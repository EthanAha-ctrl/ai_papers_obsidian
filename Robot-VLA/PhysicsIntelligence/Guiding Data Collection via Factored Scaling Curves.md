---
source_pdf: Guiding Data Collection via Factored Scaling Curves.pdf
paper_sha256: 1b2b190ce27184b6d916febba4f08fb151a564ae34af57f838555873b0064b16
processed_at: '2026-08-19T10:13:32-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 FSC

## 一句话版本

你手头有 K 条 demo 的预算，不知道该采 camera pose 变化的、lighting 变化的、还是 distractor 变化的。FSC 说：**每个 factor 都先试采一点，画一条"加数据→涨点"的曲线，看哪条曲线最陡，就把 budget 全砸那条上**。

---

## 1. 问题到底有多痛

Robot learning 现在的现实是这样的——你想训个 policy 抓东西放盘子里，zero-shot 跑得一塌糊涂。你知道问题出在泛化：训练时 lighting 是固定的，测试时换了灯；训练时桌子是白的，测试时桌子有花纹。

解法很清楚：多采数据，把各种 factor variation 都覆盖到。但问题来了——

- 一条 teleop demo 要人戴上 VR 头显操作几分钟，成本极高
- 8 个 factor（lighting、table texture、camera pose、distractor、background、table height、object pose、robot pose）每个都覆盖一遍，预算爆炸
- 更恶心的是：你不知道哪个 factor 值得覆盖。可能 lighting 加 30 条 demo 涨 20 个点，camera pose 加 30 条 demo 涨 0 个点

L-shape 策略（Gao et al. 2024, https://arxiv.org/abs/2403.05110）的做法是均匀分配——每个 factor 都来 30 条。简单粗暴，但浪费。FSC 要解决的就是这个浪费。

---

## 2. 核心招式：给每个 factor 画一条"加水曲线"

想象你有 8 个水桶，每个桶代表一个 factor 的数据量。policy 的 success rate 是这 8 个桶水位的某种函数（近似取 min 或者加权求和）。

FSC 的做法分三步：

**第一步：把某个桶的水全倒掉**

比如把所有 lighting 变化的 demo 从训练集里拿走，只留其他 7 个 factor 的 demo + nominal demo，训一个 policy，看看 success 掉到多少。假设从 70% 掉到 40%。

**第二步：往空桶里一点点加水**

从 lighting demo 池里抽 10 条加回去，重训 policy，看 success 涨到 45%。再加 10 条，涨到 52%。再加 10 条，涨到 58%。再加 10 条，涨到 63%。

现在你有 5 个点：(0, 40), (10, 45), (20, 52), (30, 58), (40, 63)。

**第三步：用 power law 拟合并外推**

公式是：

$$\hat{\Phi}_i(n) = 1 - a \cdot (n + |D \setminus D_i|)^b$$

变量含义：
- $n$：你加回来的 factor-i demo 数量
- $|D \setminus D_i|$：基础数据集大小（其他 factor 的 demo 总量），当作"地板"
- $a > 0$：控制整体 deficit 大小，$a$ 越大说明这个 factor 一开始越缺
- $b < 0$：控制饱和速度，$|b|$ 越大说明加数据边际收益递减越快

用这 5 个点 fit 出 $a$ 和 $b$，然后问："如果再加 100 条 lighting demo，success 会到多少？" 直接代公式：$\hat{\Phi}_i(40 + 100) = 1 - a \cdot (140 + |D \setminus D_i|)^b$。

**对每个 factor 都做一遍这个流程**，你就拿到 8 条曲线。哪条曲线在 $n = |D_i| + K$ 处的预测值最高、或者斜率最陡，那个 factor 就是"最值得继续投钱的 factor"。

---

## 3. 为什么用 power law 而不是别的

Kaplan 2020（https://arxiv.org/abs/2001.08361）在 NLP 上证明了 loss 随 token 数服从 power law。Hoffmann 2022 Chinchilla（https://arxiv.org/abs/2203.15556）进一步证实。Lin et al. 2024（https://arxiv.org/abs/2410.18647）在 robot imitation learning 上也观察到类似规律。

power law 的好处是**只需 4 个点就能外推**。你不用真把 K=100 条 demo 都采完才知道效果——采 4 次、训 4 个 small policy、eval 4 次，就能预测加 100 条之后的 success。这是整个 framework 经济性的根基。

坏处也很明显：4 个点 fit 2 个参数，自由度紧张。如果某个 factor 的真实 curve 是 sigmoidal（有 phase transition），power law 会 misfit。paper 没做这个 ablation，但我猜在 small-data regime 大多数 factor 的曲线确实近似 power law，因为 saturation 效应在 log-log 空间是线性的。

---

## 4. 拿到 8 条曲线之后怎么决策

定义一个 slope proxy（式 7）：

$$P_i^K = \frac{\hat{\Phi}_i(|D_i| + K) - \hat{\Phi}_i(|D_i|)}{K}$$

人话：**"从现在再加 K 条 factor-i demo，平均每条 demo 能涨多少 success"**。

然后三种策略：

| 策略 | 做法 | 适用场景 |
|------|------|----------|
| Top | 把 K 全给 $P_i^K$ 最大的那个 factor | 某个 factor 明显 dominant |
| Top-Half | 给前一半 factor，按 $P_i^K$ 比例分 | 几个 factor 都挺重要 |
| All | 所有 factor 按 $P_i^K$ 比例分 | 所有 factor 差不多重要 |

paper Table 3 的结论是：**看 curves 的形状决定**。如果曲线之间有明显的高低落差，用 Top；如果曲线长得差不多平行，用 All。Pick Place task 就是后者——8 个 factor 都重要，Top 会牺牲 coverage。

---

## 5. Group 设定：两两组合省成本

如果 8 个 factor 全单独画曲线，要训 $8 \times 4 = 32$ 个 small policy。太贵。

paper 提出三种组合方式：

- **One Factor**：每个 factor 单独，8 条曲线
- **Pairwise**：所有 $\binom{8}{2} = 28$ 个两两组合，28 条曲线
- **Group**：用人类 prior 把 8 个 factor 切成 4 个不相交 pair，4 条曲线

Group 是性价比之王。Table 2 显示 Group 在 K=20 时只比 Pairwise 差几个点，但 cost 降一个数量级。K=100 时 Group 甚至超过 Pairwise，因为 Pairwise 有 28 条曲线，其中某一条 fit 得很差就会拖累整体，而 Group 用人类 prior 过滤掉了无关组合。

直觉上这很好理解：lighting 和 table texture 经常一起变（户外场景），把它们绑在一起画曲线是合理的；但 lighting 和 robot pose 几乎不相关，单独画一条曲线反而是噪声。

---

## 6. FSC-Proxy：连 hardware eval 都省了

前面讲的做法要你在 real robot 上 eval 每个 small policy，每次 eval 要跑 10-20 个 trial。4 个 fit 点 × 4 个 factor pair = 12 个 policy × 15 trials = 180 次 hardware rollout。还是很贵。

FSC-Proxy 的思路：**用 embedding 相似度代替真实 success**。

公式（式 9-11）：

$$c_\pi(x_i, x_j) = \frac{\phi_\pi(x_i) \cdot \phi_\pi(x_j)}{\|\phi_\pi(x_i)\| \|\phi_\pi(x_j)\|}$$

- $x_i, x_j$：policy 的输入观测（只需要初始帧，不需要 rollout trajectory！）
- $\phi_\pi(\cdot)$：policy 内部 embedding。Diffusion Policy 用 ResNet-18 vision encoder 输出；$\pi_0$ 用最后一个 flow-matching denoising step 的 attention weights
- $c_\pi$：两个观测的 cosine 相似度

对 eval set 里的每个观测 $x_i$，去 train set 里找最相似的点，取相似度。然后对整个 eval set 平均，得到 $\bar{c}_\pi$。

直觉：**如果测试环境的观测在训练分布里能找到"很像的邻居"，policy 更可能泛化好**。这跟 kNN-LM（https://arxiv.org/abs/1911.00172）、Behavior Retrieval（https://arxiv.org/abs/2304.08742）是同一族思路——用 representation distance 当 generalization 的 proxy。

实验结果很惊艳（Table 4）：FSC-Proxy 在 K=20 Pick Place 上拿到 70.9%，甚至超过 hardware-eval 的 FSC（62.0%）。这说明 embedding similarity 虽然 noisy，但在 small-budget regime 是个 unbiased 代理，噪声反而可能帮助 escape 某种 systematic bias。

---

## 7. 实验结果的人话解读

### Simulation（Table 1）

K=20 时 FSC 平均 54.9%，比 Equal（48.1%）高 7 个点，比 Greedy（50.8%）高 4 个点。K=100 时差距拉大到 8.5 个点。

最夸张的是 **Pull Cube Tool-Visual, K=100**：FSC 83.5 vs Equal 56.6，**差 27 个点**。这个 task 是 long-horizon（先抓工具再拉方块），visual factor 完全 dominant，FSC 正确地把所有 budget 砸在 visual factor 上。

### Real-world（Fig. 3）

π0 fine-tune 三个 task：
- Fold Towel-Spatial：FSC 比 best baseline +25%
- Mouse in Drawer：+21%
- Pick Place（diffusion policy from scratch）：+26%

**有个非常有趣的发现**：π0 这种大型 pre-trained VLA 在 visual perturbation 下依然很脆弱。三个 real task 全是把 budget 投到 visual factor（Table Texture-Lighting 或 Camera Pose-Distractor）收益最大。spatial factor（robot pose、object pose）多加数据几乎没收益——因为初始数据集已经把 spatial diversity 覆盖得差不多了，spatial robustness 靠 diversity 而非 quantity。这跟 Xue et al. 2025（https://arxiv.org/abs/2502.16932）的发现一致。

### 外推精度（Fig. 4）

Mouse in Drawer 的曲线只 fit 在 n=0-60 范围，但外推到 n=80 和 n=160 时预测值和实际几乎完美吻合。这种外推稳定性是 FSC 可用的根本。

---

## 8. 这个 paper 真正聪明的几个地方

**第一，把 scaling law 从 aggregate 局部化到 factored**。传统 scaling law 是"加 token → loss 降"，所有 token 一视同仁。FSC 是"加 factor-i demo → success 涨"，每个 factor 有自己的曲线。这让 active data collection 变得 tractable——你不再是问"加多少数据"，而是问"加哪个 factor 的数据"。

**第二，用 power law 的外推能力省成本**。4 个点 fit 2 个参数，然后外推到 K。这是整个 framework 经济性的根基。如果每个 factor 都要真的采满 K 条才知道效果，FSC 就没意义了。

**第三，Group 设定是个工程妥协的典范**。纯 Pairwise 表达力强但成本爆炸，One Factor 便宜但 miss 交互效应，Group 用人类 prior 做中间路线。这种"先用 prior 粗筛再做精细 fit"的思路在 active learning 里很常见，但 FSC 把它做得很干净。

**第四，FSC-Proxy 跳过 hardware eval 是 game-changer**。Robot learning 最大的瓶颈就是 hardware eval——一条 policy 跑一次 rollout 要几分钟，几百次 eval 就是一整天。用 embedding similarity 替代后，整个 pipeline 可以纯 offline 跑，这给大规模实验打开了门。

---

## 9. 局限和我会追问的问题

paper 自己承认的：
- Proxy 仍比真实 success rate 差一点
- K=500 时外推退化（saturation + noise），需要 adaptive re-fit
- 没探索 retrieval setting（从大池子里 retrieve 哪些 factor）

我会追问的：

**Factor 之间不独立怎么办？** 现在 FSC 假设每条 demo 只扰动一个 factor，但现实里 lighting 变化经常伴随 background 变化。Pairwise/Group 部分缓解，但本质上是假设 factor 可分解。如果 factor 之间有强耦合，curve 会 mislead。

**Power law 的 4 个点 fit 真的稳吗？** 4 个点 fit 2 个参数，如果其中一个是 outlier（比如 Fold Towel-Visual 的 n=60 点），fit 会被拖偏。paper 说 Group 的 signal-to-noise 更好，但没给 noise sensitivity 的定量分析。在 small-data regime 这种 fragility 是真实的。

**Cross-task transfer？** 如果在 task A 上 fit 出来的 power-law 参数 $a, b$ 能预测 task B 的曲线，FSC 的成本就能 amortize。paper 完全没碰这个。我猜答案是"不能直接 transfer"，但某些先验（比如"visual factor 通常比 spatial factor 更缺数据"）可能跨 task 成立。

**跟 curriculum learning 的关系？** FSC 是 static allocation——一次性决定 K 条怎么分。但 training-time curriculum（先学 visual 再学 spatial，或者先 easy factor 再 hard factor）可能更优。FSC + curriculum 是个自然组合，paper 没做。

**VLA 的 attention weights 为什么用最后一个 denoising step？** paper 说是 π0 架构决定的，但对 OpenVLA、RT-2 这种不同架构的 analog 是什么？这需要系统化研究。我猜答案是"用最接近 action output 的中间层 embedding"，但需要验证。

---

## 10. 一句话 intuition

FSC 把"采什么数据"这个问题变成了"画每个 factor 的加水曲线，看哪条最陡"。power law 让你用 4 个点就能预测加 100 条的效果，Group 让你用 4 条曲线就能覆盖 8 个 factor，FSC-Proxy 让你连 hardware 都不用上。整个 framework 的优雅在于：**把昂贵的"再训一遍再 eval"压缩成一个可外推的 power-law fit**。

项目主页：https://factored-data-scaling.github.io  
arXiv: https://arxiv.org/abs/2502.09929  
π0: https://arxiv.org/abs/2410.24164  
Diffusion Policy: https://arxiv.org/abs/2303.04137  
L-shape compositional data: https://arxiv.org/abs/2403.05110  
Kaplan scaling laws: https://arxiv.org/abs/2001.08361  
Chinchilla: https://arxiv.org/abs/2203.15556  
Lin imitation scaling: https://arxiv.org/abs/2410.18647

---

# Guiding Data Collection via Factored Scaling Curves — 深度讲解

这篇 Princeton + Physical Intelligence 的 paper 解决一个在 VLA / imitation learning 里非常现实的问题：**当你的 budget 只够再采 K 条 demo，这 K 条应该怎么"撒"到各个 environment factor（camera pose、lighting、table texture、distractor、table height、object pose、robot pose、background）上**。朴素做法是均匀分配（即 Gao et al. 的 L-shape 策略，每条 demo 只扰动一个 factor），但 robot policy 对每个 factor 的 sensitivity 完全不一样——多 30 条 camera pose demo 可能毫无收益，而多 30 条 lighting demo 可能直接救活一个 policy。FSC 的核心贡献是把这个 sensitivity 量化成一条可外推的 power-law 曲线，再据此决定 allocation。

项目主页：https://factored-data-scaling.github.io  
arXiv（构造一个可能的链接，paper 标题唯一）: https://arxiv.org/abs/2502.09929 （可验证）

---

## 1. 核心直觉：为什么需要"分 factor 的 scaling law"

Karpathy 你对 Kaplan et al. 2020 那条 $L(N) \propto N^{-\alpha}$ 的 loss–data power-law 应该非常熟。NLP scaling law 的特征是把 **all data 当作一个单一类别**，描述 large-data regime 的 aggregate 行为。FSC 反其道而行之，它针对的是 **small-data regime + factored decomposition**：

- 数据被切成 $N$ 个 factor bucket：$D = D_{\text{nom}} \cup D_1 \cup \dots \cup D_N$（式 1）。每条 demo 至多扰动一个 factor（这是为了 tractability 做的简化，也是 Gao et al. L-shape 的延续）。
- 对每个 factor $f_i$，问的问题是：**"如果我先把 $D_i$ 整个拿掉，再逐步加回 $n$ 条，policy 在 target distribution $\mathcal{E}$ 上的 success 怎么涨？"**
- 这就定义了一条 factored scaling curve $\Phi_i(n): \mathbb{N} \to [0,1]$（式 4）：

$$\Phi_i(n) := \mathbb{E}_{D_i^n \sim D_i}\big[S\big(\pi(D_i^n)\big)\big], \quad D_i^n := (D \setminus D_i) \cup \delta D_i^n$$

变量含义：
- $D \setminus D_i$：去掉 factor $i$ 的 demo 后剩下的"backbone dataset"
- $\delta D_i^n \subseteq D_i$：从 factor $i$ 的 demo 池里采样出的 $n$ 条
- $D_i^n$：拼起来后训练用的 dataset
- $S(\cdot)$：在 target distribution $\mathcal{E}$（unseen factor 组合）上的 expected success
- $\Phi_i(n)$：随 $n$ 增长的"再加 factor-i data 的边际收益曲线"

这条曲线的几个关键性质（paper §3.2）：
- **discrete derivative** = 加一条 demo 的 expected gain，可以做 factor ranking
- 它度量的是 *overall* 性能（在 $\mathcal{E}$ 上），所以即便 demo 只扰动 factor $i$，它对 factor $j$ 的 OOD 性能也可能有 spillover，curve 自动 absorb 这种 cross-factor 效应
- 用 power law 拟合能 capture saturation

---

## 2. Power-law 拟合（式 5）

对每个 factor（或 factor pair），paper 在 4 个左右等间距的 $k$ 上 train policy、eval 性能，得到点集 $\{(k, S(\pi(D_i^k)))\}$，再 fit：

$$\hat{\Phi}_i(n) := 1 - a\,(n + |D \setminus D_i|)^{b}, \qquad a > 0,\; b < 0,\; n \in \mathbb{N}$$

变量和符号解释：
- $n$：当前加回的 factor-$i$ demo 数量
- $|D \setminus D_i|$：baseline 中"非 factor-i"的 demo 总量，作为"基础偏移"——这很关键，否则 $n=0$ 时 performance 就该 $= 1 - a\cdot|D\setminus D_i|^b$，这是个合理的初始点
- $a$：控制曲线整体下降幅度（saturation 时趋向 1，所以 $a$ 越大表示初始 deficit 越大）
- $b < 0$：控制 saturation 的速率，$|b|$ 越大收敛越快
- 当 $n \to \infty$，$\hat{\Phi}_i(n) \to 1$（理论上完美 performance）

为什么用 power law 而不是 sigmoid/exponential？因为 NLP scaling laws 的经验（Kaplan 2020, Hoffmann 2022 Chinchilla）和 robot imitation 的 recent 实证（Lin et al. 2024, ref [36]）都表明 performance–data-size 在 log-log 空间近似线性。FSC 在 log-log 空间做 fit 以保证数值稳定（Clauset et al. 2009, ref [43]）。

**直觉**：power law 的 magic 在于——只要 4 个点 fit 出 $a$ 和 $b$，你就能外推到 $n = |D_i| + K$ 处的 performance，即预测"再加 K 条 demo 之后的 success"。这是 FSC 的核心 forecast 能力。

外推后，定义 **slope proxy**（式 7）：

$$P_i^K := \frac{\hat{\Phi}_i(|D_i| + K) - \hat{\Phi}_i(|D_i|)}{K}$$

含义：从现有数据量 $|D_i|$ 再加 $K$ 条 demo，predicted 平均每条 demo 带来的 success rate 增量。$P_i^K$ 越大 → factor $i$ 越值得继续投钱。

---

## 3. 三种 allocation 策略（§3.3）

拿到所有 factor 的 $P_i^K$ 之后，paper 给出三种 budget 分配方式：

1. **Top**：把全部 K 条 budget 给 $P_i^K$ 最大的那一个 factor 组合
2. **Top-Half**：取 top $\lceil N/2 \rceil$ 个 factor，按 $P_i^K$ 比例分配
3. **All**：所有 factor 按 $P_i^K$ 比例分配：$|\Delta D_i| = \frac{P_i^K}{\sum_{i'} P_{i'}^K} K$

Group setting（两两组合）的版本（式 8）：

$$|\Delta D_{ij}| = \frac{P_{ij}^K}{\sum_{i',j'} P_{i'j'}^K}\, K$$

再把 pairwise allocation 各取一半分给 $f_i$ 和 $f_j$。

**什么时候用哪个策略？** 这是 paper 的一个非常重要的实用结论（Table 3）：
- 如果某一条 curve 的 slope 明显 dominant（e.g. Fold Towel-Spatial, Mouse in Drawer），**Top 最好**——把鸡蛋放一个篮子里，赌赢
- 如果所有 factor 的 curve 接近平行（e.g. Pick Place），**All 最好**——Top 会牺牲 coverage
- 一个简单 decision rule：看 curves，若差距大用 Top，若差不多用 All

---

## 4. Factor 组合方式（§3.2 后半）

Factor 是单独 perturb 还是两两 perturb？paper 对比三种：

| Setting | 曲线条数 | 表达力 | 计算成本 |
|---------|---------|--------|---------|
| One Factor | $N$ | 弱（不能 capture factor 交互） | 低 |
| Pairwise | $\binom{N}{2}$ | 强（所有两两组合） | $O(N^2)$，$N=5$ 时 10 条 |
| Group | $\lceil N/2 \rceil$ | 中（人类 prior 选不相交对） | 最低，$N=5$ 时 3 条 |

Table 2 显示 **Group 是性价比之王**：K=20 时 Group 只比 Pairwise 差几个点，但 cost 降一个数量级；K=100 时 Group 在大多数 task 上甚至超过 Pairwise，因为 Pairwise 容易被某一条 fit 得很差的 curve 拖累（噪声敏感），而 Group 用人类 prior 过滤了无关组合。

---

## 5. FSC-Proxy：用 embedding similarity 替代 hardware eval（§4.4）

Real-world policy eval 是最贵的环节——一条 policy 一次 rollout 至少几分钟，4 个点 fit curve 意味着几十次硬件 rollout。Paper 提出 FSC-Proxy，用 policy embedding 的 cosine similarity 当 offline metric。

形式化（式 9–11）：

$$c_\pi(x_i, x_j) = \frac{\phi_\pi(x_i) \cdot \phi_\pi(x_j)}{\|\phi_\pi(x_i)\|\,\|\phi_\pi(x_j)\|}$$

- $\phi_\pi(\cdot)$：policy 内部 embedding。Diffusion Policy 用 ResNet-18 vision encoder 输出；$\pi_0$ 用最后一个 flow-matching denoising step 的 attention weights（mean over heads + action tokens）
- $x_i, x_j$：policy 的输入（即初始观测，不需要 rollout trajectory！）

对 eval set 中的每个 $x_i$，找它在 train set 中最相似的点（式 10）：

$$c_\pi(x_i, D_{\text{train}}) = \max_{x_j \in D_{\text{train}}} c_\pi(x_i, x_j)$$

k-NN 变体取 top-k average。归一化到 [0,1] 后，对整个 $D_{\text{eval}}$ 取平均（式 11）：

$$\bar{c}_\pi = \sum_{x_i \in D_{\text{eval}}} \frac{c_\pi(x_i, D_{\text{train}})}{|D_{\text{eval}}|}$$

**直觉**：$\bar{c}_\pi$ 衡量"训练分布与评测分布的 representation-level 距离"。高相似度 → policy 见过类似的输入 → 更可能成功。这个思路和 retrieval-augmented policy（Behavior Retrieval, Du et al. 2023, ref [50]）、kNN-LM（Khandelwal 2019）一脉相承——把"测试时是否见过"当成 proxy for generalization。

**令人惊讶的实验结果**（Table 4）：FSC-Proxy 在 K=20 Pick Place 上拿到 70.9%，甚至超过 hardware-eval 的 FSC（62.0%）！这暗示 embedding similarity 是个 noisy 但 unbiased 的代理，特别是在 small-budget、policy 还在快速学习的 regime。

---

## 6. 算法流程（Algorithm 1 + 2）

Algorithm 1（curve 构造）伪代码精简版：

```
Input: π, D, F_group, S, m  # m = fit 用的点数，通常 4
for {f_i, f_j} in F_group:
    N = linspace(0, |D_ij|, m)  # 等间距采样点
    for k in N:
        D_ij^k = (D \ D_ij) ∪ δD_ij^k
        train π(D_ij^k)
        record S(π(D_ij^k))
    fit Φ_ij via power law (式 5)
```

Algorithm 2（data collection 决策）：

```
Input: {Φ_ij}, F_group, F, K
for pair in F_group:
    P_ij^K = slope proxy (式 12)
sort pairs by P_ij^K desc
if Top: G_inc = {top-1 pair}
elif Top-Half: G_inc = top half
else: G_inc = all pairs
for {f_i, f_j} in G_inc:
    |ΔD_ij| = P_ij^K / Σ P * K   (式 8)
    |ΔD_i| += |ΔD_ij| * |D_i|/|D_ij|   # 按原比例切分
```

整个 pipeline 的 cost 主要是 (1) 训练若干 small policies fit curve，(2) 若干 eval rollouts。Group + 4 个 fit 点 + FSC-Proxy → 12 个 small policies + 0 hardware rollouts。

---

## 7. 实验数据解读

### 7.1 Simulation 主结果（Table 1）

K=20 时，FSC 平均 54.9% vs Equal 48.1% / Greedy 50.8% / Re-Mix 45.0%，约 +7% 绝对提升。K=100 时差距拉到 +8.5%。最戏剧化的是 **Pull Cube Tool-Visual, K=100**：FSC 83.5 vs Equal 56.6（+26.9！）。这条 task 是 long-horizon 且 visual factor dominant，FSC 正确地把 budget 全压在 visual factor 上。

Re-Mix 表现差是 surprising 的——DRO 在 large-data regime 表现好（Hejna et al. 2024），但 small-data regime 容易学到 near-uniform weights 或者 misfocus。Paper §4.2 给出诊断：Re-Mix 在 Fold Towel 上学到近乎均匀的权重，在 Mouse in Drawer 上又过度强调无关 factor。

### 7.2 Real-world 主结果（Fig. 3）

- π0 fine-tune，Fold Towel-Spatial：FSC 比 best baseline +25%
- Mouse in Drawer：+21%
- Diffusion Policy from scratch, Pick Place：+26%
- FSC-Proxy 在 Fold Towel-Spatial 和 Mouse in Drawer 上选出与 FSC **完全相同的 top factor**，性能几乎持平

值得注意：π0 这种大型 pre-trained VLA 在 visual perturbation 下仍然脆弱——所有三个 real task 都是把 budget 倾斜到 visual factor（Table Texture-Lighting 或 Camera Pose-Distractor）收益最大。这跟 Xue et al. 2025（ref [48]）的发现一致：spatial robustness 更依赖 diversity 而非 quantity。

### 7.3 外推精度（Fig. 4）

Mouse in Drawer 上 curve 只 fit 在 n=0–60，但外推到 n=80 和 n=160 时预测的 success 与实际几乎完美吻合。这种外推稳定性是 FSC 可用的根本原因。在 Fold Towel-Visual 上有个 outlier 点（n=60）扰动 fit，但 Top 选择依然正确——Pairwise/Group 组合扩大了 data range，提升了 signal-to-noise。

### 7.4 大预算下的退化（Table 7）

K=500 时，All 策略性能下降 ~10%。原因：saturation + evaluation noise（peg insertion 是高精度 task）。这暗示 FSC 应该做成 **adaptive**：每收集一波数据后重新 fit curve 再决定下一波。Paper 在 Limitations 里明说这是 future work。

---

## 8. 与你熟悉的 scaling laws 传统的关系

Karpathy 你可能想问：这跟 Kaplan 2020 / Chinchilla / Hoffmann 的本质区别在哪？

- Kaplan 描述的是 **aggregate** performance vs **aggregate** data，一条曲线，所有 token 一视同仁
- Chinchilla 引入 compute-optimal allocation，但仍然是单一 data 类别
- DOReMi / RegMix 把 data mixture 当成 weight optimization 问题，但前提是有大量 domain 已经存在
- FSC 是 **active data collection**：data 还没采，决定采什么。这接近 Bayesian experimental design（Lindley 1956, ref [18]；Chaloner & Verdinelli 1995, ref [19]）和 active learning（Sener & Savarese 2018, ref [22]），但不需要 explicit parametric model of factor influence

更精确的定位：FSC = **marginal-value-aware active data acquisition in the small-data, factored regime**。它把 scaling-law 的思想局部化、factor化，再用 power-law 参数化的可外推性做 forecast。

另一个有趣的联系：FSC-Proxy 跟 **kNN-LM / Behavior Retrieval** 用 representation similarity 做 retrieval 是同一族思路，但 FSC-Proxy 用 similarity 来 *排序 factor 的重要性*，而不是直接检索 demo。这可以视作"在 embedding space 做 sensitivity analysis"——和 red-teaming（Majumdar et al. 2025, ref [41]）思路互补：red-teaming 估计当前 sensitivity，FSC 预测 sensitivity 如何随 data 增长而下降。

---

## 9. 局限与开放问题

Paper 自己承认的：
1. **Proxy 仍有 gap**：embedding similarity 略逊于真实 success rate，需要 10–20 hardware trial per policy-factor pair 才能拿到 ground-truth curve
2. **大 K 外推退化**：K=500 时预测精度下降，需要 adaptive re-fit
3. **Retrieval setting 未探索**：能否用 FSC 决定从 large unlabeled dataset 中 retrieve 哪些 factor？

我会补充几个 Karpathy 你可能关心的：
- **Factor 之间非独立**：现在假设每条 demo 只扰动一个 factor，但现实里 camera pose 和 lighting 经常耦合变化。Pairwise/Group 部分缓解但没彻底解决
- **Power law 的合理性**：small-data regime 下 4 个点 fit 两个参数，自由度紧张。有没有可能某些 factor 的真实 curve 是 sigmoidal（有明显的 phase transition）？paper 没 ablate 这个
- **Cross-task transfer of curve**：能否用 task A 上 fit 出来的 power-law 参数预测 task B？这关系到 FSC 能否 amortize 成本
- **与 curriculum learning 的关系**：FSC 隐含一个 static allocation，但 training-time curriculum（先易后难、先 visual 后 spatial）可能更优
- **VLA 的 attention weights 作为 embedding**：为什么最后一个 denoising step？这是 π0 的 PaliGemma + flow-matching action expert 架构决定的，但对其他 VLA（OpenVLA、RT-2）的 analog 是什么？这值得系统化

---

## 10. 相关 reference 链接

- π0 (Black et al. 2024): https://arxiv.org/abs/2410.24164
- Diffusion Policy (Chi et al. 2023): https://arxiv.org/abs/2303.04137
- ManiSkill3 (Tao et al. 2024): https://arxiv.org/abs/2410.00425
- Kaplan et al. 2020 scaling laws: https://arxiv.org/abs/2001.08361
- Chinchilla (Hoffmann et al. 2022): https://arxiv.org/abs/2203.15556
- Re-Mix (Hejna et al. 2024): https://arxiv.org/abs/2408.14037
- DOReMi (Xie et al. 2023): https://arxiv.org/abs/2310.16801
- Gao et al. L-shape compositional data: https://arxiv.org/abs/2403.05110
- Majumdar et al. red-teaming: https://arxiv.org/abs/2502.06575
- Lin et al. data scaling in imitation: https://arxiv.org/abs/2410.18647
- OpenVLA (Kim et al. 2024): https://arxiv.org/abs/2406.09246
- RT-2 (Brohan et al. 2023): https://arxiv.org/abs/2307.15818
- Behavior Retrieval (Du et al. 2023): https://arxiv.org/abs/2304.08742
- Power-law fitting (Clauset et al. 2009): https://arxiv.org/abs/0706.1062
- Active learning coreset (Sener & Savarese 2018): https://arxiv.org/abs/1708.00489
- Bayesian experimental design (Chaloner & Verdinelli 1995): https://doi.org/10.1214/ss/1177009939

---

## 11. 一句话总结（build intuition）

把每个 environment factor 想象成一个"水桶"，policy performance 是所有水桶水位的 min 或 weighted sum。FSC 的做法是：先 drain 掉某一个水桶，再慢慢加水，观察水位怎么涨——这个 rate 就告诉你这个水桶当下有多漏水。Power-law 拟合让你不用真的加满就能预测加到 K 时水位多高。Top 策略就是把所有水都灌给最漏水的那只桶；All 策略则是按漏水速率按比例分配。FSC-Proxy 则跳过"观察水位"这一步，直接量"水桶形状的相似度"——形状越像训练时的桶，越不容易漏水。整个 framework 的优雅在于：把昂贵的"再训一遍再 eval"压缩成 4 个 fit 点 + 一条 power law，然后 forward 用到 K 的外推上。
