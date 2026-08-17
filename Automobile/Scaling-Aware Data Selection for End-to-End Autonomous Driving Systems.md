---
source_pdf: Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems.pdf
paper_sha256: 99bd589b089f1c61ce9b74591b50212f3fef3f4bc9b91a23ac679bdef329e179
processed_at: '2026-08-12T03:45:59-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MOSAIC - 用人话讲

## 一句话版本

你有一堆自动驾驶录像，想挑出最有用的来训练模型，但模型好不好要看 9 个指标（不撞车、不出道、不闯红灯、舒服、等等），这些指标还互相打架——MOSAIC 的做法是：先把录像按场景分堆，测每堆对每个指标的"边际收益曲线"，然后一个一个挑，每次挑当前最划算的那个。

---

## 用做饭类比讲清楚

想象你要训练一个厨师，手上有 4 堆食材：Boston 龙虾、Pittsburgh 土豆、Singapore 辣椒、Vegas 糖。你要凑一道菜，评分看 9 个维度：咸、甜、鲜、辣、口感、温度、摆盘、香气、成本。

痛点在哪？

第一，每堆食材对每个维度的贡献不一样。龙虾主要提鲜，糖主要提甜，你不能一锅乱炖。

第二，9 个维度互相竞争。多放糖甜了但鲜味被盖住，多放辣辣了但成本飙升。

第三，每堆内部有好有坏。同样是 Boston 龙虾，有的新鲜有的不新鲜，得先挑好的用。

MOSAIC 的解法分三步：

**Step 1: 分堆 + 排序**
把食材按来源分 4 堆，每堆内部按"新鲜度"排序，先用新鲜的。

**Step 2: 测边际收益曲线**
先做 2 次小实验：从每堆各取 100g、200g 试做，拟合出"这堆食材加多少克，鲜味提升多少"的曲线。你会发现 Boston 龙虾曲线早期陡但很快饱和，Pittsburgh 土豆曲线稳定线性增长，Vegas 糖早期贡献小但也很快饱和。

**Step 3: 贪心迭代挑选**
每次加 1 克食材，算哪堆当前边际收益最大就加哪堆。早期 Boston 龙虾边际收益最高，先猛加；加到一定程度 Boston 饱和了，边际收益下降，Pittsburgh 土豆反而成最划算的了，转去加土豆；最后 Boston 和 Pittsburgh 都饱和了，Vegas 糖的边际收益反而相对最高，开始加糖。

这就是 MOSAIC 的全部 intuition。

---

## 为什么这个事情难

### 难点 1: 多 metric 互相竞争

LLM 训练只看一个 validation loss，所以 data mixture 方法（DoReMi、DOGE、Chameleon、ADO）都假设加数据的收益是标量。但自动驾驶看 **EPDMS**，它是 9 个 metric 的聚合：

$$
\text{EPDMS} = \underbrace{(NC \times DAC \times DDC \times TLC)}_{\text{乘法 penalty}} \times \underbrace{\frac{5 \cdot EP + 5 \cdot TTC + 2 \cdot LK + 2 \cdot HC + 2 \cdot EC}{16}}_{\text{加权平均}}
$$

变量含义：
- **NC** (No Collision)：不撞车，0 或 1
- **DAC** (Drivable Area Compliance)：不出车道
- **DDC** (Driving Direction Compliance)：不逆行
- **TLC** (Traffic Light Compliance)：不闯红灯
- **EP** (Ego Progress)：往前走，别趴窝
- **TTC** (Time-to-Collision)：碰撞时间裕度
- **LK** (Lane Keeping)：车道保持
- **HC** (Hard Comfort)：无急刹急转
- **EC** (Ego Comfort)：综合舒适

intuition：前 4 个是**乘法**，任何一个为 0 整个 EPDMS 归零。这意味着如果模型 DAC = 0（开出车道），其他指标再好也没用。所以 data selection 必须优先补短板，不能均衡铺数据。

### 难点 2: 数据池异质

OpenScene 有 31539 个 10s clips，来自 Las Vegas、Boston、Singapore、Pittsburgh 等地。Vegas 是密集城市路口，主要影响红绿灯合规 TLC 和行人避让 NC；Pittsburgh 是郊区弯道，主要影响车道保持 LK。如果用 LLM 的 homogeneous domain 假设，等于假设所有 clip 对所有 metric 影响相同，这显然错。

### 难点 3: 数据池内不可预先分

即使知道 geolocation，同一个城市内部场景也千差万别。Boston 既有高速也有路口。所以需要更细的 clustering（paper 试了用 Qwen-2.5-VL 生成 caption 再 TF-IDF 聚类）。

---

## 数学公式用大白话讲

### 核心目标（公式 1）

$$
\max_{\mathcal{D}_{sel} \subset \mathcal{D}_{pool}} U\bigg(\{\mathcal{G}_r(\mathcal{D}_{train} \cup \mathcal{D}_{sel})\}_{r=1}^{R}\bigg)
$$

人话：从数据池 $\mathcal{D}_{pool}$ 里挑 $B$ 个 clip 加到训练集 $\mathcal{D}_{train}$ 上，让 9 个 metric $\mathcal{G}_r$ 聚合后的 utility $U$（即 EPDMS）最大化。

变量：
- $\mathcal{D}_{train}$：已有训练集（Navtrain 实验里初始 460 clips）
- $\mathcal{D}_{pool}$：候选池（4141 clips 待选）
- $\mathcal{D}_{sel}$：要选的子集，大小为 budget $B$
- $\mathcal{G}_r$：第 $r$ 个 metric，$R=9$
- $U$：聚合函数，这里就是 EPDMS

### 拆解为 mixture 问题（公式 2）

$$
\max_{n_1, \ldots, n_M, \sum n_i = B} \Delta U_{mix}(n_1, \ldots, n_M)
$$

人话：把池子分成 $M$ 堆（比如 4 个城市），决定从每堆拿 $n_i$ 个，加起来等于 $B$，让 utility 增益最大。

变量：
- $M$：cluster 数量（geolocation 时 $M=4$）
- $n_i$：从第 $i$ 个 cluster 拿多少

### 线性可分假设（公式 3）

$$
\Delta U_{mix}(n_1, \ldots, n_M) \approx \sum_{i=1}^M \Delta U_i(n_i)
$$

人话：假设每堆的贡献独立可加，忽略堆间交互。比如 Boston 龙虾 + Pittsburgh 土豆的联合效果 = Boston 龙虾单独效果 + Pittsburgh 土豆单独效果。

变量：$\Delta U_i(n)$ 是只从 cluster $i$ 加 $n$ 个样本时的 utility 增益。

paper 在 Section 11 验证这个假设：实际 vs 估计 EPDMS 偏差最多 ~1 分（Table 5：1600 clips 实测 90.2 vs 估计 91.1），所以交互项存在但小。

### Saturating exponential scaling law（公式 4）

$$
\Delta U_i(n) \approx \widehat{\Delta U_i}(n) := a_i(1 - e^{-n/\tau_i})
$$

人话：每堆数据的收益曲线是"先快后慢饱和"的指数曲线。$a_i$ 是这堆数据能给你带来的总增益上限，$\tau_i$ 控制饱和速度。

变量：
- $a_i$：cluster $i$ 的**渐近增益上限**，即加无限多数据能达到的最大提升
- $\tau_i$：cluster $i$ 的**饱和速率**，$\tau_i$ 小则快饱和
- $n$：从该 cluster 加的样本数

intuition：
- 当 $n \to 0$，$\widehat{\Delta U_i} \approx a_i \cdot n / \tau_i$，斜率 $a_i/\tau_i$
- 当 $n \to \infty$，$\widehat{\Delta U_i} \to a_i$，饱和
- 二阶导数 $-a_i/\tau_i^2 \cdot e^{-n/\tau_i} < 0$，所以是 concave，diminishing return

为什么用 exponential 而非 power law？因为 power law $L \propto N^{-\alpha}$ 描述 **loss decay**（越加数据 loss 越低），而 MOSAIC 描述 **gain accumulation**（越加数据 EPDMS 越高），gain 的自然形式是有上限的 saturating 函数。

### 边际增益（公式 5）

$$
\delta_i(b_i) := \widehat{\Delta U_i}(b_i + 1) - \widehat{\Delta U_i}(b_i) = \frac{a_i}{\tau_i} \cdot e^{-b_i/\tau_i}
$$

人话：当前已经从 cluster $i$ 拿了 $b_i$ 个样本，再拿 1 个能多赚多少 utility。

变量：
- $b_i$：已从 cluster $i$ 取的样本数
- $\delta_i(b_i)$：再加 1 个的边际增益

由于 $\widehat{\Delta U_i}$ 是 concave，$\delta_i(b_i)$ 单调递减——这就是 diminishing return 的数学表达。

### 迭代算法

```
每次循环:
  对每个 cluster i，算 δ_i(b_i)  # 当前边际增益
  j = argmax_i δ_i(b_i)          # 选边际最大的 cluster
  从 cluster j 按重要性排序取下一个样本
  b_j += 1
```

intuition：这本质是 **concave maximization 的贪心 ascent**。由于目标函数 concave，KKT 条件下最优解满足所有非零 $b_i^*$ 的边际增益相等：$\delta_i(b_i^*) = \delta_j(b_j^*)$。贪心算法自然收敛到这个平衡。

---

## 实验数字怎么读

### 主结果 Table 1 (OpenScene, budget=4000)

| Method | EPDMS | BRMR |
|---|---|---|
| Random | 80.38 | 1.00 |
| Uncertainty | 73.46 | 2.00 |
| Coreset | 83.63 | 0.25 |
| Chameleon | 82.92 | 0.39 |
| **MOSAIC** | **84.25** | **0.18** |

人话：
- MOSAIC 比 Random 高 4 分，比第二好的 Coreset 高 0.6 分
- **BRMR = 0.18 意味着只用 18% 的数据就能达到 Random 全量效果**，即省 82% 数据
- Uncertainty 比 Random 还差，因为高 entropy 的样本可能是噪声而非 informative

### 子项分解 Table 2 (OpenScene, budget=4000)

Base 模型 DAC = 83.9 是最大短板（其他都在 90+）。MOSAIC 把 DAC 从 83.9 拉到 93.59（+9.69 分）。

为什么 MOSAIC 优先补 DAC？因为 DAC 是乘法 penalty 项，且 base 模型最弱。Scaling law 自动识别出"加 Pittsburgh 数据对 DAC 提升最大"，所以 early iteration 猛加 Pittsburgh。

EPDMS 的乘法结构意味着 DAC 从 83.9 → 93.59 对最终 EPDMS 的 leverage 是 93.59/83.9 = 1.115，即直接 11.5% 拉动。如果改成提升 NC 从 94 → 97（+3），leverage 只有 97/94 = 1.03，3% 拉动。所以 MOSAIC 的 scaling-aware 策略自动识别了**高 leverage metric + 高 margin domain** 的组合。

### Scaling dynamics (Figure 3, 4)

四个城市的 scaling 曲线形态：
- **Boston**：$a$ 大 $\tau$ 小，早期陡但快饱和（~500 clips 后接近 plateau）
- **Singapore**：类似 Boston，saturation 稍慢
- **Pittsburgh**：$a$ 大 $\tau$ 大，稳定线性增长，高 budget 主导
- **Las Vegas**：$a$ 小 $\tau$ 小，早期 gain 最小也最快饱和

迭代过程：
- Iteration 0-500：主要选 Boston + Singapore（边际增益最高）
- Iteration 500-3700：转 Pittsburgh（Boston/Singapore 饱和后 Pittsburgh 仍线性增长）
- Iteration 3700+：Pittsburgh 用完，转 Vegas

这展示了 MOSAIC 的 **regime-dependent allocation**：低 budget 用快饱和高初始 gain 的 cluster，高 budget 用慢饱和稳定增长的 cluster。静态 mixture weight（Chameleon）做不到这种动态切换。

---

## Baselines 用人话讲

### Random
从池子随机抽 $B$ 个。最朴素，作为 benchmark 基线。

### Uncertainty (Algorithm 2)
对每个 clip 算 trajectory logits 的 Shannon entropy $H = -\sum p \log p$，选 entropy 最大的 $B$ 个。intuition：模型不确定的样本可能最有学习价值。实际效果差（OpenScene budget=4000 时 EPDMS 73.46 < Random 80.38），因为高 entropy 也可能是 OOD 噪声，模型不确定不等于 informative。

### Coreset (Algorithm 3)
几何 diversity 选择：每次选离已选集最远的样本，欧氏距离。intuition：覆盖 feature space。效果不错（OpenScene 83.63），但 ignore metric relevance——它选 diverse 样本但 diverse 不一定对 EPDMS 有用。

### Chameleon (Algorithm 4)
Kernel ridge regression based mixture weighting：
1. 每个 cluster 用 model embedding 算 centroid $x_i$
2. 构 affinity matrix $\Omega = X X^\top$
3. 用 KRLS 算每个 cluster 的 score $S_i$
4. Softmax 归一化得到 mixture weight $\alpha_i$
5. 按 $\alpha_i$ 比例采样

intuition：相似 cluster 互相"竞争"权重，独特 cluster 获得更多。弱点：对 cluster 结构敏感，caption-based clustering 时性能崩塌（Table 8 中 BRMR 高达 3.32）。

---

## Ablation 的人话解读 (Figure 6)

paper 做了两个 ablation：
- **w/o Clustering**：不分堆，直接按 importance score 排序 greedy 取 top-B
- **w/o Ranking**：分堆但每个 cluster 内随机采样

结果：
- **低 budget (<800 clips)**：w/o Clustering 和 MOSAIC 接近，说明 ranking 主导
- **高 budget**：w/o Clustering 开始落后 MOSAIC，说明 scaling-aware cluster selection 主导

人话总结：**ranking 是 data-efficient regime 的引擎**（先用最有用样本），**scaling-aware cluster selection 是 data-abundant regime 的引擎**（决定何时换 cluster）。两者缺一不可。

---

## Compute 成本怎么看

MOSAIC 需要前期 pilot runs 拟合 scaling law（每个 cluster 2 个 pilot 点），但整体 compute 仍更优：
- vs Coreset 同等 EPDMS 省 16% compute（490 A100 GPU hours）
- vs Random 省 57% compute（1700 A100 GPU hours）

intuition：pilot 成本是 upfront 固定开销，训练成本随 budget 线性增长。大 budget 时 pilot 成本被摊薄，scaling-aware 选样的效率优势显现。

---

## 一些容易被忽略的细节

### Virtual clip 创建
原始 driving log 30s-50min，MOSAIC 切成 10s virtual clip（20 frames @ 2Hz）。理由：
1. 单个 log 太长，不能当独立样本
2. 与 industry practice（Waymo）对齐
3. non-overlapping sequential 切，尾部 <10s 丢弃

Navtrain 切完 4601 clips（丢 10.9% frames），OpenScene 切完 32539 clips（丢 1.8%）。

### Caption-based clustering
paper 用 Qwen-2.5-VL-32B-Instruct 给每个 clip 生成 <150 words 描述，TF-IDF 提取 top 1024 unigrams/bigrams，聚成 6 类（Table 3）：

| Cluster | Top terms |
|---|---|
| 1 | calm, day, street, trees, signs, yellow |
| 2 | signals, crossing, crosswalks, pedestrians |
| 3 | highway, vehicles, busy urban, palm trees |
| 4 | building, area, large, paved, parking |
| 5 | city street, major city, moderate |
| 6 | precipitation, potential rain, overcast, cloudy |

对应：郊区 / 行人路口 / 高速 / 停车场 / 城市主路 / 雨天。比 geolocation 4 cluster 更细，MOSAIC 在其上依然 robust，BRMR 0.37-0.48。

### Ranking signal 依赖
$\mathcal{T}(x) = U(\{\mathcal{G}_r(f(\cdot; \mathcal{D}_{train}), x)\}_{r=1}^R)$ 需要 9 个 rule-compliance 评估，依赖 bounding box 等密集标注。Section 10 ablation 试了 trajectory imitation loss、gradient norm、gradient perturbation，都与 EPDMS ranking 相关性极低（Kendall-Tau ≈ 0）。这是 MOSAIC 的实际部署痛点：**需要标注成本**。

---

## 这篇 paper 的真正 insight

1. **Multi-metric disentanglement via clustering**：传统 data mixture 假设 domain homogeneous 对 metric 影响一致，MOSAIC 通过 clustering 把 data vs metric 的 heterogeneous influence 结构化。

2. **Gain-based scaling law**：用 exponential saturating $\Delta U = a(1-e^{-n/\tau})$ 而非 power law $L \propto N^{-\alpha}$，因为描述 gain accumulation 而非 loss decay。

3. **First-difference greedy = concave maximization**：迭代选边际最大 cluster 等价于 KKT 条件下的 gradient ascent，理论收敛到全局最优（因为 concave）。

4. **Regime-dependent allocation**：scaling-aware 自动实现"低 budget 用快饱和 cluster，高 budget 用慢饱和 cluster"的动态分配，静态 mixture weight 做不到。

5. **Ranking + Clustering 互补**：ranking 管 within-cluster data efficiency，clustering + scaling 管 cross-cluster allocation，两者覆盖不同 regime。

---

## 局限性的诚实分析

1. **Linear separability 假设**：公式 (3) 忽略 cross-cluster interaction $\Delta U_{ij}$。如果 clustering 不 semantic，交互项会主导，MOSAIC 退化。

2. **Pilot cost**：每个 cluster 2 个 pilot run。pool 很大且 cluster 很多时 pilot 成本爆炸。

3. **Ranking 依赖 dense annotation**：$\mathcal{T}(x)$ 需要 9 个 metric 评估，cheap signal 难找。

4. **单一 saturating 曲线假设**：cluster 内部若有 sub-domain（Boston 内既有高速也有路口），实际 scaling 可能是 multiple saturation 叠加，单一 $a_i, \tau_i$ 拟合不准。

5. **Cluster 数 $M$ 敏感性**：paper 没系统 ablate $M$。$M$ 太小 cluster 同质性差，$M$ 太大 pilot 成本线性增长。

---

## 我联想到的相关工作

### 与 LLM data mixture 的对比
- **DoReMi** (Xie et al. 2023, https://arxiv.org/abs/2305.10429)：用 proxy model 算 excess loss 估 domain weight，假设单一 loss
- **DOGE** (Fan et al. 2023, https://arxiv.org/abs/2310.15393)：跟踪 domain gradient 做 reweighting
- **Chameleon** (Xie et al. 2025, https://arxiv.org/abs/2410.11368)：kernel similarity based mixture，对 cluster 结构敏感
- **ADO** (Jiang et al. 2025, https://arxiv.org/abs/2405.18392)：on-the-fly scaling fit + gradient reweighting，但无 per-domain isolation scaling
- **Data Mixing Laws** (Ye et al. 2025, https://arxiv.org/abs/2403.16952)：训练多个小 proxy model 拟合 mixture → performance regressor

MOSAIC 与这些的区别：**multi-metric**、**per-cluster saturating scaling**、**iterative first-difference**。

### 与 Active Learning 的对比
- **CoreSet** (Sener & Savarese 2017, https://arxiv.org/abs/1708.00489)：geometric diversity，MOSAIC 用作 baseline
- **Influence Selection** (Liu et al. 2021)：用 influence score 选样本，计算成本高
- **Forget Score** (Toneva et al. 2019)：基于 forgetting events 选样本

MOSAIC 在 spirit 上接近 batch active learning with scaling forecasts，但假设 unlabeled pool 和固定 inference cost，是 AL 的 data mixture extension。

### 与 Data Pruning 的对比
- **SemDeDup** (Abbas et al. 2023, https://arxiv.org/abs/2303.09540)：CLIP feature space 余弦相似度去重
- **Beyond Neural Scaling Laws** (Sorscher et al. 2022, https://arxiv.org/abs/2206.14491)：理论上证明 pruning 可改善 power law scaling
- **AdaDeDup** (Kang et al. 2025, https://arxiv.org/abs/2507.00049)：自适应 hybrid pruning for object detection

MOSAIC 与 pruning 的区别：pruning 是从已选集删冗余，MOSAIC 是从 pool 选增益最大的。

### 与 AD E2E 的对比
- **Hydra-MDP** (Li et al. 2024, https://arxiv.org/abs/2406.06978)：NAVSIM 2024 冠军，MOSAIC 用作 base model
- **TransFuser** (Chitta et al. 2022)：transformer-based sensor fusion
- **VAD** (Jiang et al. 2023)：vectorized scene representation
- **NAVSIM** (Dauner et al. 2024, https://arxiv.org/abs/2406.08291)：non-reactive simulation benchmark，EPDMS 来源

### 与 Scaling Laws 的对比
- **Kaplan et al. 2020** (https://arxiv.org/abs/2001.08361)：LLM power law $L \propto N^{-\alpha}$
- **Henighan et al. 2020** (https://arxiv.org/abs/2010.14701)：autoregressive generative scaling
- **Hoffmann et al. 2022 (Chinchilla)**：compute-optimal scaling

MOSAIC 的 exponential saturating 与这些 power law 形态不同，因为描述 gain 而非 loss。

---

## 推广潜力

### Robotics manipulation
multi-task metric：success rate、energy、smoothness。可 cluster 成 pick-and-place、precision-insertion、assembly 等 domain。

### VLM training
multi-benchmark：VQA、captioning、OCR、grounding。可 cluster 成 natural image、document、screenshot、diagram 等。

### RLHF
multi-objective reward：helpfulness、harmlessness、honesty。可 cluster 成 factual QA、creative writing、safety-sensitive、refusal 等。

关键前提：能 cluster 成对 metric 有 differential scaling 的 domain。

---

## 给 Karpathy 的 open questions

1. **Scaling law 形态**：exponential saturating vs mixture of exponentials vs power law with offset？2 个 pilot 点拟合 2 参数是否过拟合？更多 pilot 点能否改善？

2. **Cluster granularity $M$**：$M=4$ (geo) vs $M=6$ (caption) 都试了，$M=50$ 会不会更好？pilot cost $\propto M$ 如何 trade-off？

3. **Online MOSAIC**：当前 offline 选好 $\mathcal{D}_{sel}$ 再训练。能否 online：边 train 边 fit scaling law 边选样？类似 ADO 的 on-the-fly 思路。

4. **Cross-cluster interaction 建模**：$\Delta U_{ij}$ 能否用 attention/transformer 学习？paper 假设可忽略，但更复杂场景可能不行。

5. **Multi-modal MOSAIC**：能否扩展到 LLM/VLM，metric 是 multiple benchmark aggregate？比如 Llama 4 训练时混合 code/math/multilingual 数据。

6. **Ranking signal cheap alternative**：Section 10 显示 imitation loss / gradient norm 与 EPDMS ranking 相关性低。能否设计更好的 proxy？比如 self-supervised representation quality？

参考：
- Data Filtering Networks: https://arxiv.org/abs/2403.02646
- Active Learning Survey: https://arxiv.org/abs/2203.13450
- Scaling Laws Survey: https://arxiv.org/abs/2405.18392

---

## 最终 takeaway

MOSAIC 把"挑数据"这件事变成了一个**有原则的、可扩展的、multi-metric aware 的**优化问题。核心 insight 是：**在 data-rich + multi-metric + heterogeneous domain 的场景，scaling-aware iterative selection 显著优于 static mixture weighting**。这呼应了 Sorscher et al. 2022 的结论，但推广到了 multi-metric physical AI。

对自动驾驶行业来说，这意味着同样 1000 小时数据，用 MOSAIC 选样训练的模型 EPDMS 比随机选样高 4 分，或同等性能只需 18% 数据。对 NVIDIA 这种自车队数据爆炸的公司，这是直接省钱省 GPU 的工程价值。

对研究界来说，这是把 LLM scaling laws 框架推广到 physical AI 的第一步，后续可扩展到 robotics、VLM、RLHF 等 multi-metric 场景。

---

# Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems - 深度解读

## 1. 问题动机：为什么需要这篇 paper

E2E autonomous driving (AD) 系统面对的核心痛点是：**数据池巨大（physical AI 可达亿小时级别），但多个 competing metrics 必须同时优化，而现有 data mixture 框架无法处理这种 heterogeneity**。

具体来看，传统 data mixture 工作（如 DoReMi、DOGE、Chameleon、ADO、Data Mixing Laws）有三个 implicit assumptions 在 AD 场景下全部不成立：

1. **Domain 同质性假设破裂**：LLM 里 "code" 和 "math" 是 homogeneous domain，但 AD 里 "Pittsburgh 的郊区弯道" 和 "Las Vegas 的高密度城市路口" 对 9 个 rule-compliance metrics 的影响完全不同方向。
2. **单一 metric 假设破裂**：LLM 通常优化单一 validation loss，但 AD 用 EPDMS 聚合 9 个 metrics (NC, DAC, DDC, TLC, EP, TTC, LK, HC, EC)，其中 4 个是 multiplicative penalty，5 个是 weighted average，相互 compete。
3. **可分 domain 假设破裂**：原始数据池不一定预先可分成对 metrics 有 consistent 影响的子集。

MOSAIC 的核心 insight：**先 cluster 把 data 池分成 driving context domain，再对每个 domain 拟合 scaling law，最后用 marginal gain 的一阶差分迭代选样**。

参考链接：
- NAVSIM benchmark: https://github.com/autonomousvision/navsim
- Hydra-MDP (NAVSIM 2024 冠军): https://arxiv.org/abs/2406.06978
- OpenScene: https://github.com/OpenDriveLab/OpenScene
- ADO (Adaptive Data Optimization): https://arxiv.org/abs/2405.18392

---

## 2. 数学公式详解

### 2.1 主问题 (公式 1)

$$
\max_{\mathcal{D}_{sel} \subset \mathcal{D}_{pool}} U\bigg(\{\mathcal{G}_r(\mathcal{D}_{train} \cup \mathcal{D}_{sel})\}_{r=1}^{R}\bigg)
$$

变量含义：
- $\mathcal{D}_{train}$：当前已用 training set (例如 Navtrain 实验中 460 clips 初始化)
- $\mathcal{D}_{pool}$：候选数据池 (例如 4141 clips 待选)
- $\mathcal{D}_{sel}$：要选出的子集，目标 $|\mathcal{D}_{sel}| = B$ (budget)
- $\mathcal{G}_r(\cdot)$：第 $r$ 个 evaluation metric，例如 $\mathcal{G}_1$ = NC (No Collision)
- $R$：metric 数量，本场景下 $R=9$
- $U(\cdot)$：utility function 聚合所有 metric，本场景为 EPDMS

intuition：这是一个 combinatorial optimization，naive 搜索空间是 $\binom{|\mathcal{D}_{pool}|}{B}$，navtrain 中 $\binom{4141}{2400}$ 完全不可枚举。

### 2.2 重构为 mixture optimization (公式 2)

$$
\max_{n_1, \ldots, n_M, \sum_{i=1}^M n_i = B} \Delta U_{mix}(n_1, \ldots, n_M)
$$

变量：
- $M$：cluster 数量 (OpenScene 实验中 $M=4$，按城市分 Las Vegas/Boston/Singapore/Pittsburgh)
- $n_i$：从第 $i$ 个 cluster 取出多少 sample
- $\Delta U_{mix}$：加这些 data 后 utility 的变化量

### 2.3 线性可分近似 (公式 3)

$$
\Delta U_{mix}(n_1, \ldots, n_M) \approx \sum_{i=1}^M \Delta U_i(n_i)
$$

变量：
- $\Delta U_i(n)$：只从 cluster $i$ 加 $n$ 个样本时 utility 的变化

**关键 assumption**：cross-cluster interaction 项 $\Delta U_{ij}(n_i, n_j) = U_{ij} - U_i - U_j + U_0$ 可忽略。paper 在 Section 11 给出实证：实际 EPDMS vs 估计 EPDMS 偏差最多 ~1 EPMS point (见 Table 5，例如 1600 clips 实测 90.2 vs 估计 91.1)，说明 interaction 项存在但小。

### 2.4 Saturating exponential scaling law (公式 4)

$$
\Delta U_i(n) \approx \widehat{\Delta U_i}(n) := a_i (1 - e^{-n/\tau_i})
$$

变量：
- $a_i$：cluster $i$ 的 **asymptotic improvement**，即数据无限多时的总 utility 增益上限
- $\tau_i$：**saturation rate**，控制边际收益衰减速度；$\tau_i$ 越小越快饱和
- $n$：从该 cluster 加的样本数

intuition：这个函数类似 RL 中的 value function，初期 gain 接近 $a_i/\tau_i$ 的斜率线性增长，然后以 $a_i$ 为上界 saturate。

**拟合方式**：用 2 个 pilot runs 拟合。OpenScene 用 200 和 400 clips per cluster (5 epochs continual training)；Navtrain 用 100 和 200 clips per cluster (10 epochs)。

### 2.5 Marginal improvement (公式 5)

$$
\delta_i(b_i) := \widehat{\Delta U_i}(b_i + 1) - \widehat{\Delta U_i}(b_i)
$$

变量：
- $b_i$：当前已从 cluster $i$ 取出 $b_i$ 个样本
- $\delta_i(b_i)$：再加 1 个样本时的边际 utility 增益

这个函数由于 $\widehat{\Delta U_i}$ 关于 $n$ 是 concave 函数 (二阶导数 $-a_i/\tau_i^2 \cdot e^{-n/\tau_i} < 0$)，所以 $\delta_i(b_i)$ 单调递减——这就是 "diminishing return" 的形式化。

### 2.6 EPDMS 公式

$$
\text{EPDMS} := \prod_{m \in \mathcal{M}_{\text{pen}}} m \cdot \frac{\sum_{m \in \mathcal{M}_{\text{avg}}} w_m \cdot m}{\sum_{m \in \mathcal{M}_{\text{avg}}} w_m}
$$

变量：
- $\mathcal{M}_{\text{pen}} := \{\text{NC, DAC, DDC, TLC}\}$：4 个 multiplicative penalty metrics
- $\mathcal{M}_{\text{avg}} := \{\text{EP, TTC, LK, HC, EC}\}$：5 个 weighted average metrics
- $w_m$：weights 依次为 $\{5, 5, 2, 2, 2\}$

各 metric 含义（参考 NAVSIM）：
- **NC** (No Collision)：无碰撞
- **DAC** (Drivable Area Compliance)：保持在可行驶区域内
- **DDC** (Driving Direction Compliance)：行驶方向合规
- **TLC** (Traffic Light Compliance)：红绿灯合规
- **EP** (Ego Progress)：自车前进
- **TTC** (Time-to-Collision)：碰撞时间裕度
- **LK** (Lane Keeping)：车道保持
- **HC** (Hard Comfort)：舒适性（无急刹急转）
- **EC** (Ego Comfort)：综合舒适性

intuition：penalty 项是乘法关系，意味着只要 NC = 0（撞了），EPDMS 直接归零；这是为什么 DAC 从 83.9 提升到 93.59（MOSAIC 在 OpenScene 4000 clips）对 EPDMS 拉动最显著——它提升了乘法项。

---

## 3. 算法流程 (Algorithm 1)

```
Input: D_pool, M, B
1. {D_pool^i,ranked} = ClusterAndRank(D_pool, M)
2. {ΔU_i(n)} = GetScalings({D_pool^i,ranked})   # 公式(4)拟合
3. D_sel ← {}
4. b_i ← 0 for all i
5. while |D_sel| < B:
6.   for i = 1 to M:
7.     δ_i(b_i) ← ΔU_i(b_i+1) − ΔU_i(b_i)      # 公式(5)
8.   end for
9.   j ← argmax_i δ_i(b_i)                      # 选边际增益最大的 cluster
10.  sample ← ReturnSample(D_pool^j,ranked, b_j)
11.  D_sel ← D_sel ∪ {sample}
12.  b_j ← b_j + 1
13. end while
14. return D_sel
```

### 关键步骤说明

**步骤 1: ClusterAndRank**
- Clustering 用两种方式：
  - **Geolocation**：按城市 metadata 分 (Boston, Pittsburgh, Singapore, Vegas)
  - **Semantic captions**：用 Qwen-2.5-VL-32B-Instruct 给每个 10s clip 生成 caption，再 TF-IDF 提取 top 1024 unigrams/bigrams，cluster 成 6 个 domain。caption 示例 prompt："Describe the driving environment that your student is driving through..."
- Ranking 用 EPDMS importance score：
  $$
  \mathcal{T}(x) := U(\{\mathcal{G}_r(f(\cdot; \mathcal{D}_{train}), x)\}_{r=1}^R)
  $$
  即用 base model 在 sample $x$ 上跑一遍得到 9 个 metric，聚合为单个 score。Ranking 时优先取 low $\mathcal{T}(x)$（模型表现差的样本优先选）。

**步骤 2: GetScalings**
- Continual training from base model checkpoint
- 每个 cluster 用 2 个 pilot 点拟合公式 (4) 的 $a_i, \tau_i$

**步骤 5-13: Iterative selection**
- 这本质是 **greedy concave maximization**，等价于一个一阶差分形式的 gradient ascent
- 由于 $\widehat{\Delta U_i}$ 是 concave，全局最优在 KKT 条件下满足 $\delta_i(b_i^*) = \delta_j(b_j^*)$ for all $i, j$ with $b_i^*, b_j^* > 0$

---

## 4. 实验数据深度解读

### 4.1 数据集与协议

| Setting | Train init | Pool | Budgets |
|---|---|---|---|
| OpenScene | 1000 clips | 31539 clips | {250, 500, 1000, 2000, 4000, 8000} |
| Navtrain | 460 clips | 4141 clips | {100, 200, 400, 800, 1600, 2400} |

- 每个 driving log 切成 **10s virtual clip (20 frames @ 2Hz)**，与 industry practice (Waymo) 对齐
- 模型：**Hydra-MDP** + VoVNetV2-99 backbone，trajectory vocabulary 16384
- OpenScene 关掉 rule-based distillation（计算成本太高）
- Random seed: 3 个 (0, 2025, 424242)，OpenScene 大 budget 用 2 个 seed

### 4.2 主结果 (Table 1)

**OpenScene (budget=4000)**：
| Method | EPDMS | BRMR |
|---|---|---|
| Random | 80.38±0.55 | 1.00 |
| Uncertainty | 73.46±0.19 | 2.00 |
| Coreset | 83.63±0.36 | 0.25 |
| Chameleon | 82.92±0.13 | 0.39 |
| **MOSAIC** | **84.25±0.14** | **0.18** |

关键观察：
- MOSAIC 比 Coreset 高 ~0.6 EPDMS，BRMR 0.18 意味着只用 18% 的 data 即可达到 random 全 budget 效果——即 **82% 数据节省**
- Uncertainty 在这个 setting 下表现比 random 还差，原因是 entropy-based selection 偏向 ambiguous 但不一定 informative 的样本

**Navtrain (budget=1600)**：
| Method | EPDMS | BRMR |
|---|---|---|
| Random | 88.62±0.22 | 1.00 |
| Uncertainty | 87.75±0.37 | 1.36 |
| Coreset | 89.30±0.19 | 0.58 |
| Chameleon | 89.50±0.20 | 0.62 |
| **MOSAIC** | **90.18±0.25** | **0.37** |

观察：在 Navtrain 这个已经 curated 过的高质量数据池上，MOSAIC 依然 +0.68 EPDMS over Chameleon。

### 4.3 EPDMS 子项分解 (Table 2)

**OpenScene, budget=4000**：base 模型 DAC=83.9 是最大短板。
| Method | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC |
|---|---|---|---|---|---|---|---|---|---|
| Base | 94.05 | 83.9 | 96.28 | 99.6 | 85.96 | 92.95 | 93.26 | 98.25 | 81.88 |
| Random | 96.32 | 90.53 | 99.06 | 99.79 | 86.36 | 95.66 | 95.68 | 98.30 | 84.46 |
| Coreset | 97.11 | 92.93 | 99.44 | 99.82 | 86.65 | **96.42** | 96.66 | 98.16 | 85.10 |
| Chameleon | 96.76 | 92.32 | 99.51 | 99.77 | 86.98 | 95.91 | 96.49 | **98.32** | **85.51** |
| MOSAIC | 96.97 | **93.59** | **99.59** | 99.80 | **87.14** | 96.18 | 96.62 | 98.28 | 85.06 |

intuition：MOSAIC 优先提升 DAC (从 83.9 → 93.59，提升 9.69 分) 是因为它**通过 scaling law 自动识别 DAC 是 base 模型最弱项**——而 DAC 作为 penalty 项，对 EPDMS 杠杆最大。

### 4.4 Scaling Dynamics (Figure 3, 4)

四个城市 scaling 曲线形态不同：
- **Boston**：低 data regime 增长最快，~500 clips 后接近饱和
- **Singapore**：类似 Boston，但 saturation 略慢
- **Pittsburgh**：稳定线性增长，高 budget (>2000) 时主导
- **Las Vegas**：早期 gain 最小，saturation 也最快

迭代选择过程 (Figure 4)：
- 第 0-500 iteration：主要选 Boston + Singapore
- 第 500-3700：转为 Pittsburgh 主导
- 第 3700+：Pittsburgh 耗尽，转向 Vegas

这表明 MOSAIC 的 scaling-aware 策略自动执行 **regime-dependent budget allocation**——这比静态 Chameleon 的固定 mixture weight 更优。

### 4.5 Ablation: Clustering vs Ranking (Figure 6)

- **w/o Clustering**：直接对所有 pool 按 $\mathcal{T}(x)$ 排序 greedy 取 top-B
- **w/o Ranking**：保留 clustering 但放弃 ranking，每个 cluster 随机采样

结果：
- 低 budget (<800)：ranking 主导，w/o Clustering 和 MOSAIC 接近
- 高 budget：clustering + scaling 主导，w/o Clustering 开始落后
- w/o Ranking 也 beats random，但不如 MOSAIC

intuition：这告诉我们 **ranking 是 data-efficient regime 的引擎**（先用最有用的 sample），**scaling-aware cluster selection 是 data-abundant regime 的引擎**（决定何时换 cluster）。

### 4.6 Compute 分析 (Figure 11)

虽然 MOSAIC 需要 pilot runs 来拟合 scaling law，但整体 compute 仍更优：
- vs Coreset：达到同等 EPDMS 省 16% compute (~490 A100 GPU hours)
- vs Random：省 57% compute (~1700 A100 GPU hours)

---

## 5. Baselines 详解

### 5.1 Uncertainty (Algorithm 2)
对每个 clip 计算轨迹 logits 的 Shannon entropy：
$$
H_i = -\sum_k p_{i,k} \log p_{i,k}, \quad p_i = \text{softmax}(z_i)
$$
取 top-B entropy 最大的 clip。**为什么效果差**：高 entropy 可能意味着模型见过类似场景但不确定，更可能意味着 OOD 噪声样本。

### 5.2 Coreset (Algorithm 3)
经典 geometric diversity：
$$
u = \arg\max_{i \in s^{pool}} \min_{j \in s} \Delta(x_i, x_j)
$$
每次选离当前已选集最远的样本，欧氏距离。Coreset 在 OpenScene 表现不错 (83.63)，但在 Navtrain 上不如 Chameleon/MOSAIC，原因是 Navtrain 已 curated 过，diversity 不是瓶颈。

### 5.3 Chameleon (Algorithm 4)
- 每个 cluster embedding: $x_i = \frac{1}{|D_i|} \sum_{a \in D_i} h_\theta^{(L)}(a)$
- Affinity matrix: $\Omega_D = X X^\top$
- KRLS scores: $S_\lambda(D_i)$ via kernel ridge regression
- Mixture weights: $\alpha_i^{PT} = \text{softmax}(S_\lambda^{-1}(D_i))$

Chameleon 的弱点：对 cluster 结构敏感，caption-based clustering 时性能崩塌（Navtrain caption 实验中 BRMR 高达 3.32）。

---

## 6. 与相关工作对比

| 方法 | 域结构 | Metric 结构 | Scaling 建模 | 适合场景 |
|---|---|---|---|---|
| DoReMi (2023) | 显式 | 单一 loss | 无 | LLM pretraining |
| DOGE (2023) | 显式 | 单一 loss | gradient tracking | LLM pretraining |
| Chameleon (2024) | 显式 | 单一 loss | kernel similarity | LLM/VLM |
| ADO (2025) | 显式 | 单一 loss | on-the-fly scaling | LLM |
| Data Mixing Laws (2025) | 显式 | 单一 loss | regression | LLM |
| **MOSAIC** | **隐式 cluster + ranking** | **multiple competing** | **per-cluster exponential** | **physical AI** |

---

## 7. Limitations

paper 自承两个 limitation，但实际上还有几个隐藏问题：

1. **Linear separability 假设**：公式 (3) 忽略 cross-cluster interaction $\Delta U_{ij}$。如果 clustering 不 semantic（例如 random cluster），interaction 项会主导，MOSAIC 退化。

2. **Pilot run 成本**：每个 cluster 需要 2 个 pilot run。对超大 pool (例如 100M clips)，cluster 数 $M$ 不能太大，否则 pilot 成本爆炸。

3. **Ranking signal 依赖 dense annotation**：$\mathcal{T}(x)$ 需要 9 个 rule-compliance 评估，依赖 bounding box 等标注。Section 10 ablation 显示 trajectory imitation loss / gradient norm / gradient perturbation 都与 EPDMS ranking 相关性极低（Kendall-Tau ≈ 0），暗示 cheap signal 替代品难找。

4. **Saturation exponential 假设**：公式 (4) 假设单一 saturating 曲线。如果 cluster 内部存在 sub-domain（例如 Boston 内部有 highway + urban），实际 scaling 可能是 multiple saturation 叠加，单一 $a_i, \tau_i$ 拟合不准。

5. **Cluster 数 $M$ 敏感性**：paper 中 $M=4$ (geolocation) 或 $M=6$ (caption)。$M$ 太小 cluster 同质性差，$M$ 太大 pilot 成本线性增长。paper 没系统 ablate $M$ 的影响。

---

## 8. 我的延伸思考

### 8.1 与 scaling laws 的深层联系
MOSAIC 的 scaling law $\Delta U_i(n) = a_i(1 - e^{-n/\tau_i})$ 与 Kaplan et al. 2020 的 power-law $L(N) \propto N^{-\alpha}$ 形态不同。**为什么不用 power law？** 因为 power law 描述 **loss decay**，而 MOSAIC 描述 **gain accumulation**。Exponential saturating 是 gain 的自然形式，类似于 RL 中的 value iteration convergence。

### 8.2 与 Active Learning 的关系
MOSAIC 在 spirit 上接近 **batch active learning with scaling forecasts**。但传统 AL 假设 oracle label cost，MOSAIC 假设 unlabeled pool 和固定 inference cost。可以说 MOSAIC 是 AL 的 "data mixture extension"。

### 8.3 推广到 Robotics / LLM
MOSAIC 框架原则上可推广到：
- **Robotics manipulation**：multi-task metric (success rate, energy, smoothness)
- **VLM training**：multi-benchmark (VQA, captioning, OCR, grounding)
- **RLHF**：multi-objective reward (helpfulness, harmlessness, honesty)

关键前提是：能够 cluster data into domain，且每个 domain 对每个 metric 有 differential scaling。

### 8.4 与 Data Filtering Networks (DFN) 的对比
近期 DFN 工作 (Fang et al. 2024) 用小模型给数据打分。MOSAIC 的 $\mathcal{T}(x)$ 也是 model-based scoring，但 aggregation 用 EPDMS 而非 loss，更适合 multi-metric physical AI。

### 8.5 Sample efficiency 解释
为什么 MOSAIC 能省 80% data？核心是它**同时利用了 cluster-level heterogeneity 和 within-cluster ranking**。Random 既不区分 cluster 也不 ranking；Coreset 只用 cluster-level diversity 但 ignore metric relevance；Chameleon 用 cluster-level metric relevance 但 ignore within-cluster ranking；MOSAIC 把两个 axes 都建模了。

---

## 9. 一些细节和数字感

### 9.1 2400 clips = full training performance

Navtrain 实验中 pool = 4141，MOSAIC 在 2400 clips 达到 full-pool performance (90.31 vs ~90.5)。这相当于 **42% data 节省**。注意 full-pool training 不是 upper bound，因为 Navtrain 本身就是 OpenScene 的 curated subset——理论上更多 curated data 还能继续提升。

### 9.2 EPDMS 的 multiplicative structure 影响

EPDMS = (NC × DAC × DDC × TLC) × weighted_avg(...)
- 4 个 penalty 项相乘，任何一个为 0 整个 EPDMS 归零
- DAC 从 83.9 → 93.59 在 multiplicative 上的 leverage = 93.59/83.9 = 1.115，相当于对 EPDMS 直接 11.5% 的拉动
- 这解释了为什么 MOSAIC 优先攻 DAC：scaling law 自动发现 DAC 在 base 模型上 margin 最大

### 9.3 Geolocation vs Caption clustering

paper Section 4.4 ablation：caption clustering 上 Chameleon 性能崩塌 (Table 8 中 BRMR 1.50-3.32)，但 MOSAIC 依然 robust (BRMR 0.37-0.48)。这暗示 MOSAIC 的 scaling-aware 机制对 cluster quality 不敏感——它可以从 noisy cluster 中识别出真正 effective 的样本。

参考链接补充：
- Data Filtering Networks: https://arxiv.org/abs/2403.02646
- DoReMi: https://arxiv.org/abs/2305.10429
- DOGE: https://arxiv.org/abs/2310.15393
- Data Mixing Laws: https://arxiv.org/abs/2403.16952
- Beyond Neural Scaling Laws (Sorscher et al.): https://arxiv.org/abs/2206.14491

---

## 10. 实现层面的小细节

### 10.1 Virtual clip 创建
- 每个 driving log (30s-50min) 切成 10s virtual clip
- 2Hz → 20 frames per clip
- non-overlapping sequential
- 不满 10s 的尾部丢弃
- Navtrain: 4,601 clips (10.9% frames discarded)
- OpenScene: 32,539 clips (1.8% frames discarded)

### 10.2 Hyperparameters
- Navtrain: 8×A100, lr=1e-4, batch=20/GPU
- OpenScene: 16×A100, lr=2e-4, batch=20/GPU, fixed 40 epochs
- Coreset distance: Euclidean
- Chameleon ridge: $\lambda = 1$
- Qwen caption prompt 限制 <150 words，无 maybe/might/possibly

### 10.3 Caption cluster 语义 (Table 3)
| Cluster | Top unigrams/bigrams |
|---|---|
| 1 | calm, day, street, trees, signs, yellow |
| 2 | signals, crossing, crosswalks, pedestrians |
| 3 | highway, vehicles, busy urban, palm trees |
| 4 | building, area, large, paved, parking |
| 5 | city street, major city, moderate |
| 6 | precipitation, potential rain, overcast, cloudy |

这 6 个 cluster 语义上对应：郊区 / 行人路口 / 高速 / 停车场 / 城市主路 / 雨天——比 geolocation 4 cluster 更细，但 MOSAIC 在其上依然 robust。

---

## 11. 总结：这篇 paper 的真正贡献

**核心 contribution 不在 SOTA 数字，而在于把 LLM scaling laws 框架扩展到 multi-metric physical AI 的方法论**：

1. **Cluster + Rank + Scale** 三段式 pipeline 把不可解 combinatorial 优化拆成可解的三步
2. **Exponential saturating scaling law** 适配 gain accumulation 而非 loss decay
3. **First-difference greedy** 把 concave maximization 转成 O(MB) 的迭代
4. **Cross-metric disentanglement** 通过 clustering 把数据 vs metric 的 heterogeneous influence 结构化

**最终 takeaway**：在 data-rich、multi-metric、heterogeneous domain 的 physical AI 场景，**scaling-aware iterative selection 显著优于 static mixture weighting**。这呼应了 Sorscher et al. (2022) "Beyond Neural Scaling Laws" 的结论，但推广到了 multi-metric 设定。

---

## 12. 给 Karpathy 的 open questions

1. **Scaling law 形态选择**：为什么是 exponential saturating 而不是 mixture of exponentials 或者 power law with offset？实证上 pilot run 2 个点拟合 2 个参数 $a_i, \tau_i$ 是否过拟合？
2. **Cluster granularity**：$M$ 选 4-6 是因为 pilot cost 还是 empirical optimum？更大 $M$ (例如 50) 会不会更好？
3. **Online MOSAIC**：当前是 offline 选好 D_sel 再 train。能否做 online 版本：边 train 边 fit scaling law 边选样？
4. **Cross-cluster interaction 建模**：paper 假设 $\Delta U_{ij}$ 可忽略，但若用 attention/transformer 学习 interaction term 是否更准？
5. **Multi-modal MOSAIC**：能否扩展到 LLM/VLM，其中 metric 是 multiple benchmark 的 aggregate？

参考：
- Beyond Neural Scaling Laws: https://arxiv.org/abs/2206.14491
- Scaling Laws for Neural Language Models: https://arxiv.org/abs/2001.08361
- Active Learning Survey: https://arxiv.org/abs/2203.13450
- CoreSet paper: https://arxiv.org/abs/1708.00489

这篇 paper 的 code 应该会开源在 NVIDIA 的 GitHub，目前看 Hydra-MDP 的 repo 在 https://github.com/OpenDriveLab/DriveAGI 可以参考 base 模型实现。
