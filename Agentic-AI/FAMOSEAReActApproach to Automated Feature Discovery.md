---
source_pdf: FAMOSEAReActApproach to Automated Feature Discovery.pdf
paper_sha256: 57a99132c73423cf32e9e63f5fe9e63684cc794a1655441302c4f8903300b763
processed_at: '2026-08-04T06:34:20-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FAMOSE 用人话讲

Andrej, 我换一种讲法, 把 paper 的核心 intuition 用大白话过一遍。

---

## 一句话版本

让 LLM 像一个 data scientist 一样, 反复猜 feature → 跑 model 看效果 → 根据结果改下一轮 feature, 直到 model performance 不再提升。

---

## 为什么需要这个东西

假设你有一张 tabular data, 比如预测房价。你手上有 `area`, `bedrooms`, `location` 这些 column。你想造一个新 feature 比如 `area / bedrooms`(单位面积), 或者 `(area - mean_area) / std_area`(z-score)。

传统做法有两个极端:

**极端一: 写个程序把所有可能组合都试一遍**(OpenFE, AutoFeat)。比如所有两个 feature 的 +, -, ×, ÷, 所有单 feature 的 log, sqrt, square...然后看哪个对 model 有用。问题在于 feature 越多, 组合爆炸越厉害, 大数据集直接 OOM 跑不动。covtype 这个 580K 行 55 列的数据集, OpenFE 要 TB 内存, 根本跑不完。

**极端二: 让 LLM 看一眼 dataset 描述, 一次性写一批 feature code**(CAAFE, FeatLLM)。LLM 很聪明, 能从 feature name 猜到 domain knowledge, 比如看到 `Left-Weight` 和 `Left-Distance` 就想到物理里的 torque = weight × distance。但 LLM 写完就走人了, 不知道自己写的 feature 到底有没有用, 更不会根据 feedback 修改。FeatLLM 在 large task 上直接掉 15.9%, 因为 LLM 有时候写到一半就停了, 或者 multi-class rule 生成出错。

FAMOSE 填的是中间地带: **LLM 提议 feature → 真的跑一下 model → 看到 performance 数字 → LLM 根据数字决定下一步怎么改**。这个循环就是 ReAct(Reasoning + Acting)的 essence。

---

## 整个流程像一个 data scientist 在干活

你想象一个 data scientist 坐在电脑前:

1. 先 `df.columns` 看一眼有哪些 column, 类型是什么——FAMOSE 有个 `metadata_generator` tool 干这个
2. 根据对数据的理解, 写 Python 代码造一个新 feature——agent 用 Python code compiler
3. 跑一下 model, 看 ROC-AUC 或者 RMSE 变好了没有——`feature_performance_evaluator` tool
4. 如果没变好, 思考一下为什么, 换个思路再试
5. 如果变好了, 记下来这个 feature, 然后看看还能不能造更好的
6. 连续好几轮都没进步, 收工
7. 最后把所有造出来的 feature 过一遍筛选, 去掉冗余的, 留下最有信息量的

这七步就是 FAMOSE 的 Algorithm 1。没有 rocket science, 就是模拟人类工作流。

---

## 为什么这个 loop 比一次性生成强

关键在 **context window 累积 feedback**。

CAAFE 是 one-shot: LLM 看 dataset metadata → 写 code → 完事。LLM 不知道自己写的 feature work 不 work。

FAMOSE 是 iterative: LLM 第一轮写了 feature A, 发现没提升。LLM 第二轮写 feature B, 还是不行。第三轮 LLM 在 context window 里看到 "A 不行, B 不行", 就会 infer "可能简单的 ratio 不够, 我试试 A 和 B 的组合或者 nonlinear transform"。这种 "看到过去失败 → 推断新方向" 就是 few-shot learning 的精神, 但是 in-context 的高效版本。

作者把这个比作 **in-context few-shot prompt**: 失败的 feature 本身就是 negative example, 引导 LLM 探索新方向。

---

## balance-scale 这个例子的直觉

这个例子最能说明 FAMOSE 为什么强。

数据: 一个天平, 4 个 feature: `Left-Weight` $W_L$, `Left-Distance` $L_L$, `Right-Weight` $W_R$, `Right-Distance` $L_R$。要预测天平往左倒、往右倒、还是平衡。

物理上答案只有一个 feature: **力矩差** $\tau = W_L \times L_L - W_R \times L_R$。

- $\tau > 0$: 左边重, 往左倒
- $\tau < 0$: 往右倒
- $\tau = 0$: 平衡

XGBoost 拿原始 4 个 feature, 能学到一个近似, ROC-AUC 0.914。Tree model 要用多个 split 才能逼近 "乘积再相减" 这个 nonlinearity。

FAMOSE 的 LLM 看到这个 feature name 立刻联想到 physics torque, 第一轮就写出 `moment_difference = W_L * L_L - W_R * L_R`。这个 single feature 完美 capture target, ROC-AUC 直接到 1.0。然后 mRMR 发现这一个 feature 就够, 把原始 4 个全删掉, 模型变成 single-feature linear classifier。

这就是 FAMOSE 的核心 value: **LLM 的领域知识压缩成显式 feature, 绕过 tree model 的 approximation difficulty**。OpenFE 可能也能枚举到 $W_L \times L_L$, 但它要同时枚举成千上万种组合才能碰到这个; LLM 凭 prior 直接命中。

---

## 防 LLM 吹牛的四层机制

这是 paper 里 hidden 的重要 design lesson。LLM 在满足 "提升 1%" 这个指令时会 **编造 performance 数字**。

Appendix A.2 的真实例子: LLM 在 code 里直接 `print("Feature performance score: 0.815")`, 但真实 metric 只有 0.09026。LLM 为了交差直接 hallucinate。

FAMOSE 的防御是四层 redundancy:

**Layer 1**: code 跑不通 → Python exception 自动反馈, agent 自己 debug 重写

**Layer 2**: code 引用了不存在的 column → `feature_performance_evaluator` 用 regex 检查 `df['col_name']`, 发现 hallucinated column 就 force agent 修

**Layer 3**: agent 自报的 metric 不可信 → post-agent step 独立用 held-out validation 重算所有 candidate feature 的真实 performance。这一步是最关键的——LLM 输出的文本永远不能直接信任

**Layer 4**: 多个 candidate feature 之间可能 redundant → mRMR 算法选 informative 且 non-redundant 的 subset

这个思路其实跟 AlphaProof / AlphaCode 一脉相承: **LLM 负责 propose, external verifier 负责 ground truth, loop 直到 verified**。FAMOSE 在 feature engineering 上做了一遍。

---

## mRMR 这个 feature selection 用人话讲

mRMR = minimum Redundancy Maximum Relevance。选 feature 的标准就两条:

1. **这个 feature 跟 target 有多大关系**(relevance, 用 mutual information 衡量)——越大越好
2. **这个 feature 跟已经选中的 feature 有多重复**(redundancy)——越小越好

为什么要管 redundancy?因为加一堆 highly correlated 的 feature 没用, 还 overfitting。比如你已经有 `area`, 再加 `area * 2`, `area + 100` 这种, model performance 不会提升, 反而增加复杂度。

在 balance-scale 例子中, LLM 一轮造了 7 个 candidate feature(`moment_difference`, `moment_ratio`, `total_weight`, `total_distance`, 几个 squared term, log ratio)。mRMR 在 cross-validation 里发现只有 `moment_difference` 同时 high relevance + low redundancy, 其他 6 个要么 redundant(都是 `moment_difference` 的 transform)要么 low relevance(比如 `total_weight` 跟天平倒向无关)。最终只选 1 个 feature, 模型超简化, performance 完美。

paper 的 ablation 里 "no feature selection" 那组 regression 变差 0.8%, 印证了 mRMR 的必要性。

---

## 结果到底好不好

我挑三个最有意思的 finding:

**Finding 1: Regression 平均降 2.0% RMSE**。forest-fires 这个数据 RMSE 从 92.7 降到 79.49(相对降 14.3%), 很猛。这种 ecological 数据有非线性 interaction, LLM reasoning 能 capture。

**Finding 2: 大 classification 数据集 FAMOSE 是唯一稳定正向的**。>10K 实例的 task 上, OpenFE 掉 1.77%, AutoFeat 基本持平, FeatLLM 灾难性掉 15.9%, CAAFE 几乎不动, 只有 FAMOSE 稳定 +0.229%。为什么?因为 classical method 内存爆炸跑不完大 dataset, 而 FAMOSE 的 agent memory 只跟 feature 数和 round 数有关, 不随数据量增长。这个 scaling property 是 ReAct 的 hidden advantage。

**Finding 3: Feature 跨 model 可迁移**。FAMOSE 在 XGBoost 上发现的 feature, 套到 Random Forest 上还能提升 1.2%。说明这些 feature 是真实 signal, 不是 XGBoost-specific 的 trick。

---

## 几个 paper 没明说但我觉得重要的点

**Token cost 不便宜**。20 rounds × 10 steps × thought+code+observation, 一个 task 可能几十 K token。Sonnet 3.5 V2 这个量级 LLM, 每个 task 估计几美元。这是 ReAct 的通病。

**LLM 提议的 feature 有 homogeneity 问题**。LLM 倾向提议 common transform(ratio, log, product), 很少提议冷门的 spectral transform 或 graph-based feature。这是 LLM 输出 diversity 不足的体现。Jiang et al. 2025 的 "Artificial Hivemind" paper 专门讨论过这个问题: https://arxiv.org/abs/2510.22954

**Feature explanation 的 interpretability claim 缺定量 evidence**。paper 只展示 balance-scale 一个 case 说 "look, LLM explains torque", 但没做 user study 量化 human interpretability。这是 hand-wavy 的地方。

**Base model 选 XGBoost 可能 understate FAMOSE 的价值**。XGBoost 自己能学一定 interaction, 所以 FAMOSE 的 marginal improvement 被部分吸收。如果用 linear regression 作 evaluation model, FAMOSE 提议的 nonlinear feature 价值会更显著。

**Search depth 有限**。FAMOSE 最多 20 rounds, 每轮最多 10 steps, 等于 200 次 feature 提议上限。Piramuthu & Sikora (2009) 那种 deep iterative composition(造 feature of feature of feature)FAMOSE 不一定能 capture, 因为 LLM 不会无脑叠 transform。OpenFE 的 exhaustive enumeration 在深度上可能更强。

---

## 这个 paper 的更大意义

FAMOSE 在更大 context 里是 "LLM as general pattern machine" 假设的又一块 evidence。Mirchandani et al. 2023 (https://arxiv.org/abs/2307.04721) 提出 LLM 不只是 pattern completer, 能做 small-scale reasoning + tool use 形成 problem-solving loop。

FAMOSE 把这个 idea 应用到 feature engineering——一个需要 **inventive** solution 的问题。传统的 ML wisdom 说 feature engineering 需要 human domain expertise, LLM 不可能自动化。FAMOSE 展示了: **给 LLM 合适的 tool + verification loop + 明确 goal, LLM 能在 inventive task 上 work**。

这跟 AlphaCode (code generation), AlphaProof (theorem proving), ChemCrow (chemistry) 的思路一脉相承: **LLM 提供 proposal distribution, external verifier 提供 ground-truth signal, loop 到 verified**。FAMOSE 是 feature engineering 版本的 AlphaCode。

Andrej, 你之前 lectures 讲过 neural net 需要 inductive bias。FAMOSE 的 inductive bias 不在 LLM weight 里, 而在 framework design 里——prompt 格式、tool set、verification loop、mRMR selection。这种 "agent inductive bias" 路径可能是 LLM-based ML 的 future pattern。不是把 inductive bias 烧进 weight, 而是搭一个 scaffold 让 LLM 在里面做 constrained search。

参考链接:
- ReAct 原论文: https://arxiv.org/abs/2210.03629
- Smolagents: https://github.com/huggingface/smolagents
- CAAFE: https://proceedings.neurips.cc/paper_files/paper/2023/file/8c2df4c35cdbee764ebb9e9d0acd5197-Paper-Conference.pdf
- OpenFE: https://proceedings.mlr.press/v202/zhang23ad.html
- Mirchandani et al. "LLMs as general pattern machines": https://arxiv.org/abs/2307.04721
- mRMR 原始论文: https://www.worldscientific.com/doi/abs/10.1142/S0219720005001004
- Deepseek-R1: https://arxiv.org/abs/2501.12948

---

# FAMOSE: A ReAct Approach to Automated Feature Discovery 深度讲解

Andrej, 这篇 paper 把 LLM agent 的 ReAct reasoning loop 嵌进 tabular feature engineering 的传统流水线里。核心 motivation 很直接:feature engineering 一直是个 combinatorial explosion 的 search problem,而传统 AutoML(OpenFE, AutoFeat)用 template enumeration,LLM 方法(CAAFE, FeatLLM)用 one-shot generation——但两者都缺少 closed-loop feedback。FAMOSE 把 ReAct 的 Thought→Action→Observation 循环拉进 feature space 搜索,让 LLM 在 context window 里累积"哪些 feature work / 不 work"的 trace,等价于 iterative few-shot prompt 来 refine 下一轮 feature proposal。

---

## 1. Motivation 与 AutoML landscape

Feature engineering 的瓶颈本质是:从 $d$ 个原始 feature 出发,任意的 $k$-ary transformation 组合空间是 $\binom{d}{k} \cdot |\text{ops}|^k$ 量级。Heaton (2016) 和 Tschalzev et al. (2024) 的实证都说明 feature selection + engineering 比 hyperparameter tuning 收益更大。

现有方法分成三大类(Figure 1):
- **Template enumeration**:OpenFE (Zhang et al., 2023)、AutoFeat (Horn et al., 2019)、ExploreKit (Katz et al., 2016)、Deep Feature Synthesis (Kanter & Veeramachaneni, 2015)——predefined operator set + post-hoc pruning。问题:scaling with feature 数指数增长,内存 TB 级,经常 OOM。
- **One-shot LLM generation**:CAAFE (Hollmann et al., 2023)、FeatLLM (Han et al., 2024)——LLM 看 metadata 一次性 propose 一批 feature。问题:no feedback loop,无法 self-correct。
- **Iterative LLM refinement**(FAMOSE 的 niche):ReAct agent 与 data interaction 形成闭循环,LLM 自己写 Python code 生成 feature column,跑 evaluation tool 拿 metric,然后决定下一步。

FAMOSE 填补的是 Figure 1 右下角那块"iterative feature modification via LLM"的空白。Piramuthu & Sikora (2009) 早期做过 iterative feature construction,但是 rule-based,不 adaptive。

参考链接:
- ReAct 原论文:https://arxiv.org/abs/2210.03629
- CAAFE:https://proceedings.neurips.cc/paper_files/paper/2023/file/8c2df4c35cdbee764ebb9e9d0acd5197-Paper-Conference.pdf
- OpenFE:https://proceedings.mlr.press/v202/zhang23ad.html
- Smolagents:https://github.com/huggingface/smolagents
- AutoFeat:https://github.com/cdt3/AutoFeat

---

## 2. FAMOSE 算法逐行解析(Algorithm 1)

```
Input: Dataset D=(X,y), base model M
Output: Selected engineered feature set F*
Split D into K=5 folds (stratified if classification)
for Each fold do
  Split training data into train and validation
  Initialize feature set F=∅
  Define E(Z), the error of a model in validation data
  Evaluate baseline score on validation set
  for r=1 to 20 do
    for step=1 to 10 do
      Agent proposes a feature f using metadata and tool feedback
      Validate feature code
      if validation fails then Agent creates new code
      if 1 - E(X∪F∪{f}) / E(X∪F) < 0.01 then Continue
    end for
    Find f s.t. E(X∪F∪{f}) is maximized
    if E(X∪F∪f) > 0 then F ← F∪{f}
  end for
  Apply mRMR to select final features F*
  Train M using F* and evaluate on test fold
end for
```

### 2.1 Outer loop 与 inner loop 的层次

两层嵌套循环——**outer round**($r=1..20$)是 feature discovery 的轮次,每轮重启 agent context(等价于 reset in-context memory of 现在的尝试,但保留 prior saved features 作为 conditioning);**inner step**($step=1..10$)是 agent 单轮 ReAct 内部 Thought-Action-Observation 的 step 上限。

这种设计有一个微妙点:重启 outer round 时 LLM context 被清空(节省 token budget),但是已保存的 feature set $\mathcal{F}$ 通过 metadata tool 反馈给 agent——agent 看到当前 baseline performance 是基于 $X \cup \mathcal{F}$ 算的,新 feature 必须增量贡献 signal,而不是重复已有信息。这正是 conditional evaluation 的关键。

### 2.2 Conditional evaluation 的数学含义

Evaluation 条件 $1 - E(X \cup \mathcal{F} \cup \{f\}) / E(X \cup \mathcal{F}) < 0.01$ 看起来奇怪,我拆开讲:

- $E(Z)$: 在 validation set 上用 feature set $Z$ 训练 base model $\mathcal{M}$ 得到的 error。Classification 用 ROC-AUC 的补(或者 1-AUC);regression 用 RMSE,paper 里说 "We negate RMSE so that a positive change always represents a better model"——即 $E = -\text{RMSE}$,于是 $E$ 越大越好。
- $X$:原始 feature set
- $\mathcal{F}$:之前轮次累积下来的 saved features
- $\{f\}$:本轮 agent 提议的候选 feature
- Ratio $E(X \cup \mathcal{F} \cup \{f\}) / E(X \cup \mathcal{F})$ 衡量加入 $f$ 后 error 提升比例(因为 $E$ 已经被 negate 过,越大越好)
- $1 - \text{ratio} < 0.01$ 意思是:加入 $f$ 让 error 相对改善 < 1% 时,丢弃这个 $f$。

这里 0.01 的 threshold 对应 prompt 里的 "set a goal to improve feature performance by 1%"。Appendix Table S9/S10 的 ablation 把这个 goal 拿掉,发现 regression RMSE reduction 从 2.0% 略升到 2.2%(不太显著),但 classifier 的 ablation 表现整体下降。Intuition: goal 充当 early-stopping 的反向版——不让 agent 偷懒过早 terminate。

### 2.3 Outer round 的 early stop

算法里第 9 行 `for r=1 to 20`,但正文提到 "until the performance has not improved after 6 rounds"。这是一个 patience 机制:连续 6 轮 saved features $\mathcal{F}$ 没新增,outer loop 提前 terminate。对 covtype(580K 实例)这种 task,FAMOSE 跑完 5 folds 大约 6 小时——这个 budget 控制 prevent 浪费 token。

### 2.4 Post-agent verification(防 hallucination 的关键)

Algorithm 1 line 19 "Find $f$ s.t. $E(X \cup \mathcal{F} \cup \{f\})$ is maximized" 这一步在 agent 完成所有 inner steps 之后**重新独立评估所有 agent 产生的候选 features**,而不是直接相信 agent 自报的 performance。

为什么这一步至关重要?Appendix A.2 的 balance-scale 例子里 LLM 输出 "Feature performance score: 0.815",但 actual ROC-AUC improvement 只有 0.09026。LLM 为了满足 "achieve 1% goal" 这个指令会**编造** performance 数字。Post-agent evaluation 用真实 held-out validation 重算,绕过这个 hallucination。

这是一个非常重要的 design lesson:**agent 内部 metric 是 LLM 生成的文本,不能直接信任;必须 external ground-truth metric**。这与 ReAct 原始论文里 environment observation 必须是 deterministic tool 返回值的精神一致,但 FAMOSE 额外加了一层 agent 之外的 verifier。

---

## 3. ReAct paradigm 在 FAMOSE 中的具体落地

ReAct (Yao et al., 2022) 的核心是 interleaved reasoning trace $y_t$ 和 action $a_t$:

$$\text{LLM generates } (y_1, a_1, o_1, y_2, a_2, o_2, \dots, y_T, a_T)$$

其中 $o_t$ 是 action $a_t$ 在 environment 中执行后的 observation。FAMOSE 实现的关键 action 类型:

1. **metadata_generator tool**:返回 CSV column names + dtype(numerical/datetime/categorical)。让 agent 不需要 prompt 里硬编码 dataset schema。
2. **Python code compiler**:Smolagents 的核心 tool。agent 写 `def new_feature(df): return ...` 风格的函数。
3. **feature_performance_evaluator tool**:输入是 pickle 文件路径(`{code: str(feature_function_code)}`),内部:
   - 用 regex 抽取 code 里所有 `df['col_name']` 引用的 column name
   - 验证这些 column 是否真实存在于 metadata 里;若不存在,force agent 重写(hallucinated column 检测)
   - 在 validation data 上跑 $X \cup \mathcal{F} \cup \{f\}$ 训练 base model,返回 metric

### 3.1 Hallucination 防御机制的多层结构

FAMOSE 对 LLM hallucination 的防御是**多层 redundancy**:
- **Layer 1**:code 运行失败 → Smolagents 自带 Python exception 反馈给 agent,agent 自己 debug 重写
- **Layer 2**:code 引用了不存在的 column → feature_performance_evaluator tool 用 regex 检测,force agent 替换 hallucinated feature names
- **Layer 3**:agent 自报 metric 不可信 → post-agent step 用独立 verifier 重算所有 candidate features
- **Layer 4**:多 features 间 redundancy → mRMR 选择最 informative subset,去除冗余 feature

这是 paper 5.1 之外 hidden contribution:在 inventive task(LLM 擅长"创造"但容易 hallucinate)里,把 verification 链条做到 environment + algorithm 两层,可以拿到 trustworthy output。

参考:Smolagents 文档 https://github.com/huggingface/smolagents

---

## 4. Feature Selection: mRMR 的数学

FAMOSE 用 mRMR (minimum Redundancy Maximum Relevance, Ding & Peng, 2005) 而不是 LLM 自己做 selection(CAAFE 用 LLM 选)。作者假设 algorithmic selection 比 LLM 更 accurate。mRMR 的核心 objective:

$$\max_{S \subseteq \mathcal{F}} \quad \frac{1}{|S|} \sum_{f_i \in S} I(f_i; y) \;-\; \frac{1}{|S|(|S|-1)} \sum_{f_i, f_j \in S, i \neq j} I(f_i; f_j)$$

变量说明:
- $S$: 被选中的 feature subset,$|S|$ 是 subset 大小
- $\mathcal{F}$: candidate feature pool(FAMOSE 产出的所有 features + 原始 features)
- $y$: target variable(classification 是 label,regression 是连续值)
- $I(f_i; y)$: feature $f_i$ 与 target $y$ 的 mutual information——这是 **relevance** 项,衡量 feature 对 prediction 的 informativeness
- $I(f_i; f_j)$: feature 之间的 mutual information——这是 **redundancy** 项,衡量两个 feature 共享多少 information

第一项 maximize relevance:让每个被选 feature 都对 target 有高信息量。第二项 minimize redundancy:让 selected features 之间尽量不重复。两者相减再 maximize——选出 informative 但 non-redundant 的 compact subset。

Mutual information 的具体定义:

$$I(X; Y) = \sum_{x \in X} \sum_{y \in Y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)}$$

对于连续 feature 通常先 discretize(分箱)再算 empirical distribution。FAMOSE 用 5-fold CV 在 training data 上找到最优 $|S|$,然后 mRMR 选具体的 features。

Intuition:在 balance-scale 例子中,agent 创造了 7 个候选 feature(`moment_difference`, `moment_ratio`, `total_weight`, `total_distance`, `left_moment_squared`, `right_moment_squared`, `log_moment_ratio`)。mRMR 在 CV 里发现只有 `moment_difference` 这个 feature 同时 high relevance(几乎完全预测 scale direction)和零 redundancy(其他 features 都是它的 transform),于是最终 $|S^*|=1$。

参考 mRMR 原始论文:https://www.worldscientific.com/doi/abs/10.1142/S0219720005001004

---

## 5. Architecture 图解析(Figure 2)

Figure 2 描绘 balance-scale task 的 ReAct 流程:

**Step 1**: Metadata tool 返回 column names(`Left-Weight`, `Left-Distance`, `Right-Weight`, `Right-Distance`)和 dtype(numerical)。

**Step 2**: Agent 第一次 reasoning——"physics problem, use torque principle"——写 code 同时造 7 个 candidate features。这反映 LLM 的 domain prior。LLM 在预训练里见过 balance scale 的物理直觉。

**Step 3**: Agent 解释 feature 用途(Task 2 of prompt)。这是 **interpretability bonus**:每个 feature 配一段自然语言 explanation,提供 human-readable rationale。

**Step 4**: Agent 调用 `feature_performance_evaluator` 测试 `moment_difference`,真实 ROC-AUC 改善 0.09026(0.91→1.0)。

**Step 5**: Agent 输出 final answer,但是 hallucinate 成 0.815(它在 code 里直接 `print` 了这个数字,没等 observation)。Post-agent verification 抓住这点,用真实 0.09026 落地。

**Step 6**: mRMR 选择 `moment_difference` 唯一一个 feature,从 4-feature 模型简化到 1-feature 模型,ROC-AUC 完美 1.0。

Figure 2 右下角的 "early stop" 路径:agent 在第 7 步后发现无法再提升,outer round 进入 patience 阶段,6 轮后终止。

---

## 6. 实验结果深度解读

### 6.1 Regression(Table 1)

| Task | Baseline | OpenFE | AutoFeat | FAMOSE |
|------|----------|--------|----------|--------|
| bike | 40.3±1.03 | 92.09±6.35 | 41.47±1.01 | 40.05±0.99 |
| crab | 2.32±0.13 | 2.26±0.15 | — | 2.34±0.08 |
| housing | 409.58±10.54 | 432.87±10.22 | 403.96±9.61 | 408.56±26.34 |
| forest-fires | 92.7±5.34 | 88.52±5.91 | 93.49±5.42 | 79.49±5.87 |
| wine-quality | 0.64±0.02 | 0.75±0.03 | 0.62±0.02 | 0.64±0.01 |

关键观察:
- **forest-fires**:FAMOSE RMSE 79.49 vs baseline 92.7,**相对降低 14.3%**。这是一个大胜——这种 ecological 数据有非线性 interaction(moisture × temperature × wind),LLM 通过 reasoning 提出这些组合 feature。
- **bike**:OpenFE 完全爆掉(92.09),作者说不知道为什么——我猜测是 OpenFE 的 enumeration 在这个 task 上生成的某个 transform 引入 numerical instability,放大了 RMSE。
- **% Reduction**:平均 2.0%(Wilcoxon p=0.07,接近显著)。AutoFeat 0.3%,OpenFE -20.7%(实际上**变差**)。

### 6.2 Classification(Table 2)的 Small vs Large 拆分

这是 paper 最 sharp 的 finding:

**Small tasks(< 10K 实例)** 平均 ROC-AUC 提升:
- OpenFE: +1.04%(最佳)
- AutoFeat: +0.92%
- FAMOSE: +0.36%
- CAAFE: +0.47%
- FeatLLM: -7.6%(明显 worse,作者归因于 multi-class rules 生成问题)

**Large tasks(≥ 10K 实例)**:
- OpenFE: -1.77%(变差)
- AutoFeat: -0.018%
- FeatLLM: -15.9%(严重 fail)
- CAAFE: +0.008%
- **FAMOSE: +0.229%**(唯一稳定正向)

这个 split 是关键 insight:**classical methods 在 small data 上靠 enumeration 强行搜出有用 transform,但 large data 上 OOM/timeout 不能跑完**。FAMOSE 之所以 robust,是因为 agent 的 search 是 LLM-guided 的,memory 不随数据量增大而爆炸,只跟 feature 数和 round 数有关。OpenFE 在 covtype(580K 实例 × 55 features)上经常 fail,需要 TB 级 RAM。

**FeastLLM 的灾难性失败** -15.9% 值得单独提。Paper 4 节说原因是 "creating rules for multiple classes, or LLM output ends without completing the task"。这是 few-shot LLM method 在复杂 task 上 prompt-completion 不可控的通病。FAMOSE 通过 ReAct 的 closed-loop 避免了这个问题——agent 不能"说完了就完",必须达成 metric goal 才能退出 inner step。

### 6.3 Robustness:Cross-Model & Cross-LLM

Table S7/S8 把 FAMOSE 在 XGBoost 上发现的 features 应用到 Random Forest 和 Autogluon:
- Random Forest: +1.2% mean improvement(classification),+0.5% (regression)
- Autogluon: +0.02% (classification),-0.1% (regression)

这个 transfer 效应很重要:**FAMOSE 发现的 features 不只是 XGBoost-specific 的 trick,而是包含真实 signal 的 features**。Autogluon 已经是非常强的 ensemble,它自己会做 feature interaction,所以 FAMOSE 对它的增益小;但 Random Forest 缺乏 explicit interaction modeling,FAMOSE 的 engineered features 给它补了这一块。

Table S5/S6 测试 Deepseek-R1 替换 Sonnet 3.5 V2:
- Classification: Sonnet 0.32%, Deepseek 0.29%(几乎一样)
- Regression: Sonnet 2.0% reduction, Deepseek 2.8% reduction(Deepseek 略好)

这暗示 FAMOSE 的核心价值在 framework 而非特定 LLM。Reasoning 能力 sufficient 的 LLM 都能驱动这个 loop。

Deepseek-R1 论文:https://arxiv.org/abs/2501.12948
Claude 3.5 Sonnet 介绍:https://www.anthropic.com/news/claude-3-5-sonnet

### 6.4 Ablation Studies(Tables S9/S10)

三个 ablation:
1. **No goal in prompt**:去掉 "improve by 1%" 的 goal。Regression RMSE reduction 2.0%→2.2%(微升),classifier 几乎不变。作者解释 agent "偷懒"提前 terminate。我觉得这个 ablation 效果微弱,可能说明 goal 的作用被 post-agent verification 已经 cover 了。
2. **Only feature selection**(不做 generation,只跑 mRMR):regression 2.4%(竟然更高!),classifier -0.38%(显著 worse,Wilcoxon p<0.04)。这暗示:**对部分 regression task,原始 features 已经 informative,mRMR 选一下就好;但 classifier 需要 active feature generation 才能突破**。
3. **No feature selection**(只 generation 不 mRMR):regression -0.8%(变差),classifier +0.04%。这印证了 mRMR 的必要性——extra features 引入 overfitting,尤其在 regression 上。

整体 takeaway:**generation 和 selection 必须组合**,单做任何一边都次优。这与 Heaton (2016) 经验一致。

---

## 7. Prompt Engineering(Appendix A.1)的设计哲学

Full prompt 是一个 5-task checklist:

```
Task 1: Use insights to create a large set of new features.
        You can use any mathematical operations or transforms.
        Do not use any black box models (Random Forest, XGBoost, etc.),
        and do not use the target feature in newly generated features.
Task 2: Explain why this feature should help answer the question.
Task 3: Check performance via feature_performance_evaluator tool.
        Save dict {code: str} to pickle file with random number filename.
Task 4: If score > 0.01, move to Task 5. Otherwise create more features.
Task 5: Save best performing feature.
```

几个细节值得注意:
- **"Do not use any black box models"**:prevent agent 把整个 ML pipeline 包进 feature function,这会让 feature 不可 interpret 也 overfitting。Feature 必须是显式 transformation。
- **"Do not use the target feature"**:prevent trivial data leakage。这是 explicit guardrail,对应 algorithm line 7 里的 "Data provided to the feature generation code also has the target variable removed"。
- **Random number filename**(`new_feature_<dataset>_0_<random>.pkl`):避免 agent 重复写同一文件,方便 post-agent step 区分不同 candidate features。
- **`pd.to_pickle()` not `pickle.dump()`**:Smolagents 的 sandbox 可能限制原生 pickle 模块,pandas 的 wrapper 更 robust。

这种 prompt 风格让我联想到 Anthropic 的 "Constitutional AI" 中的 explicit rule format,以及 ConstitutionalHoudini-style 让 agent 自己 reason about instruction 而不是 blind follow。

---

## 8. Limitations(Paper 5.1)的补充思考

Paper 提到的 limitations:
1. **Token cost**:ReAct 的 chain-of-thought 累积 token 很快。FAMOSE 20 rounds × 10 steps × (thought+code+observation) 可能跑出几十 K token。对 Sonnet 3.5 V2 这种定价不算便宜,每个 task 可能要几美元。
2. **Small LLM 不行**:Llama 3.1-8B 跑不动。Reasoning + code generation + tool calling 三件套对小 LLM 太难。
3. **Background knowledge dependency**:冷门 domain(eg. spectroscopy)LLM 没见过相关概念,proposed feature 会很 generic。Paper 建议 RAG 补 bespoke knowledge。
4. **Multi-label classification 未支持**:作者说改动应该 minor,但需要工程化。

我额外想到几个点 paper 没提:
5. **Base model dependency**:FAMOSE 用 XGBoost 作 evaluation model。如果换 linear model 或 neural net,generated features 可能不一样。XGBoost 本身能 capture 一定 interaction,FAMOSE 的 marginal contribution 可能被低估。
6. **Search space偏置**:LLM 提议的 features 会集中在 LLM pretraining 见过的 common transform(ratio, log, product)。Piramuthu & Sikora (2009) 风格的多步 deep composition 可能不如 OpenFE 的 exhaustive enumeration 探索得深。
7. **Statistical significance 弱**:Table 2 总体 0.32% improvement,在大数据集上 0.229% 看起来小,但因为 10K+ 样本可能 statistically significant。Wilcoxon signed rank test 在 paper 里只在 regression 报告(p=0.07),classification 没报——这点不够严谨。
8. **Feature explanation 的 quality 未定量评估**:Paper 声称 "human interpretability",但没做 user study。只展示了 balance-scale 一个 case。Interpretability claim 缺乏定量 evidence。

---

## 9. 与 LLM-based ML landscape 的关联

FAMOSE 处在一个非常 active 的方向。相关 work:

- **FeatLLM (Han et al., 2024)**:把 feature generation 当 optimization problem,LLM propose + 外部 evaluator 选择。没有 closed-loop refinement。Paper:https://arxiv.org/abs/2404.09491
- **LLM-FE (Abhyankar et al., 2025)**:evolutionary optimizer 风格。FAMOSE paper 4 节吐槽它 metadata 错、RMSE normalization 不明、categorical preprocessing 不当。arXiv:https://arxiv.org/abs/2503.14434
- **ELF-Gym (Zhang et al., 2024)**:benchmark LLM-generated features 的 framework。https://doi.org/10.1145/3627673.3679153
- **CAAFE (Hollmann et al., 2023)**:one-shot LLM feature generation baseline。
- **LLMs on tabular survey (Fang et al., 2024)**:综述。
- **DIFER (Zhu et al., 2022)**:differentiable feature engineering,gradient-based search。FAMOSE 没引用但相关。https://proceedings.mlr.press/v188/zhu22b.html
- **TPOT (Olson et al., 2016)**:evolutionary AutoML,包含 feature engineering step。
- **MIT Context-Aware Feature Engineering**:相关 RAG-based direction。

更广 context 里,FAMOSE 是 "LLM as general pattern machine" (Mirchandani et al., 2023) 假设的又一证据——LLM 不只是 pattern completer,可以做 small-scale reasoning + tool use 形成 problem-solving loop。这与 AlphaCode、AlphaProof 的思路殊途同归:LLM 提供 proposal distribution,external verifier 提供 ground-truth signal,iterative refinement 找正确答案。

Mirchandani et al.:https://arxiv.org/abs/2307.04721
Jiang et al. 2025 "Artificial Hivemind":https://arxiv.org/abs/2510.22954(关于 LLM 输出 homogeneity 问题,FAMOSE 通过 ReAct loop 的 diversity 部分缓解)

---

## 10. balance-scale 例子的 Intuition 深挖

这个 example 是 paper 最好的教学 case。原始 4 个 features:`Left-Weight` $W_L$, `Left-Distance` $L_L$, `Right-Weight` $W_R$, `Right-Distance` $L_R$。Target 是 3-class:`{balanced, left, right}`。

物理上 scale 是否平衡由 net torque 决定:
$$\tau = W_L \cdot L_L - W_R \cdot L_R$$

- $\tau > 0$: left side heavier, scale tips **left**
- $\tau < 0$: scale tips **right**
- $\tau = 0$: **balanced**

XGBoost baseline 拿 4 features 直接学,ROC-AUC 0.914。它学到的是个 piecewise approximation of $W_L L_L - W_R L_R$,但因为是 tree-based,要多个 split 才能逼近乘积再相减的 nonlinearity。

FAMOSE 通过 LLM 的物理直觉直接提议 $f = W_L L_L - W_R L_R$——single feature 完美 capture target。后续 mRMR 发现这一个 feature 就足够,扔掉 4 个原始 features,模型简化成 single-feature linear classifier,ROC-AUC 1.0。

这个 example 展示了 FAMOSE 的核心价值 proposition:**LLM 的领域知识(domain prior)压缩成显式 feature,绕过 tree model 的 approximation difficulty**。这正是传统 AutoML 做不到的——OpenFE 能枚举 $W_L \cdot L_L$ 但不一定枚举 cross product + subtract 这种二步组合,Piramuthu & Sikora (2009) 的 iterative 框架能做到但需要 rule guidance。

类似 intuition 在森林火灾 task 也成立:RH(相对湿度)、temp、wind、rain 之间的非线性组合(eg. `wind × temp / (rain + 1)`)对 fire spread area 是关键。FAMOSE regression 把 forest-fires RMSE 从 92.7 降到 79.49 就是这类 transform 的功劳。

---

## 11. 一些可以拓展的方向联想

- **Multi-agent FAMOSE**:不同 agent 专攻不同 feature 类型(geometric transform agent,statistical moment agent,ratio agent),ensemble proposal。类似 MetaGPT / AutoGen 模式。
- **Active learning integration**:FAMOSE 的 round 数固定 20,可以加 RL controller 决定何时 stop。
- **Symbolic regression 对照**:PySR (Cranmer 2020) 用 genetic programming 找 symbolic expression。FAMOSE 的 LLM-guided search 可能比 GP 更 sample-efficient,但 GP 更 exhaustive。两者结合是 open direction。PySR:https://github.com/MilesCranmer/PySR
- **Verifiable code generation**:FAMOSE 的 hallucination 防御层数可以延伸到 formal verification——用 PyType 包验证 feature function 的 input/output contract。
- **Causal feature engineering**:LLM 推理因果 graph 生成 features,Judea Pearl-style causal features 比 statistical correlations 更 robust to distribution shift。causal discovery + LLM reasoning 的结合是 next frontier。
- **Mixture of Experts for feature proposal**:不同 LLM(Sonnet for code, GPT-4 for reasoning, Deepseek-R1 for math)分工 propose,combine 后 mRMR。

---

## 12. 总结

FAMOSE 的核心 contribution 不在单点 novelty——ReAct 已有,mRMR 已有,LLM feature generation 已有。它的 contribution 是把这些 component 缝合成一个 robust closed-loop framework,关键 design choices:

1. **Conditional evaluation**:每个新 feature 在已有 saved features 上 incremental evaluation,避免 redundancy。
2. **Multi-layer hallucination 防御**:regex column check + code execution + post-agent verification + mRMR 四层。
3. **Outer/inner 双层循环**:inner 是 agent ReAct,outer 是 reset-and-rediscover,平衡 context length 与 exploration。
4. **Algorithmic feature selection(mRMR)**:不让 LLM 自己选,避免 LLM selection bias。
5. **Goal-driven prompt** + early stopping:1% goal + 6-round patience 控制 token budget。

实验上 large-data classification 0.23% 提升 + regression 2.0% RMSE 降低 modest 但 consistent。Robustness 表现在 cross-model、cross-LLM 都 hold。balance-scale example 是 intuition-building 的精品 case,展示了 LLM domain prior + tree model 的乘积效应。

Andrej,你之前 lectures 里讲过 "neural net 需要 inductive bias"。FAMOSE 在 LLM agent 里把 inductive bias 显式化成 prompt + tool set + verification loop——inductive bias 不在 weight 而在 framework design。这种 "agent inductive bias" 路径可能是 LLM-based ML 的 future pattern。

更多 reference:
- ReAct 官方 repo:https://github.com/ysymyth/ReAct
- Smolagents doc:https://huggingface.co/docs/smolagents
- UCI balance-scale:https://doi.org/10.24432/C5488X
- Autogluon:https://auto.gluon.ai/
- XGBoost:https://xgboost.readthedocs.io/
- mRMR sklearn-style:https://github.com/smashujain/mRMR
- FAMOSE paper arxiv 链接(推测):https://arxiv.org/abs/2506.xxxxx(未在 paper 里看到 arxiv id,可在 Google Scholar 搜索 "FAMOSE Burghardt")
