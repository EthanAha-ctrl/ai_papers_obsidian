---
source_pdf: Vision Language Models are In-Context Value Learners.pdf
paper_sha256: 621819fea0df5069d035bacfb7b7cb717e40a3e1ea505c41a19ee7a3e8324704
processed_at: '2026-08-13T01:25:44-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 GVL

## 一句话说清楚

VLM其实"知道"怎么判断任务进度，但你直接问它，它会偷懒。GVL的trick就是：把视频帧顺序打乱，逼它真的去看内容。

---

## 为什么直接问不行

你给VLM一段视频，问"每帧完成了百分之多少"。VLM一看——哦，帧是按时间排的，那我直接输出0%, 3%, 6%, 9%... 递增就行了。它根本没看图像内容，就是看帧号猜。

这就像考试你问学生"第3题答案"，他看题号直接猜"C"。你给他一沓打乱顺序的卷子，他就只能真做题了。

VLM为什么有这个毛病？因为它训练时见过大量chronological video（captioning、QA），时间顺序本身就是一个strong cue。它学到了"时间往后走 = 内容往后发展"的shortcut。这个shortcut在captioning任务上work，但在value prediction上完全没用——因为不管trajectory好坏，它都输出递增。

---

## Shuffling 怎么破这个

把30帧随机打乱，告诉VLM"这些帧顺序乱了，你给每帧打完成度分数"。

现在VLM没法靠帧号偷懒了。它只能：
1. 看第一帧（保留作anchor）→ "哦，这是初始状态，0%"
2. 看其他帧 → 真的parse图像内容 → "这个手还没碰到杯子，大概20%"，"这个杯子已经在架子上了，85%"

打乱顺序把一个"pattern matching on temporal index"的问题，变成了"true visual-semantic reasoning"问题。看似更难，实际逼出了VLM真正的能力。

---

## 为什么第一帧不shuffle

这里有个细节：第一帧保留原位，作为anchor。

为什么不能全shuffle？因为很多task的正放和倒放在物理上都plausible。比如pick-and-place，正放是"拿起→放下"，倒放是"放下→拿起"——都合理。你全shuffle，VLM根本不知道哪边是起点哪边是终点。

第一帧告诉VLM"从这里开始"，breaks这个symmetry。简单但关键。

---

## Autoregressive 提供一致性

传统value function靠Bellman equation enforce consistency：$V(o_t) = R(o_t) + V(o_{t+1})$。训练时minimize这个等式的误差。所以你query single frame，它也能给你consistent的值。

VLM没这个训练。你单独问它第5帧，再单独问第6帧，两次答案可能完全没关系。

GVL的解法：让VLM一次性看完所有帧，autoregressive地输出所有values。模型生成第6个value时，context里已经有前5个value了。它会自然地避免自相矛盾——比如context里已经说了50%，它不太会再说30%。

这是"implicit Bellman"——不通过训练enforce，通过generation的autoregressive nature enforce。每个token都conditioned on previous tokens，产生globally coherent的output。跟chain-of-thought一个道理。

---

## VOC 这个metric很clever

你怎么evaluate一个universal value function？传统方法是看下游policy学得好不好。但universal value function要跨300+ task，你不可能每task都fine-tune一个policy来eval。太贵。

Prior work的做法是"eye-test"——肉眼看value curve平不平滑。但只能看几个video。

GVL提出VOC（Value-Order Correlation）：
- 拿一段expert demo（我们知道value应该随时间递增）
- shuffle后给GVL
- 看GVL预测的value排序，和真实时间排序的rank correlation

Expert demo的value按构造是monotonically increasing。好的value model应该能reconstruct这个order，VOC高。

Clever的地方在VOC同时是两个东西：
- Fix trajectory quality（expert）→ 高VOC说明model好
- Fix model（GVL）→ 低VOC说明trajectory差

第二个用处特别valuable。低质量trajectory有重复frame（robot卡住了反复尝试）、遮挡、camera角度差——这些都会让value不monotonic，VOC低。所以GVL的VOC score直接能当dataset quality indicator用。

---

## 三大downstream application

### 1. Dataset Quality Estimation

跑GVL算每个dataset的平均VOC，排名。

结果特别interpretable：
- RT-1: 0.74（human teleop, fixed camera, 高质量）
- Bridge: 0.51
- DROID: -0.01（发现很多camera angle差/遮挡的trajectory）
- RoboNet: -0.85（autonomous motor babbling, suboptimal）

DROID排名低这个发现特别有意思——prior work（OpenVLA）也发现从training data里移除DROID反而提升performance。GVL zero-shot就捕捉到了这个信号。

### 2. Success Detection

混合quality的数据集（比如autonomous collection, 50%成功率），你想filter掉失败的trajectory。

做法：跑GVL算VOC，低于threshold（如0.5）的标记为failure。

为什么这个work？失败的trajectory通常有repetitive frame（robot卡住重试）、没有clear progress。GVL没法shuffle-reorder这种video，VOC自然低。

对比SuccessVQA（直接问VLM"这个成功了吗"）：GVL-SD的precision高很多。SuccessVQA系统性偏向输出"failure"（可能因为VLM训练数据里negative example多）。

### 3. Advantage-Weighted Regression

最fine-grained的应用：不只filter trajectory，还给每个transition加权。

公式：$w_k = \exp(\tau \cdot (\nu_{k+1} - \nu_k))$

如果某个action让value涨了很多（$\nu_{k+1} - \nu_k$ 大正数），这个action被upweight。如果value没涨甚至降了，downweight。

这相当于offline RL——从mixed quality data里extract好的behavior。

实验结果有个特别clean的correlation：GVL+DP vs DP的improvement，和该task的VOC score强相关。VOC高的task（如banana-handover 0.73），AWR help大；VOC低的task（如open-drawer 0.09，top-down camera看不清progress），AWR反而hurt。

这validate了整个pipeline：value prediction quality直接决定下游performance。

---

## In-Context Learning 的惊喜

GVL zero-shot已经不错，但加in-context example能steadily improve。

几个surprising发现：
- **Cross-embodiment work**：用人类做同样task的视频作为example，能提升robot video的value prediction。说明VLM学到的是embodiment-agnostic的"progress concept"。
- **Cross-task work**：用不同task的example也有help（虽然不如same-task）。说明in-context example不只提供task-specific信息，还提供output format和generic progress notion。
- **Scaling**：5个in-context example（150张shuffled image）Gemini-1.5-Pro还能utilize full context，没saturate。

这整个不需要任何fine-tuning。纯prompt-time improvement。

---

## 核心intuition总结

1. **Foundation models已经"知道"很多东西**，关键是问对问题。Naive prompting让VLM走shortcut；shuffling堵死shortcut，逼出真正能力。

2. **看似更难的任务反而更好解**。直接predict value → VLM偷懒。Shuffle后predict value → VLM必须reason。这跟"让模型解释推理过程比直接要答案更准"一个道理。

3. **Autoregressive generation是implicit consistency constraint**。不需要Bellman training，generation本身enforce coherence。

4. **VOC是dual-purpose metric**。同时evaluate value model quality和trajectory quality，free of charge。

5. **Capability extraction vs capability training**。GVL没train任何东西，纯靠prompt engineering extract VLM的implicit capability。这跟RLHF/fine-tuning是不同哲学——后者是"教模型新东西"，前者是"让模型用已有的东西"。

整个工作最漂亮的地方是它的simplicity。Shuffling这个idea说出来感觉"就这？"——但它actually work，而且有clear的mechanistic explanation。这种"简单trick + 深刻insight"的组合，是好research的标志。

---

# Vision Language Models are In-Context Value Learners 深度解析

## 一、Core Problem & Motivation

这篇paper解决的核心问题是 **universal value function estimation** for robotics。在robotics中，value function（预测task temporal progress）是critical的，因为如果robot能判断"我现在做的动作是否在进步"，就能learn/adapt/improve。但universal value learning面临三个fundamental challenges：

1. **Broad generalization** to new tasks and scenes
2. **Accurately estimate state** in partially observed environments (POMDP)
3. **Temporal consistency** (satisfying Bellman equation) over long horizons

Prior work（如VIP、LIV）trained on relatively small vision-only data，缺乏semantic/spatial/temporal understanding，无法ground task progress到video的space-time manifold，generalization差。

**Key insight**: 现代VLMs（如Gemini-1.5-Pro）已经天然具备解决上述三个challenge的能力：
- VLMs有strong spatial reasoning + temporal understanding（解决challenge 1）
- Large transformer VLMs有requisite context window来reason over large historical information（解决challenge 2）
- VLMs auto-regressively commit to their own outputs as inputs for subsequent predictions，impose consistency constraints（解决challenge 3）。例如，如果context中已经有50% completion prediction，VLM不太可能再预测50% completion。

但naive prompting VLM to predict per-frame values fails——因为video的strong temporal correlation让VLM走shortcut，输出uninformative monotonic values。

---

## 二、Method: Generative Value Learning (GVL)

### 2.1 Problem Setup

Formalize为goal-conditioned POMDP：

$$\mathcal{M}(\phi) := (O, A, R, P, T, \mu, G)$$

变量解释：
- $O$: observation space
- $A$: action space
- $R$: reward function
- $P$: transition function
- $T$: task horizon
- $\mu(o)$: initial state distribution
- $G$: goal space，specifies the task semantically

Agent $\pi: O \to A$ 最大化value function:

$$V^{\pi}(o_1; g) = \mathbb{E}_{\mu, \pi, P}[r(o_1; g) + \cdots + r(o_T; g)]$$

但robotics中reward/value难以heterogeneously定义，所以采用**universal notion of value = task progress**，定义为temporal value function：

$$V: O \times \mathcal{G} \to [0, 1]$$

其中initial observations对应value 0，goal-satisfying observations对应value 1。Expert trajectory $\tau = (o_1, \ldots, o_T) \sim \pi_E$ 的value为：

$$V^{\pi_E}(o_t; g) = \frac{t}{T}$$

即线性progress。GVL的目标就是获得这样的temporal value function $V$，能预测每帧 $\nu_1, \ldots, \nu_T$。

### 2.2 GVL的三个key components

#### Component 1: Autoregressive Value Prediction

传统value function通过Bellman equation训练为self-consistent：

$$V^{\pi}(o_t) = R(o_t) + \mathbb{E}_{\pi, P}\left[V(o_{t+1})\right] \quad \text{(Eq. 1)}$$

变量解释：
- $V^{\pi}(o_t)$: policy $\pi$下$t$时刻的value
- $R(o_t)$: immediate reward
- $\mathbb{E}_{\pi, P}[\cdot]$: 在policy $\pi$和transition $P$下的expectation

Feed-forward NN通过minimize MSE来enforce这个等式。由于同一trajectory内的不同observations通过Bellman equation相关，即使single observation query也能保持consistent。

但VLMs没有inherently trained with consistency objective。如果independently query VLM with不同observations，会得到inconsistent values。

**Insight**: 提供整个trajectory作为input而非single observation，给VLM更多opportunity生成self-consistent estimates：

$$\nu_t = \mathrm{VLM}(o_1, \dots, o_T; \nu_1, \dots, \nu_{t-1}; l_{\mathrm{task}}), \forall t \in [2, T] \quad \text{(Eq. 2)}$$

变量解释：
- $\nu_t$: 第$t$帧的predicted task completion value
- $o_1, \dots, o_T$: 整个trajectory的$T$个video frames
- $\nu_1, \dots, \nu_{t-1}$: 已经autoregressively生成的之前所有values
- $l_{\mathrm{task}}$: 任务的language description
- $\mathrm{VLM}(\cdot)$: frozen VLM的next-token prediction

简写：$\nu_1, \ldots, \nu_T = \mathrm{VLM}(o_1, \ldots, o_T; l_{\mathrm{task}})$

这个机制让VLM在predict下一个value时attend to所有之前的predictions和frames，enable globally consistent estimates over long horizons，无需像classical feed-forward value function那样训练。

但naive prompting这样会导致degenerate monotonic values。

#### Component 2: Input Observation Shuffling

**Empirical observation**: 当VLM看到chronological frames，会走shortcut，输出monotonically increasing values，忽略task description和actual trajectory quality。

**Hypothesis**: VLMs在ordered video frames上训练captioning和QA，chronology本身成为cue。这导致naive prompting产生unfaithful low-quality values。

**Solution**: 随机shuffle input frames，强制VLM关注每individual frame，使用context中所有信息输出faithful values：

$$\nu_{\tilde{1}}, \dots, \nu_{\tilde{T}} = \mathrm{VLM}(o_{\tilde{1}}, \dots, o_{\tilde{T}}; l_{\mathrm{task}}, o_1), \quad \text{where} \quad (\tilde{1}, \dots, \tilde{T}) = \mathtt{permute}(1, \dots, T) \quad \text{(Eq. 3)}$$

变量解释：
- $\tilde{1}, \dots, \tilde{T}$: permuted indices，随机shuffle后的时间索引
- $\mathtt{permute}$: 随机置换operator
- $o_{\tilde{1}}, \dots, o_{\tilde{T}}$: shuffled frames
- $o_1$: 第一帧作为anchor point condition

**关键细节**: 不能shuffle所有frames！如果完全shuffle，video的arrow of time会ambiguous——很多情况下reverse video也physically plausible，ground-truth order无法预测。所以condition VLM on第一帧 $o_1$ 作为anchor point。

这个设计极其巧妙：把value estimation problem转化为**temporal ordering problem over shuffled frames**。看似更难，实际逼VLM更充分exploit其semantic和temporal grounding capabilities，根据perceived task progress区分frames，从而产生更好的value predictions。

#### Component 3: In-Context Value Learning

Large models有in-context learning能力。GVL通过prepend shuffled videos和ground-truth task progress作为in-context examples来boost value prediction quality：

$$\nu_{\tilde{1}}, \dots, \nu_{\tilde{T}} = \mathrm{VLM}\left(o_{\tilde{1}}, \dots, o_{\tilde{T}}, l_{\mathrm{task}} \mid \mathtt{permute}\left((o_1, \nu_1), (o_2, \nu_2), \dots, (o_M, \nu_M)\right)\right) \quad \text{(Eq. 4)}$$

变量解释：
- $|$ 左边: target prediction query
- $|$ 右边: in-context examples
- $(o_i, \nu_i)$: observation-value pair
- $M$: in-context example的frame数
- $\mathtt{permute}$: 同样shuffle in-context examples

In-context examples的来源非常flexible——可以来自same task、different task、甚至不同embodiment（human videos）。

### 2.3 Practical Implementation

- VLM输出0-100的integer-valued percentages
- 所有videos subsample到30 frames（保证cross-dataset comparable）

完整prompt（来自Appendix A）：
```
You are an expert roboticist tasked to predict task completion
percentages for frames of a robot for the task of {task_description}.
The task completion percentages are between 0 and 100, where 100
corresponds to full task completion. We provide several examples of
the robot performing the task at various stages and their
corresponding task completion percentages. Note that these frames are
in random order, so please pay attention to the individual frames
when reasoning about task completion percentage.
Initial robot scene: [IMG]
In the initial robot scene, the task completion percentage is 0.
Now, for the task of {task_description}, output the task completion
percentage for the following frames that are presented in random
order. For each frame, format your response as follow: Frame {i}:
Frame Description: {}, Task Completion Percentages:{}%
Frame 1: [IMG]
...
Frame n: [IMG]
```

---

## 三、Evaluation Metric: Value-Order Correlation (VOC)

传统value function evaluation（如downstream policy performance）对universal value function太expensive，需要per-task fine-tune。Prior work用qualitative "eye-test"看value curve smoothness，但只few videos。

GVL formalize并scale up这个intuition：

$$\mathrm{VOC} = \mathtt{rank\text{-}correlation}\left(\mathtt{argsort}(\nu_{\tilde{1}}, \dots, \nu_{\tilde{T}}); \mathtt{arange}(T)\right) \quad \text{(Eq. 5)}$$

变量解释：
- $\nu_{\tilde{1}}, \dots, \nu_{\tilde{T}}$: GVL预测的shuffled frames的values
- $\mathtt{argsort}(\cdot)$: 返回按value排序后的indices，即模型预测的时间顺序
- $\mathtt{arange}(T)$: $[0, 1, 2, \ldots, T-1]$，ground-truth的chronological order
- $\mathtt{rank\text{-}correlation}$: Spearman rank correlation
- VOC范围: $[-1, 1]$

**Intuition**: 
- VOC=1: predicted order与ground-truth order完美aligned
- Expert demonstrations按构造value monotonically increases with time，所以好的value model在expert videos上应有高VOC
- 低quality trajectories有redundant/repeated frames（re-attempt actions, poorly-placed cameras），values不monotonic，导致低VOC
- 所以固定好的value model，低VOC indicates低trajectory quality

这个metric有strong predictive power for downstream policy learning performance。

---

## 四、Experiments

### 4.1 Large-scale Real-World Evaluation

Backbone VLM: Gemini-1.5-Pro（也ablate了GPT-4o）。

#### Open X-Embodiment (OXE) Dataset

- 50 standalone academic robot datasets
- 每dataset随机sample 20 trajectories
- 总共1000 (50×20) trajectories
- 没有language annotation时用last frame作为goal specification
- Baseline: LIV (contrastive vision-language model fine-tuned with value learning on human videos)

**Results (Figure 3)**:
- Language goals: GVL的VOC histogram heavily right-skewed，LIV基本random（LIV embedding space缺乏sufficient knowledge预测arbitrary robot videos的dense values）
- Image goals: LIV相对好一些（image similarity本身能产生ascending values），但GVL略高

**Selected OXE Dataset Quality (Table 1)**:

| Dataset | Avg. VOC |
|---------|----------|
| RT-1 | 0.74 |
| Dobb-E | 0.53 |
| Bridge | 0.51 |
| QT-OPT | 0.19 |
| DROID | -0.01 |
| RoboNet | -0.85 |

Interpretation：
- Human teleoperation + fixed camera (RT-1, Dobb-E, Bridge): 高VOC
- Autonomous scripted/motor babbling (QT-OPT, RoboNet): 低VOC，suboptimal trajectories无smooth temporal structure
- DROID低VOC：many trajectories有poor camera angles或heavy occlusion。这与prior work发现remove DROID from training能improve performance一致

#### Challenging Bimanual ALOHA Dataset

- 250 distinct household tabletop tasks
- 长horizon、fine-grained（如remove 3 gears sequentially from NIST board、fold dress in eighth-fold、hang t-shirt on cloth rack）
- 每 task 2 human teleoperated demos
- 总共500 (250×2) trajectories

**Results (Figure 4)**:
- Zero-shot: median VOC 0.12, 60% positive
- One-shot (in-context example): median VOC 0.37, 90% positive

### 4.2 Multi-Modal In-Context Value Learning

#### Few-shot In-Context Learning

在ALOHA-13 subset（13 tasks with >500 demos each）上，up to 5 in-context examples：

平均VOC随in-context examples数量steadily improves。即使5 in-context trajectories（150 total shuffled images），Gemini-1.5-Pro也能utilize full context exhibit strong generalization。

#### Cross-Embodiment In-Context Learning

人类执行相同task的videos作为in-context examples，能有效improve over zero-shot。这说明GVL学到的是general notion of "task progress"，embodiment-agnostic。

#### Cross-Task In-Context Learning (Figure 14)

Random pair tasks作为in-context examples。仍beneficial但不如same-task examples——cross-task examples提供output format和generic task progress notion，但缺乏task-specific信息。

### 4.3 Downstream Applications

#### Application 1: Dataset Quality Estimation

GVL的VOC score作为dataset quality indicator。Table 1的ranking interpretable且match human intuition。

#### Application 2: Success Detection (GVL-SD)

将GVL用于success detection：filter trajectories with VOC < threshold。

**Comparison (Table 2)**:

| Method | Accuracy | Precision | Recall |
|--------|----------|-----------|--------|
| GVL-SD (Zero-Shot) | 0.71 | 0.71 | 0.71 |
| GVL-SD (One-Shot) | 0.75 | 0.85 | 0.70 |
| SuccessVQA | 0.62 | 0.33 | 0.73 |
| SuccessVQA-CoT | 0.63 | 0.44 | 0.68 |

- GVL-SD在所有metrics上outperforms或matches SuccessVQA
- SuccessVQA low precision: base VLM systematic bias towards outputting failure
- One-shot进一步improve所有metrics

**Key qualitative insight (Figure 7)**:
- Successful trajectories: GVL产生coherent VOC distribution
- Failed trajectories: GVL产生uniform VOC distribution（无法uncover temporal order）
- No-shuffling: success和failure的histograms基本相同——失去discriminability！

**Filtered Imitation Learning (Figure 8)**:
- ACT作为IL algorithm
- GVL-SD consistently outperforms SuccessVQA
- SuccessVQA经常hurt performance（low precision导致训练在false positive failure trajectories上）
- VOC threshold在{-1.0, 0, 0.25, 0.5, 0.75}都improve over ACT，robust to threshold choice
- Threshold=0.75略dip（dataset太小）

#### Application 3: Advantage-Weighted Regression (AWR)

在real-world ALOHA上，用GVL values做AWR，对每个transition加权：

$$\mathcal{L}(\theta) := -\mathbb{E}\left[\exp\left(\tau \cdot (\nu_{k+1} - \nu_k)\right) \cdot \log \pi_{\theta}(a_k \mid o_k)\right] \quad \text{(Eq. 6)}$$

变量解释：
- $\theta$: policy network parameters
- $\tau$: temperature hyperparameter，控制weighting的sharpness
- $\nu_{k+1} - \nu_k$: estimated advantage，即该action带来的value增量
- $\pi_{\theta}(a_k \mid o_k)$: policy likelihood
- $\exp(\cdot)$: exponential weighting，positive advantage对应大weight
- $\log \pi_{\theta}(\cdot)$: log-likelihood
- 负号: minimize negative weighted log-likelihood = maximize weighted log-likelihood

如果一个action被认为make progress towards goal，则future value $\nu_{k+1}$ 显著高于present value $\nu_k$，导致大positive weight。通过upweight最有promise的actions，AWR能在diverse human-collected datasets上outperform imitation learning。

**Results (Table 3)**: Diffusion Policy (DP) backbone，7 real-world tasks，10 trials per task:

| Task | GVL+DP | DP | Avg. VOC |
|------|--------|-----|----------|
| bowl-in-rack | 7/10 | 6/10 | 0.57 |
| banana-handover | 7/10 | 5/10 | 0.73 |
| close-laptop | 9/10 | 6.5/10 | 0.59 |
| open-drawer | 4/10 | 6/10 | 0.09 |
| remove-gears | 4.67/10 | 7/10 | 0.19 |
| pen-handover | 1.5/10 | 0/10 | 0.43 |
| fold-dress | 7/10 | 7/10 | 0.66 |

**Key insight**: GVL+DP vs DP的improvement与VOC score强相关。High VOC tasks（如banana-handover 0.73）AWR help大；Low VOC tasks（如open-drawer 0.09, remove-gears 0.19，top-down view分辨率不足以distinguish progress）value predictions noisy，hurt policy learning。这说明GVL的可信度直接影响下游performance。

### 4.4 Ablations

#### Is Autoregressive Value Prediction Necessary?

VLM (Single Frame): independent query each frame → VOC仅-0.08（vs GVL 0.74 on RT-1）。Pre-trained VLMs by themselves是poor value estimators，产生inconsistent noisy values。

#### Is Input Observation Shuffling Necessary?

Figure 11: no-shuffling的predictions collapse到几个linear ascending patterns，regardless of trajectory quality。失去discriminate success/failure的能力（Figure 7 right）。

#### Does GVL pay attention to task specification? (Figure 16, 17)

对ALOHA-13的每对(task video, language description)计算VOC。9/13 tasks上GVL在matched pair上VOC最高；unmatched cases中model经常refuses to output， stating frames和language不相关。No-shuffling ablation: quality急剧下降，高VOC与task description无关。

#### Different VLM Backbone (Figure 13)

GPT-4o作为backbone也work，VOC histogram heavily right-skewed。Difference主要在refusal rate和template conforming（-1.0 bar高度）。

#### Different Camera Viewpoints (Figure 15)

ALOHA 4个cameras：top-down, table, left wrist, right wrist。Zero-shot在table view最好（更in-distribution with natural images）。One-shot ICL在所有cameras上都improve。

---

## 五、Intuition Building: 为什么GVL Work

让我提炼几个核心insight：

### Insight 1: Shuffling as Anti-Shortcut Regularization

VLMs在chronological video上pretrained，所以chronology本身是一个strong cue。Naive prompting让VLM走"lazy path"——直接输出 $t/T$ 的linear mapping，忽略actual content。这就像学生不看题直接按行号给答案。

Shuffling把这个shortcut堵死。VLM被迫actually parse每帧内容，比较其与goal的semantic/spatial distance。这把value estimation从"pattern matching on temporal index"转化为"true visual-semantic reasoning"。

### Insight 2: Autoregressive as Implicit Bellman

Classical value function通过Bellman equation $V(o_t) = R(o_t) + \mathbb{E}[V(o_{t+1})]$ enforce consistency。VLM没有这个training signal。但autoregressive prediction提供了implicit consistency：模型看到自己之前的predictions，会避免产生self-contradictory sequence。这就像chain-of-thought——每个token都conditioned on previous tokens，产生globally coherent generation。

### Insight 3: First-Frame Anchor for Time Arrow

完全shuffle会ambiguous forward/backward direction。很多task的reverse也physically plausible（如pick-place的reverse是place-pick）。第一帧作为anchor告诉VLM"start from here"，breaks this symmetry。

### Insight 4: VOC as Universal Quality Signal

VOC的brilliance在于它freezes both axes：
- Fix trajectory quality → good model should give high VOC on experts
- Fix model → bad trajectories should give low VOC

这使得VOC同时是value model quality metric和trajectory quality metric，一举两得。

### Insight 5: Foundation Model as Implicit Value Function

GVL表明VLMs的world knowledge已经包含了"task progress"的implicit representation。GVL只是把这个implicit capability extract出来，通过prompt engineering (shuffling + autoregressive + ICL) 而非fine-tuning。这与LLMs as reasoners的哲学一致——foundation models已经"知道"很多东西，关键是问对问题。

---

## 六、Limitations & Future Work

- 没有investigate VLM fine-tuning是否能improve value prediction
- Multi-view observations可能能improve quality（未investigate）
- VOC metric最适合a-periodic tasks（unique ordering）；wiping/stirring等periodic tasks难以discern
- Camera viewpoint影响大（top-down for fine-grained tasks不够）

---

## 七、Related References

- Paper website: https://generative-value-learning.github.io
- Gemini 1.5 Pro: https://arxiv.org/abs/2403.05530
- LIV (baseline): https://arxiv.org/abs/2306.08647
- VIP: https://arxiv.org/abs/2210.00030
- Open-X-Embodiment: https://arxiv.org/abs/2310.08864
- ALOHA: https://arxiv.org/abs/2304.13705
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- AWR (Advantage-Weighted Regression): https://arxiv.org/abs/1910.00177
- SuccessVQA: https://arxiv.org/abs/2303.07280
- DROID dataset: https://arxiv.org/abs/2403.12945
- Bridge Data V2: https://arxiv.org/abs/2308.12952
- CLIP (used in LIV): https://arxiv.org/abs/2103.00020
- Octo: https://arxiv.org/abs/2405.12213
- OpenVLA: https://arxiv.org/abs/2406.09246

总结：GVL是一个非常elegant的work，把universal value estimation这个RL经典难题，通过re-framing为"shuffled temporal ordering" + autoregressive VLM prompting，zero-shot解决，并且scaling with in-context examples。VOC metric和downstream applications (data filtering, success detection, AWR) 让它从纯academic exercise变为practical toolkit。Core takeaway——foundation models的capability往往在"问对问题"时才能unlock，shuffling这个看似简单的trick实际上破了VLM的temporal shortcut bias。
