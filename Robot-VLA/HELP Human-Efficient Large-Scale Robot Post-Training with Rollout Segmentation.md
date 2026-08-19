---
source_pdf: HELP Human-Efficient Large-Scale Robot Post-Training with Rollout Segmentation.pdf
paper_sha256: d2e4ed7c936731438c327dbcbf02032faf68ad75c090f8ba035dd6bb0edf513b
processed_at: '2026-08-19T10:52:49-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HELP 人话版 —— 给Karpathy的intuition dump

好，我把上一篇technical拆解用大白话再过一遍，重点讲**为什么这么做**而不是**怎么做**。

---

## 一句话概括这篇paper在干嘛

你在VLA post-training的时候，**人比GPU贵**。这篇paper就是想办法让2个人管12个robot，还让这2个人产出的每一分钟human labor都换到最大的policy improvement。

paper里的核心数字：在**完全相同**的HITL recovery trajectory数量下，加上VLAC-CUT去做autonomous rollout segmentation，throughput gain额外放大**2.15×**，success rate gain额外放大**2.21×**。意思就是：人花的功夫一样，policy进步翻倍。

这个framing本身就跟 $\pi_{0.6}$（https://arxiv.org/abs/2511.14759）、ROVE（https://arxiv.org/abs/2606.17011）、SOP（https://arxiv.org/abs/2601.03044）不一样——那些工作追的是"model怎么从experience里学"，HELP追的是"人怎么被deploy"。

---

## 为什么human是瓶颈，说得具体一点

每一轮post-training你都要做这几件事：

1. 让policy跑autonomous rollout，看它怎么挂的。
2. 人在旁边盯着，看到快出问题就上手intervene，录一段recovery demonstration。
3. 录完之后robot回到初始状态，人手动把场景reset好。
4. 把所有data curate一下，混进训练集，flow-matching SFT一轮。
5. 评估，再进下一轮。

这里头**贵的是2和3**。第2步要的是经过几个月训练的teleop skill，那种能在test tube插入时自动减速、自动抬一下end-effector增加clearance的肌肉记忆。第3步要的是人走来走去搬物体、按按钮。一个人同时干这两件事，task switching的cognitive cost极其巨大——你刚进入"teleop心流"，又被叫去reset一个robot，回来时心流没了，teleop质量下降。

HELP的解法很直白：**把这两个职责分给两个人**。

- **Teleoperator**：只坐VR前面teleop，永远不走动、不reset。训练成本高（weeks到months），所以让他每一分钟都花在产出high-value recovery data上。
- **Floor Operator**：只reset + monitor + 按Bluetooth keyboard触发takeover。训练成本低（不用练manipulation skill），但因为他长时间盯着12个并发robot的autonomous rollout，他对policy的failure mode有一种statistical intuition，知道"这个policy在第3步拿beaker的时候有70%概率会撞到rim"，于是他会故意把场景reset成那个状态，让Teleoperator录recovery data。

这里有一个我挺喜欢的insight：**12 robot并行不只是throughput乘12**。一个operator看1个robot跑12次，跟看12个robot同时跑1次，信息量不一样。后者让你看到failure的**统计分布**——你会注意到"哦这12个里有5个在同一个位置挂了"，单robotsequential跑你很难有这种pattern recognition。这是一种scale带来的认知杠杆，跟SOP那种fleet-scale RL（https://arxiv.org/abs/2601.03044）思路接近，但HELP是用人来observe fleet，让model去collect。

---

## VLAC-CUT要解决什么problem，用人话说

policy被HITL recovery data训过几轮之后，会发展出**recovery能力**。这件事听起来是好事，实际上是个trap。

为什么？因为recovery data教policy"从error state怎么recover回来"，于是autonomous rollout里你会看到这样的pattern：

```
[正确progress] → [手抖了一下] → [policy自己试了3次都没对] → [终于recover了] → [继续做对] → 完成
```

如果你把这条trajectory直接丢进flow-matching SFT里训练，会发生什么？flow-matching的target就是这些action，policy会学到"试错3次然后recover"这个pattern本身。结果就是：任务成功率没掉（最后还是成了），但**throughput崩了**——每次任务都要先错再recover，平均执行时间从60s涨到90s。

你可能会问：那把这种trajectory整个扔掉不就行了吗？不行，因为trajectory里**前面的正确progress和后面的recover动作本身是有价值的训练data**，扔掉就浪费了。

所以你需要一个**segmentation tool**，把trajectory切成段，留好的，扔坏的。这就是VLAC-CUT。

VLAC-CUT把rollout切成4类segment：
- **Progress-making**：正常推进，留。
- **Recovery**：从error state回来的动作，留——这是有价值的data，教policy怎么recover。
- **Idle**：原地不动，扔。
- **Failure-inducing**：把任务搞砸的那些action，扔。

为什么failure-inducing不能当negative sample用？因为flow-matching SFT没有"负样本"这个概念，它的loss就是"让action distribution靠近target action"。要把failure当负信号得用RL，但RL在flow-matching decoder上还不太稳定（参考Flow Q-Learning, https://arxiv.org/abs/2502.02538，paper里也提到了这点）。所以HELP选择"切掉而不是利用"，这是个工程妥协。

---

## VLAC-CUT跟普通reward model有什么不一样

普通reward model $R(s)$ 是state的函数：同一个state，无论怎么到达，reward都一样。它是个snapshot scorer。

VLAC-CUT更像**policy-conditioned value estimator**：同一个state在不同trajectory context下得分不同，因为它要考虑"policy是怎么到的这个state，以及在这个state下policy后续期望会怎么样"。所以它打的分是trajectory-aware的，比snapshot reward更细。

这也意味着VLAC-CUT不是个universal reward function，它是跟deployed policy耦合的——policy换了，segmentation结果也会变。这点其实跟VLAC（https://arxiv.org/abs/2509.15937）的design philosophy一致。

---

## VLAC-CUT怎么训练的，直白讲

收集28,167条robot manipulation video，每条video由人annotator标记稀疏的progress point。注意是**signed** progress，范围 $[-100, 100]$：

- $p = 100$：任务完成
- $p = 0$：initial state
- $p < 0$：比initial还差，比如把object碰掉了

为什么signed这么重要？因为recovery过程往往要经过"比initial还差"的中间态——你抓object掉了，得先重新捡起来，捡起来之前的状态比initial state更糟。unsigned $[0, 1]$ 的reward scale根本表达不出这种regression。这跟PRM（Process Reward Model）在math reasoning里用signed step score的思路一致，dense supervision比binary success label信息量大得多。

数据来源里最特别的是**ARX-data**——他们自己用ARX-5 robot master-slave teleop收集的real-world dataset，**deliberately包含失败、掉落、wrong-object interaction、re-grasp、pose correction这些non-monotonic trajectory**。

为什么要搞ARX-data？因为public dataset（DROID https://arxiv.org/abs/2403.12945、LIBERO、VLABench）里几乎全是expert demonstration，policy没见过失败长什么样，训出来的critic默认"时间越往后任务越完整"，一遇到regression就傻掉。Reverse video augmentation（倒放）能fake一些decreasing progress sample，但visual dynamics不realistic——倒放里object会"反重力"飞回手里，跟real failure dynamics差太远。ARX-data补的就是这个gap。

每个progress point还配5个language component：
1. State description（"gripper正在下降"）
2. Action description（"gripper张开"）
3. Progress explanation（"gripper还没碰到object，progress停滞"）
4. Success/failure analysis（"gripper位置偏左，未对齐"）
5. Correction plan（"gripper应该抬起、右移、再下降"）

加上optional grounding（gripper + object的bbox和keypoint）。这些annotation让VLAC-CUT不止输出一个score，还能输出**diagnostic language**，告诉post-training pipeline"这一段为什么是failure-inducing"。

VLAC-CUT backbone是**Qwen3-VL-30B-A3B-Instruct**（MoE，激活3B），fine-tune在144×A800-80GB上跑16天，batch 2304，max seq len 15240 tokens。这个size不算小，因为它需要offline跑segmentation，对latency不敏感，可以大一点保证segmentation质量。

---

## ConSFT —— 一句话讲清intuition

ConSFT（https://arxiv.org/abs/2605.08879）的目标函数：

$$\mathcal{L}_{\mathrm{ConSFT}}(\theta) = \mathrm{sg}\left[\exp\left(-\frac{\mathcal{L}_{\mathrm{SFT}}(\theta)}{\tau}\right)\right] \cdot \mathcal{L}_{\mathrm{SFT}}(\theta) \tag{3}$$

人话解释：
- $\mathcal{L}_{\mathrm{SFT}}$ 是普通SFT loss。
- $\exp(-\mathcal{L}/\tau)$ 是个gating coefficient，$\tau$ 是温度。
- $\mathrm{sg}[\cdot]$ 表示这个gating不参与backprop，只作为系数。

intuition：一个新recovery action刚进来，policy还没学过，loss大，$\exp(-\mathcal{L}/\tau)$ 接近0，整个gradient被指数suppress——保护policy不被这个OOD样本搞坏。随着policy学进去，loss降下来，gating weight逐渐放大，正常学习。

这等价于一个self-regulating的adaptive learning rate，per-sample scaling由loss自己决定。好处是**不需要reference network、不需要experience replay buffer**——PPO那种RLHF需要reference model做KL penalty，experience replay需要维护buffer管理，工程上都很烦。ConSFT把这些都去掉，用loss自身做confidence proxy。代价是失去了replay对catastrophic forgetting的explicit防护，但paper里claim这个mechanism足够。

我个人觉得这跟DeepSeek-R1（https://arxiv.org/abs/2501.12948）里GRPO简化RL training的philosophy有点像——把依赖reference的复杂mechanism换成self-contained的简单formulation，靠scale和good data弥补理论上的不严谨。

---

## 实验设计最聪明的地方 —— matched HITL budget

这是这篇paper实验上最干净的设计。

他们做controlled comparison：HITL-only和HELP从**同一个checkpoint出发**，用**相同数量的HITL recovery trajectories**，HELP额外加上VLAC-CUT curated的autonomous rollout segments。然后比较policy improvement。

这把"human supervision"作为control variable固定住，把"autonomous rollout curation"作为treatment variable。如果HELP更好，那这个gain完全归功于VLAC-CUT的rollout segmentation能力，跟human effort无关。

这是个很strict的ablation，比"我们method在standard benchmark上更好"有说服力得多。

定义amplification factor：

$$A_M = \frac{M(\pi_{\mathrm{HELP}}) - M(\pi_{\mathrm{start}})}{M(\pi_{\mathrm{HITL}}) - M(\pi_{\mathrm{start}})} \tag{4}$$

$M$ 是throughput或success rate，$\pi_{\mathrm{start}}$ 是共同起点，$\pi_{\mathrm{HITL}}$ 和 $\pi_{\mathrm{HELP}}$ 是两种training后的policy。$A_M > 1$ 意味着VLAC-CUT从同一个human supervision budget里挤出更多policy gain。

7个matched task-round comparison的数字：

| Task-Round | $A_{TP}$ | $A_{SR}$ |
|---|---|---|
| Refrigerator-1 | 2.25× | 1.67× |
| Microplate-1 | 1.67× | 1.50× |
| Test Tube-1 | 1.20× | 2.00× |
| Stirrer-1 | 2.71× | 2.33× |
| Refrigerator-2 | 1.64× | 2.33× |
| Microplate-2 | **3.43×** | **3.00×** |
| Test Tube-2 | 2.13× | 2.67× |
| **Mean** | **2.15×** | **2.21×** |

注意Test Tube-1的 $A_{TP}$ 只有1.20×——这是整个table里最低的。为什么？因为这是**第一轮**，policy还弱，autonomous rollout质量差，curated segments里信息量低。到Test Tube-2，policy强了，autonomous rollout质量上来了，$A_{TP}$ 就跳到2.13×。

这就是compound return的intuition：stronger policy → higher quality rollout → better curated training data → even stronger policy。第一轮的ROI低是因为policy还没"起飞"，第二轮才显现效果。这个pattern在Microplate上更夸张——i=1的 $A_{TP}$ 1.67×，i=2的 $A_{TP}$ 3.43×，**翻倍**。

Microplate-2为什么特别dramatic？看Table 4，base 30%→HITL 60%→HELP 90%。HITL的human effort已经把success rate从30%拉到60%，HELP从60%拉到90%。第二个30pp jump全靠curated autonomous rollout，不需要额外human recovery data——这就是VLAC-CUT价值的直接量化。

---

## Execution time的"先升后降"现象 —— paper里一个挺诚实的observation

看Table 4，三个做了两轮的task（Refrigerator、Microplate、Test Tube）的execution time变化：

```
Refrigerator:  72.3 → 89 / 85  → 65.6 / 67.2    (第一轮升，第二轮降)
Microplate:    83.3 → 88 / 91  → 85.1 / 77.7    (第一轮升，第二轮降)
Test Tube:      93  → 83 / 105 → 90.7 / 93.4    (第一轮HITL降、HELP升，第二轮降)
```

第一轮execution time升，因为HITL recovery data教policy"recover from error state"，但recover需要时间。policy学会了recover，成功率上来了，代价是每次都要先错再recover，平均时间拉长。

第二轮execution time降，因为curated rollout data告诉policy"别再错了，直接走正确路径"。policy不再先错再recover，直接做对，时间就下来了。

throughput在第二轮的jump比第一轮大得多，原因就是**time和success rate同时改善**。Refrigerator: i=1 throughput +90%，i=2 throughput +120%。Microplate更夸张：i=1 +38%，i=2 +133%。

这个现象的intuition：第一轮training主要在补"会不会做"，第二轮training主要在补"做得快不快"。前者的边际收益是把失败转成功，后者的边际收益是把trial-and-error转成direct execution。这跟RL里"explore→exploit"的两阶段dynamic很像，只是这里是用data curation而非exploration bonus来驱动。

---

## VPB Benchmark的三个metric family，每个测什么

VPB（Video Progress Benchmark）held-out 3,515 records，分4个bucket：expert-seen, expert-unseen, non-expert-seen, non-expert-unseen。

### Global-level metrics

- **PRC**（Progress Rank Correlation）：Spearman rank correlation between预测的progress trajectory和真实trajectory。测"顺序对不对"。
- **VOC**（Value-Order Correlation）：只在expert data上算，测prediction是否monotonic with chronological frame order。这是GVL（https://arxiv.org/abs/2310.12931）原版评测，看model有没有chronological prior。
- **MAE**：绝对calibration error。测"数值准不准"。

VLAC-CUT在Overall MAE 7.56，PRC 0.926，两个都最好。最强的competitor Chrono-GVL-Gemini-3.1-Pro MAE 12.8，PRC 0.900——chronological ordering能恢复形状，但绝对值calibration差。Non-expert bucket上差距更大：VLAC-CUT MAE ~9.5, PRC ~0.83；Chrono-GVL MAE ~15-18, PRC ~0.72-0.76。chronological prior在non-monotonic trajectory上失效。

### Terminal-state metrics

90%作near-completion threshold。VLAC-CUT的TSA 84.30%不是最高（Chrono-GVL-Gemini-3.5-Flash 86.69%更高），但failed/incomplete class的F1高很多（61.02% vs 42.08%），MacroF1_T 75.59% vs 67.28%。

intuition：Chrono-GVL对"看起来像success"的terminal state过度敏感——chronological prior让最后frame默认高progress，所以visually plausible的success被正确识别。但对failed/incomplete terminal识别差——一个任务在最后一步失败，visual上跟前几步差不多，chronological prior会预测"高progress"，实际是failure。VLAC-CUT通过dense signed supervision避免了这种success bias。

### Local progress direction metrics —— segmentation能力的直接测试

对每对adjacent semantic anchor：
- $\Delta p^{\mathrm{key}}_{i,m} = p^{\mathrm{key}}_{i,m+1} - p^{\mathrm{key}}_{i,m}$：真实progress差。
- $\Delta \hat{p}^{\mathrm{key}}_{i,m}$：预测progress差。

两个ranking：
- $\mathrm{AP}_+$：预测 $\Delta \hat{p}$ 排序真实"进步"transition的能力。
- $\mathrm{AP}_-$：预测 $-\Delta \hat{p}$ 排序真实"后退"transition的能力。

VLAC-CUT $\mathrm{AP}_-$ 39.46%，最好的Chrono-GVL只有22.00%。这是**signed supervision的核心payoff**——能识别regression，这是segmentation的必备能力。Chrono-GVL的 $\mathrm{AP}_+$ 很高（94-95%），说明它assume monotonic progress，一旦真的regression了就识别不出来。VLAC-CUT在 $\mathrm{AP}_+$ 95.90%基础上还能保住 $\mathrm{AP}_-$ 39.46%，这才是真正能做segmentation的critic。

---

## VR Teleoperation Assistance —— 一个小但聪明的工程设计

VR teleop的痛点：packet loss、network jitter导致某些control slot没有valid command。简单fix（hold pose、repeat last command、temporal interpolation）会引入pause或stale motion。

HELP的trick：**复用正在deploy的 $\pi_\theta$ 做action inpainting**。

$$\hat{\mathbf{A}}_t \sim p_\theta(\mathbf{A} \mid \mathbf{o}_t, \mathbf{s}_t, \ell, \mathbf{A}_t^{\mathrm{obs}}, \mathbf{m}_t) \tag{21}$$

把VR command stream当成partially observed action chunk：observed slot硬projection保留，missing slot由deployed VLA inpaint成context-consistent action sequence。Signal-quality gate在command太stale时disable model-generated motion，让robot hold pose。

妙处：**teleop assistance不需要单独训一个model**，直接复用post-training正在迭代的那一个。policy变强，teleop assistance跟着变强，形成正反馈。跟ALOHA shared autonomy（https://arxiv.org/abs/1802.01537）思路接近，但工程更seamless——operator根本感知不到背后有model在辅助，只觉得VR变"丝滑"了。

---

## 几个我觉得比较subtle的设计选择

### 1. 为什么是2个operator，1:6 ratio

12 robot平均每个5-6分钟一次takeover，VR切换 + teleop操作 + verbal handoff大概30s-1min。Teleoperator单人limit大概就在12个robot附近。再加robot就得加Teleoperator，1:6是个自然的plateau。

paper没测2:24，但从throughput放大规律推，理论上可行，Floor Operator那边还有headroom（他主要是物理reset，时间不surge）。

### 2. 为什么是signed progress [-100, 100]

[0, 1] scale表达不出regression。recovery过程常常经过"比initial state还差"的中间态——object掉地上比initial state更糟。signed scale让VLAC-CUT能区分"还没开始"（0）和"开始但搞砸了"（-30），这是segmentation的semantic基础。

### 3. 为什么VLAC-CUT是30B-A3B MoE

30B dense太大，offline segmentation跑不动batch。30B-A3B MoE激活3B，inference快很多，但quality接近dense 30B。VLAC-CUT跑offline，latency不敏感，所以可以用30B的expressive power。

### 4. 为什么failure-inducing segment只切不用

flow-matching SFT没有negative gradient概念，只能让target distribution靠近某些action。要把failure当负信号得用RL，但flow-matching decoder上的RL还没ready（Flow Q-Learning https://arxiv.org/abs/2502.02538 是个promising方向但还不成熟）。所以HELP选择"切掉"，是个工程妥协，放弃了一部分training signal。future work如果能搞stable RL on flow-matching，failure-inducing segment就能做negative sample用了。

### 5. 为什么1:1:1 mixing是heuristic

paper没justify这个ratio，纯empirical。其实从information角度，HITL recovery是最稀缺最valuable的data，理论上应该over-weight它。但实际实验里1:1:1 work得最好，可能是curated rollout的量足够大且分布广，能稳定base capability，避免policy被HITL的narrow distribution拽偏。

### 6. 为什么predictive takeover model $\phi$ 是必要的

12 robot同时跑，Floor Operator的眼睛不够用。$\phi$ 提前预测"这个robot快出问题了"，让Floor Operator在"看着的robot"和"快出事的robot"之间prioritize。注意 $\phi$ 是**辅助**Floor Operator，最终takeover决策权在人——Floor Operator可以override $\phi$ 的prediction，也可以在没有 $\phi$ 信号时主动takeover。

paper没单独report $\phi$ 的accuracy，这是个system-level的missing experiment。如果 $\phi$ false positive高，Floor Operator被spam；false negative高，policy在error state浪费时间。这是HELP system的隐藏bottleneck。

---

## 我对这篇paper的overall take

这是篇**system engineering paper**，novelty不在单个component，而在于**把多个已知component assemble成一个coherent pipeline，然后在真实2-operator/12-robot setup上跑出可复现的数字**。

核心贡献：
1. **Human-economics framing**：把human labor explicitly放进post-training的cost function。
2. **Role specialization**：2个operator分工，让昂贵的teleop skill不被task switching稀释。
3. **VLAC-CUT**：process-level signed progress critic做rollout segmentation，是这套pipeline里最有technical depth的component。
4. **Controlled comparison**：matched HITL budget下的amplification factor，量化了autonomous rollout curation的价值。

最有technical depth的是VLAC-CUT，最有engineering含量的是distributed architecture + dual-process client + dynamic VR routing，最有experimental说服力的是matched HITL budget的controlled comparison。

如果你要从这篇paper里take away一个idea回去用，我会推荐**process-level signed progress supervision for rollout segmentation**这个concept。它在任何"从mixed-quality data学imitation"的setting都适用，远超robotics本身。比如LLM post-training里，一段reasoning trajectory可能有correct steps、wrong steps、recovery steps，用process-level signed critic做segmentation再SFT，理论上跟HELP的recipe同构。这就是为什么PRM在math reasoning里work（DeepSeek-R1, https://arxiv.org/abs/2501.12948），为什么HELP的VLAC-CUT在robot manipulation里work——core intuition是同一个：**dense signed supervision > sparse binary success label**。

---

主要reference links汇总：

- HELP / VLAC-CUT predecessor: https://arxiv.org/abs/2509.15937
- $\pi_{0.5}$: https://arxiv.org/abs/2504.16054
- $\pi_{0.6}$: https://arxiv.org/abs/2511.14759
- ROVE: https://arxiv.org/abs/2606.17011
- SOP: https://arxiv.org/abs/2601.03044
- Learning While Deploying: https://arxiv.org/abs/2605.00416
- ConSFT: https://arxiv.org/abs/2605.08879
- DROID: https://arxiv.org/abs/2403.12945
- LIBERO: https://arxiv.org/abs/2306.03310
- OpenVLA: https://arxiv.org/abs/2406.09246
- Robo-Dopamine: https://arxiv.org/abs/2512.23703
- Robometer: https://arxiv.org/abs/2603.02115
- RoboReward: https://arxiv.org/abs/2601.00675
- ProgressLM: https://arxiv.org/abs/2601.15224
- GVL: https://arxiv.org/abs/2310.12931
- TOPReward: https://arxiv.org/abs/2602.19313
- Flow Q-Learning: https://arxiv.org/abs/2502.02538
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- RewardBench: https://arxiv.org/abs/2403.13787
- VLRewardBench: https://arxiv.org/abs/2411.17451
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- ALOHA shared autonomy: https://arxiv.org/abs/1802.01537
- World-Env: https://arxiv.org/abs/2509.24948
- AtomVLA: https://arxiv.org/abs/2603.08519
- ConRFT: https://arxiv.org/abs/2502.05450
- Qwen-VL: https://arxiv.org/abs/2308.12972

---

# HELP: Human-Efficient Large-Scale Robot Post-Training

你好 Andrej，我把这篇paper按你的intuition-building需求拆开讲，尽量做到细节足够、能让你在脑子里建模出这套系统的data flow和trade-off。我会从顶层motivation一路打到公式变量、数据split、训练mixture和实验数字。

---

## 1. 为什么这篇paper存在 —— 重新定义efficiency

绝大部分VLA post-training工作都在追**sample efficiency**（fixed dataset上的loss下降速度），或者追**rollout volume**（多少个autonomous episode）。HELP这篇工作把矛头指向一个被忽视的瓶颈：**human labor**。

核心命题：在real-world VLA post-training里，每一轮迭代都要human去 (i) 看failure，(ii) 上手intervene，(iii) reset物理场景，(iv) curate data。这些成本**累积 across iterations**，最终成为scale的真正瓶颈。所以metric应该是：

$$\text{HumanEfficiency} = \frac{\Delta \text{SuccessRate / Throughput}}{\text{Human Labor} \times \text{Wall-clock Time}}$$

这个framing本身就跟pi0.6、ROVE、SOP那种偏纯RL或偏fleet-scale RL的工作拉开距离。pi0.6（Physical Intelligence, 2025, https://arxiv.org/abs/2511.14759）关心的是VLA怎么从experience里学；ROVE（https://arxiv.org/abs/2606.17011）关心humanoid的imperfect intervention；SOP（https://arxiv.org/abs/2601.03044）关心online fleet的scalable learner。HELP明确把**人**放在cost function里，这是一种operator-economics视角的post-training。

---

## 2. HELP系统架构 —— 三层distributed software

Architecture（Figure 1）分三层，每层都对应一个human-centric requirement。

### 2.1 Centralized GPU Workers

放在一个centralized server上，用Ray做distributed framework。每个control step，asynchronous coordination module把所有robot的多模态observation聚合，dynamic load balancing分发到parallel inference workers。同时跑三个模型：

$$\mathbf{a}_t^i = \pi_\theta(s_t^i), \quad \gamma_t^i = \phi(s_t^i), \quad d_t^i = \psi(s_t^i) \tag{1}$$

变量解释：
- $\mathbf{a}_t^i$ —— 第 $i$ 个robot在step $t$ 的**action chunk**（注意是chunk，不是单步action；这跟 $\pi_{0.5}$ 用flow-matching出chunk的范式一致，参考 https://arxiv.org/abs/2504.16054）。
- $\gamma_t^i \in \{0,1\}$ —— **predictive takeover model** $\phi$ 输出的flag，预测当前是否需要human接管。
- $d_t^i \in \{0,1\}$ —— **task termination model** $\psi$ 输出的flag，预测任务是否完成（用于触发reset）。
- $s_t^i$ —— 第 $i$ 个robot在step $t$ 的state（包含proprioceptive + camera）。
- $\theta$ —— VLA policy参数，会被异步gradient descent更新。
- $\phi$ —— takeover predictor，用2B Qwen-VL fine-tune，positive sample来自Floor Operator按takeover键那一帧，negative sample来自正常autonomous rollout帧。
- $\psi$ —— termination model。

这个并发inference的设计非常关键：$\phi$和$\psi$不是离线跑的reward model，而是**online和policy同step inference**，直接驱动client的state machine。这就要求三个模型共享GPU workers，dynamic load balancing的意义就在这里。

### 2.2 Distributed Robot Clients

跑在每个robot的onboard computer上，dual process + async multi-threaded。Hardware control loop跟network I/O和VR telemetry stream**显式隔离**，用ZeroMQ做async command fetch，IPC做inter-process payload。

Client是一个finite state machine，优先级hierarchy：

$$S_{\mathrm{client}}^i \gets \begin{cases} S_{\mathrm{takeover}} & \text{if } u_t^i = \mathrm{takeover}, \\ S_{\mathrm{hitl}} & \text{if } S_{\mathrm{client}}^i = S_{\mathrm{takeover}} \text{ and VR streaming is active}, \\ S_{\mathrm{reset}} & \text{if } u_t^i = \mathrm{reset}, \\ S_{\mathrm{model}} & \text{if } u_t^i = \mathrm{step}(\mathbf{a}_t^i). \end{cases} \tag{2}$$

四种state：
- $S_{\mathrm{model}}$ —— VLA policy自己执行action chunk。
- $S_{\mathrm{takeover}}$ —— standby state，等待VR stream同步。
- $S_{\mathrm{hitl}}$ —— active human control，VR telemetry driving robot。
- $S_{\mathrm{reset}}$ —— robot locked，等Floor Operator物理reset。

优先级takeover > hitl > reset > model。这里有个intuition点：从 $S_{\mathrm{model}}$ 到 $S_{\mathrm{hitl}}$ 不是直接跳的，要先过 $S_{\mathrm{takeover}}$ 这个"purge stale actions + wait VR sync"的中间态。这是为了让handover不卡顿，避免VR还没sync就动作导致robot瞬移。

### 2.3 Operator Console

Web frontend + VR application + Bluetooth keyboard + backend service。
- Teleoperator用Web frontend看camera feed，dictate VR action的routing destination。
- VR用UDP dynamic routing gateway（高频telemetry）。
- Floor Operator用Bluetooth keyboard触发 $S_{\mathrm{takeover}}$（ZeroMQ socket触发对应robot）。
- 还有个VLA-based teleoperation assistance module（Appendix B后面讲）。

---

## 3. Role Specialization —— Teleoperator vs Floor Operator

这是HELP的核心human-economics设计。2个operator管12个robot。

### Teleoperator
- **训练成本高**：要花weeks到months练出specific manipulation patterns（比如test tube插入时降速、抬升end-effector增加clearance）。
- **任务单一**：只做remote manipulation，不reset、不走动、不monitor fleet。
- **stationary**：坐在VR前，等takeover signal被route过来。
- 价值：产出**high-value recovery demonstration**，这是post-training最稀缺的data type。

### Floor Operator
- **任务单一**：物理reset + 持续monitor fleet + 用Bluetooth keyboard强制takeover。
- **训练成本低**：不需要练manipulation skill，只需要理解policy的failure mode。
- **认知杠杆**：因为长时间观察多个robot的autonomous rollout，Floor Operator对policy的weakness有intuitive理解，能精准判断"什么时候takeover + 把场景reset到哪个state"。
- 价值：在policy频繁失败的state下收集targeted data，最大化recovery data的"信息密度"。

这里有个subtle intuition：**12 robot并行不只是线性放大throughput**，更关键的是它**broadens policy-behavior coverage**。一个operator盯着12个并发执行，比盯着1个执行12次更容易看到recurring failure mode。这种statistical view of failure是单robot setup无法提供的。这跟Learning While Deploying（https://arxiv.org/abs/2605.00416）的fleet-scale思路有相通之处，但HELP把"人看fleet"这件事显式建模成failure mode discovery。

---

## 4. Post-Training in HELP —— 两种optimization策略

数据collection会产出两种trajectory：
1. **纯VLA rollout**（homogeneous trajectory）。
2. **VLA prefix + HITL recovery suffix**（hybrid trajectory）。

HITL event除了做recovery demonstration，还**同时**做takeover prediction model的训练样本（positive: Floor Operator按takeover那一帧；negative: 正常autonomous rollout里的帧）。Takeover predictor是2B Qwen-VL fine-tune。

### 4.1 Fully Online Incremental Optimization —— ConSFT

用ConSFT（Conservative SFT，Zhang et al. 2026b, https://arxiv.org/abs/2605.08879），目标：

$$\mathcal{L}_{\mathrm{ConSFT}}(\theta) = \mathrm{sg}\left[\exp\left(-\frac{\mathcal{L}_{\mathrm{SFT}}(\theta)}{\tau}\right)\right] \cdot \mathcal{L}_{\mathrm{SFT}}(\theta) \tag{3}$$

变量解释：
- $\mathcal{L}_{\mathrm{SFT}}(\theta)$ —— 标准SFT loss（这里用的是flow-matching loss，跟 $\pi_{0.5}$ 一致）。
- $\tau > 0$ —— temperature，控制conservative scaling的sensitivity。
- $\mathrm{sg}[\cdot]$ —— stop gradient operator，意思是方括号里的部分只作为scaling coefficient，不参与backprop。

Intuition：当一个样本是out-of-distribution（比如某个新的recovery action），$\mathcal{L}_{\mathrm{SFT}}$ 大，$\exp(-\mathcal{L}/\tau)$ 就小，整个gradient contribution被**指数suppress**。随着policy慢慢学进去，loss下降，scaling weight放大，正常学习。

这个机制的好处：**不需要reference network、不需要replay buffer**，纯粹靠"loss自身作为confidence proxy"来自适应scaling。这跟RLHF里PPO用KL penalty + reference model的思路完全不同，ConSFT把KL penalty换成了一个self-regulating的exponential gating。我个人觉得这跟DeepSeek-R1（https://arxiv.org/abs/2501.12948）里GRPO的某些simplification思想有点类似——把reference dependence去掉。

### 4.2 Periodic Batched Optimization

跟online对比的另一种config：discrete training interval，每个batch到了就joint train，把所有historical data from all preceding rounds混进去。这就避免了streaming continual learning里的catastrophic forgetting，但代价是没有online responsiveness。

实际HELP real-world实验用的是哪种？看Section 5.2.1的task描述，他们其实是iterative round-based，每round用best checkpoint去collect，再混合base data + HITL + curated rollout大约1:1:1训练。这更像periodic batched，但ConSFT在 continual setting下也能用。

---

## 5. VLAC-CUT —— Rollout Segmentation Critic

这是这篇paper最technical novel的部分。先说为什么需要它。

### 5.1 问题motivation

VLA policy经过几轮HITL recovery训练后，会发展出**recovery能力**。这意味着autonomous rollout里经常出现这样的pattern：

```
[正确progress] → [错误动作] → [试错] → [recover] → [继续progress] → ...
```

如果直接把整条trajectory丢进去训练（flow-matching SFT），policy会学到"试错"本身——因为flow-matching的target就是这些action，它会reinforce trial-and-error pattern，最终任务成功率虽然高，但throughput下降（每次都要先错再recover）。

如果整条trajectory丢弃，又浪费了progress segment和recovery segment。

VLAC-CUT做的事：把autonomous rollout切成4类segment：
- **Progress-making** —— 保留，reinforce正确行为。
- **Recovery** —— 保留，这是valuable data。
- **Idle** —— 丢弃。
- **Failure-inducing** —— 丢弃。

注意：在flow-matching objective下，没有reliable way把failure-inducing action作为negative signal用。所以只能"切掉"，不能"当负样本用"。这跟RL里的negative gradient不一样，是SFT范式的局限。

### 5.2 跟reward function的区别

普通的reward function $\mathcal{R}(s)$ 是policy-agnostic的：同一个state无论怎么到达，reward都一样。

VLAC-CUT是**policy-conditioned value estimator**性质的：同一个state在不同trajectory context下会得到不同score，因为它考虑了"policy怎么到的这个state + 后续action expected产生什么consequence"。这一点很重要——它是process-level critic，而不是snapshot reward。

这个设计跟VLAC（Zhai et al. 2025, https://arxiv.org/abs/2509.15937）一脉相承，VLAC-CUT是VLAC在HELP这个specific setting下的specialization。

### 5.3 训练数据 —— Progress Annotation Dataset

总量：**28,167 records, 22,978 episodes, 15,206 task units, 375,172 progress points**。3,515 records held-out for VPB。

来源：
- **DROID**（Khazatsky et al. 2024, https://arxiv.org/abs/2403.12945）—— 大规模real-world manipulation diversity。
- **LIBERO**（Liu et al. 2023a）—— simulated task variation。
- **VLABench**（Zhang et al. 2025b）—— simulated benchmark-style。
- **ARX-data** —— 自己收集的，in-house real-world ARX-5 robot master-slave teleop，重点是**physically executed non-monotonic trajectories**：grasp failure、object drop、wrong-object interaction、re-grasp、pose correction。这是关键——public dataset大多是expert demonstration，无法训练critic去识别regression和recovery。

**Signed progress label**：$p \in [-100, 100]$，$p=100$ 是full completion，$p=0$ 是initial state，$p<0$ 是regressive state（比initial还差）。这个signed设计很关键，它直接supervise "progress vs regression vs recovery"的区分。

### 5.4 Annotation Schema

每个progress point包含5个component：
1. State description
2. Action description  
3. Progress explanation
4. Success/failure analysis
5. Correction plan

加上optional grounding（gripper end-effector + task-relevant object bbox + keypoint）。

Annotation interface支持kinematic mode：加载robot position + gripper state，rescale，unwrap rotation，filter，compute velocity/acceleration，extract candidate keyframes under weighted position/rotation/gripper cost，然后map到video frame。annotator可以add/remove/modify。

### 5.5 Instruction Tuning的4个ability

1. **Task decomposition and grounding** —— language goal → semantic milestone + object + gripper position + keypoint。
2. **Temporal progress prediction** —— 从task-conditioned visual evidence估timestamped progress（不是从elapsed time）。
3. **Diagnostic reasoning** —— 把scalar progress supervision转成"为什么是progress/idle/failure-inducing/recoverable"的explanation。
4. **In-context and action-conditioned supervision** —— 条件在reference episode上，预测relative action delta（translation in mm, rotation in degree, gripper in %）。

两个targeted augmentation：
- **Counterfactual reverse-progress augmentation** —— 从annotated progress point pair构造"先倒退再恢复"的clip，discourage "later frame = more complete"这个shortcut。
- **Grounding-based rationale generation** —— 当object box + gripper keypoint available且geometric evidence跟progress方向一致时，verbalize成rationale。

### 5.6 VLAC-CUT backbone

Qwen3-VL-30B-A3B-Instruct（MoE）。有效global batch 2304，max seq len 15240 tokens，peak lr $3\times10^{-5}$ cosine decay。MS-SWIFT + DeepSpeed ZeRO-3，144×A800-80GB，16天。

训练mixture（Table 8）：
- Annotation-derived robot data: 4.76M samples
- LLaVA-style instruction: 200K
- RoboReward: 45K
- RoboVQA: 50K
- Spatial QA: 250K
- ProgressLM CoT: 24K
- MMSI-Video-Bench: 0.7K

---

## 6. VPB Benchmark —— 三个metric family

VPB = Video Progress Benchmark，held-out 3,515 records。两个axis切分：
- Semantic task familiarity: seen / unseen
- Progress pattern type: expert / non-expert

四bucket: expert-seen, expert-unseen, non-expert-seen, non-expert-unseen。

### 6.1 Global-level metrics

- **PRC**（Progress Rank Correlation）—— Spearman rank correlation between $\hat{\mathbf{p}}_i$ 和 $\mathbf{p}_i$，measures temporal ordering preservation。

$$\mathrm{PRC}_i = \rho_S(\hat{\mathbf{p}}_i, \mathbf{p}_i), \quad \mathrm{PRC}(\mathcal{S}) = \frac{1}{|\mathcal{S}|}\sum_{i\in\mathcal{S}} \mathrm{PRC}_i \tag{6}$$

- **VOC**（Value-Order Correlation）—— 来自GVL（Ma et al. 2024, https://arxiv.org/abs/2310.12931）的rank correlation评估，只在expert data上算，看prediction是否monotonic with chronological frame order。

$$\mathrm{VOC}_i = \rho_S(\hat{\mathbf{p}}_i, (1, 2, \ldots, T_i)) \tag{7}$$

- **MAE** —— 绝对calibration error。

$$\mathrm{MAE}_i = \frac{1}{T_i}\sum_{t=1}^{T_i} |\hat{p}_{i,t} - p_{i,t}| \tag{8}$$

dense ground-truth trajectory $\mathbf{p}_i$ 通过sparse keyframe linear interpolation得到：

$$p_{i,t} = p_{i,m}^{\mathrm{key}} + \frac{t - t_{i,m}}{t_{i,m+1} - t_{i,m}}(p_{i,m+1}^{\mathrm{key}} - p_{i,m}^{\mathrm{key}}) \tag{5}$$

### 6.2 Terminal-state metrics

90%作为near-completion threshold。

$$y_i^T = \mathbb{I}[p_{i,T_i} \geq 90], \quad \hat{y}_i^T = \mathbb{I}[\hat{p}_{i,T_i} \geq 90] \tag{9}$$

- TSA（Terminal-State Accuracy）= $\frac{TP + TN}{TP + FP + TN + FN}$
- $\mathrm{F1}_S = \frac{2TP}{2TP + FP + FN}$ （成功class的F1）
- $\mathrm{F1}_F = \frac{2TN}{2TN + FP + FN}$ （failed/incomplete class的F1）
- $\mathrm{MacroF1}_T = \frac{1}{2}(\mathrm{F1}_S + \mathrm{F1}_F)$

### 6.3 Local progress direction metrics —— 这是segmentation能力的核心

定义在adjacent semantic-anchor transition上。

$$\Delta p_{i,m}^{\mathrm{key}} = p_{i,m+1}^{\mathrm{key}} - p_{i,m}^{\mathrm{key}}, \quad \Delta\hat{p}_{i,m}^{\mathrm{key}} = \hat{p}_{i,m+1}^{\mathrm{key}} - \hat{p}_{i,m}^{\mathrm{key}} \tag{13}$$

两个one-vs-rest ranking：
- Improvement target: $y_{i,m}^+ = \mathbb{I}[\Delta p_{i,m}^{\mathrm{key}} > 0]$, score $s_{i,m}^+ = \Delta\hat{p}_{i,m}^{\mathrm{key}}$
- Regression target: $y_{i,m}^- = \mathbb{I}[\Delta p_{i,m}^{\mathrm{key}} < 0]$, score $s_{i,m}^- = -\Delta\hat{p}_{i,m}^{\mathrm{key}}$

$\mathrm{AP}_+$ 和 $\mathrm{AP}_-$ 各自的average precision，macro是平均。

$\mathrm{AP}$ 的标准定义：

$$\mathrm{AP} = \frac{1}{N_+}\sum_{r=1}^{M} y_{\pi_r}\frac{\sum_{q=1}^{r} y_{\pi_q}}{r} \tag{20}$$

其中 $\pi$ 是按score降序的permutation，$N_+ = \sum_j y_j$。

为什么这个metric重要：它直接测**segmentation的能力**——model能不能区分"这一段在进步"和"这一段在后退"。Chronological prompting方法（Chrono-GVL）在 $\mathrm{AP}_+$ 上很好，但 $\mathrm{AP}_-$ 很差——因为它们assume monotonic progress，对regression不sensitive。VLAC-CUT因为有signed progress supervision，$\mathrm{AP}_-$ 大幅领先。

---

## 7. VPB实验结果 —— Table 1, 2, 3

### Table 1: Global-level

| Method | Overall MAE↓ | Overall PRC↑ |
|---|---|---|
| **VLAC-CUT** | **7.5600** | **0.9260** |
| Chrono-GVL-GPT-5.5 | 12.3511 | 0.8768 |
| Chrono-GVL-Gemini-3.1-Pro | 12.7982 | 0.8997 |
| Chrono-GVL-Gemini-3.5-Flash | 12.8388 | 0.8961 |
| GVL-Gemini-3.5-Flash | 22.8278 | 0.5370 |
| RoboReward | 19.7949 | 0.7521 |
| Robometer | 21.1340 | 0.6605 |

VLAC-CUT在overall MAE和PRC都最好，每个bucket的MAE都最低。最强的非VLAC-CUT方法（Chrono-GVL-Gemini-3.1-Pro）PRC 0.8997 vs VLAC-CUT 0.9260，差距小但MAE差距大（12.8 vs 7.6）—— 说明chronological ordering能恢复形状但**绝对值calibration不好**。

Expert-unseen bucket上Chrono-GVL-Gemini-3.1-Pro的PRC 0.9869 vs VLAC-CUT 0.9859，反过来一点点。但MAE还是VLAC-CUT更低。这说明expert trajectory上chronological prior很强，但calibration仍然需要signed progress supervision。

Non-expert bucket是真正的discriminator：VLAC-CUT MAE ~9.5，PRC ~0.83；最好的Chrono-GVL MAE ~15-18，PRC ~0.72-0.76。这里有regression和recovery行为，chronological prior失效。

### Table 2: Terminal-state

| Method | TSA↑ | F1_S↑ | F1_F↑ | MacroF1_T↑ |
|---|---|---|---|---|
| **VLAC-CUT** | 84.30 | 90.17 | **61.02** | **75.59** |
| Chrono-GVL-Gemini-3.5-Flash | **86.69** | **92.48** | 42.08 | 67.28 |
| Chrono-GVL-Gemini-3.1-Pro | 86.15 | 91.99 | 48.90 | 70.44 |

VLAC-CUT的TSA不是最高，但failed/incomplete class的F1高很多（61.02 vs 42-48）。MacroF1_T 75.59 vs 67-70。Intuition：Chrono-GVL对"看起来像success"的terminal state很敏感（因为chronological prior让最后frame默认高progress），但对failed/incomplete terminal的识别差——容易被visually plausible successful ending骗。VLAC-CUT通过dense signed progress supervision更平衡。

### Table 3: Local progress direction

| Method | Overall AP_+↑ | Overall AP_-↑ | MacroAP_D↑ |
|---|---|---|---|
| **VLAC-CUT** | 95.90 | **39.46** | **67.68** |
| Chrono-GVL-Gemini-3.5-Flash | 94.40 | 22.00 | 58.20 |
| Chrono-GVL-Gemini-3.1-Pro | 93.65 | 17.92 | 55.78 |
| Chrono-GVL-GPT-5.5 | 94.83 | 16.00 | 55.42 |
| ProgressLM-RL | 91.41 | 9.35 | 50.38 |

最大差距在 $\mathrm{AP}_-$：VLAC-CUT 39.46 vs 最好的Chrono-GVL 22.00。这正是process-level signed supervision带来的优势——能识别regression。

---

## 8. Real-World Post-Training实验 —— 4个task

### 8.1 Task描述

- **Refrigerator**：开冰箱门 → 抓beaker → 放进去 → 关门。100s limit。drops算失败。
- **Microplate**：开reader lid → 拿microplate → 放进去 → 关lid。100s limit。
- **Test Tube**：transfer 4个test tube从transparent rack到yellow wooden rack。200s limit。
- **Stirrer**：抓magnetic stir bar → 放进beaker → 把beaker放上stirrer。60s limit。

任务难度梯度：Refrigerator和Test Tube是long-horizon + precision-demanding，Stirrer是short-horizon，Microplate中等。

### 8.2 Base model + 迭代设置

Base是 $\pi_{0.5}$ VLA（Physical Intelligence, 2025, https://arxiv.org/abs/2504.16054），用flow-matching训练。

迭代流程：
1. 用上一round的best checkpoint做autonomous rollout收集 + HITL收集。
2. VLAC-CUT segment autonomous rollout。
3. 构造新训练set：base demos + HITL recovery + curated rollout ≈ 1:1:1。
4. Flow-matching SFT训练。
5. 评估，best checkpoint进入下一round。

Refrigerator / Microplate / Test Tube做2轮，Stirrer做1轮（单轮就90%）。

### 8.3 Table 4: 主结果

```
Task           Model        TP  SR    Time   FailProg
Refrigerator   Base         10  20%   72.3   40%
Refrigerator   HITL(i=1)    14  35%   89     42.3%
Refrigerator   HELP(i=1)    19  45%   85     49%
Refrigerator   HITL(i=2)    33  60%   65.6   56%
Refrigerator   HELP(i=2)    42  80%   67.2   62.5%

Microplate     Base         13  30%   83.3   30%
Microplate     HITL(i=1)    16  40%   88.35  27.5%
Microplate     HELP(i=1)    18  45%   91.4   34.5%
Microplate     HITL(i=2)    25  60%   85.1   51.3%
Microplate     HELP(i=2)    42  90%   77.7   20%

Test Tube      Base         14  35%   93     38.9%
Test Tube      HITL(i=1)    19  45%   83     41.2%
Test Tube      HELP(i=1)    20  55%   105    45%
Test Tube      HITL(i=2)    28  70%   90.7   65%
Test Tube      HELP(i=2)    37  95%   93.4   76%

Stirrer        Base         66  55%   30     33.3%
Stirrer        HITL(i=1)    83  70%   30.5   35%
Stirrer        HELP(i=1)    112 90%   29     45%
```

关键observation：
1. **第一轮execution time增加，第二轮execution time下降**。第一轮HITL recovery data教policy recover from error state，但recover需要时间，所以execution time长。第二轮curated rollout data让policy直接走正确路径，不再先错再recover，所以time下降。
2. **第二轮throughput增益远大于第一轮**。Refrigerator: 19→42 (+120%) vs 10→19 (+90%)。Microplate: 18→42 (+133%) vs 13→18 (+38%)。Test Tube: 20→37 (+83%) vs 14→20 (+43%)。这跟RL里的"compound return"类似——stronger policy产生更高质量rollout，更高质rollout产生更强policy。
3. **Microplate HELP(i=2)的failure progress降到20%**——这是因为剩下的失败集中在最难的step（开reader lid），early failure拉低average。这其实是个good sign：policy已经能在lid之后稳定执行，failure集中在一个local bottleneck。

### 8.4 Table 5: 数据组成

Refrigerator i=1: base 94 + HITL 201 + (HELP only) curated 100  
Refrigerator i=2: base 94 + HITL 242 + (HELP only) curated 117  
Microplate i=1: base 112 + HITL 168 + curated 173  
Microplate i=2: base 112 + HITL 134 + curated 116  
Test Tube i=1: base 150 + HITL 352 + curated 142  
Test Tube i=2: base 150 + HITL 300 + curated 141  
Stirrer i=1: base 101 + HITL 230 + curated 334

Stirrer的curated rollout特别多（334），因为这个任务简单，autonomous rollout成功率高，curated segments多。Test Tube的HITL特别多（352→300），因为这个long-horizon任务failure点多，需要更多human intervention。

### 8.5 Human-supervision amplification factor

$$A_M = \frac{M(\pi_{\mathrm{HELP}}) - M(\pi_{\mathrm{start}})}{M(\pi_{\mathrm{HITL}}) - M(\pi_{\mathrm{start}})} \tag{4}$$

$M$ 可以是throughput或success rate。$\pi_{\mathrm{start}}$ 是matched comparison的common starting checkpoint。HITL和HELP用**相同数量的HITL recovery trajectories**，HELP**额外**用VLAC-CUT curated的autonomous rollout segments。所以 $A_M > 1$ 意味着VLAC-CUT从相同human supervision预算里挤出更多policy gain。

### Table 6: amplification factor

| Task-Round | HITL budget | ΔTP_HITL | ΔTP_HELP | A_TP | ΔSR_HITL | ΔSR_HELP | A_SR |
|---|---|---|---|---|---|---|---|
| Refrigerator-1 | 201 | +4 | +9 | 2.25× | +15pp | +25pp | 1.67× |
| Microplate-1 | 168 | +3 | +5 | 1.67× | +10pp | +15pp | 1.50× |
| Test Tube-1 | 352 | +5 | +6 | 1.20× | +10pp | +20pp | 2.00× |
| Stirrer-1 | 230 | +17 | +46 | 2.71× | +15pp | +35pp | 2.33× |
| Refrigerator-2 | 242 | +14 | +23 | 1.64× | +15pp | +35pp | 2.33× |
| Microplate-2 | 134 | +7 | +24 | **3.43×** | +15pp | +45pp | **3.00×** |
| Test Tube-2 | 300 | +8 | +17 | 2.13× | +15pp | +40pp | 2.67× |
| **Mean** | — | — | — | **2.15×** | — | — | **2.21×** |

Throughput gain amplification 1.20×–3.43×，success rate gain amplification 1.50×–3.00×。Microplate-2的3.43×最dramatic——这个task的base 30%→HITL 60%→HELP 90%，三倍success rate jump说明curated rollout在这个setting下信息密度极高。

注意Test Tube-1的A_TP只有1.20×——这跟i=1时policy还不够强、autonomous rollout质量不够好有关。到了i=2，policy强了，curated rollout质量上来了，A_TP就到2.13×。这佐证了compound return的intuition。

---

## 9. Appendix B —— VR Teleoperation Assistance

VR teleop常见的issue：packet loss、network jitter、packet reordering、deadline miss导致某些control slot没有valid command。简单fix（hold pose、repeat last command、interpolation）会引入pause或stale motion。

HELP的解法：**context-aware human-intent inpainting**。复用正在deploy的 $\pi_\theta$，把VR command stream当成partially observed action trajectory：

$$\hat{\mathbf{A}}_t \sim p_\theta(\mathbf{A} \mid \mathbf{o}_t, \mathbf{s}_t, \ell, \mathbf{A}_t^{\mathrm{obs}}, \mathbf{m}_t) \tag{21}$$

$$\hat{\mathbf{A}}_{t,i} = \mathbf{A}_{t,i}^{\mathrm{obs}} \quad \forall i \text{ such that } m_{t,i} = 1$$

变量：
- $\mathbf{o}_t$ —— visual observation。
- $\mathbf{s}_t$ —— robot state。
- $\ell$ —— task instruction。
- $\mathbf{A}_t^{\mathrm{obs}}$ —— 部分观察到的VR action chunk（fixed-rate control slots）。
- $\mathbf{m}_t$ —— availability mask（哪些slot有VR command）。
- $p_\theta$ —— deployed VLA induced的conditional action distribution。

Inference时unknown slots联合生成short-horizon context-consistent sequence，observed slots硬projection保留。Signal-quality gate在command太stale或interruption太长时disable model-generated motion，让robot hold pose。

这个设计的妙处：**teleoperation assistance的inpainted action跟policy post-training是同一个model，不需要额外训练separate teleop policy**。随着policy变得更强，teleop assistance也跟着变强，形成正反馈。这跟ALOHA-style的shared autonomy（https://arxiv.org/1802.01537）思路类似，但更加seamless。

---

## 10. 跟相关工作的联系

| 工作 | 关系 |
|---|---|
| **$\pi_{0.5}$** (https://arxiv.org/abs/2504.16054) | Base VLA，flow-matching decoder |
| **$\pi_{0.6}$** (https://arxiv.org/abs/2511.14759) | Near-online real-world RL，混合demonstration + rollout + intervention |
| **ROVE** (https://arxiv.org/abs/2606.17011) | Humanoid VLA post-training，从imperfect human intervention里prioritize high-value behavior |
| **SOP** (https://arxiv.org/abs/2601.03044) | Scalable online post-training for VLA fleet |
| **Learning While Deploying** (https://arxiv.org/abs/2605.00416) | Fleet-scale RL streaming |
| **VLAC** (https://arxiv.org/abs/2509.15937) | VLAC-CUT的predecessor，process reward modeling |
| **ConSFT** (https://arxiv.org/abs/2605.08879) | Conservative SFT，避免catastrophic forgetting |
| **Robo-Dopamine** (https://arxiv.org/abs/2512.23703) | General process reward modeling baseline |
| **Robometer** (https://arxiv.org/abs/2603.02115) | Trajectory comparison reward model baseline |
| **RoboReward** (https://arxiv.org/abs/2601.00675) | General-purpose VLM reward baseline |
| **ProgressLM** (https://arxiv.org/abs/2601.15224) | Progress reasoning from partial observation |
| **GVL** (https://arxiv.org/abs/2310.12931) | Generative Value Learning，shuffled frame prompting |
| **TOPReward** (https://arxiv.org/abs/2602.19313) | Token probabilities as zero-shot reward |
| **DROID** (https://arxiv.org/abs/2403.12945) | Large-scale real-world manipulation dataset |
| **LIBERO** (https://arxiv.org/abs/2306.03310) | Lifelong robot learning benchmark |
| **OpenVLA** (https://arxiv.org/abs/2406.09246) | Open-source VLA |
| **Diffusion Policy** (https://arxiv.org/abs/2303.04137) | Diffusion-based visuomotor policy |
| **RewardBench** (https://arxiv.org/abs/2403.13787) | Reward model evaluation for LLM |
| **VLRewardBench** (https://arxiv.org/abs/2411.17451) | Multimodal reward model evaluation |
| **DeepSeek-R1** (https://arxiv.org/abs/2501.12948) | RL training simplification, GRPO |
| **Qwen-VL** (https://arxiv.org/abs/2308.12972) | VLM backbone基础 |
| **RoboVQA** (https://arxiv.org/abs/2402.09155) | Multimodal long-horizon robot reasoning |
| **World-Env** (https://arxiv.org/abs/2509.24948) | World model as virtual env for VLA post-training |
| **AtomVLA** (https://arxiv.org/abs/2603.08519) | Latent world model post-training |
| **ConRFT** (https://arxiv.org/abs/2502.05450) | Reinforced fine-tuning for VLA via consistency policy |

---

## 11. Intuition总结

我把paper里的几个非trivial设计选择列一下，你可以判断它们是不是真的"必须"：

1. **为什么是2个operator而不是1个**：1个operator要同时VR teleop + 物理reset + monitor fleet，task switching cost巨大，而且teleop training成本高（weeks-months），不能浪费在走路上。2个operator让specialization成为可能。这个2:12 ratio是个interesting number，能不能2:24？paper没说，但理论上Teleoperator是瓶颈——12个robot平均每个5-6分钟一次takeover，VR切换+操作+verbal handoff大概30s-1min，2:12是接近极限的。

2. **为什么VLAC-CUT是process-level而不是episode-level**：episode-level success label无法区分"高效执行"和"试错后成功"，也无法识别哪个intermediate action导致stagnation。process-level signed supervision让segmentation成为可能。这跟PRM（Process Reward Model）在math reasoning里的成功（参考Math-PRM、DeepSeek-R1）是同一种intuition：dense supervision比sparse更好。

3. **为什么是flow-matching而不是RL**：因为 $\pi_{0.5}$ 是flow-matching decoder，RL在flow-matching上的稳定训练还不太成熟（参考Flow Q-Learning, Park et al. 2025, https://arxiv.org/abs/2502.02538）。SFT on curated segments是简单稳定的workaround，代价是无法用failure-inducing action作为negative signal。

4. **为什么用Qwen3-VL-30B-A3B做VLAC-CUT**：30B dense太大，30B-A3B是MoE，inference只激活3B，CPU/GPU memory友好。VLAC-CUT跑offline segmentation，不需要real-time，所以模型大小可以大一点。

5. **为什么signed progress是[-100, 100]而不是[0, 1]**：因为要区分"还没开始"（0）和"开始但比initial还差"（<0）。recovery过程中常常比initial state更差（比如把object碰掉到地上，要重新抓起来），这种regression在[0, 1] scale里没法表达。

6. **为什么需要ARX-data**：因为public dataset都是expert demonstration，没有physically executed failure。纯synthetic reverse augmentation（rewind video）虽然能制造decreasing progress sample，但visual dynamics不realistic。ARX-data补这个gap。

7. **为什么predictive takeover model是必要的**：12个robot同时跑，Floor Operator不可能每秒盯着所有robot。takeover predictor $\phi$ 帮Floor Operator在"看着的robot"和"快出问题的robot"之间prioritize。这个$\phi$是**辅助**Floor Operator，不是替代——最终takeover决策权在人。

8. **为什么ConSFT而不是experience replay**：experience replay需要maintain buffer，跨robot跨task的buffer管理复杂。ConSFT用loss itself作为confidence proxy，无buffer无reference model，工程上simple。代价是失去了replay的explicit anti-forgetting能力。

---

## 12. 可能的局限和open question

paper没有很深入讨论的：

1. **Takeover model $\phi$ 的accuracy没单独report**。如果$\phi$ false positive高，Floor Operator会被spammed；false negative高，policy会卡在error state浪费rollout时间。这是system bottleneck但没数据。

2. **VLAC-CUT的inference cost**：30B-A3B MoE对每条autonomous rollout做segmentation，latency多少？paper没说。如果是离线跑，可以batch overnight；如果是online filter，可能成瓶颈。

3. **Curated rollout segments里recovery segment的定义边界**：recovery segment跟failure-inducing segment往往是连续的，怎么决定切点？paper里的"signed progress + local direction"应该能处理，但具体阈值没给。

4. **Generalization到unseen task**：real-world实验都是4个specific task，VPB有unseen task bucket，但real-world post-training没测unseen task。HELP在unseen task上能不能scale？

5. **HITL data和curated rollout的1:1:1 mixing是heuristic还是有理论支撑**？paper没justify。

6. **Refrigerator task saturate在80%**，paper归因于camera viewpoint + 机械结构。这是honest reporting但说明HELP对"perceptual bottleneck"无能为力。

7. **Stirrer只做1轮**，因为90% success rate后认为不需要更多iteration。但throughput 112还能更高吗？longer training会不会天花板比90%高？

---

## 13. 我对这篇paper的整体判断

这是篇很"系统"的工作，不是"新方法"的工作。它的核心贡献是把多个已知component（role specialization、teleop assistance、process reward modeling、conservative SFT、predictive takeover）assemble成一个coherent pipeline，然后在一个真实的2-operator / 12-robot setup上跑出可复现的数字。

最有technical depth的是VLAC-CUT（process-level signed progress supervision + 4-class segmentation），其次是人类角色分解（Teleoperator + Floor Operator）。最有工程含量的是distributed architecture + dual-process client + dynamic VR routing。

它的实验说服力来自"matched HITL budget"的controlled comparison——同样数量HITL recovery trajectories，VLAC-CUT额外amplify 2.15×/2.21×。这个数字量化了"autonomous rollout curation"的价值，比简单的"我们method更好"有说服力得多。

如果你要从这篇paper里take away一个idea回去用，我会推荐**process-level signed progress supervision for rollout segmentation**这个concept——它在很多imitation learning from mixed-quality data的setting都适用，不止robotics。

主要reference links：
- HELP/VLAC-CUT: https://arxiv.org/abs/2509.15937 (VLAC)
- $\pi_{0.5}$: https://arxiv.org/abs/2504.16054
- $\pi_{0.6}$: https://arxiv.org/abs/2511.14759
- ConSFT: https://arxiv.org/abs/2605.08879
- DROID: https://arxiv.org/abs/2403.12945
- Robo-Dopamine: https://arxiv.org/abs/2512.23703
- GVL: https://arxiv.org/abs/2310.12931
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- RewardBench: https://arxiv.org/abs/2403.13787
- OpenVLA: https://arxiv.org/abs/2406.09246
- Diffusion Policy: https://arxiv.org/abs/2303.04137
