---
source_pdf: Sparse Autoencoders Reveal Interpretable and.pdf
paper_sha256: 3dad10788f216489c86e2add923d70292f3a570d83377a94fe8e2127014d1012
processed_at: '2026-08-12T08:48:36-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

---

## 一句话版本

他们给VLA model的"大脑"装了个fMRI机器，想看看robot policy fine-tune之后，脑子里到底装的是"通用技能"还是"背台词"。答案有点让人心凉：**90%以上是在背台词**，但好在确实有一小撮真正通用的"技能神经元"混在里面，而且你还能手动拨动这些神经元来操控机器人行为。

---

## 他们到底在问什么问题

你train了一个VLA，在LIBERO上跑90%+ success rate，paper写得很漂亮。然后你拿去稍微perturb一下——换个object位置、改个lighting——success rate直接崩到接近0。LIBERO-PRO (https://arxiv.org/abs/2510.03827) 把这个现象钉死了。

大家嘴上都在说"memorization"、"overfitting"、"brittle generalization"，但这些都是**behavioral层面**的描述——你看到robot失败了，你猜它在memorize，但你不知道model内部到底发生了什么。就像医生看到病人发烧，但没拍X光。

这篇paper干的事就是拍X光。工具是Sparse Autoencoder (SAE)，从LLM interpretability那边搬过来的。核心idea：neural network的activation空间里，concept是被"叠"在一起的（superposition，https://arxiv.org/abs/2209.10652），你直接看activation看不懂，但如果你训一个autoencoder把它拆成sparse的"feature字典"，每个feature会对应一个human-interpretable的概念。Anthropic用这个在Claude里找到了"Golden Gate Bridge"神经元（https://transformer-circuits.pub/2024/scaling-monosemanticity/），steer一下就能让Claude满嘴Golden Gate。

Stanford这帮人把同样的方法用到了$\pi_{0.5}$和OpenVLA上。

---

## SAE到底怎么工作的——用最intuitive的方式

想象residual stream是一个2048维的"意念空间"。每个timestep，model把当前的视觉+语言+状态压缩成这个空间里的一个点。问题是这个点里"叠"了无数个concept——grasp的概念、red color的概念、approach goal的概念——全部挤在同一个2048维向量里。

SAE做的事情：

**第一步**，把input activation normalize到一个unit sphere上（Eq. 1）。为什么？因为不同timestep的activation magnitude可能差很多，但direction才是真正carry semantic meaning的部分。先把magnitude nuisance干掉。

$$\tilde{\mathbf{x}} = \frac{(\mathbf{x} - \mathbf{b}_{\mathrm{pre}}) - \mu}{\|(\mathbf{x} - \mathbf{b}_{\mathrm{pre}}) - \mu\|_2}$$

- $\mathbf{x}$：raw activation，$\mathbf{b}_{\mathrm{pre}}$是prebias（初始化为geometric median，比mean更robust to outliers），$\mu$是scalar mean
- 输出$\tilde{\mathbf{x}}$是unit vector

**第二步**，用encoder把2048维input映射到一个dictionary上，然后TopK只保留top 100个feature激活（Eq. 2）：

$$\mathbf{z} = \mathrm{ReLU}(\mathrm{TopK}(\mathbf{W}_{\mathrm{enc}}(\tilde{\mathbf{x}})))$$

- $\mathbf{W}_{\mathrm{enc}}$：encoder权重矩阵，把input投影到每个feature的方向上
- TopK：只保留100个最强响应，其余直接置零
- ReLU保证非负

**为什么TopK而不是L1 penalty？** L1 sparsity penalty有个烦人的side effect——它会把所有activation往小压（shrinkage bias），而且sparsity level很难精确控制。TopK直接说"我就要恰好100个active features"，简单粗暴，no hyperparameter sensitivity。这是从Winner-Take-All Autoencoder (https://arxiv.org/abs/1409.2752) 那条线来的idea。

**第三步**，decoder把sparse code重建回activation space，loss就是reconstruction error + 一个auxiliary loss（Eq. 3）：

$$\mathcal{L} = \frac{\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2}{C_{\mathrm{MSE}}} + \alpha \cdot \frac{\|\tilde{\mathbf{e}} - \hat{\tilde{\mathbf{e}}}_{\mathrm{aux}}\|_2^2}{C_{\mathrm{MSE}}}$$

- $C_{\mathrm{MSE}}$：initialization时centered activations的variance，让loss scale-invariant
- $\alpha = 1/32$：auxiliary loss权重
- $\tilde{\mathbf{e}} = \tilde{\mathbf{x}} - \hat{\tilde{\mathbf{x}}}$：main SAE没重建好的residual
- $\hat{\tilde{\mathbf{e}}}_{\mathrm{aux}}$：用dead latents去重建这个residual

**AuxK的intuition**：SAE训练有个大问题——很多feature训着训着就死了，再也不激活，白占dictionary capacity。AuxK loss让这些dead latents去"捡残渣"，把main SAE没建好的residual兜底重建。这样dead features被reactivate，dictionary utilization提升。$k_{\mathrm{aux}}=512$远大于main $k=100$，因为dead latents要承担大量residual work。

**一个反直觉的选择：expansion ratio = 1**。Anthropic在Claude上用32×甚至更高（dictionary远大于input维度），这里只用1×（dictionary = input维度）。为什么？因为robotics dataset比LLM pretraining data小3-4个数量级，overcomplete dictionary会"过度解剖"已经稀疏的信号，导致大量dead features。Figure 7的ablation证实了这点。

---

## 怎么判断一个feature是"通用技能"还是"背台词"

训完SAE你拿到几千个features，每个都是一个direction in activation space。现在的问题：哪些是真正general的motion primitive，哪些只是在memorize某个specific episode？

他们定义了4个temporal metrics。先约定符号：$f_j(\mathbf{x}_t^{(e)})$是feature $j$在episode $e$的timestep $t$的激活值，$E_j^+$是feature $j$至少激活过一次的episode集合。

### Metric 1: Episode Coverage $c_j$ (Eq. 4)

$$c_j = \frac{|E_j^+|}{|E|}$$

大白话：这个feature在多少比例的episode里至少fire过一次。如果在1693个LIBERO episode里1500个都fire了，coverage ≈ 0.89，很general。如果只在3个episode里fire，coverage ≈ 0.002，大概率是在背那3个episode的台词。

### Metric 2: Mean Onset Count $\bar{o}_j$ (Eq. 5-7)

这个稍微subtle。先定义一个带hysteresis的state machine：

$$s_t = \begin{cases} 1 & \text{if } f_j(x_t) > \tau_{\mathrm{on}} \\ 0 & \text{if } f_j(x_t) = 0 \\ s_{t-1} & \text{otherwise} \end{cases}$$

- $\tau_{\mathrm{on}} = 0.1$：threshold，低于这个值且非零时保持原状态（hysteresis防抖动）
- 然后数0→1的transition次数，就是onset count $o_j$
- $\bar{o}_j$是只在active episode里平均

**Intuition**：想象一个"grasp event"的feature。一个pick-and-place episode里有2次grasp，这个feature就会bursty地fire两次，$\bar{o} = 2$。如果是two-object task就是4次。而memorized feature一旦激活就持续整个episode，$\bar{o} \approx 1$。

**这个metric的妙处**：它capture的是"事件性"——feature是不是在响应某个discrete sensorimotor event，而non持续地编码某个static scene property。

### Metric 3: Mean Activation Magnitude $\bar{a}_j$ (Eq. 8)

$$\bar{a}_j = \frac{1}{|E_j^+|} \sum_{e \in E_j^+} \max_t f_j(\mathbf{x}_t^{(e)})$$

每个active episode里的peak activation，再平均。Capture firing的典型强度。辅助metric，主要是为了给classifier提供更多信息。

### Metric 4: Relative Run Length $\bar{\ell}_{r,j}$ (Eq. 9-10)

$$\bar{\ell}_{r,j} = \frac{1}{|E_j^+|} \sum_{e \in E_j^+} \frac{r_j^{(e)}}{T^{(e)}}$$

- $r_j$：每个onset的平均持续timestep数
- $T^{(e)}$：episode $e$的总长度
- 除一下得到fraction

**Intuition**：接近0说明feature是transient的——fire一下就灭了，典型event-driven。接近1说明feature一旦fire就持续整个episode——典型memorized scene encoding。

$\bar{o}$和$\bar{\ell}_r$是互补的。Memorized feature是$\bar{o}$低、$\bar{\ell}_r$高（once on, stays on）。General feature是$\bar{o}$高、$\bar{\ell}_r$低（bursty events）。

---

## Classifier：自动化区分general vs memorized

手动标30个features（15 general, 15 memorized），训一个logistic regression（Eq. 11）：

$$P(\mathrm{general} \mid \mathbf{m}) = \sigma(\beta_0 + \beta_1 \bar{o} + \beta_2 c + \beta_3 \bar{a} + \beta_4 \bar{\ell}_r)$$

- $\sigma$：logistic sigmoid
- 输入是4个metrics组成的vector $\mathbf{m}$
- 输出是"这个feature是general"的概率

**LIBERO上的fitted coefficients**：$\beta_0 = -4.20, \beta_1 = 1.89, \beta_2 = 1.80, \beta_3 = 0.52, \beta_4 = -0.36$

读这个数：$\beta_1 \approx \beta_2$，说明在LIBERO里onset count和episode coverage同等重要。LIBERO场景少（~20个visual scenes），光有breadth不够，还得bursty。$\beta_4 < 0$确认sustained activation = memorized。LOO-CV accuracy 100%。

**DROID上的fitted coefficients**：$\beta_0 = -1.78, \beta_1 = 0.74, \beta_2 = 2.36, \beta_3 = 0.35, \beta_4 = -1.04$

读这个数：$\beta_2$是$\beta_1$的3.2倍——在DROID上episode coverage dominate。为什么？DROID有1545个unique tasks vs LIBERO的130个，dataset本身diverse，所以"能跨很多episode fire"这件事本身就strongly signal generality。$\beta_4 = -1.04$比LIBERO的$-0.36$强3倍，因为DROID episode更长更variable（$\mu=283.5, \sigma=219.2$ timesteps, CV=0.77 vs LIBERO的$\mu=161.5, \sigma=68.2$, CV=0.42），run length是更强的differentiator。

**Deep insight**：同一个metric在不同dataset上的discriminative power完全不同。Generality这个concept本身是dataset-relative的。在低diversity dataset上你看burstiness，在高diversity dataset上你看breadth。这有点像evaluating ML model——metric的选择depends on data distribution。

---

## 最striking的实验结果

### 结果1：VLA model里90%+都是memorization

Table 2的数据令人清醒：

| Model | Dataset | Layer | Features总数 | Memorized % |
|-------|---------|-------|--------------|-------------|
| $\pi_{0.5}$ | LIBERO | PG5 | 2044 | 98.43% |
| $\pi_{0.5}$ | LIBERO | PG avg | 7175 | 97.37% |
| $\pi_{0.5}$ | DROID | PG5 | 2046 | 94.92% |
| $\pi_{0.5}$ | DROID | PG avg | 6649 | 89.19% |
| OpenVLA | LIBERO Goal | L8 | 1775 | 99.55% |
| OpenVLA | LIBERO Goal | LM avg | 9389 | 99.55% |

OpenVLA在LIBERO Goal上99.55% memorized——428个episode基本被rote learn干净。$\pi_{0.5}$在DROID上89% memorized，看起来好一些但绝对值仍然高。

Trend很清晰：LIBERO Goal (99.55%) → full LIBERO (97.37%) → DROID (89.19%)。Dataset越大越diverse，memorization比例越低。这给"more diverse data helps generalization"这个folk wisdom提供了**mechanistic evidence**，而non只是behavioral observation。

### 结果2：但确实有真正general的features，而且很beautiful

LIBERO PG5上找到4个super general features（episode coverage > 0.99）：

**F1129 (grasp/place)**: 每次grasp或place时burst fire。Single-object task → 2个onset（一次grasp一次place）。Two-object task → 4个onset。**Onset count随sub-goal数量scale**——这是compositional structure的直接证据！$P(\mathrm{general}) = 0.91$。

**F1902 (transport)**: 在F1129的两个onset之间fire，对应carrying phase。Magnitude随end-effector接近goal position**线性增加**——这个feature不仅encode"在transport"，还encode"离goal多近"。$P(\mathrm{general}) = 0.89$。

**F128 (pre-grasp alignment)**: End-effector在object正上方时fire。Magnitude随object进入wrist camera中心ramp up。Post-placement如果arm回到vertical alignment会reactivate。$P(\mathrm{general}) = 0.92$。

**F445 (task completion)**: 接近goal location时fire，主要在final sub-goal。Compound task里magnitude更低且prefer second placement——encode overall task success而non sub-task completion。$P(\mathrm{general}) = 0.58$（ borderline，classified as general但weakly）。

这4个features构成一个**hierarchical motion primitive decomposition**：
```
pre-grasp alignment (F128) 
    → grasp/place (F1129) 
        → transport (F1902) 
            → task completion (F445)
```

每个feature的temporal firing pattern精确锁在sensorimotor event上。这就是你期望一个"真正学到了技能"的policy应该有的内部结构。

DROID上也有类似发现：
- **F158 (sub-task checkpoint)**：在8-bottle pick-and-place episode里fire 16次，每次bottle进入wrist camera或goal plate可见时onset。Highest $\bar{o}$ across all SAEs。
- **F586 (pinch grasp)**：Precision grasp of thin objects时fire。对wider grasps以~0.5× magnitude fire——**encode grip aperture as continuous signal**，non just binary grasp event。
- **F165 (open gripper over target)**：Target在open gripper jaws之间可见时fire。
- **F399 (grasp acquisition/placement)**：Grasp closing和placement时fire，transport时deactivate。

### 结果3：Steering能causally操控robot行为

这是最能build intuition的部分。他们把SAE decoder的某个column（即某个feature的direction vector）直接加到residual stream上：

$$\mathbf{v}_i = \mathbf{W}_{\mathrm{dec}}[:, i], \quad \mathbf{y}' = \mathbf{y} + \alpha \cdot \mathbf{v}$$

- $\mathbf{v}_i$：decoder第i列，unit norm by construction
- $\alpha$：scalar steering magnitude
- 加到每个token position、每个denoising step

**Steer F128 (pre-grasp alignment)** across 3个tasks：

1. "Move black bowl from drawer to plate"：robot接近bowl，然后**悬停在bowl上方不grasp**。
2. "Turn on stove and put moka pot on stove"：接近moka pot，悬停。
3. "Pick up alphabet soup and place in basket"：接近soup，悬停。

完美consistent with "pre-grasp alignment"的interpretation——你把这个feature强行一直激活，model就以为"我已经在pre-grasp alignment状态了"，于是stuck在那里。

**Steer F1902 (transport)** across 3个tasks：

1. "Move black bowl to plate"：robot**直接skip grasp**，move到plate位置。
2. "Put white mug on plate"：直接move到plate。
3. "Pick up alphabet soup and place in basket"：直接move到basket，甚至**collide with basket把它push开**。

你把transport feature一直激活，model就skip掉grasp phase直接进入"transport to goal"模式。

**一个重要observation**：即使high magnitude steering，policy仍然保持goal-directed behavior，会attempt approach/grasp/pursue alternative subtasks，non退化成random motion。VLA的computation对single-feature perturbation有相当redundancy。这和LLM steering的体验类似——steer Golden Gate feature，Claude不会崩，只是变得obsessed with Golden Gate。

### 结果4：Knowledge Insulation能reverse memorization趋势

Figure 6是关于training dynamics的。随着fine-tuning steps增加：
- Episode coverage下降（features越来越只在小范围episode fire）
- Relative run length上升（features越来越sustained）
- 两者都指向**generality退化，memorization增强**

这就算$\pi_{0.5}$ base是在10,000+小时cross-embodiment robot data上pretrain的，catastrophic forgetting依然从fine-tuning一开始就发生。

**Knowledge Insulation (KI)** (https://arxiv.org/abs/2505.23705) 能reverse这个趋势。KI的核心idea：用stop gradient + architectural separation让VLM backbone的representation不被fine-tuning破坏，让一个separate expert处理continuous control。

P(general) median across DROID PG5 features：
- 10k steps: 0.190
- 30k steps: 0.181  
- 60k steps: 0.179
- 90k steps: 0.181
- KI: **0.206**

KI把distribution往generality方向shift。虽然shift不大，但statistically meaningful。这给了KI的effectiveness一个mechanistic explanation，non只是"benchmark数字高了一点"。

---

## 一些更subtle的技术细节

### Per-token vs mean-pooled activations

主paper用mean-pooled activations——每个timestep把所有token（768个image patch + ~20个text token）average成一个vector。

为什么non per-token？Storage。768 tokens × 2048 dim × 2000 episodes × 4 bytes ≈ 3.5TB per layer。Robotics lab的compute budget扛不住。

Per-token experiments (Section A.7) 是promising但less interpretable。他们把每个image的256 patches sum成1个vector，保留individual text token。发现一些有趣feature：
- **F1881**: 主camera上uniform activation，noun "pot"和modifier "black"几乎identical activation pattern——link co-referent tokens
- **F225**: 纯textual semantic，camera零激活，noun强激活，verbs不激活
- **F1659**: Memorized，只在含lid/pot的episode fire

但general per-token features少（<10 per layer），且top-activating text tokens常常是function words（"the", "and"）。这暗示VLA里的visual-textual alignment比pure VLM **less crisp**。

### Steering的layer dependence

Figure 11的ablation：Action Expert layer 0最steerable，随depth急剧衰减。PaliGemma layer 17产生**exactly zero displacement**——essentially frozen representation。

**Intuition**：Early action expert layers是action computation的"入口"，perturb这里能直接shift action output。Later layers已经converge to specific action trajectory，perturb被"吸收"掉了。PG L17 zero effect说明KI确实在保护VLM backbone——high-level semantic representation被lock死了。

### Classifier的false negatives

Section A.5.1讨论了两个misclassification case：

**F1939 (LIBERO PG5)**: $\bar{o} = 1.00, c = 0.732$。Fire in first 20 timesteps of every episode，encode robot "home" position。Scene-invariant, task-invariant, present in 73% episode。显然general。但$\bar{o} = 1.00$让classifier误判为memorized——因为classifier假设general features一定multi-onset，而home position feature恰好once-per-episode。

**F1381 (DROID PG5)**: $\bar{o} = 1.00, c = 0.226, \bar{\ell}_r = 0.990$。Lid grasp feature，在86% lid-related episode fire。但lid episode只占DROID的6.7%，coverage低于decision boundary。而且lid grasp是sustained的（整个grasp过程持续），$\bar{\ell}_r$高，被额外penalize。

这两个case暴露一个structural limitation：**classifier confuses low burstiness with memorization**，但有些general concept（home position, lid grasp）天然是once-per-episode的。需要dataset-diversity-aware normalization of coverage，或者额外的cross-scene consistency metric。

---

## 这篇paper的big picture意义

### 对VLA research community的message

1. **你benchmark上的90% success rate可能99%来自memorization**。OpenVLA在LIBERO Goal上99.55% features是memorized。LIBERO-PRO的behavioral finding现在有了mechanistic correlate。

2. **General features确实emerge，但需要足够diverse data**。LIBERO Goal → full LIBERO → DROID，memorization比例单调下降。但即使DROID (1545 tasks) 仍89% memorized。Robotics dataset相比LLM pretraining data小太多数量级，scale远未到generalization dominate的regime。

3. **Compositional structure在feature层面是可见的**。F1129的onset count随sub-goal数量scale，F1902在F1129 onset pair之间fire——这是temporal compositionality的直接证据。只是这些general features被海量memorization features淹没了。

4. **SAE metrics可作training-time proxy for generalization**。无需rollout，只需看episode coverage和run length的trend。这给cheap training-time diagnostic开了条路。

### 对你的直觉的build

这篇paper的core insight对你（Karpathy）来说可能特别resonate：你一直强调neural network的"magic"在于learned representation，而non surface behavior。这篇paper正是在representation层面打开了VLA的黑箱。

几个可能让你兴奋的方向：

**Feature-level regularization**：训VLA时直接penalize memorization features的activation，或者reward general features的activation。这比behavior-level regularization（如LIBERO-PRO的perturbation training）更precise。

**Steering as RL reward shaping**：在RL fine-tuning中，用SAE features作为auxiliary reward signal。Reward general features激活，penalize memorized features激活。这可能比pure task success reward更好的induce generalization。

**SAE-guided data curation**：如果一个feature被classified as memorized，说明model在"背"某些episode。可以反向identify这些episode，从training set里downweight或remove，force model学general feature。这类似"hard example mining"的inverse。

**Per-token SAE + attention pattern**：这篇paper的per-token SAE是preliminary attempt。真正理解VLA的visual-textual grounding，需要per-token SAE + attention head analysis的组合。你之前在micrograd/makemore里讲attention时的那些intuition（attention as communication protocol）在这里直接applicable。

**Real robot steering**：Sim里的robustness可能non transfer到real。$\pi_{0.5}$在real robot上steering会怎样？会不会sim的redundancy在real上breaks？这connects到你一直强调的sim2real gap。

**Connection to你的"software 2.0"vision**：SAE features本质上是在"反编译"software 2.0的learned program。每个general feature对应program里的一个"函数"或"子程序"，memorized feature对应"硬编码的lookup table"。这篇paper显示VLA的"program"里90%+是lookup table，只有少量真正的函数。如何让software 2.0写出更多函数、更少lookup table，是open problem。

---

## 相关links汇总

**这篇paper本身**:
- 项目主页: https://drvla.github.io
- Paper PDF (arxiv): 暂未找到正式arxiv link，project page上应该有

**VLA models**:
- $\pi_0$: https://arxiv.org/abs/2410.24164
- $\pi_{0.5}$: https://arxiv.org/abs/2504.16054
- OpenVLA: https://proceedings.mlr.press/v270/kim25c.html
- RT-2: https://proceedings.mlr.press/v229/zitkovich23a.html

**SAE & Mechanistic Interpretability**:
- Towards Monosemanticity (Anthropic): https://transformer-circuits.pub/2023/monosemantic-features
- Scaling Monosemanticity (Claude): https://transformer-circuits.pub/2024/scaling-monosemanticity/
- Scaling SAEs (Gao et al., TopK+AuxK): https://arxiv.org/abs/2406.04093
- Toy Models of Superposition: https://arxiv.org/abs/2209.10652
- SAEs find interpretable features (Cunningham et al.): https://arxiv.org/abs/2309.08600
- Winner-Take-All Autoencoders: https://arxiv.org/abs/1409.2752
- SAEs in VLMs: https://arxiv.org/abs/2504.02821
- Golden Gate Claude (steering demo): https://www.anthropic.com/news/golden-gate-claude

**Robotics benchmarks & datasets**:
- LIBERO: https://arxiv.org/abs/2306.03310
- LIBERO-PRO (memorization analysis): https://arxiv.org/abs/2510.03827
- DROID: https://arxiv.org/abs/2403.12945
- Open X-Embodiment: https://arxiv.org/abs/2310.08864

**VLA generalization techniques**:
- Knowledge Insulation: https://arxiv.org/abs/2505.23705
- Don't Blind Your VLA (visual representation regularizer): https://arxiv.org/abs/2510.25616

**Backbone models**:
- PaliGemma: https://arxiv.org/abs/2407.07726
- Gemma 2: https://arxiv.org/abs/2408.00118

**Related VLA interpretability work**:
- Linear probes on OpenVLA (Lu et al.): https://arxiv.org/abs/2502.04558
- Emergent world representations in OpenVLA: https://arxiv.org/abs/2509.24559
- Mechanistic interpretability for steering VLAs (Haon et al.): https://arxiv.org/abs/2509.00328
- Controlling VLAs through sparse latent directions (Khan et al.): https://openreview.net/forum?id=wtf3ww1EOL
- Observing and controlling features in VLAs (Buurmeijer et al.): https://arxiv.org/abs/2603.05487

---

# Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models — 深度技术解析

这篇paper来自Stanford (Swann, Kennedy, Schwager等人)，把mechanistic interpretability的核心工具——Sparse Autoencoders (SAEs)——从LLM领域迁移到Vision-Language-Action (VLA) models上，核心问题是：**VLA fine-tuning后到底学到了什么？是generalizable的motion primitives，还是episode-specific的trajectory memorization？** 这与你在LIBERO-PRO、policy memorization等问题上的concerns高度一致。项目主页：https://drvla.github.io

---

## 1. 核心动机与问题定位

LIBERO-PRO (arXiv:2510.03827, https://arxiv.org/abs/2510.03827) 显示，一个在standard LIBERO protocol上>90% success rate的VLA，经过systematic perturbations后success rate collapse到near-zero。这暗示这些"成功"的policy在做trajectory-level rote memorization，而non compositional skill learning。这篇paper想用mechanistic interpretability从**模型内部**给出证据，而non behavioral/anecdotal的failure analysis。

关键insight：行为层面的brittleness已经被观察到很久，但内部到底发生了什么一直不清楚。SAE提供了一个unsupervised的"显微镜"。

---

## 2. SAE Architecture 技术细节：TopK + AuxK

他们采用Gao et al. (arXiv:2406.04093, https://arxiv.org/abs/2406.04093) 的TopK + AuxK架构，这是Anthropic/Claude SAE工作的延续。

### 2.1 Per-sample normalization (Eq. 1)

$$\tilde{\mathbf{x}} = \frac{(\mathbf{x} - \mathbf{b}_{\mathrm{pre}}) - \mu}{\|(\mathbf{x} - \mathbf{b}_{\mathrm{pre}}) - \mu\|_2}$$

变量解释：
- $\mathbf{x} \in \mathbb{R}^d$：从VLA residual stream抽出的activation（PaliGemma layer $d=2048$，action expert $d=1024$，OpenVLA $d=4096$）
- $\mathbf{b}_{\mathrm{pre}}$：learned prebias，初始化为training activations的**geometric median**（robust to outliers，比mean更稳定）
- $\mu$：沿$d_{\mathrm{model}}$维度的scalar mean
- 最终$\tilde{\mathbf{x}}$是L2-normalized unit vector

**Intuition**: 这一步把每个timestep的activation拉到一个统一的"sphere"上，消除不同timestep间magnitude的nuisance variability，让SAE学习的是directional structure而不是scale。

### 2.2 TopK encoding (Eq. 2)

$$\mathbf{z} = \mathrm{ReLU}(\mathrm{TopK}(\mathbf{W}_{\mathrm{enc}}(\tilde{\mathbf{x}})))$$

- $\mathbf{W}_{\mathrm{enc}}$：encoder权重，初始化为$\bar{\mathbf{W}}_{\mathrm{dec}}^\top \sqrt{k/n}$（decoder转置的scaled版本，$n$是residual stream维度）
- TopK保留k个最大pre-activation，其余置零
- $\pi_{0.5}$ PaliGemma用 $k=100$，action expert用 $k=64$
- 后接ReLU保证非负

**Key insight**: TopK替代传统L1 sparsity penalty，直接控制sparsity level（active features数量是exact $k$），避免L1带来的feature shrinkage bias和hyperparameter sensitivity。Winner-take-all autoencoder (arXiv:1409.2752, https://arxiv.org/abs/1409.2752) 的精神。

### 2.3 AuxK auxiliary loss (Eq. 3)

$$\mathcal{L} = \frac{\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2}{C_{\mathrm{MSE}}} + \alpha \cdot \frac{\|\tilde{\mathbf{e}} - \hat{\tilde{\mathbf{e}}}_{\mathrm{aux}}\|_2^2}{C_{\mathrm{MSE}}}$$

- $C_{\mathrm{MSE}}$：centered activations在initialization时的variance，用作normalization constant让loss scale-invariant
- $\alpha = 1/32$：auxiliary loss权重
- $\tilde{\mathbf{e}} = \tilde{\mathbf{x}} - \hat{\tilde{\mathbf{x}}}$：normalized space下的reconstruction residual
- $\hat{\tilde{\mathbf{e}}}_{\mathrm{aux}}$：用top-$k_{\mathrm{aux}}$个**dead latents**（500步内未激活）重建residual

**Intuition**: Dead latents是SAE训练的大问题——很多feature学完就死了，浪费dictionary capacity。AuxK让dead latents去"收拾残局"，把main SAE没重建好的residual捡起来。这既reduces dead feature ratio，又提升reconstruction quality。$k_{\mathrm{aux}}=512$ 远大于main $k=100$，因为dead latents需要承担较多residual work。

Decoder columns约束unit norm，gradient投影到tangent plane（OpenAI的manifold constraint技巧），gradient norm clip 1.0。

### 2.4 Expansion ratio = 1

这是与LLM SAE工作的**major departure**。Anthropic用expansion ratio 32×甚至更高。这里只用1×（OpenVLA用0.5×匹配feature数量）。

Figure 7的ablation显示：higher expansion ratio导致substantially更多dead features，且interpretable features在ER=1也能emerge。Reasons：
- Robotics dataset小3-4个数量级
- Mean-pooled tokens已经compressed
- 大dictionary会"过度dissect"已经稀疏的信号

---

## 3. Generality Quantification Metrics — 核心创新

这是paper的methodologically novel part。他们定义4个per-feature的temporal activation statistics，用来区分general vs. memorized features。

### 3.1 Episode Coverage $c_j$ (Eq. 4)

$$c_j = \frac{|E_j^+|}{|E|}$$

$E_j^+ = \{e : \exists t, f_j(\mathbf{x}_t^{(e)}) > 0\}$：feature $j$ 至少激活一次的episode集合。High coverage → feature跨diverse tasks激活 → general。

### 3.2 Mean Onset Count $\bar{o}_j$ (Eq. 5-7)

State machine with hysteresis：
$$s_t = \begin{cases} 1 & \text{if } f_j(x_t) > \tau_{\mathrm{on}} \\ 0 & \text{if } f_j(x_t) = 0 \\ s_{t-1} & \text{otherwise} \end{cases}$$

- $\tau_{\mathrm{on}} = 0.1$：activation threshold，suppress noise
- 第三行是hysteresis：低于threshold但非零时保持原状态，避免抖动

$$o_j = \sum_{t=1}^T \max(0, s_t - s_{t-1})$$

每个0→1 transition计一次onset。

$$\bar{o}_j = \frac{1}{|E_j^+|} \sum_{e \in E_j^+} o_j^{(e)}$$

**Intuition**: General feature（如"grasp event"）在一个episode内会bursty地fire多次——每次grasp一次onset。Memorized feature是sustained activation，once on, stays on，$\bar{o} \approx 1$。这个metric decouples from coverage，单独capture burstiness。

### 3.3 Mean Activation Magnitude $\bar{a}_j$ (Eq. 8)

$$\bar{a}_j = \frac{1}{|E_j^+|} \sum_{e \in E_j^+} \max_t f_j(\mathbf{x}_t^{(e)})$$

Per-episode peak activation的均值。Capture典型firing intensity。

### 3.4 Relative Run Length $\bar{\ell}_{r,j}$ (Eq. 9-10)

$$r_j = \frac{1}{o_j} \sum_{t=1}^T s_t$$

每个onset的平均持续timestep数。

$$\bar{\ell}_{r,j} = \frac{1}{|E_j^+|} \sum_{e \in E_j^+} \frac{r_j^{(e)}}{T^{(e)}}$$

Normalize by episode length得到fraction。接近0 → transient/bursty；接近1 → sustained across全episode。

**关键对比**: $\bar{o}$和$\bar{\ell}_r$是**互补**的——onset count capture firing pattern的"事件性"，run length capture firing的"持续性"。Memorized feature两者都低/高（$\bar{o}$低，$\bar{\ell}_r$高），general feature反之。

---

## 4. Logistic Regression Classifier (Eq. 11)

$$P(\mathrm{general} \mid \mathbf{m}) = \sigma(\beta_0 + \beta_1 \bar{o} + \beta_2 c + \beta_3 \bar{a} + \beta_4 \bar{\ell}_r)$$

用30个manually labeled features（15 general, 15 memorized）训练。LOO-CV accuracy: LIBERO 100%, DROID 96.7%。

### LIBERO classifier coefficients:
$\beta_0 = -4.20, \beta_1 = 1.89, \beta_2 = 1.80, \beta_3 = 0.52, \beta_4 = -0.36$

- $\beta_1 \approx \beta_2$：onset count和coverage贡献相当。LIBERO场景少（~20个visual scenes），burstiness和breadth都需要。
- $\beta_4 < 0$：sustained activation → memorized

### DROID classifier coefficients:
$\beta_0 = -1.78, \beta_1 = 0.74, \beta_2 = 2.36, \beta_3 = 0.35, \beta_4 = -1.04$

- $\beta_2$是$\beta_1$的3.2×：DROID有1545个unique tasks vs LIBERO的130，episode coverage成为dominant signal
- $\beta_4 = -1.04$：比LIBERO的$-0.36$强3×。DROID episodes更长更variable（$\mu=283.5, \sigma=219.2, CV=0.77$ vs LIBERO $\mu=161.5, \sigma=68.2, CV=0.42$），run length是更强differentiator

**Intuition**: 同样的metric在不同dataset上的discriminative power完全不同。这暴露了一个deep issue——generality是dataset-relative的concept。在低diversity dataset上burstiness是key，在高diversity dataset上breadth是key。

---

## 5. 关键实验结果

### 5.1 Feature Interpretability (Table 1)

采样120个features across 6 model/layer组合，79.17% interpretable。PaliGemma layer 5在LIBERO上90%，layer 11降到80%。Early-middle layer最interpretable。

### 5.2 Memorization dominates (Table 2)

| Model | Layer | % Memorized |
|-------|-------|-------------|
| $\pi_{0.5}$-LIBERO PG5 | 2044 features | 98.43% |
| $\pi_{0.5}$-LIBERO PG avg | 7175 features | 97.37% |
| $\pi_{0.5}$-DROID PG5 | 2046 features | 94.92% |
| $\pi_{0.5}$-DROID PG avg | 6649 features | 89.19% |
| OpenVLA-LIBERO Goal L8 | 1775 features | 99.55% |
| OpenVLA LM avg | 9389 features | 99.55% |

**Critical finding**: 即使是SOTA的$\pi_{0.5}$在DROID上fine-tune，仍有~89-95% features是memorization。OpenVLA在LIBERO Goal上99.55% memorized——基本上整个model在memorize 428个episodes。

Trend: LIBERO Goal (99.55%) → full LIBERO (97.37%) → DROID (89.19%)。Dataset size/diversity增加 → general feature比例增加。这给"more diverse data helps generalization"提供了mechanistic evidence。

### 5.3 Specific General Features

**LIBERO PG5的4个highlighted general features**（episode coverage > 0.99）:

1. **F1129 (grasp/place)**: $P(\mathrm{general})=0.91$. Initial grasp和placement时fire，onset count随pick-and-place sub-goals数量scale（single-object task→2 onsets, two-object task→4 onsets）。Compositional structure！

2. **F1902 (transport)**: $P(\mathrm{general})=0.89$. 在F1129的两个onset之间fire，对应carrying phase。Magnitude随接近goal position线性增加——encode proximity to placement target。

3. **F128 (pre-grasp alignment)**: $P(\mathrm{general})=0.92$. End-effector位于object正上方时fire。Magnitude随object进入wrist camera中心ramp up。Post-placement如果arm回到vertical alignment会reactivate。

4. **F445 (task completion)**: $P(\mathrm{general})=0.58$. End-effector接近goal location时fire，predominantly在final sub-goal。Compound task中magnitude更低且prefer second object placement——encode overall task success。

这4个features构成了一个**hierarchical motion primitive decomposition**: pre-grasp → grasp/place → transport → task completion。每个feature的onset timing精确锁在sensorimotor event上。

**DROID PG5的4个general features**:

1. **F158 (sub-task checkpoint)**: $P(\mathrm{general})=0.91$. Highest $\bar{o}$ across所有SAEs。在8-bottle pick-and-place episode中fire 16次，每次bottle进入wrist camera或goal plate可见时onset。

2. **F586 (pinch grasp)**: $P(\mathrm{general})=0.78$. Precision grasp of thin objects（plate edges, sugar packets, utensil handles, towels）。对wider grasps以~0.5× magnitude fire——encode grip aperture as continuous signal。

3. **F165 (open gripper over target)**: $P(\mathrm{general})=0.83$. Target object在open gripper jaws之间可见时fire。

4. **F399 (grasp acquisition/placement)**: $P(\mathrm{general})=0.79$. Grasp closing phase和placement时fire，transport时deactivate。

### 5.4 Closed-Loop Steering (Eq. 12-13)

$$\mathbf{v}_i = \mathbf{W}_{\mathrm{dec}}[:, i] \in \mathbb{R}^d$$
$$\mathbf{y}' = \mathbf{y} + \alpha \cdot \mathbf{v}$$

Decoder column是unit-normalized，所以steering magnitude完全由scalar $\alpha$控制。Steering vector broadcast到所有token positions，在每个denoising step都perturb。

**F128 steering**: 跨3个tasks（move black bowl to plate, turn on stove + put moka pot, pick alphabet soup to basket）。Steer后robot consistently接近target grasp object然后**hover above it**而non grasp。完全consistent with pre-grasp alignment interpretation。

**F1902 steering**: 跨3个tasks。Robot完全**skip object grasp**直接move到goal location。在alphabet soup task中甚至collide with basket把它push开。

**Robustness observation**: 即使high magnitude steering，policy仍保持goal-directed behavior，attempt approach/grasp/pursue alternative subtasks，non degenerate random motion。这说明VLA的computation对single-feature perturbation有相当的redundancy。

### 5.5 Knowledge Insulation (Figure 6)

Training step增加 → episode coverage下降，relative run length上升 → generality退化。**Knowledge Insulation (KI)** (arXiv:2505.23705, https://arxiv.org/abs/2505.23705) reverse这个trend。

P(general) median across DROID PG5 features:
- AE 10k steps: 0.190
- AE 30k steps: 0.181
- AE 60k steps: 0.179
- AE 90k steps: 0.181
- KI: 0.206

KI用stop gradient和architectural separation让VLM backbone保留semantic capability，expert handle continuous control。这里给出mechanistic evidence：KI确实在feature level shift distribution toward generality。

---

## 6. Per-Token SAEs (Section A.7)

Mean-pooled tokens是compromise——storage限制（per-token在DROID 2000 episodes需要~3.5TB per layer）。他们也做了per-token experiments，把每个image的256 patches sum成1个vector。

Per-token SAEs less interpretable than summed-token counterparts，但有一些promising features：

- **F1881**: 主camera token上strong uniform activation，wrist camera sparse lower magnitude。Noun "pot"和modifier "black"几乎identical activation patterns——link co-referent tokens。
- **F225**: 纯textual semantic structure。Camera token零激活，noun上强激活，verbs不激活。
- **F1659**: Memorized feature，只在含lid/pot的episode fire，"pot" token最高activation。

Per-token general features少（<10 per layer），且top-activating text tokens常常是function words（"the", "and"），non semantically informative content words。这暗示visual-textual alignment在VLA里**less crisp** than in pure VLMs。

---

## 7. Limitations & Open Problems

1. **Top activation ≠ steerability**: 很多clean features有limited/unpredictable causal impact。Hypothesis: flow matching的nonlinear downstream interactions，或predictive ≠ causal。

2. **Binary classification过简化**: 
   - F1939 (LIBERO PG5): $\bar{o}=1.00, c=0.732$ — fire in first 20 timesteps regardless of scene/task，encode robot "home" position。Single-onset-per-episode导致classifier误判为memorized。
   - F1381 (DROID PG5): $\bar{o}=1.00, c=0.226, \bar{\ell}_r=0.990$ — lid grasp feature，在86% lid-related episode fire，但lid episode只占6.7% dataset。Coverage低于decision boundary。
   
   这两个case暴露structural limitation：**dataset-diversity-aware normalization of episode coverage**是需要的。

3. **Steering随layer depth衰减** (Figure 11): Action Expert layer 0最steerable，deeper layers急剧衰减。PaliGemma layer 17产生**exactly zero** displacement——essentially frozen representation。

4. **Hardware evaluation缺失**: 只有simulation closed-loop，无real robot验证。

---

## 8. 对你的直觉build的几个关键insights

### 8.1 VLA的"成功"可能主要是memorization
即使在$\pi_{0.5}$这种SOTA + DROID这种"large" dataset上，90%+ features仍是memorization。OpenVLA在LIBERO Goal上99.55% memorized——428 episodes基本被rote learn。这给LIBERO-PRO的behavioral finding提供了mechanistic correlate：behavioral brittleness源于feature-level memorization dominance。

### 8.2 General features确实emerge，且compositional
F1129的onset count随sub-goal数量scale，F1902在F1129 onset pair之间fire——这是**compositional temporal structure**。说明VLA内部确实学到了hierarchical motion primitives，just被memorization features heavily outnumbered。

### 8.3 Dataset diversity是generality的必要条件
LIBERO Goal → full LIBERO → DROID的memorization比例单调下降。但即使DROID (1545 tasks)仍89% memorized。Robotics dataset相比LLM pretraining data小太多数量级。**Scale matters but robotics is far from the regime where generalization dominates**。

### 8.4 SAE metrics可作training-time proxy
无需rollout就能监控generality。Episode coverage下降 + run length上升 = forgetting。KI能reverse这个趋势。这给training-time diagnostic提供了cheap signal。

### 8.5 Steering的layer-dependence暗示representation冻结
PG L17 zero displacement是striking。暗示high-level semantic representations在fine-tuning中被"locked"，只有early action expert layers保持plasticity。这与KI的设计意图一致——protect VLM backbone。

---

## 9. 与你工作的潜在connection

你在policy gradient、optimization、memorization方面的工作与这篇paper的finding有deep connection。LIBERO-PRO的behavioral perturbation发现 → 这篇paper的feature-level memorization evidence，两者构成**behavioral ↔ mechanistic**的dual characterization。

几个可能的延伸方向：
- 用SAE metrics作为regularization target，训练时直接penalize memorization features的activation
- 在RL fine-tuning中reward general features、ablate memorized features
- Per-token SAEs + attention pattern analysis来理解visual-textual grounding
- Real robot上的steering evaluation——sim的robustness可能在real上不hold

References:
- 项目主页: https://drvla.github.io
- $\pi_0$ paper: https://arxiv.org/abs/2410.24164
- $\pi_{0.5}$ paper: https://arxiv.org/abs/2504.16054
- OpenVLA: https://proceedings.mlr.press/v270/kim25c.html
- LIBERO: https://arxiv.org/abs/2306.03310 (NeurIPS 2023)
- LIBERO-PRO: https://arxiv.org/abs/2510.03827
- Knowledge Insulation: https://arxiv.org/abs/2505.23705
- DROID: https://arxiv.org/abs/2403.12945
- Scaling SAEs (Gao et al.): https://arxiv.org/abs/2406.04093
- Towards Monosemanticity (Bricken et al.): https://transformer-circuits.pub/2023/monosemantic-features
- Scaling Monosemanticity (Templeton et al.): https://transformer-circuits.pub/2024/scaling-monosemanticity/
- Toy Models of Superposition: https://arxiv.org/abs/2209.10652
- SAEs find interpretable features (Cunningham et al.): https://arxiv.org/abs/2309.08600
- Winner-take-all autoencoders: https://arxiv.org/abs/1409.2752
- SAEs in VLMs: https://arxiv.org/abs/2504.02821
- PaliGemma: https://arxiv.org/abs/2407.07726
- RT-2: https://proceedings.mlr.press/v229/zitkovich23a.html
