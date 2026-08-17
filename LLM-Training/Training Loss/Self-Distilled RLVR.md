---
source_pdf: Self-Distilled RLVR.pdf
paper_sha256: 2fefb889decef0baa61b9bccd41222df904987db8147a1519a4dd19d60a12f71
processed_at: '2026-08-12T04:45:35-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 RLSD

## 一句话总结

**OPSD 这条路本质上走错了——拿 teacher 当"模仿目标"会泄露答案；RLSD 的思路是：teacher 只告诉你"这个 token 该奖励多少"，但"该奖励还是惩罚"由 environment reward 说了算。**

---

## 一、背景：现在的训练范式都各有毛病

### GRPO（DeepSeek 那套）

给模型一道数学题，让它生成 G 个回答，用 verifier 打分（对/错），算 group-relative advantage。

**问题**：一道题生成几百个 token，但只有一个 scalar reward。哪个 token 是关键推导？哪个 token 是废话填充？模型分不清——所有 token 拿同样的 advantage。这就是 "credit assignment problem"。

### OPD（用大 teacher）

让一个更大的 teacher model 评估 student 的 trajectory，给每个 token 提供 dense 的 logits 信号。

**问题**：要养一个大的 teacher，贵；而且 teacher 和 student 必须 share vocabulary，scalability 差。

### OPSD（自己做 teacher）

同一个 model 当 teacher 和 student。Teacher 多看一点 "privileged information"（比如参考答案、verified reasoning trace），student 只看 question。

**看起来很美**：不用额外大模型，dense 信号，token efficient。

**实际**：前 20 步猛涨，之后持续退化，模型开始 hallucinate 它根本看不到的"参考答案"。

---

## 二、OPSD 为什么会崩？

### 直觉解释

想象老师考试时偷看了答案，然后教学生。老师知道答案说 "No"，于是老师会在每个 token 上倾向于 "导向 No" 的表达。学生没看答案，但被强迫模仿老师的分布。

学生学会的是 "如何装作知道答案"，而不是 "如何真正推导答案"。训练越久，学生越会把 question 和 answer 的统计关联 encode 到参数里——这就是 leakage。

### 数学上的根因

核心定理：

$$\mathcal{L}_{\text{OPSD}} = \mathcal{L}^* + I(Y_t; R | X, Y_{<t})$$

翻译成人话：**OPSD 的 loss = 理想 loss + 一项 irreducible 的互信息 gap**。

这个 gap 就是 "teacher 因为看了 r 而对 token 预测更有信心" 这部分信息。**Student 永远无法消除它**，因为 student 架构上就不能看 r。无论 student 怎么优化，这个 gap 都是 strictly positive 的常数。

更糟糕的是，在 per-sample gradient 层面：

$$g(\theta; r) = g^*(\theta) + \delta(\theta; r)$$

- $g^*$：有益的 marginal matching 部分
- $\delta$：r-specific deviation，方差正比于那个 irreducible gap

**关键**：$\delta$ 的期望为零，但 SGD/Adam 是 path-dependent 的，零均值的扰动不会自动 cancel。训练早期 $g^*$ 主导（猛涨），后期 $g^*$ 消失但 $\delta$ 还在（因为它的方差与 $\theta$ 无关），于是 $\delta$ 的累积把参数往 "encode $x \to r$ correlation" 的方向推。

### Leakage Bandwidth 实验

作者设计了三个变体验证理论预测：
- Full OPSD：整个 vocabulary 都受影响
- Teacher's Top-1：只对 teacher 最喜欢的 token —— 泄漏最严重
- Student's Top-1：只对 student 最喜欢的 token —— 泄漏最轻

**三种都泄漏**，只是程度不同。因为只要 $P_T(\cdot|r)$ 进入 gradient direction，泄漏就不可避免。

### Impossibility Trilemma

在 shared parameters 设定下，三个性质不可兼得：
- (a) Objective stability（teacher 不漂移）
- (b) Sustained improvement（信号不消失）
- (c) Leakage-free training（参数不被 $\delta$ 带偏）

Frozen teacher 满足 (a) 但违反 (b)（容量被初始 checkpoint 锁死）。Online teacher 满足 (b) 但违反 (a)（teacher 漂移方向不可控）。而且无论哪种都违反 (c)。还有 self-reinforcing feedback loop：$\delta$ 让参数更 r-predictive → teacher 更会用 r → $\delta$ 方差更大 → ... 最终 collapse。

---

## 三、RLSD 怎么解决？

### Core Insight

Direction 和 magnitude 的需求是不对称的：
- **Direction 必须可靠**：一旦错了直接毁掉 policy
- **Magnitude 越细越好**：但即便有点 noise 也能容忍

所以：**environment reward 决定方向，teacher 决定幅度**。

### 三个步骤

**Step 1：算 privileged information gain**

$$\Delta_t = \text{sg}\big(\log P_T(y_t) - \log P_S(y_t)\big)$$

同一个 model 两次 forward：一次只看 question，一次看 question + 答案。两个 log-prob 的差就是 "r 对这个 token 的 marginal 贡献"。Stop-gradient 保证这只是个标量信号。

**Step 2：Direction-aware reweighting**

$$w_t = \exp(\text{sign}(A) \cdot \Delta_t) = \left(\frac{P_T(y_t)}{P_S(y_t)}\right)^{\text{sign}(A)}$$

关键设计：
- $A > 0$（做对了）：$w_t = P_T/P_S$，r 支持的 token 拿更大 credit
- $A < 0$（做错了）：$w_t = P_S/P_T$，倒过来，r 反对的 token 拿更大 blame

因为 $\exp(\cdot) > 0$，$w_t$ 永远为正，**advantage 的符号不会被翻转**。Environment reward 对 direction 有 exclusive authority。

**Step 3：Clipping + Mixing**

$$\hat{A}_t = A \cdot \text{clip}(w_t, 1-\epsilon_w, 1+\epsilon_w)$$

像 PPO 那样 clip，防止单个 token 影响过大。再线性插值 uniform advantage 和 reweighted advantage（$\lambda$ 从 0.5 decay 到 0），早期 dense 信号加速收敛，后期退化为 vanilla GRPO，sustained improvement。

### 为什么这样就不 leak？

三个层面的 isolation：

1. **Directional isolation**：$\text{sign}(\hat{A}_t) = \text{sign}(A)$，r 永远不能改变 gradient 方向，只能调 magnitude
2. **Support isolation**：gradient 只作用于 student 自己采样的 token，teacher 因 r 而 "喜欢" 的那些不在 support 里的 token，gradient 为零
3. **Magnitude boundedness**：$w_t$ 被 clip 到 $[1-\epsilon_w, 1+\epsilon_w]$ 且被 $\lambda$ 衰减，bound 住影响

对比 OPSD 的 per-sample gradient $\sum_v P_T(v|r) \nabla \log P_S(v)$——它跨整个 vocabulary，teacher 因 r 喜欢的 token 全部接收 gradient contribution，主动把 unseen privileged patterns 拉进参数更新。RLSD 完全消除了这条通道。

### Bayesian Interpretation

**Theorem 4**：$w_t = P_T(y_t)/P_S(y_t) = P(r|x, y_{\leq t})/P(r|x, y_{<t})$

这就是 Bayesian belief update ratio：生成 $y_t$ 之前对 r 的 posterior belief，除以之后——这个 token 让你更相信还是更不相信正确答案 r。

- $w_t > 1$：positive evidence（比如解 $2x+3=7$ 写 "2x=4"）
- $w_t < 1$：negative evidence（比如写 "5x=7"）
- $w_t = 1$：informationally neutral（比如 "therefore"）

Telescoping 性质：$\prod_t w_t = P(r|x,y)/P(r|x)$，sequence-level 的 total belief update 分解到每个 token。这就是把 sequence-level credit assignment 从 GRPO 的 coarse 颗粒度 elevate 到 token level。

### 同一个数学量，不同命运

最 striking 的 insight 在 Appendix A.5.6：**OPSD gradient 里的 importance weight $P_T(v|r)/P_S(v)$ 和 RLSD 的 evidence ratio $w_t$ 数学上完全一样**。

差别只在于**怎么用**：
- OPSD：作为 gradient weight，跨整个 vocabulary，drive student 模仿 teacher 分布的 shape——leakage 的源头
- RLSD：作为 stop-gradient 的 scalar multiplier，只作用于 student on-policy 采样的 token——precise credit attribution 工具

**同一个量，从泄漏源变成 attribution 工具，关键在用法的转变**。

---

## 四、实验结果

### Main Results（Table 2）

Qwen3-VL-8B-Instruct 上五个 multimodal reasoning benchmarks：

| Method | Avg |
|---|---|
| Base LLM | 51.49 |
| GRPO | 53.86 |
| OPSD | 52.49 |
| SDPO | 52.74 |
| GRPO+OPSD | 52.91 |
| **RLSD** | **56.18** |

RLSD vs Base +4.69%，vs GRPO +2.32%，vs OPSD +3.69%。

**关键观察**：
- OPSD 比 GRPO 还差——验证理论分析，leakage 导致 long-term degradation
- GRPO+OPSD（简单线性组合）也差——bounded reward 和 unbounded KL loss 的 scale mismatch，训练不稳定
- RLSD 通过 multiplicative modulation + strict sign preservation 避开了这个坑

### Training Dynamics（Figure 5）

- **Reward**：RLSD 初期上升更陡，天花板更高，避免了 OPSD 的后期 collapse
- **Entropy**：GRPO 熵快速塌缩（uniform reward 压制所有 token），RLSD 保持更高 entropy（selectively strengthen 关键 token，不均匀压制）
- **Clip ratio**：稳定在 3%-6%，证明 clipping 机制确实 engage 了

### Case Study（Figure 6）

做对的 trajectory：credit 集中在真正决定正确性的 token（识别关键立方体 + 最终减法），废话 token（"Looking at the image, I see..."）被 downweight。

做错的 trajectory：blame 集中在误读的关系 "3x=28.5" 和推导出的错误答案 "x=9.5"，中性的 setup token 拿较小惩罚。

---

## 五、直觉构建

### 类比：开车

- **Direction = 方向盘**：必须可靠，错了直接撞墙
- **Magnitude = 油门**：越精细越好，有点 noise 能容忍

Environment reward 是方向盘（sparse 但 reliable），teacher evidence ratio 是油门（dense 但只调 magnitude）。

### 类比：考试辅导

- **OPSD**：老师偷看答案后，让学生逐字模仿自己说的每句话——学生学到的是"装作知道"
- **RLSD**：老师批改时，在每个 token 上标"这一步对推导有帮助/没帮助/有害"，但**对错的最终判断由 verifier 决定**，老师只调整每一步的权重——学生学到的是"什么是有价值的推导"

### 为什么 OPSD 的理论分析这么 elegant？

Theorem 1 的分解 $\mathcal{L}_{\text{OPSD}} = \mathcal{L}^* + I(Y_t; R|X, Y_{<t})$ 在结构上类似 ELBO 分解 $\log p(x) = \text{ELBO} + \text{KL}(q(z|x)\|p(z|x))$。两者都揭示一个 **irreducible gap**：VI 中是 approximating family 与 true posterior 的 gap，OPSD 中是 student（不 condition r）与 teacher（condition r）的 marginal vs conditional gap。

### 为什么 stop-gradient + scalar modulation 这套 pattern 能 work？

四个机制联合：
1. **Stop-gradient**：teacher 信号不进入反向传播路径
2. **Scalar modulation**：只调 magnitude，不进 direction
3. **Clipping**：bound 单 token 影响
4. **On-policy sampling**：gradient 只作用于 student 自己采样的 token

这是一个**general design pattern**——任何想利用 privileged information 但不让它 contaminate optimization direction 的场景都可以 apply。比如 tool use in reasoning、multi-step planning、multi-agent cooperation。

---

## 六、核心 Take-aways

1. **OPSD 失败的根因**：information asymmetry 下 distribution matching 产生 irreducible mutual information gap，这个 gap 在 path-dependent optimizer 中累积，把参数往 encode $x \to r$ correlation 的方向推

2. **RLSD 的 core design**：decouple direction（environment reward）和 magnitude（teacher evidence ratio），通过 stop-gradient + scalar modulation + clipping + on-policy sampling 四个机制 structurally guarantee leakage-free

3. **Bayesian interpretation**：evidence ratio $P_T/P_S$ = sequential belief update ratio，提供 principled 的 token-level credit assignment

4. **实验验证**：5 个 multimodal reasoning benchmarks 上 best average accuracy（56.18%），比 Base +4.69%，比 GRPO +2.32%，比 OPSD +3.69%

5. **Trilemma resolution**：RLSD 同时满足 objective stability、sustained improvement、leakage-free training——OPSD 下 impossible

6. **最 elegant 的 insight**：**同一个数学量（evidence ratio），在 distribution matching 中是 leakage source，在 credit assignment 中是 principled tool，关键在 usage 的转变**。从 gradient weight 到 scalar multiplier，从 "逼你模仿" 到 "告诉你的每一步有多重要"。

---

## 七、开放思考

1. **推广到其他 self-referential optimization**：Impossibility trilemma 的 framework 可能可以推广到 GAN、self-training、iterative DPO 等任何 self-referential system with shared parameters

2. **Connection 到 VI / IB / GFlowNet**：$I(Y_t; R|X, Y_{<t})$ 作为 irreducible gap 与 Information Bottleneck 的 $I(X;Z) - \beta I(Z;Y)$ 有 structure similarity；evidence ratio 与 GFlowNet 的 reward signal 有 conceptual 相似性

3. **Multi-teacher ensemble**：多个 privileged contexts 的 evidence ratio aggregation

4. **General design pattern**：stop-gradient + scalar modulation + clipping 作为利用 privileged info 的通用 pattern，可能 apply 到 tool use、multi-step planning、multi-agent cooperation

5. **Curriculum 的 adaptive scheduling**：目前 $\lambda$ 是 linear decay，如果基于 student policy quality adaptive 调整可能更好

总的来说：**这篇 paper 的优雅之处在于，从 "为什么 OPSD 会失败" 的诊断出发，推导出一个 "direction 和 magnitude 的 asymmetric 需求" 的 deep insight，然后设计出 RLSD——同一个数学量，换个用法，从 leakage 源变成 attribution 工具**。理论分析、method design、experimental validation 三者逻辑链非常 clean。

---

# Self-Distilled RLVR 深度讲解

## 一、整体定位与核心问题

这篇paper处理的是post-training LLM的一个长期矛盾：**RLVR提供sparse但reliable的environment signal vs OPD/OPSD提供dense但可能problematic的teacher signal**。核心claim可以总结为一句话：**在self-distillation setting下，teacher不应该作为distribution matching target，应该作为credit assignment的magnitude modulator，direction则由environment reward锚定**。

这呼应了最近MIMO-v2-Flash [6] 用MOPD (Mixing OPD with RLVR)、Thinking Machines Lab的OPD blog (https://thinkingmachines.ai/blog/on-policy-distillation) [5]、Kimi K2.5 [3] 等工作都在探索dense token-level signal的方向。RLSD提供了一个无需external teacher的principled alternative。

参考链接：
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- On-Policy Distillation (TML blog): https://thinkingmachines.ai/blog/on-policy-distillation
- MIMO-v2-Flash: https://arxiv.org/abs/2601.02780

---

## 二、背景：GRPO、OPD、OPSD的三角关系

### 2.1 GRPO及其sparse signal problem

GRPO对每个question $x$ 采 $G$ 个responses $\{y^{(1)}, \dots, y^{(G)}\}$，然后基于verifier reward计算sequence-level advantage。**公式(1)**：

$$A^{(i)} = \frac{R(x, y^{(i)}) - \mu_G}{\sigma_G}$$

变量含义：
- $A^{(i)}$：第 $i$ 个response的sequence-level advantage
- $R(x, y^{(i)}) \in \{0,1\}$：verifier给出的binary reward
- $\mu_G$：group内 $G$ 个reward的mean
- $\sigma_G$：group内reward的standard deviation

**公式(2)**的clipped surrogate objective：

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y^{(i)}|}\sum_{t=1}^{|y^{(i)}|}\min\Bigl(\rho_t^{(i)} A^{(i)}, \text{clip}\bigl(\rho_t^{(i)}, 1-\epsilon, 1+\epsilon\bigr) A^{(i)}\Bigr)\right]$$

变量含义：
- $\rho_t^{(i)} = \pi_\theta(y_t^{(i)}|x,y_{<t}^{(i)}) / \pi_{\theta_{\text{old}}}(y_t^{(i)}|x,y_{<t}^{(i)})$：current policy与old policy的importance sampling ratio
- $\epsilon$：clip bound（实验里 $\epsilon_{\text{low}}=0.2, \epsilon_{\text{high}}=0.28$）
- $|y^{(i)}|$：第 $i$ 个trajectory的token长度

**关键问题**：response内所有token共享同一个advantage $A^{(i)}$，没有token-level discrimination。reasoning trajectory中真正决定正确性的关键token（如推导某步公式）与stylistic filler（如"Looking at the image, I see..."）获得identical credit。这就是GRPO的credit assignment bottleneck，也是后续entropy collapse的根源。

### 2.2 OPD vs OPSD的information symmetry/asymmetry

OPD (On-Policy Distillation) [4, 5] 用一个**独立的更大teacher model** $\pi_{\hat{\theta}}$ 评估student的on-policy trajectory，提供dense token-level logits。关键在teacher和student看到**同样的input $x$**（information-symmetric）。

OPSD (On-Policy Self-Distillation) [7, 8] 用**同一个model**充当teacher和student，但teacher额外看到privileged information $r$（如verified reasoning trace或final ground-truth answer）。这就产生了**information asymmetry**——teacher看到了student看不到的 $r$。

学生和teacher分布的形式化定义（**公式3-5**）：
- Student: $P_S(\cdot|y_{<t}) \triangleq \pi_\theta(\cdot|x, y_{<t})$
- OPD Teacher: $P_T(\cdot|y_{<t}) \triangleq \pi_{\hat{\theta}}(\cdot|x, y_{<t})$
- OPSD Teacher: $P_T(\cdot|y_{<t}) \triangleq \pi_\theta(\cdot|x, r, y_{<t})$

共享objective（**公式6**）：

$$\mathcal{L}_{\text{OP(S)D}}(\theta) = \mathbb{E}_{(x,r)\sim S}\mathbb{E}_{\hat{y}\sim P_S(\cdot|x)}\left[\frac{1}{|\hat{y}|}\sum_{t=1}^{|\hat{y}|}D(P_T \| P_S)\right]$$

变量含义：
- $S = \{(x_i, r_i)\}_{i=1}^N$：training dataset
- $\hat{y}$：student on-policy采样得到的trajectory
- $D$：divergence measure，如generalized Jensen-Shannon divergence
- 关键：gradient只通过 $P_S$ backprop，$P_T$ 作为fixed target

参考：
- On-Policy Distillation: https://openreview.net/forum?id=3zKtaqxLhW
- Self-Distilled Reasoner: https://arxiv.org/abs/2601.18734
- RL via Self-Distillation: https://arxiv.org/abs/2601.20802

### 2.3 OPSD的三个failure现象（Figure 2, 3）

**Phenomenon 1: Privileged information leakage**。OPSD-trained model在inference时explicitly reference它在training时看过、但inference时无法access的"reference solution"。Figure 2给了一个example："...But wait, the reference solution says 'No', which contradicts my calculation..."

**Phenomenon 2: Performance degradation**。Figure 3(b)显示validation accuracy在前10-20步达到peak后持续下降，与leakage frequency的单调递增同步（Figure 3(a)）。

**Phenomenon 3: KL divergence stagnation**。Figure 3(c)显示OPSD的teacher-student KL divergence在first few steps下降后plateau在接近initial value的水平，而OPD的KL divergence在整个training过程中steady下降。这预示着存在某种**irreducible gap**。

---

## 三、理论分析：为什么OPSD会fail

### 3.1 Theorem 1: Irreducible Mutual Information Gap

**Setup**：$r \sim P(r|x)$，由于同一问题 $x$ 可能有多个semantically valid reasoning paths，$P(r|x)$ 是非degenerate分布，entropy非零。即便每个training instance $x_i$ 只paired一个reference trace $r_i$，从student的epistemic perspective（既不能observe $r$ 也不能从 $x$ deterministically derive $r$），privileged information仍是不确定latent variable。

**Optimal student policy**应该recover teacher的marginal distribution（**公式7**）：

$$P_S^*(y_t|x, y_{<t}) = \mathbb{E}_{r\sim P(r|x, y_{<t})}[P_T(y_t|x, r, y_{<t})]$$

变量含义：
- $P_S^*$：optimal student distribution（不condition on $r$）
- $P(r|x, y_{<t})$：给定 $x$ 和已生成tokens $y_{<t}$ 时 $r$ 的posterior distribution
- $P_T(y_t|x, r, y_{<t})$：teacher（看 $r$）的conditional distribution

定义marginal teacher distribution：$\bar{P}_T(y_t) \triangleq \mathbb{E}_r[P_T(y_t|x, r, y_{<t})]$

**Ideal objective**（**公式8**）：

$$\mathcal{L}^*(\theta) = \mathbb{E}_x\Big[D_{\text{KL}}\bigl(\bar{P}_T(\cdot) \big\| P_S(\cdot|x)\bigr)\Big]$$

但OPSD实际强制的是**per-sample matching**（**公式9**）：

$$\mathcal{L}_{\text{OPSD}}(\theta) = \mathbb{E}_x\mathbb{E}_{r\sim P(r|x)}\bigl[D_{\text{KL}}\bigl(P_T(\cdot|x, r) \| P_S(\cdot|x)\bigr)\bigr]$$

这相当于让一个**不condition on $r$ 的 $P_S$**去match一个**condition on $r$ 的 $P_T(\cdot|x,r)$**，是fundamentally ill-posed的。

**Theorem 1 (KL Decomposition)**：

$$\mathcal{L}_{\text{OPSD}} = \mathcal{L}^* + I(Y_t; R | X, Y_{<t})$$

变量含义：
- $I(Y_t; R | X, Y_{<t})$：在给定 $X, Y_{<t}$ 条件下，current token $Y_t$ 与privileged information $R$ 的条件互信息
- 衡量teacher的token-level prediction在多大程度上依赖于privileged information

**证明思路**（Appendix A.1）：将 $\bar{P}_T(v)/\bar{P}_T(v)$ 插入logarithm：

$$\mathcal{L}_{\text{OPSD}} = \mathbb{E}_r\left[\sum_v P_T(v|r)\log\frac{P_T(v|r)}{\bar{P}_T(v)}\right] + \mathbb{E}_r\left[\sum_v P_T(v|r)\log\frac{\bar{P}_T(v)}{P_S(v)}\right]$$

第一项 $= \mathbb{E}_r[D_{\text{KL}}(P_T(\cdot|r) \| \bar{P}_T)] = I(Y_t; R|X, Y_{<t})$（条件互信息定义）。

第二项中 $\log(\bar{P}_T(v)/P_S(v))$ 与 $r$ 无关，所以 $\mathbb{E}_r$ 只作用于 $P_T(v|r)$：

$$\mathbb{E}_r\left[\sum_v P_T(v|r)\log\frac{\bar{P}_T(v)}{P_S(v)}\right] = \sum_v \bar{P}_T(v)\log\frac{\bar{P}_T(v)}{P_S(v)} = D_{\text{KL}}(\bar{P}_T \| P_S) = \mathcal{L}^*$$

**关键insight**：$I(Y_t; R | X, Y_{<t})$ **独立于 $\theta$**，完全由teacher的conditional distribution和 $P(r|x)$ 决定。Student的优化**无法消除这个gap**。Feasible set $\mathcal{F} = \{Q : Q(\cdot|x, y_{<t}) \text{ does not condition on } r\}$ 下的global optimum是 $P_S^* = \bar{P}_T$，此时residual loss恰好等于 $I(Y_t; R|X, Y_{<t}) > 0$，一个strictly positive的irreducible lower bound。

**这与Figure 3(c)的KL stagnation完全consistent**：student在前几步快速逼近 $\bar{P}_T$ 之后，residual $I(Y_t; R|X, Y_{<t}) > 0$ 无法通过legitimate optimization继续减小。更critical的是，这个irreducible residual持续给optimizer非零的loss signal，驱动model吸收有害noise到parameters中。

**Leakage的数学origin**：由于student架构无法directly condition on $r$，唯一pathway是encode $x \to r$ 的statistical correlation到 $\theta$ 里。这就是privileged information leakage。

对比OPD：external teacher的prediction不condition on student不可访问的privileged information，所以mutual information gap不出现，KL divergence稳步下降。

**Intuition构建**：可以把这个gap理解为"teacher因为看了r所以对Y_t更有信心"这部分信息——这部分信心对student来说永远是latent。这与Variational Inference中ELBO的KL项结构相似，也与Information Bottleneck的 $I(X;Z) - \beta I(Z;Y)$ 有形式上类比。

### 3.2 Proposition 1: Per-Sample Gradient Decomposition

Theorem 1的 $I(Y_t; R|X)$ $\theta$-independent性可能让人觉得它不影响gradient。但这是**expected gradient**的视角。实际优化是在concrete samples $(x, r)$ 上进行。

**Benign expected gradient**：

$$\nabla_\theta \mathcal{L}_{\text{OPSD}} = \nabla_\theta \mathcal{L}^* = -\sum_v \bar{P}_T(v) \nabla_\theta \log P_S(v)$$

**Pathological per-sample gradients**（**公式11**）：

$$g(\theta; r) = -\sum_{v \in \mathcal{V}} P_T(v|r) \cdot \nabla_\theta \log P_S(v)$$

**Proposition 1 (Per-Sample Gradient Decomposition)**（**公式12**）：

$$g(\theta; r) = \underbrace{-\sum_v \bar{P}_T(v) \nabla_\theta \log P_S(v)}_{g^*(\theta): \text{marginal matching}} + \underbrace{-\sum_v [P_T(v|r) - \bar{P}_T(v)] \nabla_\theta \log P_S(v)}_{\delta(\theta; r): \text{r-specific deviation}}$$

两条性质：
- **(i)** $\mathbb{E}_r[\delta(\theta; r)] = 0$ — deviation在 $r$ 上zero-mean
- **(ii)** $\mathbb{E}_r[\|\delta(\theta; r)\|^2] = \sum_v \text{Var}_r[P_T(v|r)] \cdot \|\nabla_\theta \log P_S(v)\|^2$ — deviation variance正比于 $P_T$ 在 $r$ 上的conditional variance，而这正比于mutual information $I(Y_t; R|X, Y_{<t})$

证明思路（Appendix A.2）：在sum里add-subtract $\bar{P}_T(v)$。性质(i)由 $\mathbb{E}_r[P_T(v|r)] = \bar{P}_T(v)$ 直接得到。性质(ii)的diagonal项正好是variance weighted gradient norm；cross terms通过Cauchy-Schwarz bounded by同样的variances。当 $I(Y_t;R|X, Y_{<t})=0$ 时 $P_T(v|r) = \bar{P}_T(v)$ 对所有 $r$ 成立，$\delta \equiv 0$。

**Critical insight**：性质(i)的zero-mean**不意味着innocuous**。任何optimizer在individual samples或mini-batches上计算gradient（如SGD、Adam [12]）都是path-dependent的。**零均值的perturbation在非线性optimization中不会自动cancel over training**。

### 3.3 Two-phase training dynamics

Proposition 1的分解给出两阶段dynamics，与Figure 3(b)的曲线perfectly对应：

**Phase 1（早期）**：student $P_S$ 远离teacher marginal $\bar{P}_T$，beneficial component主导：$\|g^*(\theta)\| \gg \|\delta(\theta; r)\|$。Gradient主要驱动marginal matching，student快速获得general reasoning capability。对应Figure 3(b)前10-20步的steep rise。

**Phase 2（后期）**：$P_S$ 逼近 $\bar{P}_T$，beneficial component $\|g^*(\theta)\|$ 趋于0。但deviation component $\|\delta(\theta; r)\|$ **保持bounded away from zero**——因为其variance由 $I(Y_t; R|X, Y_{<t})$ 决定，而这**与 $\theta$ 无关，不随optimization progress衰减**。Parameter updates越来越被 $\delta$ 主导，path-dependent accumulation of these perturbations驱动model toward $x \to r$ correlations的parameter region，触发self-reinforcing degradation。

### 3.4 Leakage Bandwidth实验：三个ablation

作者设计了三个variant测试理论预测：**任何teacher's privileged evaluation $P_T(\cdot|r)$ 进入gradient direction的variant都会leak，无论distillation target如何压缩**。

**Variant (i) Full OPSD**：gradient over整个vocabulary $\mathcal{V}$：

$$g_t(\theta; r) = -\sum_{v \in \mathcal{V}} P_T(v|r) \nabla_\theta \log P_S(v)$$

每个token都receive $P_T(v|r)$-weighted的gradient contribution。Widest leakage bandwidth。

**Variant (ii) Teacher's Top-1**：collapse teacher到point mass at $v_T^* = \arg\max_v P_T(v|r)$：

$$g_t(\theta; r) = -\nabla_\theta \log P_S(v_T^*)$$

$v_T^*$ 完全由 $r$ 决定。Most concentrated leakage injection。

**Variant (iii) Student's Top-1**：target support限制在 $v_S^* = \arg\max_v P_S(v)$。Gradient weight正比于 $P_T(v_S^*|r)/P_S(v_S^*)$，仍然依赖 $r$。Narrowest bandwidth，但leakage persists。

实验结果（Figure 3a, 3b）confirm预测：
- 三种variant都exhibit monotonically increasing leakage
- Severity排序：Teacher's Top-1 > Full OPSD > Student's Top-1（与bandwidth理论prediction一致）

附录A.3给出formal proof：在所有三种variant中，$\hat{A}_t$ 都涉及 $P_T(\cdot|r)$，所以deviation component $\delta$ 都携带r-specific information进入parameter update direction。**Directional dependence on $r$ is irreducible**。

### 3.5 Theorem 3: Impossibility Trilemma

在shared parameter setting下（teacher和student共享同一个 $\theta$），三种desirable properties**不可能同时满足**：

- **(a) Objective stability**：optimization target在successive steps间不drift
- **(b) Sustained improvement**：distillation signal不vanish
- **(c) Leakage-free training**：deviation component不驱动parameter drift

**Strategy A: Frozen Teacher**（**公式27**）

$$\mathcal{L}_A(\theta_k) = \mathbb{E}_r\Big[D_{\text{KL}}\big(P_T^{\theta_0}(\cdot|r) \| P_S^{\theta_k}(\cdot|x)\big)\Big]$$

满足 **(a)**：target不drift（$\Delta_T = 0$）。但违反 **(b)**：student capacity ceiling到initial checkpoint $\theta_0$ 的quality，无进一步improvement空间（Proposition 2）。

**Strategy B: Online Teacher**（**公式28**）

$$\theta_{k+1} = \theta_k - \eta \nabla_\theta \mathcal{L}_k(\theta_k), \quad \mathcal{L}_k(\theta) \triangleq \mathbb{E}_r\big[D_{\text{KL}}(P_T^{\theta_k}(\cdot|r) \| P_S^\theta(\cdot|x))\big]$$

满足 **(b)**：teacher持续evolve提供non-vanishing signal。但违反 **(a)**：每步update同时改变 $P_T$ 和 $P_S$，**Theorem 2**给出：

$$\mathcal{L}(\theta_{k+1}) - \mathcal{L}(\theta_k) = \underbrace{\Delta_S}_{\leq 0 \text{ (student improvement)}} + \underbrace{\Delta_T}_{\text{sign uncontrolled (teacher drift)}}$$

$\Delta_S \leq 0$ 由gradient descent保证。但 $\Delta_T$ 的sign **uncontrolled**——asymmetry of KL divergence没有保证teacher drift的方向。

**Proposition 3: Self-Reinforcing Feedback Loop**：

定义model对privileged information的sensitivity：$S(\theta) \triangleq \mathbb{E}_r[D_{\text{KL}}(P_T^\theta(\cdot|r) \| \bar{P}_T^\theta)] = I(Y_t; R|X, Y_{<t})$。

Cycle：
1. Per-sample deviation $\delta(\theta; r)$ 驱动parameters toward r-predictive features
2. 这些features被encode在shared $\theta$，增强teacher利用 $r$ 的能力
3. $S(\theta_{k+1}) \geq S(\theta_k)$，放大 $\text{Var}_r[P_T(v|r)]$
4. 由Proposition 1(ii)，deviation variance增大，reinforce step (i)

定义 $\rho_k = \mathbb{E}_r[\|\delta(\theta_k; r)\|^2] / \mathbb{E}_r[\|g(\theta_k; r)\|^2]$ 为deviation fraction。Feedback loop驱动 $\rho_k$ 单调上升：早期 $\rho_k$ 小，后期 $\rho_k \to 1$，training collapse。

**Trilemma的resolution**：Hybrid strategies（如periodic teacher snapshots）仅interpolate between (a)和(b)。Property (c)始终不满足，因为underlying mutual information gap无论snapshot schedule如何都persists。

---

## 四、RLSD方法：Self-Distillation作为RLVR的Wingman

### 4.1 核心insight

前述分析pinpoint root cause：**distribution matching失败因为privileged information进入gradient direction，contaminating optimization trajectory**。

但evidence ratio $P_T(y_t)/P_S(y_t)$ 本身**也携带useful signal**：它衡量privileged information对每个token的belief revision程度。所以问题不是discard这个signal，而是改变它怎么被used。

**Key insight: direction和magnitude的需求asymmetric**：
- **Direction signal**：可以sparse但必须reliable——错误direction直接毁坏policy
- **Magnitude signal**：越dense越好——fine-grained discrimination among tokens

所以environment reward决定direction，teacher的privileged assessment决定magnitude。这就是RLSD的设计哲学。

### 4.2 Step 1: Privileged Information Gain

**公式(13)**：

$$\Delta_t = \mathbf{sg}\big(\log P_T(y_t) - \log P_S(y_t)\big)$$

变量含义：
- $\Delta_t$：token $t$ 处的privileged information gain
- $P_T(y_t) = \pi_\theta(y_t|x, r, y_{<t})$：teacher context（看 $r$）下 $y_t$ 的log-prob
- $P_S(y_t) = \pi_\theta(y_t|x, y_{<t})$：student context（仅看 $x$）下 $y_t$ 的log-prob
- $\mathbf{sg}$：stop-gradient operator

由于teacher和student share同一个model，$\Delta_t$ isolated了 $r$ 对prediction $y_t$ 的marginal contribution。$\Delta_t > 0$ 表示 $r$ 强烈支持这个token；$\Delta_t < 0$ 表示 $r$ 反对。

Stop-gradient确保 $\Delta_t$ 只作为weighting signal，不引入auxiliary gradient pathways——这是与OPSD的关键区别。

### 4.3 Step 2: Direction-aware Evidence Reweighting

**公式(14)**：

$$w_t = \exp(\text{sign}(A) \cdot \Delta_t) = \left(\frac{P_T(y_t)}{P_S(y_t)}\right)^{\text{sign}(A)}$$

变量含义：
- $w_t$：per-token evidence weight
- $A$：sequence-level advantage
- $\text{sign}(A) \in \{+1, -1\}$：direction-aware modulation

两种case：
- **$A > 0$（correct trajectory）**：$w_t = P_T/P_S$，$r$ 支持的token得到更大weight，concentrate positive credit到与correct reasoning trace对齐的tokens
- **$A < 0$（incorrect trajectory）**：$w_t = P_S/P_T$，ratio inverted，$r$ 反对的token得到更大blame，$r$ 支持的token得到attenuated punishment

由于 $\exp(\cdot) > 0$，$w_t$ 始终为正，**保证token-level advantage的sign不被flip**。Environment reward对direction有exclusive authority。

### 4.4 Bayesian Interpretation: Evidence Ratio作为Belief Update

**Theorem 4 (RLSD Weights as Belief Update Ratios)**（**公式34**）：

$$w_t = \frac{P_T(y_t|x, r, y_{<t})}{P_S(y_t|x, y_{<t})} = \frac{P(r|x, y_{\leq t})}{P(r|x, y_{<t})}$$

即 $w_t$ 等于在生成 $y_t$ **之后**对 $r$ 的posterior belief与生成**之前**的belief的比值——这是sequential Bayesian belief update的ratio。

**证明**（Appendix A.5）：用Bayes theorem on joint distribution $P(r, y_t|x, y_{<t})$：

$$P(y_t|x, r, y_{<t}) = \frac{P(r, y_t|x, y_{<t})}{P(r|x, y_{<t})} = \frac{P(r|x, y_{\leq t}) \cdot P(y_t|x, y_{<t})}{P(r|x, y_{<t})}$$

两边除以 $P(y_t|x, y_{<t})$ 并应用Assumption 1（model是consistent approximation of true conditional）即得。

**Assumption 1 (Consistent Conditional Approximation)**：
- $P_S(y_t|x, y_{<t}) \approx P(y_t|x, y_{<t})$ （model近似prior predictive）
- $P_T(y_t|x, r, y_{<t}) \approx P(y_t|x, r, y_{<t})$ （model近似posterior predictive）

当model capacity足够 + $r$ 通过in-context conditioning提供时assumption合理。

**Interpretation**：
- $w_t > 1$：生成 $y_t$ 增加对 $r$ 的belief，**positive evidence** for correct reasoning（例如求解 $2x+3=7$ 时写"2x=4"大大增加trajectory leads to $x=2$ 的belief）
- $w_t < 1$：生成 $y_t$ 减弱对 $r$ 的belief，**negative evidence**（如错误步骤"5x=7"会sharp降低belief）
- $w_t = 1$：informationally neutral，如formatting connectives（"therefore", "we have"）

**Telescoping identity**（**公式37**）：

$$\prod_{t=1}^T w_t = \prod_{t=1}^T \frac{P(r|x, y_{\leq t})}{P(r|x, y_{<t})} = \frac{P(r|x, y)}{P(r|x)}$$

右侧是sequence-level Bayesian evidence ratio——observing完整trajectory $y$ 后对 $r$ 的total belief update。Per-token weights $\{w_t\}_{t=1}^T$ 提供这个sequence-level evidence的fine-grained token-level decomposition。

**与OPSD的fundamental对比**：
- **OPSD performs behavioral cloning**：要求student replicate teacher的output distribution $P_T(\cdot|x, r)$。如果teacher因看了reference solution而favor某phrasing（如"according to the hint"），OPSD驱动student也采用这phrasing，即便它无reasoning value
- **RLSD performs logical credit attribution**：不要求student imitate teacher的任何specific token choice。$w_t$ 只衡量一个property——student-generated token是否对correct answer $r$ 构成positive Bayesian evidence

**与Active Learning的联系**：Bayesian evidence ratio $w_t$ quantifies per-token information gain toward correct answer，这与Active Learning中的acquisition functions（如expected information gain, BALD score）有concept相似性。

### 4.5 Step 3: Clipped Credit Assignment

**公式(15)**：

$$\hat{A}_t = A \cdot \text{clip}(w_t, 1-\epsilon_w, 1+\epsilon_w)$$

变量含义：
- $\hat{A}_t$：modified token-level advantage
- $A$：sequence-level advantage
- $\epsilon_w$：per-token credit deviation bound（实验中 $\epsilon_w = 0.2$）

类似PPO [9]/GRPO的clip机制，但clip的是credit redistribution magnitude而非policy update step size。都是trust-region constraint。

为避免training开始时的abrupt transition，用 $\lambda \in [0,1]$ 线性插值uniform advantage和reweighted advantage：

$$\hat{A}_t = A \cdot \big((1-\lambda) + \lambda \cdot \text{clip}(w_t, 1-\epsilon_w, 1+\epsilon_w)\big)$$

实验中 $\lambda$ 从0.5线性decay到0（前50步），smoothly transition到vanilla GRPO。这个schedule反映curriculum-aware design：早期dense credit assignment加速收敛，后期environment reward alone足够。

**最终objective**（**公式16**）：

$$\mathcal{L}_{\text{RLSD}}(\theta) = \mathbb{E}\left\{\frac{1}{G}\sum_{i=1}^G\frac{1}{|y^{(i)}|}\sum_{t=1}^{|y^{(i)}|}\min\Big[w_t A^{(i)}, \text{clip}(w_t, 1-\epsilon_w, 1+\epsilon_w) A^{(i)}\Big]\right\}$$

### 4.6 Algorithm 1解析

Algorithm 1给出完整训练流程：

```
1. for each training iteration:
2.   Sample batch of questions {x}
3.   for each x with privileged r:
4.     # Step 1: On-policy rollout
5.     Sample G responses {y^(1),...,y^(G)} ~ π_θ(·|x)
6.     # Step 2: Sequence-level advantage
7-10. for i: get reward R(x, y^(i)); end for
11.    Compute A^(i) = (R(x,y^(i)) - μ_G) / σ_G
12.    # Step 3: Token-level credit via self-distillation
13.    for i: forward pass with (x, r, y^(i))  # single extra forward
14-17. for t: compute Δ_t, w_t, Â_t^(i)
21.    # Step 4: Policy update
22. Update θ by maximizing L_RLSD
```

**关键实现细节**：
- Step 13只需要**single additional forward pass** per response获得teacher logits，相对rollout generation的wall-clock cost可忽略
- Teacher model parameters每10步sync一次with student，期间frozen（介于Strategy A和B之间的hybrid，但gradient direction anchoring确保了leakage-free）
- **No auxiliary distillation loss** introduced——RLSD是standard GRPO pipeline的drop-in replacement，只modify trajectory内credit的internal redistribution

### 4.7 Unified Token-Level Advantage Perspective

GRPO、OPSD、RLSD都可用**公式(17)**的single policy gradient template表达：

$$\Delta\theta \propto \mathbb{E}_{y\sim P_S(\cdot|x)}\left[\sum_{t=1}^{|y|}\hat{A}_t \nabla_\theta \log P_S(y_t|x, y_{<t})\right]$$

三种method只在 $\hat{A}_t$ 定义上differ：

- **GRPO**：$\hat{A}_t = A$（uniform advantage），方向fully grounded in environment reward但无token-level discrimination
- **OPSD**（reverse KL via log-derivative trick）：$\hat{A}_t = \Delta_t = \log P_T(y_t) - \log P_S(y_t)$。Environment reward $R(x,y)$ **entirely absent** from $\hat{A}_t$——即便trajectory给出错误答案（$A<0$），teacher favored的token（$\Delta_t > 0$）仍receive positive advantage，**decoupling optimization direction from verifiable correctness signal**
- **RLSD**：$\hat{A}_t = A \cdot \text{clip}(w_t, 1-\epsilon_w, 1+\epsilon_w)$。Environment reward决定direction（sign），teacher决定magnitude

### 4.8 Leakage-Free Guarantee (Theorem 5)

RLSD在**三个level上isolated** privileged information $r$：

**(i) Directional isolation**：由 $\text{sign}(\hat{A}_t) = \text{sign}(A)$，$r$ 无法influence任何token的gradient sign。Parameter updates在 $\theta$-space的方向由student-sampled token $y_t$（via $\nabla_\theta \log \pi_\theta(y_t|x, y_{<t})$）和environment correctness（via $\text{sign}(A)$）完全决定，**两者都不contain information about $r$**

**(ii) Support isolation**：expectation over $y \sim \pi_\theta(\cdot|x)$ sampled from student's own policy（无 $r$ access）。Log-derivative $\nabla_\theta \log \pi_\theta(y_t|x, y_{<t})$ 只在student自己generate的tokens上evaluated。任何只存在于privileged mode的token $y_{\text{leak}} \notin \text{supp}(\pi_\theta(\cdot|x, y_{<t}))$ 严格zero sampling probability，贡献zero expected gradient

**(iii) Magnitude boundedness**：$w_t$ 被clip到 $[1-\epsilon_w, 1+\epsilon_w]$ 并进一步被 $\lambda$ attenuate。当 $P_S \to P_T$ 时 $w_t \to 1$，RLSD自动degrade到vanilla GRPO

**对比OPSD per-sample gradient**：$g(\theta; r) = -\sum_{v\in\mathcal{V}} P_T(v|r) \nabla_\theta \log P_S(v)$ 跨越entire vocabulary $\mathcal{V}$——teacher因 $r$ 强烈favor的token receive $P_T(v|r)$-weighted gradient contribution，主动pull unseen privileged patterns进入parameter updates。RLSD完全eliminate这个channel。

**Trilemma resolution**（Appendix A.6.3）：
- **(a) Objective stability**：RLSD优化的是environment reward $R(x,y) \in \{0,1\}$，external signal independent of $\theta$。Teacher只通过stop-gradient scalar $w_t$ contribute，不构成optimization objective的一部分。Theorem 2的teacher drift term $\Delta_T$ **不applicable**
- **(b) Sustained improvement**：RLSD的two-phase curriculum design——早期dense credit assignment加速收敛，$\lambda$ linearly decay to 0 transition到vanilla GRPO——确保GRPO policy gradient保持non-zero，training不stall
- **(c) Leakage-free training**：由Theorem 5，gradient direction完全由environment reward和student on-policy samples决定。Proposition 1的deviation $\delta$ 不arise（无distribution matching，无 $[P_T(v|r) - \bar{P}_T(v)]$ bias）。Proposition 3的self-reinforcing feedback loop也absent

Table 3清晰对比：
| Property | OPSD (Frozen) | OPSD (Online) | RLSD |
|---|---|---|---|
| (a) Objective stability | √ | × | √ |
| (b) Sustained improvement | × | √ | √ |
| (c) Leakage-free training | × | × | √ |

---

## 五、实验

### 5.1 Setup详解

**Training data**: MMFineReason-123K [13]，从MMFineReason-1.8M corpus中通过difficulty filtering得到。具体：用Qwen3-VL-4B-Thinking做4个独立rollout，**只保留model在所有4次attempt都失败的样本**。这个conservative criterion丢弃trivial examples，concentrate training signal on challenging problems，yields faster convergence + 更efficient compute use。

**5个multimodal reasoning benchmarks**：
- **MMMU** [14]：college-level subjects across science/engineering/humanities，要求perception + domain knowledge
- **MathVista** [15]：visual contexts中的mathematical reasoning
- **MathVision** [16]：competition-level visual math problems
- **ZeroBench** [17]：designed to be unsolvable by current frontier models，stress test for reasoning robustness
- **WeMath** [18]：fine-grained mathematical problem-solving with structured difficulty levels

**Base model**: Qwen3-VL-8B-Instruct [19]

**Baselines**：
- **GRPO** [1]：standard RLVR
- **OPSD** [7]：on-policy self-distillation with privileged reasoning traces
- **SDPO** [8]：self-distillation extended to RL with rich feedback，用successful previous rollout作为privileged context
- **GRPO+OPSD**：linear interpolation of GRPO loss和KL distillation loss（inspired by MOPD in MIMO-v2-Flash [6]）
- **Base LLM**：reference

**Implementation**：VERL [20] + EasyR1 [21] frameworks。4 compute nodes × 8 NVIDIA H200 140GB GPUs。

**Hyperparameters**：
- Max context length: 8192（max prompt 4096, max response 4096）
- LR: $1 \times 10^{-6}$ for GRPO/GRPO+OPSD/RLSD；$1 \times 10^{-5}$ for OPSD/SDPO（按原implementation）
- Batch size: 256
- 8 rollouts/prompt，temperature 1.0
- Clip thresholds: $\epsilon_{\text{low}} = 0.2, \epsilon_{\text{high}} = 0.28$
- 省略KL penalty loss和entropy regularization loss
- RLSD: $\lambda$ init 0.5 linearly decay to 0 over first 50 steps，$\epsilon_w = 0.2$
- Teacher sync每10步

**Privileged information requirements**：
- OPSD: verified reasoning traces（distilled from Qwen3-VL-235B-A22B-Thinking并verified correct in MMFineReason-123K）
- SDPO: successful previous rollout作为privileged context
- **RLSD: 只需final ground-truth answer，无需reasoning trace**——least demanding

参考链接：
- MMMU: https://arxiv.org/abs/2311.16502
- MathVista: https://arxiv.org/abs/2310.02255
- MathVision: https://arxiv.org/abs/2402.14804
- ZeroBench: https://arxiv.org/abs/2502.09696
- WeMath: https://arxiv.org/abs/2407.01284
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- VERL (HybridFlow): https://arxiv.org/abs/2409.19256
- EasyR1: https://github.com/hiyouga/EasyR1

### 5.2 Main Results (Table 2)

| Method | MMMU | MathVista | MathVision | ZeroBench | WeMath | Avg |
|---|---|---|---|---|---|---|
| Base LLM | 62.44 | 73.80 | 47.37 | 19.76 | 54.10 | 51.49 |
| GRPO | 65.11 | 76.20 | 48.82 | 22.60 | 56.57 | 53.86 |
| OPSD | 63.82 | 75.10 | 47.53 | 21.06 | 54.95 | 52.49 |
| SDPO | 65.11 | 74.00 | 47.27 | 25.15 | 52.19 | 52.74 |
| GRPO+OPSD | 63.22 | 75.90 | 48.52 | 22.16 | 54.76 | 52.91 |
| **RLSD** | **67.22** | **78.10** | **52.73** | 24.85 | **58.00** | **56.18** |

**关键观察**：

1. **RLSD vs Base LLM**: +4.69% avg
2. **RLSD vs GRPO**: +2.32% avg，其中MathVision +3.91%, MathVista +1.9%（fine-grained reasoning discrimination最benefit的场景）
3. **RLSD vs OPSD/SDPO**: 大幅超越。OPSD even below GRPO，validating理论分析——OPSD的leakage导致long-term degradation
4. **RLSD vs GRPO+OPSD**: +3.27%。GRPO+OPSD的linear interpolation of bounded reward（$R \in \{0,1\}$）with unbounded high-variance KL loss causes **severe scale mismatch**，forcing suboptimal trade-off destabilize training。RLSD通过**multiplicative modulation** + exponentiation into dynamically bounded relative multiplier，**mathematically guarantees strict sign preservation**

注意ZeroBench上SDPO最高（25.15 vs RLSD的24.85）。这是个edge case——ZeroBench是designed to be unsolvable，可能SDPO用successful previous rollout作为context在extreme difficulty下更effective，但overall average RLSD胜出。

### 5.3 Training Dynamics (Figure 5)

**Figure 5(a) Reward dynamics**: RLSD初始ascent更陡，converge到更高accuracy reward ceiling，避免OPSD的late-stage collapse。这exactly对应理论two-phase dynamics——RLSD没有Phase 2的 $\delta$ 主导问题。

**Figure 5(b) Entropy dynamics**: GRPO rapid entropy collapse（uniform sequence-level reward导致每个token都被uniformly suppress），RLSD保持consistently higher entropy（selectively strengthen critical reasoning tokens without uniformly suppress alternatives at every position）。这与"reasoning with exploration" [31]、high-entropy minority tokens [32]、Seed-GRPO [33] 等工作的entropy perspective呼应。

**Figure 5(c) Clip ratios**: clip ratios稳定在3%-6%，证明clipping mechanism actively engaged，成功bound teacher的per-token influence，作为trust-region constraint analogous to importance ratio clipping in PPO/GRPO。

参考：
- Reasoning with Exploration (entropy perspective): https://arxiv.org/abs/2510.10649
- Outcome-grounded advantage reshaping: https://arxiv.org/abs/2601.07408

### 5.4 Case Study (Figure 6)

**Correct cube-counting example**：RLSD分配larger credit给真正决定正确性的tokens——identifying relevant yellow cube + executing final subtraction，downweight generic narration如"Looking at the image, I see..."

**Incorrect bar-model example**：RLSD集中strongest blame在misread relation "3x=28.5"和derived wrong answer "x=9.5"，neutral setup tokens得到相对较小惩罚。

这与design goal consistent——environment reward仍决定trajectory是reinforced还是penalized，privileged teacher只modulate token-level credit的relative magnitude。Update pattern既非uniform（GRPO）也非distribution matching（OPSD），是**targeted token-level credit assignment anchored to verifier-grounded correctness**。

---

## 六、与其他方法的深入关系

### 6.1 与PRM (Process Reward Models)对比

PRM类方法（如Let's Verify Step by Step [22]、Math Shepherd [23]、Automated Process Supervision [25]、Step-level Value Preference Optimization [26]、Generative Verifiers [27]、PRIME [30]）通过step-level reward modeling实现fine-grained credit assignment。

**Limitations**：
- 需要costly human step annotation（Let's Verify）或automated supervision（Math Shepherd, PRIME）
- 需要auxiliary reward modeling和extra computation beyond base policy
- Estimate仍是noisy

**RLSD advantage**：无需auxiliary model，仅需single additional forward pass，自然利用self-distillation的dense token-level assessment。 Yang et al. 自己的earlier work如Test-time Prompt Intervention [24]、Dynamic Early Exit [28]、S-GRPO [29] 也在reasoning model efficiency方向，但RLSD的mechanism完全不同。

参考：
- Let's Verify Step by Step: https://openreview.net/forum?id=v8L0pN6EOi
- Math Shepherd: https://aclanthology.org/2024.acl-long.510
- Generative Verifiers: https://arxiv.org/abs/2408.15240

### 6.2 与entropy/attention-based credit assignment对比

Recent work用model-internal proxies做token-level credit：
- **Entropy-based**: Reasoning with Exploration [31], High-entropy minority tokens [32], Seed-GRPO [33], Beyond high-entropy exploration [36]
- **Uncertainty-aware**: Uncertainty-aware advantage shaping [10]
- **Key-token statistics**: KTAE [34]
- **Attention dynamics**: Attention illuminates LLM reasoning [35]
- **Outcome sensitivity**: Outcome-grounded advantage reshaping [11]

这些都是**heuristic proxies**。RLSD不依赖heuristic，通过privileged context下的self-distillation提供rigorous token-level assessment，同时keep update direction anchored to verifier reward。

### 6.3 与TRRD的对比

TRRD (Reinforcement-aware knowledge distillation, arxiv 2602.22495) [42] 是concurrent work，也识别了additive KL penalty与reward maximization的conflict，提出inject teacher probabilities into policy importance ratio。

**Key difference**：TRRD operates on trust region anchor，RLSD直接通过Bayesian evidence ratio modulate advantage magnitude。从structural perspective，RLSD的formulation更principled——直接利用Bayesian belief update的telescoping property。

参考：TRRD: https://arxiv.org/abs/2602.22495

### 6.4 与其他self-distillation variants对比

On-policy self-distillation variants已被explored for:
- **Continual learning from demonstrations**: Self-distillation enables continual learning [38]
- **Context internalization**: On-policy context distillation [39]
- **Reasoning compression**: On-policy self-distillation for reasoning compression [40]
- **Privileged information distillation**: Privileged information distillation for language models [41]

这些methods都retain distribution-matching objective。RLSD departs from this paradigm——不use teacher作为generative target，repurpose privileged discrepancy only as scalar multiplier for credit assignment。

---

## 七、Intuition构建与联想

### 7.1 关键intuition：Direction vs Magnitude的asymmetric需求

这是RLSD的core design principle。可以把这想象成汽车驾驶：
- **Direction = steering wheel**：必须reliable，一旦错了直接撞墙（policy被毁）
- **Magnitude = gas pedal**：越fine-grained越好，但即便有点noise也能tolerate

Environment reward给出sparse但reliable的direction；teacher给出dense但只用于magnitude modulation。这种decoupling让两者各司其职。

### 7.2 Bayesian evidence ratio的intuition

考虑solving $2x + 3 = 7$：
- 写"2x = 4"：观察这个token后，"x=2"是correct answer的posterior belief大幅increase → $w_t > 1$，positive evidence
- 写"5x = 7"：观察这个token后，posterior belief sharp decrease → $w_t < 1$，negative evidence
- 写"therefore"：观察这个token后，posterior belief不变 → $w_t = 1$，neutral

这个telescoping到sequence-level：$\prod_t w_t = P(r|x,y)/P(r|x)$，即observe整个trajectory后的total belief update。Per-token weights是这sequence-level evidence的fine-grained token-level decomposition。这把credit assignment从sequence level（GRPO）elevate到token level，principled且practically cost-free。

### 7.3 为什么从gradient weight到scalar multiplier是关键转变

注意Appendix A.5.6的observation：OPSD per-sample gradient中的importance weight $P_T(v|r)/P_S(v)$ 与RLSD的evidence ratio $w_t$ **mathematically identical**。这不是coincidence——两种method operate on同样的underlying quantity，但employ它fundamentally differently：

- **OPSD**：$P_T/P_S$ 作为gradient weight spanning entire vocabulary $\mathcal{V}$，drive $P_S$ match $P_T$ 的shape。Per-sample deviation $[P_T(v|r) - \bar{P}_T(v)]$ 与expected signal **inseparable**（Proposition 1），r-specific patterns infiltrate parameter updates

- **RLSD**：$P_T/P_S$ 作为stop-gradient scalar credit multiplier applied only to on-policy sampled token $y_t$。Modulate magnitude of existing advantage但不进入gradient direction

**这从source of leakage转变为tool for precise credit attribution，关键在usage的转变**。这让人想起GAN中mode collapse的diagnosis——不是generator/discriminator的capacity问题，而是objective formulation的问题。

### 7.4 与Variational Inference的结构类比

Theorem 1的分解 $\mathcal{L}_{\text{OPSD}} = \mathcal{L}^* + I(Y_t; R|X, Y_{<t})$ 在structure上类似ELBO分解：

$$\log p(x) = \text{ELBO} + \text{KL}(q(z|x) \| p(z|x))$$

两者都通过KL decomposition揭示一个**inreducible gap**——在VI中是approximating family与true posterior的gap，在OPSD中是student（不condition on $r$）与teacher（condition on $r$）的conditional vs marginal的gap。这个structural similarity可能预示着更深的connection，比如用VI的amortized inference视角理解self-distillation。

### 7.5 与Information Bottleneck的联系

$I(Y_t; R|X, Y_{<t})$ 作为irreducible gap，与Information Bottleneck (IB) [Tishby]的objective $\min I(X;Z) - \beta I(Z;Y)$ 有structure similarity。可能可以formally connect RLSD到IB framework——$r$ 作为"compressed representation" of correct answer，student学到的marginal $\bar{P}_T$ 是 $r$ 的"minimal sufficient statistic"。

### 7.6 与GFlowNet的潜在connection

RLSD的Bayesian evidence ratio与GFlowNet [Bengio]的reward signal有conceptual相似：
- GFlowNet: $P(x)/R(x)$ ratio matching to amortized sampling
- RLSD: $P_T(y_t)/P_S(y_t)$ ratio作为credit

可能可以formally connect这些frameworks——RLSD的evidence ratio可能是GFlowNet的per-token generalization。

### 7.7 Impossibility Trilemma的generalization

Theorem 3的trilemma在concept上类似：
- **GAN的mode collapse**：generator/discriminator co-evolve导致instability
- **Recommendation系统的echo chamber**：self-referential feedback amplifies biases
- **Self-training的confirmation bias**：model用自己的predictions reinforce自己

任何**self-referential optimization with shared parameters**可能都有类似trilemma。这个framework可能可以推广到其他self-distillation场景，如continual learning、model editing、iterative DPO等。

### 7.8 RLSD与curriculum learning的connection

$\lambda$ 从0.5 decay到0的设计反映了一个curriculum：
- **早期**：student policy coarse-grained，dense credit assignment最有价值，teacher guidance强
- **后期**：student已经internalize reasoning ability，environment reward alone足够drive continued improvement，smoothly transition到vanilla GRPO

这与curriculum learning的difficulty schedule有concept相似性，但更principled——transition criterion是student policy quality而非arbitrary difficulty。

### 7.9 关于"stop-gradient + scalar modulation"作为general design pattern

RLSD的stop-gradient + scalar modulation + clipping four mechanism共同保证leakage-free。这是一个**general design pattern**——anytime我们想利用privileged information但不让其contaminate optimization direction，可以apply同样模式。可能可以推广到：
- Tool use in reasoning（tool output作为privileged info）
- Multi-step planning（future state作为privileged info）
- Multi-agent cooperation（other agent's observations作为privileged info）

### 7.10 关于trilemma的philosophical reflection

Impossibility trilemma揭示一个deep principle：**任何self-referential system with shared parameters都面临stability、improvement、faithfulness的三角张力**。这可能与Gödel incompleteness、Arrow's impossibility theorem等foundational impossibility results有philosophical connection——都是关于self-referential system的structural limits。

---

## 八、Limitations和Future Work

Paper的Limitations部分坦诚：focus主要在OPSD structural limitations的理论分析和RLSD paradigm的motivation/validation。Experiments集中在multimodal reasoning scenarios。但作者preliminarily validated RLSD across broader settings：
- Pure text reasoning
- Video understanding
- Additional model families beyond Qwen series

观察到consistent gains，会在forthcoming version报告。

**可能的extension方向**：
1. **Multi-step privileged information**：从final answer扩展到hierarchical reasoning traces
2. **Cross-modal privileged information**：用一种modality作为另一种的privileged info
3. **Online curriculum learning**：基于student policy quality adaptively schedule $\lambda$
4. **Multi-teacher ensemble**：多个privileged contexts的evidence ratio aggregation
5. **Connection to GFlowNet / VI / IB**：formalize这些connections
6. **Generalization to other self-referential optimization**：apply trilemma framework to GANs, self-training, iterative DPO

---

## 九、总结

这篇paper做了一件elegant的事：**通过rigorous theoretical analysis（Theorem 1, Proposition 1, Theorem 3）揭示OPSD的structural deficiency，然后基于deep insight（direction vs magnitude的asymmetric需求）设计RLSD**。

核心take-aways：

1. **OPSD fail的root cause**：information asymmetry下distribution matching产生irreducible mutual information gap $I(Y_t; R|X, Y_{<t})$，该gap的per-sample deviation $\delta(\theta; r)$ 在path-dependent optimizer中累积，drive model encode $x \to r$ correlations

2. **RLSD的core design**：decouple direction（environment reward）和magnitude（teacher evidence ratio），通过stop-gradient + scalar modulation + clipping + on-policy sampling四个机制structurally guarantee leakage-free

3. **Bayesian interpretation**：evidence ratio $P_T/P_S$ = sequential belief update ratio $P(r|x, y_{\leq t})/P(r|x, y_{<t})$，提供principled token-level credit assignment

4. **Empirical validation**：5个multimodal reasoning benchmarks上RLSD取得best average accuracy（56.18%），比Base LLM +4.69%，比GRPO +2.32%，比OPSD +3.69%

5. **Trilemma resolution**：RLSD同时满足objective stability、sustained improvement、leakage-free training——这在OPSD下impossible

参考paper：
- arxiv (推测): https://arxiv.org/abs/2601.18734 (SDR)
- arxiv (推测): https://arxiv.org/abs/2601.20802 (RL via self-distillation)
- arxiv: https://arxiv.org/abs/2602.22495 (TRRD)

这work对post-training LLM community有几个implications：
1. **OPD vs OPSD的categorical distinction**：information symmetry是关键，self-distillation需要careful design
2. **Direction-magnitude decoupling作为general principle**：可能可以apply到其他dense signal sources
3. **Trilemma framework**：可能可以推广到其他self-referential optimization场景

总的来说，这是一篇理论分析与method design紧密结合的elegant paper，从diagnosis到solution的逻辑链非常clean。核心insight——"同一个数学量（evidence ratio）在distribution matching中是leakage source，在credit assignment中是principled tool，关键在usage"——值得deeply internalize。
