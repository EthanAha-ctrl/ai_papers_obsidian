---
source_pdf: Learning from Trials and Errors Reflective Test-Time Planning for Embodied
  LLMs.pdf
paper_sha256: 4edf776d9ef3b3eed08a3fe3113d8f9741d0565cc693a4c774ceb55fa465225b
processed_at: '2026-08-05T13:02:24-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Reflective Test-Time Planning

Andrej，我把这篇paper"嚼碎"了再讲一遍，尽量用我们在 MLcoffee chat 时的那种语气——不端着，直击intuition，但该上公式的地方还是上。

---

## 这篇paper到底在解决什么"痛点"

 embodied LLM 现在 deployment 的姿势太僵了。模型 pretrain 完，参数 freeze，扔进一个新厨房——它该怎么蠢还怎么蠢。失败了？失败就失败了，下次同一个 task 它还是踩同一个坑。这就像一个学生考试错了题，老师不发答案，下次月考继续错同一题。

Reflexion（https://arxiv.org/abs/2303.11366 ）那套是想用 verbal critique 解决这个——每次失败让 LLM 写一段"我哪里错了"塞进 context。听起来挺好，但问题在于：**critique 只是文本，没改参数**。分布一漂移，context 里的 critique 就成了"过期的鸡汤"——读了感动，执行没用。

另一拨人搞 world model（3D-VLA、DreamerV3 https://arxiv.org/abs/2301.04104 ），让模型在脑内 rollout 未来。这是 reflection-in-action 的雏形，但 world model 是 frozen 的——你家厨房的橱柜跟训练集不一样，它脑内 rollout 出来的未来是错的，然后基于错的未来决策。

Yining 这篇 paper 的 insight 就一句话：**这两个 mode 必须闭环，而且闭环的接口是"把 verbal reflection 转成 gradient"**。光说不练假把式，光练不说傻把式。要又练又说。

---

## Schön 的理论其实很接地气

Donald Schön（https://en.wikipedia.org/wiki/Donald_Sch%C3%B6n ）这套 reflective practitioner 理论，说白了就两种"想"：

- **reflection-in-action**：你在打网球挥拍那一瞬间，脑子里飞快闪过"这球会出界吗"——这是边做边想，mental simulation。
- **reflection-on-action**：打完比赛复盘"我第二盘那个反手为啥总是下网"——这是做完再想，把结果反哺认知。

人类牛就牛在两个 mode 自由切换。LLM agent 之前最多有一个，且都不到位。

Yining 又加了个 **retrospective reflection**——专门解决"当时看着对、5步后发现挖坑了"这种 long-horizon credit assignment。你把小车先放进隔间A，执行 feedback 是 success，外部 reflection 打80分。结果5步后想放大熊，发现只有A能放——小车把唯一位置占了。这时 retrospective 回头重评 a_3，把80分改成25分。这个"事后诸葛亮"产生的训练信号才是真金白银。

---

## 三个 LLM 的分工——类比 actor-critic

部署时跑三个 LLM copy，都是 LLaVA-3D-7B（https://arxiv.org/abs/2505.00624 ）的同款：

| 符号 | 角色 | 类比 | test-time 更新？ |
|---|---|---|---|
| π_θ | action 生成 | actor | 是，REINFORCE |
| V_{φ_i} | 候选打分 | critic | 是，supervised |
| V_{φ_e} | 执行后 + retro 评判 | frozen reward model | 否 |

为什么 V_{φ_e} 冻结？因为它是"现实裁判"。如果裁判也跟着学，标准会漂——你让它今天觉得"放小车挺好"，明天又"放小车糟糕"，policy 就懵了。RLHF 里 reward model 在 policy 训练时通常也是冻结的，逻辑一样。参考 InstructGPT（https://arxiv.org/abs/2203.02155 ）。

---

## Reflection-in-Action：公式慢慢拆

**标准 greedy**：

$$a_t = \arg\max_{a \in \mathcal{A}} p_\theta(a | o_t)$$

- $a_t$：第 t 步动作（自然语言 token 序列，比如 "pick up the red cup"）
- $\mathcal{A}$：合法动作空间
- $p_\theta(a|o_t)$：参数 θ 的 action LLM 在观测 $o_t$ 下给 a 的概率
- $o_t$：当前 multimodal observation，含 RGB-D 点云 + 历史上下文

这就是个 greedy decoding，问题在于 commit 过早——都没"想"就动手。

**改成 sample N 个 candidate**：

$$a_t^k \sim p_\theta(\cdot | x_{\text{action}}; T), \quad k=1,\ldots,N$$

- $a_t^k$：第 t 步第 k 个候选
- $x_{\text{action}}$：prompt，里面装了 task τ、当前观测 $o_t$、上一步动作 $a_{t-1}$、上一步 external reflection $f_e^{t-1}$
- $T$：采样温度，论文取 T=2.0（Long-Horizon Household）或 T=1.25（Cupboard）——高温保 diversity
- $N$：候选数，主实验 N=4，ablation 显示 N=6 最优

**用 V_{φ_i} 给每个 candidate 打分**：

$$f_i^{t,k}, s_i^{t,k} = V_{\phi_i}(x_{\text{internal}}^k)$$

- $f_i^{t,k}$：自然语言反思文本，例如"这个 placement 会堵住未来大物体的空间"
- $s_i^{t,k} \in [0, 100]$：数值分数
- $x_{\text{internal}}^k$：在 $x_{\text{action}}$ 后面 append 候选动作 $a_t^k$，让模型评估"如果做这个会怎样"

**选最高分执行**：

$$a_t^* = a_t^{(\arg\max_{k \in [N]} s_i^{t,k})}$$

注意没有 MCTS 那种 lookahead rollout——embodied 环境的 rollout 太贵了，一步真实执行就够慢的，更别说在脑内 simulate 多步。这里全靠 V_{φ_i} 一次 forward 给的预估。所以 V_{φ_i} 必须准——这也是为什么 V_{φ_i} 必须 test-time update，不然 SFT 出来的偏差会一直放大。

这个套路其实跟 Best-of-N rejection sampling（https://arxiv.org/abs/2204.05862 ）和 Self-Consistency（https://arxiv.org/abs/2203.11171 ）是表亲，只不过这里 verifier 是个 verbal LLM 而不是 majority vote 或外部 reward model。

---

## Reflection-on-Action：三层结构

### 第一层：immediate external reflection

执行完 $a_t^*$ 看到 $(o_{t+1}, e_t)$，让 V_{φ_e} 写反思：

$$f_e^t, s_e^t = V_{\phi_e}(x_{\text{external}})$$

- $e_t$：execution flag（success / fail）
- $x_{\text{external}}$：$x_{\text{action}}$ + $a_t^*$ + $e_t$ + $(o_t, o_{t+1})$ 这对前后观测

关键限制：$f_e^t$ 只能看到下一步 immediate outcome。一个看似成功的 pick-up，可能5步后让大物体无处可放——immediate reflection 看不到这种"延迟后果"。

### 第二层：working memory buffer

$$\mathcal{W}_t = \{(o_j, a_j, e_j, f_e^j) \mid j = t-K+1, \ldots, t\}$$

K=5 的 sliding window。当 $|\mathcal{W}| = K$ 或者遇到 milestone（房间切换、连续失败），就触发 consolidation + test-time training。K=5 是工程权衡——太小 retro 看不到因果链，太大 prompt 爆炸。

### 第三层：retrospective reflection with hindsight

到了 milestone，对 working memory 里每个历史动作 $a_j$ 重新打分：

$$f_r^j, s_r^j = V_{\phi_e}(x_{\text{retro}}^j)$$

$x_{\text{retro}}^j$ 包含：
1. 整个 $\mathcal{W}_t$（"完整经历给你看"）
2. 待重评的 $a_j$
3. $a_j$ 上一次的 reflection（$f_e^j$ 或上一轮 $f_r^j$）
4. 当前观测 $o_{t+1}$（hindsight 证据）

这个 prompt 结构让 V_{φ_e} 能看到"做了 $a_j$ 之后，世界变成了什么样"，从而给 $a_j$ 重新打分。这就把 delayed consequence 拉回到当时那个决策点上——本质是 credit assignment 的 verbal 版本。

存进 retro-buffer $\mathcal{D}_{\text{retro}}$，只保留最新一次 retro（避免历史污染）：

$$\mathcal{D}_{\text{retro}} = \{(a_j, f_r^j, s_r^j)\}$$

---

## Test-Time Training：把 reflection 焊成 gradient

这一步是 paper 最核心的工程创新——reflection 不再只是 context 里的文字，它变成梯度。

### 数据集构造

$\mathcal{D}_{\text{train}}$ 由两部分拼：

**Retro-supervised pairs**（公式 7）：

$$\mathcal{D}_{\text{retro}} = \{(a_j, f_r^j, s_r^j)\}$$

这些是 hindsight 修正后的"标准答案"，既训 V_{φ_i} 也训 π_θ。

**Regularization pairs**（公式 8）：

$$\mathcal{D}_{\text{reg}} = \{(a_l, f_i^l, s_i^l)\}, \quad f_i^l, s_i^l = V_{\phi_i}(x_{\text{internal}}^l)$$

$a_l$ 是随机采样的未执行动作，$(f_i^l, s_i^l)$ 是 V_{φ_i} **当前**对它们的预测。把这些"当前预测"当 anchor，防止 V_{φ_i} 在那 5-15 个 retro sample 上 overfit 然后对其他动作乱漂。这就是 continual learning 里的 replay buffer 思路（参考 https://arxiv.org/abs/2404.00418 ）。

### 训练 V_{φ_i}：supervised NLL

$$\mathcal{L}_{\text{internal}}(\phi_i) = \mathbb{E}_{(x_{\text{internal}}, f, s) \sim \mathcal{D}_{\text{train}}} \left[ -\log p_{\phi_i}(f | x) \right]$$

- $f$：目标 reflection 文本（retro 后的 $f_r$ 或 reg pair 的 $f_i$）
- $p_{\phi_i}(f|x)$：V_{φ_i} 自回归生成 f 的概率
- 注意分数 $s$ 没进 loss——只监督 reflection 文本。Design choice：reflection 写对了分数自然就对了。

5 个 epoch，LoRA 学习率 $5 \times 10^{-3}$，base-weight 学习率 $5 \times 10^{-5}$：

$$\phi_i^{(e+1)} = \phi_i^{(e)} - \eta_\phi \nabla_{\phi_i} \mathcal{L}_{\text{internal}}$$

### 训练 π_θ：REINFORCE

**Token 级 log prob**（公式 11）：

$$\log p_\theta(a | x_{\text{action}}) = \sum_{i=1}^{|a|} \log p_\theta(a_i | a_{<i}, x_{\text{action}})$$

- $|a|$：动作 token 长度
- $a_i$：第 i 个 token
- $a_{<i}$：前 i-1 个 token
- 标准自回归分解

**Reward rescaling**：

$$r = 2 \cdot \frac{s_r}{100} - 1 \in [-1, 1]$$

把 0-100 分映射到 [-1, 1]。**这个细节很关键**——如果直接用 $s_r/100$，最差也有 0 reward，policy gradient 会偏向"反正执行就行"。映射到 [-1, 1] 让差动作真的有 negative gradient，把它的概率压下去。

**REINFORCE loss**（公式 12）：

$$\ell_\theta = -r \cdot \log p_\theta(a | x_{\text{action}})$$

注意没有 baseline subtraction——标准 REINFORCE 通常会减 baseline 减方差，这里简化了。因为 $r \in [-1, 1]$，方差还可以接受。

**Update**（公式 13）：

$$\theta^{(s+1)} = \theta^{(s)} - \eta_\theta \nabla_\theta \sum \ell_\theta$$

LoRA 学习率 $1 \times 10^{-2}$，base-weight 学习率 $1 \times 10^{-3}$，3 RL steps，gradient clipping 0.5。

LoRA 配置：rank 4-8, alpha 8-16, dropout 0.1-0.15, target 是 q_proj/v_proj 或者 all linear layers except lm_head/embed_tokens/vision encoder。

---

## 实验结果：哪些数字最 striking

### Long-Horizon Household（Table 1）

| 方法 | Fitting | Selection | Preparation | Hybrid | Avg |
|---|---|---|---|---|---|
| Reflexion | 8.51 | 8.82 | 15.9 | 6.45 | 9.92 |
| Self-Refine | 10.6 | 11.8 | 12.7 | 9.68 | 11.20 |
| ReflectVLM | 2.12 | 5.88 | 14.3 | 6.45 | 7.19 |
| PPO | 0 | 2.94 | 7.94 | 3.23 | 3.53 |
| DreamerV3 | 4.26 | 11.8 | 11.1 | 12.9 | 10.02 |
| 3DLLM-Mem | 10.6 | 14.7 | 9.52 | 9.68 | 11.13 |
| **Ours** | **44.7** | **32.4** | **31.7** | **25.8** | **33.65** |

3 倍以上的提升。PPO 在 Fitting 上 0%——这种 sparse reward long-horizon 任务，RL exploration 根本走不通。DreamerV3 卡在 4.26% 因为 world model frozen，新场景就废。

### Cupboard Fitting（Figure 3）

| 方法 | Fit | Correct |
|---|---|---|
| Vanilla | ~44.5 | ~10 |
| w/o RIA | 53.5 | ~15 |
| w/o ROA | 45.2 | ~12 |
| Full (Base-Weight) | 57.4 | ~22 |
| Full (LoRA) | **60.2** | **25.3** |

LoRA 反而比 Base-Weight 好！60.2 vs 57.4。这点反直觉但合理——test-time 训练数据极少（10-15 个 retro sample per iteration），全参数更新容易 overfit，LoRA 的低秩约束本身就是强 regularizer。这跟 LoRA 在 few-shot finetuning 上的优势一致（https://arxiv.org/abs/2106.09685 ）。

---

## Ablation 最 striking 的发现：RIA 和 ROA 互相依赖

> "Sometimes removing just one component performs worse than removing both. For instance, without RIA, Preparation falls to 3.17% and Hybrid to 3.23%, while removing both components yields 11.1% and 12.9% respectively."

这是**反协同**——加一个比两个都不加还差。怎么理解？

- **RIA without ROA**：V_{φ_i} 是 SFT 训出来的，对当前环境有 systematic bias。没有 ROA 纠正，高温采样出 N 个 candidate 被同一个 bias 打分，反而过度自信地挑错。Greedy 至少随机性高，偶尔撞对。
- **ROA without RIA**：训练数据来自 greedy 选的"差动作"，hindsight 后才知道差。但这些动作没探索环境（没去别的房间看），hindsight 也学不到啥。Garbage in garbage out。

两个一起才形成 virtuous cycle：RIA 保证 sample 高质量 → ROA 学到的 reflection 有信息量 → V_{φ_i} 变强 → 下次 RIA 更准。这是 **data quality flywheel**。

类比 GAN——generator 和 discriminator 必须一起变强，单方面变强反而崩。或者 actor-critic 里 critic 太强 actor 学不动，critic 太弱 actor 学错的——必须 coordinated update。

Loss ablation 也显示同样 dependency：去掉 internal loss 或 action loss 单独，都比两个都去掉差。这呼应 double-loop learning（https://hbr.org/1977/09/double-loop-learning-in-organizations ）——single loop（只 update policy）是行为修正，double loop（同时 update value）是认知修正。

---

## Compute-Matched 对照：反思 ≠ 多算

附录 B 做了个特别重要的对照：给 vanilla baseline 3× steps（匹配 full model 的 3× wall-clock），看是不是"多花时间就行"。

| 方法 | Avg |
|---|---|
| Vanilla | 8.79% |
| Vanilla (3× steps) | 8.46% |
| Full | 33.65% |

Vanilla 给 3 倍时间反而略降（8.79 → 8.46）。这证明 reflection 的价值不是单纯多算——多算只是让 agent 多犯几次同样的错。Reflection 改变的是 decision process 本身。

这点很关键，回应了 LLM test-time scaling 文献里 "compute is all you need" 的简化叙事（https://arxiv.org/abs/2408.03314 ）。单纯增加 inference compute（多 sample 多数 majority vote）在某些任务上有效，但在需要"改变决策机制"的 task 上，compute 换不来 competence。

---

## HM3D 泛化：reflection 学到的是 meta-skill

附录 C 把只在 BEHAVIOR 上训练的模型 zero-shot 丢到 HM3D（https://arxiv.org/abs/2109.08238 ）的 photorealistic scene 上跑 Preparation：

| 方法 | HM3D Prep |
|---|---|
| 3DLLM-Mem | 7.32% |
| Reflexion | 2.44% |
| PPO | 0% |
| ReflectVLM | 0% |
| **Ours** | **19.5%** |

绝对数字从 BEHAVIOR 的 31.7% 掉到 19.5%，掉 12.2 pp，但仍保持相对优势。这表明 reflection 学到的不完全是 BEHAVIOR-specific 的知识，而是某种"如何从失败中恢复"的元能力——这种能力能跨 domain transfer。

我觉得这是 paper 最有价值的发现之一：test-time adaptation 不只是 narrow fit，它学到的是 meta-skill。这个 meta-skill 是"如何反思"，不是"如何做家务"。

---

## Single-Step vs Receding Horizon：少即是多

附录 E 对比"单步生成 + retro" vs "5-step sequence planning + 执行第一步"：

| 方法 | Fit | Compute |
|---|---|---|
| Receding Horizon | 57.8% | 5× |
| Single-Step | 60.2% | 1× |

Receding Horizon 多花 5 倍算力还更差。作者的解读：

> "Test-time training fundamentally relies on learning from actual execution feedback... However, generating 5-action sequences forces the model to predict these outcomes before they occur... creates optimization interference where gradients from test-time training (learned from reality) fight against the inductive bias from sequence generation (learned from imagination)."

翻译成人话：receding horizon 让模型在脑内预演一个未来并 commit，但 test-time training 又用 reality 的反馈推翻这个 imagination——两个目标冲突，gradient 互相打架。Single-step 干脆不想象，让 retro 来做 implicit long-horizon reasoning——把 multi-step lookahead 蒸馏进 V_{φ_i} 的 single-step 评估里。

这点跟 Sutton 在 RL 里强调的 "model-based vs model-free trade-off" 深度呼应：用 world model 做 long rollout 在 model 错时会 amplify error；不如 model-free + value approximation，把 long-horizon 信息压进 value function。参考 Rich Sutton 的 "The Bitter Lesson"（http://www.incompleteideas.net/IncIdeas/BitterLesson.html ）。

---

## Hyperparameter 的几个 takeaway

**N（candidate 数）**：N=1 53.0% → N=6 峰值 60.0% → N=10 58.8%。N=6 是 sweet spot。超过之后 candidate pool 里低质量样本变多，V_{φ_i} 打分噪声上升。

**Temperature T**：T=0 → 52.6%，T=1.25 峰值 60.0%，T=2.0 → 47.0%。Inverted U。低温 candidate 太像，V_{φ_i} 没区分度；高温 candidate 太乱，物理不可行，V_{φ_i} 救不回来。

**LoRA rank**：(r=8, α=16) 峰值 60.0%；(r=32, α=32) 灾难性 collapse 到 34.8%——mode collapse，对所有输入输出同样的动作。这是 LoRA 在小数据 + 高 LR 下的经典 failure。工程上要警惕。

**Action budget**：50 步最优，100 步反而下降。说明无限 budget 让 agent 倾向"探索而不 commit"，长 horizon 累积错误。适度 budget 是个 implicit inductive bias，逼 agent 决策果断。

---

## 我自己的几点 intuition

### Intuition 1：verbal reflection 是 lossless compression of experience

每次失败，raw 存 (obs, action, outcome) 下次 prompt 时 context 爆炸。verbal reflection 把失败抽象成"为什么"——一句"小车占了唯一大空间"足够让 LLM 下次注意。这是 **symbolic distillation**，比 replay buffer 紧凑得多。这跟人类写 debug log 一个道理——你不存所有 stack trace，你写一句"这个函数在 None 输入时 crash，记得加 guard"。

### Intuition 2：double-loop learning 的数学对应

Single-loop：只 update π_θ 让它偏好高分动作——这是 RL，行为修正。
Double-loop：同时 update V_{φ_i} 让它"知道哪些动作 retro 后会得低分"——这是 credit assignment 的 meta-update，认知修正。

在 actor-critic 里 critic 也 update，但 critic update 来自 TD error（temporal difference），这里 critic update 来自 **hindsight re-evaluation**——后者更稀疏（只有 milestone 触发），但更 interpretable。可以把 retro score 看成一种 "perfect information value"——已知未来后重新估值，类似 chess 里的 retrospective analysis。

### Intuition 3：test-time training 在 embodied 上的天然合理性

embodied 任务 deployment environment 跟 training environment 必然有 gap——每家厨房都不一样。Train-time generalization 是上限，deployment adaptation 决定下限。Test-time training 文献（Tent https://arxiv.org/abs/2006.10926 用 entropy minimization 更新 BN params）在 perception 上有效，这里用 self-generated verbal reflection 作为更丰富的 supervisory signal，是 TTA 在 embodied 上的自然延伸。

### Intuition 4：N=4-6 + T=1.25-2.0 是 test-time scaling 的 sweet spot

这呼应 o1、DeepSeek R1 类 reasoning model 的 test-time compute 分配思路——不是越多越好，要 generate diverse + verify。Verification 这里由 V_{φ_i} 完成，本质是 LLM-as-judge（https://arxiv.org/abs/2306.05685 ）在 embodied 上的应用。

### Intuition 5：mode collapse 警示

LoRA rank 32 在小数据上 mode collapse 是个 useful negative result。提醒我们 test-time training 比 offline training 脆弱——没大规模数据兜底，regularization 选择极关键。这对实际部署 embodied agent 有直接工程意义。也提示 LoRA rank 不是越大越好，要跟数据量匹配。

---

## 这篇 paper 在更大图景中的位置

把它放在三条线交汇处看就清楚了：

1. **Test-time adaptation** 线：从 Tent（BN stats update）→ TTT（self-supervised）→ LoRA-TTT（https://arxiv.org/abs/2502.02069 ）→ 这里（verbal reflection as signal）。信号越来越 rich，从 entropy 到 reconstruction 到 verbal critique。

2. **LLM agent reflection** 线：从 Reflexion（context only）→ Self-Refine（iterative）→ CRITIC（https://arxiv.org/abs/2305.11738 tool-augmented）→ 这里（parameter update）。Reflection 越来越"重"，从 text 到 gradient。

3. **Embodied LLM** 线：从 RT-2 → OpenVLA → 3D-LLM → 3D-LLM-Mem（https://arxiv.org/abs/2505.22657 ）→ 这里。从 static oracle 到 adaptive practitioner。

Yining 的工作把三条线焊在一起——TTA 的 update 机制 + agent reflection 的 verbal 形式 + embodied 的实际场景。这是少见的 cross-area synthesis。

---

## 局限和可以延伸的方向

Paper 自己承认的：
1. 只测了 vision，tactile / audio 没碰——但 framework modality-agnostic，加 MultiPLY（https://arxiv.org/abs/2401.08547 ）那种 multisensory 直接适配。
2. V_{φ_e} 冻结，意味着"裁判"不成长。人类反思是裁判也在进化的——未来可以让 V_{φ_e} 用 EMA slow update。
3. Test-time compute 3× 对 real-time robot 控制有点重，但对 household 分钟级任务可接受。
4. Retro buffer 在超长 episode（>100 步）上管理策略没说清。

我能想到的后续：
- 把 reflection 文本压缩成 structured representation（scene graph patch 而非自然语言），降 V_{φ_i} context 负担。
- 用 DPO 替代 REINFORCE，把 retro 高分 vs 低分动作做成 preference pair，可能更稳。参考 DPO（https://arxiv.org/abs/2305.18290 ）。
- 多 agent 协作反思：让 V_{φ_e} 是 ensemble，多个 critic 投票，减 single model bias。
- 把 retro score 当成训练 reward model 的数据，离线训更强 V_{φ_e}，下次部署又用上去——形成 cross-task 数据飞轮。
- 把这套 framework 搬到 tool use、web navigation、code generation——执行 → LLM 反思 → reflection 当训练数据 update policy + value。这本质是 "RLHF from AI feedback"（RLAIF，https://arxiv.org/abs/2309.00267 ）的 test-time 版本。
- 跟 process reward model（PRM，https://arxiv.org/abs/2305.20050 ）结合——retro score 是天然 step-level reward 信号。
- 扩展到 hierarchical reflection——low-level retro 评单步，high-level retro 评整段策略，类似 hierarchical RL。

---

## 一句话总结

这篇 paper 的核心 contribution 是把 Schön 的 reflective practitioner 理论"可操作化"——用 verbal reflection 作为 self-supervised signal，在 test-time 同时 update actor 和 critic，闭环 reflection-in-action 和 reflection-on-action，并用 retrospective reflection 解决 long-horizon credit assignment。在 Long-Horizon Household 和 Cupboard Fitting 两个 benchmark 上比 baseline 翻 3 倍，且通过 compute-matched 对照证明反思 ≠ 多算。

更深层的意义在于：它给 embodied agent 指了一条从 "static oracle" 到 "adaptive learner" 的路径，且这条路不依赖外部 reward、不依赖额外标注、不依赖重训——agent 用自己的语言反思自己，再用反思更新自己。这跟人类专家习得技能的过程惊人一致。

参考链接汇总：
- 项目主页：https://reflective-test-time-planning.github.io/
- 代码：https://github.com/Reflective-Test-Time-Planning/Reflective-Test-Time-Planning
- LLaVA-3D：https://arxiv.org/abs/2505.00624
- BEHAVIOR-1K：https://behavior.stanford.edu/
- Reflexion：https://arxiv.org/abs/2303.11366
- Self-Refine：https://arxiv.org/abs/2303.17651
- DreamerV3：https://arxiv.org/abs/2301.04104
- PPO：https://arxiv.org/abs/1707.06347
- Qwen2.5-VL：https://qwenlm.github.io/blog/qwen2.5-vl/
- LoRA：https://arxiv.org/abs/2106.09685
- Tent TTA：https://arxiv.org/abs/2006.10926
- LoRA-TTT：https://arxiv.org/abs/2502.02069
- 3D-LLM-Mem：https://arxiv.org/abs/2505.22657
- DPO：https://arxiv.org/abs/2305.18290
- RLAIF：https://arxiv.org/abs/2309.00267
- PRM：https://arxiv.org/abs/2305.20050
- InstructGPT：https://arxiv.org/abs/2203.02155
- LLM-as-judge：https://arxiv.org/abs/2306.05685
- Self-Consistency：https://arxiv.org/abs/2203.11171
- Best-of-N：https://arxiv.org/abs/2204.05862
- o1 scaling：https://arxiv.org/abs/2408.03314
- CRITIC：https://arxiv.org/abs/2305.11738
- MultiPLY：https://arxiv.org/abs/2401.08547
- HM3D：https://arxiv.org/abs/2109.08238
- Double-loop learning：https://hbr.org/1977/09/double-loop-learning-in-organizations
- Schön：https://en.wikipedia.org/wiki/Donald_Sch%C3%B6n
- Bitter Lesson：http://www.incompleteideas.net/IncIdeas/BitterLesson.html

---

# Reflective Test-Time Planning 深度讲解

这篇论文由 Yining Hong、Li Fei-Fei、Jiajun Wu、Yejin Choi 等人合作完成，核心思想是把 Schön 的"反思实践者"理论搬到 embodied LLM 的 test-time adaptation 中。项目主页见 https://reflective-test-time-planning.github.io ，代码见 https://github.com/Reflective-Test-Time-Planning/Reflective-Test-Time-Planning 。

## 1. 论文要解决的根本问题

现有 embodied LLM（如 RT-2 https://arxiv.org/abs/2307.15818 、OpenVLA https://openvla.github.io/ 、3D-LLM https://3d-llm.github.io/ ）在 deployment 阶段是 **static oracle**——参数冻结，每次失败都从零开始。失败不能积累为经验，导致 agent 在同一个 trap 上反复掉坑。

作者认为问题在于现有方法至多抓住了反思的某一个 mode：

- 第一类方法如 Reflexion https://arxiv.org/abs/2303.11366 、Self-Refine https://arxiv.org/abs/2303.17651 把 verbal critique 当成 context 塞进 prompt。这本质是 reflection-on-action，但 critique 只作为文本存在，不更新参数，分布偏移下就会失效。
- 第二类方法如 3D-VLA、DreamerV3 https://arxiv.org/abs/2301.04104 用预训练 world model 做 anticipation。这是 reflection-in-action，但 world model 是 frozen 的，部署时遇到的新 dynamics 它没法吸收。

论文的核心 claim 是：**只有把两种 reflection 闭环起来、并且把 verbal reflection 转换成 self-supervised training signal 去更新参数，才能实现 double-loop learning**（Argyris 1977 https://hbr.org/1977/09/double-loop-learning-in-organizations ）。

## 2. Schön 的反思框架如何映射到 embodied agent

Donald Schön 的《The Reflective Practitioner》(https://en.wikipedia.org/wiki/Donald_Sch%C3%B6n ) 把人类专业实践分为两个 mode：

- **reflection-in-action**：在行动过程中进行，"边做边想"，本质是 mental simulation。在 agent 上对应 test-time scaling——在执行前生成多个候选动作并通过 internal LLM 打分。
- **reflection-on-action**：在行动之后进行，"做完再想"，把 outcome 反馈回 belief 与 strategy。在 agent 上对应 test-time training——用执行结果产生的 verbal reflection 更新参数。

论文还加了第三个反思 **retrospective reflection**，专门解决 long-horizon credit assignment：当下一个动作看似成功，但 5 步之后才发现它占用了唯一能放下大物体的空间，外部反馈没法立刻捕捉。retrospective reflection 用 hindsight 重评历史动作，把延迟后果回填到当时那个决策上。这对应 Schön 框架中"事后回看重新理解"的环节。

## 3. 三个 LLM 的角色与数据流

部署时系统里有三个 multimodal LLM，都是 LLaVA-3D-7B（https://arxiv.org/abs/2505.00624 ）的 copy：

| 符号 | 角色 | 触发时机 | 是否在 test-time 更新 |
|---|---|---|---|
| π_θ | action generation LLM | 每个 timestep 生成候选动作 | 是，REINFORCE |
| V_{φ_i} | internal reflection LLM | 候选动作打分（pre-action） | 是，supervised |
| V_{φ_e} | external reflection LLM | 执行后评估 + 后期 retro | 否，frozen oracle |

为什么 V_{φ_e} 冻结？因为它是"现实裁判"——必须保持稳定，作为 ground-truth signal。如果它也跟着 update，reflection 信号会自己漂移，feedback loop 失稳。这点很关键，类似于 RLHF 里 reward model 在 policy 训练时通常冻结。

架构图（Figure 2）的左半（a）是 reflection-in-action：π_θ 高温采样 N 个候选 → V_{φ_i} 给每个 candidate 打分 → 选 max score 的执行。右半（b）是 reflection-on-action：执行后 V_{φ_e} 给 external reflection → 存进 working memory W → 达到 K 步或 milestone 时触发 retrospective → 把 retro 后的 reflection 作为训练数据，通过 LoRA 同时 update π_θ 和 V_{φ_i}。

## 4. Reflection-in-Action 的公式逐项解析

**Vanilla action generation**（公式 1）：

$$a_t = \arg\max_{a \in \mathcal{A}} p_\theta(a | o_t)$$

- $a_t$：第 t 步的动作，自然语言 token 序列
- $\mathcal{A}$：合法动作空间
- $p_\theta$：参数为 θ 的 action LLM 给出的条件概率
- $o_t$：第 t 步的 multimodal observation（包含 RGB-D 转的点云 + 历史上下文）

这是标准 greedy decoding，问题是**commit 过早**，没"想一想后果"。

**Candidate generation**（公式 2）：

$$a_t^k \sim p_\theta(\cdot | x_{\text{action}}; T), \quad k=1,\ldots,N$$

- $a_t^k$：第 t 步第 k 个候选动作
- $x_{\text{action}}$：prompt，包含 task τ、observation $o_t$、上一动作 $a_{t-1}$、上一外部反思 $f_e^{t-1}$
- $T$：采样温度，论文取 T=2.0，比较激进
- $N$：候选数，论文 N=4（cupboard 实验里 ablation 显示 N=6 最优）

高温采样的目的是 diversity，让 V_{φ_i} 有"挑"的余地。这点让我想起 Best-of-N rejection sampling 和 Self-Consistency（https://arxiv.org/abs/2203.11171 ）的思路，但这里 N 个样本由 reflector 评分，而不是用 majority vote。

**Internal reflection scoring**（公式 3）：

$$f_i^{t,k}, s_i^{t,k} = V_{\phi_i}(x_{\text{internal}}^k)$$

- $f_i^{t,k}$：自然语言形式的反思，例如"这个动作会把大物体占住唯一的小隔间，后续无法放置 X"
- $s_i^{t,k} \in [0, 100]$：数值化分数
- $x_{\text{internal}}^k$：在 $x_{\text{action}}$ 基础上加了候选动作 $a_t^k$，要求模型评估"如果做这个动作，会怎样"

**Best action selection**（公式 4）：

$$a_t^* = a_t^{(\arg\max_{k \in [N]} s_i^{t,k})}$$

注意是 argmax 不是期望，意味着只挑分最高的。这和 Tree Search 中的 UCB、AlphaGo 的 MCTS 不一样，这里没有 lookahead rollout，全靠 V_{φ_i} 一次 forward 给的预估。这有好处：不用在环境里实际 rollout（embodied 环境的 rollout 极贵）；坏处是 V_{φ_i} 的准确性是 bottleneck，所以必须让 V_{φ_i} 在 test-time 也能 update。

## 5. Reflection-on-Action 的三层结构

### 5.1 Immediate external reflection（公式 5）

$$f_e^t, s_e^t = V_{\phi_e}(x_{\text{external}})$$

$x_{\text{external}}$ 在 $x_{\text{action}}$ 之上加了 $a_t^*$、execution flag $e_t$、以及 $(o_t, o_{t+1})$ 这对前后观测，让模型做 change detection——例如"抓取失败，物体仍在原位"。

**关键限制**：$f_e^t$ 只能看到 immediate next state。一个看似成功的 pick-up，可能让后续大物体无处可放。所以需要 retrospective。

### 5.2 Working memory buffer

$$\mathcal{W}_t = \{(o_j, a_j, e_j, f_e^j) \mid j = t-K+1, \ldots, t\}$$

K=5，sliding window。当 $|\mathcal{W}| = K$ 或遇到 milestone（如房间切换、连续失败）时，触发 consolidation。这里 K=5 是个工程权衡：太小则 retro 看不到足够长的因果链，太大则 retro prompt 过长、cost 高。

### 5.3 Retrospective reflection（公式 6）

$$f_r^j, s_r^j = V_{\phi_e}(x_{\text{retro}}^j)$$

$x_{\text{retro}}^j$ 的内容很关键：
1. 整个 $\mathcal{W}_t$ 作为上下文（"看完整段经历"）
2. 待重评的历史动作 $a_j$
3. 该动作最近一次的反思 $f_{\text{recent}}^j$（来自 $\mathcal{W}_t$ 中的 $f_e^j$，或来自 retro-buffer $\mathcal{D}_{\text{retro}}$ 的上一轮 $f_r^j$）
4. 当前 observation $o_{t+1}$（hindsight 证据）

这个 prompt 结构让 V_{φ_e} 能看到"做了 $a_j$ 之后，未来发生了什么"，从而给 $a_j$ 重新打分。例如：

> "Step 3 我把小车放进隔间 A，外部 reflection 当时打了 80 分。但现在 step 7 我需要放大熊，唯一能放下的是 A——小车把 A 占了。重评 $a_3$：分数应该从 80 降到 25。"

这个重打分产生的 $(a_j, f_r^j, s_r^j)$ 元组存进 $\mathcal{D}_{\text{retro}}$，只保留最新的一次 retro reflection，避免历史污染。

## 6. Test-Time Training 数据集构造

$\mathcal{D}_{\text{train}}$ 由两部分拼接：

**Retro-supervised pairs**（公式 7）：

$$\mathcal{D}_{\text{retro}} = \{(a_j, f_r^j, s_r^j)\}$$

这些是 hindsight 修正后的"标准答案"。注意它们既用来训练 V_{φ_i}（supervised），也用来训练 π_θ（RL with reward）。

**Regularization pairs**（公式 8）：

$$\mathcal{D}_{\text{reg}} = \{(a_l, f_i^l, s_i^l)\}, \quad f_i^l, s_i^l = V_{\phi_i}(x_{\text{internal}}^l)$$

这里 $a_l$ 是**未执行**的随机采样动作，$f_i^l, s_i^l$ 是 V_{φ_i} **当前**对它们的预测。把这些"当前预测"当 anchor，目的是 **catastrophic forgetting 的防御**——如果只在 retro 后的样本上更新，V_{φ_i} 会在那 5-15 个动作的小子集上 overfit，对其他动作的判断漂移。Regularization pairs 让模型保持对"未经历动作"的原有判断，类似 replay buffer 在 continual learning 里的作用（参考 https://arxiv.org/abs/2404.00418 ）。

## 7. 两个 Loss 函数的细节

### 7.1 Internal LLM 的 supervised loss（公式 9-10）

$$\mathcal{L}_{\text{internal}}(\phi_i) = \mathbb{E}_{(x_{\text{internal}}, f, s) \sim \mathcal{D}_{\text{train}}} \left[ -\log p_{\phi_i}(f | x) \right]$$

- $f$：目标 reflection 文本（来自 retro 后的 $f_r$ 或 reg pair 的 $f_i$）
- $p_{\phi_i}(f | x)$：V_{φ_i} 在输入 $x$ 下生成 $f$ 的自回归概率

这是标准 NLL，在 test-time 跑 E=5 epochs：

$$\phi_i^{(e+1)} = \phi_i^{(e)} - \eta_\phi \nabla_{\phi_i} \mathcal{L}_{\text{internal}}$$

$\eta_\phi = 5 \times 10^{-3}$（LoRA）或 $5 \times 10^{-5}$（base-weight）。注意分数 $s$ 没进 loss——loss 只监督 reflection 文本。这是因为分数本质是 reflection 文本的"摘要"，把 reflection 写对了分数自然就对了。这是 design choice，也可以把 $s$ 当 regression target 拼上。

### 7.2 Action LLM 的 REINFORCE loss（公式 11-13）

**Token-level log prob**（公式 11）：

$$\log p_\theta(a | x_{\text{action}}) = \sum_{i=1}^{|a|} \log p_\theta(a_i | a_{<i}, x_{\text{action}})$$

- $|a|$：动作的 token 长度
- $a_i$：第 i 个 token
- $a_{<i}$：前 i-1 个 token
- 这就是标准自回归分解 $\log p(a) = \sum \log p(a_i | a_{<i})$

**Reward rescaling**：

$$r = 2 \cdot \frac{s_r}{100} - 1 \in [-1, 1]$$

把 0-100 分映射到 [-1, 1]。这是为了让差动作真的有负 reward——如果直接用 $s_r/100$，最差也有 0 reward，policy gradient 会偏向 "any execution"。

**REINFORCE loss**（公式 12）：

$$\ell_\theta = -r \cdot \log p_\theta(a | x_{\text{action}})$$

注意没有 baseline subtraction（标准 REINFORCE 通常有 $\sum r \log p$，这里简化了）。这等价于 expected reward 的梯度 $\nabla \mathbb{E}[r]$，因为：

$$\nabla \mathbb{E}_{a \sim \pi_\theta}[r] = \mathbb{E}[r \nabla \log \pi_\theta(a)]$$

所以负号是为了 gradient descent。$r > 0$ 时梯度让 $\log p(a)$ 增大（更可能选），$r < 0$ 时减小。

**Update**（公式 13）：

$$\theta^{(s+1)} = \theta^{(s)} - \eta_\theta \nabla_\theta \sum_{(x_{\text{action}}, f, s_r) \in \mathcal{D}_{\text{train}}} \ell_\theta(x_{\text{action}}, f, s_r)$$

- $\eta_\theta = 1 \times 10^{-3}$（base-weight）或 $1 \times 10^{-2}$（LoRA），LoRA 学习率更高是因为只更新低秩矩阵
- 3 RL steps，accumulate gradients over all training pairs per step
- gradient clipping 0.3 (supervised) / 0.5 (RL) 防 LoRA 训练崩坏

LoRA 配置：rank=4-8, alpha=8-16, dropout=0.1-0.15，target 是 q_proj 和 v_proj（在 Long-Horizon Household 任务）或 all linear layers except lm_head/embed_tokens/vision encoder（在 Cupboard 任务）。

## 8. Long-Horizon Household Benchmark

基于 BEHAVIOR-1K（https://behavior.stanford.edu/ ），但 BEHAVIOR 原版任务"对了就能做完"，没专门 stress failure recovery。作者用 GPT-5（注释里写了 GPT-5，时间戳看起来是 2025 年后的工作）配合 scene graph 生成 4 类任务：

| 类别 | 失败模式 | 例子 |
|---|---|---|
| Fitting | 几何/容量/occlusion | 把小玩具先放，结果唯一大空间被占 |
| Selection | 探索不足 | 进了第一个房间拿了番茄，没看到隔壁的生菜 |
| Preparation | 序列依赖 | 没拿底座就想加 topping |
| Hybrid | 多种混合 | 单 episode 内多种失败模式 |

数据生成流程：GPT-5 写 task spec → BEHAVIOR OmniGibson 物理仿真验证（剔除不可能的 placement）→ 生成 (obs, action, reflection, score) 四元组做 SFT。

SFT 训练三个 LLM 共享一个 backbone，cross-task 学习。Evaluation 用 disjoint task descriptions/scenes/placements，零重叠。

**主结果 Table 1**（按平均成功率）：

| 方法 | Fitting | Selection | Preparation | Hybrid | Avg |
|---|---|---|---|---|---|
| Reflexion | 8.51 | 8.82 | 15.9 | 6.45 | 9.92 |
| Self-Refine | 10.6 | 11.8 | 12.7 | 9.68 | 11.20 |
| ReflectVLM | 2.12 | 5.88 | 14.3 | 6.45 | 7.19 |
| PPO | 0 | 2.94 | 7.94 | 3.23 | 3.53 |
| DreamerV3 | 4.26 | 11.8 | 11.1 | 12.9 | 10.02 |
| 3DLLM-Mem | 10.6 | 14.7 | 9.52 | 9.68 | 11.13 |
| **Ours** | **44.7** | **32.4** | **31.7** | **25.8** | **33.65** |

3 倍以上的提升。注意 PPO 在 Fitting 上 0%——这是因为 RL baseline 训练数据量跟 SFT 持平，但 PPO 的 exploration 在这种 sparse-reward long-horizon 任务上根本走不通。DreamerV3 也卡在 4.26%，因为 world model 是 frozen 的，遇到新场景就失效。

## 9. Cupboard Fitting Benchmark

为了 isolate 几何失败，作者在 MuJoCo 里做了一个可控 cupboard 环境：6-8 个隔间 + 6-10 个 colored 几何体 + Franka Panda arm，命令是自然语言"pick up red apple, put in green compartment"。

两个 metric：
- **fit rate**：物体成功放进任意隔间的比例
- **correct rate**：物体放进**正确**隔间（每个物体有指定的 target compartment）的比例

base model 是 Qwen2.5-VL-3B（https://qwenlm.github.io/blog/qwen2.5-vl/ ），3B 比 7B 小，正好用来验证 framework 在小模型上也 work。

**主结果 Figure 3**：

| 方法 | Fit | Correct |
|---|---|---|
| Vanilla | ~44.5 | ~10 |
| w/o RIA | 53.5 | ~15 |
| w/o ROA | 45.2 | ~12 |
| Full (Base-Weight) | 57.4 | ~22 |
| Full (LoRA) | **60.2** | **25.3** |

LoRA 反而比 Base-Weight 好（60.2 vs 57.4）！这有点反直觉但合理：test-time 训练数据极少（10-15 个 retro sample per iteration），全参数更新容易 overfit 到这 10-15 个 sample，LoRA 的低秩约束本身就是强 regularizer。这点和 LoRA 在 few-shot finetuning 上的优势一致（参考 https://arxiv.org/abs/2106.09685 ）。

## 10. Ablation 的关键发现：RIA 和 ROA 是 mutually dependent

最 striking 的 ablation 现象：

> "Sometimes removing just one component performs worse than removing both. For instance, without RIA, Preparation falls to 3.17% and Hybrid to 3.23%, while removing both components yields 11.1% and 12.9% respectively."

这是**反协同**——单独加一个反而比两个都不加更差。作者的解读：

- **RIA without ROA**：V_{φ_i} 是 SFT 训出来的，但它对当前环境的判断可能 systematic 偏差。没有 ROA 来纠正它，high-temperature 采样出的 N 个 candidate 都被同一个偏差打分，反而**过度自信地挑错**。Greedy 至少随机性高，有时候撞对。
- **ROA without RIA**：训练数据来自 greedy 选出的"差动作"，hindsight 后才知道差。但因为这些动作本身没暴露 environment affordance（比如没去探索别的房间），hindsight 也学不到什么有用信息。垃圾进垃圾出。

只有两个一起才形成 virtuous cycle：RIA 保证 sample 高质量 → ROA 学到的 reflection 有信息量 → V_{φ_i} 变强 → 下次 RIA 更准。这本质是个 **data quality flywheel**。

Loss ablation 也显示类似 dependency：去掉 internal loss 或 action loss 单独都比两个都去掉差。这呼应 double-loop learning 的核心：单 loop（只 update policy）是 single-loop learning，效果差；同时 update policy + value 是 double-loop，效果才好。

## 11. Compute-Matched Experiment

附录 B 做了非常重要的对照：给 vanilla baseline 3× steps（匹配 full model 的 3× wall-clock），看是不是"多花时间就行"。

Table 2 结果：

| 方法 | Avg |
|---|---|
| Vanilla | 8.79% |
| Vanilla (3× steps) | 8.46% |
| Full | 33.65% |

Vanilla 给 3 倍时间**反而略降**（8.79 → 8.46）。这证明 reflection 的价值**不是单纯多算**——多算只是让 agent 多犯几次同样的错。Reflection 改变的是 decision process 本身。这点很关键，回应了 LLM test-time scaling 文献里 "compute is all you need" 的简化叙事。

## 12. HM3D 泛化实验

附录 C 把训练只在 BEHAVIOR 上的模型，直接 zero-shot 丢到 HM3D（https://arxiv.org/abs/2109.08238 ）的 photorealistic scene 上跑 Preparation：

| 方法 | HM3D Prep |
|---|---|
| 3DLLM-Mem | 7.32% |
| Reflexion | 2.44% |
| PPO | 0% |
| ReflectVLM | 0% |
| **Ours** | **19.5%** |

绝对数字从 BEHAVIOR 的 31.7% 掉到 19.5%，掉了 12.2 pp，但仍保持相对优势。这表明 reflection 学到的不完全是 BEHAVIOR-specific 的知识，而是某种 **"如何从失败中恢复"的元能力**——这种能力能跨 domain transfer。我觉得这是这篇论文最有价值的发现之一：test-time adaptation 不只是 narrow fit，它学到的是 meta-skill。

## 13. Hyperparameter 分析

附录 D 的 ablation 很 informative，几个 take-aways：

**N（candidate 数）**：N=1（greedy）53.0% → N=6 峰值 60.0% → N=10 58.8%。N=6 是 sweet spot。超过 6 之后 candidate pool 里低质量样本变多，V_{φ_i} 的打分噪声上升。这呼应 Best-of-N 文献里的 N vs quality 曲线。

**Temperature T**：T=0 → 52.6%，T=1.25 峰值 60.0%，T=2.0 → 47.0%。Inverted U。低温 candidate 太像，V_{φ_i} 没区分度；高温 candidate 太乱，物理上不可行，V_{φ_i} 也救不回来。T=2.0 是 Long-Horizon Household 的设置，T=1.25 是 Cupboard 的——不同任务 optimal T 不同，可能跟 action space 离散度有关。

**LoRA rank**：(r=8, α=16) 峰值 60.0%；(r=32, α=32) 灾难性 collapse 到 34.8%——mode collapse，对所有输入输出同样的动作。这是 LoRA 在小数据 + 高 LR 下的经典 failure，工程上要警惕。论文特别强调 "overparameterized adapters converge to degenerate solutions that ignore input variations"。

**Action budget**：50 步最优，100 步反而下降（60.0 → 59.4）。这说明无限 action budget 让 agent 倾向于"探索而不 commit"，长 horizon 上累积错误。适度 budget 是个 implicit inductive bias，逼 agent 决策果断。

## 14. Single-Step vs Receding Horizon

附录 E 对比"单步生成 + retro" vs "5-step sequence planning + 执行第一步"。结果：

| 方法 | Fit | Compute |
|---|---|---|
| Receding Horizon | 57.8% | 5× |
| Single-Step | 60.2% | 1× |

Receding Horizon 多花 5 倍算力**还更差**！作者的解释很深刻：

> "Test-time training fundamentally relies on learning from actual execution feedback... However, generating 5-action sequences forces the model to predict these outcomes before they occur... creates optimization interference where gradients from test-time training (learned from reality) fight against the inductive bias from sequence generation (learned from imagination)."

也就是说，receding horizon 让模型在 imagination 里预先 commit 一个未来，但 test-time training 又用 reality 的反馈去推翻这个 imagination——两个目标冲突。Single-step 干脆不想象，让 retro 来做 implicit long-horizon reasoning：把 multi-step lookahead 蒸馏进 V_{φ_i} 的 single-step 评估里。

这点跟 Sutton 在 RL 里强调的 "model-based vs model-free trade-off" 有呼应：用 world model 做 long rollout 在 model 错的时候会 amplify error；不如 model-free + value function approximation，把 long-horizon 信息压进 value。

## 15. Real-Robot 验证

附录 I 在真实 Franka Panda 上验证 cupboard fitting。top-down 相机视角跟 sim 一致。每个动作后 binary execution feedback（物体在 target compartment 内 = success）作为 V_{φ_e} 的输入。Figure 5(b) 的 qualitative 显示，agent 把 reflection 学到的"先放小、后放大、留出大空间"的策略真的迁移到物理世界。

## 16. 我对这篇论文的几点 intuition

**直觉 1：verbal reflection 是 lossless compression of experience**。

每次失败，如果把 (obs, action, outcome) raw 存起来，下次 prompt 时 context 就爆炸。verbal reflection 把失败抽象成 "为什么"——一句"小车占了大空间唯一可放位置"足够让 LLM 下次注意。这是种 **symbolic distillation**，比 replay buffer 更 compact。

**直觉 2：double-loop learning 的数学对应**。

Single-loop：只 update π_θ 让它偏好高分动作——这是 RL。
Double-loop：同时 update V_{φ_i} 让它"知道哪些动作 retro 后会得低分"——这是 credit assignment 的 meta-update。
在 actor-critic 里 critic 也是被 update 的，但 critic update 来自 TD error，这里 critic update 来自 **hindsight re-evaluation**——后者更弱（只有 milestone 触发），但更 interpretable。

**直觉 3：test-time training 在 embodied 上的合理性**。

embodied 任务的特点是 deployment-time environment 跟 training environment 必然有 gap（每家厨房都不一样）。Train-time generalization 是上限，但 deployment adaptation 决定下限。Test-time training（TTA）文献（如 Tent https://arxiv.org/abs/2006.10926 ）用 entropy minimization 更新 BN params；这里用 self-generated verbal reflection 作为更丰富的 supervisory signal，是 TTA 在 embodied 上的延伸。

**直觉 4：N=4-6 候选 + T=1.25-2.0 的组合是 test-time scaling 的 sweet spot**。

这呼应 OpenAI o1、DeepSeek R1 类 reasoning model 的 test-time compute 分配思路（https://arxiv.org/abs/2408.03314 ）：不是越多越好，是要 generate diverse + verify。Verification 这里由 V_{φ_i} 完成。

**直觉 5：mode collapse 警示**。

LoRA rank 32 在小数据上 mode collapse 是个 useful negative result。提醒我们 test-time training 比 offline training 更脆弱——没大规模数据兜底，regularization 选择极关键。这点对实际部署 embodied agent 有直接工程意义。

## 17. 与现有方法的联系

- **Reflexion**（https://arxiv.org/abs/2303.11366 ）：只有 verbal reflection-as-context，没有参数更新。本论文可以理解为 "Reflexion + 参数化"。
- **Self-Refine**（https://arxiv.org/abs/2303.17651 ）：iterative critique refinement。本论文是 single-step decision + post-hoc reflection，不重复 refine 同一动作。
- **AlphaGo MCTS**（https://www.nature.com/articles/nature16961 ）：MCTS 用 value + policy network 做 lookahead。本论文没有 rollout，用 verbal anticipation 替代，是 LLM-era 的"verbal value function"。
- **DreamerV3**（https://arxiv.org/abs/2301.04104 ）：world model frozen 后没法 adapt。本论文 V_{φ_i} 可 update，是 "non-frozen verbal world model"。
- **Inner Monologue**（https://arxiv.org/abs/2207.05608 ）：robotics 上的早期 verbal reasoning 工作。本论文把它升级到有训练信号的 reflection。
- **LoRA-TTT**（https://arxiv.org/abs/2502.02069 ）：LoRA-based test-time training for VLM。本论文在 embodied 设定下用类似机制。

## 18. 局限与可延伸方向

论文自己也承认：
1. 只测了视觉模态，tactile / audio 没碰——但 framework 本身是 modality-agnostic 的，理论上加 multisensory（参考 MultiPLY https://arxiv.org/abs/2401.08547 ）就直接适配。
2. V_{φ_e} 冻结，意味着"裁判"不会随经验变强。真实人类反思是裁判也在成长的——未来可以让 V_{φ_e} 用 slower update（如 EMA）。
3. Test-time compute 3× 是个不小的 cost，real-time robot 控制上要权衡。但对 household 这种分钟级任务，3× 完全可接受。
4. Retro buffer 在超长 episode（>100 步）上怎么管理？论文提到 "subsample if necessary"，但没给具体策略。这是个可以做 ablation 的点。
5. Double-loop 的 meta-update 是 milestone-triggered（K=5 或 room transition）。能不能做成 online、连续的？比如用一个 small head 预测"现在的 retro 是否值得 trigger"。

可以想到的后续工作：
- 把 reflection 文本压缩成 structured representation（不是自然语言，而是 scene graph patch），降低 V_{φ_i} 上下文负担。
- 用 DPO 替代 REINFORCE，把 retro 高分 vs retro 低分动作做成 preference pair，可能更稳。
- 多 agent 协作反思：让 V_{φ_e} 是 ensemble，多个 critic 投票，减少 single model bias。
- 跟 RLHF 的 reward model 联动：把 retro score 当成训练 reward model 的数据，离线训练更强的 V_{φ_e}，下次部署又用上去。

## 19. 总结

这篇论文最让我欣赏的是它**把 Schön 的哲学框架真的可操作化**了——不是停留在"agent 应该反思"的口号，而是给出了：
- 三种 reflection 的精确定义与触发条件
- reflection 转换为 self-supervised signal 的两条路径（supervised for V_{φ_i}，REINFORCE for π_θ）
- retro 解决 long-horizon credit assignment 的工程实现
- 充分的 ablation 揭示 RIA/ROA 的 mutually dependency
- compute-matched 对照证明反思 vs 多算的区别
- 跨 domain (BEHAVIOR → HM3D) 的泛化证据
- 真实机器人的初步验证

它提出的 "verbal reflection as test-time training signal" 范式我认为有普遍意义——不只 embodied，agent 在 tool use、web navigation、code generation 上都可以套这个框架：执行 → 用 LLM 反思 → 把反思当训练数据 update 策略 + value。

参考链接汇总：
- 项目主页：https://reflective-test-time-planning.github.io/
- 代码：https://github.com/Reflective-Test-Time-Planning/Reflective-Test-Time-Planning
- LLaVA-3D：https://arxiv.org/abs/2505.00624
- BEHAVIOR-1K：https://behavior.stanford.edu/
- Reflexion：https://arxiv.org/abs/2303.11366
- Self-Refine：https://arxiv.org/abs/2303.17651
- RT-2：https://arxiv.org/abs/2307.15818
- OpenVLA：https://openvla.github.io/
- 3D-LLM：https://3d-llm.github.io/
- DreamerV3：https://arxiv.org/abs/2301.04104
- PPO：https://arxiv.org/abs/1707.06347
- Qwen2.5-VL：https://qwenlm.github.io/blog/qwen2.5-vl/
- LoRA：https://arxiv.org/abs/2106.09685
- Tent TTA：https://arxiv.org/abs/2006.10926
- Double-loop learning：https://hbr.org/1977/09/double-loop-learning-in-organizations
- Schön reflective practitioner：https://en.wikipedia.org/wiki/Donald_Sch%C3%B6n
- HM3D：https://arxiv.org/abs/2109.08238
- LoRA-TTT：https://arxiv.org/abs/2502.02069
- 3D-LLM-Mem：https://arxiv.org/abs/2505.22657
- AlphaGo：https://www.nature.com/articles/nature16961
- Self-Consistency：https://arxiv.org/abs/2203.11171
- Inner Monologue：https://arxiv.org/abs/2207.05608
- MultiPLY：https://arxiv.org/abs/2401.08547
- Test-time scaling (o1)：https://arxiv.org/abs/2408.03314
