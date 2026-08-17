---
source_pdf: VideoAuto-R1 VideoAutoReasoningviaThinking Once,AnsweringTwice.pdf
paper_sha256: 45673792d97e8e1c376160fed070715b419e4b6b2c0c810e96de72f4a8fd3d98
processed_at: '2026-08-13T00:52:23-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 VideoAuto-R1

Andrej你好！咱们换个更relaxed的方式聊这篇paper。

## 故事开头：一个反直觉的发现

Meta团队在做video reasoning的时候发现了一个挺"打脸"的事情。大家都在follow DeepSeek-R1那套CoT recipe，让model先think一大段再answer。结果他们拿来三个RL-trained video model（Video-R1、Time-R1、VideoChat-R1）做实验，对比两种inference方式：

- **Direct**: 直接给答案，不解释
- **CoT**: 先step-by-step think，再给答案

按理说CoT应该碾压direct对吧？结果Table 1的数据非常savage：

Video-R1 on Charades-STA，direct 42.0，CoT只有34.9，**掉了7.1个点**。Video-R1 on VideoMME，direct 64.6，CoT 64.3，也掉了。而且CoT平均要生成386个token，direct只要17个。

只有VideoMMMU这种math/physics-heavy的benchmark，CoT才有consistent的gain（大概+1到+3）。

**Intuition是这样的**：math problem是symbolic的、clean的，你确实需要step-by-step推导。但video understanding本质是个perception task——你看到一个人在stirring a pan，答案就是"stirring"，没什么好推导的。一旦visual perception对了，后面的reasoning非常shallow。强行让model写一大段"Let me analyze the video frame by frame..."，反而会让model在reasoning trace里hallucinate（Figure 7那个例子特别典型：VideoChat-R1把dancing moves描述错了，CoT答案错了，但direct答案反而对了）。

这和text/image domain的"overthinking"现象是同一个道理：
- "Do Not Think That Much for 2+3=?": https://arxiv.org/abs/2412.21187
- "Stop Overthinking" survey: https://arxiv.org/abs/2503.16419

## 核心idea：该想的时候想，不该想就别想

既然CoT不是万能的，那理想的video model应该**adaptively**决定要不要think。简单问题直接答，hard问题才reason。

但怎么做这个adaptive decision？之前AdaptThink（https://arxiv.org/abs/2505.13417）的做法是training-based：给每个sample打think/no-think label，训练一个mode-switching policy。这在text domain还行，到video就崩了——因为video里真正的"must-think"sample太少了，training signal太sparse，model容易mode collapse（要么always think要么never think）。

paper Table 7实测了reproduced AdaptThink，在VideoMME上think ratio只有1%（basically never think），MVBench上甚至比no-think baseline还差。

VideoAuto-R1换了个思路：**把"when to think"和"how to think"decouple**。

## Training：Thinking Once, Answering Twice

training的时候，每个response都遵循同一个template：

$$\boxed{a_1} \text{} \boxed{a_2}$$

翻译成人话：
1. 先给个initial answer $a_1$（short，通常<10 token）
2. 再写一段reasoning rationale $r$
3. 最后给个reviewed answer $a_2$

关键点：**两个answer都被supervise**，不需要per-sample think/no-think label。model学会的是"both直接答和reviewed答"，什么时候用哪个完全在inference time决定。

这个设计很巧妙——它avoid了mode-switching training的unstable问题，因为model永远在同一个format里learn，没有binary mode的label noise。

### Fallback mechanism

有个edge case：math-heavy的问题，model可能没法直接猜出 $a_1$。如果硬逼它猜，会得到low-confidence wrong answer，污染training signal。

paper的解法是designated一个fallback string `"Let's analyze the problem step by step."`。当model觉得没法immediate answer时，第一个box里填这个string，然后proceed to reasoning。这样：
- Output grammar保持一致（还是两个box + 一个think block）
- Early-exit mechanism在inference time保持unambiguous（fallback string直接force reasoning）
- Training reward能distinguish "honest defer"和"wild guess"

### Reward设计

用GRPO（DeepSeek-R1的RL framework，https://arxiv.org/abs/2501.12948），但reward是dual-answer的：

$$R = w_1 R_{\text{task}}^{(1)}(a_1) + w_2 R_{\text{task}}^{(2)}(a_2) + \lambda R_{\text{fmt}} + \alpha R_{\text{fallback}}$$

变量解释：
- $w_1 = 0.9$：initial answer的weight
- $w_2 = 1.1$：reviewed answer的weight（**$w_2 > w_1$ 很关键**）
- $\lambda = 1$：format reward weight
- $\alpha = 0.3$：fallback bonus weight
- $R_{\text{task}}^{(1)}, R_{\text{task}}^{(2)}$：两个answer各自的task correctness reward
- $R_{\text{fmt}}$：format check reward（regex enforce两个box + 一个think block）
- $R_{\text{fallback}} \in \{0, 1\}$：1 iff $a_1$是fallback string且 $a_2$ correct

**为什么 $w_2 > w_1$？** 想一个case：$a_1$对了但 $a_2$错了，vs $a_1$错了但 $a_2$对了。如果 $w_1 = w_2$，这两种case的total reward一样（都是1），model区分不出来。但我们希望prioritize reviewed answer——user如果allow thinking mode，期望final answer reliable。设 $w_2 > w_1$后，"wrong→correct"得1.1，"correct→wrong"只得0.9，明确incentive model去improve reviewed answer。

Table 9的ablation验证了这点：$w_1:w_2 = 1:1$时VideoMMMU 56.1，改成0.9:1.1变56.4，加fallback bonus $\alpha=0.3$变58.6。

### GRPO背景

快速回顾下GRPO（变量含义之前讲过，这里讲intuition）。给定prompt $q$，sample $G$个responses，计算每个的verifiable reward，然后用group statistics normalize：

$$A_i = \frac{r_i - \mu}{\sigma + \varepsilon}$$

- $r_i$：第 $i$个response的reward
- $\mu$：group mean reward
- $\sigma$：group standard deviation
- $\varepsilon$：numerical stability的小常数
- $A_i$：第 $i$个response的advantage

Objective是PPO-style的clipped importance ratio + KL penalty：

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^{G} \min(\rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i) + \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

- $\rho_i = \pi_\theta(o_i|q) / \pi_{\theta_{\text{old}}}(o_i|q)$：importance ratio between current policy $\pi_\theta$ 和behavior policy $\pi_{\theta_{\text{old}}}$
- $\epsilon$：clipping range
- $\beta = 0.01$：KL penalty strength
- $\pi_{\text{ref}}$：frozen reference policy

GRPO相比PPO的核心trick是去掉learned critic，用group statistics当baseline。对verifiable-reward task特别合适——answer对就是对、错就是错，不需要value function估计。

### Training data filtering

这个细节挺important的。Training pool 137K → filtered到83K：
- Text: 6.4K (DAPO-Math)
- Image: 27.5K (ViRL, ThinkLite-Hard)
- Video: 49.4K (Video-R1, TVBench, STI-Bench, MMR-VBench, Charades-STA, ActivityNet, Time-R1, NExT-GQA)

Filtering pipeline（Figure 5）：
1. 对每个sample用base model generate 8个responses（high temperature）
2. 用Qwen3-30B-A3B judge correctness
3. **去掉8/8 correct（too easy）和0/8 correct（too hard）的samples**

**Intuition**：GRPO的advantage是group-normalized的。如果8个rollout全对或全错，advantage接近0，gradient信号弱。这种difficulty filtering让training signal density最大化。Table 11验证：filtering让VideoMMMU从55.4→56.4，同时dataset size从138K→83K。

### 跳过SFT cold-start

paper做了个important choice：直接RL，不SFT。Appendix F.3的ablation（Table 17）显示SFT with Video-R1-CoT data反而让VideoMME从66.0掉到60.1。CoT SFT data质量不够好，会distort strong base model的behavior。Qwen2.5-VL已经pretrained on massive data，CoT format用system prompt enforce就够了，不需要SFT教。

R1-zero也有similar observation: https://arxiv.org/abs/2503.05132

## Inference：Confidence-Based Early Exit

training的时候永远answer→think→answer，inference的时候怎么decide要不要think？

paper的解法非常简洁：**看第一个answer的confidence**。

### Algorithm

1. Greedy decode直到第一个`\boxed{}`结束，拿到 $a_1 = (t_1, \dots, t_L)$
2. 计算length-normalized mean log probability：

$$s(a_1) = \frac{1}{L} \sum_{\ell=1}^{L} \log p_\theta(t_\ell | t_{<\ell}, q)$$

- $L$：$a_1$的token数
- $t_\ell$：第 $\ell$个token
- $p_\theta$：model的next-token distribution
- $s(a_1)$：confidence score

3. 如果 $a_1$是fallback string，设 $s(a_1) = -\infty$，force继续reasoning
4. 如果 $s(a_1) \geq \log \tau$（$\tau = 0.97$），early exit，accept $a_1$
5. 否则继续decode reasoning $r$ 和reviewed answer $a_2$

### Why this works

Table 8验证了confidence和thinking necessity的correlation：

| Benchmark | Avg confidence | Think ratio | Gain from thinking |
|---|---|---|---|
| MVBench (perception) | 0.948 | 25% | +0.1 |
| MMVU (perception) | 0.933 | 39% | +0.4 |
| VideoMMMU (reasoning) | 0.874 | 51% | +4.0 |

Perception-heavy的benchmark confidence高（>93%），think ratio低，thinking gain marginal。Reasoning-heavy的VideoMMMU confidence低（87%），think ratio高，thinking gain明显（+4.0）。

这和Liao et al. (2025)的finding一致：token-level confidence和answer correctness强相关。VideoAuto-R1直接leverage这个signal，不需要external calibrator。

### Threshold $\tau$ 的trade-off

Figure 3显示 $\tau$ 是个continuous knob：
- $\tau$ 高 → think ratio高 → accuracy高但latency高
- $\tau$ 低 → think ratio低 → accuracy可能掉但latency低

VideoMMMU上 $\tau$ 从0.86→0.98，accuracy从57.5→58.7，think ratio从29%→55%。
VideoMME上 $\tau$ 变化accuracy基本不变——perception task thinking的diminishing return。

Paper选 $\tau = 0.97$ 作为robust default，不需要per-dataset tuning。

## 实验：Accuracy + Efficiency双赢

### Video QA（Table 3）

VideoAuto-R1 (Qwen2.5-VL-7B) vs prior SOTA:
- VideoMME: 67.3 (+5.5 vs Video-R1, +2.1 vs VideoChat-R1.5)
- VideoMMMU: 58.6 (+3.9 vs baseline, +6.2 vs Video-R1)
- MVP: 39.4 (+6.4 vs Video-R1)
- **Average response length: 44 tokens**（vs Video-R1的386）

Think ratio分布很有意思：
- MVBench (perception): 25%
- VideoMMMU (reasoning): 51%
- MVP (reasoning): 44%

这exactly印证了"reason when necessary"的philosophy。

### Temporal grounding（Table 4）

- Charades-STA mIoU: 60.0 (+7.1 vs baseline)
- ActivityNet mIoU: 47.6 (+20.7!)
- NExT-GQA Acc: 80.6 (+27.3)

Grounding task上initial answer就sufficient了，CoT对localization improvement有限。paper在Appendix F.2分析：grounding不需要multi-step deduction，model直接map event到time span。Table 16显示first answer和second answer的mIoU完全一样，所以grounding task默认early exit。

### Image reasoning（Table 5）

虽然是为video设计的，但image reasoning也有提升：
- MathVista: 69.4 → 73.7
- MMMU-Pro: 36.1 → 39.8

这是因为training data里加了image-centric math data，dual-answer design自然transfer到static images。

### Ablation对比（Table 6）

| Training | Inference | Length | VideoMME | VideoMMMU |
|---|---|---|---|---|
| Baseline | Direct | 3.0 | 66.0 | 54.7 |
| SFT | Direct | 2.3 | 67.0 | 56.5 |
| RL without thinking | Direct | 2.5 | 66.0 | 54.4 |
| RL with thinking | CoT | 149 | 66.1 | 56.4 |
| **VideoAuto-R1** | Auto | **44** | **67.3** | **58.6** |

RL with thinking在perception task上gain有限（VideoMME +0.1）但token暴涨到149。VideoAuto-R1在VideoMMMU上比RL with thinking还高2.2个点——因为dual-answer supervision让both answers都optimized，而不是只supervise final answer。

## 为什么这个work：深层intuition

### 1. Decoupling的设计哲学

Traditional auto-thinking把"when to think"和"how to think"耦合在training里，需要per-sample mode label。VideoAuto-R1的decoupling：
- Training只学"how to think"（answer→think→answer format + dual reward）
- Inference只学"when to think"（confidence threshold）

这种separation of concerns让training更stable（no mode label noise），inference更controllable（threshold可调）。

### 2. Dual-answer作为self-verification

$a_1 \to r \to a_2$的structure本质上是一种in-context self-verification。Model先commit一个answer，然后通过reasoning verify/correct它。这比pure CoT（直接think→answer）多了一层anchor——initial answer给reasoning一个starting point，避免reasoning trace完全diverge。

Dual reward $w_2 > w_1$ ensure model不会把correct $a_1$改成wrong $a_2$（这种情况reward只有0.9而不是2.0）。

### 3. Fallback作为honest defer signal

Fallback mechanism让model学会"知道自己不知道"。与其wild guess，不如honest说"我需要reason一下"。这在epistemology上很sound——calibrated uncertainty是intelligent behavior的key component。

$\alpha$ reward让fallback比wrong guess更profitable，incentive model学会defer。

### 4. Confidence作为necessity signal

Token-level log probability本质上反映model的epistemic uncertainty。高confidence意味着model的knowledge已经sufficient，low confidence意味着需要更多computation。这和LLM test-time scaling的intuition一致——more compute应该allocated to harder problems。

参考Yue et al.的efficient R1 survey: https://arxiv.org/abs/2508.02120

## Limitations & Future directions

paper自己在Section G列了几个：
1. **Training没explicitly shape confidence distribution**：现在confidence只用在inference，没进training objective。Future work可以把 $p(a_1)$ 加进reward，让model学会calibrated confidence。
2. **纯language-based reasoning**：对perception error的correction有限。Future direction是"thinking with frames"——reasoning时revisit visual features。
3. **Benchmark limitation**：现有video reasoning benchmark不够challenging，需要更多long-range temporal / compositional logic / counterfactual reasoning的data。
4. **Must-think video data稀缺**：constructing大规模genuine reasoning-required video dataset是urgent方向。

## 和相关工作的positioning

- vs DeepSeek-R1 (https://arxiv.org/abs/2501.12948): 同样用GRPO + verifiable reward，但针对video domain的overthinking问题做了adaptive thinking
- vs AdaptThink (https://arxiv.org/abs/2505.13417): AdaptThink是training-based mode switching，VideoAuto-R1是inference-based，更stable
- vs R-4B (https://arxiv.org/abs/2508.21113): R-4B做image auto-thinking用bi-mode policy optimization，VideoAuto-R1用dual-answer format避免binary mode
- vs Video-R1 (https://arxiv.org/abs/2503.21776): Video-R1是always-thinking，VideoAuto-R1是reason-when-necessary
- vs Time-R1 (https://arxiv.org/abs/2503.13377): Time-R1把reasoning用到temporal grounding，VideoAuto-R1发现grounding其实不需要CoT

## 总结

VideoAuto-R1的story用一句话讲：**video understanding的CoT不是free lunch，得adaptive地用**。Technical contribution集中在三点：
1. Empirical finding：CoT对video often redundant甚至harmful
2. Thinking once answering twice的training paradigm，decouple when/how to think
3. Confidence-based early exit的inference strategy，simple但effective

这个work的spirit我觉得很对——不是blindly apply text domain的recipe到video，而是先understand domain characteristic再design method。这种domain-aware的research attitude值得follow。

Project page: https://ivul-kaust.github.io/projects/videoauto-r1

---

# VideoAuto-R1: 深度技术讲解

Hi Andrej! 这篇paper来自Meta AI / KAUST，core insight非常elegant，让我从底层intuition开始讲起。

## 1. Core Motivation: CoT 在 video 上其实"overkill"

### 1.1 关键empirical observation

paper开头就抛出一个反直觉的finding（Table 1）：对RL-trained video models（Video-R1, Time-R1, VideoChat-R1），**direct answering 经常matches甚至outperforms CoT**，同时生成token数从386降到17.6。

具体看Table 1的data：
- Video-R1 on VideoMME: Direct 64.6 vs CoT 64.3 (-0.3)，token从17.6 → 386
- Video-R1 on Charades-STA: Direct 42.0 vs CoT 34.9 (**-7.1**!)，CoT反而伤performance
- Time-R1 on VideoMME: Direct 65.9 vs CoT 63.8 (-2.1)
- VideoChat-R1 on VideoMME: Direct 65.7 vs CoT 63.9 (-1.8)

只有在VideoMMMU（symbolic/math-heavy）上CoT有consistent gain（+1.0 ~ +3.4）。

**Intuition**: video understanding和math不同。math是symbolic、noise-free，需要step-by-step deduction；video主要是perception task，一旦perception准确了，剩下的reasoning非常shallow。强行让model think verbose，反而触发"overthinking"——model在reasoning trace里hallucinate visual details，最终drag down final answer（Figure 7的failure case非常illustrative：VideoChat-R1把dancing moves描述错了，结果CoT answer错了但direct answer对了）。

参考overthinking现象：
- "Do Not Think That Much for 2+3=?": https://arxiv.org/abs/2412.21187
- "Stop Overthinking" survey: https://arxiv.org/abs/2503.16419

### 1.2 The cost side

CoT还有一个efficiency问题：autoregressive LLM的latency随token数linearly增长。386 tokens vs 17 tokens，差22倍latency。production deployment完全不可接受。

## 2. 核心idea：Thinking Once, Answering Twice

### 2.1 Design philosophy

paper的key insight是把"when to think"和"how to think"decouple：
- **training time**: 永远answer → think → answer，让model学会both直接回答和reviewed回答
- **inference time**: 用confidence score决定是否early exit

vs. AdaptThink (https://arxiv.org/abs/2505.13417) 那种training-based mode-switching：AdaptThink需要per-sample think/no-think labels，在video domain因为"must-think" samples稀缺导致training unstable、容易mode collapse。

paper在Table 7里做了对比：reproduced AdaptThink的training-based方法在VideoMME上think ratio只有1%（basically no-think），MVBench上甚至**underperforms** no-think baseline（70.5 vs 71.1）。而VideoAuto-R1 think ratio在VideoMMMU上达到51%，且always outperforms。

### 2.2 Output format

每个training response严格遵循：

$$
\boxed{a_1} \text{} \boxed{a_2}
$$

其中：
- $a_1$: 第一个short、verifiable answer（通常 <10 tokens）
- $r$: free-form reasoning rationale
- $a_2$: reviewed final answer

System prompt（Table 2）强制exactly两个`\boxed{}`和一个` block，无extra text。

### 2.3 Fallback tolerance mechanism

这是paper的一个subtle但critical的设计。对math/symbolic-heavy问题，model可能在没有intermediate reasoning的情况下猜不出 $a_1$。如果硬逼model给个guess，会得到low-confidence wrong answer，污染training signal。

paper的解法：指定一个fallback string `"Let's analyze the problem step by step."`。当model无法immediate answer时，它在第一个box里输出这个string，然后proceed to reasoning，最后给出correct $a_2$。这个mechanism有两个好处：
1. preserve output grammar（仍然两个box + 一个think block）
2. 让early-exit在inference time保持unambiguous（fallback string直接trigger reasoning）

## 3. Training: Dual-Answer GRPO

### 3.1 GRPO background

先回顾GRPO（DeepSeek-R1, https://arxiv.org/abs/2501.12948）。给定prompt $q$，behavior policy $\pi_{\theta_{\text{old}}}$ 采样 $G$ 个candidates $\{o_1, \dots, o_G\}$。对每个output计算verifiable reward $r_i$，然后用group-wise normalization得advantage：

$$A_i = \frac{r_i - \mu}{\sigma + \varepsilon}
$$

其中：
- $\mu$: group mean of rewards
- $\sigma$: group standard deviation
- $\varepsilon$: small constant for numerical stability

Importance ratio：

$$\rho_i = \frac{\pi_\theta(o_i | q)}{\pi_{\theta_{\text{old}}}(o_i | q)}
$$

Final objective：

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^{G} \min(\rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i) + \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
$$

变量：
- $G$: rollout size（paper里是16）
- $\rho_i$: importance sampling ratio between current policy $\pi_\theta$ 和behavior policy $\pi_{\theta_{\text{old}}}$
- $\epsilon$: PPO clipping range
- $\beta$: KL penalty strength（paper里0.01）
- $\pi_{\text{ref}}$: reference policy（usually frozen initial policy），防止policy drift太远

**Intuition**: GRPO相比PPO的核心trick是去掉learned critic，用group statistics当baseline。这对verifiable-reward task特别合适——answer对就是对、错就是错，不需要value function估计。

### 3.2 Dual-answer reward

这是paper的core contribution。在standard GRPO的 $R_i = w R_{\text{task}}(o_i) + \lambda R_{\text{fmt}}(o_i)$ 基础上，VideoAuto-R1把single answer拆成dual answer：

$$R = w_1 R_{\text{task}}^{(1)}(a_1) + w_2 R_{\text{task}}^{(2)}(a_2) + \lambda R_{\text{fmt}} + \alpha R_{\text{fallback}}
$$

变量：
- $w_1 = 0.9$: initial answer weight
- $w_2 = 1.1$: reviewed answer weight，**$w_2 > w_1$ 是critical**
- $\lambda = 1$: format reward weight
- $\alpha = 0.3$: fallback bonus weight
- $R_{\text{fallback}} \in \{0, 1\}$: 1 iff $a_1$ 是fallback string 且 $a_2$ correct

**Why $w_2 > w_1$?** 看Table 12的reward matrix：

| $a_1$ | $a_2$ | $w_1=w_2=1$ | $w_1=0.9, w_2=1.1$ |
|---|---|---|---|
| ✓ | ✗ | 1 | 0.9 |
| ✗ | ✓ | 1 | 1.1 |
| ✓ | ✓ | 2 | 2.0 |

如果 $w_1 = w_2$，model区分不出"correct→wrong"和"wrong→correct"。我们要prioritize reviewed answer的correctness——因为user如果allow thinking mode，期望final answer reliable。设 $w_1 < w_2$ 后，"wrong→correct"得1.1，"correct→wrong"只得0.9，incentive model去improve reviewed answer。

**Why fallback bonus $\alpha$?** 如果不加 $\alpha$，model区分不出"$a_1$是wrong guess"和"$a_1$是honest fallback string"。Fallback是honest defer，应该reward而不是penalize。加 $\alpha=0.3$ 后，"fallback + correct $a_2$"得 $0.9 \cdot 0 + 1.1 \cdot 1 + 1 \cdot 1 + 0.3 \cdot 1 = 2.4$，而"wrong guess + correct $a_2$"只得 $0 + 1.1 + 1 + 0 = 2.1$，明确prefer honest defer over wild guess。

Table 9的ablation证实这点：
- $w_1:w_2 = 1:1$, no $\alpha$: VideoMMMU 56.1
- $w_1:w_2 = 0.9:1.1$, no $\alpha$: VideoMMMU 56.4
- $w_1:w_2 = 0.9:1.1$, $\alpha=0.3$: **VideoMMMU 58.6**

### 3.3 Task-specific rewards

paper考虑3种task type（Appendix B）：

**QA**: $R_{\text{QA}}(o_i) \in \{0, 1\}$，用math-verify（math problems）或normalized string comparison。

**Temporal grounding**: ground-truth segments $\mathcal{G} = \{[s_j, e_j]\}_j$，predicted $\hat{\mathcal{G}} = \{[\hat{s}_k, \hat{e}_k]\}_k$，取best matching pair的tIoU：

$$R_{\text{TG}}(o_i) = \max_{[\hat{s}, \hat{e}] \in \hat{\mathcal{G}}, [s, e] \in \mathcal{G}} \text{tIoU}([\hat{s}, \hat{e}], [s, e]) \in [0, 1]
$$

其中 tIoU = $|[\hat{s}, \hat{e}] \cap [s, e]| / |[\hat{s}, \hat{e}] \cup [s, e]|$ 是temporal intersection-over-union。

**Grounding QA**: $R_{\text{GQA}}(o_i) = R_{\text{QA}}(o_i) + R_{\text{TG}}(o_i) \in [0, 2]$，既要answer对又要localization准。

**Format reward**: $R_{\text{fmt}}(o_i) \in \{0, 1\}$，严格regex check，要求exactly两个`\boxed{}` + 一个``。

### 3.4 Training data & filtering

Training pool 137K → filtered到83K（Table 10）：
- Text: 6.4K (DAPO-Math, https://arxiv.org/abs/2503.14476)
- Image: 27.5K (ViRL, ThinkLite-Hard)
- Video: 49.4K (Video-R1, TVBench, STI-Bench, MMR-VBench, Charades-STA, ActivityNet, Time-R1, NExT-GQA)

**Filtering pipeline**（Figure 5）非常important：
1. 去掉invalid ground-truth samples
2. 对每个sample用base model（Qwen2.5-VL-7B）generate 8 responses with high temperature
3. 用smaller LLM（Qwen3-30B-A3B）judge每个response correctness
4. **去掉8/8 correct（too easy）和 0/8 correct（too hard）的samples**

**Intuition**: GRPO的advantage是group-normalized，对all-correct或all-wrong samples，advantage都接近0，gradient signal弱。这种difficulty filtering让training signal density最大化。

Table 11的ablation验证：filtering在text+image+video配置下，VideoMME从65.4→66.1，VideoMMMU从55.4→56.4，同时dataset size从138K→83K，efficiency↑。

### 3.5 Direct RL without cold-start SFT

paper做了一个important choice：**跳过SFT cold-start，直接RL**。

Appendix F.3的ablation（Table 17）：
- Qwen2.5-VL baseline: VideoMME 66.0, VideoMMMU 54.7
- SFT with Video-R1-CoT data: 60.1, 53.8 (**worse than baseline!**)
- RL with thinking (direct): 66.1, 56.4
- SFT → RL with thinking: 61.7, 53.5 (SFT污染了initialization)

**Intuition**: low-quality CoT SFT会distort strong base model的behavior。Qwen2.5-VL已经pretrained on massive data，CoT format可以用system prompt enforce，不需要SFT教。SFT反而引入了distribution shift。

参考R1-zero的similar observation: https://arxiv.org/abs/2503.05132

### 3.6 Training hyperparameters

- Base model: Qwen2.5-VL-7B-Instruct, Qwen3-VL-8B-Instruct
- Max video tokens: 4,096 (Qwen2.5-VL) / 128K (Qwen3-VL)
- Max frames: 256
- Optimizer: AdamW, lr $1 \times 10^{-6}$, weight decay 0.01, max grad norm 1.0
- Constant LR schedule, no warmup
- $\beta = 0.01$ (KL penalty)
- $w_1 = 0.9, w_2 = 1.1, \lambda = 1, \alpha = 0.3$
- Global batch size 256, 1 epoch
- Visual encoder frozen, projector + LLM fine-tuned
- DeepSpeed (https://arxiv.org/abs/2007.00072) + vLLM (https://arxiv.org/abs/2309.06180) acceleration
- GRPO rollout $G=16$, temperature 1.0
- **32 H100 GPUs, ~35 hours**

## 4. Inference: Confidence-Based Early Exit

### 4.1 Algorithm

Algorithm 1的完整流程：

1. 对input $(v, q)$ greedy decode直到第一个`
