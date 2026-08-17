---
source_pdf: Unified Multimodal Chain-of-Thought Reward Model.pdf
paper_sha256: 067661563c2be49548feb3c3f07c156cdc12c0d868b2019715a89092b4e58bde
processed_at: '2026-08-12T19:38:34-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# UnifiedReward-Think 人话版

## 一、这paper到底在干嘛

想象你在当摄影比赛评委，面对两张照片要打分。你脑子里会怎么想？大概会拆成几个维度——构图、光线、主题契合度、色彩，每个维度给个分，最后综合判断。这paper就是想让reward model也这么干。

当前multimodal reward models有个尴尬处境。大多数RM（像LLaVA-Critic、VisionReward、UnifiedReward这些）直接吐一个score或者pairwise ranking出来。你问它为啥给这个分？它没法说清。有些RM会generate一段brief justification，但那种justification常常很表面，甚至reasoning process本身有flaw，导致final answer也错了。

作者的intuition很simple：**如果reward signal本身是经过deep reasoning得出的，那这个signal就更reliable**。而且更妙的是，一旦模型真的internalize了这种structured thinking，即使你让它直接给答案别废话，它内部也会"偷偷"reason一遍再output，这就是所谓的implicit reasoning。

这paper就干了一件事：**把long Chain-of-Thought reasoning塞进multimodal reward model里**，而且是unified的——image generation、video generation、image understanding、video understanding全cover。

Reference: 类似思想在DeepSeek-R1 [Guo et al., 2025, arXiv:2501.12948](https://arxiv.org/abs/2501.12948) 里已经用在math/code上，这paper是把它搬到multimodal reward modeling这个新domain。

## 二、三阶段Training Pipeline——用大白话拆解

整个training像教一个聪明但没见过reward task的学生去做judge。三步走：

### Stage 1: Cold Start——先学会"说话的格式"

**问题**：VLM（比如基于LLaVA-OneVision或Qwen2.5-VL的UnifiedReward）pre-training时没接触过reward modeling这个task format。你让它做CoT reward reasoning，它不知道输出长什么样。

**解决**：从image generation preference data里随机抽10K samples，喂给GPT-4o蒸馏出CoT reasoning trace。关键filter——只留final answer跟ground truth label一致的那些，最终筛出5K high-quality samples，叫ImageGen-CoT-Reward-5K。

**训练目标**就是standard next-token prediction：

$$\mathcal{L}_{cold\_start}(\theta) = -\sum_{i=1}^{T} \log p(\mathbf{y}_i \mid \mathbf{x}, \mathbf{y}_{<i}; \theta)$$

变量含义：
- $\theta$：reward model的参数（就是你要train的那些weights）
- $\mathbf{x}$：input，包括image pair、caption、instruction
- $\mathbf{y} = \{y_1, \dots, y_T\}$：蒸馏出来的CoT output，一共$T$个token
- $\mathbf{y}_{<i}$：position $i$之前的所有token，autoregressive的条件
- $p(\cdot)$：模型在当前参数下生成某个token的conditional probability
- 负号 + log：这是standard cross-entropy loss，最大化log-likelihood等价于minimize这个loss

**为什么只用image generation数据就够**：作者的insight很巧妙。video本质上是multi-image的temporal extension，image understanding跟image generation在input-output structure上unified（都是visual content + text → reward judgment）。所以学会image generation的CoT format之后，video和understanding任务能通过model的prior knowledge自然generalize过去。Appendix A专门解释了这点，Table 7还做了ablation——用Qwen2.5-VL-72b代替GPT-4o蒸馏，结果几乎一样（VLReward 73.2 vs 73.8），说明cold start主要学format而非knowledge。

### Stage 2: Rejection Sampling——放大正确的reasoning pattern

**问题**：Cold start之后model会了format，但只见过5K image generation样本。需要scale到large-scale unified multimodal preference data上。

**数据准备**：
- Image Generation: HPD (25.6K) + OIP (7.4K) + EvalMuse (3K) + OpenAI-4o_t2i (6.7K) ≈ 43K
- Video Generation: VideoDPO (10K) + Text2Video-Human (5.7K) ≈ 16K
- Image Understanding: LLaVA-Critic-113K采样30K
- Video Understanding: ShareGPTVideo-DPO (17K)

**做法**：让cold-started model对每个input generate一条CoT reasoning。只留final answer跟GT label match的samples做SFT。相当于让model自己"投票"，只把答对的留下。

**为什么不能直接上GRPO**：GRPO每个input要sample N=8条responses来算relative advantage，计算开销巨大。如果model已经能答对的简单case也丢进去，纯属浪费compute。Rejection sampling先把model已经mastered的case reinforce掉，剩下的hard cases才喂给GRPO做exploration。

**Training objective**：跟Stage 1一样，Eqn. 1的形式，但用的是filtered data $(\mathbf{x}', \mathbf{y}')$。

**Ablation验证**：Table 5显示去掉rejection sampling，VLReward从73.8掉到70.4，GenAI-Video从82.3掉到79.2。说明这stage在shaping correct reasoning distribution上确实有用。

### Stage 3: GRPO——让模型在hard cases上explore

**问题**：Rejection sampling filter完之后，剩下一批model还答不对的samples。这些samples往往reasoning pattern更complex，model还没mastered。直接SFT没办法，因为这些case model自己生成的reasoning就是错的，你没法self-distill。

**方案**：用Group Relative Policy Optimization，让model探索diverse reasoning paths，用verifiable reward来guide优化。

#### Verifiable Rewards——rule-based的reward信号

两个reward，都是binary的：

**Format reward** $R_{fmt}$：检查output里有没有`⟨think⟩`和`⟨answer⟩`两个tag，有就1，没有就0。这是safety check，防止model偶尔format出错。

**Accuracy reward** $R_{acc}$：

$$R_{acc} = \begin{cases} 1 & \text{if } o = \text{ground truth} \\ 0 & \text{otherwise} \end{cases}$$

$o$是`⟨answer⟩` tag里的final answer（比如"Image 2 is better"），跟GT精确match才给1。

**Total reward**：$R = R_{fmt} + R_{acc}$

这里有个很关键的design——**只对final answer做verification，不对中间reasoning steps做监督**。为啥能trust这个？因为作者设计了multidimensional scoring strategy（下一节讲），让final answer必须从各个维度的score中aggregated出来，这就在结构上强制了reasoning和final answer的一致性。

#### GRPO的objective

给定input $\mathbf{x}$，从old policy $\pi_{\theta_{old}}$ sample N=8条responses $\{o^{(1)}, \dots, o^{(N)}\}$，每条算reward $\{R^{(1)}, \dots, R^{(N)}\}$。

**Advantage**（标准化reward，衡量relative quality）：

$$\hat{A}^{(i)} = \frac{R^{(i)} - \text{mean}(\{R^{(1)}, \dots, R^{(N)}\})}{\text{std}(\{R^{(1)}, \dots, R^{(N)}\})}$$

含义：第$i$条response比group里平均好多少（normalized）。用mean减、用std除是为了standardization，让advantage在different groups间comparable。

**Importance ratio**（衡量policy update magnitude）：

$$r^{(i)} = \frac{\pi_{\theta_{new}}(o^{(i)} \mid \mathbf{x})}{\pi_{\theta_{old}}(o^{(i)} \mid \mathbf{x})}$$

分子是新policy生成这条response的概率，分母是旧policy的。如果ratio=1说明new policy跟old没区别；ratio>1说明new policy更倾向生成这条response。

**Clipping**：把ratio限制在$[1-\delta, 1+\delta]$区间内，防止update太激进导致policy collapse。这跟PPO [Schulman et al., 2017, arXiv:1707.06347](https://arxiv.org/abs/1707.06347) 一样。

**Final objective**：

$$\mathcal{L}_{grpo}(\theta) = \mathbb{E}_{\mathbf{x} \sim \mathcal{X}, o^{(i)} \sim \pi_{\theta_{old}}}\left[\min\left(r^{(i)}\hat{A}^{(i)}, \text{clip}(r^{(i)}, 1-\delta, 1+\delta)\hat{A}^{(i)}\right) - \beta \cdot D_{KL}(\pi_{\theta_{new}} \parallel \pi_{ref})\right]$$

逐项解释：
- $\mathcal{X}$：训练集inputs的集合
- $\min(\cdot, \cdot)$：取ratio和clipped ratio的较小值，这是PPO的标准trick，确保 clipped objective是unclipped objective的下界（pessimistic bound）
- $\beta = 0.04$：KL penalty的权重，training details里写的
- $D_{KL}(\pi_{\theta_{new}} \parallel \pi_{ref})$：new policy跟reference policy（通常是training开始时的policy）的KL divergence，防止new policy drift太远
- $\pi_{ref}$：fixed reference model，不更新

直觉上：GRPO让model在sampled responses里学，advantage > 0的response（比平均好）会被 reinforce，advantage < 0的（比平均差）会被penalize。通过explore multiple paths + verifiable reward feedback，model逐渐收敛到correct reasoning trajectory。

Reference: GRPO本身来自DeepSeek-Math [Shao et al., 2024, arXiv:2402.03300](https://arxiv.org/abs/2402.03300)，在DeepSeek-R1里大放异彩。这paper是首次把它用在multimodal reward modeling上。

## 三、Multidimensional CoT Scoring——解决reasoning-answer inconsistency

这是个很巧妙的design，解决了一个常见问题。

**问题**：CoT reasoning有个well-known failure mode——reasoning steps看起来plausible，但final answer是靠shortcut/intuition跳出来的，跟reasoning process脱节。这种情况即使final answer对了，reasoning其实不可靠，拿来做self-distillation或者RL supervision会带来noisy signal。

**解决方案**：强制structured multi-dimensional scoring。看Figure 1的例子，对image generation task，model要按3个维度分别打分：
1. **Semantic consistency**：image跟caption在semantic上匹不匹配（7/10, 9/10）
2. **Aesthetics**：构图、光线、color palette好不好（8/10, 9/10）
3. **Authenticity**：realistic程度（6/10, 9/10）

然后aggregate成total score：Image 1 = 7+8+6 = 21，Image 2 = 9+9+9 = 27。Image 2 better。

这个design的妙处在于：**final decision必须从dimension-wise scores中mathematically derive出来**。你没法"reasoning说一堆好话但最后突然说Image 1好"——因为如果各维度score都倾向Image 2，aggregated结果自然就是Image 2。这就在结构上强制了reasoning process和final answer的consistency。

不同task用不同dimensions：
- Image generation: semantic consistency, aesthetics, authenticity
- Video generation: semantic consistency, temporal coherence, authenticity
- Image/Video understanding: semantic accuracy, factual correctness, clarity

这样有个training上的好处：你只需要verify final answer是否correct就能trust整条reasoning chain，因为structure已经隐式保证了一致性。所以在rejection sampling和GRPO里，只对final answer做监督就够了，这大大简化了reward design。

## 四、实验数据表深度解析

### Table 1: Image Understanding Assessment (VLRewardBench)

| Models | General | Hallu. | Reason. | Overall Accuracy | Macro Accuracy |
|--------|---------|--------|---------|-----------------|----------------|
| Gemini-1.5-Pro | 50.8 | 72.5 | 64.2 | 67.2 | 62.5 |
| GPT-4o | 49.1 | 67.6 | 70.5 | 65.8 | 62.4 |
| LLaVA-Critic | 47.4 | 38.5 | 53.8 | 46.9 | 46.6 |
| UnifiedReward | 76.5 | 58.1 | 65.1 | 67.5 | 66.6 |
| Ours (w/o CoT) | 77.9 | 70.5 | 65.4 | 73.1 | 71.3 |
| Ours | 78.1 | 72.7 | 66.0 | 73.8 | 72.3 |

**关键observations**：

1. **Hallucination维度提升最dramatic**：UnifiedReward只有58.1，加了CoT后跳到72.7，绝对提升14.6个点。这intuitively说得通——hallucination detection需要step-by-step verification（image里到底有没有这个object？caption描述的细节对不对？），shallow reasoning很容易miss。

2. **Reasoning维度提升相对小**：65.1 → 66.0。这task本身就难，CoT能help但天花板有限。

3. **w/o CoT vs with CoT**：这组对比是paper最amazing的claim。即使不输出CoT trace（w/o CoT），只要模型经过CoT训练，implicit reasoning就让Overall Accuracy从67.5（baseline）升到73.1。这说明CoT能力internalized之后，model的direct response也变强了。这是对"CoT only helps inference-time compute"观点的有力反驳——CoT训练能改变model的latent representation。

4. **vs GPT-4o**：有趣的是GPT-4o在Reasoning维度（70.5）比这篇paper的方法（66.0）还高，但在Hallucination（67.6 vs 72.7）和Overall（65.8 vs 73.8）上输了。说明specialized RM with CoT在某些维度能beat general-purpose VLM。

### Table 2: Image & Video Generation Assessment

| Method | GenAI-Bench Image (diff) | Method | GenAI-Bench Video (diff) | VideoGen-Reward (diff) |
|--------|--------------------------|--------|--------------------------|------------------------|
| PickScore | 67.2 | VideoScore | 70.6 | 49.9 |
| HPSv2 | 68.4 | LiFT | 60.1 | 58.3 |
| ImageReward | 65.0 | VisionReward | 73.1 | 68.2 |
| VisionReward | 66.4 | VideoReward | 73.3 | 73.9 |
| UnifiedReward | 70.9 | UnifiedReward | 77.2 | 79.3 |
| Ours (w/o CoT) | 71.9 | Ours (w/o CoT) | 81.6 | 79.9 |
| Ours | 72.5 | Ours | 82.3 | 80.5 |

这里作者用"diff" metric（排除tie pairs）来report，因为paper的方法intentionally不handle tie case。Appendix F解释了——他们专注于discriminative ability，即区分better vs worse，tie case不是target scenario。

**Observations**：
1. Video generation task上improvement最明显：UnifiedReward 77.2 → Ours 82.3，+5.1 absolute。
2. "tau" metric（包含tie）上有些baseline反而更高（比如VideoReward tau=57.4 vs Ours tau=57.8，几乎打平），但diff metric上Ours明显领先。这印证了作者说的——他们method在discriminative scenarios更强，不擅长tie。
3. w/o CoT again表现出色，consistent with implicit reasoning hypothesis。

### Table 3 & 4: Ablation Studies

最有信息量的ablation，每个stage的contribution：

**Table 3 (Image Understanding)**:
- UnifiedReward baseline: 67.5
- + GRPO (w/o CoT): 69.0（只+1.5，说明无CoT的GRPO效果有限）
- + cold start: 66.9（actually下降了！cold start只学format，knowledge没boost）
- + rejection sampling: 72.1（+5.2，大幅提升，说明correct pattern reinforcement重要）
- + GRPO (Ours): 73.8（最终，再+1.7）

**关键insight**：Cold start单独不仅没提升反而略降，这paper的解读是——cold start只教format，但5K数据不够覆盖复杂场景，反而可能让model过度依赖distilled pattern。但cold start是必要的prerequisite，没有它后面stage无从开始。

**Table 4 (Generation)**:
类似pattern。+ GRPO (w/o CoT)在image generation上甚至比baseline还低（54.1 vs 54.8 in tau），说明没有CoT structure的GRPO在generation task上几乎无效。Video generation上w/o CoT GRPO有微弱提升（77.2 → 78.4），但远不如with CoT的82.3。

**这组ablation最重要的结论**：**CoT reasoning structure是GRPO能work的关键**。没有structured reasoning，GRPO只能reinforce final answer，model学不到underlying reasoning process。这跟DeepSeek-R1的发现一致——RL需要structured output space才能explore effectively。

Reference: DeepSeek-R1-Zero showed that RL alone without SFT can elicit reasoning, but in multimodal domain, cold start helps format learning significantly. See [Guo et al., 2025](https://arxiv.org/abs/2501.12948) and follow-up discussion at [Visual-RFT, Liu et al., 2025b, arXiv:2503.01785](https://arxiv.org/abs/2503.01785).

### Table 6: 不同backbone的robustness

| Backbone | GenAI Image | GenAI Video | VLRewardBench Overall |
|----------|-------------|-------------|----------------------|
| LLaVA-OneVision-7B | 72.5 | 82.3 | 73.8 |
| Qwen2.5-VL-7B | 76.6 | 84.0 | 75.9 |

Qwen2.5-VL更强，所以最终性能更高，consistent improvement across tasks。说明这method是backbone-agnostic的，不依赖某个specific VLM architecture。

### Table 7: 蒸馏model对cold start的影响

| Distillation source | VLReward | GenAI-Image | GenAI-Video | VideoGen |
|---------------------|----------|-------------|-------------|----------|
| GPT-4o | 73.8 | 72.5 | 82.3 | 80.5 |
| Qwen2.5-VL-72b | 73.2 | 73.6 | 81.7 | 80.2 |

差距很小！这验证了作者的claim——cold start主要学format而非knowledge。即使从更弱的model蒸馏，只要format对了，后续rejection sampling + GRPO能弥补knowledge gap。这是个很practical的finding，因为GPT-4o蒸馏有成本和license问题。

## 五、一些intuitive的联想

### 1. 为什么implicit reasoning能work

这个现象在认知科学里有对应概念——**System 1 vs System 2 thinking**（Kahneman的框架）。Explicit CoT是System 2，slow但deliberate；direct response是System 1，fast但可能shallow。这paper的发现是：训练System 2能力能让System 1也变强。

技术上的解释可能是：CoT training改变了model的latent representation。通过step-by-step reasoning的梯度反传，model在latent space里学到了更structured的feature representation。即使inference时不输出reasoning trace，这些latent representation已经encode了reasoning process，所以direct response也benefit。

类似的phenomenon在[Snell et al., 2024, arXiv:2408.03314](https://arxiv.org/abs/2408.03314)里讨论过——test-time compute和model parameters之间有trade-off，而CoT training能改善这个trade-off frontier。

### 2. Verifiable reward的设计哲学

这paper的verifiable reward很minimal——只有format check和final answer match。为什么这么简单就够？

深层原因是**multidimensional scoring structure提供了implicit process supervision**。Model必须先对各维度打分，再aggregate，这强制了reasoning chain的internal consistency。如果reasoning错了，aggregated final answer自然也错，accuracy reward就是0。所以accuracy reward其实indirectly supervises了整个reasoning process，不用显式对每个step打分。

这跟process reward model (PRM) [Lightman et al., 2023, "Let's Verify Step by Step", arXiv:2305.20050](https://arxiv.org/abs/2305.20050) 的思路不同。PRM要对每一步打分，annotation成本高。这paper用structured output design来bypass这个需求，更scalable。

### 3. 跟RLHF/DPO的关系

传统RLHF [Ouyang et al., 2022, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58804a9791bb-Abstract-Conference.html) 用一个单独的RM来给policy model提供reward signal。DPO [Rafailov et al., 2023, NeurIPS 2023](https://arxiv.org/abs/2305.18290) 则直接从preference data里学policy，绕过explicit RM。

这paper的位置在RM这一侧——它改进的是RM本身，让RM更reliable。更好的RM可以喂给RLHF，也可以用于DPO variants。最近还有PrefGRPO [Wang et al., 2025c, arXiv:2508.20751](https://arxiv.org/abs/2508.20751) 这种把pairwise preference直接塞进GRPO的工作，跟这paper的direction有些overlap但不同——PrefGRPO是image generation policy optimization，这paper是reward model本身。

### 4. Limitations和future work

作者自己提了两个limitation：
1. **Inference time增加**：CoT output长，inference慢。但implicit reasoning的发现部分mitigate这点——不需要CoT时也能受益。
2. **RL不能fundamentally extend capability**：引用了[Yue et al., 2025, arXiv:2504.13837](https://arxiv.org/abs/2504.13837)的研究，RL只能amplify SFT已经acquire的potential。所以如果想push boundary，scale up high-quality CoT supervision还是关键。

我自己的额外思考：

1. **Multidimensional scoring的dimensions是手工设计的**。Image generation用semantic/aesthetics/authenticity，video用temporal coherence替代aesthetics。这些dimension choice很task-specific，可能有bias。如果能learned dimensions就更elegant，类似Aspect-Based Sentiment Analysis的aspect extraction。

2. **Pairwise comparison only**。目前只做pairwise（A vs B谁好），没有pointwise scoring（给A单独打分）。Pairwise的好处是annotation容易（人类比较更reliable than absolute rating），但downside是没法直接做best-of-N sampling的calibrated score。Reference: RLHF里pointwise和pairwise的trade-off在[Stiennon et al., 2020, arXiv:2009.01325](https://arxiv.org/abs/2009.01325)有讨论。

3. **Cold start只从image generation蒸馏**。虽然证明够用，但可能限制了format的generality。如果从multi-task蒸馏可能format更rich，但成本更高。这是个trade-off。

4. **N=8 in GRPO**：作者用的N=8，sample数量相对小。增大N可能让advantage estimation更stable，但compute成倍增长。Future work可能探索adaptive N或者variance reduction techniques。

5. **没有human evaluation on reasoning quality**。Paper验证了final answer accuracy提升，但没有human study验证CoT reasoning本身是否真的make sense。有可能model学到"fake reasoning"——reasoning trace看起来对但其实没用到。不过multidimensional scoring structure应该部分mitigate这个风险。

## 六、Implementation细节

从paper的training details section提取：

**Cold start和Rejection sampling**:
- Batch size: 1 per GPU
- Gradient accumulation steps: 16
- Learning rate: 2.5e-6
- Warm-up ratio: 0.3
- Hardware: 8× NVIDIA H100 (80GB)

**GRPO**:
- Batch size: 1 per GPU
- Gradient accumulation steps: 1
- Learning rate: 1e-6
- KL penalty coefficient $\beta = 0.04$
- N (sampled responses per input): 8
- Hardware: 64× NVIDIA H20 (97GB)

注意GRPO用了64块H20，远多于前两stage的8块H100。这是因为GRPO的compute intensive——每个input要sample 8条responses，还要算KL divergence，memory和compute都大。

Reference: H20是H100的export-compliant版本，memory bandwidth和compute都lower一些，但显存大（97GB vs 80GB），适合large batch sampling。选择H20可能跟export control有关，或者因为需要更大显存装policy + reference model + sampled responses。

## 七、一句话总结

这paper的核心contribution是**第一次把long CoT reasoning引入multimodal reward modeling**，通过三stage pipeline（cold start学format → rejection sampling generalize → GRPO explore hard cases）让RM学会multi-dimensional step-by-step reasoning，最终不仅explicit CoT让reward signal更reliable，还意外发现implicit reasoning让direct response也变强。

Method本身不算fundamentally novel（CoT + GRPO + rejection sampling都是existing techniques），但combination和application到multimodal reward modeling是first attempt，而且实验结果很有说服力。特别interesting的发现是implicit reasoning phenomenon，这给test-time compute和training-time compute的关系提供了新insight。

希望这些细节能帮你build intuition。如果想深入某个具体aspect，比如GRPO的gradient analysis、multidimensional scoring的information-theoretic interpretation、或者跟其他RL methods的theoretical comparison，可以继续聊。

---

# UnifiedReward-Think 深度解析

## 一、核心问题与Motivation

当前multimodal reward models存在一个fundamental tension：它们要么直接输出score/ranking（缺乏interpretability），要么产生shallow reasoning（容易flawed）。作者的key insight在于：**reward signal的accuracy本质上依赖reasoning的depth**。这就像一个grading rubric——如果teacher只给final score，学生无法理解为什么；如果teacher给brief comment，可能miss关键细节；只有multi-dimensional step-by-step analysis才能提供trustworthy signal。

作者提出三个hypothesis，这是整个paper的logical backbone：

1. **Explicit long CoT → reward reliability**：通过structured multi-step reasoning，可以分解complex judgment为verifiable sub-judgments
2. **Internalized CoT → implicit reasoning**：一旦模型学会CoT，这种structured thinking会"沉淀"为latent capability，即使不输出reasoning trace也能提升direct response
3. **VLMs already have latent reasoning ability**：只需要合适的training strategy来elicit，类似DeepSeek-R1的insight

## 二、Method Pipeline 架构解析

整个pipeline分为三stage，逻辑递进：

### Stage 1: Cold Start (Format Learning)

**目标**：教模型long CoT的format和structure，而non-task-specific knowledge。

**数据构建**：从HPD、EvalMuse、OIP中随机采样10K image generation preference pairs，喂给GPT-4o蒸馏reasoning trace。关键filter：只保留final answer与GT label一致的samples，最终得到5K high-quality CoT samples (ImageGen-CoT-Reward-5K)。

**Loss function**：

$$\mathcal{L}_{cold\_start}(\theta) = -\sum_{i=1}^{T} \log p(\mathbf{y}_i \mid \mathbf{x}, \mathbf{y}_{<i}; \theta)$$

变量解释：
- $\theta$：reward model参数
- $\mathbf{x}$：input (image pair + caption + instruction)
- $\mathbf{y} = \{y_1, y_2, \dots, y_T\}$：distilled CoT output，长度为$T$的token序列
- $\mathbf{y}_{<i}$：position $i$之前的token (autoregressive conditioning)
- $p(\cdot)$：模型在$\theta$下的conditional likelihood

这就是standard next-token prediction loss，但关键在data quality而不在loss本身。

**Critical insight**：cold start只用了image generation数据，但能generalize到video、understanding等所有任务。作者的解释是video本质上是multi-image understanding，且unified input-output structure（visual content + text → reward judgment）让model学到的format是task-agnostic的。

### Stage 2: Rejection Sampling (Generalization Fine-tuning)

**目标**：利用model的prior knowledge + generalization capability，在large-scale unified multimodal preference data上elicit CoT reasoning。

**数据规模**：
- Image Generation: HPD (25.6K) + OIP (7.4K) + EvalMuse (3K) + OpenAI-4o_t2i (6.7K)
- Video Generation: VideoDPO (10K) + Text2Video-Human (5.7K)
- Image Understanding: LLaVA-Critic-113K中采样30K
- Video Understanding: ShareGPTVideo-DPO (17K)

**Filtering策略**：模型生成CoT后，只保留final answer match GT的samples做SFT。这相当于self-distillation + correctness filtering。

**为什么需要这stage**：直接对所有data用GRPO计算成本太高（需要sample N=8 responses per input）。Rejection sampling先解决model已经能答对的case，reinforce正确reasoning pattern distribution，剩下hard case留给GRPO。

Ablation (Table 5)显示：去掉rejection sampling，VLReward从73.8降到70.4，GenAI-Video从82.3降到79.2。说明这stage对distribution shaping很重要。

### Stage 3: GRPO (Reinforcement Fine-tuning)

**目标**：针对model still fails的hard cases，通过exploration-driven RL来incentivize deeper reasoning。

#### Verifiable Rewards设计

两个rule-based reward：

**Format reward** $R_{fmt}$：检查output
