---
source_pdf: Do What You Say.pdf
paper_sha256: 2d67a5e2306372773bb268235eeb0cc4a4b360b25beee66ffc0f42a185e83374
processed_at: '2026-08-03T22:57:41-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

Andrej，我换个讲法，先讲故事，再把技术细节埋进去。

## 核心story：嘴上说一套，手上做一套

想象你在调教一个robot。你给它看任务："把alphabet soup和butter都放进basket"。这个robot很聪明，它会先在脑子里"说"一段plan：

> "First I pick up the soup, then place it in the basket. Now I need to pick up the butter..."

说完这段话，它才开始生成low-level action（end-effector的pose序列）。这就是所谓的**reasoning VLA**——先think，再act，模仿LLM里的Chain-of-Thought。

但是这里有个尴尬的事：**它说的plan是对的，它做的action经常是错的**。比如plan说"拿soup"，手却伸向了cream cheese。paper里Fig.1就画了这个场景。

作者把这种现象命名为 **"Embodied CoT Faithfulness Gap"**。翻译成人话：模型的text reasoning和它的motor action之间，faithfulness断了。LLM里的CoT faithfulness问题是"模型生成的reasoning是否真的反映了它内部的计算过程"（参考Turpin et al. 2023, https://arxiv.org/abs/2305.04388 ），这里借用了这个概念，但关注的是**action的物理outcome是否匹配text plan的语义**。

这个gap在in-distribution任务上就有8%的success rate损失，在OOD场景（尤其visual-viewpoint shift）上能掉到20%。作者的实验数据里，text generation accuracy接近100%，但action execution失败导致整体task failure——这印证了gap的存在。

---

## 为什么不直接训练解决？要runtime steering？

作者argue了两个理由，我觉得都挺合理的：

**理由1：训练loss和alignment objective本质不同。** 训练时的loss（Eq.2）是这样的：

$$
\mathcal{L}_{\mathrm{r\text{-}vla}} = \lambda_{\mathrm{reason}} \mathcal{L}_{\mathrm{reason}} + \lambda_{\mathrm{act}} \mathcal{L}_{\mathrm{act}}
$$

- $\lambda_{\mathrm{reason}}, \lambda_{\mathrm{act}} \in (0,1)$ 且 $\lambda_{\mathrm{reason}} + \lambda_{\mathrm{act}} = 1$：两个loss的权重，paper里都设0.5
- $\mathcal{L}_{\mathrm{reason}} = -\log \pi_\theta(\ell_j^r | o_t, \ell_{j-1}^r, \ell^g)$：text plan的next-token prediction loss，$\ell_j^r$是第j段sub-plan，$\ell_{j-1}^r$是上一段（给模型"running context"）
- $\mathcal{L}_{\mathrm{act}} = -\sum_{t'} \log \pi_\theta(a_{t'} | o_{t'}, \ell_j^r, \ell^g)$：action的flow-matching loss，条件在当前observation和当前text plan上

这个loss只要求"在expert demo上token likelihood高"，不要求"action rollout后的物理结果满足plan语义"。所以training根本没显式优化alignment。

**理由2：直接优化alignment很难。** Eq.3是alignment objective：

$$
\mathcal{L}_{\mathrm{align}} = -\mathbb{E}_{\mathcal{P}, \pi_\theta}\left[ R_{\mathrm{align}}(o_{t:t+H}, \hat{\ell}^r) \right]
$$

- $\mathcal{P}$：environment dynamics，non-differentiable
- $\pi_\theta$：policy本身，sampling也是non-differentiable的
- $R_{\mathrm{align}}$：alignment reward，要判断"observation序列是否满足text plan描述的subgoal"——这个reward本身也很难analytically定义

所以你没法backprop through $\mathcal{P}$ 和sampling。RL理论上可以，但VLA这种几B参数的模型做RL成本极高。于是作者选了**test-time scaling**这条路——这正是现在LLM社区流行的方向（参考Snell et al. 2024, https://arxiv.org/abs/2408.03314 ）。

---

## 方法：SEAL = Hypothesize + Predict + Verify

这个method的核心idea，用一句话讲：**既然模型自己sample出来的action有diversity，那就sample一堆，挑一个和plan匹配的执行。** 把action diversity从error source变成strength。

三阶段流程（每个reasoning step跑一次）：

### Stage 1: Hypothesize（采样候选action）

从reasoning VLA $\pi_\theta^{\mathrm{r\text{-}vla}}$ 并行sample K个action sequence：

$$
A_t = \{ \hat{\mathbf{a}}_t^{(k)} \sim \pi_\theta^{\mathrm{r\text{-}vla}}(\cdot | o_t, \hat{\ell}^r, \ell^g) \}_{k=1}^{K}
$$

- $\hat{\mathbf{a}}_t^{(k)} = a_{t:t+H_k}^{(k)}$：第k个candidate action序列
- $H_k$：这个序列的horizon长度，**每个序列不一样长**（这是关键设计，后面讲）
- $K$：候选数，paper里默认 $K=10$

这里有个很巧妙的architecture细节。他们用的base model是 $\pi_0$ （ https://arxiv.org/abs/2410.24164 ），按OneTwoVLA（ https://arxiv.org/abs/2510.12161 ）的recipe fine-tune。模型里有两个special token：
- `_intf`：表示"接下来要生成text reasoning"
- `<act>`：表示"接下来要生成action token"

Action生成是diffusion process（ $\pi_0$ 本身就是flow-matching模型）。**当模型生成 `_intf` token时，就标志着当前action sequence结束**。所以每个candidate会自然生成到不同长度就停——这个设计让action sequence的边界由模型自己决定，对应于它自己的text plan的语义范围，而不是人为切chunk。

### Stage 2: Predict（前向仿真）

对每个candidate action sequence $\hat{\mathbf{a}}_t^{(k)}$，用dynamics model $\hat{\mathcal{P}}$ 预测observation序列 $\hat{\mathbf{o}}_t^{(k)}$：

- 在simulation里：直接用K个parallel environment instance做high-fidelity forward simulation（LIBERO支持，参考 https://arxiv.org/abs/2306.03310 ）
- 在real world里：可以用learned world model（比如Cosmos， https://arxiv.org/abs/2501.03575 ）或digital twin（比如Robotwin， https://arxiv.org/abs/2503.20254 ）

这个Hypothesize-Predict loop是interleaved的：因为action是autoregressive生成的，生成 $a_{t'}$ 需要先有 $\hat{o}_{t'}$。所以采样和仿真交替进行，直到模型吐出 `_intf` token为止。

### Stage 3: Verify（VLM当裁判）

用pre-trained VLM（GPT-4o）作为proxy reward $R_\psi$，给每个candidate打分。输入给VLM的是：
- 初始image $I_1$（让VLM知道object初始位置）
- 这个candidate的final image $\hat{I}_{t+H_k}^{(k)}$（agentview + wristview）
- Text plan $\hat{\ell}^r$

VLM输出binary score $R_\psi \in \{0, 1\}$：1表示"final state满足plan"，0表示不满足。**选第一个被verify成功的candidate执行**（early exit，不等所有K个都verify完）。

paper的Appendix A.3里给了prompt template，是structured prompting——让VLM先step-by-step分析图像，再给结论。这种chain-of-thought式verifier prompting在 https://arxiv.org/abs/2403.09025 里被证明比direct query更准。

---

## 为什么这个方案能work？我的intuition

**关键在于：reasoning VLA的action distribution是有diversity的，但mode不集中。** 

想象action distribution $p(a | o_t, \hat{\ell}^r, \ell^g)$ 是个多峰分布。correct action是其中一个mode，但不是dominant mode。Base model greedy sampling经常落到错的mode上。

这其实和LLM里best-of-N sampling的logic完全一致（参考Snell et al. https://arxiv.org/abs/2408.03314 ）：如果你的verifier能区分好bad和good sample，那sample K个里挑best的，比greedy sample一个强。而且这个gain随K增长——paper Fig.6显示K从1到10，composition task success rate从~38%涨到~53%。

这里有几个让我觉得聪明的设计：

**1. Variable-length action sequence。** 因为 `_intf` token让模型自己决定action chunk在哪结束，每个candidate的 $H_k$ 不同。这对应了text plan的自然语义范围——一个"pick up soup"的plan可能对应15个action step，一个"open drawer"可能对应8个。比固定chunk size（像 $\pi_0$-V-GPS那样）更flexible。

**2. Asynchronous verification + early exit。** K个candidate并行生成，哪个先生成完就先送VLM verify。一旦有candidate被verify成功，立即执行，不等其他。这把latency从"K个串行"降到"~最快1个 + VLM latency"。Paper里报K=10时347ms，比RoboMonkey（ https://arxiv.org/abs/2510.08324 ，ref [6]）的520ms还快。

**3. 用VLM不用Q-function。** 这是和 $\pi_0$-V-GPS（ https://arxiv.org/abs/2410.13774 ，ref [16]）的核心区别。 $\pi_0$-V-GPS训了个offline Q-function（IQL），runtime时选Q值最高的action chunk。问题：Q-function在OOD场景下会over-estimate（Fig.3右图显示composition task上Q值虚高），给错信号。VLM有commonsense reasoning，泛化更好，虽然有时也错，但至少不会系统性地misleading。

---

## 实验结果：哪些结果最能说明问题？

### Result 1: ID task的提升（Table II）

在LIBERO-10 ID task上：

| Training Data | $\pi_0$ | $\pi_0$-V-GPS | $\pi_0$-reason | **SEAL** |
|---|---|---|---|---|
| LIBERO-10 | 85% | 90% | 92% | **96%** |
| LIBERO-100-Basket | 85% | 89% | 86% | **94%** |
| LIBERO-100 | 85% | 87% | 89% | **97%** |

关键观察：
- $\pi_0$-reason > $\pi_0$：reasoning确实有用，+4-7%
- SEAL > $\pi_0$-reason：verification再+4-8%
- **SEAL的提升随training data scale增大而增大**（96%→94%→97%），这暗示data diversity让base policy的candidate pool更好，verifier能挑到更好的

### Result 2: OOD robustness（Table VI，Fig.4）

这是最能体现SEAL价值的地方。四种OOD：

| OOD Type | $\pi_0$ | $\pi_0$-V-GPS | $\pi_0$-reason | **SEAL** |
|---|---|---|---|---|
| Lang-Rephrase | 73% | 71% | 86% | **95%** |
| Lang-Object-Property | 72% | 81% | 81% | **91%** |
| Visual-Scene | 84% | 83% | 91% | **98%** |
| Visual-Viewpoint | 28% | 23% | 25% | **45%** |

Visual-Viewpoint是最难的（换background+camera），所有方法都崩，但SEAL还有45%，比第二名高17个百分点。这说明：**visual shift会让action distribution严重degrade，但只要还能sample出几个correct mode，verifier就能救回来。**

### Result 3: Behavior Composition（Table VII，Fig.2a）

Composition task是测试"skill重组"——训练时见过"put A,B in basket"和"put C in basket"，测试时让它"put A,C in basket"。

| Training Data | $\pi_0$ | $\pi_0$-V-GPS | $\pi_0$-reason | **SEAL** |
|---|---|---|---|---|
| LIBERO-10 | 14% | 17% | 18% | 16% |
| LIBERO-100-Basket | 11% | 13% | 23% | 26% |
| LIBERO-100 | 16% | 16% | 38% | **53%** |

**SEAL的scaling trend最明显**：从16%涨到53%。而 $\pi_0$-V-GPS基本stagnate在15%左右。作者解释： $\pi_0$-V-GPS用的MUSE encoder（ https://arxiv.org/abs/2007.09052 ，ref [26]）language understanding太弱，OOD instruction下Q值信号是错的。

这个结果让我联想到LLM里的test-time scaling law——compute换performance，且gain随base model能力提升而提升（Snell et al.发现strong base model + best-of-N > weak base model + best-of-N，因为candidate pool质量更高）。

### Result 4: Runtime scaling（Fig.6）

K从1到10：
- Success rate：composition task上从~38%涨到~53%（+15%）
- Episode length：变短（更efficient execution）
- Latency：147ms → 347ms，**sub-linear scaling**（10x sample只2.1x sampling time，因为batched inference + KV cache）
- VLM verification是bottleneck：61ms → 163ms

---

## 和相关工作的关系

**1. 和RoboMonkey（ref [6]）的区别。** RoboMonkey也是test-time sampling + verification，但它的verifier是fine-tuned VLM（用synthetic preference label）。问题是这些preference label和真实task success不一定correlate。SEAL用GPT-4o zero-shot verify，靠VLM的commonsense。泛化更好，但latency更高（GPT-4o API call）。

**2. 和 $\pi_0$-V-GPS（ref [16]）的区别。** $\pi_0$-V-GPS用offline Q-function做verifier，chunk-level verification。SEAL用VLM做plan-level verification。区别在于：
- Q-function：dense signal，但OOD下over-estimate；chunk-level，没法判断"这一段action是否完成了plan描述的subgoal"
- VLM：sparse binary signal，但能做semantic judgment；plan-level，直接判断action sequence outcome的语义

**3. 和ECOT/Embodied CoT（ https://arxiv.org/abs/2407.08693 ，ref [7]）的关系。** ECOT是reasoning VLA的pioneer工作，但它的runtime没有verification——假设action会faithfully执行plan。SEAL补上了这块。

**4. 和LLM CoT faithfulness研究的呼应。** Turpin et al.（ https://arxiv.org/abs/2305.04388 ）发现LLM的CoT有时候"说一套做一套"——生成的reasoning不反映真实computation。SEAL相当于在robotics setting下，用runtime intervention强制faithfulness。

---

## 我的几个takeaway

1. **Test-time scaling在robotics里也work。** 这paper证明best-of-N + verifier在VLA上有效，且gain随base model能力提升而提升。这暗示robotics也可以走"train once, scale inference compute"的路线。

2. **Reasoning VLA的bottleneck在action不在plan。** Paper里说text generation accuracy接近100%，但action execution经常错。这意味着：提升reasoning VLA的关键可能不在更好的language model，而在更好的action generation或action verification。

3. **VLM作为robotics verifier的潜力。** GPT-4o zero-shot就能做binary alignment judgment，这在OOD场景比learned Q-function强。未来可能可以做更fine-grained的VLM verifier，或者fine-tune小VLM专门做verifier（降低latency）。

4. **Action diversity是feature不是bug。** 以前大家觉得VLA的action stochasticity是噪声，要尽量reduce。这paper反过来利用diversity——sample一堆，挑好的。这和diffusion policy社区的发现类似（ https://arxiv.org/abs/2303.04137 ）：diffusion的mode coverage能力是优势。

5. **Limitation也很明显。** Verifier依赖VLM，VLM在wrist-view image和occlusion场景下不准。Simulation dependence——real world没有high-fidelity simulator，得用world model，但world model本身在OOD下也不准。Latency还是高，real-time control难。

---

## 几个可以深挖的方向

如果你想build on这个工作，我觉得几个有意思的angle：

1. **Verifier的fine-tuning。** 用RLAIF思路fine-tune一个小VLM专门做alignment verifier，用SEAL的binary signal做reward，能降低latency且可能更准。

2. **World model的integration。** 把real-world world model（如Cosmos）接进来，做real robot实验。这是paper目前缺的。

3. **Verifier的uncertainty modeling。** 现在VLM给binary score，如果给probability + uncertainty estimate，可以做更principled的selection（比如Thompson sampling over candidates）。

4. **和hierarchical planning的结合。** SEAL现在在plan-level verify，如果加上更高层的goal-level verification（用VLM判断"整个task是否完成"），可能能处理更长horizon task。

5. **Active re-sampling。** 如果K个candidate都不被verify，现在就直接执行最好的。可以让模型"重新reasoning"——生成新plan或重新sample，类似LLM里的self-correction（ https://arxiv.org/abs/2310.01798 ）。

Paper project site: https://yilin-wu98.github.io/steering-reasoning-vla/

希望这个讲法更intuitive了。核心idea其实就一句话：**既然模型嘴上说"拿soup"，手却可能伸向cream cheese，那就让它试10次，看哪次真的拿到soup，就执行那次。** 剩下的都是工程细节。

---

# "Do What You Say": 深度解析这篇reasoning VLA alignment paper

 Andrej,这篇paper我觉得是一个挺有意思的工作,它把LLM里"CoT faithfulness"的概念移植到了embodied AI领域,然后用test-time scaling的方式去解决。下面我从几个层面拆解。

## 1. 核心问题:Embodied CoT Faithfulness Gap

这篇paper抓的痛点非常具体。在reasoning VLA中,模型会先输出一段textual plan,比如 "First I pick up the soup, then place it in the basket...",然后才生成low-level action。作者发现一个尴尬的现象:**text generation基本是对的,但action generation经常执行错**。

他们把这个形式化定义为一个"faithfulness gap"。让我把Eq.(3)拆开看:

$$
\mathcal{L}_{\mathrm{align}}(\theta; o_t, \hat{\ell}^r, \ell^g) = -\mathbb{E}_{\mathcal{P}, \pi_\theta^{\mathrm{r\text{-}vla}}}\left[ R_{\mathrm{align}}(o_{t:t+H}, \hat{\ell}^r) \right]
$$

- $o_t$:当前时刻observation,包含image $I_t$ 和proprioception $p_t \in \mathbb{R}^d$
- $\hat{\ell}^r$:模型自己生成的intermediate text plan(比如 "pick up the soup can")
- $\ell^g$:high-level goal instruction
- $\mathcal{P}$:environment true dynamics,non-differentiable
- $R_{\mathrm{align}}$:reward function,衡量future observation $o_{t:t+H}$ 是否满足 $\hat{\ell}^r$ 描述的subgoal
- 期望是对environment dynamics $\mathcal{P}$ 和policy $\pi_\theta^{\mathrm{r\text{-}vla}}$ 两个分布取的

关键insight是这个objective和training时的behavior cloning loss(Eq.2)是不同的——训练时没显式优化这个alignment,所以action的输出自然和plan脱节。

Eq.(2)是reasoning VLA的训练loss:

$$
\mathcal{L}_{\mathrm{r\text{-}vla}} = \lambda_{\mathrm{reason}} \mathcal{L}_{\mathrm{reason}} + \lambda_{\mathrm{act}} \mathcal{L}_{\mathrm{act}}
$$

- $\mathcal{L}_{\mathrm{reason}} = -\log \pi_\theta^{\mathrm{r\text{-}vla}}(\ell_j^r | o_t, \ell_{j-1}^r, \ell^g)$:text plan的cross-entropy loss
- $\mathcal{L}_{\mathrm{act}} = -\sum_{t'=t} \log \pi_\theta^{\mathrm{r\text{-}vla}}(a_{t'} | o_{t'}, \ell_j^r, \ell^g)$:action token的loss(实际是flow-matching loss)
- $\lambda_{\mathrm{reason}} = \lambda_{\mathrm{act}} = 0.5$(paper里用的值)

作者的意思是:这个loss只关心"在expert data上token-level的likelihood",没关心"action rollout的结果是否满足plan语义"。这是gap的根源。

---

## 2. Reasoning Data Annotation Pipeline

这部分挺重要的,因为他们需要一个reasoning-annotated dataset来训练base model。pipeline的flow:

1. 从LIBERO原始dataset取出一个episode的agentview image,编成10fps video
2. 把video + task instruction $\ell^g$ 喂给Gemini-2.5-Pro
3. Gemini一次pass生成:一系列一句话的sub-plan $\hat{\ell}_j^r$ + 每个sub-task的结束timestep $t_j'$
4. 由于一个sub-task的结束就是下一个的开始,直接得到标注序列:$(\hat{\ell}_1^r, 1), (\hat{\ell}_2^r, t_1'), \dots, (\hat{\ell}_L^r, t_{L-1}')$
5. L(子任务数)每个episode不同,由Gemini决定

reasoning label的format是:
- **Plans**: <all text plans in the task>
- **What has been done**: <completed plans>
- **Now I need to do**: <the next text plan to execute>

这个"running state"格式的设计很关键——它让模型在每一步都明确知道"已经做了什么,接下来要做什么",这给runtime verification提供了清晰的对照点。

参考一下他们的project site: https://yilin-wu98.github.io/steering-reasoning-vla/

---

## 3. Architecture:Adaptive Switching via Special Tokens

他们基于的base model是 $\pi_0$ (Physical Intelligence的工作, https://arxiv.org/abs/2410.24164 ),按OneTwoVLA( https://arxiv.org/abs/2510.12161 ,ref [8])的recipe fine-tune。

核心机制是两个special token:

- **
