---
source_pdf: Agent Learning via Early Experience.pdf
paper_sha256: 8b6a5a51fc9ed8a36c119f642bf1a98eefefdeb2088780ca5c428fd1b9d4ecef
processed_at: '2026-08-18T00:06:20-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Early Experience：用人话讲清楚

## 一、先说痛点在哪

想象你在教一个agent订机票。传统做法是给它一堆expert demonstrations：在这个页面点这个按钮，在那个页面填这个表单。agent就是死记硬背这些轨迹。

问题来了。agent部署时会遇到训练时没见过的页面，或者自己点错按钮进入奇怪状态。这时候它懵了——因为它训练时从来没见过"如果我点错了会发生什么"。它只见过expert的成功路径，完全不知道失败长什么样。

这就像你教小孩骑自行车，只给他看别人平稳骑行的视频，从来没让他摔过。他一旦失去平衡就不知道怎么recover。

RL理论上能解决这个问题——让agent自己试错，环境给reward。但现实很骨感：网站不告诉你"你填错了"，form提交成功不等于填对了；多步tool use序列太长，reward稀疏，credit assignment一团糟。

所以现在卡在一个尴尬位置：IL太弱（只会模仿，不会应对失败），RL太贵（环境不给reward，训练不稳定）。

## 二、Early Experience的核心insight

作者发现一个被忽视的事实：**agent执行action后环境返回的next state，本身就是supervision signal，不需要reward**。

举个具体例子。WebShop里你想买"蓝色、低于20块的耳机"。Expert action是"click蓝色15块耳机"。Agent自己propose一个alternative："click红色30块耳机"。执行后环境返回一个页面——可能是红色耳机的product page，价格显示30块。

这个next state告诉你：红色耳机超预算了。这个信息是**reward-free**的——环境没说"你错了"，但page content本身就编码了"这个action导致超出budget constraint"。

Early experience就是：让agent在每个expert state自己探索K个alternative actions，收集这些(s, a, s')三元组，然后用两种方式从中学习。

## 三、两个Method的intuition

### Method 1: Implicit World Modeling (IWM)

直觉很简单：**先让agent理解环境怎么变化，再学怎么在环境中行动**。

具体来说，对每个(s, a, s')三元组，训练model预测"在state s下做action a，下一个state长什么样"。因为state是natural language，这就是标准next-token prediction。

比如在WebShop：
- Input: current page + action "click[non-ears blue]"
- Target: "After clicking, this page is a product-details page with color options, price $24.99, Buy Now button..."

Model通过预测这些next states，internalize了"click不同按钮会导致什么样的page变化"。这就像小孩通过大量观察学会"推这个会倒，拉那个会开"的物理直觉。

关键设计点：**用同一组参数做world modeling和policy learning**。不是搞个separate simulator，而是把predictive signal直接bake进policy。这避免了额外module的overhead，natural fit LLM fine-tuning。

Two-stage pipeline：
1. 先用world modeling loss训练，让model建立dynamics prior
2. 再用expert data做SFT，学具体action

### Method 2: Self-Reflection (SR)

这个更有意思。直觉是：**让agent对比expert action和自己alternative action的结果，生成"为什么expert更好"的reasoning，然后学这个reasoning**。

流程：
1. Expert action $a_i$ → 得到 $s_{i+1}$（好的结果）
2. Alternative $a_i^j$ → 得到 $s_i^j$（可能差的结果）
3. Prompt LLM：给定 $s_i, a_i, s_{i+1}, a_i^j, s_i^j$，解释为什么 $a_i$ 比 $a_i^j$ 好
4. 把这个reasoning $c_i^j$ 作为训练target

训练时让model预测 $c_i^j \circ a_i$（reasoning + expert action）。

为什么这比纯IL强？看个具体例子：

WebShop中：
- Expert: "click $15 blue shirt"
- Alternative: "click $30 red shirt"
- Reflection: "Red shirt matches color preference but exceeds $20 budget constraint. Blue shirt satisfies both style requirement and budget limit."

这个reflection教model的是**prioritize constraints**这个decision criteria，而不是specific item。下次遇到类似但不同的场景，这个criteria能transfer。

相比之下，纯IL只教"在这个state做这个action"，换个页面就废了。

## 四、实验结果的high-level takeaway

### 4.1 In-domain效果

8个环境，3个model family，early experience全面碾压IL。几个highlight：

- WebShop上IWM给Llama-3.2-3B带来+18.4的jump（41.8 → 60.2）
- TravelPlanner上SR给Llama-3.1-8B带来+15.0（17.2 → 32.2）
- ScienceWorld上SR给Llama-3.1-8B带来+13.3（54.7 → 68.0）

**什么时候IWM强，什么时候SR强？**

这取决于environment的特性。看两个维度：

**Action space维度**：
- Closed/finite（ALFWorld, ScienceWorld, TravelPlanner）：IWM internalize transition regularities；SR在long-horizon planning上gain大
- Structured but large（BFCLv3, Tau-Bench）：early experience减少tool misuse；SR在logical error上更强
- Open（SearchQA, WebArena）：最难，但仍有reliable gains

**Observation space维度**：
- State transitions predictable（WebShop）→ IWM excels，因为predict next state容易学
- Failures from reasoning errors（TravelPlanner, ScienceWorld）→ SR delivers larger gains，因为需要对比reasoning

这两个method是complementary的，根据环境特性选。

### 4.2 Out-of-Domain效果

OOD setting下early experience的优势更明显。几个case中OOD gains甚至大于in-domain（如SearchQA）。

这很make sense：IL过拟合training distribution，遇到新state就废；early experience让agent见过自己alternative actions的consequences，这些experience generalize到新场景。

### 4.3 RL bridge效果（最重要）

这个实验直接验证paper核心claim：early experience是通向RL的bridge。

在WebShop、ALFWorld、SearchQA上用GRPO做RL，固定hyperparameters，只变starting checkpoint。结果：early experience checkpoints consistently yield higher post-RL ceilings。

具体数据（ALFWorld, Llama-3.1-8B）：
- IL warm-start → RL: 80.5 → 97.7
- IWM warm-start → RL: 85.9 → 97.7（相同最终但起点更高）
- SR warm-start → RL: 85.2 → 98.5（最高ceiling）

WebShop, Llama-3.1-8B：
- IL → RL: 47.3 → 80.5
- IWM → RL: 58.6 → 91.4（起点和终点都更高）
- SR → RL: 58.2 → 89.8

直接从pretrained model做GRPO最差且unstable。

**Intuition**：early experience相当于mid-training。IL只学了static state→action mapping，缺乏dynamics understanding。Early experience让model先建立environment representation和decision prior，RL在此基础上optimize reward时，better initialization → higher ceiling。这解释了为什么post-RL performance gap maintained or amplified。

### 4.4 Ablation的insight

**Branching factor K**：
- IWM: 随K增大steadily improve（更多transition data，学richer dynamics）
- SR: small-moderate K最好，very large K有diminishing returns。原因有二：一是comparing很多alternatives时偶尔包含other success-leading actions，减少与expert的contrast；二是current models在single context中reason over many alternatives能力有限

实践建议：IWM favor larger K；SR用moderate K（2-4）。

**Expert data scaling**：
- WebShop上1/8 expert data + early experience > full data IL
- ALFWorld上1/2 expert data + early experience > full data IL

这说明early experience提供的supervision**beyond what demonstrations alone can supply**。Agent自己的exploration触发了demonstrations覆盖不到的states。

### 4.5 vs Baselines

三个baseline对比很有启发性：

**Long CoT (test-time)**：强制更长CoT。在prompt模型上modest gains，但在fine-tuned model上**performance degradation**（WebShop 47.3 → 0.0）。因为fine-tuned model缺乏inherent rationales，长推理ungrounded反而有害。

**STaR-style**：synthesize rationales for expert actions。但rationales ungrounded（不看alternative actions和结果states）。WebShop 47.3 → 25.0。可能含hallucinated facts。

**DPO**：用off-expert rollouts作rejected，expert作chosen。Weaker and brittle，training在几十步内collapse。Preference signal over off-expert rollouts是weaker training target than grounding in observed next states。

**关键insight**：early experience的supervision是**grounded in observed environment responses**。Long CoT和STaR的reasoning是ungrounded，可能hallucinate。DPO只用了preference signal（好/差），没用next state的rich content。这解释了为什么early experience robust而alternatives brittle。

### 4.6 Model Scaling

WebArena上Llama-3.2-3B → 3.1-8B → 3.3-70B。Early experience在every scale上outperform IL，gap在70B上persist。

这说明supervision complements model size rather than substituting for it。大模型也能从early experience中benefit，不是因为模型capacity不够，而是因为这种supervision signal本身提供了IL无法提供的信息。

## 五、核心Intuition的深层理解

### 5.1 为什么early experience比IL强

三层原因：

**1. Supervision density**：IL有N个state-action pairs；early experience有N×(K+1)个triplets。数据量直接多一个数量级。

**2. Coverage**：expert只覆盖narrow success path。Agent自己的exploration触及failure states, error states, non-optimal states。这让agent学到recovery和robustness。

**3. Information content**：IL的signal只是"在这个state做这个action"。Early experience的signal是"在这个state做这个action会导致这个结果"——后者信息量大得多，且grounded。

### 5.2 IWM和SR为什么complementary

IWM学的是**environment dynamics**：state怎么变化。这是model-based的knowledge。

SR学的是**decision criteria**：为什么这个action比那个好。这是value-based的knowledge。

两者针对不同failure mode：dynamics misunderstanding → IWM fix；reasoning error → SR fix。这也是为什么不同环境上两者表现不同——dynamics predictable的环境IWM强，reasoning-heavy的环境SR强。

### 5.3 为什么是RL的bridge

IL warm-start的问题：只学了static state→action mapping，policy对environment dynamics没理解。RL直接optimize reward时，如果policy对环境完全不理解，exploration效率极低，training unstable。

Early experience相当于mid-training stage：让model先建立environment dynamics的mental model和decision criteria的prior。RL在此基础上optimize reward时，better prior → more efficient exploration → higher ceiling。

这就像AlphaGo先学人类棋谱（IL），再做RL。但early experience比纯IL更进一步——它让agent见过自己失败action的consequences，这更接近RL的trial-and-error spirit，只是没有reward signal。

## 六、一些更细节的technical insight

### 6.1 Rollout dataset构造的细节

不同环境的rollout构造有讲究：

**ALFWorld**: closed action space，从admissible action list uniform sample K=8个（excluding expert）。这保证alternatives都是valid action。

**WebShop**: open-ish action space，mix model-proposed（temperatures {0.5, 0.8, 0.9}）和uniform sampled admissible。温度mixing让alternatives更多样。

**SearchQA**: 30个alternatives with temperature 1.0。数量大是因为search query空间太大，需要dense exploration。

**TravelPlanner**: exhaustive augmentation——ALL valid actions at each state。这最大化coverage，生成70,000+ samples。

**WebArena**: 5个alternatives，但next state用same model generate concise summary。这step很关键——raw accessibility tree太noisy，summary提取task-relevant info。

这种per-environment tuning说明：early experience的effectiveness依赖rollout quality，不能盲目apply。

### 6.2 Next state representation的选择

不同环境next state的处理不同：

- ALFWorld/ScienceWorld: 原始textual observation
- WebShop: offline textual summary（avg 345 chars）
- SearchQA: summarize retrieved docs，predict summary not full text
- WebArena: model-generated concise summary

这个pattern揭示一个insight：**next state需要task-relevant的compression**。Raw state太noisy，直接predict会浪费model capacity在irrelevant details上。Summary or distillation是必要的preprocessing。

这也引发一个question：能否jointly learn summarization和world modeling？或者用attention机制让model自己learn what to attend to in next state？这可能是future work方向。

### 6.3 SR prompt的设计哲学

看Appendix B.1的prompt template，几个设计principle：

1. **Grounding in observed outcomes**: prompt包含 $s_{i+1}$ 和 $s_i^j$，要求reflection grounded在这些actual observations，不允许speculation
2. **Step-by-step reasoning**: 要求natural monologue，analyze situation → compare actions → justify expert → highlight constraints
3. **No meta-commentary**: 不允许"作为AI"之类的meta-talk，保持task-focused
4. **Strict information boundary**: "Stay strictly within the provided information"，防止hallucination

这些constraint确保SR生成的reasoning是high-quality、grounded、transferable的。如果prompt设计松散，reflection可能degenerate成generic statement如"expert action is better because it's more appropriate"，失去supervision价值。

### 6.4 Training recipe的细节

Two-stage pipeline for IWM:
1. 1 epoch world modeling loss（include expert transitions $(s_i, a_i, s_{i+1})$）
2. Continue with $\mathcal{L}_{\mathrm{IL}}$ on $\mathcal{D}_{\mathrm{expert}}$

Total updates equal IL budget，no extra steps。这保证fair comparison。

For SR:
- Mix $\mathcal{D}_{\mathrm{refl}}$ 和 $\mathcal{D}_{\mathrm{expert}}$
- Train same number of epochs as IL
- Expert trajectories中已有的CoT保留

这种设计保证early experience不extra compute（除了rollout collection），但获得better supervision。

### 6.5 Why DPO fails but early experience works

DPO用同样的rollout data，只取preference signal（expert好，alternative差）。结果training collapse。

这对比很有启发性。DPO只用了**binary preference**：expert > alternative。但next state包含的information远比binary preference丰富——它告诉你alternative action具体导致了什么后果，是error message？是wrong product page？是format error？

Early experience的IWM直接predict这个rich next state；SR基于next state差异生成reasoning。两者都**exploit the full information content of next state**，而DPO把它压缩成1-bit signal。这解释了为什么DPO weaker and brittle——它waste了大部分supervision signal。

## 七、Limitations和Open Questions

Paper没有explicit limitations section，但从分析可推断：

1. **K的选择需要per-environment tuning**：SR在大K时diminishing returns，说明current models在long context中reason over many alternatives能力有限。Future work可能需要hierarchical reflection或iterative comparison。

2. **Rollout collection的compute overhead**：虽然reward-free，但environment interaction仍需cost。对slow environment（如real website），这可能prohibitive。

3. **Alternative action quality依赖initial policy**：如果initial policy太差，proposed alternatives可能都uninformative（如全部是format error）。Paper用mix model-proposed和uniform sampled来mitigate，但没系统study这个issue。

4. **Open action spaces仍是hard regime**：WebArena上绝对performance仍低（8.5%）。虽然early experience有gain，但open action space的combinatorial explosion仍是fundamental challenge。

5. **Next state summarization需要手工设计**：不同环境用不同summarization strategy（raw, offline summary, model summary）。能否自动learn optimal summarization？

6. **SR的reflection quality**：依赖LLM自身的reasoning能力。如果LLM reasoning弱，generated reflection可能low quality。Paper用filter mitigate，但没quantify reflection quality对final performance的影响。

7. **Only K alternative actions per state**：没explore deeper rollouts（如alternative action后继续alternative）。Multi-step alternative rollouts可能提供更rich supervision但也exponentially expensive。

## 八、更broad的implications

### 8.1 对agent training paradigm的启示

当前language agent training主要靠SFT on expert data。这篇paper展示一个practical alternative：用agent自己的experience作为supervision，不需要reward。

这降低了对expert data的依赖。Figure 4a显示1/8 expert data + early experience > full data IL。如果这个trend holds，early experience可能显著reduce对high-quality expert demonstrations的需求。

### 8.2 对RL for language agents的启示

RL for language agents目前不mature——reward难定义，long-horizon rollout难处理。Early experience提供一个practical bridge：先reward-free learning，建立strong prior，再seamlessly integrate RL when available。

这类似AlphaGo的recipe（先IL学人类棋谱，再RL自我对弈），但更进一步：early experience让agent见过自己失败action的consequences，这更接近RL的trial-and-error spirit。IL只是passive imitation，early experience是active exploration without reward。

### 8.3 对world model research的启示

传统world model是separate simulator，用于model-based RL的planning。这篇paper提出**implicit world model**：把predictive signal直接bake进policy，不用separate module。

这更fit LLM paradigm——LLM本来就是next-token predictor，world modeling作为auxiliary task natural fit。且避免了separate simulator的inaccuracy问题——implicit world model和policy share parameters，policy直接learned in context of dynamics understanding。

Concurrent work如CWM (team et al. 2025)、Dyna-think (Yu et al. 2025)也explore类似idea，说明这是promising direction。

### 8.4 对self-improvement的启示

Self-reflection最初是prompting technique（Reflexion, Self-Refine），但inference-time方法often fail without external feedback（Huang et al. 2024）。这篇paper把reflection作为training signal，grounded in observed state transitions，解决ungrounded reflection的问题。

这suggest一个更broad的principle：inference-time techniques如果能grounded in environment observations并作为training signal，可能unlock更强self-improvement。

## 九、Future work方向推测

基于paper的analysis和limitations，几个可能的future direction：

1. **Hierarchical reflection**: 对大K，用hierarchical structure先compare小groups，再compare winners。解决SR在大K时diminishing returns问题。

2. **Multi-step alternative rollouts**: 不只explore 1-step alternative，而是explore multi-step alternative trajectories。这提供更long-horizon的supervision，但compute expensive。可能需要smart sampling策略。

3. **Automatic summarization learning**: Jointly learn how to summarize next state和how to predict/use it。Remove per-environment summarization design。

4. **Curriculum over alternative difficulty**: 先explore easy alternatives（near-expert），再explore hard alternatives（far-from-expert）。这像curriculum learning，可能加速learning。

5. **Early experience + active exploration**: Agent主动select which alternatives to explore based onuncertainty或information gain。这更接近RL的exploration spirit，但需要uncertainty quantification。

6. **Cross-environment transfer**: Early experience在一个环境学到dynamics understanding能否transfer到similar环境？比如WebShop的dynamics能help WebArena吗？

7. **Multi-modal early experience**: Paper只study text-based environments。对visual environments（如GUI screenshots），early experience怎么adapt？Next state prediction变成next image prediction，更复杂。

## 十、总结

这篇paper的beauty在于**conceptual simplicity + practical effectiveness**。

Conceptually，insight很简单：agent执行action后环境返回的next state本身就携带supervision signal，不需要reward。这个observation被overlook了，但一旦pointed out就make sense。

Practically，两个method（IWM和SR）都是标准LLM fine-tuning的variant，容易integrate进现有pipeline。Eight environments, three model families上consistent gains，且作为RL warm-start更strong。

这work的positioning很smart：不claim解决RL for language agents的难题，而是提供practical bridge。在reward signal unavailable的now，先让agent从experience中learn something useful，等RL infrastructure成熟后再seamlessly integrate。

对researcher的启示：当fully reward-driven RL不available时，look for alternative supervision sources。Agent自己的experience和environment responses是rich supervision，别让它浪费掉。这可能是从"era of human data"过渡到"era of experience"的practical path。

参考链接：
- Paper: [arXiv:2507.08625](https://arxiv.org/abs/2507.08625)
- Silver & Sutton "Era of Experience": [Google AI Blog](https://blog.google/technology/ai/google-ai/)
- GRPO (DeepSeekMath): [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- DPO: [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- STaR: [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
- Reflexion: [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
- WebArena: [arXiv:2307.13854](https://arxiv.org/abs/2307.13854)
- WebShop: [arXiv:2207.01206](https://arxiv.org/abs/2207.01206)
- ALFWorld: [arXiv:2010.03768](https://arxiv.org/abs/2010.03768)
- TravelPlanner: [arXiv:2402.02822](https://arxiv.org/abs/2402.02822)
- Search-R1: [arXiv:2503.09516](https://arxiv.org/abs/2503.09516)
- DAgger: [arXiv:1011.0686](https://arxiv.org/abs/1011.0686)
- Hindsight Experience Replay: [arXiv:1707.01495](https://arxiv.org/abs/1707.01495)
- Dreamer (world model): [arXiv:1910.01341](https://arxiv.org/abs/1910.01341)

---

# Agent Learning via Early Experience 深度解析

## 一、Core Problem & Motivation

这篇paper要解决一个非常根本的tension。当前language agent training面临一个dilemma：

**Imitation Learning (IL)** 用expert demonstrations做SFT，是当前主流，但存在致命缺陷：
- Agent never observes consequences of its own actions
- Distribution shift：部署时π_θ偏离expert policy，错误compound
- 数据scaling难，generalization差

**Reinforcement Learning (RL)** 理论上是silver bullet，但在real-world language agent settings中：
- 很多环境**缺verifiable reward**（如website，form提交成功不代表填对了）
- Multi-turn tool use需要**inefficient long-horizon rollouts**
- Credit assignment困难，training unstable

作者的insight是：在IL和RL之间存在一个**middle ground** —— agent自己propose action，环境返回future state，这个future state本身就携带supervision signal，**不需要reward**。

这呼应了Silver & Sutton的"era of experience"vision，但承认fully reward-driven RL还不成熟，所以提出early experience作为practical bridge。

---

## 二、Formalization: MDP Setup

作者用标准MDP来formalize：

$$\mathcal{M} = (S, A, T, R, \gamma, \rho_0)$$

变量含义：
- $S$：state space，在language agent中是webpage content、tool output、textual environment description
- $A$：action space，discrete choices如click element、invoke tool、generate text
- $T: S \times A \to \Delta(S)$：transition function，$\Delta(S)$是state space上的probability simplex
- $R: S \times A \to \mathbb{R}$：reward function，**关键在于training时可能unknown或unverifiable**
- $\gamma \in [0,1]$：discount factor
- $\rho_0 \in \Delta(S)$：initial state distribution

Policy：$\pi_\theta: S \to \Delta(A)$，参数为$\theta$。

### Imitation Learning的limitation数学化

给定expert dataset $\mathcal{D}_{\mathrm{expert}} = \{(s_i, a_i)\}_{i=1}^N$：

$$\mathcal{L}_{\mathrm{IL}}(\theta) = -\sum_{i=1}^N \log \pi_\theta(a_i \mid s_i) \tag{1}$$

这个loss的问题：agent只看到$(s_i, a_i)$对，**never observes what happens with non-expert action**。当部署时π_θ不可避免偏离expert policy，agent进入training data没覆盖的state，错误compound（Ross et al. 2011的DAgger motivation）。

---

## 三、Early Experience的核心机制

### 3.1 Rollout Dataset构造

对每个expert state $s_i$，从initial policy $\pi_\theta(\cdot | s_i)$ sample $K$ 个alternative actions：

$$\mathcal{A}_i = \{a_i^1, a_i^2, \ldots, a_i^K\}$$

执行expert action $a_i$ 得到 $s_{i+1}$，执行每个alternative $a_i^j$ 得到 $s_i^j \sim T(s_i, a_i^j)$。

构造rollout dataset：

$$\mathcal{D}_{\mathrm{rollout}} = \{(s_i, a_i^j, s_i^j) \mid i \in [N], j \in [K]\} \tag{2}$$

**关键intuition**：next state $s_i^j$ 编码了action quality的implicit feedback。比如在WebShop中，点击错误颜色按钮会得到不同的product page；在BFCL中调用错误tool会得到error message。这些都是**reward-free但informative**的supervision。

实际中 $\mathcal{D}_{\mathrm{rollout}}$ 可以比 $\mathcal{D}_{\mathrm{expert}}$ 大一个数量级（如ALFWorld中21,031 expert pairs → 189,279 triplets with K=8）。

### 3.2 Method 1: Implicit World Modeling (IWM)

**核心思想**：把world modeling作为policy的auxiliary prediction task，让agent内化environment dynamics。

形式化为next-token prediction（因为state是natural language）：

$$\mathcal{L}_{\mathrm{IWM}} = -\sum_{(s_i, a_i^j, s_i^j) \in \mathcal{D}_{\mathrm{rollout}}} \log p_\theta(s_i^j \mid s_i, a_i^j) \tag{3}$$

变量：
- $p_\theta$：language model的output distribution
- $s_i$：current state（input）
- $a_i^j$：alternative action（input）
- $s_i^j$：target next state

**关键设计**：用**同一组参数θ**同时做state prediction和action prediction。这与传统world model不同——传统world model是separate simulator，这里把predictive signal直接integrate进policy learning。

Two-stage pipeline：
1. 先用 $\mathcal{L}_{\mathrm{IWM}}$ 训练（include expert transitions $(s_i, a_i, s_{i+1})$），internalize coarse dynamics
2. 再在 $\mathcal{D}_{\mathrm{expert}}$ 上用 $\mathcal{L}_{\mathrm{IL}}$ fine-tune

**Intuition building**：这类似于mid-training的思路。先让model"理解环境怎么变化"，再学"在环境中怎么做"。比如在WebShop中，model先学到"click[non-ears blue] → 进入product details page with color options"，再学具体的purchase策略。

### 3.3 Method 2: Self-Reflection (SR)

**核心思想**：让agent比较expert action和自己的alternative action，基于结果state差异生成natural language reasoning。

具体流程：
1. 对每个expert state $s_i$，执行expert action $a_i$ 得到 $s_{i+1}$
2. 对每个alternative $a_i^j$（$j \in \{1, \ldots, K\}$）得到 $s_i^j$
3. Prompt同一个LLM生成chain-of-thought $c_i^j$，解释为什么 $a_i$ 比 $a_i^j$ 好
4. 收集triplets $(s_i, a_i^j, c_i^j)$ 到 $\mathcal{D}_{\mathrm{refl}}$

训练loss：

$$\mathcal{L}_{\mathrm{SR}} = -\sum_{(s_i, a_i^j, c_i^j) \in \mathcal{D}_{\mathrm{refl}}} \log p_\theta(c_i^j, a_i \mid s_i) \tag{4}$$

变量：
- $c_i^j$：chain-of-thought reasoning（target）
- $a_i$：expert action（target）
- $s_i$：current state（input）
- $p_\theta$：与policy $\pi_\theta$ aligned的output distribution

实际训练：mix $\mathcal{D}_{\mathrm{refl}}$ 和 $\mathcal{D}_{\mathrm{expert}}$，用next-token prediction。Expert trajectories中已有的CoT保留。

**Intuition building**：SR不只是让model记住"在这个state做这个action"，而是让model学到**transferable decision criteria**。例如WebShop中expert action是"click $15 blue shirt"，alternative是"click $30 red shirt"，reflection会说："red shirt matches color preference但exceeds $20 budget constraint；blue shirt satisfies both style requirement和budget limit"。这教会model prioritize constraints，而不是specific item。

---

## 四、实验结果深度分析

### 4.1 In-domain Effectiveness (Table 1)

八个环境、三个模型family的完整结果。几个highlight：

| Benchmark | Model | IL | IWM | SR |
|-----------|-------|-----|-----|-----|
| WebShop | Llama-3.2-3B | 41.8 | **60.2 (+18.4)** | 52.7 (+10.9) |
| TravelPlanner | Llama-3.1-8B | 17.2 | 25.0 (+7.8) | **32.2 (+15.0)** |
| ScienceWorld | Llama-3.1-8B | 54.7 | 57.0 (+2.3) | **68.0 (+13.3)** |
| BFCLv3 | Llama-3.2-3B | 21.3 | 25.3 (+4.0) | **29.3 (+8.0)** |

**Action-space perspective的insight**：
- **Closed/finite action sets**（ALFWorld, ScienceWorld, TravelPlanner）：IWM internalizes transition regularities；SR在long-horizon planning上gain大
- **Structured but large action sets**（BFCLv3, Tau-Bench）：early experience减少tool misuse和ordering错误；SR在logical error上更强
- **Open action sets**（SearchQA, WebArena）：最难，但early experience仍reliable gains

**Observation-space perspective**：
- State transitions predictable（WebShop）→ IWM excels
- Failures from reasoning errors（TravelPlanner, ScienceWorld）→ SR delivers larger gains
- 这说明IWM和SR是complementary的，根据环境特性选择

### 4.2 Out-of-Domain Generalization (Table 2)

OOD设置：ALFWorld和SearchQA用原始OOD splits；BFCLv3的OOD是missing function/argument/long context averaged。

关键观察：
- OOD scores全面下降，但early experience**recover substantial portion of the gap**
- 几个case中OOD gains甚至大于in-domain（如SearchQA）
- IWM在dynamics stable时好（ALFWorld）；SR在distribution shift affect tool availability时好（BFCLv3）

这表明agent的own rollouts提供generalizable supervision，超越了demonstrations的coverage。

### 4.3 RL Following Early Experience (Figure 3)

这是paper最重要的claim之一：early experience是通向RL的bridge。

实验：在WebShop、ALFWorld、SearchQA上用GRPO（Shao et al. 2024），固定hyperparameters，只变starting checkpoint。

结果pattern：early experience checkpoints consistently yield higher post-RL ceilings。有时gap在RL过程中扩大（ALFWorld），有时缩小但never reverses。直接从pretrained model做GRPO最差且unstable。

**Intuition**：early experience相当于mid-training，让model先建立environment dynamics的representation和decision principles的prior，RL再在此基础上optimize reward。这比cold start IL warm-up更有效。

### 4.4 Ablations

**Branching factor K (Figure 4b)**：
- IWM：随K增大steadily improve（学到richer transition regularities）
- SR：small-moderate K最好，very large K有diminishing returns。原因：comparing很多alternatives时偶尔包含other success-leading actions，减少与expert的contrast；且current models在single context中reason over many alternatives能力有限
- 实践建议：IWM favor larger K；SR用moderate K（2-4）

**Demonstration budget (Figure 4a)**：
- WebShop上1/8 expert data + early experience > full data IL
- ALFWorld上1/2 expert data + early experience > full data IL
- 说明early experience提供的supervision**beyond what demonstrations alone can supply**

### 4.5 Comparison to Baselines (Table 3)

三个baseline：

1. **Long CoT (test-time)**：用delimiter truncation鼓励更长CoT。结果：modest gains on prompt模型，但在fine-tuned model上**performance degradation**（WebShop 47.3 → 0.0）。原因：fine-tuned model缺乏inherent rationales，长推理不grounded。

2. **STaR-style (training-time)**：synthesize rationales for expert actions。但rationales ungrounded（不看alternative actions和结果states）。结果：performance degradation（WebShop 47.3 → 25.0）。可能含hallucinated facts。

3. **DPO**：用off-expert rollouts作rejected，expert作chosen。结果：weaker and brittle，training在几十步内collapse。说明preference signal over off-expert rollouts是weaker training target than grounding in observed next states。

**关键对比insight**：early experience的supervision是**grounded in observed environment responses**，而Long CoT和STaR的reasoning是ungrounded。这解释了为什么early experience robust而alternatives brittle。

### 4.6 Model Scaling (Figure 5)

WebArena上比较Llama-3.2-3B、3.1-8B、3.3-70B。70B用LoRA。

结果：early experience在every scale上outperform IL，gap在70B上persist。Absolute performance随scale上升，early-experience checkpoints consistently occupy top curve。说明supervision complements model size rather than substituting for it。

---

## 五、Self-Reflection Prompt Template解析

Paper Appendix B.1给出了SR的prompt template，这是理解method的关键：

```
You will be presented with a situation where you need to choose 
between multiple possible actions. Your task is to analyze the 
situation and provide reasoning about why we decide to take the 
expert action.

• Situation Description (s_i): {Situation Description}
• Expert Action (a_i): {Expert Action}
• Expected Outcome (s_{i+1}): {Future State of Expert Action}
• Alternative Actions:
  1. Action a_i^1: {Alt Action 1}, resulting state s_i^1: {Obs 1}
  2. Action a_i^2: {Alt Action 2}, resulting state s_i^2: {Obs 2}
```

要求monologue包含：
1. Analyze situation and goal
2. Compare actions, explain why each less optimal
3. Justify expert action, grounded in expected outcome
4. Highlight clues, constraints, consequences

**设计哲学**：reflection必须grounded在实际observed state transitions，不允许meta-commentary，必须natural step-by-step reasoning。这确保了SR的supervision quality。

---

## 六、Per-Environment Implementation Details

### ALFWorld (Appendix C.1)
- 21,031 expert state-action pairs
- IWM: K=8，21,031 × 9 = 189,279 triplets
- SR: temperature 1.0 propose up to 3 alternatives，canonicalize后keep unique
- Training: batch 16, lr 1e-5, 2 epochs

### WebShop (Appendix C.2)
- 1,571 human trajectories → 15,464 state-action pairs
- IWM: 122,954 triplets（temperatures {0.5, 0.8, 0.9} + uniform sample 5）
- SR: 6,235 examples（filter trajectories < 15 steps）
- 关键：next state用offline textual summary（avg 345 chars）

### SearchQA (Appendix C.5)
- 7,000 tasks from Musique，用Search-R1生成expert data
- IWM: 30 alternative actions per state（temperature 1.0），predict summaries not full text
- SR: 2 alternatives per state，7,691 training instances
- RL: F1 score as reward，max 6 retrieval interactions

### TravelPlanner (Appendix C.7)
- 45 training trajectories → 1,395 state-action pairs
- IWM: exhaustive augmentation，ALL valid actions per state → 70,000+ samples
- SR: up to 30 alternatives，temperature 0.9，8-way tensor parallelism

### WebArena (Appendix C.8)
- 554 successful trajectories from leaderboard agents → 7,044 state-action pairs
- IWM: 5 alternatives per state，next state用same model generate concise summary
- SR: 5 alternatives，filter low-quality → 3,190 examples
- 165 tasks in WebArena-Lite evaluation

---

## 七、关键Intuition总结

### 7.1 为什么early experience work

1. **Supervision density**：IL只有N个state-action pairs；early experience有N×(K+1)个triplets，supervision密度高一个数量级

2. **Coverage**：expert demonstrations只覆盖narrow range of scenarios；agent自己的exploration触及non-expert states，学到recovery和robustness

3. **Grounding**：与ungrounded reasoning不同，early experience的supervision来自实际observed environment responses，no hallucination

4. **Complementarity**：IWM学dynamics，SR学decision criteria，两者针对不同failure mode

### 7.2 IWM vs SR的何时用何者

- **IWM**：environment dynamics stable & predictable，state transitions carry rich information（如WebShop的product page变化）
- **SR**：failures from reasoning errors，long-horizon constraint satisfaction，distribution shift affects tool availability（如TravelPlanner、ScienceWorld）

### 7.3 为什么是RL的bridge

- Early experience相当于**mid-training stage**：建立environment representation和decision prior
- IL warm-start只学了static mapping，缺乏dynamics understanding
- RL在此基础上optimize reward时，better initialization → higher ceiling
- 这解释了为什么post-RL performance gap maintained or amplified

### 7.4 与classical methods的关系

- vs **Hindsight Experience Replay**：HER需要goal relabeling；early experience直接用next states，不需要
- vs **DAgger**：DAgger需要interactive expert correction；early experience用static expert data + own rollouts
- vs **Classical World Models**：传统world model是separate simulator用于planning；IWM是implicit的，integrate进policy，无planning overhead
- vs **STaR**：STaR的rationales ungrounded；SR的reflection grounded in observed state transitions

---

## 八、Limitations和Open Questions

Paper没有explicitly讨论limitations section，但从analysis可推断：

1. **K的选择**：SR在大K时diminishing returns，说明current models在long context中reason over many alternatives能力有限
2. **Compute overhead**：rollout collection需要environment interaction，虽然reward-free但仍需rollout
3. **Alternative action quality**：依赖initial policy的exploration quality，如果initial policy太差，alternatives可能uninformative
4. **Open action spaces**：WebArena上绝对performance仍低（8.5%），说明open action spaces仍是hard regime

---

## 九、个人Reflections

这篇paper的核心贡献是**conceptual clarity**：把"agent从自己experience中学习但不要reward"这个中间地带formalize清楚，并给出两个concrete的training strategy。

IWM的insight特别elegant：把world modeling作为auxiliary next-token prediction task，用同一组参数。这避免了separate simulator的overhead，且natural fit LLM fine-tuning paradigm。

SR的insight是**contrastive learning from own experience**：不是直接学expert action，而是学"为什么expert比我的alternative好"，这学到的decision criteria更generalizable。

实验设计rigorous：8个环境跨多个domain，3个model family，ablations全面，baseline comparison fair。特别是RL after early experience的实验直接验证了"bridge to RL"的claim。

这工作的positioning很聪明：不claim解决RL for language agents的难题，而是提供practical bridge。在reward signal unavailable的now，先让agent从experience中learn something useful，等RL infrastructure成熟后再seamlessly integrate。

参考链接：
- Paper: [arXiv:2507.08625](https://arxiv.org/abs/2507.08625)
- Silver & Sutton "Era of Experience": [Google AI blog](https://ai.googleblog.com/)
- GRPO (DeepSeekMath): [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- DPO: [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- STaR: [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
- WebArena: [arXiv:2307.13854](https://arxiv.org/abs/2307.13854)
