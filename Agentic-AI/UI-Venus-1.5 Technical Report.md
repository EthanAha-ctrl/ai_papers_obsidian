---
source_pdf: UI-Venus-1.5 Technical Report.pdf
paper_sha256: 59576ad84b083b24cfd560fbc41eb5b6617b2561cae9cea601d2f5524f8d7136
processed_at: '2026-08-12T19:03:20-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# UI-Venus-1.5 用人话版讲解

Andrej，咱们抛开那些学术黑话，用最直白的方式聊聊这篇paper到底干了啥。

---

## 1. 这帮人想解决什么问题？

想象一下，你想让AI帮你操作手机——比如"帮我在淘宝买双Nike跑鞋，尺码42，收货地址用公司"。听起来简单，但对AI来说这是地狱级难度。

**为什么难？** 三个核心痛点：

**痛点一：模型看不懂界面。** 现在的VLM（比如Qwen3-VL）虽然能描述图片内容，但它看GUI截图就像普通人看埃及象形文字——能认出几个图标，但不懂"这个按钮点了会跳转到哪""这个输入框该填什么"。通用预训练数据里压根没有足够的GUI结构化数据。

**痛点二：单步正确 ≠ 整体成功。** 这是最反直觉的发现。你训练模型，每一步action的accuracy都在涨，看起来很美好。但完整跑完一个task的成功率，到后期反而开始掉。为啥？因为step-level reward只管"这一步对不对"，不管"这十步连起来能不能完成任务"。就像考试每道题都会做，但组合起来就挂了。

**痛点三：静态数据训出来的模型在真实世界会拉胯。** 你用静态dataset训，模型记住的是"在这个特定screenshot下该点哪"。一到真实app，界面稍有变化，或者弹个广告、加载慢一点，模型就懵了。

UI-Venus-1.5就是针对这三个痛点设计的三连击解决方案。

Reference: [UI-Venus GitHub](https://github.com/inclusionAI/UI-Venus)

---

## 2. 他们的四步打法，用大白话讲

### 第一步：Mid-Training —— "恶补GUI知识"

**类比**：把一个聪明但没见过电脑的小孩，先送去上电脑基础课。

通用VLM就像个博学但没怎么碰过GUI的学者。你直接让它做RL，它连"这个红色按钮是删除""那个输入框是搜索框"都分不清，reward signal太sparse，根本learn不动。

所以他们搞了10B tokens的GUI data，分成四类：
- **Perception（20.8%）**：教模型认图标、widget state、OCR-free captioning
- **GUI-VQA（22.1%）**：教模型理解"这个组件是干嘛的"
- **Grounding（24.8%）**：教模型"找东西"——given一个instruction，在screenshot里定位element
- **Navigation-Reasoning（剩余）**：教模型chain-of-thought，把high-level goal拆成intermediate steps

**数据清洗的妙招**：开源dataset噪声很大，很多trace标着"完成"但实际没完成。他们用Qwen3-VL-235B-A22B当judge，给每条trace打0-10分：

- **7分以上**：gold pool，直接用
- **4-6分**：送给rewriting model重新加工instruction
- **0-3分**：扔掉或重建

这么一轮轮筛下来，high-fidelity data从69.7%涨到89.7%。

**更有意思的是Data Generation Loop**：他们搞了个闭环——让MLLM生成task prompt → 在真机上跑 → 收集trajectory → 把成功的trajectory喂回给MLLM当in-context example → 生成更好的prompt。循环几轮，trajectory generation success rate从17.9%飙到70%+。

这个intuition很简单——**你见过越多成功案例，就越会生成靠谱的任务描述**。这就是个bootstrapping的过程。

**验证Mid-Training效果**：他们用t-SNE可视化latent space，算了一堆metric：

| Metric | 之前 | 之后 | 变化 |
|--------|------|------|------|
| Silhouette Score | 0.235 | 0.315 | +34% ↑ |
| Intra-class Consistency | 0.448 | 0.396 | -11.6% ↓ |
| Inter-class Similarity | 0.220 | 0.223 | 几乎没变 |

翻译成人话——Mid-Training之后，GUI-related features在latent space里cluster更紧凑了，模型对fine-grained的GUI差异更敏感了，同时没有collapse掉通用能力。这说明GUI知识确实被"装进去了"。

---

### 第二步：Offline-RL —— "分科训练"

**类比**：高考前分文科理科集训，先把单科能力拉满。

这一步针对三个domain分别训练specialist model：
- **Grounding specialist**：专门练"找元素"
- **Mobile specialist**：专门练"操作手机"
- **Web specialist**：专门练"操作网页"

#### Grounding的reward，公式(1)：

$$R = R_{\text{format}} \cdot w_1 + R_{\text{point-in-box}} \cdot w_2$$

逐个拆解：
- $R_{\text{format}}$：检查output是不是合法的 `[x, y]` 格式。合法给1，不合法给0。这是binary reward。
- $R_{\text{point-in-box}}$：检查你predict的center point是不是落在ground-truth的bounding box里。落在里面给1，否则0。
- $w_1, w_2$：两个reward的权重，调trade-off的。

**Refusal capability是这里的大创新**。以前的grounding model，你问它"找到那个紫色独角兽按钮"，就算界面上根本没有，它也会硬给你predict一个坐标。这就导致hallucination——agent会去点不存在的东西。

UI-Venus-1.5教模型说："找不到就输出 `[-1, -1]`"。这是个很小的改动但意义深远——**从"永远要给答案"变成"知道什么时候该说没有"**。在VenusBench-GD的refusal subset上，他们的30B模型达到73.1%，而MAI-UI-32B、GTA1-32B这些baseline全是0.0%。

#### Navigation的reward，公式(2)：

$$R = w_1 \cdot R_{\text{format}} + w_2 \cdot R_{\text{action}}$$

$R_{\text{action}}$ 拆成两部分：
- $R_{\text{type}}$：binary，action type对不对（Click / Scroll / Type等）
- $R_{\text{content}}$ 或 $R_{\text{coord}}$：如果是Type这种文本action，算token-level F1；如果是Click这种坐标action，用hierarchical reward，gradually relax tolerance

**Hierarchical coordinate reward的intuition**：一开始tolerance很松，"大概在那个区域"就给reward，让model先学到rough location。然后慢慢收紧tolerance，逼model refine到precise location。这就是curriculum learning的思想——先学走路再学跑步。

#### Step-Trace Discrepancy现象

这一步训练时，他们观察到一个非常key的现象。看Figure 6的两条曲线：

- **Step-level success rate**：一路上涨，很稳定
- **Trace-level success rate**：先涨后跌，在中后期peak之后开始decline

为什么？因为step-level reward只管"这一步action对不对"，但忽略了：
1. **Error accumulation**：每一步差一点，十步下来就偏到十万八千里了
2. **Compositional effect**：单个action可能对，但组合起来不solve task
3. **Distribution shift**：训练数据里的GUI state和真实环境有gap

这个现象直接motivate了下一步——Online-RL，专门针对trace-level reward做优化。

---

### 第三步：Online-RL —— "真刀真枪在真机上练"

**类比**：驾校学车（Offline-RL）vs 上路实练（Online-RL）。驾校里你每个动作都对，但一到真实路况，遇到加塞的、闯红灯的、突然下雨的，你就慌了。

这一步是UI-Venus-1.5相对1.0最大的升级，也是工程量最大的部分。

#### DaaS：Device as a Service

Online-RL需要agent和真实environment交互。但你要训几千个concurrent的agent，就需要几千台device同时serve rollout。这是典型的systems bottleneck。

他们搞了个DaaS架构，核心组件：

**Group Control Gateway (GCGW)**：
- 本质是个高性能reverse proxy
- 抽象不同协议：ADB（Android）、CDP（Chrome）、SSH（Linux容器）
- **Secondary hash routing**：确保同一台device的requests路由到同一个gateway node。为啥重要？因为ADB和CDP这种stateful protocol依赖long-lived connection，如果M个gateway node都要连N台device，就会有M×N条connection，直接爆炸。Hash routing把这个问题降到M+N。
- Zero-copy I/O + streaming transmission：内部转发几乎零latency
- Coroutine-based concurrency model：适合device操作这种"高并发低频次"pattern

**Unified Client SDK**：封装device lifecycle管理（preemption、heartbeat、release）和unified semantic interaction interface。下游团队不用管protocol细节，专心训model就行。

**性能数字**：数千台heterogeneous devices，每日millions of requests，millisecond-level resource allocation，支持hundreds to thousands concurrent devices的RL training。

这个infrastructure的intuition——**Online-RL的scalability bottleneck从来不在算法本身，而在"怎么让几千台真机同时serve training"**。这是典型的systems + ML co-design问题，和强化学习中的REINVENT、AICA等infrastructure思路一脉相承。

Reference: [Ray RLlib](https://docs.ray.io/en/latest/rllib/) | [DeepMind IMPALA](https://arxiv.org/abs/1802.01561)

#### Task Generation：怎么造任务？

Online-RL的performance ceiling由task pool的diversity和quality决定。他们用混合策略：

**Static task library**：给MLLM一组app和website，让它extract functional map并生成common tasks。

**Dynamic trajectory-based generation**，公式(3)：
$$\mathcal{T}_{new} = \{q' \ : | \ : \psi(q', \mathcal{T}_{pool}) < \epsilon\}$$

拆解：
- $q'$：从offline trajectory里随机sample一个screenshot $s_t$，让MLLM推断"这个state下plausible的task是什么"
- $\mathcal{T}_{pool}$：已有task pool
- $\psi(q', \mathcal{T}_{pool})$：deduplication function，算semantic similarity
- $\epsilon$：threshold，新task必须和已有task足够不同才加入pool

这个design的intuition——**静态task pool覆盖不了long-tail interaction pattern，从真实trajectory里反推task能capture更丰富的场景**。

**Stratified Sampling by difficulty**：
- Easy: $N_{steps} \leq 10$
- Medium: $10 < N_{steps} \leq 20$  
- Hard: $N_{steps} > 20$

每个iteration按比例从三个bucket采样。这个curriculum learning的intuition——early training阶段reward sparse，先用easy task让policy bootstrap起来；后期逐渐增加hard task比例，学long-horizon planning。

#### Composite Reward：怎么给反馈？

公式(4)是整个Online-RL最核心的设计：

$$R(\tau) = \mathbb{1}_{success} \cdot R_{comp} \cdot \eta^{\frac{T - T_{min}}{T_{min}}} + \sum_{t=0}^{T} R_p(a_t)$$

逐项拆解：
- $\tau = (a_0, a_1, \dots, a_T)$：长度为$T$的trajectory
- $\mathbb{1}_{success}$：task成功的indicator，成功=1失败=0
- $R_{comp}$：task completion reward，比如固定值
- $\eta \in (0, 1]$：decay coefficient，比如0.9
- $T_{min}$：同一组trajectories里最短成功路径的步数
- $R_p(a_t)$：每一步的penalty

**Decay term的intuition**：$\frac{T - T_{min}}{T_{min}}$ 是"你比最短路径多走了多少步"的normalized ratio。比如最短路径10步，你走了20步，那excess ratio = $\frac{20-10}{10} = 1$，reward乘以 $\eta^1 = 0.9$。如果你走了30步，excess ratio = 2，reward乘以 $\eta^2 = 0.81$。

这个design直接打击一个常见问题——**agent学会绕圈子、做redundant action来"拖时间"**。有了decay term，走最短路径拿到的reward最高，agent自然学会高效完成任务。

公式(5) - Penalty：
$$R_p(a_t) = \begin{cases} -\lambda, & \text{if } a_t \text{ is unparseable} \\ 0, & \text{otherwise} \end{cases}$$

如果model输出的action无法被parser识别（格式错误、语法不对），给 $-\lambda$ 的penalty。这个design reduce invalid attempts，提升sample efficiency——**别让agent在format error上浪费rollout**。

**Dual-track verification**判断task是否成功：
- **Rule-based**：URL跳转、文件生成、系统设置变更这类deterministic outcome，直接query system API验证
- **MLLM-as-a-Judge**：semantically ambiguous的任务，把initial task $q$ 和final keyframe $s_i$ 喂给MLLM判断逻辑intent是否satisfied

这个dual-track的intuition——**有些task有明确的programmatic signal，有些task需要semantic understanding**。比如"把这篇文档设为只读"可以用API验证，但"帮我找到最便宜的机票"就需要MLLM判断结果是否合理。

#### GRPO训练算法

公式(6) - GRPO loss：

$$L_{GRPO}(\theta) = -\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|\tau_i|} \sum_{t=1}^{|\tau_i|} \min\left(r_{i,t}(\theta) \hat{A}_i, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i\right)$$

这个公式看着复杂，拆开看：
- $G$：group size，每个task采样$G$条trajectory
- $|\tau_i|$：第$i$条trajectory的length
- $r_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t} | s_{i,t}, q)}{\pi_{\theta_{old}}(a_{i,t} | s_{i,t}, q)}$：importance sampling ratio，新policy和旧policy在某个action上的概率比
- $\epsilon$：PPO的clip range，防止policy update太大
- $\hat{A}_i$：trajectory-level advantage

这就是标准PPO的clipped objective，只是apply到trajectory level。

**Key design——Trajectory-level Advantage**，公式(7)：

$$\hat{A}_i = \frac{R(\tau_i, q) - \text{mean}(\{R(\tau_j, q)\}_{j=1}^G)}{\text{std}(\{R(\tau_j, q)\}_{j=1}^G) + \epsilon}$$

拆解：
- $R(\tau_i, q)$：第$i$条trajectory的composite reward（公式4算出来的）
- $\text{mean}(\{R(\tau_j, q)\}_{j=1}^G)$：同组$G$条trajectory的mean reward
- $\text{std}(\{R(\tau_j, q)\}_{j=1}^G)$：同组reward的standard deviation
- $\epsilon$：numerical stability的small constant

这就是z-score normalization。每条trajectory的advantage是"相对于group平均好多少/差多少"。

**为什么用trajectory-level而不用step-level？** 这直接address了Section 2.2的step-trace discrepancy。GUI task的critical action难以identify——你很难说"第5步那个click是task成功的关键"。与其guess哪个step重要，不如把整个trajectory当做一个unit来评估。

Group-relative normalization的intuition——**同一task的多条trajectory面临相同的environment stochasticity，group内部competition可以filter掉环境噪声，提供更stable的credit assignment signal**。

#### Training Stability Mechanisms

Online-RL容易collapse，他们加了两个regularization：

**Adaptive KL Constraint**，公式(8)：
$$L_{KL}(\theta) = \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref})$$

$\beta$ 是KL penalty的weight，$\pi_{ref}$ 是reference policy（通常是SFT/Offline-RL后的model）。

但固定reference policy会限制progress，所以他们做了adaptive update，公式(9)：
$$\pi_{ref} \leftarrow (1-\alpha) \pi_{ref} + \alpha \pi_\theta$$

当current policy在validation set上outperform $\pi_{ref}$ by margin $\delta$时，把两个policy做exponential moving average。这个design的intuition——**reference policy要跟着policy一起进步，但变化不能太快否则constraint失效**。

**Annealed Entropy Regularization**，公式(10)：
$$L_{entropy}(\theta) = -\lambda_t \mathbb{H}(\pi_\theta(\cdot | s, q))$$

$\mathbb{H}$ 是policy distribution的entropy。公式(11)给annealing schedule：
$$\lambda_t = \lambda_0 \cdot \sigma^k, \quad \sigma \in (0, 1)$$

- $\lambda_0$：initial entropy coefficient，设得比较大
- $\sigma$：每步decay rate，比如0.999
- $k$：training step

这个annealing的intuition——early training阶段，model从SFT/Offline-RL继承来的policy过于deterministic，high entropy coefficient鼓励exploration；后期逐渐降低entropy weight，让policy converge到deterministic optimal。

公式(12) - Total objective：
$$J(\theta) = L_{GRPO}(\theta) - L_{KL}(\theta) + L_{entropy}(\theta)$$

注意符号——KL是penalty所以减，entropy是bonus所以加。这和标准RLHF的formulation一致。

Reference: [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300) | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | [PPO原论文](https://arxiv.org/abs/1707.06347)

---

### 第四步：Model Merge —— "三合一"

**类比**：三个专科医生（grounding专家、mobile专家、web专家）合成一个全科医生。

这一步解决deployment问题——总不能让用户根据task类型switch model吧？

#### Linear Merge，公式(12)：

$$\theta_{linear} = \sum_{i=1}^{3} w_i \cdot \theta_i, \quad \text{subject to} \quad \sum_{i=1}^{3} w_i = 1$$

- $\theta_i$：第$i$个specialist model的parameters
- $w_i$：weight，比如grounding 0.4 / mobile 0.3 / web 0.3

这就是weighted average，简单粗暴。

#### TIES-Merge：更聪明的merge方式

TIES分两步：

**Step 1 - Task vector pruning**：
算task vector = fine-tuned model - base model。这个vector编码了"fine-tuning学到了什么"。然后prune low-magnitude updates——只保留significant changes，去掉noise。

**Step 2 - Sign election**：
对每个parameter，看三个specialist的task vector sign是什么。如果有两个是正一个是负，就elect正为dominant sign，只aggregate sign-aligned的updates。

**TIES优于Linear的intuition**：Linear merge直接average，conflicting updates会互相cancel掉。TIES通过pruning和sign election减少interference，保留每个specialist的核心knowledge。

#### 实验对比

UI-Venus-1.5-30B-A3B：
- **Merge前**：ScreenSpot-Pro 71.0%, AndroidWorld 75.5%
- **Linear Merge**：68.1% (-2.9%↓), 73.2% (-2.3%↓) —— 两个都掉了
- **TIES-Merge**：69.6% (-1.4%↓), 77.6% (+2.1%↑) —— grounding掉一点，navigation反而涨了

TIES在navigation上的+2.1% gain很interesting。可能的解释——**grounding specialist的precise localization能力和navigation specialist的sequential planning能力通过merge发生了positive transfer**。grounding能力帮助navigation更准确地click target，从而提升整体success rate。

Reference: [TIES-Merging](https://arxiv.org/abs/2311.03003) | [Model Soups](https://arxiv.org/abs/2203.05482) | [Task Arithmetic](https://arxiv.org/abs/2212.04089)

---

## 3. 实验结果：到底有多强？

### Grounding Benchmark（Table 1）

挑几个highlight：

| Benchmark | UI-Venus-1.5-30B-A3B | 最强baseline | 差距 |
|-----------|---------------------|-------------|------|
| VenusBench-GD | **75.0%** | UI-Venus-1.0-72B (70.2%) | +4.8% |
| ScreenSpot-Pro | **69.6%** | MAI-UI-32B (67.9%) | +1.7% |
| OSWorld-G-R | **76.4%** | MAI-UI-32B (73.9%) | +2.5% |
| UI-Vision | **54.7%** | MAI-UI-32B (47.1%) | +7.6% |

UI-Vision上+7.6%的margin特别impressive。这个benchmark强调real-world application和fine-grained reasoning，说明UI-Venus-1.5在实际应用场景的grounding能力有明显优势。

**Scaling consistent**：ScreenSpot-Pro上，2B 57.7% → 8B 68.4% → 30B-A3B 69.6%。越大约越强，符合scaling law预期。

### Navigation Benchmark

**AndroidWorld**：
- UI-Venus-1.5-30B-A3B: **77.6%**
- MAI-UI-32B: 73.3%
- +4.3% margin

**VenusBench-Mobile**（他们的internal benchmark，特别难）：
- UI-Venus-1.5-30B-A3B: **21.5%**
- UI-Venus-1.0-72B: 15.4%
- +6.1% margin

**WebVoyager**：
- UI-Venus-1.5-30B-A3B: 76.0%
- Holo2-30B-A3B: 83.0%
- OpenAI-CUA: 87.0%

WebVoyager上和Holo2、OpenAI-CUA有gap。说明**web scenario的specialization还有提升空间**。可能原因——web interaction比mobile更复杂（Hover、Hotkey、DoubleClick等action），而且web layout diversity更高。

### Ablation Study（Table 7）——每个stage贡献多少？

| Stage | 2B on AW | 8B on AW | 30B-A3B on AW |
|-------|---------|---------|---------------|
| Mid-Training | 39.0 | 57.0 | 67.1 |
| +Offline-RL | 45.3 (+6.3) | 63.5 (+6.5) | 68.0 (+0.9) |
| +Online-RL | 59.8 (+14.5) | 72.7 (+9.2) | 75.5 (+7.5) |
| +Model Merge | 55.6 (-4.2) | 73.7 (+1.0) | 77.6 (+2.1) |

几个insight：

1. **Online-RL是最critical的stage**，尤其对小model。2B model在Online-RL阶段gain了+14.5%，这非常夸张。intuition——small model的prior knowledge弱，更需要exploration来compensate。

2. **Model Merge对大model反而有gain**。2B merge后掉了4.2%，但30B-A3B merge后涨了2.1%。说明大model的parameter space更冗余，merge时interference更小，cross-task transfer更容易发生。

3. **Offline-RL的gain随scale递减**。2B gain 6.3%，30B-A3B只gain 0.9%。大model在Mid-Training阶段已经学得差不多了，Offline-RL的marginal收益递减。

---

## 4. 一些值得深挖的技术联想

### 4.1 Step-Trace Discrepancy和Reward Hacking

这个现象其实和reward hacking有关。step-level reward优化local action correctness，但model可能学到"每一步看起来都对但整体不solve task"的policy。这和LLM alignment里的reward hacking本质相同——reward signal和true objective之间有gap。

相关work：
- [Reward Hacking](https://arxiv.org/abs/2204.06574)
- [Speculative Decoding for RL](https://arxiv.org/abs/2302.01318)

### 4.2 Refusal Capability和Hallucination Mitigation

教模型说"找不到"这个idea非常practical。传统grounding model总是output坐标，即使element不存在。这导致agent在real deployment中会hallucinate——点不存在的东西。

这个思想可以extend到更多scenario：
- "这个button当前不可点击" → refusal
- "这个task在当前界面无法完成" → refusal
- "需要更多信息才能继续" → CallUser

和LLM领域的selective prediction、abstention相关：
- [Self-Consistency for Abstention](https://arxiv.org/abs/2305.17306)
- [Calibration of LLMs](https://arxiv.org/abs/2305.14975)

### 4.3 Model Merge和Modular Knowledge

TIES-Merge的效果暗示了一个interesting direction——**不同task的knowledge可以modular地存储在model parameters里，通过arithmetic operation组合**。

这和LLM领域的task arithmetic、model soups一脉相承。未来可能的方向：
- **Dynamic merging at inference time**：根据input type动态选择merge weight
- **Continual learning via merging**：新task训一个specialist然后merge进去，avoid catastrophic forgetting
- **Mixture of Experts as implicit merging**：MoE本质上就是soft merging

Reference: [Model Soups](https://arxiv.org/abs/2203.05482) | [Task Arithmetic](https://arxiv.org/abs/2212.04089)

### 4.4 DaaS和Scalable RL Infrastructure

DaaS架构本质上解决了一个scale问题——**怎么让几千台real device同时serve RL training的rollout需求**。

这和以下系统思路相关：
- [Ray](https://docs.ray.io/)：distributed computing framework
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)：MS的training optimization
- [Anyscale](https://www.anyscale.com/)：Ray背后的公司

Key challenge——device operation是"高并发低频次"pattern，和typical ML workload（低并发高频次）不同。Coroutine-based concurrency model + hash routing + zero-copy I/O是合理的engineering choices。

### 4.5 Entropy Annealing和Exploration-Exploitation

公式(11)的entropy annealing：
$$\lambda_t = \lambda_0 \cdot \sigma^k$$

这和Simulated Annealing的思想完全一致——early stage高entropy鼓励exploration，late stage低entropy converge到optimal。

更deep的connection——这和LLM RLHF里的KL penalty annealing、AlphaGo里的temperature scheduling都是同一个pattern：**训练初期鼓励diversity，后期收敛到deterministic optimal**。

Reference: [Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing) | [AlphaGo Nature paper](https://www.nature.com/articles/nature16961)

### 4.6 Trajectory-level Advantage和Credit Assignment

公式(7)的trajectory-level advantage：
$$\hat{A}_i = \frac{R(\tau_i, q) - \text{mean}(\{R(\tau_j, q)\}_{j=1}^G)}{\text{std}(\{R(\tau_j, q)\}_{j=1}^G) + \epsilon}$$

这是对credit assignment problem的一个practical solution。传统RL的credit assignment很难——trajectory有几十步，哪一步是critical action？

他们的approach——**放弃step-level credit assignment，直接用trajectory-level reward做group-relative normalization**。同一task的多条trajectory相互competition，filter out环境噪声。

这和以下方向相关：
- [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)
- [Reward Shaping](https://arxiv.org/abs/1907.02057)
- [Advantage Weighted Actor-Critic](https://arxiv.org/abs/2006.09359)

### 4.7 Decay Coefficient和Path Efficiency

公式(4)的decay term $\eta^{\frac{T - T_{min}}{T_{min}}}$ 很有意思。这直接penalize冗余step，鼓励shortest path。

这和robotics里的minimum-time optimal control、以及LLM里的token efficiency optimization异曲同工。本质都是——**不仅要完成任务，还要高效完成**。

Reference: [Optimal Control](https://en.wikipedia.org/wiki/Optimal_control) | [Token Efficiency in LLMs](https://arxiv.org/abs/2305.03393)

---

## 5. Limitations和未来方向

### 潜在limitations

1. **WebVoyager gap**：76% vs Holo2的83%和OpenAI-CUA的87%。web scenario需要更多specialization。

2. **Chinese app focus**：40+ Chinese apps的optimization可能limit cross-cultural generalization。西方app的UI pattern、interaction logic可能不同。

3. **Merge的grounding trade-off**：即使TIES-Merge，grounding还是掉1.4%。有没有更好的merge方法？比如learnable merge weight、attention-based merge？

4. **Online-RL的cost**：需要上千台real device，financial cost高。能不能用simulator pre-train再fine-tune到real device？类似sim-to-real transfer。

5. **VenusBench-Mobile绝对值低**：21.5%说明complex mobile task仍然是hard problem。有些task可能需要20+ steps，error accumulation严重。

### 未来可能的方向

1. **Test-time scaling**：结合GTA1的test-time scaling思想，inference时sample多条trajectory然后select best。参考[GTA1 paper](https://arxiv.org/abs/2507.05791)

2. **Multi-modal fusion**：融合XML / accessibility tree和visual input。很多GUI agent用screenshot only，但XML信息可以提供structured layout info。

3. **Self-improving loop**：类似UI-Genie的self-improving approach，让agent自动发现weakness并generate针对性training data。参考[UI-Genie](https://arxiv.org/abs/2505.21496)

4. **Cross-platform transfer**：Desktop / IoT / VR等更多platform。action space需要extend。

5. **Memory mechanism**：当前model只有previous_actions作为memory。可以引入external memory（类似retrieval-augmented generation）来处理ultra-long-horizon task。

6. **Hierarchical planning**：high-level planner分解goal → sub-goals，low-level executor执行每个sub-goal。参考Mobile-Agent-E的hierarchical framework。

7. **Active learning**：让model自己标记uncertain的state，主动query human feedback。减少annotation cost。

8. **World model for GUI**：learn一个predictive model预测"action执行后GUI会变成什么样"，用于model-based RL planning。这能大幅reduce real interaction cost。

Reference: [Mobile-Agent-E](https://arxiv.org/abs/2501.11733) | [World Models](https://arxiv.org/abs/1803.10122) | [Model-Based RL](https://arxiv.org/abs/1906.08226)

---

## 6. 总结：核心intuition

用最简单的话总结UI-Venus-1.5的四个stage：

**Mid-Training**：先让model"上电脑课"，恶补GUI基础知识。解决"看不懂界面"的问题。

**Offline-RL**：分科集训，grounding / mobile / web三个specialist各自练强。解决"不会做具体task"的问题。但发现step-level优化和trace-level success有gap。

**Online-RL**：上真机实练，在dynamic environment里学error recovery和long-horizon planning。用trajectory-level reward直接优化trace success rate。解决"静态数据训出来在real world会拉胯"的问题。

**Model Merge**：三个specialist合成一个end-to-end agent。用TIES-Merge减少interference。解决"deployment要simple"的问题。

这四步构成了一个从**knowledge → skill → adaptation → unification**的完整pipeline。每一步都address前一步的limitation，层层递进。这种layered design philosophy对于build practical AI system非常有启发——**不要指望一个stage solve所有问题，每个stage focus on一件事，然后把它们compose起来**。

最后的最后，这篇paper给我最大的启发是——**GUI Agent的bottleneck不在算法novelty，而在engineering execution**。DaaS infrastructure、data refinement pipeline、reward design的细节，这些"脏活累活"才是决定实际performance的关键。这和Tesla FSD、Google Gemini等industrial system的insight一致——**scale和execution比algorithm insight更重要**。

Reference:
- [UI-Venus GitHub](https://github.com/inclusionAI/UI-Venus) | [HuggingFace Models](https://huggingface.co/collections/inclusionAI/ui-venus)
- [Qwen3-VL](https://arxiv.org/abs/2511.21631) | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | [GRPO](https://arxiv.org/abs/2402.03300)
- [TIES-Merging](https://arxiv.org/abs/2311.03003) | [Model Soups](https://arxiv.org/abs/2203.05482)
- [AndroidWorld](https://arxiv.org/abs/2405.14573) | [ScreenSpot-Pro](https://arxiv.org/abs/2412.05685)
- [UI-TARS](https://arxiv.org/abs/2501.12326) | [MAI-UI](https://arxiv.org/abs/2512.22047) | [Holo2](https://github.com/huggingface/open-r1)
- [PPO](https://arxiv.org/abs/1707.06347) | [AlphaGo](https://www.nature.com/articles/nature16961)
- [UI-Genie](https://arxiv.org/abs/2505.21496) | [GTA1](https://arxiv.org/abs/2507.05791)

---

# UI-Venus-1.5 技术报告深度解析

Andrej，这篇paper来自Ant Group的Venus Team，是GUI Agent领域一个相当solid的工程化工作。让我从intuition层面帮你拆解。

## 1. 核心问题与设计哲学

GUI Agent领域目前面临的fundamental tension在于：**step-level accuracy与trace-level accuracy的mismatch**。这篇paper的核心观察是——在SFT和Offline-RL阶段，模型的per-step success rate持续上升，但完整trajectory的成功率会先peak后decline。这个现象的本质是：step-level reward只能优化local action，却无法credit assignment到long-horizon的compositional success。

UI-Venus-1.5的设计哲学可以概括为**"分而治之，最后融合"**——先用Mid-Training注入GUI domain knowledge，再用Offline-RL做task-specific specialization，接着用Online-RL解决dynamic environment的exploration问题，最后用Model Merge把三个specialist统一成一个end-to-end agent。

Reference: [UI-Venus GitHub](https://github.com/inclusionAI/UI-Venus) | [HuggingFace Models](https://huggingface.co/collections/inclusionAI/ui-venus)

---

## 2. 四阶段Training Pipeline深度拆解

### 2.1 Stage 1: Mid-Training —— GUI Knowledge Injection

**Motivation的intuition**: General VLM（如Qwen3-VL）在pre-training corpus中缺乏GUI-specific的结构化建模。这导致进入RL阶段时，reward signal过于sparse，policy无法bootstrap。Mid-Training的本质是**在进入RL之前，先把GUI的"先验知识"通过supervised方式注入到latent space**。

**数据构成**（10B tokens, 30+ datasets）：
- Semantic Perception: 20.8%
- GUI-VQA: 22.1%
- Grounding: 24.8%
- Hybrid Navigation-Reasoning: 剩余部分

数据来源包括Mind2Web、ShowUI、AITW等。这里有一个值得注意的design choice——他们把数据分层为perception / reasoning / action alignment三个axis，这其实对应了GUI Agent需要的三个核心能力维度。

**Iterative Data Refinement**:
使用Qwen3-VL-235B-A22B作为judge，对traces打0-10分：
- Score ≥ 7: 保留到gold pool
- Score 4-6: 送入rewriting model重新refine instruction
- Score 0-3: 重建或丢弃

经过recursive refinement，high-fidelity samples比例从69.7% → 89.7%。这里有一个intuition——**开源GUI dataset的噪声主要来源于trace的不完整性**（task没真正完成但被记录为完成），用teacher model做reachability check是合理的。

**Data Generation Loop（DaaS-based）**:
这是工程上最impressive的部分之一。他们构建了一个闭环：
1. MLLM从seed instructions生成candidate task prompts
2. Embedding similarity做deduplication
3. 在cloud-hosted real devices上执行
4. GUI trajectory scraping + multi-annotator verification
5. Verified trajectories作为in-context examples反馈给MLLM

Success rate从17.9% → 70%+，总共收集30,000+ verified trajectories。这个iterative bootstrapping的intuition是——**static dataset无法捕捉execution failure和GUI dynamics，real-device interaction data才能反映deployment distribution**。

**Latent Space验证**（Table 6）:
用t-SNE可视化验证Mid-Training的效果：

| Metric | Qwen3-VL | After Mid-Training | Change |
|--------|----------|-------------------|--------|
| Silhouette Score | 0.235 | 0.315 | +34.0% ↑ |
| Intra-class Consistency | 0.448 | 0.396 | -11.6% ↓ |
| Inter-class Similarity | 0.220 | 0.223 | +1.4% (stable) |

这里的intuition——Silhouette Score上升表示cluster更separable；Intra-class Consistency下降说明模型对fine-grained functional variance更sensitive；Inter-class Similarity稳定说明没有representation collapse。这是一个非常clean的验证，说明Mid-Training确实enriched了GUI-specific features。

---

### 2.2 Stage 2: Offline-RL —— Task-Specific Specialization

#### 2.2.1 Grounding的Reward设计

公式(1):
$$R = R_{\text{format}} \cdot w_1 + R_{\text{point-in-box}} \cdot w_2$$

变量解析：
- $R_{\text{format}}$：binary reward，检查output string是否符合 `[x, y]` 的预定义syntax
- $R_{\text{point-in-box}}$：经典的point-in-box reward，预测的center point是否落在ground-truth bounding box内
- $w_1, w_2$：control format correctness与location precision的相对权重

**Refusal Samples** 是一个key innovation。当instruction指向的element不存在于image中时，模型被训练输出 `[-1, -1]`。这个设计的intuition是——**hallucination在GUI grounding中是致命的**，因为agent会去点击不存在的元素导致任务失败。引入refusal capability虽然在ScreenSpot-Pro这类没有refusal examples的benchmark上略有trade-off，但在VenusBench-GD和OSWorld-G-Refine上显著提升。

#### 2.2.2 Navigation的Reward设计

公式(2):
$$R = w_1 \cdot R_{\text{format}} + w_2 \cdot R_{\text{action}}$$

其中 $R_{\text{action}}$ 进一步分解：
$$R_{\text{action}} = R_{\text{type}} + \begin{cases} R_{\text{content}} & \text{if text-based action} \\ R_{\text{coord}} & \text{if coordinate-based action} \end{cases}$$

- $R_{\text{type}}$：binary，预测的action type是否match ground-truth type（Click / Scroll / Type等）
- $R_{\text{content}}$：token-level F1 score between predicted and ground-truth content
- $R_{\text{coord}}$：hierarchical reward，gradually relax tolerance on coordinate errors

**Hierarchical coordinate reward的intuition**：如果一开始就用tight tolerance，reward landscape过于sparse，policy gradient信号弱；gradually relax可以让model先学到rough location再refine到precise location，这是一个curriculum learning的思想。

**Step vs Trace Discrepancy**（Figure 6）:
这是整篇paper最重要的empirical observation之一。在Mobile和Web两个scenario下，step-level success rate持续上升，但trace-level success rate在中后期开始decline。

这个现象的root cause——Offline-RL的step-level reward只优化local action correctness，但忽略了：
1. Action之间的compositional effect
2. Error accumulation across long horizon
3. State distribution shift between training data and real environment

这个observation直接motivate了Stage 3的Online-RL。

---

### 2.3 Stage 3: Online-RL —— Dynamic Environment Interaction

这是UI-Venus-1.5相对于1.0最大的upgrade，也是工程量最大的部分。

#### 2.3.1 DaaS架构

**Group Control Gateway (GCGW)**:
- Centralized reverse proxy + orchestration core
- 抽象heterogeneous protocols: ADB (Android), CDP (Chrome), SSH (Linux containers)
- **Secondary hash routing algorithm**: 确保同一device的requests路由到同一gateway node，避免 "M×N connection explosion" problem
- Streaming transmission + zero-copy I/O for internal forwarding
- Coroutine-based high-concurrency model

**Unified Client SDK**:
- 自动化device lifecycle management: preemption, heartbeat maintenance, resource release
- Unified semantic interaction interface across protocols

性能benchmark：
- 数千台heterogeneous devices
- 每日millions of operation requests
- Millisecond-level resource allocation
- 支持hundreds to thousands concurrent devices的RL training

这里的intuition——**Online-RL的scalability bottleneck不在算法，而在infrastructure**。如何让上千台real device同时serve RL training的rollout需求，是一个典型的systems + ML co-design问题。

#### 2.3.2 Task Generation与Stratified Sampling

公式(3) - Dynamic trajectory-based generation:
$$\mathcal{T}_{new} = \{q' \ : | \ : \psi(q', \mathcal{T}_{pool}) < \epsilon\}$$

变量解析：
- $q'$：MLLM从sampled screenshot $s_t$ 推断出的plausible task query
- $\mathcal{T}_{pool}$：已有task pool
- $\psi(\cdot, \cdot)$：deduplication function（基于semantic similarity）
- $\epsilon$：similarity threshold，promote uniform coverage of task space

**Stratified Sampling by difficulty**:
- Easy: $N_{steps} \leq 10$
- Medium: $10 < N_{steps} \leq 20$
- Hard: $N_{steps} > 20$

每个training iteration按比例从三个bucket采样，支持structured curriculum learning。这个design的intuition——**如果只用hard tasks，early training阶段reward signal过于sparse，policy无法improve；如果只用easy tasks，model无法学到long-horizon planning**。

#### 2.3.3 Composite Reward设计

公式(4):
$$R(\tau) = \mathbb{1}_{success} \cdot R_{comp} \cdot \eta^{\frac{T - T_{min}}{T_{min}}} + \sum_{t=0}^{T} R_p(a_t)$$

变量解析：
- $\tau = (a_0, a_1, \dots, a_T)$：长度为$T$的execution trajectory
- $\mathbb{1}_{success}$：indicator function，task是否成功完成
- $R_{comp}$：task completion reward
- $\eta \in (0, 1]$：trace length decay coefficient
- $T_{min}$：同组trajectories中最少的成功步数
- $R_p(a_t)$：step $t$ 的action constraint penalty

**Decay coefficient $\eta$的intuition**：$\frac{T - T_{min}}{T_{min}}$ 是一个normalized excess step ratio。如果trajectory用的步数比最短路径多，reward会exponentially decay。这鼓励agent学习shortest operational path，suppress redundant或circular actions。

公式(5) - Penalty term:
$$R_p(a_t) = \begin{cases} -\lambda, & \text{if } a_t \text{ is unparseable} \\ 0, & \text{otherwise} \end{cases}$$

$\lambda$ 是negative penalty for invalid actions。这个设计reduce invalid attempts during exploration，提升sample efficiency。

**Dual-track verification**:
- Rule-based: URL redirection, file generation, system setting changes等deterministic outcomes
- MLLM-as-a-Judge: semantically ambiguous tasks，用MLLM judge final keyframe screenshot是否satisfy logical intent

#### 2.3.4 GRPO训练算法

公式(6) - GRPO loss:
$$L_{GRPO}(\theta) = -\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|\tau_i|} \sum_{t=1}^{|\tau_i|} \min\left(r_{i,t}(\theta) \hat{A}_i, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i\right)$$

变量解析：
- $G$：group size，每个task采样$G$条trajectories
- $|\tau_i|$：第$i$条trajectory的长度
- $r_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t} | s_{i,t}, q)}{\pi_{\theta_{old}}(a_{i,t} | s_{i,t}, q)}$：importance sampling ratio
- $\epsilon$：PPO clip range
- $\hat{A}_i$：trajectory-level advantage

公式(7) - Trajectory-level advantage:
$$\hat{A}_i = \frac{R(\tau_i, q) - \text{mean}(\{R(\tau_j, q)\}_{j=1}^G)}{\text{std}(\{R(\tau_j, q)\}_{j=1}^G) + \epsilon}$$

**Key design choice**: 用trajectory-level reward而非step-level reward。这直接解决了Section 2.2中观察到的step-trace discrepancy问题。$\hat{A}_i$ uniformly assigned to all action steps within trajectory。

这里的intuition——**GUI任务的critical action难以identify**，step-wise reward信号noisy。用group-relative normalization可以filter out environmental stochasticity，提供stable credit assignment signal。

#### 2.3.5 Training Stability Mechanisms

**Adaptive KL Constraint** (公式8-9):
$$L_{KL}(\theta) = \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref})$$

$$\pi_{ref} \leftarrow (1-\alpha) \pi_{ref} + \alpha \pi_\theta$$

当current policy在held-out validation set上outperform $\pi_{ref}$ by margin $\delta$时，smoothly blend两个policies。这个adaptive update的intuition——**static reference policy会变成stationary constraint限制progress，而fully dynamic reference又会导致training不稳定**。Adaptive blending是两者的trade-off。

**Annealed Entropy Regularization** (公式10-11):
$$L_{entropy}(\theta) = -\lambda_t \mathbb{H}(\pi_\theta(\cdot | s, q))$$

$$\lambda_t = \lambda_0 \cdot \sigma^k, \quad \sigma \in (0, 1)$$

变量解析：
- $\lambda_0$：initial entropy coefficient
- $\sigma$：decay rate per training step $k$
- $\mathbb{H}$：entropy of policy distribution

**Annealing的intuition**——early training阶段policy过于deterministic（从SFT/Offline-RL继承），high $\lambda_t$鼓励exploration；后期逐渐降低$\lambda_t$，让policy converge到optimal deterministic policy。这是经典的exploration-exploitation trade-off。

公式(11) - Total objective:
$$J(\theta) = L_{GRPO}(\theta) - L_{KL}(\theta) + L_{entropy}(\theta)$$

---

### 2.4 Stage 4: Model Merge —— Unification

#### Linear Merge (公式12):
$$\theta_{linear} = \sum_{i=1}^{3} w_i \cdot \theta_i, \quad \text{subject to} \quad \sum_{i=1}^{3} w_i = 1$$

- $\theta_i$：第$i$个specialized model（grounding / web / mobile）的parameters
- $w_i$：relative importance weight

#### TIES-Merge:
两步关键操作：
1. **Task vector pruning**: 计算task vectors（fine-tuned model - base model），prune low-magnitude updates，只retain most significant changes
2. **Sign election**: 对每个parameter elect一个dominant sign direction，只aggregate aligned updates

**TIES优于Linear的intuition**——Linear merge会average掉conflicting updates，导致performance regression。TIES通过pruning noisy updates和resolving sign conflicts，显著降低interference。

**实验对比**（Section 3.4）:
UI-Venus-1.5-30B-A3B在merge前：ScreenSpot-Pro 71.0%, AndroidWorld 75.5%
- Linear Merge: 68.1% (-2.9%↓), 73.2% (-2.3%↓)
- TIES-Merge: 69.6% (-1.4%↓), 77.6% (+2.1%↑)

TIES-Merge不仅在grounding上loss更小，在navigation上反而有gain。这个现象的intuition——**cross-task knowledge transfer通过merge发生**，grounding specialist的precise localization能力与navigation specialist的sequential planning能力互相enhance。

Reference: [TIES-Merging paper](https://arxiv.org/abs/2311.03003) | [Deep Model Fusion survey](https://arxiv.org/abs/2309.15698)

---

## 3. 实验结果深度解读

### 3.1 Grounding Benchmarks（Table 1）

| Benchmark | UI-Venus-1.5-30B-A3B | Strongest Baseline | Margin |
|-----------|---------------------|-------------------|--------|
| VenusBench-GD | **75.0%** | UI-Venus-1.0-72B (70.2%) | +4.8% |
| ScreenSpot-Pro | **69.6%** | MAI-UI-32B (67.9%) | +1.7% |
| ScreenSpot-V2 | 96.2% | MAI-UI-32B (96.5%) | -0.3% |
| MMBench-GUI L2 | 88.6% | MAI-UI-32B (91.3%) | -2.7% |
| OSWorld-G-R | **76.4%** | MAI-UI-32B (73.9%) | +2.5% |
| OSWorld-G | **70.6%** | MAI-UI-32B (67.6%) | +3.0% |
| UI-Vision | **54.7%** | MAI-UI-32B (47.1%) | +7.6% |

几个key observations：
1. **Scaling consistent**: ScreenSpot-Pro 57.7% → 68.4% → 69.6% (2B / 8B / 30B-A3B)
2. **Refusal capability的trade-off**: 在VenusBench-GD的refusal subset上，UI-Venus-1.5-30B-A3B达到73.1%，而几乎所有baseline都是0.0%
3. **UI-Vision的+7.6% margin**特别impressive，这个benchmark强调real-world applications和fine-grained reasoning

### 3.2 Navigation Benchmarks

**AndroidWorld**（Table 2）:
- UI-Venus-1.5-30B-A3B: **77.6%**
- MAI-UI-32B: 73.3%
- Margin: +4.3%

**AndroidLab**（Table 3）:
- UI-Venus-1.5-8B: 55.1% / 68.1%† (human-verified)
- UI-Venus-1.5-30B-A3B: 52.9% / 68.1%†
- UI-Venus-1.0-72B: 49.3%

这里有一个interesting observation——8B模型在这个benchmark上反而比30B-A3B略好（未verified版本）。Paper解释是AndroidLab官方evaluation code有bug，human-verified后两者持平。

**VenusBench-Mobile**（Table 4）:
- UI-Venus-1.5-30B-A3B: **21.5%**
- UI-Venus-1.0-72B: 15.4%
- Margin: +6.1%

这个benchmark特别challenging，绝对值低但相对gain大。

**WebVoyager**（Table 5）:
- UI-Venus-1.5-30B-A3B: 76.0%
- Holo2-30B-A3B: 83.0%
- OpenAI-CUA: 87.0%

WebVoyager上UI-Venus-1.5没有达到SOTA，与Holo2和OpenAI-CUA有gap。这表明**web scenario的specialization还有提升空间**。

### 3.3 Ablation Studies（Table 7）

这是理解每个stage贡献的关键table：

| Model | Mid-Training | Offline-RL | Online-RL | Model Merge |
|-------|-------------|-----------|----------|------------|
| | SS-Pro / AW | SS-Pro / AW | SS-Pro / AW | SS-Pro / AW |
| 2B | 52.3 / 39.0 | 59.0 / 45.3 | - / 59.8 | 57.7 / 55.6 |
| 8B | 63.1 / 57.0 | 70.0 / 63.5 | - / 72.7 | 68.4 / 73.7 |
| 30B-A3B | 65.2 / 67.1 | 71.0 / 68.0 | - / 75.5 | 69.6 / 77.6 |

Key insights：
1. **Offline-RL**: consistent +6-7% on ScreenSpot-Pro, +0.9-6.5% on AndroidWorld
2. **Online-RL**: 最critical的stage for navigation，2B model +14.5% on AndroidWorld
3. **Model Merge**: grounding略降~1.4%，但navigation对大模型有+2.1% gain

Online-RL对2B model的+14.5% gain特别impressive——这说明**small model从dynamic environment interaction中获益更大**，因为small model的prior knowledge更弱，更需要exploration来compensate。

---

## 4. Action Space设计（Table 8）

UI-Venus-1.5的action space统一了mobile和web interaction：

**Mobile primitives**: Click, Drag, Scroll, Type, Launch, Wait, Finished, CallUser, LongPress, PressBack, PressHome, PressEnter, PressRecent

**Web-specific additions**: Hover, DoubleClick, Hotkey

**Domain-specific constraints**:
- Mobile Scroll: 必须predict precise start和end coordinates
- Web Scroll: 只需specify direction (up/down)

这个unified action space的intuition——**mobile和web的interaction modality有overlap也有divergence**。Click / Type / Scroll是shared primitives，Hover / Hotkey是web-specific（mobile没有mouse hover概念）。Domain-specific constraints则反映了platform的different operational logics。

---

## 5. 技术联想与相关work

### 5.1 与UI-Venus-1.0的对比
1.0版本只有RL，没有Mid-Training；是分开的GD和Navi模型，没有unified agent。1.5版本的三个key advances都是针对1.0的limitation。

### 5.2 与concurrent works的关系
- **T-GRPO** (Chen et al., 2025): UI-Venus-1.5的Online-RL inspired by T-GRPO的trajectory-wise GRPO思想
- **DeepSeek-R1**: RL training paradigm的inspiration来源
- **Qwen3-VL**: base model backbone

### 5.3 Model Merging的理论背景
TIES-Merge基于task vector arithmetic的理论——fine-tuned model与base model的difference vector编码了task-specific knowledge。这个方向的相关工作还包括：
- [Model Soups](https://arxiv.org/abs/2203.05482)
- [Task Arithmetic](https://arxiv.org/abs/2212.04089)

### 5.4 Online-RL for GUI的挑战
GUI Agent的Online-RL相比传统RL有unique challenges：
1. **Sparse reward**: 只有task completion才给positive reward
2. **Long horizon**: 有些task需要20+ steps
3. **Environment stochasticity**: GUI state变化有随机性
4. **Expensive rollout**: real device interaction比simulator慢

UI-Venus-1.5的solutions：
1. Trajectory-level advantage normalization
2. Stratified sampling by difficulty
3. Group-relative baseline
4. DaaS infrastructure for scalable rollout

### 5.5 Refusal Capability的深远意义
传统的grounding model总是输出坐标，即使element不存在。这导致hallucination——agent会去点击不存在的元素。UI-Venus-1.5引入refusal capability `[-1, -1]`，这是一个从"always predict"到"know when to abstain"的paradigm shift。

这个思想与LLM领域的hallucination mitigation相关：
- [When Not to Retrieve](https://arxiv.org/abs/2402.10857)
- [Self-Consistency for Abstention](https://arxiv.org/abs/2305.17306)

---

## 6. Limitations与未来方向

基于paper内容推测的潜在limitations：

1. **WebVoyager performance gap**: 76.0% vs Holo2的83.0%和OpenAI-CUA的87.0%，web scenario的specialization不足
2. **Chinese app ecosystem focus**: 40+ Chinese apps的optimization可能limit cross-cultural generalization
3. **Model Merge的grounding trade-off**: TIES-Merge仍然导致~1.4%的grounding性能损失
4. **Online-RL的cost**: 需要上千台real device，computational和financial cost高
5. **VenusBench-Mobile的绝对值低**: 21.5%说明complex mobile tasks仍然是hard problem

未来可能的方向：
- **Test-time scaling**: 结合GTA1的test-time scaling思想
- **Multi-modal fusion**: 融合XML / accessibility tree与visual input
- **Self-improving loop**: 类似UI-Genie的self-improving approach
- **Cross-platform transfer**: Desktop / IoT等更多platform

---

## 7. 总结：Build Intuition

UI-Venus-1.5的核心intuition可以归纳为四个层次：

**Level 1 - Knowledge Layer**: Mid-Training通过10B tokens的GUI data建立foundation。这解决的是"model不知道GUI是什么"的问题。

**Level 2 - Skill Layer**: Offline-RL通过task-specific reward学习grounding / web / mobile三个domain的specialized skills。这解决的是"model不会做具体task"的问题。

**Level 3 - Adaptation Layer**: Online-RL通过real environment interaction学习dynamic adaptation和error recovery。这解决的是"model在real world会fail"的问题。

**Level 4 - Unification Layer**: Model Merge把三个specialist融合成一个end-to-end agent。这解决的是"deployment simplicity"的问题。

这四个层次构成了一个从knowledge → skill → adaptation → deployment的完整pipeline，每一层都address了前一层的limitation。这种layered design philosophy对于build practical GUI Agent非常有启发。

Reference:
- [UI-Venus GitHub](https://github.com/inclusionAI/UI-Venus)
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)
- [T-GRPO Paper](https://arxiv.org/abs/2506.08440)
- [TIES-Merging](https://arxiv.org/abs/2311.03003)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [AndroidWorld Benchmark](https://arxiv.org/abs/2405.14573)
- [ScreenSpot-Pro](https://arxiv.org/abs/2412.05685)
- [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [Mind2Web](https://arxiv.org/abs/2306.06001)
- [ShowUI](https://arxiv.org/abs/2411.17465)
- [AITW](https://arxiv.org/abs/2311.11708)
- [WebVoyager](https://arxiv.org/abs/2401.13919)
- [UI-TARS](https://arxiv.org/abs/2501.12326)
- [Mobile-Agent-v3](https://arxiv.org/abs/2508.15144)
- [MAI-UI](https://arxiv.org/abs/2512.22047)
- [Holo2](https://github.com/huggingface/open-r1)
- [OpenCUA](https://arxiv.org/abs/2508.09123)
- [GTA1](https://arxiv.org/abs/2507.05791)
- [Step-GUI](https://arxiv.org/abs/2512.15431)
- [GUI-Owl](https://arxiv.org/abs/2508.15144)
- [VenusBench-GD](https://arxiv.org/abs/2512.16501)
