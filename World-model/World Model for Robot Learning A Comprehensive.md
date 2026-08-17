---
source_pdf: World Model for Robot Learning A Comprehensive.pdf
paper_sha256: 1b035d79d5c135fd437e1bbe7625a7e284a405e0451a240e816ce5dcaf35f586
processed_at: '2026-08-13T05:03:57-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# World Model for Robot Learning — 用人话再聊一遍

好，刚才那个版本可能太"教科书"了，我换个方式，就像咱俩在咖啡馆里聊这篇paper一样，把核心东西掰开揉碎讲。

---

## 这篇paper到底在说啥

一句话总结：**机器人现在光会"看图说话"不够用，得会"脑补未来"才能做好control**。

你想想现在的VLA model（比如π₀、OpenVLA），给它一张图加一句instruction，它直接regress出action chunk。这就像一个人只会reactive反射，不会想"我抓起这个杯子之后会发生什么"。short-horizon task没问题，long-horizon、contact-rich、物理交互重的任务就崩了。

这篇survey就是把"world model"这个概念拎出来，systematically梳理一遍：**field里所有人怎么用predictive model帮robot policy变聪明的**。作者阵容太豪华了——Pieter Abbeel、Jitendra Malik、Yilun Du、Jiajun Wu、Zhuang Liu都在，基本是这个领域最top的一批人联合发声明说"world model is the next big thing for robotics"。

Links:
- https://ntumars.github.io/wm-robot-survey/
- https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy

---

## 最核心的intuition：Probabilistic Unification

paper里最让我"啊哈"的一点在Section 3.1。它用一个probabilistic view把四种seemingly不同的model统一了。

你先想象一个joint distribution：

$$p(o_{t+1:t+k}, a_{t+1:t+k} | o_t, l) \tag{4}$$

这个distribution说的是：given当前observation $o_t$ 和instruction $l$，未来k步的observation序列和action序列的联合分布。

然后paper告诉你，field里所有model都是这个joint distribution的不同query方式：

**Policy model**就是把这个joint distribution里的future observation marginalize掉：
$$p(a | o, l) = \int p(o_{future}, a_{future} | o, l) \, do_{future} \tag{5}$$

——只问"给我action就行，future observation我不管"。

**Passive world model**（普通video generation）就是marginalize掉action：
$$p(o_{future} | o, l) = \int p(o_{future}, a_{future} | o, l) \, da_{future} \tag{6}$$

——只问"未来video长啥样，不管中间什么action"。

**Controllable world model**就是condition on action：
$$p(o_{future} | o, a_{future}, l) \tag{7}$$

——问"如果我执行action A，世界会变成什么样"。这才是robot真正需要的。

**Inverse dynamics model**：
$$p(a | o_{current}, o_{future}, l) \tag{8}$$

——问"给定现在和未来，中间什么action能实现这个transition"。

**这个insight对你的intuition特别重要**：这四个model不是四个不同的architecture，它们是同一个idealized joint distribution的四种factorization。所以当你在选architecture的时候，本质上是在选"我要explicitly model哪个factorization"，而非在build不同的object。

这跟你做JEPA的philosophy其实underlying一致——都是在说"prediction是understanding的核心"，只不过这里把prediction和action generation耦合在一起了。

---

## Field的五大architecture paradigm

paper把"怎么把world model塞进policy"分成五大类，按coupling tightness递增。我用类比给你讲：

### Paradigm 1: Decoupled IDM (predict-then-act)

像两个独立工人。World model先generate未来video，另一个IDM worker看这个video反推action。

$$\hat{o}_{future} = \mathcal{W}(o_t, l) \tag{9}$$
$$\pi(a | o_t, l) = P(a | E_{img}(o_t), E_{text}(l), \Phi(\hat{o}_{future})) \tag{11}$$

变量解释：
- $\mathcal{W}$: world model (通常video diffusion)
- $\Phi(\cdot)$: 在predicted future上的feature extractor
- $E_{img}, E_{text}$: image和text encoder

代表工作evolution：UniPi (2023) → VidMan → Vidar → Gen2Act → VPP → Video2Act → MimicVideo → TC-IDM → Say-Dream-ACT

trend很清晰：**从raw pixel-space rollout逐渐shift到compact latent representation**。VPP和Video2Act干脆不从pixel generate video，直接从video diffusion的latent space抽predictive feature塞进action head。

**问题**: visual plausible但action-inconsistent的future会corrupt downstream control，error accumulation。

### Paradigm 2: Single-backbone (shared generative process)

像一个人同时想未来画面和未来action，用同一个brain。

$$\hat{y} = f_\theta(\tilde{x}_\tau, o_t, l, \tau), \quad x = [z^v; z^a] \tag{13}$$

变量解释：
- $x = [z^v; z^a]$: future visual representation和action representation拼一起
- $\tilde{x}_\tau$: 在denoising step $\tau$ 被corrupt的input
- $f_\theta$: shared backbone
- $y$: target (diffusion noise / velocity field / masked token)

代表：UVA, UWA, VideoVLA, VideoPolicy, Cosmos Policy, DreamZero, UD-VLA, GigaWorld-Policy

**核心motivation（这个对build你的intuition很重要）**: pretrained video diffusion backbone天然encode了motion continuity、temporal causality、approximate physical dynamics的prior，因为它training时就是learn temporally ordered prediction。而VLM backbone（image-text alignment pretrained）偏向semantic correspondence。当action generation嵌入同一个denoising process时，policy能inherit这个temporal propagation bias。

Cosmos Policy的设计特别聪明——保持pretrained video diffusion architecture完全不变，把robot action、future state、value都encode成额外的latent "frames"放进original diffusion sequence里。Inference时direct policy mode只取action output，planning mode用future-state和value prediction来rank trajectory。

### Paradigm 3: MoE/MoT (expert fusion + deep interaction)

像两个专家分别处理video和action，但通过shared attention反复交流。

$$(h_{\ell+1}^v, h_{\ell+1}^a) = \mathcal{F}_\ell^{mix}(h_\ell^v, h_\ell^a; o_t, l) \tag{15}$$

变量解释：
- $h_\ell^v, h_\ell^a$: layer $\ell$ 处video expert和action expert的hidden state
- $\mathcal{F}_\ell^{mix}$: layer-wise interaction operator

代表：Motus, LingBot-VA, BagelVLA, DiT4DiT, Fast-WAM, LDA-1B, FRAPPE

**Motivation**: video prediction和action generation有不同的temporal frequency、representational scale、optimization requirement，full parameter sharing不一定optimal。这跟你做MoE的intuition完全align——specialization + interaction > monolithic。

Fast-WAM有个重要empirical finding：主要benefit来自training时的video co-training，inference时explicit future imagination可能不需要。这暗示video branch可能主要是training-time regularizer，不一定非要inference-time rollout。

### Paradigm 4: Unified VLA (internalized future modeling)

没有显式video world model，但在VLA backbone内部learn future-oriented predictive structure。

三个subclass：
1. **Explicit future-state prediction**: GR-1, UP-VLA, WorldVLA — 直接predict future image
2. **Latent/implicit future modeling**: DreamVLA, UniVLA, CoWVLA — predict compact future-aware representation
3. **Multi-expert unified**: F1, InternVLA-A1, HALO, TriVLA — training统一但内部specialization

### Paradigm 5: Latent-space world modeling (no pixels)

完全在representation space做prediction，no image/video generation。这跟你JEPA philosophy高度align。

代表：FLARE, VLA-JEPA, JEPA-VLA, WoG, DIAL

VLA-JEPA的设计很elegant——leakage-free state prediction，future frame只用来produce latent target for supervision，防止model通过pixel variation shortcut。JEPA-VLA更直接——用V-JEPA 2学到的predictive embedding作为VLA backbone，argue这比static visual representation强。

---

## World Model as Simulator

除了当policy的predictive conditioning module，world model还能直接当environment用——你可以在imagination里train policy，不碰真实robot。

### RL in imagination

World model提供transition：
$$(\hat{o}_{t+1}, \hat{r}_t, \hat{d}_t) \sim p_\phi(\cdot | o_{\leq t}, a_{\leq t}, l) \tag{16}$$

变量：
- $\hat{o}_{t+1}$: predicted next observation  
- $\hat{r}_t$: predicted reward
- $\hat{d}_t$: predicted termination signal

然后GRPO-style优化：
$$\mathcal{L}_{RL}(\theta) = -\mathbb{E}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)] \tag{18}$$

其中 $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{old}}(a_t|s_t)$ 是importance ratio，$\hat{A}_t$是advantage，$\epsilon$是PPO经典clip range。

**重要empirical result**: World-Gymnast显示RL inside video world model能outperform supervised finetune AND software simulators。这是很强的evidence——learned simulator在某些场景下比hand-crafted simulator还强。

### Co-evolution (最前沿的idea)

但learned simulator本身imperfect，所以更前沿的方向是simulator和policy co-evolve：

$$\phi^{k+1} \gets \text{UpdateWM}(\phi^k, D_{real} \cup D_{policy}(\pi_{\theta^k}))$$
$$\theta^{k+1} \gets \text{UpdatePolicy}(\theta^k, \hat{D}(\phi^{k+1})) \tag{19}$$

变量：
- $\phi^k, \theta^k$: 第k轮的world model和policy参数
- $D_{real}$: real-world data
- $D_{policy}(\pi_{\theta^k})$: 当前policy rollout产生的data
- $\hat{D}(\phi^{k+1})$: improved world model生成的imagined data

代表：World-VLA-Loop, VLAW, WoVR。WoVR特别强调simulator reliability是central bottleneck，引入Keyframe-Initialized Rollouts来稳定long-horizon rollout。

### Evaluation in imagination

World model还能当evaluator——rollout candidate action看哪个future最好，或者rank不同policy checkpoint。WorldEval做了个有意思的事：完全在imagination里rank不同policy，看这个ranking是否能match真实世界的ranking。

---

## Robotic Video World Model的四个stage

paper把robotic video generation按capability成熟度分成四阶段：

**Stage 1: Imagination for supervision** — Dreamitate, RoboDreamer, DreMa, ManipDreamer, DreamGen。主要用途是generate future execution作为supervision signal或visual plan。

**Stage 2: Action-controllable** — IRASim, RoboEnvision, RoboMaster, Ctrl-World, EnerVerse-AC, Interactive World Simulator, EVA。核心shift是从"generate plausible future"到"future必须faithfully follow commanded action"。

**Stage 3: Structure-aware** — Mask2IV, TesserAct, RoboVIP。引入mask、geometry、viewpoint identity cue来preserve contact relation和scene structure。TesserAct做4D embodied world model over RGB+depth+normal。

**Stage 4: Foundation video world model** — Vid2World, Genie Envisioner, DreamDojo, WoW, UnifoLM-WMA-0, Cosmos Predict 2.5, GigaWorld-0, ABot-PhysWorld。Video generation变成reusable substrate for simulation, planning, evaluation, data production。

**Field bottleneck**: 不是generate realistic future，而是generate future that is **causally aligned with robot action, physically and kinematically self-consistent over long horizon, coherent across view and embodiment, stable under interaction, executable enough to support real policy improvement**。

---

## 实验数据告诉你什么

LIBERO 4-suite results（Table 5）最有意思的几个pattern：

| Paradigm | Method | Spatial | Object | Goal | Long | Avg |
|---------|--------|---------|--------|------|------|-----|
| Decoupled | Say-Dream-ACT | 99.4 | 99.2 | 98.6 | 95.4 | 98.1 |
| Single-backbone | Cosmos Policy | 98.1 | 100.0 | 98.2 | 97.6 | 98.5 |
| MoE/MoT | LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| Latent WM | VLA-JEPA | 96.2 | 99.6 | 97.2 | 95.8 | 97.2 |
| Latent WM | JEPA-VLA | 97.2 | 98.0 | 95.6 | 94.8 | 96.4 |

**三个key insight**:

1. **没有一种paradigm垄断** — decoupled (98.1), single-backbone (98.5), MoE (98.5), latent WM (97.2)都competitive。这strongly suggest——对embodied control，predictive representation quality比visual fidelity更重要。VLA-JEPA完全不generate pixel，avg 97.2，跟Cosmos Policy（full video diffusion）只差1.3 point。

2. **Long-horizon是真正的differentiator** — 几乎所有method在Spatial/Object都strong，Long suite是关键drop点。UD-VLA Long 89.6 vs Spatial 94.1（drop 4.5）。Long-horizon考验sustained, action-grounded consistency。

3. **Benchmark fragmentation** — RoboTwin/CALVIN/SIMPLER results显示，strong on one benchmark不transfer到another。Current embodied world model仍sensitive to embodiment/action space/task composition。

---

## 对你build intuition最重要的几个point

**1. World model ≠ video generation model**

Formal definition（公式1）是action-conditioned state-transition predictor。Video generation只是visual observation space的instantiation。Generic video generation不qualify as world model unless它capture environment evolution in form relevant to robot interaction。

**2. Photorealism不是necessary condition**

Table 5的VLA-JEPA（latent, no pixel）和Cosmos Policy（full video diffusion）performance gap只有1.3 point。这是你JEPA philosophy在robotics domain的empirical validation——prediction in representation space can be as effective as pixel-space prediction for control。

**3. Coupling tightness spectrum**

Decoupled IDM → Single-backbone → MoE/MoT → Unified VLA → Latent WM

这个spectrum的evolution反映一个deep trend：world modeling从auxiliary predictor逐渐integrate进core learning和decision-making loop。但paper明确说——**which predictive substrate is most effective仍open empirical question**。

**4. Causal conditioning是fundamental gap**

当前大部分WM的future conditioning更多bind to historical context和task intent，而不是specific pending robot action。这导致semantically plausible但action-inconsistent future。WorldVLA用implicit unified training来mitigate，但这仍是open problem。

**5. Simulator reliability > simulator realism**

WoVR的finding：hallucination和long-horizon error直接corrupt assessment signal。所以evaluation的realism + action-faithfulness + long-horizon reliability缺一不可。Co-evolution（公式19）是必须的——simulator本身需要continuously improve。

**6. Long-horizon是真正瓶颈**

几乎所有method在Spatial/Object都strong，Long suite是differentiator。Long-horizon要求sustained, action-grounded consistency——这恰好是reactive VLA最弱的地方，也是world model最该help的地方。

**7. Field方向**

Figure 2总结两条trend：
- Policy侧：decoupled video+IDM → single-backbone → MoE/MoT → unified VLA → latent WM
- Simulator侧：rollout validation → learned simulator for RL → co-evolving optimization

两条trend都指向同一结论：**world model从auxiliary predictor变成core learning/decision-making loop的integral part**。

---

## 几个你可能感兴趣的extension

**JEPA-style world model for robotics**: V-JEPA 2.1已经show latent action-conditioned world model支持zero-shot robot planning with image goal。LeWorldModel做end-to-end JEPA from pixel，show latent predictive model能detect physically implausible event。这跟你一直push的"prediction in representation space"完全align，第一次在robotics domain有concrete deployment evidence。
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- LeWorldModel: https://arxiv.org/abs/2603.19312

**Flow matching for action**: π₀用flow matching，ProphRL的FA-GRPO适配flow-based action head。Action-to-Action flow matching (Jia et al.)把base distribution从Gaussian换成action history——跟rectified flow思想有connection。
- π₀: https://arxiv.org/abs/2410.24164
- Action-to-Action flow matching: https://arxiv.org/abs/2602.07322

**MPC + neural dynamics**: TD-MPC2和LeWorldModel都show latent-space MPC能significantly enhance long-horizon reasoning。公式17的objective在latent space instantiate后planning complexity大幅降低。
- TD-MPC2: https://arxiv.org/abs/2310.16828

**MoE for robotics**: MoT-style architecture（Motus, LingBot-VA, BagelVLA）保留video和action expert specialization，通过shared attention深度交互。跟你sparse expert architecture有直接技术关联。
- Motus: https://arxiv.org/abs/2512.13030
- LingBot-VA: https://arxiv.org/abs/2602.09849 (BagelVLA)

---

## 最后一句人话

这篇survey本质上在说一件事：**robot要变聪明，不能只会"看到什么做什么"，得会"想象做完之后世界变成什么样，再决定做什么"**。而field正在从"world model是辅助工具"往"world model是policy核心组件"走。至于到底是用pixel-space video generation，还是latent-space prediction，还是symbolic abstraction——还没定论，但latent-space WM的empirical performance已经证明photorealism不是必要条件，这跟你JEPA philosophy在robotics domain得到了validation。

想深入聊哪一块（比如co-evolution algorithm的细节、或者latent-space WM的具体实现、或者evaluation protocol的设计），随时说。

---

# World Model for Robot Learning: A Comprehensive Survey — 深度技术解读

这篇paper是NTU MARS lab联合Berkeley/Stanford/Oxford/ETH/Princeton/Harvard等机构搞的一篇world model for robot learning的综合survey，作者阵容豪华（Pieter Abbeel, Jitendra Malik, Yilun Du, Jiajun Wu, Zhuang Liu, Marc Pollefeys等都在列）。下面我从你的视角——一个关心model scaling、representation learning、generative modeling的practitioner视角——把这篇paper的核心技术骨架、关键公式、architecture taxonomy、实验数据全部拆解给你，目标是build你的intuition about why world models matter for robotics, 以及the field is converging toward什么架构。

Reference links:
- Paper: https://ntumars.github.io/wm-robot-survey/
- GitHub repo: https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy
- arXiv (搜索): https://arxiv.org/abs/2507.XXXX (paper目前是survey版本，具体arXiv号以github为准)

---

## 1. The Core Thesis: 为什么robotics需要world model

paper开头直接点明了一个核心问题：**purely reactive VLA policies fail in long-horizon, contact-rich, physically grounded tasks**，原因在于缺少explicit predictive structure。这跟你一直在neural nets领域强调的"model needs to predict to understand"的intuition完全一致。

具体来说，目前的VLA（Vision-Language-Action）policy比如RT-2、OpenVLA、π₀、π₀.₅都是直接学一个mapping $p(a_{t+1:t+k} | o_t, l)$，这种reactive mapping有几个本质缺陷：

1. **Long-horizon credit assignment**：action chunk长度k有限（通常8-16），超过这个horizon就没有foresight
2. **Compounding error**：每一步的小误差在闭环执行中累积
3. **No counterfactual reasoning**：policy无法回答"如果我现在做action A而不是B，世界会变成什么样"

paper给出的核心论点是：**world model提供了predictive bridge from semantic intent to physically realizable behavior**。一个actionable world model需要三个核心能力：
- **Foresight**：执行前预测future states/consequences
- **Imagination-driven planning**：用imagined rollout来比较candidate behaviors
- **Data amplification**：synthesize额外demonstration来improve learning

---

## 2. Probabilistic Lens: The Unified Predictive-Control Distribution

这是paper最有intellectual depth的部分（Sec 3.1），paper用probabilistic view把policy、world model、inverse dynamics model统一成一个joint distribution的不同marginal/conditional。这是build intuition的关键。

设observation $o_t$，action $a_t$，instruction $l$。考虑joint conditional distribution over future observations AND future actions：

$$p(o_{t+1:t+k}, a_{t+1:t+k} | o_t, l) \tag{4}$$

paper指出，看似不同的paradigms其实都是这个joint distribution的不同query方式：

**Policy Model**（marginalize out future observation）：
$$p(a_{t+1:t+k} | o_t, l) = \int p(o_{t+1:t+k}, a_{t+1:t+k} | o_t, l) \, do_{t+1:t+k} \tag{5}$$

变量解释：$o_{t+1:t+k}$是未来k步observation序列（被marginalize掉），$a_{t+1:t+k}$是未来k步action序列，$o_t$是当前observation，$l$是language instruction。这是standard VLA直接学的目标。

**Passive World Model**（marginalize out action）：
$$p(o_{t+1:t+k} | o_t, l) = \int p(o_{t+1:t+k}, a_{t+1:t+k} | o_t, l) \, da_{t+1:t+k} \tag{6}$$

这是普通video generation model在学的东西——只关心"given当前frame和instruction，未来video长什么样"，没有action conditioning。这种model对robotics用处有限，因为它无法回答counterfactual。

**Controllable World Model**（condition on action）：
$$p(o_{t+1:t+k} | o_t, a_{t+1:t+k}, l) \tag{7}$$

这是robotics真正需要的——给定candidate action sequence，预测future observation。注意这里action变成conditioning variable，不是被marginalize掉的。这是"action-conditioned"的精确定义。

**Inverse Dynamics Model**：
$$p(a_{t+1:t+k} | o_{t:t+k}, l) \tag{8}$$

给定observation sequence（当前+未来），推断中间发生了什么action。这是UniPi那类decoupled pipeline的核心——先generate future video，再用IDM recover action。

**Key insight for your intuition**：这四个model不是独立的architecture，而是同一个idealized joint distribution $p(o_{future}, a_{future} | o_{current}, l)$ 的不同factorization。这解释了为什么world model和policy可以natural耦合：policy可以把future observation当作intermediate latent variable，IDM-style decoder再从这个predicted future recover executable action。

这个unified view实际上也是你在Jepa work里看到的"prediction in representation space"思想的generalization——区别在于这里是joint predictive-control distribution，而JEPA是predictive representation learning。

---

## 3. Architectural Taxonomy for World Model + Policy (Sec 3)

这是paper最dense的部分。paper把world-model-based policy methods分成**五大paradigm**，按coupling tightness递增排列。我给你拆解每个paradigm的architecture、优缺点和representative work。

### 3.1 IDM-style (Decoupled): Predict-then-Act

**Architecture**: 两个独立module。World model先生成future observation/latent，separate policy module再从(current obs, predicted future)推断action。

公式表达：
$$\hat{o}_{t+1:t+H} = \mathcal{W}(o_t, l) \tag{9}$$
$$\pi(a_{t+1:t+H'} | o_t, l) = P(a_t | E_{img}(o_t), E_{text}(l), \Phi(\hat{o}_{t+1:t+H})) \tag{11}$$

变量解释：
- $\mathcal{W}$：world model（通常是video diffusion）
- $H$：prediction horizon
- $E_{img}, E_{text}$：image和text encoder
- $\Phi(\cdot)$：predicted future上的feature extractor
- $H'$：action chunk size（可以与$H$不同）

**Representative work演化路径**（看你build intuition）：
- **UniPi** (Du et al., 2023) — 最早work，task-conditioned video generation + IDM从相邻frame比较recovery action representation
- **VidMan / Vidar** — 引入masked IDM强调action-relevant region
- **Gen2Act** — condition on generated human video而不是robot-centric rollout
- **VPP / Video2Act** — 关键shift：不再显式pixel-space rollout，而是从pretrained video diffusion的latent space提取predictive representation注入action head
- **MimicVideo** — 用partially denoised latent visual plan替代显式video prediction
- **TC-IDM / LVP** — 把future翻译成tool-centric geometric trajectory或retargetable visual plan
- **Say-Dream-ACT** — 把generated video plan作为in-context visual guidance而不是IDM target

**Latent form**（更紧凑的版本）：
$$\hat{z}_{t+1:t+H} = \mathcal{W}(E_{img}(o_t), E_{text}(l)) \tag{10}$$
$$\pi(a_{t+1:t+H'} | o_t, l) = P(a_t | E_{img}(o_t), E_{text}(l), \hat{z}_{t+1:t+H}) \tag{12}$$

**3D-aware extension**：AVDC, VidBot, Object-centric 3D Motion Field, NovaFlow——这些work从video里extract dense correspondence / 3D hand trajectory / 3D motion field / actionable 3D flow作为structured intermediate。

**Pros**: modular, reusable video prior, interpretable future prediction  
**Cons**: decoupling导致error accumulation，visually plausible但action-inconsistent的future会corrupt downstream control

### 3.2 Single-Backbone (Shared): Joint Generative Process

**Architecture**: 一个shared backbone同时generate future visual evolution AND future action。Observation token和action token在同一个denoising/generative process里处理。

**核心motivation**（这点对你的intuition很重要）：pretrained video diffusion backbone是trained on temporally ordered prediction objective，天然encode了motion continuity, temporal causality, approximate physical dynamics的prior。而VLM backbone（OpenVLA, π₀用的）是image-text alignment pretrained，偏向semantic correspondence。当action generation嵌入同一个denoising process时，policy能inherit这个temporal propagation bias。

公式表达：
$$\hat{y} = f_\theta(\tilde{x}_\tau, o_t, l, \tau), \quad x = [z^v; z^a] \tag{13}$$
$$\mathcal{L}_{unified} = \mathbb{E}[\ell(\hat{y}, y)] \tag{14}$$

变量解释：
- $x = [z^v; z^a]$：future visual representation和action representation的concatenation
- $\tilde{x}_\tau$：corrupted input at denoising step $\tau$
- $f_\theta$：shared backbone
- $y$：target（可以是diffusion noise / velocity field / masked token，取决于具体instantiation）

**Representative work演化**：
- **UVA** (Unified Video-Action) — joint video-action latent space，lightweight modality-specific decoding head让policy inference bypass explicit video generation
- **UWA** — 在diffusion process里直接integrate video和action，single transformer under modality-specific timesteps
- **VideoVLA** — 把Video Diffusion Transformer扩展成Video-Action Diffusion Transformer
- **VideoPolicy** — 把video generation当作primary policy substrate，action prediction变成layered on top的lightweight interface
- **Cosmos Policy** — 关键设计：保持pretrained video diffusion架构不变，把robot action / future state / value encode成额外的latent "frames"放在original diffusion sequence里。Inference时direct policy mode只取action output，planning mode用future-state和value prediction来rank trajectory
- **DreamZero** — autoregressive flow-matching video-action DiT，closed-loop chunk-wise joint denoising（限制compounding error）
- **UD-VLA** — discrete multimodal，future-image token和action token在single synchronous denoising trajectory里couple
- **GigaWorld-Policy** — causal design让visual branch在inference时optional

**Key distinction across these methods**：不是它们是否都render full future video online，而是visual branch在control时保留多少active computation。Some保留explicit future prediction for planning，others marginalize/truncate/discard visual branch for efficiency。

### 3.3 MoE/MoT-Style: Expert Fusion with Deep Interaction

**Architecture**: 保留separate expert streams for video prediction和action generation，通过shared attention / cross-attention / interleaved autoregressive sequence深度交互。

**核心motivation**：full parameter sharing不一定optimal——video prediction和action generation有不同的temporal frequency、representational scale、optimization requirement。这跟你熟悉的MoE思想一致：specialization + interaction > monolithic sharing。

公式表达：
$$(h_{\ell+1}^v, h_{\ell+1}^a) = \mathcal{F}_\ell^{mix}(h_\ell^v, h_\ell^a; o_t, l) \tag{15}$$

变量解释：
- $h_\ell^v, h_\ell^a$：layer $\ell$ 处video expert和action expert的hidden state
- $\mathcal{F}_\ell^{mix}$：layer-wise interaction operator（joint attention / cross-attention / shared-attention fusion）

**三种pattern**：

1. **Parallel expert coupling**: pretrained video diffusion backbone + lighter action branch  
   - GE-Act: parallel flow-matching action pathway + pretrained video diffusion，deep cross-attention注入visual latent feature

2. **Mixture-of-Transformers-based deep interaction**: 多个expert全程retained，repeatedly fused via shared attention  
   - Motus: explicit MoT with dedicated experts for understanding, video generation, action
   - LingBot-VA: causal world modeling，interleave video和action token进shared autoregressive sequence，dual-stream MoT with shared attention
   - BagelVLA: long-horizon manipulation，interleave linguistic planning + visual forecasting + action generation，Residual Flow Guidance用single-step denoising替代full video rollout
   - DiT4DiT: video branch的intermediate denoising feature guide action prediction
   - Fast-WAM: 关键empirical finding——主要benefit来自training时的video co-training，inference时explicit future imagination可能不需要

3. **Latent-space expertization**: shift world modeling到structured latent dynamics  
   - LDA-1B: visual forecasting在DINO latent space，shared self-attention inside multimodal diffusion transformer
   - FRAPPE: parallel expert streams with separate adapters，align到visual foundation model在latent space

**Intuition for you**：video branch从"必须faithfully rendered的output"变成"predictive latent process whose hidden states guide action generation"。这是非常关键的conceptual shift。

### 3.4 Unified VLA: Internalized Future Modeling

**Architecture**: 没有显式video world model，但在unified VLA backbone内部learn future-oriented predictive structure。

**Three subclass**：

1. **Explicit future-state prediction**: 直接预测future image
   - GR-1: GPT-style transformer里jointly predict action和future image
   - UP-VLA: future-image prediction改善action generation和visual generalization
   - WorldVLA: unified action和image understanding/generation，future-image prediction主要作为training signal而不是inference output

2. **Latent/implicit future modeling**: 预测compact future-aware representation
   - DreamVLA: 预测structured world knowledge（dynamic, spatial, semantic cue）
   - UniVLA: post-training over native multimodal tokenization吸收causal dynamics
   - CoWVLA: latent motion和compact future visual target替代redundant future frame reconstruction

3. **Multi-expert/multi-system unified**: training统一但architecture内部specialization
   - F1: future visual state as planning target in MoT architecture
   - InternVLA-A1: lightweight latent visual foresight + joint optimization
   - HALO: visual subgoal prediction + embodied reasoning
   - TriVLA: grounding + episodic dynamics perception + control as coordinated subsystems

**Key insight**：区别不在于是否有explicit standalone world model，而在于future-oriented predictive modeling是否internalized进同一个multimodal policy backbone。

### 3.5 Latent-Space World Modeling: Internalized Without Pixels

**Architecture**: 完全在representation space做future prediction，no explicit image/video generation。

**核心motivation**：避免explicit generative decoding的computational overhead和redundancy，同时retain predictive structure的inductive bias。这跟你JEPA work的philosophy高度align——prediction in embedding space rather than pixels。

**Representative work**：
- **FLARE**: "Future Latent Representation Alignment"——action denoising network的hidden feature与future observation的latent embedding对齐
- **VLA-JEPA**: leakage-free state prediction，future frame只用来produce latent target for supervision（防止model通过pixel variation shortcut）
- **JEPA-VLA**: 直接用V-JEPA 2学到的predictive embedding作为VLA backbone（argue这比static visual representation强）
- **WoG** (World Guidance): shift world modeling到condition space——预测compact future-oriented condition together with action
- **DIAL**: latent visual foresight in VLM feature space作为structured bottleneck

**Symbolic extension**（non-pixel abstraction）：
paper还提到symbolic/planner-facing world model——predict transition over predicates, object relations, affordances, causal processes，被symbolic planner query产生high-level skill sequence。Representative: VisualPredicate, ExoPredicate, From Pixels to Predicates (Athalye et al.)。

---

## 4. World Model as Simulator (Sec 4)

这里paper shift到另一个视角：world model不只是predictive conditioning module，而是直接作为interactive simulator stand in for environment。

### 4.1 For Reinforcement Learning: Imagined Policy Improvement

**Setup**: 在learned simulator里roll out trajectory，receive reward，improve policy。

World model $p_\phi$ 提供imagined transition：
$$(\hat{o}_{t+1}, \hat{r}_t, \hat{d}_t) \sim p_\phi(\cdot | o_{\leq t}, a_{\leq t}, l) \tag{16}$$

变量解释：
- $\hat{o}_{t+1}$：predicted next observation
- $\hat{r}_t$：predicted reward
- $\hat{d}_t$：predicted termination signal
- $p_\phi$：parameterized world model

Policy optimization objective（GRPO-style，因为VLA action head常用这个）：
$$J(\theta) = \mathbb{E}_{\hat{\tau} \sim (\pi_\theta, p_\phi)}\left[\sum_t \gamma^t \hat{r}_t\right] \tag{17}$$
$$\mathcal{L}_{RL}(\theta) = -\mathbb{E}_t\left[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)\right] \tag{18}$$
$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$$

变量解释：
- $\gamma$：discount factor
- $\hat{\tau}$：imagined trajectory
- $r_t(\theta)$：importance ratio
- $\hat{A}_t$：estimated advantage
- $\epsilon$：clip range（PPO经典超参）

**Representative work**:
- UniSim, World-Env, VLA-RFT — 基础recipe
- DiWA — frozen world model from play data支持fully offline diffusion policy adaptation
- World4RL — diffusion world model for higher-fidelity manipulation refinement
- World-Gymnast — RL inside video world model outperform supervised finetune AND software simulators（重要empirical result）
- PlayWorld — 从autonomous play学robot world model
- RehearseVLA — physically consistent world simulator + instant reflector for reward/termination
- WMPO — pixel-space imagination + on-policy GRPO
- ProphRL — FA-GRPO for flow-based action head
- RISE — compositional dynamics + progress value estimation
- GigaBrain-0.5M — scale world-model RL to pretrained VLA adaptation

**Second-level paradigm: Co-evolution**——recognize simulator本身imperfect，必须与policy co-improve：

$$\phi^{k+1} \gets \text{UpdateWM}(\phi^k, D_{real} \cup D_{policy}(\pi_{\theta^k}))$$
$$\theta^{k+1} \gets \text{UpdatePolicy}(\theta^k, \hat{D}(\phi^{k+1})) \tag{19}$$

变量解释：
- $\phi^k, \theta^k$：第k次iteration的world model和policy参数
- $D_{real}$：real-world data
- $D_{policy}(\pi_{\theta^k})$：当前policy rollout产生的data
- $\hat{D}(\phi^{k+1})$：improved world model生成的imagined data

Representative: World-VLA-Loop, VLAW, WoVR。WoVR特别强调simulator reliability是central bottleneck，引入controllable action-conditioned video modeling和Keyframe-Initialized Rollouts。

### 4.2 For Evaluation: Decision-Time Selection

**Setup**: 不retrain policy，而是用world model在执行前score/verify candidate action。

**Three form**：

1. **Rollout-based candidate assessment**: policy propose多个action，world model predict outcome，select best
   - GPC: augment frozen generative robot policy at deployment with action-conditioned world model，online ranking
   - IRASim: simulate multiple trajectory，select highest predicted value
   - World-in-World: closed-loop planning，rollout → evaluate by revision policy → revise
   - DreamPlan: 从world-model rollout构造preference pair作为training signal

2. **MPC-style active optimization**: 在world model的imagined trajectory里gradient-based optimize action sequence
   - TD-MPC2, LeWorldModel: latent-space MPC

3. **Policy evaluator**: world model作为scalable proxy for real-world policy evaluation
   - Evaluating Gemini Robotics Policies in Veo World Simulator: video world simulator for offline policy evaluation, OOD testing, safety probing
   - WorldEval: rank different policy AND different checkpoint of same policy entirely in imagination
   - WorldArena: policy evaluation as core downstream use

4. **Feedback-head augmented**: simulator有explicit reward/termination head
   - World-Env: continuous reward prediction + action termination prediction
   - RISE: progress value model scores imagined outcome by task advancement

**Critical insight from WoVR**: hallucination和long-horizon error不只reduce visual quality，而是直接corrupt assessment signal。所以evaluation要求realism + action-faithfulness + long-horizon reliability，三个缺一不可。

---

## 5. Robotic Video World Model (Sec 5): Four Capability Regimes

paper把robotic video generation按capability成熟度分成四个stage，这对你理解field evolution很有帮助。

### 5.1 Imagination for Policy Learning (Stage 1)

**Core idea**: 用strong generative prior合成future task execution，转化为supervision。

- Dreamitate: fine-tune video diffusion on task-specific human demo，test time用synthesized execution as visual plan
- RoboDreamer: compositional world modeling——instruction decompose成reusable primitive，generation condition on这些structured component
- ManipDreamer: action tree + depth + semantic visual guidance
- DreMa: Gaussian Splatting + physics simulator reconstruct explicit manipulable scene
- PhysWorld: 从generated video reconstruct物理world model，object-centric residual RL ground motion to action
- DreamGen: adapt strong video generator到target embodiment，synthesize neural trajectory，latent action modeling recover executable action

### 5.2 Action-Controllable (Stage 2)

**Core shift**: 从"generate plausible future"到"future follows commanded action with sufficient precision"。

- IRASim: trajectory-to-video formulation，frame-level action conditioning在每个transformer block里
- RoboEnvision: long-horizon multi-task，preserve semantic+temporal consistency
- RoboMaster: collaborative trajectory control，decompose manipulation成multiple phase，model coupled motion of arm和object
- Ctrl-World: joint multi-view prediction + frame-level action control + memory-based long-horizon generation，支持policy-in-the-loop rollout
- EnerVerse-AC: action-conditional multi-view generator，data engine AND evaluator
- Interactive World Simulator: high-frequency, long-horizon, stable policy-conditioned interaction for closed-loop rollout
- EVA: inverse-dynamics reward align video world model与smooth, embodiment-consistent action sequence

### 5.3 Structure-Aware (Stage 3)

**Core idea**: encode mask, geometry, viewpoint, identity cue来preserve contact relation和scene structure。

- Mask2IV: 两stage——先predict actor和object的interaction trajectory，再generate video condition on这些trajectory
- TesserAct: 4D embodied world model over RGB + depth + normal
- RoboVIP: visual identity prompting guide multi-view video diffusion，scalable augmentation

### 5.4 Foundation Video World Model (Stage 4)

**Core idea**: video generation不再是downstream augmentation tool，而是reusable substrate for simulation, planning, evaluation, large-scale data production。

- Vid2World: systematically transform pretrained video diffusion into interactive world model
- Genie Envisioner: unified world foundation platform integrating video world modeling + action decoding
- DreamDojo: pretrain on large-scale human egocentric video，continuous latent action bridge unlabeled human interaction和robot control
- WoW: physical intuition不能从passive video observation学到，必须train on extensive robot interaction trajectory
- UnifoLM-WMA-0, Cosmos Predict 2.5: reusable world backbone
- GigaWorld-0: controllable video branch + physically grounded 3D branch for large-scale embodied data synthesis
- ABot-PhysWorld: physics-aligned world foundation model

**Field的核心bottleneck**（这点对你的intuition很重要）：不是generate realistic future，而是generate future that is causally aligned with robot action, physically and kinematically self-consistent over long horizon, coherent across view and embodiment, stable under interaction, executable enough to support real policy improvement。

---

## 6. Benchmarks & Datasets (Sec 7)

### 6.1 Three-Layer Evaluation Framework

paper强调embodied world model evaluation和conventional video generation根本不同——visual realism alone neither necessary nor sufficient。分三层：

1. **Open-loop predictive quality**: given current obs + action sequence, autoregressively generate future obs, evaluate faithfulness
   - RBench: structural consistency, physical plausibility, action completeness across task/embodiment
   - EWMBench: factorized——scene consistency, motion correctness, semantic alignment
   - DreamGen Bench: instruction following + physics alignment
   - EVA-Bench: long-horizon anticipation + OOD robustness

2. **Closed-loop task utility and policy evaluation**: world model嵌入decision loop
   - WorldArena: 不只perceptual criteria，还有functional role（synthetic data generation, policy evaluation, action planning）
   - WorldEval: comparative policy assessment，rollout in learned world model preserve relative ordering of policy
   - WorldGym: Monte Carlo evaluation，estimated policy value是否match real world
   - World-in-World: world model直接进closed-loop planning pipeline

3. **Physical consistency, controllability, executability diagnostics**: probe specific failure mode
   - WorldSimBench: perceptual + manipulative evaluation，inverse-dynamics recovery test
   - WoW-World-Eval: physical-law + execution-oriented criteria，IDM-based Turing Test
   - WM-ABench: atomic capability decomposition——spatial/temporal understanding, motion perception, mechanistic simulation, counterfactual reasoning

### 6.2 Datasets

paper用multi-axis view而非disjoint category。重要dimension：
- Embodiment coverage
- Action supervision
- Observation/3D support (multi-view, depth, LiDAR, 3D annotation)
- Language conditioning
- Multimodal/contact-rich signal (force, tactile, audio)

Key datasets:
- Open X-Embodiment, DROID, BridgeData V2: 大规模general trajectory corpus
- AgiBot World, Galaxea Open-World, Humanoid Everyday, RoboMIND 2.0: modern real robot
- UMI / MV-UMI / ActiveUMI: human-to-robot prior via handheld interface
- TWIST2, DexWild, EgoMimic, PHSD, UniHand: human video + robot
- FreeTacMan, Humanoid Visual-Tactile-Action, VTDexManip, RH20T: contact-rich

**Critical limitation**：failure recovery, decision-sensitive variation, dense physically grounded supervision仍然scarce——绝大部分是successful demonstration。

### 6.3 Representative Results (LIBERO + RoboTwin + CALVIN + SIMPLER)

LIBERO 4-suite (Spatial/Object/Goal/Long)的representative result（Table 5）：

| Paradigm | Method | Spatial | Object | Goal | Long | Avg |
|---------|--------|---------|--------|------|------|-----|
| Decoupled | MimicVideo | 94.2 | 96.8 | 90.6 | 94.0 | 93.9 |
| Decoupled | Say-Dream-ACT | 99.4 | 99.2 | 98.6 | 95.4 | 98.1 |
| Single-backbone | Cosmos Policy | 98.1 | 100.0 | 98.2 | 97.6 | 98.5 |
| Single-backbone | UD-VLA | 94.1 | 95.7 | 91.2 | 89.6 | 92.7 |
| MoE/MoT | Motus | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| MoE/MoT | LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| Unified VLA | RynnVLA-002 | 99.0 | 99.8 | 96.4 | 94.4 | 97.4 |
| Unified VLA | UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| Unified VLA | CoWVLA | 97.2 | 97.8 | 94.6 | 92.8 | 95.6 |
| Unified VLA | F1 | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 |
| Latent WM | VLA-JEPA | 96.2 | 99.6 | 97.2 | 95.8 | 97.2 |
| Latent WM | JEPA-VLA | 97.2 | 98.0 | 95.6 | 94.8 | 96.4 |

**Three key observations from Table 5**:

1. **Strong result across multiple paradigm**——decoupled (Say-Dream-ACT 98.1), single-backbone (Cosmos Policy 98.5), MoE/MoT (LingBot-VA 98.5), unified VLA (RynnVLA-002 97.4), latent WM (VLA-JEPA 97.2)都达到competitive performance。这说明world modeling对embodied control的utility不tied to one specific implementation。**Photorealistic video generation is NOT necessary for effective embodied control**——latent-space world model也能很强。

2. **Long-horizon是关键differentiator**：很多method在Spatial和Object suite都strong，但Goal和Long suite drop明显。MimicVideo Long 94.0 vs Spatial 94.2（基本一致），但UD-VLA Long 89.6 vs Spatial 94.1（drop 4.5 point）。Long-horizon考验的是sustained, action-grounded consistency。

3. **Avg不能完全capture能力**：Cosmos Policy Avg 98.5但Long 97.6，Say-Dream-ACT Avg 98.1但Long 95.4——前者long-horizon更稳。

RoboTwin/CALVIN/SIMPLER result（Table 6）更进一步support这些point，但benchmark fragmentation严重——strong performance on one不一定transfer到another。这说明current embodied world model仍sensitive to embodiment/action space/task composition/evaluation protocol difference。

---

## 7. Open Challenges (Sec 8): What's Still Hard

### 7.1 Causal Conditioning Gaps

**问题**: 很多predictive world-model objective train主要从observation history + task intent，future可以plausible但不causally tied to即将执行的robot action。这limit了precise closed-loop control的usefulness。

**核心requirement**: 不只predict likely future，而是predict future如何under robot's intervention改变。这是causal conditioning的根本要求。

**Current mitigation**: WorldVLA的implicit unified training——couple future-state prediction with action generation，encourage more policy-aligned predictive dynamics。

### 7.2 Efficiency Bottlenecks

**Training/inference overhead**: world-model-based policy远比VLA计算intensive。要么joint predict future video+action，要么fine-tune before policy learning。

**Inference latency**: diffusion-based video prediction的iterative denoising导致high latency。

**Mitigation**:
- Parameter-efficient: lightweight adapter保持base model frozen
- Partial denoising: MimicVideo, LingBot-VA prioritizemotion dynamics over fine-grained visual detail
- Latent-space: LeWorldModel focus on predictive representation而非high-dim generation
- Train-time only: Fast-WAM——world modeling只在training用，inference skip掉

### 7.3 Multi-Modal Perception Bottlenecks

**问题**: current world model excel at visual synthesis但decouple from real-world interaction物理dynamics。Vision+proprioception无法capture friction, stiffness, contact stability这些unobservable property。

**Architecture challenge**: 异步signal with divergent frequency/dimension。Tactile sensor capture high-frequency transient event但low-dim signal在joint latent optimization里被high-dim visual feature dilute或overwhelm。

**Mitigation**: visuo-tactile model学习joint latent representation（Higuera et al. 2026, OmniVTA 2026）。

### 7.4 Classical Control Integration

**MPC bottleneck**: iterative world model rollout for action optimization的computational overhead限制real-time deployment。

**Frontier**: reconcile neural expressivity与formal control guarantee（Lyapunov stability, robust control）。Fuse learned dynamics with mature control principle。

### 7.5 Symbolic Structure Integration

**问题**: pixel-based rollout的long-horizon error accumulation degrade planning reliability。Symbolic representation abstract away low-level detail，model discrete/rule-based transition。

**Limitation**: require suitable abstraction + perception grounding。High-dim observation无法cleanly map到predefined symbol。

**Future**: hybrid world model combining learned perceptual representation + symbolic structure。

### 7.6 Evaluation Metrics

**问题**: 没有widely accepted evaluation metric。Visual plausible future可以fail to preserve action-conditioned dynamics。Limited visual realism不necessarily preclude utility for planning/evaluation。

**Future direction**: function-aware evaluation framework——joint assessment of predictive realism, action sensitivity, long-horizon consistency, control utility。Standardized metric: task success, policy-ranking fidelity, executability-oriented diagnostics。

---

## 8. My Intuition Summary for You

我给你提炼几个对build intuition最关键的point：

**1. World model ≠ Video generation model**

Paper的formal definition（公式1）是action-conditioned state-transition predictor，而video generation（公式2）只是visual observation space的instantiation。Generic video generation不qualify as world model unless it captures environment evolution in form relevant to robot interaction。

**2. Probabilistic unification是key insight**

Policy / passive WM / controllable WM / IDM都是 $p(o_{future}, a_{future} | o_{current}, l)$ 的不同factorization。这告诉你：architecture choice本质上是在select which factorization to model explicitly，而非在build different object。

**3. Coupling tightness spectrum**

Decoupled (IDM) → Single-backbone → MoE/MoT → Unified VLA → Latent-space WM

这个spectrum的evolution反映一个deep trend：world modeling从auxiliary predictor逐渐integrate进core learning和decision-making loop。但paper明确说——**which predictive substrate is most effective仍open empirical question**，video-pretrained backbone不necessarily superior to VLM/latent/structured/symbolic alternative。

**4. Photorealism不是necessary condition**

Table 5显示VLA-JEPA（latent-space, no pixel generation）avg 97.2，与Cosmos Policy（single-backbone video diffusion）avg 98.5差距很小。这strongly suggest——对embodied control，predictive representation quality比visual fidelity更重要。这是你JEPA philosophy在robotics domain的empirical validation。

**5. Long-horizon是真正瓶颈**

几乎所有method在Spatial/Object都strong，Long suite是differentiator。Long-horizon要求sustained, action-grounded consistency——这恰好是reactive VLA最弱的地方，也是world model最该help的地方。

**6. Simulator reliability > simulator realism**

WoVR的empirical finding：hallucination和long-horizon error直接corrupt assessment signal。所以evaluation的realism + action-faithfulness + long-horizon reliability缺一不可。这也解释了为什么co-evolution（公式19）是必须的——simulator本身需要continuously improve。

**7. Causal conditioning是fundamental gap**

当前大部分WM的future conditioning更多bind to historical context和task intent，而不是specific pending robot action。这导致semantically plausible但action-inconsistent future。这是closed-loop control的fundamental limitation。

**8. Field的方向**

Paper Figure 2总结两条trend：
- Policy侧：decoupled video+IDM → single-backbone → MoE/MoT → unified VLA → latent WM
- Simulator侧：rollout validation → learned simulator for RL → co-evolving optimization

两条trend都指向同一个结论：world model从auxiliary predictor变成core learning/decision-making loop的integral part。

---

## 9. 额外technical联想

基于paper内容，我想到几个你可能感兴趣的extension方向：

**关于JEPA-style world model for robotics**：V-JEPA 2.1 (Mur-Labadia et al. 2026)已经show latent action-conditioned world model支持zero-shot robot planning with image goal。LeWorldModel (Maes et al. 2026)做end-to-end JEPA from pixel，show latent predictive model能detect physically implausible event。这跟你一直push的"prediction in representation space"完全align，且第一次在robotics domain有concrete deployment evidence。

**关于flow matching for action**：π₀用flow matching，ProphRL的FA-GRPO适配flow-based action head。Action-to-Action flow matching (Jia et al. 2026b)是NTU MARS lab的工作，把flow matching的base distribution从Gaussian noise换成action history——这跟你熟悉的rectified flow思想有连接。

**关于MPC与neural dynamics融合**：TD-MPC2和LeWorldModel都show latent-space MPC能significantly enhance long-horizon reasoning。这与你的"model-based RL + neural network dynamics"研究方向高度relevant。公式17的objective在latent space instantiate后，planning complexity大幅降低。

**关于MoE for robotics**：MoT-style architecture（Motus, LingBot-VA, BagelVLA）保留video和action expert specialization，通过shared attention深度交互。这是MoE思想在robotics domain的具体应用，与你熟悉的sparse expert architecture有直接技术关联。

---

## Reference Links

- Survey paper: https://ntumars.github.io/wm-robot-survey/
- GitHub repo: https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- TD-MPC2: https://arxiv.org/abs/2310.16828
- π₀: https://arxiv.org/abs/2410.24164
- OpenVLA: https://arxiv.org/abs/2406.09246
- UniPi: https://arxiv.org/abs/2306.11335
- WorldVLA: https://arxiv.org/abs/2506.19850 (Cen et al. RynnVLA)
- Cosmos Policy: https://arxiv.org/abs/2601.16163 (Kim et al.)
- Motus: https://arxiv.org/abs/2512.13030 (Bi et al.)
- VLA-JEPA: https://arxiv.org/abs/2602.10098 (Sun et al.)
- WoVR: https://arxiv.org/abs/2602.13977 (Jiang et al.)
- DreamGen: https://arxiv.org/abs/2505.23504 (Jang et al.)
- WorldEval: https://arxiv.org/abs/2505.19017 (Li et al.)
- WorldArena: https://arxiv.org/abs/2602.08971 (Shang et al.)

如果你想深入某个paradigm（比如latent-space WM的具体实现细节，或者co-evolution algorithm的伪代码），告诉我，我可以再展开讲。
