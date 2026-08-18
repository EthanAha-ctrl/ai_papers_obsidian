---
source_pdf: FineVLA Fine-Grained Instruction Alignment.pdf
paper_sha256: 38d03031575fef6b4510865b2df90510d85bd4b5ede63c653763a0818a84d6b0
processed_at: '2026-08-18T12:57:59-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FineVLA 用人话讲

## 这篇paper到底在干嘛

很简单一句话——**robot learning现在有个很尴尬的gap**：action是dense的（每一帧都有continuous vector），但language supervision是sparse的（一整条trajectory只配一个task name比如"pick up the cup"）。

你想啊，一条trajectory几十秒、上百个action step，policy要从这么一个超级压缩的label去infer出几十个low-level decision。这signal-to-target ratio严重失衡。在language model里我们token-level supervision是dense的，每个token都有梯度。VLA里language是一整个trajectory共享一句task name，这gap太大了。

FineVLA的thesis就是：language supervision应该dense起来，要把execution manifold上每个control-relevant的factor都说清楚——哪只arm动、从哪接近、碰到哪里、怎么走、往哪转。这些factor在goal-level instruction里全都是unspecified的，policy只能靠implicit co-occurrence statistic瞎猜。

## 数据怎么搞的——FineVLA-Tool

### 核心problem：open dataset很脏很冗余

他们从10个open-source dataset聚合了972,247条trajectory（Table 6），来源包括DROID、BridgeData V2、RT-1、RoboMIND、Galaxea、RDT这些。但有两个problem：

1. **Format乱七八糟**：有的dataset action是joint space，有的是end-effector space；有的用absolute coordinate，有的用delta（相对当前），有的用relative to first frame；rotation有rotvec、quat、wxyz、euler四种encoding。根本没法直接拼。

2. **Trajectory高度冗余**：同一个"pick up cup"任务，DROID里可能有几百条demonstration，只在speed、minor spatial offset、camera viewpoint上不同，但underlying action pattern是一回事。全部annotate太贵了。

### Format统一

把所有trajectory转成LeRobot 2.1 format。Core canonicalization rule就两条公式（公式1、2）：

$$\hat{s}_{t+1}^{\text{joint}} = s_t^{\text{joint}} + \Delta a_t^{\text{joint}}$$

变量含义：
- $s_t^{\text{joint}} \in \mathbb{R}^d$：当前时刻的joint state vector，$d$是关节维度
- $\Delta a_t^{\text{joint}}$：raw delta action command（很多dataset存的是delta）
- $\hat{s}_{t+1}^{\text{joint}}$：转换后的absolute next-state target

第二公式：

$$a_{t+1,\text{rel}}^{\text{joint}} = \hat{s}_{t+1}^{\text{joint}} - s_1^{\text{joint}}$$

- $s_1^{\text{joint}}$：first frame的joint state
- $a_{t+1,\text{rel}}^{\text{joint}}$：relative-to-first-frame action representation

这样所有dataset都normalize成同一种temporal reference。EEF的处理还要先把rotation统一到quaternion xyzw format，再compose delta pose。

### DTW clustering去冗余

这是最有engineering含量的部分。用Dynamic Time Warping算两条trajectory之间的similarity，再做hierarchical clustering。DTW的recursion（公式3）：

$$D_{\text{DTW}}(i,j) = c(\mathbf{x}_i, \mathbf{y}_j) + \min\{D_{\text{DTW}}(i-1,j-1), D_{\text{DTW}}(i-1,j), D_{\text{DTW}}(i,j-1)\}$$

变量含义：
- $\mathbf{x}_{1:T}$, $\mathbf{y}_{1:U}$：两条action sequence，长度分别是$T$和$U$
- $D_{\text{DTW}}(i,j)$：前$i$个x frame和前$j$个y frame的optimal cumulative cost
- $c(\mathbf{x}_i, \mathbf{y}_j)$：frame-level distance
- 三个min项分别对应：both advance / x advance only / y advance only——这就是允许temporal warping的关键

EEF的frame cost（公式5）：

$$c_{\text{eef}}(\mathbf{x}, \mathbf{y}) = w_{\text{pos}} \cdot \|\mathbf{p}_x - \mathbf{p}_y\|_2 + w_{\text{rot}} \cdot d_{\text{geo}}(\mathbf{q}_x, \mathbf{q}_y) + w_{\text{grip}} \cdot |g_x - g_y|$$

变量含义：
- $\mathbf{p}$：3D position
- $\mathbf{q}$：quaternion
- $g$：gripper state（binary open/close）
- $d_{\text{geo}}(\mathbf{q}_x, \mathbf{q}_y) = 2\arccos(|\mathbf{q}_x \cdot \mathbf{q}_y|)$：quaternion geodesic distance，取绝对值是因为quaternion有double cover（$\mathbf{q}$和$-\mathbf{q}$是同一个rotation）

**Weight选择的intuition很关键**：$w_{\text{pos}} = 1.0, w_{\text{rot}} = 1.0, w_{\text{grip}} = 100.0$。gripper weight比其他大100倍。原因是gripper open/close transition是个discrete event，是区分manipulation strategy的key signal——抓取前必open，抓取后必close，这个binary state翻转点定义step boundary。如果weight太小，continuous position/rotation的小差异会drown掉这个critical binary signal，clustering就分不出execution mode了。

结果：972,247条raw trajectory → 47,159条representative trajectory（约20.6x压缩）。Average word count从9.3（原始coarse instruction）涨到96.8（fine-grained annotation），10.4x density increase。最夸张的Galaxea从4.7涨到219.9（47.1x）。

## Annotation schema的10个dimension

这是paper的core design contribution（Table 9）：

| Dimension | 说什么 |
|---|---|
| Action sequence | 顺序的primitive action + gripper state |
| Active actor | 哪只arm/gripper/body part执行 |
| Target object | Object category + disambiguating attribute |
| Initial config | Pre-action的pose/state |
| Final config | Post-action的pose/state |
| Contact & approach | 接触哪 + 从哪接近 |
| Trajectory & orientation | 平移路径 + 旋转行为 |
| Object interaction | 对其他object的secondary effect |
| Failure & recovery | Retry, slippage, fail-then-succeed |
| Body motion | Base/torso/camera移动 |

**为什么这10个dimension重要**：它们覆盖了execution manifold上所有control-relevant axis。每个dimension都是goal-level instruction omit掉的、且能从video visual确认的。paper很强调"describe only visible facts, do not infer hidden actions"——这点防止hallucination。

Annotation pipeline是model-assisted + human-verified：Qwen3.5-Plus先生成structured step decomposition，human annotator再verify 7个dimension（temporal order, object identity, actor identity, contact region, motion direction, state transition, hallucinated events，见Table 10）。

## Benchmark设计——RoboFine-Bench

500个video（每source dataset 50个，uniform分配防dominance），32种embodiment，11,631个atomic fact，1,030个VQA question。严格disjoint from training set。

**两个track互补**：
- **VQA track**：测discriminative understanding，10个dimension聚合成3个reporting axis（Gnd. / Act. / State，见Table 16）
- **Caption track**：测generative understanding，分easy（给task instruction）和hard（只给video，要infer整个process）两种setting

### Caption track的metric设计很精巧

公式1-4。给定generated caption和GT atomic fact set，做alignment得到label count：
- $M$ = match数
- $P$ = partial match数
- $C$ = contradiction数
- $O$ = omission数
- $A = M + P + C$（caption实际address的GT fact总数）
- $G = M + P + C + O$（总GT fact数）
- $H$ = hallucinated action event数
- $S$ = caption总step数

$$\text{Consistency} = \frac{M + 0.5P}{A} \quad (\text{precision-like})$$

$$\text{Coverage} = \frac{M + 0.5P}{G} \quad (\text{recall-like})$$

$$\text{Anti-Hallucination} = 1 - \frac{H}{S}$$

$$\text{Overall} = \frac{\text{Consistency} + \text{Coverage} + \text{Anti-Hallucination}}{3}$$

**Partial match给0.5 weight的intuition**：caption说"approach from above"但GT是"approach from top"，这算partial，既不过分punish coarseness也不完全forgive imprecision。Hard setting是真正能测出model有没有capture execution process的——它不能靠task-level language prior偷懒。

## RoboFine-VLM的result

在Qwen3.5-397B-A17B上full-parameter SFT，256×H200 GPU，903 step，~40小时，~105 GB/GPU memory。

**VQA**（Table 2）：
- RoboFine-VLM Overall: 68.2%
- GPT-5.4（最强baseline）: 60.2%
- 最大gain在Act. axis：75.7% vs 64.6%（+11.1）
- vs base Qwen3.5-Plus：55.9% → 68.2%（+12.3）

**Caption**（Table 3）：
- Easy Overall: 83.2% vs GPT-5.4的81.4%
- **Hard Overall: 82.2% vs GPT-5.4的78.0%**（+4.2）

Hard setting的result尤其重要——它证明RoboFine-VLM真正capture了execution process，不是依赖task-level language prior。

**Human alignment**: 10个rater的ranking和benchmark score相关性极高（Easy: Pearson 0.937, Spearman 0.943；Hard: 0.922 / 0.943）。这验证metric validity。换Gemini-3.1-Pro做judge也保持RoboFine-VLM最强（Table 18），证明结论对judge choice robust。

## Policy experiment——核心ablation

### Setup

两个framework，共享Qwen3.5-4B backbone：
- **StarVLA-OFT**: MLP regression head读action token hidden state，L1 objective回归continuous action chunk
- **StarVLA-GR00T**: dual-system design，VL backbone做System 2，DiT-based flow-matching module做System 1

三个(dataset, framework)组合：RDT-OFT, RDT-GR00T, AlohaMix-OFT。AlohaMix约13x大于RDT in episode count（86,662 vs 6,061），限制在ALOHA-compatible dual-arm避免cross-embodiment confound。

### 最clean的design choice

**这是paper最聪明的地方**：FG dataset和Raw dataset共享identical trajectory、action label、visual observation，唯一变量是paired language instruction。FG:Raw ratio控制每个training step sample每个dataset的概率。

这isolates了language supervision effect from data scale, embodiment, action distribution所有confound。七个ratio：Raw-only, 1:4, 1:2, 1:1, 2:1, 4:1, FG-only。

### Inverted-U现象——最核心的finding

AlohaMix-OFT的RoboTwin result（Table 4）：

| FG:Raw | Easy | Hard |
|---|---|---|
| Raw-only | 71.8 | 71.4 |
| 1:4 | 75.3 | 74.3 |
| 1:2 | 82.8 | 78.6 |
| **1:1** | **86.8** | **82.5** |
| 2:1 | 80.9 | 79.3 |
| 4:1 | 79.5 | 78.5 |
| FG-only | 78.3 | 76.1 |

Peak在FG:Raw = 1:1，gain over Raw-only是+15.0/+11.1。三个(dataset, framework)组合都show一致inverted-U trend。

**Inverted-U的deep intuition**——

**Raw instruction做的事**：preserve compact goal semantics——what task to complete。它是个高度压缩的abstraction。

**FG instruction做的事**：expose execution constraint——how task to perform。它是step-level dense decomposition。

- **Raw-only下**：execution-level choice（arm, approach, rotation）靠implicit co-occurrence statistic，policy猜得太弱
- **FG-only下**：policy有explicit process-level guidance，但lose goal-level abstraction，weaken generalization to unseen instruction phrasing；且FG description更长、distributionally更different from natural user command
- **Mixed下**：policy同时access task semantic（来自raw）和execution constraint（来自FG），两信号complementary

这就像signal processing里的two-frequency band——一个high-level低频band给envelope，一个low-level高频band给detail。两band都给，signal才完整。只给一个就distortion。

### Architecture gap narrows——supervision比architecture更fundamental

Table 23 Panel B，RDT上OFT vs GR00T的gap：

| Supervision regime | OFT - GR00T gap (Easy/Hard) |
|---|---|
| Raw-only | 6.4 / 6.6 |
| Best mixed | 4.7 / 4.2 |
| FG-only | 0.8 / 0.5 |

**Intuition**：Raw-only下language supervision不够informative，policy必须靠action decoder的capacity弥补。OFT的MLP head直接efficient，GR00T的flow-matching DiT绕一圈，所以OFT更强，gap大。

FG-only下，supervision本身已经给了足够execution constraint，architecture choice变得less critical，gap collapse到0.8/0.5。

这个现象在language model里也有类比——induction head的capacity依赖。Strong signal能compensate弱architecture。这暗示**fine-grained supervision是比architecture更fundamental的scaling axis**。

### Data scale effect——FG在大scale上更值

FG-only improvement over Raw-only：
- RDT（小）: +1.4/+2.0
- AlohaMix（大13x）: +6.5/+4.7

**Intuition**：trajectory diversity越大，dense action-aligned language有更多distinct execution pattern可以bind到。小dataset上pattern不够多，FG的marginal value有限；大dataset上每个distinct pattern都能从FG supervision受益。这说明fine-grained supervision是**scalable axis**——未来更大scale的VLA training收益会更大，不是incidental improvement。

## Real-world steerability——factor-level analysis

### Suite设计

Cobot Magic dual-arm，14 joint + 2 gripper continuous command，3 RGB camera（2 wrist + 1 third-person），action chunk length 50，30Hz controller。12 tabletop task，600 demonstration。

**Paired variant设计**是最精妙的部分：每个factor下，两个variant共享**same visual scene**，只改**一个**language-specified control factor：
- Color: "put red pen" vs "put blue pen"
- Pose: "lying cup" vs "standing cup"
- Approach: "from above" vs "from right side"
- Rotate: "clockwise" vs "counter-clockwise"
- Arm: "right hand to right bowl" vs "left hand to left bowl"
- **OOD probe**: "right hand to right bowl" vs "left hand to **right** bowl"——unseen actor-target binding

这设计让我们能在固定visual context下纯粹测instruction following能力。

### Per-factor result

Table 5，FG:Raw = 1:1 vs Raw-only的gain：

| Factor | Raw-only | 1:1 | Gain |
|---|---|---|---|
| Pose | 24 | 47 | **+23** |
| Color | 22 | 40 | **+18** |
| Approach | 60 | 78 | **+18** |
| Rotate | 76 | 86 | +10 |
| Arm | 60 | 64 | +4 |
| Clean Table | 72 | 84 | +12 |
| Stack Block | 35 | 40 | +5 |
| OOD Arm | 0 | 10 | +10 |

**Gain magnitude排序的intuition**——

排序：Pose (+23) ≈ Color (+18) ≈ Approach (+18) > Rotate (+10) > Arm (+4)

这精准对应了instruction underspecification程度：
- Pose, Color, Approach在goal-level language里**完全unspecified**——gain最大
- Rotate在task context里偶尔implied（"rotate"这个词有时暗示方向）
- Arm selection在task context里更常implied（dual-arm task自然有arm choice）

**这从另一个angle验证了paper核心论点**：FG supervision补的恰好是raw instruction没说清楚的部分。Gain magnitude跟underspecification degree正相关，这是paper最强的empirical evidence之一。

### OOD compositional limitation

OOD probe（L→R，unseen actor-target binding）：mixed model从0涨到10/100。

Table 26的language-critical subgoal success rate（只看是否选对factor，不看最终completion）：
- Raw-only: 3/10
- 1:1: 6/10
- 2:1, 4:1: 4/10

**Intuition**：增加FG supervision确实improve active-arm grounding——model学会了听instruction选arm。但unseen actor-target binding仍未解决——model能选对arm，但不能正确把arm binding到指定的target receptacle。

这是**factor-level grounding和compositional generalization之间的gap**——前者是table内interpolation，后者是table外extrapolation。FineVLA解决了前者，后者仍是open problem。这跟你做micrograd时强调的"interpolation easy, extrapolation hard"的divide一样。

## Failure mode分析

Paper诚实承认两类failure：

1. **Grounding failure**：policy选错object/arm/target，即使language specify了correct factor。这是perception + language grounding问题。
2. **Execution failure**：选对了factor但physical manipulation失败（unstable grasp, incomplete rotation, inaccurate placement）。这是low-level motor control问题。

这两类failure是orthogonal的——FineVLA主要address前者，但execution failure仍需separate解决，可能需要force feedback, tactile sensing等。

## 一些critical的concern

1. **Real-world evaluation scale有限**：只有12 tabletop task，600 demonstration。相比真real-world deployment仍太小。
2. **RoboFine-VLM未用于policy supervision**：paper明确说policy experiment用human-verified label，RoboFine-VLM只是future scalable annotator。这说明RoboFine-VLM的annotation质量仍不够直接用，human-in-the-loop不能完全去除。
3. **Caption track的LLM judge dependency**：用GPT-5.4-Pro做alignment judge，虽然swap到Gemini-3.1-Pro结论robust（Table 18），但仍是closed-source dependency。
4. **OOD compositional generalization未解决**：L→R从0到10远未实用。
5. **Cross-embodiment未验证**：AlohaMix限制在ALOHA-compatible，cross-embodiment effect未测。

## 我整体读下来的intuition

FineVLA本质上是把**language supervision从trajectory-level sparse变成step-level dense, action-aligned**。这解决的就不是policy capacity问题，是**signal density问题**——VLA model的capacity够，但language supervision不够informative，policy学不到execution manifold的结构。

Inverted-U现象是paper最深刻的insight：fine-grained supervision是signal **augmentation**，跟raw instruction是complementary不是replacement。Raw instruction提供task semantic abstraction（高维compression），FG instruction提供execution constraint（低维decomposition）。两者在information bottleneck的两端，mixed supervision让policy同时access两端。

Architecture gap narrows的发现进一步支持这点——当supervision signal足够dense，policy的architecture capacity不再bottleneck，这意味着fine-grained supervision是比architecture更fundamental的scaling axis。

Real-world的per-factor gain排序（pose/color/approach > rotate > arm）精准对应了instruction underspecification程度，这从另一个angle验证了"FG supervision补的是raw instruction没说清楚的部分"这个核心论点。

未来最重要的open question是**compositional generalization**——单factor grounding可以学，但unseen factor combination binding仍未解决。这可能需要从instruction-level supervision走向structural compositional representation，类似language model里compositional generalization的研究方向。

## Reference Links

- FineVLA Project Page: https://finevla.xlang.ai/
- LeRobot (HuggingFace): https://github.com/huggingface/lerobot
- DTW (Wikipedia): https://en.wikipedia.org/wiki/Dynamic_time_warping
- RoboTwin: https://github.com/TianxingChen/RoboTwin
- OpenVLA-OFT: https://github.com/moojink/openvla-oft
- NVIDIA Isaac GR00T: https://github.com/NVIDIA/Isaac-GR00T
- Qwen3-VL: https://qwenlm.github.io/blog/qwen3-vl/
- BridgeData V2: https://arxiv.org/abs/2308.12952
- DROID: https://droid-dataset.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- StarVLA: https://arxiv.org/abs/2604.05014
- π (Physical Intelligence): https://arxiv.org/abs/2604.15483
- RDT-1B: https://arxiv.org/abs/2410.07864
- RoboMIND: https://arxiv.org/abs/2510.17801
- Galaxea: https://arxiv.org/abs/2509.00576
- RoboCOIN: https://arxiv.org/abs/2511.17441
- RH20T: https://arxiv.org/abs/2307.00595

---

# FineVLA: Fine-Grained Instruction Alignment 深度解析

## 1. 核心问题与Motivation的Intuition

VLA model当前的language supervision存在一个根本性的**density mismatch**问题。Action space是dense的（每个timestep都有continuous action vector），但language supervision是sparse的——一条trajectory几十秒、上百个action step，只配一个task name。这相当于让model从一个极度under-determined的signal去推一个over-determined的output。Andrej你应该熟悉这种signal-to-target ratio不平衡的问题，在language model里token-level supervision是dense的，但VLA里language是trajectory-level的。

FineVLA的核心论点可以formalize成：language supervision应该**span整个execution manifold的factor**，包括actor choice（哪只arm）、approach direction（从哪接近）、contact region（碰到哪里）、motion path（怎么走）、rotation direction（往哪转）等。这些factor在goal-level instruction里完全unspecified，policy只能靠implicit co-occurrence statistics猜测。

## 2. FineVLA-Tool Pipeline的Engineering Detail

### 2.1 Heterogeneous Data Unification

从10个source dataset聚合972,247 trajectory（Table 6），转换到LeRobot 2.1 format。这里的关键challenge是action/state representation异构：

- **Temporal reference axis**: absolute vs delta vs rel（relative to first frame）
- **Kinematic axis**: joint space vs EEF space，且rotation有rotvec/quat/wxyz/euler四种encoding

Canonicalization的formal rule（公式1和2）：

$$\hat{s}_{t+1}^{\text{joint}} = s_t^{\text{joint}} + \Delta a_t^{\text{joint}}$$

这里$s_t^{\text{joint}} \in \mathbb{R}^d$是当前joint state（$d$是关节维度），$\Delta a_t^{\text{joint}}$是raw delta action command。$\hat{s}_{t+1}^{\text{joint}}$是预测的next absolute state target。

$$a_{t+1,\text{rel}}^{\text{joint}} = \hat{s}_{t+1}^{\text{joint}} - s_1^{\text{joint}}$$

$s_1^{\text{joint}}$是first frame的joint state，$a_{t+1,\text{rel}}^{\text{joint}}$是relative-to-first-frame action。这统一了不同dataset的temporal reference。

对于EEF，需要先统一rotation representation到quaternion xyzw format，处理delta pose需要compose with当前absolute pose。

### 2.2 DTW-Based Clustering的Intuition

Open robot dataset高度冗余。比如DROID里同一个"pick up cup"任务可能有几百条demonstration，只在speed、minor spatial offset、camera viewpoint上不同，但underlying action pattern相同。

DTW的recursion（公式3）：

$$D_{\text{DTW}}(i,j) = c(\mathbf{x}_i, \mathbf{y}_j) + \min\{D_{\text{DTW}}(i-1,j-1), D_{\text{DTW}}(i-1,j), D_{\text{DTW}}(i,j-1)\}$$

- $\mathbf{x}_{1:T}$和$\mathbf{y}_{1:U}$是两条action sequence，长度分别是$T$和$U$
- $D_{\text{DTW}}(i,j)$是前$i$个frame和前$j$个frame的optimal cumulative cost
- $c(\mathbf{x}_i, \mathbf{y}_j)$是frame-level cost
- 三个min项分别对应：match对齐（i-1,j-1）、advance on x only、advance on y only——这允许temporal warping

Frame cost function按action space区分。Joint-space（公式4）：

$$c_{\text{joint}}(\mathbf{x}, \mathbf{y}) = w_{\text{pos}} \cdot \|\mathbf{j}_x - \mathbf{j}_y\|_2 + w_{\text{grip}} \cdot |g_x - g_y|$$

$\mathbf{j}$是min-max normalized joint vector，$g$是gripper state。

EEF-space（公式5）：

$$c_{\text{eef}}(\mathbf{x}, \mathbf{y}) = w_{\text{pos}} \cdot \|\mathbf{p}_x - \mathbf{p}_y\|_2 + w_{\text{rot}} \cdot d_{\text{geo}}(\mathbf{q}_x, \mathbf{q}_y) + w_{\text{grip}} \cdot |g_x - g_y|$$

$\mathbf{p}$是3D position，$\mathbf{q}$是quaternion。这里**quaternion geodesic distance**很关键：

$$d_{\text{geo}}(\mathbf{q}_x, \mathbf{q}_y) = 2 \arccos(|\mathbf{q}_x \cdot \mathbf{q}_y|)$$

quaternion有**double cover**性质：$\mathbf{q}$和$-\mathbf{q}$表示同一个rotation。取$|\cdot|$处理这个ambiguity。$2\arccos(\cdot)$给出angular distance in $[0, \pi]$。

**Intuition on weight choice**: $w_{\text{pos}} = 1.0, w_{\text{rot}} = 1.0, w_{\text{grip}} = 100.0$。gripper weight远大于其他，因为gripper open/close transition是**discrete event**，是区分manipulation strategy的关键signal。比如抓取前必open，抓取后必close，这个binary state翻转点决定step boundary。如果weight太小，continuous position/rotation difference会drown掉这个critical signal。

Hierarchical clustering用average linkage，自动通过largest relative gap in merge heights决定cluster数。每个cluster选2-3个representative trajectory，从972,247降到47,159（约20.6x压缩）。

### 2.3 Fine-Grained Annotation Schema的10个Dimension

这个schema是paper的核心设计贡献（Table 9）：

| Dimension | Captures |
|---|---|
| Action sequence | Ordered primitive + gripper state |
| Active actor | 哪只arm/gripper/body part执行 |
| Target object | Object category + disambiguating attribute |
| Initial config | Pre-action pose/state |
| Final config | Post-action pose/state |
| Contact & approach | Contact region + approach direction |
| Trajectory & orientation | Translation path + rotation behavior |
| Object interaction | Secondary effect on other object |
| Failure & recovery | Retry, slippage, fail-then-succeed |
| Body motion | Base/torso/camera movement |

**Intuition**: 这10个dimension覆盖了execution manifold的所有control-relevant axis。它们都是goal-level instruction omit掉的，且能从video中visual确认（这点很重要——paper强调"describe only visible facts, do not infer hidden actions"）。

Annotation pipeline是model-assisted + human-verified：Qwen3.5-Plus先生成structured step decomposition，human annotator再verify。结果average word count从9.3涨到96.8（10.4x density increase），最极端的Galaxea从4.7涨到219.9（47.1x）。

## 3. RoboFine-Bench的Evaluation Design

### 3.1 Benchmark Statistics

- 500 video（每source dataset 50个，uniform allocation防dominance）
- 32 embodiment
- 11,631 atomic fact，avg 4.3 step + 23.3 fact per sample
- 1,030 VQA question
- 严格disjoint from training set

### 3.2 VQA Track的Dimension Aggregation

10个fine-grained dimension聚合成3个reporting axis（Table 16）：
- **Gnd.** (Entity & Scene Grounding): Active actor + Target object + Initial config
- **Act.** (Action & Motion Understanding): Action sequence + Contact & approach + Trajectory & orientation + Body motion
- **State** (Interaction & State Reasoning): Object interaction + Final config + Failure & recovery

Question type有multiple choice（4-8 option）/yes-no/number三种，answer scoring deterministic（option matching / string comparison / value extraction）。

### 3.3 Caption Track的Metric设计

这组metric设计得很有意思（公式1-4）。给定一个generated caption，与GT atomic fact set做alignment，得到label count：
- $M$ = match count
- $P$ = partial match count
- $C$ = contradiction count
- $O$ = omission count

定义$A = M + P + C$（caption实际address的GT fact总数），$G = M + P + C + O$（总GT fact数）。$H$是hallucinated action event数，$S$是caption总step数。

$$\text{Consistency} = \frac{M + 0.5P}{A}$$

衡量caption说的东西里有多少是对的（precision-like）。

$$\text{Coverage} = \frac{M + 0.5P}{G}$$

衡量GT fact有多少被caption cover到（recall-like）。

$$\text{Anti-Hallucination} = 1 - \frac{H}{S}$$

衡量caption step里有多少是fabricated（penalize hallucination）。

$$\text{Overall} = \frac{\text{Consistency} + \text{Coverage} + \text{Anti-Hallucination}}{3}$$

**Intuition**: 这个设计巧妙在partial match给0.5 weight，既不过分punish coarseness，也不完全forgive imprecision。Easy vs Hard setting区分了"given task instruction"和"vision-only"两种condition，hard setting能真正测出model是否capture了execution process而不依赖task-level language prior。

## 4. RoboFine-VLM的Result

RoboFine-VLM在Qwen3.5-397B-A17B上full-parameter SFT，256×H200，903 step，~40小时，~105 GB/GPU memory。

**VQA result** (Table 2):
- Overall: 68.2%（vs GPT-5.4的60.2%，+8.0 point）
- 最大gain在Act. axis: 75.7% vs 64.6%
- vs base Qwen3.5-Plus: 55.9% → 68.2%（+12.3 point）

**Caption result** (Table 3):
- Easy Overall: 83.2%（vs GPT-5.4的81.4%）
- **Hard Overall: 82.2%**（vs GPT-5.4的78.0%，+4.2 point）

Hard setting的result尤其重要——它证明RoboFine-VLM不是靠task-level language prior工作，而是真正capture了execution process。

**Human alignment**: 10个rater的ranking和benchmark score相关性极高（Easy: Pearson 0.937, Spearman 0.943; Hard: 0.922 / 0.943）。这验证了metric validity。

## 5. FineVLA-Policy的Core Experiment

### 5.1 Architecture Setup

两个framework，共享Qwen3.5-4B backbone：
- **StarVLA-OFT**: lightweight MLP regression head读predefined action token的hidden state，L1 objective回归continuous action chunk（follow OpenVLA-OFT）
- **StarVLA-GR00T**: dual-system design，VL backbone做System 2（slow reasoning），DiT-based flow-matching module做System 1（fast action generation），follow GR00T N1.5

### 5.2 Data Mixing的关键Design

**这是paper最clean的experiment design**：FG dataset和Raw dataset共享identical trajectory、action label、visual observation——唯一变量是paired language instruction。

FG:Raw ratio控制training step中sample每个dataset的概率。FG:Raw = 2:1意味着约2/3 step用FG instruction，1/3用Raw。这isolates了language supervision effect from data scale, embodiment, action distribution的confound。

七个ratio配置：Raw-only, 1:4, 1:2, 1:1, 2:1, 4:1, FG-only。

三个(dataset, framework)组合：RDT-OFT, RDT-GR00T, AlohaMix-OFT。其中AlohaMix约13x大于RDT in episode count（86,662 vs 6,061），限制在ALOHA-compatible dual-arm embodiment避免cross-embodiment confound。

### 5.3 RoboTwin Result的Inverted-U现象

Table 4的核心数据，AlohaMix-OFT：

| FG:Raw | Easy | Hard |
|---|---|---|
| Raw-only | 71.8 | 71.4 |
| 1:4 | 75.3 | 74.3 |
| 1:2 | 82.8 | 78.6 |
| **1:1** | **86.8** | **82.5** |
| 2:1 | 80.9 | 79.3 |
| 4:1 | 79.5 | 78.5 |
| FG-only | 78.3 | 76.1 |

**Inverted-U peak在FG:Raw = 1:1**，gain over Raw-only是+15.0/+11.1。

**Inverted-U的intuition**（这是paper最深刻的insight之一）：

- **Raw instruction preserve compact goal semantics**——what task to complete
- **FG instruction expose execution constraint**——how task to perform
- **Raw-only下**：execution-level choice（arm, approach, rotation）靠implicit co-occurrence statistic，policy guess得太弱
- **FG-only下**：policy有explicit process-level guidance，但lose goal-level abstraction，weaken generalization to unseen instruction phrasing；且FG description更长、distributionally更different from natural user command
- **Mixed下**：policy同时学task semantic和execution constraint，两者complementary

这个inverted-U是典型的"两信号complementary"现象，在多任务learning, multi-modal training里也常见——signal太多或太少都不行。

### 5.4 Architecture Gap Narrows with FG Supervision

Table 23 Panel B的关键数据，RDT上OFT vs GR00T的gap：
- Raw-only: 6.4/6.6（Easy/Hard）
- Best mixed: 4.7/4.2
- FG-only: 0.8/0.5

**Intuition**: 当language supervision不够informative（Raw-only），policy必须靠action decoder的capacity弥补——OFT的MLP head比GR00T的flow-matching DiT更直接高效，所以gap大。当language supervision足够dense（FG-only），supervision本身已经给了足够execution constraint，architecture choice变得less critical，gap collapse到0.8/0.5。

这让我联想到language model里**induction head的capacity依赖**——strong signal能compensate弱architecture。

### 5.5 Data Scale Effect

FG-only improvement over Raw-only：
- RDT: +1.4/+2.0
- AlohaMix: +6.5/+4.7

**Intuition**: trajectory diversity越大，dense action-aligned language有更多distinct execution pattern可以bind到。在小dataset上，FG instruction的marginal value有限——因为pattern不够多；在大dataset上，每个distinct pattern都能从FG supervision中benefit。这暗示fine-grained supervision是**scalable axis**，而非incidental improvement——未来scale越大收益越大。

## 6. Real-World Steerability的Factor-Level Analysis

### 6.1 Suite Design

Cobot Magic dual-arm，14 joint + 2 gripper continuous command，3 RGB camera（2 wrist + 1 third-person），action chunk length 50，30Hz controller。

12 tabletop task，600 demonstration，100k step fine-tune。

**Paired variant设计**是最精妙的部分：每个factor下，两个variant共享same visual scene，只改一个language-specified control factor：
- Color: red pen vs blue pen
- Pose: lying vs standing cup
- Approach: from above vs from right side
- Rotate: clockwise vs counter-clockwise
- Arm: right hand to right bowl vs left hand to left bowl
- **OOD probe**: right hand to right bowl vs left hand to **right** bowl（unseen actor-target binding）

### 6.2 Per-Factor Result

Table 5的关键数据，FG:Raw = 1:1 vs Raw-only的gain：

| Factor | Raw-only | FG:Raw=1:1 | Gain |
|---|---|---|---|
| Pose | 24 | 47 | **+23** |
| Color | 22 | 40 | **+18** |
| Approach | 60 | 78 | **+18** |
| Rotate | 76 | 86 | +10 |
| Arm | 60 | 64 | +4 |
| Clean Table | 72 | 84 | +12 |
| Stack Block | 35 | 40 | +5 |
| OOD Arm | 0 | 10 | +10 |

**Gain magnitude的排序与raw instruction underspecification程度正相关**：

- Pose, Color, Approach在goal-level language里完全unspecified——gain最大
- Rotate在task context里偶尔implied（"rotate"可能暗示方向）
- Arm selection在task context里更常implied

这验证了"FG supervision直接improve factor-level steerable control on attribute that raw instruction provide no guidance"。

### 6.3 OOD Compositional Limitation

OOD probe（L→R）显示：mixed model从0涨到10/100，improvement存在但远未解决。Table 26显示language-critical subgoal success rate（只看是否选对factor，不看最终completion）：
- Raw-only: 3/10
- 1:1: 6/10
- 2:1, 4:1: 4/10

**Intuition**: 增加FG supervision确实improve active-arm grounding（model学会了听instruction选arm），但unseen actor-target binding仍未解决——model能选对arm，但不能正确把arm binding到指定的target receptacle。这是**factor-level grounding和compositional generalization之间的gap**——前者是table内的，后者是table外的。

## 7. Failure Mode Analysis

Paper诚实地承认两类failure：
1. **Grounding failure**: policy选错object/arm/target，即使language specify了correct factor
2. **Execution failure**: 选对了factor但physical manipulation失败（unstable grasp, incomplete rotation, inaccurate placement）

这两类failure是orthogonal的——前者是perception+language grounding问题，后者是low-level motor control问题。FineVLA主要address前者，但execution failure仍需separate解决（可能需要force feedback, tactile sensing等）。

## 8. Critical Reflection与Open Question

### 8.1 Strong Point

1. **Cleanest experiment design**: 控制variable isolate language supervision effect，这种abstraction在VLA paper里很罕见
2. **Inverted-U现象很深刻**: 不是"more data better"的简单narrative，而是signal complementarity的subtle现象
3. **Factor-level evaluation suite**: paired variant设计让我们能看到per-factor gain而不只是aggregate success rate
4. **Schema设计系统化**: 10个dimension覆盖execution manifold
5. **Architecture-independence**: 两个framework都show一致trend，避免architecture-specific artifact

### 8.2 潜在Concern

1. **Real-world evaluation scale有限**: 只有12 tabletop task，600 demonstration，相比真real-world deployment仍太小
2. **RoboFine-VLM未用于policy supervision**: paper明确说policy experiment用human-verified label，而RoboFine-VLM只是future scalable annotator。这说明**RoboFine-VLM的annotation质量仍不够直接用**，human-in-the-loop不能完全去除
3. **Caption track的LLM judge dependency**: 用GPT-5.4-Pro做alignment judge，虽然swap到Gemini-3.1-Pro结论robust，但仍是closed-source dependency
4. **OOD compositional generalization未解决**: L→R从0到10仍远未实用
5. **跨embodiment未验证**: AlohaMix限制在ALOHA-compatible，cross-embodiment effect未测

### 8.3 对未来工作的Implication

1. **Fine-grained supervision是scalable axis**——data scale越大收益越大，未来更大scale的VLA training应考虑
2. **Inverted-U trend暗示mixed instruction是better training recipe**——不要完全替换raw instruction
3. **Architecture gap narrows的发现**: 暗示当supervision足够dense，architecture choice可以simplify，可能节省compute
4. **Factor-level evaluation的methodology值得推广**: paired variant设计可以扩展到更多factor和更大scale

## 9. Reference Link

- FineVLA Project Page: https://finevla.xlang.ai/
- LeRobot (HuggingFace): https://github.com/huggingface/lerobot
- DTW (Wikipedia): https://en.wikipedia.org/wiki/Dynamic_time_warping
- RoboTwin: https://github.com/TianxingChen/RoboTwin
- OpenVLA-OFT: https://github.com/moojink/openvla-oft
- NVIDIA Isaac GR00T: https://github.com/NVIDIA/Isaac-GR00T
- Qwen3-VL: https://qwenlm.github.io/blog/qwen3-vl/
- BridgeData V2: https://arxiv.org/abs/2308.12952
- DROID: https://droid-dataset.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- StarVLA: https://arxiv.org/abs/2604.05014
- π (Physical Intelligence): https://arxiv.org/abs/2604.15483
- RDT-1B: https://arxiv.org/abs/2410.07864
- RoboMIND: https://arxiv.org/abs/2510.17801
- Galaxea: https://arxiv.org/abs/2509.00576
- RoboCOIN: https://arxiv.org/abs/2511.17441
- RH20T: https://arxiv.org/abs/2307.00595

## 10. 总结的Intuition

Andrej，让我把这篇paper的intuition用最compact的方式总结：

FineVLA本质上是把**language supervision从trajectory-level sparse变成step-level dense, action-aligned**。这解决的不是policy capacity问题，而是**signal density问题**——VLA model的capacity够，但language supervision不够informative，policy学不到execution manifold的结构。

Inverted-U现象是核心insight：fine-grained supervision是signal augmentation而非replacement。Raw instruction提供task semantic abstraction（高维compression），FG instruction提供execution constraint（低维decomposition）。两者在information bottleneck的两端，mixed supervision让policy同时access两端。

Architecture gap narrows的发现进一步支持这点——当supervision signal足够dense，policy的architecture capacity不再bottleneck，这意味着fine-grained supervision是比architecture更fundamental的scaling axis。

Real-world的per-factor gain排序（pose/color/approach > rotate > arm）精准对应了instruction underspecification程度，这从另一个角度验证了"FG supervision补的是raw instruction没说清楚的部分"这个核心论点。

未来最重要的open question是**compositional generalization**——单factor grounding可以学，但unseen factor combination binding仍未解决。这可能需要从instruction-level supervision走向structural compositional representation，类似language model里compositional generalization的研究方向。
