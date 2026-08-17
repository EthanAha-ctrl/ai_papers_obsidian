---
source_pdf: World-Ego Modeling for Long-Horizon Evolution in Hybrid Embodied Tasks.pdf
paper_sha256: d02e593aa682f277e378b78d0528a3ce0cb6c3a4d2f8c777e7007650c4b22107
processed_at: '2026-08-13T05:13:48-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 WEM

## 1. 这篇paper到底在解决什么问题?

想象你在玩一个游戏,你操控一个机器人在房间里:先走过去(nav),然后开柜门、拿杯子、放桌上(manip),再走到厨房(nav),开冰箱、拿牛奶(manip)……

现在你想训一个model,让它**看完开头几帧 + 你的指令,自动脑补出后面的视频**。这就是video world model在embodied AI里的角色——它是一个"未来模拟器",预测"如果我这么做,世界会怎么变"。

问题出在哪?现有model把**所有东西塞进一个generation stream**一起生成。房间里墙怎么摆、柜子在哪、地板纹理……这些是**几乎不随你动作变的**。同时你的机器人手臂在动、杯子被夹起来……这些是**完全由你指令驱动的**。

这两类信息的时间尺度差了几十倍——墙几小时都不动,手臂几帧就变。你让一个stream同时denoise这两类东西,它就左右为难:

- 关注scene consistency吧,manipulation动作就糊了
- 关注action sharpness吧,背景慢慢漂移,长horizon下来房间变形

这paper的核心claim:**把这两类信息显式拆开建模,长horizon rollout就不会崩**。

## 2. "World"和"Ego"到底是什么?

paper最漂亮的contribution是它没直接给一个定义,而是先说:**这个词太泛了,我们得先界定边界在哪**。它给了三种划法:

### 画法一:按"谁在动"分(Motion-based)
房间里所有运动要么是相机移动导致的(你走的时候,墙也"动"),要么是物体被你碰导致的(杯子被夹起来)。

你走的时候,墙往反方向"流",这个流是可以预测的——就叫world。杯子突然飞起来,这个流跟相机运动对不上——就叫ego。

直觉上很合理,但实际不work:你手挡住杯子的时候,optical flow算不出来;你转身的瞬间,墙的flow也不规则。

### 画法二:按"谁是谁"分(Semantic-based) ← 这个赢了
机器人自己 + 正在被你操作的物体 = ego。其他全部 = world。

**关键insight**:这个边界是**会动的**。柜子门,你还没碰的时候属于world;你抓住把手那几秒它变成ego;你松手之后它又回归world。这种"interaction-dependent boundary"我觉得是整篇paper最深刻的思想——边界本身是task-driven的动态量。

### 画法三:按"信息从哪来"分(Intention-based)
历史视频里能看到的 = world;当前这条新指令要产生的 = ego。

不切割pixel,只切割conditioning source。让model自己学怎么从两类conditioning里取信息。问题是太implicit,分离效果弱。

实验结果:Semantic (61.48) > Motion (59.36) > Intention (58.69)。

**为什么semantic赢**:它给的signal最直接、最对应task-relevant区域。Motion受flow estimation noise污染,intention太软。

## 3. 拆开之后怎么合起来?三种"解耦程度"

光定义边界还不够,得在architecture里真的让两部分分开算。paper又给了三种方案:

### Pre-disentanglement:前面分开,后面合
token分两组过restricted cross-attention,但后面还是shared computation。等于"前期分流,后期混回",分离不彻底。

### Post-disentanglement:前面合,后面分开再融合
两个expert都看完整token sequence,各算各的,最后用mask加权融合。

问题:每个expert都看full sequence,会互相干扰,world expert在偷偷学ego的东西,反之亦然。

### Full disentanglement:前分、中专、后合 ← WEM用的
- 用semantic mask把token**路由**到不同expert
- 各expert只处理自己那片区域(+ 邻域,避免接缝处artifact)
- 算完用同一个mask**unroute**回单一序列

这是最强分离。EWMScore 61.48 vs w/o disent 58.40,3分的gap全来自disentanglement。

**这3点gap很关键**:告诉我们video world model的long-horizon瓶颈不在capacity,在structure。

## 4. WEM具体长什么样?

### State Predictor:先"读懂"历史
基于Qwen3-VL-2B,输入序列像这样:

```
[初始帧] [指令1] [视频1] [指令2] [视频2] ... [当前指令] [192个world query] [64个ego query]
```

VLM forward一遍,最后那些query位置的hidden state就是 S_k^w 和 S_k^e。

两个关键设计:

**不对称预算**:192 world query vs 64 ego query。World要存长期scene structure(信息量大),ego只存当前action的局部dynamics(信息量小)。强行对称是错的prior。

**Role-Conditioned Attention**:world query只能看历史视觉+过去指令,**看不到当前指令**;ego query只能看当前指令+最近几轮,**看不到远古历史**。

这个attention mask设计很优雅——同一个backbone,通过mask就实现了soft modularity,world和ego不会互相污染,即使共享参数。

### Generator:基于Wan2.2-TI2V-5B的DiT,改造成CP-MoE

DiT的block分成两段:
- 前面一段shared,叫**Preceding Expert**——同时condition on S_k^w和S_k^e,理解"scene和robot在什么joint configuration"
- 后面一段复制成两份,**World Expert**只condition S_k^w,**Ego Expert**只condition S_k^e

Preceding Expert中间接出来一个**Semantic Head**(DPT架构),预测一个binary mask:哪些video patch属于world,哪些属于ego。这个mask就是routing signal。

Routing的时候有个细节:每个expert的active区域要往外扩几个pixel,不然mask边界处会出现接缝artifact。这个细节ablation下来影响最大(-1.91分),说明hybrid task里phase transition的地方boundary变化剧烈,seam问题特别严重。

### 训练
两个loss:
- Flow-matching loss(继承自Wan的diffusion训练)
- Mask loss(BCE + Dice,因为ego区域通常很小,class imbalance严重)

mask loss权重λ=0.3,训练过程中anneal到20%——early stage监督最强,等model学到mask之后逐渐放松,让它自己refine。

mask的ground truth来自BEHAVIOR-1K simulator的instance segmentation——这是个**privileged signal**,real-world部署时拿不到,是paper最大的limitation。

## 5. HTEWorld benchmark — 填了一个空白

现有benchmark要么测短horizon manipulation(WorldArena),要么测单prompt生成(VBench),没有一个测"nav-manip-nav-manip"这种长链条混合任务。

HTEWorld基于BEHAVIOR-1K构建:
- 125K训练clips,4.5M frames
- 300个multi-turn eval trajectories,2K+ instructions
- 16 FPS, 480×480

6个新metric,设计哲学都是"测结构化演化而非单帧质量":

**RCBD**:测chunk边界处的appearance和motion连续性。不奖励over-smoothing——如果你生成的边界比GT还平滑,说明model在"偷懒"把所有东西都smooth掉,ratio对不上,score会低。

**LPSA**:后期chunk权重更高,因为error是累积的,后期更难。

**CISR**:用retrieval task测instruction alignment——给定生成的chunk,能在所有GT chunk里检索到对应step吗?

**PMPA**:把每个chunk的motion时间profile提取成4维特征(median flow、top-20% flow、log ratio、flow entropy),比较和GT的profile距离。Nav和Manip的motion profile本质不同,这个metric分别测。

**CPDM**:测phase可分性——生成的nav chunk和GT的nav chunk应该比和GT的manip chunk更相似,margin越大越好。

**FPHS**:phase切换那几帧(nav→manip的瞬间),在变化最剧烈的局部region测consistency。这是autoregressive rollout最容易崩的地方。

## 6. 实验结果讲了什么故事?

### Main result(Table 3)
WEM 61.48,比PAN-style baseline高3分,比Cosmos-14B高6分。Gains集中在:
- Boundary Consistency: +7.68(最大)
- Scene Consistency: +7.28
- Instruction alignment: +4.47

Local visual quality(IQ/AQ)提升很小。这完美对应paper的hypothesis:disentanglement的收益在long-horizon structure,不在单帧quality。

### Ablation的关键发现
1. Semantic view比motion/intention view好2-3分
2. Full disentanglement比pre/post好1-2分
3. w/o semantic proxy: 58.59 → w/ proxy: 61.09(3分全来自proxy)
4. w/o neighbor-expanded routing: 59.57(最大降幅,seam artifact是主要failure mode)

第3点是个critical question:WEM的gain到底来自architecture还是来自privileged mask supervision?Paper没完全回答这个。

### Compatibility(Table 5)
在原版WorldArena(manipulation-only),WEM 57.90,低于IRASim和CtrlWorld。这是generalist vs specialist的classic trade-off——为hybrid优化,在单task上牺牲specialization。

## 7. 我的intuition和思考

### 为什么这个idea work?本质上是个time-scale separation
长horizon rollout有两条error accumulation path:
1. Scene drift:慢,几百帧才看出来
2. Action misalignment:快,几十帧就崩

Monolithic model用一个stream denoise两者,时间尺度差2个数量级,必然顾此失彼。WEM的factorization本质是**multi-timescale inductive bias**——这是LSTM/Highway Network/Transformer-XL里multi-timescale idea在video diffusion的reincarnation。

### Interaction-dependent boundary是深刻的思想
传统semantic segmentation是static的——杯子永远是杯子。但WEM说:**物体的"角色"会随interaction状态变**。杯子在桌上是world的一部分,你拿起来那几秒它是ego的一部分,放下又变回world。

这个dynamic boundary idea我觉得可以extend到很多场景:
- Autonomous driving:前面的车,你不碰它是world,你变道要绕它时变成ego-relevant
- Multi-agent:每个agent有自己的ego,world是shared
- Tool use:工具本身是world,被抓起来用的时候是ego的extension

### 和JEPA家族的对照
V-JEPA 2(LeCun线)在latent space做类似的separation——representation learning和action head分开。WEM在pixel/video space做。两者的共同intuition:**predictive responsibility应该modular化**。

差别:JEPA不生成pixel,所以不需要routing;WEM生成pixel,必须决定每个pixel由谁负责,所以需要explicit boundary。

### 跟MoE的关系
WEM的CP-MoE和standard sparse MoE(Switch Transformer、Mixtral)很不一样:
- Sparse MoE:router是learned的,token可以走任何expert,有load balancing问题
- CP-MoE:routing是predefined的(由semantic mask决定),所有expert永远active,没有load balancing问题

这其实是**structured MoE**——用task prior替代learned routing。好处是interpretable,坏处是放弃了router的学习能力。在structured task(embodied)上,prior > learned;在open-domain text generation上,learned > prior。这暗示MoE设计应该task-aware。

### Critical的几个点
1. **Privileged supervision问题**:semantic mask来自simulator的instance segmentation,real world没有。Paper的future work提到self-supervised boundary discovery,这是make-or-break的问题。如果real world必须靠instance segmentation,WEM的实用价值就受限。

2. **Gain的attribution**:w/o proxy 58.59 → w/ proxy 61.09,3分来自proxy本身。那baselines如果也给semantic mask supervision,差距还会这么大吗?Paper没做这个controlled comparison。

3. **Architecture复杂度**:RCA + asymmetric query + CP-MoE + DPT head + neighbor expansion,很多moving parts。是否minimal sufficient?会不会有simpler的architecture达到类似效果?

4. **Long-horizon到底多long**:HTEWorld的eval trajectory多少chunk?如果是5-10个chunk,和真实embodied task的几百步还差很远。Error accumulation的scaling law是什么?

5. **Sim-to-real gap**:BEHAVIOR-1K是simulation,visual diversity、contact dynamics都简化了。Real world的perception noise会直接污染semantic mask prediction,可能cascade成routing错误。

## 8. 这个工作在field里的位置

WEM属于一个emerging trend:**structured inductive bias for video world models**。同期工作:
- VideoREPA:relational alignment with foundation model
- Tesseract:4D embodied world model
- Longscape:context-aware MoE for long-horizon

WEM的独特angle是**concept-first**——先build world-ego的taxonomy(3 views × 3 strategies),再instantiate。这种"先把概念理清楚再写代码"的方法在RL/embodied field里undervalued,但这paper显示它yield principled design decisions。

它和policy端的modular policies(MOKA, RT-2 with semantic routing)也是同一线——**structured modularity beats monolithic model** when task有compositional structure。

## 9. 如果我来extend这个工作

1. **3D-aware boundary**:用NeRF/Gaussian Splatting的geometry做boundary,而不是2D mask。Occlusion处理更好,egomotion下boundary更consistent。

2. **Causal boundary discovery**:用counterfactual——"如果指令换掉,哪些pixel会变?"变的就是ego。这给self-supervised boundary estimation提供principled框架。

3. **World-ego policy learning**:paper只做video prediction,但这个decomposition对policy直接relevant——world state提供planning context,ego state提供action grounding。可以训一个policy head,让它从S_k^w读"在哪",从S_k^e读"做什么"。

4. **Autonomous driving迁移**:World-Ego Modeling对自动驾驶天然契合——road scene regularities(world) vs ego-vehicle behavior(ego)。BEHAVIOR-1K格式和nuScenes/ Waymo有结构相似性。

5. **Adaptive capacity allocation**:固定192/64不optimal。复杂scene应该给world更多query,复杂manipulation应该给ego更多。可以学一个meta-controller动态分配。

## 10. 一句话总结

WEM证明了一件事:**长horizon embodied video generation的瓶颈不在model capacity,在predictive structure**。把"世界怎么变"和"agent怎么动"显式分开建模,听起来像common sense,但这paper第一次系统地operationalize这个idea,给出清晰的taxonomy、可控的architecture、专门的benchmark。3分的提升不大,但来自structure而非scale,这暗示下一个decade的embodied world model应该是modular的,不是monolithic的。

## 关键links

- Paper: https://zgca-hmi-lab.github.io/WEM
- Code: https://github.com/ZGCA-HMI-Lab/WEM
- Model: https://huggingface.co/Zoorao/WEM
- Dataset: https://huggingface.co/datasets/Zoorao/HTEWorld
- Wan video model: https://arxiv.org/abs/2503.20314
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- Cosmos-Predict 2.5: https://arxiv.org/abs/2511.00062
- WorldArena: https://arxiv.org/abs/2602.08971
- BEHAVIOR-1K: https://arxiv.org/abs/2210.04426
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- GEM: https://arxiv.org/abs/2505.14698
- Iso-Dream: https://arxiv.org/abs/2205.13521
- DPT: https://arxiv.org/abs/2104.06254
- Flow Matching: https://arxiv.org/abs/2210.02747
- WoW: https://arxiv.org/abs/2509.22642
- PAN: https://arxiv.org/abs/2511.09057

---

# World-Ego Modeling for Long-Horizon Evolution in Hybrid Embodied Tasks — 深度讲解

## 1. Paper 的核心 thesis: World-Ego Entanglement 问题

这篇paper要解决一个我在做video world model研究时也经常困扰的问题:当embodied agent做长horizon任务时,传统的monolithic world model把两类**本质不同的预测责任**塞进一个生成stream里。

**World** 承载persistent、instruction-agnostic的scene regularities(layout、object permanence、background geometry);**Ego** 承载robot-centric、instruction-conditioned的dynamics(gripper motion、contact interaction、被操作object的运动)。这两类信息的时间尺度、信息来源、物理本质都不同,把它们collapse到一个stream里会导致:

- Long-horizon scene consistency退化(navigation阶段background漂移)
- Instruction alignment弱(manipulation阶段action不精确)
- 尤其是hybrid navigation-manipulation任务里两类dynamic交替出现,单stream model很难兼顾

Paper的central claim:**显式decompose world和ego可以align predictive structure with underlying physical reality**。

Paper link: https://zgca-hmi-lab.github.io/WEM
Code: https://github.com/ZGCA-HMI-Lab/WEM

## 2. 三种 World-Ego Boundary 定义 — 这是核心conceptual contribution

paper探索了三种把"world"和"ego"边界划在哪的方式,这是非常思辨性的taxonomy:

### 2.1 Motion-based view
基于**motion source**。Static-scene assumption下,camera ego-motion induces a predictable scene flow over background。匹配scene flow的pixel → world(scene-induced viewpoint change);deviate from scene flow的pixel → ego(contact-driven object dynamics)。

技术proxy:**object residual flow**。

具体地,使用 RAFT (https://arxiv.org/abs/2003.12039) 估计dense optical flow F,然后用 RANSAC fit homography H 得到camera-induced flow F_cam,残差:
$$\mathbf{F}_{obj} = \mathbf{F} - \mathbf{F}_{cam}$$

- F: raw dense optical flow between consecutive frames
- F_cam: camera-induced flow rendered by applying homography H to every pixel coordinate
- F_obj: residual object flow,large magnitude区域 = 接触驱动object运动

这个view的问题:**大视角变化和contact-rich interaction下,flow decomposition噪声大**。Camera motion不再是简单homography,物体可能同时被相机跟随和被manipulate。

### 2.2 Semantic-based view (WEM default)
基于**embodied role of scene entities**。Robot body + 当前被manipulate的object = ego region;background + 未被manipulate的object = world region。

**关键洞察**:world-ego boundary是**interaction-dependent**的——一个movable object在interaction前属于world,被acted upon时变成ego-related,interaction完成后又回归world。这个dynamic boundary定义在paper里很优雅。

技术proxy:**semantic mask**(来自BEHAVIOR-1K simulator的instance segmentation)。

这个view的优势是直接separate interaction regions和persistent scene regions,同时允许两者都在egomotion下移动(不像motion-based那样把egomotion本身都归到world)。

### 2.3 Intention-based view
基于**conditioning source**。World = visual history确立的内容;Ego = current instruction诱导的dynamics。

这个view和 Iso-Dream (NeurIPS 2022, https://arxiv.org/abs/2205.13521) 隔离controllable/noncontrollable dynamics的思路相关,但是paper做了重要的reframe:不再是controllable/noncontrollable,而是instruction-induced ego dynamics vs. history-established world regularities。

技术proxy:不做pixel-space partition,而是partition conditioning sources,model implicit learn如何整合。具体实现是:
- World state S_k^w 注入到每个decoder layer作为cross-attention memory tokens
- Ego state S_k^e 通过AdaLN modulation全局控制

且world state用GRU-style更新:
$$\mathbf{S}_k^w = \mathbf{G}_k \odot \mathbf{S}_{k-1}^w + (1-\mathbf{G}_k) \odot \tilde{\mathbf{S}}_k^w$$

- S_k^w: 第k步world state,shape [N, D]
- G_k: keep gate,shape [N, D],由 S_{k-1}^w、Ŝ_k^w (candidate)、S_k^e的mean-pooled summary共同计算
- Ŝ_k^w: candidate update,用单独的reset gate生成
- ⊙: element-wise product

G_k conditioned on ego state意味着当ego action是transient的,gating可以抑制world state update,preferentially保留stable scene info。

**实验结果(Table 1)**:
- Intention-based: 58.69
- Motion-based: 59.36
- Semantic-based: 61.48 ← winner

intuition:explicit spatial boundary比implicit conditioning separation更强;semantic boundary比motion boundary更鲁棒,因为后者受flow estimation noise影响。

## 3. 三种 Disentanglement 策略 — 架构层面的设计

给定一个boundary定义,如何把disentanglement注入到generator架构里?Paper提出三种策略:

### 3.1 Pre-disentanglement (Fig 3b)
在rear stage的**input端**引入分离。Preceding expert输出一个proxy,把token分成world和ego两组,通过一个shared rear module但用**restricted cross-attention**(world token只attend S_k^w,ego token只attend S_k^e)。

问题:**downstream computation是shared的**,分离不彻底。

### 3.2 Post-disentanglement (Fig 3c)
在rear stage的**output端**引入分离。Rear stage复制成World Expert和Ego Expert,两者都处理**完整token sequence**但condition各自的state;proxy作为soft mask融合两个输出。

变体:可以移除semantic proxy用constant assignment。这个ablation很informative——
- Post-disent. w/o semantic proxy: 58.59
- Post-disent. w/ semantic proxy: 61.09

**这3点提升基本都来自semantic proxy这个routing signal**,告诉我们explicit boundary supervision重要。

问题:每个branch都看full token sequence,会有**cross-role interference**。

### 3.3 Full disentanglement (Fig 3d, WEM default)
组合三件事:
- **Routing**: 用proxy先分token到World/Ego Expert
- **Branch processing**: 各expert只处理自己assign的token + 邻域token(避免seam artifact)
- **Unrouting**: 用同一个proxy把两个expert的输出recompose回single sequence

这是最强的分离。EWMScore: 61.48。

注意这个CP-MoE和standard sparse MoE (Switch Transformer, Mixtral)不同——**all three experts always active**, specialization来自predefined role assignment,不是learned routing。这避免了MoE常见的load balancing问题,但也放弃了router的学习能力。

## 4. WEM 架构详解

### 4.1 State Predictor (Vision-Language State Predictor Φ_φ)

基于frozen Qwen3-VL-2B-Instruct (https://arxiv.org/abs/2511.21631),input sequence结构是:

```
[O_0] [a_1] [V_1] [a_2] [V_2] ... [a_{k-1}] [V_{k-1}] [a_k] [World Queries ×192] [Ego Queries ×64]
```

- O_0: initial egocentric observation (image tokens)
- a_i: 第i条instruction (text tokens)
- V_i: 第i个generated video chunk (video tokens)
- 末尾append learnable query tokens

**两个关键设计**:

**(1) Asymmetric query budgets** — 192 world queries + 64 ego queries
intuition:world需要encode persistent scene structure accumulated across long histories(信息量大);ego只需要encode当前step的instruction-conditioned dynamics(信息量小)。Forcing equal capacity会误assume两者comparable。

**(2) Role-Conditioned Attention (RCA)** — 限制每个query group的attention horizon:
- World queries: attend to O_0 + 所有 {V_1..V_{k-1}} + 所有 {a_1..a_{k-1}} + 其他world queries;**blocked from a_k and ego queries**
- Ego queries: attend to其他ego queries + a_k + 最近K个instruction-video pairs;**distant history和world queries被mask**

这个mask设计很关键——它阻止 S_k^w 和 S_k^e collapse到shared representation,即使共享backbone。本质上是用attention mask做soft modularity。

**Ablation (Table 6)**:
- w/o Asymmetric Query Budget: 60.50 (-0.98)
- w/o RCA: 60.64 (-0.84)
- w/o Neighbor-Expanded Routing: 59.57 (-1.91, 最大降幅)

Neighbor-Expanded Routing影响最大,说明**seam artifact**在hybrid task里特别严重,因为navigation-manipulation phase transition时boundary变化剧烈。

### 4.2 CP-MoE Generator (基于 Wan2.2-TI2V-5B, https://arxiv.org/abs/2503.20314)

DiT block分割:
- **Early shared blocks** → Preceding Expert (shared encoder)
- **Later blocks** → 复制成 World Expert + Ego Expert (specialized)

#### Preceding Expert
每个block:
1. Self-attention on noise latent
2. **两条parallel cross-attention streams**:
   - Stream 1: attend to text instruction(继承pretrained DiT的原生text conditioning)
   - Stream 2a: attend to S_k^w
   - Stream 2b: attend to S_k^e
3. 两stream输出相加 → FFN

**关键**:Preceding Expert同时condition on S_k^w和S_k^e,这样才能learn joint configuration of scene context + embodied interaction——这是预测world-ego boundary的前提。

#### Role Experts
- 拓扑和Preceding Expert相同
- 但只condition**一个state**(S_k^w或S_k^e)
- Text cross-attention保留(从pretrained继承)
- 处理disjoint region of future video

#### Semantic Head, Routing, Unrouting
- **Semantic Head**: DPT-style (https://arxiv.org/abs/2104.06254) lightweight dense prediction transformer
- Tap encoder features at layers {5, 9, 13, 17, 21, 23} — coarse-to-fine fusion
- 同时fuse S_k^e的token
- 输出binary mask M ∈ {0, 1}^{T×H×W}(per video patch)

- **Routing**: M-assigned world tokens → World Expert; ego-assigned → Ego Expert;每个expert的active set扩展到spatial neighbors避免seam
- **Unrouting**: 用同一个M把两个expert的输出recompose回single sequence → video decoder → clean latent

#### Training Objectives
Total loss:
$$\mathcal{L} = \mathcal{L}_{flow} + \lambda \mathcal{L}_{mask}$$

- L_flow: flow-matching loss(继承自pretrained DiT,https://arxiv.org/abs/2210.02747)
- L_mask: 
$$\mathcal{L}_{mask} = \mathcal{L}_{BCE} + \mathcal{L}_{Dice}$$
  - L_BCE: class-balanced binary cross-entropy
  - L_Dice: Dice loss (https://arxiv.org/abs/1911.02855),处理class imbalance(ego region通常远小于world region)
- λ = 0.3,anneal to 20% of initial value over training → semantic supervision在early stage最强

Mask supervision来自BEHAVIOR-1K simulator的instance segmentation——这是一个**privileged supervision** signal,real-world部署时需要weakly-supervised alternative。

## 5. HTEWorld Benchmark — 这是另一个重要contribution

现有benchmark的gap:
- VBench/VBench++ (https://arxiv.org/abs/2410.21672): 通用video quality,不测embodied
- WorldModelBench (NeurIPS 2025): short-horizon
- WorldArena (https://arxiv.org/abs/2602.08971): manipulation-only
- 没有一个benchmark测long-horizon + hybrid navigation-manipulation

HTEWorld基于 BEHAVIOR-1K (https://arxiv.org/abs/2210.04426):
- 125K video clips
- 4.5M+ frames
- Fine-grained action-centric annotations
- 300 multi-turn evaluation trajectories
- 2K+ instructions
- 16 FPS, 480×480

### 6个 HTEWorld-specific metrics(很有意思的设计)

**Multi-Turn Continuous Generation metrics**:

**(1) RCBD (Rollout Chunk-Boundary Dynamics)** — 测chunk边界的appearance和motion dynamics fidelity
对每对consecutive (V_k, V_{k+1}):
- Appearance gap: $b_k = d_p(V_k^{(-1)}, V_{k+1}^{(1)})$ (LPIPS between last frame of k and first frame of k+1)
- Motion gap: $m_k$ = optical flow discontinuity across boundary
- Match score: $S(x,y) = \exp(-|\log(x/y)|)$ — ratio alignment
- Boundary score: $\sqrt{S(b_k, b_k^*) \cdot S(m_k, m_k^*)}$ (geometric mean)
- RCBD = mean over all K-1 boundaries

intuition:这个metric不奖励over-smoothing。如果generated boundary比GT太smooth(常见failure mode),ratio会偏离1,S会低。

**(2) LPSA (Late-Prefix State Alignment)** — 强调later chunks
$$LPSA = \frac{\sum_k k \cdot r_k}{\sum_k k}$$
- r_k: cosine similarity of CLIP features of last W=4 frames of V_k vs V_k^*
- 线性weight让later chunks(accumulated error大)contribute更多

**(3) CISR (Chunk Instruction-Step Retrieval)** — retrieval-based alignment
对每个 V_k,在所有GT chunks里retrieval,看正确match的reciprocal rank。
$$CISR = MRR = \frac{1}{K}\sum_k \frac{1}{\text{rank}_k}$$

**Navigation-Manipulation Generation metrics**:

**(4) PMPA (Phase-Matched Motion Profile Alignment)** — 4-dim motion profile
$$[\bar{u}/L, \tilde{u}/L, \log(1+\tilde{u}/\bar{u}), \mathcal{E}(u)]$$
- ū: median optical flow magnitude
- ũ: top-20% mean
- L: frame diagonal length (normalization)
- E(u): normalized flow entropy (motion complexity)
- Resample to 16 time steps, L2 distance → score $\exp(-\delta/\tau)$

**(5) CPDM (Cross-Phase Discriminative Margin)** — phase可分性
$$\text{score} = \sigma\left(\frac{r^+ - r^-}{\tau}\right), \tau = 0.05$$
- r+: 同phase GT chunk similarity
- r-: max over不同phase GT chunks (hard negative)

**(6) FPHS (Frontier Phase-Hop State Consistency)** — phase transition处的local consistency
- 找phase switch的boundary
- R=4 frames window on each side
- Change region: accumulate GT optical flow magnitudes, top-20% spatial area
- Crop到该region算feature similarity

这6个metric的设计哲学:**测的不是单帧质量,而是long-horizon structured evolution**。RCBD和FPHS特别精妙,因为它们针对autoregressive rollout的failure modes(boundary discontinuity和phase transition artifact)。

## 6. Experiments 详解

### 6.1 Main Comparison (Table 3, WorldArena metrics)

| Model | EWMScore | Flow | MS | BC | SC | Persp | Act | Inst |
|-------|----------|------|-----|-----|-----|-------|-----|------|
| WoW-7B | 53.44 | 25.49 | 67.74 | 66.86 | 63.06 | 95.14 | 78.42 | 80.90 |
| Cosmos-2B | 54.83 | 27.23 | 69.33 | 71.54 | 66.88 | 95.28 | 79.60 | 83.02 |
| Cosmos-14B | 55.41 | 32.34 | 71.63 | 73.85 | 68.65 | 94.40 | 80.20 | 84.70 |
| PAN-style | 58.40 | 47.43 | 79.47 | 80.24 | 74.79 | 95.08 | 80.40 | 86.33 |
| **WEM** | **61.48** | 49.21 | 82.70 | 87.92 | 82.07 | 97.60 | 82.00 | 90.80 |

WEM相对PAN-style baseline提升3.08分,主要gains在:
- **BC (Boundary Consistency): 87.92 vs 80.24** (+7.68) — chunk间一致性大幅提升
- **SC (Scene Consistency): 82.07 vs 74.79** (+7.28) — long-horizon scene stability
- **Persp (Perspective): 97.60 vs 95.08** — 3D consistency
- **Inst (Instruction alignment): 90.80 vs 86.33** (+4.47) — instruction跟随

这正符合paper的hypothesis:disentanglement收益在consistency和control,而非local visual quality(IQ/AQ提升较小)。

### 6.2 Hybrid Task Metrics (Table 4)

WEM在所有6个metric都领先,但提升幅度较小(0.01-0.08)。这说明HTEWorld-specific metric可能已经饱和,或者long-horizon evaluation本身有noise floor。

### 6.3 Compatibility with Manipulation-Only (Table 5)

WEM在原版WorldArena: 57.90,低于IRASim(58.10)和CtrlWorld(58.12),但仍competitive。

intuition:WEM是为hybrid task优化的,在纯manipulation上sacrifice了一些specialization。这是reasonable的trade-off——generalist vs specialist的classic tension。

## 7. 与相关工作的context — build intuition

### 7.1 JEPA family (Yann LeCun line)
- V-JEPA 2 (https://arxiv.org/abs/2506.09985): 在representation space预测,separate representation learning from action head
- WEM和JEPA的相似处:**都强调separate不同predictive responsibility**
- 不同:JEPA在latent space,WEM在pixel/video space;JEPA分离的是representation vs action,WEM分离的是world vs ego

### 7.2 GEM (CVPR 2025, https://arxiv.org/abs/2505.14698)
- GEM: ego-vision world model,factor未来video成ego motion + object dynamics + scene composition
- 和WEM motion-based view接近,但WEM更systematic——3个view × 3个disentanglement strategy的grid

### 7.3 Iso-Dream (NeurIPS 2022)
- Isolate controllable vs noncontrollable dynamics
- WEM intention-based view是这个idea的扩展:把"controllable"refine成"instruction-conditioned ego dynamics"

### 7.4 WoW (https://arxiv.org/abs/2509.22642) 和 CtrlWorld (https://arxiv.org/abs/2602.08971)
- WoW: large-scale robot trajectory + inverse dynamics
- CtrlWorld: pose-conditioned memory retrieval + frame-level action conditioning
- 两者都是monolithic stream,WEM的contribution是显式factorize

### 7.5 PAN (https://arxiv.org/abs/2511.09057)
- Generative latent prediction for long-horizon
- WEM的PAN-style baseline就是这个
- WEM可以看作PAN + world-ego decomposition

### 7.6 Cosmos-Predict 2.5 (https://arxiv.org/abs/2511.00062)
- NVIDIA的video world foundation model
- Single-stream,WEM在HTEWorld上outperform它2.6分

## 8. 我的intuition和critical analysis

### 8.1 为什么world-ego decomposition有效 — information-theoretic视角
长horizon rollout的error累积有两条path:
1. **Scene drift**: background geometry/object layout在navigation阶段slowly drift
2. **Action misalignment**: manipulation阶段instruction没有精确follow

Monolithic model必须用一个stream同时denoise这两类error,且它们的时间尺度差几个数量级(scene regularities ~100 frames, action dynamics ~10 frames)。WEM的factorization本质上是**inductive bias matching the time-scale separation**——这是RNN/LSTM里multi-timescale idea的reincarnation,在video diffusion context。

### 8.2 Semantic view胜出的深层原因
Semantic mask直接对应**task-relevant interaction regions**。Motion view依赖optical flow decomposition,但contact-rich interaction时:
- Robot gripper遮住object,flow估计noisy
- Manipulated object可能rigid-body运动,flow magnitude不一定大
- Camera motion和object motion可能耦合

Intention view的weakness在于没有**explicit spatial routing**,所有token都看两个state,分离是soft的implicit。这和"explicit structure beats implicit learning"的一般intuition一致。

### 8.3 关键open question
Paper limitations section很坦诚:
1. **Sim-to-real**:BEHAVIOR-1K是simulation,real-world的perception noise、object variability更强。Semantic mask supervision在real world如何获得?Self-supervised mask discovery是key future direction。
2. **Boundary construction dependency**:需要instance segmentation,这是privileged signal。Weakly-supervised / self-supervised boundary estimation是critical。
3. **Residual long-horizon degradation**:disentanglement减轻但没消除error accumulation。需要hierarchical planning + explicit memory refresh + uncertainty-aware rollout。
4. **Adaptive state capacity**:固定192/64 query budget不optimal。Slot-based memory / variable-length tokens可能更好。

### 8.4 我会怎么extend
- **3D-aware boundary**:用NeRF/Gaussian Splatting的geometry做boundary,而不是2D semantic mask。这样occlusion处理更好,且egomotion下boundary更consistent
- **Causal disentanglement**:用counterfactual——"如果instruction换掉,哪些pixel会变?"这就是ego region。可以用causal inference framework做boundary discovery
- **Multi-agent extension**:把ego推广到multi-agent,每个agent有自己的ego state,world是shared。这对应cooperative manipulation场景
- **World-ego policy**:paper只做video prediction,但这个decomposition对policy learning也relevant——world state提供planningcontext,ego state提供action grounding
- **Autonomous driving迁移**:World-Ego Modeling对自动驾驶很natural——road scene regularities (world) vs ego-vehicle behavior (ego)。BEHAVIOR-1K的格式和Waymo/nuscenes有相似性

### 8.5 Critique
- **Evaluation noise**:Table 4的HTEWorld-specific metric提升很小(0.01-0.04),可能在metric noise floor内。需要更大scale的evaluation
- **Comparison fairness**:WEM用semantic mask supervision,而baselines没有。这个privileged signal是否是main driver?Ablation里Post-disent. w/o proxy (58.59) vs w/ proxy (61.09)显示3点提升来自proxy本身——这是critical question
- **Architecture complexity**:CP-MoE + RCA + DPT semantic head + asymmetric queries——很多moving parts,ablation只incremental验证。是否minimal sufficient design?
- **Long-horizon definition**:HTEWorld的"long-horizon"是300 trajectories × 2K instructions,但每个trajectory多少chunk?multi-turn到什么程度?需要看dataset stats更详细
- **Comparison with V-JEPA 2**:V-JEPA 2在representation space做similar decomposition,但没有pixel-level evaluation。WEM应该做latent-space ablation看representation是否也disentangle

## 9. 关键implementation细节

### 9.1 Caption生成pipeline(Appendix A)
用了Gemini-3-flash,动态prompt构建融合4个信息源:
1. Robot grounding(Galaxea R1 humanoid描述)
2. Episode trajectory context(完整action list,当前step highlighted)
3. Temporal phase hint("clip 3 of 5, 60% through")
4. Structured output rules(30 words以内,egocentric directional language)

**Intent sanitization**:去掉action label里全是consonant的trailing tokens——很pragmatic的data cleaning

### 9.2 Baseline adaptation(Appendix D)
Cosmos-Predict和WoW是single-turn的,需要adapt到multi-turn:
- 90% clips conditioned on preceding chunk's latent(multi-turn continuation)
- 10% conditioned on first-frame(trajectory start)
- 这个90/10 mixture mirrors evaluation rollout

WoW的chunk k=0用repeated-frame conditioning(第一帧tile 41次),chunk k>0用tail 41 frames。Generate 82 frames internally,discard 41 conditioning frames,subsample到37 output frames。

Cosmos用image2world(k=0)和video2world(k>0,last 10 latent frames)。

**所有模型**:37 frames/chunk, 480×480, 16 FPS, 35 diffusion steps。Guidance scale 5 (Cosmos) / 7 (WoW)。

### 9.3 Training setup
- 16×A100 80GB
- LR 1e-5
- 4 epochs
- EMA decay 0.99
- Full fine-tuning(LoRA underperform,因为large domain gap from internet-video pretraining to egocentric manipulation)

## 10. 总结与position in field

这篇paper的核心intellectual contribution是**把world-ego这个模糊概念operationalize**——给出3个boundary定义 × 3个disentanglement策略的设计空间,systematic study哪个组合最优。

它属于一个emerging trend: **structured inductive bias for video world models**。前面有:
- VideoREPA (NeurIPS 2025, https://arxiv.org/abs/2510.02503): relational alignment with foundation models
- Tesseract (ICCV 2025): 4D embodied world models
- Longscape (https://arxiv.org/abs/2509.21790): context-aware MoE for long-horizon

WEM的独特angle:**interaction-dependent boundary**——world-ego line随task progress移动。这个dynamic boundary idea我觉得是paper最深刻的insight,值得future work深挖。

对embodied AI field的影响:这paper证明**explicit decomposition beats monolithic generation** for long-horizon composite tasks。这suggests下一代embodied world model应该有modular structure,一个planner + 多个specialized predictors。和Modular Policies (MOKA, https://arxiv.org/abs/2405.08192)在policy端的trend呼应。

对video generation field的影响:WEM的CP-MoE可以看作**structured MoE for video diffusion**——和CogVideoX的expert transformer (https://arxiv.org/abs/2408.06072)、HunyuanVideo的双stream (https://arxiv.org/abs/2412.03603)同一线,但用role-based routing而非learned routing。这暗示**semantic prior to routing helps** when we have structured task。

对Karpathy自己的研究interest:这paper和他在micrograd/eureka-labs强调的"build intuition first"哲学一致——paper先build world-ego conceptual taxonomy,再instantiate。这种concept-first的方法在RL/embodied AI里undervalued,而这paper显示它yield principled design decisions。

## Reference Links

- **Paper**: https://zgca-hmi-lab.github.io/WEM
- **Code**: https://github.com/ZGCA-HMI-Lab/WEM
- **Model weights**: https://huggingface.co/Zoorao/WEM
- **Dataset**: https://huggingface.co/datasets/Zoorao/HTEWorld
- **Wan video model**: https://arxiv.org/abs/2503.20314
- **Qwen3-VL**: https://arxiv.org/abs/2511.21631
- **Cosmos-Predict 2.5**: https://arxiv.org/abs/2511.00062
- **WoW**: https://arxiv.org/abs/2509.22642
- **PAN**: https://arxiv.org/abs/2511.09057
- **WorldArena**: https://arxiv.org/abs/2602.08971
- **BEHAVIOR-1K**: https://arxiv.org/abs/2210.04426
- **V-JEPA 2**: https://arxiv.org/abs/2506.09985
- **GEM**: https://arxiv.org/abs/2505.14698
- **Iso-Dream**: https://arxiv.org/abs/2205.13521
- **CogVideoX**: https://arxiv.org/abs/2408.06072
- **HunyuanVideo**: https://arxiv.org/abs/2412.03603
- **Flow Matching**: https://arxiv.org/abs/2210.02747
- **DPT**: https://arxiv.org/abs/2104.06254
- **Dice Loss**: https://arxiv.org/abs/1911.02855
- **RAFT**: https://arxiv.org/abs/2003.12039
- **LPIPS**: https://arxiv.org/abs/1801.03924
- **Switch Transformer**: https://arxiv.org/abs/2101.03961
- **Mixtral**: https://arxiv.org/abs/2401.04088

如果想deep dive到具体某个module的代码实现,我可以继续展开(比如RCA的attention mask具体怎么写、CP-MoE的routing tensor操作、DPT semantic head的feature fusion细节)。
