---
source_pdf: EgoSteer A Full-Stack System Towards Steerable Dexterous Manipulation
  from Egocentric Videos.pdf
paper_sha256: 1818bc3b5f5a0d41d44c2e33a5b71f9255a68b7b395ba06895e3e11477cb7dfe
processed_at: '2026-08-18T10:20:26-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版

Andrej, 我之前讲太dense了,让我重新用"人话"聊聊这个工作。

## 一句话版

这帮人把"从YouTube式第一人称视频里学人类操作"这件事从头到尾做了一遍——data清洗、机器人遥操、模型训练、human-in-the-loop修正,全栈打通,在40个真实灵巧操作任务上跑出75%平均成功率。

## 他们到底在解决什么问题

机器人圈现在有个尴尬:你想让机器手跟人一样灵活,但灵巧手的real-robot data极难collect。人类每天有116K小时的第一人称操作视频自然生成,但raw视频是垃圾——camera晃、occlusion、没标注、一堆走路无关镜头。

所以核心问题是: **怎么把这些"野生的"人类操作视频,变成能训练robot policy的高质量数据,然后真正迁移到物理机器人上?**

之前的工作要么只做data curation,要么只做model architecture,要么只做deployment,这帮人意识到——**任何一个环节掉链子都run不起来**。

## 三个组件,各管一段

### 1. EgoSmith: 把垃圾视频洗成训练数据

核心story: raw egocentric video → 9.6K小时的fully-annotated训练数据。

他们做的事其实很工程化,但每一步都有intuition:

**Pre-filtering**: 用optical flow检测locomotion(走路的frame会有large global translation,操作时camera相对稳),用YOLO检测手(只保留画面下半部分、面积在2%-50%之间、至少两只手都可见的frame)。简单heuristics但effective。

**4D motion estimation** (这是核心改进): 之前的SOTA HaWoR用DROID-SLAM做camera tracking,慢且在fast head movement下drift。他们换成DPVO(更轻量更稳),然后用Any4D的metric depth去recover物理scale。具体就是用median ratio对齐两个depth的scale。

直觉: DPVO告诉你"camera轨迹的shape",Any4D告诉你"每一帧的物理深度",两者对齐就得到了metric-scale的world-space camera trajectory。然后再把hand reconstruction贴到这个世界坐标系里。

Throughput 9×于HaWoR,精度还更好(RPE降了一半)。

**Language labeling**: 用Qwen3.5-VL生成5个level的annotation,从"Open the drawer"(5词)到step-by-step物理描述(100词)。有意思的是他们强制verb-first、禁止"then/next"这种transitional word——让language更像"executable instruction"而不是natural description。

**Post-filtering**: 三层质量把控,episode/chunk/frame级别。chunk级别有个不错的intuition——先canonicalize到wrist frame再比坐标,避免confound with body movement。

### 2. Unified Robot Stack: 让human能smoothly接管机器人

这部分其实有个我挺喜欢的mathematical insight。

Human-in-the-loop correction最大的痛点是: 当人踩foot pedal要接管时,机器人在某个pose,人的手在另一个pose,直接映射会有state jump。

他们的solution是**relative motion mapping**:

$$\widetilde{\mathbf{T}}_{t'}^{\text{R}} = \mathbf{T}_t^{\text{R}} \cdot (\mathbf{T}_t^{\text{H}})^{-1} \cdot \mathbf{T}_{t'}^{\text{H}}$$

人话: 算人的relative motion $(\mathbf{T}_t^{\text{H}})^{-1}\mathbf{T}_{t'}^{\text{H}}$(frame-invariant),然后compose到robot当前state上。Operator只需要mimic机器人当前的运动方向,不需要精确match绝对pose。

这操作85%的handover成功率,听起来简单但engineering上很clean。

Hardware方面用了RealMan双臂+Ruiyan RY-H2六自由度手,RealSense D455 head+chest双视角。控制频率100Hz(wrist/arm)和80Hz(glove/hand)。

### 3. EgoSteer: model本身

这部分几个有意思的design choice:

**Unified action space**: 48维 = 每只手24维(3D wrist translation + 6D rotation + 15D fingertip keypoints)×2。关键是fingertip keypoints——human hand和robot hand的joint configuration不同,但fingertip positions在task space里是equivalent的,这就enable了cross-embodiment transfer。

**World model expert** (这是我最喜欢的trick): 

直觉解释——VLA模型缺乏"future imagination",这限制action generation精度。但这帮人没有deploy一个完整的world model(贵),而是train了一个**training-only**的小module(70M params,4层Transformer),让它predict action-induced future **DINOv3 features**。

为什么不predict pixels? DINOv3 features已经abstract掉low-level details,encode semantic content,gradient更stable。为什么不keep at inference? 因为它的真正作用是**shape backbone representation**——gradient通过KV cache projection回传到backbone,让backbone更action-aware。Inference时直接discard,zero overhead。

这有点像JEPA的哲学——在latent space里predict future,avoid pixel-level reconstruction的负担。但EgoSteer的twist是把它当成representation shaping tool,不是deploy-time predictor。

**Training-time RTC**: 从π0.5借来的trick。把action chunk切成prefix(已执行,fill inference latency)和suffix(要denoise的新actions)。部署时保留32-step chunk的前12步,减4个prefix steps,实际执行8 new steps per inference cycle。实现了seamless的closed-loop control,没有execution pause。

**Conditional Flow Matching**: standard OT path的flow matching,timestep η用Beta(1.5, 1)采样偏向high noise levels。每个sample采4个random η expand effective batch size。

## 实验结果其实挺convincing

**40 tasks, 75%平均成功率**: 覆盖PnP-easy/medium/hard、non-prehensile、reorient、bimanual、contact-rich七类。22个task达到80%+。Compositional generalization 65%,unseen tasks 62%。

**DAgger的sample efficiency真的惊人**:
- EgoSteer-FT (只用teleop data): 22.5%
- EgoSteer-DG (3轮DAgger, 8.3 hours correction data): 62.5%

40个百分点的提升只用8.3小时correction data。Intuition是pre-trained model已经有good priors,DAgger精准target deployment failures,这些"hard cases"正是model需要improve的地方。

**Pre-training scaling**: scratch / 3K / 6K / 9.6K → 30% / 40% / 43% / 60%。随着scale增长,观察到failure recovery、instruction-following、action accuracy的**emergent capabilities**。这跟LLM的emergent abilities现象类似。

**vs baselines**: π0.5 22%, Being-H0.5 39%, EgoSteer 74%。但要注意baseline没用EgoSteer的unified action space,所以不完全fair comparison。

**Few-shot long-horizon**: Box-folding (18步,40秒) 75%, Cake-unboxing (9步,1分钟) 83%。DP、IMLE、from-scratch全部0% fail。这pre-trained priors的价值在long-horizon contact-rich task上尤其明显。

## 我的real take

### 真正的contribution

1. **Full-stack co-design**: 这paper的central insight是steerable dexterous manipulation需要data + system + model co-design。这个trend会越来越主流——从single-paper innovation转向integrated systems。

2. **Data quality > data scale**: 9.6K curated hours > 100K raw hours。EgoSmith的pipeline工程量大但worth it。

3. **Training-only auxiliary module**: world model expert这个设计很聪明,zero inference overhead,纯representation shaping。我觉得这个pattern会被更多work借鉴。

4. **Relative motion mapping**: clean的mathematical structure,$SE(3)$上frame-invariant的operator。这种工程细节常被忽略但实际deploy很关键。

### 我觉得的weaknesses

1. **6-DoF robot hand vs 21-DoF human hand**: 这是fundamental bottleneck。很多human manipulation primitives在kinematic层面就丢失了。Future work需要更高DoF的hand。

2. **No tactile feedback**: contact-rich tasks仅靠vision是suboptimal的。Wipe whiteboard、fold box这种需要force-controlled contact。

3. **Pre-training scale**: 9.6K vs 116K available。他们自己admit这个,需要更大scale fully validate scaling law。

4. **Baseline comparison不完全fair**: π0.5和Being-H0.5没用EgoSmith的unified action space,所以numbers comparison有caveat。

### 一些speculation

基于这个work,我觉得几个direction会火:

1. **Tactile-conditioned policies**: 加BioTac/DIGIT sensors,让policy condition on tactile feedback
2. **Higher-DoF hands**: LEAP Hand 21-DoF或Shadow Hand 24-DoF,unlock更多human dexterity
3. **Audio-visual policies**: sound对contact detection很有用,人其实很依赖audio feedback
4. **100K+ hour pre-training**: scale up到Ego4D全规模
5. **Latent world models + planning**: 把training-only的world model extend成deploy-time的planner
6. **Cross-embodiment generalization**: 用更多种hand design验证unified action space

## 一句话总结

这是个工程量极大的full-stack工作,展示了从wild egocentric videos到real-robot steerable dexterous manipulation的完整pipeline。核心insight是data quality、world model作为representation shaper、relative motion mapping、DAgger sample efficiency——任何一个环节掉链子都run不起来。我认为这种co-design philosophy是robotics research的未来。

Hope this version更readable,Andrej! 想deep dive哪个部分告诉我。

---

# EgoSteer: 全栈式可操纵灵巧操作系统深度解析

Andrej, 这是一篇非常dense的full-stack paper,从data curation到model architecture到real-robot deployment都覆盖了。我来一层一层剥开讲,重点build你的intuition。

## 1. 整体动机与Problem Statement

paper的核心claim: **steerability**是generalist robot policy的defining capability,但在dexterous-hand系统上still largely absent。bottleneck在data和system scalability。

直接在dexterous robot上collect大规模demonstration data非常expensive,特别是针对特定embodiment。egocentric human videos是一个scalable alternative,因为human hand manipulation包含rich interaction knowledge,且naturally generated at massive scale。但raw videos是noisy的,lack reliable language和action annotations。

paper的贡献是close this gap,通过三个co-design的component:
- **EgoSmith**: 数据curation pipeline
- **Unified Robot Stack**: teleoperation + DAgger correction
- **EgoSteer**: world-model-enhanced VLA

Project page: https://egosteer.github.io/

## 2. EgoSmith: 数据Pipeline详解

这是paper最engineering-heavy的部分。从12个egocentric datasets curate出9.6K hours的高质量annotated data。整体throughput是prior SOTA HaWoR的9×。

### 2.1 Stage 1: Pre-filtering

两个specialized gates:

**Camera gate** (检测locomotion):
- 在每frame上建立128-point grid
- 用pyramidal Lucas-Kanade (Bouguet et al., [59]) track back 15 frames
- 用RANSAC (Hartley & Zisserman, [60]) fit一个similarity transform到correspondences
- 如果translation > 10% image's longer dimension,则frame无效
- intuition: locomotion会induce large global translation,manipulation时camera相对stable

**Hand gate** (检测visibility anomalies):
- 用YOLOv3 [44] / WiLoR [45]检测hands
- valid bounding box需满足:
  - confidence ≥ 0.30 (reject false positives)
  - area ∈ [2%, 50%] of image
    - 上限50%排除abnormally close hands
    - 下限2%基于Egocentric-10K/100K manual labeling校准,filter out别人的手
  - 与lower-central region [0.075, 0.925] × [0.075, 1.0]相交
  - 每frame至少2个valid detections (preserve bimanual manipulation)

连续3帧无效才prune segment,isolated failures保留。这是engineering wisdom: 一次YOLO failure不应该废弃整段video。

### 2.2 Stage 2: 4D Motion Estimation (核心创新)

这是computational bottleneck,也是EgoSmith相对HaWoR的主要改进。

**原HaWoR的问题**:
- 依赖DROID-SLAM [46]做camera tracking
- DROID-SLAM是dense SLAM,computationally expensive
- 在rapid head movements和textureless环境下有drift

**EgoSmith的改进**:
1. 用**DPVO** [47]替换DROID-SLAM
   - 更robust in long-range egocentric scenarios
   - much lower computational cost
   - 输出up-to-scale camera poses $\hat{\mathbf{T}}_t = (\mathbf{R}_t, \hat{\mathbf{p}}_t) \in SE(3)$ + focal length
   - 注意hat notation表示up-to-scale

2. 用**Any4D** [48]做frame-wise metric-scale depth prediction
   - 对local windows做cross-chunk alignment,得到temporally consistent的metric depth sequence

3. **Scale recovery** via median ratio:

$$s = \text{median}_{t, (u,v) \in \mathcal{B}_t} \frac{\mathbf{D}_t^{\text{Any4D}}(u,v)}{\hat{\mathbf{D}}_t^{\text{DPVO}}(u,v)}$$

变量解释:
- $s$: 全局scale factor,把DPVO的up-to-scale trajectory anchor到physical scale
- $\mathbf{D}_t^{\text{Any4D}}(u,v)$: Any4D预测的metric depth在pixel $(u,v)$
- $\hat{\mathbf{D}}_t^{\text{DPVO}}(u,v)$: DPVO估计的up-to-scale depth
- $\mathcal{B}_t$: valid background region,通过排除hand region(从reconstructed 3D hand mesh project回去)得到

intuition: 用median而不是mean是为了robustness against outliers。background pixels的depth ratio应该是consistent的,hand region的depth则不可靠。

4. **Metric trajectory**: $\mathbf{p}_t = s\hat{\mathbf{p}}_t$

5. **Camera-to-world transform**:

$$\mathbf{x}_t^{\text{world}} = \mathbf{R}_t^\top(\mathbf{x}_t^{\text{cam}} - \mathbf{p}_t)$$

变量:
- $\mathbf{x}_t^{\text{cam}} \in \mathbb{R}^3$: camera-space 3D coordinate (hand vertices/joints)
- $\mathbf{R}_t \in SO(3)$: camera rotation
- $\mathbf{p}_t \in \mathbb{R}^3$: metric camera translation
- $\mathbf{x}_t^{\text{world}}$: world-space coordinate

**Hand reconstruction**: 用HaWoR的ViT module作为off-the-shelf camera-frame hand reconstructor,regress MANO [61] parameters:
- pose parameters $\pmb{\theta}_t \in \mathbb{R}^{51}$
- shape parameters $\beta_t \in \mathbb{R}^{10}$
- camera-relative root translation $\mathbf{t}_t \in \mathbb{R}^3$

**Efficiency optimizations**:
- Parallelized batching: 把多个16-frame windows group到一个batch
- Asynchronous I/O pipelining: overlap CPU frame decoding/cropping with GPU inference
- Result: 8×A800上 ~9× speedup over HaWoR

**Accuracy benchmark** (Table 2, on TACO, H2O, OakInk-v2, Ego-Verse subsets):

| Method | RPE↓ | ATE↓ | WA-MPJPE↓ | W-MPJPE↓ |
|--------|------|------|-----------|----------|
| HaWoR  | 5.17 | 9.44 | 38.7      | 106.9    |
| EgoSmith | 2.42 | 7.60 | 25.9      | 86.0     |

(all in mm)

Metrics intuition:
- **RPE** (Relative Pose Error): local tracking drift和frame-to-frame jitter,no global alignment。sensitive to metric-scale inaccuracies
- **ATE** (Absolute Trajectory Error): overall trajectory shape + long-term drift,aligned via global Sim(3)。insensitive to absolute scale
- **WA-MPJPE** (World-Aligned MPJPE): hand joint errors with per-100-frame Sim(3) alignment
- **W-MPJPE** (World MPJPE): strictest metric,只align first frame via SE(3),penalize scale errors + temporal drift + orientation misalignment

EgoSmith在所有4个metrics上都better,RPE降了50%+,说明DPVO-based trajectory有superior local consistency。

参考链接:
- DPVO: https://github.com/princeton-vl/DPVO  
- DROID-SLAM: https://github.com/princeton-vl/DROID-SLAM
- Any4D: https://any4d.github.io/
- HaWoR: https://github.com/haowaru/hawor

### 2.3 Stage 3: Language Labeling

用**Qwen3.5-VL-Plus** [49]做multi-granularity annotation。先filter掉3.5%没有meaningful hand-object manipulation的clips,然后对剩余的generate 5-level语言annotations:

- **Level 1** (Verb + Object): max 5 words, e.g., "Open the drawer."
- **Level 2** (Gist): max 15 words,concise summary
- **Level 3** (Object-Centric): max 30 words,描述manipulated object的parts, state, spatial features
- **Level 4** (Hand-Centric): max 50 words,specify left/right hand roles, grasp style, contact points
- **Level 5** (Dense Sequence): max 100 words,step-by-step physical breakdown with spatial anchors和motion trajectories

这种hierarchical annotation同时提供task-level semantic grounding和action-level spatiotemporal grounding,让downstream model可以respond to instructions across varying levels of abstraction。

paper的prompt template有几个notable的设计:
- **Verb-first imperative**: 一定要start with action verb,NO subjects
- **Definite object references**: 用"the"而不是"a/an"
- **No transitional words**: 禁止"then", "next", "afterwards"
- **Action Precision**: 用specific physical verbs (Grip, Lift, Slide, Scoop, Peel, etc.)

这些constraint的intuition: 让language description更接近"executable instruction"的form,而不是natural language description。这对policy learning更friendly。

Qwen3.5 blog: https://qwen.ai/blog?id=qwen3.5

### 2.4 Stage 4: Post-filtering (Quality Control)

从coarse到fine三层:

**Episode level**:
- 计算episode的camera translation/rotation statistics
- 与同dataset其他episode比较,用dataset-specific IQR criterion
- Outlier定义: statistics在$[Q_1 - 2.5\text{IQR}, Q_3 + 2.5\text{IQR}]$之外

intuition: 不同dataset有不同device/scene/manipulation style,所以用dataset-specific distribution而不是universal threshold。

**Chunk level**:
- 把过去5秒+未来30帧的sliding window的hand states/actions transform到current camera frame
- 在这个canonical system里:wrist positions相对camera,finger joints相对wrist
- 用同样的IQR criterion检测outliers
- **Universal physical ceiling**: 每个coordinate axis 1.5米(human hand不可能reach更远)

intuition: 直接比较绝对hand coordinates会有confounding with camera/body movement,所以先canonicalize到wrist frame。

**Frame level**:
- 检测frame-to-frame jumps
- 用fixed physical thresholds (因为human hand/head speed有universal limit):
  - camera translation ≤ 0.20 m/frame
  - wrist and finger translation ≤ 0.30 m/frame
  - camera rotation ≤ 28°/frame
  - wrist rotation ≤ 41°/frame

任何frame violation就discard整个episode。

### 2.5 Curated Dataset Statistics

最终corpus:
- **9.60K hours**
- **2.09M episodes**
- **1.04B frames**
- 来自12 datasets

主要来源:
- Egocentric-100K: 8,049 hours (83.8%)
- EgoVerse: 690 hours (7.2%)
- EgoDex: 370 hours (3.9%)
- Egocentric-10K: 288 hours (3.0%)
- Ego4D: 138 hours (1.4%)
- Epic-Kitchens: 49 hours (0.5%)
- 其他6个dataset共~22 hours

semantic diversity:
- 8,969 distinct object nouns
- 623 action verbs
- Long-tailed分布

Ego4D: https://ego4d-data.org/  
EPIC-KITCHENS: https://epic-kitchens.github.io/  
Egocentric-100K: https://huggingface.co/datasets/builddotai/Egocentric-100K

## 3. Unified Robot Stack

### 3.1 Hardware Setup

两个embodiment:

**Primary: RealMan**
- 2× RealMan RM75-6F 7-DoF robotic arms
- 2× Ruiyan RY-H2 6-DoF dexterous hands
- 1× head-mounted + 1× chest-mounted Intel RealSense D455
- 提供dual egocentric viewpoints

**Secondary: AgiBot G1**
- Adapted from AgiBot G1 humanoid
- Default end-effectors替换为2× Ruiyan RY-H2 hands
- 1× head-mounted D455 + 2× wrist cameras (unused)
- Neck/waist/mobile base固定

控制频率:
- wrist tracker, arm kinematic solvers, joint control ROS 2 nodes: 100 Hz
- glove, hand solvers, control ROS 2 nodes: 80 Hz
- Cameras: 30 Hz
- Data最终resample到30 Hz for training

这种high-frequency loop是关键,enable intuitive bimanual teleoperation和fine-grained manipulation data collection。

### 3.2 Teleoperation

- **PsiBot SynGlove-Air gloves** + **Vive Trackers**捕获operator的:
  - $SE(3)$ wrist poses
  - hand joint angles
- 用**mink** [53] (MuJoCo-based IK) compute inverse kinematics驱动arms
- Joint mapping驱动6-DoF hands

mink: https://github.com/kevinzakka/mink

### 3.3 DAgger Relative Motion Mapping (核心创新)

这是paper的key engineering contribution之一。Human-in-the-loop intervention的最大challenge是**防止handover boundary的sudden state jumps**。

**Setup**: 当operator在step $t$踩foot pedal请求intervention,系统记录(对于每个arm/hand $i \in \{1, 2\}$):
- Robot end-effector poses: $\mathbf{T}_t^{\text{R},i} \in SE(3)$
- Human wrist poses: $\hat{\mathbf{T}}_t^{\text{H},i} \in SE(3)$
- Robot hand joint states: $\mathbf{q}_t^{\text{R},i} \in \mathbb{R}^6$
- Glove states: $\mathbf{q}_t^{\text{H},i} \in \mathbb{R}^6$

(变量中$\mathbf{T}$是$SE(3)$ rigid transform,$\mathbf{q}$是6D joint angles;$R$表示robot,$H$表示human)

**Relative motion计算** (在任意$t' \geq t$):

$$\Delta\mathbf{T}_{t \to t'}^{\text{H},i} = (\mathbf{T}_t^{\text{H},i})^{-1}\mathbf{T}_{t'}^{\text{H},i}$$

$$\Delta\mathbf{q}_{t \to t'}^{\text{H},i} = \mathbf{q}_{t'}^{\text{H},i} - \mathbf{q}_t^{\text{H},i}$$

**Mapping到robot**:

$$\widetilde{\mathbf{T}}_{t'}^{\text{R},i} = \mathbf{T}_t^{\text{R},i}\Delta\mathbf{T}_{t \to t'}^{\text{H},i}$$

$$\mathbf{q}_{t'}^{\text{R},i} = \mathbf{q}_t^{\text{R},i} + \Delta\mathbf{q}_{t \to t'}^{\text{H},i}$$

intuition (这是关键!):
- 想象operator看到robot处于某个pose $\mathbf{T}_t^{\text{R},i}$
- 如果直接用human的absolute pose $\mathbf{T}_{t'}^{\text{H},i}$映射到robot,会有mismatch
- 但如果operator做的是**relative motion** $\Delta\mathbf{T}$,这个relative motion是在human frame里计算的
- 然后compose到robot state:$\mathbf{T}_t^{\text{R},i} \cdot \Delta\mathbf{T}_{t \to t'}^{\text{H},i}$
- 这相当于:robot从自己的当前state出发,执行了human的"motion delta"
- operator只需mimic robot的motion方向,不需要精确match绝对pose

数学上:$(\mathbf{T}_t^{\text{H},i})^{-1}\mathbf{T}_{t'}^{\text{H},i}$计算的是human wrist从$t$到$t'$的relative transform,与human wrist在$t$时刻的绝对pose无关。这个relative transform compose到robot state上,自然avoid了human-robot pose mismatch。

**Result**: 85% handover success rate。Corrections的segment被collect用于DAgger training。

### 3.4 Teleoperation Dataset

- **187 hours**
- **193 semantically-diverse tasks**
- **55K trajectories**
- 每task约300 randomized trajectories

任务分类:
- **Common Tasks** (56): everyday manipulations,高teleoperation success rate
- **Long-Tail Tasks** (137): infrequent, contact-sensitive,低success rate,但保证semantic coverage

按motion characteristics分7类:
1. **PnP-Easy**: single-step tabletop pick-and-place
2. **PnP-Medium**: non-planar/3D spatial PnP with containers
3. **PnP-Hard**: multi-step or high-precision PnP, e.g., "stack paper cups"
4. **Non-prehensile**: pushing, pulling, pressing
5. **Reorient**: rotation/reorientation, e.g., "pour water"
6. **Bimanual**: high synchronization required, e.g., "plug cable into charger"
7. **Contact-rich**: frequent complex contact, e.g., "wipe whiteboard"

数据collection protocol的key design:
- **Cluttered scenarios**: 避免model从visual inputs alone识别task
- **Randomized configurations**: tablecloths, object instances, initial configurations都randomized
- **Natural, human-like execution**: 不同demonstrations有substantial trajectory variations

intuition: 最大化覆盖human-to-robot action space mapping的distribution,而不是memorize fixed trajectories。

## 4. EgoSteer模型架构

### 4.1 整体架构

三大模块:
1. **Qwen3-VL 2B backbone** [54]: vision-language understanding
2. **DiT-based action expert** (~300M params): action chunk generation via flow-matching
3. **World model expert** (~70M params, training-only): predict future DINOv3 features

### 4.2 Unified Action Space (48D)

定义episode $\tau = \{l, \mathbf{K}, (\mathbf{I}_t, \mathbf{D}_t, \mathbf{T}_t^{w2c}, \mathbf{s}_t^w, \mathbf{a}_t^w)_{t=0}^{N-1}\}$

变量:
- $l$: language instruction
- $\mathbf{K} \in \mathbb{R}^{3\times3}$: camera intrinsics
- $\mathbf{I}_t \in \mathbb{R}^{H\times W\times 3}$: RGB image
- $\mathbf{D}_t \in \mathbb{R}^{H\times W\times 1}$: depth image (model不用)
- $\mathbf{T}_t^{w2c} \in SE(3)$: world-to-camera extrinsics
- $\mathbf{s}_t^w, \mathbf{a}_t^w \in \mathbb{R}^{48}$: bimanual world-frame states and actions

**48维breakdown** (per hand 24D, 两个手共48D):
- 3D wrist translation: 3维
- 6D wrist rotation (Zhou et al. continuous representation): 6维
- 15D fingertip keypoints: 5 fingers × 3D position = 15维
- 总计 per hand: 3 + 6 + 15 = 24维

intuition on **why fingertip keypoints**:
- Human hand和robot hand的kinematic structure不同
- 但fingertip positions在task space里是equivalent的
- 这是**cross-embodiment transferable**的representation
- 与human hand通过15D keypoints自然aligned

参考: Zhou et al.的6D rotation representation (CVPR 2019): https://arxiv.org/abs/1812.07035

### 4.3 Training-time RTC (Real-Time Chunking)

这是从π0.5 [24]借鉴的engineering trick,解决chunk-based policy的execution pause问题。

**问题**: 传统chunk-based policy每$h$步预测一个action chunk,执行完才infer下一个chunk。这导致execution pauses。

**EgoSteer的solution**:
- 把action chunk分成prefix和suffix
- $\mathbf{a}_{\text{pre}} = \mathbf{a}_{t:t+d-1}$: clean prefix,长度$d$(randomly sampled delay)
- $\tilde{\mathbf{a}}_{\text{suf}} = \tilde{\mathbf{a}}_{t+d:t+h-1}$: noisy suffix,model denoise这个

变量:
- $h = 32$: chunk length
- $d$: simulated delay
- 训练时: $d \sim \mathcal{U}([0, 5])$ (post-training),$d = 0$ (pre-training, max supervision)
- 部署时: $d = 4$ (cover physical inference latency)

**Deployment mechanism**:
1. Robot执行reserved prefix $\mathbf{a}_{\text{pre}}$ during asynchronous VLA inference
2. Inference完成后,seamlessly transition到new chunk $\mathbf{a}_{\text{suf}}$
3. 保留32-step chunk的前12步,减4 prefix steps,实际执行8 new action steps per inference cycle
4. 这就实现了high-frequency, asynchronous closed-loop control

intuition: 训练时让model学习"给定prefix,预测suffix",部署时prefix填满inference latency gap,suffix是new action steps。No execution gap。

### 4.4 Conditional Flow Matching训练

模型$\pi$在context $\mathbf{C}_t = \{l, \mathbf{K}, \mathbf{I}_{t-k+1:t}, \mathbf{s}_{t-k+1:t}, \mathbf{a}_{\text{pre}}\}$下,通过CFM [56]回归target suffix $\mathbf{a}_{\text{suf}}$的linear velocity field:

$$\mathcal{L}_{\text{CFM}}(\pi) = \mathbb{E}_{t, \eta, \epsilon}\left[\|\pi(\tilde{\mathbf{a}}_{\text{suf}}, \eta, \mathbf{C}_t) - (\mathbf{a}_{\text{suf}} - \epsilon)\|^2\right]$$

变量详解:
- $t$: 当前timestep
- $\eta \in [0, 1]$: flow matching timestep (noise level)
- $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: standard Gaussian noise
- $\tilde{\mathbf{a}}_{\text{suf}} = (1-\eta)\epsilon + \eta\mathbf{a}_{\text{suf}}$: interpolated noisy action
- $\mathbf{a}_{\text{suf}} - \epsilon$: target velocity field (linear interpolation path的velocity)
- $\pi(\tilde{\mathbf{a}}_{\text{suf}}, \eta, \mathbf{C}_t)$: model预测的velocity

intuition: 这是Optimal Transport (OT) path的flow matching。Interpolation是linear的:$\tilde{\mathbf{a}}_{\text{suf}}$在$\eta=0$时是pure noise $\epsilon$,在$\eta=1$时是clean action $\mathbf{a}_{\text{suf}}$。Velocity field是constant $(\mathbf{a}_{\text{suf}} - \epsilon)$,与$\eta$无关。Model学习predict这个constant velocity,然后通过ODE integration从noise走到clean action。

**Timestep $\eta$ sampling**:

$$P(\eta) = \text{Beta}\left(\frac{s-\eta}{s}; 1.5, 1\right), s = 0.999$$

让我分析一下: 令$u = (s-\eta)/s$,则$\eta = s(1-u)$,Jacobian $|du/d\eta| = 1/s$。

$P(\eta) = \text{Beta}(u; 1.5, 1) \cdot 1/s = \frac{1.5 u^{0.5}}{s}$

- 当$u = 1$ ($\eta = 0$,high noise): density = $1.5/s$ (max)
- 当$u = 0$ ($\eta = s$,low noise): density = 0 (min)

所以model spend more capacity on high-noise levels,与π0 paper一致。

每个sample采4个random $\eta$来expand effective batch size和improve loss gradient。

### 4.5 World Model Expert (训练时辅助)

VLA的limitation是缺乏"future imagination",这限制action generation accuracy。EgoSmith引入world model expert来predict action-induced future DINOv3 features [21]。

**Architecture**:
- 4-layer Transformer (identical to Qwen3's text layer)
- Hidden dim: 1024
- Intermediate size: 4096
- 8 attention heads, head dim $d_{\text{head}} = 128$
- ~70M params

**Inputs**:
- Ground-truth $\mathbf{a}_{t:t+h-1}$
- Relative camera motion $\Delta\mathbf{T} = \mathbf{T}_t^{w2c}(\mathbf{T}_{t+h-1}^{w2c})^{-1}$,flatten成16D vector,通过2-layer MLP编码成single continuous token
- Learnable query tokens $\mathbf{z}_{0:L_z-1}$,长度$L_z$

**DINOv3 feature extraction**:
- DINOv3 ViT-L/16 [21]
- Input resolution: 384×384
- Spatial resolution: $H_v = W_v = 24$
- Feature dim: $C_{\text{DINO}} = 1024$
- Token merge后: $L_z = H_v' \times W_v' = (H_v/2) \times (W_v/2) = 12 \times 12 = 144$

**Output**: $\hat{\mathbf{Y}} \in \mathbb{R}^{H_v' \times W_v' \times d_{\text{WM}}}$,其中$d_{\text{WM}} = 1024$
然后通过2×2 linear upsampling projection layer恢复到$\hat{\mathbf{Z}} \in \mathbb{R}^{H_v \times W_v \times C_{\text{DINO}}}$

**World model loss** (MSE):

$$\mathcal{L}_{\text{WM}} = \frac{1}{H_v \cdot W_v}\sum_{u=1}^{H_v}\sum_{v=1}^{W_v}\|\mathbf{Z}_{u,v} - \hat{\mathbf{Z}}_{u,v}\|_2^2$$

变量:
- $\mathbf{Z}_{u,v} \in \mathbb{R}^{C_{\text{DINO}}}$: ground-truth DINOv3 feature at spatial location $(u,v)$
- $\hat{\mathbf{Z}}_{u,v}$: reconstructed feature
- $H_v = W_v = 24$: spatial resolution

**Why DINOv3 features而不是pixels?**
1. Semantic abstraction: filter out lighting variations和background noise
2. Stable gradient guidance vs generative loss in pixel space
3. 与backbone representation自然aligned
4. Training-only,inference时discard,zero overhead

**Joint attention机制**: World model expert在layer $\ell$ attends to:
- 自己的sequence
- VLM backbone的KV cache from layer $f(\ell) = 7\ell$ (after linear projection)

intuition: 这与世界模型类似LeCun的JEPA philosophy——predict latent representation of future state,而不是pixel-level reconstruction。这avoid了generative model的high-frequency detail burden,而focus on semantic understanding。

但EgoSteer的twist是: world model expert是training-only auxiliary module。它的gradient通过backbone的KV cache projection matrix $\mathbf{W}_\ell^K, \mathbf{W}_\ell^V$回传到backbone,shape backbone representation使其更action-aware。

参考:
- DINOv3: https://arxiv.org/abs/2508.10104
- JEPA: https://openreview.net/forum?id=6Y24HvJdtfP

### 4.6 Action Expert (DiT-based)

**Architecture**:
- 14 layers
- Hidden dim: 1024
- Intermediate size: 2816
- 8 attention heads, head dim $d_{\text{head}} = 128$
- ~300M params

**Action chunk**: length $h = 32$ at 30Hz

**Positional encoding**: Interleaved MRoPE (与VLM backbone一致)

**Joint attention公式** (Eq. 2, layer $\ell$, head $m$):

$$\text{Softmax}\left(\frac{1}{\sqrt{d_{\text{head}}}}\mathbf{Q}_{\ell,m}^{\text{AE}}\left(\text{concat}[\mathbf{K}_{f(\ell),m}^{\text{B}}\mathbf{W}_\ell^{\text{K}}, \mathbf{K}_{\ell,m}^{\text{AE}}]\right)^\top\right)\text{concat}[\mathbf{V}_{f(\ell),m}^{\text{B}}\mathbf{W}_\ell^{\text{V}}, \mathbf{V}_{\ell,m}^{\text{AE}}]$$

变量详解:
- $\mathbf{Q}_{\ell,m}^{\text{AE}}, \mathbf{K}_{\ell,m}^{\text{AE}}, \mathbf{V}_{\ell,m}^{\text{AE}} \in \mathbb{R}^{h \times d_{\text{head}}}$: action expert的query, key, value at layer $\ell$, head $m$
- $\mathbf{K}_{f(\ell),m}^{\text{B}}, \mathbf{V}_{f(\ell),m}^{\text{B}} \in \mathbb{R}^{N_B \times d_{\text{head}}}$: backbone的key, value at layer $f(\ell) = 2\ell$, head $m$
- $N_B$: backbone的input sequence length
- $\mathbf{W}_\ell^{\text{K}}, \mathbf{W}_\ell^{\text{V}} \in \mathbb{R}^{d_{\text{head}} \times d_{\text{head}}}$: learnable projection matrices on backbone的K, V

intuition:
- Action expert不只attend自己的action tokens,还attend backbone的KV cache
- 但backbone的K/V需要先project ($\mathbf{W}^K, \mathbf{W}^V$),为了align semantic spaces
- $f(\ell) = 2\ell$是cross-layer attention:action expert的第$\ell$层attend backbone的第$2\ell$层。这类似progressive feature integration。

### 4.7 Backbone Input Scheme

- History downsampled到6 frames at 1 FPS,覆盖5s window
- Proprioceptive states在corresponding timestamps采样
- Language instructions和camera intrinsics格式化成textual inputs
- Proprioceptive state history通过2-layer MLP编码成continuous tokens

**防shortcut learning的trick**:
- 每frame的proprioceptive history有75%概率被replace with learnable mask token
- 这迫使model attend to visual inputs和task instructions,而不是over-rely on proprioception

intuition: 因为proprioception与target actions强相关,model容易学到"直接copy proprioception"的shortcut。Masking部分proprioception force model学习visual-language grounding。

**Dual-camera setup**:
- Chest-camera sequence有50%概率被drop
- 防止over-reliance on chest-view observations

### 4.8 VLM Co-Training Data

为了preserve general vision-language reasoning能力,同时cultivate robotic task comprehension,curate了10.4M-sample的VLM co-training mixture,7 datasets:

| Dataset | Samples | Coverage |
|---------|---------|----------|
| FineVision [63] | 3.5M | General VLM,防catastrophic forgetting |
| RefSpatial [64] | 2.5M | Multi-step spatial referring, 2D visual grounding |
| RoboInter-VQA [65] | 1.6M | Embodied QA, temporal reasoning |
| RoboPoint [66] | 1.3M | Spatial affordance, 2D grounding |
| RoboAfford [67] | 765K | Fine-grained manipulation affordance |
| Robo2VLM [68] | 678K | Embodied QA |
| ShareRobot [69] | 13K | Task planning |

**Standardization**: 
- 2D bounding box和point coordinates normalized到Qwen3-VL native的[0, 1000] scale
- 只用single-image samples
- 排除context length超过budget的samples

### 4.9 数据采样权重

每个dataset $i$有subjective quality score $w_i \in [1, 10]$,final sampling weight:

$$W_i = w_i \cdot n_i^{0.5}$$

变量:
- $w_i$: subjective quality score
- $n_i$: total frame count

intuition: 用sqrt(n)而不是n是为了mitigate large datasets dominating training。这是典型的Pareto-style sampling,类似于Chinchilla paper里的domain mixing strategy。

### 4.10 Joint Optimization Objective

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CFM}} + \mathcal{L}_{\text{WM}} + 0.05\mathcal{L}_{\text{VLM}}$$

三个loss term:
- $\mathcal{L}_{\text{CFM}}$: action flow-matching loss (primary)
- $\mathcal{L}_{\text{WM}}$: world-model feature regression loss (auxiliary)
- $\mathcal{L}_{\text{VLM}}$: autoregressive next-token prediction loss of VLM (preserve VL capability)

Weight 0.05 on VLM loss是为了align数值scales。

## 5. Training Infrastructure

这是engineering细节,但对scaling至关重要:

- **Hybrid Sharded Data Parallel (HSDP)** [57]: scale batch size + overlap computation with communication
- **Mixed precision** training
- **torch.compile**: kernel fusion
- **FlexAttention** [58]: optimize attention
- **WebDataset**: sequential streaming,大幅reduce I/O pressure
  - Shuffle buffer size: 16,384
  - 20% random retention probability
  - Randomized shard reading
  - Random sample dropping

**Performance**:
- 44.5% Model FLOPs Utilization (MFU)
- 97 samples/s on 8-A800 node
- 近线性scale到128 GPUs

**Training hyperparameters** (Table 4):

| Hyperparameter | Pre-Training | Post-Training |
|----------------|--------------|---------------|
| Camera setup | Head | Head & Chest |
| Resolution | 384×384 | 640×480 |
| GPUs | 128 A800 | 96 A800 |
| Gradient accumulation | 2 | 1 |
| Global batch size | 4608 | 384 |
| Training steps | 175K | 60K |
| LR (VLM / AE / WM) | 1e-4 / 3e-4 / 3e-4 | 1e-5 / 3e-5 / 3e-5 |
| Freeze-VLM steps | 5000 | 0 |
| Warmup steps | 2000 | 2000 |
| Training time | 164h | 29h |

intuition: Pre-training时先freeze VLM 5K steps,warmup action expert和world model 2K steps。这是staged warmup strategy,避免backbone早期被auxiliary loss corrupt。

参考:
- FlexAttention: https://arxiv.org/abs/2412.05496
- torch.compile: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- WebDataset: https://github.com/webdataset/webdataset
- PyTorch FSDP: https://arxiv.org/abs/2304.11277

## 6. 实验结果

### 6.1 Main Results (40 tasks)

- 22 tasks达到80%+ success rate
- Overall average: **75%**
- Compositional generalization: **65%** (recombine seen primitives into novel sequences)
- Unseen tasks: **62%** (completely novel action semantics)

tasks覆盖7 categories:
- PnP-Easy/Medium/Hard
- Non-prehensile
- Reorient
- Bimanual
- Contact-rich

**Free-form instruction following**: 在cluttered, randomized layouts里,policy严格adhere to language instructions regarding:
- Target objects
- Hand selections
- Specific actions

even for fine-grained manipulation of flat and small objects。

**Failure recovery**: policy能execute multiple retries if previous step fails。

### 6.2 DAgger Efficacy

| Method | Avg. |
|--------|------|
| EgoSteer-FT (teleop only) | 22.5% |
| EgoSteer-DG (after 3 DAgger iterations) | 62.5% |

DAgger细节:
- 3.7K trajectories
- 56 tasks
- 8.3 hours of correction data

intuition: 这是**sample efficient performance leap**。Pre-trained model已经有good priors,但可能misaligned with robot embodiment和deployment distribution。DAgger精准collect deployment failures的corrections,所以8.3 hours就能带来巨大提升。

Per-task results (Table 5):

| Task | EgoSteer-DG | EgoSteer-FT |
|------|-------------|-------------|
| Stack tableware | 80.0% | 50.0% |
| Close laptop | 70.0% | 10.0% |
| Place phone on stand | 50.0% | 0.0% |
| Flip cup | 50.0% | 30.0% |
| Average | 62.5% | 22.5% |

DAgger在所有4个dexterous且failure-prone tasks上都better,某些tasks从0%到50%。

### 6.3 Pre-training Scaling Behavior

EgoSteer-0 (scratch) / 3K / 6K / 9.6K:

Pre-training loss随data scale减小。Real-robot performance:

| Pre-training | Avg. (10 tasks) |
|--------------|-----------------|
| Scratch | 30% |
| EgoSteer-3K | 40% |
| EgoSteer-6K | 43% |
| EgoSteer-9.6K | 60% |

Per-task results (Table 7)的highlights:
- "Grasp object": 80% → 80% → 70% → 100%
- "Hand over object": 70% → 80% → 80% → 100%
- "Place items into container": 40% → 80% → 90% → 100%
- "Stack paper cups": 0% → 20% → 10% → 40%
- "Stack tableware": 0% → 20% → 10% → 20%

intuition: 随pre-training scale增加,观察到**emergent capabilities**:
- Failure recovery
- Enhanced instruction-following
- Improved action accuracy

这是scaling law的manifestation,与EgoScale [6]发现的log-linear scaling law一致。

### 6.4 Baseline Comparison

| Method | Avg. (10 easier tasks) |
|--------|------------------------|
| π0.5 [2] | 22% |
| Being-H0.5 [8] | 39% |
| EgoSteer-9.6K | 74% |

Baseline comparison的highlights (Table 8):
- "Hand over object": Ours 100% vs π0.5 20% vs Being-H0.5 60%
- "Place items into container": Ours 100% vs π0.5 0% vs Being-H0.5 50%
- "Put tennis ball into bucket": Ours 90% vs π0.5 0% vs Being-H0.5 0%

EgoSteer显著outperform两个baselines。Baseline的weaknesses:
- Inconsistent action representations between pre- and post-training phases
- Smaller resolution
- Lack deployment optimizations

参考:
- π0.5: https://arxiv.org/abs/2504.16054
- Being-H0.5: https://arxiv.org/abs/2601.12993
- EgoScale: https://arxiv.org/abs/2602.16710

### 6.5 Ablation Studies

EgoSteer-1K (pre-trained on 1K hours)的ablations (Table 9):

| Variant | Avg. |
|---------|------|
| No WM-objective | 31% |
| No training-RTC | 39% |
| Noisy data | 33% |
| Ours | 44% |

intuition:
- **No WM-objective** (31%): fine-grained manipulation accuracy下降,confirm world model的关键role in action generation accuracy
- **No training-RTC** (39%): severe action pauses,disrupts execution dynamics,contact-rich tasks完全fail due to continuous jitter
- **Noisy data** (33%): unfiltered egocentric data,training不converge,instruction-following和manipulation precision都degrade

这三个ablation分别validates:
1. World model作为auxiliary representation learning
2. Training-time RTC作为deployment-time enabler
3. EgoSmith data curation作为foundation

### 6.6 Few-Shot Adaptation to Long-Horizon Tasks

两个challenging long-horizon tasks:

| Task | Steps | Duration | Demos | Success Rate |
|------|-------|----------|-------|--------------|
| Box-Folding | 18 steps | 40 sec | 120 | 75% |
| Cake-Unboxing | 9 steps | 1 min | 200/229 | 83% |

Comparison (Table 1d):

| Method | Box-Folding | Cake-Unboxing |
|--------|-------------|---------------|
| DP [25] | 0% | 0% |
| IMLE [26] | 0% | 0% |
| Ours (scratch) | 0% | 0% |
| Ours (EgoSteer-9.6K) | 75% | 83% |

intuition: 这是**pre-trained priors真正价值的manifestation**。
- Long-horizon, contact-rich tasks对sample-efficient methods (DP, IMLE)是extremely difficult
- From-scratch baseline也fail,说明问题在priors而不是architecture
- EgoSteer-9.6K的pre-trained manipulation priors能few-shot adapt到新embodiment (AgiBot G1)和complex tasks

Fine-tuning hyperparameters (Table 11):

| Hyperparameter | Box-Folding | Cake-Unboxing |
|----------------|-------------|---------------|
| Pre-training checkpoint | EgoSteer-9.6K @ 155K steps | 同 |
| Resolution | 384×384 | 384×384 |
| GPUs | 8 A800 | 8 A800 |
| Global batch size | 144 | 144 |
| Fine-tuning steps | 44K | 12K |
| Demos | 120 | 229 |

参考:
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- IMLE Policy: https://arxiv.org/abs/2502.12371

## 7. 我的Intuition和Deep Analysis

### 7.1 Full-Stack Co-Design的哲学

paper的central insight是: steerable dexterous manipulation需要data + system + model的co-design,任何一个component fail都会prevent realization。这与recent trend一致——从single-paper innovations转向integrated systems。

每个component都有其scaling bottleneck:
- Data scale bottleneck → EgoSmith解决
- Embodiment gap bottleneck → Unified Robot Stack解决
- Action generation accuracy bottleneck → World Model Expert解决
- Real-time control bottleneck → Training-time RTC解决
- Deployment failure bottleneck → DAgger解决

### 7.2 Egocentric Videos作为Pre-training Data

为什么egocentric videos是理想的pre-training data source?
1. **First-person perspective alignment**: 与robot camera setup (head + chest mounted)的viewpoint一致
2. **Natural task distribution**: human自然perform的manipulation覆盖大部分core primitives
3. **Massively scalable**: 大约116K hours of egocentric video可用
4. **Hand kinematic similarity**: human hand和dexterous robot hand在task space里通过fingertip keypoints可以align

但是,egocentric videos有几个issues需要curation:
- Camera jitter
- Frequent occlusions
- Lack of annotations
- Locomotion干扰

EgoSmith的四个stage分别address这些问题:
1. Pre-filtering: 去除locomotion和occlusion
2. 4D motion estimation: 重建metric-scale world-space trajectory
3. Language labeling: 自动generate 5-level annotations
4. Post-filtering: multi-granularity quality control

### 7.3 World Model作为Auxiliary Objective的深度分析

这是paper的key technical innovation之一。VLA model的fundamental limitation: 缺乏future state imagination,这导致action generation accuracy受限。

**EgoSteer的approach**:
- 不直接generate future pixels
- 不引入expensive generative decoder
- 而是predict action-induced future **DINOv3 features**

为什么DINOv3 features是好的supervision target?
1. **Semantic abstraction**: DINOv3是self-supervised vision foundation model,其features已经abstract掉low-level details,encode semantic content
2. **Stable gradients**: 相比pixel-space prediction,DINOv3 features的MSE loss更stable
3. **Natural alignment**: backbone的visual representations应该与DINOv3-like features有natural alignment
4. **Zero inference overhead**: world model expert是training-only module

更深的intuition: 这与LeCun的JEPA philosophy一致——在latent space里predict future state,避免generative model的high-frequency detail burden。但EgoSteer的twist是: world model expert的gradient通过backbone的KV cache projection回传,shape backbone representation使其更action-aware。换句话说,world model不是deploy-time的predictor,而是training-time的representation shaping tool。

这也与recent "World Action Models are Zero-Shot Policies" [16]的工作呼应:world model和policy可以是同一个model的两个view。

### 7.4 Relative Motion Mapping的数学Beauty

Human-in-the-loop intervention的核心challenge是state jump。EgoSteer的relative motion mapping有一个clean的mathematical structure:

$$\widetilde{\mathbf{T}}_{t'}^{\text{R},i} = \mathbf{T}_t^{\text{R},i} \cdot (\mathbf{T}_t^{\text{H},i})^{-1} \cdot \mathbf{T}_{t'}^{\text{H},i}$$

这可以解读为:
1. $\mathbf{T}_{t'}^{\text{H},i}$: human在$t'$时刻的wrist pose
2. $(\mathbf{T}_t^{\text{H},i})^{-1} \cdot \mathbf{T}_{t'}^{\text{H},i}$: human从$t$到$t'$的relative motion (在human $t$ frame里)
3. $\mathbf{T}_t^{\text{R},i} \cdot [\text{relative motion}]$: robot从自己的$t$ state出发,执行同样的relative motion

关键: relative motion是**frame-invariant**的operator。无论human和robot在$t$时刻的absolute pose如何mismatch,只要relative motion一致,compose的结果自然smooth。

这有点类似于differential drive或者velocity-based control的philosophy,但lift到$SE(3)$上。

### 7.5 Joint Attention Pattern的Design Choice

Action expert和world model expert都jointly attend to backbone的KV cache,但**不互相attend**。这是一个值得思考的design:

为什么不让action expert和world model expert互相attend?
1. **Computational efficiency**: 减少attention complexity
2. **Training stability**: 避免两个auxiliary module的gradient互相干扰
3. **Modular design**: world model是training-only,discard时不影响action expert的inference graph

为什么用$f(\ell) = 2\ell$和$f(\ell) = 7\ell$不同的backbone layer mapping?
- Action expert ($2\ell$): 浅层cross-attention,more frequent integration
- World model expert ($7\ell$): 深层cross-attention,more abstract features
- 这可能是empirical choice,反映不同module需要不同semantic level的backbone features

### 7.6 6D Rotation Representation和Action Space Design

EgoSteer用6D rotation representation (Zhou et al. CVPR 2019)而不是quaternions或Euler angles。intuition:
- Quaternions在continous control里有double cover问题 ($q$和$-q$表示同一rotation)
- Euler angles有gimbal lock和non-uniformity
- 6D representation (前两列 of rotation matrix)是continuous且unique,适合neural network regression

48维action space的breakdown:
- 3D wrist translation
- 6D wrist rotation
- 15D fingertip keypoints (5 fingers × 3D position)
- Per hand: 24D
- Bimanual: 48D

这设计supports cross-embodiment transfer:
- Human hand和robot hand的finger count和joint configuration不同
- 但fingertip positions在task space里是equivalent的
- Wrist $SE(3)$也是embodiment-agnostic的

### 7.7 DAgger的Sample Efficiency

DAgger [20] (Ross, Gordon, Bagnell 2011)是classic imitation learning算法,核心idea:
- Standard imitation learning有covariate shift问题
- Train时states分布与deploy时states分布不同
- Model错误accumulation导致trajectory drift
- DAgger: iteratively collect states visited by learned policy,由expert label,aggregate到training set

EgoSteer的DAgger results:
- EgoSteer-FT: 22.5%
- EgoSteer-DG: 62.5% (after 3 iterations, 8.3 hours)

这是40 percentage points的performance gain,只用8.3 hours of correction data。Sample efficiency的intuition:
1. Pre-trained model有good priors,大部分baseline capability已经存在
2. DAgger corrections精准target deployment failures
3. 8.3 hours主要collect "hard cases"的corrections
4. These hard cases are exactly where model needs to improve

这与recent "Human-in-the-loop RL"工作 [40]和compliant residual DAgger [41]的philosophy一致。

参考:
- DAgger: https://proceedings.mlr.press/v15/ross11a.html
- Human-in-the-loop RL: https://www.science.org/doi/10.1126/scirobotics.ads5033

### 7.8 Limitations和Future Work

paper坦诚地列出三个limitations:

1. **Robotic DoF limitations**: 6-DoF RY-H2 hand无法完全transfer human hand的fine-grained dexterity。Human hand有21+ DoF,robot hand只有6 DoF,所以很多human manipulation primitives丢失。

2. **Lack of tactile feedback**: datasets, model, embodiment都lack tactile sensing。这限制contact-rich tasks的表现。Wipe whiteboard, fold box等需要force-controlled contact的任务,仅靠visual feedback是suboptimal的。

3. **Pre-training scale可以expand**: 9.6K hours相对116K hours available的egocentric video还是minority。Future work可以scale up pre-training to capture broader priors。

可能的future directions:
- 加入tactile sensors和tactile-conditioned policies
- 用更高DoF的robot hand (e.g., Shadow Hand, LEAP Hand)
- Expand pre-training到100K+ hours
- 加入audio modality (sound cues对contact detection有用)
- 加入force/torque sensing

### 7.9 与其他VLA工作的Positioning

EgoSteer在VLA landscape里的position:

| 工作 | Pre-training Data | Robot Data | Steerability | Dexterous? |
|------|---------------------|------------|---------------|------------|
| RT-1/2 [13,14] | Web data | Real robot | Limited | No (gripper) |
| OpenVLA [29] | Open X-Embodiment | Open X-Embodiment | Limited | No (gripper) |
| Octo [28] | Open X-Embodiment | Open X-Embodiment | Limited | No (gripper) |
| π0 [1] | - | Real robot | Yes | No (gripper) |
| π0.5 [2] | Web data | Real robot | Yes | No (gripper) |
| Being-H0 [7] | Egocentric | - | Limited | Yes |
| Being-H0.5 [8] | Egocentric + Web | - | Yes | Yes |
| EgoScale [6] | Egocentric | Real robot | Yes | Yes |
| **EgoSteer** | Egocentric (curated) | Real robot | Yes | Yes |

EgoSteer的unique aspects:
- Full-stack (data + system + model)
- 9.6K curated pre-training data (high quality, not just scale)
- World model as auxiliary representation shaping
- DAgger refinement with relative motion mapping
- Unified action space enabling cross-embodiment transfer

### 7.10 关于Emergent Capability的思考

paper提到pre-training scale增长带来emergent capabilities:
- Failure recovery
- Enhanced instruction-following
- Improved action accuracy

这有点类似于LLM的emergent abilities (Wei et al. 2022)。intuition:
- Small pre-training data: model学到local trajectory patterns,fragile to distribution shift
- Medium pre-training data: model学到task abstractions,but still relies on teleop data
- Large pre-training data: model学到"physical common sense",知道action的consequences,所以能recover from failures

Failure recovery需要model理解"current state"和"goal state"的差距,并generate corrective actions。这需要world understanding,而large-scale pre-training正是提供这种understanding的途径。

### 7.11 关于Hierarchical Language Annotation的Intuition

EgoSmith的5-level language annotation是很有趣的设计:

- Level 1 (Verb + Object, 5 words): "Open the drawer."
- Level 2 (Gist, 15 words): concise summary
- Level 3 (Object-Centric, 30 words): object features
- Level 4 (Hand-Centric, 50 words): grasp style, contact points
- Level 5 (Dense Sequence, 100 words): step-by-step breakdown

intuition: 这creates一个**instruction abstraction hierarchy**。Model可以conditioned on任意level的instruction:
- 高层task descriptions对应coarse action sequences
- 低层dense descriptions对应fine-grained action chunks

这也支持compositional generalization: model学到"verb + object"组合的语义,可以recombine成novel instructions。

与RT-2 [14]等工作的"web-scale language understanding transfer到robot control"的philosophy一致,但更structured。

## 8. Critical Assessment

### Strengths

1. **Truly full-stack**: 不just propose a model,而是co-design data pipeline, robot stack, model architecture, training infra
2. **Strong empirical results**: 75% avg across 40 tasks,75+% few-shot on long-horizon tasks
3. **World model insight**: training-only auxiliary module,zero inference overhead
4. **DAgger with relative motion mapping**: clean mathematical formulation,85% handover success
5. **Open-source**: 完整系统,data,model都release
6. **Ablation thorough**: validates每个component的necessity

### Weaknesses / Open Questions

1. **DoF mismatch**: 6-DoF robot hand vs 21+ DoF human hand。How much manipulation knowledge is lost in this bottleneck?
2. **No tactile feedback**: contact-rich tasks仍然challenging
3. **Pre-training scale**: 9.6K hours vs 116K hours available。需要更大规模来fully validate scaling
4. **Baseline comparison**: π0.5和Being-H0.5没有用EgoSteer的unified action space,所以comparison不完全fair。但这是inevitable due to action representation differences
5. **Cost of data curation**: 9× speedup over HaWoR是great,但仍然需要8×A800 servers,成本high
6. **Long-horizon task complexity**: Box-folding (18 steps)和Cake-unboxing (9 steps)虽然impressive,但仍然远不如human-level dexterity

### Speculative Future Directions

基于这个work,我 speculate几个future research directions:

1. **Multi-finger tactile-conditioned policies**: 加入tactile sensors (e.g., BioTac, DIGIT)和tactile-conditioned action generation
2. **Higher-DoF robot hands**: 用21-DoF LEAP Hand或24-DoF Shadow Hand,enable更fine-grained dexterity
3. **Audio-visual policies**: 加入audio modality for contact detection和interaction understanding
4. **100K+ hour pre-training**: scale up到Ego4D的全部规模
5. **Latent world models**: 把world model expert的predictive capability extend到latent space,可能enable planning
6. **Cross-embodiment generalization**: 用更多种embodiment (different hand designs, arm kinematics)验证unified action space的generalization
7. **Closed-loop RL refinement**: 在DAgger的基础上,加入RL with human preferences (RLHF-style)或asynchronous RL with simulators
8. **Foundation model for tactile**: 类似DINOv3 for vision,develop self-supervised tactile foundation model

## 9. 关键References和Web Links

### Core Papers
- EgoSteer project page: https://egosteer.github.io/
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- π0.6*: https://arxiv.org/abs/2511.14759
- Being-H0: https://arxiv.org/abs/2507.15597
- Being-H0.5: https://arxiv.org/abs/2601.12993
- EgoScale: https://arxiv.org/abs/2602.16710
- World Action Models as Zero-Shot Policies: https://arxiv.org/abs/2602.15922

### Datasets
- Ego4D: https://ego4d-data.org/
- EPIC-KITCHENS: https://epic-kitchens.github.io/
- Egocentric-10K: https://huggingface.co/datasets/builddotai/Egocentric-10K
- Egocentric-100K: https://huggingface.co/datasets/builddotai/Egocentric-100K

### Methods
- HaWoR: https://github.com/haowaru/hawor
- DPVO: https://github.com/princeton-vl/DPVO
- DROID-SLAM: https://github.com/princeton-vl/DROID-SLAM
- Any4D: https://any4d.github.io/
- DINOv3: https://arxiv.org/abs/2508.10104
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- DAgger: https://proceedings.mlr.press/v15/ross11a.html
- Flow Matching: https://arxiv.org/abs/2210.02747
- DiT: https://arxiv.org/abs/2212.09748
- 6D Rotation Representation: https://arxiv.org/abs/1812.07035

### Tools
- Qwen3-VL: https://qwen.ai
- mink (MuJoCo IK): https://github.com/kevinzakka/mink
- FlexAttention: https://arxiv.org/abs/2412.05496
- WebDataset: https://github.com/webdataset/webdataset
- PyTorch FSDP: https://arxiv.org/abs/2304.11277

### VLA Surveys and Foundations
- VLA Survey (Zhong et al.): https://arxiv.org/abs/2507.01925
- RT-1: https://arxiv.org/abs/2212.06817
- RT-2: https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246
- Octo: https://arxiv.org/abs/2405.12213
- GR00T N1: https://arxiv.org/abs/2503.14734

## 10. 总结

EgoSteer是一个truly full-stack的工作,展示了从egocentric videos出发,通过data curation → robot teleoperation → model training → DAgger refinement的完整pipeline,实现steerable dexterous manipulation。

Key takeaways for building intuition:

1. **Data quality > data scale**: 9.6K curated hours > 100K raw hours。EgoSmith的4-stage pipeline (pre-filter, 4D estimation, language labeling, post-filter)是必要的。

2. **World model as representation shaper**: training-only的world model expert,通过predict future DINOv3 features,shape backbone representation使其更action-aware,而inference时zero overhead。

3. **Relative motion mapping for smooth human-in-the-loop**: 在$SE(3)$上的relative motion mapping自然avoid state jumps,enables 85% handover success。

4. **DAgger的sample efficiency**: pre-trained model + DAgger corrections是extremely sample efficient的post-training strategy。8.3 hours带来40%+ performance gain。

5. **Unified action space enables cross-embodiment transfer**: wrist $SE(3)$ + fingertip keypoints的48D representation是embodiment-agnostic的,enable few-shot adaptation到新embodiment (RealMan → AgiBot G1)。

6. **Full-stack co-design**: steerable dexterous manipulation需要data + system + model的co-design,任何一个component fail都会prevent realization。

7. **Emergent capabilities from scale**: pre-training scale增长带来failure recovery, instruction-following, action accuracy的emergence。

这paper让我想起recent trend in robotics: 从single-paper innovations转向integrated systems。未来的robotics research会越来越像这样,需要full-stack thinking。

Hope this builds your intuition, Andrej! 让我知道哪个部分你想dive deeper。
