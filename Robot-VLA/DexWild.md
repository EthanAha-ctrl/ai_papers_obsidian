---
source_pdf: DexWild.pdf
paper_sha256: 83733a7671dd6f0612dde6146d03c2008cfa31423ad7e61be6411ee8821131fa
processed_at: '2026-08-03T20:59:26-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DexWild 人话版

好，那我换个讲法，不堆jargon，讲讲这篇paper到底在干啥，为什么work，哪些地方clever，哪些地方有坑。

---

## 一句话story

你想让robot手学会干各种活——抓spray bottle、倒水、折衣服、插花。传统做法是找个老练的operator坐robot前面teleop，一小时憋40来条demonstration，搬到新地方还得重新架robot。Deepak Pathak组说：**别折腾了，让人戴个手套直接用自己的手干，一小时collect 200条，然后把人的数据和少量robot数据混一起训练，完事。**

---

## 为什么这个idea现在才work

其实"用人的手collect data给robot学"这个想法10年前就有人提过。一直没work的原因是三个gap同时存在：

**Gap 1: 人的手和robot手长得不一样**

人手5个手指、thumb能oppose、非常compliant。LEAP Hand是rigid的、underactuated的、thumb DoF有限。直接把人手的joint angles map到robot手，动作会变形。

**Gap 2: 视觉观察的角度不一样**

人是从第一人称看自己的手，robot的camera装在手背上或者外部架着。训练时看到的画面domain差太远，学到的visual feature没法transfer。

**Gap 3: 数据collection的scalability**

以前的做法要么让人拿着一个fake gripper（UMI那种），要么穿戴笨重的exoskeleton手套。前者限制了动作naturalness，后者穿戴不舒服没法scale到wild collection。

DexWild的contribution是**三件事同时解掉**：
- Gap 1用fingertip position representation + IK retargeting解
- Gap 2用palm-mounted camera mirror placement解
- Gap 3用ArUco + EMF glove + mini-PC的portable kit解

每一件单拎出来都不算特别novel，但三个一起凑齐了才让整个pipeline跑通。这是system paper的典型value——不在于单个component多牛逼，而在于integration让某个idea从"理论上work"变成"实际work"。

---

## Hardware的clever之处

### ArUco vs SLAM

DexCap和UMI用SLAM做wrist tracking，问题是SLAM在feature-sparse的地方会fail——白墙、空桌面、手被物体长时间挡住。你想象一下开drawer的过程，手伸进drawer里，SLAM看不到feature直接丢了。

DexWild用ArUco marker——就是那种黑白方块二维码，贴在手套手背上，一个单目相机看到就能解出6DoF pose。优点是：
- 不需要feature点
- 不需要bundle adjustment
- 一帧搞定，没有drift

缺点是marker必须在视野里，所以手被完全挡住的时候还是会丢。但比起SLAM已经是huge improvement。

**关键trick是relative action representation**：
$$\Delta p_i = p_i - p_{i-1}$$

每个action是"相对于上一时刻的位移"，而不是"相对于world frame的绝对位置"。这意味着tracking camera放哪都行——egocentric头戴也行，exocentric三脚架也行，甚至collection中途挪一下相机都行。完全不需要calibration。

这个设计的beauty在于：**把calibration问题转化成representation问题**。你不需要保证相机和world有固定关系，只需要保证相机能连续追踪marker的relative motion。这是个非常clean的工程insight。

参考UMI: https://universal-zju.github.io/Universal-Manipulation-Interface/

### EMF Glove vs Vision-based Hand Pose

为什么不直接用camera看人的手然后estimate pose？因为在manipulation过程中手几乎总是被物体遮挡。你抓一个杯子的时候，从外部看你的手指基本看不见。

EMF glove用electromagnetic field sensing，每个fingertip有一个sensor，glove能直接测出fingertip的3D position。不受occlusion影响，latency低，精度够用。

输出是**fingertip positions**而不是joint angles——这点很关键。因为任何robot hand只要有IK solver，都能从fingertip positions反解出joint angles。如果输出的是joint angles，那human hand的kinematic structure就hardcode进数据里了，换robot hand就没法用。

这叫**embodiment-agnostic representation**。人手数据今天是retarget到LEAP Hand，明天LEAP Hand出了V3，后天换成Shadow Hand，同一份数据还能用，只需要换一个IK solver。

参考LEAP Hand: https://sites.google.com/view/leap-hand

### Palm Camera的Mirror Placement

这是我觉得最subtle的设计。人在collect数据时，两只palm camera分别装在pinky侧和thumb侧，FoV主要对着environment，尽量少看到手本身。Robot collect数据时，camera装在完全对应的位置。

为什么这样？想象两种极端：

**如果camera看到很多手**：ViT encoder会学到"这是LEAP hand的指节形状""这是人手的纹理"这种embodiment-specific feature。Co-training时，human data的"人手feature"和robot data的"robot手feature"打架，encoder不知道该听谁的。

**如果camera主要看environment + contact zone**：encoder学到的是"这是一个spray bottle的handle""这个object需要从这个角度approach"。这种feature和embodiment无关，human和robot看到的是同一个task structure，encoder被forced学shared representation。

这不是adversarial domain adaptation，不是optimal transport，不是任何fancy的alignment method——纯靠hardware的physical design让两个domain的observation visually aligned。Very elegant。

---

## Co-training的intuition

### 1:2 ratio为什么是sweet spot

这是paper最核心的experimental finding，我详细讲讲为什么。

先看data：
- Human data $D_H$: 9,290 demos，93个environments，visual超级diverse
- Robot data $D_R$: 1,395 demos，3个environments，action super precise

两种data的"价值"完全不同：

**Human data的价值**：给ViT encoder看各种各样的scene，让它学到什么是spray bottle、什么是flower vase、什么是clothes——不管在什么光照、什么桌面、什么角度。这是visual generalization的基础。

**Robot data的价值**：告诉policy"在LEAP hand的joint space里，要精确到什么角度才能grasp成功"。这是action grounding的基础。

想象你在学做菜：
- Human data = 看了1000个不同厨房的做菜视频（visual grounding强，但你的手不是视频里厨师的手）
- Robot data = 在一个厨房里亲手做了100道菜（action grounding强，但只会在这个厨房做）

Co-training就是两者混着学。Ratio问的是：**每batch里，让policy看几个"做菜视频"vs亲手做几次菜？**

- **1:1（等比例）**：video太多，亲手做太少，action grounding不够
- **1:2（robot:human）**：每1次亲手做配2次看video，video给visual diversity，亲手做给action precision，balance最好
- **1:5**：video压倒一切，policy学到一堆"看别人怎么做"的strategy，但自己执行不出来——因为human的retargeted action distribution不在robot workspace里

Paper里Table IV的数据：

| Scenario | Robot Only | 1:2 | Human Only |
|----------|-----------|-----|------------|
| In-Domain | 64.7% | 79.8% | 3.6% |
| In-Wild | 28.5% | 75.1% | 7.3% |
| In-Wild Extreme | 22.0% | 62.7% | — |

两个极端都fail：
- **Robot only**：action grounding强但visual overfit，换个scene就懵
- **Human only**：visual grounding强但action完全没calibration，动起来全是乱动作
- **1:2**：两个都能学到

为什么不是1:1或1:5？因为1:2时robot data的action signal刚好够"锚定"policy的action distribution到robot workspace，而human data的visual signal刚好够"撑开"encoder的visual invariance。再少robot data（1:5）锚不够，再少human data（1:1）diversity不够。

这个ratio本质上是**两个loss landscape的trade-off点**，类似multi-task learning里的task balancing。不是universal的，但对这个dataset和task suite，1:2是best。

### 1:2的std最小也说明问题

Table IV-a里1:2的std是0.050，1:1是0.114，1:5是0.232。std小说明这个ratio是"稳定区"——换几个random seed、换几个data subset，performance都差不多。1:5的std大说明human data太多时policy对具体data sample敏感，不稳定。

这个intuition和LLM的data mixing很像：pre-training data和SFT data的ratio也有sweet spot，太多SFT会overfit，太少SFT又学不会instruction following。

---

## 三个最有意思的实验结果

### 1. Cross-Task Transfer（Pour task）

这是最impressive的。Setup是：

- Pour task：完全没有任何robot data，只有621条human pour demos
- 混上：388条robot Spray demos（完全不同task）
- Co-train 1:2
- 在**In-the-Wild Extreme**场景测试pour

结果：**94% success rate**

对比：
- Robot only（用spray的robot data测pour）: 0%
- Human only（只用pour的human data）: 11%

为什么能work？因为Spray和Pour共享motor primitives——grasp bottle, lift, tilt, orient。Robot spray data教会policy这些primitive的精确执行，Human pour data教会policy在pour task的visual context下什么时候trigger哪个primitive。

这就像你学开车：
- 先在停车场学会踩油门、打方向盘、踩刹车（=Spray robot data，学low-level motor）
- 然后看一堆上路视频（=Pour human data，学high-level strategy）
- 第一次上路就能开（=94% success）

这个结果对robot learning community非常重要：**如果设计好task suite，让tasks共享primitives，就能用少量labeled robot data + 大量human data实现zero-shot cross-task transfer**。这和NLP的pre-train-then-finetune非常类似——robot data是SFT，human data是pre-training。

### 2. Cross-Embodiment Transfer

两个实验：
- **Cross-arm**：xArm data → Franka Panda arm，4.5% → 37.5%（8.3×）
- **Cross-hand**：LEAP V2 Advanced → Original LEAP Hand，13.3% → 65.3%（4.9×）

意思是：今天collect的human data，5年后hardware换代了还能用。这对industrial deployment是huge——你不需要每次换robot都重新collect全部data。

为什么work：Human data的action space是fingertip positions（embodiment-agnostic），retarget到新hand只需要新hand的IK solver。Visual observation通过palm camera mirror placement自动aligned。所以co-trained policy的backbone学到的visual-action mapping可以zero-shot transfer。

### 3. Scaling曲线的非线性

| Scale | Avg Success |
|-------|-------------|
| 0% | 28.7% |
| 25% | 35.1% |
| 50% | 52.3% |
| 100% | 67.8% |

25-50%之间有个steep gain（+17.2%），之前比较平（+6.4%），之后还在涨（+15.5%）。

我的interpretation：25%之前policy在"memorize specific scenes"，跨过threshold后开始"learn invariances"。这和LLM的emergent abilities很像——某些capability在scale超过threshold后突然出现。这里threshold是"环境的visual diversity足够覆盖test-time distribution的core modes"。

而且100%还没plateau，说明继续collect更多data还能improve。这对future work很important——DexWild的framework没有bottleneck，纯粹是"data多就performance高"。

---

## Policy架构的几个choice

### 为什么用Diffusion而不是ACT

DexWild的data来自10个不同的untrained collector，每个人有自己的"strategy"——有人power grasp，有人precision grasp，有人先orient再grasp，有人先grasp再orient。

这种inherent multimodality用unimodal loss（ACT的MSE）会导致**mode averaging**——policy学到所有strategy的平均，实际执行时不是任何一个valid grasp。

Diffusion policy通过iterative denoising能represent arbitrary multimodal distributions。Table I的ACT vs Diffusion对比验证了这点：

| Task | ACT 1:2 | Diffusion 1:2 |
|------|---------|---------------|
| Pour (cross-task) | 50.8% | **95.8%** |

Pour是multimodality最强的task（pour的动作可以很多种），Diffusion的优势最明显。

### Action Chunk Size 48

Policy每次输出48个future actions，不是1个。这是ACT [Zhao et al. 2023]引入的设计，好处是：
1. 减少compounding error（不会每一步都积累误差）
2. 让policy"look ahead"，输出temporally consistent trajectory
3. Multi-modal action distribution的representation

LLM类比：action chunk size 48相当于"predict 48 tokens ahead"而不是next-token prediction。LLM是autoregressive逐token生成，robot policy是一次性生成48步。Trade-off：
- Small chunk → reactive但high compounding error
- Large chunk → smooth但can't react to disturbances

48是paper选的，没ablate。这个数字可能也值得systematic study——像LLM的context length tuning一样。

参考ACT: https://tonyzhaozh.github.io/aloha/

### ViT Encoder from Soup 1M

Visual encoder用pre-trained ViT，从Dasari et al.的Soup 1M初始化。这是在多个robotics dataset上pre-train的ViT。

为什么不从头train？因为DexWild的visual diversity太大了——93个environment的光照、纹理、object外观变化巨大。从头train ViT需要的数据量远超9290 demos。Pre-trained ViT已经学到了generic visual features，DexWild只需要fine-tune到task-specific features。

为什么不用ResNet？paper引用[Ha et al. 2024, Lin et al. 2024]说ViT在in-the-wild manipulation上明显优于ResNet。Intuition是ViT的global self-attention能capture long-range spatial relationships（object和end-effector的relative pose），ResNet的local convolutional inductive bias在diverse visual scenes下反而limiting。

参考Soup 1M: https://arxiv.org/abs/2310.00103
参考R3M: https://r3m.github.io/

---

## RMP做low-level control

Policy输出的是end-effector target（3D position + 6D rotation），不是joint angles。转换由Riemannian Motion Policy (RMP)在Isaac Lab里完成。

为什么需要RMP：
1. **Smoothness**：policy输出可能noisy，RMP能smooth它
2. **Collision avoidance**：实时防双臂自撞、防撞桌面
3. **Joint limit awareness**：保证输出在joint limits内

RMP的核心是把multiple task objectives（target tracking, collision avoidance, joint limits）formulate为Riemannian manifold上的velocity fields，然后用Riemannian sum组合：

$$\dot{q} = J^T (J J^T)^{-1} f_{task}$$

其中$J$是Jacobian，$f_{task}$是task space的desired acceleration。

这个设计让policy不需要learn low-level control，专注high-level strategy。类似你在学开车时不需要管油门怎么喷fuel，只管steering和brake的高层决策。

参考RMPflow: https://arxiv.org/abs/1801.02854

---

## 几个Limitations的深入思考

### 1. Retargeting的systematic loss

人手的thumb能oppose（和其他手指对面），这是人类dexterity的核心。LEAP Hand的thumb DoF有限，没法完全replicate。

比如人用"three-jaw chuck" grasp（thumb + index + middle三点夹持）时，LEAP可能只能做"two-finger pinch"。这种systematic difference会让co-trained policy学到"compromise grasp"——既不是human的optimal style也不是robot的optimal style。

未来方向可能是：learned retargeting with differentiable IK + contact-aware optimization，让retargeting solver知道object geometry和contact constraints。

### 2. 没有failure recovery data

Human collectors是"performing task"不是"teaching recovery"。因为人做这些任务基本都成功，所以demos里几乎没有failure case。Policy遇到disturbance就崩。

解决思路：
- Collection时让user故意失败然后recover（counter-intuitive但可设计）
- 用RL fine-tune with reset-free reinforcement learning
- Collect "failure demonstrations" + human-in-the-loop correction

参考MimicPlay: https://mimic-play.github.io/

### 3. ArUco的occlusion limit

ArUco比SLAM鲁棒，但marker必须visible。Drawer opening这种occlusion-heavy task，marker可能完全被挡住。这解释了为什么paper的5个task都是grasp-and-move类型，没有真正occlusion-heavy的manipulation（比如开drawer、拧螺丝、button push）。

### 4. 没有tactile sensing

Contact-rich task（插花、fold clothes的precise edge alignment）visual feedback不够。Glove本身没tactile sensor。

未来方向：在glove加tactile sensor，collect contact-rich human data，用tactile-conditioned diffusion policy。

### 5. Inference latency

Diffusion U-Net的16步denoising在real-time control里可能有latency问题。Environment frequency是30Hz，16步denoising如果cascade执行可能跑不到30Hz。可能需要parallel denoising或consistency model来accelerate。

参考Consistency Models: https://arxiv.org/abs/2303.01469

---

## 和你工作的几个联想

### Data Scaling Law for Robotics

Karpathy你之前讲过LLM的scaling law（Chinchilla, Kaplan）。Robot learning的scaling law还很不成熟。DexWild的Figure 7右是concrete data point——human data的marginal value在不同scale上nonlinear，存在threshold。

更深的问题：**robot learning的Chinchilla-like law是什么？** 给定model size和data size，optimal compute分配在哪？这个paper没ablate model size，只ablate了data size。但Table I的ACT vs Diffusion对比暗示**policy capacity和data multimodality需要match**——Diffusion容量更大所以能利用更多data的multimodality。

参考Chinchilla: https://arxiv.org/abs/2203.15556
参考Kaplan scaling law: https://arxiv.org/abs/2001.08361

### Pre-training and Transfer的analogy

ViT从Soup 1M pre-trained = "language model的pre-training"
Human DexWild data = "unlabeled internet text"（diverse, large, but embodiment-mismatched）
Robot data = "supervised fine-tuning"

Co-training 1:2 ratio = "混合pre-training和SFT"

这个analogue可能指向unified framework：**robot learning = visual pre-training + action grounding + embodiment alignment**，三个loss按某个ratio组合。

### Cross-embodiment和Cross-lingual的类比

Cross-embodiment transfer 5.8× improvement非常像cross-lingual transfer in NLP。Pre-train on high-resource "language"（human data），transfer to low-resource "language"（new robot hand）。关键是找到**shared representation space**——对language是universal grammar/semantic space，对robotics是fingertip position + visual observation。

### Action Chunking和LLM的next-token prediction

Action chunking size 48相当于"predict 48 tokens ahead"。LLM是next-token prediction，robot policy是next-action-chunk prediction。两者trade-off类似：
- Small chunk → reactive but high compounding error
- Large chunk → smooth but can't react to disturbances

DexWild选48，没ablate这个数字。像LLM的context length tuning一样，action chunk size也应该是重要hyperparameter。

### Diffusion Policy vs Autoregressive Policy

Diffusion生成action chunk是一次性生成48个actions，不是autoregressive逐个generate。避免了"exposure bias"（训练时用ground truth previous action，inference时用predicted previous action）。但失去了online re-planning的灵活性。

未来方向：**autoregressive transformer policy with KV-cache**（像LLM一样），这样能reactive又能long-horizon。Diffusion Policy作者Chi et al.后来也在探索这个方向。

---

## 一个更深的思考：Data-centric Robotics vs Model-centric Robotics

DexWild让我更相信robot learning的下一个phase是**data-centric**而不是model-centric。现在的model architecture（Diffusion, ACT, ViT）已经够好了，bottleneck是data。

DexWild的thesis是：**与其bend over backwards设计更好的architecture来处理有限data，不如design更好的data collection pipeline来scale data**。

这和ImageNet时代computer vision的shift一样：AlexNet之后大家发现model architecture不是bottleneck，data才是。然后有了ImageNet、COCO、OpenImages这些大规模dataset，model性能跟着涨。

Robot learning现在处于pre-ImageNet时代。Droid、Open X-Embodiment这些dataset在scale up，但都是teleop-based，collection cost高。DexWild指出了另一条路——**human-in-the-wild collection + embodiment-agnostic representation + co-training**。

如果这条路work，未来robot data的collection可能像Wikipedia一样crowdsource——untrained volunteers用自己的手collect几小时数据，上传到shared dataset，所有robot都能用。

这需要：
1. Hardware像DexWild一样portable + cheap（现在可能$2000-5000，需要降到$500）
2. Representation像fingertip position一样embodiment-agnostic
3. Co-training framework像1:2 ratio一样work

DexWild把这三件事都demonstrate了，虽然还不完美，但作为proof-of-concept已经很有说服力。

参考Open X-Embodiment: https://robotics-transformer-x.github.io/
参考DROID: https://droid-dataset.github.io/

---

## 最后的take

DexWild这篇paper最让我喜欢的是它的**clean engineering taste**。每一个design choice都有明确的reasoning，没有一处是"加个module试试看"：

- ArUco vs SLAM：明确权衡calibration cost vs robustness
- EMF glove vs vision：明确权衡occlusion robustness vs deployability
- Palm camera mirror：明确权衡embodiment feature vs task feature
- 1:2 ratio：明确权衡visual diversity vs action grounding
- Diffusion vs ACT：明确权衡multimodality capacity vs training stability

这种"每个decision都能讲清楚why"的paper读起来很舒服。不是所有paper都这样——很多paper的architecture里塞了一堆module，ablation study跑出来每个都"提升2%"，但没人能讲清楚为什么。

DexWild的message也很简单：**让人用自己的手collect数据，scale起来比teleop快5倍，co-train with少量robot数据就能在wild generalize**。这个message会被future work build on——不管是加tactile、加failure recovery、还是scale到更多task。

Paper的video results推荐看，尤其是Bimanual Florist和Clothes Folding。DexWild policies在失败时也会"meaningfully attempt"，而robot-only baseline失败时是"erratic unstructured behavior"。这个qualitative difference比success rate数字更说明问题——DexWild学到了task structure而不只是memorize trajectory。

Project page: https://dexwild.github.io

---

# DexWild: 用 Human Hands 解锁 Dexterous Policy 的 In-the-Wild Generalization

这篇paper来自CMU的Deepak Pathak组（一作Tony Tao和Mohan Kumar Srirama， Kenneth Shaw也是co-author， Deepak Pathak是PI）， 核心thesis非常直接： **如果想让dexterous robot policy generalize到unseen environments/objects/embodiments， 就不要再用expensive的teleoperation在固定scene里憋数据了， 让untrained humans用自己的手在93个environments里collect 9,290 demos， 然后co-train with 1,395 robot demos， 在in-the-wild extreme场景下达到62.7% success rate（vs robot-only的22.0%）**。 

Project page: https://dexwild.github.io

---

## I. 核心Insight和Positioning

Robot learning community目前主要的三条data路线：

1. **Teleoperation** （UMI [7], Gello [56], ALOHA [59]）—— high fidelity但scales poorly， 43 demos/hour， 还要搬robot到每个新location
2. **Internet video** （VideoDex [40], HOP [42], LAPA [57]）—— diversity巨大但vision-based hand pose太noisy， 而且没有episode structure
3. **In-the-wild human mocap** （DexCap [55], Robotic Telekinesis [44]）—— 介于两者之间

DexWild的关键差异点在于**三项同时achieved**：

- **Calibration-free**： 用ArUco marker而不是SLAM， 单目相机随便摆
- **Embodiment-agnostic**： action用fingertip position定义， retarget到任意robot hand
- **Palm-camera aligned**： human和robot的palm camera placement严格mirror， visual observation跨embodiment一致

这三点合在一起让co-training变得trivial——不需要任何domain adaptation或adversarial alignment， behavior cloning objective直接work。

参考DexCap: https://dex-cap.github.io/
参考UMI: https://universal-zju.github.io/Universal-Manipulation-Interface/

---

## II. Hardware System深度解析

### 2.1 三个核心组件

```
DexWild-System
├── Wrist Tracking: 单目相机 + ArUco markers on glove
├── Onboard Compute: battery-powered mini-PC (便携式)
└── Sensor Pod
    ├── Motion Capture Glove (EMF-based finger tracking)
    └── 2× Global-Shutter Palm Cameras (stereo, wide FoV)
```

### 2.2 为什么不用SLAM做wrist tracking

DexCap [55]和UMI [7]用SLAM做wrist pose estimation， 但SLAM在以下场景失败：
- Feature-sparse环境（白墙、空桌面）
- 长时间occlusion（开drawer、tool use时手被物体遮挡）
- Dynamic lighting

DexWild选择用**ArUco marker**——single fiducial marker就能给出6DoF pose， 不需要feature点， 不需要bundle adjustment， 一帧搞定。Trade-off是： marker必须在视野内， 所以marker放在手背朝外的位置。

关键设计是**relative state-action representation**：
$$\Delta p_i = p_i - p_{i-1}$$

其中$p_i \in SE(3)$是end-effector pose at timestep $i$。 因为action是relative的， 所以不需要global frame——这意味着tracking camera可以随便摆放， 可以egocentric（头戴）也可以exocentric（三脚架）， 都work。 这是一个非常clean的设计， 完全eliminate了calibration的必要性。

### 2.3 Hand Tracking: 为什么用EMF glove

对比方案：
- **Vision-only hand pose** （HAMER [28], FrankMocap [35]）: deployment容易但occlusion-fragile， 在manipulation task里hand几乎总是被object遮挡
- **Exoskeleton gloves** （DogLove [58]）: 高fidelity + haptic feedback， 但太笨重不适合长时间wild collection
- **EMF sensing glove** （DexWild采用）: Electromagnetic field sensing， 估计fingertip position， lightweight + occlusion robust

EMF glove的输出是fingertip 3D positions（不是joint angles）， 然后通过**inverse kinematics in PyBullet** retarget到robot hand的joint space。 这种keypoint-based representation是embodiment-agnostic的关键——任何有inverse kinematics solver的hand都能用同一份human data。

参考LEAP Hand: https://sites.google.com/view/leap-hand

### 2.4 Palm Camera的Mirror Placement

这是整个system最subtle但最重要的design choice之一。 Figure 3展示了： human palm camera和robot palm camera的安装位置是mirror的——两个camera分别对着pinky侧和thumb侧， FoV主要集中在environment而不是手本身。

为什么这样设计： 
- **如果camera看到太多hand**， ViT encoder会学到embodiment-specific的visual features（"这是LEAP hand的指节"）
- **如果camera主要看到environment + contact zone**， encoder学到的是task-centric representation（"这是一个spray bottle的handle"）
- Mirror placement保证human和robot看同一个object的角度geometrically aligned， 让co-training时visual encoder被"forced"学习shared representation

这是一个implicit的domain alignment trick， 不需要adversarial loss或者optimal transport——纯靠hardware design就实现了。

---

## III. Data Scale和Collection Efficiency

### 3.1 规模数据

| Dataset | Demos | Environments | Tasks |
|---------|-------|--------------|-------|
| $D_H$ (Human) | 9,290 | 93 | 5 |
| $D_R$ (Robot) | 1,395 | ~3 (In-Domain only) | 5 |

**Ratio**: Human : Robot ≈ 6.66 : 1

Task distribution（$D_H$）：
- Spray Bottle: 3,000 demos / 30 envs
- Toy Cleanup: 3,000 demos / 30 envs  
- Pour: 621 demos / 6 envs
- Florist (bimanual): 1,545 demos / 15 envs
- Clothes Folding (bimanual): 1,124 demos / 12 envs

### 3.2 Collection Efficiency (Figure 8)

- **DexWild-System**: 201 demos/hour
- **Bare hand baseline** (理论上限): ~210 demos/hour
- **Gello teleoperation** [56]: 43 demos/hour

**4.6× speedup**的来源（paper Section V-C列了三点）：
1. Gello缺乏haptic feedback， fine manipulation变难
2. Scene reset需要第二个operator或pause
3. Robot hardware setup overhead， 每个新location都要搬robot + 校准

DexWild的untrained user setup： 一页说明书， 戴glove， 开mini-PC， 用Bluetooth clicker或foot pedal控制recording， 每个location collect 100 episodes。

参考DROID: https://droid-dataset.github.io/
参考Gello: https://github.com/wuphilipp/gello_software

---

## IV. Policy Architecture和Training

### 4.1 Observation Space

每个transition $x_i$ at timestep $i$包含：

**Observation** $o_i$:
- $I_{pinky}$: pinky侧palm camera image (RGB, 224×224)
- $I_{thumb}$: thumb侧palm camera image (RGB, 224×224)
- State history: $\{\Delta p_i, \Delta p_{i-\text{step}}, ..., \Delta p_{i-H}\}$ where $H$ is proprioception horizon

注意： **Bimanual tasks**的observation dim从9扩到27， 多出来的是**inter-hand pose**——left hand relative to right hand的relative pose。 paper在Appendix D特别强调这个对Florist task的success rate关键——双臂协调需要explicit的inter-hand geometry信号。

**Action** $a_i \in \mathbb{R}^{26}$:
$$a_i = [a_{arm}, a_{hand}]$$
- $a_{arm} \in \mathbb{R}^9$: relative end-effector的3D position (3) + 6D rotation representation (6)（应该是Zhou et al. 2019的continuous 6D rotation representation， 比quaternion更适合NN learning）
- $a_{hand} \in \mathbb{R}^{17}$: 17维的finger joint position targets

LEAP Hand V2 Advanced的DoF配置： 16个finger joints + 1个wrist/palm相关的额外DoF（具体配置需要查LEAP hand V2的spec sheet， 但17维对应这个hand的full DoF）

### 4.2 Action Chunking

Policy输出action chunk of size $n = 48$：
$$a_{i:i+n-1} = \{a_i, a_{i+1}, ..., a_{i+47}\}$$

这是ACT [59] / Diffusion Policy [6]引入的标准设计， 主要是为了：
1. 减少distribution shift和compounding error
2. 让policy "look ahead"， 输出temporally consistent trajectories
3. Multi-modal action distribution的representation

### 4.3 Visual Encoder

Pre-trained **ViT** (Vision Transformer)， 从**Soup 1M**初始化， 这是Dasari et al. [11]的工作——visuomotor pre-training的unbiased evaluation， 在多个robotics dataset上pre-train的ViT。

为什么不用ResNet： paper引用[16, 23]指出ViT在in-the-wild manipulation tasks上明显优于ResNet。 直觉是ViT的global self-attention能capture long-range spatial relationships（比如object和end-effector的relative pose）， 而ResNet的local convolutional inductive bias在diverse visual scenes下反而limiting。

ViT的output token序列：
$$Z_i = \phi_{vit}(o_i)$$

$Z_i$会作为condition传入diffusion U-Net。

参考Soup 1M / R3M: https://r3m.github.io/
参考Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

### 4.4 Diffusion U-Net Policy

为什么用Diffusion而不是ACT或GMM：

DexWild的data来自多个untrained collectors， 每个人有自己的"strategy"——有人用power grasp， 有人用precision grasp， 有人先orient再grasp， 有人先grasp再orient。 这种**inherent multimodality**用GMM很难拟合（需要预定义mode数）， 用ACT的unimodal MSE loss会导致"mode averaging"——policy学到所有strategy的平均， 实际执行时不是一个valid grasp。

Diffusion policy通过iterative denoising能represent arbitrary multimodal distributions。

#### 4.4.1 Training Objective (Algorithm 1)

核心loss：
$$\mathcal{L}_\theta = \|\epsilon_t - \hat{\epsilon}_\theta(Z_i, a_{i:i+n-1} + \epsilon_t, t)\|_2^2$$

变量解释：
- $t \sim \mathcal{U}(1, T)$: noise scale， 均匀采样， $T = 100$ (training)
- $\epsilon_t \sim \mathcal{N}(0, \sigma_t)$: Gaussian noise with variance $\sigma_t$ corresponding to noise scale $t$
- $a_{i:i+n-1}$: ground truth action chunk (48 steps × 26 dims)
- $a_{i:i+n-1} + \epsilon_t$: noisy action chunk (forward diffusion step)
- $\hat{\epsilon}_\theta$: U-Net预测的noise
- $Z_i$: ViT编码的observation token

这是标准DDPM formulation， 但conditioning on visual tokens $Z_i$而不是text embedding（不像Stable Diffusion）。

#### 4.4.2 U-Net Architecture (Table V)

```
Diffusion U-Net
├── Down Channels: [256, 512, 1024]
├── Kernel Size: 3
├── Groups (GroupNorm): 8
├── Dropout: 0.1
├── Train Diffusion Steps: 100
└── Eval Diffusion Steps: 16
```

Eval时只用16步denoising（vs training的100步）——这是DDIM-style的加速inference， 用deterministic sampling减少denoising步骤。

#### 4.4.3 ACT Baseline对比 (Table I)

| Task | Policy | In-Wild Robot Only | In-Wild 1:2 | In-Wild Extreme Robot Only | In-Wild Extreme 1:2 |
|------|--------|-------------------|-------------|---------------------------|---------------------|
| Spray | ACT | 0.000 | 0.680 | 0.115 | 0.395 |
| Spray | Diffusion | 0.050 | 0.628 | 0.120 | 0.520 |
| Pour | ACT | 0.025 | 0.508 | 0.000 | 0.350 |
| Pour | Diffusion | 0.000 | 0.958 | 0.000 | 0.917 |

Diffusion在co-training后的gain明显大于ACT， 尤其Pour task（cross-task transfer）： Diffusion 95.8% vs ACT 50.8%。 这直接验证了multimodality hypothesis——Pour task的动作分布是从Spray task的motor primitives重组的， 本身就highly multimodal。

---

## V. Co-training的关键发现 (Table IV)

这是整篇paper最informative的实验， 三张sub-table分别对应三种evaluation scenarios。

### 5.1 In-Domain (Table IV-a)

| Ratio | Spray | Toy | Avg | Std |
|-------|-------|-----|-----|-----|
| Robot only | 0.690 | 0.604 | **0.647** | 0.061 |
| 1:1 | 0.630 | 0.792 | 0.711 | 0.114 |
| **1:2** | **0.763** | **0.833** | **0.798** | 0.050 |
| 1:5 | 0.381 | 0.708 | 0.545 | 0.232 |
| Human only | 0.030 | 0.042 | 0.036 | 0.008 |

### 5.2 In-the-Wild (Table IV-b)

| Ratio | Spray | Toy | Avg | Std |
|-------|-------|-----|-----|-----|
| Robot only | 0.050 | 0.521 | 0.285 | 0.333 |
| 1:1 | 0.625 | 0.646 | 0.635 | 0.015 |
| **1:2** | **0.628** | **0.875** | **0.751** | 0.175 |
| 1:5 | 0.393 | 0.625 | 0.509 | 0.164 |
| Human only | 0.063 | 0.083 | 0.073 | 0.015 |

### 5.3 In-the-Wild Extreme (Table IV-c)

| Ratio | Spray | Toy | Florist | Clothes | Avg | Std |
|-------|-------|-----|---------|---------|-----|-----|
| Robot only | 0.120 | 0.500 | 0.063 | 0.198 | 0.220 | 0.195 |
| **1:2** | 0.520 | 0.625 | 0.623 | 0.740 | **0.627** | 0.090 |

### 5.4 关键Intuitions

**Q1: 为什么Robot-only在in-domain强但in-wild差36个点？**

Robot data只在~3个environments collect， ViT encoder overfit到这些scenes的specific visual features（特定桌面纹理、特定光照、特定camera viewpoint）。 在新environment， 这些features out-of-distribution， policy"看着陌生"。

**Q2: 为什么Human-only完全失败（3-7% success）？**

虽然human data有visual diversity， 但human hand的kinematics和robot hand差异巨大：
- Human fingers非常compliant， 能comply到object shape
- LEAP hand是rigid underactuated， 需要精确的joint angle
- Retargeting lossy——human thumb的opposition能力LEAP无法完全replicate

所以human-only的policy学到的是"affordance + approach direction"的高层策略， 但execution时动作out-of-reach或不precise， 全部失败。

**Q3: 为什么1:2是sweet spot而不是1:1或1:5？**

我的interpretation：

设batch里robot data fraction为$f_r$， human data fraction为$f_h = 1 - f_r$。

- $f_r = 1$（robot only）: visual grounding narrow但action grounding strong → overfit to scenes
- $f_r = 0.5$（1:1）: 平衡但robot data的action grounding signal被稀释
- $f_r = 0.33$（1:2）: robot data足够frequent让policy学到precise action， human data足够diverse让visual encoder generalize
- $f_r = 0.17$（1:5）: human data主导， policy学到human-style action distribution， retargeting后的distribution可能不在robot workspace内， 导致1:5反而比1:2差

**这个ratio tuning本质上是在balance两个loss landscape**： 一个是"visual diversity → generalization"的landscape， 一个是"action grounding → execution accuracy"的landscape。 1:2不是universal， 但对这个dataset和task suite， 它让两个landscape的gradient同时下降。

**Q4: 为什么Std在1:2时最小？**

Table IV-a里1:2的std是0.050， 1:1是0.114， 1:5是0.232。 这暗示1:2是"鲁棒点"——微小的data perturbation不会大幅改变policy performance。 1:5的高std说明human data太多时policy对random seed和具体data sample敏感， 不稳定。

---

## VI. Cross-Task Transfer (Figure 7 Left)

这是paper最impressive的结果之一。

**Setup**: Pour task完全没有任何robot data， 只用：
- Human Pour demos (621 demos from 6 envs)
- Robot Spray demos (388 demos)

**Result in In-the-Wild Extreme**:
- Robot only (spray data + human spray, 但task是pour): 0%
- Human only (pour data): 11%
- **Co-trained 1:2**: **94%**

为什么能work： Spray和Pour share motor primitives——grasp bottle, lift, orient, tilt。 Diffusion policy通过Spray task的robot data学到这些primitive的precise execution， 然后通过Human Pour data学到Pour task的visual context和high-level strategy。 两者在latent action space里compose起来。

这个结果对robot learning community有重要implication： **如果设计好task suite， 让tasks share primitives， 就能用少量labeled robot data + 大量human data实现zero-shot cross-task transfer**。 这和NLP里pre-train-then-finetune的paradigm有点类似——robot data是"supervised fine-tuning"， human data是"self-supervised pre-training"。

---

## VII. Cross-Embodiment Transfer (Figure 7 Middle)

两个transfer实验：

### 7.1 Cross-Arm: xArm → Franka Panda

| Method | Success Rate |
|--------|--------------|
| Robot only (xArm data) on Franka | 4.5% |
| **DexWild co-trained** on Franka | **37.5%** |

**8.3× improvement**

### 7.2 Cross-Hand: LEAP V2 Advanced → Original LEAP Hand

| Method | Success Rate |
|--------|--------------|
| Robot only (LEAP V2 data) on LEAP original | 13.3% |
| **DexWild co-trained** on LEAP original | **65.3%** |

**4.9× improvement**

为什么work： Human data的action space是fingertip positions（embodiment-agnostic）， retargeting到新hand只需要新hand的IK solver。 Visual observations也通过palm camera的mirror placement自动aligned。 所以co-trained policy的backbone学到的visual-action mapping可以zero-shot transfer到新embodiment， 只需要少量新embodiment的retargeting calibration。

这个结果对industrial deployment非常重要——意味着human data collected today在5年后hardware换代时仍然有用， 不需要重新collect。

---

## VIII. Scaling Behavior (Figure 7 Right, Table III)

| Scale | Spray | Toy | Avg | Std |
|-------|-------|-----|-----|-----|
| 0% (Robot only) | 0.060 | 0.514 | 0.287 | 0.321 |
| 25% | 0.260 | 0.442 | 0.351 | 0.129 |
| 50% | 0.605 | 0.440 | 0.523 | 0.116 |
| **100%** | **0.565** | **0.792** | **0.678** | 0.160 |

关键观察：
1. **0% → 25%: +6.4%**（小增益）
2. **25% → 50%: +17.2%**（steep gain， critical threshold）
3. **50% → 100%: +15.5%**（持续增益）
4. **没有plateau** → 说明继续collect更多data还能improve

这个nonlinear curve很有意思——25-50%的steep gain说明policy需要"足够多"的visual diversity才能开始generalize， 之前是"memorize specific scenes"， 跨过threshold后是"learn invariances"。

这和LLM的emergent abilities有点像——某些capability在scale超过threshold后突然出现。 这里threshold是"环境的visual diversity足够覆盖test-time distribution的core modes"。

**2.36× improvement from 20% to 100%**说明data collection还在"值得继续"的阶段。

参考Data Scaling Laws in Imitation Learning: https://arxiv.org/abs/2410.24862

---

## IX. Low-Level Control: RMP (Appendix E)

Policy输出的是end-effector targets， 不是joint angles。 转换由**Riemannian Motion Policy (RMP)** [34]完成， 在Isaac Lab [26]里实现。

RMP的核心思想： 把multiple task objectives（target tracking, collision avoidance, joint limits）formulate为Riemannian manifold上的velocity fields， 然后用Riemannian sum组合。 Output是smooth的joint space trajectory。

$$\dot{q} = J^T (J J^T)^{-1} f_{task}$$

其中$J$是Jacobian， $f_{task}$是task space的desired force/acceleration。

好处：
- Smoothness（policy输出可能noisy， RMP smooth它）
- Real-time collision avoidance（防止双臂自撞， 防止撞桌面）
- Joint limit awareness

这个设计让policy不需要learn low-level control， 专注high-level strategy。

参考RMPflow: https://arxiv.org/abs/1801.02854
参考Isaac Lab: https://isaac-sim.github.io/IsaacLab/

---

## X. Limitations和我的思考

paper自己提了三点：

### 10.1 仍需少量robot data

Human-only的3.6% success说明光靠retargeting不够bridge embodiment gap。 未来方向：
- **Better retargeting**： 比如learned retargeting with differentiable IK + contact-aware optimization
- **Online adaptation**： deployment时用少量trial-and-error fine-tune policy到新embodiment

### 10.2 缺少error recovery

Human demos几乎都是成功的， 因为collectors是"performing task"而不是"teaching recovery"。 这导致policy遇到disturbance就崩。

潜在解决方案：
- 在collection时让user"故意失败然后recover"（counter-intuitive但可设计）
- 用RL fine-tune policy with reset-free reinforcement learning
- Collect "failure demonstrations"——和MimicPlay [Liu et al. 2023]等human-in-the-loop correction work类似

### 10.3 没有tactile sensing

Contact-rich task（比如插花、fold clothes的precise edge alignment）visual feedback不够。 Glove本身没tactile sensor。

未来方向：
- 在glove加tactile sensor， collect contact-rich human data
- Tactile-conditioned diffusion policy

### 10.4 我自己补充的几点

**10.4.1 Retargeting fidelity loss**

Human thumb的opposition能力极强， LEAP hand的thumb DoF有限。 Human用"three-jaw chuck" grasp时， LEAP可能只能做"two-finger pinch"。 这种systematic difference会让co-trained policy学到"compromise grasp"——既不是human style也不是robot-optimal style。

**10.4.2 ArUco的trade-off**

虽然paper说ArUco比SLAM鲁棒， 但ArUco需要marker visible。 在drawer opening这种occlusion-heavy task里， marker可能完全被object挡住。 这可能解释了为什么paper的5个task都是grasp-and-move类型， 没有真正occlusion-heavy的manipulation。

**10.4.3 Inference latency**

Diffusion U-Net的16步denoising在real-time control里可能有latency问题。 Environment frequency是30Hz， 16步denoising如果是cascade执行可能跑不到30Hz。 需要parallel denoising或consistency model来accelerate。

**10.4.4 Bimanual的inter-hand pose设计**

paper说inter-hand pose对Florist关键。 但这个pose是relative的， 没有绝对world frame信息。 在unseen environment， 如果table height不同， inter-hand pose可能不足以disambiguate。 可能需要补充base pose或scene-level features。

**10.4.5 Action representation的6D rotation**

$a_{arm}$用6D rotation representation（Zhou et al. CVPR 2019 "On the Continuity of Rotation Representations in Neural Networks"）， 这个比quaternion好， 但对diffusion的noise injection可能还有问题——加noise到6D representation后再project回$SO(3)$会损失信息。 可能需要用$SE(3)$-equivariant diffusion或geodesic diffusion。

---

## XI. 和你工作（Karpathy）的关联思考

### 11.1 Data Scaling Law for Robotics

Karpathy你之前提过LLM的scaling law（Chinchilla, Kaplan等）， robot learning的scaling law还很不成熟。 DexWild的Figure 7右是一个concrete data point——human data的marginal value在不同scale上nonlinear， 存在threshold。

更深的问题： **robot learning的Chinchilla-like law是什么？** 给定model size和data size， optimal compute分配在哪？ 这个paper没有ablate model size， 只ablate了data size。 但Table I的ACT vs Diffusion对比暗示**policy capacity和data multimodality需要match**——Diffusion容量更大所以能利用更多data的multimodality。

### 11.2 Pre-training and Transfer

ViT从Soup 1M pre-trained初始化——这相当于"language model的pre-training"。 Human DexWild data相当于"unlabeled internet text"的analogue（diverse, large, but embodiment-mismatched）。 Robot data相当于"supervised fine-tuning"。

Co-training 1:2 ratio相当于"混合pre-training和SFT"。 这个analogue可能指向一个unified framework： **robot learning = visual pre-training + action grounding + embodiment alignment**， 三个loss按某个ratio组合。

### 11.3 Cross-embodiment和Cross-lingual的类比

Cross-embodiment transfer 5.8× improvement非常像cross-lingual transfer in NLP。 Pre-train on high-resource "language"（human data）， transfer to low-resource "language"（new robot hand）。 关键都是找到**shared representation space**——对language是universal grammar/semantic space， 对robotics是fingertip position + visual observation。

### 11.4 Action Chunking和LLM的next-token prediction

Action chunking size 48相当于"predict 48 tokens ahead"。 LLM是next-token prediction， robot policy是next-action-chunk prediction。 两者的trade-off类似：
- Small chunk → reactive but high compounding error
- Large chunk → smooth but can't react to disturbances

DexWild选48， 没ablate这个数字。 像LLM的context length tuning一样， action chunk size也应该是重要hyperparameter。

### 11.5 Diffusion Policy vs Autoregressive Policy

Diffusion生成action chunk是一次性生成48个actions， 不像autoregressive那样一个一个generate。 这避免了"exposure bias"（训练时用ground truth previous action， inference时用predicted previous action）。 但失去了online re-planning的灵活性。

未来可能的方向： **autoregressive transformer policy with KV-cache**（像LLM一样）， 这样能reactive又能long-horizon。 Diffusion Policy作者Chi et al.后来也在探索这个方向。

参考ACT (Action Chunking Transformer): https://tonyzhaozh.github.io/aloha/
参考Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

---

## XII. 总结

DexWild是一个非常clean的system paper， 三件事都做得很扎实：

1. **Hardware**: 用ArUco + EMF glove + palm cameras实现calibration-free, embodiment-agnostic, high-fidelity的portable system
2. **Co-training**: 1:2 ratio让visual diversity和action grounding balance
3. **Generalization**: 68.5% in-wild success rate (4× over robot-only), 5.8× cross-embodiment, 94% cross-task

最值得follow-up的方向（按我personal ranking）：
1. **Cross-task transfer的systematic study**——Pour transfer只是single example， 需要"primitive library"理论框架
2. **Adding failure recovery data**——让human collectors故意失败和recover
3. **Tactile sensing extension**——contact-rich task的下一个frontier
4. **Action representation design**——$SE(3)$-equivariant diffusion可能进一步提升cross-embodiment
5. **Continual learning**——新hardware发布时incremental adapt而不是zero-shot

paper的代码、hardware instructions都在project page： https://dexwild.github.io

强烈推荐看看video results， Bimanual Florist和Clothes Folding的policy behavior很有意思——尤其DexWild policies在失败时也会"meaningfully attempt"， 而robot-only baseline失败时是"erratic unstructured behavior"。 这个qualitative difference可能比success rate数字更说明问题——DexWild学到了task structure而不只是memorize trajectory。

---

## XIII. 补充References

- DexWild: https://dexwild.github.io
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- LEAP Hand: https://sites.google.com/view/leap-hand
- DexCap: https://dex-cap.github.io/
- UMI: https://universal-zju.github.io/Universal-Manipulation-Interface/
- Gello: https://github.com/wuphilipp/gello_software
- ALOHA / ACT: https://tonyzhaozh.github.io/aloha/
- DROID: https://droid-dataset.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- R3M: https://r3m.github.io/
- VideoDex: https://video-dex.github.io/
- Octo: https://octo-models.github.io/
- Ego4D: https://ego4d-data.org/
- EPIC-KITCHENS: https://epic-kitchens.github.io/2024/
- RMPflow: https://arxiv.org/abs/1801.02854
- Isaac Lab: https://isaac-sim.github.io/IsaacLab/
- Soup 1M / Visuomotor Pre-training: https://arxiv.org/abs/2310.00103
- Zhou et al. 6D Rotation: https://arxiv.org/abs/1812.07035
