---
source_pdf: Imagine2Real Towards Zero-shot Humanoid-Object Interaction via Video Generative
  Priors.pdf
paper_sha256: aee7e3bb42429fed9d706739d557e08438b9323e721de7d4c1c6dfed480b0124
processed_at: '2026-08-05T09:18:34-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Imagine2Real 大白话版

## 一、这paper到底在干啥

你给它一张图(机器人站在box旁边)+ 一句话指令("走过去把箱子抱起来走两步"),它先让video生成模型"想象"出一段交互视频,再从视频里抽出几条点轨迹,最后让真机zero-shot照着执行。整个过程从想象到现实,**不用任何CAD模型,不用retargeting,不训练task-specific reward**。

一句话: **把video generator当成免费的3D交互数据来源,绕开所有几何对齐的痛苦。**

---

## 二、为什么这事特别难——用人话讲两个坑

### 坑1: Representation Misalignment(表示对不上)

你想从2D video里恢复robot和object的3D motion,主流做法是用CAD模型当先验。问题在于真实世界里你根本拿不到box、椅子、pillar的精确CAD。即便你硬上,用 [GVHMR](https://github.com/zju3dv/GVHMR) 估robot、用 [FoundationPose](https://github.com/StanfordVL/BundleTrack) 估object,两套估计各自漂,拼到一起空间和深度都对不齐,tracking就崩了。

有些工作更暴力,直接把object"焊"在robot手上跟着走。这种做法扭曲了object的真实运动,推箱子这种场景直接失效——box本来应该在地上滑动,你硬把它粘在手上就完全错位了。

### 坑2: Retargeting Complexity(retarget太痛苦)

从human video抽出来的dense motion(joint角度那种)得retarget到humanoid embodiment。HOI场景下还得把robot和object的mesh一起morph对齐,这个morph过程会**放大video源头的估计误差**。video本来就有噪声,你再做一次mesh对齐,误差层层叠加,出来的reference motion基本没法用。

已有sparse keypoints方法的短板也很明显:
- [VisualMimic](https://arxiv.org/abs/2509.20322): 5个keypoints tracking,没joint信息,步态不自然
- [HEAD](https://arxiv.org/abs/2508.03068): 3个keypoints + AMP保步态,但AMP限制了scalability,只能做navigation,做不了interaction

---

## 三、他们的解法——一句话insight

**把robot和object都表示成4D点轨迹(3D位置+时间),只track 3个critical points(base + 两只手),用预训练好的BFM latent space当policy的搜索域,这样sparse信号也能产生natural gait。**

这招的本质是**把"难"的部分(几何对齐、retargeting)直接跳过,把"软"的部分(步态自然性)甩给已经训好的BFM兜底**。

---

## 四、三阶段progressive training——用人话+公式讲

数据金字塔长这样:

| Stage | 数据量 | 数据来源 | 用途 |
|-------|--------|---------|------|
| 1. BFM Backbone | ~68.5h, ~10,000 clips | AMASS + LAFAN1 + 100STYLE | 学natural motion的latent space |
| 2. Keypoints Tracker | ~8.86h, ~4,000 clips | LAFAN1 + OMOMO + AMASS loco-manipulation | 学sparse keypoints tracking |
| 3. Interaction Adaptor | ~0.43h, ~200 clips | OMOMO box carry/push | 学contact-rich精细控制 |

数据量从68.5h断崖式跌到0.43h,所以必须用progressive strategy,每一层都站在前一层prior的肩膀上,否则0.43h根本学不出contact interaction。

### Stage 1: BFM Backbone — 学一个"natural motion字典"

#### 三个核心模块

**Motion Encoder** $\mathcal{E}$: 把一段$\delta$步长的motion片段压成latent code
$$\mathbf{z}_t = \mathcal{E}(\mathbf{m}_{t:t+\delta}) \in \mathcal{Z}$$
- $\mathbf{m}_{t:t+\delta}$: 从$t$到$t+\delta$时刻的motion subsequence,$\delta$是窗口长度
- $\mathbf{z}_t \in \mathbb{R}^{32}$: 32维latent code
- $\mathcal{Z}$: latent motion space

**Autoregressive Predictor** $\mathcal{P}$: 只看proprioceptive history预测下一步latent
$$\hat{\mathbf{z}}_t = \mathcal{P}(\mathbf{o}_{0:t}^{prop}, \mathbf{z}_{t-1})$$
- $\mathbf{o}_{0:t}^{prop}$: 从0到$t$时刻的本体感知observation history,通过GRU(hidden 256)压缩
- $\mathbf{z}_{t-1}$: 上一时刻的latent state
- $\hat{\mathbf{z}}_t$: 预测的latent prior

Instantaneous observation的构成:
$$\mathbf{o}_t^{prop} = [\mathbf{q}_t, \dot{\mathbf{q}}_t, \omega_t, \psi_t, \mathbf{p}_t, \mathbf{a}_{t-1}]$$
- $\mathbf{q}_t$: joint positions(关节角度)
- $\dot{\mathbf{q}}_t$: joint velocities(关节角速度)
- $\omega_t$: base angular velocity(底盘角速度)
- $\psi_t$: projected gravity(重力方向在底盘坐标系的投影,告诉你robot是不是歪了)
- $\mathbf{p}_t$: end-effector相对base的local位置
- $\mathbf{a}_{t-1}$: previous action(上一步动作,给policy记忆)

**Decoder** $\mathcal{D}$: latent + proprioceptive history → joint-level action
$$\mathbf{a}_t = \mathcal{D}(\mathbf{o}_{0:t}^{prop}, \mathbf{z}_t)$$

#### 训练loss

Predictor的prediction loss:
$$\mathcal{L}_{pred} = \|\hat{\mathbf{z}}_t - sg(\mathbf{z}_t)\|_2^2$$
- $sg(\cdot)$: stop-gradient operator,阻止gradient回传到encoder,防止encoder和predictor互相collapse
- 这其实是 [VQ-VAE](https://arxiv.org/abs/1711.00937) 风格的latent prediction,让predictor学会internalize forward dynamics

#### Intuition(为什么predictor这么关键)

Predictor在downstream被freeze后,它就变成一个**autonomous motor engine**——即使没有dense reference motion,它也能基于proprioceptive history持续输出一个物理上plausible的default motion prior $\hat{\mathbf{z}}_t$,维持natural locomotion和balance。后续的sparse tracking policy只需要在这个default prior上做小幅residual modulation。

类比: 想象一个人在走路(predictor在自动输出走路步态),突然天上掉个苹果要接(sparse keypoints target),他不需要重新设计整个全身运动,只需要在走路步态基础上稍微调整手臂位置(latent residual)。这就是BFM的power——它把"走路"这个default压缩进了latent space。

### Stage 2: Keypoints Tracker — 在latent space做residual

Predictor $\mathcal{P}^*$和Decoder $\mathcal{D}^*$都freeze(上标$*$表示frozen),只训latent residual planner $\mathcal{H}$:

$$\mathbf{z}_t^{res} = \mathcal{H}(\mathbf{o}_{0:t}^{prop}, \mathbf{m}_{t:t+\delta}^{kp}, \mathbf{z}_{t-1}', \hat{\mathbf{z}}_t)$$
$$\mathbf{z}_t' = \hat{\mathbf{z}}_t + \mathbf{z}_t^{res}$$
$$\mathbf{a}_t' = \mathcal{D}^*(\mathbf{o}_{0:t}^{prop}, \mathbf{z}_t')$$

变量含义:
- $\mathbf{m}_{t:t+\delta}^{kp}$: sparse keypoints target subsequence(就是base + 两只手的3D位置轨迹)
- $\mathbf{z}_{t-1}'$: 上一时刻的modified latent(带residual)
- $\hat{\mathbf{z}}_t$: frozen predictor输出的default motion prior
- $\mathbf{z}_t^{res}$: planner输出的latent residual
- $\mathbf{z}_t'$: 最终latent command
- $\mathbf{a}_t'$: decoder输出的full-body action

#### Intuition

这就是 [ResMimic](https://arxiv.org/abs/2510.05070) 思想的latent space版本。在latent space做residual有两个好处:
1. **低维**: 32维 vs joint space几十维,搜索空间小,容易学
2. **天然constrained**: BFM decoder把latent映射到natural motion manifold,residual再大也跑不出natural space,这就从架构层面杜绝了jitter和unnatural posture

类比: 在joint space做residual就像在钢琴上随便按键,可能弹出噪音;在BFM latent space做residual就像在一个music grammar约束的symbol space里挑,出来的永远是符合语法的旋律。

### Stage 3: Interaction Adaptor — joint space精细微调

BFM只见过non-interactive motion,latent space缺乏object manipulation的fine-grained dexterity。所以第三层residual回到joint space:

$$\Delta\mathbf{a}_t = \pi_{adapt}(\mathbf{o}_{0:t}^{prop}, \mathbf{m}_{t:t+\delta}^{kp}, \mathbf{m}_{t:t+\delta}^{obj})$$
$$\mathbf{a}_t^{int} = \mathbf{a}_t' + \Delta\mathbf{a}_t$$

- $\Delta\mathbf{a}_t$: joint-level residual action
- $\mathbf{m}_{t:t+\delta}^{obj}$: target object的6-DoF pose trajectory
- $\mathbf{a}_t^{int}$: 最终execution command

#### 为什么要在joint space再做一层

Latent residual只能表达"粗粒度whole-body motion",contact-rich interaction需要的精细手指/手腕调整在32维latent里根本表达不出来。所以必须下到joint space直接intervene。

但是,有了前两层的强prior,这层adaptor可以用最简单的tracking reward学会,完全不需要 [InterMimic](https://privileged-learning.github.io/) 那种object force curriculum、contact rewards、Physical State Initialization(PSI)等复杂机制。这就是progressive training的payoff——每一层都让下一层的problem变得更简单。

#### 两层residual的intuition

整个policy本质上是两层叠加:

$$\mathbf{a}_t^{int} = \underbrace{\mathcal{D}^*\Big(\mathbf{o}_{0:t}^{prop}, \underbrace{\mathcal{P}^*(\mathbf{o}_{0:t}^{prop}, \mathbf{z}_{t-1}')}_{\text{frozen BFM prior}} + \underbrace{\mathcal{H}(\cdot)}_{\text{latent residual}}\Big)}_{\text{Keypoints Tracker output}} + \underbrace{\pi_{adapt}(\cdot)}_{\text{joint residual}}$$

- 第一层(latent residual): 控制"走向哪里、伸手到哪里"这种粗粒度whole-body motion
- 第二层(joint residual): 控制"手指怎么贴合box表面、手腕怎么调整推力方向"这种精细contact

这种coarse-to-fine的分层residual是control领域的经典思想,但用BFM latent作为coarse layer的搜索域是这篇paper的novelty。

---

## 五、Reward设计——极简哲学

两个stage都用exponential kernel tracking reward:
$$r = \exp(-\|\mathbf{x} - \hat{\mathbf{x}}\|^2 / \sigma)$$
- $\mathbf{x}$: current state
- $\hat{\mathbf{x}}$: reference target
- $\sigma$: kernel bandwidth,越小越严苛(对精度要求越高)

### Keypoints Tracker的reward table

| Reward Term | Weight | $\sigma$ | 作用 |
|---|---|---|---|
| Base Position | 4.0 | 0.2 | 底盘位置追踪 |
| Base Rotation | 2.0 | 1.2 | 底盘朝向(宽松) |
| Marker Position | **8.0** | 0.2 | 手+base keypoint位置(核心!) |
| Marker Rotation | 2.0 | 1.2 | marker朝向(宽松) |
| Joint Position | 4.0 | 20.0 | 关节角度soft regularization |
| Joint Velocity | 2.0 | 50.0 | 关节速度soft regularization |
| Feet Height | 4.0 | 0.15 | 抬腿高度(防止拖步) |
| Termination | -10 | - | 摔倒惩罚 |
| Joint Torques | -2 | - | 能耗约束 |
| Action Rates | -2 | - | 一阶smoothness |
| Action Smoothness | -2 | - | 二阶smoothness |

### Interaction Adaptor的reward table(差异部分)

| Reward Term | Weight | $\sigma$ | 备注 |
|---|---|---|---|
| Base Position | 4.0 | **0.1** | bandwidth从0.2 sharpen到0.1 |
| Marker Position | 4.0 | **0.1** | bandwidth sharpen,精度要求更高 |
| **Object Position** | **12.0** | **0.1** | 新增,最高权重 |
| **Object Rotation** | **3.0** | **1.0** | 新增 |

#### Intuition

- Marker Position在Stage 2是最高权重(8.0),因为tracking hands + base是核心任务
- 到Stage 3,Object Position接管最高权重(12.0),因为object的6-DoF pose才是interaction的真正目标
- $\sigma$从0.2 sharpen到0.1,精度要求翻倍——interaction对位置敏感度远高于locomotion
- 负向reward全是smoothness和能耗约束,这些是sim2real的保命符

---

## 六、Real-world部署pipeline——人话流程

完整闭环:

```
一张图 + "走过去抱起箱子"
        ↓
Seedance 2.0 Fast (ByteDance video generator)
        ↓
一段interaction video (3-5秒)
        ↓
SAM3 分割: left hand / right hand / pelvis / object 4个mask
        ↓
SpaTrackerV2: 对每个mask内pixels做4D点追踪
        ↓
Filtering: 按visibility和confidence过滤outliers
        ↓
Geometric average + smoothing: 每个keypoint出1条reference trajectory
        ↓
Mocap在初始帧抓robot+object的global position → scale calibration
        ↓
Keypoints Tracker + Interaction Adaptor 控制Unitree G1执行
        ↓
真机物理交互!
```

### 关键设计点

**1. 为什么用mocap做scale calibration**

合成video没有metric depth,抽出来的trajectory只有相对运动,没真实尺度。Mocap在初始帧抓real global position,把relative trajectory scale到real-world dimension。Paper明确说mocap比monocular metric depth和stereo vision精度高得多。

**2. Video generation prompt engineering**

为了video物理合理,paper在task description后面固定加一组constraints:

```
[Task Description] Constraints:
- Camera: 固定静止,视角匹配原图
- Robot: 刚体,只有revolute joint,不允许软变形
- Physics: 物理准确,自然smooth realistic
- Environment: 背景地板光照不变
```

这是inference-time的implicit physical regularization,让video generator别太放飞。

---

## 七、实验数据怎么读

### Table 2: BFM vs Direct vs DAgger(核心ablation)

| Method | SR (%) | $E_{hands}$ (cm) | $E_{base}$ (cm) | $E_{mpjae}$ (rad) | $A_{rate}$ | $A_{smooth}$ |
|---|---|---|---|---|---|---|
| Direct | 99.16 | **1.95** | **1.66** | 0.44 | 1.65 | 0.64 |
| DAgger | 99.32 | 3.73 | 3.32 | 0.36 | 0.61 | 0.20 |
| Ours (BFM) | **99.36** | 3.08 | 3.78 | **0.25** | **0.22** | **0.09** |

变量含义:
- SR: success rate,tracking error全程<0.2m的episode比例
- $E_{hands}$ / $E_{base}$: 手/base位置tracking error(cm)
- $E_{mpjae}$: Mean Per Joint Angle Error,衡量whole-body posture自然度,对比ground truth dense motion
- $A_{rate}$: action一阶导数,衡量抖动
- $A_{smooth}$: action二阶导数,衡量平滑度

#### 怎么读

Direct方法tracking error最低(1.95, 1.66cm),看着好,但这是它aggressively overfit到point matching,完全没physical constraint。代价是action jitter严重($A_{rate}=1.65$),posture unnatural($E_{mpjae}=0.44$ rad ≈ 25°关节误差)——这种policy在真机上根本没法deploy,会jitter到散架。

BFM方法tracking error略高(3.08, 3.78cm),但**action smoothness比Direct好7.5倍**($A_{rate}$ 0.22 vs 1.65),**naturalness好76%**($E_{mpjae}$ 0.25 vs 0.44 rad)。

#### Intuition

这就是BFM设计的payoff: 在natural motion manifold上搜一个点,必然要稍微牺牲一点tracking精度,但换来deployable的smoothness和naturalness。仿真里Direct看着SR高,真机上必死。这本质上是"search space restriction"的力量——你把policy限制在一个learned natural manifold上,自动得到了smoothness prior。

### Table 1: HOI task execution(Adaptor的必要性)

| Task | Method | SR (%) | $E_{obj}$ | $E_{obj-f}$ | $E_{hands}$ | $E_{hands-f}$ | $E_{base}$ | $E_{base-f}$ |
|---|---|---|---|---|---|---|---|---|
| Carry Box | w/o Adaptor | **0.00** | - | - | - | - | - | - |
| Carry Box | w/ Adaptor | **82.65** | 6.34 | 7.03 | 7.33 | 4.46 | 5.16 | 4.05 |
| Push Box | w/o Adaptor | 29.82 | 11.11 | 17.87 | 4.42 | 2.39 | 5.85 | 4.58 |
| Push Box | w/ Adaptor | **64.91** | 9.00 | 13.02 | 7.25 | 4.81 | 6.49 | 4.40 |

变量:
- $E_{obj}$ / $E_{obj-f}$: object位置tracking error / 最后帧error
- 下标-f表示final frame

#### 怎么读

**Carry Box: 0% vs 82.65%**——这是最dramatic的对比,直接证明Stage 3 adaptor对contact-rich interaction是必须的。没有adaptor,Keypoints Tracker只能kinematic track手的位置,无法实现物理grasp和lift,所以SR=0%。

**Push Box: 29.82% vs 64.91%**——push相对容易,因为只需要contact不需要stable grasp,所以adaptor增益较小但仍显著。

**Push Box SR为什么比Carry Box低**: Paper说原因是缺marker orientation input。6-DoF object pose的rotation从video里很难robust提取,导致robot没法fine-tune hand position during pushing,容易推歪。

#### Intuition

三个critical points(base + 2 hands)是interaction的minimal sufficient statistics。Carry需要hands lock在box上,base提供locomotion;Push需要hands contact box surface,base提供推力。两种task都成功,说明unified 4D point representation + 3-keypoint tracking确实captured了HOI的核心affordance。

### Real-world Zero-shot部署

部署在 [Unitree G1](https://www.unitree.com/g1/) 上,定性结果(Figure 4)显示:
- 成功lift各种boxes
- 击打"Iron Man" pillar(semantically rich task,训练时没见过)

第二点是video generative prior的power——只要video model能"想象"出来,pipeline就能deploy,不需要事先训练每个task。

---

## 八、训练细节的工程经验

### 频率分层

- Physics simulation: 250Hz ($\Delta t = 0.004$s)
- Low-level PD controller: 50Hz(decimation=5)
- High-level policy: 10Hz($\Delta t = 0.1$s)

这是typical humanoid setting,high-level给target,low-level PD跟踪。

### Noise injection

后两个stage注入5cm Gaussian noise到reference keypoints,模拟real-world tracker noise。这是sim2real的关键——paper的SpaTrackerV2在video上抽的trajectory本来就有噪声,训练时如果不inject noise,policy在真机会被噪声打挂。

### Domain Randomization(Table 5)

覆盖4个维度:
- Robot dynamics: link mass [0.8, 1.2]×, CoM offset [-0.1, 0.1]m, friction [0.3, 1.5]
- Control: PD stiffness/damping [0.9, 1.1]×, joint noise ±0.01 rad
- Perturbations: push base linear ≤0.5m/s, angular ≤0.6 rad/s, body force up to 20N for 0.1-0.2s
- Object properties: mass 0.05-1.5kg, friction [0.1, 1.2], scale randomize

### Action noise策略

- BFM: constant 0.005(小,因为要stable latent learning)
- Keypoints Tracker: constant 0.02
- Adaptor: **learnable** [0.01, 2.0](大范围,因为contact-rich phase需要exploration)

---

## 九、Limitations + 我的延伸思考

Paper自己提了两个limitation:

**1. 依赖mocap**: close-contact时marker容易被occlude。Future work: on-device SLAM + multi-camera fusion(像 [HTC VIVE Tracker](https://www.vive.com/us/accessory/vive-tracker3/))。

**2. Open-loop执行**: 当前只在初始帧scale calibrate后open-loop跑。True closed-loop需要joint fine-tune video generator和motion tracker,但最强video model都是closed-source,开源的 [Wan2.2](https://github.com/Wan-Video/Wan2.1) 物理consistency还不够。

### 我额外想到的几个问题

**A. Sparse representation的表达能力上限**

Paper在3.2.2过滤掉了"dancing和flipping"动作,因为3个keypoints表示不了。这暗示了一个根本问题——3个critical points的表达能力ceiling在哪?能否用adaptive keypoints(比如5-7个,加上feet和head)扩展到更广task space?会带来的副作用是video generation和point tracking的复杂度也上升。

**B. BFM latent space的locality**

32维latent是否足够capture全body interaction的多样性?$\delta$(subsequence length)具体多少?Paper没给。如果$\delta$太小,predictor学不到long-range dynamics;太大,encoder感受野不够。这是个关键hyperparameter。

**C. Scale calibration的时间衰减**

Mocap只在初始帧scale calibrate,但video生成motion可能有temporal deformation(比如box在video里越走越小,因为camera perspective)。这种deformation会在trajectory后期累积误差,初始帧calibration救不了。可能需要在execution中持续用mocap object position做online scale correction。

**D. Closed-loop的真正实现路径**

Future work提到joint fine-tune video generator和motion tracker,但closed-source model没法fine-tune。一个可行的workaround: 用video model生成multiple hypotheses(不同seed或不同prompt perturbation),motion tracker做Bayesian filtering选最physically plausible的。这相当于在action level做粒子滤波,效率比iterative prompting高得多。

**E. Failure mode的深度分析**

Push Box SR只有64.91%,主因是marker orientation缺失。能不能用video model直接predict object 6-DoF pose而不是point trajectory?或者用一个learned [object pose estimator](https://github.com/StanfordVL/BundleTrack)作为补充信号?这会引入geometric prior dependency,但只在adaptor阶段用,可以接受。

**F. Generalization bounds**

Zero-shot部署在train时没见过的Iron Man pillar上,但这个generalization的边界在哪?Object几何/质量/dynamics的范围有多大?Paper没给failure case,这其实是个挺重要的缺失——zero-shot的failure mode才能告诉你方法的真实能力边界。

---

## 十、和Dreamer系列的类比

这paper让我强烈联想到 [Dreamer V3](https://danijar.com/dreamerv3/) 系列思想:

- **Dreamer**: 用RL学一个world model,在imagination里plan,在reality里execute
- **Imagine2Real**: 用internet-scale video generator作为implicit world model,在video(imagination)里生成交互,在reality里execute

区别在于Dreamer的world model是task-specific RL学的,Imagine2Real的"world model"是internet-scale video generator pretrain的,用人类视频作为implicit physical prior。

这是一个非常general的思路——**用大模型的generative prior作为robotics的"imagination engine"**。后续应该会被大量follow,因为:
1. Video model越来越强([Sora](https://openai.com/sora/), [Wan2.2](https://github.com/Wan-Video/Wan2.1), [Seedance](https://www.seedance.ai/))
2. Video model cap free physical priors,不需要手工design
3. Point tracking技术([SAM3](https://github.com/facebookresearch/sam2), [SpaTrackerV2](https://github.com/hoanganhpham1006/spatial_tracker))成熟
4. BFM作为motion prior space的技术([BFM-zero](https://arxiv.org/abs/2511.04131), [UniTracker](https://arxiv.org/abs/2509.13780))成熟

四个enabling技术同时成熟,所以这个范式正好踩在time点上。

---

## 十一、一句话总结

**Imagine2Real = Video Generator(想象) + 4D Point Tracker(抽象) + BFM Latent Search(兜底natural motion) + Two-layer Residual Control(粗细分工),四件已有技术巧妙组合,绕开geometric prior和retargeting两个老大难,实现zero-shot HOI。**

最妙的不是任何单个技术,而是这种**"先把简单问题做好,再用residual补复杂部分"**的分层哲学——progressive training让每一层都站在前一层肩膀上,0.43h的interaction data就能学到carry/push box,这本身就是个值得记下的工程艺术。

---

## 参考链接

- [AMASS Dataset](https://amass.is.tuebingen.mpg.de/)
- [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
- [100STYLE](https://github.com/eborboihun/100STYLE)
- [OMOMO](https://github.com/JiamanLi/OMOMO)
- [SAM3 / SAM2](https://github.com/facebookresearch/sam2)
- [SpatialTrackerV2](https://github.com/hoanganhpham1006/spatial_tracker)
- [Seedance 2.0](https://www.seedance.ai/)
- [Wan 2.1/2.2 Video Model](https://github.com/Wan-Video/Wan2.1)
- [Unitree G1](https://www.unitree.com/g1/)
- [Isaac Gym](https://developer.nvidia.com/isaac-gym)
- [MuJoCo](https://mujoco.org/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [VQ-VAE](https://arxiv.org/abs/1711.00937)
- [DeepMimic](https://xbpeng.github.io/projects/deepmimic/)
- [OmniH2O](https://omni-h2o-umd.github.io/)
- [ResMimic](https://arxiv.org/abs/2510.05070)
- [GenMimic](https://arxiv.org/abs/2512.05094)
- [HDMI](https://arxiv.org/abs/2509.16757)
- [VisualMimic](https://arxiv.org/abs/2509.20322)
- [HEAD](https://arxiv.org/abs/2508.03068)
- [BeyondMimic](https://arxiv.org/abs/2508.08241)
- [InterMimic](https://privileged-learning.github.io/)
- [BFM-zero](https://arxiv.org/abs/2511.04131)
- [UniTracker](https://arxiv.org/abs/2509.13780)
- [HumanX](https://arxiv.org/abs/2602.02473)
- [ZeroWBC](https://arxiv.org/abs/2603.09170)
- [InterReal](https://arxiv.org/abs/2603.07516)
- [Pro-HOI](https://arxiv.org/abs/2603.01126)
- [OmniRetarget](https://arxiv.org/abs/2509.26633)
- [Gen2Real](https://arxiv.org/abs/2509.14178)
- [DexMan](https://arxiv.org/abs/2510.08475)
- [Geometry-aware 4D video](https://arxiv.org/abs/2507.01099)
- [FoundationPose](https://github.com/StanfordVL/BundleTrack)
- [GVHMR](https://github.com/zju3dv/GVHMR)
- [Pink IK](https://github.com/stack-of-tasks/pink)
- [Holosoma (Amazon FAR)](https://amazon-far-science.github.io/)
- [Dreamer V3](https://danijar.com/dreamerv3/)
- [HTC VIVE Tracker](https://www.vive.com/us/accessory/vive-tracker3/)
- [OpenVLA](https://openvla.github.io/)
- [RT-2](https://robotics-transformer2.github.io/)
- [π0.7 Physical Intelligence](https://www.physicalintelligence.company/)
- [AMP (Adversarial Motion Priors)](https://xbpeng.github.io/projects/AMP/)

---

# Imagine2Real: 深度技术解析

这篇paper由Zhejiang University、Shanghai AI Lab、CUHK合作完成, 提出了一个zero-shot humanoid-object interaction (HOI)的完整pipeline。核心insight是利用video generative models作为upstream motion priors, 通过unified 4D point trajectories表示和BFM-based sparse keypoints tracker, 实现从image+text instruction到physical robot execution的端到端部署。

## 1. 问题动机: 为什么现有方法不够用

### 1.1 HOI的数据困境

Whole-body HOI的根本瓶颈在于**高保真3D interaction data极度稀缺**。这导致了几条不同的技术路径, 各有硬伤:

- **VLA路线** [OpenVLA](https://openvla.github.io/), [RT-2](https://robotics-transformer2.github.io/), [π0.7](https://www.physicalintelligence.company/): 依赖end-to-end训练, 但humanoid-specific的大规模dataset不存在, generalization ceiling低
- **AMP路线** [AMP](https://xbpeng.github.io/projects/AMP/): 用sparse高质量数据学specialized behavior, 但容易mode collapse, 需要task-specific reward engineering
- **Motion tracking路线** [DeepMimic](https://xbpeng.github.io/projects/deepmimic/), [OmniH2O](https://omni-h2o-umd.github.io/): 单轨迹replay, 缺乏autonomy

### 1.2 Video generative priors的两难

用internet-scale video generation models提供visual/physical priors是很自然的想法, [GenMimic](https://arxiv.org/abs/2512.05094)已经在pure motion tracking上验证了可行性。但是扩展到HOI会遇到两个critical challenges:

**Representation Misalignment**: 从2D video提取3D motion通常需要strong geometric priors (e.g., explicit CAD models), 真实场景几乎拿不到。即便用成熟工具独立估计robot motion (GVHMR)和object pose (FoundationPose), 也会引入spatial和depth discrepancies, 导致reference motion质量差、tracking失败。很多工作干脆rigid attach object to robot, 但这扭曲了真实object motion。

**Retargeting Complexity**: 从human video提取的dense motion必须retarget到specific humanoid embodiment。HOI场景下, object geometry的引入使得robot和object mesh必须做intensive morphing [OmniRetarget](https://arxiv.org/abs/2509.26633), 这种morphing会**放大video source的estimation errors**。

## 2. 核心insight: 用统一表示+稀疏信号绕过两个坑

Imagine2Real的设计哲学可以总结为两点:

1. **统一4D point trajectories**: 把robot和object motion都用同一套3D point trajectory表示, 不需要任何geometric priors, 也天然aligned在同一frame里
2. **Sparse keypoints tracking + BFM latent search**: 只track三个critical points (base + 两只手), 完全bypass retargeting; 同时用BFM的latent space作为policy search domain, 保证sparse signals也能产生natural gaits

这其实是把"难"的部分(几何对齐、retargeting)直接绕过去, 把"软"的部分(motion naturalness)交给预训练好的BFM来兜底。

## 3. 方法详解: 三阶段progressive training

整个pipeline分为训练侧(三阶段)和部署侧(video-to-motion-to-real)。

### 3.1 Stage 1: BFM Backbone — 构建motion prior space

这一阶段的目标是学一个compact、physically grounded的latent motion representation space, 让后续sparse tracking有"软约束"可依。

#### 3.1.1 数据与架构

训练数据: AMASS + LAFAN1 + 100STYLE, 共~68.5h, ~10,000 motion clips, 通过[pink](https://github.com/stack-of-tasks/pink)做IK retarget到robot。

BFM包含三个核心模块:

**Motion Encoder** $\mathcal{E}$: 把一段$\delta$步长的local motion subsequence压缩成latent code
$$\mathbf{z}_t = \mathcal{E}(\mathbf{m}_{t:t+\delta}) \in \mathcal{Z}$$
其中$\mathbf{m}_{t:t+\delta} = [\mathbf{m}_t, \mathbf{m}_{t+1}, \dots, \mathbf{m}_{t+\delta}]$是从$t$到$t+\delta$时刻的motion subsequence, $\mathbf{z}_t \in \mathbb{R}^{32}$是32维latent code, $\mathcal{Z}$是latent space。

**Autoregressive Predictor** $\mathcal{P}$: 仅基于proprioceptive history预测当前latent prior
$$\hat{\mathbf{z}}_t = \mathcal{P}(\mathbf{o}_{0:t}^{prop}, \mathbf{z}_{t-1})$$
其中$\mathbf{o}_{0:t}^{prop}$是从$0$到$t$时刻的proprioceptive observation history, $\mathbf{z}_{t-1}$是上一时刻的latent state。history通过GRU (hidden size 256)递归压缩。

instantaneous observation定义为:
$$\mathbf{o}_t^{prop} = [\mathbf{q}_t, \dot{\mathbf{q}}_t, \omega_t, \psi_t, \mathbf{p}_t, \mathbf{a}_{t-1}]$$
- $\mathbf{q}_t$: joint positions
- $\dot{\mathbf{q}}_t$: joint velocities
- $\omega_t$: base angular velocity
- $\psi_t$: projected gravity (基底在gravity方向的投影)
- $\mathbf{p}_t$: local end-effector positions relative to base
- $\mathbf{a}_{t-1}$: previous action

**Decoder** $\mathcal{D}$: 把latent command + proprioceptive history解码为low-level action target
$$\mathbf{a}_t = \mathcal{D}(\mathbf{o}_{0:t}^{prop}, \mathbf{z}_t)$$

#### 3.1.2 训练loss

Predictor通过prediction loss匹配encoded reference motion:
$$\mathcal{L}_{pred} = \|\hat{\mathbf{z}}_t - sg(\mathbf{z}_t)\|_2^2$$
其中$sg(\cdot)$是stop-gradient operator, 防止gradient回传到encoder导致collapse。这种设计类似[VQ-VAE](https://arxiv.org/abs/1711.00937)或[VPP](https://arxiv.org/abs/2403.16085)的latent prediction, 让predictor学会internalize forward dynamics和natural motion transitions。

**Intuition**: 关键insight是predictor在downstream任务里被freeze后, 它变成了一个autonomous motor engine — 没有dense reference motion时, 它会持续输出物理上plausible的default motion prior $\hat{\mathbf{z}}_t$来maintain natural locomotion和balance, 只允许higher-level sparse tracking policy做小幅度modulation。这就实现了"自然步态由BFM兜底, 任务行为由sparse信号drive"的解耦。

BFM training还用了latent consistency losses: overlap loss (coef=0.35), commitment loss (coef=0.25), triplet loss (coef=5.0), 这些是representation learning的标准技术。

### 3.2 Stage 2: Keypoints Tracker — 稀疏信号的latent residual

这一阶段不再学whole-body controller, 而是在frozen BFM上做task-conditioned biasing。

#### 3.2.1 架构

Predictor $\mathcal{P}^*$和Decoder $\mathcal{D}^*$都freeze (上标$*$表示frozen), 只训练一个latent residual planner $\mathcal{H}$:

$$\mathbf{z}_t^{res} = \mathcal{H}(\mathbf{o}_{0:t}^{prop}, \mathbf{m}_{t:t+\delta}^{kp}, \mathbf{z}_{t-1}', \hat{\mathbf{z}}_t)$$
$$\mathbf{z}_t' = \hat{\mathbf{z}}_t + \mathbf{z}_t^{res}$$
$$\mathbf{a}_t' = \mathcal{D}^*(\mathbf{o}_{0:t}^{prop}, \mathbf{z}_t')$$

其中:
- $\mathbf{m}_{t:t+\delta}^{kp}$: sparse keypoints tracking target subsequence (base + 两个hands的3D位置)
- $\mathbf{z}_{t-1}'$: 上一时刻的modified latent (带residual的)
- $\hat{\mathbf{z}}_t = \mathcal{P}^*(\mathbf{o}_{0:t}^{prop}, \mathbf{z}_{t-1}')$: frozen predictor输出的default motion prior
- $\mathbf{z}_t^{res}$: planner输出的latent residual, 加到$\hat{\mathbf{z}}_t$上形成新的latent command $\mathbf{z}_t'$
- $\mathbf{a}_t'$: decoder输出的full-body action

**Intuition**: 这是residual policy在latent space的化身, 类似[ResMimic](https://arxiv.org/abs/2510.05070)在joint space做residual, 但在latent space做有两个好处:
1. Low-dimensional (32维) → 搜索空间小, 容易学
2. BFM decoder天然映射到natural motion manifold → residual再大也跑不出natural space

Actor network: GRU (256) + 3-layer MLP [256, 256, 128]
Critic: 5-layer MLP [512, 512, 512, 512, 256]

#### 3.2.2 训练数据

LAFAN1 + OMOMO + AMASS中的loco-manipulation motions, 过滤掉难以用三个critical point表示的动作 (e.g., dancing, flipping), 共~8.86h, ~4,000 clips。这个过滤步骤实际上承认了稀疏表示的表达能力有上限, 但取舍是值得的。

### 3.3 Stage 3: Interaction Adaptor — Joint-level精细控制

BFM只训练在non-interactive motion上, latent space缺乏object manipulation所需的fine-grained dexterity。所以需要第三阶段做residual refinement, 这次是在joint space。

#### 3.3.1 公式

$$\Delta\mathbf{a}_t = \pi_{adapt}(\mathbf{o}_{0:t}^{prop}, \mathbf{m}_{t:t+\delta}^{kp}, \mathbf{m}_{t:t+\delta}^{obj})$$
$$\mathbf{a}_t^{int} = \mathbf{a}_t' + \Delta\mathbf{a}_t$$

其中:
- $\Delta\mathbf{a}_t$: residual joint-level action
- $\mathbf{m}_{t:t+\delta}^{obj}$: target object motion subsequence (6-DoF pose trajectory)
- $\mathbf{a}_t^{int}$: 最终execution command, 加到Keypoints Tracker的output $\mathbf{a}_t'$上

**Intuition**: 这是一个两层residual的架构:
- 第一层residual: latent space ($\mathbf{z}_t^{res}$), 控制粗粒度whole-body motion
- 第二层residual: joint space ($\Delta\mathbf{a}_t$), 控制精细contact interaction

之所以分两层, 是因为interaction的fine-grained dexterity在latent space里表达不出来 (BFM没见过object interaction), 必须在joint space直接intervene。但有了前两层的prior, adaptor只需要简单的tracking reward就能学会, 不需要[InterMimic](https://privileged-learning.github.io/)那种object force curriculum、contact rewards或Physical State Initialization (PSI)。

#### 3.3.2 训练数据

OMOMO中的box-carrying和pushing任务 (不需要dexterous hand manipulation), 通过[Holosoma](https://amazon-far-science.github.io/)做HOI retargeting, 共~0.43h, ~200 clips。注意inference时不需要retargeted data, 只在training时用来correct human-robot morphological mismatch和提供auxiliary tracking rewards。

Actor: [256, 256, 128] MLP, 用learnable action noise (init=1.0, bounded [0.01, 2.0]) encourage contact-rich exploration
Target KL = 0.02

### 3.4 Reward设计: 极简哲学

两个stage都用exponential kernel tracking reward:
$$r = \exp(-\|\mathbf{x} - \hat{\mathbf{x}}\|^2 / \sigma)$$
其中$\mathbf{x}$是current state, $\hat{\mathbf{x}}$是reference target, $\sigma$是kernel bandwidth (越小越严苛)。

**Keypoints Tracker的reward terms** (从Table 4):
- Base Position: weight=4.0, $\sigma=0.2$
- Base Rotation: weight=2.0, $\sigma=1.2$
- Marker Position: weight=8.0, $\sigma=0.2$ (最高权重, 因为是核心tracking target)
- Marker Rotation: weight=2.0, $\sigma=1.2$
- Joint Position: weight=4.0, $\sigma=20.0$ (soft, 只做regularization)
- Joint Velocity: weight=2.0, $\sigma=50.0$
- Feet Height: weight=4.0, $\sigma=0.15$
- Penalties: Termination -10, Joint Torques -2, Action Rates -2, Action Smoothness -2

**Interaction Adaptor的reward terms**:
- 多了Object Position: weight=12.0, $\sigma=0.1$ (最高权重, 最严苛bandwidth)
- Object Rotation: weight=3.0, $\sigma=1.0$
- Marker Position bandwidth从0.2 sharpen到0.1 (要求更紧的精度)
- 其余保持不变

这里有个细节值得注意: **bandwidth $\sigma$从Stage 2的0.2 sharpen到Stage 3的0.1**, 说明adaptor阶段要求更高的tracking precision, 因为object interaction对位置精度更敏感。

## 4. Real-world Deployment Pipeline

### 4.1 Video-to-Motion提取

完整pipeline:

1. **Image + Text instruction → Video**: 用[Seedance 2.0 Fast](https://www.seedance.ai/) (ByteDance)生成interaction video, 它的instruction-following能力最好
2. **Video → Masks**: 用[SAM3](https://github.com/facebookresearch/sam2)分割robot的left hand、right hand、pelvis和target object的masks
3. **Masks → 4D trajectories**: 用[SpatialTrackerV2](https://github.com/hoanganhpham1006/spatial_tracker)追踪mask内的pixels, 得到4D point trajectories (3D position + time)
4. **Filtering & Smoothing**: 基于visibility和confidence scores过滤, 移除spatio-temporal outliers, 然后对每个mask内的valid points做geometric average和smoothing, 得到每个keypoint的单条稳定reference trajectory

### 4.2 Mocap部署

这里有个很smart的设计: mocap系统在部署中承担两个角色:
1. **实时input**: 提供robot和object keypoints的real-time global positions作为policy input
2. **Scale calibration**: 解决video生成motion的scale ambiguity — 因为合成video没有metric depth, 提取的trajectory只是relative motion。用mocap在初始帧捕捉precise global positions, 用来scale和calibrate generated trajectories到real-world dimensions

**为什么不用monocular metric depth或stereo vision?** Paper明确说mocap calibration精度更高。这是合理的trade-off: mocap牺牲了generality, 但获得了sub-mm精度, 对于zero-shot demo是可接受的。

### 4.3 Video Generation Prompt工程

这个细节很关键。为了让生成的video物理上合理, paper在task-specific instruction后面加了一组standardized constraints:

```
[Task Description] Constraints:
• Camera: 固定静止, perspective匹配原图
• Robot: 刚体, 只允许revolute joint, 不允许soft/elastic deformation  
• Physics: 物理准确, 自然smooth realistic
• Environment: 保持background/floor/lighting不变
```

这种prompt engineering实际上是在给video model做implicit physical regularization, 类似于在训练时加domain randomization, 但这里是在inference-time约束生成分布。

## 5. 实验结果深度分析

### 5.1 Keypoints Tracking的Ablation (Table 2)

这是验证核心claim的关键实验: BFM latent search vs direct joint-space tracking。

| Method | SR (%) | E_hands (cm) | E_base (cm) | E_mpjae (rad) | A_rate | A_smooth |
|--------|--------|-------------|-------------|---------------|--------|----------|
| Direct | 99.16 | **1.95** | **1.66** | 0.44 | 1.65 | 0.64 |
| DAgger | 99.32 | 3.73 | 3.32 | 0.36 | 0.61 | 0.20 |
| Ours (BFM) | **99.36** | 3.08 | 3.78 | **0.25** | **0.22** | **0.09** |

**关键观察**:
- Direct方法tracking error最低 (1.95, 1.66cm), 但这是通过aggressive overfitting达成的 — 几乎没有physical constraint, 所以action jitter严重 ($A_{rate}=1.65, A_{smooth}=0.64$) 且posture不自然 ($E_{mpjae}=0.44$ rad ≈ 25°)
- DAgger (distill privileged whole-body tracker到3-point tracker) 有所缓解, 但errors仍高
- BFM方法虽然tracking error略高 (3.08, 3.78cm), 但**action smoothness比Direct好7.5倍** ($A_{rate}$ 0.22 vs 1.65), **naturalness好76%** ($E_{mpjae}$ 0.25 vs 0.44 rad)

**Intuition**: 这正是BFM设计的payoff — 你用稀疏信号去搜一个natural motion manifold上的点, 必然会损失一点tracking精度, 但换来的是deployable的smoothness和naturalness。Direct方法在仿真里SR高, 但在真机上会jitter到散架。

### 5.2 HOI Task Execution (Table 1)

| Task | Method | SR (%) | E_obj (cm) | E_obj-f (cm) | E_hands (cm) | E_hands-f (cm) | E_base (cm) | E_base-f (cm) |
|------|--------|--------|-----------|--------------|--------------|----------------|-------------|--------------|
| Carry Box | w/o Adaptor | 0.00 | - | - | - | - | - | - |
| Carry Box | w/ Adaptor | **82.65** | 6.34 | 7.03 | 7.33 | 4.46 | 5.16 | 4.05 |
| Push Box | w/o Adaptor | 29.82 | 11.11 | 17.87 | 4.42 | 2.39 | 5.85 | 4.58 |
| Push Box | w/ Adaptor | **64.91** | 9.00 | 13.02 | 7.25 | 4.81 | 6.49 | 4.40 |

**关键观察**:
- **Carry Box**: w/o Adaptor = 0% SR, w/ Adaptor = 82.65% SR。这是最dramatic的对比, 证明Stage 3 adaptor对contact-rich interaction是必要的。没有adaptor, Keypoints Tracker只能kinematic track, 无法grasp和lift
- **Push Box**: w/o Adaptor = 29.82%, w/ Adaptor = 64.91%。push相对容易, 因为不需要stable grasp, 只需contact, 所以adaptor的增益较小但仍显著
- **Final tracking errors**: Carry Box的$E_{obj-f}=7.03$cm, $E_{hands-f}=4.46$cm — 在0.2m threshold下符合success criteria
- **Push Box的lower SR**: Paper指出原因是缺少marker orientation input — 6-DoF object pose的orientation从video里很难robust提取, 这限制了robot fine-tune hand position during pushing

**Intuition**: 三个critical points (base + 2 hands)是interaction的minimal sufficient statistics。Carry Box需要hands lock在box上, base提供locomotion; Push Box只需要hands contact box surface, base提供推力。这两种任务都成功, 说明unified 4D point representation + 3-keypoint tracking确实captured了HOI的核心affordance。

### 5.3 Real-world Zero-shot部署

部署在[Unitree G1](https://www.unitree.com/g1/)humanoid上, mocap系统辅助。定性结果 (Figure 4)显示robot成功lift各种boxes, 还执行了一个semantic-rich task: 击打"Iron Man"pillar。这是video generative prior的power — 你不需要事先训练"打Iron Man"这个task, 只要video model能想象出来, pipeline就能deploy。

## 6. 局限性与未来方向

Paper坦诚讨论了两个limitation:

1. **依赖mocap**: 接触任务时marker容易被occlude。Future work: on-device SLAM + multi-camera fusion (e.g., [HTC VIVE Tracker](https://www.vive.com/us/accessory/vive-tracker3/))
2. **Open-loop**: 当前pipeline只在初始帧scale calibrate后open-loop执行。虽然可以iterative prompt video model做pseudo-closed-loop, 但efficiency低。True closed-loop需要joint fine-tune video generator和motion tracker, 但最强video model都是closed-source API, 开源替代 ([Wan2.2](https://github.com/Wan-Video/Wan2.1))物理consistency还不够

## 7. 我的Intuition总结

整篇paper的精髓在于三层解耦:

**Layer 1 — Representation解耦**: 用unified 4D point trajectories同时表示robot和object motion, 绕开了geometric prior dependency。这是一个非常clean的abstraction, 把HOI从"robot + object mesh alignment"问题简化为"两条trajectory alignment"问题。

**Layer 2 — Control解耦**: BFM predictor提供natural gait的"默认值", sparse keypoints tracker只学latent residual。这相当于把policy network的工作量从"生成humanoid motion"降为"在natural motion manifold上做小幅adjustment"。这种"先建立强prior, 再做residual"的范式在control领域越来越常见, [ResMimic](https://arxiv.org/abs/2510.05070)、[AnyTeleop](https://anyteleop.org/)都有类似思想, 但Imagine2Real把它推广到了sparse keypoints setting。

**Layer 3 — Granularity解耦**: Latent residual (粗粒度whole-body) + Joint residual (细粒度contact interaction)。两层residual的bandwidth从$\sigma=0.2$ sharpen到$\sigma=0.1$, 体现了从"approximate tracking"到"precise contact"的精度升级。

如果让我用一个公式概括整个方法, 那就是:

$$\mathbf{a}_t^{int} = \underbrace{\mathcal{D}^*\big(\mathbf{o}_{0:t}^{prop}, \overbrace{\mathcal{P}^*(\mathbf{o}_{0:t}^{prop}, \mathbf{z}_{t-1}')}^{\text{BFM默认prior}} + \overbrace{\mathcal{H}(\cdot)}^{\text{latent residual}}\big)}_{\text{Keypoints Tracker output}} + \underbrace{\pi_{adapt}(\cdot)}_{\text{joint residual}}$$

这里$\mathcal{P}^*$和$\mathcal{D}^*$都是frozen, 只学$\mathcal{H}$和$\pi_{adapt}$。这种设计让整个系统的trainable parameters极少, sample efficiency极高 — 最终interaction stage只用~0.43h数据就学到了carry/push box的能力, 这是progressive training + BFM prior的power。

## 8. 相关工作的延伸联想

- [**BeyondMimic**](https://arxiv.org/abs/2508.08241): 同样用guided diffusion扩展motion tracking到versatile control, 但仍依赖dense reference
- [**HDMI**](https://arxiv.org/abs/2509.16757): 从human videos学interactive whole-body control, 用3-keypoint tracking + AMP, 但依赖AMP导致scalability受限 — Imagine2Real用BFM替代AMP, 解决了这个问题
- [**VisualMimic**](https://arxiv.org/abs/2509.20322): 5-keypoint tracking, 但因缺乏joint-level info无法natural gait — Imagine2Real的BFM latent search解决了这个
- [**HumanX**](https://arxiv.org/abs/2602.02473): 从human videos学agile humanoid interaction, 但仍需heuristic interaction generation
- [**ZeroWBC**](https://arxiv.org/abs/2603.09170): 直接从egocentric video学natural visuomotor control, 思路更激进但尚未达到zero-shot deployment
- [**InterReal**](https://arxiv.org/abs/2603.07516) / [**Pro-HOI**](https://arxiv.org/abs/2603.01126): 统一physics-based imitation for HOI, 但没解决video-driven zero-shot问题

从更广视角看, Imagine2Real代表的是"video generative model → robot action"这一范式的humanoid版本。类似的manipulation工作有:
- [Gen2Real](https://arxiv.org/abs/2509.14178): demo-free dexterous manipulation via generated video
- [DexMan](https://arxiv.org/abs/2510.08475): bimanual dexterous manipulation from human + generated videos
- [Geometry-aware 4D video generation](https://arxiv.org/abs/2507.01099): 给manipulation生成geometry-aware的4D video

但这些都在fixed-base manipulator上, 没有humanoid的dynamic balance问题。Imagine2Real的独特之处在于把BFM作为"动态稳定性的兜底", 让video生成的内容可以被一个sparse tracker可靠执行。

## 9. 一些可以深挖的方向

如果我是reviewer, 我会关心几个问题:

1. **Sparse representation的表达能力上限**: Paper在3.2.2节过滤掉了"dancing和flipping"动作, 因为3个keypoints无法表示。这个上限是什么? 能否用adaptive keypoints (e.g., 5-7个, 包括feet)扩展到更广task space?

2. **BFM latent space的locality**: 32维latent是否足够capture全bodyinteraction的多样性? Encoder的$\delta$ (subsequence length)如何选择? Paper没给具体值。

3. **Scale calibration的局限**: 用mocap做initial-frame scale calibration, 但如果video生成的motion有deformation (e.g., box的大小在video里随时间变化), scale calibration只在初始帧有效, 后续会累积误差。

4. **Closed-loop的真正实现**: Future work提到joint fine-tune video generator和motion tracker, 但closed-source video model没法fine-tune。一个可能的替代方案是用video model生成multiple hypotheses, motion tracker做Bayesian filtering选最physically plausible的。

5. **Failure mode分析**: Push Box的SR只有64.91%, 主因是marker orientation缺失。能否用video generation model直接predict object 6-DoF pose (而不是point trajectory)? 或者用learned object pose estimator作为补充信号?

6. **Generalization bounds**: Zero-shot部署在train时没见过的object上 (e.g., Iron Man pillar), 但这种generalization的边界在哪里? Object的几何/质量/dynamics范围有多大?

## 10. 工程实现的细节启发

从training细节可以学到一些工程经验:

- **Simulation频率分层**: Physics 250Hz, PD 50Hz, high-level 10Hz。decimation factor=5是typical humanoid setting
- **8192 parallel environments** on single RTX 4090 — Isaac Gym的scaling效率很高
- **5cm Gaussian noise injection**到reference keypoints (后两阶段) — 这是为了模拟real-world tracker noise, 是sim2real的关键
- **Domain randomization** (Table 5) 覆盖了robot dynamics、control、perturbations、object properties, 范围设置合理 (e.g., mass [0.8, 1.2]×default)
- **Action noise**: BFM用constant 0.005, Keypoints Tracker用0.02, Adaptor用learnable [0.01, 2.0] — learnable noise在contact-rich phase很重要, 因为需要exploration

## 总结

Imagine2Real是一个设计哲学很清晰的系统工作: 它没有发明新的deep learning module, 而是把video generation、point tracking、BFM、residual policy这些已有技术, 通过unified 4D representation和three-stage progressive training巧妙组合, 解决了HOI的data scarcity和retargeting complexity两个根本难题。最终的zero-shot real-world deployment是令人impressive的, 证明了"想象即现实"(Imagine2Real)的可行性。

这个工作让我想起[Dreamer](https://danijar.com/dreamer/)系列的思想: 用learned world model作为prior, 在imagination里plan, 在reality里execute。只不过Dreamer的world model是RL学出来的, 而Imagine2Real的"world model"是internet-scale video generator, 用人类视频作为implicit physical prior。这是一个非常general的思路, 应该会被大量后续工作follow。

参考链接:
- [Paper PDF (假设openreview/arxiv)](https://arxiv.org/)
- [AMASS Dataset](https://amass.is.tuebingen.mpg.de/)
- [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
- [100STYLE](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
- [OMOMO](https://github.com/JiamanLi/OMOMO)
- [SAM3 / SAM2](https://github.com/facebookresearch/sam2)
- [SpatialTrackerV2](https://github.com/hoanganhpham1006/spatial_tracker)
- [Isaac Gym](https://developer.nvidia.com/isaac-gym)
- [PPO](https://arxiv.org/abs/1707.06347)
- [Unitree G1](https://www.unitree.com/g1/)
- [Seedance](https://www.seedance.ai/)
- [Wan 2.1/2.2 Video Model](https://github.com/Wan-Video/Wan2.1)
- [ResMimic](https://arxiv.org/abs/2510.05070)
- [GenMimic](https://arxiv.org/abs/2512.05094)
- [HDMI](https://arxiv.org/abs/2509.16757)
- [VisualMimic](https://arxiv.org/abs/2509.20322)
- [OmniH2O](https://omni-h2o-umd.github.io/)
- [BeyondMimic](https://arxiv.org/abs/2508.08241)
- [InterMimic](https://privileged-learning.github.io/)
- [Pink IK](https://github.com/stack-of-tasks/pink)
- [Holosoma (Amazon FAR)](https://amazon-far-science.github.io/)
- [Dreamer V3](https://danijar.com/dreamerv3/)
