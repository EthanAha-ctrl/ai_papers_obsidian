---
source_pdf: Evaluating Real-World Robot Manipulation Policies in Simulation.pdf
paper_sha256: 6a9481f3c0fc43fa6736e5f5fc8ddb72ebf05aad7dbd0ff31721e4d6bffec9e5
processed_at: '2026-08-04T05:30:14-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说SIMPLER

## 一句话概括

**用simulation来测试真实世界训练的robot policy,省得每次都得跑真机器人。**

## 1. 为什么要搞这个

现在robotics社区火得一塌糊涂,RT-1、RT-2、Octo、Open-X-Embodiment一堆generalist policy冒出来。但有个尴尬的事: **评估这些policy比训练它们还痛苦**。

想想RT-1-X要跨22个robot embodiment,上千个task,你要在每个真实robot setup上跑几千个trial,每个trial要setup scene、reset、人工盯。Stanford的人跑一次eval要花几天,Google DeepMind的人要花几周。跨lab还无法reproduce,因为你家的灯光、桌面纹理、camera角度和我家的差一点,policy行为就不一样。

autonomous driving早就解决了这个问题——用CARLA这类simulator先在virtual world里跑自动驾驶policy,过了再去real test。但manipulation很难,因为robot和物体有tight physical interaction,稍微差一点整个trajectory就雪崩。

这篇paper reverse了思路: **既然sim-to-real training能transfer,那real-to-sim evaluation应该也行**。关键是,你不需要做完美的digital twin,只要sim足够像real,使得policy的relative ranking保持一致就够了。

## 2. 核心思想: 只要排序对就行

假设你训练了5个policy checkpoint,real world里跑出success rate是:

```
Policy A: 0.85
Policy B: 0.76
Policy C: 0.62
Policy D: 0.29
Policy E: 0.13
```

你不需要sim也跑出0.85, 0.76, ...。sim里跑出0.78, 0.71, 0.55, 0.22, 0.10也行,只要**排序一致**,你就知道policy A > B > C > D > E。

这个insight很重要,因为完美复现real几乎不可能(你不知道物体的精确mass、friction、物体表面微观几何),但保持排序是有希望的。

## 3. 两个核心Gap要解决

从real到sim,主要差两件事:

### 3.1 Control Gap: 动起来不一样

Real robot的motor有cable friction、joint backlash、temperature drift,这些在sim里没建模。直接把real的PD参数拷到sim里,sim的robot动起来就会偏。你发同一个action序列,real里end-effector画一条曲线,sim里画另一条,偏差累积导致grasp失败。

**解法**: System Identification(SysID)。拿一段real demonstration,记录action序列和对应的end-effector轨迹,在sim里replay同一action序列,调PD参数让sim轨迹尽量贴合real轨迹。

具体loss就是translation误差(L2 norm)加上rotation误差(用Frobenius norm反算geodesic angle)。用simulated annealing三轮coarse-to-fine搜PD参数。

效果: Figure 4里SysID前抓coke can失败,SysID后成功。挺直观的。

### 3.2 Visual Gap: 看起来不一样

Policy是吃image的,real和sim的image差距太大,policy直接OOD。这个gap分两部分:

**Background**: 用green-screening。从real eval video第一帧,用online inpainting工具(就是那种橡皮擦)把robot和前景物体擦掉,得到纯背景。然后在sim渲染时,把sim的foreground segment出来,贴到real背景上。公式就一行: `I' = M * I_sim + (1-M) * I_real`,M是binary mask。

**Foreground object/robot texture**: 这个麻烦点。用SAM在real图里抠出物体,在sim里粗调object pose让segmentation mask重叠,用Nvdiffrast做differentiable rendering精调pose,然后把real RGB"unproject"到sim mesh的UV map上。背面看不见的部分用Zero123++生成novel view补全。

Robot arm的texture也得tune,而且因为arm在运动中颜色会变(light angle变化),作者tune了好几个版本的arm color,eval时取平均消除这个confound。

**有个坑**: 不能只tune一部分。Table III显示,只green-screen或只tune drawer都没用,必须三个一起搞: green-screen + drawer texture + robot texture。原因可能是policy对scene里各部分的**相对关系**敏感,只改一部分反而让关系更不协调。

### 3.3 替代方案: Variant Aggregation

另一种思路是domain randomization的反向版本: 不matching,而是造一堆variant环境(不同background、lighting、distractor、table texture),eval完取平均。

直觉上这个应该work,因为sim-to-real training里domain randomization很强。但实验显示它比Visual Matching差不少(MMRV 0.143 vs 0.056)。

为啥? 因为eval的目标是测policy在某specific real setting的performance,你造variant引入额外noise,模糊了真实signal。training时要policy robust,eval时要measure precise performance,目标不同方法就该不同。

## 4. 评估指标: 为什么Pearson r不够

Pearson correlation是标准做法,但有俩问题:

1. 它要求线性关系。如果sim success rate和real success rate是sigmoid关系(也保持排序但非线性),Pearson r会偏低,即使这个pipeline其实很好。

2. 它对数据range敏感。如果几个policy在real里performance都集中在0.7-0.8之间(真实评估的noise就有这么大),Pearson r会随便跳。

作者提了MMRV (Mean Maximum Rank Violation):

对每对policy (i, j), 如果sim把它们的排序搞错了,就记一个violation,violation的"严重度"等于real performance的margin。最后对每个policy取它和所有其他policy的最大violation,再对N个policy取平均。

直觉: **排序错没关系,但只有当real margin很大时才算严重错误**。两个policy在real里差0.05,sim里排反了不算事(real eval本来就有噪声);real里差0.5排反了就是大问题。

这个metric比Spearman rank correlation好,因为Spearman只看rank不看margin,所有violation一视同仁。

## 5. 实验结果讲了啥

### 5.1 主结果: Sim和Real强相关

Google Robot的4个task, Bridge V2的4个task,跑了6-7个policy checkpoint (RT-1各阶段、RT-1-X、RT-2-X、Octo-Base、Octo-Small),大部分task的MMRV都接近0,Pearson r都接近1。

Bridge V2的"stack block"任务Pearson r=1.000,完美排序。

### 5.2 一个重要baseline: Validation MSE

imitation learning圈子里,大家习惯用validation action MSE选model checkpoint。这篇paper打脸了这个practice。

Table XII: Validation MSE的Pearson r在Bridge任务上是**负的**(-0.951, -0.857, -1.000),也就是说validation loss越低,real success rate反而越低。

为什么? Action prediction MSE和task success是脱钩的。同一个observation,action A和action B差0.01的MSE,但A能成功B会失败,这种事情多了去了。而且action的某些维度critical,某些维度完全redundant, MSE给它们等权重是不合理的。

SIMPLER的Pearson r基本都在0.85-1.00,完全吊打validation MSE。这告诉我们: **不要用validation loss选policy checkpoint,要用sim eval**。

### 5.3 Distribution Shift实验: sim能预测real的robustness

作者设计了5个shift axis: background、lighting、distractor、table texture、camera pose。比较RT-1 with/without image augmentation两个policy在real和sim下的performance drop。

结果: sim和real里,camera pose都是最大drop factor(40%+), lighting和distractor影响都很小(<10%), table texture中等。趋势完美对齐。

更cool的是,作者发现Octo-Base在sim里对robot arm texture超敏感(0% untuned, 29.3% tuned),RT-1-X不敏感。于是设计了一个real novel OOD test: 用gift wrapping paper包住real robot arm。结果验证了sim的预测: Octo-Base从0.293掉到0,RT-1-X从0.760掉到0.520。

**Sim里观察到的sensitivity ranking,在real world里也成立**。这意味着SIMPLER不只是测average performance,还能预测policy对未知distribution shift的robustness。这对policy开发超有价值。

### 5.4 Physical property不敏感

作者故意把coke can mass从10g变到80g,gripper friction从0.25变到2.0,MMRV纹丝不动都是0.031,Pearson r都在0.95以上。

也就是说,**物体物理参数即使不准,sim eval的ranking照样对**。这极大降低了环境搭建成本,你不需要精确测量每个物体的mass、inertia、friction。

### 5.5 不绑定具体simulator

SAPIEN和Isaac Sim两个engine都跑了Google Robot eval,结果一致。说明这套方法论不是SAPIEN特有的trick,可以迁移到别的physics engine。

## 6. 工程细节值得学的

- **Ruckig** (https://github.com/pantor/ruckig): time-optimal joint trajectory planning,同时约束velocity/acceleration/jerk,工业级库,比手写PID强太多
- **Nvdiffrast**: differentiable rendering,做texture alignment必备
- **CoACD**: convex decomposition,生成collision mesh,没这个sim里物体碰撞行为会乱
- **fSpy**: 从单张图反推camera intrinsics,省得手动标定
- **SAM**: segmentation,抠物体
- **Zero123++**: 单图到多视图3D重建,补全texture背面

Google Robot controller里有个细节: simulation跑501Hz, policy每秒输出3次action, 每个action对应167个sim step。用Ruckig在这个区间内做time-optimal joint trajectory,这样action之间平滑过渡,不会硬切。

SAPIEN默认的Projected Gauss-Seidel solver在grasping时会有mesh penetration,作者切到Temporal Gauss-Seidel才正确。这种simulator坑踩到了分享一下挺有价值。

## 7. 局限性

作者自己说了几个:

1. **只搞了rigid body**: 软体、可变形物体、流体没碰。这是physics engine本身的限制。
2. **Green-screening固定camera**: 不能handle moving camera,阴影和反射也不准确。
3. **半自动pipeline**: 还得人手curate asset,离"全自动生成千个environment"还远。
4. **没覆盖dexterous manipulation**: in-hand rotation、bimanual这种高动态任务没测。

## 8. 我的intuition

### 8.1 为什么real-to-sim比sim-to-real容易

Sim-to-real training时,你不知道policy部署到哪个specific real setting,所以domain randomization要覆盖所有可能性,这是**分布覆盖**问题,难度极高。

Real-to-sim eval时,policy已经在某个real setting上学好了,你只需让sim匹配那个specific setting。这是**单点匹配**,难度低一个数量级。

### 8.2 为什么Visual Matching比Variant Aggregation好

你要测policy在real setting A的performance,sim就该尽量像A。Variant Aggregation绕一圈去平均化,引入额外noise,把signal洗掉了。

直觉上,training要robustness,eval要precision,目标不同方法就该不同。

### 8.3 Validation MSE为啥这么烂

这是这篇paper最重要的副产品发现。Imitation learning圈子里其实大家隐约知道validation loss和real performance脱钩,但没人quantitative证明。这篇paper把Pearson r算出来,在Bridge任务上甚至是-1.000,完美负相关。

Reason: action是连续的,success是binary的,这俩的mapping高度non-monotonic。Compounding error让小action偏差累积成大trajectory drift,但MSE对每个action step等权,完全没capture这个。

这就解释了为什么imitation learning圈一直不用validation loss选model,大家都靠real eval,但real eval贵啊。SIMPLER填的就是这个gap: 比validation loss准得多,比real eval便宜得多。

### 8.4 MMRV的精髓

它体现了"评估的实用性": sim可以不准,但不能误导。如果sim告诉你policy A比B好,但real里B比A好,你就白迭代了。MMRV用real margin加权,意味着**严重错误才算严重**,小margin的rank flip可以容忍(反正是noise)。

这比Spearman只数rank violation个数合理多了。

### 8.5 对社区的意义

我觉得这是infrastructure级别的贡献。当Open-X-Embodiment这类跨embodiment dataset成为趋势,real eval根本无法scale。SIMPLER给出的promise是: **同一套sim env可以公平评估所有Open-X policy**,这对policy research democratization很关键。

接下来几年,我expect会有大量paper用SIMPLER作为eval backbone,大量新policy用SIMPLER iterate design。这种"evaluate, iterate, ship"的rapid cycle,是任何data-driven field scale up的前提。

Lerrel Pinto的RB2、TOTO benchmark、DROID dataset这是一条线,SIMPLER是这条线上的关键节点:从distributed real eval走向sim-based eval。

### 8.6 未来方向我猜

- Learnable simulator: 用neural surrogate替代手工asset
- 自动asset生成: Gaussian Splatting + physics,直接从real video生成可交互3D scene
- Differentiable pipeline: 从real video自动tune sim参数
- VLM in the loop: 用GPT-4V判断sim observation够不够像real
- Cross-embodiment SIMPLER: 一个env支持多种robot同时eval
- Beyond rigid body: 把IPC这类soft body simulator集成进来

## 9. 总结

这篇paper技术上没有惊人突破,但**methodologically极其重要**。它把"sim eval real policy"从intuition变成validated methodology,给出了:

- 可复现的工程pipeline (SysID + Visual Matching)
- 严谨的评估metric (MMRV)
- 跨simulator、cross-embodiment、cross-policy的验证
- 开源environment collection

对整个robot manipulation community是infrastructure级别贡献。如果你的policy在SIMPLER里跑得好,real world里大概率也好; 如果SIMPLER里跑得差,real world里也大概率差。这种scalable、reproducible、reliable的eval tool,是任何领域从research走向engineering的必经之路。

Project page: https://simpler-env.github.io
Code: https://github.com/simpler-env/SIMPLER

---

# SIMPLER: Real-to-Sim Policy Evaluation 深度解析

## 1. Paper 核心命题与动机

这篇paper tackle的问题非常fundamental: 当robotics社区大力推动generalist manipulation policies (RT-1, RT-2, Octo, Open-X-Embodiment)的同时, 评估方法学严重滞后。real-world eval的痛点在于:

- **Scalability瓶颈**: 一个policy要跨100+ task/scenario评估, 每个trial要手动resetup, 单次episode可能几十秒到几分钟
- **Reproducibility困难**: lighting、camera pose、object微小差异都会影响policy行为, 跨lab无法对齐
- **Cost**: 一组paper的real eval动辄几千个episode, 工程师要现场盯设备

作者reverse了传统的sim-to-real方向, 提出**real-to-sim evaluation**: 用真实数据训练的policy, 放到purpose-built的simulator里跑, 看sim performance是否correlate with real performance。这是autonomous driving领域早就有的事(CARLA, LidarSim), 但manipulation的特殊性在于tight closed-loop interaction, 稍微的action差异就会导致后续state完全不同。

关键intuition来自Section III.A: **目标不是1:1复现real-world performance, 而是relative performance ordering保持一致**。即policy A 在real比policy B好, 那么在sim里也应该看到同样的ordering。这降低了对simulator fidelity的要求。

Project page: https://simpler-env.github.io
Code: https://github.com/simpler-env/SIMPLER

---

## 2. 评估指标设计: 为什么Pearson r不够, 要提出MMRV

### 2.1 Pearson correlation coefficient的局限

Pearson $r$ 衡量两个变量的线性一致性, 公式:

$$r = \frac{\sum_{i}(R_i - \bar{R})(R_{S,i} - \bar{R_S})}{\sqrt{\sum_i (R_i - \bar{R})^2 \sum_i (R_{S,i} - \bar{R_S})^2}}$$

其中 $R_i$ 是第 $i$ 个policy在real的success rate, $R_{S,i}$ 是其在sim的success rate, $\bar{R}, \bar{R_S}$ 是均值。

作者用Figure 3的四个scatter plot说明Pearson $r$ 的两个致命缺陷:

1. **要求线性关系**: 中右图里sim正确恢复了ranking, 但非线性映射(比如sigmoid-like), Pearson $r$ 会偏低, 即使这是一个很好的eval pipeline
2. **对range敏感**: 右侧图当real performance都在narrow range内时, 小噪声会引起Pearson $r$ 大幅波动

### 2.2 MMRV (Mean Maximum Rank Violation)的设计

Spearman rank correlation只看ranking不看margin, 也有问题: Figure 3最左和中间左两个pipeline都只有1个rank violation, 但前者margin小(0.1), 后者margin大(0.6), 显然后者更糟。

MMRV公式:

$$\text{RankViolation}(i, j) = |R_i - R_j| \cdot \mathbf{1}[(R_{S,i} < R_{S,j}) \neq (R_i < R_j)]$$

$$\text{MMRV}(R, R_S) = \frac{1}{N} \sum_{i=1}^{N} \max_{1 \leq j \leq N} \text{RankViolation}(i, j)$$

变量含义:
- $N$: 评估的policy数量
- $R_i, R_j$: policy $i$ 和 $j$ 在real的performance
- $R_{S,i}, R_{S,j}$: policy $i$ 和 $j$ 在sim的performance
- $\mathbf{1}[\cdot]$: indicator function, 当sim错误rank两个policy时取1
- $|R_i - R_j|$: rank violation的"代价", 用real performance margin加权

核心insight: **rank violation只有在real performance有显著差距时才算严重**。如果两个policy在real里就差0.05, 那么sim里rank错了不算大问题(可能就是noise); 但如果real差0.5, sim里rank错就是大问题。

聚合方式: 对每个policy $i$, 取它和所有其他policy $j$ 的最大rank violation, 然后对N个policy取平均。这给每个policy一个"最坏邻居"视角的violation score。

Intuition: MMRV ∈ [0, 1], 越低越好。0意味着sim完美保持ordering。

---

## 3. 控制Gap的弥合: System Identification

### 3.1 问题形式化

给定real-world rollout一段action trajectory $\{\mathbf{a}_i\}_{i=1}^T$, 记录对应end-effector pose trajectory $\{(\mathbf{x}_i, R_i)\}_{i=1}^T$, 其中 $\mathbf{x}_i \in \mathbb{R}^3$ 是translation, $R_i \in SO(3)$ 是rotation matrix。

在simulator里open-loop重放同一action sequence, 用PD参数 $(\mathbf{p}, \mathbf{d})$ 控制joints, 得到sim trajectory $\{(\mathbf{x}'_i, R'_i)\}_{i=1}^T$。初始条件对齐: $\mathbf{x}'_0 = \mathbf{x}_0, R'_0 = R_0$。

### 3.2 System Identification loss

Translation loss (Eq. 3):
$$\mathcal{L}_{\text{transl}}(\mathbf{p}, \mathbf{d}) = \frac{1}{T} \sum_{i=1}^T \|\mathbf{x}_i - \mathbf{x}'_i\|_2$$

直接是end-effector位置误差的L2 norm平均。

Rotation loss (Eq. 4):
$$\mathcal{L}_{\text{rot}}(\mathbf{p}, \mathbf{d}) = \frac{1}{T} \sum_{i=1}^T \arcsin\left(\frac{1}{2\sqrt{2}} \|R_i - R'_i\|_F\right)$$

这里有个细节: $R_i - R'_i$ 的Frobenius norm $\|\cdot\|_F$ 与rotation之间的geodesic distance有关。对于两个rotation matrix $R_1, R_2$, 它们的相对rotation $R_1 R_2^{-1}$ 的旋转角 $\theta$ 满足:

$$\|R_1 - R_2\|_F = 2\sqrt{2} \sin(\theta/2)$$

所以 $\arcsin(\frac{1}{2\sqrt{2}}\|R_i - R'_i\|_F) = \arcsin(\sin(\theta/2)) = \theta/2$ (当 $\theta \leq \pi$), 这给出了half-angle的geodesic distance, 单位是radian。

Total loss (Eq. 5):
$$\mathcal{L}_{\text{sysid}}(\mathbf{p}, \mathbf{d}) = \mathcal{L}_{\text{transl}} + \mathcal{L}_{\text{rot}}$$

简单相加, 没有加权(可能是个简化)。

### 3.3 优化策略

用simulated annealing (Kirkpatrick 1983, 经典的Monte Carlo优化方法)优化。流程:
1. 初始PD参数 $(\mathbf{p}_0, \mathbf{d}_0)$ 从real controller copy来
2. 设定搜索范围 $[\mathbf{p}_{\text{low}, 0}, \mathbf{p}_{\text{high}, 0}] \times [\mathbf{d}_{\text{low}, 0}, \mathbf{d}_{\text{high}, 0}]$
3. Normalize到 $[0, 1]$
4. 运行simulated annealing, 选最优作为 $(\mathbf{p}_1, \mathbf{d}_1)$
5. 缩小搜索范围, 重新跑annealing
6. **总共3轮** (coarse-to-fine)

Intuition: PD参数在sim里和real里有差异, 因为simulator的物理引擎不精确建模了friction, motor inertia, cable friction等。直接拷贝real PD会导致sim里joint tracking不准确, end-effector trajectory偏差大, 然后policy看到的observation就偏了, action又进一步放大偏差, closed-loop下雪崩。

Figure 4对比了SysID前后: 原始参数sim里抓coke can失败, SysID后能成功grasp。

### 3.4 实际控制器实现

Google Robot controller (Algorithm 1) 关键设计:

```
Simulation frequency: H_sim = 501 Hz
Control frequency: H_ctrl = 3 Hz (policy每秒输出3次action)
所以每个control step对应 501/3 = 167 sim steps
```

每个control step流程:
1. **Forward Kinematics**: $({\bf x}, R) = \text{FK}(q_{\text{arm}})$, 算当前end-effector pose
2. **目标pose计算**: $({\bf x}_{\text{goal}}, R_{\text{goal}}) = ({\bf x}_a + {\bf x}, R_a \cdot R_{\text{arm}})$, 即current pose + delta action
3. **Inverse Kinematics**: $q_{\text{goal}} = \text{IK}({\bf x}_{\text{goal}}, R_{\text{goal}}, q_{\text{arm}})$, 注意seed用current joint positions
4. **Trajectory planning**: 用Ruckig库(https://github.com/pantor/ruckig)做time-optimal joint trajectory, 约束:
   - Arm: velocity=1.5, acceleration=2.0, jerk=50.0
   - Gripper: velocity=1.0, acceleration=7.0, jerk=50.0
5. **执行**: 在每个sim step里, $t = i / H_{\text{sim}}$, 查planned trajectory取target, 设置joint position和velocity target

Gripper有"小动作过滤": 当 $|g_a| < 0.01$ 时不更新目标, 防止抖动。

WidowX controller (Algorithm 2)更简单些, 没有Ruckig trajectory planning, 直接设joint position target。

物理引擎设置上有个重要细节: **SAPIEN默认用Projected Gauss-Seidel solver, 在grasping时会出现mesh penetration**, 作者切换到Temporal Gauss-Seidel solver来获得正确grasping行为。这是仿真器选择里很容易踩的坑。

---

## 4. 视觉Gap的弥合: Visual Matching vs Variant Aggregation

### 4.1 Green-Screening

步骤:
1. 从real eval video第一帧 $I_{\text{real}}$ 用online inpainting工具(https://cleanup.pictures)擦除robot和foreground objects, 得到背景
2. 在sim渲染图 $I_{\text{sim}}$ 中用ground truth segmentation mask查询foreground (robot arm + interactable objects) 的binary mask $M$
3. 合成: $I' = M \odot I_{\text{sim}} + (1-M) \odot I_{\text{real}}$

其中 $\odot$ 是element-wise乘法。这就是经典的alpha blending, 但用binary mask。

**局限性**: 固定camera (没法handle moving camera), 阴影不正确(因为sim渲染的object不会有正确的阴影投到real背景上)。

### 4.2 Texture Matching

对foreground objects做texture tuning:

1. **SAM segmentation** (Kirillov et al., https://segment-anything.com)在real image里抠出object
2. **粗略pose估计**: 把sim object import进来, 手动调整位置使segmentation mask重叠
3. **Differential rendering优化**: 用Nvdiffrast (https://github.com/NVlabs/nvdiffrast)优化object pose, 让sim渲染的segmentation mask和real的对齐
4. **Unproject texture**: 把real RGB值project回sim mesh的UV坐标
5. (Optional)用Zero123++(https://github.com/SUDO-AI-3D/zero123plus)生成novel views, 补全背面看不见的部分, 再unproject

代码: https://github.com/Jiayuan-Gu/GeTex

对于robot arm的texture: 因为visual mesh已经有texture map, 直接用GIMP的bucket-paint工具拷贝real颜色值到texture map。

**重要发现**: robot arm颜色在task执行过程中会变化(可能因为lighting angle变化), 所以作者tune了多个robot arm colors, 在eval时取平均来消去这个confounding factor。具体地Google Robot用了4个版本, WidowX因为是黑色, 跳过这一步。

### 4.3 Variant Aggregation (替代方案)

借鉴domain randomization思想:
- Base environment + 4个axis的variants
- Background, Lighting, Distractors, Table Texture
- 每个axis 2个variants, 总共覆盖ReplicaCAD场景
- 对所有variants的evaluation结果取平均

Figure 11展示了一些variant例子。

### 4.4 两种方法对比

Table I (Google Robot 6个policy checkpoint):

| Protocol | Pick Coke Can MMRV | Move Near MMRV | Drawer MMRV | Avg MMRV |
|---|---|---|---|---|
| Validation MSE | 0.412 | 0.408 | 0.306 | 0.375 |
| SIMPLER-VarAgg | 0.084 | 0.111 | 0.235 | 0.143 |
| **SIMPLER-VisMatch** | **0.031** | 0.111 | **0.027** | **0.056** |

| Protocol | Pick Coke Can r | Move Near r | Drawer r | Avg r |
|---|---|---|---|---|
| Validation MSE | 0.464 | 0.230 | 0.231 | 0.308 |
| SIMPLER-VarAgg | 0.960 | 0.887 | 0.486 | 0.778 |
| **SIMPLER-VisMatch** | **0.976** | 0.855 | **0.942** | **0.924** |

VisMatch全面优于VarAgg, 也就是**直接弥合gap比平均化gap更有效**。这有点反直觉, 因为domain randomization在sim-to-real training里通常很好用, 但在eval场景下, 我们要的是测量policy的真实performance, 引入更多variation反而模糊了signal。

---

## 5. SIMPLER Environments构建流程

### 5.1 Asset获取pipeline

**Robot URDF**:
- Google Robot: 从公开MuJoCo .mjcf转换成URDF
- WidowX: 从Interbotix ROS package直接export
- Camera intrinsics未知时用fSpy(https://fspy.io/)交互式GUI标定

**Object assets**:
- 普通rigid物体: Objaverse (https://objaverse.allenai.org)
- 不常见物体: 3D扫描 或 One-2-3-45++(https://github.com/One-2-3-45/One-2-3-45)单图重建
- Blender调整尺寸匹配real
- CoACD(https://github.com/CoACD/CoACD)生成convex collision mesh
- Uniform density: GPT-4查询材料密度, 或mass/volume
- 摩擦系数: 基于材料性质赋值

**Articulated objects**: 手工建模, 比如cabinet, 这是流程里最费人工的部分, 作者pointed out这是future work, 可以用multi-view articulated object reconstruction (Ditto, https:// Ditto-3D.github.io)

**Scene对齐**: tune robot和camera pose, 让fixed object edges和gripper初始化位置在sim和real里大致对齐

### 5.2 性能数据

- 单个environment渲染速度: **3.5k sim steps/sec** on NVIDIA 4090, 640×512分辨率
- Simulation frequency 500Hz下, 这是**7x real-time speedup**
- pip install直接装, Gym API交互

### 5.3 Tasks列表

**Google Robot** (来自RT系列):
- "pick coke can": 3 orientations × 25 positions = 75 trials
- "move {obj1} near {obj2}": 5 triplets × 2 triangle patterns × 6 source/target/distactor perms = 60 trials
- "open/close (top/middle/bottom) drawer": 9 positions × 3 drawers × 2 directions = 54 trials
- "open top drawer; place apple": 3 robot positions × 9 apple positions = 27 trials

**WidowX Bridge V2** (来自BridgeData):
- "put spoon on towel": 2 orientations × 12 positions = 24 trials
- "put carrot on plate": 同上
- "stack green block on yellow block": 2 square sizes × 12 positions = 24 trials
- "put eggplant into yellow basket": 24 trials

---

## 6. 核心实验结果

### 6.1 主结果: Sim vs Real Performance Correlation

Figure 6 (Google Robot) 和 Figure 7 (Bridge V2) 的scatter plot显示, 在大多数task上policy point都基本沿对角线分布, correlation强。

具体数据见Table IV (Google Robot):
- Pick Coke Can (VisMatch): MMRV=0.031, Pearson r=0.976
- Move Near: MMRV=0.111, r=0.855
- Open Drawer: MMRV=0.000, r=0.983
- Close Drawer: MMRV=0.123, r=0.768
- Open Drawer & Place Apple: MMRV=0.000, r=0.969

Table V (Bridge V2):
- Put Spoon on Towel: MMRV=0.000, r=0.778
- Put Carrot on Plate: MMRV=0.000, r=0.995
- Stack Block: MMRV=0.000, r=1.000 (完美!)
- Put Eggplant in Basket: MMRV=0.000, r=0.990

### 6.2 与Validation MSE的对比

这是一个非常重要的baseline: 在imitation learning里, 大家常用validation action MSE来选model。Table XII对比了两种方法:

| Task | Validation MSE (MMRV/r) | Sim Eval (MMRV/r) |
|---|---|---|
| Pick Coke Can | 0.412 / 0.464 | 0.031 / 0.976 |
| Move Near | 0.408 / 0.230 | 0.111 / 0.855 |
| Open/Close Drawer | 0.346 / 0.264 | 0.055 / 0.915 |
| Open Drawer & Place Apple | 0.265 / 0.198 | 0.000 / 0.969 |
| Put Spoon on Towel | 0.389 / -0.951 | 0.000 / 0.827 |
| Put Carrot on Plate | 0.194 / -0.342 | 0.111 / 0.575 |
| Stack Block | 0.125 / -0.857 | 0.000 / 1.000 |
| Put Eggplant in Basket | 0.366 / -1.000 | 0.000 / 0.990 |

Validation MSE的Pearson r在Bridge任务上甚至是**负的**! 这意味着validation MSE根本不能预测real performance, 甚至反着来。这是imitation learning里非常重要的发现: action prediction error和task success基本脱钩, 因为action space的local geometry和task success metric没有直接对应。

Intuition: 同一个observation, 几乎正确的action和稍微偏一点的action, MSE差很小, 但前者成功后者失败。MSE是连续的, success是binary的, 这俩之间的mapping非常non-monotonic。

### 6.3 Distribution Shift实验

作者设计了5个distribution shift axis (inspired by Xie et al., https://arxiv.org/abs/2307.03659):
- Background
- Lighting
- Distractors
- Table Texture
- Camera Pose

用RT-1 with/without image augmentation两个policy对比, 公式6:
$$\Delta\text{Success}(\text{shift}) = \frac{1}{2} \sum_{k=1}^2 (\text{Success}(\text{shift}, k) - \text{Success}(\text{base}))$$

即对每个axis的2个variants取平均变化。

Table VI结果:
- RT-1 w/o Aug: Real的 |ΔSuccess| = (background 0.028, lighting 0.083, distractors 0.111, table texture 0.389, camera pose 0.458)
- RT-1 w/o Aug: Sim的 |ΔSuccess| = (0.048, 0.057, 0.080, 0.144, 0.473)
- RT-1 w/ Aug: Real = (0.167, 0.042, 0.083, 0.167, 0.375)
- RT-1 w/ Aug: Sim = (0.123, 0.075, 0.059, 0.189, 0.394)

**关键观察**:
1. Camera pose是最大的distribution shift factor(都损失40%+), sim和real都反映了这一点
2. Lighting和distractors影响很小(都在10%以内)
3. Table texture在中间
4. Data augmentation在sim和real里都improved robustness, 趋势一致

这是一个相当强的结果: SIMPLER不只预测平均performance, 还预测**per-factor的robustness profile**。

### 6.4 预测新distribution shift的robustness

作者发现Octo-Base在sim里对robot arm texture极敏感(0% untuned vs 29.3% tuned), 而RT-1-X不敏感。基于这个sim观察, 设计了real-world novel distribution shift: 用gift wrapping paper包住real robot arm。结果(Table VIII):

| Policy | Sim Success Range | Real (Orig Arm) | Real (OOD Arm) |
|---|---|---|---|
| RT-1-X | [0.507, 0.653] | 0.760 | 0.520 |
| Octo-Base | [0.000, 0.293] | 0.293 | 0.000 |

Real world验证了sim的预测: Octo-Base对arm texture变化从0.293掉到0, RT-1-X从0.760掉到0.520。**Sim里观察到的sensitivity rank, 在real里也成立**, 这是predict novel distribution shift的强证据。

### 6.5 Ablation: System Identification效果

Table II (Pick Coke Can):

| Control Parameters | Control Loss | MMRV |
|---|---|---|
| Setting 1 (perturbed) | 0.267 | 0.070 |
| Setting 2 (perturbed) | 0.432 | 0.100 |
| SIMPLER SysID | 0.131 | 0.031 |

Control loss和MMRV单调相关, 验证了SysID的核心作用。

### 6.6 Ablation: Visual Matching各组件

Table III (Open/Close Drawer):

| Green Screen | Drawer Matching | Robot Matching | MMRV | Real-Sim Gap |
|---|---|---|---|---|
| ✗ | ✗ | ✗ | 0.087 | 0.272 |
| ✗ | ✓ | ✗ | 0.087 | 0.266 |
| ✗ | ✗ | ✓ | 0.087 | 0.272 |
| ✗ | ✓ | ✓ | 0.087 | 0.328 |
| ✓ | ✗ | ✗ | 0.087 | 0.198 |
| ✓ | ✓ | ✗ | 0.142 | 0.253 |
| **✓** | **✓** | **✓** | **0.050** | **0.136** |

**关键发现**: 
1. 单独tune drawer或robot texture没用(MMRV不变)
2. 单独green-screening降低gap但不降低MMRV
3. **必须jointly apply所有3个**: green-screening + drawer + robot, 才能同时降低MMRV和gap

作者的hypothesis: scene里不同部分appearance不一致会让policy confused。如果只有background是real-styled但前景不匹配, 反而更糟。这解释了为什么partial visual matching失败。

### 6.7 Ablation: Physical Property Gap Sensitivity

Table X(a) (Pick Coke Can): 改变coke can mass (10-80g) 和gripper friction (0.25-2.0):

所有12个组合MMRV都是0.031, Pearson r在0.957-0.990之间。**Physical property简化对评估pipeline不敏感**。这是因为即使物理参数偏了导致absolute success rate变化, relative ranking还保持。

Table X(b) (Drawer): 改变cabinet joint friction (0.0125-0.20), MMRV保持0.055, Pearson r 0.903-0.941。同样结论。

这是好消息: 不需要精确测量每个物体的mass和friction, 简化模型就够用。这极大降低了环境搭建成本。

### 6.8 Ablation: Physics Simulator Independence

Figure 10, Table XI: 在Isaac Sim(https://developer.nvidia.com/isaac-sim)上reproduce Google Robot eval。

Pick Coke Can: MMRV=0.064, r=0.973
Move Near: MMRV=0.053, r=0.865

SAPIEN和Isaac Sim结果一致, 说明**method不绑定具体simulator**。这给了framework普适性。

### 6.9 Single-Task Policy效果

Table XIII: 加入只在"Pick Coke Can"上训练的RT-1:
- Real: 0.680
- Sim: 0.403
- 整体7个policy的MMRV=0.027, r=0.959

说明SIMPLER不只对multi-task large-data policy有效, 对single-task small-data policy也work。

### 6.10 Kruskal-Wallis Test: Absolute Distribution Alignment

除了relative ranking (MMRV), 作者还检验absolute distribution。对每个policy, 用Kruskal-Wallis test比较real和sim的trial success indicators $\mathbf{r}_i = (r_{ij})$ 和 $\mathbf{s}_i = (s_{ij})$。这是非参数检验, 不假设normality。

Table XIV: VisMatch下, 大部分task/policy组合 $p \geq 0.05$, 即sim和real的success distribution无显著差异。这说明VisMatch不只是恢复了ranking, 还恢复了absolute performance distribution, 是很强的validation。

---

## 7. Limitations和Future Work

作者诚实地列出了几个限制:

1. **Rigid body only**: 软体、可变形物体、流体等没有覆盖。这是physics engine本身的限制。参考IPC (Incremental Potential Contact, https://ipc-sim.github.io)做soft body。

2. **Green-screening固定camera**: 不支持moving camera, 阴影不准确, 反射不准确。UniSim(https://unisim-cvpr2023.github.io)的neural rendering approach可能能解决。

3. **半自动pipeline**: 仍需人工curate assets, 距离全自动生成千个environment的目标还很远。需要结合text-to-3D (Objaverse-XL, https://objaverse.allenai.org), 自动texture transfer, articulated object扫描(Ditto)。

4. **没有覆盖dexterous manipulation**: in-hand rotation, bimanual等高动态任务。Mobile ALOHA(https://mobile-aloha.github.io)这类可能更难sim。

---

## 8. 我的intuition总结

### 8.1 为什么real-to-sim比sim-to-real更"容易"

Sim-to-real训练时, 要让policy在sim里学到的behavior generalize到real, 需要handle sim不模拟的物理细节, 通常是domain randomization暴力覆盖。

Real-to-sim eval反过来: policy已经在real分布上学好了, 我们只需让sim足够像real的某个specific setting。这是**单点匹配**而非**分布覆盖**, 难度低很多。

### 8.2 为什么Visual Matching > Variant Aggregation

直觉是: 你要测量policy在某个real setting下的performance, 那sim就该尽量像那个setting, 而不是绕一圈去平均化。Variant Aggregation引入额外变量(noise), 模糊了policy的真实performance signal。

### 8.3 为什么Validation MSE那么差

这是paper最重要的副产品发现之一。Imitation learning里action prediction MSE和task success脱钩, 因为:
- Action space的高维误差不一定对应task失败
- Action的某些维度critical, 某些维度redundant
- Success是binary的, MSE是连续的, mapping高度non-linear
- Compounding error: 小MSE可能仍导致trajectory严重drift

这解释了为什么社区从来不用validation loss做model selection, 而是依赖real eval, 但real eval又很贵。SIMPLER填补了这个gap。

### 8.4 MMRV的设计哲学

Pearson r要求linear, Spearman只看rank, MMRV看rank+margin。这是为评估场景量身定制的metric: 我们关心的是sim能否正确告诉practitioner "policy A比B好多少"。

设计思路: 用real margin作为rank violation的"严重程度"weight, 用max aggregation体现最坏邻居原则。这避免了小margin噪声导致metric波动。

### 8.5 SIMPLER对社区的实用价值

1. **Policy development loop加速**: 不用每次都run real robot, 用sim快速iterate
2. **Cross-paper comparison**: 不同paper的policy可以在同一SIMPLER env上公平对比
3. **Ablation studies on policies**: 比如test data augmentation的效果, 在sim里可以批量做
4. **Debugging tool**: real失败时, 用sim复现case, 诊断failure mode

### 8.6 与ManiSkill, RLBench, CALVIN等benchmark的关系

那些sim benchmark是**sim-train, sim-eval**。SIMPLER是**real-train, sim-eval**。两者互补: SIMPLER评估真实数据的policy, 不需要sim训练数据, 直接给real performance proxy。

### 8.7 未来方向猜想

- **Learnable sim**: 用neural surrogate替代手工asset curation
- **Differentiable asset creation**: 从real video自动generate textured 3D asset (Gaussian Splatting + physics)
- **Closed-loop sim parameter tuning**: 让sim自动对齐real observation
- **Foundation model in-the-loop**: 用VLM判断sim observation是否"像real"
- **Cross-embodiment SIMPLER**: 一个env里支持多种robot同时eval

### 8.8 工程上值得学习的细节

- **Ruckig** (https://github.com/pantor/ruckig)做time-optimal joint trajectory, 处理velocity/acceleration/jerk约束, 这个库工业级, 比PID简单方法好很多
- **Nvdiffrast** (https://github.com/NVlabs/nvdiffrast)做differentiable rendering, 用于texture alignment
- **CoACD** (https://github.com/CoACD/CoACD)生成convex decomposition, 是collision mesh必备
- **fSpy** (https://fspy.io)从单张图反推camera intrinsics
- **SAM** (https://segment-anything.com)做object segmentation
- **Zero123++** (https://github.com/SUDO-AI-3D/zero123plus)单图到多视图3D

### 8.9 与RT-1/RT-2/Open-X-Embodiment ecosystem的关系

SIMPLER本质是为Open-X-Embodiment时代准备的eval infra。当dataset跨多embodiment, 多task, 多lab, 用real robot eval根本无法scale。SIMPLER给出的promise: **同一套sim env可以复现性评估所有Open-X policy**, 这是policy research democratization的关键enabler。

### 8.10 对Lerrel Pinto, Chelsea Finn, Sergey Levine这些大佬的工作脉络的影响

- Lerrel Pinto的RB2 (https://arxiv.org/abs/2203.01773): distributed real eval
- TOTO benchmark (https://arxiv.org/abs/2211.12984): shared real hardware
- DROID dataset (https://droid-dataset.github.io): 大规模real data
- SIMPLER: sim-based eval

这是一条线: real eval越来越scale up的瓶颈, 最终需要sim来break out。SIMPLER是这条线上的关键节点。

---

## 9. 数学补充: 为什么Visual Matching同时要tune background + foreground

设observation $o = f(B, F)$, 其中 $B$ 是background, $F$ 是foreground features, $f$ 是policy network。

Real observation: $o_{\text{real}} = f(B^*, F^*)$
Sim observation: $o_{\text{sim}} = f(B, F)$

Policy output: $\pi(o)$, 期望 $\pi(o_{\text{sim}}) \approx \pi(o_{\text{real}})$

如果只tune $B$ 让 $B \approx B^*$, 但 $F$ 仍偏离 $F^*$, 那么policy可能依赖 $B$ 和 $F$ 的某种joint statistic (比如相对位置, contrast, semantic alignment), 单方面匹配会让这个joint statistic更偏, 输出更不稳定。所以需要joint matching。

Table III的实验数据印证了这个理论: 单方面matching时, MMRV=0.087不变, joint matching时, MMRV掉到0.050。这是emergent consistency phenomenon的实证。

---

## 10. 结语

这篇paper技术上不是大突破, 但是**methodologically极其重要**。它把"sim eval real policy"从intuition变成validated methodology, 给出了:

- 工程上可复现的pipeline (SysID + Visual Matching)
- 严谨的评估指标 (MMRV)
- 跨simulator, cross-embodiment, cross-policy的验证
- 开源的environment collection

对整个robot manipulation community是infrastructure级别的贡献。我expect接下来几年会有大量paper用SIMPLER作为eval backbone, 大量新policy用SIMPLER iterate design。这种"evaluate, iterate, ship"的rapid cycle, 是任何data-driven field scale up的前提。

References:
- Project: https://simpler-env.github.io
- Code: https://github.com/simpler-env/SIMPLER
- SAPIEN: https://sapien.ucsd.edu
- Isaac Sim: https://developer.nvidia.com/isaac-sim
- Open-X-Embodiment: https://robotics-transformer-x.github.io
- BridgeData V2: https://bridgedata.github.io/bridge-v2
- RT-1: https://robotics-transformer.github.io
- Octo: https://octo-models.github.io
- Objaverse: https://objaverse.allenai.org
- SAM: https://segment-anything.com
- CoACD: https://github.com/CoACD/CoACD
- Ruckig: https://github.com/pantor/ruckig
- Nvdiffrast: https://github.com/NVlab/nvdiffrast
- Zero123++: https://github.com/SUDO-AI-3D/zero123plus
- One-2-3-45++: https://github.com/One-2-3-45/One-2-3-45
- Xie et al. "Decomposing the Generalization Gap in Imitation Learning": https://arxiv.org/abs/2307.03659
- Kadian et al. "Sim2Real Predictivity": https://arxiv.org/abs/1912.06321
