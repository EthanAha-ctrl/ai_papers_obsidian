---
source_pdf: GAUGE A Measurement-Grounded Benchmark for Physical Fidelity in Simulation
  Engines and.pdf
paper_sha256: 57314698a5f496a1f4f2368272cae2f517b7bac06e1d8a77091a265cf41b180c
processed_at: '2026-08-19T08:39:15-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 GAUGE

## 这篇paper到底在干嘛

一句话：**给"物理模拟器"和"AI生成的video world model"做一个"物理准不准"的考试卷，而且这考试有标准答案——来自真实世界的高精度实验测量。**

你训练机器人，要仿真。仿真有两种：一种是传统物理引擎（Isaac Sim、Genesis、Newton），一种是最近火起来的video generation model（Cosmos、Wan、Seedance、Genie这些）。问题是：**你怎么知道它们模拟的物理对不对？**

以前的做法基本是"看着像不像"——人打分，或者学个evaluator。GAUGE说这不行，看着像不等于真对。

---

## 怎么造"考题"

他们搭了个2米×2米×2米的motion capture棚子，16台红外相机，180Hz，亚毫米精度。然后设计了22种实验：

**8个刚体实验**：
- 从斜坡上掉下来（测碰撞、friction）
- 斜坡上滑木块（测static/kinetic friction）
- 转盘上放个方块（非惯性系里的摩擦）
- 球弹跳（测restitution）
- Newton's cradle（动量传递）
- 单摆（周期运动、能量守恒）

**6个布料实验**：
- 拉伸、弯曲、抖动、塞过洞、转盘上拖、桌布抽走

**7个3D可变形物体实验**：
- foam的拉伸、压缩、剪切、扭转、弯曲、stick-on-plane、悬臂梁

每种实验换材料：刚体用wood/plastic/metal，布料用6种fabric（rayon, satin, uniform cloth, Oxford, synthetic leather, nylon），3D用soft/hard foam。

**关键**：他们不only拍video，还**用仪器测了所有物理参数**——friction coefficient用斜面法测，restitution用air-bearing rail消除friction后测，布料的stretch/bending stiffness用Style3D的仪器测，3D的Young's modulus和Poisson's ratio用DIC（Digital Image Correlation）测。

每个实验做20次repetition，存trajectory + 校准的物理参数 + uncertainty。

---

## 给物理引擎打分怎么打

他们rebuild了Isaac Sim (PhysX)、Genesis (XPBD/PBD)、Newton (MuJoCo/VBD)在14个任务上的scene，用GAUGE测出来的真实参数配置，其他保持default，看"out-of-the-box"的表现。

### 核心metric

$$E_{\text{RMSE}} = \sqrt{\frac{1}{T}\sum_{t=1}^{T}\|g_{\text{sim}}(t) - \bar{g}_{\text{real}}(t)\|_2^2}$$

其中$g(t)$是"generalized trajectory"——刚体用3D position $\mathbf{P}(t)$，布料用marker点的Gaussian curvature向量$\mathbf{K}(t)$，可变形体用mesh face的面积向量$\mathbf{A}(t)$。

**为什么用Gaussian curvature**？因为布料是distributed deformation，你不可能用一个position描述。他们在每个marker点上用angle deficit method算离散Gaussian curvature：

$$k_i = \Theta_i - \sum \theta_i$$

$\Theta_i$是reference full angle（interior点$2\pi$，edge点$\pi$，corner点$\pi/2$），$\theta_i$是adjacent triangle的interior angle。这个$k_i$本质是angular deficit，捕获局部是否"皱了"。

结果都normalize by baseline，**值<1意味着比真实实验trial之间的variation还小**。

### 结果怎么样

**简单的刚体任务还行**：
- Turntable: Isaac Sim normalized RMSE=0.17，比real-to-real的variation还小
- Slope slider: Genesis 0.58，Isaac Sim 0.61，都OK

**一旦碰到impulsive contact就崩**：
- Bouncing ball: 最好的Isaac Sim也是baseline的**15.63倍**
- Newton's cradle: Isaac Sim和Newton的**Longest Stationary Duration = 0**（完全没产生应有的静止间隔），Momentum Transfer Efficiency只有0.20和0.26（理想是~1）。Genesis直接崩了，产生不出valid rollout
- Pendulum: period还凑合（1.10, 1.09），但Energy Loss从-0.041到0.034，说明数值积分在累积误差

**布料和3D变形更惨**：
- Textile stretching: 还行，≤1
- Textile bending: 最好RMSE=7.94（Isaac Sim），已经是baseline的8倍
- **Textile flinging**: Isaac Sim RMSE=128.26（baseline的**128倍**！），Genesis最好也要8.54。说明quasi-static对得好不代表rapid deformation对
- 3D foam: 最好也约一个order of magnitude above baseline

**核心结论**：**没有一个引擎是universal winner**。Isaac Sim强在刚体contact，Genesis强在dynamic textile和3D，Newton强在selected deformation case。不同regime的failure mode完全不同。

---

## 给video world model打分——这才是有意思的部分

他们测了6个model：Cosmos3-Nano, Cosmos3-Super-I2V, Wan-2.2, Wan-2.7, Seedance 2.0, Genie 3。每个model喂同样的initial frame + text prompt，让它生成future video。然后用SAM3做segmentation + tracking，把object的centroid trajectory提取出来，跟真实物理比。

### 他们发现了什么

**最striking的发现：trajectory的"形状"对了，但"物理参数"全错。**

举个例子——slope slider任务（木块从斜坡滑下，理想是uniformly accelerated motion，$s = \frac{1}{2}at^2$，所以$s$ vs $t^2$应该是直线）：

他们定义QFI（Quadratic Form Improvement）来测"是不是直线":

$$\text{QFI} = \frac{(\text{SSE}_L - \text{SSE}_Q)/1}{\text{SSE}_Q/(N-3)}$$

就是比较linear model vs quadratic model的残差。低QFI意味着linear就够（形状对），高QFI意味着有额外曲率（形状错）。

**结果**：
- Cosmos3-Super-I2V在wood上QFI=13.61（形状还行），但fitted acceleration = 0.088 m/s²，真实值2.58 m/s²，**差30倍**
- Bouncing ball: 最好的QFI=12.50（Cosmos3-Super-I2V + negative prompt），但acceleration = 0.088 m/s²，**只有g的0.9%**
- Seedance 2.0给出最接近的acceleration 1.84 m/s²，仍然比9.81低81%
- Genie 3甚至给出negative acceleration（轨迹反着走）

**Pendulum更明显**：
- Wan-2.2和Genie 3都达到R²=0.99（damped sinusoid拟合极好，**形状完美**）
- 但period是1.93s和1.90s，真实1.06s——**慢了快一倍**
- 最接近的Wan-2.7 period=1.83s，仍然长73%
- 更诡异的是fitted damping coefficient的sign在不同model间不同——有的model产生amplitude growth而非decay

**Newton's cradle**：6/10个配置产生不了valid sequence。最好的Wan-2.2 MTE=0.76，意味着momentum只传了76%。

### 这说明什么

**Video world model学到的是"motion pattern"，不是"physical law"**。

它能生成一条看起来像自由落体的trajectory（在$t^2$空间接近linear），但完全没学到"gravity = 9.81 m/s²"这个尺度。它能生成看起来像pendulum的oscillation，但period是错的。

这就是为什么GAUGE坚持要**分开测equation form和parameter accuracy**——只看visual plausibility或只看trajectory distance会被骗。

### Negative prompt的诡异效果

他们对Cosmos3和Wan系列还试了加negative prompt（描述"物理不对的样子"）。结果：

- Cosmos3-Super-I2V bouncing ball QFI: 270.69 → 12.50（**大幅改善**）
- Cosmos3-Super-I2V wood slope slider QFI: 13.61 → 569.36（**大幅恶化**）
- Wan-2.2加了negative prompt才能生成Newton's cradle（之前直接崩）

**Negative prompt对physics没有consistent effect**。这强烈暗示：evaluation必须用fixed prompt template + report paired result，不能从single favorable generation下结论。

---

## 最技术的一段：怎么把测量参数喂给不同solver

这部分是Appendix A.2.2，是paper里最hardcore的。

问题：你测了布料的$K_s$（stretch stiffness）和$K_{\text{sh}}$（shear stiffness），但Isaac Sim、Genesis、Newton用的constitutive model和discretization完全不同，你不能直接把同一个数字塞进去。

**Isaac Sim (PhysX)**：用isotropic membrane plane stress model
$$K_s = Et, \quad K_{\text{sh}} = \frac{Et}{2(1+\nu)}$$
反解出$E$和$\nu$。Bending用`surfaceBendStiffness`，对他们的固定regular square grid with diagonal triangulation：
$$\text{surfaceBendStiffness} \simeq \frac{K_\theta}{4t^3}$$
**那个1/4是来自两种edge class的averaging，不是mesh-independent identity**。

**Genesis (XPBD)**：用discrete distance/dihedral constraint，per-constraint compliance就是stiffness的倒数：
$$\text{stretch\_compliance} = \frac{1}{K_s}, \quad \text{bending\_compliance} = \frac{1}{K_\theta}$$
但XPBD的effective response还取决于substep size、relaxation factor、projection iteration数，所以这只是initialization。而且Genesis没有独立$K_{\text{sh}}$ input channel。

**Newton (VBD, stable Neo-Hookean)**：用Lamé parameters $\mu, \lambda$。匹配free-transverse uniaxial test和engineering-shear test：
$$K_{\text{sh}} = \mu, \quad K_s = \frac{4\mu(\lambda+\mu)}{\lambda + 2\mu}$$
反解出$\mu, \lambda$。当measured $K_s/K_{\text{sh}}$超出physical range（$2 < \text{ratio} < 4$对应positive Poisson's ratio），用$K_s/2.7$的engineering prior兜底——**这个2.7是fixed engineering prior，不是理论常数**。

**Insight**：不同solver的参数mapping本身就是effective approximation。你测的macroscopic参数到solver internal coefficient之间有irreducible的model mismatch，这是sim-to-real gap的一部分。

---

## 这篇paper的真正价值

### 1. 它定义了"physical fidelity"应该是多维的

以前大家说"物理准不准"就是一个scalar。GAUGE说至少要分三个维度：
- **Equation form**: trajectory符不符合expected数学结构
- **Parameter accuracy**: fitted parameter对不对
- **Temporal stability**: 长时间horizon的energy/amplitude行为

一个model可以在第一个维度满分，第二第三维度全挂。Visual plausibility和single trajectory distance都capture不了这个。

### 2. 它给embodied intelligence researcher一个明确warning

如果你用video world model做implicit simulator或policy training environment——**小心**。你的policy可能在learn一个period慢一倍、gravity小30倍的"假物理"。它生成的trajectory看起来合理，但underlying dynamics的scale完全错。

如果你用physics engine做sim-to-real——**选engine要按task regime**。Isaac Sim适合rigid contact，Genesis适合dynamic textile，没有universal winner。impulsive contact和rapid deformation是所有人的weak spot。

### 3. 它是第一个cross-regime + measurement-grounded的benchmark

以前要么只测刚体（RIGIDBENCH），要么只测布料（RGBench, Cloth Sim-to-Real），要么只测volumetric（PokeFlex），要么用synthetic ground truth。GAUGE是第一个把rigid + textile + 3D deformable放在一个standardized framework里，全部用real-world measurement校准，同时测engine和generative WM的。

### 4. 它暴露了video WM最根本的limitation

**形式正确但参数错误**——这个发现我觉得是paper最深的insight。它说明当前video generation model的training objective（基本是perceptual loss）不足以learn physical law的quantitative aspect。model学到了"自由落体应该加速"这个qualitative pattern，但完全没学到"加速度是9.81"这个quantitative scale。

这对未来world model的训练有明确implication：需要physics-grounded loss、auxiliary physical parameter supervision、或者把physical law作为inductive bias硬编码进去。纯visual training不够。

---

## 一句话总结

**GAUGE搭了个精密的物理实验台，用真实测量去考物理引擎和AI生成video model，发现两者在"看起来对"和"真对"之间存在系统性gap——尤其是AI生成的video，形式像但参数错，这个gap目前没人好好量化过，GAUGE把它量化了，并指出了未来该往哪使劲。**

Project page: https://internrobotics.github.io/GAUGE/

---

# GAUGE: 真实测量驱动的物理保真度Benchmark深度解析

## 1. 核心动机：为何需要GAUGE

Embodied intelligence的real-to-sim-to-real pipeline严重依赖底层环境的物理保真度。SIMPLER等work已经证明carefully aligned的simulation实验可以可靠预测robot policy的real-world performance。问题在于：**高visual fidelity不implies准确physical dynamics**。一个simulator可能产生realistic-looking的observation，同时misrepresent motion、contact或deformation，导致policy exploit simulation artifact或给出错误的ranking。

现有benchmark的根本缺陷：
- Physics-IQ / VideoPhy-2: 主要依赖human judgment或learned evaluator，给出perceptual score，无法localize trajectory或physical parameter的error
- IRIS: 虽然提供governing-equation label和parameter target，但collision coupling是effective quantity而非measured contact parameter
- PokeFlex: 报告object-level effective stiffness而非material elastic modulus
- RGBench: 限制在cloth regime
- WorldBench / RIGIDBENCH: coverage或coarse judgment的trade-off

GAUGE的关键区别在于：它是第一个将**cross-regime物理参数标注**（rigid-body contact + cloth constitutive + volumetric soft-body mechanics）与**dedicated protocols for both engines and video WMs**结合的standardized real-world benchmark。Table 1的comparison很清楚展示了这一点——只有GAUGE在所有维度（Real dyn., Exp. param., R, C, S, Engine, Video WM, Eq. form, Param. acc.）全部打√。

## 2. Dataset Pipeline：22个任务族的设计哲学

### 2.1 任务族分类（Table 2）

| Category | Tasks | Target Physical Process |
|---|---|---|
| Rigid body (8) | slope contact, nonsmooth contact, slope slider, turntable, bouncing ball, Newton's cradle, wall breaking, pendulum | collision, friction, momentum/energy conservation, oscillation |
| Quasi-1D (1) | rope winding | self-collision, stretch modulus |
| Quasi-2D textile (6) | textile stretching, bending, flinging, funnel, rotating ball, tablecloth pulling | stretch/bending modulus, locally high acceleration, friction |
| 3D deformable (7) | foam stretching, compression, shearing, twisting, bending, stick-stack, cantilever beam | Young's modulus, Poisson's ratio, coupling with large stiffness ratio |

Material variant增加divers性：
- Rigid: wood, plastic, metal
- Textile: rayon (R), satin (S), uniform cloth (U), Oxford fabric (O), synthetic leather (L), nylon taslan (N) — 六种fabric覆盖不同的warp/weft structure
- 3D deformable: soft/hard foam和rubber

### 2.2 Data Acquisition的精度

Motion capture system: 16台NOKOV Mars9H红外相机，180 Hz，sub-millimeter 3D localization。2m×2m×2m的capture volume，三高度层级部署消除occlusion。

关键设计：marker的mass negligible（6mm retroreflective sphere约0.5g），不会影响object dynamics。对于deformable object，adjacent marker连接形成triangulated surface mesh，系统重建local skeletal structure。

Parameter calibration方法论（Section A.2）:
- **Friction coefficient**: inclined plane method，测量sliding velocity $v(t)$，$\mu = (g\sin\alpha - \dot{v})/(g\cos\alpha)$，其中$g$是gravitational acceleration，$\alpha$是slope inclination angle
- **Restitution coefficient**: inclined air-bearing rail消除friction，$e = v_{\text{out}}/v_{\text{in}}$
- **Textile**: Style3D stretch and bending measurement instruments，warp/weft/diagonal三方向
- **3D deformable**: compression test + Digital Image Correlation (DIC)测Young's modulus和Poisson's ratio

## 3. Physics Engines Pipeline技术细节

### 3.1 模拟器选择和Solver对应

| Object class | Isaac Sim | Genesis | Newton |
|---|---|---|---|
| Rigid body | PhysX | Native solver | MuJoCo |
| Textile | Surface FEM | PBD | VBD |
| 3D deformable | FEM | Explicit MPM | Implicit MPM |

注意Genesis用的是XPBD-based cloth solver（参见[Genesis world repo](https://github.com/Genesis-Embodied-AI/genesis-world)），Newton用VBD cloth implementation（[Newton physics repo](https://github.com/newton-physics/newton)），Isaac Sim用NVIDIA OmniPhysics的Surface Deformable Bodies（[OmniPhysics docs](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/deformables/deformable_bodies.html)）。

### 3.2 Generalized Trajectory表示（Eq. 1）

$$g(t) = \begin{cases} \mathbf{P}(t) \in \mathbb{R}^3, & \text{rigid bodies} \\ \mathbf{K}(t) \in \mathbb{R}^{N_m}, & \text{textiles} \\ \mathbf{A}(t) \in \mathbb{R}^{N_f}, & \text{deformable bodies} \end{cases}$$

- $\mathbf{P}(t)$: rigid body的3D position
- $\mathbf{K}(t)$: textile marker上的Gaussian curvature向量，维度$N_m$（marker数）
- $\mathbf{A}(t)$: deformable body mesh face的面积向量，维度$N_f$（face数）

**为何这样设计**: 不同object class的物理state space维度和语义完全不同。Position对rigid body足够（6-DoF pose可以单独处理），但textile和deformable body需要distributed representation。Gaussian curvature捕获textile的局部弯曲状态，face area捕获volumetric deformation的局部体积变化。

### 3.3 Gaussian Curvature计算（Eq. 13-14, Appendix B.1）

基于discrete Gauss-Bonnet theorem的angle deficit method。对每个vertex $i$，adjacent triangle的interior angle:

$$\theta = \arccos\left(\frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}\right)$$

其中$\mathbf{a}, \mathbf{b}$是vertex $i$到该triangle另两个vertex的edge vector。

Discrete Gaussian curvature:
$$k_i = \Theta_i - \sum \theta_i$$

$\Theta_i$是reference full angle，根据vertex的topological position：interior点$2\pi$，edge点$\pi$，corner点$\pi/2$。这个$k_i$本质是angular deficit（单位rad），未normalize by local area，所以不是严格意义的Gaussian curvature（length$^{-2}$维度），但作为相对指标足够。

### 3.4 Evaluation Metrics（Eq. 2）

$$E_{\text{RMSE}} = \sqrt{\frac{1}{T}\sum_{t=1}^{T}\|g_{\text{sim}}(t) - \bar{g}_{\text{real}}(t)\|_2^2}$$

$$E_{\text{DTW}} = \frac{1}{|\pi^*|}\sum_{(i,j)\in\pi^*}\|g_{\text{sim}}(i) - \bar{g}_{\text{real}}(j)\|_2$$

$\pi^*$是optimal dynamic time warping alignment path。RMSE是frame-aligned error，DTW允许monotonic temporal alignment——对oscillatory或phase-shifted motion更fair。

**Baseline normalization**: Table 3中的simulator值都normalize by baseline mean。Within-trial RMSE = $\text{mean}_i[E_{\text{RMSE}}(g_{\text{real},i}, \bar{g}_{\text{real}})]$，per-trial RMSE std = $\text{std}_i[E_{\text{RMSE}}(g_{\text{real},i}, \bar{g}_{\text{real}})]$。Normalized值<1意味着sim-to-real的discrepancy小于real-to-real的trial-to-trial variation。

### 3.5 Pendulum/Cradle的Task-Specific Metrics

**Longest Stationary Duration (LSD)**（Eq. 15）:
$$\text{LSD} = \max_s\{|S|/f_s\} \quad \text{s.t.} \quad \max(z_S) - \min(z_S) \leq \epsilon_z, \quad |\nu_S| \leq \nu_{\text{th}}$$

$S$是连续满足stationary condition的frame set，$f_s$是sampling frequency，$\epsilon_z = 0.5$mm，$\nu_{\text{th}} = 5$mm/s。这个metric专为Newton's cradle设计——ideal impulse-like motion应该有明显的stationary interval。

**Momentum Transfer Efficiency (MTE)**（Eq. 16）:
$$\text{MTE} = \sqrt{\frac{\Delta H_{\text{out}}}{\Delta H_{\text{in}}}}$$

$\Delta H_{\text{in}}$是incoming ball的initial height difference，$\Delta H_{\text{out}}$是outgoing ball的max swing height difference，都相对于equilibrium position。Ideal lossless equal-mass cradle的MTE ≈ 1。

**Energy Loss (EL)**（Eq. 17）:
$$\text{EL} = \frac{E_{\text{start}} - E_{\text{end}}}{E_{\text{start}}}$$

在pendulum oscillation的peak position计算（kinetic energy完全转化为gravitational potential energy）。EL<0表示数值积分error导致系统能量unphysical增长。

### 3.6 Physics Engine实验结果分析（Table 3）

#### Rigid body的关键发现

Isaac Sim在slope contact、nonsmooth contact、turntable motion上error最低。Turntable的normalized RMSE=0.17、DTW=0.61，远小于within-trial variation。Genesis在slope-slider上最优（0.58 RMSE, 0.69 DTW）。

**Demanding event暴露巨大gap**:
- Bouncing ball: 即使最好的Isaac Sim，RMSE也是baseline的15.63倍，DTW是5.58倍。这指向impulsive contact resolution的根本困难
- Newton's cradle: Isaac Sim和Newton的LSD都是0（完全无法产生stationary interval），MTE只有0.20和0.26。Genesis甚至产生不了valid rollout
- Pendulum: 频率相对容易（normalized PD 1.10/1.09），但EL从-0.041到0.034，所有engine都有能量drift

**核心insight**: 匹配low-frequency motion不implies准确impact resolution或long-horizon energy behavior。

#### Textile和deformable body的regime-dependent fidelity

- Textile stretching: Isaac Sim和Newton的normalized error ≤ 1（接近baseline variation）
- Textile bending: 最好的RMSE=7.94，DTW=11.78（Isaac Sim），已经远超baseline
- **Textile flinging是最大failure mode**: Genesis最低RMSE=8.54，Isaac Sim达到128.26。这contrast说明quasi-static agreement无法predict rapid spatially varying deformation下的fidelity
- 3D deformable: Genesis在stretching/shearing/twisting上最低error，Newton在bending上最优。但即便最好也约比baseline高一个数量级

#### 没有universal winner

Isaac Sim强在rigid contact，Genesis强在dynamic textile和大多数deformable body，Newton强在selected deformation case。**Complementary solver strength**意味着不同engine的failure mode不同，选engine要按task regime。

## 4. World Models Pipeline：Law Form vs Parameter Accuracy

### 4.1 评估的6个模型

| Model | Type | Access |
|---|---|---|
| Cosmos3-Nano | I2V | 1×RTX 4090, Diffusers |
| Cosmos3-Super-I2V | I2V | 8×RTX 4090, Diffusers ([Cosmos 3 paper](https://arxiv.org/abs/2606.02800)) |
| Wan-2.2 | I2V | 8×RTX 4090, official script ([Wan paper](https://arxiv.org/abs/2503.20314)) |
| Wan-2.7 | I2V | DashScope API |
| Seedance 2.0 | I2V | Volcano Engine Ark API ([Seedance 2.0 paper](https://arxiv.org/abs/2604.14148)) |
| Genie 3 | Interactive WM | Project Genie interface ([Genie paper](https://arxiv.org/abs/2402.15391), [World Action Models](https://arxiv.org/abs/2602.15922)) |

### 4.2 Trajectory Recovery via SAM3

[Segment Anything with Concepts (SAM3)](https://arxiv.org/abs/2511.16719)对generated video做segmentation和tracking。对每帧$t$产生binary mask，foreground pixel的centroid:

$$c_x(t) = \frac{1}{N}\sum_{i=1}^{N}x_i, \quad c_y(t) = \frac{1}{N}\sum_{i=1}^{N}y_i$$

已知object dimension提供image-to-world scale $c$（m/pixel），centroid sequence转换为2D trajectory。

### 4.3 三个核心Metrics

#### Dynamic Error (DE)（Eq. 3）

$$\text{DE} = \frac{1}{T}\sum_{t=1}^{T}|m\ddot{x}_t - f_t|$$

$T$是frame数，$m$是object mass，$x_t$是position，$f_t$是net external force at frame $t$。对turntable task，$f_t$的最大值是friction limit $\mu mg$，DE具体形式见Eq. 31-32：

$$F_{\text{fric,max}} = \mu m g$$
$$e_i = [ma_i - F_{\text{fric,max}}]_+$$
$$\text{DE} = \frac{1}{M}\sum_{i=1}^{M}e_i$$

$[z]_+ = \max(0, z)$是positive-part operator。DE单位是Newton，零意味着required force不超过friction limit。

#### Coefficient of Determination $\mathbb{R}^2$（Eq. 44）

对pendulum的damped oscillation fit:

$$\hat{\theta}(t) = \theta_{\text{eq}} + e^{-\beta t}[C\cos(\omega_d t) + D\sin(\omega_d t)]$$

$\theta_{\text{eq}}$是equilibrium offset，$\beta$是damping coefficient，$\omega_d$是damped angular frequency。拟合时$\beta$不强制为正，所以nonphysical amplitude growth不会被hide。

$$R^2 = 1 - \frac{\sum_{i=1}^{N}(\theta_i - \hat{\theta}_i)^2}{\sum_{i=1}^{N}(\theta_i - \bar{\theta})^2}$$

#### Quadratic Form Improvement (QFI)（Eq. 28）

对slope slider和bouncing ball，理想uniformly accelerated motion下displacement $s$与$t^2$应linear。Linear model:

$$\hat{s}_i = b + ku_i, \quad u_i = t_i^2, \quad a = 2k$$

Quadratic model:
$$\hat{s}_i^Q = b_Q + k_Q u_i + c_Q u_i^2$$

$$\text{QFI} = \frac{(\text{SSE}_L - \text{SSE}_Q)/1}{\text{SSE}_Q/(N-3)}$$

这是nested-model F statistic。低QFI意味着$t^4$项提供little additional explanatory power（ideal constant acceleration的signature），高QFI揭示systematic curvature。

### 4.4 World Model实验结果分析（Table 4和Fig. 4）

#### 关键发现1: Law form和parameter accuracy是不同能力

对slope sliding:
- Wood: 最低QFI是Cosmos3-Super-I2V（13.61），但最接近的acceleration是Cosmos3-Nano的$2.06$ m/s²（baseline $2.58$ m/s²）
- Plastic: 最低QFI是Seedance 2.0（8.69），但最接近的acceleration只有$0.75$ m/s²（baseline $2.57$ m/s²）
- Metal: 最低QFI是Cosmos3-Nano（4.41），最接近acceleration是Genie 3的$0.43$ m/s²（baseline $2.67$ m/s²）

**Ranking在不同material间lack consistency**，更重要的是**lowest QFI model不是best acceleration model**。Fig. 4(a)可视化：多条trajectory在$t^2$空间近似linear，但slope差距巨大。

#### 关键发现2: Bouncing ball的extreme case

Cosmos3-Super-I2V with negative prompt: QFI=12.50（最低），但fitted acceleration = $0.088$ m/s²，**只有gravity的0.9%**。

Seedance 2.0给出最接近的acceleration $1.84$ m/s²，仍然比$g \approx 9.81$ m/s²低81%。

Genie 3甚至给出negative acceleration $-0.052$ m/s²（trajectory反向）。

Fig. 4(b)清楚展示了：generated motion可以在short interval内resemble uniformly accelerated trajectory，同时encode incorrect physical scale。这直接论证了equation structure和parameter accuracy必须分开evaluate。

#### 关键发现3: Momentum transfer的failure

10个model-prompt配置中6个无法产生valid Newton's cradle sequence。Wan-2.2 with negative prompt达到最高MTE=0.76，仍远低于ideal的~1。Wan-2.7的0.55/0.48，Seedance 2.0只有0.24。

#### 关键发现4: Pendulum的period错配

Wan-2.2和Genie 3都达到$R^2 = 0.99$（damped sinusoid拟合极好），但period分别是1.93s和1.90s，而baseline是1.06s。

最接近的period是Wan-2.7的1.83s，仍然比real period长73%。

**Insight**: 一个generated sequence可以look periodic、achieve high fitting score，同时violate underlying temporal parameter和energy behavior。Fig. 4(c)还显示fitted damping的sign在不同model间不同——有些model产生amplitude growth而非decay。

#### 关键发现5: Negative prompt的sensitivity

Negative prompt没有consistent improvement：
- Cosmos3-Super-I2V bouncing ball QFI: 270.69 → 12.50（大幅改善）
- Cosmos3-Super-I2V wood slope slider QFI: 13.61 → 569.36（大幅恶化）
- Wan-2.2使Newton's cradle能产生valid rollout（之前无法生成）

Direction和magnitude依赖于model和task。这强烈建议physics evaluation应该用fixed prompt template、report paired result，不能从single favorable generation下结论。

## 5. Textile Parameter Mapping的工程细节（Appendix A.2.2）

这部分是paper最technical的section之一，展示了如何将measured macroscopic参数映射到不同solver的internal coefficient。

### 5.1 Isaac Sim (PhysX)

Isotropic membrane under plane stress的free-transverse uniaxial和shear stiffness:

$$K_s = Et, \quad K_{\text{sh}} = \frac{Et}{2(1+\nu)}$$

$E$是Young's modulus，$t$是thickness，$\nu$是Poisson's ratio。反解：

$$\nu = \frac{K_s}{2K_{\text{sh}}} - 1, \quad E = \frac{K_s}{t}$$

当$\nu$超出physical range时用fallback（Eq. 6）。

PhysX的bending通过`surfaceBendStiffness`参数，resulting edge stiffness $\propto t^3$。对固定regular square grid with diagonal triangulation:

$$\text{surfaceBendStiffness} \simeq \frac{K_\theta}{4t^3}$$

$K_\theta$是discrete bending hinge的torque-per-angle stiffness。Factor $1/4$来自这个triangulation的两种edge class的averaging，**不是mesh-independent identity**（引用[Discrete shells, Grinspun et al. 2003](https://doi.org/10.2312/SCA03/062-067)）。

### 5.2 Genesis (XPBD)

Genesis用discrete distance和dihedral constraint。Per-constraint compliance:

$$\text{stretch\_compliance} = \frac{1}{K_s}, \quad \text{bending\_compliance} = \frac{1}{K_\theta}$$

注意这些reciprocal **不是** exact mesh-independent conversion。Effective response也取决于mesh、substep size、relaxation factor和projection iteration数（[PBD original paper, Müller et al. 2007](https://doi.org/10.1016/j.jvcir.2007.01.005), [XPBD, Macklin et al. 2016](https://doi.org/10.48550/ARXIV.2011.08985)）。Genesis的textile model没有独立$K_{\text{sh}}$ input channel。

### 5.3 Newton VBD (Stable Neo-Hookean)

Newton的VBD membrane用stable Neo-Hookean elasticity（[Stable Neo-Hookean flesh simulation, Smith et al. 2018](https://doi.org/10.1145/3180491)）。Input coefficient对应small-strain Lamé parameters$\mu, \lambda$。Internal stable Neo-Hookean: $\mu_{\text{NH}} = \mu$, $\lambda_{\text{NH}} = \lambda + \mu$。

Matching free-transverse uniaxial test和engineering-shear test:

$$K_{\text{sh}} = \mu, \quad K_s = \frac{4\mu(\lambda+\mu)}{\lambda + 2\mu}$$

Newton coefficient:
$$\text{tri\_ke} = \tilde{K}_{\text{sh}}, \quad \text{tri\_ka} = \frac{2\tilde{K}_{\text{sh}}(K_s - 2\tilde{K}_{\text{sh}})}{4\tilde{K}_{\text{sh}} - K_s}$$

其中$\tilde{K}_{\text{sh}}$有fallback（Eq. 11）: 当$2 < K_s/K_{\text{sh}} < 4$（positive-$\nu$ regime）用measured值，否则用$K_s/2.7$（fixed engineering prior，不是theoretical constant）。

Bending kernel用effective hinge coefficient $k_e = \text{edge\_ke} \cdot \ell_e$:
$$\text{edge\_ke} = \frac{K_\theta}{\bar{\ell}_e}$$

$\bar{\ell}_e$是nearly uniform mesh的mean rest-edge length。

## 6. 建立Intuition的核心Take-away

### 6.1 物理保真度是多维的

GAUGE的核心贡献是把"physical fidelity"分解为至少三个独立维度：
1. **Equation form consistency**: trajectory是否符合expected mathematical structure（$s \propto t^2$, $\theta \propto e^{-\beta t}\cos(\omega_d t)$等）
2. **Parameter accuracy**: fitted parameter是否match measured value（acceleration, period, MTE等）
3. **Temporal stability**: 长horizon的energy和amplitude behavior是否physical

一个model可以在维度1上score很高，同时在维度2、3上完全错。这就是为什么perceptual plausibility或single trajectory distance都不够。

### 6.2 Sim-to-real gap是regime-specific的

Physics engine没有universal winner。Isaac Sim的PhysX在rigid contact上calibration最好，但textile flinging上RMSE=128.26（baseline的128倍）。Genesis的XPBD在rapid textile motion上相对好，但Newton's cradle产生不了valid rollout。这种complementarity意味着embodied intelligence的researcher需要按task选engine，或者hybrid pipeline。

### 6.3 Cross-regime parameter grounding是难点

不同solver的constitutive model和discretization差异巨大。同一个measured $K_s, K_{\text{sh}}, K_\theta$无法直接transfer成identical numerical coefficient。GAUGE的mapping（Eq. 4-12）是effective isotropic initialization，明确标注了mesh-dependent factor（如$1/4$）和engineering prior（如2.7）。这些conversion本身引入了误差源，需要在interpretation中考虑。

### 6.4 Generative WM的"形式正确但参数错误"现象

这是paper最striking的finding。Video WM似乎学到了physical law的structural form（low QFI, high $R^2$），但完全没有recover正确的physical scale。Possible explanation:
- Training objective是perceptual loss，不直接penalize physical parameter
- Model学到的是"motion pattern"而非"physical law"
- Timescale的miscalibration（pendulum period长73%）暗示model对temporal dimension的encoding有systematic bias

这对未来world model的training有重要implication：需要physics-grounded loss或auxiliary physical parameter supervision，单纯visual loss不够。

### 6.5 Limitations和future work

Paper坦诚承认limitation：
- Material和parameter range仍limited（同nominal categorymaterial因surface treatment、manufacturing process不同而异）
- World model track只focus on rigid body的2D image trajectory，对textile和deformable body的3D distributed deformation、self-occlusion、self-contact无能为力
- Future需要3D point cloud/mesh-based metric：local strain, area/volume change, Gaussian/mean curvature, bending energy, geodesic distortion, self-contact, oscillation mode, damping, energy dissipation

## 7. Reference

- [Project page: GAUGE](https://internrobotics.github.io/GAUGE/)
- [Genesis world repo](https://github.com/Genesis-Embodied-AI/genesis-world)
- [Newton physics repo](https://github.com/newton-physics/newton)
- [NVIDIA OmniPhysics Surface Deformable Bodies](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/deformables/deformable_bodies.html)
- [Discrete shells (Grinspun et al. SCA 2003)](https://doi.org/10.2312/SCA03/062-067)
- [Position Based Dynamics (Müller et al. 2007)](https://doi.org/10.1016/j.jvcir.2007.01.005)
- [XPBD (Macklin et al. 2016)](https://doi.org/10.48550/ARXIV.2011.08985)
- [Stable Neo-Hookean flesh simulation (Smith et al. 2018)](https://doi.org/10.1145/3180491)
- [Cosmos 3 (NVIDIA 2026)](https://arxiv.org/abs/2606.02800)
- [Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/abs/2503.20314)
- [Seedance 2.0](https://arxiv.org/abs/2604.14148)
- [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391)
- [SAM3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719)
- [World Action Models are Zero-Shot Policies](https://arxiv.org/abs/2602.15922)
- [Physics-IQ: Do Generative Video Models Understand Physical Principles?](https://arxiv.org/abs/2501.09038)
- [Physics-IQ Verified](https://arxiv.org/abs/2606.18943)
- [PISA Experiments](https://arxiv.org/abs/2503.09595)
- [WorldBench](https://arxiv.org/abs/2601.21282)
- [RIGIDBench (ICLR 2026 workshop)](https://arxiv.org/abs/2502.20694)
- [VideoPhy-2](https://arxiv.org/abs/2503.06800)
- [PhyWorldBench](https://arxiv.org/abs/2507.13428)
- [IRIS](https://arxiv.org/abs/2603.16432)
- [PokeFlex](https://arxiv.org/abs/2410.07688)
- [RGBench](https://arxiv.org/abs/2511.06434)
- [DifCloud](https://arxiv.org/abs/2204.03139)
- [SORS](https://arxiv.org/abs/2512.15994)
- [FysicsEval / OmniFysics](https://arxiv.org/abs/2602.07064)
- [How Far is Video Generation from World Model](https://arxiv.org/abs/2411.02385)
- [Validating Robotics Simulators on Real-World Impacts](https://arxiv.org/abs/2110.00541)
- [SIMPLER: Evaluating Real-World Robot Manipulation Policies in Simulation](https://arxiv.org/abs/2405.05941)
- [Scalable Real2Sim](https://arxiv.org/abs/2503.00370)
- [Reconciling Reality through Simulation (Real-to-Sim-to-Real)](https://arxiv.org/abs/2403.03949)
- [Hungarian Method for the Assignment Problem (Kuhn 1955)](https://doi.org/10.1002/nav.3800020109)
- [DayDreamer: World Models for Physical Robot Learning](https://arxiv.org/abs/2206.14176)
- [WorldGym: World Model as an Environment for Policy Evaluation](https://arxiv.org/abs/2506.00613)
- [Physical Validation of Simulators in Computer Graphics (Romero et al. 2021)](https://doi.org/10.1145/3450626.3459939)
- [Cloth Sim-to-Real Benchmarking](https://doi.org/10.1109/lra.2024.3360814)
- [A User's Guide to Calibrating Robotics Simulators](https://arxiv.org/abs/2011.08985)
- [Benchmarking Simulated Robotic Manipulation (Collins et al. 2020)](https://doi.org/10.1109/lra.2019.2953663)
- [PhyGenBench: Towards World Simulator](https://arxiv.org/abs/2410.05363)

---

**总结的intuition**: GAUGE建立的核心认识是——物理保真度无法用单一scalar score刻画。Visual fidelity、trajectory distance、parameter estimation各自只capture一个aspect。真正的physical fidelity需要joint evaluation of law form、physical parameter和temporal behavior against real measurement。当前无论是numerical engine还是generative WM都存在regime-specific的failure mode，impulsive contact、rapid textile motion和volumetric deformation是residual error的主要来源。Generative WM暴露了一个特别深刻的gap：形式正确但参数错误的trajectory，这对未来用generative model做implicit simulator或policy training environment提出严肃质疑。
