---
source_pdf: Rotograb Combining Biomimetic Hands with Industrial Grippers.pdf
paper_sha256: 3ddc701f7dc46fcbe7dba31cccd2a5a3ea6aa5195c79e4ea66c3de318d752ecd
processed_at: '2026-08-12T02:22:08-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Rotograb

## 先说一个让人头疼的困境

你想造一只 robot hand 去抓东西。摆在你面前有两条路，每条都不太对劲。

**第一条路：照抄人手。** 人手有 21 个 DoF，thumb 自己就 5 个 DoF，能 oppose，能 pinch，能 power grasp，能 twist 一根 pen。你去抄，抄出来叫 Shadow Hand，大概 $100,000 起，里面 20+ actuators，tendon 像意大利面一样缠绕，用一段时间 tendon 松了、joint 磨了，修一次半个月。Research lab 玩得起，工厂不敢用。

**第二条路：工业 gripper。** 两个手指对向夹，1 个 DoF，气缸一推，啪，夹住。$500，robust，啥都能夹——前提是 object 形状别太奇怪，别要求 in-hand manipulation。Robotiq 2F-85 就是这种，工厂里到处都是。

这两条路之间是空的。你想要人手的 dexterity，又想要工业 gripper 的简单、便宜、robust，怎么做？

这就是 Rotograb 要回答的问题。

---

## Rotograb 的核心 idea，一句话讲完

**让 thumb 在 palm 上能旋转。**

就这一个 idea，扛起了整篇 paper。

为什么这一个 idea 这么 powerful？因为 thumb 的 opposability 是人手 dexterity 的核心。Shadow Hand 花大力气用 5 个 DoF 去实现 thumb 的 full opposability。Rotograb 说：我用 1 个 rotating plate DoF，把 thumb 推到 left / middle / right 三个位置，对应三种 mode——left hand mode、right hand mode、industrial gripper mode。

- **Thumb 在 left**：像你右手捏一根 pen，precision grasp
- **Thumb 在 right**：像你左手，ambidexterity，corner 里也能抓
- **Thumb 在 middle**：thumb 和其他 4 根 finger 对向，变成一个 industrial-style gripper，power grasp 大物体

这个 mode switching 的妙处在于——**robot arm 本身不需要 reorient**。你把 Rotograb 装在机械臂末端，机械臂不动，就靠 thumb 转一圈，gripper 就从"右手抓"切换成"左手抓"或者"crane 夹"。在狭窄空间（比如货架上靠墙的 box）这个能力就值钱了。

---

## Mechanical design：几个我觉得聪明的地方

### 1. 5 根 finger 全部用同一套设计

这听起来没什么，其实很重要。你如果让 thumb 跟其他 finger 用不同 design，CAD、fabrication、maintenance、spare parts 全部翻倍。Rotograb 把 thumb 就当成一根普通 finger，只是多了一个 rotating plate 当 base。Unified design 是 engineering 美德。

Paper 里说 "All fingers follow the same design"——一句轻描淡写，背后是 iteration 成本降一个数量级。你在 3D 打印机上改一次 finger design，5 根 finger 全部更新，不用单独调 thumb。

### 2. Rolling contact joint + 中间挖一个 cutout

这个 cutout 是 paper 的另一个核心创新，但很容易被忽略。我详细讲讲。

普通的 rolling contact joint 长这样：两个 link 的端面是两个 matching 的 circle，靠 4 根 ligament（交替 top/bottom）绑在一起。弯曲的时候，两个 circle 互相 roll，没有 sliding friction，特别 smooth。这是好东西，Birmingham 大学、哈佛 soft robotics 一直在用。

问题在 tendon routing。你 tendon 要 actuate 这个 joint，自然想法是把 tendon 贴在 circle 外面绕，就像人手 flexor tendon 贴着 phalanx 走一样。但这样 tendon 会 wrap 在 circle 的 arc 上，wrap 的 arc length 跟 joint angle 成非线性关系。你想精确控制 joint angle，就得做一堆 trigonometric compensation，kinematics 难写，控制难调。

Rotograb 的 hack：**在 link 中间挖一个洞，让 tendon 直接从洞里穿过去**。tendon 不再 wrap 在 circle 外，而是从 circle 的 diameter 端点直接拉到下一个 link 的 diameter 端点。

这个 hack 的直接效果：tendon 的几何路径变成了**两个 circle 几何的简单叠加**，每一段 tendon length 都是 closed-form 可算的，不需要数值积分。Paper 的 Section III 那两个公式就是从这个设计直接推出来的。

代价？rolling contact joint 的 lateral compliance 削弱一点（因为中间挖空了，ligament 走的路径变短，lateral 刚度可能下降）。Paper 说 "minimal impact on robustness"，没给量化数据，但 prototype 能用，应该问题不大。

**Intuition**：这是一个典型的 "用 mechanical design 换 control simplicity" 的例子。你不挖洞，你就得在 software 里做非线性 compensation，模型 drift、tendon 蠕变全来。你挖了洞，kinematics 变成高中几何题。Hardware 解决的问题，不要留给 software。

参考 Toshimitsu et al. 2023 的工作 (https://ieeexplore.ieee.org/document/10081911)，他们用的是没挖洞的 rolling contact joint，kinematics 就复杂得多。Rotograb 是站在他们肩膀上的 next iteration。

### 3. Spool 半径比 2:1

每根 finger 2 个 motor：
- Motor 1 actuate joint 1
- Motor 2 actuate joint 2 + 3（coupled，通过一根额外的 coupling tendon 联动）

Joints 2&3 coupled 的副作用：当你 bend joint 2，extensor tendon 要回退的长度是 flexor tendon 拉进的长度的 2 倍（因为 joint 3 跟着 bend，extensor 同时管两个 joint 的回退）。

如果 flexor spool 和 extensor spool 半径一样，那 motor 转同样的角度，flexor 拉 1 单位，extensor 放 1 单位，但实际需要 extensor 放 2 单位。怎么办？**extensor spool 半径做 2 倍**。这样 motor 转 1 度，extensor 放出的 length 是 flexor 拉进的 2 倍，刚好 match。

这是个 micro-design choice，但解决了 coupled joint 的 kinematic consistency。简单、漂亮。

### 4. Thumb 的 tendon 走 plate 的 rotation axis

这个最妙。Rotating thumb 有两种 motion：
- Plate 转动（thumb 整体绕 palm center 转）
- Thumb 自己弯曲（thumb 的 3 个 joint flex/extend）

如果 thumb 的 flexor/extensor tendon 随便挂在 plate 上某个位置，那 plate 一转，tendon anchor 跟着转，tendon length 就变了，thumb 就会被动弯曲——你不想让它弯它自己弯了，这是 mechanical coupling，灾难。

Rotograb 的解法：**让 thumb 的 flexor/extensor tendon 穿过 plate 的 rotation axis**。在 axis 上，无论 plate 怎么转，那个点不动。所以 tendon length 跟 plate rotation 无关，thumb 弯曲和 plate 旋转完全 decouple。

这是 mechanical decoupling 的标准 trick，在 differential gear、gyroscopes 里都有类似 idea。用 mechanical geometry 解决 coupling，software 控制层面零负担。

### 5. Hand 倾斜 10° mounting

Paper 里轻描淡写说 "10° with respect to horizontal wrist plane"，模仿鹰爪 (eagle talon)。这个 10° 的作用：让 hand 从上方 grasp，gravity 主要沿 wrist 轴向，wrist motor 的 holding torque 最小。同时这也让 hand 在受限空间（比如桌面上方 5cm）能塞进去操作。

参考他们组之前 Raptor 工作 (https://ieeexplore.ieee.org/document/9981964)，那个是飞行 drone 上装 gripper 抓东西，10° mounting 的 idea 应该是从那里来的。

---

## Kinematics：公式背后的 intuition

### Finger 的 tendon length 公式（公式 1）

$$
\vec{l}_{P_1 P_1'}(\theta_1) = \begin{bmatrix} r\sin\theta_1 + 2r\cos\frac{\theta_1}{2} \\ r(1-\cos\theta_1) + 2r\sin\frac{\theta_1}{2} \end{bmatrix}
$$

变量含义：
- $\theta_1$: joint 1 弯曲角度
- $r$: rolling contact circle 的半径
- $P_1$ 到 $P_1'$: extensor tendon 在两个 link 之间的 path

**两个分量各来自什么**：
- $r\sin\theta_1$ 和 $r(1-\cos\theta_1)$：这两个是 $O_1'$（real rotation center）相对 $O_1$（virtual center）的位移。简单 trig，因为两个 circle 半径相等，rolling 时 $O_1'$ 走的是 cycloid 路径，但在小角度下近似为 $r\sin\theta$ 和 $r(1-\cos\theta)$。
- $2r\cos\frac{\theta_1}{2}$ 和 $2r\sin\frac{\theta_1}{2}$：这两个是 tendon 本身在两个 circle diameter 端点之间的距离。半角 $\theta_1/2$ 出现是因为 rolling contact joint 的运动学——当两个 circle 互相 roll $\theta_1$ 度，每个 circle 实际转了 $\theta_1/2$ 度（对称分配），所以 diameter 端点的位移对应半角。

**Intuition**：这个公式的 beauty 在于——你把 cutout 的几何约束代进去，tendon path 就是一个 2-DOF 几何问题，可以直接 closed-form solve。如果不挖 cutout，tendon 要走 arc，arc length = $r \cdot \theta_1$，但 tendon 还会脱离 circle、re-attach，建模起来一团乱。

### Rotating thumb 公式（公式 3）

$$
l_{\text{rot}}(\theta) = \sqrt{r_{\text{palm}}^2 + r_{\text{plate}}^2 - 2 r_{\text{palm}} r_{\text{plate}} \cos\left(\frac{\pi}{2} - (\theta + \gamma)\right)}
$$

这就是余弦定理，标准三角形问题。$r_{\text{palm}}$ 是 palm 上 tendon anchor 到 rotation center 的距离，$r_{\text{plate}}$ 是 plate 上 anchor 到 center 的距离，$\theta$ 是 plate 转角，$\gamma$ 是初始 geometry offset。

$\frac{\pi}{2} - (\theta + \gamma)$ 是因为 palm anchor 在 plate anchor 的"上方"（坐标定义），所以夹角是 90° 减去转角。

这个公式没什么 fancy 的，但它的意义在于——rotating thumb 的 tendon length 是个简单 trigonometric function，控制时直接反解就行，不需要 look-up table 或 numerical optimization。

---

## Control：teleop 和 RL 两条线

### Teleoperation

用 OAK-D Pro depth camera (https://shop.luxonis.com/products/oak-d-pro) 拍人手，跑 Google Mediapipe Hand Tracking (https://google.github.io/mediapipe/solutions/hands.html)，提取 21 个 3D landmarks，然后映射到 Rotograb 的 11 个 joint angle。

Index/middle/ring/pinky 直接 angle mapping。Thumb 不一样——Mediapipe 检测人手是 left hand 还是 right hand，自动切换 Rotograb 的 plate position。这就用上了 rotating thumb 的 mode switching 能力。

Mapping 的挑战：人手 shape 跟 Rotograb shape 不同。人手 thumb 有 5 DoF，Rotograb thumb 只有 3 DoF（joints 1-3）+ 1 DoF rotation。所以 mapping 是 lossy 的，需要重新 scale 和 tune。Paper 说 "All angles are scaled and tuned"，意思是手工调的，没自动化。这是个 future work。

### Reinforcement Learning

用 Isaac Gym (https://developer.nvidia.com/isaac-gym) 跑 4096 个并行环境，PPO algorithm，40 分钟训练完一个 ball rotation policy。这部分明显是 follow Toshimitsu et al. 2023 (https://ieeexplore.ieee.org/document/10081911) 的 pipeline。

**Action space**: 11 个 joint angular velocity
**Observation**: 11 个 joint angle + 11 个 joint velocity + object 的 pose + velocity（位置、四元数 orientation、线速度、角速度），总共大约 35 维
**Reward**: ball 绕 x 轴的 angular velocity $\omega_x$，目标范围 1-3 rad/s。$|\omega_x|$ 落在 1 到 3 之间拿满分，外面线性衰减。

训练完的 policy 怎么 deploy 到 real robot？**完全 feedforward**。Paper 老实承认：physical robot 上没有 tactile sensor，没有 closed-loop feedback，policy 在 sim 里 record 成 joint angle 序列，直接在 real hardware 上回放。

这就有问题了。Sim 和 real 有 gap（tendon friction、joint stiffness、object mass distribution 全不一样），feedforward 回放很容易 drift。Paper 用的缓解策略：domain randomization——训练时给所有物理参数加 noise，让 policy 学到一个 robust policy，能 tolerate 参数 variation。

但 tennis ball 这种 symmetric、uniform texture 的 object 还能 feedforward work。Rubik's cube 就不行了——edge 跟 thumb 的 contact 不连续，feedforward 一旦 drift 就 fail。Paper 诚实地说，cube rotation 是 **hardcoded**，不是 RL learned。

这是 Rotograb 现在最大的 limitation：**没有 closed-loop control**。Future work 提到加 camera-based position measurement、Kalman filter、IMU、angular deflection sensors。这些都是必要的，但还没做。

参考 OpenAI Rubik's cube (https://openai.com/blog/solving-rubiks-cube/)，他们 Shadow Hand 上用了 LSTM policy + state estimation + massive domain randomization + 13 个月训练，才解决 cube in-hand rotation。Rotograb 现在的 hardware 没有 sensor，达不到那个 level。

---

## Experiments：到底 work 不 work

### Workspace

用 Qualisys motion capture 测 5 个 fingertip 的 3D workspace。Thumb 的 workspace 明显大于其他 finger（因为能扫 -65° 到 +65°）。其他 4 个 finger workspace 类似（unified design 的好处）。

Gap：finger 之间没有 adduction/abduction DoF，workspace 不连续。Paper 承认这是 next step。

### Grasping YCB objects

YCB benchmark (http://www.ycbbenchmarks.com/) 里的 object：ping pong ball、pen、mug、drill、handball、pylon。Small to large，都抓得起来。Thumb 切换 L/M/R 三个 position 来适配。

所有 grasping 都是 teleop 完成。这点要记住——**RL 只在 ball rotation 上 work，grasping 还没自动化**。这是个 limitation。

### Ambidexterity

把 ball 放在 box 里，靠近左墙 / 右墙。Ball 靠左墙，用 right-hand mode（thumb 在 right）；ball 靠右墙，用 left-hand mode。两种都成功。

这个实验展示了 rotating thumb 的真正 value：**confined space 操作**。工厂货架、装配线、家居环境，到处都是 constrained space。传统 gripper 遇到墙就要 reorient 整个机械臂，Rotograb 转 thumb 就行。

### In-hand manipulation

- Tennis ball rotation：RL policy，5.45 rounds/min（约 0.09 Hz）
- Rubik's cube rotation：hardcoded，5.62 rounds/min（约 0.09 Hz）

速度不快，但 demo 出了 capability。Rubik's cube 那个 video 里，thumb 转 palm axis，其他 finger 提供 countertorque，这种 motion 在 fixed-thumb hand 上做不出来——fixed-thumb hand 要转 cube，必须靠 finger adduction/abduction，而大多数 hand 这个方向 DoF 不够。Rotograb 的 rotating thumb 提供了一个新的 manipulation axis，这是 hardware innovation 直接 enable 的 new capability。

---

## 这篇 paper 的真正贡献

我重排一下，从我视角看：

1. **Mechanical innovation**：Rotating thumb + cutout rolling contact joint。这两个是真正的 hardware contribution，让 dexterity + power grasp 在一个 $1200 的 platform 上共存。
2. **Design philosophy**：Unified finger design + underactuated coupling + mechanical decoupling。这是好的 engineering，让 system 简单、便宜、可维护。
3. **Demo of capability**：YCB grasping、ambidexterity、in-hand rotation。Proof of concept 完成。
4. **Limitation 诚实披露**：RL 只在 ball 上 work，cube 是 hardcoded，没有 closed-loop control，no tactile。这种诚实让 paper 更可信。

它**不是**解决了 in-hand manipulation 问题。它**是**展示了一种新的 hardware design 方向，让 in-hand manipulation 变得可能且 affordable。

---

## 我想吐槽的几个点

1. **No quantitative grasping success rate**。YCB grasping 那一节全是 qualitative demo，没说"我们试了 N 次，成功 K 次"。这是 robotic grasping paper 的硬指标，缺了就不严谨。Yale OpenHand Project、Dex-Net 都会报 success rate。
2. **No comparison with baselines**。应该至少跟 Robotiq 2F-85 或者 LEAP Hand 在同样 task 上对比一下。Paper 只展示 Rotograb 自己能干啥，没说比 baseline 好多少。
3. **RL policy 没 evaluate on real robot**。Feedforward 在 tennis ball 上 work，但没量 success rate。如果 tennis ball 换成稍微 asymmetric 的 object，feedforward 还 work 吗？没数据。
4. **Tendon creep / fatigue 完全没讨论**。Tendon-driven system 长期使用会松，rolling contact joint 的 ligament 会磨损。Paper 没说 maintenance cycle。
5. **Rotating plate range 只有 ±65°**。如果做 ±90°，thumb 能完全 fold 到 palm 另一侧，ambidexterity 更彻底。机械 limit 在哪？没讨论。
6. **Wrist 10° mounting 没有量化对比**。说"reduces wrist torque"，但没给数据。Mounting angle 0°、5°、10°、15° 各有什么 trade-off？

---

## 跟相关工作的关系网

让我画个 mental map：

- **Toshimitsu et al. 2023** (https://ieeexplore.ieee.org/document/10081911)：前驱工作，tendon-driven rolling contact hand + RL ball rotation。Rotograb 继承了 rolling contact joint idea，加了 cutout 和 rotating thumb。
- **LEAP Hand** (https://leap-hand.com/)：CMU 2023，direct-driven 16 DoF，$2000。Rotograb 同时代、更便宜、更 compliant，但 DoF 少。
- **Shadow Hand** (https://www.shadowrobot.com/)：高端 benchmark hand，$100k+，Rotograb 是 affordable alternative。
- **iHY Hand** (https://ieeexplore.ieee.org/document/6316233)：Yale 的 underactuated hand，philosophy 接近 Rotograb（用 underactuation 换 simplicity）。
- **OpenAI Rubik's cube** (https://openai.com/blog/solving-rubiks-cube/)：in-hand manipulation 的 milestone，Rotograb 离这个 level 还有距离。
- **Raptor** (https://ieeexplore.ieee.org/document/9981964)：同 lab 前作，drone + gripper，10° mounting angle idea 来源。
- **Vision-controlled jetting** (https://www.nature.com/articles/s41586-023-06684-3)：同 lab 2023 Nature paper，multi-material 3D printing，未来 Rotograb 可能用这个 fabrication method。

---

## 一句话总结

Rotograb 用 **rotating thumb + cutout joint** 两个 mechanical innovation，在 $1200 的 tendon-driven hand 上同时实现了 industrial gripper 的 power grasp 和 humanoid hand 的 ambidexterity + in-hand manipulation。它没有解决 in-hand manipulation 的 control 问题（cube 还得 hardcoded），但它展示了一种新的 hardware design direction——**用 mechanical cleverness 替代 software complexity**。这个 direction 比"堆 DoF、堆算力"更可持续，是 robotics hardware design 的好范例。

如果你 (Karpathy) 想 build intuition，我会说：好的 robotics hardware design 是**让 software 的 life 更容易**，而不是让 hardware 看起来更 impressive。Rotograb 的 cutout 让 kinematics 变成高中几何题，rotating thumb 让 mode switching 变成 1-DoF 控制，thumb tendon 走 rotation axis 让 decoupling 变成 free。每一个 mechanical choice 都在减少 software 需要处理的 complexity。这种"为 downstream 着想"的 design thinking，是 robotics hardware 的核心 skill，比堆 sensor、堆 actuator 难得多。

参考资料：
- Paper (假设 arXiv): 搜 "Rotograb" on arXiv
- Robert Katzschmann Lab: https://rsl.ethz.ch/
- Toshimitsu 2023: https://ieeexplore.ieee.org/document/10081911
- LEAP Hand: https://leap-hand.com/
- Shadow Hand: https://www.shadowrobot.com/
- OpenAI Rubik's: https://openai.com/blog/solving-rubiks-cube/
- Isaac Gym: https://developer.nvidia.com/isaac-gym
- PPO: https://arxiv.org/abs/1707.06347
- Mediapipe Hands: https://google.github.io/mediapipe/solutions/hands.html
- YCB Benchmark: http://www.ycbbenchmarks.com/
- Cutkosky Grasp Taxonomy: https://ieeexplore.ieee.org/document/6316233

---

# Rotograb: Hybrid Biomimetic Hand + Industrial Gripper 深度解析

这篇paper来自ETH Zurich的Robert Katzschmann实验室，2024年左右的工作。核心idea非常巧妙：用一根**rotating thumb**桥接humanoid hand的dexterity与industrial gripper的power grasp。我会从mechanical design、kinematics、control、experiments四个层面深入讲解，并补充相关背景和intuition。

---

## 1. Motivation & Design Philosophy

机器人gripper领域长期存在一个dichotomy：

| 类别 | 代表 | DoF | 特点 | 缺陷 |
|------|------|-----|------|------|
| Industrial grippers | Robotiq 2F-85 | 1-3 | 简单、robust、力大 | 无in-hand manipulation |
| Soft grippers | Fin Ray, Raptor [1] | passive | 适应性强、safe | 力量弱、精度差 |
| Mid-tier hands | iHY Hand [6], Eagle Shoal [7] | 8-16 | 低成本 | dexterity有限 |
| High-end biomimetic | Shadow Hand, Schunk SVH | 20+ | 全dexterous | 昂贵、fragile |
| Direct-driven | LEAP Hand [9], Allegro [10] | 16 | 控制简单 | compliance差 |

Rotograb的insight在于：人手之所以versatile，**thumb的opposability是关键**，而工业gripper之所以powerful，在于**对向夹持几何**。如果让thumb能在palm上**旋转**，就可以在两种mode之间切换：

- Thumb居中：模仿crane/claw，大物体power grasp（industrial mode）
- Thumb左侧/右侧：模仿人手left/right，precision grasp + ambidexterity

这个10° wrist mounting angle模仿了**鹰爪抓取角度**——参考他们组之前的Raptor工作 [1] (https://ieeexplore.ieee.org/document/9981964)。

---

## 2. Mechanical Design细节

### 2.1 整体architecture

- **5 fingers**，所有finger用**统一design**（简化建模与fabrication）
- **11 DoF total**：2 per finger × 5 fingers + 1 rotating plate
- 每finger 4 links: base → lower link → upper link → tip
- 每finger 3 rolling contact joints
- Hand相对wrist平面倾斜10°（减少wrist torque，grasp from above）
- Finger base相对palm 45°
- 总高27.5 cm，finger length 9.6 cm，palm width 9.4 cm，hand总长17.9 cm —— 接近成年男性hand尺寸

### 2.2 Rolling Contact Joint with Cutout（核心创新之一）

传统rolling contact joint（如Toshimitsu et al. [17], https://ieeexplore.ieee.org/document/10081911）的问题：tendon必须wrap在joint曲面外，导致tendon length随joint angle非线性变化，kinematics复杂。

Rotograb的解法：**在每个link中心切出一个cutout**，让tendon走cutout内部，从circle的diameter端点直接拉。

这带来三个关键simplification：

1. **Decoupling**：tendon for joint 2/3穿过joint 1的rotation center $O_1$和$O_1'$，所以joint 1弯曲不会改变upper joints的tendon length
2. **Constant center distance**：$|O_1 O_1'|$ 在整个rotation过程中保持不变（因为两个circle半径相等，ligament对称耦合）
3. **Linearized actuation**：tendon的pull distance与joint angle成正比

代价：rolling contact joint本身的lateral compliance被cutout削弱一点，但paper说robustness影响minimal。

### 2.3 Tendon Routing

每根finger 2 motors：
- Motor 1: actuate joint 1 (flexor + extensor)
- Motor 2: actuate coupled joints 2&3 (flexor + extensor + coupling tendon)

Spool radius设计：extensor spool radius = 2 × flexor spool radius。原因：joints 2&3耦合后，single extensor tendon需要同时控制两个joint的extension，tendon length变化是flexor的2倍，所以spool半径翻倍保持motor angle mapping一致。

### 2.4 Rotating Thumb

- Plate通过pin joint + bearing安装在palm中心
- 范围：-65° 到 +65°
- 两根tendon左右对拉，symmetric
- 关键设计：**thumb的flexor/extensor tendons穿过plate的rotation axis**，这样plate旋转不会影响thumb本身的flexion/extension tendon length
- 这是一种**mechanical decoupling**，比软件compensation更优雅

### 2.5 Cost breakdown

| Component | Cost (€) |
|-----------|----------|
| 11× Robotis XC330-T288-T servos | 896 |
| U2D2 control unit | 81 |
| U2D2 Power Hub | 45 |
| SMPS 12V 5A | 50 |
| 3D printed parts + tendons + silicon + bearings + screws | ~28 |
| OAK-D Pro AF camera (teleop) | 316 |
| **Total** | **~1100 + 316** |

对比Shadow Hand ~$100k+，Allegro Hand ~$15k，LEAP Hand ~$2000。Rotograb在$1200价位段做到了11 DoF + rotating thumb + tendon actuation，是非常cost-effective的设计。

---

## 3. Kinematics详解

### 3.1 Joint 1运动学（公式1-2）

参考Figure 6a，设定：
- $O_1$: virtual rotation center（下link的circle center）
- $O_1'$: real rotation center（base的circle center）
- $r$: circle radius
- $P_1$: extensor tendon在base上的exit point
- $P_1'$: extensor tendon在下link上的entry point
- $\theta_1$: joint 1的bending angle

**Tendon length vector公式（公式1）**：

$$
\vec{l}_{P_1 P_1'}(\theta_1) = \begin{bmatrix} l_x(\theta_1) \\ l_y(\theta_1) \end{bmatrix} = \begin{bmatrix} r\sin\theta_1 + 2r\cos\frac{\theta_1}{2} \\ r(1-\cos\theta_1) + 2r\sin\frac{\theta_1}{2} \end{bmatrix}
$$

**变量含义解析**：
- $l_x$分量：水平方向位移。第一项 $r\sin\theta_1$ 来自 $O_1'$ 相对 $O_1$ 的水平偏移；第二项 $2r\cos\frac{\theta_1}{2}$ 是tendon在两个circle上wrap的几何投影（半角是因为rolling contact的运动学特性——当joint转动 $\theta_1$，每个circle相对另一circle滚动 $\theta_1/2$，对应arc length为 $r \cdot \theta_1/2$，但投影成 $2r\cos(\theta_1/2)$ 是因为tangent方向的几何关系）
- $l_y$分量：垂直方向位移。第一项 $r(1-\cos\theta_1)$ 是 $O_1'$ 抬起的高度；第二项 $2r\sin\frac{\theta_1}{2}$ 是tendon wrap在垂直方向的投影

**Tendon length change（公式2）**：

$$
\Delta l(\theta_1) = \|\vec{l}_{P_1 P_1'}(\theta_1)\| - \|\vec{l}_{P_1 P_1'}(\theta_{1,\text{init}})\|
$$

其中 $\theta_{1,\text{init}} = -45°$ 是calibration angle（finger base 45° mounting，flat position对应-45°）。

**Intuition**：这个公式的beauty在于——通过cutout设计，tendon的几何路径变成了**两个circle几何的简单叠加**，没有非线性wrap。如果不cutout，tendon会wrap在circle外弧上，长度 = $r \cdot \theta_1$（弧长）+ 其他项，非线性且与cutout case不同。

### 3.2 Rotating Thumb运动学（公式3-4）

参考Figure 7，用cosine theorem：

$$
l_{\text{rot}}(\theta) = \sqrt{r_{\text{palm}}^2 + r_{\text{plate}}^2 - 2 r_{\text{palm}} r_{\text{plate}} \cos\left(\frac{\pi}{2} - (\theta + \gamma)\right)}
$$

**变量含义**：
- $r_{\text{palm}}$: palm上tendon anchor到plate rotation center的距离
- $r_{\text{plate}}$: plate上tendon anchor到rotation center的距离
- $\theta$: plate rotation angle (0 = middle, +65° = right, -65° = left)
- $\gamma$: 初始geometry offset angle
- $\frac{\pi}{2} - (\theta + \gamma)$: 实际夹角（用 $\frac{\pi}{2}$ 减是因为坐标定义）

$$
\Delta l_{\text{rot}}(\theta) = l_{\text{rot}}(\theta) - l_{\text{rot}}(\theta_0), \quad \theta_0 = 0
$$

由于symmetric design，左侧tendon缩短量 = 右侧tendon伸长量。两个servo通过differential-like方式控制plate rotation。

**Intuition**：这是标准三角函数应用，但key是**thumb自身的flexion tendon不走这条路径**——它们走plate rotation axis，所以thumb弯曲和plate旋转完全decouple。这种mechanical decoupling比任何software-level compensation都robust。

### 3.3 Joint 2 & 3耦合运动学

Joints 2&3 share一个motor，通过一根green coupling tendon（Fig 5a）连接。当motor拉flexor，joint 2弯曲，coupling tendon同时拉joint 3 flexor，使joint 3也同步弯曲。

这种underactuation（2 motors控制3 joints）的trade-off：
- Pro: 简化控制、减少motor数量、降低cost
- Con: 无法独立控制distal joint，限制in-hand manipulation精细度

人手其实也有类似的tendon coupling（FDP tendon连接multiple fingers），所以这种设计也符合biomimetic principle。

---

## 4. Control架构

### 4.1 Teleoperation Pipeline

```
Human hand → OAK-D Pro depth camera → Google Mediapipe Hand Tracking
       → 21 landmarks → joint angle extraction → mapping → Rotograb servos
```

- **OAK-D Pro AF** (https://shop.luxonis.com/products/oak-d-pro): Stereo + RGB depth camera，on-board Myriad X芯片可跑neural network
- **Mediapipe Hand** (https://google.github.io/mediapipe/solutions/hands.html): 21 3D landmarks，30 FPS real-time
- Mapping策略：
  - Index/middle/ring/pinky: 直接joint angle mapping (scaled & tuned)
  - Thumb: 检测human hand是left还是right，自动switch Rotograb的plate position
- 挑战：人手和Rotograb的shape/size不同，mapping需要重新scale，特别是rotating thumb的workspace mapping

### 4.2 Reinforcement Learning Pipeline

参考Toshimitsu et al. [17] (https://arxiv.org/abs/2305.08558)的pipeline：

**Simulator**: NVIDIA IsaacGym (https://developer.nvidia.com/isaac-gym) - GPU-based physics sim，可以跑4096个并行环境

**Algorithm**: PPO (Proximal Policy Optimization) (https://arxiv.org/abs/1707.06347)

```
PPO核心：
- Policy: π_θ(a|s) Gaussian分布
- Value: V_φ(s)
- Clipped surrogate objective:
  L = E[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]
  其中 r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
  ε = 0.2 (clipping range)
- Advantage A_t = GAE (Generalized Advantage Estimation)
```

**Training specs**:
- 4096 parallel environments
- 2000 epochs
- 40 minutes on single RTX 3060
- 实现: rl_games (https://github.com/Denys88/rl_games)

**Observation space**:
- Joint angles (11)
- Joint angular velocities (11)
- Object position (3)
- Object orientation (4, quaternion)
- Object linear velocity (3)
- Object angular velocity (3)
- Total: ~35 dim

**Action space**: Joint angular velocities (11 dim)

**Reward function for ball rotation**:
```
R = max(0, 1 - |ω_x| - target_ω) if target_min < |ω_x| < target_max
    线性decrease outside this region
target: 1 < |ω_x| < 3 rad/s
```

其中 $\omega_x$ 是object绕x轴的angular velocity。Flip sign of $\omega_x$ 可训练reverse方向rotation policy。

**Domain Randomization** (为sim2real):
- Observation noise: Gaussian
- Action noise: Gaussian
- Tendon damping & stiffness: uniform
- Joint damping & stiffness: uniform
- Joint range of motion: uniform
- Robot & object mass: uniform
- Friction coefficient: uniform
- Object scale: uniform

**Sim2Real策略**: 由于physical robot无sensor feedback（无tactile、无joint encoderfeedback loop），policy被record成joint angle sequence，**feedforward execution** on real hardware。这是limitation，paper承认future work需要加camera-based position measurement + Kalman filter + IMU。

参考OpenAI的Rubik's cube工作 (https://openai.com/blog/solving-rubiks-cube/)，他们用了大量domain randomization + LSTM policy + state estimation。Rotograb简化到feedforward，可能是因为rolling contact joint本身的compliance提供了implicit robustness。

---

## 5. Experiments & Results

### 5.1 Workspace Analysis (Fig 8)

用Qualisys motion capture系统测量5个fingertip的3D workspace。

Key findings:
- Rotating thumb的workspace明显大于其他fingers（因为可以扫过 -65° 到 +65°）
- 4个固定fingers的workspace相似（因为unified design）
- Workspace之间有gap——缺少adduction/abduction

### 5.2 YCB Object Grasping (Fig 9)

测试objects（来自YCB benchmark, http://www.ycbbenchmarks.com/）:
- 小物：ping pong ball, pen
- 中物：mug, drill
- 大物：handball, pylon

Grasps categorized by Cutkosky taxonomy [16] (https://ieeexplore.ieee.org/document/6316233):
- Precision grasp (thumb L or R position)
- Power grasp (thumb M position)

所有task通过teleoperation完成。

### 5.3 Ambidexterity Test (Fig 10)

实验设计：ball放在box里，靠近左墙或右墙
- Ball near left wall → use right-hand mode (thumb on right)
- Ball near right wall → use left-hand mode (thumb on left)

这个实验validates了rotating thumb的核心value proposition：**不需要reorient robot arm**就能在confined space操作，这对工业场景（货架、装配线）意义重大。

### 5.4 In-Hand Manipulation (Fig 11)

**Tennis ball rotation**:
- Policy: learned via RL in IsaacGym
- Thumb alternates between lateral and central configuration
- Speed: **5.45 rounds per minute** (≈0.09 Hz)

**Rubik's cube rotation**:
- Policy: **hardcoded** (不是RL，因为cube edges和thumb的interaction复杂，sim2real gap大)
- Thumb pushes cube edges, other fingers provide countertorque
- Speed: **5.62 rounds per minute** (≈0.094 Hz)

Intuition: rotating thumb让cube能在palm axis方向旋转，这是传统fixed-thumb hand做不到的（需要finger adduction/abduction，而大多数hand这方向DoF不足）。

---

## 6. Critical Analysis & Related Work

### 6.1 与LEAP Hand对比

LEAP Hand (https://leap-hand.com/, Shaw et al. [9]) 是CMU 2023年的low-cost dexterous hand：
- 16 DoF, direct-driven (motor in joint), $2000
- 每joint独立actuation，控制简单
- 适合RL（IsaacGym sim ready）

Rotograb vs LEAP:
| 维度 | Rotograb | LEAP Hand |
|------|----------|-----------|
| DoF | 11 | 16 |
| Actuation | Tendon | Direct |
| Compliance | High (rolling contact) | Low (rigid) |
| Cost | ~$1200 | ~$2000 |
| In-hand manipulation | Yes (with rotating thumb) | Yes (with RL) |
| Sim2Real ease | Harder (tendon nonlinearity) | Easier (direct drive) |

Rotograb的优势是**mechanical compliance**和**rotating thumb带来的ambidexterity**，劣势是tendon control更复杂。LEAP的优势是direct drive简化control和sim，劣势是rigidity可能damage objects。

### 6.2 与Shadow Hand对比

Shadow Hand (https://www.shadowrobot.com/):
- 24 DoF, 20 actuators
- Tendon-driven, ~$100k+
- 5 fingers with full thumb opposability
- 标准in-hand manipulation benchmark

Rotograb用1个rotating plate DoF + 5 fingers × 2 DoF 实现了ambidexterity + dexterity的折中。Shadow靠full thumb的5 DoF实现类似功能，但成本高2个数量级。

### 6.3 与Toshimitsu et al. [17]的关系

Toshimitsu et al. 2023 (https://ieeexplore.ieee.org/document/10081911)是同一field的先驱工作，他们做了**tendon-driven hand with rolling contact joints** for ball rolling。Rotograb:
- 继承了rolling contact joint设计
- **创新**: cutout简化kinematics
- **创新**: rotating thumb扩展workspace
- 用了类似的RL pipeline (PPO + IsaacGym)

可以说Rotograb是Toshimitsu工作的next iteration，重点解决kinematics复杂度和ambidexterity问题。

### 6.4 与OpenAI Rubik's Cube [8]对比

OpenAI 2019 (https://openai.com/blog/solving-rubiks-cube/, Andrychowicz et al.):
- Shadow Hand
- 5 fingers, full dexterity
- Massive domain randomization
- LSTM + ADR (Automatic Domain Randomization)
- 13 months training

Rotograb简化到hardcoded policy for cube，是honest的limitation disclosure。这反映了一个现实：**rotating thumb虽然提供了新的manipulation axis，但control复杂度也上升了**，对sim2real提出新挑战。

### 6.5 Soft Robotics方向

Katzschmann lab本身也做soft robotics (https://rsl.ethz.ch/)。Rotograb用rolling contact joint + silicone tip实现**semi-soft** design，介于rigid industrial gripper和fully soft gripper之间。这种hybrid approach可能是未来trend——纯soft力量不足，纯rigid不safe。

---

## 7. Limitations & Future Work

Paper承认的:
1. **No finger adduction/abduction** → workspace gap
2. **No tactile sensing** → feedforward RL policy
3. **Hardcoded cube policy** → RL没能solve cube
4. **Teleoperation mapping** → 人手和robot hand shape差异大

我补充的潜在issues:
1. **Tendon creep & fatigue**: Tendon-driven system长期使用会stretch，需要re-tensioning。Paper没讨论maintenance cycle。
2. **Rolling contact joint wear**: 长期rolling会导致ligament磨损，paper提到"pop instead of break"但没量化lifetime。
3. **Spool radius ratio fixed at 2:1**: 如果未来想decouple joints 2&3，需要重新设计actuation。
4. **Rotating plate只1 DoF**: 真人手thumb有5 DoF，Rotograb的thumb只能rotate，不能flex跨palm（无法触其他finger base）。这限制了precision pinch tasks。
5. **RL只test了ball**: 真正的dexterous manipulation需要test更多object shape/texture，paper的YCB grasping只通过teleop，没通过RL。
6. **No closed-loop control**: 完全开环执行RL policy在physical robot上，对object variation的robustness未充分验证。

---

## 8. Key Takeaways for Intuition

1. **Mechanical decoupling > Software compensation**: Rotating thumb的tendon走rotation axis，cutout让tendon length不依赖wrap geometry——这种"在hardware层面解决coupling问题"的设计哲学值得学习。

2. **Underactuation is a feature, not bug**: 2 motors per 3 joints通过coupling tendon实现natural grasp motion，类似人手FDP tendon共享。这降低cost同时保留必要dexterity。

3. **Mode switching vs. continuous dexterity**: Rotograb选择了discrete modes (L/M/R thumb position)而不是continuous thumb reconfiguration，这是pragmatic trade-off——简化control，覆盖主要use cases。

4. **Biomimetic abstraction**: 不照搬人手21 DoF，而是提取核心principle (thumb opposability) + industrial principle (对向夹持)，merge成新设计。这是好的engineering thinking。

5. **Cost-aware design**: $1200的11 DoF tendon hand + rotating thumb，配合3D printing快速迭代，是research prototype的优秀template。

6. **Sim2Real gap remains open**: 即使用了domain randomization，cube task还是hardcoded。这说明tendon-driven + rolling contact + rotating plate的复合非线性仍是sim2real的nightmare。

---

## 9. 相关Reference Links

- Paper PDF (假设arXiv): 通常此类ETH工作会发到arXiv，建议查 https://arxiv.org/abs/搜索 "Rotograb"
- Robert Katzschmann Lab: https://rsl.ethz.ch/
- Toshimitsu et al. 2023 (前驱工作): https://ieeexplore.ieee.org/document/10081911
- LEAP Hand: https://leap-hand.com/ | GitHub: https://github.com/leap-hand/LEAP_Hand_Sim
- Shadow Hand: https://www.shadowrobot.com/products/dexterous-hand/
- Allegro Hand: http://www.wonikrobotics.com/Allegro-Hand.htm
- PPO paper: https://arxiv.org/abs/1707.06347
- Isaac Gym: https://developer.nvidia.com/isaac-gym | Docs: https://docs.omniverse.nvidia.com/isaacsim/
- rl_games: https://github.com/Denys88/rl_games
- Google Mediapipe Hands: https://google.github.io/mediapipe/solutions/hands.html
- OAK-D Pro: https://shop.luxonis.com/products/oak-d-pro
- YCB Benchmark: http://www.ycbbenchmarks.com/
- Cutkosky Grasp Taxonomy: https://ieeexplore.ieee.org/document/6316233
- OpenAI Rubik's Cube: https://openai.com/blog/solving-rubiks-cube/
- Raptor (同组前作): https://ieeexplore.ieee.org/document/9981964
- Vision-controlled jetting (同组Nature 2023): https://www.nature.com/articles/s41586-023-06684-3

---

总结一句：这篇paper是**hardware design-driven**而非**algorithm-driven**的工作，核心贡献在mechanical innovation (rotating thumb + cutout joint)，control层面借用existing tools (PPO + IsaacGym + Mediapipe)。这种"用mechanical cleverness简化control complexity"的思路，比堆DoF和堆RL算力更可持续，是robotics hardware design的范例工作。
