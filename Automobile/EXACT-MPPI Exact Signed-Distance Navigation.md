---
source_pdf: EXACT-MPPI Exact Signed-Distance Navigation.pdf
paper_sha256: 2e0e9f6cc17da778e0209bd17046e384f08d440f9b577982d0dc527d38f07a12
processed_at: '2026-08-18T11:51:15-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---
想象你开着一辆叉车，叉车前面还叉着一个大托盘。整个"叉车+托盘"从天上往下看，形状像一个大写的 T——叉车主体是下面的竖杠，托盘是上面的横杠。T 这个形状有个特点：中间凹进去一块。

现在你要让这辆 T 形叉车穿过一个窄窄的过道，过道两边各有一个障碍物。过道的宽度恰好让 T 的横杠刚刚能塞进去，前提是 T 的竖杠必须对准过道中央，让两边障碍物正好"卡进"T 的凹槽里。

人类司机看到这个场景，脑子里会想："我把车摆正，让障碍物正好进到 T 的凹槽里，就能穿过去。"

但传统机器人导航算法做不到。原因有两个：

**第一个原因：算法嫌 T 形太麻烦，给它套了个壳。** 大多数算法会把机器人简化成最简单的形状。T 形不好算，那就把 T 形的四个角连起来，用一个凸多边形把它包住——这叫 convex hull。T 形的凸多边形大概像一个五边形，把 T 的凹槽也填实了。这样一来，凹槽里本来能容纳障碍物的空间没了，算法一看："过道这么窄，我这个五边形肯定过不去。"于是干脆放弃，原地停车。但其实 T 形本体的凹槽正好能容纳障碍物，人眼一看就知道能过。

**第二个原因：算法嫌弃点云太乱，非要先画个格子地图。** LiDAR 扫一圈回来，给一堆散点。算法觉得这堆点不好用，先把它栅格化成 occupancy grid——把空间切成方格，标记每个方格有没有障碍物。问题是栅格化会丢精度。一个 10 厘米宽的柱子，如果在 20 厘米的格子里，可能被画成 20 厘米宽的格子。本来就紧张的 clearance，被栅格这么一放大，又把可行解给抹掉了。

这篇 paper 说：**别简化 footprint，别画地图，直接干。** LiDAR 的点云拿来就用，footprint 是什么形状就是什么形状，哪怕是凹的、带凹陷的、奇形怪状的，直接算障碍物点到 footprint 的精确距离，然后把这个距离塞到 MPPI 控制器里做采样，GPU 上并行算几百万次，毫秒级出结果。

---

## 关键 trick：别动机器人，动机器人周围的点

这一招特别漂亮。想象你在算"障碍物到机器人有多近"。最直觉的做法是：footprint 在每个 rollout 里都跟着机器人跑，每跑一步都要重新旋转、平移 footprint 顶点，然后跟障碍物点算距离。这样算可以，但效率低，因为每个 rollout 都要重新构造 footprint 的几何。

这篇 paper 反过来做：**把 footprint 钉在机器人自己的坐标系里不动，把所有障碍物点变换到 footprint 所在的坐标系里再算距离。** 这就相当于你站着不动，让世界绕着你转。为什么这样更好？因为 footprint 的顶点坐标在程序里是常数，写死在 JAX tensor 里，一次编译，永远不变。变的是障碍物点，障碍物点的变换是一次矩阵乘加（rotation + translation），GPU 上是超级便宜的 broadcasting 操作。

于是 footprint 的顶点数据结构长这样：(B, 2) —— B 个顶点，每个顶点 2 维坐标。障碍物数据结构长这样：(K, T, N, 2) —— K 个 rollout，每个 rollout 有 T 个时间步，每个时间步有 N 个障碍物点，每个点 2 维坐标。整个 batched 距离计算就是在这两个 tensor 上做 elementwise 运算和 reduction。JAX 编译后，这就是一个 GPU kernel，一次 call 算完所有 5 百万个距离查询。

---

## 两种 footprint 表示，看情况选

paper 准备了两套计算距离的方法，根据 footprint 形状挑着用。

### Rectangle-cover 路线：适合方方正正的 footprint

很多机器人的 footprint 是"横平竖直"的——比如底盘是个长方形，前面叉个托盘也是长方形。这种 footprint 可以用几个轴对齐的长方形的并集来表示。点到单个长方形的 signed distance 有一个超简洁的 closed-form：

$$d_{\mathrm{box},j}^{\pm}(\mathbf{p}) = \|\max(\mathbf{a}_j(\mathbf{p}), 0)\|_2 + \min\left(\max(a_{j,x}, a_{j,y}), 0\right)$$

这里 $\mathbf{a}_j(\mathbf{p}) = |\mathbf{p} - \mathbf{c}_j| - \mathbf{s}_j$，$\mathbf{c}_j$ 是长方形中心，$\mathbf{s}_j$ 是半边长，$|\cdot|$ 是逐元素取绝对值。$a_{j,x}$ 和 $a_{j,y}$ 是 $\mathbf{a}_j$ 的两个分量。

这个公式看着吓人，其实就是小学几何：

- $\mathbf{a}_j$ 的分量表示点 $\mathbf{p}$ 在 x 和 y 两个方向上"超出"长方形边界多少。负值意味着点在长方形内部
- 第一项 $\|\max(\mathbf{a}_j, 0)\|_2$：如果点在长方形外面，这是到最近角点的欧氏距离；如果点在里面，这一项是 0
- 第二项 $\min(\max(a_{j,x}, a_{j,y}), 0)$：如果点在长方形里面，这是穿透深度——离最近边有多远

整个表达式没有 if-else 分支，全是 elementwise 运算加一个 norm。GPU 上最爱的就是这种无分支的算术流。

### Polygon-edge 路线：适合奇形怪状的 footprint

如果 footprint 不方正，比如菱形、星形、箭头形，那就要用一般多边形。思路是：对多边形的每一条边，算点到这条边的最短距离；取所有边的最小值；用 ray-casting 判断点在多边形里面还是外面，决定符号。

点到一条边的最短距离，关键参数是投影参数 $\alpha_b$：

$$\alpha_b(\mathbf{p}) = \max\left(0, \min\left(1, \frac{(\mathbf{p} - \mathbf{v}_b)^\top (\mathbf{v}_{b+1} - \mathbf{v}_b)}{\|\mathbf{e}_b\|^2}\right)\right)$$

$\mathbf{v}_b$ 是边的起点，$\mathbf{v}_{b+1}$ 是终点，$\mathbf{e}_b = \mathbf{v}_{b+1} - \mathbf{v}_b$ 是边向量。这个 $\alpha_b$ 是把点 $\mathbf{p}$ 投影到边的延长线上后，投影点离起点 $\mathbf{v}_b$ 的距离占整条边的比例。Clamp 到 [0, 1] 是为了不让投影点跑到边的外面去——如果投影落在外面，最近点就取端点。

最终点到这条边的距离就是：

$$d_{\mathrm{seg},b}(\mathbf{p}) = \|\mathbf{p} - (\mathbf{v}_b + \alpha_b(\mathbf{p})\mathbf{e}_b)\|_2$$

$\mathbf{v}_b + \alpha_b \mathbf{e}_b$ 是边上离 $\mathbf{p}$ 最近的那个点。点到它的欧氏距离就是边距离。

对所有 B 条边取 min，再乘以内外判断的符号 $\sigma = \pm 1$，就得到 signed distance。

**这种表示最大的好处是：凹多边形也能直接处理，不用把凹多边形拆成几个凸多边形再并起来。** 一个 simple polygon，无论凸凹，统一一个 evaluator 搞定。

实验数据显示：对 L 形、T 形、F 形这些方正的 footprint，rectangle-cover 比 polygon-edge 快 2 到 3.34 倍。所以 paper 把两条路都留着——方正的走快路，歪七扭八的走通用路。

---

## 距离算出来之后，MPPI 怎么用

MPPI 是个"采样 + 加权平均"的控制器。每个控制周期做这么几件事：

**第一步：采样一堆候选控制序列。** 假设有 1000 个 rollout（K=1000），每个 rollout 预测未来 5 秒（T=50 步，每步 0.1 秒）。每个 rollout 都是从当前 nominal 控制序列上加噪声扰动得到的。

**第二步：把每个 rollout 往前推演。** 用运动学模型把状态推 forward。比如 differential-drive 模型下：

$$\dot{x} = v \cos\theta, \quad \dot{y} = v \sin\theta, \quad \dot{\theta} = \omega$$

每个 rollout 推 50 步，得到一条 50 个状态点的预测轨迹。

**第三步：算每条轨迹的 cost。** Cost 是三部分之和：

$$J^{(r)} = \sum_h \left[\phi_{\mathrm{task}} + \phi_{\mathrm{ctrl}} + \phi_{\mathrm{obs}}\right]$$

- $\phi_{\mathrm{task}}$：离 goal 有多远、有没有跟上 reference path
- $\phi_{\mathrm{ctrl}}$：控制量有多激进，惩罚猛打方向盘
- $\phi_{\mathrm{obs}}$：障碍物惩罚，用前面算的 signed distance 喂进来

障碍物惩罚长这样：

$$\phi_{\mathrm{obs}}(d) = w_{\mathrm{coll}} \mathbb{I}(d < 0) + w_{\mathrm{rep}} \max(d_{\mathrm{safe}} - d, 0)^2$$

$d$ 是某个 rollout 某个时间步的最近障碍物距离。第一项是硬碰撞惩罚——$d<0$ 意味着 footprint 被穿透了，给一个 $w_{\mathrm{coll}}$ 的大 penalty。第二项是软避障——离障碍物太近（在 $d_{\mathrm{safe}}$ 以内），就给二次惩罚，越近惩罚越大。

**第四步：softmax 加权平均。** 把 cost 转成权重：

$$\omega^{(r)} = \frac{\exp(-(J^{(r)} - \beta)/\lambda)}{\sum_j \exp(-(J^{(j)} - \beta)/\lambda)}$$

$\beta$ 是 1000 个 rollout 里的最低 cost，做数值稳定用。$\lambda$ 是温度参数——$\lambda$ 小，权重集中在最好的几个 rollout 上；$\lambda$ 大，权重更平均。

然后用这些权重去加权平均所有 rollout 的扰动：

$$\mathbf{u}_h \gets \mathbf{u}_h + \sum_r \omega^{(r)} \boldsymbol{\epsilon}_h^{(r)}$$

更新后的 nominal 控制序列就是一个"软最优"的轨迹。

**第五步：执行第一个 command，往右移一步，下个周期再来。** 这是 receding horizon 的标准做法。

---

## 三道安全防线

paper 在安全上下了狠手，设了三道防线：

**第一道：soft penalty。** signed distance 离 footprint 太近，cost 就上来，rollout 的权重就降。MPPI 自然倾向于选 clearance 大的 rollout。

**第二道：hard flag。** 如果某个 rollout 在任何时间步上 clearance 低于 $d_{\mathrm{safe}}$，直接给这个 rollout 加一个巨大的 penalty（$w_{\mathrm{inf}}$），让它在 softmax 里几乎拿不到权重。

**第三道：post-update validation。** MPPI 算完 nominal 控制序列后，重新 rollout 一次 nominal 轨迹，再检查一遍每个时间步的 clearance。如果 nominal 轨迹有任何一个时间步违反 $d_{\mathrm{safe}}$，**执行 zero-velocity hold，原地不动**，把 nominal 序列 reset 为 0。

第三道是真正的"最后一公里"保险。MPPI 是随机算法，加权平均理论上可能把权重分布到一些 borderline rollout 上，结果平均出来反而落到了 unsafe region。Validation 是兜底，确保发出去的 command 一定是安全的。

---

## 凹 footprint 为什么赢 convex hull

实验里最 striking 的结果是 DoN=1.0 的 corridor 场景。DoN 是 Degree of Narrowness：

$$\mathrm{DoN} = \frac{W_r}{W_p}$$

$W_r$ 是机器人的有效宽度，$W_p$ 是通道的最窄宽度。DoN=1.0 意味着通道恰好等于机器人宽度，再窄一点就过不去了。

T 形 footprint 的有效宽度在某个方向上其实比 T 的凸包要小——因为 T 的凹槽里可以容纳障碍物。DoN=1.0 时，convex hull 算法看到通道宽度等于凸包宽度，直接放弃。EXACT-MPPI 看到 T 的凹槽正好能容纳障碍物，于是穿过去了。

实验结果：DoN 0.6 到 0.9，所有方法都能过；DoN 1.0，只有 EXACT-MPPI 过了，花了 70 秒。Convex-MPPI 和 NeuPAN 全部 fail。

这个 case 直接对应了真实场景：仓库叉车在货架间穿行，叉车前端的叉子和托盘形成的 T 形或 L 形 footprint，凹槽刚好能容纳货架腿。用 convex hull 就过不去，必须用 exact geometry。

---

## 跟 NeuPAN 的对比

NeuPAN 是最直接的 baseline，也是 perception-to-control 范式。它用神经网络（DUNE）学了一个点到机器人距离的 encoder。训练时把 footprint 编码进网络权重，推理时输入点云输出距离。

问题在于：**footprint 一变，网络就得重训。** 实验里有个 case 很说明问题——AgileX Ranger Mini 上加了一个额外负载，footprint 变了。NeuPAN 的 DUNE 重训了，但下游的 NRMP planner 参数是为原 footprint 调的，transfer 过去后机器人卡在 trap 里出不来。

EXACT-MPPI 不需要重训，直接改 footprint 的 polygon 描述就行。这是 analytic 方法的核心优势：**几何变了就改几何参数，不需要重新 amortize。**

性能上，EXACT-MPPI 的 analytic evaluator 比 DUNE 在 GPU 上快 12-19 倍。原因是点到多边形的距离本来就是个 closed-form 计算，神经网络 inference 是矩阵乘加 + 激活函数，前者在 GPU 上是 elementwise arithmetic，后者有更多 memory access 和计算开销。

---

## 跨平台部署

paper 在三个机器人上做了 real-world 实验：

1. **Differential-drive 双臂机器人在 indoor office**：footprint 是带手臂的底盘，过窄门、绕行人、进 tight workspace
2. **AgileX Ranger Mini 在 trap 场景**：这个平台有三种模式——dual-Ackermann、parallel、spin。parallel 模式下，EXACT-MPPI 9 秒脱困，Convex-MPPI 因为凸包太胖直接 fail
3. **Unitree Go2 四足机器狗扛着一根长杆**：footprint 因为长杆变得很长。在 garden 里走，EXACT-MPPI 108 秒走完，Falco 114 秒。在极端窄通道里，EXACT-MPPI 过去了，Falco 因为只用矩形近似 footprint 过不去

这三个平台的 motion model 都不一样，但 EXACT-MPPI 的核心代码——signed distance evaluator + MPPI update——一行没改。改的只是 footprint 的 polygon 描述和 motion model 的 $\mathcal{F}_m$ 函数。这验证了 paper 的核心 claim：**框架是 platform-agnostic 的，换平台只需要换两个配置。**

---

## Hybrid mode 怎么搞

AgileX Ranger Mini 这种 4WS/4WD 平台支持多种 non-skidding motion mode。如果让 MPPI 在连续 command space 里采样，会生成需要轮胎打滑才能执行的动作。所以 paper 把每个 mode 单独跑一遍 MPPI，最后比 cost 选 mode。

每个 mode $m$ 独立跑 Algorithm 1，得到 candidate sequence $\mathbb{U}_m$ 和 cost $J_m$。加 mode switching penalty：

$$\bar{J}_m = J_m + \lambda_{\mathrm{switch}} \mathbb{I}(m \neq m_{\mathrm{prev}})$$

$\lambda_{\mathrm{switch}}$ 是切换惩罚，$m_{\mathrm{prev}}$ 是上一周期的 mode。再加一个 cooldown 计时器防止频繁切换。最后选 $\arg\min_m \bar{J}_m$。

实验里，hybrid mode（dual-Ackermann + parallel + spin 三种都能选）在 DoN=0.9 的窄空间里 106 秒走完，只用 dual-Ackermann 要 140 秒。原因是在某些弯道里 sideways 平移比 forward + 转向高效得多。

---

## 局限性

paper 老实说了几个局限：

1. **只是 local planner**，不管全局路径、语义理解、task-level decision。guidance 假设由上游 module 提供
2. **只用 kinematic model**，没验证 dynamic feasibility，高速、激进机动、崎岖地形、腿足接触这些场景没覆盖
3. **2D planar footprint**，不管 3D body geometry、悬挂障碍物、姿态相关的形状变化
4. **动态障碍物当 quasi-static 处理**，不做显式运动预测。低速度场景够用，密集人群或高速变化场景不够

---

## 我觉得有意思的地方

**Analytic beats learned for structured problems。** 这对当前 "everything should be learned" 的氛围是个好的提醒。点到多边形的距离是个已知 closed-form 函数，没有理由用神经网络去 approximate。神经网络擅长的是 unknown 或难以解析的函数，比如图像特征、语言模式。对 structured geometric computation，直接写公式更快更准更灵活。

**Body-frame transformation + broadcasting 是 GPU 友好设计的关键。** 这个 trick 本身不新，robotics 里古典用法。但 paper 把它做到了极致——固定 footprint 在 body frame，把障碍物点变换过来，全 batched 在 JAX 里编译成 GPU kernel。这种"老 trick + 现代 compute"的组合往往比"全新 trick"更有工程价值。

**Defense-in-depth 的安全设计。** soft penalty + hard flag + post-update validation 三道防线，最后一道是 zero-velocity hold。这是 real-robot deployment 的成熟工程思路——MPPI 是 stochastic controller，理论上有概率输出 unsafe command，必须兜底。

**DoN 这个 metric 揭示了 footprint modeling 的价值边界。** DoN < 0.9 时所有方法都能过，footprint 准不准不重要。DoN 接近 1.0 时，convex hull 算法开始 fail，exact geometry 才显出价值。这告诉我们：exact geometry 不是"永远更好"，是在 clearance 紧张时才决定 feasible vs infeasible。open environment 里用 convex hull 完全够，省下来的计算可以用来采样更多 rollout。

**References 里值得挖的几篇**：
- MPPI 原始论文 Williams et al. 2017: https://arc.aiaa.org/doi/10.2514/1.G001950
- NeuPAN 对比 baseline: https://ieeexplore.ieee.org/document/10885165
- OBCA convex duality 路线: https://ieeexplore.ieee.org/document/9142219
- JAX: https://github.com/jax-ml/jax

总而言之，这篇 paper 给了一个很清晰的 lesson：**不是所有问题都需要 learning，有些问题用 analytic + modern compute 就够了，而且更优雅、更灵活、更快。** 在 robotics 这种 data-scarce、platform-diverse 的领域，这个 lesson 格外重要。

---

# EXACT-MPPI Paper 深度讲解

## 一、核心 Motivation 和 Big Picture

这篇 paper 解决的核心问题是：**地面机器人的 effective footprint 经常是非凸的复杂形状**（比如 forklift 托着 pallet、agricultural platform 拖着 implement、mobile manipulator 携带 object），但在 clearance 接近 footprint 几何细节的狭窄环境中，传统 local planner 的两个简化会丢掉可行解：

1. **Footprint 简化**：将 robot 用 circle、rectangle 或 convex hull 近似，inflatation 了 collision boundary
2. **Map 栅格化**：把 LiDAR point cloud 先栅格化成 occupancy grid 或 ESDF，丢失原始几何细节，并对 grid resolution 和 inflation radius 敏感

EXACT-MPPI 的 idea 非常干净：**跳过 map representation，直接从 point cloud 到 control command**；同时**analytically 计算 point 到 exact polygonal footprint 的 signed distance**，不需要 convex decomposition 或 learned encoder。把这个 signed distance 作为 MPPI 的 safety cost，在 GPU 上 batched 评估成千上万个 rollout。

这是 **perception-to-control** 范式，与 NeuPAN（learning-based 的 perception-to-control）类似，但用 analytic geometry 替代 learned distance encoder，避免 footprint 改变时 retraining。

---

## 二、核心方法：Exact Signed-Distance Evaluation

### 2.1 为什么"signed distance"而不是 unsigned

这里很关键。signed distance $d^{\pm}(\mathbf{p}, \mathcal{B}_{\mathrm{eff}})$ 定义为：
- **negative** 在 footprint 内部（collision penetration 深度）
- **zero** 在 boundary 上
- **positive** 在 footprint 外部（clearance）

为什么需要 signed？因为 MPPI 的 cost function 需要一个**连续可微的标量**来同时表达"碰撞程度"和"接近碰撞程度"。unsigned distance 在 footprint 内部恒为 0，无法区分"刚刚碰上"和"深度嵌入"。Signed distance 提供了 gradient-like 信息让 MPPI 的 importance weight $\omega^{(r)}$ 能合理区分 rollout 质量。

### 2.2 Rollout-frame 变换（关键 trick）

公式 (24)：

$$\mathbf{p}_{h,i}^{b,(r)} = \mathbf{R}(\theta_h^{(r)})^\top (\mathbf{o}_i - \mathbf{t}_h^{(r)})$$

变量解释：
- $\mathbf{o}_i \in \mathbb{R}^2$：obstacle point 在 planning frame 中的坐标
- $\mathbf{t}_h^{(r)} = [x_h^{(r)}, y_h^{(r)}]^\top$：rollout $r$ 在 horizon step $h$ 的位置
- $\theta_h^{(r)}$：rollout $r$ 在 horizon step $h$ 的朝向
- $\mathbf{R}(\theta_h^{(r)})$：2D rotation matrix
- $\mathbf{p}_{h,i}^{b,(r)}$：obstacle point 在 rollout $r$ 的 body frame 中的坐标

**Intuition**：与其在每个 rollout 都旋转、平移 footprint 去匹配 obstacle，不如固定 footprint 在 body frame 中，然后把 obstacle points 变换到 body frame。这是一个非常聪明的 batched-friendly 设计：footprint 顶点 $\mathbf{v}_b$ 只需在程序启动时定义一次，整个 MPPI cycle 不变；只有 obstacle points 被变换。在 GPU 上，对一个固定的 footprint 顶点 tensor 做广播运算比每次重新构造 polygon tensor 高效得多。

### 2.3 Rectangle-cover specialization（rectilinear footprints）

对于 axis-aligned 的正交 footprint（很多 vehicle chassis 和 implement），用 axis-aligned rectangles 的 union 表示：

$$\mathcal{B}_{\mathrm{eff}} = \bigcup_{j=1}^{R} \mathcal{R}_j$$

每个 rectangle 用 center $\mathbf{c}_j$ 和 half-extent $\mathbf{s}_j$ 参数化。点到单个 box 的 signed distance（公式 26）：

$$d_{\mathrm{box},j}^{\pm}(\mathbf{p}) = \|\max(\mathbf{a}_j(\mathbf{p}), 0)\|_2 + \min\left(\max(a_{j,x}, a_{j,y}), 0\right)$$

其中 $\mathbf{a}_j(\mathbf{p}) = |\mathbf{p} - \mathbf{c}_j| - \mathbf{s}_j$（公式 25），$|\cdot|$ 是 elementwise absolute value。

**这个公式的 intuition**：
- $\mathbf{a}_j$ 表示 query point 到 box 各个边的 signed 偏移量（正：在 box 外侧；负：在 box 内侧）
- 第一项 $\|\max(\mathbf{a}_j, 0)\|_2$：当 query 在 box 外时，给出到最近 corner 的 Euclidean 距离；当 query 在 box 内时为 0
- 第二项 $\min(\max(a_{j,x}, a_{j,y}), 0)$：当 query 在 box 内时给出 penetration 深度（取两个轴偏移中较大的那个，即"离最近边的距离"），且只在 box 内为负

最终对 R 个 rectangles 取 min（公式 27）：

$$d_{\mathrm{rect}}^{\pm}(\mathbf{p}, \mathcal{B}_{\mathrm{eff}}) = \min_{j=1,\ldots,R} d_{\mathrm{box},j}^{\pm}(\mathbf{p})$$

这个公式没有 branching、sorting、topology test，**纯 elementwise arithmetic + 一个 vector norm**，对 GPU batched 友好到极致。实验显示在 L-, T-, F-shape 上比 polygon-edge route 快 2-3.34×（Table III）。

### 2.4 General simple polygon（凸或非凸）

对一般 simple polygon（不自交的闭合曲线），用顶点 cyclic sequence $\mathcal{V} = (\mathbf{v}_1, ..., \mathbf{v}_B)$ 表示，B 是顶点数（也等于边数）。

对每条边 $\mathbf{e}_b = \mathbf{v}_{b+1} - \mathbf{v}_b$，先计算 clipped projection parameter（公式 28）：

$$\alpha_b(\mathbf{p}) = \max\left(0, \min\left(1, \frac{(\mathbf{p} - \mathbf{v}_b)^\top (\mathbf{v}_{b+1} - \mathbf{v}_b)}{\|\mathbf{e}_b\|^2}\right)\right)$$

变量解释：
- $\alpha_b(\mathbf{p})$ 是 query point $\mathbf{p}$ 在边 $\mathbf{e}_b$ 上的**投影参数**（clamp 到 [0,1]）
- 分子是 $\mathbf{p} - \mathbf{v}_b$ 在 $\mathbf{e}_b$ 方向上的投影长度
- 分母是边的长度平方，归一化后 $\alpha_b \in [0,1]$ 表示投影落在 segment 内部，$\alpha_b=0$ 或 $1$ 表示 clamp 到端点

对应的 point-to-segment distance（公式 29）：

$$d_{\mathrm{seg},b}(\mathbf{p}) = \|\mathbf{p} - (\mathbf{v}_b + \alpha_b(\mathbf{p})\mathbf{e}_b)\|_2$$

这是 query 到边 $b$ 的最近点距离。

符号通过 ray-casting 的 point-in-polygon test 决定（公式 30）：

$$\sigma(\mathbf{p}, \mathcal{B}_{\mathrm{eff}}) = \begin{cases} -1, & \mathbf{p} \in \mathcal{B}_{\mathrm{eff}} \\ +1, & \mathbf{p} \notin \mathcal{B}_{\mathrm{eff}} \end{cases}$$

最终 signed distance（公式 31）：

$$d_{\mathrm{poly}}^{\pm}(\mathbf{p}, \mathcal{B}_{\mathrm{eff}}) = \sigma(\mathbf{p}, \mathcal{B}_{\mathrm{eff}}) \min_{b=1,\ldots,B} d_{\mathrm{seg},b}(\mathbf{p})$$

**Intuition**：对每条边算 unsigned 最近距离，取 min 得到 boundary 距离；ray-casting 决定内外符号。这个公式的好处是统一处理 convex 和 concave polygon，**不需要 convex decomposition**。对非凸形状（如 Star、Arrow、T-shape），传统的 convex hull 近似会"填满"凹陷部分，丢掉那些本可通行的配置；这个 evaluator 保留了所有几何细节。

### 2.5 计算复杂度

每个 control cycle 评估 $K \times T \times N$ 个 signed-distance query：
- Rectangle-cover：$O(KTNR)$
- Polygon：$O(KTNB)$

paper 用 $K=1000, T=50, N=100$，对应 $5 \times 10^6$ 个 query/cycle，GPU 内存占用约 500MB。

---

## 三、MPPI 控制框架

### 3.1 MPPI 的核心思想

MPPI 是一种 sampling-based predictive control。核心流程：

1. 维护 nominal control sequence $\mathbb{U} = \{\mathbf{u}_0, ..., \mathbf{u}_{T-1}\}$
2. 通过加 perturbation 采样 $K$ 条 rollout：$\mathbf{u}_h^{(r)} = \mathbf{u}_h + \boldsymbol{\epsilon}_h^{(r)}$
3. 用 forward-Euler 离散 kinematics 推 forward（公式 15）：

$$\mathbf{q}_{h+1}^{(r)} = \mathbf{q}_h^{(r)} + \mathcal{F}_m(\mathbf{q}_h^{(r)}, \mathbf{u}_h^{(r)}) \Delta t$$

4. 对每条 rollout 算 finite-horizon cost $J^{(r)}$（公式 16）：

$$J^{(r)} = \sum_{h=0}^{T-1} \left[\phi_{\mathrm{task}}(\mathbf{q}_h^{(r)}, \mathbf{u}_h^{(r)}) + \phi_{\mathrm{ctrl}}(\mathbf{u}_h^{(r)}) + \phi_{\mathrm{obs}}(d_h^{\mathrm{min},(r)})\right]$$

变量解释：
- $\phi_{\mathrm{task}}$：navigation 目标（goal seeking、path following）
- $\phi_{\mathrm{ctrl}}$：control regularization（平滑控制）
- $\phi_{\mathrm{obs}}$：collision penalty（基于 signed distance）
- $d_h^{\mathrm{min},(r)}$：rollout $r$ 在 horizon step $h$ 上的最小 signed distance

5. 用 softmax 归一化的 importance weight 更新 nominal control（公式 17-19）：

$$\beta = \min_r J^{(r)}$$
$$\omega^{(r)} = \frac{\exp(-(J^{(r)} - \beta)/\lambda)}{\sum_{j=1}^K \exp(-(J^{(j)} - \beta)/\lambda)}$$
$$\mathbf{u}_h \gets \mathbf{u}_h + \sum_{r=1}^K \omega^{(r)} \boldsymbol{\epsilon}_h^{(r)}$$

变量解释：
- $\beta$：最低 cost，做数值稳定（避免 overflow）
- $\lambda > 0$：temperature，控制 weight 分布的 sharpness。$\lambda$ 小 → 更 deterministic 地选最优 rollout；$\lambda$ 大 → 更 average 地用所有 rollout

**Intuition**：MPPI 是 path-integral control 的近似。理论上，最优 control 是所有 rollout 的加权平均，权重由 $e^{-J/\lambda}$ 决定。这相当于一个"soft min" over 所有 trajectory，比 hard selection 更稳健。$\lambda$ 起到 temperature annealing 的作用。

### 3.2 为什么 MPPI 而不是 MPC

paper 选择 MPPI 而不是 gradient-based MPC 的理由非常具体：

1. **Non-smooth cost**：signed distance 在 polygon boundary 上不可微（point-to-edge projection 有 clamp），gradient-based 方法需要 smoothing 或 subgradient
2. **GPU-batched friendly**：sampling 天然 parallel，JAX + GPU 编译后可以一次评估百万级 query
3. **Multi-kinematic 通用**：换 kinematic model 不需要 redesign，只需要换 $\mathcal{F}_m$
4. **No gradient information**：避开了非凸优化的 local minima 问题

代价是：MPPI 是 stochastic improvement step，不是 optimal control 的精确解；需要大量采样才能逼近最优；high-dimensional control space 下采样效率低。

### 3.3 Motion models（公式 7-11）

paper 列出 5 种 kinematic model，所有都用 forward-Euler 离散：

**Differential-drive (unicycle)**，输入 $\mathbf{u} = [v, \omega]^\top$：
$$\mathcal{F}_{\mathrm{diff}}(\mathbf{q}, \mathbf{u}) = [v\cos\theta, v\sin\theta, \omega]^\top$$

**Ackermann (bicycle)**，输入 $\mathbf{u} = [v, \delta]^\top$，$\delta$ 是 steering angle：
$$\mathcal{F}_{\mathrm{ack}}(\mathbf{q}, \mathbf{u}) = [v\cos\theta, v\sin\theta, v/L \tan\delta]^\top$$
其中 $L$ 是 wheelbase。

**Omni-motion**，输入 $\mathbf{u} = [v_x, v_y, \omega]^\top$：
$$\mathcal{F}_{\mathrm{omni}}(\mathbf{q}, \mathbf{u}) = [v_x\cos\theta - v_y\sin\theta, v_x\sin\theta + v_y\cos\theta, \omega]^\top$$
注意 paper 中公式 (9) 似乎有 typo，应该是 3 分量 vector。

**Spin-in-place**：$\mathcal{F}_{\mathrm{spin}}(\mathbf{q}, \mathbf{u}) = [0, 0, \omega]^\top$

**Parallel motion**（Ranger Mini 这种 4WS/4WD 平台的 sideways 模式）：
$$\mathcal{F}_{\mathrm{para}}(\mathbf{q}, \mathbf{u}) = [-v_{\mathrm{para}}\sin\theta, v_{\mathrm{para}}\cos\theta, 0]^\top$$
$v_{\mathrm{para}}$ 是 lateral body-frame velocity（沿 y 轴）。

**Intuition building**：所有这些 model 都被写成 $\mathbf{q}_{h+1} = \mathbf{q}_h + \mathcal{F}_m(\mathbf{q}_h, \mathbf{u}_h) \Delta t$ 的统一形式，MPPI 只看到 $\mathcal{F}_m$ 接口。这正是 cross-platform 部署的核心：换平台 = 换 $\mathcal{F}_m$ + 换 footprint polygon。

---

## 四、Safety Penalties 和 Trajectory Validation

### 4.1 Soft obstacle penalty（公式 33）

$$\phi_{\mathrm{obs}}(d) = w_{\mathrm{coll}} \mathbb{I}(d < 0) + w_{\mathrm{rep}} \max(d_{\mathrm{safe}} - d, 0)^2$$

变量解释：
- $d$：当前 rollout state 的最小 signed distance
- $w_{\mathrm{coll}}$：collision penalty（大数，paper 中是 hard penalty）
- $w_{\mathrm{rep}}$：clearance penalty weight
- $d_{\mathrm{safe}}$：desired safety margin
- $\mathbb{I}(\cdot)$：indicator function

**Intuition**：
- 第一项是**碰撞 hard penalty**：$d < 0$ 说明 footprint 被 penetrate，给大权重
- 第二项是**quadratic repulsion**：当 $d < d_{\mathrm{safe}}$ 时（在 safety margin 内）给二次惩罚，离 boundary 越近惩罚越大；$d \geq d_{\mathrm{safe}}$ 时为 0

这个设计的妙处：quadratic 让 cost 在 $d = d_{\mathrm{safe}}$ 处连续可微（在 $d_{\mathrm{safe}}$ 邻域内），有助于 MPPI 的 weight 平滑分布。

### 4.2 Hard feasibility flag + post-update validation

paper 不仅用 soft penalty，还加了**hard feasibility screening**（公式 34-36）：

每个 rollout 维护一个 unsafe flag：

$$\chi^{(r)} = \bigvee_{h=0}^{T-1} (d_h^{\mathrm{min},(r)} < d_{\mathrm{safe}})$$

如果任意 horizon step 违反 safety margin，就给这个 rollout 加 $w_{\mathrm{inf}}$ 大 penalty（公式 35）：

$$\tilde{J}^{(r)} = J^{(r)} + w_{\mathrm{inf}} \mathbb{I}(\chi^{(r)})$$

$w_{\mathrm{inf}}$ 足够大，让 unsafe rollout 的 softmax weight 接近 0。

更严格的是 **post-update validation**：MPPI 更新完 nominal control 后，**重新 rollout 一次 nominal trajectory**，验证每个 horizon step 的 $d_h^{\mathrm{min,nom}} \geq d_{\mathrm{safe}}$。如果失败，**执行 zero-velocity hold**，reset nominal sequence 为 0。

**Intuition**：这是一个 **defense-in-depth** 设计：
1. Soft penalty 让 MPPI 的 importance weight 倾向 safe rollout
2. Hard flag 进一步把 unsafe rollout 从 weighting 中剔除
3. Post-update validation 是 last line of defense：即使 MPPI 的加权平均"漂移"到 unsafe region，validation 也会 catch 并执行 emergency stop

这种设计在 real-robot deployment 中是必要的：MPPI 是 stochastic controller，理论上 always 存在 sampling noise 导致 updated nominal trajectory 落入 unsafe region 的可能。

---

## 五、Hybrid-Mode 扩展

### 5.1 为什么需要 hybrid-mode

AgileX Ranger Mini 这种 4WS/4WD 平台支持多种 non-skidding motion mode（dual-Ackermann、parallel、spin-in-place）。如果把 command space 当成 fully continuous，会生成需要 wheel slip 才能执行的 motion。Selecting among discrete modes 避免这个问题，但引入 mode-switching decision。

### 5.2 Hybrid MPPI formulation

对每个 mode $m \in \mathcal{M}_{\mathrm{hyb}}$，独立跑 Algorithm 1，得到 candidate sequence $\mathbb{U}_m$ 和 cost $J_m$。共享 signed-distance evaluator 和 footprint，只有 rollout dynamics 和 admissible command structure 不同。

加 switching penalty（公式 37）：

$$\bar{J}_m = J_m + \lambda_{\mathrm{switch}} \mathbb{I}(m \neq m_{\mathrm{prev}})$$

变量解释：
- $\lambda_{\mathrm{switch}}$：mode switching penalty
- $m_{\mathrm{prev}}$：上一 cycle 的 active mode

Failed validation 的 mode 设 $\bar{J}_m = +\infty$。还有 cooldown variable $\tau_{\mathrm{cool}}$ 阻止频繁切换。

最终选 $m^* = \arg\min_m \bar{J}_m$。

### 5.3 Actuator post-processing

real hardware 部署需要 deadzone correction：如果 command magnitude 低于 threshold，scale 到 minimum executable value（保持方向 / 符号）。这对 deadzone 大的 actuator 很重要，避免低 command 被硬件 interpret 为 zero。

**Intuition**：这个 hybrid extension 的核心 insight 是把 mode selection 嵌入到 MPPI 的 cost minimization 中。不同 mode 的 cost 是 apples-to-apples comparable 的（都是 finite-horizon cost），所以可以直接 $\arg\min$。这比手工设计 mode-specific navigation rule 优雅得多。

---

## 六、实验结果深度解读

### 6.1 Experiment 1: Signed-Distance Evaluator Benchmark

**对比对象**：DUNE（NeuPAN 的 learned distance encoder）

**结果**（Table II, Fig. 6）：
- 在 100,000 query points 下，EXACT-MPPI 比 DUNE 快：
  - Rectangle: 14.0×
  - Trapezoid: 12.6×
  - Sprayer: 18.9×
  - Double-sided pruner: 16.0×
- Scaling with obstacle count：EXACT-MPPI 一直 favorable，因为 analytic geometry 是 tensorized arithmetic + reduction；DUNE 是 neural network inference，batch size 大时 overhead 显现

**Rectangle-cover vs polygon-edge**（Table III）：
- L-shape: 3.25× speedup
- T-shape: 3.34× speedup
- F-shape: 2.03× speedup

**Deployment overhead**：
- EXACT-MPPI: JIT compilation <1s，subsequent calls reuse
- DUNE: ~1 hour training per footprint

**Intuition**：这个 benchmark 的核心结论是 **analytic > learned for this specific structured problem**。原因：
1. Point-to-polygon distance 是 well-defined closed-form computation，没有"learning"的必要——学习只是 amortize 一个已知解
2. Analytic 实现是 pure arithmetic，和 GPU 的 SIMD/SIMT 架构 perfect match
3. Learned encoder 有 inference overhead（matrix multiplication、activation），且必须经过 training-amortize 才能 efficient

但 paper 诚实地承认：rectangle-cover 内部 overlap 区域的 interior penetration magnitude 与 true signed distance 可能不一致（collision classification 仍然正确）。

### 6.2 Experiment 2: Clearance-Limited Navigation

**核心 metric: Degree of Narrowness (DoN)**（公式 38-39）：

$$\mathrm{DoN} = W_r / W_p$$

其中 $W_r$ 是 effective robot width，$W_p$ 是 minimum passable width。DoN → 1 表示 clearance 极限。

Directional width（公式 39）：

$$W_r(\mathbf{n}) = \max_{\mathbf{x} \in S} \mathbf{n}^\top \mathbf{x} - \min_{\mathbf{x} \in S} \mathbf{n}^\top \mathbf{x}$$

$\mathbf{n}$ 是 translation direction 的 orthogonal unit vector。

**关键 caveat**：paper 自己指出 DoN 是粗略 metric，因为它对 $S$ 和 $\mathrm{conv}(S)$ 给出相同值（凸包不变 directional extrema）。所以 DoN=1 时 convex hull 不可行，但 concave footprint 可能可行——这正是 EXACT-MPPI 的优势所在。

#### Test Case 1: Differential-Drive Corridor DoN Sweep（Table IV）

T-shaped footprint，对比 EXACT-MPPI / Convex-MPPI / NeuPAN。

- DoN 0.6-0.9：所有方法都成功，速度差不多
- DoN 1.0：**只有 EXACT-MPPI 成功**（70.4s），Convex-MPPI 和 NeuPAN 都 fail

这个 case 是 paper 最强的证据：在 DoN=1.0 时，convex hull 把 T-shape 的"凹槽"填满，正好堵死了唯一可行的 passage。EXACT-MPPI 保留了凹槽，让 obstacle point 可以"穿过" T-shape 的凹陷部分。

#### Test Case 2: Omni-Directional Gap（Table V）

L-shaped footprint，omni-motion dynamics。

- DoN 0.83-1.00：EXACT-MPPI 比 Convex-MPPI 快约 5%
- DoN 1.05：**只有 EXACT-MPPI 成功**（0.98 m/s, 20.04s）

#### Test Case 3: IR-SIM Dynamic-Obstacle Corridor（Table VI）

8m 宽 corridor，2 个 dynamic + 8 个 static obstacles，50 trials。

| Method | Success | Time | Path | Speed |
|---|---|---|---|---|
| EXACT-MPPI | **0.92** | 44.25s | 63.62m | 1.482 m/s |
| Convex-MPPI | 0.86 | 41.79s | 63.57m | 1.556 m/s |
| Rectangle-MPPI | 0.78 | 39.66s | 63.00m | 1.609 m/s |
| NeuPAN (convex hull) | 0.76 | 42.95s | 64.05m | 1.491 m/s |

**重要 observation**：EXACT-MPPI **success rate 最高，但 mean speed 不是最快**。Rectangle-MPPI / Convex-MPPI 成功的 trial 更快——因为它们更 conservative，更愿意走"宽路"。EXACT-MPPI 敢于走窄路，所以成功率高，但相应地走窄路时速度低。

这是一个非常重要的 trade-off：**exact geometry 提升的是 feasibility 和 robustness，不是 speed**。

#### Test Case 4: Gazebo Dynamic-Obstacle（Table VII）

Limo + 0.7m extra load → T-shaped footprint。50 trials。

| Method | Success | Time | Path | Speed |
|---|---|---|---|---|
| EXACT-MPPI | **0.96** | 62.47s | 25.05m | 0.40 m/s |
| NeuPAN (convex hull) | 0.65 | 64.47s | 26.49m | 0.41 m/s |

31% 的 success rate gap 非常显著。

### 6.3 Experiment 3: Cross-Platform Deployment

三个平台：
1. **Differential-drive dual-arm robot**（indoor office，Fig. 11）：narrow gate + pedestrian + workspace
2. **AgileX Ranger Mini**（trap scenario，Fig. 12-13）：
   - Parallel motion + EXACT-MPPI: 9s escape
   - Parallel motion + Convex-MPPI: fail（DoN=1.0）
   - Dual-Ackermann + EXACT-MPPI: 13s
   - Dual-Ackermann + Convex-MPPI: 35s
   - Dual-Ackermann + NeuPAN (nominal): pass
   - Dual-Ackermann + NeuPAN (added load, retrained DUNE, transferred NRMP params): **fail**

最后这个 case 非常 important——展示了 NeuPAN 的 deployment 痛点：footprint 变了，DUNE 必须重训，但 NRMP 的 planner params 是为原 footprint tuned 的， transferred 后性能 degrade。

3. **Unitree Go2 + carried bar**（garden + extreme narrow passage，Fig. 14-17）：
   - Garden: EXACT-MPPI 108.96s/45.72m vs Falco 114.78s/50.90m
   - Extreme narrow passage: EXACT-MPPI pass, Falco fail

### 6.4 Experiment 4: Hybrid-Mode Navigation（Fig. 18-19）

Ranger Mini，DoN=0.90 窄空间。

- Dual-Ackermann only: 140s
- Hybrid (dual-Ackermann + parallel + spin): **106s**（24% reduction）

velocity profile 显示 hybrid 用 lateral velocity $v_y$ 走 parallel mode，避开了 dual-Ackermann 必须的 steering correction。

---

## 七、与 Related Work 的对比

### 7.1 Modular Map-Based Navigation

代表方法：APF、VFH、DWA、Hybrid A*、TEB、EGO-Planner、OBCA、CBF。

**三个 limitation**：
1. Map rasterization 引入 resolution sensitivity
2. Footprint 简化成 convex / smooth / inflated proxy
3. Modular separation 让 raw sensor data 的几何细节在 mapping stage 丢失

EXACT-MPPI 跳过 mapping，直接 point cloud → control。

### 7.2 Learning-Based Navigation

**End-to-end**（ALVINN、NVIDIA、ViNT）：data hunger 严重，warehousing/orchard 等 domain 数据稀缺，OOD degrade。

**Model-based learning**（iPlanner、NeuPAN）：
- NeuPAN 用 DUNE 学 distance encoder，NRMP 解 biconvex motion planning
- 与 EXACT-MPPI 三个不同：
  1. NeuPAN 把 footprint encode 到 network weights，footprint 变要 retrain；EXACT-MPPI 只更新 polygon description
  2. NeuPAN 把 robot 建模成 convex set，非凸要 union of convex sets；EXACT-MPPI 处理 simple polygon（含 concave）作为单 object
  3. NeuPAN 用 gradient-based proximal alternating minimization，需要 smoothness；EXACT-MPPI 用 gradient-free MPPI，处理 non-smooth cost

### 7.3 Sampling-Based Predictive Control

MPPI [Williams et al. 2017] 是代表。Log-MPPI 用 heavy-tailed sampling，smooth variants 减少 chatter。但 **collision avoidance 在 MPPI 中较少系统研究**——大多 delegate 给 costmap lookup 或 grid-based SDF。EXACT-MPPI 填补这个 gap。

---

## 八、Intuition Building：为什么这个方法 work

### 8.1 为什么 analytic beats learned for structured geometric problems

Karpathy 你应该 appreciate 这个：learning 的优势是 amortize **未知或难以解析**的函数。但当函数本身有 closed-form 解（如点到 polygon 的距离），learning 反而引入不必要的 approximation error 和 inference overhead。NeuPAN 的 DUNE 是在 amortize 一个 convex optimization subproblem（OBCA 的对偶形式），这个 amortization 在 fixed footprint 下 work，但 footprint 一变就要 retrain。

EXACT-MPPI 的 insight 是：**对 structured problem，直接 analytic evaluation 比 learned surrogate 更快、更准、更灵活**。这与 differentiable physics learning vs analytical physics 的 debate 类似。

### 8.2 为什么 MPPI 比 MPC 更适合这里

MPPI 的核心 advantage 在这个 setting 下格外突出：
1. **Cost non-smooth**：signed distance 在 polygon edge 附近不可微，gradient-based MPC 需要 smoothing
2. **Cost multi-modal**：在 cluttered 环境中，cost landscape 有多个 local minima（不同 passage），MPPI 的 sampling 自然 explore 多个 basin
3. **Batched-friendly**：K=1000 rollout 一次性 evaluate，GPU 用满
4. **Model-agnostic**：换 kinematic model 不需要 redesign optimizer

代价是 sample efficiency 低（high-dim control space 会需要天文数字的 K），但 paper 的 control input dimensionality 是 2-3，K=1000 完全够。

### 8.3 "Body-frame transformation" 的设计哲学

固定 footprint 在 body frame，变换 obstacle points 到 body frame——这个 reverse trick 在 robotics 中其实是 classical idea（如 polygon collision checking 中的"frame locking"）。但 paper 的 contribution 是把它做成 fully batched、JAX-compiled、对每个 rollout × horizon step × obstacle point 都广播：

- Footprint tensor: shape (B, 2)，编译时常量
- Obstacle tensor: shape (N, 2)，每 cycle 更新一次
- Rollout pose tensor: shape (K, T, 3)，MPPI sampling 后产生
- Transformed obstacle tensor: shape (K, T, N, 2)，broadcast + matmul

这种 **broadcasting + JIT compilation** 的组合让 5M query 在 GPU 上完成于 ms 级别。

### 8.4 为什么 DoN=1.0 时只有 EXACT-MPPI 成功

这是 paper 的核心 selling point。在 DoN=1.0 时，passable width 恰好等于 robot 的 effective directional width。Convex hull 把 T-shape 的凹槽填满，导致 convex hull 的 directional width 大于实际 T-shape 的 directional width（在某些方向上）。结果：convex hull 不可行，但 actual T-shape 可行，因为凹槽刚好让 obstacle 穿过去。

这种 scenario 在 warehouse aisle（forklift 托 pallet）、orchard row（tractor 拖 implement）中很常见，是 paper 的 motivation。

---

## 九、Limitations 和 Future Work

paper 自己列了几个：

1. **Local planner only**：不解决 global route、semantic understanding、task-level decision
2. **Kinematic model only**：不验证 dynamic feasibility、aggressive maneuver、rough terrain、contact-rich legged locomotion、articulated trailer
3. **2D planar footprint**：不建模 3D body geometry、height-dependent clearance、overhanging obstacle
4. **No obstacle motion prediction**：dynamic obstacle 当 quasi-static，receding-horizon replanning 补偿。在 dense crowd 或 fast-changing 环境下不够

### 9.1 我（Karpathy）会 push 的方向

1. **3D extension**：把 2D polygon 换成 3D mesh，signed distance 用 point-to-mesh 距离。但 point-to-mesh 的 closed-form 比 point-to-polygon 复杂得多（需要 face、edge、vertex 三种 case）。可以考虑 SDF 网络 + analytic refinement。
2. **Differentiable MPPI for end-to-end learning**：JAX 让 MPPI 全可微。可以用 MPPI 作为 differentiable planner layer，端到端学习 cost weight 和 guidance prior。这就是不同的范式：EXACT-MPPI 是 training-free，但 +differentiable 可以变成 training-amortized。
3. **Obstacle motion prediction**：把 obstacle point 加 time index，预测未来轨迹。可以 learned (social pooling、social GAN) 或 model-based (constant velocity、linear Kalman)。
4. **Adaptive sampling**：MPPI 的 sampling distribution 用 diagonal Gaussian。可以用 normalizing flow 或 diffusion model 学 trajectory-level sampling distribution，让 sampling 集中在 promising region。Log-MPPI 是简单版本。
5. **Multi-robot coordination**：footprint-aware MPPI 可以自然扩展到 multi-robot——把其他 robot 当 dynamic obstacle，footprint 是 their effective polygon。
6. **Risk-sensitive MPPI**：当前 cost 是 expectation。可以用 CVaR 或其他 coherent risk measure 让 planner 更 conservative in tail risk。
7. **Sensor fusion**：目前只用 LiDAR point cloud。可以加 camera semantic 信息，让 $\phi_{\mathrm{obs}}$ 对不同 obstacle class 有不同 weight（e.g. 对 pedestrian 比 static obstacle 更保守）。

---

## 十、Implementation Detail 和工程 Insight

### 10.1 JAX 的角色

JAX 提供：
- **NumPy-like syntax**：写出 vectorized 的 analytic distance code
- **JIT compilation**：XLA 编译成优化 GPU kernel
- **Automatic vectorization (vmap)**：处理 batched 维度
- **Just-in-time shape specialization**：第一次 cycle 编译 <1s，后续 reuse

这是 paper 能 real-time 的关键。同等 NumPy 实现会慢几个数量级。

### 10.2 Padding + validity mask

obstacle count 不固定（LiDAR return count 变化），但 GPU batched 要求 fixed shape。Solution：固定 N，少的用 padding，配 validity mask 在 reduction 时忽略 padded entry。这是 standard trick。

### 10.3 Warm-start nominal control

paper 用 current chassis velocity 做 nominal control 的 warm-start。这让 MPPI 的 nominal sequence 不是从 0 开始，而是从当前状态平滑过渡。重要 for control smoothness。

### 10.4 Receding-horizon update

每 cycle 只执行 nominal sequence 的第一个 command，然后 shift forward。这是 MPC 的 standard receding horizon。但 paper 还加了"validation 失败时 reset 到 0"——这是一个 emergency brake。

---

## 十一、可能的 Critique 和 Open Questions

### 11.1 Computational cost 的隐藏问题

K=1000, T=50, N=100 → 5M query/cycle。如果 cycle @ 30 Hz，全年 GPU 算力 150M query/s。这在 RTX 4060 Ti 上 work，但 embedded platform（Jetson Orin Nano 之类）可能撑不住。paper 没讨论 embedded deployment。

### 11.2 Point cloud preprocessing 的 transparency

paper 说"height filtering and downsampling to fixed budget N"，但没给具体算法。downsampling strategy 会影响 collision evaluation——如果关键 obstacle point 被 downsample 掉，planner 会漏检。Voxel grid？Random？Farthest point sampling？

### 11.3 Weak guidance 的 robustness

paper 假设 weak guidance 可用（target pose / waypoint sequence）。但 guidance quality 直接影响 MPPI 的 task cost $\phi_{\mathrm{task}}$。如果 guidance 错（指向 obstacle），MPPI 会挣扎。没讨论 guidance failure 的 robustness。

### 11.4 MPPI 的 optimality gap

paper 自己承认 MPPI 是 stochastic improvement step，不是最优控制解。没有讨论与 global optimal 的 gap。可以加 lower bound（如 obstacle-free 下的 optimal trajectory）作为 reference。

### 11.5 Rectangle-cover 的 sign correctness

paper 说 rectangle-cover 的 interior penetration magnitude 在 rectangle overlap 区域可能与 true signed distance 不一致，但 collision classification 正确。如果 cost function 依赖 penetration depth（不只是 binary collision），这会引入 noise。不过 $\phi_{\mathrm{obs}}$ 的 $w_{\mathrm{coll}} \mathbb{I}(d<0)$ 项是 binary，所以这个 inconsistency 不影响 hard penalty；只影响 $w_{\mathrm{rep}} \max(d_{\mathrm{safe}} - d, 0)^2$ 的 soft term，且只在 footprint 内部，paper 的 cost 主要关注 exterior clearance。

---

## 十二、相关联想和 Reference

### 12.1 MPPI 相关

- **Williams et al. 2017** "Model Predictive Path Integral Control: From Theory to Parallel Computation" - MPPI 原始论文：https://arc.aiaa.org/doi/10.2514/1.G001950
- **Williams et al. 2018** "Information-Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving" - IT-MPCI：https://ieeexplore.ieee.org/document/8558820
- **Log-MPPI 2022** heavy-tailed sampling for AGV：https://ieeexplore.ieee.org/document/9817020
- **Smooth MPPI 2022**：https://ieeexplore.ieee.org/document/9812077

### 12.2 NeuPAN 和 learning-based navigation

- **NeuPAN 2025** - 直接对比 baseline：https://ieeexplore.ieee.org/document/10885165
- **NeuPAN ROS repo**：https://github.com/hanruihua/neupan_ros
- **iPlanner 2023** - learned perception + classical optimization：https://roboticsconference.org/program/papers/064/
- **ViNT 2023** - foundation model for visual navigation：https://portal-corners.github.io/

### 12.3 Map-based navigation

- **Voxblox 2017** - incremental ESDF：https://ieeexplore.ieee.org/document/8202315
- **OBCA 2020** - optimization-based collision avoidance with convex duality：https://ieeexplore.ieee.org/document/9142219
- **Hybrid A* 2010** - 创始 paper：https://journals.sagepub.com/doi/10.1177/0278364909359210
- **TEB 2017**：https://ieeexplore.ieee.org/document/8202317
- **EGO-Planner 2020**：https://ieeexplore.ieee.org/document/9341305
- **DWA 1997**：https://ieeexplore.ieee.org/document/580977
- **Falco 2020** - sampled-based local planner：https://onlinelibrary.wiley.com/doi/10.1002/rob.21936
- **CBF 2019**：https://ieeexplore.ieee.org/document/8792330

### 12.4 JAX 和 differentiable simulation

- **JAX GitHub**：https://github.com/jax-ml/jax
- **DiffSim / Brax** - JAX-based differentiable physics：https://github.com/google/brax

### 12.5 Robot platforms

- **AgileX Ranger Mini**：https://global.agilex.ai/products/ranger-mini
- **Unitree Go2**：https://www.unitree.com/products/Go2
- **Cartographer 2016**：https://ieeexplore.ieee.org/document/7489212
- **FAST-LIO2 2022**：https://ieeexplore.ieee.org/document/9372441

### 12.6 Agricultural / warehouse robotics

- **Peng et al. 2024** JFR paper on agricultural vehicle motion planning（同一作者 group）：https://onlinelibrary.wiley.com/doi/10.1002/rob.22308
- **Vivaldini et al. 2010** Robotic forklifts：https://ieeexplore.ieee.org/document/5408420

### 12.7 Differentiable robot navigation and learning

- **Value Iteration Networks 2016**：https://arxiv.org/abs/1606.04495
- **MPNet 2019**：https://ieeexplore.ieee.org/document/8791164
- **GNM 2023** - General Navigation Model：https://general-navigation-models.github.io/

### 12.8 Swept volume 和 exact geometry

- **Implicit Swept Volume SDF 2024**：https://dl.acm.org/doi/10.1145/3658120
- **Efficient swept volume 2025**：https://ieeexplore.ieee.org/document/10686794

---

## 十三、总结

EXACT-MPPI 是一个**简洁但深刻**的工作。它的核心 insight 是：当 footprint 复杂且 clearance 紧张时，**analytic geometry + sampling-based control + GPU batched execution** 的组合 beats both (a) map-based modular pipelines with simplified footprints 和 (b) learning-based perception-to-control with learned distance encoders。

**三个核心 contribution 的高度凝练**：

1. **Analytic > learned for closed-form problems**：point-to-polygon distance 是 known function，learning 只引入 overhead
2. **Body-frame transformation + broadcasting**：让百万级 query 在 GPU 上 ms 级完成
3. **Footprint-aware MPPI with hard validation**：soft penalty + hard flag + post-update validation 的 defense-in-depth

**最值得思考的 meta-lesson**：在 robotics 中，**不是所有问题都需要 learning**。当 problem structure 允许 analytic 解时，直接用 analytic，搭配 modern compute (JAX/GPU)，往往 beats end-to-end learning——尤其在 data-scarce、platform-diverse 的 domain。

这个 lesson 对你 Karpathy 来说应该 resonate——你一直在强调 "software 2.0" 的边界，而 EXACT-MPPI 展示了一个 Software 1.5（analytic + parallel compute）依然有强大生命力的 domain。
