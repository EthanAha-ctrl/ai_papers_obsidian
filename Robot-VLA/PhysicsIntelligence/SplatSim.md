---
source_pdf: SplatSim.pdf
paper_sha256: 0c8741dcb67b1cdb5afa3ee85bb2dd185d4b8de583519d96e1b501c42242d247
processed_at: '2026-08-12T10:14:01-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SplatSim 人话版：用真实扫描"画"出仿真机器人

## 1. 一句话版本

你想让 robot 在 simulator 里学会 task，然后直接 deploy 到真实 world。难点在于 simulator 渲染的 image 太假，robot 一看 real camera 就懵了。SplatSim 的 trick 是：**把真实场景拍下来重建一遍，然后让 simulator 里的 robot 在这个重建场景里"摆拍"**。

Project page: https://splatsim.github.io

---

## 2. 为什么 RGB Sim2Real 一直这么难

先 build 一个核心 intuition。

你看那些 zero-shot sim2real 成功的案例——ANYmal 在野外跑酷、ANYmal 跳障碍、in-hand 旋转 cube、dexterous grasping——它们清一色用 depth、point cloud、tactile、proprioception。这些 modality 在 sim 里能精确模拟，因为它们本质是几何与物理量，sim 和 real 的 distribution 几乎一致。

RGB 就完全不一样了。RGB image 是 camera 的"主观感受"，它依赖于 texture、lighting、shadow、material reflectance、camera response function、lens distortion 等几百个 factor。传统 simulator（PyBullet 的 EGL renderer、MuJoCo 的 mujoco renderer）用 mesh + material model + 点光源来渲染，出来的 image 像 2010 年的视频游戏。Robot policy 在这种 image 上训练，到了 real world 一看 real image，就是完全 out-of-distribution。

所以 RGB sim2real 本质上是个 OOD generalization 问题。这个 problem 在 ML 里都是 unsolved，更别说 robotics。

参考 Haonan Yu 的博客: https://www.haonanyu.blog/post/sim2real/

---

## 3. SplatSim 的核心 Insight

传统思路是"**让 sim 渲染得更像 real**"——给 mesh 加 PBR material、加环境光、加 soft shadow、加 tone mapping。这条路走得通但代价巨大，每个 scene 都要 modeling 一遍。

SplatSim 换了个思路：**直接用 real world 数据来渲染**。

具体怎么做？用 3D Gaussian Splatting。Gaussian Splatting 是 2023 年 SIGGRAPH 的爆款工作（Kerbl et al., ACM TOG 2023, https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/），它从一组 RGB image 重建出 scene，用 millions of 3D Gaussian primitives 表示。每个 Gaussian 有 mean、covariance、opacity、color。渲染时把 Gaussian 投影到 image plane 做 alpha blending，出来的 image 与真实 image 几乎像素级一致。

关键 observation：**3D Gaussians 是 explicit 的、point-cloud-like 的 primitive**。它们可以被任意 rigid transform（旋转 + 平移）。这跟 NeRF 完全不同——NeRF 是 implicit neural field，你没法把一个 NeRF 切成两半再各自旋转。

所以 SplatSim 的 trick 是：
1. 拍一段真实场景视频（包括 robot 在 home position）
2. 重建出 Gaussian Splat
3. 把 robot 各个 link 对应的 Gaussians 分割出来
4. 把 object 对应的 Gaussians 分割出来
5. 在 simulator 里跑 trajectory，每个 time step 给出 joint angle 和 object pose
6. 用 forward kinematics 计算每个 link 应该在哪
7. 把对应 Gaussians rigid transform 到那个位置
8. 渲染一张 photorealistic image
9. 把这些 image 喂给 Diffusion Policy 训练
10. Deploy 到 real world

Image: 真实重建 + 模拟器驱动 = photorealistic synthetic data。

---

## 4. 技术细节"人话版"

### 4.1 Rigid Body 变换的公式

每个 3D Gaussian 是个椭球，用 mean $\mu$ 和 covariance $\Sigma$ 描述。当你要把它 rigid transform（$R$ 旋转 + $t$ 平移），公式是：

$$\mu' = R\mu + t \quad (1)$$
$$\Sigma' = R\Sigma R^T \quad (2)$$

人话解释：
- $\mu$ 是 Gaussian 的"中心点"，平移 $t$ 直接加，旋转 $R$ 直接乘
- $\Sigma$ 是 Gaussian 的"形状椭球"，描述它在 x/y/z 三个方向延伸多远。平移不改变形状（你把一个篮球从 A 点搬到 B 点，它还是球）；但旋转改变形状的"朝向"（你把一根长棍从横放转成竖放）。所以 $\Sigma' = R\Sigma R^T$ 这个 conjugation 就是把椭球转过去
- $R^T$ 是 $R$ 的转置，对 rotation matrix 来说等于 $R^{-1}$

### 4.2 坐标系对齐（conjugation 的 intuition）

这部分最容易绕晕。我用一个**翻译类比**：

想象你是个英语小说家，写了一本英文小说，主角叫 "John"。现在你要把这本书翻译成中文。你想加一段新情节"John 去了巴黎"。你怎么加？
1. 先把整本书从中文翻回英文
2. 在英文版里加 "John went to Paris"
3. 再把整本书翻回中文

数学上写就是：$T_{中文}^{-1} \cdot T_{新情节}^{英文} \cdot T_{中文}$。这就是 conjugation。

SplatSim 里：
- "中文版" = splat frame（Gaussians 所在的坐标系，COLMAP 重建的任意起点）
- "英文版" = simulator frame（PyBullet 里的 robot base frame）
- "新情节" = forward kinematics 给出的某个 link 的 transformation $T_{fk}^l$（在 simulator frame 下定义）

把 Gaussians 从 splat frame 搬到 simulator frame，应用 $T_{fk}^l$，再搬回 splat frame，整体公式就是：

$$T = (T_{\mathcal{F}_{robot}}^{\mathcal{F}_{splat}})^{-1} \cdot T_{fk}^l \cdot T_{\mathcal{F}_{robot}}^{\mathcal{F}_{splat}} \quad (3)$$

变量含义：
- $T_{fk}^l$：simulator 算出来的 link $l$ 相对于 robot base 的 transformation
- $T_{\mathcal{F}_{robot}}^{\mathcal{F}_{splat}}$：splat frame 到 simulator frame 的变换（ICP 算出来的）
- $(\cdot)^{-1}$：矩阵逆
- 整个 $T$ 就是最终要在 splat frame 里执行的变换

记住这个 pattern $T_{base}^{-1} \cdot T_{action} \cdot T_{base}$，在 robotics 里到处都是——Jacobian 转换、力矩转换、velocity twist 转换，本质都是 conjugation。

### 4.3 整体 pipeline 用一段话讲

你在 real world 摆好 robot + 物体，拍 100 张 image，跑 Gaussian Splatting 重建出场景。你手动（或用 CAD bounding box）把 robot 的每个 link、gripper 的每个 finger、每个 object 对应的 Gaussians 分组切出来。每个 component 单独存为一个 "Gaussian cloud"。

然后你在 PyBullet 里建一个简化版 scene，用最简单的 mesh（不需要好看，只要几何对就行）。你用 motion planner 或 teleop 在 sim 里收 200 条 demonstration trajectory。

每条 trajectory 的每个 time step，simulator 给你：
- robot 各 joint angle $q_t$
- 各 object 的 position $p_t^k$ 和 orientation $R_t^k$

你用 forward kinematics 把 $q_t$ 转成各 link 的 pose，再用公式 (3) 把对应 Gaussians rigid transform 过去。object 也类似（公式 (4)）。然后调 Gaussian Splatting 的 renderer 渲染一张 image。这条 trajectory 渲染完，就得到一条 photorealistic demo。

最后把这些 demo 喂给 Diffusion Policy 训练。Diffusion Policy 的 input 是 RGB + end-effector pose，output 是 action sequence。训练完直接 deploy 到 real robot。

---

## 5. 实验数据讲了什么故事

主表（Table I）：

| Task | Sim2Sim | Real2Real | Sim2Real (SplatSim) |
|---|---|---|---|
| T-Push | 100% | 100% | 90% |
| Pick-Up-Apple | 100% | 100% | 95% |
| Orange-On-Plate | 97.5% | 95% | 90% |
| Assembly | 85% | 90% | 70% |
| **平均** | **95.62%** | **97.5%** | **86.25%** |

人话解读：

1. **86.25% zero-shot sim2real**——这是 RGB manipulation policy 的 SOTA 水平。之前 RGB zero-shot 几乎不可行，能到 50% 都算奇迹。

2. **退化模式很有意思**：粗 grasp 任务（Pick-Up-Apple）只退化 5%；中精度 task（T-Push、Orange-On-Plate）退化 10%；高精度 task（Assembly）退化 20%。说明 visual gap + dynamics gap 在精度敏感任务上被放大。

3. **数据收集效率**：real world 收 demo 要 20.5 小时（人在现场 teleop），sim 收 demo 只要 3 小时（其中 3 个 task 完全自动化，0 小时人工）。约 6.8× 时间节省。

4. **Assembly 为什么这么难**：这个 task 要把一个 cube 精确放到另一个 cube 上。Splat 渲染没有 dynamic shadow（splat 是 static scan，shadow 是 baked-in 的），gripper 渲染精度也有限，policy 学到的视觉信号有微小偏差，到 real world 就放大成 placement error。

### Augmentation 的"沉默英雄"角色

最戏剧性的 ablation：

| 配置 | Sim2Real 成功率 |
|---|---|
| 不加 augmentation | 21% |
| 加 augmentation | 86.25% |

65 个百分点的提升！augmentation 包括 Gaussian noise、color jitter、random erasing。

为什么这么重要？因为 splat 渲染有几个 systematic artifact：
- Shadow 不跟随 object 移动（baked-in 的）
- Reflection 是 view-dependent 的，但 splat 用 spherical harmonics 只能近似
- Robot cable 这种 flexible component 用 rigid body 近似会有 gap

augmentation 让 policy 学会忽略这些 artifact，专注于真正 invariant 的 feature（object 的位置、robot 的姿态）。

我有个 critical thought：如果给传统 mesh-based sim 也加同样 aggressive 的 augmentation，差距会缩小多少？Paper 没做这个 baseline。我猜 augmentation 能把 mesh-based sim2real 从 5% 提到 40%，但很难超过 60%——因为 mesh-based 的 texture/lighting 偏差是 systematic 的，augmentation 难以完全消除。SplatSim 给了一个更好的"基础"，让 augmentation 能把剩余 gap 跨过去。

---

## 6. 为什么这个方法 Work：4 个 Pillar

我总结 SplatSim 之所以 work 的 4 个核心要素：

### Pillar 1: Photorealism from data, not from modeling

传统 renderer 要你给每个 mesh 配 material、配 texture、配 lighting，调半天。Splat 直接从真实 image 重建，自然 capture 了所有 visual phenomena。你不需要懂 subsurface scattering、不需要懂 BRDF、不需要懂 tone mapping——一段 video 全部包含。

### Pillar 2: Physics from simulator, not from neural network

Splat 不学 dynamics。dynamics 完全 offload 给 PyBullet 这种成熟 physics engine。这种 decoupling 非常 clean——vision 用 photorealistic splat，dynamics 用 accurate physics。两者各做各擅长的事。

对比 Embodied Gaussians（CoRL 2024, https://openreview.net/forum?id=AEq0onGrN2），它直接学 forward model，需要 real world interaction data。SplatSim 不需要任何 real world interaction 数据。

### Pillar 3: Rigid body transformation as the bridge

Gaussian Splatting 的 explicit primitive 让 rigid transform 成为可能。这是 NeRF 做不到的（NeRF 的 implicit field 没法 segment 和 transform）。Rigid body assumption 又恰好匹配大部分 manipulation 任务的特性。这是个 sweet spot。

### Pillar 4: Augmentation as the safety net

残余 gap（shadow、reflection、cable deformation）用 augmentation 弥补。这是工程化的"最后一公里"。

四个 Pillar 缺一不可。这让我想到当年 AlexNet 之所以 work——不只因为 ReLU、不只因为 dropout、不只因为 GPU、不只因为 data augmentation——是所有这些 trick 组合在一起，跨过了 deep learning 的 critical threshold。SplatSim 也是同样的系统工程胜利。

---

## 7. Limitations 与我的延伸联想

Paper 自己承认的 limitation：
- 只能处理 rigid body（cloth、liquid、plant 都不行）
- Dynamic shadow 缺失
- Robot cable 渲染不准

我个人的延伸思考：

### 7.1 与 4D Gaussian Splatting 的整合

Wu et al. (CVPR 2024) 的 4D Gaussian Splatting 可以处理 dynamic scene。如果给 SplatSim 引入 4D Gaussians，理论上可以处理 cable 这种 dynamic component。这是个很自然的 next step。

参考: https://arxiv.org/abs/2310.08528

### 7.2 与 Foundation Model 整合做自动 segmentation

Paper 现在用 CAD bounding box + KNN 做 segmentation，需要 robot 的 URDF。未来可以用 SAM (Segment Anything Model) 或 DINOv2 自动从 splat 里切出 rigid body，连 URDF 都不需要。这会让 framework 更 generalizable。

### 7.3 与 RL 的整合

Paper 只用 BC（Diffusion Policy）。RL 的优势是能探索 BC 没见过的 state。SplatSim 渲染是 real-time 的（Gaussian Splatting 的速度优势），完全支持大规模 RL 训练。如果做 SplatSim + RL，可能能学到比 BC 更 robust 的 policy。但需要解决 reward shaping 的问题——在 sim 里定义 reward 容易，但 splat 渲染不直接影响 reward，所以这条路其实没有额外难度。

参考 Maniwhere (CoRL 2024, https://openreview.net/forum?id=jart4nhCQr) 的 RL sim2real 思路。

### 7.4 与 World Model 的关系

SplatSim 本质是个 "non-learned world model"——rendering 用 splat，transition 用 physics engine。如果未来用 generative world model（如 DeepMind 的 Genie）替代 splat + physics，可能学到更广义的 dynamics。但 sim2real 的 fidelity 可能下降。SplatSim 可以作为 world model 的 ground truth generator。

### 7.5 农业场景的想象空间

Paper 在 conclusion 提到农业应用。我觉得这个方向非常有想象力：
- 摘苹果需要 RGB（颜色判断成熟度）
- 田野环境数据采集极贵
- 但 fruit 有 deformable 特性（4D Gaussians 用得上）
- lighting 在田野完全 uncontrolled（augmentation 必须更强）

如果 SplatSim + 4D Gaussians + 强 augmentation，做 strawberry picking、apple harvesting，可能是个 game-changer。

### 7.6 Photorealism 与 augmentation 的相对贡献（critical thought）

我前面提过这个 ablation 缺失。让我把这个 critical thinking 再展开：

如果 SplatSim 不加 augmentation 也只有 21%，而 mesh-based + augmentation 能到 40%，那么 SplatSim 的边际贡献其实只有 -19% 到 +46% 这段区间。这个数字仍然 impressive，但不是"碾压级"。

更深的问题：**photorealism 真的是 sim2real 的瓶颈吗**？还是 **distribution shift 中的 systematic component** 才是瓶颈？augmentation 主要抹平 random shift，splat 渲染主要抹平 systematic shift。两者可能 equally important，但 paper 的 ablation 没分别 isolate 这两个因素。

如果让我做这个 ablation：
1. Mesh-based sim, no aug → baseline（猜 5%）
2. Mesh-based sim, with aug → augmentation 单独效果（猜 30-40%）
3. SplatSim, no aug → photorealism 单独效果（21%，paper 给了）
4. SplatSim, with aug → 完整方法（86.25%）

这样能 cleanly 分离两个 factor。可惜 paper 跳过了 (1)(2)，只给了 (3)(4)。

---

## 8. 最后的 intuition 总结

如果让我给 Karpathy 你一个 mental model 来记住 SplatSim：

**SplatSim 是个"实景搭台、模拟器唱戏"的 framework**。

你搭个真实舞台（Gaussian Splat 重建），把所有 props（robot links、gripper、objects）都做成真实的"零部件"（分割 Gaussians）。然后让模拟器当导演，喊"joint angle 30 度！"，对应零部件立刻 rigid transform 到位，相机咔嚓一张。几千张照片出来，拿去训练 policy。最后这个 policy 上真实舞台表演，发现布景跟训练时一模一样——sim2real 自然 work。

这个 analogy 抓住了三个 essence：
1. 舞台是真实的（photorealism from data）
2. 导演是模拟器（physics from simulator）
3. 演员移动靠 rigid transform（explicit representation 的优势）

剩下的 shadow/reflection/cable 这些"破绽"，靠 augmentation 这个"后期特效"补一下。整个 framework 就成立了。

---

## 参考资料

1. SplatSim 项目主页: https://splatsim.github.io
2. 3D Gaussian Splatting (Kerbl et al., ACM TOG 2023): https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
3. Gaussian Splatting 官方代码: https://github.com/graphdeco-inria/gaussian-splatting
4. Diffusion Policy (Chi et al., RSS 2023): https://diffusion-policy.cs.columbia.edu/
5. Diffusion Policy 代码: https://github.com/real-stanford/diffusion_policy
6. Gello teleop: https://arxiv.org/abs/2309.09674
7. PyBullet: https://pybullet.org
8. RialTo (RSS 2024): https://arxiv.org/abs/2403.07988
9. Embodied Gaussians (CoRL 2024): https://openreview.net/forum?id=AEq0onGrN2
10. Maniwhere (CoRL 2024): https://openreview.net/forum?id=jart4nhCQr
11. 4D Gaussian Splatting (CVPR 2024): https://arxiv.org/abs/2310.08528
12. PhysGaussian: https://arxiv.org/abs/2311.12198
13. NeRF2Real (ICRA 2023): https://arxiv.org/abs/2305.20046
14. Sim2Real 综述博客 (Haonan Yu): https://www.haonanyu.blog/post/sim2real/
15. SAM (Segment Anything, Meta): https://segment-anything.com/
16. DINOv2 (Meta): https://dinov2.metademolab.com/
17. Robotiq 2F-85 Gripper: https://robotiq.com/products/2f85-140-adaptive-robot-gripper
18. Intel RealSense D455: https://www.intelrealsense.com/depth-camera-d455/
19. Murray, Li, Sastry《A Mathematical Introduction to Robotic Manipulation》: https://www.cds.caltech.edu/~murray/mlswiki/
20. Splat-MOVER (CoRL 2024): https://openreview.net/forum?id=8XFT1PatHy

希望这个"人话版"帮你 build 起来 SplatSim 的 mental model。如果你想深入任何一部分（比如 conjugation 的 Lie group 解释、Diffusion Policy 为什么 robust、或者怎么用 SAM 自动做 segmentation），告诉我，我可以再展开。

---

# SplatSim 详解：用 Gaussian Splatting 消除 RGB Manipulation Policy 的 Sim2Real Gap

## 1. Motivation 与核心 Insight

### 1.1 问题本质

Sim2Real transfer 在 robotics 中一直是 fundamental challenge。观察到一个有趣的现象：所有 zero-shot sim2real 成功的案例（legged locomotion [Anymal parkour]、in-hand rotation [DexTube、Rotating without seeing]、grasping [DextrAh-G]）几乎都使用 **depth、point cloud、tactile、proprioception** 这些 modalities。原因很简单——这些 modalities 在 simulator 中可以被精确模拟，real 与 sim 的 distribution gap 小。

而 RGB image 一直难以 sim2real，本质上是 **out-of-domain generalization 问题**：simulator 渲染出的 image distribution $p_{sim}(I)$ 与 real world 的 $p_{real}(I)$ 之间存在巨大 domain shift。传统 mesh-based rendering pipeline（如 PyBullet、MuJoCo 的 EGL renderer）产生的图像缺乏真实的 texture、lighting、soft shadows、subsurface scattering、reflectance 等细节。

### 1.2 关键 Insight

SplatSim 的核心 idea 非常 elegant：**与其费力去改进 simulator 的 mesh-based renderer，不如直接用 Gaussian Splatting 作为 photorealistic rendering primitive**。

关键观察：
- Gaussian Splatting 的 explicit 3D Gaussian primitive 可以像 point cloud 一样被 rigid transform
- 3D Gaussians 可以从一段真实视频重建，capture 了真实 scene 的所有 visual fidelity
- 只要把每个 rigid body（robot link、object、gripper）的 Gaussians segment 出来，并知道其相对于 simulator 的 transformation，就能渲染任意 configuration 下的 photorealistic image

这就把 sim2real 从 "make sim look real" 的问题，转化为 "use real scan to render sim" 的 Real2Sim2Real 范式。

参考链接：
- SplatSim project page: https://splatsim.github.io
- 3D Gaussian Splatting 原始 paper (Kerbl et al., 2023): https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

---

## 2. Method 技术详解

### 2.1 Rigid Body Transformations in Gaussian Splatting

每个 3D Gaussian 由 mean position $\mu \in \mathbb{R}^3$ 和 covariance $\Sigma \in \mathbb{R}^{3\times 3}$ 表示（外加 opacity $\alpha$ 和 spherical harmonics coefficients $c$，但这两个对 rigid transform 不变）。

对 rigid transformation $T = (R, t)$，其中 $R \in SO(3)$ 是旋转矩阵、$t \in \mathbb{R}^3$ 是平移向量：

$$\mu' = R\mu + t \quad (1)$$
$$\Sigma' = R\Sigma R^T \quad (2)$$

**变量含义**：
- $\mu$：3D Gaussian 在原坐标系下的中心位置（mean of the Gaussian distribution）
- $\mu'$：变换后的中心位置
- $R$：$3\times 3$ rotation matrix，描述旋转
- $t$：$3\times 1$ translation vector，描述平移
- $\Sigma$：$3\times 3$ symmetric positive semi-definite covariance matrix，描述 Gaussian 的形状与朝向
- $\Sigma'$：变换后的 covariance
- $R^T$：$R$ 的转置（对 rotation matrix 也等于其逆 $R^{-1}$）

**Intuition**：
- Mean 是位置，直接 affine transform
- Covariance 描述 "Gaussian blob 的形状"，rotation 会改变它的朝向，translation 不会改变 covariance（因为 covariance 是二阶矩，translation 是一阶矩，平移对中心差分无影响）
- 公式 (2) 是标准的 covariance 旋转公式 $\Sigma' = R \Sigma R^T$，保证变换后仍然是合法的 covariance matrix（symmetric positive semi-definite）

这种 transformation 在 GPU 上可大规模 parallel 处理，所以渲染速度极快（real-time）。

### 2.2 坐标系定义（Coordinate Frames）

Paper 定义了 4 个坐标系来处理 real / sim / splat 之间的对齐：

- $\mathcal{F}_{real}$：real-world coordinate frame，主参考系
- $\mathcal{F}_{sim}$：simulator coordinate frame，与 $\mathcal{F}_{real}$ 对齐
- $\mathcal{F}_{robot}$：real-world robot base frame，与 $\mathcal{F}_{real}$ 对齐
- $\mathcal{F}_{splat}$：Gaussian Splat 重建中 robot base 的 frame（通常与 $\mathcal{F}_{real}$ 有偏差，因为重建是从任意 COLMAP 起点开始）
- $\mathcal{F}_{k-obj,sim}$：simulator 中第 k 个 object 的 frame（origin、no rotation）
- $\mathcal{F}_{k-obj,splat}$：splat 中第 k 个 object 的 frame

关键 transformation：$T_{\mathcal{F}_{robot}}^{\mathcal{F}_{splat}}$ 表示从 splat frame 到 robot frame 的变换矩阵。

**Intuition**：因为 Gaussian Splatting 重建时，COLMAP/SfM 用的是任意初始坐标系，所以 splat point cloud 的 robot base 与 simulator 中的 robot base 不在同一 frame。需要 ICP 把它们对齐，这样 simulator 给出的 joint angle 才能正确地驱动 splat 中的 Gaussians。

### 2.3 Robot Splat Models（核心方法）

分三步：

#### Step 1: ICP 对齐

把 splat 中 robot 的 Gaussians 的 means 提取成 point cloud $P_{splat}$，与 simulator 中 robot 在 home position 时的 ground truth point cloud $P_{sim}$ 用 **Iterative Closest Point (ICP)** 算法对齐，得到 $T_{\mathcal{F}_{robot}}^{\mathcal{F}_{splat}}$。

ICP 的目标函数：
$$T^* = \arg\min_{T} \sum_{i} \| p_i^{sim} - T \cdot p_i^{splat} \|^2$$

通过迭代：1) 找最近邻对应；2) 求 SVD 最优 R, t；3) 收敛判定。

#### Step 2: Robot Link 分割

利用 robot 的 CAD model 提供的每个 link 的 axis-aligned bounding box (AABB)，把 splat 中落在每个 link 的 bounding box 内的 Gaussians 分配给该 link，记为 $\bar{S}_{real}^l$，其中 $l$ 是 link index。

#### Step 3: Forward Kinematics Transformation

当 simulator 给出 joint angle $q_t$ 时，PyBullet 的 forward kinematics 模块计算每个 link 在 simulator frame 下的 transformation $T_{fk}^l$。

那么在 splat frame 下应用此变换的公式为：

$$T = (T_{\mathcal{F}_{robot}}^{\mathcal{F}_{splat}})^{-1} \cdot T_{fk}^l \cdot T_{\mathcal{F}_{robot}}^{\mathcal{F}_{splat}} \quad (3)$$

**变量含义**：
- $T_{fk}^l$：simulator frame 下第 $l$ 个 link 相对于 robot base 的 homogeneous transformation matrix（$4\times 4$）
- $T_{\mathcal{F}_{robot}}^{\mathcal{F}_{splat}}$：splat frame 到 simulator/robot frame 的变换
- $(T_{\mathcal{F}_{robot}}^{\mathcal{F}_{splat}})^{-1}$：simulator/robot frame 到 splat frame 的变换（逆变换）
- $T$：最终在 splat frame 下施加的 transformation

**Intuition（重要！）**：这是经典的 **conjugation** 操作 $T_{base}^{-1} \cdot T_{action} \cdot T_{base}$。这种形式在 Lie group / linear algebra 中无处不在。本质是：
1. 把 Gaussians 从 splat frame 变换到 simulator frame（左乘 $T^{-1}$）
2. 在 simulator frame 下应用 forward kinematics 给出的 link transformation
3. 再变换回 splat frame（右乘 $T$）

为什么需要 conjugation 而不是直接乘？因为 forward kinematics 是在 robot base frame 下定义的（"joint angle = 0 → link 在某个 position"），而 Gaussians 是在 splat frame 下表示的。要在 splat frame 下复用 forward kinematics，必须做坐标变换的共轭。

这一点很关键：很多 sim2real 工作忽略了这种 coordinate frame alignment 的精细处理，导致 Gaussians 渲染位置错误。

### 2.4 Object Splat Models

对于每个 object $k$，单独做一次 splat 重建得到 $S_{obj}^k$，然后用 ICP 对齐到 simulator 中该 object 的 ground truth point cloud，得到 $T_{\mathcal{F}_{k-obj,sim}}^{\mathcal{F}_{k-obj,splat}}$。

当 simulator 给出 object $k$ 的 pose $(p_t^k, R_t^k)$ 时，其在 simulator frame 下的 transformation 为 $T_{fk}^{k-obj}$。则 splat frame 下的最终变换为：

$$T = (T_{\mathcal{F}_{robot}}^{\mathcal{F}_{splat}})^{-1} \cdot T_{fk}^{k-obj} \cdot T_{\mathcal{F}_{k-obj,sim}}^{\mathcal{F}_{k-obj,splat}} \quad (4)$$

**注意**：这里和公式 (3) 不同——object 不在 robot base frame 内，所以需要先从 $k-obj,splat$ 变到 $k-obj,sim$（用 ICP 得到的 alignment），然后从 $k-obj,sim$ 通过 $T_{fk}^{k-obj}$ 变到 simulator 的 world frame，最后从 simulator world frame 变到 splat frame。

### 2.5 Articulated Object（Gripper 分割）

对于 Robotiq 2F-85 这种 parallel jaw gripper，其 links（两个 finger、base）在 3D 空间中不是 axis-aligned 的，bounding box segmentation 失败。

解决方案：用 **KNN classifier**。
- 从 URDF 生成 simulator 中 ground truth 的 labeled point cloud（每个 point 有 link label）
- 训练 KNN classifier（$k$ 近邻投票）
- 把 splat 中每个 Gaussian 的 mean 作为 query point，KNN 预测其 link 归属

这是 simple but effective 的 trick，避免了训练复杂的 segmentation network。

### 2.6 Rendering Pipeline 全流程

对于一条 trajectory $\tau_{\mathcal{E}} = \{(s_1, a_1), ..., (s_T, a_T)\}$，其中 $s_t = (q_t, x_t^1, ..., x_t^n)$，$a_t = (p_t^e, R_t^e)$：

1. 从 $q_t$ 通过 forward kinematics 计算每个 robot link 的 $T_{fk}^l$
2. 从 $x_t^k$ 计算 object $k$ 的 $T_{fk}^{k-obj}$
3. 用公式 (3)、(4) 把每个 rigid body 的 Gaussians 变换到目标位置
4. 用标准 Gaussian Splatting renderer 渲染一张 RGB image $I_t^{sim}$
5. 输出 trajectory $\tau_{\mathcal{G}} = \{(I_1^{sim}, a_1), ..., (I_T^{sim}, a_T)\}$

Gaussian Splatting 的渲染是通过 **$\alpha$-blending** 的 splatting：

$$C(u) = \sum_{i \in \mathcal{N}} c_i \alpha_i \prod_{j<i}(1 - \alpha_j)$$

其中 $c_i$ 是 Gaussian $i$ 的 color（通过 spherical harmonics 计算），$\alpha_i = \sigma_i \exp(-\frac{1}{2}(u-\mu_i')^T \Sigma_i'^{-1} (u-\mu_i'))$，$u$ 是像素投影到 image plane 的位置。

### 2.7 Policy Training 与 Augmentation

**Policy**: Diffusion Policy [Chi et al., RSS 2023]。
- Input: RGB image $I^{sim}$ + end-effector pose $(p^e, R^e)$
- Output: action sequence
- 训练用 conditional denoising diffusion model

**Augmentation（极其关键）**：因为 splat 渲染存在 systematic artifacts：
- 缺少 dynamic shadows（splat 是 static scan，无法随 object 移动产生新 shadow）
- Rigid body assumption 不能处理 robot 电缆等 flexible components
- 反射（specular reflection）的 view-dependent 效果有限

augmentation 包括：
- Gaussian noise addition
- Color jitter（brightness、contrast、saturation、hue）
- Random erasing

**实验数据点**：没有 augmentation 时 Sim2Real 成功率只有 21%，加上 augmentation 后跃升至 86.25%——augmentation 的贡献占比约 65 个百分点，是整个方法的"沉默英雄"。

---

## 3. 实验数据深度解读

### 3.1 主表 Table I 分析

| Task | Sim2Sim | Real2Real | Sim2Real (SplatSim) | Sim human hrs | Real human hrs |
|---|---|---|---|---|---|
| T-Push | 100% | 100% | 90% | 3.0 | 3.5 |
| Pick-Up-Apple | 100% | 100% | 95% | 0.0* | 3.5 |
| Orange-On-Plate | 97.5% | 95% | 90% | 0.0* | 6.0 |
| Assembly | 85% | 90% | 70% | 0.0* | 7.5 |
| **Total** | **95.62%** | **97.5%** | **86.25%** | **3.0** | **20.5** |

(* = automated by motion planner)

**深度分析**：

1. **Sim2Sim vs Real2Real gap (95.62 → 97.5)**：SplatSim 内部其实比 real 略低（虽然差不多），说明 splat rendering 本身在 sim 内就有 small fidelity loss（PSNR 22.62、SSIM 0.78 表明）。

2. **Real2Real vs Sim2Real gap (97.5 → 86.25)**：约 11.25% 的 degradation，这个数字是相当 impressive 的。对比传统 mesh-based sim2real，RGB policy 几乎不可能 zero-shot。

3. **Assembly task 退化最严重（90 → 70）**：Assembly 需要毫米级精度，shadow 缺失、gripper 渲染不准都会放大误差。Pick-Up-Apple 退化最小（100 → 95），因为 grasp tolerance 较大。

4. **Human effort 节省 (20.5 → 3.0 hours)**：约 6.8× 时间节省。三个 task 完全自动化（0 hours），只有 T-Push 用了 Gello teleoperation 收 160 demos。这个效率提升在 robotics 数据采集中是 game-changer。

### 3.2 渲染质量定量（Sec V-C）

- PSNR: 22.62 dB（中等-较好水平；>30 算优秀，>20 算可接受）
- SSIM: 0.7845（>0.8 较好）

这表明 SplatSim 渲染与真实图像之间存在残余 gap，但被 augmentation + diffusion policy 的 robustness 部分弥补。

### 3.3 Augmentation 的 ablation

| 配置 | 成功率 |
|---|---|
| 无 augmentation | 21% |
| 有 augmentation (Gaussian noise + color jitter + random erasing) | 86.25% |

这暗示了：**SplatSim 的核心 photorealism 可能不比 augmentation 贡献更大**。如果 augmentation 能把 21% 提到 86%，那么纯 rendering 改进是否值得这么复杂？这个 ablation 实际上同时揭示了两个事实：
- photorealistic rendering 是基础（没有它，augmentation 没法救）
- 残余 gap 必须靠 augmentation 弥补

一个 critical 的思考：如果给传统 mesh-based renderer 也加上类似的 augmentation，是否能达到接近的效果？Paper 没做这个对比，这是个 open question。

---

## 4. 系统架构图解析（Fig. 2）

整个 pipeline 是 Real2Sim2Real 闭环：

**Top (Training)**:
- (a) Physics simulator (PyBullet) 收集 expert demonstrations，source 可以是 Gello teleop 或 privileged-info motion planner
- (b) Trajectory 喂给 SplatSim-aligned 的 splat models（scene + object）
- 用公式 (3)、(4) 对 3D Gaussians 做 rigid transform
- 渲染出 photorealistic RGB observations $I^{sim}$
- (c) Diffusion Policy 接受 $(I^{sim}, \text{end-effector state})$ 作为 input
- 训练时对 end-effector state 也做 augmentation

**Bottom (Deployment)**:
- Freeze policy
- 直接 deploy 到 real-world robot
- 输入是真实 RGB + 真实 end-effector state

这个架构的 elegance 在于：sim 与 real 用同一个 policy 架构、同一个 observation space（RGB + ee state），无需任何 sim2real adaptation module。

---

## 5. 与相关工作的深度对比

### 5.1 vs RialTo [Villasevil et al., RSS 2024]

RialTo 也是 Real2Sim2Real，但：
- RialTo 用 point cloud（需要 depth camera）
- RialTo 用 Gaussian Splatting 仅为 scene reconstruction + 调参，不用于渲染 robot
- SplatSim 用 RGB only，渲染 robot + object 的所有 interactions

SplatSim 的优势：部署时不需要 depth，对硬件要求低。
RialTo 的优势：point cloud 对 shadow/reflection 不敏感，更鲁棒。

### 5.2 vs Maniwhere [Yuan et al., CoRL 2024]

Maniwhere 做 large-scale RL sim2real，但 deployment 时仍需要 depth。SplatSim 是 pure RGB deployment，这是本质区别。

### 5.3 vs Embodied Gaussians [Abou-Chakra et al., CoRL 2024]

Embodied Gaussians 直接学习 forward model for robot-object interaction（即直接学 neural dynamics），需要 real-world data。
SplatSim 把 dynamics offload 给 physics engine，把 vision 留给 splat——这种 decoupling 让框架更 modular。

### 5.4 vs RoboStudio [Lou et al., 2024]

RoboStudio 用 Gaussian Splatting 做 system identification（参数标定）。
SplatSim 用 Gaussian Splatting 做 data generation for policy training。

### 5.5 vs NeRF2Real [Byravan et al., ICRA 2023]

NeRF2Real 用 NeRF 渲染 bipedal motion sim2real，思路相似。但 NeRF 是 implicit representation，难以做 rigid body manipulation（要把 scene 分割并 transform）。Gaussian Splatting 的 explicit nature 让 rigid transformation 自然——这是为什么 SplatSim 选择 Gaussian Splatting 而非 NeRF 的核心原因。

### 5.6 vs PhysGaussian [Xie et al., 2023]

PhysGaussian 把 physics integration 进 3D Gaussians 做 generative dynamics。SplatSim 反其道而行——把 physics 留给 simulator，把 Gaussians 作为 pure rendering primitive。这是两种不同的 philosophy。

### 5.7 vs Gaussian Splatting to Real World Flight [Quach et al., CoRL 2024]

这篇用 Liquid Networks 做 navigation sim2real，scene 是 static 的，agent 不与 scene 交互。SplatSim 处理 manipulation，agent 直接改变 scene 配置，难度更大。

---

## 6. Build Intuition: 为什么这个方法 Work？

让我帮你 build 几个关键 intuition：

### Intuition 1: Photorealism comes from reconstruction, not rendering

传统 simulator 的渲染管线（mesh + material + lighting model）需要人工建模每一个 component。而 Gaussian Splatting 从真实图像 reconstruction，自然 capture 了所有真实世界的 visual phenomena：subtle lighting gradients、surface texture、camera response function、lens distortion 等等。所以 splat 渲染本质是 "sampling from real visual distribution"，而不是 "synthesizing real-looking images"。

### Intuition 2: Explicit > Implicit for Manipulation

NeRF 也能做 photorealistic rendering，但 NeRF 的 implicit field 难以 "split" 和 "transform"。Gaussian Splatting 的每个 primitive 是 explicit entity，可以被 assign 到 rigid body、被 transformed、被 rendered independently。这种 explicitness 让 manipulation 的 rigid body assumption 完美契合——只要物体是 rigid，Gaussians 就可以跟着 transform。

### Intuition 3: Augmentation 是补丁，不是缺陷

很多人可能觉得 augmentation 占了 65% 的改进是 weakness，但换个角度：augmentation 让 policy 学到了 invariance，这种 invariance 是 sim2real 的真正 robustness 来源。splat 渲染提供了 photorealistic baseline，augmentation 提供了 distributional robustness。

### Intuition 4: Conjugation 是 coordinate frame alignment 的数学本质

公式 (3) 看似复杂，本质是 Lie group 上的 adjoint action: $Ad_g(h) = g h g^{-1}$。这种 pattern 在 robot control 中无处不在：
- Jacobian 在不同 frame 间的转换
- Force/torque 在 sensor frame 与 end-effector frame 间的转换
- Velocity twist 在 base frame 与 tool frame 间的转换

理解这一点后，公式 (3) 不再是 ad hoc，而是标准操作。

### Intuition 5: Sim2Real 退化随 task precision 单调增加

从实验数据看：
- Pick-Up-Apple（粗 grasp）：5% 退化
- T-Push、Orange-On-Plate：10% 退化
- Assembly（毫米级 placement）：20% 退化

这暗示了 sim2real gap 不仅是 visual 的，还有 dynamics 的。Assembly 的 dynamics 涉及 contact、friction、friction-induced positioning error，这些 splat 渲染无法 capture。

---

## 7. Limitations 与 Open Questions

Paper 自己承认的 limitations：
- 只能处理 rigid body（无法处理 cloth、liquid、plants）
- 缺少 dynamic shadows 和 reflections
- 不能很好地处理 cables 等 flexible components

我认为更深层的 open questions：

1. **Augmentation 与 photorealism 的相对贡献**：如果给 mesh-based sim 也加同样 augmentation，能到多少？如果 SplatSim 不加 augmentation 也只有 21%，那么 SplatSim 的核心贡献是否真的有那么大？这个 ablation 没做，是个 gap。

2. **Gripper 的 KNN segmentation 鲁棒性**：如果 gripper 有 reflective surface（金属），splat 重建可能不准。KNN 训练用的 ground truth 是 URDF 的，但真实 gripper 表面可能与 URDF 有偏差。

3. **Dynamic lighting 与 shadow**：当 object 移动到不同位置时，原来 static splat 中 baked-in 的 shadow 不会更新。这意味着当 robot arm 遮挡某个区域时，object 上仍然显示原来未遮挡时的 shadow pattern。这是 systematic artifact。

4. **Generalization to unseen objects**：如果一个新物体没在 splat 重建中被 scan，整个 pipeline 失败。需要 object splat model。未来可能需要 few-shot splat 重建。

5. **Multi-view 一致性**：如果有两个 camera（paper 用 2 个 Intel Realsense D455），每个都需要 splat。但 splat 重建是从一个 video 做的，多个 view 间的 photometric consistency 需要 verify。

6. **与 diffusion policy 的协同**：为什么 diffusion policy 在 SplatSim 上表现这么好？可能因为 diffusion model 的 denoising training objective 本身就是一种 implicit data augmentation，让 policy 更 robust to distribution shift。

---

## 8. 我的一些延伸联想（hallucination 区）

### 8.1 与 Genie / World Models 的关系

SplatSim 本质上是一个 "non-learned world model"：用 Gaussian Splatting 做 rendering，用 PyBullet 做 transition dynamics。这让我想到 Genie（DeepMind 的 generative interactive environment）。如果用 Genie 这种 generative world model 替代 splat + physics，是否可以做 sim2real？反过来，SplatSim 可以作为 Genie 的 ground truth training data generator。

### 8.2 与 Differentiable Rendering 的关系

公式 (1)(2) 的 transformation 是 differentiable 的。这意味着可以 backprop 从 image loss 到 joint angle，可能用于 differentiable simulation、system identification、或 visual servoing。这是一个 unexplored direction。

### 8.3 与 4D Gaussian Splatting 的关系

Wu et al. (CVPR 2024) 的 4D Gaussian Splatting 可以处理 dynamic scene。如果未来 SplatSim 引入 4D Gaussians，可以处理 cables、flexible components、甚至 soft robotics。这将极大扩展其 applicability。

### 8.4 与 Foundation Models 的整合

CLIP / DINOv2 / SAM 等 vision foundation model 可以用来：
- 自动 segmentation（替代 manual segmentation）
- 评估 rendered image 与 real image 的 embedding 距离（更 semantic 的 PSNR）
- 提供 language-conditioned manipulation

### 8.5 农业应用（paper 提到的 future work）

Paper 在 conclusion 中提到农业应用（pruning、harvesting）。这是非常 promising 的方向：
- 田野条件下数据采集 expensive
- 农产品（fruit、leaf）有丰富 color/texture，RGB 必要
- 但农产品有 deformable 特性，需要 4D Gaussians 或 PhysGaussian 扩展

### 8.6 与 RL 的整合

Paper 只用 behavior cloning（diffusion policy）。如果用 SplatSim 做 RL，splat 渲染速度（real-time）让大规模 RL 训练成为可能。这对探索性任务（探索新策略）特别有用，因为 BC 需要 expert demo，但 RL 可以自主探索。

参考: Maniwhere、ISAACTAB、Legged Robotics 都在做 RL sim2real，但都用 depth。SplatSim 是 RGB RL sim2real 的可能 path。

### 8.7 与 Digital Twin 的关系

SplatSim 本质是在构建一个 photorealistic digital twin。这可以扩展到：
- Remote operation (splat 渲染 + haptic feedback)
- Predictive maintenance (用 splat 渲染不同 failure mode)
- Human-robot collaboration (splat 渲染 future robot actions 给 human 看)

---

## 9. 总结性 Intuition

SplatSim 之所以 work，是因为它找到了一个 sweet spot：
1. **Photorealism from data, not from modeling**：用 splat 重建 capture 真实视觉
2. **Physics from simulator, not from neural network**：用 PyBullet 保证 dynamics 准确
3. **Rigid transformation as the bridge**：用 rigid body assumption 连接两者
4. **Augmentation as the safety net**：弥补残余 gap

这四个要素缺一不可。它不是 "用 splat 替代 mesh" 这么简单，而是一种 systems-level thinking：让每种 component 做自己最擅长的事。

---

## 参考资料

1. **SplatSim 项目主页**: https://splatsim.github.io
2. **3D Gaussian Splatting (Kerbl et al., ACM TOG 2023)**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
3. **Diffusion Policy (Chi et al., RSS 2023)**: https://diffusion-policy.cs.columbia.edu/
4. **Diffusion Policy IJRR 版本**: https://arxiv.org/abs/2303.04137
5. **Gello teleoperation framework**: https://arxiv.org/abs/2309.09674
6. **PyBullet**: https://pybullet.org
7. **RialTo (RSS 2024)**: https://arxiv.org/abs/2403.07988
8. **Maniwhere (CoRL 2024)**: https://openreview.net/forum?id=jart4nhCQr
9. **Embodied Gaussians (CoRL 2024)**: https://openreview.net/forum?id=AEq0onGrN2
10. **RoboStudio**: https://arxiv.org/abs/2408.14873
11. **PhysGaussian**: https://arxiv.org/abs/2311.12198
12. **4D Gaussian Splatting (CVPR 2024)**: https://arxiv.org/abs/2310.08528
13. **NeRF2Real (ICRA 2023)**: https://arxiv.org/abs/2305.20046
14. **Intel RealSense D455**: https://www.intelrealsense.com/depth-camera-d455/
15. **Robotiq 2F-85 Gripper**: https://robotiq.com/products/2f85-140-adaptive-robot-gripper
16. **Gaussian Splatting to Real World Flight (CoRL 2024)**: https://openreview.net/forum?id=ubq7Co6Cbv
17. **Splat-MOVER (CoRL 2024)**: https://openreview.net/forum?id=8XFT1PatHy
18. **GraspSplats (CoRL 2024)**: https://openreview.net/forum?id=pPhTsonbXq
19. **DeformGS (WAFR 2024)**: https://arxiv.org/abs/2404.01382
20. **ManiSkill3**: https://arxiv.org/abs/2410.00425

---

如果你想 build 更深 intuition，我建议从几个角度切入：
1. 实际跑一遍 Gaussian Splatting 的官方 repo（https://github.com/graphdeco-inria/gaussian-splatting），感受 explicit primitive 的 power
2. 看 Diffusion Policy 的 code（https://github.com/real-stanford/diffusion_policy），理解为什么 diffusion objective 对 sim2real 自然 robust
3. 重新看一下 robotics 中的 coordinate frame transformation 与 Lie group 的关系（推荐 Murray, Li, Sastry 的《A Mathematical Introduction to Robotic Manipulation》）

这个 paper 看似只是 sim2real 的一个 incremental improvement，但它揭示了 robotics 中 representation 选择的深层哲学：在 explicit、physical、photorealism 之间找 balance。
