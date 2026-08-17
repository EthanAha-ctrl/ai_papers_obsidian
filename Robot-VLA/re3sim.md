---
source_pdf: re3sim.pdf
paper_sha256: 6439f06ea9d7a3d32b014a3af67822072ffd79ddf7f3e471e5d62cb0541a11da
processed_at: '2026-08-11T21:11:25-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 RE³SIM

## 一句话总结

给真实桌面拍个 3D 照片, 在 simulator 里复刻一个看起来一样的场景, 让 robot 在 simulator 里自己练几千次, 练完直接搬到真机上能用, 中间不需要任何 fine-tune.

Paper link: http://xshenhan.github.io/Re3Sim/

---

## 为什么这件事以前做不好

sim-to-real 一直有两个老大难:

**Visual gap**: simulator 渲染太干净. 你看 Isaac Sim 渲染出来的图, 像 Pixar 动画, 桌面是纯色, 光照完美, 物体纹理规整. 真机 RealSense 拍出来的图带 sensor noise, 有 motion blur, 桌面上有划痕, 光照还随时间变. policy 在 sim 学完, 真机一拍就 OOD, 直接废.

**Geometric gap**: simulator 里的物体都是完美 CAD model. 一个 cube 就是 perfect cube. 真实世界里的 cube 边角磨圆了, 表面有指纹, 桌面也不是水平. 这种小差异让 collision 计算不准, gripper 抓的位置一偏就掉.

---

## RE³SIM 的核心 trick

把 reconstruction 拆成两半, 各干各的:

**Background (桌面、墙、basket)**: 用 3DGS 重建 + 渲染. 3DGS 渲染出来视觉上跟真实照片几乎一样, 但 3DGS 是一组 static Gaussians, 不能跟 physics engine 互动. 不过 background 本来就不动, 所以没关系.

**Foreground (被 manipulate 的物体)**: 用 OpenMVS / ARCode 重建 mesh, 在 Isaac Sim 里跑 physics, 用 ray-tracing 渲染. 因为物体要被抓起来移动, 必须有 mesh 才能算 collision.

最后用 ground-truth depth 做 Z-buffer, 把 background 的 3DGS 图和 foreground 的 mesh 图合成一张.

这个分离是整篇 paper 最聪明的地方. 一起用 3DGS 重建? 物体没法跑 physics. 一起用 mesh 重建? 背景视觉不够逼真. 分开做, 各取所长.

3DGS 原理可以看这个: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

---

## 怎么跟真机对齐

sim 里的 3DGS 坐标系和真机 robot base 坐标系不一样, 不对齐就抓空.

做法很土但 work:
1. 真实桌上放一个 ArUco marker
2. 用 depth camera 拍一张, 拿到 marker 周围的 partial point cloud
3. 从重建的 mesh 上采一个 complete point cloud
4. 跑 ICP (Iterative Closest Point), 求一个 6-DoF transform 把两个 cloud 对齐

ICP 简单讲就是反复做两件事: (a) 找最近的点对, (b) 算一个 R, t 让这些点对距离最小. 跑几百次迭代就收敛了.

ICP 介绍: https://en.wikipedia.org/wiki/Iterative_closest_point

---

## 数据怎么来——这是最爽的部分

不用人 teleop. 写一个 "作弊 policy" $\pi_{\text{priv}}$:

- 它知道物体的 ground-truth 6D pose (真机拿不到这种信息)
- 给它几个 keyframe: 抓取前 pose, 放置位置
- 用 RRT-Connect 在 joint space 自动规划轨迹
- 失败的 rollout 直接扔掉 (rejection sampling)
- 物体位置随机, robot base 位置随机

8 张 RTX 4090, 16 个 parallel Isaac Sim, 12 分钟采 100 条 trajectory. 同样数据量真机 teleop 要人坐那儿 50 分钟, 还要熟练 operator.

100 条 sim trajectory 训 ACT + DINOv2, 真机 zero-shot 58% average success rate. 这就是 paper 的 headline 数字.

ACT 介绍: https://tonyzhaozh.github.io/aloha/
DINOv2 介绍: https://arxiv.org/abs/2304.07193

---

## 关键实验数字

| 任务 | RialTo (baseline) | 真机数据 50 eps | RE³SIM 100 eps sim |
|---|---|---|---|
| Pick bottle → basket | 0.4 | 0.8 | 0.75 |
| Stack cubes | 0 | 0.15 | 0.25 |
| Place vegetable on board | 0.45 | 0.6 | 0.75 |

几个有意思的观察:

1. RE³SIM 在 place vegetable 任务上**超过**真机数据训出来的 policy (0.75 vs 0.6). 反直觉. 原因是 motion planner 走最短路径, trajectory 干净; human teleop 有 pauses, jitter, 反而干扰学习.

2. Stack cubes 全员低分. 因为 stacking 对精度要求极高, 5mm 误差就让 cube 倒. 这任务真的难.

3. Sim 和 real success rate 的 Pearson correlation 是 0.924. 这数字很重要, 意味着你可以信 sim 上的成绩, 不用每次迭代都跑真机.

---

## augmentation 重要得很

3DGS 渲染太干净, 没有 sensor noise, 没有 motion blur. 直接拿去训, 真机一上就废 (Real SR 0.25).

加了 Gaussian Noise 之后, Real SR 从 0.25 跳到 0.8. 就这么简单. 因为 RealSense D435i 的 RGB sensor 本来就有 readout noise, sim 里没模拟这个 noise, policy 就 overfit 到 "perfect pixel" 上.

| Augmentation | Sim SR | Real SR |
|---|---|---|
| None | 0.77 | 0.25 |
| + Gaussian Noise | 0.94 | **0.8** |

ColorJitter 模拟光照变化, 在 darkness 测试里发挥作用. 训练时见过 jittered color, 部署时房间暗一点也能 work.

---

## 时间成本 (这是 paper 卖点)

| 步骤 | 时间 |
|---|---|
| 拍背景 video | 51.5 秒 |
| 拍单个物体 (ARCode) | 60.5 秒 |
| 重建 + 对齐 | < 3 分钟 |
| 采 100 条 sim trajectory | 12-14 分钟 |
| 真机 teleop 50 条 | ~50 分钟 + 人工 |

整个 real-to-sim pipeline 一个人 5 分钟搞定, 之后机器自己跑数据. 这才是 paper 真正的 contribution: **让 sim-to-real 变成一个 repeatable, 低人力成本的工程流程**, 而不是每次换 task 都要重新 teleop.

---

## 还做不了什么 (limitations)

1. **Rigid object only**. 3DGS 是一组 static Gaussians, 想模拟 deformable (衣服、海绵) 或 articulated (抽屉、剪刀) 都不行. 这是 3DGS 本身的限制.

2. **Transparent / reflective 物体不行**. MVS 和 3DGS 都假设物体是 Lambertian (漫反射). 玻璃杯、镜面金属重建出来全是 noise. Evo-NeRF 那种专门为透明物体设计的方法是必要的补充. Evo-NeRF: https://arxiv.org/abs/2210.11989

3. **Rule-based policy 写不了复杂任务**. "Pick nearest object" 这种简单规则能写, 但 "把书翻到第 50 页" 这种就完蛋. 复杂任务还是要 human demo 或 RL.

4. **Physics 参数靠 default**. 物体的 mass, friction 都是猜的, 没有 system identification. 对 pick-and-place 影响不大, 但对 push, pivot 这种 friction-sensitive 任务就有问题.

---

## 这篇 paper 真正的意义

它没提出新 algorithm, 没新 theory. 价值在 **engineering integration**: 把 COLMAP, OpenMVS, ARCode, 3DGS, Isaac Sim, PhysX, RRT-Connect, DINOv2, ACT 这堆 off-the-shelf 东西拼成一个端到端 pipeline, 用 58% zero-shot success rate 证明 sim-to-real gap 是个工程问题, 不是 fundamental problem.

对社区的信号是: 你不用再纠结 "sim 渲染太假怎么办" 了, 3DGS + 物理引擎 hybrid 已经够好. 你可以放心大规模采 sim 数据训 policy. 接下来的瓶颈转移到 (a) 怎么重建 deformable 物体, (b) 怎么写复杂 task 的 privileged policy, (c) 怎么做 system identification 把 mass / friction 估准.

---

## 给你的直觉

如果你想在自家 lab 复现这套东西, 最短路径大概是:

1. 装个 Isaac Sim, 写个简单 task
2. 用 Polycam 手机 app 扫桌面 (虽然质量差点但能用)
3. 用 ARCode 扫物体
4. 在 Isaac Sim 里写个 rule-based policy + RRT-Connect, 跑 100 条 trajectory
5. ACT + DINOv2 训 policy
6. 真机部署, 加 Gaussian Noise augmentation

不一定能复现 58%, 但 30-40% 应该可行. 重点是 reconstruction + alignment 这一坨弄准, 别的都不难.

Polycam: https://poly.cam
ARCode: https://ar-code.com/
Isaac Sim: https://developer.nvidia.com/isaac-sim

---

要再展开哪块都行, 比如 3DGS 训练细节、RRT-Connect 双向搜索、ACT 的 action chunking 怎么 work、ICP 收敛性怎么样.

---

# RE³SIM: Real-to-Sim-to-Real 的一个完整工程化方案

Karpathy 你好, 这篇 paper 我读完之后觉得它真正贡献不在 algorithmic novelty, 而是在 engineering integrations —— 把 MVS reconstruction, 3D Gaussian Splatting (3DGS), physics engine, privileged policy, IL 这条链条端到端跑通, 并用 zero-shot 58% average success rate 来证明 sim-to-real gap 真的可以小到 deployable. 下面我把整个系统从底层几何到顶层 policy 拆开讲, 顺便 build 一些 intuition.

Paper link: http://xshenhan.github.io/Re3Sim/
3DGS 原始 paper: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
COLMAP: https://colmap.github.io/
OpenMVS: https://cdcseacave.github.io/openMVS/
ARCode: https://ar-code.com/
Polycam: https://poly.cam/
SuGaR: https://anttwo.github.io/sugar/
NeRF: https://www.matthewtancik.com/nerf
DINOv2: https://arxiv.org/abs/2304.07193
ACT / ALOHA: https://tonyzhaozh.github.io/aloha/
AnyGrasp: https://github.com/graspnet/anygrasp_sdk
RialTo (Torne et al. 2024): https://arxiv.org/abs/2407.14000
SplatSim: https://arxiv.org/abs/2409.10161
RoboGSim: https://arxiv.org/abs/2411.11839
Isaac Sim: https://developer.nvidia.com/isaac-sim
PhysX: https://nvidia-omniverse.github.io/PhysX/
MuJoCo: https://mujoco.org/
SAPIEN: https://sapien.ucsd.edu/
MimicGen: https://mimicgen.github.io/
RoboCasa: https://robocasa.ai/
Franka FR3: https://www.franka.de/

---

## 1. Paper 的核心 thesis

Sim-to-real 之所以痛, 主要是两类 gap:

1. **Geometric gap**: simulation 里的 CAD model 是 idealized 的, 真实世界里的 object / table 有凹凸、划痕、不规则边缘; collision shape 跟 visual mesh 跟真实物体对不齐, 直接导致 grasp / contact 时 physics 出错.
2. **Visual gap**: simulator 渲染出来太干净 (texture mapping, PBR shader 都太完美), 而真机 RGB 带噪声、motion blur、defocus、lighting drift; vision-based policy 训练在 sim, 部署到 real 就 OOD.

RE³SIM 想做的, 是把整个真实 scene 通过 reconstruction → rendering 重新 import 到 simulator, 让 simulator 里的 observation 视觉上和真机接近, collision 形状也接近真实物体. 在这之上用 privileged policy 大规模自动采 expert demo, 训 IL policy, 直接 zero-shot 部署到真机.

实验结论是: 100 条 sim trajectory, 训 ACT + DINOv2, 真机 zero-shot 58% 平均 success rate, 在 pick-and-drop bottle 任务上达到 0.75, 跟 50 条 human teleop 数据训出来的 0.8 持平.

---

## 2. 系统架构 (Fig. 2 对应)

RE³SIM 分成 4 个 module, 严格 sequential:

### 2.1 Mesh Recovery
- **Background mesh**: 用 COLMAP 做 SfM 估计 camera pose + sparse point cloud, 再用 OpenMVS 算 dense mesh. ARCode / Polycam 这类商业软件也行, 但作者实测 OpenMVS 在 flat plane 上更稳, 没有零碎 protrusions / depressions, 对后续 physics simulation 重要 (mesh 上的小毛刺会让 object 在 table 上抖动).
- **Foreground object mesh**: 用 ARCode, 自动 segmentation, 质量足够做 collision body.
- **Post-processing**: void filling + surface smoothing, 否则 reconstructed mesh 上有空洞和锐边, 物体在 sim 里放不平.
- Mesh 用 USD 格式存储, 方便 Isaac Sim ingest.

### 2.2 Hybrid Visual Rendering
这是 paper 设计上最 tricky 的地方. 直觉做法是把 background + object 一起 3DGS 重建一遍, 但作者故意分开做:

- **Background**: 用 3DGS 重建 + 渲染. 因为 background 占据视野的绝大部分 pixel, 视觉 fidelity 决定 sim-to-real visual gap 的主成分.
- **Foreground object**: 用 mesh-based ray-tracing 渲染. 因为 object 在 manipulation 中要被 pick, place, drop, 需要在 physics simulator 里实时更新位姿; 而 3DGS 是 unstructured point set, 给每个 Gaussian 做 rigid transform 是可以的, 但 deformation, contact response 都很难处理; object 的 visual fidelity 对整体视觉感知贡献又小 (object 在图像里占比小).

Hybrid composition: 用 ground-truth depth (来自 physics simulator, 知道每个 object 的 pose) 做 Z-buffer, 决定每个 pixel 取 background 的 3DGS 渲染还是 foreground 的 mesh 渲染.

SuGaR 是被作者明确 reject 的方案, 因为 SuGaR 把 3DGS align 到 mesh 上做 reconstruction, 对 flat plane (比如桌面) 容易出现 protrusions / indentations, 一旦 object 在桌面上滑动就会感觉不真实.

### 2.3 Real-World Alignment
重建出来的 3DGS 坐标系和 robot base 坐标系不一致, 需要一个 6-DoF alignment:

1. 在真实桌面放 ArUco marker, 给出 marker 在 robot base frame 的位姿.
2. 拍一张图, 用 depth camera 拿到 marker 周围的 partial point cloud.
3. 把 reconstructed mesh 上对应的 marker 区域采样成 complete point cloud.
4. 用 **ICP (Iterative Closest Point)** 配 point-to-plane metric 求 R, t, 让 partial cloud 对齐到 complete cloud.

为什么用 point-to-plane 而不是 point-to-point? point-to-plane 把 residual 投影到 target surface 法向上, 对 flat plane (桌面、墙面) 的 alignment 收敛更快, 因为它鼓励 source point 沿 target surface "滑动" 而不是垂直穿透. 公式:

$$
\min_{R,\,t} \sum_i \big( (R\,p_i + t - q_i)^\top n_i \big)^2
$$

- $p_i$: source point cloud 的第 i 个点 (depth camera 拿到的 partial)
- $q_i$: target point cloud 的第 i 个点 (reconstructed mesh 上对应的点)
- $n_i$: target surface 在 $q_i$ 处的法向
- $R$: 3×3 rotation matrix
- $t$: 3D translation vector
- $(R\,p_i + t - q_i)$: 把 source 变换到 target frame 后的 residual
- 投影到 $n_i$ 上: 只惩罚沿法向的偏差

ICP reference: https://en.wikipedia.org/wiki/Iterative_closest_point ; Low 2004 的 point-to-plane 实现: https://www.cs.unc.edu/~low/ ...

### 2.4 Expert Data Collection
有了 reconstructed scene + aligned robot, 就可以用 **privileged policy** $\pi_{\text{priv}}(a_t \mid o_{\text{priv},t})$ 自动采数据:

- $o_{\text{priv},t}$: privileged observation, 包含 object 的 ground-truth 6D pose, gripper state, etc. (这些真机拿不到)
- $a_t$: action (joint command 或 end-effector pose)
- 用 RRT-Connect (Kuffner & LaValle 2000) 在 joint space 做 motion planning, 给出一系列 keyframe, 之间用 RRT-Connect 连接.
- Domain randomization: 物体位置、机械臂 base 位置都随机.
- Rejection sampling: 失败 rollout 直接扔掉, dataset 全是 successful trajectories.

最终得到 dataset $D = \{(o_t, a_t)\}$, 其中 $o_t$ 是真机能拿到的 observation (RGB + proprioception), $a_t$ 是 expert action. IL 在这上面直接训.

---

## 3. 3D Gaussian Splatting 数学细节 (Section 2.2)

3DGS 是这篇 paper 视觉部分的基石, 公式很值得展开讲.

### 3.1 单个 Gaussian primitive

每个 Gaussian 在 world space 中心 $\mu$, 形状由 covariance matrix $\Sigma$ 决定:

$$
G(x) = \exp\left(-\frac{1}{2} x^\top \Sigma^{-1} x\right)
$$

- $x$: 相对 Gaussian 中心 $\mu$ 的 3D offset (3D 向量, $\mathbb{R}^3$)
- $\Sigma$: 3×3 covariance matrix, 必须是 positive semi-definite
- $\Sigma^{-1}$: $\Sigma$ 的逆, 决定 Gaussian 的 "等值面" 形状 (椭球)

直接优化 $\Sigma$ 难以保证 PSD, 所以 3DGS 把 $\Sigma$ 拆成 rotation + scaling:

$$
\Sigma = R\,S\,S^\top R^\top
$$

- $R$: 3×3 rotation matrix, 由四元数 $q = (q_w, q_x, q_y, q_z)$ 参数化, 表示 Gaussian 在 3D 的朝向
- $S$: 3×3 scaling matrix, 通常用 diagonal $(s_x, s_y, s_z)$, 表示沿 Gaussian 局部坐标系三轴的尺度
- $S\,S^\top$: 3×3 diagonal matrix, 元素是 $s_x^2, s_y^2, s_z^2$
- 整体 $R\,S\,S^\top R^\top$: 这是一个典型的 $R\,D\,R^\top$ 形式, 保证 $\Sigma$ 是 PSD

所以每个 Gaussian 优化 11 个参数: 3 (中心 $\mu$) + 4 (四元数 $q$) + 3 (scale $s$) + 1 (opacity $\alpha$). 颜色用 spherical harmonics (通常 degree 3, 48 dims for RGB), paper 里简化记作 $c_i$.

### 3.2 Alpha blending 渲染

给定一个 pixel, 把所有覆盖它的 Gaussian 按 depth 排序 (z-buffer), 然后 front-to-back 累积:

$$
C = \sum_{i=1}^{N} \left[\prod_{j<i} (1 - \alpha_j)\right] \alpha_i\,c_i
$$

- $N$: 该 pixel 上 depth-sorted 的 Gaussian 数量
- $i, j$: index, 越小越靠前 (近 camera)
- $\alpha_i$: 第 i 个 Gaussian 在该 pixel 上的 2D opacity, 是 3D Gaussian 在 image plane 投影后再乘以原始 opacity 的结果
- $c_i$: 第 i 个 Gaussian 的 color (or SH-evaluated color)
- $\alpha_i\,c_i$: 第 i 个 Gaussian 对 pixel 颜色的直接贡献
- $\prod_{j<i}(1-\alpha_j)$: 前 i-1 个 Gaussian 的累积 transmittance, 表示光线穿过前面所有 Gaussian 后还剩多少没被吸收
- $C$: 最终渲染的 RGB 颜色

这套公式的好处: (1) 可微, 直接 backprop 到 $\mu, q, s, \alpha, c$; (2) explicit 表示, 不像 NeRF 要 query MLP, 渲染快. RE³SIM 实测 480p 双 camera 渲染 12.93 ms, 加 physics 总共 41.46 ms/step, 约 24 FPS, 刚好够 real-time simulation.

3DGS 原始 paper: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

---

## 4. 实验: 把每个数字读细一点

### 4.1 Reconstruction quality (Table 1)

| Method | PSNR | SSIM |
|---|---|---|
| Polycam | 11.52 ± 1.40 | 0.34 ± 0.04 |
| OpenMVS | 13.40 ± 0.96 | 0.27 ± 0.03 |
| 3DGS | 13.29 ± 1.11 | 0.37 ± 0.04 |

直觉解读:
- PSNR 整体都很低 (~13 dB). 普通 NeRF 在标准 benchmark 能到 30+ dB. 这里低主要因为 sim-real 之间还有 pixel-level misalignment (object 手动 align, background 也有微小偏差), 不是单纯渲染质量问题.
- 3DGS 的 PSNR 跟 OpenMVS 差不多 (13.29 vs 13.40), 但 SSIM 明显高 (0.37 vs 0.27). SSIM 衡量结构相似度, 对 patch 内方差敏感. OpenMVS 的 textured mesh 有 cracks (Table 1 + Appendix B 解释), crack 区域和 background 颜色接近导致 PSNR 不敏感, 但 crack 内方差大, SSIM 直接掉. 这就是为什么作者选 3DGS.
- Polycam 在两个指标上都最差, 但在 RialTo (Torne et al. 2024) 里被用, 所以作者把它作为 baseline.

### 4.2 Zero-shot sim-to-real (Table 2)

| Task | RialTo+IL | AnyGrasp+Prim. | Real+IL (50 eps) | RE³SIM+IL (100 eps) |
|---|---|---|---|---|
| Pick & drop bottle | 0.4 | 0.9 | 0.8 | 0.75 |
| Stack cubes | 0 | — | 0.15 | 0.25 |
| Place vegetable | 0.45 | — | 0.6 | 0.75 |

几个值得注意的 observation:

1. **RE³SIM 在 2/3 任务上超过 real-data trained policy**. 这是 anti-intuitive 的, 因为我们一直相信 "real data is king". 作者解释 (Appendix C) 主要因为 rule-based motion planner 在 sim 里走最短路径, trajectory 干净短; 而 human teleop 用 spacemouse, 有 pauses, jitter, trajectory 长, 反而让 policy 学不到 clean motor priors.
2. **AnyGrasp+Prim 在 pick bottle 上 0.9, 远超 RE³SIM 的 0.75**. 但 AnyGrasp 是专门的 grasp pose predictor, 配合 hand-coded primitive (move-to, close-gripper, lift), 不输出整条 trajectory. 对 stack / place 这种 long-horizon 任务根本无法用, 所以作者只在 pick 任务上做对比.
3. **Stack cubes 全员低分** (Real+IL 也只有 0.15). 因为 stacking 对放置精度要求高, 任何 ~5mm 误差都会让第二个 cube 掉下来. RE³SIM 用 mesh 重建 cube, collision shape 比 CAD 略有偏差, 但 sim-real 还有 gap, 所以叠加失败率高.
4. **RialTo 在 stack cubes 上 0**. RialTo 用 Polycam 重建, mesh 质量差, 物体在桌面上放不稳, 直接 stacked cube 永远倒.

### 4.3 Sim-Real Consistency (Fig. 1d)

Pearson correlation coefficient = **0.924** 在 pick bottle 任务上, 跨 0 到 120k training step, 3 seeds. 这是 paper 最 strong 的一个 claim, 因为它意味着:
- 可以用 sim 上的 success rate 来 predict 真机 success rate
- Sim 是 real 的 reliable proxy
- 不需要每次迭代都跑真机评估

直觉: 这只有当 sim-real gap 足够小才成立. 如果 gap 大, sim 100% 真 机 0%, correlation 是负的. 0.924 说明 sim 和 real 的 difficulty landscape 几乎一致.

### 4.4 Time cost breakdown (Table 5)

| Process | Time (ms) |
|---|---|
| Physics Simulating | 26.64 |
| Gaussian Splatting Rendering | 12.93 |
| Motion Planning | 0.36 |
| Others | 1.53 |
| **Total** | **41.46** |

直觉: physics 占了 64% 时间, 因为 reconstructed mesh 顶点数多, collision detection 慢. 3DGS 渲染占 31%, 已经很快了 (传统 NeRF 一张图要几秒). motion planning 只占 0.36 ms 因为 RRT-Connect 是非常高效的 bidirectional sampling, 加上 keyframe 是预先计算的.

总 41.46 ms/step ≈ 24 FPS, 双 camera 480p. 这够 privileged policy 跑, 但要做到 real-time inference on robot 还需要再优化 (e.g. surface mesh decimation for collision).

### 4.5 Reconstruction human effort (Table 3)

| Input | Human Effort (s) |
|---|---|
| Video | 51.5 |
| Images | 84.5 |
| ARCode (object) | 60.5 |

Video 最快但有 motion blur, images 慢但质量好. ARCode 是 foreground object 重建, 单个 object 约 1 分钟. 一个 scene 总重建时间 = background + N×object, 大约 3 分钟以内可以搞定一个新 scene, 这是 paper 卖点之一.

### 4.6 Data collection cost (Table 4)

| Task | Time for 100 eps (min) |
|---|---|
| Pick & drop bottle | 12.35 |
| Place vegetable | 13.78 |
| Stack blocks | 6.45 |

用 8 张 RTX 4090, 每张跑 2 个 Isaac Sim process (即 16 个 parallel simulator). 100 条 trajectory 12 分钟左右, 平均 ~7 秒/episode. 对比 ALOHA 2 类 teleop 系统, 熟练 operator 大约 30 秒/episode, 100 条要 50 分钟, 还要 operator 在场. RE³SIM 完全 offline.

### 4.7 Large-scale sim-to-real (Fig. 4, "clear table" task)

四个 setup:
- **Seen**: 4 个训练时见过的物体 (bottle, cucumber, corn, eggplant)
- **Unseen**: 4 个未见过的物体 (green pepper, banana, red bowl, momordica charantia)
- **Cluttered**: 全部 8 个混在一起
- **Darkness**: 4 个 seen, 但环境亮度明显低于训练

每个 setup 测 10 次, 失败允许 2 次自动 grasp retry.

直觉 reading: 
- Trained on 5 个物体, generalize 到 unseen, 说明 policy 学到的是 "pick graspable object on this table" 这一类, 不是 overfit 到具体物体.
- Darkness 鲁棒性来自 training 时的 ColorJitter augmentation (见 Table 7).
- 作者 hypothesis: 因为 scene 固定, robot 用 background color 分布区分 "object vs table", 所以 object shape 变化没关系.

### 4.8 Data scaling (Fig. 5)

Success rate 随 dataset size 单调上升, 在 ~200 episodes 后开始 plateau. 100 episodes 已经够 zero-shot 58%. 这跟 LLM scaling law 不一样, 因为 IL 不像 next-token prediction 那么吃 data, 更吃 trajectory quality 和 diversity.

### 4.9 Image augmentation (Table 7)

| Augmentation | Sim SR | Real SR |
|---|---|---|
| None | 0.77 | 0.25 |
| + Gaussian Blur | 0.45 | 0.4 |
| + Defocus | 0.97 | 0.6 |
| + ColorJitter | 0.37 | 0.6 |
| + Gaussian Noise | 0.94 | **0.8** |

直觉:
- 3DGS 渲染的 image 太 "干净" 了, 没有 camera noise, 没有 motion blur. 真机 RealSense D435i 的 RGB sensor 有 readout noise, 加上 robot motion 时帧间 blur.
- Gaussian Noise 最 work (Real SR 0.25 → 0.8), 因为 sensor noise 是真机和 sim 视觉差的主要 source.
- ColorJitter sim 上掉到 0.37 (过强), 但 real 上还是涨到 0.6, 说明 color robustness 有用.
- 单 Gaussian Blur sim 掉到 0.45 但 real 涨到 0.4, 因为 blur 模拟 motion, 但 sim 里其实没 motion blur (motion planner 平稳), 加 blur 反而让 policy 学不到精细信息.

Augmentation 用 Albumentations library: https://github.com/albumentations-team/albumentations

### 4.10 JPEG compression (Fig. 7)

Training 用 JPEG quality=40 压缩存储, deployment 用未压缩. quality 40 几乎无 perceptual loss, 但能省大量存储. 这是 data scale 时的实用 trick.

---

## 5. Policy 架构 (Section 4.1 + Appendix B)

- Backbone: **ACT** (Action Chunking with Transformers), Zhao et al. 2023
- Visual encoder: **DINOv2-small** 替换 ACT 原始的 ResNet
- Input: 双 camera RGB (wrist + third-person) + proprioception
- Output: future k-step action sequence (action chunking)
- 评估时用 **temporal ensemble** (ACT paper 里的 trick, 多个 step 的 prediction 平均, 平滑 rollout)
- VAE 结构: CVAE, KL weight = 10
- Hidden dim 512, dim feedforward 3200, 100 epochs, batch size 8

为什么 DINOv2 比 ResNet work better: DINOv2 是 self-supervised ViT, 在 142M 图像上 pretrain, 学到的 feature 对 photometric invariance 强; ResNet 是 ImageNet supervised, feature 偏向 object classification, 对 manipulation 的 spatial-aware representation 不够强.

ACT paper: https://tonyzhaozh.github.io/aloha/
DINOv2 paper: https://arxiv.org/abs/2304.07193

---

## 6. 跟 related work 的精确对比

| System | Geometry | Vision | Sim-to-Real Validation |
|---|---|---|---|
| RialTo (Torne et al. 2024) | Polycam mesh | Polycam texture | Yes, 但是 SR 低 |
| SplatSim (Qureshi et al. 2024) | Pre-obtained CAD | 3DGS re-render | Yes, 但是需要预先有 CAD |
| RoboGSim (Li et al. 2024a) | — | 3DGS, novel pose synth | 有限 sim-to-real |
| RoboStudio (Lou et al. 2024) | URDF | 3DGS | Robot reconstruction, 非 scene |
| URDFormer (Chen et al. 2024) | Image → URDF | — | Yes, 但 articulated env |
| Evo-NeRF / LERF-TOGO | NeRF / LERF | NeRF | Grasping only |
| GaussianGrasper / SplatMover / GraspSplats | 3DGS | 3DGS | Manipulation-specific |
| **RE³SIM** | MVS mesh + ARCode mesh | 3DGS + ray-traced mesh | Yes, 多任务 zero-shot 58% |

RE³SIM 的特殊之处在于: (1) 不需要预先有 object CAD; (2) geometry 和 vision 分开重建, 各取所长; (3) 真机做了 systematic sim-to-real validation.

SplatSim: https://arxiv.org/abs/2409.10161
RoboGSim: https://arxiv.org/abs/2411.11839
URDFormer: https://arxiv.org/abs/2405.11656

---

## 7. Co-training vs Fine-tune (Appendix D)

- Sim-only model: 速度快, motion planner 走最短路径, 学到 "direct, fast motion" prior
- Real-only model: 速度慢, human teleop 有 pauses
- Co-training (sim + real mixed): 速度居中
- Pretrain on sim → finetune on real: 接近 real-only, 但收敛快

直觉: sim data 提供 "motor prior" (怎么从 A 到 B), real data 提供 "perception calibration" (怎么从 visual 看到目标). 这两件事是 orthogonal 的, 所以 co-train 不会 collapse, finetune 也 work.

---

## 8. Limitations (作者自己提的)

1. 只支持 rigid object. Articulated, deformable, liquid 都不行. 3DGS 本质上是一组静态 Gaussians, 做 deformation 需要额外 As-Rigid-As-Possible (ARAP) 或 neural skinning.
2. Physics 参数 (mass, friction) 用 default, 没有 system identification. 这是 gap 的另一个 source, 但作者实测对 manipulation 影响不大.
3. Rule-based privileged policy 在 task 复杂时不好设计. "Clear table" 还能用 "pick nearest", 但更复杂的 task (e.g. open drawer, pour water) 几乎写不了 rule-based policy.
4. Reconstruction 对 transparent / reflective 物体不好, 因为 MVS / 3DGS 都依赖 Lambertian 假设.

---

## 9. 一些跨 paper 的联想

### 9.1 跟 MimicGen / RoboCasa 的关系
MimicGen (Mandlekar et al. 2023) 和 RoboCasa (Nasiriany et al. 2024) 也是自动生成 sim demo 的系统, 但他们走的是 "human demo → transform → new scene" 路线, 需要预先有少量 human teleop. RE³SIM 完全不需要 human demo, 用 rule-based policy. 代价是 task 复杂度受限. 未来两者结合 (rule-based for simple subtask + MimicGen for composition) 可能是个方向.
- MimicGen: https://mimicgen.github.io/
- RoboCasa: https://robocasa.ai/

### 9.2 跟 π0 / RT-2 / OpenVLA 的关系
这些是 VLA 大模型, 用大规模 internet pretrain + 小规模 robot data finetune. RE³SIM 生成的是 "small but high-fidelity" data, 不在 scale 上竞争. 但 RE³SIM 这套 pipeline 可以作为 VLA 模型的 sim-side data engine, 给 VLA 提供 "in-domain high-fidelity demos" 来 fine-tune.
- π0: https://arxiv.org/abs/2410.24164
- RT-2: https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246

### 9.3 跟 DROID / Open X-Embodiment 的关系
DROID / OpenX 是真机数据集. RE³SIM 的 sim 数据如果质量足够高, 可以作为 DROID 的 augmentation 或 pretraining source. 重要的是 RE³SIM 的 visual gap 小, 这样 sim data 跟 real data 之间的 distribution shift 小, 不会 poison VLA training.
- DROID: https://arxiv.org/abs/2403.12945
- Open X-Embodiment: https://arxiv.org/abs/2310.08864

### 9.4 跟 Genesis / SAPIEN / Isaac Lab 的关系
新一代 simulator (Genesis, Isaac Lab) 都在强化 physics fidelity + rendering fidelity. RE³SIM 用的是 Isaac Sim + PhysX + 3DGS 的 hybrid. 未来 Genesis 如果能 native 支持 3DGS 渲染, RE³SIM 这种 hybrid 渲染就不需要了.
- Genesis: https://genesis-embodied-ai.github.io/
- Isaac Lab: https://isaac-sim.github.io/IsaacLab/

### 9.5 跟 GraspSplats / GaussianGrasper 的关系
这一类方法直接在 3DGS 上做 grasp prediction, 把 3DGS 当 scene representation. RE³SIM 把 3DGS 只当渲染工具, grasp 决策完全在 IL policy 内部. 后者 task-agnostic, 前者 grasp-specific. 两者可能互补: 3DGS scene representation + IL policy.
- GraspSplats: https://arxiv.org/abs/2409.02084
- GaussianGrasper: https://arxiv.org/abs/2403.09637

### 9.6 跟 SuGaR / GSDF / 2DGS 的关系
SuGaR 把 3DGS align 到 mesh, GSDF 把 3DGS 和 SDF 结合. 这些 method 都在解决 "3DGS 不可直接用于 collision" 的问题. RE³SIM 直接绕过这个问题: object 用 MVS mesh 做 collision, 3DGS 只渲染 background. 这是 pragmatic 但 limited 的方案, 因为 background 不动. 如果未来要让 robot 推 background 物体 (e.g. 推开椅子), 就需要 GSDF 这类方法.
- SuGaR: https://anttwo.github.io/sugar/
- GSDF: https://arxiv.org/abs/2403.16964
- 2DGS: https://surh-nwpu.github.io/2d-gaussian-splatting/

### 9.7 跟 Evo-NeRF 的关系
Evo-NeRF (Kerr et al. 2023) 对 transparent object 做 sequential grasping, 用 NeRF 重建 + 更新. RE³SIM 对 transparent object 不行, 因为 MVS 在 transparent surface 上 depth 估计完全错. Evo-NeRF 这种 "专门为 transparent object 设计的 NeRF" 是必要的补充.
- Evo-NeRF: https://arxiv.org/abs/2210.11989

### 9.8 跟 Future Predictive Model 的关系
Tian et al. 2024 (PIDM, predictive inverse dynamics) 是 Karpathy 你自己关注的方向. RE³SIM 的 sim data 可以用来训 PIDM, 因为 PIDM 需要 "next-state prediction", sim 里 next-state 是 ground-truth, real 里要靠 vision 估, sim 训 PIDM 可能更 sample-efficient.
- PIDM: https://arxiv.org/abs/2412.15109

### 9.9 跟 V-JEPA / VideoMAE 的关系
DINOv2 是 image-level self-supervised, 但 manipulation 是 video task. V-JEPA / VideoMAE 这类 video self-supervised encoder 在 manipulation 上可能更强. 未来 RE³SIM + V-JEPA encoder 是一个 obvious next step.
- V-JEPA: https://ai.meta.com/research/publications/v-jepa/
- VideoMAE: https://arxiv.org/abs/2203.12602

---

## 10. 一个 end-to-end intuition 总结

把 RE³SIM 想成给 robot 造一个 "photorealistic mirror world":

1. **Reconstruction**: 给真实 scene 照一个 "3D 全息相" — mesh + 3DGS
2. **Simulation**: 在 mirror world 里复现 physics, 让 robot 在里面 "排练"
3. **Privileged supervision**: 让一个 oracle (rule-based + RRT-Connect) 在 mirror world 里给 robot 做 demo
4. **IL training**: robot 看 oracle 的 demo, 学 "what to do" (perception + action mapping)
5. **Real-world deploy**: robot 在真实世界执行, 因为 mirror world 视觉/几何接近真实, policy 不需要 adaptation

这套工作的真正意义不在 algorithm, 而在 **工程化 demonstration**: 用一组 off-the-shelf 组件 (COLMAP, OpenMVS, ARCode, 3DGS, Isaac Sim, PhysX, RRT-Connect, DINOv2, ACT), 串成一个端到端 pipeline, 并用 zero-shot 58% average success rate 证明 sim-to-real gap 是可被工程化的, 而不是 fundamental. 这个 framing 对整个 robotics sim-to-real 社区是个有用的信号.

RE³SIM 的 limitation 也是这个领域的 limitation: rule-based privileged policy 很难 scale, deformable / articulated object 没法 reconstruct, transparent / reflective 物体重建失败. 这些都是接下来 1-2 年内 community 可能突破的方向.

---

如果你对其中任何一个 sub-thread (3DGS 的可微 rendering pipeline, ICP point-to-plane 推导, RRT-Connect 双向搜索, ACT CVAE 结构, DINOv2 feature 在 manipulation 上的 inductive bias) 想深入聊, 我可以再展开.
