---
source_pdf: Structured 4D Latent Predictive Model for Robot Planning.pdf
paper_sha256: def8594b9171b20a8c8d44aa9a76771abb84ba4382d9413663b9ecd02dac9ac0
processed_at: '2026-08-12T11:22:28-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 Paper

Karpathy, 我换个更接地气的角度重新讲一遍。之前那份偏 paper 解读, 这次偏"如果我自己来设计这个系统, 我会怎么想"。

Project: https://structured-4d-model.github.io/

---

## 1. 一句话总结这篇 paper 在干啥

想象你教 robot 干活, 比如"把红方块放到蓝方块上"。两种主流思路:

**思路 A (Diffusion Policy, OpenVLA, π0)**: 直接拿摄像头画面, 神经网络吐出关节角度。优点是快, 缺点是画面一变 (灯暗了、相机歪了、背景换了) 就傻眼, 因为它学的是"这个特定画面 → 这个动作"的 mapping, 没学到物理本质。

**思路 B (UniPi, TesserAct)**: 先让模型"想象"未来会怎么样——"我会先伸手、抓住方块、抬起来、放上去", 想象成一段视频, 再从视频反推每一步该动哪个关节。好处是可解释、可 long-horizon、遇到新场景能重新 plan。问题是, 想象出来的视频是 2D 的, 多个相机视角对不上, cube 在一个视角里到了桌上, 另一个视角里还悬空, 物理一致性崩了。

**这篇 paper 的做法**: 与其在 2D 视频里想象 future, 不如直接在 3D 结构里想象。模型预测的 future 是一个完整的 3D scene 演化 (4D = 3D + time), 想成什么视角就 decode 成什么视角, 想转成 point cloud 就转 point cloud, 天然 multi-view consistent。然后再用 inverse dynamics 把"3D future"翻译成关节动作。

就这么个事, 但实现细节很讲究。

---

## 2. 为什么 2D video planner 物理不一致

先讲清楚 2D video planner 的问题到底出在哪。

UniPi 的做法: 给定起始画面 + 文字指令 "pick up the red cube", 用 video diffusion 生成一段未来视频。多相机怎么办? naive 的做法是对每个相机独立生成一段, 再 fuse。问题来了: 第 1 个相机生成的视频里 cube 在 t=5 时到了桌上, 第 2 个相机生成的视频里 cube 在 t=5 时还在空中。两个视角描述同一时刻的同一 scene, 却互相矛盾。

Table 1a 里有个指标叫 cPSNR (cross-view PSNR), 专门测多视角一致性。Wan-2.1 的 cPSNR 只有 16.86 dB, 我们的方法 27.42 dB, 差了 10 dB, 这是一个数量级的提升。Chamfer Distance (两个视角分别重建 3D 点云再对齐的误差) Wan-2.1 是 43.09, 我们是 5.95, 差 7 倍。

更直观的例子在 Figure 4: 训练时所有模型都只见过固定的全局相机视角, 测试时换成 novel local view。Baseline (Wan-2.1, TesserAct, OpenSora) 生成的画面里 cube 直接"穿透"了 gripper, 因为 2D 模型从来没学过这个视角下 cube 和 gripper 应该是什么遮挡关系。我们的方法生成的 3D 结构从一开始就是 view-agnostic 的, 任意视角都能正确 decode 出遮挡关系。

**核心 insight**: 2D video planner 的 multi-view inconsistency 并不是一个可以通过加更多训练数据或多视角 fusion module 解决的"工程问题", 而是 representation 层面的根本缺陷。你在 pixel space 建模, 就注定要面对"多视角描述同一 3D scene"这个 underdetermined 问题。

---

## 3. Structured 3D Latent: 为什么选这个 representation

这部分是 paper 最核心的设计决策。

### 3.1 候选 representation 的权衡

要做 4D 动态建模, 先得选一个 3D representation。候选有:

| Representation | 几何准确 | 视觉质量 | 动态建模难度 | 计算成本 |
|---|---|---|---|---|
| Mesh | 好 | 差 (没纹理) | 中 | 低 |
| Point cloud | 好 | 差 | 中 (GNN) | 低 |
| NeRF | 好 | 好 | 极难 (per-scene 优化) | 极高 |
| 3DGS | 好 | 好 | 难 (连续参数不好 dynamic) | 高 |
| Dense voxel | 好 | 好 | 难 (64³ = 262144 个 token) | 高 |
| **Sparse voxel latent** | 好 | 好 (decode 后) | 中 (8000 个 token) | 中 |

NeRF / 3DGS 视觉质量好但动态建模难, 因为它们是 per-scene 优化的连续表示, 你很难用 transformer 直接 rollout 出下一帧的 NeRF 参数。Dense voxel 可以但 token 数太多, transformer self-attention 是 O(N²), 64³ 直接爆显存。

Sparse voxel latent 是 TRELLIS (Xiang et al., 2025) 提出的, 这篇 paper 直接借用。核心 idea: 在 64×64×64 的离散 grid 里, 只保留有内容的 voxel (大约 8000 个), 每个 voxel 存一个 8 维 feature 向量。

### 3.2 形式化定义

$$z_t = \{(p_i, f_i)\}_{i=1}^{L}$$

- $z_t$ — 时刻 t 的整个 3D scene latent
- $p_i \in \{0, 1, ..., 63\}^3$ — 第 i 个 active voxel 在 64³ grid 中的整数坐标
- $f_i \in \mathbb{R}^8$ — 这个 voxel 的 feature 向量, 编码局部几何和颜色
- $L \approx 8000$ — active voxel 总数

8000 个 token × 8 维 feature = 64000 个数, 比 dense 64³ (262144 个 voxel, 每个 feature 8 维 = 2M 个数) 压缩 32 倍, 比 2D 视频 (512×512×3 = 786432 个 pixel) 也压缩 12 倍。这个 compactness 让 flow matching 在 latent 上训练变得可行。

### 3.3 关键 inductive bias: 位置和外观解耦

$p_i$ 给出离散 3D 位置 (structural prior), $f_i$ 给出 continuous 外观 (appearance)。这种解耦让 dynamic model 只需要学两件事:
1. 哪些 voxel 在下一帧该 active (Single Dynamics Model, SD)
2. 这些 active voxel 的 feature 长什么样 (Latent Generator, LG)

如果用纯 point cloud (无 voxel anchor), transformer 需要从零学 3D 空间关系, 没有 structural prior。如果用 dense voxel, token 数太多, 训练不稳定。Sparse voxel 是"先验 + 自由度"的 sweet spot。

### 3.4 Encoding: 多视角 RGB-D → 3D latent

具体流程 (Figure 2 左侧):
1. 多个 RGB-D 相机同时拍, 拿到 $\{o_t^{(i)}\}$ (彩色图 + 深度图) 和相机内外参
2. 每张深度图用相机参数 unproject 成 3D point cloud
3. 多视角 point cloud 合并, voxelized 成 64³ sparse grid, 确定 $\{p_i\}$
4. 每张 RGB 图用 DINOv2 (Meta 的 self-supervised vision encoder) 抽 patch embedding
5. 这些 embedding 通过相机参数反投影到 voxel grid, 同一 voxel 收到多个视角的 feature 就取平均
6. Latent encoder $\mathcal{E}$ (TRELLIS 预训练) 把 averaged feature 压成最终的 $f_i \in \mathbb{R}^8$

这步相当于"多视角信息 fuse 到一个 unified 3D representation", 从此 multi-view consistency 就被 representation 结构性地保证了, 不需要训练时 explicit 约束。

### 3.5 Decoding: 3D latent → 可渲染的 3D Gaussians

Latent decoder $\mathcal{D}$ 把每个 $f_i$ 映射到 K 个 3D Gaussians, 每个 Gaussian 有:

$$\{(\sigma_i^k, c_i^k, s_i^k, \alpha_i^k, r_i^k)\}_{k=1}^K$$

- $\sigma_i^k \in \mathbb{R}^3$ — Gaussian 中心相对 voxel 中心的 offset
- $c_i^k \in \mathbb{R}^3$ — RGB 颜色
- $s_i^k \in \mathbb{R}^3$ — 各轴 scale
- $\alpha_i^k \in \mathbb{R}$ — opacity
- $r_i^k \in \mathbb{R}^4$ — rotation (quaternion)

约束:
$$x_i^k = p_i + \tanh(\sigma_i^k)$$

$\tanh$ 把 offset 限制在 $(-1, 1)$, 让 Gaussian 中心始终在 voxel 附近, 不会飘到 scene 外面去。这个约束很关键——它把"生成 3DGS"这个连续生成问题锚定在离散 voxel 上, flow matching 不用学大范围 spatial translation, 只学小范围 refinement。

Decode 后的 3D Gaussians 可以:
- Render 成任意视角的 RGB 图 (用标准 3DGS α-blending)
- 取 Gaussian centers 作为 point cloud (给 inverse dynamics 用)

所以一个 latent $z_t$ 既可以是 image, 也可以是 point cloud, 也可以是 3DGS, 适应性很强。

### 3.6 为什么这个 representation 能 zero-shot generalize 视角

因为 $z_t$ 是 view-agnostic 的。训练时模型只见过固定 40 个 spherical 视角, 但 latent 学的是"3D scene 的 intrinsic structure", 不是"某视角下的纹理"。测试时给个 novel view, encoder 还是能把这视角的 image 投到同一个 latent space (camera 参数变了, 但 3D 结构不变), decoder 也能从 latent render 这个新视角。

Table 4 显示, 视角旋转 10° 时, DP 从 56% 掉到 25%, DP3 从 47% 掉到 45%, 我们的方法从 84% 掉到 83%, 几乎无损失。这不是"训练时加了 view augmentation"这种数据层面的 trick, 是 representation 层面的 inductive bias 直接吃掉了视角变化。

---

## 4. 4D Predictive Model: 怎么在 latent 上学 dynamic

### 4.1 生成式目标

$$g(z_{t+1}, ..., z_{t+T} | z_t, c)$$

给定当前 latent $z_t$ 和文字指令 $c$, 生成未来 T 步的 latent sequence。自回归 rollout:

$$\hat{z}_{t+1} = LG(SD(z_t, c), z_t, c)$$

### 4.2 为什么要拆成 SD 和 LG 两步

直接用一个模型生成完整 $z_{t+1} = \{(p_i, f_i)\}_{i=1}^L$ 听起来更简洁, 但 paper 拆成两阶段:

**Single Dynamics Model (SD)**: 预测 $\{p_i\}_{t+1}$, 即下一时刻哪些 voxel 该 active
- 运行在 coarse 16×16×16 grid (比 LG 的 64³ 更粗, 计算量小)
- 学的是"结构演化"——cube 从桌上被搬到另一个位置, voxel 集合该怎么变
- Condition: text (CLIP embedding) + $z_t$ (3D conv 压到 16³)

**Latent Generator (LG)**: 给定 SD 预测的 $\{p_i\}_{t+1}$, 填充 $\{f_i\}_{t+1}$
- 运行在 64×64×64 grid
- 学的是"外观演化"——voxel 位置确定了, 颜色、纹理该怎么变
- Condition: text + $z_t$ + SD 输出的 positions

两阶段分解的 intuition: 位置是骨架, 外观是皮肤。位置错了 (cube 跑到错地方), 再精细的 feature 也救不回来。先解决 skeleton, 再补 skin, 是 cascaded generation 的通用哲学 (DALL-E 3 的 hierarchical、Stable Diffusion 的 latent + decoder 都是类似思路)。

另一个深层原因: SD 学的是 "set prediction" (哪些 voxel active, 是离散选择), LG 学的是 "feature regression" (continuous)。两个 loss 的 gradient scale 不同, 联合训练需要仔细调权重, 分开训练更稳。

### 4.3 Flow Matching: 为什么用这个而不是 diffusion

两个模型都用 conditional flow matching (Lipman et al., 2023), 训练目标:

$$\mathcal{L}_{FM}(\theta) = \mathbb{E}_{t, \epsilon, x_0} \| v_\theta(x(t), t, c) - (\epsilon - x_0) \|^2$$

变量解释:
- $\theta$ — 模型参数 (SD 用 $\theta_{SD}$, LG 用 $\theta_{LG}$)
- $t \in [0,1]$ — flow 时间步, 0 是 clean data, 1 是 pure noise
- $x_0$ — clean target (SD 是 voxel occupancy pattern, LG 是 feature vectors)
- $\epsilon \sim \mathcal{N}(0, I)$ — Gaussian noise
- $x(t) = (1-t) x_0 + t \epsilon$ — linear interpolation, constant velocity flow
- $v_\theta$ — 神经网络预测的 velocity field

物理直觉: flow matching 学一个 vector field $v$, 从 noise $\epsilon$ 沿 $v$ 积分能到达 clean data $x_0$。Linear interpolation 对应 straight-line path, 是 Optimal Transport 的特例, 比 DDPM 的 forward/reverse Markov chain 更简单, 训练更稳定, sampling 用 Euler 积分就够, 不需要 1000 步去噪。

Flow matching 和 DDPM 在数学上等价 (都是 score-based generative model), 但 flow matching 的 training loss 更简洁, sampling 更快, 最近在 image generation (Stable Diffusion 3)、video generation (Wan 2.1) 都开始替代 DDPM。

### 4.4 为什么生成式而非回归式

如果用 MLP 直接回归 $\hat{z}_{t+1} = MLP(z_t, c)$, 给定 $(z_t, c)$ 只能输出一个 $z_{t+1}$。但 robot planning 的 future 是 multi-modal 的: 同一句 "pick up the red cube", 可以从左边抓, 也可以从右边抓, 可以先张开 gripper 再靠近, 也可以边靠近边张开。用 regression 会输出这些 mode 的 average, 通常是个模糊的、物理上不合理的"平均动作"。

Flow matching / diffusion 通过 sample noise $\epsilon$ 引入 stochasticity, 能 generate diverse futures, 每个 sample 对应一个合理的 trajectory mode。这对 planning 的 exploration 和 inverse dynamics 的 robustness 都重要——如果 future 只有单一 mode, inverse dynamics 学不到"多种 action 都能到达同一 subgoal"的 flexibility。

### 4.5 Condition Augmentation: 模拟 observation 不全

训练时随机 dropout $z_t$ 的 voxel features + 加 Gaussian noise 到 $\{f_i\}$。这模拟"真实场景下某视角被遮挡、深度传感器有噪声"的情况, 让 model 训练时就见过 partial observation, inference 时即使观察不全也能 robust。

这和 classifier 训练时的 dropout 类似, 但作用在 condition (input) 而非 network 内部, 相当于 input-space data augmentation。

### 4.6 Classifier-Free Guidance (CFG)

训练时 10% 概率把 condition 替换成 unconditional token, 让 model 同时学 conditional 和 unconditional 两种分布。Inference 时:

$$v_{cfg} = v_{uncond} + w \cdot (v_{cond} - v_{uncond})$$

$w$ 是 guidance scale, 控制 condition 影响强度。CFG 是 diffusion / flow matching 的标准技巧, 让生成质量 vs 多样性可调, 这里照搬。

---

## 5. Inverse Dynamics: latent future → robot action

### 5.1 为什么需要这个模块

4D predictive model 只告诉你"未来 3D scene 长什么样", 但 robot 需要的是"每个关节该转到什么角度"。中间需要一个翻译层: 给定当前 3D state $z_t$ 和未来 3D subgoal $z_{t+1}$, 输出 H 步的关节位置 action chunk $a_{1:H}$。

$$ID(z_t, z_{t+1}) \to a_{1:H}$$

### 5.2 Learned version: diffusion-based

Paper 主要的实验用这个。把 $z_t, z_{t+1}$ decode 成 point cloud $pc_t, pc_{t+1}$, 加上 robot proprioceptive state, 输入 diffusion policy 预测 action:

$$\mathcal{L}_{ID}(\theta_{ID}) = \mathbb{E}_{t, \epsilon_t, u_0} \| \epsilon_t - \epsilon_{\theta_{ID}}(u_0 + \epsilon_t, pc_t, pc_{t+1}, t) \|^2$$

- $u_0 = a_{1:H}$ — clean action chunk
- $\epsilon_t$ — noise
- $\epsilon_{\theta_{ID}}$ — denoising network

Architecture:
- Point cloud encoder: 4-layer 1D conv, hidden 128 (来自 DP3 的 backbone)
- Action decoder: 1D conditional UNet, channels [256, 512, 1024], kernel 5
- 100 denoising steps inference

1000 demos/task, 20,000 epochs, 8h/task on A100。

### 5.3 Learning-free version: geometric registration

Real-world block-in-basket 实验用这个, 更简单, 不需要 action-labeled 训练数据:

1. Decode $\hat{z}_{t+1} \to \hat{pc}_{t+1}$ (predicted future point cloud)
2. 在 $\hat{pc}_{t+1}$ 里提取 gripper 区域的 point cloud
3. 计算 FPFH (Fast Point Feature Histograms) 局部几何 feature
4. RANSAC 找 gripper 模板和 predicted gripper 之间的 rigid transformation $T \in SE(3)$ 初始对齐
5. ICP (Iterative Closest Point) refine 对齐
6. 得到 target end-effector pose $T_{ee, t+1}$
7. Motion planner (RRT / CHOMP) 规划 trajectory

这个 variant 很巧妙: 它假设 gripper 是 rigid body, 只要能从 predicted point cloud 里 register 出 gripper 的 6-DOF pose, 就能用现成 motion planner 规划动作, 完全跳过"学 action"这个步骤。

优点: 不需要 action-labeled data, 跨 robot / 跨 setup 友好 (换 robot 只需换 gripper 模板)
缺点: 对 contact-rich 任务不适用 (gripper 在 contact 中会 deform, rigid registration 失效), 所以 peg insertion 还是得用 learned version

### 5.4 为什么 input 用 point cloud 而非 latent

Table 5 ablation 显示 point cloud > latent > voxel。深层原因:

- **Latent 是 predictive model 的内部 representation, 直接拿来当 inverse dynamics input 会 entangle 两个模块**, 训练时 predictive model 的 latent 分布变化会 break inverse dynamics
- **Point cloud 是 "physical interface"**, 是 3D scene 的显式几何描述, 跨模块、跨 task 通用
- **Decoupling 让 modular training 更稳**, 两个模块可以独立训练、独立迭代

这是系统工程上的重要 insight, 类似 OS 的 ABI: 模块之间用稳定的物理 representation 通信, 内部用 compact latent 计算, 各自优化互不干扰。

---

## 6. Planning Pipeline: open-loop vs closed-loop

Algorithm 1 的核心循环:

```
Observe multi-view RGB-D
z_0 ← encode
for t = 0..T-1:
    ẑ_{t+1} ← predict next latent (SD then LG)
    pc_t, pĉ_{t+1} ← decode to point cloud
    a_{1:H} ← inverse dynamics (pc_t, pĉ_{t+1})
    Execute a_{1:H}
    if closed-loop:
        Observe new multi-view
        z_{t+1} ← re-encode from real obs  # 修正 drift
    else:
        z_{t+1} ← ẑ_{t+1}  # 信任预测
```

**Open-loop**: 一次性预测整个 future trajectory, 然后逐步执行, 不再感知环境。快, 但 autoregressive rollout 累积误差, 几步就 drift。

**Closed-loop**: 每步执行后重新观察、重新 encode, 修正预测误差。慢 (需要 sensing latency), 但误差不累积。

Table 6b 的数据非常有意思:

| Training views | Open-loop | Closed-loop |
|---|---|---|
| 4 views | 57% | 84% |
| 10 views | 72% | 82% |
| 40 views | 84% | 85% |

**Closed-loop 把 4-view 训练的 model 从 57% 拉到 84%, 几乎追平 40-view open-loop 的 84%**。这说明 closed-loop 是"穷人版多视角训练"——training-time 多视角用来学 representation, inference-time re-observation 用来纠正 rollout error, 两者某种程度上可以互换。

这与 MPC (Model Predictive Control) 的 receding horizon 哲学一致: 与其一次性 plan 整个 trajectory, 不如每步 re-plan, 用新 observation 修正 model 不完美带来的偏差。

---

## 7. 实验数据深度解读

### 7.1 4D generation quality (Table 1)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | CD↓ | cPSNR↑ | Robot IoU |
|---|---|---|---|---|---|---|
| Wan-2.1 | 19.87 | 0.84 | 0.09 | 43.09 | 16.86 | 0.74 |
| TesserAct | 21.63 | 0.86 | 0.07 | 42.79 | 17.91 | 0.83 |
| OpenSora | 19.89 | 0.82 | 0.09 | 44.07 | 16.67 | 0.72 |
| **Ours** | 22.45 | 0.79 | 0.13 | 5.95 | 27.42 | 0.91 |

**反直觉的点**: 我们的 SSIM (0.79) 比 baseline 低, LPIPS (0.13) 比 baseline 高。SSIM/LPIPS 通常越接近 1/0 越好, 但这两个指标偏好"smooth/blurry"输出。Video diffusion 倾向 generate blurry 的 average frame, 在 SSIM/LPIPS 上占便宜。我们的方法 generate 锐利但有 high-frequency error 的输出, perceptual metric 上吃亏, 但物理上更对。

**关键指标**:
- **Chamfer Distance 5.95 vs 43**, 差 7 倍——3D 几何一致性碾压
- **cPSNR 27 vs 17**, 差 10 dB——multi-view consistency 碾压
- **Robot mask IoU 0.91 vs 0.73**——robot 自身几何 predict 得准, downstream inverse dynamics 才有 reliable goal

### 7.2 ManiSkill3 planning (Table 2)

| Method | StackCube | ToolPull | PegInsert | Avg |
|---|---|---|---|---|
| DP | 56% | 87% | 24% | 55.7% |
| DP3 | 47% | 94% | 7% | 49.3% |
| UniPi | 35% | 49% | 4% | 29.3% |
| TesserAct | 31% | 43% | 3% | 24.7% |
| **Ours** | 84% | 84% | 16% | **61.3%** |

**PegInsert 全员拉胯**, 最好的 DP 也只有 24%。这是 contact-rich fine-grained 任务, 需要 millimeter-level 精度, 当前 64³ voxel grid 分辨率不够。这也是 paper Limitation 明确指出的。

**DP3 在 ToolPull 上反超我们 (94% vs 84%)**, 因为 DP3 是 task-specific imitation, 在 narrow domain 上见过更多 expert demonstration, 而 our method 是 generative planner + inverse dynamics 两阶段, pipeline 串联误差累积。但 DP3 的代价是 zero-shot generalization 很差 (Table 4)。

### 7.3 RLBench (Table 3)

| Method | Close Box | Sweep Dustpan | Water Plants | Avg |
|---|---|---|---|---|
| UniPi | 81% | 49% | 35% | 55.0% |
| TesserAct | 88% | 56% | 41% | 61.7% |
| **Ours** | 93% | 69% | 64% | 75.3% |

RLBench 比 ManiSkill3 简单 (没 peg insertion), +13.6 pt over TesserAct, 显示 representation 优势在中等复杂度任务上更明显。

### 7.4 Zero-shot generalization (Table 4) — 最强 selling point

| Perturbation | DP | DP3 | Ours |
|---|---|---|---|
| Light ↓ | 7% | 47% | **78%** |
| Noise + | 5% | 47% | **80%** |
| BG color change | 1% | 47% | **84%** |
| View 5° | 43% | 49% | **85%** |
| View 10° | 25% | 45% | **83%** |

DP 几乎全垮 (1%-43%)——end-to-end policy 对 visual distribution shift 极敏感, 训练分布外直接失效。

DP3 稳定在 47%——3D point cloud policy 有一定 robustness (几何信息抗 lighting/texture 变化), 但 plateau 在 47%, 因为 point cloud encoder 仍依赖训练分布。

我们的方法稳定在 78%-85%——representation-level inductive bias 直接吃掉 perturbation。3D latent 本质 view-invariant, lighting/noise 主要影响 texture, 几何不变。5°/10° view shift 在 latent space 里只是 encoding 时 camera 参数变化, decoder 出来仍是同一 3D structure。

**这不是"训练时加了 augmentation"的数据层面 trick, 是 representation 层面的根本优势。**

### 7.5 与 3D Diffuser Actor 对比 (Table 7)

3D Diffuser Actor 是 SOTA 3D policy, 原设置 90% vs 我们 91%, 持平。但 perturbation 下:

| Perturbation | 3D Diffuser Actor | Ours |
|---|---|---|
| Original | 90% | 91% |
| Noise 0.05 | 85% | 86% |
| Noise 0.08 | **48%** | 90% |
| Noise 0.10 | **7%** | 86% |

3D Diffuser Actor 用 3D representation 但 **不做 predictive rollout**, 只做 obs→action mapping。Observation noise 通过 point cloud encoder 直接漏到 action。

我们的方法有 **predictive rollout 充当 denoising prior**: noisy observation → encode 到 latent (噪声在 latent space 被压低) → rollout 出 clean future latent → decode 到 clean point cloud → inverse dynamics 输出 action。中间的 generative rollout 起到 "filter" 作用, 把 observation noise 过滤掉。

这跟 LLM 里 "let's think step by step" 让 model 内部多做几步推理来稳定输出是类似哲学: 中间多走几步 generative computation, 抖动就被平均掉了。

---

## 8. 我的几个直觉 (build your intuition)

### 8.1 为什么 sparse voxel 比 dense voxel 好

Dense 64³ = 262144 个 token, transformer self-attention O(N²), 显存爆炸。Sparse 8000 token 只占 3%, 计算量降 90%, 而且 active voxel 本身就是 scene 的 structural prior——空白区域根本不需要计算。

更深层: dense voxel 在稀疏区域 (空气) 浪费 capacity, 而 sparse voxel 把所有 capacity 都花在 scene 内容上, 信息密度更高。

### 8.2 为什么 $\tanh$ 约束 Gaussian 位置

$$x_i^k = p_i + \tanh(\sigma_i^k)$$

如果不约束, flow matching 可能学出"Gaussian 中心飘到 scene 外"的 degenerate solution。$\tanh$ 把 offset 限制在 $(-1, 1)$, 让 Gaussian 始终在 voxel 附近, 生成过程被锚定在 grid 上。

这和 DDPM 里 clip 预测、VQ-VAE 里 codebook 约束是同一类技巧: 生成模型的输出空间需要 bounded, 否则训练不稳定。

### 8.3 为什么 SD 用 coarse grid, LG 用 fine grid

SD 学"哪些 voxel active", 是结构判断, 不需要 fine 分辨率, 16³ 足够。LG 学"voxel feature 长啥样", 是外观细节, 64³ 保留更多 high-frequency 信息。

这种 "coarse-to-fine, structure-to-appearance" 的分工, 和图像生成里的 cascaded resolution (256→512→1024)、文本生成里的 "先 plan 大纲再写细节" 是同一哲学: 先解决 low-entropy 的全局结构, 再解决 high-entropy 的局部细节。

### 8.4 Closed-loop 为什么这么有效

Autoregressive rollout 的误差是 multiplicative 累积, 每步误差 $\epsilon$, T 步后总误差 $\sim T \epsilon$。Closed-loop 每步 re-encode 真实 observation, 把误差变成 additive, 每步独立, 总误差 $\sim \sqrt{T} \epsilon$。

Table 6b 显示 closed-loop 在 4-view training 下能把 success rate 从 57% 拉到 84%, 几乎追平 40-view open-loop。这说明 closed-loop 不只是"修正 drift", 它根本改变了 error 累积模式。

### 8.5 这个范式的 scaling potential

如果 4D latent predictive model 训练在足够 diverse 的 robot + scene + task 数据上, 它可以成为 "robotics foundation world model":

- 给定任意 instruction + scene, roll 出 4D future
- 用任意 inverse dynamics (learning-free registration 或 task-specific learned) 解出 action
- World model 跨 robot 迁移 (只学 3D dynamic, 不管 action space)
- Inverse dynamics 跨 task 迁移 (只学 goal→action, 不管 scene understanding)

这种 decoupling 让两个模块各自 scale: world model 用海量 passive video 数据 (YouTube、simulation、不同 robot), inverse dynamics 用少量 task-specific demonstration。这是相比 VLA (vision-language-action 端到端) 的架构优势——VLA 把所有东西塞进一个 model, scaling 时所有维度同时紧张, 这个范式可以分工。

### 8.6 跟 LLM 的类比

LLM 里, "predict next token" 是 self-supervised pretraining objective, 学到 language 的 world model。
Robotics 里, "predict next 3D latent" 也应该是 self-supervised pretraining objective, 学到 physical world 的 dynamic model。

两者都是 generative modeling, 都是 autoregressive, 都可以用 flow matching / diffusion。区别只在 representation: LLM 用 discrete token, robotics 这里用 sparse voxel latent。如果 robotics 也找到像 LLM 一样可以 internet-scale 收集的 pretraining data (e.g. YouTube 视频 → 3D latent via off-the-shelf 重建), 这个范式就能 scale。

---

## 9. Limitations 和 future directions

Paper 自己承认的 (Appendix D):

1. **Calibrated multi-view 依赖**: 真实世界 multi-view setup 难部署, 单目 / 弱 calibration 场景不行。可以结合 single-image-to-3D (LRM, TripoSR) 缓解
2. **Contact-rich 任务弱**: PegInsert 16%, voxel grid 分辨率不够
3. **Tabletop only**: 没测 mobile manipulation, humanoid

我补充几个可能的后续方向:

1. **Single-view → 3D latent**: 用 LRM-style 单图重建 latent, 摆脱 multi-view calibration
2. **Active perception**: closed-loop 时主动选择下一个 view, 减少 observation cost
3. **Latent-space RL**: 把 4D predictive model 当 world model, 在 latent 里做 planning 或 RL, 不需要 decode 到 pixel (类似 Dreamer)
4. **Contact-aware representation**: voxel + SDF 补充 fine geometry, 解决 peg insertion
5. **Object-centric factorization**: latent 里加 object-level decomposition, 支持 "pick up the red cube" compositional reasoning
6. **Diffusion forcing 集成**: 让 rollout 时长更 long-horizon 而不退化
7. **Hierarchical latent**: task → subtask → subgoal 的 semantic hierarchy (类似 Compositional Foundation Models)
8. **Cross-embodiment inverse dynamics library**: 像 RT-X 那样统一不同 robot 的 inverse dynamics, 让 world model 真正 embodiment-agnostic

---

## 10. 一句话总结

这篇 paper 把 robot world model 从 2D pixel 推到 structured 4D latent, 用 sparse voxel + flow matching 学 3D dynamic, 用 inverse dynamics 翻译成 action。最强 result 是 zero-shot generalization 几乎不掉——这是 representation-level inductive bias 的胜利, 也是 generative planner 范式相对 end-to-end policy 的根本优势。

如果打个比方: end-to-end policy 是"看见即反应" (System 1), UniPi 是"想象 future video 再反应" (System 2 但 2D), 这篇 paper 是"想象 4D 结构再反应" (System 2 + 3D grounded)。Robot manipulation 的 fine-grained 特性, 注定需要 3D-grounded 的想象。

---

## Reference Links

- Paper project: https://structured-4d-model.github.io/
- TRELLIS (base representation): https://trellis3d.github.io/
- UniPi: https://universal-policy.github.io/
- TesserAct: https://arxiv.org/abs/2504.20995
- DP3: https://3d-diffusion-policy.github.io/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Flow Matching (Lipman et al.): https://openreview.net/forum?id=PqvMRDCJT9t
- ManiSkill3: https://github.com/haosulab/ManiSkill
- LIBERO: https://libero-project.github.io/
- RLBench: https://sites.google.com/view/rlbench
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- DINOv2: https://dinov2.metademolab.com/
- CLIP: https://openai.com/research/clip
- 3D Diffuser Actor: https://3d-diffuser-actor.github.io/
- π0: https://arxiv.org/abs/2410.24164
- GR00T N1: https://arxiv.org/abs/2503.14734
- GWM (Gaussian World Model): https://arxiv.org/abs/2506.14135
- Dreamer V3: https://arxiv.org/abs/2301.04104
- Diffusion Forcing: https://boyuan.space/diffusion-forcing/
- LRM (Large Reconstruction Model): https://yiconghong.me/lrm/
- OpenVLA: https://openvla.github.io/
- RT-X: https://robotics-transformer-x.github.io/

---

# Structured 4D Latent Predictive Model for Robot Planning 深度解析

Karpathy 你好, 这篇 paper 来自 MIT 的 Yilun Du 组 (Ruojin Cai 等), Kempner Institute 赞助, 与 PickleRobotics、Amazon AGI Labs 合作。它本质上是对 UniPi (Du et al., 2023b)、TesserAct (Zhen et al., 2025) 这一脉 "video-as-planner" 范式的 3D-ization, 把 world model 从 2D pixel space 推到 structured 4D latent space。我把核心 intuition、representation 选择、训练目标、planning pipeline、实验数据全部拆开来讲。

Project page: https://structured-4d-model.github.io/
TRELLIS (基础 representation): https://trellis3d.github.io/
Flow Matching 原始 paper: https://openreview.net/forum?id=PqvMRDCJT9t
UniPi: https://universal-policy.github.io/
TesserAct: https://arxiv.org/abs/2504.20995
DP3: https://3d-diffusion-policy.github.io/

---

## 1. 核心动机:为什么 2D video planner 不够用

当前 robot learning 的两大流派:
- **End-to-end policy** (DP, DP3, OpenVLA, π0, GR00T N1): obs→action 直接映射, narrow task 上很强, distribution shift 下崩盘
- **Generative planner / world model** (UniPi, TesserAct, Sora-style models): 先 generate future, 再用 inverse dynamics 解出 action, 优点是 long-horizon、可解释、可组合

2D video planner 的根本问题:
1. **3D inconsistency**: 多视角各自 rollouts 后 fusion 失败, Table 1a 显示 Wan-2.1 的 cross-view PSNR (cPSNR) 只有 16.86 dB, Chamfer Distance 高达 43.09
2. **Occlusion / viewpoint shift** 下物理不一致, Fig 4 显示 baselines 在 novel view 下 cube 直接 "穿透" gripper
3. **Pixel space 计算昂贵**, long horizon rollout 会累积误差

TesserAct 试图用 RGB+Depth+Normal 的 2.5D 方法缓解, 但本质还是 surface projection, multi-view coherence 仍然丢失。

本文的核心 claim: **在 structured 3D latent space 里建模 4D dynamic, 直接绕过 2D pixel 的 bottleneck**。这并不是 "换个 representation", 而是把 multi-view consistency 从 training objective 里 (很难学的) 移到 representation 设计里 (结构性保证)。

---

## 2. Structured 3D Latent Representation (来自 TRELLIS / SLAT)

### 2.1 形式化定义

Latent state 定义为:

$$z_t = \{(p_i, f_i)\}_{i=1}^{L}$$

变量解释:
- $z_t$ — 时刻 t 的整个 3D scene latent
- $p_i \in \{0, 1, ..., N-1\}^3$ — 第 i 个 active voxel 的 3D 整数坐标 (N=64, 即在 64×64×64 离散 grid 中)
- $f_i \in \mathbb{R}^d$ — 该 voxel 的 feature vector, d=8
- $L \approx 8000$ — active voxel 数量 (sparse)

对比 dense grid: 64³ = 262144 vs 8000×8 = 64000, 大约 4x 压缩, 但更重要的是 **sparse 结构**让 transformer attention 可以只在 active tokens 间计算, 复杂度从 O(N³) 降到 O(L²)。

### 2.2 为什么选 sparse voxel 而非 NeRF / 3DGS / point cloud

| Representation | 几何完整 | Photorealism | 动态建模友好 | 计算成本 |
|---|---|---|---|---|
| Mesh / Point cloud | ✓ | ✗ | ✓ (GNN) | 低 |
| NeRF | ✓ | ✓ | ✗ (per-scene 优化) | 极高 |
| 3DGS | ✓ | ✓ | ✗ (不易动态) | 高 |
| Dense voxel | ✓ | ✓ | ✗ (内存爆炸) | 高 |
| **Sparse voxel latent (SLAT)** | ✓ | ✓ (decode 后) | ✓ (transformer) | 中 |

关键 insight: SLAT 把 **结构信息 (3D coordinate)** 和 **语义/外观 (feature vector)** 解耦, structural bias 通过 $p_i$ 显式注入, 让 transformer 不必从 0 学 3D, 同时 feature 维度低 (d=8) 让 flow matching/diffusion 训练稳定。

### 2.3 Encoding: Multi-view RGB-D → z_t

Pipeline:
1. Multi-view RGB-D images $\{o_t^{(i)}\}$ + camera intrinsics/extrinsics
2. Depth unprojection → 合并成 3D point cloud
3. Voxelize → 得到 active voxels $\{p_i\}_{i=1}^L$
4. 每张 image 用 **DINOv2** encoder 抽 patch embeddings
5. Patch embeddings 通过 camera poses 反投影到 voxel grid, 同 voxel 多 view feature **取平均**
6. Latent encoder $\mathcal{E}$ (pre-trained from TRELLIS) 把 averaged features 压成 $f_i$

这一步对应 Figure 2 左侧 "3D reconstruction"。

### 2.4 Decoding: z_t → 3D Gaussians → Images / Point clouds

Latent decoder $\mathcal{D}$ 把每个 $f_i$ 映射到 K 个 3D Gaussians:

$$\{(\sigma_i^k, c_i^k, s_i^k, \alpha_i^k, r_i^k)\}_{k=1}^K$$

变量含义:
- $\sigma_i^k \in \mathbb{R}^3$ — Gaussian center 的 offset (相对 voxel 中心)
- $c_i^k \in \mathbb{R}^3$ — color (RGB)
- $s_i^k \in \mathbb{R}^3$ — scale (各轴 std)
- $\alpha_i^k \in \mathbb{R}$ — opacity
- $r_i^k \in \mathbb{R}^4$ — rotation quaternion

**关键约束** (防止 Gaussian 飘出 voxel):

$$x_i^k = p_i + \tanh(\sigma_i^k)$$

$\tanh$ 把 offset 限制在 $(-1, 1)$, 保证 Gaussian 中心在该 voxel 单位立方体内。这是个很重要的设计——它把 "生成 3DGS" 这个连续问题, 通过 voxel anchor 离散化, 让 flow matching 不需要学习大范围 spatial translation。

Render 出 image 用标准 3DGS α-blending; 转 point cloud 直接取 Gaussian centers。

### 2.5 为什么能 zero-shot generalize viewpoint

因为 latent $z_t$ 是 **view-agnostic** 的——同一个 latent 可以 decode 到任意 viewpoint。所以训练时即使只见过固定视角, latent space 学到的是 **真正 3D structure**, 不是 "view-conditioned texture"。Table 4 的 view(10°) 实验, DP 从 56% → 25%, 我们的方法从 84% → 83%, 几乎不掉。这是 representation-level inductive bias 的胜利。

---

## 3. 4D Predictive Model: 把 Dynamic 拆成 SD + LG

### 3.1 整体 generator

$$g(z_{t+1}, ..., z_{t+T} | z_t, c)$$

自回归 rollout:
$$\hat{z}_{t+1} = LG(SD(z_t, c), z_t, c)$$

为什么不一次生成整个 sequence? 因为 full 3D latent state 太大 (8000 tokens × 8 dim × 多步), autoregressive + 两阶段分解让每步计算可控, 同时让 SD 和 LG 各自专注。

### 3.2 Single Dynamics Model (SD): 预测 coarse geometry

$$SD(\{p_i\}_{t+1} | z_t, c)$$

- **作用**: 预测下一时刻的 active voxel positions (粗结构)
- **运行空间**: coarse 16×16×16 grid (相对 LG 的 64×64×64)
- **生成方法**: Conditional Flow Matching
- **Backbone**: 24 层 transformer, 16 heads, hidden dim 1024
- **Conditioning**:
  - Text → CLIP embedding
  - $z_t$ → 3D conv 压到 16³ resolution → cross-attention
  - Positional encoding 让 voxel token 与 condition 对齐
- **Condition augmentation**: 训练时随机 dropout voxel features + 加 Gaussian noise 到 $\{f_i\}$, 模拟 partial observation 鲁棒性

为什么先预测 position? **结构是 skeleton, appearance 是 skin**。位置错了 (cube 跑到错地方), 再 fancy 的 feature 也救不回来。这种 coarse-to-fine 的解耦和 Cascaded Diffusion、DALL-E 3 的 hierarchical 生成哲学一致。

### 3.3 Latent Generator (LG): 填充 fine features

$$LG(\{f_i\}_{t+1} | \{p_i\}_{t+1}, z_t, c)$$

- **作用**: 给定 SD 预测的 voxel positions, 填充 feature vectors
- **运行空间**: 64×64×64 grid
- **生成方法**: Conditional Flow Matching (同样 transformer backbone)
- **Conditioning**: 同 SD (text + $z_t$ + 预测的 positions)

为什么 SD 和 LG 分开训练? 论文给的理由是 "modularity, easier to train"。深层原因可能是 **两者 loss landscape 不同**: SD 学 discrete set prediction (类似 DETR 的 set prediction), LG 学 continuous feature regression。混合训练容易 conflict。

### 3.4 Flow Matching objective

两个模型都用 conditional flow matching (Lipman et al., 2023), 是 diffusion / score matching 的等价但更简洁的形式:

$$\mathcal{L}_{FM}(\theta) = \mathbb{E}_{t, \epsilon, x_0} \| v_\theta(x(t), t, c) - (\epsilon - x_0) \|^2$$

变量逐项解释:
- $\theta$ — 模型参数 (SD 用 $\theta_{SD}$, LG 用 $\theta_{LG}$)
- $t \in [0, 1]$ — flow time step, $t=0$ 是 clean data, $t=1$ 是 pure noise
- $x_0$ — clean target (SD 中是 voxel occupancy, LG 中是 feature vectors)
- $\epsilon \sim \mathcal{N}(0, I)$ — Gaussian noise
- $x(t) = (1-t) x_0 + t \epsilon$ — linear interpolation (constant velocity flow, Optimal Transport 特例)
- $v_\theta(x(t), t, c)$ — neural net 预测 velocity field

物理直觉: flow matching 学一个 vector field $v$, 让从 $\epsilon$ (noise) 沿 $v$ 积分能到达 $x_0$。Linear interpolation 对应 straight-line trajectory, 比 DDPM 的 forward/reverse Markov chain 简单, 训练更稳定, sampling 也快 (Euler 积分即可)。

Classifier-free guidance (CFG) 训练时 unconditional dropout p=0.1, inference 时用 CFG scale。EMA decay 0.9999 稳定训练。

### 3.5 Training schedule

- 300,000 steps, lr=1e-4, AdamW, FP16
- Batch size 8/GPU × 4 H100 = 32
- SD: 2 gradient accumulation steps
- LG: 4 gradient accumulation steps
- 训练时间: ~3 天 / 模型 × 4 H100 (80GB)

数据准备: 每条 trajectory $(z_1, ..., z_T, c)$, 均匀采样 T 个 subgoal timesteps。训练时随机采样 $(z_t, z_{t+1}, c)$ 配对。ManiSkill3 每任务 1000 demos, LIBERO-90 每任务 50 demos, 每 trajectory 抽 4-10 intermediate frames, render 40 spherical views。

---

## 4. Inverse Dynamics: latent → action

### 4.1 Learned version (diffusion-based)

$$ID(z_t, z_{t+1}) \to a_{1:H}$$

Input: 解码后的 point clouds $pc_t, pc_{t+1}$ + robot proprioceptive state
Architecture:
- Point cloud encoder: 4-layer 1D conv, hidden 128 (来自 Ze et al., 2024a, DP3 backbone)
- Action decoder: 1D conditional UNet, channels [256, 512, 1024], kernel 5, 8 groups
- Diffusion head

训练目标 (DDPM-style):

$$\mathcal{L}_{ID}(\theta_{ID}) = \mathbb{E}_{t, \epsilon_t, u_0} \| \epsilon_t - \epsilon_{\theta_{ID}}(u_0 + \epsilon_t, pc_t, pc_{t+1}, t) \|^2$$

- $u_0 = a_{1:H}$ — clean action chunk (H 步 joint positions, 绝对位置)
- $\epsilon_t \sim \mathcal{N}(0, I)$ — noise
- $\epsilon_{\theta_{ID}}$ — denoising network
- $pc_t, pc_{t+1}$ — 当前 + subgoal 的 point cloud

100 denoising steps inference, 20,000 epochs, 8h/task on A100。

### 4.2 Learning-free version (geometric registration)

这个 variant 比较巧妙, 对 real-world 实验 (block-in-basket) 用:

1. 解码 $\hat{z}_{t+1} \to \hat{pc}_{t+1}$
2. 提取 predicted gripper region 的 point cloud
3. 计算 **FPFH** (Fast Point Feature Histograms) features 做粗匹配
4. **RANSAC** 找 rigid transform $T \in SE(3)$ 的初始对齐
5. **ICP** (Iterative Closest Point) refine
6. 得到 target end-effector pose $T_{ee, t+1}$
7. Motion planner (e.g. RRT, CHOMP) 生成 trajectory

为什么 learning-free 在某些场景更稳? 因为不需要 action-labeled training data, 只要 point cloud registration 收敛就行, 更适合跨机器人 / 跨 setup 迁移。但对 contact-rich (peg insertion) 不够精细, 因为 gripper geometry 在 contact 中会 deform, rigid registration 失效。

---

## 5. 完整 Planning Pipeline

Algorithm 1 (paper 里) 的解读:

```
Observe {o_0^(i)}  # multi-view RGB-D
z_0 ← E({o_0^(i)})  # encode to 3D latent
for t = 0..T-1:
    ẑ_{t+1} ← LG(SD(z_t, c), z_t, c)  # autoregressive rollout
    pc_t ← D(z_t), pĉ_{t+1} ← D(ẑ_{t+1})  # decode to point cloud
    a_{1:H} ← ID(pc_t, pĉ_{t+1})  # inverse dynamics → action chunk
    Execute a_{1:H}
    if closed loop:
        Observe {o_{t+1}^(i)}  # new obs
        z_{t+1} ← E({o_{t+1}^(i)})  # re-encode (correct drift)
    else:
        z_{t+1} ← ẑ_{t+1}  # trust prediction
```

关键 design choice: **closed-loop vs open-loop**。Closed-loop 每步重新 encode 真实 observation, 修正 predictive rollout 的 drift, 但需要更多 sensing。Open-loop 直接信任预测, 快但会累积误差。Table 6b 显示 closed-loop 在 4-view training 下能从 57% 拉到 84%, 几乎追平 40-view open-loop 的 84%。

---

## 6. 实验数据深度分析

### 6.1 4D generation quality (Table 1)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | CD↓ | depth↓ | cPSNR↑ | cSSIM↑ | cLPIPS↓ |
|---|---|---|---|---|---|---|---|---|
| Wan-2.1 | 19.87 | 0.84 | 0.09 | 43.09 | 25.06 | 16.86 | 0.62 | 0.24 |
| TesserAct | 21.63 | 0.86 | 0.07 | 42.79 | 23.87 | 17.91 | 0.65 | 0.23 |
| OpenSora | 19.89 | 0.82 | 0.09 | 44.07 | 25.82 | 16.67 | 0.60 | 0.25 |
| **Ours** | **22.45** | 0.79 | 0.13 | **5.95** | **9.38** | **27.42** | **0.86** | **0.07** |

关键观察:
- **PSNR 略胜, 但 SSIM / LPIPS 反而略输** — 这很反直觉。原因是 SSIM/LPIPS 偏好 "smooth / blurry" outputs, video models 倾向 generate blurry 平均, 我们的方法 generate 锐利但有 high-frequency error, 在 perceptual metric 上吃亏。
- **Chamfer Distance 暴跌 7x** (5.95 vs 43.09) — 这是 3D consistency 的硬指标, video baseline 的 depth map 在多 view 间完全不 consistent
- **cPSNR (cross-view PSNR) 暴涨 10 dB** — 直接反映 multi-view consistency
- **Robot mask IoU 0.91 vs 0.73** — robot 自身的几何 predict 得准, downstream inverse dynamics 才有 reliable goal

### 6.2 Robot planning on ManiSkill3 (Table 2)

| Method | StackCube | ToolPull | PegInsert | Avg |
|---|---|---|---|---|
| DP | 56% | 87% | 24% | 55.7% |
| DP3 | 47% | 94% | 7% | 49.3% |
| UniPi | 35% | 49% | 4% | 29.3% |
| TesserAct | 31% | 43% | 3% | 24.7% |
| **Ours** | 84% | 84% | 16% | **61.3%** |

观察:
- DP3 在 ToolPull 反而更强 (94% vs 84%) — DP3 专注任务 imitation, narrow domain 优势
- PegInsert 全员拉胯 (7%-24%) — contact-rich, fine-grained, 我们 16% 略胜, 但绝对值低, 这也是 paper Limitation 里讲的
- UniPi/TesserAct 全面塌方 — video planner + image inverse dynamics 串联 error 太大

### 6.3 RLBench (Table 3)

| Method | Close Box | Sweep Dustpan | Water Plants | Avg |
|---|---|---|---|---|
| UniPi | 81% | 49% | 35% | 55.0% |
| TesserAct | 88% | 56% | 41% | 61.7% |
| **Ours** | **93%** | **69%** | **64%** | **75.3%** |

RLBench 比 ManiSkill3 简单 (没有 fine-grained peg insertion), 我们的方法 +13.6 pt over TesserAct, 显示 representation 优势在中等复杂任务上更明显。

### 6.4 Zero-shot generalization (Table 4) — 最惊艳的结果

| Perturbation | DP | DP3 | Ours |
|---|---|---|---|
| Light | 7% | 47% | **78%** |
| Noise | 5% | 47% | **80%** |
| BG color | 1% | 47% | **84%** |
| View 5° | 43% | 49% | **85%** |
| View 10° | 25% | 45% | **83%** |

- DP 几乎全垮 (1%-43%) — end-to-end policy 对 visual distribution shift 极敏感
- DP3 稳定但 plateau 在 47% — 3D point cloud policy 有一定 robustness, 但训练分布固定
- 我们的方法稳定在 78%-85% — **representation-level inductive bias 直接吃掉 perturbation**

这是 paper 最强的 selling point: 不是 "在 seen setting 上更好", 而是 "在 unseen setting 上几乎不掉"。3D latent 的本质就是 view-invariant, lighting / noise 主要影响 texture, 几何不变; 5°/10° view shift 在 latent space 里只是 encoding 时的 camera parameter 变化, decoder 出来仍是同一个 3D structure。

### 6.5 Ablation: Inverse Dynamics input (Table 5)

| Input | OL | CL |
|---|---|---|
| Pointcloud-40 (training) | 84% | 85% |
| Pointcloud-4 | 57% | 84% |
| 3D latent-4 | 66% | 80% |
| Voxel-4 | 57% | 73% |

- Point cloud > voxel > 3D latent (在 4-view 下)
- 为什么 latent 不如 point cloud? 论文说 "computational cost high", 但更深层: latent 是 generative model 自己的内部 representation, 直接拿来当 inverse dynamics 输入会 entangle 两个模块, 而 point cloud 是 "physical interface", 更可迁移
- **Closed-loop 把 4-view 从 57% 拉到 84%**, 几乎追平 40-view — 极其重要的实践结论

### 6.6 Ablation: Training views (Table 6)

Training views 4/10/40, inference 固定 4 views:
- CD: 7.10 → 7.06 → 6.81 (more views 略好)
- Planning OL: 57% / 72% / 84% (强依赖 training views)
- Planning CL: 84% / 82% / 85% (closed-loop 抹平了少 view 的劣势)

**Insight**: closed-loop 是 "穷人版 multi-view training" — training-time 多 view 用来学 representation, inference-time re-observation 用来纠正 rollout error, 两者在某种程度上可互换。

### 6.7 Real-world experiment (Fig 5)

- Task: block-in-basket
- 200 human demos, 4 RGB-D cameras
- Learning-free inverse dynamics (FPFH + RANSAC + ICP)
- 50 episodes, success rate 高于 baseline (具体数字在 Fig 5g, paper 文字没明确)

真实世界的关键挑战:
1. **Calibration 噪声** — multi-view setup 要求 camera extrinsic 准, 真实世界难免有误差
2. **Depth noise** — RGB-D sensor 的 depth 不准, unprojection 后 point cloud 有 jitter
3. **Gripper geometry registration** — real gripper 有反光 / 遮挡, FPFH 可能 match 不上

Paper 用 learning-free 路径走通了, 说明 latent prediction 的几何 quality 足够好, 即使 registration 不完美也能 work。

---

## 7. 与相关工作的定位

### 7.1 World Model 谱系

- **Dreamer 系列** (Hafner et al.): latent dynamic model + actor-critic, 强 RL 导向, latent 是 unstructured vector
- **UniPi**: video 作为 world model, 2D pixel
- **TesserAct**: video + depth + normal, 2.5D
- **GAussian World Model (GWM, Lu et al., 2025)**: 直接在 3DGS 上做 dynamic, 但 per-scene fitting, 训练 dynamic 时耦合 appearance
- **本文**: structured 3D latent (TRELLIS-style) + flow matching dynamic, 解耦 appearance 与 geometry, latent 是 task-agnostic 的

### 7.2 与 3D Diffuser Actor (Ke et al., 2024) 对比 (Table 7)

3D Diffuser Actor 是 SOTA 3D policy, 在原 setting 上 90%, 我们 91%, 持平。但在 perturbation 下:
- Noise 0.08: 3D Diffuser Actor 48% vs Ours 90%
- Noise 0.10: 3D Diffuser Actor 7% vs Ours 86%

3D Diffuser Actor 仍用 3D representation 但不做 predictive rollout, 只做 obs→action mapping, 因此 visual perturbation 通过 point cloud encoder 漏到 action 上。我们的 predictive rollout 充当 "denoising prior", 把 observation noise 在 latent generation 阶段过滤掉。

### 7.3 与 π0, GR00T N1 等 VLA 对比

VLA (vision-language-action) 直接预训练 large-scale 数据, generalist 但 robot-specific tokenization, 跨 embodiment 困难。我们的方法 decouple "world understanding" (4D latent) 与 "control" (inverse dynamics), inverse dynamics 可以 learning-free 替换, 因此跨 robot setup 更友好。

---

## 8. Limitations 和 failure modes (paper Appendix D)

1. **Calibrated multi-view 依赖** — 单目 / 弱 calibration 场景不行。可结合 single-image-to-3D (e.g. LRM, TripoSR) 缓解
2. **Contact-rich 任务弱** — PegInsert 16%, 因为 fine geometry 误差导致 contact mismatch
3. **Tabletop only** — 没测 mobile manipulation, full-body humanoid
4. **Latent resolution 限制** — 64³ grid 对小物体 (e.g. key hole) 不够精细

Failure 来源:
- 3D rollout 的 fine geometric error (peg insertion 主要)
- Inverse dynamics 的 registration / prediction drift

---

## 9. 我的几个直觉 (build your intuition)

### 9.1 为什么 sparse voxel latent 比 dense grid / transformer token 更好

如果用 dense 64³, transformer token 数 = 262144, self-attention 是 O(N²), 显存爆炸。如果用纯 point cloud (无 voxel anchor), 没有 spatial inductive bias, transformer 难学 3D 结构。Sparse voxel 是 "锚点 + feature" 的折中: voxel 给离散位置先验, feature 给 continuous appearance, 这样 flow matching 只需学 "在已知 anchor 上分配 feature", 大幅降低学习难度。

### 9.2 为什么 SD 和 LG 分开而不是 joint

Joint training 会遇到 multi-modal conflict: SD 的目标是 "voxel 在不在那" (binary / set prediction), LG 的目标是 "feature 长啥样" (continuous regression)。两个 loss 的 gradient scale 不同, joint training 需要仔细 balance。分开训练还能让 SD 用更 coarse 的 16³ grid 加速, LG 用 64³ grid 保精度, 各自的 compute budget 搭配更优。

### 9.3 Condition augmentation 的作用

Random dropout voxel features + Gaussian noise 模拟 "观察不全 / 有噪声" 的情况。这相当于 training-time data augmentation, 让 model 见过 partial observation, inference 时即使某 view 被遮挡也能 robust。这和 dropout 在 classifier 中的作用相似, 但作用在 condition 而非 network 内部。

### 9.4 为什么不用 latent direct regression 而用 flow matching

Direct regression (e.g. MLP head 预测 $f_i$) 是 unimodal 的, 给定 $(z_t, c)$ 只能输出一个 $z_{t+1}$。但 robot planning 的 future 是 multi-modal 的 (同一 instruction 可以有多种解法, e.g. 抓 cube 从左还是从右)。Flow matching / diffusion 通过 sample noise $\epsilon$ 引入 stochasticity, 能 generate diverse futures, 这对 planning 的 exploration 和 inverse dynamics 的 robustness 都重要。

### 9.5 Closed-loop 的重要性

Open-loop 假设 predictive model 完美, 但 autoregressive rollout 累积误差, 几步后就 drift。Closed-loop 每步用真实 observation 重 encode, 等于 "predict 一步, 验证一步", 把 long-horizon 拆成多个 short-horizon, 误差不累积。代价是 sensing latency, 但对 manipulation 这种慢动力学任务可接受。这与 MPC (Model Predictive Control) 的 receding horizon 哲学一致。

### 9.6 这个范式对 foundation model 的启示

如果 4D latent predictive model 训练在足够 diverse 的 robot + scene 数据上, 它可以成为 "robotics foundation world model": 给定任意 instruction + scene, roll 出 4D future, 再用任意 inverse dynamics (learning-free registration 或 task-specific learned) 解出 action。这种 decoupling 让 world model 跨 robot 迁移, inverse dynamics 跨 task 迁移, 各自 scale。

---

## 10. 可能的后续方向 (hallucinate 一下)

1. **Single-view → 3D latent**: 用 LRM-style (Large Reconstruction Model) 单图重建 latent, 摆脱 multi-view calibration 依赖
2. **Active perception**: closed-loop 时主动选择下一个 view, 减少 observation cost
3. **Diffusion forcing (Chen et al., 2024) 集成**: 让 rollout 时长更 long-horizon 而不退化
4. **Hierarchical latent**: coarse-to-fine 之外加 "task → subtask → subgoal" 的 semantic hierarchy, 类似 Compositional Foundation Models (Ajay et al., 2023b)
5. **Latent-space reward / RL**: 把 4D predictive model 当 world model, 在 latent 里做 planning 或 RL, 不需要 decode 到 pixel
6. **Contact-aware representation**: 当前 voxel grid 对 contact-rich 任务精度不够, 引入 SDF / occupancy 补充几何细节
7. **Symbolic / relational latent**: 在 voxel feature 里加入 object-centric factorization, 支持 "pick up the red cube" 这种 compositional reasoning
8. **Cross-embodiment inverse dynamics library**: 像 RT-X 那样把不同 robot 的 inverse dynamics 统一, 让 world model 真正 embodiment-agnostic

---

## 11. 总结 takeaway

这篇 paper 的核心 contribution 不是 "新模型", 而是 "新 representation + 新 training paradigm":
- **Representation**: structured sparse voxel latent (TRELLIS) 提供 view-invariant 3D prior
- **Training**: flow matching 在 latent 上学 dynamic, 分解 SD (coarse geometry) + LG (fine feature)
- **Planning**: latent rollout + inverse dynamics, learning-free registration 让 modular

最强 result 不是绝对 success rate, 而是 **zero-shot generalization 几乎不掉**——这是 representation-level inductive bias 的胜利, 也是 generative planner 范式相对 end-to-end policy 的根本优势。

如果让我打个比方: end-to-end policy 是 "看见即反应" (System 1), UniPi 是 "想象 future video 再反应" (System 2 但 2D), 我们的方法是 "想象 4D 结构再反应" (System 2 + 3D grounded)。Robot manipulation 的 fine-grained 特性, 注定需要 3D-grounded 的想象。

---

## Reference Links

- Paper: https://structured-4d-model.github.io/
- TRELLIS (base representation): https://trellis3d.github.io/
- UniPi: https://universal-policy.github.io/
- TesserAct: https://arxiv.org/abs/2504.20995
- DP3: https://3d-diffusion-policy.github.io/
- DP (Diffusion Policy): https://diffusion-policy.cs.columbia.edu/
- Flow Matching (Lipman et al.): https://openreview.net/forum?id=PqvMRDCJT9t
- ManiSkill3: https://github.com/haosulab/ManiSkill
- LIBERO: https://libero-project.github.io/
- RLBench: https://sites.google.com/view/rlbench
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- DINOv2: https://dinov2.metademolab.com/
- CLIP: https://openai.com/research/clip
- 3D Diffuser Actor: https://3d-diffuser-actor.github.io/
- π0: https://arxiv.org/abs/2410.24164
- GR00T N1: https://arxiv.org/abs/2503.14734
- GWM (Gaussian World Model): https://arxiv.org/abs/2506.14135
