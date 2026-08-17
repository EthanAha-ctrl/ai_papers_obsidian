---
source_pdf: PerpetualWonder Long-Horizon Action-Conditioned 4D Scene Generation.pdf
paper_sha256: 9562d4a1f635e645d32b2c499ef5eb4fb049a42f81b3bd06cc54a0cf25800a98
processed_at: '2026-08-06T02:43:36-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 PerpetualWonder

Andrej，我换一种讲法，咱们像在咖啡馆白板上聊这个事。

---

## 这帮人到底想干嘛

Stanford 这组人就在问一个问题：**给一张照片，再告诉它"推一下"、"吹一阵风"、"挖一铲子"，能不能生成一段 3D 的、能从任意角度看、还能连续做多步动作的视频？**

听起来简单，做起来是个深坑。之前的方案要么能控制动作但画面假，要么画面真但没法响应动作，最好的折中方案 WonderPlay 也只能做"一步"，做多步就崩。

---

## 为什么做"多步"这么难

想一下 WonderPlay 的做法，它分两步走：

**第一步**：传统物理引擎跑出粗 dynamics——球该往哪滚、布该怎么飘，物理算出来。

**第二步**：把这个粗结果丢给 video generation model，让它"美化"——加阴影、加水花、加真实材质感。

听起来挺好。问题在 representation 上。物理引擎用的是 **particles**，video model 美化的是 **Gaussian splatting primitives**。这两套东西是**分开存的**。物理 particle 驱动 gaussian 位置没问题，但 video model 把 gaussian 优化得更好看之后，这个改进**回不到 particle**。

打个比方：你有个机器人手臂（physics），手臂上挂个手套（visual）。物理算出手臂动了 10cm，手套跟着动。但 video model 说"手套其实应该旋转 5°才更像真手"，它优化了手套的旋转。下一轮你让手臂再做动作，手臂还从原位置出发——手套上次的旋转改进全丢了。

所以 WonderPlay 做第二步、第三步动作时，每步都从"原始状态"重来，误差越积越大，castle 那个例子里铲子插进去就裂开了。

---

## PerpetualWonder 的核心 trick：VPP

VPP 全称 Visual-Physical Aligned Particle。**一句话概括：让每个 physics particle 同时当 K 个 gaussian 的"锚点"。**

这样物理量和视觉量就长在同一棵树上。particle 动，gaussian 跟着动；gaussian 被 video model 优化，优化结果通过 averaging 回灌给 particle。

这个"回灌"就是闭环的关键。下一轮 physics simulation 就基于 corrected particle 状态做，误差不累积。

### 几何约束怎么做

gaussian 的位置不是随便挂的，有个硬约束（公式 1）：

$$\mu_{j,k} = p_j + \tanh(\tilde{p}_{j,k}) \cdot \delta$$

- $p_j$：第 $j$ 个 physics particle 的位置
- $\tilde{p}_{j,k}$：可学习的小偏移
- $\delta$：particle 的半径（$10^{-2}$）
- $\tanh$：把偏移压在 $(-\delta, \delta)$ 内

$\tanh$ 是关键——gaussian 想跑也跑不出 particle 周围一个半径的范围。这就保证了 binding 不散。

### 还有个 loss 兜底（公式 4）

$$\mathcal{L}_{\text{sim}} = \frac{1}{T \cdot J} \sum_{t=1}^{T} \sum_{j=1}^{J} \left\| p_{j,t} - \frac{1}{K} \sum_{k=1}^{K} \mu_{j,k,t} \right\|_2^2$$

意思是：每个 particle 锚定的 K 个 gaussian 的**平均位置**必须等于 particle 位置。gaussian 可以在 particle 周围散开表达细节，但中心不能偏。

这两层约束（$\tanh$ 几何约束 + $\mathcal{L}_{\text{sim}}$ 损失约束）合起来，就是"bidirectional bridge"。

### 材质自适应

不是所有材质都用一样的绑定方式（Supplementary D）：

- **Rigid body、cloth**：K=1，一对一严格绑，gaussian scale = $\delta$。因为表面紧致。
- **Gas、liquid、sand、snow、elastic**：K=20，一对多，gaussian scale = $0.5\delta$。因为体积物质需要一片 gaussian 撑起一个 particle 的体积。

---

## 第二个 trick：Multi-View Optimization

光有 VPP 还不够。video model 从单一视角 refine 视频，只能约束那个视角的渲染。换角度看就崩——因为 3DGS 有歧义，多个 gaussian 配置能产生相同 2D 图。

WonderPlay 就栽在这——single-view optimization，novel view 渲染烂掉（imaging score 只有 36.80）。

### 先把 scene 建全

PerpetualWonder 从单图出发，用 **GEN3C**（NVIDIA 的 camera-controlled video model）生成 242 个 dense surrounding views。Supplementary A 说分两条轨迹——"arc left" 和 "arc right"，各 90°，避免一条 180° 大轨迹导致 inconsistency。

然后 COLMAP 重建 point cloud，3DGS 优化，SAM2 分割前后景，Gaussian Grouping 分离物体。rigid body 还特殊处理——用 Hunyuan3D 直接生成高质量 mesh，再 6-DoF 对齐到 scene 里。

这样得到的 scene 是真 3D，能任意视角渲染。

### 三阶段 progressive 优化

video model 从不同视角 refine 出来的视频**天然不一致**——frontal view 可能 hallucinate 苹果是红的，side view 没这个细节。直接一起优化就打架，结果模糊闪烁。

所以用 progressive 策略（Section 3.2）：

1. **阶段 1**：只用输入图视角的 refined video 优化
2. **阶段 2**：加其他视角，但 control weight 调小
3. **阶段 3**：所有视角一起优化，最终一致

实际用 3 个 key views：frontal、left-side、right-side。

这就像 curriculum learning——先在简单约束下建立基础，再慢慢引入冲突信号。

---

## 闭环怎么转

整个 simulation loop 三个阶段循环（Section 3.3）：

**Forward Pass $\Phi_p$**：Genesis physics engine 推 392 步，每步 $10^{-3}$ 秒，10 个 sub-steps。支持 cloth、sand、snow、liquid、smoke、elastic、rigid body。物理参数（Young's modulus、Poisson's ratio、friction angle 等）用 GPT-4o 从输入图估算，可选 manual tuning。

**Backward Optimization $\Psi_n$**：渲染 RGB + optical flow（704×1280，49 frames），CogVideoX + Go-with-the-flow 做 video refinement（bimodal control），然后 progressive multi-view optimization。

**Loop Closure**：把 refined state 转成下一 window 的初始 state。particle 位置更新为：

$$p_j \leftarrow \frac{1}{K} \sum_{k=1}^{K} \mu_{j,k,T}$$

velocity 直接继承 physics solver 在 $T$ 时刻算出来的值。理由是 $\mathcal{L}_{\text{sim}}$ 已经限制了 gaussian 偏移很小，所以 particle 位置修正幅度小，velocity 不用重估。

这一步就是"闭环"——refined visual state 回流到 physics state，下一轮 forward 基于修正后的 state。

---

## 实验数据说了什么

### 自动指标（Table 1，World-Score）

| Method | Camera Ctrl | 3D Consist | Imaging |
|---|---|---|---|
| Veo3.1 | 60.61 | 73.93 | 67.82 |
| WonderPlay | 75.95 | 63.93 | 36.80 |
| **PerpetualWonder** | **93.26** | **80.41** | 66.98 |

Camera Ctrl 93.26 碾压——因为真 3D scene，相机随便摆。3D Consist 80.41 也是最高。Imaging 66.98 不算顶尖（Veo3.1 67.82 更高），这是 trade-off：为了 3D consistency 和 controllability，sacrifice 了一些 visual fidelity。WonderPlay 的 imaging 只有 36.80，novel view 崩了。

### 人类评估（Table 2，350 人 2AFC）

vs Veo3.1：physics plausibility 62.0%，motion fidelity 70.8%。Veo3.1 是最强 video model 之一，implicit physics 很强，但 PerpetualWonder 的 explicit physics + closed loop 还是赢。

vs WonderPlay：80.8% / 86.3%，差距明显。

vs GEN3C：93.5% / 83.5%，GEN3C 完全不响应 action，physics plausibility 输得最惨。

### Ablation

**VPP vs 标准 3DGS（Figure 6）**：标准 3DGS 没 $\mathcal{L}_{\text{sim}}$ 约束，只 photometric loss 驱动，结果 chaotic degenerate。VPP 干净。

**Progressive vs Direct（Figure 7）**：直接多视角一起优化，apple 模糊闪烁。Progressive 干净。

### Runtime（Table S2）

每个 time window ~16 分钟（initialization 8 min + forward <1 min + backward 7 min）。3 个 window 接近 50 分钟。不是 real-time，backward optimization 是瓶颈。

---

## 我觉得最 elegant 的点

**Representation 决定信息流的可能性**。decoupled representation 只能单向流；unified representation 才能 closed loop。这是论文最深的 insight，也是 Karpathy 你在 Software 2.0 里反复强调的——正确的 representation 让 gradient flow 自然走通。

VPP 的 $\tanh$ + $\mathcal{L}_{\text{sim}}$ 双约束，像极了 differentiable physics 的弱化版。不用真正 backprop through physics solver，用 $\mathcal{L}_{\text{sim}}$ 把 visual gradient 间接传到 particle position。velocity 直接继承是 hack，但有效。

---

## 我觉得可以质疑的点

1. **velocity 直接继承是 hack**。如果 visual refinement 改变了物体速度（小球弹得更高、布料飘得更远），这个改动会丢。理论上应该从 refined positions 用 finite difference 重新估 velocity。论文没做这个 ablation。

2. **$\mathcal{L}_{\text{sim}}$ 用 L2 平均约束**，可能限制 visual primitives 表达大幅度形变。布料大幅翻折时，gaussian 平均位置可能偏离 particle 中心。论文没讨论 $\lambda_{\text{sim}}$ 的 sensitivity。

3. **16 分钟一个 window 太慢**，没法做 interactive embodied AI 或 RL training。需要 distillation 或加速。

4. **Imaging quality 66.98 不算高**。Veo3.1 的 67.82 说明 pure video model 在 visual fidelity 上还是强。hybrid 的优势在 controllability + consistency，不在 raw fidelity。

5. **Veo3.1 在 3D Consist 上已经 73.93**，说明 video model 自己也在学 implicit 3D consistency。如果未来 video model 进化到完全 3D consistent，hybrid simulator 的优势会缩小。

6. **hockey stick 失败案例**（Figure S3）暴露 single-image 4D 根本限制——不能 hallucinate 未见几何。3D 表示的 rigidity 限制了 generative 能力。

7. **多物体复杂碰撞场景没展示**。论文里都是单物体交互，多物体碰撞、堆叠、流体-刚体耦合这些复杂场景表现如何不清楚。

---

## 更深的联想

**VPP vs NeRF**：VPP 本质是 particle-anchored 4D gaussian，把 NeRF 的 implicit volume 换成 explicit particle + gaussian。explicit 的好处是 physics solver 直接可作用，不用 differentiable rendering through neural field。

**VPP vs Differentiable Physics**：闭环是 differentiable physics 的弱化版。真正 differentiable physics 要 backprop through solver，数值不稳定且慢。PerpetualWonder 用 $\mathcal{L}_{\text{sim}}$ 把 visual gradient 间接传到 particle，绕过了 backprop through solver 的难题。

**VPP vs Embodied AI Environment**：embodied AI 训练需要可交互 world model。PerpetualWonder 这种 closed-loop action-conditioned 4D simulator 正是所需，但 16 分钟一个 window 太慢。需要 distillation 成更小模型，或者把 physics solver 嵌入 video model 的 latent space。

**Hybrid vs Pure Video World Model**：Sora、Veo 把 physics 隐式编码在网络里，泛化性好但不可控。PerpetualWonder 把 physics 显式放在 solver 里，可控但需要手工材质参数。未来路线可能是 video model 学到足够强 implicit physics 后，hybrid 优势缩小；或者 hybrid 把 VLM 估算的物理参数也 learnable，减少人工。

**未来方向猜想**：把 video refiner 换成 distilled 小模型，把 physics solver 嵌入 video model latent space，把 VLM 物理参数估算也 learnable，可能达到 real-time。或者把 VPP representation 直接做进 video model 的 latent 里，让 video model 原生支持 action-conditioned 4D generation。

---

## 一句话总结

**PerpetualWonder 把 hybrid generative simulator 从开环变闭环，靠 VPP 让 physics particle 和 visual gaussian 共享同一套 anchor 结构，配合 progressive multi-view optimization 消除 refinement 歧义。Representation 决定信息流，unified representation 才能 closed loop。**

参考链接：
- 项目主页：https://johnzhan2023.github.io/PerpetualWonder/
- WonderPlay 前作：https://wonderplay-2025.github.io/
- GEN3C：https://research.nvidia.com/labs/toronto-ai/GEN3C/
- CogVideoX：https://github.com/THUDM/CogVideo
- Go-with-the-flow：https://withertoy.github.io/go_with_the_flow/
- Genesis Physics Engine：https://github.com/Genesis-Embodied-AI/Genesis
- 3D Gaussian Splatting：https://github.com/graphdeco-inria/gaussian-splatting
- Gaussian Grouping：https://github.com/lkeab/gaussian-grouping
- SAM2：https://github.com/facebookresearch/sam2
- Hunyuan3D 2.0：https://github.com/Tencent/Hunyuan3D-2
- WorldScore benchmark：https://worldscore-3d.github.io/
- PhysDreamer：https://physdreamer.github.io/
- PhysGaussian：https://github.com/yanqinJiang/PhysGaussian
- FreeTimeGS（temporal opacity 灵感来源）：https://zkzhou.net/FreeTimeGS/

---

# PerpetualWonder 深度解读

Andrej，这篇 paper 我从你 build intuition 的角度讲透。Stanford 的 Jiahao Zhan、Zizhang Li、Hong-Xing Yu、Jiajun Wu 这组人做的事，本质上是在问一个深层问题：**物理 simulation 与 neural generation 之间能不能形成一个真正的 closed loop？** WonderPlay 之前的答案是"半开环"——physics 推 dynamics，video model refine appearance，但 refined appearance 永远回不到 physics state。PerpetualWonder 把这个回路闭合上了。

项目主页：https://johnzhan2023.github.io/PerpetualWonder/

---

## 1. 任务定义

输入：单张图 $I$ + 一串 actions $\{\mathcal{A}_t\}_{t=0}^{T-1}$，其中 action 可以是 global force field $\mathbf{f}(x,y,z,t)$（gravity、wind）或者 local force $\mathbf{f}(t)$（push、poke）。

输出：4D scene sequence $\{\mathcal{S}_t\}_{t=0}^{T}$，每个 scene state $\mathcal{S}_t = (\mathcal{B}_t, \mathcal{F}_t)$ 分为 background 和可交互 foreground。

关键约束：long-horizon，即支持连续多个 time window 的 sequential actions。WonderPlay 只能做 single window，PerpetualWonder 能 perpetual cycle。

---

## 2. WonderPlay 的根本病灶：Decoupled Representation

要理解 PerpetualWonder 的设计动机，必须先理解 WonderPlay 为什么会 fail at long-horizon。

WonderPlay 的 representation 设计：
- **Physics side**：MPM/PBD particles $\{p_j\}$ 表达 dynamics
- **Visual side**：独立的 3D Gaussian Splatting primitives 表达 appearance
- **Binding**：单向。Physics particles 移动 → 驱动 gaussians 位置，但 gaussians 优化后不会反过来更新 particles。

后果：在每个 time window 末尾，video model refine 出来的 gaussian 位置、形变、新细节，全部丢失——下一个 window 启动时，physics simulator 还是用初始的 particle 状态做 forward。Error 累积，castle 那个例子里铲子插进去就崩了。

这就像你写 RL 的时候，environment 的真实 state 和 agent 观测到的 state 是脱节的——agent 永远学不到 environment 的真实反馈。这里 physics simulator 是 environment，video model 是 refiner，但 refiner 的修正没回流到 environment。

Karpathy 视角的 intuition：**representation 决定了信息流的可能性**。如果物理 state 和 visual state 用两套数据结构存，它们之间只能做单向 copy；要做 closed loop，必须用一套数据结构同时表达两件事。

---

## 3. 核心创新一：VPP（Visual-Physical Aligned Particle）

### 3.1 设计哲学

VPP 的核心思想是：**让每个 physics particle 成为 K 个 gaussian 的 anchor**。这样 physics particle 既是物理量（有位置 $p_j$、速度 $v_j$），又是 visual 量的根（gaussians 锚在它上面）。Forward 时 particle 动，gaussians 跟着动；backward 时 gaussians 优化，优化结果通过 averaging 回灌给 particle。

这是非常优雅的"shared substrate"设计，类似你在 micrograd 里讲的——用一个 computation graph 同时承载 forward 和 backward。

### 3.2 公式解析

**Position offset（公式 1）**：

$$\mu_{j,k} = p_j + \tanh(\tilde{p}_{j,k}) \cdot \delta$$

变量解释：
- $\mu_{j,k}$：第 $j$ 个 physics particle 锚定的第 $k$ 个 gaussian 的最终 3D 位置
- $p_j$：第 $j$ 个 physics particle 的位置（来自 physics solver）
- $\tilde{p}_{j,k}$：可学习的 position offset，是网络参数
- $\delta$：physics particle 的尺寸（论文里 $\delta = 10^{-2}$）
- $\tanh$：关键！把 offset 限制在 $(-\delta, \delta)$ 区间内

为什么用 $\tanh$？因为它保证 gaussian 不会漂离 anchor particle 超过一个 particle radius。这就是 bidirectional bridge 的几何约束。如果没有这个 $\tanh$，gaussians 在 backward optimization 中会被 photometric loss 推到任意位置，binding 就失效了。

**Temporal opacity（公式 2）**：

$$o_t(t) = \exp\left(-\frac{1}{2}\left(\frac{t - \mu_t}{s_d}\right)^2\right)$$

变量解释：
- $\mu_t$：temporal center（这个 gaussian 在时间轴上的"激活中心"）
- $s_d$：temporal duration（高斯的标准差）
- $o_t(t)$：时刻 $t$ 的 temporal opacity

最终 opacity 是 $o(t) = o_s \times o_t(t)$，其中 $o_s$ 是标准 spatial opacity。

这个设计来自 FreeTimeGS（论文 ref [42]）。直觉：每个 gaussian 在时间维度上像一个"脉冲"，只在某个时刻附近显眼。这对 4D 表达很重要——一个 gaussian 不需要全程都"亮"，可以只在它该出现的时刻亮。这对 smoke、liquid、splash 这种 emitter 物质尤其有用，因为新物质是逐渐生成的，老物质是逐渐消失的。

**Scale**：固定 isotropic，不大于 $\delta$。Supplementary 里 Figure S1 说明 isotropic 比 anisotropic 在 novel view 下表现更好，因为 anisotropic 容易 overfit 输入视图。

### 3.3 VPP 的超参配置（Supplementary D）

非常工程化的设计——根据材质自适应：

| 材质类型 | K（每个 particle 锚定几个 gaussian） | gaussian scale |
|---|---|---|
| Solid/Surface（rigid body、cloth） | K=1 | $\delta$（一对一严格绑定） |
| Volumetric/Emitter（gas、liquid、sand、snow、elastic） | K=20 | $0.5\delta$（一对多，覆盖体积） |

Intuition：rigid body 一个 particle 对一个 gaussian 就够了，因为表面紧致；volumetric 需要一个 particle 撑起一片体积，所以挂 20 个小 gaussian，scale 减半以表达细粒度。

---

## 4. 核心创新二：Multi-View Optimization

### 4.1 为什么要 multi-view？

Video model 从单一视角 refine 出来的视频，只能约束该视角下的渲染。当你想从 novel view 渲染时，因为 3D 表示没被多个视角约束过，会有严重 ambiguity——3DGS 的多个 gaussian 配置可以产生相同的 2D 渲染。

WonderPlay 用 single-view refinement，所以在 novel view 下崩了（Figure 4 中间行可以看到）。

### 4.2 3D Scene Initialization

这一步是 PerpetualWonder 比 WonderPlay 强的另一关键。

WonderPlay 的初始化：单图 depth unprojection + 手动物体放置。窄 baseline，只能小范围转视角。

PerpetualWonder 的初始化：
1. 用 GEN3C（camera-controlled video model）从单图生成 dense surrounding views。Supplementary A 说总共 242 views，分成 "arc left" 和 "arc right" 两条轨迹，每条 90°，避免一条 180° 大轨迹导致 inconsistency。
2. COLMAP 重建 point cloud
3. 3DGS 优化得到 $N$ 个 gaussian primitives $\{G_i\}_{i=1}^N$，每个有 position $p_i$、orientation $q_i$、scale $s_i$、opacity $o_i$、color $c_i$，加上一个 learnable feature $g_i$ 用于 segmentation
4. SAM2 在 dense views 上抠 mask，supervise learnable feature（这是 Gaussian Grouping 的做法）
5. 分离前后景；foreground gaussians 通过 TSDFusion 转 mesh
6. rigid body 特殊处理：用 Hunyuan3D 直接生成高质量 mesh，再用 6-DoF pose + scale 优化对齐到 scene（Supplementary B）
7. Mesh 体采样 physics particles $\mathcal{P}_0$
8. 再做一轮 3DGS 优化，把原始 foreground gaussians 替换成 VPP 形式

这样得到的 scene 是真正的 3D，可以任意视角渲染。

### 4.3 Loss function（公式 3 和 4）

总 loss：

$$\mathcal{L} = \mathcal{L}_p(\text{Render}(\mathcal{B}_t) \odot (1-\mathbf{M}), \mathbf{V}_t \odot (1-\mathbf{M})) + \mathcal{L}_p(\text{Render}(\mathcal{G}_t), \mathbf{V}_t \odot \mathbf{M}) + \lambda_{\text{sim}} \mathcal{L}_{\text{sim}}$$

变量解释：
- $\mathbf{M}$：foreground 二值 mask
- $\text{Render}(\cdot)$：gaussian rendering function
- $\mathcal{L}_p$：photometric loss = L1 + SSIM
- $\mathbf{V}_t$：video model refine 出来的视频帧
- $\mathcal{B}_t$：background gaussians（也带 spatial + temporal opacity，捕捉阴影等 secondary effect）
- $\mathcal{G}_t$：foreground VPP
- $\lambda_{\text{sim}}$：simulation consistency loss 的权重
- $\mathcal{L}_{\text{sim}}$：公式 4

第一项：背景渲染与视频背景对齐
第二项：前景 VPP 渲染与视频前景对齐
第三项：simulation consistency，关键正则项

**Simulation consistency loss（公式 4）**：

$$\mathcal{L}_{\text{sim}} = \frac{1}{T \cdot J} \sum_{t=1}^{T} \sum_{j=1}^{J} \left\| p_{j,t} - \frac{1}{K} \sum_{k=1}^{K} \mu_{j,k,t} \right\|_2^2$$

变量解释：
- $T$：time window 的时间步数
- $J$：physics particle 数
- $K$：每个 particle 锚定的 gaussian 数
- $p_{j,t}$：时刻 $t$ 第 $j$ 个 particle 位置
- $\mu_{j,k,t}$：时刻 $t$ 第 $j$ 个 particle 锚定的第 $k$ 个 gaussian 位置

Intuition：强制每个 particle 锚定的所有 gaussians 的**平均位置**等于 particle 位置。这允许 gaussians 在 particle 周围分布（表达形变细节），但中心不能偏移。

这个 loss 是 VPP closed loop 的关键——它把 visual optimization 的结果约束在 physics particle 的"附近"，使下一步 backward update particle 时有意义。

### 4.4 Progressive Multi-View Optimization

Video model 从不同视角 refine 出来的视频**天然不一致**——frontal view 可能 hallucinate 一个红色苹果，side view 可能没有。直接用所有视角的 loss 一起优化，会冲突，得到模糊或闪烁的结果。

Progressive 策略三步走：
1. **Step 1**：只用输入图像视角的 refined video 做 optimization，建立基础
2. **Step 2**：渲染其他视角，用 video model refine，但用**更小的 control weight**（论文里没给具体数值，应该是 heuristic）
3. **Step 3**：用所有视角的 refined videos 同时优化，得到最终 consistent 4D scene

Intuition：先单视角建基础，再慢慢引入其他视角，避免一开始就被冲突信号打乱。这有点像 curriculum learning，也像 Adam 优化器的 warmup。

实现里用了 3 个 key views：frontal、left-side、right-side。

---

## 5. Closed-Loop Simulation（核心闭环）

### 5.1 Forward Pass $\Phi_p$

$$\hat{\mathcal{S}}_{t+1} = \Phi_p(\hat{\mathcal{S}}_t, \mathcal{A}_t)$$

用 Genesis physics simulator 推进 $T = 392$ 步，每步 step time $10^{-3}$，10 个 sub-steps。支持多种材质：cloth、sand、snow、liquid、smoke、elastic、rigid body。

物理参数用 GPT-4o（VLM）从输入图估算，可选 manual tuning（Table S1 给了完整列表）。

输出：coarse sequence $\{\hat{\mathcal{S}}_t\}_{t=0}^T$。

### 5.2 Backward Optimization $\Psi_n$

把 coarse sequence 渲染成 RGB + optical flow，分辨率 H=704, W=1280，每 8 步采 1 frame，共 49 frame。

用 CogVideoX + Go-with-the-flow 做 video refinement，bimodal control（RGB + optical flow 双控）。

然后做 progressive multi-view optimization（见第 4 节），得到 refined sequence $\{\mathcal{S}_t\}_{t=0}^T$。

### 5.3 Loop Closure（关键！）

这一步是 PerpetualWonder 的灵魂——把 refined state 转化为下一个 time window 的初始 state。

**Particle 位置更新**：

$$p_j \leftarrow \frac{1}{K} \sum_{k=1}^{K} \mu_{j,k,T}$$

即用时刻 $T$ 的所有 anchor gaussians 的平均位置作为新 particle 位置。

**Velocity 继承**：

$$v_j \leftarrow v_{j,T}$$

直接继承 physics solver 在 $T$ 时刻算出来的 velocity。理由：因为 $\mathcal{L}_{\text{sim}}$ 已经限制了 visual primitives 不能偏离 particle 太远，所以 particle 位置的更新是小幅度的"correction"，velocity 不需要重新估计。

这就是 closed loop 的本质——refined visual state 回流到 physics state，下一个 forward pass 就基于 corrected state 进行。

---

## 6. 实验数据深度解读

### 6.1 World-Score 自动指标（Table 1）

| Method | Camera Ctrl | 3D Consist | Imaging |
|---|---|---|---|
| Wan2.2 | 59.73 | 65.35 | 67.03 |
| GEN3C | 80.29 | 61.69 | 66.25 |
| WonderPlay | 75.95 | 63.93 | 36.80 |
| Tora | 51.80 | 60.77 | 54.37 |
| Wan2.6 | 64.75 | 70.49 | 66.09 |
| DaS | 78.96 | 62.18 | 60.23 |
| Veo3.1 | 60.61 | 73.93 | 67.82 |
| **PerpetualWonder** | **93.26** | **80.41** | 66.98 |

关键观察：
- **Camera Ctrl 93.26**：碾压级。因为 PerpetualWonder 是真正的 3D scene，相机随便摆。
- **3D Consist 80.41**：也是最高。Veo3.1 73.93 第二，说明 Veo3.1 video prior 很强，但还是输给 PerpetualWonder 的几何一致性。
- **Imaging 66.98**：中等。注意这是 trade-off——PerpetualWonder 为了 3D consistency 和 controllability 牺牲了一些 imaging quality。WonderPlay 的 imaging 只有 36.80，崩了，因为 single-view optimization 导致 novel view 渲染烂。

### 6.2 2AFC Human Study（Table 2）

350 个 participant，每个评 10 个 scene，二选一。

| 对比 baseline | Physics Plausibility | Motion Fidelity |
|---|---|---|
| vs Wan2.2 | 74.1% | 71.8% |
| vs GEN3C | 93.5% | 83.5% |
| vs WonderPlay | 80.8% | 86.3% |
| vs Veo3.1 | 62.0% | 70.8% |
| vs Wan2.6 | 68.5% | 77.3% |
| vs Tora | 83.5% | 85.3% |
| vs DaS | 80.9% | 81.9% |

注意 vs GEN3C 93.5%——GEN3C 完全不响应 action，所以 physics plausibility 上输得最惨。
vs Veo3.1 只有 62.0% / 70.8%——Veo3.1 是最强 video model 之一，video prior 很强，能学到一些 implicit physics。但 PerpetualWonder 仍然赢，因为 explicit physics + closed loop 比 implicit physics 更可控。

### 6.3 Ablation

**Ablation 1：VPP vs 标准 3DGS（Figure 6）**
- VPP：visual primitives 受 $\mathcal{L}_{\text{sim}}$ 约束，不能漂移，dynamics 干净
- 标准 3DGS：unconstrained，只 photometric loss 驱动，结果 chaotic，degenerate

**Ablation 2：Progressive vs Direct optimization（Figure 7）**
- Progressive：先单视角建基础，再引入其他视角，apple 干净
- Direct：一开始就多视角同时优化，apple 模糊闪烁

### 6.4 失败案例（Figure S3）

Hockey stick 从画面外移入——mesh 不完整，看起来比真实短。PerpetualWonder 不能 hallucinate 未见区域的几何。这是 single-image 4D generation 的根本限制。

### 6.5 Runtime（Table S2）

| Stage | Time |
|---|---|
| Initialization | ~8 min |
| Forward Pass | <1 min |
| Backward Opt. | ~7 min |
| **Total (1st Loop)** | ~16 min |

非 real-time。每个 time window ~16 分钟，3 个 window 就是 48 分钟左右。Backward optimization 是瓶颈。

---

## 7. 关键 intuition 总结

1. **Representation 决定信息流**：decoupled representation（WonderPlay）只能单向信息流；unified representation（VPP）才能 closed loop。这是论文最深的 insight。

2. **VPP 的 bidirectional bridge**：physics particle 是 anchor，gaussian 是 leaf。Forward 时 particle 推 gaussian，backward 时 gaussian 优化结果通过 averaging 回灌 particle。$\tanh$ 和 $\mathcal{L}_{\text{sim}}$ 是 binding 的两个约束。

3. **Multi-view 解决 ambiguity**：3D 表示从单视图优化有歧义，多视图 supervise 消除歧义。但 video model 多视图不一致，所以需要 progressive 策略 + $\mathcal{L}_{\text{sim}}$ 正则。

4. **Loop closure 是 long-horizon 的关键**：没有 closure，每个 time window 都从原 state 重新开始，error 累积。有了 closure，refined state 是下一 window 的 corrected initial state。

5. **Hybrid simulator 的优势**：physics solver 给 controllability，video model 给 realism，closed loop 让两者协同进化。这比纯 video world model（Sora、Veo）更可控，比纯 physics simulator（PhysGen3D）更真实。

---

## 8. 与相关工作的关系网

- **WonderPlay** [21]：直接前作，decoupled representation，single-view，single-window。
- **PhysMotion** [38]：另一个 hybrid simulator，类似 limitation。
- **PhysDreamer** [58]：physics-based interaction，但更早期。
- **PhysGaussian** [48]：physics-integrated 3DGS，但纯 physics，没 video prior。
- **GEN3C** [34]：PerpetualWonder 用它做 dense view 生成。它本身是 3D-aware camera-controlled video model。
- **CogVideoX** [50] + **Go-with-the-flow** [5]：video refiner 组合，bimodal control。
- **Genesis** [19]：physics simulator，支持多材质。注意 ref [19] 在 paper 里指向一个 NLP paper，应该是引用错误，实际指的 Genesis physics engine（https://github.com/Genesis-Embodied-AI/Genesis）。
- **Gaussian Grouping** [52]：scene decomposition 方法。
- **SAM2** [31]：mask supervision。
- **Hunyuan3D** [61]：rigid body mesh 生成。
- **FreeTimeGS** [42]：temporal opacity 的灵感来源。

---

## 9. 可能的延伸联想

- **VPP 与 Neural Radiance Field 的关系**：VPP 本质是 particle-anchored 4D gaussian，类似把 NeRF 的 implicit volume 表达换成 explicit particle + gaussian。explicit 的好处是 physics solver 直接可作用。
- **VPP 与 Differentiable Physics 的关系**：闭环本质是 differentiable physics 的弱化版——不用真正 backprop through physics solver，而是用 $\mathcal{L}_{\text{sim}}$ 把 visual gradient 间接传到 particle position。Velocity 直接继承是 hack，但有效。
- **VPP 与 Embodied AI 的关系**：如果你要训 embodied agent，需要可交互的 world model。PerpetualWonder 这种 closed-loop action-conditioned 4D simulator 正是 embodied AI 需要的 environment。但是 16 分钟一个 window，太慢，没法做 RL training loop。需要 distillation 或加速。
- **VPP 与 World Model 的本质区别**：Sora、Veo 这种 latent world model 把 physics 隐式编码在网络里；PerpetualWonder 把 physics 显式放在 solver 里。前者泛化性好但不可控，后者可控但需要手工材质参数。Hybrid 是中间路线。
- **未来方向**：把 VLM 估算的物理参数也 learnable，把 video refiner 换成更小的 distilled model，把 physics solver 嵌入到 video model 的 latent space，可能能达到 real-time。

---

## 10. 代码层面可能实现的关键点

如果我要复现，关键代码片段思路：

```python
class VPP:
    def __init__(self, particles, K, delta, material_type):
        self.p = particles  # [J, 3] physics particle positions
        self.v = ...  # [J, 3] velocities
        if material_type in ['rigid', 'cloth']:
            K = 1
            scale = delta
        else:  # volumetric
            K = 20
            scale = 0.5 * delta
        self.offset = nn.Parameter(torch.randn(J, K, 3) * 0.01)  # \tilde{p}_{j,k}
        self.delta = delta
        self.spatial_opacity = nn.Parameter(torch.randn(J, K))
        self.temporal_center = nn.Parameter(torch.randn(J, K))
        self.temporal_duration = nn.Parameter(torch.randn(J, K))
    
    def gaussian_positions(self, t):
        # \mu_{j,k} = p_j + tanh(\tilde{p}_{j,k}) * \delta
        mu = self.p[:, :, None, :] + torch.tanh(self.offset) * self.delta
        return mu
    
    def temporal_opacity(self, t):
        # exp(-0.5 * ((t - \mu_t) / s_d)^2)
        return torch.exp(-0.5 * ((t - self.temporal_center) / self.temporal_duration)**2)
    
    def simulation_consistency_loss(self, t):
        # ||p_{j,t} - (1/K) \sum_k \mu_{j,k,t}||_2^2
        mu = self.gaussian_positions(t)  # [J, K, 3]
        mu_mean = mu.mean(dim=1)  # [J, 3]
        return ((self.p - mu_mean)**2).sum() / (self.p.shape[0])
    
    def update_particles_from_optimized_gaussians(self, T):
        # Loop closure: p_j <- (1/K) \sum_k \mu_{j,k,T}
        mu = self.gaussian_positions(T)
        self.p = mu.mean(dim=1).detach()
        # velocity inherited from physics solver
```

Backward optimization 主循环大概：

```python
for window in range(num_windows):
    # Forward pass
    coarse_seq = physics_simulator(scene, actions[window])
    
    # Render coarse
    rgb_frames, flow_frames = render_coarse(coarse_seq, viewpoints)
    
    # Video refinement
    refined_videos = {view: video_model(rgb_frames[view], flow_frames[view]) 
                      for view in viewpoints}
    
    # Progressive optimization
    # Step 1: only input view
    optimize(scene, {input_view: refined_videos[input_view]}, 
             control_weights={input_view: 1.0})
    
    # Step 2: add other views with smaller weight
    optimize(scene, refined_videos, 
             control_weights={v: 0.5 if v != input_view else 1.0 
                              for v in viewpoints})
    
    # Step 3: all views equal
    optimize(scene, refined_videos, 
             control_weights={v: 1.0 for v in viewpoints})
    
    # Loop closure
    scene.update_particles_from_optimized_gaussians(T=window_end)
```

---

## 11. 个人评价与潜在质疑

**优点**：
- Representation 设计非常 elegant。VPP 的 $\tanh$ + $\mathcal{L}_{\text{sim}}$ 双约束解决 binding 问题很 clean。
- Closed-loop 的 motivation 充分，ablation 清楚证明 VPP 和 progressive multi-view 各自的贡献。
- 多材质支持（cloth、sand、snow、liquid、smoke、elastic、rigid）展示性强。

**潜在质疑**：
- Velocity 直接继承是 hack。如果 visual refinement 改变了物体速度（比如让小球弹得更高），这个改动会被丢失。理论上应该从 refined positions 用 finite difference 重新估 velocity。
- $\mathcal{L}_{\text{sim}}$ 用 L2 平均约束，可能限制 visual primitives 表达大幅度形变（比如布料大幅翻折）。论文里没讨论 $\lambda_{\text{sim}}$ 取值范围和 sensitivity。
- 16 分钟一个 window，3 个 window 近 50 分钟。没法做 interactive embodied AI。
- Imaging quality 66.98 不算高（Veo3.1 是 67.82，GEN3C 是 66.25），说明 visual fidelity 牺牲了。
- 失败案例（hockey stick）暴露了 single-image 4D 的根本限制——不能 hallucinate 未见几何。这需要 video model 的 generative 能力补足，但 3D 表示的 rigidity 限制了它。
- 242 views 的 dense view generation 也是 costly 的，GEN3C 推理时间应该不短。
- Table 1 里 Veo3.1 在 3D Consist 上 73.93 已经很强，说明 video model 自己也在学 implicit 3D consistency。如果未来 video model 进化到完全 3D consistent，hybrid simulator 的优势会缩小。

**值得追问的问题**：
- $\mathcal{L}_{\text{sim}}$ 的权重 $\lambda_{\text{sim}}$ 取多少？sensitivity 如何？
- Progressive optimization 中 "smaller control weight" 具体多少？是 heuristic 还是 searched？
- Velocity 重新估计的 ablation 没做。如果用 finite difference 重新估 velocity 会怎样？
- VPP 在多物体碰撞场景下表现如何？论文里没展示多物体交互的复杂场景。
- 把 video model 换成 Sora、Veo 这种更强的模型，结果会有多少提升？还是 VPP 的 binding 约束反而限制了 video model 的发挥？

---

## 相关参考链接

- PerpetualWonder 项目主页：https://johnzhan2023.github.io/PerpetualWonder/
- WonderPlay（前作）：https://github.com/hzxie/PonderV2 或 https://wonderplay-2025.github.io/
- GEN3C：https://research.nvidia.com/labs/toronto-ai/GEN3C/
- CogVideoX：https://github.com/THUDM/CogVideo
- Go-with-the-flow：https://withertoy.github.io/go_with_the_flow/
- Genesis Physics Engine：https://github.com/Genesis-Embodied-AI/Genesis
- 3D Gaussian Splatting：https://github.com/graphdeco-inria/gaussian-splatting
- Gaussian Grouping：https://github.com/lkeab/gaussian-grouping
- SAM2：https://github.com/facebookresearch/sam2
- Hunyuan3D 2.0：https://github.com/Tencent/Hunyuan3D-2
- WorldScore：https://worldscore-3d.github.io/
- PhysDreamer：https://physdreamer.github.io/
- PhysGaussian：https://github.com/yanqinJiang/PhysGaussian

---

总结一句话：**PerpetualWonder 把 hybrid generative simulator 从开环变闭环，关键 trick 是 VPP 让 physics particle 和 visual gaussian 共享同一套 anchor 结构，配合 progressive multi-view optimization 消除 refinement 歧义。** 这是一个 representation design 解决 systemic limitation 的好例子，类似你在 Software 2.0 里讲的——正确的 representation 让 gradient flow 自然走通。
