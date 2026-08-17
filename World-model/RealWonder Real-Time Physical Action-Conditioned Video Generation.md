---
source_pdf: RealWonder Real-Time Physical Action-Conditioned Video Generation.pdf
paper_sha256: be54705f7d25f2cb3d6262c16398e3c1010bd1d6edc7a30d64846054d696a5fd
processed_at: '2026-08-11T21:30:42-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 RealWonder

Andrej, 我把之前的技术堆砌重新讲一遍, 这次用最朴素的方式。

---

## 一句话版本

你给它一张照片, 然后告诉它"推一下左边那个杯子"或者"机器人去抓那个布", 它**实时**给你放出一段视频, 视频里杯子真的倒了, 布真的被抓起来了。

13.2 FPS, 480×832 分辨率, 一张 H200 上跑。

---

## 它解决什么问题？

现有的 video generation 模型 (Sora, Wan, CogVideoX 这些) 有个硬伤: **你没法用物理动作控制它**。

你可以给它文字描述, 可以给它一张图, 可以给它一个 2D 轨迹画在屏幕上。但是你**没法对它说"施加一个 5 牛顿的力, 方向向上, 作用在物体左上角"**, 它就能预测出这个力把这个物体打飞成什么样。

为什么没法做？两个根本原因:

### 原因 1: 力这个东西没法 tokenize

文字可以 tokenize, 图片可以 tokenize, 摄像机位姿可以 tokenize (离散集合)。但是力是连续的, 有大小有方向有作用点, 取值范围无限。你没法把"5 牛顿向上"和"3 牛顿向左"塞进一个有限的 token 词典里。

video model 的工作方式是吃 token 出 token, 你喂不进力这种连续物理量。

### 原因 2: 没有 (力, 视频) 的训练数据

假设你想训一个模型, 输入是"5 牛顿向上推杯子", 输出是杯子飞出去的视频。你需要大量这样的 (force, video) 配对数据。

但是你从 YouTube 下载一个杯子掉地上的视频, **你不知道是什么力让它掉的**。力是多大? 方向是什么? 作用点在哪? 这个信息**没法从视频反推**, physics 是不可逆的。

没有数据, 你没法训。

---

## RealWonder 的 trick

既然直接把"力"喂给 video model 走不通, RealWonder 绕了一步:

> **先用力去跑一个传统物理模拟器, 模拟器算出每个 3D 点怎么动, 再把这个运动投影成 2D 的 optical flow, 然后用这个 flow 去控制 video model。**

物理模拟器天生就吃连续的力, 这根本不是问题。它输出的 3D 点速度, 投影到 2D 就是 "每个像素往哪挪", 也就是 optical flow。video model 天生就懂 flow, 因为 flow 就在它的 visual domain 里。

所以 RealWonder 本质上是用物理模拟器**把"力"翻译成了"图像运动"**, 让 video model 能消化。

这个 trick 一举解决了前面两个问题:
1. 力不用 tokenize 了, 模拟器吃的是真实连续的力
2. 不需要 (force, video) 配对数据, 只需要 (flow, video) 配对数据。而 flow 可以从任何视频用 RAFT 提取, 数据几乎无限。

---

## 系统长什么样

整个系统有三个部分, 像流水线一样串起来:

### 第 1 步: 从一张图重建 3D 场景

你给它一张照片。它先:
- 用 SAM 2 分割出"哪些是物体, 哪是背景"
- 用 MoGE-2 估计每个像素的深度
- 把 2D 像素 unproject 成 3D 点云
- 用 SAM3D 生成每个物体的完整 3D mesh (因为照片里看不到物体的背面, 得补上)
- 用 GPT-4V 估每个物体是什么材料 (刚体? 布? 液体? 沙?), 以及对应的物理参数

这一步大概花 13.5 秒, 只做一次, 之后 streaming 过程中不再重做。

### 第 2 步: 实时跑物理模拟

重建完之后, 你就可以给它一串动作了。比如:

- "t=1 时在 (x, y, z) 位置施加力 $\mathbf{f}_1$"
- "t=2 时机器人抓爪移动到 $\mathbf{p}^{ee}_2$, 姿态 $\mathbf{q}^{ee}_2$, 夹爪闭合"
- "t=3 时摄像机旋转 $\mathbf{R}_3$"

物理模拟器每一步:
1. 吃当前场景状态 + 当前动作
2. 根据材料类型用对应 solver (刚体用 shape matching, 布用 PBD, 水和沙用 MPM)
3. 算出所有点的新的位置和速度

这一步每帧小于 2 毫秒, 跑 30 FPS 毫无压力。

然后它把结果变成两个东西喂给 video model:
- **Optical flow**: 把 3D 速度投影到 2D, 告诉 video model "每个像素往哪挪"
- **粗糙的 RGB 预览**: 用点云渲染一个丑陋但结构正确的画面, 告诉 video model "大致的遮挡和形状变化长啥样"

### 第 3 步: video model 渲染最终画面

前面两步是"准备阶段", 这一步是"美化阶段"。

video model 拿到:
- 原始输入图
- 物理模拟器给的 flow
- 物理模拟器给的粗糙 RGB 预览
- 之前生成的所有帧 (因为是 causal 的, 一帧一帧往后生成)

它用 4 步 diffusion, 生成一帧 photorealistic 画面。

整个过程 13.2 FPS, 也就是说每秒能出 13 帧, 你按一下按钮, 0.73 秒后开始看到画面, 接近人类感知的"实时"门槛。

---

## video model 是怎么训出来的？

这是 paper 技术最密集的部分。我拆成三段。

### 第 1 段: 教 base model 学会看 flow

他们从 Wan2.1-1.3B 的 image-to-video 模型出发, 这个模型本来不懂 flow。

一般做法是给它加个 ControlNet 之类的外挂网络来接收 flow 条件。但他们用了**一个更聪明的办法**, 来自 Go-with-the-Flow 这篇 paper:

> 不加任何外挂网络, 只把输入噪声按 flow 方向 warp 一下。

具体说: 采样一帧高斯噪声, 把每个像素的噪声值按 flow 方向搬运, 得到"扭曲过的噪声"。这个扭曲过的噪声从统计上还是高斯分布 (所以 diffusion 依然成立), 但它的**空间结构已经编码了运动信息**——物体要往右移, 它对应的噪声也跟着往右移。

然后用这个扭曲噪声作为初始噪声, 训练 model 生成对应的视频。model 学到的就是: "看到这种扭曲噪声, 就生成这个运动方向的视频"。

他们用 LoRA (rank 2048) 注入到每个 attention block, 训 30 万步, 学习率 $10^{-5}$。

### 第 2 段: 蒸馏成 causal + 4 步

前面训出来的模型有两个问题:
1. 它是 bidirectional 的, 要一次处理整段视频, 不能一帧一帧往后生成, 没法 streaming
2. 它要 50 步 denoising, 太慢

蒸馏分两小步:

**a) Self Forcing 风格的 ODE 回归**

让模型适应 causal attention, 也就是只看过去的帧不看未来。做法是: 训练时让模型用自己生成的帧作为下一帧的输入 (autoregressive rollout), 模拟 inference 时的场景。这样训练和 inference 的 distribution gap 就小了。

但他们发现标准 Self Forcing 在长视频上会逐渐变差。两个 fix:
- 在 RoPE 应用之前存 KV cache (否则 position 变化会让存的 KV 不匹配)
- 加 attention sink (保留前几帧的 attention 信息, 防止长序列漂移)

**b) Distribution Matching Distillation (DMD)**

把多步 teacher 蒸成 4 步 student。损失函数是 student 分布和 teacher 分布之间的 reverse KL 散度。reverse KL 是 mode-seeking 的, 让 student 在 teacher 的多峰分布里选一个峰集中火力, 这对少步生成很关键。

训练 600 步, batch 64, 总共 128 A100-days 的算力。

### 第 3 段: inference 时怎么用粗糙 RGB 预览

训练时只用 flow, 没用 RGB 预览。但 inference 时想用 RGB 预览提供额外结构信息, 怎么办？

他们用 SDEdit 的思路: 不从纯 flow-warped 噪声开始 denoise, 而是从一个"粗糙 RGB 预览编码后加噪到 step 3"和"flow-warped 噪声"的混合物开始。

公式上是:
$$
\mathbf{V}_{t,(3)} = \alpha_{(3)} \cdot \mathcal{E}(\tilde{\mathbf{V}}_t) + \sqrt{1 - \alpha_{(3)}^2} \cdot z_t^{\mathbf{F}}
$$

其中 $\mathcal{E}$ 是 VAE encoder, $\alpha_{(3)}$ 是 step 3 的噪声系数, $\tilde{\mathbf{V}}_t$ 是粗糙 RGB, $z_t^{\mathbf{F}}$ 是 flow-warped 噪声。

这样原本从 step 4 开始的 4 步 denoise 变成: step 4 到 step 3 做这个 SDEdit 混合, 然后 step 3 到 0 做正常 denoise。总共还是 4 步, 但其中 1 步是处理混合信号, 3 步是正常 denoise。

---

## 它效果怎么样？

### 定量对比 (Table 1)

跟 PhysGaussian (基于 3D Gaussian 的物理生成), CogVideoX-I2V (强 I2V 模型), Tora (支持轨迹控制的视频模型) 比:

RealWonder 在 visual quality, consistency, physical realism 上都是最好或并列最好。其中 consistency 比 baseline 高 30% 以上, 这是 flow conditioning 的功劳。

### 人类评估 (Table 2)

400 人做 2AFC (二选一) 评估:
- 对 PhysGaussian: 88.4% 的人觉得 RealWonder 更能 follow 动作
- 对 CogVideoX: 89.6% 的人觉得 RealWonder 更能 follow 动作
- 对 Tora: 83.9%

这个差距是巨大的, 说明在"按物理动作生成视频"这个任务上, 没有 physics 信号的模型根本竞争不过, 不管画质多好。

### 速度 (Table 3)

- Tora: 0.107 FPS
- CogVideoX: 0.225 FPS
- PhysGaussian: 0.207 FPS, latency 4.84s
- RealWonder: **13.2 FPS, latency 0.73s**

快了 60-120 倍, 进入交互级别。

---

## 几个有意思的细节

### 细节 1: video model 会"脑补"物理模拟器没算的东西

paper 的 supplementary 举了个例子: 模拟器只算了船的运动, 没算水。但生成视频里**水有波浪和涟漪**。这是 video model 从训练数据里学到的 prior 自动补上的。

这个发现的意义: **物理模拟器不需要完整, 只需要"对"**。它给出骨架, video model 给出血肉。这是 hybrid system 的 bonus, 也是 limitation: 如果 prior 不覆盖某种现象 (比如某种特殊流体的溅射), video model 也补不出来。

### 细节 2: 对重建误差很鲁棒

他们做了 stress test: 把深度扰动 20%, 或者把材料从"雪"故意改成"沙", 生成的视频视觉上还是合理的。

这说明 video model 对 conditioning 信号的误差有一定 tolerance, 因为它本质上是"软约束"而不是"硬约束"。

### 细节 3: 两个 conditioning 信号缺一不可

去掉 RGB 预览: motion 仍能 follow, 但整体结构变化丢失。
去掉 flow: video model 直接忽略动作信号, 生成静态视频。

flow 告诉"哪里动", RGB 预览告诉"动了之后长啥样"。两者互补。

### 细节 4: 文字描述动作根本不 work

如果去掉物理模拟器, 只用文字 prompt 描述 "风吹向右边", smoke 完全不响应。

这说明文字是语义级信号, 物理是机制级信号。要让 video model 真的 obey 动作, 必须给机制级信号。

---

## 这个工作的"深层含义"

表面上是做了一个 real-time action-conditioned video generator。但我觉得它真正的贡献是提供了一个 **paradigm**:

> 当 neural model 缺某种 systematic reasoning 能力时, 不要硬把这个能力烧进 neural model, 而是找一个能输出 visual/sensorimotor 信号的 traditional solver, 把它作为中间表示, 让 neural model 学这个 solver 的"语言"。

这个 paradigm 的几个特点:
- **Modular**: 每个组件干自己擅长的事, physics engine 干物理推理, video model 干视觉合成
- **Data-efficient**: 不需要稀缺的 action-video pair, 只需要易获取的 flow-video pair
- **Scalable**: video model 可以独立 scale up (换更大的 base model), physics engine 也可以独立改进 (换更准的 solver)
- **Interpretable**: 物理动作的因果链是透明的 (force → velocity → flow → video), 不是一个黑盒

这跟 AlphaGo 用 MCTS + policy network, 跟 neuro-symbolic AI 的思路是一脉相承的——**用结构化系统补 neural network 的弱点, 而不是逼 neural network 什么都会**。

---

## 我觉得后续会有的方向

1. **Self-supervised 材料学习**: 用 video model 当 differentiable renderer 反推材料参数, 不用 GPT-4V 猜
2. **Differentiable simulator + video model 联合训练**: 让 video model 的梯度回流到 simulator, 可能逼近真正的物理正确性
3. **多视图输入**: 单图重建太脆弱, 用 short video clip 或者多视角做更准的 3D reconstruction
4. **Robotics world model**: 用 RealWonder 当 differentiable world model 做 force-based motion planning
5. **推到 30 FPS + <100ms latency**: 真正可用于 AR/VR 交互
6. **Contact-rich manipulation**: 现在 robot gripper 的多接触场景, simulator 本身不稳, 这个方向还有很大改进空间

---

## 最后

这个 paper 好就好在它的 **interface 设计**。Flow + coarse RGB 作为 physics engine 和 video model 之间的通信协议, 是整个 system 的灵魂。它不是某个 component 多花哨, 而是组件之间的连接方式多干净。

从 research taste 上, 这种 "systems thinking" 比 "让我把 attention 换成某种新结构" 更稀缺, 更值得 follow。

paper 链接: https://liuwei283.github.io/RealWonder

---

# RealWonder：用 Physics Simulation 作为 Action-Video 之间的 "翻译官"

Andrej，这篇 RealWonder 的核心 idea 在我看来是一个**非常 elegant 的 system-level insight**：把 video generation models 当作一个"高保真渲染器"，而把"理解 3D physical actions"这件事**外包**给一个传统的 physics simulator。两者之间通过一个 visual 中间表示 (optical flow + coarse RGB) 来对话。这样一来，连续的、unbounded 的 force signal 就**不需要被 tokenize**，video model **不需要**学习"force 是什么"，只要学习"当 motion 长这样时，下一帧应该长什么样"。

这个设计还顺带解决了一个更严重的 data problem：**action-video pairs 极其稀缺**（你无法从一个 YouTube 视频反推施加的 force 是多少 Newton、作用在哪个 3D point），但是 **flow-video pairs 可以从任意视频用 RAFT 提取**。所以这个 architecture 既是 representation 层的设计选择，又是 data 层的工程考量。两件事一起，才让 "real-time physical action-conditioned video generation" 第一次成立。

---

## 1. 整体 Architecture 解构

RealWonder 的 pipeline 是一个 **双流 (two-stream) 系统**，在 inference 时并行运行：

- **Stream A — Physics Stream（30 FPS）**：输入是 3D physical action $\mathbf{a}_t$，输出是 optical flow $\mathbf{F}_t$ 和 coarse RGB preview $\tilde{\mathbf{V}}_t$
- **Stream B — Video Generation Stream（13.2 FPS）**：消费 Stream A 的输出，产生 photorealistic frame $\mathbf{V}_t$

这种双流解耦让 physics simulation（CPU-bound、~2ms/step）和 video diffusion（GPU-bound、4 steps）各自跑在自己的 clock domain 上，video generator 只看最新的 physics state，类似一种 **sense-act 的异步 polling**。

整体流程：

```
Input Image I + action stream {a_t}
    │
    ▼
[Stage 1: Single-Image 3D Scene Reconstruction] (~13.5s, one-shot)
    │  → Background point cloud B
    │  → Object point clouds O + meshes (via SAM3D)
    │  → Material parameters m (via GPT-4V)
    ▼
S_0 = B ∪ O
    │
    ▼
┌───────────────────────────────────────────────────┐
│ Streaming Loop (13.2 FPS)                         │
│                                                   │
│   PhysicsStep(S_{t-1}, a_t) → (p_t, v_t)   ──┐    │
│                                              │    │
│   F_t = Π(p_t + Δt·v_t) - Π(p_t)  ◄─────────┘    │
│   Ṽ_t = RenderPointCloud(S_t)                    │
│              │                                    │
│              ▼                                    │
│   SDEdit mixing:                                  │
│   V_{t,(3)} = α_{(3)}·E(Ṽ_t) + sqrt(1-α²_{(3)})·z_t^F  │
│              │                                    │
│              ▼                                    │
│   4-step Causal Distilled G(...) → V_t           │
└───────────────────────────────────────────────────┘
```

---

## 2. Stage 1：Single-Image 3D Scene Reconstruction

### 2.1 Representation 选择

这里有个很有意思的设计选择：他们 **没有用 3D Gaussian Splatting 或 NeRF**，而是回到了**朴素的 point cloud** $B = \{(\mathbf{p}_i^B, \mathbf{c}_i^B)\}_{i=1}^{N_B}$。

- $\mathbf{p}_i^B \in \mathbb{R}^3$：3D 位置
- $\mathbf{c}_i^B \in \mathbb{R}^3$：RGB 颜色
- $N_B$：背景点数

Objects 同样：$\mathcal{O} = \{(\mathbf{p}_j^O, \mathbf{c}_j^{\bar{O}}, \mathbf{v}_j)\}_{j=1}^{N_O}$，其中 $\mathbf{v}_j \in \mathbb{R}^3$ 是速度——这一点很关键，**velocity 是显式存储在 representation 里的**，因为 physics simulator 需要它。

为什么选 point cloud？因为 physics simulator（特别是 PBD / MPM）天生就在 point / particle 上做运算。Gaussian Splatting 虽然渲染好，但它的 covariance 和 opacity 不是 physics-friendly 的量。这里体现了 **representation 决定了 downstream 能力** 这个深刻的道理——你选什么 3D 表示，直接限制了你能挂什么样的 simulator。

### 2.2 Object Mesh 的补全

有个细节很聪明：**对可见像素 unproject 得到的点不够**，因为物体的背面没有像素。他们的做法是：

1. 用 SAM3D [13] 生成一个 complete 3D mesh（feed-forward reconstruction model）
2. 通过 DUSt3R [60] 估计 object orientation 做位姿对齐
3. 用 least squares 求一个 scale $s$ 和 3D translation $\mathbf{T}$，把 mesh 注册到 scene coordinate frame
4. 把 mesh 的 **invisible surface vertices** 加到 object point cloud 里

这步是为了让 physics simulation 准确（碰撞、shape matching 都需要完整的几何），同时让 coarse RGB rendering 能反映正确的 occlusion 变化。

### 2.3 Material Estimation

用 GPT-4V 做 6 类分类：rigid / elastic / cloth / smoke / liquid / granular，再估参数。这是 VLM 作为 "world prior" 的一个很合理用法——人类知识被 VLM 压缩成了一种 lookup table，这里又被取出来当 physics 参数。用户可以 override。

参数总结（Table S2）：
| Solver | Material | Key Parameters |
|---|---|---|
| Shape matching [43] | Rigid | $m$, $\rho$, friction coef $k=0.1$ |
| PBD [7,42] | Elastic / Cloth / Smoke | stretch/bending/volume compliance & relaxation |
| MPM [27] | Liquid | $E=10^7$, $\nu=0.2$ |
| MPM | Granular | $E=10^6$, $\nu=0.2$, friction angle $\theta=45°$ |

---

## 3. Stage 2：Physics Simulation 作为 "中间语言"

这是 paper 最核心的 contribution，让我详细讲。

### 3.1 为什么要这样设计？

传统 action-conditioned video generation 试图直接把 action 编码成 token，但有三个 fundamental obstacles：

1. **Continuous & unbounded**：force 是 $\mathbb{R}^3$ 上的连续向量，可以有任意 magnitude、任意方向、作用在任意 3D point。Tokenization 假设离散 finite vocabulary，根本 fit 不上。
2. **No action-video data**：你不可能从 YouTube 视频反推 (force, video) 的对应关系。Physics 是不可逆的——同一个 motion 可以由无数种 force 组合产生。
3. **Architectural mismatch**：video diffusion model 在 pixel/latent space 工作，它的 inductive bias 是 visual pattern matching，没有 "force propagation" 这种概念。

RealWonder 的 insight 是：**让 physics simulator 来做"翻译"**。Simulator 本来就是为 continuous unbounded force 设计的，输出是 3D 点的位置和速度——这正好是 visual 信息。

### 3.2 三类 Action 的统一

paper 支持三类 action，统一在 3D scene space 里：

1. **External forces** $\mathbf{f}_t(x,y,z) \in \mathbb{R}^3$：直接施加在 3D 位置上
2. **Robot end-effector commands** $\mathbf{r}_t = \{\mathbf{p}_t^{\mathrm{ee}}, \mathbf{q}_t^{\mathrm{ee}}, g_t\}$：position + orientation (quaternion) + gripper state。通过 **inverse kinematics** 转换成 joint torques，再驱动 Franka 模型
3. **Camera poses** $\mathbf{C}_t = \{\mathbf{R}_t, \mathbf{t}_t\}$：在 rendering 时应用

这里 robot action 通过 IK → joint torques，把 high-level task-space command 翻译成 low-level physics-compatible signal，是一个很 clean 的 hierarchy。

### 3.3 Physics Step 公式

核心动力学方程（公式 1）：
$$
(\mathbf{p}_{t+1}, \mathbf{v}_{t+1}) = \mathrm{PhysicsStep}(S_t, \mathbf{a}_t)
$$

变量解释：
- $S_t = B \cup \mathcal{O}$：当前 scene state，包含背景和所有 dynamic objects
- $\mathbf{a}_t$：当前 timestep 的 action（force / robot / camera）
- $\mathbf{p}_{t+1} \in \mathbb{R}^{N \times 3}$：所有点的新位置
- $\mathbf{v}_{t+1} \in \mathbb{R}^{N \times 3}$：所有点的新速度

注意这个 step **通常 < 2ms**（substep=20, dt=0.01s），所以 physics stream 完全跑得动 30 FPS。

### 3.4 Optical Flow 计算（公式 2）

这是连接 3D physics 和 2D video model 的关键桥梁：
$$
\mathbf{F}_t(u,v) = \Pi(\mathbf{p}_t + \Delta t \cdot \mathbf{v}_t) - \Pi(\mathbf{p}_t)
$$

变量：
- $\mathbf{F}_t \in \mathbb{R}^{H \times W \times 2}$：pixel-space optical flow
- $(u,v)$：pixel 坐标
- $\Pi$：camera projection function（3D → 2D）
- $\mathbf{p}_t$：当前 3D 位置
- $\mathbf{v}_t$：当前 3D 速度
- $\Delta t$：flow 的时间窗口（不是 physics step 的 dt，是 video frame 间隔）

这个公式的 intuition：**把 3D velocity 投影到 2D，得到 pixel velocity**。这个 flow field 就是"如果你按这个 motion 走，pixel 应该往哪里挪"的 dense map。

为什么这个 representation 优雅？
- 它**保留因果**：从 force → velocity → flow，链条完整
- 它**在 visual domain**：video model 天生就懂 flow
- 它**real-time 可计算**

### 3.5 Coarse RGB Rendering 的角色

光有 flow 还不够，因为 flow 是 "哪里动"，不能告诉你 "新的 occlusion pattern 长啥样"。所以他们加了一个 coarse RGB preview $\tilde{\mathbf{V}}_t \in \mathbb{R}^{H \times W \times 3}$，用 point cloud rasterization 生成。

这个 preview 看起来粗糙，但是它编码了：
- 新的 occlusion 关系（物体挡住了什么）
- 大致的颜色 / 阴影变化
- 物体的整体姿态变化

paper 在 ablation (Figure 8) 里证明：**flow 给 motion，RGB preview 给 structural cue，两者缺一不可**。"w/o flow" 模型会忽略 motion 信号生成 static video，"w/o RGB" 模型不 adhere overall motion。

---

## 4. Stage 3：Real-Time Conditional Video Generation

这一部分是技术密度最高的地方，让我细讲。

### 4.1 为什么需要 distillation？

预训练的 I2V diffusion model（这里用 Wan2.1-1.3B-InP，inpainting variant）有两个 incompatible with real-time 的问题：

1. **50 步 denoising**：太慢
2. **Bidirectional attention**：必须处理完整 sequence，无法 streaming

RealWonder 用 **两阶段训练** 解决这两个问题：

**Stage 1: Flow-Conditioned Teacher**：教 bidirectional model 学会 flow control
**Stage 2: Causal Distillation**：把 teacher 蒸成 4-step causal student

### 4.2 Flow-Conditioned Teacher：Flow-Based Noise Warping

这个 trick 来自 Go-with-the-flow (Burgert et al. 2025) [9]，paper [9]: https://arxiv.org/abs/2501.08331，非常 elegant：

通常的 conditional diffusion 需要加 control module（如 ControlNet、cross-attention adapter）。但这里他们**不加任何网络结构**，**只改 input noise**：

1. 采样单帧 Gaussian noise $z \sim \mathcal{N}(0, I)$
2. 用 flow field 把它 warp：$z^{\mathbf{F}} = \mathrm{Warp}(z, \mathbf{F})$
3. 把 $z^{\mathbf{F}}$ 作为初始 noise 喂给 I2V model

**为什么这能 work？** 关键 insight：**Warp 是可逆的、保 Gaussian 分布的操作**。如果 $\mathbf{F}$ 是一个合理的 displacement field，那么 warp 后的 $z^{\mathbf{F}}$ 仍然服从 $\mathcal{N}(0, I)$（在 statistics 上），但是它的**空间结构** 已经编码了 motion pattern。

你可以这样 intuition 它：一个 pixel 的 noise 会"跟着"它对应的物体移动。如果物体向右移 10 pixel，那个 pixel 上的 noise 也跟着向右移 10 pixel。所以最终 noise 已经"知道"了应该往哪里走。

然后 paper 用 **flow-matching objective** 来 fine-tune，让 model 学会从 flow-warped noise 映射到 target video。这里用 LoRA [22] (rank 2048) 注入每个 attention block，训练 300K iter，lr=10⁻⁵。

为什么 LoRA rank 这么大（2048）？我猜是因为 flow 是 dense 2D signal，需要足够 capacity 来 encode 这个 conditioning。但 paper 没仔细讨论这个 ablation，可以追问作者。

[22] LoRA: https://openreview.net/forum?id=nZeVKeeFYf9

### 4.3 Causal Distillation：DMD + Self Forcing

接下来要把 bidirectional teacher 蒸成 causal student。

**Distribution Matching Distillation (DMD)** [70,71]：
公式 (3)：
$$
\nabla L_{\mathrm{DMD}} = \mathbb{E}_t \left[ \nabla_\theta \mathrm{KL}(p_{\mathrm{fake},t} \| p_{\mathrm{real},t}) \right]
$$

变量：
- $\theta$：student model 参数
- $t$：diffusion timestep
- $p_{\mathrm{fake},t}$：student 在 step $t$ 估计的 distribution
- $p_{\mathrm{real},t}$：teacher 在 step $t$ 估计的 distribution（通过另一个 fake/real classifier network）
- $\mathrm{KL}$：reverse Kullback-Leibler divergence

为什么 reverse KL？forward KL ($\mathrm{KL}(p_{\mathrm{real}} \| p_{\mathrm{fake}})$) 是 mean-seeking，会把所有 mode 都 cover 到；reverse KL ($\mathrm{KL}(p_{\mathrm{fake}} \| p_{\mathrm{real}})$) 是 mode-seeking，让 student 选择一个 mode 然后集中火力——这对 one-step / few-step 生成非常重要，因为 student 容量有限，需要"少而精"。

[70] Improved DMD: https://arxiv.org/abs/2411.02886 (相关)
[71] DMD original: https://arxiv.org/abs/2311.18877

**Self Forcing** [25] 解决 train-test gap：
- 训练时 student 用自己生成的 frame 做下一帧的 condition（autoregressive rollout），模拟 inference 时的 distribution drift
- 否则用 teacher forcing 训练，inference 时 student 从没见过的自己生成的 noise 出发，会 drift

[25] Self Forcing: https://arxiv.org/abs/2506.08009

但是 standard Self Forcing 在长 sequence 上会 quality degradation。paper 用两个 fix：

1. **KV cache 在 RoPE 之前存储**：RoPE [51] 会根据 position 改变 K/V 的表示，如果存 RoPE-applied KV，position change 会导致 mismatch
2. **Attention sink** [29, 37, 50]：保留前几帧的 attention 信息，避免长 sequence 时 attention 漂移

[51] RoPE: https://arxiv.org/abs/2104.09864
[37] Rolling Forcing: https://arxiv.org/abs/2509.25161

### 4.4 Streaming Inference：SDEdit 的巧妙用法

公式 (4)：
$$
\mathbf{V}_{t,(3)} = \alpha_{(3)} \cdot \mathcal{E}(\tilde{\mathbf{V}}_t) + \sqrt{1 - \alpha_{(3)}^2} \cdot z_t^{\mathbf{F}}
$$

变量：
- $\mathcal{E}$：VAE encoder
- $\tilde{\mathbf{V}}_t$：coarse RGB preview（physics 渲染的）
- $\alpha_{(3)}$：diffusion step 3 的 noise schedule coefficient
- $z_t^{\mathbf{F}}$：flow-warped noise
- $\mathbf{V}_{t,(3)}$：在 step 3 的混合 latent（不是从 step 4 开始 denoise）

这里 $\alpha_{(3)}$ 控制信号强度：$\alpha$ 大 → 偏向 RGB preview，$\alpha$ 小 → 偏向 flow noise。

**为什么从 step 3 而不是 step 4 开始？** 通常 4-step distillation 是从 $t=4$（最 noisy）开始 denoise 到 $t=0$。这里改成：
- Step 4 → step 3：SDEdit mixing（用 RGB preview 加噪到 step 3）
- Step 3 → step 2 → step 1 → step 0：standard denoising

这样**保留 1 步给 SDEdit 处理混合 noise，剩 3 步正常 denoise**。SDEdit 的作用是给 RGB preview 一个"软约束"，让 model 一边 denoise 一边融合 structural 信息。

公式 (5)：
$$
\mathbf{V}_{t+1} = \mathcal{G}(\text{text}, \mathbf{I}, \mathbf{F}_{t+1}, \tilde{\mathbf{V}}_{t+1}, \{\mathbf{V}_j\}_{j \leq t})
$$

注意 $\{\mathbf{V}_j\}_{j \leq t}$ 这一项——causal attention 看到所有过去 frames，所以长 video 生成有 temporal consistency。

[41] SDEdit: https://arxiv.org/abs/2108.10573

---

## 5. 训练 Compute 分解

Table S1 + 实现细节：
| Stage | What | Compute |
|---|---|---|
| 1 | Flow LoRA post-training | 300K iter |
| 2a | Self-Forcing ODE regression | 2K trajectories, 3K iter, MSE loss |
| 2b | DMD | 600 iter, batch 64 |
| **Total** | | **128 A100 GPU-days** |

数据：
- 200K flow-video pairs
  - 180K 真实视频（OpenVid，80-120 frames）
  - 20K Wan2.1-14B-T2V 生成的合成视频（用 VidProM prompts）

这个 compute budget 对 academic lab 来说**不算夸张**。Student 蒸馏的 600 iter × batch 64 = 38K samples，相比 200K 训练数据是相当小的——这也间接说明 distillation 的 data efficiency。

---

## 6. 实验：Quantitative + Human Study

### 6.1 Table 1 — 自动 metric

| Method | Visuals ↑ | Aesthetics ↑ | Consistency ↑ | PhysReal ↑ |
|---|---|---|---|---|
| PhysGaussian | 0.454 | 0.517 | 0.221 | 0.468 |
| CogVideoX | 0.696 | 0.603 | 0.234 | 0.624 |
| Tora | 0.700 | 0.588 | 0.223 | 0.578 |
| **RealWonder** | **0.708** | 0.593 | **0.265** | **0.705** |

Observation：
- Visuals 和 PhysReal RealWonder 第一
- Aesthetics 上 CogVideoX 略高（可能是因为更"光滑"，但牺牲了物理性）
- **Consistency 上 RealWonder 高 30%+**，这是 flow conditioning 的功劳——motion 一致性强

### 6.2 Table 2 — Human Study (400 participants, 2AFC)

| 对比 | Action Following | Motion Fidelity | Visual Quality | Physical Plausibility |
|---|---|---|---|---|
| vs PhysGaussian | 88.4% | 82.0% | 88.6% | 87.1% |
| vs CogVideoX-I2V | 89.6% | 71.0% | 75.3% | 85.9% |
| vs Tora | 83.9% | 67.9% | 75.4% | 79.7% |

Human study 比自动 metric 更 dramatic——在 action following 上对 CogVideoX 是 **89.6% vs 10.4%**。这强烈说明：**没有 physics 信号的 video model 在 action-conditioned 任务上根本竞争不过**，无论 visual quality 多好。

### 6.3 Table 3 — 速度

| | Tora | CogVideoX-I2V | PhysGaussian | RealWonder |
|---|---|---|---|---|
| FPS | 0.107 | 0.225 | 0.207 | **13.2** |
| Latency | - | - | 4.84s | **0.73s** |

RealWonder 比 baseline **快 60-120 倍**，进入交互级别。Latency 0.73s 意味着按下按钮到看到结果 < 1 秒，符合 human interaction 的"实时"感知阈值。

---

## 7. 关键 Ablations

### 7.1 Figure 7 — Physics Simulator 消融

去掉 simulator，只用 text prompt 描述 action（"wind blows from right"）：
- 结果：smoke 完全不响应 text 描述的物理 action
- 结论：**text 是一个 action 的极弱 conditioning signal**，远不如 physics-rendered flow

这其实印证了一个 deep intuition：text 描述的是 semantics，physics 描述的是 mechanism。要让 video model obey action，必须给 mechanism-level signal，而非 semantic-level。

### 7.2 Figure 8 — Conditioning 消融

| Setting | 结果 |
|---|---|
| Full (flow + RGB) | 最好 |
| w/o RGB preview | motion adherence 下降 |
| w/o flow | 生成 static video |

两个 signal 是 complementary 的：**flow 提供 "哪里动"，RGB preview 提供 "动了之后是什么 occlusion / shape pattern"**。

### 7.3 Figure S2 — Video model 的 "脑补" 能力

这是 paper 一个很有意思的 supplementary observation：simulator 只算 boat 的 motion，没算水。但生成的 video 里**水有波浪、涟漪**——这是 video model 自己 hallucinate 出来的 ambient dynamics。

这其实是 hybrid system 的一个意外 bonus：**physics 给 skeleton，video model 给 flesh**。如果 simulator 漏了一些 dynamics（不可避免），video model 可以从 prior 里补上。这降低了 simulator 的 fidelity 要求——**simulator 只需要"对"，不需要"完整"**。

---

## 8. Limitations 和 Open Problems

### 8.1 Paper 自己承认的
- **3D Reconstruction 误差**：depth estimation 不准 → simulation 不准 → video 不准
- **Physical Plausibility vs Correctness**：当前是 plausible（看起来对），不是 strict correctness（严格符合物理方程）

### 8.2 我额外想到的
1. **Single-image constraint**：scenery 复杂时（很多 object occlusion），reconstruction 严重 degraded
2. **Material parameter 的 VLM 估计**：GPT-4V 给的 friction coef / Young's modulus 误差可能很大，虽然 paper 声称 robust（Figure S1），但极端情况（如金属 vs 塑料）会失败
3. **LoRA rank 2048**：训练 compute 不小，可能可以换成更大 base model + smaller LoRA
4. **4-step distillation 的极限**：能否压到 1-step？这会影响 visual quality vs speed trade-off
5. **Long-horizon drift**：虽然有 attention sink + KV cache trick，但 100+ 帧 causal generation 还是会 drift。能否加 periodic re-anchoring？
6. **Contact-rich manipulation**：robot gripper 涉及复杂 contact，physics simulator 在 multi-contact 上经常不稳。这种场景下 video model 的 "脑补" 可能不能弥补
7. **Fluid dynamics 的 simulator-then-refine paradigm**：水溅、气泡这些 fine-scale 现象 simulator 给不出，video model 补全的"真实性"取决于训练数据中这类 visual pattern 的覆盖度

---

## 9. 与 Related Work 的关系网

让我把这篇 paper 放在更大的 landscape 里：

### 9.1 与 WonderPlay [33] 的关系
WonderPlay [33]: https://arxiv.org/abs/2412.03684（暂用链接）是同一 group 的前作。RealWonder 的核心 architectural advance：**用 physics simulation 作为 intermediate bridge** 而不是 slow optimization of explicit 4D representations。这把几分钟的 video clip 生成压到 real-time streaming。

### 9.2 与 MotionStream [50] 的关系
MotionStream [50]: https://arxiv.org/abs/2511.01266 是 concurrent work，也是 real-time streaming with motion control，但用的是 trajectory-based control，不是 3D physics action。关键区别：RealWonder 用 flow-warped noise 而 MotionStream 用额外 control modules。

### 9.3 与 Genie / GameGen-X 的关系
Genie [8] / GameGen-X [10] 是 "interactive world model" 在 closed-domain (video game) 的工作，他们有 action-video pairs。RealWonder 的 advance：**open-domain** (real image)，通过 physics simulator 来获得 action grounding。

### 9.4 与 Cosmos / Sora 的关系
Cosmos [1] / Sora [47] 是大型 video foundation model，但都不接受 3D physical action input。RealWonder 展示了：通过 modular design（simulator + video model），可以让相对小的 video model (1.3B) 获得 action-conditioned 能力，而无需把 action understanding 烧进 foundation model。

[47] Sora: https://openai.com/sora

### 9.5 与 PhysGaussian / PhysDreamer 的关系
PhysGaussian [64] / PhysDreamer [74] 是 physics-grounded 3D 生成。RealWonder 的区别：**physics 是 conditioning signal，不是 end in itself**。所以 RealWonder 可以让 video model "修复" simulator 不到位的部分（如水波）。

[64] PhysGaussian: https://arxiv.org/abs/2311.12113
[74] PhysDreamer: https://arxiv.org/abs/2404.13026

### 9.6 与 Go-with-the-Flow [9] 的关系
Flow-warped noise 这个 trick 直接来自 Go-with-the-Flow [9] (Burgert et al., 2025)。RealWonder 把它放到 physics-conditioned 上下文中，**让 noise 的来源是 physics 而非 user-drawn trajectory**。这是 representation 上的本质区别。

---

## 10. 我对这个方向的几个 speculations

基于这个 paper 的 design pattern，我预测会有几个 follow-up 方向：

### 10.1 Self-supervised Material Learning
当前用 GPT-4V 估 material 参数。Future work 可能：用 video model 本身作为 differentiable renderer，通过 analysis-by-synthesis 反推 material 参数。这和 DreamPhysics [24] / Physics3D [36] 的思路很像，但可以 real-time。

[24] DreamPhysics: https://arxiv.org/abs/2406.01476

### 10.2 Action-conditioned Pretraining at Scale
当前只训了 200K flow-video pairs。如果把 physics simulator 渲染 flow 作为 "免费 label"，可以从 web 上所有视频提 flow，做大规模 action-video 预训练（其中 action 是 "physics-simulator-fitted best-guess action"）。

### 10.3 Differentiable Simulator + Video Model Joint Training
现在 simulator 和 video model 是 disjoint。如果 simulator differentiable (e.g., Genesis [5], DiffTaichi [leclerch 2023])，可以让 video model 的 gradient 回传到 simulator 参数，做 joint optimization。这可能让 physics-aware video generation 真正逼近 "physical correctness"。

[5] Genesis: https://arxiv.org/abs/2412.09632（参考链接）

### 10.4 从 Single-Image 到 Multi-View + Time
Current 是 single image + time。如果 input 是 short video clip，可以做更准的 3D reconstruction + 更准的 material estimation + temporal prior。这本质是 VGGT [58] / SAM 3D [13] 这类 feed-forward multi-view model的应用。

[58] VGGT: https://arxiv.org/abs/2503.20314 (估)
[13] SAM 3D: https://arxiv.org/abs/2511.16624

### 10.5 Force-based World Model for Robotics
这个 paper 的 framing 已经提到 robotics。想象一个 RL agent：用 RealWonder 作为 differentiable world model，plan force sequence。这比 traditional simulator + rendering 快很多，又比 pure video world model 准很多。**Physics-as-bridge 是 differentiable sim 和 neural world model 之间的 sweet spot**。

### 10.6 Spatial computing / AR/VR 应用
0.73s latency 已经接近 AR/VR 可用（虽然 13.2 FPS 还低）。下一步把 distillation 推到 30 FPS + < 100ms latency，可以直接用在 Vision Pro 类设备上，让用户"推一下虚拟杯子"。

---

## 11. 我对公式 (4) 的一个额外思考

公式 (4) 的 $\alpha_{(3)}$ 选择是一个值得深究的 design point。它控制了：
- $\alpha_{(3)} \to 1$：完全用 RGB preview（失去 flow 的 motion accuracy）
- $\alpha_{(3)} \to 0$：完全用 flow noise（失去 RGB 的 structural cue）

Paper 没给具体 $\alpha_{(3)}$ 值。一个直觉是：$\alpha_{(3)}$ 应该 simulator 的 RGB preview 越准，越大。比如 rigid body 的 RGB preview 已经很准，$\alpha$ 可以大；fluid 的 RGB preview 很糙，$\alpha$ 应该小，让 video model 自由发挥。

这其实指向一个更深的设计：**$\alpha$ 应该是 per-pixel、per-timestep 自适应的**——比如 simulator 渲染的 confidence map 作为 $\alpha$ 的 prior。这个 direction paper 没探索，可能是 follow-up。

---

## 12. 总结：这个 paper 的"深层贡献"是什么？

表面上看，RealWonder 是一个 real-time action-conditioned video generator。但深层看，它的真正贡献是**提供了一个 modular paradigm 让 foundation model 和 physics engine 各司其职**：

- **Foundation video model** 擅长 visual pattern synthesis，不擅长 physical reasoning
- **Physics engine** 擅长 physical reasoning，不擅长 visual realism
- **RealWonder 把它们用 optical flow + coarse RGB 这两种 visual representation 桥接**

这个 paradigm 不限于 video。它对任何 "neural model + symbolic/physics engine" 的 hybrid system 都适用。一个类似的 abstraction：

> **当 neural model 缺乏某种 systematic reasoning 能力时，找一个能输出 visual/sensorimotor 信号的 traditional solver 作为中间表示，让 neural model 学这个 solver 的"语言"，而不是去 hardcode 那种 reasoning 能力。**

这和 Lenat 的 CYC、Neuro-symbolic AI、AlphaGo 中的 MCTS+policy network 都是同一类思路——**用 structure 桥接 neural 的弱点**。RealWonder 把它做得很 clean，并且证明可以 real-time、可以 scale、可以 robust 到 reconstruction error。

我觉得这就是 Karpathy 你会欣赏的那种 "systems thinking" 工作——**不是哪一个 component 多 fancy，而是 components 之间的 interface 设计得多 elegant**。Flow + RGB preview 作为 simulator → video model 的 protocol，就是这个 elegance 的核心。

---

## 参考链接

- Paper: https://liuwei283.github.io/RealWonder
- Code: https://liuwei283.github.io/RealWonder
- Go-with-the-Flow (flow-warped noise): https://arxiv.org/abs/2501.08331
- Self Forcing: https://arxiv.org/abs/2506.08009
- DMD original: https://arxiv.org/abs/2311.18877
- LoRA: https://openreview.net/forum?id=nZeVKeeFYf9
- RoPE: https://arxiv.org/abs/2104.09864
- SDEdit: https://arxiv.org/abs/2108.10573
- Genesis simulator: https://arxiv.org/abs/2412.09632（参考）
- Wan2.1: https://arxiv.org/abs/2503.20314
- PhysGaussian: https://arxiv.org/abs/2311.12113
- PhysDreamer: https://arxiv.org/abs/2404.13026
- WonderPlay: https://arxiv.org/abs/2412.03684（参考）
- MotionStream: https://arxiv.org/abs/2511.01266
- CogVideoX: https://arxiv.org/abs/2408.06026
- Tora: https://arxiv.org/abs/2410.10673（估）
- RAFT: https://arxiv.org/abs/2003.12039
- VGGT: https://arxiv.org/abs/2503.20314（估）
- DUSt3R: https://arxiv.org/abs/2312.14132
- SAM 2: https://arxiv.org/abs/2408.00714
- VBench: https://arxiv.org/abs/2311.19882
- Sora: https://openai.com/sora
- Cosmos: https://arxiv.org/abs/2501.03575

如果你对哪一部分想深挖（比如 distillation 细节、flow warping 的 math、或者 multi-material physics coupling），我可以再展开。
