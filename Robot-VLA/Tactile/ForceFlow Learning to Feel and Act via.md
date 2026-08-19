---
source_pdf: ForceFlow Learning to Feel and Act via.pdf
paper_sha256: 30830360460ba69dd426a01b8ed570480818812a1d95f37c8bc60ef3060dd918
processed_at: '2026-08-18T22:36:03-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 ForceFlow

## 一句话版本

机器人插 USB、擦花瓶、按弹簧瓶这些活儿，光靠眼睛看是不行的，得有"手感"。ForceFlow 就是给机器人装了个"手感系统"，让它在碰东西的时候能感觉到力，而不是像个瞎子一样硬怼。

---

## 问题到底出在哪

想象你闭着眼睛插 USB。你看得到大概位置，但插不进去的时候，你是靠手感知道"嗯，撞墙了，往左偏一点试试"。这个"撞墙的力"就是 force/torque feedback。

现在的 imitation learning 方法（Diffusion Policy、$\pi_{0.5}$、ACT、OpenVLA）全是 **视觉中心主义**。它们把 RGB 图片当主食，force/torque 信号当配菜。问题是：

**一张图片 23 万维度，force 信号 6 个数字。** 在 end-to-end 训练里，gradient 流回去的时候，network 发现"靠视觉 reconstruct action 更省事"，于是 force pathway 逐渐被边缘化。这在 multimodal learning 里叫 **modality masking** —— 模态之间互相竞争 capacity，弱的那方被挤掉。

结果就是：机器人能看到 USB 孔在哪，但不知道自己怼没怼上，更不知道该不该调整。所以插 USB 失败、擦曲面打滑、盖章要么力气太小要么砸坏纸。

---

## ForceFlow 怎么解决：两个层次的"解耦"

### 第一层：把任务拆成两半（V2F Handover）

人干活的自然流程：
1. **先看**：眼睛找到 USB 孔在桌上哪个位置
2. **后摸**：手伸过去之后，控制权交给触觉，开始"感觉着插"

ForceFlow 把这个流程 explicit 化：

- **Approach Stage**：让 VLM（本质是个会看图说话的语言模型）负责"在全局视角里找到 target 的 pixel 坐标"，然后用 depth 信息反投影成 3D 空间点，motion planner 把机械臂导航过去
- **Interaction Stage**：到达后，ForceFlow policy 接管，只做 local 的 force 闭环控制

为什么要拆？因为 **找位置** 和 **调节力** 是两个完全不同的能力。找位置靠 semantic reasoning（"USB 长这样，孔在桌上那个角"），调节力靠 high-frequency feedback control（"撞到了，扭矩变大，往左挪 0.5mm"）。把它们塞进同一个 end-to-end network，两个能力会互相拖累。

Table 9 是个铁证：给所有 baseline 也装上 V2F 模块，结果 baseline 依然失败。这说明 V2F 不是"作弊帮 ForceFlow 导航"，而是真正解开了一个 bottleneck —— contact regulation 本身的难度。

### 第二层：架构里给 force 开 VIP 通道（Asymmetric Fusion）

这是 paper 最核心的 contribution。Observation 有四个：
- 双视角 RGB 图片（arm camera + fixed camera）
- Proprioception（机械臂自己的 pose，7 维）
- Force/torque 历史（10 步 × 6 维 = 60 维）

ForceFlow 把这四个 observation 不对称地分到两条 pathway：

**Force 走 VIP 通道（AdaLN）**

把 force history 和 proprio 拼成一个 vector，通过 **Adaptive Layer Normalization** 注入到 DiT 的每一层。AdaLN 的公式：
$$\text{AdaLN}(h, c_{\text{vec}}) = \gamma(c_{\text{vec}}) \odot \text{LayerNorm}(h) + \beta(c_{\text{vec}})$$

说人话：force 信号被用来生成一对 scale $\gamma$ 和 shift $\beta$ 参数，直接修改每一层的 feature 分布的均值和方差。

这意味着什么？**force 信号在每一层都会重塑整个 feature 的统计特性**，它不会被 cross-attention 的 softmax 稀释，不会被 concatenate 后的 MLP 淹没。它是 **全局 regulatory constraint**，强制参与每一层的计算。

类比：普通 fusion 像"force 在会议最后举一下手表决"，AdaLN 像"force 是会议主席，每个议题开场都要先表态"。

**Vision 走普通通道（Cross-Attention）**

视觉 feature 编码成 sequence，通过 cross-attention 让 action token 主动去"查询"视觉信息：
$$\text{MHCA}(h, c_{\text{seq}}) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

说人话：action token 是 query，visual feature 是 key-value 库，network 自己决定要从哪个 visual token "读取"信息。视觉信息是 **可选的参考库**，不强制全局参与。

这种不对称设计妙在哪？**force 决定"全局物理状态"，vision 决定"局部空间参考"**。两个 modality 各司其职，不会互相挤占 capacity。

Ablation Table 7 验证了这点：在 interaction stage 完全 mask 视觉，Stamp 和 Clean Whiteboard 几乎不掉（85→80, 100→90），因为单轴压力调节 force 就够；但 Plug、Insert、Clean Vase 直接归零，因为需要视觉提供 trajectory 跟踪。**两个 modality 缺一不可，但角色不同**。

---

## 为什么用 Flow Matching 不用 Diffusion

Contact-rich 任务要 high-frequency 闭环控制（30Hz），意味着 policy inference 必须很快。

- **Diffusion**：stochastic Markov chain，几十步采样，慢
- **Flow Matching**：deterministic ODE，路径拉直后几步 Euler solver 就能采样

实测 ForceFlow inference latency 83.3ms，跟 ForceVLA 差不多，比 Diffusion Policy（109.9ms）快不少。而且 action horizon 64 步，执行 32 步就 replan，replanning cycle ~1.15s 在 contact 任务里够用。

---

## 最骚的设计：Joint Prediction of Force

ForceFlow 的 output 是 hybrid action：$[\Delta \mathbf{p}_t, \hat{\mathbf{f}}_{t+1}]$，即同时预测 motion command 和下一步的 contact force。

但注意：**预测的 force $\hat{\mathbf{f}}_{t+1}$ 不直接参与 low-level force control**，只用来训练。执行时只把 $\Delta \mathbf{p}_t$ 发给 robot controller。

那为什么要预测它？

**让 network 学到 forward model**。

要让 $\hat{\mathbf{f}}_{t+1}$ 预测准确，network 必须理解：
- "如果我下一步往下压 1mm，根据当前 force history 和接触状态，下一步力会变成多少"
- 这等价于学了 $f: \text{motion} \to \text{force}$ 的 forward dynamics

这个 forward model 反过来约束 motion generation：network 在生成 motion 时，会隐式考虑"这个 motion 会产生什么力"，避免生成会撞墙的动作。

Ablation 证实这点：去掉 force prediction，success rate 只掉 5%（85→80），但 force cost 显著上升（10.61N → 12.52N）。说明 force prediction 是 **compliance regularizer**，主要影响执行质量而非可行性。

---

## 实验里的"惊人时刻"

### 1. Clean Vase 任务

所有 vision-centric baseline 都是 **0%**。因为花瓶是曲面，normal vector 持续变化，vision 提供不了"我现在压在曲面的哪个角度"的信息。ForceFlow 通过 force prediction 做 proactive compliance matching，达到 65%。这是 force 信号不可替代的最纯粹证据。

### 2. Stamp 任务的物理直觉

纸堆从 1 张到 50 张厚度差几十倍，但视觉上几乎看不出区别。Vision-only policy 倾向于 converge 到 mean terminal height，导致薄纸压力不够、厚纸撞坏。ForceFlow 用 10 步 force history 检测 resistance onset（"力突然变大，说明碰到了"），无论纸多厚都在正确时刻停。85% vs 0-20% 的差距。

### 3. ForceFlow (w/o Force) 在 Press 任务上 0%

去掉 force input 后，按弹簧瓶任务完全崩溃。因为不同瓶子 spring constant 差异大，vision 看不出弹簧硬度，只有 force 信号能告诉你"是不是按到底了"。

### 4. 1-step Force vs 10-step History

只用单步 force 信号，success rate 从 85% 掉到 55%。原因是 force sensor 噪声大，单点读数无法区分真实接触事件和 sensor noise。10 步 history 起到 implicit low-pass filter 作用，类似人类抓握物体也是依赖持续触觉反馈而非瞬时单点压力。

### 5. OOD 泛化

替换成不同弹性、不同摩擦系数的工具，ForceFlow 显著好于 ForceVLA。原因是 temporal force history 提供了物理属性 inference signal：按不同 spring constant 的瓶盖，force ramp-up 斜率不同，policy 通过 history 隐式识别这种差异并自适应。

---

## 这个 Paper 的真正价值

ForceFlow 没有发明新数学。Flow Matching 是已有的（[Lipman 2022](https://arxiv.org/abs/2210.02747)），AdaLN 是已有的（[Peebles & Xie DiT 2023](https://arxiv.org/abs/2212.09748)），VLM pointing 是已有的，joint prediction 是 multi-task learning 的常规操作。

它的价值在 **把已有的组件用物理上 meaningful 的方式组装起来**：

1. **架构层面**：用 AdaLN 把 force 提升为"全局控制变量"，用 cross-attention 把 vision 降为"可选参考库"，从架构上消除 modality masking 的可能
2. **任务层面**：把 spatial localization 和 contact regulation 拆开，因为它们是两种根本不同的能力
3. **训练层面**：用 joint force prediction 逼 network 学 motion-force 的 forward model，获得 proactive compliance

37% 的性能提升完全来自这种 architectural inductive bias，没有更多 data、没有更好的 sensor、没有更大的 model。这在 deep learning 里是最纯粹的胜利 —— **你设计对了，network 自己就能学会**。

---

## 参考

- [ForceFlow Project Page](https://jokeresc.github.io/ForceFlow-page)
- [ForceFlow GitHub](https://github.com/JokerESC/ForceFlow)
- [Lipman et al. Flow Matching](https://arxiv.org/abs/2210.02747)
- [Peebles & Xie DiT](https://arxiv.org/abs/2212.09748)
- [Yu et al. ForceVLA](https://openreview.net/forum?id=2845H8Ua5D)
- [Wang et al. What Makes Multi-Modal Training Hard](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.html)

---

# ForceFlow: Contact-Driven Flow Matching 深度技术解析

## 一、Paper 核心定位与问题动机

这篇 paper 来自 Tianjin University 与 Huawei Noah's Ark Lab，核心要解决的是 **contact-rich manipulation** 中长期被忽视的 **modal masking** 问题。在 vision-centric 的 imitation learning 范式下（ACT、Diffusion Policy、$\pi_{0.5}$、OpenVLA），high-dimensional visual feature 会在 end-to-end training 中轻松淹没 low-dimensional 但 temporally rich 的 force/torque 信号。

直观地说，一张 $320 \times 240 \times 3$ 的 RGB image 展开后是 ~230k 维，而 6D force/torque 只有 6 维。当 backprop 的 gradient 同时流过这两个 pathway，optimizer 会倾向于走 vision shortcut，因为 vision 信息量大、对 reconstruction loss 贡献显著。这是 multimodal learning 中经典的 **modality competition / modality laziness** 问题（参考 [Wang et al. CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.html) 与 [Wu et al. ICML 2022](https://proceedings.mlr.press/v162/wu22d.html)）。

ForceFlow 给出的答案是：**架构层面的不对称设计 + 任务层面的层次分解**。

---

## 二、整体架构解析：两层解耦

### 2.1 Task-Level 分解：V2F Handover

human 在做 contact-rich 任务时（插 USB、盖章、擦曲面）的自然感知切换流程：

- **Approach Stage**：vision dominant，用眼睛定位 target 位置
- **Interaction Stage**：force dominant，一旦接触发生，控制权切换到触觉反馈

ForceFlow 将这个流程显式化为 **Vision-to-Force (V2F) handover**：

1. **VLM Pointing Mechanism**：用 VLM（具体使用 Embodied-R1, [Yuan et al. 2025](https://arxiv.org/abs/2508.13998)）从 global camera view $I_{\text{fix}}$ 与 language instruction 预测 target 的 pixel coordinate $(\hat{u}, \hat{v})$。
2. **3D Waypoint Deprojection**：用 depth + camera intrinsics 反投影到 robot base frame，得到 3D approach waypoint。
3. **Motion Planning**：navigator 到达 waypoint。
4. **Trigger Handover**：基于 spatial arrival condition（严格 position-based）触发切换。
5. **ForceFlow 接管**：进入 local interaction window，只做 closed-loop force regulation。

这个设计的核心 insight 是：**spatial generalization 与 physical interaction generalization 是两个根本不同的能力**。前者是 semantic reasoning 问题（"USB 孔在哪"），后者是 high-frequency feedback control 问题（"为什么这把刀切不进去"）。把它们塞进同一个 end-to-end policy 会造成 **mutual performance degradation**。

### 2.2 Policy-Level：Asymmetric Multimodal Fusion

观察空间定义：
$$\mathcal{O}_t = \{I_{\text{arm},t}, I_{\text{fix},t}, \mathbf{q}_t, \mathbf{F}_t^{\text{hist}}\}$$

- $I_{\text{arm},t}, I_{\text{fix},t}$: 双视角 RGB
- $\mathbf{q}_t \in \mathbb{R}^{d_q}$: proprioception（6D pose + 1D gripper = 7D）
- $\mathbf{F}_t^{\text{hist}} \in \mathbb{R}^{H \times d_f}$: force/torque 历史，$H=10$，$d_f=6$（3 force + 3 torque）

**关键设计：两条 conditioning pathway 不对称**

#### Force-Centric Vector Condition $c_{\text{vec}}$

force history + proprioception 编码为 global vector $c_{\text{vec}}$，通过 **Adaptive Layer Normalization (AdaLN)** 注入每个 DiT block。

数学上：
$$\text{AdaLN}(h, c_{\text{vec}}) = \gamma(c_{\text{vec}}) \odot \text{LayerNorm}(h) + \beta(c_{\text{vec}})$$

变量含义：
- $h$: 当前 transformer block 的 hidden state
- $c_{\text{vec}}$: 由 force history + proprio 拼接后经 MLP 编码得到的全局条件向量（dim=256）
- $\gamma(\cdot), \beta(\cdot)$: 由 $c_{\text{vec}}$ 通过线性回归得到的 scale 和 shift 参数
- $\odot$: element-wise product

**直觉**：AdaLN 调制 feature statistics（mean 和 variance），这意味着 force 信号在每一层都会重塑整个特征分布。force 信号不会被 cross-attention 的 softmax 稀释掉，它直接修改 normalization 的 scale/shift，是一种 **persistent regulatory constraint**。

这个设计来自 DiT ([Peebles & Xie, ICCV 2023](https://arxiv.org/abs/2210.02747))，原本用于 class label 与 timestep conditioning，ForceFlow 借用它把 force 信号提升为类似"全局控制变量"的角色。

#### Visual Sequence Condition $c_{\text{seq}}$

多视角 RGB 通过 ResNet-18 编码为 spatial feature sequence（dim=512），通过 **Multi-Head Cross-Attention (MHCA)** 注入：

$$Q = h W_Q, \quad K = c_{\text{seq}} W_K, \quad V = c_{\text{seq}} W_V$$

$$\text{MHCA}(h, c_{\text{seq}}) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

变量含义：
- $h$: action tokens 作为 query
- $c_{\text{seq}}$: visual feature sequence 作为 key/value source
- $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$: 可学习投影矩阵
- $d_k$: 每个 attention head 的 dimension
- $QK^\top$: 计算 action token 与 visual token 的相似度
- $\sqrt{d_k}$: 标准 scaling factor，防止点积过大导致 softmax 饱和

**直觉**：visual 信息被当作"可选查询库"，action token 主动决定从哪里"读取"视觉信息。视觉信号参与生成的方式是 **selective** 的，而不是全局强制的。

这种不对称 fusion 的妙处在于：force 决定"全局物理状态"（AdaLN scale/shift，每一层都强制作用），vision 决定"局部空间参考"（cross-attention 选择性查询）。两个 modality 各司其职，不会互相挤占 capacity。

---

## 三、Flow Matching Backbone：为什么不是 Diffusion？

### 3.1 Linear Probability Path 构造

Hybrid action space 定义：
$$\mathbf{a}_t = [\Delta \mathbf{p}_t, \hat{\mathbf{f}}_{t+1}] \in \mathcal{A}$$

- $\Delta \mathbf{p}_t \in \mathbb{R}^{d_p}$: motion command（pose 增量 + gripper 状态），$d_p=7$
- $\hat{\mathbf{f}}_{t+1} \in \mathbb{R}^{d_f}$: 下一时刻预测接触力，$d_f=6$
- 总 action dim = 13

Flow Matching 构造一条 **linear probability path** 连接 expert distribution 与 Gaussian prior：

$$\mathbf{a}_t^k = (1-k)\mathbf{a}_t^0 + k\mathbf{a}_t^1$$

变量含义：
- $k \in [0,1]$: flow time step（**superscript** 表示在 flow 上的位置，不是 temporal time）
- $\mathbf{a}_t^0 \sim p_{\text{data}}(\mathbf{a})$: 从 expert demonstration 采样的 clean action
- $\mathbf{a}_t^1 \sim \mathcal{N}(0, \mathbf{I})$: 标准 Gaussian prior
- $\mathbf{a}_t^k$: 在 flow 上的中间状态

对应 ground truth velocity field:
$$\mathbf{u}_t^k(\mathbf{a}_t^k, k) = \frac{d\mathbf{a}_t^k}{dk} = \mathbf{a}_t^1 - \mathbf{a}_t^0$$

这是一个 **rectified flow**（参考 [Liu et al. Rectified Flow 2022](https://arxiv.org/abs/2209.14530) 与 [Lipman et al. Flow Matching 2022](https://arxiv.org/abs/2210.02747)）。线性插值的妙处在于 target velocity 是常数 $\mathbf{a}_t^1 - \mathbf{a}_t^0$，与 $k$ 无关，这意味着路径可以被 **rectified**（拉直），可以用更少的 ODE steps 完成采样。

### 3.2 训练目标

$$\mathcal{L}_{FM}(\theta) = \mathbb{E}_{k \sim [0,1], \mathbf{a}_t^0, \mathbf{a}_t^1} \left[ \| v_\theta(\mathbf{a}_t^k, k, c_{\text{vec}}, c_{\text{seq}}) - (\mathbf{a}_t^1 - \mathbf{a}_t^0) \|^2 \right]$$

变量含义：
- $v_\theta$: 神经网络参数化的 velocity field
- $\mathbf{a}_t^k$: flow 上的 noised 中间状态
- $k$: flow time step
- $c_{\text{vec}}, c_{\text{seq}}$: 上文定义的两个 condition pathway
- $\mathbf{a}_t^1 - \mathbf{a}_t^0$: 常数 target drift

**对比 Diffusion Policy**：
- Diffusion 用 stochastic Markov chain，需要数十到数百 step 采样
- Flow Matching 用 deterministic ODE，路径直线化后 Euler solver 几步就能采样
- Contact-rich 任务需要 high-frequency 闭环（30Hz），diffusion sampling 的 latency 是致命的

Table 10 显示 ForceFlow inference latency ~83.3ms，与 ForceVLA（~84.1ms）相当，与 Diffusion Policy（~109.9ms）相比有显著优势，且 action horizon 长达 64（vs DP 的 16），replanning cycle ~1.15s 仍然在可接受范围。

### 3.3 Inference：ODE 求解

$$d\mathbf{a}_t^k = v_\theta(\mathbf{a}_t^k, k, \mathcal{O}_t) \, dk$$

从 $k=1$（noise）积分到 $k=0$（clean action），用 Euler solver 即可。

**预测后处理**：
- 只把 $\Delta \mathbf{p}_t$ 发给 robot controller 执行
- $\hat{\mathbf{f}}_{t+1}$ **不直接参与 low-level force control**，它只是 joint training 用的辅助 head

这是个微妙的设计选择：force prediction 的作用是 **regularize 网络内部表示**，让它隐式学习 motion-force 的 causal coupling，而不是显式做 force feedback control。这避免了 force prediction 误差直接转化为 control error 的风险。

---

## 四、Joint Prediction 的 Intuition：Why Predict Force?

为什么 force prediction head 即使不直接控制也很有用？

考虑这个场景：USB 插入时，你下达一个 $\Delta \mathbf{p}_t$，预测对应的 $\hat{\mathbf{f}}_{t+1}$。要让预测准确，网络必须理解：
- 如果这个 motion 会让插头撞墙，那 $\hat{\mathbf{f}}_{t+1}$ 会突然变大
- 如果这个 motion 让插头对准孔口，那 $\hat{\mathbf{f}}_{t+1}$ 应该接近 zero

所以 joint prediction 等价于让网络学到 **forward model** $f: \text{motion} \to \text{force}$，这个 forward model 反过来约束 motion generation 时考虑物理后果。这类似于 model-based RL 中的 **latent forward model** 思路，但巧妙地通过 multi-head regression 实现，没有任何显式的 dynamics computation。

Ablation Table 5 显示：
- Full ForceFlow: SR=85%, Cost=10.61N
- w/o Force Prediction: SR=80%, Cost=12.52N
- 力预测主要影响 force cost（fidelity），而非 success rate，证实它是 **compliance regularizer** 的角色。

---

## 五、实验数据深度解读

### 5.1 Main Result (Table 1)

| Method | Stamp | Plug | Press | Insert | Clean WB | Clean Vase | Avg |
|---|---|---|---|---|---|---|---|
| $\pi_{0.5}$ | 0% | 60% | 30% | 45% | 10% | 0% | 24.17% |
| ACT | 0% | 30% | 5% | 0% | 15% | 0% | 8.33% |
| Diffusion Policy | 0% | 40% | 20% | 50% | 75% | 0% | 30.83% |
| ForceVLA | 20% | 70% | 65% | 15% | 100% | 0% | 45% |
| ForceFlow (w/o Force) | 20% | 75% | 0% | 40% | 100% | 30% | 44.17% |
| **ForceFlow** | **85%** | **90%** | **90%** | **60%** | **100%** | **65%** | **81.67%** |

关键观察：

1. **Clean Vase 是 game-changer**：所有 vision-centric 方法都是 0%。因为曲面 normal vector 持续变化，仅靠 vision 无法感知 local contact angle。ForceFlow 通过 force prediction 做 proactive compliance matching，达到 65%。这是 ForceFlow 设计哲学最纯粹的体现。

2. **Stamp 的物理直觉**：1-50 张纸厚度视觉上几乎无法区分。Vision-only policy 倾向于 converge 到 mean terminal height，导致薄纸压力不够、厚纸撞坏。ForceFlow 用 10-step force history 检测 resistance onset，无论纸多厚都在正确时刻触发。85% vs 0-20% 的差距直接体现了 force 作为 primary modality 的价值。

3. **ForceFlow (w/o Force) 在 Press 上 0%**：这是个惊人的结果。去掉了 force input，在 press button 这种 spring constant 变化大的任务上完全崩溃。这暗示 force 信号在 Press 任务里是 **唯一可靠的 state transition signal**。

4. **Clean WB 上 w/o Force 也能 100%**：白板是平面，normal vector 恒定，vision 提供的 trajectory 足以维持接触，force 只是辅助。但 Table 2 显示 w/o Force 的 force cost 是 7.16N vs full model 4.59N，意味着虽然 success 一样，但物理 compliant 程度差。

### 5.2 Force Fidelity (Table 2) - 关键定量证据

| Method | Avg MAE Cost (N) |
|---|---|
| $\pi_{0.5}$ | 23.58 |
| ACT | 28.38 |
| Diffusion Policy | 21.42 |
| ForceVLA | 23.31 |
| ForceFlow (w/o Force) | 22.67 |
| **ForceFlow** | **8.23** |

ForceFlow 的 force cost 比 ForceVLA 降了 **65%**。这是 joint prediction 的直接效果。

Force Fidelity Metric 定义：
$$\mathcal{T}_{\text{force}} = \frac{1}{N} \sum_{i=1}^N |\hat{F}_{\text{policy}}^{(i)} - F_{\text{expert}}|$$

- $N=20$: 试验次数
- $\hat{F}_{\text{policy}}^{(i)}$: 第 $i$ 次试验的 force statistic
- $F_{\text{expert}}$: expert demonstration 中的参考 force

**task-dependent 定义**：
- Short-Horizon Contact Tasks（Stamp, Plug, Press, Insert）：取 peak contact force $\max \|\mathbf{f}_t\|$，捕捉 impact intensity
- Continuous Contact Tasks（Clean WB, Clean Vase）：取 contact phase（$\|\mathbf{f}_t\| > 5N$）平均 effective force，评价 tracking consistency

这种分任务 metric 设计很巧妙：瞬时接触和持续接触的物理本质完全不同，前者关心冲击峰值，后者关心跟踪精度。

### 5.3 Ablation: Force History vs Force Prediction (Table 5)

| Variant | SR (%) | Cost (N) |
|---|---|---|
| w/o Force History (1-step) | 55% | 15.50 |
| w/o Force Prediction | 80% | 12.52 |
| w/o Both | 40% | 18.21 |
| **ForceFlow (Full)** | **85%** | **10.61** |

关键 insight：
- **Force history 是决策必需**（85% → 55% drop 巨大）
- **Force prediction 是质量优化**（85% → 80% 微降但 cost 上升）
- 两者协同（w/o Both 只剩 40%）

为什么 1-step force 不够？Force/torque sensor 数据有显著噪声，单点读数无法区分真实接触事件与 sensor noise。Figure 7 的可视化对比显示：1-step variant 有严重的高频 jitter 和 abrupt spike，而 10-step history 起到 **temporal smoothing** 作用，类似一个 implicit low-pass filter。

这呼应了 signal processing 的经典直觉：低信噪比的信号需要时间窗积分才能可靠检测。也是为什么人类抓握物体时也是基于持续的触觉反馈，而非瞬时单点压力。

### 5.4 Visual Modality Ablation (Table 7)

| Method | Stamp | Plug | Press | Insert | Clean WB | Clean Vase | Avg |
|---|---|---|---|---|---|---|---|
| Full | 85% | 90% | 90% | 60% | 100% | 65% | 81.6% |
| w/o vis | 80% | 0% | 15% | 0% | 90% | 0% | 30.8% |

非常深刻的 ablation：在 interaction stage 完全 mask 视觉。

- **Stamp 和 Clean WB 几乎不掉**（85→80, 100→90）：单轴压力调节，force + proprio 就够
- **Plug, Insert, Clean Vase 直接归零**：需要 spatial trajectory tracking，没有 vision 就完全 lost

这印证了 asymmetric fusion 的设计必要性：**force 调节接触物理，vision 提供空间 grounding**，两者缺一不可。

---

## 六、OOD 泛化分析

### 6.1 Physical Interaction Generalization (Table 3)

替换工具（不同橡皮擦、不同弹性瓶），零样本测试：
- ForceVLA: 40% / 90% / 0%
- ForceFlow: 80% / 100% / 60%

ForceFlow 在 unseen 物理属性上显著优于 ForceVLA。原因：temporal force history 提供了物理属性 inference signal。比如按不同 spring constant 的瓶盖，force ramp-up 的斜率不同，policy 通过 history 隐式识别这种差异并自适应。

### 6.2 Spatial Generalization (Table 4 & Table 9)

把目标物放到训练分布外的空间位置：
- 没有 V2F 的所有方法（包括 ForceFlow）：**全部 0%**
- 加上 V2F 的 ForceFlow: 40% / 10% / 50%

最关键的是 Table 9：给所有 baseline 也加上 V2F 模块，结果 baseline 仍然 0%（ForceVLA 加 V2F 也只有 13.33%）。这说明：
- V2F 提供的 spatial grounding 不是"作弊"的 navigation 优势
- 真正的瓶颈是 contact regulation，而 ForceFlow 的 force-centric 设计在这里起决定作用

---

## 七、与相关工作的关联与区别

### 7.1 Force-Aware IL 谱系

按 sensor modality 分类（Table 8）：

| Sensor Type | Representative | ForceFlow 区别 |
|---|---|---|
| High-dim Tactile Array | OmniVTLA, RDP, ViTaL | 触觉阵列 vs 全局 F/T，前者捕捉 local geometry，后者捕捉 global wrench |
| 3D Point Cloud | FeelTheForce, FoAR, ForceMimic | 不需要点云重建，节省算力 |
| Joint Torque | TA-VLA | 用 EEF wrench 比 joint torque 更直接反映 contact |
| 6D EEF Wrench | **ForceVLA, ForceFlow** | 严格对齐，公平对比 |

ForceFlow 选 6D EEF Wrench 是为了与 ForceVLA 严格对齐 input modality，确保性能差异完全来自架构设计而非 sensor 优势。这种 controlled comparison 的严谨性值得称道。

### 7.2 与 ForceVLA 的核心差异

ForceVLA ([Yu et al. NeurIPS 2025](https://openreview.net/forum?id=2845H8Ua5D)) 用 MoE 架构处理 force。MoE 的 routing 机制会动态选择 expert，但 force signal 在 routing 决策中可能被视觉驱动的 router 忽略。ForceFlow 用 AdaLN 强制 force 在每一层都参与 modulation，从架构层面消除了这种"被忽视"的可能。

37% 的性能 gap 完全归因于架构设计，而非 data 或 sensor。

### 7.3 与 Reactive Diffusion Policy / FoAR 的关系

RDP ([Xue et al. ICRA 2025 Workshop](https://openreview.net/forum?id=zRhjjLGUAp)) 和 FoAR ([He et al. ICRA 2025 Workshop](https://openreview.net/forum?id=cbjluXVaJz)) 也探索 force-aware reactive policy，但都用 diffusion backbone。ForceFlow 选择 Flow Matching 的核心 motivation 是 **inference latency**：contact-rich 任务的闭环控制需要 high-frequency response，deterministic ODE 的几步 sampling 远快于 stochastic diffusion 的数十步 sampling。

### 7.4 与 Diffusion Policy / CleanDiffuser 的工程关系

CleanDiffuser ([Dong et al. NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/9e08a1db869a9646418e3371b24c6ae6-Abstract-Datasets_and_Benchmarks_Track.html) 是这个 lab 之前开源的 diffusion-based decision making 库，ForceFlow 大概率是基于 CleanDiffuser 的工程化扩展，将 diffusion 替换为 flow matching，并加入 force-specific pathway。"Conditioning Matters" ([Dong et al. NeurIPS 2025](https://openreview.net/forum?id=pKQcmLHoGG)) 也是同 lab 工作，强调 conditioning design 对 diffusion policy 训练效率的影响，ForceFlow 的 asymmetric fusion 可以视为这个思想在 force modality 上的极致应用。

---

## 八、Hyperparameter 与 Architecture 细节 (Table 6)

值得注意的工程细节：

- **DiT-1D backbone**：Model dim=384, heads=6, depth=12。规模相对小，对应 4×RTX 4090 训练 8-10 小时即可，远小于 VLA 量级
- **Action horizon $H_a$=64**：远长于 Diffusion Policy 的 16，意味着一次 inference 覆盖 ~2s，减少 replanning frequency
- **Executed steps = 32**：执行一半就 replan，类似 [ACT 的 chunking 策略](https://openreview.net/forum?id=e8Eu1lqLaf)，平衡 smoothness 与 responsiveness
- **Force history $H_{\text{force}}$=10**：在 30Hz 下约 333ms，足够覆盖一次完整 contact transient
- **Observation horizon $H_o$=2**：仅 2 帧视觉，对应 ~67ms，说明 vision 主要用于 spatial grounding 而非 motion estimation
- **AdaLN timestep embedding**：Fourier embedding with scale 0.2，untrainable。固定 timestep encoding 是 stable diffusion 一直沿用的设计
- **Image resolution 320×240**：相对低，但配合 ResNet-18 dual view 足够，再次体现 vision 是 auxiliary 而非主导

---

## 九、Limitations 与未来方向

Paper 自承的 limitation：
- 依赖高保真 6D F/T sensor，限制低成本平台部署
- 未来工作：adaptive V2F switching，让 handover 时机更智能

我可以补充几个潜在方向：

1. **Force signal 稀疏性问题**：当前 force history 是 fixed window，但接触事件本身是 sparse 的（大部分时间无接触）。可以借鉴 [Tactile-VLA](https://arxiv.org/abs/2507.09160) 的 attention 机制让 policy 自己学习何时关注 force。

2. **Force prediction 与 model-based RL 的桥接**：force prediction head 本质上学了 forward model $f: \text{action} \to \text{force}$。可以把它拿出来做 MPC 或者 planning，类似 [Diffusion Forcing](https://arxiv.org/abs/2407.01502) 的 next-state prediction 思路。

3. **Tactile array integration**：6D wrench 是 global signal，缺少 local contact geometry。结合 GelSight 类 sensor 可以补全 contact patch 的丰富信息。OmniVTLA 已经在探索这个方向。

4. **Sim-to-Real for Force**：force signal 在 sim 中很难准确仿真（contact dynamics 对 mesh、friction model 极度敏感）。这限制了大规模 sim pretraining 的可行性。可以考虑 domain randomization on contact stiffness 或者 system identification。

5. **Active Compliance 与阻抗控制的结合**：当前 force prediction 不直接做 control，但可以把预测 force 作为 impedance controller 的 reference，形成 hybrid force-position control。这是 classical robotics 的 [Hybrid Force-Position Control](https://ieeexplore.ieee.org/document/1087267) 在 learned policy 框架下的复兴。

---

## 十、Take-away Intuition

如果让我给 Karpathy 一句话总结 ForceFlow 的 essence：

**"Don't ask the network to figure out how to fuse vision and force. Tell it architecturally that force is a global state variable (via AdaLN) and vision is a queryable spatial reference (via cross-attention). Then train it to predict force alongside action, so it must learn the causal physics. And separate spatial localization from contact regulation into different policies, because they're fundamentally different problems."**

这个 work 在我看来是 force-aware IL 领域的一个 **architectural baseline**：它不发明新数学（Flow Matching 是已有的，AdaLN 是已有的，VLM pointing 是已有的），但它把这些组件用一种 **物理上 meaningful 的方式** 组装起来，让 force 信号在架构层面获得应有的地位。37% 的性能提升完全来自这种 architectural inductive bias，这本身就是 deep learning 中最纯粹的胜利。

---

## 参考 Links

- [ForceFlow Project Page](https://jokeresc.github.io/ForceFlow-page)
- [ForceFlow GitHub Code](https://github.com/JokerESC/ForceFlow)
- [ForceFlow HuggingFace Datasets](https://huggingface.co/datasets/JokerESC/ForceFlow)
- [Lipman et al. Flow Matching (NeurIPS 2022)](https://arxiv.org/abs/2210.02747)
- [Liu et al. Rectified Flow (2022)](https://arxiv.org/abs/2209.14530)
- [Peebles & Xie DiT (ICCV 2023)](https://arxiv.org/abs/2212.09748)
- [Chi et al. Diffusion Policy (RSS 2023)](https://diffusion-policy.cs.columbia.edu/)
- [Zhao et al. ACT (RSS 2023)](https://tonyzhaozh.github.io/aloha/)
- [Yu et al. ForceVLA (NeurIPS 2025)](https://openreview.net/forum?id=2845H8Ua5D)
- [Yuan et al. Embodied-R1 (2025)](https://arxiv.org/abs/2508.13998)
- [Black et al. $\pi_{0.5}$ (CoRL 2025)](https://openreview.net/forum?id=vlhoswksBO)
- [Kim et al. OpenVLA (CoRL 2024)](https://openvla.github.io/)
- [Bjorck et al. GR00T N1 (2025)](https://arxiv.org/abs/2503.14734)
- [He et al. FoAR (ICRA 2025 Workshop)](https://openreview.net/forum?id=cbjluXVaJz)
- [Xue et al. Reactive Diffusion Policy (ICRA 2025 Workshop)](https://openreview.net/forum?id=zRhjjLGUAp)
- [Wang et al. What Makes Multi-Modal Training Hard (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.html)
- [Wu et al. Greedy Nature of Multi-Modal Learning (ICML 2022)](https://proceedings.mlr.press/v162/wu22d.html)
- [Dong et al. CleanDiffuser (NeurIPS 2024)](https://papers.nips.cc/paper_files/paper/2024/hash/9e08a1db869a9646418e3371b24c6ae6-Abstract-Datasets_and_Benchmarks_Track.html)
- [Dong et al. Conditioning Matters (NeurIPS 2025)](https://openreview.net/forum?id=pKQcmLHoGG)
- [Huang et al. Tactile-VLA (2025)](https://arxiv.org/abs/2507.09160)
- [Adeniji et al. Feel The Force (2025)](https://arxiv.org/abs/2506.01944)
- [Zhao et al. Touch Begins Where Vision Ends (RSS 2025 Workshop)](https://openreview.net/forum?id=vbW7BVKAeb)
