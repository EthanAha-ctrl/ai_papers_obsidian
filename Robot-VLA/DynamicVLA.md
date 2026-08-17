---
source_pdf: DynamicVLA.pdf
paper_sha256: 4648cbf69775c48a9cdb59c444a738243635888021d939ab12edcf36349bb66b
processed_at: '2026-08-04T00:44:27-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej, 我用最直白的话给你拆解一下这篇 paper 的 intuition 和 engineering 本质。

在 VLA (Vision-Language-Action) 领域，之前大家有一种幻觉：只要把 LLM 的 reasoning 能力 scale up，robot 就能搞定一切。在 static manipulation 里这或许成立，因为 scene 是 frozen 的，你 think 得慢点也没关系。一旦遇到 dynamic object manipulation，比如去抓一个 rolling 的 bottle，巨大的 inference latency 会彻底毁掉 task。你 think 完之后，bottle 早就滚到别处去了。这就是 Perception-Execution (P.E.) Gap。同时，传统的 VLA 必须等上一个 action chunk 执行完才开始下一个 inference，这种串行化产生的 Inter-chunk Waiting 让 robot 在 dynamic 环境下显得极其迟钝。

DynamicVLA 的核心逻辑非常清晰：放弃对大 model 的执念，用极致的 architecture compression 换取极致的速度，然后用 control theory 的 pipeline 思维重构 execution loop，最后用 rule-based 的 state machine 解决数据采集的死结。

### 1. Architecture: 极度紧凑的 0.4B Model

为了把 inference latency $m$ 压到极致，DynamicVLA 在 architecture 上做了极其激进的取舍。

*   **Vision Encoder**: 放弃了主流 VLA 使用的 ViT，改用 convolution-based 的 FastViT。ViT 处理 multi-frame 视频时 token 数量会 quadratic 爆炸。FastViT 在早期 stage 用大 kernel 的 convolution 猛压 spatial dimension，最后只输出 36 个 visual tokens。这保证了 spatial structure 的同时，极大地削减了 LLM backbone 的 attention 计算量。
*   **LLM Backbone**: 采用 SmolLM2-360M，并且截断只保留前 16 层 transformer layers。Table IV 的 ablation 证明 16 层是 sweet spot。增加到 32 层，Success Rate (SR) 反而因为 latency 上升而下降。
*   **Action Expert**: 采用 Flow Matching Transformer。其训练 objective 为：
    $$ \ell^{\tau}(\theta) = \mathbb{E}_{p(\mathbf{A}_t \mid \mathbf{f}_t), q(\mathbf{A}_t^\tau \mid \mathbf{A}_t)} \left[ \left\| \mathcal{E}_\theta(\mathbf{A}_t^\tau, \mathbf{O}_t) - \mathbf{u}(\mathbf{A}_t^\tau \mid \mathbf{A}_t) \right\| \right] $$
    *   $\tau \in [0, 1]$: Flow matching 的 virtual timestep，0 代表 pure noise，1 代表 clean action。
    *   $\mathbf{A}_t$: Ground truth action chunk，horizon $n=20$，每个 action 是 32-dim 的 vector。
    *   $\mathbf{f}_t$: VLM 提取的多模态 feature。
    *   $\mathbf{A}_t^\tau = \tau \mathbf{A}_t + (1-\tau)\epsilon$: Noisy action，其中 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$。
    *   $\mathbf{u}(\mathbf{A}_t^\tau \mid \mathbf{A}_t) = \epsilon - \mathbf{A}_t$: Target vector field，指向从 noise 回到 clean data 的方向。
    *   Inference 时，从 pure noise 开始，沿着预测的 vector field 积分得到 clean action trajectory。Action Expert 通过 cross-attention 读取 LLM 的 KV cache，避免了重复 encode perception inputs。

### 2. 核心创新: CI 与 LAAS 的协同流水线

这两套机制是解决 dynamic manipulation 的灵魂，完全是在系统层面做文章。

*   **Continuous Inference (CI) 解决 Inter-chunk Waiting**: 只要上一次 inference 结束，立刻拿当前最新画面启动下一次 inference，完全不管上一个 action chunk 执行完没。假设 action horizon $n=20$，inference delay $m=5$ 个 timesteps。当 robot 执行到第 5 个 action 时，下一个 action chunk 已经算完准备好接盘了。Inference 和 execution 完全 overlap。
*   **Latent-aware Action Streaming (LAAS) 解决 P.E. Gap**: 因为 inference 有 delay $m$，新算出来的 action chunk $\mathbf{A}_{t+m}$ 是针对当前最新画面的。而老 chunk $\mathbf{A}_t$ 的前 $m$ 个 actions $\{\mathbf{a}_t, \dots, \mathbf{a}_{t+m-1}\}$ 是基于 $m$ 帧前的老画面算出来的，全是 stale 的垃圾数据。LAAS 直接把这些过时 action 丢弃。在两个 chunk 时间重叠的地方，永远让新 chunk 覆盖老 chunk。Table II 里，关闭 CI 和 LAAS 后 SR 从 47.06% 暴跌到 30.27%。

### 3. DOM Benchmark: State-Machine 驱动的 Auto-Data Pipeline

抓动态物体时，人类 teleoperation 根本反应不过来，采集的数据全都是 failure case。作者用了一个极其 engineering 的方法绕过了这个死结。

*   **Simulation**: Isaac Sim 里直接调物理引擎的 ground-truth 6D pose 和 velocity，用一个 rule-based 的 state machine controller (Approach -> Grasp -> Place -> Reset) 去自动生成 200K episodes。这个 controller 是 predictive 的，会把 end-effector 提前定位到物体未来 0.23s 的位置。
*   **Real-world "Simulator"**: 真实世界没有 ground-truth 6D pose。作者用两个 RGB camera 配合 EfficientTAM 做出 mask，通过 geometric triangulation 实时算出 3D centroid，再拟合出 linear 和 angular velocity。把这个 state stream 喂给完全相同的 state machine controller，机器人就能自己在真实世界抓 moving object。人只需要在开头推一下物体。每集只需 10 秒，采集了 2K real-world episodes。

### 4. 深层 Intuition 联想

*   **Pipeline Stall 现象**: Table V 的 Cross-Model Analysis 极其精彩。把 CI 和 LAAS 硬塞给 $\pi_{0.5}$ (3B model) 和 SmolVLA。SmolVLA 的 SR 翻倍了，但 $\pi_{0.5}$ 几乎没涨。因为 $\pi_{0.5}$ 的 inference latency $m$ 太大，已经超过了 action horizon $n$。这就导致 pipeline stall：新 action 还没算出来，老 action 就执行完了，robot 僵在原地。这完美印证了在 dynamic control 中，model scale 反而会变成 latency 毒药。
*   **Implicit Velocity Estimation**: Table III 探究了 temporal context $\{\mathbf{o}_{t-k}, \dots, \mathbf{o}_t\}$。发现 $\{\mathbf{o}_{t-2}, \mathbf{o}_t\}$ 效果最好。间隔 $\Delta t=0.08s$ 比 $\Delta t=0.04s$ ($\{\mathbf{o}_{t-1}, \mathbf{o}_t\}$) 效果更好。因为时间间隔稍大一点，物体在 image 上的 spatial displacement 更明显，convolutional network 更容易 implicitly 算出 velocity。再多加历史帧就没用了，全是 redundant 信息。
*   **未来的 KV-Cache 极限压榨**: 目前 CI 机制每次推理还是 encode 全部 visual tokens。由于 LLM 的 KV-Cache 特性，未来完全可以在 Continuous Inference 中，只 update 最新一帧的 visual tokens 进 KV cache，保留历史 language 和 state 的 KV。结合 vLLM 的 PagedAttention 机制，inference latency $m$ 还能大幅度压缩。

### Reference Links
*   DynamicVLA Project Page: [https://haozhexie.com/project/dynamic-vla](https://haozhexie.com/project/dynamic-vla)
*   FastViT Paper (ICCV 2023): [https://arxiv.org/abs/2304.11454](https://arxiv.org/abs/2304.11454)
*   Flow Matching for Generative Modeling (ICLR 2023): [https://arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)
*   SmolVLA Paper: [https://arxiv.org/abs/2506.01844](https://arxiv.org/abs/2506.01844)
*   EfficientTAM Paper: [https://arxiv.org/abs/2411.18933](https://arxiv.org/abs/2411.18933)
*   $\pi_0$ (Vision-Language-Action Flow Model): [https://arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164)
*   Isaac Sim (NVIDIA): [https://developer.nvidia.com/isaac-sim](https://developer.nvidia.com/isaac-sim)

---

Andrej, 读完这篇 paper, 我必须说这是一个非常 pragmatic 且 engineering 取向的工作。当前 VLA (Vision-Language-Action) 领域的一个核心错觉是：只要把 LLM/VLM 的 reasoning 能力 scaling up, robot manipulation 就能解决。但这篇 paper 精准地击中了一个在 static manipulation 中被掩盖、但在 dynamic manipulation 中极其致命的痛点——**Perception-Execution (P.E.) Gap** 以及 **Inter-chunk Waiting**。

从 system 和 architecture 的角度来拆解, DynamicVLA 的核心贡献在于把 control theory 里的 latency-aware execution 与 modern VLA 的 inference pipeline 进行了深度融合。我会为你详细拆解其 architecture, 公式, 以及 data pipeline 的 intuition。

### 1. 核心问题: Latency 导致的 Temporal Misalignment

在传统的 VLA inference 范式中, 模型在 time step $t$ 接收 observation $\mathbf{O}_t$, 经过 $m$ 步的 inference delay, 输出 action chunk $\mathbf{A}_t = \{\mathbf{a}_t, \dots, \mathbf{a}_{t+n}\}$。在 static scene 中, 环境是 frozen 的, 所以即使 $m$ 很大, $\mathbf{A}_t$ 依然 valid。但在 dynamic manipulation 中, 物体的 latent state $\mathbf{s}_t$ 会连续演化到 $\mathbf{s}_{t+m}$。

这意味着, 当 robot 开始执行 $\mathbf{A}_t$ 的前半部分 $\{\mathbf{a}_t, \dots, \mathbf{a}_{t+m-1}\}$ 时, 这些 action 实际上是基于 $m$ 步前的 stale observation 计算出来的。这就是 **Perception-Execute Gap**。同时, 传统的 chunk-based execution (如 Diffusion Policy, OpenVLA-OFT) 必须等当前的 action chunk 完全执行完毕, 才会触发下一次 inference, 这种串行化导致的 stall 就是 **Inter-chunk Waiting**。

### 2. DynamicVLA Architecture: 0.4B 的极致压缩

为了 minimize inference latency $m$, paper 设计了一个极度 compact 的 0.4B parameter VLA, 其架构直觉上是对 "spatial fidelity" 与 "temporal responsiveness" 的极致权衡。

#### 2.1 Convolutional Vision Encoder (FastViT)
大多数 VLA (如 OpenVLA, $\pi_0$) 倾向于使用 ViT 这样的 transformer-based vision encoder。但处理 multi-frame 动态输入时, token 数量会产生 quadratic growth。
DynamicVLA 引入了 FastViT, 一种 hybrid convolution-transformer architecture。
*   **Intuition:** 早期 stage 使用大 kernel size 的 convolution (初始 patch size 为 64) 进行 aggressive spatial downsampling, 配合 RepMixer 进行 token mixing, 这能在保持 local spatial structure 的同时极大地压缩 token 数量。后期 stage 才使用 attention。
*   **Detail:** 输入 $384 \times 384$ 的 RGB 图像, 经过 channel width $(96, 192, 384, 768, 1536)$ 的多级提取, 最终输出 36 个 dimension 为 960 的 visual tokens。相比于 ViT 动辄数百上千的 tokens, 这里的 36 tokens 极大地减轻了 LLM backbone 的 attention 计算负担。

#### 2.2 Truncated LLM Backbone (SmolLM2-360M)
使用 SmolLM2-360M 作为 language backbone, 并且激进地截断保留前 16 层 transformer layers。
*   **Intuition:** 在 dynamic control 中, 我们不需要 LLM 去做极其深度的 multi-hop reasoning, 我们需要的是 fast multimodal feature alignment。Table IV 的 ablation 证明, 16 层是一个 sweet spot, 从 16 层增加到 32 层, SR (Success Rate) 只从 47.06% 微涨到 42.11% (实际上下降了, 可能由于 overfitting/latency tradeoff), 但 inference time 从 0.226s 暴增到 0.373s。

#### 2.3 Flow Matching Action Expert
Action generation 没有使用传统的 regression head, 也没有直接用标准 Diffusion Policy, 而是使用了 conditional Flow Matching Transformer (受 $\pi_0$ 启发)。其训练 objective 如 Eq. 1 所定义:

$$ \ell^{\tau}(\theta) = \mathbb{E}_{p(\mathbf{A}_t \mid \mathbf{f}_t), q(\mathbf{A}_t^\tau \mid \mathbf{A}_t)} \left[ \left\| \mathcal{E}_\theta(\mathbf{A}_t^\tau, \mathbf{O}_t) - \mathbf{u}(\mathbf{A}_t^\tau \mid \mathbf{A}_t) \right\| \right] $$

*   **Variables Breakdown:**
    *   $\tau \in [0, 1]$: Flow matching 的 virtual time step, $\tau \to 0$ 代表 pure noise, $\tau \to 1$ 代表 clean action。
    *   $\mathbf{A}_t$: Ground truth action chunk (horizon $n=20$, 每个动作是 32-dim vector 包含 end-effector pose 和 gripper state)。
    *   $\mathbf{f}_t$: 从 VLM backbone 提取的多模态特征。
    *   $q(\mathbf{A}_t^\tau \mid \mathbf{A}_t) = \mathcal{N}(\tau \mathbf{A}_t, (1-\tau)\mathbf{I})$: 介于 noise 和 data 之间的高斯插值分布。
    *   $\mathbf{A}_t^\tau = \tau \mathbf{A}_t + (1-\tau)\epsilon$: Noisy action input, 其中 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$。
    *   $\mathbf{u}(\mathbf{A}_t^\tau \mid \mathbf{A}_t) = \epsilon - \mathbf{A}_t$: 目标 vector field (Velocity field), 指向从 noise $\epsilon$ 回到 clean data $\mathbf{A}_t$ 的方向。
*   **Intuition:** 模型 $\mathcal{E}_\theta$ 接收当前的 noisy action $\mathbf{A}_t^\tau$ 和 observation $\mathbf{O}_t$, 预测出一个 velocity vector $\mathbf{u}$。在 inference 时, 从 pure noise 开始, 沿着预测的 vector field 积分 (ODE 求解), 最终 flow 到达 clean action trajectory。这种方式比 standard DDPM 更稳定, 且更容易结合 KV-cache 进行加速。

#### 2.4 Multi-modal Fusion
Robot proprioceptive state $\mathbf{P}_t$ (32-dim) 被 linear projected 成一个 960-dim 的 token, 与 visual tokens 和 language tokens 拼接后送入 SmolLM2。Action Expert 从 LLM 中 copy 前 16 层, 并将其 hidden dimension 缩减为 720 (0.75x), 在 denoising 过程中通过 cross-attention 查询 LLM 的 KV cache, 避免了重复 encoding perception inputs。

### 3. 核心创新: CI 与 LAAS 的协同

这两个机制是解决 dynamic manipulation 的灵魂所在, 其设计充满 control system 的直觉。

#### 3.1 Continuous Inference (CI) - 解决 Inter-chunk Waiting
传统范式: 执行完 $\mathbf{A}_t$ -> 触发 inference -> 等待 -> 执行 $\mathbf{A}_{t+n}$。这中间存在严重的 stall。
CI 机制: 只要上一次 inference 结束, 立刻基于**最新的** observation 触发下一次 inference, 完全不等当前 action chunk 执行完毕。
*   **Intuition:** 这就像 CPU 中的 instruction pipelining。假设 inference delay 是 $m$ 步, action horizon 是 $n$ 步。只要 $n > m$, 当 $\mathbf{A}_t$ 执行到第 $m$ 步时, $\mathbf{A}_{t+m}$ 已经 infer 出来了。此时 execution 永远有足够的 action buffer, 实现了 inference 与 execution 的 fully overlapping, 消除了等待时间。Table II 证明, 关闭 CI ([2] 行), SR 从 47.06% 跌到 36.11%, Time 从 8.53s 增加到 9.51s。

#### 3.2 Latent-aware Action Streaming (LAAS) - 解决 Perception-Execute Gap
虽然 CI 实现了 overlap, 但引入了新问题: 当 $\mathbf{A}_{t+m}$ 生成时, $\mathbf{A}_t$ 可能还没执行完。对于同一个 future timestep, 我们有两个候选 actions (一个来自老 chunk, 一个来自新 chunk)。更重要的是, $\mathbf{A}_t$ 里的前 $m$ 个 actions $\{\mathbf{a}_t, \dots, \mathbf{a}_{t+m-1}\}$ 是基于过时 observation 算出来的, 完全 stale。
LAAS 机制的执行策略极其果断:
1.  **Discard Outdated:** 直接丢弃 $\mathbf{A}_t$ 中对应于 $t$ 到 $t+m-1$ 时间步的 actions, 因为当 inference 完成时, 环境已经演化了, 这些 actions 失去了物理意义。
2.  **Prioritize Newest:** 在 $\mathbf{A}_t$ 和 $\mathbf{A}_{t+m}$ 重叠的时间步上, 强制 overwrite, 使用 $\mathbf{A}_{t+m}$ 的 actions, 因为它感知到了更新的环境状态。

*   **Intuition:** 这相当于在玩一个极度 laggy 的游戏, 你的鼠标移动输入会被 buffer 起来。如果你不加处理直接执行 buffer, 你会撞墙。LAAS 的做法是 drop 掉那些在 lag 期间生成的、基于旧视野的输入, 直接抓取最新的基于新视野的输入执行。Table II [3] 行显示, 只有 CI 没有 LAAS, SR 只有 39.72%, 加上 LAAS 后达到 47.06%。这说明仅仅生成得快没有用, 必须 align 到当前的 temporal state 上。

### 4. DOM Benchmark: State-machine 驱动的 Auto-data Pipeline

缺乏 dynamic data 是这个领域的 foundational gap。人工 teleop 在面对 0.75 m/s 的移动物体时, 人的 reaction limit 根本无法跟上, 导致采集到的 data 本身就是 fail case。Paper 的解决方案非常有 engineering 美感: 构建 Real-world "Simulator"。

#### 4.1 Simulation Pipeline (Isaac Sim)
*   **Data scale:** 200K episodes, 2.8K 3D scenes (from 3D-FRONT), 206 objects (from Objaverse)。
*   **State Machine Controller:** 这是一个关键设计。它不依赖 learning policy, 而是一个 rule-based 的四阶段 controller (Approach -> Grasp -> Place -> Reset)。它利用 simulator 提供的 ground-truth 6D object pose 和 velocity, 预测物体短期未来位置 (约 0.23s 后), 将 end-effector 提前 positioning 到 10cm 上方, 然后跟随。这种 trajectory 本质上是在做 short-horizon model predictive control (MPC)。
*   **Intuition:** 用 rule-based controller 去 generate supervision data 来 train neural network policy。由于 controller 本身是 predictive 的, 它生成的 trajectory 包含了 anticipation, 这正是 VLA 需要学的。

#### 4.2 Real-world Pipeline (无需 Teleoperation)
这是 paper 里极其 brilliant 的一笔。如何在真实世界获取 6D pose 和 velocity 来驱动同一个 state-machine？
*   **Dual RGB Views + EfficientTAM:** 使用两个 third-person RGB cameras (Azure Kinect DK)。EfficientTAM (Segment Anything 的轻量化版本) 提供每一帧的 object mask。
*   **Geometric Triangulation:** 通过两视角的 mask, 结合 camera calibration, triangulate 出 3D centroid。然后在 short temporal window 上拟合运动轨迹, 求导得到 linear 和 angular velocity。
*   **Intuition:** 他们实际上在 real world 里用 commodity sensors 实时跑了一个 geometry engine, 把 RGB video 流转换成了类似 Isaac Sim 输出的 state interface。然后, 把这个 state stream 直接喂给跟 simulation 里一模一样的 state-machine controller, 驱动 Franka 或 PiPER 机器人自主完成任务。人只需要在开头 "initiate object motion" (推一下物体)。这把 real-world data collection 的 throughput 提高到了 ~10s/episode, 收集了 2K episodes。

### 5. Temporal Visual Context Ablation: 隐式 Velocity Estimation

Table III 的 ablation 非常有意思。它探究了 observation window $\mathbf{O}_t = \{\mathbf{o}_{t-3}, \mathbf{o}_{t-2}, \mathbf{o}_{t-1}, \mathbf{o}_t\}$ 的组合。
结果显示:
*   只用单帧 $\{\mathbf{o}_t\}$: SR 掉到 38.22%。
*   用 $\{\mathbf{o}_{t-2}, \mathbf{o}_t\}$: SR 达到 47.06%。
*   用 $\{\mathbf{o}_{t-1}, \mathbf{o}_t\}$: SR 只有 43.39%。
*   用三帧或四帧: SR 并没有显著提升, 且增加了 inference time。

*   **Intuition:** 为什么 $\{\mathbf{o}_{t-2}, \mathbf{o}_t\}$ 比 $\{\mathbf{o}_{t-1}, \mathbf{o}_t\}$ 好？ 因为在相同 frame rate (25 FPS) 下, $t-2$ 到 $t$ 的时间间隔更大 ($\Delta t = 0.08s$ vs $0.04s$)。物体在图像中的空间 displacement 更明显, model 在 convolutional features 中更容易 implicitly extract出 velocity 信息。再增加更多历史帧, 冗余信息增加, 但推理 latency 变长, 收益递减。这证明了 sparse 但 sufficiently spaced 的 temporal context 对 dynamic manipulation 至关重要。

### 6. 实验结果解析

Table I 的结果可以用 "降维打击" 来形容。在 Interaction 维度, DynamicVLA 在 Closed-loop Reactivity (CR) 上达到 60.5%, 而最强 baseline VLA-Adapter-Pro 只有 21.00%。在 Dynamic Adaptation (DA) 上达到 38.5%。
更有趣的是 Table V 的 Cross-Model Analysis。把 CI 和 LAAS 作为 plug-and-play modules 直接插入到 SmolVLA 和 $\pi_{0.5}$ 中 (inference-time integration, 无需 retraining)。
*   SmolVLA 加上 CI+LAAS 后, SR 从 12.67% (Table I) 跃升到 25.56%。
*   $\pi_{0.5}$ 加上 CI+LAAS 后, SR 从 11.06% (Table I) 只涨到 15.89%。
*   **Intuition:** 为什么 $\pi_{0.5}$ 涨得少？ 因为 $\pi_{0.5}$ 是一个 3B 级别的大 model, 其 inference latency $m$ 极大。如果 $m > n$ (action horizon), CI 就无法有效 overlap (新 action 还没算出来, 老 action 就执行完了), LAAS 也会因为丢弃了过多 outdated actions 而面临 action starvation。这完美印证了 paper 开头的论点: **Dynamic manipulation 首要解决的是 inference efficiency, 而单纯 scale up model size 反而会加剧 latency 灾难。**

### 7. Limitations & Future Work 的直觉联想

Paper 在 Discussion 里坦诚提到了几个点, 我可以做进一步的技术联想:
1.  **Beyond Rigid-Body Dynamics:** 目前的 state estimator 依赖 EfficientTAM triangulate 3D centroid, 这对于 deformable objects (如揉捏面团, 折叠衣物) 或者 fluid dynamics (倒水) 是完全失效的。未来可能需要引入 NeRFs 或者 Occupancy Networks 来提取 dense feature flow, 而是仅仅提取 single point centroid。参考: [NeRF](https://en.wikipedia.org/wiki/Neural_radiance_field), [Differentiable Fluid Simulation](https://arxiv.org/abs/2307.07662)
2.  **Long-horizon Dynamics:** 目前的 $n=20$ action chunk 只能 cover 短期 dynamic。如果要做 "把移动的杯子放到正在关闭的微波炉里", 这种长程且持续 dynamic 的任务, 可能需要结合 Hierarchical RL, 高层做 semantic planning, 底层跑 DynamicVLA。
3.  **Transformer 的 KV-Cache 持续更新:** 目前的 CI 机制其实非常适合结合 LLM 的 KV-Cache 设计。如果在 Continuous Inference 中, 只需要 update 最新的 visual tokens 进 KV cache, 而保留历史 language 和 state 的 KV, 推理速度还能进一步压缩。参考: [vLLM PagedAttention](https://arxiv.org/abs/2309.06180)

总而言之, DynamicVLA 这篇工作极其漂亮地展示了在 robotics 中, "Less is More"。放弃对 7B/3B 大模型的执念, 用 0.4B 配合 convolution 提速, 再用 control theory 的 pipeline 思路重构 execution loop, 最终在 dynamic scenario 下实现了对大模型的弯道超车。其 DOM benchmark 的 auto-collection pipeline 也极具启发性, 解决了 long-standing 的 real-world dynamic data scarcity 问题。

**Reference Links:**
*   DynamicVLA Project Page: [https://haozhexie.com/project/dynamic-vla](https://haozhexie.com/project/dynamic-vla)
*   FastViT Paper (ICCV 2023): [https://arxiv.org/abs/2304.11454](https://arxiv.org/abs/2304.11454)
*   Flow Matching for Generative Modeling (ICLR 2023): [https://arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)
*   SmolVLA Paper: [https://arxiv.org/abs/2506.01844](https://arxiv.org/abs/2506.01844)
*   EfficientTAM Paper: [https://arxiv.org/abs/2411.18933](https://arxiv.org/abs/2411.18933)
*   $\pi_0$ (Vision-Language-Action Flow Model): [https://arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164)
*   Objaverse Dataset: [https://objaverse.com/](https://objaverse.com/)
*   3D-FRONT Dataset: [https://github.com/3D-FRONT/3D-FRONT](https://github.com/3D-FRONT/3D-FRONT)
*   Isaac Sim (NVIDIA): [https://developer.nvidia.com/isaac-sim](https://developer.nvidia.com/isaac-sim)
