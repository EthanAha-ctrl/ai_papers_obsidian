---
source_pdf: ClosedLoop-UTNV-Report.pdf
paper_sha256: 9479deca4819657a3d024f097720b7c015e47cec5d8ba94e6572b95717c3d9e2
processed_at: '2026-08-03T15:56:27-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 Paper

---

## 1. 这篇 Paper 在干啥

一句话概括：**别人的 foundation model 已经训练好了，输出一条轨迹（一系列 waypoints），但是这条轨迹直接丢进 simulator 执行会撞车、会颠簸、会开得扭扭捏捏。作者团队做的就是一个 lightweight 的 post-processing（后处理），把这条轨迹"修整"一下，让它在 closed-loop 仿真里跑得好很多。**

结果：HUGSIM 2025 Challenge 第一名。

---

## 2. 为什么需要这个修整？先讲清楚 OL/CL Gap

### 2.1 Open-loop training 是怎么训练的

现在所有 end-to-end driving model 几乎都是用 behavior cloning (BC) 训练的。BC 就是模仿学习——给一堆 expert driver 的轨迹数据 $(o_t, a_t)$，让模型 $f_\theta$ 学着预测 expert 的 action：

$$\mathcal{L}_{BC} = \sum_{t=1}^{T} \| f_\theta(o_{1:T}) - a_t^{GT} \|^2$$

这里 $o_{1:T}$ 是过去 $T$ 帧的 observation（图像、LiDAR等），$a_t^{GT}$ 是 expert 在时刻 $t$ 的真实 action（通常是 $(x, y, \theta)$ waypoint）。

关键点：**训练时每一帧的 input observation 都来自 expert 的真实轨迹**——也就是 ground truth 数据。这意味着 model 永远在"看 expert 的视角"做预测。

### 2.2 Closed-loop 评估是怎么执行的

CL 评估时，model 自己预测 waypoint，simulator（HUGSIM）执行这个 waypoint，渲染出下一帧 observation，然后 model 再基于这一帧预测下一段 trajectory。这是一个真正的闭环：

```
Model predicts trajectory → Simulator executes → New observation rendered → Model predicts again → ...
```

### 2.3 Gap 的本质：Compounding Error

这里的问题叫 **compounding error**，最早由 Ross et al. DAgger (https://arxiv.org/abs/1011.0689) 系统分析过。原理是这样：

假设 model 在每一步有 $\epsilon$ 的小误差。在 OL 评估中，下一步的 input 还是 ground truth，所以误差不会累积。但在 CL 评估中，下一步的 input 是基于上一步预测执行的，误差会复合：

$$\text{Total Error} \sim \mathcal{O}(T \cdot \epsilon + T^2 \cdot \epsilon^2 + ...)$$

即使每步误差很小，累积下来也会让 ego vehicle 偏离 expert distribution，进入 model 从未见过的 state，然后 model 在这些 OOD state 上预测更差，进入恶性循环。

DAgger 的解决方案是在训练时主动 inject model 的预测，让 expert 在 model 的 state 分布上给 label。但这篇 paper 没改训练，只改 inference。

### 2.4 Paper 的关键观察

作者团队在 HUGSIM 上跑了 VaVAM 的 closed-loop，统计了每个 waypoint 位置发生 collision 的次数（Figure 3），发现一个特别有意思的 pattern：

- 第一个 waypoint（t=0.5s，实际被执行的那个）：collision 少
- 中间 waypoints（t=1s 到 2.5s）：collision 多
- 最后一个 endpoint（t=3s）：collision 少

这是一个 "U-shape" 的 error 分布。

为什么 endpoint 靠谱？因为 endpoint 实际上是被 navigation command 强约束的——"左转"这个 command 基本上决定了 3 秒后车在哪。模型学到了这个 strong prior。

为什么中间 waypoints 不靠谱？因为它们是 "free-form" 生成的，没有 strong constraint。flow matching / diffusion 的 stochasticity 在这些位置更容易产生不稳定。

这个观察直接启发了 ECO 的设计：**保留 endpoint，修整中间**。

---

## 3. VaVAM 这个基础模型

VaVAM 是这篇 paper 用的 foundation model，来自 Valeo AI 实验室（https://arxiv.org/abs/2502.15672）。它的架构拆解：

### 3.1 VaVIM: Video Foundation

VaVIM 是一个 large-scale generative video model，类似 Sora / Stable Video Diffusion，在 internet-scale driving videos 上预训练。它的作用是学习 visual representations——理解 driving scene 的语义。

### 3.2 VaVAM Action Head: Flow Matching

VaVAM 不用 diffusion 而是 flow matching（Lipman et al. 2023, https://arxiv.org/abs/2210.02747）。Flow matching 学一个 conditional vector field $v_\theta$，把一个简单分布（Gaussian）"流动"到 data 分布（driving trajectories）：

$$\frac{d\mathbf{x}_t}{dt} = v_\theta(\mathbf{x}_t, t, \mathbf{c})$$

变量解释：
- $\mathbf{x}_t \in \mathbb{R}^{N \times 3}$: trajectory 的 waypoints，每个 waypoint 是 $(x, y, \theta)$，$N$ 是 waypoint 数量
- $t \in [0, 1]$: flow matching 的 denoising time（注意：这个 $t$ 是 inference 时的 denoising step，和 trajectory 的物理时间是两回事）
- $\mathbf{c}$: conditioning，包含 visual features $\mathbf{f}_{vis}$ 和 navigation command $\mathbf{n}$

训练 loss 是简单的 regression：

$$\mathcal{L}_{FM} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1} \| v_\theta(\mathbf{x}_t, t, \mathbf{c}) - u_t(\mathbf{x}_t | \mathbf{x}_1, \mathbf{x}_0) \|^2$$

其中 $\mathbf{x}_0 \sim \mathcal{N}(0, I)$ 是 noise，$\mathbf{x}_1$ 是真实 trajectory，$u_t$ 是条件 vector field（通常取 linear path $u_t = \mathbf{x}_1 - \mathbf{x}_0$）。

Flow matching 相比 diffusion 的优势：
- ODE 而非 SDE，inference 更 deterministic
- 不需要 noise schedule tuning
- 训练更稳定

### 3.3 三个 Size: S / B / L

VaVAM 有 Small、Base、Large 三个 size。作者发现 **B 在 closed-loop 上最好**（Figure 2），L 反而更差。这是一个反常的 scaling 现象——通常 foundation model 是越大越好。

可能的解释：L 模型 capacity 大，更容易 overfit OL 训练分布，导致 CL generalization 变差。这呼应了经典 ML 中的 "overfitting hurts OOD generalization" 现象。

---

## 4. Causal Confusion：最 Striking 的发现

Table 1 是这篇 paper 我个人觉得最有意思的实验：

| Frames Used | Route Completion | HD-Score |
|---|---|---|
| 1 | 0.5905 | 0.4100 |
| 2 | 0.1558 | 0.0915 |
| 4 | 0.0919 | 0.0515 |
| 8 | 0.1034 | 0.0430 |

**给模型看更多历史帧，性能断崖式下降。** 这违反直觉——通常认为多帧输入能提供更丰富的 context。

### 4.1 Causal Confusion 是什么

这个现象最早由 de Haan et al. NeurIPS 2019（https://arxiv.org/abs/1905.11179）系统描述。核心机制：

模型在 IL 训练中容易把 **effect 当成 cause**。

举个 driving 的例子：车在红灯前停下。Expert 在这一刻的 action 是 "保持停止"。但 expert 为什么保持停止？是因为看到了红灯（cause）。但是从 history 来看：
- 过去几帧车都在减速
- 当前帧车已经停止

模型如果同时看 history 和 current frame，可能学到的 spurious rule 是："过去几帧在减速 + 当前静止 → 保持静止"。这看起来对，但模型没学到真正的 cause 是红灯。一旦在 CL 中遇到 OOD 情况（e.g., 绿灯但前面有缓行），模型会错误地保持停止。

### 4.2 为什么单帧反而好

单帧输入逼模型聚焦 current observation 的真正 causally-relevant features（红绿灯、行人、车道）。多帧输入给了模型"作弊"的机会——用 history 推断当前场景类型，学到 spurious correlation。

这个发现对 community 非常有启发：**Foundation model 的多模态输入可能反而是 curse**。多帧 history 听起来是 free lunch，但实际会让 IL 模型 causally confused。

可能的 follow-up 方向：
- Counterfactual training（e.g., IRIS, https://arxiv.org/abs/2203.00546）
- Causal attention masking
- 用 RL fine-tune 替代 BC，让模型通过 environment feedback 学到真正 cause

---

## 5. ECO 方法：两阶段 Post-processing

### 5.1 Stage 1: Trajectory Smoothing

VaVAM 输出的轨迹经常抖动（Figure 1 显示 ego vehicle 在 simulator 里 oscillate）。作者团队用一个 constrained optimization 来 smooth 它。

**目标函数**：

$$\text{Cost}(\mathbf{x}) = w_s \cdot S(\mathbf{x}) + w_c \cdot C(\mathbf{x}) + w_{dev} \cdot D(\mathbf{x}, \mathbf{x}^{(0)})$$

变量：
- $\mathbf{x} = [(x_0, y_0, \theta_0), ..., (x_{N-1}, y_{N-1}, \theta_{N-1})]$: 优化变量（要 refine 的轨迹）
- $\mathbf{x}^{(0)}$: VaVAM 原始输出
- $w_s = 3, w_c = 8, w_{dev} = 0.05$: 实验调出来的权重

三个 cost term：

**Smoothness term $S(\mathbf{x})$**：

$$S(\mathbf{x}) = \sum_{i=0}^{N-3} \| (x_{i+2}, y_{i+2}) - 2(x_{i+1}, y_{i+1}) + (x_i, y_i) \|^2 + \sum_{i=0}^{N-3} \text{wrap}[(\theta_{i+2} - \theta_{i+1}) - (\theta_{i+1} - \theta_i)]^2$$

变量：
- 下标 $i$ 表示 waypoint index，从 0 到 $N-3$（因为二阶差分需要 $i, i+1, i+2$ 三个点）
- 第一项是 position 的二阶差分，相当于离散 jerk
- 第二项是 yaw angle 的二阶差分，相当于 yaw acceleration
- `wrap[·]` 把角度映射到 $[-\pi, \pi]$，处理 $2\pi$ 周期性

直觉：二阶差分衡量"加速度的变化"，惩罚 jerk——开车的颠簸感主要来自 jerk 而非 acceleration 本身。

**Curvature term $C(\mathbf{x})$**：

$$C(\mathbf{x}) = \sum_{i=1}^{N-2} \mathbb{1}(\|\mathbf{v}_1\| > 0.1, \|\mathbf{v}_2\| > 0.1, |c_i| > 0.5) \cdot (|c_i| - 0.5)^2$$

其中：
- $\mathbf{v}_1 = (x_{i+1} - x_i, y_{i+1} - y_i)$: segment $i$ 的 displacement vector
- $\mathbf{v}_2 = (x_{i+2} - x_{i+1}, y_{i+2} - y_{i+1})$: segment $i+1$ 的 displacement vector
- $c_i = v_{1x} v_{2y} - v_{1y} v_{2x}$: 2D cross product，衡量两个连续 segment 的转向角
- Indicator function 三个条件：
  - segment 1 不是停滞（防止 zero-velocity 边界情况）
  - segment 2 不是停滞
  - curvature 超过 threshold 0.5
- Penalty 是 $(|c_i| - 0.5)^2$，hinge-like quadratic

直觉：cross product 衡量转弯的"急"。直线行驶时 $\mathbf{v}_1 \parallel \mathbf{v}_2$，$c_i = 0$。急转弯时 $|c_i|$ 大。Threshold 0.5 是说允许 moderate curve，但急转弯要被惩罚。

**Deviation term $D(\mathbf{x}, \mathbf{x}^{(0)})$**：

$$D(\mathbf{x}, \mathbf{x}^{(0)}) = \sum_{i=0}^{N-1} [(x_i - x_i^{(0)})^2 + (y_i - y_i^{(0)})^2 + (\theta_i - \theta_i^{(0)})^2]$$

简单的 L2 deviation，确保优化不会偏离原始预测太远。权重 $w_{dev} = 0.05$ 很小，意味着这是软约束——optimizer 有空间调整，但不能完全 discard VaVAM 的输出。

**约束**：
- First and last waypoints fixed（硬约束，对应 endpoint constrained 的名字）
- Kinematic limits: max speed, acceleration, yaw rate, yaw acceleration, jerk

**求解器**：L-BFGS-B（Zhu et al. 1997, https://dl.acm.org/doi/10.1145/279232.279231）。这是 limited-memory quasi-Newton 方法，适合 high-dimensional bound-constrained 问题。scipy.optimize 里有现成实现（https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html）。

**效果**（Table 3）：comfort score 从 0.1486 提升到 0.8014。HD-Score 从 0.2280 提升到 0.3623。

### 5.2 Stage 2: Trajectory Re-timing

Smoothing 之后的轨迹形状 OK 了，但时间分配还有问题。问题在于：VaVAM 在 OL training 下学得比较 conservative——前面慢悠悠，后面赶进度。这导致 ego vehicle 在 simulator 里 progress 不够，route completion 低。

Re-timing 用一个 warp function 把时间重新映射：

$$w(t) = 1 - (1-t)^{1+\alpha}$$

变量：
- $t \in [0, 1]$: normalized original time
- $w(t) \in [0, 1]$: normalized trajectory progress after warping
- $\alpha$: hyperparameter，控制 warp 强度

**性质分析**：
- $\alpha = 0$: $w(t) = t$，no warp
- $\alpha > 0$: $w(t) > t$ for $t \in (0, 1)$，progress 比 time 快
- 一阶导数 $\frac{dw}{dt} = (1+\alpha)(1-t)^\alpha$：初始 slope 是 $(1+\alpha)$，末端 slope 接近 0
- 二阶导数 $\frac{d^2w}{dt^2} = -\alpha(1+\alpha)(1-t)^{\alpha-1} < 0$：concave function

效果上：
- 早期 progress 加速 → ego vehicle 更早到达 endpoint 附近
- 末期 progress 减速 → 留出 braking margin，避免冲撞

经验最佳 $\alpha = 0.6$（Figure 5）。Route Completion 从 0.5131 提升到 0.5905。

### 5.3 整个 Pipeline

```
VaVAM raw output (N waypoints, 3s horizon)
        ↓
[Stage 1: L-BFGS-B Optimization]
   - Fix first & last waypoint
   - Optimize middle N-2 waypoints
   - Subject to kinematic constraints
   - Minimize w_s*S + w_c*C + w_dev*D
        ↓
[Stage 2: Re-timing Warp]
   - Apply w(t) = 1 - (1-t)^(1+α), α=0.6
   - Redistribute waypoints along arc length
        ↓
Final Trajectory → HUGSIM Execution
```

---

## 6. HUGSIM 和评估 Metric

### 6.1 HUGSIM 是什么

HUGSIM（https://arxiv.org/abs/2412.01718）是基于 3D Gaussian Splatting 的 photorealistic closed-loop simulator。和 CARLA 这种 synthetic simulator 不同：
- 用真实 driving scene 重建（NeRF / 3DGS）
- Photo-realistic rendering
- Closed-loop：ego vehicle 的 action 影响后续渲染

49 scenes / 345 scenarios，每个 scene 多个 difficulty level（外部 agent 的 aggressiveness 不同）。

### 6.2 HD-Score 公式

$$\text{HD-Score}_t = \left( \prod_{m \in \{NC, DAC\}} \text{score}_m \right) \times \left( \frac{\sum_{w \in \{TTC, COM\}} \text{weight}_w \times \text{score}_w}{\sum_{w \in \{TTC, COM\}} \text{weight}_w} \right)$$

$$\text{HD-Score} = R_c \times \frac{\sum_{t=0}^{T} \text{HD-Score}_t}{T}$$

变量：
- $R_c \in [0, 1]$: Route Completion
- NC (No Collision): 0/1 binary
- DAC (Drivable Area Compliance): 0/1 binary
- TTC (Time-to-Collision): continuous safety metric
- COM (Comfort): comfort metric
- $T$: trajectory 总时长
- $t$: 时间 index

**乘性 vs 加性的设计**：
- NC, DAC 是 hard safety constraints——一旦违反，整段轨迹 score 接近 0
- TTC, COM 是 soft comfort constraints——可以 trade-off，用 weighted average

这种 metric 来自 nuPlan（https://arxiv.org/abs/2106.11810），被 NAVSIM、Bench2Drive（https://arxiv.org/abs/2406.07542）继承。

---

## 7. 实验数据深度分析

### 7.1 主榜（Table 3）

| Model | Route Completion | HD-Score |
|---|---|---|
| VaVAM-B-ECO (1st Place) | 0.5905 | 0.4190 |
| Team NVIDIA (2nd Place) | 0.4601 | 0.4012 |
| LTF (Official Baseline) | 0.3449 | 0.2182 |
| VaVAM-B (Our Baseline) | 0.4917 | 0.2280 |
| VaVAM-B + Smoothing | 0.5131 | 0.3623 |

**逐步贡献**：
- VaVAM-B → + Smoothing: HD-Score +58.9% (主要来自 comfort 0.1486→0.8014)
- + Smoothing → + Re-timing: Route Completion +15.1%, HD-Score +15.6%

**vs Team NVIDIA（第二名）**：
- HD-Score 接近（0.4190 vs 0.4012）
- 但 Route Completion 差距大（0.5905 vs 0.4601）
- ECO 更 aggressive（走得更远），NVIDIA 更 conservative（safety 更好但走得少）
- 如果 metric 把 NC, DAC 的乘性权重调更严，NVIDIA 可能反超

**vs LTF baseline**：
- LTF 是 NAVSIM 官方 baseline（基于 TransFuser，https://arxiv.org/abs/2110.09028）
- 训练数据 OpenScene（nuPlan 子集）远小于 VaVAM 的 OpenDV-2k + nuPlan + nuScenes
- LTF 是 deterministic policy，VaVAM 是 generative（flow matching），后者能 better handle multi-modal driving

### 7.2 Limitations

Paper 自己提到：
- 不能 consistently 减速/停止面对 oncoming agents（紧急制动差）
- Generalization gap：VaVAM action head 训练在 nuPlan，但 HUGSIM 用 Waymo dataset（https://arxiv.org/abs/1912.04838）
- 即使 comfort score 高，轨迹仍可能看起来 erratic

---

## 8. 我个人觉得有意思的延伸联想

### 8.1 把 ECO 反灌进训练

现在 ECO 是 inference-time post-processing。但 deviation cost $D(\mathbf{x}, \mathbf{x}^{(0)})$ 把 model output 当 prior，反过来可以：

$$\mathcal{L}_{ECO-aware} = \| f_\theta(\mathbf{c}) - \text{ECO}(f_\theta(\mathbf{c})) \|^2$$

让 model 直接学 refined output，省去 inference optimization。这相当于 self-distillation，让 model 内化 ECO 的 priors。

### 8.2 Multi-modal Trajectory Selection

VaVAM 是 generative model，可以 sample 多个 candidate trajectories。可以：
- Sample K 条
- 每条 ECO post-process
- 选 ECO cost 最小的

类似 Diffusion Planner（https://arxiv.org/abs/2311.18819）的 top-k selection。

### 8.3 Offline RL Fine-tune

Paper 引用了 Wagenmaker et al. 2025（https://arxiv.org/abs/2506.15799）和 Critic-Regularized Regression（CRR, https://arxiv.org/abs/2006.15134）。Offline RL 可以：
- 用 HUGSIM rollouts 作 replay buffer
- 不需要 expert labels，只需 reward signal
- HUGSIM 没 explicit reward，但可以从 HD-Score 反推

CQL（https://arxiv.org/abs/1910.00107）、IQL（https://arxiv.org/abs/2106.12151）都是候选。Diffusion Q-Learning（https://arxiv.org/abs/2308.10172）更直接适用 flow matching 架构。

### 8.4 Causal Confusion 的系统性研究

Table 1 是 single experiment，但启发很多问题：
- 是否所有 video-conditioned driving models 都有这现象？
- temporal attention vs spatial attention 各自的作用？
- 能否用 counterfactual data augmentation 解决？

### 8.5 Endpoint Reliability 的 further 利用

Paper 用 endpoint 做 anchor，但只 fix 它。可以更进一步：
- 训练时给 endpoint 一个 explicit loss bonus
- 推理时用 endpoint 做 planning horizon 的 anchor（类似 MPC 的 terminal cost）

### 8.6 与传统 MPC 的对比

ECO 的 trajectory smoothing 本质是 MPC 的 trajectory refinement。区别：
- 传统 MPC 从 scratch 优化，需要 cost function + dynamics model
- ECO 用 model output 作 prior，只 refine 不重建

类似 Optimo（https://arxiv.org/abs/2203.05712）、MPDM（https://arxiv.org/abs/1710.06038）的 sampling-based planning。

### 8.7 Diffusion Policy Refinement

Diffusion policy 也有 post-hoc refinement 工作（e.g., beam search in diffusion, https://arxiv.org/abs/2304.03262）。ECO 的 specificity 在于 autonomous driving 的 kinematic constraints 和 endpoint-anchored 设计。

---

## 9. 几个直觉性的 Takeaway

1. **Foundation model 训练好之后，输出还是要 polish**——OL training 学到 semantic understanding，但 CL execution 的细节（smoothness, timing）需要 explicit handling
2. **Multi-frame input 反而是 curse**——causal confusion 让模型走捷径学 spurious correlation
3. **Endpoint 是靠谱的 anchor，中间 waypoints 是 failure mode**——这个 insight 可以指导未来 loss design
4. **Simple post-processing 可以打败复杂方法**——ECO 只是 L-BFGS-B + 一个 warp function，但拿了第一
5. **HUGSIM 这种 photorealistic CL benchmark 是 game-changer**——以前被 OL metric 掩盖的问题被暴露
6. **Foundation model 越大未必越好**——VaVAM-L 在 CL 上反而比 B 差，这是反 scaling law 的现象

---

## 参考链接

- VaVAM/VaVIM: https://arxiv.org/abs/2502.15672
- HUGSIM: https://arxiv.org/abs/2412.01718
- Flow Matching (Lipman et al.): https://arxiv.org/abs/2210.02747
- DAgger: https://arxiv.org/abs/1011.0689
- Causal Confusion in IL: https://arxiv.org/abs/1905.11179
- nuPlan: https://arxiv.org/abs/2106.11810
- Bench2Drive: https://arxiv.org/abs/2406.07542
- TransFuser: https://arxiv.org/abs/2110.09028
- LTF / NAVSIM: https://github.com/autonomousvision/navsim
- Diffusion Planner: https://arxiv.org/abs/2311.18819
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- Waymo Open Dataset: https://arxiv.org/abs/1912.04838
- L-BFGS-B: https://dl.acm.org/doi/10.1145/279232.279231
- scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
- CQL: https://arxiv.org/abs/1910.00107
- IQL: https://arxiv.org/abs/2106.12151
- CRR: https://arxiv.org/abs/2006.15134
- Steering Diffusion Policy with Latent RL: https://arxiv.org/abs/2506.15799
- Diffusion Q-Learning: https://arxiv.org/abs/2308.10172
- IRIS (Causal): https://arxiv.org/abs/2203.00546
- Beam Search in Diffusion: https://arxiv.org/abs/2304.03262
- Optimo: https://arxiv.org/abs/2203.05712
- MPDM: https://arxiv.org/abs/1710.06038

---

# VaVAM-ECO Paper 深度解析

这篇 paper 是 ICCV 2025 RealADSim (HUGSIM) Challenge 第一名的工作，由 University of Toronto、Vector Institute 和 NVIDIA Research 合作完成。核心思想非常简洁但有效：**不要重新训练 foundation model，而是用 lightweight post-processing 修复 closed-loop 执行中的中间 waypoints**。下面我从多个层面展开技术细节。

---

## 1. 核心问题：OL/CL Gap 的具体形态

### 1.1 OL vs CL 的根本 misalignment

Open-loop behavior cloning (OL BC) 训练的 loss 通常是逐 waypoint 的 L2/L1 regression：

$$\mathcal{L}_{OL} = \sum_{t=1}^{T} \| \hat{\mathbf{x}}_t - \mathbf{x}_t^{GT} \|^2$$

其中 $\hat{\mathbf{x}}_t = (\hat{x}_t, \hat{y}_t, \hat{\theta}_t)$ 是模型预测的 vehicle state，$\mathbf{x}_t^{GT}$ 是 ground truth。这种 loss 在 OL 评估下表现良好，因为：
- 每一步状态都是从 expert trajectory 的 ground-truth previous state 推出来的
- 没有 error accumulation

但在 CL 执行中，step $t$ 的输入是 step $t-1$ 的 predicted state + simulator-rendered observation，errors 会复合放大 (compounding error, 参考 DAgger [Ross et al. 2011])。

### 1.2 Paper 的关键观察

Paper 通过 HUGSIM 上的 CL 评估，发现一个有趣的现象 (Figure 3)：
- **Endpoint (t=3s) 通常是 well-aligned 的**
- **中间 waypoints (t=1s-2.5s) 是 high-error 区域**
- 第一帧 (t=0.5s，实际被执行的) 也相对可靠

这个 error distribution 的 "U-shape" 启发了 ECO 的设计：**保留 endpoint 这个 anchor，调整中间 waypoints**。

直觉上：在 OL 训练中，endpoint 通常对应于 navigation command (turn left/right/straight) 的最终落点，被 navigation conditioning 强约束；而中间 waypoints 是 "free-form" 生成，更容易被 diffusion/flow matching 的 stochasticity 扰动。

参考：DAgger (https://arxiv.org/abs/1011.0689), Compounding Error 分析 (https://arxiv.org/abs/2103.01942)

---

## 2. VaVAM 基础架构回顾

VaVAM (https://arxiv.org/abs/2502.15672) 是基于 VaVIM 的 video-action foundation model。

### 2.1 VaVIM (Generative Video Model)

VaVIM 类似于 video diffusion / latent video models (e.g., Sora, Stable Video Diffusion)，在 internet-scale driving video 上预训练，学习 visual representations 和 future prediction。

### 2.2 VaVAM Action Head: Flow Matching

VaVAM 的 action prediction 不用 diffusion 而用 **flow matching** (Lipman et al., 2023, https://arxiv.org/abs/2210.02747)。Flow matching 学习一个 conditional vector field $v_\theta(\mathbf{x}_t, t, \mathbf{c})$，将一个 noise distribution $p_0$ (通常是 Gaussian) 流动到 data distribution $p_1$ (driving trajectories)：

$$\frac{d\mathbf{x}_t}{dt} = v_\theta(\mathbf{x}_t, t, \mathbf{c})$$

其中：
- $\mathbf{x}_t \in \mathbb{R}^{N \times 3}$：ego trajectory 的 waypoints，每个 waypoint 是 $(x, y, \theta)$
- $t \in [0, 1]$：flow matching 的时间参数 (注意：这里和 trajectory 的物理时间不同，是 denoising time)
- $\mathbf{c}$：conditioning，包含：
  - Visual features $\mathbf{f}_{vis}$ (来自 VaVIM)
  - Navigation command $\mathbf{n} \in \{\text{left, right, straight}\}$

Flow matching 相比 diffusion 的优势：
- 训练更稳定
- ODE 而非 SDE，inference 时不需要 stochastic noise scheduling
- Loss 是简单的 regression：

$$\mathcal{L}_{FM} = \mathbb{E}_{t, \mathbf{x}_0 \sim p_0, \mathbf{x}_1 \sim p_{data}} \| v_\theta(\mathbf{x}_t, t, \mathbf{c}) - u_t(\mathbf{x}_t | \mathbf{x}_1, \mathbf{x}_0) \|^2$$

其中 $u_t$ 是 conditional vector field (例如 linear path: $u_t = \mathbf{x}_1 - \mathbf{x}_0$)。

### 2.3 模型规模

VaVAM 有三个 size：S (Small)、B (Base)、L (Large)。Paper 发现 **B 在 CL 上最好** (Figure 2)，这是一个 scaling 的反常现象——更大的 L 反而更差。原因：L 模型更容易 overfit OL 分布，导致 CL generalization 变差。

---

## 3. Causal Confusion 现象 (Table 1)

这是这篇 paper 最 striking 的发现之一。Table 1 显示：

| Frames Used | Route Completion | HD-Score |
|---|---|---|
| 1 | 0.5905 | 0.4100 |
| 2 | 0.1558 | 0.0915 |
| 4 | 0.0919 | 0.0515 |
| 8 | 0.1034 | 0.0430 |

**单帧 → 多帧：性能断崖式下降。** 这是 classic causal confusion in imitation learning (de Haan et al., NeurIPS 2019, https://arxiv.org/abs/1905.11179) 的体现。

### 3.1 Causal Confusion 的机制

Causal confusion 的核心：IL 模型把 **effect 当成 cause**。在 driving 中：
- Expert 的动作 $a_t$ 通常取决于 history $(o_{t-k}, ..., o_t)$
- 但 expert 在某些状态下几乎不动 (e.g., 红灯前静止)，此时 $a_t \approx 0$ 主要由 "current stop state" 决定
- 模型如果同时看到 history 和 current frame，可能学到 "history 长 + 当前静止 → 仍静止" 这种 spurious correlation

更具体地，当 multi-frame 输入让模型可以 "推断" 当前是哪种场景 (e.g., 红灯停车 vs. 拥堵慢行)，它会学到错误因果链。单帧输入反而逼模型聚焦 current observation 的真正 causally-relevant features。

### 3.2 与现有 literature 的对比

- de Haan et al. (https://arxiv.org/abs/1905.11179) 最早在 IL 中正式定义 causal confusion
- Codeficients (https://arxiv.org/abs/2203.00546) 提出反事实方法
- 在 autonomous driving 中，UniAD 等也观察到类似现象

这个观察对 community 非常有启发：**Foundation model 的多模态输入可能反而是 curse**。

---

## 4. ECO 方法详解

ECO = Endpoint Constrained Optimization，两阶段 post-processing pipeline。

### 4.1 Stage 1: Trajectory Smoothing

**优化目标** (公式 2)：

$$\text{Cost}(\mathbf{x}) = w_s \cdot S(\mathbf{x}) + w_c \cdot C(\mathbf{x}) + w_{dev} \cdot D(\mathbf{x}, \mathbf{x}^{(0)})$$

其中 $\mathbf{x} = [(x_0, y_0, \theta_0), ..., (x_{N-1}, y_{N-1}, \theta_{N-1})]$ 是优化变量，$\mathbf{x}^{(0)}$ 是 VaVAM 原始输出。

**三个 cost terms 的细致分析**：

#### 4.1.1 Smoothness $S(\mathbf{x})$

$$S(\mathbf{x}) = \sum_{i=0}^{N-3} \| (x_{i+2}, y_{i+2}) - 2(x_{i+1}, y_{i+1}) + (x_i, y_i) \|^2_2 + \sum_{i=0}^{N-3} \left( \text{wrap}[(\theta_{i+2} - \theta_{i+1}) - (\theta_{i+1} - \theta_i)] \right)^2$$

- 第一项：position 的二阶差分 (discrete acceleration 的代理)，惩罚 jerk
- 第二项：yaw angle 的二阶差分，惩罚 yaw acceleration
- `wrap[·]` 函数把角度映射到 $[-\pi, \pi]$，处理 $2\pi$ periodicity
- 下标 $i$ 表示 waypoint index，$i \in [0, N-3]$ 是因为二阶差分需要 $i, i+1, i+2$ 三个点

#### 4.1.2 Curvature $C(\mathbf{x})$

$$C(\mathbf{x}) = \sum_{i=1}^{N-2} \mathbb{1}(\|\mathbf{v}_1\|_2 > 0.1, \|\mathbf{v}_2\|_2 > 0.1, |c_i| > 0.5) \cdot (|c_i| - 0.5)^2$$

其中：
- $\mathbf{v}_1 = (x_{i+1} - x_i, y_{i+1} - y_i)$: segment $i$ 的 displacement
- $\mathbf{v}_2 = (x_{i+2} - x_{i+1}, y_{i+2} - y_{i+1})$: segment $i+1$ 的 displacement
- $c_i = v_{1x} v_{2y} - v_{1y} v_{2x}$: signed 2D cross product，衡量 turn 的 sharpness
- Indicator function 三个条件：
  - $\|\mathbf{v}_1\| > 0.1$: segment 1 不是停滞
  - $\|\mathbf{v}_2\| > 0.1$: segment 2 不是停滞
  - $|c_i| > 0.5$: curvature 超过 threshold
- 惩罚是 $(|c_i| - 0.5)^2$，是一个 hinge-like quadratic penalty

直觉：cross product 衡量两个连续 segment 的转向角度。对于直线 $\mathbf{v}_1 \parallel \mathbf{v}_2$，$c_i = 0$。Turn 越急，$|c_i|$ 越大。

#### 4.1.3 Deviation $D(\mathbf{x}, \mathbf{x}^{(0)})$

$$D(\mathbf{x}, \mathbf{x}^{(0)}) = \sum_{i=0}^{N-1} \left[ (x_i - x_i^{(0)})^2 + (y_i - y_i^{(0)})^2 + (\theta_i - \theta_i^{(0)})^2 \right]$$

简单的 L2 deviation，确保优化不会偏离原始预测太远。

**权重选择**：$w_s = 3, w_c = 8, w_{dev} = 0.05$。$w_c$ 最大表明 curvature 是最关心的 safety 相关项；$w_{dev}$ 很小，意味着 deviation 是软约束。

**约束条件**：
- Fixed start and end waypoints (硬约束)
- Kinematic limits:
  - Maximum speed $v_{max}$
  - Maximum acceleration $a_{max}$
  - Maximum yaw rate $\dot{\theta}_{max}$
  - Maximum yaw acceleration $\ddot{\theta}_{max}$
  - Maximum jerk $j_{max}$

**求解器**：L-BFGS-B (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)，是 limited-memory quasi-Newton 方法，适合 high-dimensional bound-constrained 问题。Zhu et al. 1997 (https://dl.acm.org/doi/10.1145/279232.279231)。

#### 4.1.4 为什么用 Optimization 而不是 Learning?

直接学一个 smoother 网络需要：
- Pairs of (rough, smooth) trajectories
- 难以收集
- Domain-specific

而 optimization-based post-processing 是 model-agnostic，可以 plug-and-play 到任何 planner 输出。

### 4.2 Stage 2: Trajectory Re-timing

**核心 idea**：原始 trajectory 在时间上分布不均匀，可能 "晚到"。Re-timing 通过 warp function 让 ego vehicle 更早 progress，留出 braking margin。

**Warp function** (公式 3)：

$$w(t) = 1 - (1-t)^{1+\alpha}$$

其中：
- $t \in [0, 1]$: normalized time (原始)
- $w(t) \in [0, 1]$: normalized trajectory progress (warped)
- $\alpha$: hyperparameter，控制 warp 强度

**性质分析**：
- $\alpha = 0$: $w(t) = t$，identity (no warp)
- $\alpha > 0$: $w(t) > t$ for $t \in (0, 1)$，i.e., progress 比 time 快
- $\frac{dw}{dt} = (1+\alpha)(1-t)^\alpha$：初始 slope $(1+\alpha)$ 大，末端 slope $\to 0$
- $\frac{d^2w}{dt^2} = -\alpha(1+\alpha)(1-t)^{\alpha-1} < 0$：concave function

效果：
- 早期 progress 加快 → ego 更早到达 endpoint 附近
- 末期 progress 慢 → 有 braking margin

**经验最佳 $\alpha = 0.6$** (Figure 5)。

### 4.3 ECO Pipeline 完整流程

```
VaVAM Output (N waypoints, 3s horizon)
        ↓
[Stage 1: L-BFGS-B Optimization]
   - Fix first & last waypoint
   - Optimize middle N-2 waypoints
   - Subject to kinematic constraints
   - Minimize Cost = w_s*S + w_c*C + w_dev*D
        ↓
[Stage 2: Re-timing Warp]
   - Apply w(t) = 1 - (1-t)^(1+α), α=0.6
   - Redistribute waypoints along arc length
        ↓
Final Trajectory → HUGSIM Execution
```

---

## 5. HUGSIM Benchmark 细节

### 5.1 HUGSIM Simulator

HUGSIM (https://arxiv.org/abs/2412.01718) 是基于 3D Gaussian Splatting 的 photorealistic closed-loop simulator。与 CARLA 等 synthetic simulator 不同：
- Real-world scenes reconstruction (NeRF / 3DGS)
- Photo-realistic rendering
- Closed-loop execution (ego vehicle 动作影响 environment)

支持 49 scenes / 345 scenarios，每个 scene 多个 difficulty levels。

### 5.2 HD-Score Metric (公式 1)

$$\text{HD-Score}_t = \left( \prod_{m \in \{NC, DAC\}} \text{score}_m \right) \times \left( \frac{\sum_{w \in \{TTC, COM\}} \text{weight}_w \times \text{score}_w}{\sum_{w \in \{TTC, COM\}} \text{weight}_w} \right)$$

$$\text{HD-Score} = R_c \times \frac{\sum_{t=0}^{T} \text{HD-Score}_t}{T}$$

**变量解释**：
- $R_c \in [0, 1]$: Route Completion (走的路程 / 参考路程)
- NC (No Collision): 碰撞指标 (0/1)，**乘性**——一旦碰撞，整段 trajectory 的 HD-Score 接近 0
- DAC (Drivable Area Compliance): 是否在可行驶区域内 (0/1)，乘性
- TTC (Time-to-Collision): 安全 metric，**加性** (有 weight)
- COM (Comfort): comfort metric (jerk, lateral accel 等)，加性
- $T$: trajectory 总时长
- $t$: 时间 index

**乘性 vs 加性的设计哲学**：
- NC, DAC 是 hard safety constraints——一旦违反，整段 trajectory 应当被严重惩罚 (close to 0)
- TTC, COM 是 soft comfort constraints——可以 trade-off，所以用 weighted average

这种 metric 设计源于 nuPlan (https://arxiv.org/abs/2106.11810)，被 NAVSIM、Bench2Drive (https://arxiv.org/abs/2406.07542) 继承。

---

## 6. 实验结果分析 (Table 3)

| Model | Route Completion | HD-Score |
|---|---|---|
| VaVAM-B-ECO (1st) | 0.5905 | 0.4190 |
| Team NVIDIA (2nd) | 0.4601 | 0.4012 |
| LTF (Baseline) | 0.3449 | 0.2182 |
| VaVAM-B (Our baseline) | 0.4917 | 0.2280 |
| VaVAM-B + Smoothing | 0.5131 | 0.3623 |

### 6.1 关键 ablation 解读

**VaVAM-B → VaVAM-B + Smoothing**：
- Route Completion: 0.4917 → 0.5131 (+4.4%)
- HD-Score: 0.2280 → 0.3623 (+58.9%)

HD-Score 大幅提升主要来自 COM (comfort) score 提升 (0.1486 → 0.8014，paper 明确提及)。Route Completion 也小幅提升，因为 smoother trajectories 减少了 simulator 执行中的 deviation。

**VaVAM-B + Smoothing → VaVAM-B-ECO (加 Re-timing)**：
- Route Completion: 0.5131 → 0.5905 (+15.1%)
- HD-Score: 0.3623 → 0.4190 (+15.6%)

Re-timing 主要提升 Route Completion——这是合理的，因为 warp function 让 ego 更早 progress。

### 6.2 与 NVIDIA (2nd place) 对比

虽然 NVIDIA 的 HD-Score 接近 (0.4012 vs 0.4190)，但 Route Completion 差距大 (0.4601 vs 0.5905)。说明：
- VaVAM-ECO 更 aggressive (走得更远)
- NVIDIA 方法更 conservative 但 safety 更好

如果 HD-Score 公式中乘性项 (NC, DAC) 更严格，NVIDIA 可能反超。这说明 ECO 的 risk-reward trade-off 比较激进。

### 6.3 vs LTF Baseline

LTF (Latent TransFuser, https://arxiv.org/abs/2312.10017) 是 NAVSIM 官方 baseline，参数量和 VaVAM-B 相近但训练数据更少。LTF 落后的主要原因是：
- 训练数据 OpenScene (nuPlan 子集) 远小于 OpenDV-2k + nuPlan + nuScenes
- LTF 是 deterministic policy，VaVAM 是 generative (flow matching)，后者 better handles multi-modal driving

---

## 7. Open Questions & Future Directions

### 7.1 Paper 自己提到的 limitations

1. **Braking/Stopping 缺陷**：ECO 仍不能 consistently 减速或停止面对 oncoming agents
2. **Generalization gap**：VaVAM action head 训练在 nuPlan，但 HUGSIM 用 Waymo dataset (https://arxiv.org/abs/1912.04838)
3. **Erratic trajectories**：即使 comfort score 高，轨迹仍可能看起来 erratic

### 7.2 我个人认为的延伸方向

**A. Offline RL for closed-loop refinement**

Paper 引用了 Wagenmaker et al. 2025 (https://arxiv.org/abs/2506.15799) 和 Wang et al. Critic-regularized regression (CRR, https://arxiv.org/abs/2006.15134)。Offline RL 可以：
- 用 HUGSIM rollouts 作为 replay buffer
- 不需要 expert labels，只需 reward signal
- 但 HUGSIM 没有 explicit reward function——需要从 HD-Score 反推 reward

CQL (Conservative Q-Learning, https://arxiv.org/abs/1910.00107)、IQL (https://arxiv.org/abs/2106.12151) 都是候选。Diffusion policy + RL (e.g., Diffusion Q-Learning, https://arxiv.org/abs/2308.10172) 更直接适用 VaVAM 的 flow matching 架构。

**B. Endpoint Constrained Training (而非 post-processing)**

ECO 是 inference-time post-processing，但思路可以 backprop 进训练。具体地：
- 在 BC loss 之外，添加一个 ECO-aware loss：
$$\mathcal{L} = \mathcal{L}_{BC} + \lambda \cdot \mathcal{L}_{smooth} + \mu \cdot \mathcal{L}_{curv}$$
- 这样模型直接输出 smooth trajectories，省去 inference-time optimization

**C. Multi-modal Endpoint Prediction**

VaVAM 的 flow matching 实际上可以输出 multi-modal trajectories (sample 多次)。可以：
- Sample K 个 candidate trajectories
- 用 ECO post-process 每一个
- 选 ECO cost 最小的作为最终输出

类似 Diffusion Planner (https://arxiv.org/abs/2311.18819) 的 top-k selection。

**D. Causal Confusion 的进一步研究**

Table 1 的 finding 非常 striking 但 paper 没深挖。可能的延伸：
- 是否所有 driving foundation models 都有这个现象？
- 是否与 training data 的 temporal correlation 有关？
- 能否用 counterfactual training (e.g., IRIS, https://arxiv.org/abs/2203.00546) 解决？

**E. Trajectory Diffusion 的 OL/CL Gap**

VaVAM 用 flow matching，但 diffusion-based planners (Diffusion Policy, https://arxiv.org/abs/2303.04137; Diffusion Planner, https://arxiv.org/abs/2311.18819) 也有类似问题。ECO 的思路可以 transfer 到这些方法上。

---

## 8. 与 Related Work 的对比

### 8.1 与 Motion Planning 的传统方法

ECO 的 trajectory smoothing + re-timing 本质上是 MPC (Model Predictive Control) 中的 trajectory refinement。区别：
- 传统 MPC：从 scratch 优化，需要 cost function 和 dynamics model
- ECO：refine 已有 trajectory，用 model 的 output 作为 prior

类似工作：
- Optimo (https://arxiv.org/abs/2203.05712): optimization-based planning
- MPDM (https://arxiv.org/abs/1710.06038): sampling-based planning

### 8.2 与 Diffusion Policy Refinement 的对比

Diffusion policy 也有 post-hoc refinement 工作：
- Beam Search in Diffusion (https://arxiv.org/abs/2304.03262)
- Diffusion Policy with Spatial Refinement

ECO 的 specificity 在于：
- 针对 autonomous driving 的 kinematic constraints
- Endpoint-anchored 设计 (利用 BC 训练的 endpoint reliability)

### 8.3 与 LTF / TransFuser 系列对比

TransFuser (https://arxiv.org/abs/2110.09028) 用 multi-modal transformer fusion，LTF 替换 LiDAR 为 latent embedding。两者都是 deterministic，缺少 multi-modal modeling 能力。VaVAM 的 generative formulation 在这方面有本质优势。

---

## 9. 关键 Takeaways

1. **Foundation model 的 OL training ≠ CL 执行能力**——但通过 lightweight post-processing 可以大幅弥补
2. **Causal confusion 在 multi-frame driving models 中严重**——single-frame 反而更鲁棒
3. **Endpoint 是 reliable anchor，中间 waypoints 是 failure mode**——这个 insight 可以指导未来 training loss 设计
4. **Trajectory refinement + re-timing 是 simple 但 effective 的工具**——值得在任何 generative driving model 上尝试
5. **HUGSIM 等 photorealistic CL benchmarks 是 community 的 game-changer**——以前被 OL metric 掩盖的问题现在被暴露

---

## 10. 个人 thoughts

如果让我做 follow-up，最 promising 的方向是 **将 ECO 的 insight 反灌到训练**：

- 当前 ECO 的 deviation cost $D(\mathbf{x}, \mathbf{x}^{(0)})$ 把 model output 当 prior
- 可以反过来：在训练时，把 ECO-refined trajectory 当作 target，让 model 直接学 refined output
- 即 $\mathcal{L}_{ECO-aware} = \|f_\theta(c) - \text{ECO}(f_\theta(c))\|^2$

这相当于 self-distillation，让 model 内化 ECO 的 priors。

另一个方向是 **Causal Confusion 的系统性研究**。Table 1 是一个 single experiment，但启发很多问题：
- Is this unique to VaVAM, or general to all video-conditioned driving models?
- What's the role of temporal attention vs. spatial attention in causal confusion?
- Can we design training data augmentation to break spurious correlations?

最后，**flow matching 的 closed-loop behavior** 仍是 under-studied。理论上 flow matching 比 diffusion 更 deterministic (ODE)，但实际 closed-loop 行为如何？这是下一个 paper 的方向。

---

## 参考链接

- VaVAM/VaVIM: https://arxiv.org/abs/2502.15672
- HUGSIM: https://arxiv.org/abs/2412.01718
- Flow Matching: https://arxiv.org/abs/2210.02747
- L-BFGS-B: https://dl.acm.org/doi/10.1145/279232.279231
- Causal Confusion in IL: https://arxiv.org/abs/1905.11179
- DAgger: https://arxiv.org/abs/1011.0689
- nuPlan: https://arxiv.org/abs/2106.11810
- Bench2Drive: https://arxiv.org/abs/2406.07542
- TransFuser: https://arxiv.org/abs/2110.09028
- Diffusion Planner: https://arxiv.org/abs/2311.18819
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- Waymo Open Dataset: https://arxiv.org/abs/1912.04838
- CQL: https://arxiv.org/abs/1910.00107
- IQL: https://arxiv.org/abs/2106.12151
- Critic-Regularized Regression: https://arxiv.org/abs/2006.15134
- Steering Diffusion Policy with Latent Space RL: https://arxiv.org/abs/2506.15799
