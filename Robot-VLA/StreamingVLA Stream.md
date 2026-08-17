---
source_pdf: StreamingVLA Stream.pdf
paper_sha256: 577598639348adf01277b414009762668a96b3ea9ab9a790e75a9d5033adf38f
processed_at: '2026-08-12T11:18:05-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# StreamingVLA 人话版

Andrej，咱们换个聊法，像在白板上画图那样讲。

## 一句话概括

机器人现在干活的时候会"卡壳"——动两下停三秒，再动两下又停三秒。这篇 paper 让它从头到尾流畅地动下去，方法是把"看-想-动"三步从串行改成 **streaming 流水线**，让它们像 LLM inference 里的 pipeline parallelism 一样 overlap 起来跑。

## 问题到底长啥样

π₀.₅ 这种 VLA 模型执行一个任务的 timeline 长这样：

```
[看 100ms][想 100ms][动 动 动 动 动 200ms][看 100ms][想 100ms][动...]
                          ↑↑↑                  ↑↑↑
                          机器人在干活          机器人发呆中
```

中间那段"看+想"的时间，机器人 **完全静止**，停在那里等大脑算完下一批动作。这个 gap 叫 **halting time**，实测大概 232ms，跟执行动作的时间差不多长。

为啥会这样？因为 π₀.₅ 用了 **action chunking**：每次观察一次，生成一坨 10 个 action，执行前 5 个，然后再观察再生成。这是为了摊薄"看+想"的固定成本。

```
T_action = (T_观察 + T_想 + T_动) / h
T_halt = T_观察 + T_想
```

你想，h=5 的时候每个 action 摊到 74ms，h=10 摊到 50ms。chunk 越大越省，但 halting 还是 232ms 一动不动。

**Model compression 路线**（quantization、token pruning）能缩短 T_观察 和 T_想，但 T_halt = T_观察 + T_想 这个公式本身改不掉——只要还是串行，halting 就摆在那里。

所以作者换了个思路：**让阶段之间 overlap**。

## 两个 overlap 维度

定义两个 overlap 量：
- $O_{ge}$：action 生成 vs 执行 的重叠
- $O_{oe}$：observation vs 执行 的重叠

公式变成：

$$T_{\text{action}} = \frac{T_o + T_g + T_e - (O_{ge} + O_{oe})}{h}$$
$$T_{\text{halt}} = (T_o + T_g) - (O_{ge} + O_{oe})$$

只要 $O_{ge} + O_{oe}$ 够大，T_halt 就能压到接近 0。两个技术分别搞定这两个 overlap。

## 技术一：Action Flow Matching (搞 $O_{ge}$)

### 传统 chunking 为啥不能并行

Diffusion policy 的生成过程是：从噪声 $\mathcal{A}^0$ 出发，迭代去噪 10 步，每步都更新整个 chunk 的所有 token。**只有最后一步结束，所有 10 个 action 才同时确定**。中间任何一步拿出来的 action 都是 garbage。

所以你没法"生成一个执行一个"——必须等整个 chunk 算完。gen 和 exec 死锁。

那直接 $N_{\text{chunk}}=1$ 行不行？理论上行，但每个 action 都得跑完整 diffusion 迭代，cost 翻 10 倍。

### Flow Matching 的"换个角度看问题"

SFP ([Streaming Flow Policy](https://arxiv.org/abs/2505.21851)) 的关键 insight：**action 序列本身就是一条 flow 轨迹**。

传统 diffusion：t 是 denoising step（0 到 1 是噪声到数据），你要迭代很多步去噪。
Flow Matching：t 是 **action 在 trajectory 里的位置**（0 是第一个 action，1 是最后一个），你要预测的是一个 velocity field $v(x, t | o)$，沿着这个 field 走就走出整条轨迹。

公式 (2)：

$$v_\xi(x, t) = \dot{\xi}(t) - k(x - \xi(t))$$

- $\xi(t)$：ground-truth 轨迹在位置 t 的值
- $\dot{\xi}(t)$：轨迹的导数（速度）
- $x$：当前预测的 state
- $k$：拉回因子，x 偏离轨迹时把它拽回来（论文取 k=5）

初始分布是个窄高斯：$p^0(x) = \mathcal{N}(\xi(0), \sigma_0^2)$，$\sigma_0^2=0.16$。

神奇的性质（公式 3）：随着 t 走，分布始终是贴着 $\xi(t)$ 的高斯，方差指数衰减：

$$p(x | t) = \mathcal{N}(\xi(t), \sigma_0^2 e^{-2kt})$$

意思是：**越往后走，分布越紧贴真实轨迹**。

### 推理时怎么 streaming

公式 (6) 离散化：

$$x_T = x_0 + \sum_{t=0}^{T-1} v_\theta(x_t, t | o) \Delta t$$

每个 action：

$$a_t = x_{t+1} - x_t = v_\theta(x_t, t | o) \cdot \Delta t$$

**一次 forward pass 直接出一个 velocity，立刻变成一个 action，立刻可以执行**。不用迭代去噪！

那么 timeline 变成：

```
[g第1][g第2][g第3][g第4]...
[e第1][e第2][e第3][e第4]...
       ↑重叠↑
```

生成第 2 个 action 的同时执行第 1 个，完美 overlap。$O_{ge} = (h-1) \cdot \min(T_g/h, T_e/h)$，论文实测 162ms。

### 搬到大模型上翻车了

SFP 原版只在 PushT 这种小任务 + 小 MLP 模型上跑过。直接套到 π₀.₅ 上有两个坑：

**坑 1：Action Space ≠ Physical Space**

PushT 里 policy 直接输出 (x, y) 物理坐标，所以 state $x_t$ 就是物理状态，加法性天然成立。

LIBERO 用 OSC_POSE controller。policy 输出 end-effector pose delta，controller 再非线性变换到整个仿真环境状态。**action 累加 ≠ 物理状态变化**。

解决：搞两套 state。**action-space state**（policy 输出累积）和 **physical-space state**（环境真实状态）。Policy 只维护 action-space state：

$$x_{t+1} = x_t + v_\theta(x_t, t) \Delta t$$

训练时为了对齐，**预先把整条 trajectory 的 action-space state 全算好**，训练每个子轨迹时把对应的初始 state 喂进去。这就是 State-based Alignment (SA)。

**坑 2：Normalization 破坏加法性**

State-based modeling 的命根子是 $x_t + a_t = x_{t+1}$。大模型里到处是 normalization，原始公式 (8)：

$$a_t' = (a_t - q_{\min}) / \text{scale} \times 2 - 1$$

那个 `-1` 的 offset 项直接把加法性搞没了：$x_t' + a_t' \neq x_{t+1}'$。

解决方案简单粗暴（公式 9）：

$$a_t' = a_t / \text{scale}$$

去掉 offset，只留 scale。这就是 Normalization Modification (NM)。

消融实验里 NM 单独用只有 61.8%，因为没 SA 对齐语义错位。NM + SA 一起用冲到 97.1%。**NM 是必要条件，SA 是充分条件**。

## 技术二：Adaptive Early Observation (搞 $O_{oe}$)

### Early Observation 的逻辑

执行 action 期间，提前启动下一轮的 VLM 观察。这样观察和执行 overlap：

```
[观察下一帧          ]
[执行当前chunk剩余   ]
```

但有个问题：如果提前得太早，观察到的环境是"还没执行完当前 action"的状态，跟真正需要决策的状态不一致。

### 关键观察：Action Saliency 不一样

不是所有 action 都一样重要。看 Fig. 4 两个例子：

- **高 saliency**：抽屉刚被拉出来的那一帧，环境剧变。这一帧之前观察，看到的是关着的抽屉，决策全错。
- **低 saliency**：手臂在空中移动，环境几乎不变。提前 1-2 帧观察无所谓。

现有方法（[RTC](https://arxiv.org/abs/2506.07339)、[SmolVLA](https://arxiv.org/abs/2506.01844)、[VLASH](https://arxiv.org/abs/2512.01031)）**uniform 跳过最后 N 个 action**，碰运气。RTC 在 long-horizon 任务上从 92.4% 跌到 25.2%，惨。

### 怎么度量 saliency

**朴素方案**：action token 的 norm 大小。物理上 norm 大 = 动得多 = 可能重要。实验证明比 uniform 好一点，但还是很烂（ANAO: 90.0%）。

**StreamingVLA 方案**：直接测 action 对 **VLM feature** 的影响。因为 feature 是 action expert 的输入，feature 错了后面全错。

训个轻量 transformer predictor（Fig. 7）：
- 输入：early frame 的 ViT embedding
- 条件：剩余的 action sequence（DiT 那套 conditioning）
- 输出：residual embedding $\Delta\text{embedding}$
- 训练：MSE loss，让 $\text{emb}_{\text{early}} + \Delta\text{emb}$ 拟合 late frame 的 ground-truth embedding

推理时算 $\|\Delta\text{embedding}\|_2$，超过阈值 $\eta$ 就 **禁止** early observation。这个阈值靠 accuracy-latency trade-off 搜出来。

### Overhead

Predictor 每 10 个 action 调一次，每次 8-10ms，平均每个 action 0.8-1ms，占标准生成时间 18ms 的 5%。训练成本 3-4 GPU 小时，比 VLASH 的 full finetune 便宜太多。

## 实验亮点

主表 (Table 1) 核心数据：

| 方法 | Success | $T_{\text{action}}$ | $T_{\text{halt}}$ |
|------|---------|-----|------|
| π₀.₅ (h=5) | 96.9 | 74.5 | 232.3 |
| π₀.₅ (h=10) | 95.1 | 49.9 | 230.8 |
| RTC | 58.55 | 50.2 | 203.6 |
| SmolVLA | 95.8 | 51.1 | 180.7 |
| VLASH | 97.1 | 40.6 | - |
| **AFM only** | **97.1** | 33.7 | 76.1 |
| **AFM+AEO (full)** | **94.9** | **31.6** | **36.0** |

几个 intuition 点：

1. **AFM 单独性能反超 baseline**（97.1 vs 95.1）。Flow 建模本身比 chunk 去噪更结构化，temporal consistency 更好。

2. **RTC 灾难**：long-horizon 从 92.4% 跌到 25.2%。Uniform early observation 碰到关键 action 就废。

3. **AEO vs NEO**：同样 early observation，NEO 跳过 2 个性能 86.2%，AEO 平均跳过 1.4 个性能 94.9%。**自适应选择何时跳过比跳过多少更重要**。

4. **AEO vs ANAO**：两者平均跳过数都是 1.4，ANAO（norm-based）90.0%，AEO（feature-based）94.9%。**Feature-level saliency 比 action-norm saliency 准**。

5. **Runtime breakdown** (Appendix C.1)：
   - $O_{ge} = 162$ ms（占大头）
   - $O_{oe} = 41.58$ ms
   - 总 overlap 200ms+，所以 $T_{\text{halt}}$ 从 232ms 压到 36ms

6. **真机部署** (Appendix A)：Franka Panda pick-and-place，从 271ms 压到 171ms，约 1.59× 加速。

## 几点我的看法

**1. 本质是 LLM inference trick 的迁移**

Continuous batching、speculative decoding、pipeline parallelism——这些 LLM inference 里成熟的 trick，搬到 embodied AI 上就是 StreamingVLA 这套。embodied 这边因为 latency sensitivity 更高（机器人不能等），反而更需要这种 overlap 设计。

**2. Additivity 这个坑很深**

NM 这个"去掉 offset"的修改看起来 trivial，但它是整个方法 work 的基石。大模型里 LayerNorm、RMSNorm 随处可见，任何破坏线性可加性的操作都会让 state-based modeling 崩。这点对想做 world model / state space model 的人都是警示。你看 Mamba 这类 SSM 也很在意加法性。

**3. Action Saliency 概念可以推广**

"这个 action 会让 VLM 看到啥不同的东西"——这个 saliency 定义比 action norm 高明多了。本质上是问 **action 对未来决策的影响**，跟 active perception、attention mechanism、甚至 model-based RL 里的 value of information 都是一脉相承的。可以推到任何 partial observation 的系统。

**4. 潜在问题**

- **离散 action**：gripper open/close 这种突变，flow 建模可能困难。Flow 假设轨迹连续可微。
- **闭环数据**：SA 需要预计算整条 trajectory 的 action states。如果数据是闭环反馈式的（每个 action 都依赖实时 observation），预计算就不成立。
- **Predictor 泛化**：threshold $\eta$ 是搜出来的，跨任务、跨 embodiment 泛化性存疑。可能需要任务特定的 calibration。
- **Horizon 固定**：h=10 是预设的，真正的 streaming 应该是变长的。这块还有空间。

**5. 跟相关工作的关系**

- vs [SFP](https://arxiv.org/abs/2505.21851)：把 SFP 从小模型小任务扩展到大 VLA 复杂 benchmark，SA 和 NM 是关键扩展。
- vs [VLASH](https://arxiv.org/abs/2512.01031)：VLASH 用 finetune 弥补 early observation 的 mismatch，StreamingVLA 用 predictor 跳过关键 action。前者重训，后者轻量化。
- vs [RTC](https://arxiv.org/abs/2506.07339)：RTC 也 early observation 但 uniform 跳过，遇到关键 action 就挂。
- vs model compression：StreamingVLA 是 orthogonal 的，可以叠加 quantization/pruning/token compression 一起用。

**6. 一个有意思的小细节**

公式 (2) 里的 $k$ 是 stabilizing factor，当 x 偏离轨迹时把它拉回来。这玩意儿有点像 Langevin dynamics 里的 drift term，也像 Kalman filter 里的 correction gain。本质都是"预测-修正"的思路。k=5 是 grid search 出来的，太大会过于刚性，太小会发散。这个超参的敏感性论文没细讲，但直觉上跟 trajectory 的平滑度有关。

## 参考链接

主 paper 与核心依赖：
- [StreamingVLA (本 paper)](https://arxiv.org/) - 论文本身
- [SFP: Streaming Flow Policy](https://arxiv.org/abs/2505.21851) - AFM 的源头
- [π₀](https://arxiv.org/abs/2410.24164) - base VLA 模型
- [π₀.₅](https://arxiv.org/abs/2504.16054) - 本文用的 base
- [OpenPI codebase](https://github.com/open-pi/open-pi) - 开源实现

对比方法：
- [RTC: Real-Time Chunking](https://arxiv.org/abs/2506.07339)
- [SmolVLA](https://arxiv.org/abs/2506.01844)
- [VLASH](https://arxiv.org/abs/2512.01031)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [ACT: Action Chunking Transformer](https://arxiv.org/abs/2304.13705)
- [DiT (Peebles & Xie)](https://arxiv.org/abs/2212.09748)

Benchmark 与相关：
- [LIBERO](https://libero-project.github.io/)
- [GR00T N1 (NVIDIA)](https://arxiv.org/abs/2503.14734)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [RT-2](https://arxiv.org/abs/2307.15818)

总结一句人话：这篇 paper 把"看一眼-想一下-动一下"的串行机器人控制，改成了"边看边想边动"的流水线。两个核心 trick——把 action 当 flow 轨迹走（一次 forward 出一个 action），和用小 predictor 判断啥时候能提前看下一眼（别在关键时刻提前看）。性能几乎不掉，2.4× 加速，6.5× 减少卡顿。思路清晰，工程落地扎实，跟 LLM inference optimization 的套路一脉相承。

---

# StreamingVLA 深度讲解

Andrej，这篇 paper 来自 Tsinghua University 和 Lenovo 的团队，核心解决 VLA (Vision-Language-Action) 模型在执行时的 **stuttering（卡顿）和 latency（延迟）** 问题。让我从 intuition 出发，逐层拆解。

## 1. 核心问题：VLA 执行的"管道阻塞"

现有的 VLA 模型（如 π₀.₅）采用三阶段同步流水线：

- **Observation stage**：VLM 处理图像+语言+机器人状态，生成 KV-cache（耗时 $T_o$）
- **Gen-action stage**：action expert 通过 diffusion/flow matching 生成一个 chunk 的 $N_{\text{chunk}}$ 个 action（耗时 $T_g$）
- **Execution stage**：执行 horizon $h$ 个 action（耗时 $h \cdot T_e / h$）

问题在于这三个阶段 **严格串行**，导致两次执行之间存在一个 **halting gap**：

$$T_{\text{halt}} = T_o + T_g \approx 232 \text{ ms}$$

这个等待时间几乎和整个执行阶段相当，机器人会明显"顿挫"。同时每个 action 的平均延迟为：

$$T_{\text{action}} = (T_o + T_g + T_e) / h$$

**Intuition**：与其压缩每个阶段的计算（model compression 路线），不如让不同阶段 **overlap**，实现异步 streaming 执行。定义两个 overlap 变量：
- $O_{ge}$：action generation 与 execution 的重叠
- $O_{oe}$：observation 与 execution 的重叠

优化目标变为：

$$T_{\text{action}} = \frac{(T_o + T_g + T_e) - (O_{ge} + O_{oe})}{h}$$

$$T_{\text{halt}} = (T_o + T_g) - (O_{ge} + O_{oe})$$

这就把问题转化为：**如何最大化 $O_{ge}$ 和 $O_{oe}$ 而不损失性能**。两个技术分别针对这两个 overlap。

## 2. Action Flow Matching (AFM)：实现 $O_{ge}$

### 2.1 为什么 Action Chunking 阻碍并行

传统 diffusion policy 对一个 chunk $\mathcal{A}_h^t$ 做迭代去噪，任何中间结果 $\mathcal{A}_h^t$ 都不包含完整的 action 信息——只有整个 denoising 过程结束，chunk 内所有 action token 才被确定。这意味着 **必须等整个 chunk 生成完才能执行第一个 action**，gen-action 和 execution 无法 overlap。

如果设 $N_{\text{chunk}}=1$（逐个生成），每个 action 都要完整 diffusion 迭代，cost 爆炸。

### 2.2 Flow Matching 的 Reformulation

借鉴 SFP (Streaming Flow Policy, [arXiv:2505.21851](https://arxiv.org/abs/2505.21851))，把 action trajectory $\xi(t)$ 直接当作 flow trajectory 来建模。关键公式 (2)：

$$v_\xi(x, t) = \dot{\xi}(t) - k(x - \xi(t)), \quad p_\xi^0(x) = \mathcal{N}(\xi(0), \sigma_0^2)$$

变量含义：
- $\xi(t)$：ground-truth action trajectory 在归一化时间 $t \in [0,1]$ 上的取值
- $\dot{\xi}(t)$：trajectory 的导数（速度）
- $x$：当前预测的 state
- $k$：stabilizing factor（论文取 $k=5$），当 $x$ 偏离 $\xi(t)$ 时把它拉回轨迹
- $\sigma_0^2$：初始分布方差（论文取 $0.16$）
- $p_\xi^0(x)$：初始分布，是以 $\xi(0)$ 为中心的小高斯

**关键性质**（公式 3）：

$$p_\xi(x | t) = \mathcal{N}(\xi(t), \sigma_0^2 e^{-2kt})$$

随着 $t$ 增大，方差指数衰减——distribution 始终是紧贴 $\xi(t)$ 的窄高斯。**重要**：这里的 $t$ 是 **action index 在 horizon 内的归一化位置**，不是 diffusion denoising timestep。这是与 diffusion policy 的本质区别。

### 2.3 训练目标

公式 (4) 的 conditional flow matching loss：

$$\mathcal{L}(v_\theta, p_\mathcal{D}) = \mathbb{E}_{(o,\xi) \sim p_\mathcal{D}} \mathbb{E}_{t \sim U[0,1]} \mathbb{E}_{x \sim p_\xi(x|t)} \| v_\theta(x, t | o) - v_\xi(x, t) \|_2^2$$

模型学习 velocity field $v_\theta(x, t | o)$，条件于 observation $o$。

### 2.4 推理时的 Streaming

公式 (5)(6) 离散化积分：

$$x_T = x_0 + \sum_{t=0}^{T-1} v_\theta(x_t, t | o) \Delta t, \quad T \in \{1, 2, \ldots, h\}$$

每个 action：

$$a_t = x_{t+1} - x_t = v_\theta(x_t, t | o) \Delta t$$

**每次 forward pass 只产生一个 velocity 估计，立刻转换成一个 action token 并执行**。这就实现了 action-by-action 的 streaming 生成，gen-action 与 execution 自然 overlap。

### 2.5 扩展到大模型的关键挑战

SFP 只在小模型（MLP-based diffusion policy）和简单任务（PushT）上验证过。直接搬到 π₀.₅ 这种大 VLA 上有两个 non-trivial 问题：

#### 挑战一：Action Space ≠ Physical Space

在 PushT 里，policy 直接输出 $(x, y)$ 物理坐标，state $x_t$ 就是物理状态。

但 LIBERO 用 OSC_POSE controller（Operational Space Controller），policy 输出的是 **end-effector pose delta**，再由 controller 非线性地转换成环境状态变化。Action 不能直接代表物理状态变化，原始 formulation 失效。

**解决方案**：引入 **action-space state** $\alpha$（区别于 physical-space state）。Generated action 用来更新 action-space state，而非 physical-space state：

$$x_{t+1} = x_t + v_\theta(x_t, t) \Delta t \quad \text{(公式 7)}$$

**训练时的对齐技巧**：原始 SFP 从 trajectory 采子轨迹训练，physical state 总是可得。但 extended formulation 中，action-space state 需要累积所有前序 action 才能得到——子轨迹内部累积会错位。所以 **预处理阶段预先计算完整 trajectory 的 action-space states**，训练每个子轨迹时输入对应的初始 action-space state。这就是 **State-based Alignment (SA)**。

#### 挑战二：Normalization 破坏 Additivity

State-based modeling 的核心是 $x_t + a_t = x_{t+1}$ 的可加性。小 MLP 天然满足。但大 VLA 的 normalization layer 会破坏它。

原始 normalization（公式 8）：

$$a_t' = (a_t - q_{\min}) / \text{scale} \times 2 - 1, \quad \text{scale} = q_{\max} - q_{\min}$$

offset 项导致 $x_t' + a_t' \neq x_{t+1}'$。

**解决方案**（Normalization Modification, NM，公式 9）：

$$a_t' = a_t / \text{scale}, \quad \text{scale} = q_{\max} - q_{\min}$$

两点修改：(1) action-space state 和 action 共享 normalization 统计量，保证 scale 一致；(2) 去掉 offset 项。

消融实验显示 **NM 是必要非充分条件**——单独 NM 只有 61.8% 成功率，必须配合 SA 才能达到 97.1%。NM 提供稳定的 scaling，SA 保证语义对齐。

## 3. Adaptive Early Observation (AEO)：实现 $O_{oe}$

### 3.1 Early Observation 的直觉

Action execution 期间提前启动下一轮的 VLM observation 处理，可以让 $O_{oe} > 0$。但如果提前得太早，observation 捕获的环境状态和实际执行完 action 后的状态不一致，导致性能崩塌。

### 3.2 Action Saliency 的关键观察

不同 action 对环境的改变程度差异巨大：
- **High saliency**：比如"抽出抽屉"这个 action 完成，环境剧变。提前观察会捕获错误状态。
- **Low saliency**：大多数 action（如移动中）环境几乎不变，提前观察无伤大雅。

现有方法（RTC [arXiv:2506.07339](https://arxiv.org/abs/2506.07339), SmolVLA [arXiv:2506.01844](https://arxiv.org/abs/2506.01844), VLASH [arXiv:2512.01031](https://arxiv.org/abs/2512.01031)）**uniformly 跳过最后几个 action**，可能恰好跳过高 saliency 的关键 action。

### 3.3 Saliency 度量的两种方案

**朴素方案**：用 action token 的 norm。物理上 norm 大→运动幅度大→可能 saliency 高。但实验证明这比 naive uniform 好有限，仍明显退化。

**StreamingVLA 方案**：直接度量 action 对 **observation feature** 的影响。因为 feature 是 action expert 的输入，feature 错了后续 action 全错。

具体做法（Fig. 7 架构）：
- 输入：early frame 的 ViT embedding（当前环境表征）
- 条件：pending action sequence（按 DiT [arXiv:2212.09748](https://arxiv.org/abs/2212.09748) 的 conditioning scheme）
- 输出：residual embedding $\Delta\text{embedding}$
- 训练目标：MSE loss，让 $\text{embedding}_{\text{early}} + \Delta\text{embedding}$ 拟合 late frame 的 ground-truth embedding

推理时用 $\|\Delta\text{embedding}\|_2$ 作为 saliency 指标，超过阈值 $\eta$ 则**禁止** early observation。阈值通过 accuracy-latency trade-off 搜索确定。

### 3.4 Overhead 分析

Predictor 每 $H$ 个 action 调用一次（$H=10$），每次 8-10 ms，平均到每个 action 约 0.8-1 ms，仅占标准生成时间（18 ms）的 5%。训练成本 3-4 GPU 小时，远低于 VLASH 的 full finetune。

## 4. 实验结果详解

### 4.1 主实验（Table 1）

Base model：π₀.₅-LIBERO，$N_{\text{chunk}}=10$，replan=5（默认 $h=5$）或无 replan（$h=10$）。

关键数据点：

| Method | Success Rate | $T_{\text{action}}$ (ms) | $T_{\text{halt}}$ (ms) |
|--------|-------------|------------------------|----------------------|
| π₀.₅ (h=5) | 96.9 | 74.5 | 232.3 |
| π₀.₅ (h=10) | 95.1 | 49.9 (1.49×) | 230.8 (1.01×) |
| RTC (d=1) | 58.55 | 50.2 | 203.6 |
| SmolVLA | 95.8 | 51.1 | 180.7 |
| VLASH | 97.1 | 40.6 (1.83×) | — |
| Temporal Ensembling | 90.0 | 279.0 (0.26×) | 231.6 |
| **StreamingVLA (AFM)** | **97.1** | 33.7 (2.21×) | 76.1 (3.05×) |
| StreamingVLA (AFM+NEO) | 86.2 | 29.3 (2.54×) | 23.0 (10.10×) |
| StreamingVLA (AFM+ANAO) | 90.0 | 30.775 (2.42×) | 27.75 (8.37×) |
| **StreamingVLA (AFM+AEO)** | **94.9** | **31.625 (2.36×)** | **36.0 (6.45×)** |

**几个关键 insight**：

1. **AFM 单独就提升性能**（97.1 vs 95.1）：把 action 建模成连续 flow 提供了更结构化的优化目标，temporal modeling 变好。

2. **RTC 灾难性失败**（58.55%）：early observation 在 long-horizon 任务上从 92.4% 跌到 25.2%。gradient-based correction 无法弥补信息丢失。

3. **NEO vs AEO 对比**：同样是 early observation，NEO（uniform 跳过 2 个）成功率 86.2%，AEO（自适应跳过平均 1.4 个）94.9%。**自适应选择何时跳过比跳过多少更重要**。

4. **ANAO vs AEO**：两者平均跳过数相同（1.4），但 ANAO（action norm based）90.0%，AEO（feature based）94.9%。说明 **feature-level saliency 比 action-norm saliency 准确**。

### 4.2 Runtime Breakdown（Appendix C.1）

对 π₀.₅：$O_{ge}=O_{oe}=0$，所以 $T_{\text{action}}=74.6$ ms, $T_{\text{halt}}=238$ ms。

对 StreamingVLA：
$$O_{ge} = (h-1) \min\left\{\frac{T_g}{h}, \frac{T_e}{h}\right\} = 162 \text{ ms}$$
$$O_{oe} = N_{eo} \cdot \frac{T_e}{h} = 41.58 \text{ ms}$$

计算得 $T_{\text{action}}=30.44$ ms (2.40×), $T_{\text{halt}}=34.42$ ms (6.71×)。

$O_{ge}$ 的公式很关键：gen-action 和 execution 都是 per-action 的，overlap 取决于**较慢的那个**。$(h-1)$ 因为第一个 action 必须先生成才能开始执行。

## 5. 算法细节（Appendix B）

**Training Algorithm 1** 核心步骤：
1. 采样 $(o_i, \alpha_i, \xi_i)$，其中 $\alpha_i$ 是预计算的初始 action-space state
2. 累积计算 action states：$\mathcal{A}_i[n] = \mathcal{A}_i[n-1] + \xi_i[n-1]$, $\mathcal{A}_i[0] = \alpha_i$
3. 采样 $t$，对应 index $T = \lfloor t \cdot h \rfloor$
4. 从 $\mathcal{N}(\mathcal{A}_i[T], \sigma_0^2 e^{-2kT/h})$ 采样 $x_T$
5. 最小化 $\|v_\xi(x_T, T/h) - v_\theta(x_T, T/h | o_i)\|^2$

**Inference Algorithm 2** 核心步骤：
- 异步 observation（`need_obs` flag）
- 异步 action 生成：$a = v_\theta(\alpha, T/h | o) \cdot \frac{1}{h}$
- 当 $h - T = N_{eo}$ 时检查 predictor $I(a) \leq \eta$ 决定是否 early observation
- 执行后更新 $\alpha \leftarrow \alpha + a$，$T \leftarrow T + 1$

**注意 $\Delta t = 1/h$**：因为 $t \in [0,1]$ 而 horizon 有 $h$ 步。

## 6. 真实世界部署（Appendix A）

Franka Panda + RGB camera，pick-and-place 任务，$H=8$。

- StreamingVLA：170.88 ms/action
- π₀.₅ baseline：271.49 ms/action

约 1.59× 加速，验证 sim-to-real 的迁移性。

## 7. 我的几点思考

**与 diffusion policy 的本质区别**：传统 diffusion/flow matching 是在 **action space** 上从 noise 到 data 的去噪；AFM 是在 **state space** 上沿 trajectory 的 flow。前者 $t$ 是 denoising step，后者 $t$ 是 action index。这个 reformulation 让 "生成一个 action" 退化成 "一次 forward pass"，代价是失去了 chunk 内 action 的全局协同优化。

**Additivity 的脆弱性**：NM 这个看似简单的修改（去掉 offset）是整个方法能 work 的基石。大模型里 normalization 随处可见，任何破坏线性可加性的操作都会让 state-based modeling 崩塌。这点值得在更广泛的世界模型 / 状态空间模型设计中警惕。

**Action Saliency 的物理意义**：用 feature-level $\Delta$embedding 而非 action norm 度量 saliency，本质上是在问"这个 action 会让 VLM 看到什么不同的东西"。这比物理运动幅度更接近"对决策的影响"。这个思路可以推广到任何部分观察 / 主动感知的系统。

**潜在局限**：
1. AFM 假设 action trajectory 是连续可微的 flow，对于突发性的离散切换（如 gripper open/close）可能建模困难。
2. Predictor 的 threshold $\eta$ 需要搜索，泛化性存疑。
3. State-based alignment 需要预计算完整 trajectory 的 action states，对闭环数据（带反馈的）处理可能复杂。

## 参考链接

- Streaming Flow Policy (SFP): https://arxiv.org/abs/2505.21851
- π₀: https://arxiv.org/abs/2410.24164
- π₀.₅: https://arxiv.org/abs/2504.16054
- RTC (Real-Time Chunking): https://arxiv.org/abs/2506.07339
- SmolVLA: https://arxiv.org/abs/2506.01844
- VLASH: https://arxiv.org/abs/2512.01031
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- DiT (Peebles & Xie): https://arxiv.org/abs/2212.09748
- LIBERO: NeurIPS 2023, https://libero-project.github.io/
- OpenPI codebase: https://github.com/open-pi/open-pi
- ACT (Action Chunking Transformer, Tony Zhao): https://arxiv.org/abs/2304.13705
- GR00T N1: https://arxiv.org/abs/2503.14734

这篇 paper 的核心贡献是把 VLA 从"chunk-based 同步管道"推进到"streaming 异步管道"，两个 overlap 维度的设计（$O_{ge}$ 通过 flow matching，$O_{oe}$ 通过 saliency-aware early observation）非常工程化且可组合，能和现有 model compression 技术（pruning/quantization/token compression）叠加。从 systems 角度看，这更像是把 LLM inference 中的 continuous batching / speculative decoding 思想迁移到 embodied AI 的 pipeline parallelism 上，思路很自然但执行细节（NM、SA、predictor）才是难点。
