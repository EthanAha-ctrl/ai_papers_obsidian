---
source_pdf: ReinboT.pdf
paper_sha256: e0729a34c74655c279a98faac6dece09b50250727498b1f6d45257e21bc694cc
processed_at: '2026-08-11T22:04:57-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

讲 ReinboT，你就把它当成给机器人装了个“势利眼”。机器人看 demonstration data，学会了专挑那些“未来收益高、动作丝滑”的优质行为去模仿，同时对那些笨拙但勉强成功的动作保持警惕。 

下面用最直白的人话，把这背后的技术细节拆解开，帮你 build 起来这个 mental model。

### 1. 痛点：好数据和烂数据混在一起怎么办？

假如你教机器人抓杯子，给了 100 个成功视频。其中 20 个动作行云流水，80 个动作哆哆嗦嗦、绕了弯路但最后也抓到了。传统的 VLA model 搞 imitation learning，本质就是在做 maximum likelihood estimation，它把这 100 个视频一视同仁地拟合。结果就是，机器人学出来的动作带着那 80 个劣质数据的“噪音”，变成了一个平庸的抓取动作。它分不清什么叫“好”，什么叫“坏”。

### 2. 打分系统：Dense Reward

为了区分好坏，ReinboT 设计了一套四维度的打分系统，也就是 dense reward，用来评判轨迹里每一步的质量。

**公式 (8): $r = \sum_{i=1}^{4} w_i r_i$**
*   **变量解释**：$r$ 是总 reward。$r_i$ 是第 $i$ 个维度的 reward。$w_i$ 是对应权重（论文里 $w_1=0.1, w_2=0.1, w_3=0.01, w_4=0.1$）。

我们来看看这四个维度怎么算的：

**维度 1: Sub-goal achievement ($r_1$)** —— “离当前阶段的小目标有多近？”
**公式 (4): $r_1 = e^{f_{\text{MSE}}(s_t, s_t^*)} + e^{f_{\text{MSE}}(o_t, o_t^*)} + e^{f_{\text{SSIM}}(o_t, o_t^*)} + e^{f_{\text{ORB}}(o_t, o_t^*)}$**
*   **变量解释**：$s_t$ 是当前时刻 $t$ 的 proprioception（关节状态）。$s_t^*$ 是 sub-goal 的 proprioception。$o_t$ 是当前图像，$o_t^*$ 是 sub-goal 图像。$f_{\text{MSE}}$ 是均方误差，$f_{\text{SSIM}}$ 是结构相似度指数，$f_{\text{ORB}}$ 是特征点匹配度。
*   **直觉**：这里用四个不同算法（状态差、像素差、视觉结构差、特征点差）算相似度。外面套了一层 $e$ 的指数函数，意思是只要稍微接近目标，分数就指数级飙升，强引导机器人靠近 sub-goal。

**维度 2: Task progress ($r_2$)** —— “在整个大任务里走到哪一步了？”
**公式 (5): $r_2 = \frac{n(s_t)}{|\{s^*\}|}$**
*   **变量解释**：$n(s_t)$ 是当前 state 所在的 sub-goal 序列编号（比如第 3 段）。$|\{s^*\}|$ 是总的 sub-goal 数量（比如 5 段）。
*   **直觉**：越往后走，分数越高。这本质上是一种 curriculum signal，告诉模型“大局进行到哪了”。

**维度 3: Behavior smoothness ($r_3$)** —— “动作抖不抖？”
**公式 (6): $r_3 = -|\dot{\mathbf{q}}|^2 - |\ddot{\mathbf{q}}|^2 - |\mathbf{a}_{t-1} - \mathbf{a}_t|^2 - |\mathbf{a}_{t-2} - 2\mathbf{a}_{t-1} + \mathbf{a}_t|^2$**
*   **变量解释**：$\dot{\mathbf{q}}$ 是关节速度，$\ddot{\mathbf{q}}$ 是关节加速度。$\mathbf{a}_t$ 是当前 action，$\mathbf{a}_{t-1}$ 是上一步 action。$|\cdot|^2$ 是平方和。
*   **直觉**：前面两项罚速度和加速度过大（费电且伤机器）。后面两项罚 action 的一阶差分和二阶差分过大（一阶差分大说明动作突变，二阶差分大说明 jerk 大，也就是猛地一抽）。因为全是负数，所以越平滑这个负数越接近 0。

**维度 4: Task completion ($r_4$)** —— “最后成没成？”
**公式 (7): $r_4 = \mathbb{1}\{\tau \text{ is successful}\}$**
*   **变量解释**：$\mathbb{1}\{\cdot\}$ 是指示函数，条件成立为 1 反之为 0。$\tau$ 是整条轨迹。
*   **直觉**：这是唯一一个 sparse 的维度，只要最后成了就给 1，锚定最终目标。

### 3. 核心魔法：Expectile Regression 怎么实现 Return Maximization？

打完分后，把从当前时刻 $t$ 到结束 $T$ 的所有分数加起来，就是未来总收益 ReturnToGo (RTG)。
**公式: $g_t = \sum_{j=t}^{T} r_j$**

传统 RL 的做法是学一个 Q-function 去估测这个未来收益，然后在 Transformer 架构里估测 Q-function 极其容易不稳定。ReinboT 的骚操作来了，它把预测 RTG 当成一种新的 data modality，丢给模型去预测，并且用了一个叫 **Expectile Regression** 的 loss 来逼模型“往高了猜”。

**公式 (9): $\mathcal{L}_{\text{RTG}} = \mathbb{E}_t[|m - \mathbb{1}(\Delta g < 0)| (\Delta g)^2]$**
**其中 $\Delta g = g_t - P_\varphi[\pi_\theta(\langle s, o\rangle_{t-h+1:t}, l)]$**

*   **变量拆解**：$m$ 是 expectile 参数，论文设为 0.9。$\mathbb{1}(\Delta g < 0)$ 是指示函数，当 $\Delta g < 0$（也就是模型预测的 RTG 大于真实 ground-truth RTG）时为 1。$g_t$ 是真实 RTG。$P_\varphi$ 是 RTG decoder。$\pi_\theta$ 是 GPT-style backbone。$\langle s, o\rangle_{t-h+1:t}$ 是长度为 $h$ 的历史状态和图像。$l$ 是 language instruction。

*   **直觉大揭秘**：你看那个 loss 的权重 $|m - \mathbb{1}(\Delta g < 0)|$。
    *   当模型**猜低了**（预测 RTG < 真实 RTG，即 $\Delta g > 0$）：指示函数为 0，权重变成 $|m - 0| = 0.9$。模型挨了重重的惩罚。
    *   当模型**猜高了**（预测 RTG > 真实 RTG，即 $\Delta g < 0$）：指示函数为 1，权重变成 $|m - 1| = |0.9 - 1| = 0.1$。模型只挨了轻轻的惩罚。

因为猜低了挨重骂，猜高了轻轻放过，模型在梯度下降的逼迫下，自然就学成了一个“乐观主义者”，总是倾向于预测比 ground-truth 更高的 RTG。这个预测出来的高 RTG，代表了在当前状态下，数据分布中“可能达到的最大收益”。然后模型根据这个高收益预期去生成 action，这就把 RL 里 maximize return 的精髓给实现了，且完全避开了 Q-learning 难训练的老大难问题。

### 4. 架构图解析：如何用 Single Inference 搞定一切？

前辈 Reinformer 也用了 expectile regression，但它在 inference 时要跑两次 forward pass：第一次预测 RTG，第二次把 RTG 塞进去预测 action。ReinboT 觉得太慢，搞了个模块化设计，一次搞定。

**架构前向传播三步曲：**

**Step 1: Backbone 提取特征**
**公式 (11): $h_{t:t+k-1}^{\text{RTG}}, h_{t:t+k-1}^{\text{action}} = \pi_\phi(l, o_{t-u+1:t}, s_{t-u+1:t})$**
*   **变量解释**：$\pi_\phi$ 是 backbone。输入语言 $l$、长度为 $u=10$ 的图像历史 $o$ 和状态历史 $s$。输出两个 token 的 hidden features：$h^{\text{RTG}}$ 对应 RTG token，$h^{\text{action}}$ 对应 ACTION token。$k$ 是预测 horizon（CALVIN 里是 5 步）。

**Step 2: RTG decoder 提取深层表征**
**公式 (12): $\hat{g}_{t:t+k-1}^{\text{hidden}} = P_\varphi(h_{t:t+k-1}^{\text{RTG}})$**
*   **变量解释**：$P_\varphi$ 是 RTG decoder。这里**非常关键**，输出的 $\hat{g}^{\text{hidden}}$ 并非最终的 RTG scalar 数值，而是 RTG decoder 最后一层的 hidden features。这个高维向量里包含了“为什么要预测这个收益”的全部 reasoning 过程信息。

**Step 3: Action decoder 拼接生成动作**
**公式 (13): $\hat{a}_{t:t+k-1} = P_\omega(h_{t:t+k-1}^{\text{action}}, \hat{g}_{t:t+k-1}^{\text{hidden}})$**
*   **变量解释**：$P_\omega$ 是 action decoder（基于 CVAE）。它直接把上一步拿到的 RTG hidden features $\hat{g}^{\text{hidden}}$ 和自己的 action features $h^{\text{action}}$ 拼在一起，生成未来 $k$ 步的 action $\hat{a}$。

这种设计妙就妙在，把 RTG 的“脑内推理过程”直接硬接到了 action 生成器上，一次网络前向就能完成“看图 → 想未来能赚多少 → 做出高收益动作”的全闭环。

### 5. 实验数据表深度剖析

来看 Table 1 在 CALVIN mixed-quality data 上的战果。重点看 Avg. Length (AL)，也就是平均能连贯完成多少条指令。

*   **GR-1 (annotated data) = 1.41** vs **GR-1 (所有数据) = 1.36**
    你发现没有，给 GR-1 喂更多数据，效果反而变差了！这直接证明了 imitation learning 在 mixed-quality 数据面前会拉胯，劣质数据把模型带偏了。
*   **ReinboT (sparse) = 1.74** vs **ReinboT (dense, single) = 1.90**
    把 sparse reward 换成 dense reward，AL 从 1.74 涨到 1.90。dense reward 提供了每一步的精细打分，监督信号强太多了。
*   **ReinboT (dense, single) = 1.90** vs **ReinboT (dense, full) = 2.26**
    single 是把四个维度的 reward 加起来求和预测一个 scalar RTG。full 是把四个维度的 RTG 分开预测。分开预测又涨了 20%！这非常符合直觉，保留各个维度的信息量远比揉成一个干瘪的数字强。
*   **ReinboT (dense, full) = 2.26** vs **RWR (dense, single) = 1.82**
    同样用 dense reward，ReinboT 大幅领先经典 offline RL 算法 RWR。RWR 靠 reweight 数据去优化，数据不均时容易崩；ReinboT 靠 expectile regression 的高维条件注入，稳且强。

### 6. 个人发散与 Intuition 沉淀

ReinboT 这套打法，底层逻辑极其优雅。它把 RL 里的 credit assignment 和 return maximization 问题，转化成了一个带有不对称惩罚的 supervised regression 问题。通过“偏心眼”的 loss function，强迫网络内化一种乐观主义的世界观。

这让我联想到大语言模型里的 RLHF，本质也是把人类偏好注入到 supervised 框架里。ReinboT 的 dense reward 就相当于一种自动化的 preference signal。而且它把 ReturnToGo 抬升为一种与 image、text 并列的 modality，这就开启了极大的想象空间。未来如果把这种 expectile regression 的 trick 搬到更大规模的 Open X-Embodiment 数据集上，或者干脆结合 Diffusion Policy 去做 action 生成，把那个 CVAE action decoder 换掉，说不定能炸出更强的火花。

总而言之，ReinboT 的精神内核是：与其悲观地平均所有的可能性，不如乐观地预测一个最好的未来，并为之采取行动。

### Web Links for Reference
*   [Decision Transformer (NeurIPS 2021)](https://arxiv.org/abs/2106.01345) - 把 RL 变成 sequence modeling 的祖师爷。
*   [Reinformer (ICML 2024)](https://proceedings.mlr.press/v235/zhuang24a.html) - Expectile regression 实现 max-return 的直接灵感来源。
*   [GR-1 (ICLR 2024)](https://openreview.net/forum?id=v2NJ2J8K3q) - ReinboT 用的 video pre-training backbone 基础。
*   [CALVIN Benchmark](https://calvinrobot.github.io/) - 测试 long-horizon manipulation 的经典 benchmark。
*   [Expectile (Wikipedia)](https://en.wikipedia.org/wiki/Expectile) - 理解不对称 least squares 的数学原理。
*   [CVAE (Original Paper)](https://arxiv.org/abs/1606.05908) - Action decoder 使用的条件变分自编码器结构。
*   [Q-Transformer](https://arxiv.org/abs/2310.10101) - 对比组，Google 的 autoregressive Q-learning VLA 方案。

---

# ReinboT: 详解 RL Return Maximization 如何赋能 VLA Model

## 1. Paper 核心问题与 Motivation

这篇 paper 要解决的核心问题是: 现有的 Vision-Language-Action (VLA) models 基于 imitation learning paradigm, 但 training data 的 quality 不均匀, 即使来自 successful demonstrations 的数据也存在 quality 差异。Imitation learning 只做 maximum likelihood estimation, 无法区分和充分利用 mixed-quality data。

ReinboT 的核心 insight 是: 将 reinforcement learning (RL) 中 "maximize cumulative return" 的 principle 集成到 end-to-end VLA model 中, 通过预测 dense return 来理解 data quality distribution, 从而 generate 更 robust 的 actions。

**Intuition**: 想象一个 robot learning to grasp cup 的场景。即使有 100 条 successful trajectories, 其中有的 smooth and efficient, 有的 jerky and slow。Imitation learning 把它们一视同仁做 maximum likelihood, 而 RL 的 return maximization 会自动 prefer 那些 high-return (smooth + efficient) 的 trajectories。ReinboT 的 dense reward 就是用来 quantify 这个 "quality" 的。

参考:
- [Decision Transformer (DT)](https://arxiv.org/abs/2106.01345) - RL as sequence modeling 的开创性工作
- [Reinformer](https://proceedings.mlr.press/v235/zhuang24a.html) - Max-return sequence modeling, ReinboT 的直接 inspiration
- [GR-1](https://openreview.net/forum?id=v2NJ2J8K3q) - ReinboT 的 backbone architecture basis
- [CALVIN benchmark](https://calvinrobot.github.io/) - 实验 dataset

---

## 2. Dense Reward Design - 4 Components

这是 paper 的第一个核心贡献。针对 long-horizon manipulation task, 他们设计了 4 个 reward component, 每个 capture 不同的 task characteristic。

### 2.1 Sub-goal Achievement ($r_1$)

首先需要理解 sub-goal division。他们用 heuristic method 把 long-horizon trajectory 分割成多个 sub-goal sequences。分割依据两个 constraint:
1. Joint velocities close to zero (robot 到达 pre-grasp pose 或 transition 到 new task phase)
2. Gripper state changes (grasping 或 releasing object)

$$r_1 = e^{f_{\text{MSE}}(s_t, s_t^*)} + e^{f_{\text{MSE}}(o_t, o_t^*)} + e^{f_{\text{SSIM}}(o_t, o_t^*)} + e^{f_{\text{ORB}}(o_t, o_t^*)}$$

**变量解释**:
- $s_t$: 当前 timestep $t$ 的 proprioception (robot joint states)
- $s_t^*$: sub-goal 的 proprioception (该 sub-goal sequence 的终点 proprioception)
- $o_t$: 当前 timestep $t$ 的 image observation
- $o_t^*$: sub-goal 的 image observation
- $f_{\text{MSE}}(\cdot, \cdot)$: Mean Square Error function, 衡量两个 vector 的直接差异
- $f_{\text{SSIM}}(\cdot, \cdot)$: [Structural Similarity Index](https://en.wikipedia.org/wiki/Structural_similarity), 衡量 image 的 visual quality similarity (考虑 luminance, contrast, structure)
- $f_{\text{ORB}}(\cdot, \cdot)$: [Oriented FAST and Rotated BRIEF](https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF) feature point matching similarity

**Intuition**: 这 4 个 metric 从不同角度衡量 "当前状态离 sub-goal 有多近":
- Proprioception MSE: robot 本身 joint 配置的接近度
- Image MSE: pixel-level 直接差异
- SSIM: 人类视觉感知层面的相似度 (比 MSE 更 robust to lighting)
- ORB: feature point level 的对应关系 (capture object 位置和 pose)

用 $e^{f(\cdot)}$ 而非 $f(\cdot)$ 是为了放大差异 (exponential amplification), 让接近 sub-goal 的 state 获得显著更高的 reward。

### 2.2 Task Progress ($r_2$)

$$r_2 = \frac{n(s_t)}{|\{s^*\}|}$$

**变量解释**:
- $n(s_t) \in \{1, 2, \ldots, |\{s^*\}|\}$: state $s_t$ 所在的 sub-goal sequence 编号
- $|\{s^*\}|$: 总 sub-goal sequence 数量

**Intuition**: 一个 long-horizon task 被分成比如 5 个 sub-goal sequences。在 sequence 1 的 state 得 $r_2 = 1/5 = 0.2$, 在 sequence 5 的 state 得 $r_2 = 5/5 = 1.0$。这 reward 越靠近 final goal 越大, 反映 "整体进度"。

### 2.3 Behavior Smoothness ($r_3$)

$$r_3 = -|\dot{\mathbf{q}}|^2 - |\ddot{\mathbf{q}}|^2 - |\mathbf{a}_{t-1} - \mathbf{a}_t|^2 - |\mathbf{a}_{t-2} - 2\mathbf{a}_{t-1} + \mathbf{a}_t|^2$$

**变量解释**:
- $\dot{\mathbf{q}}$: robot arm joint velocity vector
- $\ddot{\mathbf{q}}$: robot arm joint acceleration vector  
- $\mathbf{a}_t$: timestep $t$ 的 action
- $\mathbf{a}_{t-1}$: timestep $t-1$ 的 action
- $\mathbf{a}_{t-2}$: timestep $t-2$ 的 action
- $|\cdot|^2$: vector 的 squared L2 norm

**Intuition**: 这 4 项 penalty 分别惩罚:
- $|\dot{\mathbf{q}}|^2$: joint 速度过大 (energy consumption)
- $|\ddot{\mathbf{q}}|^2$: joint 加速度过大 (energy + jerk)
- $|\mathbf{a}_{t-1} - \mathbf{a}_t|^2$: action 的一阶差分 (action 突变)
- $|\mathbf{a}_{t-2} - 2\mathbf{a}_{t-1} + \mathbf{a}_t|^2$: action 的二阶差分 (action 变化率突变, 即 "jerk")

注意符号是负的, 所以这其实是 penalty。Smooth, energy-efficient 的 motion 得到 higher (less negative) reward。

### 2.4 Task Completion ($r_4$)

$$r_4 = \mathbb{1}\{\tau \text{ is successful}\}$$

**变量解释**:
- $\mathbb{1}\{\cdot\}$: binary indicator function, 条件成立返回 1, 否则返回 0
- $\tau$: 整个 trajectory
- $\tau \text{ is successful}$: trajectory 成功完成 language instruction

**Intuition**: 这是唯一的 sparse component, 只在整个 trajectory 成功时为 1。它 anchor 了整个 reward signal, 确保 "success" 仍是最高 priority。

### 2.5 Total Dense Reward

$$r = \sum_{i=1}^{4} w_i r_i$$

**变量解释**:
- $w_i$: reward weight for component $i$, 论文设置 $w_1=0.1, w_2=0.1, w_3=0.01, w_4=0.1$

**Intuition**: 注意 $w_3=0.01$ 比 others 小一个量级, 因为 $r_3$ 本身是无界的负数 penalty, 而其他 component 是 bounded positive。权重平衡确保各 component 数量级 comparable。

---

## 3. ReturnToGo (RTG) 与 Expectile Regression - 核心创新

### 3.1 ReturnToGo 定义

$$g_t = \sum_{j=t}^{T} r_j$$

**变量解释**:
- $g_t$: timestep $t$ 的 ReturnToGo
- $r_j$: timestep $j$ 的 dense reward (由上面的公式计算)
- $T$: trajectory 的 terminal timestep

**Intuition**: RTG 是从当前 timestep 到结束的累积 reward。它 encode 了 "从当前状态出发, 这条 trajectory 还能获得多少总收益"。

### 3.2 Expectile Regression Loss - Return Maximization 的核心

$$\mathcal{L}_{\text{RTG}} = \mathbb{E}_t\left[|m - \mathbb{1}(\Delta g < 0)| (\Delta g)^2\right]$$
$$\text{with } \Delta g = g_t - P_\varphi\left[\pi_\theta(\langle s, o\rangle_{t-h+1:t}, l)\right]$$

**变量解释**:
- $m \in (0, 1)$: expectile regression 的 hyperparameter, 论文用 $m=0.9$
- $\mathbb{1}(\Delta g < 0)$: indicator function, 当 $\Delta g < 0$ (预测 > ground-truth) 时为 1
- $\Delta g = g_t - \hat{g}_t$: ground-truth RTG $g_t$ 减去 predicted RTG $\hat{g}_t$
- $P_\varphi$: ReturnToGo decoder (参数 $\varphi$)
- $\pi_\theta$: backbone network (GPT-style transformer, 参数 $\theta$)
- $\langle s, o\rangle_{t-h+1:t}$: 从 timestep $t-h+1$ 到 $t$ 的 proprioception 和 image state 历史
- $l$: language instruction
- $h$: history window length (= 10)

**Expectile Regression 的 Intuition 深度解析**:

考虑 $\Delta g$ 的两种情况:

**Case 1: $\Delta g > 0$ (predicted $\hat{g}_t < g_t$, prediction 偏低)**
- $\mathbb{1}(\Delta g < 0) = 0$
- Loss weight = $|m - 0| = m = 0.9$
- Loss = $0.9 \cdot (\Delta g)^2$

**Case 2: $\Delta g < 0$ (predicted $\hat{g}_t > g_t$, prediction 偏高)**
- $\mathbb{1}(\Delta g < 0) = 1$  
- Loss weight = $|m - 1| = 1 - m = 0.1$
- Loss = $0.1 \cdot (\Delta g)^2

**关键 insight**: 当 prediction 偏高 ($\hat{g}_t > g_t$) 时, loss weight 只有 $0.1$; 当 prediction 偏低时, loss weight 是 $0.9$。这创建了一个 **asymmetric penalty**: 模型被 "鼓励" 预测一个偏高的 RTG。

**Degenerate cases**:
- $m = 0.5$: 两边 weight 相等, degenerate 成 MSE → pure imitation learning
- $m \to 1$: 极度 prefer 高 prediction → aggressive return maximization (但可能 over-optimize)
- $m = 0.9$: 论文最优设置, moderate return maximization

**为什么这实现 "return maximization"**: 模型学到的 RTG 分布会被 push 向 higher value, 而不是简单拟合 ground-truth。在 inference 时, 模型预测的 $\hat{g}_t$ 代表了 "在当前 state 下, data distribution 中可能达到的 maximum return"。这个 $\hat{g}_t$ 然后 condition action generation, 让 action 朝向 high-return region 偏移。

参考:
- [Expectile (Wikipedia)](https://en.wikipedia.org/wiki/Expectile)
- [Expectile regression 原始 paper](https://www.jstor.org/stable/2526293)

### 3.3 与 Reinformer 的对比

Reinformer 用相同的 expectile regression idea, 但需要两次 inference:
$$\begin{cases} \hat{g}_t = \pi(\langle s, g, a\rangle_{t-h:t-1}, s_t) \\ \hat{a}_t = \pi(\langle s, g, a\rangle_{t-h:t-1}, s_t, \hat{g}_t) \end{cases}$$

**变量解释**: 第一次 inference 预测 $\hat{g}_t$, 第二次用 $\hat{g}_t$ 作为输入预测 $\hat{a}_t$。

ReinboT 通过 modular design 实现 **single inference**, 将在 architecture 部分详述。

---

## 4. ReinboT Architecture - 详解

### 4.1 Overall Architecture (Figure 1)

```
Inputs:
  - Language instruction l → CLIP encoder → 768-dim language features
  - Image observation o → ViT + Perceiver Resampler → 768-dim visual features
  - Proprioception s → MLP encoder → embedding features

Backbone (GPT-style Transformer):
  - 12 layers, 12 attention heads, 384-dim embedding
  - Input: token sequence including [RTG], [ACTION], [IMAGE] tokens + encoded features
  - Output: hidden features for each token

Decoders:
  - ReturnToGo decoder P_φ (Transformer): takes h^RTG → predicts RTG + hidden features
  - Action decoder P_ω (CVAE-based): takes h^action + RTG hidden features → predicts action
  - Image decoder P_ν (Transformer): takes h^image → predicts future image
```

### 4.2 Action Prediction Flow - Single Inference 的关键

**Step 1: Backbone forward pass**
$$h_{t:t+k-1}^{\text{RTG}}, h_{t:t+k-1}^{\text{action}} = \pi_\phi(l, o_{t-u+1:t}, s_{t-u+1:t})$$

**变量解释**:
- $\pi_\phi$: backbone network (GPT-style transformer)
- $l$: language instruction
- $o_{t-u+1:t}$: image observation 历史, 长度 $u=10$
- $s_{t-u+1:t}$: proprioception 历史, 长度 $u=10$
- $h_{t:t+k-1}^{\text{RTG}}$: backbone 输出的对应 [RTG] token 的 hidden features, 覆盖 prediction horizon $t$ 到 $t+k-1$
- $h_{t:t+k-1}^{\text{action}}$: backbone 输出的对应 [ACTION] token 的 hidden features
- $k$: action prediction horizon (CALVIN: 5, UR5: 64)

**Intuition**: 一次 forward pass 同时获取 RTG 和 action 的 features, 因为 backbone 是 shared 的。

**Step 2: RTG decoder 提取 hidden features**
$$\hat{g}_{t:t+k-1}^{\text{hidden}} = P_\varphi(h_{t:t+k-1}^{\text{RTG}})$$

**变量解释**:
- $P_\varphi$: ReturnToGo decoder (Transformer, 128 hidden dim)
- $\hat{g}_{t:t+k-1}^{\text{hidden}}$: RTG decoder 的最后一层 hidden features (不是 final RTG prediction)

**关键 design**: 这里取的是 **最后一层 hidden features** 而不是 final output。这些 hidden features 包含了 RTG prediction 的 "reasoning process" 信息, 可以 inject 到 action prediction。

**Step 3: Action prediction with RTG injection**
$$\hat{a}_{t:t+k-1} = P_\omega(h_{t:t+k-1}^{\text{action}}, \hat{g}_{t:t+k-1}^{\text{hidden}})$$

**变量解释**:
- $P_\omega$: action decoder (CVAE-based, 128 hidden dim)
- $h_{t:t+k-1}^{\text{action}}$: backbone 的 action features
- $\hat{g}_{t:t+k-1}^{\text{hidden}}$: RTG decoder 的 hidden features (通过 concatenation 注入)

**Intuition**: RTG 的 "internal representation" 直接 inject 到 action decoder, 让 action prediction 被 "maximum return reasoning" 所 condition。这就是 single inference 实现 return-conditioned action generation 的核心。

### 4.3 Total Loss Function

$$\mathcal{L} = \lambda \mathcal{L}_{\text{RTG}} + \mathcal{L}_{\text{arm}} + 0.01 \mathcal{L}_{\text{gripper}} + 0.1 \mathcal{L}_{\text{image}}$$

**变量解释**:
- $\lambda = 0.001$: RTG loss weight (小, 因为 RTG 是 auxiliary supervision)
- $\mathcal{L}_{\text{arm}}$: smooth L1 loss on arm actions (continuous)
- $\mathcal{L}_{\text{gripper}}$: cross entropy loss on gripper actions (discrete: open/close)
- $\mathcal{L}_{\text{image}}$: pixel-level MSE on future image predictions
- 0.01 和 0.1 是经验性权重

### 4.4 Action Decoder 的 CVAE 结构

Action decoder 用 [Conditional Variational Autoencoder (CVAE)](https://arxiv.org/abs/1606.05908):
- **Encoder**: 把 ground-truth action trajectory encode 成 style vector embedding (32-dim latent)
- **Decoder**: style vector + [ACTION] token output + $k$ learnable tokens → Transformer → predict $k$-step action trajectory

**Intuition**: CVAE 让 action prediction 有 multi-modality 能力 - 同一个 state 可以有多个 reasonable actions, CVAE 的 latent variable 捕获这个 diversity。Training 时 encoder 用 ground-truth, inference 时从 prior 采样。

### 4.5 Network Hyperparameters (Table 3)

| Parameter | Value |
|-----------|-------|
| Action prediction horizon $k$ | 5 (CALVIN) / 64 (UR5) |
| History stack $u$ | 10 |
| Action encoder latent dim | 32 |
| Action encoder/decoder hidden dim | 128 |
| RTG decoder hidden dim | 128 |
| Visual feature dim | 768 |
| Language feature dim | 768 |
| Embedding dim | 384 |
| Backbone layers | 12 |
| Attention heads | 12 |
| Activation | ReLU |

---

## 5. Inference Pipeline - Algorithm 1

```
Input: ReinboT model π_φ, P_φ, P_ω, initial state o_{0,test}, s_{0,test}, instruction l_test, environment Env
  // Note: NO initial RTG value required!

t ← 0
while t ≤ T_test:
    1. h^RTG, h^action = π_φ(l_test, o_{t-u+1:t}, s_{t-u+1:t})  // Backbone forward
    2. ĝ^hidden = P_φ(h^RTG)  // RTG decoder
    3. â = P_ω(h^action, ĝ^hidden)  // Action decoder with RTG injection
    4. o_{t+1}, s_{t+1} = Env.Step(â)  // Execute action
    // Note: NO reward needed during inference!
    t ← t + 1
```

**Key advantage vs Decision Transformer**: 
- DT 需要手动指定 initial RTG value (critical hyperparameter, 难以 tune)
- DT 需要环境 reward 来 update RTG
- ReinboT 自主预测 RTG, 无需这些, 大大简化实际部署

---

## 6. Experiments 详解

### 6.1 CALVIN Mixed-Quality Data 实验 (Table 1)

**Dataset 构造**:
- CALVIN ABC: ~50 trajectories per task with language instructions (少量 annotated data)
- CALVIN autonomous data: >20,000 trajectories without language instructions (human teleoperation)
- Failure data: >10,000 trajectories from RoboFlamingo interacting with CALVIN D (with Gaussian noise 0.05, 0.1, 0.15 added to actions)

**Protocol**: Train on mixed-quality data → fine-tune on small annotated data → test on CALVIN D

**Results Analysis (Table 1)**:

| Algorithm | 1 instr | 2 instr | 3 instr | 4 instr | 5 instr | Avg. Length |
|-----------|---------|---------|---------|---------|---------|------------|
| RoboFlamingo (annotated only) | 0.55 | 0.19 | 0.07 | 0.02 | 0.00 | 0.83 |
| GR-1 (annotated only) | 0.67 | 0.37 | 0.20 | 0.11 | 0.07 | 1.41 |
| PIDM (annotated only) | 0.60 | 0.45 | 0.32 | 0.23 | 0.13 | 1.73 |
| GR-1 (all data) | 0.62 | 0.31 | 0.18 | 0.14 | 0.10 | 1.36 |
| GR-MG | 0.65 | 0.35 | 0.24 | 0.11 | 0.05 | 1.41 |
| RWR (sparse) | 0.63 | 0.36 | 0.21 | 0.12 | 0.07 | 1.38 |
| RWR (sub-goal, sparse) | 0.71 | 0.46 | 0.27 | 0.19 | 0.11 | 1.73 |
| RWR (dense, single) | 0.75 | 0.52 | 0.27 | 0.18 | 0.11 | 1.82 |
| ReinboT (sparse) | 0.70 | 0.44 | 0.29 | 0.19 | 0.12 | 1.74 |
| ReinboT (sub-goal, sparse) | 0.74 | 0.50 | 0.28 | 0.17 | 0.12 | 1.80 |
| ReinboT (dense, single) | 0.77 | 0.53 | 0.32 | 0.18 | 0.11 | 1.90 |
| **ReinboT (dense, full)** | **0.79** | **0.58** | **0.40** | **0.28** | **0.21** | **2.26** |

**Key observations**:

1. **Imitation learning baselines 限制**: GR-1 with all data (AL=1.36) 甚至比 GR-1 with annotated only (AL=1.41) 还差! 这说明 mixed-quality data 对 imitation learning 是 harmful 的, 因为它做 maximum likelihood 无法 filter bad data。

2. **Dense vs Sparse reward**: 
   - ReinboT sparse → dense single: AL 1.74 → 1.90 (+9.2%)
   - RWR sparse → dense single: AL 1.38 → 1.82 (+31.9%)
   - Dense reward 显著优于 sparse, 因为提供了 per-step supervision

3. **Full ReturnToGo vector vs single scalar**:
   - ReinboT dense single → dense full: AL 1.90 → 2.26 (+18.9%)
   - Predicting 每个 reward component 的 return 比 predict aggregated scalar 更 informative
   - **Intuition**: 这就像 predicting each class probability vs predicting only argmax - 保留了更多信息

4. **ReinboT vs RWR**: 都用 dense reward, ReinboT (2.26) > RWR (1.82). RWR 是经典 offline RL, 用 reward-weighted regression。ReinboT 的优势来自:
   - Modular architecture (single inference)
   - RTG as modality (vs RWR 的 reweighting)
   - Expectile regression 实现 implicit maximization

### 6.2 Ablation Study (Table 2)

| Variant | AL | Change |
|---------|-----|--------|
| ReinboT (dense, full) | 2.26 | - |
| W/o ReturnToGo | 1.36 | -39.8% |
| W/o $r_1$ (sub-goal achievement) | 1.87 | -17.2% |
| W/o $r_2$ (task progress) | 1.79 | -20.8% |
| W/o $r_3$ (behavior smoothness) | 1.90 | -15.9% |
| W/o $r_4$ (task completion) | 1.93 | -14.6% |

**Key insights**:

1. **RTG 是最 critical component**: 去掉 RTG, AL 暴跌 39.8%, 几乎回到 GR-1 (all data) 的水平。这证明 return maximization principle 是 ReinboT 的核心价值。

2. **$r_2$ (task progress) 影响最大 (-20.8%)**: 这有点 surprising, 但想想也有道理 - task progress reward 提供 "全局进度" signal, 对 long-horizon task 的 credit assignment 最有帮助。

3. **$r_1$ (sub-goal achievement) 次之 (-17.2%)**: 这是 dense reward 的主体, 衡量局部 progress。

4. **$r_3$ 和 $r_4$ 影响相近 (~-15%)**: 
   - Smoothness 让 action 更 stable
   - Task completion anchor 了 success signal
   - 两者都 important 但不是最 critical

### 6.3 Hyperparameter Analysis (Figure 2)

**ReturnToGo loss weight $\lambda$**:
- $\lambda = 0.001$ 最优
- 过大: RTG loss dominate, action prediction 受影响
- 过小: RTG supervision 太弱, return maximization effect 不足

**Expectile regression parameter $m$**:
- $m = 0.9$ 最优
- $m = 0.5$: degenerate to MSE, 无 return maximization
- $m = 0.99$: over-optimistic, 预测 OOD RTG, performance 下降

### 6.4 RTG Distribution Analysis (Figure 3)

论文分析了 predicted RTG 的 distribution:
- 随着 $m$ 增大, predicted RTG distribution 向 higher value shift
- $m=0.9$ 时, predicted RTG 略高于 ground-truth, 是 "moderate optimism"
- $m=0.99$ 时, predicted RTG 远高于 ground-truth, 是 "over-optimism"

**Intuition**: 这验证了 expectile regression 确实 push model 预测 higher return。但过高的 $m$ 导致 predicted RTG 超出 training distribution, action generation 无法响应这个 unrealistic target。

### 6.5 Real-World Experiments (Figure 4-6)

**Setup**: UR5 robot arm, 530 successful trajectories total
- Few-shot learning: 30 trajectories per task (cup, bowl, plush toy)
- OOD generalization: unseen instructions, backgrounds, distractors, objects

**Results (Figure 6)**:
- ReinboT 在 few-shot 和 OOD 上都显著优于 baselines
- GR-1 和 RWR 表现相近 (RWR 在 data 不足时 overfit)

**Key insight from Figure 5**: 即使全是 successful trajectories, 在 dense reward metric 下, quality distribution 仍然 uneven。这证明 "successful ≠ high quality", 需要 RL principle 来 distinguish and prefer high-quality demonstrations。

参考:
- [GR-MG](https://arxiv.org/abs/2503.01949) - Hierarchical VLA baseline
- [PIDM](https://arxiv.org/abs/2412.15109) - Predictive inverse dynamics model baseline  
- [RoboFlamingo](https://arxiv.org/abs/2311.01378) - VLA baseline
- [Q-Transformer](https://arxiv.org/abs/2310.10101) - Auto-regressive Q-function for VLA

---

## 7. 与 Related Work 的深度对比

### 7.1 vs Decision Transformer (DT)

| Aspect | DT | ReinboT |
|--------|-----|---------|
| RTG role | Input condition | Predicted output + condition |
| Inference RTG | Need manual init | Auto-predicted |
| Environment reward | Needed | Not needed |
| Return maximization | No (fit ground-truth) | Yes (expectile regression) |
| Architecture | Pure Transformer | Multi-modal VLA |

### 7.2 vs Reinformer

| Aspect | Reinformer | ReinboT |
|--------|------------|---------|
| Expectile regression | ✓ | ✓ |
| Inference passes | 2 | 1 |
| RTG injection | Input token | Hidden features |
| Architecture | Single-modal | Multi-modal VLA |
| Reward | Task-specific | General dense reward |

### 7.3 vs Q-Transformer

| Aspect | Q-Transformer | ReinboT |
|--------|--------------|---------|
| RL formulation | Q-learning | Return prediction |
| Sequence length | Grows with action dim | Fixed |
| Inference time | Grows with action dim | Constant |
| Value estimation | Explicit Q-function | Implicit via expectile |
| Reward | Sparse | Dense |

### 7.4 vs RWR (Reward-Weighted Regression)

RWR 的 gradient:
$$\nabla_\pi \mathcal{L}_a = \frac{1}{N} \sum_\tau \nabla_\pi \log \pi(a | l, \langle o, s\rangle_{t-h:t}) \left[\sum_{i=t}^{T} \gamma^{i-t} \cdot r(l, \langle o, s, a\rangle_{t-h:t})\right]$$

**变量解释**:
- $\gamma = 0.9$: discount factor
- $r(\cdot)$: reward function
- gradient 被 return $\sum \gamma^{i-t} r$ weighted

**Intuition**: RWR 直接 reweight action log-likelihood by return。问题: 当 data distribution uneven, reweighting 可能 unstable。ReinboT 通过 expectile regression implicit 实现 maximization, 更 stable。

参考:
- [RWR original paper](https://dl.acm.org/doi/10.1145/1273496.1273590)

---

## 8. Training Details (Table 4)

| Parameter | Value |
|-----------|-------|
| RTG loss weight $\lambda$ | 0.001 |
| Expectile parameter $m$ | 0.9 |
| Gradient clip | 1.0 |
| Epochs | 50 |
| Warm-up epochs | 1 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Dropout rate | 0.1 |
| Reward weights $w_{1:4}$ | 0.1, 0.1, 0.01, 0.1 |
| Optimizer | Adam ($\beta_1=0.9, \beta_2=0.999$) |

**Pre-training**: ReinboT 用 [Ego4d](https://ego4d-data.org/) dataset 上的 video generation pre-training 初始化 weights (与 GR-1 一致), 这提供了 rich visual representation。

**Handling missing language**: 对于没有 language instruction 的 autonomous data, 用 empty string 作为 input, 允许 model 利用 all data。

---

## 9. Core Intuition Summary

让我总结一下 ReinboT 的 core intuition, 帮助 build your mental model:

### 9.1 为什么 Imitation Learning 在 Mixed-Quality Data 上 Fail

Imitation learning 优化:
$$\min_\theta \mathbb{E}_{(s,a) \sim \text{data}} [-\log \pi_\theta(a|s)]$$

这 uniform 地 treat 所有 data points, 无论 quality。Mixed-quality data 中, bad trajectories 拖累 model, 让它学到 "average behavior" 而非 "optimal behavior"。

### 9.2 为什么 Dense Reward 比 Sparse Reward 好

Sparse reward 的 credit assignment problem: 只有 trajectory 结束才知道 success/failure, 中间哪一步 good/bad 无法区分。

Dense reward 提供 per-step supervision:
- $r_1$: 这一步离 sub-goal 多近?
- $r_2$: 这一步在整个 task 进度哪里?
- $r_3$: 这一步的 motion smooth 吗?
- $r_4$: 整个 task 成功了吗?

每个 component 从不同角度 evaluate action quality, 提供 rich signal。

### 9.3 为什么 Expectile Regression 实现 Return Maximization

传统 offline RL 需要学 value function $Q(s,a)$, 但在 transformer 中 value estimation 不 stable (参考 [Catformer](https://arxiv.org/abs/2105.05181), [Stabilizing Transformers for RL](https://arxiv.org/abs/1910.06791))。

Expectile regression 的 trick: 不直接 maximize return, 而是 learn 预测一个 "optimistic" return。这个 optimistic return 作为 condition, 间接引导 action 朝向 high-return region。

数学上, expectile regression 的 solution 是 distribution 的 expectile, 而非 mean。当 $m > 0.5$, expectile 大于 mean, shift 到 distribution 右侧。这相当于 "imagine" 一个 better-than-average return, 然后让 action 朝这个 target 学习。

### 9.4 为什么 Single Inference 重要

Reinformer 的两次 inference 在实际 robot deployment 中是 significant overhead:
- 每个 timestep 需要两次 forward pass
- 第一次预测 RTG, 第二次用 RTG 预测 action
- Latency 翻倍

ReinboT 的 modular design:
- 一次 forward pass 获取所有 features
- RTG decoder 的 hidden features 直接 inject 到 action decoder
- 只需一次 inference

### 9.5 为什么 RTG Hidden Features 而非 Final RTG Value

Final RTG $\hat{g}_t$ 是 scalar, information 有限。RTG decoder 的 hidden features $\hat{g}^{\text{hidden}}$ 是 high-dimensional vector, 包含 RTG prediction 的 "reasoning representation"。

这个 representation 包含:
- 哪些 sub-goal 还要 reach
- 当前 progress 状态
- 预期 future trajectory quality

Inject 这些 rich features 到 action decoder, 比 inject scalar 更 informative。

---

## 10. Potential Extensions & Open Questions

基于我对 paper 的理解, 一些 potential directions:

1. **Reward learning**: 当前 dense reward 是 hand-crafted。能否用 [RLHF](https://arxiv.org/abs/2203.02155) style 的 human preference learning 自动学 reward?

2. **Scaling**: Paper 提到 future work 是 scaling models and data。能否 combine ReinboT 的 principle 与 [Open X-Embodiment](https://robotics-transformer-x.github.io/) scale 的 data?

3. **Online RL**: 当前是 offline RL。能否 extend 到 online RL fine-tuning (参考 [Policy Agnostic RL](https://arxiv.org/abs/2412.06685))?

4. **Multi-robot**: Dense reward 的 component (尤其 $r_3$ smoothness) 是否需要 per-robot customization? 能否 learn robot-agnostic reward?

5. **Hierarchical structure**: 能否 combine ReinboT 与 hierarchical VLA (like [GR-MG](https://arxiv.org/abs/2503.01949))? High-level predict RTG, low-level execute?

6. **World model connection**: Future image prediction $\mathcal{L}_{\text{image}}$ 与 [Dreamer](https://arxiv.org/abs/1912.01603) style world model 有 connection? 能否用 world model 做 imagination-based RL?

7. **Diffusion policy**: [Diffusion Policy](https://arxiv.org/abs/2303.04137) 在 action generation 上很强。能否 replace CVAE action decoder with diffusion, 用 RTG condition diffusion process?

---

## 11. Critical Analysis

### 11.1 Strengths

1. **Principled RL integration**: 不是简单加 RL loss, 而是通过 expectile regression implicit 实现 return maximization, 避免 value estimation difficulty。

2. **General dense reward**: 4 个 component 覆盖 manipulation task 的 key aspects, 设计 general, 不 task-specific。

3. **Practical inference**: Single inference, no RTG init, no reward needed - 实际 deployment 友好。

4. **Comprehensive experiments**: Sim + real, few-shot + OOD, ablation thorough。

### 11.2 Limitations & Questions

1. **Reward engineering**: 4 个 component + 4 个 weight 仍是 hand-crafted。Sensitivity analysis 只 show single-component ablation, 没 show weight sensitivity。

2. **Sub-goal division heuristic**: 基于 joint velocity 和 gripper state, 对 non-arm robots (e.g., quadruped, [QUAR-VLA](https://arxiv.org/abs/2410.15801)) 是否适用?

3. **$m=0.9$ 的 universal validity**: 只在 CALVIN 上 validate。其他 task distribution 的最优 $m$ 是否不同?

4. **Comparison with modern offline RL**: 没对比 [IQL](https://arxiv.org/abs/2110.06152), [TD3+BC](https://arxiv.org/abs/2106.01345) 等更强 offline RL baselines。

5. **RTG distribution shift analysis**: Figure 3 只 show distribution, 没 quantitatively analyze OOD rate vs performance。

6. **Long-horizon scaling**: 5 instructions chained 已是 limit? 10+, 20+ instructions 的表现如何?

---

## 12. 个人 Reflection (从 RL/sequence modeling 视角)

ReinboT 让我想到几个 deep 的 connection:

### 12.1 Sequence Modeling as RL

[Decision Transformer](https://arxiv.org/abs/2106.01345) 开创了 "RL as sequence modeling" paradigm。ReinboT 把这个 push 到 VLA scale。这 suggest 一个 trend: RL 的 "classical" elements (value function, policy iteration) 可能被 sequence modeling 的 supervised paradigm 替代, 至少在 offline setting。

### 12.2 Expectile Regression 与 Distributional RL

Expectile regression 与 [Distributional RL](https://arxiv.org/abs/1707.06887) (C51, QR-DQN) 有 connection。Distributional RL 学 return distribution, expectile 是 distribution 的 summary statistic。ReinboT 用 expectile 来 implicitly capture distribution 的 upper tail, 类似于 quantile regression 但 smooth。

### 12.3 Condition vs Maximization 的哲学

经典 RL: maximize $Q(s,a)$ w.r.t. $a$ → optimal policy
DT/ReinboT: condition $\pi(a|s, g)$ on high $g$ → implicit maximization

这是两种不同 philosophy。前者 explicit optimization, 后者 conditional generation。LLM/VLA 的 success 让后者越来越 attractive - 用 supervised learning + condition trick 实现 RL goal。

### 12.4 Dense Reward 与 Curriculum Learning

$r_2$ (task progress) 本质上是 curriculum signal。它告诉 model "你现在在 task 的哪一步", 类似 curriculum learning 的 stage label。这 suggest dense reward 设计可以借鉴 curriculum learning 的思想。

### 12.5 RTG 作为 "Planning" 代理

预测 RTG 本质上是 "imagine future return"。这与 model-based RL 的 "planning" 有 connection。ReinboT 没有 explicit world model, 但 RTG prediction 隐式 encode 了 "future will be good/bad" 的 belief。结合 future image prediction $\mathcal{L}_{\text{image}}$, ReinboT 有 weak world model 的味道。

---

## 13. Web Resources Summary

主要 references:
- [Reinformer (ICML 2024)](https://proceedings.mlr.press/v235/zhuang24a.html) - Expectile regression for max-return sequence modeling
- [Decision Transformer (NeurIPS 2021)](https://arxiv.org/abs/2106.01345) - RL as sequence modeling
- [GR-1 (ICLR 2024)](https://openreview.net/forum?id=v2NJ2J8K3q) - Video pre-training for VLA
- [CALVIN benchmark](https://calvinrobot.github.io/) - Long-horizon manipulation benchmark
- [CLIP](https://arxiv.org/abs/2103.00020) - Language-image encoder
- [ViT](https://arxiv.org/abs/2010.11929) - Vision transformer
- [Perceiver](https://arxiv.org/abs/2103.03206) - Efficient multi-modal perception
- [Expectile regression](https://en.wikipedia.org/wiki/Expectile) - Asymmetric least squares
- [Ego4d](https://ego4d-data.org/) - Egocentric video pre-training data
- [GPT-2](https://openai.com/research/better-language-models) - Backbone architecture
- [CVAE](https://arxiv.org/abs/1606.05908) - Conditional VAE for action decoder
- [Q-Transformer](https://arxiv.org/abs/2310.10101) - Auto-regressive Q for VLA
- [Open X-Embodiment](https://robotics-transformer-x.github.io/) - Robot learning at scale
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) - Diffusion for action generation
- [IQL](https://arxiv.org/abs/2110.06152) - Implicit Q-learning offline RL
- [Distributional RL](https://arxiv.org/abs/1707.06887) - Learning return distributions
- [Dreamer](https://arxiv.org/abs/1912.01603) - World model based RL
- [RLHF](https://arxiv.org/abs/2203.02155) - Reward learning from preferences

---

这篇 paper 的 core contribution 在于: 它把 offline RL 的 "return maximization" principle, 通过 expectile regression + dense reward + modular architecture, elegant 地集成到 end-to-end VLA model 中, 既享受了 RL 的 data efficiency 和 quality awareness, 又保持了 supervised learning 的 stability 和 scalability。这是 "RL + Foundation Model" 融合的一个 promising direction。

希望这个 detailed walkthrough 帮你 build 出 ReinboT 的 mental model! Core intuition 是: **用 dense reward quantify trajectory quality, 用 expectile regression predict optimistic return, 用 这个 optimistic return condition action generation** - 这三步实现了 implicit return maximization within supervised VLA framework。
