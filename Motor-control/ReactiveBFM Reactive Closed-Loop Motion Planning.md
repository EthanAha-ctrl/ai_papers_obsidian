---
source_pdf: ReactiveBFM Reactive Closed-Loop Motion Planning.pdf
paper_sha256: f83c0ec9315ad1cf9dc0bce54219c57ba2657f6c5743ded6d90f027b26c48d72
processed_at: '2026-08-11T21:12:36-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：ReactiveBFM 到底干了啥

## 一句话概括

他们让一个生成式 motion planner 能真正闭环跑在真实 humanoid 上，核心是**训练时主动给模型"喂脏数据"，让它学会从歪掉的状态自我恢复**，而不是部署时才发现自己不会 recover。

---

## 为什么这事本来很难

humanoid control 现在的套路是两层：

- 上面一个 **planner**（diffusion model 或 AR transformer）负责"想"——给定 text、target、历史状态，生成未来 40 帧该长啥样
- 下面一个 **BFM / tracker**（PPO 训出来的 RL policy）负责"做"——把 reference 转成 joint torque，让机器人真的动起来

BFM 已经很强了，SONIC、UniTracker 这些 zero-shot 就能 track 陌生轨迹。问题不在 control 层。

问题在 **planner 训练时见过的世界和部署时的世界不是同一个世界**。

具体讲：planner 训练用 teacher forcing，每一步输入的"历史 prefix"都是 ground-truth 轨迹，干干净净。但部署时它要闭环——上一帧 controller 执行完的真实状态是带噪声、带 tracking error、甚至被外力踹歪过的，这个 prefix 和训练分布完全对不上。

模型一 OOD 就开始瞎预测，瞎预测出来的 reference 让 controller 执行得更歪，下一帧 prefix 更歪，planner 更 OOD……**这就是 exposure bias 的雪崩**。

在 NLP 里 exposure bias 顶多让生成文本质量下降一点，在 humanoid 上是直接摔倒。Table 1 里 open-loop baseline 的 fall rate 23% 就是这么来的。

---

## 他们怎么解的：核心就一个 trick

**Scheduled prefix sampling**。

intuition 特别直白：既然部署时模型会遇到"歪掉的 prefix"，那训练时就主动给它看歪掉的 prefix，逼它学会怎么从歪掉的状态拉回来。

具体三段 curriculum：

1. **开头**：纯 teacher forcing，prefix 用 GT。让模型先学会 motion manifold 长啥样，别一开始就学坏了。
2. **中间**：线性衰减用 GT 的概率，剩下时候让模型**拿自己之前预测的 chunk 当下一个 window 的 prefix**。这就是 self-rollout——模型看到自己生成的不完美输出，必须接着往下生成，学着"自救"。
3. **最后**：在 prefix 上加 Gaussian noise 做域随机化，模拟 sim-to-real 的 tracking error 和外力扰动。

这本质上是把"部署时会遇到的脏数据"在训练时 explicit 喂给模型。就像教小孩骑车不能只在室内平地练，得让他出去摔几跤，学会从倾斜状态恢复平衡。

ablation 证实这是最关键的设计：去掉 self-rollout，success 从 93.1% 掉到 70.5%，掉 22.6 个点。这是全文最重的 single ablation。

---

## 其他几个配套 trick

### Compact 36-dim representation

公式 $\mathbf{x}_i = [\mathbf{p}_i, \mathbf{q}_i, \boldsymbol{\theta}_i] \in \mathbb{R}^{36}$ 就是 root 位置 + root 朝向四元数 + 29 个 joint position。

之前的 MDM 风格喜欢加 contact label、global velocity、foot heading 这些 dense feature。问题是这些 feature 之间物理上是耦合的（contact=true 时 foot velocity 必须近零），模型一处预测错就传染给 controller，drift 累积更快。

砍到 36 维就是把"error 能钻的口子"缩到最小。ablation 显示 dense representation 让 success 从 93.1% 掉到 89.1%。

### Temporal consistency loss

标准 MSE 是 point-wise，对时间维无感。replan 时新 chunk 和历史 prefix 拼接处会出现 velocity 不连续，机器人执行起来是 jerk。

加一阶、二阶差分 penalty 强制边界 velocity/acceleration 连续。去掉这个 loss，smoothness 从 96.9mm 恶化到 118.3mm。

### Condition dropout

text 和 target 位置独立 dropout。两个目的：学 unconditional motion prior（防止模型完全依赖 condition），以及模拟用户中途换指令——换指令时模型靠 unconditional prior 做平滑过渡，不会硬切。

---

## 工程上怎么落地

### 异步 replanning

controller 跑 50Hz（20ms 一帧），planner 单次推理 19.3ms，加通信开销根本追不上。

解法是**事件驱动的异步**：control buffer 持续消耗 reference frame，当剩余 frame < 10 就 trigger 一个非阻塞 planner 线程，planner 想好了就把新 40 帧 chunk 推进 buffer，controller 该干啥干啥。

留 10 frame 余量是为了吸收 Wi-Fi 抖动、OS scheduling jitter、TensorRT 偶发慢推理。同时 controller process 设 real-time CPU priority，防止被 planner 线程抢资源。

这就是为什么 50Hz control loop 永远不会被 planner 拖垮——两层频率完全解耦。

### Trajectory chunking + temporal ensembling

异步 replanning 会产生 overlap：新 chunk 推进来时旧 chunk 还有 ~10 帧没执行完。直接覆盖就是 jitter。

解法是 ALOHA 那套 temporal ensembling——overlap 区间内两套 reference 做加权 blending，指数衰减权重，保证拼接处 spatio-temporal 连续。

### Ego-centric reset（zero-shot 动态目标）

这是最巧的 generalization trick。

问题：planner 训练时只见 **静态** target reaching（PhysHSI-Reach 数据集，target 不动）。部署时要 reach **移动** target（人手持 tracker 走动），这是严重 OOD。

insight：把"追踪移动目标"重新 cast 成一串"静态 reach 子任务"。

每次 replan：
1. 把 robot 当前 root pose 设为新的全局原点
2. 把 target 在世界系下的位置 transform 到这个 egocentric frame
3. planner 看到的就是"我站在这里，target 在我相对位置 $(\Delta x, \Delta y, \Delta z)$"——完美匹配训练分布
4. 物理 momentum 由 AR prefix 里的历史 proprioception 隐式处理，planner 不需要显式知道

结果：zero-shot 90% success rate，连续执行 40 秒+。**训练数据里一条移动目标的样本都没有**，全靠 reframe 把 OOD 任务塞回 in-distribution。

---

## 结果说明了什么

几个关键 read：

1. **Open-loop 下 ReactiveBFM 已经最强 (75.2%)**：数据 curation + compact representation 本身就比 baseline 强。
2. **同方法 closed-loop 比 open-loop 提升 17.9 个点**：闭环的价值。
3. **TWIST2 controller 配 TextOp closed-loop 只有 56.1% success**：controller 太弱直接拖垮整个系统。说明 BFM 的 tracking fidelity 是闭环能不能跑的前提。
4. **self-rollout 贡献最大 (-22.6%)**，temp loss 次之 (-10.1%)，compact repre 最小 (-4%)。

最深的一层意思：**foundation model 时代的 robotics，关键不是把单模块做到极致，而是把闭环每一层的 distribution shift 都 explicit 处理掉**。BFM 已经把 low-level control 解决得差不多了，剩下的是怎么围绕它构建 robust closed-loop system。ReactiveBFM 是一个相当完整的 reference implementation。

---

## 我的 takeaway

读完最大的感受是：这 paper 没有发明任何新算法，scheduled sampling 2015 年 Bengio 就搞了，temporal ensembling ALOHA 2023 年就有了，async execution 是 robotics 老套路。但**组合起来在 humanoid 这个对 distribution shift 极度敏感的载体上 work，本身就是贡献**。

humanoid 比 manipulator 难，因为 manipulator 闭环错了最多撞坏东西，humanoid 闭环错了直接摔地上，fall rate 是刚性指标。这逼迫你必须每一层 mismatch 都 explicit 处理，不能假装看不见。

这种 "defense-in-depth" 式的 robustness 设计思路，比任何一个 single trick 都值得借鉴。下一个阶段 humanoid 研究的核心战场，大概就从"训更强的 BFM"转移到"围绕 BFM 构建更鲁棒的闭环系统"了。

---

参考：
- Project page: https://xiao-chen.tech/reactivebfm  
- BFM: https://arxiv.org/abs/2509.13780  
- SONIC: https://arxiv.org/abs/2511.07820  
- CLoSD: https://openreview.net/forum?id=RrXhQy3KbE  
- Scheduled Sampling: https://arxiv.org/abs/1506.03099  
- ALOHA: https://arxiv.org/abs/2304.13705  
- Mobile ALOHA: https://arxiv.org/abs/2401.02117

---

# ReactiveBFM 深度解析：从 Open-Loop Cascade 到 Closed-Loop Reactive Humanoid Control

这篇 paper 解决的核心问题非常本质：**如何把一个生成式的 motion planner 真正闭环地接到 humanoid 上运行**。表面上看是系统集成，实际触及的是一个深层的 training-inference mismatch 问题——exposure bias 在 robotics 闭环控制中的致命放大。我来逐层拆解。

---

## 1. 为什么 Open-Loop Cascade 必然失败

先建立 intuition。当前 humanoid control 的主流范式是两段式：
- **High-level planner**：生成 reference trajectory（通常是 diffusion model 或 AR transformer）
- **Low-level BFM/tracker**：tracking controller 把 reference 转成 joint torque

BFM 已经很强了（SONIC、UniTracker、ExBody 这些），可以 zero-shot track 未见过的 reference。问题在 planner 这层。

Open-loop 的崩塌路径是这样的：

```
planner 预测 X̂_{1:L}  →  controller 执行  →  实际状态 X^real_{1:L} ≠ X̂_{1:L}
                                                    ↓
                                            下一帧 replan 时
                                            planner 仍然以为
                                            当前状态 = X̂
                                                    ↓
                                            GT-prefix 训练分布
                                            被打破
                                                    ↓
                                            error 累积 → fall
```

关键 insight：**planner 训练时用 teacher forcing，看到的 prefix 永远是 ground-truth 轨迹。部署时它看到的 prefix 是真实物理状态，两者分布完全不同。** 这就是 exposure bias。在 NLP 里这只是一点 quality 退化，在 humanoid 上是 fall，是 catastrophic。

Table 1 的数据很说明问题：
- TextOp+SONIC open-loop：success 64.5%，fall rate 23.5%
- 同样的 TextOp+SONIC closed-loop：success 76.4%，fall rate 14.7%（改善但还不够）
- ReactiveBFM：success 93.1%，fall rate 2.0%

仅仅闭环还远远不够，必须在训练阶段就打破 exposure bias。这是 paper 的核心立论。

参考：
- CLoSD (closed-loop diffusion)：https://openreview.net/forum?id=RrXhQy3KbE  
- Scheduled Sampling (Bengio 2015)：https://arxiv.org/abs/1506.03099  
- SONIC：https://arxiv.org/abs/2511.07820

---

## 2. 核心方法：Scheduled Prefix Sampling Curriculum

### 2.1 AR Formulation

planner 是一个 **Auto-Regressive Motion Diffusion Model (AR-MDM)**，类似 CLoSD / DartControl 的架构。每次给定：
- 文本指令 c_text
- 目标位置 c_target
- 历史 proprioception prefix（前 20 帧）
- 当前 noisy chunk X_τ at diffusion step τ

预测 clean chunk X̂_0 = G(X_τ, c, τ)，长度 40 帧。

### 2.2 Compact 36-Dim Representation

公式 (1)：

$$\mathbf{x}_i = [\mathbf{p}_i, \mathbf{q}_i, \boldsymbol{\theta}_i] \in \mathbb{R}^{36}$$

变量含义：
- $\mathbf{p}_i \in \mathbb{R}^3$：root 的 global translation（世界系下位置）
- $\mathbf{q}_i \in \mathbb{R}^4$：root 的 quaternion rotation（四元数表示的全局朝向，比 Euler 稳定，无 gimbal lock）
- $\boldsymbol{\theta}_i \in \mathbb{R}^{29}$：Unitree G1 的 29 个 DoF 的 joint position

为什么是 36 维而非传统 dense representation？传统 MDM / Motion-X 风格会加 contact label、global velocity、foot heading 等，这样 over-parameterize 之后：
1. 生成模型需要预测更多冗余维度
2. 这些冗余维度之间是耦合的（比如 contact 和 foot velocity 物理上必须一致）
3. 任何一处预测不准都会传染，导致 controller 执行时 kinematic misalignment，物理 drift 累积

最小但 kinematically complete 的表示严格 bound 住 error space。Ablation 在 Table 1 里：dense representation 让 success 从 93.1% 掉到 89.1%，fall rate 从 2.0% 升到 2.7%。

### 2.3 Diffusion Loss

公式 (2)：

$$\mathcal{L}_{diff} = \mathbb{E}_{\mathbf{X}_0 \sim p(\mathbf{X}_0|\mathbf{c}),\, \tau \sim [1,T]} \left[\|\mathbf{X}_0 - \hat{\mathbf{X}}_0\|_2^2\right]$$

变量：
- $\mathbf{X}_0$：ground-truth clean motion chunk
- $\hat{\mathbf{X}}_0 = G(\mathbf{X}_\tau, \mathbf{c}, \tau)$：模型预测
- $\tau \in [1,T]$：diffusion timestep，从均匀分布采样
- $p(\mathbf{X}_0|\mathbf{c})$：conditioned data distribution
- $\mathbf{c}$：text + target + prefix 拼起来的 condition

直接在 motion space 做 MSE，没有用 latent diffusion。因为 36 维够小，不需要 VAE 压缩，省掉一个 encoder-decoder 的 inconsistency。

### 2.4 Scheduled Sampling Curriculum 的本质

这是全文最关键的 training trick。直觉是这样的：

**问题**：teacher forcing 让模型只见过"理想 prefix → ideal continuation"。Deploy 时 prefix 是真实的、带噪声的、偏离 GT 的状态，模型没见过这种 OOD prefix，预测就崩。

**解法**：训练时**主动让模型见到自己生成的不完美 prefix**，强迫它学会 "从一个歪掉的状态如何拉回正轨"。

具体 schedule：

1. **Phase 1 (warmup)**：纯 teacher forcing，prefix 全用 GT。让模型快速学到 motion manifold 的基本结构。
2. **Phase 2 (linear decay)**：用 GT prefix 的概率 $p_{GT}$ 线性衰减，$p_{GT}(t) = 1 - t/T_{decay}$。其余时候 $1 - p_{GT}$ 概率走 **self-rollout**：用模型自己之前预测的 chunk 作为下一个 window 的 prefix。
3. **Phase 3 (noise injection)**：在 prefix 上加 Gaussian noise 做域随机化，进一步模拟 sim-to-real gap 和 controller tracking error。

这相当于让模型在线训练时不断被"踹一下"，然后学会如何恢复。结果在 Table 1 ablation：

| Method | Success | Fall Rate |
|---|---|---|
| ReactiveBFM | 93.1% | 2.0% |
| w/o self-rollout | 70.5% | 9.3% |

去掉 self-rollout（即只用 teacher forcing）success 掉 22.6 个百分点。这是 paper 最强的 ablation 之一，直接证明 exposure bias 是核心 bottleneck。

### 2.5 Temporal Consistency Loss

标准 MSE 是 point-wise 的，对时间维 agnostic。 replan 时新的 chunk 和历史 prefix 的拼接边界会出现 velocity/acceleration 不连续，机器人执行起来就是 jerk。

加一阶、二阶差分 penalty：

$$\mathcal{L}_{temp} = \lambda_1 \|\Delta \hat{\mathbf{X}}_0 - \Delta \mathbf{X}_{GT}\|_2^2 + \lambda_2 \|\Delta^2 \hat{\mathbf{X}}_0 - \Delta^2 \mathbf{X}_{GT}\|_2^2$$

其中 $\Delta$ 是沿时间轴的一阶差分（velocity），$\Delta^2$ 是二阶差分（acceleration）。

更重要的设计意图：在 **prefix-generation boundary**（即第 20 帧和第 21 帧之间）强制 zero-order continuity，保证拼接处位置连续。

Ablation：去掉 temp loss → success 83.0%，smoothness 118.3mm（vs 96.9mm）。

### 2.6 Condition Dropout

text 和 target 位置以独立概率 dropout。两个目的：
1. **学 unconditional motion prior**：当 condition 缺失时仍能生成 reasonable motion，避免 condition collapse。
2. **模拟 streaming command switching**：用户中途换指令时，模型依赖 unconditional prior 做平滑 transition，而不是 hard switch 导致的 discontinuous joint reference。

这一点在 Fig. 4 (e-f) 的 streaming interactive control 实验里非常直观——文本指令在线切换，机器人不会突然 freeze 或 jerk。

---

## 3. Closed-Loop System 工程细节

### 3.1 Asynchronous Replanning

低层 controller 跑 50Hz（20ms 周期）。Planner 单次推理 19.3ms，加上数据搬运和通信，根本追不上。强行同步会 block control loop。

解法是**事件驱动异步**：

```
control buffer 持续消耗 reference frame
                ↓
剩余 frame 数 < N_buf = 10
                ↓
trigger 非阻塞 planner 线程
                ↓
planner 生成 40-frame chunk
                ↓
推入 buffer，controller 继续跑
```

$N_{buf}=10$ 这个阈值的设计：planner 推理 ~20ms，相当于 1 frame @ 50Hz，留 10 frame 余量是为了吸收 Wi-Fi 抖动、OS scheduling jitter、TensorRT 偶发慢推理。

CPU scheduling 上把 controller process 设为 real-time priority，防止被 planner 线程 preempt。这是 hard real-time control 的标准做法。

### 3.2 Trajectory Chunking + Temporal Ensembling

异步 replanning 带来一个问题：新的 40-frame chunk 和上一个 chunk 在时间上 **overlap**（因为 buffer 还有 ~10 frame 没执行完）。直接覆盖会导致：
- overlap 区段两套 reference 突然切换 → jitter
- 时间轴对齐错位 → 整段轨迹"跳变"

解法是 ALOHA / Mobile ALOHA 那套 temporal ensembling：在 overlap 区间内对两套 reference 做加权 blending（指数衰减权重），保证 spatio-temporal continuity。

参考：
- ALOHA: https://arxiv.org/abs/2304.13705  
- Mobile ALOHA: https://arxiv.org/abs/2401.02117

### 3.3 Ego-Centric Reset for Zero-Shot Dynamic Reaching

这是 paper 最有意思的 generalization trick。

**问题**：planner 训练时只见过 **static** target reaching（PhysHSI-Reach 数据集，target 位置固定）。但部署时要求 reach **moving** target（人手持 tracker 移动）。这是严重的 distribution shift。

**Insight**：把"动态追踪"分解成一串"静态 reach 子任务"。

每个 replan step：
1. 把 robot 当前的 root pose 设为新的全局 origin
2. 把 target 在世界系下的位置 transform 到这个 egocentric frame
3. planner 看到的就是"我现在站在这里，target 在我相对位置 $(\Delta x, \Delta y, \Delta z)$ 处"——这恰好匹配训练分布！
4. 物理上 robot 还在惯性运动，但 planner 不需要知道——AR prefix 里的历史 proprioception 已经 encode 了 momentum

这是把 continuous tracking problem 重新 cast 成 training distribution 内的 static reach + 动力学由 prefix 隐式处理。结果：zero-shot 90% success rate，连续执行 40 秒+。

---

## 4. BFM 架构细节

paper 在 Appendix B.2 透露了 controller 的内部结构（虽然是引用 [35] 的 anonymous 工作）：

```
Input:
  historical proprioception s^P (length L)
  historical actions a
  future goal states s^g (length N)
                ↓
modality-specific tokenizers → embeddings
                ↓
sequence-interleaved: (z^s^p_{t-L+1}, z^a_{t-L+1}, ..., z^s^p_t, e)
                                                ↑
                                          query token
                ↓
Transformer backbone:
  - Self-attention (causal mask: context 不能 attend query，query 能 attend 全部 context)
  - Cross-attention (注入 future goal tokens Z^g_t)
  - SwiGLU FFN
  - RMSNorm
  - RoPE positional embedding
                ↓
e 的 hidden rep → linear projection head → action / value
```

几个值得注意的点：

**Cross-attention 注入 goal** 而不是 concatenation：这允许 flexible 数量的 goal tokens，支持变长 future horizon。RMSNorm 把 goal embedding 投到 bounded sphere——这是个 manifold constraint，鼓励结构化 representation 涌现。

**Causal masking 的非对称设计**：context tokens 之间互相 causal，但 query token 能"看"所有 context。这是典型的 prefix-LM / decoder-with-query pattern，类似 Perceiver、Decision Transformer 的 query 设计。

训练规模：**102 million frames @ 50FPS**，64 GPU × 10 days。这是 BFM 之所以叫 foundation model 的原因——数据规模远超任何 single-task policy。PPO 优化，8192 parallel environments in IsaacLab。

参考：
- IsaacLab: https://arxiv.org/abs/2511.04831  
- PPO: https://arxiv.org/abs/1707.06347  
- SwiGLU: https://arxiv.org/abs/2002.05202  
- RMSNorm: https://arxiv.org/abs/1910.07467  
- RoPE: https://arxiv.org/abs/2104.09864

---

## 5. 实验数据深度解读

### 5.1 Sim-to-Sim 主表 (Table 1)

扰动设置：100N 外力打在 torso 和 pelvis 上，持续 0.1s。这是相当狠的扰动——一个 35kg 量级的 humanoid，100N × 0.1s = 10 N·s 的 impulse，足以让 walking 中的 robot 失衡。

| 类别 | Method | Success↑ | Fall↓ | MPJPE^r↓ | Smooth↓ | Survival↑ |
|---|---|---|---|---|---|---|
| Open-loop | DART+GMR+SONIC | 51.0% | 12.5% | 46.2mm | - | - |
| Open-loop | TextOp+SONIC | 64.5% | 23.5% | 40.1mm | - | - |
| Open-loop | Kimodo+SONIC | 70.4% | 16.0% | 41.2mm | - | - |
| Open-loop | ReactiveBFM (open-loop) | 75.2% | 4.3% | 36.5mm | - | - |
| Closed-loop | TextOp+TWIST2 | 56.1% | 31.9% | 51.7mm | 154.6mm | 20.1s |
| Closed-loop | TextOp+SONIC | 76.4% | 14.7% | 42.3mm | 128.5mm | 27.8s |
| Closed-loop | ReactiveBFM (dense repre.) | 89.1% | 2.7% | 40.3mm | 110.9mm | 29.2s |
| Closed-loop | ReactiveBFM (w/o temp loss) | 83.0% | 7.5% | 38.2mm | 118.3mm | 26.9s |
| Closed-loop | ReactiveBFM (w/o self-rollout) | 70.5% | 9.3% | 41.9mm | 124mm | 26.8s |
| Closed-loop | **ReactiveBFM** | **93.1%** | **2.0%** | **34.6mm** | **96.9mm** | **29.8s** |

几个有意思的 read：

1. **Open-loop 下 ReactiveBFM 已经是最好的 (75.2%)**：说明数据 curation + compact representation 本身就比 baseline 强。
2. **Closed-loop 比同方法 open-loop 提升 17.9%**：闭环的价值。
3. **TWIST2 controller 在 closed-loop 下 success 只有 22.3% (Appendix Table 3)**，配上 TextOp 后整体 closed-loop success 56.1%——controller 太弱直接拖垮整个系统。说明 BFM 的 tracking fidelity 是 closed-loop 能不能跑的前提。
4. **Self-rollout 贡献最大**：去掉掉 22.6%。Temp loss 贡献次之 (-10.1%)，dense representation 损失最小 (-4%)。

### 5.2 Modular Evaluation (Table 2, 3)

**Planner alone** (text-to-motion)：
- FID 2.10, R@3 0.45——比 Kimodo (FID 3.83)、TextOp (FID 18.20) 都好
- 但这个 metric 是 kinematic quality，不直接反映 closed-loop 价值

**Controller alone** (motion tracking)：
- Ours-Global: success 94.1%, MPJPE^r 29.2mm, MPJPE^a 97.5mm
- SONIC: success 71.0%, MPJPE^r 42.7mm, MPJPE^a 1171.2mm（这个 absolute MPJPE 1171mm 非常大，说明 SONIC 在 global frame 下漂移严重）

Ours-Global vs Ours-Local 差距很大（94.1% vs 82.4%, 29.2mm vs 36.6mm），说明 BFM 内部有 local 和 global 两种 tracking mode，global 模式显然更难但更鲁棒——这对 closed-loop 部署至关重要，因为 closed-loop 必须 global consistent。

### 5.3 Latency (Fig. 6)

- Planner: 19.3ms (TensorRT 优化后)
- Controller: 5.9ms
- Control loop: 50Hz = 20ms 周期
- Controller 占 5.9ms，剩 ~14ms 给 OS 和通信

这个数字说明 asynchronous 设计是必须的——同步的话 planner 19.3ms 加 controller 5.9ms 共 ~25ms，超 20ms 周期，必丢帧。

---

## 6. 数据 Curation

Table 6:

| Dataset | #Motions | #Frames | Avg Len | Duration @60fps |
|---|---|---|---|---|
| AMASS-HumanML3D | 11,424 | 1.16M | 101.2 | 5.35h |
| 100STYLE | 8,100 | 4.06M | 500.7 | 18.78h |
| Kungfu | 1,032 | 0.62M | 598.7 | 2.86h |
| PhysHSI-Reach | 9,994 | 2.19M | 219.3 | 10.15h |
| **Total** | **30,550** | **8.02M** | **262.6** | **37.14h** |

关键 pipeline：
1. AMASS-HumanML3D 提供 daily activity + text label
2. 100STYLE 提供 locomotion 多样性
3. Kungfu 提供 high-dynamic maneuver
4. PhysHSI-Reach 用 pre-trained PhysHSI policy 合成 10000 条 reach 轨迹，**只保留 reach phase**（截断 carry 部分）——这是为了给 planner 注入 manipulation prior

每条 motion 都过 PyBullet 做 **kinematic canonicalization + physical correction**，去掉 foot sliding 和 self-penetration。这一步至关重要，否则 planner 学到物理不可行的 prior，闭环时 controller 执行不了就崩。

参考：
- AMASS: https://amass.is.tue.mpg.de/  
- HumanML3D: https://github.com/EricGuo5583/HumanML3D  
- 100STYLE: https://homepages.inf.ed.ac.uk/tkomura/projects/100-style/  
- PhysHSI: https://arxiv.org/abs/2510.11072  
- PyBullet: https://pybullet.org/

---

## 7. Limitations 的诚实

Paper 自己承认两点：

1. **No explicit HOI (humanoid-object interaction)**：没建模 contact-rich loco-manipulation。要碰东西时可能 fail。
2. **Condition 太 compact**：只有 text + target position + proprioception。没有 vision，没有 tactile。复杂任务时 sensory 信息不够。

这两点指向下一步方向：把 vision-language foundation model 接进来做 condition，把 tactile feedback 接进来做 contact-rich 任务。但这又回到 latency 问题——VLM 推理几百 ms，怎么和 50Hz control loop 协同？这是一个新的异步层级问题。

---

## 8. 我的整体 intuition

读完这篇 paper，我脑海里的 mental model 是这样的：

**humanoid control 的本质是 distribution mismatch 的层层叠加**：
- Sim 和 real 的 mismatch
- Kinematic prior 和 physical feasibility 的 mismatch
- Planner 训练分布和部署时 OOD state 的 mismatch
- AR generation latency 和 control frequency 的 mismatch

每一层 mismatch 都会累积放大。Open-loop cascade 假装这些 mismatch 不存在，结果 fall rate 23%。ReactiveBFM 的贡献是**在每一层都主动 inject mismatch**：
- Scheduled sampling: inject 训练时的 prefix mismatch
- Noise injection: inject sim-to-real mismatch
- Compact representation: 缩小 representation 维度上 mismatch 的 surface area
- Async + chunking: absorb latency mismatch
- Ego-centric reset: 把 dynamic target 的 distribution shift 重新映射回 training distribution

这是一种 "defense-in-depth" 式的 robustness 设计。每一个 trick 单独都不新（scheduled sampling 2015 年就有了，temporal ensembling ALOHA 2023 年就有了），但组合起来在 humanoid 这个极端敏感的载体上 work，就是工程和科研的胜利。

更深层看，这篇 paper 在告诉我们：**foundation model 时代的 robotics，关键不在于把单一模块做到极致，而在于把闭环的每一段 distribution shift 都 explicit 处理掉**。BFM 已经把 low-level control 解决得差不多了，剩下的是怎么围绕它构建一个 robust 的 closed-loop system。这个方向上 ReactiveBFM 是一个相当完整的 reference implementation。

参考链接汇总：
- Project page: https://xiao-chen.tech/reactivebfm  
- BFM (Shanghai AI Lab): https://arxiv.org/abs/2509.13780  
- SONIC: https://arxiv.org/abs/2511.07820  
- UniTracker: https://arxiv.org/abs/2507.07356  
- CLoSD: https://openreview.net/forum?id=RrXhQy3KbE  
- DartControl: https://arxiv.org/abs/2412.06158  
- Scheduled Sampling: https://arxiv.org/abs/1506.03099  
- MDM (Human Motion Diffusion): https://arxiv.org/abs/2209.09106  
- ALOHA: https://arxiv.org/abs/2304.13705  
- Mobile ALOHA: https://arxiv.org/abs/2401.02117  
- PhysHSI: https://arxiv.org/abs/2510.11072  
- Unitree G1: https://www.unitree.com/g1  
- IsaacLab: https://arxiv.org/abs/2511.04831  
- PPO: https://arxiv.org/abs/1707.06347  
- AMASS: https://amass.is.tue.mpg.de/  
- HumanML3D: https://github.com/EricGuo5583/HumanML3D  
- PyBullet: https://pybullet.org/  
- HTC VIVE Ultimate Tracker: https://www.vive.com/us/accessory/vive-ultimate-tracker/
