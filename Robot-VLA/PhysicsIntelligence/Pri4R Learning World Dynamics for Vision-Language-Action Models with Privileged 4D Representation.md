---
source_pdf: Pri4R Learning World Dynamics for Vision-Language-Action Models with Privileged
  4D Representation.pdf
paper_sha256: 13d61ef429c099b7bfdd3cf02fb17ca3047773be11e9e699f711463d96f41a37
processed_at: '2026-08-06T06:07:29-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Pri4R 人话版

## 一句话讲清楚

**Pri4R 就是：训练时给 VLA 模型"开小灶"喂 3D point tracks，推理时啥都不加，但模型的脑子已经被这个辅助信号重塑了，action 预测变得更 physically aware。**

类比一下：师傅带徒弟修车。师傅一边修一边嘴里念叨"这个螺丝拧了之后那个件会往左偏 2 毫米"。徒弟本来只看着师傅的手（action label），现在被迫也听师傅的讲解（privileged 3D track）。时间长了，徒弟自己修车时虽然嘴上不念，但手上的动作已经带着那种"动了之后会发生什么"的直觉。

这就是 Learning Using Privileged Information (LUPI) 的核心思想，Vapnik 早在 2009 年就提出来了（https://arxiv.org/abs/1107.2118），Pri4R 把它搬到了 VLA 上。

---

## 为什么这事儿值得做

现在的 VLA models（OpenVLA、π0、π0.5 这些）有个通病：**它们会模仿动作的样子，但不懂动作的后果。**

你给它一个任务"把门拉开"，它可能直接去抓手柄往外拽。但门有铰链，是绕轴转的，直直往外拽就卡住了。模型不知道"我拽手柄，门会绕铰链转"这个物理事实，它只学到了"人类的手往这个方向移动过"。

问题出在训练信号上。VLA 的 loss 就是：

$$\mathcal{L}_{\text{act}} = \|\hat{\mathbf{a}}_{t:t+H} - \mathbf{a}_{t:t+H}\|_1$$

这个 loss 只告诉模型"人类在这个状态下做了什么动作"，但完全没告诉它"做了这个动作之后世界变成了什么样"。模型学到的是 action 的 marginal distribution，缺失了 action 到 world state 的因果链条。

人类不是这么学东西的。你抓杯子的时候，脑子里其实在预测"我手过去，杯子会被我推倒还是被抓住"。这个 forward model 是 implicit 的，但你时时在用。Pri4R 就是想把这个 forward model 塞进 VLA 的 representation 里。

参考这个 forward model 的心理学背景：Wolpert 1995 的 internal model 理论（https://www.science.org/doi/10.1126/science.7569931）。

---

## 怎么做的：极简架构

核心就一句话：**在 VLA backbone 上挂一个轻量 point track head，预测未来 H 步内 3D 点怎么移动，推理时把这个 head 丢掉。**

### 数据是啥

先看 supervision target 长什么样。对于每条 demonstration，在第一帧 sample 1024 个 3D 点（在 robot 周围的 cube 里，往 mesh 表面上撒），然后在整个 trajectory 里 track 这同一批点。

每个点的形式：

$$p_j^\tau = (x_j^\tau, y_j^\tau, z_j^\tau) \in \mathbb{R}^3$$

- $j$ = 点的 index，$j \in \{1, \ldots, N_p\}$，$N_p = 1024$
- $\tau$ = 时间步，$\tau \in \{t, \ldots, t+H+1\}$
- $x, y, z$ = 这个点在 world coordinate 里的 3D 位置

模型预测的不是绝对位置，是 **displacement**：

$$\Delta p_j^\tau = p_j^{\tau+1} - p_j^\tau$$

意思是"第 $j$ 个点从 $\tau$ 到 $\tau+1$ 这一步移动了多少"。

为啥用 displacement？后面讲。

### 架构长啥样

非常简单，就两个 MLP：

```
当前点集 P_t (1024×3)
    │
    ▼
PointMLP ──► per-point features e_t (1024×d)
                                    │
                                    ▼
backbone 出来的 z_t (H×d) ──► broadcast 成 (H×1024×d)
                                    │
                                    ▼
                            concat ⊕  (H×1024×2d)
                                    │
                                    ▼
                            FusionMLP
                                    │
                                    ▼
                    预测的 displacement ΔP̂ (H×1024×3)
```

公式版：

$$\widehat{\Delta P}_{t:t+H} = \text{MLP}_{\text{fusion}}(\mathbf{z}_t \oplus \mathbf{e}_t) \in \mathbb{R}^{H \times N_p \times 3}$$

逐项拆：
- $\mathbf{z}_t = \phi(\mathbf{o}_t) \in \mathbb{R}^{H \times d}$：backbone $\phi$ 从当前 observation $\mathbf{o}_t$ 提取的 multi-modal embedding，跨 horizon H 个 slot
- $\mathbf{e}_t = \text{PointMLP}(P_t) \in \mathbb{R}^{N_p \times d}$：当前点集编码后的 per-point feature
- $\oplus$：concatenation。$\mathbf{z}_t$ 在 point 维度 broadcast，$\mathbf{e}_t$ 在 time 维度 broadcast，拼起来送进 FusionMLP
- 输出 shape $H \times N_p \times 3$：H 步、1024 个点、每个点 3D displacement

### $\mathbf{z}_t$ 从哪来：两种 VLA 不同接口

**OpenVLA-OFT** 直接拿 action query token 的 final layer hidden states 当 $\mathbf{z}_t$。因为 OpenVLA-OFT 本来就要把这些 hidden states 喂给 action MLP 预测 action，现在同时喂给 point track head，等于让同一个 representation 干两件事。

**π series** 麻烦点，因为 π 的 action expert 是通过 masked self-attention 跟 backbone 交互的，没有一个现成的 query embedding 可以拿。作者加了个小 transformer module：learnable query tokens 做 cross-attention 到 backbone 的 final-layer image+language tokens，输出 $\mathbf{z}_t \in \mathbb{R}^{H \times d}$。

这个设计的 ablation（Table V）很有意思：直接复制 action expert 的结构做 "point expert" 只 +0.5%，把 query 注入 backbone 能 attend action 的 +1.5%，不能 attend action 的 +1.9%，用 cross-attention 提 final-layer hidden states 的 +4.1%。直觉是：别扰动预训练 representation，老老实实从最后层提 context 最干净。

### Loss

$$\mathcal{L} = \mathcal{L}_{\text{act}} + \omega_{\text{pt}} \|\widehat{\Delta P}_{t:t+H} - \Delta P_{t:t+H}\|_1$$

- $\mathcal{L}_{\text{act}}$：原始 action loss（OpenVLA-OFT 是 $\ell_1$，π 是 flow matching）
- $\omega_{\text{pt}}$：权重，最优点是 1.0
- 第二项：3D point displacement 的 $\ell_1$ loss

$\omega_{\text{pt}}$ 的 ablation（Table S7）：0.1 → 54.7%, 1.0 → 57.0%, 10.0 → 50.7%。1.0 刚好，说明 displacement 形式跟 action 量级匹配，几乎不用 tune。

---

## 几个关键设计选择的人话解释

### 1. 为啥用 displacement 不用 absolute position

Absolute position 有个大问题：模型可以靠"这个物体平时在桌上"这种 prior 蒙混过关，根本不需要学 dynamics。你给它当前点云 + 让它预测未来绝对位置，它可能直接输出"桌面附近的平均位置"就完事了。

Displacement 强制它学"这个特定的 scene，在这个特定的 action 下，怎么变化"。它没法靠 prior 偷懒，必须看当前 frame 和当前 action 才能预测下一步点会怎么动。

而且 displacement 量级跟 action（delta joint angle）量级匹配，loss 不用 tune weight，$\omega_{\text{pt}} = 1.0$ 直接 work。这个设计很 elegant。

### 2. 为啥 $P_t$ 不进 backbone，只进 point track head

这是 Pri4R 最聪明的地方，Table IV 是核心证据。

| 方案 | 推理时要 3D 输入? | Avg SR |
|---|---|---|
| Baseline | 否 | 33.1 |
| $P_t$ 进 backbone | 是 | 33.3 (+0.2) |
| $P_t$ 进 backbone + track 监督 | 是 | 34.5 (+1.4) |
| $P_t$ 只进 track head（Ours） | 否 | **46.3 (+13.2)** |

如果把 $P_t$ 作为额外 token 塞进 VLM 的 input sequence，有两个坏处：
- 推理时必须给 3D 点云，破坏原 VLA 接口
- VLM 的 embedding space 被新 token 类型扰动，预训练 representation 受损（Pick-and-Place 反而 -7.8%，因为 language 信号被稀释）

如果只把 $P_t$ 喂给 point track head（轻量旁路），backbone 输入还是只有 image + language，跟预训练分布一致，零 distribution shift。3D 信息通过 backprop "倒灌"进 backbone 的 shared representation，把 dynamics 知识压进去。推理时把 head 丢掉，但 backbone 已经"被训练好了"。

这就是 privileged information 的精髓：辅助信号塑造 representation，但不进入推理路径。

更极端的 ablation（Table S5）：如果连 point track head 都不给 $P_t$，让它从 $\mathbf{z}_t$ 凭空 generate 整个 track sequence，性能掉到 28.7%（比 baseline 还低）。说明 point track head 必须看到当前 scene 的 3D 结构才能预测演化，任务得是"这个特定 scene 怎么动"，不能是"凭空生成 scene"。

### 3. 为啥 head 越轻越好

Table VI 的 ablation 反直觉：

| Point Encoder | Fusion Module | SR | Δ |
|---|---|---|---|
| 无 head | - | 89.2 | - |
| PointNet | Ours | 80.8 | -8.4 |
| Point Transformer | Ours | 92.4 | +3.2 |
| Ours (PointMLP) | Transformer | 92.2 | +3.0 |
| **PointMLP** | **FusionMLP** | **94.4** | **+5.2** |

PointNet 反而损害性能 -8.4%。PointNet 的 max-pooling 丢失 per-point identity，而 Pri4R 要的是 per-point displacement，identity 必须保留。重型 Point Transformer 也不如简单 MLP。

为啥？因为 head 重了，dynamics 信号在 head 内部就被"消化"了，gradient 不回流到 backbone。head 越轻，越被迫依赖 backbone 提供 rich feature，dynamics 学习压力就被推到 backbone 身上。Backbone 学到了，action head 才能受益。

这跟 LoRA、adapters 的哲学一致：auxiliary 越弱，main model 越强。这条直觉可以推广到所有 privileged learning 的设计。

### 4. 为啥 3D track 比 2D track、depth、goal point 都好

Table III 是核心 evidence：

| Supervision | SR | Δ |
|---|---|---|
| Baseline | 33.1 | - |
| Goal point set (只预测终点) | 33.8 | +0.7 |
| 2D point track | 37.0 | +3.9 |
| Depth map (VAE latent) | 42.3 | +8.3 |
| **3D point track** | **46.3** | **+13.2** |

三个维度拆开看：

**Temporal density**：Goal point set 只在 horizon 终点监督一次，信号稀疏。3D track 在每个 timestep 都监督，捕捉 fine-grained interaction。+0.7 vs +13.2 的差距说明"未来某时刻长啥样"远不如"整个 trajectory 怎么演化"有用。

**Metric 3D structure**：2D track 投影到单视角，保持 temporal density 但丢 metric depth。2D 没法区分"物体离相机变远"还是"物体变小"，这种 ambiguity 在 manipulation 里致命。+3.9 vs +13.2 说明 metric 3D 是刚需。

**Spatial redundancy & identity**：Depth map 空间 dense，但 fixed camera 下大部分像素是静态背景，temporal 冗余。更糟的是 depth 没有 identity registration——同一像素在不同时刻不一定是同一物理点，所以 depth 学不到"这个点怎么动"。+8.3 vs +13.2。Point track 的 identity consistency 是关键优势。

**Robot-scene interaction**：Table III 下半部分：

| 监督哪些点 | SR | Δ |
|---|---|---|
| 只 track scene | 35.2 | +2.1 |
| 只 track robot | 43.8 | +10.7 |
| Robot + scene | 46.3 | +13.2 |

只 track robot 主要学 self-motion，只 track scene 弱化 contact sensitivity。World dynamics 必须 cover robot-environment interaction，这是 manipulation 的本质。两者一起 +13.2 远超单独之和 +12.8，有协同效应。

---

## 实验结果白话

### LIBERO（Table I）

LIBERO-Long 上 OpenVLA-OFT + Pri4R 涨 +9.8%（85.5 → 95.3）。Long suite 是多步任务，需要 planning + 中间状态判断 + contact，对 dynamics 最敏感，提升最大符合直觉。

π0.5 + Pri4R 在 LIBERO-Long 涨 +3.8%（90.5 → 94.3），LIBERO-Spatial 涨 +1.1%（96.1 → 97.2），LIBERO-Object 涨 +0.6%（88.3 → 88.9），LIBERO-Goal 持平。π0.5 本身已经很强（92.6 avg），Pri4R 在它基础上还能 +1.4 avg，说明这个 supervision signal 对 SOTA 模型也 work。

### RoboCasa（Table II）

RoboCasa 是 articulated interaction（门、抽屉、旋钮、扳手、按钮、insertion），全是需要理解 kinematic constraint 的任务。OpenVLA-OFT + Pri4R 拿到 +13.2 avg：

- Lever +30.7（36 → 66.7）：扳手最依赖"按下去之后会绕轴转多少"
- Press +23.3（56 → 79.3）：按钮需要"按到位"的 contact awareness
- Drawers +21.0（59 → 80）：抽屉的滑轨约束
- Knobs +17.0（8 → 25）：旋钮的旋转轴
- Doors +16.0（45.7 → 61.7）：铰链约束

这些任务 baseline 经常"动作看起来对但物理上卡住"。Pri4R 通过预测点怎么动，强迫模型理解"我拉手柄，门会绕铰链转"这种 dynamics。

### Training dynamics（Figure 3）

特别有意思：训练前 20K steps，Pri4R 比 baseline 慢（point track loss 占 representation capacity）。20K 之后快速上升，达到 baseline peak 速度是 baseline 的 **2.7×**，省约 8× H200 GPU-days。

直觉：前期模型在学"怎么预测 scene 演化"这个 skill，还没成熟到帮 action。一旦 dynamics representation 学好，action head 立刻 leverage 这个 physically-aware context，快速收敛。说明 world dynamics 是 action prediction 的 **bottleneck feature**，baseline 在低效地隐式学这个 feature，Pri4R 直接喂给它。

### Real-world（Table VII）

四个 task：
- **Height**（避障 pick-and-place）：83.3 → 96.7
- **Spatial**（bin 放入，seen/unseen 位置）：unseen 60 → 80
- **Depth**（拿最远的物体）：seen 45.9 → 79.2
- **Tracking**（物体移动中抓取）：seen 75 → 100, unseen 41.7 → 66.7, OOD 50 → 66.7

Tracking task 最能体现 dynamics 价值：物体在 robot approaching 时被移动，robot 必须持续 update grasp plan。Baseline 经常在 outdated 位置停下，close gripper on empty space，然后还继续执行剩余 action chunk（说明它根本没意识到 grasp 失败了）。Pri4R 持续 track 物体新位置，update 到新位置抓取。

Figure 6 的 qualitative 对比很直观：baseline 撞障碍物、抓错位置、空抓还继续执行；Pri4R 绕障、relocalize、几何一致的 approach。

---

## 跟其他方法的区别

### vs. Explicit world model（DreamerV3、3D-VLA、DreamVLA）

这些方法生成未来 image/state，policy 在 imagined rollout 里训练。优点：能做 planning。缺点：依赖 world model 质量，inference 有 latency。

Pri4R 是 implicit world model，把 dynamics 压进 representation，action head 自己用。优点：零 inference overhead，不依赖生成质量。缺点：没法显式 planning/search。

参考：
- DreamerV3: https://arxiv.org/abs/2301.04104
- 3D-VLA: https://arxiv.org/abs/2403.09631
- DreamVLA: https://arxiv.org/abs/2502.04899

### vs. SpatialForcing（Table S6）

SpatialForcing 用 VGGT（3D 几何 foundation model）的 feature 做 alignment，注入 static 3D structure awareness，同样不需要 test-time 3D 输入。

| 方法 | LIBERO Avg |
|---|---|
| OpenVLA-OFT | 92.7 |
| + SpatialForcing | 94.2 |
| + Pri4R | 95.0 |

Pri4R 略胜。区别在 supervision 性质：SpatialForcing 给静态 3D 结构，Pri4R 给 temporally dense 4D dynamics。后者直接监督 interaction，前者只给场景几何先验。

参考：SpatialForcing https://arxiv.org/abs/2510.12276

### vs. VLA with 3D input（PointVLA、3D-CAVLA、GeoVLA）

这些方法把 3D 点云 / depth 作为额外 input 喂给 backbone。缺点：推理时必须提供 3D 输入，破坏接口，引入 distribution shift。

Pri4R 的 privileged 范式更干净：训练时用，推理时丢。

参考：
- PointVLA: https://arxiv.org/abs/2503.07511
- 3D-CAVLA: https://arxiv.org/abs/2505.05800

---

## 我的 Intuition 和吐槽

### 直觉 1：Privileged supervision 的本质是"知识蒸馏"

Pri4R 可以看成：一个 oracle（知道 3D track）的知识被蒸馏到一个只看 image+language 的 student。Student 通过模仿 oracle 的 output（point displacement），被迫学到 oracle 的 internal reasoning（scene dynamics）。

这跟 distillation 的差别：distillation 蒸 logits/feature，Pri4R 蒸的是一个 derived task 的 output。但效果类似——都让 student 学到 teacher 的 implicit knowledge。

参考 Hinton distillation: https://arxiv.org/abs/1503.02531

### 直觉 2：为什么 MLP head 反而好——gradient 路径的重要性

这个 ablation（Table VI）我觉得是全 paper 最反直觉、最深刻的发现。重型 head 自己把活干了，gradient 不回流 backbone。轻量 head 必须依赖 backbone 提供好 feature，所以 backbone 被迫学。

这跟 prompt tuning vs. full fine-tuning 的 trade-off 有点像：prompt tuning 弱，所以必须依赖 frozen backbone 做重活，backbone 的 representation 反而被"激活"得更好。Full fine-tuning 太自由，容易 catastrophic forgetting。

类比到 auxiliary task 设计：auxiliary head 越弱，main representation 越强。这是个可推广的设计原则。

### 直觉 3：Point track 是 "Goldilocks representation"

太稀疏（goal point）：信号不够。
太 dense（depth map）：冗余、没 identity。
2D（投影）：缺 metric。
3D track + identity + dense：刚刚好。

而且 3D track 跟 action 在同一个 spatiotemporal metric space，supervision 信号天然 aligned with control。这是别的 representation（language、feature embedding、image）都做不到的。

### 吐槽 1：Real-world 的 track 质量没分析

Real-world 用 SpatialTrackerV2 pseudo-label，但没分析 tracker 失败率对性能的影响。如果 fast motion / occlusion 下 tracker 漂了，supervision 就是 noisy 的。这是个 clear robustness 风险。

### 吐槽 2：Point sampling 策略太朴素

作者就是 uniform sample on mesh + segmentation 偏向 foreground。但 manipulation 的关键是 contact region（gripper-object 接触点），这些区域应该 sample 更密。现在的策略可能浪费点 budget 在无关紧要的背景上。

一个改进方向：用 attention map 或 contact prediction 自动 focus sampling。这跟 active learning 的思路类似。

### 吐槽 3：Horizon 限制

$H = 10$ 大约 1 秒，dynamics 监督只覆盖这 1 秒。LIBERO-Long 提升大可能恰恰因为它需要更长 dynamics reasoning，而 Pri4R 只能间接帮助（通过 representation 泛化）。

如果 action chunk 再长，或者用 hierarchical structure（short-term track + long-term goal），可能效果更好。这跟 RT-2 的 hierarchical 思路（https://arxiv.org/abs/2307.15818）可以结合。

### 吐槽 4：Pretraining 阶段的潜力被浪费了

作者自己承认，Pri4R 只在 fine-tuning 上做。如果在 OpenX / Embodiment X 大规模 pretraining 阶段就加 point track supervision，representation 的 dynamics awareness 会从源头建立，下游 fine-tune 更高效。

这需要大规模 3D track 标注，但 SpatialTrackerV2 这种 model 已经可以 pseudo-label。Cost 是一次性 的，收益是所有下游任务共享。这是 clear next step。

参考 OpenX: https://arxiv.org/abs/2310.08864

### 联想 1：其他 privileged signal

Pri4R 用 3D track，但 privileged signal 的选择空间很大：
- **Tactile / force-torque**：接触力作 privileged，训练时监督，推理时不用
- **Contact graph**：哪些点接触、接触力方向，作 privileged supervision
- **Object pose**：6D pose 作 privileged（仿真里有 ground truth）
- **Future camera pose**：如果机器人有移动 base，未来视角作 privileged

这些都遵循 Pri4R 的范式：训练时 shaping representation，推理时丢弃。

### 联想 2：跟 contrastive learning 的关系

Pri4R 的 point track loss 其实可以改成 contrastive 形式：让模型预测"下一步这个点会去哪个位置"，用 contrastive loss 拉近正确位置、推远负样本。这在 track 数据 noisy 时可能更 robust。

参考 SimCLR: https://arxiv.org/abs/2002.05709

### 联想 3：跟 video prediction 的关系

3D point track 是 video 的稀疏版。如果未来 3D tracking model 强到能在任意 video 上 track，那所有 robot video 数据都能自动标 3D track，pretraining 阶段的 privileged supervision 就有了无限来源。

这跟 PointWorld（https://arxiv.org/abs/2601.03782）的思路有交集——用 4D representation 作为 robotic 任务的通用 interface。

### 联想 4：与 Joint Embedding Predictive Architecture (JEPA)

LeCun 的 JEPA 思路：predict latent representation of future，不 predict pixel。Pri4R 有点像 JEPA 的特例——predict 3D point 的 future position（一种 structured latent），不 predict future image。

如果 point track 看成"scene 的 latent state"，那 Pri4R 就是 VLA 版的 JEPA。区别是 JEPA 用 contrastive/non-contrastive learning，Pri4R 用 regression（因为有 ground truth track）。

参考 JEPA: https://arxiv.org/abs/2301.08243

---

## 设计哲学总结

把 Pri4R 的所有设计选择提炼成可推广原则：

1. **Privileged > Test-time auxiliary**：辅助信号训练时用，推理时丢
2. **Displacement > Absolute**：预测变化量，避免 prior 作弊
3. **Identity-preserving > Dense**：point track 比 depth/video 好，temporal identity 是关键
4. **Light head > Heavy head**：auxiliary 越弱，main backbone 越强
5. **Auxiliary-only input > Backbone input**：3D 输入不进 backbone，避免 distribution shift
6. **Robot + Scene > Either alone**：world dynamics 必须 cover interaction

这六条不只适用于 point track。任何 privileged signal（tactile、force、contact graph、object pose）都能套这套框架设计 VLA 的 auxiliary supervision。

---

## 最后的 take

Pri4R 是个典型的"小而美"工作：idea 简单（privileged 3D track supervision），实现轻量（两个 MLP），结果扎实（LIBERO +10, RoboCasa +13, real-world 多个 task 显著提升），ablation 全面（每个设计选择都有 controlled experiment）。

最大贡献不是某个具体 trick，而是把 privileged information 这个老 idea 在 VLA 时代重新激活，并用清晰的 ablation 证明"为什么 3D point track 是特别合适的 privileged signal"。

对未来 VLA pretraining 的启示：与其堆更多 action data，不如给现有 data 标 3D track（即使 pseudo-label），用 privileged supervision 重塑 backbone。这可能是 scaling VLA 的另一条路径——垂直做深（更好的 supervision signal），而不只是水平做大（更多 data）。

参考一下 paper 的项目主页看 visualization：https://jiiiisoo.github.io/Pri4R/ ，Figure 4 和 S3 的 point track 可视化很直观，能看到模型确实学到了"门绕铰链转、抽屉沿滑轨动"这种 dynamics。

---

# Pri4R: 用 Privileged 4D Representation 学习 World Dynamics

## 核心直觉

VLA models 目前的训练信号只有 action labels，只告诉模型"怎么动"，但没告诉模型"动了之后会发生什么"。Pri4R 的核心 insight 是：**用 3D point tracks 作为 privileged supervision，强迫 VLM backbone 学会预测 scene geometry 如何随 action 演化**，从而让 shared representation 编码 world dynamics，最终让 action head 受益。

这是一个典型的 **privileged information / learning using privileged information (LUPI)** [Vapnik 2009] 的思路：训练时有额外信息（3D tracks），推理时不用，但 representation 已经被改善了。

参考链接：
- 项目主页：https://jiiiisoo.github.io/Pri4R/
- π0 paper: https://arxiv.org/abs/2410.24164
- π0.5 paper: https://arxiv.org/abs/2504.16054
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- SpatialTrackerV2 (用来标注 real-world 3D tracks): https://arxiv.org/abs/2411.16842

---

## I. VLA Formulation 详解

每个 time step $t$，policy 接收 observation：

$$\mathbf{o}_t \triangleq (\mathbf{I}_t, \mathbf{t}_t, \mathbf{q}_t)$$

变量含义：
- $\mathbf{I}_t = \{\mathbf{I}_t^i\}_{i=1}^{N}$：N 个 multiview camera images（LIBERO 用 2 个，RoboCasa 和 real-world 用 3 个）
- $\mathbf{t}_t$：tokenized language instruction（如 "pick up the doll and place it on the right"）
- $\mathbf{q}_t$：robot proprioceptive state（关节角、gripper 状态）

输出是 action chunk $\mathbf{a}_{t:t+H}$，horizon 为 $H$。OpenVLA-OFT 用 $H=10$，π series 也类似。

训练目标（behavior cloning）：

$$\max_\theta \; \mathbb{E}_{(\mathbf{a}_{t:t+H}, \mathbf{o}_t) \sim \mathcal{D}} \log p_\theta(\mathbf{a}_{t:t+H} \mid \mathbf{o}_t) \tag{1}$$

这里 $\theta$ 是 VLM backbone + action head 的所有参数。这个目标只看到了 action label，**没有显式的 dynamics 约束**，模型容易学到"看起来对但物理上不 robust"的 action。

### 两个 baseline 架构

**OpenVLA-OFT**：backbone-centric
- 基于 Prismatic VLM（7B），LoRA fine-tune
- 最终层 action query token 的 hidden states → MLP regression head → 连续 action chunk
- Parallel decoding（bidirectional attention），不是 autoregressive
- $\ell_1$ regression loss

**π0 / π0.5**：expert-style
- VLM backbone + 独立的 transformer action expert
- Action expert 通过 flow matching 生成 continuous actions
- Blockwise causal mask：action tokens 可以 attend 全部 context，但 language/image tokens 不能 attend action tokens（保护 VLM representation）
- Action expert 输入：proprioception $\mathbf{q}_t$ + noisy action chunk，输出 velocity field 做 iterative denoising

参考：
- Flow Matching: https://arxiv.org/abs/2210.02747
- Prismatic VLMs: https://arxiv.org/abs/2403.09001

---

## II. Pri4R 架构详解

### A. Point Track Head

核心公式：

$$\widehat{\Delta P}_{t:t+H} = \text{MLP}_{\text{fusion}}(\mathbf{z}_t \oplus \mathbf{e}_t) \in \mathbb{R}^{H \times N_p \times 3} \tag{2}$$

逐项拆解：

- $P_t \in \mathbb{R}^{N_p \times 3}$：当前 time step 的 3D point set，$N_p = 1024$ 个点
- $\mathbf{e}_t = \text{PointMLP}(P_t) \in \mathbb{R}^{N_p \times d}$：per-point features，$d$ 是 feature dim
- $\mathbf{z}_t = \phi(\mathbf{o}_t) \in \mathbb{R}^{H \times d}$：从 backbone $\phi$ 提取的 multi-modal embeddings，跨 action horizon H 个 slot
- $\oplus$：feature concatenation。$\mathbf{z}_t$ broadcast 到 $\mathbb{R}^{H \times N_p \times d}$，$\mathbf{e}_t$ broadcast 到 $\mathbb{R}^{H \times N_p \times d}$，然后 concat 成 $\mathbb{R}^{H \times N_p \times 2d}$
- $\text{MLP}_{\text{fusion}}$：把 fused feature map 到 $\mathbb{R}^3$（每个 point 的 displacement）
- 输出 $\widehat{\Delta P}_{t:t+H}$：预测的 per-step 3D displacement，shape 是 $H \times N_p \times 3$

关键设计：**预测 displacement 而不是 absolute position**。这和 action prediction 的形式对齐（action 也是 displacement-like），让 loss 量级匹配，$\omega_{pt} = 1.0$ 几乎不需要 tune。

### B. $\mathbf{z}_t$ 的来源：两种 VLA 的不同接口

**OpenVLA-OFT**：
- $\mathbf{z}_t$ = action query token 的 final-layer hidden states
- 这些 hidden states 本来就要喂给 action MLP，所以 inject 进 point track head 等于在同一个 representation 上同时做 action 和 track 预测
- backbone 被迫让这些 hidden states 同时编码 "如何动" + "动了之后 scene 如何演化"

**π family**：
- π 的 action expert 不直接产出 query embedding，而是通过 masked self-attention 交互
- 所以作者加了一个 **lightweight transformer embedding module**：
  - 一组 learnable query tokens
  - cross-attention 到 backbone 的 final-layer image + language tokens
  - 输出 $\mathbf{z}_t \in \mathbb{R}^{H \times d}$
- 这相当于在 π 的架构上额外加一个 "query 通道" 从 backbone 提取 dynamics-aware feature

Table V 的 ablation 说明这个 embedding module 的设计很重要：
- "Point expert"（mirror action expert 的结构）：+0.5
- "Backbone query token (attend action)"（query 注入 backbone 且能 attend action tokens）：+1.5
- "Backbone query token"（不能 attend action）：+1.9
- "Ours"（cross-attention 取 final-layer hidden states）：+4.1

直觉：直接从 backbone final layer 提取最完整的 multimodal context，避免 perturb 预训练 representation，同时不让 point track head 干扰 action expert 的 flow matching。

### C. Loss Function

$$\mathcal{L} = \mathcal{L}_{\text{act}} + \omega_{pt} \|\widehat{\Delta P}_{t:t+H} - \Delta P_{t:t+H}\|_1$$

- $\mathcal{L}_{\text{act}}$：原始 action loss（OpenVLA-OFT 用 $\ell_1$，π 用 flow matching）
- $\omega_{pt} = 1.0$（最优，Table S7 显示 0.1 → 54.7%，1.0 → 57.0%，10.0 → 50.7%）
- 第二项：3D point displacement 的 $\ell_1$ loss

**关键：point track head 只在训练时存在，推理时完全丢弃。** 这就是 "privileged" 的含义——4D 信息只在训练时可见，用来 shape representation。

---

## III. 为什么 3D Point Tracks 是好的 privileged signal？

作者对比了多种 supervisory target（Table III, RoboCasa）：

| Method | SR | Δ |
|---|---|---|
| OpenVLA-OFT baseline | 33.1 | - |
| + Goal point set | 33.8 | +0.7 |
| + 2D point track | 37.0 | +3.9 |
| + Depth (via VAE latent) | 42.3 | +8.3 |
| + 3D point track (Ours) | **46.3** | **+13.2** |

三个维度的分析：

**1. Temporality**：Goal point set 只预测终端点集 $P_{t+H}$，相当于 mean-pool backbone embedding。提升只有 +0.7。说明 "未来某个时刻 scene 长什么样" 的稀疏信号太弱。3D track 在整个 horizon H 上 dense 监督，捕捉 fine-grained interaction。

**2. Spatiality**：2D point track 把 3D 投影到单视角，保持 temporal density 但失去 metric 3D 结构。+3.9 vs +13.2，说明 metric 3D 很关键。2D track 无法区分"物体远离相机"vs"物体变小"。

**3. Spatial redundancy**：Depth map 空间 dense，在 fixed camera 下大量像素是静态背景，temporal 冗余。而且 depth 没有 identity registration（同一个像素在不同时刻不一定对应同一物理点），无法建模 contact/articulation。+8.3 vs +13.2。

**4. Scene-robot interaction**（Table III 下半部分）：
- Only scene points：+2.1（只看环境如何动）
- Only robot points：+10.7（只看机器人如何动）
- Both（Ours）：+13.2

直觉：robot-only 主要捕获 self-motion，scene-only 弱化 contact sensitivity。World dynamics 必须同时建模 robot-environment interaction，这正是 manipulation 的本质。

---

## IV. 3D Point Track 的构造

### Simulation（LIBERO, RoboCasa）

直接从 simulator（MuJoCo）拿 ground-truth mesh：
1. 第一帧在 robot-centered 3D cube 内 crop mesh，在 mesh faces 上 sample $N_p = 1024$ 个 query points
2. 记录每个 point 的 face index + barycentric coordinates
3. Roll out action sequence，每帧用 face index + barycentric coords retrieve 同一个 surface point 的 3D 位置
4. 得到 $\{P_\tau\}_{\tau=1}^{T}$

这种方法保证 **identity consistency**：第 $j$ 个点在所有时刻对应同一个 mesh surface point。这是 point track 区别于 depth 的关键。

### Real World

用 off-the-shelf 3D point tracking model（SpatialTrackerV2）：
- Segmentation model（应该是 SAM 之类）在前景区域 sample 更多点
- 背景区域 uniform sample
- Fixed camera setup → 可以恢复稳定的 world coordinates

参考：SpatialTrackerV2 https://arxiv.org/abs/2411.16842

### Supervision target

定义 displacement：

$$\Delta p_j^\tau = p_j^{\tau+1} - p_j^\tau, \quad \tau \in \{t, \ldots, t+H\}$$

- $p_j^\tau = (x_j^\tau, y_j^\tau, z_j^\tau) \in \mathbb{R}^3$：第 $j$ 个 point 在 time $\tau$ 的 3D 位置
- $j \in \{1, \ldots, N_p\}$，$N_p = 1024$
- $\tau \in \{t, \ldots, t+H+1\}$：horizon 覆盖 $H+1$ 个时间点（因为 $H$ 个 displacement 需要 $H+1$ 个 position）

监督 $\Delta P_{t:t+H} = \{\Delta P_\tau\}_{\tau=t}^{t+H}$，每个 $\Delta P_\tau = \{\Delta p_j^\tau\}_{j=1}^{N_p}$。

---

## V. 关键 Ablation：$P_t$ 该怎么输入？

Table IV 是理解 Pri4R 设计哲学的核心：

| Method | Need $P_t$ at test? | Avg SR |
|---|---|---|
| OpenVLA-OFT baseline | No | 33.1 |
| + $P_t$ input to backbone | Yes | 33.3 (+0.2) |
| + $P_t$ input + track supervision | Yes | 34.5 (+1.4) |
| + Track supervision only (Ours) | No | **46.3 (+13.2)** |

直觉分析：

**如果 $P_t$ 输入到 backbone**：
- 推理时必须提供 3D point set（破坏原 VLA 接口）
- 在 VLM embedding space 引入新的 token type，perturb 预训练 representation
- Pick-and-Place 任务反而下降 -7.8（language 信号被稀释）
- 即便加上 track supervision，PnP 还是不恢复

**如果 $P_t$ 只输入到 point track head**：
- 推理时完全不需要 3D 输入
- Backbone 输入只有 image + language，和预训练分布一致，零 distribution shift
- Track head 通过 backprop 把 dynamics 信息"蒸馏"进 backbone 的 shared representation
- PnP 也提升 +1.2

这是 **privileged learning 的精髓**：辅助信号在训练时塑造 representation，推理时完全不可见。

Table S5 进一步验证：如果把 $P_t$ 从 point track head 也移除，让 head 从 $\mathbf{z}_t$ 直接 generate 整个 point track sequence，性能降到 28.7%（甚至低于 baseline）。说明 **point track head 需要看到当前 scene 的 3D 结构作为起点**，任务是"预测这个特定 scene 如何演化"而不是"凭空生成 scene"。

---

## VI. Point Track Head 的架构 Ablation（Table VI, LIBERO Long）

| Point Encoder | Fusion Module | SR | Δ |
|---|---|---|---|
| (no head) | - | 89.2 | - |
| PointNet | Ours | 80.8 | -8.4 |
| PtTransformer | Ours | 92.4 | +3.2 |
| Ours (PointMLP) | Transformer | 92.2 | +3.0 |
| **Ours (PointMLP)** | **Ours (FusionMLP)** | **94.4** | **+5.2** |

观察：
- PointNet 反而伤害性能（-8.4）。PointNet 的 max-pooling 丢失 per-point identity，而 Pri4R 需要 per-point displacement，identity 必须保留
- 重型 Point Transformer 不如简单 MLP
- Transformer fusion 也不如 MLP fusion

直觉：point track head 应该是"轻量的 readout module"，重活交给 backbone。重型 head 反而让 backbone "偷懒"，不让 dynamics 信号回流到 shared representation。这和 LoRA、adapters 的设计哲学一致——auxiliary module 越轻，main backbone 学到的越多。

---

## VII. 主要实验结果

### LIBERO（Table I）

| Model | Avg | Spatial | Object | Goal | Long |
|---|---|---|---|---|---|
| OpenVLA | 76.5 | 84.7 | 88.4 | 79.2 | 53.7 |
| π0 | 87.4 | 87.8 | 84.9 | 91.2 | 85.7 |
| π0 + Pri4R | 90.6 | 92.8 | 88.6 | 95.3 | 85.6 |
| π0.5 | 92.6 | 96.1 | 88.3 | 95.6 | 90.5 |
| π0.5 + Pri4R | 94.0 | **97.2** | 88.9 | 95.6 | 94.3 |
| OpenVLA-OFT | 92.7 | 90.8 | 98.2 | 96.4 | 85.5 |
| OpenVLA-OFT + Pri4R | **96.3** | 93.2 | **98.6** | **98.1** | **95.3** |

LIBERO-Long 上 OpenVLA-OFT + Pri4R 拿到 +9.8 的提升。Long suite 是 long-horizon 任务，需要多步 planning + 中间 contact，对 world dynamics 最敏感。

### RoboCasa（Table II）

| Model | Avg | PnP | Doors | Drawers | Knobs | Lever | Press | Insert |
|---|---|---|---|---|---|---|---|---|
| π0 | 38.8 | 24.0 | 45.0 | 78.0 | 29.0 | 57.3 | 57.3 | 0.0 |
| π0 + Pri4R | 42.2 | 24.0 | 49.0 | 84.0 | 30.0 | 71.3 | 59.3 | 2.0 |
| π0.5 | 52.9 | 54.3 | 51.0 | 75.0 | 28.0 | 79.3 | 60.0 | 4.0 |
| π0.5 + Pri4R | 57.0 | 52.0 | 68.5 | 89.0 | 33.0 | 86.7 | 54.7 | 5.0 |
| OpenVLA-OFT | 33.1 | 21.8 | 45.7 | 59.0 | 8.0 | 36.0 | 56.0 | 27.0 |
| OpenVLA-OFT + Pri4R | **46.3** | 23.0 | 61.7 | 80.0 | 25.0 | 66.7 | 79.3 | 34.0 |

RoboCasa 是 articulated interaction（doors, drawers, knobs, levers），这些任务**强依赖 world dynamics**——必须理解铰链、滑轨、旋转轴的 kinematic constraint。OpenVLA-OFT + Pri4R 拿到 +13.2 的平均提升，其中：
- Lever：+30.7（最显著）
- Press：+23.3
- Drawers：+21.0
- Doors：+16.0
- Knobs：+17.0

这些全都是 articulated object，baseline 容易在 "看起来对但物理上不对" 的 action 上失败。Pri4R 通过预测 point track，强迫模型理解 "把手拉了之后门会绕轴旋转" 这种 dynamics。

### Training Dynamics（Figure 3）

非常有趣的现象：
- 前 ~20K steps，Pri4R 性能低于 baseline（因为 point track loss 占用 representation capacity）
- 20K 之后迅速上升
- 达到 baseline peak 性能的速度是 baseline 的 **2.7×**
- 节省约 8× H200 GPU-days

直觉：早期模型在学"如何预测 scene 演化"，这个 skill 还没成熟到能帮助 action。一旦 dynamics representation 学好，action head 就能 leverage 这个 physically-aware context，迅速收敛。这说明 world dynamics 是 action prediction 的 **bottleneck feature**，baseline 在用低效的方式学这个 feature。

---

## VIII. Real-World 实验（Table VII, Figure 6）

四个 task：

1. **Height**（Pick-and-place over obstacle）：需要避障，理解 3D 几何
   - OpenVLA-OFT：83.3 → +Pri4R：96.7
   
2. **Spatial**（Pick-and-place into bin，seen/unseen 位置）：
   - Seen：100.0 → 100.0（已经 saturate）
   - Unseen：60.0 → 80.0（明显提升）
   
3. **Depth**（Pick farthest object，需要 depth-dependent geometry reasoning）：
   - Seen：45.9 → 79.2
   - Unseen：50.0 → 50.0
   
4. **Tracking**（Pick moving object，物体在 robot approaching 时被移动）：
   - Seen：75.0 → 100.0
   - Unseen：41.7 → 66.7
   - OOD：50.0 → 66.7

Tracking task 最能体现 world dynamics 价值：物体在移动，robot 必须持续 update grasp plan。Baseline 经常在 outdated location 停下并 close gripper on empty space，然后继续执行剩余 action chunk（说明它根本没意识到 grasp 失败）。Pri4R 持续 track 物体并 update 到新位置。

π0.5 在 real-world 上提升相对小（60.0 → 66.7），作者没深入解释。我猜测 π0.5 已经在 OpenX 上大规模预训练，representation 已经部分包含 dynamics，privileged supervision 的边际收益较小。

---

## IX. 与 SpatialForcing 的对比（Table S6）

SpatialForcing 用 VGGT（3D geometric foundation model）的 feature 做 alignment，给 VLA 注入 3D 结构 awareness，同样不需要 test-time 3D 输入。

| Method | Avg |
|---|---|
| OpenVLA-OFT | 92.7 |
| + SpatialForcing | 94.2 |
| + Pri4R | 95.0 |

Pri4R 略胜。作者归因：SpatialForcing 注入的是 **static 3D structure**，Pri4R 注入的是 **temporally dense 4D dynamics**。后者直接监督 interaction，前者只给场景几何先验。

参考：
- SpatialForcing: https://arxiv.org/abs/2510.12276
- VGGT: https://arxiv.org/abs/2503.11651

---

## X. 我的 Intuition 构建

让我从几个角度 build intuition：

### 1. Privileged Information 的本质

经典 SVM+LUPI（Vapnik 2009）的思想：训练时给模型更丰富的信息（如 medical diagnosis 训练时给 lab test，推理时只给症状）。模型学会"如果做了 lab test 会怎么判断"，这个 reasoning pattern 沉淀到 representation，即使推理时没有 lab test 也能 work。

Pri4R 是这个思想在 VLA 上的应用：训练时给 3D point track（lab test），推理时只用 image+language（症状）。Backbone 学会"如果理解了 scene dynamics 应该怎么 act"，这个理解能力沉淀到 shared representation。

### 2. 为什么 displacement 而不是 absolute position

两个原因：
- **Scale match**：action 也是 displacement（delta joint angle），point displacement 在同一量级，$\omega_{pt}$ 不用 tune
- **Identity-aware**：absolute position 让模型可以"作弊"——只学静态 scene prior（这个物体一般在哪），displacement 强制学"这个特定 scene 在这个特定 action 下怎么演化"

### 3. 为什么 MLP 比 Transformer fusion 好

这违反"越大越好"的直觉。关键在于 **gradient flow**：

- 重型 fusion module 自己有足够 capacity 处理 point track prediction
- Backbone 的 gradient signal 被稀释（fusion module 已经"消化"了大部分信号）
- 轻量 MLP 必须依赖 backbone 提供 rich feature，所以 dynamics 信号被强制 push 回 backbone

类似 LoRA 的哲学：auxiliary 越弱，main model 越强。

### 4. 为什么 LIBERO-Long 提升最大

Long-horizon 任务需要：
- 多步 planning
- 中间 contact 状态判断
- 子目标达成检测

这些全都需要 world dynamics awareness。Short-horizon task（如 Spatial）更多依赖 perception，dynamics 价值小。这印证了 "dynamics 是 action 的 bottleneck feature"。

### 5. 与世界模型的关系

Pri4R 不是一个 explicit world model（不生成未来 image/state）。它更像 "implicit world model"——把 dynamics 知识压缩进 representation，让 action head 自己用。

对比 DreamerV3 / DIAMOND 等 explicit world model：
- Explicit：生成未来 observation，policy 在 imagined rollout 中训练
- Pri4R：未来 observation 作为辅助 supervision，policy 直接在真实数据上训练

Pri4R 的优势：零 inference overhead，不依赖 world model 质量。劣势：dynamics 只在 representation 层面 implicit，无法做 planning/search。

参考 DreamerV3: https://arxiv.org/abs/2301.04104

### 6. Potential Limitations 我觉得作者没充分讨论

- **3D track 质量依赖**：real-world 用 SpatialTrackerV2 的 pseudo-label，如果 tracker 在 fast motion / occlusion 下失败，supervision 会 noisy。作者没分析 track 质量对性能的 sensitivity。
- **Point sampling 策略**：random mesh sampling 可能不是最优。关键 contact region（gripper-object 接触点）应该 sample 更密。
- **Long-horizon beyond action chunk**：$H = 10$ 大约 1 秒，更长 horizon 的 dynamics 没被监督。LIBERO-Long 提升大可能恰恰因为它需要更长 dynamics reasoning，而 Pri4R 只能间接帮助。
- **Pretraining 阶段的潜力**：作者自己提到，在 Embodiment X 等大规模 pretraining 上用 point track supervision 可能收益更大。这是 clear future work。

---

## XI. 相关工作的更广阔 context

让我联想一些相关工作构建更完整的 landscape：

### VLA with forecasting
- **Gen2Act** [Bharadhwaj 2024]：用 human video generation 做 high-level planner
- **3D-VLA** [Zhen 2024]：3D vision-language-action generative world model
- **DreamVLA** [Zhang 2025]：用 comprehensive world knowledge 做 VLA
- **VideoVLA** [Shen 2025]：video generator 作为 generalizable robot manipulator
- **CoT-VLA** [Zhao 2025]：visual chain-of-thought reasoning

这些方法都需要 inference-time generation，有额外 latency。Pri4R 的差异化是 zero inference overhead。

### Point track for manipulation
- **SpatialTrackerV2** [Xiao 2025]：3D point tracking 的 SOTA
- **ST4RTrack** [Feng 2025]：simultaneous 4D reconstruction and tracking
- **TAPVid-3D**：3D tracking benchmark
- **Any-point trajectory modeling** [Wen 2023]：policy learning from trajectories
- **RoboPoint** [Yuan 2024]：spatial affordance prediction

Pri4R 用 point track 做 supervision，这些方法用 point track 做 representation 或 affordance。

### VLA with 3D awareness
- **3D-CAVLA** [Bhat 2025]：depth + 3D context for VLA
- **PointVLA** [Li 2025]：inject 3D world into VLA
- **GeoVLA** [Sun 2025]：3D representations in VLA
- **SpatialForcing** [Li 2025]：implicit spatial alignment

这些方法多数需要 test-time 3D input 或 feature alignment。Pri4R 的 privileged 范式更干净。

### Privileged information in robotics
- **DAgger** [Ross 2011]：用 expert oracle 作 privileged
- **Imitation from observation**：用 human video 作 privileged
- **RL with privileged critic**：sim-to-real 中用 sim state 训 critic，real 中只用 image

Pri4R 在 imitation learning 中用 4D geometry 作 privileged，是这个 family 的新成员。

参考：
- DAgger: https://arxiv.org/abs/1011.0686
- PointVLA: https://arxiv.org/abs/2503.07511
- 3D-VLA: https://arxiv.org/abs/2403.09631

---

## XII. 实施细节的关键数字

**Simulation**（Table S3）：
- LIBERO：4 GPU，batch 64（OpenVLA-OFT）/ 128（π），90K / 30K steps
- RoboCasa：4 GPU，batch 64/128，120K / 30K steps

**Real-world**（Table S1, S2）：
- 4× H200
- OpenVLA-OFT：LR 5e-4, batch 32, LoRA rank 32, action chunk 10
- π0.5：LR 5e-5, AdamW, cosine decay, 10K warmup, EMA 0.999
- $\omega_{pt} = 1.0$ across all settings
- $N_p = 1024$ points（Table S8 显示 256 → 48.8%, 512 → 53.9%, 1024 → 57.0%）

1024 是 sweet spot：太少不足以 capture interaction-relevant geometry，太多 supervision 信号变 noisy 且 memory 贵。

---

## XIII. 总结：Pri4R 的设计哲学

把 Pri4R 抽象成几个原则：

1. **Privileged > Test-time auxiliary**：4D 信息训练时用，推理时丢弃，避免 inference overhead
2. **Displacement > Absolute**：和 action 形式对齐，scale match，避免 cheating
3. **Identity-preserving > Dense**：point track 比 depth/video 好，因为 temporal identity registration
4. **Light head > Heavy head**：让 backbone 承担 dynamics learning，auxiliary 越轻 main 越强
5. **Auxiliary-only input > Backbone input**：$P_t$ 不进 backbone，避免 distribution shift
6. **Robot + Scene > Either alone**：world dynamics 必须 cover robot-environment interaction

这套原则不只适用于 point track，可以推广到任何 privileged supervision 的 VLA 设计。比如未来用 tactile signal、force/torque、contact graph 作 privileged，都可以套这个框架。

这是一个简洁、可推广、实验扎实的 work。最大亮点是把 privileged information 这个老 idea 在 VLA 时代重新激活，并给出清晰的 ablation 证明每个设计选择的必要性。对未来 VLA pretraining 的启示是：大规模 3D track 标注（即使 pseudo-label）可能比单纯堆 action data 更高效。
