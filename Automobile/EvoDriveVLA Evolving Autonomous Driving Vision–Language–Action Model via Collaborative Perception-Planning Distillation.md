---
source_pdf: EvoDriveVLA Evolving Autonomous Driving Vision–Language–Action Model via
  Collaborative Perception-Planning Distillation.pdf
paper_sha256: 72b39f1218fab206635e59fba0ac194893b1e68c41559257fdd04056cda53e40
processed_at: '2026-08-04T05:59:30-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# EvoDriveVLA 用人话讲

## 这篇 paper 到底在干嘛？

想象你在教一个新手司机开车。传统做法是你自己坐副驾,看一样的路况,然后说"往左打方向盘"。问题是——你自己也是新手,你能教他啥？

这篇 paper 的核心 idea 就是:**找个"开过这条路的老司机"来当教练**。老司机知道前面 3 秒会发生什么（前方有车变道、前面要转弯），所以他给的指导肯定比你自己瞎猜准。

但光有老司机还不够,还有一个问题：新手在学开车的时候,容易把"基本视觉能力"也练坏了。比如他为了适应某个特殊路况,把"认红绿灯"这种通用能力给忘了。所以 paper 又加了一个"保镖"——冻结新手学车前的视觉能力,随时把他拉回原来的水平。

这两个东西合起来就是 EvoDriveVLA。

---

## 两个核心问题

### 问题 1: visual encoder 一 fine-tune 就"变傻"

VLA 模型里 visual encoder 就是"眼睛"。预训练时这双眼睛在几亿张图上学会了认东西——车、人、路、天空。但是当你 unfreeze 它在 driving 数据上 fine-tune 时,会发生一件尴尬的事：

眼睛为了适配 driving 任务,**开始"过拟合"**,把一些通用视觉能力给丢了。这就好比一个摄影师为了拍婚纱照,把眼睛练得只会看白纱,结果连红绿灯都认不清了。

学术界一直吵这个问题：到底 freeze 还是不 freeze？
- Freeze：保住通用能力,但适应不了 driving 特殊场景
- Unfreeze：能适应,但通用能力掉

**EvoDriveVLA 的解法**：让 student encoder 自由学 driving 任务,但是拿一个 SFT 前的 frozen copy 当 "self-anchor teacher",用 MSE loss 把 student 拉回来,别跑太远。而且这个"拉"不是均匀的——**对驾驶关键的 region（前方道路、车道线）拉得紧,对天空、远景拉得松**。

### 问题 2: teacher 跟 student 一样菜,蒸馏个寂寞

之前的 distillation 方法（DiMA、DistillDrive）有个致命问题：**teacher 和 student 输入完全一样**。既然输入一样,凭啥 teacher 能教 student？这就好比你和你同桌一起考试,你俩水平差不多,你抄他能抄出什么花来？

**EvoDriveVLA 的解法**：给 teacher 喂"未来信息"——未来 3 秒的图像 + 未来 3 秒的 ego status（速度、加速度、steering）。这样 teacher 在 planning 上绝对吊打 student,蒸馏才有意义。

这个叫 **privileged information**——teacher 训练时作弊,student 推理时老老实实。但 teacher 把"看到未来之后怎么做决策"的 reasoning 过程蒸馏给 student,让 student 学会"预判式驾驶"。

---

## 怎么实现的？拆开看

### Part 1: Self-Anchored Visual Distillation

**Step 1**: 训练前,把 student 的 visual encoder 复制一份,冻结。这个就是 self-anchor teacher。

**Step 2**: 设计一个 AnchorFormer,它的工作是给每个 visual token 打个权重（0 到 1 之间）,表示"这个 token 对当前驾驶任务有多重要"。

AnchorFormer 怎么算权重？它看三个东西：
- 当前 visual tokens（图像特征）
- Prompt + ego state（任务描述 + 车的状态）
- **Ground-truth future trajectory**（未来要去哪）

为什么要看 future trajectory？因为如果未来要左转,那左前方 region 就重要；如果直行,正前方 region 重要。**用未来的去向反推现在该看哪里**——这是 paper 的一个很妙的设计。

公式（这就是个 attention-style gating）：
$$
\mathbf{S}_a = \text{AnchorScorer}(\tilde{\mathbf{z}}_v^{tea} \odot \tilde{\mathbf{q}})
$$

- $\tilde{\mathbf{z}}_v^{tea}$：teacher 的 visual tokens,shape (N_v, d),N_v 是 token 数,d 是 hidden dim
- $\tilde{\mathbf{q}}$：query tokens,融合了 prompt/ego/trajectory 信息
- $\odot$：Hadamard product（逐元素相乘）
- $\mathbf{S}_a$：每个 token 的 score

然后过个温度 sigmoid：
$$
\mathbf{W}_a = \frac{1}{1 + \exp(-\mathbf{S}_a / \tau_v)}, \quad \tau_v = 2.0
$$

$\tau_v$ 是温度,2.0 让权重分布平滑点,别太极端。

**Step 3**: 用加权 MSE 做 distillation：
$$
\mathcal{L}_a = \frac{1}{N_v} \sum_{i=1}^{N_v} \mathbf{W}_a^{(i)} \|\mathbf{z}_v^{tea(i)} - \mathbf{z}_v^{stu(i)}\|_2^2
$$

重要 token（W_a 接近 1）：student 必须紧跟 teacher,保住 representation。
不重要 token（W_a 接近 0）：student 自由发挥,适应 driving 任务。

**人话总结**：关键 region 严格"保鲜",次要 region 允许"进化"。

---

### Part 2: Oracle-Guided Trajectory Distillation

这部分是 paper 的重头戏。

**Step 1: 构造 oracle teacher**

拿一个 Qwen2.5-VL 3B,输入是：
- Current images（6 个视角,和 student 一样）
- **Future images**（未来 0s/1s/2s/3s,每秒一张,GT 的）
- **Future ego status**（未来 3 秒的速度、加速度、steering,GT 的）
- Historical trajectory + ego status（和 student 一样）

输出：未来 3 秒的 waypoints（6 个点,0.5 秒间隔）

**Step 2: Coarse-to-Fine refinement**

Oracle teacher 先做一次粗预测 $W_t^c$（coarse）,然后把粗预测喂回去再做一次精预测 $W_t^f$（fine）。

这其实就是让 LLM 做 "given this rough guess, refine it" 的任务。prompt 里明确写了：
> "I can provide the trajectory predictions from the student model here, and you can also refine them based on this: [...]"

**为什么这么做？** 粗预测给大方向（转弯还是直行）,精预测做局部微调（轨迹平滑、物理可行）。模仿人类开车"先看大方向,再精控方向盘"。

**Step 3: MC-Dropout 采样**

只有一条 coarse 和一条 fine 还不够,作者想要"多条候选轨迹"挑最好的。怎么做？

对 coarse 和 fine 的 hidden states 做 10 次 stochastic dropout（p=0.1）,每次 dropout 等于从模型 posterior 里采一个 slightly different sub-network,给出 slightly different 轨迹。

$$
\mathbf{h}^{(n)} = \text{Dropout}(\mathbf{h}; p=0.1), \quad n = 1, \ldots, 10
$$

10 次采样得到 10 条轨迹,加上原本的 coarse + fine,一共 12 条候选。

**Step 4: 挑最优轨迹做 soft target**

拿这 12 条候选轨迹跟 ground truth 算 cross-entropy,挑 loss 最小的那条：
$$
\hat{k} = \arg\min_{\mathbf{l}_k \in S_l} \mathcal{L}_{\text{CE}}(\mathbf{l}_k, \mathbf{W}^*)
$$

**Step 5: 双层蒸馏**

挑出来的最优轨迹的 hidden states 和 logits 都用来教 student：
- Hidden state 层面：MSE loss,让 student 学 teacher 的"中间推理"
- Logits 层面：KL divergence,让 student 学 teacher 的"输出分布"

$$
\mathcal{L}_h = \frac{1}{N_t} \sum_{i=1}^{N_t} \|\mathbf{h}_{stu}^{(i)} - \mathbf{h}_{\hat{k}}^{(i)}\|_2^2
$$

$$
\mathcal{L}_l = \text{KL}\big(\text{softmax}(\mathbf{l}_{\hat{k}}/\tau_t) \,\|\, \text{softmax}(\mathbf{l}_{stu}/\tau_t)\big), \quad \tau_t = 5
$$

$\tau_t = 5$ 是温度,让 softmax 变软,传递更多 "dark knowledge"（Hinton 原版 KD 的精髓）。

---

## Total Loss 怎么配的？

$$
\mathcal{L}_{all} = \mathcal{L} + 0.05 \cdot \mathcal{L}_a + 0.1 \cdot \mathcal{L}_h + 0.2 \cdot \mathcal{L}_l
$$

- $\mathcal{L}$：原始 trajectory prediction loss（NLL）
- $0.05 \cdot \mathcal{L}_a$：anchor visual distillation,权重很小,只是 regularization
- $0.1 \cdot \mathcal{L}_h$：hidden state distillation,中等权重
- $0.2 \cdot \mathcal{L}_l$：logit distillation,权重最大,主要 supervision

**权重设计透露了作者意图**：最信"输出对齐"（logit）,次信"中间推理对齐"（hidden）,最不信"视觉 representation 对齐"（anchor,只当保底）。

---

## 实验结果说了什么？

### Open-loop（nuScenes）

**ST-P3 split**：
- L2 error: EvoDriveVLA 0.26, DiMA 0.27, OpenDriveVLA 0.33
- Collision rate: EvoDriveVLA 0.07, OpenDriveVLA 0.32（降 78%！）

**UniAD split**：
- L2 error: EvoDriveVLA 0.52, DiMA 0.57（降 9%）
- Collision: EvoDriveVLA 0.12, DiMA 0.07（略差）

**亮点**：collision rate 在 ST-P3 split 上大幅下降,说明 oracle teacher + coarse-to-fine 在长 horizon（3s）上的稳定性收益最大。

### Closed-loop（NAVSIM）

| Model | PDMS |
|---|---|
| Qwen2.5-VL 3B (baseline) | 81.9 |
| Qwen2.5-VL 8B | 83.3 |
| InternVL3-8B | 83.3 |
| **EvoDriveVLA (3B distilled)** | **85.3** |

**这是最亮眼的结果**：3B 蒸馏版打 8B 原版,说明 distillation 比 scaling up 更高效。

Ego Progress（EP）从 77.6 涨到 81.1,说明蒸馏让 student "开得更进取"而不只是"更保守"。

### Ablation

| 配置 | L2 Avg |
|---|---|
| 无任何蒸馏 | 0.55 |
| + Trajectory KD | 0.54（几乎没提升）|
| + Traj Refine | 0.53 |
| + MC-Dropout | 0.53 |
| + Visual KD | 0.52 |

**有意思的观察**：每个 component 单独加都没什么用,但全部加起来才能到 0.52。这是一个 **system-level design**——各模块协同才有用,单独拆开效果不明显。

---

## Oracle teacher 自己有多强？

| Split | L2 Avg | Collision Avg |
|---|---|---|
| ST-P3 | 0.14 | 0.04 |
| UniAD | 0.20 | 0.04 |

对比 student 最优结果 0.26/0.52,oracle teacher 的 L2 几乎是 student 的一半。**这就是它能教好学生的根本原因**——teacher 真的比 student 强很多,不是"平辈互相教"。

---

## 我觉得最妙的几个点

### 1. Privileged information 的用法

之前 driving distillation 工作的 teacher 和 student 同输入,本质上没有信息增益。EvoDriveVLA 给 teacher 喂 future info,让 teacher 真正"更强",这才是 distillation 的正确姿势。

这其实是 **Learning by Cheating** (Chen et al. 2019) 的思路——privileged learning。但 EvoDriveVLA 把它用在 supervised distillation 而不是 RL 上,更简洁。

reference: https://arxiv.org/abs/1912.12294

### 2. Trajectory-guided anchor

AnchorFormer 用 future trajectory 反推当前 visual token 重要性,这是一个很 elegant 的设计。本质上是**用 planning 信号指导 perception 的 attention**——perception 和 planning 联合优化。

### 3. MC-Dropout 当 test-time augmentation 用

MC-Dropout 原本是估计 Bayesian uncertainty 的,这里用来**采样多条候选轨迹挑最好的**。这其实是 test-time augmentation (TTA) 的思路,但用在 distillation 的 target 生成上。

reference: https://arxiv.org/abs/1506.02142

### 4. 3B 打 8B

这是 paper 最有说服力的结果。distillation 比 scaling up 更高效,这对算力有限的团队是个好消息。

---

## 几个我存的疑

### 1. Future images 是 GT 还是生成的？

Paper 没明确说,但从 prompt 看（Future 0s/1s/2s/3s）像是 nuScenes 的 GT future frames。这意味着 oracle teacher 只能 **offline 训练时用**,online 部署时没有 future info。

这其实没问题（teacher 只在训练时存在,student 推理时不依赖 future）,但 paper 应该说清楚。如果未来用 world model 生成 future 而不是用 GT,整个 pipeline 可以变成 fully online。

### 2. Coarse-to-Fine 是 prompt-level 的

Refinement 是靠在 prompt 里喊 "refine this trajectory",依赖 LLM 的 instruction following 能力。如果 LLM 不够强（比如换成 1B 模型）,这个 refine 可能失效。

更好的做法可能是 architecture-level refinement（比如专门的 refinement decoder）,而不是 prompt-level。

### 3. Ablation 里每个 component 收益都太小

0.55 → 0.52 的总收益,单 component 贡献都不到 0.02。这在 L2 metric 上其实不太显著。真正亮点是 collision rate,但 ablation 没拆 collision。

这种"单 component 没用,组合才有用"的方法,通常 reproducibility 会有挑战——去掉任何一个 component 都可能失去大部分收益。

### 4. MC-Dropout N=10 是怎么定的？

论文没做 sensitivity analysis。N 越大肯定越好但成本越高,这个 trade-off 在哪？N=10 是 heuristic 还是经过实验验证？

---

## 联想到的相关方向

### 1. World model 替代 GT future

Oracle teacher 现在用 GT future images。如果换成 world model 生成的 future（比如 GAIA-1, OccWorld, Drive-WM）,整个 pipeline 可以变成 fully online,不依赖 dataset 的 GT future。

- GAIA-1: https://arxiv.org/abs/2309.17080
- OccWorld: https://arxiv.org/abs/2404.01825
- Drive-WM: https://arxiv.org/abs/2312.07744

### 2. EMA teacher 替代 frozen snapshot

Self-anchor teacher 现在是 SFT 前的 frozen copy。但如果训练 step 很多,student 漂移太远,这个 frozen anchor 可能反而拖后腿。

更好的做法可能是 EMA（exponential moving average）更新,像 MOCO/BYOL/DINO 那样。teacher 慢慢跟着 student 走,既保持 anchor 作用又不会太"过时"。

- MOCO: https://arxiv.org/abs/1911.05722
- BYOL: https://arxiv.org/abs/2006.07733
- DINO: https://arxiv.org/abs/2104.14294

### 3. Diffusion policy 的多模态轨迹

DiffusionDrive 用 diffusion 生成多模态轨迹,EvoDriveVLA 用 MC-Dropout 采样。两者其实可以结合——用 diffusion 生成候选,用 MC-Dropout 在 diffusion 的 hidden states 上采样,多样性可能更好。

- DiffusionDrive: https://arxiv.org/abs/2411.15239
- Diffusion Policy: https://arxiv.org/abs/2303.04137

### 4. Privileged RL 的 driving 版本

EvoDriveVLA 是 privileged learning 的 supervised 版本。RL 版本是 teacher 用 simulator state,student 用 observation。如果 driving 有一个好的 simulator（比如 CARLA,或者 3DGS-based simulator）,可以做 privileged RL distillation。

- Learning by Cheating: https://arxiv.org/abs/1912.12294
- Privileged RL: https://arxiv.org/abs/1511.07279

### 5. VLM visual encoder degradation 的通用解法

EvoDriveVLA 的 trajectory-guided anchor 是 task-specific 的（针对 driving）。但 VLM 社区普遍有 visual encoder fine-tune degradation 问题。是否能设计一个 task-agnostic 的 anchor distillation？比如用 CLIP pre-train feature 做 anchor,而不是 self-anchor。

- Cambrian-1: https://arxiv.org/abs/2406.16860
- Eagle: https://arxiv.org/abs/2408.15998

---

## 最后总结一句

EvoDriveVLA 的核心 insight 就一句话：**distillation 的 teacher 必须真的比 student 强,否则就是瞎折腾**。它通过给 teacher 喂 future info 实现了这一点,同时用 self-anchor 保住了 student 的 visual encoder 不被 fine-tune 打坏。

实验上 3B 打 8B 的结果很扎实,说明这个设计确实有效。但每个 component 单独贡献都小,系统性强,replicability 可能是挑战。

---

相关 link 汇总：
- EvoDriveVLA GitHub: https://github.com/heycjj/EvoDriveVLA
- nuScenes: https://www.nuscenes.org/
- NAVSIM: https://github.com/autonomousvision/navsim
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- DiMA: https://arxiv.org/abs/2503.04218
- DistillDrive: https://arxiv.org/abs/2507.07291
- Hinton KD: https://arxiv.org/abs/1503.02531
- MC-Dropout: https://arxiv.org/abs/1506.02142
- TinyBERT: https://arxiv.org/abs/1909.10351
- Learning by Cheating: https://arxiv.org/abs/1912.12294
- DINO: https://arxiv.org/abs/2104.14294
- BYOL: https://arxiv.org/abs/2006.07733
- MOCO: https://arxiv.org/abs/1911.05722
- DiffusionDrive: https://arxiv.org/abs/2411.15239
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- GAIA-1: https://arxiv.org/abs/2309.17080
- OccWorld: https://arxiv.org/abs/2404.01825
- Cambrian-1: https://arxiv.org/abs/2406.16860
- Eagle VLM: https://arxiv.org/abs/2408.15998
- UniAD: https://github.com/OpenDriveLab/UniAD
- VAD: https://github.com/hustvl/VAD
- OpenDriveVLA: https://arxiv.org/abs/2503.23463

---

# EvoDriveVLA 深度讲解：一个 collaborative perception-planning distillation framework

很高兴能从 Karpathy 视角来拆解这篇 paper。它做的事情很有意思，把 knowledge distillation 在 autonomous driving VLA 上的"两个老问题"重新打了一遍，并且提出了一个非对称 teacher-student 的设计。下面我会非常详细地把架构、公式、动机、实验一起捋一遍，重点放在 build intuition。

---

## 1. 这篇 paper 想解决什么问题？

VLA (Vision-Language-Action) 模型在 autonomous driving 上的一个核心痛点：**当你 unfreeze visual encoder 做 SFT (supervised fine-tuning) 时，pre-training 阶段学到的 general visual representation 会被打掉**。这很像 LLM 里 QLoRA 的研究观察：frozen encoder 能保 representation，unfrozen encoder 能 adapt downstream task，但二者很难兼得。论文里把这个叫做 **degradation-adaptation dilemma**。

第二个痛点：**trajectory planning 在长 horizon 上不稳定**。现有 distillation 方法 (DiMA, DistillDrive) 的 teacher 模型和学生用同样的输入 (current observation)，所以 teacher 在 planning 上根本不比 student 强多少，蒸馏几乎是"平辈之间相互教"，收益有限。

EvoDriveVLA 给出的对策：
- **Self-Anchored Visual Distillation**：用 SFT 前的 visual encoder 拷贝作为 "self-anchor teacher"，让它提供 anchor 约束，student encoder 可以学 task-specific 特征但又不偏离 original representation 太远。同时引入 trajectory-guided token-level 加权，把 anchor 约束集中在"驾驶相关的关键区域"。
- **Oracle-Guided Trajectory Distillation**：构造一个 future-aware oracle teacher，喂给它 future images + future ego status 这种 "privileged information"，这样 teacher 在 planning 上确实比 student 强。再用 coarse-to-fine refinement + MC-Dropout 采样得到多条候选轨迹，挑最优的作为 soft target 蒸馏。

reference:
- DiMA paper: https://arxiv.org/abs/2503.04218
- DistillDrive (ICCV 25): https://arxiv.org/abs/2507.07291
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Hinton original KD: https://arxiv.org/abs/1503.02531
- MC-Dropout (Gal & Ghahramani): https://arxiv.org/abs/1506.02142

---

## 2. Preliminary: VLA 在 driving 上的 formulation

在每个时间步 t，模型接收：
- Multi-view camera images $\mathcal{T}_t = \{I_t^{(v)}\}_{v=1}^{V}$，V 是视角数 (nuScenes 通常是 6 个 cam)
- Textual instruction prompt $P_t$
- Ego-vehicle state $S_t = (x_t, y_t, v_t, a_t, \delta_t)$，分别是 position, velocity, acceleration, steering angle

输出未来 waypoints 序列：
$$W_t = \{w_{t+\tau}\}_{\tau=1}^{T}$$
其中每个 waypoint $w_{t+\tau} = (x_{t+\tau}, y_{t+\tau})$ 是车在未来 t+τ 时刻的 2D 位置 (ego coordinate)。T 通常是 6 (3 秒，0.5 秒间隔)。

把所有 observation 写成 $\mathcal{O}_t = (\mathcal{T}_t, P_t, S_t)$，目标学一个 policy $p_\theta$ 最小化 NLL：

$$
\mathcal{L} = - \sum_{\tau=1}^{T} \log p_\theta\big(\mathbf{w}_{t+\tau} = \mathbf{w}_{t+\tau}^{*} \mid \mathcal{O}_t, \mathbf{w}_{<t+\tau}^{*}\big) \tag{1}
$$

这里：
- $\theta$ 是 model parameters
- $\mathbf{w}_{<t+\tau}^{*} = \{w_{t+1}^{*}, \ldots, w_{t+\tau-1}^{*}\}$ 是 ground-truth 的前 τ-1 个 waypoint，做 teacher forcing
- 上标 $*$ 表示 ground truth
- 下标 t+τ 表示时刻

**Intuition**：这是一个 autoregressive sequence modeling 问题，每个 waypoint 条件于前面的所有 waypoint + observation，所以本质上是 next-token prediction，只是 token 是连续坐标。这就让 LLM 的训练范式可以直接迁移过来。

---

## 3. Self-Anchored Visual Distillation：保住 visual encoder 的"灵魂"

### 3.1 为什么需要 self-anchor？

VLM SFT 的一个争议话题：visual encoder 到底要不要 freeze？
- 一派（Tong et al. 2024, Shi et al. 2024）：unfreeze 能 cross-domain adapt
- 另一派（Karamcheti 2024, Kachaev 2025）：直接 fine-tune 会把大规模预训练 representation 打掉，造成 OOD 泛化变差

EvoDriveVLA 的做法：**用一个 SFT 前的 frozen encoder copy 作为 teacher，做 anchor distillation**。这个 teacher 不是来自外部更强模型，而是来自 student 自己的"过去快照"。这有点像 EMA teacher (MOCO 风格) 或者 self-distillation (DINO 风格) 的简化版本，但 anchor 的作用是 **regularization**，让 student 不要偏离原 representation 太远。

reference:
- DINO self-distillation: https://arxiv.org/abs/2104.14294
- MOCO: https://arxiv.org/abs/1911.05722

### 3.2 AnchorFormer：trajectory-guided token-level anchoring

普通的 anchor distillation 是 sample-level 的（整张图一个 loss），这里升级为 **token-level**，并且权重由 trajectory 信息引导——驾驶场景里车前方、车道线这些 region 显然比天空重要。

**输入准备**：
- Self-anchor teacher visual tokens: $\mathbf{z}_v^{tea}$
- Student visual tokens: $\mathbf{z}_v^{stu}$
- Prompt tokens: $\mathbf{z}_p$
- Ego state tokens: $\mathbf{z}_s$
- Ground-truth future trajectory tokens: $\mathbf{z}_{w^{*}}$
- 一组 learnable query tokens: $\mathbf{q}$

把 observation tokens 拼起来 $\mathbf{z}_o = [\mathbf{z}_v^{tea}, \mathbf{z}_p, \mathbf{z}_s]$，和 trajectory tokens $\mathbf{z}_{w^{*}}$ 以及 queries $\mathbf{q}$ 一起喂进 AnchorLayer：

$$
\tilde{\mathbf{z}}_o, \tilde{\mathbf{z}}_{w^{*}}, \tilde{\mathbf{q}} = \text{AnchorLayer}(\mathbf{z}_o, \mathbf{z}_{w^{*}}, \mathbf{q}) \tag{2}
$$

AnchorLayer 结构上等价于一个 LLM decoder layer (multi-head self-attention + FFN)，初始权重从 Qwen2.5-VL 3B 的最后一个 LLM layer copy 过来。这样 query tokens 可以 attend 到 visual/prompt/state/trajectory 所有信息。

**Anchor score 计算**：用 Hadamard product (逐元素乘) 把 visual tokens 和 query tokens 融合，再过一个 linear layer AnchorScorer：

$$
\mathbf{S}_a = \text{AnchorScorer}\big(\tilde{\mathbf{z}}_v^{tea} \odot \tilde{\mathbf{q}}\big) \tag{3}
$$

- $\odot$ 是 Hadamard product，shape 必须一致 (N_v × d)
- $\tilde{\mathbf{z}}_v^{tea}$ 是 teacher 的视觉 token，shape (N_v, d)，N_v 是 visual token 数，d 是 hidden dim
- $\tilde{\mathbf{q}}$ 是 broadcast 或者 pooling 后 query tokens

**温度 sigmoid 归一化**：

$$
\mathbf{W}_a = \frac{1}{1 + \exp(-\mathbf{S}_a / \tau_v)} \tag{4}
$$

- $\tau_v = 2.0$ 是温度，控制 sigmoid 的陡峭程度。温度大一点让权重分布更平滑，不会过度集中在某些 token 上导致梯度爆炸
- $\mathbf{W}_a \in (0, 1)^{N_v}$ 是每个 visual token 的 anchor 权重

**Intuition**：这其实是一个 attention-style gating。query tokens 充当"驾驶任务专家"，visual tokens 是"图像特征提供者"，二者融合后判断每个 visual token 对当前 driving 任务的重要程度。Trajectory 信息进入 query，所以 anchor 权重会被"未来要去哪儿"反向引导——前方道路、转弯方向对应 region 权重更高。

### 3.3 Visual distillation loss

用 MSE 把 student 拉向 teacher，但每个 token 的 weight 不同：

$$
\mathcal{L}_a = \frac{1}{N_v} \sum_{i=1}^{N_v} \mathbf{W}_a^{(i)} \|\mathbf{z}_v^{tea(i)} - \mathbf{z}_v^{stu(i)}\|_2^2 \tag{5}
$$

- $N_v$ 是 visual token 数
- $i$ 是 token index
- $\mathbf{W}_a^{(i)} \in (0,1)$ 是 anchor 权重，重要的 token 权重接近 1，约束强；不重要的接近 0，约束弱
- 这其实是 **selective distillation**：关键区域严格保持 pre-training representation，次要区域允许 student 自由 adapt

---

## 4. Oracle-Guided Trajectory Distillation：用"作弊"的 teacher 教学生

### 4.1 核心想法：teacher 要比 student 强才有意义

现有 distillation 方法 (DiMA, DistillDrive) 的 teacher 和 student 输入一样，所以 teacher 在 trajectory 预测上没有"信息优势"。EvoDriveVLA 给 teacher 喂 **privileged information**：

- **Future images** $I_{t+1}, \ldots, I_{t+T}$（未来 T 秒的相机图）
- **Future ego status** $S_{t+1}, \ldots, S_{t+T}$（未来速度、加速度、steering）

这是"作弊"，但只是 teacher 在训练时作弊，student 在推理时仍然只看 current observation。这就是 privileged learning / learning by cheating 的思路。

reference:
- Learning by Cheating (Chen et al.): https://arxiv.org/abs/1912.12294
- Privileged learning: https://arxiv.org/abs/1511.07279

### 4.2 Coarse-to-Fine Trajectory Refinement

Oracle teacher 不只是看 future 一次，而是 **两阶段预测**：

**Stage 1: Coarse prediction** — 基于所有未来观测生成粗轨迹：
$$
p_\theta(W_t^c \mid \cdot) = \prod_{\tau=1}^{T} p_\theta(w_{t+\tau} \mid \mathcal{O}_{<t+T}, w_{<t+\tau}) \tag{6a}
$$

**Stage 2: Fine prediction** — 把 coarse trajectory 喂回去再做一次：
$$
p_\theta(W_t^f \mid \cdot) = \prod_{\tau=1}^{T} p_\theta(w_{t+\tau} \mid \mathcal{O}_{<t+T}, W_t^c, w_{<t+\tau}) \tag{6b}
$$

其中 $\mathcal{O}_{<t+\tau} = \{\mathcal{O}_{t+1}, \ldots, \mathcal{O}_{t+\tau-1}\}$ 是 t+1 到 t+τ-1 时刻的观测集合。

**Intuition**：这模仿人类驾驶的"先看大方向，再精确控制"两阶段思维。Coarse 阶段用全局 future 信息决定大方向 (转弯？直行？)；Fine 阶段在 coarse 基础上做局部修正，使轨迹更平滑、更物理可行。本质上是一个 iterative refinement，类似 diffusion 的 denoising step 或者 LLM 的 self-refine。

**训练时**两个 trajectory head 同时优化，loss 都对 ground-truth 做 NLL。

### 4.3 MC-Dropout trajectory sampling

Coarse 和 fine 各给出一条轨迹还不够。论文想做 **多候选轨迹蒸馏**，但又不想像 DistillDrive 那样依赖预定义 planning vocabulary。于是用 MC-Dropout：

对每个 hidden state $\mathbf{h} \in S_h = \{\mathbf{h}_c, \mathbf{h}_f\}$（coarse 和 fine 的 hidden states），加 N 次 stochastic dropout：

$$
\mathbf{h}^{(n)} = \text{Dropout}(\mathbf{h}; p), \quad n = 1, \ldots, N \tag{7}
$$

- $p = 0.1$ 是 dropout 率
- $N = 10$ 是采样次数
- model parameters 保持不变，只对 hidden state 做 perturbation

然后过 lm_head 得到 logits：

$$
S_h \gets S_h \cup \{\mathbf{h}^{(n)}\}_{n=1}^N, \quad S_l \gets S_l \cup \{\mathbf{l}^{(n)} = \text{lm.head}(\mathbf{h}^{(n)})\}_{n=1}^N \tag{8}
$$

**Intuition**：MC-Dropout 在 inference 阶段近似 Bayesian model uncertainty。每次 dropout 等价于从 posterior 上采一个 sub-network，不同 sub-network 给出略微不同的轨迹，从而产生 trajectory diversity。这比 vocabulary-based 多样性更"data-driven"——多样性是从模型本身的不确定性里来的，而不是预先定义的。

reference:
- MC-Dropout Gal & Ghahramani: https://arxiv.org/abs/1506.02142
- Dropout as Bayesian: https://arxiv.org/abs/1506.02142

### 4.4 Trajectory distillation loss

从候选 logits 集合 $S_l$ 里挑出与 ground-truth 最接近的那条作为 soft target：

$$
\hat{k} = \arg\min_{\mathbf{l}_k \in S_l} \mathcal{L}_{\text{CE}}(\mathbf{l}_k, \mathbf{W}^{*}) \tag{9}
$$

然后用 $\mathbf{h}_{\hat{k}}$ (hidden state) 和 $\mathbf{l}_{\hat{k}}$ (logits) 双层蒸馏：

$$
\mathcal{L}_h = \frac{1}{N_t} \sum_{i=1}^{N_t} \|\mathbf{h}_{stu}^{(i)} - \mathbf{h}_{\hat{k}}^{(i)}\|_2^2 \tag{10a}
$$

$$
\mathcal{L}_l = \text{KL}\big(\text{softmax}(\mathbf{l}_{\hat{k}}/\tau_t) \,\|\, \text{softmax}(\mathbf{l}_{stu}/\tau_t)\big) \tag{10b}
$$

- $N_t$ 是 trajectory token 数（T=6 在 nuScenes 上）
- $\tau_t = 5$ 是温度，让 softmax 分布更软，传递更多 "dark knowledge"
- $\mathcal{L}_h$ 是 hidden state 层面的 MSE，让学生 internalize teacher 的 reasoning process
- $\mathcal{L}_l$ 是 logits 层面的 KL，让学生匹配 teacher 的输出分布

**Intuition**：两层蒸馏 (hidden + logit) 类似 TinyBERT 的 "feature-level + logit-level" 双蒸馏。Hidden state 蒸馏传递的是"中间推理"，logit 蒸馏传递的是"最终决策分布"。这种做法让 student 不只学输出，还学思考过程。

reference:
- TinyBERT: https://arxiv.org/abs/1909.10351
- Hint-based distillation (FitNets): https://arxiv.org/abs/1412.6550

---

## 5. Overall training loss

总 loss 是各项加权和：

$$
\mathcal{L}_{all} = \mathcal{L} + \lambda_a \cdot \mathcal{L}_a + \lambda_h \cdot \mathcal{L}_h + \lambda_l \cdot \mathcal{L}_l \tag{11}
$$

权重设置：
- $\lambda_a = 0.05$ — anchor visual distillation 权重很小，说明主要是 regularization 作用
- $\lambda_h = 0.1$ — hidden state distillation 中等权重
- $\lambda_l = 0.2$ — logit distillation 权重最大，是主要的 trajectory supervision

**Intuition**：权重的相对大小反映了作者的"信念"——logit 蒸馏最重要（直接传递输出知识），hidden state 次之（中间表示辅助），anchor 最弱（仅作保 representation 不漂移）。如果 $\lambda_a$ 太大，student 会变成 teacher 的复制，丢失 task adaptation 能力；太小则 anchor 失效。$\lambda_l / \lambda_h = 2$ 表明作者更信"行为对齐"而非"特征对齐"。

---

## 6. 架构总览

整体架构 (Figure 2) 分三块：

```
[Left: Self-Anchored Visual Distillation]
  Multi-view images → student ViT → z_v^stu
                     → self-anchor teacher ViT (frozen) → z_v^tea
                                       ↓
  Prompt + Ego state + GT trajectory → tokens
                                       ↓
  AnchorFormer (AnchorLayer + AnchorScorer) → W_a (token weights)
                                       ↓
  L_a = MSE(z_v^tea, z_v^stu) weighted by W_a

[Right: Oracle-Guided Trajectory Distillation]
  Current images + FUTURE images + FUTURE ego status
            ↓
  Oracle teacher (Qwen2.5-VL 3B, frozen)
            ↓
  Coarse trajectory W_t^c → Fine trajectory W_t^f (refine)
            ↓
  MC-Dropout sampling × 10 → candidate set S_h, S_l
            ↓
  Pick best k by CE with GT → h_k, l_k
            ↓
  L_h = MSE(h_stu, h_k),  L_l = KL(l_k || l_stu)

[Center: Collaborative Distillation]
  Student model (Qwen2.5-VL 3B, trainable)
    - visual encoder (trainable, but anchored)
    - LLM decoder (trainable)
  Total loss = L + 0.05 L_a + 0.1 L_h + 0.2 L_l
```

注意几个细节：
1. **Student 和 oracle teacher 架构完全一样** (都是 Qwen2.5-VL 3B)，区别只在输入（teacher 多吃 future info）和是否 freeze
2. **Self-anchor teacher 是 student visual encoder 的 pre-SFT snapshot**，本质上是同一个 encoder 的"前世"
3. **AnchorFormer 跟 student 一起训练**，是唯一额外的可训练模块

---

## 7. 实验设置

### 7.1 Implementation details
- Student & Oracle teacher: Qwen2.5-VL 3B (共享架构)
- Self-anchor teacher: student visual encoder 的 SFT 前拷贝
- AnchorLayer: 用 Qwen2.5-VL 3B 最后一层 LLM layer 的权重初始化
- 训练时只有 student 和 AnchorFormer 可训练，两个 teacher 全部 frozen
- GitHub: https://github.com/heycjj/EvoDriveVLA

### 7.2 Datasets
**Open-loop**: nuScenes (https://www.nuscenes.org/)
- 1000 scenes, 每个 ~20s
- 6 cameras
- Metric: L2 error at 1s/2s/3s + avg collision rate
- 两种 split：ST-P3 split 和 UniAD split

**Closed-loop**: NAVSIM (https://github.com/autonomousvision/navsim)
- navtrain: 1192 scenes
- navtest: 136 scenes
- Metric: PDM-Score (PDMS)，包含 NC (No Collision), DAC (Drivable Area Compliance), TTC (Time to Collision), Comfort, Ego Progress 五个子指标
- 预测 horizon: 4s

---

## 8. Open-loop 结果分析 (Table 1)

在 nuScenes 上对比三大类 baseline：traditional / LLM-based / distillation-based。

**ST-P3 split 上的对比**：
| Method | L2 1s | L2 2s | L2 3s | Avg | Col 1s | Col 2s | Col 3s | Avg |
|---|---|---|---|---|---|---|---|---|
| ST-P3 | 1.44 | 2.11 | 2.90 | 2.15 | 0.23 | 0.62 | 1.27 | 0.71 |
| VAD | 0.17 | 0.34 | 0.60 | 0.37 | 0.04 | 0.27 | 0.67 | 0.33 |
| DiffusionDrive | 0.27 | 0.54 | 0.90 | 0.57 | 0.03 | 0.05 | 0.16 | 0.08 |
| UniAD | 0.44 | 0.67 | 0.96 | 0.69 | 0.04 | 0.08 | 0.23 | 0.12 |
| DriveVLM | 0.18 | 0.34 | 0.68 | 0.40 | 0.10 | 0.45 | - | 0.27 |
| OmniDrive | 0.17 | 0.31 | 0.55 | 0.34 | 0.05 | 0.25 | 0.80 | 0.37 |
| OpenDriveVLA | 0.14 | 0.30 | 0.55 | 0.33 | 0.02 | 0.19 | 0.67 | 0.32 |
| DistillDrive | 0.28 | 0.54 | 0.83 | 0.55 | 0.00 | 0.17 | - | 0.06 |
| DiMA | 0.12 | 0.25 | 0.44 | 0.27 | 0.04 | 0.06 | 0.15 | 0.08 |
| **EvoDriveVLA** | **0.12** | **0.24** | **0.43** | **0.26** | **0.02** | **0.12** | - | **0.07** |

**关键观察**：
1. EvoDriveVLA 在 L2 上和 DiMA 持平 (0.12 @ 1s) 或略好 (Avg 0.26 vs 0.27)
2. Collision rate 显著优于 OpenDriveVLA (0.07 vs 0.32)，相对降幅 ~78%
3. 与 DistillDrive 相比，L2 降 ~54% (0.26 vs 0.55)，collision 持平 (0.07 vs 0.06)
4. 比 DriveVLM、OmniDrive 这类纯 LLM-based 方法显著好

**UniAD split 上的对比**：EvoDriveVLA 在 L2 上 0.52，比 DiMA (0.57) 降 ~9%；collision 0.12 比 DiMA (0.07) 略高，但仍是 SOTA 水平。

**Intuition**：open-loop 上 collision 的下降特别明显，说明 oracle teacher + coarse-to-fine 在长 horizon (3s) 上的稳定性收益最大——这正是 L2 误差累积导致 collision 的关键时段。

---

## 9. Closed-loop 结果 (Table 2, NAVSIM navtest)

| Method | NC↑ | DAC↑ | TTC↑ | Comf.↑ | EP↑ | PDMS↑ |
|---|---|---|---|---|---|---|
| Constant Velocity | 68.0 | 57.8 | 93.0 | 77.3 | 50.0 | 20.6 |
| Ego Status MLP | 65.6 | 100 | 62.8 | 100 | 83.6 | 65.6 |
| VADv2 | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| UniAD | 97.7 | 92.8 | 92.9 | 100 | 78.8 | 83.4 |
| TransFuser | 97.8 | 91.9 | 92.8 | 100 | 79.2 | 84.0 |
| PARA-Drive | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| Qwen2.5-VL 3B (baseline) | 97.3 | 90.4 | 92.9 | 99.6 | 77.6 | 81.9 |
| Qwen2.5-VL 8B | 97.8 | 92.1 | 92.8 | 100 | 78.3 | 83.3 |
| InternVL3-8B | 97.0 | 92.4 | 91.8 | 100 | 78.9 | 83.3 |
| **EvoDriveVLA** | **98.0** | **93.3** | **93.1** | **100** | **81.1** | **85.3** |

**关键发现**：
1. EvoDriveVLA (3B) 比 Qwen2.5-VL 3B baseline PDMS 高 3.4 分 (4.2% 相对提升)
2. **3B 蒸馏版甚至超过 8B 和 InternVL3-8B**——这是 paper 最亮眼的结果，证明 distillation 比 scaling up 更高效
3. Ego Progress (EP) 提升明显 (77.6 → 81.1)，说明蒸馏让学生"开得更进取"而不只是"更安全"
4. Comfort 100 分，舒适性没有牺牲

**Intuition**：closed-loop 比 open-loop 更能反映真实驾驶能力，因为 closed-loop 里 agent 的 action 会影响后续 state，错误会复合。EvoDriveVLA 在 closed-loop 上的提升说明 oracle teacher 的 future-aware reasoning 被有效 transfer 到 student，让 student 在没有 future info 的情况下也能做出"预判式"决策。

---

## 10. Ablation study (Table 3, UniAD metric on nuScenes)

| Traj KD | Traj Refine | MC-Dropout | Visual KD | L2 1s | 2s | 3s | Avg |
|---|---|---|---|---|---|---|---|
| ✗ | ✗ | ✗ | ✗ | 0.17 | 0.47 | 1.02 | 0.55 |
| ✓ | ✗ | ✗ | ✗ | 0.17 | 0.46 | 1.00 | 0.54 |
| ✓ | ✓ | ✗ | ✗ | 0.16 | 0.46 | 0.99 | 0.53 |
| ✓ | ✓ | ✓ | ✗ | 0.16 | 0.45 | 0.98 | 0.53 |
| ✓ | ✓ | ✓ | ✓ | **0.16** | **0.44** | **0.96** | **0.52** |

**观察**：
1. Trajectory KD 单独加入几乎无提升 (0.55 → 0.54)，说明只蒸馏单一轨迹收益有限
2. + Traj Refine 略有提升 (0.54 → 0.53)，coarse-to-fine 有效
3. + MC-Dropout 略有提升 (0.53 → 0.53)，几乎持平但稍好
4. + Visual KD 进一步提升到 0.52

**重要直觉**：每个 component 单独贡献都不大，但**累积效应显著**。这暗示这是一个 **system-level design**——各模块协同才能发挥效果。单一组件"性价比"看起来低，但整体系统是 SOTA。这种类型的 method 通常 reproducibility 会有挑战，因为去掉任何一个 component 都会失去大部分收益。

### Oracle teacher 自身性能 (Table 4)

| Split | L2 1s | 2s | 3s | Avg | Col 1s | 2s | 3s | Avg |
|---|---|---|---|---|---|---|---|---|
| ST-P3 | 0.10 | 0.14 | 0.18 | 0.14 | 0.02 | 0.03 | 0.05 | 0.04 |
| UniAD | 0.13 | 0.20 | 0.27 | 0.20 | 0.02 | 0.05 | 0.05 | 0.04 |

Oracle teacher 自己的 L2 = 0.14 (ST-P3) / 0.20 (UniAD)，远好于所有 published 方法。这说明 future info 确实给了 teacher 巨大优势——这也是它能教好学生的根本原因。

---

## 11. 关键的可视化分析

### 11.1 Coarse-to-Fine refinement 的效果 (Figure 3)
作者用 KDE (Kernel Density Estimation) 画了 refine 前后 trajectory loss 分布：
- Refine 前：loss 分布偏右，长尾明显
- Refine 后：分布显著左移，零附近密度大增，长尾被压缩

**Intuition**：refinement 主要砍掉了"明显错误"的预测（长尾），但对已经很好的预测改变不大。这符合 coarse-to-fine 的语义——粗预测给方向，fine 预测做局部修正。

### 11.2 MC-Dropout sampling 的效果 (Figure 4)
- 近零区域 loss 减少 ~50%
- ~30% 的 teacher 轨迹 L2 loss < 0.1

**Intuition**：MC-Dropout 的作用是"采样找最好"。10 个 dropout 样本里有几个会"幸运地"给出非常准确的轨迹，挑这些做 soft target，蒸馏效果自然好。这其实是一种 test-time augmentation (TTA) 思想，但用在 distillation 上。

### 11.3 定性对比 (Figure 5)
对比 VAD, OmniDrive, EvoDriveVLA 在不同场景（sunny/overcast，straight/curved）的 long-horizon 预测：
- VAD: longitudinal 预测过短（保守）
- OmniDrive: lateral 偏差（容易偏车道）
- EvoDriveVLA: long-horizon 上轨迹更贴近 ground truth

---

## 12. Prompt 工程细节（Appendix A）

从附录的 prompt 例子可以看到设计：

**Student prompt** 包含：
- 6 个 view 的 camera image
- Historical trajectory (last 2 seconds)
- Historical ego status (velocity, accel_x, accel_y, steer) over last 2s
- 指令："please output the plan waypoints (0.5s intervals) for the next 3 seconds"

**Oracle teacher prompt** 额外包含：
- Future images (Future 0s/1s/2s/3s，每秒一张)
- Future ego status over next 3 seconds
- 同样的输出格式要求

**Refine-teacher prompt** 在 oracle teacher 基础上多加：
- "I can provide the trajectory predictions from the student model here, and you can also refine them based on this: [...]"
- 把 coarse 轨迹作为 hint 输入

**Intuition**：这是一个非常 prompt-engineering-driven 的设计。teacher 看到的 future info 是显式的 image + ego state，而不是隐式的 feature。这让 teacher 可以"看到"未来 1 秒前方的车、行人、转弯，所以 trajectory 预测自然准。Refine 阶段把 coarse 当 hint，相当于让 LLM 做 "given this guess, refine it" 的任务，对 LLM 来说是一个 well-defined 任务格式。

---

## 13. 我的一些 critical thoughts

### 13.1 优势
1. **Privileged teacher 是真正合理的 distillation 范式**——之前的方法 teacher 和 student 同输入，本质上没有信息增益。Oracle teacher 用 future info 做到 "teacher >> student"，distillation 才有意义。
2. **Token-level anchor distillation** 比传统 sample-level 蒸馏更精细，trajectory-guided 权重对 driving 任务来说语义合理。
3. **3B 打 8B 的结果**非常有说服力，说明 distillation 在 driving 上是 efficient 的 scaling 替代方案。

### 13.2 潜在问题
1. **Oracle teacher 的 future images 从哪儿来？** Paper 没说清楚是 nuScenes 里的 ground-truth future frames (offline) 还是预测出来的。从 prompt 看像是用 nuScenes 的 future frame（GT），这意味着 teacher 不能 deploy 到 online 场景——teacher 只能 offline 训练时用。这其实没问题，但 paper 没明确说，读者会困惑。
2. **Ablation 里每个 component 收益都很小**，0.55 → 0.52 的总收益在 L2 metric 上其实不太显著。Collision 指标的下降才是真正亮点，但 ablation 没拆 collision。
3. **MC-Dropout 采样数 N=10** 看起来 heuristic，论文没做 sensitivity analysis。N 越大肯定越好但成本越高，trade-off 在哪？
4. **Coarse-to-fine 是 prompt-level 而不是 architecture-level**——refinement 靠 prompt 喊话 "refine this trajectory"，这依赖 LLM instruction following 能力。如果 LLM 不够强（比如 1B 模型），这个 refine 可能失效。
5. **Self-anchor teacher 是 SFT 前的 snapshot**，但 student 训完之后，anchor teacher 还能不能继续用？后续 continual learning 怎么处理？

### 13.3 与相关工作对比

| 维度 | DiMA | DistillDrive | EvoDriveVLA |
|---|---|---|---|
| Teacher 输入 | = student | = student | + future info |
| Trajectory 多样性 | 单轨迹 | vocabulary-based | MC-Dropout |
| Visual encoder 处理 | 没特殊处理 | 没特殊处理 | self-anchor distillation |
| Refinement | 无 | 无 | coarse-to-fine |
| 蒸馏层级 | logit | logit | hidden + logit |

EvoDriveVLA 真正的创新是**把 visual encoder degradation 和 trajectory precision 两个问题分开处理但协同优化**——visual 用 anchor，trajectory 用 oracle，两者在 loss 里加权和。

### 13.4 联想到的相关方向

1. **World models for driving**：Oracle teacher 用 future images，本质是"使用 world model 的输出做 planning teacher"。如果未来有更强的 driving world model (GAIA-1, OccWorld, Drive-WM)，可以直接替换 oracle teacher，效果可能更好。
   - GAIA-1: https://arxiv.org/abs/2309.17080
   - OccWorld: https://arxiv.org/abs/2404.01825

2. **Privileged reinforcement learning**：这其实是 privileged RL 的 supervised 版本。RL 版本是 teacher 用 simulator state，student 用 observation。这里 teacher 用 future GT，student 用 current obs。思路同源。
   - Learning by Cheating: https://arxiv.org/abs/1912.12294

3. **Self-distillation & EMA teacher**：BYOL, DINO, SimSiam 这套 self-supervised 蒸馏的方法这里被简化——没用 EMA 更新，而是直接 frozen pre-SFT snapshot。简单但可能不够 robust，如果训练 step 数很多，student 漂移太远 anchor 可能反而拖后腿。
   - BYOL: https://arxiv.org/abs/2006.07733
   - DINO: https://arxiv.org/abs/2104.14294

4. **Diffusion policy distillation**：DiffusionDrive 用 diffusion 生成多模态轨迹，EvoDriveVLA 用 MC-Dropout 采样多模态轨迹。前者是 generative diversity，后者是 model uncertainty diversity。两者其实可以结合。
   - DiffusionDrive: https://arxiv.org/abs/2411.15239
   - Diffusion Policy: https://arxiv.org/abs/2303.04137

5. **VLM visual encoder fine-tuning 的 degradation 问题**：这是一个 VLM 社区普遍问题。Cambrian-1, Eagle, Prismatic VLMs 都讨论过。EvoDriveVLA 给出了一个 task-specific 的解决方案（trajectory-guided anchor），但通用 VLM 上是否有效需要进一步验证。
   - Cambrian-1: https://arxiv.org/abs/2406.16860
   - Eagle: https://arxiv.org/abs/2408.15998

---

## 14. 总结

EvoDriveVLA 的核心 thesis 是：**autonomous driving VLA 的 distillation 应该是 collaborative 的 perception-planning 联合优化，而不是单纯 trajectory KD**。

具体来说：
1. **Perception 端**：用 self-anchor teacher 做 token-level anchor distillation，让 visual encoder 在 adapt task 的同时保留 pre-training representation。
2. **Planning 端**：用 oracle teacher (吃 future info) 做 coarse-to-fine + MC-Dropout 的多候选蒸馏，让 student 学到"未来感知的 reasoning"。
3. **Loss 整合**：$\mathcal{L}_{all} = \mathcal{L} + 0.05\mathcal{L}_a + 0.1\mathcal{L}_h + 0.2\mathcal{L}_l$，权重设计反映了对各 component 作用的判断。

实验上：
- nuScenes open-loop SOTA (L2 0.26, collision 0.07 on ST-P3 split)
- NAVSIM closed-loop 3B 模型打 8B 模型 (PDMS 85.3 vs 83.3)
- Ablation 显示各 component 单独贡献小，协同效应显著

对我来说，这篇 paper 最大的启示是：**distillation 的设计要遵循"teacher 必须比 student 强"这个基本原则**，而现有 driving distillation 工作很多都违反了这一点。Oracle teacher with privileged information 是一个非常 clean 的解决方案，可以推广到任何有"未来信息"的 offline training 场景。

进一步延展：如果把 oracle teacher 换成更强的 world model (生成 future 而非用 GT future)，整个 pipeline 可以变成 fully online，这是一个值得探索的方向。同时 self-anchor 这个 idea 也可以反过来用——在 SFT 过程中动态更新 anchor teacher (EMA 风格)，可能比 frozen snapshot 更 robust。

相关 link 集合：
- EvoDriveVLA GitHub: https://github.com/heycjj/EvoDriveVLA
- nuScenes: https://www.nuscenes.org/
- NAVSIM: https://github.com/autonomousvision/navsim
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- UniAD: https://github.com/OpenDriveLab/UniAD
- VAD: https://github.com/hustvl/VAD
- DiMA: https://arxiv.org/abs/2503.04218
- DistillDrive: https://arxiv.org/abs/2507.07291
- Hinton KD: https://arxiv.org/abs/1503.02531
- MC-Dropout: https://arxiv.org/abs/1506.02142
- TinyBERT: https://arxiv.org/abs/1909.10351
- Learning by Cheating: https://arxiv.org/abs/1912.12294
- DINO: https://arxiv.org/abs/2104.14294
- BYOL: https://arxiv.org/abs/2006.07733
- DiffusionDrive: https://arxiv.org/abs/2411.15239
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- Cambrian-1: https://arxiv.org/abs/2406.16860
- Eagle VLM: https://arxiv.org/abs/2408.15998
- OccWorld: https://arxiv.org/abs/2404.01825
- DriveVLM: https://arxiv.org/abs/2402.12289
- OmniDrive: https://github.com/wzjwmm/OmniDrive
- OpenDriveVLA: https://arxiv.org/abs/2503.23463

希望这个详细的拆解对你 build intuition 有帮助！这个 paper 的 privileged teacher + token-level anchor 这套组合拳，其实是把 RL 里的 privileged learning、self-supervised 里的 EMA teacher、NLP 里的 hidden+logit 双蒸馏全部 recombine 到 driving VLA 上，是非常 elegant 的工程化设计。
