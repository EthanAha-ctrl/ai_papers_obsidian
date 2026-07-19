---
source_pdf: ABot-PhysWorld Interactive World Foundation Model.pdf
paper_sha256: f2d2a6d86f2c465638ece0d085e11933ae6c26fedb83735c8266fced83068273
processed_at: '2026-07-17T22:59:10-07:00'
target_folder: World-model/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# ABot-PhysWorld 深度拆解：让 World Model 真正"懂"物理

Andrej，这篇 2026 年 3 月的 paper 我读下来，核心 takeaway 可以浓缩成一句话：**MLE 学不会物理，但 DPO + decoupled discriminator + action map 可以把物理"贴"回去**。下面我把每个模块的机制、公式变量、以及背后的直觉都摊开讲，顺便把我自己脑子里蹦出来的联想和质疑也放进来。

---

## 1. 问题为什么根本：MLE 的物理盲区

Sora v2 Pro 和 Veo 3.1 这类 SOTA 视频模型在 robotic manipulation 上经常翻三种车：
- **Object penetration**（夹爪穿过苹果）
- **Contactless / anti-gravity motion**（物体被"磁吸"起来）
- **Unnatural deformation**（接触瞬间几何崩坏）

作者归因到两点，我认为这点诊断很准：
1. **训练数据缺 embodied interaction signal**——generic web video 里摩擦、碰撞响应、质量分布这些细粒度 dynamics 信号稀薄；
2. **MLE 目标把 pixel error 当一回事**——一个"夹爪穿过苹果"的 frame，和"夹爪停在苹果表面"的 frame，在 L2 / flow matching loss 上差距可能就几个 pixel。likelihood 没法区分 *physically valid transition* 和 *physically invalid transition*。

直觉上：MLE 在做"匹配训练分布"，但训练分布本身就有物理违和的 sample（来自数据噪声、sensor 同步问题、或者只是渲染瑕疵），且 MLE 没有"哪个更好"的序。DPO 正好补这一刀。

参考：
- Diffusion-DPO 原论文: https://arxiv.org/abs/2311.18808
- DPO 原论文: https://arxiv.org/abs/2305.18290
- PhyDPO（相关的 physics-DPO 工作）: https://arxiv.org/abs/2512.24551

---

## 2. Data Curation：embodied 数据不是 web video

### 2.1 Filtering 四道闸
3M clips，5 个数据源聚合：**AgiBot**（https://arxiv.org/abs/2503.06669）、**RoboCoin**（https://arxiv.org/abs/2511.17441）、**RoboMind**（https://arxiv.org/abs/2412.13877）、**Galaxea**（https://arxiv.org/abs/2509.00576）、**OXE**（https://robotics-transformer-x.github.io/）。

每道闸都有具体的工程含义：

| Stage | 操作 | 直觉 |
|---|---|---|
| Video quality gate | 80–500 frames, static camera, 异常 resolution 丢 | 机械臂视频是固定背景，scene-cut detector（Cosmos-Curate、VideoX-Fun）会误切 |
| Optical flow (Farnebäck) | 2 FPS 灰度 → dense flow → 极坐标幅值平均 → kinematic score | 滤掉静止黑屏与"鬼影抖动" |
| CLIP temporal coherence | 8 等距帧, 768D feature, 相邻 cosine similarity 阈值 | 抓黑屏 / cut / stitch artifact |
| Vision-action alignment | 校准 action map 投到 frame 上，Qwen3-VL 判时空对齐 | sensor calibration / 时间戳错位的 clip 必须丢，否则 DPO 学到的"物理"会被噪声污染 |

直觉上这一段给我两点启发：
- (a) **Cosmos-Curate（https://github.com/nvidia-cosmos/cosmos-curate）的设计是为 web video**——scene-cut 在 robotic 静态背景上完全失效。说明你不能拿通用 curation infra 直接套 embodied。
- (b) **Vision-action alignment 这一步非常关键**——后面 action-conditioned generation 的前提是 action 和 frame 严格对齐，这里筛不干净，A2V 训练就是垃圾进垃圾出。

### 2.2 Hierarchical Distribution Balancing
四级采样，直觉是"防止 head task 把模型拽进 memorization"：

- **Level 1（intra-dataset）**：OXE 内部子集原样保留；
- **Level 2（cross-robot）**：upweight 罕见 embodiment（dual-arm、非标 kinematics）；
- **Level 3（task-aware quota）**：
  - Head tasks → cap **8–15%**（防 over-fit dominant category）
  - Body tasks → **40–50%** uniform
  - Long-tail tasks → **100% 保留**
- **Level 4（macro-dataset scale）**：大集 uniform subsample，小集 minimum coverage；**3-round supplementation**：（1）base quota →（2）reallocate unused →（3）random fallback。

这个 task-aware quota 设计我特别欣赏——它直接对应了 *"How Far is Video Generation from World Model?"*（https://arxiv.org/abs/2411.02385）的结论：scaling 重复数据只会 memorization。把 head 压到 15% 是非常激进的下采样，但正是这种"刻意打散分布"才能让 long-tail 出现。

### 2.3 Physics-Aware Captioning
这是整个 data pipeline 最有意思的部分。caption 分四层：

1. **Macroscopic**：task intent（natural language）
2. **Mesoscopic**：verb-noun action segmentation（long-horizon planning）
3. **Microscopic**：Cartesian trajectory / relative motion / gripper state
4. **Scene-level**：contact / support / containment / outcome

并且作者明确做三件事压 hallucination：
- **Few-shot in-context with 正例+负例**（给"穿透"和"接触"两种描述，让模型选）
- **Dynamic vocabulary for grasp type**（pinch / wrap / lateral 等精确 grasp 语义）
- **Visible-fact baseline**（只描述可观测证据，杜绝"imagined spatial relations"）

caption 结构是 **four-stage narrative**：scene construction → action flow → final state → camera summary。

我读完这段的直觉是：这种 caption 已经在做"因果物理建模"——gravity-induced dropping、surface deformation、force feedback 这些被显式写进 caption。等于把物理 annotation 当成一种 *weak supervision*，让 SFT 阶段就先吃进物理先验，DPO 阶段再 sharpen。

perception 用 **Qwen3-VL 32B** 抽 structured physical attribute，writing 用 **Qwen3 32B FP8** 生成四阶段 caption。FP8 量化是为了 throughput——3M clips 全跑 caption 必须量化。

---

## 3. Method：从 SFT 到 Physics-Aligned DPO 再到 Action Control

### 3.1 Backbone
**Wan2.1-I2V-14B-480P**（https://github.com/Wan-Video/Wan2.1），full fine-tune。81 frame @ 480×832。

### 3.2 Physical Preference Alignment——这是 paper 的灵魂

#### 3.2.1 Decoupled VLM Discriminator
直觉上最大的工程亮点：**生成问题和回答问题的不能是同一个 VLM**，否则就是 self-evaluation hallucination（自己提问自己答，会 sycophant，会 reward hack）。

- **Proposer = Qwen3-VL 32B Thinking**：看 first frame + instruction → 动态生成 task-specific physical checklist。
  - **Tier 1 metric**：fatal violation（penetration / anti-gravity）→ **single-vote veto**（一票否决）
  - **Tier 2 metric**：micro-physical fidelity、contact dynamics → 做 fine differentiation
  - 强制 **positive + negative 混合**问题（例如"夹爪是否穿苹果"配"夹爪是否停在苹果表面"），防止 scorer 走捷径全答 "no violation"。

- **Scorer = Gemini 3 Pro**（https://deepmind.google/technologies/gemini/）：CoT 推理，global scan → mark suspicious frames → backtrack confirm。

- **Tournament sampling**：N 个 candidate 里要选最优 $y_w$ 和最差 $y_l$。全排列 $\mathcal{O}(N^2)$ 太贵。他们用 **knockout tournament 选最优** + **loser-bracket 选最差**，复杂度 $\mathcal{O}(N)$，构造出 DPO triplet $(x, y_w, y_l)$ 且 margin 清晰。

这里我自己的联想：tournament sampling 这个思路其实在 LLM RLHF 里也应该用。通常 RLHF reward model 给一对 pair 打分，但视频生成里 N 个 candidate 是同一 prompt 蒸出来的，做 round-robin pairwise 反而浪费——锦标赛 + 败者组这种结构能 $O(N)$ 提取极值对，工程上很漂亮。

#### 3.2.2 Diffusion-DPO 公式拆解

公式（1）：
$$
\mathcal{L}_{DPO} = -\mathbb{E}_{z,\epsilon,t}\Bigg[\log\sigma\Bigg(-\frac{\beta}{2}\Big[\underbrace{(L_\theta(z_w) - L_\theta(z_l))}_{\text{Policy Diff.}} - \underbrace{(L_{ref}(z_w) - L_{ref}(z_l))}_{\text{Ref. Diff.}}\Big]\Bigg)\Bigg]
$$

逐项解释：

- $z$：video latent（VAE 编码后的潜变量）
- $\epsilon \sim \mathcal{N}(0, I)$：标准 Gaussian noise
- $t \sim \mathcal{U}(0, T)$：diffusion timestep，从 $[0, T]$ 均匀采样
- $z_t$：latent 加噪后的状态
- $L(\theta, z) = \|\epsilon_\theta(z_t, t, c) - \epsilon\|_2^2$：单步 denoising MSE，$c$ 是 condition（prompt + initial frame）
- $L_\theta(\cdot)$：policy model $\pi_\theta$ 的 denoising error
- $L_{ref}(\cdot)$：reference model $\pi_{ref}$（SFT baseline）的 denoising error
- $z_w, z_l$：physics-compliant video / physics-violating video 的 latent
- $\beta$：distribution divergence 控制参数——**paper 里设 $\beta = 5000$**，非常激进
- $\sigma(\cdot)$：sigmoid

直觉解读：
- 内层 $\big(L_\theta(z_w) - L_\theta(z_l)\big) - \big(L_{ref}(z_w) - L_{ref}(z_l)\big)$ 的含义是 **policy 相对 reference，在 $w$ vs $l$ 上的 preference gap 的变化**。
- 当这个 gap 变正（policy 在 $z_w$ 上比 ref 更"喜欢"，在 $z_l$ 上比 ref 更"不喜欢"），sigmoid 输入变大，loss 减小。
- $\beta = 5000$ 这么大，是因为 diffusion 的 $L$ 本身量级在 MSE 级别（很小），不放大 preference signal 会被噪声淹没。这跟 LLM DPO 里 $\beta$ 通常 0.1 量级完全是两个世界。

**LoRA 工程技巧**（这部分是 14B DiT 跑 DPO 不 OOM 的关键）：
- Freeze DiT backbone
- 插 **rank-64 LoRA** 到 self-attention 的 $q, k, v, o$ 和 FFN 的 $fn.0, fn.2$
- 算 $L_{ref}$ 时**临时禁用 LoRA**，于是同一个参数集既是 policy 又是 ref，**零额外显存**

这个 trick 我觉得值得单独拎出来——在 LLM DPO 里通常 maintain 两个完整 model（policy + frozen reference），显存翻倍。LoRA + disable 这套思路把 ref cost 压成 0，对 14B 级别 diffusion model 几乎是 must-have。参考 LoRA 原文：https://arxiv.org/abs/2106.09685

### 3.3 Action-Conditioned Generation

#### 3.3.1 Action Map Construction
输入 action $\mathbf{a} \in \mathbb{R}^7$：
- 3D position $(x, y, z)$
- 3D orientation（quaternion 或 rotation matrix 的三主轴）
- 1D gripper openness

双臂扩展到 $\mathbb{R}^{14}$。

转换流程：
1. 用 camera intrinsics/extrinsics 把 $(x,y,z)$ 投影到 image plane $(u, v)$
2. Orientation 三个主轴 → 投影到 image plane → 渲染成 **colored arrows**，**arrow length 编码 depth**
3. Gripper state → 在 $(u, v)$ 处画 **circular mask**，**opacity 线性映射 openness**
4. 双臂用 **R / B 通道区分**，输出 multi-channel action map

直觉：把低维 7D command "渲染"成图像空间的结构化 tensor，**让 action 和 visual 在同一表征空间**。这有几个好处：
- cross-embodiment 泛化好——不同机器人 kinematics 不同，但都投影到同一图像坐标系
- 不破坏 backbone 的 pre-trained 视觉先验（因为 action 看起来像"图像 patch"）
- arrow + circle 这种几何图元比 raw vector 更"可读"

#### 3.3.2 Action Injection: Parallel Context Block (VACE-style)
公式（2）：
$$
\mathbf{x}_i = \mathrm{DiT}_i(\mathbf{x}_{i-1}) + \alpha \cdot W_{\mathrm{zero}}^{(i)} \mathbf{h}_i
$$

变量：
- $\mathbf{x}_{i-1}, \mathbf{x}_i$：第 $i$ 个主 DiT block 的输入输出
- $\mathrm{DiT}_i(\cdot)$：第 $i$ 个主 DiT block（frozen）
- $\mathbf{h}_i$：第 $i$ 个 **context block** 的输出——这些 block 是从主 DiT clone 出来的，参数独立训练
- $W_{\mathrm{zero}}^{(i)}$：**zero-initialized 1×1 conv**，把 context block 输出投影到主 DiT 维度
- $\alpha$：control scale（推理时可调，类似 ControlNet 的 conditioning scale）

关键设计点：
- **Selective duplication**：只复制 layers 0, 5, 10, 15, 20, 25, 30, 35（每 5 层取 1 层），节省参数
- **Zero-init**：训练开始时 $W_{\mathrm{zero}}$ 输出全 0，context branch 对主 backbone **零扰动**——pre-trained physical priors 完全保留，action control 能力**逐步学到**
- 不用 AdaLN（cross-embodiment 泛化差）也不直接 concatenate（catastrophic forgetting）

VACE 原文：https://arxiv.org/abs/2503.18812（ICCV 2025）

我的直觉：这个架构跟 ControlNet（https://arxiv.org/abs/2302.05543）精神上很像——都是"克隆分支 + zero-init 投影 + residual 融合"。区别在于 ControlNet 克隆整个 encoder，VACE 这里只克隆每 5 层中的 1 层，密度更低，参数更省，更适合 14B DiT。

---

## 4. EZSbench：第一个 training-independent embodied zero-shot benchmark

### 4.1 Evaluation Set 构造
**双分支 initial observation pool**：
- **Branch 1（合成）**：Nano Banana（Google T2I）合成图，正交控制 robot / scene / task / perspective 四个变量。覆盖 morphological / scene / task generalization。
- **Branch 2（真实 + 编辑）**：VLM-guided **background editing** on real images，**保留 foreground physical interaction**——只换 background，foreground 物理不变。

**Dense description synthesis 三阶段**：
1. **Visual anchoring**：scene layout + object coordinates
2. **Action simulation**：kinematically compliant trajectory + micro-physical interaction
3. **Narrative synthesis**：documentary-style caption，整合 initial state + trajectory + final state

直觉：Branch 1 测纯 OOD（morphology + task 组合），Branch 2 测 background shift。这两支加起来覆盖了 "未见 robot × 未见 scene × 未见 task" 的组合空间，这是 in-distribution benchmark（像 PBench）测不到的。

### 4.2 Decoupled Dual-Model Evaluation
- **Question generator = Qwen3-VL-32B-Thinking**：System 2 reasoning，9 个 criteria across spatial / temporal / physical，**30–50% negative question 强制配比**（防止 shortcut learning——一直猜"没有 violation"）
- **Answerer = Qwen2.5-VL-72B-Instruct**：与 generator 不同模型，**消除 self-evaluation bias**

评分公式：
$$
S_v = \frac{1}{|Q_v|} \sum_{q \in Q_v} \mathbb{I}\big(\mathrm{VQA}(v, q) = \mathrm{GT}(q)\big)
$$

- $Q_v$：对视频 $v$ 生成的 checklist
- $\mathrm{VQA}(v, q)$：answerer 对问题 $q$ 给出的答案
- $\mathrm{GT}(q)$：checklist 的 ground truth
- $\mathbb{I}(\cdot)$：indicator function

直觉：这个协议把"提问"和"回答"分离，再加上强制 negative question 配比，理论上让 benchmark 不能被一个会"全 yes"的 VLM 蒙混过去。这套思路其实跟 self-rewarding LLM 的 critique 思路反着来——你要的是 *adversarial* 评估，不是 *cooperative* 评估。

---

## 5. Experiments——数字说话

### 5.1 实施细节
128× NVIDIA H20 GPUs。

三阶段训练：

| Stage | 配置 |
|---|---|
| TI2V (SFT) | 480×832, 81 frames, 6000 steps, global batch 128, lr 1e-5 |
| DPO | LoRA rank 64 (q,k,v,o,fn.0,fn.2), AdamW, lr 1e-6, 10-step warmup, **β=5000**, BF16, gradient checkpoint, per-device batch 1, 500 steps/epoch × 100 epochs |
| A2V (VACE) | duplicate layers [0,5,10,15,20,25,30,35], backbone frozen, batch 16, lr 5e-5, 20000 steps |

注意 DPO 阶段 **per-device batch 1** + 100 epochs，这意味着大量小步精修。β=5000 配合 LoRA 的小有效参数空间，确实需要更多 epoch 才能收敛。

### 5.2 PBench 结果（PAI-Bench robot subset, 174 videos, 886 questions）

| Model | Avg | Domain Score | Quality Score |
|---|---|---|---|
| Veo 3.1 | 0.8045 | 0.8350 | **0.7740** |
| Sora v2 Pro | 0.7652 | 0.7626 | 0.7679 |
| Wan 2.5 | 0.8096 | 0.8644 | 0.7548 |
| **Our Model (SFT only)** | 0.8232 | 0.8785 | 0.7678 |
| **Our Model + DPO** | **0.8491** | **0.9306** | 0.7676 |

直觉解读：
- **DPO 把 Domain Score 从 0.8785 → 0.9306**（+5.2 个点），同时 Quality Score 几乎不掉（0.7678 → 0.7676）。这说明 preference alignment 是**近乎 free lunch**——物理对齐了，视觉质量不退。
- Veo 3.1 和 Sora v2 Pro Quality 高但 Domain 低，验证了 paper 的核心论点：generic video model 学的是 *perception*，不是 *physics*。
- Wan 2.5 是个有趣 baseline——它和 ABot 都来自 Wan 家族，Quality 接近但 Domain 差 6.6 个点，说明 DPO + embodied data curation 这套增量是真有效。

参考 PAI-Bench: https://arxiv.org/abs/2512.01989

### 5.3 EZSbench 结果（OOD 评测）

| Model | Avg | Quality | Domain |
|---|---|---|---|
| WoW-wan 14B | 0.7780 | 0.7609 | 0.7951 |
| GigaWorld-0 | 0.7549 | 0.7272 | 0.7826 |
| Cosmos-Predict 2.5 | 0.7394 | 0.7089 | 0.7698 |
| UnifoLM-WMA-0 | 0.6294 | 0.7355 | 0.5232 |
| **Our Model** | **0.8030** | **0.7694** | **0.8366** |

OOD 条件下 domain 仍能保持 0.8366，说明物理对齐学到的不是 in-distribution shortcut，而是某种 generalizable 的物理 reasoning。这点很重要——如果只是 over-fit PBench 的 question pattern，EZSbench 上会崩。

### 5.4 Action-Conditioned 结果

| Model | PSNR | SSIM | Traj. Consis. |
|---|---|---|---|
| Enerverse-AC | 20.42 | 0.7542 | 0.8157 |
| Gen-Sim | 18.05 | 0.7413 | 0.6195 |
| **Ours** | **21.09** | **0.8126** | **0.8522** |

Trajectory Consistency 用 **nDTW** 度量——fine-tuned YOLO 检测 gripper，提取轨迹和 ground truth 比 normalized dynamic time warping。这个 metric 选得很对，因为 pixel-level PSNR 对"轨迹稍微偏一点但视觉 plausible"的 case 不敏感。

参考 Enerverse-AC: https://arxiv.org/abs/2505.09723
参考 Gen-Sim 系: https://arxiv.org/abs/2508.05635

---

## 6. 我的几个直觉与质疑

### 6.1 直觉层面
1. **DPO β=5000 的物理含义**：diffusion loss 量级（MSE on latent）很小，相比 LLM 的 logit difference，preference signal 必须 scale up 才能学到。这个值 5000 的设定大概率是 sweep 出来的——但也意味着训练对 β 极其敏感，reproduce 难度高。

2. **Decoupled discriminator + tournament 的工程价值**：把 RLHF 里那套 reward model + preference pair 的范式搬到 diffusion，但避开了"训练独立 reward model"的代价——直接用现成 VLM（Qwen3-VL + Gemini 3）当 reward。这是 *API-as-reward-model* 的实例化，工程上极轻量。

3. **Zero-init parallel context block**：这是 paper 里最 elegant 的设计。**zero-init 保证训练初时刻 backbone 完全等于 SFT model**，物理先验原封不动，然后 action 信号从零慢慢注入。这种"渐进唤醒"模式让 pre-training 的 physical knowledge 不会被 action 训练冲掉。

4. **Action map 设计**：7D → multi-channel 2D map 这一步，本质上是在做 **action 跨 embodiment 的统一表征**。所有机器人投影到同一 image plane，arrow 表方向、circle 表 gripper、color 表手臂。这相当于在 data level 做了 embodiment normalization，cross-embodiment 泛化因此提升是合理的。

### 6.2 质疑层面
1. **Self-evaluation 真的解决了吗？** Proposer (Qwen3-VL) 和 Scorer (Gemini 3 Pro) 确实不同 model，但它们可能共享相似的视觉盲点——比如对"夹爪稍微穿透"这种细粒度 violation 都不敏感。Decouple 缓解了 same-model bias，但没消除 *VLM-family-shared blind spot*。理想情况下应该引入一个 *physics-simulator-based verifier*（哪怕粗糙的 rigid body sim）做 ground truth 抽样校验。

2. **Tournament sampling 的 margin**：knockout + loser-bracket 给出 $y_w, y_l$，但 DPO 的有效性依赖 margin。如果 N 个 candidate 物理上都差不多（都 compliant 或都 violation），margin 太小，DPO 学不到东西。Paper 里没看到对 margin distribution 的统计。

3. **EZSbench 的 ground truth**：checklist 的 GT 是怎么定的？如果是 Qwen3-VL Thinking 生成的问题 + Qwen2.5-VL 回答 + 一致性打分，那么 GT 本身就是"两个 VLM 的共识"——不是真正的物理 GT。一个真正物理违规但两个 VLM 都没看出来的 case，会被错判为 compliant。这是 VLM-as-judge 的本质局限。

4. **Closed-loop 评估缺失**：作者在 conclusion 自己承认 "lacks closed-loop evaluation"。这意味着目前所有评估都是 *open-loop* —— 给 initial frame + action，generate video，然后静态打分。但 world model 真正的价值在 *closed-loop rollout* —— 用生成的 frame 喂回 VLA policy，看 policy 是否崩。这个缺口是接下来必须补的。

5. **Fixed-viewpoint 限制**：现在所有训练数据是固定相机视角。action map 的投影依赖 camera intrinsics/extrinsics 已知。如果相机移动或者多视角，这套 action map 构造会失效。这也限制了真实部署（机器人头戴相机走路时视角一直在变）。

---

## 7. 与相关工作的坐标定位

- **vs Cosmos / GigaWorld / UnifoLM-WMA**：这些都是 embodied world foundation model，但大多用 AdaLN 注入 action 或者直接 concat。ABot 用 parallel context block + zero-init，更优雅，灾难性遗忘风险低。
- **vs Veo 3.1 / Sora v2 Pro**：通用 video model，视觉质量顶级但物理崩。ABot 表明 *physics alignment 不需要牺牲 visual quality*——前提是有 embodied data + DPO。
- **vs PhyDPO**（https://arxiv.org/abs/2512.24551）：这是 ABot 的直接前序工作，ABot 把它扩展到 embodied + 加了 decoupled discriminator + tournament sampling。
- **vs VACE**（https://arxiv.org/abs/2503.18812）：VACE 提供了 selective duplication + zero-init residual 的 architectural primitive。ABot 把它用到 action injection 上。
- **vs World Action Models**（https://arxiv.org/abs/2602.15922）：WAM 把 world model 当 zero-shot policy 直接用。ABot 现在是 *predictor*，还没到 *actor*，但架构上已经具备 WAM 的潜质——把 action 输入换成 policy 输出即可。

---

## 8. 一句话总结我的 intuition

> **MLE 学到的是"分布匹配"，DPO + decoupled discriminator 加的是"物理偏好序"。把 7D action 渲染成图像空间 action map + zero-init parallel context block，相当于在 14B DiT 上做"无损 action 接口"。整套 pipeline 的精髓是：先用 data curation + SFT 把物理 prior 灌进 backbone，再用 DPO 把"物理违和"的 likelihood 压下去，最后用 zero-init context branch 把 action 控制能力"无痛"加上去——三步各管一摊，互不打架。**

参考链接汇总：
- Wan2.1: https://github.com/Wan-Video/Wan2.1
- VACE: https://arxiv.org/abs/2503.18812
- DPO: https://arxiv.org/abs/2305.18290
- Diffusion-DPO: https://arxiv.org/abs/2311.18808
- PhyDPO: https://arxiv.org/abs/2512.24551
- LoRA: https://arxiv.org/abs/2106.09685
- ControlNet: https://arxiv.org/abs/2302.05543
- AgiBot World: https://arxiv.org/abs/2503.06669
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- RoboMind: https://arxiv.org/abs/2412.13877
- Galaxea: https://arxiv.org/abs/2509.00576
- RoboCoin: https://arxiv.org/abs/2511.17441
- Cosmos-Curate: https://github.com/nvidia-cosmos/cosmos-curate
- PAI-Bench: https://arxiv.org/abs/2512.01989
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- World Action Models: https://arxiv.org/abs/2602.15922
- Enerverse-AC: https://arxiv.org/abs/2505.09723
- How Far is Video Generation from World Model: https://arxiv.org/abs/2411.02385
- ABot-PhysWorld Project Page: https://github.com/amap-cvlab/ABot-PhysWorld

如果你想我深入挖某一块（比如 DPO β 为什么是 5000 而不是 500、tournament sampling 的具体 bracket 构造、或者 action map 在双臂场景的 channel 分配细节），告诉我，我再展开。
