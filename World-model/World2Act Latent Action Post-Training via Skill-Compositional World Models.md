---
source_pdf: World2Act Latent Action Post-Training via Skill-Compositional World Models.pdf
paper_sha256: b189a7f6917a70e84308780760daa65884f2d33821a678b230d4a869c568160c
processed_at: '2026-08-13T05:32:12-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 World2Act

## 先讲个故事

假设你在训练一个 robot，让它学会"拿起杯子放到盘子上"。你给它看 50 个示范视频，它学会了基本动作。但你把它换到新厨房，灯光变了，杯子位置变了，它就傻眼了。

这时候你想：要是有个能"预测未来"的 World Model (WM) —— 给它当前画面，它能脑补出接下来杯子怎么被抓起来 —— 那这个 WM 脑子里的物理常识，能不能传给 robot policy，让它更 robust？

这就是这篇 paper 干的事。

---

## 问题出在哪

之前的人怎么传？路线是这样的：

```
WM 生成一段假视频 → 从假视频里反推 robot 该做什么动作 → 用这些动作训练 policy
```

听起来合理，但有个致命缺陷：**WM 生成的视频会 hallucinate**。杯子可能多长出一个把手，drawer 的把手可能凭空消失，不同摄像头的画面可能对不上。这些像素级别的瑕疵，经过"反推动作"这一步，会被放大成完全错误的训练信号。

打个比方：你让一个画师画"抓杯子"的过程，他画得七七八八，杯子形状都变了。然后你拿这画去反推"手该怎么动"，那推出来的动作肯定也是歪的。

DreamGen (https://arxiv.org/abs/2410.24164) 就是走这条路的，实验里它很不稳定 —— 数据加到一半 success rate 反而掉下来。

---

## World2Act 的核心 insight

**别看像素，看 latent**。

即使 WM 画的杯子多出个把手，它 latent space 里编码的"手在靠近、抓取、抬起"这个 dynamics 是对的。就像一个人讲故事，细节记错了（杯子颜色说成红色），但情节走向（抓起来放下去）是对的。

所以应该直接对齐 video 的 latent 和 action 的 latent，跳过脆弱的 pixel-to-action 反推。

---

## 但还有个前置问题：WM 画不了长视频

Robot 任务时长差异巨大。短的两秒，长的二十秒。但 video diffusion model 都在固定长度（比如 16 帧）上 pretrain，让它生成长视频会 error accumulation —— 画到后面越来越崩。

World2Act 的解法很朴素：**把长任务拆成短的 atomic skill**。

"Pick cup from cabinet and place on counter" → 拆成 ["pick cup", "place cup"]。每个 skill 短，WM 能稳定生成；skill 之间用上一段的最后一帧作为下一段的起点，保证连贯。

怎么拆？两路并行：

- **视觉路**：看 gripper 的开合。夹爪合上 = contact，张开 = non-contact。一个 "reach → grasp → release" 循环就是一个 skill segment
- **语言路**：用 DeepSeek 把 high-level instruction 拆成 atomic prompts

然后对齐：第 116 帧 gripper 合上 = "pick" 开始，第 230 帧 gripper 再次合上 = "place" 开始。LLM 负责这个 alignment，而且 prompt 里严格规定"不许 hallucinate frame number，indices 数对不上就 discard"。

结果：RoboCasa-Skill 同步率 96.2%，video length distribution 从 long-tail 变成集中分布，训练 stability 大幅提升。

---

## 两阶段方法

### Stage 1：学一个对齐空间

想象你要让两种"语言"能互译：video dynamics 是一种语言，robot action 是另一种。你需要一个 shared latent space，让表达相同行为的两种 modality 在这个空间里靠近。

具体做法：
- Frozen WM 提取 video latent（每个 latent 对应 4 帧）
- Video adapter（一个小 CNN）把 video latent 压到 32 维 embedding
- Action adapter（一个小 MLP）把 4 帧的动作 concat 起来也压到 32 维
- 用 **contrastive loss** 拉近 matching pair，推开 non-matching pair

公式核心就是 InfoNCE，双向的：

$$\mathcal{L} = -\log \frac{e^{\text{sim}(z^v, z^a)/\tau}}{\sum_j e^{\text{sim}(z^v, z_j^a)/\tau}} - \log \frac{e^{\text{sim}(z^a, z^v)/\tau}}{\sum_j e^{\text{sim}(z^a, z_j^v)/\tau}}$$

- $z^v, z^a$：video 和 action 的 embedding sequence
- $\tau = 0.1$：temperature，控制 softmax 的 sharpness
- $\text{sim}$：chunk-averaged cosine similarity，按时间步对齐

关键 design choice：**chunk-wise alignment**。不要先 global pool 再算相似度，那样模型会偷懒用"这是什么任务"来匹配，而不是"这一步在干嘛"。Ablation 证明 chunk-wise 比 global pooling 高 3.3% success rate。

同时还有个 reconstruction loss 保证 action latent 能还原回原始动作，防止 collapse。

### Stage 2：Residual post-training

直接 finetune 整个 VLA backbone 又慢又容易忘掉之前学的。World2Act 的选择是：**freeze backbone，学一个小 residual correction**。

$$a_{\text{final}} = a_{\text{base}} + a_{\text{residual}}$$

- $a_{\text{base}}$：frozen GR00T-N1.6 输出的 base action
- $a_{\text{residual}}$：小 network（2 层 transformer, hidden=32）预测的修正量

这个 residual network 输入当前 observation + base action，输出 correction。训练信号来自 Stage 1 学会的对齐空间：

```
当前 policy 跑 rollout → 拿到 action sequence
WM 根据初始条件生成 video latent → 投影到 video embedding
把 action sequence 也投影到 action embedding
算 contrastive loss，拉近这两个 embedding
```

这就是 reward-free 的。不需要 success signal，不需要 reward model，完全靠"我的 action 要和 WM 想象的 video dynamics 一致"这个 objective 驱动。

---

## 为什么能 work —— 三个直觉

### 1. Latent 比 Pixel robust

Pixel space 太高维太敏感。杯子多一个像素的把手，对 IDM 来说就是完全不同的 action label。Latent space 把这些 high-frequency noise 抹掉，保留 task-relevant dynamics。Fig. S3 里那些 hallucination 案例在 latent space 影响小得多。

### 2. Skill decomposition 稳住 WM

Video diffusion 在 fixed-length 上训练，长视频会 drift。拆成 atomic skill 后，每个 segment 长度均匀，WM 能稳定生成。Fig. 5a 显示 Skill-WM 在 5K steps 收敛，Base-WM 到 20K 还没收敛。稳定的 WM → 稳定的 latent → 稳定的 post-training signal。

### 3. Residual 利用 pre-aligned space

LoRA 是从头学 low-rank adapter，没法利用 Stage 1 学好的 cross-modal structure。Residual policy 把 base action 通过 frozen action adapter 转成 token，再和 observation token 一起 attention —— 它直接在 pre-aligned 空间里工作，sample efficiency 高。Table S2 显示 residual 比 LoRA rank=32 快 2.25×，success rate 还高 0.5%。

---

## 结果怎么样

几个关键数字：

- **RoboCasa**：GR00T-N1.6-ft + World2Act 达到 72.6% SR，比 DreamGen 高 2.1%，用同样 50 条 synthetic trajectory
- **LIBERO**：DreamGen 反而 degrade（0.970→0.926），World2Act 稳定提升到 0.981
- **Scaling**：World2Act 随数据量单调提升，DreamGen 在 N=500 时崩一下
- **Real robot**：Franka arm 上 3 个任务平均 +6.7%
- **Inference**：251.9 Hz，完全 real-time

还有个漂亮的发现：**cosine similarity 和 success rate 强相关**。训练时看 cosine similarity $\text{sim}(z^v, z^a)$ 就能判断 post-training 有没有效，不用跑完整 evaluation。这对工程实践很有用。

---

## 局限性（作者自己也承认）

1. **Backbone sensitive**：GR00T-N1.6 上 gain 大（+2.5%），Cosmos Policy 上 gain 小（+0.6%）。因为 Cosmos Policy 自己已经是 joint video-action model，留给 latent alignment 的空间小
2. **Real-world gain modest**：+6.7% 不错但绝对值不高，说明 WM 对复杂物理（contact-rich, deformable）建模还不够
3. **Skill decomposition 依赖 gripper signal**：对 non-prehensile manipulation（用胳膊关门）效果差
4. **Imagination-to-execution gap**：Fig. S4 那个 failure case 很典型 —— WM 想象里成功抓到 stove knob，实际执行没抓牢。Visual plausibility ≠ motor feasibility

---

## 一句话总结

**别让 robot 照着 hallucinate 的像素学动作，让它的 action latent 和 world model 的 dynamics latent 对齐就够了。** 长任务拆成短 skill 稳住 WM，frozen backbone + 小 residual 高效 post-training，contrastive loss 做 reward-free 的 alignment。

整个 framework clean，每个 design choice 有 ablation，工程上 tractable。本质上是把 contrastive learning 那套哲学（CLIP, SimCLR, V-JEPA）应用到 WM→VLA transfer 这个具体问题上。

Project page: https://wm2act.github.io/

---

# World2Act: 通过 Skill-Compositional World Models 实现 Latent Action Post-Training

## 1. 核心问题与动机

这篇paper解决一个 embodied AI 中非常核心的问题：**如何把 World Model (WM) 学到的 dynamics priors 有效迁移到 Vision-Language-Action (VLA) policy 中**。

### 1.1 现有方法的痛点

先理解一下 VLA post-training 的 landscape。当前 strong VLA models 像 GR00T-N1.6 (https://research.nvidia.com/labs/gear/gr00tn1_6/) 或 Cosmos Policy (https://arxiv.org/abs/2601.16163) 主要通过 behavior cloning 训练，但在 OOD (out-of-distribution) 场景下 generalization 很差。原因是它们缺乏 robust dynamics priors —— 知道"做什么"但不知道"物理上怎么演进"。

World Models 比如 Cosmos-Predict2 (https://arxiv.org/abs/2501.03575) 或 V-JEPA 2 (https://arxiv.org/abs/2506.09985) 能学到这种 dynamics priors，但如何迁移？Prior work 如 DreamGen (https://arxiv.org/abs/2410.24164 走的路线) 主要走 **pixel-space supervision**：

```
WM 生成 video rollout → Inverse Dynamics Model (IDM) 从 pixels 推断 pseudo-action → 用这些 action 训 VLA
```

这条 pipeline 有个根本性的弱点：**pixel rollout 会 hallucinate**。WM 生成的视频里杯子可能多出一个 handle，drawer 的把手可能消失，wrist view 和 third-person view 不一致（见 paper Fig. S3）。这些 pixel artifacts 经过 IDM 后被 **放大** 成错误的 action labels，导致 policy drift。DreamGen 在 scaling 实验中（Fig. 5b）表现不稳定，N=500 时 success rate 反而掉到 69.1%，就是这个原因。

### 1.2 World2Act 的核心 insight

**Latent space 比 pixel space 更 robust**。即使 WM 生成的 pixels 有局部 artifacts，其 latent dynamics（表征层面的时序+物理交互信息）依然准确。所以应该直接在 latent space 对齐 video dynamics 和 action，跳过 fragile 的 pixel-to-action inversion。

这让我想到 contrastive learning 的哲学 —— 与其 reconstruct 原始信号（容易过拟合 noise），不如学习一个 alignment 的表示空间。SimCLR (https://arxiv.org/abs/2002.05709) 和 CLIP (https://arxiv.org/abs/2103.00020) 都是这个思路。

---

## 2. Skill-Compositional World Model —— 解决 arbitrary-length generation

### 2.1 为什么需要 skill decomposition

Video diffusion backbones 都在 **fixed-length clips** 上 pretrain（通常 16 frames 或更短）。但 robotic tasks 的 horizon 差异巨大 —— RoboCasa 里一个 "pick and place" 可能 50 frames，一个 "arrange kitchen" 可能 500 frames。直接生成长视频会 error accumulation，这点在 LIVE (https://arxiv.org/abs/2602.03747) 和 StableWorld (https://arxiv.org/abs/2601.15281) 中都有讨论。

### 2.2 Data decomposition pipeline

这个 pipeline 很 elegant，分两路走：

**Visual stream segmentation** —— 用 gripper width 作为 contact 信号：

$$\delta_t = w_0 - w_t$$

其中 $w_0$ 是初始 fully-open width（calibrated），$w_t$ 是 time $t$ 的 width。$\delta_t \approx 0$ 表示 gripper 完全打开（non-contact），$\delta_t \geq \Delta$ 表示 contact（threshold $\Delta = 0.05$m）。

每个 segment 定义为一个完整的 action cycle：从 non-contact 开始，到 contact event 结束。这很符合 manipulation 的 atomic 结构 —— reach → grasp → lift/place → release。

**Language stream decomposition** —— 用 DeepSeek-R1 (https://www.nature.com/articles/s41586-025-09422-z) 把 high-level instruction 拆成 atomic skill prompts。比如 "pick the hot dog from the cabinet and place it on the counter" → ["pick hot dog", "place hot dog on counter"]。

**Synchronization** —— 把 visual segments 和 language prompts 按时序对齐。关键是 LLM prompt 设计（见 paper Section E）：LLM 必须严格用提供的 gripper indices，不能 hallucinate frame numbers；如果 indices 数和 schema steps 不匹配就 discard 这条 trajectory。这保证了 data quality。

结果：RoboCasa-Skill 同步率 96.2%，LIBERO-Skill 86.9%。从 Fig. 3 看，分解后 video length distribution 从 long-tail 变成 concentrated unimodal —— RoboCasa median 附近 density +17%，LIBERO +72%。这个 distribution shaping 对训练 stability 极其关键。

### 2.3 Skill-compositional inference

推理时 autoregressive：LLM 先生成 atomic prompt list → WM 为每个 prompt 生成一个 sub-video → 用前一个 sub-video 的 last frame 作为下一个的 initial condition → concatenate。

这其实是个 **hierarchical decomposition**：global task = ordered composition of atomic skills，每个 skill 内部用 fixed-length generation（WM 擅长的），skill 之间用 frame-level conditioning 保证 continuity。比 RoboEnvision (https://arxiv.org/abs/2501.06605) 那种 keyframe + interpolation 的两阶段方法更简洁。

---

## 3. World2Act 方法详解 —— 两阶段 latent alignment

### 3.1 Stage 1: Aligning Video Dynamics and Robot Actions

**目标**：学习一个 shared latent space，让 video dynamics 和 robot action 在这个空间里对齐。

**Frozen Skill-WM** $W$ 提取 video latents $\mathbf{V} = \{V_t\}_{t=1}^{T}$，每个 $V_t \in \mathbb{R}^{C \times H \times W}$ 对应 $M$ 个 low-level frames（temporal compression，Cosmos-Predict2 的 latent resolution 是 $16 \times 60 \times 104$，即 $C=16, H=60, W=104$）。

**Video Adapter** $B_v$：CNN-based，把 $V_t$ 映射到 $z_t^v \in \mathbb{R}^D$（$D=32$）：
```
Conv2d(C, 64, k=3, s=2, p=1) → GroupNorm → GELU
Conv2d(64, 128, k=3, s=2, p=1) → GroupNorm → GELU
AdaptiveAvgPool2d((1,1)) → Flatten → FC(128, D)
```

**Action Adapter** $B_a$：MLP-based，把 raw actions $\mathbf{a}_{gt}$ 按 $M$-frame window concat 成 chunk，再 encode 到 $z_t^a \in \mathbb{R}^D$：
```
Flatten → FC(A×M, 128) → GELU → FC(128, 64) → GELU → FC(64, D)
```
这里 $A$ 是 action dimension（RoboCasa $A=12$，LIBERO $A=7$），$M=4$ 是 chunk size。Concat 而非 average 保留了 high-frequency control variations（rapid joint movements）。

**两个 loss**：

(1) **Reconstruction loss**（保证 action latent 保留 kinematics）：
$$\mathcal{L}_{\text{recon.}} = \|\mathbf{a}_{gt} - \hat{\mathbf{a}}\|^2$$
Action Decoder $\mathcal{D}_a$ 把 chunk latent 解码回 $M$ 个 frame-level actions。

(2) **Bidirectional chunk-wise InfoNCE**（核心 alignment）：

$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i^v, \mathbf{z}_i^a)/\tau)}{\sum_{j=1}^{B} \exp(\text{sim}(\mathbf{z}_i^v, \mathbf{z}_j^a)/\tau)} - \log \frac{\exp(\text{sim}(\mathbf{z}_i^a, \mathbf{z}_i^v)/\tau)}{\sum_{j=1}^{B} \exp(\text{sim}(\mathbf{z}_i^a, \mathbf{z}_j^v)/\tau)}$$

变量解释：
- $\mathbf{z}_i^v, \mathbf{z}_i^a$：第 $i$ 个 sample 的 video 和 action embedding sequences（长度 $T$）
- $\tau = 0.1$：temperature，控制分布 sharpness
- $B = 16$：batch size
- $\text{sim}(\mathbf{z}_i^v, \mathbf{z}_j^a) = \frac{1}{T}\sum_{t=1}^{T} \cos(z_{i,t}^v, z_{j,t}^a)$：chunk-averaged cosine similarity

**关键设计**：chunk-wise alignment 而非 global pooling。如果先 global pool 再算 cosine（ablation in Table S3），success rate 掉到 0.693（vs 0.726）。因为 global descriptor 会让模型用 coarse task identity 匹配（"这是 pick 任务"），而不是 fine-grained temporal dynamics（"这一帧在 reach，下一帧在 grasp"）。

**Negative sampling** 策略很精细：
- Easy negatives：不同 atomic skills 的 samples
- Hard negatives：同一 skill 的不同 demonstrations（prevent representational collapse，hard negative ratio 0.25）

总 loss：$\mathcal{L} = \mathcal{L}_{\text{recon.}} + \mathcal{L}_{\text{contrastive}}$，训练 30K steps。Fig. S2 显示 $\mathcal{L}_{\text{recon.}}$ 收敛到 ~0.01，$\mathcal{L}_{\text{contrastive}}$ 收敛到 ~0.05。

### 3.2 Stage 2: Latent Action Post-Training via Residual Policy

这里用了一个很聪明的设计 —— **residual policy learning** (Silver et al., https://arxiv.org/abs/1812.06298)，而不是直接 finetune VLA backbone。

**为什么 residual？**
1. VLA backbone 参数巨大（GR00T-N1.6 是 billion-scale），直接 finetune sample-inefficient 且 catastrophic forgetting
2. Frozen backbone 只需 forward pass，不用 backprop through 整个 VLA —— Table S2 显示 residual policy 训练 6.8 hrs vs LoRA rank=32 的 15.3 hrs，2.25× speedup
3. Residual 能利用 Stage 1 pre-aligned 的 shared latent space（LoRA 是从头初始化 low-rank matrices，无法 exploit 这个结构）

**Policy 结构**：
$$\pi_{\text{final}} = \pi_{\text{base}} + f^\theta$$

每个 chunk $t$：
- Frozen $\pi_{\text{base}}$ 输出 $a_{\text{base},t} = \pi_{\text{base}}(s_t)$
- Residual policy $f^\theta$ 预测 correction $a_{\text{residual},t} = f^\theta(s_t)$
- 执行 $a_{\text{final},t} = a_{\text{base},t} + a_{\text{residual},t}$，open-loop 跑 $M$ 步到 $s_{t+1}$

**Residual network $f^\theta$ 架构**（Table S6）：
- State encoder：FC($D_{\text{proprio}}=53$, 128) → ReLU → FC(128, D) → LayerNorm
- Visual encoder：3 层 Conv2d（3→16→32→32，stride 4/2/2）→ Flatten → FC(6272, D) → LayerNorm
- 把 base action $a_{\text{base},t}$ 通过 frozen $B_a$ 得到 action token $\mathbf{x}^{(0)}$
- Concat $\{\mathbf{x}^{(0)}, \mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}\}$（$N_{\text{src}}=3$ tokens）
- Self-Attention Transformer（2 layers, 4 heads, $d_{\text{model}}=D=32$）
- 取 $\mathbf{h}^{(0)}$ → MLP → $\mathcal{D}_a$ → $a_{\text{residual},t}$

**Training signal**：
- Online rollout（在 post-training split，与 test scene 严格 disjoint）
- Frozen Skill-WM $W$ 生成 video latents $\mathbf{V}$ conditioned on rollout initialization
- Frozen $B_v$ 把 $\mathbf{V}$ 投影到 $\mathbf{z}^v$（target）
- Frozen $B_a$ 把 $\mathbf{a}_{\text{final}}$ 投影到 $\mathbf{z}^a$
- 算 Stage 1 的 $\mathcal{L}_{\text{contrastive}}$，更新 $f^\theta$

这是 **reward-free** 的 —— 不需要 success signal 或 reward model，完全靠 cross-modal consistency 驱动。In-batch $B$ 个 parallel environments，same environment = positive，different initial conditions same task = hard negatives。

**Intuition**：WM 的 video latent 编码了 "task 应该怎么演进"，VLA 的 action latent 编码了 "policy 实际让 robot 怎么动"。Contrastive loss 拉近这两者，就是让 policy 的行为向 WM 的 dynamics prior 靠拢。即使 WM pixel 有 artifact，latent dynamics 依然 informative。

---

## 4. 实验结果深度解析

### 4.1 RoboCasa 主结果（Table 1）

| Method | Real Demos | Synthetic | SR |
|--------|-----------|-----------|-----|
| GR00T-N1.6-ft | 350 | 0 | 0.701 |
| + DreamGen | 350 | +50 | 0.705 |
| + VLA-RFT | - | - | 0.710 |
| + Ctrl-World | - | - | 0.698 |
| **+ World2Act** | 350 | +50 | **0.726** |

关键对比：DreamGen 用同样 50 synthetic trajectories 只提升 0.4%，World2Act 提升 2.5%。这说明 **latent supervision 的 data efficiency 远高于 pixel supervision**。

而且对比 UWM (https://arxiv.org/abs/2504.02792) 用 1000 real demos 达到 0.608，World2Act 用 50 real + 50 synthetic 达到 0.663，data efficiency 极高。

### 4.2 LIBERO 主结果（Table 2）

GR00T-N1.6-ft + DreamGen 反而 **degrade** 到 0.926（baseline 0.970），因为 LIBERO 的 IDM pseudo-labeling 质量差。World2Act 稳定提升到 0.981。Long-horizon 任务上 World2Act 0.940 vs DreamGen 0.876，差距明显 —— 这正验证了 skill decomposition 对 long-horizon 的价值。

### 4.3 Cosine similarity 与 success rate 的相关性（Fig. 5a）

这是个很有意思的 finding。Cosine similarity $\text{sim}(\mathbf{z}^v, \mathbf{z}^a)$ 与 success rate **strong positive correlation**。World2Act 的 cosine similarity 在 5K steps 内快速爬升到接近 1.0，success rate 从 70.1% → 72.6%。

这给了一个可观测的 proxy metric —— 不用跑 full evaluation，看 cosine similarity 就能判断 post-training 是否有效。对 practical deployment 很有用。

### 4.4 Scaling behavior（Fig. 5b）

World2Act 从 N=0 到 N=1000 **monotonically** 提升 70.1% → 72.6%。DreamGen 在 N=500 时 drop 到 69.1%（pixel noise 注入），N=1000 才恢复到 70.5%。这个 stability 差异是 latent vs pixel supervision 的直接体现。

### 4.5 Cross-task generalization（Fig. 5c）

24 个 RoboCasa tasks 分成 12 seen + 12 unseen。Pick-and-Place (PnP) 系列 12 个任务完全 held-out。训练 seen tasks 数量增加时，unseen PnP success rate 持续提升 —— GR00T-N1.6-base +3.3%，Cosmos Policy +1.6%。

这很关键：**WM 从未见过 PnP 的 post-training trajectories**，但 generalization 依然提升。说明 WM 学到的是 **transferable dynamics priors**（reach, grasp, lift, place 这些 atomic motions），而非 task-specific patterns。Skill decomposition 在这里起了双重作用 —— 既稳定了 WM 训练，又让 atomic skills 可以 recompose 到新任务。

### 4.6 Real-world experiments（Fig. 7）

Franka Research 3 arm，3 个任务（pick cup & place on plate, pick up bowl, close drawer），每任务 20 real demos + 100 synthetic trajectories。Average **+6.7%** improvement。Inference speed 251.9 Hz（GR00T-N1.6-ft + World2Act），完全满足 real-time control。

Fig. S4 展示了一个 failure case：WM imagination 成功 grasp stove knob，但 VLA 执行时 grip 不牢。这揭示了 **imagination-to-execution gap** —— visual plausibility ≠ motor feasibility。这是 future work 的方向。

---

## 5. 与相关工作的对比

### 5.1 vs DreamGen / pixel-space post-training
- DreamGen：WM pixel rollout → IDM → pseudo-action → behavior cloning
- World2Act：WM latent → contrastive align with action latent → residual correction
- 优势：跳过 fragile IDM，对 pixel artifact robust

### 5.2 vs Cosmos Policy (https://arxiv.org/abs/2601.16163)
- Cosmos Policy：joint embedding space + 3 prediction heads (action, future frames, values)
- World2Act：cross-modal contrastive，不 joint train，而是 align frozen representations
- 优势：避免 high-dimensional joint embedding 的 training instability（Guo et al. https://arxiv.org/abs/2501.04565 提到的问题）

### 5.3 vs V-JEPA 2 (https://arxiv.org/abs/2506.09985)
- V-JEPA 2：model-based planning，用 $\ell_1$ loss on hidden states 对齐 final frame
- World2Act：supervise 整个 trajectory 的 latent dynamics，而非 terminal frame
- 优势：fine-grained temporal alignment，不只看结果

### 5.4 vs LAPA (https://arxiv.org/abs/2410.21258)
- LAPA：pretrain latent action space from cross-embodiment data，global task-level code
- World2Act：in-distribution synthetic rollouts + step-by-step alignment
- 优势：task-specific fine-grained supervision

---

## 6. Limitations 与个人思考

1. **Backbone sensitivity**：GR00T-N1.6 上 gain 显著（+2.5%），Cosmos Policy 上 marginal（+0.6%）。作者归因于 Cosmos Policy 的 joint video-action representation 已经 capture 了 cross-modal correlation，留给 latent alignment 的空间小。这暗示 World2Act 对 **decoupled video-action architecture** 更有效。

2. **Real-world gap**：+6.7% real-world improvement 不错但绝对值仍 modest。WM 对 complex physical dynamics（contact-rich, deformable objects）的建模仍不足。

3. **Skill decomposition 依赖 gripper signal**：对于 non-prehensile manipulation（用 arm 关门）效果差（LIBERO 同步率 86.9% < RoboCasa 96.2%）。更通用的 contact detection（force-torque, tactile）可能更好。

4. **Residual policy 的 capacity**：小 network（2-layer transformer, D=32），对于需要大幅修正 base policy 的场景可能不够。一个可能的 extension 是 adaptive residual magnitude 或 hierarchical residual。

5. **LLM dependence**：Skill decomposition 依赖 DeepSeek 的 instruction following。如果 task 描述 ambiguous 或 schema 不完整，alignment 会失败。一个 self-supervised 的 segmentation（比如从 video 自身学 contact boundaries）会更 robust。

---

## 7. 对你（Karpathy）的 intuition building

这个工作其实在重复一个 deep learning 里的经典 lesson：**representation alignment > signal reconstruction**。从 image generation 到 VAE，从 BERT 到 CLIP，从 pixel RL 到 latent RL —— 我们反复发现，在 latent space 做 alignment 比 reconstruct 原始信号更 sample-efficient、更 robust to noise。

World2Act 把这个 lesson 应用到 WM→VLA transfer：不 reconstruct pixels，不 invert pixels to actions，而是直接在 latent space 用 contrastive objective 对齐 video dynamics 和 action dynamics。这和 V-JEPA (https://arxiv.org/abs/2301.08243) 的哲学一脉相承 —— joint-embedding predictive architecture 比 pixel-reconstructive 更 powerful。

Residual policy 的选择也很自然 —— 你之前在 Tesla AI Day 讲过类似思路：大模型提供 base capability，轻量 module 做 task-specific adaptation。这里 frozen VLA = base，residual = adapter，contrastive latent loss = alignment signal。整个 framework 很 clean，工程上 tractable，且每个 design choice 都有 ablation 支撑。

Project page: https://wm2act.github.io/

值得 follow 的方向：把这个 framework 扩展到 humanoid whole-body control，或结合 diffusion policy 的 iterative refinement 做 latent-space planning。
