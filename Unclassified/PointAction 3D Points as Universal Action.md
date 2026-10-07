---
source_pdf: PointAction 3D Points as Universal Action.pdf
paper_sha256: f0aa3184fef616446167f9d85660a1bb7db1f6db1eb9b00a498dccceb5b48fa6
processed_at: '2026-08-06T04:56:51-07:00'
target_folder: Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 PointAction

## 一句话总结

**让video model不光predict未来RGB帧, 还predict每个pixel对应的3D坐标怎么动, 然后用一个轻量decoder把"robot身上那些点的3D运动轨迹"翻译成joint commands。**

---

## 为什么要搞这个

想象你教一个robot "把杯子从桌上搬到柜子里"。两种主流做法都有坑:

**VLA路线** (像π0, GR00T, OpenVLA): 给VLM看图+指令, 直接output action tokens。问题在于VLM很懂语义 (知道"杯子是杯子"), 但它脑子里没有"物理世界怎么演化"的model。它不知道杯子被推一下会滑多远, 不知道gripper夹住杯壁需要多大aperture。它只能从大量paired (observation, action) data里死记硬背dynamics。数据贵, 换个新场景就崩。

**VAM路线** (像VPP, Cosmos Policy): 先用video diffusion生成"未来会怎样"的RGB视频, 再从视频里decode出action。听起来很美 — video prior在大规模internet video上学过世界怎么动。但**RGB视频歧义太大**。你看一帧RGB, 知道gripper在杯子旁边, 但不知道gripper离杯子3cm还是5cm, 不知道杯子是空还是满, 不知道gripper是刚闭合还是正要闭合。这些metric信息全丢了, action decoder只能从像素appearance变化反推control, 非常困难。

PointAction的bet: **把3D geometry显式predict出来, 当作video和action之间的"中间语言"**。这个中间语言embodiment-agnostic — 不管哪个arm执行"把杯子搬进柜子", 杯子在3D空间里的运动轨迹是同一个; 只是不同arm需要不同的joint angle去实现这个轨迹。

---

## 怎么做的

整个pipeline分成两段, 两段解耦:

### 第一段: 4D Video Model (universal, 贵, pretrain一次)

在LVP (一个robot video foundation model)基础上加XYZ prediction。输入: 当前RGB帧 + 指令。输出: 未来49帧的RGB video + 对应的49帧XYZ pointmap (每个pixel一个3D坐标)。

关键技术trick是**WidthConcat**:

最naive的做法是把RGB (3通道)和XYZ (3通道)拼成6通道输入。但预训练video DiT是在3通道RGB上学的, 突然多3个channel, backbone的texture prior全乱了, 两个modality对不齐。

WidthConcat的做法: RGB和XYZ分别过同一个VAE encoder, 各得到latent $z^o, z^u \in \mathbb{R}^{C \times h \times w}$。然后沿width dimension concat, 得到 $z^{joint} \in \mathbb{R}^{C \times h \times 2w}$。

效果是: latent space里每个RGB patch的右边紧挨着它的spatially对应XYZ patch。DiT的self-attention天然就把它们关联起来。RGB backbone的channel structure完全保留, 没破坏pretrained prior。

训练用Diffusion Forcing + Flow Matching + LoRA (rank 128, 只更209M params)。history context有50%概率clean (不加noise), 这鼓励模型对noisy context也robust。

### 第二段: Action Decoder (lightweight, 便宜, 每个arm单独训)

4D model预测出49帧RGB+49帧XYZ。但decoder不需要整个scene的XYZ — 只需要**robot表面那些点的3D轨迹**。

怎么拿到robot mask? 用SAM 3在predicted RGB video上跑open-vocabulary "robot" segmentation, 得到mask $\tilde{\alpha}$, 然后:

$$\tilde{u}_{robo} = \tilde{u}_{xyz} \odot \tilde{\alpha}$$

把non-robot pixel全zero out。

然后每帧用FPS下采样到512个点, 过一个PointNet MLP得到per-frame feature。这些feature作为conditioning, 喂给一个轻量DiT decoder (6层, hidden 256), 用DDIM 10步一次decode出整个49步action chunk。

Global condition (初始robot state $s_t$ + diffusion timestep $j$) 通过AdaLN注入, clean separation: point feature走token pathway, state走AdaLN pathway。

---

## 为什么work (核心insight)

Ablation table (Table 3)是最informative的, 逐行看:

| 配置 | ID Success |
|---|---|
| Decoder只吃RGB | 25.1% |
| Decoder只吃全scene XYZ | 27.1% |
| Decoder吃post-hoc depth (DA3 on generated RGB) | 28.4% |
| Decoder吃RGB + robot-only XYZ | 37.2% |
| Decoder吃robot+scene XYZ (两个encoder) | 40.3% |
| **Decoder只吃robot-only XYZ (joint预测)** | **47.7%** |

几条intuition:

**1. RGB Only → XYZ Only (Robot): +22.6%**。这个gap巨大。说明action grounding的dominant failure mode确实是geometric ambiguity — 像素告诉你"那里有个东西", 但不告诉你"那个东西3D上在哪"。把3D拎出来, decoder一下子轻松了。

**2. RGB + XYZ反而比纯XYZ差** (37.2% vs 47.7%)。lightweight decoder容量有限, RGB里的光照、纹理、阴影都是task-irrelevant noise, 反而distract它。这跟VLA的"global scene understanding → action"哲学相反 — 在action层面, 你只需要知道robot怎么动, 不需要知道背景墙什么颜色。

**3. Full scene XYZ << Robot+Scene < Robot-only**。全scene的点喂给单个PointNet, 27.1%。加一个separate encoder处理robot点, 恢复到40.3%。完全mask掉scene, 只留robot点, 47.7%。说明scene points对action prediction是噪音 — 它不携带robot state信息, 注入ambiguity。这个结论有点反直觉: 我们以为"理解整个场景"有助于action, 结果action decoder只需要"理解机器人自己怎么动"。

**4. Post-hoc depth (28.4%) << Joint prediction (47.7%)**。先生成RGB, 再跑Depth-Anything-V3估深度, 远差于joint generation。这是cascaded pipeline的经典问题 — error accumulation + distribution shift。depth estimator没见过generated RGB的artifact, 估出来的depth就off。joint generation让RGB和XYZ在denoising过程中互相inform, 输出temporally + spatially consistent的4D。

---

## Cross-embodiment transfer为什么是关键validation

论文最强的evidence不是simulation数字, 是**两个pretraining完全没见过的arm的真机transfer**:

**xArm7** (50 demos/task, 100 rollouts/task):
- GR00T N1.7: 14.7% avg
- π0.5: 22.7% avg  
- **PointAction: 43.0% avg**

**YAM arm** (20 demos/task, 20 rollouts/task):
- GR00T N1.5: Pick Pens 20%, Insert Cups 15%, Stack Cubes 0%
- π0: 几乎全崩
- **PointAction: Stack Cubes 20%, Pick Pens 60%, Insert Cups 50%**

注意YAM这个setup: pretraining完全没见过的arm, 只20条demo。VLA baseline几乎完全失败 (stack cubes 0%)。PointAction还能做到20-60% success。

这cleanly验证了核心claim: **3D point dynamics是embodiment-agnostic的action interface**。universal 4D model预训练的"物体怎么动、gripper怎么闭合"的3D prior, 跟具体arm无关。新arm只需要学"怎么用我的joint去实现这个3D轨迹", 这个mapping相对简单, 20-50条demo够学。

---

## 我觉得有意思的几个点

**1. Embodiment-agnostic的根源**: 4DVM **不condition on $s_t$** (arm-specific state)。它只看RGB + 指令, 预测"3D世界怎么演化"。这个演化规律是物理决定的, 跟哪个arm执行无关。Decoder才condition on $s_t$, 因为只有decoder需要知道"我这个arm的joint现在在哪"才能算出下一步joint command。这个factorization很clean。

**2. 为什么是pointmap而不是别的3D表示**: pointmap是pixel-aligned的, 每个pixel一个XYZ。好处是: (a)跟RGB天然spatially aligned, 可以WidthConcat; (b)不需要reconstruction (不像NeRF要volume render); (c)直接可actionable (知道pixel p的XYZ轨迹, 就知道那个表面点怎么动)。对比TesserAct预测RGB+depth+normal再integrate成4D, 多一步reconstruction, 而且normal integration在real scene上有noise。PointAction直接输出actionable geometry, 跳过reconstruction。

**3. SAM 3 mask是个hack但work**: 4DVM预测dense XYZ时不知道哪些是robot。理想情况应该joint learn mask, 但这需要embodiment-specific supervision, 破坏universal pretraining。用SAM 3 post-hoc segment "robot"是个clever workaround — SAM 3是open-vocabulary的, 跨embodiment都能用, 不需要重新标注。代价是: SAM 3如果segment错, decoder吃garbage。这是self-occlusion failure mode的部分原因。

**4. Open-loop 49步chunk是当前天花板**: 一次forward pass出49步action, 6分钟inference (B200上), 然后open-loop执行。物体slip了, 碰到意外了, 都没法recover。这限制了实际部署。Future work提到distill成autoregressive + KV cache, 这是正道, 但还没做。现在这个版本更像是一个"paradigm validation"而不是"ready-to-deploy system"。

**5. Video model的capacity够不够jointly学RGB+XYZ**: 答案是够了。Table 4显示PointAction的RGB质量 (PSNR 19.631, SSIM 0.821, FVD 320)跟LVP (RGB-only foundation model, 19.613, 0.816, 330)持平甚至略好。LoRA fine-tune没破坏RGB prior, 同时adapt出XYZ能力。这说明video diffusion backbone的capacity远没饱和, 加一个modality不挤占原有能力。这个结论对未来的"video model + more modalities"路线很鼓舞。

---

## Limitation和我的疑问

**1. Self-occlusion**: robot在generated video里挡住自己的一部分, XYZ pointmap不完整, decoder收到degraded input。Single-view的根本限制。多view或active view selection可能是方向。

**2. SAM 3依赖**: mask质量直接决定decoder input质量。end-effector被occlude时SAM 3可能seg不准。能不能在4DVM里joint learn一个robot mask channel? 但这又需要embodiment-specific supervision...

**3. WidthConcat的attention cost**: latent width翻倍, self-attention是$O((2w)^2)$, 约4×cost。大模型上可能bottleneck。有没有更高效的spatially-aligned fusion? 比如cross-attention或gated fusion。

**4. VAE对XYZ的smoothing**: VAE在RGB上训的, 有KL regularization, 对continuous metric signal可能over-smooth。XYZ的fine-grained geometry (比如gripper tip的精确位置)可能被smooth掉。试过专门train XYZ VAE或在pixel space做XYZ diffusion吗?

**5. PointNet是不是太弱**: 2017年的encoder, 现在有PointTransformer、Point-MAE。lightweight decoder的capacity瓶颈可能在point encoding。换更强encoder能涨多少?

**6. Closed-loop怎么真正实现**: distill成autoregressive是个方向, 但video diffusion的temporal coherence怎么在autoregressive框架下preserve? Self Forcing和Causal Forcing给出了初步答案, 但都没处理action grounding。

---

## TL;DR

PointAction把VLA/VAM的action grounding问题reformulate成: **video model学"世界怎么4D演化" (universal), decoder学"我这个arm怎么实现这个演化" (specific), 中间用robot表面点的3D轨迹当interface**。这个interface选得准 (metric 3D, embodiment-agnostic, pixel-aligned), 所以两边都能cleanly optimize, 还能cross-embodiment transfer。核心trick是WidthConcat保持RGB prior + SAM 3 robot masking + joint prediction避免cascaded error。Limitation是open-loop + slow inference, 需要distillation才能真正deployable。

---

# PointAction: 3D Points as Universal Action Representations — 深度解析

## 1. 核心intuition: 为什么是3D pointmaps?

当前robot learning的两条主线都各有bottleneck:

- **VLA (Vision-Language-Action)** 路线 (RT-2, OpenVLA, π0, GR00T, π0.5): VLM backbone提供semantic understanding, 但没有explicit model of *how scenes evolve through contact, motion, and long-horizon interaction*。policy只能从paired observation-action data学习dynamics, 超出distribution就崩。
- **VAM (Video-Action Model)** 路线 (VPP, Cosmos Policy, DreamVLA): 用预训练video diffusion作为implicit world model, 先rollout未来RGB帧, 再用inverse dynamics解码action。问题是**RGB-only rollout under-specified**: metric 3D motion在哪里? contact geometry长什么样? gripper尖端和物体接触面是什么形状? 这些在RGB帧里都implicit, action module被迫学一个从appearance changes到controls的困难映射。

PointAction的key insight: **dynamic 3D pointmaps是embodiment-agnostic的action interface**。同一个"把杯子从counter搬到stove"任务, 不管是Franka Panda、WidowX 250、xArm7还是YAM arm执行, task-relevant 3D geometry的演化规律是共享的 — 杯子要从位置A移动到位置B, 路径要避开obstacle, gripper要闭合到特定aperture。这些都可以表达为3D space里的metric motion + contact constraints, 而且supervision可以从大规模video data获取 (multi-view reconstruction, monocular depth, point tracking), 完全bypass昂贵embodiment-specific action label。

这就把video-to-action learning factorize成两块:
1. **Universal video-to-point model** (π_θ^{4DVM}): 在大规模video data上预训练, 预测RGB rollout + dynamic 3D pointmaps, embodiment-agnostic
2. **Embodiment-specific point-to-action decoder** (π_ψ^{DEC}): 轻量级, 用少量paired robot data训练, 把point dynamics映射成具体robot的joint commands

参考:
- Project page: https://oriontmt.github.io/pointaction/
- Diffusion Forcing (Boyang Chen et al.): https://arxiv.org/abs/2507.01892
- LVP backbone: https://arxiv.org/abs/2512.15840

---

## 2. 形式化: probabilistic factorization

VAM的基本setup: 给定t时刻observation $o_t \in \mathbb{R}^{H\times W\times 3}$, proprioceptive state $s_t$, language instruction $l$, 输出Δ步action chunk $\tilde{a} = a_{t:t+\Delta-1} \in \mathbb{R}^{\Delta \times D}$。

End-to-end VAM的joint distribution:

$$
(\tilde{o}, \tilde{a}) \sim \pi_\theta^{\text{VAM}}(\cdot \mid s_t, o_t, l) \quad (3.1)
$$

PointAction引入一个中间latent variable $\tilde{u}$ — 4D pointmap rollout, 写成marginalization:

$$
\pi(\tilde{o}, \tilde{a} \mid s_t, o_t, l) \approx \int \pi_\theta^{\text{4DVM}}(\tilde{o}, \tilde{u} \mid o_t, l) \cdot \pi_\psi^{\text{DEC}}(\tilde{a} \mid \tilde{u}, s_t) \, d\tilde{u} \quad (3.2)
$$

变量含义:
- $\tilde{o} \in \mathbb{R}^{\Delta \times H \times W \times 3}$: 未来RGB rollout
- $\tilde{u} \in \mathbb{R}^{\Delta \times H \times W \times 4}$: 每个pixel存一个3D坐标 + binary robot mask
- $u_t(p) = (x_p, y_p, z_p, \alpha_p)$: pixel $p$处的3D coordinate $(x_p, y_p, z_p)$加上$\alpha_p \in \{0, 1\}$指示该pixel是否在robot surface上
- $s_t$: arm-specific proprioceptive state (joint position/velocity等), 只给decoder用, **不给4DVM用** — 这正是embodiment-agnostic的来源

practical approximation: 先sample一个 $(\tilde{o}, \tilde{u})$ pair, 再从$\tilde{u}$解码$\tilde{a}$。**关键设计**: $\pi_\theta^{\text{4DVM}}$不condition on $s_t$, 所以同一个4D backbone可以跨robot重用; $\pi_\psi^{\text{DEC}}$ lightweight, 只需要少量paired data适配新embodiment。

---

## 3. 4D Video Model架构: spatially aligned modality fusion

### 3.1 为什么不用6-channel输入?

最naive的做法是把pointmap $u$当成额外channel拼到RGB后面, 形成6-channel input ($H\times W\times 6$)。但这破坏了预训练video DiT的channel structure — backbone在massive RGB数据上学到的texture prior会被新建geometric channel打乱, 难以align。

PointAction借鉴4DNeX的思路: **WidthConcat**。先让冻结VAE encoder $\mathcal{E}$独立encode RGB和XYZ:

$$
z^o = \mathcal{E}(o), \quad z^u = \mathcal{E}(u), \quad z \in \mathbb{R}^{C \times h \times w}
$$

然后沿**width dimension** concat:

$$
\tilde{z}^{\text{joint}} = \text{WidthConcat}(z^o, z^u) \in \mathbb{R}^{C \times h \times 2w} \quad (3.3)
$$

好处:
1. **保留pretrained RGB backbone的channel structure** — RGB latent通道数没变, backbone的texture prior仍然有效
2. **Spatial alignment**: 每个RGB patch的右边紧挨着它的spatially对应XYZ patch, DiT self-attention自然能建模local RGB-XYZ interaction (类似dual-stream但token-aligned)
3. **Modality embedding**: 加separate learnable vectors给RGB/XYZ做disambiguation
4. **RoPE沿width dimension重复**, 适应doubled layout

(这里有个有意思的设计选择: 把pointmap也丢进同一个VAE, 而不是用专门的pointmap VAE。VAE是在RGB上训的, 对XYZ这种geometric signal可能会过度smooth, 但作者发现dataset statistics normalization ($\mu = -0.227444, \sigma = 1.437663$)后训练稳定。这说明VAE的inductive bias对XYZ也算合理, 不一定要retrain VAE。)

参考:
- 4DNeX: https://arxiv.org/abs/2508.13154
- DiT (Peebles & Xie): https://arxiv.org/abs/2212.09748
- VAE (Stable Diffusion): https://arxiv.org/abs/2112.10752

### 3.2 Diffusion Forcing训练

follow Diffusion Forcing (Boyang Chen et al., NeurIPS 2025)的next-token + full-sequence diffusion思路: 每个temporal sequence随机split成history context $\hat{z}$ (长度$m$)和future trajectory $\tilde{z}$。两者各加独立noise level:

- Future frames: 给定noise level $\tau \in [0, 1]$和$\epsilon \sim \mathcal{N}(0, I)$:

$$
\tilde{z}_\tau = (1 - \tau)\tilde{z} + \tau \epsilon \quad (A.1)
$$

  $\tau$是flow matching的time variable, $\tau = 0$对应clean data, $\tau = 1$对应pure noise。interpolation用linear, 这是flow matching (Lipman et al.)的标准构造。

- History context: 用独立noise level $\tau'$, 50%概率设$\tau' = 0$ (clean context) — 这个trick鼓励robust conditioning, 防止模型过度依赖clean history, 提升inference时对noisy/partial context的robustness。

Flow matching的target velocity field是 $v = \epsilon - \tilde{z}$ (从data到noise的flow direction), loss:

$$
\mathcal{L}_{\text{flow}} = \mathbb{E}_{z^{\text{joint}}, \epsilon, \tau, \tau', m}\left[\left\| v_\theta(\tilde{z}_\tau, \hat{z}_{\tau'}, l, \tau) - v \right\|_2^2\right] \quad (A.2)
$$

变量:
- $v_\theta$: DiT backbone预测的flow field
- $\tilde{z}_\tau$: 加噪后的future latent
- $\hat{z}_{\tau'}$: 加噪后的history latent  
- $l$: language instruction (text condition)
- $\tau$: 当前noise level

参数高效fine-tune: 用LoRA (rank=128, $\alpha=64$, dropout=0), 只更新~209M参数, text encoder和VAE冻结。这preserves backbone的RGB generation quality同时adapt到joint RGB-XYZ target。

参考:
- Flow Matching: https://openreview.net/forum?id=PqvMRDCJT9t
- LoRA: https://arxiv.org/abs/2106.09685

### 3.3 Robot-centric point extraction

4DVM预测dense XYZ pointmap时不知道哪些pixel是robot surface (没有embodiment mask supervision)。inference时用**SAM 3** (open-vocabulary "robot" prompt)在generated RGB trajectory $\tilde{o}$上跑segmentation, 得到mask $\tilde{\alpha}$:

$$
\tilde{u}_{\text{robo}} = \tilde{u}_{xyz} \odot \tilde{\alpha}
$$

$\odot$是broadcast over 3 geometric channels。non-robot pixel被zero out, decoder只看robot部分的3D motion。这个设计很关键 — Table 3的ablation显示, **full scene point cloud反而比robot-only差** (27.1% vs 47.7% ID), 因为scene points不携带robot state信息, 注入ambiguity。

参考:
- SAM 3: https://arxiv.org/abs/2511.16719

---

## 4. Action Decoder: 3D Diffusion Policy风格

### 4.1 输入预处理

每个predicted frame的robot-centric point cloud先**Farthest Point Sampling (FPS)**下采样到$N = 512$点。FPS保证spatially均匀采样, 比random sampling更鲁棒。然后用3-layer **PointNet-style MLP** $\Phi$编码:

$$
\tilde{\mathcal{P}} = \text{FPS}(\tilde{u}_{\text{robo}}, N), \quad \text{per-frame feature} = \Phi(\tilde{\mathcal{P}})
$$

PointNet的max-pooling保证permutation invariance, 对无序point set友好。

### 4.2 Conditional DiT decoder

action decoder是个轻量DiT (6 transformer blocks, hidden dim 256, 4 heads), 用$\epsilon$-prediction (区别于4DVM用的flow matching)。给定ground-truth action sequence $a$, 在diffusion step $j$加Gaussian noise:

$$
a^{(j)} = \text{add\_noise}(a, \epsilon', j), \quad \epsilon' \sim \mathcal{N}(0, I)
$$

训练loss:

$$
\mathcal{L}_{\text{dec}} = \mathbb{E}_{a, \epsilon', j}\left[\left\| \epsilon' - \epsilon_\psi(a^{(j)}, \Phi(\tilde{\mathcal{P}}), s_t, j) \right\|_2^2\right] \quad (A.3)
$$

变量:
- $\epsilon_\psi$: DiT decoder预测的noise
- $a^{(j)}$: noised action
- $\Phi(\tilde{\mathcal{P}})$: per-frame point features作为token-aligned conditioning (concat到noisy action tokens前)
- $s_t$: initial robot state, 经AdaLN注入
- $j$: diffusion timestep, 也经AdaLN注入

**AdaLN (Adaptive Layer Norm)** 是DiT的核心conditioning机制: 用$s_t$和$j$通过MLP预测scale和shift参数$\gamma, \beta$, 然后对每个transformer block的activation做:

$$
\text{AdaLN}(h) = \gamma \cdot \text{LayerNorm}(h) + \beta
$$

这cleanly separates modality roles: point features走token pathway, global state和timestep走AdaLN pathway, 不互相干扰。

Inference用DDIM 10步采样, 一次性denoise整个49-step action chunk (parallel decoding, 不是autoregressive)。

参考:
- 3D Diffusion Policy: https://arxiv.org/abs/2403.03954
- DDIM: https://arxiv.org/abs/2010.02502
- PointNet: https://arxiv.org/abs/1612.00593

---

## 5. 数据和实验设置

### 5.1 Pretraining data

从BridgeData V2 (WidowX 250)和DROID (Franka Panda) curate ~75K trajectories:
- **DROID**: 50K clips, raw sensor depth noisy, 用**FoundationStereo**从binocular pairs重算depth
- **BridgeData V2**: 25K clips, monocular only, 用**Depth-Anything-V3**生成metric depth + pseudo-camera intrinsics

所有video统一downsample到49 frames, resize到$832 \times 480$。

### 5.2 三个evaluation regime (RoboCasa365)

- **ID (In-Distribution)**: seen task + seen environment, 100 rollouts per (method, task) cell
- **OOD-Env**: seen task + unseen environment (novel backgrounds, textures), 测visual robustness
- **OOD-Task**: unseen task + seen environment, zero-shot instruction following, 测semantic generalization

15个task (10 seen + 5 unseen), 每个task 100个teleoperated episodes做post-training。

参考:
- DROID: https://arxiv.org/abs/2403.12945
- BridgeData V2: https://arxiv.org/abs/2308.12952
- FoundationStereo: https://arxiv.org/abs/2507.07878
- Depth-Anything-V3: https://arxiv.org/abs/2511.10647
- RoboCasa365: https://arxiv.org/abs/2406.02523

---

## 6. 主要结果

### 6.1 Simulation结果 (Table 1)

| Setting | GR00T N1.7 | π0.5 | VPP | Cosmos Policy | **PointAction** |
|---|---|---|---|---|---|
| ID (10 seen) | 44.5 | 39.8 | 34.5 | 45.2 | **47.7** |
| OOD-Env | 37.6 | 35.2 | 32.2 | 42.9 | **44.1** |
| OOD-Task (5 unseen) | 8.6 | 6.9 | 7.4 | 14.0 | **17.0** |

观察:
- PointAction在所有regime都是SOTA, 比最强baseline Cosmos Policy高+2.5% (ID) / +1.2% (OOD-Env) / +3.0% (OOD-Task)
- 相比VLA baseline (GR00T N1.7, π0.5) gap更大, +4.1% / +6.5% / +8.4%~+10.1%
- **OOD-Env的degradation**: GR00T N1.7跌−6.0%, PointAction只跌−3.6% — explicit 3D interface对visual distractor更鲁棒
- **OOD-Task**: PointAction 17.0% ≈ 2× GR00T N1.7 (8.6%), 2.5× π0.5 (6.9%) — geometric interface迁移到novel task composition更容易

### 6.2 Cross-embodiment real-world (Table 2)

xArm7 (unseen during pretraining, 50 demos/task, 100 rollouts/task):

| Method | Pick&Place | Stack Cubes | Stack Cups | Avg |
|---|---|---|---|---|
| GR00T N1.7 | 30.0 | 7.0 | 7.0 | 14.7 |
| π0.5 | 42.0 | 12.0 | 14.0 | 22.7 |
| **PointAction** | **67.0** | **28.0** | **34.0** | **43.0** |

YAM arm (20 demos/task, 20 rollouts/task):

| Method | Stack Cubes | Pick Pens | Insert Cups |
|---|---|---|---|
| GR00T N1.5 | 0 | 20 | 15 |
| π0 | 0 | 10 | 15 |
| **PointAction** | **20** | **60** | **50** |

**PointAction在两个pretraining-unseen arm上, 每个task都胜过VLA baselines**。在YAM这种低数据regime (只20 demos), VLA几乎完全失败, PointAction还能做到20%~60% success。这直接验证了核心claim: **3D point dynamics是embodiment-agnostic action interface**。

---

## 7. Ablation (Table 3) — 最informative的部分

| Decoder Input | ID Success | OOD-Env Success |
|---|---|---|
| RGB Only | 25.1 | 20.3 |
| RGB + XYZ (Robot Only) | 37.2 | 30.9 |
| XYZ Only – Full Scene | 27.1 | 19.4 |
| XYZ Only – Robot + Scene | 40.3 | 33.7 |
| XYZ Only – Robot Only (DA3 source, post-hoc depth) | 28.4 | 21.7 |
| **XYZ Only – Robot Only (Ours, joint prediction)** | **47.7** | **44.1** |

关键insight:

1. **RGB Only → XYZ Only (Robot Only)**: 25.1% → 47.7%, **+22.6%**。这是最大single jump。证明geometric ambiguity是dominant failure mode, 不是别的什么次要问题。

2. **RGB + XYZ反而比XYZ only差** (37.2% vs 47.7%): task-irrelevant visual artifacts (lighting, textures)会distract lightweight decoder。说明explicit 3D geometry已经够用, 加RGB反而注入noise。

3. **Full scene XYZ (27.1%) << Robot+Scene XYZ (40.3%) < Robot-only XYZ (47.7%)**: scene points不含robot state信息, single encoder处理混合scene+robot points会注入ambiguity。两个separate encoder分开处理能recover大部分performance, 但robot-only masking最强。

4. **Post-hoc depth (DA3 on generated RGB, 28.4%) vs Joint prediction (47.7%)**: cascaded设计累积误差。生成RGB → 再estimiate depth的pipeline, depth quality受RGB quality限制, 而且depth estimator没见过generated RGB的distribution shift。joint generation让geometry和texture在denoising过程中互相inform, 输出更consistent。

这个ablationtable是整篇paper的intuition核心, 完全justifies design choices。

---

## 8. 4D Generation质量 (Table 4)

| Method | PSNR↑ | SSIM↑ | FVD↓ | AbsRel↓ | δ_1↑ | Chamfer L_1↓ |
|---|---|---|---|---|---|---|
| TesserAct | 12.225 | 0.487 | 746 | 0.403 | 0.641 | 0.389 |
| 4DNeX | 13.858 | 0.542 | 818 | 0.348 | 0.681 | 0.370 |
| LVP | 19.613 | 0.816 | 330 | – | – | – |
| Wan 2.1 14B | 14.532 | 0.674 | 671 | – | – | – |
| Ours (RGB) + StreamVGGT | – | – | – | 0.382 | 0.675 | 0.341 |
| **PointAction (joint)** | **19.631** | **0.821** | **320** | **0.176** | **0.890** | **0.122** |

指标含义:
- **PSNR** (Peak Signal-to-Noise Ratio): RGB质量, 越高越好
- **SSIM** (Structural Similarity): 结构相似度, 越高越好
- **FVD** (Fréchet Video Distance): video distribution距离, 越低越好
- **AbsRel** (Absolute Relative Error): depth误差, $\text{AbsRel} = \text{mean}(|d_{\text{pred}} - d_{\text{gt}}| / d_{\text{gt}})$, 越低越好
- **$\delta_1$**: thresholded accuracy, $|d_{\text{pred}}/d_{\text{gt}}| \in [1.25^{-1}, 1.25]$的比例, 越高越好
- **Chamfer L_1**: 两个point cloud的双向chamfer distance, $CD(P, Q) = \frac{1}{|P|}\sum_{p\in P}\min_{q\in Q}\|p-q\|_1 + \frac{1}{|Q|}\sum_{q\in Q}\min_{p\in P}\|p-q\|_1$

**关键观察**: 
- LVP (RGB-only foundation video model)在RGB metric上很好 (PSNR 19.613), 但没有geometry output
- 给LVP + StreamVGGT后处理得到geometry (cascaded variant), Chamfer只能到0.341
- PointAction joint generation在RGB metric上和LVP持平, geometry metric远超所有baseline (Chamfer 0.122 vs 0.341)

Table 7进一步验证: 在RoboCasa365的OOD场景上, PointAction joint (AbsRel 0.118, Chamfer 0.151) > MegaSAM on our RGB (0.187, 0.327) > DepthAnything-V3 on our RGB (0.198, 0.361)。说明geometry是**genuinely learned** by joint model, 不是从RGB post-hoc recoverable。

参考:
- TesserAct: https://arxiv.org/abs/2504.20995
- StreamVGGT: https://arxiv.org/abs/2507.11539
- Wan 2.1: https://arxiv.org/abs/2503.20314

---

## 9. Failure modes和limitations (Section 6, E)

两个dominant failure:

1. **Self-occlusion**: robot在generated video里occlude自己的part, robot-centric XYZ pointmap不完整, decoder收到degraded geometric input。end-effector pose partial在gripper后面时最严重。 — 这暴露了single-view的根本限制, 多view或active view selection可能是未来方向。

2. **Open-loop execution**: 49-step action chunk一次forward pass后直接执行, 不re-plan。物体slip或环境扰动无法recover。论文future work提到: distill video backbone成autoregressive + KV cache架构, 实现real-time closed-loop control。

这也指向一个更大的research方向: **slow video diffusion backbone → fast autoregressive distillation**。引用了Self Forcing (Huang et al., NeurIPS 2025)和Causal Forcing (Zhu et al., 2026)。

参考:
- Self Forcing: https://openreview.net/forum?id=mSiN7i0BYH
- Causal Forcing: https://arxiv.org/abs/2602.02214
- 从slow bidirectional到fast autoregressive (Yin et al.): https://arxiv.org/abs/2503.07878

---

## 10. 与相关工作的positioning

### 10.1 VLA vs VAM vs PointAction

- **VLA** (RT-2, OpenVLA, π0, π0.5, GR00T N1.5/N1.7): VLM backbone + action head, semantic prior强, world dynamics implicit, 需大量paired observation-action supervision
- **VAM** (VPP, Cosmos Policy, DreamVLA, WorldVLA): 预训练video diffusion作为world model, RGB-only rollout, 还是从appearance推action
- **PointAction**: VAM + explicit 4D pointmap, 把metric geometry拎出来作为action interface, **universal pretrainable + lightweight embodiment-specific decode**

### 10.2 与4D generation work的关系

- **早期4D** (4D-Fy, DreamGaussian4D, D-NeRF): optimization-based, slow, 没法做manipulation需要的real-time dynamics
- **Feed-forward 4D** (4DNeX, WorldReel, Geo4D, GeoVideo): video diffusion直接生成4D, 但target static scene / natural scene, miss manipulation dynamics
- **TesserAct**: predict RGB-DN (depth + normal) sequences for robot scenes, 用normal integration合成4D, **decoupling generation from visuomotor control** — 这是个不同的philosophy, TesserAct focus on scene reconstruction, PointAction focus on actionable 4D
- **4DGen (concurrent)**: 类似joint RGB-XYZ generation, 但需要predefined gripper CAD + continuous end-effector visibility, 限制更强

PointAction的差异化: **reconstruction-free, directly actionable, robot-centric masking**, 4D generation的output直接是action decoder的input。

参考:
- TesserAct: https://arxiv.org/abs/2504.20995
- 4DGen: https://openreview.net/forum?id=18gC6pZVVc
- 4D-Fy: https://arxiv.org/abs/2311.18484

---

## 11. 我的intuition takeaways

1. **3D是VLA/VAM的missing piece**: RGB prior已经很强 (LVP, Cosmos), 但action grounding需要metric geometry, 这个gap只能靠explicit 4D modeling close。论文的核心bet是: video diffusion backbone的capacity足够jointly model RGB + XYZ, 不需要单独geometry module。

2. **Embodiment-agnostic universal + lightweight specific**: 这个factorization思路非常简洁。universal部分用video data (便宜、大规模), specific部分用robot data (贵、小规模)。中间interface选得对 (3D points), 两者能cleanly分开。

3. **Robot-centric masking是关键trick**: 完整scene XYZ + simple decoder < robot-only XYZ + simple decoder。说明action decoder不需要理解世界, 只需要理解"robot怎么动"。把scene信息filter掉, 反而让decoder更focus。这跟VLA里"global scene understanding → action"的思路相反, 更接近motor primitive的视角。

4. **Joint > Cascaded**: 在所有level都成立 — RGB+depth joint > RGB+post-hoc depth; joint RGB-XYZ generation > RGB generation + 4D reconstruction。原因都是error accumulation + distribution shift。这个pattern在2D generation里也常见 (joint multi-modal > pipeline)。

5. **Open-loop 49-step action chunk是当前bottleneck**: 6分钟一次forward pass, 物理扰动无法recover。future work的autoregressive distillation方向是对的, 也是能fundamentally解决这个问题的路径。

6. **Pretraining-unseen arm (xArm7, YAM)的transfer results是最强的validation**: 不只是simulation数字好看, 真实新硬件 + 20-50 demos就能做到SOTA, 说明这个paradigm有实际部署价值。

---

## 12. 几个我想深挖的问题

- **WidthConcat vs Channel Concat的trade-off**: doubled token width增加attention complexity ($O((2w)^2)$vs$O(w^2)$, ~4×), 是否有更高效的spatially-aligned fusion方式? 比如cross-attention而不是self-attention?
- **VAE对XYZ的over-smoothing**: XYZ是continuous metric signal, VAE的KL regularization可能损失fine-grained geometry。试过专门train一个XYZ VAE吗? 或者直接在pixel space做XYZ diffusion?
- **SAM 3 mask的failure propagation**: 如果SAM 3 segment错robot mask, action decoder直接吃garbage input。有没有end-to-end joint learn mask的方式? 类似4DGen假设continuous end-effector visibility, 但PointAction想handle occlusion。
- **History context clean-or-noise的50/50 schedule**: 这个ratio是heuristic还是swept过? clean context过多可能overfit clean inference, 过少可能underfit conditioning signal。
- **PointNet vs 更强的point encoder**: PointNet是2017年的baseline, 现在有PointTransformer, Point-MAE, PointNeXt。lightweight decoder的bottleneck可能在point encoder, 试过更强的encoder吗?

整体上, 这篇paper的核心贡献是把4D video modeling和robot action learning clean地bridge起来, 用一个minimal modification (joint RGB-XYZ generation + robot masking + lightweight diffusion decoder)就实现了cross-embodiment transfer的SOTA。设计choices都很justified, ablation informative, real-world deployment convincing。我认为这个direction (explicit 3D as action interface)会成为VLA/VAM之后的重要branch, 特别是当video diffusion backbone capacity继续提升时。

主要参考链接汇总:
- PointAction project page: https://oriontmt.github.io/pointaction/
- Diffusion Forcing: https://arxiv.org/abs/2507.01892
- Flow Matching: https://openreview.net/forum?id=PqvMRDCJT9t
- LVP: https://arxiv.org/abs/2512.15840
- DiT: https://arxiv.org/abs/2212.09748
- LoRA: https://arxiv.org/abs/2106.09685
- DDIM: https://arxiv.org/abs/2010.02502
- PointNet: https://arxiv.org/abs/1612.00593
- 3D Diffusion Policy: https://arxiv.org/abs/2403.03954
- SAM 3: https://arxiv.org/abs/2511.16719
- FoundationStereo: https://arxiv.org/abs/2507.07878
- Depth-Anything-V3: https://arxiv.org/abs/2511.10647
- DROID: https://arxiv.org/abs/2403.12945
- BridgeData V2: https://arxiv.org/abs/2308.12952
- RoboCasa365: https://arxiv.org/abs/2406.02523
- TesserAct: https://arxiv.org/abs/2504.20995
- 4DNeX: https://arxiv.org/abs/2508.13154
- 4DGen: https://openreview.net/forum?id=18gC6pZVVc
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://openreview.net/forum?id=vlhoswksBO
- GR00T N1: https://arxiv.org/abs/2503.14734
- Cosmos Policy: https://arxiv.org/abs/2601.16163
- VPP: https://proceedings.mlr.press/v267/hu25g.html
- Wan 2.1: https://arxiv.org/abs/2503.20314
- StreamVGGT: https://arxiv.org/abs/2507.11539
- Self Forcing: https://openreview.net/forum?id=mSiN7i0BYH
- Causal Forcing: https://arxiv.org/abs/2602.02214
- OpenVLA: https://arxiv.org/abs/2406.09246
- RT-2: https://arxiv.org/abs/2307.15818
