---
source_pdf: MotionWAM- Towards Foundation World Action Models for Real-Time Humanoid
  Loco-Manipulation.pdf
paper_sha256: 7f1f828566788c1552b9a989b78e0ed525216d58218c2ac2247847e172f3c041
processed_at: '2026-08-05T20:50:13-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MotionWAM 人话版

好的 Karpathy，我换一种方式讲，就像我们坐下来喝咖啡聊这个工作，不堆公式，讲 intuition。

---

## 这篇 paper 到底想干啥

你想象一个场景：你有一台 Unitree G1 humanoid，头上装一个 camera，你想让它自己完成"从桌下拿出篮子放桌上"这种 task。

现有方法大概分两派，都有硬伤：

**第一派：VLA（Vision-Language-Action model）**。就是 GR00T-N1、π_0.5、OpenVLA 这类，本质上是把 LLM/VLM 的 backbone 接一个 action head。这些 model 见过海量 internet image-text，semantic prior 很强，能理解"把瓶子放进篮子"是什么意思。但它们从来没有真正"见过"物理世界怎么 evolve——不知道物体掉下来会怎么样、不知道接触力怎么传递。所以一到 humanoid 这种需要 whole-body coordination、需要 balance、需要 contact 的场景就崩。paper 里 π_0.5 在 humanoid loco-manipulation 上 overall success rate 不到 20%，Qwen3DiT（参数 matched 的 ablation）在 locomotion-heavy task 上直接 collapse 到接近 0。

**第二派：WAM（World Action Model）**。就是 Cosmos Policy、DiT4DiT、WorldVLA 这类。思路很漂亮：先用 video diffusion model 学会"预测未来画面"，然后把这个 future prediction 当成 prior 来 condition action。等于让 policy 先"想象一下接下来会发生什么"，再决定怎么动。这个 prior 比 image-text prior 强太多了，因为它直接 encode 了物理 dynamics、temporal coherence、contact pattern。

但 WAM 有两个 fatal issue：

第一个 issue 是 **太慢**。video diffusion model 要 denoise 几十步才能生成未来帧，每次 action inference 都要跑一遍这个 denoising loop。Cosmos Policy 在 A100 上只有 **0.7 Hz**。humanoid 的 low-level balance controller 跑 200-1000 Hz，high-level policy 至少 5 Hz 才能闭环。0.7 Hz 意味着机器人每 1.4 秒才更新一次 decision，balance 早崩了。

第二个 issue 是 **action space 分裂**。绝大多数 humanoid autonomous system 是 hierarchical 结构：high-level policy 只管上半身（手臂 joint targets），low-level controller 只接受 base velocity / torso height / orientation 这种 coarse 命令。结果是腿永远只做"保持平衡"这一件事，从来不能 task-driven 地用脚。你没法让它踩踏板、踢球、用脚推东西，因为 base velocity 这个旋钮表达不出"踢球"这种 motion。

MotionWAM 的论点就是：**这两个问题必须一起解，WAM 才能真正 deploy 到 humanoid 上**。

参考：
- GR00T-N1: https://arxiv.org/abs/2503.14734
- π_0.5: https://arxiv.org/abs/2504.16054
- Cosmos Policy: https://arxiv.org/abs/2601.16163
- DiT4DiT: https://arxiv.org/abs/2603.10448
- WorldVLA: https://arxiv.org/abs/2506.21539

---

## MotionWAM 怎么解决"太慢"的问题

这是整个 paper 最 clever 的地方。

传统 WAM 的 pipeline 是：
```
当前帧 → Video DiT → denoise 50 步 → 生成完整未来帧 → 用未来帧 condition policy → 输出 action
```
50 步 denoise 就是慢的根源。

MotionWAM 的 pipeline 是：
```
当前帧 → Video DiT → 跑 1 步 forward → 在某个 transformer block 挂个 hook，把 activation 抽出来 → 这个 activation 直接 condition Motion DiT → 输出 action
```

**Video DiT 根本不完成 denoise**。它只跑一次 forward pass，在 flow timestep τ_f ≈ 1（就是 input 基本是 pure noise 的那端）就停了，把中间层的 hidden state 抽出来用。

为什么这样能行？直觉是这样的：diffusion model 在 denoise 的早期 step，其实已经"决定"了未来要去哪里。你想想，你给一个 image diffusion model 纯噪声 + "一只猫"的 prompt，它第一步 forward 就已经在 latent 空间形成了"猫的大致构图"，后面 49 步只是在 refine 细节——毛发纹理、眼睛颜色这些。policy 需要的恰恰是那个"大致构图"级别的 dynamics plan，不需要 pixel-level 的渲染细节。

所以 MotionWAM 等于把 Video DiT 当成一个 **"imagination module"**，只取它的第一个 thought 作为 plan，跳过 render 阶段。这就是为什么它 2.5B 参数还能跑 4.9 Hz，比 2.0B 的 Cosmos Policy 快 7 倍。

这个 insight 我觉得跟你 Karpathy 一直讲的"latent space is where thinking happens"完全一致。pixel-level rendering 是 presentation layer，latent activation 才是 reasoning layer。MotionWAM 直接在 reasoning layer 截取 signal，不等你 render 完。

---

## MotionWAM 怎么解决"action space 分裂"的问题

这个问题的根源是：humanoid 有 29+ DoF，但传统 hierarchical 系统把这个 DoF 空间切成两半，上半身用 joint angles，下半身用 base velocity。这两个 representation 完全不兼容，腿的表达力被严重限制。

MotionWAM 的解法是引入一个 **unified motion latent** $\mathbf{m}_t = (\mathbf{m}_t^{\text{cont}}, \mathbf{k}_t)$：

- $\mathbf{m}_t^{\text{cont}}$：连续部分，对应 gripper / dexterous hand 的精细控制
- $\mathbf{k}_t$：SONIC token，一个 64 维的离散向量，压缩了 locomotion + torso + height + foot interaction 的 whole-body intent

SONIC（arxiv 2511.07820）是一个 pre-trained universal whole-body controller，它被训练成"给一个 latent 就能产生 coordinated whole-body motion"。MotionWAM 只需要在 latent 空间预测，不需要直接预测 29-DoF raw joint angles。

这里的 beauty 是：SONIC 在训练时见过各种 whole-body motion——走路、深蹲、踢腿、踩踏板、弯腰，所以它的 latent space 天然有一个 rich 的 motion vocabulary。MotionWAM 只需要在这个 vocabulary 里选对 token，腿就能 task-driven 地参与任务。

对比一下：
- 传统 hierarchical：腿 = balance keeper，action space = {base velocity, height, orientation}
- MotionWAM：腿 = task participant，action space = SONIC latent（覆盖踩、踢、蹲、走、弯腰...）

这就是为什么 MotionWAM 能做 Kick Soccer、Load Cart 这种 task，而 GR00T-N1.7、π_0.5 这些 hierarchical baseline 做不了。在 paper 的 Figure 5 里，Kick Soccer 上 MotionWAM 60%，GR00T-N1.7 只有 20%，差 40 个点。这 40 个点不是 policy learning 能力差，是 action space 表达不出来。

SONIC 用的 quantization 方法是 FSQ（Finite Scalar Quantization，arxiv 2309.15505），不是传统 VQ-VAE 那套 codebook + commitment loss。FSQ 直接把连续值分到固定 grid 上，训练 stability 好很多。2 tokens × 32 levels = 1024 个 whole-body motion cluster，这个 compression ratio 相当 aggressive（29-DoF × T horizon → 64-dim），能 work 说明 whole-body loco-manipulation 的 behavior mode 实际上 low-dimensional，大部分 task 是几个 canonical motion primitive 的组合。

参考：
- SONIC: https://arxiv.org/abs/2511.07820
- FSQ: https://arxiv.org/abs/2309.15505
- OmniH2O: https://arxiv.org/abs/2406.08858
- HOMIE: https://arxiv.org/abs/2502.13013

---

## 三阶段训练：为什么不能 joint train from scratch

paper 反复强调 video branch 和 motion branch 要 **specialise in turn**，不能一上来 joint train。这一点直觉上很合理。

### Stage 1：只训 Video DiT on egocentric video

用 ~2136 小时 egocentric human video + humanoid robot video，只训 video prediction，Motion DiT 还没接上。

key insight 是：**egocentric visual dynamics，不是 action diversity，才是 bottleneck**。internet video 大多是 third-person viewpoint，egocentric 的几何、occlusion pattern、self-motion cue 完全不同。先纯视频训就把这个 distribution shift 学好，不用被小量 action-labelled data 限制住。

这一步等于把 Cosmos-Predict2.5 这个 generic video model fine-tune 成一个"懂第一人称视角"的 video model。

### Stage 2：接上 Motion DiT，cross-embodiment 训练

接上 Motion DiT，用 heterogeneous Unitree G1 数据（不同 end-effector、不同 action annotation format）joint train。Loss 是 video loss + motion loss 一起。

这里有个细节：**video loss 作为 representation regulariser 保留**，防止 Motion DiT 的 noisy gradient 在早期把 Video DiT 的 dynamics prior 破坏掉。这是 dual-DiT 训练的常见 trick，DiT4DiT 也这么干。

cross-embodiment 的实现是 per-embodiment input/output projector 包住 shared trunk，action vector right-pad 到 66 维 + mask。这让不同 embodiment 的不同 action layout 能共享同一个 Motion DiT trunk。

这一步等于让 model 学会"visual dynamics → action"的 grounding，但不局限于单一 hardware config。

### Stage 3：whole-body fine-tune on Unitree G1

全网络 fine-tune，用 teleop 收集的 whole-body demonstration。每个 task 200 episodes，9 个 task，50 Hz。Teleop pipeline 是 PICO VR + 3 个 tracker → SMPL 24-joint → SONIC retarget → 29-DoF G1。

这一步把 cross-embodiment grounding 收敛到 target embodiment，action space 切换到 unified whole-body motion token。

三阶段的 ablation（Table 1）很有说服力：
- w/o Stage 2（只 Stage 1+3）：42% avg
- w/o Stage 1（只 Stage 2+3）：59% avg
- Full（1+2+3）：70% avg

去掉 Stage 2 比 Stage 1 损失更大（28% vs 11%）。说明 dynamics prior 从 video 学相对容易，但 action grounding 必须有 cross-embodiment 数据 anchor，否则 Motion DiT 直接 attach 到 Stage 1 trunk 上 fine-tune，没有 grounding 信号，action 预测就崩了。

---

## 实验结果讲什么

### 主结果（Figure 5）

9 个 real-world task，每个 20 trials：

| Method | Avg Success Rate |
|--------|------------------|
| MotionWAM | **76.1%** |
| GR00T-N1.7 | 43.9% |
| π_0.5 | < 20% |
| Qwen3DiT | collapse on locomotion |
| ACT / Diffusion Policy | fail on most |

MotionWAM 比 SOTA VLA baseline（GR00T-N1.7）高 **32 个点**。最大的 gap 出现在需要 whole-body coordination 的 task：Kick Soccer +40%, Load Cart +40%, Retrieve Item +40%, Wipe Board +45%, Do Laundry +30%。这些 task 在 hierarchical action space 下根本表达不出来。

### Qwen3DiT ablation 最关键

Qwen3DiT 是把 Cosmos-Predict2.5 Video DiT 换成 Qwen3-VL 2B（参数量 matched），其他架构、训练 pipeline 完全一样。结果：**VLM-only baseline 在 locomotion-heavy task 上 collapse**。

这说明静态 image-text prior 不够 transfer 到 closed-loop physics humanoid。Video world model 的 temporal dynamics 才是关键。这点跟你 Karpathy 在 Tesla 反复强调的"predict next token → world understanding"思想完全一致——只是这里 predict 的是 visual dynamics 而不是 text token。

### 实时性（Table 2）

| Model | Params | Frequency |
|-------|--------|-----------|
| GR00T-N1.7 | 1.6B | 6.5 Hz |
| Qwen3DiT | 2.3B | 9.0 Hz |
| Cosmos Policy | 2.0B | 0.7 Hz |
| MotionWAM | 2.5B | **4.9 Hz** |

MotionWAM 比 Cosmos Policy 快 7 倍，且参数更大。single-pass hidden state vs full denoising loop 的差距。4.9 Hz 对 humanoid high-level policy 够用。

---

## 整体直觉总结

如果用一段话 build intuition：

**MotionWAM 把 video diffusion model 当成一个"想象器"，但只取它的第一个 thought（τ_f ≈ 1 的 hidden state）作为 plan，不等你把未来帧 render 完。这个 plan 通过 cross-attention condition 一个轻量 Motion DiT，Motion DiT 输出一个 unified motion latent，这个 latent 同时覆盖 locomotion、torso、height、foot interaction、hand manipulation——腿和手在同一个 representation space 里协调。最后用 SONIC 这个 pre-trained whole-body controller 把 latent 翻译成 29-DoF joint commands。训练上分三步走：先纯视频把 Video DiT 训成懂 egocentric dynamics，再 cross-embodiment 把 Motion DiT 接上学会 action grounding，最后 fine-tune 到 G1 specific task。**

整个 pipeline 的 elegance 在于：
- Video prior 不浪费：不渲染完整 video，只取 plan-level activation
- Action space 统一：SONIC token 让腿和手在同一个 latent space 协调
- 训练渐进：video → cross-embodiment → single embodiment，每阶段只学一件事
- Inference 实时：4 步 flow matching + 1 步 video forward pass = 4.9 Hz

---

## 我的延伸思考

几个方向我觉得值得 drill：

**1. Hidden state as planning representation**。MotionWAM 在 τ_f ≈ 1 读 hidden state，等于承认 video diffusion model 的早期 transformer block 已经在做 abstract planning。这个 plan-level activation 是 System 2 thinking 的 latent 表征。后续工作可以探索：能不能不 hook 单层，而是 fuse 多层 activation？不同 flow timestep 的 hidden state 编码什么不同信息？

**2. SONIC token 的 capacity**。2 tokens × 32 levels = 1024 个 motion cluster，这个 vocabulary 是否够大？对于更复杂的 task（比如开冰箱、叠衣服这种 multi-step fine manipulation），1024 个 primitive 够不够？能不能 scale 到 4 tokens 或更多？

**3. Stage 1 数据 scale**。2136 小时 egocentric video 相比 internet-scale text-video 仍小。如果能用 Ego4D 全集（4000+ 小时）+ 更大规模 humanoid simulation video，dynamics prior 应该再上一台阶。

**4. 单 egocentric camera 的 limitation**。paper 自承物体离开 FoV 就 fail。这其实是所有 single-camera egocentric policy 的通病。解法可能是 multi-camera fusion（head + wrist + body camera），或者加 active vision 让 head 主动 track task-relevant object。

**5. Transfer 到其他 humanoid**。paper 只在 G1 上验证。三阶段 paradigm 是否 transfer 到 H1、GR1、Tesla Optimus？SONIC 是 G1 specific 的 whole-body controller，换 hardware 需要换 low-level，但 Video DiT + Motion DiT 的高层 structure 应该 portable。

**6. VLM-only 为什么 collapse**。Qwen3DiT ablation 很说明问题：静态 image-text prior 在 closed-loop physics humanoid 上不够用。原因是 image-text 只 encode "what"，不 encode "how things evolve over time"。video model 的 temporal dynamics 才是 physics reasoning 的载体。这暗示未来 VLA 路线必须 merge video dynamics，不能停在 static image-text。

**7. 为什么不直接 VLA + video auxiliary loss**。常见替代方案是 VLA 主干 + video prediction auxiliary loss。但 Qwen3DiT 间接回答了：只有当 video dynamics 是 primary pathway（conditioning action）而非 side loss 时，prior 才真正 transfer。Auxiliary loss 在 2B 参数规模下 signal 太弱，被 main action loss 淹没。MotionWAM 让 video hidden state 直接进 cross-attention，是 primary pathway。

参考：
- Ego4D: https://ego4d-data.org/
- Cosmos-Predict2.5: https://arxiv.org/abs/2511.00062
- Cosmos-Reason1: https://arxiv.org/abs/2503.15558
- Flow Matching: https://arxiv.org/abs/2210.02747
- DiT (Peebles & Xie): https://arxiv.org/abs/2212.09748
- ALOHA 2: https://arxiv.org/abs/2405.02292
- Unitree G1: https://www.unitree.com/g1
- Exbody2: https://arxiv.org/abs/2412.13196
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- ACT: https://arxiv.org/abs/2304.13705
- OpenVLA: https://arxiv.org/abs/2406.09246
- UniVAM: https://arxiv.org/abs/2503.00200
- Mimic-Video: https://arxiv.org/abs/2512.15692

---

如果还想 drill 某个模块（比如 SONIC 的 FSQ 细节、Cosmos-Predict 的 causal spatiotemporal VAE、flow matching 的 schedule 选择、或者 Stage 2 的 cross-embodiment projector 设计），可以继续问。

---

# MotionWAM 深度技术讲解

非常高兴深入聊这篇 paper，Karpathy。这篇工作在你熟悉的 model scaling + robotics 交叉点上做了一个相当 elegant 的工程取舍。我把它拆成几个层次来 build intuition。

---

## 1. 核心问题：为什么现有 WAM 跑不到 humanoid 上

先把 problem space 摊开。WAM (World Action Model) 这一类方法本质上是 **predict video dynamics, then invert for action**。代表工作比如 Cosmos Policy、UniVAM、WorldVLA、DiT4DiT。它们的 motivation 很合理：video foundation model（像 Cosmos-Predict、Sora 类）已经在 internet-scale 数据上学到了物理先验、物体交互、时空连贯性，把这些 prior 注入 policy，应该比纯 image-text pretrained 的 VLA（如 OpenVLA、π_0.5、GR00T-N1）更能处理 contact physics 和 temporal coherence。

但落到 humanoid 上有两个 fundamental bottleneck：

### Bottleneck 1: Denoising latency
WAM 通常要 iterative denoise 高维 video-action latent（典型 10~50 步 DDIM/flow steps），单次 action chunk 推理几百 ms 到几秒。Tabletop arm 的控制频率 1~3 Hz 还能接受，humanoid 闭环 balance 控制要求 50+ Hz 的 low-level，high-level policy 至少要 5 Hz 以上才能稳态。Cosmos Policy 在这个 paper 的对比里只有 **0.7 Hz**，基本没法闭环。

### Bottleneck 2: Hierarchical action space decoupling
绝大多数 humanoid autonomous system 是两层结构：
- **High-level manipulation policy**：只输出 upper body joint targets（手臂 + gripper）
- **Low-level locomotion controller**：只接受 coarse base commands（velocity, torso height, orientation）

结果是 upper body 和 lower body 生活在两个完全不同的 action space，腿被降级成"保持平衡的支架"，**永远做不了 task-driven foot interaction**（踩踏板、踢球、用脚推东西）。这一点其实在 OmniH2O、Exbody2、HOMIE 这些 whole-body controller 工作里已经隐含了，但 autonomous policy 层面没人解决。

MotionWAM 的论点就是：**这两个问题必须同时解决**，real-time WAM 才能 deploy 到 humanoid 上，且能产生 human-like 的 whole-body coordination。

参考：
- Cosmos Policy: https://arxiv.org/abs/2601.16163
- DiT4DiT: https://arxiv.org/abs/2603.10448
- OmniH2O: https://arxiv.org/abs/2406.08858
- HOMIE: https://arxiv.org/abs/2502.13013

---

## 2. MotionWAM 的核心架构直觉

### 2.1 Dual-DiT：Video DiT + Motion DiT

架构上 motionWAM 借鉴了 DiT4DiT 的 dual-DiT 思路，但做了一个关键改造。整体数据流：

```
egocentric RGB o_t  →  VAE  →  z_t^0 (clean latent)
                                 ↓
                          Video DiT (Cosmos-Predict2.5-2B)
                                 ↓ (forward hook at τ_f ≈ 1)
                          h_t^{τ_f}  (intermediate hidden state, ONE forward pass)
                                 ↓
   proprioception p_t  ──→  Motion DiT (DiT-B)  →  velocity field v_φ
                                                              ↓
                              flow matching integration → m_t (motion latent)
                                                              ↓
                                          SONIC decoder → a_t (29-DoF joint cmds)
```

关键 insight：**Video DiT 永远不完成 denoise**。它跑一次 forward pass，在 flow timestep τ_f ≈ 1（pure noise end）就通过 forward hook 把某个 transformer block 的 activation 抽出来。这就是 paper 里说的 "one-shot imagination regime"。

为什么这个能行？直觉是这样：flow matching 的 schedule 是
$$\mathbf{z}_{t+1}^{\tau_v} = (1-\tau_v)\mathbf{z}_{t+1}^{0} + \tau_v \boldsymbol{\epsilon}_v, \quad \tau_v \in [0,1]$$

这里 $\tau_v$ 是 flow timestep（$\tau_v=0$ 表示 clean，$\tau_v=1$ 表示 pure noise），$\mathbf{z}_{t+1}^0$ 是未来帧的 clean VAE latent，$\boldsymbol{\epsilon}_v \sim \mathcal{N}(0, I)$ 是 Gaussian noise。

当 $\tau_f \approx 1$，输入给 Video DiT 的几乎是纯噪声 + clean conditioning frame $\mathbf{z}_t^0$。DiT 在这一步的 hidden state 已经"决定"了未来要去哪里（conditional generation 的早期 step 本质是 plan formation），只是还没把 pixel 细节 denoise 出来。**这个 plan-like activation 就是 policy 需要的 dynamics prior**，不需要等 50 步 denoise 把未来帧渲染完整。

这就是 MotionWAM 能实时跑 4.9 Hz 的根本原因——**single forward pass of Video DiT**，对比 Cosmos Policy 的完整 denoising loop。

公式 (2) 形式化这个 hook：
$$\mathbf{h}_t^{\tau_f} = \mathcal{H}[v_\theta^{\text{video}}]\left(\mathbf{z}_{t+1}^{\tau_f}, \tau_f \mid \mathbf{z}_t^0, l\right)$$

- $\mathcal{H}[\cdot]$：forward hook operator，从一个 transformer block 读 activation
- $v_\theta^{\text{video}}$：Video DiT 的 velocity network（参数 $\theta$）
- $\mathbf{z}_{t+1}^{\tau_f}$：未来帧在 flow step $\tau_f$ 的 noise-perturbed latent
- $\mathbf{z}_t^0$：clean conditioning latent
- $l$：language goal 的 embedding（来自 Cosmos-Reason1）

### 2.2 Motion DiT 怎么吃 hidden state

Motion DiT 是 DiT-B 配置（hidden size 2560，max seq 1024），输入三路：
1. **$\mathbf{h}_t^{\tau_f}$**：来自 Video DiT 的 visual dynamics prior，通过 cross-attention 注入
2. **$p_t$**：proprioceptive state（关节角度、速度等 64 维），embed 后进 self-attention
3. **$e$**：embodiment index，per-embodiment projector 在 Stage 2 用，Stage 3 固定为 Unitree G1

输出是 velocity field $v_\phi^{\text{motion}}$，通过 flow matching 积分得到 motion latent $\mathbf{m}_t$。Inference 时只跑 4 步 denoise（Table 4 里 "Num inference timesteps: 4"），所以 Motion DiT 自己也很轻。

注意 **interleaved self/cross-attention** 这个设计——Motion DiT 不只是简单把 $\mathbf{h}_t^{\tau_f}$ 作为一个 token concat 进去，而是在 transformer block 之间交替做 self-attention（motion latent 内部 token 互动）和 cross-attention（motion latent 对 visual hidden state 查询）。这让 action representation 既能保持时序 coherent，又能被 visual dynamics 持续 modulate。

### 2.3 Unified Motion Latent：核心设计

这是 paper 最漂亮的部分。$\mathbf{m}_t = (\mathbf{m}_t^{\text{cont}}, \mathbf{k}_t)$，分两块：

- **$\mathbf{m}_t^{\text{cont}}$**：continuous channels，对应 SONIC 没覆盖的 dexterous end-effector（左右 gripper 或灵巧手）
- **$\mathbf{k}_t$**：SONIC token，64 维离散向量，来自 Finite Scalar Quantization (FSQ)，2 个 token × 32 levels = 64 dim。这个 token 压缩了 locomotion + torso + height + foot interaction 的 whole-body intent

为什么用 FSQ 而不是 VQ？FSQ 不需要 codebook 学习 + commitment loss，直接把连续值分桶到固定 grid 上，训练 stability 好很多。FSQ 原文：https://arxiv.org/abs/2309.15505

为什么用 SONIC 作为 low-level？SONIC（arxiv 2511.07820）是一个 universal whole-body controller，它本身被训练成"接受一个 latent 就能产生 coordinated whole-body motion"。它已经把"高级意图 → 29-DoF joint target"这个映射学好了，所以 MotionWAM 只需要在 latent 空间预测，不需要直接预测 raw joint angles——避免了 high-dimensional action space 的 sample inefficiency。

关键观察：**这个设计让 lower body 第一次有了 task-driven 的 action vocabulary**。传统 hierarchical 系统里腿只有 base velocity 一个旋钮，现在通过 SONIC token，腿可以"踩踏板"、"踢球"、"做深蹲"，因为 SONIC 在训练时就见过这些 whole-body motion。

公式 (6) 展示 inference pipeline：
$$\mathbf{m}_t = (\mathbf{m}_t^{\text{cont}}, \tilde{k}_t) \xrightarrow{\text{Eq. (4)}} \hat{\mathbf{m}}_t = (\hat{\mathbf{m}}_t^{\text{cont}}, \hat{\tilde{k}}_t) \xrightarrow{\hat{k}_t = \text{round}(\tilde{k}_t)} (\hat{\mathbf{m}}_t^{\text{cont}}, \hat{k}_t) \xrightarrow{\text{SONIC}} \mathbf{a}_t$$

注意 $\tilde{k}_t \in \mathbb{R}$ 是一个 scalar slot（不是 categorical head），整个 $\mathbf{m}_t$ 一起走 flow matching regression，inference 时再 round 成离散 SONIC index $k_t \in \{0, \ldots, K-1\}$。这个 trick 避免了离散 categorical head 训练时的 straight-through estimator 复杂性，又保留了 SONIC codebook 的结构。

---

## 3. 三阶段训练 recipe：渐进式 specialization

paper 反复强调 video branch 和 motion branch 要 **specialise in turn** 而不是 jointly from scratch。这点很重要，直觉上：你不想让 Motion DiT 早期的 noisy gradient 把 Video DiT 的 dynamics prior 破坏掉。

### Stage 1: Egocentric Video Pretraining
- 只训 Video DiT，loss 只有 $\mathcal{L}_{\text{video}}$
- 数据：~2136 小时 egocentric human + humanoid video（详见表 5）
- Mixture: human 30%, G1-class humanoid 50%, other robots 20%
- 目的：把 Cosmos-Predict2.5 的 generic video distribution shift 到 **first-person viewpoint dynamics**

这里 paper 给了一个反直觉的 insight：**egocentric visual dynamics，不是 action diversity，才是 bottleneck**。internet video 大多是 third-person，egocentric 的几何、occlusion pattern、self-motion cue 完全不同。先纯视频训就把这个 shift 学好，不用被小量 action-labelled data throttle。

### Stage 2: Cross-Embodiment Action Post-Training
- 接上 Motion DiT，joint loss = $\mathcal{L}_{\text{motion}} + \mathcal{L}_{\text{video}}$（公式 5）
- Video loss 作为 **representation regulariser** 防止 dynamics prior 被 overwrite
- Heterogeneous Unitree G1 data，不同 end-effector 和 action annotation format
- Per-embodiment input/output projector 包住 shared trunk，action vector right-pad 到 66 维 + mask

公式 (3) Video loss：
$$\mathcal{L}_{\text{video}} = \mathbb{E}_{\tau_v, \mathbf{z}_{t+1}^0, \epsilon_v}\left[\left\|v_\theta^{\text{video}}(\mathbf{z}_{t+1}^{\tau_v}, \tau_v \mid \mathbf{z}_t^0, l) - (\epsilon_v - \mathbf{z}_{t+1}^0)\right\|_2^2\right]$$

- $v_\theta^{\text{video}}$：Video DiT 预测的 velocity field
- Target 是 $\epsilon_v - \mathbf{z}_{t+1}^0$：这是 flow matching 的标准 target，对应从 $\tau_v$ 处的 tangent vector 指向 noise 方向
- $\tau_v$ 从 uniform $[0,1]$ 采样，确保整个 flow trajectory 都被覆盖

公式 (4) Motion loss：
$$\mathcal{L}_{\text{motion}} = \mathbb{E}_{\tau_a, \mathbf{m}_t^0, \epsilon_m}\left[\left\|v_\phi^{\text{motion}}(\mathbf{m}_t^{\tau_a}, \tau_a \mid \mathbf{h}_t^{\tau_f}, p_t, e) - (\epsilon_m - \mathbf{m}_t^0)\right\|_2^2\right]$$

- $\tau_a$：action flow timestep（区别于 $\tau_v$ video flow timestep）
- $\mathbf{m}_t^0$：clean motion latent（来自 teleop demonstration）
- $\mathbf{m}_t^{\tau_a} = (1-\tau_a)\mathbf{m}_t^0 + \tau_a \epsilon_m$：noise-perturbed motion latent
- $e$：embodiment index，让 shared trunk 能区分不同机器人
- 同样 target 是 $\epsilon_m - \mathbf{m}_t^0$

两个 loss 共享同一类 flow-matching 数学结构，只是 conditioning 不同。这让 Stage 2 的 joint training 在数学上很干净。

### Stage 3: Whole-Body Fine-Tuning
- 全网络 fine-tune，loss 仍是公式 (5)
- 数据：每个 task 200 episodes，9 个 task，50 Hz，Unitree G1，通过 VR teleop
- Teleop pipeline：PICO VR + 3 个 tracker → SMPL 24-joint → SONIC retarget → 29-DoF G1
- Action space 切换到 unified whole-body motion token

这里 stage 2 的 cross-embodiment projector 被 swap 成 G1 专属 projector，但 trunk 保留。这让 stage 2 学到的"action grounding"能 transfer。

---

## 4. 实验：数据说话

### 4.1 主结果（Figure 5）

9 个 real-world task，每个 20 trials：

| Method | Avg Success Rate |
|--------|------------------|
| MotionWAM | **76.1%** |
| GR00T-N1.7 | 43.9% |
| π_0.5 | < 20% |
| Qwen3DiT (matched capacity) | collapse on locomotion tasks |
| ACT, Diffusion Policy | fail on most loco-manipulation |

最大的 gap 出现在需要 whole-body coordination 的 task：Kick Soccer (+40%), Load Cart (+40%), Retrieve Item (+40%), Wipe Board (+45%), Do Laundry (+30%)。这些 task 在 hierarchical action space 下根本表达不出来——你没法用 base velocity 命令"踢球入网"。

### 4.2 Qwen3DiT ablation 的关键性

Qwen3DiT 是 paper 里最 important 的 ablation：把 Cosmos-Predict2.5 Video DiT 换成 Qwen3-VL 2B（参数量 matched），其他架构、训练 pipeline 完全一样。结果：**VLM-only baseline 在 locomotion-heavy task 上 collapse**。

这说明 **静态 image-text prior 不足以 transfer 到 closed-loop physics humanoid**。Video world model 的 temporal dynamics 才是关键。这点和你 Karpathy 之前在 Tesla 讲过的"predict next token → world understanding"思想高度吻合，只是这里预测的是 visual dynamics 而不是 text token。

### 4.3 三阶段 ablation（Table 1）

| Variant | Avg Success |
|---------|-------------|
| w/o Stage 2 (only Stage 1+3) | 42.0% |
| w/o Stage 1 (only Stage 2+3) | 59.0% |
| Full (1+2+3) | 70.0% |

去掉 Stage 2（cross-embodiment grounding）损失 28%，比去掉 Stage 1 还严重。这印证了：dynamics prior 容易从 video 学，但 action grounding 必须有 cross-embodiment 数据 anchor。

### 4.4 实时性（Table 2）

| Model | Params | Frequency |
|-------|--------|-----------|
| GR00T-N1.7 | 1.6B | 6.5 Hz |
| Qwen3DiT | 2.3B | 9.0 Hz |
| Cosmos Policy | 2.0B | 0.7 Hz |
| MotionWAM | 2.5B | **4.9 Hz** |

MotionWAM 比 Cosmos Policy 快 **7 倍**，且参数更大。关键就是 single-pass hidden state 而非 full denoising。4.9 Hz 对 humanoid high-level policy 够用了（low-level balance controller 是 200-1000 Hz，high-level 5 Hz 是常见 spec）。

---

## 5. 我的延伸思考

### 5.1 Hidden state as planning representation
MotionWAM 在 τ_f ≈ 1 读 hidden state，本质上是把 Video DiT 当成一个 **"imagination module"**——给定当前帧 + language，DiT 在第一个 step 就形成一个 latent plan，后续 denoise 只是把这个 plan render 成 pixel。这个 plan 直接 condition action，跳过 render。

这跟你 Karpathy 反复提的 "system 2 thinking via latent" 思路完全一致。video diffusion model 的早期 transformer block 已经在做某种 abstract planning，MotionWAM 只是把这个 plan 显式 extract 出来用。

### 5.2 为什么不直接训 VLA + video auxiliary loss
常见替代方案是 VLA 主干 + video prediction auxiliary loss（让 VLA 同时预测未来帧）。但 paper 的 Qwen3DiT ablation 间接回答了：**只有当 video dynamics 是 primary pathway（conditioning action）而非 side loss 时，prior 才真正 transfer**。Auxiliary loss 在 2B 参数规模下太弱，signal 被 main action loss 淹没。

### 5.3 SONICFSQ 的 bottleneck 设计
SONIC 用 2 tokens × 32 levels = 1024 个 whole-body motion cluster。这是个相当 aggressive 的 compression（29-DoF × T horizon → 64-dim token）。能 work 说明 whole-body loco-manipulation 的行为模式实际上 low-dimensional——大部分任务是几个 canonical motion primitive 的组合。这跟你在 "YouTube-down task" 之类的论点呼应：robot behavior 的 intrinsic dimension 远低于 joint space 维度。

### 5.4 局限与未来方向
paper 自承两个 limitation：
1. **只在 G1 上验证**：三阶段 paradigm 是否 transfer 到其他 humanoid（如 H1、GR1、Tesla Optimus）未知
2. **没有 novel-object generalization study**：train/test object 视觉相似
3. **单 egocentric camera**：物体离开 FoV 就 fail（Figure 6 failure case）

我认为还有一个 implicit limitation：**Stage 1 用 2136 小时 egocentric video**，这个数据规模虽然比 action-labelled data 大，但相比 internet-scale text-video 仍小。如果能用上 Ego4D 全集（4000+ 小时）+ 更大规模 humanoid simulation video，dynamics prior 应该还能再上一台阶。

参考：
- Ego4D: https://ego4d-data.org/
- SONIC: https://arxiv.org/abs/2511.07820
- Cosmos-Predict2.5: https://arxiv.org/abs/2511.00062
- DiT (Peebles & Xie): https://arxiv.org/abs/2212.09748
- Flow Matching: https://arxiv.org/abs/2210.02747

---

## 6. 与相关工作的 positioning

| 工作 | 类型 | Real-time? | Whole-body? | 关键差异 |
|------|------|-----------|-------------|---------|
| Cosmos Policy | WAM | 0.7 Hz ❌ | Tabletop only | Full denoising，慢 |
| DiT4DiT | WAM | 中 | Tabletop | MotionWAM 的 dual-DiT 灵感来源 |
| WorldVLA | WAM | 慢 | Tabletop | Autoregressive action world model |
| UniVAM | WAM | 慢 | Tabletop | Unified video action latent |
| GR00T-N1.7 | VLA | 6.5 Hz | Whole-body | 静态 VLM prior，无 video dynamics |
| π_0.5 | VLA | < 5 Hz | Whole-body | Open-world generalization 强，但 humanoid loco-manipulation 弱 |
| OmniH2O | Teleop | N/A | Whole-body | 不是 autonomous policy |
| HOMIE | Teleop+IL | N/A | Whole-body | Exoskeleton teleop，no WAM |
| **MotionWAM** | **WAM** | **4.9 Hz ✅** | **Whole-body ✅** | **First real-time WAM on humanoid** |

---

## 7. 总结直觉

如果用一句话 build intuition：**MotionWAM 把 video diffusion model 当成一个"想象器"，但只取它的第一个 thought（τ_f ≈ 1 的 hidden state）作为 plan，再用一个轻量 Motion DiT 把这个 plan 翻译成 unified whole-body action token，最后用 SONIC 这个 pre-trained low-level controller 把 action token 渲染成 29-DoF joint commands**。

整个 pipeline 的 elegance 在于：
- **Video prior 不浪费**：不渲染完整 video，只取 plan-level activation
- **Action space 统一**：SONIC token 让腿和手在同一个 latent space 协调
- **训练渐进**：video → cross-embodiment → single embodiment，每一阶段只学一件它最擅长的事
- **Inference 实时**：4 步 flow matching + 1 步 video forward pass = 4.9 Hz

这个工作在我看来是 robotics foundation model 路线上的一个重要 data point：**证明了 video-pretrained world model 可以从 tabletop extend 到 dynamically balancing humanoid，且实时**。下一步值得关注的方向是把 Stage 1 数据 scale 上去（Ego4D 全量 + synthetic humanoid simulation video），以及把 SONIC 换成更高容量的 whole-body controller（更细的 motion vocabulary）。

---

参考链接汇总：
- MotionWAM（本论文）: https://arxiv.org/abs/2511.07820 (基于文中引用的 SONIC ID 推测)
- Cosmos-Predict2.5: https://arxiv.org/abs/2511.00062
- Cosmos-Reason1: https://arxiv.org/abs/2503.15558
- SONIC: https://arxiv.org/abs/2511.07820
- DiT4DiT: https://arxiv.org/abs/2603.10448
- GR00T-N1: https://arxiv.org/abs/2503.14734
- π_0.5: https://arxiv.org/abs/2504.16054
- FSQ: https://arxiv.org/abs/2309.15505
- DiT: https://arxiv.org/abs/2212.09748
- Flow Matching: https://arxiv.org/abs/2210.02747
- Ego4D: https://ego4d-data.org/
- Unitree G1: https://www.unitree.com/g1
- OmniH2O: https://arxiv.org/abs/2406.08858
- HOMIE: https://arxiv.org/abs/2502.13013
- Exbody2: https://arxiv.org/abs/2412.13196
- ALOHA 2: https://arxiv.org/abs/2405.02292

如果你想 drill deeper 到任何一个模块（比如 SONIC 的 FSQ 细节、Cosmos-Predict 的 causal spatiotemporal VAE、或者 flow matching 的 schedule 选择），我可以再展开讲。
