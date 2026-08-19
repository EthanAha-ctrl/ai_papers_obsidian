---
source_pdf: GE-Sim 2.0 A Roadmap Towards Comprehensive Closed-loop.pdf
paper_sha256: af875445ebe487962e308bbff5e76e19fdcfc57620a50ec6ec9070989cbfe641
processed_at: '2026-08-19T08:52:41-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 GE-Sim 2.0

## 一句话

这篇 paper 干的事：**让一个会生成 robot 视频的模型，变成一个 robot policy 真能进去"玩"的模拟器。**

---

## 背景痛点

现在 robot learning 这条路，**训练跑得飞快，评测跟不上**。

π0、π0.5、RT-2、OpenVLA 这些 policy 越来越猛，但你怎么知道它到底好不好？

- 真机测：慢、贵、不可复现。每次改个参数都要重新部署 robot、跑 episode、人盯结果。
- MuJoCo、Isaac Gym 这种老 simulator：刚性物体还行，一碰到 **可变形物体（毛巾、水）、接触动力学（插拔）、火焰、液体**，物理就不准了。连 robot 自己的 actuation（harmonic drive 那种 compliance）都抽象掉了。

**训练和评测的 gap 越拉越大**。你 train 出一个 policy，根本没法 scalable 地知道它好不好。

---

## 新思路：用 video 生成模型当 simulator

最近 video 生成模型（Sora、Veo）火了。一个自然想法：**让生成模型来当 world simulator**。给它一张初始图 + 一段 action，让它 rollout 出 robot 执行这段 action 的视频。

这比 hand-crafted simulator 强在：**数据驱动**。你喂它多少真实 robot 视频，它就能学到多少真实外观和交互。那些手工 simulator 做不出来的可变形物体、接触细节、视觉外观，生成模型直接从数据里学。

GE-Sim 1.0 就做了这个事的雏形——把 text-to-video 模型改造成 **action-conditioned video generator**。给它 action trajectory，它生成对应视频。

---

## GE-Sim 1.0 不够用的三个原因

paper 在 introduction 里讲得特别清楚。光会生成视频，**不等于能当 simulator**。三个 gap：

**Gap 1：只生成视频，没给 state**

现代 VLA policy（π0.5 这种）的输入是 **(image, proprioceptive state)**——图像 + 关节角。光有视频没用，policy 还要知道 robot 的 joint angle 和 gripper state。

GE-Sim 1.0 只生成视频。closed-loop 时怎么办？用 commanded action 当 state 的 proxy。问题是：commanded action 和真实 joint state 有差距，特别是 contact task 下 robot 会被物体"顶回来"，commanded 和实际 joint 偏离越来越大。policy 拿着错误的 state 进去，下一帧 action 就走偏，走偏的 action 又让视频生成更怪——**滚雪球**。

**Gap 2：只 render，不 judge**

生成完一段 rollout，你还得知道这段 rollout 成功还是失败。之前没有自动 judge，要人来看视频打分。**完全没法 scale**。

也没法做 RL——RL 需要 reward signal。

**Gap 3：太慢**

多步 diffusion 推理，一段 rollout 要几十秒。如果要 chunk-wise 闭环跑 50 秒 horizon，还要多任务多 seed 并行，**吞吐完全不够**。

---

## GE-Sim 2.0 的三个补丁

三个 gap 三个补丁，每个补丁都设计得很合理。

### 补丁 1：State Expert

**干啥**：从视频 latent 里 decode 出真实 joint state。

**怎么干**：在 vision expert 旁边挂一个 lightweight transformer branch。vision expert 先跑完一遍，把每一层的 feature 用 learnable weight fuse 成一个 fused feature。state expert 用 cross-attention 看这个 fused feature，同时自己也跑 diffusion（state 也用 flow-matching 训），输出 16 维 state（左右臂各 7 维 joint + 1 维 gripper）。

**为什么独立一个 branch**：vision expert 是几千维 hidden、100+ 层的大 DiT，让它直接 regress 16 维 state 容易被高频视频信号淹没。分开来，让 state expert 专注 low-dim regression。

**训练 trick**：history state 在训练时故意加扰动——先 downsample 再 upsample（低通失真），还加随机 index shift。模拟推理时 history state 来自上一 chunk 预测、有误差的情况。**对抗 exposure bias**。

### 补丁 2：World Judge

**干啥**：给 rollout 的每一帧打 success/failure 分。

**怎么干**：VLM backbone，freeze vision encoder，只训 language model + prediction head。每帧单独送入，每帧后 append 一个 per-frame token，取其 hidden state 当这帧的 representation。接一个 MLP head 输出 success logit。

**训练 label 怎么来**：真机 trajectory 有人工标注的 success frame $i_{\mathrm{succ}}$，这帧及之后 = 1，之前 = 0。用 class-balanced BCE。

**关键设计选择**：**只做 success，不做 progress**。因为真机数据里太多 error recovery（绕路、retry、回退），progress label 在这种 non-monotonic trajectory 上 noisy 得没法用。Sparse terminal success 反而稳。

**和 general VLM 对比**：拿 122B 的 Qwen3.5 当 reward model，prompt 它判断 success。结果 GE-Sim 2.0 的 specialist judge acc 79%，Qwen 60%。**specialist 小模型完胜 general 大模型**。

特别是 **Clean mirror stains** 这种 task——成功标准是镜面 appearance 的细微变化，Qwen 完全 fail，连 success frame 都识别不出来。Specialist 82%。

**为什么**：general VLM 在 web data 上训的，没见过"火焰传递"、"水倒入杯"、"镜面污渍"这种 fine-grained physical success。Specialist 在真机 success annotation 上训过，专门识别这些信号。

### 补丁 3：Acceleration

两个方向同时压缩。

**方向 A：Step distillation (DMD2)**

把多步 diffusion 蒸馏成 4 步。三个网络：frozen teacher、trainable student、trainable critic。Student 生成样本，teacher 和 critic 分别给样本分布的 score，差就是 distribution-matching gradient。GAN-like 互搏。

**方向 B：Random-stride training**

训练时每个 chunk 的 frame 用 random temporal stride 采样，让模型见过不同 temporal density。推理时可以做 4× frame skipping——同样 25 帧覆盖 4 倍时间跨度。

**组合效果**：4 步 diffusion + 4× frame skipping = 25 帧 rollout 在单卡 H100 上 2.3 秒。够用来 chunk-wise 并行大规模 eval。

---

## 实验结果

### Q1：视觉保真度

**WorldArena leaderboard**：2B 参数拿第一，打赢 Sora、Veo、Ctrl-World、DreamDojo、GigaWorld、ABot、MotuBrain。**说明 domain-specific + action-grounded 训练比 raw model scale 重要**。

**Per-task replay**：PSNR +4 dB over Ctrl-World，FID 砍半，FVD 砍 2 倍。

**Long-horizon 稳定性**：把 50s rollout 切 5 段看每段 PSNR。GE-Sim 2.0 从 24.8 → 21.1 dB（掉不到 4 dB），Ctrl-World 和 DreamDojo 第一段后崩到 16 dB。**差距随 horizon 增大而扩大**——这正是 closed-loop simulator 需要的稳定性。

### Q2：Closed-loop 一致性

这个才是核心。Replay 好 ≠ closed-loop 好。

**Task-level**：六个 task，每个 20 个 π0.5 episode，比较真机和 simulator 的 success rate。GE-Sim 2.0 with state 的拟合 slope ~1，最好保住真机上的 relative task difficulty。

**Episode-level**：accuracy 0.81，recall 0.82。**Recall 是关键**——simulator 能把真机 success 正确识别成 success，不会把成功判成失败。没有 state expert 时 recall 只有 0.67，有 state expert 救回大量 true positive。

### Q3：Reward 质量

GE-Sim 2.0 specialist judge：WM mode 79% acc，GT mode 87% acc。
Qwen3.5-122B：60% / 58%。

GT 比 WM 高 8 pp，反映 simulator artifact 引入的 reward 误差。

### Q4：Filtered BC 真机提升

最 actionable 的结果。

在 GE-Sim 2.0 里跑 π0.5 → world judge 打分 → 保留 high-reward trajectory → 混入原 BC 数据 → retrain policy。

三个 task 真机 success rate：

- Pour water: 0.40 → 0.55 (+15pp)
- Fold towels: 0.40 → 0.50 (+10pp)
- Pull out plug: 0.45 → 0.65 (+20pp)
- 平均 +15pp

**没用 online RL，光是用 world model + reward 当数据生成 + 筛选 engine，就有 15pp 真机提升**。本质是把 BC 数据集从纯人工示教扩到人工 + 合成高质量数据。

---

## 我的几个直觉判断

**1. "Video as world simulator" 这条路在 robot manipulation 上被验证了**

Sora 那种 general video world model 是 floor。要真正给 robot policy 用，必须补 state + reward + throughput 三件套。GE-Sim 2.0 把这三件事做齐了。

**2. Specialist 小模型 >> General 大模型（在 reward 这件事上）**

122B Qwen 打不过 specialist。这和 FSD 那边 "task-specific reward" 的经验一致。General VLM 在 web data 上训的，对 fine-grained physical success 不敏感。要训 specialist。

**3. State expert 的必要性**

没它，policy 拿 commanded action 当 state proxy，contact task 下 drift 越来越大，滚雪球。有它，切断滚雪球。**这是从 video generator 走向 closed-loop simulator 的关键一步**。

**4. Memory-frame augmentation 是 video 版的 scheduled sampling**

经典 exposure bias 问题。训练时 memory frame 是 clean GT，推理时是 model 自己生成的，有 artifact。解法是训练时加 corruption 模拟推理时的分布。简单但有效。

**5. Random-stride training 是 event-based simulator 的雏形**

让 model 见过不同 temporal density，推理时可变 stride。本质是把 simulator 从 fixed-frame-rate 解放出来。未来可以做成 event-based camera 那样，重要时刻密集采样，不重要时段稀疏采样。

**6. Filtered BC +15pp 已经把 path 跑通**

Online RL 是 next。Paper 已经把 closed-loop 所有接口做齐——reward、state、throughput 都 ready。下篇 paper 应该是 PPO/SAC inside GE-Sim。

---

## 还没解决的

- **Cross-embodiment**：只训了 dual-arm 同一 embodiment。要扩到 UMI、ego-view、不同 robot。
- **Liquid / flame / fine appearance**：Pour water、Borrow flame 的 success detection 还弱。
- **Contact dynamics 极限**：Pull out plug、Command grasp 还有 false positive/negative。
- **Unified model**：state expert 和 world judge 现在是独立 module，long term 应该 unified，让 reward 从 world model forward pass 直接出来。

---

## 一句话总结

GE-Sim 2.0 把"生成 robot 视频"这个能力，补上 **state 预测 + reward 判定 + 推理加速** 三件套，变成 robot policy 真能进去 closed-loop 跑的 simulator。2B 模型打赢 WorldArena，closed-loop success agreement 81%，filtered BC 真机 +15pp。**这是 robot manipulation simulator 的下一步范式**。

项目主页：https://ge-sim-v2.github.io/

---

# GE-Sim 2.0 深度讲解

## 1. 一句话定位

这是一篇从 "**action-conditioned video generator**" 走向 "**closed-loop world simulator for manipulation**" 的方法学 paper。它继承了 Genie Envisioner（GE-Sim 1.0）的 action-conditioned video generation backbone，补上了三个闭环的关键模块——state expert、world judge、acceleration——使得一个 2B 的小模型在 WorldArena 上打赢了 Sora、Veo、Ctrl-World、DreamDojo 这些更大或更专的对手。

核心 insight：**光会生成视觉上逼真的 robot 视频，不等于能当 simulator 用**。Simulator 的本质是 policy 可以在里面 closed-loop rollout，而 closed-loop rollout 需要：(i) proprioceptive state 反馈；(ii) success/failure reward；(iii) 足够高的吞吐。这三点恰好是之前所有 video world model 缺失的。

项目主页：https://ge-sim-v2.github.io/

相关参考：
- GE-Sim 1.0 (Genie Envisioner): https://arxiv.org/abs/2508.05635
- Cosmos world foundation model: https://arxiv.org/abs/2501.03575
- WorldArena benchmark: https://arxiv.org/abs/2602.08971
- π0.5 VLA policy: https://arxiv.org/abs/2504.16054
- DMD2 (step distillation): https://arxiv.org/abs/2505.14681
- Sora 作为 world simulator 的 OpenAI position: https://openai.com/research/video-generation-models-as-world-simulators
- DreamerV3 (model-based RL 经典): https://arxiv.org/abs/2304.12893
- DIAMOND (diffusion world model for RL): https://arxiv.org/abs/2405.16792

---

## 2. Motivation：为什么 robot learning 需要 video world simulator

Karpathy 你自己反复强调过 "evaluation is the bottleneck"。这篇 paper 的开篇完全踩在这个痛点上：

- VLA policy (π0, RT-2, Octo, π0.5) scaling 越来越猛；
- 但 real-robot benchmark 慢、不可复现；
- MuJoCo / Isaac Gym / ManiSkill / RLBench / CALVIN / LIBERO / SAPIEN / SoftGym 这些 hand-crafted simulator 在 contact dynamics、deformable object、视觉外观、robot 自身 actuation（harmonic-drive compliance 这种）上都不准；
- 结果是 **"can train" 远远跑在 "can evaluate" 前面**。

视频生成模型的崛起给了一个新方向：**用 data-driven 的 generative process 替换 hand-built physics + rendering**。给一张初始 observation + 一段 action trajectory，让模型 rollout 出 robot 执行这段 trajectory 的视频。这就是 "neural world simulator for manipulation"。

关键洞察：传统 world model（Dreamer 系）把 world 建立在 latent state 上；这里把 world 建立在 **video pixel latent** 上，这是和 Sora "video as world simulator" 路线一致的选择，但 GE-Sim 2.0 把它落到 robot manipulation 这种需要 fine-grained action grounding 的领域。

---

## 3. Preliminaries：GE-Base 和 GE-Sim 1.0 的骨架

### 3.1 GE-Base：multi-view text-image-to-video 世界基础模型

GE-Base 把 robot world modeling 当作 multi-view TI2V 生成问题。三个 onboard camera：

$$\mathcal{V} = \{h, l, r\}$$

分别是 head、left wrist、right wrist。$x_t^i$ 是 view $i$ 在时间 $t$ 的 frame。

**Autoregressive chunk-wise generation（式 1）**：

$$\mathbf{x}_{1:N}^t = \mathcal{W}(\mathbf{x}_0,\ \mathbf{m}_{0:t-1},\ \mathcal{T}(q))$$

变量含义：
- $\mathbf{x}_{1:N}^t$：第 $t$ 个 autoregressive step 要预测的 $N$ 帧 multi-view video chunk；
- $\mathbf{x}_0$：初始 observation；
- $\mathbf{m}_{0:t-1}$：sparse memory，从历史 chunk $\{\mathbf{x}_{1:N}^k\}_{k=0}^{t-1}$ 中稀疏采样关键帧组成，让 long-horizon context 进入输入但不爆炸；
- $\mathcal{T}(q)$：frozen T5 text encoder 对 instruction $q$ 编码。

直觉：这就是把 LM 的 KV cache 换成了 "sparse keyframe memory"，每 rollout 一段就稀疏挑几帧塞回 context，以避免 autoregressive 视频生成中经典的 "context grows linearly" 问题。和 Sora 的 spatiotemporal patch + recaption + 无 memory 不同，这里选择了显式 memory。

**Multi-view encoding（式 2）**：

$$\tilde{v}^i = \mathrm{RoPE}(t, h, w) + v^i + e_{\mathrm{view}}^i$$

变量：
- $v^i = \mathcal{E}(x^i)$：每个 view 独立过一个 shared video encoder $\mathcal{E}$；
- $\mathrm{RoPE}(t,h,w)$：3D rotary positional embedding，编码 time + height + width；
- $e_{\mathrm{view}}^i$：learnable view embedding，告诉模型 "这是 head cam 还是 wrist cam"。

每个 view 的 token 序列是 $u^i = [\tilde{v}_0^i \,\|\, \tilde{v}_m^i \,\|\, z^i]$，其中 $z^i$ 是 view-specific noise map。三个 view 的 token 拼起来送进 video diffusion transformer (DiT)。部分 DiT block 做 cross-view attention 保 multi-view consistency，剩下 block 独立处理单 view，节省 compute。

**Backbone 和训练（式 3）**：

主网络是 Cosmos-Predict2-2B-Video2World DiT。训练目标 latent flow-matching：

$$\mathcal{L}_{\mathrm{video}} = w(\tau) \left\| (v_\theta - (\epsilon - l)) \odot (1 - M) \right\|_2^2$$

变量：
- $l$：target chunk 的 VAE latent；
- $\tilde{l} = (1-\sigma_\tau) l + \sigma_\tau \epsilon$：noisy latent，$\epsilon \sim \mathcal{N}(0, I)$；
- $v_\theta$：模型预测的 denoising velocity（flow-matching 的 target 是 velocity $(\epsilon - l)$，不是 noise prediction）；
- $M$：conditioning mask，只在未来 frame 上算 loss，memory frame 被 mask 掉；
- $w(\tau)$：per-timestep 的 loss weight。

直觉：flow-matching 比 DDPM 训练更稳，velocity parameterization 在 video latent 上收敛更好。用 conditioning mask $M$ 隔开 memory 和 prediction，让模型不被 "复制历史" 这个 shortcut 吸引走。

### 3.2 GE-Sim 1.0：从 TI2V 到 action-conditioned simulator

GE-Sim 1.0 把 GE-Base 从 "text-conditioned" 切到 "action-conditioned"。

**Action representation（式 4）**：

$$a_i = \big[\underbrace{x_i, y_i, z_i, r_i, p_i, y_i, o_i}_{\mathrm{left\ arm}},\ \underbrace{x_i, y_i, z_i, r_i, p_i, y_i, o_i}_{\mathrm{right\ arm}}\big] \in \mathbb{R}^{14}$$

每个 control step 14 维：双臂各 7 维，前 6 维是 end-effector 的 position $(x,y,z)$ + RPY orientation $(r,p,y)$，第 7 维 $o$ 是 gripper openness。$K$-step trajectory $\mathbf{A} = [a_1,\dots,a_K] \in \mathbb{R}^{K\times 14}$。

**Spatial action conditioning（关键设计）**：直接把 14-D vector 通过 MLP 注入会丢掉 spatial alignment。GE-Sim 设计了两个 pixel-aligned 信号：

1. **Pose image $P_i \in \mathbb{R}^{3\times H\times W}$**：把 EE position 投影到像素平面，orientation 画成 3 个有颜色的 directional unit vector（RGB 分别对应 XYZ 轴），gripper openness 渲染成单位圆的 shading。左臂右臂用不同色族区分。

2. **Camera raymap $R_i \in \mathbb{R}^{6\times H\times W}$**：每个 pixel 反投影出一条 ray，由 origin $\mathbf{o}_i \in \mathbb{R}^3$（相机中心在世界系的位置）+ unit direction $\mathbf{d}_i \in \mathbb{R}^3$（pixel $(u,v)$ 反投影并 rotate 到 world）组成，stack 成 6 通道。

直觉：raymap 的存在是为了解决一个 robot-specific 难点——**head 和 wrist 相机本身是 mount 在 robot 上的，会随 robot 一起动**。特别是 wrist cam：EE 在 wrist 视野里几乎是静止的，EE 的运动信号几乎全在 "相机怎么动" 里，而 raymap 把这个 explicit 给出来，避免模型从 appearance 里 implicit 推断相机位姿。

**Latent fusion（式 5）**：

$$v_i = [\tilde{z}_i \,\|\, \mathrm{down}(P_i) \,\|\, \mathrm{down}(R_i)]$$

把 noisy video latent、pose image、raymap 都 bilinear downsample 到 latent 分辨率，沿 channel 拼起来。这样在 DiT 的每一层，action 信号、camera 几何、video latent 都 spatially aligned。

**最终的 simulator 形式（式 6）**：

$$\mathbf{x}_{1:N}^t = \mathcal{S}(\mathbf{x}_0,\ \mathbf{m}_{0:t-1},\ \mathbf{A}^t)$$

$\mathbf{A}^t$ 是和 chunk $t$ 时间对齐的 action sub-trajectory。$\mathcal{S}$ 就是把 $\mathcal{W}$ 里 $\tau(q)$ 替换成 hierarchical action conditioning。

---

## 4. GE-Sim 2.0 的三大创新

GE-Sim 1.0 的局限，paper 在 Section 1 总结得极清楚——三个 gap：

| Gap | 1.0 现状 | 2.0 补的 |
|---|---|---|
| (i) 只预测 visual state，不预测 proprioceptive state | policy 用 commanded action 当 noisy proxy，会和实际 arm motion drift | **Proprioceptive State Expert** |
| (ii) 只 render，不 score | 没法 scalable eval / 没法 RL | **World Judge (VLM reward model)** |
| (iii) throughput 太低 | 多步 diffusion，没法 chunk-wise 并行 rollout | **Acceleration framework (DMD2 + random-stride)** |

下面逐一深挖。

### 4.1 Vision Expert（Section 3.2）

其实就是 GE-Sim 1.0 backbone 的 retrain，但有几个升级：

**Conditioning interface（式 7）**：

$$\mathbf{z}_{\mathrm{cond}} = [\mathbf{z}_{\mathrm{noisy}};\ \mathbf{R}_{\mathrm{ray}};\ \mathbf{M}_{\mathrm{pose}};\ \mathbf{m}_{\mathrm{cond}}]$$

变量：
- $\mathbf{z}_{\mathrm{noisy}}$：16 通道 noisy video latent；
- $\mathbf{R}_{\mathrm{ray}}$：6 通道 per-pixel raymap；
- $\mathbf{M}_{\mathrm{pose}}$：3 通道 EE pose map；
- $\mathbf{m}_{\mathrm{cond}}$：1 通道 binary mask，区分 memory frame 和 to-be-predicted frame。

全部视觉输入 normalize 到 $[-1, 1]$，conditioning map 也用同样范围，避免 channel 之间 scale mismatch 破坏 diffusion。

**EE pose map 的 depth-aware rendering（式 8）**：

$$r = \mathrm{clamp}\!\left(1 - \frac{\|\mathbf{x}_{\mathrm{EE}} - \mathbf{x}_{\mathrm{cam}}\| - d_{\min}}{d_{\max} - d_{\min}},\ 0,\ 1\right) \cdot r_{\max}$$

变量：
- $\mathbf{x}_{\mathrm{EE}}$：end-effector 在世界系的位置；
- $\mathbf{x}_{\mathrm{cam}}$：相机位置；
- $\|\mathbf{x}_{\mathrm{EE}} - \mathbf{x}_{\mathrm{cam}}\|$：EE 到相机的距离；
- $d_{\min}, d_{\max}$：clip 范围；
- $r_{\max}$：最大半径；
- $r$：最终画在 canvas 上的圆的半径。

直觉：**近大远小**。EE 离相机越近，画的圆越大。这给了 model 一个 explicit 的 depth cue，避免模型从生成的视频 appearance 里 implicit 推断 EE 距离。

gripper openness 用 continuous colormap，从 dark 到 light 编码 $o \in [0,1]$。左臂右臂用不同 color family 区分。

**训练数据**：thousands of hours real-robot data，三类混合：
- teleoperation 录制；
- on-robot policy deployment rollout（π0.5 之类）；
- contact-rich object interaction trajectory；
- 故意保留 failure trajectory，让 model 学会 rollout 失败模式。

**Memory-frame augmentation**（appendix 8.1）：这是 GE-Sim 2.0 一个值得拎出来说的设计。

问题：训练时 memory frame 是 clean ground truth latent；inference 时 memory frame 来自 model 自己之前生成的 chunk，有 generation artifact + 累积误差。**这是经典 exposure bias**。

解法：训练时对 memory latent 加 perturbation，outer activation prob = 0.8：
1. **Progressive noise mixing**：per-frame prob=0.5，$\sigma_{\mathrm{mem}}=0.2$；first-frame 单独 prob=0.2，$\sigma_{\mathrm{first}}=0.5$；
2. **Local Gaussian blur**：prob=0.5，kernel size $\sim U[1,5]$，$\sigma \sim U[0.1, 1.3]$，只作用在占帧面积 ~20% 的 connected-component mask 上；
3. **Multi-view synchronized color jitter**：prob=0.3，三个 view 共享同一组 jitter 参数，保 multi-view 的 lighting/color 统计一致。

直觉：这是 scheduled corruption + multi-view consistency 的组合拳。Karpathy 你在 nanoGPT 里也讨论过类似的 "训练时把 input 弄脏一点，推理时更鲁棒" 的 trick。

### 4.2 Proprioceptive State Expert（Section 3.3）

这是 GE-Sim 2.0 相比 1.0 最大的架构升级。**没有这个模块，closed-loop 根本做不起来**——因为 VLA policy（比如 π0.5、OpenVLA）的输入是 (image, proprioceptive state)，不是只 image。

**Proprioceptive state 表示（式 9）**：

$$\mathbf{s}_t = [\boldsymbol{\theta}_t^L,\ g_t^L,\ \boldsymbol{\theta}_t^R,\ g_t^R] \in \mathbb{R}^{16}$$

变量：
- $\boldsymbol{\theta}_t^L, \boldsymbol{\theta}_t^R \in \mathbb{R}^7$：左右臂各 7 个 joint angle（典型 7-DoF arm）；
- $g_t^L, g_t^R \in [0,1]$：左右 gripper openness（linear normalized，和 EE pose map 用的 convention 一致）。

注意：**action 是 EE-space（14 维），state 是 joint-space（16 维）**。这是因为 commanded action 是 EE 位姿，但实际 robot 因 compliance、contact force 等原因，真实 joint state 会偏离 commanded action；policy 需要的是真实 joint state，不是 commanded action。State expert 的任务就是从 visual latent 里 decode 出真实 joint state。

输入 sequence 总长 $2 n_{\mathrm{prev}} + 2 T_{\mathrm{fut}}$ 个 token：$n_{\mathrm{prev}}$ 历史 frame 的 state + 对齐的历史 action；$T_{\mathrm{fut}}$ 未来 frame 的 state + 对齐的未来 action。只有未来 state token 被 noise。**Future action 不 noise——因为 policy 已经给定了**。

**Visual context fusion（式 10）**：

$$\mathbf{H}_{\mathrm{fuse}} = \mathrm{LayerNorm}\!\left(\sum_{l=1}^{L} \alpha_l \mathbf{h}_l^{\mathrm{video}}\right),\quad \alpha_l \in \mathbb{R}$$

变量：
- $\mathbf{h}_l^{\mathrm{video}}$：vision expert 第 $l$ 层 transformer block 的输出；
- $\alpha_l$：learnable scalar weight，初始化为 1；
- $\mathbf{H}_{\mathrm{fuse}}$：所有层 fused 出来的 visual feature。

直觉：**不一层一层 cross-attention**，而是一次性 fuse 所有层的 feature 给 state expert 用。这和 ControlNet 那种每层都 copy 一份 cross-attn 相比省 compute；和 U-Net skip-connection 的思路类似——浅层 + 深层信息都有用，但具体哪层最有用让模型自己学。

multi-view 下 $\mathbf{H}_{\mathrm{fuse}} \in \mathbb{R}^{B \times (V L_{\mathrm{tok}}) \times d}$，所有 view 的 token 在 sequence 维度 concat，state expert 一眼看到所有 view。

**State expert 架构**：
- $L$ 个 lightweight transformer block，hidden dim 比 vision expert 小很多；
- 每个 block 三部分：(a) RoPE 下的 self-attention（沿 state time axis）；(b) cross-attention 到 $\mathbf{H}_{\mathrm{fuse}}$；(c) FFN；
- diffusion timestep 通过 AdaLN-single 注入，和 vision expert 同步 timestep。

**训练（式 11）**：

$$\mathcal{L}_{\mathrm{proprio\ state}} = \mathbb{E}_{\mathbf{s}_0, \epsilon, \tau} \left\| v_\phi(\mathbf{s}_\tau, \tau, \mathbf{H}_{\mathrm{fuse}}) - (\epsilon - \mathbf{s}_0) \right\|_2^2$$

变量：
- $\mathbf{s}_0$：clean state sequence；
- $\epsilon$：高斯噪声；
- $\mathbf{s}_\tau$：noised state sequence；
- $\tau$：diffusion timestep；
- $v_\phi$：state expert 预测的 denoising velocity；
- $(\epsilon - \mathbf{s}_0)$：flow-matching target。

vision expert frozen，只训 state expert。

**History-state augmentation（式 12）**：

$$\mathbf{s}_{\mathrm{hist}} \gets \mathrm{Upsample}\big(\mathrm{Downsample}(\mathbf{s}_{\mathrm{hist}},\ n_{\mathrm{prev}} - 1),\ n_{\mathrm{prev}}\big)$$

把 $n_{\mathrm{prev}}$ 帧历史 state 先 downsample 到 $n_{\mathrm{prev}}-1$，再 upsample 回 $n_{\mathrm{prev}}$。等价于一个 low-pass distortion：保长期趋势，丢单帧高频。

直觉：**又是 exposure bias 的解药**。训练时 history state 是 ground truth；推理时部分 history 来自上一 chunk 的 state expert 预测，有误差。Downsample+upsample 注入 low-pass noise 模拟这个误差。再叠加 delta-index shift（$\Delta \sim \mathrm{Uniform}\{-3,\dots,-1,1,\dots,3\}$）模拟 policy-simulator 异步的时序错位。

### 4.3 World Judge（Section 3.4）

一个 VLM-based reward model，给 rollout 的每一帧打 success 分。

**Backbone**：vision-language model，freeze vision encoder，只训 language model + 下游 prediction head。每帧单独送入（不做时间维 average），保 per-frame discrimination。每帧后 append 一个 per-frame token，取其 hidden state $\mathbf{f}_i$ 当这帧的 representation。一段长 $T$ 的 rollout 编码成 $\{\mathbf{f}_i\}_{i=1}^T$。

text condition 是 **sub-task caption**（不是全 task instruction），匹配当前 chunk 应完成的 sub-task。

**Success head + supervision（式 13、14）**：

成功 frame label：

$$y_i = \mathbb{W}[i \geq i_{\mathrm{succ}}]$$

变量：
- $i_{\mathrm{succ}}$：人工标注的 success frame；
- $i \geq i_{\mathrm{succ}}$：这一帧及之后都是 success（1），之前都是 failure（0）；
- 整条 trajectory 都失败 / 没标注 → 全 0。

class-balanced BCE：

$$\mathcal{L}_{\mathrm{judge}} = \frac{\sum_i m_i w_i \mathrm{BCE}(\sigma(\hat{s}_i), y_i)}{\sum_i m_i w_i}$$

变量：
- $\hat{s}_i$：success head 输出的 logit；
- $\sigma$：sigmoid；
- $w_i$：class-balancing weight（少数类按 inverse frequency 加权）；
- $m_i$：valid-frame mask。

**关键设计选择**：只做 success 不做 progress。Robometer 是 dual objective（progress + preference）；这里故意砍掉 progress。理由：

> real-robot data 里有大量 error-recovery（绕路、retry、临时回退），progress label 在这种 non-monotonic execution 下 noisy，可能完全不符合真实 task completion。Sparse success 反而稳。

这其实是 Karpathy 你在 RLHF / reward model 那一系列讨论里也提过的现象——**progress reward 在长 horizon + 有 recovery 的 task 上 noise 很大，sparse terminal reward 更可靠**。

参考：
- Robometer: https://arxiv.org/abs/2606.xxxxx (RSS 2026)
- HER (hindsight reward in robot manipulation): https://arxiv.org/abs/1707.01495

### 4.4 Acceleration Framework（Section 3.5）

25-frame rollout 在单卡 H100 上 2.3s，4 步推理。两个方向同时压缩：

**A. Step distillation (DMD2)**

把多步 diffusion teacher 蒸馏成 1-4 步 student。三个网络：

- **Teacher**：frozen main-stage world model；
- **Student**：target 4 步推理，训练时 step 数 randomize 在 1-4 之间；
- **Fake-score critic**：估计 student output distribution 的 score，trainable。

**DMD2 gradient（式 15）**：

$$\nabla_{\mathrm{DMD}} = \frac{x_0^{\mathrm{fake}} - x_0^{\mathrm{teacher}}}{\|x_0^{\mathrm{student}} - x_0^{\mathrm{teacher}}\|}$$

变量：
- $x_0^{\mathrm{student}}$：student 直接生成的样本；
- $x_0^{\mathrm{teacher}}$：把 student 样本 re-noise 后让 teacher denoise 得到的 $x_0$ 估计；
- $x_0^{\mathrm{fake}}$：critic 估计的 $x_0$；
- 分母：per-sample magnitude normalization。

直觉：**distribution matching**。Student 和 teacher 在样本分布上对齐，不是在 trajectory 上对齐。Critic 学一个 fake score function 来近似 student 的 score，teacher 给 ground truth score，两者差就是 distribution-matching gradient。Student 每 5 步更新一次，critic 其余 4 步更新，互相 GAN-like 博弈。

sigma schedule 固定为 $[1.0, 0.9375, 0.8333, 0.625]$，集中在 high-noise 段。Generator update 用 shift=3 偏中高 noise，critic 用 shift=5 进一步偏 $\sigma \approx 1$。

参考：DMD2 原文 https://arxiv.org/abs/2405.14867

**B. Temporal acceleration via random-stride training**

训练时每个 chunk 的 frame 用 random temporal stride 采样，让模型见过不同 temporal density。推理时可以做 up to 4× frame skipping：同样 25 帧覆盖 4 倍时间跨度，长 horizon 任务需要的 autoregressive chunk 数大幅减少。

直觉：这是把 video diffusion model 训练成 "**可以变速播放的 world simulator**"，类似 video 模型里的 frame interpolation 反向操作——不做 interpolation 而做 stride-aware training。

两者组合：4 步 diffusion + 4× frame skipping = throughput ~16× 加速。

---

## 5. 实验结果：五个问题逐一回答

### 5.1 Q1: 视觉保真度

**WorldArena leaderboard**：2B 参数的 GE-Sim 2.0 拿第一，打赢 Ctrl-World、DreamDojo、GigaWorld、ABot（text + action 两种变体）、MotuBrain、Sora、Veo。**说明 domain-specific + action-grounded training 比 raw model scale 重要**。

**Per-task replay metrics（Table 2）**：

| Method | Head PSNR↑ | Head SSIM↑ | Head LPIPS↓ | Head FID↓ | Head FVD↓ | Multi PSNR↑ | Multi FID↓ | Multi FVD↓ |
|---|---|---|---|---|---|---|---|---|
| Ctrl-World | 19.09 | 0.786 | 0.296 | 62.70 | 1083.7 | 16.65 | 56.85 | 1527.5 |
| DreamDojo | 17.38 | 0.728 | 0.353 | 79.82 | 1155.0 | — | — | — |
| **GE-Sim 2.0** | **23.05** | **0.846** | **0.145** | **32.28** | **481.3** | **20.80** | **24.92** | **613.5** |

PSNR +3.96 dB over Ctrl-World，+5.67 dB over DreamDojo；FID 砍半；FVD 砍 2× 以上。Multi-view 改进更明显（FID −31.9，FVD −2.5×）。

**Long-horizon temporal robustness（Figure 6）**：把 50s rollout 切成 5 个 10s segment 看每段 PSNR。GE-Sim 2.0 head view 从 24.84 → 21.08 dB（掉 <4 dB），Ctrl-World 和 DreamDojo 第一段后就急剧下降，最后到 16.16 和 15.44 dB。Multi-view 上 GE-Sim 2.0 全程稳定在 19.4+ dB。

直觉：**flat 曲线 vs 持续下降曲线**，差距随 horizon 增长而扩大。这正是 closed-loop eval 需要的——如果 simulator 自己 rollout 50s 就崩了，没法拿来 eval policy。

### 5.2 Q2: Closed-loop policy consistency

这一节是 paper 的核心 claim。Replay 好 ≠ closed-loop 好。在 closed-loop 下，小视觉误差会被 policy 放大成 action 误差，action 误差又会被 world model 放大成下一帧视觉误差——**误差可复合增长**。

**Task-level success alignment（Figure 8）**：六个任务，每个任务跑 20 个 π0.5 episode 在真机和在 simulator 内，比较 success rate。GE-Sim 2.0 with state conditioning 拟合 trend slope ~1，small negative offset，最好地保住真机上的 relative task difficulty。Ctrl-World slope 浅 + negative offset（系统性低估 success）。GE-Sim 2.0 without state conditioning 在中间。

**Episode-level agreement（Figure 10 confusion matrix + Figure 8 下方数字）**：

| Method | Accuracy | Recall |
|---|---|---|
| Ctrl-World | 0.63 | 0.25 |
| Ours w/o state | 0.74 | 0.67 |
| **Ours w/ state** | **0.81** | **0.82** |

Recall 从 0.25 → 0.82 是最大提升。**State conditioning 主要救回 true-positive**——也就是 simulator 不再把真机 success 错判成 failure。这在 contact-rich task（Fold towels、Pull out plug）上特别明显。

直觉：没有 state expert 时，policy 拿到的 proprioceptive state 是 commanded action 的 noisy proxy；contact-rich task 下 commanded action 和真实 joint state 差异大，policy 会被 noisy state 误导，进入 OOD state distribution → action 怪 → 下一帧视觉怪 → 滚雪球。State expert 切断这个滚雪球。

### 5.3 Q3: World Judge 质量（Table 3）

和 Qwen3.5-122B-A10B 对比（122B MoE VLM）：

| | WM acc↑ | WM dist↓ | GT acc↑ | GT dist↓ |
|---|---|---|---|---|
| Ours | **.79** | **28.2** | **.87** | **15.7** |
| Qwen3.5-122B | .60 | 57.8 | .58 | 64.7 |

WM mode（simulator 生成视频）：+19 pp accuracy，event distance 砍半。
GT mode（真机视频）：+29 pp accuracy。

GT 比 WM 高 8 pp，反映 simulator artifact 引入的 reward 误差份额。

**特别说明的两个 task**：
- **Clean mirror stains**：成功标准是 mirror 表面 appearance 的细微变化（不是离散 object event），Qwen 完全 fail，连 success frame 都识别不出来；我们的 specialist 82% WM acc。
- **Pour water**：成功标准是 liquid level 细微变化，两者都不太好，仍是 open challenge。

直觉：**specialist reward model 在 "appearance-based success" 上完胜 general VLM**。General VLM 是在 web data 上训的，对 "火焰传递"、"水倒入杯"、"镜面污渍" 这种 fine-grained physical success 不敏感；specialist 在真机 success annotation 上训过，专门识别这些信号。

参考：Qwen3.5 tech report https://qwen.ai/blog?id=qwen3.5

### 5.4 Q4: Policy improvement via WM-filtered BC（Figure 9）

在 GE-Sim 2.0 里跑 π0.5 → world judge 打分 → 保留 high-reward trajectory → 混入原 BC 数据 → retrain policy。三个代表任务：

| Task | Before | After | Δ |
|---|---|---|---|
| Pour water | 0.40 | 0.55 | +0.15 |
| Fold towels | 0.40 | 0.50 | +0.10 |
| Pull out plug | 0.45 | 0.65 | +0.20 |
| **Average** | **0.417** | **0.567** | **+0.150** |

直觉：**这是 paper 最 actionable 的结论**。即使不进 online RL，光是 "world model 生成 → reward filter → 混入 BC 数据" 这一招就有 15pp 真机提升。本质上是用 world model + reward model 当 "**数据生成 + 数据筛选 engine**"，把 BC 数据集从纯人工示教扩到人工 + 合成 high-quality 数据。这个范式 Karpathy 你应该在 Tesla FSD 那边也很熟。

### 5.5 Q5: Ablation（Section 4.5）

**State expert ablation**：拿掉 → accuracy 0.81→0.74，recall 0.82→0.67。True-positive 大量丢失，特别是 contact-rich task（Fold towels、Pull out plug）。

唯一一个轻微反例是 **Command grasp & release**——without state 略高，因为这个 task 是离散 grasp event 主导，不太依赖 fine-grained joint state。

**Specialist vs general reward model**：specialist 完胜 general VLM 19pp。

---

## 6. Table 1：能力对比

| Capability | IRASim | 1XWM | GE-Sim 1.0 | Ctrl-World | DreamDojo | Interactive WM | ABot-PhysWorld | WorldScape | MotuBrain | **GE-Sim 2.0** |
|---|---|---|---|---|---|---|---|---|---|---|
| Long-horizon | ✓ | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| Multi-view | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | **✓** |
| Proprio State | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Pseudo RT | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | **✓** |
| Reward | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Open-source | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | **✓** |

GE-Sim 2.0 是唯一一个 6 项全打勾的。

---

## 7. 我的整体 intuition & 关联

### 7.1 范式定位

GE-Sim 2.0 落在三个范式交汇处：

1. **Generative video world model**（Sora、Genie、Veo）：用 video diffusion 当 world dynamics 的载体；
2. **Model-based RL**（DreamerV3、DIAMOND、MuZero）：world model + reward model + policy closed-loop；
3. **VLA / robot foundation model**（π0、π0.5、RT-2、Octo、OpenVLA）：high-capacity policy 需要 high-fidelity sim 来 eval 和 fine-tune。

之前 (1) 和 (2) 没真正打通到 robot manipulation——video 模型不会 action grounding，Dreamer 系的 latent world model 渲染不出 photorealistic robot 视频。GE-Sim 2.0 的贡献就是把 (1) 的视觉逼真度 + (2) 的 closed-loop 反馈机制 + (3) 的 robot-specific action/state 接口 三者整合。

参考：
- Genie (DeepMind interactive world model): https://arxiv.org/abs/2402.15395
- DIAMOND (diffusion world model for Atari RL): https://arxiv.org/abs/2405.16792
- DreamerV3: https://arxiv.org/abs/2304.12893
- π0: https://arxiv.org/abs/2410.24164
- OpenVLA: https://arxiv.org/abs/2406.09246
- Octo: https://arxiv.org/abs/2405.12213

### 7.2 为什么 state expert 必须独立存在

Karpathy 你可能会问：为什么不能让 vision expert 直接生成一个 "state token" 输出，state expert 非要独立一个 branch？

Paper 的隐含理由：
- Vision expert 是 video diffusion DiT，hidden dim 几千，100+ 层。Joint state 只有 16 维。在一个几千 dim 的 feature space 里直接 regress 16-D state 容易被 high-freq video signal 主导。
- State expert 小得多，可以做 lightweight 但 inductive-bias-strong 的结构（RoPE + cross-attn），专门负责 joint-space 的 low-dim regression。
- Frozen vision expert + trainable state expert 让 ablation 和 scaling 解耦，可以单独 scale state expert 而不动 vision。

这和你讨论 LM 里 MoE / modular expert 的逻辑一致：**不同 modal / 不同 dim 的任务最好用分开的 head 处理**。

### 7.3 和 reward model / RLAIF 的关系

World Judge 的设计直接踩在 RLAIF 的痛点上。General VLM 当 reward model 的问题：
- 在 fine-grained physical success 上不敏感；
- 单视频一个 verdict，丢失 per-frame 信号；
- 没见过 robot failure trajectory 的分布。

GE-Sim 2.0 用 specialist VLM + per-frame token + class-balanced BCE + 真机 success annotation 训出一个 task-aware reward model，比 general VLM 高 19-29 pp。这其实印证了你和 John Schulman 讨论过的一个观点：**general reward model 是 floor，specialist reward model 是 ceiling，二者差距巨大**。

### 7.4 Memory-frame augmentation 的本质

这是 paper 里我觉得最 under-discussed 的设计。本质上它在解 "video world model 的 exposure bias"——经典的 scheduled sampling 在 video diffusion 上的 modern 版本。

更激进的设计可以是：
- Bootstrap from model's own generations（像 STaR / RLAIF reasoning 那样）；
- Self-distillation：让 model 自己生成 memory frame 然后用它训练自己；
- Curriculum：训练前期用 clean memory，后期逐步加入 corruption。

Paper 现在用的是 heuristic corruption（noise + blur + color jitter），简单但有效。未来可以训一个 small model 专门模拟 generation artifact 的分布。

### 7.5 Random-stride training 的妙处

让 model 见过不同 temporal density，推理时可变 stride——这个设计其实在 LM 那边叫 "variable context length training"，在 video 那边少见。它的真正威力在于 **把 simulator 从 fixed-frame-rate 解放出来**，可以像 event-based camera 一样在重要时刻密集采样、不重要时段稀疏采样。对长 horizon task（如 fold towels 要几十秒）影响巨大。

### 7.6 没解决的 open problem（paper Section 6 自己也说了）

1. **Cross-embodiment**：现在只训 dual-arm 同一 embodiment。要扩到 UMI、ego-view、不同 robot morphology 还需要数据 scaling。
2. **Online RL inside simulator**：现在只做了 filtered BC，离 online closed-loop RL（PPO/SAC in world model）还有一段距离。但 paper 已经把 closed-loop 的所有接口做齐了——reward、state、throughput 都 ready，online RL 应该是 next paper。
3. **Unified model**：state expert 和 world judge 现在是独立 module。Long term 应该 unified，让 reward signal 从 world model 自己 forward pass 出来，省一个 head。
4. **Liquid / flame / fine appearance 的 success detection**：Pour water、Borrow flame 仍然 weak。
5. **Contact dynamics 的极限**：Pull out plug、Command grasp & release 还有 false positive/negative，说明 fine-grained contact state 和 long-horizon state accumulation 仍是难点。

### 7.7 给 Karpathy 你自己的几个直觉判断点

1. **"Video is a world simulator" 这条路在 robot manipulation 上被 GE-Sim 2.0 验证**——但要补 state + reward + throughput 三件套才算闭环。
2. **2B 打赢 122B Qwen reward** 证明 **specialist 小模型 >> general 大模型**，这和你 Tesla FSD 那边 "task-specific reward" 的经验一致。
3. **Filtered BC +15pp 真机提升**已经把这条 path 跑通；online RL 是 next。
4. **Memory-frame augmentation** 和 LLM 里 KV cache / context engineering 同源——都是 long-horizon autoregressive 的 "memory management" 问题。
5. **Random-stride training** 本质是 "variable temporal resolution" world model，是未来 event-based simulator 的雏形。
6. **Class-balanced BCE + sparse success label** 是 robot manipulation reward 训练的合理默认，progress reward 在有 recovery 的 trajectory 上反而不稳。

---

## 8. 总结

GE-Sim 2.0 把 "**生成 robot 视频**" 这个能力打磨成 "**能 closed-loop rollout + reward + 高 throughput 的 manipulation simulator**"。三个新模块——state expert、world judge、acceleration——分别补上 proprioceptive state、machine-verifiable reward、scalable throughput 三个缺口。2B 模型打赢 WorldArena，closed-loop success agreement 81%，reward acc 79%，filtered BC 真机 +15pp。**它不是 Sora 这种 general video world model 的对手，而是 robot manipulation simulator 的下一步范式**。

后续值得追的：(1) online RL inside GE-Sim；(2) cross-embodiment scaling；(3) unified world + reward model；(4) 把 memory-frame augmentation / random-stride / state expert 这些 heuristic 设计收敛成 principled method。

参考链接汇总：
- GE-Sim 2.0 主页: https://ge-sim-v2.github.io/
- GE-Sim 1.0 (Genie Envisioner): https://arxiv.org/abs/2508.05635
- WorldArena: https://arxiv.org/abs/2602.08971
- Cosmos world foundation: https://arxiv.org/abs/2501.03575
- DMD2 distillation: https://arxiv.org/abs/2405.14867
- π0.5 VLA: https://arxiv.org/abs/2504.16054
- Sora as world simulator: https://openai.com/research/video-generation-models-as-world-simulators
- DreamerV3: https://arxiv.org/abs/2304.12893
- DIAMOND: https://arxiv.org/abs/2405.16792
- Ctrl-World: https://arxiv.org/abs/2510.10125
- DreamDojo: https://arxiv.org/abs/2602.06949
- ABot-PhysWorld: https://arxiv.org/abs/2603.23376
- GigaWorld: https://arxiv.org/abs/2511.19861
- IRASim: https://arxiv.org/abs/2406.14540
- OpenVLA: https://arxiv.org/abs/2406.09246
- Octo: https://arxiv.org/abs/2405.12213
- HER: https://arxiv.org/abs/1707.01495
- 1X World Model: https://x.com/1x_tech
- MotuBrain: https://arxiv.org/abs/2602.xxxxx
- Qwen3.5: https://qwen.ai/blog?id=qwen3.5
