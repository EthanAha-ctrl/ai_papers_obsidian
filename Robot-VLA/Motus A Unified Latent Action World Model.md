---
source_pdf: Motus A Unified Latent Action World Model.pdf
paper_sha256: 4f9be253165cbb3567f56d23d200ba93bda1242195e579abaac987ea37cc264f
processed_at: '2026-08-05T20:55:08-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Motus 用人话说

Hey Andrej，给你一个 plain Chinese 版本，少公式多直觉。

## 一句话总结

现在的 robot learning 模型都是"单科状元"——有专做看图说话的（VLM），有专做生成视频的（VGM），有专做动作控制的（VLA），有专做"给定未来反推动作"的（IDM），有专做"给定动作预测未来"的（World Model）。Motus 把这五个"单科"合并成一个"全科"模型，靠三招：**共享注意力的 MoT 架构** + **用噪声时间步切换模式的 UniDiffuser 调度器** + **用光流作为跨机器人通用动作语言**。结果是仿真上 +15%~45%，真实世界 +11%~48%，尤其是煮咖啡、磨豆子这种 long-horizon 任务提升最猛。

## 之前为什么是分裂的

你想想机器人做任务这件事，本质上涉及五种能力：

- **看懂**（VLM）：当前画面里有个杯子，指令是"把杯子放到盘子上"
- **想象**（VGM）：脑子里放电影——手伸过去、抓住、抬起、移动、放下
- **规划动作**（VLA）：具体输出每个关节怎么动
- **反推动作**（IDM）：看了别人怎么做，反推每一步动作是什么
- **推演后果**（WM）：如果做这个动作，下一帧画面会变成什么样

这五件事其实是**同一个联合分布** $p(o, a \mid \text{context})$ 的不同 marginal/conditional。但是历史上大家各做各的——OpenVLA 只管 VLA，[Genie](https://arxiv.org/abs/2402.15391) 只管 latent action，[$\pi_{0.5}$](https://arxiv.org/abs/2504.16054) 只管 VLA + 一点 reasoning，[VidAR](https://arxiv.org/abs/2507.12898) 只管生成视频再提 action。

问题来了：如果你只训 VLA，模型不知道"未来会怎么样"；如果你只训 VGM，模型不知道"动作和视觉的物理对应关系"。这就好比一个人只会背菜单不会做菜，或者只会做菜不会点单——干不了完整工作。

## Motus 怎么把它们合起来的

### 第一招：MoT 架构（三个 expert 共享注意力）

Motus 拿了三个 pretrained model 拼起来：
- **Wan 2.2 5B**（[视频生成](https://arxiv.org/abs/2503.20314)）当 "generative expert"
- **Qwen3-VL 2B**（[VLM](https://arxiv.org/abs/2502.13923)）当 "understanding expert"
- 自己搭一个 253M 的 Transformer 当 "action expert"

关键设计叫 **Tri-model Joint Attention**：三个 expert 的 **self-attention 层是共享的**，但每个 expert 有自己的 **FFN**。

直觉：attention 层负责"让三个 modality 互相看到对方在说什么"（cross-modal routing），FFN 层负责"各自记自己的 specialization"。这跟 [Bagel](https://arxiv.org/abs/2505.14683)、[Mixture-of-Transformers](https://openreview.net/forum?id=HwW8tYbvgZ) 思路一致——你之前也在 tweet 里提过，理解和生成不该强行 fuse 在一个 FFN 里，但可以让 attention 做 cross-modal knowledge flow。

对比 [UWM](https://arxiv.org/abs/2504.02792)（Unified World Model）直接把 obs token 和 action token concat 喂进同一个 transformer stack，Motus 这种"attention 共享、FFN 分离"避免了 representation 冲突，同时保留各 expert 的 specialization。

### 第二招：UniDiffuser 调度器（一个模型五个模式）

这是我觉得最 elegant 的部分。Motus 给 video 和 action **分别**分配一个噪声时间步 $\tau_o$ 和 $\tau_a$。推理时通过固定其中一个 modality 在 clean（$\tau=0$）或 pure noise（$\tau=T_\tau$），就能切换出五种模式：

| 模式 | $\tau_o$ 起始 | $\tau_a$ 起始 | 推理什么 | 给定什么 |
|------|------------|------------|---------|---------|
| **VGM** | $T_\tau$（denoise） | $T_\tau$（保持 noise） | 视频 | 当前帧 + 语言 |
| **WM** | $T_\tau$（denoise） | 0（保持 clean） | 视频 | 当前帧 + 动作 |
| **IDM** | 0（保持 clean） | $T_\tau$（denoise） | 动作 | 视频序列 |
| **VLA** | $T_\tau$（保持 noise） | $T_\tau$（denoise） | 动作 | 当前帧 + 语言 |
| **Joint** | $T_\tau$（denoise） | $T_\tau$（denoise） | 视频 + 动作 | 当前帧 + 语言 |

人话：同一个网络，**完全相同的权重**，只在推理时改一下"哪些 token 是噪声、哪些是 clean"，就能做完全不同的事情。这跟 [UniDiffuser](https://arxiv.org/abs/2303.06555) 在 image-text 上的做法一脉相承，Motus 把它搬到 video-action 上。

训练时，$\tau_o$ 和 $\tau_a$ 都从 $[1, T_\tau]$ 随机采样，所以模型同时学了 marginal、conditional、joint 三种 distribution。这就跟你之前讲 "next-token prediction 隐式学了所有 conditional" 的道理一样——一个 generative model 学了 joint，自然能 condition 出任何 marginal。

### 第三招：光流作为"跨机器人的通用动作语言"

这是 Motus 最有意思的 insight。问题：互联网视频、人类 ego-centric 视频、不同机器人轨迹——这些数据的 action space 完全不一样（人的手是肌肉+骨骼，Aloha 是 14-DoF 双臂，Franka 是 7-DoF 单臂）。怎么让 action expert 从这些异构数据里学 motion prior？

答案：**用 optical flow**。

直觉：不管谁在动——人的手抓杯子、Aloha 抓方块、Franka 推东西——在 pixel 空间看，都是"某些像素从 A 位置移到 B 位置"。Optical flow 就是这个"像素位移场"。所以 flow 是 **embodiment-agnostic** 的 motion 语言。

具体 pipeline：
1. 用 [DPFlow](https://openaccess.thecvf.com/content/CVPR2025/html/Morimitsu_DPFlow_Adaptive_Optical_Flow_Estimation_With_a_Dual-Pyramid_Framework_CVPR_2025_paper.html) 算光流，转成 RGB 表示（horizontal + vertical 两个通道）
2. 用 [DC-AE](https://arxiv.org/abs/2412.19399) 压缩成 4 个 512-dim token
3. 轻量 encoder 投射到 **14 维**——刚好对应 Aloha 双臂的 DoF
4. 用 90% 无标签视频做 reconstruction + 10% 有标签机器人轨迹做 alignment

loss 是：
$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_a \| a_{\text{real}} - a_{\text{pred}} \|^2 + \beta \mathcal{L}_{\text{KL}}$$

人话：主要让 VAE 重建光流，同时用少量真实 action 标签把 latent 拉向真实 action 分布，让它最后能映射回真实控制信号。$\beta = 10^{-6}$ 几乎不开 KL，他们想让 reconstruction 主导。

这个 latent action 思路跟 [Genie](https://arxiv.org/abs/2402.15391)、[LAPA](https://arxiv.org/abs/2410.11758)、[AdaWorld](https://arxiv.org/abs/2506.10978) 是一个 lineage，但 Motus 用 optical flow 而非 RGB pixel 作为 reconstruction target，避免了 latent 把 appearance 信息也编进去（[Zhang et al. "What do latent action models actually learn?"](https://arxiv.org/abs/2506.15691) 指出这个问题）。也跟 [VideoJam](https://arxiv.org/abs/2502.02492)、[FlowVLA](https://arxiv.org/abs/2508.18269) 思路接近。

## 训练 recipe：三阶段 + 六层数据金字塔

这是 paper 里最"工程化"的部分，但其实是关键。

### 三阶段

1. **Stage 1（8000 GPU 小时）**：只用 VGM，在人类视频 + 仿真数据 + 多机器人轨迹上学视频生成。让模型先懂"物理世界长什么样、物体怎么动"。

2. **Stage 2（10000 GPU 小时）**：三个 expert 联合训练，VLM frozen，用 **latent action** 作为 action 监督。这一步是核心——把视频里的 motion prior 通过光流 latent action 注入 action expert。

3. **Stage 3（400 GPU 小时）**：用目标机器人的少量真实轨迹（~2000 条）做 SFT，把 latent action 对齐到真实 action space。

人话：先用大量"杂"数据学通用物理和 motion，最后用极少目标机器人数据"窄化"到具体 embodiment。这个 18000:400 的 GPU hour 比例，跟你讲 "pretraining is the main dish, fine-tuning is the dessert" 的逻辑完全一致。

### 六层数据金字塔

从下到上：web image-text → egocentric human video → 仿真数据 → task-agnostic（用 Curobo 随机采样的动作）→ 多机器人轨迹 → 目标机器人轨迹。数据量从 230k+ 到 2k 递减，质量递增。这跟 LLM pretraining 用 web data → SFT 用 instruction data 的层级一致。

## 效果怎么说

### 仿真（RoboTwin 2.0，50 个任务）

| 方法 | Clean | Randomized |
|------|-------|------------|
| $\pi_{0.5}$ | 43% | 44% |
| X-VLA | 73% | 73% |
| Motus | **89%** | **87%** |

比 $\pi_{0.5}$ 提升 +45%，比 X-VLA 提升 +15%。所有方法都只 fine-tune 40k steps，所以差距主要来自 pretraining strategy。

### 真实世界

**AC-One 平台（9 个任务平均）**：
- $\pi_{0.5}$: 14.79%
- Motus: **63.22%**（+48%）

亮点：
- 磨咖啡豆：$\pi_{0.5}$ 8% → Motus 92%
- 煮咖啡：$\pi_{0.5}$ 0% → Motus 62%
- 浇花：5% → 65%

这些 long-horizon 任务正是 unified model 的优势区——需要想象未来（WM）、理解指令（VLM）、规划动作（VLA）、反推动作（IDM）协同。

**Agilex-Aloha-2（5 个任务平均）**：
- $\pi_{0.5}$: 48.60%
- Motus: 59.30%（+11%）

提升幅度比 AC-One 小，我猜是因为 Aloha 任务相对简单，baseline 已经 48%，ceiling 效应 + Stage 3 数据少。

## 我的几个直觉

### 为什么 unification 真的有用

你想想一个具体场景：煮咖啡。要完成这个任务，机器人需要：
1. 看懂"咖啡机"是什么、按钮在哪（VLM）
2. 想象"按按钮后水会流出来"（WM/VGM）
3. 规划"伸手 → 按按钮 → 拿杯子 → 接水"（VLA）
4. 如果看别人演示过，反推每一步动作（IDM）

如果你只训 VLA，模型不知道按按钮的后果，出错没法 self-correct。如果你只训 VGM，模型能想象但不会动作。Motus 的 ablation 也验证了这点：Stage 1 only（只 VGM）只有 82%，加 Stage 2 latent action 后到 87%——**video 里的 motion prior 通过 latent action 注入 action expert 是关键 transfer 来源**。

### 为什么光流是个好选择

Optical flow 本质上是 **"什么在动、往哪动"** 的纯几何信息，剥离了 appearance、texture、lighting。这跟 [JEPA](https://arxiv.org/abs/2506.07863) 思路一致——LeCun 一直强调"在 abstract representation space 做 prediction"，光流恰好是一个介于 pixel 和 semantic 之间的中间层。

潜在问题：deformable object（毛巾、衣服）的光流估计噪声大，[DPFlow](https://openaccess.thecvf.com/content/CVPR2025/html/Morimitsu_DPFlow_Adaptive_Optical_Flow_Estimation_With_a_Dual-Pyramid_Framework_CVPR_2025_paper.html) 在 non-rigid 上误差本来就高。Paper 里 Fold Towel real-world 才 14-39%，可能这部分受限。改进方向：用 [DINOv2 feature flow](https://arxiv.org/abs/2304.07193) 或 [Tracking-Any-Point](https://arxiv.org/abs/2307.15992) 替代 raw optical flow。

### MoT 架构的哲学

你之前说过"VLM 不该强行 unify 理解和生成"，因为两个任务对 representation 的要求冲突。[Bagel](https://arxiv.org/abs/2505.14683) / [Janus](https://arxiv.org/abs/2410.13848) 的 MoT 思路验证了这点——attention 共享做 routing，FFN 分开做 specialization。Motus 把这个思路推到三方（VLM + VGM + Action），我觉得是合理的下一步。

类比：这就像公司里有产品、工程、设计三个部门，他们需要**开会**（attention 共享）对齐信息，但各自有**专业工具**（FFN 分开）。强行让一个部门干所有事会乱套。

### 跟你 Eureka Labs 思路的呼应

你之前讲 "education is about the model, not the data"——pretrained prior 比 downstream fine-tuning 重要。Motus 印证了这点：18000 GPU hours 学通用 prior，400 GPU hours 适配目标机器人。这个 ratio 是 embodied AI 正确的方向。

对比 [AgiBot World Colosseo](https://arxiv.org/abs/2503.06669) 直接用 750k 真实轨迹暴力训 VLA 的路子，Motus 用 latent action 从视频借 motion prior，sample efficiency 高得多。这跟你讲"LLM 的 few-shot 能力来自 pretraining"的逻辑同构。

## 我会问作者的问题

1. **5 个 mode 真的都用了吗**？实验主要展示 VLA + Joint + VGM + WM，IDM 只在 Table 7 验证。真实部署时怎么选 mode？Joint vs VLA 性能差 3%，是否默认该用 Joint？

2. **Action expert 253M 够吗**？跟 VGM 2.13B 差一个量级。Unified training 时 attention 共享让 action expert 借力 VGM，但精细控制是否够？sim→real gap 还在。

3. **Latent action 14 维怎么扩展到 humanoid**（30+ DoF）？Paper 没讨论。这跟你最近关注 [humanoid](https://figure.ai/) 的方向有 gap。

4. **Optical flow 在 deformable object 上可靠吗**？Fold Towel 任务效果一般，可能是光流估计噪声导致。

5. **开源吗**？[Project page](https://motus-robotics.github.io/motus) 在但没说 release code。Wan 2.2 和 Qwen3-VL 都开，复现应该可行。

## 一句话再总结

Motus 的 contribution 用人话讲：**"把 VLM + VGM + VLA + IDM + WM 这五个之前各自为战的能力，用一个共享注意力、分离 FFN 的 MoT 架构合到一个模型里，靠光流 latent action 让 action expert 能从互联网视频里学 motion prior，靠 UniDiffuser 调度器实现一个模型五个推理模式，最后用 18000:400 的 pretraining:fine-tuning GPU hour 比例验证了 pretrained prior 在机器人任务上的重要性。"**

方向跟你一直讲的 "pretraining is the main dish" 高度一致，方法上把 [Bagel](https://arxiv.org/abs/2505.14683) 的 MoT、[UniDiffuser](https://arxiv.org/abs/2303.06555) 的 scheduler、[Genie](https://arxiv.org/abs/2402.15391) 的 latent action 三个 lineage 巧妙缝合。实验数据 +15%~48% 提升很扎实，尤其 long-horizon real-world task。

---

**Reference Links:**

核心 paper 和架构：
- [Motus Project Page](https://motus-robotics.github.io/motus)
- [Mixture-of-Transformers](https://openreview.net/forum?id=HwW8tYbvgZ)
- [Bagel](https://arxiv.org/abs/2505.14683)
- [UniDiffuser](https://arxiv.org/abs/2303.06555)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)

Latent Action lineage：
- [Genie](https://arxiv.org/abs/2402.15391)
- [LAPA](https://arxiv.org/abs/2410.11758)
- [AdaWorld](https://arxiv.org/abs/2506.10978)
- [UniVLA](https://arxiv.org/abs/2505.06111)
- [VideoJam](https://arxiv.org/abs/2502.02492)
- [FlowVLA](https://arxiv.org/abs/2508.18269)

VLA baselines：
- [$\pi_{0.5}$](https://arxiv.org/abs/2504.16054)
- [X-VLA](https://arxiv.org/abs/2510.10274)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [RDT-1B](https://arxiv.org/abs/2410.07775)
- [UWM](https://arxiv.org/abs/2504.02792)
- [$\mathcal{F}_1$](https://arxiv.org/abs/2509.06951)

Datasets：
- [AgiBot World Colosseo](https://arxiv.org/abs/2503.06669)
- [Egodex](https://arxiv.org/abs/2505.11709)
- [RoboMind](https://arxiv.org/abs/2412.13877)
- [RoboTwin 2.0](https://arxiv.org/abs/2506.18088)

Components：
- [Wan 2.2](https://arxiv.org/abs/2503.20314)
- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923)
- [DC-AE](https://arxiv.org/abs/2412.19399)
- [DPFlow](https://openaccess.thecvf.com/content/CVPR2025/html/Morimitsu_DPFlow_Adaptive_Optical_Flow_Estimation_With_a_Dual-Pyramid_Framework_CVPR_2025_paper.html)

如果你想我再展开某个具体部分（比如 latent action VAE 的实现、UniDiffuser 调度器的数学推导、或者某个具体实验 task 的 subgoal breakdown），说一声。

---

# Motus: A Unified Latent Action World Model - 详细技术讲解

Hey Andrej，这篇 paper 我仔细读了几遍，给你一个 deep dive。整体来看，这是 Tsinghua 团队（Bi Hongzhe, Tan Hengkai 等）做的 unified embodied foundation model，思路是把你之前在 VLA, World Models, IDM, Video Generation 这几个分支上分裂开的工作合到一个 backbone 里。下面我从 architecture, math, data, experiments 几个层面拆开讲。

## 1. 核心问题：为什么需要 unification

Paper 识别了 5 个 embodied intelligence 的核心 distribution：

$$
\begin{aligned}
\text{VLA} &: p(a_{t+1:t+k} \mid o_t, \ell) \\
\text{WM} &: p(o_{t+1:t+k} \mid o_t, a_{t+1:t+k}) \\
\text{IDM} &: p(a_{t+1:t+k} \mid o_{t:t+k}) \\
\text{VGM} &: p(o_{t+1:t+k} \mid o_t, \ell) \\
\text{Joint} &: p(o_{t+1:t+k}, a_{t+1:t+k} \mid o_t, \ell)
\end{aligned}
$$

符号解释：$o_t$ 是 time $t$ 的 observation（视觉输入），$a_{t+1:t+k}$ 是 action chunk（chunk size = $k$，他们用 $k=16$），$\ell$ 是 language instruction，下标 $t:t+k$ 表示从 $t$ 到 $t+k$ 的整段 trajectory。

这五个 conditional distribution 看起来像是 VLA, World Model, IDM, VGM 这些孤立工作各自优化一个，但其实它们是同一个 joint distribution $p(o_{t:t+k}, a_{t+1:t+k} \mid o_t, \ell)$ 的不同 marginal/conditional。UWM (Unified World Models, [Zhu et al. 2025](https://arxiv.org/abs/2504.02792)) 先做了这件事，但 Motus 想更进一步：把 VLM 和 VGM 的 pretrained priors 也接进来。

## 2. Architecture: Mixture-of-Transformers (MoT) + Tri-model Joint Attention

### 2.1 三个 Expert 的组合

Motus 用三个 pretrained expert 拼装：
- **Generative Expert**: Wan 2.2 5B ([Wan 2.2](https://arxiv.org/abs/2503.20314)) — video diffusion backbone
- **Understanding Expert**: Qwen3-VL-2B ([Qwen2.5-VL](https://arxiv.org/abs/2502.13923)) — 提供 vision-language 理解
- **Action Expert**: 一个跟 Wan 同深度的 Transformer（hidden size 1024, 30 layers, 24 heads）

总规模 5B 参数（VGM 2.13B + VLM 1.5B + Act 253.5M + Und 8M + overhead）。

### 2.2 Tri-model Joint Attention 的关键设计

这里 paper 的 Figure 1 应该仔细看。MoT 思路来自 [Bagel (Deng et al. 2025)](https://arxiv.org/abs/2505.14683) 和 [Mixture-of-Transformers (Liang et al. 2025)](https://openreview.net/forum?id=HwW8tYbvgZ)。

每个 expert 保持自己的 FFN / AdaLN（action expert）或 LayerNorm（understanding expert），但是 **multi-head self-attention 是跨 expert 共享的**——这就是 "Tri-model Joint Attention"。

直觉上讲，这种设计避免了把 obs token 和 action token 简单 concat 进同一堆 UWM block 导致的 representation 冲突（UWM 的局限）。每个 expert 的 FFN 可以学自己的 specialization，但 attention 层提供 cross-modal knowledge fusion。

形式上：对每个 expert $i \in \{v, a, u\}$（video, action, understanding）的 token $X_i$：
$$
\text{Attn}_{\text{joint}}([X_v, X_a, X_u]) = \text{softmax}\left(\frac{Q_v Q_a Q_u \cdot K_v K_a K_u}{\sqrt{d}}\right) V_v V_a V_u
$$
（这里简化写法，实际是把三个 modality 的 token 拼成 sequence 做 self-attention）

每个 expert 自己的 FFN 独立处理：
$$
X_i \leftarrow X_i + \text{FFN}_i(\text{Attn}_{\text{joint}}(X_i))
$$

### 2.3 UniDiffuser-style Scheduler

这是最 tricky 的部分。Motus 借鉴 [UniDiffuser (Bao et al.)](https://arxiv.org/abs/2303.06555) 的思路，给 video 和 action 分配 **独立的时间步** $\tau_o$ 和 $\tau_a$，从 $[1, T_\tau]$ 采样。这样通过控制 $(\tau_o, \tau_a)$ 的组合，可以在 inference 时切换 5 种 mode。

Rectified flow 训练目标：
$$
\begin{aligned}
l_{\text{action}}^\theta &= \mathbb{E}_{(o_{t:t+k}, a_{t+1:t+k}, \ell) \sim \mathcal{D}} \left\| v_a^\theta - (\epsilon_a - a_{t+1:t+k}) \right\|_2^2 \\
l_{\text{obs}}^\theta &= \mathbb{E}_{(o_{t:t+k}, a_{t+1:t+k}, \ell) \sim \mathcal{D}} \left\| v_o^\theta - (\epsilon_o - o_{t+1:t+k}) \right\|_2^2 \\
l^\theta &= l_{\text{action}}^\theta + l_{\text{obs}}^\theta
\end{aligned}
$$

符号：
- $\epsilon_a, \epsilon_o \sim \mathcal{N}(0, I)$：Gaussian noise
- $\tau_a \sim \mathcal{N}(0, T_\tau)$, $\tau_o \sim \mathcal{M}(0, T_\tau)$：时间步（这里 paper 写法有点诡异，$\tau_a$ 用 Normal, $\tau_o$ 用另一个 distribution $\mathcal{M}$——可能是 logit-normal，与 Table 11 的 "Logit Normal" 对应）
- $v_a^\theta, v_o^\theta$：模型预测的 velocity field
- ground truth 是 $(\epsilon_a - a)$ 和 $(\epsilon_o - o)$，即从 noise 到 data 的方向向量

### 2.4 Inference 时的 5 种模式切换

这部分参考 supplementary 的 Algorithm 2-6。核心 trick 是在推理时固定其中一个 modality 的 $\tau$ 在 0（clean）或 $T_\tau$（pure noise）：

| Mode | $\tau_o$ 起始 | $\tau_a$ 起始 | 推理对象 | Condition |
|------|--------------|---------------|---------|-----------|
| VGM | $T_\tau$ | $T_\tau$ (固定 noise) | $o_{t+1:t+k}$ | $o_t, \ell$ |
| WM | $T_\tau$ | 0 (固定 clean) | $o_{t+1:t+k}$ | $o_t, a_{t+1:t+k}$ |
| IDM | 0 (固定 clean) | $T_\tau$ | $a_{t+1:t+k}$ | $o_{t:t+k}$ |
| VLA | $T_\tau$ (固定 noise) | $T_\tau$ | $a_{t+1:t+k}$ | $o_t, \ell$ |
| Joint | $T_\tau$ | $T_\tau$ | both | $o_t, \ell$ |

很 elegant 的设计。一个模型，五个 inference mode，全靠 noise schedule 控制。

## 3. Action-Dense Video-Sparse Prediction

这是 paper 里一个小但重要的 trick。Action chunking 用 $k=16$，但 video frames 如果按 30Hz 全预测就是 16 帧。问题是：
1. Video tokens 数量远超 action tokens（每帧 ~几千 token vs. action 14 维 × 16）
2. 训练/推理慢
3. **Tri-model Joint Attention 失衡**，模型偏向 video prediction

解决方案：**video frame rate = action frame rate / 6**。具体：8 frames @ 5Hz video + 48 actions @ 30Hz（看 Table 11）。

直觉：人对未来视觉细节的高频预测其实不需要那么 dense，但 action 需要高频精细控制。这个 asymmetric sampling 在 VLA 里也算常见做法（[Diffusion Policy](https://arxiv.org/abs/2303.04137) 也用类似思路），但 paper 在 unified 模型里专门强调，是为了让 attention token 数量平衡。

## 4. Latent Actions: 从 Optical Flow 学 motion prior

### 4.1 为什么不用 RGB 直接 reconstruct latent action

Latent action model (LAM) 一支参考 [Genie (Bruce et al.)](https://arxiv.org/abs/2402.15391), [LAPAdo (Ye et al.)](https://arxiv.org/abs/2410.11758), [AdaWorld (Gao et al.)](https://arxiv.org/abs/2506.10978)。早期用 RGB reconstruct next frame 作为 IDM 监督，但会把 appearance 也编进 latent（[Zhang et al. "What do latent action models actually learn?"](https://arxiv.org/abs/2506.15691)）。后续工作要么 bottleneck latent 维度，要么换 DINOv2 feature、keypoints、language。

Motus 选择 **optical flow** 作为 motion 的"通用语言"：
- DPFlow ([Morimitsu et al. CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Morimitsu_DPFlow_Adaptive_Optical_Flow_Estimation_With_a_Dual-Pyramid_Framework_CVPR_2025_paper.html)) 算 flow
- 转成 RGB image 表示（horizontal + vertical 通道）
- 用 [DC-AE (Chen et al. ICLR 2025)](https://arxiv.org/abs/2412.19399) compress 成 4 个 512-dim token
- 轻量 encoder 投射到 14 维（≈ 通用 robot action 维度，比如 Aloha 14-DoF bimanual）

为什么 14 维？这样 latent action 和真实 robot action 维度对应，便于用少量标签做 alignment supervision。

### 4.2 训练 loss

Latent Action VAE 的训练目标：

$$
\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_a \| a_{\text{real}} - a_{\text{pred}} \|^2 + \beta \mathcal{L}_{\text{KL}}
$$

- $\mathcal{L}_{\text{recon}}$: flow reconstruction（pixel-level MSE on flow RGB）
- $\| a_{\text{real}} - a_{\text{pred}} \|^2$: action alignment，把 latent 拉向真实 action
- $\mathcal{L}_{\text{KL}}$: KL regularizer，约束 latent space
- $\lambda_a = 1.0$, $\beta = 1 \times 10^{-6}$（Table 11）——KL 几乎不开，说明他们想让 reconstruction 主导

Data mix: **90% unlabeled** (互联网视频/ego-centric) + **10% labeled** (task-agnostic + 任务数据)。Task-agnostic data 来自 [AnyPos (Tan et al.)](https://arxiv.org/abs/2506.18088) 用 Curobo 在 simulation 里 random sample action space 收集 image-action pair。

### 4.3 直觉：为什么 optical flow 是 cross-embodiment 的桥梁

这点跟 [VideoJam (Chefer et al.)](https://arxiv.org/abs/2502.02492), [FlowVLA (Zhong et al.)](https://arxiv.org/abs/2508.18269) 思路接近。Optical flow 是 **embodiment-agnostic** 的——人的手、Aloha 双臂、Franka 单臂、AC-One 全在 pixel 空间表现为同样的位移场。所以 latent action 学到的 "motion primitive" 可以 transfer 到不同 robot。

这点其实是你之前在 nanoGPT 和 Eureka Labs 一直强调的：找一个 "modality-agnostic" 的中间表征比直接对齐 raw action space 更通用。

## 5. Three-Phase Training Pipeline + Six-Layer Data Pyramid

### 5.1 三阶段

| Stage | 数据 | 训练什么 | GPU Hours |
|-------|------|---------|----------|
| 1. Video Generation | Level 2 (Egodex human) + Level 3 (RoboTwin sim) + Level 5 (multi-robot) | 只 VGM | ~8000 |
| 2. Unified Training with Latent Actions | Level 2-5 (不含 target robot) | Motus 三个 expert 联合，用 latent action | ~10000 |
| 3. SFT | Level 6 (target robot trajectories) | Motus 三 expert + 真 action | ~400 |

Stage 2 关键：VLM frozen，只训 VGM + Action expert + Understanding expert。用 latent action 作为 action 监督，让 action expert 吸收 video 里的 motion prior。

Stage 3 才用真 robot action 微调。LR 从 $8 \times 10^{-5}$ 降到 $1\sim5 \times 10^{-5}$，AdamW + weight decay 0.01, batch 256。

### 5.2 Six-Layer Data Pyramid

参考 Figure 4 + Table 12：

| Level | 数据类型 | 代表 dataset | 规模 |
|-------|---------|-------------|------|
| 1 | Web Data (image-text) | 公开预训练 | web-scale |
| 2 | Egocentric Human Videos | [Egodex](https://arxiv.org/abs/2505.11709) | 230,949 |
| 3 | Synthetic Data | RoboTwin 2.0 | 27,500 |
| 4 | Task-Agnostic | AnyPos | 1,000 |
| 5 | Multi-Robot Trajectory | [Agibot World Colosseo](https://arxiv.org/abs/2503.06669), [RDT](https://arxiv.org/abs/2410.07775), [RoboMind](https://arxiv.org/abs/2412.13877) | 750k+ |
| 6 | Target-Robot Trajectory | In-house | 2,000 |

直觉：从下到上数据量递减、质量递增。Stage 1 用 1-3 级学视觉动态；Stage 2 加上 4-5 级学 motion prior；Stage 3 才用 6 级做 embodiment-specific 对齐。这跟你之前讲 LLM pretraining "broad → narrow" 的逻辑一致。

## 6. Experiments

### 6.1 RoboTwin 2.0 Simulation（50 任务）

Table 2 给的关键数字（avg success rate）：

| Method | Clean | Randomized |
|--------|-------|------------|
| $\pi_{0.5}$ ([Black et al. 2025](https://arxiv.org/abs/2504.16054)) | 42.98% | 43.84% |
| X-VLA ([Zheng et al. 2025](https://arxiv.org/abs/2510.10274)) | 72.80% | 72.84% |
| w/o Pretrain | 72.8% | 77.00% |
| Stage 1 only | 82.86% | 81.86% |
| **Motus** | **88.66%** | **87.02%** |

vs $\pi_{0.5}$ 提升 **+45%**，vs X-VLA 提升 **+15%**。

训练条件很苛刻：所有 model 都只 fine-tune 40k steps 从 pretrained checkpoint 起步。这意味着 Motus 的 advantage 主要来自 pretraining strategy，不是模型容量（Motus 5B 跟 $\pi_{0.5}$ 量级接近）。

注意几个有意思的 case：
- "Pick Dual Bottles"：Stage1 only 反而崩到 7%/17%，但 full Motus 96%/90% → 说明 Stage 2 latent action 预训练对这个 task 关键
- "Scan Object"：所有 baseline 都挣扎在 14-69%，Motus 67%/66% → 视觉理解类任务
- "Hanging Mug"：所有方法都很烂（最高 38%），deformable + precise insertion 还是开放问题

### 6.2 Real-World（AC-One + Agilex-Aloha-2）

Table 3 的 partial success rate（任务被分解成 subgoal，按 subgoal 完成度评分）：

**AC-One 平台（9 个任务 avg）**：
- $\pi_{0.5}$: 14.79%
- w/o Pretrain: 25.86%
- **Motus: 63.22%** (提升 +48%)

亮点：
- "Grind Coffee Beans": $\pi_{0.5}$ 8% → Motus 92%
- "Brew Coffee": $\pi_{0.5}$ 0% → Motus 62%
- "Pour Water to Flowers": 5% → 65%

这些 long-horizon 任务正是 unified model 受益最大的地方——WM 能想象 future，VLA 能 plan，IDM 能反推 action，三者协同。

**Agilex-Aloha-2（5 个任务 avg）**：
- $\pi_{0.5}$: 48.60%
- w/o Pretrain: 26.60%
- **Motus: 59.30%** (提升 +11%)

提升幅度小于 AC-One，作者没明说原因，我猜：Aloha-2 任务相对简单（baseline 已经 48%），ceiling 效应 + Stage 3 fine-tune data 有限。

### 6.3 World Model 生成质量（Table 6）

| Platform | FID↓ | FVD↓ | SSIM↑ | LPIPS↓ | PSNR↑ |
|----------|------|------|-------|--------|-------|
| Agilex-Aloha-2 | 9.46 | 49.28 | 0.886 | 0.054 | 26.10 |
| AC-One | 12.96 | 73.13 | 0.846 | 0.073 | 24.04 |

FID ~10 级别在真实 robot 视频生成里算不错（对比 [Sora-style](https://openai.com/sora) 在 web data 上的水平）。说明 unified training 没有显著牺牲 VGM 的视觉质量。

### 6.4 IDM 测试（Table 7）

Action MSE on RoboTwin 2.0：

| Model | MSE |
|-------|-----|
| ResNet18+MLP | 0.044 |
| DINOv2+MLP | 0.122 |
| **Motus** | **0.014** |

比专门 train 的 IDM baseline 还好——说明 unified training 反而帮助 IDM 能力，可能因为 WM 和 VLA 学到的 prior 给 IDM 提供了 context。这跟我直觉一致：从未来反推 action 比单纯从过去推 action 信息更多。

### 6.5 LIBERO-Long 和 VLABench

LIBERO-Long：Motus 97.6% → match X-VLA SOTA
VLABench：In-distribution +5%, Cross-category +3% vs $\pi_{0.5}$（绝对值仍低，cross-category 才 25%，说明 generalization 还有大空间）

## 7. Ablation 关键发现

Figure 6 + Table 3 ablation:
- **w/o Pretrain**: 跟 Motus 差距最大，尤其 real-world（25.86% vs 63.22%）
- **Stage 1 only**（只 VGM pretrain, 没用 latent action）: 81.86% sim / 真实世界大幅低于 full Motus

直觉：Stage 2 latent action pretraining 是最重要的 transfer 来源。只靠 VGM 学视觉 dynamic 不够，必须把 video 里的 motion 通过 latent action 注入 action expert。

## 8. 跟 Related Work 的定位

Paper 的 related work 部分把 Motus 放在一个清晰的 lineage：

- **Unified Multimodal**: [Bagel](https://arxiv.org/abs/2505.14683), [Chameleon](https://arxiv.org/abs/2405.09818), [Emu3](https://arxiv.org/abs/2409.18869), [Janus](https://arxiv.org/abs/2410.13848), [Show-o](https://arxiv.org/abs/2408.12528), [MMaDA](https://arxiv.org/abs/2505.15809) → MoT 思路来源
- **Latent Action**: [Genie](https://arxiv.org/abs/2402.15391), [LAPA](https://arxiv.org/abs/2410.11758), [AdaWorld](https://arxiv.org/abs/2506.10978), [LAOM](https://arxiv.org/abs/2502.00379), [MOTO](https://arxiv.org/abs/2412.04445), [CoMo](https://arxiv.org/abs/2505.17006), [UniVLA](https://arxiv.org/abs/2505.06111) → latent action 学习
- **VLA**: [$\pi_0$](https://arxiv.org/abs/2410.24164), [$\pi_{0.5}$](https://arxiv.org/abs/2504.16054), [OpenVLA](https://arxiv.org/abs/2406.09246), [RDT-1B](https://arxiv.org/abs/2410.07775), [GR00T-N1](https://arxiv.org/abs/2503.14734), [H-RDT](https://arxiv.org/abs/2506.07634) → VLA baseline
- **VGM for control**: [Gen2Act](https://arxiv.org/abs/2409.16283), [VidAR](https://arxiv.org/abs/2507.12898), [RoboDreamer](https://arxiv.org/abs/2406.10993), [Video2Policy](https://arxiv.org/abs/2502.09886)
- **$\mathcal{F}_1$** ([Lv et al. 2025](https://arxiv.org/abs/2509.06951)): VLA + IDM 但没 VGM
- **UWM** ([Zhu et al. 2025](https://arxiv.org/abs/2504.02792)): 5 个 distribution 都 unify，但 from scratch

Motus 的位置：UWM 的 unify 思路 + Bagel 的 MoT architecture + optical flow latent action + 用 pretrained VGM/VLM。一句话总结："how to inherit both internet-scale visual-language priors AND robot-scale interaction priors in one model."

## 9. 我对这篇 paper 的几个观察 / 直觉

### 9.1 强的地方

1. **Tri-modal Joint Attention 是合理的归纳偏置**。把 attention 共享、FFN 分开，比纯 concat token 好——attention 做 routing，FFN 做 specialization，类似 [Switch Transformer](https://arxiv.org/abs/2101.03961) 的 sparse MoE 思路，但 modality 之间 dense 路由。

2. **UniDiffuser scheduler 的 elegance**：用 $(\tau_o, \tau_a)$ 的组合做 mode 切换，避免多个 head 或多 model。这种 design pattern 在 [Stable Diffusion + ControlNet](https://arxiv.org/abs/2302.05543) 那类工作里也见过——一个 base model，多个 conditioning 路径。

3. **Optical flow 作为 cross-embodiment 语言**：这跟 [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) 的 "action conditioning via latent" 思路类似，但 Motus 显式用 optical flow 而非 latent VAE，更具物理可解释性。

4. **Stage 2 latent action pretraining 是关键 transfer 来源**：ablation 显示这点提升最大。直觉：互联网 video 没有 action label，但 latent action 把 video 的 motion 翻译成可学信号。

### 9.2 我会问的问题

1. **Optical flow 在 deformable object 上可靠吗？** Towel folding 这种任务，flow 估计噪声很大（[DPFlow](https://openaccess.thecvf.com/content/CVPR2025/html/Morimitsu_DPFlow_Adaptive_Optical_Flow_Estimation_With_a_Dual-Pyramid_Framework_CVPR_2025_paper.html) 在 non-rigid 上误差本来就高）。Paper 里 Fold Towel 任务 real-world 才 14-39%，可能这部分受限。

2. **Action expert 容量太小（253M）**？跟 VGM 2.13B 比差一个量级。Unified training 时 attention 共享可能让 action expert 借力 VGM，但 action 的精细控制是否够？Table 7 IDM MSE 0.014 比 baseline 好很多，但 sim→real gap 还在。

3. **Stage 3 SFT 只有 400 GPU hours，2k target trajectories**——这跟 [AgiBot World Colosseo](https://arxiv.org/abs/2503.06669) 750k 规模比小得多。是不是因为 unified pretraining 把 sample efficiency 提上来了？这其实是论文最强的 selling point 之一：**用很少 target data 适配新 robot**。

4. **5 个 mode 真的都用了吗**？Paper 实验 section 主要展示 VLA + Joint + VGM + WM，IDM 只在 Table 7 验证。真实部署时 inference mode 怎么选？Joint vs VLA 性能差 3.12%（87.02 vs 83.90），是不是默认应该用 Joint？

5. **Latent action 14 维够吗**？双臂 bimanual + gripper 大概 14 DoF 刚好，但人形 30+ DoF 怎么办？Paper 没讨论 scaling 到 humanoid。这跟你最近关注 humanoid ([Unitree, Figure](https://figure.ai/)) 的方向有 gap。

### 9.3 跟你之前想法的呼应

你之前 tweet 过 "VLM 是理解和生成的 trade-off 不该强行 unify"，[Janus](https://arxiv.org/abs/2410.13848) / [Bagel](https://arxiv.org/abs/2505.14683) 的 MoT 思路验证了这点——共享 attention、分离 encoder/FFN。Motus 把这个思路推到 VLA + VGM + VLM 三方。我觉得这是合理的下一步。

你 [Eureka Labs](https://www.eurekalabs.ai/) 的"教育 is about the model, not the data"角度其实跟 Motus 的 philosophy 一致——pretrained prior 比 downstream fine-tuning 重要。Stage 1+2 投入 18000 GPU hours 学 general prior，Stage 3 只用 400 GPU hours 适配，这个 ratio 在 embodied AI 里是正确的方向。

## 10. 一些可能的延伸联想

- **跟 LeCun 的 JEPA 思路对比**：[V-JEPA 2](https://arxiv.org/abs/2506.07863) 也在 latent space 学 predictive model，但 JEPA 不显式生成 video，更关注 representation。Motus 走相反路——显式生成 + latent action。两种思路的对比值得专门研究。

- **跟 Genie 2 对比**：DeepMind 的 Genie 2 也是 latent action world model，但更游戏化（keyboard/mouse）。Motus 是 manipulation 专精。两者 latent action 设计可能可以互通。

- **Diffusion vs Flow Matching**：Motus 用 rectified flow（[Lipman et al. 2023](https://arxiv.org/abs/2209.03003)），不用传统 DDPM。Flow matching 在 robot control 里越来越主流（[$\pi_0$](https://arxiv.org/abs/2410.24164) 也是）。跟 [Consistency Policy](https://arxiv.org/abs/2407.03598) 比会怎样？

- **Open-source 问题**：Paper 提了 [project page](https://motus-robotics.github.io/motus) 但没说 release code。Wan 2.2 是 open 的，Qwen3-VL 也是。如果 release 应该能复现。

- **跟 [Mamba in robotics](https://arxiv.org/abs/2403.18410) 的对比**：长 sequence modeling 在 long-horizon manipulation 上可能比 attention 更合适。Motus 用 Transformer，但 video chunk $k=16$ 不长，attention 够用。如果 $k=64$ 或 128 可能要换 SSM。

## 11. 实操角度：怎么复现 / 改进

如果你想 build on top of Motus：

1. **复现 latent action VAE**：可以单独训——DPFlow + DC-AE + lightweight encoder + 90/10 mix。这部分 self-contained，可以独立 verify。

2. **复现 MoT**：Bagel 的 [开源代码](https://github.com/ByteDance-Seed/Bagel) 应该可以直接改，加 action expert + understanding expert。

3. **改进点**：
   - 把 latent action 从 optical flow 换成 [DINOv2 feature flow](https://arxiv.org/abs/2304.07193) 或 [Tracking-Any-Point](https://arxiv.org/abs/2307.15992)，对 deformable object 更鲁棒
   - 加一个 [Flow Matching + Consistency](https://arxiv.org/abs/2407.03598) 的 inference 加速
   - 把 action expert 扩到 1B+，看 sim→real 是否更好
   - Multi-embodiment inference 时显式注入 embodiment embedding（类似 [HPT](https://arxiv.org/abs/2409.20537)）

## 12. 总结一句

Motus 的 contribution 是把"unified modeling"和"pretrained priors"两件事同时做到 embodied foundation model 上，技术 lever 是 MoT (architecture) + UniDiffuser scheduler (mode switching) + optical-flow latent action (cross-embodiment bridge) + 3-stage pretrain (data pyramid)。

数据上 +15% ~ +48% 提升很扎实，特别是 long-horizon real-world task（brew coffee, grind beans）。这个方向跟你之前讲 "pretraining is the main dish" 的 thesis 高度一致。

---

**Reference Links:**

Paper:
- [Motus Project Page](https://motus-robotics.github.io/motus)
- [arXiv (待 release)](https://arxiv.org)

Architecture:
- [Mixture-of-Transformers](https://openreview.net/forum?id=HwW8tYbvgZ)
- [Bagel: Unified Multimodal Pretraining](https://arxiv.org/abs/2505.14683)
- [UniDiffuser](https://arxiv.org/abs/2303.06555)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)
- [Wan 2.2 Video Generation](https://arxiv.org/abs/2503.20314)
- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923)
- [DC-AE](https://arxiv.org/abs/2412.19399)
- [DPFlow](https://openaccess.thecvf.com/content/CVPR2025/html/Morimitsu_DPFlow_Adaptive_Optical_Flow_Estimation_With_a_Dual-Pyramid_Framework_CVPR_2025_paper.html)

Related Robotics:
- [$\pi_{0.5}$](https://arxiv.org/abs/2504.16054)
- [X-VLA](https://arxiv.org/abs/2510.10274)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [RDT-1B](https://arxiv.org/abs/2410.07775)
- [GR00T-N1](https://arxiv.org/abs/2503.14734)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [Unified World Models](https://arxiv.org/abs/2504.02792)
- [$\mathcal{F}_1$ VLA+IDM](https://arxiv.org/abs/2509.06951)
- [VidAR](https://arxiv.org/abs/2507.12898)
- [Gen2Act](https://arxiv.org/abs/2409.16283)

Latent Action:
- [Genie](https://arxiv.org/abs/2402.15391)
- [LAPA](https://arxiv.org/abs/2410.11758)
- [AdaWorld](https://arxiv.org/abs/2506.10978)
- [UniVLA](https://arxiv.org/abs/2505.06111)
- [MOTO](https://arxiv.org/abs/2412.04445)
- [VideoJam](https://arxiv.org/abs/2502.02492)
- [FlowVLA](https://arxiv.org/abs/2508.18269)

Datasets:
- [AgiBot World Colosseo](https://arxiv.org/abs/2503.06669)
- [Egodex](https://arxiv.org/abs/2505.11709)
- [RoboMind](https://arxiv.org/abs/2412.13877)
- [RoboTwin 2.0](https://arxiv.org/abs/2506.18088)
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864)

如果你想要我 deep dive 某个具体部分（比如 latent action VAE 的实现细节、UniDiffuser scheduler 的数学推导、或者 real-world experiment 的 subgoal breakdown），告诉我一声，我可以再展开。
