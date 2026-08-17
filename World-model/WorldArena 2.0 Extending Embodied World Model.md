---
source_pdf: WorldArena 2.0 Extending Embodied World Model.pdf
paper_sha256: c712ff5fc8366d6278477a71d3f7219f07c61ccef66dcb1aa0da27f5177b5f0e
processed_at: '2026-08-13T05:35:14-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 WorldArena 2.0

好，把公式和表格全扔掉，用大白话重讲一遍。

---

## 这篇 paper 在干嘛

你就想象一个场景：你在教机器人倒水。机器人脑子里有个"世界模型"——它先在脑子里"放映"一段未来视频："如果我这样转手腕，水会从杯口流出来，落到桌上"——然后根据这段想象决定怎么动手。

这就是 embodied world model 的核心卖点：**不用真试，先想一遍，想象得准就能干活**。

问题是，现在整个领域都在 claim "我的 world model 很强"，但到底强在哪、弱在哪，没人说得清。你说你的视频生成得好？那跟机器人能不能干活是两码事。你说你的 policy success rate 高？那只在模拟器里测的，搬到真机器人上还行不行？

WorldArena 2.0 就是来当这个裁判的。它把评估分成三把尺子，每把尺子都专门戳现有方法的痛处。

---

## 第一把尺子：加触觉

你想想，机器人插 HDMI 线的时候，眼睛看插孔能看个大概，但**最后那 2 毫米对准没对准、插进去那一下的阻力大小、有没有卡住**——这些信息在视频像素里几乎读不出来。人类拧螺丝的时候也不靠看，靠手感。touch signal 的 frequency 比 vision 高一个数量级，是 contact-rich 任务的命门。

但所有现有 benchmark 都只评 vision。WorldArena 2.0 加了一路 tactile signal，做法很聪明——不重训你的 video model，而是加一个 tactile VAE 把触觉信号"翻译"进 video model 的 latent space，再让两路 denoising 一起跑，最后接一个 diffusion policy head。

结果非常有意思：

- **Wan 2.2（通用视频大模型）触觉预测质量反而比专门的机器人模型 Vidar 高**。为啥？因为 Wan 2.2 预训练见过的跨模态知识多，latent space 更通用，能更好地容纳新模态。这又是一个"foundation model 的通用性 > 专用架构"的例证。
- **短任务（Insert HDMI）加了触觉直接 100% 成功**——因为这种任务成败就决定在接触那一下，触觉预测准就等于任务准。
- **长任务（Lift Bottle）所有 world model 全挂 0%**——ACT 这种 closed-loop policy 反而 80%。因为 long-horizon 力控需要每一步重新读触觉 sensor 做 feedback，但 world model 是"开环想象"，一次性 rollout 到底，没有 closed-loop feedback 循环。这是 generative world model 的根本缺陷：它生成的是**电影**，不是**控制信号**。

---

## 第二把尺子：让它当 RL 环境

以前怎么评 world model？让它生成一段未来视频，看看漂亮不漂亮，或者让一个 frozen policy 跑一下看 success rate。这都是**一次性评估**，没测它**反复 rollout 会不会越滚越离谱**。

WorldArena 2.0 干脆把 world model 当成 RL 的虚拟环境用：policy 在 world model 里想象 rollout，拿到 reward，更新自己，再 rollout……几百个回合下来，最后把训好的 policy 放到真实环境测。

这就考三件事：world model 长期稳不稳？transition 准不准？能不能给 policy 提供有用的 learning signal 而不是错误信号？

结果呢：

- **最好的 world model（WoVR）在 Click Bell 任务上 75%，simulator 是 87%**——还差 12 个点，但已经接近了。这说明 world model 当 RL env 这条路快通了。
- **reward model 的设计比 world model 本身还重要**。他们对比了三种 reward：proxy-based（专门训的小网络，最准）、VLM-based（用 Qwen 当裁判，没微调，拉胯）、similarity-based（看视觉相似度，太依赖生成质量）。proxy-based 全场最佳。这告诉你的直觉是：**world model 不行可以调，reward model 不行全完蛋**。就像你给学生一个错误答案的参考，学生再聪明也学歪。
- **OpenSora / IRASim / iVideoGPT 几乎没提升**——这些 naive video model 在 RL 场景里只能提供 marginal signal。光会生成漂亮视频 ≠ 能当 dynamics model。

---

## 第三把尺子：搬到真机器人

最扎心的部分。所有 simulator 里的高分，都可能是假象。

他们用三个平台搭了个阶梯：

- **RoboTwin 2.0**：模拟器，但加了大量 domain randomization（光照、背景、桌子高度、物体姿态全随机），专治过拟合。
- **LIBERO**：模拟器，但 design 上能**诊断到底是哪种 knowledge transfer 不了**——是物体关系不行？还是 articulated body 不行？还是 goal 没对齐？
- **AgileX ALOHA**：真机器人，双臂，倒水和擦桌子两个任务。sensor noise、摩擦变化、actuator 不完美全在这里。

结论：

- **视觉质量指标（visual/motion/physics/3D）跨平台相关性还可以**——低层的 fidelity 和几何 reasoning 能 transfer。
- **语义级指标（content consistency、controllability）跨平台就散了**——指令理解和 action 响应跟 domain 绑死。
- **task success rate 从 sim 到 real 大跳水**：24 个 real-world cell 里大部分是 0，偶尔几个 40%、50% 的 outlier。real-world 评估不可替代。
- **Veo 3.1 视觉最好但 trajectory accuracy 烂**（0.12 vs CtrlWorld 的 0.48）。又一句"漂亮 ≠ 准"。
- **CtrlWorld 在三个平台的 trajectory accuracy 都领先**，因为它的 controllable generation 把物理 dynamics 真的 encode 进去了。这是 design choice 决定的，不是 scaling 决定的。

---

## 这套 benchmark 想告诉你什么

如果只记五句话：

1. **视频好看跟机器人能干活几乎无关**。physics encoding 才是真正的 differentiator，CtrlWorld 和 WoW 证明了这点。
2. **触觉是短 horizon contact 任务的银弹，但解决不了长 horizon 力控**。current world model 是开环的，闭环 force control 仍然是死穴。
3. **world model 当 RL env 已经接近 simulator，但还没追上**。bottle neck 在 reward model，不在 world model。
4. **sim-to-real gap 在 functional level 极其严重**，视觉指标能 transfer，task success 基本不能。
5. **通用 foundation model 在跨模态/跨平台 transfer 上强，专用 model 在 in-domain physics 上强**。Wan 2.2 触觉预测好，Vidar real-world robust，各有所长。

---

## Andrej 你大概会怎么想

你之前讲 "video models as world models" 的时候就提过这个问题：Sora 生成的视频漂亮，但它不知道自己生成的是物理世界还是抽象图像序列。WorldArena 2.0 本质上就是**把这个直觉量化、标准化、做成可比较的 benchmark**。

它没解决任何问题，但它让所有人都没法藏了。你不能再 claim "我的 world model 在 X 任务上 95%"——你得在三个模态、两种功能、三个平台下都过关才算数。这是把整个 field 往诚实方向推了一步。

如果让我赌一个方向，我会押 **latent-space visuotactile world model + closed-loop RL with proxy reward**——把触觉做在 latent 里（避免高维 tactile 生成），用短 horizon imagination + 真实环境 fallback 做 closed-loop。这是 WorldArena 2.0 暴露的最 actionable 的 open problem。

---

# WorldArena 2.0 详解 — Embodied World Model 的三轴扩展评估

Andrej，这篇 paper 你应该会很有共鸣。它本质上是在追问一个你已经思考很久的问题：**当一个 generative model 能"预测未来视频"时，它到底是不是一个 world model？** WorldArena 1.0 已经迈出第一步（perceptual quality + functional utility 联合评估），但 2.0 把这个问题推到了更严肃的层面——加上触觉、加上 closed-loop RL、加上 real robot。我下面把整套设计拆开讲，重点放在能 build intuition 的地方。

paper 链接: https://world-arena.ai  
arXiv (WorldArena 1.0): https://arxiv.org/abs/2602.08971  
相关 survey (Shang et al. 2026): https://arxiv.org/abs/2602.08971

---

## 1. 为什么需要 WorldArena 2.0 —— 三个被忽视的 gap

现有 embodied world model (EWM) benchmark 的三个 latent 问题：

**(a) Modality gap**: 几乎所有 benchmark（EWM-Bench, WorldSimBench, WorldEval, WoW-World-Eval, WorldArena 1.0）都只用 vision。但 contact-rich manipulation（插拔、按压、抓取）的真实物理量在 visual stream 里是部分可观测的——force、slip、friction、material compliance 这些量在 RGB 像素里被严重 alias。Vision-only 训练出来的 "world model" 在 contact moment 附近会产生 systematic hallucination，因为它从没见过 ground truth 的 contact signal。

**(b) Functionality gap**: 现有 functional evaluation 全部停留在 open-loop planning 或 frozen policy evaluation。这相当于只测了 world model "一次预测得准不准"，没测 "反复 rollout 1000 次会不会 compounding error 爆炸"。Dreamer 系列早就证明 world model 的真正价值在于 closed-loop imagination rollout，但没人 standardized 这个评测。

**(c) Platform gap**: 所有数字都来自 simulator。Sim-to-real gap 在 manipulation 上极大，没有 real-world 数据等于自欺欺人。

这三点合起来，就构成了 2.0 的 modality × functionality × platform 三轴扩展。

```
            Modality
              ↑
   vision-only ───→ visuotactile
              │
              │
Platform ←───┼───→ Functionality
sim-only ─────→ sim+real        offline ───→ online RL env
```

---

## 2. Modality Extension: Visuotactile World Model 标准化 pipeline

### 2.1 架构解析（Figure 2）

这个 pipeline 的核心 design choice 是 **plug-in augmentation, no architectural surgery**——保留现有 video world model 的 latent space 和 diffusion backbone，只在外围加三个模块：

**(1) Tactile VAE**  
输入是 tactile deformation map 序列 $x^T_{1:t} = \{x^T_1, ..., x^T_t\}$（通常来自 DIGIT/GelSight 类 sensor 的 RGB-D 形变图）。VAE 将其编码到 latent $z^T_t$，并通过一个 alignment projection $g_\psi(\cdot)$ 映射到 video world model 的 latent space $\mathcal{Z}^V$：

$$z^T_t \in \mathcal{Z}^T \xrightarrow{g_\psi} \tilde{z}^T_t \in \mathcal{Z}^V$$

这样 tactile 信息和 video latent 在同一空间内可被 cross-attention 共享，避免改 backbone。

**(2) Visuotactile Two-Stream World Model**  
两条 denoising stream 同步进行：

$$\epsilon_V \sim \mathcal{N}(0, I), \quad \epsilon_T \sim \mathcal{N}(0, I)$$
$$z^V_t = D_\theta(z^V_t, t, c, \tilde{z}^T_{<t}), \quad z^T_t = D_\phi(z^T_t, t, c, z^V_{<t})$$

其中 $c$ 是 conditioning（action, language instruction），$D_\theta, D_\phi$ 是两个 denoising network，通过 cross-attention 共享 latent。这种 two-stream 设计保留了 modality-specific dynamics，同时允许 cross-modal coordination——比简单 concatenate 强很多。

**(3) Action Diffusion Head**  
基于 Diffusion Policy (Chi et al. RSS 2023, https://diffusion-policy.cs.columbia.edu/) 的思路：

$$a_{t:t+H} = \text{Denoise}(a^N_{t:t+H} | o_{t-K:t}, a_{t-K:t}, \hat{z}^V_{t:t+H}, \hat{z}^T_{t:t+H})$$

$H$ 是 action horizon，$K$ 是 observation history length，$N$ 是 diffusion steps。关键：action diffusion 不是从纯 noise 还原，而是 conditioned on **predicted visuotactile latents**——这意味着 policy 可以"看见"自己即将引发的 contact 后果再决定 action。

### 2.2 UniVTAC 实验结果（Table 1）

| Model | PSNR↑ | SSIM↑ | Insert HDMI (%) | Lift Bottle (%) | Avg (%) |
|---|---|---|---|---|---|
| ACT (baseline) | — | — | 20 | **80** | 50 |
| Vidar | 13.97 | 0.278 | 70 | 0 | 35 |
| Genie Envisioner | 13.36 | 0.456 | 0 | 0 | 0 |
| **Wan 2.2** | **21.26** | **0.746** | **100** | 0 | 50 |

UniVTAC simulator: https://arxiv.org/abs/2602.10093

**关键 insight 1**: Wan 2.2（general-purpose video model）的 tactile PSNR/SSIM 比专门 embodied model（Vidar）高得多。Why？因为 Wan 2.2 预训练时见过海量 cross-modal data（虽然没明确做触觉），它的 latent space 更 rich，能 better align 到 tactile modality。这印证了一个 intuition：**foundation model 的 representation 通用性 > task-specific architecture**。

**关键 insight 2**: Lift Bottle 这个 long-horizon task 全部 world model 都挂了（0%），但 ACT 拿到 80%。这是 paper 里最诚实也最有信息量的一行数据。Lift Bottle 需要 sustained force control，触觉主要是 high-frequency feedback（slip detection, force closure），但 **world model 一次只能 rollout 固定 horizon**，无法形成 closed-loop force feedback。ACT 是 closed-loop policy，每 step 都重新条件化于真实 sensor 读数。这暴露了 generative world model 在 long-horizon contact control 上的根本缺陷：它生成的是"开环 future"，不是"闭环 control signal"。

**关键 insight 3**: Insert HDMI Wan 2.2 100% 成功——这个任务短 horizon、接触瞬间决定成败（成功 insert 需要精确感知 HDMI 口位置 + 力对齐），触觉 prediction 准确直接转化为任务成功。说明 tactile injection 对**短 horizon、contact-critical** task 极其有效。

### 2.3 我的扩展联想

这个 pipeline 让我想到几个相关工作值得对照：
- **VTAM** (Yuan et al. 2026, https://arxiv.org/abs/2603.23481): Video-Tactile-Action model，处理复杂 physical interaction，比 VLA 强很多
- **Visuo-Tactile World Models** (Higuera et al. 2026, https://arxiv.org/abs/2602.06001): Meta/CMU 工作
- **OmniVTA** (Zheng et al. 2026, https://arxiv.org/abs/2603.19201): visuotactile world modeling for contact-rich
- **DIGIT tactile sensor**: https://arxiv.org/abs/2005.14097
- **SNESS / GelSight**: 触觉 sensor 标准化还在早期，UniVTAC 是一次尝试

---

## 3. Functionality Extension: World Model 作为 RL Environment

### 3.1 形式化为 POMDP

paper 用一个 POMDP $\mathcal{M} = (\mathcal{O}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma, \rho_0)$ 刻画真实环境，然后用 $\hat{\mathcal{P}}^\phi$ 近似 $\mathcal{P}$。变量含义：

- $\mathcal{O}$: observation space（这里主要是 visual frame，部分 case 加 tactile）
- $\mathcal{A}$: action space（机器人关节/末端执行器 command）
- $\mathcal{P}(o_{t+1}|o_t, a_t)$: 真实环境 transition kernel，**未知**
- $\mathcal{R}(o_t, a_t)$: 真实 reward，**未知**
- $\gamma \in [0,1)$: discount factor，weighs future reward
- $\rho_0$: 初始 observation 分布

World model 的角色就是学一个 $\hat{\mathcal{P}}_\theta(o_{t+1}|o_t, a_t)$ 去近似 $\mathcal{P}$，从而把 RL training 从真实环境（昂贵、危险）搬到 imagined environment（free, safe, infinite data）。

### 3.2 四组件 + 三阶段 pipeline（Figure 3）

**四个组件**：

| Component | 参数 | 形式 | 作用 |
|---|---|---|---|
| World Model Env | $\phi$ | $\hat{\mathcal{P}}_\theta(o_{t+1}|o_t, a_t)$ | 替代 simulator 提供 transition |
| Reward Model | $\psi$ | $\hat{r}_t = \hat{\mathcal{R}}^\psi(o_t, \hat{a}_t)$ | 替代真实 reward signal |
| Policy Model | $\theta$ | $a_t \sim \pi_\theta(\cdot|o_t)$ | 待优化的 VLA policy |
| Optimization | — | maximize $\mathcal{I}(\theta)$ | GRPO / PPO 等 |

**三个阶段**：
1. **Stage 1**: World model training on real dataset $\mathcal{D} = \{(o_t, a_t, o_{t+1}, r_t)\}_{i=1}^N$
2. **Stage 2**: RL policy optimization inside world model (imagination rollout)
3. **Stage 3**: Deploy optimized policy 到 real env 测 success rate

### 3.3 公式 (1)-(3) 详解

**公式 (1) — 闭环 trajectory 生成**：
$$o_0 \sim \rho_0, \quad a_t \sim \mathcal{P}_\phi(\cdot|\hat{o}_t, a_t), \quad \hat{r}_t = \hat{\mathcal{R}}_\psi(o_t, a_t)$$

注意这里有个 typo 嫌疑：第二项应该是 $a_t \sim \pi_\theta(\cdot|\hat{o}_t)$，$\hat{o}_{t+1} \sim \hat{\mathcal{P}}_\phi(\cdot|\hat{o}_t, a_t)$。意思是：从初始 observation 采样，policy 给出 action，world model 给出 next observation，reward model 给出 reward，递归生成 imagined trajectory $\tau = (o_0, a_0, r_0, o_1, a_1, r_1, ...)$。

**公式 (2) — World model 训练**：
$$\phi^* = \text{argmin}_\phi \mathcal{L}_{\text{WM}}(\phi; \mathcal{D})$$

$\mathcal{L}_{\text{WM}}$ 可以是 diffusion loss（如 Wan/CogVideoX/Cosmos）、autoregressive next-token loss（如 iVideoGPT）、或 latent dynamics loss（如 Dreamer 的 ELBO）。这个 abstraction 让 framework 兼容所有 world model 架构。

**公式 (3) — Policy gradient**：
$$\nabla \mathcal{I}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta, \hat{\mathcal{P}}_\phi} \left[ \left(\sum_{t=0}^{T-1} \nabla \log \pi_\theta(a_t|o_t)\right) \hat{A}_t(\tau) \right]$$

- $\mathbb{E}_{\tau \sim \pi_\theta, \hat{\mathcal{P}}_\phi}$: trajectory 从 policy 和 world model 联合采样
- $\nabla \log \pi_\theta(a_t|o_t)$: score function，policy 在该 state 下选该 action 的 log 概率梯度
- $\hat{A}_t(\tau)$: advantage estimator at timestep $t$，可以是 GAE、return-to-go、或 GRPO 的 group-relative baseline
- $T$: trajectory horizon

这就是标准 policy gradient theorem，区别在于 transition 来自 world model 而非真实环境。这里 compounding error 是核心隐患：$\hat{\mathcal{P}}_\phi$ 累积误差会让 imagined trajectory 偏离真实 manifold，policy 在想象中学到的策略可能 useless 甚至 harmful。

### 3.4 实验设置细节

- Base policy: **π_0.5** (Physical Intelligence, https://www.physicalintelligence.company/blog/pi05)
- Tasks: Adjust Bottle, Click Bell (RoboTwin 2.0)
- Data: 3000 trajectories（两个 performance level 的 policy + simulator action planner 生成）
- SFT initialization: 1000 expert trajectories
- Optimizer: **GRPO** (DeepSeekMath, https://arxiv.org/abs/2402.03300)
- Evaluation: 100 real-world rollouts 测 success rate

GRPO 的核心思想：对同一 prompt 采样 $G$ 个 responses，用 group mean 作为 baseline 计算 advantage，避免训一个 critic network。在 world model RL 场景下，这个选择很合理——因为 critic 也得在 world model 里训，引入双重 model error。

### 3.5 Table 2 解读 —— 三种 reward model × 8 个 world model

| Method | Proxy Click Bell | Proxy Adjust Bottle | VLM Click Bell | VLM Adjust Bottle | Sim Click Bell | Sim Adjust Bottle |
|---|---|---|---|---|---|---|
| SFT | 43.75 | 55.08 | 43.75 | 55.08 | 43.75 | 55.08 |
| **Simulator RL** | **87.30** | **78.90** | 87.45 | 78.90 | 87.45 | 78.90 |
| OpenSora | 56.25 | 60.16 | 55.27 | 57.03 | 53.13 | 58.00 |
| IRASim | 53.13 | 61.33 | 53.52 | 58.98 | 50.78 | 59.38 |
| iVideoGPT | 52.53 | 56.25 | 48.44 | 58.59 | 52.15 | 60.93 |
| Cosmos-Predict-2.5(action) | 67.38 | 63.48 | 54.10 | 58.40 | 63.09 | 61.13 |
| RoboScape | 68.75 | 60.74 | 55.46 | 59.38 | 63.48 | 59.18 |
| Ctrl-World | 69.53 | **70.70** | 66.80 | 65.04 | 69.92 | 66.02 |
| **WoVR** | **75.00** | 67.19 | 69.38 | 64.45 | 72.07 | 61.35 |

**关键观察**：

1. **Simulator RL 是 upper bound**: 87.30 vs 78.90。所有 world model 都还没追上，但 top performer (WoVR 75.00 on Click Bell) 已经接近 simulator 的 86%。这意味着 world model RL 即将 cross the chasm。

2. **WoVR 擅长短 horizon (Click Bell)**: Click Bell 是一次按压动作，short-horizon contact，world model 的 rollout 准确性直接决定 policy 质量。WoVR (https://arxiv.org/abs/2602.13977) 专门 design 成 reliable simulator for VLA post-training。

3. **Ctrl-World 擅长 long horizon (Adjust Bottle)**: 70.70 vs WoVR 的 67.19。Adjust Bottle 需要持续调整姿态，long-horizon。Ctrl-World (https://arxiv.org/abs/2510.10125) 的 controllable generation 优势在此显现。

4. **Proxy-based reward 全面胜出**: 对比三个 reward model 列，Proxy-based 在所有 model 上几乎都最高。VLM-based（Qwen-3.5 没在任务上微调）和 similarity-based（依赖 observation prediction quality）都有明显 failure mode。这个发现很重要——**reward model 的 task-specific 微调比其 architecture 选择更关键**。

5. **OpenSora / IRASim / iVideoGPT 改进微弱**: 这三个 video generation quality 不足以支撑可靠 RL，marginal improvement over SFT。说明 naive video model ≠ world model。

### 3.6 Figure 5 的 training curve 分析

Figure 5 显示 Click Bell 上 policy success rate vs environment steps。几乎所有 model 都能引导 policy 改进，但收敛速度和 plateau 不同。WoVR 收敛最快且 plateau 最高，IRASim/iVideoGPT 早期甚至会 dip（world model 早期不稳定的 rollout 让 policy 学到错误信号），然后才缓慢恢复。这个 curve shape 提示一个 design pattern：**world model 在 RL 早期应该 short rollout，逐步 extend horizon**——类似 Dreamer 的 curriculum。

### 3.7 相关工作联想

- **Dreamer V3** (Hafner et al. Nature 2025, https://danijar.com/project/dreamerv3/): latent world model + actor-critic in imagination，是这条 line 的开山之作
- **World-Env** (Xiao et al. 2025, https://arxiv.org/abs/2509.24948): world model 作为 VLA post-training env
- **RoboScape-R** (Tang et al. 2025, https://arxiv.org/abs/2512.03556): unified reward-observation world model
- **WMPO** (Zhu et al. 2025, https://arxiv.org/abs/2511.09515): world model-based policy optimization for VLA
- **PlayWorld** (Yin et al. 2026, https://arxiv.org/abs/2603.09030): from autonomous play
- **RLinf** (Zang et al. 2025, https://arxiv.org/abs/2510.06710): 这篇 paper 的 pipeline 基础

---

## 4. Platform Extension: Cross-Embodiment Sim-to-Real

### 4.1 三个平台的 diagnostic 作用

| Platform | 角色 | Tasks | 关键 stress test |
|---|---|---|---|
| **RoboTwin 2.0** | domain randomization 压力测试 | Adjust Bottle, Click Bell | visual/spatial distribution shift |
| **LIBERO** | structured knowledge transfer 诊断 | Turn on the Stove | 物体关系 + articulated body dynamics |
| **AgileX ALOHA** | physical reality check | Pour Water, Wipe Table | sensor noise, variable friction, imperfect actuation |

RoboTwin 2.0: https://arxiv.org/abs/2506.18088  
LIBERO: https://libero-project.github.io/  
AgileX: https://www.agilex.ai/  
ALOHA / Mobile ALOHA: https://mobile-aloha.github.io/

这个三段式设计很 elegant——RoboTwin 测 robustness（"任何场景都能行"），LIBERO 测 transferability（"哪些 knowledge 能 transfer"），AgileX 测 deployability（"现实中真能行"）。任何只在某一平台刷高分的 model 都会被这套设计 expose。

### 4.2 两个评估协议

**(1) Embodied Data Engine**:  
World model 生成 synthetic trajectories $\rightarrow$ train downstream policy $\rightarrow$ 在真实 task 上测 success rate。这个 protocol 测的是 world model 作为 **数据增强器** 的能力。

**(2) Embodied Action Planner**:  
World model 直接预测 closed-loop action sequence $\rightarrow$ 在真实环境执行 $\rightarrow$ 测 task completion rate。这个 protocol 测的是 world model 作为 **policy** 的能力。

### 4.3 Table 3 —— 跨平台 functional 结果

| Model | RoboTwin DE T1 | RoboTwin DE T2 | RoboTwin AP T1 | RoboTwin AP T2 | LIBERO DE | LIBERO AP | Real DE T1 | Real DE T2 | Real AP T1 | Real AP T2 |
|---|---|---|---|---|---|---|---|---|---|---|
| GigaWorld | 2 | 13 | 6 | 19 | 0 | 0 | 0 | 0 | 0 | 0 |
| Genie Envisioner | 7 | 21 | 10 | 20 | 2 | 6 | 0 | 0 | 0 | 20 |
| TesserAct | 1 | 35 | 1 | 35 | 34 | 38 | 0 | 0 | 0 | 30 |
| Vidar | 13 | 53 | 2 | 19 | 22 | 14 | **40** | 0 | 30 | 10 |
| Wan 2.2 | 15 | 41 | 12 | 20 | 10 | 24 | 10 | 0 | 10 | 0 |
| CogVideoX | 3 | 28 | 8 | 16 | 0 | 2 | 10 | 10 | 0 | **50** |

**关键 insight**:

1. **Real-world 是 graveyard**: 6 个 model × 4 个 real-world settings = 24 个 cell，多数是 0。Vidar 在 Real Data Engine Task 1 拿到 40% 是 best case，CogVideoX 在 Action Planner Task 2 拿到 50% 是另一个 outlier。这说明 sim-to-real gap 在 functional level 极其严重——visual generation quality 几乎不 transfer 到 task success。

2. **TesserAct 的 strange pattern**: RoboTwin T1 几乎 0% 但 T2 35%，LIBERO 表现最好（34, 38），Real-world 0%。这种 extreme variance 暴露了 TesserAct 在某些 task distribution 上的 catastrophic overfitting。LIBERO 的高分可能是 procedural generation 偶然匹配了 TesserAct 训练分布。

3. **Vidar 是最 robust 的**: 在所有四个平台都有 non-zero 表现，特别是 Real Data Engine Task 1 40%——Vidar (https://arxiv.org/abs/2507.12898) 的 video diffusion + 3D scene representation 设计让它在跨平台 generalization 上领先。

4. **Wan 2.2 / CogVideoX 商业 general model**: Wan 2.2 在 RoboTwin 表现不错（15, 41, 12, 20）但 real-world 退化严重。CogVideoX (https://arxiv.org/abs/2408.06072) 在 Real Action Planner Task 2 突然跳到 50% 是个意外，可能是 Wipe Table 这种 task 对 visual fidelity 的依赖 > 对物理 accuracy 的依赖。

### 4.4 Cross-platform correlation 分析（Figures 6-9）

paper 分析了六维 16 metrics 的跨平台 Spearman/Kendall 相关性：

**强相关维度**（跨平台稳定）：
- Visual quality (Image, Aesthetic, JEPA similarity)
- Motion quality (Dynamic degree, Flow score, Smoothness)
- Physics adherence (Interaction quality, Trajectory accuracy)
- 3D accuracy (Depth, Perspectivity)

**弱相关维度**（domain-sensitive）：
- Content consistency (Subject, Background, Photometric consistency)
- Controllability (Instruction following, Semantic alignment, Action response sensitivity)

**Task success correlation**: 两个 simulator 之间正相关，但与 real-world 大幅下降。

**核心结论**: **simulation 数字 ≠ real-world 数字**。Visual fidelity 和 geometry reasoning 能 transfer，但 semantic alignment 和 instruction following 不能。这暗示一个 design 方向：world model 应该把 **physics engine-style 的 structure** 嵌进 generation process，而非纯 data-driven generation。

### 4.5 Table 4-9 视觉质量数据分析

我对比了三个平台的 video quality 数字，几个 standout：

- **Veo 3.1 / Wan 2.6**: 商业模型，visual quality / motion / instruction following 全面领先，但 trajectory accuracy 不一定高（Veo 3.1 在 RoboTwin trajectory accuracy 仅 0.1231，远低于 CtrlWorld 0.4766）。这印证 **visual fidelity ≠ dynamics fidelity**。

- **CtrlWorld**: 在三个平台的 trajectory accuracy 都领先（RoboTwin 0.4766, Real 0.6865），3D depth accuracy 也高（0.9300, 0.9888）。说明 CtrlWorld 的 controllable generation design 真的 encode 了物理 dynamics。

- **WoW** (https://arxiv.org/abs/2509.22642): physics adherence + content consistency 综合最强，是 task success 强相关 predictor。

- **iVideoGPT**: autoregressive 架构，JEPA similarity 高（0.9330 on RoboTwin）但 flow score 低——autoregressive 在 frame-level consistency 强但 motion smoothness 弱。

---

## 5. 综合直觉：WorldArena 2.0 想说什么

我把整套实验结果抽象成几条 principle：

**Principle 1: Visual fidelity is necessary but wildly insufficient for embodied deployment.**  
Veo 3.1 视觉最好但 task success 不一定高；WoW 视觉中等但 task success 强。Gap 在 physics encoding。

**Principle 2: Tactile modality unlocks short-horizon contact tasks but doesn't solve long-horizon control.**  
Wan 2.2 + tactile injection 让 Insert HDMI 100%，但 Lift Bottle 0%。Closed-loop force control 仍是 world model 的死穴。

**Principle 3: World model as RL env works, but reward model design dominates.**  
Proxy-based reward 显著优于 VLM-based 和 similarity-based。Reward model 是这条 pipeline 的 bottleneck，而非 world model 本身。

**Principle 4: Sim-to-real gap is real and large, especially at functional level.**  
Perceptual metrics 部分跨平台 transfer，task success 几乎不 transfer。Real-world evaluation 不可省略。

**Principle 5: General-purpose foundation model (Wan 2.2) > specialized embodied model (Vidar) on transferable representation, but specialized model wins on in-domain physics.**  
Wan 2.2 在 tactile alignment 上碾压 Vidar（因为 richer cross-modal priors），但 Vidar 在 Real Data Engine 上比 Wan 2.2 更稳（因为 in-domain physics priors）。

---

## 6. 我对 paper 的批判性思考

**优点**：
- 三轴扩展的 framing 非常 clean，modality × functionality × platform 是个真正的 Cartesian product 评估空间
- Standardized tactile injection pipeline 是 engineering 上的贡献，让现有 vision model 不重训就能加触觉
- Real-world 数据是稀缺且昂贵的，敢做就是 contribution
- 三种 reward model 的 ablation 是 actionable insight

**可质疑的地方**：
1. **Visuotactile 实验只 2 个 task，4 个 baseline**: Insert HDMI + Lift Bottle 数据点太少，无法 disentangle "tactile 帮助大" vs "短 horizon 任务 tactile 帮助大"。需要更多不同 horizon 的 contact-rich tasks。
2. **RL 实验只测了 Click Bell + Adjust Bottle**: long-horizon RL task 完全缺失。World model 在 10-step rollout vs 100-step rollout 的 failure mode 完全不同。
3. **Real-world 只测了 6 个 model，部分 cell 全是 0**: 数据点 sparse，难做统计 significant claim。
4. **AgileX ALOHA platform 只测了 pour water + wipe table**: 这两个 task 都是 deformable / fluid dynamics，是 generative model 的传统弱项，可能过度悲观。
5. **GRPO 在 world model 中的 compounding error 没有专门分析**: 这是 closed-loop RL 的核心问题，paper 只在 Figure 5 展示了曲线但没有 error decomposition。
6. **Tactile VAE 的 alignment quality 没有 ablation**: $\tilde{z}^T_t$ 是否真在 video latent space 内，alignment loss 是什么，没说清。

---

## 7. 你可能感兴趣的延伸方向

基于这篇 paper 的结论，我推测接下来 12 个月 embodied world model 的研究方向：

1. **Latent world model + tactile**: 把 visuotactile 做在 latent space（Dreamer-style），而非 pixel space，避免高维 tactile generation 难题。参考 Dreamer V3 + DIGIT tactile。
2. **Closed-loop short-horizon rollout in world model**: 不要一次 rollout 100 步，而是 rollout 5 步 + 真实环境 1 步 + world model 重置，混合 imagination 和 reality。
3. **Reward model scaling**: proxy-based reward 胜出暗示 reward model 应该 task-specific 微调，类似 RLHF 中的 reward model。
4. **3D/4D world model for sim-to-real**: 比如 4DWorldBench (https://arxiv.org/abs/2511.19836) 这条 line，把 4D scene representation 嵌入 generation。
5. **World action models**: 直接 predict action 而非 observation (MoTUS https://arxiv.org/abs/2512.13030, World Action Models https://arxiv.org/abs/2602.15922)，绕过生成高维 observation 的 cost。
6. **Differentiable physics + generative prior**: RoboScape (https://arxiv.org/abs/2506.23135) 这条 line，physics-informed loss + diffusion，是 CtrlWorld trajectory accuracy 高的根本原因。
7. **World model + VLA co-training**: π_0.5 已经是 VLA，WorldArena 2.0 把 π_0.5 当 base policy。下一代应该是 world model 和 VLA joint train，类似 Dreamer 的 actor-critic 在 imagination 里共同优化。
8. **Real-world evaluation scaling**: AgileX 之外可以加 Franka, UR5, Kuka 等不同 embodiment，测 cross-embodiment world model。

---

## 8. 关键 references

paper 主页: https://world-arena.ai  
UniVTAC: https://arxiv.org/abs/2602.10093  
RoboTwin 2.0: https://arxiv.org/abs/2506.18088  
LIBERO: https://libero-project.github.io/  
AgileX: https://www.agilex.ai/  
π_0.5: https://www.physicalintelligence.company/blog/pi05  
GRPO/DeepSeekMath: https://arxiv.org/abs/2402.03300  
Diffusion Policy: https://diffusion-policy.cs.columbia.edu/  
Dreamer V3: https://danijar.com/project/dreamerv3/  
VTAM: https://arxiv.org/abs/2603.23481  
Visuo-Tactile World Models (Meta): https://arxiv.org/abs/2602.06001  
OmniVTA: https://arxiv.org/abs/2603.19201  
WoW: https://arxiv.org/abs/2509.22642  
Vidar: https://arxiv.org/abs/2507.12898  
Ctrl-World: https://arxiv.org/abs/2510.10125  
RoboScape: https://arxiv.org/abs/2506.23135  
iVideoGPT: https://arxiv.org/abs/2410.20045  
IRASim: https://arxiv.org/abs/2406.14540  
WoVR: https://arxiv.org/abs/2602.13977  
CogVideoX: https://arxiv.org/abs/2408.06072  
Wan 2.2: https://arxiv.org/abs/2503.20314  
Cosmos: https://arxiv.org/abs/2511.00062  
World Action Models: https://arxiv.org/abs/2602.15922  
MoTUS: https://arxiv.org/abs/2512.13030  
WorldScore: https://arxiv.org/abs/2504.00983  
VBench: https://vchitect.github.io/VBench-project/  
4DWorldBench: https://arxiv.org/abs/2511.19836  
GigaWorld: https://arxiv.org/abs/2511.19861  
Genie Envisioner: https://arxiv.org/abs/2508.05635  
DIGIT tactile: https://arxiv.org/abs/2005.14097  
Mobile ALOHA: https://mobile-aloha.github.io/

---

## 9. 一句话总结

WorldArena 2.0 是一个 **诚实的 benchmark**——它告诉你 embodied world model 在 visual quality 上已经很好，在 in-simulator functional utility 上接近 simulator RL，在 real-world deployment 上还差一大截，在 long-horizon contact control 上是 dead。Tactile 是 short-horizon contact 的 silver bullet，closed-loop RL env 是 next frontier，real-world 是 final arbiter。任何 claim "我们的 world model 是 embodied AGI" 的工作，都得先过 WorldArena 2.0 这关。

如果你下次想动手做点东西，我会建议挑 **latent-space visuotactile world model + closed-loop RL with proxy reward + real-world eval on AgileX** 这个组合——这是 WorldArena 2.0 暴露的最 actionable 的 open problem。
