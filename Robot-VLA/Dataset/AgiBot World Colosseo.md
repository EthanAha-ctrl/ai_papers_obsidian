---
source_pdf: AgiBot World Colosseo.pdf
paper_sha256: 200a3029108f26c3408b827f4d45d16dc19d7468864fb5a46458172756c5e737
processed_at: '2026-07-18T06:14:02-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# AgiBot World Colosseo 深度解读

Hey Andrej, 这篇 paper 我读了之后感觉特别有"robotics 终于开始走 LLM 老路"的味道。我把它当成一个 data + model co-design 的 case study 来拆给你看，重点 build 一下 intuition。

---

## 1. 这篇 paper 在 robot learning 谱系里的定位

先把它放在你熟悉的 scaling 谱系里看：

| 维度 | Open X-Embodiment (OXE) | DROID | AgiBot World |
|---|---|---|---|
| Trajectories | 1.4M (聚合) | 76k | **1M+** (统一采集) |
| Hours | ~2000h | ~101h | **2976h** |
| Skills | 217 | 86 | 87 |
| Scenes | 311 | 564 | 106 (但1:1复刻) |
| Arm type | single+dual | single | **dual (humanoid)** |
| Dex hand | ✗ | ✗ | **✓** |
| Failure recovery | ✗ | ✗ | **✓ (~1%)** |
| Human-in-the-loop | ✗ | ✗ | **✓** |
| Trajectory 长度 | <5s 主导 | 5-20s | **30-60s, 部分超120s** |

直觉上，OXE 像 "Internet text scraped and dumped"，DROID 像 "crowd-sourced Reddit data"，AgiBot World 则像 "Common Crawl + RLHF + human-annotated corpus"——后者刻意把 data quality loop 闭起来。这一点我后面会重点讲，因为这是 robot data scaling 真正卡住的地方。

Project page: https://agibot-world.com/
Code: https://github.com/OpenDriveLab/AgiBot-World

---

## 2. Hardware：为什么"统一 embodiment"很重要

他们用的是自家 AgiBot G1 humanoid：
- dual 7-DoF arms + mobile base + adjustable waist
- end effector 可换：standard gripper / 6-DoF dexterous hand / visuo-tactile gripper
- **8 cameras**：1 front RGB-D + 3 front fisheye + 2 wrist-mounted + 2 rear fisheye
- control freq **30 Hz**，记录 joint position + EEF position

直觉上的关键点：OXE 最大的问题是 embodiment heterogeneity——你把 22 种 robot 的数据 pool 在一起，policy 要学一个"universal action"几乎不可能，因为 action space 物理含义都不同。AgiBot 用 100 台 homogeneous robot 采集，相当于把"tokenizer 统一"这件事在硬件层就解决了。这是 LLM 没遇到过的问题——text 的 token 天然统一，robot 的"token"必须靠硬件统一。

teleoperation 两条路：
1. **VR headset**：hand gesture → EEF translation/rotation → IK 解到 joint angle。问题：dexterous hand 只能 few preset gestures
2. **Whole-body motion capture**：直接 capture 人体关节（含手指）→ map 到 robot posture，解锁 individual finger movement + torso + head

后者对应了他们 long-horizon + dexterous 任务的可行性。我猜 motion capture 的 latency 和 mapping residual 是隐藏的工程难点，paper 没展开。

---

## 3. Data Collection Pipeline：human-in-the-loop 是核心 contribution

三阶段流程（Fig. 2）：

**Phase 1 - Preliminary**：先小批量采集验证 task feasibility，建立 collection standard。这一步类似 "spec the dataset"。

**Phase 2 - Formal Collection**：skilled teleoperator 按 standard 采集 → 本地 valid check (no missing frame) → upload to cloud。

**Phase 3 - Post-processing**：annotator 核对每个 episode + 加 language annotation (task-level + sub-step-level)。

### 3.1 Failure Recovery Data（~1%）

这里 intuition 很关键：传统 BC dataset 把失败 trajectory 直接丢掉，但这相当于把"失败 → 反思 → 恢复"这一段 valuable behavior 扔了。AgiBot 保留这些 trajectory 并标 timestamp + failure reason，专门留给 policy alignment (类似 [GRAPE](https://arxiv.org/abs/2411.19309)) 和 failure reflection (类似 [Reflexion](https://arxiv.org/abs/2303.11366))。

类比 LLM：这就是 RLHF 里 "rejected response" 的角色——policy 不仅要学"做什么"，还要学"不做什么 / 怎么从坏状态 recover"。

### 3.2 Human-in-the-loop 的真正含义

这里我要强调，他们的 HITL 不是简单事后标注，而是 **policy-in-the-loop data curation**：

> 收一小批 demo → train policy → deploy → 看 policy 在哪里挂 → 反过来修 collection protocol

paper 给的具体例子：policy 在 action 起始处会"pause 很久"。annotator 反馈说 demo 里有 inconsistent transition + excessive idle frame。修复方法：(1) 改 collection protocol，(2) post-processing 里 cut idle frame。

这个 loop 在 LLM 里对应 "iterative data quality improvement via SFT-RLHF-SFT cycle"。Robotics 之前没人系统做这件事，因为成本太高——你需要 100 台 robot 和一个 4000 m² facility 才能闭得上 loop。

---

## 4. Dataset 统计的深层直觉

### 4.1 Long-horizon 的物理含义

Fig. 3(b) 显示 OXE 大部分 trajectory <5s，DROID 5-20s，AgiBot World 30-60s，部分 >120s。

直觉上，<5s 的 trajectory 基本是 "pick → place" 这种 single atomic skill。30-60s 意味着 trajectory 里串了 5-10 个 atomic skill（比如 "make coffee" = grasp cup → place under machine → press button → wait → grasp cup → pour → serve）。

这对 policy learning 的含义：
- **action chunking 必须更长**：single-step chunk 学不到 sub-task 切换
- **需要 sub-goal / latent plan**：直接从 image → 30s 的 low-level action 几乎不可能泛化
- **language conditioning 必须分层的**：一句 "make coffee" 要能被 decompose 成 sub-instruction

这三个点直接 motivate 了 GO-1 的 ViLLA 架构。

### 4.2 Skill Distribution 的"长尾"刻意保留

Fig. 3(c) 显示大部分 atomic skill 至少有 100 trajectories（红色虚线）。他们刻意保留了 "chop"、"plug" 这种 less frequent 但 valuable 的 skill。

这跟 LLM pre-training data 的 power-law distribution 思路不一样——LLM 让长尾自然出现，AgiBot 是人为 floor 住长尾。直觉上这是因为 robot policy 对 rare skill 的 catastrophic forgetting 更严重，必须人为保底。

---

## 5. GO-1：ViLLA 架构（核心 contribution）

这部分我重点拆一下，因为这是这篇 paper 真正的 algorithmic novelty。

### 5.1 范式对比

- **VLA (OpenVLA, RT-2)**：image+text → 直接 predict discretized action token
- **ViLLA (GO-1)**：image+text → predict **latent action token** → condition action expert → low-level action

直觉上，VLA 把 action 当成"另一种 language token"直接 MLM 出来，问题是 action space 远比 vocab 大且连续，discretization 丢精度。ViLLA 在中间插了一层 "latent plan"，相当于把"high-level intent"和"low-level motor control"解耦——这跟人类 cognitive architecture 里 "system 2 planning + system 1 motor execution" 的分离很像。

跟 [LAPA](https://arxiv.org/abs/2410.11758) 的关键区别：LAPA 也用 latent action pre-training，但 latent planning capability 没保留到 downstream。GO-1 把 latent planner 作为 first-class citizen 保留下来，并且和 action expert joint train。

### 5.2 三阶段训练

**Stage 1: Latent Action Model (LAM) on web-scale video**

数学上：
- Input: consecutive frames $\{I_t, I_{t+H}\}$（H 是 horizon）
- Encoder: $\mathbf{I}(z_t \mid I_t, I_{t+H})$ — inverse dynamics，给两帧推 latent action
- Decoder: $\mathbf{F}(I_{t+H} \mid I_t, z_t)$ — forward dynamics，给初始帧 + latent action 预测未来帧
- $z_t = [z_t^0, \dots, z_t^{k-1}]$，**k=4** 个 discrete latent action tokens
- 通过 VQ-VAE objective 量化，codebook size $|C|$
- Encoder 是 spatial-temporal transformer with causal temporal mask（参考 [Genie](https://arxiv.org/abs/2402.14044)）
- Decoder 是 spatial transformer

变量直觉：
- $I_t$：第 $t$ 帧图像
- $I_{t+H}$：未来第 $t+H$ 帧图像（H 是预测 horizon）
- $z_t \in \mathbb{R}^{k \times d}$：latent action，描述"从 $I_t$ 到 $I_{t+H}$ 之间发生了什么动作"
- $z_t^i$：第 $i$ 个 latent action token，$i \in \{0, \dots, k-1\}$
- $|C|$：VQ codebook 大小

**Stage 1 的关键 intuition**：用 Ego4D 这种 web-scale human video，**没有 action label**，但通过 inverse dynamics 可以"自动发现"latent action。这就把 robot pre-training 的 data pool 从 "1M trajectory" 扩到 "web-scale video"。这是把 LLM 式 web pre-training 引入 robot 的真正突破口。

**Stage 2: Latent Planner**

- VLM backbone：**InternVL2.5-2B**（[paper](https://arxiv.org/abs/2412.05271)）
- Multi-view input：$(I_t^h, I_t^l, I_t^r)$ — head cam + left wrist + right wrist
- Language instruction：$l$
- Predict：$\mathbf{P}(z_t \mid I_t^h, I_t^l, I_t^r, l)$
- Supervision：来自 LAM encoder 的 $z_t := \mathbf{I}(I_t^h, I_{t+H}^h)$（用 head view）
- Latent planner 是 24 层 transformer，full bidirectional attention，layer-by-layer condition 在 VLM 上

直觉：
- VLM 已经在 web-scale image-text 上 pretrain 好，有 scene understanding + reasoning
- Latent planner 在上面加一层"在 latent action space 上做 planning"
- 比 OpenVLA 直接 discretize low-level action 强，因为 latent action space 小几个数量级 → MLM 更高效

**Stage 3: Action Expert**

- 同样的 transformer 架构
- 但用 **diffusion objective** 建模 continuous low-level action distribution
- 输出 action chunk：$A_t = [a_t, a_{t+1}, \dots, a_{t+H}]$，**H=30**
- 条件：$\mathbf{A}(A_t \mid I_t^h, I_t^l, I_t^r, p_t, l)$，$p_t$ 是 proprioceptive state
- 和 latent planner hierarchical conditioning

变量直觉：
- $a_t$：第 $t$ 步的 low-level action（joint torque/velocity）
- $p_t$：第 $t$ 步的 proprioception（自身 joint state）
- $A_t$：长度 30 的 action chunk，对应 1 秒 @ 30Hz
- diffusion denoising process 条件于 latent planner 输出的 $z_t$

**Inference 时**：VLM → latent planner → $z_t$ → action expert (diffusion denoising) → 最终 control signal。

直觉上这就是 "high-level plan condition low-level motor execution"，类似 [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) + latent plan 的组合。和 [π0](https://www.physicalintelligence.company/blog/pi0) 的 flow matching action expert 思路接近，但 π0 没有 explicit latent plan layer。

---

## 6. Experiments：数据 scaling 和 quality 两个 ablation 最值得看

### 6.1 AgiBot World vs OXE pretrain（Fig. 6）

RDT model：
- in-distribution：AgiBot World pretrain **0.77** vs OXE 0.47
- out-of-distribution：AgiBot World **0.67** vs OXE 0.38
- Table Bussing 任务：AgiBot World 接近 3x performance

最 striking 的点：**AgiBot World alpha（仅 14% 数据，236h）就比 OXE（~2000h）高**。这强烈表明 robot data scaling 不是 simple power law on hours，data quality + embodiment consistency + task diversity 的权重远大于 raw hours。这跟 LLM 的 Chinchilla scaling 不完全一样，robot data 的 "information density" 信号更强。

### 6.2 GO-1 vs RDT vs π0（Fig. 5）

6 个任务：
- Restock Bag
- Table Bussing
- Pour Water（最考验 position robustness）
- Restock Beverage（最考验 instruction following）
- Fold Shorts（deformable + long-horizon）
- Wipe Table（tool use + tactile）

GO-1 比 RDT 平均 +32%，比 π0 也有显著 margin。Latent planner 平均贡献 +0.12 completion score。

直觉上，latent planner 在 "Fold Shorts" 这种 long-horizon + deformable 任务上贡献最大，因为这个任务需要 sub-goal reasoning（"fold left first, then right, then bottom up"），low-level diffusion policy 单独学不到这个 high-level structure。

### 6.3 Data Scaling Law（Fig. 7a）

- 10% alpha (9.2k) → 100% alpha → beta (1M)
- Pearson **r = 0.97** 的 log-log power law

直觉：robot data scaling 真的存在 power law，但前提是 data quality controlled。这跟 [Lin et al.](https://arxiv.org/abs/2410.18665) ICLR 2025 的"data scaling laws in imitation learning"一致。Karpathy 你应该会喜欢这个 figure，因为它证明了 robot 也能走 LLM 老路。

### 6.4 Data Quality Ablation（Fig. 7b）

Wipe Table 任务 fine-tune RDT：
- 528 verified trajectories → score X
- 482 unverified trajectories → score X - 0.18

**少 10% 但 verified 的 data 反而高 0.18 score**。这个结论对应到 LLM 里就是 "LIMA 1k high quality > 10k random"。Robot data 的 quality 信号比 LLM 更强，因为 robot 没有任何"间接监督信号"（不像 LLM 有 next-token 这种天然 dense supervision），一条坏 trajectory 会直接把 policy 带歪。

---

## 7. 我对这篇 paper 的几个 concern / 联想

### 7.1 latent action 的 codebook 大小是关键超参

paper 里只说 "codebook of size $|C|$" 但没给具体数值。这个值太大 → latent action 退化成 raw action，VLM 没法 efficient predict；太小 → information loss 太多，action expert 学不出来。Genie paper 用的也是 4 token × codebook，我猜他们 follow 了这个 setting。这是后续可解释性研究的关键 handle。

### 7.2 30Hz × H=30 chunk 的物理含义

每 chunk = 1 秒 action。如果 task horizon 60s，整条 trajectory 需要 60 次 plan + denoise。Latency 累积是不是 deployment bottleneck？paper 没给 inference latency。我猜他们做了 action chunk overlap（类似 [ACT](https://tonyzhaozh.github.io/aloha/)），但没明说。

### 7.3 Failure recovery 只占 1%

1% 听起来太少。直觉上 RLHF 里 rejected response 占比远高于 1%。我担心这 1% 的 failure recovery 数据不足以让 policy 学到 robust recovery behavior。后续可能需要主动 collect failure-heavy dataset（类似 [RH20T](https://rh20t.github.io/) 的 failure subset）。

### 7.4 motion capture teleop 的 IK gap

paper 没讨论 motion capture → robot mapping 的 kinematic mismatch（人手 vs 6-DoF dex hand）。我猜这是为什么 dex hand task success rate 还没 saturate 的原因之一。后续如果 dex hand DoF 提升到 16+，这个 mapping 会更复杂。

### 7.5 与 OpenVLA/OpenPI 对比缺失

paper 主要对比 RDT 和 π0，但 OpenVLA 是 single-arm 的，理论上不可比。这点需要在 community 里 push 一下，让 leaderboard 更公平。

### 7.6 真正的 scaling test：10x data → 10x robot

现在 100 robot + 4000 m² facility 已经是天文成本。下一个 10x scaling 需要 1000 robot + 多 facility coordination，这是 capitalism-level problem 而不是 research-level problem。Karpathy 你之前讲过 compute 是 LLM 的 bottleneck，对 robot 来说 bottleneck 是 **physical data collection infrastructure**。

---

## 8. 这篇 paper 对 robot learning 的 meta-level 启示

### 8.1 Robot learning 正在重演 LLM 历史

- 2020-2022 robot：single-task BC，对应 LLM 的 pre-GPT 时代
- 2023 OXE：dataset aggregation，对应 C4/Common Crawl
- 2024 DROID：crowd-source，对应 OpenAssistant
- 2025 AgiBot World：human-in-the-loop + homogeneous embodiment + standardized pipeline，对应 InstructGPT+RLHF 的 curated data 时代
- 下一步：robot RLHF/DPO + internet-scale video pretrain，对应 GPT-4 时代

### 8.2 ViLLA 范式可能成为主流

VLA 的 bottleneck 是 action space 太大、太连续。ViLLA 把 "language → latent action → continuous action" 三层解耦，相当于把 LLM 的"concept → token → phoneme"层级结构搬到 robot。我个人 bet 这种 hierarchical 范式会在 2026 年成为 default。

### 8.3 Failure recovery + reflection 是下一个 research gold mine

AgiBot 把 failure data 保留并标注，但只用了一点点。结合 [Reflexion](https://arxiv.org/abs/2303.11366)、[Voyager](https://arxiv.org/abs/2305.16291)、[GRAPE](https://arxiv.org/abs/2411.19309) 的思路，下一步应该是让 policy 在 deployment 时遇到 failure 自动 trigger verbal reflection + recovery plan。这需要 VLM + latent planner + memory module 联合训练。

---

## 9. 一些有用的 reference links

- AgiBot World project: https://agibot-world.com/
- AgiBot World GitHub: https://github.com/OpenDriveLab/AgiBot-World
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- DROID: https://droid-dataset.github.io/
- RDT-1B: https://thu-rdol.github.io/RDT/
- π0 (Physical Intelligence): https://www.physicalintelligence.company/blog/pi0
- LAPA (Latent Action Pretraining): https://latentactionpretraining.github.io/
- Genie (DeepMind): https://sites.google.com/view/genie-interactive
- VQ-VAE: https://arxiv.org/abs/1711.00937
- InternVL2.5: https://github.com/OpenGVLab/InternVL
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- OpenVLA: https://openvla.github.io/
- LeRobot (HuggingFace): https://github.com/huggingface/lerobot
- RoboMIND: https://github.com/Zhidong-Robo/RoboMIND

---

## 10. 一句话总结直觉

AgiBot World Colosseo 真正在做的事是 **把 LLM 的 data-engineering + hierarchical representation learning 范式移植到 robot manipulation**——硬件统一解决"tokenizer"问题，human-in-the-loop 解决"RLHF data quality"问题，LAM 解决"web-scale pretrain without action label"问题，latent planner + action expert 解决"high-level reasoning + low-level motor control 解耦"问题。这条路径如果跑通，robot foundation model 的 scaling law 会跟 LLM 越来越像，但 physical data collection infrastructure 的成本会一直是 unique bottleneck。

Karpathy 你之前在 [Intro to LLMs](https://www.youtube.com/watch?v=zjkBMFhNj_g) 里讲过 "the bitter lesson" 在 LLM 上的体现，我感觉 robot learning 正在写自己版本的 bitter lesson——只不过这次 compute 不是唯一变量，**data collection pipeline 本身就是一个 slow-iterating physical compute**。

你看看哪些点想再 deep dive，我可以再展开 LAM 的 VQ-VAE 训练细节、latent planner 的 bidirectional attention 设计动机、或者 action expert 的 diffusion schedule 选择。
