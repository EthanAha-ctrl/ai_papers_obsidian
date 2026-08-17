---
source_pdf: DREAMGEN.pdf
paper_sha256: 4621edafc6d4035c6c084fc1275f19e4a3087e60801b893e825d1ccb3b7e4bd4
processed_at: '2026-08-03T23:24:24-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DREAMGEN 人话版

---

## 一句话概括

让机器人 **做梦** —— 用 video generation 模型给机器人"想象"出自己干活儿的视频，再从这些梦里反推出动作，拿来训练 robot policy。

---

## 为什么需要这个

现在训练 robot 干活儿，标准做法是人拿着遥控器，一遍一遍 teleoperate robot 做 task。一条 trajectory 要几分钟，一个 task 要几百条，要 generalize 到新环境新 task 又要重新采。

**瓶颈**：人慢、贵、累。一个 lab 一周采几百条 data 已经很猛了。

DREAMGEN 说：别采了，让 video model 帮你"编"几十万条出来。

---

## 四步走，特别简单

### 第 1 步：教会 video model "你长这样"

拿一小撮人 teleoperate 的 video（比如 1200 条 pick-and-place），用 LoRA 微调一下 video generation model（WAN2.1 之类）。

目的：让 video model 学会"这个 robot 长这样、关节这样动、gripper 这样开合"。

不用改 architecture，LoRA rank 4 就够——因为 robot 相对于 internet video 是个 low-rank 偏移。

### 第 2 步：让它做梦

给一张初始图 + 一句话（比如 "pour water into the cup"），让 fine-tuned video model 生成一段 robot 做这个动作的 video。

**神奇之处**：你只 fine-tune 了 pick-and-place，但你说 "iron shirt"、"use vacuum"、"water flowers"，它居然能生成出 robot 做这些事的 video。

为什么 work？因为 video model 在 internet 上见过几十亿段人类和物体交互的 video，它知道 "pour" 这个 verb 视觉上长啥样。fine-tune 加的是 robot 的身体知识，没覆盖掉这些 verb prior。两套知识在 model 里 **共存**。

### 第 3 步：从视频里抠出动作

video model 只给你 video，没给你 action（关节角、gripper 开合）。

两个办法抠：

**办法 A（IDM）**：训一个小模型，喂它两帧 video（当前帧 + 未来帧），让它预测中间 robot 应该执行什么 action。训这个模型用的是 real teleop data（video + 真实 action pair）。

**办法 B（LAPA）**：训一个 latent action model，看两帧 video 之间的 visual 变化，编码成一个 latent vector。不需要 real action，纯视觉。

**结果**：video + action 配对，叫 **neural trajectory**。

### 第 4 步：拿 neural trajectory 训 policy

就跟普通 imitation learning 一样，拿 neural trajectory 当 dataset 训 policy（Diffusion Policy、π0、GR00T N1 都行）。

一个 hack：neural trajectory 没有真实 proprioceptive state，就把 state 填 0。

可以纯用 neural trajectory 训，也可以和 real trajectory 混着训（1:1 ratio）。

---

## 为什么这事 work

关键 insight 是 **video model 学到的东西和 robot 专有的东西是可分离的**：

- Internet video 教 model：physics 怎么样（液体怎么流、布料怎么折）、verb 视觉上长啥样（pour、wipe、hammer）
- Fine-tune 教 model：这个 robot 长什么样、关节怎么动、gripper 怎么开

这两套知识 factorize 得很干净。所以你 prompt 一个新 verb，model 能"想象"出 robot 做新 verb 的 video；你 prompt 一个新环境的初始帧，model 能在新环境里想象 robot 干活儿。

---

## 结果有多炸

### Data augmentation

用 10 条真实 data + 几百条 neural trajectory：
- GR1 humanoid 4 个 task：37% → **46%**
- Franka 3 个 task：23% → **37%**
- SO-100 2 个 task：21% → **45%**

### Behavior generalization

只 teleop 了 pick-and-place，训完 policy 能做 14 个全新 verb（pour water、open microwave、iron shirt、use vacuum……）：11.8% → **43.2%**

### Environment generalization

只在 1 个 lab 环境采过 data，在 10 个新环境干活儿：0% → **28.5%**

### Scaling law

neural trajectory 从 0 加到 24 万条，policy performance **log-linear 上升**，没饱和。这在 robot learning 里从来没见过——以前只有 teleop data 量，从来没出现过"合成 data scaling"。

---

## 这事像什么

像 LLM 里用 GPT-4 生成 synthetic data 训 GPT-3.5。但这里有个 twist：**没有更强的 robot model 来生成 data**。video world model 本身扮演了那个 "stronger model"——它有 internet 全量 visual+physical prior，比你的 robot policy 强太多了。

所以这是一个 robot learning 里全新的 **data scaling axis**：以前是 "more teleop"，现在是 "more dreams"。

---

## 代价

240k video samples 花了 54 小时 × 1500 张 L40 GPU。贵。但 GPU 时间可 scale，teleop data 不可 scale。这是质的区别。

初始帧还要人手动拍，没完全自动化。Evaluator 用 7B VLM，会 hallucinate。任务还比较简单，short-horizon 单 verb。

---

## 一句话直觉

**Video world model 是 robot 的 "imagination engine"**。你教会它"你长这样"，它就能用 internet 上学到的所有物理和动词知识，想象出你在任何环境做任何事的样子。你把这些想象翻译成 action，就成了无穷无尽的 training data。

这件事一旦 work，robot learning 的 bottleneck 从"采数据"变成"生视频"，从人瓶颈变成 GPU 瓶颈。GPU 可 scale，人不可 scale。这就是这篇 paper 的意义。

---

# DREAMGEN: 用 Video World Models 解锁 Robot Learning 的 Generalization

Andrej，这篇 paper 我读了三遍，最让我激动的不是单点突破，而是它把几个之前被分开处理的问题用一条 elegant pipeline 串起来了：**how do you get a robot to do things it has never been teleoperated to do, in places it has never been, by essentially "dreaming" videos of itself?**

我会从 design philosophy、数学细节、architectural intuition、实验解读、以及为什么这个方向有可能是 robot learning 的下一个 "data scaling" axis 这几个层面来讲。

---

## 1. Design Philosophy: 把 World Model 当 Data Generator, Not Real-time Planner

之前的工作（[Du et al. 2023 UniPi](https://arxiv.org/abs/2302.00638), [Video Language Planning](https://openreview.net/forum?id=9pKtcJcMP3), [RoboDreamer](https://arxiv.org/abs/2404.12377)）大多把 video world model 当作 **online planner**：每一步 query model 生成下一步，然后用 IDM 提取 action，循环执行。这条路有几个 hard problems：inference latency、long-horizon drift、video generation 不 deterministic 导致的累积误差。

DREAMGEN 把它彻底反过来：**video world model 是 offline data factory**。一次性生成成千上万条 video，离线提取 action，然后当成普通 dataset 训任意下游 policy。这把问题解耦了：
- video model 可以很慢很贵（offline 生成）
- policy 可以很快很小（inference 在 robot 上）
- 两边可以独立迭代

这个 decoupling 像极了 RL 中把 model-based rollout 和 policy learning 分开做的好处，但又避开了 RL 中 value function 估计的难题。

---

## 2. 4-Stage Pipeline 的细节

### Stage 1: Fine-tune Video World Model

**Base model**: WAN2.1（多数实验），也测了 Cosmos, CogVideoX, Hunyuan。

**关键 trick**: LoRA rank 4, alpha 4, LR 1e-4。这个 rank 4 非常小——意味着作者认为 robot domain shift 在 internet video pretraining 的 manifold 上是一个 low-rank 偏移。这也合理：robot 的 visual statistics（金属质感、特定 embodiment、特定光照）+ motion statistics（gripper motion）相对于整个 internet video distribution 确实是低维 shift。

**Multi-view 处理**: RoboCasa 和 DROID 有多视角，作者直接把 cameras 拼成 2×2 grid（左相机、右相机、wrist 相机，加一格黑），让 video model 在 pixel space 学到 cross-view consistency。这是个很 pragmatic 的 hack，避免了改 model architecture。

**两个监控 metric**：
- **Instruction Following (IF)**：video 内容是否匹配 language instruction
- **Physics Following (PF) / Physics Alignment (PA)**：video 是否遵循物理（不穿模、不瞬移）

paper 中明确说 "optimal amount of fine-tuning required for each video world model and fine-tuning data pair differed"——意思是 fine-tune 太少 → 不能 follow robot dynamics；fine-tune 太多 → 忘掉 internet 知识（比如 "pour" 这个动词的 visual prior）。LoRA 是个有效缓解，但 epoch 数仍要手调。

### Stage 2: Video Rollout

给定 $(o_0, \ell)$ —— 初始帧 + 语言指令，让 model 生成 video rollout $V = \{o_0, o_1, \dots, o_T\}$。

关键 design choices：
- **Initial frames 的随机化**：手动拍新 initial frame，object 位置随机化
- **对于 environment generalization**：只 fine-tune 在单一 environment，但 rollout 时给 new environment 的 initial frames
- **对于 behavior generalization**：手工构造 novel behavior prompts（"pour water", "iron shirt", "use vacuum" 等 14 个）

这里有个非常 remarkable 的 empirical finding：fine-tune 在 pick-and-place 上，prompt 说 "water flowers"，video model 居然能 generate 出 GR1 robot 在执行 pouring motion 的视频。这说明 **internet video prior 没被 fine-tune 完全覆盖**，而是和 embodiment kinematics 通过某种 factorized 方式 coexist。

### Stage 3: Pseudo Action Labeling

这是 paper 的技术核心之一。两条路：

#### (a) Inverse Dynamics Model (IDM)

**Architecture**: Diffusion Transformer + SigLIP-2 vision encoder + flow matching objective。

输入：$(o_t, o_{t+H})$ —— 当前帧和 future 帧（间隔 H 步）
输出：$\hat{a}_{t:t+H}$ —— action chunk

Flow matching 的 loss 形式（paper 没给完整公式，我根据上下文重建）：

$$\mathcal{L}_{\text{IDM}} = \mathbb{E}_{t, o_t, o_{t+H}, a_{t:t+H}, \epsilon \sim \mathcal{N}(0, I)} \left\| v_\theta\big((o_t, o_{t+H}),\, a_t^{\tau},\, \tau\big) - (a_{t:t+H} - \epsilon) \right\|_2^2$$

其中：
- $v_\theta$ 是 velocity field（DiT 网络，参数 θ）
- $a_t^{\tau} = (1-\tau) \epsilon + \tau \cdot a_{t:t+H}$ 是线性插值的 noisy action
- $\tau \in [0, 1]$ 是 flow time
- $\epsilon$ 是从 prior（通常是 $\mathcal{N}(0, I)$）采样的 noise
- $(o_t, o_{t+H})$ 是 conditioning（通过 SigLIP-2 编码后 cross-attention 进 DiT）

**关键 design**: **no language, no proprioception 输入**。作者明确说："we want the IDM model to only capture the dynamics of the robot"。这是个干净的设计——IDM 只回答"如果我现在 frame A、之后 frame B、robot 应该执行什么 action"。这让 IDM 可以跨 task 复用，不被 task-specific language bias 污染。

**Sliding window inference**：模型一次预测 H 步 action chunk，然后滑动一格再预测下一段。这意味着 overlap region 有多个预测——paper 没明说怎么 fuse（大概率是直接用 first prediction，类似 receding horizon）。

公式上：
$$\hat{a}_{t:t+H} = \text{IDM}(o_t, o_{t+H})$$
$$\hat{a}_{t+1:t+1+H} = \text{IDM}(o_{t+1}, o_{t+1+H})$$
$$\dots$$

最后形成完整 trajectory $\hat{a}_{0:T}$。

#### (b) LAPA Latent Action Model

**Architecture**: Transformer encoder-decoder + VQ-VAE objective。

**Pre-training data**: paper Table 3 列了一个庞大的 mixture——438M frames、5721 小时，包含：
- GR-1 teleop: 6.4M frames
- DexMG: 4.4M frames  
- DROID: 23.1M frames
- RT-1: 3.7M frames
- Language Table: 7.0M frames
- Bridge-v2: 2.0M frames
- RoboCasa: 19.3M frames
- Agibot-Alpha: 213.8M frames
- Sth-v2 + Ego4D: 158.4M frames (人类视频)

**VQ-VAE objective** 形式：

$$z = E_\phi(o_t, o_{t+\Delta t}) \quad \text{(encoder 输出 continuous latent)}$$
$$\hat{z} = \text{Quantize}(z, C) \quad \text{(在 codebook C 上量化)}$$
$$\hat{o}_{t+\Delta t} = D_\psi(o_t, \hat{z}) \quad \text{(decoder 重建 future frame)}$$

$$\mathcal{L}_{\text{VQ}} = \underbrace{\|o_{t+\Delta t} - \hat{o}_{t+\Delta t}\|_2^2}_{\text{reconstruction}} + \underbrace{\|\text{sg}[z] - \hat{z}\|_2^2}_{\text{codebook loss}} + \underbrace{\beta \|z - \text{sg}[\hat{z}]\|_2^2}_{\text{commitment loss}}$$

- $\text{sg}$ 是 stop-gradient（torch 中的 `.detach()`）
- $\beta$ 是 commitment loss 权重
- Codebook size = 8, sequence length = 16（很小的 codebook！）
- Latent action = pre-quantized continuous embedding（不是离散 token）

**Paper 用 pre-quantized continuous embedding 作为 latent action**，这其实是 follow GR00T N1 的做法。continuous embedding 比 discrete token 保留更多信息，适合 VLA-style policy 直接当 action space。

**为什么 LAPA 和 IDM 都给了**？因为两者各有利弊：
- LAPA 不需要 target robot 的 ground-truth action（只看 visual delta），适合 cross-embodiment generalization
- IDM 给的是真 action space，可以 solely train policy on neural trajectories 而无需 co-training
- Paper 最后 main experiments 偏向 IDM，因为 IDM actions 可以直接和 real trajectory 1:1 比较、可以 replay 到 simulation 中验证

### Stage 4: Policy Training on Neural Trajectories

给定 $o_t$（image observation）和 $i_t$（task instruction），train policy $\pi_\theta$ 输出 $\hat{a}_{t:t+H}$。

**关键 hack**: state information 填 zero。因为 neural trajectory 没有真实 proprioceptive state（关节角、速度等），只能 zero out。这听起来很糟糕，但作者发现效果还行——可能因为 VLA policy 主要依赖 visual observation，state 是 auxiliary signal。

**Two training regimes**:
1. **Co-training**：neural trajectory 和 real trajectory 用 1:1 sampling ratio 混合。对 GR00T N1，两种 trajectory 用 **separate action encoder 和 decoder**，当成两个 embodiment 处理。这是个 key insight——把 IDM action space 和 real action space 视为不同 modality，让 policy 学一个映射函数。
2. **Only neural trajectory**：完全用 IDM actions 训练，不需要 real data。这测了 24 个 RoboCasa tasks，达到 20.6% average success。

### Test 3 个 policy 架构

- **Diffusion Policy** (Chi et al.): 用 diffusion 生成 action，CNN-based
- **π0** (Black et al.): VLA flow model
- **GR00T N1** (Bjorck et al.): NVIDIA 的 humanoid foundation model

---

## 3. 实验结果的关键 Insight

### 3.1 Log-linear Scaling in RoboCasa (Figure 4)

这是 paper 最 strong 的 empirical finding：随着 neural trajectory 数量增加（从 0 → 240k，相当于 333× 的原始 data），下游 policy performance 在 log-linear 关系上提升。

类比 LLM 的 scaling law：$\mathcal{L}(N) \propto N^{-\alpha}$。这里相当于 success rate $S$ 和 neural trajectory count $N$ 满足：

$$S(N) = S_0 + c \cdot \log(N/N_0)$$

在不同 ground-truth data 量（low 720 / mid 2.4k / high 7.2k）下都成立。这说明：
1. Video world model 作为 data generator **没有饱和**
2. 真实 data 是 multiplicative boost，不是 additive
3. 这种 scaling 在 robot learning 中以前从没见过——以前都是要更多 teleop data

**Compute cost**: 240k samples × 54 小时 × 1500 L40 GPUs。这意味着 log-linear slope 暂时还很贵，但 GPU 时间是可 scale 的，teleoperated data 不可 scale。

### 3.2 Real-world Data Augmentation (Figure 5)

9 个 real-world tasks，跨 3 个 embodiments：

| Robot | Tasks | Baseline avg | + Neural Traj. |
|---|---|---|---|
| GR1 humanoid | Hammering, Wiping, Folding, Stacking | 37.0% | 46.4% |
| Franka | Pick&Place, Cube Stacking, Tool Use | 23.3% | 37.0% |
| SO-100 | Strawberry, Tic-Tac-Toe | 21.0% | 45.5% |

每个 task 只用 10-13 条 real trajectory + 100-300 条 neural trajectory。

**为什么 GR00T N1 涨幅最大（37→46）而 DP 涨幅最小（22→27）**？paper 的假设是 GR00T N1 的 separate action encoder/decoder 设计让它能更好地吸收 zero-state 的 neural trajectory，而 DP 把 state 强行 zero out 后 representation 受损。

我自己再加一个 hypothesis：GR00T N1 是 VLA 模型，pre-training 见过海量 internet video-text pairs，对 "dreamed" trajectory 的 visual distribution shift 更鲁棒。DP 没有 language pretraining，更依赖 data distribution 一致性。

### 3.3 Behavior Generalization (Table 1)

GR1 fine-tune 在 2,884 条 pick-and-place trajectory（单一 verb），prompt novel behavior：

- Baseline (GR00T N1 + pick-and-place only): **11.8%** on 14 novel tasks（部分得分来自 partial credit，比如 Pour Water 任务中"拿起瓶子"给 0.5）
- + Neural Trajectories: **43.2%**

22 个 novel behavior 包括：pour, open microwave, open macbook, close lunchbox, hit tambourine, hit keyboard, grab button, water flowers, light candle, use vacuum, iron shirt, take spoon out, unroll mat, move mouse 等。

这相当于 zero-to-one 的 generalization——pick-and-place policy 在这些 verb 上完全不行（0% pure success），但通过 video world model 的 "imagination"，policy 学会了从未 teleoperated 过的 motion primitive。

### 3.4 Environment Generalization (Table 1)

Single environment (lab) fine-tune，10 个 new environments 做 zero-shot rollout：
- Seen behavior (6 tasks): **28.5%** vs baseline 0%
- New behavior (7 tasks): **28.5%**（也列在 env generalization 一栏）

这极其令人惊讶。Video world model 见过 internet 上各种环境的视频，fine-tune 时只看到 lab environment，但 prompt 时给一张新环境的初始帧，它居然能 generate 出 robot 在新环境执行任务的视频。这是 **internet prior + short-horizon adaptation** 的 combined effect。

paper 明确对比 [π0.5](https://arxiv.org/abs/2504.16054) 等工作：那些用大规模 multi-environment teleop data 来实现 environment generalization，而 DREAMGEN 只需要 single environment + new initial frames。

### 3.5 DreamGen Bench (Table 2)

8 个 video world models × 2 个 robots × 3 个 dimensions (object/behavior/env)：

**Two metrics**:
- IF (Instruction Following): GPT-4o + Qwen2.5-VL + human eval (Pearson r > 0.9 验证)
- PA (Physics Alignment): VideoCon-Physics + Qwen2.5-VL

Top performers on GR1 (after fine-tune):
- Cosmos-sft: IF 90.0 (object), 59.6 (behavior), 69.0 (env); PA 73.0, 64.9, 65.5
- WAN2.1-sft: IF 72.0, 72.3, 48.3; PA 69.0, 74.5, 67.4

**关键 finding**: DreamGen Bench 和下游 policy performance 正相关（Figure 6）。意味着 video model researchers 不需要 robot hardware，可以靠这个 benchmark 间接贡献 robot learning。

---

## 4. 架构图解析 (Figure 2 + Figure 3)

### Figure 2 整体 Pipeline

```
Teleop Trajectories ──→ [Step 1: Fine-tune Video WM (LoRA)] ──→ Fine-tuned WM
                                                                       │
                                                                       ▼
Initial Frame + Language Instruction ──→ [Step 2: Rollout Video WM] ──→ Synthetic Video
                                                                              │
                                                                              ▼
                                                              [Step 3: Pseudo Action]
                                                                  /            \
                                                          IDM            LAPA
                                                              \            /
                                                               ▼          ▼
                                                          Real action  Latent action
                                                                  \          /
                                                                   ▼        ▼
                                                          Neural Trajectory (Video + Action)
                                                                              │
                                                                              ▼
                                                          [Step 4: Policy Training]
                                                                  /     |     \
                                                              DP    π0    GR00T N1
```

### Figure 3 两个 Action Extraction 架构

**(a) IDM Architecture**:
```
Frame o_t ──┐
            ├──→ SigLIP-2 ──→ patch tokens ──┐
Frame o_{t+H} ─────────────────→ patch tokens ─┤
                                                ▼
                                         Concat / Cross-Attn
                                                │
                                       Diffusion Transformer
                                                │
                                    (conditioned on flow time τ)
                                                │
                                                ▼
                                       Action Chunk â_{t:t+H}
```

**(b) LAPA Architecture**:
```
Frame o_t ──┐
            ├──→ Encoder ──→ continuous z ──→ Quantize ──→ ẑ (codebook idx, size 8, seq len 16)
Frame o_{t+Δ} ─┘                                                     │
                                                                     ▼
                                                              Decoder(o_t, ẑ)
                                                                     │
                                                                     ▼
                                                            Reconstructed ô_{t+Δ}
                                                              (training only)
```

Inference 时：只取 $\hat{z}$ 的 pre-quantized continuous embedding 作为 latent action 喂给下游 policy。

---

## 5. 关键公式和符号汇总

| Symbol | 含义 |
|---|---|
| $o_t$ | 第 t 步的 image observation |
| $i_t$ / $\ell$ | language instruction |
| $a_{t:t+H}$ | action chunk，从 step t 到 t+H |
| $\hat{a}_{t:t+H}$ | predicted action chunk (IDM 输出) |
| $V = \{o_0, ..., o_T\}$ | video rollout |
| $v_\theta$ | flow matching velocity field (DiT) |
| $\tau \in [0,1]$ | flow time (类似 diffusion 的 timestep) |
| $\epsilon \sim \mathcal{N}(0, I)$ | base distribution sample |
| $z = E_\phi(o_t, o_{t+\Delta t})$ | LAPA encoder 输出 |
| $C$ | LAPA codebook (size 8) |
| $\hat{z}$ | quantized latent action |
| $\text{sg}[\cdot]$ | stop-gradient |
| $\beta$ | commitment loss weight |
| $H$ | action chunk length / horizon |
| $\Delta t$ | LAPA 用的 future frame offset (paper 说 1 second ahead) |
| $N$ | number of neural trajectories |

---

## 6. Limitations 和 Open Questions

paper Section 7 自承：
1. **Compute**: 54 小时 × 1500 L40 GPUs 生成 240k samples。这对 academic lab 几乎不可行。
2. **Initial frames 仍需手动采集**。这违背了 "fully automated" 的理想。
3. **Evaluation 用 small VLMs (7B)**，会 hallucinate。
4. **Task complexity 仍 simple**——22 个 behavior 都是单 verb short-horizon，没测试 long-horizon 多阶段任务。

我自己观察的 open questions：
- **Action chunk overlap fusion**：paper 没说 sliding window 的 overlap region 怎么处理。如果直接 average，会 smooth 掉 fine-grained motion；如果用 first prediction，会有 discontinuity。这影响实际 deployment smoothness。
- **State=0 hack 的极限**：对需要精确 proprioception 的任务（in-hand manipulation、compliance control），这个 hack 应该会 fail。Paper 没测试这类任务。
- **Distribution shift between dream and reality**：video model 可能 dream 出物体凭空消失、瞬移的"超自然"运动。Physics Alignment metric 是 0-1 binary，可能掩盖了细粒度差异。
- **Causal confusion in IDM**：IDM 看两帧预测 action，但两帧之间可能有多种 action 路径。Diffusion IDM 通过 sampling 能 capture multimodality，但实际部署时取哪个 sample？
- **Log-linear slope 能持续多久**：会不会和 LLM 一样遇到 "data wall"，因为 video world model 本身能 generate 的 behavior diversity 是有限的？

---

## 7. 与相关工作的位置定位

| 路线 | 代表作 | 与 DREAMGEN 区别 |
|---|---|---|
| Video as online planner | [UniPi](https://arxiv.org/abs/2302.00638), [VLP](https://arxiv.org/abs/2310.10625) | DREAMGEN 是 offline data generation，不分摊 inference latency |
| Latent action pretraining | [LAPA](https://openreview.net/forum?id=VYOe2eBQeh), [UniVLA](https://arxiv.org/abs/2505.06111), [GR-2](https://arxiv.org/abs/2410.06158) | DREAMGEN 把 latent action 用在 synthetic data 上，不直接做 internet video → policy |
| Synthetic data via sim | [MimicGen](https://arxiv.org/abs/2310.17596), [DexMimicGen](https://arxiv.org/abs/2410.24185), [RoboGen](https://arxiv.org/abs/2311.01455) | DREAMGEN 不需要 simulator，避开 sim2real |
| Generative augmentation | [GenAug](https://arxiv.org/abs/2302.06671), [ROVI-Aug](https://arxiv.org/abs/2409.03403), [Cosmos Transfer](https://arxiv.org/abs/2503.14492) | DREAMGEN 生成完整 video rollout，不只是 in-painting 或 style transfer |
| Unified video-action model | [UVA](https://arxiv.org/abs/2503.00200), [UniSim](https://arxiv.org/abs/2310.05632), [UniWorld](https://arxiv.org/abs/2504.02792) | DREAMGEN 显式分两阶段，让 SOTA video model 单独 maximally strong |
| VPT-style IDM | [VPT](https://arxiv.org/abs/2206.11795) | DREAMGEN 用 IDM 提取 action 而非 train policy 直接 |
| Internet video learning | [VRB](https://arxiv.org/abs/2311.01398), [Track2Act](https://arxiv.org/abs/2405.01527), [MimicPlay](https://arxiv.org/abs/2302.12422), [VideoDex](https://arxiv.org/abs/2310.18091) | DREAMGEN 生成 robot videos 而非 human videos，避免 human-robot embodiment gap |

paper 里 **"Our approach deliberately separates these components to fully make use of the state-of-the-art video generative models"** 这句话很关键。Unified video-action model（UVA 等）虽然优雅，但 video model 和 action model 互相牵制，难以同时达到 SOTA。DREAMGEN 的 decoupling 是务实的 engineering choice。

---

## 8. 我对这篇 paper 的直觉总结

**为什么这件事 work**：video world model 在 internet 上学到的是 "physics-aware visual dynamics"——它知道 pouring 时液体该怎么流、hammering 时钉子该怎么动、unrolling mat 时布料该怎么变形。这些 prior 是 **embodiment-agnostic** 的。Fine-tune 在 specific robot 上加的是 **embodiment-specific kinematics**。两者 factorize 的很好，所以能 generalize 到 new verb + new environment。

**为什么以前没 work**：以前缺两块。一是 video model 不够好（一年前生成的 video 还充满 morphing artifacts），二是 IDM 不够准（VPT 时代的 IDM 在 Minecraft 上 work，但泛化到 real robot 很难）。SOTA video model（WAN2.1, Cosmos）+ LAPA-style pretraining latent action 是 enabling factor。

**这件事对未来意味着什么**：robot learning 的 data scaling axis 之前一直是 "more teleop"，现在多了一条 "more dreaming"。如果 log-linear slope 能持续，未来可能是 "1 demo + 1M dreams" 训练一个 task。这非常像 LLM 中 "human-written" vs "synthetic from stronger model" 的 dynamic——只是这里没有 "stronger model"，video world model 本身扮演了这个角色。

**关于 generalization 的本质**：behavior generalization 和 environment generalization 的来源不同。Behavior generalization 主要靠 **video model 的 internet prior**（"pour water" 这个 verb 在 internet video 中被见过）。Environment generalization 主要靠 **video model 的 conditional generation 能力**（看到 new environment 的初始帧，generate 出 follow physics 的 rollout）。Paper 没有显式 ablation 这两个来源，但 Table 1 的数据暗示 behavior gen (43.2%) > env gen (28.5%)，可能因为 verb prior 比 spatial generalization 更 robust。

---

## References

- [DREAMGEN Project Page](https://research.nvidia.com/labs/gear/dreamgen)
- [WAN2.1](https://arxiv.org/abs/2503.20314) - 主要 base video model
- [Cosmos](https://arxiv.org/abs/2501.03575) - DreamGen Bench 最佳 fine-tuned model
- [CogVideoX](https://arxiv.org/abs/2408.06072)
- [HunyuanVideo](https://arxiv.org/abs/2412.03603)
- [LAPA](https://openreview.net/forum?id=VYOe2eBQeh) - latent action 模型
- [VPT](https://arxiv.org/abs/2206.11795) - IDM 思想来源
- [GR00T N1](https://arxiv.org/abs/2503.14734) - 下游 policy 之一
- [π0](https://arxiv.org/abs/2410.24164) 和 [π0.5](https://arxiv.org/abs/2504.16054) - 下游 policy 之一 / env generalization 对照
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) - 下游 policy 之一
- [RoboCasa](https://robocasa.ai/) - 主要 simulation benchmark
- [DROID](https://droid-dataset.github.io/) - Franka 实验 pretrain data
- [VideoCon-Physics](https://arxiv.org/abs/2406.03520) - PA metric
- [UniPi](https://arxiv.org/abs/2302.00638) - 早期 video-as-planner
- [Video Language Planning](https://openreview.net/forum?id=9pKtcJcMP3)
- [RoboDreamer](https://arxiv.org/abs/2404.12377)
- [UniVLA](https://arxiv.org/abs/2505.06111) - latent action VLA
- [GR-2](https://arxiv.org/abs/2410.06158) - generative VLA
- [MimicGen](https://arxiv.org/abs/2310.17596) / [DexMimicGen](https://arxiv.org/abs/2410.24185) - simulation-based data generation
- [Cosmos Transfer 1](https://arxiv.org/abs/2503.14492) - conditional world generation
- [UVA](https://arxiv.org/abs/2503.00200) / [UniWorld](https://arxiv.org/abs/2504.02792) - unified video-action model
- [Track2Act](https://arxiv.org/abs/2405.01527) / [MimicPlay](https://arxiv.org/abs/2302.12422) / [VideoDex](https://arxiv.org/abs/2310.18091) - learning from human video

---

如果你想 build deeper intuition，我建议关注三个后续方向：（1）video world model 的 causal 控制能力（能不能 fine-tune 成"我说让 hand 停在 X，就停在 X"）；（2）IDM 在多解 motion 下的 calibration（diffusion IDM sampling strategy）；（3）neural trajectory + real trajectory 的混合比该怎么 schedule（paper 用 1:1 固定比例，curriculum 可能更好）。如果其中一个 work，DREAMGEN 这条路可能就是 robot learning 的 GPT-3 moment。
