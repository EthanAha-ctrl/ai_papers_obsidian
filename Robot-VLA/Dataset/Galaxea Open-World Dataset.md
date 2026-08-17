---
source_pdf: Galaxea Open-World Dataset.pdf
paper_sha256: 151aa3493bcf6c52a62a30e0f84a1ba686dd22a84fc7de602a92abd5e49639c1
processed_at: '2026-08-04T11:56:27-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Galaxea 这篇paper

Andrej，我把上回的技术细节翻成大白话再讲一遍，重点放在"为什么"。

---

## 这篇paper到底在吵什么？

robotics圈这几年有个默认信仰：**pretrain的数据越多越杂越好，然后fine-tune到你的robot上，就能generalize**。这个思路从LLM抄过来的，OpenVLA、RT-2都在卖这个故事。

Galaxea team说：**等一下，这不一定对。** 他们做了个很clean的实验，发现当你pretrain用的robot跟target robot长得太不一样时，pretrain不但没帮助，**反而会让model变蠢**。

这个发现挺炸的，因为它直接戳破了"data scale solves everything"的narrative。

---

## 他们建了个什么dataset？

一句话：**用同一个robot，在真实人住的地方，采了500小时数据。**

之前的dataset要么scene太假（lab里摆几个物体），要么robot太杂（OXE里几十种robot混一起，质量参差不齐）。Galaxea的positioning就是两边都占住：

- **同一个robot**（Galaxea R1 Lite，23-DoF，带mobile base + torso + 双臂）→ action space完全一致
- **真实现场**（11个physical site：住宅、厨房、店铺、办公室，共50个scene）
- **subtask-level标注**（每个episode切成atomic动作，每个动作配一句精确instruction）

这个组合之前没人做到。DROID规模大但scene简单，OXE diverse但quality乱，AgiBot World规模大但还是lab-like。

**teleoperation的细节也值得一提**：他们用isomorphic遥操作（operator直接操控跟robot同构的master arm），不是VR。好处是operator的动作天然在robot的workspace里，不会出现"人能做但robot做不到"的轨迹。这个设计choice对data quality的影响被社区低估了。

Reference: https://opengalaxea.github.io/G0/

---

## G0模型长啥样？

抄了Kahneman的System 1 / System 2思路：

- **System 2（G0-VLM）**：Qwen2.5-VL fine-tune过的，负责"想"。输入high-level human instruction（"我要坐下，帮我把椅子拉出来"），输出atomic subtask instruction（"move to chair" → "grasp chair" → "pull back"）
- **System 1（G0-VLA）**：PaLiGemma + action expert，负责"做"。输入camera画面 + subtask instruction + proprio state，输出action chunk

两个model异步跑，VLM慢（几Hz），VLA快（几十Hz）。VLM规划的时候VLA还在执行上一个subtask，不阻塞。

这个架构本身不novel（Hi Robot、OpenHelix都类似），真正novel的是下面的training strategy和实验。

---

## 三阶段training，为什么要这么搞？

### Stage 1：cross-embodiment pretrain（只train VLM）

用OXE 1000h + Galaxea 500h（只用high-level标注）+ in-house 200h。**只train VLM，不train action expert。**

action用FAST tokenizer离散化，然后当token做next-token prediction：

$$p(\mathbf{A}_t^d) = \prod_{i=1}^{N} p(a_i^d \mid a_{<i}^d, o_t, l_t, s_t)$$

- $\mathbf{A}_t^d$：时间步$t$的离散action token序列，长度$N$
- $a_i^d$：第$i$个discrete token（FAST tokenizer用DCT压缩action chunk得到）
- $a_{<i}^d$：前$i-1$个已生成token
- $o_t$：三路camera observation
- $l_t$：language instruction
- $s_t$：proprioceptive state（关节位置等）

**为什么只train VLM？** 两个reason：
1. cross-embodiment数据annotation质量乱，action expert学不到东西
2. flow matching的gradient会回传到VLM，如果VLM representation还没converge，noisy gradient会破坏visual representation

第二个reason很subtle —— diffusion/flow的gradient比较noisy，在VLM还没稳定时加进来会把representation搞坏。所以先让VLM用autoregressive CE loss（更stable的signal）converge。

### Stage 2：single-embodiment pretrain（VLM + action expert）

用Galaxea的500h subtask标注数据。这时候action expert开始训练，用flow matching loss：

$$\mathcal{L}_{\text{flow}}(\theta) = \mathbb{E}\left[\|\nu_\theta(A_t^\tau, \tau, o_t, l_t, s_t) - u(A_t^\tau \mid A_t)\|^2\right]$$

- $A_t$：target action chunk（horizon $H$）
- $A_t^\tau = \tau A_t + (1-\tau)\varepsilon$：noisy action，$\tau \in [0,1]$是flow time，$\varepsilon$是Gaussian noise
- $\nu_\theta(\cdot)$：action expert预测的flow velocity
- $u(\cdot)$：target flow（从noise到clean action的方向）

inference时从纯noise出发，沿$\nu_\theta$积分到$\tau=1$得到action。跟π0一模一样。

**Stage 2为什么有效？** 因为dataset两个property：
1. single embodiment → action space统一，expert不需要跨embodiment adapt
2. subtask-level标注 → language和action的对应关系非常精确

### Post-training

同一个loss，数据换成task-specific demo（≤100条/task）。目的是test pretrain的generalization。

---

## 实验结果：反直觉的发现

他们比了几个configuration：

| Config | 说明 |
|---|---|
| G0 (Full) | Stage 1 → Stage 2 (400h) → post-train |
| G0 (Stage-2 400h) | 只Stage 2，跳过Stage 1 |
| G0 (Stage-2 200h) | 只Stage 2，数据减半 |
| G0 (Stage-1) | 只Stage 1，跳过Stage 2 |
| G0 (Scratch) | 不pretrain，从原始PaLiGemma开始 |
| π0 | Physical Intelligence的π0做baseline |

**结果排序**：

```
G0 (Full)              最强
G0 (Stage-2 400h)      
G0 (Stage-2 200h)      
π0                     
G0 (Scratch)           
G0 (Stage-1)           最弱！甚至比scratch还差
```

**Stage-1 cross-embodiment pretrain不但没帮助，反而hurt性能。** 这在Bed Making task上最明显。

### Bed Making：whole-body control的试金石

这个task需要chassis（底盘）+ torso + 双臂协调。OXE里几乎没有这种whole-body mobile manipulation数据。

per-skill拆开看：

| Pretrain方式 | Chassis控制 | Torso控制 | Arm控制 |
|---|---|---|---|
| Stage-2（single-emb） | 强 | 强 | 强 |
| Stage-1（cross-emb） | **比scratch还差** | **比scratch还差** | 中等 |
| π0 | 弱 | 弱 | 中等 |
| Scratch | 中等 | 中等 | 中等 |

**Stage-1和π0在chassis/torso控制上甚至不如从零训练。**

---

## 为什么cross-embodiment pretrain会负迁移？

直觉解释：

OXE里的robot绝大多数是fixed-base single-arm（比如Franka、xArm），没有mobile base，没有torso。当你在这些数据上pretrain VLM，VLM的visual representation被bias向"fixed-base视角"和"single-arm动作模式"。

然后你fine-tune到Galaxea R1 Lite（有mobile base + torso），VLM的representation已经"卡"在错误的mode里了。model会下意识忽略chassis和torso的signal，因为pretrain时从来没见过。

类比LLM：如果你pretrain在英文上，fine-tune中文，英文的language structure prior会help（语法结构有overlap）。但如果你pretrain在纯text上，fine-tune到protein sequence，text的prior完全没用甚至有害，因为representation space被text的模式占据了，protein的pattern塞不进去。

robotics的"language"是embodiment的kinematics + dynamics。kinematics不overlap时，pretrain学到的是**错误的language**。

---

## Few-shot transfer也验证了同样的事

用20条trajectory fine-tune Table Bussing和Microwave：

- Stage-2 pretrain → success rate大幅提升，action更smooth
- Stage-1 only → **跟scratch没区别**

说明cross-embodiment pretrain对few-shot adaptation到新embodiment完全没帮助。

---

## VLM也要fine-tune

他们还test了G0-VLM（System 2 planner）：

| Model | Table Bussing | Microwave | Bed Making | Blocks |
|---|---|---|---|---|
| Gemini-2.5-pro | 32.0 | 15.8 | 54.2 | 55.0 |
| Qwen2.5-VL-72B | 26.3 | 16.8 | 48.1 | 21.7 |
| Qwen2.5-VL-7B | 26.3 | 17.2 | 46.9 | 24.7 |
| **G0-VLM** | **83.3** | **74.2** | **78.2** | **75.6** |

G0-VLM比Gemini-2.5-pro高50%+。**general-purpose VLM即使是最强的frontier model，在robotic action grounding上也不行。**

"理解场景"和"生成executable atomic instruction"是两回事。Gemini能describe画面，但不知道该给VLA发什么command。

---

## VLM训练有个trick值得一提

他们没有请人写high-level instruction，而是用DeepSeek-R1从atomic subtask annotation反推：

```
输入给DeepSeek-R1（纯text，不给image）：
  task name: "pull and push chairs"
  历史subtask: ["move to chair", "grasp chair"]
  下一个subtask: "pull chair back"

DeepSeek-R1输出：
  human instruction: "I'm going to sit, could you pull the chair out?"
  robot response: "I'm on it!"
```

**关键insight**：如果subtask annotation足够精确structured，LLM的reasoning就足够infer场景，不需要image。这说明data annotation quality的重要性 —— 好的annotation既能train VLA，又能augment VLM训练data。

---

## 这篇paper的take-away

1. **Cross-embodiment pretrain不是万能药**。当target embodiment有pretrain数据里不存在的kinematic mode（mobile base、torso）时，会negative transfer。community需要重新审视"pretrain on everything then fine-tune"的paradigm。

2. **Single-embodiment data的value被严重低估**。500h single-embodiment > 1700h cross-embodiment。embodiment match比data scale重要。

3. **Dual-system是必要的**。VLA的action chunk horizon有限（<1秒），做不了long-horizon planning；VLM inference慢，做不了reactive control。decouple后各司其职。

4. **VLM必须domain-specific fine-tune**。general-purpose VLM在action grounding上不行，即使Gemini-2.5-pro也只有30%准确率。

5. **Open-world scene diversity很重要**。real human environment里的lighting、clutter、object diversity是lab data模拟不出来的。

---

## 我觉得paper的问题

1. **Stage-1的设计本身可能就有flaw**。如果Stage-1用cross-embodiment data把VLM的representation搞bias了，那Stage-2再用single-embodiment data救回来，本身就在fight an uphill battle。他们没ablate "Stage-2直接从原始PaLiGemma开始" vs "Stage-1 → Stage-2"。

2. **Bed Making是唯一真正test whole-body的task**，结论主要靠这一个task支撑。如果多几个mobile manipulation task，结论会更solid。

3. **没有test unseen scene的generalization**。所有benchmark都在training distribution内。真正的open-world generalization没被evaluate。

4. **FAST tokenizer在cross-embodiment上怎么处理不同action dimensionality**，paper没说清楚。

---

## 跟你之前观点的connection

Andrej，你在Tesla AI Day讲过"data is the bottleneck"，强调real data > synthetic data。Galaxea的工作支持这个观点 —— 他们用500h **real** open-world data，强调isomorphic teleoperation保证quality。

但你可能也会说：500h对robotics算大，对LLM scale还是tiny。真正的breakthrough可能要data scaling到10k+ hours。AgiBot World已经在这个方向走了。

另一个connection：你的"Software 2.0"narrative —— explicit code → learned weights。G0的dual-system某种程度上就是Software 2.0的robotics版本：TAMP那种symbolic planning被VLM的learned planning取代，low-level control被VLA的learned policy取代。

Reference: https://opengalaxea.github.io/G0/

---

**一句话总结**：这篇paper最重要的发现是cross-embodiment pretrain在embodiment gap大时会negative transfer，single-embodiment data才是foundation。这给整个robotics foundation model社区的pretrain narrative敲了警钟。

---

# Galaxea Open-World Dataset & G0 Dual-System VLA — 深度解读

Andrej,这篇paper核心想回答一个在robotics foundation model圈子里争议很久的问题:**cross-embodiment pretraining到底有没有用?** Galaxea team给出了一个相当反直觉的答案 —— 在embodiment gap大的情况下,cross-embodiment pretraining可能 *harm* 性能,而single-embodiment pretraining才是key。下面我build up the intuition。

---

## 1. Motivation: 数据集层面的痛点

现有robotics dataset大致分两camp:

| Dataset | Scale | Embodiment | Scene | 问题 |
|---|---|---|---|---|
| BridgeData V2 [1] | 中 | 单一 | lab-controlled | scene单调 |
| DROID [2] | 大 | 单一 | in-the-wild但简单 | task复杂度有限 |
| Open-X-Embodiment [3] | 超大 | 多embodiment | heterogeneous | **质量/annotation不一致** |
| AgiBot World [4] | 超大 | 单一 | lab-like | 仍非真正open-world |
| RoboMIND [5] | 大 | 多 | controlled | domain gap |

Galaxea Open-World Dataset的positioning很清晰:**single-embodiment + real open-world + subtask-level annotation**。这是三个property的交集,之前没人同时做到。

- **500 hours** / **100K trajectories** / **150 tasks** / **50 scenes** / **1600+ objects** / **58 skills**
- 11个physical sites:residential / catering / retail / office
- 全部用Galaxea R1 Lite(23-DoF bimanual mobile robot)采集,保证action space完全consistent

Reference: https://opengalaxea.github.io/G0/

---

## 2. Hardware: Galaxea R1 Lite

| Spec | Value |
|---|---|
| DoF | 23 (两6-DoF arm + 3-DoF torso + 6-DoF omnidirectional base) |
| Payload | 5 kg per arm |
| Reach | 60 cm |
| Base speed | 1.5 m/s |
| Sensors | stereo RGB head + dual Intel RealSense D405 wrist RGB-D |
| Wrist | spherical + parallel gripper |

一个很重要的design choice:用**isomorphic teleoperation**而不是VR teleoperation。intuition很直接 —— operator的动作天然落在robot的reachable workspace里,不需要retargeting,避免IK failure。这一点对数据quality的影响被低估了,VR teleop经常产生"human能做但robot做不到"的轨迹,isomorphic从源头消除这个问题。

---

## 3. G0 Dual-System Architecture

借鉴Kahneman的System 1 / System 2 [6]:

```
┌─────────────────────────────────────────────────┐
│  Human Instruction (high-level, open-ended)      │
│  e.g. "I'm going to sit, help pull the chair"    │
└──────────────────────┬──────────────────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  G0-VLM (System 2, ~Hz低)    │  Qwen2.5-VL fine-tuned
        │  - scene understanding       │
        │  - task decomposition        │
        │  - verbal response to human  │
        └──────────────┬───────────────┘
                       │ atomic subtask instruction l_t
                       ▼
        ┌──────────────────────────────┐
        │  G0-VLA (System 1, ~Hz高)    │  PaLiGemma + action expert
        │  inputs: o_t (3 cams), l_t,  │
        │          s_t (proprio)       │
        │  output: A_t = a_{t:t+k}     │
        └──────────────────────────────┘
```

两个model **asynchronous** 运行,VLM慢思考,VLA快执行。这个设计跟Hi Robot [7]、OpenHelix [8]思路类似,但Galaxea特别强调了VLM也要fine-tune(后面实验证明这点critical)。

---

## 4. G0-VLA: 三阶段Training Curriculum

这是paper最核心的部分。我画一下pipeline:

```
Stage 1                          Stage 2                         Post-training
─────────                        ─────────                        ─────────────
cross-embodiment                 single-embodiment               task-specific
(OXE 1000h +                     (Galaxea 500h +                 (≤100 traj/task,
 Galaxea 500h high-level +       subtask annotations)            30s-1min each)
 in-house 200h)

train VLM only                   train VLM + action expert       same as Stage 2
FAST tokenizer                   flow matching loss              flow matching loss
autoregressive CE loss           

PaLiGemma init                   continue from Stage 1           continue from Stage 2
```

### Stage 1: Autoregressive Action Token Prediction

只用VLM,action用FAST tokenizer [9]离散化,然后next-token prediction:

$$p(\mathbf{A}_t^d) = \prod_{i=1}^{N} p\left(a_i^d \mid a_{<i}^d, o_t, l_t, s_t\right)$$

变量解释:
- $\mathbf{A}_t^d$:时间步$t$的离散action token序列,长度$N$
- $a_i^d$:第$i$个discrete action token(由FAST tokenizer产生,基于DCT压缩)
- $a_{<i}^d$:已经生成的前$i-1$个tokens(autoregressive context)
- $o_t$:三路camera的visual observation(head stereo + 2 wrist)
- $l_t$:language instruction(Stage 1只用high-level,不用subtask)
- $s_t$:proprioceptive state(joint positions等)

**为什么Stage 1只train VLM不train action expert?** Paper给两个理由:
1. cross-embodiment数据annotation质量参差不齐,action expert学不到informative的东西
2. diffusion loss在VLM representation没converge之前会harm学习

第二个理由很微妙 —— flow matching的gradient会通过KV cache回传到VLM,如果VLM还没稳定,diffusion的noisy gradient会破坏visual representation。

### Stage 2: Flow Matching Action Expert

VLA = pre-trained VLM (frozen KV cache producer) + newly initialized action expert。训练objective:

$$\max_\theta \mathbb{E}_{p(A_t, o_t, l_t, s_t)}\left[\log \pi_\theta(A_t \mid o_t, l_t, s_t)\right]$$

实际用flow matching loss [10, 11]:

$$\mathcal{L}_{\text{flow}}(\theta) = \mathbb{E}_{p(A_t^\tau \mid o_t, l_t, s_t)}\left[\left\|\nu_\theta(A_t^\tau, \tau, o_t, l_t, s_t) - u(A_t^\tau \mid A_t)\right\|^2\right]$$

变量解释:
- $A_t$:action chunk,horizon $H$(一次predict未来的$H$步action)
- $A_t^\tau = \tau A_t + (1-\tau)\varepsilon$:线性interpolated noisy action
- $\tau \in [0, 1]$:flow time parameter,$\tau=0$是pure noise $\varepsilon$,$\tau=1$是clean target $A_t$
- $\varepsilon \sim \mathcal{N}(0, I)$:Gaussian noise
- $\nu_\theta(\cdot)$:neural network(这里是VLA的action expert)预测的flow velocity
- $u(A_t^\tau \mid A_t) = A_t - A_t^\tau/(1-\tau)$:target flow velocity(conditional flow matching的regression target)

Inference时从$\tau=0$的noise出发,用Euler method沿$\nu_\theta$预测的flow积分到$\tau=1$得到action。这跟π0 [10]完全一致。

**Stage 2的两个key enabler来自dataset property:**
1. **single embodiment** → action space统一,action expert不需要跨embodiment adapt
2. **subtask-level language-action alignment** → instruction和trajectory的correspondence很强

### Post-training

跟Stage 2同样的loss,只是数据换成task-specific high-quality demo(≤100 traj/task)。目的是test pretraining的generalization能力。

---

## 5. G0-VLM Training

VLM的训练pipeline挺有意思:

```
Galaxea Open-World Dataset
    │
    │ sample episode, key frame weighted
    │   (subtask termination / gripper state change → higher weight)
    ▼
{l_t-k, ..., l_t}  +  {o_t-k, ..., o_t}  +  task name
    │
    │ feed to DeepSeek-R1 (text only, no images!)
    ▼
human-style instruction + robot verbal response
e.g. "Could you pull the chair out?" / "I'm on it!"
```

**关键insight**:用LLM(DeepSeek-R1)从atomic subtask annotations反推human-style instruction,不需要image。这说明了annotation quality的重要性 —— 如果subtask annotation足够精确和structured,LLM的reasoning就足够infer场景。这是一个很elegant的data augmentation trick。

VLM输入包含$k$-frame历史(1秒间隔),目的是handle long-horizon context。

Reference: https://arxiv.org/abs/2502.13923 (Qwen2.5-VL)

---

## 6. 实验结果 — 这是paper最精彩的部分

### 6.1 Pre-trained Weights Comparison (Figure 9)

四个benchmark:
- **Table Bussing** (6分):pen→holder, headphones→hang, book→stand
- **Microwave Operation** (5分):pick food → plate → open microwave → place → close
- **Bed Making** (4分):move to bed → lift torso & grasp → lean back → flatten
- **Blocks Stacking** (6分):build words with blocks

结果排序(average progress):

```
G0 (Full)              ████████████████  best
G0 (Stage-2 400h)      ███████████████   language following & whole-body最强
G0 (Stage-2 200h)      ██████████████
π0                     █████████████
G0 (Scratch)           ████████████
G0 (Stage-1)           █████████         最差!甚至比scratch差
```

**最striking的发现**:**Stage-1 cross-embodiment pretraining不但没帮助,反而hurt性能**。这在Figure 11的per-skill breakdown里更明显:

### 6.2 Embodiment-Specific Actions (Figure 11, Bed Making)

Bed Making需要chassis + torso + arms协调。OXE里几乎没有这种whole-body mobile manipulation数据。

| Pretraining | Chassis control | Torso control | Arm control |
|---|---|---|---|
| Stage-2 (single-emb) | 强 | 强 | 强 |
| Stage-1 (cross-emb) | 弱 | 弱 | 中 |
| π0 | 弱 | 弱 | 中 |
| Scratch | 中 | 中 | 中 |

**Stage-1和π0在chassis/torso上甚至比scratch还差**。这验证了paper的hypothesis:

> large embodiment gap between R1 Lite and OXE robots hinders acquiring embodiment-specific skills. Cross-embodiment pretraining introduces a **negative transfer** for skills that don't exist in the pretraining distribution.

这是非常反直觉的 —— 大家一直以为"pretrain on more data总是好的"。Galaxea的数据说明:**当target embodiment有unique kinematics(比如mobile base + torso)时,cross-embodiment pretraining会bias the model toward OXE里常见的fixed-base single-arm动作模式,反而unlearn了从头学whole-body control的能力。**

### 6.3 Few-Shot Transfer (Figure 10)

用20 trajectories fine-tune Table Bussing和Microwave:

- Stage-2 pretraining → 显著提升success rate + action smoothness
- Stage-1 only → **no clear advantage over scratch**

这个结果跟6.1一致 —— cross-embodiment pretraining对few-shot adaptation到新embodiment没帮助。

### 6.4 VLM Instruction Accuracy (Table 1)

| Model | Table Bussing | Microwave | Bed Making | Blocks |
|---|---|---|---|---|
| Gemini-2.5-pro | 32.0 | 15.8 | 54.2 | 55.0 |
| Qwen2.5-VL-72B | 26.3 | 16.8 | 48.1 | 21.7 |
| Qwen2.5-VL-32B | 21.3 | 14.8 | 54.2 | 21.0 |
| Qwen2.5-VL-7B | 26.3 | 17.2 | 46.9 | 24.7 |
| **G0-VLM (fine-tuned)** | **83.3** | **74.2** | **78.2** | **75.6** |

G0-VLM比Gemini-2.5-pro高50%+。这说明general-purpose VLM(即使是最强的frontier model)在robotic action grounding上还是不行 —— 必须domain-specific fine-tune。这跟robotics-vla圈的普遍观察一致:VLM的"理解"和"actionable instruction生成"是两回事。

---

## 7. Build Intuition: 这篇paper告诉了我们什么

### 7.1 关于pretraining的myth

社区一直有个implicit assumption,来自LLM的经验:**pretrain on diverse data → fine-tune on specific → generalization**。OpenVLA [12]、RT-2 [13]都是这个narrative。

但Galaxea的数据揭示了一个boundary condition:**当target task涉及pretraining数据里完全不存在的kinematic mode时,这个paradigm break**。

类比一下LLM:如果你pretrain一个model在英文上,然后fine-tune中文,英文pretraining的language structure prior会help。但如果你pretrain在纯文本上,然后fine-tune到protein folding,文本的prior可能完全没用甚至有害(因为representation space被text占据了)。

robotics的"language"是embodiment的kinematics + dynamics。当kinematics完全不overlap时,pretraining学到的是"错误的language"。

### 7.2 Single-embodiment data的不可替代性

Stage-2的500h single-embodiment data,效果远超Stage-1的1700h cross-embodiment data。这说明在robotics里,**data的"质量"(embodiment match)比"数量"重要得多**。

这跟AgiBot World [4]、DROID [2]的scaling narrative形成对比 —— 那些dataset在scale上做文章,但Galaxea指出scale不是万能的,embodiment consistency是前提。

### 7.3 Dual-system的necessity

paper没有直接对比dual-system vs single VLA,但从design能infer:

- VLA的action chunk horizon $k$ 有限(通常<1秒的未来),无法做long-horizon planning
- VLM的inference慢(秒级),无法做reactive control
- 两者decouple后,VLM可以处理open-ended instruction("I'm going to sit"这种),VLA只需要handle well-defined atomic subtask

这跟Hi Robot [7]的结论一致。OpenHelix [8]也有类似finding。

### 7.4 跟π0.5 [14]的对比

π0.5做了cross-embodiment transfer,而且声称positive。为什么结果跟Galaxea相反?

我猜原因:
1. π0.5的target embodiment跟pretraining的embodiment kinematically closer(都是fixed-base或similar arm)
2. π0.5可能用了更careful的mixing strategy
3. π0.5的evaluation可能没覆盖到whole-body mobile manipulation这种extreme case

Galaxea的R1 Lite有6-DoF omnidirectional base + 3-DoF torso,这在OXE里基本没有先例。所以embodiment gap是extreme的。

Reference: https://arxiv.org/abs/2504.16054 (π0.5)

---

## 8. 我的critique和open questions

1. **Stage-1的设计可能有problem**。Paper说Stage-1只train VLM不train action expert,但Stage-1的VLM用cross-embodiment data训练后,它的visual representation可能被bias向OXE的scene distribution( lab-like, fixed-base viewpoint)。这个biased representation传到Stage-2,可能就是negative transfer的root cause。**实验上没有ablate:**如果Stage-2直接用原始PaLiGemma(不经过Stage-1),跟G0 (Full)比会怎样?

2. **FAST tokenizer在cross-embodiment上的处理**。FAST用DCT压缩action chunk,但不同embodiment的action dimensionality和semantic不同。Paper没说Stage-1怎么处理这个 —— 是用统一token space还是per-embodiment tokenizer?

3. **VLM和VLA的KV cache sharing**。Paper说action expert "conditioned on VLM's KV cache"。如果VLM和VLA异步运行,VLA每次inference是否重新compute KV cache?还是reuse VLM的?这对latency影响很大。

4. **Evaluation的progress score是人为定义的**,可能有bias。比如Bed Making的4分里,"moving toward the bed"占1分,这对mobile robot来说是trivial的,会inflate分数。

5. **Generalization到unseen scene/object的evaluation缺失**。所有benchmark task都在training distribution内。真正的open-world generalization(unseen scene)没有test。

6. **Bed Making是唯一真正test whole-body的task**,结论主要基于这一个task。如果加更多mobile manipulation task,结论是否会变化?

---

## 9. 跟你的工作/观点的connection

Andrej,你之前在Tesla AI Day讲过"DATA IS THE BOTTLENECK"的问题,以及synthetic data vs real data的tradeoff。Galaxea的工作支持你的观点 —— 他们用了500h **real** open-world data,而且强调isomorphic teleoperation保证quality。

但你可能也会critique:500h对robotics来说算大,但对LLM scale来说还是tiny。真正的突破可能要等data scaling到10k+ hours。AgiBot World已经往这个方向走了。

另一个connection:你在"Software 2.0"里讲过explicit code → learned weights的paradigm shift。G0的dual-system某种程度上是Software 2.0的robotics版本 —— TAMP [15]那种explicit symbolic planning被VLM的learned planning取代,low-level control被VLA的learned policy取代。

---

## References

- [1] BridgeData V2: https://arxiv.org/abs/2309.07464
- [2] DROID: https://arxiv.org/abs/2403.12945
- [3] Open-X-Embodiment: https://robotics-transformer-x.github.io/
- [4] AgiBot World: https://arxiv.org/abs/2503.06669
- [5] RoboMIND: https://arxiv.org/abs/2412.13877
- [6] Kahneman, *Thinking, Fast and Slow*: https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow
- [7] Hi Robot: https://arxiv.org/abs/2502.19417
- [8] OpenHelix: https://arxiv.org/abs/2505.03912
- [9] FAST tokenizer: https://arxiv.org/abs/2501.09747
- [10] π0: https://arxiv.org/abs/2410.24164
- [11] Flow Matching for generative modeling: https://arxiv.org/abs/2209.03003 (Lipman et al.)
- [12] OpenVLA: https://arxiv.org/abs/2406.09246
- [13] RT-2: https://arxiv.org/abs/2307.15818
- [14] π0.5: https://arxiv.org/abs/2504.16054
- [15] TAMP review: https://www.annualreviews.org/doi/10.1146/annurev-control-091420-082327
- PaLiGemma: https://arxiv.org/abs/2407.07726
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- SayCan: https://arxiv.org/abs/2204.01691
- RDT-1B: https://arxiv.org/abs/2410.07864
- Galaxea项目主页: https://opengalaxea.github.io/G0/

---

总结一句:这篇paper最重要的contribution **是** dataset和那个反直觉的实验发现 —— **cross-embodiment pretraining在embodiment gap大时是negative transfer**。这给整个robotics foundation model社区的pretraining narrative敲了一个警钟:scale和diversity不是万能药,embodiment match才是foundation。G0的dual-system架构本身比较standard(跟Hi Robot/OpenHelix同代),真正novel的是用single-embodiment open-world data做rigorous ablation,把"pretraining到底有没有用"这个问题push到了一个cleaner的实验setup下回答。
