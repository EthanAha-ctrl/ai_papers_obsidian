---
source_pdf: Chain-of-Modality Learning Manipulation Programs from.pdf
paper_sha256: 2df31f796ae54ba6140f9efb4716dc7ec6c3ec8c1f11a08d031c2929e2c0d7fe
processed_at: '2026-08-18T03:16:03-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇 paper

## 一句话版本

让 robot 看一个人类示范 video 学做事，但光看 video 学不会"用力"的活儿——所以加了个测肌肉的电信号 armband，然后让 VLM 按"先看力、再看手、最后看画面"的顺序一步步推理，成功率高了三倍。

---

## 这 paper 要解决什么真实问题

想象你教 robot 开瓶盖。你自己开瓶盖的时候，手在干啥？你会先轻轻捏住瓶盖转一下试探，然后突然使劲拧一下，松开，再使劲拧一下。这个"使劲的 timing"是 task 的灵魂。

但你拍个 video 给 robot 看——robot 能看到啥？能看到你的手在动，能看到瓶盖。但 robot **看不到你用了多大劲**。video frame 里没有 force 这个 dimension。

所以纯 video learning 的 robot，学到的 plan 是"grasp → twist → release"，但不知道什么时候 grasp 要轻、什么时候 grasp 要重。执行出来就废了。

这篇 paper 的核心 insight 很朴素：**力的信息，你得用测力的 sensor 去拿**。vision 无法 infer force，这是 information-theoretically 不可逾越的。

---

## 为什么直接把所有 signal 塞给 VLM 不 work

Google 的 Gemini 1.5 Pro 和 OpenAI 的 GPT-4o 都支持 multimodal long-context input。你可以把 video frames、force signal（一堆 float）、hand pose（一堆坐标）全部 concat 在一起喂进去。

但实验发现这帮 VLM 在这种 merged input 下表现极差。原因作者没细讲，我从 intuition 推断：

VLM 的 attention 机制在处理长 context 时，会 suffer **attention dilution**。视觉 token 太多了——100 frame × 几百个 patch，把 force signal 那 100 个 float 淹没了。VLM 就 neglect force，整个 plan 就垮了。

类比一下：你在喧闹的 party 上让一个人同时听你说话、看你手势、读你手机屏幕上的字，他多半顾此失彼。但如果你先让他听清楚你说什么，再让他看手势补充，最后给他手机看细节——他每一步都能专注，结果好得多。

这就是 CoM 的核心思想。

---

## CoM 具体怎么做的

**Chain-of-Modality**：把多模态融合拆成 sequential pipeline。

### Step 1：先喂 force signal

VLM 看到 force signal 上有 3 个 peak。VLM 推理："这个人施加了 3 次力。"

但 VLM 还不知道在干啥，只是知道"有 3 次用力"。

### Step 2：加上 hand pose

VLM 看到 force peak 期间，手指在做 grasp + counterclockwise twist 180° → release → clockwise twist 180°。

VLM 推理："这个人 grasp 之后 CCW 转 180°，松开，再 CW 转 180°（回位）。"重复 3 次。

但还不知道在转什么。

### Step 3：加上 image

VLM 看到画面里左手持 bottle，右手操作 bottle cap。

VLM 推理："哦，是开瓶盖。左手 grasp bottle，右手 grasp cap 然后 twist。"

最终 output 是 structured task plan：
```
t=11: Grasp(left, bottle)
t=27: Grasp(right, cap)
t=27→37: Twist(right, CCW, 180°)
t=37: Release(right)
...
```

**关键 design**：每一步的 output 都作为下一步的 context，VLM 是在 refine 前序 hypothesis，而不是从零推理。

---

## 为什么这个顺序是 force → hand → image

我推断的 intuition：

- **Force 是最 coarse 的 segmentation signal**。它告诉你"什么时候有事发生"，是 temporal boundary detector。
- **Hand pose 是 motion grammar**。它告诉你"在做事的时候手在做什么动作"，是 action classifier。
- **Image 是 object grounding**。它告诉你"在做这个动作时操作的是什么东西"，是 semantic label。

从 abstract timing → motion grammar → object identity，是 information entropy 递减的 ordering。每一步把 search space 缩小，下一步的推理就更容易。

反过来如果先看 image：VLM 一上来看到 bottle 就 commit "这是开瓶盖"，然后 hand pose 和 force 都被这个 prior bias 了，可能错过细节。

---

## 拿 force 用的什么 sensor

两个 option：

1. **EMG armband**：戴在前臂上，测肌肉电活动。8 个 channel，200 Hz 采样。处理方式是 downsample 到 60 Hz（和 camera 对齐），然后 8 个 channel 取 max 作为该时刻的 force value。

   $$F_t = \max_{c \in \{1,...,8\}} \text{EMG}_{c,t}$$

   取 max 是因为肌肉活动是 sparse firing，mean 会把 peak 抹平。

2. **Microphone**：录物体 interaction 的声音。处理方式是 per-frame 算 loudness（推测是 RMS envelope）。drum 击打声、plug 插入声的音量峰值就是 force peak。

**EMG 的聪明之处**：它测的是 **muscle activation intention**，不是 contact force 本身。所以不需要在 robot end-effector 上装 tactile sensor，人类正常做事就行。缺点是 EMG → actual Newton 是非线性映射，且因人而异。但 CoM 不需要精确 Newton 值，只需要 relative level（high/medium/low），所以这个 trade-off 成立。

---

## 然后 VLM 把 plan 变成 robot code

CoM 输出的是 structured task plan text，还不能直接跑在 robot 上。还需要一步 code generation。

Prompt VLM：
```python
from skills import Grasp, Release, Twist, Find, Move_to, Insert
# Based on video analysis, generate python code:
```

VLM 输出：
```python
Move_to('left', Find('bottle'))
Grasp('left')
Move_to('right', Find('bottle_cap'))
for _ in range(3):
    Grasp('right')
    Twist('right', 'counterclockwise', 180)
    Release('right')
    Twist('right', 'clockwise', 180)
```

注意 VLM 自动生成了 `for _ in range(3)` loop——因为 CoM analysis 里识别到 force 有 3 个 peak，VLM 推断这是个循环动作。这是 structured program synthesis，不是简单 action sequence 模仿。

force-sensitive task 比如插 USB：
```python
Grasp('right', 'plug', 100)      # firm grasp
Move_to('right', 'box', 20)      # light push to wall, in-hand re-orient
Insert('right', 'power_strip', 100)
```

VLM 把 EMG 的 low/high level 映射到 discrete force parameter（20 vs 100），是把 continuous signal 离散化为 API token。

---

## `Find()` API 怎么实现

`Find('bottle')` 要返回 object 的 3D position。实现方式：

1. 把 RGB-D image + object name 喂给 Gemini 1.5 Pro，得到 2D bounding box。
2. 在 bbox 内取 depth，反投影到 3D point cloud。
3. 取 point cloud 的 centroid 作为 object position。

$$\mathbf{p} = \frac{1}{N} \sum_{i=1}^{N} \pi^{-1}(u_i, v_i, d_i; K)$$

$\pi^{-1}$ 是 pinhole camera back-projection，$(u_i, v_i, d_i)$ 是 bbox 内 pixel 的坐标和 depth，$K$ 是 camera intrinsic matrix，$N$ 是 valid depth pixel 数。

这个 API 是 open-vocabulary 的——你说什么 object name，Gemini 就找什么，不需要训 detector。这是 VLM 作为 perception primitive 的便利。

---

## 实验结果的核心数字

### Modality ablation（Table I）

| Setting | Success Rate (Gemini/GPT-4o) |
|---|---|
| Image-only | **0% / 0%** （所有 task） |
| w.o. img（force + hand） | 0%（猜对 action 但不知道 object） |
| w.o. force（image + hand） | 0%（visual 推不出 force-dependent plan） |
| All（force + hand + image） | 37-80% |

**最 striking 的 finding**：纯 image 在所有 force-sensitive task 上 success rate 是零。这是 paper 最强的 claim 之一——vision-only 学 manipulation 有 hard ceiling。

### Reasoning procedure ablation（Fig. 5）

| Method | CoM vs best baseline |
|---|---|
| Gemini 1.5 Pro | +19% |
| GPT-4o | +17% |

CoM 的 sequential refine 比 parallel analyze-then-merge 好 17-19%。

### Real robot（Table II）

| Task | CoM | Oracle（human code） |
|---|---|---|
| Opening Bottle (ViperX) | 60% | 80% |
| Opening Bottle (KUKA) | 75% | 100% |
| Insert Plug | 75% | 90% |
| Wiping Board | 70-80% | 80-100% |
| Playing Drum | 80% | 100% |

**Average：CoM 73%，Oracle 92%**。Gap 主要是 perception 噪声和 gripper 物理限制，不是 plan 错误。

**Cross-embodiment**：同一 generated program 跑在 ViperX 和 KUKA 两个不同 robot 上都 work，因为 code API 抽象掉了 embodiment。这是 code-as-policy paradigm 的核心优势。

---

## 为什么 CoM 比 merged 好（深层直觉）

我构建一个 mental model：

VLM 是 attention-limited reasoning engine。input 的 information rate（每秒有用信号 bit 数）如果超过 VLM 当前 forward pass 的 reasoning capacity，会 information loss。

Multimodal merged input 一次性 push 所有信息，导致两个问题：

1. **Attention dilution**：force signal 的 100 个 float 被 video 的几万个 visual token 淹没。
2. **Modality interference**：VLM 可能从 image 推断出 object identity 后，就用这个 prior 去"解释"force signal，而不是让 force signal 独立 inform plan。

CoM 的 sequential prompting 等价于 **information bottleneck regularization**：先在小信息量上 commit 一个 hypothesis，再用后续 modality refine。类似 Bayesian update——先验（force peak → 3 次 twist）+ 似然（hand pose 显示 twist motion）+ 后验（image 显示 bottle cap）。

这和 Chain-of-Thought 在 LLM reasoning 上的成功机制同构。CoT 把 single forward pass 中的 implicit reasoning 拆成 explicit intermediate steps；CoM 把 multi-modal fusion 拆成 explicit per-modality steps。

参考 CoT: https://arxiv.org/abs/2201.11903

---

## 与已有工作的关系

### Code as Policies [Liang et al., ICRA 2023]

Code as Policies 是 language → code。CoM 是 multimodal video → code。可以看作 Code as Policies 的 multimodal extension，但多了 force parameter inference 能力。

参考: https://arxiv.org/abs/2209.07758

### RT-2 [Google DeepMind, 2023]

RT-2 是 VLA model，端到端 finetune VLM 输出 action token。需要大规模 robot data 训练。CoM 是 zero-training pure prompting，只看一个 video。

参考: https://arxiv.org/abs/2307.15818

### Mimicplay [Wang et al., 2023]

也从 human video 学习，但用 play data + latent plan，是 end-to-end policy learning。CoM 完全不训练。

参考: https://arxiv.org/abs/2302.12422

### ReKep [Huang et al., 2024]

ReKep 用 VLM 生成 keypoint constraint 做 closed-loop trajectory optimization。CoM 是 open-loop。两者可以结合——CoM 生成 high-level plan，ReKep 在每个 sub-skill 内做 closed-loop refinement。

参考: https://arxiv.org/abs/2409.01652

---

## 局限

作者自述：

1. **Audio 只用 loudness**，没用 frequency/pitch/timbre。drum 音色、plug 插入"咔哒声"都包含 rich frequency 信息。
2. **Open-loop execution**：generated program 是固定 sequence，不适应 unexpected perturbation。未来需要 closed-loop。

我补充的潜在 limitation：

3. **EMG 个体差异**：不同人肌肉 activation 模式差异大，需要 calibration。paper 未讨论跨 user 泛化。
4. **Action library 有限**：9 个 API（Grasp, Twist, Hit, etc.），覆盖的 manipulation primitive 有限。扩展到 in-hand manipulation（如 pen spinning）需要更丰富 primitive。
5. **One-shot 的 boundary**：只测 single video → single task。Multi-task video 或 long-horizon video（如 cooking）未测。

---

## 这篇 paper 的 Karpathy-style 启发

这篇 paper 的哲学很合我口味：

**Work with what you have, structure the problem to fit the model's strength.**

VLM 在 multimodal fusion 上有 architectural weakness？不要 retrain VLM，用 prompt engineering 绕过。Vision 无法 capture force？用 cheap EMG armband 补。Robot embodiment 各异？用 code API 抽象掉。

整个 pipeline 没有任何训练。zero training。pure prompting + sensor fusion + code generation。这是 prompt engineering for robot learning 的极简主义路线，与 nanoGPT、micrograd 的精神一脉相承。

未来 robot learning 不一定要 train 大型 VLA model——也许 prompt 一个 general VLM 加上结构化 sensor data，就够了。至少对 manipulation task 来说，这条路线被这篇 paper 验证是 viable 的。

参考 nanoGPT: https://github.com/karpathy/nanoGPT
参考 micrograd: https://github.com/karpathy/micrograd

---

## Project page

https://chain-of-modality.github.io

Google DeepMind Robotics 团队 + Stanford Vision Lab 合作。作者 Chen Wang、Fei Xia、Jie Tan、Jacky Liang 都是 DeepMind Robotics 的人，Li Fei-Fei、C. Karen Liu、Ruohan Zhang 来自 Stanford。

---

# Chain-of-Modality (CoM) 深度讲解

## 1. 核心问题与 motivation

这篇 paper 关注一个具体问题：**robot 如何从一个 single human hand video demonstration 中学习 manipulation task，尤其是那些需要 varying force 的 contact-rich task**。

关键 insight：**pure vision 是信息不充分的**。举例——开瓶盖需要"twist + release + twist"的循环，插入 USB plug 需要"light grasp for re-orient → firm grasp for insertion"。这些 force timing 信息在 RGB pixel 中是 latent 的，但可以由 **EMG armband** 或 **microphone** 之类廉价 sensor 直接 capture。

**Challenge**: 现代 VLM（如 Gemini 1.5 Pro、GPT-4o）虽然支持 long-context multimodal input，但**直接把 video frames + force time-series + hand pose keypoints 全部 concatenate 后一次性 query VLM，会产生 cross-modality misalignment**：VLM 会 neglect 部分 input，或者从错误的 modality 提取信息。这就是 CoM 要解决的核心痛点。

---

## 2. Chain-of-Modality 方法详解

### 2.1 整体架构（对应 Fig. 2）

CoM 的核心是 **sequential modality refinement**，与传统的 merged prompting 形成对比：

**Baseline "Merged"**：
$$\text{VLM}(\text{concat}(V, F, H)) \rightarrow \text{Task Plan}$$
其中 $V$ = video frames, $F$ = force/audio signal, $H$ = hand pose keypoints。一次性输入，一次性输出。

**CoM**：
$$\text{plan}_0 = \text{VLM}(F)$$
$$\text{plan}_1 = \text{VLM}(\text{plan}_0, H)$$
$$\text{plan}_2 = \text{VLM}(\text{plan}_1, V)$$
$$\text{Task Plan} = \text{plan}_2$$

**变量含义**：
- $F \in \mathbb{R}^{T \times 1}$: force signal sequence，$T$ 为总 timestep 数（这里指 down-sampled 后的 frame 数）。
- $H \in \mathbb{R}^{T \times 2 \times 2}$: hand pose，包含 thumb 与 middle fingertip 的 2D pixel location $(x_t, y_t)$，共 2 个 finger × 2D 坐标。
- $V \in \mathbb{R}^{T \times H_{\text{img}} \times W_{\text{img}} \times 3}$: RGB frames。
- $\text{plan}_k$: 第 $k$ 阶段 partial task plan，是一个 structured text（包含 timestep、action name、参数）。

**为什么这个顺序**？作者的 ablation 暗示 force 提供最 coarse 的 segmentation boundary（"什么时候施加力"），hand pose 提供 motion 类别（grasp/twist/release）和方向（counterclockwise 180°），image 最后 ground 到具体 object identity。从 abstract 到 concrete，是信息 entropy 递减的 ordering，类似 top-down parsing。

### 2.2 Prompt 结构（对应 III-B）

CoM 的 prompt 由三部分组成：

1. **Modality description**: 解释 force signal 是 normalized float in $[0, 1]$，hand pose 是 thumb/middle fingertip pixel coordinates 等。
2. **Action set definition**: 可用 skill 库 $\mathcal{A} = \{\text{Grasp}, \text{Release}, \text{Twist}, \text{Move\_to}, \text{Find}, \text{Insert}, \text{Push\_towards}, \text{Wipe}, \text{Hit}\}$，以及每个 action 的参数 schema。
3. **One-shot example**: 给一个 task-irrelevant 的 demonstration（如"按并旋转 apple 和 can"），展示输出格式。**关键设计**：example 中的 object 与 task plan 不与 evaluation task 重合，避免 in-context leakage。

### 2.3 CoM 的实际 trace（对应 Fig. 2 示例）

以 bottle-opening 为例：

| Stage | Input modality | 新增信息 | Partial plan |
|---|---|---|---|
| 1 | Force $F$ | 3 个 force peaks → 3 次 "apply force" | `[t1: force_on] [t1→t2: force] ...` |
| 2 | + Hand $H$ | grasp+twist 动作识别，CCW 180° → CW 180° 交替 | `[grasp+twist CCW 180] [release] [twist CW 180]` |
| 3 | + Image $V$ | left hand 持 bottle，right hand 操作 cap | `Grasp(left, bottle), Grasp(right, cap), Twist(right, CCW, 180)` |

每一 stage 的 output 作为下一 stage 的 conditioning context，形成 **modality-conditioned chain**。

### 2.4 为何 sequential 优于 merged？（intuition 构建）

我从信息论角度推断（论文未明示，但是底层 intuition）：

- VLM 的 attention 机制在 long-context 中存在 **attention dilution**。当 force signal（~100 个 float）和 image patches（~100 frames × ~256 patches each）同时输入时，force 这种低熵、稀疏 peak 的 signal 会被视觉 token 淹没。
- Sequential prompting 等价于 **information bottleneck regularization**：先让 VLM 在小信息量上 commit 一个 hypothesis，再用后续 modality refine，类似 EM 算法的 E-step 和 M-step 交替。
- 这与 Chain-of-Thought (CoT) 在 LLM reasoning 中的成功机制同构：CoT 把 single forward pass 中的 implicit reasoning 拆解成 explicit intermediate steps，CoM 把 multi-modal fusion 拆成 explicit modality-wise reasoning。

参考链接：
- CoT 原始 paper: https://arxiv.org/abs/2201.11903
- Vision-Language models long context: https://arxiv.org/abs/2403.05530 (Gemini 1.5)

---

## 3. 多模态数据采集（对应 III-A, III-D）

### 3.1 Sensor stack

| Modality | Sensor | Sampling | Processing |
|---|---|---|---|
| RGB image | Camera | 60 Hz | 原始 frame |
| EMG force | Armband (8 channels) | 200 Hz | downsample 到 60 Hz，对 8 channel 取 $\max$ |
| Audio | Microphone | (未明示，推测 44.1 kHz) | per-frame loudness（ RMS 或 envelope） |
| Hand pose | HaMeR [Pavlakos et al., CVPR 2024] | 60 Hz | fingertip 2D pixel location |

**Force signal 处理公式**（推测，基于论文描述）：
$$F_t = \max_{c \in \{1, \dots, 8\}} \text{EMG}_{c, t}, \quad \text{EMG}_{c, t} \in \mathbb{R}$$

其中 $c$ 是 EMG channel index，$t$ 是 60 Hz 下的 timestep。取 max 而非 mean 的 intuition：肌肉活动是 sparse firing，max 能更好地捕捉瞬时 peak。

HaMeR 论文链接：https://arxiv.org/abs/2306.10285

### 3.2 为何选 EMG 而非 tactile sensor

EMG 测的是 **muscle activation**，不是 contact force 本身。这是一个聪明的 trade-off：
- Tactile sensor（如 GelSight）需要装在 end-effector 上，但人类 demonstration 时无法安装。
- EMG armband 是 wearable，不干扰人类 manipulation，能 capture "意图施加的力"。
- 缺点：EMG → actual force 是非线性映射，且因人而异。但 CoM 不需要精确牛顿值，只需 **relative force level**（如 20 / 50 / 100 三档）作为 control parameter。

---

## 4. Robot Code Generation（对应 III-C）

### 4.1 API 设计

```python
from skills import Grasp, Release, Twist, Find, Move_to, Insert, Push_towards, Wipe, Hit
```

每个 API 的参数 schema（基于论文 example 反推）：

| API | 参数 | 语义 |
|---|---|---|
| `Find(obj_name)` | str | 返回 object 3D location，由 Gemini 1.5 Pro + RGB-D 实现 |
| `Move_to(hand, target, force=0)` | hand ∈ {left, right}, target=3D pos, force ∈ [0,100] | 移动 gripper 到 target |
| `Grasp(hand, obj_name, force=100)` | hand, obj, force | 闭合 gripper 到指定 force |
| `Twist(hand, direction, angle)` | direction ∈ {CW, CCW}, angle ∈ degrees | 旋转 wrist |
| `Insert(hand, target, force)` | hand, target obj, force | 推进并施加力 |
| `Hit(hand, target, force)` | hand, target, force | 打击 |

### 4.2 Bottle-opening 生成代码

```python
Move_to('left', Find('bottle'))
Grasp('left')
Move_to('right', Find('bottle_cap'))
for _ in range(3):
    Grasp('right')
    Twist('right', 'counterclockwise', 180)
    Release('right')
    Twist('right', 'clockwise', 180)
```

**关键观察**：VLM 自动生成了 `for _ in range(3)` 循环，因为 CoM analysis 中识别到 force signal 有 3 个 peaks。这是 **structured program synthesis** 的体现，而非简单 action sequence 生成。

### 4.3 Plug insertion 中 force 参数

```python
Grasp('right', 'plug', 100)        # firm grasp
Move_to('right', 'box', 20)       # light push to wall for in-hand re-orient
Insert('right', 'power_strip', 100)
```

VLM 把 EMG signal 的 low/high level 映射到 discrete force level（20 vs 100），这是 **symbolic abstraction**——把 continuous signal 离散化为 robot API 可接受的 token。

### 4.4 Find API 的实现

```
Find(obj_name) = depth_backproject( Gemini_1_5_Pro.bbox( RGB-D image, obj_name ) )
```

具体步骤：
1. 将 RGB-D image + object name 喂给 Gemini 1.5 Pro，得到 2D bounding box $(x_1, y_1, x_2, y_2)$。
2. 在 bbox 内取 depth，反投影到 3D point cloud。
3. 取 point cloud centroid 作为 object 3D position $\mathbf{p} \in \mathbb{R}^3$。

$$\mathbf{p} = \frac{1}{N} \sum_{i=1}^{N} \pi^{-1}(u_i, v_i, d_i; K)$$

其中 $\pi^{-1}$ 是 pinhole camera back-projection，$(u_i, v_i, d_i)$ 是 bbox 内 pixel 的 $(u, v, depth)$，$K$ 是 camera intrinsic，$N$ 是 bbox 内 valid depth pixel 数。

参考 Code as Policies: https://arxiv.org/abs/2209.07758

---

## 5. 实验设计

### 5.1 Baseline 矩阵

论文设计了两类 baseline，非常 thorough：

**Modality ablation**：
- Image-only
- w.o. img（force + hand）
- w.o. force（image + hand）
- w.o. hand（image + force）
- All（image + force + hand）

**Reasoning procedure ablation**（对应 Fig. 5）：
- Merg：merge all modalities, single output
- Merg-Sep：merge input, separate output per modality
- Sep-Merg：separate input, single merged output
- Sep-Sep：separate input, separate output, final merge
- Ours (CoM)：separate input, sequential refinement

### 5.2 评估指标

$$\text{Success Rate} = \frac{\text{correct task plans}}{\text{total queries}}$$

$$\text{Similarity Score} = \frac{\text{LCS}(\text{output}, \text{ground truth})}{\max(|\text{output}|, |\text{ground truth}|)}$$

LCS = Longest Common Subsequence。这是 token-level 字符串匹配，捕捉 plan 结构相似度，而非要求 exact match。

### 5.3 任务设计

**Video analysis 任务**（10 videos/task）：
1. Pressing Cube（force-sensitive，单手）
2. Inserting Plug（force + bi-manual 配合）
3. Playing Drum（force + timing）
4. Opening Bottle（bi-manual，长 horizon）

**Robot evaluation 任务**（20 trials/task）：
- Opening Bottle（7 种 bottle，6 unseen）
- Inserting Plug（randomized object placement）
- Wiping Board（不同 marker shape/position）
- Playing Drum（不同 beat）

跨 embodiment：ViperX 与 KUKA 两个 bi-manual platform。

---

## 6. 关键实验结果分析

### 6.1 Table I 解析（modality ablation）

| Setting | Pressing Cube (Gemini/GPT-4o) | Opening Bottle | Inserting Plug | Playing Drum |
|---|---|---|---|---|
| Image-only | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| w.o. img | 0.00 / 0.45 | 0.00 / 0.32 | 0.00 / 0.80 | 0.00 / 0.31 |
| w.o. force | 0.00 / 0.68 | 0.00 / 0.64 | 0.00 / 0.72 | 0.00 / 0.03 |
| w.o. hand | 0.70 / 0.96 | 0.00 / 0.49 | 0.47 / 0.96 | 0.57 / 0.90 |
| All | 0.67 / 0.92 | 0.37 / 0.75 | 0.53 / 0.93 | 0.80 / 0.96 |

**关键 takeaways**：

1. **Image-only 在所有 task 上 success rate = 0**。这是论文最强的 claim 之一：pure vision 无法 capture force-dependent task plan。但 similarity score 非零（如 Pressing Cube 的 0.68），说明 VLM 能猜对一些 action name 但参数错。

2. **w.o. img** 成功率也是 0，说明没有 image 就无法 ground 到具体 object。但 similarity 较高（如 Inserting Plug 0.80），因为 force + hand 已能识别 action skeleton。

3. **w.o. hand 在 Opening Bottle 上 = 0**：这个 task 需要识别 twist 方向（CW/CCW）和 angle（180°），缺 hand pose 完全无法做到。证明 hand pose 对 fine-grained manipulation 是必要的。

4. **All（包含全部 modality）** 才在 Opening Bottle 上有非零 success rate（0.37/0.75），说明这个 task 真正需要全部信息。

5. **Gemini 1.5 Pro 在 Opening Bottle 上 0.37，GPT-4o 上 0.75**——VLM 选择显著影响 fine-grained task 表现。

### 6.2 Fig. 5 解析（reasoning procedure ablation）

Sep-Sep 与 CoM 都优于 Merged 类方法。CoM 比 Sep-Sep 在 Gemini 上 +19%、GPT-4o 上 +17%。

**Intuition**：Sep-Sep 是 "并行分析每个 modality 然后合并"，CoM 是 "sequential refine"。后者优势在于：
- 后续 modality 可以利用前序 commitment 来 disambiguate（例如 force 已识别 3 peaks，hand pose 阶段只需在这些 peak window 内分析 motion）。
- 前序 modality 的输出作为 in-context example，引导 VLM 的 attention。

### 6.3 Table II 解析（real robot）

| Task | Ours | Oracle |
|---|---|---|
| Opening Bottle (ViperX) | 12/20 (60%) | 16/20 (80%) |
| Opening Bottle (KUKA) | 15/20 (75%) | 20/20 (100%) |
| Insert Plug | 15/20 (75%) | 18/20 (90%) |
| Wiping Board (red) | 16/20 (80%) | 20/20 (100%) |
| Wiping Board (blue) | 14/20 (70%) | 16/20 (80%) |
| Playing Drum | 16/20 (80%) | 20/20 (100%) |

**Average**：Ours 73%，Oracle 92%。Oracle 是 human-written code，作为 upper bound。Gap 来源主要是 perception 噪声与 gripper 物理限制，而非 plan 错误。

**Cross-embodiment 验证**：同一个 generated program 在 ViperX 和 KUKA 上都能运行，因为 code API 抽象掉了 robot embodiment。这是 **code-as-policy paradigm 的核心优势**。

---

## 7. 局限性与未来工作

论文自述 limitations：

1. **Audio modality 仅用 loudness**：未利用 frequency、pitch、timbre。Drum 击打的音色、plug 插入的"咔哒声"都包含 rich frequency 信息，未来可引入 spectrogram 或 audio foundation model（如 AudioCLIP, Whisper encoder）。
   - Whisper: https://arxiv.org/abs/2212.04356

2. **Open-loop execution**：generated program 是固定 sequence，不适应 unexpected perturbation。未来需要 closed-loop control，可能结合 Visual Prompting（PIVOT: https://arxiv.org/abs/2402.07772）或 ReKep（https://arxiv.org/abs/2409.01652）实现 constraint-based reactive control。

**我补充的潜在 limitations**：

3. **EMG 个体差异**：不同人肌肉 activation 模式不同，需要 calibration。论文未讨论跨 user 泛化。

4. **Action library 有限**：当前 9 个 API（Grasp, Twist, Hit, etc.）覆盖的 manipulation primitive 有限。扩展到 in-hand manipulation（如 pen spinning）需要更丰富 primitive。

5. **One-shot generalization 的 boundary**：论文只测试 single video → single task。Multi-task video 或 long-horizon video（如 cooking）未测试。

---

## 8. 相关工作联想与对比

### 8.1 与 Code as Policies [Liang et al., ICRA 2023] 的对比

| 维度 | Code as Policies | CoM |
|---|---|---|
| Input | Language instruction | Multimodal video |
| Output | Python code | Python code |
| Force parameter | 无法指定 | 从 EMG 推断 |
| Generalization | Object-level | Object + force level |

CoM 可以看作 Code as Policies 的 **multimodal input extension**。

### 8.2 与 Mimicplay [Wang et al., 2023] 的对比

Mimicplay 也从 human video 学习，但用 play data + latent plan，是 end-to-end policy learning。CoM 是 zero-training approach，完全靠 VLM in-context reasoning。

链接：https://arxiv.org/abs/2302.12422

### 8.3 与 RT-2 / RT-X 的对比

RT-2 是 VLA model，端到端 finetune VLM 输出 action token。CoM 不训练任何参数，pure prompting。RT-2 需要大规模 robot data，CoM 只需一个 video。

RT-2: https://arxiv.org/abs/2307.15818
RT-X: https://arxiv.org/abs/2310.08864

### 8.4 与 ReKep [Huang et al., 2024] 的联想

ReKep 用 VLM 生成 relational keypoint constraint，做 closed-loop trajectory optimization。CoM 生成的 open-loop program 可以与 ReKep 结合：用 CoM 生成 high-level plan，用 ReKep 在每个 sub-skill 内做 closed-loop refinement。

ReKep: https://arxiv.org/abs/2409.01652

### 8.5 与 Chain-of-Thought, Tree-of-Thoughts 的同构

CoM 是 CoT 在 multimodal fusion 上的对应：
- CoT：将 single-modality (text) reasoning 拆解为 explicit intermediate steps。
- CoM：将 multi-modality fusion 拆解为 explicit per-modality analysis steps。

ToT: https://arxiv.org/abs/2305.10601

### 8.6 与 Ego4D / EPIC-Kitchens 的关系

这些 egocentric video dataset 是 potential training/test 数据源，但它们都缺 force modality。CoM 启示：future egocentric dataset 应该 multimodal（video + audio + IMU/EMG）。

Ego4D: https://arxiv.org/abs/2110.07058
EPIC-Kitchens: https://arxiv.org/abs/1804.02748

### 8.7 与 XSkill, XIRL 的 cross-embodiment 联想

XSkill [Xu et al., CoRL 2023] 做 cross-embodiment skill discovery，用 IRL。CoM 通过 code API abstraction 实现 cross-embodiment，更 lightweight 但需要 predefined API。

XSkill: https://arxiv.org/abs/2310.08062

### 8.8 与 Vision-based Manipulation from Single Human Video [Zhu et al., 2024] 的对比

同期工作，也做 single video → manipulation，但用 open-world object graph，未利用 force modality。CoM 与之正交。

链接：https://arxiv.org/abs/2405.20321

---

## 9. Intuition 总结：CoM 为什么 work

构建一个 mental model：

**VLM 是一个 attention-limited reasoning engine**。当 input information rate（bit/second of useful signal）超过 VLM 的 reasoning capacity 时，会出现 information loss。Multimodal merged input 把所有信息一次性 push 给 VLM，导致 attention dilution 和 modality interference。

**CoM 是一种 information scheduling strategy**：按 modality 的 information entropy 从低到高、abstraction level 从 abstract 到 concrete 依次输入，让 VLM 在每个 stage 做出 partial commitment，再用下一 modality refine。这类似于：

- **Bayesian update**：先验（force peak → 3 次 twist）+ 似然（hand pose 显示 grasp-twist motion）+ 后验（image 显示 bottle cap）逐步 refine posterior。
- **Compiler pipeline**：lexical analysis（force tokenization）→ parsing（hand motion grammar）→ semantic analysis（object grounding）。
- **Hierarchical task decomposition**：从 coarse timing 到 fine motion 到 object grounding。

这种 sequential refinement 的本质，是把 VLM 的 single forward pass 拆成 multiple forward pass，每次只关注一个 abstraction level，从而绕过 VLM 在 long-context multimodal fusion 上的 weakness。

---

## 10. 可延伸的研究方向

基于这篇 paper，我能想到的延伸：

1. **Modality ordering ablation**：CoM 用 force → hand → image 顺序。如果反过来（image → hand → force）会怎样？理论上 force 作为最后 refine 可能不 work，因为 image 已经 commit 了 object identity。值得 ablation。

2. **Automatic modality selection**：不是所有 task 都需要全部 modality。Playing Drum 主要靠 audio，Opening Bottle 主要靠 force。可以用 VLM 自己 decide 用哪些 modality，类似 Toolformer 思路。
   - Toolformer: https://arxiv.org/abs/2302.04761

3. **Closed-loop CoM**：在 robot execution 中持续 capture force/audio，让 VLM 实时 refine plan。把 open-loop code generation 升级为 online MPC-style control。

4. **Multimodal ICL scaling**：当前只用了 1 个 example。研究 multimodal ICL 的 scaling law——更多 example 是否带来 better generalization？是否有 emergent ability？

5. **Force-conditioned policy distillation**：用 CoM 生成的 (video, force, code) tuple 作为 pseudo-label，distill 一个小型的 force-conditioned policy network，部署到 edge device。

6. **Cross-human generalization**：不同 demonstrator 的 EMG 模式差异。需要 learn 一个 user-invariant EMG representation。

7. **Touch sensor integration**：除了 EMG，robot 端可装 tactile sensor（DIGIT, GelSight），形成 human-EMG ↔ robot-tactile 的 cross-modal alignment，做 closed-loop force matching。
   - GelSight: https://arxiv.org/abs/2203.07957

---

## 11. 论文 project page & code

- Project page: https://chain-of-modality.github.io
- 作者 Chen Wang 主页（推测）: Google DeepMind Robotics
- Fei Xia, Jie Tan, Jacky Liang 都是 Google DeepMind Robotics 团队成员
- Li Fei-Fei, C. Karen Liu, Ruohan Zhang 来自 Stanford Vision Lab

参考 Stanford VL Lab: https://svl.stanford.edu/
Google DeepMind Robotics: https://deepmind.google/discover/blog/

---

## 12. 一句话总结

CoM 的核心贡献是把 **multimodal video understanding** 这个 VLM 的 weakness，通过 **modality-sequential prompting** 转化为 VLM 擅长的 **single-modality step-by-step reasoning**，并利用 EMG/audio 这种 cheap wearable sensor 补足 vision-only 无法 capture 的 force 信息，最终通过 VLM 的 code generation 能力把 analysis 转化为 cross-embodiment executable robot program。

它本质上是 **prompt engineering for sensor fusion**——一个 lightweight、training-free 的方法，绕开了 VLM 在 multimodal fusion 上的 architectural limitation，并展示了 robot learning 可以从 "train large VLA model" 转向 "prompt general VLM with structured sensor data"。

这种思路非常 Karpathy-style：**work with what you have, structure the problem to fit the model's strength**，与 nanoGPT、micrograd 等 minimalism 哲学一脉相承。

参考 Karpathy 工作：
- micrograd: https://github.com/karpathy/micrograd
- nanoGPT: https://github.com/karpathy/nanoGPT
- "State of GPT" talk: https://www.youtube.com/watch?v=bZQun8Y4L2A
