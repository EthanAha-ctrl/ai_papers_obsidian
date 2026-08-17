---
source_pdf: DrivingDojo Dataset.pdf
paper_sha256: 5d57f51cfdd073218053b40bfde38c4a955d739a3ec9b65ef96e040439071b81
processed_at: '2026-08-03T23:52:05-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 DrivingDojo

## 一句话概括

这群人发现现有的 autonomous driving dataset 都是给 perception 用的，用来 training world model 根本不够用，所以他们从美团配送车的 fleet data 里 curated 出 18k 个 video clip，专门为 world model 设计了三个 subset，并且搞了个 action instruction following 的 benchmark 来测 world model 到底听不听话。

## 为什么这事儿值得做

Karpathy 你想啊，world model 这个概念在 Sora 之后火得一塌糊涂。Sora 能 generate 一分钟的 coherent video，大家就开始幻想：是不是可以把 video generation 当成 real-world dynamics modeling 来做？autonomous driving 这个 domain 特别 natural，因为车就是靠 sensor input 和 action 来 navigate 的。

但是问题来了。当你真的想 train 一个 driving world model，你去翻现有的 dataset：

- **nuScenes**: 1000 个 scene，5.5 hours。基本都在 Boston 和 Singapore 的固定区域兜圈子
- **Waymo**: 1000 个 scene，11 hours。perception 标注很全但 driving behavior 很单一
- **ONCE**: 一百万个 scene 但几乎都是 straight driving
- **OpenDV-2K**: 2000 hours，从 YouTube 上扒的，量大但没 curation

这些 dataset 的 design principle 是 sensor coverage 和 annotation density，你拿它们 training perception model 没问题，但是 training world model 就尴尬了。World model 要的是 behavioral diversity，要的是 ego vehicle 做各种 maneuver、和其他 agent 互动、遇到 rare event 的 video。这些 dataset 里 lane change 比例低得可怜，更别说 U-turn、cut-in、动物横穿马路这种 case 了。

结果就是你 train 出来的 world model，你让它 generate 一个左转的 video，它生成的画面里车还是往前直走。因为它压根没见过左转长啥样。

## 三个 Subset 的 Intuition

DrivingDojo 把 data 分成三块，每一块针对 world model 的一个核心 capability。

### DrivingDojo-Action：让 world model 学会各种动作

这个 subset 7.9k videos，核心 idea 就是 action balance。他们统计了每小时各种 action 发生的次数：

- **nuScenes**: lane change 约 2 次/小时，turn 约 1 次/小时
- **ONCE**: 几乎全是直行
- **DrivingDojo**: lane change 约 15 次/小时，turn 约 12 次/小时，emergency brake 约 8 次/小时

这个差距很直观。你给一个 model 看 100 小时全是直行的 video，然后让它 generate 左转，它生成的画面里地面纹理、路边的树都会变，但是 ego-perspective 的 motion 还是直的，因为它学到的 prior 就是 "车往前开"。

### DrivingDojo-Interplay：让 world model 学会和别人互动

这个 subset 6.2k videos，focus 在 ego vehicle 和其他 dynamic agent 的 interaction。curate 的标准很巧妙：他们用了 Meituan autonomous driving stack 的 PNC signals。

PNC 就是 Planning and Control module。当 PNC 检测到前方的 obstacle 无法通过轻微 steering 或减速 avoid collision 时，就会 trigger 一个 "dangerous interaction" 的 flag。他们直接从 fleet data 里挖这些 flagged 的 clip。

包含的场景：
- **Cut-in/cut-off**: 别的车切进来或被你切出去
- **Meeting**: 对向会车
- **Blocked**: 被 vehicle、motorcycle、pedestrian 挡住
- **Overtaking**: 超车或被超

这个 subset 的价值在于，world model 不光要 generate 正确的 ego motion，还要 generate 其他 agent 对 ego action 的 reasonable response。比如你刹车了，旁边的车看到你减速就 cut in 到你前面。这种 multi-agent dynamics 是单纯 perception dataset 里 capture 不到的。

### DrivingDojo-Open：让 world model 学会处理长尾

这个 subset 3.7k videos，是最有意思的一块。World model 在 pixel space 操作，capacity 远大于 perception model 的 vector representation。理论上它可以 model 任何 visual dynamics，包括长尾的 rare event。

他们从 fleet data 里 manual 筛选了这些 case：
- Unusual weather（暴雨、大雪）
- Foreign objects on road（倒下的树、construction barrier、abandoned vehicle）
- Floating obstacles（绳子、电线）
- Falling objects（快递箱、瓶子、头盔）
- Takeover cases（safety inspector 接管）
- Traffic light 和 boom barrier 的 interaction

每个 video 都有 text description。Figure 3b 的 word cloud 里高频词包括 "barrier"、"construction"、"animal"、"pedestrian" 这些。

这个 subset 的 intuition 是：autonomous driving 的 safety 主要受 long-tail event 威胁。如果 world model 能 simulate 这些 rare case，planner 就可以提前 experience 这些 scenario，类似于一个 data augmentation for policy training。

## AIF Task 到底在测什么

Action Instruction Following 是这篇 paper 的核心 benchmark innovation。

formulation 很简单：

$$\{I_{t+1}, ..., I_{t+k}\} = f_\theta(I_t, \{A_t, ..., A_{t+k}\})$$

变量含义：
- $I_t$：初始 frame 的 image
- $A_t = (\Delta x_t, \Delta y_t)$：第 $t$ 帧的 action，是 ego-centric 的 relative motion
- $f_\theta$：world model，参数 $\theta$
- $k$：prediction horizon

这里的 action 不是 vehicle control signal（steering angle、acceleration），而是 camera trajectory。具体计算公式在 Appendix 里：

$$\begin{pmatrix} A_n \\ 1 \end{pmatrix} = \begin{pmatrix} R_n & T_n \\ 0^3 & 1 \end{pmatrix}^{-1} \begin{pmatrix} T_{n+1} \\ 1 \end{pmatrix}$$

变量解释：
- $R_n$：第 $n$ 帧 camera 到 world coordinate 的 $3 \times 3$ rotation matrix
- $T_n$：第 $n$ 帧 camera 到 world coordinate 的 $3 \times 1$ translation vector
- $0^3$：$1 \times 3$ 的 zero row，用来构造 homogeneous coordinate
- $A_n$：$3 \times 1$ vector，表示第 $n+1$ 帧 camera position 在第 $n$ 帧 camera coordinate system 里的坐标

intuition 就是：把下一帧的 world coordinate 位置转换到当前帧的 camera coordinate 里，得到一个 relative displacement。这个 representation 的好处是 ego-centric，不依赖 absolute world coordinate。

## AIF Error 的评估 Trick

这里有个非常 clever 的设计。world model 输出的是 video frame，不是 trajectory。怎么知道生成的 video 是否符合 action instruction？

他们用 SfM（Structure from Motion）从生成的 video 反推 camera trajectory，然后和给定的 action instruction 对比。

具体公式：

$$E_x^{AIF}, E_y^{AIF} = \frac{\sum_{i=0}^{k} |A_{t+i} - \tilde{A}_{t+i} \cdot \hat{S}|}{k+1}$$

变量：
- $A_{t+i}$：ground-truth action instruction at frame $t+i$
- $\tilde{A}_{t+i}$：SfM 从 generated video 估计的 trajectory（up-to-scale）
- $\hat{S}$：scale factor，用来 align SfM 估计的 scale 和 ground-truth 的 metric scale
- $k+1$：总 frame 数

Scale factor 怎么算：

$$\hat{S} = \arg\min_S \sum_{i=0}^{N} |A_{t+i} - \tilde{A}_{t+i} \cdot S|$$

用前 $N=10$ 帧 minimize error 估计 scale，然后 evaluate 全部 $k+1$ 帧的 mean absolute error。lateral error 记为 $E_y^{AIF}$，longitudinal error 记为 $E_x^{AIF}$。

这个 metric 的妙处在于：它不依赖 model 内部 representation，直接从 visual output 反推 action，测的是 "你生成的 video 视觉上看起来像不像在执行这个 action"。

实现细节：moving objects 会干扰 SfM，所以他们用 instance mask 把前景移动物体 mask 掉，只 reconstruct static background。对于 DrivingDojo 自己的 video，camera intrinsic 用 ground-truth；对于 OpenDV-2K 的 unknown intrinsic，SfM 同时 estimate intrinsic 和 extrinsic。

## Baseline Model 怎么改的

baseline 用 Stable Video Diffusion (SVD)，这是一个 latent diffusion 的 image-to-video model，本身没有 action conditioning。

改动很 minimal：
1. 用 MLP 把 action sequence 编码成 1024 维 vector
2. action feature 和 first-frame image feature concatenate
3. 一起喂进 SVD 的 U-Net

training setup：
- 初始化：SVD-XT checkpoint
- EDM framework [Karras et al. 2022]
- fps=5，motion_bucket_id=127
- AdamW optimizer，learning rate $10^{-5}$
- 16 张 A100 80G，batch size 32，50K iterations
- Classifier-free guidance：action feature dropout 20%
- 两种 resolution：1024×576 for 14 frames（visual quality），576×320 for 30 frames（AIF）

inference 用 DDIM sampler 25 steps。

这个 baseline 基本就是 "把 SVD 拿来 fine-tune + 加个 action encoder"，没有任何 architecture innovation。paper 自己也承认这只是 baseline，focus 在 dataset 上。

## 实验结果的核心发现

### Visual Quality（Table 3）

| Fine-tuning Dataset | FID↓ | FVD↓ |
|---|---|---|
| OpenDV-2K (2059h) | 18.27 | 321.05 |
| nuScenes (5.5h) | 24.17 | 580.94 |
| DrivingDojo (150h) | 19.20 | 343.91 |

DrivingDojo 只有 150 hours 但 FVD 343.91，比 2059 hours 的 OpenDV-2K 只差一点点，比 nuScenes 好非常多。这说明 curation quality 比 raw scale 重要 for visual fidelity。

### AIF Performance（Table 4）

| Action Type | Test Dataset | $E_x^{AIF}$ | $E_y^{AIF}$ |
|---|---|---|---|
| GT (real video) | DrivingDojo | 0.036m | 0.019m |
| In-Domain action | DrivingDojo | 0.100m | 0.062m |
| Random action | DrivingDojo | 0.173m | 0.110m |
| Random action | OpenDV-2K (zero-shot) | 0.238m | 0.136m |

GT row 是 SfM 在 real video 上的 reconstruction error，作为 lower bound，大约 3cm。In-domain generation 的 error 约 10cm，相当 reasonable。Zero-shot 到 OpenDV-2K 的 error 24cm，但还算 follow action。

### Cross-dataset Comparison（Table 5）

| Training | $E_x^{AIF}$ | $E_y^{AIF}$ |
|---|---|---|
| DrivingDojo | 0.238m | 0.136m |
| ONCE | 0.255m | 0.239m |
| nuScenes | - | - |

ONCE-trained model 的 $E_y^{AIF}$ 是 0.239m，几乎是 DrivingDojo 的两倍。paper 里说 ONCE-trained model 即使给 turn left 的 instruction，生成的 video 还是直行。这就是 data bias 的直接体现——ONCE 几乎没有 lateral maneuver data，model 学不到 lateral motion 的 visual prior。

## 一些 Failure Mode

Figure 8 展示了 hallucination：
- Object 突然消失（比如前面有辆车，生成几帧后车没了）
- 给一个 unrealistic action（比如在直路上强制右转），model 会凭空 hallucinate 出一条不存在的 road

这揭示了 world model 的一个 fundamental issue：它学的是 $p(video | action)$ 的 conditional distribution，当 action 落到 training distribution 之外，generation 会 break down。model 倾向于 satisfy action instruction 而破坏 scene consistency。

## 我的 Take

这篇 paper 的价值主要在 data 和 benchmark，不在 model。几个思考点：

**Data curation 的哲学**。perception dataset 追求 sensor coverage，world model dataset 追求 behavioral diversity。这是两个 fundamentally 不同的 curation principle。DrivingDojo 的 PNC signal-based curation 是个很 pragmatic 的方案——直接利用 autonomous driving stack 已经在做的 scenario classification。

**AIF metric 的可推广性**。用 SfM 反推 action 这个 idea 可以推广到其他 domain。任何有 ego-motion 的 video generation task 都可以用类似 metric。但 SfM 的 limitation 也明显——generated video visual quality 不够时 SfM 会 fail。未来可能需要 learning-based 的 action estimator。

**Single camera 的 trade-off**。他们为了 maximize video diversity 只用单 camera，但 multi-view 对 downstream planning task 很重要。这是 diversity vs. sensor richness 的 trade-off。未来可能需要 multi-camera 但保持 action diversity 的 dataset。

**和 model-based RL 的 connection**。DrivingDojo 的 action annotation 是 ego-centric relative motion，这和 Dreamer-style 的 world model formulation 很 compatible。如果能加 reward signal，这个 dataset 可以直接用于 model-based RL 的 offline training。

**长尾的真正价值**。DrivingDojo-Open 的 3.7k videos 量不大，但覆盖了非常有价值的长尾。autonomous driving 的 safety 瓶颈就在长尾。如果 world model 能 simulate 这些 case，planner 就能提前 experience，类似一个 targeted data augmentation。

## 参考链接

- DrivingDojo Project: https://drivingdojo.github.io
- Stable Video Diffusion: https://arxiv.org/abs/2311.15127
- Sora (OpenAI tech report): https://openai.com/sora
- Genie (DeepMind): https://arxiv.org/abs/2402.15391
- UniSim: https://arxiv.org/abs/2310.06114
- GAIA-1 (Wayve): https://arxiv.org/abs/2309.17080
- DriveDreamer: https://arxiv.org/abs/2309.09777
- Drive-WM: https://arxiv.org/abs/2312.06855
- OpenDV-2K / Vista: https://arxiv.org/abs/2403.09630
- Think2Drive: https://arxiv.org/abs/2402.16720
- DayDreamer: https://arxiv.org/abs/2206.14176
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- LeCun AMI: https://openreview.net/pdf?id=bZ59a4rYOOF
- DreamerV3: https://arxiv.org/abs/2301.04104
- COLMAP: https://colmap.github.io
- EDM: https://arxiv.org/abs/2206.00364
- Classifier-free Guidance: https://arxiv.org/abs/2207.12598
- nuScenes: https://www.nuscenes.org
- Waymo Open Dataset: https://waymo.com/open
- ONCE: https://once-for-auto-driving.github.io
- nuPlan: https://www.nuplan.org
- CARLA: https://carla.org

Karpathy，你觉得这种 data-centric 的 approach 对 world model 发展够不够？还是说 architecture innovation 才是真正的 bottleneck？我个人觉得当前阶段 data 是更 limiting 的 factor，model architecture (SVD-based) 已经能 generate 不错的 video，关键是 training data 的 behavioral coverage 决定了 world model 能 simulate 什么。

---

# DrivingDojo Dataset 深度解析

Andrej，这篇 paper 是一个非常有意思的工作，来自中科院自动化所和美团自动驾驶团队的 collaborative effort。核心 motivation 在于：当前的 driving world models 受限于 dataset 的 video diversity，无法充分释放其 modeling capability。他们构建了一个 tailor-made 的 dataset 来 training interactive world models。

## 1. Motivation 与 Core Problem

当前 driving world models 的 bottleneck 在于 data。已有的 autonomous driving datasets 比如 nuScenes、Waymo、ONCE 都是 perception-oriented 的 curation 思路，这导致 driving patterns 单一，multi-agent interactions 稀缺。Perception dataset 的 design principle 是 coverage of sensor configurations 和 annotation density，而 world model 需要的是 behavioral diversity 和 dynamics richness。

Key insight：world model 作为 real-world simulator，需要 follow action prompts accurately，这要求 training data 包含 complete set of driving maneuvers 和 diverse multi-agent interplay。Perception-oriented data 倾向于 highway cruising 或者 straight driving，lateral maneuvers 如 lane change、U-turn 极度稀缺。

## 2. Dataset 三大 Pillars

DrivingDojo 设计了三个 subsets，每个针对 world model 的不同 capability：

### 2.1 Action Completeness (DrivingDojo-Action, 7.9k videos)

这个 subset 强调 longitudinal 和 lateral maneuvers 的 balance。包含：
- **Longitudinal**: acceleration, deceleration, emergency braking, stop-and-go
- **Lateral**: lane keeping, lane changing, U-turn, overtaking

Figure 3a 对比了 nuScenes、ONCE 和 DrivingDojo 的 hourly event counts。ONCE dataset 几乎只有 straight driving，nuScenes 稍好但 turn 和 lane change 极少。DrivingDojo 显著更 balanced。

这个 design 的 intuition 很清晰：如果 training data 中 turn/lane change 极度稀缺，world model 在 inference 时无法 generate lateral motion 的 video，即使给 action instruction 也不行。Table 5 的 zero-shot evaluation 证明了这一点——ONCE-trained model 始终 generate 直行视频，即使 instruction 是 turn left/right。

### 2.2 Multi-agent Interplay (DrivingDojo-Interplay, 6.2k videos)

这个 subset 关注 ego vehicle 与其他 dynamic agents 的 interaction。Curated scenarios 包括：
- **Cutting in/off**: 其他 vehicle 突然切入 ego lane，或者 ego 切入他人 lane
- **Meeting**: 对向 vehicle encounter
- **Blocked**: 被 vehicles、motorcycles、pedestrians 阻挡
- **Overtaking / being overtaken**

Curation 策略基于 PNC (Planning and Control) signals。当 ego vehicle 无法通过 steering wheel 或轻微减速避免 collision 时，定义为 PNC interaction case。这是一个非常聪明的 heuristic——直接从 autonomous driving stack 的 intervention signal 中挖掘 interaction-rich 的 clips。

### 2.3 Open-world Knowledge (DrivingDojo-Open, 3.7k videos)

这是最 interesting 的 subset。World model 在 pixel space 操作，modeling capacity 远高于 perception model 的 low-dimensional vector representation。这意味着 world model 理论上可以 capture open-world 的 long-tail dynamics，比如 animals crossing、falling objects、debris on road。

这个 subset 来自 fleet data 中的 unusual cases：
- Unusual weather
- Foreign objects on road surface (fallen trees, construction barriers, abandoned vehicles)
- Floating obstacles (ropes, power lines)
- Falling objects (parcels, bottles, helmets)
- Takeover cases (safety inspector intervention)
- Interactions with traffic lights 和 boom barriers

每个 video 都有 text description，这是用 LLM-style 的 annotation 方式描述 rare event。Figure 3b 的 word cloud 显示了 high-frequency keywords。

## 3. Data Curation Pipeline

从 900,000 videos（约 7500 hours）的 pool 中 curate 出 18k clips。Data sources 包括：

1. **Intervention data**: safety inspector 接管时的 data，通常是 challenging scenario
2. **AEB data**: automatic emergency braking 触发，通常是有 obstacle 或 sudden stop
3. **Random 30-second general videos**: baseline diversity
4. **Selected distinct scenarios**: traffic light changes, barrier opening, turns, crossings, encounters, lane changes, pedestrian interactions
5. **Manually sorted rare data**: foreign objects, floating obstacles, falling/rolling objects

PII removal 用 high-precision license plate 和 face detector (YOLO-based [31]) 检测并 blur，然后 manual double-check。

## 4. Action Instruction Following (AIF) Task Formulation

这是 paper 的核心 contribution 之一。给定 initial image $I_t$ 和 action sequence $\{A_t, ..., A_{t+k}\}$，world model $f_\theta$ 预测 future states：

$$\{I_{t+1}, ..., I_{t+k}\} = f_\theta(I_t, \{A_t, ..., A_{t+k}\})$$

其中 $A_t = (\Delta x_t, \Delta y_t)$ 是 trajectory，表示下一帧 camera position 在当前 camera coordinate system 中的坐标。$k$ 是 prediction horizon。

### 4.1 Action Representation 的数学细节

Action info $A_n$ 的计算公式（Appendix 中的 Equation 3）：

$$\begin{pmatrix} A_n \\ 1 \end{pmatrix} = \begin{pmatrix} R_n & T_n \\ 0^3 & 1 \end{pmatrix}^{-1} \begin{pmatrix} T_{n+1} \\ 1 \end{pmatrix}$$

变量解释：
- $R_n$: 第 $n$ 帧 camera 到 world coordinate system 的 rotation matrix (3×3)
- $T_n$: 第 $n$ 帧 camera 到 world coordinate system 的 translation vector (3×1)
- $0^3$: 1×3 的 zero row vector
- $A_n$: 3×1 vector，表示第 $n+1$ 帧 camera position 在第 $n$ 帧 camera coordinate system 中的坐标

World coordinate system 是 ENU (East-North-Up)，camera coordinate system 中 x 轴向右、y 轴向下、z 轴向前。第一帧的 world coordinate 被 normalize 到 origin。

这个 formulation 的 intuition：action 不是绝对的 vehicle pose，而是 relative motion 在 ego-centric frame 中的表示。这符合 world model 的 Markovian assumption——下一帧 state 只依赖当前 frame 和当前 action。

## 5. Model Architecture

Baseline 基于 Stable Video Diffusion (SVD) [2]。SVD 是 latent diffusion model for image-to-video generation，原本没有 action conditioning。

### 5.1 Architecture 修改

1. **Action Encoder**: 用 MLP 将 action sequence 的 value 编码成 1024-dimensional vector
2. **Conditioning Fusion**: action feature 与 first-frame image feature concatenate，传入 U-Net [40]

这是一个相对 minimal 的修改。SVD 的 U-Net 保留原结构，只是 input channel 增加。

### 5.2 Training Details

- 初始化：SVD-XT checkpoint
- Framework: EDM [32]
- fps: 5, motion_bucket_id: 127
- Optimizer: AdamW [35], learning rate $1 \times 10^{-5}$
- Hardware: 16× NVIDIA A100 (80G), batch size 32
- Iterations: 50K
- Classifier-free guidance [25]: action feature dropout ratio 20%
- Resolution: 1024×576 (14 frames) for visual prediction, 576×320 (30 frames) for AIF

Inference 用 DDIM sampler 25 steps。

## 6. Evaluation Metrics

### 6.1 Visual Quality

- **FID** (Frechet Inception Distance) [23]: 随机选 5000 frames 评估
- **FVD** (Frechet Video Distance) [46]: 生成 256 videos 评估，用 UCF FVD evaluation code

### 6.2 Action Instruction Following Error

这是 paper 的核心 metric innovation。Definition：

$$E_x^{AIF}, E_y^{AIF} = \frac{\sum_{i=0}^{k} |A_{t+i} - \tilde{A}_{t+i} \cdot \hat{S}|}{k+1}$$

其中：
- $A_{t+i}$: ground-truth action instruction at frame $t+i$
- $\tilde{A}_{t+i}$: 从 generated video 用 SfM (COLMAP) 估计的 camera trajectory
- $\hat{S}$: scale factor，通过 minimize 前 $N$ 帧的 error 估计

Scale factor $\hat{S}$ 的估计：

$$\hat{S} = \arg\min_S \sum_{i=0}^{N} |A_{t+i} - \tilde{A}_{t+i} \cdot S|$$

Intuition：SfM 估计的 trajectory 是 up-to-scale 的，需要 align 到 ground-truth 的 metric scale。用前 $N=10$ 帧 align scale，然后 evaluate 全部 $k+1$ 帧的 mean absolute error。

这个 metric 的 clever 之处：不需要 model 内部输出 trajectory，直接从 visual output 反推 action，这 measure 的是 video 是否 visually consistent with 指定的 action。

Implementation detail：moving objects 会影响 SfM reconstruction 质量，所以用 instance masks occlude foreground moving objects。

## 7. Experimental Results Analysis

### 7.1 Visual Prediction (Table 3)

| Method | Fine-tuning | FID | FVD |
|--------|-------------|-----|-----|
| SVD | OpenDV-2K | 18.27 | 321.05 |
| SVD | nuScenes† | 24.17 | 580.94 |
| SVD | DrivingDojo | 19.20 | 343.91 |

注意 DrivingDojo 的 FVD (343.91) 显著优于 nuScenes (580.94)，但略逊于 OpenDV-2K (321.05)。这可能因为 OpenDV-2K 有 2059 hours，远多于 DrivingDojo 的 150 hours。但 DrivingDojo 在 action-controllable generation 上更优。

### 7.2 Action Instruction Following (Table 4)

| Action Type | Test Dataset | FID | FVD | $E_x^{AIF}$ | $E_y^{AIF}$ |
|-------------|--------------|-----|-----|-------------|-------------|
| In-Domain | DrivingDojo (GT) | - | - | 0.036m | 0.019m |
| In-Domain | DrivingDojo | 37.07 | 658.72 | 0.100m | 0.062m |
| Out-of-Domain | DrivingDojo | 38.30 | 716.44 | 0.173m | 0.110m |
| Out-of-Domain | OpenDV-2K* | 24.27 | 442.67 | 0.238m | 0.136m |

关键观察：
- **GT row**: 用 real images 测试 SfM 的 reconstruction error，作为 lower bound
- **In-Domain**: 用 training data 的 initial frame + ground-truth action，generation 的 AIF error 约 10cm
- **Out-of-Domain action**: 用 training data 的 initial frame + random action，error 上升到 17cm
- **Zero-shot to OpenDV-2K**: 用 OpenDV-2K 的 initial frame + random action，error 24cm，但仍然 reasonable

### 7.3 Cross-dataset Comparison (Table 5)

| Training set | Test set | FID | FVD | $E_x^{AIF}$ | $E_y^{AIF}$ |
|--------------|---------|-----|-----|-------------|-------------|
| DrivingDojo | OpenDV-2K* | 24.27 | 442.67 | 0.238m | 0.136m |
| ONCE | OpenDV-2K* | 28.37 | 473.59 | 0.255m | 0.239m |
| nuScenes | OpenDV-2K* | - | - | - | - |

ONCE-trained model 的 $E_y^{AIF}$ (0.239m) 几乎是 DrivingDojo (0.136m) 的两倍。这证实了 hypothesis：ONCE 缺乏 lateral maneuver data，导致 model 无法 follow lateral action instructions。

## 8. Qualitative Results

### 8.1 Action Generalization (Figure 7a)

Model 能 generalize 到 OOD actions，比如强行开上人行道。还能 zero-shot 应用到 OpenDV-2K 做 lane change，到 nuScenes 做 backing maneuver。这说明 model 学到了某种 action-conditioned 的 generative capability，而非简单 memorize training distribution。

### 8.2 Interaction Simulation (Figure 7b)

Model 能根据 ego action 改变其他 agents 的 behavior：
- 如果 ego 前进，pedestrian yield
- 如果 ego 停下，delivery person 停在 narrow road 等待

这是 emergent 的 multi-agent interaction modeling，虽然 paper 没有定量 evaluate。

### 8.3 Hallucination (Figure 8)

Model 的 failure modes：
- Object 突然消失
- 给定 unrealistic action（如强制右转）时，model 会 hallucinate 一条不存在的 road

这揭示了当前 world model 的 limitation：action-conditioned generation 在 OOD action 时会 break down，model 倾向于 satisfy action instruction 而 invent 不存在的 scene structure。

## 9. Limitations 与 Future Directions

1. **Single camera only**: 为了 maximize video diversity，减少了 sensor 数量
2. **Baseline model 没有专门 design**: 这只是 SVD + action encoder 的 minimal modification
3. **Hallucination 问题**: object disappearance 和 non-existent road generation
4. **Short-horizon prediction**: 当前只能 generate 短视频，long-horizon 留给 future work
5. **Driving policy 未探索**: long-tail cases 对 policy learning 的价值未挖掘

## 10. Personal Reflections 与 Related Connections

这个 work 让我想到几个 broader 的 research threads：

### 10.1 World Model 的 Data Scaling

Sora [Sora tech report] 证明了 video data scaling 对 world model 的重要性。DrivingDojo 是 domain-specific 的 data scaling 尝试。Key question: 是 general video data 更有效，还是 domain-specific curated data 更有效？OpenDV-2K (2059 hours) vs DrivingDojo (150 hours) 的 FVD 对比 (321 vs 343) 暗示 curation 比 raw scale 更重要 for action controllability。

### 10.2 Action-conditioned Generation 的 Future

当前的方法是 action sequence 作为 condition 输入。但更 powerful 的 formulation 可能是：
- **Language-conditioned**: "turn left at the next intersection"
- **Goal-conditioned**: 给定 target pose
- **Interactive**: agent 可以在 generation 过程中调整 action

Genie [5] 和 UniSim [55] 都探索了 interactive generation，但 driving domain 的 interactivity 更 challenging because of safety constraints。

### 10.3 SfM-based Evaluation 的 Insight

用 SfM 从 generated video 反推 action 是一个巧妙的 evaluation strategy。但这 assume 了 generated video 的 visual quality 足够支持 SfM。如果 model generate 的 video 有 hallucination，SfM 可能 fail 或给出 noisy estimate。未来可能需要 learning-based 的 action estimator 来替代 SfM。

### 10.4 Connection to Model-based RL

Think2Drive [34] 用 world model 在 CARLA 中做 model-based RL。DrivingDojo 的 real-world data 可以 enable 类似的方法在 real-world driving 中。关键 challenge 是 real-world 的 reward signal 和 safety constraints。

### 10.5 Data Curation 的 Meta-learning

DrivingDojo 的 curation pipeline 用 PNC signals 来 identify interaction-rich clips。这启发了一个想法：能否用 meta-learning 自动 learn curation policy？比如用 world model 的 prediction error 来 identify challenging clips，然后 prioritize 这些 clips in training。

## Reference Links

- DrivingDojo Project Page: https://drivingdojo.github.io
- Stable Video Diffusion: https://arxiv.org/abs/2311.15127
- Genie: https://arxiv.org/abs/2402.15391
- UniSim: https://arxiv.org/abs/2310.06114
- GAIA-1: https://arxiv.org/abs/2309.17080
- DriveDreamer: https://arxiv.org/abs/2309.09777
- Drive-WM: https://arxiv.org/abs/2312.06855 (CVPR 2024)
- ADriver-I: https://arxiv.org/abs/2311.13549
- DriveDreamer2: https://arxiv.org/abs/2403.06845
- OpenDV-2K (Vista): https://arxiv.org/abs/2403.09630
- Think2Drive: https://arxiv.org/abs/2402.16720
- DayDreamer: https://arxiv.org/abs/2206.14176
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- LeCun's Path Towards AMI: https://openreview.net/pdf?id=bZ59a4rYOOF
- nuScenes: https://www.nuscenes.org
- Waymo Open Dataset: https://waymo.com/open
- ONCE Dataset: https://once-for-auto-driving.github.io
- nuPlan: https://www.nuplan.org
- COLMAP: https://colmap.github.io
- EDM: https://arxiv.org/abs/2206.00364
- Classifier-free Guidance: https://arxiv.org/abs/2207.12598

这个 paper 的核心 contribution 是 data-centric 的——它 argue 了 data curation 对 world model 的关键作用，并通过 AIF benchmark 量化了 action controllability。虽然 baseline model 只是 minimal modification，但 dataset 本身为 future research 提供了 valuable resource。
