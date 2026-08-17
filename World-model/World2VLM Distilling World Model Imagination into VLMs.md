---
source_pdf: World2VLM Distilling World Model Imagination into VLMs.pdf
paper_sha256: a0aff8c483da5e2b4d42be3db6d9e0b2496e3f94467f518c8850152e0deb957a
processed_at: '2026-08-13T05:33:54-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 World2VLM

## 一句话讲清楚

现在的VLM看图很行，但一问"你往前走三米再右转50度，电视机会在哪"就懵了。之前的解法是推理时调一个world model帮它想象。这paper说：别推理时作弊，训练时用world model出一万道题刷VLM，刷完它自己就会算了。

---

## 问题是什么

你给GPT-4o看张房间照片，问它"电视右边那个柜子离墙多远"，它能答个大概。但你问它"如果我往前走2米然后左转30度，电视还会在画面里吗"，它就开始胡说八道。

原因很直觉：当前VLM训练数据里90%是"看一张图答问题"，缺的是"看一张图然后脑补camera移动后场景怎么变"这种**dynamic spatial reasoning**。模型没练过这个肌肉。

---

## 之前的两条路都不太行

**第一条路：造更多spatial QA数据**

就是用simulator生成一堆"电视在左边""桌子在中间"的标注，堆给VLM训。问题：这些数据是static的，不教模型"action之后场景怎么演化"。就像你给学生一万道"求三角形面积"的题，但从来不教他"这个三角形旋转30度后面积变不变"。

**第二条路：推理时挂个world model**

MindJourney这派说，既然VLM不会想象，那推理时挂一个generative world model，让world model生成"你move forward之后看到的画面"，VLM在那画面上答。就像考试带个计算器。

问题是慢。每答一个问题，要跑world model好几次生成候选视角，VLM再给每个视角打分，最后选最好的。而且VLM本身还是不会，下次还得挂这个外挂。

---

## World2VLM的核心idea

把world model从"考试时的计算器"变成"考前的一万道练习题"。

具体流程：

1. 拿一堆房间照片当anchor
2. 给每张照片编一段camera动作："往前走3米然后左转50度"
3. 让world model（SVC或HY-WorldPlay）生成动作后应该看到的画面
4. 从这"前后两帧+已知动作"造出8种不同问法的题
5. 用这10万道题训Qwen2.5-VL，先SFT再GRPO
6. 训完之后world model扔掉，VLM自己就会想象了

类比AlphaZero：MCTS是想象工具，但最终policy network把想象压缩成直觉，下棋时几乎不搜索。World2VLM一样，world model是训练时的想象工具，最终VLM把想象能力内化进参数，推理时零外部调用。

---

## 三个关键设计直觉

### 设计一：Max-Displacement Pairing

不把anchor和邻近帧配对（往前走0.1米的那帧），而是配trajectory里最远的那帧（比如往前走5米+转50度后的帧）。

为什么？0.1米的two frames视觉上几乎一样，模型靠pixel-level小差异就能蒙对答案，没学到真东西。5米+50度rotation的两帧大量像素disocclusion/occlusion，模型必须真mental simulate才能答对。

就像教小孩做加法，别给他1+1这种trivial题，给他37+58这种必须真正算的。

### 设计二：Bidirectional Task Suite

每条transition造8种题，分两个方向：

**Inverse方向**（看现象推因）：
- 给两帧问"camera移动了几米"
- 给两帧问"camera转了几度"
- 给两帧问"camera的action sequence是什么"

**Forward方向**（知因推果）：
- 给一帧+action问"这个物体现在的bbox在哪"
- 给一帧+action问"这个物体还visible吗"
- 给一帧+action问"两个bbox是不是同一个object"

为什么双向？只学inverse，模型靠view diff模式匹配就能猜motion，没真懂。只学forward，模型当simulator但不会反推。两边一起学，强制模型建立一个既能解释观察又能预测后果的spatial representation。

这跟因果推理里forward query和abduction必须对称支持一个道理。完整的spatial understanding = 解释能力 + 预测能力，缺一不可。

### 设计三：SFT然后GRPO

SFT阶段教知识：10万道transition题，模型学会"motion和scene change怎么对应"。

GRPO阶段calibrate表达：1千道精选题，每个采4个rollout，用task-aware reward做group-relative advantage optimization。

GRPO的reward分5个项：
- format reward：输出格式对不对
- semantic reward：答案语义对不对
- numeric reward：数值精度（比如translation估了2.3米，gt是2.5米，在容忍带内得分）
- geometric reward：bbox的IoU/center/size/L1
- order reward：action sequence顺序对不对

不同task激活不同reward项。比如A1（translation estimation）用format+semantic+numeric，D1（bbox prediction）用format+geometric+validity。

看Table 8的case study最直观：

| Task | GT | SFT输出 | GRPO输出 |
|------|----|---------|-----------|
| 估计depth | 5.4m | 10.5m（严重高估） | 5.1m（接近） |
| 估计物体size | 42cm | 450cm（量级错误） | 50cm（calibrated） |
| perspective判断 | right | left（错） | right（对） |

SFT让模型"知道答案是什么类型"，但数值和离散决策没校准。GRPO用reward把输出层calibrate到位。这就是为什么SFT+GRPO比纯SFT在所有benchmark上都更高。

---

## 结果讲什么

Base Qwen2.5-VL-7B在SAT-Real上44.67分。

Test-time挂world model（MindJourney风格）：31.33，**反而掉了13分**。这个数据很反直觉，但仔细想想有道理——真实动态任务里，world model生成的candidate view可能跟query视角对不齐，VLM评分时被误导。

World2VLM-SFT（SVC当teacher）：64.00，涨19分。

World2VLM-GRPO（SVC当teacher）：72.67，涨28分。

Avg四个benchmark：base 36.63 → 52.07（SVC）/ 52.61（HY），涨15分左右。test-time coupling只涨2分。

SAT-Real的task-level breakdown最说明问题：

Egocentric Movement：base 52.17 → GRPO 100.00。直接saturate，因为任务足够窄+supervision足够密，模型完全学会了"camera移动后scene怎么更新"。

Object Movement：base 21.74（接近随机猜）→ GRPO 65.22。从几乎完全不会到大部分会，但没saturate，说明cross-view object tracking更难。

Perspective Taking：base 48.48 → SFT 42.42（反而降）→ GRPO 57.57。SFT阶段下降了，GRPO救回来。说明perspective taking需要transition supervision + 输出层calibration双管齐下。

---

## 两个teacher的互补很有意思

**SVC**：camera-conditioned novel view synthesis，diffusion-based。几何严格，viewpoint一致性强，但大motion时有渲染artifact（拉伸边缘、texture tearing）。在SAT-Real和VSI-Bench上更强——这俩benchmark直接考camera geometry。

**HY-WorldPlay**：基于HunyuanVideo的action-conditioned video world model。感知真实性强，temporal continuity强，但可能"reinterpret"fine-grained camera motion让它看起来plausible但不严格match。在SAT-Synth和MindCube上更强——这俩考mental simulation和temporal reasoning。

不同teacher把不同prior注入student。SVC注入tight camera-geometry alignment，HY注入stronger perceptual realism和temporal continuity。Paper没说哪个更好，而是说不同teacher适合不同下游能力。

---

## Teacher error怎么传播（最诚实的limitation）

World2VLM的根本瓶颈：teacher质量决定student上限。Paper的Appendix G很坦诚地讲了这个。

Teacher出错通过四种方式污染supervision：

**Action-observation mismatch**：world model渲染的endpoint跟标称action对不上（over-rotate或drift），那A1-A4的motion标签就和image pair视觉不一致，model学到错误的motion-to-observation mapping。

**Geometric distortion**：object silhouette扭曲、bbox变形，D1的localization target变得没意义。

**Appearance hallucination/disappearance**：world model错误生成/删除了object，D2/D4的visibility和identity标签就错了。

**Large-motion instability**：max-displacement策略把样本放在trajectory里teacher最容易出错的位置。这是设计trade-off——信息量大但噪声也大。

目前的缓解：多teacher ensemble、object-level filter（confidence/area/border-margin/track-consistency）、prompt-level validation、mixed-source training。但没做explicit teacher reliability modeling，所有通过hard filter的样本被同等对待。未来方向是confidence-weighted distillation和task-aware teacher selection。

---

## 我的几个take

**这个framing会stuck**：把world model从inference-time tool重定位成training-time teacher，这个概念框架很clean，大概率会被后续工作follow。跟AlphaZero的MCTS-to-policy-distillation精神一致，会成为embodied AI/spatial intelligence训练的一个基础design pattern。

**Bidirectional supervision是key insight**：inverse+forward不是task多样性trick，是对同一transition施加两端约束。完整spatial representation必须同时支持解释和预测。这个原则可以推广到其他causal reasoning任务。

**GRPO的作用被underestimated**：SFT后perspective taking反而降了，GRPO救回来。说明纯SFT会过拟合到某些task pattern，RL refinement通过reward把输出层重新calibrate。这跟R1论文里SFT和RL的division of labor一致——SFT学format和knowledge，RL学reasoning和calibration。

**Action space太窄是limitation**：6个single-step action+3个multi-step preset，translation 0.1m grid，rotation 10度grid。真实embodied agent的action space是连续的。当前是toy setup，扩展到continuous action space是下一步。

**Inference efficiency是真的**：test-time coupling每个query要6+次world model forward，train-time distillation训完后单次VLM forward就行。部署成本降几个数量级。这在实际应用里很关键。

---

## Reference Links

- World2VLM paper: https://arxiv.org/abs/2505.19310 (这篇)
- MindJourney (test-time coupling baseline): https://arxiv.org/abs/2505.17079
- AlphaZero (distillation精神源头): https://arxiv.org/abs/1712.01815
- Qwen2.5-VL (base model): https://arxiv.org/abs/2502.13923
- Stable Virtual Camera (teacher 1): https://arxiv.org/abs/2505.14
- HunyuanVideo (teacher 2 base): https://arxiv.org/abs/2503.12
- SAT benchmark: https://arxiv.org/abs/2412.07755
- VSI-Bench: https://arxiv.org/abs/2412.14171
- DeepSeekMath GRPO: https://arxiv.org/abs/2402.03300
- SpatialDreamer (相关工作): https://arxiv.org/abs/2512.07733
- LoRA: https://arxiv.org/abs/2106.09685

---

## 最终一句话

World2VLM把world model从推理时的"外脑"重定位成训练时的"老师"，通过10万道bidirectional transition题+SFT→GRPO两阶段post-training，把dynamic spatial imagination蒸馏进VLM参数。推理时零外部调用，benchmark平均涨15分，远超test-time coupling的2分。核心insight是bidirectional supervision（inverse+forward）对同一transition施加两端约束，强制模型学一个同时能解释和预测的spatial representation。这跟AlphaZero把MCTS想象蒸馏进policy network的精神完全一致，大概率会成为spatial intelligence训练的基础范式。

---

# World2VLM — 把 world model 从 test-time 外脑蒸馏成 train-time 老师

## 1. Paper 的核心 thesis（一句话版）

现有 VLM 在 static visual QA 上很强，但遇到 dynamic spatial reasoning（egocentric motion 下推演 scene 演化、预测 action consequence、mental simulate 未观察视角）就拉胯。之前的两条路线都有问题：

- **Synthetic data scaling 路线**：堆 spatial VQA 数据，但 supervision signal 是 static 的，不显式 model "observation 如何在 action 下演化"
- **Test-time world-model coupling 路线**（MindJourney-style [1]）：推理时调 world model 生成 imagined views，VLM 在那些 view 上推理。计算贵，且 VLM 本体没变化

World2VLM 提出第三条：把 generative world model 当 **training-time teacher**，offline 用它生成"action-aligned view transition"，转成 structured supervision，SFT + GRPO 蒸馏进 VLM 参数。Inference 时纯 VLM，零外部 generator 调用。

类比直觉：MindJourney 像开卷考试带 calculator，World2VLM 像闭卷考试前刷一万道题把心算练满级。AlphaZero 的精神：MCTS 提供想象，最终 policy/value net 把想象内化进参数 [2]。

---

## 2. 三个核心组件

### 2.1 World-Model-Guided Transition Construction

输入三件套：anchor observation $s_t$ + 参数化 egocentric action $a_t$ + controllable world model $\mathcal{G}$。输出一条 trajectory。

公式 Eq.(3)：

$$s_{t+\Delta}^{WM} = \mathcal{G}(s_t, a_t^{(\Delta)})$$

- $s_t$: anchor RGB 帧
- $a_t^{(\Delta)}$: 累积到第 $\Delta$ 步的 camera motion prefix（translation + rotation + multi-step composition）
- $s_{t+\Delta}^{WM}$: 沿 trajectory 第 $\Delta$ 步的合成 view
- $\mathcal{G}$: view-consistent world model（实验里用了两个：SVC 和 HY-WorldPlay）

**Max-Displacement Pairing**：把 anchor 和 trajectory 中"最远的有效 endpoint"配成 source-target pair，不取近邻帧。这点很关键：0.1m 平移的两帧视觉几乎无变化，模型可以靠光流/pixel drift 蒙混过去；5m + 50° rotation 的两帧必须真正 mental simulate 才能预测。这跟 contrastive learning 里 hard negative mining 是同一思路——挑信息量最大的样本。

**Spatial Anchoring**：detector-tracker 提取 source/target view 的 bbox $B_t, B_{t+\Delta}$，序列化到 text prompt 里（normalized 到 [0, 1000] 整数坐标），不画到图上。这个设计很微妙：把 bbox 渲染到 image 会让视觉 shortcut 学习（看红框位置就行），序列化到 prompt 强制 model 用语言 + 视觉联合 reasoning。

### 2.2 Bidirectional Task Suite for Spatial Internalization

每条 transition 转成 8 个 task，分两个方向：

| Task | 内容 | Direction |
|------|------|-----------|
| A1 | Translation distance estimation | Inverse |
| A2 | Rotation angle estimation | Inverse |
| A3 | Multi-step action-sequence prediction | Inverse |
| A4 | Action-sequence verification | Forward |
| D1 | Post-action object bbox prediction | Forward |
| D2 | Post-action object visibility | Forward |
| D3 | Box-guided action-sequence inference | Inverse |
| D4 | Cross-view object consistency judgment | Forward |

**关键设计：complementarity**

- **Inverse**：给两个 view 推断背后的 motion（"为什么会变成这样"）
- **Forward**：给 anchor + action 预测 outcome（"会发生什么"）

这俩约束同一个 transition 的两端，缺一不可。只学 inverse 的 model 会变成 pattern matcher（看 view diff 猜 action）；只学 forward 的 model 会变成 simulator 但不会"反演"。完整的 spatial representation 必须 both 解释观察 AND 预测后果——跟 causal model 的 forward/inverse query 完全对称 [3]。

### 2.3 Two-Stage Post-Training

**Stage I: SFT** — 在 100K 样本上 supervised distillation

公式 Eq.(4):

$$\mathcal{L}_{SFT} = \lambda_{inv}\mathcal{L}_{inv} + \lambda_{fwd}\mathcal{L}_{fwd}$$

其中：

$$\mathcal{L}_{inv} = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{inv}} \log P_\theta(y|x)$$

$$\mathcal{L}_{fwd} = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{fwd}} \log P_\theta(y|x)$$

- $\mathcal{D}_{inv}, \mathcal{D}_{fwd}$: inverse/forward 训练集
- $\lambda_{inv}, \lambda_{fwd}$: 平衡权重
- 两个 term 形式都是 NLL，但语义完全不同——一个推 motion，一个推 consequence

**Stage II: Task-Aware GRPO** — 在 1K balanced 子集上 RL refinement

公式 Eq.(7):

$$\max_\theta \mathbb{E}_{\hat{y} \sim \pi_\theta(\cdot|x)} \left[ A(\hat{y}) \log \pi_\theta(\hat{y}|x) \right] - \beta \text{KL}(\pi_\theta \| \pi_{ref})$$

- $A(\hat{y})$: group-relative advantage（同一 prompt 采 4 个 rollout，reward 减 group mean，无 critic baseline，这是 GRPO 的核心 trick，源自 DeepSeekMath [4]）
- $\pi_{ref}$: SFT 后的 reference policy
- $\beta = 0.01$: KL 系数，防 policy 漂太远

Reward 设计（Eq.(6)）：

$$r(\hat{y}, y^*) = \alpha_{fmt}r_{fmt} + \alpha_{sem}r_{sem} + \alpha_{num}r_{num} + \alpha_{geo}r_{geo} + \alpha_{ord}r_{ord}$$

- $r_{fmt}$: 格式合法性
- $r_{sem}$: 答案语义正确性  
- $r_{num}$: 数值精度（distance/angle）
- $r_{geo}$: bbox 几何质量
- $r_{ord}$: sequence 顺序一致性

不同 task 启用不同项。比如 A1/A2 启用 $\{fmt, sem, num\}$，D1 启用 $\{fmt, geo\}$ 加 validity 子项。

Numeric precision 用 piecewise-linear 函数（Eq.(8)）：

$$s_{num}(\hat{v}, v^*; \tau) = \begin{cases} 1, & |\hat{v}-v^*| \le \tau_{low} \\ 0, & |\hat{v}-v^*| \ge \tau_{high} \\ 1 - \frac{|\hat{v}-v^*| - \tau_{low}}{\tau_{high}-\tau_{low}}, & \text{otherwise} \end{cases}$$

- $\hat{v}$: 预测值
- $v^*$: ground truth
- $\tau_{low}, \tau_{high}$: 容忍区间
- Translation: $(\tau_{low}, \tau_{high}) = (0.5m, 5m)$
- Rotation: $(\tau_{low}, \tau_{high}) = (5°, 90°)$
- 小区间满分，大区间零分，中间线性插值——很 clean 的容忍带设计

GRPO 不重新教 spatial knowledge（这部分 SFT 已经做完），只 sharpen 输出层的表达：numeric calibration、format consistency、discrete decision boundary。

---

## 3. 实验数据深度解读

### 3.1 主表 Table 2

| Method | SAT-Real | SAT-Synth | VSI-Bench | MindCube | Avg |
|--------|----------|-----------|-----------|----------|-----|
| Qwen2.5-VL-7B (base) | 44.67 | 39.60 | 33.00 | 29.26 | 36.63 |
| + WM test-time (MindJourney-style) | 31.33 | 51.75 | 37.68 | 33.85 | 38.65 |
| World2VLM-SFT (SVC) | 64.00 | 50.00 | 39.84 | 33.14 | 46.75 |
| **World2VLM-GRPO (SVC)** | **72.67** | 59.20 | **41.55** | 34.86 | 52.07 |
| World2VLM-SFT (HY-WorldPlay) | 68.66 | 57.20 | 38.63 | 36.57 | 50.27 |
| **World2VLM-GRPO (HY-WorldPlay)** | 69.33 | **65.20** | 39.07 | **36.85** | **52.61** |

几个直觉性观察：

1. **Test-time coupling 在 SAT-Real 上 -13.34**：这是个 anti-pattern 信号。candidate trajectory search 在真实动态任务里反而干扰 VLM——可能想象出的 view 跟 query 的 view 不对齐，反而误导评分。

2. **Train-time distillation 全局提升 +15.44 (SVC) / +15.98 (HY)**：远超 test-time coupling 的 +2.02，证明内化 >> 外挂。

3. **Teacher 互补**：
   - SVC 强在 SAT-Real (+28.00) 和 VSI-Bench (+8.55) — 这些 benchmark 直接依赖 camera-conditioned viewpoint 几何
   - HY-WorldPlay 强在 SAT-Synth (+25.60) 和 MindCube (+7.59) — 这些依赖 mental simulation 和 temporal continuity
   - 不同 teacher 把不同 prior 注入 student

### 3.2 SAT-Real 任务分解 Table 3（这个表最 informative）

| Variant | All | Ego Move | Obj Move | Goal Aim | Action Cons | Perspective |
|---------|-----|----------|----------|----------|-------------|-------------|
| Base | 44.67 | 52.17 | 21.74 | 50.00 | 45.95 | 48.48 |
| +WM | 31.33 | 35.00 | 45.00 | 16.67 | 34.29 | 34.38 |
| SFT (SVC) | 64.00 | 73.91 | 43.48 | 79.41 | 75.68 | 42.42 |
| **GRPO (SVC)** | **72.67** | **100.00** | **65.22** | 79.41 | 67.57 | 57.57 |

让我对几个数字 build intuition：

- **Ego Movement 52.17 → 100.00**：从勉强过随机到 perfect。说明 SFT + GRPO 把"camera 移动后 scene 怎么更新"完全学会了。这种 saturate 现象在 narrow task 上常见——任务足够窄 + supervision 足够密，就吃满了。
- **Object Movement 21.74 → 65.22**：base 几乎随机猜（4 选 1 baseline 25%）。说明 base VLM 对"物体是否移动"这种 cross-view 动态判断完全不会，distillation 后大幅提升但没 saturate，说明这是更难的 task。
- **Perspective 48.48 → 42.42 (SFT) → 57.57 (GRPO)**：唯一一个 SFT 阶段反而比 base 差的类别，GRPO 把它救回来。这说明 perspective taking 需要 transition supervision（SFT 教的方向）+ 输出层 calibration（GRPO）双重加持。
- **Action Cons SFT 75.68 → GRPO 67.57**：唯一一个 GRPO 后下降的。作者解释是这 task 在 SFT 已经基本吃满，GRPO 在 trade-off 上为了 sharpen 其他类别牺牲了一点。这是 multi-task RL 的经典 trade-off。

### 3.3 Ablation Table 4

**(a) Distillation Direction Ablation:**

| Variant | SAT-R | SAT-S | VSI | MindCube | Avg |
|---------|-------|-------|-----|----------|-----|
| Base | 44.67 | 39.60 | 33.00 | 29.26 | 36.63 |
| Forward only | 52.00 | 50.40 | 38.03 | 33.33 | 43.44 |
| Inverse only | 56.67 | 38.60 | 39.51 | 31.05 | 41.46 |
| Bidirectional | **64.00** | 50.00 | **39.84** | **33.14** | **46.75** |

观察：

- Forward only 在 SAT-Synth 上比 inverse only 强（50.40 vs 38.60）——SAT-Synth 是"predict consequence"型任务，forward 直接对应
- Inverse only 在 SAT-Real 上比 forward only 强（56.67 vs 52.00）——SAT-Real 多数是"recover motion"型任务，inverse 直接对应
- Bidirectional 在 SAT-Synth 上 50.00 跟 Forward only 50.40 几乎持平，但在 SAT-Real 上 64.00 远超 Inverse only 56.67
- 平均 +10.12 (Bidirectional) vs +6.81 (Forward) vs +4.83 (Inverse)，**bidirectional 不是简单的 task 多样性，而是同一 transition 的两端约束互相强化**

**(b) Training Source Composition:**

| Variant | Avg |
|---------|-----|
| Real-scene only | 42.20 (+5.57) |
| Simulated only | 42.67 (+6.04) |
| Mixed | **46.75 (+10.12)** |

Real 提供 clutter/occlusion/真实 appearance 噪声；Sim 提供 cleaner、controllable viewpoint transition。两者单独都不够，混合 +4.55 平均提升——典型的 domain complementary 现象，跟 sim2real 文献里 mixed training 思路一致 [5]。

### 3.4 Data Scaling Figure 3

Power-law fit 风格 scaling curve：
- 0 → 100K samples
- 低数据 regime 快速提升
- 高数据 saturate
- SAT-Real/SAT-Synth 提升最陡（依赖 motion-conditioned reasoning）
- VSI-Bench/MindCube 提升较平（依赖更广的能力，transition supervision 只是一部分）

这暗示 world-model-generated supervision 是 scalable 的，但每个 benchmark 都有 ceiling。

---

## 4. 八个 Build-Intuition 点

### Intuition 1: 为什么 train-time 蒸馏 > test-time 耦合

Test-time coupling 每次推理要：
1. 采样 candidate action
2. 调 world model 生成 imagined view
3. VLM 评分每个 view
4. 选 best trajectory 出答案

每个 query 要 N×M 次 world model 调用 + VLM 调用，N = search depth, M = beam width。Paper 里 baseline 用 3 步 + beam 2，所以每个 query 至少 6 次 world model forward。

Train-time 蒸馏只做一次：offline 生成 100K 例子，训完后 inference 就是单次 VLM forward。Inference cost 降几个数量级，且 spatial knowledge 内化进参数（更彻底）。

类比 AlphaGo Lee vs AlphaZero：前者 test-time 用 MCTS 大量搜索；后者通过 self-play 把 MCTS 的 search prior 蒸馏进 policy/value net，inference 几乎不搜索也能下出强手 [2]。

### Intuition 2: 为什么 inverse + forward 互补而非冗余

考虑一个 transition：camera 移动 + 转 90° 后电视从右边缘移到中央。

- **Inverse 问**："什么 action 让电视从右边到中央？"——学习"看现象推因"
- **Forward 问**："如果 camera forward + right turn，电视会从右边移到中央吗？"——学习"知因推果"

这俩 query 同一个 transition 的两端。只学 inverse 的 model 可以靠 view diff 模式匹配（看物体相对位置变化猜 motion），不真正理解。只学 forward 的 model 在 inverse 任务上要 brute-force 试每个 action。两者一起训，强制 model 学一个能 both 解释 AND 预测的 spatial representation。

这跟 causal model 的 forward/inverse 一致：完整因果模型必须支持 do-intervention 和 abduction 双向 query [3]。

### Intuition 3: 为什么 max-displacement pairing 重要

近邻帧配对（如 0.1m translation）：
- 视觉变化 ≈ 0
- 模型可以靠 pixel flow / 图像 registration 蒙混
- supervision signal 弱

远端帧配对（如 5m + 50° rotation）：
- 大量像素 disocclusion / occlusion
- 必须真正 mental simulate camera 移动后场景怎么变
- supervision signal 强

跟 contrastive learning 里 hard negative mining 同理：挑 model 答错的、信息量大的样本。

### Intuition 4: 为什么 bbox 序列化到 prompt 而不画图上

两种可能：
- **画到 image 上**：模型可能走捷径"看红框位置直接读"
- **序列化到 text**：保留视觉输入原貌，强制 model 把 bbox 当语言 token 处理

后者强制 visual + language 联合 reasoning，防止 shortcut。类似解数学题给参数但不把答案画在草稿纸上。

### Intuition 5: 两个 teacher 的互补 prior

**SVC** (Stable Virtual Camera [6]):
- Camera-conditioned diffusion novel-view synthesis
- 几何严格，viewpoint 一致性强
- 局部 artifact（拉伸边缘、重复结构）在 large motion 时多
- 适合需要精确 camera 几何的任务（SAT-Real, VSI-Bench）

**HY-WorldPlay** (基于 HunyuanVideo [7]):
- Action-conditioned video world model
- 感知真实性强，temporal prior 强
- 可能"reinterpret" fine-grained motion（让它 plausible 但不严格 match camera）
- 适合 mental simulation 和 temporal reasoning（SAT-Synth, MindCube）

Paper 的实验清楚地展示了这个互补：每个 teacher 在自己擅长的 benchmark 上更强。这暗示未来方向是 task-aware teacher selection。

### Intuition 6: GRPO 在做什么（vs SFT）

SFT 教"知识"——通过 transition supervision 让 model 学会 motion-conditioned spatial structure。

GRPO 教"表达"——通过 task-aware reward sharpen 输出层。

Table 8 的 case study 很说明：

| Benchmark | GT | SFT Output | GRPO Output |
|-----------|----|------------|--------------|| SPARBench depth | 5.4 | 10.5 (overestimate) | 5.1 (close) |
| VSI-Bench size | 42 cm | 450 cm (order-of-magnitude err) | 50 cm (calibrated) |
| SAT-Real perspective | "right" | "left" (wrong) | "right" (correct) |

SFT 已经"知道"大致答案类型，但 magnitude 和 discrete decision 都没 calibrate。GRPO 用 numeric reward + format reward 把这些 sharpen。

这跟 AlphaZero 类似：MCTS 提供想象，policy net 通过 RL 把想象压缩成直觉。World2VLM 里 world model 是想象来源，SFT + GRPO 是压缩机制。

### Intuition 7: Teacher error propagation（paper Appendix G 的关键 caveat）

World2VLM 的根本限制：teacher 质量决定 student 上限。

Teacher errors 通过四种方式污染 supervision：

1. **Action-observation mismatch**：渲染 endpoint 不准（over-rotate, drift）→ A1/A2/A3/A4 的 motion 标签视觉不一致
2. **Geometric distortion**：bbox 变形、object silhouette 破碎 → D1 localization target 无意义
3. **Appearance hallucination/disappearance**：object 被错误生成/移除 → D2/D4 visibility 和 identity 标签错
4. **Large-motion instability**：max-displacement pairing 把样本放在 teacher 错误最严重的 trajectory 部分

缓解策略（目前实现）：
- 多 teacher ensemble（SVC + HY）
- Object-level filter（confidence ≥ 0.3, area ratio [0.01, 0.6], track consistency）
- Prompt-level validation
- Mixed-source training（real + sim）
- GRPO format regularization 防止 verbose 错误答案

未来方向：confidence-weighted distillation, task-aware teacher selection, multi-teacher agreement filtering。

### Intuition 8: 跟 AlphaZero / policy distillation 路径的精神联系

AlphaGo Lee (2016):
- Policy net + Value net + MCTS test-time search
- Inference：大量 MCTS rollouts

AlphaZero (2017):
- 纯 RL self-play + MCTS 蒸馏进 policy/value
- Inference：几乎不搜索，policy net 直接给强招

World2VLM 类似 AlphaZero 路径：
- World model = MCTS（提供想象）
- VLM = policy/value net
- SFT + GRPO = 蒸馏 + RL sharpening
- Inference：纯 VLM forward，无 world model 调用

区别：AlphaZero 自蒸馏（self-play），World2VLM 用 external teacher（world model）。未来可以想象 student 反过来训 teacher 的 iterative self-improvement loop。

---

## 5. 局限性 & 我的 take

**Paper 自己列的 limitations（Appendix G）**：
- Teacher 质量瓶颈
- 无 explicit teacher reliability modeling
- 当前 hard filter 而非 soft weighting
- 未来：confidence-weighted distillation, task-aware teacher selection

**我额外想到的几个点**：

1. **Action space 太窄**：只 6 single-step + 3 multi-step preset，translation 0.1m grid, rotation 10° grid。真实世界 action space 是连续的。当前是 toy discretization。
2. **只 LoRA rank 256**：可能限制了能学到的 spatial representation 容量。Full fine-tune 也许更彻底。
3. **GRPO 只 1K 样本 + 1 epoch**：可能 underfit。更长的 RL 阶段或许能再 push。
4. **Benchmark 都偏 indoor scene**：outdoor / large-scale scene generalization 没验证。
5. **没比较 RLHF/DPO 等其他 post-training 方法**：GRPO 是选的，没说为什么 not DPO。
6. **Teacher inference cost 没量化**：100K 数据生成花了多少 GPU-hour，paper 没明说。

**Take-home**：这篇 paper 的 elegance 在于把"想象"从 test-time 移到 train-time，把"用工具"转化为"成为工具"。这是经典 distillation idea 在 world model 和 spatial reasoning 上的应用，跟 AlphaZero 路径精神一致。最大贡献是 methodological framing——把 world model 从 inference-time tool 重新定位成 training-time teacher。后续可以扩展到更多模态（audio, tactile）、更长 horizon 的 planning、continuous action space。

---

## 6. Web Reference Links

- [1] MindJourney (Yang et al. 2025): https://arxiv.org/abs/2505.17079
- [2] AlphaZero (Silver et al. 2018): https://arxiv.org/abs/1712.01815
- [3] Causal reasoning (Pearl): https://www.cambridge.org/core/books/causality/
- [4] DeepSeekMath GRPO: https://arxiv.org/abs/2402.03300
- [5] Sim2Real mixed training: https://arxiv.org/abs/1910.11083
- [6] Stable Virtual Camera: https://arxiv.org/abs/2505.14 (ICCV 2025)
- [7] HunyuanVideo: https://arxiv.org/abs/2503.12
- [8] Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- [9] SpatialVLM: https://arxiv.org/abs/2401.12168
- [10] VSI-Bench: https://arxiv.org/abs/2412.14171
- [11] MindCube: https://arxiv.org/abs/2505.19310
- [12] SAT benchmark: https://arxiv.org/abs/2412.07755
- [13] SpatialDreamer: https://arxiv.org/abs/2512.07733
- [14] LoRA: https://arxiv.org/abs/2106.09685
- [15] Think3D: https://arxiv.org/abs/2505.23 (近作)

---

## 7. 一句话总结

World2VLM 把 world model 从"inference-time calculator"重定位成"training-time teacher"，通过 bidirectional (inverse + forward) transition supervision + SFT→GRPO 两阶段 post-training，把 dynamic spatial imagination 蒸馏进 VLM 参数，实现内化即效率——inference 零外部调用，benchmark 提升 +15.4 平均，远超 test-time coupling 的 +2.0。这个 framing 值得记住，可能成为后续 embodied AI / spatial intelligence 训练范式的基础设计。
