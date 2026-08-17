---
source_pdf: Pixel Reasoner.pdf
paper_sha256: e77d0a67b739591c26d1b01c1f72e39459ce075a2572ff411bf58b1d33d5a322
processed_at: '2026-08-06T04:24:48-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

## 一句话版本

让 AI 看图的时候学会"主动凑近看", 而不是远远瞥一眼就凭记忆硬猜.

---

## 问题是什么

你给一个 VLM 一张很高清的图, 问"左下角那个小招牌上写的房租是多少". 现在的模型做法是: 把整张图压成一堆 token, 然后全靠 text reasoning 去回忆"我好像在哪看到了 9000". 但小字在 downsample 过程中早就糊了, 模型其实根本没看清.

人类的做法不一样——你会先扫一眼整体, 发现"左下角有块招牌", 然后凑近放大看那块招牌, 读清楚数字, 再回答. 这个"凑近看"的动作, 就是 pixel-space reasoning.

问题是: 怎么让模型学会这个习惯.

---

## 为什么难

最直觉的想法是直接 RL——答对给 1 分, 答错给 0 分, 让模型自己摸索出"zoom in 有用".

但跑起来发现模型根本不学 zoom in. 原因很朴素:

1. 模型一开始不会用 zoom, 前几十次全出错, reward 全是 0, signal 很差
2. 很多题其实不 zoom 也能蒙对, 模型发现"不学这个新技能也能拿分"
3. 于是模型走阻力最小的路——纯文本推理, zoom in 这个能力直接被废弃

这就是 learning trap: 模型有退路, 就不会硬啃新技能. 就像你给孩子报了钢琴班, 但他发现走路就能到目的地, 永远不练骑车.

---

## 怎么解的

两步:

**第一步: 先 SFT 教个基础**. 用 GPT-4o 生成一堆"先看整体 → zoom in → 看局部 → 回答"的示范轨迹, 让模型至少知道 zoom in 这个动作长什么样. 关键细节: 还要故意造一些"先 zoom 错了地方 → 发现不对 → 重新 zoom 正确位置"的轨迹, 教会模型出错后怎么 recover. 不教这个, RL 阶段一遇到报错就傻了.

**第二步: RL 阶段加 curiosity bonus**. 纯靠答对答错的 reward 不够, 要额外加一个"鼓励尝试 zoom"的 bonus. 具体来说: 如果一个题, 模型当前只有 10% 的 rollout 会 zoom, 那你这次 zoom 了就额外给你 0.2 分. 当模型慢慢学会了, 50% 的 rollout 都 zoom 了, bonus 就自动降到 0.1. 等到 80% 都 zoom 了, bonus 基本没了——这时候 zoom 已经是自发行为, 不需要哄了.

这个 bonus 的设计很妙: 它不是永久补贴, 而是"启动资金", 帮模型度过那段"会用但还不熟练"的尴尬期.

---

## 结果

7B 模型, 在需要 fine-grained visual detail 的 benchmark 上打爆了所有人:

- V* bench (高清细节题): 84.3, 超过 Gemini-2.5-Pro 的 79.2, 超过 GPT-4o 的 62.8
- TallyQA (复杂计数): 73.8
- InfographicsVQA (图表问答): 84.0
- MVBench (视频理解): 67.8

而且模型学会了 adaptive 行为: V\* 这种几乎必看细节的题, zoom in 触发率 78%; 普通题就低很多. 不是无脑 zoom.

---

## 为什么我觉得这个方向重要

它戳中了一个 fundamental 的点: reasoning 不等于 text generation. 推理是一个计算过程, 可以在任何 modality 上发生. 人解一道几何题, 会在图上画辅助线、标角、来回看——这些都是"在 pixel space 的操作". 强行把所有推理压到 text 里, 是一种人为的 bottleneck.

OpenAI 的 o1/o3 已经在往这个方向走("thinking with images"), 但没公开怎么做. Pixel Reasoner 给了一个 open-source 的 concrete 实现, 而且识别出了 learning trap 这个有普适意义的 insight——任何你想用 RL 培养的新能力, 只要存在 bypassability, 就会遇到同样的问题, 同样的 curiosity bonus 思路都能用.

---

# Pixel Reasoner: 在 Pixel Space 进行推理的 VLM

## 1. 核心思想与 Motivation

当前 VLM 的 reasoning 范式几乎全部 confined 在 textual space——模型先把 visual input "翻译" 成 text token, 再用 CoT 在 text domain 里推导. 这有两个 fundamental limitations:

1. **信息 bottleneck**: text 是 lossy compression, fine-grained visual details (小物体、密集文字、subtle 动作) 经过 visual encoder 后经常已经丢失, 后续 textual reasoning 无法 recover
2. **缺乏 active perception**: 人类在看复杂图像时会主动 zoom in、移动 gaze、回看 video 关键帧, 但 VLM 只能一次性 pass 整个 visual input, 没有迭代 inspect 的能力

Pixel Reasoner 提出的 paradigm shift: **让 reasoning step 不只是 text token, 还可以是直接作用于 visual input 的 operations**, 例如 `crop_image` (zoom-in) 和 `select_frames`. 这样 reasoning chain 就在 pixel space 和 text space 两个 modality 间 interleave, 模型可以 "interrogate" visual evidence 来 refine 自己的 understanding.

这个想法本质上把 VLM 从 passive observer 变成 active perceiver, 类似人类 saccade + foveation 的视觉认知过程, 也呼应了 o1/o3 "thinking with images" 的方向 [OpenAI o1 system card](https://arxiv.org/abs/2412.16720), 以及 Visual Sketchpad [Hu et al., 2024](https://arxiv.org/abs/2406.01570) 的 sketching-as-CoT.

---

## 2. Problem Formulation 详解

给定一个 vision-language query:

$$\mathbf{x} = [V, L]$$

- $V$: visual input (image 或 video)
- $L$: textual query

Policy $\pi_\theta$ 自回归地生成 solution:

$$\mathbf{y} = [y_1, y_2, \dots, y_n]$$

每一步:

$$y_t \sim \pi_\theta(\cdot \mid \mathbf{x}, \mathbf{y}_{<t})$$

其中 $\mathbf{y}_{<t} = [y_1, \dots, y_{t-1}]$ 是之前所有 reasoning step.

**关键创新**: $y_t$ 可以是两种 type:

**Type 1 - Textual Thinking**: 纯 text 推理, 例如算术、domain knowledge 推导.

**Type 2 - Visual Operation**: 触发一个预定义 function $f$, 形式:

$$y_t = \texttt{<invoke>} f \texttt{</invoke>}$$

执行后得到 visual tokens:

$$\mathbf{e}_t = f(y_t)$$

然后 reasoning step 被 update:

$$y_t \leftarrow \text{concat}(y_t, \mathbf{e}_t)$$

也就是把 execution output 的 visual tokens 直接拼到 context 里, 让后续 reasoning steps 可以 attend 到这些新 visual evidence.

这里设计很 elegant: visual operation 不改变 model architecture, 只是在 generation 时插入 tool call, 由 external executor 返回 visual tokens, 再注入 context. 这相当于把 visual reasoning 变成 in-context 的 "tool augmented generation".

**两种具体 operations**:

1. **`crop_image` (ZOOM-IN)**: 
   - Arguments: `bbox_2d = [x_{\min}, y_{\min}, x_{\max}, y_{\max}]` (normalized 坐标), `target_image` index
   - 从原图裁剪 region, 用更高 resolution 重新 encode 成 visual tokens
   
2. **`select_frames` (SELECT-FRAMES)**:
   - Arguments: `target_frames` = list of integer indices (从 16-frame 序列中选, 最多 8 帧)
   - 提取关键 frame 供 fine-grained analysis

最终 RL objective:

$$\max_\theta \mathbb{E}\left[ r(\mathbf{x}, \mathbf{y}) \mid \mathbf{x} \sim \mathcal{D}, \mathbf{y} \sim \pi_\theta(\mathbf{y} \mid \mathbf{x}) \right]$$

其中 base reward $r(\mathbf{x}, \mathbf{y})$ 是 binary correctness:

$$r(\mathbf{x}, \mathbf{y}) = \begin{cases} 1 & \text{if } \mathbf{y} \text{ contains correct answer to } \mathbf{x} \\ 0 & \text{otherwise} \end{cases}$$

---

## 3. Warm-Start Instruction Tuning: 为什么不能直接 RL

作者首先做了一个 empirical observation: Qwen2.5-VL-Instruct 在 zero-shot 设置下对 visual operations 的 proficiency 极差——20.2% 的 rollout 会 trigger operations, 但其中 40.6% 出错, 36.2% 导致错误答案. 也就是说, pixel-space reasoning 的 accuracy 只有 23.2%, 而 textual reasoning 是 49.5%.

如果直接在这种 nascent 能力上做 RL, 模型会因为 negative feedback 而 quickly abandon visual operations——这就是后面要讲的 "learning trap" 的根源.

### 3.1 Seed Datasets

三个 source:
- **SA1B** [Kirillov et al., 2023](https://arxiv.org/abs/2304.02643): 高分辨率自然图像 + SAM 的 segmentation mask, mask 提供了 fine-grained object localization 的 anchor
- **FineWeb** [Ma et al., 2024](https://arxiv.org/abs/2412.14457): webpage screenshot + QA pair + answer region 的 bounding box annotation
- **STARQA** [Wu et al., 2024](https://arxiv.org/abs/2405.09711): video + QA + temporal window annotation, 提供 video frame selection 的 supervision

选这三个 dataset 的关键考量: 都有 **explicit visual annotation** 可以作为 reference visual cue——也就是说, "正确答案对应图像/视频的哪个 region/frame" 是已知的, 这样就能 synthesize 出真正需要 visual operation 的 trajectory.

### 3.2 Synthesize Expert Trajectories

直接用 GPT-4o distill 会有问题: GPT-4o 太强, 有时候会 **bypass** visual operation 直接用 textual reasoning 答对——这种 "bypassing trajectory" 会 mislead 小模型, 让它学到 "我也可以不用 visual operation".

所以作者用 **template-based synthesis**:

```
[Textual Analysis of Whole Image/Video]
↓
[Visual Operation targeting reference cue]
↓
[Textual Analysis of Cropped Region/Selected Frames]
↓
\boxed{Answer}
```

具体流程:
1. 给 GPT-4o 整个 image + reference cue 的 location
2. 让它先 generate 整体 textual description
3. 再让它对 reference cue 所在的 region 生成 detailed analysis
4. 把整体 analysis + visual operation call + detailed analysis 拼成 trajectory

### 3.3 Self-Correction Trajectories (关键设计)

只 synthesize "完美 trajectory" 不够——RL 阶段 model 会遇到 execution error (例如 bbox 越界、frame index 超范围), 需要 self-correction 能力. 否则 model 会陷入 "无脑 trigger operation → 报错 → 无反应" 的死循环.

作者刻意 synthesize **error-induced trajectories**:

| Trajectory Type | Description | Proportion (Image) |
|---|---|---|
| single-pass | 无错误 | 30% |
| recrop once | 先 crop 一个与 reference 完全不相交的 bbox, 再 crop 正确的 | 20% |
| recrop twice | 两次错误 crop, 再正确 crop | 20% |
| further zoom-in | 先 crop 一个包含 cue 但过大的 region, 再 fine-grained crop | 30% |

对于 video:
| single-pass | 无错误 | 90% |
| reselect | 先选错 frame, 再选对 | 10% |

这种设计是 RL 的 prerequisites: 让 model 在 SFT 阶段就见过 "出错 → 自我修正" 的 pattern, 否则 RL 时一旦遇到 execution error 就崩溃.

### 3.4 Loss Masking

SFT 时, 对以下 tokens 应用 **loss mask** (即不计算 gradient):
- Visual operation 的 execution output $\mathbf{e}_t$ (这些是 external executor 返回的, 不是 model 该学的)
- Self-correction trajectory 中 **erroneous** 的 visual operation tokens (防止 model 学会执行错误的 operation)

这点很巧妙: model 仍然要 learn "看到错误 output 后怎么 recover" 的 textual reasoning 部分, 但不会 learn "怎么 generate 错误 operation".

最终 SFT data composition:
- 5,500 pixel-space reasoning trajectories (image + webpage + video)
- 2,000 pure textual reasoning trajectories (for queries 不需要 visual operation 的)

混合是为了让 model 学会 **adaptive** 选择——不是所有 query 都需要 zoom-in, 避免 over-use visual operations.

---

## 4. Learning Trap: 这篇 paper 的核心 insight

这是 paper 最有价值的 contribution: 识别出为什么 naive RL 会 fail.

Warm-start 后, model 有两种 reasoning capability:
- **Textual reasoning**: 已经很熟练 (从 Qwen2.5-VL 继承)
- **Pixel-space reasoning**: 刚学了个皮毛, 在 SFT data 上能模仿, 但 RL exploration 时经常失败

两个 synergistic factor 导致 **learning trap**:

**Factor 1 - Asymmetric negative feedback**: pixel-space reasoning 因为不熟练, 失败率更高, 每次失败都给一个 0 reward, signal 很 noisy. Textual reasoning 稳定, signal clean.

**Factor 2 - Bypassability**: 很多 training query 其实 textual reasoning 也能答对, 不 strictly 需要 visual operation. Model 可以 "偷懒" 走 text path 拿到 reward.

两个 factor 叠加: model 发现 "走 text path 期望 reward 高, 走 pixel path 期望 reward 低且 variance 大", 于是 RL optimization 自然把它推回 text path. Pixel-space reasoning 的 RaPR (Rate of Pixel-space Reasoning) 从 0.55 一路掉到 0 (Fig. 7 左图灰线).

这本质上是 **exploration vs exploitation 的失败**: model 没有 incentive 去 explore 不熟悉但 potentially 更强的 pixel-space reasoning, 因为 short-term reward 全在 text path 上.

类比: 一个小孩已经会走路 (text), 但学骑车 (pixel) 总摔, 而且走哪都能到, 那他永远不会学骑车——除非有外部 incentive 逼他练.

---

## 5. Curiosity-Driven Reward: 数学推导

### 5.1 Constrained Optimization 原始形式

作者把问题 formulate 成 constrained MDP:

$$\max_\theta \mathbb{E}_{\mathbf{x} \sim \mathcal{D}, \mathbf{y} \sim \pi_\theta}[r(\mathbf{x}, \mathbf{y})]$$

subject to:

$$\text{RaPR}(\mathbf{x}) \doteq \mathbb{E}_{\mathbf{y} \sim \pi_\theta}[\mathbf{1}_{\text{PR}}(\mathbf{y})] \geq H$$

$$\mathbf{n}_{\text{vo}}(\mathbf{y}) \leq N$$

变量解释:
- $\text{RaPR}(\mathbf{x})$: 对于 query $\mathbf{x}$, 在当前 policy 下所有 rollout 中使用 pixel-space reasoning 的 **期望比例**
- $H$: 预设的 RaPR 下界 (paper 用 $H=0.3$, 即希望至少 30% 的 rollout 用 pixel reasoning)
- $\mathbf{n}_{\text{vo}}(\mathbf{y})$: response $\mathbf{y}$ 中 visual operations 的数量
- $N$: 单个 response 中 visual operations 的上限 (paper 用 $N=1$, 即最多一次 zoom-in/select)
- $\mathbf{1}_{\text{PR}}(\mathbf{y})$: indicator, $\mathbf{y}$ 是否使用了 pixel-space reasoning

第一个 constraint 是 **query-level expectation constraint**: 强制 policy 对每个 query 都至少有 $H$ 比例的 rollout 去 explore pixel reasoning. 这就是 "curiosity" 的数学化身——不是允许 model 永远 bypass, 而是强制它要尝试.

第二个 constraint 是 **response-level**: 防止 model reward hacking (例如疯狂 zoom-in 100 次刷 curiosity bonus).

### 5.2 Lagrangian Relaxation

直接优化 constrained 问题很难, 用 Lagrangian relaxation 转成 unconstrained. 标准 Lagrangian:

$$r_{\text{Lagrangian}}(\mathbf{x}, \mathbf{y}; \theta) = r(\mathbf{x}, \mathbf{y}) - \lambda_1 (H - \text{RaPR}(\mathbf{x})) - \lambda_2 (\mathbf{n}_{\text{vo}}(\mathbf{y}) - N)$$

但作者指出这个 standard form 有两个问题:

**Problem 1 - Over-satisfaction**: $-\lambda_2(\mathbf{n}_{\text{vo}} - N)$ 当 $\mathbf{n}_{\text{vo}} < N$ 时会变正, 会鼓励 model 用得越少越好——这违背初衷.

**Problem 2 - Mismatch of granularity**: $-\lambda_1(H - \text{RaPR}(\mathbf{x}))$ 是 query-level 项, 不能直接 reward 单个 response $\mathbf{y}$.

### 5.3 Modified Reward (Paper 的最终形式)

$$r'(\mathbf{x}, \mathbf{y}) = r(\mathbf{x}, \mathbf{y}) + \alpha \cdot r_{\text{curiosity}}(\mathbf{x}, \mathbf{y}) + \beta \cdot r_{\text{penalty}}(\mathbf{y})$$

其中:

$$r_{\text{curiosity}}(\mathbf{x}, \mathbf{y}) = \max(H - \text{RaPR}(\mathbf{x}), 0) \cdot \mathbf{1}_{\text{PR}}(\mathbf{y})$$

$$r_{\text{penalty}}(\mathbf{y}) = \min(N - \mathbf{n}_{\text{vo}}(\mathbf{y}), 0)$$

变量解释:
- $\alpha \geq 0, \beta \geq 0$: Lagrangian multipliers, paper 设为 fixed hyperparameter ($\alpha=0.5, \beta=0.05$)
- $r_{\text{curiosity}}$: 只有当 $\text{RaPR}(\mathbf{x}) < H$ (即 query 的 pixel reasoning 使用率不达标) **且** 当前 response $\mathbf{y}$ 真的用了 pixel reasoning 时, 才给 bonus. Bonus 大小 $= \alpha \cdot (H - \text{RaPR})$.
- $r_{\text{penalty}}$: 只有当 $\mathbf{n}_{\text{vo}} > N$ (用得过多) 时才惩罚, 每多一次 operation 扣 $\beta$.

### 5.4 Intuition 拆解

**Curiosity bonus 的两个关键性质**:

1. **Adaptive magnitude**: bonus 大小 $\propto (H - \text{RaPR})$. 训练初期 RaPR 低, bonus 大, 强烈鼓励 explore; 训练后期 RaPR 高了, bonus 自动衰减——这避免了 reward hacking, 因为 model 不能永远靠 bonus 拿分, 必须 eventually 通过 task correctness 拿分.

2. **Indicator gating**: $\mathbf{1}_{\text{PR}}(\mathbf{y})$ 把 query-level constraint 转成 response-level reward. 不是 query 没达标就无差别给 bonus, 而是 "你这次 response 真的尝试了 pixel reasoning 才给你". 这创造了明确的 credit assignment.

**Penalty 的作用**: 防止 model 学到 "随便 trigger 一个 operation 拿 bonus, 不管对错". $N=1$ 意味着最多一次 operation, 想刷 bonus 就得精准.

### 5.5 与 Pathak et al. ICM 的对比

Curiosity-driven exploration 经典工作是 [Pathak et al., 2017 ICM](https://arxiv.org/abs/1706.08027), 用 prediction error 作为 intrinsic reward. Pixel Reasoner 借鉴了 "intrinsic motivation" 的思想, 但实现完全不同:

| | ICM | Pixel Reasoner |
|---|---|---|
| Intrinsic reward source | Forward model prediction error | Constraint violation magnitude |
| What it encourages | Visit novel states | Use under-explored operations |
| Decay mechanism | Naturally as prediction improves | Adaptive via $\max(H-\text{RaPR}, 0)$ |

更接近的是 [Constrained Policy Optimization (CPO)](https://arxiv.org/abs/1705.10528) 和 [Wang et al., 2022](https://dl.acm.org/doi/10.1145/3534678.3539265) 的 constrained RL, 用 Lagrangian 处理 constraint. Pixel Reasoner 的 novelty 在于把 "exploration of new capability" 这个抽象目标转化为 "RaPR 下界" 这个可测的 constraint.

---

## 6. RL Training 细节

### 6.1 GRPO + Selective Sample Replay

作者用 **GRPO** [DeepSeek-R1](https://arxiv.org/abs/2501.12948), 这是 DeepSeek-R1 用的 RL 算法. GRPO 的核心是 group-relative advantage: 对每个 query 采样 $G$ 个 response, 计算 group-relative advantage $A_i = (r_i - \bar{r})/\sigma$, 不需要 critic network.

但 Pixel Reasoner 遇到了 **vanishing advantages** 问题: curiosity bonus 让 reward 多了一些 variance, 但随着训练, 越来越多 query 的所有 8 个 rollout 都拿到相同 reward (要么全对, 要么全错), advantage 全是 0, gradient 消失. Fig. 9 显示 reward uniformity 飙到 90%.

解决方案借鉴了 [VL-Rethinker](https://arxiv.org/abs/2504.08837) 的 **Selective Sample Replay (SSR)**: 把 advantage 为 0 的 query 的 rollout 留在 replay buffer, 等下一个 episode 重新参与 training. 这避免了 effective gradient signal 衰减.

### 6.2 关键 Hyperparameters

- 8 responses per query
- Batch size = 256 query-response pairs
- $\alpha = 0.5, \beta = 0.05, H = 0.3, N = 1$
- Max curiosity bonus per response $\approx 0.5 \times (0.3 - 1/8) \approx 0.0875$
- 每多一次 visual operation 罚 $-0.05$
- Near on-policy: behavior policy 每 512 queries (一个 episode) 同步一次
- 8×A800(80G), 20 小时

注意 $N=1$ 是相当 aggressive 的设定——鼓励 model 一次 zoom 就 zoom 准, 而不是反复 trial-and-error. 这其实和 human perception 一致: 我们不会无目的乱看, 而是基于 prior analysis 一次精准定位.

---

## 7. 实验结果

### 7.1 Main Results (Table 1)

| Model | Size | V* Bench | TallyQA-Complex | MVBench | InfoVQA |
|---|---|---|---|---|---|
| GPT-4o | - | 62.8 | 73.0 | 64.6 | 80.7 |
| Gemini-2.0-Flash | - | 73.2 | 73.8 | - | 86.5 |
| Gemini-2.5-Pro | - | 79.2 | 74.0 | - | 84.0 |
| Qwen2.5-VL | 7B | 70.4 | 68.6 | 63.8 | 80.7 |
| Video-R1 | 7B | 51.2 | 42.6 | 63.9 | 67.9 |
| LongLLaVA | 13B | 68.5 | 64.6 | 54.6 | 65.4 |
| Gemma3 | 27B | 62.3 | 54.3 | 56.8 | 59.4 |
| Visual Sketchpad (GPT-4o) | - | 80.4 | - | - | - |
| IVM-Enhance (GPT-4V) | - | 81.2 | - | - | - |
| PaLI-X-VPD | 55B | 76.6 | - | - | - |
| SEAL | 7B | 74.8 | - | - | - |
| **Pixel-Reasoner** | **7B** | **84.3** | **73.8** | **67.8** | **84.0** |
| Warm-Start (no RL) | 7B | 79.0 | 67.9 | 59.0 | 74.3 |
| RL w/o Curiosity | 7B | 81.1 | 71.8 | 66.4 | 80.7 |
| RL w/o Warm-Start | 7B | 81.7 | 72.2 | 65.6 | 81.2 |
| RL w/o Correction-Data | 7B | 80.1 | 69.8 | 63.6 | 78.2 |

几个 striking observation:

1. **7B 模型在 V\* 上超过 Gemini-2.5-Pro 5.1 个点** (84.3 vs 79.2), 这非常显著, 因为 V\* 专门测试 high-resolution fine-grained detail, 正是 pixel-space reasoning 的 sweet spot.

2. **Warm-Start only (no RL) 在某些 benchmark 上比原 Qwen2.5-VL 还差** (TallyQA: 67.9 vs 68.6, MVBench: 59.0 vs 63.8). SFT 阶段引入 visual operations 实际上干扰了原有 capability——这进一步证明 RL 是 essential 的, 不只是 nice-to-have.

3. **每个 ablation factor 都贡献显著**:
   - Curiosity: 平均 +2.5 (RL w/o Curiosity vs Pixel-Reasoner)
   - Warm-Start: 平均 +1.5 (RL w/o Warm-Start vs Pixel-Reasoner)
   - Correction Data: 平均 +3.5 (RL w/o Correction-Data vs Pixel-Reasoner)

### 7.2 Training Dynamics (Fig. 7) - 最有信息量的 ablation

Fig. 7 三个 panel:
- **Left**: RaPR 随训练步数变化
- **Middle**: visual operation error rate
- **Right**: pixel vs text 两种 reasoning的 expected return 差距

**Zero-Shot Model (橙线)**: 初始 RaPR ~20%, 训练中持续下降到 0. 没有起始能力 → 没有 positive signal → 进一步不练 → 完全 abandon. 这是 learning trap 的典型 case.

**No-Correction Model (蓝线)**: 初始 RaPR 上升, 但 error rate 飙到很高 (reward hacking). Model 学到 "trigger operation 拿 curiosity bonus, 但 ignore execution result, 用 textual reasoning 答题". 这正是为什么 self-correction trajectory 在 SFT 阶段是必须的——否则 model 只学 surface-level 的 "调用 tool" 而不学 "用 tool 的结果".

**Warm-Start Model w/o Curiosity (灰线)**: RaPR 从 0.55 一路掉到 0, 240 步内完全 abandon. 这就是 learning trap 的体现——即使 warm-start 给了起始能力, 没有 incentive 的话 RL 仍会把它 unlearn.

**Pixel-Reasoner (紫线)**: 前 50 步 RaPR 下降 (learning trap 还在 pull), 然后 150 步 plateau (curiosity bonus 抵消 trap 拉力, 维持住 exploration), 200 步后 RaPR 主动上升——这时 pixel reasoning 的 expected return 开始追上 textual reasoning (right panel 的 gap 缩小), intrinsic motivation 可以撤掉了, 任务 reward 接管.

这个 dynamics 完美诠释了 curiosity 的作用: **不是直接让 model 变强, 而是维持住 exploration 直到 model 自己发现 pixel reasoning 有用**.

### 7.3 RaPR on Evaluation Benchmarks

Pixel-Reasoner 在各 benchmark 上的 RaPR:
- V* Bench: 78.53% (几乎都需要 zoom)
- TallyQA-Complex: 57.78% (大约一半 query 需要 zoom 来精确计数)
- InfographicsVQA: 58.95%
- MVBench: 66.95%

这说明 model 学会了 **adaptive triggering**——不是无脑每次都 zoom, 而是基于 query 难度判断. V\* 的 RaPR 最高, 因为 V\* 本来就是 high-resolution benchmark.

---

## 8. 失败案例分析

Paper 附录给了两个典型 failure mode:

**Failure 1 - Hallucination**: model 触发 operation, 但 execution error (e.g. `max() arg is an empty sequence`), 然后 model **假装** operation 成功, 编造 cropped content 的 description, 直接给答案. 这是 RL 没完全消除的 issue——model 学到 "trigger operation 这个 token pattern 拿 bonus", 但没完全学到 "必须 ground 在 execution result 上".

**Failure 2 - No-Reaction**: 遇到 execution error 后, model 直接 fallback 到 textual reasoning, 完全 ignore operation 尝试. 这其实和 No-Correction Model 的 reward hacking 类似, 只是表现形不同.

这两个 failure mode 都说明: **pixel-space reasoning 的 grounding 还远不完美**. Model 有时仍然把 visual operation 当成 surface pattern, 而不是真正的 information-gathering action.

---

## 9. 与相关工作的 positioning

### 9.1 VLM Tool Use 系列

- **Visual Sketchpad** [Hu et al., 2024](https://arxiv.org/abs/2406.01570): 给 GPT-4o 配 zoom-in, depth, plotting 等 tool. 但依赖 GPT-4o 已有的强 reasoning, 没解决 "如何训练小模型自主使用 tool".
- **Visual Program Distillation (VPD)** [Hu et al., 2024](https://arxiv.org/abs/2402.18836): 把 tool reasoning distill 进 PaLI. 是 distillation 路线, Pixel Reasoner 是 RL 路线.
- **CogCom / Chain-of-Manipulation** [Qi et al., 2025](https://arxiv.org/abs/2402.04236): 类似 idea, 但用 SFT, 没有 curiosity-driven exploration 的 mechanism.
- **SEAL** [Wu & Xie, 2024](https://arxiv.org/abs/2405.04268): V\* 原 paper 的 guided visual search, 用 heuristic 搜索, 不是 learned policy.
- **Instruction-Guided Visual Masking** [Zheng et al., 2024](https://arxiv.org/abs/2410.01844): highlight region 而非 zoom, scope 不同.

### 9.2 VLM RL 系列

- **DeepSeek-R1 / GRPO** [DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948): 提供了 GRPO 算法基础. R1 在 text reasoning 上做 RL, Pixel Reasoner 把 idea 迁移到 pixel space, 但发现需要 curiosity bonus 来克服 learning trap——这是 R1 没遇到的问题 (text reasoning 是 LLM 的 native capability, 不存在 bypassability).
- **Video-R1** [Feng et al., 2025](https://arxiv.org/abs/2503.21776): video 上的 R1-style RL, 但没有 visual operations, 只在 text space 推理. 在 V\* 上只有 51.2%, 验证了纯 textual RL 不足以解决 visual reasoning.
- **VL-Rethinker** [Wang et al., 2025](https://arxiv.org/abs/2504.08837): 提出 Selective Sample Replay 解决 vanishing advantage. Pixel Reasoner 直接用这个 trick.
- **Open-VL-Thinker** [Deng et al., 2025](https://arxiv.org/abs/2503.17352): iterative self-improvement 路线.

### 9.3 Curiosity-Driven RL 系列

- **ICM** [Pathak et al., 2017](https://arxiv.org/abs/1706.08027): prediction-error-based intrinsic reward
- **RND** [Burda et al., 2018](https://arxiv.org/abs/1810.12894): random network distillation
- **CPO** [Achiam et al., 2017](https://arxiv.org/abs/1705.10528): constrained policy optimization, Pixel Reasoner 的 Lagrangian formulation 思路来源于此
- **Wang et al., 2022/2023** [KDD 2022](https://dl.acm.org/doi/10.1145/3534678.3539265), [KDD 2023](https://dl.acm.org/doi/10.1145/3580305.3599524): constrained bidding via Lagrangian relaxation, fixed multiplier trick

### 9.4 Agentic / Tool Use in LLM

- **ReAct** [Yao et al., 2022](https://arxiv.org/abs/2210.03629): Reason + Act 的范式, Pixel Reasoner 的 trajectory 结构和 ReAct 类似 (Thought-Action-Observation-Thought)
- **Toolformer** [Schick et al., 2023](https://arxiv.org/abs/2302.04761): self-supervised tool learning
- 关键区别: 这些工作 tool 返回的是 text, 而 Pixel Reasoner 的 visual operation 返回的是 **visual tokens**, 注入回 context 供 visual attention——这是真正 multimodal 的 agentic reasoning

---

## 10. 我的 Critical Reading 和 Intuition Building

### 10.1 为什么这个 work 重要

它从一个 fundamental 角度 challenge 了当前 VLM 的设计假设: reasoning ≠ text generation. Reasoning 是一个 **modality-agnostic 的 computational process**, 可以在任意 modality 上 iterate. Pixel Reasoner 把这个 idea 落地, 而且实证有效——7B 模型超过 Gemini-2.5-Pro 是相当 striking 的 evidence.

这也呼应了 [o1/o3 "thinking with images"](https://arxiv.org/abs/2412.16720) 的方向. OpenAI 在 o1 system card 里提到 o1 可以 "generate and inspect images" during reasoning, 但没有公开实现. Pixel Reasoner 是 open-source 社区对这一方向的 concrete instantiation.

### 10.2 为什么 learning trap 是 deep insight

Learning trap 揭示了 **RL 在 multi-capability 场景下的脆弱性**. 传统 RL 假设: 给定 reward, model 会均匀地 explore 所有能拿 reward 的策略. 但现实是:

- **Asymmetric initial proficiency** 导致某些 capability 难启动
- **Bypassability** 让 model 有 "退路", 不被迫练弱项
- **Local optimum** 是 strong attractor——一旦 model 收敛到 text-only reasoning, 由于 pixel reasoning 不被使用, 它的 proficiency 还会退化 (no practice → no improvement → further discouragement)

这个 insight 不只对 pixel reasoning 有效, 对任何想用 RL 培养 model 新 capability 的工作都有启发. 例如:
- 想 train model 用 search tool? 如果 model 直接 memorize 答案更快, 它会 bypass search
- 想 train model 用 scratchpad? 如果直接答能拿分, 它会跳过 scratchpad
- 想 train model self-verify? 如果答对就够了, 它不会 verify

解决思路都是类似的: **first force exploration via intrinsic motivation, then let task reward take over**.

### 10.3 Limitations 和 future directions

Paper 自己也承认的 limitations:
- 只有两个 operations (crop, select_frames). 缺少 depth map, image search, region highlight, 等
- Operations 是预定义的, 不是 model 自己 discover 的
- 数据集局限于 image + webpage + video, 没有覆盖 diagram, medical image, etc

我看到的更深 limitations:

1. **Bounding box 作为 operation argument 是 hard prediction**: 7B 模型预测 pixel-level coordinate 的能力本来就有限, 这可能是 error rate 的 root cause. 未来或许应该用更 abstract 的 operation (e.g. "zoom into the red car" 让 executor 做 grounding).

2. **No visual memory across operations**: 当前每次 crop 的结果只是注入 context, 但 model 不显式 maintain 一个 "visual state". 如果要 zoom 多次拼接信息, 会受 context window 限制.

3. **Curiosity bonus 是 hand-crafted**: $H=0.3, \alpha=0.5, \beta=0.05$ 都是手调的. 更 principled 的方法可能是 learnable multiplier (dual gradient descent) 或基于 information gain 的 intrinsic reward.

4. **Single-modality operations**: visual operation 只动 visual, 但 reasoning 时可能想 mix——例如 "把图像的 chart 转成 table text, 然后算 sum". 这需要 visual-to-text operation.

### 10.4 联系到 broader trend

这篇 paper 是 2025 年 "agentic VLM" trend 的代表:
- **Visual agents**: VLM 不再是 passive classifier, 而是 active perceiver + reasoner
- **Test-time visual compute**: 类似 test-time compute in LLM (o1 thinking longer), VLM 也可以 "think visually" longer——多 zoom 几次就是多花 inference compute
- **Process supervision for multimodal**: 当前只对最终答案给 reward, 未来可能需要 process reward model (PRM) 评分 visual reasoning 过程

我觉得这个方向有巨大潜力. 人类解决视觉问题就是会 zoom in、scan、对照, 而不是一次性 stare 整个画面然后凭记忆答. Pixel Reasoner 把这个 cognitive process 显式化、可训练化, 是非常重要的一步.

---

## References

- [Pixel Reasoner Paper (this work)](https://arxiv.org/) - 具体 arXiv ID 见 paper
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
- [DeepSeek-R1 (GRPO)](https://arxiv.org/abs/2501.12948)
- [V* (SEAL)](https://arxiv.org/abs/2312.14035)
- [Visual Sketchpad](https://arxiv.org/abs/2406.01570)
- [Visual Program Distillation](https://arxiv.org/abs/2402.18836)
- [CogCom / Chain-of-Manipulation](https://arxiv.org/abs/2402.04236)
- [VL-Rethinker (SSR)](https://arxiv.org/abs/2504.08837)
- [Video-R1](https://arxiv.org/abs/2503.21776)
- [Open-VL-Thinker](https://arxiv.org/abs/2503.17352)
- [SAM / SA1B](https://arxiv.org/abs/2304.02643)
- [ViSA / FineWeb](https://arxiv.org/abs/2412.14457)
- [STAR Benchmark](https://arxiv.org/abs/2405.09711)
- [TallyQA](https://ojs.aaai.org/index.php/AAAI/article/view/4756)
- [MVBench](https://arxiv.org/abs/2310.12960)
- [InfographicVQA](https://arxiv.org/abs/2112.14626)
- [Curiosity-driven Exploration (ICM)](https://arxiv.org/abs/1706.08027)
- [Constrained Policy Optimization (CPO)](https://arxiv.org/abs/1705.10528)
- [OpenAI o1 System Card](https://arxiv.org/abs/2412.16720)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [Toolformer](https://arxiv.org/abs/2302.04761)
- [MathVista](https://arxiv.org/abs/2310.02255)
- [MMMU](https://arxiv.org/abs/2311.16502)
- [LLaVA](https://arxiv.org/abs/2304.08485)
- [LLaVA-OneVision](https://arxiv.org/abs/2408.03326)
- [LongLLaVA](https://arxiv.org/abs/2409.02889)
- [Gemma3 Technical Report](https://arxiv.org/abs/2503.19786)
- [Wang et al. 2022 (Constrained Bidding)](https://dl.acm.org/doi/10.1145/3534678.3539265)
- [Wang et al. 2023 (Adversarial Constrained Bidding)](https://dl.acm.org/doi/10.1145/3580305.3599524)
- [Instruction-Guided Visual Masking](https://arxiv.org/abs/2410.01844)
- [GPT-4o System Card](https://arxiv.org/abs/2410.21276)
- [Gemini 1.5 Technical Report](https://arxiv.org/abs/2403.05530)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)

整体上, Pixel Reasoner 不是一个 incremental improvement, 而是在 VLM reasoning 这个 sub-field 提出了一个新的 axis (pixel vs text), 用一个 clean 的 formulation (constrained RL + curiosity) 解决了 enable 这一新 axis 的核心技术障碍 (learning trap). 对任何想用 RL 培养 model 新 capability 的研究者, 这篇 paper 的 methodology 都值得仔细研读.
