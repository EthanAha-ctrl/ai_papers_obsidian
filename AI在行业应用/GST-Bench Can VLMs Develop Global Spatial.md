---
source_pdf: GST-Bench Can VLMs Develop Global Spatial.pdf
paper_sha256: f4c42b3bbdbbd627f839796e8ceba86b853a7b711affc478a7f954dc81b21a38
processed_at: '2026-08-19T10:09:51-07:00'
target_folder: AI在行业应用
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 GST-Bench

Andrej，咱们这么聊。这篇 paper 你可以把它想成一句话：**"你给 VLM 看一段在屋里乱走的 egocentric 视频，然后给它一张它从没见过的视角的照片，问它'你现在在哪''之前那个咖啡杯在哪''这屋长啥样'。结果发现所有 VLM 全扑街。"**

就这么个事儿。我下面用大白话把关键点拆开。

---

## 1. 为什么要搞这么个 benchmark

以前那些 spatial benchmark，说白了都是这个套路："看一张图，告诉我 A 在 B 左边还是右边"。这种任务 VLM 用 appearance cue 就能糊弄过去，根本没真正在做 spatial reasoning。

video 类 benchmark 也有，但有个老问题 —— **有些问题一帧就能答，你分不清 model 到底是跨帧推理答对的，还是恰好那一帧就有答案**。比如 VSI-Bench 问 object size，一张图就够；OST-Bench 问 existence，一帧就够。这把 benchmark 的信噪比稀释得很厉害。

GST-Bench 干的第一件事，就是从 **construction 上** 把所有 single-frame shortcut 堵死。具体怎么做？**凡是问 object localization 的，target object 在 query view 里必须不可见**。这条 constraint 是整个 benchmark 的灵魂。

Project page: <https://qwerirwq.github.io/GST-Bench/>

---

## 2. 它问的三个核心问题

embodied agent 在屋里走完一圈，必须能回答三件事：

**Where am I?** （self localization）  
给一段 exploration video、一张 medium 抽象度的 top-down 图、一张从没在 video 里出现过的 query view，让 model 说出这个 query view 的 camera position / orientation 在 top-down 上哪儿。

**Where is the target?** （object localization）  
target 在 query view 里看不见，但 model 得根据 video 里曾经看到过 target 的位置，推断 "从我现在这个角度，target 在我哪个方向、多远、在 top-down 上哪个位置"。target 可以用 category name 指定（semantic），也可以用 object-annotated video 上的红 bbox 指定（visual）。

**What does the scene look like?** （scene structure）  
给四张 top-down 候选图，让 model 选哪张是它刚探索过的那个 scene。或者给一段 short trajectory clip，让 model 在 top-down 上选对应的 trajectory。

top-down 还分三档：easy 是 photo-realistic 鸟瞰图（有 texture），medium 是 occupancy map（只剩 footprint），hard 是 bare floor plan（只剩 walls）。这个递进设计真的聪明 —— 它让你能看出 model 到底是靠 appearance matching 还是真的理解了结构。

---

## 3. 数据是怎么造的

用 OmniGibson + BEHAVIOR-1K simulation。simulation 的好处就一个字：**可控**。

- 可以 sample off-trajectory novel view，天然 anti-shortcut
- camera pose、object pose、distance、angle 这些 ground truth 全从 geometry 直接算出来，**answer 不用人标，只用人 verify**
- 多 trajectory per scene，perturbed viewpoints，防止死记硬背

人 verify 主要 check 两件事：
1. target 在 video 里能不能真的被认出来（否则这个 sample 无效）
2. off-trajectory query view 和 exploration video 有没有足够 overlap 让人能 localize（否则这个 sample 不可答）

最终 2,762 个 human-verified question，累积 6,790 分钟 video。

OmniGibson: <https://behavior.stanford.edu/omnigibson>  
BEHAVIOR-1K: <https://behavior.stanford.edu/>

---

## 4. Metric 这块得说清楚，否则你看不懂 Table 1

四种 question format，每种 metric 不同。我挨个拆。

### Distance: Mean Relative Accuracy (MRA)

$$\mathrm{MRA} = \frac{1}{|\mathcal{C}_d|} \sum_{\theta \in \mathcal{C}_d} \mathbb{1}\!\left( \frac{|\hat{y} - y|}{y} < 1 - \theta \right)$$

人话：算相对误差 $|\hat{y}-y|/y$，看它在 10 个不同严格度 threshold 下是不是都达标，取平均。  
- $\hat{y}$: model 预测距离  
- $y$: ground truth 距离  
- $\mathcal{C}_d = \{0.50, 0.55, \dots, 0.95\}$: 10 个 threshold，从宽松到严格  
- $\theta = 0.95$ 时要求相对误差 < 5%，非常严  
- $\theta = 0.50$ 时要求相对误差 < 50%，很松  

用相对误差是为了 scale-invariant —— 1m 的距离误差 0.5m 和 10m 的距离误差 5m，都算 50% 错。

### Angle: circular distance + multi-threshold

$$e_a = \min\!\left( |\hat{\alpha} - \alpha|, \ 360^\circ - |\hat{\alpha} - \alpha| \right)$$

人话：角度是环状的，350° 和 10° 其实只差 20°，不能直接减。所以取较短那个弧。  
- $\hat{\alpha}$: 预测角度  
- $\alpha$: ground truth 角度  
- $e_a$: 真正的角度误差  

然后：

$$\mathrm{Acc}_{\mathrm{angle}} = \frac{1}{|\mathcal{C}_a|} \sum_{\tau \in \mathcal{C}_a} \mathbb{1}(e_a < \tau)$$

- $\mathcal{C}_a = \{15^\circ, 30^\circ, 45^\circ\}$: 三档 tolerance  
- $45^\circ$ 大致是 8 个方位之一 (360/8=45)  
- $15^\circ$ 接近 "我能精确说出它在我前面偏左一点"  

### Point: Euclidean distance + multi-threshold

$$e_p = \|\hat{\mathbf{p}} - \mathbf{p}\|_2, \quad \mathrm{Acc}_{\mathrm{point}} = \frac{1}{|\mathcal{C}_p|} \sum_{\tau \in \mathcal{C}_p} \mathbb{1}(e_p < \tau)$$

- $\hat{\mathbf{p}}, \mathbf{p} \in \mathbb{R}^2$: top-down 上 predicted / ground-truth pixel 坐标  
- $\|\cdot\|_2$: 欧氏距离  
- $\mathcal{C}_p = \{100, 150, 200, 250, 300\}$ pixels  

注意单位是 pixel 不是 meter，因为 top-down 是渲染的 2D 图。100 像素大概对应几米，300 像素大概是"大致在哪个房间"的尺度。

### Multiple choice: 直接 accuracy

top-down selection 和 trajectory selection 都是四选一，直接算 accuracy。

### Overall

$$\mathrm{Score} = \frac{1}{12}\sum_{i=1}^{12} s_i$$

12 个 subtask 算术平均，不加权。

---

## 5. 主结果：人跟 model 差多少

Table 1 里的几个关键 cell：

| 指标 | 最强 model | Human | Gap |
|---|---|---|---|
| Avg | Gemini-3-Pro 42.68 | 79.08 | -36.40 |
| Ori (orientation) | Gemini-3-Pro 21.52 | 85.00 | -63.48 |
| GP (global position) | Gemini-3-Pro 42.23 | 93.00 | -50.77 |
| EDist_v (egocentric distance) | Gemini-2.5-Pro 42.00 | 41.50 | +0.50 |
| TDS_E (top-down selection easy) | Gemini-3-Pro 98.15 | 100.00 | -1.85 |

**几个值得咂摸的点：**

**(a) Orientation 最惨。** Gemini-3-Pro 21.52 vs human 85.00，差 63 点。这个数字说明 VLM 连"我现在面朝哪"都搞不准。pose 错了，后面所有 spatial reasoning 都跟着错。这指向 architectural defect —— 当前 VLM 没有 explicit pose representation。

**(b) EDist 看起来 model 追平 human，其实是假象。** Human 本身也才 41.50 MRA，因为从单目视频恢复绝对 metric distance 是 ill-posed 的，人也做不好。这条线不是 "model 已经 solved"，是 "这条 task ceiling 被感知极限压住了"。

**(c) Top-Down Selection (easy) 几乎所有 model 都 80+，但去掉 appearance 就崩。** Easy → Medium → Hard，分数一路下滑。Easy 靠 texture/color template matching 能糊弄；Medium 去 appearance 只留 footprint，立刻掉；Hard 只剩 walls，掉得更狠。这条曲线直接证明 VLM 在 top-down 匹配上靠的是 appearance-level matching，不是真正的结构理解。

**(d) Embodied-tuned models 反而更差。** RoboBrain2.5-8B (24.61) < Qwen3-VL-8B (25.89)；Robix-32B (29.26) ≈ Qwen3-VL-32B (30.43)。这说明现有 embodied post-training 专攻 local affordance 和 local spatial relation，**完全没碰 long-horizon spatial memory 这条线**。RoboBrain / Robix / Cosmos-Reason 这帮 model 训练时 reward 的是 "next action grounded in current frame"，没人 reward "build a map over 5 minutes"。

RoboBrain: <https://arxiv.org/abs/2601.14352>  
Robix: <https://arxiv.org/abs/2509.01106>  
Cosmos-Reason: <https://docs.nvidia.com/cosmos/latest/reason2/index.html>

---

## 6. 最漂亮的部分：GST-Bench-Local 的三档剥离诊断

paper 第 4.3 节，我强烈建议你细看。他们对 ED (semantic) 和 EDist (semantic) 这两个 task 构造三个 setting：

- **Global**: 原始 GST-Bench，target 在 query view 不可见，必须跨帧。
- **Local-Video**: 把 query view 换成 target 可见的帧，exploration video 保留但逻辑上 redundant。测 "model 会不会被 distractor 干扰"。
- **Local-Image**: 直接把 exploration video 拿掉，只剩一张含 target 的图。任务退化成 single-image perception。

结果 Table 2 出来，两类 model 表现截然不同：

**Proprietary（Gemini-3-Pro / Gemini-2.5-Pro / GPT-5）**  
ED 从 Global 22.11 跳到 Local-Image 61.20，**+39 分**。EDist 也类似，+28 分。  
意思：这些 model **single-image 其实很强**，一加 video 跨帧就崩。failure mode 是 **integration bottleneck** —— 它们能局部看见，但无法把分散的观察整合成 coherent 的 global representation。

**Open-source（Qwen3-VL-2B/8B, InternVL3.5-2B/8B）**  
ED 上 4 个 model 里 2 个在 Local-Image 反而掉分（Qwen3-VL-8B -8.03, Qwen3-VL-2B -4.55）。EDist 全涨，但绝对分仍远低 —— InternVL3.5-2B Local-Image 25.40 vs Gemini-2.5-Pro 66.49。  
意思：开源 model **perception 和 integration 都 fail**，且 perception 是更紧迫的瓶颈。2B-8B 这种 scale 连 basic visual grounding 都没学好。

**一个反直觉现象**：GPT-5 在 EDist Local-Video 上 -1.38。给它一个 redundant 的 video，它反而被干扰。这是非常强的 signal —— 说明 model 没有 "select most informative evidence" 的策略，会被 spurious correlation 拉走。这跟最近 LLM reasoning 文献里 "more context hurts" 是一类问题，这里第一次在 spatial 上被量化。

---

## 7. GST-Train + Fine-tune：supervision 能补多少

用同一 pipeline 在 BEHAVIOR-1K、ArtVIP、HyperSim 和内部 sim 上造训练数据（scene 与 eval split 不重叠，防 leakage），拿 Qwen3-VL-8B 做 SFT，混入 general 多模态 instruction data 保通用能力。

结果（Table 1 最后一行）：
- Avg: 25.89 → 53.52 (**+27.63**)
- ED_v: 18.75 → 53.72
- EDist_v: 5.57 → 39.70
- Ori: 17.64 → 44.39
- TDS_E: 96.76 → 99.07

**几个值得咂摸的点：**

**(a) SFT 把 8B 拉到超过所有 proprietary zero-shot。** Gemini-3-Pro 42.68 < 53.52。这说明这个 gap 不是 architectural ceiling，是 **data + objective gap**。

**(b) 但仍差 human 25.56 分。** Paper 诚实地说 "improves, but does not fully solve"。剩下的是更难的，可能需要 architecture-level 改造（explicit 3D representation / map module）。

**(c) EDist 从 5.57 暴涨到 39.70，要警惕。** 我猜测 model 学到的不是 "真的从单目恢复 metric depth"，而是 "学到了 simulation 里的尺度先验 + 物体类别尺度先验"。OmniGibson 里 desk 高度、bed 长度都有规律，model 大概率在 memorize 这些 prior。这跟 monocular depth estimation 文献里 "models actually predict category-scale priors" 的发现（DPT、ZoeDepth）完全吻合。后续工作需要 out-of-distribution scene scale 的 test 来 verify 这点。

DPT: <https://arxiv.org/abs/2103.13413>  
ZoeDepth: <https://arxiv.org/abs/2302.12288>

---

## 8. 相关联想（给 Karpathy 你顺着看）

**(1) Thinking in Space / VSI-Bench [28]**：GST-Bench 直接继承 VSI-Bench 的 MRA metric 和 egocentric video setup，但把 single-frame solvable 任务彻底剔掉，并加入 top-down image 这把 probe。同一研究组（Stanford + NYU）的延续路线。  
<https://thinking-in-space.github.io/>

**(2) Cambrian-S / VSI-Super [29]**：强调 long-horizon recall + continual counting，偏 memory load。GST-Bench 偏 cross-view alignment，两者互补。

**(3) MMSI-Bench [30]**：多图 spatial intelligence，也要求 integrate across images，但没 top-down 这种 explicit global representation。GST-Bench 多走了一步。  
<https://github.com/OpenGVLab/MMSI-Bench>

**(4) OpenEQA [19]**：真实世界 episodic memory + active exploration QA。GST-Bench 与之形成 "sim vs real" 对照 —— sim 的好处是精确 geometry ground truth + off-trajectory query view，real 数据做不到。  
<https://open-eqa.github.io/>

**(5) SLAM literature**：GST-Bench 的 self-localization 本质上就是 monocular SLAM 里的 "global relocalization"。SLAM 的解法是 feature matching + PnP，VLM 想用 pure feed-forward 推理实现。这让我想到：VLM 可能需要 explicit map module（类似 Droid-SLAM、NICER-SLAM 这种 differentiable SLAM）而非纯 hidden state 承载全局 map。  
Droid-SLAM: <https://github.com/princeton-vl/DROID-SLAM>

**(6) BLINK [6]**：发现 "VLMs can see but not perceive"。GST-Bench 在 spatial + video + global scale 上印证了这个 finding，top-down selection 的 easy → medium → hard 下滑曲线就是证据。

**(7) SpatialBot / SpatialRGPT [31]**：从 single-image 教 VLM 做 depth / relative position。GST-Train 算是这条线在 video + global scale 上的延伸。

**(8) 跟你讲过的 "microscope vs world model" 框架的关系**：GST-Bench 测的是 world model 的一个具体切面 —— "VLM 能不能在 hidden state 里维持一个 allocentric map"。结果是不能。这跟 LLM world model 的争论（你在几个 podcast 里提过）是一个具体的 empirical 数据点。当前 transformer 的 attention 在 long context 上虽然能 attend，但 "把分散的 token 整合成 coherent 的 allocentric representation" 是另一回事。GST-Bench 给了这个直觉一个可量化的 benchmark。

---

## 9. 我读完的 take-aways（直觉版）

1. **GST-Bench 真正的杀手锏是 construction-level anti-shortcut**。"target invisible from query view" 这条 constraint 比 "用数值不用 categorical" 更根本。它让 benchmark 从根上无法被 single-frame VLM 作弊。

2. **Top-down image 是最聪明的一手**。它把 model 的 "internal representation" 强制 externalize 到 explicit 2D allocentric frame，让你能问 "你脑子里那张地图对不对"。比单纯问 egocentric relation 信息量大得多。

3. **Proprietary vs Open-source 的 failure mode 分裂**对社区很重要。开源圈需要先补 perception（basic spatial grounding data），闭源圈需要补 integration（跨帧 consistency loss、map reconstruction auxiliary task）。两类 model 下一步训练方向应该不同。

4. **GST-Train 的 27 分提升里，EDist 暴涨（5.57 → 39.70）最值得警惕**。可能是 prior memorization 而非真正的 metric perception。后续需要 OOD scene scale test。

5. **未来 architectural 方向我赌 "VLM + explicit differentiable map module"**。Pure VLM 用 hidden state 承载全局 map，ceiling 不够。neural SLAM + VLM 的 hybrid 可能是正路。

6. **Open problem**：怎么拓到 outdoor / unbounded scene。OmniGibson 是 indoor，户外 embodied agent（自动驾驶、delivery robot）的 global spatial awareness，top-down representation 都得重新设计。

7. **另一个我特别想看的 follow-up**：把 GST-Bench 和 RL post-training 结合。现在 GST-Train 是 SFT，如果用 RL（GRPO 之类）让 model 自己探索 scene 并 reward 正确 spatial answer，可能比 SFT 更能学到真正的 integration 能力，因为 SFT 容易 memorize prior。这跟 DeepSeek R1 / o1 的推理学习路线是同一思路。

---

一句话总结：GST-Bench 把 "VLM 能不能在脑子里建一张连贯的全局地图" 这件模糊的能力，变成了 12 个可精确数值评测的子任务，并用 construction-level constraint 杜绝 single-frame shortcut。结果发现所有 VLM 都扑街，最强 Gemini-3-Pro 42.68 vs human 79.08。但 SFT 能补回 27 分，说明这不是 architecture ceiling，是 data + objective gap。剩下 25 分怎么补，可能需要 architecture-level 改造。

Project page 再贴一次: <https://qwerirwq.github.io/GST-Bench/>

---

# GST-Bench: VLMs 的 Global Spatial Awareness 体检报告

Andrej, 这篇paper戳中了当前 VLMs 一个非常根本却长期被绕开的痛点 —— **long-horizon cross-frame spatial reasoning**。我会按 intuition 优先的顺序展开, 并把公式、变量、实验数据拆细来谈。

---

## 1. Motivation: 为什么 "global spatial awareness" 是被忽视的一块

现有 spatial benchmark 大致分两派:

- **Single-image 派** (What'sUP [10], CV-Bench [25], 3DSRBench [18], Spatial457 [27]): 判断 left/right/under 这类 categorical 关系, 或者做 relative depth ordering。这类任务 VLM 只需在一张图里做几何推理。
- **Multi-image / Video 派** (MM-Spatial [4], MMSI-Bench [30], VSI-Bench [28], VSI-Super [29], STI-Bench [13], OST-Bench [15]): 开始 push cross-view, 但有三个共同缺陷, 这正是 GST-Bench 要打中的三根钉子:
  1. 不区分 "single-frame solvable" vs "strictly cross-frame" 任务。比如 VSI-Bench 里 object-size estimation 或者 OST-Bench 里 existence query, 有时一帧就能答, 把 benchmark 的信噪比稀释了。
  2. Direction reasoning 还停留在 coarse categorical (front/back/left/right), 没有精细角度。
  3. 几乎从不要求 model 把 egocentric observation **显式 align 到 global representation** (top-down image)。

直觉上, 这对应 embodied AI 的一个老问题: 一个 household robot 在房间里走完一圈, 它必须 incremental 地 accumulate spatial memory, 然后 "在脑子里重建一张鸟瞰图", 才能回答 "我现在在哪 / 之前看到的咖啡杯在我后面多远 / 这个房间长什么样"。这件事人类不费力, 但 VLMs 至今做得很差。GST-Bench 的目的就是把这个能力单独剥离出来量化。

Project page: <https://qwerirwq.github.io/GST-Bench/>

---

## 2. GST-Bench 的三条腿和十二个 subtask

三大 competencies 对应 embodied agent 必须回答的三个问题:

### 2.1 Self Localization (Where am I?)
- **Global Orientation**: 给 exploration video + medium top-down + novel current view, 推断 current view 在 top-down 上的 camera orientation。
- **Global Position**: 同上, 推断 camera position。

### 2.2 Object Localization (Where is the target?)
关键设计 —— **target object 必须 invisible from current view**。这条 constraint 是 GST-Bench 区别于其他 benchmark 的灵魂, 它从 construction 上杜绝了 single-frame shortcut。又分 semantic (用 category name 指定) 和 visual (用 object-annotated video 上的 red bbox 指定) 两种 modality:
- Egocentric Direction (semantic / visual)
- Egocentric Distance (semantic / visual)
- Global Position on top-down (semantic / visual)

### 2.3 Scene Structure Understanding (What does the scene look like?)
- **Top-Down Selection (easy / medium / hard)**: 三种 abstraction level —— easy 是 photo-realistic bird's-eye view (去 ceiling, 保留 texture), medium 是 occupancy-style map (保留 footprint, 去 appearance), hard 是 bare floor plan (只剩 walls)。这个递进设计非常聪明, 它直接 probe model 在多大程度上依赖 appearance cue vs 真正的几何/拓扑结构。
- **Trajectory Selection**: 给一段 short trajectory clip + top-down 上四条候选 trajectory, 选匹配的。

直觉上, 这 12 个 subtask 覆盖了 "where am I / where is X / what does the world look like" 的完整闭环, 任何一个环节断了都能被独立抓出来。Table 1 里你可以看到每个 subtask 的 random baseline, 这对后续诊断很重要。

---

## 3. Data Pipeline: simulation 是这盘棋的关键

paper 选了 OmniGibson + BEHAVIOR-1K [11] 作为底层 sim。这里有几个非常重要的工程决策:

**a) Simulation 给的可控性**:
- 可以 sample arbitrary viewpoints (off-trajectory novel view), 这是天然 anti-shortcut 的。
- Camera pose, object pose, visibility, distance, angle, top-down projection 全部从 scene geometry 直接计算, **answer 不需要人标, 只需要人 verify**。
- 多 trajectory per scene (perturbed viewpoints, 不同 start point), 防止 model 走 canonical route 的死记硬背。

**b) 五类 input**:
- Exploration video (primary memory source)
- Object-annotated video (visual modality 的 target specification, 红框覆盖所有 target 可见帧)
- Short trajectory video (trajectory selection 任务的局部 motion observation)
- Top-down images (easy/medium/hard, global representation)
- Current view (off-trajectory novel viewpoint)

**c) Human verification 双重 check**:
1. target object 在 exploration/object-annotated video 里 **能不能被识别出来** (answerability)
2. off-trajectory current view 与 exploration video 之间 **有没有足够 overlap 让人能 localize** (answerability)

最终拿到 2,762 human-verified questions, 累积 6,790 minutes of video。量级上比 VSI-Bench (5k+ 长视频) 略小但 per-question 平均 video input 长得多, 因为 embodied exploration 路径天然就长。

OmniGibson: <https://behavior.stanford.edu/omnigibson>  
BEHAVIOR-1K: <https://behavior.stanford.edu/>

---

## 4. Metrics 详解 (这是大家最容易跳过的部分, 但其实最影响解读)

paper 用了 4 种 question format, 每种对应一个 metric。这里我重点拆公式, 因为这是 build intuition 的关键。

### 4.1 Distance: Mean Relative Accuracy (MRA)

$$\mathrm{MRA} = \frac{1}{|\mathcal{C}_d|} \sum_{\theta \in \mathcal{C}_d} \mathbb{1}\!\left( \frac{|\hat{y} - y|}{y} < 1 - \theta \right). \tag{1}$$

变量说明:
- $\hat{y}$: model 预测的距离
- $y$: ground truth 距离
- $\mathcal{C}_d = \{0.50, 0.55, \dots, 0.95\}$: 一组 confidence threshold, 共 10 个值, 间隔 0.05
- $\theta$: 遍历 $\mathcal{C}_d$ 中每个 threshold
- $\mathbb{1}(\cdot)$: indicator function, 条件成立为 1, 否则 0
- $|\mathcal{C}_d|$: 集合元素个数, 这里是 10

直觉: 把 "$\frac{|\hat{y}-y|}{y} < 1 - \theta$" 重排, 等价于 "$\frac{|\hat{y}-y|}{y} < 1-\theta$", 即 **相对误差小于 $1-\theta$**。当 $\theta = 0.95$ 时, 要求相对误差 < 5% (严格); 当 $\theta = 0.50$ 时, 要求相对误差 < 50% (宽松)。MRA 就是在 10 个不同严格度上各打一个分, 然后取平均。这比单一 threshold 更能反映 distribution of error, 而且对**距离的尺度无关** (因为用了相对误差)。这个设计直接来自 VSI-Bench [28]。

VSI-Bench 原文: <https://thinking-in-space.github.io/>

### 4.2 Angle: circular distance + 多 threshold accuracy

$$e_a = \min\!\left( |\hat{\alpha} - \alpha|, \ 360^\circ - |\hat{\alpha} - \alpha| \right). \tag{2}$$

- $\hat{\alpha}$: 预测角度
- $\alpha$: ground truth 角度
- $e_a$: 角度误差, 用 circular distance 处理 "350° vs 10°" 这种 wrap-around 的情况 —— 如果用绝对值, 误差会是 340°, 但实际上只差 20°。$\min(\cdot, \cdot)$ 取较小的那个就是绕最短弧。

$$\mathrm{Acc}_{\mathrm{angle}} = \frac{1}{|\mathcal{C}_a|} \sum_{\tau \in \mathcal{C}_a} \mathbb{1}\!\left( e_a < \tau \right). \tag{3}$$

- $\mathcal{C}_a = \{15^\circ, 30^\circ, 45^\circ\}$: 三档 tolerance
- $\tau$: tolerance threshold

这里 threshold 选 15° / 30° / 45° 是有讲究的 —— 45° 大致是 8 个方位之一 (360/8=45), 15° 接近 "我能精确定位到这个物体在我前面偏左一点" 的水平。MRA 的 multi-threshold 思路在这里再次复用。

### 4.3 Point: Euclidean distance + 多 threshold

$$e_p = \|\hat{\mathbf{p}} - \mathbf{p}\|_2 \tag{4a}$$

$$\mathrm{Acc}_{\mathrm{point}} = \frac{1}{|\mathcal{C}_p|} \sum_{\tau \in \mathcal{C}_p} \mathbb{1}\!\left( e_p < \tau \right). \tag{4b}$$

- $\hat{\mathbf{p}}, \mathbf{p} \in \mathbb{R}^2$: predicted / ground-truth pixel 坐标 (top-down 上)
- $\|\cdot\|_2$: L2 norm, 即 $\sqrt{(\hat{p}_x - p_x)^2 + (\hat{p}_y - p_y)^2}$
- $\mathcal{C}_p = \{100, 150, 200, 250, 300\}$ pixels

注意 unit 是 pixel 而非 metric meter, 因为 top-down 是渲染出来的 2D 图。100 像素大致对应一个中等房间内的若干米, 300 像素则是 "大致在哪个房间" 的尺度。多 threshold 的好处是曲线能反映 error distribution, 而不是 single point estimate。

### 4.4 Overall

$$\mathrm{Score} = \frac{1}{12} \sum_{i=1}^{12} s_i. \tag{5}$$

- $s_i$: 第 $i$ 个 subtask 在其对应 metric 上的得分
- 12: subtask 数量, **简单算术平均, 不加权**

不加权这个决策 paper 没明说, 但我推测是因为他们想让每个 subtask 都有 equal diagnostic value, 而不是被容易的 task 主导。

---

## 5. 主结果 Table 1 的几个关键读法

让我挑几个 cell 来 build intuition:

| 维度 | Best model | Human | Gap |
|---|---|---|---|
| Avg | Gemini-3-Pro 42.68 | 79.08 | -36.40 |
| Ori (orientation) | Gemini-3-Pro 21.52 | 85.00 | -63.48 ← 最大 |
| GP (global position) | Gemini-3-Pro 42.23 | 93.00 | -50.77 |
| EDist (egocentric distance) | Gemini-2.5-Pro 42.00 | 41.50 | +0.50 ← 反常 |

几个非显然的观察:

**(1) Orientation 是最 catastrophic 的 task。** Gemini-3-Pro 21.52 vs human 85.00, 差 63 点。这背后反映的是: VLM 看图时缺乏稳定的 camera-relative frame, 把 "我在哪、朝哪" 这种 self-pose estimation 当作空间推理的零点。这一点和 SLAM 文献里 "estimated pose 精度直接决定后续 metric" 的逻辑完全一致 —— pose 错了, 后面 object localization 全错。这是这个 paper 最重要的 finding 之一, 因为它直接指向 architectural defect: 当前 VLM 没有 explicit 的 pose / pose-graph representation。

**(2) EDist 上 human 本身就只有 41.50 MRA, 而模型 ~42 MRA。** Paper 里特别 caution 不要把这个误读成 "model 已经追上人"。绝对 metric distance 从单目视频恢复是 ill-posed 的, 人也做不好。这里 "打平" 反而说明这个 subtask 的 ceiling 被感知极限压住了, 应该把它看作 "noisy reference" 而非 "solved"。

**(3) Top-Down Selection (easy)** 几乎所有 model 都 80+, Gemini-3-Pro 甚至 98.15, 而 random 只有 23.15。这是 12 个 subtask 里 VLMs 唯一像样的地方。但只要去掉 appearance (medium), 立刻掉到 60-80; 再去掉 object footprint 只剩 wall (hard), 又掉到 30-50。这条曲线直观证明: VLMs 主要靠 **appearance-level template matching** 而非真正的 **structural / topological reasoning** 在做 top-down 匹配。这点和 BLINK [6] 的发现 "VLMs can see but not perceive" 形成呼应。

**(4) Embodied-tuned models 反而更差。** RoboBrain2.5-8B (24.61) 和 Cosmos-Reason2-8B (21.64) 都低于 general-purpose Qwen3-VL-8B (25.89); Robix-32B (29.26) ≈ Qwen3-VL-32B (30.43)。Paper 的解释是: 现有 embodied post-training 强调 local affordance 和 local spatial relation, 而非 long-horizon spatial memory 和 cross-viewpoint alignment。这是非常重要的 negative result, 它说明 "spatial intelligence for robotics" 这个 narrative 在数据/任务设计层面就被局部化锁死了。这跟 RoboBrain / Robix / Cosmos-Reason 这条线整体的训练分布是吻合的 —— 它们 reward 的是 "next action grounded in current frame", 而非 "build a map over 5 minutes"。

RoboBrain: <https://arxiv.org/abs/2601.14352>  
Robix: <https://arxiv.org/abs/2509.01106>

---

## 6. 真正的 key insight: GST-Bench-Local 的三档剥离诊断

这一节是 paper 最漂亮的部分, 但很容易被快速翻过。Andrej 你一定要细看。

### 6.1 三个设置
对 ED (egocentric direction, semantic) 和 EDist (egocentric distance, semantic) 两个 task, 构造三个变体:

- **Global**: GST-Bench 原始设置, target 在 current view 不可见, 需要跨帧。
- **Local-Video**: 把 current view 换成 target 可见的帧, exploration video 保留但逻辑上 redundant。这个设置 probe "model 会不会被 distractor 干扰"。
- **Local-Image**: 直接把 exploration video 拿掉, 只剩一张含 target 的 current view。任务退化成 single-image spatial perception。

### 6.2 Table 2 的两幅截然不同的画像

**Proprietary (Gemini-3-Pro, Gemini-2.5-Pro, GPT-5)**: 
- ED (Global → Local-Image): Gemini-3-Pro 22.11 → 61.20 (+39.09), Gemini-2.5-Pro 23.08 → 66.49 (+43.41), GPT-5 31.09 → 65.61 (+34.52)
- EDist (Global → Local-Image): Gemini-3-Pro 27.39 → 55.79 (+28.40)

读法: 顶级模型在 **single-image 上其实很强**, 但一加 video + 跨帧, 掉 30+ 分。这说明 failure mode 是 **integration bottleneck**: 它们能 "局部看见", 但无法 "全局整合"。这呼应了你 (Karpathy) 在多个 talk 里讲过的 intuition —— 当前 transformer 的 attention 在长 context 上虽然能 attend, 但 "把分散的 token 整合成一个 coherent 的 allocentric representation" 是另一回事。

**Open-source (Qwen3-VL-2B/8B, InternVL3.5-2B/8B)**:
- ED 上 4 个 model 里 2 个在 Local-Image 反而 **掉分** (Qwen3-VL-8B -8.03, Qwen3-VL-2B -4.55)
- EDist 全部涨, 但绝对分仍远低: InternVL3.5-2B Local-Image 25.40 vs Gemini-2.5-Pro 66.49

读法: open-source model **同时在 perception 和 integration 两个阶段都 fail**, 且 perception 是更紧迫的瓶颈。这非常符合 scaling literature 的直觉 —— 小模型 (2B-8B) 连 basic visual grounding 都没学好, 你让它做 cross-frame integration 是奢望。这点和 NVILA [17]、LLaVA-OV [1] 在 BLINK / MM-Spatial 上的表现也是一致的。

### 6.3 一个反直觉现象 (paper 没强调但我觉得重要)

GPT-5 在 EDist Local-Video 上 -1.38。也就是说, 给它一个 redundant 的 video, 它反而被干扰。这是一个非常强的 signal: 它说明 model 的 reasoning 没有形成 robust 的 "select the most informative evidence" 策略, 而是被 spurious correlation pull 走。这和最近 LLM reasoning literature 里 "more context hurts" 现象 (比如 FRAMES, LongBench v2 上的 finding) 是同一类问题, 这里第一次在 spatial 上被量化出来。

---

## 7. GST-Train + Fine-tune: 用 supervision 部分缝合

用同一 pipeline 在 BEHAVIOR-1K, ArtVIP [9], HyperSim [21] 和内部 sim scene 上造训练数据 (与 eval scene disjoint, 防 leakage), 把 Qwen3-VL-8B 拿来 SFT, 同时混入 general 多模态 instruction data 保住通用能力。

结果 (Table 1 最后一行):
- Avg: 25.89 → 53.52 (+27.63)
- ED_v: 18.75 → 53.72 (+34.97)
- EDist_v: 5.57 → 39.70 (+34.13)
- GP_v: 17.35 → 48.97 (+31.62)
- Ori: 17.64 → 44.39 (+26.75)

这里有几个值得注意的点:

**(a) SFT 把 8B 拉到超过所有 proprietary zero-shot 模型** (Gemini-3-Pro 42.68 < 53.52)。这是 paper 的核心 claim: 这个 gap 不是 architectural ceiling, 而是 **data + objective gap**。这跟 VSI-Bench [28] 后 Cambrian-S [29] 的逻辑很像 —— 先 benchmark 揭示 gap, 再 supervised SFT 收一部分回来。

**(b) 但仍差 human 25.56 分。** Paper 诚实地说 "improves, but does not fully solve"。这暗示剩下的是更难的: 可能是 architecture-level 的 (需要 explicit 3D representation / map module), 也可能是 data-level (SFT 数据多样性不够)。

**(c) EDist 从 5.57 跳到 39.70**, 4 倍以上。这非常引人联想 —— fine-tune 之后 model 学到的可能不是 "真的从单目恢复 metric depth", 而是 "学到了 sim 里的尺度先验 + 物体类别尺度先验"。OmniGibson 里 desk 高度、bed 长度都是有规律的, model 大概率是 memorize 这些 prior 来 estimate 距离。这跟 monocular depth estimation 文献里 "models actually predict category-scale priors" 的发现一致 (e.g. DPT, ZoeDepth)。

---

## 8. 和我想到的相关 paper / thread (扩展联想)

Andrej, 你应该会觉得下面这些联系有意思:

**(1) Thinking in Space / VSI-Bench [28]**: GST-Bench 直接继承 VSI-Bench 的 MRA metric 和 egocentric video 的 setup, 但把 "single-frame solvable" 任务彻底剔掉, 并加入 top-down image 这把 probe。这是同一研究组 (Stanford + NYU) 的延续路线。

**(2) Cambrian-S [29]**: VSI-Super, 强调 long-horizon recall + continual counting, 思路接近但偏 memory load。GST-Bench 偏 cross-view **alignment**, 两者形成互补。

**(3) MMSI-Bench [30]**: 多图 spatial intelligence, 也要求 "integrate across images", 但没有 top-down 这种 explicit global representation。GST-Bench 的 top-down 这一手比 MMSI-Bench 多走一步。

**(4) SpatialBot / SpatialRGPT [31]**: 从 single-image 教 VLM 做 depth / relative position; GST-Train 算是这条线在 video + global scale 上的延伸。

**(5) OpenEQA [19]**: 真实世界 episodic memory + active exploration QA。GST-Bench 与 OpenEQA 形成 "sim vs real" 对照, sim 的好处就是精确 geometry ground truth + off-trajectory query view, 这是 real 数据做不到的。

**(6) SLAM literature**: GST-Bench 的 self-localization 任务在形式上就是 monocular SLAM 的 "global relocalization" 子问题。SLAM 的解法是用 feature matching + PnP, 而 VLM 想用 pure feed-forward 推理实现。这个对照让我想到: 也许 VLM 需要一个 explicit 的 map module (类似 Differentiable SLAM, e.g. Droid-SLAM, NICER-SLAM) 而非纯 hidden state。

**(7) Embodied AI 的 map module 路线**: 和 Habitat/WebArena 上的 "Object-Goal Navigation" 任务高度相关。GST-Bench 本质上就是把这个 nav task 拆成可单独 eval 的 sub-skill。

OpenEQA: <https://open-eqa.github.io/>  
MMSI-Bench: <https://github.com/OpenGVLab/MMSI-Bench>

---

## 9. 我读完之后的几个 take-aways (intuition)

1. **GST-Bench 真正的杀手锏是 construction-level anti-shortcut**: "target invisible from query view" 这一条 constraint, 比 "metric 用数值不用 categorical" 更重要。它让 benchmark 从根上无法被 single-frame VLM 作弊通过。

2. **Top-down image 是这盘棋最聪明的一步**: 它把 model 的 "internal representation" 强制 externalize 到一个 explicit 2D allocentric frame, 让你可以问 "你脑子里那张地图对不对"。这比单纯问 egocentric relation 信息量大得多。

3. **Proprietary vs Open-source 的 failure mode 分裂**: 这对社区很重要。开源圈需要先补 perception, 闭源圈需要补 integration。两类 model 的下一步训练方向应该不同 —— open-source 应该加更多 basic spatial grounding data, proprietary 应该加 long-horizon integration 的训练 (e.g. 跨帧 consistency loss, map reconstruction auxiliary task)。

4. **GST-Train 的 27 分提升里, 距离预测的暴涨 (5.57 → 39.70) 最值得警惕**。它可能是 prior memorization 而非真正的 metric perception。后续工作需要设计 out-of-distribution scene scale 的 test 来 verify。

5. **未来 architectural 方向**: 我会 bet 在 "VLM + explicit differentiable map module" 上, 类似 neural SLAM 与 VLM 的 hybrid。Pure VLM 用 hidden state 承载全局 map, 看起来 ceiling 不够。

6. **Open problem**: 如何把这个 benchmark 拓到 outdoor / unbounded scene。OmniGibson 是 indoor, 但真正 embodied agent (e.g. 自动驾驶, 户外 delivery robot) 需要 outdoor global spatial awareness, 那时 top-down image 的 representation 都要重新设计。

---

总结一句, GST-Bench 的核心贡献是 **把 "VLM 能不能在脑子里建一张连贯的全局地图" 这件事, 从一个模糊的能力, 变成了 12 个可精确数值评测的子任务, 并用 construction-level constraint 杜绝 single-frame shortcut**。这个 benchmark 应该会成为后续 embodied VLM 评测的标配之一。

Project page 再贴一次: <https://qwerirwq.github.io/GST-Bench/>
