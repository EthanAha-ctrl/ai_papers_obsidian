---
source_pdf: Embodied Navigation with Adaptive Reasoning and Visual-Assisted Linguistic
  Memory.pdf
paper_sha256: 1a8fd53081b5115cf0c2f4bfa4951fce9a852cb955789da95c379c9058a931e6
processed_at: '2026-08-04T03:33:32-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VLingNav 论文的人话版

## 一句话总结

让 robot 学会像人一样走路——简单的时候自动驾驶，遇到岔路口才停下来思考，思考完把环境关键信息写成"日记本"存起来，下次不用重复探索。

## 1 这篇 paper 想解决什么真问题

现在做 embodied navigation 的 VLA model 有个尴尬处境：你看 NaVid、Uni-NaVid 这些工作，把 video 喂给 VLM 让它直接 output action，表面上 unified 美观，实际三个硬伤：

**硬伤一：不会"想"**。Model 从 observation 到 action 是一个 reactive mapping，没有 explicit reasoning。遇到岔路口、目标被遮挡、room 都查过一遍还没找到——这些场景需要"停下来想一想"，但 model 只会一味往前冲，撞墙、打转、无限循环。

**硬伤二：记不住**。Video-based VLA 用 visual features 做历史 memory，听着合理，但 visual features 每经过一次 attention layer 就被压缩一次，传到第 50 步时，第 1 步的 semantic 信息已经面目全非。Robot 进了 room A，出来，走了一大圈又走回 room A——因为它"忘了"自己来过。

**硬伤三：模仿天花板**。SFT 训练的 model 上限就是 expert policy 的水平。Expert 走最短路，model 学会走最短路，但 real-world 不可能总走最短路——遇到 dynamic obstacle、dead end、goal position 微调，model 就懵了，因为 training data 里没见过这种 distribution。

VLingNav 的核心 insight：**navigation 不只是 perception-to-action 问题，是 cognitive resource allocation 问题**。Robot 应该像人一样，大多数时候用"直觉"走（System 1），关键决策点才停下来"思考"（System 2），思考完用语言记下关键信息，下次不用重复劳动。

## 2 AdaCoT：什么时候该想

### 2.1 核心机制的直觉

想象你走进一个没去过的 office 找"会议室"：
- **走廊直走**：不需要想，腿自己动
- **岔路口**：停下来，看看左右标志，决定往哪走
- **看到一个 room**：进去扫一眼，没找到目标，记住"这个 room 是空的"，出来
- **目标被遮挡**：停下来分析遮挡物后面是不是目标

AdaCoT 就是让 model 学会这种"该想的时候想，不该想的时候别浪费 computation"。

具体实现：model 先预测一个 indicator token——`<think_on>` 或 `<think_off>`。`<think_on>` 就生成 reasoning 内容 + 一个 `<summary>` 标签的环境总结；`<think_off>` 就直接输出 action。

这个 `<summary>` 内容会被存进 memory，作为后续输入的一部分。所以 reasoning 不是为了"看起来在思考"，是为了产生可持久化的 linguistic memory。

### 2.2 为什么 2.1% 触发率就够了

Table 6 的 ablation 数据非常有意思：

| 策略 | 触发率 | ObjNav SR |
|---|---|---|
| 不思考 | 0% | 36.2 |
| 每步都思考 | 100% | 25.3 |
| 每 5 步思考一次 | 20% | 42.5 |
| **Adaptive** | **2.1%** | **50.1** |

每步都思考的 SR 反而最低（25.3）！这个结果乍一看反直觉，但仔细想就明白了：

CoT token 占据 context window，每步都 reasoning 等于把一堆 reasoning 文本塞进 input。这些 reasoning 文本本身是有 noise 的（model 不可能每步都 reasoning 正确），noise 累积会干扰 action prediction。更糟的是，CR（碰撞率）从 5.51 飙到 26.3——model 想太多反而走不好。

Adaptive 2.1% 触发率意味着 100 步里只有 2 步触发 reasoning，但这两步是真正关键的决策点。Information theory 角度：action entropy 在不同 step 是 non-uniform 的，corridor 直走 entropy ≈ 0，intersection entropy 高。把 computation 分配到 high-entropy step 是 optimal allocation。

Reference: Kahneman 的 dual-process theory
https://en.wikipedia.org/wiki/Dual_process_theory

## 3 VLingMem：用语言记事

### 3.1 为什么用语言不用图

这是我觉得这篇 paper 最有 insight 的设计选择。

Visual memory 的根本问题：visual features 是 dense high-dimensional representation，每次被 attention 处理都会被 implicit 压缩。压缩 50 步之后，第 1 步的"我进过 room A"这个 semantic fact 已经被压成噪声。

Linguistic memory 的优势：language 是 discrete symbolic representation，不会因为多次 attention 而退化。"我已经查过 room A" 这句话作为 token 输入，第 1 步和第 100 步语义完全一样。

更关键的：VLM backbone 经过 trillions tokens 预训练，language 是它的 native modality。你给它一段自然语言描述，它能立刻理解；给它一段 compressed visual features，它需要重新 decode。Linguistic memory 直接利用了 pre-trained alignment。

### 3.2 Memory 的更新时机

从 Algorithm 1 可以看出一个关键设计：**memory 只在 `<think_on>` 时更新**。

这意味着 memory 是 episodic 的——只在 critical decision point 记录环境 summary，而不是每步都记。这就像人写日记——只在重要事件发生时写，不会把"我从厨房走到客厅"也记下来。

Sparse updating 有三个好处：
1. Memory 体积小，不爆 context window
2. 每条 memory 都是 high-information 的，不是 trivial 的
3. 与 AdaCoT 形成 synergy——`<think_off>` 时虽然不 reasoning，但 memory 仍提供历史 context

### 3.3 Ablation 数据验证

Table 7：

| Memory Mode | ObjNav SR |
|---|---|
| 无 memory | 15.4 |
| 只用 visual | 45.2 |
| 只用 language | 18.8 |
| **Visual + Language** | **50.1** |

几个有意思的点：
- 完全没 memory 灾难性（15.4），证明 long-horizon 必须 memory
- 只用 visual 比 只用 language 好（45.2 vs 18.8）——visual 携带的信息量确实大
- 但 visual + language 远超单一 modality（50.1 vs 45.2 vs 18.8）——两者 complementary

Visual 给 fine-grained perceptual detail，language 给 semantic abstraction。单独 visual 会忘，单独 language 信息量不够。组合起来：visual 处理 short-term 感知，language 维护 long-term 语义，分工明确。

## 4 Dynamic FPS：怎么处理长视频

### 4.1 Ebbinghaus 启发

Video-based VLA 有个现实问题：navigation episode 可能几百步，每步一帧 RGB，全塞进 model 不可行。

现有方案两个极端：
- Token merging（Uni-NaVid）：压缩历史 visual tokens，但 distort semantic features
- Uniform sampling（NaVILA）：均匀采样，低采样率丢 short-term 信息

VLingNav 用 Ebbinghaus forgetting curve 启发的设计：

$$f_s(i) = f_s^{max} \cdot e^{-\frac{\Delta T}{s}}$$

变量解释：
- $f_s(i)$：frame $i$ 的 sampling rate
- $f_s^{max}$：最新 frame 的采样率（最高）
- $\Delta T = t - i$：当前 frame $t$ 到历史 frame $i$ 的时间间隔
- $s$：memory stability，控制衰减速度

直觉：近期 frame 高采样率（short-term memory，detail 重要），远期 frame 低采样率（long-term memory，coarse 语义足够）。这跟人脑记忆一致——你能记住 5 分钟前走过的 corridor 大概什么样，但记不住 30 分钟前的具体细节。

配合 grid pooling：

$$g(i) = \lfloor e^{-\frac{\Delta T}{g}} \rfloor$$

老 frame 用更大 stride 做 spatial pooling，进一步减少 token 数。新 frame 保留 fine-grained spatial detail 用于 immediate decision。

### 4.2 Temporal-aware Token

Dynamic FPS 有个 side effect：frame 之间时间间隔不均匀。Model 需要知道"这帧是 5 秒前还是 50 秒前"，否则 temporal reasoning 会混乱。

解决方案：每个 frame 前加一个 temporal indicator token，用 RoPE 编码时间间隔：

$$E^T(\Delta T) = E_{base}^T + \text{RoPE}(\Delta T)$$

RoPE 用 rotary position encoding 让 model 感知绝对时间间隔。这个细节虽小，但对 dynamic FPS 的 coherence 很关键。

Reference: RoPE 原理
https://arxiv.org/abs/2104.09864

## 5 Online RL Post-training：突破模仿天花板

### 5.1 为什么需要 RL

SFT model 的极限就是 expert policy。但 expert（shortest path planner）走的是 ideal trajectory，real-world 不可能复制：
- Dynamic obstacle 出现，expert 没见过
- Model 走偏了，需要 recovery，但 training data 没有 recovery 示例
- Covariate shift：model 自己走的轨迹 distribution 与 expert 不同，越走越偏

Pure RL 又有 navigation 的特殊困难：long-horizon + sparse reward。一个 episode 几百步，只有最后找到 goal 才有 reward，credit assignment 极其困难。

### 5.2 Hybrid Rollout 的巧思

VLingNav 的 hybrid rollout 是我觉得最实用化的设计：

**Naive rollout**：model 自己走，保留成功 trajectory。提供 on-policy positive examples，让 model 强化"哪些 action sequence 能成功"。

**Expert-guided rollout**：model 卡住（oscillate 15 步）或失败时，expert planner 接管，演示 recovery path。提供 corrective examples，教 model "遇到 dead state 怎么逃出来"。

这种设计的精髓：把 long-horizon credit assignment 问题分解为多个 short-horizon supervised learning 问题。Expert 介入相当于在 trajectory 中插入 anchor points，model 不需要从 episode 终点反推 credit，而是直接学"从这个 stuck state 怎么 recovery"。

### 5.3 Loss 设计的微妙平衡

$$\mathcal{L}_{post}(\theta) = \lambda \mathcal{L}_{RL}(\theta) + (1-\lambda) \mathcal{L}_{SFT}(\theta)$$

其中 $\lambda = 0.01$。

这个 0.01 的选择很微妙——RL loss 只占 1%，SFT loss 占 99%。直觉上 RL 应该主导才对，但这里反过来。

为什么？因为 pure RL 在 long-horizon navigation 下 unstable。如果 RL loss 权重大，policy gradient 的 high variance 会让 model 忘掉 expert 教的基础能力（catastrophic forgetting）。

$\lambda = 0.01$ 的实际效果：SFT loss 保持 model 的基础 navigation 能力不退化，RL loss 提供微调信号让 model 探索 expert 之外的策略。这是"在 expert 基础上微调"而非"从零学起"。

### 5.4 Probabilistic Action 的设计

$$\pi_\theta(\mathbf{a}_t | \mathbf{s}_t) = \mathcal{N}\left(\boldsymbol{\mu}_\theta(\mathbf{h}_t), \text{diag}\left(\boldsymbol{\sigma}_\theta(\mathbf{h}_t)^2\right)\right)$$

变量：
- $\mathbf{h}_t$：VLM 输出的 hidden state
- $\boldsymbol{\mu}_\theta(\mathbf{h}_t)$：Gaussian mean，作为 deterministic action
- $\boldsymbol{\sigma}_\theta(\mathbf{h}_t)$：std，控制 exploration 程度

Rollout 时 sample：$\mathbf{a}_t \sim \pi_\theta(\cdot | \mathbf{s}_t)$，引入 stochastic exploration。
Validation 时用 mean：$\mathbf{a}_t = \mu_\theta(\mathbf{h}_t)$，deterministic execution。

这个设计避免了两个极端：
- Discrete tokenization（action 离散化）：损失 precision
- Diffusion/flow matching：iterative denoising 慢

Multivariate Gaussian 是最 lightweight 的 continuous action 参数化，inference 时只一次 forward pass。

## 6 Nav-AdaCoT-2.9M 数据集

### 6.1 规模

2.9M navigation steps + 472K CoT annotations + 1.6M open-world video samples = 4.5M total。

之前的 dataset 最多 110K CoT（Nav-CoT-110K），VLingNav 直接 472K，4 倍以上。更关键的是第一次集成 ObjectNav + EVT + ImageNav 三个 task。

### 6.2 Adaptive CoT 标注 pipeline

用 Qwen2.5-VL-72B 给 2.9M navigation steps 自动标注 CoT，最终 472K steps 触发 CoT（约 16.3%）。这个 16.3% 是 labeling 阶段的触发率，最终 model 学到的触发率只有 2.1%——model 自己学会了比 labeling 更 sparse 的 reasoning pattern。

Composite prompt 五要素：
1. Navigation instruction
2. 最近 10 frame egocentric view
3. Prior memory content
4. Expert trajectory at current step
5. 格式要求

两阶段过滤：
1. Rule-based check：丢弃 incomplete 或 inconsistent response
2. Quality verification：与 expert trajectory 交叉验证

### 6.3 Open-world Video Co-training

三个 open-world dataset：
- LLaVA-Video-178K：non-CoT subset
- Video-R1：CoT subset（challenging video QA）
- ScanQA：non-CoT subset

分类逻辑：Video-R1 的 challenging video QA 用 CoT 标注，让 model 学会"难问题才 reasoning"；其他 dataset 不用 CoT，让 model 学会"简单问题直接答"。这种 categorization 直接训练了 adaptive reasoning 能力。

Table 8 显示 co-training 对 ImageNav 提升最大（+10.6 SR），因为 open-world video 增强 cross-modal grounding，而 ImageNav 最依赖 image-text 对齐能力。

## 7 实验结果的关键 insights

### 7.1 MP3D 的巨大提升

MP3D 是 long-range exploration 主导的场景，VLingNav 对前 SOTA CogNav 提升 +12.3 SR (+26.4%) 和 +10.4 SPL (+32.8%)。

这个巨大提升直接验证 VLingMem 的 long-horizon memory 能力——long-range 场景最容易陷入重复探索，linguistic memory "我已经查过 room A" 这种 semantic fact 防止 loop。

### 7.2 SFT vs RL post-training

| Benchmark | SFT SR | Final SR | 提升 |
|---|---|---|---|
| HM3Dv1 | 70.6 | 79.1 | +8.5 |
| MP3D | 47.4 | 58.9 | +11.5 |

MP3D 的 RL 提升比 HM3Dv1 更大（+11.5 vs +8.5），说明 long-range 场景更受益于 RL exploration——RL 让 model 发现 expert 之外的更优策略。

### 7.3 ImageNav 的 SPL 飞跃

VLingNav 对 UniGoal 在 SR 上略高（+0.6），但 SPL 大幅提升 +13.7 (+57.8%)。

SR 提升小说明两者都能找到目标，SPL 提升大说明 VLingNav 走的路短得多。直接证据：linguistic memory 防止 redundant exploration，让 trajectory 更 direct。

### 7.4 Real-World Zero-Shot Transfer

最 impressive 的结果：simulation 训练的 model weights 直接部署到 Unitree Go2 quadruped，no real-world fine-tuning，在 home/office/outdoor 三种场景完成 ObjectNav/EVT/ImageNav 三种 task。

Inference latency <300ms，加上 100ms communication 共 ~400ms，实现 2.5 FPS。这个速度对 quadruped navigation 已经足够。

Zero-shot transfer 成功的三个因素：
1. Multi-task training 学到 generalizable representation
2. Open-world video co-training 缩小 sim-to-real gap
3. Adaptive CoT + linguistic memory 提供 robust cognitive scaffold，遇到 real-world noise 时 model 能 trigger reasoning 重新分析

### 7.5 Emergent Cross-Task Composition

Figure 9 展示了训练时没见过的 composition：
- "找到 X 然后 tracking X"——训练时 ObjectNav 和 EVT 是分开的 task，real-world 却能 compose
- "找到 image goal 然后 tracking"——ImageNav + EVT 的 composition

这种 compositionality 来自 unified VLA architecture + multi-task co-training。Model 学到的不是 task-specific policy，而是 shared navigation priors，这些 priors 可以自由组合。

## 8 我的直觉性思考

### 8.1 为什么 Adaptive 比 Fixed 好——Information Theoretic 视角

Navigation 的 action entropy 在不同 step 是高度 non-uniform 的：
- Corridor 直走：action entropy ≈ 0（continue forward）
- Intersection：action entropy 高（left/right/forward）
- Goal occlusion：action entropy 高（绕左/绕右/前进查看）

Fixed interval CoT 无论 entropy 都花 same computation，要么 over-thinking routine step（waste），要么 under-thinking critical step（miss）。Adaptive CoT 把 computation 分配到 high-entropy step，是 optimal resource allocation。

这个 insight 对 LLM reasoning 也有启发——DeepSeek-R1 这类 reasoning model 每个问题都 long CoT，但很多简单问题根本不需要。Adaptive reasoning 是 future direction。

### 8.2 Linguistic Memory 的深层意义

从 cognitive science 角度，人类 episodic memory 就是 linguistic 的。你能回忆"昨天在咖啡店见了朋友"，这个 memory 是以语言形式存储的，不是 visual pixels 重放。

VLingMem 实际上是把 human episodic memory 的 computational analog 实现了。Visual features 像 sensory memory（短暂、丰富、易退化），linguistic summary 像 episodic memory（持久、semantic、compact）。

这种 design choice 暗示一个 future direction：VLA model 的 memory 系统应该分层，short-term 用 visual，long-term 用 language，very-long-term 可能用 structured graph。

### 8.3 RL 在 Navigation 的特殊性

Navigation RL 的核心困难：long-horizon + sparse reward。一个 episode 几百步，只有终点有 reward，中间 step 的 credit 怎么分配？

VLingNav 的 hybrid rollout 给了一个实用答案：expert 介入提供 dense supervised signal。这相当于把 RL 问题转化为"在 expert anchor 之间做 short-horizon RL"，credit assignment 难度大幅降低。

这个 insight 可以推广到其他 long-horizon RL task——manipulation、autonomous driving、game playing。Expert guidance 不只是 imitation learning，可以理解为"为 RL 提供 curriculum"。

### 8.4 Multi-task Synergy 的本质

ObjectNav、EVT、ImageNav 表面是不同 task，但共享 underlying cognitive primitives：
- Visual grounding（识别目标）
- Spatial exploration（搜索未见区域）
- Trajectory planning（移动到目标）
- Memory maintenance（避免重复）

Single-task training 容易 overfit 到 task-specific pattern（比如 ObjectNav 学会"看到目标就 stop"，但 EVT 需要"看到目标 continue tracking"）。Multi-task training force model 学到更 abstract 的 primitive（"目标在视野中" vs "目标在视野中且需要接近" vs "目标在视野中且需要 maintain distance"）。

这种 abstract primitive 更 generalizable，能 transfer 到 unseen task composition。

### 8.5 Sim-to-Real 成功的关键

很多 VLA model 在 simulation 表现好但 real-world 失败。VLingNav zero-shot transfer 成功的核心原因：

1. **Linguistic memory 是 modality-agnostic 的**。"我已经查过 room A" 这个 fact 在 simulation 和 real-world 都成立，不受 visual appearance 差异影响
2. **Adaptive CoT 提供 noise robustness**。Real-world visual 有 noise（lighting、viewpoint、camera intrinsic 变化），model 遇到 ambiguous 场景 trigger reasoning 重新分析，而非 reactive 错误 action
3. **Open-world video co-training**。1.6M open-world video 提供 visual diversity，缩小 sim-real distribution gap

这三个因素组合起来，比单纯"用更多 simulation data 训练"有效得多。

### 8.6 与 π0.5 的对比

π0.5（Physical Intelligence）也是 VLA model with reasoning，但 focus on manipulation。VLingNav 把类似思想 extend 到 navigation：

- π0.5 用 text reasoning 做 task decomposition
- VLingNav 用 adaptive CoT 做 spatial decision-making
- π0.5 用 flow matching action
- VLingNav 用 Gaussian action head（更 lightweight）
- π0.5 没有 explicit memory（manipulation 不太需要）
- VLingNav 有 VLingMem（navigation 必须）

两者都体现一个趋势：VLA model 从 reactive mapping 进化到 cognitive agent。Future work 可能是 unified VLA 同时处理 manipulation + navigation + interaction。

Reference: π0.5
https://arxiv.org/abs/2504.16054

## 9 这篇 paper 的最大贡献

我个人觉得这篇 paper 最有价值的 contribution 是这个 conceptual insight：

**Embodied navigation 是 cognitive resource allocation 问题，perception-to-action mapping 问题**。

具体体现：
1. AdaCoT 让 model 学会 "when to think"
2. VLingMem 让 model 学会 "what to remember"
3. Hybrid RL 让 model 学会 "how to explore beyond expert"

这三个 capability 组合起来，让 VLA model 从 "reactive system" 进化为 "cognitive agent"。

更深层：这篇 paper 暗示 AGI 路径可能不在 "更大的 model + 更多 data"，而在 "让 model 学会 cognitive science 揭示的人类思维结构"。Adaptive thinking、episodic memory、dual-process theory 这些 cognitive science concept 可能是 next-generation AI architecture 的 design principle。

## 10 Limitation 和未来方向

作者承认三个 limitation：

1. **Monocular FOV**：单目视野有限，real-world 容易 miss 侧面 obstacle。Future：multi-view input（like NavFoM）

2. **Single-system architecture**：当前 prediction frequency 受限，dynamic environment 反应慢。Future：dual-system structure（like StreamVLN 的 slow-fast context）

3. **MPC locomotion**：当前用 waypoint controller，movement speed 和 reachable area 受限。Future：integrate learned locomotion policy（like Miki et al. 的 perceptive locomotion）

我个人的 additional thoughts：

4. **Memory 的 structure**：当前 linguistic memory 是 unstructured text。Future 可以 explore structured memory（graph、tree、key-value），更 efficient retrieval

5. **CoT 的 verification**：当前 CoT 是 model 自生成，没有 verification 机制。Future 可以加入 environment feedback verification（"我说前面是 room A，但 visual 确认吗？"）

6. **Reasoning 的 compositionality**：当前 CoT 是 linear 的。Future 可以 explore tree-structured reasoning（"如果往左走会怎样？如果往右走会怎样？"），更像 human deliberation

## 11 相关工作的 web links

**核心参考论文：**
- NaVId: https://navid.github.io/
- Uni-NaVid: https://github.com/MarkWuQ/Uni-NaVid
- TrackVLA: https://arxiv.org/abs/2505.23189
- NavFoM: https://arxiv.org/abs/2509.12129
- OctoNav: https://arxiv.org/abs/2506.09839
- Nav-R1: https://arxiv.org/abs/2509.10884

**RL 和 reasoning 相关：**
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- SimpleVLA-RL: https://arxiv.org/abs/2509.09674
- ReinFlow: https://arxiv.org/abs/2505.22094
- REINFORCE++: https://arxiv.org/abs/2501.03262
- PPO: https://arxiv.org/abs/1707.06347

**Cognitive science 启发：**
- Dual-process theory: https://en.wikipedia.org/wiki/Dual_process_theory
- Ebbinghaus forgetting curve: https://en.wikipedia.org/wiki/Forgetting_curve
- RoPE: https://arxiv.org/abs/2104.09864

**Foundation models：**
- LLaVA-Video: https://arxiv.org/abs/2410.02713
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- SigLIP: https://arxiv.org/abs/2303.15343

**项目主页：**
- VLingNav: https://wsakobe.github.io/VLingNav-web/

## 12 最后

VLingNav 这篇 paper 对我的最大启发：AI 的下一步可能不在单纯 scaling，而在让 model 学会人类思维的 structure。Adaptive thinking、episodic memory、dual-process theory 这些 cognitive science concept 给了 VLA architecture design 一个 new axis。

当 model 学会"什么时候该想、想什么、记住什么"，它就从 reactive system 进化为 cognitive agent。这个进化路径可能比"更大的 transformer + 更多 data"更接近 AGI。

---

# VLingNav 论文深度解析

## 1 论文的核心问题与动机

Embodied navigation 领域当前 VLA (Vision-Language-Action) models 存在三个根本性缺陷：

**第一个问题**：现有 VLA models 都是 reactive systems，从 observation 直接映射到 action，缺乏 explicit reasoning 机制。模型 inference budget 固定，遇到 ambiguity 无法增加 deliberation。

**第二个问题**：缺乏 persistent semantic memory。现有模型依靠 visual features 隐式维护历史，但随着 context window 增长，visual features 反复压缩导致 semantic 信息退化。

**第三个问题**：现有 VLA training 依赖 pure SFT (imitation learning)，导致 covariate shift 和 causal confusion 问题，模型无法超越 expert demonstrations。

VLingNav 从 cognitive science 视角切入，借鉴人类 dual-process theory (Kahneman 的 System 1/System 2)，引入 adaptive reasoning + linguistic memory + RL post-training 三位一体的解决方案。

Reference: Kahneman dual-process theory https://en.wikipedia.org/wiki/Dual_process_theory

## 2 VLingNav 整体架构解析

### 2.1 Architecture Overview

VLingNav 基于 LLaVA-Video-7B 作为 backbone，集成一个 MLP-based action model 实现文本 token generation 和 trajectory planning 的同步输出。

输入序列的构造：
$$\text{Input} = [\mathbf{E}_t^V, \mathbf{E}^T, \mathbf{E}^I, \mathbf{E}^M]$$

其中：
- $\mathbf{E}_t^V$：当前 frame 经过 vision encoder + projector 后的 visual tokens
- $\mathbf{E}^T$：temporal-aware indicator tokens（通过 RoPE 编码时间间隔）
- $\mathbf{E}^I$：instruction tokens
- $\mathbf{E}^M$：linguistic memory tokens（来自 VLingMem）

输出流程：
1. VLM backbone 先 autoregressive 预测 CoT indicator token (`<think_on>` 或 `<think_off>`)
2. 如果 `<think_on>`，继续生成 reasoning content + `<summary>` 内容
3. 最后一个 token 的 hidden state $\mathbf{h}_t^{pred}$ 输入 action model 生成 trajectory

这种设计的 intuition：模型有一个"meta-cognitive"开关，先决定是否需要思考，再决定思考什么。这与人类面对简单 vs 复杂决策时的行为模式高度一致。

### 2.2 Dynamic FPS Sampling Strategy

这是解决 video-based VLA 在 online inference 时 visual tokens 数量爆炸的关键创新。

公式 (1)：
$$f_s(i) = f_s^{max} \cdot e^{-\frac{\Delta T}{s}}$$

变量含义：
- $f_s(i)$：frame $i$ 的 sampling rate
- $f_s^{max}$：maximum sampling rate（最新 frame 的采样率）
- $\Delta T = t - i$：当前 frame $t$ 到历史 frame $i$ 的时间间隔
- $s$：memory stability 参数，控制 forgetting curve 的衰减速度

这个公式灵感来自 Ebbinghaus forgetting curve（https://en.wikipedia.org/wiki/Forgetting_curve），核心思想是 recent frames 采样率高（short-term memory），old frames 采样率低（long-term memory）。

与现有方法对比：
- Uni-NaVid 用 token merging 压缩历史，会 distort semantic features
- NaVILA/StreamVLN 用 uniform sampling，低 sampling rate 会导致 short-term 信息丢失

### 2.3 Grid Pooling Strategy

公式 (2)(3)：
$$g(i) = \lfloor e^{-\frac{\Delta T}{g}} \rfloor$$
$$\mathbf{V}_{t_i}' = \mathcal{G}(\mathbf{V}_{t_i}, g(i))$$

变量含义：
- $g(i)$：frame $i$ 的 grid pooling stride
- $\mathcal{G}(\cdot)$：grid pooling 操作
- $g$：pooling stability 参数

intuition：老的 frame 用更大 stride 做 spatial pooling，因为 coarse-grained semantic 对 long-term memory 足够；新 frame 保留 fine-grained spatial detail 用于 immediate decision。

### 2.4 Temporal-aware Indicator Token

公式 (4)：
$$E^T(\Delta T) = E_{base}^T + \text{RoPE}(\Delta T)$$

变量含义：
- $E^T(\Delta T)$：时间间隔 $\Delta T$ 对应的 indicator token
- $E_{base}^T$：base temporal embedding
- $\text{RoPE}(\Delta T)$：用 Rotary Position Embedding 编码时间间隔

Reference: RoPE paper https://arxiv.org/abs/2104.09864

这个设计解决了 dynamic FPS sampling 带来的 temporal inconsistency 问题——不同 frame 之间时间间隔不均匀，需要让模型感知这种不均匀性。

## 3 Adaptive Chain-of-Thought (AdaCoT)

### 3.1 核心机制

AdaCoT 的关键创新在于让模型自主决定 "when to think"，而非固定频率的 CoT。

模型首先预测 indicator token：
- `<think_on>`：进入 slow thinking mode，生成 reasoning + summary
- `<think_off>`：进入 fast thinking mode，直接输出 action

当 `<think_on>` 时，输出包含两部分：
1. **Reasoning content** (``)：包含 visual perception、task decomposition、location revisit check、next action determination
2. **Environmental summary** (`<summary>...</summary>`)：作为 linguistic memory 注入后续输入

### 3.2 AdaCoT vs 其他 CoT 策略

从 Table 6 的 ablation 可以看出：

| CoT Strategy | ObjNav SR | ObjNav SPL | Track SR | Track TR | Track CR | ImageNav SR | ImageNav SPL | $r_{CoT}$ (%) |
|---|---|---|---|---|---|---|---|---|
| w/o CoT | 36.2 | 16.5 | 62.7 | 68.5 | 6.28 | 56.3 | 27.3 | 0.0 |
| Dense CoT (Per-step) | 25.3 | 13.0 | 59.8 | 70.1 | 26.3 | 19.6 | 13.2 | 100.0 |
| Fixed Interval (k=5) | 42.5 | 23.5 | 68.5 | 74.2 | 9.18 | 48.2 | 28.7 | 20.0 |
| Fixed Interval (k=20) | 39.7 | 19.4 | 66.2 | 70.8 | 11.9 | 51.3 | 31.2 | 5.0 |
| **Adaptive CoT (Ours)** | **50.1** | **24.6** | **67.6** | **73.5** | **5.51** | **60.8** | **37.4** | **2.1** |

关键 insights：
1. Dense CoT (per-step) 性能反而最差（ImageNav SR 仅 19.6），因为过度 reasoning 干扰了 action quality，CR 飙升到 26.3
2. Adaptive CoT 仅在 2.1% 的 steps 触发 reasoning，却获得最佳性能——说明绝大多数 navigation steps 是 routine 的（如直线行走），只有 critical decision points 需要 deliberation
3. Fixed Interval 策略即使频率很低（k=20，5%）也不如 Adaptive，因为固定频率无法捕捉 environment 的实际 complexity 变化

intuition：这就像人类走路——大多数时候是自动的（System 1），只有在路口、障碍物、目标识别时才会停下来思考（System 2）。AdaCoT 让模型学会识别这些 critical moments。

## 4 Visual-Assisted Linguistic Memory (VLingMem)

### 4.1 设计哲学

作者对比了四种 memory 方案：
1. **Latent memory** (RoboFlamingo)：用 LSTM 传播 latent tokens，coarse-grained
2. **Visual-only memory** (video-based VLA)：implicit visual features 反复压缩
3. **Map-based memory** (MapNav, Mem2Ego)：VLM backbone 不原生支持 map input
4. **Linguistic memory** (VLingMem)：用 language summary 作为 cross-modal semantic memory

为什么 linguistic memory 最好？因为：
- Language 是 VLM 的 native modality，经过大规模预训练对齐
- Linguistic representation 对 information decay 更 robust（"我已经检查过这个房间"比 compressed visual features 更持久）
- Linguistic memory 可以与 AdaCoT 形成 synergy——即使不触发 CoT，summary 仍提供历史 context

### 4.2 Memory Update Mechanism

从 Algorithm 1 可以看到 memory 更新逻辑：
```
if E^CoT = <think_on> then
    c ← LLM.generate(...)  # 生成 reasoning + summary
    M ← UpdateMemory(M, c_t)  # 用 summary 更新 memory
end if
```

关键点：memory 只在 `<think_on>` 时更新。这意味着 memory 是 episodic 的，记录 critical decision points 的环境 summary，而非每个 step 的琐碎信息。

### 4.3 VLingMem Ablation

Table 7：

| Memory Mode | ObjNav SR | ObjNav SPL | Track SR | Track TR | Track CR | ImageNav SR | ImageNav SPL |
|---|---|---|---|---|---|---|---|
| w/o Memory | 15.4 | 3.5 | 37.5 | 59.1 | 1.90 | 21.0 | 3.7 |
| Visual-only | 45.2 | 20.3 | 66.8 | 70.6 | 7.85 | 57.9 | 33.7 |
| Language-only | 18.8 | 4.4 | 40.2 | 55.2 | 3.25 | 23.3 | 7.5 |
| **VLingMem (Ours)** | **50.1** | **24.6** | **67.6** | **73.5** | **5.51** | **60.8** | **37.4** |

关键 insights：
1. w/o Memory 性能灾难性下降（ObjNav SR 从 50.1 到 15.4），证明 memory 对 long-horizon navigation 是 essential 的
2. Visual-only memory 比 Language-only 好得多——这说明 visual features 携带的信息量大于 language summary
3. VLingMem (visual + linguistic) 显著优于单一 modality——visual 提供 perceptual detail，linguistic 提供 semantic abstraction，两者 complementary

## 5 Nav-AdaCoT-2.9M 数据集

### 5.1 数据集规模对比

Table 1 展示了与现有 dataset 的对比：

| Dataset | $N_{scene}$ | ObjNav | Track | ImageNav | Modality | $N_{step}$ | $N_{cot}$ | Action |
|---|---|---|---|---|---|---|---|---|
| OctoNav-Bench | 438 | √ | × | √ | V, L | 45K | 10K | Des. |
| Nav-CoT-110K | 342 | √ | × | × | V, L | 110K | 110K | Des. |
| **Nav-AdaCoT-2.9M (Ours)** | **718** | **√** | **√** | **√** | **V, L** | **2.9M** | **472K** | **Traj.** |

Nav-AdaCoT-2.9M 的优势：
- 第一个集成三种 navigation task 的 dataset
- 最大规模的 CoT annotations（472K）
- 使用 trajectory-based annotation（比 discrete action 更 fine-grained）

### 5.2 Adaptive CoT Labeling Pipeline

使用 Qwen2.5-VL-72B 生成 CoT labels，composite prompt 包含 5 个 components：
1. Navigation instructions
2. Egocentric visual stream（最近 10 frames）
3. Prior memory content
4. Expert trajectories at each step
5. Explicit formatting requirements

生成 472K CoT responses from 2.9M samples（约 16.3% 的 steps 触发 CoT）。

两阶段 filtering：
1. Rule-based checks：丢弃 incomplete 或 logically inconsistent 的 responses
2. Quality verification：与 expert trajectories 交叉验证

### 5.3 Open-World Video Co-training

额外引入 1.6M open-world video samples：
- LLaVA-Video-178K：non-CoT subset
- Video-R1：CoT-annotated subset（challenging video QA pairs）
- ScanQA：non-CoT subset

这种 categorization 让模型学会根据 input difficulty 自主决定是否 reasoning。

Table 8 显示 co-training 的效果：

| Training Data | ObjNav SR | Track SR | ImageNav SR |
|---|---|---|---|
| w/o Co-training | 43.1 | 66.5 | 50.2 |
| w/ Co-training | 50.1 | 67.6 | 60.8 |

ImageNav 提升 +10.6 SR，说明 open-world video data 显著增强了 cross-modal grounding 能力。

## 6 Training Recipe 三阶段

### 6.1 Stage 1: Pre-train

在 open-world adaptive CoT video dataset 上训练 1 epoch，赋予模型 adaptive visual reasoning 的基础能力。Loss 为 standard cross-entropy loss，只有 visual encoder frozen。

### 6.2 Stage 2: Supervised Fine-Tuning

混合 embodied navigation data + open-world video data，co-training 20K steps，batch size 512。

公式 (6)：
$$\min_\theta \mathcal{L}_{SFT}(\theta) = \alpha \mathcal{L}_{MSE}(\hat{\tau}_t, \tau_t^{gt}) + (1-\alpha) \mathcal{L}_{CE}(E_t^{pred}, E_t^{gt})$$

变量含义：
- $\mathcal{L}_{MSE}$：监督 action trajectory 的 Mean Squared Error loss
- $\hat{\tau}_t$：predicted trajectory
- $\tau_t^{gt}$：ground-truth trajectory
- $\mathcal{L}_{CE}$：监督所有 textual outputs（CoT + VQA）的 Cross-Entropy loss
- $E_t^{pred}$：predicted text tokens
- $E_t^{gt}$：ground-truth text tokens
- $\alpha$：balance hyperparameter（设为 0.5）

### 6.3 Stage 3: Online Expert-guided Post-training

这是超越 pure imitation learning 的关键阶段。

#### 6.3.1 Probabilistic Continuous Action Model

公式 (7)：
$$\pi_\theta(\mathbf{a}_t | \mathbf{s}_t) = \mathcal{N}\left(\boldsymbol{\mu}_\theta(\mathbf{h}_t), \text{diag}\left(\boldsymbol{\sigma}_\theta(\mathbf{h}_t)^2\right)\right)$$

变量含义：
- $\pi_\theta(\mathbf{a}_t | \mathbf{s}_t)$：policy，给定 state $\mathbf{s}_t$ 时 action $\mathbf{a}_t$ 的概率分布
- $\mathbf{h}_t$：VLM backbone 提取的 visual-linguistic features
- $\boldsymbol{\mu}_\theta(\mathbf{h}_t)$：Gaussian 分布的 mean
- $\boldsymbol{\sigma}_\theta(\mathbf{h}_t)$：Gaussian 分布的 standard deviation
- $\text{diag}(\cdot)$：对角协方差矩阵

设计 intuition：
- Rollout 时从分布中 sample：$\mathbf{a}_t \sim \pi_\theta(\cdot | \mathbf{s}_t)$，实现 stochastic exploration
- Validation 时用 mean：$\mathbf{a}_t = \mu_\theta(\mathbf{h}_t)$，实现 deterministic execution
- 相比 discrete tokenization（sacrifices precision）和 diffusion/flow matching（high computational cost），multivariate Gaussian 实现了 precision 和 efficiency 的平衡

#### 6.3.2 Hybrid Rollout Strategy

两种 rollout 模式交替进行：

**Naive rollout**：当前 policy $\pi$ 独立与环境交互，只保留 successful trajectories 加入 hybrid buffer。提供 on-policy positive examples。

**Expert-guided rollout**：当 agent 触发 irrational condition（oscillating 或 stuck for k=15 steps）或最终失败时，expert policy $\pi^*$（Shortest Path planner）接管，提供 corrective trajectories。

这种设计的关键 insight：pure RL 在 sparse rewards + long horizons 下 sample-inefficient；pure imitation learning 有 covariate shift。Hybrid approach 用 expert demonstrations 稳定学习，同时用 on-policy data 探索更好策略。

#### 6.3.3 Augmented Loss

公式 (8)：
$$\min_\theta \mathcal{L}_{post}(\theta) = \lambda \mathcal{L}_{RL}(\theta) + (1-\lambda) \mathcal{L}_{SFT}(\theta)$$

其中：
$$\mathcal{L}_{RL}(\theta) = -\mathbb{E}_t\left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

变量含义：
- $\mathcal{L}_{RL}$：PPO-style policy-gradient objective
- $r_t(\theta)$：importance sampling ratio
- $A_t$：advantage，用 REINFORCE++ 计算
- $\epsilon$：PPO clip parameter
- $\mathcal{L}_{SFT}$：公式 (6) 定义的 imitation loss
- $\lambda$：balance hyperparameter（设为 0.01）

$\lambda = 0.01$ 说明 SFT loss 仍占主导，RL loss 作为 fine-grained adjustment——这避免了 RL 训练的不稳定性，同时允许模型探索 expert 之外的策略。

Reference: REINFORCE++ https://arxiv.org/abs/2501.03262
Reference: PPO https://arxiv.org/abs/1707.06347

## 7 实验结果深度分析

### 7.1 Object Goal Navigation

Table 2 (HM3Dv1, HM3Dv2, MP3D)：

| Method | HM3Dv1 SR | HM3Dv1 SPL | HM3Dv2 SR | HM3Dv2 SPL | MP3D SR | MP3D SPL |
|---|---|---|---|---|---|---|
| Uni-NaVid | 73.7 | 37.1 | - | - | - | - |
| CogNav | 72.5 | 26.2 | - | - | 46.6 | 16.1 |
| ApexNav | 59.6 | 33.0 | 76.2 | 38.0 | 39.2 | 17.8 |
| VLingNav (SFT) | 70.6 | 38.2 | 76.4 | 32.6 | 47.4 | 25.8 |
| **VLingNav** | **79.1** | **42.9** | **83.0** | **40.5** | **58.9** | **26.5** |

关键 insights：
1. VLingNav vs Uni-NaVid：HM3Dv1 提升 +5.4 SR (+7.3%) 和 +3.9 SPL (+15.6%)
2. MP3D 上对 CogNav 提升 +12.3 SR (+26.4%) 和 +10.4 SPL (+32.8%)——MP3D 是 long-range exploration 主导的场景，这个巨大提升直接验证了 VLingMem 的 long-horizon memory 能力
3. SFT → RL post-training 提升：HM3Dv1 从 70.6 到 79.1 SR（+8.5），MP3D 从 47.4 到 58.9 SR（+11.5）——RL post-training 在 long-range 场景效果更显著

Table 3 (HM3D OVON - open vocabulary)：

| Method | Val Seen SR | Val Seen Syn SR | Val Unseen SR |
|---|---|---|---|
| Uni-NaVid | 41.3 | 43.9 | 39.5 |
| Nav-R1 | 58.4 | 48.1 | 42.2 |
| VLingNav (SFT) | 45.9 | 44.8 | 41.5 |
| **VLingNav** | **59.3** | **56.8** | **50.1** |

VLingNav 在 Val Seen Synonyms 上对 Nav-R1 提升 +8.7 SR (+18.1%)，证明 adaptive CoT + linguistic memory 显著增强了 synonym generalization。

### 7.2 Embodied Visual Tracking

Table 4 (EVT-Bench)：

| Method | Single Target SR | Single Target TR | Single Target CR | Distracted SR | Distracted TR | Distracted CR |
|---|---|---|---|---|---|---|
| TrackVLA | 85.1 | 78.6 | 1.65 | 57.6 | 63.2 | 5.80 |
| NavFoM | 86.0 | 80.5 | - | 61.4 | 68.2 | - |
| TrackVLA++ | 86.0 | 81.0 | 2.10 | 66.5 | 68.8 | 4.71 |
| VLingNav (SFT) | 87.2 | 78.9 | 1.23 | 66.1 | 69.7 | 4.78 |
| **VLingNav** | **88.4** | **81.2** | **2.07** | **67.6** | **73.5** | **5.51** |

Distracted Tracking 是最 challenging 的 setting，VLingNav 对 TrackVLA++ 提升 +1.1 SR (+1.7%) 和 +4.7 TR (+6.8%)。这个提升主要来自 AdaCoT 在 occlusion 后的 re-identification 能力——当目标被遮挡后重新出现，model 触发 `<think_on>` 重新分析环境。

### 7.3 Image Goal Navigation

Table 5 (HM3D Instance ImageNav)：

| Method | SR | SPL |
|---|---|---|
| UniGoal | 60.2 | 23.7 |
| VLingNav (SFT) | 51.1 | 32.6 |
| **VLingNav** | **60.8** | **37.4** |

VLingNav 对 UniGoal 在 SR 上略高 (+0.6)，但 SPL 大幅提升 +13.7 (+57.8%)。UniGoal 用 LightGlue keypoint matching 作为额外 criterion，而 VLingNav 完全依靠 implicit reasoning。

SPL 的巨大提升说明 VLingNav 不仅找到目标，还走更 direct path——这是 linguistic memory 防止 redundant exploration 的直接证据。

### 7.4 Online Post-training Ablation

Figure 11 的 ablation 比较三种 rollout：
- **Naive Rollout**：性能无提升，因为 sparse reward + long horizon 使 value estimation 太困难
- **Expert Rollout**（DAgger-like）：有提升，但仍不及 Hybrid
- **Hybrid Rollout**：最佳，combines exploration (naive) 和 stabilization (expert)

这验证了 RL post-training 中 expert knowledge 的关键作用——pure RL 在 navigation 这种 long-horizon + sparse reward 场景下无法独立 work。

### 7.5 Multi-task Synergy

Figure 12 显示 multi-task training 的 emergent behavior：
- Single-task models 在各自 specialized benchmark 上都不如 multi-task model
- Multi-task training foster cross-domain 和 cross-task capabilities

这说明 ObjectNav、EVT、ImageNav 共享 underlying navigation priors，multi-task training 让模型学到 universal cognitive structures。

### 7.6 SFT Training Steps Analysis

Figure 10 显示 training steps 与性能的关系：
- 性能随 steps 正向 scaling
- Excessive training 导致 diminishing returns 和 eventual degradation（overfitting on simulation data）
- 1 epoch ≈ 10K training steps

## 8 Real-World Experiments

### 8.1 Hardware Setup

- Robot: Unitree Go2 quadruped
- Camera: Intel RealSense D457（RGB 1280×800，HFOV 90°）
- Compute: Remote server with NVIDIA RTX 4090
- Communication: Wi-Fi，image compression before transmission
- Control: NMPC (Nonlinear Model Predictive Control) for trajectory tracking

### 8.2 Inference Efficiency

VLingNav 在 500 video frames 下维持 <300ms inference latency，加上 ~100ms communication overhead，实现 ~2.5 FPS effective inference speed。

关键优化：
- 只 encode 最新 frame（历史 visual tokens 已 cached）
- Dynamic FPS sampling 控制输入 token 数量
- Grid pooling 压缩历史 frame 的 spatial resolution

### 8.3 Real-World Results (Figure 8)

在 home/office/outdoor 三种场景下，对每个 target 进行 10 trials：

**ObjectNav**：VLingNav 在所有场景显著超越 Uni-NaVid
**EVT**：在 distracted setting（occlusion + distractors）优势最大
**ImageNav**：在所有 category 超越 UniGoal

### 8.4 Zero-Shot Sim-to-Real Transfer

Real-world 用的是与 simulation 完全相同的 model weights，no fine-tuning on real-world data。这证明：
1. Multi-task training 学到的 representations 是 generalizable 的
2. Adaptive CoT + linguistic memory 提供 robust 的 cognitive scaffold
3. Open-world video co-training 缩小 sim-to-real gap

### 8.5 Emergent Cross-Task Capabilities

Figure 9 展示了 zero-shot 的 cross-task compositionality：
1. Search language-described target → switch to tracking
2. Search image-goal target → track after locating

这种 compositionality 来自 VLA model 的 unified architecture + multi-task co-training——模型学到 common navigation priors 可以跨 task transfer。

### 8.6 Cross-Domain Generalization

虽然只训练 tracking humans，VLingNav 能可靠 tracking dynamic non-human targets；能 navigate 到 OOD objectives（color-specified, spatially constrained, detailed description）。这表明 multi-task learning + general visual understanding data 大幅增强 cross-domain generalization。

## 9 Discussion 核心洞察

### 9.1 Adaptive Thinking 的认知科学基础

Dual-process theory 启发 AdaCoT：
- **System 1 (fast)**：`<think_off>` mode，traverse straight corridor 等 routine 场景，real-time fluid navigation
- **System 2 (slow)**：`<think_on>` mode，intersection direction choice, occluded object search 等 critical decision points

平均仅 2.1% steps 触发 slow thinking 就能显著提升 overall success——证明 deliberate thought 只需在 key steps 就足够。这对 resource-constrained robot platforms 至关重要。

### 9.2 Linguistic Memory 的 Synergy

VLingMem 的两大优势：
1. **Robustness against information decay**：相比 compressed visual features，linguistic summary 更持久（"我已经检查过这个房间"不会随时间衰减）
2. **Synergy with AdaCoT**：即使 `<think_off>`，persistent linguistic memory 仍提供 historical context，确保 coherent decision-making

这种 synergy 是 long-horizon + complex environment 下 robustness 的关键。

### 9.3 Beyond Imitation Learning

Pure SFT 受限于 expert data 的 quality 和 coverage，容易产生 causal confusion 和 covariate shift。Online expert-guided RL 让模型从 "imitator" 变成 "problem solver"：
- Autonomous exploration 发现 expert 之外的更优策略
- Expert knowledge 稳定学习，防止 catastrophic forgetting
- Direct rewards from prior expert policy 提高 sample efficiency

### 9.4 Generality 和 Real-World Generalization

VLingNav 用 single unified model weights 在所有 task 上达到 SOTA，no task-specific fine-tuning。这证明 approach 成功 capture 了 embodied navigation 的 universal cognitive structures。

Zero-shot real-world transfer 证明：通过 high-quality simulation training + powerful cognitive architecture，模型学到的是 space/language/action 的 generalizable representations，而非 simulator-specific patterns。

## 10 Limitations 和 Future Work

1. **Monocular FOV limitation**：单目 egocentric 视角 FOV 受限，计划 integrate multi-view observations（following NavFoM [74]）

2. **Single-system architecture**：当前 prediction frequency 受限，impede rapid decision-making in highly dynamic environments。计划升级为 dual-system structure 支持 high-frequency action outputs

3. **Locomotion model**：当前用 MPC-based waypoint controller，缺乏 flexible locomotion model。计划 integrate locomotion capabilities（following Miki et al. [35]）提高 movement speed 和 reachable areas

## 11 个人思考与 Intuition Building

### 11.1 为什么 Adaptive CoT 比 Fixed Interval CoT 好？

从 information theory 角度：navigation 场景的 entropy 是 non-uniform 分布的。Straight corridor 的 action entropy 接近 0（continue forward），intersection 的 action entropy 高（left/right/forward）。Adaptive CoT 让模型把 "cognitive computation" 分配到 high-entropy decision points，实现 optimal resource allocation。

Fixed interval 无论 entropy 如何都花费 same computation，要么 over-thinking routine steps（wasting computation），要么 under-thinking critical points（missing key reasoning）。

### 11.2 为什么 Linguistic Memory 比 Visual Memory 好？

从 representation compression 角度：visual features 是 high-dimensional dense representation，即使经过 grid pooling 仍占用大量 tokens。Linguistic summary 是 ultra-compressed semantic representation——"房间 A 有桌子但无目标物体" 比 visual features 更 compact 且语义明确。

从 VLM backbone 角度：language 是 VLM 的 native modality，经过 trillions tokens 预训练对齐。Linguistic memory 直接利用这种 pre-trained alignment，而 visual memory 需要 visual encoder 重新 encode。

### 11.3 为什么 Hybrid Rollout 比 Pure RL 好？

从 credit assignment 角度：navigation 是 long-horizon task，一个 episode 可能 100+ steps。Pure RL 面临严重的 credit assignment 问题——很难判断哪一步导致 success/failure。

Expert-guided rollout 提供 dense supervised signal：当 agent stuck 时立即获得 expert correction，相当于把 long-horizon credit assignment 分解为多个 short-horizon supervised learning problems。

### 11.4 Multi-task Synergy 的根本原因

ObjectNav、EVT、ImageNav 共享相同的 underlying skills：
- Visual perception（识别目标）
- Spatial exploration（搜索未见区域）
- Trajectory planning（移动到目标）
- Memory maintenance（避免重复探索）

Multi-task training 让模型学到这些 shared skills 的 universal representations，而 single-task training 容易 overfit 到 task-specific patterns。

## 12 相关工作与 Reference Links

### 12.1 Embodied Navigation VLA Models
- NaVId: https://navid.github.io/
- Uni-NaVid: https://github.com/MarkWuQ/Uni-NaVid
- NaVILA: https://navila-lgbot.github.io/
- StreamVLN: https://arxiv.org/abs/2507.05240
- TrackVLA: https://arxiv.org/abs/2505.23189
- NavFoM: https://arxiv.org/abs/2509.12129

### 12.2 Embodied Chain-of-Thought
- Embodied-CoT: https://embodiedcot.github.io/
- CoT-VLA: https://cot-vla.github.io/
- OctoNav: https://arxiv.org/abs/2506.09839

### 12.3 Memory in VLA Models
- MemoryVLA: https://arxiv.org/abs/2508.19236
- MapNav: https://arxiv.org/abs/2502.13781
- Mem2Ego: https://arxiv.org/abs/2502.14254

### 12.4 RL Post-training for VLA
- SimpleVLA-RL: https://arxiv.org/abs/2509.09674
- ReinFlow: https://arxiv.org/abs/2505.22094
- VLN-R1: https://arxiv.org/abs/2506.17221
- Nav-R1: https://arxiv.org/abs/2509.10884
- DeepSeek-R1: https://arxiv.org/abs/2501.12948

### 12.5 Foundation Models
- LLaVA-Video: https://arxiv.org/abs/2410.02713
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- SigLIP: https://arxiv.org/abs/2303.15343
- RoPE (RoFormer): https://arxiv.org/abs/2104.09864

### 12.6 Benchmarks
- HM3D: https://arxiv.org/abs/2109.08238
- MP3D: https://arxiv.org/abs/1707.06834
- HM3D OVON: https://arxiv.org/abs/2405.05535
- EVT-Bench: https://arxiv.org/abs/2505.23189
- HM3D Instance ImageNav: https://arxiv.org/abs/2211.15876

### 12.7 Project Page
- VLingNav 官方项目页: https://wsakobe.github.io/VLingNav-web/

## 13 总结

VLingNav 的核心贡献是从 cognitive science 视角重新思考 VLA architecture：

1. **Adaptive CoT** 实现了 dual-process theory 的计算化，模型学会 "when to think" 和 "what to think about"
2. **VLingMem** 用 linguistic representation 作为 persistent memory，克服 visual memory 的 information decay 问题
3. **Online expert-guided RL** 通过 hybrid rollout 突破 imitation learning 的局限
4. **Nav-AdaCoT-2.9M** 提供了最大规模的 adaptive CoT navigation dataset

这套设计在 simulation 和 real-world 都取得 SOTA，且展现 emergent cross-task 和 cross-domain generalization，证明 cognitive-inspired architecture 是 embodied AI 的 promising direction。

这篇论文的 deepest insight：embodied navigation 不只是 perception-to-action mapping 问题，更是 cognitive resource allocation 问题。将 cognitive science principles（adaptive thinking, episodic memory, dual-process theory）与 ML engineering（VLA, RL, data scaling）结合，是通往 generalist embodied agents 的 promising path。
