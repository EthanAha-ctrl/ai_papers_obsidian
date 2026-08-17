---
source_pdf: Wow, wo, val.pdf
paper_sha256: aef22be60a1ee060d8a74ff0c157f9b69090af532086b0dcd145a206684f70fc
processed_at: '2026-08-13T06:13:10-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇Paper

好，咱们抛开那些学术黑话，用最直白的话来说说这篇paper到底在干嘛。

## 一句话说清楚

**现在的video generation模型，生成视频看起来很牛，但你真让它去当机器人的"大脑"，让机器人按着视频去干活，基本全崩。** 这篇paper就是做了一个test，专门来戳破这个"看起来很美"的幻象。

## 为什么要有这个Benchmark

### 先说背景

现在大家都 hype 一个概念叫 **world model**——就是让 AI 在脑子里"想象"世界会怎么变化。如果机器人能准确预测"我抓起这个杯子，杯子会跟着动"，那它就能在脑子里先"彩排"一遍再行动，这比 trial-and-error 高效多了。

于是大家就想：**video generation models 不就是在"预测未来"吗？** Sora、Kling、Hailuo 这些模型能生成那么逼真的视频，那能不能直接拿它们当 robot 的 world model 来用？

**听起来很合理，但有几个问题没人真正回答：**

1. 这些模型生成的视频，人看着觉得"真"吗？还是只是 pixel 层面好看？
2. 这些视频里发生的事情，**物理上对吗**？robot 真能照着做吗？

### 现有Benchmark的问题

你去看现有的 video generation benchmark，比如 VBench、T2V-CompBench、Physics-IQ，它们主要看这些：

- **画面美不美**（FVD、PSNR、SSIM 这些 pixel 级指标）
- **是不是符合 text prompt**
- **有没有 basic physics violation**

但是 **robotic world model** 的需求完全不同。举个例子：

一个 video 生成出来，画面超清晰，color 很准，motion 很 smooth，FVD 分很高。但你仔细一看，robot arm 直接穿过了桌子去抓物体，或者物体莫名其妙 teleport 了一下。**这个视频在传统 benchmark 上可能得分很高，但对 robot 来说完全没用，甚至有害。**

这就是这篇 paper 要解决的问题：**搞一个专门针对 embodied world model 的 test，让它能真正反映 model 能不能当 robot 的"大脑"。**

## WoW-World-Eval 怎么设计的

### 五个核心能力的直觉

Paper 把一个合格的 embodied world model 需要的能力拆成五个维度，我用大白话讲讲每个维度在测什么：

**1. Perception（感知）**  
就是你给它一张图，它能不能看懂里面有什么。红的、方的、几个、多大，物体之间什么关系，哪个地方能抓。这是基础，看都看不懂，后面别玩了。

**2. Planning（规划）**  
给它一个复杂任务，比如"把面包放进盘子再放到抽屉里"，它能不能把这个任务拆成几个步骤，按正确顺序执行。这测的是它有没有"先想再做"的能力。

**3. Prediction（预测）**  
给它一个 initial state 和一个 action，它生成的未来视频里，物体有没有突然消失、突然出现、穿透桌面？重力对不对？碰撞反应对不对？这是测它的"内置物理引擎"靠不靠谱。

**4. Execution（执行）**  
这是最狠的一测。让 model 生成视频，然后把这个视频喂给一个 Inverse Dynamics Model，让 IDM 从视频里"反推"出 robot 应该做什么 action，然后**把这个 action 拿到真 robot 上执行**，看任务能不能成功。**这就是 Turing Test 的精髓：你说你生成的视频像真的？那我们看看真 robot 能不能照着干。**

**5. Generalization（泛化）**  
给它训练时没见过的场景——比如用 GPT 把训练场景的风格改一改，或者干脆用名画《戴珍珠耳环的少女》这种图当初始帧——看它还能不能正确执行任务。

### 数据怎么来的

609 个 robot manipulation 样本，来源混合：
- 公开数据集（RoboMIND、DROID）
- 他们自己 in-house 的 robot trajectories
- 用 GPT 生成的 OOD（out-of-distribution）样本

清洗流程是 **GPT-4o 粗筛 + 人工精筛**：
- GPT-4o 当"intelligent annotator"，给每个 video-instruction pair 打分，看它匹配哪个能力维度
- 人工再 verify 一遍，处理 edge case
- 标注 keypoint（robot arm 的 gripper、joints、manipulated object）

## 22 个 Metrics 讲人话

这块是 paper 最 technical 的部分，我挑几个最关键的讲。

### Video Quality 类（这个最 standard）

就是看视频像不像真的，包括：

- **FVD**：把 generated video 和 real video 都通过 I3D 网络提 feature，然后算两个 Gaussian 分布之间的 Fréchet distance。越小说明分布越接近。
- **PSNR**：pixel-level 的 MSE 取 log，越高说明 pixel 越接近。
- **SSIM**：看结构相似性，比 PSNR 更接近人眼感受。
- **DINO**：用 DINOv2 提 embedding 算 cosine similarity，看 semantic 层面像不像。
- **DreamSim**：CLIP + DINO 融合，用人类偏好数据 fine-tune 过的，更接近人眼判断。

**这里的问题是：这些指标高，不代表物理上对。** Paper 后面用实验证明了这点。

### Instruction Understanding 类（用 GPT-4o 当 judge）

三个分数：
- **Caption Score**：让 GPT 从 generated video 和 GT video 里各提取一个结构化描述（initial state、processing state、final state、action、object），然后比对像不像
- **Sequence Match Score**：指令里说"先拿 A 再放 B"，视频里是不是真的先动了 A 再动了 B
- **Execution Quality Score**：动作做到什么程度了。1 分=完全没动，5 分=完美完成

### Physical Consistency 类（这个最 novel）

这块 paper 做了很多创新。

**Mask-guided Regional Consistency**  
这个 idea 是：**一个视频里可能整体看着 OK，但 robot arm 在抖，或者 object 在闪烁**。传统的整体 metric 看不出来这种局部问题。

怎么做呢：
1. 用 GroundedSAM2 把 robot arm、object、background 分别 segment 出来
2. 用 DINOv3 提 patch-level feature
3. 对每个 region 分别算 temporal consistency（frame 之间 cosine similarity）
4. 这样就能 pinpoint 说"这个视频 background 稳定，object 稳定，但 robot arm 在第 30 帧突然变了"

公式核心是：

$$\mathrm{MRC}^r = \frac{1}{T-1} \sum_{t=2}^{T} \left[\frac{1}{2}\mathrm{Consist}^r(1,t) + \frac{1}{2}\mathrm{Consist}^r(t-1,t)\right]$$

其中 $\mathrm{Consist}^r(a,b)$ 就是 region r 在 frame a 和 frame b 的 feature 的 cosine similarity。$\frac{1}{2}$ 那部分是 combine long-range（frame 1 到 t）和 short-range（frame t-1 到 t）的一致性。

**Trajectory Consistency**  
用 SAM2 跟踪 robot end-effector 和 object 的 keypoint，得到轨迹，然后比对 generated trajectory 和 GT trajectory。

这里有个很 clever 的细节：**Camera Motion Correction**。generated video 里可能有 camera drift，导致你看到的 robot 运动其实是 robot 真实运动 + camera 漂移的混合。Paper 用 Lucas-Kanade optical flow + RANSAC 估计 camera 的 affine transform，然后从观测轨迹里减掉 camera motion：

$$\mathbf{p}_t^{\text{true}} = \hat{\mathbf{p}}_t - \mathbf{c}_t$$

其中 $\hat{\mathbf{p}}_t$ 是观测到的 end-effector 位置，$\mathbf{c}_t$ 是估计的 camera offset，相减得到"真实"的 robot 运动。这是从 SLAM 借来的技巧。

然后算三个距离：
- **L2norm**：逐帧位置差
- **DTW**：允许时间轴 warp 的轨迹相似度（你快我慢没关系，形状对就行）
- **Fréchet Distance**：最严格的，模拟一个人牵着狗走两条路，狗绳最短要多长

### Physical Common Sense（用 GRPO 训了一个 evaluator）

这个最 fancy。Paper 拿 Qwen-2.5-VL (7B) 当 base，用 GRPO 算法分两阶段 fine-tune：

**Stage 1**：先用 50,000 个 video QA 样本教它理解视频内容和物理常识。Reward 就是答对给 1 分，答错 0 分。准确率从 60.83% 提到 71.51%。

**Stage 2**：再用 1,297 个人工标注的评分数据，教它按 1-5 分给视频的四个维度（video quality、instruction following、physical consistency、planning logic）打分，让它对齐人类判断。Reward 是它打的分和人工分的差距的负数。

GRPO 的核心 formula：

$$\mathcal{J}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \left(\min\left[r_i(\theta)\hat{A}_i, \mathrm{clip}(r_i(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_i\right] - \beta D_{KL}[\pi_\theta \| \pi_{ref}]\right)\right]$$

其中 $r_i(\theta) = \pi_\theta(o_i|q)/\pi_{\theta_{old}}(o_i|q)$ 是 importance sampling ratio，$\hat{A}_i$ 是 group-relative advantage（在 G 个 sample 里标准化 reward），$\beta D_{KL}$ 是防止 policy 偏离 reference model 太远的正则项。

**直觉就是**：PPO 要训一个 value network 来估计 baseline，GRPO 直接用一组 sample 的平均 reward 当 baseline，省事。对每个 query 采 G=8 个 output，谁比平均好谁 advantage 正，谁比平均差谁 advantage 负。

### Overall Score 怎么算的

不同 metric 的 scale 不一样，FVD 可能上千，PSNR 才几十，怎么加权？

Paper 的做法是 **三步走**：

1. **Clip 到固定范围**：比如 PSNR clip 到 [0, 50]，FVD clip 到 [0, 2000]
2. **线性映射到 [0, 1]**：HIB（越高越好）的直接除，LIB（越低越好）的用 1 减
3. **再过一个 monotone transform**：有 gamma、logit-temperature、tanh-slope 几种选择，参数 $\theta_m$ 通过最大化 Pearson correlation with human ratings 来选

比如 FVD 用 $f(x) = x^{1.52}$，PSNR 用 tanh slope 4.71。**这是 data-driven 的 metric calibration，比拍脑袋定权重科学多了。**

## 两个 Turing Test

### Human Turing Test

简单粗暴：给 13 个人看一堆视频，有真的有生成的，让他们猜哪个是假的。统计每个 model 的视频骗过人的比例。

结果：**WoW-World-Eval 的 Overall Score 和 Deceive Human Ratio 的 Pearson correlation 是 0.93**。说明这个 benchmark 打分和人感觉是高度一致的。**其中 Video Quality (r=0.874) 和 Physical Law (r=0.753) 的相关性最高**——人最在意"看着像不像"和"物理对不对"。

### IDM Turing Test（最有意思）

这个 idea 真的很高明。**你说你生成的视频像真的？好，那我们让真 robot 试试能不能照着做。**

具体流程：
1. 拿一个在真实数据上训好的 Inverse Dynamics Model（GC-IDM）
2. 把 model 生成的视频喂给它
3. IDM 从视频里反推出 robot 应该执行什么 action
4. 把这个 action 发给真 robot 执行
5. 看任务成不成功

**这个 test 的逻辑是**：如果生成的视频物理上 OK，那 IDM 从中提取的 action 在真 robot 上应该能 work。如果视频有物理 violation（比如物体穿透、teleport），IDM 提取的 action 也会是错的，真 robot 执行就失败。

先验证 IDM 本身靠不靠谱：在 GT video 上 replay，GC-IDM 达到 90% success rate，证明 IDM 没问题。

然后看 model 生成的视频：

| Model | Real-world Succ. Rate |
|-------|----------------------|
| Kling | 9.88% |
| Hailuo | 2.47% |
| CogVideoX | 0.00% |
| Cosmos-Predict1 | 0.00% |
| Wan2.1 | 0.00% |
| Cosmos-Predict2 | 8.64% |
| WoW-wan | **40.74%** |
| WoW-cosmos2 | 18.52% |

**结论很扎心**：那些在传统 metric 上得分很高的 commercial model，在这个 test 上几乎全崩。Kling、Hailuo 这种看起来很牛的模型，真 robot 照着做基本做不成。**只有专门用 robot 数据训过的 WoW 系列还行，最好也才 40.74%。**

## 主要实验结论

我挑几个最重要的讲：

**1. Planning 是最大瓶颈**  
所有 model 在 Planning Reasoning 上的得分都低得可怜。最好的 Hailuo 才 17.27，CogVideoX 才 4.55。说明现在的 video model 在 long-horizon reasoning 上根本不行。给它一个多步任务，它能把单步动作生成出来，但把它串成一个合理的 plan，做不到。

**2. Visual Quality 好 ≠ Physical Executable**  
Kling 在 Video Quality 和 Physical Law 上都是顶级，但 IDM Turing Test 只有 9.88%。**看起来像和能用是两回事。**

**3. Dense Prompt 有用但有上限**  
用 InternVL3-78B 把短 prompt 扩展成包含环境、子目标、物理约束的 dense prompt，Video Quality 和 Instruction Understanding 都涨了。但 **Planning 几乎没涨**。说明光靠 prompt engineering 解决不了 planning 的根本问题，得改 architecture。

**4. Real Robot Data 很关键**  
WoW 系列是专门在 robot data 上训的，所以 IDM test 表现远好于 general video model。这说明要当 embodied world model，光靠 internet video 不够，必须有 robot interaction data 的 inductive bias。

## 我的直觉和联想

### 1. 关于 "World Model" 这个概念的反思

现在大家把 "video generation" 和 "world model" 混为一谈，这篇 paper 实际上是在 challenge 这个 conflation。**能生成漂亮视频 ≠ 理解世界 physics。**

这让我想到 LeCun 的 JEPA 那一套。他一直说 generative model 是错的路径，因为生成 pixel 级别的未来既不可能也没必要。这篇 paper 的结果某种程度上印证了：pixel-level generation 可以很逼真，但 internal representation 并没有真正 capture 物理。**也许真正的 world model 应该在 abstract representation space 里做 prediction，而不是在 pixel space。**

### 2. IDM Turing Test 的深层含义

这个 test 实际上把 video generation model 和 robot policy 放在了一个 closed loop 里：**world model 预测未来 → IDM 从未来反推 action → action 在真 world 执行**。

这让我想到 **Hindsight Experience Replay** 和 **Inverse Dynamics** 的关系。如果 world model 生成的 video 足够好，那 IDM 提取的 action 就是对的，robot 就能干活。这等价于说：**world model 的质量 = 它生成的 video 能被 IDM 正确解码成 executable action 的概率**。

这其实暗示了一种 training paradigm：**用 IDM 的 execution success rate 当 reward signal 来 fine-tune world model**。你生成视频 → IDM 提取 action → 真 robot 执行 → 成功与否当 reward → 回去更新 world model。这不就是 RL 的 closed loop 吗？只是把 environment 换成了 world model 本身。

### 3. Camera Motion Correction 的细节让我想到 SLAM

那个用 Lucas-Kanade + RANSAC 估计 camera affine transform 的 trick，本质上是 2D SLAM 的简化版。在 robotic video evaluation 里，camera 和 robot motion 是 entangled 的，这是 video model evaluation 独有的问题。**传统 video benchmark 不需要 care 这个，因为它们不关心 robot 执行。**

这也说明：**embodied world model 的 evaluation 比 general video generation 复杂得多，因为它要对接到 robot control 这个下游 task。**

### 4. GRPO 训 Evaluator 的 idea

用 RL fine-tune VLM 当 evaluator 这个 idea 不错，但 stage 2 只用 1,297 个样本，这量级有点小。Paper 说 Pearson correlation 很高，但我怀疑在 OOD scenario 上会不会 generalize。**如果生成的视频是 model 从没见过的 failure mode（比如很微妙的光影错误），这个 evaluator 可能就不准了。**

不过这个 paradigm 本身很有潜力：**用人类偏好数据 + GRPO 来 align VLM evaluator**，比 manually design rule 强多了。以后 evaluation metric 可能都这么搞。

### 5. 关于 Dense Prompt 的 null result

Dense prompt 让 Video Quality 和 Instruction Understanding 涨了，但 Planning 没涨。**这说明 planning 不是 input 信息不足的问题，是 model 内部 representation 的问题。**

这跟 LLM 里那个现象很像：你给 LLM more context 不代表它能更好地做多步推理。**Planning 需要的是 internal 的 structured reasoning mechanism，不是更多的 input。** 也许需要 explicit 的 planning module，或者 chain-of-thought 那种 explicit intermediate representation。

### 6. 那 40.74% 意味着什么

WoW-wan 的 40.74% 是最好的，但这意味着 **5 次里有 3 次真 robot 照着生成视频干，干不成**。这个数字其实还是太低。**如果 world model 想真正 deploy 到 robot 上，这个数字至少得 80% 以上。**

现在的差距说明：**我们离真正可用的 embodied world model 还有很远。** Video generation 的 progress 很快，但 physical grounding 的 progress 跟不上。

## 总结一句

这篇 paper 用两个 Turing Test（Human + IDM）戳破了一个幻象：**现在那些漂亮的 video generation model，离能当 robot brain 的 world model 还差得远。** 它能生成看起来真的视频，但那个"真"是 pixel-level 的，不是 physics-level 的。要真正有用，需要 physical consistency、planning ability、real-world executability，这些是现在 video model 最缺的。

## References

- Paper: [WoW: Towards a World Omniscient World Model](https://arxiv.org/abs/2509.22642)
- Benchmark comparison: [VBench-2.0](https://arxiv.org/abs/2503.21755), [Physics-IQ](https://arxiv.org/abs/2501.09038), [VideoPhy](https://arxiv.org/abs/2406.03520)
- Models: [Kling](https://app.klingai.com/cn/), [Hailuo](https://hailuoai.video/), [Cosmos-Predict2](https://research.nvidia.com/labs/dir/cosmos-predict2/), [Wan2.1](https://github.com/Wan-Video/Wan2.1), [CogVideoX](https://github.com/THUDM/CogVideo)
- Tools: [SAM2](https://github.com/facebookresearch/sam2), [DINOv2](https://github.com/facebookresearch/dinov2), [DreamSim](https://dreamsim-night.github.io/), [GroundedSAM2](https://github.com/IDEA-Research/Grounded-SAM-2)
- RL: [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300), [Qwen-2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- Datasets: [RoboMIND](https://github.com/RoboMIND-Platform/RoboMIND), [DROID](https://droid-dataset.github.io/)
- Related ideas: [LeCun JEPA](https://openreview.net/forum?id=BZ5a1r-kVsf), [World Models (Ha & Schmidhuber)](https://arxiv.org/abs/1803.10122), [Sora as World Simulator](https://openai.com/research/video-generation-models-as-world-simulators)

---

# Wow, wo, val! 论文深度解析

这篇paper来自Peking University、Beijing Innovation Center of Humanoid Robotics和HKUST的团队，提出了一个针对embodied world model的comprehensive benchmark，核心贡献是用Turing Test的思路来系统评估video foundation models能否真正作为robotic agents的"internal brain"。

## 核心Motivation：为什么需要这样一个Benchmark

当前video generation领域有一个根本性的认知gap：很多研究者把video foundation models当作predictive world models来用，期望它们能完成3D prediction、interactive generation等downstream tasks。但是在这之前，两个关键问题没人真正回答：

**Question 1**: 这些模型的generative generalization是否足够维持人类观察者眼中的perceptual fidelity？

**Question 2**: 它们是否足够robust，能作为real-world embodied agents的universal prior？

现有的video generation benchmarks（如VBench-2.0、T2V-CompBench、Physics-IQ等）大多focus在general-purpose video quality或者isolated dimensions上，overlook了robotic world models的特殊需求。比如一个model可能在conventional video metrics上得分很高，但在robotic scenario下产生physically impossible或contextually incorrect的predictions。Paper中通过实验证实了这种misalignment——standard video-quality scores在embodied settings下与human judgments关联很弱。

这里的核心intuition是：**embodied world model需要的不仅是"看起来像"，更需要"物理上对"和"action上executable"**。所以benchmark设计必须围绕embodied AI的特定需求展开。

## WoW-World-Eval的设计哲学

### 五个Core Capabilities

Paper提出一个competent embodied world model必须在五个fundamental且orthogonal的dimensions上展现mastery：

**1. Perception Understanding**
World model首先必须accurately perceive和represent环境。这通过tasks requiring fine-grained object recognition（attributes如color, shape, number, size）、spatial understanding（relative positions和arrangements）、affordance recognition（identifying interactive parts of objects）来评估。

**2. Decision-making and Planning**
Embodied agents必须execute long-horizon tasks。所以评估planning ability通过challenge model生成coherent video sequences for complex instructions。这要求implicitly understanding task decomposition into key sub-goals和respecting their causal dependencies。Paper收集了25个long-term planning samples。

**3. Predictive Reasoning**
这个dimension评估model的internal physics engine。Given initial state和action，model必须生成respect core physical principles的future，如object permanence、collision dynamics、trajectory plausibility。这直接probes model作为world simulator的能力。

**4. Interactive Execution**
与real world交互并在real robot上execute是embodied world model的ultimate goal。Paper收集了9个不同task从easy到hard，让model生成videos，然后用Gripper-Centric Inverse Dynamics Model (GC-IDM) interpret generated videos成actions在real robot上执行。

**5. Generative Generalization**
Universal world model应该不仅在In-Distribution data上表现好，还要generalize到unseen data。Paper通过GPT-5 style transfer或image editing生成unseen images，还收集了world-famous masterpiece paintings（如"Girl with a Pearl Earring"），让world model execute task instructions。

### Data Curation Pipeline

这是paper中很关键的部分。整个pipeline是semi-automated的：

**数据来源**：Open-source robotics data（RoboMIND、DROID）、in-house trajectories、AI-generated OOD samples。

**两阶段cleaning**：
1. **GPT-4o作为intelligent annotator**：scoring video-instruction pairs的matching level基于五个capability dimensions及其subdivisions，examining哪些instructions match哪些dimensions，实现large-scale filtering和coarse categorization。
2. **Human experts verification**：verify all samples保证category accuracy，resolve edge cases。Five additional annotators选择best initial frames for generation（robotic arm和manipulated object都在同一frame）和key point annotations（robotic arm gripper、joints、manipulated object）。

最终dataset：**609 samples**。Prediction (50.57%) 和 Perception (40.89%) dominate能力分布。Physical interaction覆盖single-object manipulation (56.08%)、multi-object interaction (41.89%)、dual-arm cooperation (2.03%)。还包含107个non-occluded views和54个semi-occluded views来test robustness。

## 22个Metrics的详细技术讲解

Paper的metrics设计是最值得深入学习的部分。我把它们分成五大类讲解。

### 1. Visual Fidelity Metrics

这是标准的video quality评估，包含五个complementary metrics：

**FVD (Fréchet Video Distance)** - Distribution-level metric：

$$\mathrm{FVD} = \|\mu_r - \mu_g\|_2^2 + \mathrm{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

其中：
- $\mu_r, \Sigma_r$：real videos的feature means和covariances（通过I3D network提取）
- $\mu_g, \Sigma_g$：generated videos的feature means和covariances
- $\|\cdot\|_2$：L2 norm
- $\mathrm{Tr}(\cdot)$：matrix trace
- $(\cdot)^{1/2}$：matrix square root

Lower FVD表示higher spatial-temporal coherence和distributional realism。FVD的intuition是衡量两个Gaussian distributions之间的Fréchet distance，类似FID but for videos。

**SSIM (Structural Similarity Index)**：

$$\mathrm{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

其中：
- $\mu_x, \mu_y$：local patches x和y的means
- $\sigma_x, \sigma_y$：variances
- $\sigma_{xy}$：covariance
- $C_1, C_2$：stability constants

**PSNR (Peak Signal-to-Noise Ratio)**：

$$\mathrm{PSNR}(x, y) = 10 \log_{10}\left(\frac{MAX^2}{\mathrm{MSE}(x, y)}\right)$$

其中$MAX$是maximum possible pixel value，MSE是mean squared error。

**DINO Score** - 使用DINOv2 self-supervised visual foundation model：

$$\mathrm{DINO}(g_t, r_t) = \frac{\langle f(g_t), f(r_t) \rangle}{\|f(g_t)\|_2 \|f(r_t)\|_2}$$

其中$f(\cdot)$是DINOv2 encoder，$g_t, r_t$是generated和reference frames at time t。

**DreamSim** - Combines CLIP、OpenCLIP和DINO embeddings，fine-tuned on human perceptual judgments：

$$\mathrm{DreamSim}(x, y) = 1 - \|E(x) - E(y)\|_2$$

其中$E(\cdot)$是fused embedding。

### 2. Instruction Semantic Alignment Metrics

这部分使用GPT-4o作为scalable evaluator，根据是否有ground-truth采用两种方法：

**With Ground-Truth**：
- **Caption Score** (1-5 scale): GPT-4o提取structured descriptions (Initial-、Processing-、Final-state) from both generated和GT videos，VLM scores their Caption Score
- **Sequence Match Score** (0-1 scale): 评估generated video action-object pairs against instruction的correctness of order
- **Execution Quality Score** (1-5 scale): correctness of action-object pairs

**Without Ground-Truth** (用于Generalization)：
- 只评估Sequence Match Score和Execution Quality Score

Execution Quality Score的五级rubric很值得注意：
1. object does not move, action not executed
2. object slightly moves OR action partially executed  
3. object slightly moves AND action partially executed
4. object reaches goal OR action fully executed
5. object reaches goal AND action fully executed

### 3. Physical Consistency and Causal Reasoning

这是paper中最novel的部分，包含三类metrics：

#### 3.1 Mask-guided Regional Consistency

这是paper提出的novel metric，用于disentangle background、robot arm和manipulated object的inconsistencies。

**Implementation**：
1. 使用GroundedSAM2 with human annotation获取robot arm、manipulated object(s)、background的masks
2. 使用DINOv3-Large提取patch-level features：$F_t \in \mathbb{R}^{H_p \times W_p \times d}$
3. Background region定义为complement of union of two tracked regions：

$$M_t^{\mathrm{bg}} = \mathbf{1} - (M_t^{\mathrm{obj}} \lor M_t^{\mathrm{arm}})$$

4. 计算normalized mask：

$$w_t^{\mathrm{r}}(i, j) = \frac{M_t^{\mathrm{r}}(i, j)}{\sum_{i', j'} M_t^{\mathrm{r}}(i', j') + \epsilon}$$

5. Mask-weighted averaging得到region feature：

$$\mathbf{f}_t^{\mathrm{r}} = \sum_{i, j} w_t^{\mathrm{r}}(i, j) F_t(i, j, :)$$

6. Cosine similarity计算temporal consistency：

$$\mathrm{Consist}^{\mathrm{r}}(a, b) = \begin{cases} \tilde{\mathbf{f}}_a^{\mathrm{r}} \cdot \tilde{\mathbf{f}}_b^{\mathrm{r}}, & \|\tilde{\mathbf{f}}_a^{\mathrm{r}}\|_2 > 0, \|\tilde{\mathbf{f}}_b^{\mathrm{r}}\|_2 > 0 \\ 0, & \text{otherwise} \end{cases}$$

7. Combine long-range和short-range temporal coherence：

$$s_t^{\mathrm{r}} = \frac{1}{2}\mathrm{Consist}^{\mathrm{r}}(1, t) + \frac{1}{2}\mathrm{Consist}^{\mathrm{r}}(t-1, t)$$

8. Video-level Mask-guided Regional Consistency：

$$\mathrm{MRC}^{\mathrm{r}} = \frac{1}{T-1} \sum_{t=2}^{T} s_t^{\mathrm{r}}$$

其中$T$是frame数量。这个metric能pinpoint temporal flaws的source，比如即使object和background稳定，也能识别出"jittery" robot arm。

#### 3.2 Trajectory Consistency

比较generated videos和ground-truth的trajectory，track end-effector和object trajectories。

**Keypoint labeling和SAM2 tracking**：
- Human annotators在first frame放置N个representative points：$\mathcal{K}_1 = \{\mathbf{u}_k^{(1)} \in \mathbb{R}^2\}_{k=1}^N$
- SAM2返回binary segmentation mask：$\mathbf{M}^{(t)} \in \{0, 1\}^{H \times W}$
- Foreground pixels：$\Omega^{(t)} = \{(x, y) | \mathbf{M}^{(t)}(x, y) = 1\}$
- Mask-to-point trajectory通过centroid：

$$\mathbf{p}_t = (x_t, y_t) = \frac{1}{|\Omega^{(t)}|} \sum_{(x, y) \in \Omega^{(t)}} (x, y)$$

**Camera-motion-aware trajectory correction**：在generated videos中，observed end-effector trajectory entangle了true robot motion和camera drift。Paper通过Lucas-Kanade optical flow估计camera trajectory，然后从observed trajectory中减去：

$$\mathbf{p}_t^{\mathrm{true}} = \hat{\mathbf{p}}_t - \mathbf{c}_t$$

Camera trajectory estimation用Shi-Tomasi keypoints on image boundary，通过RANSAC估计2D affine transform：

$$\mathbf{A}_t = \begin{bmatrix} a_{11} & a_{12} & t_x \\ a_{21} & a_{22} & t_y \end{bmatrix}$$

Camera displacement（与background motion相反）：

$$\Delta \mathbf{c}_t = -\mathbf{t}_t = (-t_x, -t_y)$$

**三个trajectory metrics**：

1. **ATE (Absolute Trajectory Error)**：

$$\mathrm{ATE} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} \|\hat{\mathbf{c}}_t^{\mathrm{gen}} - \hat{\mathbf{c}}_t^{\mathrm{gt}}\|_2^2}$$

2. **RPE (Relative Pose Error)**：

$$\mathrm{RPE} = \sqrt{\frac{1}{T-\Delta} \sum_{t=1}^{T-\Delta} \|\mathbf{v}_t^{\mathrm{gen}} - \mathbf{v}_t^{\mathrm{gt}}\|_2^2}$$

其中$\mathbf{v}_t = \hat{\mathbf{c}}_{t+\Delta} - \hat{\mathbf{c}}_t$是normalized relative motion。

3. **L2Norm Error**：

$$\mathrm{L2norm} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} \|\mathbf{q}_t^{\mathrm{gen}} - \mathbf{q}_t^{\mathrm{gt}}\|_2^2}$$

4. **DTW (Dynamic Time Warping)**：

$$\mathrm{DTW}(\mathbf{q}^{\mathrm{gen}}, \mathbf{q}^{\mathrm{gt}}) = \min_\pi \sum_{(t, s) \in \pi} \|\mathbf{q}_t^{\mathrm{gen}} - \mathbf{q}_s^{\mathrm{gt}}\|_2$$

其中$\pi$是monotonic warping path，allowing nonlinear temporal alignment。

5. **Fréchet Distance**：

$$\mathrm{FD}(\mathbf{q}^{\mathrm{gen}}, \mathbf{q}^{\mathrm{gt}}) = \inf_{\alpha, \beta} \max_{t \in [0, 1]} \|\mathbf{q}_{\alpha(t)}^{\mathrm{gen}} - \mathbf{q}_{\beta(t)}^{\mathrm{gt}}\|_2$$

其中$\alpha, \beta$是continuous, non-decreasing reparameterizations。Fréchet Distance capture minimum leash-length，提供strict measure of geometric similarity。这里关键的intuition是：DTW允许speed variations，Fréchet强制simultaneous forward progression。

#### 3.3 Physical Common Sense

这是paper中最technically sophisticated的部分。使用fine-tuned Qwen-2.5-VL (7B)来score六个dimensions：object interaction、basic physical properties、temporal and causal consistency、lighting/shadows/reflections、fluid and particle behavior、local anomalies。

**Two-stage GRPO fine-tuning**：

GRPO (Group Relative Policy Optimization)的核心objective：

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G} \sum_{i=1}^{G} \left(\min\left[\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} \hat{A}_i, \mathrm{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\varepsilon, 1+\varepsilon\right) \hat{A}_i\right] - \beta D_{KL}[\pi_\theta \| \pi_{ref}]\right)\right]$$

其中：
- $\pi_\theta$：policy model
- $\pi_{\theta_{old}}$：old policy（before update）
- $\pi_{ref}$：reference model（KL regularization target）
- $G$：group size（sampled outputs数量）
- $o_i$：第i个sampled output
- $q$：input query
- $\varepsilon$：clipping hyperparameter
- $\beta$：KL-divergence penalty coefficient
- $\hat{A}_i$：advantage estimate
- $D_{KL}$：KL divergence

Advantage计算采用group-relative formulation：

$$\hat{A}_i = \frac{r_i - \mathrm{mean}(\{r_1, \dots, r_G\})}{\mathrm{std}(\{r_1, \dots, r_G\}) + \epsilon}$$

其中$r_i$是第i个output的reward。这种formulation作为dynamic baseline，effectively reducing variance without requiring separate value network。这是GRPO相对于standard PPO的关键优势。

**Stage 1: Foundational Video Understanding Fine-tuning**
- Dataset: ~50,000 samples from six video understanding benchmarks
- Format: multiple-choice Video Question Answering (VQA)
- Group size G=8
- Rule-based reward: correct answer → r_i=1.0, else r_i=0.0
- Result: average accuracy从60.83% (Qwen-2.5-VL 7B backbone) 提升到71.51%

**Stage 2: Scoring Alignment Fine-tuning**
- Dataset: 1,297 internally annotated data points
- Output: structured JSON with four keys (video_quality, instruction_following, physical_consistency, planning_logic)
- Reward function:

$$e_k = \frac{|s_k'^{gt} - s_k'^{out}|}{4.0}$$

其中$s_k'^{gt}, s_k'^{out}$是clipped到[1.0, 5.0]的scores，4.0是normalization factor（5-1）。

Mean error：

$$\bar{e} = \frac{1}{|\mathcal{K}_{match}|} \sum_{k \in \mathcal{K}_{match}} e_k$$

Final reward：

$$R(y_{out}, y_{gt}) = \max(0.0, \min(1.0, 1.0 - \bar{e}))$$

### 4. Planning and Task Decomposition

基于RoboBench的Directed Acyclic Graphs (DAGs)方法。

**Process**：
1. Parse natural language instruction和ground-truth video成ground-truth plan DAG
2. Nodes是atomic actions parameterized by $\langle\text{skill, object, args}\rangle$
3. Edges代表dependencies
4. Compare model-generated plan to ground-truth DAG

**两个scores**：
1. **Node Correctness**：fraction of correctly predicted nodes aligning with ground-truth nodes
2. **Task Completion**：使用MLLM conduct lightweight world-simulation rollout

**LongHorizon Score**：

$$S_{\mathrm{LongHorizon}} = (S_{\mathrm{NodeCorrectness}} + S_{\mathrm{TaskCompletion}}) \times 50$$

### 5. Overall Benchmark Score - Monotone Parametric Mappings

这是paper中一个很精巧的设计，用于把不同metrics统一到0-100 scale。

**Step 1: Pre-scale to [0, 1]**

For "higher-is-better" (HIB) metrics:

$$\hat{x}_{i,m}^{\mathrm{HIB}} = \frac{\mathrm{clip}(x_{i,m}; L_m, U_m) - L_m}{U_m - L_m} \in [0, 1]$$

For "lower-is-better" (LIB) metrics:

$$\hat{x}_{i,m}^{\mathrm{LIB}} = 1 - \hat{x}_{i,m}^{\mathrm{HIB}} \in [0, 1]$$

其中$L_m, U_m$是fixed anchors for metric m。例如：
- PSNR (HIB): $L_{\mathrm{PSNR}} = 0, U_{\mathrm{PSNR}} = 50$
- FVD (LIB): $L_{\mathrm{FVD}} = 0, U_{\mathrm{FVD}} = 2000$

**Step 2: Monotone parametric mapping**

$$s_{i,m} = 100 f_m(\hat{x}_{i,m}; \theta_m), \quad s_{i,m} \in (0, 100)$$

四种mapping families（all strictly increasing on [0, 1]）：

1. **Simple**: $f(x) = x$

2. **Power (Gamma)**: $f_\gamma(x) = x^\gamma, \gamma > 0$
   - $\gamma > 1$ accentuates high end
   - $\gamma < 1$ expands low end

3. **Logit temperature**: 
   $$f_T(x) = \sigma(\mathrm{logit}(x)/T), T > 0$$
   $$\sigma(t) = \frac{1}{1 + e^{-t}}$$
   - $T < 1$ expands mid-range, compresses extremes

4. **Tanh slope**:
   $$f_\kappa(x) = \frac{1}{2}(\tanh(\kappa(2x-1)) + 1), \kappa > 0$$
   - $\kappa > 1$ expands mid-range

**Parameter selection**：对每个metric m，$\theta_m$在fixed development set上通过maximizing Fisher-z averaged Pearson correlation between $f_m(\hat{x}; \theta)$和human ratings across K-fold CV来选择，Spearman correlation作为tie-breaker。Chosen $\theta_m$然后frozen并applied to all evaluations。

Paper在Table 6中给出了所有metrics的mapping parameters，比如：
- FVD: gamma, 1.52
- PSNR: tanh, 4.71
- SSIM: gamma, 0.61
- DINO: gamma, 3.06

**Step 3: Aggregation (weighted arithmetic mean)**

Metrics grouped into categories g。对每个model i，$\mathcal{M}_g$是group g中available metrics的set，$N_{i,g} = |\mathcal{M}_g|$是其cardinality。Overall score：

$$O_i = \sum_{g \in \mathcal{G}_i} \frac{W_g}{\sum_{h \in \mathcal{G}_i} W_h} \left(\frac{1}{|\mathcal{M}_g|} \sum_{m \in \mathcal{M}_g} s_{i,m}\right)$$

其中$\mathcal{G}_i = \{g : N_{i,g} > 0\}$是model i有至少一个valid metric的groups集合。Setting $W_g \equiv 1$ yields unweighted overall mean across all available groups。

## 两个Turing Test设计

### Human Turing Test

基于psychophysics的2AFC (two-alternative forced-choice) methodology。13个participants区分real videos和generated videos，compute proportion of generated videos that successfully fool human evaluators。

**关键发现**：
- Deceive-human ratio与Overall score强相关（r = 0.679）
- Video Quality (r = 0.874)和Physical Law (r = 0.753)的correlation最高
- 这表明**visual quality和physical plausibility都是让generated videos appear real的关键**

### IDM (Inverse Dynamics Model) Turing Test

这是paper中最有创意的贡献。Machine Turing Test使用real-world-trained IDM来evaluate生成的videos是否exhibit physically executable dynamics。

**核心idea**：If generated videos lead IDM to output plausible actions that are executable in real world，说明model的outputs在physical和action plausibility方面indistinguishable from real data。

**GC-IDM (Gripper-Centric Inverse Dynamics Model)** validation：
- 在9个manipulation tasks上evaluate
- 每个task 10个ground-truth execution videos
- Tasks分Easy/Medium/Hard三级

GC-IDM在ground-truth video replay上达到**90% overall replay accuracy**，远超ResNet-based IDM和AVDC baselines。这证明GC-IDM是reliable evaluator of physical plausibility。

## 实验结果深度分析

### Quantitative Results

**Overall Performance**：
- Best closed-source model: Hailuo (52.55)
- Best open-source model: WoW-cosmos2 (50.74)
- Kling虽然Video Quality和Physical Law强，但Instruction Understanding和Planning弱，导致overall只有37.93

**Video Quality**：
- Hailuo: 56.09 (closed-source best)
- WoW-wan: 55.38 (open-source best)
- WoW-cosmos2在DreamSim (64.42)和FVD (85.71)上甚至slightly exceeding Hailuo

**Instruction Understanding**：
- WoW-cosmos2: 70.36 (open-source best, surpassing Hailuo 70.11)
- Hailuo在Caption Score (88.78)和Exec. Quality (70.48)强
- WoW-cosmos1在Seq. Match (63.33)最强

**Physical Law**：
- Kling: 68.02 (best overall)
- WoW-cosmos2: 66.18 (open-source best)
- Camera motion基本saturated（所有models都很stable）
- Commercial models focus more on object manipulation，WoW-cosmos2更accurate in handling robot arm movements

**Planning Reasoning** - 这是primary bottleneck：
- Hailuo: 17.27 (best)
- Cosmos-Predict2: 13.41 (open-source best)
- 所有scores都markedly lower且compressed
- 表明long-horizon planning和temporal reasoning仍然underexplored

### Dense Prompts Results

使用InternVL3-78B将short prompts扩展为dense prompts（包含explicit environment和object references、sub-goals、physical constraints）。

**Key findings**：
- 所有dimensions都有improvements
- Video Quality: WoW-cosmos1达到60.55，超过closed-source baseline (56.09)
- Instruction Understanding: Cosmos-Predict2增加近20分
- Physical Law: WoW-cosmos1增加最大(+7)
- **Planning Reasoning: DAG metric minimal change**

最后一点特别important：detailed prompting alone不能compensate current world models在long-horizon reasoning和structured task decomposition方面的limitations。这需要deeper architectural advances beyond prompt refinement。

### Turing Test Results

**Human Turing Test**：
- Hailuo和WoW-cosmos2 variants最effective at fooling humans
- Deceive-human ratio与Overall score强相关（r = 0.679）

**IDM Turing Test**：
- Kling: 9.88%
- Hailuo: 2.47%
- CogVideoX, Cosmos-Predict1, Wan2.1: 0.00%
- Cosmos-Predict2: 8.64%
- WoW-wan: **40.74%** (best)
- WoW-cosmos2: 18.52%

**关键insight**：most high-scoring models on WoW-World-Eval仍然fail this test。这表明visual realism alone不足以achieve embodied execution。需要both physics-grounded modeling和real-world exposure。当前embodied world models生成的videos与real physical world之间仍有significant gap。

## 核心Insights总结

1. **四个dimensions不可indispensable**：Video Quality, Instruction Understanding, Physical Law, Planning Reasoning各自capture distinct failure modes。一个model可能在某些dimension强但overall差。

2. **Planning是primary bottleneck**：所有models在long-horizon planning上表现poor。需要更explicit的planning representations或control mechanisms。

3. **Visual realism ≠ Physical executability**：Commercial video models虽然视觉quality高，但在IDM Turing Test上几乎collapse。只有经过real-robot data training的WoW系列models表现较好。

4. **Dense prompts有用但有limit**：rich textual conditioning可以meaningfully enhance generation quality，但无法解决fundamental planning gaps。

5. **Human alignment强**：Pearson Correlation > 0.93证明benchmark能reliably approximate human judgment，作为Human Turing Test的reliable proxy。

## 参考资源

- Paper arXiv链接：https://arxiv.org/abs/2509.22642 (WoW原始paper)
- GitHub: https://github.com/Karpathy/reasoning (Karpathy的零样本推理)
- RoboMIND: https://github.com/RoboMIND-Platform/RoboMIND
- DROID: https://droid-dataset.github.io/
- Cosmos: https://research.nvidia.com/labs/dir/cosmos-predict2/
- Wan: https://github.com/Wan-Video/Wan2.1
- CogVideoX: https://github.com/THUDM/CogVideo
- SAM2: https://github.com/facebookresearch/sam2
- DINOv2: https://github.com/facebookresearch/dinov2
- GroundedSAM: https://github.com/IDEA-Research/Grounded-SAM-2
- DreamSim: https://dreamsim-night.github.io/
- Kling: https://app.klingai.com/cn/
- Hailuo: https://hailuoai.video/
- Qwen-2.5-VL: https://github.com/QwenLM/Qwen2.5-VL
- InternVL3: https://github.com/OpenGVLab/InternVL
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- RoboBench: https://arxiv.org/abs/2510.17801
- VideoPhy: https://arxiv.org/abs/2406.03520
- VBench-2.0: https://arxiv.org/abs/2503.21755

## 对Karpathy的特别Insights

作为Karpathy，你可能特别感兴趣的几点：

1. **GRPO的应用**：这个原本用于math reasoning的RL algorithm在visual evaluation alignment上的应用很有创意。Two-stage设计（先learn physical understanding，再align to human scoring）是一个值得借鉴的paradigm。

2. **IDM Turing Test的哲学**：这本质上是问"一个real-world trained policy能否被generated video欺骗"。这和你之前讨论过的"model作为environment simulator"的思想很契合，但这里更极端——直接用generated video作为policy的input，看output action能否在real robot上execute。

3. **Camera-motion-aware trajectory correction**：这个细节很关键。在generated videos中，robot motion和camera motion是entangled的。通过Lucas-Kanade optical flow + RANSAC affine estimation来disentangle，是从robotics SLAM借鉴的technique应用到video evaluation的有趣案例。

4. **Monotone parametric mappings**：选择different mapping families (gamma, logit temperature, tanh slope)来maximize correlation with human ratings，这是一个数据驱动的metric calibration方法，比hardcoded normalization更principled。

5. **Planning bottleneck**：你的 intuitions about LLMs需要explicit planning representations在这篇paper中得到了empirical confirmation。即使是dense prompts也无法解决这个fundamental limitation。
