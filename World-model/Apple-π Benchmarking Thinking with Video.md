---
source_pdf: Apple-π Benchmarking Thinking with Video.pdf
paper_sha256: f2112d08a3d732ca0c0fa7037ee1fc97d2081e991ac97fab37a50e4d57ec1164
processed_at: '2026-08-18T01:04:39-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 Apple-π

## 一句话总结

**现在的video generation model根本不懂物理，它们只是在"画得像"，而不是在"算得对"。**

## 他们到底在测什么？

想象你给模型看一张图：一个球在斜坡顶上，旁边标了 mass = 5kg，angle = 30°，g = 9.8。然后你问它：

1.  **你能把这些标注读出来吗？**（Perception）
2.  **你知道该用哪个物理公式吗？**（Formulation）
3.  **你能按这个公式把球滚下来的过程画出来吗？**（Deduction）

就这么简单。但结果非常惨。

## 为什么这件事重要？

现在所有人都在喊"Sora是world model！Veo是world model！"。但Apple-π的人说：**等一下，你说它是world model，那它到底懂不懂Newton第二定律？**

之前的benchmark只看"生成出来的视频顺不顺眼"，这就像考试只看最终答案对不对，完全不管解题过程。一个学生答案对了，可能是真懂，也可能是蒙的。Apple-π要做的是**把解题过程也拿出来看**。

## 他们怎么测的？

### Dataset: Orchard
400个视频，涵盖经典力学的主要场景：
- 自由落体、抛体运动、斜面、圆周运动
- 弹性碰撞、非弹性碰撞
- 静止、匀速直线运动

数据来自三个地方：
- **Isaac Sim 模拟器**（243个）：精确到pixel的ground truth
- **实验室自己拍的**（121个）：真实世界的光照和材质
- **YouTube物理频道**（36个）：更多样化的场景

物体只有四种：球、方块、圆柱、圆锥。**故意用简单几何体**，不希望模型靠"我见过这个茶杯"这种semantic prior来蒙混过关。

### 五个子测试

这是最精彩的地方。他们把"物理推理"拆成了五个可诊断的环节：

| Subtrack | 干什么 | 类比 |
|----------|--------|------|
| P-T | 把图上的数字标注抄出来 | OCR |
| P-G | 把物理物体抠出来，去掉背景和标注 | Image Segmentation |
| F-T | 四选一选正确物理公式，代入数字 | 做选择题 |
| F-G | 给定时间t*，画出那一刻的物体位置和速度 | 算中间状态 |
| Deduction | 生成完整物理过程视频 | 做大题 |

**关键设计**：输入是一个annotated first frame，所有物理量（mass、velocity、angle等）直接标在图上。这样做是为了**排除"模型看不懂题目"这个干扰因素**——如果模型连初始条件都读不到，后面测什么都没意义。

## 结果有多惨？

| Model | 总分 | 
|-------|------|
| Seedance 2.0 | 0.473 |
| Wan2.2 | 0.267 |
| HunyuanVideo-1.5 | 0.177 |
| Veo 3.1 | 0.313 |
| GPT Image 2 | 0.704 |
| Nano Banana 2 | 0.699 |

满分是1.0。最好的video model只有**0.473**。

### 两个核心发现

**发现一：Unified model碾压纯video model**

GPT Image 2和Nano Banana 2（都是understanding+generation的unified架构）把所有纯video model按在地上摩擦。

为什么？因为纯video model只会"画"，不会"读"和"想"。你让它抄图上的数字，它抄不对；你让它选公式，它选不出。它没有**explicit understanding**能力。

但注意：即使是GPT Image 2，在Deduction上也只有0.406分。**生成符合物理定律的时序动态，对当前所有模型来说都是地狱级难度。**

**发现二：Reasoning Funnel——越往后越崩**

Perception分数 > Formulation分数 > Deduction分数

模型能抄标注，但不会选公式；会选公式，但不会推导动态。这说明中间步骤是断的——**表面上的成功不等于真正的理解**。就像学生抄对了已知条件，但根本不会做题。

## 一个特别有意思的技术细节

### Time Alignment
不同模型生成的视频长度和fps都不一样。有人生成5秒24fps，有人生成8秒30fps。怎么比较？

Apple-π的做法：**prompt要求生成T_c秒，那就把模型输出的所有帧当成T_c秒的轨迹**。如果prompt说"生成4秒运动"，模型吐了192帧过来，那就当48fps处理。模型想用慢动作蒙混过关？没门。

## 我的 Intuition

这篇paper本质上在说一件事：**video generation和physical simulation是两回事。**

当前的video model学到了"球从高处掉下来会越来越快"这种visual pattern，但它不是通过 $v = v_0 + gt$ 算出来的，而是从训练数据里memorize的。所以一旦你给它精确的初始条件，要求它按特定公式推演特定时长，它就露馅了。

这就像一个人看了很多篮球比赛，能模仿投篮动作，但你让他计算抛物线轨迹，他做不到——因为他从来没有真正"理解"过抛物线方程。

**Apple-π的价值在于：它第一次把"看起来对"和"算得对"区分开了，并且证明了当前所有模型都还停留在"看起来对"的阶段。**

## 相关链接

- Project Page: https://21yrm.github.io/Apple-PI-homepage/
- Sora as World Simulator (OpenAI): https://openai.com/research/video-generation-models-as-world-simulators
- Kang et al. "How far is video generation from world model" (ICML 2025): https://proceedings.mlr.press/v267/kang25g.html
- Chain-of-frames reasoning: https://arxiv.org/abs/2506.00318

---

这篇名为《Apple-π: Benchmarking Thinking with Video》的paper由NTU S-Lab和CUHK的研究人员完成，它对当前video generation models是否真正具备“world model”的能力提出了深刻的质疑，并提供了一套极其精细的diagnostic framework。以下是对该paper的深度技术拆解与intuition构建。

### 1. Core Motivation: 从 Aristotle 到 Newton 的跨越

当前业界弥漫着一种乐观情绪，认为通过海量视频数据训练出的video generation models（如Sora、Veo）已经internalize了physical laws，成为了emerging world models。这篇paper指出，现有的benchmarks仅仅停留在**output-level plausibility**（生成结果看起来对不对），完全忽略了**reasoning process**（是否基于物理定律推导出来的）。

作者借用物理学史隐喻：当前的video models更像Aristotle，依靠visual intuition和pattern matching生成看起来合理的运动；而真正的world model需要像Newton，通过感知物理量、提取governing law、推导时序动态来实现law-grounded deduction。为了量化这两者之间的gap，Apple-π诞生了。

### 2. 核心架构：三大组件深度解析

Apple-π的设计逻辑极其严密，通过三个组件将黑盒的video generation过程彻底白盒化。

#### 2.1 Orchard Dataset
为了排除confounder，Orchard dataset采用了**law-first**的设计哲学。它包含400个video，分为三种来源：Simulated (243 cases，基于NVIDIA Isaac Sim)、Self-recorded (121 cases，实验室控制环境)、Internet-sourced (36 cases，YouTube物理频道)。

为了控制nuisance variable，所有动态物体被标准化为四种primitive solids：sphere, cube, cylinder, cone。这种设计保证了center of mass、contact surface等物理属性的精确可测。
Dataset分为两层：
*   **Single-law branch**：包含三个pillars。
    *   *Law of universal gravitation*：自由落体 ($h = \frac{1}{2}gt^2$)，抛体运动 ($\vec{r}(t) = \vec{r}_0 + \vec{v}_0 t + \frac{1}{2}\vec{g}t^2$)，斜面运动 ($a = g\sin\theta$)，圆周运动 ($mg = mv^2/r$)。
    *   *Conservation of momentum*：完全弹性碰撞 ($e=1$)，完全非弹性碰撞 ($e=0$)，非弹性碰撞 ($0<e<1$)。
    *   *Newton's first law*：静止 ($\sum\vec{F}=0, \vec{v}=0$)，匀速直线运动 ($\sum\vec{F}=0, \vec{v}(t)=\vec{v}_0$)。
*   **Multi-law branch**：组合上述定律，例如斜面运动接抛体运动。这用于测试模型在law transition时的**state transfer**能力。

#### 2.2 Benchmark Protocol: Five Subtracks
这是本文最精妙的设计。作者将科学推理分为三个阶段，并细分为五个subtracks。所有subtrack的input是infographic-style annotated first frame（将mass、velocity、angle等物理量直接标注在图像对应物体上，消除reference resolution负担），output是chain-of-frames video。

1.  **Perception-Text (P-T)**：类似OCR。要求模型将第一帧的annotation提取出来，输出在纯白背景上。测试模型能否读取物理量。
2.  **Perception-Graphic (P-G)**：类似Instance Segmentation。要求模型抹除所有annotation和背景，仅保留物理实体在纯白背景上。测试模型能否grounding物理实体。
3.  **Formulation-Text (F-T)**：四选一选择题。选项包含正确公式、符号易混淆公式、无关公式、伪造公式。要求模型选出正确公式并代入数值。测试模型是否internalize了物理定律。
4.  **Formulation-Graphic (F-G)**：给定目标时间 $t^\star$，预测该时刻的scene configuration，并画出velocity arrow。测试模型能否将公式instantiate为特定时刻的物理状态。
5.  **Deduction (Ded.)**：生成符合物理定律的完整轨迹视频。测试模型在时间维度的law-consistent dynamics。

#### 2.3 Evaluation Suite: Subjective + Objective
评估体系同样采取混合策略：MLLM-based (Gemini 3 Flash)评估format、layout、visual quality；Physics-law-grounded objective metrics评估物理正确性。

这里有一个极其关键的**Time and FPS Alignment**技术细节。由于不同模型输出的video长度和fps各异，不能直接逐帧比对。Apple-π强行将生成的视频总时长映射为prompt要求的物理时长 $T_c$。
公式定义如下：
$$ i_c(t) = \text{clip}(\text{round}(t f_c^{\text{GT}}), 0, N_c^{\text{GT}}-1) $$
*   $i_c(t)$: 在物理时间 $t$ 时对应的ground truth帧索引。
*   $f_c^{\text{GT}}$: case $c$ 的原生ground truth帧率。
*   $N_c^{\text{GT}}$: case $c$ 的GT总帧数。
*   $\text{clip}$: 截断函数，防止越界。

对于生成的视频，计算其有效评估帧率：
$$ \hat{f}^{\text{gen}} = \frac{N^{\text{gen}}}{T_c} $$
*   $\hat{f}^{\text{gen}}$: 模型生成视频的有效评估帧率。
*   $N^{\text{gen}}$: 解码后的生成视频总帧数。
*   $T_c$: prompt要求的物理持续时间。

这意味着如果prompt要求生成4秒的视频，模型却生成了8秒24fps（共192帧）的视频，系统会强行将这192帧视为4秒的轨迹（即48fps）进行对齐评估。这消除了模型生成慢动作或快进带来的歧义。

在Deduction的评分融合上，公式如下：
$$ S_{\text{Ded}} = 0.20 S_{\text{integrity}} + 0.20 S_{\text{fidelity}} + 0.60 S_{\text{physics}} $$
*   $S_{\text{integrity}}$: 评估annotation是否清除、object consistency。
*   $S_{\text{fidelity}}$: 评估visual quality、motion smoothness、PSNR。
*   $S_{\text{physics}}$: 权重高达0.60，包含spatiotemporal mask overlap、3D velocity error等。
这种权重分配明确宣示：视觉效果再好，物理定律算错也是零分。

### 3. 实验数据解析与 Intuition 构建

作者测试了11个model，包括5个video generation models和6个unified understanding-generation models。Table 2的数据揭示了当前AI领域的深层真相。

#### 3.1 Unified Models 的碾压性优势
GPT Image 2 (0.704) 和 Nano Banana 2 (0.699) 总分远超表现最好的video model Seedance 2.0 (0.473)。在Perception和Formulation阶段，unified models具有统治级表现。

**Intuition**: 纯粹的video diffusion models极度缺乏explicit understanding能力。它们在拟合像素分布时表现优异，但在要求精确读取数值、执行符号逻辑推导时显得无能为力。Unified models将VLM的理解能力与生成能力耦合，显著改善了interface skills和law selection能力。然而，即使是GPT Image 2，在Deduction阶段的得分也仅有0.406，这表明无论架构如何，long-term temporal dynamics consistency仍是当前技术的死穴。

#### 3.2 Reasoning Funnel: 越往后越坍塌
Figure 4展示了Perception -> Formulation -> Deduction的分数递减趋势。

**Intuition**: 这说明video models并没有建立真正的causal reasoning chain。它们或许能通过视觉先验抄写标注，或者盲猜出一个正确的公式，但这些intermediate success并不能transfer到最终的时序推演。就像LLM的CoT，如果中间步骤只是表面拟合，一旦要求长时间跨度的一致性rollout，误差就会指数级放大。

#### 3.3 Multi-law 与 Sim-to-Real Gap
*   **Multi-law failure**: 在涉及多个物理定律组合的case中，模型得分极低。这揭示了模型在处理**state transfer**时的严重缺陷。例如，小球从斜面滑下进入抛体运动阶段，斜面底部的exit velocity本应作为抛体的initial velocity。但当前模型在law transition的瞬间，极易发生velocity突变或identity drift。
*   **Sim-to-Real gap**: 所有模型在real-world video上的表现均差于simulated video。尽管governing law不变，但real-world的optical and material cues严重干扰了模型的grounding能力。这暗示模型的物理感知依然停留在texture matching层面，未能提取出invariant physical variables。

### 4. 深度联想与未来方向

从Apple-π的发现中，我们可以延伸出关于AI发展路线的深度联想：

1.  **Pixel-level Generation 是 World Model 的死胡同吗？**
    目前业界有一种幻觉，认为只要scale up video diffusion models，物理定律就会自然emerge。Kang等人在ICML 2025发表的《How far is video generation from world model》也指出，video generation models不理解物理。Apple-π进一步从精细的stage-wise evaluation证明了这一点。当前自回归或扩散模型在pixel space进行预测时，绝大部分capacity被用来拟合光照、纹理、背景等高频噪声，留给rigid body dynamics的表征能力极其匮乏。这为Yann LeCun提倡的JEPA架构（在latent space预测状态）提供了强有力的背书。
2.  **Explicit Understanding 在 Generation 中的必要性**
    GPT Image 2等unified models的优势表明，未来的world simulator必须具备“先理解、后生成”的机制。SenseNova-U1-8B-MoT-Think在F-T上得分0.154，而GPT Image 2达到0.824，这种差距不仅仅来自参数量，更来自mature post-training for instruction following和text rendering。未来的系统可能需要内嵌一个physics engine或者symbolic reasoning module，在生成视频帧之前，先在latent space或symbolic level演算好整条trajectory。
3.  **State Transfer 与 Autoregressive Rollout 的极限**
    Multi-law任务的失败，本质上是autoregressive generation在长时序rollout中error accumulation的体现。每一帧的生成误差都会作为下一帧的input，这在多阶段物理过程中是致命的。未来的架构或许需要引入global trajectory constraint，或者采用hierarchical generation（先生成keyframe physics state，再进行temporal interpolation）。

### 5. 参考资源

*   Apple-π Project Page: https://21yrm.github.io/Apple-PI-homepage/
*   Sora as World Simulators (OpenAI): https://openai.com/research/video-generation-models-as-world-simulators
*   Kang et al. "How far is video generation from world model" (ICML 2025): https://proceedings.mlr.press/v267/kang25g.html
*   Wiedemer et al. "Chain-of-frames: Advancing video understanding in multimodal llms via frame-aware reasoning": https://arxiv.org/abs/2506.00318
*   VideoPhy: Evaluating physical commonsense for video generation: https://arxiv.org/abs/2406.03520
