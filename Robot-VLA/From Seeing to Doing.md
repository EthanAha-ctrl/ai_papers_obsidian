---
source_pdf: From Seeing to Doing.pdf
paper_sha256: 07d60b41f9a41ee7d573ceb4278c7b498a367caca5da65d94c13865fd95e1136
processed_at: '2026-08-04T10:58:03-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 FSD

Karpathy,我用最直白的方式给你讲讲这篇 paper 在干啥。

---

## 这帮人想解决什么问题?

你手头有个 VLM(像 GPT-4V 那种能看图说话的模型),你想让它直接控制机械臂干活——比如"把胡萝卜放盘子上"。现在的标准做法是把这个 VLM 改造成 VLA,让它直接输出机械臂的关节角度或者末端坐标。

听起来简单,实际上有两个大坑:

**第一个坑:数据不够**。语言和图片的 training data 互联网上一抓一大把,但机器人数据呢?得真去开机械臂录,或者人工遥操作。Open X-Embodiment 这种数据集已经是顶配了,规模跟 web data 比还是差几个数量级。scaling law 想触发都触发不了。

**第二个坑:机械臂五花八门**。WidowX、xArm、Franka、Kuka 这些 robot,关节数不一样、动作空间维度不一样、连坐标系都不一样。你录了 Franka 的数据想 transfer 到 xArm 上,action label 压根对不上号。end-to-end 让一个模型直接吐出 action,它很容易就 overfit 到某个具体的 robot 上去了。

---

## 他们的 core insight

这帮 Tianjin University 的人想明白一件事:**别让 VLM 直接预测机械臂怎么动,让它预测"被操作的那个物体应该怎么动"**。

物体怎么动跟机械臂长啥样没关系。你预测"胡萝卜从 A 点移动到 B 点"这种描述,在 WidowX 上是这样,在 xArm 上也是这样,甚至在没有机械臂的场景里也是这样。这就把"embodiment heterogeneity"这个老大难问题绕过去了。

那机械臂怎么动呢?交给传统 motion planner 去算。CuRobo 这种工具,你给它目标点,它能自己规划一条 collision-free 的轨迹出来。这种几何规划问题早就被工程界解决得很完善了,没必要让神经网络去学。

**所以 FSD 的 philosophy 就是:让 VLM 做它擅长的事(看图、推理语义),让传统机器人工具做它擅长的事(几何规划),中间用"visual aids"这种坐标表示来桥接**。这个 philosophy 跟你和 Yann LeCun 这些人一直讲的"模块化"思路挺像——别把所有东西都塞进一个 monolith network 里。

---

## Visual Aids 是啥?

就是模型吐出来的三种坐标表示:

1. **Affordance box**:一个矩形框,表示"物体应该被放在这个区域里"。比如"把寿司放进银锅",模型吐出锅内部的一个矩形。

2. **Affordance points**:一组点(8个),更精细地标记放置位置。

3. **Visual trace**:8 个点组成的有序序列,描述物体从起点到终点的运动路径。比如胡萝卜先抬起来、再平移、最后放下,就是 8 个点连成的轨迹。

所有坐标都归一化到 [0, 1000] 区间。为啥是这个数?因为它要被当 text token 喂给 LLM,LLM 处理离散 token 比处理连续浮点数自然。这个 trick 来自 Shikra 那篇 paper。

---

## SrCoT ——这篇 paper 最核心的东西

SrCoT 全称是 Spatial Relationship-Focused Visual Chain-of-Thought。

直接让 VLM 吐坐标会 overfit。因为坐标跟图片像素的映射关系太复杂了,数据又不够,模型学到的全是死记硬背。

这帮人观察人类怎么干活:你说"把西兰花放锅里",人脑子里其实是这么转的——先看到西兰花在哪、锅在哪、它俩啥关系(西兰花在锅右边),然后规划路径(先抬起来、再往左移、再放下去)。每一步都 reference 物体的具体位置。

SrCoT 就是把这种人类推理过程强制塞进 VLM 里。它分两步:

**第一步 Description**:模型先描述场景,把每个物体的坐标和物体间的关系标出来。比如:
```
<ref>green plate</ref><box>[[264, 324, 504, 516]]</box>
The carrot is <pred>right</pred> of the green plate
```

这就建立了一个 spatial relationship graph——节点是物体(带坐标),边是关系(上下左右)。

**第二步 Reasoning**:用这个 graph 当 anchor,一步步推导轨迹。比如:
```
Step 1: carrot 在 (680, 708)
Step 2: 先抬起来到 (663, 663) 避免撞东西
Step 3: 往左移到 (607, 560)
...
Step n: 放到盘子上 (390, 416)
```

最后把所有点拼起来就是 visual trace。

**关键 intuition**:VLM 直接从 image+instruction 映射到 16 维坐标(8 个点 × 2D),这个 mapping 太复杂了。但如果你把它分解成 8 步,每步只推一个点,每步都基于已知的 spatial graph,每步的搜索空间就小多了。这就像 ReAct 之于 tool use——把 hard task 拆成 reasoning + grounding 的交替序列。

Ablation 数据很 dramatic:去掉 SrCoT,FSD 在 VABench-Point 上从 61.82% 掉到 26.21%,几乎跟 RoboPoint 一个水平了。这证明 SrCoT 是性能的全部来源,不是什么 data scale 或 architecture 改进带来的。

---

## 数据怎么造?

SrCoT 要求模型同时具备 grounding、空间理解、复杂指令跟随能力,主流 VLM 都不太行。所以他们搞了个 5 级渐进式数据 pipeline,从弱到强一步步训:

**Level 1 - Region Grounding (145k)**:让模型能在 caption 里嵌入物体坐标。用 GPT-4o 选 task-relevant 物体,GroundedSAM 提取 bbox,把坐标嵌进描述句子里。

**Level 2 - Spatial Relationship (86k)**:这个最 technically interesting。从 RGB 图直接推空间关系不准,所以他们绕了一圈:
- Metric3Dv2 做 depth estimation
- WildCamera + PerspectiveFields 估计相机内外参
- 把 2D 图反投影成 3D point cloud
- 从 point cloud 算物体间真实空间关系
- 再把这个关系 label 回到 2D 训练数据

巧妙之处在于:他们只生成 **relative depth sorting** label,不要求绝对深度精确。还专门挑 depth gap ≥ 20% 的物体对,避免 depth 噪声把 label 搞乱。这其实就是个 round-trip self-supervision——用 3D 信息生成 2D label,再让模型从 2D 推回 3D 能力。

**Level 4 - Spatial Affordance (24k)**:从 demonstration video 的 terminal frame 提取 manipulated object 的最终位置。先 GroundingDINO+GroundedSam 检测 mask,mask 的 bbox 就是 affordance box。然后对 mask 做 **erosion**(腐蚀)再采样 8 个点——erosion 是为了让采样点更靠中心,别采到 mask 边界外头去。

**Level 5 - Visual Trace (26k)**:两阶段——先用 self-supervised keypoint extraction 找 grasp points,再用 CoTracker 在视频上 track 这些点的运动轨迹。选最长的 trajectory,做 cubic spline 平滑,均匀采 8 个点。

数据质量控制很严:size threshold、trajectory length threshold 都要调,每个数据集先在 100 个样本上人工验证准确率 >95% 才正式跑全量。

---

## Self-Consistency ——个小但聪明的 trick

坐标这个 modality 在 VLM pretraining 数据里几乎没出现过,所以 LLM 其实不太理解坐标的物理含义。它可能只是死记硬背了"看到这种图片就吐这种数字"的 pattern。

FSD 的解法是 **双向训练**:
- Forward:图+指令 → 坐标(generation)
- Inverse:图+坐标 → 指令(understanding)

Inverse 任务强迫模型真的理解坐标在指哪里,不然反推不出 instruction。这就像 CycleGAN 的 cycle consistency——用反向任务当 auxiliary supervision,把 latent space 对齐到 meaningful geometry。

Ablation 显示去掉 alignment,VABench-Point 从 61.82% 掉到 55.92%,掉 6 个点。没有 SrCoT 那么戏剧性,但也是实打实的提升。

---

## 训练设置

架构就是标准 LLaVA-1.5:
- CLIP-ViT-L-336px 当 vision encoder(冻结)
- 2-layer linear projector(可训)
- Vicuna-13B 当 LLM(可训)
- 在 ASMv2 上继续微调(因为 ASMv2 已经有 basic grounding 能力)

两阶段:
- **Stage 1**:1.4M mixed data(838k general VQA + 295k spatial reasoning + 250k FSD Level 1-3),72 小时
- **Stage 2**:50k FSD Level 4-5,8 小时

Stage 2 只要 8 小时这个细节很有意思——说明 spatial reasoning foundation 建好后,visual aids generation 只是薄薄一层 capability,很容易 fine-tune 上去。反过来验证了 reasoning-driven 比 data-driven 更 sample-efficient。

8× A100 40G,batch 128,AdamW,lr 2e-5,3% warmup + cosine decay。

---

## 怎么执行?从 2D 坐标到机械臂动作

FSD 只输出 2D 坐标,但机械臂要在 3D 空间里动。怎么 bridge?

### Pinhole camera model 反投影

公式很标准:
$$s_i \begin{bmatrix} u_i \\ v_i \\ 1 \end{bmatrix} = K \begin{bmatrix} x_i \\ y_i \\ z_i \end{bmatrix}$$

$K$ 是相机内参矩阵,$(u_i, v_i)$ 是图像坐标,$(x_i, y_i, z_i)$ 是 3D 坐标,$s_i$ 是 depth 归一化因子。有 depth camera(他们用 RealSense L515)就能反推出 3D 坐标。

### 一个小优化:避免轨迹贴物体表面

直接用 raw depth 会有个问题:轨迹会紧贴物体表面,机械臂执行时容易撞。FSD 的解法是固定起点和终点的 depth(这两个点通常是 grasp point 和 place point,depth 可靠),只优化中间点的 depth,让总轨迹长度最小:
$$\hat{d}_i = \arg\min_{d_{2:T-1}} \sum_i d(\mathbf{P}_i, \mathbf{P}_{i+1})$$

用 scipy gradient descent 解。本质就是把"贴表面"问题转化成 trajectory smoothing 问题。

### Grasp pose 匹配

第一个点 $\mathbf{x}_1$ 通常是 grasp point,在 GraspNet 预计算的 grasp candidates 里找最近的:
$$G^* = \arg\min_{G \in \mathcal{G}} \|G, \mathbf{x}_1\|$$

### Motion Planning

- 用 visual trace 时:在 SE(3) 空间做 gradient descent interpolation
- 用 spatial affordance 时:用 CuRobo(NVIDIA 的 parallelized collision-free motion planner)

---

## 实验结果亮点

### Spatial Reasoning(8 个 benchmark)

FSD-13B 在 18 个 subtask 上平均 rank 1.3,跟 GPT-4o 打平,超越其他所有 13B VLM。3D depth perception 88.0%,3D distance estimation 86.7%。这主要归功于 Level 2 那套 depth-aware 数据 pipeline。

### Object/Free Space Reference

RoboRefIt 上 FSD 56.7%,GPT-4o 只有 15.3%。这点挺 striking 的——GPT-4o 在 fine-grained spatial grounding 上其实很弱,因为 web data 里几乎没有 pixel-level coordinate supervision。FSD 比 RoboPoint 高 7 个点,几乎全靠 SrCoT。

### VABench(他们自己提的 benchmark)

VABench-Point:FSD 61.82%,RoboPoint 19.09%,GPT-4o 9.30%。FSD 是 RoboPoint 的 3 倍。去掉 SrCoT 掉到 26.21%——这就是为什么我前面说 SrCoT 是全部来源。

VABench-VisualTrace:他们训了个 DINOv2+Transformer 的 end-to-end baseline 作对比(用同样数据),FSD 的 RMSE 是它的一半。这证明 **reasoning-driven > data-driven**,即使在相同数据量下。

### SimplerEnv(仿真,zero-shot)

FSD zero-shot 40.6% 平均成功率,超过 RT-1-X(1.1%)、OpenVLA(1.0%)、RoboVLM-ZS(13.5%)、Octo-S(30.0%)、SpatialVLA-ZS(34.4%)。比 fine-tuned SpatialVLA(42.7%)只低 2 个点。

注意 FSD 在某些 task 上不如 fine-tuned baseline——比如 Eggplant→Basket 上 SpatialVLA-FT 是 100%,FSD 只有 37.5%。但在 Spoon→Towel 和 Carrot→Plate 上反超。zero-shot reasoning 的 strength 在 unseen task 泛化,weakness 在特定 task 极致优化。这是合理的 trade-off。

### Real-World(xArm 6,8 个 task)

zero-shot 72% 成功率,超最强 baseline 30%。最 impressive 的是 cloth folding(叠布),baseline 完全做不了——因为它们只能预测 start/end points,叠布需要完整 trajectory。FSD 用 visual trace 提供了完整 guidance。

---

## 跟相关工作的对比

**vs LLaRVA**:LLaRVA 也预测 visual trace,但需要 task-specific fine-tuning,泛化不行。FSD 用 reasoning 替代 brute-force supervision,实现了 zero-shot。

**vs EmbodiedCoT**:在 OpenVLA 上加 CoT,但还是直接 output action。FSD 不 output action,只 output visual aids,然后接传统 motion planner。

**vs RoboPoint**:单步预测,无 reasoning chain。FSD 几乎全面碾压它,尤其 VABench-Point 上是 3 倍性能。

**vs RoboBrain**:RoboBrain 也做 visual trace,但是 agent-centric(机械臂末端轨迹)。FSD 是 object-centric(被操作物体轨迹),跨 embodiment 更泛化。

**vs SpatialVLA**:end-to-end VLA,需要 fine-tuning。FSD zero-shot 几乎打平它。

**vs RT-2/OpenVLA/π0**:纯 end-to-end,把感知+推理+控制全塞一个模型。优点 simple,缺点数据需求巨大、embodiment-specific、debug 难。FSD 选了模块化路线:VLM 做 reasoning → visual aids → 传统 motion planner 做控制。这种 modularity 让每个 component 都可以用最适合的方法优化。

---

## 局限与潜在问题

作者自己列了三个:
1. **Long-horizon tasks**:现在主要针对 explicit instructions,复杂长任务需要 instruction decomposition。
2. **Downstream execution**:用 training-free motion planner,在 dynamic 场景可能成 bottleneck。可以让 visual aids 作为下游 VLA 的 explicit guidance,替代 language conditioning。
3. **2D → 3D**:2D 利用了 VLM 的 REC-style 能力,但复杂场景 3D 可能更有效(像 ReKep 那种方向)。

我个人还想补充几个:
- **Depth camera 依赖**:FSD 执行依赖 RealSense L515 这种 depth camera,没 depth 就做不了 2D→3D mapping,deployment 受限。
- **Static scene assumption**:SrCoT 的 spatial graph 是单帧推出来的,如果场景 dynamic(物体被人推动、机械臂操作过程中其他物体变化),graph 会失效。可能需要 temporal extension。
- **8 个点的固定长度**:visual trace 固定 8 个点,简单 task 浪费,复杂 task 不够。可以做成 adaptive length。
- **Object-centric 的 grasp 选择**:用 GraspNet 找最近 grasp,但如果 GraspNet 候选里没有合适的(比如 deformable object、transparent object),就挂了。cloth folding 这种可能就是用预设 grasp 凑合过去的。

---

## 这 work 的本质直觉

我觉得 FSD 的成功可以归结为一个 meta-insight:**当数据不够时,reasoning 是 data 的 substitute**。

纯 end-to-end VLA 的逻辑是"我有足够数据,让模型自己学 mapping"。FSD 的逻辑是"数据不够,所以我用 reasoning 把 hard mapping 分解成简单 sub-problems,每个 sub-problem 都落在模型已有 capability 内"。

这跟你和 Yann 这些人一直讲的 "system 2 thinking"、"deliberative reasoning" 思路一致——当 system 1(快思考、pattern matching)不够用时,call system 2(慢思考、explicit reasoning)。SrCoT 本质就是给 VLM 装了个 spatial domain 的 system 2。

**跟 Anthropic tool use 范式的类比**:LLM 不直接做计算,而是 call calculator tool。FSD 不直接做 action prediction,而是 call motion planner tool。中间的"reasoning"是 LLM 该干的事。

**潜在延伸**:
- SrCoT 可以扩展成 test-time self-verification:模型生成 visual aids 后,用 inverse task 验证 aids 跟指令是否一致,不一致就 re-generate。这是 self-consistency CoT 在 spatial domain 的应用。
- 可以把 spatial graph 扩展成 spatio-temporal graph,预测物体状态变化,支持 dynamic scene。
- 用 diffusion policy 替代 CuRobo 执行 visual aids,diffusion 能建模 multi-modal action distribution,在 cluttered 场景可能更鲁棒。

---

## 一句话总结

**FSD 把 VLA 的 action prediction 问题重新 cast 成 spatial reasoning 问题,用 SrCoT 把 hard coordinate prediction 分解成 multi-hop spatial reasoning chain,借传统 motion planner 执行,绕过了 data scarcity 和 embodiment heterogeneity 两大坑**。

核心 innovation 就是这个 reframing + reasoning chain 的设计。架构、训练、数据 pipeline 都是围绕这个核心 idea 的工程支撑。

希望这个人话版本帮你 build intuition 了。如果你想再深入聊某个细节——比如 SrCoT 的 prompt 模板、depth optimization 的几何含义、或者跟 ReKep / π0 / Diffusion Policy 这些工作的更深对比——随时说。

---

# FSD (From Seeing to Doing) 深度解析

Karpathy 你好,这篇 paper 来自 Tianjin University 的工作,我尽量从 first principles 帮你 build intuition。项目的链接如下:
- Project page: https://embodied-fsd.github.io/
- Code: https://github.com/pickxiguapi/Embodied-FSD
- Datasets: https://huggingface.co/IfYuan

---

## 1. 核心问题与动机

当前 VLA (Vision-Language-Action) models 如 RT-2, OpenVLA, π0 的核心 assumption 是:VLM 在 internet-scale data 上 pretrain 获得的 generalization 能力能 transfer 到 robotic manipulation。但实证证据(Zheng et al. 2024, Zawalski et al. 2024, Liu et al. 2024c)显示这种 transfer 在 completely novel tasks 上**严重掉链子**。

FSD 作者诊断出两个 root causes:

**(1) Data scarcity vs scaling law mismatch**
Robotics data 相比 language/vision data 还差几个数量级,无法触发类似 Kaplan et al. 2020 提出的 scaling laws。Open X-Embodiment 等数据集虽有增长,但 coverage 和 diversity 远不够。

**(2) Embodiment heterogeneity**
不同 robot (WidowX, xArm, Franka, Kuka...) 的 action space 维度、关节配置、动力学差异巨大,导致同一个 task 在不同 embodiment 下的 action label 几乎不可比。Wang et al. 2024 (HPT) 也讨论过这个问题。直接做 end-to-end supervised learning from (vision, language) → diverse action outputs 容易 overfit 到某个 embodiment。

**FSD 的核心 insight**:与其强行让模型直接预测 raw action,不如先预测一个 **embodiment-agnostic 的 mid-level representation**——即 visual aids (空间坐标 + 视觉标记)。这样 VLM 的 general visual understanding 能力可以直接复用,而不必强行吸收 embodiment-specific 的 action manifold。

这个思路和 LLaRVA (Niu et al. 2024), RoboPoint (Yuan et al. 2024b), EmbodiedCoT (Zawalski et al. 2024) 在精神上一致,但 FSD 的差异化在于:不把 visual aids 当作一个新的 prediction target 去做监督学习,而是当作 **reasoning 的产物**——通过 spatial chain-of-thought 推导出来。

---

## 2. Visual Aids 的数学定义

所有坐标定义在 normalized image space:
$$\mathbf{x} = (p, q) \in [0, 1000]^2 \subset \mathbb{R}^2$$

这里 [0, 1000] 是离散化区间(整数化),将连续像素坐标映射到固定的 token vocabulary 上,让 LLM 可以像处理 text token 一样处理坐标。这个 trick 来自 Shikra (Chen et al. 2023)。

三种 visual aids:

**Spatial affordance box**:
$$\mathbf{B} = [x_1, y_1, x_2, y_2]$$
表示放置目标物体的 free region (如"把寿司放进银锅",锅内部的 free space)。

**Spatial affordance points**:
$$\mathcal{P} = \{(x_i, y_i) \mid i = 1, 2, ..., n\}$$
更精细的放置点集合,减少 box 的冗余。

**Object-centric visual trace**:
$$\tau = \{\mathbf{x}_t \mid t = 1, 2, ..., T\}$$
其中 T 是序列长度,FSD 固定 T=8(通过 cubic spline 插值)。

**关键 design choice**:用 2D 而非 3D,原因是高质量 3D data 稀缺 (Zhang et al. 2024)。Object-centric 而非 agent-centric 避开了 embodiment heterogeneity 问题——轨迹描述的是被操作物体的运动,而非机械臂末端,这样可以 transfer 到任何 robot。

---

## 3. SrCoT (Spatial Relationship-Focused Visual Chain-of-Thought)

这是 FSD 最有意思的部分。直接 SFT (vision, language) → coordinates 容易 overfit。作者从人类认知出发:人执行"把西兰花放进锅里"时,先定位物体,再根据相对位置规划路径,中间不断 reference 物体位置建立 spatial relationships。

SrCoT 分两阶段:

### Phase 1: Description
生成 object-centric region captions,建立 **spatial relationship graph** $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
- Nodes $\mathcal{V}$:objects with coordinates
- Edges $\mathcal{E}$:relative relationships (above, below, left, right, ...)

输出格式(从 paper Appendix A 摘录):
```
The image shows a <ref>green plate</ref><box>[[264, 324, 504, 516]]</box>...
The carrot<box>[[553, 506, 751, 844]]</box> is currently <pred>right</pred> of the green plate.
```

`<ref>` 标记 object name,`<box>` / `<point>` 标记坐标,`<pred>` 标记 predicate。这种 explicit binding 解决了 VLM "hallucinate 坐标"的问题——每个坐标必须 anchor 到具体 object。

### Phase 2: Reasoning
用 spatial relationship graph 作为 anchor points,确定 start/end 坐标,然后 **iteratively 推导中间点**,每步都有 explicit logical connection。

例如 Figure 9 的推理:
```
Step 1: start by identifying the current position of the carrot at <point>[[680, 708]]</point>
Step 2: First, lift the carrot slightly upwards to <point>[[663, 663]]</point> to clear any obstacles
...
Step n: place the carrot on the plate at <point>[[390, 416]]</point>, within <box>[[208, 437, 440, 521]]</box>
Answer: <point>[[680, 708], [663, 663], [607, 560], ..., [390, 416]]</point>
```

**Intuition**:VLMs 直接从 image+instruction → 未来轨迹坐标 很难,因为这种映射高度 nonlinear 且训练信号稀疏。但如果把问题 decompose 成"在已知 spatial graph 上做 multi-hop analysis",每个 hop 只需要做局部 reasoning(start point → 障碍物规避点 → ... → end point),每个子问题都落在 VLM 已有的 capability 内。本质上这是把 spatial 问题转化成 LLM 已经擅长的 "graph reasoning" 问题。

类比一下:这就像 ReAct (Yao et al. 2023) 之于 LLM tool use,把一个 hard task 分解成 reasoning + grounding 的交替序列。

---

## 4. Weak-to-Strong 数据 Pipeline

SrCoT 要求 VLM 同时具备:
- 精确的 reference grounding
- 空间理解
- complex instruction following
- 直接预测未来轨迹

主流 VLM 在这些能力上都有短板,所以作者设计了 **5-level 渐进式数据 pipeline**:

| Level | Capability | 数据规模 |
|-------|-----------|---------|
| 1 | Region Grounding | 145k |
| 2 | Spatial Relationship | 86k |
| 3 | Spatial Reasoning QA | 19k |
| 4 | Spatial Affordance Generation | 24k |
| 5 | Visual Trace Generation | 26k |

总计 300K SFT data across 10+ embodiments,来源是 BridgeDataV2 (Walke et al. 2023), RT-X (O'Neill et al. 2023), Droid (Khazatsky et al. 2024)。

### 自动化数据构造的技术细节

**Level 1 (Region Grounding)**:
- GPT-4o 根据任务指令 nominate task-relevant objects,排除 out-of-range 或过复杂 items
- GroundedSAM (Ren et al. 2024) 提取 bounding boxes + segmentation masks
- 将坐标嵌入到 caption 中形成 grounded caption

**Level 2 (Spatial Relationship)**:
这是最 technically interesting 的部分。直接从 RGB 推 spatial relation 不准,所以:
1. **Metric3Dv2** (Hu et al. 2024) 做 depth estimation
2. **WildCamera** (Zhu et al. 2024) + **PerspectiveFields** (Jin et al. 2023) 估计 camera intrinsics/extrinsics
3. 2D RGB → 3D point cloud(用 pinhole camera model 反投影)
4. 基于 object segmentation 提取每个 object 的 3D 位置和 size
5. 计算 spatial relationships → 导出 spatial relationship graph

**关键 design 决策**:只生成 **relative depth sorting** data,所以对绝对 depth 精度要求不高。为提升质量,只选 depth gap ≥ 20% 的 objects 对——这避免了 depth 噪声主导 relation label。

**Level 4 (Spatial Affordance)**:
从 terminal frame 提取 manipulated object 的最终位置 → 结合 reference object 定位 → 计算 affordance region → re-render 到 initial frame。

具体实现(Appendix A):
1. GroundingDINO + GroundedSam 检测 terminal frame 中 manipulated_object 的 mask
2. mask 的 bbox = Affordance Box
3. 对 mask 做 **erosion**(腐蚀)操作,缩小面积,使得采样点更靠中心
4. 在 eroded mask 内均匀采样 8 个点 = Affordance Points

erosion 这个 trick 很巧妙,避免采样到 mask 边缘的点(往往在物体实际边界外)。

**Level 5 (Visual Trace)**:
两阶段:
1. **Self-supervised keypoint extraction** (Huang et al. 2024) 识别 grasp points
2. **CoTracker** (Karaev et al. 2024) 在 video sequence 上跟踪这些 points 的 temporal dynamics
3. 选最长 trajectory 作为代表
4. Cubic spline interpolation 平滑
5. 均匀采样 8 个点 = Visual Trace

**数据质量控制**:严格 rule-based filtering(size thresholds, trajectory length thresholds),每个数据集先调规则,在 100 个样本上人工验证准确率 > 95% 才正式生成。

---

## 5. Self-Consistency Mechanism

坐标在 VLM pretraining 数据中几乎没出现过(这是 LLaVA 系列的通病),coordinate space 与 image-text modality 之间存在 alignment gap。

FSD 的解法是 **bidirectional training**:
- Forward (generation): $(X_v, X_q) \rightarrow \tau$
- Inverse (understanding): $(X_v, \tau) \rightarrow X_q$

即:给一张图和一组 visual traces,让模型反推可能的 task instruction。这种"反过来问"的方式 forcing 模型理解 coordinate 的 physical meaning,而不是死记硬背 generation pattern。

这个思路和 cycle consistency (CycleGAN, Zhu et al. 2017)、bidirectional translation 一脉相承,本质上是用 inverse task 作为 auxiliary supervision,把 latent space 对齐到 meaningful geometry。

Ablation 数据(Table 3):
- FSD (full): 61.82% accuracy on VABench-Point
- w/o SrCoT: 26.21% (掉一半多)
- w/o Alignment: 55.92% (掉 6 个点)

SrCoT 是最重要的 component,alignment 是 secondary 的 polish。

---

## 6. 架构与训练

### 架构(标准 LLaVA-1.5 范式)
- **Vision encoder**: CLIP-ViT-L-336px (Gao et al. 2024),frozen
- **Projector**: 2-layer linear,trainable
- **LLM**: Vicuna-13B (Zheng et al. 2023b),trainable
- **Foundation**: 在 ASMv2 (Wang et al. 2025) 上继续微调,因为 ASMv2 已具备 basic relation conversation + reference grounding

### 两阶段训练

**Stage 1: General Spatial Reasoning Enhancement**
- 1.4M mixed samples
- 838k general VQA(ShareGPT4V, VQAv2, OCR-VQA, Visual7W, RefCOCO, VG, AS-Core, TextVQA 等)
- 295k general spatial reasoning(LLaVA-OneVision, RoboPoint, SpatialBot, SAT 训练集)
- 250k FSD Level 1-3

混合 general VQA + spatial data 是关键——避免 catastrophic forgetting,FSD 既要保留 general instruction following,又要获得 spatial capability。

**Stage 2: Visual Aids Generation & Understanding**
- Level 4 (24k) + Level 5 (26k),共 50k
- 包含 forward + inverse 任务
- 训练 3 epochs

**Hyperparameters**:
- Batch size: 128
- Optimizer: AdamW, $\beta_1 = 0.9$, $\beta_2 = 0.999$, weight decay = 0
- Learning rate: $2 \times 10^{-5}$,3% warmup + cosine decay 到 0
- Image resolution: 336×336
- Hardware: 8× A100 40G
- Stage 1: 72 hours,Stage 2: 8 hours

Stage 2 只需要 8 小时这一数据点很重要——意味着 spatial reasoning 是更难学的能力(需要 72 小时),而 visual aids generation 一旦有了 spatial reasoning foundation,只需少量数据 + 短时间就能 fine-tune 上去。这反过来验证了"reasoning-driven 比 data-driven 更 sample-efficient"的假设。

---

## 7. Action Execution 细节

FSD 输出 2D 坐标,需要 mapping 到 3D 空间才能控制机械臂。

### Pinhole Camera Model 反投影

$$s_i \begin{bmatrix} u_i \\ v_i \\ 1 \end{bmatrix} = \underbrace{\begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}}_{K} \begin{bmatrix} x_i \\ y_i \\ z_i \end{bmatrix}$$

变量解释:
- $(u_i, v_i)$:第 $i$ 个 keypoint 的 image pixel coordinate
- $(x_i, y_i, z_i)$:对应的 3D Cartesian coordinate
- $K$:camera intrinsic matrix
- $f_x, f_y$:focal length(像素单位)
- $c_x, c_y$:principal point
- $s_i = d_i / \text{depth\_scale}$:normalized depth,从 depth camera 读到的 raw depth value $d_i$ 除以 depth_scale

由此得到 3D trajectory:
$$\tau^{3d} = \{\mathbf{x}_t^{3d} \mid t = 1, 2, ..., T\}$$

### 深度优化(避免轨迹贴着物体表面)

直接用 raw depth 会导致轨迹紧贴物体表面,不实用。FSD 的解法:

**固定** $d_1$ 和 $d_T$(起点和终点深度,通常是抓取点和放置点,深度可靠)
**优化** $d_{2:T-1}$(中间点深度)

目标函数:
$$\hat{d}_i = \arg\min_{d_{2:T-1}} \sum_i d(\mathbf{P}_i, \mathbf{P}_{i+1})$$

其中 $d(\mathbf{P}_i, \mathbf{P}_{i+1})$ 是 3D Euclidean distance。用 scipy 的 gradient descent 求解。

这是一个**最小化总轨迹长度**的目标——让中间点尽量平滑过渡,在 3D 空间中走一条 short path。这个 trick 本质上是把"轨迹贴表面"问题转化为一个 trajectory smoothing 问题。

### Grasp Pose 匹配
基于 first point $\mathbf{x}_1$ 的 spatial position,在 **GraspNet** (Fang et al. 2020) 预计算的 grasp candidates $\mathcal{G}$ 中找最近 grasp pose:
$$G^* = \arg\min_{G \in \mathcal{G}} \|G, \mathbf{x}_1\|$$

### Motion Planning
- 用 visual trace 时:gradient descent-based interpolation in SE(3) space
- 用 spatial affordance 时:**CuRobo** (Sundaralingam et al. 2023) 做 motion planning

CuRobo 是 NVIDIA 出的 parallelized collision-free motion planner,速度很快。

---

## 8. VABench 设计

作者发现现有 benchmark 缺失:
- Where2Place 只有 100 张图,指令简单
- 没有 visual trace 的 benchmark

所以提出 **VABench**,300 manually annotated 问题,来自 OXE, BridgeData, Droid, Libero。

### VABench-Point
- Metric: predicted point 落在 target region 的比例
- 对只输出 bbox 的 model:在 bbox 内 uniform sample 9 个点取中心

### VABench-VisualTrace
Ground truth $\tau = \{\mathbf{x}_t \mid t = 1, 2, ..., T\}$,预测 $\hat{\tau} = \{\hat{\mathbf{x}}_t \mid t = 1, 2, ..., \hat{T}\}$。

**MAE**:
$$\text{MAE} = \frac{1}{T} \sum_{t=1}^{T} \|\mathbf{x}_t - \hat{\mathbf{x}}_t\|$$

**RMSE**:
$$\text{RMSE} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} \|\mathbf{x}_t - \hat{\mathbf{x}}_t\|^2}$$

当 $T \neq \hat{T}$ 时插值对齐,坐标归一化到 $[0, 1000]^2$。

**GPT Score**:由于一条 instruction 有多个 valid 轨迹,纯几何距离不够,所以用 MLLM(GPT-4.1)对可视化轨迹打 1-10 分,基于三个 criteria:
1. Task alignment and success(最重要)
2. Feasibility(物理合理性)
3. Obstacle avoidance

这个 metric 设计有借鉴 VLM-as-a-judge (Zheng et al. 2023) 的思路,但加了 task-specific criteria。

---

## 9. 实验结果深度解读

### 9.1 Spatial Reasoning (Table 1)

FSD-13B 在 18 个 subtasks 上平均 rank 1.3,超越其他 13B VLMs,与 GPT-4o 抗衡。

关键数据点:
- 3D depth perception: 88.0%
- 3D distance estimation: 86.7%
- Spatial relationship: 78.3%
- CVBench Average: 80.9%

**为什么 FSD 在 3D depth/distance 上这么强?** 因为 Level 2 数据 pipeline 用 Metric3Dv2 做了 depth-aware spatial relation labeling,模型学到了把 2D 视觉 cues 推回 3D 几何关系的能力。这种"用 3D 信息生成 2D 训练数据,再让模型从 2D 推回 3D"的 round-trip 训练,本身就是一种 self-consistency。

### 9.2 Object/Free Space Reference (Table 2)

- RoboRefIt: FSD 56.7%, GPT-4o 15.3%, RoboPoint 49.8%
- Where2Place: FSD 45.8%, RoboPoint 46.0% (持平)

GPT-4o 在 RoboRefIt 上只有 15.3% 这点很 striking——closed-source VLM 在 fine-grained spatial grounding 上其实很弱,因为它们主要在 web image-caption pair 上训练,缺乏 pixel-level coordinate supervision。

FSD 比 RoboPoint 高出 7 个点,主要归功于 SrCoT——通过 reasoning 链式推导出坐标,而不是单步预测。

### 9.3 VABench (Table 3)

**VABench-Point**:
- FSD: 61.82%
- RoboPoint: 19.09%(FSD 是它的 3 倍)
- GPT-4o: 9.30%
- w/o SrCoT: 26.21%(掉到接近 RoboPoint 水平)

这个 ablation 是论文最有说服力的证据之一:**SrCoT 几乎是 FSD 性能的全部来源**。去掉 SrCoT,FSD 就退化为一个普通的 visual grounding model,和 RoboPoint 同档。

**VABench-VisualTrace**:
- FSD: RMSE 78.26, MAE 63.44, GPT Score 6.21
- DINOv2 Predictor (end-to-end baseline): RMSE 128.32, MAE 117.49, GPT Score 4.01
- GPT-4o: RMSE 136.13, MAE 113.53, GPT Score 4.37

DINOv2 Predictor 是用同样的 visual trace 数据训练的 end-to-end transformer(visual encoder DINOv2 + language encoder T5-Base + transformer),它的 RMSE 比 FSD 高 64%。这证明了 **reasoning-driven 范式 > data-driven 范式**,即使在相同数据量下。

直觉解释:end-to-end model 要学一个 image+instruction → 8 个坐标的 mapping,这个 mapping 极其复杂(8 个 point × 2D = 16 维输出,且依赖几何约束);FSD 通过 SrCoT 把这个 mapping 分解成 8 步局部 reasoning,每步只需基于 spatial graph 推一个点,搜索空间大幅缩小。

### 9.4 SimplerEnv (Table 4)

SimplerEnv (Li et al. 2024e) 是为评估 real-world robotic manipulation 设计的仿真平台,基于 WidowX robot。8 个 tasks:

FSD zero-shot 平均 **40.6% success rate**,超越:
- RT-1-X: 1.1%
- OpenVLA: 1.0%
- RoboVLM (ZS): 13.5%
- SpatialVLA (ZS): 34.4%
- Octo-S: 30.0%

FSD 比 fine-tuned SpatialVLA (42.7%) 只低 2 个点,这是非常强的 zero-shot 表现。

**值得注意的是 FSD 在某些 task 上不如 fine-tuned SpatialVLA**,例如 Eggplant→Basket (SpatialVLA-FT: 100% vs FSD: 37.5%)。但在 Spoon→Towel (FSD: 41.7% vs SpatialVLA-FT: 16.7%) 和 Carrot→Plate (FSD: 50% vs SpatialVLA-FT: 25%) 上反超。这说明 zero-shot reasoning-driven 方法的 strength 在于 unseen task 的泛化,weakness 在于特定 task 的极致优化——这是合理的 trade-off。

### 9.5 Real-World (xArm 6, 8 tasks)

FSD zero-shot **72% success rate**,超最强 baseline 30%。

注意一个细节:visual trace 用于 sponge 和 folding tasks(需要轨迹规划),affordance points 用于其他 tasks。这说明 FSD 是一个 **multi-tool 框架**——根据 task 类型选择不同的 visual aid。

最 impressive 的是 cloth folding(叠布),这是 baseline 完全做不了的,因为 baseline 只能预测 start/end points,而 cloth folding 需要完整 trajectory。FSD 通过 visual trace 提供了完整 trajectory guidance。

---

## 10. FSD vs 相关工作

### vs LLaRVA (Niu et al. 2024)
LLaRVA 也预测 visual traces 来 align visual 和 action space,但用了大量 task-specific fine-tuning,泛化到新 task 困难。FSD 用 reasoning 替代 brute-force supervision,实现了 zero-shot。

### vs EmbodiedCoT (Zawalski et al. 2024)
EmbodiedCoT 在 OpenVLA 上 fine-tune 加入 CoT 中间推理,但仍然 output action。FSD 不 output action,只 output visual aids,然后接 traditional motion planner。

### vs RoboPoint (Yuan et al. 2024b)
RoboPoint 是 spatial affordance prediction 的 VLM,但单步预测,无 reasoning chain。FSD 在 RoboRefIt 上比 RoboPoint 高 7 个点,在 VABench-Point 上是它的 3 倍,几乎全部归功于 SrCoT。

### vs RoboBrain (Ji et al. 2025)
RoboBrain 也生成 visual trace,但是 **agent-centric**(机械臂末端轨迹),FSD 是 **object-centric**(被操作物体轨迹)。Object-centric 在 heterogeneous embodiment 下更泛化,因为不依赖机械臂几何。Appendix I 的对比显示 FSD 的 zero-shot 轨迹更准确。

### vs SpatialVLA (Qu et al. 2025)
SpatialVLA 是 end-to-end VLA,带 spatial representation。需要 fine-tuning 才能达 42.7%。FSD zero-shot 40.6%,几乎打平。

---

## 11. 局限与未来方向

作者在 Appendix J 列出:

1. **Long-horizon tasks**:FSD 主要针对 explicit instructions,长 horizon 复杂任务需要 instruction decomposition 到 atomic sub-tasks,每个 sub-task 生成 visual aids。
2. **Downstream execution**:当前用 training-free motion planner,在 dynamic 场景下可能成 bottleneck。可以让 visual aids 作为下游 VLA 的 explicit guidance,替代 language conditioning(Bharadhwaj et al. 2024, Zheng et al. 2024 已经初步验证 affordance/trajectory 比 language prompt 更有效)。
3. **2D → 3D**:2D visual aids 利用了 VLM 的 REC-style 能力,但 task complexity 提高时 3D visual trace 可能更有效。这是 ReKep (Huang et al. 2024) 等工作的方向。

我个人觉得还有一个潜在问题:**depth camera 依赖**。FSD 的 action execution 假设有 depth camera(RealSense L515),如果没有 depth 信息,2D → 3D mapping 就做不了。这限制了 deployment 的 universality。

---

## 12. 我的整体 intuition

FSD 的核心贡献在于:**把 VLA 的 action prediction 问题重新 cast 成 spatial reasoning 问题**。这个 reframing 带来三个好处:

1. **Sample efficiency**:VLM 已经有 spatial understanding capability(虽然弱),SrCoT 把它"激活"出来,而不是从零学一个 action head。

2. **Embodiment agnostic**:visual aids 描述物体运动,不描述机械臂运动,所以 transfer 跨 embodiment。这是 FSD 最深的 insight——把 embodiment-specific 的部分推给传统 motion planner(CuRobo, GraspNet),让 neural network 只做它擅长的部分(semantic + spatial reasoning)。

3. **Compositional generalization**:SrCoT 把任务分解成 description + 多步 reasoning,每步是局部决策,这天然支持 compositional generalization 到新组合。

**与 RT-2 / OpenVLA / π0 等纯 end-to-end VLA 的对比**:
纯 end-to-end VLA 把感知、推理、控制全塞进一个 model,优点是 simplicity,缺点是数据需求巨大、embodiment-specific、debugging 困难。FSD 选择了一个**模块化**路线:VLM 做 reasoning → visual aids → 传统 motion planner 做控制。这种 modularity 让每个 component 都可以用最适合的方法优化(VLM 用 LLaVA-style instruction tuning,motion planning 用 CuRobo 这种 GPU-parallelized 几何 planner)。

**这个 work 让我联想到**:Anthropic 的 Tool Use 范式(LLM + tools),以及你和同事的 nanoGPT / minGPT 哲学——简单、模块化、可解释的 component 比一个大型 monolith 更可控。

**潜在延伸方向**:
- SrCoT 可以扩展到 multi-modal reasoning:不只是 spatial graph,还可以加入 temporal graph(预测物体随时间的状态变化)。
- Self-consistency 机制可以扩展成 test-time 自我验证:模型先生成 visual aids,再用 inverse task 验证 aids 是否 consistent,不一致就 re-generate。这是类似 self-consistency CoT 在 spatial domain 的应用。
- 用 diffusion policy 替代简单的 motion planner 来执行 visual aids——可能比 CuRobo 更鲁棒,因为 diffusion 可以学习 multi-modal action distribution。

希望这些分析对你 build intuition 有帮助!如果你想深入讨论某个 component(比如 SrCoT 的 prompt 设计、depth optimization 的几何含义、或 VABench 评估的细节),我可以再展开。
