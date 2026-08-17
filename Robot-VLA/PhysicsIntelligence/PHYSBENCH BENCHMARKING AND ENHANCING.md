---
source_pdf: PHYSBENCH BENCHMARKING AND ENHANCING.pdf
paper_sha256: 0ad58867a2880bfa94883f482895b5bb4ddf4979fb0f22361047060a7d39a6cb
processed_at: '2026-08-06T03:13:03-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 PhysBench

Andrej，我换个风格，像咱们在 NeurIPS 走廊上喝咖啡聊天那样讲。

---

## 这篇 paper 到底在干嘛

一句话总结：**它证明了当前所有的 VLMs 都是"物理白痴"**。

你看啊，GPT-4o 在 MMMU 上能考 60+ 分，在 MMBench 上能考 70+ 分，看起来挺厉害。但给它一段视频问"球会掉进哪个罐子"，或者给它两张图问"哪个物体更硬"，它就崩了——PhysBench 上 GPT-4o 只有 49.5% 的准确率，人类是 95.9%。

这个 gap 太大了，大到说明这不是"模型不够大"的问题，而是**训练数据的根本性缺陷**。

---

## 为什么之前没人做这个 benchmark

其实有人试过，但都做得不够"狠"。

CLEVRER (2019) 是最早的物理推理 benchmark，但它用的全是渲染出来的简单几何体——红球撞蓝立方体这种。ComPhy (2022) 加了流体和绳子，但还是 simulation only。Physion (2021) 做得不错，但只有 17k 个样本，而且只测"会不会碰撞"这种 binary 判断。

这些 benchmark 的共同问题：

1. **太简单**——只有 cube, sphere, 几种基本形状
2. **太窄**——只测 collision 或 fluid 的一两个维度
3. **太 synthetic**——distribution 跟 real world 差太远
4. **没 interleaved**——物理属性本质上需要多帧或多图才能表达，单图根本不够

PhysBench 的 authors（USC + Berkeley + Toyota Research）觉得，要真正测 VLM 的物理理解，得搞一个 **comprehensive + real-world mix + interleaved format** 的 benchmark。于是他们花了 4000 个标注小时，搞了 10,002 个 QA pairs。

参考资料：
- [CLEVRER](https://arxiv.org/abs/1909.06525)
- [ComPhy](https://arxiv.org/abs/2205.04191)
- [Physion](https://arxiv.org/abs/2106.05803)

---

## PhysBench 测什么

四个大 domain，我一个个说：

### 1. Physical Object Property（物体属性）

这个最直觉——物体有 mass, color, number, 还有更抽象的 attribute（rigid, soft, elastic, brittle, sharp...）。

难点在于：很多属性**从单张图看不出来**。你怎么从一张图判断一个球是 rubber 还是 steel？你得看它碰撞后的形变，或者丢到水里的浮力。所以这个 domain 大量用 video 和 multi-image。

举例：给两个图，都说"施加相同的力，哪个物体 stiffness 更高？"——你得从形变量推断。

### 2. Physical Object Relationships（物体关系）

这个相对简单，VLM 表现也最好。size, location, depth, distance, motion。

为什么 VLM 在这个 domain 相对 OK？因为 2D 视觉特征够用了。你不需要物理先验就能判断"A 在 B 左边"。GPT-4o 在这个 domain 拿了 64.8%，是四个 domain 里最高的。

但 depth 和 distance 还是难——因为需要 metric 感知，不是单纯 semantic。

### 3. Physical Scene Understanding（场景理解）

**这是 VLM 的灾难区**。GPT-4o 只有 30.15%，比 random 的 25% 高不了多少。

测什么？温度、光照、气压、camera viewpoint。这些是**环境因素**，不是物体属性。

比如：
- "这个视频里是加了冷水还是热水？"（看水蒸气凝结模式）
- "camera 的 focal length 怎么变了？"（看 background compression）
- "光源从哪移到哪了？"（看阴影方向变化）
- "杯子里水位为什么变了？"（空气燃烧导致气压下降）

这些问题对人类来说不难——我们有丰富的物理常识。但 VLM 训练数据里几乎没有这类标注。

### 4. Physics-based Dynamics（物理动力学）

collision, throwing, manipulation sequence, fluid, chemistry, others。

比如：
- "哪个场景先发生？"（时间序列推理）
- "球最可能掉进哪个罐子？"（轨迹预测）
- "哪种液体粘度最低？"（流体属性推断）
- "正确的操作顺序是什么？"（manipulation sequence 排序）

GPT-4o 在这个 domain 46.99%——比 Scene 好多了，但仍然远低于人类 95.7%。

参考资料：
- [PhysBench examples](https://physbench.github.io/)

---

## 数据怎么来的

三个来源混合，这是这个 benchmark 设计上很聪明的点：

### Simulation（Blender）

679 个 3D assets + 470 个 HDR environment。关键是他们**同时存了 depth map, normal map, albedo map**——这些中间表征以后可以用来训练 model 生成物理中间表征，潜力很大。

为了确保 simulation 生成的图能被 detection model 识别，他们用 GroundingDINO 验证（box_threshold=0.2, text_threshold=0.2，grid search 调出来的），只有 detect 出来的物体跟生成的 label 完全匹配才保留。

### Web

Unsplash 6k 张图，nuScenes 自动驾驶数据 1356 个 QA（左转/直行/右转判断），DROID/Ego4D/MimicPlay 用来做 manipulation sequence。

### Real-world

iPhone 13 Pro Max 录的 RGBD 视频，主要覆盖 light/camera/collision 这几个 sub-task。

这个 mixed-source design 的好处：**既保证 ground truth 可控，又避免纯 synthetic 的 distribution shift**。

---

## 评测 setup 有点 tricky

他们把 model 分成三类来测：

- **Image VLMs**（LLaVA-1.5, BLIP-2, InstructBLIP）：只能看单张图。视频怎么办？把帧 concat 成一张大图喂进去（叫 "merge" 方法）
- **Video VLMs**（Video-LLaVA, Chat-UniVi, PLLaVA）：能看视频，帧按顺序输入（"seq" 方法）
- **General VLMs**（VILA, GPT-4o, Mantis, LLaVA-interleave）：支持多图 + interleaved 输入

**Caveat**：Image VLM 和 Video VLM 测的是 PhysBench 的子集（去掉 interleaved 数据），General VLM 测完整数据集。所以不同类别的分数不能直接横向比。

但即使如此，结论已经很清楚了。

---

## 最 striking 的发现：Scaling 在 PhysBench 上失效

这个我之前没强调够，但我觉得是整篇 paper 最重要的发现。

### Model size 不 work

VILA-1.5 从 3B 扩到 8B，在普通 VQA 上涨 7.1%，在 PhysBench 上**反而掉 3.8%**。

InstructBLIP 从 7B 扩到 13B，普通 VQA 涨 3%，PhysBench 涨 6%（但 base 太低，从 23.82 到 29.94）。

这个 pattern 在 Figure 6a 里很清楚：普通 QA 的曲线陡峭上升，PhysBench 的曲线基本平的甚至下降。

**直觉解释**：language model capacity 增加让你更会"说话"，但如果训练数据里没有物理知识，你只是更流畅地胡说八道物理。

### Data scaling 也不 work

PLLaVA 在 LLaVA-Next 基础上加了 783k 视频-文本数据，PhysBench 性能从 40.45 掉到 37.70。

VILA-1.5 训练数据比 LLaVA-1.5 多得多，PhysBench 从 40.45 掉到 37.15。

**为什么？** authors 做了 word cloud 分析（Figure 18-21）：
- LLaVA 训练数据高频词：'description', 'phrase', 'summary', 'region'
- PLLaVA 加了视频数据，出现了 'collides', 'camera' 这些词，但**只用来描述现象**，不解释机制
- 物理关键词 'direction', 'phenomenon', 'effects' 频率极低

**关键 insight**：Web caption 数据描述的是 *what*（"一个球在滚动"），不描述 *why*（"球在滚动是因为重力沿斜面分量"）。物理推理需要因果和 counterfactual，这种数据在 web caption 分布里几乎不存在。

### Frame scaling 也不 work

Figure 6c 测试 1/2/4/8 帧输入，三个开源 model 性能持平甚至下降。Mantis 超过 8 帧就直接 refuse to answer。

**直觉**：当前 video VLM 的 temporal modeling 是把 frame embedding 当 sequence-of-image 处理，没有真正的 dynamics modeling。它捕捉的是 appearance 变化（颜色、位置），不是 mechanics（速度、加速度、冲量）。

---

## Error Analysis 给了 PhysAgent 的设计 motivation

他们人工分析了 500 个错误案例，三个 model（GPT-4o, Gemini-1.5-flash, Phi-3V）的错误分布：

| 错误类型 | GPT-4o | Gemini-flash | Phi-3V |
|---|---|---|---|
| Perception Error | 37% | 40% | 45% |
| Lack of Knowledge | 34% | 35% | 23% |
| Reasoning Error | ~15% | ~15% | ~20% |
| 其他 | <15% | <10% | <12% |

**Perception error 占 37-45%**：VLM 看不清 depth，分不清 light direction，识别不准小物体。这指向 vision encoder 的 representation bottleneck——CLIP-style encoder 优化的是 image-text alignment，不是 metric depth 或 3D geometry。

**Lack of knowledge 占 23-35%**：模型不知道"全反射"、"气压与沸点关系"、"粘度与温度反相关"。

**Reasoning error 只占 15%**：LLM backbone 的 chain-of-thought 能力 OK，瓶颈不在推理能力。

**这个分布直接 motivate 了 PhysAgent 的两支设计**：perception 错就接 vision foundation models 补感知，knowledge 错就加 knowledge memory 补知识。

---

## PhysAgent 是什么

一个 inference-time framework，给 VLM 装两个"外挂"：

### 外挂 1：Vision Foundation Models

- **Depth Anything V2**：metric depth estimation
- **SAM**：segmentation
- **GroundingDINO**：open-vocabulary detection

这些 model 专门处理 VLM 不擅长的 perception 任务——depth 估计、精确物体定位、距离计算。

### 外挂 2：Knowledge Memory

针对不同 task type 激活不同的物理知识 prompt。比如问光照问题，就检索"光源移动与阴影方向关系"的知识塞给 VLM。

### 三步流程

```
Question 进来
  ↓
Step 1: 分类问题 + 激活 task-specific prompt 和物理知识
  ↓
Step 2: 调 foundation models 处理视觉信息
  - GroundingDINO 检测物体
  - Depth Anything 估计深度
  - SAM 分割
  ↓
Step 3: VLM 做 chain-of-thought reasoning + self-verification
  ↓
输出答案
```

### 效果

GPT-4o 在 PhysBench 上从 49.49% 提升到 **+18.4%**，特别是 Scene 子任务从 30.15% 跳到 49.5%。

**对比 baselines**：
- Chain of Thought 几乎无效——说明难度不在 multi-step reasoning
- Description-CoT 反而下降——先描述会 commit 到错误 perception
- Pure Language Reasoning（把图换成 description）灾难性崩溃——说明视觉信息 essential
- ContPhy 这种 neuro-symbolic pipeline 也不行——因为它用 R-CNN 处理视觉，对 GPT-4o 是 information loss

**PhysAgent 为什么 work 的 intuition**：它把 VLM 的 weakness（perception, knowledge）外包给 specialized tools，保留 VLM 的 strength（reasoning, generalization）。跟 Toolformer / ViperGPT 的 philosophy 一致，但加了 task-aware knowledge retrieval。

参考资料：
- [Depth Anything V2](https://arxiv.org/abs/2406.09414)
- [SAM](https://arxiv.org/abs/2304.02643)
- [GroundingDINO](https://arxiv.org/abs/2303.05499)
- [Toolformer](https://arxiv.org/abs/2302.04761)
- [ViperGPT](https://arxiv.org/abs/2303.08128)

---

## Embodied 验证：MOKA 实验

光在 benchmark 上涨分不够，得看能不能帮 robot。

他们用 MuJoCo + Franka 机械臂，5 个 task：affordance（抓物体）, force（抓 fragile/soft/rigid 物体）, color（抓特定颜色物体）, location（抓特定位置物体）, tool（选对工具）。

两种增强方式：
1. Fine-tune GPT-4o with PhysBench subset
2. Zero-shot 用 PhysAgent

结果（Figure 9c）：

| Task | Baseline | + PhysAgent | + Fine-tune |
|---|---|---|---|
| Affordance | 0.6 | 0.8 | 0.9 |
| Force | 0.2 | 0.5 | 0.6 |
| Color | 0.7 | 0.8 | 0.8 |
| Location | 0.7 | 0.7 | 0.8 |
| Tool | 0.4 | 0.5 | 0.7 |

**Force task 提升最大**（0.2 → 0.5 zero-shot, 0.6 fine-tune），因为 force task 需要理解 deformable/fragile/rigid 物体属性，正好是 PhysBench Attribute 子任务训练的能力。

**直觉**：Fine-tune 比 zero-shot PhysAgent 更好，但 PhysAgent 不需要训练数据——这是 trade-off。生产场景下 PhysAgent 更适合 zero-shot deployment。

参考资料：
- [MOKA paper](https://arxiv.org/abs/2403.03174)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)

---

## 我觉得 paper 没做好的地方

1. **PhysAgent 细节太模糊**：
   - Knowledge memory 是 hardcoded prompt library 还是真正的 RAG？paper 写得不清楚
   - Task classifier 是 manual 还是 learned？说 "manually or automatically" 太 hand-wavy
   - Self-verification 怎么实现？另一个 LLM call 还是 rule-based？

2. **Claude 评测有 confound**：
   - Claude-3.5-sonnet 图像被压到 128×128，这对需要细粒度视觉信息的物理任务不公平
   - Paper 应该 ablate 这个

3. **Video VLM instruction following 问题**：
   - Chat-UniVi-13B 只有 10.36%（比 random 低），这是 instruction following 失败，不是物理理解失败
   - 应该用 answer extraction LLM 兜底

4. **PhysAgent 的 OOD generalization 没测**：
   - 在 PhysBench-test 上 +18.4%，但在完全 OOD 的物理任务上呢？

5. **MOKA 实验的 task 太简单**：
   - 5 个 task 都是 single-step pick-and-place
   - 应该测 long-horizon manipulation

---

## 跟你的工作有什么连接

Andrej，我觉得 PhysBench 对 VLA (Vision-Language-Action) model 训练有几个直接启示：

### 1. Data composition > Data scale

VLA 训练不应该只 scale up demonstration 数据。应该 inject 物理知识——counterfactual reasoning, causal physics。

Web caption 描述的是 *what*（"手在抓杯子"），不描述 *why*（"因为杯子要被移动到桌子上"）。物理推理需要因果和 counterfactual，这种数据在 web caption 分布里几乎不存在。

### 2. Foundation models as perception backbone

与其让 VLA 从头学 depth estimation，不如像 PhysAgent 那样蒸馏 Depth Anything 的输出作为 auxiliary supervision。

这跟你的 nanoGPT philosophy 一致——**用强的 teacher 蒸馏出好的 representation**，而不是让 student 从头学。

### 3. Interleaved training data 是对的

VLA 应该训练在 interleaved (image_t, image_{t+1}, action) 而不是单帧，因为物理 dynamics 本质是时序的。

### 4. Scene understanding 是缺的拼图

当前 VLA 工作（OpenVLA, Octo, RT-2）都没 explicitly 评估 light condition, camera viewpoint 变化下的 robustness。PhysBench-Scene 子集可以做这个 evaluation。

### 5. Physics-grounded pretraining

这是终极方向。就像 LLaVA 用 GPT-4V 生成 instruction data 训练 next-gen VLM，应该用物理 simulator 生成 counterfactual 物理 QA 训练 next-gen physics-VLM。

物理 simulator 的优势：**infinite counterfactual data**。你可以同一个场景生成 1000 种变体（不同 mass, 不同 friction, 不同 elasticity），让 model 学会"属性 → outcome"的因果映射。这是 web data 永远无法提供的。

参考资料：
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [Octo](https://octo-models.github.io/)
- [RT-2](https://robotics-transformer2.github.io/)
- [LLaVA](https://arxiv.org/abs/2304.08485)

---

## 最重要的 takeaway

PhysBench 揭示的 core insight：

**VLMs 的物理理解缺失本质是 data composition 问题，不是 model capacity 问题。**

Web-scale caption 数据描述 *what* 不描述 *why*，缺乏 counterfactual 物理推理。Scaling up parameters 或 scaling up caption data 都解决不了——因为问题在数据分布，不在模型大小。

PhysAgent 是当前 pragmatic 最优解：用 tool use + knowledge memory 外包 perception 和 knowledge 两个 bottleneck。但长期看，**native physics-aware pretraining** 才是终极方案。

类比一下你的 nanoGPT 工作：LLM 的 reasoning 能力是从 next-token prediction 涌现的。物理理解的涌现可能需要类似的 "physics-grounded next-frame prediction" training objective——把 video prediction + physical property 预测作为 auxiliary loss。

这条路如果走通，可能是通往 embodied AGI 的关键一步。

---

## 公式补充：Pearson Correlation

Paper 用 Pearson 相关系数构造了 PhysBench 与 15 个其他 benchmark 的 correlation map（Figure 4a）：

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \cdot \sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

变量说明：
- $x_i$: 第 $i$ 个 model 在 benchmark X（比如 PhysBench-Scene）上的 score
- $y_i$: 第 $i$ 个 model 在 benchmark Y（比如 POPE）上的 score
- $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$: benchmark X 上所有 model score 的均值
- $\bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i$: benchmark Y 上的均值
- $n = 10$: model 数量（LLaVA-1.5-7B/13B, InstructBLIP-7B/13B, Qwen-VL-Chat, VILA-1.5-3B/3B-s2/8B/13B, BLIP-2，见 Table 23）
- $r \in [-1, 1]$: 接近 1 说明两个 benchmark 测同一能力维度，接近 0 说明 orthogonal

**关键发现**：
- PhysBench-Scene 与 POPE（hallucination detection）相关性最高 (~0.7)
- PhysBench-Dynamics 与传统 VQA（VQAv2, GQA）相关性低 (<0.3)
- 这证明 **PhysBench 测的是 orthogonal 于传统 VQA 的能力维度**

这个 correlation map 的 intuition 是：如果两个 benchmark 测同一能力，那在同一组 model 上分数应该正相关。如果测 orthogonal 能力，相关性低。

---

好，这就是用大白话讲的 PhysBench。核心就一句话：**当前 VLMs 的物理理解很烂，烂在训练数据不在模型大小，PhysAgent 用 tool use 能补一部分，但最终需要 physics-grounded pretraining 才能根治**。

---

# PhysBench 深度技术讲解

Andrej，这篇 paper 触及了当前 VLMs 一个非常本质的盲点——**spatial intelligence / physical world understanding**。我把它拆成几个 layer 讲，目的是帮你 build intuition about 为什么 VLMs 在 physical reasoning 上崩溃，以及 PhysAgent 这种 agentic 框架能不能补上这个洞。

---

## 1. Motivation & Positioning

Embodied AI 的关键 bottleneck 一直是 VLMs 的 "intuitive physics" 缺失。人类婴儿期就具备的 naive physics（McCloskey 1983, Carey 2000）——比如物体恒存性、重力直觉、刚体碰撞——VLMs 完全没有 internalize。这篇 paper 的核心 contribution 是把这件事量化：**PhysBench 是第一个 system-level 评估 VLMs 物理感知能力的 benchmark**，而不是物理 *知识*（physics exam）或 commonsense reasoning。

关键区分：
- 之前的 CLEVRER (Yi 2019)、ComPhy (Chen 2022)、Physion (Bear 2021)、ContPhy (Zheng 2024b) 都是 synthetic + 简单 primitive（cube, sphere）+ narrow task type。
- PhysBench 引入了 **interleaved video-image-text** 格式，覆盖 real-world capture + simulation + web，并要求模型真的"看"video 推断 elasticity, temperature, viscosity 等隐藏属性。

这一点我觉得是 essential design choice——因为单张 image 根本无法表达 "elasticity" 或 "mass" 这种只能从 dynamics 中推断的属性。

参考资料：
- [PhysBench project page](https://physbench.github.io/)
- [ContPhy (ICML 2024)](https://proceedings.mlr.press/v235/zheng24a.html)
- [Physion (NeurIPS 2021)](https://arxiv.org/abs/2106.05803)

---

## 2. Dataset 设计的 Intuition

### 2.1 四个 domain 的 taxonomy

| Domain | 子类 | 能力维度 |
|---|---|---|
| **Physical Object Property** | Number, Mass, Color, Attribute | Identify, Comparison |
| **Physical Object Relationships** | Size, Location, Depth, Distance, Motion | Static, Dynamic |
| **Physical Scene Understanding** | Temperature, Viewpoint, Air, Light | Prediction, Judgment, Reasoning, Perception |
| **Physics-based Dynamics** | Collision, Throwing, Manipulation, Fluid, Chemistry, Others | Prediction, Judgment, Reasoning, Perception |

注意 **Scene** 和 **Dynamics** 共享 4 个能力维度（perception/prediction/judgment/reasoning），但内容不同——Scene 是环境因素（光照、温度、气压），Dynamics 是事件（碰撞、流体、化学反应）。

### 2.2 数据统计直觉

总 10,002 entries，分布很关键：
- only image: 18.6%（1,766）
- only video: 44.8%（2,749）——这个比例非常大，说明大量任务必须依赖 temporal dynamics
- interleave: 20.1%（1,902）——这是 interleaved 多模态输入

平均 question length 16.5 words，平均 choice length 4.4 words——保持简洁，避免让 LLM 部分 dominate 推理。

数据来源混合：
- **Simulation (Blender)**：679 个 3D assets + 470 HDR environment，生成 depth/normal/albedo maps，用 GroundingDINO 验证 object detectable（box_threshold=0.2, text_threshold=0.2，grid search 调出来的）
- **Web**：Unsplash 6k 张，nuScenes 1,356 QA pairs，DROID/Ego4D/MimicPlay 用于 manipulation
- **Real-world**：iPhone 13 Pro Max RGBD 录制 light/camera/collision

**关键 intuition**：simulation 提供 ground truth 的 depth map, normal map, albedo，这意味着未来可以用 PhysBench 训练模型生成这些中间表征——这是 paper 没明说但 latent 的潜力。

---

## 3. Evaluation Setup 的技术细节

### 3.1 三种 model setup

这是 paper 里技术含量最高的部分之一：

```
Image VLMs (LLaVA-1.5, BLIP-2, InstructBLIP, ...)
  └── merge: 视频帧 concat 成单张图 (从左到右、从下到上)

Video VLMs (Video-LLaVA, Chat-UniVi, PLLaVA)
  └── seq: 帧按顺序输入

General VLMs (VILA-1.5, GPT-4o, Mantis, LLaVA-interleave)
  └── seq + interleaved text-image 支持
```

**关键 caveat**：Image VLM 和 Video VLM 评测时，interleaved 数据被剥离了，只在子集上测。只有 General VLM 在 full dataset 上测——这导致 Image/Video VLM 的分数和 General VLM 不严格可比。这一点 paper 写得有点模糊，需要警惕。

### 3.2 Hyperparameter 细节

- 大多数 open-source: `torch_dtype=torch.float16`
- InternVL: `bfloat16, max_new_tokens=512, num_beams=1`
- GPT-4o: `max_new_tokens=300, temperature=0, seed=42`
- Claude-3.5-sonnet 图像 resize 到 128×128（cost 考虑）——这个细节很重要，因为 Claude 表现差可能部分来自 resolution loss
- GPT-4V 图像 resize 到 512×512

**My intuition**：Claude 系列在 PhysBench 上没显示优势，和它的 image resolution 被压到 128×128 有强相关。这个 confound paper 没充分讨论。

参考资料：
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
- [MOKA paper](https://arxiv.org/abs/2403.03174)

---

## 4. 主结果表 (Table 3) 的 Intuition

最关键的数据点：

| Model | Property | Relationships | Scene | Dynamics | Avg |
|---|---|---|---|---|---|
| Random | 25.0 | 25.0 | 25.0 | 25.0 | 25.0 |
| Human | 97.1 | 95.7 | 94.9 | 95.7 | 95.9 |
| **GPT-4o** | 56.91 | 64.80 | **30.15** | 46.99 | 49.49 |
| Gemini-1.5-pro | 57.26 | 63.61 | 36.52 | 41.56 | 49.11 |
| GPT-4o-mini | 53.54 | 44.24 | 30.59 | 42.90 | 43.15 |
| InternVL-Chat1.5 (best OS) | 53.08 | 70.14 | 37.01 | 44.78 | 47.51 |
| LLaVA-1.5-13B | 41.31 | 42.50 | 34.40 | 44.38 | 40.45 |

**核心 takeaways**：

1. **Scene 是最 catastrophic 的**：GPT-4o 才 30.15%，比 random 只高 5 个点。这说明 VLMs 完全不 understand light source 移动、camera focal length 改变、air pressure 变化这种环境因素。这是 VLM 的"暗物质"。

2. **Relationships 相对好**：GPT-4o 64.80%。这是因为 spatial relationship（size, location）可以用 2D 视觉特征直接 infer，不需要物理先验。

3. **Closed vs Open-source gap = 20.7%**：GPT-4o (49.49) vs LLaVA-interleave (41.00)。这个 gap 比 MMBench / MMMU 上大得多，说明物理理解是 closed-source model 训练 data 中 latent 包含的特权信息。

4. **Chat-UniVi-13B 崩溃到 10.36%**——甚至低于 random。这是 instruction following 失败，不是物理理解失败。Video VLM 的 instruction tuning 数据严重不足。

参考资料：
- [GPT-4o technical report](https://arxiv.org/abs/2303.08774)
- [InternVL](https://arxiv.org/abs/2312.14238)

---

## 5. Scaling Laws 在 PhysBench 上的失效（最重要的发现）

这是 paper 最 striking 的发现，我详细展开：

### 5.1 Model size scaling 失效

Figure 6(a) 数据：

| Model | Common QA (14 tasks avg) | PhysBench |
|---|---|---|
| VILA-1.5-3B | ~55% | 34.11 |
| VILA-1.5-8B | ~62% (+7.1%) | 32.85 (**-3.8%**) |
| VILA-1.5-13B | ~65% | 37.15 |
| InstructBLIP-7B | ~50% | 23.82 |
| InstructBLIP-13B | ~53% | 29.94 |

**直觉解释**：parameter count 增加 → language capacity 增加 → 在 descriptive QA 上提升；但物理知识是 *training data composition* 决定的，不是 *capacity* 决定的。More parameters don't help if the data doesn't contain physical priors.

### 5.2 Data scaling 失效

| Model | 训练数据 | PhysBench |
|---|---|---|
| LLaVA-1.5-13B | 665K instruction tuning | 40.45 |
| PLLaVA-13B (LLaVA-Next + 783K video) | +783K video-text | 37.70 (**-2.75%**) |
| VILA-1.5-13B (more data than LLaVA-1.5) | millions | 37.15 (**-3.3%**) |

**关键 insight from word cloud analysis** (Figure 18-21)：
- LLaVA-1.5 训练数据高频词：'description', 'phrase', 'summary', 'region'
- PLLaVA 包含 'collides', 'camera' 但只用于描述现象，不解释机制
- 物理关键词 'direction', 'phenomenon', 'effects' 频率极低

**My interpretation**：Data scaling 在 vision-language pretraining 上 saturate 的根本原因是 caption-style 数据的"语义稠密度"不够。Web caption 描述的是 *what* 不是 *why*。物理推理需要 counterfactual reasoning（"如果光从左侧移到右侧，阴影如何变化？"），这种数据在 web caption 分布里几乎不存在。

### 5.3 Frame scaling 失效

Figure 6(c) 测试了 frame 数 1/2/4/8：
- LLaVA-1.5, VILA-1.5, Mantis 三个模型，frame 增加性能持平甚至下降
- Mantis 超过 8 帧会 refusal 或不 follow instruction

**直觉**：当前 video VLM 的 temporal modeling 本质上是把 frame embedding 拼起来当 sequence-of-image 处理，没有真正的 temporal dynamics modeling。Video-LLaVA, PLLaVA 的 pooling / aggregation 机制对物理 dynamics 无效——它们捕捉的是 appearance 变化（颜色、位置），不是 mechanics（速度、加速度、碰撞冲量）。

参考资料：
- [Scaling laws for neural language models (Kaplan)](https://arxiv.org/abs/2001.08361)
- [LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/)

---

## 6. Error Analysis 的 6 类 taxonomy

人工分析 500 questions × 3 models (GPT-4o, Gemini-1.5-flash, Phi-3V)：

| Error Type | GPT-4o | Gemini-1.5-flash | Phi-3V |
|---|---|---|---|
| Perception Error | 37% | 40% | 45% |
| Lack of Knowledge | 34% | 35% | 23% |
| Reasoning Error | ~15% | ~15% | ~20% |
| Refuse to Answer | <5% | <5% | <5% |
| Fail to Follow Instruction | <5% | <5% | <5% |
| Annotation Error | <5% | <5% | <5% |

**直觉解析**：

1. **Perception Error 主导**（37-45%）：VLMs 看不清深度、看不见小物体、分不清 light direction。这指向 vision encoder 的 representation bottleneck。CLIP-style encoder 优化的是 image-text alignment，不是 metric depth / 3D geometry。

2. **Lack of Knowledge 显著**（23-35%）：模型不知道 "total internal reflection"、"air pressure 与 boiling point 的关系"、"viscosity 与 temperature 的反相关"。这是 training corpus 中物理知识稀疏的直接证据。

3. **Reasoning Error 较小**：说明 LLM backbone 的 reasoning 能力 OK，瓶颈不在 chain-of-thought 能力，而在 perception + knowledge。

**对 PhysAgent 设计的指导意义**：如果 perception 错占 37-45%，那就接 vision foundation models 补 perception；如果 knowledge 错占 23-35%，那就加 knowledge memory。这就是 PhysAgent 的 two-pronged 设计 motivation。

---

## 7. PhysAgent 架构深度解析

### 7.1 三步流程

```
[Question] 
  ↓
Step 1: Task-specific Prompt Activation
  - 分类问题（manual 或 automatic）
  - 激活 task-specific prompt + 物理知识
  - 例：light 问题 → 检索"光源移动与阴影方向关系"知识
  ↓
Step 2: Foundation Models Integration
  - GroundingDINO: 检测场景中物体
  - SAM: segmentation
  - Depth Anything: depth estimation
  - 从 knowledge memory 检索物体属性
  ↓
Step 3: Chain-of-Thoughts Reasoning
  - VLM 用 LLM 推理能力整合上述信息
  - self-verification step 确保逻辑一致性
  ↓
[Final Answer]
```

### 7.2 与 baselines 的对比 (Figure 9a)

| Method | Property | Relationships | Scene | Dynamics | Avg |
|---|---|---|---|---|---|
| GPT-4o (zero-shot) | 56.91 | 64.80 | 30.15 | 46.99 | 49.49 |
| + CoT | ~56 | ~64 | ~30 | ~46 | ~49 |
| + Desp-CoT | ↓ | ↓ | ↓ | ↓ | ↓ |
| + PLR (pure language) | catastrophic | | | | |
| + ContPhy (oracle) | ↓ in 3/4 tasks | | | | |
| **+ PhysAgent** | **↑** | **↑** | **49.5% (+18.4%)** | **↑** | **+18.4%** |

**关键 insight**：
- **CoT 几乎无效**：这说明 PhysBench 的难度不在 multi-step reasoning，而在 perception + knowledge retrieval。CoT 给的是 "思考时间" 但没有 "外部信息"。
- **Desp-CoT 反而下降**：先描述再推理会让模型 commit 到错误的 perception，然后基于错误 perception 推理出错误答案。
- **PLR 灾难性**：把 image 替换成 description 后，性能崩溃。这说明 image 信息是 essential 的，无法用 text 替代。
- **ContPhy 也不行**：因为它依赖 R-CNN 处理视觉，对 GPT-4o 是 information loss。这证明 neuro-symbolic pipeline 在 VLM 时代是 anti-pattern。

### 7.3 PhysAgent 为什么 work

**Intuition**：PhysAgent 本质上是把 VLM 的 weakness（perception, knowledge）外包给 specialized tools，保留 VLM 的 strength（reasoning, generalization）。这和 Toolformer / ViperGPT / ViperGPT 的 philosophy 一致，但 PhysAgent 的关键是 task-aware knowledge retrieval——不同物理问题激活不同的物理知识 prompt。

Foundation models 的具体作用：
- **Depth Anything V2** (Yang 2024b): metric depth estimation，解决 "Which object is closer to camera?" 类问题
- **SAM** (Kirillov 2023): segmentation，让 VLM 关注 specific region
- **GroundingDINO** (Liu 2023b): open-vocabulary detection，识别 "blue cube" / "yellow ball" 等

**我没看到的具体细节**（paper 没充分展开）：
- Knowledge memory 是 RAG 还是 hardcoded prompt？从描述看更像是 task-class-conditioned prompt template，不是真正的 vector DB retrieval。
- 三步之间是否有 feedback loop？看起来是 linear pipeline。

参考资料：
- [Depth Anything V2](https://arxiv.org/abs/2406.09414)
- [SAM](https://arxiv.org/abs/2304.02643)
- [GroundingDINO](https://arxiv.org/abs/2303.05499)
- [ViperGPT](https://arxiv.org/abs/2303.08128)

---

## 8. MOKA Embodied Integration

### 8.1 Setup

- MuJoCo + Franka Emika 7-DoF（来自 Menagerie）
- 5 个 task: Affordance, Force, Color, Location, Tool
- 用 GPT-4o 作为 VLM
- 两种增强方式：
  1. Fine-tune GPT-4o with PhysBench subset
  2. Zero-shot with PhysAgent

### 8.2 Algorithm 1 (MOKA Pipeline)

```
Algorithm 1: MOKA Pipeline
Input: VLM M, instruction l, prompt p_low, observation s
1: Get observation s from top-down camera
2: Propose keypoint/waypoint candidates → annotated image f(s_k)
3: Query M for low-level motion: y_low = M([p_low, l, f(s)])
4: Execute y_low on robot
```

**关键变量解释**：
- `M`: vision-language model (GPT-4o)
- `l`: task instruction (e.g., "Grasp the blue cube")
- `p_low`: low-level reasoning prompt template
- `s`: observation (RGBD from top-down camera)
- `f(s_k)`: annotated image with keypoint candidates (red circles, see Figure 25)
- `y_low`: low-level motion output (keypoint + waypoint + attributes)

### 8.3 Embodied 结果（Figure 9c）

| Task | MOKA baseline | + PhysAgent | + Fine-tune |
|---|---|---|---|
| Affordance | 0.6 success | 0.8 | 0.9 |
| Force | 0.2 | 0.5 | 0.6 |
| Color | 0.7 | 0.8 | 0.8 |
| Location | 0.7 | 0.7 | 0.8 |
| Tool | 0.4 | 0.5 | 0.7 |

**Intuition**：
- **Force task 提升最大**（0.2 → 0.5 zero-shot, 0.6 fine-tune）：因为 force task 需要理解 deformable / fragile / rigid 物体属性，这正好是 PhysBench Attribute sub-task 训练的能力。
- **Affordance 也显著提升**：因为 PhysAgent 的 GroundingDINO 帮助精确识别物体。
- **Location 提升小**：因为 baseline 已经 0.7，spatial reasoning 是 VLM 相对擅长的。

**My interpretation**：Fine-tune 比 zero-shot PhysAgent 更好，但 PhysAgent 不需要训练数据——这是 trade-off。在生产场景，PhysAgent 更适合 zero-shot deployment，fine-tune 适合特定 robot platform。

参考资料：
- [MOKA paper](https://arxiv.org/abs/2403.03174)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)

---

## 9. 相关 Benchmark Positioning（Table 1, 25, 26）

PhysBench 与已有 benchmark 的对比矩阵：

| Benchmark | Property | Attribute | Location | Motion | Temp | Viewpoint | Light | Collision | Manipulation | Fluid | Interleaved | Size |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CLEVRER | ✓ | | | ✓ | | | | ✓ | | | ✗ | 300k |
| ComPhy | ✓ | ✓ | | ✓ | | | | ✓ | | ✓ | ✗ | 99k |
| Physion | ✓ | ✓ | | ✓ | | | | ✓ | | | ✗ | 17k |
| Physion++ | ✓ | ✓ | | | | | | ✓ | | | ✗ | 2k |
| ContPhy | ✓ | | | ✓ | | | | ✓ | | ✓ | ✗ | 6.5k |
| SuperCLEVR | ✓ | ✓ | ✓ | ✓ | | | ✓ | ✓ | | | ✗ | 1.2k |
| **PhysBench** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** | 10k |

**关键差异**：
- **Interleaved format**：PhysBench 是唯一同时 interleave video + image + text 的。这对 manipulation sequence（"哪个 image 先发生"）类问题 essential。
- **Temperature / Viewpoint / Light / Air**：这四个是 PhysBench 独有的 scene-level physical understanding 维度。
- **Real-world + simulation mix**：避免 synthetic-only 的 distribution shift 问题。

---

## 10. 我对这篇 paper 的 critical assessment

### 10.1 Strengths

1. **Taxonomy 设计出色**：4×19×8 的 hierarchy 把物理理解分解成可测量单元，避免了"physical understanding"作为模糊概念。
2. **Interleaved format 是 correct design**：物理属性本质是多模态 + 时序的，单图无法表达 elasticity。
3. **Error analysis 是 informative 的**：6 类 taxonomy 直接 motivate 了 PhysAgent 的两支设计。
4. **Scaling analysis 揭示 fundamental issue**：物理理解不 scale with parameters/data，这是 *data composition* 问题，不是 *capacity* 问题。

### 10.2 Weaknesses / Open Questions

1. **PhysAgent 缺乏 technical detail**：
   - Knowledge memory 是什么？hardcoded prompt library 还是 RAG?
   - Task classifier 是 manual 还是 learned？paper 写 "manually or automatically" 太 hand-wavy
   - Self-verification 机制怎么实现？是另一个 LLM call 还是 rule-based?

2. **Claude 的 evaluation confound**：
   - Claude-3.5-sonnet 图像被压到 128×128，这是非常不公平的对比。Paper 应该 ablate 这个。

3. **Video VLM 的 instruction following 问题**：
   - Chat-UniVi-13B 只有 10.36% 比 random 低，这是 instruction following 失败。Paper 应该用 answer extraction LLM 兜底（像 Claude 那样用 GPT-4o-mini 提取答案）。

4. **PhysAgent 的 generalization 没充分测试**：
   - 在 PhysBench-test 上 +18.4%，但在 OOD 物理任务上呢？比如 real robot video？

5. **MOKA 实验的 task 太 simple**：
   - 5 个 task 都是 single-step pick-and-place，不是 long-horizon manipulation。PhysBench 训练的 knowledge transfer 到 complex task 没验证。

### 10.3 与你的工作 (Eureka Labs / nanoGPT / VLA) 的潜在连接

我觉得 PhysBench 对 VLA (Vision-Language-Action) model 训练有几个启示：

1. **Data composition matters more than scale for physical reasoning**：VLA 训练不应该只 scale up demonstration 数据，应该 inject physical knowledge（counterfactual reasoning, causal physics）。

2. **Foundation models as perception backbone**：与其让 VLA 从头学 depth estimation，不如像 PhysAgent 那样蒸馏 Depth Anything 的输出作为 auxiliary supervision。

3. **Interleaved training data**：VLA 应该训练在 interleaved (image_t, image_{t+1}, action) 而不是单帧，因为物理 dynamics 本质是时序的。

4. **Scene understanding 是缺的拼图**：当前 VLA 工作（OpenVLA, Octo, RT-2）都没 explicitly 评估 light condition, camera viewpoint 变化下的 robustness。PhysBench-Scene 子集可以做这个 evaluation。

参考资料：
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [Octo](https://octo-models.github.io/)
- [RT-2](https://robotics-transformer2.github.io/)

---

## 11. 公式与数学表达

虽然 PhysBench 本身不强调公式，但 Pearson correlation map (Figure 4a) 的构造用到了：

**Pearson Correlation Coefficient**:

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \cdot \sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

变量说明：
- $x_i$: 第 $i$ 个模型在 benchmark X 上的 score
- $y_i$: 第 $i$ 个模型在 benchmark Y 上的 score
- $\bar{x}, \bar{y}$: 各自的均值
- $n$: 模型数（这里 n=10，见 Table 23）
- $r \in [-1, 1]$: 接近 1 表示两个 benchmark 测同一能力

**Key finding from correlation map**：
- PhysBench-Scene 与 POPE (hallucination detection) 相关性最高 (~0.7)
- PhysBench-Dynamics 与传统 VQA (VQAv2, GQA) 相关性低 (<0.3)
- 这证明 PhysBench 测的是 orthogonal 能力维度

---

## 12. 我会建议的后续研究方向

1. **PhysBench-Real**：当前 simulation + web + real-world 混合，应该有 pure real-world 子集做 OOD evaluation。

2. **Counterfactual PhysBench**：在 PhysBench 基础上加 counterfactual 问题（"如果球是 rubber 而不是 glass，碰撞后会怎样？"）。这是 physical reasoning 的真正 test。

3. **PhysBench-Action**：把 PhysBench 的 QA 形式扩展到 action prediction，与 VLA training 结合。

4. **PhysAgent + RL**：当前 PhysAgent 是 inference-time framework，应该用 RL 在线学习 tool use policy。

5. **Physics-grounded VLM pretraining**：把 PhysBench 89,998 训练集 + depth/normal/albedo 中间表征作为 pretraining signal，训练一个 native physics-aware VLM，而不是用 PhysAgent 这种 plug-in 方式。

---

## 总结 Intuition

PhysBench 揭示的核心 insight：**VLMs 的物理理解缺失不是 model capacity 问题，而是 data composition 问题**。Web-scale caption 数据描述 *what* 而不 *why*，缺乏 counterfactual 物理推理。PhysAgent 通过 tool use + knowledge memory 外包 perception 和 knowledge 两个 bottleneck，让 VLM 专注于 reasoning——这是当前 pragmatic 最优解。但长期看，native physics-aware pretraining 才是终极方案，类似 LLaVA 用 GPT-4V 生成 instruction data 训练 next-gen VLM 的方式，应该用物理 simulator 生成 counterfactual 物理 QA 训练 next-gen physics-VLM。

Andrej，你的 Eureka Labs 课讲了 LLM 从 next-token prediction 涌现 reasoning——物理理解的涌现可能需要类似的"physics-grounded next-frame prediction"训练 objective，把 video prediction + physical property 预测作为 auxiliary loss。这是我会探索的方向。

参考资料汇总：
- [PhysBench project page](https://physbench.github.io/)
- [PhysBench arXiv](https://arxiv.org/abs/2412.04132)
- [MOKA](https://arxiv.org/abs/2403.03174)
- [Depth Anything V2](https://arxiv.org/abs/2406.09414)
- [SAM](https://arxiv.org/abs/2304.02643)
- [GroundingDINO](https://arxiv.org/abs/2303.05499)
- [ContPhy](https://proceedings.mlr.press/v235/zheng24a.html)
- [Physion](https://arxiv.org/abs/2106.05803)
- [ViperGPT](https://arxiv.org/abs/2303.08128)
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [Scaling Laws (Kaplan)](https://arxiv.org/abs/2001.08361)
