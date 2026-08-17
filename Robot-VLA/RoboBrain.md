---
source_pdf: RoboBrain.pdf
paper_sha256: ca163824a22550d1e5778d419704b18732f2fac8505c7599d45438af8f5c2796
processed_at: '2026-08-12T00:22:36-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RoboBrain 人话版

好，那我换个方式，用大白话给你讲一遍这篇 paper 到底干了啥。

---

## 一句话总结

这群人觉得现在的 multimodal LLM 做 robotics 不够用，因为它们只会"说人话"，但不会"做人事"。具体来说，它们缺三个能力：**把复杂任务拆成小步骤**、**知道物体该抓哪儿**、**预测机械臂怎么动过去**。所以这群人造了一个大 dataset 叫 ShareRobot，又训了一个 model 叫 RoboBrain，把这三件事一次搞定。

---

## 为什么要搞这个

Andrej 你肯定懂，现在 LLM + vision encoder 的组合（MLLM）已经很猛了，image captioning、VQA 这些任务 GPT-4o 都能干得不错。但放到 robotics 上就拉胯了。

举个例子，你给 robot 一个指令："拿起 teapot，把水倒进 cup 里"。

一个真正的 robot brain 需要干三件事：

**第一步，Planning（拆任务）**
"拿起 teapot 倒水"这个 high-level instruction 要拆成：
- 走到 teapot 旁边
- 抓住 teapot 的 handle
- 提起来
- 移到 cup 上方
- 倾倒

现在的 MLLM 拆这种 long-horizon task 不太行，因为训练数据里几乎没有这种 fine-grained 的 sub-task decomposition label。

**第二步，Affordance Perception（知道抓哪儿）**
"抓住 teapot 的 handle"——teapot 的哪个区域是 handle？哪个区域是 spout？哪个区域是 lid？model 要能输出一个 bounding box，告诉你"抓这块"。现有 MLLM 对 affordance 的 grounding 能力很差，实验里 Qwen2-VL 只有 12.5% AP。

**第三步，Trajectory Prediction（怎么动过去）**
知道了抓哪儿，还得知道从当前位置到 grasp point 的 path 是什么。paper 里用的是 2D visual trace，就是一系列 $\{x, y\}$ 坐标点，表示 end-effector 在画面上画一条线。

这三步是 **从 abstract 到 concrete** 的 cascade：language instruction → sub-task sequence → affordance region → 2D trajectory。RoboBrain 的 thesis 就是：你得同时 supervise 这三个 level，model 才能真正 ground 到 physical world。

---

## ShareRobot：这个 dataset 才是核心

### 数据从哪来

从 **Open X-Embodiment (OXE)** 里筛。OXE 是一个大杂烩，60 个 dataset、22 种 robot body 合在一起。但 OXE 的标注质量参差不齐，很多只有 high-level description（比如"pick up the cup"），没有 fine-grained 的 sub-task 分解。

RoboBrain 团队从 OXE 里选了 51,403 条 instance，筛选标准：
- 分辨率 > 128px（太糊的 ViT 编码不出有用信息）
- 有准确的文字描述
- 任务执行成功（failed demo 会教坏 model）
- 视频帧数 > 30（太短没法拆 sub-task）
- 物体和 end-effector 没被遮挡（affordance 和 trajectory 要 visible）
- 轨迹清晰可见

### 三维标注

这 51K 条数据，他们做了三种标注：

**Planning 标注**
- 每条 video 抽 30 帧
- 用 Gemini 把 high-level description（比如"make tea"）分解成 low-level instructions（"reach for kettle" → "grasp kettle" → "lift kettle" → ...）
- 3 个 human annotator review 和 refine
- 然后用 10 种 question type × 5 个 template，随机选 2 个 template，把每条 instance 扩展成 QA pairs
- 最终：51,403 instances → **1,027,990 QA pairs**

这个量级很大。对比一下 RoboVQA 只有几十万 QA pairs，而且 scene 多样性远不如 ShareRobot。

**Affordance 标注**
- 6,522 张图片
- 每张标 bounding box：$\{l^{(x)}, l^{(y)}, r^{(x)}, r^{(y)}\}$
  - $l$ = left，$r$ = right，$(x)$ 和 $(y)$ 是 pixel 坐标
  - $l^{(x)}, l^{(y)}$ 是 box 左上角
  - $r^{(x)}, r^{(y)}$ 是 box 右下角
- 这里的 affordance 定义比较 specific：专指 human hand 接触物体的区域

**Trajectory 标注**
- 6,870 张图片
- 每张标 end-effector 的 2D path，至少 3 个 waypoint
- $P_{t:N} = \{(x_i, y_i) \mid i = t, t+1, \ldots, N\}$
  - $t$ 是起始时间步
  - $N$ 是总时间步数
  - $(x_i, y_i)$ 是第 $i$ 个 waypoint 的 2D 坐标

### 数据多样性

- 23 个 source dataset
- 102 个 scene（bedroom, kitchen, lab, office...）
- 12 种 robot body
- 107 种 atomic task type，top 5 是 pick、move、reach、lift、place

这个 diversity 保证了 cross-embodiment generalization 的可能性。

---

## RoboBrain 模型架构

### 基础架构

RoboBrain 建立在 **LLaVA** 上，三个组件：

```
Image/Video → SigLIP (ViT) → 2-layer MLP Projector → Qwen2.5-7B LLM → Text Output
```

- **Vision Encoder**：SigLIP `siglip-so400m-patch14-384`
  - 27 层 transformer
  - patch size 14×14，input resolution 384×384
  - 输出 729 个 visual token per image（因为 $\frac{384}{14} = 27.4$，取整后 $27 \times 27 = 729$）
  - SigLIP 和 CLIP 的区别：用 sigmoid loss 替代 softmax loss，不需要 global normalization，batch size 可以 scale 得更大

- **Projector**：2-layer MLP，把 visual token 映射到 LLM 的 embedding space

- **LLM**：Qwen2.5-7B-Instruct
  - 28 层
  - 128K context window
  - 支持多语言

### 两个 LoRA 模块

基础 model 负责 planning（输出 text），再挂两个 LoRA 做 affordance 和 trajectory：

**A-LoRA（Affordance LoRA）**
- LoRA rank = 64
- 加在 Projector 和 LLM 的 FFN 层上
- 28M trainable parameters（vs base model 的 8B）
- 输出格式：bounding box 坐标 $\{l^{(x)}, l^{(y)}, r^{(x)}, r^{(y)}\}$

**T-LoRA（Trajectory LoRA）**
- 同样 rank = 64，28M params
- 输出格式：waypoint 序列 $(x_1, y_1), (x_2, y_2), \ldots$
- 坐标 normalize 到 $[0, 1000)$（跟随 Qwen2-VL 的做法）

### 推理流程

实际用的时候是 pipeline：

1. 给 instruction + visual input
2. Base model 生成 detailed plan（text）
3. 把 plan 拆成 sub-tasks
4. 每个 sub-task：
   - A-LoRA 输出 affordance bounding box
   - T-LoRA 输出 2D trajectory waypoints

这是一个 **coarse-to-fine** 的 cascade。

---

## 训练策略：5 个 stage

这是 paper 的另一个核心。分两个 phase，共 5 个 stage。

### Phase 1：General MLLM 训练

这个 phase 的目标是：先让 model 变成一个强大的 general-purpose MLLM。

| Stage | 干什么 | 数据 | 训练什么 |
|-------|--------|------|----------|
| S1 | Vision-language alignment | LCS-558K（image-text pairs） | 只训 Projector（17M params） |
| S1.5 | 多模态知识注入 | 4M 高质量 image-text | 全模型（8B） |
| S2 | 高分辨率 + 视频 | 3.2M single-image + 1.6M video | 全模型（8B） |

S2 的关键细节：resolution 是 `Max 384×{6×6}`，意思是把图片切成最多 36 个 tile，每个 tile 729 tokens，所以最多 729×37 ≈ 26,973 个 visual token（37 是 36 tile + 1 个 base view）。这个 token budget 让 model 能同时看到全局布局和局部细节。

### Phase 2：Robotics 专项训练

| Stage | 干什么 | 数据 | 训练什么 |
|-------|--------|------|----------|
| S3 | Robotics planning | 1.3M robotic + 1.7M general（混入防 forgetting） | 全模型（8B） |
| S4-A | Affordance | 10K affordance images | A-LoRA（28M） |
| S4-T | Trajectory | 400K trajectory images | T-LoRA（28M） |

S3 的 robotic data 组成：
- RoboVQA-800K
- ScanView-318K（3D 场景扫描数据：MMScan + 3RScan + ScanQA + SQA3D）
- ShareRobot-200K（他们自己造的数据的子集）

**为什么混 1.7M general data？** 防 catastrophic forgetting。纯 robotic data fine-tune 会把 general capability 冲掉。robot : general = 4 : 6 这个比例是 ablation 出来的最优值。

**为什么 S4 用 LoRA 而不是 full fine-tune？** 因为 S3 已经把 planning 能力训到 65.05 分了，如果 S4 全量 fine-tune，planning 能力会崩。LoRA 的 low-rank 约束相当于在已经学好的 planning representation 上加一个稀疏旁路，学 affordance/trajectory 的同时保留 planning。这和 LoRA 原始论文 [LoRA](https://arxiv.org/abs/2106.09685) 的 motivation 完全一致。

---

## 实验结果：哪些数字值得关注

### Planning：碾压级提升

RoboVQA benchmark（BLEU 分数，越高越好）：

| Model | BLEU-1 | BLEU-4 |
|-------|--------|--------|
| GPT-4V | 32.23 | 23.94 |
| LLaVA-OV-7B | 38.12 | 30.97 |
| RoboMamba | 54.9 | 36.3 |
| **RoboBrain** | **72.05** | **55.05** |

BLEU-4 比 second-best 高了 18.75 分。这个 gap 非常大，主要归功于 ShareRobot 的 fine-grained planning 标注。

### Affordance：从 12.5% 到 27.1%

AGD20K test set，AP metric：

| Model | AP |
|-------|-----|
| LLaVA-NeXT-7B | 9.8% |
| Qwen2-VL-7B | 12.5% |
| **RoboBrain** | **27.1%** |

baseline MLLM 的 affordance grounding 能力极差，RoboBrain 靠 A-LoRA + 专门 supervision 翻倍。

### Trajectory：special token 是关键

| Variant | DFD ↓ | HD ↓ | RMSE ↓ |
|---------|-------|------|--------|
| Base | 0.191 | 0.171 | 0.133 |
| + Start Points | 0.176 | 0.157 | 0.117 |
| + Max Points | 0.185 | 0.163 | 0.125 |
| + Spec Token & End Points | **0.109** | **0.010** | **0.091** |

最 interesting 的发现：

**加 start points 为什么有用？** 因为如果不告诉 model end-effector 起点在哪，它生成的 trajectory 会有一个 systematic translational offset——形状对了但整体偏了。给了 start point 就消除了这个平移误差。

**HD 从 0.171 暴跌到 0.010（94.2% 降幅）是什么概念？** HD = Hausdorff Distance，衡量两条 trajectory 之间最大的偏离点。降到 0.010 意味着预测 trajectory 和 ground truth 几乎完全重合，没有任何一个点偏离很远。这说明 special token 让 model 特别关注 start/goal 这些 critical points，整体 trajectory 的 max deviation 被极大压缩。

**为什么 max points 单独用反而更差？** 因为 uniform sampling 到 10 个 waypoint 会丢掉关键中间点。但配合 special token 就有 regularization 效果——special token 强调关键点，uniform sampling 保证长度可控。

### Data ratio ablation：4:6 是 sweet spot

| Robot : General ratio | RoboVQA | ShareRobot-Eval | General Avg |
|----------------------|---------|-----------------|-------------|
| 3:7 | 45.96 | 61.73 | 67.67 |
| **4:6** | **48.29** | **63.11** | 68.25 |
| 5:5 | 49.34 | 63.35 | 67.03 |
| 6:4 | 49.22 | 64.57 | 67.39 |
| 7:3 | 47.74 | 65.22 | 67.72 |

robotic data 太少（3:7），robotic benchmark 低；太多（7:3），general benchmark 掉。4:6 是 general 和 robotic 的 Pareto 最优解。

### ShareRobot 数据本身的价值验证

最 convincing 的实验是 Table 8：把 ShareRobot 数据喂给不同的 base model：

| Model | 有 ShareRobot？ | RoboVQA | ShareRobot-Eval |
|-------|----------------|----------|-----------------|
| OpenVLA-7B | 没有 | 4.11 | 21.44 |
| OpenVLA-7B | 有 | 54.79 | 60.56 |
| Qwen2-VL-7B | 没有 | 24.05 | 28.17 |
| Qwen2-VL-7B | 有 | 58.94 | 58.86 |

OpenVLA 在没 ShareRobot 时 RoboVQA 只有 4.11 分（基本不能用），加了 ShareRobot 直接跳到 54.79。这说明 **ShareRobot 数据本身具有很强的 transferability**，不管你用什么 architecture 都能受益。

这其实印证了你 Karpathy 一直说的那句话：data is the bottleneck，architecture 是 commodity。

---

## General Benchmark 也没掉

RoboBrain 在 general MLLM benchmark 上的表现：

| Benchmark | RoboBrain | GPT-4o |
|-----------|-----------|--------|
| RealWorldQA | **68.89** | 58.6 |
| AI2D | 82.03 | 94.2 |
| ChartQA | 80.48 | 85.7 |
| MMStar | 61.23 | 63.9 |

RealWorldQA 超过 GPT-4o 是一个很有意思的发现：**robotic data 反而增强了 real-world visual understanding**。这和 embodied AI 的核心 hypothesis 一致——grounded physical learning 帮助 general perception。

---

## 失败案例分析：model 的 blind spot

### Planning 失败

"Clean the desk"（桌上有打翻的咖啡）：
1. 把 tissue 识别成 disinfectant wipe → object recognition error
2. 忘了"从 tissue box 里抽 tissue"这一步 → critical step omission
3. 擦整个 desk 而非先擦咖啡渍 → priority deviation

Root cause：spilled coffee 和 desk 颜色太像，model 的 visual grounding 不够 robust。

### Trajectory 失败

- 开冰箱门时忽略了门的 articulated（铰链）性质，预测的 trajectory 物理上不可行
- 折衣服时没考虑衣服是 deformable 的，把衣服当 rigid body 来规划 path

这些都指向 **physical constraint 和 world knowledge 的缺失**——model 缺少对物体物理性质的理解。

---

## 几个直觉性的 Takeaway

### 1. 为什么 2D trajectory 而不是 3D joint angles？

因为 2D visual trace 和 visual input 在同一个 modality（pixel space），MLLM 天然能 handle。3D joint angles 需要专门的 motor policy，脱离了 MLLM 的优势区。2D trajectory 可以作为 downstream policy head 的 conditioning signal，类似 [RT-Trajectory](https://rt-trajectory.github.io/) 的思路——用 2D sketch 作为 task specification。

### 2. 为什么 modular LoRA 比 joint training 好？

S3 训完后 planning 已经 65.05 分了。如果 S4 继续 full fine-tune，affordance/trajectory 的 gradient 会干扰 planning representation，导致 catastrophic forgetting。LoRA rank=64 相当于在 8B 参数空间里只更新一个 28M 的低维子空间，物理上限制了新 skill 对旧 skill 的干扰。

### 3. 数据标注的 multi-dimensionality 是关键创新

同一个 robotic demo video，传统做法只标一个 action label。ShareRobot 从三个角度标：
- Planning（text-level sub-task decomposition）
- Affordance（spatial region grounding）
- Trajectory（temporal waypoint sequence）

每个角度对应不同 cognitive level，三个 level 的 supervision signal 叠加，让 model 学到更丰富的 representation。这个 "multi-view supervision" 思路我觉得可以推到其他 domain。

### 4. Open-loop 的根本局限

RoboBrain 是 **open-loop**：给定 instruction + 当前 visual observation，一次性生成 plan + affordance + trajectory，然后执行。没有 closed-loop replanning。

但真实 robotics 需要的是：执行过程中如果状态变了（比如 cup 被碰倒了），要能 re-perceive + re-plan。RoboBrain 的 pipeline 没有这个 reactive loop。这是 [Inner Monologue](https://inner-monologue.github.io/) 和 [Reflexion](https://arxiv.org/abs/2304.03342) 那条线在做的事。

### 5. Sim-to-real 和 embodiment gap

训练数据全是 real-world demo，但 inference 时如果换了一个不同的 robot arm（比如训练数据里是 Franka，inference 时是 UR5），2D trajectory 的 generalization 如何？paper 没有充分讨论这个。cross-embodiment generalization 是 [Open X-Embodiment](https://robotics-transformer-x.github.io/) 和 [RT-X](https://robotics-transformer-x.github.io/) 在攻关的核心问题，RoboBrain 虽然用了 12 种 embodiment 的数据，但没给出 embodiment-level 的 generalization 实验。

---

## 和你 Karpathy 可能关心的方向的关系

### 和 VLA（Vision-Language-Action）的关系

RoboBrain 不是 VLA model。VLA model（如 [OpenVLA](https://openvla.github.io/)、[RT-2](https://robotics-transformer2.github.io/)）直接输出 low-level action（joint angles 或 end-effector pose），是 end-to-end 的。RoboBrain 输出的是 plan text + 2D bounding box + 2D trajectory waypoints，还需要一个 downstream policy 把这些变成 actual motor commands。

所以 RoboBrain 更像是一个 **cognitive layer**，坐在 VLA policy 之上。你 Karpathy 之前在 Tesla 讲的 "system 1 / system 2" 的架构里，RoboBrain 对应 system 2（slow thinking, planning），VLA 对应 system 1（fast reflex, motor control）。

### 和 data-centric AI 的关系

这篇 paper 再次印证了你一直强调的观点：**data engineering 是 AI 进步的真正 bottleneck**。RoboBrain 的 architecture 没什么 novelty（LLaVA + LoRA，都是现成的），真正的 contribution 是 ShareRobot 的数据质量。Table 8 证明换什么 backbone 都能从 ShareRobot 受益。

这和 [ImageNet](https://www.image-net.org/) 当年的作用类似——不是 AlexNet 有多 novel，是 ImageNet 这个 dataset 让 deep learning 成为可能。

### 和你可能感兴趣的 future direction

1. **Closed-loop replanning**：把 RoboBrain 放进一个 reactive loop里，每执行一步就 re-perceive + re-plan
2. **3D understanding**：从 2D trajectory 升级到 3D（或至少 depth-aware），用 [NeRF](https://arxiv.org/abs/2003.08934) 或 [Gaussian Splatting](https://arxiv.org/abs/2308.14737) 做 scene representation
3. **Physical constraint injection**：让 model 理解 articulated objects（门、抽屉）、deformable objects（衣服、绳子）、liquids 的物理性质
4. **Long-horizon reasoning**：现在的 planning 大概是 5-10 步的 sub-task，真正的 long-horizon（比如"做一顿饭"）需要 hierarchical planning 和 memory

---

## 参考链接

- [RoboBrain Project Page](https://robobrain.github.io/)
- [ShareRobot Dataset](https://github.com/RoboBrain-and-ShareRobot/ShareRobot)
- [LLaVA-OneVision](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/)
- [Open X-Embodiment](https://robotics-transformer-x.github.io/)
- [OpenVLA](https://openvla.github.io/)
- [RT-Trajectory](https://rt-trajectory.github.io/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Inner Monologue](https://inner-monologue.github.io/)
- [RoboVQA](https://robot-vqa.github.io/)
- [OpenEQA](https://open-eqa.github.io/)
- [SigLIP](https://arxiv.org/abs/2303.15343)
- [Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/)

---

总之，这篇 paper 的 core message 就一句话：**robotics 需要的不只是 LLM 会说话，还需要会拆任务、会找抓取点、会画运动轨迹。要教 model 这三件事，你得有标注好这三层信息的大 dataset。ShareRobot 就是这个 dataset，RoboBrain 就是吃这个 dataset 训出来的 model。** 架构上没什么新东西，数据工程是真正的 contribution。

---

# RoboBrain 深度解析：从 Abstract 到 Concrete 的 Robotic Brain

Hey Andrej，这篇 paper 挺有意思，我给你做一个 deep dive。核心 idea 是把 robotic manipulation 分解成三个 cognitive level：**planning**（abstract reasoning）、**affordance perception**（mid-level grounding）、**trajectory prediction**（concrete execution），用一个 unified MLLM 全包下来。下面我拆开讲。

---

## 1. 核心洞察：为什么需要 RoboBrain

现有 MLLMs 在 robotics 上卡在一个 fundamental gap：它们能理解 high-level language，但 missing 三个从 abstract 到 concrete 的桥梁能力。举个例子——"lift teapot and pour water into cup"：

- **Planning**：要分解成 "approach teapot → lift it → move spout over cup → tilt to pour"
- **Affordance perception**：对每个 sub-task，比如 "grasp teapot"，要识别 teapot 上哪块区域是 graspable 的
- **Trajectory prediction**：要预测 end-effector 从起点到 grasp point 的完整 2D path

paper 的 thesis 是：缺少大规模、fine-grained 的 robotic operation dataset 导致 MLLMs 无法学到这三层能力。所以他们同时造数据（ShareRobot）+ 训模型。

Project page: [RoboBrain](https://robobrain.github.io/)  
相关 reference: [PaLM-E](https://palm-e.github.io/), [RT-2](https://robotics-transformer2.github.io/), [OpenVLA](https://openvla.github.io/)

---

## 2. ShareRobot Dataset：数据工程是核心

这部分是 paper 的真正 contribution，我重点讲。

### 2.1 数据筛选 pipeline

从 **Open X-Embodiment (OXE)** 里筛 51,403 instances，filtering criteria 非常讲究：

| Criterion | Rationale |
|-----------|-----------|
| Resolution > 128px | 避免 ViT 编码出 garbage token |
| Accurate description | vague description 会污染 planning label |
| Success status | failed demos 会让 model 学到错误因果 |
| Frames > 30 | 太短的视频 atomic tasks 不足 |
| Object not covered | affordance 必须 visible 才能标 bounding box |
| Clear trajectories | 模糊 trajectory 无法 supervise T-LoRA |

这个 filtering 思路其实和 LLaVA 系列做 data curation 的逻辑一致，只是搬到了 robotic domain。

### 2.2 三维标注

**Planning Labeling**：
- 抽 30 frames + high-level description
- 用 **Gemini** 分解成 low-level instructions
- 3 个 annotators review & refine
- 设计 5 templates × 10 question types（来自 RoboVQA），随机选 2 templates per question type 生成 QA pairs
- 结果：51,403 instances → 1,027,990 QA pairs

**Affordance Labeling**：
- 6,522 images，每个标 bounding box
- 格式：$\{l^{(x)}, l^{(y)}, r^{(x)}, r^{(y)}\}$
  - $l^{(x)}, l^{(y)}$：affordance 区域 top-left 坐标
  - $r^{(x)}, r^{(y)}$：affordance 区域 bottom-right 坐标
- 这里的 affordance 定义比较窄——专指 "human hand makes contact with objects" 的区域

**Trajectory Labeling**：
- 6,870 images，每个标 end-effector 的 2D path
- 至少 3 个 $\{x, y\}$ waypoints
- 形式化：$P_{t:N} = \{(x_i, y_i) \mid i = t, t+1, \ldots, N\}$
  - $t$：起始 time step
  - $N$：episode 总 time step 数
  - $(x_i, y_i)$：第 $i$ 个 waypoint 的 2D 坐标

### 2.3 数据多样性

- **23 个 source datasets**（来自 OXE）
- **102 个 scenes**（bedroom, lab, kitchen, office...）
- **12 种 embodiments**
- **107 种 atomic task types**，top 5 是 "pick", "move", "reach", "lift", "place"

这个 diversity 很关键，cross-embodiment generalization 是当下 robotics foundation model 的核心 challenge。

---

## 3. RoboBrain Architecture

基于 **LLaVA** 框架，三个模块：

```
Image/Video X_v → ViT g(·) → Z_v → Projector h(·) → H_v → LLM f(·) → Text Response
                                     (SigLIP)        (2-layer MLP)    (Qwen2.5-7B)
```

### 3.1 Foundational Model

- **Vision Encoder**: SigLIP `siglip-so400m-patch14-384`
  - 27 hidden layers
  - Patch size 14×14，resolution 384×384 → 729 visual tokens per image
  - SigLIP vs CLIP：用 sigmoid loss 替代 softmax，避免 global normalization，batch size scaling 更高效
- **Projector**: 2-layer MLP
- **LLM**: Qwen2.5-7B-Instruct
  - 28 hidden layers
  - 128K context window
  - 29+ languages

### 3.2 A-LoRA Module（Affordance）

数学定义：对图像 $I$ 中的 object $i$，其 affordance 集合为：

$$O_i = \{A_i^0, A_i^1, \ldots, A_i^N\}$$

其中 $A_i^j = \{l^{(x)}, l^{(y)}, r^{(x)}, r^{(y)}\}$ 表示第 $i$ 个 object 的第 $j$ 个 affordance 区域。

- LoRA rank = 64
- 加到 Projector 和 LLM 的 feed-forward network layers
- 28M trainable parameters
- 训练数据：10K affordance samples

### 3.3 T-LoRA Module（Trajectory）

输出 2D visual traces（参考 [RT-Trajectory](https://rt-trajectory.github.io/) 的概念）：

$$P_{t:N} = \{(x_i, y_i) \mid i = t, t+1, \ldots, N\}$$

- 坐标 normalize 到 [0, 1000)（follow Qwen2-VL）
- LoRA rank = 64
- 28M trainable parameters
- 训练数据：400K trajectory samples

### 3.4 推理 pipeline

```
Instruction + Visual Input
        ↓
   RoboBrain (Base) → generate detailed plan
        ↓
   Split into sub-tasks
        ↓
   For each sub-task:
       ├─ A-LoRA → affordance bounding box
       └─ T-LoRA → 2D trajectory waypoints
```

---

## 4. Multi-Stage Training Strategy

这是 paper 的另一大 contribution，5 个 stage，我详细拆解每个 stage 的数据、resolution、tokens、learning rate。

### Phase 1: General OV Training

| Stage | Data | Resolution | #Tokens | Trainable | LR (ViT) | LR (Proj/LLM) | Batch | Epoch |
|-------|------|-----------|---------|-----------|----------|---------------|-------|-------|
| **S1** | LCS-558K | 384 | 729 | Projector (17M) | — | 1e-3 | 8×1 | 1 |
| **S1.5** | 4M image-text | Max 384×{2×2} | Max 729×5 | Full (8B) | 2e-6 | 1e-5 | 2×2 | 1 |
| **S2** | 3.2M single + 1.6M video | Max 384×{6×6} | Max 729×37 | Full (8B) | 2e-6 | 1e-5 | 1×2 | 1 |

- **S1**：纯 alignment，只训 Projector，让 visual token 对齐 LLM 的 semantic space
- **S1.5**：full model warmup，用 4M 高质量 image-text 对
- **S2**：高分辨率 + 视频，`384×{6×6}` 意味着最多 36 个 tiles，每个 729 tokens → max 26,244 visual tokens，这个 token budget 对 robotic 场景很重要（要同时看全局 + 细节）

### Phase 2: Robotic Training

| Stage | Data | Resolution | #Tokens | Trainable | LR | Batch | Epoch |
|-------|------|-----------|---------|-----------|-----|-------|-------|
| **S3** | 1.3M robotic + 1.7M general | Max 384×{6×6} | Max 729×37 | Full (8B) | 1e-5 (Proj/LLM) | 1×2 | 1 |
| **S4-A** | 10K affordance | Max 384×{6×6} | Max 729×37 | A-LoRA (28M) | 1e-5 | 4×2 | 1 |
| **S4-T** | 400K trajectory | Max 384×{6×6} | Max 729×37 | T-LoRA (28M) | 1e-5 | 4×2 | 1 |

**S3 数据组成**（1.3M robotic data）：
- RoboVQA-800K
- ScanView-318K（MMScan-224K + 3RScan-43K + ScanQA-25K + SQA3D-26K）
- ShareRobot-200K（subset）

**Anti-catastrophic forgetting**：混入 1.7M general image-text data，比例 robot:general ≈ 4:6。这个比例是 ablation 出来的最优值（见下面的实验）。

---

## 5. 实验结果深度分析

### 5.1 Planning Task

**RoboVQA**（BLEU-1~4）：

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|-------|--------|--------|--------|--------|
| GPT-4V | 32.23 | 26.51 | 24.65 | 23.94 |
| LLaVA-OV-7B | 38.12 | 33.56 | 31.76 | 30.97 |
| RoboMamba | 54.9 | 44.2 | 39.5 | 36.3 |
| Qwen2-VL-7B | 33.22 | 26.11 | 20.98 | 17.37 |
| **RoboBrain** | **72.05** | **65.35** | **59.39** | **55.05** |

BLEU-4 比 second-best (RoboMamba 36.3) 高 18.75 points，这个 gap 非常大。主要原因是 ShareRobot 的 fine-grained planning 标注质量。

**OpenEQA**（8 个子任务）：

| Subtask | RoboBrain | GPT-4V | LLaVA-OV | Qwen2-VL |
|---------|-----------|--------|----------|----------|
| OBJECT-STATE-RECOG | 70.4 | 63.2 | 72.02 | 72.06 |
| OBJECT-RECOG | 49.54 | 43.4 | 51.73 | 61.91 |
| FUNCTIONAL-REASONING | 57.14 | 57.4 | 55.53 | 54.23 |
| SPATIAL-UNDERSTANDING | 46.46 | 33.6 | 48.98 | 50.39 |
| ATTRIBUTE-RECOG | 66.7 | 57.2 | 75.52 | 73.88 |
| DISCRIMINATIVE | **99.02** | — | 57.9 | — |

注意 SPATIAL-UNDERSTANDING RoboBrain 46.46 比 GPT-4V 33.6 高，但仍弱于 Qwen2-VL 50.39，说明 spatial intelligence 还有提升空间。

### 5.2 Affordance Prediction

在 **AGD20K** test set 上用 AP metric：

| Model | AP ↑ |
|-------|------|
| LLaVA-NeXT-7B | 9.8% |
| Qwen2-VL-7B | 12.5% |
| **RoboBrain** | **27.1%** (+14.6 / +17.3) |

这个 gap 巨大，说明 A-LoRA + 专门的 affordance supervision 是关键。baseline MLLMs 几乎无法 grounding affordance。

### 5.3 Trajectory Prediction

这是 paper 里最 interesting 的 ablation，4 个 variant：

| Method | DFD ↓ | HD ↓ | RMSE ↓ |
|--------|-------|------|--------|
| Base (VQA-only) | 0.191 | 0.171 | 0.133 |
| + Start_Points | 0.176 | 0.157 | 0.117 |
| + Max_Points (uniform sample to 10) | 0.185 | 0.163 | 0.125 |
| + Spec_Token & End_Points | **0.109 (42.9%↓)** | **0.010 (94.2%↓)** | **0.091 (31.6%↓)** |

关键洞察：
- **Start points** 修正 translational offset——模型知道 end-effector 起点位置后，trajectory 整体平移误差消除
- **Special tokens** 强调 waypoints 和 start/goal，让 LLM 更关注 critical points
- **Max points** 单独用反而比 base 差，因为 uniform sampling 丢失了关键 waypoints；但配合 special token 就有 regularization 效果

HD 从 0.171 降到 0.010（94.2%↓）是 dramatic improvement，说明 max deviation 被极大压缩——trajectory 整体形状对了。

### 5.4 Data Ratio Ablation

Table 7 的 ablation 非常 informative：

| Exp | General | ShareRobot | Others | RoboVQA | ShareRobot-Eval | Avg |
|-----|---------|------------|--------|---------|-----------------|-----|
| A (40% R, 20% SR) | 60% | 20% | 20% | 48.29 | 63.11 | 62.48 |
| B (40% R, no SR) | 60% | 0% | 40% | 49.20 | 27.03 | 55.66 |
| C (30% R) | 70% | 15% | 15% | 45.96 | 61.73 | 61.22 |
| D (= A) | 60% | 20% | 20% | 48.29 | 63.11 | 62.48 |
| E (50% R) | 50% | 25% | 25% | 49.34 | 63.35 | 61.92 |
| F (60% R) | 40% | 30% | 30% | 49.22 | 64.57 | 62.07 |
| G (70% R) | 30% | 35% | 35% | 47.74 | 65.22 | 62.14 |

两个结论：
1. **ShareRobot 很关键**：Exp A vs Exp B，ShareRobot-Eval 从 63.11 掉到 27.03（57% 降幅）
2. **4:6 是 sweet spot**：general benchmark 没掉太多，robotic benchmark 涨明显。Robot data 比例再高（7:3）会伤 general capability，反而 avg 下降。

### 5.5 Cross-Architecture 验证

Table 8 测试 ShareRobot 在不同 backbone 上的泛化：

| Model | SFT (G:R) | RoboVQA | ShareRobot | MMMU |
|-------|-----------|---------|------------|------|
| LLaVA-OV-7B | 6:0 | 36.29 | 27.04 | 49.65 |
| LLaVA-OV-7B | 6:4 | 43.63 | 54.66 | 48.83 |
| Qwen2-VL-7B | 6:0 | 24.05 | 28.17 | 52.10 |
| Qwen2-VL-7B | 6:4 | 58.94 | 58.86 | 52.33 |
| OpenVLA-7B | 6:0 | 4.11 | 21.44 | 35.07 |
| OpenVLA-7B | 6:4 | 54.79 | 60.56 | 37.25 |

OpenVLA 在 6:0 时 RoboVQA 只有 4.11（因为它没对齐过 vision-language），但加 ShareRobot 后跳到 54.79，说明 ShareRobot 数据本身具有很强的 transferability。

### 5.6 Stage-wise Ablation

Table 9 是 stage-by-stage 的能力增长：

| Stage | RoboVQA | ShareRobot | Affordance ↑ | Trajectory ↓ |
|-------|---------|------------|--------------|--------------|
| S1.5 | 2.60 | 9.81 | 0.00 | 1.00 |
| S2-si | 28.90 | 13.31 | 3.11 | 1.00 |
| S2-ov | 31.81 | 34.84 | 8.50 | 1.00 |
| S3 | 62.96 | 65.05 | 7.14 | 1.00 |
| S4-A | 62.96 | 65.05 | **27.1** | — |
| S4-T | 62.96 | 65.05 | — | **0.09** |

可以清晰看到：
- S1.5 → S2：planning 从 9.81 跳到 34.84（video + high-res 贡献）
- S2 → S3：planning 从 34.84 跳到 65.05（robotic data 注入）
- S3 → S4：affordance 从 7.14 跳到 27.1，trajectory 从 1.00 降到 0.09（LoRA 专门训练）
- S4 不影响 planning（frozen base model）

这个 stage-wise 的 isolation 验证了 modular 设计的合理性。

---

## 6. General Benchmark 表现

RoboBrain 在 general MLLM benchmark 上也没掉链子（Table 5）：

| Benchmark | RoboBrain | GPT-4V | LLaVA-OV-7B | Qwen2-VL-7B |
|-----------|-----------|--------|-------------|-------------|
| AI2D | 82.03 | 78.2 | 81.4 | — |
| ChartQA | 80.48 | 78.5 | 80 | 83 |
| RealWorldQA | **68.89** | 61.4 | 66.3 | 70.1 |
| MMStar | 61.23 | 57.1 | 61.7 | 60.7 |
| MMBench-en-dev | 81.52 | 81.3 | 83.2 | — |

特别值得注意的是 **RealWorldQA 68.89** 超过 GPT-4o (58.6) 和 GPT-4V (61.4)，说明 robotic data 反而增强了 real-world understanding，这和 embodied AI 的 hypothesis 一致——grounded visual learning 帮助 general perception。

---

## 7. Qualitative 分析与 Failure Modes

### 7.1 Planning Success Cases

"Cluster blocks of the same color into different corners" 这种 long-horizon task，RoboBrain 能：
1. Step 1-2：分析桌上每种 color 的 block 数量
2. Step 3：分解成 4 个 sub-movements（top-left, top-right, bottom-left, bottom-right）

### 7.2 Planning Failure Case

"Clean the desk"（spilled coffee scene）：
1. **Object recognition error**：把 tissue 识别成 disinfectant wipe
2. **Critical step omission**：忘了 "extract tissue from box"
3. **Action priority deviation**：擦整个 desk 而非先擦 spilled coffee

Root cause：spilled coffee 和 desk 颜色相似，model 的 segmentation/grounding 不够 robust。

### 7.3 Affordance Failure Modes

- 物体识别错误
- 场景中其他物体干扰
- 完全没识别到物体

这些失败指向 noisy environment 下的 perception robustness 不足。

### 7.4 Trajectory Failure Modes

- 杯子定位不准
- 忽略冰箱门的 articulated 性质
- 折叠衣服时没考虑 deformable properties

这些都指向 **physical constraints + world knowledge** 的缺失。

---

## 8. 一些 Intuitive Takeaways

### 8.1 为什么 Modular LoRA（A-LoRA / T-LoRA）比 joint training 好？

S3 已经把 base model 训到 65.05 ShareRobot score，如果继续 full fine-tune 做 affordance/trajectory，会 catastrophic forgetting planning 能力。LoRA 的 low-rank constraint（rank=64, 28M params vs 8B total）相当于在 high-level planning representation 上加一个 sparse 旁路，preserve base capability 的同时学新 skill。这和 [LoRA](https://arxiv.org/abs/2106.09685) 原始 motivation 一致。

### 2D Trajectory 为什么有效？

paper 用 2D visual traces 而非 3D joint angles，这个选择很聪明：
- 2D 在 pixel space，和 visual input 同 modality，MLLM 天然 handle
- 3D joint angles 需要专门的 motor policy，脱离 MLLM 优势
- 2D trajectory 可以作为 downstream policy head 的 supervision，类似 [RT-Trajectory](https://rt-trajectory.github.io/) 的思路

### 8.3 Data Centric > Model Centric

RoboBrain 的架构没什么 novelty（就是 LLaVA + LoRA），真正的 contribution 是 **ShareRobot 的数据质量**。Table 8 证明：换 backbone（LLaVA-OV / Qwen2-VL / OpenVLA / LLaVA1.5 + 各种 LLM）都能从 ShareRobot 受益，说明 dataset 本身是 transferable asset。

这呼应了 Karpathy 你一直强调的 "data is all you need"——数据工程是 AI 进步的 bottleneck，不是 model architecture。

---

## 9. 局限与 Future Work

Paper 自己提到的 future direction：
- Spatial understanding（[Thinking in Space](https://arxiv.org/abs/2412.14171)）
- Embodied reasoning（[Gemini Robotics](https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/)）
- Tool utilization
- Long-context comprehension
- Model efficiency（[RoboMamba](https://arxiv.org/abs/2406.04339)）
- Safety（[BadRobot](https://arxiv.org/abs/2412.04455), [GuardReasoner](https://arxiv.org/abs/2501.18492)）

我观察到 paper 没充分讨论的：
1. **Sim-to-real gap**：训练数据全是 real-world demo，但 inference 时 embodiment 是否匹配？
2. **Temporal reasoning**：trajectory 只有 2D，没有显式 time modeling
3. **Closed-loop control**：RoboBrain 是 open-loop 生成 plan + trajectory，没有 reactive replanning
4. **3D understanding**：2D trajectory 无法 capture depth，复杂 6-DoF manipulation 怎么办？

---

## 10. 相关工作脉络

我给你梳理一下这个方向的工作谱系：

**VLA Models**：
- [RT-1](https://arxiv.org/abs/2212.06817), [RT-2](https://robotics-transformer2.github.io/), [RT-H](https://arxiv.org/abs/2403.01823)
- [OpenVLA](https://openvla.github.io/)
- [PaLM-E](https://palm-e.github.io/)

**Robotic Datasets**：
- [Open X-Embodiment](https://robotics-transformer-x.github.io/)
- [DROID](https://droid-dataset.github.io/)
- [BridgeDataV2](https://github.com/alexanderkhazatsky/BridgeDataV2)
- [RH20T](https://rh20t.github.io/)

**MLLM for Robotics**：
- [RoboVQA](https://robot-vqa.github.io/)
- [RoboMamba](https://arxiv.org/abs/2406.04339)
- [LLaVA-OneVision](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/)
- [Qwen2-VL](https://qwenlm.github.io/blog/qwen2-vl/)

**Affordance Learning**：
- [AGD20K / Affordance Grounding](https://arxiv.org/abs/2205.14003)
- [RT-Trajectory](https://rt-trajectory.github.io/)
- [LLaRVA](https://arxiv.org/abs/2406.11815)

---

## 总结

RoboBrain 的核心 contribution 是证明了一件事：**用 fine-grained heterogeneous annotation（planning + affordance + trajectory）supervise MLLM，可以让它从 abstract language instruction 推到 concrete 2D action**。技术 novelty 有限，但 data engineering + multi-stage training recipe 很扎实。

对你 Andrej 来说，这个 paper 可能最值得思考的点：data annotation 的 multi-dimensionality——同一个 robotic demo 可以从 planning（text）、affordance（bounding box）、trajectory（waypoints）三个角度 supervise，每个角度对应不同的 cognitive level。这种 "多视角 supervision" 思路能不能推到其他 domain？比如 autonomous driving 的 perception-planning-control stack？

希望这个 deep dive 帮你 build 出对 robotic MLLM 当前 state-of-art 的 intuition。如果有 specific 想深挖的点（比如 trajectory 的 special token 设计、LoRA rank 选择、或者 S2 的 tile 策略），可以继续聊。
