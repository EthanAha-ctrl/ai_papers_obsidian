---
source_pdf: GeneralVLA Generalizable Vision–Language–Action Models with Knowledge-Guided
  Trajectory Planning.pdf
paper_sha256: b594c5788ba7d426cfbd6ceab842948c2c14744799aa9c64cf14db02b510fffb
processed_at: '2026-08-04T13:37:50-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GeneralVLA 用人话讲

## 一句话说清楚

这篇 paper 想解决一个问题：**怎么让 robot 不用收集 real-world data 就能 zero-shot 干活**。

现在 monolithic VLA（RT-2、OpenVLA 那些）的问题很简单——你把 VLM fine-tune 去输出 action，结果发现 VLM 原本很强的 reasoning 能力被搞坏了，因为 model 的 capacity 被 visual understanding 和 action prediction 互相抢。而且你还得花大价钱去 robot 上 collect data。

作者的 solution：**别把所有事压在一个 model 里，拆成三层，每层干自己擅长的事**。

## 三层架构，打个比方

想象你要让 robot "把绿色 block 放到红色 mat 上"：

**第一层 ASM**——相当于 robot 的"眼睛+初级视觉皮层"。给它一张 RGB image 和 task 描述，它告诉你："绿色 block 在这几个 pixel 位置，红色 mat 在那几个 pixel 位置"。关键是它用了 SAM 做精确 segmentation，还做了 iterative refinement——先粗分一次，再让 MLLM 看看哪里分错了，给 positive/negative points 让 SAM 再分一次，最多迭代 n 次。

**第二层 3DAgent**——相当于 robot 的"前额叶皮层"，做 planning。它拿到第一层的 2D points，用 depth map 投影成 3D points，然后把 3D coordinates 当成 text 喂给 LLM。LLM 就开始推理："block 在 (0.25, 0.21, 0.11) 附近，mat 在那边，我得先 move 到 block 上方，close gripper，lift 起来，move 到 mat 上方，open gripper"。输出一个 waypoint 序列，最多 20 个点。

**第三层 HGM**——相当于 robot 的"运动皮层+小脑"。3D path 是粗略的，到了真正要 grasp 的时候，它用 GraspNet 在 cropped point cloud 上估计 precise 6-DoF grasp pose，再 filter 掉会碰撞的，选最接近 object center 的那个。

## 为什么这么设计？核心 insight

**Insight 1：VLM 不擅长给精确坐标**

你看 Table III 的数据，GPT-4o 在 RoboRefIt 上 affordance prediction accuracy 只有 15.3%，Qwen2.5-VL 24.1%，而 ASM 做到 63.4%。差距 3 倍。VLM 能理解 "这是 block"，但让它给 pixel-level 精确位置，它不行。SAM 行。所以让 VLM 做 reasoning + SAM 做 precision，组合起来。

**Insight 2：把 3D 问题变成 text 问题**

这是最 clever 的 trick。现在的 VLM 做 3D visual reasoning 很烂，但是 LLM 做 text reasoning 很强。那干嘛非要让 model 直接看 3D scene？把 3D points 写成 text 喂给 LLM 就行了。

比如输入给 LLM 的 prompt 长这样：
```
objects: [[cube, (0.25, 0.21, 0.11), (0.22, 0.23, 0.10), (0.23, 0.24, 0.11)],
          [mat, (0.74, 0.21, 0.05), ...]]
task: pick up cube and place on mat
```
LLM 一看就懂了，直接输出 trajectory。3 个 points 就能确定一个 plane，LLM 就能推断 object 的朝向。

消融实验证明了这点——只用 2D points，umbrella task 只有 4% success（不知道往哪个方向拔）；给 3D points，67%。

**Insight 3：Hierarchy 让每层保留自己的 strength**

monolithic VLA 的痛点是 fine-tune 的时候互相干扰。你 fine-tune VLM 输出 action，它的 visual understanding 就退化了。GeneralVLA 分开后，ASM 只管 2D affordance，3DAgent 只管 text-based planning，HGM 只管 grasp pose。各管各的，互不影响。

## KnowledgeBank：让 robot 记住经验

这个设计很有意思。每次执行任务后：
1. 用 LLM-as-judge 判断这次成功还是失败
2. 成功的 → 提取有效策略存进去
3. 失败的 → 提取 counterfactual signals，"这种情况下别这么做"
4. 下次遇到类似任务，先 retrieve top-k 相关经验注入 prompt

本质上就是给 robot 加了个 external memory，做 test-time learning。类似 MemGPT 或 RETRO 的思路，但用在 robotics 上。

## 实验结果有多强？

**Zero-shot（Table I）**：14 个 RLBench tasks，GeneralVLA 在 10 个上超过 baselines。有些 task baselines 全是 0%，GeneralVLA 能到 80%+。比如 Play Jenga——需要推断 block 的 3D pose 和 pulling direction，VoxPoser/CAP/Scaling-up 全部 0%，GeneralVLA 84.67%。

**Data generation（Table II）**：这是最 impressive 的。用 GeneralVLA 生成的 data 训练 RVT-2 policy，效果跟用 human demonstrations 训练的差不多，平均差 2.7%。有些 task 甚至更好——Put_block 用 GeneralVLA data 86.67%，用 RLBench human data 才 20%。

**Scaling（Fig. 7）**：GeneralVLA data 的 scaling slope 是 0.539，RLBench data 是 0.178。GeneralVLA data 越 scale 效果越好，3 倍效率。

这说明什么？**Foundation model 生成的 trajectory 可能比 human teleoperation 的更 consistent、更 optimal**。人操作 robot 有噪声、有 sub-optimal 决策，foundation model 的 planning 基于更系统的 reasoning。

## 几个关键的技术细节

**ASM 的公式**（Eq. 1）：
- LLM 输出里有个 special token `<SEG>`
- 取 LLM 最后一层对应这个 token 的 hidden state $\tilde{h}_{\mathrm{seg}}$
- 过个 MLP $\gamma$ 得到 segmentation embedding $h_{\mathrm{seg}}$
- SAM encoder 对 image 做 encoding 得到 features $f$
- SAM decoder 用 $h_{\mathrm{seg}}$ 和 $f$ 生成最终 mask $\hat{M}$

就是 LLM 负责 "我要分割什么"，SAM 负责 "怎么分割得精确"。

**每个 object 至少 3 个 points**：3 个点确定一个 plane，LLM 就能推断 object 的 spatial pose。1 个点不行（不知道朝向），2D points 不行（没有 depth 信息）。

**HGM 的 grasp selection**：GraspNet 会给多个 candidate grasp poses，HGM 先 filter 掉会碰撞的，再选 grasp center 最接近 object center 的。消融实验显示 w/o 3D point info 的话 HGM 直接 0%——没有 object 类别信息根本没法 grasp。

## 为什么我觉得这篇 paper 重要？

**第一，它给 robotics data scarcity 问题指了一条路**。如果 GeneralVLA 能 zero-shot 生成高质量 demonstration data，而且 scaling efficiency 比 human data 还好，那 robotics 的 data bottleneck 可能被 fundamentally 解决。这跟 LLM 领域用 strong model 生成 synthetic data 训练 small model 的思路一样。

**第二，"3D-as-text" 这个 formulation 很 pragmatic**。现在 VLM 做 3D reasoning 不行，与其等 VLM 进化，不如把 3D 信息转成 text 让 LLM 推理。Long term 我们可能需要 native 3D reasoning 的 foundation model（像 SpatialLLM 那个方向），但 short term 这个 workaround 很有效。

**第三，hierarchical decomposition 是 robotics 的 general principle**。Rod Brooks 的 Subsumption Architecture 时代就在讲这个，现在 foundation model 时代又重新验证了一遍。每个 module 专注自己擅长的，比 trying to do everything in one model 强。

## Limitations 也得说

- **Non-prehensile tasks 不行**：Push_block 22.67%，Open_box 35.33%，Insert_block 32.67%。这些需要 dynamic adjustment，当前架构搞不定。
- **Inference latency**：ASM + 3DAgent + iterative refinement，这个 pipeline 肯定慢。论文没 report timing，但 real-time dexterous manipulation 可能做不到。
- **Error propagation**：hierarchical 的天然问题——ASM 分错了，3DAgent 就 plan 错了，HGM 就 grasp 错了。Failure analysis 显示大部分 failure 在 action execution phase，but upstream errors 也有贡献。
- **单 viewpoint**：虽然 setup 有 4 个 camera，实际 reasoning 只用 front view。Multi-view fusion 没做。
- **GraspNet generalization**：HGM 依赖 M2T2，遇到完全 novel 的 object 可能不行。

## 跟其他工作的关系

- **VoxPoser**：用 LLM 生成 3D voxel value maps，只能做简单 value function，>4-DoF 就挂了。GeneralVLA 直接 plan 3D path，能力强很多。
- **Code-as-Policies**：LLM 生成 code 调 hand-crafted primitives。新场景没有 primitive 就废了。GeneralVLA 不需要 primitives。
- **LLARVA**：最接近的 monolithic VLA，把 trajectory prediction 当 auxiliary task。GeneralVLA 是 fully hierarchical，不做 trade-off。
- **RoboPoint**：ASM 的 data source 和 baseline。ASM 加了 SAM + iterative refinement，accuracy 从 24% 提到 63%。

## 我的 takeaway

这篇 paper 的核心 message 我觉得是：**foundation model 时代的 robotics，不要执着于 end-to-end**。每个 foundation model 有自己的 strength——VLM 擅长 semantic understanding，SAM 擅长 precise segmentation，LLM 擅长 text reasoning，GraspNet 擅长 grasp pose estimation。把它们 hierarchically 组合起来，比强行 fine-tune 一个 model 干所有事效果好得多。

而且这个 paradigm 的 bonus 是 data generation——如果 GeneralVLA 能持续生成高质量 robot data，那我们可能不需要那么依赖 expensive 的 human teleoperation 了。这跟 LLM 领域 synthetic data generation 的 trend 呼应，可能是 robotics scaling 的关键。

---

References:
- [GeneralVLA Paper](https://arxiv.org/abs/2506.07566)
- [GeneralVLA GitHub](https://github.com/AIGeeksGroup/GeneralVLA)
- [GeneralVLA Project Page](https://aigeeksgroup.github.io/GeneralVLA)
- [VoxPoser - CoRL 2023](https://proceedings.mlr.press/v229/huang23b.html)
- [Code as Policies - ICRA 2023](https://arxiv.org/abs/2209.07753)
- [RoboPoint - CoRL 2024](https://proceedings.mlr.press/v270/yuan25c.html)
- [LLARVA - CoRL 2024](https://proceedings.mlr.press/v270/niu25a.html)
- [RT-Trajectory - ICLR 2024](https://openreview.net/forum?id=F1TKzG8LJO)
- [RVT-2 - RSS 2024](https://doi.org/10.15607/RSS.2024.XX.055)
- [M2T2/GraspNet - CoRL 2023](https://proceedings.mlr.press/v229/yuan23a.html)
- [LISA - Reasoning Segmentation](https://arxiv.org/abs/2308.00692)
- [SegAgent](https://arxiv.org/abs/2503.08625)
- [RLBench](https://doi.org/10.1109/LRA.2020.2974707)
- [Scaling-up Distilling-down - CoRL 2023](https://proceedings.mlr.press/v229/ha23a.html)
- [OpenVLA - CoRL 2024](https://proceedings.mlr.press/v270/kim25c.html)
- [LLM-as-a-judge survey](https://www.sciencedirect.com/science/article/pii/S2590238524000)
- [SpatialLLM - CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Ma_SpatialLLM_CVPR_2025_paper.html)
- [SAM 2](https://arxiv.org/abs/2408.00714)

---

# GeneralVLA 技术深度讲解

## 1. 核心问题与 motivation

当前 monolithic VLA models（如 RT-2、OpenVLA、π0）面临一个根本矛盾：foundation models 在 vision/language domain 展现了惊人的 zero-shot generalization，however 在 robotics domain 却无法复现这种能力。GeneralVLA 的作者认为原因在于：

- VLM 虽然具备 semantic understanding，但是缺乏 fine-grained coordinate prediction 能力（不像 SAM、YOLO、Deformable-DETR 这类 task-specific models）
- monolithic 架构存在 trade-off：要 preserve long-horizon planning capability 就会 sacrifice visual understanding，反之亦然
- inference frequency 受限，无法实现 dexterous manipulation
- 严重依赖 expensive on-robot data

作者的 insight 是：与其试图把所有能力压进一个 end-to-end model，不如采用 hierarchical decomposition，让每一层专注于自己擅长的任务，thus 充分利用 foundation models 各自的 prior knowledge。

这种思路让我联想到你之前在 Tesla 提到的 "system 1 / system 2" 划分——fast reactive policy vs. slow deliberative planning。GeneralVLA 实际上把这个想法推向了三层：perception → reasoning/planning → control。

## 2. 整体架构解析

GeneralVLA 的 hierarchy 分为三层，我把它理解为三个 abstraction level：

### High-level: ASM (Affordance Segmentation Module)
- **输入**：RGB image + task text
- **输出**：2D affordance points + semantic labels（每个 object 至少 3 个 points）
- **作用**：scene understanding，识别 task-relevant objects 的 affordance regions

### Mid-level: 3DAgent (Knowledge-Guided Trajectory Planning)
- **输入**：task instruction + 3D points（由 2D points 通过 depth map back-project 得到）+ object semantics
- **输出**：3D path（waypoint 序列，最多 20 个 points，包含 gripper open/close 状态）
- **作用**：long-horizon planning，spatial reasoning，obstacle avoidance

### Low-level: 3D-aware control policy + HGM
- **输入**：3D path + RGB-D + localized point cloud
- **输出**：6-DoF grasp pose + executed trajectory
- **作用**：precise manipulation，grasp pose estimation

作者把 ASM + 3DAgent 的组合称为 **Hierarchical World Model**——因为这两个模块都 infuse 了 massive world priors，并且 dedicated to trajectory planning like classic world models。

这个架构的精妙之处在于 decoupling：
- ASM 只负责 2D affordance prediction，不需要 sacrifice visual understanding
- 3DAgent 只处理 text-based 3D reasoning，利用 LLM 的 textual generalization
- Low-level policy 专注于 3D spatial awareness + proprioceptive control

## 3. ASM 技术细节

### 3.1 核心公式解析

公式 (1) 描述了 ASM 的 segmentation 过程：

$$h_{\mathrm{seg}} = \gamma(\tilde{h}_{\mathrm{seg}})$$

- $\tilde{h}_{\mathrm{seg}}$：LLM 最后一层对应 `<SEG>` special token 的 hidden representation。这个 token 是在 LLM 输出 binary segmentation mask 时生成的。
- $\gamma$：MLP projection function，将 LLM 的 hidden space 映射到 SAM 的 embedding space
- $h_{\mathrm{seg}}$：最终的 segmentation embedding

$$f = \mathcal{F}_{\mathrm{enc}}(x_{\mathrm{img}})$$

- $\mathcal{F}_{\mathrm{enc}}$：SAM 的 image encoder
- $x_{\mathrm{img}}$：输入 RGB image
- $f$：image features

$$\hat{M} = \mathcal{F}_{\mathrm{dec}}(h_{\mathrm{seg}}, f)$$

- $\mathcal{F}_{\mathrm{dec}}$：SAM 的 mask decoder
- $\hat{M}$：最终的 segmentation mask

这个设计借鉴了 LISA（Lai et al., 2023）的思路——让 LLM 生成 `<SEG>` token 作为 semantic prompt，然后 SAM 根据这个 prompt 和 image features 生成 precise mask。这种架构同时利用了 LLM 的 reasoning 能力和 SAM 的 segmentation precision。

### 3.2 Iterative Refinement Mechanism

单次 segmentation 容易出现 over-segmentation 或 under-segmentation，which 会 propagate errors 到 3D geometry reconstruction。作者借鉴 SegAgent（Zhu et al., 2025）的设计，引入了 iterative refinement：

1. MLLM 评估 initial segmentation 结果
2. MLLM 提供 positive points（正确分割区域）和 negative points（错误分割区域）
3. 这些 points 作为 prompt 指导 SAM 进行下一轮 refinement
4. 重复最多 $n$ iterations，直到没有 negative points 产生

这种 human-annotator-like 的 interactive segmentation 思路非常 powerful，because 它把 segmentation 从 one-shot prediction 变成了 iterative correction process。

### 3.3 Spatial Affordance Prediction 的 multi-round 设计

在 Appendix 中，作者还提到了一个 trick：对每个 detected point，crop 原图的 1/4 区域 centered on that point，作为下一轮 recognition 的输入。每个 point 识别 3 次。这个 multi-scale / multi-round 的设计能显著提升 precision——类似 coarse-to-fine 的 detection pipeline。

### 3.4 VLM Fine-tuning Dataset

ASM 的训练数据来自 5 个 source（Appendix VII）：
1. **Pixel Point Pred Data**：来自 RoboPoint，347k samples，labels 是 unordered 2D points 或 bounding boxes
2. **LVIS**：从 LVIS dataset 随机采样 mask 内的 2D points，attach 语义信息
3. **Robot Data**：100k points 来自 Open X-Embodiment（Jaco Play arm）+ SIM-PLER 模拟数据
4. **VQA Data**：667K conversations from VQA dataset

消融实验（Table IV）显示每个 component 都贡献显著：
- No VQA: 52.1（↓11.3）
- No LVIS: 32.1（↓31.3，影响最大！）
- No Pixel: 48.2（↓15.2）
- No Sim: 51.9（↓11.5）
- No Robo: 56.2（↓7.2）
- All: 63.4

LVIS 的贡献最大，说明 precise semantic + location information 对 affordance prediction 至关重要。

### 3.5 ASM 性能对比

Table III 展示了 ASM 在 RoboRefIt benchmark 上的表现：

| Method | Accuracy % |
|--------|------------|
| GPT-4o | 15.3 ± 1.3 |
| LLaVA-NeXT | 20.0 ± 0.9 |
| Qwen2.5-VL | 24.1 ± 0.9 |
| SpatialLLM | 21.3 ± 0.9 |
| **ASM** | **63.4 ± 1.4** |

ASM 几乎是其他 VLM 的 3 倍 accuracy！这个 gap 非常惊人，说明单纯的 VLM 在 spatial affordance prediction 上确实存在 fundamental limitation，而 ASM 的 iterative refinement + SAM integration 是有效的解决方案。

## 4. 3DAgent 技术细节

### 4.1 设计思路

3DAgent 的核心 insight 是：现有 visual foundation models 在 3D scene understanding 上能力不足，但是 LLM 在 text-based reasoning 上非常强大。于是作者把 3D 问题转化为 text problem：

1. 2D points → depth map projection → 3D points
2. 3D points + object semantics → text representation
3. Text representation + task instruction → LLM input
4. LLM 输出 → 3D trajectory

这种 "3D-as-text" 的 formulation 非常巧妙，because 它完全 bypass 了 3D visual reasoning 的难题，转而利用 LLM 已经在 text domain 证明了的 generalization 能力。

### 4.2 为什么每个 object 至少 3 个 points？

作者发现当每个 object 的 points 数量 ≥ 3 时，LLM 能有效理解 object 的 spatial pose。Table VI 的消融实验验证了这一点：

| Method | Take umbrella | Put block |
|--------|---------------|-----------|
| 3DAgent-2D | 4.00 ± 4.00 | 19.33 ± 3.06 |
| 3DAgent-1point | 24.00 ± 4.00 | 82.00 ± 7.21 |
| 3DAgent w/o obstacle | 23.33 ± 7.57 | 91.33 ± 4.16 |
| 3DAgent | **67.33 ± 14.05** | **93.33 ± 3.06** |

- 3DAgent-2D（只用 2D）：umbrella 任务几乎完全失败（4%），因为无法判断 umbrella bag 的 3D orientation
- 3DAgent-1point（每个 object 只有 1 个 3D point）：umbrella 任务 24%，因为单点无法推断 object pose
- 3DAgent w/o obstacle：umbrella 任务 23.33%，因为无法判断 obstacle orientation 来选择正确的 pulling direction

3 个 points 能确定一个 plane，which 足以推断 object 的 approximate pose。这是非常 elegant 的几何 insight。

### 4.3 KnowledgeBank 机制

3DAgent 配备了一个 KnowledgeBank，用于跨 task 的经验积累。这是一个 closed-loop process：

**Step 1: Knowledge Retrieval**
- Agent 用 current query context 查询 KnowledgeBank
- 通过 embedding-based similarity search 找到 top-k relevant experiences
- Retrieved items 注入 agent 的 system instruction

**Step 2: Knowledge Construction**
- 任务完成后，使用 LLM-as-a-judge（Gu et al., 2024）评估 trajectory 的 success/failure
- 成功经验 → validated manipulation strategies
- 失败经验 → counterfactual signals + pitfalls（用于 sharpen guardrails）
- 每个 trajectory/experience 提取多个 knowledge items

**Step 3: Knowledge Consolidation**
- 通过 simple addition operation 将新 knowledge items 加入 KnowledgeBank
- 维持一个不断 evolved 的 knowledge repository

这种设计让我联想到 RETRO、MemGPT 等 retrieval-augmented LLM 架构，以及你的 Eureka leverages LLM for reward design 的工作。KnowledgeBank 本质上是一个 external memory，让 agent 能 test-time learning。

### 4.4 Trajectory 格式

3DAgent 输出的 trajectory 格式（来自 prompt）：

```
<ans>[(0.25, 0.32, 0.10), (0.32, 0.17, 0.10),
<action>CloseGripper</action>, (0.13, 0.24, 0.10),
<action>OpenGripper</action>, (0.74, 0.21, 0.20),
<action>Grasp</action>, ...]</ans>
```

- 每个 tuple $(x, y, z)$ 是 end-effector 在 3D space 的位置，normalized 到 [0, 1]
- `<action>` tags 表示 gripper 状态变化
- 最多 20 个 points（保证 planning stability）

## 5. HGM (Hybrid Grasping Module) 技术细节

### 5.1 设计动机

3D path 是 macroscopic coarse trajectory，but 在 grasp point 需要精确的 grasp pose estimation。HGM 的作用是 bridge 这个 gap。

### 5.2 工作流程

1. **3D Spatial Range Determination**：用 3D point information 定位 object 的 3D spatial range
2. **Point Cloud Cropping**：RGB + depth fusion，通过 inverse projection 得到 point cloud，then crop 到 object 附近
3. **Grasp Pose Estimation**：使用 GraspNet（M2T2, Yuan et al., 2023）估计 grasp pose
4. **Collision Filtering**：过滤掉会碰撞的 grasp candidates
5. **Nearest Selection**：选择 grasp center 最接近 object center point 的 grasp pose

### 5.3 消融实验

Table VII：

| Method | Play jenga | Take umbrella |
|--------|------------|---------------|
| HGM w/o rgb | 56.67 ± 4.53 | 32.33 ± 14.03 |
| HGM w/o 3D point | 0.00 ± 0.00 | 0.00 ± 0.00 |
| HGM w/o filter-C | 58.00 ± 5.52 | 53.00 ± 12.41 |
| HGM w/o filter-N | 76.33 ± 7.24 | 54.67 ± 14.00 |
| HGM | **84.67 ± 11.02** | **67.33 ± 14.05** |

关键发现：
- **HGM w/o 3D point = 0%**：完全没有 3D point 信息，HGM 无法确定 grasped object 类别，完全失败
- **HGM w/o rgb**：只用 depth，性能下降明显，说明 RGB 的 visual information 对 grasp pose estimation 重要
- **filter-C（collision）和 filter-N（nearest）**：各自贡献约 8-14% 的提升

## 6. 实验结果深度分析

### 6.1 Zero-shot Performance (Table I)

14 个 RLBench tasks 的 zero-shot success rate：

GeneralVLA 在 10/14 tasks 上超过 baselines（VoxPoser、CAP、Scaling-up）。特别值得注意的几个 task：

- **Play_jenga**：GeneralVLA 84.67%，所有 baselines = 0%。这个 task 需要精确推断 jenga block 的 3D pose 并选择正确的 pulling direction，只有 GeneralVLA 的 3D reasoning 能力能解决
- **Open_jar**：GeneralVLA 84%，Scaling-up 78.67%，其他 = 0%
- **Close_box**：GeneralVLA 52%，其他全部 = 0%
- **Open_box**：GeneralVLA 35.33%，其他全部 = 0%

VoxPoser 在需要 >4-DoF arm movement 的 tasks 上完全失败，because 它的 value map formulation 无法 handle complex trajectories。

GeneralVLA 表现较差的 3 个 tasks（Push_block 22.67%、Open_box 35.33%、Insert_block 32.67%）都是 non-prehensile 或需要 fine-grained dynamic adjustment 的 tasks——这暴露了当前方法在 dynamic manipulation 上的 limitation。

### 6.2 Behavior Cloning Data Quality (Table II)

这是论文最 impressive 的结果之一。用 GeneralVLA 生成的 data 训练 RVT-2 policy：

- **GeneralVLA data vs. RLBench human demonstrations**：平均差异仅 2.7%！
- 在 10/12 tasks 上，GeneralVLA data 训练的 policy 是所有 autonomous data generation 方法中最好的
- Variance：GeneralVLA data 训练的 policy std = 6.24，而 zero-shot deployment std = 11.02

特别值得关注的 task：
- **Put_block**：GeneralVLA data 86.67% vs. RLBench 20%（GeneralVLA data 反而更好！）
- **Lamp_on**：GeneralVLA data 88.67% vs. RLBench 84%
- **Take_umbrella**：GeneralVLA data 87.33% vs. RLBench 58.67%

这个结果暗示 GeneralVLA 生成的 trajectories 比 human demonstrations 更加 consistent 和 diverse——human demonstrations 可能存在 sub-optimal trajectories，而 GeneralVLA 的 planning 是基于 foundation model 的 optimal reasoning。

### 6.3 Scaling Experiment (Fig. 7)

Linear fit slope：
- GeneralVLA data: **0.539**
- RLBench data: 0.178

GeneralVLA data 的 scaling efficiency 是 RLBench 的 3 倍！这意味着随着 data 量增加，GeneralVLA data 训练的 policy 性能提升更快。这个结果对于 robotics data scarcity 问题非常有意义——GeneralVLA 不仅能生成 data，而且生成的 data quality 更适合 scaling。

### 6.4 Real-world Experiments (Table V)

4 个 real-world tasks 的 zero-shot success rate：

| Task | CAP | Robopoint | GeneralVLA |
|------|-----|-----------|------------|
| Move_spray_bottle | 6.67 | 0.00 | **63.33** |
| Open_drawer | 0.00 | 0.00 | **36.67** |
| Open_jar | 36.67 | 20.00 | **50.00** |
| Sort_object | 70.00 | 63.33 | **76.67** |

Real-world 结果证实了 simulation 结果的 generalization。特别值得注意的是 Open_drawer task：CAP 完全失败（没有 pre-designed primitive for opening drawers），而 GeneralVLA 通过 3D reasoning 能推断 drawer orientation 并规划 trajectory。

## 7. 关键 Insights 与 Intuition Building

### 7.1 为什么 hierarchical 比 monolithic 好？

monolithic VLA 的 fundamental problem 是 "capacity competition"——visual understanding、semantic reasoning、spatial planning、motor control 这些能力在一个 model 里互相竞争 capacity。GeneralVLA 的 hierarchical design 让每个 layer 专注于自己的 strength：

- VLM/SAM 擅长 visual perception → ASM
- LLM 擅长 text-based reasoning → 3DAgent
- Specialized grasp models 擅长 motor control → HGM

这种 "divide and conquer" 的思路在 robotics 领域非常 powerful，because robotics task 本身就是 multi-modal 的。

### 7.2 "3D-as-text" 的 formulation

这是论文最 clever 的设计之一。作者没有试图让 VLM 直接理解 3D scene（which 是当前 VLM 的 weak point），而是把 3D 信息转化为 text representation，然后利用 LLM 的 textual reasoning 能力。这种 problem reformulation 的思路值得借鉴——当某个 modality 的 direct reasoning 能力不足时，可以转化到 model 擅长的 modality。

### 7.3 Knowledge Bank 的 test-time learning

KnowledgeBank 的设计让 agent 能在 deployment 过程中不断 accumulate experience。这与 traditional zero-shot methods（每次都从头 planning）形成对比。这种 test-time learning 思路与 DeepMind 的 Algorithm Distillation、你的 work on in-context learning 有相似的精神——利用 history 来 improve current performance。

### 7.4 Data Generation 的价值

GeneralVLA 的另一个重要价值是作为 data generation engine。Table II 和 Fig. 7 证明了 GeneralVLA 生成的 data quality 接近甚至超过 human demonstrations，且 scaling efficiency 更好。这对于解决 robotics 的 data scarcity bottleneck 非常关键。

这个 insight 让我联想到 LLM 领域的 synthetic data generation（如 Constitutional AI、self-instruct）——用 strong model 生成 data 训练 specialized model。GeneralVLA 把这个 paradigm 带到了 robotics 领域。

## 8. Limitations 与 Future Work

论文承认的 limitations：
1. **VLM 的 spatial perception 限制**：当前只用 VLM 做 2D point estimation，future work 可以增强 VLM 的 3D pose estimation 能力
2. **Non-prehensile tasks 表现较差**：Push_block、Open_box、Insert_block 等需要 dynamic adjustment 的 tasks 表现不佳
3. **Single viewpoint**：当前只用 front viewpoint（虽然提到 four-camera setup，但实际 reasoning 只用 front）

我额外观察到的一些潜在 issues：
- **Inference speed**：ASM + 3DAgent 的 hierarchical pipeline 在 inference 时可能有显著 latency，论文没有详细 report timing
- **Error propagation**：hierarchical design 的 downside 是 error 会从 high-level propagate 到 low-level。论文的 failure analysis（Fig. 11）显示 Play Jenga task 82% 成功，most failures 在 action execution phase，but ASM 和 3DAgent 的 errors 也会 contribute
- **KnowledgeBank 的 scalability**：随着 knowledge items 增加，retrieval 的 relevance 可能下降，需要更好的 organization mechanism
- **GraspNet 的 generalization**：HGM 依赖 GraspNet（M2T2），其 generalization 到 novel objects 的能力可能有限

## 9. 与相关工作的关联

### 9.1 VoxPoser [22]
VoxPoser 用 LLM 生成 3D voxel value maps 来 guide manipulation。它的 limitation 是只能 produce simple value functions，struggle with complex long-horizon tasks，且无法 handle >4-DoF movements。GeneralVLA 通过 explicit 3D path planning 解决了这些问题。

### 9.2 Code-as-Policies (CAP) [39]
CAP 用 LLM 生成 code 调用 hand-crafted primitive actions。它的 limitation 是 heavily dependent on pre-designed primitives，无法 handle 新场景。GeneralVLA 不需要 primitive actions，直接生成 3D trajectories。

### 9.3 RT-Trajectory [15]
RT-Trajectory 提出 trajectory-based task specification 来 condition low-level policies。GeneralVLA 借鉴了这个思路，但在 3D space 进行 planning 并实现了 zero-shot completion。

### 9.4 LLARVA [51]
LLARVA 是最接近的 monolithic VLA，它 predict end-effector trajectories 作为 auxiliary task 来 improve action prediction。但 LLARVA 只是辅助任务，GeneralVLA 则是 fully hierarchical approach。

### 9.5 Manipulate-Anything [10]
Manipulate-Anything 也用 VLMs 自动化 real-world robots，但 GeneralVLA 的 hierarchical design 和 KnowledgeBank 机制使其在 long-horizon planning 和 cross-task generalization 上更强。

### 9.6 RoboPoint [73]
RoboPoint 是 ASM 的重要 baseline 和 data source。ASM 在 RoboPoint 基础上加入了 SAM integration 和 iterative refinement，显著提升了 precision（63.4% vs. 24.1% for Qwen2.5-VL）。

## 10. 对 Robotics 领域的 broader implications

### 10.1 Data Generation Paradigm
GeneralVLA 验证了一个重要的 paradigm：用 foundation models 作为 data generation engine 来 train specialized policies。如果这个 paradigm 成立，robotics 的 data bottleneck 问题可能被 fundamentally 解决——we can generate unlimited training data using VLMs + LLMs + simulation。

### 10.2 Hierarchy as General Principle
GeneralVLA 的成功进一步验证了 hierarchy 在 robotics 中的价值。这与 Rod Brooks 的 Subsumption Architecture、 recent work on foundation models for robotics 的 trend 一致。未来的 VLA 可能会采用更复杂的 hierarchy，包含更多 layers（如 task decomposition、skill selection、motion planning、control）。

### 10.3 Test-time Learning
KnowledgeBank 的 test-time learning 机制是一个 promising direction。如果 robot 能在 deployment 中不断 accumulate experience 并 improve performance，这将是一个重要的 step towards truly general robot intelligence。

### 10.4 3D Reasoning 的 Text-based Formulation
"3D-as-text" 的 formulation 可能是一个重要的 methodological contribution。如果 LLM 在 3D visual reasoning 上能力有限，but 在 text-based reasoning 上强大，那么把 3D 问题 reformulate 为 text problem 是一个 pragmatic 的解决方案。当然，long-term 来看，我们可能需要 native 3D reasoning 能力的 foundation models（如 SpatialLLM [44]）。

## 11. References

- [GeneralVLA Paper](https://arxiv.org/abs/2506.07566) - 论文 arXiv
- [GeneralVLA GitHub](https://github.com/AIGeeksGroup/GeneralVLA) - 官方代码
- [GeneralVLA Project Page](https://aigeeksgroup.github.io/GeneralVLA) - 项目主页
- [VoxPoser](https://proceedings.mlr.press/v229/huang23b.html) - CoRL 2023
- [Code as Policies](https://arxiv.org/abs/2209.07753) - ICRA 2023
- [RT-Trajectory](https://openreview.net/forum?id=F1TKzG8LJO) - ICLR 2024
- [LLARVA](https://proceedings.mlr.press/v270/niu25a.html) - CoRL 2024
- [RoboPoint](https://proceedings.mlr.press/v270/yuan25c.html) - CoRL 2024
- [LISA](https://arxiv.org/abs/2308.00692) - Reasoning segmentation via LLM
- [SegAgent](https://arxiv.org/abs/2503.08625) - Interactive segmentation refinement
- [RVT-2](https://doi.org/10.15607/RSS.2024.XX.055) - RSS 2024
- [M2T2 / GraspNet](https://proceedings.mlr.press/v229/yuan23a.html) - CoRL 2023
- [RLBench](https://doi.org/10.1109/LRA.2020.2974707) - Robot learning benchmark
- [Scaling-up Distilling-down](https://proceedings.mlr.press/v229/ha23a.html) - CoRL 2023
- [OpenVLA](https://proceedings.mlr.press/v270/kim25c.html) - CoRL 2024
- [LLM-as-a-judge survey](https://www.sciencedirect.com/science/article/pii/S2590238524000) - 评估方法
- [SpatialLLM](https://openaccess.thecvf.com/content/CVPR2025/html/Ma_SpatialLLM_CVPR_2025_paper.html) - CVPR 2025
- [SAM 2](https://arxiv.org/abs/2408.00714) - Segment Anything in images and videos
- [TAPIR](https://doi.org/10.1109/ICCV51070.2023.00923) - Tracking any point

---

**总结**：GeneralVLA 的核心贡献在于证明了 hierarchical decomposition 能 effectively leverage foundation models 的 prior knowledge，从而实现 robotics 的 zero-shot generalization。它的 "3D-as-text" formulation、iterative segmentation refinement、KnowledgeBank test-time learning 都是值得借鉴的技术创新。更重要的是，它作为 data generation engine 的价值可能比 zero-shot deployment 本身更有意义——如果这个 paradigm 成立，robotics 的 data scarcity bottleneck 可能被 fundamentally addressed。
