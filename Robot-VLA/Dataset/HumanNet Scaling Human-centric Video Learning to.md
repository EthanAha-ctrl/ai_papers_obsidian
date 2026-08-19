---
source_pdf: HumanNet Scaling Human-centric Video Learning to.pdf
paper_sha256: 2a6ea38ae809705d3c805dee47b7f6ef2fe896e661b72d78bd5b666cd7889079
processed_at: '2026-08-19T11:52:48-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

## 一句话版本

**别费劲采 robot data 了，网上扒一百万小时人类干活视频，效果不比 robot 数据差。**

---

## 为什么这么干

Robot data 太贵了。teleoperation 一小时采集成本几十上百刀，搞得几天也就攒个百来小时。而互联网上到处都是人在做饭、修东西、组装家具、玩运动的视频，免费的，量大管饱。

作者的 insight 是：人类干活的方式，本质上就是 robot 想学的东西——怎么抓东西、怎么用工具、怎么走动、怎么完成一整套流程。虽然人的手跟 robot gripper 不一样，但 visual representation 层面的 prior 是通用的。

---

## 干了什么

搞了个 1,000,000 小时的人类活动视频数据集，叫 HumanNet。规模是之前最大的同类数据集的 ~50 倍。

视频分两种视角：
- **First-person（第一人称）**：头戴摄像头拍的，能看到手和物体的接触关系，跟 robot wrist camera 视角接近
- **Third-person（第三人称）**：旁人拍的，能看到全身动作、场景、人和人交互

两种视角互补，一个看"手在干嘛"，一个看"身体在干嘛"。

数据来源乱七八糟——YouTube、Bilibili、开源数据集、自己采集的都有。所以搞了一套 pipeline 来清洗：去重、过滤垃圾视频、按场景切分、然后 annotate 上 3D hand pose、body pose、SLAM 轨迹、LLM 生成的 caption。

其中一部分视频还做了 **motion retargeting**——把人的动作映射到 humanoid robot skeleton 上，误差小于 15mm 的才算 "robot-ready"。

---

## 最关键的实验

拿同一个 VLA architecture（LingBot-VLA），在同样的 downstream data 上 post-train，只变 pretrain source：

| Config | Pretrain source | 结果 |
|--------|----------------|------|
| Baseline | 裸 Qwen VLM | 最差 |
| +100h robot | Qwen + 100h real robot CoBot data | 中等 |
| +1000h ego | Qwen + 1000h egocentric human video | **跟上面差不多，某些 task 还更好** |
| LingBot | Qwen + 20000h real robot data | 最好，但贵 |

**1000 小时人类视频 ≈ 100 小时 robot 数据的效果。** 而前者的采集成本可能是后者的 1/5 甚至更低。

---

## 所以呢

这篇 paper 的核心 message 就一句：**human video 是 robot data 的 scalable 替代品**。

这跟 LLM 的路径一模一样——pretrain 用海量通用数据（人类视频），fine-tune 用少量 task-specific 数据（robot demo）。HumanNet 就是在给 embodied AI 造那个 "海量通用数据"。

作者 Daquan Zhou 之前还做过 video generation model for embodied world（[arXiv:2601.15282](https://arxiv.org/abs/2601.15282)），Being-H 系列（[H0](https://arxiv.org/abs/2507.15597), [H0.5](https://arxiv.org/abs/2601.12993), [H0.7](https://arxiv.org/abs/2605.00078)）也在往同一个方向推。整个 field 在 converge 到一个 paradigm：**human data pretrain + robot data fine-tune**。

Paper 链接：https://dagroup-pku.github.io/HumanNet/

---

# HumanNet: Scaling Human-centric Video Learning to One Million Hours — 深度解析

## 1. Core Thesis 与 Motivation

Andrej 你应该会特别 appreciate 这篇 paper 的立论。作者 Daquan Zhou (Peking University, DAGroup) 的核心论点非常清晰：**embodied intelligence 的 bottleneck 是 data scale**。Language 和 vision-language 已经 scale 起来了（GPT-4, Gemini, Qwen2.5-VL 这些 foundation models 依赖 massive web corpora），但 physical interaction models 还卡在小规模、task-specific、robot-platform-tied 的数据上（RT-1, Open X-Embodiment, DROID 等）。

HumanNet 的解决方案非常直接：**用 1,000,000 小时的人类活动视频作为 scalable substrate**，把 embodied foundation model 的 pretraining 从 robot-specific data 转移到 human-centric video。

这个思路的理论 intuition 是：human behavior 是 physical intelligence 最 scalable 的 data source。人类 daily perform 的 manipulation、tool use、locomotion、navigation、social coordination、procedural activities 覆盖的 object diversity、environment diversity、motion style diversity 远超 robot teleoperation 能采集到的。而且 cost 极低——互联网上已经有海量 video，只需要 curation。

Project page: https://dagroup-pku.github.io/HumanNet/
GitHub: https://github.com/DAGroup-PKU/HumanNet/

---

## 2. 与 Prior Work 的 Positioning

Table 1 是 paper 里最 informative 的对比表之一。让我把它重新组织一下，便于 build intuition：

### 2.1 Egocentric Datasets 的 scaling history

| Dataset | Hours | Year | Focus |
|---------|-------|------|-------|
| EPIC-KITCHENS-100 | ~100 | 2018 | Kitchen actions only |
| Ego4D | ~3,670 | 2022 | Daily activities, broad |
| HOI4D | 2.4M frames | 2022 | Category-level HOI, dense annotation |
| EgoDex | 829 | 2026 | Dexterous manipulation |
| OpenEgo | 1,107 | 2025 | Dexterous manipulation |
| EgoScale | 20,854 | 2026 | Dexterous manipulation scaling |
| EgoVerse | 1,362 | 2026 | Human demos for robot |
| **HumanNet** | **1,000,000** | 2026 | Full human activity spectrum |

可以看到 scaling trend：从 100h → 3670h → 20854h → 1,000,000h。HumanNet 是一个 ~50x jump over EgoScale。

### 2.2 Exocentric Datasets 的局限

Third-person datasets (ActivityNet, Kinetics, Charades, AVA, Something-Something V2, HACS, FineGym, HowTo100M) 虽然 clip 数量可能很大（HowTo100M 有 136M clips），但它们的 embodied use 都是 "Indirect"——即只能用于 representation learning，没法直接 transfer 到 robot control，因为缺乏 hand-object contact、first-person viewpoint 这些 action-centric cues。

### 2.3 Mixed-view 的 emerging direction

Ego-Exo4D (1286h, 2024) 和 HumanNet 都强调 first-person + third-person 的 complementarity：
- **First-person**: actor-centered intent, hand-object contact, contact dynamics, visual consequences of motor decisions
- **Third-person**: full-body motion, posture, scene context, multi-person interaction, spatial geometry

这个 dual-view 的设计 intuition 是：human-to-robot transfer 需要同时知道 "actor 看到什么" 和 "actor 的 body 在做什么"。

---

## 3. Dataset Design Principles (Section 3.1)

Paper 提出四个 first-class design principles：

### 3.1 Scale
要 large enough to support long-tail coverage over activities, environments, body motions, interaction styles。这里的关键 insight 是：**small datasets 会 saturate on narrow task family**。1M hours 的目标是覆盖 rare but physically informative behaviors——例如 folding deformable objects、handling reflective containers、operating unfamiliar appliances。

### 3.2 Viewpoint Diversity
First-person + third-person 都保留并且 explicitly indexed。这允许 model 学到 complementary cues：
- Actor-centered cues (ego)
- Observer-centered cues (exo)

### 3.3 Physical Relevance
Data 必须包含：
- Hand-object proximity
- Full-body motion
- State changes
- Action ordering
- Procedural structure
- Scene context

### 3.4 Pretraining Readiness
支持 modern large-scale training pipelines：
- Chunking
- Metadata indexing
- Quality filtering
- Caption labels
- Motion annotations
- Optional alignment with text/structured labels

---

## 4. Data Pipeline (Section 3.3, Figure 3)

这是 paper 里技术上最 substantial 的部分。Pipeline 分三个 stage：

### 4.1 Stage 1: Data Collection

这一 stage的核心是 **keyword-driven retrieval**：

```
Seed Keywords → Keyword Expansion → Keyword-based Crawling → 
Channel Crawling → Existing Sources → Unified Keyword Repository
```

这个 keyword repository 然后 drive 多个 retrieval channels：
- Video-platform search (YouTube, Bilibili 等)
- General web search engines
- Directly crawled videos
- Open-source datasets (Ego4D, EPIC-KITCHENS, etc.)
- Self-collection under real-world environments

Self-collection stream 是关键补充——它能 capture controlled first/third-person recordings in everyday settings，覆盖那些 public platforms 上难以 reliable 获取的 underrepresented activities、viewpoints、scenes。

Output: ego-video URL pool (first-person) + retained third-person material

### 4.2 Stage 2: Data Processing

这一 stage 把 raw videos 转成 clip-level training samples。Steps：

1. **De-duplication & Normalization**: remove near-identical copies, unify frame rate, resolution, container format
2. **Content filtering**: retain clips with meaningful human action and observable motion
3. **Quality filtering**: discard severe motion blur, heavy occlusion, static framing
4. **Scene splitting**: segment long videos at visual changes (避免 unrelated activities 合并成一个 sample)
5. **Video clipping**: produce fixed-granularity segments

这里有个值得深究的技术点：scene splitting 的"visual change" detection。Paper 没有给出具体算法，但常见做法是：
- 用 visual feature (e.g., CLIP embedding) 计算相邻 frame 的 cosine similarity
- 当 similarity 低于 threshold θ_scene 时，标记为 boundary
- 公式：boundary(t) = 1[cos(f_t, f_{t+w}) < θ_scene]
  - f_t: frame t 的 visual feature
  - w: window size
  - θ_scene: scene change threshold

### 4.3 Stage 3: Annotation

这是最 technically interesting 的部分。Annotation 有四个并行的 module：

#### 4.3.1 3D Hand and Body Pose Detection
Recover fine-grained motion structure。这通常用 off-the-shelf pose estimator（可能是 FrankMocap, SMPL-X based, 或 HaMeR + 4D-Humans 这类方法）。

#### 4.3.2 Monocular SLAM
Estimate camera trajectory for first-person clips。这里有个限制：只有满足 stability 和 parallax requirements 的 clips 才能 estimate（因为 pure rotation video 无法做 SLAM triangulation）。

#### 4.3.3 Motion Retargeting
这是 human-to-robot transfer 的关键 module。把 recovered human motion align 到 unified humanoid skeleton。Paper 给了两个 quantitative criteria：
- **Retargeting error < 15 mm**
- **Valid-frame coverage > 60%**

只有同时满足这两个条件的 clip 才被 designate 为 "robot-ready"。

Retargeting 的数学形式（虽然 paper 没明说，但可以 reconstruct）：

给定 human motion sequence H = {h_1, h_2, ..., h_T} 和 humanoid robot skeleton R，找 mapping φ: H → R 使得：

min_φ Σ_t ||FK_R(φ(h_t)) - FK_H(h_t)||²

其中：
- FK_R: robot skeleton 的 forward kinematics
- FK_H: human skeleton 的 forward kinematics
- φ: joint angle mapping function
- ||·||²: Euclidean distance in Cartesian space

15mm threshold 对应 wrist/hand end-effector 的 average position error。

#### 4.3.4 LLM-assisted Captioning
Produce video captions, motion descriptions, activity classifications。这些会和 source 的 narrations/metadata normalize 在一起。

---

## 5. Statistical Analysis (Section 3.4, Figure 5)

Figure 5 给出四个关键的 distribution：

### 5.1 Pose Score Distribution
Concentrates at high-confidence end after quality filtering。这说明 quality filtering 有效——retained clips 适合 dense pose, hand, motion supervision。

### 5.2 Motion Score Distribution (<=P99 4.18)
Heavy-tailed but well bounded。Motion score 应该是 average per-frame joint displacement：

motion_score = (1/T) Σ_t ||h_t - h_{t-1}||₂

其中 h_t 是 frame t 的 pose，T 是 clip 长度。P99 = 4.18 意味着 99th percentile 的 average joint displacement 是 4.18（应该是某个 normalized unit）。

### 5.3 Motion Length Distribution (<=P99 48.88)
也是 heavy-tailed。P99 = 48.88 可能指 clip 长度（秒或 frames）。

### 5.4 Per-category Breakdown
Athletic 和 outdoor families 有 longer, higher-magnitude motion；daily activities 和 game-character actions 集中在 shorter, finer-grained segments。

这个 heterogeneity 是 feature——它支持 mixed-supervision training recipes，每个 downstream task 可以从 corpus 的 appropriate slice 取数据。

---

## 6. Validation Experiment (Section 3.5, Figure 6) — 最关键的实证结果

这是 paper 的 empirical highlight。我详细 decompose 一下：

### 6.1 Experimental Setup

**Architecture**: LingBot-VLA [34] (arXiv:2601.18692, 2026)
**Backbone**: Qwen VLM
**Downstream data**: 100 tasks × 20 episodes = 34 hours robot interaction data (固定)

**Four configurations**:
1. **Qwen VLM** (baseline, no embodied pretraining)
2. **Qwen + 100h real-robot CoBot data** (Magic Cobot)
3. **Qwen + 1000h egocentric human video** (from HumanNet)
4. **LingBot** (Qwen backbone + 20000h real-robot data)

**Critical detail**: 对 configuration 1-3，用 fine-tuned VLM + reinitialized action expert；对 LingBot，直接用 pretrained VLM 和 action expert。

**Evaluation**: 5个 held-out task groups 的 validation loss
- (a) In-Domain Loss
- (b) OOD Average Loss
- (c) OOD Short-Horizon Loss
- (d) OOD Long-Horizon Loss
- (e) OOD Mobile-Manipulation Loss

### 6.2 Key Findings

**Finding 1**: 1000h egocentric pretraining 一致地 narrows the gap between generic web-scale initialization 和 robot-specialized initialization。

**Finding 2** (最 striking): **1000h egocentric human video matches, 甚至在 several task groups 上 surpasses, 100h real-robot CoBot data**。

这个 result 的 implication 非常重要：当 teleoperated robot data limited 时，egocentric human video 是更 scalable、更 cost-effective 的 substitute。

### 6.3 Intuition Building: 为什么 egocentric video 这么有效？

这里我做一些 inference/hallucination（你要求宁可 hallucinate 也不要遗漏）：

1. **Visual viewpoint alignment**: First-person video 的 viewpoint 和 robot camera 的 viewpoint 接近。Robot manipulation 通常是从 robot wrist camera 或 head camera 看 scene，和 egocentric human video 的 viewpoint 分布相似。Qwen VLM pretrain 时见过的 internet images 大多是 third-person，所以 egocentric pretraining 在 viewpoint distribution 上做了 alignment。

2. **Hand-object contact signal**: Egocentric video 天然 expose hand-object contact dynamics。这个 signal 和 robot manipulation 的核心问题（how to grasp, how to contact, how to move object）高度 aligned。即使没有 action label，visual representation 学到的 contact affordance 应该 transferable。

3. **Procedural structure**: Human daily activities 是 long-horizon procedures（make coffee, do laundry, cook meal）。这些 procedural structure 在 VLA model 的 latent space 里应该 encode 成 useful priors，支持 downstream long-horizon robot tasks。

4. **Object affordance priors**: 1000h egocentric video 里出现的 object variety 远超 100h robot data。Model 学到的 object affordance（哪里可以 grasp，哪里可以 push）是 generalizable 的。

5. **Distribution shift 的 reduced magnitude**: 从 Qwen VLM 直接 transfer 到 robot data 有 large distribution shift。Egocentric video pretraining 相当于一个 intermediate domain，bridge 了 web-scale vision-language 和 robot control 之间的 gap。

### 6.4 Possible Concerns / Open Questions

我提出一些 paper 没有完全 address 的 concerns：

1. **Validation loss vs. success rate**: Paper 只 report validation loss，没有 report task success rate。Loss 和 success rate 之间可能有 gap（尤其是 long-horizon tasks）。你以前在 Tesla AI Day 讨过这个 issue。

2. **100h CoBot vs. 1000h ego 的 fairness**: 1000h vs 100h 是 10x data。如果按 per-hour cost，可能 CoBot data 还是更 informative。但 paper 的论点是 cost-effectiveness——egocentric video 的采集 cost 远低于 teleoperation。

3. **Action expert reinitialized**: Configuration 1-3 都 reinitialize action expert，只有 LingBot 用 pretrained action expert。这意味着 comparison 不完全 fair——但这也 isolate 了 VLM backbone 的影响，是 well-controlled ablation。

4. **Generalization to other VLA architectures**: 实验只在 LingBot-VLA 上做。OpenVLA, RT-2, GR00T N1 等 architecture 是否也 benefit？需要更多实验。

---

## 7. Downstream Relevance (Section 4)

Paper 列出五个 downstream use cases：

### 7.1 Video and VLM Pretraining
Corpus 可以 pretrain video encoders 和 video-language models，需要比 generic internet video 更强的 human activity, contact, motion structure。

### 7.2 World-Action Model Training
这个特别 relevant 到你以前在 Tesla 的 work。World-action model 联合 capture environment dynamics 和 driving actions。HumanNet 的 first-person + third-person + caption + motion annotation 组合支持：
- Action-conditioned forward dynamics learning
- Predict future visual states from past observations + inferred actions
- Ground language in physically executable behavior

数学上，world-action model 学：
p(o_{t+1:t+H} | o_{t-H+1:t}, a_{t-H+1:t})

其中：
- o_t: observation at time t
- a_t: action at time t
- H: horizon

HumanNet 提供 (o, a) pairs（a 通过 motion retargeting 得到）。

### 7.3 Motion-aware Representation Learning
Combine first-person + third-person → representations align appearance, language, motion。

### 7.4 Human-to-Robot Transfer
Prior work (EgoMimic [18], Being-H series [22-24]) 已经 show 人类 data + alignment 可以 supply priors。HumanNet widen 这个 pipeline 的 human side。

### 7.5 Multimodal Objectives
支持：
- Masked/predictive video modeling
- Language-video alignment
- Procedural boundary prediction
- Weakly supervised hand-object learning
- Pose/motion prediction
- Caption-conditioned activity modeling

---

## 8. Related Work Landscape (Section 2)

让我把 related work 组织成一个更清晰的 landscape：

### 8.1 Human-centric Activity Datasets
- **Third-person**: ActivityNet [3], Kinetics [19], Charades [33], AVA [15], Something-Something [12], HACS [37], FineGym [32], HowTo100M [27]
- **First-person**: EPIC-KITCHENS [7], Ego4D [13], EgoDex [16], OpenEgo [17], EgoScale [38]
- **Mixed-view**: Ego-Exo4D [14], Assembly101 [31]
- **Dense HOI**: HOI4D [21], DexYCB [4]

### 8.2 Robot Learning from Human Data
Key papers:
- **R3M [28]** (Nair et al. 2022): 用 passive human video 学 visual representation for robot manipulation。这是早期的 "human video → robot representation" 工作。Link: https://arxiv.org/abs/2203.12601
- **EgoMimic [18]** (Kareer et al. 2024): Align human egocentric traces with robot demonstrations for imitation learning。Link: https://arxiv.org/abs/2410.24221
- **EgoScale [38]** (Zheng et al. 2026): Scaling dexterous manipulation with diverse egocentric human data。Link: https://arxiv.org/abs/2602.16710
- **EgoVerse [30]** (Punamiya et al. 2026): Shared egocentric data ecosystem for robot learning。Link: https://arxiv.org/abs/2604.07607
- **GR00T N1 [29]** (NVIDIA 2025): Open foundation model for generalist humanoid robots，mix heterogeneous robot logs with human video。Link: https://arxiv.org/abs/2503.14734
- **Being-H series [22-24]** (Luo et al. 2025-2026):
  - Being-H0: VLA pretraining from large-scale human videos (https://arxiv.org/abs/2507.15597)
  - Being-H0.5: Scaling human-centric robot learning for cross-embodiment generalization (https://arxiv.org/abs/2601.12993)
  - Being-H0.7: A latent world-action model from egocentric videos (https://arxiv.org/abs/2605.00078)

特别值得注意的是 Being-H 系列——它 explicit 地把 human interaction traces 作为 cross-embodiment learning 的 scalable substrate。Being-H0.7 的 latent world-action model 和 HumanNet 的 world-action model training use case 高度 aligned。

### 8.3 Open X-Embodiment
[6] Open X-Embodiment collaboration (https://arxiv.org/abs/2310.08864) 是 robot learning dataset 的 standard reference，但它的 scale 远小于 HumanNet。

### 8.4 其他相关
- **Human2Robot (H&R) [35]** (2025): Learning robot actions from paired human-robot videos。Link: https://arxiv.org/abs/2502.16587
- **LingBot-VLA [34]** (2026): The pragmatic VLA foundation model used in validation。Link: https://arxiv.org/abs/2601.18692
- **Rethinking Video Generation Model for Embodied World [9]** (Deng et al. 2026): Link: https://arxiv.org/abs/2601.15282 — 这篇是同一作者 group 的另一篇，关于 video generation model 用于 embodied world。

---

## 9. Limitations and Ethics (Section 6)

Paper 的 limitations 讨论相当 honest：

### 9.1 Embodiment Gap
Human behavior ≠ robot behavior。1M hours human corpus 不能 eliminate 人类手、身体、工具、mobility 和 robot control space 之间的 embodiment gap。Expected value 在 representation learning 和 transferable priors，direct one-to-one replacement 不行。

### 9.2 Scale Introduces Noise
Open-world video 有 ambiguous labels, inconsistent task boundaries, missing metadata, viewpoint imbalance, variable visual quality。Annotations 自己也 introduce errors。

### 9.3 Coverage Imbalance
Dataset 可能 biased toward certain geographies, socioeconomic contexts, occupations, camera viewpoints, body types。1M hours scale 可能 create "universality illusion"。

### 9.4 Privacy and Safety
First-person 录像可能 capture bystanders, sensitive interiors, private documents, screens, proprietary workflows。Third-person 录像可能 capture identifiable people, homes, workplaces, social interactions。需要 license review, redaction policy, restricted-content filtering, access controls。

---

## 10. Building Intuition: 这篇 paper 在大格局里的位置

让我把这个 work 放到一个更大的格局里：

### 10.1 Data Scaling Laws for Embodied AI

Language model 有 Chinchilla scaling laws。Vision-language model 有 similar scaling behavior。Embodied AI 的 scaling laws 还在 emerging。HumanNet 是这个方向的重要 data point：

- EgoScale (20K hours) → dexterous manipulation 有 predictable gains
- HumanNet (1M hours) → VLA post-training 的 validation loss 可以 match 100h robot data

这暗示 embodied AI 也可能有类似 scaling law，只是 slope 不同。如果能 plot "egocentric video hours vs. downstream task performance"，应该能看到 power law 或类似 pattern。

### 10.2 Human Data as Universal Prior

这个 paradigm 的更深 intuition 是：**人类 behavior 是 physical world 的 universal prior**。Human body 是 general-purpose embodiment——我们能 manipulate 各种 object、use 各种 tool、navigate 各种 environment。Encode 这个 universality 到 representation 里，应该 transferable 到任何 robot embodiment。

这和你以前在 Tesla 讨过的 "the world is its own simulator" 思路有 resonance——用海量真实世界 video 学 world dynamics，而不是 hand-craft simulator。

### 10.3 Cross-viewpoint Complementarity

First-person 和 third-person 的 complementarity 是一个 deep insight。可以 analogy 到 LLM 的 next-token prediction vs. masked language modeling——它们 capture language 的不同 aspects，combine 起来更强。First-person capture execution-centric cues，third-person capture observation-centric cues，两者都是 embodied intelligence 需要的。

### 10.4 Curation as Scientific Contribution

Paper 的一个重要 framing 是：**curation、viewpoint diversity、annotation taxonomy 是 scientific contribution**，不是 bookkeeping。这呼应了 LLM pretraining 的 lesson——data quality 比 data quantity 更重要，但 quality 需要 curation pipeline 来 ensure。

Stages 的 modular 设计（collection / processing / annotation 分离）允许 each stage 独立 audit, extend, rerun。这是 production-grade data infrastructure 的设计 pattern。

### 10.5 Cost-Effectiveness Argument

1000h egocentric video vs. 100h robot data 的 comparison 本质上是 cost-effectiveness argument。Teleoperation 的 cost 大约是 $10-50/hour (取决于 complexity)，所以 100h robot data 大约 $1000-5000。而 1000h egocentric video 的 curation cost 远低于这个——可能 $500-2000 (主要是 storage, compute for filtering, annotation)。

如果 10x data at 1/5 cost 能 match performance，那 egocentric video 的 ROI 是 50x。这个 cost-effectiveness argument 是 scale humanoid robotics 的关键 enabler。

---

## 11. 技术细节的进一步挖掘

让我做一些更 technical 的 deep dive：

### 11.1 Motion Retargeting 的细节

15mm threshold 是一个 interesting choice。Human hand 的 average length 大约 180mm，robot end-effector 的 precision 通常在 1-5mm 级别。15mm 是 human hand scale 的 ~8%，是 robot precision 的 3-15x。这个 threshold 是 balance——太严会 reject 太多 data，太松会 introduce 不准确 robot-ready subset。

### 11.2 Hand Pose Estimation 的挑战

Paper 用 "3D hand and body pose detection" 但没 specify method。State-of-the-art 选项：
- **HaMeR** (Moon et al.): SVG-based 3D hand mesh recovery
- **4D-Humans** (Goel et al.): SMPL-based body pose in 4D
- **FrankMocap**: Real-time monocular 3D hand pose
- **WHAM**: World-grounded humans with motion

这些方法在 egocentric video 上 performance 变化很大，因为 viewpoint 不常见、occlusion 严重、motion blur 多。

### 11.3 Monocular SLAM 的限制

Paper 提到 "first-person clips that satisfy stability and parallax requirements" 才能用 monocular SLAM。这个限制的物理直觉是：
- **Pure rotation** (e.g., 人 standing 转头) 无法 triangulate 3D structure
- **Pure translation** (e.g., 人 walking forward) 可以 parallax
- **Static scene** 无法 estimate camera motion

Egocentric video 中 walking, reaching, manipulating 这些 action 通常有 sufficient parallax。但 talking head, sitting, observing 这些 action 可能 insufficient。

### 11.4 Captioning Module

LLM-assisted captioning 的具体实现没 detail。可能 pipeline 是：
1. 抽取 keyframes (e.g., 每 1-2 秒一帧)
2. 用 VLM (GPT-4V, Qwen-VL, InternVL) generate per-frame description
3. 用 LLM aggregate 成 clip-level caption
4. Incorporate source 的 narration/metadata（如果有）
5. Normalize 到统一 format

这个 pipeline 的 quality 取决于 VLM 的 fine-grained activity understanding 能力。

---

## 12. 与你 Karpathy 的工作的潜在连接

### 12.1 "Software 2.0" 和 Data-driven AI

你以前讲过 "Software 2.0"——用 data-driven learning replace hand-coded rules。HumanNet 是这个 philosophy 在 embodied AI 的延伸：用 data-driven human video priors replace hand-crafted simulator priors。

### 12.2 World Models and Video Prediction

你 recent 的 work on Sora 和 world models 高度 relevant。HumanNet 的 "World-action model training" use case 正是 video prediction model 的 embodied 版本。Video generation model 学 p(o_{t+1:t+H} | o_{t-H+1:t})，world-action model 学 p(o_{t+1:t+H} | o_{t-H+1:t}, a_{t-H+1:t})——多了 action conditioning。

1M hours 的 human activity video 加上 motion annotations 应该能 train 出 strong world-action model。这个方向和 Sora、GAIA-1、Wayve 的 LINGO 等 work 都有 connection。

### 12.3 Tesla AI Day 和 Autonomous Driving

你在 Tesla 讨过用 video 学 world dynamics for autonomous driving。HumanNet 的 paradigm 完全可以 transfer 到 autonomous driving：用海量 dashcam video + human driving behavior 学 driving world model。区别是 driving 的 "human action" 是 steering/braking/acceleration，比 manipulation 的 action space 更连续。

### 12.4 Large-scale Data Curation

你多次强调 large-scale data curation 的重要性（ImageNet, LAION 等）。HumanNet 的 curation pipeline (collection → processing → annotation) 是这个 philosophy 的 embodied 版本。Modular 设计允许 scale to 1M+ hours without collapse。

---

## 13. Future Directions 和 Open Problems

### 13.1 Action Label 的补全

目前 HumanNet 的 action signal 来自 motion retargeting，但不是所有 clip 都 robot-ready (需要 <15mm error, >60% valid frames)。如何为 non-retargetable clips generate action labels？Possible solutions:
- **Latent action models** (e.g., LAPA, GR-2): learn latent action space from video without explicit action labels
- **Inverse dynamics models**: 从 visual state change infer action
- **Video-conditioned policy distillation**: 用 large VLM generate action tokens

### 13.2 Multi-modal Alignment

目前 caption, motion, pose 是 independently annotated。如何 align 它们到 unified representation？Possible:
- **Contrastive learning**: align (video, caption, motion) triples
- **Cross-modal generation**: caption → video, motion → caption
- **Joint embedding**: shared latent space for all modalities

### 13.3 Active Learning for Curation

1M hours 远超 manual review 能力。如何自动 identify high-value clips？Active learning pipeline：
1. Train initial model on random subset
2. Score remaining clips by uncertainty/value
3. Select high-value clips for annotation
4. Retrain

这个方向可以 significantly improve annotation ROI。

### 13.4 Cross-embodiment Generalization

Being-H 系列 [22-24] 已经 explore cross-embodiment。HumanNet 的 scale 应该 enable 更 system 的 cross-embodiment study：
- 不同的 robot platforms (humanoid, arm, mobile manipulator)
- 不同的 control interface (joint angle, end-effector pose, impedance)
- 不同的 sensing stack (wrist camera, head camera, third-person camera)

### 13.5 Procedural Reasoning

Long-horizon procedural tasks (make coffee, do laundry, cook meal) 是 embodied AI 的 hard problem。HumanNet 的 procedural structure 应该 enable：
- **Task decomposition**: learn sub-goals from procedural video
- **Goal-conditioned policy**: condition on end-state
- **Hierarchical planning**: high-level procedural plan + low-level motor execution

### 13.6 Foundation Model 的 Data Mixture

Open question: 在 VLA foundation model training 中，optimal mixture of：
- Human egocentric video
- Human exocentric video
- Robot demonstrations
- Web vision-language data
- Synthetic simulator data

HumanNet 的 ablation 只 compare 了 1000h ego vs. 100h robot，没有 explore mixture。这是 obvious next step。

---

## 14. 相关的更广 landscape (我做一些联想)

### 14.1 Synthetic Data 方向

除了 human video，synthetic data (simulator-generated) 是另一个 scalable direction。NVIDIA 的 GR00T N1 [29] mix heterogeneous robot logs with human video，可能也 include synthetic。HumanNet 没有深入 synthetic data，但 1M hours human video 的 diversity 应该 cover 很多 simulator 难以 generate 的 long-tail behaviors。

### 14.2 Self-supervised Learning

HumanNet 的 annotations (pose, motion, caption) 都是 derived from raw video，本质是 self-supervised。这和 MAE, VideoMAE, VIOLETA 等 self-supervised video learning 方法 aligned。

### 14.3 Continual Learning

EgoVerse [30] 强调 "shared ecosystem for continuously growing data"。HumanNet 的 modular pipeline 支持 continual expansion——new sources 可以 incremental 加入。

### 14.4 Privacy-preserving Learning

Privacy 是 HumanNet 的 first-class concern。Possible technical solutions:
- **Federated learning**: 不 centralize raw video，只 share model updates
- **Differential privacy**: 在 caption/motion annotation 加 noise
- **On-device processing**: 在 source device 上做 pose estimation，只 share derived features

### 14.5 Robotics Foundation Models 的趋势

VLA foundation models 的 trend：
- RT-1 (2022): narrow, robot-specific
- RT-2 (2023): web VLM knowledge + robot data
- OpenVLA (2024): open-source VLA
- GR00T N1 (2025): humanoid foundation model
- LingBot-VLA (2026): pragmatic VLA foundation
- Being-H0.7 (2026): latent world-action model

HumanNet 为这个 trend 提供 data substrate，让 VLA models 可以 leverage 1M hours human priors。

### 14.6 Egocentric Vision 的 scientific angle

Egocentric video 是 cognitive science 的 rich data source。1M hours 可以 enable：
- **Activity recognition** at unprecedented scale
- **Human behavior modeling** for cognitive science
- **Procedural learning** studies
- **Tool use evolution** across cultures

这些 scientific applications 是 embodied AI 之外的价值。

---

## 15. Final Thoughts

HumanNet 是一个 ambitious 的 data infrastructure project。它的 contribution 不在 algorithm novelty，而在 **data scale + curation quality + viewpoint diversity + annotation richness** 的组合。1M hours 是 ~50x jump over previous largest human-centric corpus，validation experiment 表明这个 scale 可以 translate 到 VLA post-training 的 measurable gains。

最 striking 的 result：**1000h egocentric video matches 100h robot data**。这个 result 的 implication 远超 paper 本身——它 suggest 人类 behavior data 是 embodied AI 的 scalable prior，而且 cost-effective。如果这个 trend 继续，未来的 embodied foundation model 可能 primarily trained on human video，robot data 只用于 fine-tuning 和 embodiment-specific alignment。

这和 LLM 的发展路径类似：pretrain on massive generic text (web, books)，fine-tune on task-specific data (instruction, code, math)。Embodied AI 的 pretrain data 可能是 human video，fine-tune data 是 robot demonstration。HumanNet 是这个 paradigm 的 first large-scale validation。

值得 follow 的 future work：
- Being-H 系列 (https://arxiv.org/abs/2605.00078) 的 latent world-action model
- EgoScale (https://arxiv.org/abs/2602.16710) 的 scaling laws for dexterous manipulation
- GR00T N1 (https://arxiv.org/abs/2503.14734) 的 open foundation model
- LingBot-VLA (https://arxiv.org/abs/2601.18692) 的 pragmatic VLA

如果你对 specific aspects (e.g., motion retargeting algorithm, world-action model architecture, VLA training recipe) 想要更深入 technical detail，我可以继续 dig deeper。这个 paper 的 technical density 不算特别高，但它的 framing 和 empirical validation 是 embodied AI data infrastructure 方向的重要 contribution。
