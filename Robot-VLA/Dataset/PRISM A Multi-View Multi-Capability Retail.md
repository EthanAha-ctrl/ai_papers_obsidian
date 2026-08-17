---
source_pdf: PRISM A Multi-View Multi-Capability Retail.pdf
paper_sha256: 62ec5185a95e0834adf4e3e48e55697f0c648a5bbf2ada1f8a63dc602411326c
processed_at: '2026-08-06T06:22:34-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# PRISM 论文人话版

## 一句话说清楚

DreamVu 这帮人发现，现在的 embodied AI 模型在超市这种真实场景里不行——不是看不清东西，是不懂空间、不懂物理、不懂人在干嘛。于是他们搞了个 27 万条样本的视频微调数据集，专门针对零售场景，给 VLM 补这三方面的课。

---

## 为什么现有数据不够用

想象一个机器人在超市里干活。它需要判断货架够不够高能够到、旁边那个人接下来会干嘛、自己这步活干完没有。这些能力现有的 VLM（比如 Cosmos-Reason2）做不好。

问题出在哪？**训练数据不对路**。

现有数据集要么是 Ego4D 那种日常生活录像，要么是 RoboVQA 那种办公室场景，要么是 SariBench 那种纯仿真。没有一个同时满足三个条件：真实超市环境 + 多视角拍摄 + 覆盖空间/物理/动作三类知识。

更关键的是，几乎所有 embodied 数据集都只用头戴相机（egocentric）。头戴相机的视野太窄了——只能拍到穿戴者关注的东西，拍不到整个超市的 aisle 结构、货架布局、其他人怎么走的。你训练出来的模型只能看到"眼前这一小块"，缺乏全局空间感知。

---

## PRISM 怎么解决

### 两个相机一起拍

PRISM 在 5 个不同的超市里拍了视频，用了两种相机：

**头戴 GoPro**（第一人称）：拍到手和商品的交互、局部决策点。比如你伸手拿一罐牛奶、推着购物车走过零食区。

**天花板 360° 全景相机**（第三人称）：拍整个场景的鸟瞰图。能看到 aisle 的结构、货架怎么排的、几个人同时在干什么。这个是 DreamVu 自家的 ALIA 相机。

这个组合很关键。Ego-Exo4D 那种数据集也有双视角，但它的 exo 是"跟着演员走的摄像头"——专门盯一个人拍。PRISM 的 exo 是固定的全景相机——看整个场景，不偏心任何人。这对"多人同时活动""社交导航"这种任务来说是必须的，你没法用跟拍镜头来做多人体轨迹分析。

### 三类知识打包训练

PRISM 不是随便扔一堆 QA 对进去，而是按照一个明确的 knowledge ontology 组织：

**Embodied Action（动作知识）**：人下一步会干嘛、任务做完没有、手在做什么、为什么做这个动作、多个视角下同一个活动怎么对应。

**Spatial（空间知识）**：东西离我多远、深度范围大不大、360° 全景下空间怎么布局、能不能够到那个东西。

**Temporal-Physical（时间物理知识）**：视频是正放还是倒放、东西消失在画面外还存不存在、物理上合不合理。

总共 20 多个 task，27 万条样本，其中 11.8M 帧、7.3 亿 token。

### 标注怎么搞

这里是 PRISM 最聪明的地方——**几乎不用人工标注**。

五种标注策略：

1. **Metadata 提取**：Gemini Robotics ER 1.5 直接看视频，吐出 goal hierarchy、sub-goal 序列、手部状态、轨迹标注这些结构化 JSON。

2. **LLM 生成 QA**：拿 Gemini 2.5 Flash，基于结构化文本 annotation 生成问答对和 chain-of-thought 推理。

3. **物理推理**：Gemini Robotics ER 1.5 直接处理视频，分析重力、动量、身体力学、时间因果。

4. **深度分析**：用 DepthCrafter 估深度，然后算每帧的 8 个统计量——平均深度、方差、近场/远场百分位、梯度幅值、左右不对称、上下不对称、前景/背景比。

5. **自监督变换**：IP-1（Arrow-of-Time）就是把视频倒过来，标签自动就有了；IP-2（Object Permanence）问"东西不在画面里了还存不存在"，答案永远是"存在"。零标注成本。

### 监督格式的配比

PRISM 刻意混合三种 answer 格式：
- 60% open-ended（直接回答）
- 30% chain-of-thought（`

---

# PRISM: 面向 Embodied VLM 的多视角零售视频 SFT 语料库

## 1. 核心问题与动机

当前 physical AI 模型（如 Cosmos-Reason2、GR00T N1、π₀）在 general-purpose visual understanding 上表现出色，但 deployment 到结构化真实环境（如 retail 商店）时出现严重的 perceptual gap。PRISM 论文的核心论点是：**失败不源于 visual recognition 不足，而源于对 space、physical dynamics、embodied action 三者的协同理解缺失**。

作者识别出三类既有数据集的根本局限：
- **Ego4D / Ego-Exo4D**：聚焦 skilled activities（cooking、sports），而非 deployment-oriented 环境；且以 short, isolated activity clips 为主，缺乏 long-horizon task structure。
- **RoboVQA**：仅 exocentric、仅 office domain、缺乏 reasoning chains。
- **SariBench**：纯仿真、纯 egocentric、benchmark 而非 training corpus。

PRISM 的设计直接针对这三个缺陷：real-world retail domain + ego/exo/360° 三视角 + ontology-structured SFT + CoT 监督。

---

## 2. 三维知识本体

### 2.1 知识维度形式化

PRISM 把"physical AI 需要的知识"形式化为三个互补子空间：

$$\mathcal{K}_{\text{physical}} = \mathcal{K}_{\text{spatial}} \oplus \mathcal{K}_{\text{temporal-physical}} \oplus \mathcal{K}_{\text{embodied-action}}$$

其中：
- $\mathcal{K}_{\text{spatial}}$：3D scene geometry $G \in \mathbb{R}^{H \times W \times D}$、layout $\mathcal{L}$、relative structure
- $\mathcal{K}_{\text{temporal-physical}}$：causality $\mathcal{C}$、motion $\mathcal{M}(t)$、ordering、physical constraints
- $\mathcal{K}_{\text{embodied-action}}$：action $\mathcal{A}$、goal hierarchy $\mathcal{G} = \{g_0 \to g_1 \to \dots \to g_n\}$、task progress $\tau(t)$

### 2.2 Capability Probes 到 Knowledge Dimension 的映射

| Knowledge Dimension | Capability Probes | Format |
|---|---|---|
| Embodied Action | ER-1~ER-9 | Und + CoT |
| Common Sense (mixed) | CS-U-1/2, CS-R-1~4 | Und + CoT |
| Spatial | SP-1, SP-2 | Und + CoT+MCQ |
| Temporal-Physical | IP-1, IP-1-CoT, IP-2 | Und + CoT |
| Cross-cutting | MCQ Overlay, MCQ Standalone | MCQ |

**关键观察**：Cosmos-Reason1 的训练 taxonomy 中仅有 2/23 个 PRISM capability 被直接覆盖，14 个为全新任务（6 个完全新，8 个是已有类别的新形式）。这表明 PRISM 不是在通用 pretraining 上做"加法"，而是填补 deployment gap。

参考: [Cosmos-Reason1](https://arxiv.org/abs/2501.18626) | [Cosmos-Reason2](https://arxiv.org/abs/2505.13477)

---

## 3. 数据采集架构

### 3.1 双相机系统

PRISM 在 5 个 retail store 中采集，覆盖不同 store layout、lighting、aisle configuration。

**Ego camera**：
- 设备：GoPro head-mounted
- 编码：480p, 4 fps, H.264
- 视角：first-person，包含 hand-object interaction、local decision points
- 活动覆盖：entering、navigating、browsing、approaching、selecting、picking、placing、carrying、cart interaction、checkout

**Exo camera**：
- 设备：DreamVu ALIA 360° omnidirectional camera
- 编码：4 fps, H.264
- 视角：scene-level，包含 aisle structure、shelf layout、navigation corridors、multi-actor trajectories
- 关键差异：Ego-Exo4D 中的 exo 是 **actor-following**（tracking primary actor），PRISM 中的 exo 是 **scene-observing**（不偏袒任何 actor），这对 multi-actor reasoning 至关重要

参考: [DreamVu ALIA](https://dreamvu.ai/#technology)

### 3.2 数据规模

| Metric | Value |
|---|---|
| Total samples | 270K |
| Tasks (incl. 2 eval-only) | 20+ |
| Video frames (4 fps) | ~11.8M |
| Total tokens (Qwen3VL) | ~730M (703M visual + 27M text) |
| Open-ended / CoT / MCQ | 58.6% / 30.4% / 9.2% |
| Domain: ER / CS / SP / IP / MCQ | 34.7 / 18.3 / 4.4 / 33.5 / 9.2% |

**Tokenization 细节**：使用 Qwen3VL 的 visual tokenizer，703M visual tokens 占比 96.3%。这反映了 video SFT 的 visual density 远高于 text——一个典型样本包含约 2700 visual tokens 和 ~100 text tokens。

---

## 4. Capability Probes 详解

### 4.1 Embodied Reasoning (ER) — 9 tasks, 93,757 samples

#### ER-1: Next Subtask Prediction (25K, Ego, Und+MCQ)
- **输入**：当前 clip + 结构化 context (goal hierarchy, sub-goal, scene description, hand states)
- **输出**：predicted next subtask in shopping workflow
- **标签来源**：episode metadata 中的 temporal sub-goal ordering
- **直觉**：这训练 model 学习 long-horizon planning，超越单帧 action recognition。例如："Sub-goal = Navigate Produce Section" → Next = "Aisle Navigation (Dry Goods)"

#### ER-2: Task Completion Verification (3K, Ego, Und)
- 二分类，class split 59.3%/40.7%
- 标签从 sub-goal transition boundary 自动推断
- 6 question templates × 8 answer templates → 模板池增强 lexical diversity

#### ER-3: Goal-Conditioned Action Reasoning (8K, Ego, CoT)
- 这是 PRISM 的核心 reasoning 任务之一
- 输入：clip + goal hierarchy + environment + hand states
- 输出：`` 提供中间 reasoning states，相当于在 token level 提供 dense supervision。形式化：

$$\mathcal{L}_{\text{template}} = -\log P(y \mid x; \theta)$$
$$\mathcal{L}_{\text{CoT}} = -\sum_{t=1}^{T} \log P(r_t \mid x, r_{<t}; \theta) - \log P(y \mid x, r; \theta)$$

其中 $r = (r_1, r_2, \dots, r_T)$ 是 reasoning tokens。**CoT 的有效 supervision token 数量 $T+1$ vs template 的 1**，信息密度高一个量级。

### 12.3 为什么 IP-1 Exo 比 IP-1 Ego 难

AoT 的核心 cue 是 motion physics：
- Ego 视角：相机与人 motion 同步，整个 frame 在动，physics cues 丰富（步态、手部 motion）
- Exo 视角：相机静止，人物在 frame 内移动，physics cues 仅限于人物 trajectory——幅度小、变化少

数据印证：Ego AoT gain +26.2% vs Exo +7.2%。

### 12.4 SP-1 (Relative Depth) 为何 gain 有限

SP-1 Error Ratio 0.84（接近 1，几乎无 improvement）。原因推测：
- 任务要求 metric depth discrimination（"0.05 vs 0.78"），需要 continuous regression
- 但 Cosmos-Reason2-2B pretraining 未见过 metric depth input
- SFT 仅提供 categorical supervision（"wide vs narrow"），不足以学到 continuous depth mapping

**Fix 路径**：未来工作提到引入 metric depth 作为 input modality。

---

## 13. 相关工作与生态定位

### 13.1 VLM for Embodied AI 演进

| 模型 | 关键贡献 |
|---|---|
| RT-2 | VLM 直接生成 robot actions |
| PaLM-E | Large-scale multimodal embodied reasoner |
| Cosmos-Reason1/2 | Physical AI VLM + CoT reasoning supervision |
| GR00T N1 | VLM backbone for humanoid control |
| π₀ | VLA flow model for general robot control |

PRISM 定位为这些 backbone 的 **domain-specific SFT substrate**，而非替代。

参考: [RT-2](https://robotics-transformer2.github.io/) | [PaLM-E](https://palm-e.github.io/) | [GR00T N1](https://arxiv.org/abs/2503.14734) | [π₀](https://arxiv.org/abs/2410.24164)

### 13.2 Instruction-Tuning 数据集演进

| 数据集 | 规模 | 特点 |
|---|---|---|
| LLaVA | GPT-4 generated visual conversations | Pioneer |
| Video-ChatGPT | Video temporal reasoning | Video-specific |
| LLaVA-OneVision | Multi-stage task-specific mixtures | Data mixing matters |
| InternVL | Scaled to millions | Scale |
| VideoChat2 | Temporal reasoning instructions | Chat-centric |
| **PRISM** | 270K retail-specific, ontology-structured | Domain + ontology |

### 13.3 Self-Supervised Pretext Tasks Repurpose

PRISM 的 IP-1 和 IP-2 是经典 self-supervised pretext tasks（Arrow-of-Time Wei et al. 2019, Jigsaw Doersch et al. 2015）的创新 repurpose——从 representation learning 转为 instruction-tuning data。这提供了一个范式：**自监督 pretext signal 可以直接作为 SFT supervision，无需 manual annotation**。

参考: [Arrow-of-Time](https://arxiv.org/abs/1903.10560) | [Jigsaw Puzzles](https://arxiv.org/abs/1503.05760)

---

## 14. 工程与实现要点

### 14.1 First-Token Diversity Audit

PRISM 强制 no task has >50% of answers sharing the same opening word。这是为了避免训练时模型学到 trivial first-token bias（如总是输出 "The"）。

### 14.2 Sqrt-Proportional Sample Allocation

Sample allocation follows sqrt-proportional scaling，no single task >9.5% of total mix。形式化：

$$N_i \propto \sqrt{N_i^{\text{max}}}$$

这种平衡防止 dominant task (如 IP-1 CoT 49K) 压制 minority task (如 CS-R-4 2K)。

### 14.3 Cosmos-Reason2 Format Compatibility

所有 sample 遵循 Cosmos-Reason2 的 three-message conversation format：
- System message
- User message (interleaved video/image + task text)
- Assistant message (response)

三种 response format：
- Understanding: direct open-ended answer
- Reasoning: `
