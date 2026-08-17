---
source_pdf: SPATIALAB.pdf
paper_sha256: f5eac1240257e0bef27f2b021098876b832afc123d20ca53b059858e80d315b7
processed_at: '2026-08-12T09:24:00-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：这篇论文在说什么

## 一、一句话版本

这篇论文搞了 1400 道看图答题的题，专门考 VLM 的空间推理能力，结果发现**所有 SOTA 模型都被人类按在地上摩擦**——MCQ 上人类 87%、最强模型 55%；open-ended 上人类 65%、最强模型 41%。更扎心的是，CoT、SFT、multi-agent 这些平时大家觉得有用的 trick，在 spatial reasoning 上基本没用甚至越用越差。

---

## 二、为什么要搞这个 benchmark

之前的 spatial reasoning benchmark 有几个通病：

1. **太 fake**：CLEVR、GQA 这种是合成的，画面干净、物体就几个、关系简单。模型在这种上面刷到 90%+ 不代表真的懂空间，只代表它学会了 synthetic dataset 的 shortcut。
2. **太窄**：大多数 benchmark 就测 2-3 个 category，比如"left of / right of"这种 binary relation，根本没覆盖 occlusion、orientation、navigation、3D geometry 这些真实世界里天天遇到的事。
3. **只有 MCQ**：选择题可以靠排除法蒙对，不测 open-ended 就不知道模型是真的会还是真会蒙。

SPATIALAB 的设计就是反着来：真实场景图片（平均每张 21 个物体、3 层 depth、有阴影有反光有透明物体），6 大类 × 5 小类 = 30 种 task type，同时测 MCQ 和 open-ended。

---

## 三、核心发现，用人话翻译

### 发现 1：scaling 不是万能药

| 模型 | MCQ |
|---|---|
| InternVL3.5-1B | 31.6% |
| InternVL3.5-72B | 54.9% |
| Human | 87.6% |

参数涨 72 倍，分数涨 23 个点。但 72B 到 human 还差 33 个点。更离谱的是 Llama-3.2-11B 只有 30.5%，比 InternVL3.5-1B 还差。**这说明问题不在"参数够不够大"，在于"训练时有没有真的教空间推理"**。Llama 系列是先做 language 再接 vision，InternVL 是 native multimodal，这个差异在 spatial reasoning 上被放大了。

类比一下：你让一个只读过课本但从没动手做实验的物理系学生，和一个天天在实验室泡着的学生，去组装一个机械装置。课本学生可能公式背得更熟，但动手就拉胯。Llama 就是那个课本学生。

### 发现 2：模型的 spatial 能力是碎片化的

同一个 GPT-4o-mini：
- Stacking Orientation：85.7%（基本满分）
- Relative Size Comparison：42.0%（接近 random）
- Perspective Distortion：50.0%

**模型不是"整体懂空间"或"整体不懂"，是某些点很强、某些点极弱，像打地鼠一样按下一个冒出另一个**。这说明模型学到的不是统一的 spatial representation，是一堆零散的 visual cue → answer 的 shortcut。

Llama-3.2-11B 在 Object Rotation 上只有 2.0%——比瞎蒙的 25% 还低 23 个点。这不是"不会"，是"学反了"，学到的 prior 和真实 rotation 是 anti-correlated。这是训练数据 sampling bias 的直接证据。

### 发现 3：CoT 在 spatial reasoning 上有害

这是最反直觉的发现。用 CoT prompting 之后：
- InternVL3-78B：62% → 59.3%（降了 2.67 个点）
- Gemini-2.5-Flash：54.7% → 36.7%（暴跌 18 个点）

唯一涨的是 Orientation（+4%）。

为什么？**因为 CoT 的逻辑是"把大问题拆成小步骤，每步推理一下"。这个 trick 在 math/code 上 work，是因为每步的 prior 是 axiomatic 的（1+1=2 是定义）。但在 spatial reasoning 上，每步的 prior 是 perceptual 的——如果模型本来对 perspective geometry 的 prior 就是错的，那 CoT 只是把错误信号放大了 8 倍**。

举个具体例子：问模型"远处的自行车和近处的汽车哪个实际更大"。模型 CoT 推理：
1. "远处东西看起来小" ✓
2. "自行车在远处，看起来比汽车小" ✓
3. "但远处视觉小不代表实际小" ✓
4. "所以实际大小要 reverse 视觉大小" ← **这一步错了**

模型没有真正的 perspective geometry model，它把"远处视觉小→实际大"当成 universal rule，遇到 toy car 这种 counterexample 就 flip 错。**CoT 不是 error correction，是 error amplification**——当且仅当每步 prior 正确时才有用。

### 发现 4：SFT 造成 "algorithmic straightjacket"

用 SPATIALAB 40% 数据 fine-tune Qwen2.5-VL-3B：
- MCQ：25.6% → 36.6%（+11 个点）
- Open-ended：34.4% → 35.5%（+1 个点）
- Size & Scale 的 open-ended：18.4% → 14.5%（**反而降了 4 个点**）

更细的 dynamics：open-ended 是 U-shape——Epoch 1 直接从 34% 跌到 19%（catastrophic forgetting），然后慢慢恢复到 35%。**SFT 在 early epoch 破坏了 base model 的 language fluency，mid-late epoch 重新学一个新的 spatial-language alignment，但永远回不到原来的 fluency**。

这说明 SFT 教会模型的是"看到 size 问题就选某个选项"这种 MCQ shortcut，而不是真的理解 size relation。external benchmark 上 OmniSpatial +7%、Mind the Gap +11.7% 看起来像 transfer learning 成功，但仔细一想这些 external benchmark 也都是 MCQ 格式，所以其实是 MCQ shortcut 的 cross-format transfer，不是 spatial reasoning 的真实进步。

### 发现 5：specialized spatial model 反而最差

SpaceOm、SpaceThinker、SpaceQwen 这三个专门为 spatial reasoning fine-tune 的模型，open-ended 只有 10-13%，比 general VLM 还差。这非常讽刺——**专门训练反而损坏了通用空间能力**。

原因：task-specific SFT 让模型 overfit 到某个 narrow task distribution，失去 general spatial representation。SpaceQwen 在 Size & Scale 上 +15% 但 Spatial Containment 上 −10%，典型的 catastrophic forgetting + narrow gain。

### 发现 6：Multi-agent 在某些 task 上有效，某些 task 上有害

SPATIOXOLVER 这个 multi-agent system：
- Orientation：open-ended +36%（大涨）
- Depth & Occlusion：open-ended −24%（大跌）
- Spatial Navigation：open-ended −12%（大跌）

规律：**当 task 的 prior 是 binary/离散的（orientation = 左/右/上/下），agent 的 decomposition 有效；当 task 的 prior 是 continuous metric 的（depth = 多少米、occlusion = 哪个在前），agent 只是把错误信号 propagate 8 步而不是 1 步**。

agent 的增益正比于 base model 在该 task 上的 prior strength。Orientation prior 强 → agent work；Depth prior 弱 → agent fail。**Agent 不是 free lunch，agent 是 prior amplifier**。

---

## 四、为什么会这样：5 类 root cause

作者做了 qualitative error analysis，所有失败归到 5 类：

### 1. Spatial mislocalization（指错对象）
问"左数第二个红 cube"，模型答到最亮的 cube 或中心 cube 上。**因为 attention/conv pooling 把相邻物体的 representation 混了，模型没有 object identity 的稳定 pointer**。

类比：你让一个近视眼不戴眼镜在人群中找"左数第二个戴帽子的人"，他可能就找了最显眼的那个戴帽子的人。

### 2. Perspective & scale mistakes（不会读 perspective cue）
模型默认"car 总比人大"，遇到 toy car 就答错。**模型把 web 上的 category frequency prior 当 ground truth，不读 pixel-level metric cue**。

类比：你问一个只看过 SUV 广告的人"远处的自行车和近处的 toy car 哪个大"，他答 car 大，因为他脑子里 car = 大。

### 3. Occlusion & ordering failures（看不见 thin structure）
面对 wire、railing、tree branch 这种细物体，模型直接 omit。**因为 large receptive field 在 pooling 时把 thin edge 抹掉了，T-junction（深度关键 cue）丢失**。

### 4. Attribute confusion（把视觉属性和功能属性混了）
看到 crosswalk 的 paint stripe 就觉得"safe"，忽略 traffic light。**因为 LM objective 奖励 plausible narrative，不奖励 visual grounding**。

### 5. Open-ended rationalization without verification
问"为什么这个物体倾斜"，答"因为表面滑"——但 slipperiness 从图里根本看不出来。**模型生成 plausible text 但没有 verification 把 text tie back to image evidence**。

这 5 类 error 在所有 architecture 上 recurring，说明不是某个模型的 bug，是整个 VLM training paradigm 的 structural weakness：**image-text next-token prediction 学到的是 "image → plausible caption"，不是 "image → grounded spatial predicate"**。

---

## 五、这对未来意味着什么

### 当前路径已经接近 ceiling

scaling + CoT + SFT + agent + specialized model，这 5 个大家正在用的 trick 在 spatial reasoning 上基本都 hit ceiling 了。继续在这条路上走，能挤 1-2 个点但不会有质变。

### 需要的是 architectural + data innovation

论文最后给的 suggestion：
- **Geometry-aware supervision**：pretraining 时显式加 pairwise spatial predicate 的 loss
- **Multi-scale feature retention**：用 edge-aware pooling 保留 thin structure
- **Slot-based object representation**：maintain object identity across reasoning step
- **Counterfactual augmentation**：对每张图做 perturbation 生成 counterfactual pair
- **Embodied pretraining**：用 simulator 生成 spatial trajectory + predicate supervision
- **Verification module**：在 generation 后加 verifier check claim vs image

这些方向都需要 grounded supervision，不是更多 RLHF 能解决的。

### Meta 启示

你之前讲过 "we're overfitting to the benchmark, not learning to reason"。SPATIALAB 给这句话提供了 quantified 证据：CoT 失败、SFT 反向、specialized model 退化、agent 部分失效——这些现象合起来证明 **当前 LLM-style training paradigm 在 spatial reasoning 上已经接近 ceiling**。

要继续进步，需要的是：把 spatial predicate 显式注入 training signal，让模型学到 $r(o_i, o_j)$ 这种 structured relation，而不是继续优化 "image → plausible caption" 的 next-token prediction。

---

## 六、一句话 intuition

**当前 VLM 的 spatial reasoning 失败，本质是 perceptual grounding 的失败，不是 reasoning chain 的失败。给一个 perceptually grounded 的 prior 加 CoT，CoT 会 work；给一个 perceptually wrong 的 prior 加 CoT，CoT 只是把错误放大。**

这就像：一个真正懂几何的学生做 CoT 会越做越对；一个只会背公式的学生做 CoT 会越做越离谱——因为每一步他都在用错的 prior 推下一步。

---

参考：
- SPATIALAB：https://spatialab-reasoning.github.io/
- VSI-Bench（video-side 互补）：https://arxiv.org/abs/2412.14171
- SpatialVLM（geometry grounding 起点）：https://arxiv.org/abs/2401.12168
- Habitat（embodied simulator）：https://arxiv.org/abs/1907.01418

---

# SPATIALAB：让 VLM 在野外做空间推理的 diagnostic benchmark

## 一、为什么这篇论文值得 Karpathy 你细读

你过去几年反复强调的一个观点：当前 LLM/VLM 在 systematic reasoning、spatial understanding、embodied grounding 上的 progress 被高估了，很多 benchmark 已经 saturated 或者被 shortcut 攻破了。SPATIALAB 这篇 paper 几乎是为这个论点量身定做的证据集。它做的事情很简单——把空间推理拆成 6 个 category、30 个 subcategory，用真实场景图片（非 CLEVR-style synthetic），同时给 MCQ 和 open-ended 两种格式，结果展示了一个让你会感到 "果然如此" 的现象：

- **最强 open-source VLM**（InternVL3.5-72B）只有 **54.93%** MCQ accuracy，**人类 87.57%**，gap 33 个百分点；
- **最强 closed-source**（GPT-5-mini）只有 **40.93%** open-ended，**人类 64.93%**，gap 24 个百分点；
- **specialized spatial models**（SpaceOm、SpaceThinker、SpaceQwen）整体表现最差（10–13% open-ended），证明 task-specific SFT 反而损坏了通用空间能力；
- **CoT prompting** 平均降低 2.67%（InternVL3-78B）和 18.0%（Gemini-2.5-Flash），唯一例外是 Orientation；
- **SFT** 在 MCQ 上 +10.97%，但 open-ended 只有 +1.07%，某些 category（Size & Scale）甚至 **−3.95%**——典型的 "algorithmic straightjacket"。

这五个数字单独看不新鲜，合起来却揭示了 VLM 当前 spatial reasoning 的 ceiling 是被 **perceptual grounding 的缺失** 卡住的，scaling + CoT + SFT + agentic decomposition 都只是在同一个错误 prior 上加 buff。

参考链接：
- 项目主页：https://spatialab-reasoning.github.io/
- 论文 PDF（ICLR-style）：项目主页可下载
- 相关 baseline：SpatialVLM (https://arxiv.org/abs/2401.12168)、RoboSpatial (https://arxiv.org/abs/2411.16537)、VSI-Bench (https://arxiv.org/abs/2412.14171)、OmniSpatial (https://arxiv.org/abs/2506.03135)、Mind the Gap (https://arxiv.org/abs/2503.19707)

---

## 二、Taxonomy 设计：为什么是 6 × 5 = 30

论文最有信息密度的部分是 Section 2 + Appendix B。作者把空间推理形式化为：

$$
f : (I, Q) \mapsto R, \quad R = \{ r(o_i, o_j) \mid o_i, o_j \in O \}
$$

变量解释：
- $I$：image 或 image sequence（视觉输入）
- $Q$：linguistic query（语言查询）
- $O = \{o_1, \ldots, o_n\}$：scene 中所有 object 的集合，每个 $o_i$ 带 position、orientation、size 等 attribute
- $R$：structured spatial relation 的集合，例如 $\text{left-of}(o_i, o_j)$、$\text{occludes}(o_i, o_j)$、$\text{supports}(o_i, o_j)$
- $f$：VLM 要学的 mapping，从 pixel + text 到 spatial predicate

这个形式化很关键——它把 spatial reasoning 视为**谓词逻辑 + 几何约束**的问题，而非 caption + VQA 的语言生成问题。当前 VLM 的训练目标（next token prediction on web-scale image-text pairs）并不显式监督 $r(o_i, o_j)$，所以学到的只是"哪种文字描述最常和哪种视觉 pattern 共现"。

**6 大类 × 5 子类的具体含义**（这是 intuition building 的关键）：

| Main Category | 直觉 | 5 个 subcategory | 为什么难 |
|---|---|---|---|
| **Relative Positioning** | "A 在 B 的左边/之间/角落" | Directional Relations, Proximity Gradients, Alignment Patterns, Betweenness Relationships, Corner/Angle Positioning | 需要 stable object index 跨 reasoning step 保持 |
| **Depth & Occlusion** | "A 挡住 B / 透过玻璃能看到 C" | Layering Order, Partial Occlusion, Complete Occlusion Inference, Transparency Effects, Reflective Surfaces | 需要 multi-scale feature 保留 thin structures 和 T-junctions |
| **Orientation** | "门朝南 / 椅子转了 30° / 这是左手剪刀" | Cardinal Direction, Object Rotation, Facing Direction, Stacking Orientation, Tool Handedness | 需要 reference frame（viewer-centric vs object-centric vs world-centric）切换 |
| **Size & Scale** | "近处的自行车比远处的汽车大——但实际汽车更大" | Distance-Size Correlation, Perspective Distortion, Relative Size Comparison, Scale Consistency, Shadow-Size Projection | 需要 perspective geometry + 光源几何推理 |
| **Spatial Navigation** | "从门走到窗的路径 / 坐沙发上能看到什么" | Pathway Existence, Obstacle Avoidance, Viewpoint Visibility, Spatial Sequence, Accessibility Constraints | 需要 multi-step relational chaining（Pearson r=0.99 with overall gap） |
| **3D Geometry** | "球放在斜面上会滚到哪 / 加一本书堆会不会塌" | Volume Comparison, Stability Prediction, Shape Projection, Spatial Containment, Gravity Effects | 需要 physics-aware reasoning，模型普遍 < 40% |

这个 taxonomy 的设计灵感来自 cognitive psychology 的 spatial cognition 文献（Baddeley 1998 working memory；Previc 1998 neuropsychology of 3D space；Trope & Liberman 2010 construal-level theory）。注意 SPATIALAB 强调的是"真实场景"（in-the-wild）——平均每张图 21.48 个 object，其中 11.88 个 partially visible，分布在 3.23 个 depth layer，spatial reference chain 平均 2.07 跳。这是和 CLEVR (https://arxiv.org/abs/1612.06890)、GQA (https://arxiv.org/abs/1902.08572)、What's Up (https://aclanthology.org/2023.emnlp-main.571/) 这种 synthetic / template-driven benchmark 的本质区别。

---

## 三、Image 复杂度的 6 个 meta-dimension

论文在 Section 3.1 提了一个很聪明的 image profiling 方法。每张图沿 6 个轴打分：

1. **Lighting condition**：high contrast / low contrast / shadows / reflective
2. **Texture complexity**：uniform / patterned / complex
3. **Edge complexity**：sharp vs smooth
4. **Dominant spatial relation**：stacked / scattered / aligned
5. **Material type**：transparent / translucent / opaque / reflective
6. **Gravity constraints**：normal / floating / unconstrained

这个设计对 build intuition 极有价值：它告诉你 VLM 的失败不是"任务难"，是"任务的 **视觉统计分布** 难"。Web 上大量图片是 frontal、indoor、均匀光照、无透明物体——而真实世界的 embodied agent 会遇到 low-contrast 阴影、透明玻璃、反光金属、悬浮物体（AR/VR），这些恰恰是当前 VLM 训练分布的 tail。这是为什么 Scaling Laws 在这里失效——你 scaling 的是 head distribution，benchmark 测的是 tail。

---

## 四、实验结果的核心 finding

### 4.1 主表（Table 2 / Table 3）

我把最值得讨论的几行单独拎出来：

**MCQ Top-5（Table 2）**：
- InternVL3.5-72B：54.93%（open-source SOTA）
- GPT-5-mini：54.29%
- o4-mini-medium：53.21%（reasoning）
- Gemini-2.5-Flash-Thinking：52.93%
- Gemini-2.0-Flash：52.50%

**Open-ended Top-5（Table 3）**：
- GPT-5-mini：40.93%
- o4-mini-medium：37.86%
- Gemini-2.5-Pro：33.61%
- Gemini-2.5-Flash-Thinking：32.77%
- Gemini-2.0-Flash：27.43%

注意一个反直觉的现象：**InternVL3.5-72B 在 MCQ 是 SOTA，但 open-ended 只有 23.36%**——MCQ→open gap 高达 31.57%，是所有模型里最大的。这说明它的 MCQ 高分很可能是 distractor elimination 的 surface cue 在起作用，而不是真的懂空间。

### 4.2 Scaling law 在这里不 work 的证据

| 模型规模 | MCQ Acc |
|---|---|
| InternVL3.5-1B | 31.64% |
| InternVL3.5-2B | 33.71% |
| InternVL3.5-4B | 43.29% |
| InternVL3.5-72B | 54.93% |
| **Human** | **87.57%** |

scale 从 1B → 72B 涨了 23 个点，但 72B → human 还有 33 个点。更 striking 的是 Llama-3.2-11B 只有 30.50%，比 InternVL3.5-1B 还差。Llama-3.2-90B 50.36% 也只比 InternVL3.5-4B 强一点。这说明**架构 + 训练数据 + objective 比 scale 重要得多**——Llama 系列在 image-text pair 上没有像 InternVL 那样做 native multimodal pretraining，所以即使参数大 10 倍也补不回来。

参考：InternVL3 (https://arxiv.org/abs/2504.10479)、Qwen2.5-VL (https://arxiv.org/abs/2502.13923)、Llama 3 (https://arxiv.org/abs/2407.21783)、GLM-4.5V (https://arxiv.org/abs/2507.01006)。

### 4.3 Sub-category 的 fragmentation

最 striking 的数据在 Appendix E（Table 6-10）。我用 GPT-4o-mini 举例：

- Stacking Orientation：**85.71%**（基本饱和）
- Relative Size Comparison：**42.0%**
- Perspective Distortion：**50.0%**

同一个模型在 orientation 上接近 human，在 size & scale 上接近 chance。这种 within-model spread 超过 50 个百分点的现象在所有模型上都出现。**VLM 的空间能力不是"全好或全差"，是碎片化的局部 peak + 全局低谷**。

更戏剧性的：Llama-3.2-11B 在 Object Rotation 上只有 **2.0%**（比 random 25% 还差 23 个点），Stacking Orientation 上 28.57%。这说明 Llama-Vision 的 attention pattern 在 rotation task 上是系统性反向的——它学到的 rotation prior 是 anti-correlated 的。这是数据 sampling bias 的直接证据。

### 4.4 Specialized spatial models 反而最差

| 模型 | MCQ | Open-ended |
|---|---|---|
| SpaceOm | 41.36% | 12.93% |
| SpaceThinker-Qwen2.5VL-3B | 40.64% | 13.36% |
| SpaceQwen2.5-VL-3B-Instruct | 40.14% | 10.36% |
| **Qwen2.5-VL-3B-Instruct (base)** | 33.71% | 12.93% |

specialized fine-tuning 在 MCQ 上比 base model 高 7 个点，但 open-ended 持平甚至下降。这是 **SFT overfitting to MCQ format** 的典型 signature——模型学到了"在 4 个选项里选最像训练分布的"，但没有学到 spatial relation 本身。SpaceQwen 在 Size & Scale 上 +15%（49.02% vs 33.33% base），但在 Spatial Containment 上只有 24.0%（base 34.0%），属于典型的 catastrophic forgetting + narrow gain。

---

## 五、为什么 CoT 在 spatial reasoning 上失效（核心 intuition）

这是论文最反直觉也最重要的发现。Table 19 显示：

| Category | InternVL3-78B w/o CoT | w/ CoT | Gain |
|---|---|---|---|
| 3D Geometry | 52.0 | 52.0 | 0.0 |
| Depth & Occlusion | 60.0 | 60.0 | 0.0 |
| Orientation | 60.0 | 64.0 | **+4.0** |
| Relative Positioning | 80.0 | 72.0 | **−8.0** |
| Size & Scale | 56.0 | 44.0 | **−12.0** |
| Spatial Navigation | 64.0 | 64.0 | 0.0 |
| **Overall** | **62.0** | **59.33** | **−2.67** |

Gemini-2.5-Flash 更夸张，Overall 从 54.67% 跌到 36.67%，跌了 18 个点。

为什么？作者的假说是：**CoT 在 language reasoning 上 work，是因为 reasoning step 之间是 logically entailed 的；但在 spatial reasoning 上，每个 reasoning step 都依赖一个 perceptual prior，如果这个 prior 是错的，多步推理只是在错的 prior 上做 multi-step 误差放大**。

举一个具体场景：模型被问"远处的自行车和近处的汽车哪个实际更大"。它做 CoT：
1. "远处的物体看起来小" ✓（prior 对）
2. "自行车在远处，看起来比汽车小" ✓
3. "但远处的东西视觉上小不代表实际小" ✓
4. "所以实际大小要 reverse 视觉大小" ← 这一步错了，因为模型没有 perspective geometry 的精确 model，它把"远处视觉小→实际大" 当成 universal rule，遇到 toy car 就 flip 错

这就是 paper 里说的 "CoT amplifies flawed priors rather than corrects them"。

这个发现对你 Karpathy 关于 System 2 thinking 的论述是个补充：CoT 不是免费的午餐，它的有效性依赖于每一步的 prior 是否正确。在 math/code 上 prior 是 axiomatic 的，所以 CoT work；在 spatial reasoning 上 prior 是 perceptual 的、可能 anti-correlated with truth，所以 CoT 反而有害。

### 5.1 CoT + Self-Reflection 的结果（Table 20）

加 self-reflection 之后：
- InternVL3-78B：MCQ −2.66%，open-ended −8.67%
- Gemini-2.5-Flash：MCQ +20.66%（Geometry +32%，Depth +24%），open-ended +0.67%

self-reflection 在 MCQ 上帮助 Gemini，因为 Gemini 的 perceptual prior 比 InternVL 强，reflection 能 prune wrong options；但在 open-ended 上几乎没用，因为 reflection 是 language-only 的 polish，没有引入新的 visual evidence。这验证了作者的论断：**没有 perceptual anchor 的 reflection 是 linguistic filter，不是 error correction**。

---

## 六、SFT 的 "Algorithmic Straightjacket"

Section 5.4 / Appendix H.4 用 Qwen2.5-VL-3B-Instruct 做 LoRA fine-tuning（rank=16, alpha=16, lr=2e-4, 4 epochs），用 40% 数据训练，60% 测试。结果：

| Category | MCQ Before | MCQ After | Gain | Open Before | Open After | Gain |
|---|---|---|---|---|---|---|
| 3D Geometry | 30.07 | 37.76 | +7.69 | 20.28 | 23.78 | +3.50 |
| Depth & Occlusion | 23.87 | 32.90 | +9.03 | 43.23 | 45.81 | +2.58 |
| Orientation | 23.33 | 36.67 | +13.33 | 39.67 | 42.15 | +2.48 |
| Relative Positioning | 26.19 | 27.78 | +1.59 | 51.18 | 50.39 | **−0.79** |
| Size & Scale | 27.63 | 44.74 | **+17.11** | 18.42 | 14.47 | **−3.95** |
| Spatial Navigation | 22.38 | 38.46 | +16.08 | 36.62 | 39.44 | +2.82 |
| **Total** | 25.63 | 36.59 | **+10.97** | 34.40 | 35.48 | **+1.07** |

Size & Scale 是最 dramatic 的：MCQ +17%，open-ended −4%。这是经典的 **catastrophic forgetting + format overfitting**——模型学到了"看到 size 问题就选某个选项"的 shortcut，但失去了生成自然语言描述 size 关系的能力。

更细的 SFT dynamics（Appendix H.4.3，Table 23，5 个 seed）显示：

- **MCQ**：monotonic 上升，25.63% → 35.74%（Epoch 4 平均）
- **Open-ended**：U-shape——Epoch 0 是 34.40%，Epoch 1 暴跌到 19.26%（catastrophic forgetting），Epoch 2-4 慢慢恢复到 34.82%

这个 U-shape 是 paper 里最重要的 dynamics 证据。它在告诉你：SFT 在 early epoch 破坏了 base model 的 language prior（用于 open-ended generation），mid-late epoch 重新学习新的 spatial-language alignment，但永远回不到 base 的 fluency。这是为什么 SFT 在 open-ended 上 net gain ≈ 0。

### 6.1 SFT 的 transfer learning 证据（Table 24）

在 OmniSpatial、SPACE、Mind the Gap 三个 external benchmark 上：

| Benchmark | Base | After SFT | Gain |
|---|---|---|---|
| OmniSpatial | 40.30% | 47.35% | +7.05% |
| SPACE (MM) | 23.43% | 28.67% | +5.24% |
| Mind the Gap | 35.86% | 47.56% | **+11.70%** |

这个 transfer learning 结果其实和前面 SFT 在 open-ended 上的失败看起来矛盾，但仔细想其实一致：external benchmark 都是 MCQ-format，所以 SFT 学到的 MCQ shortcut 在 transfer 时仍然 work。这恰恰是 paper 想警告的——**很多 spatial benchmark 的提升是 SFT 在 MCQ format 上的 overfitting**，不是 spatial reasoning 的真实进步。

参考 critique fine-tuning (Wang et al. 2025, https://arxiv.org/abs/2501.17703) 的类似观察。

---

## 七、SPATIOXOLVER：Multi-agent 的局限

Appendix H.5 描述了作者自己搭的 multi-agent system，基于 Xolver (https://arxiv.org/abs/2506.14234)。架构：

1. **Base Visual Analysis Agent**：Gemini-2.5-Flash 生成 detailed description
2. **Object Segmentation Agent**：把 description 转成 Obj1, Obj2, ... 列表
3. **Attribute Extraction Agent**：每个 object 转成 JSON（shape, size, color, orientation, ...）
4. **Spatial Relation Agent**：生成三元组 (ObjectA, Relation, ObjectB)
5. **Grouping and Symmetry Agent**：识别 row, grid, symmetry pattern
6. **Transformation Tracking Agent**（多帧）：translation/rotation/scaling log
7. **Representation Standardization Agent**：合并成 unified JSON
8. **Open-Ended Reasoning Agent**：基于 unified JSON 回答

结果（Table 25）：

| Category | MCQ Normal | MCQ Agent | Gain | Open Normal | Open Agent | Gain |
|---|---|---|---|---|---|---|
| 3D Geometry | 48.0 | 44.0 | −4.0 | 60.0 | 48.0 | −12.0 |
| Depth & Occlusion | 48.0 | 44.0 | −4.0 | 40.0 | 16.0 | **−24.0** |
| Orientation | 56.0 | 64.0 | **+8.0** | 0.0 | 36.0 | **+36.0** |
| Relative Positioning | 60.0 | 60.0 | 0.0 | 50.0 | 36.0 | −14.0 |
| Size & Scale | 48.0 | 52.0 | +4.0 | 25.0 | 24.0 | −1.0 |
| Spatial Navigation | 68.0 | 64.0 | −4.0 | 40.0 | 28.0 | −12.0 |

Orientation 上 +36% open-ended 是 dramatic gain——证明 sequential alignment 任务能从 explicit decomposition 受益。但 Depth & Occlusion −24% 是 dramatic loss，证明**当 perception prior 本身就弱时，multi-step agent 只是把错误信号 propagate 8 步而不是 1 步**。

这个发现对 agentic AI 的 hype 是个冷水：agent 不是 free lunch，agent 的增益正比于 base model 在该 task 上的 prior strength。Orientation 的 prior 是 binary direction（左/右/上/下），prior 强，agent work；Depth 的 prior 是 continuous metric + occlusion reasoning，prior 弱，agent 失败。

---

## 八、Error Taxonomy（Appendix I）：5 类系统性 failure

作者做了 qualitative error analysis，把失败归成 5 类。这是 build intuition 最直接的部分：

### 8.1 Spatial Mislocalization & Reference Confusion
模型在被问"左数第二个红色 cube"时，会 drift 到"最亮的 cube"或"中心的 cube"。Root cause：feature pooling（attention/conv）把相邻 object 的 representation 混在一起，没有 slot-based 或 pointer-token 机制 maintain object identity across reasoning steps。

### 8.2 Perspective & Scale Mistakes
模型对"toy car 比真人小"这种反 prior 场景会失败。Root cause：训练数据里 car 总是比人大，模型把 category-frequency prior 当 ground truth，不读 pixel-level metric cue。这是 distribution shortcut 而非 geometric inference。

### 8.3 Occlusion & Ordering Failures
面对 thin structure（wire、railing、tree branch），模型直接 omit。Root cause：large receptive field 在 pooling 时抹掉 thin edge，T-junction（深度关键 cue）丢失。

### 8.4 Attribute Confusion & Semantic Swap
把 crosswalk 的 paint stripe 当 "safe" 信号，忽略 traffic light state。Root cause：LM objective 奖励 plausible narrative，不奖励 visual grounding。

### 8.5 Open-Ended Rationalization Without Verification
问"为什么这个物体倾斜"，答"因为表面滑"——但 slipperiness 从图像里无法 infer。Root cause：decoder 没有 verification layer 把生成 token tie back to image region。

这 5 类 error 在所有 architecture 上 recurring，说明不是某个 model 的 bug，是整个 VLM training paradigm 的 structural weakness：**pretrain on web-scale image-text pairs + SFT on instruction-following + RLHF on helpfulness → 学到的是 "image → plausible caption"，不是 "image → grounded spatial predicate"**。

---

## 九、Statistical Robustness（Appendix K）：benchmark 自身的可靠性

为了证明 1400 QA pair 的结论不是 noise，作者做了详细统计检验：

### 9.1 ICC (Intra-Class Correlation)

$$
\text{ICC}(3, k) = \frac{\sigma^2_{\text{between}}}{\sigma^2_{\text{between}} + \frac{\sigma^2_{\text{within}}}{k}}
$$

变量解释：
- $\sigma^2_{\text{between}}$：item 之间的 variance（不同题难度差异）
- $\sigma^2_{\text{within}}$：同一 item 跨 run 的 residual variance
- $k$：run 数（这里 R=3）
- ICC > 0.75 表示 good reliability，> 0.9 表示 excellent

实测：所有 setting ICC > 0.98，证明 item 难度排序在 run 之间稳定。

### 9.2 Cohen's Kappa（LLM-as-judge reliability）

$$
\kappa = \frac{p_o - p_e}{1 - p_e}, \quad p_e = \sum_{c \in \{0,1\}} p_c^{(1)} p_c^{(2)}
$$

变量：
- $p_o$：observed agreement（人类和 LLM 一致的比例）
- $p_e$：expected agreement by chance
- $p_c^{(1)}, p_c^{(2)}$：两个 rater 各自把样本归到 class $c$ 的 marginal probability

实测：LLM judge (Gemini-2.5-Flash) 和人类 majority 的 $\kappa = 0.738$，accuracy 0.880，Fleiss' $\kappa$（多 rater）= 0.774。证明 open-ended evaluation 的 LLM judge 是 reliable 的。

### 9.3 Cronbach's Alpha

$$
\alpha = \frac{R}{R-1} \left( 1 - \frac{\sum_{r=1}^{R} s_r^2}{s_T^2} \right)
$$

变量：
- $R$：run 数
- $s_r^2$：run $r$ 的 variance
- $s_T^2$：跨 run 总分的 variance

实测：所有 setting $\alpha > 0.98$，证明 dataset 内部一致性极高。

### 9.4 Resampling study

随机抽 S=20 vs S=25 子集，1000 次重采样，Wilcoxon signed-rank test：

$$
z = \frac{W - \frac{n(n+1)}{4}}{\sqrt{\frac{n(n+1)(2n+1)}{24}}}
$$

变量：
- $W = \min(W^+, W^-)$，正/负 rank sum 的较小者
- $n$：非零配对数

实测：1000 次 trial 中 100% 的 p > 0.05，证明 subcategory size 即使减到 20 也 statistically indistinguishable from full set。说明 25 subcategory 的设计已经 over-sufficient。

这整段 statistics 是给 reviewer 看的，但对 build intuition 也有用：**VLM 在 spatial reasoning 上的 50% accuracy 不是 noise artifact**，是 reproducible 的 capability ceiling。

---

## 十、对 Karpathy 的 intuition 总结

把整篇 paper 压成 5 句话：

1. **Spatial reasoning 的 bottleneck 是 perceptual grounding，不是 reasoning chain length**——CoT 和 multi-agent 在 Orientation（binary direction prior 强）上 work，在 Depth & Navigation（continuous metric prior 弱）上 fail。
2. **Scaling laws 在 spatial reasoning 上 partially broken**——Llama-3.2-11B 比 InternVL3.5-1B 还差，说明 architecture + data 比 parameter count 重要。
3. **SFT 制造 "algorithmic straightjacket"**——MCQ accuracy 上升，open-ended 下降，因为模型学的是 format shortcut 而非 spatial relation。External benchmark 的 transfer gain 是同一个 overfitting 的另一面。
4. **Specialized spatial model 反而最差**——SpaceOm/SpaceThinker/SpaceQwen 证明 task-specific fine-tuning 损坏 general spatial ability，需要的是 general geometric representation 而非 task-specific classifier。
5. **Error 是 systematic 而非 random**——5 类 error 在所有 architecture 上 recurring，说明是 VLM training paradigm 的 structural weakness：image-text next-token prediction 不会学到 $r(o_i, o_j)$ 这种 structured predicate。

你之前在 state of GPT 系列讲过 "we're overfitting to the benchmark, not learning to reason"——SPATIALAB 给这句话提供了 quantified 证据。CoT 失败、SFT 反向、specialized model 退化，这三个现象合起来证明：在 spatial reasoning 上，当前 LLM-style training paradigm 已经接近 ceiling，要继续进步需要 architectural innovation（slot attention、geometry-aware encoder、physics-informed objective）+ data innovation（embodied simulation、counterfactual augmentation），而不是继续 scaling + RLHF。

---

## 十一、可以挖的 follow-up 方向（如果你想动手）

我顺着这篇 paper 联想到几个 concrete 的方向：

1. **Geometry-aware pretraining objective**：在 image-text pretraining 之外加一个 auxiliary loss，显式监督 pairwise spatial predicate。SpatialVLM (https://arxiv.org/abs/2401.12168) 的 depth + metric grounding 是个起点，但没有 cover occlusion / orientation / navigation。
2. **Slot-based object representation in VLM**：用 slot attention 或 object-centric learning (https://arxiv.org/abs/2006.15055) 在 visual encoder 里 maintain object identity across transformer layer，直接 attack "spatial mislocalization" error class。
3. **Counterfactual augmentation for spatial reasoning**：对每张图做 perturbation（移动 object、改变 occlusion、旋转 viewpoint），生成 counterfactual pair，强制 model 区分 perceptual evidence vs category prior。这能直接 attack "perspective & scale mistakes"。
4. **Embodied pretraining**：用 Habitat (https://arxiv.org/abs/1907.01418) 或 ThreeDWorld 这种 simulator 生成 embodied trajectory + spatial predicate supervision，让 VLM 学到 reference frame switching 和 multi-step navigation 的 grounded representation。RT-2 (https://arxiv.org/abs/2307.15818)、PaLM-E (https://arxiv.org/abs/2303.12989) 是这个方向的早期尝试。
5. **Verification module on top of generation**：在 VLM 输出后加一个 lightweight verifier，check 生成 claim 是否和 image region consistent。Self-RAG (https://arxiv.org/abs/2310.11511) 是 language-only 的版本，spatial 版本还没人做。
6. **Active perception for spatial reasoning**：把 VLM 接到 robot head，允许它主动改变 viewpoint 来 disambiguate occlusion / scale，把 spatial reasoning 从 passive single-image 推到 active multi-view。这直接 attack Depth & Occlusion 的 −24% agent 失败。

---

## 十二、最后的 meta 评论

这篇 paper 的 writing 不算特别 clean（1400 sample 在 30 subcategory 上 over-sampled，很多 statistical 重复保证；reasoning model 的 GPT-5-mini / o4-mini 命名是 placeholder，结果应该用 API timestamp 标注 reproducibility），但 diagnostic 价值很高。它做的事情很 Karpathy-style：build 一个 hard benchmark，让所有 SOTA model 在上面 expose systematic failure mode，然后做 controlled intervention（CoT / CoT-SR / SFT / Agent）证明这些 intervention 都不 work，最后给出 architectural root cause analysis。

它让你直观感受到的是：**当前 VLM 的 "intelligence" 主要来自 web-scale language pattern matching，而 spatial reasoning 是这个 pattern matching 的盲区**，因为 web image-text pair 里 spatial predicate 的标注密度极低、且 mostly implicit（caption 说 "cat on the table"，但不标注 cat 是否被 table 遮挡、table 的尺寸是否 perspective-distorted、cat 朝哪个方向）。要在 spatial reasoning 上有 human-level performance，需要的不是更多 RLHF，而是把 spatial predicate 显式注入 training signal。

这个结论对你 Eureka Labs (https://eurekalabs.ai) 那种 education-LLM 的方向也有 implication：teaching model to "reason about space" 需要 grounded supervision，而不是更多 reasoning chain。Spatial reasoning 是 System 1（perceptual）+ System 2（relational chaining）的 integrated task，单独优化任一边都不够。

如果你想深入，我推荐同时看 VSI-Bench (https://arxiv.org/abs/2412.14171)——它用 video 而非 image，测的是 spatial memory（"刚才视频里的桌子上有什么"），和 SPATIALAB 的 static-image spatial reasoning 是互补维度。两者合起来 cover 了 embodied spatial intelligence 的 image-side 和 video-side。

---

主要参考链接汇总：
- SPATIALAB：https://spatialab-reasoning.github.io/
- SpatialVLM (Chen et al. CVPR 2024)：https://arxiv.org/abs/2401.12168
- VSI-Bench (Yang et al. 2024)：https://arxiv.org/abs/2412.14171
- RoboSpatial (Song et al. 2024)：https://arxiv.org/abs/2411.16537
- OmniSpatial (Jia et al. 2025)：https://arxiv.org/abs/2506.03135
- Mind the Gap (Stogiannidis et al. 2025)：https://arxiv.org/abs/2503.19707
- InternVL3 (Zhu et al. 2025)：https://arxiv.org/abs/2504.10479
- Qwen2.5-VL (Bai et al. 2025)：https://arxiv.org/abs/2502.13923
- GLM-4.5V/4.1V-Thinking：https://arxiv.org/abs/2507.01006
- Llama 3 herd：https://arxiv.org/abs/2407.21783
- CLEVR (Johnson et al. 2016)：https://arxiv.org/abs/1612.06890
- GQA (Hudson & Manning 2019)：https://arxiv.org/abs/1902.08572
- What's Up (Kamath et al. EMNLP 2023)：https://aclanthology.org/2023.emnlp-main.571/
- Xolver (Hosain et al. 2025)：https://arxiv.org/abs/2506.14234
- RT-2 (Zitkovich et al. 2023)：https://arxiv.org/abs/2307.15818
- PaLM-E (Driess et al. 2023)：https://arxiv.org/abs/2303.12989
- LoRA (Hu et al. 2021)：https://arxiv.org/abs/2106.09685
- Critique Fine-Tuning (Wang et al. 2025)：https://arxiv.org/abs/2501.17703
