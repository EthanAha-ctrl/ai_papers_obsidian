---
source_pdf: Open-o3VideoGroundedVideoReasoning with Explicit Spatio-Temporal Evidence.pdf
paper_sha256: 792e5d13b8cca7d076b068b68b2bb9b28746a687bd204d516944d1d8625553ef
processed_at: '2026-08-06T00:03:18-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Open-o3 Video

好，我换个讲法，像咱们在 whiteboard 前面聊天那样讲。

---

## 这个 paper 到底在干嘛

你想象一个场景：给你一段视频，问你"小兔子被放到地上之后干了啥？" 

现在的 video LLM（Video-R1、Qwen2.5-VL 这些）会吐一段 reasoning text，然后给你答案。问题是 —— 你不知道它说的那些话到底对应视频里的哪一帧、哪个位置。它可能就是在 hallucinate，编一个听起来 plausible 的故事。

OpenAI o3 在 image 上解决这个问题的方法是让 model "think with images" —— reasoning 过程中会 emit crop、zoom 这些 visual operation 作为 evidence。这篇 paper 就是把这个 idea 搬到 video 上：让 model 在 reasoning 的时候，**吐出 timestamp 和 bounding box**，比如 `<obj>bunny</obj><box>[110,198,302,337]</box>at<t>18.0</t>s`。这样它说的每句话都能 verify —— 你去那一帧那个位置看，是不是真的有兔子。

听起来简单，但做起来极难，难到他们得发明两个新的 trick 才能训得动。下面讲为什么难。

---

## 为什么 naive 做法会崩

你可能会想：直接把 image grounding 的 RL recipe 搬过来不就行了？给 model 一个 reward，告诉它 "你给的 box 和 GT 的 IoU 越高越好，你给的 timestamp 离 GT 越近越好"。然后跑 GRPO。

会崩。崩在 **reward coupling** 上。

你看，spatial reward（box IoU）只在 timestamp 对的时候才有意义。你在第 18 秒画一个 box，GT 在第 30 秒，这个 box 的 IoU 就是 garbage —— 你在错帧上画对了物体也是错的，因为那个时刻那个位置根本没有这个动作。

所以 reward signal 是这样耦合的：

```
temporal 错 → spatial reward ≈ 0（不管 box 画多准）
temporal 对 → spatial reward 才有意义
```

训练早期 model 的 temporal 预测很烂，于是 spatial reward 几乎永远是 0。SGD 看不到梯度，model 学不动。这就叫 **spatial collapse** —— spatial 能力永远学不出来，因为 temporal 这个 prerequisite 没满足。

反过来，如果你把 temporal reward 设计得太宽松（"差不多对就行"），model 很快就把 temporal reward 吃满、saturate 了，没有动力继续把 timestamp refine 到精确值。但 timestamp 不精确，spatial reward 又不可靠。死循环。

这就是这篇 paper 要解决的核心 problem。

---

## 他们的两个 trick

### Trick 1: Adaptive temporal proximity（让 temporal reward 从松到紧）

Temporal reward 用一个 Gaussian kernel：

$$r_t = \frac{1}{M}\sum_m \exp\left(-\frac{\Delta t_m^2}{2\sigma^2}\right)$$

$\sigma$ 是带宽。$\sigma$ 大 → 只要 timestamp 落在 GT 附近几秒内就有 reward，signal 很 dense；$\sigma$ 小 → 只有非常接近 GT 才有 reward，signal 稀疏但精确。

他们的做法是 **$\sigma$ 从 4 anneal 到 1**。训练初期 $\sigma=4$，model 即使 timestamp 差个几秒也能拿到 reward，于是 spatial reward 也能开始 fire，整个 training 动起来。随着训练进行，$\sigma$ 缩小，model 被迫把 timestamp 越来越精确。最后 $\sigma=1$，timestamp 基本要精确到秒级。

直觉上这就是个 **curriculum**：先让 model 大致找对时间区域，再逼它精确。和你在 nn-zero-to-hero 里讲 "SGD 需要平滑的 loss landscape 才能起步" 是一个道理 —— 早期需要 dense gradient signal，后期才需要 precise objective。

### Trick 2: Temporal gating（让 spatial reward 只在时间靠谱时才激活）

Spatial reward 加一个 hard gate：

$$r_s = \frac{1}{M}\sum_m \mathbb{1}\{|t_m - t_{j^\star}^{gt}| \leq \tau\} \cdot \text{IoU}(b_m, b^{gt})$$

$\tau = 3$ 秒。只有当预测 timestamp 离 GT 在 3 秒以内，spatial reward 才非零。否则直接置零。

这个 trick 防的是另一种 hack：model 在错误的帧上框一个 salient object（比如背景里的人脸、显眼的红色汽车），IoU 可能很高，但其实和 question 完全无关。gating 强制 "你框的物体必须出现在正确的时间，才算数"。这把 spatio-temporal 强行绑成一个 coherent 的 3D evidence，而不是两个独立可 hack 的 dimension。

两个 trick 合在一起 —— adaptive proximity 提供 smooth 的 temporal 学习曲线，temporal gating 提供 hard 的 spatio-temporal coherence 约束 —— 训练就稳定了。Ablation 里去掉 adaptive 掉 0.7% mAM，去掉 gating 掉 1.4% mAM，gating 影响更大，说明防 hack 比 smooth curriculum 更关键。

---

## 数据这事儿其实是最难的

你以为 trick 是核心？其实数据才是。现有的 video dataset 没有一个同时给 timestamp + box + CoT 三样东西的。他们不得不用 Gemini 2.5 Pro 现造 5.9k 样本。

Pipeline 三步：
1. Gemini 2.5 Pro 生成 question + answer + key frame + box + reasoning
2. 用 Qwen2.5-VL-7B 做 visual verification —— crop 出来问 "这是不是 {object}"，过滤 hallucinated box
3. Self-consistency check —— reasoning text 里每个 entity 必须有对应 box 和 timestamp，否则丢弃

这个 "强模型生成、弱模型验证" 的 pattern 你应该很熟悉，类似 constitutional AI 的 critique，但 grounding 到 visual evidence 而不是 text rubric。

Ablation 显示：没有 spatio-temporal 数据，mAM 28.3%；加 VideoEspresso 的 5k 样本，31.1%；加他们自己造的 5.9k，33.7%。每 5k 高质量数据值 2-3 个点。数据质量 >> 数据数量。

---

## 为什么用 GSPO 不用 GRPO

GRPO 是 token-level importance ratio + token-level clipping。问题是 video reasoning 的 response 很长（CoT 几百 token），reward 是 sequence-level 的（一个 reward 给整个 response），但 gradient 更新是 token-level 的。这导致 high variance，长 CoT 训练不稳定。

GSPO (https://arxiv.org/abs/2507.18071) 把 importance ratio 和 clipping 都提到 sequence level：

$$s_i(\theta) = \left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}\right)^{1/|y_i|}$$

取 length-normalized 的 sequence-level ratio，然后整体 clip。这样 reward 和 update 的 granularity 对齐了。Ablation 显示 GSPO 比 GRPO 高 0.9% mAM，Chain1 temporal IoU 高 2.9% —— 长 CoT 的 temporal localization 收益最大，因为最长最不稳。

---

## Test-time scaling 这个 bonus

既然 model 现在会吐 evidence，inference 时候就可以用 evidence 做 self-verification。做法：

1. 让 model 生成 N=8 个 response（temperature=1.0）
2. 每个 response 里的 `<obj><box><t>` 提取出来，crop 对应视频区域
3. 把 crop 喂回 model，问 "这个 evidence 对回答问题有用吗"，打分 $s \in \{0, 1, 2\}$
4. 按 confidence 加权投票选最终答案

这比 naive majority voting 高 1 个点（WorldSense 37.5→38.5，VideoMMMU 52.3→54.1）。关键的 intuition 是：majority voting 会被 spurious pattern 误导（8 次里 5 次都错答 C，但 C 是 hallucination），而 confidence-aware voting 能识别出 "虽然我 5 次都说 C，但 5 次的 evidence 都不 supportive"，从而转向 evidence 更 consistent 的答案。这正是 o3-style grounded reasoning 的额外价值 —— evidence 本身是 test-time scaling 的 fuel。

---

## 结果一句话总结

V-STAR 上：相对 Qwen2.5-VL-7B base，mAM +14.4%，mLGM +24.2%，超过 GPT-4o。What（QA accuracy）从 33.5 涨到 61.0，这个涨幅巨大。说明 grounded reasoning 不只是让 model 会框 box，反而让 model 的 QA 能力也大幅提升 —— 因为 forced grounding 抑制了 hallucination，model 必须基于真实 visual evidence 回答。

VideoMME / WorldSense / VideoMMMU / TVGBench 上都有 1-4 个点的提升，说明这个 grounding 能力 generalize 到 general video understanding，没有 overfit 到 grounding task。

---

## 给你的几个联想点

1. 这和你之前讲 "vision model 应该在 pixel space 而不是 token space reasoning" 的 intuition 完全一致。Open-o3 Video 本质上就是让 video LLM 在 spatio-temporal coordinate space 里 emit reasoning step，而不只在 vocabulary space。

2. Reward coupling problem 在 single-agent RL 里很普遍 —— 比如 robotics 里 reach reward 依赖 grasp reward 先成功。他们的 anneal + gating 组合其实是个通用的 "prerequisite-aware reward shaping" 模板，可以搬到别的有 prerequisite 结构的任务上。

3. 数据 pipeline 用 Gemini 生成 + Qwen-VL 验证，这预示着 future —— strong model 当 teacher、weak model 当 verifier 的 asymmetry 会成为 data curation 的主流 pattern，因为 verifier 比 generator 容易。

4. 一个 limitation 他们没明说但我觉得重要：16 frame 采样对 long video 是 lossy 的。虽然他们 prepend absolute timestamp，但 16 frame 之间的 motion 信息丢失很多。这可能是为什么 long video 上提升没有 short video 大。要真做 hour-level video reasoning，可能得搭 hierarchical frame selection，类似 RAG 之于 long context。

5. Confidence-aware test-time scaling 那个 trick 其实就是 "让 model 自己 judge 自己的 evidence quality"，这和 self-consistency (https://arxiv.org/abs/2203.11171) 是一脉相承的，但 grounding 到 visual crop 让 judge 信号更可靠。这条路走下去可能会演化出 "visual chain-of-thought + visual self-critique" 的完整 paradigm。

---

核心 reference：
- Paper: https://marinero4972.github.io/projects/Open-o3-Video/
- V-STAR benchmark: https://arxiv.org/abs/2503.11495
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- GSPO: https://arxiv.org/abs/2507.18071
- Video-R1: https://arxiv.org/abs/2503.21776
- OpenAI o3: https://openai.com/index/introducing-o3-and-o4-mini/

---

# Open-o3 Video 深度技术解读

Andrej，这篇paper 让我想到你之前在Eureka Labs 和 tweet 里反复强调的那个 intuition —— LLM reasoning 的 bottleneck 不在 language fluency，而在 grounded perception。这篇工作 essentially 把 OpenAI o3 的 "thinking with images" 范式 (https://openai.com/index/introducing-o3-and-o4-mini/) 拓展到 video domain，但 key insight 是：**单纯把 image-grounding recipe 复制到 video 会 collapse**，因为 video reasoning 同时需要 temporal tracking 和 spatial localization，二者耦合在一起时 reward signal 会变得 extremely sparse。

下面我从 motivation、data、architecture、reward design、训练动力学、test-time scaling 几个维度拆解，目标是 build 你对这套设计为何能 work 的 intuition。

---

## 1. Motivation: 为什么 Text-only Video Reasoning 不够

Video-R1 (https://arxiv.org/abs/2503.21776) 和 VideoRFT (https://arxiv.org/abs/2505.12434) 这类工作把 GRPO 直接搬到 video QA 上，让 model 输出 chain-of-thought，但 reasoning trace 全是 text。问题在于：

- Model 可以 hallucinate 一个看起来合理的 rationale，但 verifier 无法 cross-check 这些 claim 是否真的对应视频中的某个 frame / region。
- 在 V-STAR (https://arxiv.org/abs/2503.11495) 这种需要 spatio-temporal grounding 的 benchmark 上，text-only reasoning 的 mIoU 几乎为 0（Table 1 里 Video-R1 系列的 Where IoU 都接近 0）。

这背后的本质是：video 的 evidence 是一个 3D entity $(t, x, y, w, h)$，而 text reasoning 把它坍缩成 1D symbol sequence。你要让 reasoning "verifiable"，就必须让 evidence 本身成为 reasoning trace 的一部分。这正是 o3 在 image 上做的事 —— 它会 emit `<crop>` / `<zoom>` 作为 reasoning token。Open-o3 Video 的对应做法是让 model emit `<obj>...</obj><box>[x_min, y_min, x_max, y_max]</box>at<t>...</t>s` 这种 structured evidence token。

---

## 2. 数据构造：STGR-CoT-30k 与 STGR-RL-36k

### 2.1 为什么需要新数据

现有 video grounding data 有三种 gap：

| 数据类型 | 时间标注 | 空间标注 | CoT |
|---|---|---|---|
| Temporal grounding (ActivityNet, DiDeMo, COIN) | ✓ | ✗ | 部分 |
| Spatial grounding (RefCOCO, VisCoT) | ✗ | ✓ | 部分 |
| Video QA (VideoMME, NExT-QA) | ✗ | ✗ | ✗ |

没有任何一个 dataset 同时提供三者。这是阻碍 joint spatio-temporal reasoning 的核心 bottleneck —— RL 需要 verifiable reward，而 reward 需要对齐的 ground truth。

### 2.2 数据 pipeline（Figure 2 left）

5.9k 新样本的构造过程：

1. **Initial annotation with Gemini 2.5 Pro**：输入是 video + dense caption（来自 PLM-Rdcap, https://arxiv.org/abs/2504.13180）或 temporal segment annotation。Prompt 让 Gemini 输出 JSON，包含 question、answer、1-5 个 key frame timestamps、每个 frame 上 1-3 个 object 的 bounding box、以及一个必须引用所有 `<obj><box><t>` 的 reasoning process。
2. **Bounding box filtering**：移除覆盖 >80% frame 的 box（uninformative），再用 Qwen2.5-VL-7B (https://arxiv.org/abs/2502.13923) crop-and-verify —— 问 "Is this a {object_name}?"，只保留 yes 的样本。这一步等价于一个 visual consistency check，防止 Gemini hallucinate 不存在的 object。
3. **Self-consistency checking**：检查 reasoning text 中每个 entity 是否都有对应 box 和 timestamp；删除 unmatched reference，丢弃 mismatched sample。

这个 pipeline 的 key insight 是：**用 strong model 生成、用 weaker-but-cheaper model verify**，类似 constitutional AI 的 self-critique，但 grounding 到 visual evidence 而不是 text rubric。

### 2.3 数据配比

SFT 集 30k 的组成：
- 4.1k temporal grounding CoT (TVG-Coldstart, https://arxiv.org/abs/2507.18100)
- 5k spatial grounding CoT (TreeVGR-SFT, https://arxiv.org/abs/2507.07999)
- 5.9k spatio-temporal (本文新构造)
- 15k Video-R1-CoT general QA

RL 集 36k 的组成：
- 5.2k temporal (Time-R1 + TVG-RL)
- 5k spatial (VisCoT, https://arxiv.org/abs/2505.15879 ... 实际是 Visual-CoT NeurIPS 2024)
- 10.9k spatio-temporal (5.9k 自有 + 5k VideoEspresso, https://arxiv.org/abs/2506.16573)
- 15k Video-R1

Table 6 的 ablation 很有意思 —— 15k general QA 是 sweet spot，加 30k 反而 hurt grounding。这说明 grounding 和 QA 之间存在 capacity competition，需要 ratio balance。这一点和你之前在 nn-zero-to-hero 里讲的 "SGD 会在不同 loss term 之间找 shortest path" 的 intuition 一致。

---

## 3. 模型架构与训练 Pipeline

### 3.1 Base model

Qwen2.5-VL-7B (https://arxiv.org/abs/2502.13923)，输入是 16 个 uniformly sampled frame，每帧 resolution ≤ 128×128 token（native resolution dynamic encoding）。如果 annotated key frame 存在，额外插入。每帧 prepend 绝对 timestamp —— 这是让 model 感知 absolute temporal position 的关键，否则 model 只能从 frame order 推 relative position。

### 3.2 两阶段训练（Figure 3）

**Stage 1: Cold-start SFT**
- 在 STGR-CoT-30k 上训练 1 epoch，lr = 1e-6
- 目的：让 model 学会输出 structured format `<answer>...</answer>`
- 没有 cold start，RL reward 会 extreme sparse，model 根本不会 emit 正确 format，更别说 align

**Stage 2: RL with GSPO**
- 在 STGR-RL-36k 上训练 1 epoch，lr = 1e-6
- 用 Group Sequence Policy Optimization (https://arxiv.org/abs/2507.18071) 而非 GRPO

---

## 4. Reward Design 的核心数学

这是这篇 paper 最 elegant 的部分。Total reward：

$$r(x, y) = r_{\text{acc}}(x, y) + r_{\text{thk}}(x, y) + r_{\text{fmt}}(x, y)$$

其中 $r_{\text{acc}}$ 是 task-specific accuracy，$r_{\text{fmt}} \in \{0, 0.5, 1.0\}$ 检查 format，$r_{\text{thk}}$ 是 thinking reward：

$$r_{\text{thk}}(x, y) = r_t(x, y) + r_s(x, y)$$

### 4.1 Temporal reward $r_t$ with adaptive proximity

设 model 输出的 timestamps 为 $\{t_m\}_{m=1}^M$，ground truth interval 为 $[s^{gt}, e^{gt}]$ 或 discrete timestamps $\{t_j^{gt}\}$：

$$r_t(x,y) = \begin{cases} \frac{1}{M}\sum_{m=1}^M \mathbb{1}\{s^{gt} \leq t_m \leq e^{gt}\} & \text{interval supervision} \\ \frac{1}{M}\sum_{m=1}^M \exp\left(-\frac{\Delta t_m^2}{2\sigma^2}\right) & \text{point supervision, } \Delta t_m = \min_j |t_m - t_j^{gt}| \\ 0 & \text{no timestamp} \end{cases}$$

变量含义：
- $M$ = model 输出的 timestamp 个数（parsed from `<t>...</t>` token）
- $t_m$ = 第 m 个预测 timestamp（秒）
- $s^{gt}, e^{gt}$ = ground truth interval 起止
- $t_j^{gt}$ = 第 j 个 ground truth point timestamp
- $\Delta t_m$ = 预测 timestamp 到最近 GT 的距离
- $\sigma$ = Gaussian kernel 的 bandwidth，**这是 annealed 的**：从 4 退火到 1

**Intuition**：这个 Gaussian kernel 等价于一个 soft indicator function。当 $\sigma$ 大时，只要 timestamp 落在 GT 附近 ±几秒就有 reward，signal dense；当 $\sigma$ 小时，只有非常接近 GT 才有 reward，signal sparse 但精确。

为什么需要 anneal？因为 spatial reward $r_s$ 依赖 temporal 预测准确（你只有在正确 frame 上 crop box 才有意义）。早期 model temporal 预测很烂，如果 $\sigma$ 一开始就小，spatial reward 永远接近 0，learning stall。但 $\sigma$ 一直大，temporal reward 提前 saturate，model 没动力继续 refine timestamp。所以 anneal 创造了一个 curriculum —— 先粗后细。

### 4.2 Spatial reward $r_s$ with temporal gating

对每个预测 timestamp $t_m$，找到最近的 GT timestamp：

$$j^\star(m) = \arg\min_j |t_m - t_j^{gt}|$$

然后 spatial reward：

$$r_s(x, y) = \frac{1}{M}\sum_{m=1}^M \mathbb{1}\{|t_m - t_{j^\star(m)}^{gt}| \leq \tau\} \cdot \max_{b \in \mathcal{B}_m, b^{gt} \in \mathcal{B}_{j^\star(m)}^{gt}} \text{IoU}(b, b^{gt})$$

变量含义：
- $\tau$ = temporal gating threshold，设为 3 秒
- $\mathcal{B}_m$ = 第 m 个 timestamp 上预测的所有 box 集合
- $\mathcal{B}_{j^\star(m)}^{gt}$ = 最近 GT frame 上的所有 GT box 集合
- $\max$ over IoU：Hungarian matching 的简化版，允许预测多个 box 中任一个对上 GT 就算

**Intuition**：gating $\mathbb{1}\{|t_m - t_{j^\star(m)}^{gt}| \leq \tau\}$ 是一个 hard gate —— 只有 timestamp 足够准，spatial reward 才非零。这防止一个 failure mode：model 在 wrong frame 上指一个 salient 但 irrelevant object（比如背景里的人脸），拿到高 IoU reward。gating 强制 spatio-temporal coherence。

这两个 mechanism 是 complementary 的：adaptive proximity 让 temporal reward 从粗到精提供 dense signal，temporal gating 让 spatial reward 只在 temporal 可靠时才激活。一个 smooth、一个 hard，组合起来避免 reward sparsity 和 reward hacking 两个极端。

### 4.3 Format reward $r_{fmt}$

- 1.0：有 `
