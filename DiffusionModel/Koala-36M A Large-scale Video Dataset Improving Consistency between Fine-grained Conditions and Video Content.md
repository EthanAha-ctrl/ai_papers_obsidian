---
source_pdf: Koala-36M A Large-scale Video Dataset Improving Consistency between Fine-grained
  Conditions and Video Content.pdf
paper_sha256: 7479b0aea72ecffc055f5f8d24f8d551b27d141dc52722cd7a7e22eb5a0170d8
processed_at: '2026-08-05T11:27:17-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Koala-36M 用人话说

Andrej，我把这篇 paper 用最朴素的语言重新讲一遍，不堆 jargon，直接讲 intuition。

---

## 一句话说清楚这 paper 在干嘛

Kuaishou 团队拿了和 Panda-70M **完全同一份 raw video**，重新做了一遍数据清洗，结果用 36M 的数据训练出来的 video generation model，在 VBench 上把用 70M 训练的 model 打了。

换句话说，**同样的原材料，加工工艺更精细，36M 打 70M**。这就是 paper 的全部 thesis。

---

## 为什么要重新加工？原始数据到底哪里脏？

你想啊，训练 video diffusion model 本质上就是让模型学习"text → video"的 mapping。如果训练数据里 text 和 video 对不上号，模型学到的就是 garbage。Panda-70M 有 70M 量级，但每条 caption 平均才 13.2 个 word——一句话都不到。这就好比训练 GPT，每个文档只给它一句话的 summary，模型永远学不到细节。

更糟糕的是，原始 video 是 long video 切出来的，切的时候用的是 PySceneDetect 这种基于阈值的土办法。如果切的位置不对，一个 clip 里可能跨了一个 scene transition——前半段是沙滩，后半段是雪山，caption 只描述了沙滩。模型学到的是"沙滩 → 雪山"这种鬼 mapping，生成出来的 video 就会莫名其妙地跳 cut。

还有一个问题：低质量 video 怎么过滤？传统做法是设一堆阈值——clarity > 0.5 AND aesthetic > 0.6 AND motion > 0.3...。听起来合理，但你仔细想，这些 sub-metric 之间是有耦合的。Table 7 显示 Clarity 和 Motion 的 Spearman 相关是 **-0.43**——motion 大的 video 本来就模糊（因为 motion blur），你拿 clarity 阈值一卡，把所有 motion 大的好 video 都卡掉了。这种累积误差导致大量 high-quality data 被误删，Fig 5 里就展示了这种惨案。

---

## 四步 pipeline 用大白话讲

### Step 1: 切视频——Color-Struct SVM (CSS)

**人话**：判断两帧之间是不是发生了 scene cut。

PySceneDetect 的做法是看相邻两帧的像素差异，差异大就判为 cut。问题是：快速运动的场景（比如跑步、动作片）帧间差异也大，会被误判为 cut；淡入淡出的转场帧间差异小，会被漏判。

Koala 的做法分两步：

**第一步**：用两个 feature 衡量帧间差异。
- **颜色差异** $d_{color}$：算两帧的 BGR 颜色直方图的 Pearson correlation。值域 [-1, 1]，1 = 颜色分布完全一样。
- **结构差异** $d_{struct}$：先把 gray image 和它的 Canny edge map 做逐像素 max（相当于"亮度 + 边缘"的复合图），再算 SSIM。

$$d_{color}(H_i, H_j) = \frac{\sum_p (H_i(p) - \bar{H}_i)(H_j(p) - \bar{H}_j)}{\sqrt{\sum_p (H_i(p) - \bar{H}_i)^2 \sum_p (H_j(p) - \bar{H}_j)^2}}$$

变量解释：
- $H_i(p)$：第 $i$ 帧的 BGR 直方图在第 $p$ 个 bin 的值
- $\bar{H}_i$：第 $i$ 帧直方图所有 bin 的均值
- $p$：遍历所有 histogram bin

**第二步**：把 $(d_{color}, d_{struct})$ 这对 2D 特征喂给一个 linear SVM，分类"是不是 cut"。训练数据怎么来？同一 video 内的帧对是 negative（大概率不是 cut），跨 video 的帧对是 positive（一定是 cut）。这是 self-supervised 的巧思——cut 在任意时刻出现概率低，跨 video 天然就是 hard cut 的近似。

**第三步**：temporal smoothing。假设正常帧间变化服从 Gaussian，从过去 $k$ 帧估出 $\hat{\mu}, \hat{\sigma}$。当前帧的变化如果超过 $\hat{\mu} + 3\hat{\sigma}$，才认为是 cut。这样 fast motion 的连续大变化会被 baseline 吸收（因为过去 $k$ 帧变化也大，$\hat{\sigma}$ 也大），不会误判；而 gradual transition 会突然冒出来一个异常值，会被检出。

**实验结果**（Table 3）：

| Method | Accuracy | Recall | Precision |
|---|---|---|---|
| Pydetect(hsl) | 0.4421 | 0.3096 | 0.5920 |
| Pydetect(hsl+edge) | 0.4574 | 0.4146 | 0.5854 |
| **Koala CSS** | **0.7741** | **0.9395** | **0.7547** |

Accuracy 从 0.44 到 0.77，几乎翻倍。Recall 0.94 意味着几乎不漏检——宁可多杀也不放过。

Runtime 在 720p 上 6.15ms vs Pydetect 30.57ms，5x 加速。原因是 histogram + SSIM 计算极其 cheap。

---

### Step 2: 写 caption——Structured Caption System

**人话**：把 caption 拆成六个维度分别描述，再拼起来。

六个维度：
1. **Subject**——画面主体是谁/什么
2. **Subject Actions**——主体在做什么
3. **Environment**——场景、背景、位置
4. **Visual Language**——风格、构图、光线（摄影术语）
5. **Camera Language**——镜头运动、角度、焦距、景别
6. **World Knowledge**——常识或文化背景

这个 decomposition 的 intuition 很直接：让 captioner 一次想六件事，每件都想清楚，比让它一口气写一段话质量高得多。就像让 LLM 写文章，你给它 outline 它写得更好。

**训练 captioner 的细节**：
- Teacher：GPT-4V 按六维结构生成少量高质量 caption
- Student：基于 LLaVA fine-tune
- **关键 trick 1**：vision encoder 也 fine-tune（不是 frozen）→ caption accuracy 明显提升
- **关键 trick 2**：high-res vision encoder + 2x2 average pooling 在 token 空间维度 → token 数减 4x，information loss 最小
- **关键 trick 3**：mixed training（static image + dynamic video）→ 缓解 video 训练数据不足

**结果**：平均 caption 长度 202.1 words，vs Panda 的 13.2 words。信息密度提升 15x。

**caption 长度分布**（Fig 4）大致 symmetric，峰值在 200 左右，分布范围从 100 到 300+，说明 captioner 不是无脑堆字数，是根据内容长度自适应。

---

### Step 3: 过滤低质量 video——Video Training Suitability Score (VTSS)

**人话**：训练一个"评分员"网络，给每个 video 打一个分，只留高分的那批。

**核心 insight**：传统多阈值过滤忽略了 sub-metrics 之间的 joint distribution。Koala 的解法是训一个网络，把 video 和所有 sub-metrics 一起喂进去，输出一个标量分数 VTSS，用这个单一分数过滤。

**标注标准三维度**：
- **Dynamic Quality**：motion area > 30% frame area + camera temporal stability（拒绝鬼畜手持抖动）
- **Static Quality**：subject detail、composition、aesthetic、color saturation
- **Video Naturalness**：无特效 / 字幕 / logo / 政治暴力内容

**标注流程**：200K videos，每个 video 由 8 个 expert 打分（1-5），然后做两种 bias correction：
- **Individual Preference Bias**：有的 expert 天生手松，有的天生手紧。用 z-score normalize 每个 expert 的打分，再 rescale 到全局 mean/variance
- **Label Fluctuation Bias**：8 个 expert 取 mean

**TSA Network 架构**（Fig 6）：

```
Video → ┬─→ 3D Swin Transformer → dynamic feat
        │
        ├─→ ConvNeXt → static feat
        │
        └─→ Sub-metrics (motion, aesthetic, clarity...) → label feat
                                                        │
                            Weight Cross-Gating Block (WCGB)
                                                        │
                                            Fusion → MLP → VTSS (scalar)
```

**WCGB** 的作用：label feature（比如 motion score）既和 dynamic 相关又和 static 相关，所以用 learned weight 把 label feature 分配到两个分支。形式化大概是：

$$f_{dyn}' = f_{dyn} \odot \sigma(W_1 f_{label}), \quad f_{sta}' = f_{sta} \odot \sigma(W_2 f_{label})$$

变量解释：
- $f_{dyn}$：dynamic branch 输出 feature
- $f_{sta}$：static branch 输出 feature
- $f_{label}$：sub-metrics 经过 embedding 后的 feature
- $W_1, W_2$：learned projection matrix
- $\sigma$：sigmoid gating function
- $\odot$：element-wise product

**Ablation 结果**（Table 5）：

| Config | PLCC↑ | SRCC↑ |
|---|---|---|
| Dynamic only | 0.8684 | 0.8580 |
| + Static | 0.8730 | 0.8637 |
| + Feature | 0.8953 | 0.8864 |
| + WCGB | **0.8974** | **0.8868** |

对比 Dover（Table 6）：Dover PLCC 0.8554，Koala 0.8974，**+4.2 pts**。Dover 是 video quality assessment 的 SOTA，但 Koala 的 TSA 显式建模 sub-metrics joint distribution，所以更准。

**阈值选择**（Fig 7）：VTSS 分布看起来像两个 Gaussian 的 mixture，交点在 2.5 附近，直接拿 2.5 做阈值。从 48M 全集过滤到 36M，过滤掉 25%。

---

### Step 4: 把 quality 信号注入 model——Metric Conditions

**人话**：Step 3 过滤掉了低质量 video，但保留下来的 36M video 质量仍然参差不齐。与其只做"过滤"，不如把每个 video 的质量分数直接告诉模型，让模型自己学着区分。

**实现**（Fig 8）：

```
motion score ─┐
aesthetic ────┤
clarity ───────┼─→ Frequency Embedding ─→ MLP ─→ + to timestep emb
... ──────────┘                                          │
                                                          ↓
                                                Adaptive LayerNorm (AdaLN)
                                                          │
                                                          ↓
                                                Transformer block
```

**Frequency embedding** 就是 Diffusion 里 timestep embedding 同款 sinusoidal：

$$\text{emb}(s) = [\sin(\omega_1 s), \cos(\omega_1 s), \sin(\omega_2 s), \cos(\omega_2 s), ...]$$

变量解释：
- $s$：normalized quality score（比如 0 到 1 之间）
- $\omega_k = 1/10000^{k/d}$：第 $k$ 个频率，$d$ 是 embedding dimension

**AdaLN injection**：

$$\gamma, \beta = \text{MLP}(c_t + c_{metric})$$
$$h' = \gamma \cdot \text{Norm}(h) + \beta$$

变量解释：
- $c_t$：timestep embedding
- $c_{metric}$：metric frequency embedding 经过 MLP 后的输出
- $\gamma, \beta$：AdaLN 的 scale 和 shift 参数
- $h$：transformer block 的 hidden state

**vs OpenSora 的做法**：OpenSora 把 metric 信息塞进 text prompt。Koala 的方法优势：
1. Frequency embedding 对数值敏感（连续），text prompt 是离散的
2. 解耦能力强——可以单独拉满 motion score 而 aesthetic 保持（见 Fig 13）
3. 不增加 FLOPs——text encoder 还是只处理 text

---

## 实验结果最亮的几个数字

### VBench 主结果（Table 2）

| Setting | Quality | Semantic | Total |
|---|---|---|---|
| Panda-70M | 0.7343 | 0.3093 | 0.6493 |
| Koala-w/o TSA (48M, 无过滤) | 0.7758 | 0.4668 | 0.7140 |
| Koala-37M-manual (手动阈值) | 0.7704 | 0.4548 | 0.7073 |
| Koala-36M (VTSS 过滤) | 0.7819 | 0.4504 | 0.7156 |
| Koala-36M + condition | **0.7846** | **0.5915** | **0.7460** |

**四个关键 insight**：

1. **Koala-w/o TSA > Panda-70M**（Total +0.065）——仅靠更准确的 split 和 200-word caption，不靠任何过滤，已经全面超越 Panda。这隔离了 splitting + captioning 的贡献。

2. **Koala-36M > Koala-37M-manual**——TSA 自动过滤比手动多阈值更准，且保留更少 data（36M vs 37M）反而 quality 更高。少即是多。

3. **+ Metric Condition 提升最大**：semantic score 从 0.4504 → 0.5915，**+31%**。把 quality signal 直接 inject 到 model，远比仅用作 filter signal 更有价值。

4. **Aesthetic Quality 单项飞跃**：Panda 0.3988 → Koala-36M cond 0.5318，**+33%**。这归功于 structured caption 里 "Visual Language" 维度。

### 跨分辨率、跨时长（Table 9）

| Setting | Quality | Semantic | Total | FVD↓ |
|---|---|---|---|---|
| Panda 256-2s | 0.7343 | 0.3093 | 0.6493 | 570.87 |
| Koala-36M cond 256-2s | 0.7846 | 0.5915 | 0.7460 | **549.79** |
| Koala-36M cond 512-2s | 0.7849 | 0.6495 | 0.7578 | **392.26** |

**512 分辨率上 FVD 从 579.57 → 392.26，相对降低 32%**。data quality 的增益在高分辨率下放大——bad data 在低分辨率下还能"伪装"，高分辨率下 noise 会被放大。

---

## 我的几个 critical thoughts

### 1. Captioner 的天花板
Captioner 是 LLaVA-based，意味着 caption 质量上限受 vision-language model 限制。如果未来用 GPT-4o / Gemini 1.5 Pro 级别的 captioner，质量还能再上一个台阶。这也意味着 Koala-36M 的 caption 可能在某些 fine-grained visual detail（比如文字、数字、小物体）上仍然有 noise。

### 2. TSA 的标注成本
200K videos × 8 experts = 1.6M annotations。这是 paper 没充分讨论的成本。reproducing 这部分需要相当大的标注投入。有没有可能用 active learning 减少 annotation 量？或者用 DPO-style 的 pairwise comparison 代替绝对打分？

### 3. Metric Condition 在 inference 时的"作弊"风险
训练时 metric 是 ground truth，inference 时需要 user 指定（或全部设 max）。Fig 13 显示设 max 时的确生成高质量 video，但 user 不一定知道每个 prompt 对应的最优 metric。比如 prompt 是 "slow cinematic shot"，你把 motion score 拉满反而错。这个 gap 需要 inference-time 的 metric prediction model 来填补。

### 4. CSS 的边界 case
依赖 BGR histogram + SSIM，对低光 / 高 dynamic range 场景可能失效。比如夜景 video 的 histogram 几乎全黑，correlation 可能虚高。HDR video 的亮度分布也会扭曲 histogram。这些都是潜在 failure mode，paper 没 ablation。

### 5. 和 MiraData 的对比缺失
MiraData 平均 caption 318 words，比 Koala 还长。虽然 scale 小（330K），但至少在 caption quality 维度上值得做 head-to-head comparison。作者可能认为 scale 差太多不可比，但从 research 角度这个对比 informative。

### 6. "Data quality > data quantity" 的更深层含义
这篇 paper 其实是在说：video diffusion 的 scaling law 不仅仅关于 parameter count 和 data count，还关于 data 的 information density。13 word caption 的 70M data 和 200 word caption 的 36M data，后者总 information content 反而更高（36M × 202 vs 70M × 13 = 7.27B vs 0.92B word tokens）。这给了一个新的 scaling 维度——**caption density × data count** 才是真正的 information budget。

---

## 类比 LLM 数据工程

如果你想用 LLM 的经验来理解这篇 paper：

| Video Dataset Pipeline | LLM Data Pipeline 对应 |
|---|---|
| Video splitting (CSS) | Document splitting（避免跨文档切分） |
| Structured captioning | High-quality instruction tuning data |
| VTSS filtering | Quality filtering（比如去掉 low-quality web text） |
| Metric conditioning | Conditioning on data source / quality metadata |

最后一项特别有意思——LLM 训练里很少有人把 data quality score 作为 condition inject 进 model。如果把这个思路搬到 LLM：训练时给每个 document 一个 quality score，用 frequency embedding + AdaLN inject，inference 时设 max quality，能不能让 model 生成更高质量 text？这是一个 open question，但 Koala 的实验数据（semantic score +31%）暗示这条路 worth trying。

---

## Limitations 作者自己承认的

36M 不足以训练 1B+ 参数的 video generation model。这暗示 Kuaishou 内部可能有更大版本的 dataset（如果他们做 1B+ model 的话）。也暗示了一个研究方向——如何把这套 pipeline scale 到 100M+ 甚至 1B+ 级别，同时保持 caption quality 和 filtering accuracy。

---

## References

- **Koala-36M Project Page**: https://koala36m.github.io/
- **Panda-70M (Snap Research)**: https://snap-research.github.io/Panda-70M/
- **MiraData**: https://github.com/xju-essentia/MiraData
- **OpenVid-1M**: https://github.com/OpenVDB/OpenVid-1M
- **VBench**: https://vchitect.github.io/VBench-project/
- **LLaVA**: https://llava-vl.github.io/
- **Stable Video Diffusion**: https://stability.ai/news/stable-video-diffusion-open-ai-video-model
- **PySceneDetect**: https://github.com/Breakthrough/PySceneDetect
- **OpenSora**: https://github.com/hpcaitech/Open-Sora
- **Sora (OpenAI)**: https://openai.com/sora
- **GPT-4V System Card**: https://openai.com/research/gpt-4v-system-card
- **ConvNeXt**: https://github.com/facebookresearch/ConvNeXt
- **3D Swin Transformer**: https://github.com/microsoft/Swin-Transformer
- **DiT (Diffusion Transformer)**: https://www.wpeebles.com/DiT
- **AdaLN (Peebles & Xie, NeurIPS 2022)**: https://arxiv.org/abs/2212.09748
- **Dover Video Quality Assessment**: https://github.com/VQAssessment/DOVER
- **FastVQA**: https://github.com/VQAssessment/FAST-VQA-and-FasterVQA
- **VidGen-1M**: https://arxiv.org/abs/2408.02629
- **HD-VILA-100M**: https://arxiv.org/abs/2111.10337
- **HowTo100M**: https://www.di.ens.fr/~miech/howto100m/

如果你想进一步 deep dive 某个 component（比如 CSS 在 long video 上的 failure cases、TSA 网络的具体实现、或者 metric condition 在不同 architecture 上的 transferability），告诉我。

---

# Koala-36M Paper 深度解析

Andrej, 这篇 paper 由 Kuaishou Technology 团队发布，核心 thesis 是：**当 video generation 模型本身的架构日趋成熟后（DiT, Sora-like），performance 的瓶颈从 model capacity 转移到了 fine-grained conditions 与 video content 之间的 consistency**。Koala-36M 通过四步 pipeline（splitting → captioning → filtering → conditioning）从同一份 Panda-70M 的 raw data 出发，筛出 36M clips，却在 VBench 上以更小的 data size 全面击败 Panda-70M。这本身就是一个非常强的 "data quality > data quantity" argument。

---

## 1. Problem Framing: 三大核心痛点

作者把现有 dataset 的问题精炼为三条，build intuition 上可以这样理解：

**(a) Text-Video Alignment 粗糙**
Panda-70M 平均 caption 只有 13.2 words（见 Table 1），vs Koala 的 202.1 words。这就好比训练 LLM 时只用一句话 summary 训练 next-token prediction，模型永远学不到 fine-grained semantics。

**(b) Transition 检测不准 → Temporal Incoherence**
原始 long video 包含大量 hard cut / fade in-out / dissolve transitions。如果一个 clip 跨过 transition，caption 描述的就是"前后两个场景"，model 学到的本质上是 noisy mapping，generated video 会随机出现 unwanted cuts。

**(c) Multi-Threshold Filtering 的累积误差**
传统做法是：clarity score > T1 AND aesthetic > T2 AND motion > T3 ...。由于 sub-metrics 之间不 orthogonal（Table 7 显示 Clarity 与 Motion 的 Spearman 相关 -0.4324），独立设 thresholds 会累计偏差，导致 high-quality data 被误删（见 Fig 5）。

---

## 2. Method 1 — Color-Struct SVM (CSS) Transition Detection

这是 paper 里最 math-y 的部分，值得仔细看。

### 2.1 形式化定义

对一对帧 $I_i, I_j$，定义两个 distance：

**Color distance**（基于 BGR histogram 的 Pearson correlation）：

$$H_i = \mathrm{Histogram}(\mathrm{bgr}(I_i)) \quad (1)$$

$$d_{color}(H_i, H_j) = \frac{\sum_p (H_i(p) - \bar{H}_i)(H_j(p) - \bar{H}_j)}{\sqrt{\sum_p (H_i(p) - \bar{H}_i)^2 \sum_p (H_j(p) - \bar{H}_j)^2}} \quad (2)$$

变量含义：
- $H_i(p)$：第 $p$ 个 bin 在 frame $i$ 的 BGR 直方图频率
- $\bar{H}_i$：frame $i$ 直方图所有 bin 的均值
- $p$：遍历所有 histogram bin

注意分母展开是 $\sqrt{\sum_p(\cdot)^2 \cdot \sum_p(\cdot)^2}$，本质就是 Pearson correlation coefficient $\rho(H_i, H_j)$，范围 $[-1, 1]$。值为 1 表示颜色分布完全一致，0 表示线性无关。

**Structural distance**（基于 Canny edge + SSIM）：

$$E_i = \max(\mathrm{Gray}(I_i), \mathrm{Canny}(\mathrm{Gray}(I_i))) \quad (3)$$

$$d_{struct}(E_i, E_j) = \mathrm{SSIM}(E_i, E_j) \quad (4)$$

变量含义：
- $E_i$：gray image 与其 Canny edge map 的逐像素 max，相当于"intensity + edge"的复合 map
- $\mathrm{SSIM}(\cdot, \cdot)$：Structural Similarity Index，对 luminance / contrast / structure 三项加权

这里有个 intuition 值得注意：用 $\max$ 而非 sum/concat，是为了让 edge 信息"叠加"在 intensity map 上，保留 spatial structure。

### 2.2 SVM + Temporal Smoothing

把 $(d_{color}, d_{struct})$ 作为 2D 特征，喂给 linear SVM 分类器：
- **Negative pairs**：同一 video 内的相邻帧
- **Positive pairs**：不同 video 的帧

这是 self-supervised 的 trick：transitions 在任意时刻出现概率低，所以"跨 video"天然就是 hard cut 的近似。

**Temporal smoothing** 是关键的 anti-false-positive 机制：
- 假设 frame-to-frame 变化服从 Gaussian，从过去 $k$ 帧估计 $\hat{\mu}, \hat{\sigma}$
- 当前 frame 变化若超过 $\hat{\mu} + 3\hat{\sigma}$ → 判为 transition
- 好处：fast-motion scene（比如跑步、动作戏）的连续大变化会被 baseline 抹平，gradual transition（fade、dissolve）则会被检出

### 2.3 实验数据（Table 3 & 4）

| Method | Accuracy | Recall | Precision |
|---|---|---|---|
| Pydetect(hsl) | 0.4421 | 0.3096 | 0.5920 |
| Pydetect(hsl+edge) | 0.4574 | 0.4146 | 0.5854 |
| **Ours** | **0.7741** | **0.9395** | **0.7547** |

特别值得关注的是 **Recall 0.9395**：作者刻意强调 high recall 是设计目标——宁可多检测几个 false positive 然后用 temporal smoothing 过滤，也不要漏掉 transition。

Runtime 对比（Table 4）：720p 上 Ours 6.15ms vs Pydetect(hsl+edge) 30.57ms，**5x 加速**。原因是 histogram + SSIM 计算非常 cheap，远比 per-pixel HSL 阈值判断高效。

---

## 3. Method 2 — Structured Caption System

### 3.1 六维结构

作者把 caption 分解成六个 orthogonal 维度：

1. **Subject** — 主体的外观、身份
2. **Subject Actions** — 主体在做什么
3. **Environment** — 场景、背景、位置
4. **Visual Language** — style, composition, lighting
5. **Camera Language** — camera movement, angle, focal length, shot size
6. **World Knowledge** — 常识 / 文化背景

这个 decomposition 的 intuition 类似 ControlNet 用多种 condition 信号源分离解耦——每个维度独立生成再 merge，避免 captioner 把所有信息揉成一锅粥。

### 3.2 Captioner 训练细节

- **Teacher**: GPT-4V 在六维系统下生成少量高质量 caption
- **Student**: 基于 LLaVA fine-tune
- **关键 trick 1**: vision encoder 也 fine-tune，不是 frozen → caption accuracy 提升
- **关键 trick 2**: high-resolution vision encoder + **2x2 average pooling** 在 token 空间维度上 → 减少 token 数 4x，information loss 最小
- **关键 trick 3**: **mixed training**（static image + dynamic video）→ 缓解 video 训练数据不足

### 3.3 Caption 长度分布（Fig 4）

平均 202.1 words，分布大致 symmetric，峰值在 200 左右。vs Panda-70M 的 13.2 words，**信息密度提升 15x**。这个数字本身就解释了 VBench 上 semantic score 的飞跃（0.3093 → 0.5915）。

---

## 4. Method 3 — Video Training Suitability Score (VTSS)

这是 paper 最 novel 的部分。

### 4.1 核心论点

**论点 1**：sub-metrics 之间存在 joint distribution（Table 7）

| Sub-metric pair | Pearson | Spearman |
|---|---|---|
| (Clarity, Aesthetic) | 0.3774 | 0.3732 |
| (Clarity, Motion) | -0.4028 | -0.4324 |
| (Motion, Aesthetic) | -0.2515 | -0.2347 |

Clarity 和 Motion 负相关 (-0.40) 很符合直觉：fast motion 帧间 motion blur 多，clarity 自然下降。手动独立设 thresholds 忽略了这种耦合。

**论点 2**：error 会累积（Table 8）

| 偏差 +10% 的 sub-metrics 数量 | 误过滤数据 |
|---|---|
| 1 (Clarity) | 250K / 48M |
| 2 (Clarity, Aesthetic) | 290K / 48M |
| 3 (Clarity, Motion, Aesthetic) | 340K / 48M |

每多一个 threshold，误过滤 +40K，linear scaling。

### 4.2 New Annotation Criteria（三维度）

- **Dynamic Quality**: motion area > 30% frame area + camera temporal stability
- **Static Quality**: subject detail, composition, aesthetic, color saturation
- **Video Naturalness**: 无特效 / 字幕 / logo / 政治暴力内容

每个 video 由 **8 experts** 打分（1-5），然后做 bias correction：
- **Individual Preference Bias**（Fig 12a）：每个 expert 用 z-score normalize 再 rescale 到全局 mean/variance
- **Label Fluctuation Bias**（Fig 12b）：用 8 个 expert 的 mean

### 4.3 TSA Network 架构（Fig 6）

三路并行：

```
Video → ┬─→ 3D Swin Transformer → dynamic feat
        │
        ├─→ ConvNeXt → static feat
        │
        └─→ Sub-metrics (motion, aesthetic, clarity, ...) → label feat
                                                        │
                            Weight Cross-Gating Block (WCGB)
                                                        │
                                            Fusion → MLP → VTSS (scalar)
```

**WCGB** 是关键模块，intuition 类似 Flamingo 的 gated cross-attention：label feature 因为同时关联 dynamic 和 static（比如 motion score 影响 dynamic，aesthetic score 影响 static），所以用 learned weight $\alpha$ 来分配 label feature 进入两个分支的比例。

形式化（论文未给 explicit 公式，但可推断为）：

$$f_{dyn}' = f_{dyn} \odot \sigma(W_1 f_{label}), \quad f_{sta}' = f_{sta} \odot \sigma(W_2 f_{label})$$

其中 $\sigma$ 是 sigmoid gating，$W_1, W_2$ 是 learned projection。$\odot$ 是 element-wise product。

### 4.4 Ablation 结果（Table 5 & 6）

| Config | PLCC↑ | SRCC↑ | KRCC↑ | RMSE↓ |
|---|---|---|---|---|
| Dynamic only | 0.8684 | 0.8580 | 0.7027 | 0.4644 |
| + Static | 0.8730 | 0.8637 | 0.7111 | 0.4555 |
| + Feature | 0.8953 | 0.8864 | 0.7397 | 0.4203 |
| + WCGB | **0.8974** | **0.8868** | **0.7406** | **0.4099** |

对比 Dover / FastVQA（Table 6）：Dover PLCC 0.8554，Koala 0.8974，**+4.2 pts**。Dover 是 video quality assessment 的 SOTA，但 Koala 的 TSA 显式建模 sub-metrics joint distribution，所以更准确。

### 4.5 VTSS 阈值（Fig 7）

分布可看作两个 Gaussian 的 mixture，**2.5** 作为天然分解点。最终过滤得到 36M clips（从 48M 全集过滤掉 25%）。这个"硬阈值 + 软信息注入"的设计非常 clean——硬阈值保证 training set 干净，软信息（见下一节）保留 quality 维度。

---

## 5. Method 4 — Metric Conditions

这是把"过滤掉的 quality 信息"通过另一种方式 inject 回 model 的 trick。

### 5.1 Pipeline（Fig 8）

```
motion score ─┐
aesthetic ────┤
clarity ───────┼─→ Frequency Embedding ─→ MLP ─→ + to timestep emb
... ──────────┘                                          │
                                                          ↓
                                                Adaptive LayerNorm (AdaLN)
                                                          │
                                                          ↓
                                                Transformer block
```

### 5.2 实现 intuition

- **Frequency embedding**：和 Diffusion 中 timestep embedding 同款 sinusoidal embedding $\sin(\omega_k \cdot s), \cos(\omega_k \cdot s)$，其中 $s$ 是 normalized score, $\omega_k = 1/10000^{k/d}$
- **加到 timestep embedding**：让 model 把"quality 维度"和"noise level"看作同类 conditioning
- **AdaLN injection**：$\gamma, \beta = \mathrm{MLN}(c_t + c_{metric})$，然后 $h' = \gamma \cdot \mathrm{Norm}(h) + \beta$

### 5.3 vs OpenSora 的 text-condition 方法

OpenSora 把 metric 信息塞进 text prompt。Koala 的方法优势：
1. **数值敏感**：frequency embedding 是连续的，text prompt 是离散的
2. **解耦能力强**：可以单独把 motion 拉满而 aesthetic 保持，反之亦然（见 Fig 13 的 style transfer demo）
3. **不增加 FLOPs**：text encoder 还是只处理 text

---

## 6. Experiments 详解

### 6.1 Base Model 架构

- **Sora-like** 3D-full attention transformer
- 每个 block：2D self-attention + 3D self-attention + text cross-attention
- Text encoder：T5
- VAE：3D causal VAE
- From scratch 训练：256×256, 2s, batch 32, lr 1e-4
- 80× A100 80G
- 140M data samples pass-through（每个 sample 看一遍）

### 6.2 VBench 主结果（Table 2）

| Setting | Quality Score | Semantic Score | Total Score |
|---|---|---|---|
| Panda-70M | 0.7343 | 0.3093 | 0.6493 |
| Koala-w/o TSA (48M) | 0.7758 | 0.4668 | 0.7140 |
| Koala-37M-manual | 0.7704 | 0.4548 | 0.7073 |
| Koala-36M | 0.7819 | 0.4504 | 0.7156 |
| Koala-w/o TSA + condition | 0.7823 | 0.5874 | 0.7433 |
| **Koala-36M + condition** | **0.7846** | **0.5915** | **0.7460** |

几个关键 insights：

1. **Koala-w/o TSA > Panda-70M**（+0.065 Total）— 这隔离了 splitting + captioning 的贡献：仅靠更准确的 temporal split 和 200-word caption，不靠任何 filtering，已经全面超越 Panda。
2. **Koala-36M > Koala-37M-manual**（虽然只 +0.008 Total）— TSA 自动 filter 比手动 multi-threshold 更准，且保留更少 data（36M vs 37M）反而 quality 更高。
3. **+ Metric Condition 提升最大**：Koala-36M condition vs Koala-36M，semantic score 从 0.4504 → 0.5915，**+31%**。这说明把 quality signal 直接 inject 到 model，远比仅用作 filter signal 更有价值。
4. **Aesthetic Quality 单项飞跃**：Panda 0.3988 → Koala-36M cond 0.5318，**+33%**。这直接归功于 structured caption 里 "Visual Language" 维度。

### 6.3 附加实验（Table 9）

跨分辨率、跨时长对比 + 加入 HD-VG-130M baseline：

| Setting | Quality | Semantic | Total | FVD↓ |
|---|---|---|---|---|
| Panda 256-2s | 0.7343 | 0.3093 | 0.6493 | 570.87 |
| HD-VG-130M 256-2s | 0.7696 | 0.4541 | 0.7065 | 590.86 |
| Koala-36M cond 256-2s | 0.7846 | 0.5915 | 0.7460 | **549.79** |
| Panda 256-4s | 0.7395 | 0.4448 | 0.6806 | 451.09 |
| Koala-36M cond 256-4s | 0.7644 | 0.4646 | 0.7045 | **354.79** |
| Panda 512-2s | 0.7439 | 0.3954 | 0.6742 | 579.57 |
| Koala-36M cond 512-2s | 0.7849 | 0.6495 | 0.7578 | **392.26** |

**512 分辨率上 FVD 从 579.57 → 392.26，相对降低 32%**。这表明 data quality 的增益在高分辨率下放大——bad data 在低分辨率下还能"伪装"过去，高分辨率下 noise 会被放大。

---

## 7. 与相关工作的脉络

| Work | Scale | Avg Caption Length | Key Innovation |
|---|---|---|---|
| WebVid-10M | 10M | 12 | Alt-text (cheap, noisy) |
| HD-VILA-100M | 103M | 17.6 | ASR-based |
| VidGen-1M | 1M | 89.3 | Generated captions |
| MiraData | 330K | 318 | Structured captions (but small scale) |
| Panda-70M | 70M | 13.2 | Multi-teacher captioning |
| **Koala-36M** | **36M** | **202.1** | **Structured + VTSS + Metric Cond** |

Koala-36M 是目前**唯一同时满足 (a) >10M scale + (b) >200 words caption** 的 dataset。MiraData 虽然更详细，但 scale 不到 1%。

---

## 8. Limitations & 我的思考

作者自承：**36M 不足以训练 1B+ 参数的 video generation model**。这暗示 Kuaishou 内部可能有更大版本的 dataset（如果他们做 1B+ model 的话）。

我的几个 critical observation：

1. **Captioner 是 LLaVA-based**：意味着 caption 质量上限受 vision-language model 限制。如果未来用 GPT-4o / Gemini 1.5 Pro 级别的 captioner，质量还能再上一个台阶。

2. **TSA 的 human eval 成本**：200K videos × 8 experts = 1.6M annotations。这是 paper 没充分讨论的成本——reproducing 这部分需要相当大的标注投入。

3. **Metric Condition 在 inference 时的"作弊"风险**：训练时 metric 是 ground truth，inference 时需要 user 指定（或全部设 max）。Fig 13 显示设 max 时的确生成高质量视频，但 user 不一定知道每个 prompt 对应的最优 metric。

4. **没有 compared to MiraData**：MiraData 是 caption length 上限最高的工作（318 words），缺这个对比略遗憾。可能因为 MiraData 太小，作者认为不可比。

5. **CSS 的 limitation**：依赖 BGR histogram + SSIM，对低光 / 高 dynamic range 场景可能失效。这是一个值得 ablation 的点，但 paper 没展开。

---

## 9. 总结 Intuition

Koala-36M 的核心方法论可以浓缩为一句：

> **Data curation 是 implicit 的 regularization。每一个 noisy pair 都在拉低 model 的 effective capacity。**

四步 pipeline 对应四个 layer 的"noise filtering"：
- **Splitting**: 去除 temporal noise（transitions）
- **Captioning**: 增加 semantic precision（去除 caption ambiguity）
- **VTSS Filtering**: 去除 distribution-level noise（low-quality videos）
- **Metric Conditioning**: 把 quality signal 从 filter-only 转化为 filter + guide，让 model 知道每个 sample 应该被 learn 到什么程度

这个思路对未来 video diffusion 的研究有很强的指导意义——**scaling data 不是简单堆 video count，而是要在四个 orthogonal 维度上同时 scaling**。

---

## References / Web Links

- **Koala-36M Project Page**: https://koala36m.github.io/
- **Panda-70M (Snap Research)**: https://snap-research.github.io/Panda-70M/
- **MiraData**: https://github.com/xju-essentia/MiraData
- **OpenVid-1M**: https://github.com/OpenVDB/OpenVid-1M
- **VBench**: https://vchitect.github.io/VBench-project/
- **LLaVA**: https://llava-vl.github.io/
- **Stable Video Diffusion**: https://stability.ai/news/stable-video-diffusion-open-ai-video-model
- **PySceneDetect**: https://github.com/Breakthrough/PySceneDetect
- **OpenSora**: https://github.com/hpcaitech/Open-Sora
- **Sora (OpenAI)**: https://openai.com/sora
- **GPT-4V System Card**: https://openai.com/research/gpt-4v-system-card
- **ConvNeXt**: https://github.com/facebookresearch/ConvNeXt
- **3D Swin Transformer (Swin3D)**: https://github.com/microsoft/Swin-Transformer
- **DiT (Diffusion Transformer)**: https://www.wpeebles.com/DiT
- **AdaLN (Peebles & Xie, NeurIPS 2022)**: https://arxiv.org/abs/2212.09748
- **Dover Video Quality Assessment**: https://github.com/VQAssessment/DOVER
- **FastVQA**: https://github.com/VQAssessment/FAST-VQA-and-FasterVQA

如果你想要进一步 deep dive 某个 component（比如 TSA 网络的具体实现细节、CSS 在 long video 上的 failure cases、或者 metric condition 在不同 architecture 上的 transferability），随时告诉我。
