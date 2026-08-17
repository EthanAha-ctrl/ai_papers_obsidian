---
source_pdf: SketchAgent Language-Driven Sequential Sketch Generation.pdf
paper_sha256: 707c6fc2574fa3c054e08ac604bf2112ce2ac31a929fab750ebaffa4442e362c
processed_at: '2026-08-12T07:06:21-07:00'
target_folder: AI美工
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SketchAgent 人话版

Andrej，我换个说法，用最直白的话讲一遍。

---

## 一句话版本

**让 Claude 3.5 这种大语言模型直接"画画"，不用训练，不用 sketch dataset，就是 design 一套"画图语言"让它 output，然后 post-process 成图片。**

就这么简单。剩下全是 engineering details。

---

## 为什么这件事 surprising

你想，LLM 只能 output text token。它不能 output pixel，不能 output image。传统上让 LLM "画图"有两条路：

**路线 A**：LLM call DALL-E，DALL-E 生成 pixel image。问题：DALL-E 是 diffusion model，一次性生成整张图，没有"先画头再画身体再画腿"这种 sequential process。而且 pixel space 的图没法编辑、没法 collaborative。

**路线 B**：让 LLM 写 SVG code。问题：LLM 写出来的 SVG 长这样——一个 perfect circle，一个 perfect triangle，一个 perfect rectangle。看起来像 PowerPoint auto-shape，完全不像人画的 sketch。人类 sketch 的魅力在于那些 subtle 的 irregularity、wobble、spontaneity。

SketchAgent 走了**路线 C**：design 一种介于"code"和"自然 sketch"之间的中间语言。

---

## 核心 idea：三个 trick 的组合

### Trick 1: Numbered Grid

给 LLM 一张 50×50 的 grid image，每个格子标了号。x 轴 1-50，y 轴 1-50。LLM 想说"在这里画一笔"，就 output `x12y34` 这样的字符串。

为什么需要这个？因为作者发现一个 embarrassing 的事实（Fig. 6）：给 GPT-4o 看一张画了一半的房子图，上面标了 1-5 号点，问它"哪两个点连起来能补全房子"。模型答对了——点 1 和点 5。但让它用 pixel coordinate 去画这条线，它**反复失败**，试了好几次都不对。

这说明什么？**LLM 的 visual understanding 和 spatial action 是 disconnect 的**。它能"看懂"图，能"说对"答案，但让它输出精确 pixel coordinate，它做不到。

Solution：把 2D 空间变成 1D token 序列。`x12y34` 就是一个 token，LLM 用它擅长的 token manipulation 来处理空间。这跟 CoT 把 reasoning 显式化是同一个 philosophy——**把 implicit capability 强制成 explicit symbol operation**。

### Trick 2: Sample Points + t values

LLM output 一条 stroke 时，给出：
- 一串 grid coordinates：`x13y27, x24y27, x24y11, ...`
- 每个 coordinate 对应一个 t value：`0.00, 0.30, 0.50, ...`
- 一个 semantic label：`<id>house base front rectangle</id>`

然后 post-processing 把这些 points 拟合成 cubic Bézier curve。

为什么不直接让 LLM output SVG 的 Bézier control points？因为 Bézier curve 有个反直觉的特性：**control points 不在曲线上**。如果 LLM 想画一条经过 (5,5)、(10,15)、(15,5) 的曲线，它给出这 3 个点当 control points，结果曲线根本不经过这些点。LLM 没法 reason about "我给的控制点会让曲线偏离多远"——这需要逆向求解，太难了。

所以作者 flip 了思路：让 LLM 给"我希望曲线经过的点"，然后数学上反解 control points。这就是 least squares fitting。

### Trick 3: ICL + CoT

System prompt 教 LLM 这套 sketching language（怎么画 circle、triangle、rectangle 等基本形状）。User prompt 给一个完整的 house sketch 作为 example。

Ablation 显示：**没有完整 house example，performance 直接崩掉**（Top1 从 0.23 掉到 0.07）。LLM 需要"看到一个完整的 sketch 长什么样"才能 generalize 这个 representation。光看 single-stroke primitives 不够。

CoT 也很关键：让 LLM 在 `<thinking>` tag 里先 plan——"我要画一个 shark，先画 body curve，再画 dorsal fin，再画 tail flukes..."，然后再 output strokes。去掉 CoT，Top1 从 0.23 掉到 0.14。

---

## Bézier Fitting 的数学（人话版）

Cubic Bézier curve 你肯定熟，但我讲一下 paper 里的用法：

一条 cubic Bézier 由 4 个 control points 定义：$P_0$（起点）、$P_1$、$P_2$（中间控制点）、$P_3$（终点）。

$$B(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t) t^2 P_2 + t^3 P_3$$

- $t \in [0,1]$：parameter，从 0 走到 1，沿曲线从起点滑到终点
- $t=0$：$B(0) = P_0$，在起点
- $t=1$：$B(1) = P_3$，在终点
- $P_1, P_2$：控制曲率，但不在曲线上

LLM 给了 $m$ 个 sample points $(x_j, y_j)$ 和对应 $t_j$。把每个 sample 代入 Bézier 方程：

$$B(t_j) = (x_j, y_j)$$

展开成矩阵形式 $AP = B$：
- $A \in \mathbb{R}^{m \times 4}$：每行是 $t_j$ 处的 4 个 Bernstein basis 值 $[(1-t_j)^3, 3(1-t_j)^2 t_j, 3(1-t_j) t_j^2, t_j^3]$
- $P \in \mathbb{R}^{4 \times 2}$：4 个 control points 的 (x, y) 坐标，这是 unknown
- $B \in \mathbb{R}^{m \times 2}$：$m$ 个 sample points 的 (x, y) 坐标

Least squares 求解：$P^* = (A^T A)^{-1} A^T B$

直觉：LLM 说"我希望曲线经过这些点，在这个时间点经过"，数学帮你找到最合适的 control points 让曲线尽量贴合。就像你告诉 tailor "我要裤子在这里宽一点，那里窄一点"，tailor 帮你调出合适的版型。

**为什么 t values 重要？** 同样的 3 个 sample points，$t = [0, 0.5, 1.0]$ 拟合出对称拱形；$t = [0, 0.2, 1.0]$ 拟合出左陡右缓的曲线。LLM 控制 $t$ distribution 就控制了 curve parameterization——"曲线在这些点之间怎么分配时间"。

---

## Sequential Sketching：为什么这是 big deal

人类画画是 stroke by stroke 的。先画 body outline，再加 eye，再加 legs。每一笔都 carry semantic meaning。这是**hierarchical planning** 的结果——你脑子里先有整个 image 的 plan，然后一笔一笔 execute。

之前的方法（SVGDreamer、CLIPDraw 等）是 optimization-based：初始化一堆 random Bézier curves，用 CLIP 或 diffusion model 当 reward，gradient descent 迭代 2000 次。问题是：
- 慢：一张图 1.6 小时
- 中间步骤 meaningless：iteration 500 的 intermediate 就是一堆 random squiggles
- 没有 semantic structure：最终结果的某条 curve 你不知道它代表什么 body part

SketchAgent 因为是基于 LLM next-token prediction，**天然 sequential**。LLM output `s1`（body curve）→ `s2`（dorsal fin）→ `s3`（tail flukes）...每一笔都有 `<id>` 标注，有 semantic meaning，有 logical order。

Fig. 11 的定量分析很有说服力：对 4-7 strokes 的 sketch，SketchAgent 和 QuickDraw human sketches 的 CLIPScore 都随 stroke 累积递增——都是"越画越 recognizable"。这说明 sequential dynamics 在 distribution 上接近人类。

---

## Collaborative Sketching：怎么实现的

这个 feature 很 cool。流程：

1. LLM 生成到第 $j$ 条 stroke 遇到 stopping token `</s{j}>`，暂停
2. 渲染前 $j$ 条 stroke 到 canvas
3. Human 在 canvas 上画自己的 stroke
4. **Reverse fitting**：把 human 的 vector stroke 采样成 (x, y, t) tuples，snap 到最近的 grid cell center，转换成 LLM 的 format
5. 把 human strokes 拼接进 LLM 的 sequence
6. LLM 继续生成 next stroke

关键 design：human 和 agent 用**同一种 representation**。agent 能 "看到" human 画了什么，因为 human 的笔画也被转成了 `x12y34` 这种格式。

User study 结果：30 人，480 sketches。Collaborative sketch 的 CLIP recognition rate 是 0.75，接近 human solo（略高），远高于 agent solo。更有意思的是：只看 collaborative sketch 里 agent 的 strokes，recognition 只有 0.10；只看 human 的 strokes，0.13。**两边单独看都不 recognizable，合起来 0.75**——这是 emergence，不是 simple addition。

---

## 效果到底多好

### CLIP zero-shot classification（Table 1）

50 个 QuickDraw category，每个 10 张 sketch，共 500 张。

| Model | Top1 | Top5 |
|-------|------|------|
| GPT-4o-mini | 0.04 | 0.10 |
| Claude3-Opus | 0.13 | 0.27 |
| GPT-4o | 0.15 | 0.30 |
| **Claude3.5-Sonnet (SketchAgent)** | **0.23** | **0.44** |
| Claude3.5-Sonnet (direct SVG) | 0.23 | 0.43 |
| **Human (QuickDraw)** | **0.27** | **0.49** |

Claude3.5-Sonnet 接近人类水平，差距 4-5 个百分点。Direct SVG 的 recognition rate 和 SketchAgent 相当，但 2AFC study 显示 SketchAgent 更 "human-like"。

### 2AFC User Study

150 人参与，每次看两张同类别 sketch，选"哪个更像人画的"：
- SketchAgent vs direct SVG：**74.9%** 选 SketchAgent
- SketchAgent vs human：54.7% 选 human（几乎 50/50）
- Direct SVG vs human：38.9% 选 direct SVG

**人类几乎分不清 SketchAgent 和 human sketch**。这是比 CLIP score 更强的 claim。

---

## Limitations

Paper 诚实地承认了几个问题：

1. **复杂 concept 画不出**：unicorn 这种，LLM 能描述 horn、mane 等部分，但 sketch 出来 unrecognizable。Language → spatial action 的 gap 还在。
2. **Human figures 太 amateur**：画 Frida Kahlo 只能捕捉 eyebrows，画 Michael Jordan 只能捕捉 dunk pose，整体太简单。
3. **Letters/numbers 画不出**：很 ironic，LLM 本质是 text model 但画不出 text。
4. **Open-source model 效果差**：Llama-3.1-405B 的 Top1 只有 0.052。需要够强的 base model。
5. **50×50 grid 限制 precision**：更高 resolution 需要 bigger grid，但 bigger grid 让 LLM 的 spatial reasoning 更难——fundamental tension。

---

## 我的 take

这篇 paper 的核心 insight 不是技术上的 breakthrough，而是 **perspective shift**：不要急着 train 新模型，先看看 frozen foundation model 加上 clever prompting 能走多远。

SketchAgent 证明了一件事：multimodal LLM 在 pretraining 里**已经学到了 sketching 所需的 knowledge**——它知道 shark 有 dorsal fin，知道 house 有 roof 和 base rectangle，知道 hat 应该戴在 head 上。这些 knowledge 来自 internet-scale text-image pretraining。问题只是：怎么把这些 latent knowledge "逼"出来。

Solution 是 design 一套合适的 representation（numbered grid + sample points + t values + semantic ID）+ prompt strategy（system prompt + ICL example + CoT），让 LLM 用它擅长的 next-token prediction 来 generate sketch。

这跟 LLaVA、Visual ChatGPT、Chain-of-Thought 是同一套哲学：**用 prompting 挖掘 latent capability**，而非用 training 注入新 capability。

如果你 2017 年告诉我，未来有一天我们能用纯 prompting 让一个 text model 画出接近人类水平的 sketch，我大概不会信。但 2024 年这就是现实。Software 2.0 的 capability 真的被 underestimate 了。

---

## References

- Project page: https://sketch-agent.csail.mit.edu/
- SketchRNN: https://arxiv.org/abs/1704.03477
- CLIPDraw: https://arxiv.org/abs/2106.07844
- SVGDreamer: https://arxiv.org/abs/2312.16376
- CLIPasso: https://arxiv.org/abs/2206.04837
- Visual Sketchpad: https://arxiv.org/abs/2406.09403
- Min et al. "Rethinking ICL": https://arxiv.org/abs/2202.12837
- Carroll et al. "Learning about humans for coordination": https://proceedings.neurips.cc/paper/2019/hash/700d9e252b5e0ce39f515fac0f9b77f0-Abstract.html

---

# SketchAgent: Language-Driven Sequential Sketch Generation 深度解读

Andrej, 这篇 paper 我读下来感觉是一个**非常 elegant 的 system paper**——它的核心贡献其实不在于发明新算法，而在于用一种很 surgical 的方式把 multimodal LLM 已经具备的 prior knowledge "解锁"出来用于 sketching 任务。下面我从多个 angle 切入，重点放在 build intuition 上。

---

## 1. 这篇 paper 在 sketch generation 谱系里的位置

先理清 sketch generation 的历史脉络，方便直觉性地理解 SketchAgent 在哪：

- **第一代：filter-based**。Canny edge detector、XDoG 这类，本质是 image processing，没有"理解"。
- **第二代：data-driven generative**。SketchRNN (Ha & Eck, 2017) 是里程碑，用 RNN 在 QuickDraw 上训练，sequential sketch generation 的开山之作。问题是只能生成 345 个 QuickDraw 类别里的东西，diversity 受限。
- **第三代：CLIP/diffusion-guided optimization**。CLIPDraw (Frans et al. 2022), VectorFusion, DiffSketcher, SVGDreamer 这些方法用 pretrained vision-language model 当 reward signal，通过 differentiable rasterizer 优化 Bezier curve 参数。优点：能画任意 concept；缺点：all strokes 同时优化，每一步 intermediate 都是 noise，没有 sequential semantic structure，而且慢（5 分钟到 1 小时一张）。
- **第四代（SketchAgent）：直接用 multimodal LLM 当 sketcher**。把"画图"这件事重新 cast 成"输出一种特殊的 sketching language"，让 LLM token-by-token 自回归地生成。

关键 insight 是：**LLM 的 next-token prediction 本身就是 sequential 的**，这一点天然契合人类 stroke-by-stroke 的绘画过程。这是 paper 的 first principle。

参考链接：
- SketchRNN: https://arxiv.org/abs/1704.03477
- CLIPDraw: https://arxiv.org/abs/2106.07844  
- SVGDreamer: https://arxiv.org/abs/2312.16376
- QuickDraw dataset: https://github.com/googlecreativelab/quickdraw-dataset

---

## 2. 核心架构：为什么 LLM 能"画图"

### 2.1 Multimodal LLM 的 output bottleneck

Multimodal LLM（GPT-4o, Claude3.5-Sonnet, Gemini, LLaVA 等）的 input 是 (text, image)，但 output 只能是 text。这个 asymmetry 是一个 fundamental constraint。要让它产 visual 内容，传统上有两条路：

1. **External tool**：LLM call DALL-E 3 / Stable Diffusion。但 diffusion 是 pixel-space one-shot，没有 sequential 过程。
2. **Code generation**：让 LLM 输出 Python / SVG / TikZ / Processing code，再渲染。问题是 LLM 输出的 SVG 往往是 rigid geometric shapes（圆、三角、矩形），看起来很 mechanical，缺少 freehand sketch 那种 subtle irregularity 和 spontaneity（见 paper Fig. 3B）。

SketchAgent 提出第三条路：**设计一种介于 SVG 和自然 sketch 之间的中间表示**。

### 2.2 Numbered grid canvas：spatial reasoning 的 hack

Paper Section 4 提到一个非常重要的观察（Fig. 6）：给 GPT-4o 一张简单线条画，上面标了 1-5 的点，问它哪些点连起来能补全房子。模型能正确识别是点 1 和点 5，但让它用 pixel coordinates (draw line API) 去连，它**反复失败**。这说明 multimodal LLM 的 visual reasoning ≠ spatial action reasoning。

他们的解决方案是 number-labeled 50×50 grid：
- x 轴 1-50，y 轴 1-50
- 每个 cell 唯一标识：`x12y34`
- LLM 直接 output 字符串 `x12y34` 来 reference 位置

这个 trick 背后的 intuition：**LLM 在 text space 里做 spatial reasoning 比在 implicit coordinate space 里强得多**。把 2D 坐标变成一维 token 序列，让 LLM 用它擅长的 token manipulation 来处理位置。这和 Chain-of-Thought 把 reasoning 显式化是同一类哲学——把 implicit capability 强制成 explicit symbol manipulation。

### 2.3 整体 pipeline（对应 Fig. 5）

```
[Input]
  ├── System Prompt（introduce sketching language + 单 stroke primitives）
  ├── User Prompt（task description + house example as ICL）
  └── Blank numbered canvas (image)

        ↓
   Frozen Multimodal LLM (Claude3.5-Sonnet)
        ↓
[Output: text]
  <thinking>...</thinking>  ← Chain-of-Thought: 规划 sketching strategy
  <strokes>
    <s1>
      <points>x13y27, x24y27, ...</points>
      <t_values>0.00, 0.30, ...</t_values>
      <id>house base front rectangle</id>
    </s1>
    ...
  </strokes>
        ↓
   Bezier curve fitting (least squares)
        ↓
   SVG → CairoSVG render → pixel canvas
        ↓
[Reuse]
  ├── Feed back to LLM (chat-based editing)
  └── Human draws on it (collaborative sketching)
```

关键点：**整个 pipeline 是 training-free 的**。所有"知识"都来自 LLM 的 pretraining，SketchAgent 只是设计 prompt + post-processing。这是我个人觉得最 surprising 的地方——你不需要 sketch dataset，不需要 fine-tune，只是 cleverly prompt + cleverly decode。

---

## 3. Sketch Representation：Bezier curve fitting 的数学

这是 paper 里我最喜欢的一部分，技术含量很高。

### 3.1 Cubic Bezier curve 回顾

Cubic Bezier curve 由 4 个 control points 定义：$P_0$ (start), $P_1, P_2$ (control), $P_3$ (end)。曲线方程：

$$B(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t) t^2 P_2 + t^3 P_3, \quad t \in [0,1]$$

变量含义：
- $t$：curve parameter，从 0 到 1 沿曲线移动。$t=0$ 时在 $P_0$，$t=1$ 时在 $P_3$
- $P_0, P_3$：端点，曲线必经
- $P_1, P_2$：control points，**不**在曲线上，但拉拽它们决定曲率
- $(1-t)^3, 3(1-t)^2t, 3(1-t)t^2, t^3$：Bernstein polynomial basis，4 个 weights 加起来恒等于 1（这是 affine invariance 的来源）

直觉：想象一根橡皮筋两端固定在 $P_0, P_3$，你用两根手指（$P_1, P_2$）去"拽"它，拽的位置和力度决定弯曲程度。这就是为什么 Bezier 这么适合 sketching——它天然捕捉"用少量控制点产生平滑曲线"的人类笔触特征。

### 3.2 三个 design choice 的对比（Fig. 7）

SketchAgent 处理 LLM 输出的 coordinate 序列时，作者对比了 3 种方案：

**Option A: Polyline**  
直接把 consecutive points 用直线连起来。问题：50×50 grid 太 sparse，导致锯齿状，不像人类 sketch。

**Option B: 把 coordinates 当 Bezier control points**  
理论上 elegant，但有一个根本问题：control points **不在曲线上**。如果 LLM 想画一条经过 (5,5), (10,15), (15,5) 的曲线，它直接给出这 3 个点当 control points，结果曲线根本不经过这些点。LLM 没办法 reason about "我给的控制点会让曲线偏离多远"——这需要逆向求解 control points from desired samples，对 LLM 来说太难。

**Option C (作者的选择): Curve fitting**  
让 LLM 输出"我希望曲线经过的 sample points"+ 每个 point 对应的 $t$ value，然后用 least squares 反解 control points。

### 3.3 数学形式化

对于 stroke $S_i$，LLM 输出：
- $m$ 个 sample points: $S_i = \{(x_j, y_j)\}_{j=1}^m$
- $m$ 个对应 $t$ values: $T_i = \{t_j\}_{j=1}^m$, $t_j \in [0,1]$

把每个 sample point 看作 $B(t_j) = (x_j, y_j)$，展开 cubic Bezier：

$$
\underbrace{\begin{bmatrix} (1-t_1)^3 & 3(1-t_1)^2 t_1 & 3(1-t_1) t_1^2 & t_1^3 \\ (1-t_2)^3 & 3(1-t_2)^2 t_2 & 3(1-t_2) t_2^2 & t_2^3 \\ \vdots & \vdots & \vdots & \vdots \\ (1-t_m)^3 & 3(1-t_m)^2 t_m & 3(1-t_m) t_m^2 & t_m^3 \end{bmatrix}}_{A \in \mathbb{R}^{m \times 4}}
\underbrace{\begin{bmatrix} P_0 \\ P_1 \\ P_2 \\ P_3 \end{bmatrix}}_{P \in \mathbb{R}^{4 \times 2}}
=
\underbrace{\begin{bmatrix} x_1 & y_1 \\ x_2 & y_2 \\ \vdots & \vdots \\ x_m & y_m \end{bmatrix}}_{B \in \mathbb{R}^{m \times 2}}
$$

每个 $t_j$ 给一行，Bernstein basis 在 $t_j$ 处求值。这是 over-determined linear system（$m > 4$），用 least squares：

$$P^* = \arg\min_P \|AP - B\|_F^2$$

闭式解：$P^* = (A^T A)^{-1} A^T B$。计算上很便宜（4×4 矩阵求逆）。

变量上下标的语义：
- $A_{j,k}$：第 $j$ 个 sample 在第 $k$ 个 Bernstein basis 上的值，$k \in \{0,1,2,3\}$ 对应 $P_0, P_1, P_2, P_3$
- $B_{j,:}$：第 $j$ 个 sample 的 (x, y) 坐标
- $P_{k,:}$：第 $k$ 个 control point 的 (x, y) 坐标
- $\|\cdot\|_F$：Frobenius norm，等价于把所有元素的 squared error 求和

### 3.4 为什么需要 LLM 给 $t$ values？

这是设计上很 subtle 的一点。光给 sample points 不够——你需要知道每个 sample 在曲线上对应哪个"位置"。

举例：LLM 给 `x5y5, x10y20, x15y5`，如果均匀采样 $t = 0, 0.5, 1.0$，会拟合出对称的拱形；如果 $t = 0, 0.2, 1.0$，会拟合出左陡右缓的曲线。**同样的 sample points，不同的 $t$ 分配产生完全不同的 curve**。

让 LLM 显式 output $t$ values，相当于让它控制 curve parameterization。这把 spatial reasoning 任务降维成"我希望曲线在这几个点之间怎么分布时间"——更接近 LLM 擅长的 symbolic reasoning。

### 3.5 Recursive splitting

如果 single cubic Bezier 拟合 error 太大（sequence 太长，复杂 curvature），就递归切分成多段 cubic Bezier。这是 standard CAD 技术（参考 Piegl & Tiller 的 The NURBS Book）。作者还支持 quadratic Bezier、linear line、single point 作为退化情况，保证 representation 的完备性。

### 3.6 为什么不用更高阶 Bezier？

直觉上可能想问：为什么不用 quintic 或 degree-7 Bezier？答案是**高阶 Bezier 数值不稳定**（Runge's phenomenon 类似问题），且 control points 影响全局，不 local。Cubic 是"smooth enough + local control + numerically stable"的 sweet spot。这也是 SVG 和 PostScript 都用 cubic 的原因。

---

## 4. Prompt 设计：In-Context Learning 的精细工程

### 4.1 System Prompt（Fig. 51）

固定 prompt，包含：
- Identity: "You are an expert artist specializing in drawing sketches..."
- Grid 说明：x 轴 1-50, y 轴 1-50, `x{i}y{j}` format
- 单 stroke primitives 示例（ellipse, circle, V-shape, triangle, rectangle, square, dot, line）
- 关键 instruction："If you want to draw a big and long stroke, split it into multiple small curves that connect to each other"
- "Think step-by-step" 触发 CoT

我注意到一个 design 细节：对于非 smooth shape（带 corner 的 triangle / rectangle），prompt 教模型用**重复点**模拟 corner——比如 V 形用 `['x13y27', 'x18y37', 'x18y37', 'x24y27']` + `[0.00, 0.55, 0.5, 1.00]`。重复点 + 相邻 t 值造成曲线在该点 "stop and turn"，制造 corner。这是 Bezier 表达 sharp corner 的 standard trick。

### 4.2 User Prompt + ICL Example

User prompt 包含 task（"draw a shark"）和一个完整的 house sketch 作为 ICL example（Fig. 53/54/55）。

Ablation study（Table 2）显示：
- Full SketchAgent: Top1 = 0.23, Top5 = 0.43
- w/o System Prompt: 0.20, 0.42（轻微下降）
- w/o CoT: 0.14, 0.29（显著下降，CoT 关键）
- Modified ICL（只给单 stroke example 不给完整 house）: 0.07, 0.16（**崩溃**）

ICL example 的完整性是 performance 的最大 contributor。这说明模型需要"看到一个完整的 sketch 长什么样"才能 generalize 这个 sketching language。光看 single-stroke primitives 不足以理解整个 representation。

### 4.3 Ablation on ICL theme（Fig. 49, 50）

作者还做了一个有趣的 ablation：把 house example 换成 cat example（同 stroke 数）。
- Animal 概念：用 cat example 时 sketching 风格更"动物化"（眼睛更像猫眼）
- Structure 概念：用 cat example 时形状更圆润，用 house example 更"建筑感"
- 复杂度：如果 ICL example 太详细（cat 加了很多细节），output 会 overfit，复制 cat 而不是 generate 新的

这是 ICL 经典的 demonstration effect：**example 的 distribution 强烈 shape 了 output distribution**。和 Min et al. 2022 "Rethinking the Role of Demonstrations" 的发现一致——ICL 的 format/pattern 比 content 更重要。

参考：https://arxiv.org/abs/2202.12837

---

## 5. Sequential Nature：核心 novelty

### 5.1 为什么 sequential 重要

人类 sketching 是 progressive 的：先画 body outline，再加 eyes，再加 legs。每一笔都 carry semantic meaning，是**hierarchical planning** 的结果。这跟 SketchRNN 学到的人类数据分布一致。

传统 optimization-based 方法（SVGDreamer 等）把所有 strokes 当作 simultaneously optimizable parameters，从 random init 开始梯度下降。中间步骤 meaningless，最后才"涌现"出 recognizable shape。这从 cognitive/expressive 角度看是错的——它丢失了 sketching 作为"thinking process"的本质。

### 5.2 Sequential sketching 的定量分析（Fig. 11）

作者做了 500 sketches 的统计：
- QuickDraw sketches 的 stroke 数分布：1-6 strokes 为主，single-stroke 占比高（人类 abstract sketching）
- SketchAgent sketches：5-10 strokes 为主，更"详细"

为了 fair comparison，他们筛选 stroke 数 4-7 的子集（204 SketchAgent + 120 QuickDraw），plot CLIPScore 随 stroke 数累积的曲线：

- 两者都呈**递增 pattern**：早期 strokes 给出 rough shape，后续 strokes refine，识别度逐渐提高
- 形状相似，说明 SketchAgent 的 sequential dynamics 在 distribution 上接近人类

### 5.3 Stroke annotation（Fig. 9, 38, 39）

每个 stroke LLM 都会自然 produce 一个 `<id>` tag，比如 "main body curve", "dorsal fin", "tail flukes", "pectoral fin", "eye"（dolphin 例子）。这是 LLM prior knowledge 的副产品——它在生成 coordinates 的同时生成 semantic label，因为它的 language model 训练让它倾向于"解释自己在做什么"。

这个 side effect 很有价值，对应到 cognitive science 里的"sketch understanding"和"part-based segmentation"任务（CreativeSeg, ContextSeg 等）。Paper 把它当作 data collection 工具的潜力提了一下。

---

## 6. Collaborative Sketching：交互机制的细节

### 6.1 Stopping token 设计

定义一个 adjustable stopping token `</s{j}>`，告诉 LLM "生成到第 $j$ 条 stroke 就停"。然后：
1. 渲染前 $j$ 条 stroke 到 canvas
2. Human 在 canvas 上画自己的 stroke
3. **Reverse fitting**：把 human 的 vector stroke 采样成 (x, y, t) tuple，snap 到最近的 grid cell center
4. 把 human strokes 插入 agent 的 sequence，agent 继续生成 next stroke

这个 reverse fitting 是很 elegant 的设计——保证了 human 和 agent 用同一种 representation，agent 能 "看到" human 画了什么。这避免了 human 输入和 agent 输入用不同 modality 导致的 alignment 问题。

### 6.2 User Study 结果（Fig. 12, Table 3）

30 个 user，480 sketches，8 concepts（butterfly, fish, rabbit, duck, sailboat, coffee mug, eyeglasses, car）。

Recognition rates：
- Collab full: 0.75 (95% CI: [0.61, 0.85])
- Solo user: 略高
- Solo agent: 较低
- Collab agent-only strokes: 0.10
- Collab user-only strokes: 0.13

最后两个数字非常 informative：**单独看任何一方的 strokes 都几乎 unrecognizable**，但合起来达到 0.75。这说明 collaboration 不是 simple addition，而是 emergent 的——双方都基于对方的 strokes 做出 contextual 决策。

平均 stroke count：collab full 7.33, solo agent 7.32, solo user 7.71——很接近，说明 detail level 没有显著差异。

### 6.3 Failure modes

14/240 (~6%) collaborative sketches 出现"双方理解不一致"的问题——比如两边都画了一个 head（Fig. 45）。这其实揭示了一个深层问题：collaboration 需要 **theory of mind**（agent 要推断 human 的意图）。当前 SketchAgent 没有显式 model 对方 mental state，只是把 human strokes 当 input 继续 generate。

这跟 Carroll et al. 2019 "On the utility of learning about humans for human-AI coordination" 的 finding 呼应——cooperation 需要 model 对方。后续工作方向可能是显式让 agent 推断 "human wants to draw X, so I should draw Y"。

参考：https://proceedings.neurips.cc/paper/2019/hash/700d9e252b5e0ce39f515fac0f9b77f0-Abstract.html

---

## 7. Chat-based Editing

Section 5.4 演示 chat 编辑能力。Edit prompt 模板：

> `<editing instruction>. Describe the location of the added concepts first in <thinking> tags. Only provide the added strokes. Respond in the same format as before. Be concise.`

实验设计三类 edits：
- **Animals**：加 glasses/hat/skirt（inferential，需要 agent 推断 placement，比如 hat 在 head 上）
- **Outdoor**：加 tree (left of), sun (top right), smaller object (right of)（spatial relations）
- **Indoor**：加 coffee mug/lamp/plant（top / left of）

54 sketches 量化结果：
- Overall 92% instruction-following
- Specified relations: 94%
- Inferred semantic relations: 88%

这是 spatial reasoning 的真实 test。"Hat on head" 需要 agent 先识别 sketch 里的 head 在哪，然后 spatially 放置 hat。LLM 在 numbered grid + canvas image input 的双重 grounding 下能做到 ~88%，相当不错。

参考 Sharma et al. 2024 "A Vision Check-up for Language Models"——LLM 的 spatial reasoning 是个已知瓶颈。

---

## 8. 量化结果深度解读

### 8.1 CLIP zero-shot classification（Table 1）

50 QuickDraw categories × 10 sketches = 500 sketches。用 `clip-vit-large-patch14` 做 zero-shot classification。

| Model | Top1 | Top5 |
|-------|------|------|
| GPT-4o | 0.15±0.04 | 0.30±0.06 |
| GPT-4o-mini | 0.04±0.03 | 0.10±0.04 |
| Claude3-Opus | 0.13±0.04 | 0.27±0.05 |
| **Claude3.5-Sonnet (SketchAgent)** | **0.23±0.05** | **0.44±0.03** |
| Claude3.5-Sonnet (direct SVG) | 0.23±0.04 | 0.43±0.06 |
| Human (QuickDraw) | 0.27±0.07 | 0.49±0.06 |

几个 observation：
1. **Claude3.5-Sonnet 显著好于其他 LLMs**。GPT-4o-mini 几乎完全 fail，说明 sketching 需要很强的 base model capability。
2. **Direct SVG 与 SketchAgent recognition rate 相当**。但 2AFC study 显示 SketchAgent 更"human-like"——这说明 CLIP score 衡量的是 semantic recognizability，而 human-likeness 是更 subjective 的 quality。
3. **Claude3.5-Sonnet 接近 human（0.23 vs 0.27 Top1，0.44 vs 0.49 Top5）**。差距大约 4-5 个百分点，已经相当接近。

### 8.2 Confusion matrix（Fig. 21）

最常见的 confusions：
- shark → fish（visual overlapping class）
- octopus → spider（多腿生物）
- snake → squiggle（abstract shape）

这些 confusion 都在 **semantically related classes** 内部，说明 SketchAgent 抓住了大类特征但 lacking distinctive feature 强调。这和 abstract sketching 风格一致——人类 abstract sketch 也会有这种 ambiguity。

### 8.3 2AFC User Study

150 MTurk workers，每对 sketch 二选一"哪个更像人画的"：
- SketchAgent vs direct prompting: **74.90 ± 3.35%** 选 SketchAgent
- SketchAgent vs human: 54.68 ± 4.61% 选 human（接近 50/50）
- Direct prompting vs human: 38.9 ± 5.55% 选 direct（人类压倒性赢）

这意味着 SketchAgent 已经在 perceptual discrimination 上**模糊了 human/machine boundary**——用户 nearly can't tell。这比 recognition rate 更 strong 的 claim，因为 recognition 是 machine metric，2AFC 是 human perceptual metric。

### 8.4 Open-source model 实验

Llama-3.2-11B-Vision：基本 fail，经常复制 ICL example（house）。
Llama-3.1-405B-Instruct：能用，但 Top1 只有 0.052，Top5 0.10。

这说明 SketchAgent 方法本身不 tied to Claude，但需要足够强的 base model。和 ICL scaling laws 一致——capability threshold 大约在 100B+ params with strong vision-language pretraining。

---

## 9. Limitations 和我的思考

### 9.1 Paper 自己承认的 limitations

1. **复杂 concept（unicorn）抽象难辨**：LLM 能 richly describe parts（horn, mane），但无法 convert 到 effective sketching actions。这是 language → spatial action 的 gap。
2. **Human figures（Frida Kahlo, Michael Jordan）**：distinctive features 在 language space captured，sketch output 太简单 amateur。
3. **Letters / numbers**：基本画不出来。这很 ironic，因为 LLM 本质是 text model。

### 9.2 我自己的观察和延伸思考

**Vision capability bottleneck**：作者把这个归因于 backbone model 的 vision 优化不足。我觉得更深层的原因是 LLM 的训练 objective（next text token prediction）和 sketch generation 的 objective（produce spatial coordinates that render to recognizable image）之间存在 **objective misalignment**。即使 vision capability 提升，LLM 仍然是 text-output-only，需要这种 indirect language → coordinate → render 的路径。真正解决可能需要 native multimodal output token 的训练（image tokens + text tokens intermixed），这是 GPT-4o 的 omni model 方向。

**Spatial reasoning 的根本局限**：numbered grid 是个聪明的 workaround，但 50×50 = 2500 cells 的 resolution 限制了 sketching precision。如果要画高细节，需要更大的 grid，但更大 grid 让 LLM 的 spatial reasoning 变难（数字更大、距离更难估计）。这是 fundamental tension。

**缺乏 learning signal**：因为是 training-free，模型不会从自己的 sketching mistake 中学习。对比 SketchRNN，它在 human sketch 上 supervised learning，能捕捉更 fine-grained 的 stroke distribution。一个 hybrid 方案可能是用 SketchAgent 生成大量 sketches，再蒸馏回一个 dedicated sketch model。

**Single sketch per turn**：当前不能一次性生成 scene-level multi-object sketches with consistent spatial layout。CLIPasso/CLIPascene 走的是 scene-level 路线，但用 optimization。如何把 SketchAgent 扩展到 scene 是 open question。

**Reference & inspiration from "Visual Sketchpad" (Hu et al. 2024)**：这个 concurrent work 让 LLM 在 reasoning 时画 sketch 辅助 thinking。和 SketchAgent 互补——一个用 sketching 当 output，一个当 intermediate reasoning tool。结合起来可能很有意思。

参考：https://arxiv.org/abs/2406.09403

**和 Andrej 你的工作的联想**：
- 你在 nanoGPT / makemore 里强调 next-token prediction 是简单但 powerful 的 paradigm。SketchAgent 把这个理念扩展到 sketching domain——sketch 就是另一种"语言"，next stroke = next token。
- 你讲过的 "Software 2.0" 概念：用 learned model 替代 hand-written heuristics。SketchAgent 走的是相反方向——用 hand-designed representation (sketching language) + frozen model 来避免训练。这是 "Software 1.5"——cleverly prompt 已经 trained 好的 Software 2.0 系统。
- 你在 CS231n 讲过 human visual cognition 是 hierarchical 的（edge → part → object）。SketchAgent 的 sequential + semantically annotated strokes 印证了这个 hierarchy 在 LLM prior 中已经存在。

---

## 10. 总结：这篇 paper 的真正 contribution

如果让我一句话概括：**SketchAgent 证明了 multimodal LLM 已经在 pretraining 里学到了足够的"sketching knowledge"，只需要一个合适的 representation 和 prompt strategy 就能解锁**。

这给整个 generative model 社区一个重要启示：**在投入大量 compute 训练 task-specific model 之前，先想想 frozen foundation model 加上 clever prompting 能走多远**。这跟 LLaVA、Visual ChatGPT、Chain-of-Thought 一脉相承，都是"用 prompting 挖掘 latent capability"的范例。

具体的技术 contribution：
1. Numbered grid canvas（解决 spatial reasoning bottleneck）
2. Sample-points + t-values 的 sketching language（让 LLM 输出可解析为 Bezier 的 representation）
3. Least-squares Bezier fitting post-processing（smooth out grid sparsity）
4. Stopping token + reverse fitting（enable human-agent collab）
5. ICL + CoT 的精细 prompt engineering

未来方向我会关注：
- Native multimodal output model（GPT-5 时代可能直接输出 sketch tokens）
- Sketch + language joint reasoning（"先画再描述" vs "先描述再画"）
- Agent 用 sketching 当 thinking tool（Visual Sketchpad 方向）
- 3D sketching（3Doodle 已经在 ACM TOG 2024 探索）
- Sketch-based design tools（Figma plugin 之类的 real-world deployment）

---

## Reference 链接汇总

- **Project page**: https://sketch-agent.csail.mit.edu/
- **SketchRNN paper**: https://arxiv.org/abs/1704.03477
- **QuickDraw dataset**: https://github.com/googlecreativelab/quickdraw-dataset
- **CLIPDraw**: https://arxiv.org/abs/2106.07844
- **SVGDreamer**: https://arxiv.org/abs/2312.16376  
- **DiffSketcher**: https://arxiv.org/abs/2306.09364
- **CLIPasso (Vinker et al. 2022)**: https://arxiv.org/abs/2206.04837
- **CLIPascene**: https://arxiv.org/abs/2310.07323
- **Visual Sketchpad**: https://arxiv.org/abs/2406.09403
- **A Vision Check-up for LMs (Sharma et al.)**: https://arxiv.org/abs/2401.04562
- **Min et al. "Rethinking ICL"**: https://arxiv.org/abs/2202.12837
- **Carroll et al. "On the utility of learning about humans"**: https://proceedings.neurips.cc/paper/2019/hash/700d9e252b5e0ce39f515fac0f9b77f0-Abstract.html
- **Differentiable rasterizer (Li et al. 2020)**: https://arxiv.org/abs/2003.10591
- **3Doodle**: https://arxiv.org/abs/2405.05912
- **Claude 3.5 Sonnet**: https://www.anthropic.com/claude
- **CairoSVG (renderer)**: https://cairosvg.org/
- **Differentiable Vector Graphics Rasterization**: https://arxiv.org/abs/2003.10591
- **OpenSketch dataset**: https://arxiv.org/abs/1909.09148
- **Fan et al. "Drawing as versatile cognitive tool"**: https://www.nature.com/articles/s44159-023-00212-w

如果你对某一部分（比如 Bezier fitting 的 numerical 细节、ICL 的 mechanism、collaboration 的 cognitive science 框架）想 deeper dive，告诉我，我可以再展开。
