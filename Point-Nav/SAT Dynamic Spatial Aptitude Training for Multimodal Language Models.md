---
source_pdf: SAT Dynamic Spatial Aptitude Training for Multimodal Language Models.pdf
paper_sha256: 363d912de76f6737ec5915bde761c2360aca119f25086cd79e5a15e7a4d4f792
processed_at: '2026-08-12T03:08:51-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SAT 用人话说一遍

Andrej，我换一种讲法，把那些学术包装剥掉，直接说这篇 paper 到底干了啥、为啥 work。

---

## 一句话说清楚

现在的多模态模型（LLaVA、GPT-4o、Gemini）看图片，能认出"这是猫那是狗"，但你问它"如果你往左走三步再转身，猫会在你左边还是右边"——它基本靠猜。SAT 的做法很简单：**在 3D 模拟器里手动操控相机和物体，记录下来动作前后空间关系怎么变，生成 17.5 万道这样的题，拿去训模型**。结果发现训完之后模型不光会做这种"动态空间题"，连普通的空间题（谁在谁左边、谁离谁近）也一起变好了。

---

## 为啥要做这件事

先说 background。你拿 LLaVA-1.5-13B 去测 CVBench 上的 2D 关系题（"酒瓶在盘子左边还是右边"），准确率只有 46.6%——**比扔硬币还差**。GPT-4o 好一点但也就 85.7%。更惨的是动态题：让 Gemini-1.5-pro 判断"如果我从 X 点朝左看，冲浪的人在左还是右"，它答 49.5%，**基本随机**。

问题出在哪？训练数据里几乎没有 spatial 标注。互联网图片的 caption 不会写"杯子在桌子的左前方 0.3 米"。之前有人想用 depth estimation model 给真实图片打 pseudo-label，但 depth model 本身就有误差，bounding box 也不准（一个湖的一部分在树前面、大部分在树后面，你用 bbox center 算关系就是错的）。

SAT 的 bet 是：**与其用 noisy 的真实数据，不如用 perfect 的模拟数据**。反正空间关系是几何规则，跟 texture 没关系，sim-to-real 应该能 transfer。

---

## 具体怎么造数据

### 场景

用 ProcTHOR-10K，就是 AI2 那个程序化生成的室内公寓场景。22K 个房间，1K 种家具物体。相机放哪儿？随机选 20 个位置，挑能看到最多物体的那个——保证画面信息密度高。

### 坐标系（这个是核心）

模拟器里每个物体有 3D 坐标 (x, y, z)。但你要判断"左还是右"，得先转到相机视角。paper 里这套归一化我拆开讲：

相机站在 (x₀, y₀, z₀)，朝某个方向看。先把所有物体平移，让相机变成原点：

$$x_{\text{shifted}} = x - x_0$$
$$z_{\text{shifted}} = z - z_0$$

- $x, y, z$ 是物体在世界坐标系的位置
- $x_0, y_0, z_0$ 是相机的世界坐标
- 减完之后相机就在 (0, 0, 0) 了

然后旋转，让相机朝 +z 方向（这样 left/right/in-front/behind 就变成了简单的正负号判断）：

$$\begin{bmatrix} x' \\ z' \end{bmatrix} = \begin{bmatrix} \cos(a) & -\sin(a) \\ \sin(a) & \cos(a) \end{bmatrix} \begin{bmatrix} x - x_0 \\ z - z_0 \end{bmatrix}$$

- $a$ 是相机绕 y 轴顺时针转的角度
- $x'$ 算完是负数 → 物体在相机左边；正数 → 右边
- $z'$ 算完是正数 → 物体在相机前方；越大越远
- $y$（高度）不动，因为相机高度不变

这样一来，"酒瓶在盘子左边吗"就变成：**酒瓶的 x' < 盘子的 x' 吗**。一行代码搞定，0 噪声。

### 五类动态题

#### 1. Egocentric Movement（6.9K 题）

模拟器里随机做一个动作：往左转 30°、往右转 50°、往前走 30cm 之类的。拍下动作前和动作后两帧。然后问模型："相机怎么转的？"

关键 trick：**转动和平移是独立的**。相机可以一边往左平移一边往右转。模型得从两张图的光流变化反推出动作序列。

直觉上这就是 embodied AI 的核心能力——你看到一段第一人称视频，得判断拍摄者怎么走的。

#### 2. Object Movement（6.9K 题）

随机挑一个物体，往随机方向移 0.25-0.5 米。问："两帧之间有物体动了吗？怎么动的？"

坑在哪：**有时候相机也动了**。模型得区分"是相机动了所以所有物体看起来都移动了" vs "是某个物体真的动了"。这个 disentangle 能力对 video understanding 极其重要。

#### 3. Allocentric Perspective（6K 题）

这个最难。在图片上标个点 "X"，问："如果你站在 X 那里朝左转 90°，那个冲浪的人在你左边还是右边？"

用 raycast 把 2D 点打到 3D 空间，把 agent teleport 过去，转 90°，再算物体的 x'。

为啥模型惨？因为它看到冲浪的人**当前**在 X 的左边，就答 left。但正确答案是站在 X 朝左看之后，冲浪的人变成了**右边**。模型在做 mental simulation 这件事上几乎完全失败。GPT-4o 只有 37.5%，连 random chance 50% 都不到。

#### 4. Goal Aiming（6.8K 题）

"我要走到厨房台面那边，该往左转还是往右转？"

算物体相对相机的角度：

$$\alpha = \arctan\left(\frac{x'}{z'}\right)$$

- $x'$ 是物体归一化后的水平偏移
- $z'$ 是物体归一化后的深度
- $\alpha < 0$ → 往左转 $|\alpha|$ 度
- $\alpha > 0$ → 往右转 $\alpha$ 度
- $|\alpha| \le 10°$ → "差不多直走"

答案做成离散选择（左/右/直），不要求精确角度——因为从单张图估精确角度太难了。

#### 5. Action Consequence（15K 题，最多）

"如果我往左转 90° 再往前走，会离沙发更近还是更远？"

这个本质上是 **world model 的语言化**——给一个 action，预测下一个 spatial state。数据量最大（15K），因为最 general，既需要 ego-motion 理解又需要 distance 判断。

---

## 训练怎么做的

- LLaVA-1.5-13B 用 LoRA（rank=256, alpha=512）
- LLaVA-Video-7B full fine-tune（模型小）
- 为了不忘旧知识，训练时 40% 概率混入 LLaVA 原来的 instruct tuning 数据
- batch size 8（gradient accumulation），lr 5e-6，cosine annealing，1 epoch
- 2 张 48GB GPU 训练，1 张就能 inference
- 问题格式统一成二选一："choose between left or right"，用文字而非 A/B/1/2——作者发现文字选项比字母编号给 baseline 高不少

一个有意思的 ablation（Figure 7）：**tune vision encoder 比多 tune LLM 参数重要得多**。说明 spatial reasoning 的瓶颈在 visual feature 层面，ViT 没有把几何信息编进去。这跟 ViT 的 translation invariance 有关——图片稍微平移一下，patch feature 几乎不变，但空间关系可能完全变了。

---

## 结果有多炸

### 动态题（Table 2）

| 模型 | SAT Real（150 真实题） | SAT Synthetic |
|---|---|---|
| GPT-4o | 57.5 | 49.4 |
| Gemini-1.5-pro | 64.8 | 49.9 |
| RoboPoint-13B（专门 spatial 训过的） | 46.6 | 53.3 |
| LLaVA-1.5-13B baseline | 41.6 | 51.1 |
| **LLaVA-1.5-13B + SAT** | **54.9** | **87.7** |
| LLaVA-Video-7B baseline | 53.5 | 57.4 |
| **LLaVA-Video-7B + SAT** | **63.4** | **78.0** |

重点看几个数字：
- Gemini-1.5-pro 在 synthetic 动态题上 49.9%，**基本 random**。这些题它从没见过类似的训练数据。
- LLaVA + SAT 在真实动态题上 54.9%，已经超过 GPT-4V（50.7%）和 RoboPoint（46.6%），接近 GPT-4o（57.5%）。
- Perspective 任务从 39.9% → 98.5%，涨了 58 个点——因为 simulator 给了 perfect label，real image 永远没法标这种题。

### 静态题（Table 3）

| 模型 | CVBench Avg | BLINK Avg |
|---|---|---|
| GPT-4o | 78.9 | 70.2 |
| Gemini-1.5-pro | 77.4 | 59.2 |
| RoboPoint-13B | 69.1 | 58.4 |
| LLaVA-1.5-13B baseline | 51.7 | 57.1 |
| **+ SAT** | **75.6** | **64.6** |

CVBench 2DRel 从 46.6% → 89.7%，涨 43 个点。这说明 LLaVA 原本根本不懂 left/right，sim 训练直接补上了。而且 75.6 超过了专门 spatial-tuned 的 RoboPoint-13B（69.1）。

### Video spatial（Table 4，VSI-Bench）

route planning 从 33.5% → 38.7%，涨 5.2 个点。注意 SAT 训练**只用 ≤2 帧**，却能在 long video 上提升——说明学到的是 spatial rule，不是 frame-level pattern。

### Outdoor 泛化（Table 5，MME-RealWorld）

SAT 训练数据全是 indoor、没有人类，但 outdoor+human 视频上 position 涨 8%、interaction 涨 7%、motion 涨 6.4%。这进一步证明 transfer 不靠 appearance match，靠的是抽象的几何规则。

### Sim vs Pseudo 的核心对比（Table 7）

这个表是整篇 paper 最关键的 evidence：

| 训练数据 | SAT Real | CVBench Avg | BLINK Avg |
|---|---|---|---|
| baseline | 41.6 | 51.7 | 57.1 |
| GQAPseudo（225K 真实图 + depth pseudo-label） | 45.1 | 54.5 | 53.1 |
| VSR/25VRD（200K 人工标注关系） | 45.4 | 55.9 | 44.7 |
| SAT Static（模拟器，127K） | 46.0 | 66.4 | 63.0 |
| **SAT Static + Dynamic（175K）** | **54.9** | **75.6** | **64.6** |

GQAPseudo 用 DepthAnything 估深度 + GQA/VG 的 bbox，看起来"真实"但 noisy。结果只在简单题涨，3DDep 反而掉（depth 估不准）。人工标注的关系数据在 in-domain 涨得好，但没法扩展到动态题，SAT Real 完全不涨。

**模拟器的 perfect annotation 全面碾压 noisy real annotation**。而且加动态数据后，连静态题也进一步涨了——动态任务像是更难的 regularizer，防止模型 collapse 到简单 shortcut。

作者还做了个 noise 实验：把 SAT 答案随机翻 20%，CVBench 掉 10%、BLINK 掉 18%。证明模型不是在学 format bias，是真的在学 spatial content。

---

## 为啥 sim-to-real 能 work

直觉上你想：模拟器的室内场景跟真实世界差这么远，凭啥能 transfer？

答案是：**空间关系是 view-invariant 的几何规则**。 "往左转 90° 后物体从左边变到右边" 这个规则，在模拟器和真实世界完全一样。模型学到的不是"这种 texture 的房间怎么导航"，而是"rotation 怎么改变 image 上的 x' 坐标"。这个规则跟房间长什么样、有没有人、室内还是室外，统统无关。

这跟 Geng et al. (2024) 的发现表面矛盾（他们说 synthetic image 有时不如 real），但区别在于：**他们的任务是 semantic classification，依赖 appearance；SAT 的任务是 geometric reasoning，依赖规则**。geometric rule 的 sim-to-real gap 远小于 appearance 的 sim-to-real gap。

---

## 几个我觉得值得深挖的点

### ViT 的几何编码缺陷

Figure 7 显示 tune vision encoder 是关键瓶颈。ViT 的 translation invariance 让它对"图片整体平移了一点点"这种变化几乎无感，但空间关系可能完全变了。一个可能的改进方向：在 ViT 上加 equivariant positional encoding（2D RoPE 之类），或者用 geometric-aware encoder（DINOv2 + depth token）。

### World model 接口

Action Consequence 任务本质上是 "given action → predict next spatial state"，这就是 world model 的语言接口。可以想象把 SAT 接到 Genie 3 或 Sora-style world model 上，用 SAT-style QA 做 next-state verification loss——把 world model rollout 的质量量化。

### Multi-step reasoning

SAT 现在都是 single-step action。扩展到 multi-step（"往左转 90°，往前走，再往右转 45°，物体在哪"）就需要 chain-of-thought 式显式推理。这是把 spatial reasoning 从 reflexive（System 1）推到 deliberative（System 2）的方向。

### 混合 sim + real 的最佳配比

SAT 证明了 sim 数据对 geometric task 好，但对 semantic task（"猫在沙发上睡觉吗"）可能 real image 更好。怎么混？Figure 3 显示 sim-only 已经有效，混入 60% real pretraining data 更好。但最佳比例和采样策略还是 open question。

### Human gap

人类在 SAT 上 92.8%，最好的 SAT-tuned 模型在 SAT Real 上 63.4%。30 个点的 gap 说明 dynamic spatial reasoning 还远没解决。Perspective taking 这种需要 mental simulation 的任务尤其难——可能需要 inference-time 的显式 reasoning（self-talk："如果我在 X 朝左看，cup 会在..."）。

---

## Reference

- SAT 项目页: https://arijitray.com/SAT/
- ProcTHOR: https://procthor.allenai.org/
- AI2THOR 模拟器: https://ai2thor.allenai.org/
- CVBench: https://github.com/cambrian-mllm/cv-bench
- BLINK: https://github.com/Zenyu/Fu-BLINK
- VSR: https://github.com/cambridgeltl/visual_spatial_reasoning
- VSI-Bench: https://arxiv.org/abs/2412.14171
- SpatialVLM: https://arxiv.org/abs/2401.12168
- SpatialRGPT: https://arxiv.org/abs/2406.01584
- RoboPoint: https://arxiv.org/abs/2406.10721
- LLaVA: https://github.com/haotian-liu/LLaVA
- LLaVA-Video: https://arxiv.org/abs/2410.02713
- SPOC: https://spoc-robot.github.io/
- MME-RealWorld: https://github.com/yfzhang114/MME-RealWorld
- Depth Anything: https://github.com/DepthAnything/Depth-Anything-V2
- Molmo/PixMo（下一步大模型验证方向）: https://arxiv.org/abs/2409.17146
- Geng et al. "Unmet Promise of Synthetic Training Images"（sim-to-real 反面证据）: https://arxiv.org/abs/2412.19488

---

# SAT: Dynamic Spatial Aptitude Training for MLMs — 深度解读

Andrej 你好，这篇 SAT 我读得很有意思，几个直觉让我觉得它踩在一个很 sweet 的点上 —— **用 simulator 的 perfect 3D ground truth 来 inject spatial reasoning 的"运动直觉"**，而不是去硬抠 real image 上的 noisy pseudo-annotation。下面我尽量把方法、公式、数据和直觉都摊开讲，并尽量补一些 paper 没明说但你能感觉到的东西。

---

## 1. 核心洞察与定位

SAT 解决的不是 "MLM 看不懂图"，而是 **MLM 在 image 上缺乏 dynamic spatial awareness** —— 即 "如果相机/物体动了，空间关系会怎么变"。这件事在 cognitive science 里被反复证明是儿童空间智能发展的关键 (Anderson et al., 2013 moving room test；Brucato et al., 2023 perspective taking；Franz & Mallot, 2000 aiming/navigation)。之前的 CV-Bench、BLINK、VSR、GQA-Spatial 基本都停留在 **static relationship**（谁在谁左、谁离相机远），SAT 把这个 boundary 推到了 5 类 dynamic 任务。

paper 的关键 claim 我归纳成三条：
- (i) Simulation 的 perfect annotation 比 real image 的 pseudo-annotation 更有效，因为有"因果可执行性"；
- (ii) Dynamic QA 不仅提升 dynamic reasoning，**还反过来提升 static**（这点很反直觉，但有道理）；
- (iii) Sim-only 训练已经有显著 sim-to-real transfer，再 mix 进原 pretraining data 会更好；并且 SAT 只用 indoor sim，却能 transfer 到 outdoor video（MME-RealWorld +4%）。

直觉解释：**spatial reasoning 的 bottleneck 不在 visual appearance，而在 "action → 3D geometry change" 的 mapping**。Simulator 给你 perfect 的这个 mapping，模型学到的不是 texture，而是 "**当我 rotate left 90°，物体在 image 上的 x' 会从正变负**" 这种几何规则 —— 这种规则是 view-invariant 的，所以 sim-to-real 工作。

---

## 2. 数据生成 pipeline 的技术细节

### 2.1 场景与相机

- 用 **ProcTHOR-10K** 的 indoor apartment scenes（22K scenes，1K assets）。ProcTHOR 是 procedurally generated，所以可以无限扩。
- 相机放置策略：随机选 20 个 candidate 位置，挑 visible objects 最多的那一个 —— 保证 scene 信息密度高。
- 相机坐标系约定（这个很关键，paper appendix 写得清楚）：
  - y 轴是 height（向上为正，ceiling y > floor y）；
  - bird's-eye view 下用 (x, z)；
  - 相机始终平行于 x-z plane，rotation 用绕 y 轴的 clockwise angle 描述，相机朝 +z 时为 0°。

### 2.2 相机坐标归一化（公式解析）

对任意一个 object 的 3D 坐标 (x, y, z)，先把相机平移到原点，再旋转使相机朝 +z：

旋转矩阵（绕 y 轴 clockwise 角度 a）：
$$R = \begin{bmatrix} \cos(a) & -\sin(a) \\ \sin(a) & \cos(a) \end{bmatrix}$$

归一化坐标：
$$\begin{bmatrix} x' \\ z' \end{bmatrix} = R \cdot \begin{bmatrix} x - x_0 \\ z - y_0 \end{bmatrix}$$

变量说明：
- $(x_0, y_0)$ 是相机的，注意这里 y_0 是相机的 z 坐标（因为 bird's-eye 下 (x,z) → 论文里写成 (x_0, y_0)，命名有点 confusing，但是是相机在 floor plan 上的 2D 位置）；
- $x'$: 归一化后水平偏移，**负 = 左，正 = 右**；
- $z'$: 归一化后深度，**正 = 远离相机方向**；
- $y$（高度）保持不变，因为不动 camera height；
- $a$: 相机绕 y 轴顺时针旋转的角度（radians）。

这套归一化让 "left/right/in-front/behind" 全部变成 $x', z'$ 的简单数值比较 —— 例如 chair 的 $x'$ < table 的 $x'$ 就直接判定 chair 在 table 左边。这是 simulator 相对 real image 的最大优势：**判定是 deterministic 的，0 噪声**。

### 2.3 Goal Aiming 的角度公式

对 randomly selected object，归一化后用 atan 计算相对相机的水平角度：
$$\alpha = \arctan\left(\frac{x'}{z'}\right)$$

- $\alpha < 0$ → 需要 "turn left by $|\alpha|$ degrees"；
- $\alpha > 0$ → 需要 "turn right by $\alpha$ degrees"；
- $|\alpha - 0| \le \epsilon$（$\epsilon = 10°$）→ "roughly straight"。

QA 形式故意做成**离散选择**（左/右/直），因为 paper 作者也承认从单张 image 精确估计角度对人和机器都太难。这其实是一个很务实的设计 —— **把 continuous 估计 collapse 成 discrete choice**，避免 exact-degree hallucination。

---

## 3. 5 类 Dynamic 任务的设计直觉

这部分我觉得是 paper 最有意思的地方。每一类都直接对应 cognitive science 的一个经典测试：

### 3.1 Egocentric Movement (EgoM, 6.9K)
来源：**moving room test** (Anderson et al., 2013)，本来是测幼儿能不能从两帧判断自己怎么动。
- Action space：RotateLeft/RotateRight ∈ {20,30,40,50,60}°，可选 MoveAhead ∈ [20,40] cm；
- 关键 trick：**rotation 和 translation 是独立维度**（"moving left while rotating right" 是合法的）；
- AI2THOR API: `controller.step(action='RotateRight', degrees=angle)`。
- 直觉：这个任务强迫模型 build 一个 "**optical flow → ego-action**" 的 inverse model，是 embodied navigation 的核心能力。

### 3.2 Object Movement (ObjM, 6.9K)
- Random pick 一个 visible & salient object（bounding box area 阈值过滤），用 `PlaceObjectAtPoint` 移动 [0.25, 0.5] m；
- 有时不动，有时相机也动 —— 模型必须 **disentangle ego-motion vs object-motion**。
- 直觉：这是 "object permanence + 因果归因"，对应婴幼儿物理直觉测试。这个能力对 video understanding、robot manipulation 都极重要。

### 3.3 Allocentric Perspective (Pers, 6K)
来源：Brucato et al., 2023 的 perspective taking test。
- 在 image 上选一个 normalized 坐标 ∈ [0.2, 0.8]² 的点 "X"；
- 用 `GetCoordinateFromRaycast` 把 2D 点 raycast 到 3D，把 agent teleport 过去，random ±90° turn，再判定物体在新视角下的 left/right。
- 直觉：这是经典的 "**theory of mind + spatial**" —— 模型必须 mental simulate "如果我在那个位置朝那个方向看会怎样"。这是 paper 里最难的一项，零样本模型几乎都低于 random chance (50%)。Table 2 里 GPT-4o 的 Pers 只有 37.5，Gemini-1.5-pro 49.5，连 random 都不到 —— 说明 strong static 模型会 fall for "**当前看到的 left/right**" 而非 "**rotation 后的 left/right**"。Figure 5 row 2 的 surfer 例子就是典型：surfer 当前在 X 左边，模型答 left，正确是 right。

### 3.4 Goal Aiming (Aim, 6.8K)
- 对应 navigation 的 pre-requisite (Franz & Mallot, 2000)；
- 用上面的 $\alpha$ 公式 + ε=10° 阈值离散化；
- 直觉：这是 "**visual → motor command**" 的 grounding，对机器人特别有用。SAT-tuned 模型在 SPOC Easy-ObjectNav 上 action prediction accuracy 从 40% → 51%（future work 里提的）。

### 3.5 Action Consequence (EgoAct, 15K) —— 最多的一类
- 给定第一帧，问 "如果我 rotate left 90° + move forward，会离 sofa 更近还是更远 / 会面对还是背对它"；
- 直觉：这是 "**world model 的语言化**" —— 给 action，预测下一帧 spatial relation。是 chain-of-thought causal reasoning 的天然前驱，paper 在 future work 里明确说可以 extend 到 language-controlled world models (Ball et al., 2025 Genie 3)。
- 直觉上 EgoAct 是 SAT 数据量最大 (15K) 的 split，可能因为它最 general —— 既需要 ego-motion 模型，又需要 distance 判断。

---

## 4. 训练 Setup 的技术细节

- **Models**: LLaVA-1.5-13B（image）和 LLaVA-Video-7B（video），用 LoRA rank=256, alpha=512 for 13B；LLaVA-Video 7B full fine-tune（因为小）；
- **Anti-forgetting**: 40% 概率从 LLaVA Instruct tuning 数据里采样混入（对 LLaVA-1.5）；LLaVA-Video paper 主结果没混，但 Figure 4 显示混了更好；
- **Hyperparameters**: batch size 8 (grad accum), lr 5e-6, cosine annealing, 1K warmup, weight decay 0, 1 epoch synthetic；
- **Hardware**: 2×48GB GPU train, 1 GPU inference；
- **Format**: binary choice ("choose between left or right")，用 text option 而非 A/B/1/2 —— 作者发现 text option 比 letter/number 给 baseline 高很多。这个细节很多 paper 不写但很关键；
- **Circular eval** on SAT Real (150 samples)，双向 prompt 平均，降 variance；
- **Prompt prefix** 那段 "A chat between a curious human..." 作者强调对 LLaVA 性能至关重要，tuning 和 testing 都加。

Figure 7 的 ablation 很有意思：**tune vision encoder 是关键瓶颈**，多 tune LLM 参数收益递减。这暗示当前 spatial reasoning 的瓶颈在 visual feature 而非 LLM reasoning 层 —— 这和你常说的 "vision tokens 的几何信息没被 ViT 编码好" 是一致的。

---

## 5. 关键实验数据解读

### 5.1 SAT test set 上的动态能力（Table 2）

| Model | SAT Real Avg | SAT Synth Avg |
|---|---|---|
| GPT-4V | 50.7 | 44.8 |
| GPT-4o | 57.5 | 49.4 |
| Gemini-1.5-pro | 64.8 | 49.9 |
| RoboPoint-13B | 46.6 | 53.3 |
| LLaVA-1.5-13B | 41.6 | 51.1 |
| **+ SAT** | **54.9 (+13.3)** | **87.7 (+36.6)** |
| LLaVA-Video-7B | 53.5 | 57.4 |
| **+ SAT** | **63.4 (+9.9)** | **78.0 (+20.6)** |

直觉解读：
- 即使是 Gemini-1.5-pro / GPT-4o，dynamic reasoning 也只有 ~50% 在 synth 上 —— **几乎 random**。这是 paper 最 striking 的 finding 之一。
- LLaVA + SAT 在 synth 上 87.7 是因为 in-domain；real 上 54.9 已经超过 Gemini-1.5-pro（64.8 略低，但超过 GPT-4V）。**Sim-to-real transfer 是真实存在的**。
- 最 dramatic 的提升在 Pers（39.9 → 98.5，+58.6）和 GoalAim（45.6 → 96.8，+51.2）—— 因为这些是 simulator 可以给 perfect label 的任务，real image 永远没法这样标。
- LLaVA-Video 在 ObjM 反而 -2.3，paper 没深挖，我猜是因为 video 模型原本就强 (82.7)，再 fine-tune 反而引入 in-domain bias。

### 5.2 Static benchmarks 上的提升（Table 3）

| Model | CVBench Avg | BLINK Avg |
|---|---|---|
| GPT-4o | 78.9 | 70.2 |
| Gemini-1.5-pro | 77.4 | 59.2 |
| RoboPoint-13B | 69.1 | 58.4 |
| LLaVA-1.5-13B | 51.7 | 57.1 |
| **+ SAT** | **75.6 (+23.9)** | **64.6 (+7.4)** |
| LLaVA-Video-7B | 65.2 | 56.7 |
| **+ SAT** | **78.4 (+13.2)** | **62.6 (+5.8)** |

- LLaVA-1.5+SAT 在 CVBench 75.6，已经超过 RoboPoint-13B (69.1) —— 而 RoboPoint 是专门 spatially-tuned 的。这是 **"sim 的 perfect annotation > real 的 noisy pseudo-annotation"** 的强证据。
- 最夸张的是 CVBench 2DRel：46.6 → 89.7 (+43.1)。说明 LLaVA-1.5 原本根本不懂 left/right in image，sim 训练直接补上。
- BLINK MV (multi-view) 几乎不涨 (-0.7 for 13B) —— paper 解释是 ViT 的 translation invariance 导致 subtle camera motion 的 feature change 太小。这个点其实指向一个 deeper 的问题：**patch-based ViT 本身在编码 geometric transformation 上就有架构性缺陷**，可能需要 equivariant design 或者 geometric-aware positional encoding。

### 5.3 Video spatial reasoning (Table 4, VSI-Bench)

| Model | Rel Dist | Rel Dir | Rt. Plan | App. Order | MC Avg |
|---|---|---|---|---|---|
| LLaVA-Video-7B | 43.9 | 42.0 | 33.5 | 32.3 | 37.9 |
| **+ SAT** | 47.9 | 39.6 | 38.7 | 40.6 | 41.7 |

- Rt. Plan (route planning) +5.2 是最显著的 —— paper 强调这对 embodied navigation 有直接意义；
- **SAT 训练只用了 ≤2 frames**，却能在 long video 上提升，这是很强的 evidence 说明学到的不是 frame-level pattern 而是 spatial rule。

### 5.4 Outdoor / human generalization (Table 5, MME-RealWorld-Lite)

Avg +4%, Position +8%, Interaction +7%, Motion +6.4%。SAT 训练数据**全是 indoor 无 human**，却能在 outdoor+human 视频上涨 —— 这进一步证明 sim-to-real 不依赖 appearance match，而是依赖 **spatial rule 的抽象性**。

### 5.5 Sim vs Pseudo vs Human annotation (Table 7) —— 这是 paper 最核心的 ablation

| Training data | SAT Real | CVBench Avg | BLINK Avg | VSR |
|---|---|---|---|---|
| LLaVA-1.5 baseline | 41.6 | 51.7 | 57.1 | 66.0 |
| + GQAPseudo (225K) | 45.1 | 54.5 | 53.1 | 62.3 |
| + VSR/25VRD (200K) | 45.4 | 55.9 | 44.7 | 67.9 |
| + SAT Static | 46.0 | 66.4 | 63.0 | 68.0 |
| **+ SAT Static + Dynamic** | **54.9** | **75.6** | **64.6** | **70.4** |

直觉解读：
- GQAPseudo 用 DepthAnything 估深度 + GQA/VG 的 bbox，看起来 "real" 但 noisy —— bbox 不 cover whole object（比如 lake 的一部分在 tree 前，大部分在后），depth estimation 也有误差。结果只在简单 Count/2DRel 涨，3DDep 反而掉（41.2 vs 53.0 baseline）；
- VSR/25VRD 人工标注在 in-domain VSR 涨得好，但没法 compose 出 dynamic 数据，SAT Real 完全不涨；
- **SAT Static 已经全面碾压**，再加 Dynamic 又在所有 split 上再涨一截。
- 作者还做了一个 noise 实验：把 SAT answer 随机 flip 20%，CVB 掉 10%, BLINK 掉 18% —— 证明模型不是在学 format bias，而是真在学 spatial content。

### 5.6 Scaling behavior (Figure 4)

Static-only 数据量加大后会 saturate（likely overfit 简单 left/right 关系），加 Dynamic 后 scaling 更线性。直觉：**dynamic 任务是更难、更 compositional 的训练信号**，等价于一个 regularizer，避免模型 collapse 到简单 shortcut。

---

## 6. Limitations & 我的联想

paper 自己承认的：
- 只 tune 了 LLaVA-1.5-13B 和 LLaVA-Video-7B，没在大模型 (Llama 3, Molmo) 上验证；
- 没测 math/science reasoning 是否退化（虽然 GQA/VQAv2/OK-VQA 没掉）；
- Dynamic reasoning best model 62% (SAT Real)，离 human 92.8% 还差 30 个点。

我的几个联想和延伸：

**(a) Architecture bottleneck**
Figure 7 显示 vision encoder tuning 是关键，但 ViT 的 translation invariance 是 geometric reasoning 的结构性敌人。一个值得做的实验：在 ViT 上加 **equivariant positional encoding**（比如 RoPE 2D 或 NAUTS 的球谐 positional encoding）或者用 **geometry-aware encoder**（DINOv2 + depth token）。SAT 在这个架构上能不能再吃一波红利？

**(b) World model 的语言化**
EgoAct 任务本质上是 "given action, predict next spatial state" —— 这就是 world model 的 language interface。可以想象把 SAT 接到 Genie 3 / Sora-style world model 上，用 SAT-style QA 做 **next-state verification loss**，把 world model 的 rollout 质量量化。

**(c) Chain-of-thought spatial reasoning**
SAT 现在都是 single-step action，可以扩展到 multi-step（"rotate left 90°, move forward, then rotate right 45°，物体在哪"），这就需要 CoT-style 显式推理。这是把 spatial reasoning 从 reflexive 推到 deliberative 的方向。

**(d) Sim 的下一个 level**
ProcTHOR 是 indoor + 物体少。下一步可以上 **Habitat 3.0**（有 humanoids）或者 **Isaac Sim** 的 outdoor 场景。也可以考虑用 **3D Gaussian Splatting** 做 real scene reconstruction + re-rendering，这样能拿到 perfect 3D + real appearance，应该比纯 synthetic 更好 transfer。

**(e) 与 SpatialVLM / SpatialRGPT / RoboPoint 的关系**
SpatialVLM (Chen et al., 2024) 用 internet image + metric depth pseudo-label；SpatialRGPT (Cheng et al., 2024) 用 depth foundation model + bbox；RoboPoint (Yuan et al., 2024b) 用 robotic affordance 标注。SAT 走的是 **opposite direction**：放弃 real image，换 perfect 3D。这条 trade-off 在 Table 7 已经很清楚 —— **noise > domain gap**，至少对 spatial 这种 geometric-heavy 任务是这样。但对 semantic-heavy 任务（比如 "is the cat sleeping on the sofa"），real image 可能还是更好。**混合 sim + real** 的最佳配比是个 open question。

**(f) 你常说的 System 1 / System 2**
SAT 现在主要是 System 1（fast spatial reflex）。但 Pers 和 EgoAct 其实是 System 2（mental simulation）。可以想象在 inference 时让模型显式 generate "如果我走到 X 朝左看，cup 会在..." 这种 self-talk，可能能再涨一波 —— 你在最近的 S1/S2 讨论里也提到过类似 idea。

**(g) 与你的 micrograd / llm.c 教学直觉**
SAT 证明了一件事：**有时候数据比模型架构更重要**。LLaVA-1.5 的架构没动，只是换了 175K sim 数据，就在 CVBench 上 +24 个点。这种 "small data, large gain" 的 case 总是让人觉得：我们离 data efficiency 的极限还很远。

---

## 7. Reference Links

- SAT project page: https://arijitray.com/SAT/
- ProcTHOR paper (Deitke et al., NeurIPS 2022): https://procthor.allenai.org/
- AI2THOR simulator: https://ai2thor.allenai.org/
- CV-Bench (Tong et al., 2024): https://github.com/cambrian-mllm/cv-bench
- BLINK (Fu et al., 2024): https://github.com/Zenyu/Fu-BLINK
- VSR (Liu et al., 2023): https://github.com/cambridgeltl/visual_spatial_reasoning
- VSI-Bench (Yang et al., 2024): https://arxiv.org/abs/2412.14171
- SpatialVLM (Chen et al., 2024): https://arxiv.org/abs/2401.12168
- SpatialRGPT (Cheng et al., 2024): https://arxiv.org/abs/2406.01584
- RoboPoint (Yuan et al., 2024): https://arxiv.org/abs/2406.10721
- Depth Anything (Yang et al., 2024): https://github.com/DepthAnything/Depth-Anything-V2
- GQA dataset: https://nlp.stanford.edu/gqa/
- LLaVA-1.5 (Liu et al., NeurIPS 2024): https://github.com/haotian-liu/LLaVA
- LLaVA-Video / OV (Zhang et al., 2024): https://arxiv.org/abs/2410.02713
- SPOC (Ehsani et al., CVPR 2024): https://spoc-robot.github.io/
- MME-RealWorld (Zhang et al., 2024): https://github.com/yfzhang114/MME-RealWorld
- Genie 3 (Ball et al., 2025): https://arxiv.org/abs/2504.05523 （世界模型方向，可作为 SAT future work 的延伸）
- Molmo / PixMo (Deitke et al., 2024) — 可作为下一步大模型验证: https://arxiv.org/abs/2409.17146

---

## 8. 一句话总结

SAT 的核心 bet 是：**spatial intelligence 的瓶颈在于缺乏 "action → geometry change" 的 perfect supervision signal**，simulator 提供了这个信号且 cost ≈ 0；real-image pseudo-annotation 因为 noise 把这个信号污染掉了。结果是 sim 训练的 LLaVA-13B 在多个 spatial benchmark 上超过 Gemini-1.5-pro 和 RoboPoint。这给 embodied AI、world model、video understanding 都开了一个口子 —— **下一步是把它 push 到 multi-step causal reasoning 和 real-scene 3DGS reconstruction**。
