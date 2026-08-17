---
source_pdf: Motion Prompting Controlling Video Generation with Motion Trajectories.pdf
paper_sha256: cf89a187a7e568bf6dee93c483929630ecf1dca88cfb21cf21a7857f01ba4079
processed_at: '2026-08-05T20:38:59-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Motion Prompting: 人话版

## 一句话总结

**让"动"本身成为指令**——你说不清怎么动，那我直接画给你看。

---

## 这篇paper到底在干嘛

### 痛点

现在text-to-video的痛点很直白：**文字说不清"动"**。

"一只熊快速转头"——快速是多快？转多大角度？加速还是匀速？ease-in-ease-out？什么时候停？一帧帧描述？那不就变成画storyboard了吗。

文字适合描述**静态**，描述不了**动态的细腻**。就像你没法用文字教会别人骑自行车——你得**示范**。

### 解法

那就别翻译了，**直接用motion作为输入**。

具体怎么输入？用**point trajectories**——在video里撒一堆点，让它们沿着你想要的轨迹走，模型看着这些点来生成视频。

就这么简单。

---

## Point Trajectories是啥

想象你在video上撒了一把沙子，每一粒沙子都有自己的轨迹：

```
t=0    t=1    t=2    t=3
●      ●      ●      ●      ← 第1个点
  ●      ●      ●      ●    ← 第2个点
●      ●      ●      ●      ← 第3个点
```

每个点有两个属性：
- **位置** $(x_t^n, y_t^n)$：第$n$个点在时间$t$在哪
- **可见性** $v[n,t]$：第$n$个点在时间$t$能不能看见（被遮挡了就置0）

**为什么不用optical flow？** 因为flow是**相邻两帧**之间的，chain起来误差累积，而且occlusion处理不了。当camera转动时，画面里的东西**进进出出**，flow没法表达这种"消失"和"出现"，但trajectories的visibility flag天然搞定。

---

## 怎么把这些轨迹喂给模型

这是技术上最有意思的地方。

### 朴素想法（错的）

把每个track的坐标直接拼成向量喂进去？问题：track数量是变的，从1个到16,384个都行，怎么搞fixed-size输入？

### Paper的做法

**给每个track一个随机ID**，然后把ID"画"到track经过的每个时空位置。

具体说：
1. 每个track $n$ 分配一个random embedding $\phi_n \in \mathbb{R}^{64}$（64维向量，纯随机，跟track位置无关）
2. 在 $T \times H \times W \times C$ 的conditioning volume里：
   $$\mathbf{c}[t, x_t^n, y_t^n] = \mathbf{v}[n,t] \cdot \phi_n$$
3. track没经过的位置全是0
4. 多个track经过同一位置就**相加**

**人话版**：就像你在video上用荧光笔沿着轨迹画线，每条线用不同颜色（random embedding就是"颜色"）。模型看这些"彩色线条"就知道要怎么动了。

### 为什么是random embedding

这是paper最妙的insight：**embedding跟位置无关**。

如果embedding encode了位置信息（比如用坐标的positional encoding），模型可能**cheat**——直接从embedding读位置，不学motion的shape。

用random embedding逼模型**只能从轨迹的几何形状**里学motion语义。就像DETR的object queries——queries本身没意义，靠attention学到meaning。

---

## 训练有多简单

### 和前人对比

| 方法 | 训练方式 |
|------|---------|
| Tora, MotionCtrl, DragNUWA | Two-stage（先dense后sparse） |
| Image Conductor | Specialized losses |
| MCDiff | Multi-stage fine-tuning |
| MOFA-Video | 每种motion类型单独adapter |
| TrackGo | Custom layers + losses |
| **Motion Prompting** | **Single stage, dense tracks采样, 啥特殊操作都没有** |

### 训练细节

- **Base model**: Lumiere（Google的video diffusion model，5秒16fps）
- **Adapter**: ControlNet结构（copy encoder + zero conv）
- **数据**: 2.2M videos，每video用BootsTAP提取16,384条dense tracks
- **采样**: 每次uniform采样1000-2000条tracks
- **优化器**: Adafactor, lr=1e-4, 70K steps

**就这么训完了**，no tricks。

---

## 两个反直觉发现

### 1. Dense训练对sparse推理也更好

Ablation结果（Table 3）：

| 训练方式 | N=4 tracks推理 | N=2048 tracks推理 |
|---------|---------------|-------------------|
| Sparse训练 | EPE 30.7 | EPE 26.7 |
| **Dense训练** | **EPE 24.6** | **EPE 4.8** |

直觉上你觉得"想用sparse就训sparse"对吧？**错了**。

Dense训练给模型**更丰富的gradient信号**——每帧16,384个点vs几个点，信息量差几个数量级。模型先在dense里学好motion prior，自然generalize到sparse。

### 2. "Sudden convergence"

Training loss一直在降，但模型生成质量**纹丝不动**，直到某一步（约step 20,000）**突然学会**follow conditioning。

这是zero convolution的锅——zero conv一开始输出全0，gradually"打开"让conditioning signal流进去。Loss看着平稳，其实模型在**偷偷学**，只是没show出来。

---

## Motion Prompt Expansion: 把"用户意图"翻译成"详细轨迹"

用户说"让猫转头"——但模型需要具体的trajectories。怎么桥接？

### 五种expansion方式

**1. 鼠标拖动 → 轨迹网格**
- 用户在图片上拖鼠标
- 自动生成以鼠标为中心的track grid
- 可以调grid大小、stride
- 可以"pin"背景（放static tracks让背景不动）

**2. 几何primitive → 物体控制**
- 在物体上放个sphere
- 鼠标拖动 → sphere旋转
- 3D点投影成2D tracks
- 实现"精确旋转"这种单条轨迹做不到的motion

**3. 深度估计 → camera控制**
- 用UniDepth估单目深度 → point cloud
- 给定camera轨迹 → 把point cloud投影到每帧camera → 2D tracks
- Z-buffering处理occlusion
- 鼠标也能控制camera（约束camera在垂直平面，鼠标锁定某个点）

**4. Motion组合**
- Object tracks + Camera tracks
- 把object tracks转成displacement，加到camera tracks上
- 2D下是approximation，extreme camera motion会崩

**5. Motion transfer**
- 源video → BootsTAP提取tracks → 应用到新图片
- 例子：人转头 → 猴子转头；猴子咀嚼 → 树木晃动（out-of-domain）

**Track数量很关键**：太多压制video prior，太少控制不够。Depth-based用~1024，face少一些，out-of-domain要更dense（~1500）。

---

## 涌现行为：物理直觉

这是paper最有意思的observation。

### 例子

**头发**：你拖一下头发，模型生成的视频里头发会**自然飘动**，像有重力、惯性、弹性。

**沙子**：你扫一下沙子，沙粒会**散落、堆积**，像流体力学。

**预测能力**：你可以只给前几帧的motion，让模型**预测后面**——"如果我这样拉头发，它会怎么飘？"

### 这意味着什么

模型从大量video里**学到了physics priors**。Motion prompts给了我们一个**probe**——通过施加motion约束，逼model用学到的prior"填空"，看它学到了什么。

### Probing by failure

Failures也分两类：
1. **Motion conditioning的锅**：cow的horns被锁在背景上，拖动时整个头被拉伸变形
2. **Base model的锅**：拖chess piece，竟然凭空生成新piece

第二类特别有意思——说明video model的**object permanence**有问题。Motion prompts帮我们**诊断**这些limitation。

---

## 和Sora的关系

Sora paper提出"video models as world simulators"。Motion Prompting正好给它**control interface**：

- Sora: "我会模拟世界"
- Motion Prompting: "那我给你指定怎么动，看你模拟得对不对"

这是通向**interactive world model**的一步——用户说"这样动"，模型用学到的physics生成符合直觉的结果。

未来embodied AI可能用这个：robot先在video model里"plan"motion，看效果对不对，再去执行。

---

## 量化结果

### DAVIS benchmark（N=2048 tracks）

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ | EPE↓ |
|--------|-------|-------|--------|------|------|
| Image Conductor | 11.6 | 0.12 | 0.54 | 1891 | 33.6 |
| DragAnything | 14.8 | 0.29 | 0.40 | 1468 | 12.5 |
| **Motion Prompting** | **19.3** | **0.61** | **0.23** | **656** | **3.9** |

**碾压级**，所有指标。

DragAnything在low track时EPE偶尔更好，因为它直接**warp latents**——motion准但视觉artifacts严重。

### Human study

180个2AFC问题，N=103-115人：

- Motion adherence: 74% prefer ours
- Motion quality: 76-80% prefer ours  
- Visual quality: 73-77% prefer ours

---

## 我的几个intuition

### 1. Representation是第一性的

这篇paper再次证明：**选对representation比设计复杂architecture更重要**。

Point tracks with visibility flag这个representation：
- 能表达sparse/dense
- 能表达occlusion
- 能表达任意temporal duration
- 统一了object/camera/scene motion

一个representation + 一个模型 → 五种能力。如果用optical flow，连camera control都做不了（occlusion问题）。

### 2. Random embedding逼模型学真本事

如果embedding encode了位置，模型会cheat。Random embedding让模型**必须从轨迹形状本身学motion语义**。这是个**generalization trick**——通过**移除shortcut**，强迫模型学更generic的representation。

### 3. Simple training beats complex engineering

前人各种two-stage、specialized loss、custom architecture——这篇paper证明：**simple recipe + diverse data**就够了。

这和LLM的lesson一样：**scaling + simple objective > clever engineering**。

### 4. Control interface定义了model能做什么

Text control只能做text能描述的事。Motion control能做motion能表达的事。**interface的expressiveness限制了model的上限**。

这让我想到console和GUI的关系——给了更好的interface，同样的underlying model能做更多事。

---

## 局限

- **非real-time**：生成一个5秒video要12分钟
- **非causal**：不能做实时交互loop
- **2D composition**：object + camera组合在extreme camera motion下崩
- **Tracking noise**：motion magnification需要smoothing
- **2D本质limit**：2D tracks处理不了3D遮挡、视角变化

未来方向：直接用3D point clouds、causal生成、结合physics engine、multi-modal conditioning。

---

## 参考链接

- **项目主页**: https://motion-prompting.github.io/
- **Lumiere**: https://arxiv.org/abs/2401.12945
- **ControlNet**: https://arxiv.org/abs/2302.05543
- **BootsTAP**: https://arxiv.org/abs/2402.00847
- **Co-Tracker3**: https://arxiv.org/abs/2410.11831
- **UniDepth**: https://arxiv.org/abs/2403.18913
- **Sora world simulators**: https://openai.com/research/video-generation-models-as-world-simulators
- **Video physics debate**: https://arxiv.org/abs/2411.02385

---

**一句话再总结**：选对representation（point tracks），用对architecture（ControlNet），保持训练简单（dense + single stage），motion就能成为和text一样强大的control interface，还能顺带probe模型学到了啥physics。

---

# Motion Prompting: 深度解析

## Core Insight: Motion作为"Language"

这篇paper的core idea非常elegant：把**motion trajectories**提升到与**text prompts**同等的地位。就像text prompts在text-to-image中是universal interface一样，motion prompts在video generation中也应该是universal interface。

这个insight的深层含义是：**motion本身就是一种richer的representation**。Text描述"a bear quickly turns its head"时存在ambiguity——"quickly"多快？exact trajectory是什么？应该accelerate吗？ease-in-ease-out timing如何？这些nuances用text难以表达，但用motion本身可以**直接encode**。

这让我联想到语言哲学中的**embodied cognition**——某些knowledge是**intrinsically spatial-temporal**的，无法被**symbolic representation**完全capture。Motion prompts本质上是在说：与其把motion**翻译**成language，不如直接**用motion本身**作为interface。

---

## Representation: Point Trajectories

### 为什么选择Point Trajectories而非Optical Flow？

Paper明确对比了两种motion representation：

| Representation | Pros | Cons |
|---------------|------|------|
| Optical Flow | Dense, well-studied | Error accumulation when chained; No occlusion handling |
| Point Trajectories | Handles occlusion; Sparse & dense; Arbitrary temporal duration | Requires tracking algorithms |

**Key insight**：occlusion handling对camera control至关重要（Sec 4.3）。当camera移动时，points会**enter和exit** the frame，optical flow无法表达这种**discontinuity**，而point trajectories通过visibility flag $\mathbf{v}[n,t] \in \{0,1\}$ 自然处理。

### Formal Definition

$$\mathbf{p} \in \mathbb{R}^{N \times T \times 2}$$

- $N$: number of trajectories（track数量）
- $T$: temporal length（timesteps）
- $2$: 2D coordinates $(x, y)$
- $\mathbf{p}[n,t] = (x_t^n, y_t^n)$: 第$n$个track在第$t$个timestep的position

$$\mathbf{v} \in \mathbb{R}^{N \times T}$$

- $\mathbf{v}[n,t] = 0$: track $n$ at time $t$ is **occluded或off-screen**
- $\mathbf{v}[n,t] = 1$: track $n$ at time $t$ is **visible**

这种representation的**expressiveness**在于：
1. **Any number** of points（从1到16,384）
2. **Object-specific** OR **global scene** motion
3. **Temporally sparse** motion constraints（某些timesteps没有约束）
4. **Occlusion** via visibility flag

---

## Architecture: ControlNet + Random Track Embeddings

### Encoding Strategy

这是paper最**technically interesting**的部分。如何把variable number of tracks encode成fixed-size conditioning signal $\mathbf{c} \in \mathbb{R}^{T \times H \times W \times C}$？

**Solution**: 每个track $n$ 分配一个**unique random embedding** $\phi^n \in \mathbb{R}^C$，然后写入track访问的每个space-time location：

$$\mathbf{c}[t, x_t^n, y_t^n] = \mathbf{v}[n,t] \cdot \phi_n \tag{1}$$

变量解释：
- $\mathbf{c}$: conditioning volume，shape是 $T \times H \times W \times C$
- $t$: timestep index
- $x_t^n, y_t^n$: track $n$ at time $t$ 的quantized 2D坐标
- $\mathbf{v}[n,t]$: visibility flag（multiply后zero-out不可见track）
- $\phi_n$: track $n$ 的unique embedding（从fixed pool随机抽取）

**关键设计决策**：
1. **Embedding与spatial location无关**——位置信息完全由trajectory本身表达
2. **Multiple tracks通过同一location时add embeddings**
3. **其他location设为0**
4. **Dense tracks时等价于forward warping a dense grid of embeddings**

### 与DETR Object Queries的类比

这种**random embedding作为identifier**的设计让我联想到DETR的object queries——queries本身没有semantic meaning，通过attention机制学习**associate** with different objects。这里 $\phi_n$ 也是pure identifier，模型必须从**trajectory shape**中learn motion semantics。

### 与NeRF Positional Encoding的对比

NeRF用positional encoding把spatial location映射到high-dim space。这里**反过来**——把identifier embedding写到spatial-temporal volume中。这是一种**inverse encoding**：位置信息在**index**中，identity信息在**value**中。

### ControlNet Architecture

- **Base model**: Lumiere（Google的video diffusion model，生成5秒16fps视频）
- **Adapter**: ControlNet（copy encoder stack + zero convolutions）
- **First conv layer**: 替换为接受 $T \times H \times W \times C$ conditioning
- $T=80, H=W=128, C=64$

**Zero convolution**导致有趣的**"sudden convergence phenomenon"**：模型在short number of steps内从completely ignoring conditioning到fully trained。Training loss与performance**不correlate**。Fig A2显示test metrics在step 20,000前无改善，然后rapidly converge。

---

## Training Recipe: Simplicity Wins

### Data Pipeline

1. **Dataset**: 2.2M videos（internal）
2. **Preprocessing**: Center crop to square, resize to 256×256
3. **Track extraction**: BootsTAP with dense grid → 16,384 tracks per video
4. **No filtering** of videos（hypothesis: diverse motions → powerful model）

### Training Protocol

- **Optimizer**: Adafactor, lr = $1 \times 10^{-4}$
- **Steps**: 70,000
- **Track sampling**: Uniform from 1000 to 2000
- **Embedding assignment**: Sinusoidal positional encoding, 64 dims, random integers from 0 to 16,384 without replacement
- **Single stage training**（vs. prior work的multi-stage）

### Ablation: Dense vs Sparse Training

Table 3的ablation非常**counterintuitive**：

| Training | N=4 tracks | N=2048 tracks |
|----------|------------|---------------|
| Sparse | PSNR 15.075, EPE 30.712 | PSNR 15.697, EPE 26.724 |
| Dense + Sparse | PSNR 15.162, EPE 29.466 | PSNR 15.294, EPE 27.931 |
| **Dense** | PSNR 15.638, EPE 24.553 | **PSNR 19.197, EPE 4.806** |

**Key finding**: Dense training对sparse tracks也更好！

**Hypothesis**: Sparse tracks提供too little training signal，dense tracks提供更**information-rich** gradient，模型learn更好的motion prior，然后**generalize**到sparse setting。Zero convolutions可能amplify这个效果。

---

## Generalization Properties

模型展现出**surprising generalization**：

1. **Spatial generalization**: 训练时spatially uniform tracks，inference时spatially localized tracks（Figs 3, 6）
2. **Count generalization**: 训练1000-2000 tracks，inference时1到2048+ tracks
3. **Temporal generalization**: 训练时tracks从frame 1开始，inference时tracks从任意frame开始（Fig 3b）

**Why does this work?** Paper hypothesize：
- **Inductive bias** from convolutions
- **Training on large variety** of trajectories

我的额外intuition：random embedding design可能help——因为embedding与spatial/temporal location无关，模型**必须**从trajectory shape本身learn，这创造了更**generic**的motion understanding。

---

## Motion Prompt Expansion

这是paper的第二个key contribution。如同text prompts有prompt expansion/rewriting，motion prompts也需要**expansion**——把high-level user requests转化为detailed tracks。

### Expansion Types

#### 1. Mouse Drags → Grid of Tracks
- User mouse drag → grid of tracks centered on cursor
- User controls: grid stride, grid size
- **Persist option**: tracks remain after drag（objects stay in place）
- **Pin option**: static tracks to keep background still

#### 2. Geometric Primitives → Object Control
- Place sphere over object
- Mouse motion → sphere rotation
- **3D points on sphere** → orthographic projection → 2D tracks
- User specifies: sphere location, radius, point density

#### 3. Depth-based Camera Control
- Monocular depth estimator (UniDepth) → point cloud
- Camera trajectory → re-project point cloud → 2D tracks
- Z-buffering for occlusion flags
- **Mouse → camera**: constrain camera to vertical plane, mouse follows specific point

#### 4. Motion Composition
- Object tracks + camera tracks = simultaneous control
- Method: convert object tracks to **displacements**, add deltas to camera control tracks
- **Limitation**: approximation in 2D, fails for extreme camera motion

#### 5. Motion Transfer
- Source video → BootsTAP → tracks
- Apply tracks to new first frame
- **Subsampling important**: too many tracks suppress video prior, too few = little control
- Sweet spot: ~1024 tracks for depth-based, fewer for face, more for out-of-domain

---

## Emergent Behaviors: Probing Video Priors

这是paper最**philosophically interesting**的部分。Motion prompts成为**probe**来understand video model学到了什么physics/world knowledge。

### Physics Emergence Examples

1. **Hair tossing** (Fig 3b): tracks toss hair → realistic hair dynamics
2. **Sand sweeping** (Fig 4d): tracks sweep sand → granular physics
3. **Prediction**: query model with short motion, let it predict future → "how will hair behave if pulled this way?"

### Probing by Failures

Paper区分两类failures：
1. **Motion conditioning failures**: cow's horns locked to background → unnatural stretching (Fig 9a)
2. **Underlying model failures**: drag chess piece → new piece spontaneously forms (Fig 9b)

**Second category is fascinating**——它reveals model's **learned representations**的limitations。Motion prompts成为**diagnostic tool** for video models。

### Connection to World Models

Paper明确提到与**world simulators**的联系（引用Sora paper [7]）。Motion prompts可能是**interface** for future generative world models——user specifies**what should move**, model simulates**how it moves** based on learned physics.

这与**embodied AI**的vision一致：video models作为visual planners for robots [15, 16, 83]。Motion prompts提供**control interface** for these planners。

---

## Quantitative Evaluation

### DAVIS Benchmark

Table 1关键结果（N=2048 tracks）：

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ | EPE↓ |
|--------|-------|-------|--------|------|------|
| Image Conductor | 11.609 | 0.120 | 0.538 | 1890.7 | 33.561 |
| DragAnything | 14.845 | 0.286 | 0.397 | 1468.4 | 12.485 |
| **Ours** | **19.327** | **0.608** | **0.227** | **655.9** | **3.887** |

**Key observations**:
- **Ours dominates** across all metrics at high track counts
- **DragAnything** sometimes better EPE at low tracks——但has visual artifacts (low PSNR/SSIM/LPIPS/FVD)
- **Trade-off**: DragAnything warps latents directly → accurate motion but artifacts

### Human Study (Table 2)

2AFC test, 180 questions, N=103-115 participants:

| Baseline | Motion Adherence | Motion Quality | Visual Quality |
|----------|------------------|----------------|----------------|
| Image Conductor | 74.3% (±1.1) | 80.5% (±1.0) | 77.3% (±1.0) |
| DragAnything | 74.5% (±1.1) | 75.7% (±1.1) | 73.7% (±1.0) |

**All categories show strong preference** for Motion Prompting.

---

## Additional Applications

### Motion Magnification (Appendix E)
- Input: video with subtle motion
- Process: BootsTAP → Gaussian blur (spatial+temporal) → magnify tracks
- Output: video with amplified motion
- **Smoothing necessary** due to tracking noise

### Human Pose Control (Appendix D)
- Estimate pose → animate keypoints → translate to tracks → feed to model
- Extends motion prompts to **structured motion** representation

---

## Broader Context & Connections

### 与Sora的关系
Sora [7] 提出**video models as world simulators**。Motion prompts提供**controllable interface** for these simulators——可以**probe** what physics/world knowledge they've learned。

### 与TAP/Co-Tracker的ecosystem
这篇paper建立在**point tracking**的recent advances上：
- **TAP** [12]: Tracking any point with per-frame initialization
- **BootsTAP** [13]: Bootstrapped training for tracking
- **Co-Tracker** [37]: Better tracking by tracking together
- **Co-Tracker3** [36]: Simpler, pseudo-labelling

这些methods使**reliable track extraction**成为可能，enabling Motion Prompting的训练data pipeline。

### 与ControlNet ecosystem
Paper在**ControlNet** [87] tradition内，但with **simpler training recipe**。Prior work如Tora, MotionCtrl, DragNUWA, Image Conductor, MCDiff需要：
- Two-stage training
- Specialized losses
- Custom architectures
- Multi-stage fine-tuning
- Data filtering pipelines

Motion Prompting的**simplicity**（single stage, uniform dense tracks, no specialized engineering）是key contribution。

### 与DragGAN lineage
Drag-based editing [48, 44, 62] 的video extension。Motion prompts generalize drag from**single image** to**temporal sequence**，enabling**video editing** via drags。

### 与World Model Physics Debate
Kang et al. [35] 质疑video models是否真正understand physics。Motion prompts提供**controlled probe**——如果model在motion constraints下表现出realistic physics，说明它**has learned** physics priors。但这**不prove**它truly understands physics，可能只是**pattern matching**。

---

## Limitations & Future Directions

### Stated Limitations
1. **Not real-time**: ~12 minutes per video
2. **Not causal**: cannot generate in real-time interaction loop
3. **2D composition approximation**: camera + object composition fails for extreme camera motion
4. **Tracking noise**: requires smoothing for motion magnification

### My Additional Thoughts
1. **3D awareness**: 2D tracks are fundamentally limited——future work could use 3D point clouds directly
2. **Causal version**: for true world model interaction, need causal generation
3. **Physics integration**: combine with explicit physics engines [41, 85, 86]
4. **Multi-modal conditioning**: combine motion + audio + text
5. **Hierarchical motion**: part-level + object-level + scene-level motion

---

## Web Links & References

**Project page**: https://motion-prompting.github.io/

**Key references**:
- Lumiere: https://arxiv.org/abs/2401.12945
- ControlNet: https://arxiv.org/abs/2302.05543
- BootsTAP: https://arxiv.org/abs/2402.00847
- TAPIR: https://arxiv.org/abs/2306.08630
- Co-Tracker: https://arxiv.org/abs/2307.07635
- Co-Tracker3: https://arxiv.org/abs/2410.11831
- UniDepth: https://arxiv.org/abs/2403.18913
- DragAnything: https://arxiv.org/abs/2402.06011
- Image Conductor: https://arxiv.org/abs/2406.15339
- MotionCtrl: https://arxiv.org/abs/2312.03641
- DragNUWA: https://arxiv.org/abs/2308.08089
- Sora (world simulators): https://openai.com/research/video-generation-models-as-world-simulators
- Video physics debate: https://arxiv.org/abs/2411.02385

---

## Final Intuition Build

**Mental model**: 想象你是一个film director。Text prompt是script——描述"what happens"。Motion prompt是choreography——精确specify "how it happens"。

Motion Prompting的elegance在于：**single representation**（point tracks）+ **single model**（one ControlNet）+ **simple training**（dense tracks, no tricks）→ **multitude of capabilities**（object/camera/motion transfer/editing/physics probing）。

**Key takeaway**: 当你have a **sufficiently expressive representation** + **sufficiently powerful base model**，你不需要engineer specialized solutions for each task。Motion prompts就像text prompts一样——**generic interface** that enables **emergent capabilities**。

This paper是**representation matters** principle的strong demonstration。选择正确的representation（point tracks with visibility） unlocks capabilities that would require complex engineering with wrong representation（optical flow without occlusion）。

---

*如果你想深入某个specific aspect——比如random embedding design的theoretical implications，或者motion composition的mathematical formulation，或者与Sora的更deep comparison——我可以elaborate further。*
