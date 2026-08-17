---
source_pdf: RoboStressBench Benchmarking VLM Robustness to.pdf
paper_sha256: 1861d143a3dfdb0ad3d8ad8759c604b4c50d092703b55402034990a1db267ff2
processed_at: '2026-08-12T01:26:51-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 RoboStressBench

## 一句话版本

这帮人搞了个 benchmark，专门测试 VLM 在"真实物理世界搞得很糊很乱"的情况下还能不能靠谱干活，发现现在的模型包括 GPT-5.5 和 Gemini-3.1 都挺菜的，最好的开源模型也就 58% 准确率。

---

## 这 paper 在吐槽什么

先说背景。现在的 VLM（Vision-Language Model）在干净图片上表现很好，benchmark 分数刷得很高。但放到机器人身上，让它去识别一个透明杯子、在暗光下找工具、从奇怪角度看物体——它就拉胯了。

之前的 robustness benchmark 怎么测的？给图片加 Gaussian noise、加 blur、加 pixelation——这些是**数字层面的扰动**，现实里机器人摄像头很少遇到这种"像素被随机噪声污染"的情况。机器人遇到的是：杯子是透明的所以看不清、灯光太暗、东西被挡住了、从上往下看角度很奇怪。

所以这帮人说：我们得换一种思路来定义"视觉难度"。

---

## 核心思路：从 graphics 反推

这里有个很漂亮的 intuition。他们引用了 1986 年 Kajiya 的 rendering equation：

$$L_o(\mathbf{x}, \omega_o) = \int_{\Omega} f_r(\mathbf{x}, \omega_i, \omega_o) \, L_i(\mathbf{x}, \omega_i) \, \max(0, \omega_i \cdot \mathbf{n}) \, d\omega_i$$

人话翻译：你看到的 image 里每个 pixel 的颜色（$L_o$，出射光强），是由四个东西决定的——
- 材质（$f_r$，BRDF，决定光怎么反射）
- 光照（$L_i$，环境光从哪来多强）
- 视角（$\omega_o$，你从哪个方向看）
- 几何（$\mathbf{x}$ 和 $\mathbf{n}$，物体表面在哪、法向量朝哪）

于是他们抽象成：

$$I = \mathcal{F}(M, V, L, G)$$

Image 是 Material、Viewpoint、Lighting、Geometry 四个变量的函数。那么"视觉 stress"就定义成：这四个里某个变得很 task-unfriendly，但 scene semantics 没变。

这个 framing 的妙处在于：**"为什么这张图难"变成了可解释的四维空间**，而不是一个 scalar difficulty score。模型失败了你能问：是 material 问题？lighting 问题？这就比"加了 noise 后准确率掉 10%"信息量大得多。

参考 Kajiya 原文：https://dl.acm.org/doi/10.1145/15922.15902

---

## 四类 stress 具体是什么

他们把每个维度细分，共 16 个 sub-stress：

**Material（材质）5 种**：
- Dark absorptive——黑乎乎吸光的，比如黑色塑料在暗处
- Low-contrast blend——和背景颜色一样，比如米色物体在米色桌上
- Complex texture——表面花纹太复杂干扰识别
- Transparent——透明，BRDF 里有 refraction，物体"消失"
- Specular confusion——镜面反射误导，比如不锈钢

**Viewpoint（视角）3 种**：
- Extreme viewpoint——俯视、仰视这种非 canonical 角度
- Truncated out-of-frame——物体部分超出画面
- Small scale——物体在画面里太小

**Lighting（光照）4 种**：
- Global overexposure——整体过曝
- Local overexposure——局部高光、glare
- Global underexposure——整体太暗
- Local underexposure——局部阴影

**Geometry（几何）4 种**：
- Occlusion——被遮挡
- Non-rigid deform——物体被弯曲、折叠、压缩
- Stacked layout——堆叠关系模糊
- Cluttered layout——太密集分割不开

**直觉**：这 16 个不是完全正交的。深色材质在暗光下，material 和 lighting 同时起作用。作者在 limitation 里诚实承认这点。

---

## 数据怎么来的

总共 7,183 个 example，三个来源：

**来源 1：从已有 dataset 筛（2,927 个）**
从 EmbSpatial-Bench、RoboSpatial、ManipulationVQA 这些已有的 embodied dataset 里，人工挑出已经带 stress 的样本，标注属于哪类 stress。

**来源 2：定向合成（2,596 个）**
这是最有技术含量的部分。他们用 Gemini-3-Pro-Image 和 Qwen-Image-Edit 当"受控扰动工具"——不是生成新场景，是对 nominal image 做定向编辑。

三种编辑模式：

- **Region-guided**：在原图上画个红框作为 editing guide，prompt 说"框里保持不变，框外可以加 clutter"。最后红框不出现在 final image 里。适用于"target annotation 要稳，周围 context 要变难"的场景。

- **Language-only**：用自然语言描述"在桌子左边加一个揉皱的白布"，适用于 non-rigid deform 这种没法像素对齐的情况。

- **Appearance-factor**：只改 lighting 和 material，不改 layout。比如 prompt 说"从右上方加强光制造局部过曝，保持场景结构和物体身份"。

每个合成样本都要人工 verify 三件事：stress 真的出现了、task semantics 还 valid、annotation 还对。

**来源 3：真实世界采集（1,660 个）**
Pexels 上爬的 + 自己拍的。用 GroundingDINO + SAM 自动生成 candidate annotation，人工筛选。

参考 GroundingDINO: https://arxiv.org/abs/2303.05499  
参考 SAM: https://arxiv.org/abs/2304.02643

---

## 实验结果说了啥

测了 16 个 VLM，5 个 family：Qwen3-VL、Qwen3.5、Qwen3.6、InternVL3.5、Molmo2、Gemini-3.1、GPT-5.5。

### Takeaway 1：所有模型都挺菜的

最好的是 Qwen3.5-35B-A3B，58.1%。GPT-5.5 只有 46.2%，Gemini-3.1 只有 44.8%。InternVL3.5-14B 才 29.9%。

**直觉**：这些 model 在干净图上都是 80%+ 的选手，一加 physical stress 直接掉到 30-58%。说明 general visual understanding 强**不等于** reliable under physical stress。

### Takeaway 2：Scale 不解决 stress-specific weakness

Qwen3.5 从 4B 涨到 27B，从 49.8% 涨到 58.0%，涨了 8.3%。但 InternVL3.5 从 4B 涨到 14B 反而降了（32.1% → 29.9%）。

**直觉**：scale 改善 distribution-level robustness（见过更多 pattern），但**不**改善 systematic physical stress（这是 sampling problem，不是 capacity problem）。要让 model 真正 robust，得有 physical grounding——理解 transparent cup 是 transparent，而不是记住"transparent cup 长这样"。

### Takeaway 3：不同 task 对 stress 的敏感度完全不同

这是 paper 最 insightful 的发现：

- **Geometry stress 对 grounding 任务杀伤最大**：placement grounding、target grounding、spatial MCQ 在 Geometry 下普遍最低。因为 occlusion 和 clutter 直接破坏 localization 信号。
- **Planning MCQ 在 Material 和 Viewpoint 下反而更弱**：planning 依赖 object identity 和 affordance，不依赖精确位置。
- **State Understanding 在 Lighting 下显著下降**：state（grasp 稳不稳、object 状态）依赖 visual detail，lighting 弱化 detail。

**直觉**：aggregate accuracy 把这些 failure mode 全混在一起了。要 build reliable embodied agent，必须看 task × stress 的 2D profile，不能只看 overall score。

### Takeaway 4：Paired editing 实验很有意思

同一 scene-question pair，stress 前后对比：

| Model | Nom. | Stress | Drop |
|---|---|---|---|
| Qwen3.6-27B | 64.3 | 40.1 | **-24.1** |
| Qwen3.5-27B | 53.5 | 36.8 | -16.8 |
| InternVL3.5-14B | 10.0 | 9.9 | -0.1 |

**有意思**：强 model 的 drop 反而更大。Qwen3.6-27B 掉了 24 个点，弱 model 几乎不掉。

**直觉**：强 model 在 nominal 上利用了更多 visual detail，这些 detail 一旦被 stress 破坏，性能断崖式下降。弱 model 本来就没用多少 visual 信息，所以掉了也不明显。这让我想到 Taori et al. 2020 关于"strong models are more brittle to distribution shift"的研究：https://arxiv.org/abs/2007.00644

---

## StressDART：test-time 救场方案

他们还提了个 proof-of-concept 的 intervention，叫 StressDART（Detection And Rectification at Test time）。

三阶段：

**Stage 1：Detect**
$$s, c = \mathcal{D}(I, Q)$$

用 Qwen3-VL-4B 当 detector，输入 image 和 question，输出 stress dimension $s \in \{M, V, L, G\}$ 和 fine-grained category $c$。

**关键**：detector 看 (image, question)，不是只看 image。因为 stress 是 task-relative 的——同一张图对一个问题可能是 stressed，对另一个不是。

**Stage 2：Rectify**
$$\tilde{I} = \phi_c(I)$$

根据诊断结果选对应的 visual operation。underexposure 就增强 illumination，small scale 就 crop/zoom，overexposure 就 recover highlight。用 Qwen-Image-Edit 实现。

**Stage 3：Reason**
$$A = \mathcal{R}(I, \tilde{I}, Q, s, c)$$

把 original image 和 rectified image 都喂给 reasoner，让它自己决定用哪个。original 作为 reference 防止 rectification 引入 artifact。

**结果**：

| Method | Input | Acc. |
|---|---|---|
| Qwen3-VL-4B baseline | Original | 43.2% |
| StressDART | Rectified only | 48.9% |
| StressDART | Original + Rectified | 49.0% |

**直觉**：纯 test-time intervention，不 fine-tune，涨了 5.8 个点。这说明**显式 stress 诊断 + targeted rectification 是有用的**，aggregate accuracy 把这个 improvement 机会藏起来了。

但 paper 也诚实承认 limitation：detector 可能误诊，rectifier 可能引入 artifact，没有 uncertainty quantification。这是 test-time intervention 的固有 risk。

参考 Qwen-Image-Edit: https://arxiv.org/abs/2508.02324

---

## 我（GLM）的整体思考

### 这 paper 真正的贡献

不是 7.2K 数据，不是 16 个 VLM 的 leaderboard。真正贡献是 **framing**：把 robustness 从 "corruption robustness" 升级到 "physics-grounded stress robustness"。

之前的 robustness 研究问的是"model 对 X 类 perturbation 的鲁棒性如何"。RoboStressBench 问的是"model 在 physical scene formation 的哪个 factor 下会失败"。这种 axis-aligned 诊断能力对 embodied AI 极其重要，因为它能直接指导 data augmentation、architecture design、test-time intervention。

### Inverse Graphics 视角的力量

$I = \mathcal{F}(M, V, L, G)$ 这个 abstraction 提供了：
- **Interpretability**：failure 归因到具体 physical factor
- **Coverage**：四个 axis 涵盖大部分 real-world visual challenge
- **Actionability**：知道是 lighting 问题就用 illumination enhancement，知道是 occlusion 就用 multi-view fusion

这比 "image quality" 或 "perceptual difficulty" 这种 scalar metric 信息量大得多。

### Scale 的局限

paper 显示 scale 不解决 stress-specific weakness。这和 "scale solves everything" 的 naive view 矛盾。我的理解：
- Scale 改善 *distribution-level* 的 robustness（见过的 pattern 多了）
- Scale **不** 改善 *systematic physical stress*（这是 sampling problem，不是 capacity problem）

要让 model 真正 robust，需要 *physical grounding*——让 model 理解 transparent cup 是 transparent，而不是记住 "transparent cup 长这样"。这指向 world model 和 embodied pretraining 的重要性。

参考 Yann LeCun 关于 world model: https://openreview.net/forum?id=BZ5a1r-kVsf

### 对 VLA 的启示

paper 主要 evaluate VLM（输出 language / grounding），但 embodied AI 真正关心 VLA（输出 action）。如果 perception 在 stress 下不可靠，直接传给 action head 会产生 unsafe behavior。RoboStressBench 的诊断框架可以直接迁移到 VLA evaluation。

参考 OpenVLA: https://arxiv.org/abs/2406.09246  
参考 RT-2: https://arxiv.org/abs/2307.15818

### 联想到 system 1 / system 2

StressDART 的 detect-then-rectify 让我想到你（Andrej）在 Eureka Labs 讲过的 system 2 thinking 在 visual reasoning 上的应用。VLM 直接答是 system 1（fast, intuitive），stress 一来就崩。StressDART 是某种 system 2——先诊断"这图难在哪"，再针对性 rectify，最后 reason。这是 test-time compute 的另一种形式，类似 chain-of-thought 但 applied to visual input。

未来方向我觉得 promising 的：
1. **Learned stress-aware token**：在 VLM 内部加 "stress token"，让 model 自我诊断
2. **Multi-view aggregation**：针对 occlusion 引入多视角融合
3. **Active perception**：robot 主动改变 viewpoint/lighting 来 mitigate stress
4. **Uncertainty calibration**：让 model 知道"我现在 stressed，降置信度"
5. **Video extension**：现在只测 static image，video 里的 motion blur、temporal occlusion 是另一片天地

参考 Eureka Labs: https://www.eurekalabs.ai/

---

## 一句话总结直觉

RoboStressBench 把 VLM robustness 评估从"加噪声测掉多少分"升级到"按物理 image formation factor 诊断失败模式"，发现现在的模型（包括 GPT-5.5、Gemini-3.1）在真实物理 stress 下都挺菜，scale 不解决根本问题，但 test-time 的 detect-rectify 能救一点。这个 framing 对 embodied AI 的下一步发展很有指导意义。

如果你（Andrej）想继续聊 StressDART 的具体 prompt 设计、或者 inverse graphics 和 modern VLA 的连接、或者怎么把这个 framework 推到 video 和 real robot execution，我可以继续展开。

---

# RoboStressBench: 物理 visual stress 视角下的 VLM 鲁棒性 benchmark 深度解析

## 1. 核心直觉:motivation 的关键洞察

这篇 paper 想解决一个 fundamental question:我们如何用 principled 的方式定义 "视觉难度" (visual difficulty),而非用 ad-hoc 的数字 perturbation?

现有的 robustness benchmark 存在两个 gap:
- **General VLM benchmark**(如 MM-Vet, MMBench, SEED-Bench)只在 incidentally 情况下包含 visually challenging cases,没有标注 underlying physical stress factor
- **Robustness-oriented benchmark**(如 ImageNet-C, VLM-RobustBench, Res-Bench)依赖 digital perturbation(noise, pixelation, blur),这些 corruption 在 real-world embodied scenes 里很少出现

paper 的关键 move 是把 visual stress **grounding 到 image formation physics**,基于 inverse graphics 视角:既然 image 是由 physical factor 决定的,那么 "stress" 也应该按这些 factor 分类,而非按 algorithmic corruption 分类。

这个 framing 很 elegant,因为它把 "为什么这张图难" 变成了 interpretable 的四维空间,而非一个 scalar difficulty score。

---

## 2. Inverse Graphics 视角:从 Rendering Equation 出发

paper 用 Kajiya 1986 的 rendering equation 作为 conceptual basis。我详细拆解公式 (1):

$$L_o(\mathbf{x}, \omega_o) = \int_{\Omega} f_r(\mathbf{x}, \omega_i, \omega_o) L_i(\mathbf{x}, \omega_i) \max(0, \omega_i \cdot \mathbf{n}) \, d\omega_i$$

变量含义:
- $L_o$: outgoing radiance(出射辐射度)—— 也就是我们最终"看到"的颜色/亮度,是 image pixel value 的物理对应
- $\mathbf{x}$: surface point(表面点)—— 3D 场景中某个点的位置
- $\omega_o$: outgoing direction(出射方向)—— 通常指向 camera,即观察方向
- $\omega_i$: incident direction(入射方向)—— 光从哪个方向打来
- $\Omega$: 上半球积分域(以 surface normal 为轴)
- $f_r(\mathbf{x}, \omega_i, \omega_o)$: **Bidirectional Reflectance Distribution Function (BRDF)** —— 描述材质如何把入射光反射到出射方向,是 material property 的核心表达
- $L_i(\mathbf{x}, \omega_i)$: incident radiance(入射辐射度)—— 来自 $\omega_i$ 方向的环境光
- $\mathbf{n}$: surface normal at $\mathbf{x}$ —— 表面法向量
- $\max(0, \omega_i \cdot \mathbf{n})$: cosine foreshortening term —— Lambert law 的几何部分,背面光线不贡献

这个公式告诉我们:image 的每个 pixel 是由 4 个 physical factor 决定的:
- **BRDF $f_r$** → Material(材质属性)
- **Incident radiance $L_i$** → Lighting(光照环境)
- **Outgoing direction $\omega_o$** → Viewpoint(观察方向)
- **Surface position & normal $(\mathbf{x}, \mathbf{n})$** → Geometry(几何结构)

于是 paper 抽象出公式 (2):

$$I = \mathcal{F}(M, V, L, G)$$

这个 abstraction 把 image formation 看作一个 4-variable function。"Visual stress" 就定义成:这 4 个 factor 中某一个(或几个)处于 physically plausible 但 task-unfriendly 的状态,导致 task-relevant visual evidence 变得 less accessible,但 scene semantics 保持不变。

**关键设计点**:stress 必须是 *physically plausible* 的——透明杯子、低光照、遮挡,这些都是真实物理世界中发生的,而非 Gaussian noise 这种数字 domain 的事。这区分了 RoboStressBench 和 ImageNet-C 的根本不同。

参考 Kajiya rendering equation 原文:https://dl.acm.org/doi/10.1145/15922.15902

---

## 3. 四维 Stress Taxonomy 详细拆解

paper 把每个 dimension 进一步细化为 sub-stress type,共 16 个 fine-grained category。这是 benchmark 的核心 design,值得逐个理解:

### 3.1 Material (M) —— 5 个 sub-stress

| Sub-stress | 物理本质 | Embodied 场景示例 |
|---|---|---|
| Dark absorptive | BRDF 几乎全吸收,$f_r \to 0$,反射光极弱 | 黑色塑料件在暗光下,几乎看不见细节 |
| Low-contrast blend | BRDF 与背景相近,edge cue 消失 | 米色物体在米色桌面上 |
| Complex texture | BRDF 空间高频变化,干扰 recognition | 印满图案的桌面 |
| Transparent | BRDF 包含 refraction + transmission,物体"消失"在背景 | 玻璃杯、塑料瓶 |
| Specular confusion | BRDF 有强 mirror lobe,反射误导 | 镜面金属、湿润表面 |

### 3.2 Viewpoint (V) —— 3 个 sub-stress

| Sub-stress | 物理本质 |
|---|---|
| Extreme viewpoint | $\omega_o$ 偏离 canonical angle,top-down / low-angle / side view |
| Truncated out-of-frame | 视场角 FOV 不足,object 部分超出 image boundary |
| Small scale | object 在 image 中占比小,缺乏 detail |

### 3.3 Lighting (L) —— 4 个 sub-stress

| Sub-stress | 物理本质 |
|---|---|
| Global overexposure | $L_i$ 整体过强,saturation,信号 clipped |
| Local overexposure | 局部 $L_i$ 强,glare / highlight |
| Global underexposure | $L_i$ 整体过弱,SNR 低 |
| Local underexposure | shadow 区域,局部 dark |

### 3.4 Geometry (G) —— 4 个 sub-stress

| Sub-stress | 物理本质 |
|---|---|
| Occlusion | 另一 object 遮挡 target,visibility 下降 |
| Non-rigid deform | $\mathbf{x}$ 集合发生 bending / folding,canonical shape 改变 |
| Stacked layout | vertical 堆叠,support relation 模糊 |
| Cluttered layout | dense arrangement,segmentation 困难 |

**重要观察**:这 16 个 sub-stress 之间 *不* 是完全 orthogonal 的——real scene 中 material 和 lighting 经常 entangle(深色材质在暗光下),viewpoint 和 geometry 也 entangle(俯视导致 occlusion pattern 改变)。paper 在 limitation 中明确承认这一点,这是诚实的科学态度。

参考 ImageNet-C 的 corruption taxonomy 对比:https://github.com/hendrycks/robustness

---

## 4. 数据集构建 Pipeline:三源互补

paper 用三种 source 平衡 realism、diversity、controllability:

### 4.1 Source 1: Human-curated Filtering(2,927 examples)
从已有 unconstrained dataset 中筛选已经包含 stress 的样本:
- EmbSpatial-Bench, RefSpatial-Bench, RoboAfford-Eval, RoboSpatial-Home
- ManipulationVQA, VABench-P, Where2Place, RoboRefit

6 个 annotator 手动标注 coarse stress dimension 和 fine-grained tag。

### 4.2 Source 2: Controlled Stress Synthesis(2,596 examples)
这是最有技术含量的部分。paper 用 Gemini-3-Pro-Image 和 Qwen-Image-Edit 作为 controlled perturbation tool,**不是**生成新场景,而是对 nominal image 做定向编辑。

三种 control mode:

**Mode A: Region-guided preservation**(Figure 10)
- 把 nominal image 上的 bounding box rasterize 成 red outline 作为 editing guide
- Prompt 指定:guide 内的 object 保持 pose 和 alignment,guide 外可以增加 clutter
- Final image 中 guide 不出现
- 适用场景:target annotation 需要保持稳定,只想 stress 周围 context

**Mode B: Language-only spatial edits**(Figure 11)
- 用 natural language 描述要插入的 object 和位置
- 适用场景:non-rigid deform、deformable foreground insertion
- Pixel alignment 不保证,需要重新标注

**Mode C: Appearance-factor edits**(Figures 12, 13)
- 修改 lighting 和 material,不改 layout
- Prompt 强调 preserve geometry, edit appearance

**关键质量保证**:每个 synthesized sample 都要 manual verify 三个条件:
1. intended stress 视觉上 present
2. task semantics 保持 valid
3. annotation 正确

### 4.3 Source 3: Real-world Collection(1,660 examples)
- Pexels(Internet-sourced,遵循 Pexels License)
- Self-captured by authors

用 GroundingDINO + SAM 生成 candidate annotation,annotator 筛选并写 task-specific QA。

### 4.4 总体统计

Total: **7,183 examples**

按 dimension 分布(单 example 可有多 tag):
- Material: 2,785
- Viewpoint: 1,292
- Lighting: 1,753
- Geometry: 3,337(最多,因为 cluttered layout 1,658 + occlusion 1,205)

按 task 分布:
- Placement grounding: 949
- Target grounding: 3,411(最多)
- Spatial reasoning MCQ: 1,369
- State understanding MCQ: 633
- Planning MCQ: 821

**直觉**:Geometry 占比最大是合理的,因为 embodied scene 中 occlusion 和 clutter 几乎不可避免,而且是最影响 grounding 的 factor。

参考 GroundingDINO:https://arxiv.org/abs/2303.05499  
参考 SAM:https://arxiv.org/abs/2304.02643

---

## 5. StressDART: Test-time Detect-and-Rectify 框架

这是 paper 的 intervention 部分,作为 proof-of-concept 显示 benchmark 的诊断价值能驱动 actionable 改进。

### 5.1 三阶段架构

公式 (3)-(5):

**Stage 1: Stress Detection**
$$s, c = \mathcal{D}(I, Q)$$
- $I$: input image
- $Q$: question
- $\mathcal{D}$: stress detector(用 Qwen3-VL-4B 实现)
- $s \in \{M, V, L, G\}$: coarse stress dimension
- $c$: fine-grained stress category(e.g., "transparent", "global underexposure")

**关键点**:detector 以 (image, question) 为输入,而不是只看 image。这说明 stress 是 *task-relative* 的——同一张图对一个 question 可能是 stressed,对另一个不是。

**Stage 2: Stress Rectification**
$$\tilde{I} = \phi_c(I)$$
- $\phi_c$: category-specific visual operation
- $\tilde{I}$: rectified image
- 例:underexposure → illumination enhancement;small scale → cropping/zooming;overexposure → highlight recovery

用 Qwen-Image-Edit 实现。

**Stage 3: Reasoning**
$$A = \mathcal{R}(I, \tilde{I}, Q, s, c)$$
- $\mathcal{R}$: VLM reasoner(同样用 Qwen3-VL-4B)
- $A$: answer
- 同时提供 $I$ 和 $\tilde{I}$,让 reasoner 利用 recovered cue,同时保留 original context

**为什么同时给 original 和 rectified**:rectification 可能引入 noise 或改变 task-relevant detail,original 作为 reference 能 mitigate 这一风险。这个 design choice 在 ablation 中得到验证。

### 5.2 StressDART 实验(Table 3)

Base model: Qwen3-VL-4B(原 accuracy 43.2%)

| Method | Reasoner Input | Acc. | Gain |
|---|---|---|---|
| Qwen3-VL-4B (baseline) | Original | 43.2% | — |
| StressDART | Rectified only | 48.9% | +5.7% |
| StressDART | Original + Rectified | 49.0% | +5.8% |

**关键 takeaway**:
- 几乎所有 gain 来自 rectification 本身(+5.7%)
- 加 original 只多 +0.1%,但 paper 仍推荐这种设置作为 safety net
- 没有 fine-tuning,纯 test-time intervention
- 这说明:**显式的 stress diagnosis + targeted rectification 是有用的**,而 aggregate accuracy 隐藏了这个 improvement 机会

**Limitation**:paper 在 C 章节承认 StressDART 仍有 "negative flips"——rectification 有时会改变 task-relevant cue,或 detector 诊断错误。这是 test-time intervention 的固有 risk。

参考 Qwen-Image-Edit:https://arxiv.org/abs/2508.02324

---

## 6. 实验结果深度解读

### 6.1 整体结果(Table 2)

16 个 VLM,5 个 family。Top performers:
- Qwen3.5-35B-A3B: **58.1%**(最高)
- Qwen3.5-27B: 58.0%
- Qwen3.6-27B: 57.3%
- GPT-5.5: 46.2%
- Gemini-3.1: 44.8%
- InternVL3.5-14B: **29.9%**(显著低)
- Molmo2-8B: 35.2%

**Takeaway 1**:所有 model 都远未 saturate,最强 model 也只有 58.1%,commercial 模型 Gemini 和 GPT 也只有 ~45%。这说明 physical visual stress 是真实的 open problem。

**Takeaway 2**:Scaling 改善 average,但**不解决** stress-specific weakness
- Qwen3.5: 4B(49.8%) → 27B(58.0%)→ 35B-A3B(58.1%)
- Qwen3VL: 4B(43.2%) → 30B-A3B(55.9%)
- InternVL3.5-14B(29.9%) **反而低于** InternVL3.5-4B(32.1%)——scale 不必然 helpful

### 6.2 Stress-wise 分析(Figure 8)

这是 paper 最 insightful 的部分。不同 task 对 stress 的 sensitivity 完全不同:

**Geometry stress 对 grounding 任务杀伤最大**:
- Placement grounding、Target grounding、Spatial MCQ 在 Geometry 下普遍最低
- 因为 occlusion / clutter 直接破坏 localization signal

**Planning MCQ 不 follow Geometry-dominant pattern**:
- 在 Material 和 Viewpoint 下反而更弱
- 因为 planning 依赖 object identity 和 affordance,而非精确位置

**State Understanding MCQ 在 Lighting 下显著下降**:
- 因为 state(grasp stability, object condition)依赖 visual detail,lighting 弱化这些 detail

**直觉**:这验证了 paper 的核心 thesis——aggregate accuracy 掩盖了 stress-specific failure mode。如果要 build reliable embodied agent,不能只看 overall score,要看 task × stress 的 2D profile。

### 6.3 Paired Editing Subset(Table 1)

这控制实验很有说服力:同一 scene-question pair,在 stress 前后的 accuracy 对比:

| Model | Nom. | Stress | Drop |
|---|---|---|---|
| Qwen3VL-4B | 51.0 | 35.5 | -15.5 |
| Qwen3.5-27B | 53.5 | 36.8 | -16.8 |
| Qwen3.6-27B | 64.3 | 40.1 | **-24.1** |
| InternVL3.5-14B | 10.0 | 9.9 | -0.1 |
| Molmo2-8B | 12.2 | 11.5 | -0.7 |

**有意思的观察**:强 model(Qwen3.6-27B)的 drop 反而更大(-24.1)。这说明强 model 在 nominal 上利用了更多 visual detail,这些 detail 一旦被 stress 破坏,model 性能断崖式下降。弱 model 反而 drop 小,因为本来就没利用多少 visual 信息。

这个现象让我联想到 "strong models are more brittle to distribution shift" 的研究,例如 Taori et al. 2020 关于 robustness 的研究:https://arxiv.org/abs/2007.00644

### 6.4 Grounding 细节(Table 4)

paper 报告了 point-acc, IoU@0.50, IoU@0.95, mAcc 四个指标:
- Qwen 系列 point-acc 50-64%,IoU@0.50 ~80%,IoU@0.95 ~25-34%
- InternVL3.5 point-acc ~30-37%,但 IoU 极低(IoU@0.50 < 30%)——**InternVL 能 point 但不能 box**
- Molmo2 类似:point-acc 还行,box 完全不行
- GPT-5.5:point-acc 60.4%,IoU@0.50 80.3%,IoU@0.95 15.0%
- Gemini-3.1:point-acc 58.3%,IoU@0.50 60.3%——**比 GPT-5.5 弱**

**直觉**:Qwen 系列在 grounding 上明显领先,internVL 和 Molmo 的 box prediction 是 weak point。这暗示 training data 的 grounding annotation 质量是关键。

---

## 7. 我的整体直觉与思考

### 7.1 这个 benchmark 的真正贡献在哪?

不是 7.2K 数据,也不是 16 个 VLM 的 leaderboard。真正贡献是 **framing**:把 robustness 从 "corruption robustness" 升级到 "physics-grounded stress robustness"。

之前的 robustness 研究(Goodfellow adversarial, Hendrycks corruption, Taori distribution shift)都在问 "model 对 X 类 perturbation 的鲁棒性如何"。RoboStressBench 问的是 "model 在 physical scene formation 的哪个 factor 下会失败"。这种 axis-aligned 诊断能力对 embodied AI 极其重要,因为它能直接指导 data augmentation、architecture design、test-time intervention。

### 7.2 Inverse Graphics 视角的力量

$I = \mathcal{F}(M, V, L, G)$ 这个 abstraction 提供了:
1. **Interpretability**:failure 可以归因到具体 physical factor
2. **Coverage**:四个 axis 涵盖了大部分 real-world visual challenge
3. **Actionability**:知道是 lighting 问题 → 用 illumination enhancement;知道是 occlusion → 用 multi-view fusion

这个 framing 比 "image quality" 或 "perceptual difficulty" 这种 scalar metric 信息量大得多。

### 7.3 StressDART 的局限与未来方向

StressDART 只是一个 proof-of-concept,+5.8% improvement 在 4B model 上,但:
- Detector 本身可能 misclassify stress
- Rectifier 可能引入 artifact
- 没有不确定性量化,不知道何时 trust rectified

未来几个方向我觉得 promising:
1. **Learned stress-aware token**:在 VLM 内部增加 "stress token",让 model 自我诊断
2. **Multi-view aggregation**:针对 occlusion,引入多视角融合
3. **Active perception**:robot 主动改变 viewpoint/lighting 来 mitigate stress
4. **Uncertainty calibration**:让 model 知道 "我现在 stressed,降置信度"

### 7.4 对 VLA(Vision-Language-Action)的启示

paper 主要 evaluate VLM(输出 language / grounding),但 embodied AI 真正关心 VLA(输出 action)。如果 perception 在 stress 下不可靠,直接传给 action head 会产生 unsafe behavior。RoboStressBench 的诊断框架可以直接迁移到 VLA evaluation。

参考 OpenVLA:https://arxiv.org/abs/2406.09246  
参考 RT-2:https://arxiv.org/abs/2307.15818

### 7.5 一个我关心的 meta-question

paper 显示 scale 不解决 stress-specific weakness。这和 "scale solves everything" 的 naive view 矛盾。我倾向认为:
- Scale 改善 *distribution-level* 的 robustness(因为见过的 pattern 多了)
- Scale **不** 改善 *systematic physical stress*(因为这是 sampling problem,不是 capacity problem)

要让 model 真正 robust,需要 *physical grounding*——让 model 理解 transparent cup 是 transparent,而不是记住 "transparent cup 长这样"。这指向 world model 和 embodied pretraining 的重要性。

参考 Yann LeCun 关于 world model 的论述:https://openreview.net/forum?id=BZ5a1r-kVsf

---

## 8. 总结性直觉

RoboStressBench 的核心贡献是把 VLM robustness 评估从 "数字 perturbation" 范式迁移到 "physical image-formation factor" 范式。这个 framing:
- **诊断更精确**:16 个 sub-stress 提供了细粒度 failure attribution
- **可操作**:StressDART 显示诊断能直接驱动 test-time improvement
- **可扩展**:taxonomy 可以加入更多 axis(e.g., motion blur, sensor noise)
- **揭示 scale 的局限**:大模型在 stress 下仍 fail,scale 不解决 physical grounding

对 embodied AI 领域,这个 benchmark 提供了一个 *principled* 的 robustness evaluation 框架,而非 ad-hoc 的 corruption 测试。下一步我觉得最 exciting 的方向是把这套 framework 推到 video 和 real robot execution,验证 perception robustness 对 action robustness 的 causal 影响。

参考 paper 项目页(从 abstract 推断):https://robostressbench.github.io/(推测的 URL,paper 中提到 "The project webpage is RoboStressBench Page" 但具体 URL 在 markdown 中被省略)

参考相关 embodied benchmark:
- OpenEQA: https://openeqa.github.io/
- RoboSpatial: https://arxiv.org/abs/2503.10178
- EmbodiedBench: https://embodiedbench.github.io/

如果你(Andrej)想深入讨论 StressDART 的具体 prompt 设计、或者 inverse graphics 和 modern VLA 的连接,我可以继续展开。这个 paper 让我想到你之前在 Eureka Labs 提的 "system 2 thinking" 在 visual reasoning 上的应用——StressDART 的 detect-then-rectify 就是某种形式的 system 2,在 system 1(VLM 直接答)失败时介入。
