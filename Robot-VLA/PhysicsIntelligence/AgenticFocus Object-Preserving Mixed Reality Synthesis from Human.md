---
source_pdf: AgenticFocus Object-Preserving Mixed Reality Synthesis from Human.pdf
paper_sha256: 51ce29020ea635250c94780df0432d863f09f7a2e9ffa383cd7d1cd6f5968403
processed_at: '2026-08-18T00:15:14-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# AgenticFocus 人话版

好, Karpathy, 我把之前那版技术拆解重新用大白话讲一遍, 但该给的公式和链接还是给, 因为你 build intuition 需要这些锚点。

---

## 1. 这 paper 想干嘛

一句话: **你拿 GoPro 拍自己做饭的视频, 它能把这个视频变成 robot 能学做饭的训练数据。**

不是随便变成"看起来像 robot 的视频"就完了, 它要产出两个东西配对好:
- 一段 video (人手被换成 robot 手, object 保留)
- 一份 action + state 文件 (robot 每个 joint 该怎么动)

这两个东西 time-aligned, 可以直接喂给 VLA policy 训练。

为什么这事重要: 你之前在 Tesla 和 OpenAI 都讲过, Physical AI 的 bottleneck 不是 model, 是 data。Humanoid dexterous manipulation 的 data 尤其贵——teleoperation 一小时几百美金, mocap 手套笨重, 跨 robot 不通用。但 human egocentric video 在 YouTube / EPIC-KITCHENS 上有 millions of hours, 白捡。问题是怎么把这个"白捡"的 data 变成 robot 能用的。

相关背景:
- EPIC-KITCHENS: https://epic-kitchens.github.io/
- OpenVLA (数据 consumer 例子): https://openvla.github.io/
- DexCap (hardware-heavy 路线对比): https://dexcap.github.io/

---

## 2. 为什么难: 三个坑

### 坑 1: 视角对不上

人拍 FPV, camera 绑在头上, 跟着头转。Robot 的 camera 也绑在头上, 但 robot 头和人头的高度、转动方式、mounting angle 全不一样。你直接拿人的视角给 robot 看, robot 会困惑——"这个画面跟我自己看到的不是一回事"。

### 坑 2: 手把 object 遮住了 (最要命的)

Dexterous manipulation 的核心是手指 wrap 在 object 上。这意味着**在最关键的 contact region, object 恰好被手指挡住**。你看不见 object 被 grasp 的那部分 geometry。

传统做法是 inpainting——把手擦掉, 让 generative model 猜背景。但它在 contact region 猜出来的 object geometry 是 hallucinate 的, 形状可能错, 边界可能糊。Robot policy 学这种 hallucinated geometry 会学到错误的 grasp point。

这就是为什么 Masquerade (https://arxiv.org/abs/2406.07788) 之类的 diffusion editing 方法在 contact-rich 场景会翻车——它们在最重要的地方在编造。

### 坑 3: 光有视频不够, 还要有 action

就算你视觉做对了, 还得有一组 robot 真能执行的 joint trajectory 跟视觉配对。光有"看起来像 robot 的视频"训不出 policy, policy 需要 (observation, action) pair。

---

## 3. 他们的核心思路: 别 generate, 用 MR

这是整篇 paper 最聪明的地方。

他们引用了一个观察: 现代 agentic robot 系统通常分层——high-level 模块 (SayCan https://arxiv.org/abs/2204.01691, PaLM-E https://arxiv.org/abs/2303.03378) 决定"做什么", low-level reactive policy 执行"怎么动"。

在这套架构下, reactive controller **不需要完整 scene reconstruction**, 它只需要:
- task-relevant object 的 focused view
- local contact geometry
- embodiment-consistent motion

所以他们不做 video-to-video translation (那是 Masquerade / Do as I Do https://arxiv.org/abs/2606.19333 的路线), 他们做 **Mixed Reality synthesis**:
- 保留 object 的真实 geometry (从 video 里抠出来, 不 generate)
- 擦掉人手, 换成 MuJoCo render 的 robot 手 (deterministic, 不 hallucinate)
- 在 robot 的 virtual camera 视角下重新合成

这个 choice 的好处: **deterministic**。Robot 手是 MuJoCo explicit render 出来的, 没有 diffusion artifact。Object 是从真实视频抠出来的, 没有 hallucinated geometry。唯一"猜"的部分是 background inpainting (人手挡住的那部分背景), 但 background 对 grasp 不 critical。

---

## 4. Pipeline 三步走

### Step 1: 把 object 抠出来, 别让人手弄坏它

先用 VLM (类似 GPT-4V) 看第一帧, 问它"这个视频里人在操作什么 object"。VLM 回答比如"knife"。然后 SAM2 (https://github.com/facebookresearch/sam2) 跟踪这个 knife 的 mask 跨整个视频。

但光有 visible mask 不够——手挡住 knife 的时候, mask 只剩 visible fragment, knife 的完整 shape 丢了。所以他们做了一个聪明的事:

**定义一个 "object-preserving inpainting mask":**

$$\mathcal{M}_{\mathrm{inpaint}}(t) = \mathcal{M}_{\mathrm{hand}}(t) \cap \neg \mathcal{M}_{\mathrm{obj}}(t)$$

人话翻译:
- $\mathcal{M}_{\mathrm{hand}}(t)$: 第 $t$ 帧里, 哪些像素是手 (1 = 是手, 0 = 不是手)
- $\mathcal{M}_{\mathrm{obj}}(t)$: 第 $t$ 帧里, 哪些像素是 object
- $\neg \mathcal{M}_{\mathrm{obj}}(t)$: 取反, 1 = 不是 object, 0 = 是 object
- $\cap$: 两个 mask 逐像素 AND
- 结果 $\mathcal{M}_{\mathrm{inpaint}}(t)$: "是手 AND 不是 object" 的像素

意思就是: **擦手的时候, 手压在 object 上的那部分别擦**。只擦手在背景上的部分。

然后把这个 mask 喂给 E2FGVI (https://github.com/Megvii-Research/E2FGVI) 做 video inpainting, 重建背景。背景干净了, object 的 visible 部分没被误擦。

但还不够。Inpainting 之后, contact region 可能还有手残影、边界 flicker。所以他们从某个干净 frame 抠一个 object template (RGBA), 用 SAM2 mask 跟踪它的位置, 重新贴回去。这样 object 的 appearance 和 geometry 来自一个 trusted snapshot, 位置由 mask 驱动, 两者解耦, 抑制 ghosting。

直觉: 传统 inpainting 想做 "擦前景, 恢复背景"。他们想做 "擦人, **但别碰 object**"。公式 (1) 就把这个意图编码成三个像素集合操作。

### Step 2: 重建人手 3D motion, retarget 到 robot

用 HaMeR (https://github.com/geopavlakos/hamer) 或 WiLoR (https://arxiv.org/abs/2501.08244) 从每帧估计 3D hand pose——wrist 的 6-DoF (位置 + 朝向) 加 finger joints。输出是 MANO 参数化的人手。

然后 retarget 到 Unitree G1 (https://www.unitree.com/g1/) arm + BrainCo dexterous hand。这里有个关键公式:

$$\mathbf{P}_{\mathrm{robot}} = \beta \mathbf{R}_{\mathrm{cam}} \mathbf{P}_{\mathrm{human}} + \mathbf{T}_{\mathrm{offset}}$$

人话:
- $\mathbf{P}_{\mathrm{human}}$: 人手 3D 点, 在人的 camera 坐标系下
- $\mathbf{R}_{\mathrm{cam}}$: rotation matrix, 把人 camera 的轴系 rotate 到 robot virtual camera 的轴系 (因为两个 camera 朝向不同)
- $\beta$: scale factor, 人手臂展 ~1.8m, G1 arm 短一些, 要 scale down 到 robot workspace
- $\mathbf{T}_{\mathrm{offset}}$: translation, robot virtual camera 在 robot torso 坐标系下的位置 (因为 robot camera 装在 head 上, 跟 torso 有 offset)
- $\mathbf{P}_{\mathrm{robot}}$: 转换后, 在 robot virtual camera 坐标系下的 3D 点

这就是个 rigid transform + scale。意思是: 人手的 motion 从人的视角"移植"到 robot 的视角, 跟原始视频怎么拍的解耦, 只依赖 robot 自己的 embodiment 几何。这是他们和 baseline 最不一样的地方——Masquerade 没有显式 camera-relative formulation。

然后用 MuJoCo (https://mujoco.readthedocs.io/) IK solver 把 Cartesian target 解成 joint configuration $\hat{\mathbf{q}}_t$。

但 IK 解出来的 raw joint 会 jitter (hand reconstructor 有噪声, IK 有 numerical noise), 所以套一层 EMA:

$$\mathbf{q}_t = \alpha \hat{\mathbf{q}}_t + (1 - \alpha) \mathbf{q}_{t-1}$$

- $\hat{\mathbf{q}}_t$: 第 $t$ 帧 raw IK estimate
- $\mathbf{q}_t$: smoothed 后的 joint config
- $\mathbf{q}_{t-1}$: 上一帧 smoothed 值
- $\alpha \in [0,1]$: smoothing coefficient, 小 = 强 smoothing + 大 lag, 大 = 弱 smoothing + 小 lag

就是个 IIR low-pass filter, $\alpha=0.3$ 在 30 FPS 下截止频率约 1.4 Hz, 压掉 hand reconstructor 的高频抖动。

最后加 fixed wrist orientation correction (因为 robot hand 装在 arm 上的角度和人 palm 朝向不同), 输出 arm trajectory + finger command + camera-relative state。

### Step 3: 分层合成, 让 contact 看起来对

这是最 crafty 的部分。

朴素做法: 把 robot 手直接 overlay 到 cleaned scene 上。问题:
- 手整个在 object 前面 → 看起来"飘"在 object 表面, 没有 grasp 感
- 手整个在 object 后面 → 看不到手指, grasp 结构丢失
- 小 object + wrap-around grasp 时尤其糟

他们的做法: **分两层 render robot 手**:

1. **Full articulated-hand pass**: 整只手 render 一遍
2. **Near-contact thumb pass**: 单独 render thumb 的 contact 部分

然后把 object template composite 到这两层之间。Object 可以 occlude 它应该 occlude 的 finger region (比如 index finger 远端被 object 挡住), 同时 thumb 单独 render 在 object 前面 (visible contact)。

这模拟的是真实抓握: thumb 在 object 一侧 (看得见), 其他 fingers 在 object 另一侧 (被挡)。Dexterous manipulation 视觉里最容易翻车的地方, 他们 explicit handle。

Rendering 在 headless MuJoCo 里用 G1 + BrainCo hand mesh, 不用 diffusion。好处: deterministic, 无 hallucination。坏处: lighting / shadow 可能和真实 scene 不 match, 但对 reactive policy 可能不 critical。

---

## 5. 实验做得怎么样

两个 metric:

### Metric 1: Trajectory 3D position error

$$e_t = 100 \|\hat{\mathbf{p}}_t - \mathbf{p}_t\|_2$$

就是重建 trajectory 和 ground truth 的 3D 距离, 单位 cm。$\hat{\mathbf{p}}_t$ 是重建的, $\mathbf{p}_t$ 是 GT, 乘 100 把 meter 换成 cm。AgenticFocus 最低 (具体数字在 Figure 3 里, 正文没给)。

### Metric 2: SPARC (smoothness)

这个 metric 比较有意思。它不直接看 trajectory, 而是看 wrist speed 的**频域特性**。

直觉: smooth motion 的 speed 信号是 low-frequency 的, 频谱集中在低频, 高频基本 0。jittery motion 有高频分量, 频谱拖尾长。SPARC 算的是 normalized 频谱曲线的 arc length, arc length 越短 = 越 smooth。

公式:

$$\mathrm{SPARC} = -\int_0^{\omega_c} \sqrt{\left(\frac{1}{\omega_c}\right)^2 + \left(\frac{d\hat{V}(\omega)}{d\omega}\right)^2} d\omega$$

- $\hat{V}(\omega) = V(\omega)/V(0)$: wrist speed 的 Fourier magnitude, 除以 DC component normalize (让 metric 和 overall speed 无关)
- $\omega_c$: cutoff frequency, 高频不算
- 根号里: arc length 元素, $\sqrt{dx^2 + dy^2}$ 形式
- 负号: convention, less negative = smoother

结果:

| Method | SPARC | 95% CI |
|---|---|---|
| **AgenticFocus** | **−5.18** | [−5.38, −4.98] |
| Masquerade | −5.56 | [−5.84, −5.31] |
| Do as I Do | −6.05 | [−6.42, −5.72] |

比 Masquerade 好 ~7%, 比 Do as I Do 好 ~14%。CI 基本不重叠 (AgenticFocus 下界 −5.38 < Masquerade 上界 −5.31, borderline), 和 Do as I Do 显著分开。

SPARC metric 原始 paper: https://ieeexplore.ieee.org/document/6127552

---

## 6. 我觉得值得吐槽的地方

### 吐槽 1: SPARC 改进 7-14% 到底有没有用

SPARC 是 movement science 领域的 metric, 它衡量 motion smoothness。但 **smoothness 和 downstream policy success rate 之间的 causal link, paper 没建立**。他们自己在 conclusion 承认: "Extending this evaluation to downstream policy training and broader embodiments is a natural direction for future work."

意思是: 这套 pipeline 产出的 data 训出来的 policy 到底行不行, **现在没 data**。这是最大的 open question。

### 吐槽 2: "Clean frame" 假设很脆弱

Step 1 最后, 他们从某个 clean frame 抠 object template。但假设视频里**存在**一个 clean frame——object 没被手挡的瞬间。Long-horizon task 里 object 从头到尾被部分 occlude, 没有 clean frame, 这套方法退化。Paper 没讨论这个 edge case。

可能的 fix: 多帧 mask 取 union, 用 generative 3D object prior (比如 3DGP) 做 shape completion。但这又引入 hallucination 风险, 和他们 deterministic 的设计哲学冲突。这是个 tension。

### 吐槽 3: Hand reconstructor 在重 occlusion 下崩

HaMeR / WiLoR 在 hand 被 object 挡住超过 ~30% 时, 重建质量显著退化。这是 community 已知问题。AgenticFocus 只用 EMA smoothing 压噪声, 没做任何 occlusion-aware refinement。Heavy occlusion 的 frame 会 inject noise 到 IK, EMA 平滑掉一部分但引入 lag。$\alpha$ 的选择是 trade-off, paper 没给具体值和 ablation。

### 吐槽 4: Fixed wrist correction 的隐藏假设

公式 (2) 是 camera-relative, 但 wrist orientation correction 是 fixed constant。这隐含假设: robot hand 装在 arm 上的角度 constant, AND human palm 朝向能稳定估计。后者在 hand 部分遮挡时也会崩。

### 吐槽 5: DoF mismatch 没讨论

人手 ~27 DoF, BrainCo hand 通常 6-12 DoF。从 MANO 到 BrainCo 的 mapping 一定丢 information。Paper 没说这个 mapping 怎么做的 (DexPilot-style? joint-to-joint? 是否有 finger synergy 压缩?)。这是 cross-embodiment retargeting 的 fundamental loss, 理应讨论。

相关: DexPilot retargeting 方法 https://arxiv.org/abs/1910.03135

### 吐槽 6: 和 diffusion 方法对比可能不公平

Masquerade / Do as I Do 用 generative editing, 它们能处理 AgenticFocus 处理不了的事 (scene 之外的 visual consistency, shadow / lighting match)。AgenticFocus 的 layered compositing 在 visual realism 上可能更差, robot 手看起来"贴"在 scene 上。对下游 policy 是否更好, 取决于 policy 对 lighting / shadow 的敏感度, 这要在 sim-to-real 实验里才能定论。

### 吐槽 7: Single primary object 假设

VLM 选 task-relevant object 假设 single primary object。但 long-horizon task 里 object 会切换 (先拿刀, 再切菜, 再放盘子里)。现在 pipeline 不 handle multi-object sequence。

---

## 7. 我觉得真正聪明的地方

### 聪明 1: 公式 (1) 的 set operation

$$\mathcal{M}_{\mathrm{inpaint}} = \mathcal{M}_{\mathrm{hand}} \cap \neg \mathcal{M}_{\mathrm{obj}}$$

它把 "preserve object under occlusion" 这个抽象意图, 直接编码成 3 个像素级集合操作。它不解决"重建被遮住的 object geometry"这个难题, 它只做"别在擦手的时候把 object 也擦了"。这是个**子问题分解**的好例子——把 hard problem 退化成 easier problem。

这和你 Karpathy 平时讲的"能 deterministic 就别 hallucinate"是同一种直觉。在 Physical AI 这种 safety-critical 又 data-starved 的领域, 这种保守主义可能反而是优势。

### 聪明 2: Camera-relative retargeting formulation

公式 (2) 把 human motion 从原始 video 的 camera frame 解耦到 robot virtual camera frame。这看似简单 (就是个 rigid transform + scale), 但它解决了一个 fundamental 问题: **retargeting 不再依赖视频是怎么拍的**, 只依赖 robot embodiment 几何。这意味着 pipeline 对任意 FPV video 都适用, 不需要 per-video calibrate camera extrinsic。

### 聪明 3: Layered compositing 替代 single overlay

把 robot 手分 full-hand pass 和 thumb pass, 让 object template "包"在中间。这模拟真实抓握的 depth ordering (thumb 在 object 一侧 visible, 其他 fingers 在另一侧 occluded)。在 dexterous manipulation 视觉里, depth ordering 错了会让 grasp 看起来完全不对, policy 学到错误的 contact 结构。这个 crafty 的分层是必要的, 不是 over-engineering。

### 聪明 4: Pipeline 的 modularity

每一段都可替换:
- VLM (selectable)
- SAM2 → 未来 SAM3
- E2FGVI → 未来更好的 video inpainter
- HaMeR/WiLoR → 未来更好的 hand reconstructor
- MuJoCo IK + render → 可换其他 physics engine

这意味着 pipeline 的 quality 会随 sub-component 进步自动变好。这是好的 system design。

---

## 8. 这 paper 在 landscape 里的位置

| 路线 | 代表 | 特点 |
|---|---|---|
| Hardware-heavy | DexCap (https://dexcap.github.io/) | 高质量, 低 scalability |
| Rich egocentric sensing | EgoDex, EgoEngine (https://arxiv.org/abs/2606.12604) | 中等 hardware |
| Generative video editing | Masquerade (https://arxiv.org/abs/2406.07788), Do as I Do (https://arxiv.org/abs/2606.19333) | 可 scale, 但 contact region hallucinate |
| Mixed Reality synthesis | ARMADA (https://arxiv.org/abs/2412.10631), **AgenticFocus** | Deterministic, contact-preserving |

AgenticFocus 在 landscape 里的独特位置: **deterministic MR + camera-relative retargeting + layered compositing for contact-rich dexterous**。它填补了 "ordinary monocular FPV video → dexterous humanoid demo" 这个具体 regime。

---

## 9. 如果让我给下一步建议

1. **最紧迫**: 跑 downstream policy training 实验。用 AgenticFocus 产出的 data 训一个 VLA policy, 在 sim 和 real 上测 success rate, 和 Masquerade 产出的 data 训的 policy 比。这是唯一能回答"这套 pipeline 到底有没有用"的实验。

2. **Object template 的 robustness**: 处理 no-clean-frame 场景。可以用 multi-frame mask union + light generative completion, 但要 quantify 引入的 hallucination error。

3. **Hand reconstruction refinement**: 在 occlusion 下用 temporal optimization 或 video diffusion refine hand pose, 但保持 deterministic 输出 (用 diffusion 做 prior, 用 optimization 做 output)。

4. **Multi-object / bimanual 扩展**: 现在 pipeline 假设 single hand + single object。扩展到双手协作 + object 切换。

5. **Active object state**: 现在 object template 是 static RGBA。但 manipulation 中 object state 变化 (门打开, 抽屉拉出, 盖子翻开)。需要 state-aware object representation, 可能用 3D object model with articulated joints。

6. **Force / tactile supervision**: 现在 supervision 只有 vision + action。如果能从 hand-object interaction 推断 contact force (用 grasp 时的 finger config + object stiffness model), 加进 supervision, 对 reactive control 更有用。

7. **Sim-to-real robustness study**: MR-synthesized data 训的 policy 在 real robot 上是否比 diffusion-synthesized data 训的 policy 更 robust? 这是个 open empirical question, 但回答它能 settle deterministic vs generative 的 debate。

---

## 10. 最 takeaway 的一句话

**公式 (1) 是整篇 paper 的灵魂。**

$$\mathcal{M}_{\mathrm{inpaint}} = \mathcal{M}_{\mathrm{hand}} \cap \neg \mathcal{M}_{\mathrm{obj}}$$

三个像素集合操作, 把"保留 object geometry under occlusion"这个 hard problem 退化成"别擦掉 object"这个 easy problem。这种"避开 generative, 用 deterministic set operation 解决子问题"的设计哲学, 在 Physical AI 这种 data-starved 又 safety-critical 的领域, 可能正是对的路子。

剩下所有东西 (camera-relative retargeting, layered compositing, EMA smoothing) 都是围绕这个核心 idea 的 supporting engineering。

如果你想深挖某一段 (比如 IK 在 MuJoCo 里的具体 formulation, SPARC 的 frequency-domain 直觉, 或者 layered compositing 的 render 顺序细节), 告诉我, 我再展开。

---

# AgenticFocus 深度解读

Karpathy 你好, 这篇 paper 正好打在你最关心的 Physical AI data bottleneck 痛点上。我用尽可能直觉化的方式拆给你看, 公式逐变量讲, 并指出几个值得深究的设计选择。

---

## 1. 一句话定位

AgenticFocus 把一段普通的 human FPV (first-person view) 视频, 转换成可以直接喂给 dexterous humanoid policy 训练的 **(vision, action, state) tuple 序列**。它的卖点是: 不需要 mocap 手套, 不需要 stereo rig, 不需要 scene-specific digital twin, 一根 monocular egocentric 视频进去, 一份 robot-trainable dataset 出来。

项目相关链接 (我尽量给到能找到的官方资源):

- SAM2 (segmentation backbone): https://github.com/facebookresearch/sam2
- E2FGVI / E2FGVI-HQ (video inpainting): https://github.com/Megvii-Research/E2FGVI
- HaMeR (3D hand reconstruction): https://github.com/geopavlakos/hamer
- WiLoR (in-the-wild hand localization): https://arxiv.org/abs/2501.08244
- MuJoCo (IK solver + renderer): https://mujoco.readthedocs.io/
- Unitree G1: https://www.unitree.com/g1/
- BrainCo dexterous hand: https://www.brainco.tech/
- EPIC-KITCHENS dataset: https://epic-kitchens.github.io/
- SPARC smoothness metric (原始 paper): https://ieeexplore.ieee.org/document/6127552
- DexCap (baseline 对照): https://dexcap.github.io/
- EgoEngine (近期 egocentric 工作): https://arxiv.org/abs/2606.12604
- Do as I Do (baseline): https://arxiv.org/abs/2606.19333
- Masquerade (baseline): https://arxiv.org/abs/2406.07788 (近似, 论文 ref [9] 是 ICRA 2026 版本)

---

## 2. 为什么这件事难: 三个耦合的 gap

paper 在 Introduction 里把问题拆成三段, 这套 framing 很值得记住, 它解释了为什么不是简单地"换一只手":

**Viewpoint gap.** Human FPV 视频是从 head- 或 chest-mounted camera 拍的, camera 是绑在 demonstrator 身上的, 跟着人转。而 humanoid 的 observation 定义在 robot-centered frame, camera 装在 robot 头上。两个 camera 的 ego-motion 完全不同, 直接套用会丢掉 viewpoint 一致性。

**Interaction-region gap.** 这是最致命的。Dexterous manipulation 中, 手指几乎一定 wrap 在 object 周围, 也就是说**在最关键的 contact region 上, object 的 geometry 恰好被 hand 遮住**。传统的 actor removal / inpainting / generative editing 会在这些 region 失败, 因为它们没有"被遮住那部分 object"的信息, 只能 hallucinate, 而 hallucinate 出来的 geometry 对 downstream control 是有毒的 (grasp point 错位, contact normal 错位)。

**Action grounding gap.** 就算你把视觉做对了, 还要有一组 embodiment-consistent 的 action 和 state 跟视觉配对。光有"看起来像 robot 的视频"不够, 还要有"robot 真的能 follow 的 joint trajectory + camera-relative state"。

---

## 3. 核心 intuition: 为什么是 Mixed Reality, 不是 video translation

这部分是整篇 paper 最有想法的地方。他们引用了一个观察: 现代 agentic robot system 通常**把 high-level reasoning 和 low-level control 解耦**——一个 deliberative module (类似 SayCan [1], Inner Monologue [7], PaLM-E [5]) 决定 goal 和 relevant objects, 一个 reactive policy 执行 fine-grained motion。

在这套架构下, reactive controller **根本不需要完整 scene reconstruction**, 它需要的是:
1. task-relevant objects 的 focused representation
2. local contact geometry
3. embodiment-consistent motion

这个 framing 直接决定了 pipeline 的形态——他们不是在做"video-to-video translation" (那是 Masquerade / Do as I Do / EgoEngine 的路线), 而是在做 **Mixed Reality synthesis**: 保留 physical interaction structure (object geometry + contact), 替换 embodiment (human hand → robot hand), 然后在 robot 的 virtual viewpoint 下重新 render。

这个 choice 有几个连锁的后果:

- **Deterministic 而非 generative**: 他们 explicit 避开 diffusion-based video editing, 用 MuJoCo explicit articulated model render robot hand。结果是 "free of diffusion-style hallucination artifacts"。这点很关键——diffusion inpainting 在 contact region 的 hallucination 恰恰是 policy 学习的 poison。
- **Object geometry 是第一公民**: 他们 build 一个 "object-preserving inpainting mask", 把 object 当成需要被保护的结构, 而不是被 hand 顺便擦掉的一部分。
- **Layered compositing 替代 single overlay**: 因为他们要保留 depth ordering, 所以分多层 render, 这点后面细讲。

Reference: Milgram & Kishino 的 MR taxonomy [13] (https://cs-east.utm.utoronto.ca/~molly/428-2016/Milgram-coloc95.pdf), Diminished Reality 综述 [14] (https://ipsjvx.ip.jsa.ip/?page_id=2732 en)。

---

## 4. Method 拆解: 三个 stage, 三个关键公式

Pipeline 是三段: (A) Object-Centric Restoration, (B) Full-Hand Humanoid Retargeting, (C) Layered Mixed-Reality Compositing。下面逐个拆。

### 4.1 Stage A: Object-Centric Restoration

**Goal**: 把 task-relevant object 从视频里 isolate 出来, 并在 hand occlusion 下保留**完整 object geometry**, 而不是只剩 visible fragments。

**Step 1 — Object selection & tracking.** VLM (paper 没指定具体哪个, 推测是 GPT-4V 或类似) 从第一帧或几帧里 pick 出 task-relevant object。然后 SAM2 [20] 跟踪这个 object 的 mask 跨整段视频。输出 per-frame: object mask $\mathcal{M}_{\mathrm{obj}}(t)$, RGBA crop, debug video。

**Step 2 — Object-preserving inpainting mask.** 这是第一个关键公式 (1):

$$\mathcal{M}_{\mathrm{inpaint}}(t) = \mathcal{M}_{\mathrm{hand}}(t) \cap \neg \mathcal{M}_{\mathrm{obj}}(t)$$

变量解释:
- $\mathcal{M}_{\mathrm{hand}}(t)$: 第 $t$ 帧 human hand/arm 的 binary mask (取 1 表示 "this pixel is hand")
- $\mathcal{M}_{\mathrm{obj}}(t)$: 第 $t$ 帧 target object 的 binary mask
- $\neg \mathcal{M}_{\mathrm{obj}}(t)$: logical NOT, 取 complement, 即 "this pixel is **not** object"
- $\cap$: pixel-wise logical AND
- $\mathcal{M}_{\mathrm{inpaint}}(t)$: 最终用于 inpainting 的 mask

直觉: 这个 mask 表示 "属于 hand、但同时**不**落在 object 上的像素"。换句话说, 只擦除 hand 在 background 上的部分, hand 在 object 上的部分**保留给下一步处理**, 这样 object 的 visible 部分不会被 inpainter 误擦。这是个非常干净的 set operation, 把 "preserve object" 这个意图直接编码进 mask。

**Step 3 — Background inpainting.** 把 $\mathcal{M}_{\mathrm{inpaint}}(t)$ 喂给 E2FGVI-HQ [10] 做 video inpainting, 重建背景。

**Step 4 — Stable object template reinsertion.** Inpainting 完背景之后, residual artifacts 还会在 contact region 留下 (hand 残影, boundary flicker)。所以他们从某个 clean frame 取一个 stable **object template** (RGBA), 然后用 per-frame object mask 跟踪其位置, 重新 composite 进去。

这步相当于 "object 的 geometry 和 appearance 从一个 trusted snapshot 取, 位置由 SAM2 mask 驱动"——把 appearance 和 tracking 解耦, 抑制 ghosting。

> 一个 build-intuition 的点: 传统 inpainting 想做的是 "remove foreground, recover background"; AgenticFocus 想做的是 "remove human, **保留 object**"。公式 (1) 就是把这个语义偏移编码进去。

### 4.2 Stage B: Full-Hand Humanoid Retargeting

**Goal**: 从 video 重建 human hand 3D motion, 再 retarget 到 Unitree G1 arm + BrainCo dexterous hand, 输出 robot joint trajectory + camera-relative state。

**Step 1 — Hand reconstruction.** 用 WiLoR [19] / HaMeR [18] 估计每帧 3D hand pose (wrist 6-DoF + finger joints)。HaMeR 是 transformer-based 的 3D hand reconstructor, 输出 MANO 参数; WiLoR 在 wild 条件下做 localization + reconstruction。

**Step 2 — Camera-relative mapping.** 这就是第二个关键公式 (2):

$$\mathbf{P}_{\mathrm{robot}} = \beta \mathbf{R}_{\mathrm{cam}} \mathbf{P}_{\mathrm{human}} + \mathbf{T}_{\mathrm{offset}}$$

变量解释:
- $\mathbf{P}_{\mathrm{human}} \in \mathbb{R}^3$: human camera frame 下的 3D 点 (来自 hand reconstructor)
- $\mathbf{R}_{\mathrm{cam}} \in SO(3)$: camera-axis transformation, 把 human camera 的轴系 rotate 到 robot virtual camera 的轴系
- $\beta \in \mathbb{R}_{>0}$: workspace scaling factor, 把 human reach scale 到 robot reach (human arm ≈ 70cm, G1 arm 可能 50cm 之类, workspace 大小不同)
- $\mathbf{T}_{\mathrm{offset}} \in \mathbb{R}^3$: virtual robot camera 相对 robot torso frame 的 position (因为 robot camera 装在 robot head 上, 跟 torso 有固定 offset)
- $\mathbf{P}_{\mathrm{robot}} \in \mathbb{R}^3$: 转换到 robot virtual camera frame 下的 3D 点

公式形式是个 standard rigid transform + scale: $s\mathbf{R}\mathbf{p} + \mathbf{t}$。直觉是: 我们把 human 在自己 camera frame 里的 motion, "transplant" 到 robot 的 virtual camera frame 里, 这样 retargeting 不再依赖原始视频是怎么拍的, 只依赖 robot 自己的 embodiment 几何。这一点是和 baseline 最不一样的地方——Masquerade 之类的方法没有显式 camera-relative formulation, 所以会有 viewpoint mismatch。

**Step 3 — IK in MuJoCo.** 用 MuJoCo [21] 的 IK solver 把 $\mathbf{P}_{\mathrm{robot}}$ ( Cartesian wrist target + finger targets) 解成 robot joint configuration $\hat{\mathbf{q}}_t$。

**Step 4 — EMA smoothing.** IK 解出来的 raw $\hat{\mathbf{q}}_t$ 会 jitter (hand reconstructor 噪声 + IK numerical noise), 所以套一层 exponential moving average (3):

$$\mathbf{q}_t = \alpha \hat{\mathbf{q}}_t + (1 - \alpha) \mathbf{q}_{t-1}$$

变量:
- $\hat{\mathbf{q}}_t$: 第 $t$ 帧 raw IK estimate (joint configuration vector, 维度 = robot arm DoF + hand DoF)
- $\mathbf{q}_t$: smoothed 后的 joint configuration
- $\mathbf{q}_{t-1}$: 上一帧 smoothed 值 (recurrence)
- $\alpha \in [0, 1]$: smoothing coefficient, $\alpha$ 越小越平滑但 lag 越大, $\alpha=1$ 等于不平滑

这是个 standard IIR low-pass filter, 传递函数是 $\frac{\alpha z}{z - (1-\alpha)}$, 截止频率 $f_c \approx \frac{\alpha F_s}{2\pi}$ (在 $F_s = 30$ FPS 下, $\alpha=0.3$ 给 $f_c \approx 1.4$ Hz, 这基本压制了 hand reconstructor 的高频抖动)。

**Step 5 — Fixed wrist orientation correction.** 因为 robot hand 装在 G1 末端的角度和 human palm 朝向不同, 他们加 fixed orientation offset 修正。这里有个隐含假设: 这个 offset 是 constant, 即所有视频里 camera mounting 都一致。这在同一段视频内 OK, 跨视频就要 per-video calibrate。

**Step 6 — Output.** 一份 synchronized action-state record: arm trajectories, finger commands, camera-relative robot states (存为 .npz / .json)。

### 4.3 Stage C: Layered Mixed-Reality Compositing

**Goal**: 把 retargeted robot render 到 cleaned scene 上, 使 contact region 看起来 plausible。

**Key challenge**: 单层 overlay 不行:
- 如果 robot hand 整个在 object 前面 → contact 看起来"飘"在 object 表面
- 如果整个在后面 → grasp 结构丢失 (看不到手指怎么 wrap)
- Small object + wrap-around grasp 时尤其糟

**他们的解决**: 把 robot rendering 拆成两个 pass:

1. **Full articulated-hand pass**: 整只手正常 render
2. **Near-contact thumb pass**: 单独 render thumb 的 contact 部分

然后把 Step 4 (Stage A) 得到的 object template composite 到这些 render 之间——具体说, object template "包"在 hand render 外面, 让 object 可以 occlude 它应该 occlude 的 finger region (比如 index finger 的远端), 同时保留 visible thumb contact (因为 thumb 是单独的 pass, render 在 object 前面)。

这个 layered 结构模拟的是真实抓握中: thumb 在 object 一侧 (visible), 其他 fingers 在 object 另一侧 (被 occlude)。这是 dexterous manipulation 视觉里最容易翻车的地方, 他们 explicit handle 它, 而不是寄希望于 single overlay 凑巧 work。

**Rendering choice**: 在 headless MuJoCo 里用 Unitree G1 + BrainCo hand meshes render, **不**用 diffusion / generative。好处是 deterministic, 没有 hallucination artifact; 坏处是 lighting / shadow 和真实 scene 不一定 match, 不过对于 reactive policy learning, 这可能不是 critical (policy 应该学到 robust to lighting)。

---

## 5. Experiments: 两个 metric, 一张 bar chart, 一张曲线图

### 5.1 Setup

- Dataset: EPIC-KITCHENS [4] + internal demos, 30 FPS
- 输出: (i) MR video, (ii) .npz/.json action-state
- Baselines: **Masquerade** [9], **Do as I Do** [17]
- 所有 trajectory 先 temporal resample 到 normalized timeline, 再 transform 到同一 coordinate frame, 然后比

### 5.2 Trajectory Accuracy — Mean 3D position error

公式 (4):

$$e_t = 100 \|\hat{\mathbf{p}}_t - \mathbf{p}_t\|_2 = 100 \sqrt{(\hat{x}_t - x_t)^2 + (\hat{y}_t - y_t)^2 + (\hat{z}_t - z_t)^2}$$

变量:
- $\mathbf{p}_t = (x_t, y_t, z_t)^T$: ground-truth 3D trajectory 在第 $t$ 帧的位置 (meter)
- $\hat{\mathbf{p}}_t = (\hat{x}_t, \hat{y}_t, \hat{z}_t)^T$: reconstructed trajectory 在第 $t$ 帧的位置
- $\|\cdot\|_2$: Euclidean norm
- $100$: 单位换算, meter → centimeter
- 最终 $e_t$ 是 per-frame error (cm), 然后 average over all frames and clips, 报 95% CI

直觉: 这就是个 $\ell_2$ 位置误差, 用 cm 量。结果 (Figure 3) AgenticFocus 最低, 具体 number paper 里没在正文给 (要去看图)。从 abstract 看 SPARC 给了数字, trajectory error 只说"lowest"。

### 5.3 Retargeting Smoothness — SPARC

这个 metric 比较有意思, 是 Balasubramanian 2012 [2] 提出的 movement smoothness metric。它**在频域**算, 比时域 jerk 之类更 robust to noise。

给定 wrist speed signal $v(t)$, 取其 Fourier magnitude spectrum $V(\omega)$, 然后 normalize:

$$\hat{V}(\omega) = \frac{V(\omega)}{V(0)}$$

- $V(\omega)$: $|\mathcal{F}\{v(t)\}(\omega)|$, wrist speed 的 Fourier magnitude
- $V(0)$: spectrum 的 DC component (即 $\int v(t) dt / T$, mean speed), 用它 normalize 使曲线 invariant to overall speed magnitude
- $\hat{V}(\omega)$: normalized spectrum, 量纲 1

然后 SPARC (公式 5):

$$\mathrm{SPARC} = -\int_0^{\omega_c} \sqrt{\left(\frac{1}{\omega_c}\right)^2 + \left(\frac{d\hat{V}(\omega)}{d\omega}\right)^2} \, d\omega$$

变量:
- $\omega$: frequency (rad/s 或 Hz, 取决于 Fourier convention)
- $\omega_c$: cutoff frequency, 高频以上不计入 (剔除 noise)
- $d\hat{V}(\omega)/d\omega$: normalized spectrum 对 frequency 的导数
- 根号里面: 这是个 arc length 项, $\sqrt{dx^2 + dy^2}$ 的形式, 其中 $dx = d\omega / \omega_c$ (normalized frequency step), $dy = d\hat{V}/d\omega \cdot d\omega$ (spectrum 变化量)
- 整个积分: spectrum curve 从 $0$ 到 $\omega_c$ 的 arc length
- 负号: 按约定让 smoother motion 对应 less negative (closer to 0) 的值

直觉 (这是关键 build intuition 处): 

- 如果 motion 完全 smooth, $v(t)$ 几乎是 slow-varying 信号, 它的 Fourier spectrum 集中在低频, 高频基本是 0, $\hat{V}(\omega)$ 在低频快速衰减到 0 然后保持 0 → arc length 短 → SPARC 接近 0 (less negative)
- 如果 motion 有 jitter / tremor / discontinuity, $v(t)$ 有高频分量, $\hat{V}(\omega)$ 在整个频域有 nontrivial 变化 → arc length 长 → SPARC 很负
- 用 normalized spectrum 而非 raw, 是为了 invariant to motion 的 overall speed (otherwise 慢 motion 看起来总是 "smooth")

**结果**:

| Method | Mean SPARC | 95% Bootstrap CI |
|---|---|---|
| **AgenticFocus** | **−5.18** | [−5.38, −4.98] |
| Masquerade [9] | −5.56 | [−5.84, −5.31] |
| Do as I Do [17] | −6.05 | [−6.42, −5.72] |

相对改进: AgenticFocus vs Masquerade 减少约 **7%** ($\frac{5.56-5.18}{5.56} \approx 6.8\%$), vs Do as I Do 减少约 **14%** ($\frac{6.05-5.18}{6.05} \approx 14.4\%$)。

CI 不重叠 (Masquerade 的上界 −5.31 高于 AgenticFocus 的下界 −5.38? 等等, 让我重算: AgenticFocus 的 95% CI 是 [−5.38, −4.98], Masquerade 是 [−5.84, −5.31]; −5.38 < −5.31, 所以 CI 几乎相接但不重叠, borderline significant)。Do as I Do 的 CI 是 [−6.42, −5.72], 完全在 AgenticFocus CI 下方, 差异显著。

Figure 4(a) 是 75 个 episode 的 rolling 7-episode mean, AgenticFocus 全程最低 magnitude, 稳定在 less negative 区间。

---

## 6. 我的 critique 和 reflection

这部分是给 Karpathy 你看的, 我直说几个我觉得 paper 没 self-disclose 的弱点:

### 6.1 SPARC 改进幅度的实际意义

7% 到 14% 的 SPARC 改进, 在 movement science 文献里其实算 marginal。SPARC 本身对 cutoff frequency $\omega_c$ 敏感 (paper 没说 $\omega_c$ 取多少, 这是 reproducibility 的小坑)。所以我会**保留态度**: 这个改进是统计 significant 但实际是否影响 downstream policy 成功率, 现在没 data。

paper 自己在 conclusion 承认: "Extending this evaluation to downstream policy training and broader embodiments is a natural direction for future work." 这是诚实的, 但也意味着 **整套 pipeline 还没经过 end-to-end policy training 的检验**, 这是目前最大的开放问题。

### 6.2 Object template from a clean frame 的脆弱性

Stage A Step 4 假设视频里**存在一个 clean frame**, 可以从中取 object template。但 dexterous manipulation 视频很多情况是 object 从头到尾都被部分 occlude (比如 long-horizon task), 没有真正 clean 的 frame。这种情况下他们的方法退化成 "use whatever fragments are visible", 这时 ghosting 和 artifact 就回来了。paper 没讨论这个 edge case。

可能的 fix: 多帧的 object mask 取 union, 再做一次 shape completion (用 generative 3D object model 之类), 但这又引入 hallucination 风险。这是个 tension。

### 6.3 Hand reconstruction 在 heavy occlusion 下的崩溃点

HaMeR / WiLoR 这类 3D hand reconstructor 在 hand 被 object 遮住超过 ~30% 时, 重建质量会显著退化 (这是 community 已知问题)。AgenticFocus 不做 hand 重建的任何 correction / temporal smoothing 之外的 robustness enhancement, 所以 heavy occlusion 的 frame 会直接 inject noise 到 IK, 然后 EMA (公式 3) 平滑掉一部分, 但 lag 会引入。

EMA $\alpha$ 的选择是个 trade-off:
- 小 $\alpha$ → 强 smoothing, 抑制 hand reconstructor noise, 但 lag 大, 在快速 manipulation 时 phase 滞后
- 大 $\alpha$ → 弱 smoothing, lag 小, 但 noise 留在 trajectory 里

paper 没给 $\alpha$ 的具体值和 ablation。

### 6.4 Fixed wrist orientation correction 的隐藏假设

公式 (2) 是 camera-relative, 但 wrist 的 mounting orientation correction 是 fixed。这隐含假设: robot hand 装在 arm 上的角度是 constant, 且 human palm 朝向能稳定估计。后者在 hand 部分 occlude 时也会崩。所以这套 retargeting 的 robustness 在 heavy occlusion 下还是会下降。

### 6.5 BrainCo hand 的 DoF mismatch

Human hand 有 27 DoF (左右) 之类, BrainCo dexterous hand 通常 6-12 DoF (具体型号要看)。从 MANO 到 BrainCo 的 mapping 一定丢 information, 这是 cross-embodiment retargeting 的 fundamental loss。paper 没讨论这个 mapping 是怎么做的 (是否用 DexPilot 之类的方法, 还是简单 joint-to-joint)。

### 6.6 与 diffusion-based 方法的对比可能不公平

Masquerade 和 Do as I Do 用 generative editing, 它们能处理 AgenticFocus 处理不了的事 (比如 robot 之外的 scene 变化, 或者 robot 的 shadow / lighting match)。AgenticFocus 的 layered compositing 在 visual realism 上可能更差 (robot hand 看起来"贴"在 scene 上), 但更 deterministic。对下游 policy 是否更好, 取决于 policy 是否对 lighting / shadow 敏感。这一点要在 sim-to-real policy 实验里才能定论。

### 6.7 一个 positive note: pipeline 的可组合性

这个 pipeline 的设计 modular 得很好:
- VLM (selectable)
- SAM2 (replaceable)
- E2FGVI (replaceable)
- HaMeR/WiLoR (replaceable)
- MuJoCo IK + render (replaceable)

任何一段都可以替换成更新的 model。比如把 SAM2 换成 SAM3, 把 HaMeR 换成更新版, pipeline 不动。这对维护和迭代友好, 也意味着这个 pipeline 的 quality 会随 sub-component 进步而自动变好。

---

## 7. Related work landscape (build your mental map)

这是我对这一片工作的归类:

**Hardware-heavy collection (high quality, low scalability):**
- DexCap [22] (https://dexcap.github.io/): wearable mocap, 高质量, 但 hardware 限制大

**Rich egocentric sensing (medium hardware):**
- EgoDex [6]: 多模态 egocentric setup for dexterous learning
- EgoEngine [12] (https://arxiv.org/abs/2606.12604): egocentric → high-fidelity dexterous demo

**Cross-embodiment video editing (generative):**
- Masquerade [9] (https://arxiv.org/abs/2406.07788): data-editing from in-the-wild human video
- Do as I Do [17] (https://arxiv.org/abs/2606.19333): dexterous data from everyday human video
- Bridging the Embodiment Gap [11]: disentangled cross-embodiment video editing

**Mixed Reality for robot data (deterministic):**
- ARMADA [15] (https://arxiv.org/abs/2412.10631): AR for robot manipulation data acquisition
- **AgenticFocus** (this paper)

**VLA / World-Action models (the consumer of this data):**
- RT-2 [3], OpenVLA [8] (https://openvla.github.io/), World Action Models [23]

AgenticFocus 在 landscape 里的位置: 它和 ARMADA 同属 "MR for robot data" 路线, 但比 ARMADA 更 focused 在 dexterous + contact-rich 场景。它和 Masquerade / Do as I Do 的差异在于 deterministic vs generative, camera-relative formulation, 和 layered compositing。

---

## 8. Open questions / Future directions (paper 自己提的 + 我加的)

**Paper 自己提的:**
1. Downstream policy training 评估 (目前没做)
2. 扩展到 broader embodiments

**我会加的:**
3. **Long-horizon tasks**: 多个 object 切换的场景, 现在 VLM 选 task-relevant object 假设 single primary object
4. **Bimanual manipulation**: 现在 pipeline 默认 single hand, 双手协作场景需要扩展
5. **Object template learning**: 用 generative 3D object prior (比如 3DGP, Object-3DIT) 在没有 clean frame 时完成 object geometry, 同时不引入过强 hallucination
6. **Hand reconstruction refinement**: 用 video diffusion 或 temporal optimization 在 occlusion 下 refine hand pose (但要 deterministic, 避免 hallucination)
7. **Policy robustness study**: MR-synthesized data 训出来的 policy 在 sim-to-real 时是否比 diffusion-synthesized data 更 robust? 这是个 open empirical question
8. **Active object state**: 现在 object template 是 static, 但 manipulation 中 object state 变化 (e.g., 门打开, 抽屉拉出), 需要 state-aware object template
9. **Force / tactile supervision**: 现在 supervision 只有 vision + action, 没有 force。如果能从 hand-object interaction 推断 contact force, 加进 supervision, 对 reactive control 更有用

---

## 9. 最 takeaway 的一点

如果只让我记一件事, 那就是 **公式 (1) 的 set operation**:

$$\mathcal{M}_{\mathrm{inpaint}} = \mathcal{M}_{\mathrm{hand}} \cap \neg \mathcal{M}_{\mathrm{obj}}$$

这个 mask 把 "preserve object geometry under occlusion" 这个抽象意图, 直接编码成 3 个像素级集合操作。它不做"重建 object 的 occluded 部分"这个难题 (那是 generative model 的活), 它只做"别在擦 hand 的时候把 object 也擦了"这件事。这是个**子问题分解**的好例子——把 hard problem (reconstruct occluded object) 退化成 easier problem (don't destroy object you can see, then composite from trusted template)。

这种"avoid generative where deterministic works"的设计哲学, 和你 Karpathy 平时讲的"让数据说话、能 deterministic 就别 hallucinate"是同一种直觉。在 Physical AI 这种 safety-critical 又 data-starved 的领域, 这种保守主义可能反而是优势。

---

如果你想深挖某一段 (比如 IK 在 MuJoCo 里的具体 formulation, 或者 SPARC 的 frequency-domain 直觉, 或者 layered compositing 的具体 render 顺序), 告诉我哪段, 我再展开。
