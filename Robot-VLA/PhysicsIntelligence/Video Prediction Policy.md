---
source_pdf: Video Prediction Policy.pdf
paper_sha256: 7a6d966b78a205efe0943e9188f16ec33e470526da1066df5406922fafb4f0d6
processed_at: '2026-08-13T00:35:30-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 VPP

## 一句话版本

你拿一个会"做梦"的video model，让它看一眼当前画面就开始"幻想"未来16帧会怎么演变，然后让robot policy去"追踪"这个幻想里的robot arm怎么动，就这么简单。

---

## 为什么传统方法不行

Andrej，你想想现在robot policy的标准配方长啥样。一个camera拍一帧图，扔给一个vision encoder（可能是CLIP、R3M、VC-1、Voltron之类），encoder吐一个feature vector，policy head拿这个vector输出action。就这么个pipeline。

问题在哪？**这些encoder都是在"认东西"，不是在"懂物理"**。VC-1用MAE重建single image，学到的是"这个图里有个杯子"——这是semantic info。R3M用two-image contrastive，学到的是"这两帧有点像"——这是weak temporal info。它们都不回答一个robot真正关心的问题：**"如果我执行action A，世界会变成什么样？"**

这个missing piece就是dynamic info。robotics本质是个sequential decision making problem，你给的encoder却是static的，这就是错配。

https://europe-nlp1.github.io/vc-1/

---

## VDM 为什么神奇

最近两年video diffusion models爆发了，Sora、SVD、Kling这些model能生成非常physically plausible的视频。你给它一个初始帧加一句prompt"把杯子推到左边"，它真能生成robot arm把杯子推过去的视频。

这说明啥？**VDM内部已经encode了physical dynamics的prior**。它知道物体怎么动、arm怎么抓、东西倒了会怎样。这个knowledge是从internet海量video里学来的。

更关键的观察：VDM的internal latent有个非常特殊的结构，shape是 **(T, H, W, C)**——T帧的时间维度。也就是说，它的latent天然就表示"1帧current + (T-1)帧future"的序列。这跟传统encoder输出的single-frame feature完全不同。

作者把这种latent叫 **"predictive visual representation"**。我觉得这名字起得很好，因为它点出了本质：这不是一个static snapshot，是一个对未来的预测。

https://stability.ai/news/stable-video-distribution-open-video-model

---

## 核心insight：Tracking就完了

好了，现在你有一个能预测未来的representation，怎么用它来控制robot？

这里有个非常漂亮的insight。想想inverse dynamics这个概念：

**Forward dynamics**：给定当前state $s_t$ 和 action $a_t$，预测下一state $s_{t+1}$。这是physics engine做的事。
**Inverse dynamics**：给定当前state $s_t$ 和 目标state $s_{t+1}$，反推该执行什么action $a_t$。

如果你有一个oracle能告诉你未来state序列 $s_{t+1}, s_{t+2}, ..., s_T$，那policy learning就变得超简单——只需要学一个inverse dynamics：观察"未来robot arm从位置A移到了位置B"，输出对应的那段action就行了。

VDM就是这个oracle。它预测的未来representation里，robot arm的轨迹是隐含在其中的。policy只需要学会**"在predicted future里track robot arm怎么动"**，就能反推出action。

为什么这个思路能generalize到新物体、新场景？因为policy只需要关注robot arm本身的运动，不关心是抓杯子还是抓球。只要VDM能正确预测未来（依靠internet-scale pretrain），policy就能在新场景work。

这个idea非常elegant，把policy learning从"理解physics + 生成action"简化为"track motion + 反推action"。

---

## 方法怎么实现

### Stage 1: 把VDM变成manipulation专家

作者用Stable Video Diffusion (SVD, 1.5B参数)作为base。原始SVD只condition在初始帧上，作者加了两个mod：

1. **加CLIP text embedding**：通过cross-attention注入language instruction。原始SVD没text condition，加这个是为了support instruction-conditioned policy。
2. **改output resolution到16×256×256**：从SVD默认的14或25帧改成16帧，效率更高。

然后在这个modified model上做fine-tune，loss就是标准diffusion loss：

$$\mathcal{L}_D = \mathbb{E}_{x_0 \sim D, \epsilon, t} \| V_\theta(x_t, l_{emb}, s_0) - x_0 \|^2$$

这里 $V_\theta$ 是modified SVD，$x_0$ 是ground truth video，$x_t$ 是加噪后的video，$l_{emb}$ 是CLIP text embedding，$s_0$ 是initial frame作为condition。

注意SVD用的是 **predict x0** 而不是predict noise的formulation，这通常更stable。

数据上用了三mix：
- **Internet human manipulation** (Something-Something-v2, 191k traj, 30%比例)：physical dynamics最丰富
- **Internet robot data** (RT-1, Bridge, BC-Z等, 175k traj, 50%比例)：domain knowledge
- **Self-collected** (Panda arm + Dexterous hand, 4.5k traj, 15%比例)：downstream alignment

训练2-3天 on 8×A100。

https://github.com/Swimmerable/CALVIN

### Stage 2: 用VDM当encoder，不用当generator

这是最聪明的地方。正常用VDM要denoise 30步，慢得要死，根本做不了closed-loop control。

作者的观察是：**第一个forward pass就够了**。

你输入pure noise $x_{t'}$ + initial frame $s_0$，做一次forward pass through $V_\theta$。输出不是干净视频，但是intermediate feature已经包含了rough的未来trajectory信息——物体往哪动、arm怎么移这种coarse info都已经在里面了。

具体怎么提取？取up-sampling layers的intermediate feature：

$$L_m = V_\theta(x_{t'}, l_{emb}, s_0)_{(m)}$$

$m$ 是第 $m$-th up-sampling layer，每层feature shape是 $(T, C_m, W_m, H_m)$。

然后不同layer捕捉不同granularity的信息，low layer抓texture，high layer抓semantic。作者的做法是把所有up-sampling layer的feature spatial resize到统一size，然后channel-wise concat：

$$F_p = \text{concat}(L_0', L_1', ..., L_m', \text{dim}=1)$$

这个 $F_p$ 就是最终的 **predictive visual representation**，shape是 $(T, \sum_m C_m, W_p, H_p)$。

multi-view场景（static camera + wrist camera）就独立predict两个view的 $F_p$，得到 $F_p^{static}$ 和 $F_p^{wrist}$。

https://diffusionhyperfeatures.github.io/

### Video Former: 压缩高维feature

$F_p$ 维度太高，直接喂给policy head不行。作者设计了一个Video Former来compress：

1. 初始化learnable tokens $Q$，shape是 $T \times L$（比如Calvin用 $16 \times 14 \times 384$）
2. 每帧做spatial attention，聚合static和wrist两个view的信息：

$$Q' = \{\text{Spat-Attn}(Q[i], (F_p^{static}[i], F_p^{wrist}[i]))\}_{i=0}^{T}$$

3. 跨帧做temporal attention + FFN：

$$Q'' = \text{FFN}(\text{Temp-Attn}(Q'))$$

输出 $Q''$ 是fixed-length tokens，这就压缩完了。

temporal attention重要，因为不同帧间的motion info需要聚合。ablation显示去掉temporal attention从4.33掉到4.18。

### Action Head: Diffusion Policy

最后一步，用 $Q''$ 作为condition，跑一个Diffusion Transformer (DiT)风格的diffusion policy输出action。

loss是标准diffusion policy loss：

$$\mathcal{L}_{diff}(\psi; A) = \mathbb{E}_{a_0, \epsilon, k} \| D_\psi(a_k, l_{emb}, Q'') - a_0 \|^2$$

$a_0$ 是ground truth action sequence（用action chunking，10步chunk），$a_k$ 是加噪后的action，$D_\psi$ 是DiT denoiser，$Q''$ 通过cross-attention注入。

---

## 为什么single forward pass就够了

这是你可能会问的问题：一次forward pass的feature真的够用吗？你看Figure 4里one-step prediction的图，texture全糊了，只剩coarse motion info。

答案是：**够用，因为policy只需要track coarse motion**。

policy的任务是"robot arm从位置A移到位置B该发什么action"，它不需要知道杯子上的花纹长啥样。粗略的motion trajectory就足以support inverse dynamics learning。

ablation也证实了这点：
- 1-step forward: Avg Len 4.33
- 2-step denoise: Avg Len 4.19

多denoise一步反而没提升，还double了latency。所以1-step是sweet spot。

https://diffusionhyperfeatures.github.io/

---

## 结果有多炸

### Calvin ABC→D (long-horizon generalization benchmark)

这个benchmark要求robot连续完成5个chained task，而且测试在unseen environment D。previous SOTA是RoboUniview 3.65：

| Method | Avg Len |
|--------|---------|
| RT-1 | 0.90 |
| Diffusion Policy | 0.56 |
| Robo-Flamingo | 2.47 |
| Uni-Pi | 0.92 |
| MDT | 1.55 |
| SuSIE | 2.69 |
| GR-1 | 3.06 |
| Vidman | 3.42 |
| RoboUniview | 3.65 |
| **VPP** | **4.33** |

**18.6% relative improvement over previous SOTA**。Task5的success rate从0.507涨到0.769，这意味着long-horizon能力极强。

更夸张的是，只用10%的Calvin ABC数据，VPP还能达到3.25，超过GR-1用100%数据的3.06。这说明video pretrain的prior非常强，少量downstream data就能align。

### 真实世界 Dexterous Hand (12-DOF)

这个最能体现VPP的价值。在tool-use tasks上：

| Tool | DP | SuSIE | GR-1 | VPP |
|------|-----|-------|------|-----|
| Spoon | 0.0 | 0.4 | 0.3 | **0.9** |
| Hammer | 0.2 | 0.2 | 0.1 | **0.6** |
| Drill | 0.0 | 0.1 | 0.2 | **0.8** |
| Pipette | 0.0 | 0.0 | 0.0 | **0.4** |

VPP在tool-use上average 0.68 vs GR-1的0.15，**4.5x improvement**。这是非常惊人的数字。

为什么？因为tool use需要理解"拿着勺子舀液体""拿电钻打孔"这种复杂physical interaction。这些knowledge从哪来？从internet human video里学来。Something-Something-v2里有人用勺子、用锤子的video，VDM把这些physical prior encode了，policy只需要track motion就行。

https://video-prediction-policy.github.io/

---

## 跟同类方法比

### vs UniPi / SuSIE

UniPi和SuSIE也是"先预测未来再控制"的思路，但它们：
1. 只predict single future frame或keyframe，丢掉了intermediate dynamics
2. 完整跑denoising，慢，导致open-loop control

VPP：16帧predictive representation + single forward pass = closed-loop 7-10Hz。

### vs GR-1

GR-1用autoregressive transformer，每步生成1帧+1action。
1. 每forward只1 frame，prediction quality不如diffusion
2. 没用video foundation model pretrain，从零学

VPP：SVD foundation model + diffusion prediction，quality和efficiency都更好。

### vs Vidman

Vidman也用video diffusion model representation，但**没fine-tune video model on downstream tasks**，导致representation不够task-specific。

VPP的关键创新就是fine-tune VDM on manipulation data，让representation对robot task更敏感。

---

## Ablation告诉了我们什么

最informative的几个ablation：

**Vision encoder对比**:
| Encoder | Avg Len |
|---------|---------|
| VDM (VPP) | 4.33 |
| Stable-VAE | 2.58 |
| VC-1 | 1.23 |
| Voltron | 1.54 |

VDM vs VC-1是 **3.5x** improvement。这直接证明predictive representation远好于static reconstruction representation。

**去掉video pretrain和internet data**:
从4.33掉到1.63。from scratch训练几乎失败。这告诉你internet-scale video pretrain是必要条件，不是锦上添花。

**Video Former**:
去掉从4.33掉到3.86，但更关键的是latency从140ms涨到450ms，**3x变慢**。Video Former既是performance boost又是efficiency hack。

**Single-view vs Multi-view**:
Single-view VPP还能达到3.58，超过RoboUniview (3.65)用两个view+depth的3D方法。这说明predictive representation本身就非常strong。

---

## 我觉得最cool的地方

1. **Conceptual elegance**: 把policy learning从"学physics + 生成action"简化为"track motion + 反推action"。这个idea非常clean。

2. **Efficiency trick**: single forward pass提取predictive representation，绕开了diffusion model的慢推理问题。140ms latency在real robot上用毫无压力。

3. **Generalization mechanism**: VDM负责理解physics + 预测未来，policy只负责track arm motion。分工明确，generalization来自VDM的internet pretrain。

4. **跟Sora的哲学一致**: OpenAI说"video models are world simulators"，VPP就是把这个idea在robotics上concrete实现了。未来这条路如果能走通，就是"大规模video pretrain → 通用robot policy"，可能不需要任何robot demonstration。

https://openai.com/research/video-generation-models-as-world-simulators

---

## 我觉得可能有问题的点

1. **TVP training cost**: 2-3天on 8×A100，academic lab跑不起。需要open release pretrain checkpoint。

2. **Inverse dynamics assumption**: 如果VDM预测错了（hallucinate了错误的未来），policy会跟着错。没有error correction机制。

3. **Video length 16的limit**: 16帧大概1-2秒的future，long-horizon planning还得靠autoregressive或hierarchical structure。

4. **Texture丢失**: single forward pass的feature很糊，如果task需要精细visual info（比如辨认小零件朝向），可能不够。但ablation显示对当前task suite够用。

5. **Multi-view扩展**: 现在是independent predict每个view，没有cross-view consistency constraint。如果view多了会expensive。

---

## 延伸思考

### World Model + Planning
VPP的predictive representation本质是个world model的internal state。完全可以接MPC或tree search做planning。想象一下：predict多个candidate action的未来，选最好的那个执行。

### VLM grounding
现在用CLIP text embedding，如果换成LLaMA这种strong LLM的token，instruction following能力可能大幅提升。特别是long-horizon task需要复杂instruction理解时。

### Self-supervised refinement
用robot实际执行轨迹作为额外supervision，online refine TVP model。这样VDM的prediction会越来越准，policy也越来越好。

### Cross-embodiment transfer
VDM不关心embodiment（它是看video的），所以理论上VPP能zero-shot transfer across embodiments。Franka学的skill能不能直接transfer到XHand？这是个非常值得explore的方向。

### Latent action pretraining
结合LAPA (Ye et al., 2024)的latent action pretraining，可能实现完全self-supervised的video → action learning，不需要任何action label。这跟VPP的结合点很多。

https://latentactionpretraining.github.io/

---

## 最直觉的类比

想象你学开车。传统方法是：你得先学physics——轮胎摩擦力、刹车距离、转向角度跟车速的关系，然后基于这些physics model去决策。

VPP的方法是：你脑子里有个"video prediction model"（其实就是你的imagination），你想象一下"如果我转方向盘，车会怎么走"，然后你just让手跟着这个想象走。你不需要精确知道physics，你的imagination已经encode了所有必要的physics prior。

这就是VPP的精髓。**让imagination（VDM）做physics，让policy做tracking**。

---

希望这个"人话版"帮你build了intuition，Andrej。如果你想深挖某个点，比如"为什么single forward pass的rough feature足够support inverse dynamics"，或者"Video Former的具体attention pattern设计"，咱们可以继续聊。

---

# Video Prediction Policy (VPP) 深度解析

## 1. Paper的核心Motivation

让我从最根本的问题开始build你的intuition。

### 1.1 传统Vision Encoder的局限

Andrej，你看现在的robotic policy stack，几乎所有方法都用类似的recipe：**image → vision encoder → latent → policy head → action**。vision encoder这一层，社区做了很多工作：

- **VC-1** (Majumdar et al., 2023): MAE reconstruction，single image
- **R3M** (Nair et al., 2022): two-image contrastive + language
- **VIP** (Ma et al., 2022): value-implicit contrastive
- **Voltron** (Karamcheti et al., 2023): MAE + language generation

这些方法的共同问题：**它们都基于单帧或帧对学习representation**。一个static snapshot能告诉你"场景里有什么"，但告诉不了你"接下来物理世界会如何演化"。对于embodied task，你需要知道：如果我推这个杯子，它会往哪倒？如果我抓住这个把手，抽屉会怎么开？这些是dynamic information。

reference: https://europe-nlp1.github.io/vc-1/

### 1.2 Video Diffusion Models的predictive property

最近video diffusion models（Sora, SVD, Kling等）展现出惊人的physical understanding。作者的关键hypothesis是：

**VDMs的internal latent representation天然包含 T 帧（1帧current + (T-1)帧predicted future）的结构化信息**，shape为。这个latent variable我称之为"predictive visual representation"。

对比一下：
- 传统encoder输出：，只有spatial info
- VDM latent：，有spatial + temporal predictive info

reference: https://stability.ai/news/stable-video-distribution-open-video-model

### 1.3 Core Insight：Inverse Dynamics via Tracking

这里是最关键的intuition。考虑policy learning的两种formulation：

**Forward dynamics**: $s_{t+1} = f(s_t, a_t)$ — 给定state和action，预测next state
**Inverse dynamics**: $a_t = g(s_t, s_{t+1})$ — 给定current state和future state，推算action

如果有一个oracle告诉你future state $s_{t+1}, s_{t+2}, ..., s_T$，那么policy只需要学一个inverse dynamics：观察current state和predicted future state之间robot arm的运动差异，输出对应的action。

VPP的核心insight就是：**VDM的predictive representation隐式给出了future state，下游policy只需要学一个inverse dynamics去"track"robot arm在predicted future中的位置变化**。

这就是为什么VPP能泛化：对新object、新background，只要VDM能正确预测future（依靠internet-scale pretraining），inverse dynamics模型只需要关注robot arm本身的运动，不需要理解新object的物理特性。

---

## 2. 方法细节

### 2.1 Stage 1: Manipulation TVP Model

#### 2.1.1 Base Model与Modification

Base是 **Stable Video Diffusion (SVD)**，1.5B参数。原始SVD只condition在initial frame $s_0$上。作者做了两个modification：

1. **添加CLIP language feature $l_{emb}$**：通过cross-attention layer注入text instruction
2. **调整output resolution**：16×256×256（原始SVD是14帧或25帧）

modified model记为 $V_\theta$。conditioning方式：将initial observation $s_0$ channel-wise concat到每个predicted frame上。

#### 2.1.2 训练Loss

公式(3):
$$\mathcal{L}_D = \mathbb{E}_{x_0 \sim D, \epsilon, t} \| V_\theta(x_t, l_{emb}, s_0) - x_0 \|^2$$

变量解释：
- $x_0 = s_{0:T}$: ground truth video sequence，包含T+1帧（initial frame + T predicted frames）
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$: noised video
- $\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$: cumulative noise schedule
- $\epsilon \sim \mathcal{N}(0, I)$: Gaussian noise
- $l_{emb}$: CLIP text embedding
- $s_0$: initial frame observation

注意这里 $V_\theta$ 是预测clean video $x_0$ 而不是预测noise $\epsilon$，这是SVD采用的方式（"x0-prediction"而不是"epsilon-prediction"），可能更稳定。

#### 2.1.3 Multi-dataset Mixture

公式(4):
$$\mathcal{L}_{video} = \lambda_H \mathcal{L}_{D_H} + \lambda_R \mathcal{L}_{D_R} + \lambda_C \mathcal{L}_{D_C}$$

三个数据源：
- $D_H$: Internet human manipulation (Something-Something-v2)
- $D_R$: Internet robot data (RT-1, Bridge, BC-Z, etc.)
- $D_C$: Self-collected downstream task data

详细mixture ratios (Table 8):
| Dataset | #Traj | Ratio |
|---------|-------|-------|
| Something-Something-v2 | 191,642 | 0.30 |
| RT-1 | 87,212 | 0.15 |
| Bridge | 23,377 | 0.15 |
| BC-Z | 43,264 | 0.08 |
| Calvin-ABC | 18,033 | 0.10 |
| Dexterous Hand | 2,476 | 0.10 |
| Metaworld | 2,500 | 0.05 |
| Panda Arm | 2,000 | 0.05 |
| Taco-Play | 3,603 | 0.01 |
| Jaco-Play | 1,085 | 0.01 |
| **Total** | **375,192** | **1.00** |

intuition: Something-Something-v2 占30%因为它是human manipulation中physical dynamics最丰富且scale最大的。RT-1和Bridge各占15%是robot数据的backbone。Self-collected data (Panda + Dexterous Hand) 占15%用于domain alignment。

reference: https://github.com/Swimmerable/CALVIN

### 2.2 Stage 2: Action Learning with Predictive Representation

#### 2.2.1 单次Forward Pass提取Predictive Representation

这里是最有创意的地方。VDM正常推理需要30+ step denoising，速度很慢。作者的观察是：

**diffusion model的第一个forward pass（输入pure noise + condition），虽然没有干净的视频输出，但已经隐含rough的future trajectory信息。**

实际做法：
1. 准备 $x_{t'}$：纯白噪声（最noisy的state）
2. Concat $s_0$（initial observation）channel-wise
3. Input到 $V_\theta$ 做一次forward
4. 提取up-sampling layers的中间feature

公式:
$$L_m = V_\theta(x_{t'}, l_{emb}, s_0)_{(m)}, \quad L_m \in \mathbb{R}^{T \times C_m \times W_m \times H_m}$$

变量解释：
- $m$: 第 $m$-th up-sampling layer
- $T$: video length (e.g., 16 for Calvin)
- $C_m$: channel dim of layer $m$
- $W_m, H_m$: spatial dim of layer $m$

#### 2.2.2 Feature Aggregation Across Layers

不同layer捕捉不同level的信息（low-layer抓texture，high-layer抓semantic）。作者采用interpolation + concat策略：

$$L_m' = \text{Interpolation}(L_m), \quad L_m' \in \mathbb{R}^{T \times C_m \times W_p \times H_p}$$

把每个layer spatial resize到统一的 $W_p \times H_p$，然后：

$$F_p = \text{concat}((L_0', L_1', ..., L_m'), dim=1) \in \mathbb{R}^{T \times (\sum_m C_m) \times W_p \times H_p}$$

最终 $F_p$ 的channel dim是所有up-sampling layer的channel sum。

ablation表明 Layer-9 单独使用最好(4.29)，但concat所有层达到4.33，说明multi-scale feature确实有用。

reference: https://diffusionhyperfeatures.github.io/

#### 2.2.3 Multi-view Extension

对于有多个camera view的设置（static view + wrist view）：
- 独立predict两个view的future
- 得到 $F_p^{static}$ 和 $F_p^{wrist}$

#### 2.2.4 Video Former

$F_p$ 维度太高，需要compress成fixed-length tokens。

Video Former结构：
1. 初始化learnable tokens $Q_{[0:T, 0:L]}$，shape为 $T \times L$（如Calvin中是16×14×384）
2. 对每帧做spatial attention：

$$Q' = \{\text{Spat-Attn}(Q[i], (F_p^{static}[i], F_p^{wrist}[i]))\}_{i=0}^{T}$$

3. 对时序维度做temporal attention + FFN：

$$Q'' = \text{FFN}(\text{Temp-Attn}(Q'))$$

这是借鉴了Blattmann的Spatial-Temporal Attention机制（LDM video paper）。

reference: https://arxiv.org/abs/2304.08818

#### 2.2.5 Diffusion Policy Head

最终action head是 **Diffusion Transformer (DiT)** 风格的diffusion policy（Reuss et al. 2024 MDT思路）：

公式(6):
$$\mathcal{L}_{diff}(\psi; A) = \mathbb{E}_{a_0, \epsilon, k} \| D_\psi(a_k, l_{emb}, Q'') - a_0 \|^2$$

变量：
- $a_0$: ground truth action sequence (action chunking, 10 steps)
- $a_k = \sqrt{\bar{\beta}_k} a_0 + \sqrt{1-\bar{\beta}_k} \epsilon$: noised action
- $\bar{\beta}_k$: action diffusion noise schedule
- $Q''$: aggregated predictive representation tokens
- $l_{emb}$: CLIP language feature
- $D_\psi$: DiT denoiser network

$Q''$ 通过cross-attention注入到DiT blocks里。

---

## 3. 实验结果深度分析

### 3.1 Calvin ABC→D 主结果 (Table 1)

Calvin ABC→D 是long-horizon的generalization benchmark：训练在ABC environment，测试在unseen D environment，要求连续完成5个chained tasks。

| Method | Task1 | Task2 | Task3 | Task4 | Task5 | Avg Len |
|--------|-------|-------|-------|-------|-------|---------|
| RT-1 | 0.533 | 0.222 | 0.094 | 0.038 | 0.013 | 0.90 |
| Diffusion Policy | 0.402 | 0.123 | 0.026 | 0.008 | 0.000 | 0.56 |
| Robo-Flamingo | 0.824 | 0.619 | 0.466 | 0.331 | 0.235 | 2.47 |
| Uni-Pi | 0.560 | 0.160 | 0.080 | 0.080 | 0.040 | 0.92 |
| MDT | 0.631 | 0.429 | 0.247 | 0.151 | 0.091 | 1.55 |
| SuSIE | 0.870 | 0.690 | 0.490 | 0.380 | 0.260 | 2.69 |
| GR-1 | 0.854 | 0.712 | 0.596 | 0.497 | 0.401 | 3.06 |
| Vidman | 0.915 | 0.764 | 0.682 | 0.592 | 0.467 | 3.42 |
| RoboUniview | 0.942 | 0.842 | 0.734 | 0.622 | 0.507 | 3.65 |
| **VPP** | **0.965** | **0.909** | **0.866** | **0.820** | **0.769** | **4.33** |

VPP比SOTA RoboUniview提升18.6% relative improvement。最impressive的是Task5的success rate从0.507 → 0.769，这意味着long-horizon capability极强。

10% ABC data ablation: VPP在只用10%数据时仍达3.25，超过GR-1用100%数据的3.06，说明video pretraining极强。

### 3.2 MetaWorld结果 (Table 2)

50个task统一policy：

| Method | Easy(28) | Middle(11) | Hard(11) | Average |
|--------|----------|------------|----------|---------|
| RT-1 | 0.605 | 0.042 | 0.015 | 0.346 |
| Diffusion Policy | 0.442 | 0.062 | 0.095 | 0.279 |
| SuSIE | 0.560 | 0.196 | 0.255 | 0.410 |
| GR-1 | 0.725 | 0.327 | 0.451 | 0.574 |
| **VPP** | **0.818** | **0.493** | **0.526** | **0.682** |

Hard task上VPP 0.526 vs GR-1 0.451，相对提升16.6%。

### 3.3 真实世界实验

#### Franka Panda (Table 6 detail):
| Task | DP | SuSIE | GR-1 | VPP | (Seen/Unseen) |
|------|-----|-------|------|-----|----------------|
| Pick | 0.36/0.24 | 0.56/0.40 | 0.52/0.32 | 0.90/0.80 | |
| Place | 0.40/0.12 | 0.42/0.44 | 0.38/0.32 | 0.86/0.72 | |
| Press | 0.65/0.50 | 0.90/0.60 | 0.80/0.60 | 0.85/0.80 | |
| Route | 0.40/0.20 | 0.55/0.50 | 0.50/0.50 | 0.75/0.70 | |
| Drawer | 0.45/0.40 | 0.60/0.50 | 0.60/0.40 | 0.85/0.60 | |

#### 12-DOF Dexterous Hand (Table 7 detail):

最impressive的是tool-use tasks:
| Tool | DP | SuSIE | GR-1 | VPP |
|------|-----|-------|------|-----|
| Spoon | 0.0 | 0.4 | 0.3 | **0.9** |
| Hammer | 0.2 | 0.2 | 0.1 | **0.6** |
| Drill | 0.0 | 0.1 | 0.2 | **0.8** |
| Pipette | 0.0 | 0.0 | 0.0 | **0.4** |

VPP在tool-use上达到0.68 vs GR-1的0.15，提升4.5x。这强烈说明video pretraining带来的physical understanding至关重要。

### 3.4 FVD指标 (Table 9)

Bridge dataset上的Fréchet Video Distance：

| Method | FVD↓ |
|--------|------|
| VideoFusion | 501.2 |
| Tune-A-Video | 515.7 |
| Seer | 246.3 |
| **VPP** | **41.4** |

VPP的video generation质量比Seer好6x，归功于SVD pretrain。

---

## 4. Ablation深度分析

### 4.1 Vision Encoder对比 (Table 3)

| Encoder | Pre-training | Avg Len |
|---------|--------------|---------|
| **VDM (ours)** | **Video Generation** | **4.33** |
| Stable-VAE | VAE Reconstruction | 2.58 |
| VC-1 | MAE Reconstruction | 1.23 |
| Voltron | MAE + Language Gen | 1.54 |

VPP vs VC-1: 4.33 vs 1.23，3.5x improvement。这直接证明predictive representation远好于static reconstruction representation。

### 4.2 Video Pretraining + Internet Data (Table 4)

| Configuration | Avg Len | Latency |
|---------------|---------|---------|
| VPP (full) | 4.33 | ~140ms |
| w/o Internet data | 3.97 | ~140ms |
| w/o Calvin video | 3.31 | ~140ms |
| w/o Internet data + w/o SVD Pretrain | 1.63 | ~140ms |
| w/o Video Former | 3.86 | ~450ms |
| w/o Feature Agg | 3.60 | ~140ms |

Key insights:
- **Internet data贡献+0.36** (4.33→3.97): human manipulation data带来physical prior
- **Calvin video贡献+1.02** (4.33→3.31): domain-specific video至关重要
- **SVD pretrain + Internet贡献+2.70** (4.33→1.63): from scratch几乎失败，证明internet-scale video pretrain是必要条件
- **Video Former贡献+0.47**: 不仅是performance，更关键latency从140ms→450ms，3x加速
- **Feature Agg贡献+0.73**: multi-scale feature有用

### 4.3 Layer Ablation (Table 10)

| Layer | Avg Len |
|-------|---------|
| Layer-3 | 3.72 |
| Layer-6 | 3.88 |
| Layer-9 | 4.29 |
| Layer-12 | 4.05 |
| All (VPP) | 4.33 |

Layer-9单层最好（4.29），但concat所有层能达到4.33，证明multi-scale aggregation有用。

### 4.4 Denoising Step Ablation (Table 11)

| Time-step | Avg Len |
|-----------|---------|
| 10 | 4.21 |
| 20 | 4.33 |
| 30 | 4.25 |

差异不大，说明one-step representation已经足够。

### 4.5 Single-view vs Multi-view (Table 12)

| Setting | Task1 | Task2 | Task3 | Task4 | Task5 | Avg Len |
|---------|-------|-------|-------|-------|-------|---------|
| VPP (full) | 0.965 | 0.909 | 0.866 | 0.820 | 0.769 | 4.33 |
| Single-view | 0.909 | 0.815 | 0.713 | 0.620 | 0.518 | 3.58 |

Single-view VPP的3.58已经超过RoboUniview (3.65)的3D方法！这说明predictive representation本身非常powerful。

### 4.6 Video Former Ablation

| Ablation | Avg Len |
|----------|---------|
| VPP | 4.33 |
| Ablation 1 (w/o Temp-Attn) | 4.18 |
| Ablation 2 (2-step denoise) | 4.19 |

Temporal attention在Video Former里贡献+0.15。

---

## 5. Hyperparameter Details (Table 13)

| Config | Calvin | MetaWorld | Franka | XHand |
|--------|--------|-----------|--------|-------|
| Video len | 16 | 8 | 16 | 16 |
| Action shape | 10×7 | 4×4 | 10×7 | 10×18 |
| Language shape | 20×512 | 20×512 | 20×512 | 20×512 |
| Video Former tokens | 16×14×384 | 8×28×384 | 14×16×384 | 14×16×384 |
| Diff. Transformer steps | 4 | 10 | 10 | 4 |
| TVP batchsize | 10 | 4 | 4 | 10 |
| Policy batchsize | 76 | 64 | 128 | 128 |
| Epoch nums | 12 | 30 | 30 | 40 |
| Learning rate | 1e-4 | 5e-5 | 1e-4 | 1e-4 |

---

## 6. 与Related Works深度对比

### 6.1 UniPi (Du et al., 2024)

UniPi: 先generate future video，然后学inverse dynamics从两个frames间。问题：
1. 只用single future step，物理dynamics不完整
2. 完整denoising process慢，导致open-loop control

VPP: 用single forward pass提取整段future representation，闭环7-10Hz。

reference: https://universal-policy.github.io/

### 6.2 SuSIE (Black et al., 2023)

SuSIE: 用Instruct-Pix2Pix生成single future keyframe，然后condition policy。

问题：仍是single frame，丢失intermediate dynamics。

VPP: 16帧predictive representation，完整temporal info。

### 6.3 GR-1 (Wu et al., 2023a)

GR-1: autoregressive transformer，每step生成1帧+1 action。

问题：
1. 每forward只1 frame，prediction quality不如diffusion
2. 没用video foundation model pretrain

VPP: 用SVD foundation model + diffusion prediction。

reference: https://unleashing-large-scale-video.github.io/

### 6.4 Vidman (Wen et al., 2024)

Vidman: 类似思路用video diffusion model representation，但没fine-tune video model on downstream tasks，导致sub-optimal。

VPP: 关键创新是fine-tune VDM on manipulation data，让representation更task-specific。

reference: https://arxiv.org/abs/2410.11758

---

## 7. 我的Critique与延伸思考

### 7.1 优点

1. **Intuition清晰**：inverse dynamics tracking这个idea非常elegant，把复杂的policy learning简化为track problem
2. **Efficiency好**：single forward pass，140ms latency，比SuSIE/UniPi快一个数量级
3. **Generalization强**：tool-use tasks上4.5x improvement over GR-1是惊人的
4. **Multi-scale feature aggregation**：借鉴Diffusion Hyperfeatures的思路，自动aggregate所有up-sampling layers

### 7.2 潜在Concerns

1. **TVP model training cost**: 2-3 days on 8×A100，对academic lab不友好
2. **Action chunking依赖**: 10-step action chunking是关键，如果chunk size不合适可能掉performance
3. **Inverse dynamics assumption**: 假设predicted future足够准确，如果VDM预测错误（hallucination），policy也会跟着错
4. **Single forward pass的representation quality**: Figure 4显示one-step prediction很模糊，texture几乎全丢，能work的核心是位置信息保留，但如果task依赖texture细节可能失败
5. **Video length 16的limitation**: 16帧只能预测短期future，长horizon planning仍需autoregressive

### 7.3 延伸方向

1. **World Model + Planning**: VPP的predictive representation本质是world model的internal state，可以接MPC或tree search
2. **VLM grounding**: 把CLIP text embedding换成更strong的LLM token（如LLaMA token），可能获得更好的instruction following
3. **Self-supervised refinement**: 用robot的actual execution轨迹作为额外监督，online refine TVP model
4. **Cross-embodiment transfer**: 现在Franka和XHand是separate训练，理论上VPP能transfer across embodiments（因为video model不关心embodiment），可以探索zero-shot cross-embodiment
5. **3D-aware extension**: 结合RoboUniview的3D occupancy loss，predictive representation加上depth/3D info可能进一步提升
6. **Latent action pretraining**: 结合LAPA (Ye et al., 2024) 的latent action pretraining，可能实现完全self-supervised的video → action learning

reference: https://latentactionpretraining.github.io/

### 7.4 与Sora-style World Simulator的关系

OpenAI的Sora paper (Brooks et al., 2024) 提出"video generation models as world simulators"。VPP实际上是这个idea在robotics落地的第一个concrete实例：

- Sora: video generation = world simulation
- VPP: video generation internal representation = world state + dynamics → policy

这条路线如果继续发展，可能lead到：**大规模internet video pretrain → 通用robot policy**，不用任何robot demonstration，纯靠video model的internal world understanding。

reference: https://openai.com/research/video-generation-models-as-world-simulators

### 7.5 公式变量含义总结

| Symbol | Meaning |
|--------|---------|
| $x_0$ | clean video sample |
| $x_t$ | noised video at step $t$ |
| $\alpha_t$ | noise schedule at step $t$ |
| $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$ | cumulative noise |
| $\epsilon$ | Gaussian noise |
| $\mu_\theta(x_t, t)$ | learnable denoiser |
| $V_\theta$ | modified SVD model |
| $s_0$ | initial observation frame |
| $l_{emb}$ | CLIP text embedding |
| $L_m$ | feature at $m$-th up-sampling layer |
| $F_p$ | aggregated predictive representation |
| $Q, Q', Q''$ | learnable Video Former tokens |
| $a_0$ | clean action sequence |
| $a_k$ | noised action at step $k$ |
| $\bar{\beta}_k$ | action noise schedule |
| $D_\psi$ | DiT action denoiser |

---

## 8. 总结

VPP这篇工作的核心贡献：

1. **Conceptual**: 提出predictive visual representation的概念，把VDM从generator重新定位为predictive encoder
2. **Methodological**: two-stage training (TVP fine-tune + inverse dynamics policy learning)
3. **Engineering**: single forward pass extraction + Video Former aggregation，实现140ms latency
4. **Empirical**: Calvin +18.6%, MetaWorld +10.8%, 真实世界tool-use +4.5x

最重要的intuition：**Policy不需要从零学physics，video model已经内化了physics。Policy只需要学习如何让robot去"track"predicted future中的自己。**

这跟你之前说的"software 2.0/3.0"思路高度契合：用大规模data训练的generative model作为认知核心，downstream task只需要少量alignment。

reference: 
- https://karpathy.medium.com/software-3-0-and-the-coming-era-of-ai-mediated-reality-297137b2d4e3
- https://video-prediction-policy.github.io/

希望这个解析对你的intuition building有帮助，Andrej。如果对某个细节想深挖，比如single forward pass为什么work的theoretical analysis，或者Video Former的具体attention pattern，可以继续聊。
