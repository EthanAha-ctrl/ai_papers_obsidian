---
source_pdf: WISA World Simulator Assistant for Physics-Aware Text-to-Video Generation.pdf
paper_sha256: bad0136d00b1fa54c8f66a19b59f97ab3b8a9ae0724586e082750f55f3a4ca61
processed_at: '2026-08-13T04:40:09-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# WISA 用人话讲

---

## 一句话说清楚它在干嘛

现在所有 T2V model (Sora, Kling, CogVideoX) 都有个尴尬问题：**画面漂亮，但物理不对**。苹果落水不溅起水花、橡皮擦擦完铅笔印更黑了、钟摆越摆越大——这些反直觉的 video 天天在刷屏。

WISA 的核心思路特别朴素：**既然 model 自己学不会物理，那就把物理当成 condition 显式喂给它**。跟 ControlNet 用 edge map、depth map 控制 image generation 一样的哲学，只不过这里 control signal 是 "physics information"。

---

## 为什么 model 学不会物理

这个问题其实在 LLM 时代大家应该有直觉。你让一个 model 看 10 million 个 video，它学到的是什么？学到的是 "水看起来长这样"、"火焰颜色分布大概这样"。它学的是 **appearance manifold**，不是 **physics manifold**。

打个比方：你看了 10 万遍火箭发射视频，你能画出火箭的样子，知道火焰往下喷。但你不知道为什么火箭往上飞——你从来没见过 F=ma 这个公式，你只见过 pixel pattern。

更具体地说，Koala-36M 这种 general-scene dataset 里，物理现象是 **背景噪声**。一个河边风景视频里，水在流，但水流不是 video 的主角，风景才是。Model 从这种 data 里提取不出 "流体力学" 的 learning signal，它只提取出 "水看起来是蓝色流动的"。

所以 WISA 干了两件事：
1. 造一个 data，让物理现象变成 video 的 **主角**
2. 设计一个 architecture，让物理信息变成 generation 的 **显式 input**

---

## Data 部分：WISA-32K

### 类别怎么定的

3 大类，17 个现象：

- **Dynamics**: collision, rigid body motion, elastic motion, liquid motion, gas motion, deformation
- **Thermodynamics**: melting, solidification, vaporization, liquefaction, explosion, combustion
- **Optics**: reflection, refraction, scattering, interference & diffraction, unnatural light sources

外加 12 个 "状态类" label（camera 是否 motion、object 是否 appearance/disappearance/mixing 之类），总共 29 个 qualitative category。

### 造 data 的 pipeline

```
人手收集 32000 个 physical phenomenon 明显的视频
    ↓ PySceneDetect 切 scene
    ↓ aesthetic filter 过滤丑的
    ↓ Qwen2-VL 生成 caption
    ↓ GPT-4o mini 从 caption 推断 physics annotation
WISA-32K
```

这里有个 pragmatic 的选择：**GPT-4o mini 不看 video，只看 caption**。

为什么？因为看 video 贵。Multi-modal GPT-4o mini 一个 sample 要 10k tokens，caption-based 只要 2k tokens。作者做了 ablation：multi-modal 准确率 78%，caption-based 76%。差 2 个百分点，但成本降 5 倍。这就是个工程权衡——把 perception 活儿外包给 Qwen2-VL，GPT-4o mini 只做 reasoning。

### Data curation 的核心 insight

WISA-32K 跟 Koala-36M 的本质区别，是 **supervisory signal density**。

Koala-36M 里一个 video 里有水流，但水流是 distractor。WISA-32K 强制要求：水流必须是 foreground salient——比如水从毛巾被挤出来、水龙头喷水。这等于把 data 从 "video of physical world" 升级成 "video ABOUT physics"。

这个 design 选择在 ablation 里被验证了：同样 32K video，同样训练 pipeline，用 Koala-36M 采样只能把 PC 从 0.33 提到 0.33（不动），用 WISA-32K 能提到 0.38。

**Data curation 比 architecture innovation 更重要**——这是这篇 paper 最 valuable 的 finding。

---

## Architecture 部分：怎么把物理塞进 model

Base model 是 CogVideoX-5B。WISA 是个 plug-in，只动 187M params（3.5%），inference 慢 5%。

### 三种物理信息，三种注入方式

| 信息 | 形式 | 注入方式 |
|---|---|---|
| Textual physical description | "amplitude of the swing gradually decreases over time" | 跟 caption concat，喂 text encoder |
| Qualitative physics category | 29-dim multi-hot vector | MoPA gating |
| Quantitative physics properties | density, time, temperature | AdaLN modulation |

直觉上：
- Text 走 natural language channel，让 model 用已有的语义理解能力处理
- Category 走 explicit routing channel，激活对应 expert head
- Quantity 走 continuous modulation channel，scale feature

### MoPA: Mixture-of-Physical-Experts Attention

这是 paper 的核心 architecture contribution，灵感来自 MoH。

**核心 idea**：multi-head attention 里不同 head 本来就会自发 specialize——有的 head 看 motion，有的看 texture。MoPA 把这个 implicit specialization **强制 explicit 化**：每个 head 钉死一个 physics category，input condition 决定激活哪些 head。

公式：

$$
\hat{P}_c = \text{Random}(P_c), \quad F_h = \text{MHSA}(F)
$$

$$
F_o = \text{Linear}(\text{Reshape}(F_h \odot \hat{P}_c))
$$

变量讲清楚：
- $P_c \in \mathbb{R}^{C}$: input category vector, $C=29$, $P_c^i \in \{0,1\}$ 表示第 $i$ 个 category 是否激活
- $\hat{P}_c$: 经过 random perturbation 的版本。如果 $P_c^i = 1$，有概率被改成 0.1（保留 10% 信号）；如果 $P_c^i = 0$，有 0.2 概率被改成 1.0（偶尔激活 "wrong expert"）
- $F$: 上一个 DiT block 的 denoising feature
- $F_h \in \mathbb{R}^{N \times d \times h}$: MHSA 输出。$N$ = token 数，$d$ = head dim，$h = C = 29$（head 数严格等于 category 数）
- $\odot$: element-wise 乘，每个 head 整体被其 category 的标量 scale
- $F_o$: linear 投回原维度

**为什么需要 perturbation**：因为 GPT-4o mini 标注只有 75% 准确率，label 是 noisy 的。Hard gating 会让 model overfit 错误标注。Perturbation 是 label smoothing + dropout 的混合体——偶尔激活 "wrong expert" 逼所有 head 都学一些 universal pattern，避免 catastrophic forgetting。

**Attention map 验证**：Figure 8 里，"rigid body motion" expert 的 attention 集中在 swing 上，"no obvious dynamic" expert 集中在 static background。证明 explicit specialization 真的发生了。这是 mechanistic interpretability 的 evidence。

### 为什么只插在最后一个 block 后面

WISA 没在每个 DiT block 后都插 Physical Module，只在 final block 后插一个。

理由：
1. 每层都插参数爆炸
2. Shallow physical module 收敛慢
3. 会破坏 base model 的 inherent capability

直觉上，这是把 physical module 当成 **late-stage refinement**——base model 负责大体 appearance/motion，physical module 在 final token level 做 physics-aware correction。这跟 ControlNet 思路有点像，但 ControlNet 是 parallel 双塔，WISA 是 serial 单 module。

### Physical Classifier

在 Physical Module 之后接一个 classifier，predict 29-dim category。

Loss 是 multi-label BCE：

$$
L_{pc} = \sum_{i=1}^{C} \left[ P_c^i \log(f_c^i) + (1 - P_c^i) \log(1 - f_c^i) \right]
$$

- $C=29$, $P_c^i$ 是 GT, $f_c^i \in (0,1)$ 是预测概率

**为什么需要这个 auxiliary head**：强迫 denoising feature 里显式编码 physics category 信息。等于在 latent space 加一个 "physics concept bottleneck"——这跟 CLIP 的 contrastive auxiliary head 同构：主 task 是 generation，auxiliary task 强制 latent geometry 对齐到 physics category manifold。

### Total Loss

$$
L = L_{\text{diffusion}} + \lambda \cdot \frac{L_{pc}}{1 + L_{pc}.\text{detach}}
$$

- $L_{\text{diffusion}}$: 基础 T2V flow-matching loss
- $\lambda$: balancing coefficient
- $L_{pc}.\text{detach}$: stop-gradient 后的 classifier loss，当分母

**这个 self-normalizing trick 有意思**：
- 早期 $L_{pc}$ 大，$\frac{L_{pc}}{1 + L_{pc}} \to 1$，auxiliary loss weight 接近 $\lambda$，strong supervision
- 后期 $L_{pc}$ 小，$\frac{L_{pc}}{1 + L_{pc}} \to 0$，auxiliary loss 自动 anneal，放手让 diffusion loss 主导

等于自动 curriculum——先学 "what physics is in this video"（classification），再内化到 generation。`detach` 避免 denominator 通过反向传播影响 classifier 自己。

---

## Experiments

### 主结果 (Table 1)

| Method | Inference (s) | VideoPhy SA | VideoPhy PC | PhyGenBench PC |
|---|---|---|---|---|
| CogVideoX-5B (baseline) | 210 | 0.60 | 0.39 | 0.41 |
| Cosmos-Diffusion-7B | 600 | 0.57 | 0.43 | 0.14 |
| PhyT2V (Round 4) | 1800 | 0.61 | 0.37 | 0.42 |
| **WISA** | **220** | **0.67** | **0.38** | **0.43** |

观察：
- WISA 比 baseline SA +0.07，PhyGenBench PC +0.02
- VideoPhy PC 从 0.39 → 0.38 略降，这是个 caveat
- 比 PhyT2V 快 9 倍（220s vs 1800s）
- Cosmos 在 PhyGenBench PC 只有 0.14 — temporal ordering 灾难性失败

VideoPhy PC 略降这件事作者没正面解释。但 Section G 给了线索：VideoCon-Physics 这个 evaluator 本身有误判。Figure 15 展示了一个明显物理正确的 "物体先入水再溅起水花" 的 video 被打 0.08 分。Qwen2.5-VL 也分不清 event 时序。**Automatic physics evaluation 是整个 field 的瓶颈**。

### Ablation (Table 2)

| Setting | SA | PC |
|---|---|---|
| Baseline | 0.60 | 0.33 |
| Only LoRA | 0.64 | 0.34 |
| w/o Physical Module | 0.64 | 0.33 |
| w/o Physical Classifier | 0.66 | 0.36 |
| 32K from Koala-36M | 0.62 | 0.33 |
| **WISA-32K (full)** | **0.67** | **0.38** |

解读：
1. Only LoRA 能提 SA（+0.04）但 PC 几乎不动——fine-tune on physics data 让语义更聚焦，但不会自动学物理
2. 去掉 Physical Module，PC 不动——MoPA + AdaLN 注入是 PC 提升的来源
3. 去掉 Physical Classifier，PC 从 0.38 掉到 0.36——auxiliary supervision 贡献 +0.02
4. **Data swap 是最大 contributor**：Koala-32K 给 SA 0.62/PC 0.33，WISA-32K 给 0.67/0.38

**Data curation 贡献 +0.05/+0.05，architecture 贡献 +0.03/+0.02，data > architecture**。

### Human Evaluation

Figure 7 的 human ranking：WISA 在 physical alignment 上大幅领先，semantic consistency 也保持竞争力。这跟 automatic metric 不完全一致，进一步印证 automatic metric 不可靠。

---

## 局限性

作者自己承认的：
1. **类别覆盖窄**：17 类，没有 sublimation, condensation, corrosion, electromagnetic, relativistic
2. **没有 equation-level constraint**：只有 semantic-level guidance，没有 Newton's law、能量守恒这种硬约束
3. **数据量小**：32K vs CogVideoX 的 millions-scale 训练数据
4. **依赖 GPT-4o mini 标注**：75% 准确率的 noisy label
5. **Evaluator 不可靠**：VideoCon-Physics 误判严重

---

## 我的几个观察

### MoPA 的严格 1-to-1 mapping 是否最优

C=29 个 head 钉死到 29 个 category。但 liquid motion 和 gas motion 在 visual manifestation 上共享 fluid dynamics 底层 structure。强制独立 head 可能 underutilize shared pattern。Alternative 是 learnable routing（真正 MoE 风格），但需要更大数据量支撑 expert specialization。

### Physical Classifier 的潜在 bottleneck

强迫 latent encode physics category 在 multi-physics-coupled 场景可能造成信息 bottleneck。一个 video 同时有 reflection + refraction + scattering，binary multi-label 信号密度低，classifier 难训。

### Latent physics vs explicit physics

WISA 在 latent diffusion feature 上加 physics structure，但 latent 中的 "physics" 是 emergent property，不是显式 quantity。真正强物理 consistency 可能需要 trajectory-level supervision（轨迹偏差、能量误差）。MotionCraft、PhysGen 这类 differentiable physics 路线更硬核，但泛化性差。WISA 在 generalization 和 physical fidelity 之间 trade off，选了 generalization。

### 对 world simulator 路线的启示

WISA 论证了一件事：**world simulator 不能光靠 scale，需要 explicit physics supervision**。但 WISA 的 supervision 停在 semantic level，下一步可能的方向：
- Hybrid: WISA-style semantic supervision + differentiable physics simulator refinement
- Trajectory-level contrastive learning：用真实物理 simulator 生成 paired counterfactual
- Equation-conditioned generation：直接把 ODE/PDE 作为 condition

---

## Reference

**核心**:
- WISA project: https://wisa-360.github.io/
- CogVideoX: https://arxiv.org/abs/2408.06072
- VideoPhy: https://arxiv.org/abs/2406.03520, https://github.com/HritikBansal/videophy
- PhyGenBench: https://arxiv.org/abs/2410.05363
- PhyT2V: https://arxiv.org/abs/2412.00596
- MoH: https://arxiv.org/abs/2410.11842

**Architecture**:
- DiT (AdaLN): https://arxiv.org/abs/2212.09748
- LoRA: https://arxiv.org/abs/2106.09685
- ControlNet: https://arxiv.org/abs/2302.05543

**Data/Annotation**:
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- Koala-36M: https://arxiv.org/abs/2410.08260
- OpenVid-1M: https://arxiv.org/abs/2407.02371

**Physics-aware generation**:
- MotionCraft: https://arxiv.org/abs/2405.13557
- PhysGen: https://arxiv.org/abs/2408.06072
- "How far is video generation from world model": https://arxiv.org/abs/2411.02385
- "Do generative video models learn physical principles": https://arxiv.org/abs/2501.09038

**World simulator**:
- Sora: https://openai.com/index/sora/
- Cosmos: https://arxiv.org/abs/2501.03575
- Kling: https://klingai.kuaishou.com/
- Pandora world model: https://arxiv.org/abs/2406.09455
- "Is Sora a world simulator" survey: https://arxiv.org/abs/2405.03520

---

## 一句话收尾

WISA 把 "physics" 这个抽象 latent concept 拆成 text + category + quantity 三层 condition，用 MoPA + Physical Classifier + curated dataset 三件套注入 CogVideoX-5B。最 valuable 的 finding 是 **data curation > architecture innovation**——这跟 LLM scaling 时代的整体直觉一致，physics-aware generation 仍是 data-bottlenecked problem。Limitation 是 supervision 停在 semantic level，没进 equation level，这是 next frontier。

---

# WISA: World Simulator Assistant for Physics-Aware Text-to-Video Generation — 深度解读

Andrej，这篇论文本质上是把 "physical priors" 作为结构化 condition 显式注入到 T2V diffusion model 里，并且强调 data curation 而非仅仅依赖 scale。我会从 motivation、data、architecture、loss、experiment 五个层面对它做 surgical 解剖。

---

## 1. Motivation: 为什么 T2V 模型不懂物理

现有的 T2V models (Sora, Kling, CogVideoX, Cosmos, HunyuanVideo) 本质上都在拟合训练分布 p_data(video | text)。物理规律在 general-scene dataset (Koala-36M, OpenVid-1M) 中是 **implicit secondary structure**：水流、烟雾、影子这些物理现象只是 background distractor，不是 foreground learning signal。模型学到的更多是 "appearence manifold"，而不是 "physics manifold"。

WISA 的核心 insight 是：**physical laws 太抽象，必须被 decomposed 成 generation model 能消化的 multi-granularity condition**。这个 decomposition 就是三个维度：

| 层次 | 信息形式 | 注入方式 |
|---|---|---|
| Textual physical descriptions | Natural language sentence | Concatenate with caption → text encoder |
| Qualitative physics categories | 29-dim multi-hot vector | MoPA gating |
| Quantitative physical properties | density, time, temperature | AdaLN modulation |

这种三层分解本身就是一个比较 elegant 的归纳偏置 (inductive bias) 设计——把 "physics" 这种 latent concept 拆成 token-level、category-level、continuous-quantity-level 三个独立可学习的入口。

参考：Sora technical report (https://openai.com/index/sora/), CogVideoX (https://github.com/THUDM/CogVideo), Cosmos (https://arxiv.org/abs/2501.03575), Koala-36M (https://arxiv.org/abs/2410.08260).

---

## 2. WISA-32K: Data Curation 哲学

### 2.1 类别定义

3 个 physics branch × 17 phenomena：

- **Dynamics (47%)**: Collision, Rigid Body Motion, Elastic Motion, Liquid Motion, Gas Motion, Deformation
- **Thermodynamics (24%)**: Melting, Solidification, Vaporization, Liquefaction, Explosion, Combustion
- **Optics (29%)**: Reflection, Refraction, Scattering, Interference & Diffraction, Unnatural Light Sources

外加 9 个 "anomaly/state" category (camera motion, object appearance/disappearance/mixing)，构成 29 个 qualitative label。

### 2.2 Pipeline

```
Raw videos (32K, manually collected)
    ↓ PySceneDetect (shot boundary detection)
    ↓ Aesthetic filtering
    ↓ Qwen2-VL → caption (≤256 tokens)
    ↓ GPT-4o mini → decompose to 3 types of physics info
WISA-32K (with structured annotations)
```

### 2.3 关键决策的 intuition

- **Caption-based 而非 video-based annotation**：作者做了一个 ablation (100 samples)，multi-modal 78% vs caption-based 76%，但 cost 是 10k vs 2k tokens/sample。这等于把 dense visual perception 工作外包给 Qwen2-VL，GPT-4o mini 只做 reasoning。这是 pragmatic 的两段式 pipeline 设计——VLM 负责 perception，LLM 负责 decomposition。
- **Explicit phenomena as primary criterion**: 这点很重要。Koala-36M 中液体运动通常是 "background flow" (e.g., water in a river), 而 WISA-32K 强制要求 phenomena 必须是 foreground salient (e.g., 水从毛巾被挤出)。这把 data 从 "video of physical world" 升级成 "video ABOUT physics"——本质是 supervisory signal density 的提升。
- **未覆盖**: sublimation, condensation, corrosion, vacuum 被排除。这是 long-tail 的 pragmatic 取舍。

### 2.4 Annotation Accuracy

人类评估 100 examples：dynamics 84%, optics 71%, thermodynamics 64%, overall 75%。Thermodynamics 准确率最低，这跟 GPT-4o mini 对温度、相变的 reasoning 能力较弱一致。

参考：Qwen2-VL (https://arxiv.org/abs/2409.12191), PySceneDetect (https://github.com/Breakthrough/PySceneDetect), VideoPhy (https://arxiv.org/abs/2406.03520).

---

## 3. Architecture: WISA on top of CogVideoX-5B

### 3.1 整体设计选择

WISA 是一个 **plug-in module**，base model 是 CogVideoX-5B。关键约束：
- 只训练 LoRA (rank=128, alpha=16) + Physical Module + Physical Classifier = 187M params
- 仅 3.5% 参数 overhead, 5% inference time overhead
- Physical Module 只插在 final transformer block 之后（重要 design choice）

为什么不每一层都插？论文给的理由：1) parameter & compute explosion; 2) shallow physical module 收敛慢; 3) 丢失 base model capability。直觉上是把 physical module 当成 **late-stage refinement** 而非 early-stage feature transformation——base model 负责大体 appearance/motion 生成，physical module 在 final token level 做 physics-aware correction。这跟 ControlNet 的思路有点像，但 ControlNet 是并行双塔，WISA 是串行单 module。

参考：LoRA (https://arxiv.org/abs/2106.09685), ControlNet (https://arxiv.org/abs/2302.05543), DiT/AdaLN (https://arxiv.org/abs/2212.09748).

### 3.2 Mixture-of-Physical-Experts Attention (MoPA)

这是论文最核心的 architecture contribution。灵感来自 MoH (Mixture-of-Head Attention, https://arxiv.org/abs/2410.11842) 和 MoE。

**直觉**：multi-head attention 中不同 head 本来就会自发学到不同 pattern。MoPA 把这个 implicit specialization **强制 explicit 化**——每个 head 钉死到一个 physics category，通过 input condition 决定激活哪些 head。

**数学形式**：

$$
\hat{P}_c = \text{Random}(P_c), \quad F_h = \text{MHSA}(F)
$$

$$
F_o = \text{Linear}(\text{Reshape}(F_h \odot \hat{P}_c))
$$

变量解释：
- $P_c \in \mathbb{R}^{C}$: 输入的 physical category one-hot/multi-hot vector，$C = 29$
- $P_c^i \in \{0, 1\}$: 第 $i$ 个 category 的 binary state
- $\hat{P}_c$: 经过 random perturbation 后的版本。若 $P_c^i = 1$，则以一定概率改为 $0.1$；若 $P_c^i = 0$，则以 0.2 概率改为 $1.0$
- $F$: 输入 denoising feature（来自上一个 DiT block 的输出）
- $F_h \in \mathbb{R}^{N \times d \times h}$: MHSA 输出，$N$ = token 数，$d$ = head dimension，$h = C = 29$（head 数量严格等于 category 数）
- $F_h \odot \hat{P}_c$: head-wise gating，每个 head 的整个输出被其对应 category 的标量缩放
- $F_o$: linear project 回原维度后的输出

**Perturbation 的 intuition**：这是一个 **label smoothing / dropout-like regularization**。multi-hot label 经常有 noise（GPT-4o mini 标注准确率 75%），如果 hard gating 会让 model overfit 到错误标注。把 $1 \to 0.1$ 等于保留 10% 信号；把 $0 \to 1.0$ 等于偶尔激活 "wrong expert"——这逼所有 head 都学到一些 universal pattern，避免 catastrophic forgetting，同时让 model 对 annotation noise 鲁棒。

**Attention map 验证**：Figure 8 显示 "rigid body motion" expert 集中在 swing 上，"no obvious dynamic phenomenon" expert 集中在 static background。这证明 explicit specialization 确实发生了——这是 mechanistic interpretability 层面的 evidence。

### 3.3 Quantitative Properties via AdaLN

时间、温度、密度三个 quantitative 变量被编码：
1. 用 scientific notation 表示 (coefficient + exponent)，因为不同 phenomena 的 scale 差异巨大 (e.g., 爆炸 ms, 冰川 melting 小时)
2. Linear projection → embedding
3. Concatenate with diffusion timestep embedding
4. 通过 AdaLN (Adaptive Layer Norm) 注入：$\gamma, \beta = \text{MLP}(c_t, c_{phys})$，然后 $x \leftarrow \gamma \cdot \text{Norm}(x) + \beta$

这跟 DiT (Peebles & Xie) 中 timestep conditional modulation 同构，只是 conditioning vector 多了 physical quantity 信息。Inference 时候 timestep 走 forward，physical quantities 作为额外 channel 注入。

### 3.4 Physical Classifier

```
Final denoising feature → Physical Classifier → sigmoid → f_c ∈ R^29
```

用 multi-label BCE loss 监督：

$$
L_{pc} = \sum_{i=1}^{C} \left[ P_c^i \log(f_c^i) + (1 - P_c^i) \log(1 - f_c^i) \right]
$$

变量解释：
- $C = 29$: 类别数
- $P_c^i \in \{0, 1\}$: ground-truth label
- $f_c^i \in (0, 1)$: 预测概率
- $L_{pc}$: 整体 multi-label BCE，所有类别 sum 起来

**为什么需要 Physical Classifier**：这是 auxiliary supervision。强迫 denoising feature 中显式编码 physics category 信息，等于在 latent space 上加一个 "physics concept bottleneck"。这跟 CLIP 的 contrastive auxiliary head 类似——主 task 是 generation，但 auxiliary task 强制 latent geometry 对齐到 physics category manifold。

### 3.5 总 Loss

$$
L = L_{\text{diffusion}} + \lambda \cdot \frac{L_{pc}}{1 + L_{pc}.\text{detach}}
$$

变量解释：
- $L_{\text{diffusion}}$: 基础 T2V flow-matching / diffusion loss
- $\lambda$: balancing coefficient（论文未给具体值，可推测在 1 附近）
- $L_{pc}.\text{detach}$: stop-gradient 后的 classifier loss，作为 normalization denominator

**这个 loss 设计有意思**：分母 $1 + L_{pc}.\text{detach}$ 是一个 self-normalizing trick。当 classifier loss 很大（model 还没学好 physics），$\lambda \cdot L_{pc} / (1 + L_{pc})$ 趋近 $\lambda$；当 classifier loss 很小（已经学好），这个项趋近 0。等于自动 anneal auxiliary loss weight——早期 strong supervision，后期放手让 diffusion loss 主导。`detach` 保证 classifier loss 不会通过 denominator 反向影响自己（避免循环依赖）。

直觉上：这是一种 **gradual curriculum**，让 model 先学会 "what physics is in this video" (classification)，再慢慢把这种 awareness 内化到 generation。

---

## 4. Experiments

### 4.1 Benchmark

- **VideoPhy** (https://arxiv.org/abs/2406.03520): 344 prompts，用 VideoCon-Physics 评估
- **PhyGenBench** (https://arxiv.org/abs/2410.05363): 160 prompts
- Metrics: SA (Semantic Alignment) ↑, PC (Physical Consistency) ↑
- 阈值：≥0.5 视为 1，否则 0

### 4.2 Quantitative Results (Table 1)

| Method | Inference (s) | VideoPhy SA | VideoPhy PC | PhyGenBench SA | PhyGenBench PC |
|---|---|---|---|---|---|
| VideoCrafter2 | – | 0.47 | – | – | – |
| HunyuanVideo | – | 0.36 | 0.46 | 0.28 | 0.33 |
| CogVideoX-5B (baseline) | 210 | 0.60 | 0.39 | 0.41 | – |
| Cosmos-Diffusion-7B | 600 | 0.57 | 0.43 | 0.14 | – |
| PhyT2V (Round 4) | 1800 | 0.59 | 0.38 | 0.42 | – |
| PhyT2V* (Round 4) | 1800 | 0.61 | 0.37 | – | – |
| **WISA** | **220** | **0.67** | **0.38** | **0.40** | **0.43** |

观察：
- WISA 在 VideoPhy 上 SA +0.07, PC -0.01 (vs baseline CogVideoX-5B)，PC 略降但 SA 显著提升
- PhyGenBench 上 PC 大幅提升 (0.39 → 0.43)
- Inference time 220s vs PhyT2V 1800s — 9× speedup，因为 PhyT2V 要跑 4 轮 Tarsier-34B 反馈
- Cosmos 在 PhyGenBench PC 只有 0.14 — temporal ordering 失败案例
- HunyuanVideo 在 VideoPhy PC 反而最高 (0.46)，但 SA 最低 (0.36) — 可能生成稳定但语义不匹配的视频

注意 PC 在 VideoPhy 上从 0.39 → 0.38 略降，**这是 WISA 的一个 caveat**。作者没展开讲，但结合 human eval (Figure 7) 和 discussion of quantitative evaluation (Section G)：VideoCon-Physics 本身存在误判。论文 Figure 15 展示了一个明显物理正确的视频被打了 0.08 分。这说明 automatic metric 不可靠——这是当前整个 physics-aware generation 评估的瓶颈。

### 4.3 Ablation (Table 2)

| Setting | SA | PC |
|---|---|---|
| Baseline | 0.60 | 0.33 |
| Only LoRA | 0.64 | 0.34 |
| w/o Physical Module | 0.64 | 0.33 |
| w/o Physical Classifier | 0.66 | 0.36 |
| 32K from Koala-36M (general data) | 0.62 | 0.33 |
| WISA-32K (curated) | 0.67 | 0.38 |

关键发现：
1. **Only LoRA 就能提 SA** (+0.04)，但 PC 几乎不动 (+0.01) — fine-tuning on physics data 让语义更聚焦，但不会自动学到 physics
2. **去掉 Physical Module** PC 不变 (0.33) — qualitative + quantitative 信息是 PC 提升的来源
3. **去掉 Physical Classifier** PC 从 0.38 掉到 0.36 — auxiliary supervision 贡献 +0.02
4. **Data quality 才是大头**：用 Koala-36M 32K 同样训练 pipeline 只能拿到 SA 0.62 / PC 0.33；用 WISA-32K 拿到 SA 0.67 / PC 0.38。Data curation 贡献了 SA +0.05 / PC +0.05，比 module 设计本身贡献更大。

这条 ablation 实际上是论文最 interesting 的发现：**在 physics-aware generation 这个问题上，data curation > architecture innovation**。这跟一般 LLM/VLM scaling 的直觉一致，但被 explicit 地验证了。

### 4.4 Human Evaluation (Figure 7)

3 个候选 model 对比 ranking (3/2/0 points)：WISA 在 physical alignment 上大幅领先，在 semantic consistency 上也保持竞争力。这跟 automatic metric 不完全一致，进一步印证 automatic metric 不可靠。

---

## 5. Failure Cases & Limitations

1. **类别覆盖窄**：17 类不包含 sublimation, condensation, corrosion, electromagnetic phenomena, relativistic effects
2. **缺 mechanism-level constraint**：只有 high-level semantic guidance，没有 Newton's law、能量守恒这种 equation-level constraint。作者明确说做这件事需要 image/3D-based motion modeling，泛化性差
3. **数据量小**：32K vs CogVideoX 的 millions-scale 训练数据。LoRA + 短训练 (8000 steps) 是合理选择，但天花板有限
4. **依赖 GPT-4o mini 标注**：annotation 75% 准确率，noise 被 Perturbation 部分缓解，但仍然是 ceiling
5. **VideoCon-Physics evaluator 不可靠**：Figure 15 显示明显正确的物理过程被打 0.08，Qwen2.5-VL 也分不清事件时序

---

## 6. 一些更深层的思考 / Open Questions

### 6.1 MoPA 的严格 1-to-1 head-category mapping 是否最优

C=29 head 是被钉死的。但某些 physics 在 visual manifestation 上共享底层 pattern (e.g., liquid motion 和 gas motion 都涉及 fluid dynamics)。强行 29 个独立 head 可能 underutilize 共享 structure。一种 alternative 是 learnable routing (像真正的 MoE)，但需要更大数据量支撑 expert specialization。

### 6.2 Physical Classifier 的潜在 negative transfer

强迫 latent encode physics category 在多 phenomenon coupled 场景可能造成信息 bottleneck。例如一个视频同时有 reflection 和 refraction，binary multi-label 还好；但如果有 8 个现象同时出现，binary 信号密度低，classifier 难训。

### 6.3 Latent space 的 physics geometry

WISA 本质是在 latent diffusion 的 denoising feature 上加 physics structure。但 latent space 中的 "physics" 是 emergent property，不是显式 quantity。真正强物理 consistency 可能需要 trajectory-level supervision（轨迹偏差、能量误差），这跟 MotionCraft (https://arxiv.org/abs/2405.13557)、PhysGen (https://arxiv.org/abs/2408.06072) 这类 differentiable physics 路线接近。WISA 的 high-level approach 在 generalization 和 physical fidelity 之间 trade off，选了 generalization。

### 6.4 对 Sora-style world simulator 的启示

WISA 的论点是：world simulator 不能光靠 scale，需要 **explicit physics supervision**。这与 Kang et al. "How far is video generation from world model" (https://arxiv.org/abs/2411.02385) 的结论一致——pure scaling 学不到 physics law。但 WISA 的 supervision 仍然停留在 semantic level，没有进入 equation level。下一步可能的 direction：
- Hybrid: WISA-style semantic supervision + differentiable physics simulator refinement
- Trajectory-level contrastive learning：用真实物理 simulator 生成 paired counterfactual
- Equation-conditioned generation：直接把 ODE/PDE 作为 condition

### 6.5 跟 PhyT2V (https://arxiv.org/abs/2412.00596) 对比

PhyT2V 是 LLM-guided iterative refinement：生成 → Tarsier-34B 分析 → 修改 prompt → 再生成。WISA 是 in-model training。PhyT2V 不改 model weights，泛化性好但 inference 慢 (1800s vs 220s)。两者其实可以 compose——WISA 训练 base physics awareness，PhyT2V 在 inference 时做 fine-grained refinement。

---

## 7. Reference & 进一步阅读

**核心论文**:
- WISA arxiv: https://arxiv.org/abs/2506.00000 (placeholder, 实际需搜 WISA Sora physics)
- WISA project page: https://wisa-360.github.io/ (作者 GitHub Pages，论文中提到)
- CogVideoX: https://arxiv.org/abs/2408.06072, https://github.com/THUDM/CogVideo
- VideoPhy: https://arxiv.org/abs/2406.03520, https://github.com/HritikBansal/videophy
- PhyGenBench: https://arxiv.org/abs/2410.05363
- PhyT2V: https://arxiv.org/abs/2412.00596
- MoH: https://arxiv.org/abs/2410.11842
- Cosmos: https://arxiv.org/abs/2501.03575, https://github.com/nvidia-cosmos
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- Sora: https://openai.com/index/sora/
- Kling: https://klingai.kuaishou.com/
- Step-Video-T2V: https://arxiv.org/abs/2502.10248
- Wan2.1: https://github.com/Wan-Video/Wan2.1
- Koala-36M: https://arxiv.org/abs/2410.08260
- OpenVid-1M: https://arxiv.org/abs/2407.02371

**架构相关**:
- DiT (AdaLN 来源): https://arxiv.org/abs/2212.09748
- LoRA: https://arxiv.org/abs/2106.09685
- MoE (Vision): https://arxiv.org/abs/1911.02570
- ControlNet: https://arxiv.org/abs/2302.05543
- Flow Matching: https://arxiv.org/abs/2210.02747

**数据 annotation**:
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- PySceneDetect: https://github.com/Breakthrough/PySceneDetect
- Tarsier-34B: https://arxiv.org/abs/2407.00634
- InternVideo2: https://arxiv.org/abs/2401.02647

**Physics-aware video generation 相关**:
- MotionCraft: https://arxiv.org/abs/2405.13557
- PhysGen: https://arxiv.org/abs/2408.06072
- DANO: https://arxiv.org/abs/2305.02284
- "How far is video generation from world model": https://arxiv.org/abs/2411.02385
- "Do generative video models learn physical principles": https://arxiv.org/abs/2501.09038
- Pandora world model: https://arxiv.org/abs/2406.09455

**Survey**:
- "Is Sora a world simulator?": https://arxiv.org/abs/2405.03520

---

## 8. 一句话总结

WISA 的核心 contribution 是把 "physics" 从 latent abstract concept 转化为 explicit multi-granularity condition (text + category + quantity)，并设计 MoPA + Physical Classifier + curated WISA-32K dataset 三个组件协同注入。在 CogVideoX-5B 上以 3.5% 参数和 5% inference time cost 取得 SA +0.07 / PC +0.04 提升。最 valuable 的 finding 是 **data curation (WISA-32K) 贡献大于 architecture innovation (MoPA)**——这跟 LLM scaling 时代的整体直觉一致，physics-aware generation 仍是 data-bottlenecked problem。Limitation 在于 supervision 停留在 semantic level，没有 equation-level / trajectory-level 的硬约束，这是 next frontier。
