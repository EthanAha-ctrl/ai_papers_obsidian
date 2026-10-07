---
source_pdf: Causal Forcing Autoregressive Diffusion Distillation Done Right for High-Quality
  Real-Time Interactive Video Generation.pdf
paper_sha256: 2d5631d346f2b1e5a957ee027db8cf30c915ba93ae0a2ae329d2913465b9e1e3
processed_at: '2026-08-03T15:11:59-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

好，我换个讲法，就当我们咖啡店聊天。

---

## 这篇paper到底在搞啥

现在做video generation最火的路线是 **bidirectional diffusion**，比如 Wan2.1、CogVideoX、Hunyuan。这类模型一次性生成所有帧，质量很顶，但有个硬伤：**你必须等整段视频算完才能看到第一帧**，跑一次几十秒到几分钟。这东西没法做交互，比如游戏、world model、avatar对话，全废。

所以大家就转向 **autoregressive (AR) video diffusion**：一帧一帧往后生成，生成第 $i$ 帧只看前面的 $\mathbf{x}_0^{<i}$。这样就能streaming，能交互。Google Genie 3 (https://arxiv.org/abs/2507.07971) 就走这条。

但AR model每帧内部还得跑几十步diffusion采样，还是太慢。所以需要 **distillation** 把它压到4步以内，做到real-time（17 FPS左右）。

这篇paper研究的核心问题：**从bidirectional teacher蒸馏AR student，为什么老掉性能，以及怎么真正做对**。

---

## 现有方法的pipeline长啥样

SOTA方法叫Self Forcing (https://arxiv.org/abs/2506.22590)，两阶段：

**Stage A (ODE distillation)**: 拿一个训练好的bidirectional model当teacher，采样它的PF-ODE轨迹（就是diffusion采样的中间noisy状态），训练一个AR student把这些noisy状态映射回clean frame。

**Stage B (Asymmetric DMD)**: 再用DMD (https://arxiv.org/abs/2310.14189)做distribution matching refinement。

听起来很合理，但实验上 Self Forcing 比标准DMD（蒸馏成bidirectional student）差一大截，从同一个teacher蒸馏出来的。这就奇怪了——同样teacher，同样distillation方法，为啥AR student就拉胯？

---

## 问题的根本原因：一个很fundamental的math issue

这里就要讲到paper最核心的insight了。

### ODE distillation要work，必须满足injectivity

啥叫injectivity？简单说就是 **每个noisy样本必须唯一对应一个clean样本**。如果同一个$\mathbf{x}_t$可以对应多个不同的$\mathbf{x}_0$，MSE回归的最优解就collapse成 **conditional expectation**：

$$G_\theta^*(\mathbf{x}_t, t) = \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t, t]$$

这个conditional expectation是啥？就是"给定noisy input，所有可能的clean输出的平均"。**平均多个不同图像 = 模糊**。这就是为什么diffusion distillation做不好会blurry的根本原因，Bishop的PRML (https://www.springer.com/gp/book/9780387310732) 第41页有标准推导。

### Bidirectional teacher天然video-level injective

对bidirectional模型，整个视频$\mathbf{x}_t^{1:N}$通过PF-ODE映射到$\mathbf{x}_0^{1:N}$，这是one-to-one的（PF-ODE解的唯一性保证）。所以如果你蒸馏一个bidirectional student，injectivity天然成立。

### 但AR student需要frame-level injective，这个就出问题了

AR生成是逐帧的，第$i$帧只看$(\mathbf{x}_t^i, \mathbf{x}_0^{<i})$，看不到未来$\mathbf{x}_t^{>i}$。所以injectivity必须在 **frame level** 成立：

> 固定第$i$帧的noisy state $\mathbf{x}_t^i$，无论其他帧长啥样，$\mathbf{x}_0^i$必须唯一确定。

**但bidirectional model的PF-ODE不满足这个条件！**

因为bidirectional model用full attention，去噪第$i$帧时会偷看所有帧（包括未来）。所以即使$\mathbf{x}_t^i$一样，只要future frames $\mathbf{x}_t^{>i}$不同，$\mathbf{x}_0^i$就不同。

这就是Lemma 3.2的核心：

$$\mathbb{P}\left(\text{Var}\left(\phi^{\text{Bi}}(\mathbf{x}_t^{1:N}, t)^i \mid \mathbf{x}_t^i, t\right) > 0\right) > 0$$

变量解释：
- $\phi^{\text{Bi}}(\mathbf{x}_t^{1:N}, t)^i$: bidirectional model的flow map，把noisy video $\mathbf{x}_t^{1:N}$映射到clean video的第$i$帧
- $\text{Var}(\cdot \mid \mathbf{x}_t^i, t)$: 给定$\mathbf{x}_t^i$和$t$的条件方差
- 这个概率大于0意味着：存在非零概率，同一个$\mathbf{x}_t^i$对应多个不同的$\mathbf{x}_0^i$

**直觉pump**：bidirectional teacher像一个能看见未来的预言家，它去噪当前帧时偷看了未来。你让一个只能看过去的学生去模仿这个老师，学生看不到老师偷看的那部分信息，只能学到"对各种可能的未来取平均"——结果就是模糊。

### Self Forcing的ODE init阶段就坏掉了

Self Forcing拿bidirectional model采PF-ODE轨迹训练AR student。训练目标：

$$\theta^* = \arg\min_\theta \mathbb{E}_{t, \mathbf{x}_t^{1:N}, i} \left[ \| G_\theta(\mathbf{x}_t^i, \mathbf{x}_t^{<i}, t) - \mathbf{x}_0^i \|^2 \right]$$

变量含义：
- $i \sim \mathcal{U}(1, N)$: 随机选一帧
- $(\mathbf{x}_t^{1:N}, \mathbf{x}_0^{1:N})$: 同一个bidirectional PF-ODE轨迹上的noisy和clean对
- $t \in S$: predefined timesteps

因为bidirectional PF-ODE违反frame-level injectivity，这个MSE回归collapse到conditional expectation（Proposition 3.3），AR student学到的就是blurry average。**ODE init阶段就已经坏了**。

### DMD为啥修不了

作者做了个controlled experiment（Fig. 2）：直接用bidirectional few-step model初始化AR student（绕过ODE init问题，只保留architectural gap），然后跑DMD。结果还是差。

**DMD本质是marginal distribution matching**，通过KL gradient让student的生成分布逼近data分布。但conditional structure一旦在init阶段坏了（blurry conditional mean），DMD只能修marginal，修不回conditional。这就像你给一个已经模糊的图做后处理，能调调色调、改改marginal statistics，但细节回不来。

---

## Causal Forcing怎么解决

idea非常simple and elegant：**用AR teacher代替bidirectional teacher做ODE init**。

### 三阶段pipeline

**Stage 1: 训练AR diffusion model (Teacher Forcing)**

先训练一个多步的AR diffusion model。训练第$i$帧时condition在clean prefix $\mathbf{x}_0^{<i}$上。这个model后续当teacher。

**Stage 2: Causal ODE distillation**

用Stage 1的AR teacher采样PF-ODE轨迹，训练AR student：

$$\theta^* = \arg\min_\theta \mathbb{E}_{\mathbf{x}_{\text{gt}}^{<i}, t \in S, i} \left[ \| G_\theta(\mathbf{x}_t^i, \mathbf{x}_{\text{gt}}^{<i}, t) - \mathbf{x}_0^i \|^2 \right]$$

变量：
- $\mathbf{x}_{\text{gt}}^{<i}$: 真实数据采的clean prefix
- $\mathbf{x}_t^i$: AR teacher采的noisy state
- $\mathbf{x}_0^i$: 对应clean frame

因为teacher是AR的，它的PF-ODE只condition on past，fixed $(\mathbf{x}_t^i, \mathbf{x}_0^{<i})$ 唯一决定 $\mathbf{x}_0^i$。**frame-level injectivity天然成立**，student能正确学到flow map，不collapse到conditional expectation。

**Stage 3: Asymmetric DMD**

跟Self Forcing一样，用Wan2.1-14B当$s_{\text{real}}$，Wan2.1-1.3B当$s_{\text{fake}}$，跑750步DMD refinement。因为init阶段已经对了，DMD能顺利提升。

### 为啥用Teacher Forcing而不是Diffusion Forcing训练AR teacher

这是paper第二个反直觉发现。Diffusion Forcing (DF, https://arxiv.org/abs/2407.01392) 训练第$i$帧时condition在noisy prefix $\mathbf{x}_t^{<i}$上，理由是这样能训练模型在不同noise level下都能work。

但作者证明DF有 **train-inference mismatch**：

- 训练时：condition在noisy prefix $\mathbf{x}_t^{<i}$
- 推理时：前面帧已经生成了，condition在clean prefix $\mathbf{x}_0^{<i}$

**Proposition 3.4**:

$$\mathbb{E}_{y \sim p_{\text{data}}(\mathbf{x}_0^{<i})} \left[ D_{\text{KL}}\left( p_{\text{DF}}(\mathbf{x}_0^i \mid y) \,\|\, p_{\text{data}}(\mathbf{x}_0^i \mid y) \right) \right] > 0$$

变量：
- $y = \mathbf{x}_0^{<i}$: clean prefix
- $p_{\text{DF}}(\mathbf{x}_0^i \mid y)$: DF训练的模型在clean prefix条件下的输出分布
- $p_{\text{data}}(\mathbf{x}_0^i \mid y)$: 真实数据分布

这个KL大于0意味着DF模型即使在最优训练下也无法匹配真实条件分布。

证明用Markov chain $X - Y - Z$（$X = \mathbf{x}_0^i, Y = \mathbf{x}_0^{<i}, Z = \mathbf{x}_t^{<i}$）加tower property反证法：如果KL=0会推出$X \perp Y$，但同一video的帧显然不独立，矛盾。

实验上Table 2验证：DF的VisionReward只有1.583，TF有3.343（翻倍）。DF的Dynamic Degree高（60 vs 50），但这是pathological——video collapse导致每个chunk独立飘，motion metric虚高。

---

## 实验结果

### 主结果（Table 1）

在100-prompt motion-rich set上：

| Method | FPS | Latency | Dynamic↑ | VisionReward↑ | Instruct↑ |
|---|---|---|---|---|---|
| Wan2.1-1.3B (bidir) | 0.78 | 103s | 61 | 5.275 | 42 |
| Self Forcing (SOTA) | 17.0 | 0.69s | 57 | 5.820 | 48 |
| **Causal Forcing** | **17.0** | **0.69s** | **68** | **6.326** | **56** |

关键数据：
- 比Self Forcing：Dynamic +19.3%, VisionReward +8.7%, Instruct +16.7%
- 比Wan2.1 bidir：throughput ×26（0.78→17 FPS），质量还略胜
- **训练budget完全一样**（3K ODE init + 750 DMD）

### Ablation的几个关键发现

**TF vs DF (Table 2)**:
- DF: Dynamic 60, VisionReward 1.583
- TF: Dynamic 50, VisionReward 3.343

DF的Dynamic虚高是collapse artifact，VisionReward才是真的quality metric。

**ODE init比较 (chunk-wise, Table 2)**:
- Self Forcing ODE + DMD: Dynamic 24, VisionReward 3.330
- Causal ODE + DMD: Dynamic 68, VisionReward 6.326

**ODE init比较 (frame-wise, Table 2)**:
- Self Forcing ODE + DMD: Dynamic 2, VisionReward 1.951
- Causal ODE + DMD: Dynamic 64, VisionReward 6.204

frame-wise差距更大（Dynamic从2到64，30倍），因为frame-wise比chunk-wise更纯causal，更暴露injectivity问题。chunk-wise有chunk内bidirectional能缓解。

**关键ablation (Appendix C.3, Fig. 9)**:

用AR teacher采的paired data，但student从bidirectional model初始化。结果跟"AR init"质量相当，都远好于Self Forcing。

**这证明performance gap由paired data决定，不由init决定**。核心是teacher必须是AR的，student init无所谓。

---

## 几个intuition总结

### 1. 为啥bidirectional→AR转换这么难

Bidirectional model的强大之处在于 **未来信息**：full attention让每帧都能看到全局。蒸馏成AR时，未来信息被强制cut掉，相当于让一个看未来的预言家退化成只看过去的凡人。这个information loss在ODE init阶段直接collapse成blurry average。

### 2. 为啥causal teacher能解决

AR model的PF-ODE只condition on past，fixed $(\mathbf{x}_t^i, \mathbf{x}_0^{<i})$ 唯一决定 $\mathbf{x}_0^i$，injectivity天然成立。这就是"causal"的mathematical content：它强制Markov性质，消除future对present的影响，让flow map well-defined。

### 3. 为啥DMD修不了init

DMD是marginal distribution matching，optimize的是KL divergence的gradient。但conditional structure一旦blurry了，marginal上看着分布差不多，conditional上已经回不去sharp了。就像你模糊一张图再做后处理，histogram能调对，细节回不来。

### 4. 为啥TF比DF对齐

训练和推理要见一样的东西。TF训练和推理都condition在clean prefix，aligned。DF训练condition在noisy prefix，推理condition在clean prefix，mismatch导致distribution shift。这是个非常general的principle：**任何train-inference gap都会被模型以奇怪的方式absorb**。

---

## 一个更general的insight

这篇paper其实点出了一个很general的pattern，我觉得可能能推广：

> **当architecture mismatch时，distillation的paired data必须由正确架构的teacher生成。否则information loss→multimodal posterior averaging→blurry regression。**

这个pattern可能出现在：
- U-Net → DiT蒸馏（虽然都是bidirectional，但inductive bias不同）
- token-level distillation（teacher和student的token粒度不同）
- 任何causal/bidirectional转换

更fundamentally，这告诉我们 **distillation本质是distribution transfer，必须保证teacher和student看到的信息对齐**。Information不对称的distillation在math上就会collapse到average。

---

## 我的几个疑问/思考

1. **Causal CD还很rudimentary**：作者用vanilla LCM实现，性能没超score distillation。Mean Flow (https://arxiv.org/abs/2505.13447)、sCM (https://arxiv.org/abs/2410.11081)的causal extension可能更强。

2. **Frame-wise vs chunk-wise的trade-off**：chunk-wise（3 latent frames/chunk）能缓解injectivity问题但牺牲纯causality。frame-wise更纯causal但需要更强teacher。是否有中间方案？

3. **TF在long video的极限**：TF condition在clean prefix，长video时context越长推理越慢。DF的noisy prefix可能更适合streaming？Appendix C.1提到DF在bidirectional long-video continuation场景是OK的（concat clean tail + noise，训练推理一致）。

4. **跟Rolling Forcing (https://arxiv.org/abs/2509.25161)、LongLive (https://arxiv.org/abs/2509.22622)、Self-Forcing++ (https://arxiv.org/abs/2510.02283)** 这些long-video方法的关系。它们也面临architectural gap吗？Causal Forcing的init策略能直接迁移吗？

5. **跟reward model guidance的结合**：如果AR teacher训练时加RLHF或reward weighting，下游全链条受益。这是interactive video generation的自然下一步。

6. **Memory机制**：TF的clean prefix在长video会变成memory bottleneck。Deep Forcing (https://arxiv.org/abs/2512.05081)用sink tokens + compression，可能跟Causal Forcing结合。

---

## 总结一句话

**Self Forcing的bug是用bidirectional teacher蒸馏AR student，违反frame-level injectivity，MSE collapse成blurry conditional mean。Causal Forcing的fix是用AR teacher做ODE init，保证causality，让student学到sharp flow map，然后DMD refine。**

Project page: https://thuml.github.io/CausalForcing.github.io/

希望这次讲得更像人话了。如果你想深挖某个proof或者某个extension（比如causal Mean Flow），我可以继续展开。

---

# Causal Forcing: Autoregressive Diffusion Distillation Done Right 技术深度解读

Andrej, 这篇paper来自清华团队（Hongzhou Zhu, Min Zhao等，朱军老师组），主要解决的问题非常fundamental：**为什么把bidirectional video diffusion model蒸馏成autoregressive (AR) student时会掉性能**。他们找到了一个理论上很优雅的解释——**frame-level injectivity**——并提出了Causal Forcing这个三阶段pipeline。我会尽量从第一性原理角度构建你的intuition。

---

## 1. 问题背景：实时交互视频生成的architectural gap

### 1.1 为什么需要AR video diffusion

Bidirectional video diffusion models（如Wan2.1, CogVideoX, HunyuanVideo）用full attention一次性生成所有帧，质量高但有两个致命问题：
- **latency bottleneck**：多步采样（通常30-50步）需要等整段视频算完才能显示第一帧
- **无交互性**：用户无法基于已生成内容steer后续帧

AR video diffusion（Pyramid Flow, MAGI-1, SkyReels-V2, Self Forcing）通过autoregressive factorization解决这个问题：

$$p_\theta(\mathbf{x}_0^{1:N}) = \prod_{i=1}^{N} p_\theta(\mathbf{x}_0^{i} \mid \mathbf{x}_0^{<i})$$

其中$\mathbf{x}_0^{i}$是第$i$帧的clean latent，$\mathbf{x}_0^{<i}$是之前的clean frames。这样就能streaming生成、交互控制。Google的Genie 3 (https://arxiv.org/abs/2507.07971) 就是这种范式，用于interactive world modeling。

但AR model也有问题：每帧内部还要做多步diffusion，太慢。所以需要distillation到few-step。

### 1.2 现有方法的pipeline

CausVid (https://arxiv.org/abs/2504.15232) 和 Self Forcing (https://arxiv.org/abs/2506.22590) 都用两阶段：

**Stage A (ODE distillation initialization)**: 用bidirectional teacher采样PF-ODE轨迹，训练AR student把noisy intermediate映射回clean video
**Stage B (Asymmetric DMD)**: 用DMD (https://arxiv.org/abs/2310.14189) 进一步refine

关键观察：即使从同一个bidirectional teacher蒸馏，Self Forcing（AR student）显著差于标准DMD（bidirectional student）。这个gap无法用"少步采样"解释，肯定有更深层的architectural问题。

---

## 2. 核心insight：architectural gap的根本原因

### 2.1 关键实验：DMD能否修复architectural gap

作者做了一个controlled experiment：用bidirectional few-step model直接初始化AR student（消除sampling-step gap，只保留architectural gap），然后跑DMD。结果还是显著差于标准DMD。这告诉我们：

**DMD stage无法修复architectural gap，必须从ODE initialization阶段解决**。

### 2.2 Frame-level injectivity的数学本质

这是paper最核心的贡献。我详细讲一下。

**Standard ODE distillation**（bidirectional→bidirectional）的目标：

$$\theta^* = \arg\min_\theta \mathbb{E}_{t, \mathbf{x}_t^{1:N}} \left[ \| G_\theta(\mathbf{x}_t^{1:N}, t) - \mathbf{x}_0^{1:N} \|^2 \right]$$

变量含义：
- $t \in [0,1]$：diffusion时间步（0表示clean，1表示纯噪声）
- $\mathbf{x}_t^{1:N} \in \mathbb{R}^{N \cdot d}$：N帧视频的noisy latent，$d$是每帧latent维度
- $\mathbf{x}_0^{1:N}$：对应的clean video
- $G_\theta$：student flow map，把$(\mathbf{x}_t, t)$映射到$\mathbf{x}_0$
- $(\mathbf{x}_t^{1:N}, \mathbf{x}_0^{1:N})$在bidirectional teacher的同一个PF-ODE轨迹上

PF-ODE的定义（flow matching形式）：

$$\mathrm{d}\mathbf{x}_t = \mathbf{v}_\theta(\mathbf{x}_t, t)\, \mathrm{d}t, \quad \mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, I), \quad t: T \to 0$$

其中$\mathbf{v}_\theta$是velocity prediction，noise schedule取$\alpha_t = 1-t, \sigma_t = t$，于是$\mathbf{v}_t = \boldsymbol{\epsilon} - \mathbf{x}_0$。

**Injectivity的必要条件**：MSE回归要well-defined，必须保证每个$\mathbf{x}_t$唯一对应一个$\mathbf{x}_0$，否则最优解会collapse成conditional expectation（Bishop的PRML里有详细推导，https://www.springer.com/gp/book/9780387310732）。

对bidirectional teacher，flow map $\phi^{\text{Bi}}: (\mathbf{x}_t^{1:N}, t) \mapsto \mathbf{x}_0^{1:N}$ 在video level天然injective（这是PF-ODE的基本性质，Liu et al. 2022 https://arxiv.org/abs/2209.03003 证明过）。

**但AR student需要在frame level injective**！为什么？因为AR生成是逐帧的，第$i$帧的生成只依赖$(\mathbf{x}_t^i, \mathbf{x}_0^{<i})$，看不到future frames $\mathbf{x}_t^{>i}$。

### 2.3 Lemma 3.2的形式化

**Definition 3.1 (Frame-level injectivity)**: 对AR flow map $\phi^{\text{AR}}: (\mathbf{x}_t^i, t) \mapsto \mathbf{x}_0^i$，frame-level injectivity要求：

$$\forall t \in (0,1], \forall \{\mathbf{x}_t^j\}_{j=1}^N, \{\mathbf{y}_t^j\}_{j=1}^N: \quad \forall i \in [N], \mathbf{x}_t^i = \mathbf{y}_t^i \Rightarrow \phi^{\text{AR}}(\mathbf{x}_t^i, t) = \phi^{\text{AR}}(\mathbf{y}_t^i, t)$$

直觉：固定第$i$帧的noisy state，无论其他帧是什么样，第$i$帧的clean版本必须唯一确定。

**Lemma 3.2 (Frame-level non-injectivity of bidirectional PF-ODE)**：设$\mathbf{x}_t^{1:N}$满足bidirectional diffusion model的PF-ODE，$\phi^{\text{Bi}}(\mathbf{x}_t^{1:N}, t)^i$是第$i$帧的flow map。如果$\phi^{\text{Bi}}(\mathbf{x}_t^{1:N}, t)^i$关于$\mathbf{x}_t^{\text{other}} := \mathbf{x}_t^{[N]\setminus\{i\}}$不是a.e. constant（这个条件对DiT-based bidirectional model几乎总成立，因为attention会让所有帧互相影响），那么：

$$\forall t \in (0,1], \forall \mathbf{x}_t^{1:N}, \exists \mathbf{y}_t^{1:N}: \mathbf{y}_t^i = \mathbf{x}_t^i \text{ 且 } \phi^{\text{Bi}}(\mathbf{x}_t^{1:N}, t)^i \neq \phi^{\text{Bi}}(\mathbf{y}_t^{1:N}, t)^i$$

且更关键的是：

$$\mathbb{P}\left(\text{Var}\left(\phi^{\text{Bi}}(\mathbf{x}_t^{1:N}, t)^i \mid \mathbf{x}_t^i, t\right) > 0\right) > 0$$

**证明sketch**（见Appendix B.1）：固定$(\mathbf{u}_1, t) := (\mathbf{x}_t^i, t)$，定义$f_{\mathbf{u}_1, t}(\mathbf{z}) := \phi^{\text{Bi}}((\mathbf{u}_1, \mathbf{z}), t)^i$，其中$\mathbf{z} = \mathbf{x}_t^{\text{other}}$。由假设，$f_{\mathbf{u}_1, t}$不是a.e. constant，所以对任意$\mathbf{z}_1$，存在$\mathbf{z}_2$使$f_{\mathbf{u}_1, t}(\mathbf{z}_2) \neq f_{\mathbf{u}_1, t}(\mathbf{z}_1)$。然后因为$\mathbf{x}_T \sim \mathcal{N}(0, I)$且$p_t(\mathbf{x}_t) > 0$ a.e.，用measure theory的标准论证（Kallenberg的Foundations of Modern Probability https://link.springer.com/book/10.1007/978-1-4757-4015-8），证明collision发生在non-zero measure的集合上。

**直觉**：bidirectional model去噪第$i$帧时会用所有帧的信息（full attention）。固定$\mathbf{x}_t^i$，改变future frames $\mathbf{x}_t^{>i}$会得到不同的clean $\mathbf{x}_0^i$。Self Forcing训练AR student时只给它看$\mathbf{x}_t^i$（隐藏了$\mathbf{x}_t^{>i}$），信息丢失，违反injectivity。

### 2.4 Proposition 3.3：为什么会产生blurry video

这是最关键的consequence。用MSE回归训练$G_\theta$：

$$\theta^* = \arg\min_\theta \mathbb{E}_{\mathbf{x}_t^{1:N}, t} \left[ \| G_\theta(\mathbf{x}_t^i, t) - \mathbf{x}_0^i \|^2 \right]$$

最优解是conditional expectation（PRML第41页的标准结果）：

$$G_\theta^*(\mathbf{x}_t^i, t) = \mathbb{E}[\mathbf{x}_0^i \mid \mathbf{x}_t^i, t]$$

**关键claim**：这个conditional expectation不服从data distribution $p_{\text{data}}(\mathbf{x}_0^i)$。

证明用$L^2$-orthogonal projection identity：

$$\mathbb{E}\|Y\|^2 = \mathbb{E}\|\widehat{Y}\|^2 + \mathbb{E}[\text{Var}(Y \mid U, t)]$$

其中$Y := \mathbf{x}_0^i$, $U := \mathbf{x}_t^i$, $\widehat{Y} := \mathbb{E}[Y \mid U, t]$。由Lemma 3.2，$\mathbb{E}[\text{Var}(Y \mid U, t)] > 0$，所以$\mathbb{E}\|\widehat{Y}\|^2 < \mathbb{E}\|Y\|^2$。如果$\widehat{Y}$和$Y$同分布，二阶矩应该相等，矛盾。所以$\widehat{Y} \not\sim Y$。

**物理直觉**：conditional expectation是L2意义下的"最优平均"，对应的就是blurry image——因为同一noisy state对应多个不同clean frame时，回归到的就是它们的mean，自然blur。这就是Fig. 3c看到的现象。

---

## 3. Causal Forcing方法

### 3.1 三阶段pipeline

**Stage 1: Autoregressive diffusion training (Teacher Forcing)**

训练一个AR diffusion model作为后续teacher。关键发现：**Teacher Forcing优于Diffusion Forcing**。

**Stage 2: Causal ODE distillation**

用Stage 1的AR teacher采样PF-ODE轨迹，训练AR student。因为teacher是AR的，它的PF-ODE在frame level天然injective，避免了Proposition 3.3的collapse。

**Stage 3: Asymmetric DMD**

标准的DMD（Self Forcing的做法），用bidirectional Wan2.1-14B作为$s_{\text{real}}$，Wan2.1-1.3B作为$s_{\text{fake}}$。

### 3.2 为什么TF优于DF（Proposition 3.4）

这是paper另一个重要发现，推翻了"Diffusion Forcing更好"的常见信念。

**Teacher Forcing (TF)**: 训练第$i$帧时condition on clean prefix $\mathbf{x}_0^{<i}$
**Diffusion Forcing (DF)** (https://arxiv.org/abs/2407.01392): 训练第$i$帧时condition on noisy prefix $\mathbf{x}_t^{<i}$

**Proposition 3.4**: 在regularity conditions下：

$$\mathbb{E}_{y \sim p_{\text{data}}(\mathbf{x}_0^{<i})} \left[ D_{\text{KL}}\left( p_{\text{DF}}(\mathbf{x}_0^i \mid y) \,\|\, p_{\text{data}}(\mathbf{x}_0^i \mid y) \right) \right] > 0$$

其中$p_{\text{DF}}$是DF训练的模型，$y$是clean prefix。**直觉**：

- DF训练时见的是noisy prefix $\mathbf{x}_t^{<i}$，inference时见的是clean prefix $\mathbf{x}_0^{<i}$（因为前几帧已经生成了），这是train-inference distribution mismatch
- TF训练和inference都见clean prefix，aligned

证明用Markov chain $X - Y - Z$（$X = \mathbf{x}_0^i, Y = \mathbf{x}_0^{<i}, Z = \mathbf{x}_t^{<i}$）和tower property，再反证法推出如果KL=0会imply $X \perp Y$，与A2矛盾。

实现细节：TF的实现是concatenate clean video和noisy video，用causal attention mask让$\mathbf{x}_t^i$ attend to $\mathbf{x}_0^{<i}$（参考MAGI-1, https://arxiv.org/abs/2505.13211）。

### 3.3 Causal ODE distillation的形式化

训练目标：

$$\theta^* = \arg\min_\theta \mathbb{E}_{\mathbf{x}_{\text{gt}}^{<i}, t \in S, i} \left[ \| G_\theta(\mathbf{x}_t^i, \mathbf{x}_{\text{gt}}^{<i}, t) - \mathbf{x}_0^i \|^2 \right]$$

变量含义：
- $\mathbf{x}_{\text{gt}}^{<i}$: 从real dataset采样的clean prefix（用于conditioning）
- $S$: predefined timesteps set，inference时用$\{1, 0.9375, 0.8333, 0.625\}$（4步采样）
- $\mathbf{x}_t^i$: 由AR teacher采样得到的第$i$帧的noisy state
- $\mathbf{x}_0^i$: 对应的clean frame

因为teacher是AR的，PF-ODE trajectory在frame level满足injectivity（Eq. (4)），student能正确学到flow map。

### 3.4 扩展到Causal Consistency Models

这部分是bonus。Consistency Distillation (https://arxiv.org/abs/2303.00948)的目标：

$$\theta^* = \arg\min_\theta \mathbb{E}_{\mathbf{x}_{\text{gt}}, \boldsymbol{\epsilon}, t, i} \left[ w(t) \, d\left(G_\theta(\mathbf{x}_t^i, \mathbf{x}_{\text{gt}}^{<i}, t),\, G_{\theta^-}(\hat{\mathbf{x}}_{t-\Delta t}^i, \mathbf{x}_{\text{gt}}^{<i}, t - \Delta t)\right) \right]$$

变量含义：
- $w(t)$: time-dependent weight
- $d(\cdot, \cdot)$: distance metric
- $\hat{\mathbf{x}}_{t-\Delta t}^i$: 用AR teacher解ODE从$\mathbf{x}_t^i$走一步到$t - \Delta t$
- $\theta^-$: EMA of $\theta$, stop-gradient

由于flow matching用v-prediction，可以直接写：

$$G_\theta(\mathbf{x}^i, \mathbf{x}_{\text{gt}}^{<i}, t) = \mathbf{x}^i - t \cdot \mathbf{v}_\theta(\mathbf{x}^i, \mathbf{x}_{\text{gt}}^{<i}, t)$$

这个形式天然满足boundary condition $G_\theta(\mathbf{x}, \cdot, 0) \equiv \mathbf{x}$，不需要额外的skip connection设计。

---

## 4. 实验结果详解

### 4.1 主实验（Table 1）

数据来自100-prompt motion-rich set + VBench (https://arxiv.org/abs/2311.16235) + VisionReward (https://arxiv.org/abs/2412.13776)：

| Method | Throughput (FPS) | Latency (s) | Dynamic↑ | VisionReward↑ | Instruct↑ | User Rating↓ |
|---|---|---|---|---|---|---|
| Wan2.1-1.3B (bidirectional) | 0.78 | 103 | 61 | 5.275 | 42 | 2.29 |
| Self Forcing (SOTA AR distilled) | 17.0 | 0.69 | 57 | 5.820 | 48 | 2.87 |
| **Causal Forcing (Ours)** | **17.0** | **0.69** | **68** | **6.326** | **56** | **1.64** |

关键数据：
- 比Self Forcing：Dynamic Degree +19.3%（57→68），VisionReward +8.7%（5.82→6.326），Instruction Following +16.7%（48→56）
- 比bidirectional Wan2.1：throughput提升2079%（0.78→17.0 FPS），同时质量相当或更好
- **相同训练budget**（3K ODE initialization + 750 DMD steps）

### 4.2 Ablation（Table 2）

#### 4.2.1 AR diffusion training: TF vs DF

| Method | Dynamic↑ | VisionReward↑ | Instruct↑ |
|---|---|---|---|
| Diffusion Forcing | 60 | 1.583 | 30 |
| Teacher Forcing | 50 | 3.343 | 32 |

DF的Dynamic Degree高（60 > 50）但VisionReward极低（1.583 vs 3.343）。这个dynamic虚高是因为video collapse产生的pathological motion（每个chunk独立飘），不是真实动态。TF在所有"质量"指标上碾压DF，印证Proposition 3.4。

#### 4.2.2 ODE initialization比较（chunk-wise）

| Method | Dynamic↑ | VisionReward↑ | Instruct↑ |
|---|---|---|---|
| Self Forcing's ODE + DMD | 24 | 3.330 | 38 |
| **Causal ODE + DMD** | **68** | **6.326** | **56** |

VisionReward +90.0%，Dynamic Degree +183.3%，Instruction Following +47.4%。

#### 4.2.3 ODE initialization比较（frame-wise）

| Method | Dynamic↑ | VisionReward↑ | Instruct↑ |
|---|---|---|---|
| Self Forcing's ODE + DMD | 2 | 1.951 | -4 |
| **Causal ODE + DMD** | **64** | **6.204** | **42** |

frame-wise设置下，差距更夸张：Dynamic Degree +3100%，VisionReward +218.0%。frame-wise比chunk-wise更暴露injectivity问题，因为frame-wise时每一帧都独立处理，更依赖frame-level injectivity。

#### 4.2.4 其他AR training策略（Table 3）

PFVG (https://arxiv.org/abs/2510.01784), BAgger (https://arxiv.org/abs/2512.12080), Resampling Forcing (https://arxiv.org/abs/2512.15702) 都没有显著超过TF。这些主要设计给long-video，5s设置下没发挥空间。

### 4.3 关键ablation：哪个因素决定ODE distillation质量（Appendix C.3）

实验设计：用AR teacher采样的paired data $\mathcal{D}_{\text{Causal}}$训练student，但student初始化从bidirectional model。结果与"AR teacher采样 + AR init"质量相当，都远好于Self Forcing。

**结论**：performance gap由paired data construction决定（即teacher应该是AR的），不由student initialization决定。这强化了Lemma 3.2的核心论点。

### 4.4 Multi-step AR作为DMD init（Appendix C.2, Table 4）

| Method | Dynamic↑ | VisionReward↑ | Instruct↑ |
|---|---|---|---|
| Self Forcing's ODE + DMD | 24 | 3.330 | 38 |
| Multi-step AR diffusion + DMD | 66 | 5.863 | 48 |
| **Causal ODE + DMD** | **68** | **6.326** | **56** |

直接用multi-step AR init DMD能拿到中间性能，但few-step时AR model会产生chunk间abrupt transitions（Fig. 7）。Causal ODE通过4-step distillation把这种abruptness抹平，让DMD有更好的起点。

---

## 5. 实现细节（Appendix D）

### 5.1 Training recipe

- Base model: Wan2.1-T2V-1.3B (https://arxiv.org/abs/2503.20314)，81-frame @ 832×480
- $\mathcal{D}_{\text{Bi}}$: 3K samples from Wan bidirectional with VidProM prompts (https://arxiv.org/abs/2412.14067)
- AR diffusion training: 2K steps
- Causal ODE distillation: 1K steps（初始化自AR teacher）
- Asymmetric DMD: 750 steps on VidProM，$s_{\text{real}}$=Wan2.1-14B, $s_{\text{fake}}$=Wan2.1-1.3B
- Batch size 64, Adam, lr $2 \times 10^{-6}$, $\beta_1 = 0.9, \beta_2 = 0.999$
- Inference: 4 steps at $t \in \{1, 0.9375, 0.8333, 0.625\}$
- Chunk-wise: each chunk = 3 latent frames

### 5.2 Causal CD的设置

- LCM-style (https://arxiv.org/abs/2310.04378)，48 discretized timesteps
- UniPC solver, EMA rate 0.99
- 3K training steps on $\mathcal{D}_{\text{Bi}}$

---

## 6. 直觉总结与思考

让我试着构建一个unified intuition：

### 6.1 整个故事的因果链

```
Bidirectional Teacher的PF-ODE → frame-level non-injectivity (Lemma 3.2)
                                      ↓
                              MSE回归collapse到conditional expectation (Prop 3.3)
                                      ↓
                              Self Forcing的ODE init得到blurry model
                                      ↓
                              DMD无法完全修复（gap太大，Fig 2实验证明）
                                      ↓
                              最终AR student质量差
```

Causal Forcing的fix：
```
AR Teacher (TF训练) → 满足frame-level injectivity (by construction)
                                      ↓
                              ODE distillation学到正确flow map
                                      ↓
                              DMD拿到好init，refine到位
                                      ↓
                              高质量few-step AR model
```

### 6.2 为什么这个工作重要

1. **理论清晰**：不是empirical trick，而是从ODE distillation的必要条件出发，用measure theory严格证明现有方法的理论缺陷
2. **简单优雅**：核心fix就是"teacher应该是AR的"，符合causality第一性原理
3. **实验充分**：ablation隔离了init、data、training strategy各个因素，结论非常robust
4. **practical impact**：real-time 17 FPS，延迟0.69s，质量超过bidirectional SOTA，对交互视频应用（world modeling, game sim, embodied AI）意义重大

### 6.3 一些值得深挖的点

- **Causal CD还是rudimentary**：作者明确说vanilla LCM实现没超score distillation，但paves the way for Mean Flow (https://arxiv.org/abs/2505.13447), sCM (https://arxiv.org/abs/2410.11081), score-regularized CD (https://arxiv.org/abs/2510.08431) 等更强方法的causal extension
- **DF在video continuation场景仍合法**：Appendix C.1澄清，DF的原始设计是给bidirectional model做long-video continuation（concat clean tail + noise），训练inference一致，所以Proposition 3.4不覆盖这种情况
- **frame-wise vs chunk-wise**：chunk-wise（3 latent frames/chunk）能缓解injectivity问题（因为chunk内有bidirectional），frame-wise更纯causal、更暴露问题
- **DiT的attention map证明**：Lemma B.1的关键假设"$\phi$不a.e. constant"由attention map的非constness验证（https://arxiv.org/abs/2502.01776, https://arxiv.org/abs/2511.20123）

### 6.4 与其他相关工作的联系

- **Self Forcing++ (https://arxiv.org/abs/2510.02283)**: 长视频场景的Self Forcing扩展，应该也面临architectural gap问题
- **Rolling Forcing (https://arxiv.org/abs/2509.25161)**: 类似real-time AR long video，可能也受益于causal ODE init
- **Deep Forcing (https://arxiv.org/abs/2512.05081)**: training-free长视频，用sink tokens + compression
- **Yume-1.5 (https://arxiv.org/abs/2512.22096)**, **WorldPlay (https://arxiv.org/abs/2512.14614)**: interactive world model，是Causal Forcing直接受益的应用方向
- **Vidarc (https://arxiv.org/abs/2512.17661)**: embodied video diffusion用于closed-loop control，AR + real-time是必须

### 6.5 与其它distillation方法的对比框架

| Distillation类型 | Teacher | Student | Injectivity level | 论文方法 |
|---|---|---|---|---|
| Standard CD | Bidir | Bidir | Video ✓ | Song et al. 2023 |
| Asymmetric CD | Bidir | AR | Frame ✗ (Lemma 3.2) | CausVid baseline |
| Causal CD (new) | AR | AR | Frame ✓ | Causal Forcing |
| Standard DMD | Bidir | Bidir | Video ✓ | Yin et al. 2024 |
| Asymmetric DMD (Self Forcing) | Bidir | AR | Frame ✗ | Huang et al. 2025a |
| Causal ODE init + Asym DMD | AR (for init) + Bidir (for DMD) | AR | Frame ✓ (init阶段) | **Causal Forcing** |

### 6.6 我的几个小思考

1. **Injectivity与Bayesian view**：blurry conditional expectation本质是后验均值，这是Bayesian optimal regression在multimodal posterior下的通病。Generative model要避免multimodal averaging，单模态MSE会collapse。这让我想到diffusion的noise conditioning其实隐式地给posterior做某种resolution，但bidirectional→AR的转换把这个resolution破坏了。

2. **Causal为什么work**：AR model的PF-ODE只condition on past，所以fixed $(\mathbf{x}_t^i, \mathbf{x}_0^{<i})$ 唯一决定$\mathbf{x}_0^i$，自然injective。这就是"causal"的mathematical content——它强制Markov性质，消除了future对present的影响。

3. **DMD的角色**：DMD本质上是用KL gradient做distribution matching，能fix marginal distribution但fix不了conditional structure（一旦init已经blurry）。这解释了Fig 2的实验现象。

4. **可能的进一步改进**：
   - 用Mean Flow (https://arxiv.org/abs/2505.13447)的causal extension，避免ODE paired data采样开销
   - 把Causal ODE init直接合并到DMD（end-to-end training）
   - 用更强的distance metric（Wasserstein, perceptual）替代MSE regression，可能放宽injectivity要求
   - 在AR teacher训练时加reward model guidance，让teacher更好（这样下游全链条受益）

---

## 7. 项目与代码

- Project page: https://thuml.github.io/CausalForcing.github.io/
- Wan base model: https://github.com/Wan-Video/Wan2.1
- Self Forcing (baseline): https://arxiv.org/abs/2506.22590, https://self-forcing.github.io/
- CausVid (另一个baseline): https://github.com/tianweiy/CausVid
- VBench (eval): https://github.com/Vchitect/VBench-project
- VisionReward (eval): https://github.com/THUDM/VisionReward

---

## 8. 总结

这篇paper给我的最大启发是：**distillation的theoretical soundness比empirical tuning更重要**。Self Forcing看起来很work（SOTA），但底层有个measure-theoretic flaw（frame-level non-injectivity），导致blurry conditional expectation。Causal Forcing不是简单换teacher，而是从ODE distillation的必要条件重新推导应该怎么做。

更深层地，这反映了一个普遍pattern：**当architecture mismatch（bidir↔AR）时，distillation的paired data必须由正确架构的teacher生成，否则会引入information loss→multimodal posterior averaging→blurry regression**。这个insight可能可以推广到其他distillation场景（如image distillation中的U-Net→DiT转换、token-level distillation等）。

希望这个解读能帮你build intuition。如果想深挖某个具体proof或者extension，我可以继续展开。
