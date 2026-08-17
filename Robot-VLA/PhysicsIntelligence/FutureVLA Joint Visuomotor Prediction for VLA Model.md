---
source_pdf: FutureVLA Joint Visuomotor Prediction for VLA Model.pdf
paper_sha256: 46b3cf66ea3f63be29e7afb02e0175682f9ccc07c5622f6e19da8f5c50da1017
processed_at: '2026-08-04T11:24:29-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

我们来用最直白的话把这篇 FutureVLA 捋一遍，核心就一件事：教 VLA 模型学会"未卜先知"，但方法必须极其讲究，不能瞎费算力。

### 1. 痛点在哪：为什么以前的 Future Guidance 都不行

现在的 VLA 模型大多是 reactive 的，看到当前帧就输出 action。如果想让它有预见性，以前的套路有两类，但都踩了坑：

*   **Explicit guidance (比如 [WorldVLA](https://arxiv.org/abs/2506.21539))**：直接预测未来的视频帧。这就像教机器人抓杯子，结果它天天在脑海里渲染背景墙的阴影怎么变、光线怎么移。大量的 capacity 被用来搞 visual rendering 了，真正的 motor intent 被淹没了。这叫 visually-dominated entanglement。
*   **Implicit guidance (比如 [LAPA](https://arxiv.org/abs/2410.11758), [Villa-X](https://arxiv.org/abs/2507.23682))**：抽取稀疏的两帧（首尾），猜中间的 latent vector 变化。机器人的动作是连续的 16 步 action chunk，你只抽两帧相当于把连续动画变成了翻页漫画，这叫 temporal discontinuity。同时，它本质上还是在重建视觉特征，latent space 依然不干净。

### 2. 核心直觉：物理属性不同，必须 Decouple + Condition

这篇 paper 的 intuition 非常物理：视觉和动作在现实世界的属性完全不同。
*   **Visual perception** 提供**静态空间约束**（哪里有桌子、哪里有杯子）。
*   **Motor execution** 包含**连续动态演化**（手臂该怎么轨迹运动）。

所以，必须把它们 decouple（解耦）开学。但是 pure decouple 不行，动作不能脱离环境凭空产生。所以解耦之后，motor stream 还得回过头去问 visual stream："我要做动作了，障碍物在哪？空间 affordance 怎么样？" 这就是 visually-conditioned supervision decoupling。

### 3. 架构拆解：到底怎么做的

整个 pipeline 分两步走。

#### Step 1: Joint Visuomotor Pretraining (学预知能力)

输入连续的 17 帧 RGB 视频，用 [WAN 2.2](https://arxiv.org/abs/2503.20314) 的 3D-VAE 压成 temporal tokens。为什么用 3D-VAE？因为它在时间维度上做 4x 下采样，能天然捕捉 inter-frame motion dynamics，比一帧一帧用 VQ-GAN 编码要好得多，而且算得更快。

拿到 tokens 后，**核心创新 Joint Visuomotor Gating** 登场。把 tokens 劈成两半：一半是 visual tokens $V_n$，一半是 motor tokens $M_n$。

*   **Visual stream**：只负责重建第一帧的 latent embedding。为啥重建第一帧？因为第一帧是静态锚点。如果去重建最后一帧，又掉进预测未来视觉的坑里了。Table 13 的 ablation 证明了这一点：重建第一帧 71.9%，重建最后一帧掉到 68.8%。
*   **Motor stream + Gating**：Motor tokens 先做 self-attention 提取纯动力学信息。然后通过 cross-attention 去 query visual tokens 获取空间约束。关键公式如下：

$$
M_{n+1} = \sigma(r) \odot M_n'' + M_n'
$$

变量解释：
*   $M_n'$：motor tokens 经过 self-attention 精炼后的纯动作表征。
*   $M_n''$：motor tokens 作为 query，去 cross-attention visual tokens 后得到的带视觉约束的表征。
*   $r$：一个可学习的标量参数。
*   $\sigma(\cdot)$：sigmoid 函数，把 $r$ 压缩到 $(0,1)$ 之间，控制要吸收多少视觉信息。
*   $\odot$：element-wise multiplication。
*   $M_{n+1}$：最终融合了视觉约束和纯动力学的 motor embedding。

这个 gating 机制迭代 3 次，最终产出 physically grounded joint visuomotor embeddings $M_f$。Table 5 的 ablation 极其雄辩：只给多帧输入不 decouple，性能反而掉 4.1%；decouple 了涨 7.2%；加上 gating 再涨 6.3%。逻辑完全闭环。

#### Step 2: VLA Post-training (把预知能力内化)

这一步解决怎么把 $M_f$ 塞进下游 VLA (比如用 [Qwen3-VL](https://arxiv.org/abs/2505.09388) 做 backbone)。这里用了 latent alignment 策略。

把预训练阶段 freeze 住，提取出 $M_f$ 当做 semantic anchors。下游 VLA 吃当前单帧图和 language instruction，产出中间特征 $F_r$，过一个 lightweight adapter 变成 $F_a$，然后用 MSE loss 硬拉对齐：

$$
\mathcal{L}_2 = \beta \| M_f - F_a \|_2 + \mathcal{L}_A
$$

变量解释：
*   $F_a$：下游 VLA 适配后的中间表征。
*   $M_f$：预训练阶段提取的带未来预知能力的 joint visuomotor embedding。
*   $\beta$：平衡系数，用 cosine decay 慢慢减小。
*   $\mathcal{L}_A$：VLA 本身的 action loss (可以是 OFT 的 MAE，也可以是 [Flow Matching](https://arxiv.org/abs/2210.02747) 的 vector field loss)。

这个设计最大的好处是：**推理时不需要多帧视频输入**。VLA 已经把 future dynamics 内化到自己的权重里了，架构完全不用改。

### 4. 实验数据背后的 Intuition

看实验数据，几个点特别 build intuition：

*   **长时序任务起飞**：在 SimplerEnv 的 Google Robot 上，"Put in Drawer" 这种长程任务，OpenVLA-OFT 只有 0.9%，FutureVLA 干到了 74.1%。这符合直觉，因为预见性对长程规划最关键。
*   **抗扰动极强**：在 [LIBERO-Plus](https://arxiv.org/abs/2510.13626) 的 perturbation 测试中，WorldVLA 在 Camera 扰动下崩溃到 0.1%，因为它过度依赖视觉重建。FutureVLA 还有 59.7%，因为 decoupling 机制保护了 motor stream 不受视觉噪音污染。Table 14 显示，加视觉噪音后，LAPA 的 embedding 偏差是 FutureVLA 的 56 倍。
*   **Contact-rich 任务大胜**：真机实验擦白板这种需要持续力控的任务，FutureVLA 比 $\pi_0$ 高出 26.7%。这说明 motor stream 学到了真实的物理动力学，而不只是视觉残差。

### 5. 发散联想与未来延伸

这篇 paper 的哲学其实和 LeCun 的 [JEPA](https://arxiv.org/abs/2301.08243) 高度一致：在 latent space 预测，坚决不碰 pixel reconstruction。FutureVLA 把这个思想在 robotics 场景推到了极致，通过 architectural decoupling 强制分离视觉和动作。

顺着这个直觉往下想，有很多可延展的联想：

1.  **多模态 Gating 的扩展**：paper 在 limitations 里提到 contact-rich 任务纯视觉不够。既然 visual 和 motor 可以 decouple 然后 cross-attention，那 tactile (触觉) 和 force-torque 完全可以作为一个新的 stream 加入进来。公式完全可以扩展成多流的 gating 机制，让 motor stream 同时 query visual 和 tactile tokens。
2.  **与 Model-Based RL 的结合**：这个 $M_f$ embedding 本质上就是一个 learned world model 的状态。在 model predictive control (MPC) 中，完全可以用这个 embedding 在 latent space 做 trajectory optimization，推导出最优的 action chunk，而不需要跑昂贵的 visual rollout。
3.  **Action Chunk 长度的自适应**：实验发现 17 帧 (对应 action chunk 16) 最好，21 帧就掉点。这暗示了时序对齐的刚性。如果引入一个动态的时间 warping 机制，或者让 gating 的 $r$ 参数随时间步变化，可能会打破这个刚性限制，处理更长的 horizon。
4.  **V-JEPA 2 的具身化**：[V-JEPA 2](https://arxiv.org/abs/2506.09985) 已经证明了视频预测可以用于理解和规划。FutureVLA 提供了一个将通用视频预测 embedding 对齐到具体机器人 action space 的范本，未来完全可以把 3D-VAE 换成 V-JEPA 2 的 encoder，学更泛化的物理常识。

总的来说，这篇 paper 用极其简单的 gating 机制和清晰的 decoupling 逻辑，解决了 VLA future guidance 里的 visual entanglement 和 temporal discontinuity 两个顽疾，非常 elegant。

---

# FutureVLA 深度解析：Joint Visuomotor Prediction for VLA

让我从intuition出发，逐层拆解这篇paper的核心思想、技术细节和实验逻辑。

## 1. 问题动机：为什么现有future guidance都不行

Future guidance在VLA中是个很自然的思想——agent如果能预测future state，current action就能被约束得更合理。LeCun在 [Path towards autonomous machine intelligence](https://openreview.net/pdf?id=bZ3aPdQwd6) 中提到的world model思想，以及JEPA ([Assran et al., CVPR 2023](https://arxiv.org/abs/2301.08243)) 的joint-embedding predictive架构，都体现了这一思想。

但现有两类future guidance方法都有根本性问题：

**Explicit methods** (如 [WorldVLA](https://arxiv.org/abs/2506.21539), [DreamVLA](https://arxiv.org/abs/2507.04447), [Predictive Inverse Dynamics](https://arxiv.org/abs/2412.15109))：预测future video frames。这相当于让模型同时学习"如何渲染场景"和"如何执行动作"，结果capacity被任务无关的visual appearance占用太多——比如背景纹理变化、光照变化这些与motor intent无关的信号都会被强行预测。

**Implicit methods** (如 [LAPA](https://arxiv.org/abs/2410.11758), [UniVLA](https://arxiv.org/abs/2505.06111), [Villa-X](https://arxiv.org/abs/2507.23682))：用sparse frame pairs学习latent embedding，然后用forward/inverse dynamics model重建future observation。问题有两个：
1. **Temporal discontinuity**: sparsely sampled frame pairs和robotic action chunk的连续多步特性不匹配
2. **Visual reconstruction仍然纠缠**: 隐式方法本质上还是在重建future visual observation，把task-irrelevant appearance变化和真实physical state transition耦合在一起

作者总结的两个fundamental flaws：
- **Visually-dominated embedding entanglement**: latent space被迫encode visual residual changes而非纯motor dynamics
- **Temporal discontinuity**: sparse sampling破坏了时序连续性

## 2. 核心insight：visuomotor decoupling + conditioning

这个insight非常物理直觉：
- **Visual perception** 提供**静态空间约束**（geometry, affordance, object positions）
- **Motor execution** 包含**连续动态演化**（action chunk, trajectory）

这两者属性完全不同，不能混在一起学。作者提出**visually-conditioned supervision decoupling**：先把visual和motor分开学，再用gating让motor selectively query visual，这样motor stream就被**显式条件化**于visual constraints，但不会被visual rendering污染。

这个思想让我想到V-JEPA 2 ([Assran et al., 2025](https://arxiv.org/abs/2506.09985))中predict latent representations而非pixels的思路，但FutureVLA走得更远——它直接在representation层面把visual和motor分成两个stream。

## 3. 架构详解：两阶段pipeline

### Stage 1: Joint Visuomotor Pretraining

#### 3.1.1 Visual Tokenization
输入17帧连续RGB视频，每帧$224 \times 224$。用 [WAN 2.2](https://arxiv.org/abs/2503.20314)的3D-VAE encoder（frozen）编码成temporal tokens $V$，分辨率$1960 \times 48$。

为什么用3D-VAE而非per-frame VQ-GAN？Table 12给出了答案：
- VQ-GAN per-frame encoding: 3.53s/step, avg 64.6%
- 3D-VAE joint encoding: 3.28s/step, avg 71.9%

3D-VAE在temporal dimension做4x下采样（17帧→5个temporal slots，因为$4N+1$的格式要求），显式捕获inter-frame motion dynamics，既快又准。

#### 3.1.2 Joint Visuomotor Gating（核心创新）

这是整篇paper的灵魂。把temporal tokens分成两半：980 visual tokens $V_n$ 和 980 motor tokens $M_n$。

**Step 1**: Visual stream refinement
$$V_f = \text{SelfAttn}(V_n)$$
$V_n$通过3层Transformer自注意力得到global visual context $V_f$。

**Step 2**: Motor stream refinement + gated cross-attention
$$M_n' = \text{SelfAttn}(M_n)$$
$$M_n'' = \text{CrossAttn}(M_n', V_f, V_f)$$
$$M_{n+1} = \sigma(r) \odot M_n'' + M_n'$$

逐项解释：
- $M_n$: 输入motor tokens
- $M_n'$: 自注意力精炼后的motor tokens（纯dynamics信息）
- $M_n''$: motor tokens对visual tokens的cross-attention结果（查询spatial constraints）
- $\sigma(\cdot)$: sigmoid函数，把scalar $r$压缩到$(0,1)$
- $r$: 可学习scalar parameter，控制cross-attended embeddings的贡献
- $\odot$: element-wise multiplication
- $M_{n+1}$: 更新后的motor tokens

这个gating机制形式上类似 [Highway Networks](https://arxiv.org/abs/1505.00387) 和GRU的update gate。残差连接保证motor stream不会丢失原始dynamics信号。

为什么这个设计很重要？Table 5的ablation给出了清晰的证据：

| Variant | MT | VT | JVG | Avg |
|---------|----|----|-----|-----|
| (a) baseline | | | | 62.5% |
| (b) MT only | ✓ | | | 58.4% (-4.1%) |
| (c) MT+VT decoupled | ✓ | ✓ | | 65.6% (+7.2%) |
| (d) full | ✓ | ✓ | ✓ | 71.9% (+6.3%) |

关键insight：
- (b)比(a)还差——直接把multi-frame tokens映射到actions，网络被迫吸收noisy visual dynamics
- (c)通过decoupling获得+7.2%提升，证明分离static visual和dynamic motor的重要性
- (d)再加JVG获得+6.3%提升，证明visual conditioning（而非纯decoupling）才是关键

这个gating mechanism迭代3次，让motor stream反复查询visual constraints，最终得到physically grounded joint visuomotor embeddings $M_f$。

#### 3.1.3 Visual Reconstruction
$$\mathcal{L}_I = \|V_r - V_t\|_2$$

- $V_r$: image decoder从$V_f$重建的latent representation
- $V_t$: 第一帧$O_t$经3D-VAE编码的target latent

关键设计：**重建第一帧而非最后一帧**！Table 13证明：
- First-frame reconstruction: 71.9%
- Last-frame reconstruction: 68.8% (-3.1%)

为什么？重建第一帧让visual tokens提供static geometric anchor；重建最后一帧则让visual tokens被迫记忆future visual dynamics，重新引入visually-dominated entanglement。这是一个非常subtle但critical的设计选择。

#### 3.1.4 Action Heads

支持两种action formulation：

**OFT-style** (regression with MAE):
$$\mathcal{L}_A = \|\hat{A}_{t:t+k} - A_{t:t+k}\|_1$$
$A_{t:t+k}$是target action chunk（16步），$\hat{A}$是预测。用2层ResNet MLP。

**GR00T-style** (conditional flow matching, 参考 [Lipman et al., ICLR 2023](https://arxiv.org/abs/2210.02747)):
$$\mathcal{L}_A = \mathbb{E}_{p(X_t|M_f), q(X_t^\tau|X_t)} \|v_\tau^\theta(X_t^\tau, M_f) - u(X_t^\tau|X_t)\|_2$$

- $X_t$: target action chunk $A_{t:t+k}$
- $\tau \in [0,1]$: flow matching time variable
- $X_t^\tau = \tau X_t + (1-\tau)\epsilon$: noisy action，由Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$和target线性插值
- $u(X_t^\tau|X_t) = \epsilon - X_t$: target vector field (denoising direction)
- $v_\tau^\theta$: 神经网络学习的vector field
- $M_f$: joint visuomotor embeddings作为条件

#### 3.1.5 Total Pretraining Loss
$$\mathcal{L}_1 = \lambda \mathcal{L}_I + \mathcal{L}_A$$

Table 11显示$\lambda=1.0$最优（71.9%），过小（0.5, 67.7%）或过大（1.5, 69.8%）都退化。

### Stage 2: Joint Visuomotor Embedding Guided VLA Post-training

这一步把pretrain好的$M_f$作为"semantic anchors"，让下游VLA的intermediate representations对齐到这些anchors。

$$\mathcal{L}_2 = \beta \|M_f - F_a\|_2 + \mathcal{L}_A$$

- $F_a$: VLA intermediate representations $F_r$经lightweight Transformer adapter后的aligned embeddings
- $M_f$: frozen pretrained model提取的future-aware joint visuomotor embeddings
- $\beta$: 平衡系数，用cosine decay schedule逐渐减小

为什么用latent alignment而非weight initialization？参考 [Chain-of-Thought prompting](https://arxiv.org/abs/2201.11903)和 [Spatial Forcing](https://arxiv.org/abs/2510.12276)的思路，latent alignment让VLA网络internalize这些dynamics priors，**推理时不需要multi-frame inputs**——这是个巨大优势。

VLA backbone用 [Qwen3-VL-4B-Instruct](https://arxiv.org/abs/2505.09388)。

## 4. 实验数据深度解读

### 4.1 SimplerEnv Google Robot (Table 1)

Visual Matching setting下：
- OpenVLA-OFT: 47.5% → FutureVLA-OT: 77.6% (**+30.1%**)
- GR00T-N1.5: 35.2% → FutureVLA-GT: 80.1% (**+44.9%**)
- π0: 52.7% → FutureVLA-GT: 80.1% (**+27.4%**)

特别值得注意的是"Put in Drawer"任务（long-horizon）：
- OpenVLA-OFT: 0.9% → FutureVLA-OT: 74.1%
- GR00T-N1.5: 7.4% → FutureVLA-GT: 85.2%

这个+77%的提升说明future guidance对long-horizon planning特别critical。

### 4.2 SimplerEnv WidowX (Table 2)

- UniVLA: 47.9% → FutureVLA-GT: 71.9% (**+24.0%**)
- Villa-X: 40.8% → FutureVLA-GT: 71.9% (**+31.1%**)

这个对比直接打脸implicit methods——同样提供future guidance，decoupling + conditioning的效果显著优于sparse frame pair reconstruction。

### 4.3 LIBERO (Table 3)

- Long suite: OpenVLA-OFT 53.7% → FutureVLA-GT 96.0% (**+42.3%**)
- Avg: UniVLA 95.2% → FutureVLA-GT 98.3% (**+3.1%**)

### 4.4 LIBERO-Plus (Table 6) —— 真正的robustness test

- WorldVLA: 25.0% → FutureVLA-GT: 79.7% (**+54.7%**)
- UniVLA: 43.9% → FutureVLA-GT: 79.7% (**+35.8%**)
- OpenVLA-OFT: 69.6% → FutureVLA-GT: 79.7% (**+10.1%**)

LIBERO-Plus引入7种perturbations：Camera, Robot, Language, Light, Background, Noise, Layout。WorldVLA在Camera perturbation下崩溃到0.1%，因为它依赖visual reconstruction；FutureVLA通过decoupling获得59.7%。

### 4.5 Real-World Franka Robot (Fig 3, Table 10)

四个contact-rich tasks：
- Make Burger, Insert Rose, Scoop Beans, Erase Handwriting
- π0: 43.3% → FutureVLA-GT: 70.0% (**+26.7%**)
- Whiteboard Erasing: 33.3% (no JVPM) → 73.3% (w/ JVPM) (**+40.0%**)

Whiteboard erasing这个任务特别有趣——它需要持续force regulation，纯视觉约束不够，作者在Limitations部分也提到未来需要tactile/force-torque feedback。

### 4.6 Temporal Density (Fig 4)

Frame sampling {2, 5, 9, 17}：
- 2 frames: 最低性能
- 17 frames: 最高性能

这直接验证了"continuous temporal modeling > sparse sampling"的核心claim。Table 15显示2帧时用[1,1,17,17,17]padding以满足$4N+1$的3D-VAE要求。

### 4.7 Temporal Horizon (Fig 5)

{9, 13, 17, 21} frames：
- 9→17: 性能稳步提升
- 17→21: 性能退化

原因：action chunk size固定为16，21帧引入redundant/misaligned context。这给出了visual observation length和motor execution length对齐的原则。

### 4.8 Fair Architecture Comparison (Fig 6)

在unified architecture下对比三种future guidance styles：
- WorldVLA (explicit)
- LAPA-style (implicit)
- Villa-X-style (implicit)
- Ours: 最佳

这说明性能提升来自方法本身，而非backbone优势。

### 4.9 Physics-Aware Action Consistency (Fig 7, 8, Algorithm 1)

这是评估embedding quality的核心metric。用DTW计算action sequence distance：

$$\mathcal{D}_{DTW}(A_i, A_j) = \min_{\pi \in \Pi} \sum_{(k,m) \in \pi} \|A_{i,k} - A_{j,m}\|_2^2$$

- $\Pi$: 所有合法单调对齐路径的集合
- $A_{i,k}, A_{j,m}$: 两个action sequences在第$k$步和第$m$步的action向量
- $\|\cdot\|_2^2$: L2距离平方，sensitive to physical magnitude

然后用RBF kernel转换成similarity：
$$S_{phys}(A_i, A_j) = \exp\left(-\frac{\mathcal{D}_{DTW}(A_i, A_j)}{2\sigma^2}\right)$$

- $\sigma$: bandwidth parameter，用median heuristic设定
- $\sigma = \text{median}(\{\sqrt{\mathcal{D}_{DTW}(A_i, A_j)} | i \neq j\})$

为什么需要PAAC？naive cosine similarity有两个问题：1) 忽略control signal magnitude; 2) 假设strict temporal alignment。DTW解决temporal misalignment，L2距离sensitive to magnitude。

Fig 8的probability density distribution显示FutureVLA的latent similarity和action consistency的相关性最强，证明它真正捕获了motor intent而非visual residual。

### 4.10 Robustness to Visual Perturbations (Table 14)

加入color jittering, Gaussian noise等perturbation：
- LAPA-style: Embedding MSE 0.3047, Action Bias 0.0061
- Villa-X-style: Embedding MSE 0.0188, Action Bias 0.0036
- Ours: Embedding MSE 0.0054, Action Bias 0.0027

FutureVLA的embedding deviation比LAPA小56倍！这直接验证了decoupling + gating机制成功shield了motor tokens from task-irrelevant visual dynamics。

## 5. 与相关工作的对比联想

### 5.1 JEPA家族
- [I-JEPA](https://arxiv.org/abs/2301.08243): image-level predictive architecture
- [V-JEPA](https://arxiv.org/abs/2304.08871): video-level
- [V-JEPA 2](https://arxiv.org/abs/2506.09985): 理解+预测+planning

FutureVLA和JEPA哲学一致：predict in latent space而非pixel space。但FutureVLA在robotics场景下做了关键的**stream decoupling**——这是JEPA没有显式做的。

### 5.2 World Models
[Genie](https://arxiv.org/abs/2402.15391), [DreamerV3](https://arxiv.org/abs/2301.04104), [3D-VLA](https://arxiv.org/abs/2403.09631)都在做world modeling，但都倾向visual reconstruction-heavy。FutureVLA选择更minimal的reconstruction（只重建第一帧latent），把剩余capacity留给motor modeling。

### 5.3 Gating机制族
- [Highway Networks](https://arxiv.org/abs/1505.00387) (Srivastava et al., 2015)
- GRU update gate
- [GLU](https://arxiv.org/abs/1612.08083) (Dauphin et al., 2017)

FutureVLA的$M_{n+1} = \sigma(r) \odot M_n'' + M_n'$是Highway Networks的直接inspiration，但用了scalar $r$而非vector gate，参数量极少但有效。

### 5.4 Flow Matching
[Lipman et al., ICLR 2023](https://arxiv.org/abs/2210.02747)提出flow matching作为diffusion的alternative。[π0](https://arxiv.org/abs/2410.24164)和[GR00T-N1.5](https://arxiv.org/abs/2503.14734)都用flow matching做action generation。FutureVLA的GR00T-style head继承了这套formulation，但conditioning input是joint visuomotor embeddings而非纯visual+language tokens。

### 5.5 Latent Alignment
参考[Chain-of-Thought](https://arxiv.org/abs/2201.11903)和[Spatial Forcing](https://arxiv.org/abs/2510.12276)的latent alignment策略，FutureVLA把这种思想用于future-aware priors transfer。这种策略的好处是**不改inference architecture**——下游VLA推理时只需要单帧observation + language instruction。

## 6. 工程细节

### 6.1 训练配置
- Pretraining: batch size 256, lr $1 \times 10^{-5}$, 5000步linear warmup
- 16×A100 GPU, 3天
- 15.6M frames，混合OXE和LIBERO数据集（Fig 9）
- Action chunk size: 16

### 6.2 Post-training
- Qwen3-VL-4B-Instruct作为VLM backbone
- 300 trajectories for real-world (75/task, 5Hz)

### 6.3 3D-VAE的$4N+1$约束
WAN的3D-VAE在temporal dimension有4x下采样，要求输入帧数$= 4N+1$：
- 17 frames = $4 \times 4 + 1$ ✓
- 21 frames = $4 \times 5 + 1$ ✓
- 2 frames需要padding到5 ($4 \times 1 + 1$)

这解释了为什么temporal density实验选择{2, 5, 9, 17}（都是$4N+1$）。

## 7. 局限与未来方向

作者在A.4部分坦承：
- Contact-rich operations（如whiteboard erasing）纯视觉约束不够
- 未来需要整合tactile / force-torque feedback
- 多感官的joint embedding预测是自然扩展

这个方向和[InternVLA-A1](https://arxiv.org/abs/2601.02456)（统一understanding, generation, action）以及[RoboInter](https://arxiv.org/abs/2602.09973)（holistic intermediate representation）的工作可以形成互补。

## 8. 我的思考

这篇paper最漂亮的地方在于**精准识别问题本质**：
1. Visual和motor在物理上属性不同，必须decouple
2. 但pure decoupling不够，motor必须conditioned on visual
3. Continuous temporal modeling优于sparse sampling
4. Reconstruction target选择first-frame而非last-frame，避免visual entanglement

这些insight都有明确的ablation支撑（Table 5, 13, 14），逻辑闭环。

潜在concern：
- 17帧是optimal，但这个数字依赖action chunk size=16，不同chunk size可能需要重新调
- Pretraining在15.6M frames上3天16×A100，算力门槛中等
- Latent alignment用MSE，可能限制了embedding distribution的多样性
- Real-world只测了4个task，contact-rich表现仍受限于纯视觉

总体而言，这篇paper为VLA的future guidance提供了一个非常clean的解决方案，decoupling + gating的设计哲学值得推广到更多multi-modal predictive modeling场景。

## References

- [FutureVLA Paper](https://arxiv.org/abs/2601.02456) (主要paper)
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)
- [I-JEPA](https://arxiv.org/abs/2301.08243)
- [WorldVLA](https://arxiv.org/abs/2506.21539)
- [DreamVLA](https://arxiv.org/abs/2507.04447)
- [LAPA](https://arxiv.org/abs/2410.11758)
- [UniVLA](https://arxiv.org/abs/2505.06111)
- [Villa-X](https://arxiv.org/abs/2507.23682)
- [π0](https://arxiv.org/abs/2410.24164)
- [GR00T-N1.5](https://arxiv.org/abs/2503.14734)
- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)
- [WAN 2.2](https://arxiv.org/abs/2503.20314)
- [Flow Matching (Lipman et al.)](https://arxiv.org/abs/2210.02747)
- [Highway Networks](https://arxiv.org/abs/1505.00387)
- [LIBERO](https://arxiv.org/abs/2306.03310)
- [SimplerEnv](https://arxiv.org/abs/2405.05941)
- [LIBERO-Plus](https://arxiv.org/abs/2510.13626)
- [Qwen3-VL](https://arxiv.org/abs/2505.09388)
- [Chain-of-Thought](https://arxiv.org/abs/2201.11903)
- [LeCun Path towards AMI](https://openreview.net/forum?id=bZ3aPdQwd6)
- [Genie](https://arxiv.org/abs/2402.15391)
- [3D-VLA](https://arxiv.org/abs/2403.09631)
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864)
- [DTW经典文献(Sakoe-Chiba)](https://ieeexplore.ieee.org/document/1163055)
