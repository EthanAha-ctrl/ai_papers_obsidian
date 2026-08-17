---
source_pdf: T-Rex- Tactile-Reactive Dexterous Manipulation.pdf
paper_sha256: 21fa27294293c84b74bed23d99e68c1c938b061a2743c6fb9d949fbd5908a7f2
processed_at: '2026-08-12T11:55:19-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# T-Rex 用人话说

## 核心矛盾

人能把钥匙插进锁孔，靠的不是眼睛——锁孔那么小，光看根本看不清。靠的是**手感**：手指碰到锁孔边缘的瞬间，你会下意识微调角度、感受阻力、顺着滑进去。这是毫秒级的反射，快到你意识不到。

现在的 robot 大模型（VLA）全是"眼睛+脑子"，每秒钟想几次，想得很深但反应慢。你让它做精细操作，它看到手快碰到锁孔了，等它想完"我该往左偏2毫米"并发指令，手早撞过去了。

**T-Rex 要解决的就是：怎么让大模型学会用"手感"做毫秒级反应。**

---

## 难在哪

**第一个难：没数据。** 训练大模型要几万小时视频，但带触觉的机器人数据，全世界加起来也没多少。人工遥操作采集，又贵又慢，100 小时就要好几周。

**第二个难：频率对不上。** 视觉大模型一次推理要 100ms 以上，触觉反应要 10ms 级别。把触觉塞进大模型一起跑，触觉信息还没来得及处理就过时了。

**第三个难：硬塞触觉会坏事。** 实验里 π0.5（一个很强的 VLA）直接把触觉信号拼到输入里，成功率从 17% 掉到 6%——模型压根没学过这玩意儿，突然多一路输入把原来的表征都搞乱了。

---

## T-Rex 的三个核心 idea

### Idea 1: 把"想"和"摸"拆开

打个比方开车：你不会每 0.1 秒都重新看地图规划路线，但你每 0.1 秒都会根据路面手感微调方向盘。这两件事频率不同、用的"脑子"也不同。

T-Rex 干的就是这个：
- **Action expert（大模型，1.4B）**：每 16 步跑一次，用视觉+语言规划大致动作。这是"看地图"。
- **Tactile expert（小模型，0.6B）**：每 4 步跑一次，只看触觉信号做微调。这是"调方向盘"。

关键在于小模型复用大模型的"思考结果"（cached KV cache），不用重新看图、重新理解指令，只处理触觉信号这一件事，所以跑得快。

---

### Idea 2: 在"雕刻"中间切一刀

Flow matching 的本质是从噪声一步步"雕刻"出动作。T-Rex 在 τ=0.4 这个点切一刀：

- **前 6 步**（从噪声到"粗坯"）：大模型用视觉和语言把动作的大致形状凿出来。这时候不需要触觉，因为手还没碰到东西。
- **后 4 步**（从"粗坯"到"成品"）：小模型用触觉做最后的打磨。这时候视觉已经没用了（手已经握住物体，看不见接触面），触觉才是关键。

为什么是 0.4？论文里做了 ablation：太小了"粗坯"没成形没法打磨，太大了小模型改不动大模型的"成品"。0.4 是个甜点。

这个设计的妙处在于：**触觉只在最需要的时候介入，不破坏视觉规划的全局结构，只做局部修正。**

---

### Idea 3: 用"菜谱组合"而不是"背特定菜"

数据采集的常规做法：给每个任务录 100 次演示。12 个任务就是 1200 次演示，很窄，泛化差。

T-Rex 的做法：**学 22 个"动词"（pick、slide、press、twist...）× 207 个"名词"（cup、card、egg...）**，只保留物理上可行的 502 种组合，每种录 16 次。

好处巨大：
- 学了"pick + cup"之后，"pick + bottle"也能部分迁移
- 100 小时数据覆盖了整个"动作×物体"空间，比 100 小时只录 11 个特定任务泛化强得多
- 论文 Fig. 6 直接证明：同样的 100 小时预算，compositional 设计的 mid-training 数据让模型在没见过的任务上 zero-shot 就能做

---

## 触觉怎么编码：把"杂音麦克风"变成"拼音"

触觉传感器有两个烦人的问题：
1. **Drift**：用着用着 calibration 就跑了，同样的力读出不同的数
2. **Noise**：高频噪声很大

直接把原始数值喂给模型，模型会 overfit 到这些噪声上。

T-Rex 用 VQ-VAE 把每个指尖的 16 帧 force 历史压缩成 64 个"离散 token"中的一个。就像把声波转成拼音——不存具体波形，只存"这个音节大概是什么类"。

有个关键 trick：**loss 对大力帧加权更高**。因为大部分时间手指没接触东西（force=0），如果平等加权，codebook 会全部坍塌到"没接触"这一类。给接触帧更高权重，强迫 codebook 学会区分"碰到了"和"没碰到"的各种状态。

deformation map（指尖皮肤的形变图）走另一条路：ResNet-18 前三层 + 自监督 pretrain 后 frozen。这部分只管空间几何（边缘、滑移、剪切），跟 force 的时间动态互补。

---

## 三阶段训练：看视频 → 戴手套练 → 精修

| 阶段 | 数据 | 干什么 |
|---|---|---|
| Pre-training | 22,889 小时 egocentric 视频 | 学"大概怎么做菜"——视觉先验和大致动作模式 |
| Mid-training | 100 小时带触觉遥操作 | 学"手该怎么动"——把视觉先验对齐到真实接触动力学 |
| Post-training | ~100 次特定任务演示 | 学"这道菜的具体细节" |

Table 3 的 ablation 最有说服力：
- 两个 stage 都不要：18%
- 只 mid-training：34%
- 只 pre-training：45%
- 两个都要：**65%**

pre-training 贡献最大（+27%），mid-training 在它基础上再 +20%。两个 stage 干的事不一样：pre-training 给语义和粗动作，mid-training 给接触细节。

---

## 实验里最有意思的发现

### 发现 1: 小模型从零训根本不行

ViTacFormer、RDP 这些从零训的方法，12 个任务平均 3-6%。100 次演示完全不够学出泛化能力。**大规模 pre-training 是刚需，不是锦上添花。**

### 发现 2: 硬塞触觉会坏事

π0.5 + tactile 从 17% 掉到 6%。这个结果反直觉但重要：**你不能给一个从没摸过东西的模型突然塞触觉输入，它会崩溃。** 触觉集成必须从架构层面设计，包括训练 curriculum、频率解耦、表征对齐。T-Rex 的 mid-training 阶段就是让模型"学会用触觉"的关键。

### 发现 3: 触觉对哪些任务帮助最大

看 task-by-task 数据：
- **Flip Page**（单指翻页，需要滑动控制）：T-Rex 96% vs EgoScale 68% → +28%
- **Open Lock**（钥匙插孔，需要触觉引导插入）：47% vs 19% → +28%
- **Transfer Egg**（拿鸡蛋不捏碎）：75% vs 44% → +31%

这些任务的共同点：**视觉看不到的关键信息藏在接触面里**。看再多遍也看不出鸡蛋该用多大力，必须摸。

### 发现 4: 最难的任务仍然只有 35%

Screw Lightbulb（双手拧灯泡）T-Rex 也只有 35%。这任务太难：双手协调 + 多圈螺纹对齐 + 力度调节 + 长时序。说明触觉 reactive 这条路还很长。

---

## 失败案例很有信息量

论文 App. H 诚实列了 6 类失败：

1. **拧灯泡撞到灯座**：视觉对齐还是不够细，动作太快撞了
2. **钥匙掉了**：小物体抓取不稳，tactile feedback 没完全解决 in-hand manipulation
3. **鸡蛋放不进槽**：BC 的分布漂移老问题，抓取成功但放置位置不准
4. **麻将牌开错盒子**：拇指位置太低意外碰到中间盒子，多指协调不够
5. **牙膏挤太多**：sequential prediction 导致力度过冲
6. **抽卡卡住**：sliding 动作需要更强的时间维度触觉 conditioning

这些失败揭示了 T-Rex 的边界：它解决了"反应式触觉"，但精细的 multi-finger coordination、长时序 force regulation、小物体 in-hand manipulation 还差得远。

---

## 一句话总结

**T-Rex 把触觉作为一等公民集成进 VLA foundation model，核心 trick 是用 cascaded flow matching 把"慢思考的视觉规划"和"快反射的触觉微调"在时间维度上切开，再用异步 KV cache 让小触觉模型复用大视觉模型的思考结果——实现了 30% 的平均成功率提升，证明了 tactile-reactive 是 dexterous manipulation 的必备能力。**

剩下的限制主要在硬件（sensor drift、缺乏 palm sensing）和长时序任务上。下一步可能是 RL post-training、whole-hand tactile、cross-sensor 统一表征。

---

# T-Rex: Tactile-Reactive Dexterous Manipulation 深度技术讲解

## 1. 高层直觉 (High-Level Intuition)

这篇paper要解决的核心矛盾非常清晰：**人类的灵巧操作依赖触觉的高频闭环反馈**（如slide card into slot、open lock with key），但当前的VLA (Vision-Language-Action) models主要在低频的视觉认知层面运作。Tactile信号天然是高频的（微slip、force variation、local deformation需要毫秒级响应），而VLM backbone的inference cost决定了它无法跑那么快。

T-Rex的解决方案可以归纳为三个核心idea：

**(1) 解耦频率的MoT架构**：用一个low-frequency的action expert处理视觉规划，用一个high-frequency的tactile expert处理触觉refinement。

**(2) Cascaded Flow Matching**：将denoising trajectory在τ_split=0.4处切开，slow stream跑完整的视觉planning，fast stream复用cached KV，只在tactile expert里完成剩余的4步denoising。

**(3) 三阶段训练recipe**：先用22,889小时egocentric human video做pre-training获得visuomotor priors，再用100小时tactile-synchronized teleop data做mid-training桥接contact dynamics，最后用~100 demonstrations做skill-specific post-training。

项目主页：https://tactile-rex.github.io/

---

## 2. Architecture深度解析

### 2.1 Mixture-of-Transformer-Experts (MoT) Backbone

T-Rex用了三个specialized experts：

**Latent Expert**（1.41B参数）：
- Backbone: Qwen3VL-2B
- Hidden dim: 2048, 28 layers
- 功能：处理visual和language observations，预测future visual representations
- 提供temporally grounded context

**Action Expert**（1.41B参数）：
- 同样的Qwen3VL-2B backbone
- Action dimension: 62（双臂14维 + 双手22×2=44维 + ...）
- Action chunk: 16
- 功能：low-frequency planning，从noise denoise到τ_split

**Tactile Expert**（0.62B参数，小很多！）：
- FFN intermediate size: 1536（比标准更小）
- 只需4个Euler steps（vs action expert的6步）
- 功能：high-frequency tactile refinement

**关键insight**：tactile expert参数量只有action expert的~44%，因为它不需要处理高维的visual-language context，只关注局部的tactile信号和cached的KV。这是计算amortization的核心。

### 2.2 Spatial-Temporal Tactile Encoder

触觉编码分两部分，公式(2)给出了完整的token构造：

$$z_t^\tau = \left[\text{Emb}_{\text{vq}}\big(E_f(\mathbf{f}_{t-15:t})\big) ; \text{Proj}_f(\mathbf{f}_t) ; \text{Proj}_d\big(E_d(\mathbf{d}_t)\big)\right]$$

变量解释：
- $\mathbf{f}_{t-15:t}$: 过去16帧的force history（6D per finger，5 fingers）
- $E_f$: 1D temporal CNN encoder
- $\text{Emb}_{\text{vq}}$: Vector Quantization embedding（codebook size K=64）
- $\mathbf{f}_t$: 当前时刻的瞬时force vector（保留高精度）
- $\mathbf{d}_t$: 当前deformation map（single-channel spatial）
- $E_d$: ResNet-18的前3个stage

**VQ-VAE Force Encoder细节**：
- 输入：T=16帧 × 6D force vector
- 网络结构：1D temporal conv with two strided blocks → temporal mean-pooling → 256D continuous embedding
- Vector quantization到K=64的codebook
- Codebook用EMA更新，underutilized entries周期性reseed（防止codebook collapse）
- Loss：**magnitude-weighted MSE**，对high-force contact frame给予更高penalty（防止codebook坍塌到dominant non-contact states）
- Convolutional weights在5个finger间共享，加上learned finger-identity embedding

**为什么用VQ-VAE？** tactile sensor有显著的drift问题，连续值表示容易overfit到sensor noise。Discrete codebook强制学习compact、drift-robust的"tactile vocabulary"。这类似于LLM中tokenization的作用——把高维连续信号压缩成语义token。

**Deformation Encoder**：
- ResNet-18 backbone，modified input stem（single-channel input）
- 只保留前3个residual stages
- 每个stage后接3×3 conv，re-project到128 channels
- 在self-supervised convolutional autoencoder框架下pre-trained，policy training时frozen

**设计哲学**：force是temporal dynamics（需要时序建模），deformation是spatial pattern（需要空间建模）。两个信号互补，分开编码后再concatenate。

### 2.3 Asynchronous Tactile-Reactive Cascaded Flow Matching

这是这篇paper最核心的技术贡献。

**Flow Matching基础**：标准flow matching的loss（公式1）：

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}\left[\| v_\theta(x_\tau, \tau | c_t) - (x_1 - x_0) \|^2\right]$$

其中：
- $x_0 = A_{t:t+H}$: clean action chunk（ground truth demonstration）
- $x_1 = \epsilon \sim \mathcal{N}(0, I)$: Gaussian noise
- $\tau \in [0, 1]$: flow time parameter
- $v_\theta$: 学习的vector field
- $c_t = \{o_t, \ell, f_{t-H_f:t}, d_t\}$: multimodal context

**Linear Interpolant**（公式3）：

$$x_\tau = (1-\tau)A^{\text{demo}} + \tau \epsilon, \quad v^* = \epsilon - A^{\text{demo}}$$

- $x_\tau$: 在demo action和noise之间的线性插值
- $v^*$: constant velocity target（从noise指向demo的方向）
- 两个experts都回归同一个$v^*$，但在disjoint的sub-intervals上

**Cascaded Denoising的核心**：把trajectory在$\tau_{\text{split}} = 0.4$切开：

**Slow Stream**（公式4）：
$$\hat{x}_{\tau_{\text{split}}} = \text{Euler}\big(f_\theta^{\text{act}}; x_1, \tau; 1 \to 0.4, K_{\text{slow}} = 6\big)$$

- 从$x_1 = \epsilon$（pure noise）开始
- 6步Euler integration，$\Delta\tau = -0.1$
- 到达$\hat{x}_{0.4}$时停止
- 这时cache KV作为stationary visual context

**Fast Stream**（公式5）：
$$\mathbf{A}_{t:t+T_a} = \text{Euler}(f_\theta^{\text{tac}}; \hat{x}_{\tau_{\text{split}}}, \tau; 0.4 \to 0, K_{\text{fast}} = 4)$$

- 从cached的$\hat{x}_{0.4}$开始
- 4步Euler integration
- Tactile expert接收real-time tactile tokens $z_t^\tau$和cached KV
- 对action chunk长度$T_a = 16$，fast stream在offsets {0, 4, 8, 12}处触发

**关键insight**：Fast stream完全bypasses heavy visual network！它复用cached KV，只跑轻量级的tactile expert。这就是"asynchronous"的来源——slow stream每16个control step跑一次，fast stream每4个control step就触发一次。

### 2.4 Training Objectives

公式(6)和(8)(9)：

$$\mathcal{L}_{\text{act}} = \big\| f_\theta^{\text{act}}\big(x_{\tau_{\text{act}}}, \tau_{\text{act}}; c^{\text{vl}}\big) - v^* \big\|_2^2$$

$$\mathcal{L}_{\text{tac}} = \big\| f_\theta^{\text{tac}}(x_{\tau_{\text{tac}}}, \tau_{\text{tac}}; c^{\text{tac}}, \text{KV}_{\tau_{\text{split}}}) - v^* \big\|_2^2$$

**重要细节**：
- $c^{\text{vl}}$ = visual-language context（head/wrist cameras + language prompts + future prediction tokens）
- $c^{\text{tac}}$ = 高频tactile tokens
- $\text{KV}_{\tau_{\text{split}}}$ = 从detached slow stream pass中extract的cache

**Training timestep sampling**：
- $\tau_{\text{act}} \sim \text{Beta}(1.5, 1.0)$ on $(0, 1]$
- $\tau_{\text{tac}} = \tau_{\text{split}} \cdot \tilde{\tau}$ where $\tilde{\tau} \sim \text{Beta}(1.5, 1.0)$ on $(0, \tau_{\text{split}}]$

Beta(1.5, 1.0)分布偏向于较大的τ值，意味着训练时更多采样靠近noise的state——这模仿了diffusion training中常见的做法，在高噪声区域给予更多训练信号。

**Delay Augmentation**：
$$\delta \sim \text{Uniform}\{0, 4, 8, 12\}$$

这个augmentation非常subtle但关键：deployment时fast ticks在intra-chunk offsets异步运行，visual cache和real-time tactile stream之间有temporal staleness。训练时随机shift frame indices匹配这种staleness distribution，防止overfit到perfectly synchronized modalities。

**Total Loss**（公式7/10）：
$$\mathcal{L} = \mathcal{L}_{\text{act}} + \lambda_{\text{tac}} \mathcal{L}_{\text{tac}} + \lambda_{\text{future}} \mathcal{L}_{\text{future}}$$

where $\lambda_{\text{tac}} = 1.0, \lambda_{\text{future}} = 0.5$

Future prediction loss作为auxiliary objective，align tactile modality with broader visual context during mid-training。

### 2.5 KV Cache Composition

$$\text{KV}_{\tau_{\text{split}}} = [\text{KV}^{\text{lat}} | \text{KV}_{\tau_{\text{split}}}^{\text{act}}]$$

- $\text{KV}^{\text{lat}}$: visual-language keys/values（latent expert的输出）
- $\text{KV}_{\tau_{\text{split}}}^{\text{act}}$: action positions在τ_split时刻re-encoding的keys/values

**为什么需要re-encode**？确保tactile expert attend到一个coherent、partially-denoised contextual manifold，而不是initial noise-time encoding。这是cascade的信息流动关键。

### 2.6 Algorithm 1 Pseudo-code解析

慢流和快流并行运行，通过execution lock实现thread safety：

**SLOW-STREAM** (line 2-16):
- 每个action chunk window触发一次
- 从$x_1 \sim \mathcal{N}(0, I)$开始
- 跑6步Euler integration（$\tau: 1 \to 0.4$）
- acquire lock → 保存$\hat{x}_{\tau_{\text{split}}}$和refreshed KV → release lock

**FAST-STREAM** (line 17-32):
- 对offsets $\delta \in \{0, 4, 8, 12\}$分别触发
- 采样real-time tactile stream $c^{\text{tac}}$
- acquire lock → clone KV → release lock
- 从$\hat{x}_{\tau_{\text{split}}}$开始跑4步Euler integration（$\tau: 0.4 \to 0$）
- 输出$\hat{A}_{t+\delta:t+\delta+T_a}$

**Computational Amortization**：每个control step的cost只由$K_{\text{fast}} = 4$步Euler的tactile expert决定（轻量级，FFN size 1536），visual tower + latent expert + action expert都不重新执行。

---

## 3. T-Rex Dataset 设计哲学

### 3.1 规模与构成

- **100小时** bimanual dexterous manipulation
- **200+** everyday objects
- **22** motor primitives
- **502** unique object-motor primitive combinations
- **7,755** episodes
- Median episode length: 29.8s (IQR: 21.0-41.1s)
- 平均每个combination ~16 demonstrations

### 3.2 Verb-Noun Compositional Design

关键insight：不收集narrow task-specific demos，而是设计verb-noun组合。207 objects × 22 primitives = 4,554 candidate combinations，pruning掉infeasible pairs（如pour a solid block），得到502个feasible combinations。

这种compositional design的好处：
1. **Coverage efficiency**: 16 demonstrations per pair就能覆盖action distribution
2. **Generalization**: 模型学到"verb"和"noun"的compositional structure
3. **Zero-shot capability**: mid-training后已经能在Fig. 6展示zero-shot contact-rich manipulation

### 3.3 22 Motor Primitives

虽然paper没有完整列出所有22个，但从Fig. 2的top-right分布和tasks描述可以推断包括：pick, slide, press, wipe, twist, pour, squeeze, insert, rotate, flip, transfer, rub, screw, etc.

### 3.4 Sensor Modalities

每个episode记录：
- 3× RGB streams（1 head ZED X Mini + 2 wrist ZED X One S）at 640×360, 30Hz
- Bimanual proprioception: 2×7 arm joints + 2×22-DoF Sharpa Wave hand joints
- SE(3) end-effector poses
- Per-fingertip tactile: 5 fingers × 2 hands = 10个sensors，每个提供single-channel deformation depth map + 6-axis net wrench
- Natural language instruction

### 3.5 Scene Diversity

- 6种tabletop backdrops
- 210+ distractor objects pool，每scene 0-5个distractors
- 随机initial object position和orientation
- 鼓励policy根据task context和language识别正确物体

### 3.6 VLM-based Language Annotation

使用commercial VLM自动标注：
- 输入：4-6 frames from head camera + minimal labels (object name + primitive name)
- 输出：单句imperative sentence描述episode
- 人工验证filter掉hallucinations

---

## 4. 三阶段Training Recipe

### 4.1 Stage 1: Large-scale Human Egocentric Pre-training

- **数据量**: 22,889 hours egocentric human video（基于EgoScale [1]）
- **训练内容**: latent expert学visual/language representations，action expert学retargeted human arm/hand motions
- **关键**: 这一阶段没有tactile expert参与
- **目的**: 提供broad semantic grounding和visuomotor priors

EgoScale paper: https://arxiv.org/abs/2602.16710

### 4.2 Stage 2: Tactile-Grounded Robot Mid-training

- **数据量**: 100 hours tactile-synchronized teleoperation
- **训练内容**: 
  - Action expert适应robot multiview observations和executable actions
  - Tactile expert训练high-frequency denoising作为fine-grained refinement
- **关键**: 这一阶段tactile expert开始参与，但只在$(0, \tau_{\text{split}}]$区间训练

### 4.3 Stage 3: Skill-Specific Post-training

- **数据量**: ~100 demonstrations per task
- **效果**: Fig. 5显示mid-training让low-data regime性能大幅提升，减少downstream data需求

**Table 3 ablation验证三个阶段**：
| Pre-training | Mid-training | Avg Success Rate |
|---|---|---|
| ✗ | ✗ | 18% |
| ✗ | ✓ | 34% |
| ✓ | ✗ | 45% |
| ✓ | ✓ | **65%** |

两个stage都有显著贡献，full recipe最佳。

---

## 5. 实验结果深度分析

### 5.1 Main Results (Table 1)

12个tactile-reactive tasks的完整对比：

| Method | Avg Success Rate | 关键观察 |
|---|---|---|
| ViTacFormer [18] | 3% | Small policy from scratch，完全fail |
| RDP [8] | 6% | Slow-fast diffusion，但foundation弱 |
| Tactile-VLA [21] | 15% | 有tactile但pre-training弱 |
| π0.5 [67] | 17% | 强VLA但无tactile |
| **π0.5 + tactile** | **6%** | Naive加tactile反而下降！|
| EgoScale [1] | 35% | 强pre-training但无tactile |
| **T-Rex (Ours)** | **65%** | +30% over best baseline |

**两个key observations**:

1. **Large-scale pre-training is essential**: ViTacFormer、RDP这种small policies trained from scratch在所有task上都fail。EgoScale因large-scale egocentric pre-training + hand-pose supervision大幅领先π0.5和Tactile-VLA。

2. **Tactile feedback is critical for contact-rich manipulation**: EgoScale虽然在pre-trained VLA中最好，但仍然fail at precise contact adjustment和force-sensitive behaviors。

**最striking的发现**：π0.5 + tactile（naive conditioning）比π0.5还差！从17%降到6%。这说明**tactile integration需要架构设计**，不能naive concatenate到state input。

### 5.2 Task-by-Task分析

观察各个task上的性能差异：
- **Flip Page**: T-Rex 96% vs EgoScale 68% — 需要精细的finger sliding control
- **Transfer Egg**: T-Rex 75% vs EgoScale 44% — 需要force control防碎
- **Apply Paste**: T-Rex 66% vs EgoScale 38% — 需要force regulation
- **Open Lock**: T-Rex 47% vs EgoScale 19% — 需要tactile-guided insertion
- **Screw Bulb**: T-Rex 35% vs EgoScale 18% — 需要bimanual coordination + tactile feedback

这些task都是vision无法单独解决的——必须通过touch感知contact state。

### 5.3 Ablation Studies (Table 2)

**Tactile Modality Ablation**:
| Configuration | Avg | Δ |
|---|---|---|
| Full Model | 65% | baseline |
| w/o Tactile | 42% | -23% |
| MLP Force + Deform | 58% | -7% |
| Deform only | 54% | -11% |
| MLP Force + VQVAE Force | 59% | -6% |

**分析**：
- 完全去掉tactile掉23%，说明tactile信号非常关键
- VQ-VAE force encoder比MLP force encoder好6%（65% vs 59%），验证了discrete codebook的优势
- Deformation单独用比force单独用差，但combined最好——两个modality互补

**Architecture Ablation**:
| Configuration | Avg | Δ |
|---|---|---|
| Full Model | 65% | baseline |
| w/o Async | 60% | -5% |

异步设计贡献5%，说明decoupling low-freq planning和high-freq tactile control确实有好处。

### 5.4 Split Step Ablation (Fig. 4)

变化$\tau_{\text{split}}$：
- $\tau_{\text{split}}$太小：action expert提供insufficient visuomotor priors
- $\tau_{\text{split}}$太大：tactile expert limited capacity incorporate tactile feedback
- 中间值（0.4附近）最优

这揭示了一个interesting trade-off：action expert和tactile expert的"工作量"分配。

### 5.5 Data Efficiency (Fig. 5)

- Blue (with mid-training): 10 demos时已经~45%，100 demos达到65%
- Green (without mid-training): 10 demos时~15%，100 demos只到~40%

Mid-training在low-data regime带来巨大提升，验证了compositional motor primitives data的高效性。

### 5.6 Mid-training Dataset Ablation (Fig. 6)

对比100小时T-Rex Dataset vs 100小时task-specific dataset（matched data budget）：
- T-Rex Dataset在6个representative tasks上更强
- 4个easier tasks上展示zero-shot transfer能力（pick, slide, press, wipe）

**Insight**: Compositional verb-noun design比task-specific data collection更efficient，提供broader generalization。

---

## 6. 12个Evaluation Tasks详解

这些tasks涵盖多个category：

**Force-Reactive Tasks**（需要precise force regulation）：
- Task II: Transfer Egg — 碎壳问题
- Task IV: Apply Toothpaste — 挤出力度
- Task III: Wipe Plate — 接触力调节

**Tactile-Deformation Sensitive**（依赖deformation sensing）：
- Task V: Split Cup — 嵌套杯的twist和rub
- Task VI: Sort Mahjong — 通过surface texture识别pattern
- Task X: Extract Card — 卡套中的sliding

**Insertion/Extraction**：
- Task VII: Open Lock — key insertion
- Task VIII: Refill Tablet — compartment button
- Task XII: Screw Bulb — thread engagement

**Bimanual Coordination**：
- Task I: Flip Page — 单指sweep
- Task XI: Deal Poker — handover + card sliding
- Task IX: Acid-Base Neutralization — dropper + beaker manipulation

**Grading rubric**: 每个task用additive rubric（partial credit per sub-step）或progress-based rubric（single score reflecting hierarchy progress）。

---

## 7. Hardware Setup

### 7.1 Robot Platform

- **Dexmate Vega-1**: bimanual robot, 7 joints per arm
- **Sharpa Wave dexterous hands**: 22-DoF per hand
- **Cameras**: 
  - 1× ZED X Mini (head, 640×360 RGB)
  - 2× ZED X One S wide-view (wrist-mounted)
- **Tactile sensors**: 5 fingertips per hand, single-channel deformation + 6-axis wrench
- **Control**: 300 Hz low-level controller, 30 Hz high-level policy

### 7.2 Teleoperation Stack

- **Manus gloves**: 获取finger target positions
- **VIVE trackers**: 获取wrist SE(3) poses
- **IK**: Pink [71] (differential IK based on Pinocchio [72])
- **Retargeting**: manufacturer-provided differential IK based on Pinocchio + CasADi [73]

---

## 8. Failure Case Analysis (App. H)

6类典型failure：

1. **Object Collision** (screw lightbulb): bulb碰撞base，缺乏fine-grained visual alignment
2. **Slipping Off** (open lock): key grip不稳，small object grasping能力不足
3. **Imprecise Position** (transfer egg): BC的distribution shift问题
4. **Multi-finger friction** (sort mahjong): thumb位置太低，意外打开两个compartments
5. **Excessive Force** (apply toothpaste): sequential prediction mechanism导致force过大
6. **Sliding Misalignment** (extract card): 需要更强temporal tactile conditioning

这些failures揭示了T-Rex的limitation：fine-grained visual alignment、small object dexterity、BC distribution shift、multi-finger coordination、force regulation、temporal tactile conditioning都有改进空间。

---

## 9. 相关工作与Context

### 9.1 VLA Models Evolution

- **RT-2** [30]: 早期VLA，直接fine-tune VLM
- **OpenVLA** [31]: open-source VLA
- **π0** [5]: flow matching VLA with action expert
- **π0.5** [67]: open-world generalization VLA
- **GR00T N1** [6]: NVIDIA的humanoid foundation model
- **Fast-in-Slow** [7]: dual-system foundation model
- **EgoScale** [1]: large-scale egocentric pre-training scaling laws

### 9.2 Tactile Sensing in Manipulation

- **Early work**: shallow MLPs [15]
- **Structured tactile modeling**: rigid-body-pose-aware encodings [16, 17]
- **Joint prediction**: future visual/tactile observations [18, 19]
- **VLA + touch**: tactile as additional modality [21, 22, 23, 24]
- **Force-aware MoE**: ForceVLA [26]
- **Reactive Diffusion Policy** [8]: slow-fast visual-tactile diffusion

### 9.3 Egocentric Human Video Pre-training

- **Ego4D** [43]: 3,000 hours egocentric video
- **EgoDex** [42]: dexterous manipulation from egocentric video
- **MimicPlay** [54]: long-horizon imitation from human play
- **EgoMimic** [3]: scaling imitation via egocentric video
- **DexWild** [4]: in-the-wild dexterous policies
- **EgoVLA** [2]: VLA from egocentric human videos

### 9.4 World Models and Future Prediction

- **Video Prediction Policy** [32]: predictive visual representations
- **F1** [33]: vision-language-action bridging
- **GR-2** [34]: generative video-language-action
- **InternVLA-A1** [35]: unified understanding/generation/action
- **MOTUS** [36]: unified latent action world model
- **DreamDojo** [50]: generalist robot world model from human videos
- **World Action Models** [51]: zero-shot policies

---

## 10. 核心Intuition总结

### 10.1 为什么Cascaded Flow Matching有效？

想象diffusion/flow matching是从noise到action的"雕刻"过程：
- **前6步（slow stream）**: 从pure noise到"rough shape" — 这需要global context（visual + language），所以用heavy action expert
- **后4步（fast stream）**: 从"rough shape"到"final polish" — 这需要local refinement（tactile），所以用lightweight tactile expert

τ_split=0.4不是任意的——它对应着"global structure已经成型，但local details还需要refine"的critical point。

### 10.2 为什么VQ-VAE编码force？

Tactile sensor的signal有两大问题：
1. **Drift**: sensor calibration随时间漂移
2. **Noise**: high-frequency noise难以直接建模

VQ-VAE通过discrete codebook强制学习"semantic tactile vocabulary"。Codebook size K=64足够表达常见的force patterns（contact onset、slip、stable grasp等），又足够compact防止overfitting到noise。

**Magnitude-weighted MSE**是关键trick：对high-force frame给予更高loss weight，防止codebook被dominant的non-contact state占据。

### 10.3 为什么π0.5 + tactile会变差？

Naive的tactile integration有几个问题：
1. **Distribution shift**: pre-training时没见过tactile signal，fine-tune时突然引入new modality破坏了原有的representation
2. **Frequency mismatch**: π0.5的backbone低频运行，tactile signal的高频信息被downsample丢失
3. **Curriculum缺失**: 没有mid-training阶段让模型学习tactile和visual的alignment

T-Rex通过专门的mid-training和cascaded架构解决了这些问题。

### 10.4 为什么Compositional Data高效？

Verb-noun compositionality让模型学到"transferable primitives"：
- 学了"pick + cup"后，"pick + bottle"也能部分transfer
- 学了"slide + card"后，"slide + paper"也能部分transfer

相比之下，task-specific data只覆盖narrow distribution，无法compositional generalization。

### 10.5 为什么Asynchronous比Synchronous好？

Synchronous设计要求每个control step都跑完整的visual + tactile pipeline，这导致：
1. **Latency**: visual backbone的延迟拖累整个loop
2. **Information redundancy**: visual context在short time window内变化很小

Asynchronous设计允许：
- Visual context以低频更新（每16步一次）
- Tactile refinement以高频更新（每4步一次）
- 两者通过cached KV解耦，互不干扰

---

## 11. Limitations & Future Directions

### 11.1 当前Limitations

1. **Long-horizon tasks**: teleoperation困难时，可能需要RL或online interaction-based refinement
2. **Hardware bottlenecks**: 
   - Sensor distortion和calibration drift
   - 缺少dense palm sensing for whole-hand manipulation
   - Heterogeneous tactile sensors之间缺乏unified representation

### 11.2 Future Research Opportunities

1. **RL post-training**: 在tactile-reactive foundation上做RL fine-tuning
2. **Cross-sensor generalization**: unified tactile representation across heterogeneous sensors
3. **Whole-hand tactile**: 扩展到palm和whole-hand contact
4. **Tighter tolerances**: contact coordination for precision assembly

---

## 12. 技术Specs汇总

### 12.1 Model Parameters

| Component | Size | Purpose |
|---|---|---|
| Latent Expert | 1.41B | Future visual prediction |
| Action Expert | 1.41B | Low-freq action denoising |
| Tactile Expert | 0.62B | High-freq tactile refinement |
| **Total** | ~3.44B | |

### 12.2 Training Hyperparameters

- Optimizer: AdamW
- Peak LR: $1 \times 10^{-4}$
- LR Scheduler: Cosine with min LR
- Weight Decay: 0
- Warmup Ratio: 0
- Gradient Clipping: 1.0
- GPU: 24× NVIDIA H100
- Deepspeed Zero Stage 1
- Per Device Batch Size: 16
- Mixed Precision: bf16

### 12.3 Inference Parameters

- Total flow steps: N=10
- Slow segment steps: $K_{\text{slow}} = 6$
- Fast segment steps: $K_{\text{fast}} = 4$
- Split point: $\tau_{\text{split}} = 0.4$
- Step size: $\Delta\tau = -0.1$
- Action chunk: $T_a = 16$
- Fast stream offsets: {0, 4, 8, 12}

### 12.4 VQ-VAE Specs

- Input: T=16 frames × 6D force per finger
- Codebook size: K=64
- Embedding dim: 256
- Codebook update: EMA
- Loss: magnitude-weighted MSE
- Conv weights shared across 5 fingers
- Distinct learned finger-identity embeddings

---

## 13. Reference Links

### 主要References

1. **EgoScale**: https://arxiv.org/abs/2602.16710
2. **EgoVLA**: https://arxiv.org/abs/2507.12440
3. **EgoMimic**: https://arxiv.org/abs/2410.24221
4. **π0**: https://arxiv.org/abs/2410.24164
5. **GR00T N1**: https://arxiv.org/abs/2503.14734
6. **Fast-in-Slow**: https://arxiv.org/abs/2506.01953
7. **Reactive Diffusion Policy**: RSS 2025
8. **ViTacFormer**: https://arxiv.org/abs/2506.15953
9. **Tactile-VLA**: https://arxiv.org/abs/2507.09160
10. **π0.5**: https://arxiv.org/abs/2504.16054
11. **VQ-VAE**: https://arxiv.org/abs/1711.00937
12. **ResNet**: https://arxiv.org/abs/1512.03385
13. **OpenVLA**: https://arxiv.org/abs/2406.09246
14. **Ego4D**: CVPR 2022
15. **MimicPlay**: https://arxiv.org/abs/2302.12422
16. **DexWild**: RSS 2025
17. **Open X-Embodiment**: https://arxiv.org/abs/2310.08864
18. **DROID**: https://arxiv.org/abs/2403.12945
19. **DreamDojo**: https://arxiv.org/abs/2602.06949
20. **World Action Models**: https://arxiv.org/abs/2602.15922
21. **ForceVLA**: NeurIPS 2025
22. **VLA-Touch**: https://arxiv.org/abs/2507.17294
23. **OmniVTLA**: https://arxiv.org/abs/2508.08706
24. **FLARE**: https://arxiv.org/abs/2505.15659
25. **Pretraining Auto-regressive Robotic Models**: https://openreview.net/forum?id=2FDsh5D2Th

### 相关Web Links

- **T-Rex Project Page**: https://tactile-rex.github.io/
- **EgoScale**: https://arxiv.org/abs/2602.16710
- **GR00T N1**: https://arxiv.org/abs/2503.14734
- **OpenPI (π0.5)**: https://github.com/openpi
- **Pinocchio**: https://github.com/stack-of-tasks/pinocchio
- **Pink IK**: https://github.com/stephane-caron/pink
- **Manus gloves**: https://www.manus-meta.com/
- **ZED cameras**: https://www.stereolabs.com/
- **CasADi**: https://web.casadi.org/

---

## 14. 对未来的思考

### 14.1 Scaling Tactile Data

T-Rex用了100小时tactile data，相比22,889小时egocentric video还是小数量级。未来的scaling可能需要：
- 自动化tactile data collection（减少teleop cost）
- Sim-to-real transfer for tactile（tactile simulation成熟度）
- Cross-embodiment tactile transfer

### 14.2 Tactile Foundation Models

类似vision的CLIP、language的GPT，tactile modality可能需要自己的foundation model。T-Rex的VQ-VAE encoder是一个small step，但真正的tactile foundation model需要：
- Cross-sensor unified representation
- Multi-modal alignment（vision-touch-language）
- Billion-scale tactile pre-training

### 14.3 Dual-System Cognitive Architecture

T-Rex的MoT架构让人联想到human cognition的System 1/System 2：
- **System 1 (fast, reactive)**: tactile expert，类似人类的reflexive grasp adjustment
- **System 2 (slow, deliberative)**: action + latent experts，类似人类的conscious planning

Fast-in-Slow [7] 和 MoTVLA [38] 都在探索类似方向。T-Rex的独特之处是把这种dual-system思想应用到tactile modality。

### 14.4 World Models Integration

Latent expert预测future visual representations，这是一个mini world model。未来可以：
- 扩展到tactile future prediction
- 学习contact dynamics的因果模型
- 支持model-based RL和planning

DreamDojo [50] 和 World Action Models [51] 是相关工作。

---

这篇paper的核心贡献在于：**把tactile modality真正集成进VLA foundation model，用cascaded flow matching和异步MoT架构解决了频率不匹配问题**。30%的平均提升不是incremental的，它展示了tactile-reactive manipulation作为new capability class的potential。未来的dexterous robot可能都需要类似的dual-frequency架构——视觉做planning，触觉做refinement，两者通过cached KV elegant地解耦。
