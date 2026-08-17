---
source_pdf: WALL-WM Carving World Action Modeling.pdf
paper_sha256: 01a6f9434131ae2ec4550837881a94e6dc75426b4beba7dec26435124e8a5d09
processed_at: '2026-08-13T03:34:10-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# WALL-WM 人话版

## 一句话概括

现在所有 VLA 模型都在干一件蠢事：用一个外部时钟（比如 "预测未来 16 步 action"）去切分训练数据，但 language、vision、action 三个模态各有各的自然节奏，硬把它们塞进同一个固定窗口，model 学到的是 short-horizon correlation，而不是真正的 world dynamics。WALL-WM 的核心主张是 **用 semantic event 作为原子训练单元**——reach、grasp、lift、place 这种在 language 能命名、video 能看见、action 能执行的自然单元。

## 核心问题：为什么 fixed-length chunk 是错的

想象你在学做菜。一个 recipe 说 "把鸡蛋打进碗里搅匀"。这句话描述的是一个完整的 semantic event，可能跨越 30 秒、50 个动作帧。如果你的训练数据是按 "每 2 秒一段" 切的，你会得到 15 段，每段都标着同一个 instruction "把鸡蛋打进碗里搅匀"。

这会发生什么？model 看到第 1 段（手伸向鸡蛋）学到 "手在动"，第 2 段（抓住鸡蛋）学到 "握紧"，第 3 段（移向碗）学到 "移动"... 每段都只学到局部 correlation，**完全不知道这 15 段连起来是一个完整的 grasp+transport+crack+pour 的因果链条**。

更糟的是，如果你的 chunk 横跨了 "grasp" 和 "lift" 两个 event，model 要么把它们混在一起学个平均，要么需要 history 才能知道 "我现在这个 chunk 到底在干什么"。这就是 paper 说的 **granularity mismatch**：

- Language 的粒度是 event 级的（"把杯子放到架子上"）
- Vision 的粒度是 continuous dynamics 级的（scene 每帧都在变）
- Action 的粒度是 control-step 级的（每 50ms 一个 action command）

三个模态有三种 native temporal structure，你硬用一把外部时钟的刀去切，就会把 semantic event 从中间切开，把多个 event 合并成一个 target，或者需要 historical context 仅仅才能搞清楚 "这个 chunk 在干什么"。

## Event 是什么

Paper 的引用很有意思——Plato 在 Phaedrus 里说的 "Carve nature at its joints"。一个 action-grounded semantic event 满足三个条件：

1. **Language 能 name 它**："reach to the cup"、"grasp the cup"、"lift the cup"、"place on shelf"
2. **Video 能 ground 它的时空演化**：从手开始动到稳定 grasp 住，video 有一段连续的 visual dynamics
3. **Action 能 realize 它**：这一段 action sequence 执行的是同一个 manipulation primitive

关键 insight：event 的边界由 underlying executable behavior change 定义，不由外部时钟定义。reach 结束、grasp 开始的瞬间就是 event boundary。这就像 LLM 里 token 的边界由 BPE 算法从数据中学出来，不是人为指定的字符数。

这个 idea 对我来说很有共鸣。你在 CS231n 讲过 "features matter more than classifiers"——这里完全类似，**the unit of learning matters more than the architecture**。你可以有再好的 transformer，如果训练单元本身是 misaligned 的，你学到的 prior 就是 misaligned 的。

## 架构怎么实现这个 idea

### Dual-Tower 而不是 Latent Action

现在有两条路线做 video+action world model：

- **Latent Action 路线**（LAPA、AdaWorld）：把 next observation 压缩成一个 vision-aligned 的 action code，下游 policy 再 decode
- **Dual-Tower 路线**（WALL-WM）：两个 DiT 并排放，video tower 和 action tower 通过 cross-attention 耦合

Paper 在 Appendix 9.1 给了一个很漂亮的分析：这两条路线其实是 **同一个 compression 的两端**。Latent action 是人为加 bottleneck；dual-tower 让 shared subspace 和 private capacity 的比例通过训练 emergent 地学到。形式化地说，设 $\mathbf{h}_t^{\text{shared}} \in \mathbb{R}^d$ 是 video tower 流向 action tower 的 cross-tower bottleneck activation，$\mathbf{z}_t$ 是 explicit latent action code。当 $d$ 匹配 codebook width 时，两者携带相同 information bottleneck，区别只在于 $\mathbf{z}_t$ 是离散的而 $\mathbf{h}_t^{\text{shared}}$ 是连续的、端到端学出来的。

**Dual-tower 是 latent action 的更 permissive 形式**，不需要你提前 guess bottleneck width 和 codebook size。如果你拧紧 shared sub-block，它就退化成 explicit latent action；放松就给 dynamics 更多表达空间。

### Video Tower：继承 Wan 的 prior

Video tower 继承自 Wan2.2 的 single-view DiT（pixel-space, native-T2V）。为什么选 pixel-space 而不是 latent-space？因为 pixel-space video foundation models 已经在 far broader 和 far thicker corpora 上 pretrained 了，weights 编码了 strong visual world prior。

为什么选 native-T2V 而不是 native-I2V？这个 trade-off 很 subtle。Native-I2V（V-JEPA-2、LeWorldModel）更 tightly track visual signal 的 temporal evolution——给定 previous frame，visual continuation 在 every pixel densely supervised。听起来很好，但 **I2V removes the semantic anchor**。没有 up-front commitment，I2V loss 的 easiest minimum 是 conditioned on previous frame 的 high-bandwidth pixel extrapolation——closer to learned optical flow than world model。任何教 high-level physical regularities 的 gradient 只 very thinly flow through that route。

Native-T2V 带来不同 flavor 的 prior：text 行为像 high-dimensional visual futures manifold 上的 low-dimensional cluster-center label。T2V objective 隐式 ask network 先 commit to trajectory 的 semantic class（"这是 grasp 还是 pour？"），再 paint it。这是 **semantically anchored visual self-supervision delivered for free at internet scale**。WALL-WM 留在 T2V side，让 cluster-center prior paid for once 并 inherited downstream。

### Multi-View Adaptation：Zero-Init Grafting

Wan 是 single-view 的，WALL-WM 需要支持 multi-view（ego + left-wrist + right-wrist）。怎么加 cross-view attention 而不破坏 pretrained prior？

公式 1：
$$\mathbf{h}_i^V \gets \mathbf{h}_i^V + g_i W_{\text{view}} \text{CrossViewAttn}_i(\mathbf{h}_i^V), \quad W_{\text{view}} \text{ initialized to } 0$$

这里 $\mathbf{h}_i^V$ 是 DiT block $i$ 的 hidden states（within-view Wan layout），$W_{\text{view}}$ 是 zero-init 的 output projector，$g_i$ 是 AdaLN gate。**Zero-init 是关键 trick**：训练开始时这个 branch 贡献为零，pretrained Wan 的 within-view 行为完全保留；cross-view exchange 只在训练中逐渐 turn on。这是一种 prior-preserving grafting——你不会因为加了一个新模块就破坏继承的视觉先验。

### Cross-View Geometric Masking：Sight-Cone + Tube

Cross-view attention 有两个 failure mode：一是 network 把它当 generic feature mixer 用，即使两个 patch 根本没有 co-visible region 也 attend；二是 network 在 single-view 内有 temporal shortcut，懒得用 cross-view。

Paper 用一对 training-only mask 同时解决两个问题。

**Sight-cone mask**（公式 2-6）：对同一 latent frame 上的两个 token $u = (\nu_u, h_u, w_u)$ 和 $u'$，把每个 token 的 patch back-project 成一个 cone $C(u) = (\mathbf{p}_0(u), \hat{\mathbf{v}}(u), \gamma(u))$，apex 在 camera center，axis 指向 patch center，half-apex angle $\gamma$ 紧紧包围 patch。然后在 depth-of-field band $[d_{\min}, d_{\max}]$ 内测试两个 cone 是否相交：

$$C(u) \text{ intersects } C(u') \iff \|\mathbf{p}(u, \hat{t}_1) - \mathbf{p}(u', \hat{t}_2)\|_2 \leq \hat{t}_1 \gamma(u) + \hat{t}_2 \gamma(u')$$

其中 $\hat{t}_1, \hat{t}_2$ 是 clamped 最近点距离对应的 depth。这产生一个 binary mask $\mathcal{M}_{\text{sc}}[u, u']$，以 $(1 - \mathcal{M}_{\text{sc}}) \cdot (-\infty)$ 作为 attention bias 加入。

**效果**：cross-view attention 只在物理上可能 co-visible 的 patch 之间发生。camera calibration 在 runtime 不需要，只在 training 时用 calibration 算 mask。

**Tube patch masking**：以概率 $p_{\text{tube}}$ 选一个 view $\nu^*$ 和一个 spatial window，把这条 "tube"（同一 spatial window 跨所有 latent frames）在 noisy input 中用纯噪声替换；以 nested 概率 $p_{\text{tube}}^{\text{cond}}$ 在 conditioning channel $y$ 上也 mask 掉。tube 内的 token 没有 within-view temporal shortcut，必须通过其他 $N_\nu - 1$ 个 views 来 recover。

为什么这两个 mask 互补？**Sight-cone 管 attention topology**（在哪能 attend），**Tube 管 input content**（哪必须 attend）。单独用 sight-cone 不 push traffic 沿允许的边走；单独用 tube 不 block geometrically nonsensical correlations。两者一起用，model 只能在几何允许且必须的地方 cross-view attend。

**两个 mask 都只在 training 用，inference 时全部 drop**，runtime 保持 calibration-free。这是个很 elegant 的 design——用 training-time 的 geometric supervision 教 model 学会 cross-view correspondence，然后 inference 时纯靠 learned representation。

### Action Tower：Layer-Wise Coupling

Action tower 是随机初始化的 action DiT，和 video tower 等深。每个 action block 做 4 件事：(a) action tokens 的 self-attention，(b) 专门对 state token 的 cross-attention，(c) 对 matched video block 的 cross-attention，(d) gated FFN。

公式 9 描述了 coupling：
$$\tilde{\mathbf{h}}_i^V = \text{ViewConcat}(\mathbf{h}_{\pi(i)}^V) + E_\tau(\tau^V) + E_{\text{abs}}(t_{\text{abs}})$$

$\pi(i)$ 是 depth map，把 action block $i$ 配对到 video block $\pi(i)$；ViewConcat 把那一层的 $N_\nu$ 个 per-view token sequences 沿 sequence axis 拼起来；$E_\tau$ 和 $E_{\text{abs}}$ 是两个 learnable temporal embedding。**Coupling 是单向的**：action 读 video，video 不被 action 改。

这里有个 subtle 的设计：**state token 有独立的 cross-attention**，不参与 action tokens 的 long video K/V sequence。absolute proprioception 在每个 depth 都直接 reachable，不被 video context 稀释。好的架构设计往往是在 "让正确的信息走正确的路径"——state proprioception 和 visual context 的路径应该分开。

### Asymmetric 1-to-$N_d$ Mapping：Video Frozen 时怎么训 Action

这个设计很 clever。Video 和 action 有各自的 denoising schedule，但 action block 要 cross-attend video feature，所以需要指定 action step $j$ 读哪个 video step。

**Symmetric 1-to-1**：$m(j) = j$，即 $t^A = t^V$。用于小数据端到端联合训练。

**Asymmetric 1-to-$N_d$**（默认）：固定一个 anchor $s^\star = 45$（50 步 schedule），所有 action step 都读同一个 anchored video forward：$m(j) = s^\star$ for all $j$。

为什么 asymmetric 是对的？**在 video frozen 的 regime 下，高噪声 video feature 和 ground truth 不匹配，近干净 feature 又太结构化**。pin 在 $s^\star$ 是一个 sweet spot，既有 faithful visual structure，又有 usable cross-attention evidence。

Action 在自己独立的 noise level $t_k^A \sim \Phi_A$ 上训练，但 video 证据是 anchored 的。这让你能在 video 只做一次 forward 的情况下，训练 action 在 full schedule 上的所有 noise level。**Throughput trick**：每个 optimizer step 可以画 $K=6$ 个独立的 action noise level，复用同一个 anchored video forward。这只是训练 trick，inference 时不用。

这个设计的深层意义：**video tower 的 role 从 "co-denoising partner" 变成 "frozen visual evidence provider"**。你 pretrained 了 visual prior，然后用它作为 action learning 的 anchored context，不让 action gradient 反过来 perturb 你好不容易学到的 visual structure。这很 clean——vision as world model prior for action，而不是 vision+action joint diffusion。

## 两种 Window Layout

### Event-Centric Window（公式 10）

用于 pretraining 和 event-mode inference。每个 token 拿到一个 integer frame index $\tau$：
$$\tau_{(f,h,w)}^V = f, \quad \tau_0^A = 0, \quad \tau_{1+k}^A = \lfloor k/K_p \rfloor + 1 \text{ for } k \in [0, T_a)$$

state token 和 zeroth latent frame 共享 $\tau = 0$，action tokens 按 $K_p$（每个 latent frame pool 的 action steps）分组对齐到 successive future latents。$E_\tau$ 在 cross-attention 两侧都加，bias 每个 action group 朝向 matching latent frame。

### Observation-Centered Window（公式 11）

用于 unified-mode deployment。窗口扩展为 $M$ history frames + 1 observation anchor + $N$ future frames：
$$\mathbf{h} += E_\tau(\tau) + E_{\text{abs}}(t_{\text{abs}})$$

$t_{\text{abs}}$ 索引 sliding window（哪个 chunk 是当前的），$E_\tau$ 跨 history indices $-M, \ldots, -1$、anchor $0$、future indices $+1, \ldots, +N$。

这里有个很巧的 **VAE-aligned video stream** 设计：Wan 的 3D VAE 是 "1+4×" temporal codec（一个 keyframe + 4× raw frames 压成 1+× latents）。观察窗口在 VAE 层面把 $1 + 4M + 4N$ 的 raw buffer 一次 encode 成 $1 + M + N$ 个 latents，history 和 future 之间没有 re-encoding seam。Chunk-DiT 只 denoise $N$ 个 trailing future latents。

## Length-Aware Caption-Drop：让 Model 学会从 Observation 推物理

公式 20 的 schedule：
$$\rho(L_e) = \begin{cases} \rho_{\min}, & L_e \leq L_{\min} \\ \rho_{\min} + (\rho_{\max} - \rho_{\min}) \frac{1 - \cos(\pi \frac{L_e - L_{\min}}{L_{\max} - L_{\min}})}{2}, & L_{\min} < L_e < L_{\max} \\ \rho_{\max}, & L_e \geq L_{\max} \end{cases}$$

参数：$\rho_{\min} = 0.1, \rho_{\max} = 0.9, L_{\min} = 129, L_{\max} = 220$。

**直觉**：长 event 更容易被 drop caption，迫使 model 从 observation 推断物理 continuation 而非 lexically specified sub-goals；短 event 保留 caption 更多，因为它太短了，从 observation 不足以确定接下来要干什么。

这是一种 **observation-anchored future synthesis** 的 curriculum——让 model 学会 "即使没有语言，也能从当前状态推物理上 plausible 的接触和末端动力学"。反过来，短 event 必须靠 caption 锚定语义，否则 model 不知道这一小段在干什么。

## Staircase Latent CoT：并行生成 continuous reasoning states

传统 CoT autoregressive 生成 discrete token，计算量大。LaDiR 等工作用 latent CoT 但仍然 serial——每个 reasoning step 要重算 lower features。

WALL-WM 的 Staircase Decoding（公式 16-17）把 reasoning 建模成 $K_c$ 个 continuous latent states：
$$\hat{y}_{1:K_c} = \{\hat{y}_1, \ldots, \hat{y}_{K_c}\} = \mathcal{F}_{\text{stair}}(x; N_r)$$

实现为 lightweight Mixture-of-Transformers (MoT) 耦合到 frozen Qwen3.5-9B backbone。在 relay depth $N_r$ 处把 Transformer 切开：**只有第一个 latent position 走 lower layers**，产生 shared relay representation 复用于所有 reasoning positions；**其余 latent states 在 upper blocks 并行生成**，各自有独立的 causal cache。

**直觉**：lower layers 编码 shared visual-language grounding（对所有 reasoning step 都一样），upper layers 渐进 specialize 到不同 reasoning step。这避免了为每个 reasoning step 重算 lower features。

### Frozen Latent-to-Text Reconstruction Supervision（公式 18-19）

生成的 $\hat{y}_{1:K_c}$ 通过 prefix projector $\mathcal{P}_{\text{pref}}$ 投影到 soft prefix $\mathbf{z}_{1:K_c}$，在 frozen Qwen3.5-0.8B 的 embedding space 中 autoregressive 地 reconstruct 对应的 textual CoT trace $r_{1:M_r}$：

$$\mathcal{L}_{\text{CoT}} = -\sum_{m=1}^{M_r} \log P_\phi(r_m \mid \mathbf{z}_{1:K_c}, r_{<m})$$

**只训练 staircase reasoning branch 和 prefix projector**，reconstruction language model 全程 frozen。

这是个很 elegant 的 supervision 设计——你不需要 latent states 完全等于文本 tokens，你只需要它们包含足够信息让一个 frozen LM 能 reconstruct 文本。这给了 latent states 一些 "compression freedom"，鼓励它们编码 compact high-level reasoning semantics 而不是 replicate exact token-level decoding trajectories。

## 数据生态：四象限 Data Map

### 四个 Quadrant

1. **General internet video**：1.2M-clip OpenVID slice + 其他 web video，提供 broadest visual-temporal dynamics prior
2. **Egocentric video**：Ego4D, EPIC-KITCHENS，narrow toward first-person manipulation geometry without robot actions
3. **Non-embodiment UMI-style**：XRZero-G0 可穿戴 rig，VR-tracked headset + 三个 egocentric cameras + handheld grippers（geometry 匹配 deployment robot end-effector），operator 不连机器人。**Key insight**：collection throughput 不再被 robot time bound
4. **Heterogeneous robot teleoperation**：DROID, AgiBot World, 自采集；四个 deployment platforms

中间是 **Human-intervention 和 failure-recovery data**——nominal demonstration 罕见提供的 contact-rich correction 信号。

### Structured vs Unstructured Collection

- **Structured**：预定义 task scope 和 reset protocol，每个 episode 是一个 named task 的 teleop demonstration
- **Unstructured**：operator 在 deployment scene 自由移动机器人，没有 task scope、reset protocol、episode boundary，产生 long multi-event in-distribution motion stream

Unstructured admission 是 scale 的关键——它去掉了 per-episode protocol overhead，让 collection throughput 突破 teleop demonstration 的 natural ceiling。Caption-then-cluster pipeline 吸收 resulting heterogeneity 进同一个 balanced sampler。

### Temporal Synchronization

[video+action] layers 只有在 visual observation 和 action stream 指向同一物理瞬间时才有用。但 camera encoding、controller logging、teleop middleware、disk writing 会引入 nearly constant video-action offset。**特别在 contact 附近**，几帧就能把 semantic state 从 "approaching" 变成 "touching"。

方法：对每个 episode，构造 visual motion signal（frame-to-frame image change across ego + wrist cameras）和 action-motion signal（left/right end-effector position 的 finite difference）。两者 smooth + normalize，sweep small integer lag window 找最大化 correlation 的 offset。20 FPS 下两帧 offset 约 100ms，synchronizer 应用对应 temporal shift 对齐 logged action stream 与 encoded video timeline。

### Hierarchical Captioning（图 12）

4-level temporal-nested caption hierarchy，spans 在 atomic manipulation actions 上 ground：

- **Task (L3)**：episode-global string，总结 overall objective
- **Subtask (L2)**：少数 contiguous semantically meaningful stages（approach target、establish grasp、transport、place/release）
- **Action (L1)**：short manipulation primitives（reach、align gripper、close fingers、lift、translate、insert、retract）
- **Segment (L0)**：finest temporal decomposition，short localized events
- **Human (optional)**：manual annotation subset，用于验证自动 hierarchy

**对 recovery 行为的意义**：很多有用 demonstration 含 regrasp、failed contact、小 pose correction、slip 后 retry。如果整个 episode 只有一个 caption，这些 corrective behaviors 被 average 进 global task description。Hierarchical captioning 让这些 events 在时间上 localize 并获得自己的 description，dataloader 可以 sample 或 reweight 特定 temporal region，不把 successful episode 所有 frame 等权对待。

### Cluster-Balanced Sampling

两轮 offline clustering：
- **Vision-Language clustering**：frozen multimodal encoder 把每个 (observation, caption) pair 映射到 joint embedding，partition 成 topic clusters
- **Action clustering**：action chunks 在 trajectory space 单独聚类，long tail 集中 non-nominal motion（recoveries、re-grasps、retries、contact-driven corrections）

**关键**：action-aligned decomposition 把 language-only 和 vision-language clustering views 变得更 evenly distributed，fewer samples 被少数 dominant topics 吸收。Training 时 dataloader 同时平衡 VL clusters 和 action clusters。

### Recovery Data via Contact-Rich Random Initialization

公式 19：
$$\tilde{p}_{\text{train}}(\mathbf{q}, e) = (1 - \alpha) p_{\text{nominal}}(\mathbf{q}, e) + \alpha \mathbb{E}_e[p(\mathbf{q} \mid e)]$$

对每个 contact event $e$，定义 local contact-pose distribution $p(\mathbf{q} \mid e)$，support 在以 nominal contact pose $\mathbf{q}_e^\star$ 为中心的小 geodesic ball $\mathcal{B}_\epsilon(\mathbf{q}_e^\star)$ 内。**实践**：perturb robot initialization around $\mathbf{q}_e^\star$，然后 replay original demonstration 或 collect fresh recovery rollout。这创造 controlled local coverage around each contact event，而非只观察穿过它的单一 nominal trajectory。

Clustering surface non-nominal trajectories already in corpus；recovery initialization actively creates such trajectories in contact-space regions where nominal data too sparse for clustering alone。

## 训练 Pipeline：5 个 Stage

| Component | Video PT | Action PT | VLM text | Staircase | Next-chunk |
|-----------|----------|-----------|----------|-----------|------------|
| 3D causal VAE | ✗ | ✗ | ✗ | ✗ | ✗ |
| T5 text encoder | ✗ | ✗ | ✗ | – | ✗ |
| VLM backbone (Qwen3.5-9B) | – | – | ✗ | ✗ | ✗ |
| VLM project-out / aux heads | – | – | ✓ | ✗ | ✗ |
| Video DiT (incl. view attention) | ✓ | ✗ | ✗ | ✗ | ✓ |
| Action DiT (incl. layer coupling) | – | ✓ | ✗ | ✗ | ✓ |
| Staircase MoT branches | – | – | – | ✓ | ✗ |

**Stage 1: Video Pretraining**。只训 video DiT，event latents 上 Wan-style $v$-prediction flow matching。Cross-view branch zero-init。Condition on current multi-view observation + event-aligned captions（frozen T5 encode）。Event span truncate 到 65 latent frames。Length-aware caption-drop。Prune quasi-static frames。

**Stage 2: Action Pretraining**。Freeze video DiT。只训 action tower，event-centric pairs。Asymmetric 1-to-$N_d$ mapping，pin $s^\star = 45$ on 50-step schedule。$K = 6$ parallel action-noise draws per optimizer step 复用同一 anchored video forward。

**Stage 3: VLM Text-Conditioner Pretraining**。公式 21 三项目标：
$$\mathcal{L}_{\text{text}} = \lambda_{\text{align}} \|\mathbf{c}_\ell^{\text{VLM}} - \mathbf{c}_\ell^{\text{T5}}\|^2 + \lambda_{\text{next}} \text{CE}(\hat{c}_{\text{next}}, c_{\text{next}}) + \lambda_{\text{time}} \text{Huber}(\widehat{\Delta t}, \Delta t)$$

第一项 align VLM conditioning features 到 original T5 encoding——这让 upgraded VLM 成为 T5 的 drop-in replacement，从 DiT 视角看 text conditioning geometry 不变。第二项是 next-event caption 的 token-level LM loss。第三项是 remaining-time scalar 的 Huber regression。VLM backbone 全程 frozen，只训 project-out head、next-event head、remaining-time regressor。

**Stage 4: Staircase Distillation**。只训 staircase reasoning branch 和 prefix projector，optimize latent-to-text reconstruction loss $\mathcal{L}_{\text{CoT}}$。

**Stage 5: Next-Chunk Adaptation**（optional）。Fine-tune event-pretrained backbone under next-chunk prediction on observation-centered layout。Global instructions 来自 Task level caption。Both DiT towers 在 fixed-shape windows 上 update，asymmetric anchor protocol 保留。Cluster-balanced sampling 在 history-conditioned windows 上重跑 clustering passes。

## 两种 Inference Mode

### Event-Mode Inference

每个 rollout step：video 和 action DiT 条件化 current multi-view observation、proprioceptive state、next-event description。Description 来自 human 或 fine-tuned Qwen3.5-9B heads。Model denoise 整个 event-aligned video+action segment（变长窗口）。Execution 完成后 observation advance，新 next-event description 条件化下一段。

### Unified-Mode Inference

Rollout fixed $H_a$-step video-action chunks。每个 step 条件化 current observation + configurable history window + global instruction。**三个 interchangeable sources** 供应 per-chunk text-side context $\mathbf{c}_\ell$，可以 mid-rollout 切换不改变 denoiser state：

1. **Gradient-continuous source**：text encoder 把 global instruction 一次映射成 continuous $\mathbf{c}_\ell$，verbatim 喂 cross-attention，所有 rolling chunks 复用
2. **Atomic-instruction source**：upstream planner 或 human 每个 fixed $H_a$-step chunk boundary 提供一个 short instruction
3. **VLM-CoT source**：Staircase decoder 在单次 parallel forward emit CoT latents

**同一个 event-pretrained backbone 同时支持 event-driven 变长 rollout 和 fixed-horizon VLA execution**，不需要两个不同的 model。

## 实验

### Embodied Video Generation（Table 2）

vs Wan2.1-1.3B 和 Wan2.2-5B：

| Model | Motion Quality | Semantic Consistency | Physical Plausibility |
|-------|----------------|----------------------|----------------------|
| Wan2.1-1.3B | 0.619 | 0.857 | 0.219 |
| Wan2.2-5B | 0.683 | 0.805 | 0.226 |
| **WALL-WM** | **0.771** | **0.886** | **0.434** |

WALL-WM 在 embodied-relevant dimensions 全面领先。最显著的是 **Physical Plausibility (0.434 vs 0.226)** 和 **Interaction Quality (0.434 vs 0.226)**——large-scale embodied training 把 inherited Wan video prior 转化为 robot-object interaction 和 contact evolution 的 physical prior。

Table 3 的 3D awareness benchmark on CO3Dv2：WALL-WM 在 point error (0.271)、depth error (0.132)、AUC@5 (0.210)、AUC@30 (0.727) 上和 WAN2.1-14B 持平或略好，complement strong multi-view consistency。

### Real-Robot Four Suites（Table 5）

**Diverse Manipulation**（7 个 task）：
- WALL-WM-E: **75.86** avg
- WALL-WM-U-Scratch: 63.00
- π0.5: 55.64
- DreamZero: 39.97
- LingBot-VA: 29.71

**Reasoning Manipulation**（5 个 task）：
- WALL-WM-E: **71.60** avg
- WALL-WM-U-Scratch: 59.50
- π0.5: 56.40
- DreamZero: 32.70
- LingBot-VA: 31.60

**Dexterous Manipulation**（2 个 task）：
- WALL-WM-E: **32.00** avg
- WALL-WM-U-Scratch: 31.25
- DreamZero: 25.00
- LingBot-VA: 24.00
- π0.5: 15.00

**Generalization**（4 个 task）：
- WALL-WM-E: **53.75** avg
- DreamZero: 28.50
- π0.5: 24.00
- WALL-WM-U-Scratch: 18.50

### Ablation（Table 4）

对比 pretrained event-mode WALL-WM vs pretrained fixed-length unified baseline without View-Interaction Self-Attention (VI-SA)：

- Reasoning Manipulation: 32.6 → **71.6** (+39.0)
- Generalization: 22.0 → **53.75** (+31.75)

Press Button in Order 从 0 → 64，Pair Up Items 从 8 → 36。这个 ablation 同时移除 VI-SA 和 event-conditioned execution，所以 measure 的是 combined effect。

## Infrastructure：Muon、Kernel Library、FP8

### DMuon

Muon optimizer 的 Newton-Schulz iteration 在 vanilla 实现中，optimizer step 单独可能接近 2× forward+backward 成本。**DMuon** 是 Muon 的 distributed 实现：

- **Pipeline Scheduling**：dedicated-ownership scheme，constrained Longest-Processing-Time assignment 把每个 matrix parameter map 到 unique owner rank。Native all-reduce/reduce-scatter 替换成 reduce/broadcast，post-step broadcast 在 dedicated CUDA stream 上异步 issue，与后续 forward pass overlap
- **Kernel Optimization**：Gramian formulation 有 symmetric structure，general-purpose GEMM 无法 exploit。用 CuteDSL kernel aligned to symmetric factor，shape-aware autotuning 维持 speedup

### Kernel Library

Stock kernels 在 single-operator granularity 操作，对 tensor-core throughput bound 的 patterns 不必要地 spill to global memory。识别 recurring fusible patterns，consolidate 成 composite kernels，让 intermediate activations 留在 registers 或 shared memory across fused region。

基于 **TVM FFI** 构建——language-agnostic foreign function interface，通过 stable C ABI 暴露 compiled kernels with zero-copy tensor handles，bypass per-call PyTorch dispatcher cost 和 GIL。训练和推理用同一 compiled kernel，maintain numerical consistency。

### Model Compression

**Distillation**：Distribution-matching distillation (DMD) 训练 few-step student generator align output distribution with multi-step teacher。Joint distillation objective 保留 original action-prediction loss alongside distributional term，所以 few-step student 同时 aligned with teacher video distribution 和 anchored to action supervision。Ablation 显示如果只做 distributional supervision，action MAE degrade 53%。

**FP8 Quantization**：post-training quantization with per-block scaling。Weights offline quantize 成 pre-packed FP8 tensors with baked-in scaling factors；on-the-fly activation quantization fuse 进 preceding operator 的 epilogue，per-block scale 和 FP8 cast 在同一 kernel 内完成。

结合 CUDA Graph capture eliminate host-side launch overhead，**full stack 让 end-to-end inference 到 10Hz**，meet closed-loop robotic control 的 latency budget。

## 我的 Take

### Event Unit 与 Token 的类比

Action-grounded semantic event 在某种意义上是 **spatiotemporal token**——一个在 language、vision、action 三模态都有 well-defined 边界的单元。这让我联想到 LLM 中 token 的角色：token 是 language 的 atomic unit，但它的边界是 syntactic/semantic 的（BPE、SentencePiece）。WALL-WM 的 event 是 embodied domain 的 "semantic token"，边界由 executable behavior change 定义。未来的方向可能是 self-supervised event boundary discovery，类似 LLM 的 tokenizer learned from data。

### 与 JEPA 家族的对比

Yann LeCun 的 V-JEPA / V-JEPA-2 / LeWorldModel 走的是 native-I2V + latent prediction in feature space 路线。WALL-WM 的设计哲学正好相反：pixel-space + native-T2V + dual-tower。

这不是谁对谁错，而是 **"你愿意花多少 upfront commitment 代价换多少 free semantic supervision"** 的 trade-off。JEPA 的 latent prediction 更 "efficient" 但失去了 T2V 的 cluster-center anchor；WALL-WM 保留 anchor 但接受 pixel-space 的更重 optimization landscape。

### Latency vs Generalization

Paper Section 8 有一段我特别认同：**"if latency is enforced too early, it may cap the reachable performance ceiling before the model has learned a sufficiently general world-action prior"**。distillation、quantization、speculative execution、systems-level overlap 可以 progressively reduce latency once a strong model exists；recovering a lost generalization ceiling from an under-scaled model is much less straightforward。

这和 "在 representation learning 中 accuracy > efficiency at training time, efficiency at inference time" 的观点类似。先把 model 学强，再把它压快。

## 总结

WALL-WM 的核心贡献是一个完整的 prior-preserving scale-up methodology：

1. **重新定义训练单元**：从 fixed-length chunk 到 action-grounded semantic event
2. **Prior-preserving grafting**：Wan video prior 通过 zero-init projector、frozen video tower during action training、T5-aligned VLM conditioning 被保留
3. **Dual-tower layer-coupled architecture**：作为 latent action 的更 permissive 形式
4. **Multi-view geometric inductive bias**：sight-cone + tube mask 在 attention topology 和 input content 两个维度强制 cross-view geometric consistency
5. **Staircase parallel latent CoT**：解决 long-horizon reasoning 的 efficiency 问题
6. **Muon-based infrastructure**：把 10Hz closed-loop control 拉到可及范围

一句话：**"don't flatten three modalities into one window; instead, find the joint where they all agree, and carve there."**

---

参考链接：
- [WALL-WM GitHub](https://github.com/X-Square-Robot/wall-x)
- [Wan Video Generation Models](https://arxiv.org/abs/2503.20314)
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)
- [π0.5](https://arxiv.org/abs/2504.16054)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [RoPE / RoFormer](https://arxiv.org/abs/2104.09864)
- [Distribution Matching Distillation](https://arxiv.org/abs/2312.13837)
- [DROID Dataset](https://droid-dataset.github.io/)
- [Agibot World](https://arxiv.org/abs/2503.06669)
- [Ego4D](https://ego4d-data.org/)
- [EPIC-KITCHENS](https://epic-kitchens.github.io/2024)
- [OpenVID-1M](https://arxiv.org/abs/2502.15892)
- [JEPA / V-JEPA](https://arxiv.org/abs/2301.08243)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [CogVideoX](https://arxiv.org/abs/2408.06036)
- [Open-Sora 2.0](https://arxiv.org/abs/2503.09642)
- [Plato, Phaedrus 265e](https://www.perseus.tufts.edu/hopper/text?doc=Perseus%3Atext%3A1999.01.0174%3Atext%3DPhaedrus)

---

# WALL-WM: 在事件的关节处雕刻 World Action Modeling

## 1. 核心直觉：Granularity Mismatch 是 VLA 的根本病灶

这篇 paper 的出发点非常 sharp。大多数 VLA 模型（RT-2, OpenVLA, π0 等）都继承了一个看似无害的默认约定：**用 fixed-length action chunk 作为原子训练单元**。比如 "给当前 observation + instruction，预测未来 16 步 action"。这个 chunk 长度是外部时钟强加的，和任务本身的语义结构毫无关系。

WALL-WM 的核心论点是：**language、vision、action 三个模态各有各的 native temporal structure**，强行用同一个固定窗口把它们对齐，会产生三种灾难：

1. **Language 的粒度太粗**：一个 instruction "把杯子放到架子上" 描述的是一个完整的 semantic event，可能跨越 50 个 control steps。如果 chunk 只有 16 步，model 根本看不到完整的语义单元，只能学到 "在某个中间状态下，action 是什么" 这种 short-horizon correlation。

2. **Vision 的演化是 continuous 的**：scene dynamics 在接触瞬间有剧烈变化，在 free-space motion 时又很 smooth。fixed chunk 可能把一个 grasp event 从中间切开，前半段是 approach，后半段是 contact，model 无法学到这个 event 的完整因果结构。

3. **Action 的 timescale 是 control-level 的**：它对 contact、timing、小扰动极度敏感。如果一个 chunk 横跨了多个语义不同的 action primitive（reach + grasp + lift），model 要么把它们混在一起，要么需要 history context 才能知道 "这个 chunk 到底在干什么"。

这个 insight 用 Plato 的话总结就是 **"Carve nature at its joints"**——在自然的关节处切分，而不是用一把外部时钟的刀乱砍。一个 action-grounded semantic event（reach、grasp、lift、place）是同时满足三个条件的单元：language 能 name 它、video 能 ground 它的时空演化、action 能 realize 它。

这让我想起你在 CS231n 讲的 "features matter more than classifiers"——这里的类比是 **"the unit of learning matters more than the architecture"**。你可以有再好的 transformer 架构，如果训练单元本身是 misaligned 的，你学到的就是 misaligned 的 prior。

## 2. 架构全景：Layer-Coupled Dual-Tower Denoiser

### 2.1 为什么是 Dual-Tower 而不是 Latent Action

Paper 在 Appendix 9.1 给了一个很精彩的设计空间分析。当前有两条路线：

- **Latent Action 路线**（LAPA、AdaWorld）：用一个 encoder/codebook/decoder 把 next observation 压缩成一个 vision-aligned 的 implicit action code，下游 policy 再 decode 它。
- **Dual-Tower 路线**（WALL-WM、LingBot-VA）：两个 DiT-scale 的 tower 并排放着，video tower 和 action tower 通过 cross-attention 耦合。

乍看是对立的，但 paper 指出它们其实是 **同一个 compression 的两端**。Latent action 是人为加一个 bottleneck；dual-tower 是让 shared subspace 和 private capacity 的比例通过训练 emergent 地学到。如果你把 dual-tower 的 shared sub-block 拧紧，它就退化成 explicit latent action；如果你放松，就给 dynamics 更多表达空间。

形式化地说，设 $\mathbf{h}_t^{\text{shared}} \in \mathbb{R}^d$ 是 V tower 流向 A tower 的 cross-tower bottleneck activation，$\mathbf{z}_t$ 是 LAPA 风格的 explicit latent action code。当 $d$ 匹配 codebook width 时，两者携带相同的 information bottleneck，区别只在于 $\mathbf{z}_t$ 是离散的而 $\mathbf{h}_t^{\text{shared}}$ 是连续的、端到端学出来的。

这个 framing 很漂亮：**dual-tower 是 latent action 的更 permissive 形式**，不需要你提前 guess bottleneck width 和 codebook size。

### 2.2 Video Tower：继承 Wan 的 Native T2V Prior

Video tower 继承自 Wan2.2 的 single-view DiT，然后 graft 了三个东西：

**Multi-View Adaptation**（公式 1）：
$$\mathbf{h}_i^V \gets \mathbf{h}_i^V + g_i W_{\text{view}} \text{CrossViewAttn}_i(\mathbf{h}_i^V), \quad W_{\text{view}} \text{ initialized to } 0$$

这里 $\mathbf{h}_i^V$ 是 DiT block $i$ 的 hidden states（within-view Wan layout），$W_{\text{view}}$ 是 zero-init 的 output projector，$g_i$ 是 AdaLN gate。**Zero-init 是关键 trick**：训练开始时这个 branch 贡献为零，pretrained Wan 的 within-view 行为完全保留；cross-view exchange 只在训练中逐渐 turn on。这是一种 **prior-preserving grafting**——你不会因为加了一个新模块就破坏继承的视觉先验。

**Camera RoPE**：给每个 camera 一个 learnable rotary identity，不需要运行时喂 calibration。RoPE 的 frequency bank 被分区为 $(f, h, w, \text{view})$，view rotation 来自一个 per-view learnable embedding，在所有 view-attention layer 间共享。加/减一个 camera 只需要改 embedding table。

**Cross-View Geometric Masking**：这一对 training-only mask 是我最喜欢的设计之一。

第一个是 **Sight-cone attention mask**（公式 2-6）。对同一 latent frame 上的两个 token $u = (\nu_u, h_u, w_u)$ 和 $u' = (\nu_{u'}, h_{u'}, w_{u'})$，把每个 token 的 patch back-project 成一个 cone $C(u) = (\mathbf{p}_0(u), \hat{\mathbf{v}}(u), \gamma(u))$，apex 在 camera center，axis 指向 patch center，half-apex angle $\gamma$ 紧紧包围 patch。然后在 depth-of-field band $[d_{\min}, d_{\max}]$ 内测试两个 cone 是否相交：

$$C(u) \text{ intersects } C(u') \iff \|\mathbf{p}(u, \hat{t}_1) - \mathbf{p}(u', \hat{t}_2)\|_2 \leq \hat{t}_1 \gamma(u) + \hat{t}_2 \gamma(u')$$

其中 $\hat{t}_1, \hat{t}_2$ 是 clamped 最近点距离对应的 depth。这产生一个 binary mask $\mathcal{M}_{\text{sc}}[u, u']$，以 $(1 - \mathcal{M}_{\text{sc}}) \cdot (-\infty)$ 作为 attention bias 加入。**效果**：cross-view attention 只在物理上可能 co-visible 的 patch 之间发生，防止网络把 cross-view attention 滥用成 generic feature mixer。

第二个是 **Tube patch masking**。以概率 $p_{\text{tube}}$ 选一个 view $\nu^*$ 和一个 spatial window，把这条 "tube"（同一 spatial window 跨所有 latent frames）在 noisy input 中用纯噪声替换；以 nested 概率 $p_{\text{tube}}^{\text{cond}}$ 在 conditioning channel $y$ 上也 mask 掉。**关键**：tube 内的 token 没有 within-view temporal shortcut，必须通过其他 $N_\nu - 1$ 个 views 来 recover。

为什么这两个 mask 是互补的？**Sight-cone 管 attention topology**（在哪能 attend），**Tube 管 input content**（哪必须 attend）。单独用 sight-cone 不 push traffic 沿允许的边走；单独用 tube 不 block geometrically nonsensical correlations。两者一起用，model 只能在几何允许且必须的地方 cross-view attend。**两个 mask 都只在 training 用，inference 时全部 drop**，runtime 保持 calibration-free。

### 2.3 Action Tower：Layer-Wise Coupling 和 Asymmetric Denoising

Action tower 是一个随机初始化的 action DiT，和 video tower 等深。每个 action block 做 4 件事：(a) action tokens 的 self-attention，(b) 专门对 state token 的 cross-attention，(c) 对 matched video block 的 cross-attention，(d) gated FFN。

公式 9 描述了 coupling：
$$\tilde{\mathbf{h}}_i^V = \text{ViewConcat}(\mathbf{h}_{\pi(i)}^V) + E_\tau(\tau^V) + E_{\text{abs}}(t_{\text{abs}})$$

其中 $\pi(i)$ 是 depth map，把 action block $i$ 配对到 video block $\pi(i)$；ViewConcat 把那一层的 $N_\nu$ 个 per-view token sequences 沿 sequence axis 拼起来；$E_\tau$ 和 $E_{\text{abs}}$ 是两个 learnable temporal embedding。**Coupling 是单向的**：action 读 video，video 不被 action 改。

这里有个很 subtle 的设计：**state token 有独立的 cross-attention**，不参与 action tokens 的 long video K/V sequence。这样 absolute proprioception 在每个 depth 都直接 reachable，不被 video context 稀释。这让我想起你在 "Software 2.0" 说的——好的架构设计往往是在 **"让正确的信息走正确的路径"**，state proprioception 和 visual context 的路径应该分开。

**Asymmetric 1-to-$N_d$ Mapping**（公式 12）是训练效率的关键。Video 和 action 有各自的 denoising schedule，但 action block 要 cross-attend video feature，所以需要指定 action step $j$ 读哪个 video step。两种 regime：

- **Symmetric 1-to-1**：$m(j) = j$，即 $t^A = t^V$。用于小数据端到端联合训练。
- **Asymmetric 1-to-$N_d$**（默认）：固定一个 anchor $s^\star = 45$（50 步 schedule），所有 action step 都读同一个 anchored video forward：$m(j) = s^\star$ for all $j$。

为什么 asymmetric 是对的？**在 video frozen 的 regime 下，高噪声 video feature 和 ground truth 不匹配，近干净 feature 又太结构化**。pin 在 $s^\star$ 是一个 sweet spot，既有 faithful visual structure，又有 usable cross-attention evidence。Action 在自己独立的 noise level $t_k^A \sim \Phi_A$ 上训练，但 video 证据是 anchored 的。这让你能在 video 只做一次 forward 的情况下，训练 action 在 full schedule 上的所有 noise level。

**Throughput trick**：每个 optimizer step 可以画 $K=6$ 个独立的 action noise level，复用同一个 anchored video forward。这只是训练 trick，inference 时不用。

## 3. Event-Centric Pretraining：把训练目标本身 ground 在 event 层级

这是 paper 的方法论核心。不是把 event 当 auxiliary condition，而是 **把训练问题本身放在 event 层级**。

### 3.1 两种 Window Layout

**Event-centric window**（公式 10）：用于 pretraining 和 event-mode inference。每个 token 拿到一个 integer frame index $\tau$，state token 和 zeroth latent frame 共享 $\tau = 0$，action tokens 按 $K_p$（每个 latent frame pool 的 action steps）分组对齐到 successive future latents：
$$\tau_{(f,h,w)}^V = f, \quad \tau_0^A = 0, \quad \tau_{1+k}^A = \lfloor k/K_p \rfloor + 1 \text{ for } k \in [0, T_a)$$

**Observation-centered window**（公式 11）：用于 unified-mode deployment。窗口扩展为 $M$ history frames + 1 observation anchor + $N$ future frames，激活两个 embedding：
$$\mathbf{h} += E_\tau(\tau) + E_{\text{abs}}(t_{\text{abs}})$$

$t_{\text{abs}}$ 索引 sliding window（哪个 chunk 是当前的），$E_\tau$ 跨 history indices $-M, \ldots, -1$、anchor $0$、future indices $+1, \ldots, +N$。这里有个很巧的 **VAE-aligned video stream** 设计：Wan 的 3D VAE 是 "1+4×" temporal codec（一个 keyframe + 4× raw frames 压成 1+× latents，leading-one/trailing-four 规则）。观察窗口在 VAE 层面把 $1 + 4M + 4N$ 的 raw buffer 一次 encode 成 $1 + M + N$ 个 latents，history 和 future 之间没有 re-encoding seam。

### 3.2 Video Flow-Matching Objective（公式 7-8）

Video tower 继承 Wan 的 $v$-prediction flow matching。给定 clean latents $\mathbf{z}_0^V$、noise $\varepsilon^V$、video timestep $t^V \sim \Phi_V$，form noisy latents $\mathbf{z}_t^V$，regress flow target $\mathbf{C}^{V\star} = \varepsilon^V - \mathbf{z}_0^V$：

$$\mathcal{L}_V = w_V(t^V) \|\hat{\mathbf{C}}^V - \mathbf{C}^{V\star}\|^2$$

当 tube masking 激活时（公式 8），masked tube $\mathcal{T}$ 内的 token 用额外权重 $\lambda_{\text{mask}}$：
$$\mathcal{L}_V = w_V(t^V) \left( \sum_{u \notin \mathcal{T}} \|\hat{C}_u^V - C_u^{V\star}\|^2 + \lambda_{\text{mask}} \sum_{u \in \mathcal{T}} \|\hat{C}_u^V - C_u^{V\star}\|^2 \right)$$

**Border masking** 是独立的：out-of-frame 和 synthetic black border regions 永远从 MSE 中排除。Prior-preserving main recipe 总是保留 border masking 和 sight-cone attention supervision，disable tube sampling。

### 3.3 Length-Aware Caption-Drop Schedule（公式 20）

这是个很有意思的 regularization。设 $L_e$ 是 event span 长度，$\rho(L_e) \in [\rho_{\min}, \rho_{\max}]$ 是 omitting event caption 的概率：

$$\rho(L_e) = \begin{cases} \rho_{\min}, & L_e \leq L_{\min} \\ \rho_{\min} + (\rho_{\max} - \rho_{\min}) \frac{1 - \cos(\pi \frac{L_e - L_{\min}}{L_{\max} - L_{\min}})}{2}, & L_{\min} < L_e < L_{\max} \\ \rho_{\max}, & L_e \geq L_{\max} \end{cases}$$

参数：$\rho_{\min} = 0.1, \rho_{\max} = 0.9, L_{\min} = 129, L_{\max} = 220$。**直觉**：长 event 更容易被 drop caption，迫使 model 从 observation 推断物理 continuation 而非 lexically specified sub-goals；短 event 保留 caption 更多，因为它太短了，从 observation 不足以确定接下来要干什么。这是一种 **observation-anchored future synthesis** 的 curriculum——让 model 学会 "即使没有语言，也能从当前状态推物理上 plausible 的接触和末端动力学"。

## 4. Staircase Latent CoT Decoding：并行生成 continuous reasoning states

这部分解决了 latent reasoning 的效率问题。传统 CoT autoregressive 生成 discrete token，计算量大；LaDiR 等工作用 latent CoT 但仍然 serial。

### 4.1 Staircase 架构（公式 16-17）

WALL-WM 把 reasoning 建模成 $K_c$ 个 continuous latent states：
$$\hat{y}_{1:K_c} = \{\hat{y}_1, \ldots, \hat{y}_{K_c}\}$$

实现为 lightweight Mixture-of-Transformers (MoT) 耦合到 frozen Qwen3.5-9B backbone。在 relay depth $N_r$ 处把 Transformer 切开：**只有第一个 latent position 走 lower layers**，产生 shared relay representation 复用于所有 reasoning positions；**其余 latent states 在 upper blocks 并行生成**，各自有独立的 causal cache。

$$\hat{y}_{1:K_c} = \mathcal{F}_{\text{stair}}(x; N_r)$$

$N_r$ 控制 shared grounding computation 和 parallel reasoning computation 的分界。**直觉**：lower layers 编码 shared visual-language grounding（对所有 reasoning step 都一样），upper layers 渐进 specialize 到不同 reasoning step。这避免了为每个 reasoning step 重算 lower features。

### 4.2 Frozen Latent-to-Text Reconstruction Supervision（公式 18-19）

不是直接 distill autoregressive hidden states，而是通过一个 frozen latent-to-text reconstruction objective 监督 latent reasoning：

生成的 $\hat{y}_{1:K_c}$ 通过 prefix projector $\mathcal{P}_{\text{pref}}$ 投影到 soft prefix $\mathbf{z}_{1:K_c}$，在 frozen Qwen3.5-0.8B 的 embedding space 中，autoregressive 地 reconstruct 对应的 textual CoT trace $r_{1:M_r}$：

$$P_\phi(r_{1:M_r} \mid \mathbf{z}_{1:K_c})$$

训练目标是 token-level reconstruction loss：
$$\mathcal{L}_{\text{CoT}} = -\sum_{m=1}^{M_r} \log P_\phi(r_m \mid \mathbf{z}_{1:K_c}, r_{<m})$$

**只训练 staircase reasoning branch 和 prefix projector**，reconstruction language model 全程 frozen。**效果**：latent reasoning states 被鼓励编码 compact high-level reasoning semantics，而不是 replicate exact token-level decoding trajectories。这是个很 elegant 的 supervision 设计——你不需要 latent states 完全等于文本 tokens，你只需要它们包含足够信息让一个 frozen LM 能 reconstruct 文本。这给了 latent states 一些 "compression freedom"。

## 5. 数据生态：Four-Quadrant Data Map

### 5.1 Data Source Map

四个 quadrant 按 viewpoint 和 action availability 分：

1. **General internet video**：1.2M-clip OpenVID slice + 其他 web video，提供 broadest visual-temporal dynamics prior。
2. **Egocentric video**：Ego4D, EPIC-KITCHENS，narrow toward first-person manipulation geometry without robot actions。
3. **Non-embodiment UMI-style**：XRZero-G0 可穿戴 rig，VR-tracked headset + 三个 egocentric cameras + handheld grippers（geometry 匹配 deployment robot end-effector），operator 不连机器人。**Key insight**：collection throughput 不再被 robot time bound。production recipe 是 "small real-robot anchor fraction + 大量 no-embodiment clips"——few-shot physical anchoring regime。
4. **Heterogeneous robot teleoperation**：DROID, AgiBot World, 自采集；四个 deployment platforms（desktop bimanual、QUANTA X1/X1 Pro mobile、QUANTA X2 wheeled humanoid with dexterous hands）。

中间是 **Human-intervention 和 failure-recovery data**——nominal demonstration 罕见提供的 contact-rich correction 信号。

### 5.2 Two Collection Protocols

- **Structured**：预定义 task scope 和 reset protocol，每个 episode 是一个 named task 的 teleop demonstration。
- **Unstructured**：operator 在 deployment scene 自由移动机器人，没有 task scope、reset protocol、episode boundary。产生 long multi-event in-distribution motion stream。

**关键**：两者喂同一个 downstream pipeline——hierarchical caption schema 在 collection 后 segment raw stream 成 Task/Subtask/Action/Segment spans，clustering 对 segmented clips 一视同仁。Unstructured admission 是 scale 的关键：它去掉了 per-episode protocol overhead，让 collection throughput 突破 teleop demonstration 的 natural ceiling。

### 5.3 Temporal Synchronization（图 11）

[video+action] layers 只有在 visual observation 和 action stream 指向同一物理瞬间时才有用。但 camera encoding、controller logging、teleop middleware、disk writing 会引入 nearly constant video-action offset。**特别在 contact 附近**，几帧就能把 semantic state 从 "approaching" 变成 "touching"、"grasping"、"recovering"。

方法：对每个 episode，构造 visual motion signal（frame-to-frame image change across ego + wrist cameras）和 action-motion signal（left/right end-effector position 的 finite difference）。两者 smooth + normalize，sweep small integer lag window 找最大化 correlation 的 offset。这个 estimate 跨 cameras 和 action channels aggregate，防止 transient occlusion 或 stationary gripper dominate 决策。

### 5.4 Hierarchical Captioning（图 12）

每个 action-paired source episode 用 4-level temporal-nested caption hierarchy，spans 在 atomic manipulation actions 上 ground（reach、grasp、close、lift、place）：

- **Task (L3)**：episode-global string，总结 overall objective
- **Subtask (L2)**：把 episode 分成少数 contiguous semantically meaningful stages（approach target、establish grasp、transport、place/release）
- **Action (L1)**：每个 subtask 细分为 short manipulation primitives（reach、align gripper、close fingers、lift、translate、insert、retract）
- **Segment (L0)**：finest temporal decomposition，capture short localized events
- **Human (optional)**：manual annotation subset，用于验证自动 hierarchy

**对 recovery 行为的意义**：很多有用 demonstration 不是 perfectly linear task execution——含 regrasp、failed contact、小 pose correction、slip 后 retry。如果整个 episode 只有一个 caption，这些 corrective behaviors 被 average 进 global task description，model 难以识别。Hierarchical captioning 让这些 events 在时间上 localize 并获得自己的 description，dataloader 可以 sample 或 reweight 特定 temporal region，不把 successful episode 所有 frame 等权对待。

### 5.5 Cluster-Balanced Sampling

两轮 offline clustering：
- **Vision-Language clustering**：frozen multimodal encoder 把每个 (observation, caption) pair 映射到 joint embedding，clustering partition 成 topic clusters，summarize corpus 的 instruction-scene coverage。
- **Action clustering**：action chunks 在 trajectory space 单独聚类，long tail 集中 non-nominal motion（recoveries、re-grasps、retries、contact-driven corrections）。

**关键**：action-aligned decomposition 把 language-only 和 vision-language clustering views 变得更 evenly distributed，fewer samples 被少数 dominant topics 吸收。Training 时 dataloader 同时平衡 VL clusters 和 action clusters。

### 5.6 Recovery Data via Contact-Rich Random Initialization

公式 19 的 recovery mixture：
$$\tilde{p}_{\text{train}}(\mathbf{q}, e) = (1 - \alpha) p_{\text{nominal}}(\mathbf{q}, e) + \alpha \mathbb{E}_e[p(\mathbf{q} \mid e)]$$

对每个 contact event $e$，定义 local contact-pose distribution $p(\mathbf{q} \mid e)$，support 在以 nominal contact pose $\mathbf{q}_e^\star$ 为中心的小 geodesic ball $\mathcal{B}_\epsilon(\mathbf{q}_e^\star)$ 内。$\alpha$ 是小 mixing weight。**实践**：perturb robot initialization around $\mathbf{q}_e^\star$，然后 replay original demonstration 或 collect fresh recovery rollout。这创造了 controlled local coverage around each contact event，而非只观察穿过它的单一 nominal trajectory。

## 6. Training Recipe：Staged Pipeline

Table 1 给出 trainable/frozen matrix。5 个 stage：

| Component | Video PT | Action PT | VLM text | Staircase | Next-chunk |
|-----------|----------|-----------|----------|-----------|------------|
| 3D causal VAE | ✗ | ✗ | ✗ | ✗ | ✗ |
| T5 text encoder | ✗ | ✗ | ✗ | – | ✗ |
| VLM backbone (Qwen3.5-9B) | – | – | ✗ | ✗ | ✗ |
| VLM project-out / aux heads | – | – | ✓ | ✗ | ✗ |
| Video DiT (incl. view attention) | ✓ | ✗ | ✗ | ✗ | ✓ |
| Action DiT (incl. layer coupling) | – | ✓ | ✗ | ✗ | ✓ |
| Staircase MoT branches | – | – | – | ✓ | ✗ |

**Stage 1: Video Pretraining**。只训 video DiT，event latents 上 Wan-style $v$-prediction flow matching。Cross-view branch 的 output projector zero-init。Condition on current multi-view observation (one keyframe per camera) + event-aligned captions（frozen T5 encode）。Video timestep uniform sample。Event span truncate 到 65 latent frames（129 raw frames under stride-2）。Length-aware caption-drop schedule。Prune quasi-static frames 让 flow matching supervision 集中在 salient end-effector motion 的 segments。

**Stage 2: Action Pretraining**。Freeze video DiT。只训 action tower，event-centric pairs，disable video flow matching loss。Asymmetric 1-to-$N_d$ mapping，pin $s^\star = 45$ on 50-step schedule。$K = 6$ parallel action-noise draws per optimizer step 复用同一 anchored video forward。

**Stage 3: VLM Text-Conditioner Pretraining**。公式 21 的三项目标：
$$\mathcal{L}_{\text{text}} = \lambda_{\text{align}} \|\mathbf{c}_\ell^{\text{VLM}} - \mathbf{c}_\ell^{\text{T5}}\|^2 + \lambda_{\text{next}} \text{CE}(\hat{c}_{\text{next}}, c_{\text{next}}) + \lambda_{\text{time}} \text{Huber}(\widehat{\Delta t}, \Delta t)$$

第一项 align VLM conditioning features 到 original T5 encoding——**这让 upgraded VLM 成为 T5 的 drop-in replacement**，从 DiT 视角看 text conditioning geometry 不变。第二项是 next-event caption 的 token-level LM loss。第三项是 remaining-time scalar 的 Huber regression。VLM backbone 全程 frozen，只训 project-out head、next-event head、remaining-time regressor。

**Stage 4: Staircase Distillation**。只训 staircase reasoning branch 和 prefix projector，optimize latent-to-text reconstruction loss $\mathcal{L}_{\text{CoT}}$。

**Stage 5: Next-Chunk Adaptation**（optional）。Fine-tune event-pretrained backbone under next-chunk prediction on observation-centered layout。Global instructions 来自 Task level caption，frozen T5 encode。Both DiT towers 在 fixed-shape windows 上 update，asymmetric anchor protocol 保留。Cluster-balanced sampling 在 history-conditioned windows 上重跑 clustering passes。

## 7. 两种 Inference Mode

### 7.1 Event-Mode Inference

每个 rollout step：video 和 action DiT 条件化 current multi-view observation、proprioceptive state、next-event description。Description 来自 human 或 fine-tuned Qwen3.5-9B heads（也 emit remaining-time estimate）。Model denoise 整个 event-aligned video+action segment（scheme A 变长窗口）。Execution 完成后 observation advance，新 next-event description 条件化下一段。

### 7.2 Unified-Mode Inference

Rollout fixed $H_a$-step video-action chunks（scheme B layout）。每个 step 条件化 current observation + configurable history window + global instruction。**三个 interchangeable sources** 供应 per-chunk text-side context $\mathbf{c}_\ell$，可以 mid-rollout 切换不改变 denoiser state：

1. **Gradient-continuous source**：text encoder 把 global instruction 一次映射成 continuous $\mathbf{c}_\ell$，verbatim 喂 cross-attention，所有 rolling chunks 复用。No discrete CoT bottleneck。
2. **Atomic-instruction source**：upstream planner 或 human 每个 fixed $H_a$-step chunk boundary 提供一个 short instruction，text encoder 映射成 $\mathbf{c}_\ell$ 驱动对应 chunk。
3. **VLM-CoT source**：Staircase decoder 在单次 parallel forward emit CoT latents，representation 作为 per-chunk text-side context。

这个设计很漂亮：**同一个 event-pretrained backbone 同时支持 event-driven 变长 rollout 和 fixed-horizon VLA execution**，不需要两个不同的 model。

## 8. Infrastructure：Muon、Kernel Library、FP8

### 8.1 DMuon

Muon optimizer 的 Newton-Schulz iteration 在 vanilla 实现中，optimizer step 单独可能接近 2× forward+backward 成本。**DMuon** 是 Muon 的 distributed 实现，decoupled from training framework：

- **Pipeline Scheduling**：dedicated-ownership scheme，constrained Longest-Processing-Time assignment 把每个 matrix parameter map 到 unique owner rank。Native all-reduce/reduce-scatter 替换成 reduce/broadcast，post-step broadcast 在 dedicated CUDA stream 上异步 issue，与后续 forward pass overlap。Adaptive runtime monitor 在 network contention 时 fallback 到 sync broadcast。
- **Kernel Optimization**：Gramian formulation 有 symmetric structure，general-purpose GEMM 无法 exploit，近一半 tile-level computation redundant。用 CuteDSL kernel aligned to symmetric factor，shape-aware autotuning 跨 training 中遇到的 matrix geometries 维持 speedup。

### 8.2 Kernel Library

Stock kernels 在 single-operator granularity 操作，对 tensor-core throughput bound 的 patterns 不必要地 spill to global memory。识别 recurring fusible patterns，consolidate 成 composite kernels，让 intermediate activations 留在 registers 或 shared memory across fused region，shift effective roofline operating point toward compute-bound regime。

基于 **TVM FFI** 构建——language-agnostic foreign function interface，通过 stable C ABI 暴露 compiled kernels with zero-copy tensor handles，bypass per-call PyTorch dispatcher cost 和 GIL。Wrapper mirror native PyTorch op interface，所以 library kernels 可以 drop-in 替换 PyTorch counterparts 不改 model code。训练和推理用同一 compiled kernel 通过同一 call surface，maintain numerical consistency。

### 8.3 Fine-Grained Overlapping

Ulysses sequence parallelism 下每个 attention 需要一对 all-to-all。Naïve 实现串行化四个 all-to-alls per layer 在 critical path，substantially inflate communication cost。Fine-grained scheduling 把 back-to-back communication hide 在 attention computation 内，消除大部分 added communication overhead。

### 8.4 Multi-Event Sequence Packing

常见 recipe offline-encode 整个 episode 的 video latents 缓存。Minibatch 形成时需要 load/rematerialize 整个 episode 的 latents；episode 长度差异大，无法 without aggressive padding or truncation pack 成 uniform tensors，effective batch size 实际 collapse。

WALL-WM 在 dataloader level 把多个 events pack 进 single long sequence，训练在 packed sequence 上 parallel，attention mask block cross-event leakage。Events concatenate 到 fixed total length，每个 training step 总是跑 full configured effective batch size，GPU stays close to compute-bound，per-step cost amortize across all packed events 而非 per episode。

### 8.5 Model Compression

两条 orthogonal compression axes：

**Distillation**：Distribution-matching distillation (DMD) 训练 few-step student generator align output distribution with multi-step teacher。不是 pointwise regress teacher trajectories。**Joint distillation objective** 保留 original action-prediction loss alongside distributional term，所以 few-step student 同时 aligned with teacher video distribution 和 anchored to action supervision。Ablation 显示如果只做 distributional supervision，action MAE degrade 53%——compressing denoising trajectory under distributional supervision alone 让 action head drift away from pre-trained calibration。

**FP8 Quantization**：post-training quantization with per-block scaling。Weights 和 activations 沿 reduction dimension partition 成 fixed-size blocks，每块独立 scaling factor 吸收 local magnitude variation 同时 keep per-tensor metadata overhead negligible。**Weight-side**：offline quantize 成 pre-packed FP8 tensors with baked-in scaling factors，runtime path 无 weight-side quantization cost。**Activation-side**：on-the-fly activation quantization fuse 进 preceding operator 的 epilogue，per-block scale 和 FP8 cast 在同一 kernel 内完成——避免 separate read/write pass，quantization overhead 降到 overall GEMM time 的 negligible fraction。

结合 CUDA Graph capture eliminate host-side launch overhead，**full stack 让 end-to-end inference 到 10Hz**，meet closed-loop robotic control 的 latency budget。

## 9. 实验：四个 Real-Robot Suite + Embodied Video Generation

### 9.1 Embodied Video Generation（Table 2）

vs Wan2.1-1.3B 和 Wan2.2-5B：

| Model | Motion Quality | Semantic Consistency | Physical Plausibility |
|-------|----------------|----------------------|----------------------|
| Wan2.1-1.3B | 0.619 | 0.857 | 0.219 |
| Wan2.2-5B | 0.683 | 0.805 | 0.226 |
| **WALL-WM** | **0.771** | **0.886** | **0.434** |

WALL-WM 在 embodied-relevant dimensions 全面领先。最显著的是 **Physical Plausibility (0.434 vs 0.226)** 和 **Interaction Quality (0.434 vs 0.226)**——large-scale embodied training 把 inherited Wan video prior 转化为 robot-object interaction 和 contact evolution 的 physical prior。

Table 3 的 3D awareness benchmark on CO3Dv2：WALL-WM 在 point error (0.271)、depth error (0.132)、AUC@5 (0.210)、AUC@30 (0.727) 上和 WAN2.1-14B 持平或略好，complement strong multi-view consistency。

### 9.2 Real-Robot Four Suites（Table 5）

**Diverse Manipulation**（Arrange Cup Inverted Triangle、Put Spoon to Bowl、Pour Water 等 7 个 task）：
- WALL-WM-E: **75.86** avg
- WALL-WM-U-Scratch: 63.00
- π0.5: 55.64
- DreamZero: 39.97
- LingBot-VA: 29.71

**Reasoning Manipulation**（Sort Headphone、Press Button in Order、Pair Up Items 等 5 个 task）：
- WALL-WM-E: **71.60** avg
- WALL-WM-U-Scratch: 59.50
- π0.5: 56.40
- DreamZero: 32.70
- LingBot-VA: 31.60

**Dexterous Manipulation**（Insert Wireline、Put Stationery in Case）：
- WALL-WM-E: **32.00** avg
- WALL-WM-U-Scratch: 31.25
- DreamZero: 25.00
- LingBot-VA: 24.00
- π0.5: 15.00

**Generalization**（Place Plates、Cover Pot、Push Cloth、Insert Screwdriver）：
- WALL-WM-E: **53.75** avg
- DreamZero: 28.50
- π0.5: 24.00
- WALL-WM-U-Scratch: 18.50

### 9.3 Ablation（Table 4）

对比 pretrained event-mode WALL-WM vs pretrained fixed-length unified baseline without View-Interaction Self-Attention (VI-SA)：

- Reasoning Manipulation: 32.6 → **71.6** (+39.0)
- Generalization: 22.0 → **53.75** (+31.75)

Press Button in Order 从 0 → 64，Pair Up Items 从 8 → 36。这个 ablation 同时移除 VI-SA 和 event-conditioned execution，所以 measure 的是 combined effect。

## 10. 设计哲学的三轴分析（Appendix 9.1）

Paper 给了一个很有思考价值的设计空间分析，三个 axis：

**(i) Pixel-space vs. Latent-space prior**：pixel-space video foundation models 已经在 far broader 和 far thicker corpora 上 pretrained，weights 编码 strong visual world prior。Latent side intrinsic advantage 是 smoother、更 structured representation manifold，减少对 stochastic score-based generation 的依赖。对 video+action world model，pixel-space prior 的 shear breadth dominates。

**(ii) Native T2V vs. Native I2V**：这轴容易 miss 但 consequential。大部分 pixel-space video foundation models（包括 formally trained under I2V objective 的）建在 upstream native-T2V pretraining 之上。Native-I2V（V-JEPA-2、LeWorldModel）直接从 I2V mapping 开始。

- **Native-I2V** 更 tightly track visual signal 的 temporal evolution——即便给定 previous frame，visual continuation 在 every pixel densely supervised。这是 genuine strength。
- **Native-T2V** 带来不同 flavor 的 prior：text 行为像 high-dimensional visual futures manifold 上的 low-dimensional cluster-center label。T2V objective 隐式 ask network 先 commit to trajectory 的 semantic class，再 paint it。这是 **semantically anchored visual self-supervision delivered for free at internet scale**。

I2V removes that anchor。没有 up-front commitment，I2V loss 的 easiest minimum 是 conditioned on previous frame 的 high-bandwidth pixel extrapolation——closer to learned optical flow than world model。任何教 high-level physical regularities 的 gradient 只 very thinly flow through that route。WALL-WM 留在 T2V side，让 cluster-center prior paid for once 并 inherited downstream。

这个分析让我想到你自己说过的 "data > model"——T2V 的 anchor 本质上是用 text label 在 internet scale 上做的 visual self-supervision，是 "免费的高层结构监督"。

**(iii) Latent action vs. Dual-tower**：前面已经讨论过，dual-tower 是 latent action 的更 permissive 形式。

## 11. 我的一些联想和直觉

### 11.1 与 JEPA 家族的对比

Yann LeCun 的 V-JEPA / V-JEPA-2 / LeWorldModel 走的是 native-I2V + latent prediction in feature space 路线，避免 explicit pixel reconstruction。WALL-WM 的设计哲学正好相反：pixel-space + native-T2V + dual-tower。这不是谁对谁错，而是 **"你愿意花多少 upfront commitment 代价换多少 free semantic supervision"** 的 trade-off。JEPA 的 latent prediction 更 "efficient" 但失去了 T2V 的 cluster-center anchor；WALL-WM 保留 anchor 但接受 pixel-space 的更重 optimization landscape。

### 11.2 Event Unit 与 "Token" 类比

Action-grounded semantic event 在某种意义上是 **spatiotemporal token**——一个在 language、vision、action 三模态都有 well-defined 边界的单元。这让我联想到 LLM 中 token 的角色：token 是 language 的 atomic unit，但它的边界是 syntactic/semantic 的（BPE、SentencePiece）。WALL-WM 的 event 是 embodied domain 的 "semantic token"，边界由 executable behavior change 定义。未来的方向可能是 self-supervised event boundary discovery，类似 LLM 的 tokenizer learned from data。

### 11.3 Asymmetric 1-to-$N_d$ 的深层意义

这个设计解决的不只是 efficiency。它实际上重新定义了 video tower 在 action training 中的角色：**不是作为 co-denoising partner，而是作为 anchored visual evidence provider**。这有点像 "vision as world model prior for action" 而不是 "vision+action joint diffusion"。video tower 的 role 从 "co-target" 变成 "frozen context provider"。这在 concept 上很 clean——你 pretrained 了 visual prior，然后用它作为 action learning 的 anchored context，不让 action gradient 反过来 perturb 你好不容易学到的 visual structure。

### 11.4 Staircase Decoding 与 Parallel Speculative Decoding

Staircase 的 "lower layers 共享，upper layers 并行" 让我想到 speculative decoding 的 "draft model + verify" 模式。但 staircase 的本质是 **不同 reasoning step 共享 grounding computation 但 specialize reasoning computation**。这和 "speculative decoding 让小模型 propose 大模型 verify" 不太一样——staircase 是同一个 model 的不同 depth 处理不同 reasoning position。更接近的类比是 **DeepSeek-V3 的 MTP（multi-token prediction）** 或者你最近讨论的 "parallel prediction of multiple positions"。

### 11.5 对 Latency vs. Generalization 的观点

Paper Section 8 有一段我特别认同的话：**"if latency is enforced too early, it may cap the reachable performance ceiling before the model has learned a sufficiently general world-action prior"**。distillation、quantization、speculative execution、systems-level overlap 可以 progressively reduce latency once a strong model exists；recovering a lost generalization ceiling from an under-scaled model is much less straightforward。这和 Leskovec 关于 "在 representation learning 中 accuracy > efficiency at training time, efficiency at inference time" 的观点类似。

## 12. 局限和未来方向

Paper 自己也承认：

1. **当前 data construction recipe 仍依赖 large-scale temporal grounding 和 fine-grained captions 来 expose event structure before training**。未来方向是 self-supervised pretraining over vision/language/action，event boundaries 不由 dense annotation 提供，而是由 training objective 本身 capture。

2. **Recover lost generalization ceiling from under-scaled model is much less straightforward**——但 paper 没有给出明确的 scaling law 实验曲线，只在 Section 8 提到 "consistent trend across model sizes from below 10B to tens-of-billions regime"。

3. **Recovery data 仍依赖 task-specific contact pose definition**——对 general-purpose WAM，如何自动 discover "哪些 event 是 contact-rich 且需要 recovery augmentation" 仍 open。

4. **Unified mode 的 history window 长度需要 hand-tune**（$M$ history frames + 1 anchor + $N$ future frames），paper 没讨论这个超参的 sensitivity。

5. **VLM-CoT source 在 inference 时的 staircase decoder 仍是 separate forward pass**，和 denoiser 的 integration 通过 cross-attention，没有 fully unified 进 denoising 主干。未来可能直接把 reasoning latents 作为 denoiser 的 initial state 而非 cross-attention context。

## 13. 总结

WALL-WM 的核心贡献不是某个单点技术创新，而是 **一个完整的 prior-preserving scale-up methodology**：

1. **重新定义训练单元**：从 fixed-length chunk 到 action-grounded semantic event，carve at the joints of language/vision/action 三模态。
2. **Prior-preserving grafting**：Wan video prior 通过 zero-init projector、frozen video tower during action training、T5-aligned VLM conditioning 等机制被保留。
3. **Dual-tower layer-coupled architecture**：作为 latent action 的更 permissive 形式，让 shared/private capacity 比例 emergent 地学到。
4. **Multi-view geometric inductive bias**：sight-cone + tube mask 在 attention topology 和 input content 两个维度强制 cross-view geometric consistency。
5. **Staircase parallel latent CoT**：解决 long-horizon reasoning 的 efficiency 问题，同时保持 gradient-continuous VLA path。
6. **Muon-based infrastructure**：DMuon、custom kernel library、FP8 PTQ、multi-event packing、fine-grained overlapping——把 10Hz closed-loop control 拉到可及范围。

实验上，event-mode WALL-WM 在四个 real-robot suite 全面领先，ablation 确认 event + VI-SA 的 combined effect 在 reasoning (+39.0) 和 generalization (+31.75) 上都有 substantial gain。

如果让我一句话概括 paper 的精神：**"don't flatten three modalities into one window; instead, find the joint where they all agree, and carve there."**

---

参考链接：
- [WALL-WM GitHub](https://github.com/X-Square-Robot/wall-x)
- [Wan Video Generation Models](https://arxiv.org/abs/2503.20314)
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)
- [π0.5](https://arxiv.org/abs/2504.16054)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [RoPE / RoFormer](https://arxiv.org/abs/2104.09864)
- [Distribution Matching Distillation](https://arxiv.org/abs/2312.13837)
- [DROID Dataset](https://droid-dataset.github.io/)
- [Agibot World](https://arxiv.org/abs/2503.06669)
- [Ego4D](https://ego4d-data.org/)
- [EPIC-KITCHENS](https://epic-kitchens.github.io/2024)
- [OpenVID-1M](https://arxiv.org/abs/2502.15892)
- [JEPA / V-JEPA](https://arxiv.org/abs/2301.08243)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [CogVideoX](https://arxiv.org/abs/2408.06036)
- [Open-Sora 2.0](https://arxiv.org/abs/2503.09642)
- [Plato, Phaedrus 265e](https://www.perseus.tufts.edu/hopper/text?doc=Perseus%3Atext%3A1999.01.0174%3Atext%3DPhaedrus)
