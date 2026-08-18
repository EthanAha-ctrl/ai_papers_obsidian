---
source_pdf: EGOWAM World Action Models Beyond Pixels.pdf
paper_sha256: dcda12fc4e871509ff95847ed92e64cfc71aaf021c3bf17e14312259e2229443
processed_at: '2026-08-18T10:23:50-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说EGOWAM

好, 咱们抛开公式聊。

## 这群人想干嘛

Georgia Tech的Danfei Xu组, 他们面对的问题特别朴素: robot data太贵了, 每个task要几百条teleop, 换个scene又得重来。但是human video便宜啊, 你戴个眼镜(Eye glasses, Meta的Project Aria)做一遍task, 数据就到手了, 而且什么场景都有。

所以大家都在想: 能不能让robot学人的video?

**难在哪儿**: 你human拿杯子和robot gripper拿杯子, action长得完全不一样。人的手有21个joint, robot就俩finger; 人侧着抓, robot得正面夹。如果你直接把这些human action塞进BC的action decoder里, policy会被这些"人样"motion带跑 — robot根本执行不了。

更糟糕的是, human data里真正有价值的东西(objects, scenes, task semantics)被这些执行细节包裹着, 一起被action decoder给"消化不良"了。作者管这个叫BC的"bitter lesson" — 唯一通往policy的channel是action, 所以transferable和non-transferable全挤在一条管子里, 后者的gap把前者堵死了。

## 他们的核心idea

简单讲: **别让human data走action那条路, 走另一条路**。

具体说, 给policy加一个"world prediction head" — 让它除了predict action, 还predict"未来场景长什么样"。这个head只看observation, 不管你是人还是robot, 反正未来杯子被拿起来了, scene演化的样子是差不多的。

所以human data通过"未来场景怎么变"这个channel来塑造shared representation, 而action labels(那些不可执行的)就不再绑架整个policy。

这个idea本身V-JEPA 2、UniVAM、Cosmos Policy那波人都在搞, 不新。EGOWAM真正新的是下面这件事。

## 他们真正问的问题

**你应该predict未来场景的什么?**

这听起来怪 — 未来场景不就是future image吗? predict pixel不就完了?

他们argue: predict pixel是个trap。因为pixel encode了太多embodiment-specific的东西 — robot lab的光照、gripper的颜色、桌面的纹理、你头怎么转的 — 这些东西在human video和robot video里完全不一样。如果world head要重建pixel, trunk就得把这些乱七八糟的东西全encode了, 真正决定task的structure反被挤掉了。

所以他们提出三个判断标准, 什么样的future prediction能跨embodiment:

1. **Appearance abstraction**: 把外观抽象掉, 不reward pixel-level reconstruction
2. **Cross-embodiment consistency**: human和robot产生同样物理effect时, supervision应该一致
3. **Ego-motion factoring**: head转动和scene change要分开 — 人戴眼镜转头时, 静止背景在画面上跑得飞快; robot头是固定的, 同一个event的"画面运动"完全不同

然后他们试了三个target, 在同一个backbone、同一个action head、同一份data mixture下对比:

| Target | 满足几条 | 效果 |
|---|---|---|
| Pixel VAE | 0条 | 几乎没用, 甚至worse than BC |
| DINO features | D1, D2 | OOD泛化涨4倍 |
| 3D Flow | 全部 | ID performance涨20-30% |

DINO features是用DINOv2 extract的semantic patch features, 自动把appearance抽象掉了, 而且语义层面"杯子"在人和robot数据里长得差不多。但它还是image-coordinate indexed, head motion会leak进来 (D3半违反)。

3D flow更elegant — 它predict每个pixel在3D空间里怎么动, 而且用VIO pose把future position re-express到time t的camera frame, 这样head转动的效应被算术上factor out了。静止背景flow是零, 被操作物体保留真实motion。三个标准全部by construction满足。

## 最反直觉的实验结果

他们做了个ablation我特别喜欢: 让人故意用"robot做不出来的方式"做task(比如手抓gripper夹不住的地方), 看各种方法怎么样。

结果: BC直接崩到robot-only baseline以下 — 它学到了这些inexecutable motion然后robot执行不了。3D-flow WAM纹丝不动, 还是比robot-only高。

为什么? 因为3D flow根本不在乎action, 它只看"物体怎么动"。你手怎么抓不重要, 杯子从A点移到B点这个effect是清楚的, WAM学的是这个。

还有个细节: 把human data从"naturalunaligned"换成"刻意mimic robot viewpoint和motion", 看不同方法涨多少。

- BC: 从崩盘到涨起来 (符合预期, BC只能吃aligned data)
- Pixel: 35% → 65% (涨30点)
- DINO: 50% → 70% (涨20点)
- 3D Flow: 85% → 85% (一动不动!)

3D flow完全invariant, 因为它的target本身就是camera-stabilized的, head动不动都一样。这quantifies了ego-motion factoring这条标准有多重要 — 别人要靠data collection时manual align来达到的, 它直接通过target设计就白嫖了。

## Pixel-PT那个失败案例特别vivid

他们用pretrained VACE 1.3B做Pixel-PT, 在bag-grocery上反而比from-scratch还差。原因: pretrained video model有强prior"bags are typically open", 所以它会hallucinate一个already-open的bag, 然后policy看到bag已经开了, 直接跳过opening stage去pick物体, 当然失败。

这说明pretrained video prior是双刃剑 — 它带来"视频看起来真实"的prior, 但这个prior跟task state不一致时会actively harmful。有意思的是, 加EgoVerse human data co-training能修正这个hallucination — 数据量增加override了noisy prior。

## 几个我个人觉得特别sharp的intuition

**1. World target的选择比backbone size更重要**

整个VLA社区现在都obsess over "更大的video model + 更多data", 但这篇paper说: 在cross-embodiment场景下, 你predict什么决定你能transfer什么。Pixel prediction看起来information-rich, 实际是trap, 因为它把不可transfer的appearance entangle进去。

这跟你以前讲过的"inductive bias through loss function"是resonant的 — loss function的choice比architecture更能shape representation。

**2. DINO和3D Flow互补的原因很深**

DINO学到的是"这是什么"(semantic) — 什么是cup, 什么是saucer, 杯子要upright放, 所以泛化到unseen objects和scenes很强。

3D Flow学到的是"它在哪里、怎么动"(geometric) — 杯子从工作空间这个位置精确移到那个位置, 所以spatial precision和ID performance最强。

两者都满足三条desiderata, 但through完全不同的abstraction axis。理想world representation可能需要融合两者, 甚至加audio/tactile这种modality。

**3. "Action as context"这个细节**

Table 1显示加action labels会降低3D flow的prediction loss。这说明action和world prediction不是independent的 — action告诉trunk"demonstrator想干嘛", 这sharpen了world prediction。

这跟UniVAM、DiT4DiT那批joint prediction-action model的方向一致, 但EGOWAM把它放在cross-embodiment transfer的framework里reformulated了 — action不只是output, 它也是给world head的context。

**4. 推理时丢掉world head**

这点很重要。他们follow了Fast-WAM的发现: WAM的收益主要来自training-time representation shaping, 不来自test-time imagination。所以deploy时world head完全丢掉, 只unroll action head, 跟同等BC policy一样的latency (30 Hz)。

这意味着这篇paper是个research instrument, 不直接是production system — 它的finding可以deploy到任何BC policy上, 只要training时加个world head就行, inference完全free。

## Limitations他们自己承认的

- Motion generalization还做不到 — 比如从T-shirt model学折短裤, 需要更unified的action representation
- 一个policy per task, multi-task scaling没试
- "Best" world representation还是open question — DINO和3D Flow beat pixel只是起点

## 大的picture

把这篇paper放回landscape里:

- **BC era**: action-only supervision, 数据scale就饱和
- **Video pretrain era** (UniSim, Cosmos, Video2Action): pixel-level video pretraining再transfer到robot, scale了但entangle appearance
- **WAM era** (UniVAM, EGOWAM): joint action + world prediction, decouple supervision channel
- **EGOWAM的specific niche**: 在WAM里systematically问"world target应该长什么样"

最重要的takeaway一句话: **supervision signal的可transfer性决定representation的形状**。Loss function的choice比architecture更能shape你学到什么。

跟Yann LeCun的JEPA philosophy其实是一脉的 — predict in abstract space而非pixel space。但EGOWAM在robotics的cross-embodiment setting下用controlled study强力validate了这个idea, 还加了ego-motion factoring这条JEPA没显式处理的标准。

## Further reading

- 项目主页: https://gatech-rl2.github.io/egowam.github.io/
- EgoVerse dataset (他们human data的来源): https://arxiv.org/abs/2604.07607
- HPT backbone: https://arxiv.org/abs/2409.20537
- DINOv2: https://arxiv.org/abs/2304.07193
- RAE (Representation Autoencoder, DINO head的理论基础): https://arxiv.org/abs/2510.11690
- Track4World (3D flow target的来源): https://arxiv.org/abs/2603.02573
- Project Aria (Meta的眼镜): https://arxiv.org/abs/2308.13561
- Conditional Flow Matching (action head的objective): https://arxiv.org/abs/2305.17143
- V-JEPA 2 (JEPA philosophy的video版): https://arxiv.org/abs/2506.09985
- Fast-WAM (影响action-only inference design): https://arxiv.org/abs/2603.16666
- UniVAM (related WAM work): https://arxiv.org/abs/2503.00200
- DINO-WM (之前在pretrained visual features上做WM): https://arxiv.org/abs/2411.04983

---

一句话总结: 这篇paper告诉你, 想让robot从human video学到东西, 别逼它学人的action, 也别让它画未来frame; 让它predict semantic features或者3D motion, 跨embodiment transfer就work了。听起来简单, 但要build这个intuition需要绕很多弯, 这篇paper的价值是把弯路走完了, 留下了一条clean的设计axis。

---

# EGOWAM 深度解读: World Action Models Beyond Pixels

Andrej, 这篇paper触及了当前robot learning里最核心的tension之一 — human video数据scalable但embodiment gap巨大, 而 robot数据干净但expensive。作者(Danfei Xu group at Gatech RL2)的核心wager是: 不要把human data硬塞进action decoder, 而是开一个"world prediction"的auxiliary channel让human data塑造shared representation。这个idea本身在V-JEPA 2、UniSim、Cosmos Policy那一脉已经很热, 但这篇paper的真正贡献是**对"world target应该是什么"做了controlled study**, 并得出了反直觉但合理的结论。

---

## 1. The Core Problem: BC的Bitter Lesson

让我先建立intuition。考虑human egocentric data $\mathcal{D}_H = \{(o_t^H, a_t^H)\}$ 和 robot data $\mathcal{D}_R = \{(o_t^R, a_t^R)\}$。BC co-training的做法是把两者retarget到shared action space, 然后joint train一个policy:

$$\mathcal{L}_{\text{BC-cotrain}}(\phi, \theta) = \sum_{\mathcal{D} \in \{\mathcal{D}_H, \mathcal{D}_R\}} \mathbb{E}_{(o,a) \sim \mathcal{D}} \mathcal{L}_{\text{BC}}(\pi_\theta(a | f_\phi(o)), a)$$

这里 $f_\phi$ 是shared encoder, $\pi_\theta$ 是shared action decoder, $o$ 是observation, $a$ 是action chunk。

作者的关键观察: **shared action decoder是human data到达policy的唯一通道**。如果human hand侧着拿杯子而robot gripper必须从前面夹, action label本身就把这种不可执行的motion inject进去了。这条channel上, transferable content (物体、scene、semantics) 被迫和non-transferable execution (morphology、style) 绑在一起, 后者的gap会block前者的transfer。

这跟你在Eureka Labs讲过的"distribution shift在BC里是killer"是同一个味道, 只是这里shift来自embodiment而不是dynamics。

---

## 2. WAM: 开第二个Supervision Channel

WAM的核心move是augment BC policy加一个auxiliary future-prediction head:

$$p_{\theta,\psi}(a_{t:t+k}, s_{t+T} | o_t) = p_\psi(s_{t+T} | z_t) \, p_\theta(a_{t:t+k} | z_t), \quad z_t = f_\phi(o_t)$$

变量含义:
- $z_t$: shared trunk的latent embedding, 是observation $o_t$ 经过encoder $f_\phi$ 后的representation
- $a_{t:t+k}$: action chunk, 从t到t+k的action序列, $k$ 是resampled chunk length (这里是100)
- $s_{t+T}$: future state at horizon $T$, $T$ 是原始时间horizon (robot 1.5s, human 1s)
- $\theta, \psi, \phi$: 分别是action head, world head, shared trunk的参数

Joint loss:

$$\mathcal{L}_{\text{WAM}} = \mathcal{L}_{\text{action}}(a_{t:t+k}) + \lambda \mathcal{L}_{\text{world}}(s_{t+T})$$

$\lambda = 1$ in practice。Key property是 $\mathcal{L}_{\text{world}}$ 通过"未来场景如何演化"监督shared trunk, 而这条channel对morphology和behavioral style基本indifferent — human hand和robot gripper产生相似物理变化时, 未来scene长得差不多。

**Crucial design choice**: 推理时把world head丢掉, 只unroll action head。这是follow了Fast-WAM (Yuan et al. 2026)和UniVAM的发现 — WAM的收益主要来自training-time representation shaping, 不来自test-time imagination。所以deployment footprint和同等大小的BC policy一样, 30 Hz。

---

## 3. 三个Desiderata: 什么样的World Representation能跨embodiment?

这是这篇paper最sharp的部分。作者提出三个判断标准:

### D1: Appearance Abstraction
Photometric reconstruction会把embodiment-specific的外观细节encode进trunk, 挤掉真正决定task outcome的structure。比如robot lab的光照、gripper颜色、桌面纹理, 这些在human data里完全不一样, 如果world target是pixel, trunk就会被迫encode这些。

### D2: Cross-Embodiment Consistency
同样的物理effect(杯子被拿起), 不论是human hand还是robot gripper产生的, supervision应该一致。如果target是"agent长什么样", transfer就死了; 如果target是"effect是什么", transfer才有可能。

### D3: Ego-Motion Factoring
Image-coordinate target会把head rotation和scene change搅在一起。Human戴Project Aria走动转头时, 静止的background在image plane上产生巨大apparent motion; Robot的head camera是static的, 同一event的supervision完全不同。这条对egocentric数据特别critical。

---

## 4. 三个World Target的Instantiation

### 4.1 Pixel VAE (Reconstruction Baseline)

$$s = \text{VAE}(I_{t+T}^{\text{ego}})$$

用frozen Wan video VAE encode未来的ego frame, latent shape是 $16 \times 16 \times 16$ (C×H×W), 输入分辨率128。Head是DiT (diffusion transformer), 有两个version:
- **Pixel**: from scratch, 6 blocks, hidden 384, 6 heads, patchified at stride 2 (8×8 = 64 tokens)
- **Pixel-PT**: 用VACE-1.3B pretrained weights初始化, 30 layers, hidden 1536, 12 heads, FFN 8960

Loss是标准noise prediction:

$$\mathcal{L}_{\text{world}}^{\text{VAE}} = \mathbb{E}\|\epsilon - \epsilon_\psi(s^\tau, \tau, f_\phi(o))\|^2$$

其中 $s^\tau = (1-\tau)\epsilon + \tau s$, $\tau \in [0,1]$, $\epsilon \sim \mathcal{N}(0, I)$ — 这是flow matching的线性interpolation path。

**违反所有三个desiderata**, 是strawman baseline。

### 4.2 DINO Features (Semantic Abstraction)

$$s = \text{DINO}(I_{t+T}^{\text{ego}})$$

用frozen DINOv2-B提取patch features, 得到 $16 \times 16$ token grid in $\mathbb{R}^{768}$。Head用RAE (Representation Autoencoder, Zheng et al. 2025)的 $\text{DiT}^{\text{DH}}$ 设计 — DiT backbone + shallow但wide的DDT-style head:

- DiT backbone: 6 blocks, 384-d, 6 heads
- Wide head: 2 blocks, **2048-d** — 这里width必须 ≥ token dim (768)才能让flow matching收敛, 这是RAE论文里的理论结果
- Conditioned on mean-pooled trunk embedding
- 50 fixed Euler steps sampling

Loss一样:
$$\mathcal{L}_{\text{world}}^{\text{RAE}} = \mathbb{E}\|\epsilon - \epsilon_\psi(s^\tau, \tau, f_\phi(o))\|^2$$

**满足D1 (semantic prior抽象掉appearance)和D2 (semantic feature对agent无关), 但部分违反D3** — DINO features仍然是image-coordinate indexed, head ego-motion还是会leak进来。

### 4.3 3D Flow (Spatial Abstraction) — 最geometric grounded的

这是最elegant的部分。Target是dense 3D motion field over $[t, t+T]$, 在time t的camera-stabilized frame里表达:

$$s = F_{[t, t+T]} = \tilde{X}_{t+T} - X_t$$

其中:
- $X_t, X_{t+T}$: 3D point positions, 由pretrained 3D point tracker (Track4World / SpatialTrackerV2) 给出
- $\tilde{X}_{t+T} = (T_t^{\text{cam}})^{-1} T_{t+T}^{\text{cam}} X_{t+T}$: 把future position re-express到time t的camera frame

$T_t^{\text{cam}}$ 是time t时camera在世界坐标系下的pose (来自Aria VIO)。这个stabilization之后, 静止background产生near-zero flow, 而被操作物体保留与physical displacement成比例的motion。

Head是flow matching decoder, 4 blocks, 256-d, 4 heads, condition on anchor positions $q$ (28×40 = 1120点, no subsampling):

$$\mathcal{L}_{\text{world}}^{\text{Flow}} = \mathbb{E}\|u_\psi(s^\tau, \tau, f_\phi(o), q) - (s_q - \epsilon_q)\|^2$$

这里 $u_\psi$ 是velocity prediction network, $q$ 是从current ego frame均匀采样的query points, $s_q$ 和 $\epsilon_q$ 是这些query点对应的target flow和noise。

Target shape: $100 \times 1120 \times 3$ (100-step horizon, 1120 points, 3D xyz)。

**满足所有三个desiderata by construction**:
- D1: 3D motion天然abstract掉appearance
- D2: 几何effect与agent无关
- D3: camera frame stabilization把head motion显式factor out

---

## 5. Architecture: HPT Backbone

EGOWAM build on Heterogeneous Pretrained Transformer (Wang et al. 2024, HPT):

- **Trunk**: 256-d, 16 blocks, 8 heads, stochastic depth 0.1, with learned domain embeddings
- **Tokenizers**: 每个embodiment有shallow stem
  - Ego-vision stem: ResNet-18 (scratch) 或 DINO pretrained, output 256-d, 通过learned query attention (16 latent queries, 8 heads, head dim 64)
  - Wrist-vision stem: 同上, only for robot batches
  - Proprioception stem: per-embodiment MLP (14 → 256)
- **Learnable tokens**: 64 action tokens + 16 future tokens (BC only用action tokens)
- **Observation history**: 1 frame

**Action head** — flow matching based:
- 6-block CrossTransformer
- Hidden width 128, 4 heads
- Conditional flow matching with $\tau \sim \text{Beta}(1.5, 1.0)$
- $a_{t:t+k}^\tau = (1-\tau)\epsilon + \tau a_{t:t+k}$, $\tau$ embedded along hidden dim
- Alternating self- and cross-attention blocks denoise tokens while injecting trunk context
- v-prediction, 50 sampling steps
- Action dim $d_a = 14$ (per-arm 6-DoF SE(3) + 1-D gripper)
- Chunk length $k = 100$

---

## 6. Cross-Embodiment Action Alignment (Sec 3.1)

为了让BC是strong baseline而不是strawman, 作者做了仔细的action alignment:

### 6.1 Unified Action Space
14-D: per-arm 6-DoF SE(3) pose + 1-D gripper。

### 6.2 Egomotion Factoring for Human Actions
Human hand poses $p_{t+i}^H \in \text{SE}(3)$ 是在moving device frame里的, transform是 $T_t^{\text{device}}$。Re-express到time t的instantaneous device frame:

$$a_{t:t+k}^H = [(T_t^{\text{device}})^{-1} T_{t+i}^{\text{device}} p_{t+i}^H]_{i=1}^k$$

这里 $(T_t^{\text{device}})^{-1}$ 把later time的device frame"拉回"到time t的frame, 这样同样的physical motion产生comparable numerical trajectory, 把head egomotion factor out。这跟3D flow的stabilization思想是mirror的。

### 6.3 Speed Alignment
Human动作快于teleoperated robot, 用embodiment-specific windows:
- $T_H = 1.0$s (30 frames)
- $T_R = 1.5$s (45 frames)
- Both discretized into $k=100$ steps — semantic aligned trajectories

### 6.4 Quantile Normalization
Map 1st和99th percentiles to $[-1, 1]$, robust to hand tracking outliers。

---

## 7. Joint Training

$$\mathcal{L}_{\text{EGOWAM}} = \underbrace{\mathcal{L}_{\text{action}}^{\text{robot}} + \mathcal{L}_{\text{action}}^{\text{human}}}_{\mathcal{L}_{\text{action}}} + \lambda \underbrace{(\mathcal{L}_{\text{world}}^{\text{robot}} + \mathcal{L}_{\text{world}}^{\text{human}})}_{\mathcal{L}_{\text{world}}}$$

每个step draw 32 robot + 32 human samples, 都过shared tokenizers和trunk, 然后action head给 $\mathcal{L}_{\text{action}}$, world head给 $\mathcal{L}_{\text{world}}$, 两者都supervise shared trunk。

**Key intuition**: 当human action labels transfer弱的时候, world prediction可以compensate并shape trunk representation。Human数据通过"未来场景怎么演化"这个channel塑造shared representation, 即使action labels不可信。

Optimizer: AdamW (lr $1 \times 10^{-4}$, weight decay $1 \times 10^{-4}$), cosine annealing ($T_{\max} = 1400$, $\eta_{\min} = 1 \times 10^{-5}$), bf16, ~2 days/task/method on L40S。

---

## 8. 实验数据深度分析

### 8.1 Setup
- **Robot platform**: 两台upright-mounted 6-DoF ARX5 arms + parallel-jaw grippers + head-mounted Project Aria + 2 wrist Intel RealSense D405
- **Tasks** (三个bimanual tasks from EgoVerse):
  - **cup-on-saucer**: 把随机朝向的杯子reorient并upright放在saucer上, 需要precise bimanual regrasping
  - **fold-clothes**: 三折T-shirt, deformable
  - **bag-grocery**: 打开购物袋并放3个item进去, long-horizon
- **Robot data**: 300-360 demos/task
- **Human data**:
  - In-Domain: 1:1比例, same scene/objects但unmatched viewpoint/behavior, 2h/task
  - EgoVerse: ~10:1比例, full EgoVerse-A flagship split, diverse scenes/objects/demonstrators, 7-21h/task
- **Evaluation**: 1800个real-world rollouts total, ID 20 + OOD 20 (10 unseen objects + 10 seen objects in novel scenes) per method per task

### 8.2 Main Results (Figure 3)

**Q1: WAM vs BC co-training**

WAM co-training consistently outperforms BC, 而且human data往往会**degrade** BC但给WAM带来gains。这正验证了bitter lesson — BC的action-only channel无法消化unaligned human data, 反而被它污染。Figure 5的qualitative comparison很直观:
- BC on fold-clothes仍然overfit到robot data, 无法适应novel scene with lower table
- BC on cup-on-saucer产生human-like motion (侧着拿), robot无法执行
- Pixel on bag-grocery hallucinates already-open bag, 跳过opening stage (Fig 12的failure analysis很精彩)

**Q2: 哪个world representation最好?**

| Representation | ID gains | OOD gains | 主要strength |
|---|---|---|---|
| Pixel VAE | weak | weak | 几乎没有 |
| DINO | 中等 | **up to 4×** | object/scene generalization |
| 3D Flow | **20-30%** | 中等 | spatial precision, ID performance |

互补的strengths:
- **DINO**: semantic prior让trunk学到object-和scene-invariant representation, OOD unseen objects泛化最强。Figure 7显示DINO能从human data refine object shape, 但只moderately。
- **3D Flow**: geometric grounding让spatial precision最高, cup-on-saucer across workspace都成功 (Figure 6)。Figure 7显示3D flow的gains远超其他 — 它能recover object motion that robot-only prediction leaves static。

**Q3: Ablation on action-misaligned human data**

在bag-grocery上故意让human用inexecutable方式拿物体 (Figure 8 right), BC直接collapse到robot-only baseline以下, 而3D-flow WAM保持robust并且仍超过robot-only。

---

## 9. 烧灼的细节: Ablation Studies

### 9.1 Aligned Human Data Ablation (Appendix A.1)

在cup-on-saucer上对比natural in-domain vs deliberately robot-aligned human data (Figure 10c):
- **BC**: aligned human data让它从negative transfer变成positive transfer (确认BC只在action distribution hand-curated时受益)
- **Pixel**: 35% → 65% (35-point jump!)
- **DINO**: 50% → 70% (20-point jump!)
- **3D Flow**: 85% → 85% (完全invariant!)

这quantifies了D3 violation的代价 — Pixel和DINO都是image-coordinate target, 当ego-motion在collection time被manual factor out时大幅提升; 3D Flow因为camera-stabilized target by construction就factor out了ego-motion, 所以manual alignment对它没用。

### 9.2 Modality Ablation (Appendix A.2)

在human batches上比较:
- Action only (BC baseline)
- 3D Flow only (no action supervision on human)
- Action + 3D Flow (full EGOWAM)

发现:
1. **3D-flow-only > action-only** across all splits, 尤其OOD Scene上action-only直接0%而3D-flow-only还有10%
2. **Joint training wins everywhere** — 互补效应:
   - Action as context: action labels condition trunk on demonstrator intent, sharpening world prediction (Table 1: 加action labels降低3D flow prediction loss)
   - Action as task-relevance signal: 标记哪些motion是task-relevant的, 让trunk focus on agent-caused dynamics

Table 1: 3D-flow world-model prediction loss对比:

| Flow stream | Flow-Only | Action + Flow |
|---|---|---|
| Human | 0.23 | 0.22 |
| Robot | 0.20 | 0.19 |

### 9.3 Pixel-PT Failure Analysis (Appendix A.3)

特别有意思。Pixel-PT (pretrained VACE 1.3B) 在bag-grocery上**underperform** Pixel (from scratch) 和BC。原因是pretrained video model有强prior — "bags are typically open", 会hallucinate一个already-open bag before gripper act on handles (Figure 12)。这导致policy跳过opening stage。

但Pixel-PT + EgoVerse (co-training with human data)能修正这种hallucination — 因为data量增加, override了noisy prior。这是human data作为prior correction mechanism的一个interesting case。

---

## 10. RoboTwin Simulation Validation (Appendix D)

为了reproducibility, 在RoboTwin 2.0 (SAPIEN)上replicate。用bimanual aloha-agilex作primary embodiment, arx-x5/franka/ur5作co-training embodiments。所有robots统一到14-D EE action space in head-camera frame。

Table 4结果 (100 seeds):

| Task | ACT-EE | DP-EE | BC s/c | Pixel s/c | DINO s/c | 3D Flow s/c |
|---|---|---|---|---|---|---|
| pick-diverse-bottles | 2 | 5 | 2/6 | 7/11 | 4/**28** | 0/16 |
| stack-bowls-three† | 0 | 0 | 0/0 | 0/0 | 0/8 | 0/**16** |
| hanging-mug | 0 | 0 | 0/0 | 0/1 | 0/0 | 0/0 |

关键发现:
1. **Cross-embodiment > single** — 每个variant从single到cross都有提升
2. **World head, not action space, drives transfer** — ACT-EE/DP-EE用same action space但no world head, 跟single-embodiment variants一样差
3. **DINO和3D Flow擅长object generalization** — pick和stack都stress这个, appearance-abstracting targets胜出
4. **stack-bowls-three的appearance-shift test** — bowl材质变了但pose/geometry没变, 只有DINO和3D Flow survive, BC和Pixel collapse to 0%
5. **hanging-mug所有方法都≤1%** — 这是millimeter-precise insertion, 瓶颈是manipulation precision不是world representation, appearance abstraction救不了sub-centimeter accuracy

---

## 11. Limitations and Open Questions

作者很诚实地列出三条:
1. **Motion generalization**: gains局限在context level, 学新motion primitive (比如从T-shirt model学折短裤)还不行, 需要更unified的action representation
2. **Multi-task scaling**: 一个policy per task, multi-task co-training on large-scale in-the-wild human data没探索
3. **Open world representation**: DINO和3D Flow beat pixels只是起点, "best" world representation in scaling robot learning还是open question

---

## 12. 我(Intuition)的几个Reflections

### 12.1 "World representation is the next critical axis"
这是这篇paper最punchy的claim。整个VLA/WAM社区目前obsess overdata scale和backbone size, 但这篇paper指出 — 在cross-embodiment transfer场景下, **你predict什么比你怎么predict更重要**。Pixel prediction看起来"rich"但实际是trap, 因为它把不可transfer的appearance entangle进去了。

### 12.2 Why DINO and 3D Flow互补?
- DINO是**object-centric semantic** — 学到"什么是cup, 什么是saucer, 杯子要upright放", 泛化到unseen objects/scenes
- 3D Flow是**agent-centric geometric** — 学到"杯子要从这里移到那里, 精确spatial relationship", 泛化到precise placement across workspace

DINO对appearance abstraction强, 但spatial precision弱 (image grid indexed); 3D Flow对geometric effect强, 但semantic abstraction弱 (raw 3D points没semantic)。两者都满足D1-D3, 但through different abstraction axes。理想world representation可能需要**两者融合**。

### 12.3 Ego-motion factoring是被低估的insight
D3这条desideratum在egocentric video研究里其实很早就有人意识到 (Ego-Exo4D, EgoBridge都有相关discussion), 但在WAM context下explicitly提出来作为design criterion, 我是第一次见。3D Flow的camera-stabilization trick其实很简单 — 用VIO pose把future positions re-express到time t的camera frame — 但效果惊人: 3D Flow WAM对aligned vs unaligned human data完全invariant (85%→85%), 而Pixel和DINO都提升20-35 points。

### 12.4 与V-JEPA 2 / JEPA的connection
Yann LeCun的JEPA philosophy — predict in latent/abstract space而不是pixel space — 这篇paper用controlled study强力validate了这个idea在robotics的应用。DINO feature prediction本质上是"predict in pretrained semantic latent space", 跟V-JEPA 2 (Assran et al. 2025)的joint-embedding predictive architecture思路一致。但EGOWAM更进一步 — 用3D geometric flow作为更grounded的target, 满足JEPA没显式处理的ego-motion factoring。

### 12.5 Action head只用flow matching不用diffusion的原因
作者用conditional flow matching (Lipman et al. 2023), 不是标准diffusion。Flow matching的好处是path可以任意选, 这里用linear path $s^\tau = (1-\tau)\epsilon + \tau s$。Beta(1.5, 1.0) prior让 $\tau$ 偏向大值, 也就是less noise — 这对action prediction有意义, 因为action本身是deterministic的, 不像image有真正的高频stochasticity。

### 12.6 Pixel-PT failure的meta-lesson
Pretrained video model的prior既blessing又curse。Bag-grocery的例子很vivid — "bag通常是open的"这个prior会让model hallucinate open bag, policy直接跳过opening stage。这暗示未来video foundation model用于robotics时, 需要**task-conditional或scene-state-conditional prior shaping**, 而不是直接拿off-the-shelf。Human co-training作为prior correction的mechanism是个intriguing方向。

### 12.7 跟UniVAM / Cosmos Policy / DreamDojo的区别
那几篇都把video model作为spatio-temporal representation的核心, 但都默认pixel-level latent。EGOWAM的controlled study实际上**质疑了那个default**。如果Video Foundation Model + Robotics要scale, world representation choice可能比video model size更重要。

### 12.8 关于"Action as context"的subtle insight
Table 1显示加action labels降低3D flow prediction loss。这暗示action和world prediction不是independent的 — action labels告诉trunk"demonstrator的intent是什么", 这sharpen了world prediction。这跟recent work on joint prediction-action models (UniVAM, DiT4DiT)的方向一致, 但EGOWAM把它放在cross-embodiment transfer的framework里reformulated了。

---

## 13. References & Further Reading

- **Project page**: https://gatech-rl2.github.io/egowam.github.io/
- **arXiv (推测)**: paper里没直接给arxiv ID, 但作者 affiliation是Gatech RL2
- **EgoVerse dataset** (Punamiya et al. 2026): https://arxiv.org/abs/2604.07607 — 大规模egocentric human data, ~10:1 ratio with robot data
- **HPT backbone** (Wang et al. 2024): https://arxiv.org/abs/2409.20537 — Heterogeneous Pretrained Transformer
- **DINOv2** (Oquab et al. 2023): https://arxiv.org/abs/2304.07193 — semantic features的source
- **RAE / DiT-DH** (Zheng et al. 2025): https://arxiv.org/abs/2510.11690 — Representation Autoencoder, 理论分析了为什么semantic latent space需要wide denoiser
- **SpatialTrackerV2** (Xiao et al. 2025): https://arxiv.org/abs/2507.12462 — 3D point tracker
- **Track4World** (Lu et al. 2026): https://arxiv.org/abs/2603.02573 — feedforward world-centric dense 3D tracking
- **Project Aria** (Engel et al. 2023): https://arxiv.org/abs/2308.13561 — Meta的egocentric multi-modal glasses
- **Conditional Flow Matching** (Lipman et al. 2023): https://arxiv.org/abs/2305.17143 — action head的training objective
- **V-JEPA 2** (Assran et al. 2025): https://arxiv.org/abs/2506.09985 — JEPA philosophy的video extension, 相关philosophical inspiration
- **Fast-WAM** (Yuan et al. 2026): https://arxiv.org/abs/2603.16666 — "Do WAMs need test-time future imagination?" — 影响了EGOWAM的action-only inference design
- **UniVAM** (Li et al. 2025): https://arxiv.org/abs/2503.00200 — Unified Video Action Model, related WAM work
- **Cosmos Policy** (Kim et al. 2026): https://arxiv.org/abs/2601.16163 — NVIDIA的video model for visuomotor control
- **EgoDex** (Hoque et al. 2026): https://arxiv.org/abs/2505.11709 — dexterous manipulation from egocentric video
- **Diffusion Policy** (Chi et al. 2023): https://arxiv.org/abs/2303.04137 — bimanual manipulation的经典baseline
- **DINO-WM** (Zhou et al. 2025): https://arxiv.org/abs/2411.04983 — 之前就在pretrained visual features上做world models for planning, 但没在cross-embodiment context
- **RoboTwin 2.0** (Chen et al. 2025): https://arxiv.org/abs/2506.18088 — simulation benchmark with multiple bimanual embodiments
- **VACE** (Jiang et al. 2025): https://arxiv.org/abs/2503.20314 — All-in-one video creation/editing, pretrained Pixel-PT head的source
- **Evaluating BC generalization** (Vincent et al. 2024): https://arxiv.org/abs/2405.15566 — statistical approach to trustworthy performance evaluation, 用在confidence interval computation

---

## 14. 总结性的intuition

把EGOWAM放在更大的landscape看:
- **Pre-video-era robot learning**: action prediction only, 受限于data scale
- **Video-1.0 era** (UniSim, Cosmos, Video2Action): pixel-level video pretraining → robot, scale但entangle appearance
- **WAM era** (UniVAM, EGOWAM): joint action + world prediction, decouple supervision
- **EGOWAM的niche**: 在WAM framework内systematically问"world target应该长什么样", 找出pixel是trap, semantic (DINO)和geometric (3D flow)各有所长

最important的intuition — **supervision signal的可transfer性决定representation的形状**。如果world target是pixel, trunk就要encode所有pixel-reconstruction需要的细节, 包括不可transfer的appearance; 如果world target是3D flow, trunk只encode几何dynamic, 自动cross-embodiment。

这跟你在"Neural Networks: Zero to Hero"里讲过的"inductive bias through loss function"思想是resonant的 — loss function的choice比architecture更能shape学到的representation。EGOWAM把这个principle apply到cross-embodiment transfer这个具体problem上, 用controlled studyvalidate了 — appearance-abstracting loss > pixel-reconstruction loss for transfer。

Open question我特别curious的:
1. 多个world targets融合 (DINO + 3D flow + audio? + tactile?)能比单独的好多少
2. 把world prediction head换成JEPA-style non-parametric predictor会怎样
3. 在large-scale (>1000h) human data上, 三个representations的scaling law分别是什么形状
4. World target对long-horizon task和deformable manipulation (cloth, bag)的影响是否systematically不同
5. 如果把3D flow换成更dense的4D representation (NeRF, 4D Gaussians), 能不能capture更多deformable dynamics

希望这些技术细节和intuition对你的思考有帮助。这篇paper虽然实验只在三个task上, 但它的controlled study framework + 三个desiderata是真正能transfer到其他lab其他setting的contribution — 它提供了一个concrete design axis for "use human data"这个抽象imperative。
