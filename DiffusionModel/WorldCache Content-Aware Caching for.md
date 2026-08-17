---
source_pdf: WorldCache Content-Aware Caching for.pdf
paper_sha256: 1c911e464cf80ea786330a0a4b41a82ea441c2c694668c54de94d901385ff7b0
processed_at: '2026-08-13T05:38:33-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# WorldCache 人话版

Andrej，我用最朴素的话讲一遍。

## 问题是什么

视频 world model 生成太慢。慢在哪？要生成一段视频，得跑很多"去噪步骤"（denoising steps），每一步都要跑整个 transformer 网络，算很多层的 attention。这个计算量是 interactive 应用的瓶颈。

## 之前的偷懒方法

聪明的 idea 是：**相邻的去噪步骤之间，中间结果变化不大，那我就别每步都从头算了，上次算过的结果直接拿来用**。

这就像你写作业，发现第 5 题跟第 4 题差不多，就抄第 4 题的答案改改交了。这就是 DeepCache、FasterCache、DiCache 这些方法干的事。

**问题在哪？** 它们抄答案是"原样照搬"——上次算出来是什么样，这次就用什么样。这在静态场景没问题（背景不变嘛），但场景里有运动的时候——比如车在开、人在走、机械臂在动——你抄过来的就是"过时的"信息，画面就会出现鬼影、模糊、动作不连贯。

## WorldCache 的核心洞察

Paper 说：之前那些方法本质上做了一个很简单的假设——**"如果变化小，就直接冻结，保持上一时刻的值不变"**。这在信号处理里叫 Zero-Order Hold（零阶保持），就是把连续信号离散化时直接"保持住"上一个采样值。

这个假设对"静态信号"没问题，但对"动态信号"会出 aliasing——就像你用很低的帧率拍快速运动的物体，画面会抖、会撕裂。World model 的 rollout 恰好就是高频动态场景，所以之前的方法在这里 fail 得最严重。

## WorldCache 怎么解决

四个模块，每个解决一个具体的"偷懒误区"：

### 误区 1：不知道现在动得快不快

之前的方法用一个固定的"偷懒标准"——变化小于某个阈值就抄。但场景有时动得快、有时动得慢，固定标准不合理。

**WorldCache 的做法**：先看输入本身在过去两步变化了多少，变化大说明动得快，就收紧偷懒标准（老实算）；变化小说明动得慢，就放宽标准（放心抄）。

*类比*：开车时遇到复杂路况就全神贯注，开高速直路就稍微放松——根据路况动态调整注意力。

### 误区 2：不区分"哪里在变"

之前的方法算"变化量"是把整个画面平均一下。但画面里背景飘一点点，跟前景里人突然动了，平均下来可能差不多——前者该抄，后者不该抄。

**WorldCache 的做法**：给画面不同区域打"重要性分"。怎么看重要不重要？看 transformer feature 在 channel 维度的 variance——variance 高的地方是边缘、纹理、物体边界这种"信息密集"的区域。重要区域的变化权重高，背景的权重低。

*类比*：判卷子时重点看核心题目的对错，别让格式问题（背景噪声）淹没了真正的错误（关键内容错了）。

### 误区 3：抄也要抄得聪明

这是最 math-heavy 的部分，但 idea 很朴素。

之前的方法抄答案是"原样照搬"。WorldCache 说：**抄也要抄得有讲究**。它会看最近几次真正算过（cache miss）时的"变化趋势"，顺着这个趋势外推，而不是直接拿上次的值。

更厉害的是，如果场景有运动，它还会先做"光流对齐"——把上一帧的特征"扭曲"到当前帧的坐标系，再拿来用。

*类比*：你要预测明天的股价，与其直接用今天的价格，不如看过去几天的走势，顺势外推。但如果趋势在拐头（方向变了），外推的幅度自动减小，避免离谱预测。

### 误区 4：前期和后期用同一个标准

Diffusion 去噪有两个阶段：前期是建立大结构（语义、构图、运动方向），后期只是修细节（纹理、高频）。

之前的方法全程用同一个偷懒标准，结果前期太宽松（漏算结构信息）或后期太保守（明明只是修细节却还在老老实实算）。

**WorldCache 的做法**：前期收紧标准（必须老老实实算），后期大幅放宽标准（放心偷懒，因为只是修细节，cache 来的已经够好）。

*类比*：画画时草稿阶段每笔都要认真（决定构图），上色阶段可以加快（在已有框架内填充）。

## 效果

在 Cosmos-Predict2.5 这个 NVIDIA 的 world model 上：
- **2.3× 加速**（54 秒 → 24 秒）
- **质量保持 99.4%**（基本看不出区别）
- 之前最好的 DiCache 只有 1.3× 加速

关键是在 dynamic scene（机器人操作、自动驾驶、人体动作）这些最难的场景上，WorldCache 明显胜出，而恰恰是这些场景对 world model 最重要。

## 为什么这个工作有意思

它没发明新网络、没训练新模型，纯靠"聪明地偷懒"就拿到 2.3× 加速。而且偷懒的方式不是工程 hack，是有数学原则的——把 caching 问题重新 cast 成"已知信号的动态近似"问题，然后借用了信号处理、系统辨识、视频编码的成熟工具（零阶保持、运动补偿、自适应采样）。

这种 **"把 ML 问题转成已知有解的工程问题"** 的思路，我觉得比"堆新模块"更优雅。

## 一句话总结

**之前的方法偷懒是"原样照抄"，WorldCache 是"看情况、看重点、看趋势、分阶段"地偷懒，所以偷得更多还不被发现。**

---

# WorldCache 深度讲解

Andrej, 这篇 paper 我觉得写得相当 solid，它本质上是把 diffusion model caching 从一个 "工程 hack" 重新 cast 成一个 "control-theoretic / system-identification" 问题。核心 insight 非常清晰：**现有 caching 方法 (FasterCache, DiCache) 隐含了一个 zero-order hold 假设 —— 当 drift 小时直接 copy stale features，这在静态场景 OK，但在 world model 的动态 rollout 里会累积 ghosting/blur**。WorldCache 用四个互相正交的模块系统地解决这个假设的三个 blindspot。

## 1. Paper 的 Core Argument — 为什么 prior caching 在 world model 上 fail

DiT-based video world model (Cosmos-Predict2.5, WAN2.1) 的推理 bottleneck 是：每个 frame 要跑 N 个 transformer block × T 个 denoising step × autoregressive rollout，latency 是 interactive simulation 的主要 obstacle。

Caching 的基本 idea：consecutive denoising steps 之间 intermediate features 变化小，recompute 是 wasteful 的。DiCache 的 probe-then-cache 范式：
1. 跑前 k 个 probe block 得到 $\mathbf{z}_t^{(k)}$
2. 计算 drift $\delta_t = \frac{\|\mathbf{z}_t^{(k)} - \mathbf{z}_{t-1}^{(k)}\|_1}{\|\mathbf{z}_{t-1}^{(k)}\|_1 + \epsilon}$
3. 若 $\delta_t < \tau$，skip 剩余 N-k 个 block，直接复用 cached deep states

**三个 blindspot**（这是 paper 的核心观察，值得 build intuition）：

| Blindspot | 现象 | 后果 |
|---|---|---|
| Global drift averaging | 静态 background 平均掉了 foreground 大变化 | 该 recompute 时 skip |
| Uniform spatial weighting | salient entity (agent/hand/object) 误差同等权重 | semantic smearing |
| Static threshold | 早期建立 structure / 晚期 refine detail 用同一 $\tau$ | 早期过保守，晚期浪费 cache hit |

## 2. 四个模块的公式精解

### 2.1 CFC — Motion-Adaptive Decisions (when to skip)

第一个模块回答："reuse 在当前 motion magnitude 下是否 safe？" 

**Velocity proxy** (Eq. 2):
$$v_t = \frac{\|\mathbf{z}_t^{(0)} - \mathbf{z}_{t-2}^{(0)}\|_1}{\|\mathbf{z}_{t-2}^{(0)}\|_1 + \epsilon}$$

变量含义：
- $\mathbf{z}_t^{(0)} \in \mathbb{R}^{B \times T \times H \times W \times D}$: 第 t 个 denoising step 的输入 latent (上标 (0) 表示未经任何 transformer block 处理)
- $v_t$: normalized two-step input change，单位是 dimensionless ratio
- $\epsilon$: numerical stability (避免除零)

**关键设计**：用 t-2 而不是 t-1 作为 anchor，因为 t-1 可能本身就是 cached approximation（没跑过 deep block）。这是 "causal" 的来源 —— 你需要一个 reliable 的 reference point。这跟 signal processing 里的 ping-pong buffer 思想一致。

**Motion-adaptive threshold** (Eq. 3):
$$\tau_{CFC}(v_t) = \frac{\tau_0}{1 + \alpha \cdot v_t}$$

变量：
- $\tau_0 = 0.08$: base threshold（drift 低于此值才 skip）
- $\alpha = 2.0$: motion sensitivity
- 当 $v_t \to 0$ (静态): $\tau_{CFC} \to \tau_0$，宽松
- 当 $v_t \to \infty$ (剧烈运动): $\tau_{CFC} \to 0$，threshold 极紧，几乎不 skip

**Intuition**: 这是一个 $1/(1+x)$ 形式的 saturation curve，类似 ReLU 的 smooth 版本 / 类似 softmax 的衰减形式。它 monotonically tightens with motion，且 bound 在 $(0, \tau_0]$ —— 数学性质好，不会 blow up。

### 2.2 SWD — Saliency-Weighted Drift (改 drift 信号本身)

第二个模块回答："drift 测的是 right thing 吗？" DiCache 的 $\delta_t$ 是 spatial uniform average，会 background 飘一点就把 foreground 漂走盖过去。

**Saliency map** (Eq. 4):
$$S_{h,w} = \mathrm{Var}_d\big(\bar{\mathbf{z}}_t^{(k)}[h, w, :]\big)$$

变量：
- $\bar{\mathbf{z}}_t^{(k)}$: probe output，沿 batch 和 temporal axis 平均
- $[h, w, :]$: spatial location $(h, w)$ 的全部 channel
- $\mathrm{Var}_d$: 沿 channel dimension d 的 variance

**为什么 channel variance 是好的 saliency proxy？** Channel 维度上 variance 高 → 这个 spatial location 的 feature 跨 channel 表达丰富 → 对应 edge / texture / object boundary（参考 DINO/ViT 的 [CLS] attention 和 channel statistics 的关系，Caron et al. DINO paper 证明 self-supervised ViT 的 channel variance 与 semantic boundary 强相关 https://arxiv.org/abs/2104.14294）。这是 free saliency，不需要额外网络。

**Saliency-weighted drift** (Eq. 5):
$$\delta_t^{SWD} = \frac{1}{HW}\sum_{h,w} \|\mathbf{z}_t^{(k)}(h,w) - \mathbf{z}_{t-1}^{(k)}(h,w)\|_1 \cdot (1 + \beta_s \hat{S}_{h,w})$$

变量：
- $\beta_s = 0.12$: saliency emphasis
- $\hat{S}_{h,w} \in [0, 1]$: normalized saliency
- weighting factor $(1 + \beta_s \hat{S}_{h,w}) \in [1, 1+\beta_s] = [1, 1.12]$

**Intuition**: 这是 reweighting 而非 gating —— background (低 $\hat S$) 的 contribution 不被 zero out，只是被 attenuate 到 1.0×，salient 区域被 amplify 到 1.12×。设计上比 hard mask 更 robust，避免 saliency estimation 噪声导致 over-aggressive skipping。Eq. 6 是 skip criterion：

$$\text{skip} \iff \delta_t^{SWD} < \tau_{CFC}(v_t)$$

### 2.3 OFA — Optimal Feature Approximation (cache hit 时怎么近似)

这是 paper 里最 math-heavy 的模块，也是我觉得最 elegant 的部分。回答："skip 之后，能不能不用 stale copy 而用更好的 approximation？"

#### 2.3.1 OSI (Optimal State Interpolation)

DiCache 用 scalar ratio $\gamma = \|\Delta_{tgt}\| / \|\Delta_{src}\|$ (L1 distance ratio)，这只 capture magnitude，丢了 directional info。当 feature trajectory 弯曲时，scalar extrapolation 沿 stale 方向，error 累积。

**Reformulation as least-squares projection**:

定义 deep computation residual (Eq. 7):
$$\mathbf{r}_t = \mathbf{z}_t^{(N)} - \mathbf{z}_t^{(0)}$$

这是 deep block 的 "净贡献"。Cache hit 时我们只有 $\tilde{\mathbf{r}}_t = \mathbf{z}_t^{(k)} - \mathbf{z}_t^{(0)}$ (probe 给的 partial residual)。

定义方向 deltas (Eq. 8):
$$\Delta_{tgt} = \tilde{\mathbf{r}}_t - \mathbf{r}_{t-2}, \quad \Delta_{src} = \mathbf{r}_{t-1} - \mathbf{r}_{t-2}$$

- $\Delta_{tgt}$: "我们想要的方向"（当前 probe 暗示的 deep residual 走向）
- $\Delta_{src}$: "我们能参考的方向"（上一次 cache miss 到上上次 cache miss 之间 deep residual 的实际走向）

**最优 gain** (Eq. 9):
$$\gamma^* = \arg\min_\gamma \|\Delta_{tgt} - \gamma \Delta_{src}\|^2 = \frac{\langle \Delta_{tgt}, \Delta_{src}\rangle}{\|\Delta_{src}\|^2 + \epsilon}$$

这是标准 orthogonal projection / least squares 解。几何含义：

- $\gamma^* = \cos\theta \cdot \frac{\|\Delta_{tgt}\|}{\|\Delta_{src}\|}$，其中 $\theta$ 是 $\Delta_{tgt}$ 与 $\Delta_{src}$ 之间的夹角
- 当 trajectory 线性（$\Delta_{tgt} \parallel \Delta_{src}$，$\theta=0$）: $\cos\theta=1$, $\gamma^* = \|\Delta_{tgt}\|/\|\Delta_{src}\|$，退化成 DiCache 的 scalar ratio —— OSI 严格 generalizes DiCache
- 当 trajectory 弯曲（$\theta \to 90°$）: $\cos\theta \to 0$, $\gamma^* \to 0$，OSI 自动 attenuate，避免沿 stale direction extrapolate
- 当 trajectory 反向（$\theta > 90°$）: $\gamma^* < 0$，但被 clamp 到 $[0, \gamma_{max}=2]$

**近似 deep output** (Eq. 10):
$$\hat{\mathbf{z}}_t^{(N)} = \mathbf{z}_t^{(0)} + \mathbf{r}_{t-2} + \gamma^*(\mathbf{r}_{t-1} - \mathbf{r}_{t-2})$$

变量：
- $\mathbf{z}_t^{(0)}$: 当前 input (known)
- $\mathbf{r}_{t-2}$: 上上次 cache miss 时的 deep residual (cached)
- $\mathbf{r}_{t-1}$: 上次 cache miss 时的 deep residual (cached)
- $\gamma^*$: 上面算出的 projection gain

**Intuition**: 这其实是在做 linear extrapolation，但 gain 不是 naive 的 ratio，而是带 "方向 confidence" 的 ratio。本质上 $\hat{\mathbf r}_t = \mathbf{r}_{t-2} + \gamma^*(\mathbf{r}_{t-1} - \mathbf{r}_{t-2})$，是在 $\mathbf{r}_{t-2} \to \mathbf{r}_{t-1}$ 这条线段上做外推，外推距离 $\gamma^*$ 衡量了"我有多信任当前 trajectory 沿用过去方向"。

这跟 system identification 里的 ARX model / Kalman filter 里的 update step 有形式上的 similarity（参考 Ljung, *System Identification: Theory for the User*）。**Inner product 是关键**：它把 "方向信任" baked into 了 caching 决策。

#### 2.3.2 Motion-Compensated Warping

OSI 修了 temporal misalignment，但 cached features 还可能 spatially misaligned (scene 有 motion 时 cached features 在旧坐标上)。

**Displacement field** (Eq. 11):
$$\mathbf{u}_{t,t-1} = \mathrm{LatentCorr}(\mathbf{z}_t^{(0)}, \mathbf{z}_{t-1}^{(0)})$$

实现细节 (Appendix E.1)：
- 在 latent space 跑 Lucas-Kanade optical flow (Bruhn et al. 2005, https://link.springer.com/article/10.1023/B:VISI.0000045414.49375.3d)
- **关键 trick**: 先 downsample 到 $s_{flow} \times H \times s_{flow} \times W$（$s_{flow}=0.5$），在低分辨率算 flow，再 upsample + scale by $1/s_{flow}$
- 这有两个作用：(1) spatial low-pass filter，过滤掉 deep feature 里的高频 noise；(2) 把 correlation matrix 面积降到 1/25，Lucas-Kanade solver 复杂度大幅降低
- 整体 overhead < 3% per cached step

**Warp** (Eq. 12):
$$\tilde{\mathbf{z}}_{t-1}^{(N)} = \mathrm{Warp}(\mathbf{z}_{t-1}^{(N)}, \mathbf{u}_{t,t-1})$$

然后用 $\tilde{\mathbf{z}}_{t-1}^{(N)}$ 替代 $\mathbf{z}_{t-1}^{(N)}$ 进入 Eq. 10 的 residual 计算。前 5 步禁用 warping (低 SNR 让 displacement estimation 不可靠)。

**Intuition**: 这跟经典 video recognition 里的 Deep Feature Flow (Zhu et al. CVPR 2017, https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhu_Deep_Feature_Flow_CVPR_2017_paper.pdf) 思路同源 —— 不在 frame i 算 CNN feature，而是在 frame i-1 算 + warp 过来。但 WorldCache 把它放到 diffusion 的 denoising trajectory 内，而不是 video frame 维度，这是一个 conceptual shift：denoising step 之间的 latent 也有 "motion"，可以被 warp。

### 2.4 ATS — Adaptive Threshold Scheduling (phase-aware)

最后一个模块回答："能不能在不破坏 fidelity 的前提下推得更激进？"

**Linear version** (Eq. 13):
$$\tau_{ATS}(t) = \tau_{CFC}(v_t) \cdot \left(1 + \beta_d \cdot \frac{t}{T}\right)$$

变量：
- $t \in [0, T]$: 当前 denoising step
- $T = 35$: 总 denoising steps
- $\beta_d = 4.0$: relaxation rate

**实际实现用 quadratic** (Appendix E.2, Eq. 30-31):
$$C(u) = \frac{u^2}{6} + \frac{u}{2} + \frac{10}{3}, \quad u = N/35.0$$
$$D(t) = 1.0 + C(u) \cdot r_t, \quad r_t = t/N$$
$$\tau_{ATS}(t) = \tau_{base} \cdot D(t)$$

边界行为：
- $t=0$: $D \approx 1.0$，threshold 紧，强制 full execution（structure formation phase）
- $t=35$ (N=35): $D \approx 5.0$，threshold 松 5×，aggressive reuse（detail refinement phase）

**Intuition**: 这对应 diffusion model 的两阶段行为（Ho et al. DDPM https://arxiv.org/abs/2006.11239, Song et al. DDIM https://arxiv.org/abs/2010.02502 的 score function 性质）：
- 早期 high noise → 网络 update 大、semantically critical（global layout / motion 建立）→ 必须用紧 threshold
- 晚期 low noise → 网络 update 小、high-frequency correction only → cached approximation 已经够好，可以激进 skip

ATS 是 **WorldCache 2.3× speedup 的主要来源**。Fig. 4 显示 fixed threshold 在 step 20 后 cache hit rate 暴跌到 36%，ATS 维持 68%。这是 paper 里最重要的一个 figure，值得仔细看。

## 3. Ablation Study — "Invest-and-Spend" 设计哲学

Table 4 的 incremental ablation 揭示了 WorldCache 的设计哲学，这是 paper 最 elegant 的地方：

| Config | Domain↑ | Quality↑ | Overall↑ | Speedup↑ | Lat(s)↓ |
|---|---|---|---|---|---|
| Base | 0.8447 | 0.7607 | 0.8027 | 1.00× | 55 |
| +CFC | 0.8457 | 0.7583 | 0.8020 | 1.52× | 36 |
| +CFC+SWD | 0.8414 | 0.7592 | 0.8003 | 1.67× | 33 |
| +CFC+SWD+OFA | 0.8468 | 0.7602 | **0.8035** | 1.49× | 37 |
| +CFC+SWD+OFA+ATS | 0.8395 | 0.7559 | 0.7977 | **2.30×** | 25 |

**关键观察**：
1. **CFC**: free speedup (1.52×)，quality 几乎不变 —— motion-aware thresholding 是 zero-cost safety net
2. **SWD**: 再加 speedup (1.67×)，quality 微降 —— 让 background drift 不再 dominate skip 决策
3. **OFA**: **speedup 反而降了** (1.67→1.49)，但 **quality 达到最高 0.8035**（甚至超过 base 0.8027）！这是 "invest" 阶段 —— OFA 用更精确的 approximation 换取 quality margin，故意 trade 一些 throughput
4. **ATS**: 把 OFA 攒的 quality margin "spend" 掉换 speedup (1.49→2.30×)，quality 只降 0.6%

这个 **"先建立 quality buffer，再 spend 它换 speed"** 的设计模式非常漂亮，本质上是一个 Lagrangian 的 trade-off 管理。OFA 不是为了直接加速，是为了让 ATS 后面可以放心激进。这种 "前向投资" 的思路在 system design 里很常见，但在 ML 加速 paper 里少见，通常大家都是 "每个模块都直接贡献 speedup"。

## 4. 实验结果全景

### 4.1 主实验 — Cosmos-Predict2.5

Table 1 (T2W) 和 Table 2 (I2W)：

| Model | Task | Method | Speedup | Overall↑ | Lat(s)↓ |
|---|---|---|---|---|---|
| 2B | T2W | Baseline | 1.0× | 0.748 | 54.34 |
| 2B | T2W | DiCache | 1.3× | 0.743 | 40.82 |
| 2B | T2W | FasterCache | 1.6× | 0.652 | 34.51 |
| 2B | T2W | **WorldCache** | **2.1×** | 0.745 | 26.28 |
| 2B | I2W | **WorldCache** | **2.3×** | 0.798 | 24.48 |
| 14B | T2W | **WorldCache** | **2.14×** | 0.771 | 98.61 |
| 14B | I2W | **WorldCache** | **2.18×** | 0.813 | 99.25 |

关键点：
- FasterCache 在 2B T2W 上 quality 从 0.748 砸到 0.652 (-13%)，证明 naive caching 在 world model 上确实有害
- WorldCache 在 14B 上 quality 反而 **超过** baseline (T2W: 0.771 vs 0.769；I2W 上保持) —— 暗示大模型本身有冗余 computation 可被 OFA 这种"更聪明的近似"替代而不损 quality
- 在 PAI-Bench 的 Domain sub-metrics 上 (RO=Robot, IN=Industry)，WorldCache 比 DiCache 提升最大（Δ +0.028, +0.011），这正是 dynamic scene 最多的地方 —— 印证了 SWD 的作用

### 4.2 Transfer — WAN2.1 (Table 3)

| Model | Task | Method | Speedup | Overall↑ |
|---|---|---|---|---|
| WAN2.1-1.3B | T2W | WorldCache | **2.36×** | 0.7721 (超 baseline 0.7727 几乎平) |
| WAN2.1-14B | I2W | WorldCache | **2.31×** | 0.7388 (超 baseline 0.7384) |

Transfer 到不同 backbone 仍然有效，证明 WorldCache 不是 Cosmos-specific 的 trick。

### 4.3 EgoDex-Eval — 机器人场景 (Table 6, Appendix D)

这是 paper 的高光时刻之一 —— 用 ground-truth video 算 PSNR/SSIM/LPIPS：

| Backbone | Method | PSNR↑ | SSIM↑ | LPIPS↓ | Speedup↑ |
|---|---|---|---|---|---|
| WAN2.1-14B | Baseline | 13.30 | 0.503 | 0.459 | 1.0× |
| WAN2.1-14B | DiCache | 12.95 | 0.491 | 0.461 | 1.88× |
| WAN2.1-14B | **WorldCache** | **13.19** | 0.498 | 0.460 | **2.30×** |
| Cosmos-2.5-2B | **WorldCache** | 12.82 | **0.466** | 0.518 | 1.62× |
| DreamDojo-2B | **WorldCache** | 23.69 | 0.737 | 0.251 | 1.90× |

特别值得注意的是 **Cosmos-2.5-2B 上 WorldCache 的 SSIM 0.466 > baseline 0.455** —— caching 竟然比 full execution 还高？这暗示 OFA 的 least-squares blending 在某种程度上对 deep network 的 "过度 correction" 做了 regularize，类似 temporal smoothing 但保留 motion（因为有 warping）。这是一个值得深挖的发现。

### 4.4 与更多 baseline 对比 (Table 5, Appendix C)

补充对比了 EasyCache 和 TeaCache (Fast/Slow)：

| Method | T2W Speedup | T2W Overall | I2W Speedup | I2W Overall |
|---|---|---|---|---|
| TeaCache (Slow) | 1.1× | 0.7454 | 1.1× | 0.7979 |
| EasyCache | 1.3× | 0.7451 | 1.3× | 0.7975 |
| DiCache | 1.3× | 0.7431 | 1.3× | 0.7941 |
| **WorldCache** | **2.10×** | 0.7450 | **2.30×** | 0.7977 |

所有 prior 方法都卡在 ~1.3×，WorldCache 直接 break 2× barrier，quality 跟最强的 TeaCache-Slow 持平。这是 Pareto frontier 上的明显突破。

### 4.5 Denoising step budget 的影响 (Fig. 6)

从 35 步增到 140 步：
- Baseline: 57s → 199s (线性 scale)
- WorldCache: 25s → 66s，speedup 从 2.3× 涨到 **3.10×**

Intuition: denoising 步数越多，late refinement phase 占比越大，ATS 能激进 skip 的比例越高。这对 long-rollout world simulation 非常友好 —— 越长的 trajectory 收益越大。

## 5. 我的 Intuition 和联想

### 5.1 这是 System Identification 思想 entering diffusion caching

Paper reference 29 是 Ljung 的 *System Identification: Theory for the User*。这个 reference 不是装饰 —— 整个 OFA 模块的本质是 **online parameter estimation of a dynamical system**：
- State: deep residual trajectory $\mathbf{r}_{t-2}, \mathbf{r}_{t-1}, \mathbf{r}_t$
- Model: linear extrapolation $\hat{\mathbf r}_t = \mathbf{r}_{t-2} + \gamma(\mathbf{r}_{t-1} - \mathbf{r}_{t-2})$
- Online estimator: least-squares projection 给 $\gamma^*$

这跟 Kalman filter 的 predictor step 形式上同构 —— Kalman 也是用过去 state + motion model 预测当前，然后用 observation 修正。WorldCache 没有 "observation 修正" 这一步（因为 cache hit 时没有 ground truth），但 $\gamma^*$ 的 directional gating 起到了类似 "innovation gating" 的作用。

**更激进的联想**: 是否可以把 OFA 升级成 full Kalman filter？引入 process noise Q 和 measurement noise R，让 cache hit 时的 approximation 有 uncertainty estimate，进而驱动 ATS 的 relaxation rate？这是一个 obvious next step，paper 在 Limitations 里也提了 "uncertainty-aware warping"。

### 5.2 与 P-DiT / DeepCache family 的关系

DeepCache (Ma et al. CVPR 2024, https://arxiv.org/abs/2312.00858) 在 U-Net 上 cache 高层 feature，FasterCache (Lv et al. ICLR 2025, https://openreview.net/forum?id=W49UjcpGxx) 扩展到 video DiT 加 CFG cache，DiCache (Bu et al. ICLR 2026, https://openreview.net/forum?id=kflYZjGumW) 加 online probe。**WorldCache 把这条线推到了 "dynamical approximation + perception constraint"**，从 "skip-or-not" binary 决策升级成了 "how well to approximate" continuous decision。

### 5.3 Zero-Order Hold 这个 framing 非常 helpful

把 prior 方法 cast 成 "zero-order hold" (信号处理里的 ZOH：sample-and-hold，把连续信号离散化时直接 hold 上一时刻值) 是一个很漂亮的 conceptual lens。它解释了为什么 dynamic scene 会 ghosting —— ZOH 在高频信号上 alias 严重。WorldCache 引入了：
- **First-order hold** (Eq. 10 的 linear extrapolation) — OSI
- **Motion compensation** — warping
- **Adaptive sampling rate** — ATS

这跟 video coding 里的 motion-compensated prediction (H.264/H.265) 思想几乎完全对应：I-frame (cache miss) → P-frame (cache hit + motion compensation)。WorldCache 本质上是把 video codec 的 inter-frame prediction 思路搬到了 diffusion 的 intra-rollout denoising trajectory 上。这个类比可以延伸：是否可以引入 B-frame (bidirectional prediction)？用 $\mathbf{r}_{t-2}$ 和 $\mathbf{r}_{t+1}$ (如果有的话) 双向插值？

### 5.4 Saliency = Channel Variance 是个聪明 hack

SWD 用 channel variance 做 saliency proxy，不需要额外网络（vs. DreamSim/LPIPS/CLIP-Score）。这跟 DINO 的 [CLS] token emergent property https://arxiv.org/abs/2104.14294 和 ViT 的 channel statistics 都是 self-supervised saliency 的观察一致。Intuition 是 transformer 的 channel 是 "basis functions"，variance 高的地方是多个 basis 都 active 的地方，对应复杂结构。

**可能的改进**: channel variance 是 pixel-level saliency，对 world model 来说 "object-level" saliency 可能更重要。可以用attention rollout (Abnar & Hutter 2020, https://arxiv.org/abs/2005.00928) 或 DINO [CLS] attention 做 object mask，但 overhead 更高。Paper 选 channel variance 是工程上 sweet spot。

### 5.5 Ping-Pong Buffer 是被低估的设计

CFC 用 t-2 而不是 t-1 做 anchor，配 ping-pong buffer (两个 alternating cache slot by step parity) —— 这避免了 "cache 依赖 cache" 的误差累积链。在 autoregressive world model rollout 里，error compounding 是 fatal 的（参考 Genie https://arxiv.org/abs/2402.15391 的 long-horizon drift 问题），任何依赖自己过去预测的方法都要小心这个。Ping-pong 强制 reuse 锚定到 fully-computed state，这是 "causal" 命名的真正含义。

### 5.6 Denoising Phase Awareness 的更深层含义

ATS 的 phase-aware relaxation 跟近期对 diffusion model 的 phase 分析工作呼应。Hang et al. 的 "Denoising Diffusion Steps" 分析表明早期 step 决定 low-frequency (semantic)，晚期 step 修正 high-frequency (texture) (参考 Rissanen et al. https://arxiv.org/abs/2204.13902 的 frequency analysis)。WorldCache 的 ATS 是把这种 frequency-aware thinking 转成 caching policy。

**值得探索**: 是否可以 multi-scale ATS？对 low-frequency content 用更激进 ATS，high-frequency 用更保守？这跟 Laplacian Pyramid (Burt & Adelson 1983, paper reference 7, https://persci.mit.edu/pub_pdfs/pyramid83.pdf) 的多尺度 representation 有天然 mapping。

### 5.7 Limitations 提到的 Online Policy Learning

Paper 最后提到 "learn or adapt caching policies online"。这是我最期待的方向 —— 当前 $\tau_0, \alpha, \beta_s, \beta_d$ 都是 hyperparameter (Table 7 显示 sensitivity)。一个轻量 RL agent（PPO/contextual bandit）online 学 caching policy，reward = (speedup, quality) 的 Pareto utility，state = (probe drift, motion velocity, saliency map, denoising phase)，action = skip/recompute + γ value。这跟 learned step skipping in fast solvers (如 Salimans & Ho progressive distillation https://arxiv.org/abs/2202.00512) 形成对照，但 WorldCache 是 token/feature level 而非 step level。

### 5.8 跟 Flow Matching 的关系

Cosmos 用 Flow Matching (Lipman et al. 2023, https://arxiv.org/abs/2210.02747) velocity prediction objective，不是 ε-prediction。这意味着 $\mathbf{z}_t^{(0)}$ 是 noisy latent，trajectory 是 ODE 解。OSI 的 linear extrapolation 假设 residual trajectory 局部线性 —— 对 ODE 解的短时 local linear approximation 是合理的（Taylor expansion first-order term），但在 trajectory curvature大时 break。这可能解释为什么 paper 要 clamp $\gamma^* \in [0, 2]$ 而不是 unbounded。

## 6. 总结性 Intuition

WorldCache 把 caching 加速从 "engineering heuristic" 升级到 "constrained dynamical approximation"。四个模块各自解决一个独立 blindspot，组合起来形成 "invest-and-spend" 的 Pareto 管理。最重要的是它把 signal processing / system identification / video coding 的成熟思想（ZOH、motion compensation、adaptive sampling、saliency weighting）以 mathematically principled 的方式引入了 diffusion 加速。这种 **"把 ML problem 转成已知有解的 engineering problem"** 的思路我觉得是 ML system design 的最 productive pattern 之一。

**Reference links**:
- DiCache (ICLR 2026): https://openreview.net/forum?id=kflYZjGumW
- FasterCache (ICLR 2025): https://openreview.net/forum?id=W49UjcpGxx
- DeepCache (CVPR 2024): https://arxiv.org/abs/2312.00858
- Cosmos-Predict: https://arxiv.org/abs/2501.03575
- PAI-Bench: https://arxiv.org/abs/2512.01989
- WAN2.1: https://arxiv.org/abs/2503.20314
- Genie (DeepMind): https://arxiv.org/abs/2402.15391
- DINO (saliency 参考): https://arxiv.org/abs/2104.14294
- DDIM: https://arxiv.org/abs/2010.02502
- Lucas-Kanade flow (Bruhn et al.): https://link.springer.com/article/10.1023/B:VISI.0000045414.49375.3d
- Deep Feature Flow for Video Recognition (CVPR 2017): https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhu_Deep_Feature_Flow_CVPR_2017_paper.pdf
- Ljung, System Identification: Theory for the User (Prentice Hall, 1999)
- Flow Matching (Lipman et al.): https://arxiv.org/abs/2210.02747
- Progressive Distillation (Salimans & Ho): https://arxiv.org/abs/2202.00512
- Laplacian Pyramid (Burt & Adelson): https://persci.mit.edu/pub_pdfs/pyramid83.pdf
- Attention Rollout (Abnar & Hutter): https://arxiv.org/abs/2005.00928
- TeaCache (Timestep embedding tells): https://openaccess.thecvf.com/content/CVPR2025
- EasyCache: https://arxiv.org/abs/2507.02860
- EgoDex: https://arxiv.org/abs/2505.11709
- DreamDojo: https://arxiv.org/abs/2602.06949

如果你想 push 这个方向，我觉得最 promising 的三个 angle：(1) OFA 升级成 Kalman filter with uncertainty，让 ATS 的 relaxation rate 自适应 uncertainty；(2) Object-level saliency via DINO [CLS] attention 替代 channel variance；(3) B-frame bidirectional prediction for non-causal cache。这三条都直接来自 video coding literature 的成熟工具，translation cost 低。
