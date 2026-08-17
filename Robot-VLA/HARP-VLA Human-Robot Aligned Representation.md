---
source_pdf: HARP-VLA Human-Robot Aligned Representation.pdf
paper_sha256: 0dd4838d00432b3ccd75b7a9a0ef30c3da9f87c2a995bbc2f776c57a04f2a682
processed_at: '2026-08-04T23:30:55-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HARP-VLA 人话版

Andrej，我换个说法，把这个 paper 的故事从头讲一遍。

---

## 一句话概括

想让 robot 学会做事，只靠 robot 自己的数据太贵太少。互联网上有海量 human 手操作视频，理论上是个金矿。但问题在于——human 长得跟 robot 不一样，摄像头视角不一样，手跟 gripper 不一样。你直接把 human video 喂给 VLA 模型，它学出来的东西是乱的。HARP 干的事就是**搭一座桥**，让 human video 和 robot video 在 feature space 里对齐，这样 human 的知识就能流到 robot policy 里。

---

## 问题在哪

你可能会想：latent action model 不是已经解决 embodiment gap 了吗？LAPA、UniVLA 不都是从 video 里学 discrete action code，bypass 了具体的 motor command？

对，但有个隐藏问题。latent action 是从**visual feature** 里提取的。你的 encoder 输入是两帧画面，输出 latent action 表示"这两帧之间发生了什么动作"。如果 human video 和 robot video 在 visual feature space 里是两个分离的 cluster——它们就没在同一套坐标系下。那 human video 学到的 latent action "抓取"，和 robot video 学到的 latent action "抓取"，落到 codebook 的不同 region。你以为 latent action 是 embodiment-agnostic 的通用语言，实际上它偷偷编码了"这是 human 还是 robot"的 domain signal。

下游你把 human + robot video 混在一起 pretrain VLA policy，policy 看到的 latent action labels 是两套不兼容的 subsystem。human data 的 supervision signal 根本流不进 robot policy。**action gap 看起来被 latent action 解决了，visual gap 又把这个 solution 给污染了**。

---

## HARP 的思路

两个 gap 必须一起解决，不能分步。

用**少量 paired human-robot video**（同一任务，human 做一遍，robot 做一遍）当 bridge——这告诉你"human 的这个动作"对应"robot 的那个动作"。
用**大量 unpaired video**（各自独立的 human video 和 robot video）当 dynamics supervision——这告诉你" manipulation 的 dynamics 长什么样"。

然后在 visual encoder 和 latent action model 上同时施加约束，让两个 domain 在同一个 space 里对齐。

---

## 三个 stage 干的事

### Stage 1：学对齐的 encoder 和 latent action model

**Visual encoder 设计**：human video 用 frozen 的 pretrained encoder（DINOv2 + SigLIP），robot video 用加了 adapter 的同款 encoder。adapter 是个小 MLP residual，挂在每个 transformer block 后面，zero-init 保证一开始输出跟原 encoder 完全一样，然后慢慢学。

为什么 freeze human 而动 robot？因为 VLM 预训练本来就是 human-centric 的，human representation 已经跟 LLM 对齐得很好。你动 human branch 会破坏这个已有的 alignment，且 human 本来就是 teacher，没什么好学的。**robot 往 human 靠，human 原地不动**。

**Latent action model（HARP-LAM）**：VQ-VAE 结构。给两帧 + language instruction，encoder 输出 continuous latent action，量化成 4 个 discrete tokens（codebook size 16），decoder 用当前帧 + latent action 预测下一帧的 visual feature。

关键创新在 **cross-prediction**。对 paired video (H, R)：
- 从 human 的 transition $(z_H^t, z_H^{t+\Delta t})$ 提取 latent action $a_H^t$
- 用这个 $a_H^t$ + robot 当前帧 $z_R^t$，要求 decoder 预测 robot 下一帧 $z_R^{t+\Delta t}$

如果你的 latent action 真的是 embodiment-agnostic 的"抓取"信号，它应该能解释 robot 视频的未来 dynamics。这个 objective 把 latent action 的语义跟 cross-embodiment alignment 绑死。你没法学到 domain-specific 的 latent action 还能在这个 loss 下 survive。

**Auxiliary cue**：object keypoint 轨迹 + wrist/end-effector 轨迹。这两个东西是 cross-embodiment invariant 的——human hand 抓杯子和 robot gripper 抓杯子，杯子移动的轨迹相似。把这个作为 supervision 注入 latent action encoder，强迫 latent action 编码 task-relevant dynamics 而非 domain appearance。

**Alignment loss（SRPD）**：这部分是算法核心，分两块。

Source-Relative（SR）term：不要求 robot feature 到 human feature 距离为 0，只要求**adapted robot feature 到 human 的距离，比 frozen robot feature 到 human 的距离小至少 margin $m_s$**。这是 relative improvement，不是 absolute collapse。

Pair-Discriminative（PD）term：triplet loss 变体。要求正对距离比 batch 内其他负对平均距离小至少 $m_t$。保证 alignment 不以牺牲 pair-level discriminability 为代价。

为什么 SR + PD 比纯 L2 好？纯 L2 强行 collapse 所有 paired feature，双向 retrieval 不对称——robot feature 全挤到 human feature 的一个小 region，robot 内部的 discriminability 丢了。SR 改成 relative target，PD 保 discriminability。Table 1 里 HARP-SRPD 双向 R@1 都到 78.50，纯 L2 只有 61.83。

### Stage 2：Pretrain VLA

Stage 1 训完，你有了对齐的 vision encoder（HARP-VE）和 latent action model（HARP-LAM）。用 HARP-LAM 给所有 video frame 打 latent action label，然后把 HARP-VE 的 adapter 权重 copy 到 VLM（Prismatic-7B 架构，DINOv2+SigLIP + LLaMA-2）的 vision encoder 里。

VLA policy 接收 visual input + language instruction，输出 4 个 latent action tokens，用 cross-entropy loss 训。

**关键**：Stage 2 时 vision encoder freeze。消融显示 freeze 比 unfreeze 在 CALVIN 上高 0.231 avg length。Intuition 是 Stage 1 学到的 alignment 是个 fragile sweet spot，Stage 2 如果解冻让 vision encoder 跟 latent action loss 调整，会破坏这个 alignment。

### Stage 3：Finetune real action

VLA 预训练完只能输出 latent action tokens，没法直接控制 robot。加个 lightweight action head，把 latent action embedding 映射到 normalized real action（10-step action chunk），L1 loss 训。VLA backbone 用 LoRA 微调。real action label 只在这个 stage 用。

---

## 数据怎么来

Paired data 是稀缺 bridge，unpaired data 是 scalable supervision。Paper 用的：

**Unpaired human**：HOI4D (8.9M frames) + OpenEgo (36.4M frames)
**Unpaired robot**：Bridge-V2 (8.6M frames) + 自采 DexHand (3.6M frames)
**Paired H-R**：RH20T (8.16M) + Human2Robot (9.9M) + 自采 (5.7M)

Paired 总共 23.76M frames，unpaired 57.5M frames。Paired 用在 cross-prediction + alignment loss 这种需要 bridge 的地方，unpaired 用在 self-prediction 这种可扩展的 dynamics learning 上。

**Paired video 时间对齐**：即使同一任务，human 和 robot 执行速度不一样。用 wrist keypoint 的 Euclidean distance 做 DTW，以 robot video 为 temporal reference，resample human video 的帧。这步很重要——cross-prediction 假设 frame $t$ 在两个 video 里语义对齐，时间错位的话 decoder 被迫学延迟补偿这种 spurious dynamics，污染 latent action。

---

## 结果怎么样

**Representation alignment**（Table 1）：双向 cross-embodiment retrieval，HARP-SRPD 把 avg R@1 从 unadapted 的 43.55 拉到 78.50。UMAP 可视化里 paired samples 明显聚拢。

**RLBench frozen encoder**（Table 2）：18 个 task，所有 method 用同样的 policy head，只换 frozen visual encoder。HARP-SRPD avg success rate 46.59 vs unadapted 37.56，**+9.03%**。

**CALVIN ABC→D**（Table 4）：最难的 long-horizon benchmark。HARP-VLA avg length 4.481 vs π₀.₅ 的 3.875，**+0.606**。Task5（5-subtask sequence 的 tail）上 75.9% vs π₀.₅ 61.0%，**+14.9%**。long-horizon 上拉开 gap 说明 representation 的 benefit 是累积的。

**Realworld**（Table 3）：Xarm7 + Robotera XHand，18 DoF，4 个 task 各 60 trials。HARP-VLA avg 76.3% vs π₀.₅ 69.2%，**+7.1%**。最难的 Flip Cup 上 61.7% vs 53.3%。

---

## 我觉得哪些 design choice 是对的

1. **Joint training visual + latent action**。Pipeline 式"先 align visual 再学 latent action"会在 LAM objective 里重新引入 domain discrepancy。cross-prediction 把 alignment 直接 bake 进 LAM training，这个 coupling 很 elegant。

2. **Robot-only adapter + zero-init**。全量 fine-tune DINOv2+SigLIP 风险大显存重，adapter 让 alignment 既 expressive 又 safe init。类似 ControlNet 的 zero-conv。

3. **Freeze human branch**。VLM 是 human-centric pretrain 的，human representation 跟 LLM 已经对齐。动 human branch 会破坏这个，且 human 本来是 teacher 没什么好学。

4. **Freeze vision encoder in Stage 2**。Stage 1 的 alignment 是 fragile sweet spot，Stage 2 解冻会被 latent action loss 破坏。

---

## 我的疑问

1. **Codebook size 只有 16**。LAPA 用 8192，UniVLA 也类似。HARP 用 16 我猜是 cross-prediction 的 constraint 让每个 code 必须能解释两个 embodiment 的 dynamics，codebook 大了难收敛。但 16 个 code 够不够表达 manipulation 全部 motion primitive？long-horizon task 上可能不够。

2. **Paired data 依赖性**。Paired data 收集成本高（需要 human 和 robot 同时做同任务）。Paper 没做"完全去掉 paired data"的 ablation。如果 paired data 完全没有，HARP 退化成只有 self-prediction，SRPD loss 无法计算，只剩 LAM + aux——这跟 UniVLA 还差多少？

3. **Realworld 实验规模**。每 task 60 trials，4 task，共 240 trials。没报告标准差。Flip Cup 上 +8.4% 如果方差大可能不显著。

4. **Auxiliary cue noise**。TAPIR 在严重 occlusion 下 visibility score 不可靠，WiLoR 在 fast motion 下 MANO regression 抖动。这些 noise 通过 L_aux 污染 latent action，paper 没量化影响。

---

## 核心直觉

一句话：**latent action 的 embodiment-agnostic 性质是从 visual representation 的 embodiment-agnostic 性质继承来的，所以这两个必须 joint train**。

整个 paper 的设计都在服务这个 thesis：cross-prediction 把 alignment 烙进 LAM objective，SRPD 保证 alignment 不破坏 discriminability，aux cue 把 latent action 锚在 embodiment-invariant 的 object motion 上，freeze human branch 保持 VLM 的 vision-language alignment，freeze vision encoder in Stage 2 保持 Stage 1 的 sweet spot。

这种"在恰当位置 freeze、在恰当位置 align、在恰当位置 ground"的层次化设计，工程上很扎实。代价是 paired data 依赖，这是真正的 bottleneck，也是 future work 必须解决的。

---

# HARP-VLA 深度解析

Andrej，这篇 paper 来自清华交叉信息研究院 Jianyu Chen 组，处理的是 VLA 领域一个我一直觉得被低估的核心问题：**如何让大规模 human video 真正流入 robot policy 的预训练管线**。我读完之后直觉上认为它做对了一件很关键的事——把 visual representation alignment 和 latent action learning **耦合在一个 joint objective 里**，这避免了"先 align 再学 action"那种 pipeline 式方法中 error 累积的问题。

---

## 1. 核心问题：两个 gap 的耦合性

现有 VLA pretraining from human video 面临两个 gap：

**Action execution gap**：human hand motion 无法直接映射到 robot motor command。LAPA / UniVLA / UniSkill 等 Latent Action Models (LAMs) 用 VQ-VAE 从 temporally adjacent frames 学离散 latent action codes，bypass 了 embodiment-specific motor space，这条路线已被验证。

**Visual representation gap**：这才是 paper 真正的 insight 所在。LAM 的 latent action 是 grounded in visual observation 的——你的 encoder $E_\theta$ 输入是 $z_X^t = \Phi_\theta(x^t, e_X)$。如果 human video 和 robot video 在 visual feature manifold 上是分离的两个 cluster（Fig.4 left 上半部分清晰可见），那么 latent action $a_X^t = E_\theta(z_X^t, z_X^{t+\Delta t}, l_X)$ 会**继承这个 domain discrepancy**——也就是说你以为是 embodiment-agnostic 的 latent action，实际上悄悄编码了 "这是 human 视频" 还是 "这是 robot 视频" 的 domain signal。

这意味着 human video 学到的 latent action codes 和 robot video 学到的 latent action codes 即使语义对应（比如都是"抓取"），也会落到 VQ codebook 的不同 region。当你在 Stage-2 把 human + robot video 混在一起 pretrain VLA 时，policy 实际上在学两个 disjunct 的 action subspace，human data 的 supervision signal 无法有效 transfer 到 robot policy 上。

**HARP 的核心 claim**：必须**jointly** align visual representation 和 latent action，用 paired human-robot demo 作为 bridge，用大量 unpaired video 作为 dynamics supervision，让两个 gap 在一个统一 objective 下同时被消除。

GitHub repo: https://github.com/anonymity35/HARP-VLA

---

## 2. 三阶段框架的 architecture

### Stage 1: Joint Visual + Latent-Action Alignment

**Embodiment-aware visual encoding** 是整个框架的基石。这个设计选择非常聪明：

$$
\Phi_\theta(X, e_X) = \begin{cases} F(X), & e_X = h \\ T_\theta(X), & e_X = r \end{cases}
$$

- $F$：frozen pretrained visual encoder（DINOv2 + SigLIP fused，遵循 Prismatic-7B 设计）
- $T_\theta$：robot-adapted encoder，通过 **robot-only adapter** 实现——在 DINOv2/SigLIP 每个 attention block 和 FFN 后面挂一个 2-layer MLP residual adapter
- $e_X \in \{h, r\}$：embodiment label
- $Z_X = \{z_X^t\}_{t=1}^{T_X}$：patch-level visual tokens，每帧 256 tokens (16×16)，fused dim 2176

**Adapter 初始化的 trick**：第一层 Gaussian init，第二层 **zero-init**，保证 $T_\theta(X) \equiv F(X)$ at initialization。这类似 ControlNet 的 zero-conv 设计，让训练从一个已经 well-aligned 的起点出发，避免破坏 web-scale pretrained features。

**关键 intuition**：为什么 freeze human branch 而非 robot branch？因为 VLM 的 pretraining data 主要是 human-centric web data，human representation 本身已经语义丰富、与 LLM 对齐良好。如果反过来去动 human branch，会破坏 VLM 已有的 vision-language alignment，下游 VLA pretrain 时 LLM 难以消化。**保持 human 作为 fixed semantic anchor，让 robot 往 human 主动靠拢**——这是 asymmetry 设计的本质。

### Latent Action Model (HARP-LAM)

架构上 follow UniVLA，但训练 objective 完全不同。LAM 是 VQ-VAE-style inverse + forward dynamics：

$$
a_X^t = E_\theta(z_X^t, z_X^{t+\Delta t}, l_X), \quad q_X^t = Q_\theta(a_X^t), \quad \hat{Y}_X^t = D_\theta(\tilde{z}_X^t, q_X^t, l_X)
$$

变量含义：
- $a_X^t \in \mathbb{R}^{d_q}$：continuous latent action，$d_q = 128$
- $q_X^t$：quantized latent action，由 codebook $Q_\theta$ 量化，codebook size $K=16$，每 transition 产 $N_q=4$ 个 discrete tokens
- $\hat{Y}_X^t$：decoder 预测的 target frame patch tokens
- $\tilde{z}_X^t$：decoder 的 conditioning feature（关键变量，决定 self vs cross-prediction）
- $Y_X^t$：target representation

**Self-prediction (unpaired video $V$)**：
$$
\tilde{z}_V^t = z_V^t, \quad Y_V^t = z_V^{t+\Delta t}
$$
这是标准的 forward dynamics：给当前帧 + latent action，预测下一帧 representation。

**Cross-prediction (paired video $(H, R)$)**——这是 HARP 最核心的创新：
$$
\tilde{z}_H^t = z_R^t, \quad Y_H^t = z_R^{t+\Delta t}; \quad \tilde{z}_R^t = z_H^t, \quad Y_R^t = z_H^{t+\Delta t}
$$

**Intuition**：从 human transition $(z_H^t, z_H^{t+\Delta t})$ 提取 latent action $a_H^t$，然后用它 condition 在 robot 当前帧 $z_R^t$ 上，要求 decoder 预测 robot 下一帧 $z_R^{t+\Delta t}$。如果 $a_H^t$ 真的是 embodiment-agnostic 的"抓取"信号，它应该能解释 robot 视频的未来 dynamics。这个 objective 直接把 latent action 的语义和 cross-embodiment alignment 绑定——你没法学到 domain-specific 的 latent action 还能在这个 loss 下表现好。

### Stage 1 完整 objective

$$
\mathcal{L}_{\text{stage1}} = \mathcal{L}_{\text{lam}} + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}} + \lambda_{\text{align}} \mathcal{L}_{\text{align}}
$$

#### $\mathcal{L}_{\text{lam}}$：latent action prediction loss

$$
\mathcal{L}_{\text{lam}} = \frac{1}{|\mathcal{B}|} \sum_{(X,t) \in \mathcal{B}} \left[ \|\hat{Y}_X^t - Y_X^t\|_2^2 + \ell_{\text{vq}}(X, t) \right]
$$

$$
\ell_{\text{vq}}(X, t) = \|\text{sg}[a_X^t] - q_X^t\|_2^2 + \beta \|a_X^t - \text{sg}[q_X^t]\|_2^2
$$

- $\text{sg}[\cdot]$：stop-gradient
- $\beta = 0.25$：commitment weight，控制 encoder 输出对 codebook 的"承诺"强度
- 第一项把 codebook entry 拉向 encoder output（只有 codebook 梯度流动）
- 第二项把 encoder output 拉向选中的 codebook entry（只有 encoder 梯度流动）

#### $\mathcal{L}_{\text{aux}}$：shared-cue auxiliary loss

这是防止 latent action 偷懒学到 embodiment-specific 外观线索的关键 regularization。对每个 video $X$ 提取：
- $K_X$：2D object position tracks（用 TAPIR，处理 occlusion 很强）
- $E_X$：2D wrist (human) 或 end-effector (robot) trajectory（human 用 WiLoR regression MANO 参数；robot 用 camera extrinsics + 3D wrist pose 做 perspective projection）

在 LAM encoder 里加 auxiliary tokens $u_{X,K}^\tau, u_{X,E}^\tau$，用 lightweight heads $G_K, G_E$ 预测：

$$
\hat{K}_X^\tau = G_K(u_{X,K}^\tau) \in \mathbb{R}^{N_K \times 2}, \quad \hat{E}_X^\tau = G_E(u_{X,E}^\tau) \in \mathbb{R}^2
$$

带 visibility mask 的 Huber loss：
$$
\ell_K(X) = \frac{\sum_\tau \sum_{k=1}^{N_K} M_{X,K}^{\tau,k} \mathcal{H}(\hat{K}_{X,k}^\tau, K_{X,k}^\tau)}{\sum_\tau \sum_k M_{X,K}^{\tau,k} + \epsilon}
$$
$$
\ell_E(X) = \frac{\sum_\tau M_{X,E}^\tau \mathcal{H}(\hat{E}_X^\tau, E_X^\tau)}{\sum_\tau M_{X,E}^\tau + \epsilon}
$$

**Intuition**：object motion 和 wrist trajectory 是 cross-embodiment invariant 的——无论 human hand 还是 robot gripper 抓杯子，杯子移动的轨迹是相似的。把这个作为 supervision 注入 latent action，强迫 $a_X^t$ 编码"杯子从 A 移到 B"这种 task-relevant dynamics，从而抑制"这是 human 视频还是 robot 视频"这种 domain signal 进入 latent action。

#### $\mathcal{L}_{\text{align}}$：Source-Relative Pair-Discriminative Alignment Loss

这是 paper 最重要的算法贡献。对 paired batch $\mathcal{B}_p = \{(H_i, R_i)\}_{i=1}^B$：

$$
f_i^H = \rho(Z_{H_i}), \quad f_i^{R0} = \rho(Z_{R_i,0}), \quad f_i^R = \rho(Z_{R_i})
$$

- $\rho(\cdot)$：video-level pooling（patch 内 spatial average + frame 间 temporal average，然后 $\ell_2$ normalize）
- $f_i^H$：frozen human feature
- $f_i^{R0}$：frozen robot feature（用原始 $F$ 编码）
- $f_i^R$：adapted robot feature（用 $T_\theta$ 编码）

余弦距离 $d(u,v) = 1 - \cos(u,v)$，正对距离 $d_i^+ = d(f_i^R, f_i^H)$。

**Source-Relative (SR) term**：
$$
\ell_{\text{SR}}(i) = [m_s + d_i^+ - d(f_i^{R0}, f_i^H)]_+
$$

- $m_s$：source-relative margin
- $[\cdot]_+ = \max(\cdot, 0)$
- **Intuition**：不强制要求 $f_i^R$ 到 $f_i^H$ 距离为 0（absolute target），只要求 adapted robot feature 到 human 的距离**比 frozen robot feature 到 human 的距离小至少 $m_s$**。这是 relative improvement 的 formulation，避免把所有 paired feature 都 collapse 到一个点，丢失 discriminability。

**Pair-Discriminative (PD) term**：
$$
\bar{d}^{R \to H}(i) = \frac{1}{B-1} \sum_{j \neq i} d(f_i^R, f_j^H), \quad \bar{d}^{H \to R}(i) = \frac{1}{B-1} \sum_{j \neq i} d(f_j^R, f_i^H)
$$

$$
\ell_\alpha(i) = [m_t + d_i^+ - \bar{d}^\alpha(i)]_+, \quad \alpha \in \{R \to H, H \to R\}
$$

$$
\ell_{\text{PD}}(i) = \lambda_{R \to H} \ell_{R \to H}(i) + \lambda_{H \to R} \ell_{H \to R}(i)
$$

- $m_t$：pair-discrimination margin
- $\bar{d}^{R \to H}(i)$：以 robot $i$ 为 anchor，到 batch 内其他 human $j$ 的平均距离（negative samples）
- **Intuition**：triplet loss 变体——要求正对距离 $d_i^+$ 比 negative 平均距离 $\bar{d}^\alpha$ 小至少 $m_t$。这保证 alignment 不会以牺牲 pair-level discrimination 为代价，下游 retrieval 和 policy 学习需要这个 discriminability。

**为什么 SR + PD 比纯 L2 或纯 contrastive 好？** Table 1 给了清晰证据：

| Method | H2R R@1 | R2H R@1 | Avg R@1 |
|---|---|---|---|
| Unadapted | 44.09 | 43.01 | 43.55 |
| HR (HR-Align baseline) | 45.16 | 45.16 | 45.16 |
| HARP-L2 | 70.97 | 52.69 | 61.83 |
| HARP-SR | 84.95 | 64.52 | 74.74 |
| **HARP-SRPD** | **87.10** | **69.89** | **78.50** |

纯 L2 强行 collapse，H2R 很高（70.97）但 R2H 只有 52.69，asymmetric。SR 把 absolute target 改成 relative improvement，对称性变好。SRPD 加上 discrimination 后双向都到 80+ 量级。**单向 retrieval 好而反向差，意味着 robot feature 被 collapse 到 human feature 的一个小 region，丢失了 robot 内部的 discriminability**——PD 正是修这个。

---

## 3. Stage 2: VLA Pretraining with Aligned Representations

用 Stage-1 的 HARP-LAM 给所有 video frame 打 latent action label：

$$
\bar{q}_X^t = Q_\theta(E_\theta(z_X^t, z_X^{t+\Delta t}, l_X))
$$

然后把 HARP-VE（aligned vision encoder）copy 到 VLM 的 vision encoder 里，pretrain 一个 VLA policy $\pi$：

$$
\mathcal{L}_{\text{pretrain}} = -\mathbb{E}_{(x^t, l_X) \sim \mathcal{D}} \left[ \sum_{i=1}^{N_q} \log \pi_\theta(\hat{q}_i = q_{X,i}^t | x^t, l_X) \right]
$$

- $\hat{q}_i$：policy 预测的第 $i$ 个 latent action token
- $q_{X,i}^t$：HARP-LAM 生成的 ground truth latent action
- $N_q = 4$：每 transition 4 个 discrete tokens

**关键设计**：Stage-2 时 vision encoder **freeze**。Table 4 消融显示 freeze 把 CALVIN avg length 从 4.250 提到 4.481，realworld success rate 从 73.3% 提到 76.3%。Intuition 是：Stage-1 学到的 human-robot alignment + web-scale pretrained visual features 是个 fragile 的 sweet spot，Stage-2 如果解冻 vision encoder 让它跟随 latent action loss 调整，会破坏这个 alignment。这跟 OpenVLA-OFT 的 freeze-then-finetune philosophy 一致。

VLA backbone 是 Prismatic-7B（DINOv2+SigLIP fused vision encoder + projector + LLaMA-2 7B LLM），跟 OpenVLA-OFT 同架构。

---

## 4. Stage 3: Finetune with Real Action Head

VLA 预训练完只能输出 latent action tokens，无法直接控制 robot。Stage-3 加一个 lightweight action head，把 latent action embedding 映射到 normalized real action：

- Action head：train from scratch，L1 loss
- VLA backbone：用 LoRA 微调
- 输出：10-step action chunk（跟 OpenVLA-OFT / π₀ 一致）
- Real action labels 只在 Stage-3 用

这个 action head 在 Stage-1/2 完全没参与，类似 LAPA / UniVLA 的 latent-to-real grounding 设计。

---

## 5. Data Pipeline 细节

Table A1 很有信息量：

| Dataset | Embodiment | Type | Frames | S1 | S2 | S3 |
|---|---|---|---|---|---|---|
| HOI4D | Human | Unpaired | 8.9M | ✓ | ✓ | ✗ |
| OpenEgo | Human | Unpaired | 36.4M | ✓ | ✓ | ✗ |
| Bridge-V2 | Robot | Unpaired | 8.6M | ✓ | ✓ | ✗ |
| Ours-Unpaired | DexHand | Unpaired | 3.6M | ✓ | ✓ | ✗ |
| RH20T | H-R | Paired | 8.16M | ✓ | ✓ | ✗ |
| Human2Robot | H-R | Paired | 9.9M | ✓ | ✓ | ✗ |
| Ours-Paired | H-DexHand | Paired | 5.7M | ✓ | ✓ | ✗ |
| CALVIN | Robot/Sim | Real-action | 1.1M | ✗ | ✗ | ✓ |
| Ours-Real | DexHand/Real | Real-action | 0.5M | ✓ | ✓ | ✓ |

**Paired data 才 23.76M frames vs Unpaired 57.5M frames**——paired 是稀缺的 bridge，unpaired 是 scalable supervision。这个比例印证了 paper 的 thesis：paired data 用在 cross-prediction + alignment loss 这种关键 bridge 任务上，unpaired data 用在 self-prediction 这种可扩展的 dynamics learning 上。

### DTW 时间对齐

Paired human-robot video 即使是"同一任务"，执行速度和 subtask 时长也很难一致。HARP 用 wrist keypoint 的 Euclidean distance 作 similarity metric，DTW 找 optimal matching path，以 robot video（更均匀）为 temporal reference，对 human video 做 frame resample（filtering + duplication）。

这步很关键——cross-prediction $\tilde{z}_H^t = z_R^t$ 假设 frame $t$ 在两个 video 里语义对齐，如果时间错位，decoder 会被迫学习"延迟补偿"这种 spurious dynamics，污染 latent action。

### Keypoint Extraction

- **Object keypoint**：Qwen3-VL-8B-Instruct 生成 object description → GroundingDINO 第一帧 localization → TAPIR 跟踪（处理 occlusion）
- **Human wrist**：WiLoR regression MANO 参数，project 到 image plane
- **Robot end-effector**：用 camera extrinsics + 3D wrist pose 做 perspective projection

TAPIR 的选择很有讲究——它用 historical context 推理 occluded 帧，并给 visibility score，可以 attenuate unreliable predictions。这对 manipulation 任务尤其重要，因为 object 几乎不可避免地被 hand/gripper occlude。

---

## 6. 实验结果深度解读

### 6.1 Representation Alignment (Table 1, Fig 4, Fig 5)

- Fig 4 UMAP：F(H) vs F(R) 是两个分离 cluster；F(H) vs T(R) 后 paired samples 拉得很近
- Fig 5 box plot：HARP variants 显著降低 paired cosine distance，HR-Align baseline 在同样的 mean-pooled evaluation space 里**没有降低 paired distance**——这暴露了 HR-Align 的 task-aware pooling 在 evaluation 时被剥离后失效的问题
- Table A3 MRR：HARP-SRPD avg MRR 0.8647 vs unadapted 0.5909

### 6.2 RLBench Frozen Encoder Evaluation (Table 2)

18 tasks × 75 episodes = 1350 evaluation episodes per method。所有 method 用同样的 policy head、action space、data，只换 frozen visual encoder。

| Method | Avg Success Rate |
|---|---|
| Unadapted | 37.56 |
| HR | 39.70 |
| HARP-HR | 35.11 |
| HARP-L2 | 40.78 |
| HARP-SR | 43.41 |
| **HARP-SRPD** | **46.59** |

**+9.03% absolute improvement over unadapted**。注意 HARP-HR（35.11）比 unadapted 还差——这暴露了 task-aware pooling 在 frozen policy head 下游时反而破坏 representation 的现象，是个很有意思的 negative result。

### 6.3 CALVIN ABC→D (Table 4)

CALVIN ABC→D 是 long-horizon manipulation benchmark，要求在 unseen environment D 执行 5-subtask sequence。

| Model | Task1 | Task2 | Task3 | Task4 | Task5 | Avg Len |
|---|---|---|---|---|---|---|
| π₀ | 92.3 | 82.4 | 72.1 | 62.2 | 53.7 | 3.627 |
| π₀.₅ | 94.4 | 86.0 | 76.4 | 69.7 | 61.0 | 3.875 |
| OpenVLA | 91.3 | 77.8 | 62.0 | 52.1 | 43.5 | 3.270 |
| UniVLA | 95.4 | 85.5 | 75.4 | 66.9 | 56.5 | 3.800 |
| OpenVLA-OFT | 94.2 | 86.4 | 78.0 | 70.4 | 62.7 | 3.917 |
| HARP-VLA (L2) | 95.8 | 89.7 | 81.3 | 72.8 | 64.8 | 4.044 |
| HARP-VLA (w/o F.) | 98.8 | 93.9 | 86.1 | 77.7 | 68.5 | 4.250 |
| **HARP-VLA (Ours)** | **99.8** | **96.7** | **91.3** | **84.4** | **75.9** | **4.481** |

HARP-VLA 在 Task5（最难的 long-horizon tail）达到 75.9% vs π₀.₅ 的 61.0%，**+14.9% absolute**。Avg Len 4.481 vs π₀.₅ 3.875，**+0.606**。这个 gap 在 long-horizon 上拉开说明 HARP 学到的 representation 真的在 horizon 累积任务上有持续 benefit，不是只在第一两个 subtask 上 shine。

消融：
- SRPD vs L2：4.481 vs 4.044，**+0.437**——alignment objective 设计的 win
- Freeze vision encoder vs 不 freeze：4.481 vs 4.250，**+0.231**——印证前面的 fragile sweet spot intuition

### 6.4 Realworld (Table 3)

Xarm7 + Robotera XHand，18 DoF (6 arm + 12 hand)，third-view + wrist camera，10-step action chunk。

| Model | Pick | Push | Press | Flip | Avg |
|---|---|---|---|---|---|
| π₀ | 58.3 | 75.0 | 56.7 | 35.0 | 56.3 |
| π₀.₅ | 71.7 | 83.3 | 68.3 | 53.3 | 69.2 |
| OpenVLA | 0.0 | 23.3 | 31.7 | 18.3 | 18.4 |
| UniVLA | 38.3 | 61.7 | 21.7 | 32.5 | 38.4 |
| OpenVLA-OFT | 51.7 | 71.7 | 76.7 | 43.3 | 60.9 |
| HARP-VLA (L2) | 70.0 | 71.7 | 81.7 | 56.7 | 70.0 |
| HARP-VLA (w/o F.) | 76.7 | 80.0 | 78.3 | 58.3 | 73.3 |
| **HARP-VLA (Ours)** | **76.7** | **81.7** | **85.0** | **61.7** | **76.3** |

**76.3% vs OpenVLA-OFT 60.9% = +15.4%**；vs π₀.₅ 69.2% = **+7.1%**。Flip Cup（最难的 dexterity 任务）上 61.7% vs π₀.₅ 53.3%，这是 dexterous manipulation 的真正 test。

---

## 7. 我的一些 Intuition 和疑问

**正向**：
1. **Joint alignment + LAM learning** 这件事是对的。Pipeline 式"先 align visual 再学 latent action"会在 latent action 里重新引入 domain discrepancy，因为 LAM objective 没有 alignment constraint。HARP 的 cross-prediction 把 alignment 直接 bake 进 LAM training，这个 coupling 很 elegant。
2. **Robot-only adapter + zero-init** 是 compute-efficient 的选择。全量 fine-tune DINOv2+SigLIP 风险大、显存重，adapter 让 alignment 既可以 expressive（每层都有）又可以 safe init。
3. **Freeze human branch** 的 asymmetry 我很 buy。VLM 是 human-centric pretrain 的，human representation 已经跟 LLM 对齐；动 human branch 会破坏这个 alignment，且没什么可学的（human data 本来就是 "teacher"）。

**疑问 / 可能的 concern**：
1. **Codebook size K=16 很小**。LAPA 用 8192，UniVLA 也类似量级。HARP 用 16 我猜测是因为 cross-prediction 的 constraint 让每个 code 必须能解释两个 embodiment 的 dynamics，codebook 大了反而难收敛。但 16 个 code 是否足以表达 manipulation 的全部 motion primitive？这个我有点怀疑，特别是 long-horizon task 上。
2. **Paired data 依赖性**。Paper 说 paired data 是 bridge，但 paired data 收集成本仍然高（需要 human 和 robot 同时做同任务）。RH20T、Human2Robot 加自采才 23.76M frames，相比 unpaired human data 45.3M frames 量级差不多。Limitation section 也承认 performance depends on bridge data diversity。如果 paired data 完全没有，HARP 退化成只有 self-prediction，那时 SRPD loss 无法计算，只剩 LAM + aux——这跟 UniVLA 还差多少？这个 ablation paper 没做。
3. **Auxiliary cue 的 noise**。TAPIR 在 occlusion 严重时 visibility score 不可靠，WiLoR 在 fast motion 下 MANO regression 会有抖动。这些 noise 会通过 L_aux 污染 latent action。Paper 用 visibility mask + Huber loss 缓解，但没量化 cue noise 对 final policy 的影响。
4. **Realworld 实验规模**。每 task 60 trials，4 task，共 240 trials。对比 π₀.₅ 这种 industrial-scale 训练的模型，HARP-VLA 的 win 是否在统计意义上 robust？标准差没报告。Flip Cup 上 +8.4% (61.7 vs 53.3)，如果 trial 间方差大，这个 gap 可能不显著。

---

## 8. 相关工作和扩展阅读

HARP 站在一个很 active 的研究方向上。我整理了几个紧密相关的工作：

**Latent Action Models**：
- LAPA: https://arxiv.org/abs/2410.11758 — Ye et al., 首篇大规模 latent action pretraining for VLA
- UniVLA: https://arxiv.org/abs/2505.06111 — Bu et al., task-centric latent actions，HARP 的 LAM 架构基础
- UniSkill: https://arxiv.org/abs/2505.08787 — Kim et al., cross-embodiment skill representations
- IGOR: https://arxiv.org/abs/2412.12010 — Chen et al., image-goal representations as control units

**Cross-Embodiment Human-Robot Alignment**：
- EgoMimic: https://arxiv.org/abs/2410.23664 — egocentric video scaling，visual alignment cues
- HR-Align: https://arxiv.org/abs/2404.13345 — Zhou et al., HARP 主要 baseline
- MimicDreamer: https://arxiv.org/abs/2509.22199 — Li et al., human-robot demo alignment for VLA
- DexUMI: https://arxiv.org/abs/2505.21864 — Xu et al., human hand as universal manipulation interface
- MimicPlay: https://arxiv.org/abs/2308.11546 — Wang et al., human play video for long-horizon imitation
- Im2Flow2Act: https://arxiv.org/abs/2407.15208 — object flow as cross-domain interface
- Dream2Flow: https://arxiv.org/abs/2512.24766 — video generation → 3D object flow for manipulation

**VLA Foundation Models**：
- OpenVLA: https://arxiv.org/abs/2406.09246 — Kim et al., open-source VLA
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645 — fine-tuning speed + success
- π₀: https://arxiv.org/abs/2410.24164 — Black et al., flow matching VLA
- π₀.₅: https://arxiv.org/abs/2504.16054 — open-world generalization VLA
- RDT-1B: https://arxiv.org/abs/2410.07864 — diffusion foundation for bimanual
- Octo: https://arxiv.org/abs/2405.12213 — open-source generalist policy

**Visual Encoders / VLM Backbones**：
- Prismatic VLMs: https://arxiv.org/abs/2402.07865 — Karamcheti et al., fused vision encoder design space
- DINOv2: https://arxiv.org/abs/2304.07193 — Oquab et al., self-supervised visual features
- SigLIP: https://arxiv.org/abs/2303.15343 — sigmoid loss for image-language pretraining

**Tracking / Hand Pose 估计**：
- TAPIR: https://arxiv.org/abs/2306.08630 — Doersch et al., any-point tracking with occlusion handling
- WiLoR: https://arxiv.org/abs/2409.12259 — 3D hand localization + reconstruction in-the-wild
- MANO: https://arxiv.org/abs/2201.02610 — Romero et al., hand body model

**Benchmarks**：
- CALVIN: https://arxiv.org/abs/2112.03227 — Mees et al., long-horizon language-conditioned manipulation
- RLBench: https://arxiv.org/abs/1909.12271 — James et al., 18-task manipulation benchmark

**关键技术 components**：
- VQ-VAE: https://arxiv.org/abs/1711.00937 — van den Oord et al., discrete latent representation
- LoRA: https://arxiv.org/abs/2106.09685 — Hu et al., low-rank adaptation
- DTW: https://en.wikipedia.org/wiki/Dynamic_time_warping
- GroundingDINO: https://arxiv.org/abs/2303.05499 — Liu et al., open-set detection
- Qwen3-VL: https://arxiv.org/abs/2505.09388 — multimodal LLM for object description

---

## 9. 总结性的 intuition

如果让我一句话概括 HARP 的核心 insight：**latent action 的 embodiment-agnostic 性质是从 visual representation 的 embodiment-agnostic 性质继承来的，所以这两个必须 joint train**。

整个 paper 的算法设计都在服务这个 thesis：
- Cross-prediction 把 alignment 烙进 LAM objective
- SRPD loss 保证 alignment 不破坏 discriminability
- Aux cue 把 latent action 锚定在 embodiment-invariant 的 object motion 上
- Freeze human branch 保持 VLM 的 vision-language alignment 不动
- Freeze vision encoder in Stage-2 保持 Stage-1 学到的 sweet spot

这种"在恰当位置 freeze、在恰当位置 align、在恰当位置 ground"的层次化设计，是工程上很扎实的部分。我觉得这篇 paper 的算法 contribution 比 π₀.₅ 这种 scale-up 更"可解释"——你能 trace 每个 design choice 到一个具体 failure mode。当然代价是 paired data 的依赖，这是 HARP 真正的 bottleneck，也是 future work 必须解决的问题。

如果接下来要 follow 这个方向，我会想看：能不能用 web-scale 的 paired data（比如 Ego4D + 同任务的 robot demo）替代精心采集的 RH20T-style paired data？或者用 synthesized paired data（video generation 模型把 human video 渲染成 robot video）？Dream2Flow 已经在 explore 类似 idea，但还没跟 HARP-style joint alignment 结合。这是我个人觉得 robot learning 下半年可能会火的方向之一。
