---
source_pdf: VLA-JEPA.pdf
paper_sha256: 5deec841d5a7da63e5bab3593798e1741d44d354507586bec8a2f9b576f12677
processed_at: '2026-08-13T02:48:37-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VLA-JEPA — 人话版

Andrej，我把上回那篇技术拆解用大白话重讲一遍。学术腔少一点，吐槽多一点，但关键的技术点和公式都留着，因为你要 build intuition，intuition 不能靠空话堆出来。

---

## 一句话讲完这篇 paper

现在的 VLA (Vision-Language-Action) 社区流行一个套路：先用海量的无标注人类视频预训练一个 "latent action" 表示，再 fine-tune 到真机器人上。问题是这套 latent action 学到的东西经常是错的——它学成了 "下一帧长什么样" 的压缩编码，跟机器人真正要控制的那些自由度几乎不沾边。VLA-JEPA 干的事就是：把 LeCun 那套 JEPA 哲学（在 latent space 做预测，target encoder 跟 student pathway 分开，future 信息只当 supervision target 永远不进 input）直接搬进 VLA，让 latent action 只能从 "当前观测 + 世界知识" 里推出来，逼它编码真正的 state transition dynamics，而不是像素变化。

就这么简单。核心 claim 一行：**future frame 只用来算 loss，永远不喂给 student pathway，这样就没有 information leakage 的捷径可走**。

---

## 先吐槽：现有 latent action 方法到底学错了什么

paper Section 1 列了四个 failure mode，我用人话翻译一下：

### 坑 1：pixel-level objective 让 representation 被外观绑架

你用 VQ-VAE 或者 frame reconstruction 当 target，supervision signal 里 variance 最大的就是 texture、lighting、background clutter 这些东西。这些 factor 特别容易预测，loss 下降得快，但它们跟机器人要控制的 DOF (degree of freedom) 几乎不耦合。你的 latent action 学完就是一个 "delta-frame encoder"，编码的是 "下一帧跟这一帧在像素上差多少"，完全没法迁移到真机器人。

### 坑 2：人类手持视频把 nuisance motion 放大

SSv2、Ego4D 这种数据集里，camera shake 的强度经常比 hand-object interaction 引起的 state change 还要大。你的 latent action 自然就被 gradient 推着去编码 camera motion，因为那是数据里最容易 minimize loss 的部分。结果 pretrain 完，latent action 里存的是 "镜头怎么动"，存的全是 camera ego-motion。

### 坑 3：information leakage 让 latent action 坍缩成 shortcut

这条最隐蔽也最致命。LAPA、UniVLA 这些方法架构上有一个共同特征：current observation 和 future observation 都喂进同一个 module，或者 future context 影响 latent action variable 的学习。这种设计给模型留了一个捷径：latent action 直接 encode "future frame 本身长什么样" 就能 minimize loss，完全不需要去理解 "state 怎么 transition"。

Zhang et al. 2025 (https://arxiv.org/abs/2506.15691) 专门做了 probing 实验验证这点——latent action 学到的几乎全是 future observation 的信息。这玩意在 train loss 上好看，迁移到 robot 上就崩，因为它根本没有 "action" 这个概念。

### 坑 4：多阶段 pipeline 脆弱

为了对抗前三个坑，主流方法堆出了三阶段流程：先 representation pretraining，再 latent action learning/alignment，最后 policy learning。每个 stage 引入新的工程 knob，stage 之间 representation 漂移，debug 困难，训练时各种 trick 拼凑。

VLA-JEPA 的 claim：这四个坑本质是同一个根——latent action objective 还隐式锚定在 pixel variation 上。把它换成 latent space 的 leakage-free state prediction，四个坑一起消失。

---

## JEPA 哲学在 VLA 上的对应

你 (Karpathy) 一直在各种 podcast 里讲的那句话——"predict in latent space, not pixel space"——正是这篇 paper 的核心 inductive bias。LeCun 那条线从 I-JEPA (https://arxiv.org/abs/2301.08243) 到 V-JEPA (https://arxiv.org/abs/2311.09079) 再到 V-JEPA2 (https://arxiv.org/abs/2506.09985) 一直在做的事情，VLA-JEPA 把它干净地嫁接到 robot learning 上。

回忆一下 I-JEPA 的 setup：

- **target encoder** 用 future context (图像的右下 patch) 算出 target embedding
- **student encoder** 只看 context patch
- **predictor** 从 context embedding 预测 target embedding
- target encoder 带 stop-gradient，防止 representation collapse

VLA-JEPA 完全对应：

- **target encoder** = frozen V-JEPA2，对 future video clip 编码出 target world state $s_{t_i}$
- **student pathway** = Qwen3-VL，只吃 current observation + language instruction，吐出 latent action tokens $z_{t_i}$
- **world model** (predictor) = auto-regressive Transformer，把 history state + latent action 映射到 future state prediction $\hat{s}_{t_{1:i+1}}$
- 在 latent space 做 L2 alignment loss

一个关键设计：target encoder 和 student pathway 是**两套完全不同的网络**。V-JEPA2 负责定义 "什么是 ground truth 的未来 latent state"，Qwen3-VL 负责在 "仅有 current observation 的条件" 下推断 latent action 应该是什么。latent action 从 future frame 里 "压"出来的这个动作根本不存在——它从 current observation + world knowledge 里 "推"出来，future 只提供 supervision target。

这就是 paper 标题 "leakage-free state prediction" 的字面含义。

---

## 架构逐 block 拆解

paper Figure 1 + Figure 2 给了两张图，我拼起来讲。

### Backbone：Qwen3-VL-2B

VLM 选了 Qwen3-VL-2B (https://arxiv.org/abs/2511.21631)。内部是 Qwen3 LLM (https://arxiv.org/abs/2505.09388) + SigLIP-2 vision encoder (https://arxiv.org/abs/2502.14786)，vision encoder 是 ViT + 3D conv。输入图像 resize 到 224×224。

VLA-JEPA 对 Qwen3-VL 做了一个非平凡的改造：在 vocabulary 里加了两个新的 learnable token type：

- `⟨latent_i⟩` — 第 $i$ 个时间步的 latent action token。输入 sequence 里同一个 `⟨latent_i⟩` 被复制 $K = 24/T$ 次，$T$ 是 future video horizon (默认 8)，所以 $K = 3$。token replication 借鉴了 OpenVLA-OFT (https://arxiv.org/abs/2509.21558) 和 π0-Fast (https://arxiv.org/abs/2509.04785)，目的是在 attention 里给 latent action 多一些"票数"，防止被 image token 淹没。
- `⟨action⟩` — embodied action token，放在 latent action tokens 之后，作为 flow-matching head 的 conditioning signal。

### World State Encoder：V-JEPA2 + multi-view concat

公式 (1)：

$$s_{t_i} = \big\Vert_v F(I_{v,t_i})$$

变量含义：

- $I_{v,t_i}$ — 视角 $v$ 在时间 $t_i$ 的图像 frame
- $F(\cdot)$ — 单视角 video encoder，paper 里就是 frozen V-JEPA2
- $\Vert$ — 向量 concatenation
- $s_{t_i}$ — 聚合多视角之后的 unified world state 表示

**直觉**：V-JEPA2 在 self-supervised 阶段已经通过 masking + latent alignment 学到了 "丢弃 pixel-level noise，保留 semantic + motion factor" 的 invariance 性质。直接拿它当 target encoder，等于把 JEPA 的 invariance 性质继承过来。这就是 paper 反复讲的 "robust to camera motion and background changes" 的来源。

multi-view 用 concat 这个保守选择，简单且可扩展。当 view 数 < 2 时复制一遍，> 2 时随机选 2 个 (Appendix A.2 写明)。

### Latent Action Pathway：VLM 输出 latent tokens

公式 (2)：

$$z_{t_i} = p_\theta^{VLM}\big(\langle \text{latent}_i \rangle \;\big|\; \{I_{j,t_0}\}_{j=0}^{v}, \ell\big)$$

变量含义：

- $p_\theta^{VLM}$ — Qwen3-VL 在参数 $\theta$ 下定义的条件分布
- $\langle\text{latent}_i\rangle$ — 第 $i$ 个 learnable latent action token，复制 $K$ 次进入 input
- $\{I_{j,t_0}\}_{j=0}^{v}$ — 各个视角在 initial time step $t_0$ 的图像
- $\ell$ — language instruction
- $z_{t_i}$ — VLM 在 latent action token 位置上输出的 hidden state，被解释为 "第 $i$ 时间步的 latent action 表示"

**关键点**：student pathway 这里只看到 $t_0$ 的 observation，future frame 根本不喂进来。future frame 仅通过 world state encoder 进入损失，损失只施加在 predictor 的输出上。这就切断了 leakage 的捷径。

### Latent World Model：auto-regressive Transformer

公式 (3)：

$$\hat{s}_{t_{1:i+1}} = p_\theta^{WM}(s_{t_{0:i}}, z_{t_{0:i}})$$

变量含义：

- $p_\theta^{WM}$ — 世界模型，12 层 Transformer，8 个 attention head，image token dim 2048，每时间步 256 image token，3 个 action token，2 个 view，future horizon 默认 8 (Table 5)
- $s_{t_{0:i}}$ — 已观测到的 history world state sequence
- $z_{t_{0:i}}$ — 已生成的 latent action sequence
- $\hat{s}_{t_{1:i+1}}$ — 预测的下一 chunk world state，horizon 从 $t_1$ 到 $t_{i+1}$

attention mask 设计讲究：

- 同一时间步内：K 个 latent action token 和 N 个 world state token 之间 bidirectional full attention
- 跨时间步：严格 causal — 时间 $t$ 的 token 只能 attend 到 $\le t$ 的 token

"frame-level 内部 bidirectional + frame 之间 autoregressive" 的 spatiotemporal 因式分解，跟 V-JEPA2 的 tube masking 哲学一致，只不过这里有显式 action tokens 作为 condition。

### JEPA ELBO 视角

公式 (4) 把整个 objective 放到 ELBO 框架里：

$$\log p(s_{t_{1:T}} \mid z_{t_{0:T-1}}) \ge \sum_{k=1}^{T} \mathbb{E}_{s_{t_k} \sim F(\cdot)} \big[\log p_\theta(\hat{s}_{t_k} \mid s_{t_k})\big] - D_{KL}\big[F(\cdot) \,\|\, p_\theta^{WM}\big]$$

变量含义：

- $F(\cdot)$ — frozen target encoder (V-JEPA2)，带 stop-gradient
- $p_\theta^{WM}$ — online predictor (world model)
- $s_{t_k}$ — target encoder 在时间 $t_k$ 算出的 ground truth latent
- $\hat{s}_{t_k}$ — world model 预测
- $D_{KL}[F(\cdot) \| p_\theta^{WM}]$ — target encoder 分布和 predictor 分布之间的 KL divergence

因为 $F(\cdot)$ 是 deterministic forward (V-JEPA2 没有 stochasticity)，KL 项退化为 0，ELBO 退化成 latent space 里的 reconstruction loss。这就给了公式 (5)：

$$\mathcal{L}_{WM} = \sum_{k=1}^{T} \mathbb{E}_{s_{t_k} \sim F(\cdot)} \,(\hat{s}_{t_k} - s_{t_k})$$

(原文没显式写 $\|\cdot\|_2$ 符号，排版省了，但根据标准 JEPA loss 实际是 L2 distance。)

### Flow-Matching Action Head：DiT-B

下游 robot data (带 action label) 时，加一个 action prediction head。paper 选 conditional flow matching (https://arxiv.org/abs/2210.02747) + DiT-B (https://arxiv.org/abs/2212.09748)。跟 π0 (https://arxiv.org/abs/2410.24164) 选 flow matching 是同一个 reasoning——action space 上比 discrete diffusion 连续，训练也稳定。

公式 (7) 定义插值轨迹：

$$a_t = (1-t)\epsilon + t\, a_{0:H}, \quad t \sim \mathcal{U}(0,1)$$

变量含义：

- $a_{0:H}$ — ground-truth action sequence，horizon $H$ (Table 6 是 7)
- $\epsilon \sim \mathcal{N}(0, I)$ — Gaussian noise
- $t$ — flow time，$[0,1]$ 均匀采样；$t=0$ 时 $a_t = \epsilon$ (纯噪声)，$t=1$ 时 $a_t = a_{0:H}$ (真值)

公式 (8) 是 flow-matching 训练目标：

$$\mathcal{L}_{FM} = \mathbb{E}_{a_{0:H}, \epsilon, t}\Big[\big\|v_\theta(a_t, t \mid z_a) - (a_{0:H} - \epsilon)\big\|_2^2\Big]$$

变量含义：

- $v_\theta(\cdot)$ — DiT-B 参数化的 velocity field，以 $z_a$ 为 condition
- $(a_{0:H} - \epsilon)$ — ground-truth velocity (从插值 $a_t = (1-t)\epsilon + t\,a_{0:H}$ 对 $t$ 求导就是 $a_{0:H} - \epsilon$)
- $z_a$ — 公式 (6) 算出来的 action-conditioning representation

公式 (6)：

$$z_a = p_\theta^{VLM}\big(\langle\text{action}\rangle \mid \{I_{i,t_0}\}_{i=0}^{v}, \ell, \langle\text{latent}_i\rangle\big)$$

**关键**：conditioning $z_a$ 已经把 visual obs + language + latent action sequence 三个信号通过 Qwen3-VL 的 causal attention 融在一起。`⟨action⟩` 在 sequence 末尾，能看到所有 latent action tokens——这相当于让 action head 显式被 "推断出的 latent action plan" 条件化。

最终联合 objective (公式 9)：

$$\mathcal{L} = \mathcal{L}_{FM} + \beta\, \mathcal{L}_{WM}$$

$\beta$ 是 tunable weight。robot data 上同时算两个 loss：既要预测未来 latent state (保持 world model 一致性)，也要生成真实 action。这是 paper Section 3.3 标题 "Joint Optimization Objectives" 的来源。

---

## Pretraining → Fine-tuning 流程

### Pretraining 数据

- **SSv2** (Something-Something v2, https://arxiv.org/abs/1706.04261)：220K human action videos，无 action label，只用 $\mathcal{L}_{WM}$
- **Droid** (https://arxiv.org/abs/2403.12945)：76K high-quality robot demonstrations，带 action label，用 $\mathcal{L}_{FM} + \beta\mathcal{L}_{WM}$

**重点**：这两个 dataset 在 pretraining 阶段是**同时**训练的 (jointly train 50K steps)，没有 "先 SSv2 再 Droid"。这跟 LAPA / UniVLA / villa-X 通常的多阶段 pipeline 不一样，这就是 paper Section 1 强调的 "streamlined two-stage pipeline" (pretrain → fine-tune) 的具体体现。

### Fine-tuning 数据

- LIBERO (https://arxiv.org/abs/2306.03310)：~2K expert demos，30K steps
- SimplerEnv (https://arxiv.org/abs/2410.24185)：Fractal + BridgeV2 datasets
- Real-world：100 demos，3 个 pick-and-place 任务，20K steps

### 训练配置

- 8× NVIDIA A100
- batch size 32 per GPU → global 256
- cosine LR + linear warmup
- peak LR：VLM 和 world model 1e-5，action head 1e-4
- image input 224×224，video clip 给 world state encoder 256×256

action 表示：end-effector delta position + delta axis-angle，各自 min-max normalize 到 [0,1]；gripper command binarize 成 {0,1}。跟 π0 的 joint-space delta 不同——VLA-JEPA 选 end-effector 因为它跟 latent action 在 task space 上更对齐。

---

## 实验结果

### LIBERO (Table 1)

LIBERO 有 4 个 suite：Spatial, Object, Goal, LIBERO-10。每个 task 跑 50 episodes，每 suite 500 episodes。

| Method | Spatial | Object | Goal | LIBERO-10 | Avg |
|---|---|---|---|---|---|
| LAPA | 73.8 | 74.6 | 58.8 | 55.4 | 65.7 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π0-Fast | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| villa-X | 97.5 | 97.0 | 91.5 | 74.5 | 90.1 |
| GR00T N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| **VLA-JEPA** | 96.2 | **99.6** | 97.2 | **95.8** | **97.2** |
| w/o human videos | 94.8 | 99.6 | 95.8 | 94.0 | 96.1 |

观察：

- VLA-JEPA 在 Object 和 LIBERO-10 上拿第一。Object 上 99.6 是全场最高，说明 latent world modeling 在 long-horizon (LIBERO-10 是 10 步 horizon 任务) 上有显著优势
- 比 OpenVLA-OFT (97.1) 高 0.1，比 π0.5 (96.9) 高 0.3。差距小，但 OpenVLA-OFT 和 π0.5 用了更大的 robot dataset pretraining，VLA-JEPA 只用 Droid (76K) + SSv2 (220K)
- "w/o human videos" 降到 96.1，说明在 LIBERO 这种 ID (in-distribution) 设定下，SSv2 帮助有限，高质量 expert demo 主导

### SimplerEnv (Table 2)

SimplerEnv 是 real-to-sim 的 OOD 设定，有 Google Robot 和 WidowX Robot 两套。

| Method | Pick | Move | Drawer | Place | Avg(G) | Spoon | Carrot | Block | Eggplant | Avg(W) |
|---|---|---|---|---|---|---|---|---|---|---|
| LAPA* | – | – | – | – | – | 70.8 | 45.8 | 54.2 | 58.3 | 57.3 |
| villa-X | 81.7 | 55.4 | 38.4 | 4.2 | 44.9 | 48.3 | 24.2 | 19.2 | 71.7 | 40.8 |
| UniVLA | – | – | – | – | – | – | – | – | – | 42.7 |
| RoboVLMs | 77.3 | 61.7 | 43.5 | 24.1 | 51.7 | 45.8 | 20.8 | 4.2 | 79.2 | 37.5 |
| GR00T N1 | 0.7 | 1.9 | 2.9 | 0.0 | 1.4 | 1.4 | 0.0 | 0.0 | 13.9 | 3.8 |
| π0 | 72.7 | 65.3 | 38.3 | – | – | 29.1 | 0 | 16.6 | 62.5 | 40.1 |
| π0-Fast | 75.3 | 67.5 | 42.9 | – | – | 29.1 | 21.9 | 10.8 | 66.7 | 48.3 |
| **VLA-JEPA** | **88.3** | 64.1 | **59.3** | **49.1** | **65.2** | **75.0** | **70.8** | 12.5 | **70.8** | **57.3** |
| w/o human videos | 85.3 | 66.7 | **75.5** | **86.1** | **78.4** | 75.0 | 54.2 | 20.8 | 79.2 | 57.3 |

观察：

- Google Robot：VLA-JEPA 65.2 第一，比 villa-X (44.9) 高 20 个点
- WidowX：VLA-JEPA 57.3 跟 LAPA* 并列第一。LAPA* 是 LAPA 用 SimplerEnv 里的成功 rollout 当 expert 来训练 (一种 oracle 设置)，VLA-JEPA 用更朴素的训练达到了 LAPA 的 oracle 性能
- "w/o human videos" 在 SimplerEnv 上反而更高 (78.4 vs 65.2 on Google Robot)。paper 在 Section 4.5 Q1 给了解释：SSv2 的 human video 缺乏 robot 的物理 action trajectory 信息，在 real-to-sim 这种需要精确 action 的场景反而引入 noise

### LIBERO-Plus (Table 3) — paper 的亮点

LIBERO-Plus (https://arxiv.org/abs/2510.13626) 系统性地在 7 个维度扰动 LIBERO：Camera, Robot, Language, Light, Background, Noise, Layout。

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | Avg |
|---|---|---|---|---|---|---|---|---|
| UniVLA | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π0 | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π0-Fast | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| **VLA-JEPA** | 63.3 | **67.1** | **85.4** | **95.6** | **93.6** | 66.3 | **85.1** | **79.5** |
| w/o human videos | 40.3 | 55.7 | 72.9 | 88.2 | 70.5 | 38.2 | 74.6 | 62.9 |

观察：

- VLA-JEPA 在 5/7 个扰动维度拿第一，平均 79.5，比 OpenVLA-OFT (69.6) 高 10 个点，比 π0 (53.6) 高 26 个点
- Camera / Robot / Noise 三个扰动上没拿第一。Camera 和 Robot 上 π0-Fast 比较好，Noise 上 OpenVLA-OFT 比较好
- Language / Light / Background / Layout 上压倒性领先
- 关键 ablation："w/o human videos" 掉到 62.9，比 VLA-JEPA (79.5) 少了 16.6 个点——这是 paper 关于 "human video 主要贡献是 robustness" 这一论点的硬证据

paper Figure 5 进一步画了 "human video 比例 vs 各扰动维度 success rate" 曲线，显示随 human video 比例增加，Language / Light / Background / Layout 这四条曲线单调上升，Camera / Robot / Noise 这三条平或略降。这个图非常 clean 地说明：human video 教给 model 的是 "task-agnostic 的视觉/语义稳定性"，robot-specific 的扰动 (camera extrinsic / robot embodiment) 还得靠 robot data。

### Real-World Franka (Figure 4 + Appendix B)

- Setup：Franka Research 3 + Robotiq 2F-85 + 3× Intel RealSense D435 (2 third-person + 1 wrist)
- 100 demos，3 个 pick-and-place (葡萄/苹果/芒果/橙子 → 盘/碗)
- 评估两种 OOD：task OOD (新任务) 和 object-layout OOD (训练任务但布局乱)
- baseline：π0 和 π0.5 在同一 demo 集上 fine-tune

观察：

- ID 和 object-layout OOD 上 VLA-JEPA 最好
- task OOD 上 VLA-JEPA 第二 (π0.5 最好)
- π0.5 在 task OOD 上 instruction-following 更准，但 position control 会突破 safety boundary，导致执行失败
- VLA-JEPA 在 grasp 失败时会主动 re-open gripper 重抓，π0 和 π0.5 都不会。paper 把这个归因于 SSv2 里有大量人类反复尝试的 grasp 行为，VLA-JEPA 学到了 "when to regrasp" 这种 temporal decision，然后内部 map 到自己的 robot dynamics

regrasp 这个观察其实蛮有意思——regrasp 是一个 temporal decision (什么时候放弃当前 grasp 重来)，它不需要新的 low-level dynamics 知识，只需要 "失败 → 退后 → 再试" 这个时序 pattern。human video 提供了大量这种 pattern。

### Ablation：future video horizon $T$ (Table 4)

| T | Spatial | Object | Goal | 10 | Avg |
|---|---|---|---|---|---|
| 4 | 95.0 | 99.2 | 95.8 | 89.0 | 94.8 |
| 8 | 94.8 | 99.8 | 95.8 | 94.0 | **96.1** |
| 16 | 92.8 | 98.8 | 98.0 | 92.2 | 95.5 |

观察：

- $T = 8$ 最好，跟 action horizon $H = 7$ 接近匹配
- $T = 4$ 信息不足，LIBERO-10 (long horizon) 掉得最多
- $T = 16$ 太长，引入冗余——Goal suite (任务简单) 受益，但 Spatial suite (细操作) 受损。latent action 跟 action horizon 应该 co-design，过长会注入 irrelevance

### Attention 可视化 (Figure 6)

paper 把 LAPA / UniVLA / VLA-JEPA 三个 model 的 latent action token 对 image token 的 attention map 可视化，在三种 input (sim / human video / real robot) 上对比。结论：

- LAPA 的 latent action attention 很 dense，关注桌面无关物体——这是 information leakage 的征兆：latent action 退化成 "future frame 的压缩"，所以它要 attend 到所有可能在下一帧变化的区域
- UniVLA 通过 language guidance 缓解了，但过于聚焦 semantic——会 attend 到 stationary pen、tablecloth texture 这种跟 task 无关但 semantic 醒目的东西
- VLA-JEPA 最聚焦：robotic arm / hand / target object。leakage-free + latent space prediction 的 inductive bias 起作用了——未来信息没进来，latent action 只能去关心 "操作上 causally relevant" 的部分

---

## 我的几个直觉

### 直觉 1：为什么 "leakage-free" 这么有效——信息论视角

考虑 latent action $z$ 和 future observation $O_{future}$ 的互信息 $I(z; O_{future})$。

LAPA / UniVLA 里，$z$ 由 $O_{current}$ 和 $O_{future}$ 共同决定 (VQ-VAE target 是 $O_{future}$，$z$ 要重建 $O_{future}$)，$I(z; O_{future})$ 被最大化到 "几乎 $z$ 编码了 $O_{future}$ 全部信息" 的程度。但其中真正跟 action causal 的部分 $\Delta s = s_{t+1} - s_t$ 只是 $O_{future}$ 的一个子流形。$z$ 浪费了大部分 capacity 在 nuisance factor (texture, lighting, background) 上。

VLA-JEPA 里，$z$ 只能从 $O_{current}$ 推出来 (公式 2 没喂 future)，$I(z; O_{future})$ 被压在 "能从 $O_{current}$ 推出的部分"，supervision target 是 latent state $\hat{s}_{t_k}$ 而非 pixel。这迫使 $z$ 编码 "哪些 factor 在 latent state 上会变化"，自然就跟 controllable DOF 对齐。

这其实是 information bottleneck 的 implicit 实现——把 future 从 input 里去掉，等于在 student pathway 上加了 "future-blind" 约束。

### 直觉 2：latent space prediction 比 pixel space prediction 鲁棒的关键

关键在 V-JEPA2 这个 target encoder 的 invariance 性质。V-JEPA2 在 self-supervised 阶段已经通过 masking + latent alignment 学到了 "丢弃 pixel-level noise，保留 semantic + motion factor"。所以 target latent $s_{t_k}$ 本身已经是 "denoised" 的。Student pathway 预测 $s_{t_k}$ 时，即使 background 变化，只要 background 不进入 V-JEPA2 的 semantic representation，$\hat{s}$ 也不需要预测它。这等价于 paper Section 1 强调的 "robust to camera motion and background changes"。

但这里有个 subtlety：V-JEPA2 的 invariance 是它自己 pretraining 数据决定的。如果 V-JEPA2 没在某类 appearance shift 上 invariance，VLA-JEPA 也不会有。VLA-JEPA 的 robustness 上限被 V-JEPA2 的 robustness 上限 ceiling 住了。这是个隐式 assumption，paper 没显式讨论。

### 直觉 3：human video 在 LIBERO 上几乎没用的反思

Table 1 "w/o human videos" 96.1 vs VLA-JEPA 97.2，差距只有 1.1 个点。Table 2 上甚至 "w/o human videos" 更高 (Google Robot 78.4 vs 65.2)。这说明在 ID 和 real-to-sim 这种 "task-specific skill 主导" 的场景，SSv2 的贡献很有限。

为什么？我怀疑是 SSv2 的 action 跟 robot action 太不共享了。SSv2 是人手操作物体，robot 是 gripper 夹物体，两者在 end-effector geometry、contact dynamics、force profile 上差太多。VLA-JEPA 只在 latent state space 上 transfer，所以 transfer 的是 "world dynamics"——但 LIBERO 的 dynamics 很简单 (刚体、无摩擦变异、固定光照)，没什么可 transfer 的。

LIBERO-Plus 上 human video 才大放异彩，因为扰动主要在 visual appearance 上 (Light / Background / Layout / Language)，这恰好是 V-JEPA2 pretrain + SSv2 fine-tune 学到的 invariance 最有用的地方。

这给我一个 intuition：**human video pretraining 的价值在于教 model "哪些视觉变化可以忽略"**。这跟 LLM 里 "pretraining 教语法，SFT 教风格" 的某种类比——pretraining 教 invariance，SFT 教 specific skill。

### 直觉 4：regrasp 现象背后的东西

paper 在 real-world 实验里观察到一个细节：VLA-JEPA 在 grasp 失败时会主动 re-open gripper 重试，π0 和 π0.5 都不会。paper 把这个归因于 SSv2 里有大量人类反复尝试的 grasp 行为。

我往深一层想：regrasp 本质上是一个 temporal decision——什么时候放弃当前 grasp 重来。它不需要新的 low-level dynamics 知识 (gripper 开合的物理都是已知的)，只需要 "失败 → 退后 → 再试" 这个时序 pattern。

human video 提供了大量这种 pattern，但 robot data 里很少有 "失败后重试" 的 demonstration (因为采集数据时 demo 都尽量做成功)。所以 VLA-JEPA 在 latent world modeling 阶段学到了 regrasp 的时序结构，fine-tune 时 map 到自己的 gripper dynamics 就能执行。

这个观察其实暗示了一个更大的主题：**human video 教的不只是 visual invariance，还有 "行为时序结构"**——什么时候 retry，什么时候切换策略，什么时候 abort。这些 temporal decision 跟具体 embodiment 无关，可以跨域 transfer。这点 paper 没深挖，但我觉得是 latent world modeling 路线的一个潜在金矿。

---

## 这篇 paper 在 landscape 里的位置

我把它放在你熟悉的几个派系里：

- **π0 / π0.5** (https://arxiv.org/abs/2410.24164, https://arxiv.org/abs/2504.16054)：VLM backbone + flow-matching action head，pretrain 在大规模 robot data 上，没有显式 latent action。VLA-JEPA 跟它共享 flow-matching head 的选择，但 VLA-JEPA 在 VLM 和 head 之间插入了 latent action token + world model
- **LAPA** (https://arxiv.org/abs/2410.11758)：latent action pretraining from video 的代表作。VLA-JEPA 直接 critique 它的 frame-difference + leakage 问题
- **UniVLA** (https://arxiv.org/abs/2505.21672)：task-centric latent action，用 VQ-VAE 离散化。VLA-JEPA 在 LIBERO 上 (Table 1) 超 UniVLA 2 个点，LIBERO-Plus 上超 36.6 个点
- **villa-X** (https://arxiv.org/abs/2503.02065)：unified latent action codebook across human + robot video，需要 multi-stage alignment。VLA-JEPA 在 SimplerEnv 上用 villa-X 1% 的数据量达到 comparable 性能
- **OpenVLA-OFT** (https://arxiv.org/abs/2509.21558)：token replication trick 的来源，VLA-JEPA 借鉴了它
- **Moto / MotoGPT** (https://arxiv.org/abs/2503.17112)：latent motion token + hierarchical pretraining
- **GR00T N1** (https://arxiv.org/abs/2503.14734)：NVIDIA 的人形 foundation model，在 LIBERO 上 93.9，SimplerEnv 上几乎全挂 (Google Robot 1.4，WidowX 3.8)——说明 GR00T N1 的 pretrain data 跟 SimplerEnv 的 visual gap 太大
- **WorldVLA** (https://arxiv.org/abs/2506.26120)：autoregressive action world model，把 next-frame generation 和 action prediction 联合做。在 LIBERO-Plus 上崩了 (25.0)，paper 解释是 pixel-level world model 在扰动下脆弱

VLA-JEPA 的位置：在 "latent action pretraining" 流派内部，但用 JEPA 替换了 frame-difference + VQ-VAE 的核心，避免 leakage。这是把 LeCun 路线 (JEPA) 显式嫁接到 robot learning 上的一个干净实例。

---

## Paper 没充分讨论的 limitation

1. **V-JEPA2 的依赖**：VLA-JEPA 的 robustness ceiling 被 V-JEPA2 决定。如果未来 V-JEPA3 出来，可以替换，但 paper 没做 ablation 验证 "换一个 target encoder 会怎样"
2. **Multi-view 简单 concat**：公式 (1) 用 $\Vert$ 简单 concat 多视角，对 view 数变化不友好。view 数从 2 变到 3，concat 后维度变，模型得重训。cross-attention 或者 view-token 的设计可能更鲁棒
3. **Action head 跟 latent action 解耦**：action head 是 DiT-B，从 $z_a$ condition 出发。$z_a$ 包含了 latent action tokens 的信息，但 action head 仍是独立网络。能否像 π0 那样把 action head 跟 VLM 更深度 entangle，是个 open question
4. **Real-world 实验规模小**：100 demos，3 个任务。π0.5 在 task OOD 上反而比 VLA-JEPA 强 (instruction following 更准)，说明 VLA-JEPA 在 language grounding 上还有 gap
5. **没有 scaling law 实验**：paper 没探究 SSv2 规模从 22K → 220K → 2.2M 时性能曲线。如果 VLA-JEPA 在 internet-scale video 上没有 scaling，JEPA 路线在 VLA 上的故事就还不完整

---

## 几个可能的延伸方向

1. **Replace V-JEPA2 with a robot-pretrained JEPA**：如果 target encoder 本身就在 robot 数据上 pretrain 过，target latent $s_{t_k}$ 会更 action-relevant。可能能解释为什么 human video 在 LIBERO 上没用——target encoder 太"human-centric"
2. **Hierarchical latent action**：目前 latent action 是 frame-level (每帧一个 `⟨latent_i⟩`)。可以做 coarse-to-fine：高层 latent action 描述 subgoal，低层描述 frame transition
3. **Joint train target encoder**：paper 把 V-JEPA2 freeze 了。如果让 target encoder 跟 student 一起更新 (用 EMA 或 stop-gradient)，可能让 target latent 更适配 robot domain。风险是 collapse
4. **Latent action consistency across embodiment**：让同一个 latent action 在 human 和 robot 上有相同语义 (e.g. "approach object")，需要 contrastive or alignment loss。villa-X 走了这条路但用了 multi-stage，VLA-JEPA 能否 single-stage 做
5. **Replace flow-matching with discrete diffusion**：最近 Discrete Diffusion VLA (https://arxiv.org/abs/2505.07840) 显示离散 diffusion 在 action 上也有竞争力。VLA-JEPA 的 latent action 是连续的，跟 discrete diffusion 不直接兼容，但可以探究
6. **Test-time scaling**：VLA-JEPA 推理时是 single-pass。能否在 inference 时用 world model 做 planning——sample 多个 latent action 候选，用 world model 预测未来 state，挑最好的？这其实跟 Value-Guided JEPA Planning (https://arxiv.org/abs/2506.15862) 的思路一致

---

## Reference 链接汇总

paper 本身：
- VLA-JEPA GitHub: https://github.com/ginwind/VLA-JEPA/
- VLA-JEPA Project Page: https://ginwind.github.io/VLA-JEPA/
- VLA-JEPA HuggingFace: https://huggingface.co/ginwind/VLA-JEPA/

核心依赖方法：
- V-JEPA2: https://arxiv.org/abs/2506.09985
- V-JEPA: https://arxiv.org/abs/2311.09079
- I-JEPA: https://arxiv.org/abs/2301.08243
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- Qwen3: https://arxiv.org/abs/2505.09388
- SigLIP-2: https://arxiv.org/abs/2502.14786
- Flow Matching: https://arxiv.org/abs/2210.02747
- DiT: https://arxiv.org/abs/2212.09748

对比 baseline：
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- π0-Fast: https://arxiv.org/abs/2509.04785
- OpenVLA: https://arxiv.org/abs/2406.09246
- OpenVLA-OFT: https://arxiv.org/abs/2509.21558
- LAPA: https://arxiv.org/abs/2410.11758
- UniVLA: https://arxiv.org/abs/2505.21672
- villa-X: https://arxiv.org/abs/2503.02065
- Moto / MotoGPT: https://arxiv.org/abs/2503.17112
- GR00T N1: https://arxiv.org/abs/2503.14734
- WorldVLA: https://arxiv.org/abs/2506.26120
- CoT-VLA: https://arxiv.org/abs/2503.02065
- RoboVLMs: https://arxiv.org/abs/2412.15071

数据集：
- Droid: https://arxiv.org/abs/2403.12945
- Something-Something v2: https://arxiv.org/abs/1706.04261
- Ego4D: https://arxiv.org/abs/2110.07058
- LIBERO: https://arxiv.org/abs/2306.03310
- LIBERO-Plus: https://arxiv.org/abs/2510.13626
- SimplerEnv: https://arxiv.org/abs/2410.24185

诊断性参考：
- "What do latent action models actually learn?": https://arxiv.org/abs/2506.15691
- Value-Guided Action Planning with JEPA World Models: https://arxiv.org/abs/2506.15862

---

## 最后一句

VLA-JEPA 把 LeCun 那套 "predict in latent space + asymmetric encoder + stop-gradient + leakage-free" 哲学干净地搬进 VLA pretraining，通过诊断并修复 latent-action 流派的四个 failure mode (pixel bias / nuisance motion / info leakage / multi-stage fragility)，用一个 streamlined 的两阶段流程 (joint pretrain on SSv2+Droid → fine-tune on task data) 在 LIBERO / LIBERO-Plus / SimplerEnv / real-world 上都拿到了 comparable-to-SOTA 的性能，在 robustness 扰动上显著领先。最深的 insight 是它把 "human video 的作用" 重新定义了——human video 不教新技能，教的是 visual invariance 和行为时序结构，这些在 task-specific 评估 (LIBERO) 上看不出来，在 robustness 评估 (LIBERO-Plus) 上才显化。

---

# VLA-JEPA 深度解读 — 给 Karpathy 的直觉版 walkthrough

Andrej, 这篇 paper 的核心 argument 其实非常对你的胃口 — 它本质上是把你反复强调的 "predict in latent space, not pixel space" 这条哲学 (I-JEPA / V-JEPA / V-JEPA2 路线) 显式地拉进 VLA (Vision-Language-Action) 这个赛道,并且系统性地诊断了当前 latent-action pretraining 流派的四个 "学错了" 的失败模式。让我把它从 intuition → architecture → 数学 → 实验 → 局限 串起来讲。

---

## 1. 这篇 paper 到底在抱怨什么

当前 robot learning 社区里有一股很强的潮流:用 internet-scale 的无标注 human video 来预训练 VLA,典型代表是 LAPA、UniVLA、Moto、villa-X、IGOR、CoMo、StaMo、AdaWorld 这一支。它们的共同 recipe 是:从相邻 frame 之间提取一个 "latent action" (要么离散 codebook via VQ-VAE,要么连续 embedding),然后让 VLA 先预测这个 latent action,再 align 到真实 robot action space。

paper 在 Section 1 列了四个 failure mode,我觉得这是全篇最值钱的部分,因为它把很多人隐约觉得"不对劲"但说不清楚的东西写明白了:

**Failure Mode 1 — Pixel-level objective bias toward appearance**

当你用 frame-difference (或 VQ-VAE 重建 next frame) 的方式定义 latent action 时,supervision signal 被那些 high-variance / low-control 的因素主导:texture、illumination、background clutter、viewpoint drift。这些因素很容易预测,但跟 policy 真正要控制的 DOF (degree of freedom) 几乎不耦合。换句话,你的 latent action 学成了一个 "delta-frame encoder"。

**Failure Mode 2 — In-the-wild video 放大 nuisance motion**

人类手持 camera 拍的视频里,camera ego-motion 强度经常远大于 interaction-induced 的 state change。你的 latent action 自然就被 incentive 去编码 camera shake,因为那是数据中 variance 最大、最容易 minimize loss 的部分。这就是为什么很多 latent action 方法在 SSv2 / Ego4D 上 pretrained 之后,迁移到 robot 上反而变脆。

**Failure Mode 3 — Information leakage → latent action collapse 成 shortcut**

这一条特别关键,而且经常被低估。很多 pipeline 把 current observation 和 future observation 都喂进同一个 module,或者让 future context 影响 latent action variable。这种架构会让模型走捷径:latent action 直接变成 future frame 的压缩 representation,而不是 "解释 state 怎么 transition" 的因子。最近 Zhang et al. 2025 (arXiv:2506.15691, "What do latent action models actually learn?") 用 probing 实验专门证实了这点 — latent action 学到的主要是 future observation 本身。

**Failure Mode 4 — Multi-stage pipeline 脆弱**

为了对抗上面三个问题,主流方法堆出了三 stage pipeline: (i) representation pretraining → (ii) latent action learning/alignment → (iii) policy learning。每个 stage 都引入新的 engineering knob,stage 之间 representation 会漂移,debug 也困难。

VLA-JEPA 的 claim: 这四个问题本质上是同一个根 — latent action objective 还隐式地锚定在 pixel variation 上。把它换掉,改成在 latent space 做 leakage-free 的 state prediction,就一并解决了。

---

## 2. JEPA 哲学在 VLA 上的对应

如果你对 I-JEPA (Assran et al. 2023, https://arxiv.org/abs/2301.08243) 和 V-JEPA (Bardes et al. 2023, https://arxiv.org/abs/2311.09079) 熟悉,这里几乎是搬运:

- I-JEPA: target encoder 用 future context (图像的右下 patch) 算出 target embedding,student encoder 只看 context patch,predictor 预测 target embedding。target encoder stop-gradient,避免 collapse。
- V-JEPA / V-JEPA2 (https://arxiv.org/abs/2506.09985): 同样的故事搬到 spatiotemporal tube 上。
- VLA-JEPA: target encoder 是 frozen V-JEPA2,它对 future video clip 编出 target world state $s_{t_i}$;student pathway 是 Qwen3-VL,它只吃 current observation + language,吐出 latent action tokens $z_{t_i}$;一个 auto-regressive world model $p_\theta^{WM}$ 把 history state + latent action 映射到 future state prediction $\hat{s}_{t_{1:i+1}}$,在 latent space 做 L2 alignment。

注意一个非平凡的设计选择:target encoder 和 student pathway 是两套完全不同的网络。V-JEPA2 是 ViT-based 的 self-supervised video encoder,负责 "什么是 ground truth 的未来 latent state";Qwen3-VL 是 language-aligned 的 multimodal transformer,负责 "在仅有 current observation 的条件下,推断 latent action 应该是什么"。这意味着 latent action 不是从 future frame 里 "压" 出来的,而是从 current observation + world knowledge 里 "推" 出来的,future 只提供 supervision target。

这正是 paper 标题里 "leakage-free state prediction" 的字面含义。

---

## 3. 架构逐 block 拆解

paper Figure 1 + Figure 2 给了两张架构图,我把它们拼起来讲。

### 3.1 VLM Backbone — Qwen3-VL

- VLM 是 Qwen3-VL-2B (https://arxiv.org/abs/2511.21631)。Qwen3-VL 内部由 Qwen3 LLM (https://arxiv.org/abs/2505.09388) + SigLIP-2 vision encoder (https://arxiv.org/abs/2502.14786) 组成。
- Vision encoder 是 ViT + 3D convolutional modules,这跟大多数现代 VLA (OpenVLA, π0, π0.5) 的 vision tower 设计一致。
- 输入图像被 resize 到 224×224。
- 关键改造:在 Qwen3-VL 的 vocabulary 里加了两个新的 learnable token type:
  - `⟨latent_i⟩` — 表示第 $i$ 个时间步的 latent action。在 input sequence 里把同一个 `⟨latent_i⟩` 重复 $K = 24/T$ 次。$T$ 是 future video horizon (默认 8),所以 $K = 3$。这种 "token replication" 是借鉴了 OpenVLA-OFT (https://arxiv.org/abs/2509.21558) 和 π0-Fast (https://arxiv.org/abs/2509.04785) 的做法,目的是在 attention 里给 latent action 多一些"票数",防止被 image token 淹没。
  - `⟨action⟩` — embodied action token,放在 latent action tokens 之后,作为 flow-matching head 的 conditioning signal。

### 3.2 World State Encoder — V-JEPA2 + multi-view concat

公式 (1):

$$s_{t_i} = \big\Vert_v F(I_{v,t_i})$$

变量解释:
- $I_{v,t_i}$ — 视角 $v$ 在时间 $t_i$ 的图像 frame。
- $F(\cdot)$ — 单视角 video encoder,在 paper 里就是 frozen V-JEPA2。
- $\Vert$ — 向量拼接 (concatenation)。
- $s_{t_i}$ — 聚合多视角之后的 unified world state 表示。

直觉:V-JEPA2 在 self-supervised 阶段已经把 "什么是 motion-relevant 的 semantic latent" 学进去了。它是 JEPA 风格,所以天然抑制 low-level pixel variation、保留 action-relevant 的 dynamics factor。直接拿它当 target encoder,等于把 JEPA 的 invariance 性质继承过来 — 这正是 paper 反复强调的 "robust to camera motion and background changes" 的来源。

multi-view 用 concat 而不是 cross-attention 是个偏 conservative 的选择,简单且可扩展。当 view 数 < 2 时复制一遍,> 2 时随机选 2 个 (Appendix A.2 写明)。

### 3.3 Latent Action Pathway — VLM 输出 latent tokens

公式 (2):

$$z_{t_i} = p_\theta^{VLM}\big(\langle \text{latent}_i \rangle \;\big|\; \{I_{j,t_0}\}_{j=0}^{v}, \ell\big)$$

变量解释:
- $p_\theta^{VLM}$ — VLM (Qwen3-VL) 在参数 $\theta$ 下定义的条件分布/函数。
- $\langle\text{latent}_i\rangle$ — 第 $i$ 个 learnable latent action token,被复制 $K$ 次进入 input。
- $\{I_{j,t_0}\}_{j=0}^{v}$ — 各个视角在 initial time step $t_0$ 的图像,作为 VLM 的视觉 input。
- $\ell$ — language instruction。
- $z_{t_i}$ — VLM 在 latent action token 位置上输出的 hidden state,被解释为 "第 $i$ 时间步的 latent action 表示"。

注意 student pathway 这里只看到 $t_0$ 的 observation,根本不喂 future frame。future frame 仅通过 world state encoder 进入损失 — 而损失只施加在 predictor 的输出上。这就切断了 leakage 的捷径。

### 3.4 Latent World Model — auto-regressive Transformer

公式 (3):

$$\hat{s}_{t_{1:i+1}} = p_\theta^{WM}(s_{t_{0:i}}, z_{t_{0:i}})$$

变量解释:
- $p_\theta^{WM}$ — 世界模型,12 层 Transformer,8 个 attention head,image token dim 2048,每个时间步 256 个 image token,3 个 action token,2 个 view,future horizon 默认 8 (Table 5)。
- $s_{t_{0:i}}$ — 已观测到的 history world state sequence。
- $z_{t_{0:i}}$ — 已生成的 latent action sequence。
- $\hat{s}_{t_{1:i+1}}$ — 预测的下一 chunk world state,horizon 从 $t_1$ 到 $t_{i+1}$。

attention mask 设计很讲究:
- 同一时间步内:K 个 latent action token 和 N 个 world state token 之间是 bidirectional full attention。
- 跨时间步:严格 causal — 时间 $t$ 的 token 只能 attend 到 $\le t$ 的 token。

这是把 "frame-level 内部 bidirectional + frame 之间 autoregressive" 的 spatiotemporal 因式分解做出来了,跟 V-JEPA2 的 tube masking 在哲学上一致,但 here 我们有显式的 action tokens 作为 condition。

### 3.5 JEPA ELBO 视角

公式 (4) 是把整个 objective 放到 ELBO 的框架里:

$$\log p(s_{t_{1:T}} \mid z_{t_{0:T-1}}) \ge \sum_{k=1}^{T} \mathbb{E}_{s_{t_k} \sim F(\cdot)} \big[\log p_\theta(\hat{s}_{t_k} \mid s_{t_k})\big] - D_{KL}\big[F(\cdot) \,\|\, p_\theta^{WM}\big]$$

变量解释:
- $F(\cdot)$ — frozen target encoder (V-JEPA2),带 stop-gradient。
- $p_\theta^{WM}$ — online predictor (world model)。
- $s_{t_k}$ — target encoder 在时间 $t_k$ 算出的 ground truth latent。
- $\hat{s}_{t_k}$ — world model 预测。
- $D_{KL}[F(\cdot) \| p_\theta^{WM}]$ — target encoder 分布和 predictor 分布之间的 KL divergence。

因为 $F(\cdot)$ 是 deterministic 的 (V-JEPA2 forward 没有 stochasticity),KL 项退化为 0,ELBO 退化成 latent space 里的 reconstruction loss。这就给了公式 (5):

$$\mathcal{L}_{WM} = \sum_{k=1}^{T} \mathbb{E}_{s_{t_k} \sim F(\cdot)} \,(\hat{s}_{t_k} - s_{t_k})$$

(原文写的是没有显式 norm,但根据上下文以及标准 JEPA loss,实际是 L2 distance;paper 排版省了 $\|\cdot\|_2$ 符号。)

### 3.6 Flow-Matching Action Head — DiT-B

当 downstream 是 robot data (带 action label) 时,加一个 action prediction head。Paper 选了 conditional flow matching (Lipman et al. 2023, https://arxiv.org/abs/2210.02747) + DiT-B (Peebles & Xie 2023, https://arxiv.org/abs/2212.09748) 作为 head。这跟 π0 (https://arxiv.org/abs/2410.24164) 选 flow matching 是一样的 reasoning — flow matching 在 action space 上比 discrete diffusion 更连续,训练也稳定。

公式 (7) 定义插值轨迹:

$$a_t = (1-t)\epsilon + t\, a_{0:H}, \quad t \sim \mathcal{U}(0,1)$$

变量:
- $a_{0:H}$ — ground-truth action sequence,horizon $H$ (Table 6 是 7)。
- $\epsilon \sim \mathcal{N}(0, I)$ — Gaussian noise。
- $t$ — flow time,$[0,1]$ 均匀采样。$t=0$ 时 $a_t = \epsilon$ (纯噪声),$t=1$ 时 $a_t = a_{0:H}$ (真值)。

公式 (8) 是 flow-matching 训练目标:

$$\mathcal{L}_{FM} = \mathbb{E}_{a_{0:H}, \epsilon, t}\Big[\big\|v_\theta(a_t, t \mid z_a) - (a_{0:H} - \epsilon)\big\|_2^2\Big]$$

变量:
- $v_\theta(\cdot)$ — DiT-B 参数化的 velocity field,以 $z_a$ 为 condition。
- $(a_{0:H} - \epsilon)$ — ground-truth velocity (从插值 $a_t = (1-t)\epsilon + t\,a_{0:H}$ 对 $t$ 求导正好是 $a_{0:H} - \epsilon$)。
- $z_a$ — 公式 (6) 算出来的 action-conditioning representation,它是 VLM 在 `⟨action⟩` token 位置输出的 hidden state。

公式 (6):

$$z_a = p_\theta^{VLM}\big(\langle\text{action}\rangle \mid \{I_{i,t_0}\}_{i=0}^{v}, \ell, \langle\text{latent}_i\rangle\big)$$

注意 conditioning 是 $z_a$,它已经把 visual obs + language + latent action sequence 三个信号通过 Qwen3-VL 的 causal attention 融在一起了。`⟨action⟩` 在 sequence 末尾,所以它能看到所有 latent action tokens,这相当于让 action head 显式地被 "推断出的 latent action plan" 条件化。

最终联合 objective (公式 9):

$$\mathcal{L} = \mathcal{L}_{FM} + \beta\, \mathcal{L}_{WM}$$

$\beta$ 是 tunable weight。这意味着 robot data 上同时算两个 loss:既要预测未来 latent state (保持 world model 一致性),也要生成真实 action。这是 paper Section 3.3 标题 "Joint Optimization Objectives" 的来源。

---

## 4. Pretraining → Fine-tuning 两阶段流程

### Pretraining 数据:
- **SSv2** (Something-Something v2, https://arxiv.org/abs/1706.04261): 220K human action videos,无 action label,只用 $\mathcal{L}_{WM}$。
- **Droid** (https://arxiv.org/abs/2403.12945): 76K high-quality robot demonstrations,带 action label,用 $\mathcal{L}_{FM} + \beta\mathcal{L}_{WM}$。

paper 强调一个关键点:这两个 dataset 在 pretraining 阶段是 **同时** 训练的 (jointly train 50K steps),不是 "先 SSv2 再 Droid"。这跟 LAPA / UniVLA / villa-X 通常的多阶段 pipeline 不一样,这是 paper Section 1 强调的 "streamlined two-stage pipeline" (pretrain → fine-tune) 的具体体现。

### Fine-tuning 数据:
- LIBERO (https://arxiv.org/abs/2306.03310): ~2K expert demos,30K steps。
- SimplerEnv (https://arxiv.org/abs/2410.24185): Fractal + BridgeV2 datasets。
- Real-world: 100 demos,3 个 pick-and-place 任务,20K steps。

### Hardware / hyperparameter:
- 8× NVIDIA A100
- batch size 32 per GPU → global 256
- cosine LR + linear warmup
- peak LR: VLM 和 world model 1e-5,action head 1e-4
- image input 224×224,video clip 给 world state encoder 是 256×256

action 表示:end-effector delta position + delta axis-angle,各自 min-max normalize 到 [0,1];gripper command binarize 成 {0,1}。这跟 π0 的 joint-space delta 不同 — VLA-JEPA 选 end-effector 是因为它跟 latent action 在 task space 上更对齐。

---

## 5. 实验表格逐张拆读

### 5.1 LIBERO (Table 1)

LIBERO 有 4 个 suite: Spatial, Object, Goal, LIBERO-10。每个 task 跑 50 episodes,每 suite 500 episodes。

| Method | Spatial | Object | Goal | LIBERO-10 | Avg |
|---|---|---|---|---|---|
| LAPA | 73.8 | 74.6 | 58.8 | 55.4 | 65.7 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π0-Fast | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| villa-X | 97.5 | 97.0 | 91.5 | 74.5 | 90.1 |
| GR00T N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| **VLA-JEPA** | 96.2 | **99.6** | 97.2 | **95.8** | **97.2** |
| w/o human videos | 94.8 | 99.6 | 95.8 | 94.0 | 96.1 |

观察:
- VLA-JEPA 在 Object 和 LIBERO-10 上拿第一,Object 上 99.6 是全场最高,说明 latent world modeling 在 long-horizon (LIBERO-10 是 10 步 horizon 任务) 上有显著优势。
- 比 OpenVLA-OFT (97.1) 高 0.1,比 π0.5 (96.9) 高 0.3。差距小,但 OpenVLA-OFT 和 π0.5 用了更大的 robot dataset pretraining。VLA-JEPA 只用 Droid (76K) + SSv2 (220K)。
- "w/o human videos" 行降到 96.1 — 说明在 LIBERO 这种 ID (in-distribution) 设定下,SSv2 帮助有限,高质量 expert demo 是主导因素。这是 Section 4.5 Q1 的核心 ablation 结论。

### 5.2 SimplerEnv (Table 2)

SimplerEnv 是 real-to-sim 的 OOD 设定,有 Google Robot 和 WidowX Robot 两套。

| Method | Pick | Move | Drawer | Place | Avg(G) | Spoon | Carrot | Block | Eggplant | Avg(W) |
|---|---|---|---|---|---|---|---|---|---|---|
| LAPA* | – | – | – | – | – | 70.8 | 45.8 | 54.2 | 58.3 | 57.3 |
| villa-X | 81.7 | 55.4 | 38.4 | 4.2 | 44.9 | 48.3 | 24.2 | 19.2 | 71.7 | 40.8 |
| UniVLA | – | – | – | – | – | – | – | – | – | 42.7 |
| RoboVLMs | 77.3 | 61.7 | 43.5 | 24.1 | 51.7 | 45.8 | 20.8 | 4.2 | 79.2 | 37.5 |
| GR00T N1 | 0.7 | 1.9 | 2.9 | 0.0 | 1.4 | 1.4 | 0.0 | 0.0 | 13.9 | 3.8 |
| π0 | 72.7 | 65.3 | 38.3 | – | – | 29.1 | 0 | 16.6 | 62.5 | 40.1 |
| π0-Fast | 75.3 | 67.5 | 42.9 | – | – | 29.1 | 21.9 | 10.8 | 66.7 | 48.3 |
| **VLA-JEPA** | **88.3** | 64.1 | **59.3** | **49.1** | **65.2** | **75.0** | **70.8** | 12.5 | **70.8** | **57.3** |
| w/o human videos | 85.3 | 66.7 | **75.5** | **86.1** | **78.4** | 75.0 | 54.2 | 20.8 | 79.2 | 57.3 |

观察:
- Google Robot: VLA-JEPA 65.2 是第一,比 villa-X (44.9)、π0-Fast (要算的话约 61.9)、π0 都高。
- WidowX: VLA-JEPA 57.3 跟 LAPA* 并列第一。LAPA* 是 LAPA 用 SimplerEnv 里的成功 rollout 当 expert 来训练 (一种 "oracle" 设置),所以这个比较其实是 VLA-JEPA 用更朴素的训练达到了 LAPA 的 oracle 性能。
- 注意 "w/o human videos" 在 SimplerEnv 上反而更高 (78.4 vs 65.2 on Google Robot),这看起来矛盾。paper 在 Section 4.5 Q1 给了解释:SSv2 的 human video 缺乏 robot 的物理 action trajectory 信息,在 real-to-sim 这种需要精确 action 的场景反而引入 noise。这其实暴露了 human video → robot transfer 的一个根本问题 — 你从 human video 学到的是 "world dynamics in latent space",但 robot 的 action space 跟 human hand 不共享,这部分 gap 必须靠 robot data fine-tune 来补。

### 5.3 LIBERO-Plus (Table 3) — 这是 paper 的"亮点"实验

LIBERO-Plus (https://arxiv.org/abs/2510.13626) 系统性地在 7 个维度扰动 LIBERO:Camera, Robot, Language, Light, Background, Noise, Layout。

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | Avg |
|---|---|---|---|---|---|---|---|---|
| UniVLA | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π0 | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π0-Fast | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| **VLA-JEPA** | 63.3 | **67.1** | **85.4** | **95.6** | **93.6** | 66.3 | **85.1** | **79.5** |
| w/o human videos | 40.3 | 55.7 | 72.9 | 88.2 | 70.5 | 38.2 | 74.6 | 62.9 |

观察:
- VLA-JEPA 在 5/7 个扰动维度拿第一,平均 79.5,比 OpenVLA-OFT (69.6) 高 10 个点,比 π0 (53.6) 高 26 个点。
- Camera / Robot / Noise 三个扰动上没有拿第一。Camera 和 Robot 上 π0-Fast 比较好,Noise 上 OpenVLA-OFT 比较好。但 VLA-JEPA 在 Language / Light / Background / Layout 上压倒性领先。
- 关键 ablation: "w/o human videos" 掉到 62.9 — 比 VLA-JEPA (79.5) 少了 16.6 个点。这是 paper 关于 "human video 主要贡献是 robustness" 这一论点的硬证据。LIBERO-Plus 测的就是 robustness,所以这里 human video 的作用才显出来。跟 LIBERO 上 "human video 几乎没用" 形成对比。

paper 在 Figure 5 进一步画了 "human video 比例 vs 各扰动维度 success rate" 的曲线,显示随 human video 比例增加,Language / Light / Background / Layout 这四条曲线单调上升,Camera / Robot / Noise 这三条平或略降。这个图非常 clean 地说明了:human video 教给 model 的是 "task-agnostic 的视觉/语义稳定性",而 robot-specific 的扰动 (camera extrinsic / robot embodiment) 还得靠 robot data。

### 5.4 Real-World Franka (Figure 4 + Appendix B)

- Setup: Franka Research 3 + Robotiq 2F-85 + 3× Intel RealSense D435 (2 third-person + 1 wrist)
- 100 demos,3 个 pick-and-place (葡萄/苹果/芒果/橙子 → 盘/碗)
- 评估两种 OOD:task OOD (新任务) 和 object-layout OOD (训练任务但布局乱)
- baseline: π0 和 π0.5 在同一 demo 集上 fine-tune

观察:
- ID 和 object-layout OOD 上 VLA-JEPA 最好。
- task OOD 上 VLA-JEPA 第二 (π0.5 最好)。
- π0.5 在 task OOD 上 instruction-following 更准,但 position control 会突破 safety boundary,导致执行失败。
- VLA-JEPA 在 grasp 失败时会主动 re-open gripper 重抓,π0 和 π0.5 都不会。paper 把这个归因于 SSv2 里有大量人类反复尝试的 grasp 行为,VLA-JEPA 学到了 "when to regrasp" 这种 temporal decision,然后内部 map 到自己的 robot dynamics。这个观察其实蛮有意思,因为 regrasp 是一个 temporal decision (什么时候放弃当前 grasp 重来),它不需要新的 low-level dynamics 知识,只需要 "失败 → 退后 → 再试" 这个时序 pattern。human video 提供了大量这种 pattern。

### 5.5 Ablation: future video horizon $T$ (Table 4)

| T | Spatial | Object | Goal | 10 | Avg |
|---|---|---|---|---|---|
| 4 | 95.0 | 99.2 | 95.8 | 89.0 | 94.8 |
| 8 | 94.8 | 99.8 | 95.8 | 94.0 | **96.1** |
| 16 | 92.8 | 98.8 | 98.0 | 92.2 | 95.5 |

观察:
- $T = 8$ 最好,跟 action horizon $H = 7$ 接近匹配。
- $T = 4$ 信息不足,LIBERO-10 (long horizon) 掉得最多。
- $T = 16$ 太长,引入冗余 — Goal suite (任务简单) 受益,但 Spatial suite (细操作) 受损。这暗示 latent action 跟 action horizon 应该 co-design,过长会注入 irrelevance。

### 5.6 Attention 可视化 (Figure 6)

paper 把 LAPA / UniVLA / VLA-JEPA 三个 model 的 latent action token 对 image token 的 attention map 可视化,在三种 input (sim / human video / real robot) 上对比。结论:
- LAPA 的 latent action attention 很 dense,关注桌面无关物体 — 这是 information leakage 的征兆:latent action 退化成 "future frame 的压缩",所以它要 attend 到所有可能在下一帧变化的区域。
- UniVLA 通过 language guidance 缓解了,但过于聚焦 semantic — 会 attend 到 stationary pen、tablecloth texture 这种跟 task 无关但 semantic 醒目的东西。
- VLA-JEPA 最聚焦:robotic arm / hand / target object。这印证了 "leakage-free + latent space prediction" 的 inductive bias 起作用了 — 既然未来信息没进来,latent action 就只能去关心 "操作上 causally relevant" 的部分。

---

## 6. 这篇 paper 跟其它 SOTA 的关系图谱

我把它放在你熟悉的 landscape 里:

- **π0 / π0.5** (https://arxiv.org/abs/2410.24164, https://arxiv.org/abs/2504.16054): VLM backbone + flow-matching action head,pretrain 在大规模 robot data 上,没有显式 latent action。VLA-JEPA 跟它共享 flow-matching head 的选择,但 VLA-JEPA 在 VLM 和 head 之间插入了 latent action token + world model。
- **LAPA** (https://arxiv.org/abs/2410.11758): latent action pretraining from video 的代表作。VLA-JEPA 直接 critique 它的 frame-difference + leakage 问题。
- **UniVLA** (https://arxiv.org/abs/2505.21672): task-centric latent action,用 VQ-VAE 离散化。VLA-JEPA 在 LIBERO 上 (Table 1) 超 UniVLA 2 个点,LIBERO-Plus 上超 36.6 个点。
- **villa-X** (https://arxiv.org/abs/2503.02065): unified latent action codebook across human + robot video,需要 multi-stage alignment。VLA-JEPA 在 SimplerEnv 上用 villa-X 1% 的数据量达到 comparable 性能。
- **OpenVLA-OFT** (https://arxiv.org/abs/2509.21558): token replication trick 的来源,VLA-JEPA 借鉴了它。
- **Moto / MotoGPT** (https://arxiv.org/abs/2503.17112): latent motion token + hierarchical pretraining。
- **GR00T N1** (https://arxiv.org/abs/2503.14734): NVIDIA 的人形 foundation model,在 LIBERO 上 93.9,SimplerEnv 上几乎全挂 (Google Robot 1.4,WidowX 3.8) — 说明 GR00T N1 的 pretrain data 跟 SimplerEnv 的 visual gap 太大。
- **WorldVLA** (https://arxiv.org/abs/2506.26120): autoregressive action world model,把 next-frame generation 和 action prediction 联合做。在 LIBERO-Plus 上崩了 (25.0),paper 解释是 pixel-level world model 在扰动下脆弱。

VLA-JEPA 的位置:它在 "latent action pretraining" 流派内部,但用 JEPA 替换了 frame-difference + VQ-VAE 的核心,从而避免 leakage。这是把 LeCun 路线 (JEPA) 显式嫁接到 robot learning 上的一个干净实例。

---

## 7. 一些我自己的 intuition 和疑问

### 7.1 为什么 "leakage-free" 这么有效?— 一个信息论视角

考虑 latent action $z$ 和 future observation $O_{future}$ 的互信息 $I(z; O_{future})$。

- 在 LAPA / UniVLA 里,$z$ 由 $O_{current}$ 和 $O_{future}$ 共同决定 (因为 VQ-VAE target 是 $O_{future}$,$z$ 要重建 $O_{future}$),所以 $I(z; O_{future})$ 被 maximize 到 "几乎 $z$ 编码了 $O_{future}$ 全部信息" 的程度。但其中真正跟 action causal 的部分 $\Delta s = s_{t+1} - s_t$ 只是 $O_{future}$ 的一个子流形。所以 $z$ 浪费了大部分 capacity 在 nuisance factor (texture, lighting, background) 上。
- 在 VLA-JEPA 里,$z$ 只能从 $O_{current}$ 推出来 (公式 2 没喂 future),所以 $I(z; O_{future})$ 被压在 "能从 $O_{current}$ 推出的部分",supervision target 是 latent state $\hat{s}_{t_k}$ 而不是 pixel。这迫使 $z$ 编码 "哪些 factor 在 latent state 上会变化",自然就跟 controllable DOF 对齐。

这其实是 information bottleneck 的一种 implicit 实现 — 把 future 从 input 里去掉,等于在 student pathway 上加了一个 "future-blind" 约束。

### 7.2 为什么 latent space prediction 比 pixel space prediction 更鲁棒?

关键在 V-JEPA2 这个 target encoder 的 invariance 性质。V-JEPA2 在 self-supervised 阶段就通过 masking + latent alignment 学到了 "丢弃 pixel-level noise,保留 semantic + motion factor"。所以 target latent $s_{t_k}$ 本身已经是 "denoised" 的。Student pathway 在预测 $s_{t_k}$ 时,即使 background 变化,只要 background 不进入 V-JEPA2 的 semantic representation,$\hat{s}$ 也不需要预测它。这等价于 paper Section 1 强调的 "robust to camera motion and background changes"。

但这里有个 subtlety:V-JEPA2 的 invariance 是它自己 pretraining 数据决定的。如果 V-JEPA2 没在某类 appearance shift 上 invariance,VLA-JEPA 也不会有。所以 VLA-JEPA 的 robustness 上限被 V-JEPA2 的 robustness 上限 ceiling 住了。这是个隐式 assumption,paper 没显式讨论。

### 7.3 公式 (4) 的 KL 项为什么 vanish?

paper 说 "$F(\cdot)$ produces deterministic embeddings, the KL term vanishes"。这里有点 hand-wavy。准确说,在 ELBO 推导里,假设 $p_\theta(\hat{s}_{t_k} \mid s_{t_k})$ 是 Gaussian with fixed variance,那 $\log p_\theta$ 等价于 $-\|\hat{s}_{t_k} - s_{t_k}\|^2 / (2\sigma^2)$ + const。KL 项在 standard ELBO 里是 $D_{KL}[q(z|x) \| p(z)]$ — 这里把 $F(\cdot)$ 当 posterior over latent,$p_\theta^{WM}$ 当 prior over latent。但 $F(\cdot)$ 是 deterministic encoder,所以 $q(z|x) = \delta(z - F(x))$ 是 point mass,KL 退化成 $\log p_\theta^{WM}(F(x))$。如果进一步假设 prior uniform,这一项也消失。所以 paper 的表述是 OK 的,但严格推导应该在 Gaussian + fixed variance + uniform prior 的假设下。

### 7.4 关于 human video 在 LIBERO 上几乎没用的反思

Table 1 "w/o human videos" 96.1 vs VLA-JEPA 97.2 — 差距只有 1.1 个点。Table 2 上甚至 "w/o human videos" 更高 (Google Robot 78.4 vs 65.2)。这说明在 ID 和 real-to-sim 这种 "task-specific skill 主导" 的场景,SSv2 的贡献很有限。

为什么?我怀疑是 SSv2 的 action 跟 robot action 太不共享了。SSv2 是人手操作物体,robot 是 gripper 夹物体,两者在 end-effector geometry、contact dynamics、force profile 上差太多。VLA-JEPA 只在 latent state space 上 transfer,所以 transfer 的是 "world dynamics" — 但 LIBERO 的 dynamics 很简单 (刚体、无摩擦变异、固定光照),没什么可 transfer 的。

LIBERO-Plus 上 human video 才大放异彩,因为扰动主要在 visual appearance 上 (Light / Background / Layout / Language),这恰好是 V-JEPA2 pretrain + SSv2 fine-tune 学到的 invariance 最有用的地方。

这给我一个 intuition:**human video pretraining 的价值不是教 model 新技能,而是教 model "哪些视觉变化可以忽略"**。这跟 LLM 里 "pretraining 教语法,SFT 教风格" 的某种类比 — pretraining 教 invariance,SFT 教 specific skill。

### 7.5 一些 paper 没充分讨论的 limitation

1. **V-JEPA2 的依赖**:VLA-JEPA 的 robustness ceiling 被 V-JEPA2 决定。如果未来 V-JEPA3 出来,可以替换,但 paper 没做 ablation 验证 "换一个 target encoder 会怎样"。
2. **Multi-view 简单 concat**:公式 (1) 用 $\Vert$ 简单 concat 多视角,这对 view 数变化不友好。如果 view 数从 2 变到 3,concat 后维度变,模型得重训。cross-attention 或者 view-token 的设计可能更鲁棒。
3. **Action head 跟 latent action 解耦**:action head 是 DiT-B,从 $z_a$ condition 出发。$z_a$ 包含了 latent action tokens 的信息,但 action head 仍是独立网络。能否像 π0 那样把 action head 跟 VLM 更深度 entangle,是个 open question。
4. **Real-world 实验规模小**:100 demos,3 个任务。π0.5 在 task OOD 上反而比 VLA-JEPA 强 (instruction following 更准),说明 VLA-JEPA 在 language grounding 上还有 gap。这跟 "w/o human videos" 在 Language 扰动上从 85.4 掉到 72.9 一致 — human video 帮了 language grounding,但还不够。
5. **没有 scaling law 实验**:paper 没探究 SSv2 规模从 22K → 220K → 2.2M 时性能曲线。如果 VLA-JEPA 在 internet-scale video 上没有 scaling,JEPA 路线在 VLA 上的故事就还不完整。

### 7.6 跟你 (Karpathy) 一直强调的几个直觉的连接

- **"Predict in latent space, not pixel space"**:这是 VLA-JEPA 的字面核心。LeCun 在 I-JEPA / V-JEPA 系列反复讲,VLA-JEPA 把它应用到 VLA。Failure mode 1 (pixel objective bias appearance) 就是 "predict in pixel space" 的代价。
- **"Surgical supervision"**:JEPA 的 target encoder 只在 target patch 上算 loss,不在 context patch 上算。VLA-JEPA 的 target encoder 只在 future clip 上算 loss,不在 current observation 上算。这避免了 model 浪费 capacity 去重建已经看到的 input。
- **"Asymmetric encoder-predictor"**:student encoder (Qwen3-VL) 和 target encoder (V-JEPA2) 是两套网络,stop-gradient 在 target side。这是 JEPA 防 collapse 的标准技巧。
- **"Multi-modal token passing"**:VLA-JEPA 用 `⟨latent_i⟩` 和 `⟨action⟩` 作为 special token 把 VLM 输出和 action head 连起来。这跟你最近在 podcast 里讲的 "token 是 LLM 的 universal interface" 一致。

---

## 8. 一些可能的延伸方向 (我自己的 speculation)

1. **Replace V-JEPA2 with a robot-pretrained JEPA**:如果 target encoder 本身就在 robot data上 pretrain 过,target latent $s_{t_k}$ 会更 action-relevant。可能能解释为什么 human video 在 LIBERO 上没用 — target encoder 太"human-centric"。
2. **Hierarchical latent action**:目前 latent action 是 frame-level (每帧一个 `⟨latent_i⟩`)。可以做 coarse-to-fine:高层 latent action 描述 subgoal,低层描述 frame transition。
3. **Joint train target encoder**:paper 把 V-JEPA2 freeze 了。如果让 target encoder 跟 student 一起更新 (用 EMA 或 stop-gradient),可能让 target latent 更适配 robot domain。但风险是 collapse。
4. **Latent action consistency across embodiment**:让同一个 latent action 在 human 和 robot 上有相同语义 (e.g. "approach object"),这需要 contrastive or alignment loss。villa-X 走了这条路但用了 multi-stage,VLA-JEPA 能否 single-stage 做?
5. **Replace flow-matching with discrete diffusion**:最近 Discrete Diffusion VLA (Liang et al. 2025) 显示离散 diffusion 在 action 上也有竞争力。VLA-JEPA 的 latent action 是连续的,跟 discrete diffusion 不直接兼容,但可以探究。
6. **Test-time scaling**:VLA-JEPA 推理时是 single-pass。能否在 inference 时用 world model 做 planning — 比如 sample 多个 latent action 候选,用 world model 预测未来 state,挑最好的?这其实跟 Value-Guided JEPA Planning (Destrade et al. 2025, https://arxiv.org/abs/2506.15862) 的思路一致。

---

## 9. Reference 链接汇总

paper 本身:
- VLA-JEPA GitHub: https://github.com/ginwind/VLA-JEPA/
- VLA-JEPA Project Page: https://ginwind.github.io/VLA-JEPA/
- VLA-JEPA HuggingFace: https://huggingface.co/ginwind/VLA-JEPA/

核心依赖方法:
- V-JEPA2: https://arxiv.org/abs/2506.09985
- V-JEPA: https://arxiv.org/abs/2311.09079
- I-JEPA: https://arxiv.org/abs/2301.08243
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- Qwen3: https://arxiv.org/abs/2505.09388
- SigLIP-2: https://arxiv.org/abs/2502.14786
- Flow Matching: https://arxiv.org/abs/2210.02747
- DiT: https://arxiv.org/abs/2212.09748

对比 baseline:
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- π0-Fast: https://arxiv.org/abs/2509.04785
- OpenVLA: https://arxiv.org/abs/2406.09246
- OpenVLA-OFT: https://arxiv.org/abs/2509.21558
- LAPA: https://arxiv.org/abs/2410.11758
- UniVLA: https://arxiv.org/abs/2505.21672
- villa-X: https://arxiv.org/abs/2503.02065
- Moto / MotoGPT: https://arxiv.org/abs/2503.17112
- GR00T N1: https://arxiv.org/abs/2503.14734
- WorldVLA: https://arxiv.org/abs/2506.26120
- CoT-VLA: https://arxiv.org/abs/2503.02065
- RoboVLMs: https://arxiv.org/abs/2412.15071

数据集:
- Droid: https://arxiv.org/abs/2403.12945
- Something-Something v2: https://arxiv.org/abs/1706.04261
- Ego4D: https://arxiv.org/abs/2110.07058
- LIBERO: https://arxiv.org/abs/2306.03310
- LIBERO-Plus: https://arxiv.org/abs/2510.13626
- SimplerEnv: https://arxiv.org/abs/2410.24185

诊断性参考:
- "What do latent action models actually learn?": https://arxiv.org/abs/2506.15691
- Value-Guided Action Planning with JEPA World Models: https://arxiv.org/abs/2506.15862

---

## 10. 一句话总结

VLA-JEPA 把 LeCun 的 JEPA 哲学 ("latent space prediction + asymmetric encoder + stop-gradient + leakage-free") 干净地搬进 VLA pretraining,通过诊断并修复 latent-action 流派的四个 failure mode (pixel bias / nuisance motion / info leakage / multi-stage fragility),用一个 streamlined 的两阶段流程 (joint pretrain on SSv2+Droid → fine-tune on task data) 在 LIBERO / LIBERO-Plus / SimplerEnv / real-world 上都拿到了 comparable-to-SOTA 的性能,并且在 robustness 扰动上显著领先。最深的 insight 是它把 "human video 的作用" 重新定义了 — human video 不教新技能,教的是 visual invariance,这个 invariance 在 task-specific 评估 (LIBERO) 上看不出来,在 robustness 评估 (LIBERO-Plus) 上才显化。
