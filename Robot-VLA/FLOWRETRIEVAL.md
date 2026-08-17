---
source_pdf: FLOWRETRIEVAL.pdf
paper_sha256: 0d82389a33f85b81ce8e46286ef160b88efedddfab3e2d618db70fef1f435bc0
processed_at: '2026-08-04T09:47:14-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话重讲 FLOWRETRIEVAL

## 一句话概括

机器人学新任务太吃数据了，几百上千条 demo 才能学会一个 task。这篇 paper 说：你之前攒的一大堆老任务数据别浪费，我用 **optical flow (光流)** 当"动作指纹"，从老数据里挑出动作相似的部分来帮新任务练手。

## 1. 这 paper 在解决什么实际问题

想象你是个机器人，今天老板让你学"把笔插进杯子"这个新动作。只给你 10 条 demo，你肯定学不会——10 条太少，policy 没法泛化。

但公司里有个大数据库，存了你之前做过的几百种任务：抓锅放水槽、开微波炉、抓东西放到别处……能不能从里面翻出有用的数据来辅助学习？

问题来了——**怎么定义"有用"？**

### 两种极端做法都不行

**做法 A：看图找相似**（Behavior Retrieval 那派）
拿 target 任务当前看到的 RGB 画面，去 prior 数据库里找画面长得像的。听起来合理，但现实很骨感：target 任务在你自己厨房里拍的，prior 数据是别人家厨房拍的，背景颜色、光照、物体外观全不一样。你按"长得像"去检索，要么啥也找不到，要么找到一堆视觉相似但动作完全错的（adversarial data）。

Paper 里专门设计了个 Square Assembly 实验来 hammer 这个点：把 useful data 放在绿背景里，adversarial data 放在和 target 一样的红背景里。你按视觉检索，全跑去检索 adversarial data 了，直接帮倒忙。

**做法 B：看语言标签找相似**（RT-H 那派）
给每个任务打文字标签，"open the door" 和 "open the door" 就匹配。问题是 language 太糙了。"open the door" 可能是拧 knob，也可能是压 handle，low-level 动作完全不同——你拿拧 knob 的数据去学压 handle，没用。反过来说，"turn on faucet" 其实是旋转动作，对学"turn doorknob"很有帮助，但语言层面这俩八竿子打不着，你检索不到。

### 那怎么办？

作者说：咱别盯着视觉也别盯着语义，**盯着"动作本身"**。一个任务的本质是"画面里的像素怎么动"——往左推、往下按、旋转、抓起来抬走……这些 motion pattern 才是跨任务可迁移的核心。

## 2. 用 Optical Flow 当"动作指纹"

Optical flow 你可以理解成：把视频前后两帧叠在一起，看每个像素往哪个方向挪了多少。输出是个 $H \times W \times 2$ 的图，每个像素一个 $(\Delta x, \Delta y)$ 向量，告诉你"这个点往哪动了"。

为什么选 optical flow？三个优点：

1. **不受外观干扰**：一只红杯子和一只蓝杯子，只要都从左往右被推走，flow 长得几乎一样。texture、color、background 全被 flow 抹掉了。
2. **比语言细**：能区分"拧"和"压"这种细粒度动作，语言做不到。
3. **跨 embodiment 能用**：WidowX 机器人抓锅和 ViperX 机器人抓锅，画面里都是"手伸过去、合拢、抬起来"这个 motion pattern，flow 一致；但 proprioception（关节角、末端位置）完全对不上。

直觉上，optical flow 就是"画面在动"这件事的最直接的数学刻画，比 RGB 抽象，比 language 具体，正好是 sweet spot。

## 3. 方法的三个 stage

整个 FLOWRETRIEVAL 分三步，每步都很直觉。

### Stage 1: 训一个"动作压缩器"（VAE）

直接拿 raw optical flow 比距离？不行。Flow 是 $H \times W \times 2$ 的高维向量，里面有大量噪声（背景像素 flow=0，镜头抖动），直接算 L2 距离没意义。

所以先训个 VAE，把 flow 压缩到一个小 latent vector：

$$\mathcal{L}_{\mathrm{VAE}}(\theta, \psi) = \mathbb{E}_{f_t \sim \mathcal{F}_{\mathrm{prior}}} \big[ ||q_\psi(p_\theta(f_t)) - f_t||_2 \big]$$

说人话：
- $f_t$ 是某段 prior data 算出来的 optical flow（一张 $H\times W\times2$ 的图）
- $p_\theta$ 是 encoder，把这张 flow 图压成一个 latent vector（比如 256 维）
- $q_\psi$ 是 decoder，从 latent vector 试图还原回原始 flow 图
- Loss 就是"压扁再展开，看和原图差多少"

训完以后，$p_\theta$ 这玩意儿就像个"动作翻译器"——你喂它一段 flow，它吐给你一个 compact 的 motion embedding。**相似动作 embedding 距离近，不同动作距离远**。

关键假设：prior 数据库足够大、足够多样，里面训出来的"动作理解器"能直接迁移到新任务上。因为 flow 的 domain gap 比 RGB 小多了——你 prior 见过各种"抓起来"的 flow，target 任务里"抓笔"的 flow 长得也差不多。

### Stage 2: 检索——找最像的动作

现在 target 任务来了，10 条 demo。怎么从 prior 里挑数据？

每条 target data 算一个 flow，每条 prior data 也算一个 flow，全送进 $p_\theta$ 拿到 embedding。然后对每个 prior data point $s_i$，算它的 embedding 和 target 所有 embedding 的最小距离：

$$\mathcal{S}(s_i) = -\min_{\forall f_j \in \mathcal{F}_{\mathrm{target}}} ||p_\theta(f_i) - p_\theta(f_j)||_2$$

说人话：prior 的某个数据点，只要它和 target 里的**任意一个**时刻动作很像，就算"有用"。距离取负号，越小（越像）分数越高。

这个 `min` 操作很巧妙——它允许 prior 的一段数据只 cover target 的某个片段。比如 prior 里有个任务是"抓起来放到右边"，target 是"抓起来插进杯子"，那 prior 里"抓起来"这段就匹配上 target 的"抓起来"那段，会被检索出来。"放到右边"那段和 target 不像，分数低，不被检索。

最后按分数排序，取 top $\delta\%$ 当 retrieved dataset $\mathcal{D}_{\mathrm{retrieved}}$。$\delta$ 这个 threshold 要手调，paper 里 1% 到 35% 都用过。

### Stage 3: 训 policy——用检索来的数据辅助

有了 retrieved data，怎么用？直接和 target data 一起 co-training。每个 batch 一半 target、一半 retrieved。

但光这样不够——你得让 policy "注意到" retrieved data 里的 motion 信息。于是加个 auxiliary loss：

$$\mathcal{L} = \mathbb{E}\big[\mathcal{L}_{BC}(s_i, a_{i:i+k}) + \lambda ||\phi_{\mathrm{aux}}(\phi_{\mathrm{enc}}(s_i)) - \mathcal{E}_{\mathrm{flow}}(s_i, s_{i+k})||_2\big]$$

说人话：policy 网络除了要预测 action（主任务），还要顺便预测"下一帧相对当前帧的 optical flow"（副任务）。$\phi_{\mathrm{enc}}$ 是图像 encoder，$\phi_{\mathrm{BC}}$ 是 action head，$\phi_{\mathrm{aux}}$ 是额外的 flow 预测 head。$\lambda=0.01$ 控制副任务权重，很小。

为什么要加这个副任务？

直觉是：让 encoder 在编码图像时，**必须保留足够的 motion 信息**才能预测出 flow。这样 encoder 学到的 representation 就会 motion-centric，对 retrieved data 的利用更高效。

注意这里有个设计取舍——paper 没让 flow 当 bottleneck（强制 action 必须从 flow 推出来），只是当 auxiliary regularizer。因为 flow 包含太多无关细节（背景像素也在动），强制 flow→action 会增加学习难度。Auxiliary loss 兼顾 representation richness 和 gradient flow 通畅。

## 4. 实验结果——到底有多 work

### 主结果（success rate）

| 任务 | BC (只 target) | BC-Co (用全部 prior) | FLOWRETRIEVAL |
|---|---|---|---|
| Square Assembly (对抗设置) | 7% | 3% | **55%** |
| LIBERO-Can | 64% | 16% | **90%** |
| Bridge-Microwave | 56% | 56% | **68%** |
| Bridge-Pot | 20% | 32% | **64%** |
| Pen-in-Cup (Wild prior) | 23% | 12% | **56%** (3.7×) |

几个看点：

**Square Assembly** 这个 task 是 paper 重点 demo 的对抗场景：useful data 视觉背景和 target 不同，adversarial data 视觉背景和 target 一样。BR/SR 这种视觉检索方法全跪了，FLOWRETRIEVAL 55%，BC 才 7%。说明 motion-based 检索能在视觉对抗的 setting 下 still work。

**Pen-in-Cup with Wild prior** 是 headline result：prior 是从 DROID 数据集里 random 抽的 400 条（只 filter 了视角大致匹配），里面啥任务都有。BC-Co 直接用全部 prior 反而把 policy 带坏了（12%），FLOWRETRIEVAL 通过 motion 检索挑出真正相关的那一小撮，做到 56%，**3.7 倍 baseline**。

这个数字很有冲击力，说明 **retrieval 的核心价值在于"挑"，挑对了 10% 的数据胜过用 100% 数据**。这跟 [Data quality in imitation learning (Belkhale et al.)](https://arxiv.org/abs/2310.14796) 的观点一致——prior data 的 quality 比 quantity 重要得多。

### 跟其他 retrieval 方法比

paper 跟 Behavior Retrieval、SAILOR 比，平均高 27%。这俩方法都把 RGB 编进 latent space，导致视觉相似性 dominate 了 motion 相似性。在 Square Assembly 这种视觉对抗场景下，它们 retrieve 一堆 adversarial data，policy 越学越烂。

### 关键 ablation

**ProprioRetrieval**：用 end-effector 位置变化代替 optical flow 做检索。在 simulation 里 viewpoint 固定，效果不错（Square 54%, LIBERO 81%）。但 Bridge 任务 viewpoint 多样化时直接崩（12%）。结论：proprioception 是 cheap but brittle 的 motion proxy，optical flow 贵但 robust。这给我个 intuition——**视觉信号的好处是"自带 viewpoint 信息"**，proprioception 跨场景没法比。

**Pretrained visual encoder 做 retrieval**：用 Voltron、R3M、CLIP、DINOv2 的 feature delta 做 retrieval，Square Assembly 上全跪。因为这些 encoder 学的是 scene semantics，对 motion 不敏感。说明"通用 visual foundation model"在 motion-specific 任务上不一定 work，得有 motion-specific representation。

## 5. 几个有意思的设计细节

### 为什么用 `min` 而不是 `mean` 做 similarity

公式 3 里 similarity 取的是 prior 和 target **任一**点距离的 min，不是 mean。这暗示了一个假设：prior 的某段数据哪怕只 cover target 的某个片段，也算"有用"。

这跟 [kNN-LM](https://arxiv.org/abs/1911.00172) 的思路类似——你不需要 prior 完整复现 target，只要能拼凑出 target 的片段即可。

### Top-$\delta\%$ vs KNN 检索策略

paper appendix 试了 KNN（每个 target point 找 top-k 最近的 prior），结果 worse 10-15%。理由是强制每个 target point 都要有匹配的 prior，会在 target 某些独特片段硬凑 dissimilar prior，反而引入 noise。

这个发现挺反直觉——我原本以为 KNN 更"公平"（uniform 覆盖），结果 top-% 让数据自己说话反而更好。

### $\lambda = 0.01$ 全程不调

auxiliary flow loss 的权重在所有 5 个任务上都是 0.01，没调过。这说明这个 auxiliary loss 是个很 robust 的 regularizer，不是 task-sensitive 的 trick。

## 6. 我对这篇 paper 的整体评价

### 强的地方

1. **核心 insight 简单且有力**：motion (flow) 比 visual 比 language 更适合做 retrieval。这个观察很 sharp。
2. **Decoupled 设计**：retrieval 阶段和 policy learning 阶段完全解耦，可以独立评估，policy backbone 可以换（diffusion policy / 其他 BC 都行）。
3. **实验 domain 多样**：simulation + real，同 embodiment + 跨 embodiment (WidowX→ViperX)，curated prior + wild prior (DROID)，说服力够。
4. **Pen-in-Cup 的 3.7× 是真硬核 result**，prior 完全是 wild 的 random subset 都能 work。

### 弱的地方

1. **Scalability**：每个新 target 都要扫一遍 prior 算 pairwise distance，prior 一大就慢。Paper 自己也承认。我觉得 retrieval 这步可以用 approximate nearest neighbor (ANN) 比如 FAISS 解决，但作者没提。
2. **Threshold $\delta$ 手调**：1% 到 35% 跨度太大，没有自动机制。这是个实际部署的痛点。
3. **GMFlow frozen 用**：没有探索 flow estimator 本身能否在 robot data 上 finetune 来更贴合 motion 分布。
4. **Auxiliary flow loss 设计偏保守**：直接用 L2 regression，没用 contrastive 或更 fancy 的 motion learning。我觉得这块还有空间。

## 7. 给我的几个直觉启发

1. **Mid-level representation 的 power**：visual 太具体，semantic 太抽象，mid-level (motion) 刚好。这个 insight 可能适用于其他领域，比如 NLP 里 syntax tree 是 mid-level，cross-task transfer 时可能比 token-level 或 semantic-level 都好用。

2. **Retrieval > Pretraining in few-shot**：在 target data 极少时，retrieval-based 的数据增强比 pretrain+finetune 更 sample efficient。因为 pretrain 的 representation 是 task-agnostic 的，retrieval 直接搬 task-relevant 的 trajectory 来用，signal-to-noise ratio 高得多。

3. **"挑胜过用全部"**：Pen-in-Cup 实验里 BC-Co（用全部 prior）反而不如 BC（不用 prior），但 FLOWRETRIEVAL（挑一小撮）远胜两者。这让我想到 [GPT-3 in-context learning](https://arxiv.org/abs/2005.14165) 的逻辑——不是把所有知识塞进参数，而是按需检索。

4. **Auxiliary task 是 regularizer 的神器**：让 encoder 顺便预测 flow，比强制它通过 flow 推 action 更好。这个思路在 self-supervised learning 里很常见（[SimCLR](https://arxiv.org/abs/2002.05709), [BYOL](https://arxiv.org/abs/2006.07733)），但用在 IL 里加 motion auxiliary 还是第一次见。

5. **Cross-embodiment 是 robotics 下一战场**：Bridge-Pot 实验 WidowX→ViperX 能 work，说明 optical flow 是 cross-embodiment 的天然 bridge。这让我想到 [Open X-Embodiment](https://robotics-transformer-x.github.io/) 的方向——如果有个 motion foundation model 能跨所有 embodiment，retrieval-based IL 就能真正 scale。

## 8. 可能的后续方向

顺着这篇 paper 的思路，我觉得有几个值得探索的方向：

1. **Flow foundation model**：训个大规模 video pretrain 的 motion encoder 替换 GMFlow+VAE，类似 [UniMatching V2](https://arxiv.org/abs/2310.05753) 或 [VideoMAE V2](https://arxiv.org/abs/2303.12027) 的思路。
2. **Hierarchical retrieval**：先 language filter（cheap，快速减候选集），再 flow refine（expensive，精确筛选）。结合 cheap 与 precise。
3. **Online retrieval during rollout**：policy 在执行过程中实时 retrieve 类似 motion 的 prior trajectory，类似 retrieval-augmented generation (RAG) for IL。
4. **Active retrieval**：policy 在不确定的 state 主动 query prior 数据库，而不是 offline 一次过。
5. **Cross-embodiment flow normalization**：把 optical flow 投影到一个 "canonical motion space" 消除 embodiment 差异，让 Franka 的 motion 和 WidowX 的 motion 直接可比。

## 参考资料与相关 reading

- [FLOWRETRIEVAL 项目主页](https://flow-retrieval.github.io)
- [Diffusion Policy (Chi et al., RSS 2023)](https://diffusion-policy.cs.columbia.edu/) — FLOWRETRIEVAL 的 policy backbone
- [Behavior Retrieval (Du et al., RSS 2023)](https://arxiv.org/abs/2305.14829) — 视觉 retrieval 的 baseline
- [SAILOR (Nasiriany et al., CoRL 2022)](https://arxiv.org/abs/2204.05036) — skill latent space 的 baseline
- [GMFlow (Xu et al., CVPR 2022)](https://arxiv.org/abs/2111.13681) — 用的 optical flow 估计器
- [BridgeData V2 (Walke et al., CoRL 2023)](https://arxiv.org/abs/2308.12952) — real robot prior dataset
- [DROID Dataset (Khazatsky et al., 2024)](https://droid-dataset.github.io/) — 大规模 wild prior
- [LIBERO Benchmark (Liu et al., NeurIPS 2023)](https://libero-project.github.io/) — simulation benchmark
- [Open X-Embodiment (Octo Model Team et al., 2023)](https://robotics-transformer-x.github.io/) — cross-embodiment dataset
- [APT - Any-Point Trajectory Modeling (Wen et al., 2024)](https://arxiv.org/abs/2401.00025) — 类似的 mid-level motion representation for policy
- [Track2Act (Bharadhwaj et al., 2024)](https://track2act.github.io/) — 从 internet video 预测 point tracks 做 zero-shot manipulation
- [RoboTap (Vecerik et al., 2023)](https://arxiv.org/abs/2310.08939) — dense correspondence 做 few-shot IL
- [DINOBot (Palo & Johns, ICRA 2024)](https://arxiv.org/abs/2403.18045) — vision foundation model 做 retrieval + replay
- [Data Quality in Imitation Learning (Belkhale et al., NeurIPS 2024)](https://arxiv.org/abs/2310.14796) — 解释为什么 data quality > quantity
- [kNN-LM (Khandelwal et al., 2020)](https://arxiv.org/abs/1911.00172) — NLP 里的 retrieval-augmented generation 思路
- [R3M (Nair et al., CoRL 2023)](https://arxiv.org/abs/2203.12601) — robot manipulation 通用 visual representation
- [Voltron (Karamcheti et al., RSS 2023)](https://github.com/surassurab/voltron) — language-driven representation for robotics
- [UniMatching V2 (Xu et al., 2023)](https://arxiv.org/abs/2310.05753) — 统一的 dense correspondence 框架
- [VideoMAE V2 (Li et al., 2023)](https://arxiv.org/abs/2303.12027) — video self-supervised pretraining

整体来看，这篇 paper 的核心贡献是把"motion 相似性"这个 intuition 用 optical flow + VAE 这个简单工具落地了，整个方法谈不上 fancy，但 insight 很 sharp，实验设计能突出方法的核心优势。Pen-in-Cup 的 3.7× result 让人信服。我觉得这是 robotics 里 retrieval-augmented learning 这个方向的一个 nice milestone，未来空间还很大。

---

# FLOWRETRIEVAL 深度解析

## 1. 核心动机与 Insight

这篇 paper 要解决的核心问题: **few-shot imitation learning 中 data scarcity 与 data relevance 的 trade-off**。机器人 policy learning 通常需要数百到上千条 demonstration, 而 few-shot 设定下只有 ~10 条 target task demo。要让 policy 学得好, 需要从 prior dataset $\mathcal{D}_{\mathrm{prior}}$ 中检索相关的数据进行 augment。

但现有 retrieval 方法存在两个极端:
- **Visual similarity (Behavior Retrieval [2], SAILOR [1])**: 在 RGB observation 的 latent space 中检索, 强耦合 visual scene 与 task。例如 target 是"转动 doorknob", prior data 里如果没有 doorknob 的视觉场景, 检索就失效。Paper 里把 Square Assembly task 设计成 adversarial setting (useful data 与 target 视觉背景不同, adversarial data 视觉背景与 target 相同), 直接证伪这类方法。
- **Language similarity (RT-H [3], Concept2Robot [4])**: 高层语义抽象, "open the door" 同时匹配 knob 与 handle, 但两者 low-level motion 完全不同; 同时 "turn on faucet" 的旋转 motion 其实对 "turn doorknob" 有帮助, 但语义检索会漏掉。

FLOWRETRIEVAL 的 key insight: **motion similarity (optical flow) 是介于 visual 与 semantic 之间的 mid-level representation**, 既摆脱 visual scene 的干扰 (因为 flow 只编码 pixel motion, 不受 texture/background 影响), 又比 language 更 fine-grained (能捕捉旋转、推拉等具体 low-level motion)。

这一点让我联想到 [Any-point Trajectory Modeling (APT)](https://arxiv.org/abs/2401.00025) 和 [Track2Act](https://track2act.github.io/) 的工作, 它们同样使用 point tracks / trajectory 作为 mid-level motion representation, 但是用于 zero-shot manipulation 而非 retrieval。还有 [RoboTap](https://arxiv.org/abs/2401.08939) 用 dense correspondence 做 few-shot imitation。FLOWRETRIEVAL 把这类 idea 用到 retrieval 阶段, 是个比较 novel 的角度。

## 2. 方法架构解析

### 2.1 Stage 1: Motion-Centric Pretraining

输入 prior dataset $\mathcal{D}_{\mathrm{prior}}$ 的所有 RGB observations, 用 [GMFlow](https://arxiv.org/abs/2111.13681) 计算每个 state $s_t$ 与 future frame $s_{t+k}$ 之间的 optical flow $f_t$:

$$\mathcal{F}_{\mathrm{prior}} = \{\mathcal{E}_{\mathrm{flow}}(s_t, s_{t+k}) \mid s_t, s_{t+k} \in \mathcal{D}_{\mathrm{prior}}\} \tag{1}$$

**变量解释**:
- $\mathcal{E}_{\mathrm{flow}}$: optical flow estimator, 论文用 GMFlow [32] (global matching with transformer, 比 RAFT 在大位移场景下更鲁棒)
- $s_t$: 时间步 $t$ 的 RGB observation
- $k$: future frame 间隔, 等于下游 policy 的 action chunk 长度 (BC 算法 $k=1$, diffusion policy $k=16$)
- $f_t \in \mathbb{R}^{H \times W \times 2}$: dense 2D displacement field, 每个 pixel 一个 $(\Delta x, \Delta y)$ 向量

然后训一个 VAE 来 embed optical flow 到一个 compact latent space:

$$\mathcal{L}_{\mathrm{VAE}}(\theta, \psi) = \mathbb{E}_{f_t \sim \mathcal{F}_{\mathrm{prior}}} \big[ ||q_\psi(p_\theta(f_t)) - f_t||_2 \big] \tag{2}$$

**变量解释**:
- $p_\theta$: encoder, 参数 $\theta$, 输入 optical flow, 输出 latent vector $z \in \mathbb{R}^d$
- $q_\psi$: decoder, 参数 $\psi$, 从 latent $z$ 重构 optical flow
- 训练时其实应该还有 KL divergence term (standard VAE), 但论文 Eq 2 只写了 reconstruction term, 推测是简化表达

**Intuition**: 为什么需要 VAE? 直接用 raw optical flow 计算距离的问题: (1) 高维 ($H \times W \times 2$), (2) 噪声大 (background pixel 的 flow 通常是 0 或 noise), (3) 缺乏 semantic abstraction。VAE encoder $p_\theta$ 学习把 flow 压缩到一个语义化 latent space, 使得相似 motion (即使 object/scene 不同) 距离接近。这其实是在做 motion 的 contrastive learning, 只是用 reconstruction 而非 contrastive loss。

**关键假设**: 论文声称 "$\mathcal{D}_{\mathrm{prior}}$ 足够大时, VAE 在 $\mathcal{F}_{\mathrm{prior}}$ 上训出来的 latent space 可以 generalize 到 $\mathcal{F}_{\mathrm{target}}$"。理由是 optical flow 的 domain gap 远小于 RGB image 的 domain gap。这点在 Bridge-Pot 实验中得到验证 (ViperX arm vs WidowX arm, 视觉差异大但 motion 类似)。

### 2.2 Stage 2: Data Retrieval

对 target dataset 每个 state 计算 optical flow, 然后 measure 每个 prior data point 与 target 的相似度:

$$\mathcal{S}(s_i) = -\min_{\forall f_j \in \mathcal{F}_{\mathrm{target}}} ||p_\theta(f_i) - p_\theta(f_j)||_2, \quad f_i = \mathcal{F}_{\mathrm{prior}}[i] \tag{3}$$

**变量解释**:
- $\mathcal{S}(s_i)$: prior data point $s_i$ 的相似度分数
- 负号: L2 距离越小 → 相似度越大
- $\min$ 操作: prior data point $s_i$ 与 target 中**最接近**的一个 data point 的距离, 这样保证 prior 中只要有一部分能匹配上 target 中的某一段, 就被认为是相关的

这个 min-over-target 的设计很关键。它意味着 retrieved data 不需要对整个 target trajectory 都相似, 只要能 cover target trajectory 的某一段即可。这正符合 retrieval-based IL 的 intuition: 从 prior 的不同片段中拼凑出 target 的完整 motion。

然后取 top $\delta\%$:

$$\eta = \mathrm{sorted}(\mathcal{S}(s_i) \mid s_i \in \mathcal{D}_{\mathrm{prior}})[\lceil \delta N \rceil] \tag{4}$$

$$\mathcal{D}_{\mathrm{retrieved}} = \{(s, a)_{t:t+k} \mid (s, a)_{t:t+k} \in \mathcal{D}_{\mathrm{prior}} \text{ and } \mathcal{S}(s_t) > \eta\} \tag{5}$$

**变量解释**:
- $\delta$: retrieval ratio (例如 Square Assembly 用 35%, LIBERO-Can 用 10%, Bridge 用 1%)
- $N = |\mathcal{D}_{\mathrm{prior}}|$: prior dataset 大小
- $\lceil \cdot \rceil$: ceiling function
- $\eta$: similarity threshold

**Threshold tuning 是个 limitation**: 论文承认 $\delta$ 需要针对 task 手动调, future work 提到用 annotated boundary data 或 active learning 自动化。这也呼应了 [Du et al. (Behavior Retrieval)](https://arxiv.org/abs/2305.14829) 提到的 "bell-shaped curve" — retrieval 太多引入 noise, 太少 insufficient, optimal 在中间。

### 2.3 Stage 3: Flow-Guided Policy Learning

Policy 学习用 co-training: 每个 batch 一半来自 $\mathcal{D}_{\mathrm{target}}$, 一半来自 $\mathcal{D}_{\mathrm{retrieved}}$。Loss 包含 BC action prediction 与 auxiliary flow prediction:

$$\mathcal{L}(\phi_{\mathrm{enc}}, \phi_{BC}, \phi_{\mathrm{aux}}) = \mathbb{E}_{(s, a)_{i:i+k} \in B} \big[ \mathcal{L}_{BC}(s_i, a_{i:i+k}) + \lambda ||\phi_{\mathrm{aux}}(\phi_{\mathrm{enc}}(s_i)) - \mathcal{E}_{\mathrm{flow}}(s_i, s_{i+k})||_2 \big] \tag{6}$$

**变量解释**:
- $\phi_{\mathrm{enc}}$: visual encoder (从 ImageNet pretrained 权重初始化)
- $\phi_{BC}$: action prediction head (在 diffusion policy 里是 U-Net denoiser)
- $\phi_{\mathrm{aux}}$: flow decoder, 从 bottleneck feature 预测 optical flow
- $\lambda = 0.01$: auxiliary loss weight (在所有实验中固定)
- $\mathcal{L}_{BC}$: 标准 BC supervised L2 loss on action

**关键设计**: auxiliary loss 仅作为 regularizer, 不像 [Wen et al. (APT)](https://arxiv.org/abs/2401.00025) 那样把 flow 作为 bottleneck layer 强制 policy 通过 flow 预测 action。理由: dense visual guidance 包含 end-effector motion 不需要的细节 (如 background motion), 强制 policy 预测它会增加学习复杂度。Auxiliary task 让 encoder "顺便" 编码 motion 信息, 但不阻塞 action gradient。

**Algorithm 1 流程**: 
- Line 1-3: pretrain VAE (一次性, 与 target 无关)
- Line 4-8: retrieve data (每个 target task做一次)
- Line 9-13: co-train policy with target + retrieved + auxiliary flow loss

## 3. 实验数据深度分析

### 3.1 主实验结果 (Table 1 与 Fig. 4 综合)

| Method | Square Assembly | LIBERO-Can | Bridge-Micro | Bridge-Pot | Pen-in-Cup |
|---|---|---|---|---|---|
| BC (target only) | 7% | 64% | 56% | 20% | 23% |
| BC-Co (all prior) | 3% | 16% | 56% | 32% | 12% |
| FlowBC (no retrieval) | 7% | 71% | 48% | 8% | 40% |
| ProprioRetrieval- (no flow loss) | 44% | 73% | 24% | 40% | - |
| ProprioRetrieval | 54% | 81% | 16% | 12% | 44% |
| **FLOWRETRIEVAL** | **55%** | **90%** | **68%** | **64%** | **56%** |

**几个关键观察**:

1. **Square Assembly (adversarial setting)**: BC 7%, BR/SR 也很低 (因为它们检索到 adversarial data), FLOWRETRIEVAL 55%。这是最能体现 motion-based retrieval 价值的 task — useful data 视觉背景与 target 不同, adversarial data 视觉背景与 target 相同, 只有 motion-based retrieval 能正确区分。

2. **Bridge-Pot**: BC 仅 20%, FLOWRETRIEVAL 64%。这里 prior 是 Bridge-V2 dataset (WidowX robot), target 是自己采集的 ViperX robot, 跨 embodiment 视觉差异大, 但 motion pattern 可转移。ProprioRetrieval 12% — 因为 proprioception 跨 embodiment 几乎不可比 (workspace 不同)。

3. **Pen-in-Cup (Wild prior)**: BC-Co 12% (用所有 prior data 反而害了), FLOWRETRIEVAL 56% — **3.7× baseline**。这是 paper headline result, 因为 prior 是 DROID 的 random subset (filter 过 viewpoint), 视觉差异极大, 只有 motion-based retrieval 能挑出真正相关的数据。

4. **ProprioRetrieval 在 LIBERO-Can 表现强 (81%)**: 因为 LIBERO 内部 viewpoint 一致, proprioception (end-effector 位置 delta) 是有效的 motion proxy。但 Bridge 任务 viewpoint 多样, proprioception 完全失效。说明 proprioception 是 cheap but brittle 的 motion representation, optical flow 是 robust 但 expensive 的。

### 3.2 Ablation: Retrieval Strategy (Appendix D.1)

论文对比了 top-$\delta\%$ vs KNN 两种 retrieval 策略:
- **Top-$\delta\%$**: 给每个 prior data point 一个 score (与 target 的 min 距离), 取 top $\delta\%$。问题是 target trajectory 中 "pause" 段 (robot 不动) 会有很多 prior data 匹配上 (因为静止状态的 flow 都是 0), 导致 retrieved data 偏向静止。
- **KNN**: 给每个 target data point 找 top-k 个最近的 prior data point。问题是要 force 每个 target point 都有匹配, 当某些 target point 没有真正相似 prior 时, 会 retrieve 到 dissimilar 数据。

实验结果 (KNN, 与 top-% 等量): Square Assembly 40% (-15%), LIBERO-Can 80% (-10%), Bridge-Micro 60% (-8%)。KNN 全面 worse, 说明 force uniform retrieval 反而引入 noise。

### 3.3 Ablation: Pretrained Visual Representations (Appendix D.2)

用 Voltron, R3M, CLIP, DINOv2 的 feature delta 作为 motion representation 做 retrieval, 在 Square Assembly 上都失败 — 这些预训练 visual encoder 偏向 scene semantics 而非 motion, 无法 bypass adversarial data。这印证了 optical flow 作为 motion-specific representation 的必要性。

这让我想到一个 open question: 是否可以训一个专门的 motion foundation model (类似 [VideoMAE](https://arxiv.org/abs/2203.12602) 或 [UniMatching](https://arxiv.org/abs/2303.01240))? 论文用 GMFlow 是 frozen 的, 只训 VAE embedding, 是否 VAE encoder 可以替换成更大规模的 motion pretraining?

### 3.4 Qualitative Retrieval Analysis (Fig. 6)

Square Assembly 中把 trajectory 分 10 个 bins, 看每个 method 在每个 bin 中 retrieve 的 useful (green) vs adversarial (red) 数据比例:
- BR: 大量 retrieve adversarial data, 尤其是 bottleneck stage (pick up 之后到 transfer 之前)
- SR: 一部分 useful, 一部分 adversarial, 不稳定
- FLOWRETRIEVAL: 在每个 stage 都能 retrieve 大量 useful, 几乎过滤掉所有 adversarial

这个 figure 直观展示了 motion-based retrieval 的 robustness — useful data 视觉不同但 motion 相似, 被 flow latent space 正确识别。

## 4. 与相关工作的对比联想

### 4.1 Retrieval-based IL 谱系

- **[Behavior Retrieval (Du et al., RSS 2023)](https://arxiv.org/abs/2305.14829)**: VAE on state-action pairs, visual similarity dominant
- **[SAILOR (Nasiriany et al., CoRL 2022)](https://arxiv.org/abs/2204.05036)**: inverse dynamics pretrained latent skill space, 对 long-horizon 任务设计
- **[DINOBot (Palo & Johns, ICRA 2024)](https://arxiv.org/abs/2403.18045)**: 用 vision foundation model 在 annotated bottleneck state 做 retrieval + replay
- **FLOWRETRIEVAL (本作)**: optical flow VAE, motion-centric, policy-agnostic

### 4.2 Flow / Trajectory-based Policy Learning

- **[Diffusion Policy (Chi et al., RSS 2023)](https://diffusion-policy.cs.columbia.edu/)**: FLOWRETRIEVAL 的 backbone, action chunking $k=16$
- **[APT (Wen et al., 2024)](https://arxiv.org/abs/2401.00025)**: any-point trajectory modeling 作为 policy bottleneck
- **[Track2Act (Bharadhwaj et al., 2024)](https://track2act.github.io/)**: 从 internet video 预测 point tracks 实现 zero-shot manipulation
- **[RoboTap (Vecerik et al., 2023)](https://arxiv.org/abs/2310.08939)**: dense correspondence for few-shot imitation
- **[Learning to Act from Actionless Video (Ko et al., 2023)](https://arxiv.org/abs/2310.08576)**: dense correspondence bridges video 与 robot action

FLOWRETRIEVAL 与这些工作的核心区别: 把 motion representation 用在 **retrieval stage** (而非 policy 内部), 解耦 "retrieve what" 与 "how to use"。

### 4.3 Foundation Models for Robotics

- **[Octo](https://octo-models.github.io/)**: open-source generalist policy, multi-task pretraining
- **[Open X-Embodiment](https://robotics-transformer-x.github.io/)**: 大规模 cross-embodiment dataset
- **[RT-2](https://robotics-transformer2.github.io/)**: VLM for robotic control, language-conditioned

这些工作倾向于 scale up pretraining, 而 FLOWRETRIEVAL 走 retrieval-based 路线, 在 small target data 场景下更 sample efficient。两条路线未来可能融合: 大模型做 semantic retrieval, flow VAE 做 motion retrieval, 互补。

## 5. Limitations 与 Future Directions

论文自己指出:
1. **Scalability**: 每个 new target task 需要重新 scan + sort 所有 prior data, $\mathcal{O}(|\mathcal{D}_{\mathrm{prior}}| \times |\mathcal{D}_{\mathrm{target}}|)$ 的 pairwise distance 计算。Caching embeddings 与 sub-sampling 可以缓解, 但实际部署仍是 challenge。
2. **Threshold tuning**: $\delta$ 需要手动调, 不同 task 最优值差异大 (1% ~ 35%)。

我补充几个潜在方向:
1. **Learned retrieval threshold**: 用 target data 自身的 intra-cluster distance 分布, 或一个小 validation set 自动调 $\delta$
2. **Flow foundation model**: 替换 GMFlow + VAE, 用大规模 video pretraining 的 motion representation (类似 [UniMatching V2](https://arxiv.org/abs/2310.05753))
3. **Hierarchical retrieval**: 先 language filter 大集合, 再 flow filter 小集合, 兼顾 speed 与 precision
4. **Online retrieval**: 在 policy rollout 过程中动态 retrieve, 而非 offline 一次, 类似 [kNN-LM](https://arxiv.org/abs/1911.00172) 的思路
5. **Cross-embodiment flow**: Bridge-Pot 跨 embodiment (WidowX → ViperX) 已经能 work, 但 Pen-in-Cup 的 Wild prior 是否能跨更大 embodiment gap (例如 DROID 中 Franka → target Franka)? 论文没明确测试这个

## 6. Take-aways for Building Intuition

- **Optical flow 是 mid-level motion representation 的 sweet spot**: 比 RGB 抽象 (摆脱 scene texture), 比 language 具体 (捕捉 low-level motion)
- **VAE embedding optical flow 是关键**: raw flow 太 high-dim, 直接距离无意义; latent space 让相似 motion 聚类
- **Auxiliary loss 设计哲学**: dense visual guidance 当 regularizer, 不当 bottleneck, 兼顾 representation richness 与 action gradient flow
- **Min-over-target similarity**: 允许 prior 数据只 cover target 的某一段, 灵活匹配
- **Top-% > KNN**: 不强制 uniform retrieval, 让相似度自然决定 retrieve 哪些 prior, 反而更 robust

参考资料:
- [FLOWRETRIEVAL Project Page](https://flow-retrieval.github.io)
- [Diffusion Policy (RSS 2023)](https://diffusion-policy.cs.columbia.edu/)
- [Behavior Retrieval (RSS 2023)](https://arxiv.org/abs/2305.14829)
- [SAILOR (CoRL 2022)](https://arxiv.org/abs/2204.05036)
- [GMFlow (CVPR 2022)](https://arxiv.org/abs/2111.13681)
- [BridgeData V2 (CoRL 2023)](https://arxiv.org/abs/2308.12952)
- [DROID Dataset](https://droid-dataset.github.io/)
- [LIBERO Benchmark (NeurIPS 2023)](https://libero-project.github.io/)
- [Open X-Embodiment](https://robotics-transformer-x.github.io/)
- [Octo Model](https://octo-models.github.io/)
- [APT - Any-Point Trajectory Modeling](https://arxiv.org/abs/2401.00025)
- [Track2Act](https://track2act.github.io/)
- [RoboTap](https://arxiv.org/abs/2310.08939)
- [DINOBot (ICRA 2024)](https://arxiv.org/abs/2403.18045)
- [RoboFlamingo / Voltron](https://github.com/RoboFlamingo/RoboFlamingo)
- [R3M](https://arxiv.org/abs/2203.12601)
- [UniMatching V2 (optical flow foundation)](https://arxiv.org/abs/2310.05753)
