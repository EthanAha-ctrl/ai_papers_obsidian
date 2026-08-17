---
source_pdf: V-JEPA 2 Self-Supervised Video Models Enable Understanding, Prediction
  and Planning.pdf
paper_sha256: 9cfcfde5fb0d9730637da5b9e7317825c3f3d09e91f3553e22eeba42c74d2226
processed_at: '2026-08-12T23:59:21-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# V-JEPA 2 人话版

## 一句话讲完

让 AI 看一百万小时 YouTube 视频,学会"世界大概怎么运作",然后只给它看 62 小时机器人操作视频,它就能在新环境里零样本抓东西放东西。关键是学习方式:别去预测每个像素,去预测"事情的要点"。

---

## 大背景:为什么这件事难

你想让机器人帮你倒杯水。传统做法有两条路:

**路线 A:手写规则**。告诉机器人"看到杯子→移动到杯子上方→下降→合拢夹爪→抬起"。问题是换个形状的杯子、桌子高度变了、光线变了,规则就失效。

**路线 B:大量示教数据**。让人类遥操作机器人几千次,机器人模仿学习。问题是数据贵、泛化差,换个 lab 就不工作。

人类是怎么学会的?小孩看大人做事,看几个月,然后自己试试就成了。不需要每个动作都教,看多了自然懂"物理"——推东西东西会动,松手东西会掉,抓得紧才拿得起来。

**V-JEPA 2 想干的就是这件事**:让 AI 通过"看视频"学会物理常识,再用很少的机器人数据把这些常识"接上"动作。

---

## 核心思想:别预测像素,预测要点

这是 LeCun 多年鼓吹的 JEPA 哲学,这篇 paper 是第一次把它真正做大并跑通。

打个比方。你看一段视频:一个人在草地上踢球。

**Generative model** (比如 Sora、Cosmos) 的做法:预测下一帧每一个 pixel 的颜色。这意味着模型要预测草地上每一根草在下一秒的具体位置、球上每一个纹路的移动、空气中每一粒灰尘的轨迹。这些东西里大部分是**不可预测**的——草怎么摆动是随机的,你预测不准还要硬预测,就浪费容量。

**JEPA** 的做法:先把视频编码成一组 abstract representation(可以理解为"要点向量"),然后只预测下一时刻的"要点"是什么。草地那根草的具体位置不在要点里,被自动丢弃;球往哪飞、人往哪跑这种结构性信息保留。

这就像读小说 vs 背小说。背小说要记住每一个字,读小说记住"主角爱上了谁、最后结局如何"就够了。前者容量大但泛化差,后者容量小但能举一反三。

V-JEPA 2 的 representation 就是这种"要点"。它通过一个简单任务学到:mask 掉视频一部分,让模型猜被 mask 掉的部分的 representation 是什么。猜着猜着,representation 就把"可预测的世界结构"编码进去了。

---

## 怎么训练:把视频打码,让模型补全

具体训练 task 非常简单:

1. 拿一段视频,切成小方块 (patches)
2. 随机扔掉一部分 patches (masking)
3. 剩下的 patches 喂给 encoder,输出每个 patch 的 representation
4. 加上一些"占位符" (mask tokens) 表示被扔掉的位置
5. 喂给 predictor,让它猜被扔掉 patch 的 representation
6. 跟一个 EMA 版本的 encoder 输出做 L1 loss

EMA + stop-gradient 是防止模型偷懒的技巧。如果没有这个,模型可以直接输出常数让 loss 变 0,啥也不学。EMA teacher 让 target 是个"缓慢变化的标准",模型必须真的去 predict 而不是 collapse。

这个 task 表面无聊,但其实很有信息量。模型要补全被 mask 掉的内容,必须懂:物体不能凭空消失、运动有惯性、遮挡关系、3D 结构...这些就是物理常识。

---

## 怎么 scale:四个方向一起使劲

这是这篇 paper 工程上最实在的部分。要 scale 必须四个方向一起:

**数据**:从 2M 视频扩到 22M 视频混在一起叫 VideoMix22M (VM22M)。包括 Something-Something v2 (手部动作)、Kinetics (各类活动)、HowTo100M (教程)、YT-Temporal-1B (YouTube 通用)、ImageNet (静态图,复制 16 帧当视频用)。

YT1B 数据脏,他们用 cluster-based retrieval 清理:用 DINOv2 提取 embedding,聚成 1.5M cluster,只保留与 Kinetics/SSv2/COIN/EpicKitchen 相似的那 210K cluster。这步让 ViT-L 平均提升 1.4 分。

**模型**:从 ViT-L (300M) 扩到 ViT-g (1B)。Predictor 固定是 ViT-small (22M),只让 encoder 变大。这步提升 1.5 分。

**训练时长**:从 90K iteration 扩到 252K。这步提升 0.8 分。

**分辨率和帧数**:这是最巧的。他们想用 64 帧 384×384 训练,但直接从头训要 60 GPU-year,不可能。他们用 progressive resolution:

- 前 240K iteration:用 16 帧 256×256 训,便宜
- 最后 12K iteration (cooldown):升到 64 帧 384×384,同时 learning rate 线性衰减到接近 0

为什么有效?低分辨率相当于低通滤波,先找到粗的 loss basin;高分辨率 fine-tune 时仍在 good basin 附近,补高频细节就行。这给 8.4× 加速,让 ViT-g 的大分辨率训练变得可行。这步提升 0.7 分。

四个加起来 +4 分,看起来不多,但每个都必要,缺一个都跑不到 SOTA。

---

## Stage 2:接上动作,变成 world model

预训练完的 V-JEPA 2 能"理解"视频,能预测被 mask 掉的部分。但它不知道"如果我做动作 A,世界会怎么变"——因为它没见过动作。

接下来是 post-training 出 V-JEPA 2-AC (Action-Conditioned)。用 Droid 数据集 (https://arxiv.org/abs/2403.12945) 里 62 小时 Franka 机械臂遥操作视频。注意:不需要成功/失败标签、不需要任务标签、不需要 reward。只要 (当前画面, 当前夹爪状态, 下一帧画面) 这种三元组就行。

具体做法:

1. Encoder 冻住,只训一个 300M 参数的新 predictor
2. 输入:当前帧的 representation + 夹爪状态 + 动作(夹爪状态的变化量)
3. 输出:下一帧的 representation 预测
4. Loss:预测的 representation 与真实下一帧 representation 的 L1 距离

用了两种 loss:
- **Teacher-forcing**:每一步都给真实历史,只预测一步。好优化,但推理时有 train-test mismatch (推理时只能用自己之前的预测)
- **Rollout**:把模型自己的预测喂回去,预测两步后。让模型见识自己的误差,减缓 error accumulation

Action 是 7 维向量:3 维位置变化 + 3 维姿态变化 + 1 维夹爪开合。

Predictor 用 block-causal attention:每个时间步的 patches 互相能看 (spatial reasoning),但只能看过去时间步 (temporal causality)。3D-RoPE 给位置编码,把维度三分给时间、高度、宽度。

---

## 怎么用:规划 = 找让想象接近 goal 的动作

这是我觉得最优雅的部分。给定一张目标图片(比如"杯子放在左边")和当前画面,怎么让机器人动起来?

思路:模型能"想象"动作后的状态。那就找一组动作,让模型想象出来的状态 representation 尽量接近目标图片的 representation。数学上就是最小化:

```
energy = || world_model.imagine(action_sequence) - goal_representation ||_1
```

这叫 energy-based planning。Energy 越低,说明这个动作序列越能让想象接近目标。

怎么优化这个 energy?用 Cross-Entropy Method (CEM),一种零阶优化:

1. 初始化一组高斯分布 (均值 0,方差 1)
2. 从分布里采样 800 个动作序列
3. 对每个序列算 energy
4. 选 energy 最低的 top-10 当"精英"
5. 用精英的均值方差更新高斯分布
6. 重复 10 次
7. 最后分布的均值就是选中的动作序列

执行第一个动作,然后重新规划 (receding horizon control)。每步 16 秒。

为什么这个能 work?看 paper 的 Figure 9,他们画了 energy landscape:在正确动作附近达到最小值,而且**平滑、局部凸**。这意味着 CEM 这种简单采样方法能高效找到最优。

对比一下 Cosmos (latent diffusion 7B) 在同样任务上:每步规划要 4 分钟,而且成功率更低。为什么?因为 pixel-space 的 energy landscape 是高度非凸、多模态的,采样优化非常难。这正是 JEPA 路线的根本优势——representation space 对 planning 友好。

---

## 结果:真的能在新 lab 抓东西

两个不同的 lab,Franka 机械臂,各 10 次试验。V-JEPA 2-AC 没在这两个 lab 收过任何数据,直接 zero-shot 部署:

| 任务 | V-JEPA 2-AC | Octo (VLA baseline) | Cosmos (generation baseline) |
|---|---|---|---|
| Reach (到指定位置) | 100% | 100% | 80% |
| Grasp Cup | 65% | 15% | 0% |
| Grasp Box | 25% | 0% | 20% |
| Pick-Place Cup | 80% | 15% | 0% |
| Pick-Place Box | 65% | 10% | 0% |

V-JEPA 2-AC 全面领先。Grasp Box 比较难(只有 25%),因为箱子需要精确的夹爪开合距离,模型还不够准。但 Pick-Place 能 80% 已经超出预期——这是组合任务 (grasp + reach + place),需要模型理解一连串动作的因果。

Pick-Place 用了 3 个 sub-goal 图片:第一个"抓住物体",第二个"物体在目标附近",第三个"物体在目标位置"。模型自动在 sub-goal 之间切换。这是 hierarchical planning 的雏形,虽然还很原始。

---

## 还有什么用:理解和预测

World model 不只用来控制机器人,本身也是"理解"的基础。

**动作分类**:在 Something-Something v2 (https://arxiv.org/abs/1706.04230,需要理解手部动作的任务) 上,V-JEPA 2 拿 77.3% top-1 accuracy。对比:
- DINOv2 (最强的 image SSL): 50.7%
- InternVideo2 (用 language supervision): 69.7%
- V-JEPA 2: 77.3%

差距巨大。证明 video SSL 学到的 representation 本质上 encode 了 motion 信息。

**动作预测**:在 Epic-Kitchens-100 (https://arxiv.org/abs/2202.05087) 上,给定视频 context 预测 1 秒后会发生什么动作。V-JEPA 2 拿 39.7 recall@5,比之前 SOTA (PlausiVL, 8B 参数,用 LLM) 提升 44%。V-JEPA 2 只用 300M 参数就 beat 8B baseline。

**视频问答**:把 V-JEPA 2 接到 LLM 上做 video QA。这是第一次"没用 language supervision 预训练的 video encoder"能做出 SOTA 的 MLLM。传统观念认为 vision encoder 必须用 image-text contrastive pretrain 才能对齐 language,V-JEPA 2 打破了这个观念。

在 PerceptionTest (https://arxiv.org/abs/2305.04657) 拿 84.0,在 MVP 拿 44.5,在 TempCompass 拿 76.9,在 TemporalBench 拿 36.7,在 TOMATO 拿 40.3,都是 8B 模型级别 SOTA。特别是在需要时间理解的 benchmark 上,V-JEPA 2 大幅领先 image encoder (DINOv2, SigLIP2, Perception Encoder)。

---

## 为什么这个工作重要

几个层次:

**对 LeCun 路线的验证**:LeCun 在 2022 年 position paper 里说 JEPA 是通向 autonomous machine intelligence 的正确路线,但当时只有 small-scale demo。V-JEPA 2 第一次把这条路 scale 到 1B 参数 + 1M 小时数据,并 demonstrate end-to-end from pretrain → robot deployment。这是 JEPA 从"哲学主张"变成"可行方案"的里程碑。

**对 self-supervised learning 的证明**:不需要 language supervision,不需要 task label,只靠"看视频 + mask denoising"就能学到 general visual representation,既能做理解也能做控制。这支持了 "perception is inference" 的认知科学观点。

**对机器人学习的启示**:传统机器人学习需要大量示教,且每个 task 单独训。V-JEPA 2-AC 只用 62 小时 unlabeled video,就能 zero-shot 部署到新环境做多个 task。这种"internet video 学常识 + 少量 robot video 学动作"的范式,可能是机器人 learning 的 scalable 路径。

**对 generative AI 的反思**:过去两年大家疯狂做 diffusion / autoregressive generation,觉得生成越逼真越接近"理解世界"。V-JEPA 2 提出反论:generation 和 understanding 是不同的能力,planning 需要的是 abstract prediction 不是 pixel synthesis。Cosmos 生成很逼真但 planning 不行;V-JEPA 2 不生成但 planning 很行。

---

## 局限和未来

**Long-horizon planning 不行**:目前只能 reliable 预测 ~16 秒。Pick-Place 必须给 3 个 sub-goal 图片,不是真的 autonomous。要做"帮我做晚饭"这种 long-horizon 任务,需要 hierarchical planning——LeCun 倡导的 H-JEPA。

**Camera 位置敏感**:模型从单目 RGB 隐式推断动作坐标轴,如果相机位置变,推断会偏。虽然可以通过 unsupervised calibration 缓解,但根本解决需要 3D-aware representation。

**Grasp Box 只有 25%**:精确 manipulation 还不够好。可能需要更精细的 action representation 或更多 data。

**Action space 受限**:每个动作被限制在 L1-ball radius 0.075 内 (~13cm 位移),大动作不行。这限制了 model 的决策范围。

**没用 language**:目前 goal 必须是 image。未来要接 language,让用户说"把杯子放左边"就行。Section 7 的 LLM alignment 是个起点,但还需要让 LLM 把 language goal 翻译到 V-JEPA 2-AC 的 representation space。

**Real-time 还差**:目前 16 秒/action,工业应用要 10Hz+。可以训个 feedforward policy 在 world model 的 imagination 里 imitation learn,作为 planning 的 initialization,大幅提速。

---

## 我自己的几个直觉

**直觉一**:JEPA 的核心 insight 是"好的 representation 是 task-agnostic 的 world model"。V-JEPA 2 用同一个 representation 服务 classification、anticipation、planning、VidQA,这不是巧合,是因为它学到的是"世界怎么运作"这种最 general 的东西。这与 LLM 用一个 backbone 做 translation、summarization、code 生成是同一种哲学。

**直觉二**:"预测要点"比"预测像素"更接近生物大脑。人脑不会逐像素想象未来,人想象的是"会发生什么",是 abstract 的。JEPA 这种 representation-space prediction 与 predictive coding (Rao & Ballard 1999, https://www.nature.com/articles/nn.210) 和 free energy principle (Friston 2010, https://www.nature.com/articles/nrn2787) 的脑科学理论一致。

**直觉三**:Progressive resolution training 是大 model 训练的通用 trick。低分辨率找 coarse basin,高分辨率 fine-tune。这跟 coarse-to-fine optimization、multi-grid methods、curriculum learning 都是一脉相承的。ViT-g 这种 1B 模型之所以能训出来,这种工程 trick 至关重要。

**直觉四**:"62 小时 robot data"这个数字看起来小,但比例是 1M:62 ≈ 16000:1。这跟 LLM 里 pretrain:text vs instruction tuning:text 的比例类似。说明 self-supervised pretrain + 少量 supervised post-train 是通用范式,不只适用于 language。

**直觉五**:V-JEPA 2-AC 的 300M predictor 只学"action 如何在 representation space 推进 state"。这意味着 representation space 必须"对 action 友好"——同一个 action 在不同状态下产生可预测的 representation 变化。这是为什么 V-JEPA 2 pretrain 重要:它学到的 representation 不只 encode 静态语义,还 encode dynamic affordances。

**直觉六**:Representation-space planning 比 pixel-space planning 快 15 倍不是工程优化,是 fundamental advantage。Energy landscape 平滑性决定了搜索效率。这跟 classical planning 里"在 abstract space plan,在 concrete space execute"的思想一致——VIQA、PDDL、AlphaGo 的 policy network 都是抽象空间决策。

**直觉七**:V-JEPA 2 没用 language 就能对齐 LLM,说明 vision encoder 不必依赖 language supervision。Language supervision (CLIP-style) 的好处是 semantic alignment,代价是丢了 pure visual 的 fine-grained 信息。V-JEPA 2 反其道而行,先学 pure visual,再在 alignment 阶段补 language。可能这条路在 long-form video、robotics、embodied AI 上更有后劲。

---

## 一个画面总结

把 V-JEPA 2 想象成一个小孩:

1. **婴儿期 (Stage 1 pretrain)**:看 100 万小时 YouTube,学会"世界大概怎么运作"——gravity、object permanence、occlusion、motion。这些是 task-agnostic 的物理常识。

2. **幼儿期 (Stage 2 post-train)**:大人给一个机械臂,让它随便玩 62 小时。它把"动作"跟"之前学的世界模型"接上——哦,这个动作会让夹爪这样动,夹爪动了那个东西也会跟着动。

3. **上学 (LLM alignment)**:再教它说话,把视觉概念跟词对应起来。现在它能回答"视频里发生了什么"。

4. **考试 (zero-shot deployment)**:带它去新厨房,说"把那个杯子放到那边"。它从没见过这个厨房,但用学到的世界模型想象"如果做这个动作,杯子会到那里",规划动作,执行。

这就是 paper 的故事。从"看视频学物理"到"用物理做规划",一条 line 走通。虽然现在还粗糙 (25% Grasp Box),但这是第一次把 LeCun 的 JEPA 路线从 paper 变成 working system。

---

## 一些值得继续挖的方向

如果你想继续 deep dive:

1. **I-JEPA (Assran et al. 2023, https://arxiv.org/abs/2301.08243)**:JEPA 的 image 版本,V-JEPA 2 的 predecessor。
2. **V-JEPA 原版 (Bardes et al. 2024, https://arxiv.org/abs/2404.08471)**:第一版 video JEPA,V-JEPA 2 直接 build on 这个。
3. **DINO-WM (Zhou et al. 2024, https://arxiv.org/abs/2411.04983)**:类似思路,但用 DINO features 做 world model。
4. **DreamerV3 (Hafner et al. 2023, https://arxiv.org/abs/2301.04104)**:RL world model 经典,需要 reward。
5. **π0 (Black et al. 2024, https://arxiv.org/abs/2410.24164)**:VLA 路线代表,对比阅读理解 planning vs policy learning 的 tradeoff。
6. **LeCun 2022 position paper (https://openreview.net/pdf?id=BZ5a1r-kVsf)**:JEPA 原始 manifesto。
7. **Cross-Entropy Method tutorial**:CEM 是 planning 的核心,可以读 Rubinstein 原始 paper (https://www.sciencedirect.com/science/article/pii/S0377221797003827) 或 De Boer et al. 2005 tutorial。

希望这人话版本能 build 你的 intuition。核心抓住一句话:**学"要点"不学"像素",representation space 的 world model 是 planning 的正确 substrate**。剩下都是 engineering 把这个 idea scale 到 work。

---

# V-JEPA 2 深度解读: Self-Supervised Video Models Enable Understanding, Prediction and Planning

## 0. 一句话直觉

V-JEPA 2 是 FAIR 把 LeCun 提倡多年的 **JEPA (Joint-Embedding Predictive Architecture)** 路线第一次 scale 到 1B 参数 + 1M 小时视频，然后只用 62 小时 unlabeled robot video post-train 出一个 action-conditioned world model，在两个 lab 的 Franka arm 上 zero-shot 完成 pick-and-place。核心 claim 是: **在 representation space 做 prediction 比 pixel space 做 generation 更适合 planning**，并且 self-supervised 视频预训练学到的 representation 本身就是某种 "world model"。

Paper link: https://ai.meta.com/vjepa  
Code: https://github.com/facebookresearch/vjepa2  
Blog: https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks

---

## 1. 哲学动机: 为什么是 JEPA

LeCun 在 2022 年的 "A Path Towards Autonomous Machine Intelligence" position paper (https://openreview.net/pdf?id=BZ5a1r-kVsf) 中提出 JEPA，与当时主流的 generative model (VAE, diffusion, autoregressive pixel prediction) 有本质区别。

Generative model 必须在 **pixel space** 重建每一个 pixel。这意味着模型要把概率质量分给所有可观察的细节，包括那些 **本质上不可预测** 的高频细节（比如草地上每一根草的具体位置，树叶在风中精确的摆动）。这些细节对 planning 没有任何用处，但会消耗 model capacity 和 optimization 信号。

JEPA 把 prediction 搬到 **learned representation space**。模型只需要预测那些 **可预测的结构性 aspects**（运动物体的轨迹、动作的因果效应），让 representation 学会自动丢弃 unpredictable details。这给 planning 带来巨大好处：energy landscape 平滑、locally convex（见 Figure 9），可以用简单的 Cross-Entropy Method (CEM) 高效优化。

V-JEPA 2 vs Cosmos 的对比就体现了这一点：Cosmos (latent diffusion 7B, Agarwal et al. 2025, https://arxiv.org/abs/2501.03575) 在 Droid 上 fine-tune 后做 MPC，需要 4 分钟/action；V-JEPA 2-AC 只要 16 秒/action，且 success rate 更高。

---

## 2. Stage 1: V-JEPA 2 Pretraining

### 2.1 Mask-Denoising Objective (公式 1)

$$
\underset{\theta, \phi, \Delta_y}{\text{minimize}} \quad \| P_\phi(\Delta_y, E_\theta(x)) - \text{sg}(E_{\bar{\theta}}(y)) \|_1
$$

变量解释：
- $E_\theta(\cdot)$: encoder，参数 $\theta$，ViT-g (1B params)
- $P_\phi(\cdot)$: predictor，参数 $\phi$，固定为 ViT-small (22M params)
- $\Delta_y$: learnable mask token，编码 dropped patches 的位置
- $x$: masked view of video $y$（部分 patches 被 drop）
- $\bar{\theta}$: encoder 的 exponential moving average (EMA) weights，$\bar{\theta} \leftarrow \alpha \bar{\theta} + (1-\alpha)\theta$
- $\text{sg}(\cdot)$: stop-gradient，target 不参与反传
- $\|\cdot\|_1$: L1 loss（比 L2 对 outlier robust）
- Loss 只在 masked patches 上计算

为什么需要 EMA + stop-gradient？这是 BYOL (Grill et al. 2020, https://arxiv.org/abs/2006.07733) 一脉相承的设计。如果 encoder 和 target 是同一 network 同时训练，模型可以 collapse 到 constant mapping 让 loss trivially 0。EMA + stop-gradient 阻止这种 trivial solution，迫使 representation 编码可预测的信息。

### 2.2 关键 Scaling Ingredients

四条 scaling axes:

| Axis | 从 | 到 | 提升 |
|---|---|---|---|
| Data | VM2M (2M videos) | VM22M (22M videos) | +1.0 avg |
| Model | ViT-L (300M) | ViT-g (1B) | +1.5 avg |
| Training | 90K iter | 252K iter | +0.8 avg |
| Resolution | 16f@256² | 64f@384² | +0.7 avg |

Cumulative +4.0 points on average accuracy across 6 understanding tasks (SSv2, Diving-48, Jester, Kinetics, COIN, ImageNet)。

### 2.3 数据组成 (VM22M)

| Source | Samples | Hours | Weight | Type |
|---|---|---|---|---|
| SSv2 (https://arxiv.org/abs/1706.04230) | 168K | 168 | 0.056 | EgoVideo |
| Kinetics 400/600/700 (https://arxiv.org/abs/1705.06950) | 733K | 614 | 0.188 | ExoVideo |
| HowTo100M (https://arxiv.org/abs/1906.03327) | 1.1M | 134K | 0.318 | ExoVideo |
| YT-Temporal-1B curated | 19M | 1.6M | 0.188 | ExoVideo |
| ImageNet | 1M | n/a | 0.250 | Images |

ImageNet 图像被 temporal-duplicated 成 16-frame video（每帧相同）来支持 joint image+video training。这个 trick 让 model 同时学到 appearance (静态) 和 motion (动态) features。

### 2.4 YT1B Curation: Cluster-Based Retrieval

YT1B 完全 uncurated，会污染 representation。作者用 DINOv2 ViT-L 提取每个 scene 的 embedding，cluster 成 1.5M clusters，然后以 Kinetics/SSv2/COIN/EpicKitchen 为 target distribution，只保留有 target 视频映射到的 cluster (210K clusters, 115M scenes)。

Cluster sampling weight:
$$
w_c = \sum_{d=1}^{D} w_d \times \frac{N_{d,c}}{N_d}
$$

变量解释：
- $w_c$: cluster $c$ 的 sampling weight
- $w_d$: target dataset $d$ 的 weight (来自 Table 11，K710=0.7, SSv2=0.125, COIN=0.125, EK=0.05)
- $N_{d,c}$: dataset $d$ 中分到 cluster $c$ 的 sample 数
- $N_d$: dataset $d$ 的 total sample 数
- $D = 4$ (4 个 target dataset)

效果：在 ViT-L 上 curated YT1B 比 uncurated 提升 +1.4 avg (Figure 4 right)。

### 2.5 Progressive Resolution Training（最巧妙的工程 trick）

如果要从头训练 ViT-g 在 64 帧 384×384 输入，需要 ~60 GPU-years（Figure 5 middle）。

解决方法：**Warmup-Constant-Decay** learning rate schedule 配合 progressive resolution。

| Phase | Iterations | Frames | Resolution | LR |
|---|---|---|---|---|
| Warmup | 0–12K | 16 | 256×256 | linear warmup → 5.25e-4 |
| Constant | 12K–240K | 16 | 256×256 | 5.25e-4 恒定 |
| Cooldown (decay) | 240K–252K | 64 | 256/384/512 | linear decay → 1e-6 |

Key insight: 大部分 capacity 在 constant phase 用 cheap short clip 学到，cooldown phase 只做最后的 resolution/duration 适配。这给 8.4× speedup（Figure 5 middle）。

为什么 cooldown 升 resolution 有效？因为在低分辨率长训找到的 loss basin 在高分辨率下仍是 good basin，只需要 fine-tune。这是 optimization geometry 的 inductive bias：低分辨率等价于 low-pass filter，找到 coarse structure；高分辨率补充 high-frequency detail。

类似 trick 在 Touvron et al. 2019 fix-restrain (https://arxiv.org/abs/1906.06423) 和 Oquab et al. 2023 DINOv2 (https://arxiv.org/abs/2304.07193) 中也用过。

### 2.6 Architecture 细节

- Patchify: tubelets size $2 \times 16 \times 16$ (T×H×W)
- Position encoding: **3D-RoPE** (Rotary Position Embedding, Su et al. 2024, https://arxiv.org/abs/2104.09864)
  - Feature dim 分成 3 段，分别对应 temporal/height/width 轴
  - 1D RoPE 在每段独立应用
  - 比 absolute sincos 在 ViT-g 上更稳定
- Masking: multi-block masking (Bardes et al. 2024, https://arxiv.org/abs/2404.08471)
  - Spatial mask scale [0.15, 0.7]
  - Temporal mask scale [1.0, 1.0] (整段时间都 mask)
  - Mask aspect ratio [0.75, 1.5]

---

## 3. Stage 2: V-JEPA 2-AC (Action-Conditioned World Model)

### 3.1 输入格式

每个 mini-batch 采样 4 秒 video clip (16 frames @ 4 fps, 256×256):
- $(x_k)_{k \in [16]}$: video frames
- $(s_k)_{k \in [16]}$: end-effector state, 7D 向量
  - $s_k[0:3]$: cartesian position (3D)
  - $s_k[3:6]$: extrinsic Euler angles (3D orientation)
  - $s_k[6]$: gripper state (open/close)
- $(a_k)_{k \in [15]}$: actions = adjacent frame state differences, 7D
  - $a_k = s_{k+1} - s_k$

数据：Droid dataset (Khazatsky et al. 2024, https://arxiv.org/abs/2403.12945)，只用 left camera view，丢弃 <4s 的 clip，留下 <62 小时 video。

### 3.2 Loss 设计

Encoder $E$ 保持 frozen，每帧独立编码：
$$
z_k := E(x_k) \in \mathbb{R}^{H \times W \times D}
$$
其中 $H \times W = 16 \times 16$, $D = 1408$ (ViT-g embedding dim)。

**Teacher-Forcing Loss** (公式 2):
$$
\mathcal{L}_{\text{tf}}(\phi) := \frac{1}{T} \sum_{k=1}^{T} \| \hat{z}_{k+1} - z_{k+1} \|_1 = \frac{1}{T} \sum_{k=1}^{T} \left\| P_\phi\left((a_t, s_t, E(x_t))_{t \leq k}\right) - E(x_{k+1}) \right\|_1
$$
其中 $T = 15$。

**Rollout Loss** (公式 3):
$$
\mathcal{L}_{\text{rollout}}(\phi) := \| P_\phi(a_{1:T}, s_1, z_1) - z_{T+1} \|_1
$$
实践中 $T = 2$，只 backpropagate through one recurrent step。

**Total Loss** (公式 4):
$$
L(\phi) := \mathcal{L}_{\text{tf}}(\phi) + \mathcal{L}_{\text{rollout}}(\phi)
$$

为什么需要 rollout loss？推理时模型 autoregressive rollout，teacher-forcing 只让模型见 ground-truth 历史，inference 时见自己的 prediction，会有 train-test mismatch。Rollout loss 让模型在训练时 experience 自己的 prediction error，缓解 error accumulation。这是 scheduled sampling (Bengio et al. 2015, https://arxiv.org/abs/1506.03099) 的精神，但用 differentiable 2-step 实现，避免 RL 的 instability。

### 3.3 Predictor Architecture

- 300M params, 24 layers, 16 heads, hidden 1024, GELU
- Block-causal attention: 每个 patch at time $k$ 可以 attend 到
  - 同时间步的所有 patches + action + state (block)
  - 所有之前时间步的 patches + actions + states (causal)
- 3D-RoPE for video patches, 1D temporal RoPE for action/state tokens
- Inputs (action, state, flattened feature map) 各自通过 learnable affine projection 到 hidden dim
- Output 通过 learnable affine projection 回到 encoder 的 embedding dim (1408)

### 3.4 Planning via Energy Minimization (公式 5)

给定：
- Current frame $x_k$, encode to $z_k$
- Current end-effector state $s_k$
- Goal image $x_g$, encode to $z_g$

求最优 action sequence:
$$
(a_i^\star)_{i \in [T]} := \underset{\hat{a}_{1:T}}{\text{argmin}} \ \mathcal{E}(\hat{a}_{1:T}; z_k, s_k, z_g)
$$
其中 energy function:
$$
\mathcal{E}(\hat{a}_{1:T}; z_k, s_k, z_g) := \| P(\hat{a}_{1:T}; s_k, z_k) - z_g \|_1
$$

物理意义：找一条 action trajectory，让 world model imagine 出来的 $T$ 步后 state representation 尽量接近 goal representation。L1 distance 衡量 representation 上的"距离"。

### 3.5 Cross-Entropy Method (CEM) 优化

CEM (Rubinstein 1997, https://www.sciencedirect.com/science/article/pii/S0377221797003827) 是 zero-order optimization，无需 gradient:

1. 初始化 Gaussian 分布 $\mathcal{N}(\mu_0, \Sigma_0)$，$\mu_0 = 0$, $\Sigma_0 = I$
2. Iteration $i$:
   - Sample $N$ 个 action sequences $\{\hat{a}^{(j)}_{1:T}\}_{j=1}^N$ from $\mathcal{N}(\mu_i, \Sigma_i)$
   - 对每个 sample 计算 energy $\mathcal{E}(\hat{a}^{(j)}_{1:T}; z_k, s_k, z_g)$
   - 选择 top-$K$ (elites) samples
   - 更新 $\mu_{i+1}, \Sigma_{i+1}$ 为 elites 的 sample mean 和 variance
3. 重复 $I$ 次，返回 $\mu_I$ 作为最优 action sequence
4. Receding horizon: 只执行 $a_1^\star$，下一步重新 plan

实践设置：800 samples, 10 refinement iterations, planning horizon $T=1$（因为任务相对 greedy），16 秒/action。

Action 约束：每个 action 限制在 L1-ball radius 0.075 内，对应 max end-effector displacement ~13 cm/step。

---

## 4. Zero-shot Robot Manipulation Results

### 4.1 任务和成功率 (Table 2)

两个 lab，10 trials/task，object 位置和 starting pose 随机化：

| Method | Reach | Grasp Cup | Grasp Box | Reach w/ Obj Cup | Reach w/ Obj Box | Pick-Place Cup | Pick-Place Box |
|---|---|---|---|---|---|---|---|
| Octo avg | 100% | 15% | 0% | 15% | 70% | 15% | 10% |
| **V-JEPA 2-AC avg** | **100%** | **65%** | **25%** | **75%** | **80%** | **80%** | **65%** |

### 4.2 vs Cosmos (Table 3)

| Method | #Samples | Time/action | Reach | Grasp Cup | Pick-Place Cup |
|---|---|---|---|---|---|
| Cosmos (latent diffusion 7B) | 80 | 4 min | 80% | 0% | 0% |
| **V-JEPA 2-AC** | **800** | **16 sec** | **100%** | **60%** | **80%** |

V-JEPA 2-AC 用 10× samples 但 15× 速度，且性能显著更好。这直接验证了 JEPA 在 planning 上的效率优势。

### 4.3 Energy Landscape 分析 (Figure 9)

对 single-goal reaching 任务，sweep action 的 $\Delta x, \Delta y$ ($\Delta z = 0$ fixed)，绘制 energy surface。

观察：energy function 在 ground-truth action 附近达到 minimum，且 **smooth + locally convex**。这是为什么 CEM 能高效找到 optimal action 的根本原因——optimization landscape friendly。

如果用 pixel-space generation model，energy 是 pixel-level reconstruction error，highly non-convex, multimodal, 难以 optimize。

### 4.4 Pick-and-Place 多 goal 编排

3 个 sub-goal:
- Goal 1: object grasped (前 4 步 optimize)
- Goal 2: object near target location (中间 10 步)
- Goal 3: object at target (最后 4 步)

Total 18 步，自动切换 sub-goal。这是 primitive 的 hierarchical planning 雏形，未来可以扩展到 long-horizon tasks。

### 4.5 Limitation: Camera Sensitivity (Appendix B.4)

V-JEPA 2-AC 隐式从 monocular RGB 推断 action 坐标轴。如果 camera 位置变，inferred axis 会旋转。Figure 16 显示 rotation error 与 camera position 大致线性关系。但有趣的是：error 主要是 rotation (condition number 1.5 接近 rotation matrix)，所以可以 unsupervised calibrate——让 robot 随机动一下，比较 inferred optimal action 与 actual action，solve $W^\star = \text{argmin}_W \|AW - B\|_2$，然后乘上 $W^\star$ 校准。

---

## 5. Understanding: Probe-based Classification (Table 4)

4-layer attentive probe on frozen encoder。Protocol: $16 \times 2 \times 3$ inputs for SSv2 (16 frames, 2 temporal crops, 3 spatial crops)，etc.

| Method | Params | Avg | SSv2 | Diving-48 | Jester | K400 | COIN | IN1K |
|---|---|---|---|---|---|---|---|---|
| DINOv2 (https://arxiv.org/abs/2304.07193) | 1.1B | 81.1 | 50.7 | 82.5 | 93.4 | 83.6 | 90.7 | 86.1 |
| PE_core G (https://arxiv.org/abs/2504.13181) | 1.9B | 82.3 | 55.4 | 76.9 | 90.0 | 88.5 | 95.3 | 87.6 |
| SigLIP2 (https://arxiv.org/abs/2502.14786) | 1.2B | 81.1 | 49.9 | 75.3 | 91.0 | 87.3 | 95.1 | 88.0 |
| V-JEPA ViT-H (https://arxiv.org/abs/2404.08471) | 600M | 85.2 | 74.3 | 87.9 | 97.7 | 84.5 | 87.1 | 80.0 |
| InternVideo2-1B (https://arxiv.org/abs/2403.15050) | 1B | 87.0 | 69.7 | 86.4 | 97.0 | 89.4 | 93.8 | 85.8 |
| **V-JEPA 2 ViT-g** | **1B** | **87.5** | **75.3** | **90.1** | **97.7** | 86.6 | 90.7 | 84.6 |
| **V-JEPA 2 ViT-g_384** | **1B** | **88.2** | **77.3** | **90.2** | **97.8** | **87.3** | **91.1** | **85.1** |

Key observations:
- 在 motion understanding (SSv2, Diving-48, Jester) 上 V-JEPA 2 大幅领先所有 image encoder (DINOv2, SigLIP2, PE)。SSv2 77.3 vs DINOv2 50.7，差距 26.6 points。
- 在 appearance understanding (K400, COIN, IN1K) 上 competitive，但 PE/SigLIP2 略好（因为它们用 language supervision）。
- V-JEPA 2 是唯一一个在 6 个 task average 上达到 88+ 的 model，证明 SSL video pretraining 能学到 general visual representation。

---

## 6. Prediction: EK100 Action Anticipation (Table 5)

Epic-Kitchens-100 (https://arxiv.org/abs/2202.05087): 100 小时 ego-centric cooking video，3568 action labels。给定 context clip (1 秒前结束)，predict future verb/noun/action。

Metric: mean-class recall@5 (因为非确定性，多 action 都可能 plausible)。

| Method | Params | Verb | Noun | Action |
|---|---|---|---|---|
| InAViT (https://arxiv.org/abs/2307.11904) | 160M | 51.9 | 52.0 | 25.8 |
| Video-LLaMA (https://arxiv.org/abs/2306.02858) | 7B | 52.9 | 52.0 | 26.0 |
| PlausiVL (https://arxiv.org/abs/2403.13820) | 8B | 55.6 | 54.2 | 27.6 |
| V-JEPA 2 ViT-L | 300M | 57.8 | 53.8 | 32.7 |
| V-JEPA 2 ViT-H | 600M | 59.2 | 54.6 | 36.5 |
| V-JEPA 2 ViT-g | 1B | 61.2 | 55.7 | 38.0 |
| **V-JEPA 2 ViT-g_384** | **1B** | **63.6** | **57.1** | **39.7** |

V-JEPA 2 ViT-g_384 比 PlausiVL 8B 提升 +12.1 action recall@5，相对提升 44%。注意 V-JEPA 2 ViT-L (300M) 就已经 beat 8B PlausiVL。

Anticipation probe 设计：
- 4 transformer blocks + cross-attention with 3 learnable query tokens
- 3 个 query 分别 predict verb, noun, action
- Focal loss (Lin et al. 2017, https://arxiv.org/abs/1708.02002) with $\alpha=0.25, \gamma=2.0$ 处理 long-tail
- Context: 32 frames @ 8 fps，duration 4 秒

### Prediction probe input ablation (Table 20)

| Encoder | Predictor | Verb | Noun | Action |
|---|---|---|---|---|
| ✓ | | 61.3 | 57.0 | 39.1 |
| | ✓ | 48.7 | 34.7 | 20.2 |
| ✓ | ✓ | 63.6 | 57.1 | 39.7 |

观察：encoder alone 已经 competitive，加 predictor 有 small consistent improvement。Predictor alone 表现差很多，说明 EK100 主要靠 semantic understanding 而非 forecasting。但 predictor 提供 temporal extension 的 information，对长 horizon anticipation 重要。

### Failure case 分析 (Figure 18 right)

VNA (全对) 是 most common。Most common failure 包含 action wrong。Verb 单独对 / Noun 单独对 / 都错的比例较低。说明 model 经常能识别 verb 和 noun，但联合 action label 错。

### Long-horizon degradation (Figure 18 left)

Anticipation time 1s → 2s → 4s → 10s，performance sharp 下降。这是 fundamental limit——action anticipation 是非确定性 task，long horizon 下 entropy 太高。

---

## 7. Video Question Answering (Section 7)

V-JEPA 2 是 **第一个** 没有 language supervision 的 video encoder 用于训练 MLLM 还能达到 SOTA 的，颠覆 conventional wisdom。

### 7.1 Controlled Setup (Table 6)

Qwen2-7B-Instruct (https://arxiv.org/abs/2407.10671) 作为 LLM backbone，frozen encoder，18M image+video-text pairs。

| Method | Avg | PerceptionTest | MVP | TempCompass | TemporalBench | TVBench | TOMATO | MVBench |
|---|---|---|---|---|---|---|---|---|
| DINOv2 ViT-g_518 | 45.7 | 67.1 | 22.4 | 62.3 | 26.8 | 47.6 | 32.0 | 61.8 |
| SigLIP2 ViT-g_384 | 48.1 | 72.4 | 26.2 | 66.8 | 25.7 | 48.7 | 33.2 | 64.0 |
| PE ViT-G/14_448 | 49.1 | 72.3 | 26.7 | 67.0 | 27.5 | 51.6 | 34.0 | 64.7 |
| **V-JEPA 2 ViT-g_512** | **52.3** | 72.0 | **31.1** | **69.2** | **33.3** | **55.9** | **37.0** | **67.7** |

V-JEPA 2 在所有需要 temporal understanding 的 benchmark (MVP, TempCompass, TemporalBench, TVBench, TOMATO) 上显著领先，在 PerceptionTest 上略低于 SigLIP2/PE。证明 video encoder 在 temporal reasoning 上 intrinsic 优势。

### 7.2 Scaling Effect (Table 7)

End-to-end (unfrozen encoder)，scale ViT-L → ViT-g，256 → 512 resolution：

| Method | Avg | PerceptionTest | TVBench | MVBench |
|---|---|---|---|---|
| V-JEPA 2 ViT-L_256 | 51.7 | 74.6 | 50.9 | 67.1 |
| V-JEPA 2 ViT-H_256 | 52.0 | 74.7 | 54.6 | 68.0 |
| V-JEPA 2 ViT-g_256 | 52.3 | 75.5 | 54.2 | 68.3 |
| V-JEPA 2 ViT-g_384 | 54.0 | 76.5 | 56.5 | 68.5 |
| V-JEPA 2 ViT-g_512 | 54.4 | 77.7 | 57.5 | 69.5 |

Resolution scaling 比 model scaling 更有效（256→512: +2.1 avg, ViT-L→ViT-g: +0.6 avg），与 Fan et al. 2025 (https://arxiv.org/abs/2504.01017) 在 image SSL 上的发现一致。

### 7.3 SOTA with Full Data (Table 8)

用 88.5M image+video-text pairs, Llama 3.1 8B backbone (https://arxiv.org/abs/2407.21783)，V-JEPA 2 ViT-g_384 + MLP projector (no pooling, 288 tokens/frame)：

| Method | PerceptionTest | MVP | TempCompass | TemporalBench | TOMATO | TVBench | MVBench |
|---|---|---|---|---|---|---|---|
| InternVL-2.5 (https://arxiv.org/abs/2412.05271) | 68.9 | 39.9 | 68.3 | 24.3 | 29.4 | 61.6 | 72.6 |
| Qwen2VL (https://arxiv.org/abs/2409.12191) | 66.9 | 29.2 | 67.9 | 20.4 | 31.5 | 46.0 | 67.0 |
| Qwen2.5VL (https://arxiv.org/abs/2502.13923) | 70.5 | 36.7 | 71.7 | 24.5 | 24.6 | 50.5 | 69.6 |
| PLM 8B (https://arxiv.org/abs/2504.13180) | 82.7 | 39.7 | 72.7 | 28.3 | 33.2 | 63.5 | 77.1 |
| **V-JEPA 2 ViT-g_384 + Llama 3.1 8B** | **84.0** | **44.5** | **76.9** | **36.7** | **40.3** | 60.6 | 73.5 |

SOTA on PerceptionTest, MVP, TempCompass, TemporalBench, TOMATO。在 TVBench/MVBench 上 V-JEPA 2 略低于 PLM 8B（PLM 用了更多 appearance supervision），但仍大幅超其他 baselines。

### 7.4 长视频理解 (Figure 19)

MLLM 训练和推理时增加 frame 数：V-JEPA 2 性能 linear 提升，DINOv2 (image encoder) flat 或下降。证明 video encoder 对 long-form video understanding 有 structural 优势——它天然 encode temporal relationships。

---

## 8. Decoder Visualization (Appendix B.3, Figure 15)

为了 visualize V-JEPA 2-AC 的 prediction，作者训了一个 feedforward ViT-L decoder 把 representation 映回 pixel space。Decoder 在 Droid 上用 L2 loss 训 150K steps。

Figure 15a 上：
- Top row: ground-truth robot trajectory frames
- Middle row: V-JEPA 2 encode + decode，背景略模糊（decoder 容量小）但 salient scene element 清晰
- Bottom row: V-JEPA 2-AC autoregressive rollout with ground-truth action，从第一帧起 predict 后续

观察：robot arm 正确 animate，背景和非交互物体（架子）保持稳定。Closed gripper 时 cup 跟随 arm 移动，说明 model 学到 object constancy + gravity + shape constancy。

Figure 15b：相同 action sequence，一次 closed gripper 一次 open gripper。Open gripper 时 cup 不动，说明 model 理解 "open gripper 不抓物体" 的因果。

但有 error accumulation——最后帧 cup 位置略低于 ground truth。这是 autoregressive prediction 的固有问题，rollout loss 只是减缓而非消除。

---

## 9. 核心直觉总结

让我提炼几个层次递进的 intuition：

### Intuition 1: JEPA = "对的世界模型的形状"

World model 的本质不是生成像素，而是预测可预测的结构性 aspects。Pixel space 包含太多不可预测的高频信息（草、水波、纹理），强迫 model 浪费 capacity。JEPA representation 通过 EMA teacher + masked prediction 隐式学到 "what is predictable" 的 projection。

### Intuition 2: Representation 是 Shared Substrate

同一个 V-JEPA 2 representation 服务于：
- Probe-based classification (understanding)
- Action anticipation (prediction)
- Action-conditioned planning (control)
- Video QA (after LLM alignment)

这不是 coincidence——good representation 应该是 task-agnostic 的 world model。LeCun 一直说 "perception is inference"，V-JEPA 2 给了实证。

### Intuition 3: Pretrain-then-Post-train 的数据效率

Internet video (1M 小时) 教 model 物理常识（gravity, object permanence, dynamics），robot video (62 小时) 只需教 "actions 如何影响 world state"。这个 1M:62 = 16000:1 的 ratio 体现 self-supervised pretraining 的威力。类比 LLM：text pretrain 学语法/常识，instruction tuning 教 follow instruction。

### Intuition 4: Representation Space 的 Planning 优势

Pixel-space generation model (Cosmos) 的 energy landscape 在 4 min/action 才能搜索，因为 pixel L1 distance 是高度非凸、多模态的。V-JEPA 2-AC 在 representation space 的 L1 distance 平滑、locally convex（Figure 9），CEM 16 sec/action 高效收敛。这是 JEPA 路线对 planning 的根本优势。

### Intuition 5: Block-Causal + 3D-RoPE 的设计哲学

Block-causal attention 让每个时间步内的 patches 互相 attend（spatial reasoning within frame），同时只能看过去帧（temporal causality）。3D-RoPE 让 positional encoding 在 T/H/W 三个轴独立旋转，给 ViT-g 这种大 model 提供训练稳定性。这两个设计组合是 video transformer 的强 inductive bias：spatial + temporal factored but joint reasoning。

### Intuition 6: Progressive Resolution 的 Optimization Geometry Insight

低分辨率训练等价于 low-pass filter，让 model 先找到 coarse loss basin。高分辨率 fine-tune 时仍在 good basin 附近，只需要补 high-frequency detail。这是 multi-resolution optimization 的 inductive bias：coarse-to-fine 比 fine-from-scratch 在大 model 上更稳定高效。8.4× speedup 让 ViT-g + 64 frames + 384² 变得可行。

### Intuition 7: 为什么 62 小时 robot data 足够

Droid 包含 Franka arm 的 teleoperation video，有 end-effector state metadata。V-JEPA 2-AC 不需要 task label、reward、success indicator——它只需要 (state, action, next_state) tuples 来学 dynamics。这把 problem 简化到 purely unsupervised dynamics modeling，62 小时足以学 table-top manipulation 的 local dynamics。同时因为 V-JEPA 2 representation 已经 encode 了 visual world 的常识，post-training 只学 "action 如何在 representation space 推进 state"。

---

## 10. 与 Related Work 的对比

### vs VideoMAE v2 (https://arxiv.org/abs/2303.11489)
- VideoMAE 用 pixel reconstruction (generative)，V-JEPA 用 representation prediction (discriminative)
- VideoMAE 在 SSv2 56.1, V-JEPA 2 77.3，差距 21 points

### vs InternVideo2 (https://arxiv.org/abs/2403.15050)
- InternVideo2 用 vision-text contrastive，需要 language supervision
- V-JEPA 2 不用 language，只在 VidQA 阶段 align LLM
- V-JEPA 2 在 motion tasks 更强，appearance tasks 略弱，average 持平

### vs DINOv2 (https://arxiv.org/abs/2304.07193)
- DINOv2 是 image SSL，无 temporal modeling
- V-JEPA 2 是 video SSL，原生处理 motion
- 在 SSv2 (motion) 上 V-JEPA 2 77.3 vs DINOv2 50.7，巨大差距
- 在 IN1K (appearance) 上 V-JEPA 2 85.1 vs DINOv2 86.1，competitive

### vs Octo VLA (https://arxiv.org/abs/2405.12213)
- Octo 是 behavior cloning，需要成功 trajectory + task label
- V-JEPA 2-AC 是 world model + planning，无 task label，可用失败 trajectory
- Pick-Place Cup 上 V-JEPA 2-AC 80% vs Octo 15%

### vs Cosmos (https://arxiv.org/abs/2501.03575)
- Cosmos 是 latent diffusion 7B in pixel latent space
- V-JEPA 2-AC 是 300M predictor in JEPA representation space
- Planning speed 16s vs 4min per action
- Pick-Place Cup 上 V-JEPA 2-AC 80% vs Cosmos 0%

### vs Genie (https://arxiv.org/abs/2402.15391)
- Genie 是 generative interactive environment，无 action grounding for real robot
- V-JEPA 2-AC 直接部署到真实 Franka arm

### vs DreamerV3 / TD-MPC2 (https://arxiv.org/abs/2301.04104, https://arxiv.org/abs/2310.16828)
- 这些是 RL world models，需要 reward signal
- V-JEPA 2-AC 是 unsupervised world model，只需 (s, a, s') tuples
- 直接用 MPC planning 而非 policy learning

### vs DINO-WM (https://arxiv.org/abs/2411.04983)
- DINO-WM 也用 pre-trained visual features 做 world model + planning
- V-JEPA 2-AC scale 更大，并且直接 video pretrain 而非 image pretrain
- V-JEPA 2-AC 在真实 robot 上 zero-shot 部署

---

## 11. Future Work 方向

Paper 自己列出的几个 limit:

1. **Long-horizon planning**: 当前只能 ~16 秒预测。需要 hierarchical model 跨 multiple spatial/temporal scales 做不同 abstraction level 的预测。LeCun 倡导的 hierarchical JEPA (H-JEPA) 是 natural extension。

2. **Language goal specification**: 当前用 image goal，未来需要 language-conditioned planning。可以借 Section 7 的 LLM alignment，让 LLM 把 language goal translate 到 V-JEPA representation space。

3. **Model scaling**: 当前只到 1B。DINOv2 试过 20B (Zhai et al. 2022, https://arxiv.org/abs/2106.04560)，video model 应该能 scale 更大，需要 scalable pretrain recipe。

4. **Camera invariance**: 当前对 camera 位置 sensitive，需要 unsupervised calibration 或 3D-aware representation。

5. **Error accumulation in rollout**: Rollout loss 只是减缓，根本解决需要 implicit world model (像 Dreamer 的 latent dynamics) 或 hierarchical temporal abstraction。

6. **Beyond table-top manipulation**: 当前只在 Franka arm table-top 场景。延伸到 locomotion (quadruped, humanoid) 需要 richer dynamics modeling。Gr00t N1 (https://arxiv.org/abs/2503.14734) 和 π0 (https://arxiv.org/abs/2410.24164) 已经在做 humanoid，JEPA 路线能否 scale 到 humanoid 是巨大 open question。

---

## 12. 我个人的 Critical Thoughts

**Strengths**:
- 第一次把 JEPA 真正 scale 到 meaningful size 并 demonstrate end-to-end from pretrain → robot deployment
- Progressive resolution training 是 elegant engineering
- Representation-space planning 的效率优势 quantitatively demonstrated (16s vs 4min)
- 62 小时 robot data zero-shot transfer 到 new labs 是 strong evidence for world model generalization

**Weaknesses / Open questions**:
- Grasp Box 只有 25-30% success rate，离实用还远
- Planning horizon T=1，本质上是 greedy，没有真正的 long-horizon planning
- Pick-Place 需要 3 个 sub-goal images 手动提供，不是 autonomous
- Encoder 1B + predictor 300M，但 deployment 需要 RTX 4090 跑 16s/action，real-time 还差
- Comparison with VLA baselines (Octo, RT-2, π0) 不够公平——V-JEPA 2-AC 用 62h Droid，π0 用更多更高质量 teleop data
- Action space 限制在 L1-ball 0.075，限制了 model 的"决策范围"

**What's next (my guess)**:
- Hierarchical JEPA：低层 short-horizon predictor + 高层 long-horizon planner，都 in representation space
- Language-conditioned planning: 把 Section 7 的 LLM-aligned V-JEPA 2 与 V-JEPA 2-AC 结合
- Scale to 10B+ params，观察 scaling laws
- 推广到 locomotion 和 dexterous manipulation
- Action tokenization: 当前 action 是 continuous 7D vector，未来可能 discrete action tokens 配合 autoregressive generation (像 VLA but in JEPA framework)

---

## References (核心 paper 链接)

- **V-JEPA 2 paper**: https://ai.meta.com/vjepa
- **V-JEPA 2 code**: https://github.com/facebookresearch/vjepa2
- **V-JEPA 2 blog**: https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks
- **V-JEPA original (Bardes et al. 2024)**: https://arxiv.org/abs/2404.08471
- **LeCun JEPA position paper**: https://openreview.net/pdf?id=BZ5a1r-kVsf
- **I-JEPA (Assran et al. 2023)**: https://arxiv.org/abs/2301.08243
- **DINOv2**: https://arxiv.org/abs/2304.07193
- **Droid dataset**: https://arxiv.org/abs/2403.12945
- **Cosmos**: https://arxiv.org/abs/2501.03575
- **Octo**: https://arxiv.org/abs/2405.12213
- **InternVideo2**: https://arxiv.org/abs/2403.15050
- **SigLIP2**: https://arxiv.org/abs/2502.14786
- **Perception Encoder**: https://arxiv.org/abs/2504.13181
- **PerceptionLM**: https://arxiv.org/abs/2504.13180
- **Epic-Kitchens-100**: https://arxiv.org/abs/2202.05087
- **CEM (Rubinstein 1997)**: https://www.sciencedirect.com/science/article/pii/S0377221797003827
- **RoPE (Su et al. 2024)**: https://arxiv.org/abs/2104.09864
- **ViT (Dosovitskiy et al. 2020)**: https://arxiv.org/abs/2010.11929
- **Scaling ViT (Zhai et al. 2022)**: https://arxiv.org/abs/2106.04560
- **Focal loss (Lin et al. 2017)**: https://arxiv.org/abs/1708.02002
- **LLaVA**: https://arxiv.org/abs/2304.08485
- **Qwen2-VL**: https://arxiv.org/abs/2409.12191
- **Qwen2.5-VL**: https://arxiv.org/abs/2502.13923
- **InternVL-2.5**: https://arxiv.org/abs/2412.05271
- **Llama 3.1**: https://arxiv.org/abs/2407.21783
- **DreamerV3**: https://arxiv.org/abs/2301.04104
- **TD-MPC2**: https://arxiv.org/abs/2310.16828
- **DINO-WM**: https://arxiv.org/abs/2411.04983
- **Genie**: https://arxiv.org/abs/2402.15391
- **Gr00t N1**: https://arxiv.org/abs/2503.14734
- **π0**: https://arxiv.org/abs/2410.24164
- **RT-2**: https://arxiv.org/abs/2307.15818
- **VideoMAE v2**: https://arxiv.org/abs/2303.11489
- **PerceptionTest**: https://arxiv.org/abs/2305.04657
- **TOMATO**: https://arxiv.org/abs/2410.23266
- **TemporalBench**: https://arxiv.org/abs/2410.10818
- **TempCompass**: https://arxiv.org/abs/2403.00476

---

希望这些细节能 build 你的 intuition。如果你想 deep dive 哪个 specific 部分（比如 progressive resolution 的实现细节、CEM 的 sampling 策略、block-causal attention 的具体 mask pattern、或者 3D-RoPE 的维度分配），我可以再展开。
