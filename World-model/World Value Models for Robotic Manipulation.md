---
source_pdf: World Value Models for Robotic Manipulation.pdf
paper_sha256: e5552ff1ea246bfe74965e5bacd2bbaf418fcd66b8c701d0aeb60afd1ca89f04
processed_at: '2026-08-13T05:11:23-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# WVM 用人话讲

Andrej，这篇paper说白了就一句话：**value model这个活在robotics里一直干不好，因为大家用的backbone是VLM，而VLM天生看不懂时间序列。改用video world model当backbone，value estimation立马就SOTA了。**

---

## 问题在哪

Robot learning现在数据量起来了，几十万条轨迹，质量参差不齐，有好有坏。你需要一个"value model"来给每帧打个分，说"这个动作让task往前走了多少"。这分数可以用来filter数据、给policy做加权、判断哪些轨迹是expert哪些是垃圾。

这个思路跟RL里的value function一模一样：reward稀疏的时候，value就是离终点还有多远。离终点近就value高，离终点远就value低。

**问题出在backbone选择上。** 现有value model——GVL、VLAC、Robometer、TopReward、RoboReward、Robo-Dopamine——全是基于VLM的。VLM的pretrain任务是"一张图配一段文字"，它不需要理解时间，一张静止的图片配一句"cat sitting on a chair"就够训练了。

结果就是VLM看到的"视频"其实是shuffled frames的集合，它对temporal dynamics的理解几乎为零。你让它给一个trajectory逐帧打value，它只能靠单帧appearance猜，完全捕捉不到hesitation（卡住了）、retry（重试退回去再抓）这些local temporal pattern。

**而video world model，比如Wan2.2，它的pretrain任务就是"预测未来几帧video"。** 它要预测下一帧长什么样，就必须理解物理世界怎么随时间变化——物体怎么动、手怎么抓、东西怎么掉。这跟value function关心的"task progress随时间怎么变"天然对齐。

所以核心insight不是"用更大的model"，是"用对任务对齐的pretrain prior"。VLM对spatial alignment友好，world model对temporal dynamics友好，value estimation恰好需要后者。

---

## WVM怎么搭

整体架构就是个双流transformer，通过asymmetric attention耦合：

**主路：video stream。** 直接拿Wan2.2-TI2V-5B的video DiT，5B参数，30层，hidden dim 3072。输入是 $(2h+1)$ 帧的video clip——1帧prefix作为context，h帧当前observation，h帧未来帧。Video VAE把这clip压缩成latent，未来帧的latent加噪后做flow matching denoising，相当于video stream在持续训练"预测未来"这个任务。

**旁路：value stream。** 轻量DiT，0.7B参数，hidden dim 512。它要预测的是当前h帧observation的value chunk $\hat{v}_{t-h+1:t} \in [0,1]^h$，就是这段窗口里每一帧的normalized task progress。

**耦合：MoT attention。** 每一层，value token会被project到3072维参与joint attention，能看video latent里的spatiotemporal feature，借力world model对dynamics的理解。但video token永远看不到value token——value stream的梯度不会污染video generation任务。

这个asymmetric mask是个关键设计。你可以想成"老师讲课、学生听课"——video stream是老师，一直在训练"未来长什么样"，它的representation里encoding了丰富的temporal prior。Value stream是学生，从老师的latent里吸收信息来预测value，但学生的问题不会反过来干扰老师。如果用symmetric attention或者naive concat，value loss的梯度会流回video stream，把generation能力搞坏。

最终loss是两部分：
$$\mathcal{L} = \mathcal{L}_{\text{value}} + \mathcal{L}_{\text{video}}$$

value loss预测value chunk，video loss继续预测未来video latent，两者权重相等 $\lambda=1.0$。

---

## 为什么预测chunk而不是scalar

这是个很关键的细节。传统value function是标量 $V(o_t)$，每帧一个数。WVM预测的是长度 $h=4$ 的chunk，一次预测连续4帧的value。

为什么？因为task progress是local temporal pattern。Hesitation就是value在几帧内基本不变（plateau），retry就是value先降后升（V形）。Scalar prediction丢掉了这个local shape——你只看一个数，不知道它是稳定不变还是在波动。

Chunk prediction逼model学local value profile：这4帧是稳步上升（正常）还是平的（hesitation）还是V形（retry）。这跟video prediction里的local frame modeling思路一致——预测连续几帧而不是单帧，能capture dynamics。

具体用flow matching做，不是分类不是回归。从noise到clean value chunk之间画条直线，让网络学velocity field。比categorical head（比如HL-Gaussian把[0,1]离散成51个bin做分类）好得多——ablation显示HL-Gaussian的Hesitation-RMSE只差一点（0.06 vs 0.05），但Retry-VOC从0.78掉到0.59，Expert-VOC从0.95掉到0.87。

原因：categorical head的fixed bin support能保留conditional mean，但丢了fine-grained density variation。VOC这种ordinal metric关心相邻chunk的ranking，需要sub-bin的precision，categorical天然不行。Flow matching是continuous distribution，不bound support也不bound resolution，能保留local value differential。

---

## 两个augmentation trick

光有架构不够，训练数据有个问题：expert trajectory全都是monotonic progress，从0到1平滑上升。Model在这种数据上训练，只会学"值在涨"，不知道hesitation和retry长什么样。

**Trick 1：Video rewinding。** 借鉴ReWiND那篇。对每个training window，随机sample三种pattern：rising（正常）、plateau（重复某些帧模拟hesitation）、descending（reverse帧模拟retry）。对应的value label重新标。这样model能见到suboptimal progress profile的样本。

主run里rewind ratio 0.5（一半样本做rewind），其中10%是plateau模式，其余是descending。

**Trick 2：Prefix randomization。** Inference时用overlapping chunk——预测[t,t+3]的value，下次预测[t+1,t+4]，重叠部分averaging。这样chunk间更连贯。但有个shortcut风险：model可能只看prefix value就直接外推，不看video evidence。

解决：训练时以概率 $p=0.5$ 把prefix value替换成[0,1]上的uniform random scalar，loss只算剩下的value token。类似classifier-free guidance里的conditioning dropout。

Ablation特别有意思：$p=0$（不做randomization），Expert-VOC高达0.98，但Hesitation-RMSE恶化到0.09，Retry-VOC掉到0.67——model在偷prefix的shortcut，对expert trajectory看起来完美，suboptimal完全废。$p=1$（全mask），suboptimal恢复一些，但Expert-VOC掉到0.91。$p=0.5$ 两边都好。

这个发现给整个社区一个warning：如果你只看Expert-VOC训练value model，很可能训出来的是shortcut learner，suboptimal detection完全失效。这也是paper为什么专门做Suboptimal-Value-Bench。

---

## Suboptimal-Value-Bench

现有evaluation方式都有blind spot：
- 人工看value曲线？不可scale
- 看downstream policy success？value质量跟policy选择混在一起
- VOC（Value-Order Correlation）？只对monotonic expert trajectory well-defined，suboptimal段（progress不变或下降）直接ill-defined

所以他们做了Suboptimal-Value-Bench：800条human-annotated trajectory，三种embodiment（AgileX real、ARX real、RoboSuite sim），15个task，每帧都有dense value curve label，分hesitation和retry两类。

标注pipeline很务实：
1. 先用proprietary VLM API做粗标注——把trajectory降采样送进VLM，让它输出non-progress segment的JSON
2. 人工annotator在custom界面里frame-level精修boundary，能replay、drag、split、merge

VLM proposal为空的trajectory也照常present给annotator，防漏标。这把人工成本压到可接受范围。

Ground-truth value curve的构造也很讲究：
- **Hesitation**：segment内progress是plateau（不变），前后段等速前进。三段斜率 $1/(T-1-x)$、$0$、$1/(T-1-x)$，$x$是segment长度
- **Retry**：segment内progress匀速回退，前后段匀速前进。三段斜率对称的V形，如果retry太长progress会负则clamp到0

评测metric两个：
- **Hesitation-RMSE**：segment内prediction drift多大，完美恒定=0
- **Retry-VOC**：只在monotonically decreasing的retry window算VOC，完美单调降=+1，反向=−1

---

## 结果

**Hesitation-RMSE**：WVM 0.05，最强baseline（GVL、Robometer）0.14。3倍改进。

**Retry-VOC**：WVM 0.78，最强baseline GVL 0.62。VLAC、Robometer、TopReward甚至负相关——这些VLM-based model完全捕捉不到"progress在下降"这个pattern，只会输出monotonic prediction。

**Expert-VOC**：WVM 0.95平均，最强baseline RoboReward 0.88。只有EgoDex一个dataset略输（0.92 vs 0.95），但RoboReward在Suboptimal-Value-Bench上远不如WVM，再次印证Expert-VOC单独看是insufficient metric。

**Downstream policy**：用$\pi_{0.5}$-base，只用10-50条suboptimal trajectory做finetune。三种WVM-guided variant（AWR加权、binary filter丢负advantage chunk、top-70% filter）都consistently beat vanilla BC。这证明WVM能从noisy data里filter出真progress，让policy从脏数据里学到有用signal。

---

## Ablation几个核心发现

1. **Video co-training必不可少。** 去掉 $\mathcal{L}_{\text{video}}$，suboptimal性能掉。Freeze video权重，掉得更狠（Hesitation-RMSE 0.12、Retry-VOC 0.45）。从零训练video stream，Retry-VOC 0.62。结论：pretrained world model prior是起点，必须继续co-training让latent对value task更友好，同时video loss防止prior漂移。

2. **Prefix randomization的trade-off**就是上面说的那个，$p=0.5$ 最优。

3. **Flow matching head > Categorical head**。Categorical能保mean但保不了ranking precision。

---

## 整个story的intuition

你看过那么多architecture，这篇paper的beauty在于它的整个逻辑链非常干净：

1. Value estimation本质是关于future的（expected discounted return）
2. Video world model本质也是关于future的（predict next frames）
3. 两者任务对齐，所以world model的spatiotemporal prior是value estimation的天然backbone
4. 用asymmetric MoT让value stream"借力"video stream但不污染它
5. 用chunk + flow matching capture local progress profile（plateau、V-shape）
6. 用prefix randomization防shortcut，用video rewinding补suboptimal supervision
7. 做一个suboptimal benchmark专门暴露现有value model的blind spot

没有任何一步是"black magic"，每一步都有清晰的why。最终5.7B参数，32卡A100训40小时，3万步，1万步就converge了。推理1步Euler solver够用，因为corpus相对capacity中等，velocity field已经够平滑。

这篇paper很Karpathy-style：**不堆参数、不堆数据，靠task对齐 + minimal but principled design打SOTA。** 整个pipeline可以复现，benchmark开源，方法清晰。

---

如果你想再深入聊哪一块，我可以展开：
- MoT的attention实现细节（value token怎么project、attention mask怎么设）
- Flow matching vs DDPM vs Rectified Flow的数学区别
- Suboptimal-Value-Bench的标注interface怎么scale到10K trajectory
- Distributional value head的design space（C51 / QR-DQN / IQN / Flow matching / HL-Gaussian对比）
- Downstream policy learning里AWR的temperature和clip ceiling怎么调

或者你想吐槽哪一块设计，我可以陪你一起吐槽。

---

# World Value Models for Robotic Manipulation 深度解析

Andrej，这篇paper对你来说应该会引发不少共鸣——它本质上是把"world model作为feature extractor"和"distributional value learning"两条线缝合起来，关键insight是：**VLM backbone是spatial-temporal sparse的，而video world model天然就具备temporal modeling和future planning能力**，所以直接把world model的spatiotemporal prior repurpose成value estimator的backbone。

参考链接：
- Project Page: https://zh1hao.wang/wvm
- Wan2.2 video model: https://arxiv.org/abs/2503.20314
- Mixture-of-Transformers: https://openreview.net/forum?id=Nu6N69i8SB
- Flow matching guide: https://arxiv.org/abs/2412.06264
- GVL (VLM in-context value learners): https://arxiv.org/abs/2402.17177 附近
- Robometer: https://arxiv.org/abs/2603.02115
- π0.5: https://arxiv.org/abs/2504.16054

---

## 1. Motivation: 为什么VLM-based value model不行

现有robotic value model（GVL、VLAC、Robometer、TopReward、RoboReward、Robo-Dopamine等）的三个bottleneck：

1. **Scalar value supervision information sparse**：在high-dim visual observation上做scalar regression，训练信号太稀疏，heterogeneous video corpus上预测brittle。
2. **Task-specific customization**：很多value model（如π*, GR-RL等）紧绑定单任务，不能当generalist progress estimator。
3. **VLM backbone的temporal建模缺陷**：pretrained VLM的representation对static或temporally sparse image优化，capture不到dense temporal dynamics。

而world model——尤其是video generation model——天然就在做temporal dynamics + forward prediction。**核心insight**：world model的spatiotemporal prior可以被"借尸还魂"用作value function的foundation。

这其实和你在Reddit/Twitter上经常讲的观点一致：next-token prediction on video本身就在逼model学习physical dynamics，而value function本质上是在估expected discounted future return——两者都对"future"敏感。

---

## 2. Problem Formulation

### 2.1 Value as chunk-wise distribution

不是预测单个scalar $v_t$，而是预测一个长度 $h$ 的value chunk：

$$p_{\psi}\big(\hat{v}_{t-h+1:t} \mid o_{t-h+1:t}, l\big)$$

其中：
- $\hat{v}_{t-h+1:t} \in [0,1]^h$ 是预测的value chunk
- $o_{t-h+1:t}$ 是 $h$-frame的observation序列
- $l$ 是language instruction
- $v_t = t/T$ 是normalized task progress，$T$ 是trajectory总长度
- $\psi$ 是value model参数

**为什么chunk而非scalar**：chunk能capture local progress profile，比如plateau（hesitation）和regression（retry），这对suboptimal data的detection至关重要。scalar prediction会丢掉这些local shape信息。

### 2.2 经典RL value function的解读

$$V(o_t) = \mathbb{E}\left[\sum_{t'=t}^{T} \gamma^{t'-t} r_{t'} \mid o_t\right]$$

变量含义：
- $V(o_t)$：状态 $o_t$ 的value function
- $r_{t'}$：step-level reward
- $\gamma \in (0,1]$：discount factor，控制对未来reward的折扣
- $t'$：从 $t$ 到 $T$ 的时间索引
- $\mathbb{E}[\cdot]$：对trajectory分布的期望

**关键观察**：在canonical sparse-reward setting下（$r_{t'}=-1$ for non-terminal, $0$ at completion），$V(o_t)$ 退化成negative expected distance-to-goal，所以value estimation $\Leftrightarrow$ task-progress prediction。

这给了world model一个特别自然的位置：value function by construction聚焦future outcomes，而world model就是干future prediction的。

### 2.3 用world model当feature extractor

$$p_{\psi}\big(\hat{v}_{t-h+1:t} \mid o_{t-h+1:t}, l\big) = p_{\psi}\big(\hat{v}_{t-h+1:t} \mid M_{\omega}(o_{t-h+1:t}, l)\big)$$

- $M_{\omega}$：pretrained video world model，参数 $\omega$
- $M_{\omega}(o, l)$：world model对video + language的latent encoding
- $\psi$：在world model latent上加的value预测头

这就是把world model当"value feature backbone"用，而value model只是个轻量头。

---

## 3. WVM架构详解

### 3.1 整体结构

WVM = Video Stream + Value Stream，通过MoT耦合。这是paper最architectural的部分。

#### Video stream
- 基于 **Wan2.2-TI2V-5B** checkpoint
- Wan2.2-VAE：temporal压缩4×、spatial压缩16×16，输出48-channel spatiotemporal latent
- Latent再patchify（patch size (1,2,2)）进transformer
- Video DiT：30层，hidden dim 3072，24个attention head（head dim 128），FFN width 14336，~5.0B params

#### Value stream
- 轻量DiT，mirror video DiT架构但参数少很多
- Hidden dim 512，8个self/cross-attention head（head dim 64），FFN width 14336
- ~0.7B trainable params
- 总共5.7B params

#### MoT coupling
每一层做joint attention：
- Video tokens：保留原Wan2.2的Q/K/V projection（hidden 3072）
- Value tokens：从hidden 512 linearly project到共享attention width 3072（24 heads, head dim 128）参与joint attention，输出时project回512

**Asymmetric attention mask**是核心设计：
- Value tokens **可以** attend to video latents
- Video tokens **不能** attend to value tokens

这是Fast-WAM那篇的思路——video generation stream不能被value stream污染，否则会破坏world model的generation能力。但value stream可以"读"video latent，借力spatiotemporal prior。

这个asymmetric mask的设计intuition：world model是"老师"，value model是"学生"，学生看老师，老师不被学生干扰。

### 3.2 Video clip构造

对anchor在 $[t-h+1, t]$ 的value chunk，video VAE吃一个 $(2h+1)$-frame的clip：

$$\underbrace{o_{t-h}}_{\text{1-frame prefix}} \parallel \underbrace{o_{t-h+1:t}}_{\text{h current frames}} \parallel \underbrace{o_{t+1:t+h}}_{\text{h future frames}}$$

- 1 frame prefix：前一个帧，给context
- h current frames：当前observation窗口，作为conditioning context保留
- h future frames：target，corrupt后做video generation denoising

VAE编码出三段temporal latents：
- Prefix latent：**discard**（丢掉）
- Current latent：**keep**作为context
- Future latent：corrupt做denoising

注意value chunk是对应current frames的，而video prediction是对应future frames的。两个stream预测的"时间窗口"错开了——这其实是paper一个巧妙的设计：value model用video model的"过去+未来"信息来预测"现在"的progress。

---

## 4. 训练目标

### 4.1 Flow matching loss

对video和value token都用flow matching。设 $y$ 是target（future video latents $\xi_{t+1:t+h}$ 或 value chunk $v_{t-h+1:t}$），$f_{\psi}$ 是velocity predictor：

$$\mathcal{L}_{\text{FM}}(y) = \mathbb{E}_{y, \epsilon, \tau}\Big[\big\|f_{\psi}(y_{\tau}, \tau, o_{t-h+1:t}, l) - (y-\epsilon)\big\|_2^2\Big]$$

变量含义：
- $y$：target data point（clean video latent或clean value chunk）
- $\epsilon \sim \mathcal{N}(0, I)$：标准Gaussian noise
- $\tau \in (0,1)$：flow time step，从noise到data的interpolation参数
- $y_{\tau} = \tau y + (1-\tau)\epsilon$：interpolated sample，沿直线从noise $\epsilon$ 流向data $y$
- $f_{\psi}(\cdot)$：neural network预测的velocity field
- $(y - \epsilon)$：ground-truth velocity（从 $\epsilon$ 到 $y$ 的方向向量）
- $\|\cdot\|_2^2$：L2 norm squared

**Intuition**：flow matching在data和noise之间画直线，让网络学velocity field。比diffusion的DDPM等更简单，rectified flow就是这种直线化思路。

### 4.2 双任务loss

$$\mathcal{L}_{\text{value}} = \mathcal{L}_{\text{FM}}(v_{t-h+1:t}), \quad \mathcal{L}_{\text{video}} = \mathcal{L}_{\text{FM}}(\xi_{t+1:t+h})$$

$$\mathcal{L} = \mathcal{L}_{\text{value}} + \lambda \mathcal{L}_{\text{video}}$$

- $\lambda$：video co-training权重，main run用 $\lambda=1.0$
- $\mathcal{L}_{\text{value}}$：value chunk的flow matching loss
- $\mathcal{L}_{\text{video}}$：future video latents的flow matching loss

**为什么co-training video stream至关重要**：从ablation看，去掉 $\mathcal{L}_{\text{video}}$，Hesitation-RMSE从0.05→0.08，Retry-VOC从0.78→0.68。完全freeze video权重，更糟（0.12 / 0.45）。这说明world model的temporal dynamics必须通过video loss持续优化，才能给value stream提供好的feature。

**关键设计哲学**：value stream只是个旁路，主任务仍然是video generation。video stream是"主干道"，value是"寄生"在主干道上的轻量分支。这与Fast-WAM的latent space prediction思路一脉相承。

---

## 5. 数据增强的两个trick

### 5.1 Prefix randomization

**问题**：inference用chunk overlapping会引入一个shortcut——value stream可以从prefix value外推，不依赖visual evidence。

**解决**：类比CFG的conditioning dropout。以概率 $p$ 把prefix value替换为 $[0,1]$ 上的uniform random scalar，否则保留。loss只施加在remaining value tokens上。

- $p=0$（无randomization）：Hesitation-RMSE 0.09，Retry-VOC 0.67，但Expert-VOC虚高到0.98 → 说明model在偷prefix的shortcut，对expert trajectory看起来完美但suboptimal检测失效
- $p=1$（全mask）：Retry-VOC 0.75，但Expert-VOC掉到0.91 → cross-chunk consistency被破坏
- $p=0.5$（default）：Hesitation-RMSE 0.05，Retry-VOC 0.78，Expert-VOC 0.95 → 完美balance

这个ablation特别有意思——它直接证明Expert-VOC作为唯一metric是insufficient的，因为 $p=0$ 时Expert-VOC最高但suboptimal性能最差。

### 5.2 Video rewinding（借鉴ReWiND）

Expert trajectory只提供monotonic progress label，plateau/regression的supervision不够。Rewind augmentation：

对每个窗口 $O_{t-h+1:t}$，sample三种temporal pattern之一：
- **Rising**：保留frames（normal forward）
- **Plateau**：repeat某些frames（模拟hesitation）
- **Descending**：reverse frames（模拟retry/回退）

对应的 $v_{t-h+1:t}$ 重新label。这exposes value stream to local progress profile：smooth advancement / hesitation / retry。

主run参数：rewind ratio 0.5，rewind plateau ratio 0.1（一半样本rewind，其中10%是plateau模式）。

---

## 6. Suboptimal-Value-Bench

这是paper的另一个重要contribution——现有evaluation的blind spot。

### 6.1 现有metric的问题

- **Qualitative curve inspection**：不可scale
- **Downstream policy success**：entangle value fidelity + policy choice，计算昂贵
- **VOC（Value-Order Correlation）**：只对monotonic expert trajectory well-defined，suboptimal segment失效

### 6.2 Benchmark构成

800条human-annotated trajectory：
- 3种embodiment（AgileX、ARX、RoboSuite）
- 15个task
- 每个task分hesitation和retry两组
- 每frame都有dense value curve annotation

Per-task分布：
- AgileX (real)：4个task，200条
- ARX (real)：5个task，300条
- RoboSuite (sim)：6个task，300条

### 6.3 标注pipeline

800条trajectory的frame-level人工标注成本太高，用两阶段：

**Stage 1**：proprietary LVLM API预分割。把trajectory降采样后送进VLM，prompt要求输出non-progress segments的JSON：
```json
{
  "non_progress_segments": [
    {"start_frame": int, "end_frame": int, "description": str}
  ],
  "task_completed": bool,
  "summary": str
}
```

**Stage 2**：human annotator在custom界面（Figure B.1）里frame-level精修boundary，可replay、drag、split、merge。

这个pipeline很务实——用VLM做"草稿"，human做"精修"。VLM proposal为空也仍然present给annotator以防漏标。

### 6.4 Ground-truth value curve构造

对每条trajectory，annotation给出：
- suboptimal type（hesitation或retry）
- segment端点 $m, n$（$0 < m < n < T-1$）
- 总长度 $T$

构造4-point piecewise-linear curve，控制点为 $(0,0), (m, v_m), (n, v_n), (T-1, 1)$，相邻控制点之间linear interpolation。

#### Hesitation
segment内progress不变（plateau），剩余 $T-1-x$ 个effective frame均匀覆盖unit progress（$x = n - m$ 是segment长度）：

$$v_m = v_n = \frac{m}{T-1-x}$$

三段斜率：$1/(T-1-x)$、$0$、$1/(T-1-x)$，前后等速，中间plateau。

#### Retry
假设retry是匀速retraction（与forward rate相同 $r$），$x$ 个frame回退 + $x$ 个frame前进补回，net forward只剩 $T-2x$ frame覆盖unit interval，所以 $r = 1/(T-2x)$：

$$v_m = \frac{m}{T-2x}, \quad v_n = \max\left(0, \frac{m-x}{T-2x}\right)$$

当 $n \leq 2m$ 时三段对称（V型）；当retry太长（$n > 2m$），$v_n$ clamp到0，避免progress变负。

---

## 7. 评测metric

### 7.1 Hesitation-RMSE

$$\text{Hesitation-RMSE} = \sqrt{\frac{1}{|\mathcal{H}|} \sum_{t \in \mathcal{H}} (\hat{v}_t - v_t)^2}$$

- $\mathcal{H}$：hesitation segment内所有frame的集合
- $\hat{v}_t$：模型预测值
- $v_t$：ground-truth（恒定）
- $|\mathcal{H}|$：frame数

完美恒定预测=0；预测漂移越大RMSE越大。

### 7.2 Retry-VOC

只对monotonically decreasing的retry window算VOC。完美单调降=+1，完全反向=−1。

---

## 8. 实验结果

### 8.1 Hesitation-RMSE (Table 1)

| Benchmark | GVL | VLAC | Robometer | TopReward | RoboReward | Robo-Dopamine | WVM |
|---|---|---|---|---|---|---|---|
| Suboptimal-AgileX | 0.11 | 0.47 | 0.13 | 0.36 | 0.12 | 0.41 | **0.07** |
| Suboptimal-ARX | 0.14 | 0.50 | 0.12 | 0.24 | 0.17 | 0.52 | **0.05** |
| Suboptimal-RoboSuite | 0.16 | 0.54 | 0.16 | 0.33 | 0.31 | 0.51 | **0.04** |
| Average | 0.14 | 0.51 | 0.14 | 0.31 | 0.21 | 0.49 | **0.05** |

WVM把average从最强baseline的0.14（GVL、Robometer）降到0.05，**~3x改进**。

### 8.2 Retry-VOC (Table 2)

| Benchmark | GVL | VLAC | Robometer | TopReward | WVM |
|---|---|---|---|---|---|
| Suboptimal-AgileX | 0.73 | -0.37 | 0.32 | 0.15 | **0.79** |
| Suboptimal-ARX | 0.76 | / | -0.27 | -0.19 | **0.79** |
| Suboptimal-RoboSuite | 0.43 | / | -0.37 | 0.00 | **0.75** |
| Average | 0.62 | -0.37 | -0.16 | 0.00 | **0.78** |

VLAC、Robometer、TopReward在retry上甚至负相关，说明VLM-based model对"progress regression"完全无能为力。WVM的0.78 vs GVL的0.62，**16%相对改进**。

"/"表示VOC ill-defined——出现RoboReward、Robo-Dopamine也 ill-defined的情况。

### 8.3 Expert-VOC (Table 3)

| Benchmark | GVL | VLAC | Robometer | TopReward | RoboReward | Robo-Dopamine | WVM |
|---|---|---|---|---|---|---|---|
| OXE | 0.67 | 0.48 | 0.63 | 0.19 | 0.92 | 0.72 | **0.94** |
| RoboCOIN | 0.70 | 0.60 | 0.77 | 0.47 | 0.85 | 0.75 | **0.95** |
| EgoDex | 0.82 | 0.62 | 0.86 | 0.37 | **0.95** | 0.88 | 0.92 |
| Self-collected | 0.93 | 0.50 | 0.93 | 0.58 | 0.84 | 0.76 | **0.99** |
| Average | 0.78 | 0.59 | 0.81 | 0.42 | 0.88 | 0.82 | **0.95** |

WVM 5个dataset中4个第一，只有EgoDex略输给RoboReward（0.92 vs 0.95）。Self-collected上达到0.99接近完美。

**EgoDex的"失败"反而是paper诚实之处**——paper后面讨论Expert-VOC本身是个insufficient metric。RoboReward在EgoDex上赢，但它在suboptimal bench上表现远不如WVM，说明Expert-VOC高分不等于value quality高。

### 8.4 Downstream policy learning (Figure 6)

用 $\pi_{0.5}$-base作为基础policy，只用suboptimal data finetune：
- RoboSuite simulation：每task 10条trajectory
- AgileX real：每task 50条trajectory

三种WVM-guided variant：
1. **AWR**（Advantage Weighted Regression）：$w_i = \min(\exp(\tau \cdot \Delta_i), \delta)$，其中 $\Delta_i = V(t_i^{\text{tail}}) - V(t_i^{\text{head}})$
2. **Filtered BC (binary)**：$w_i = \mathbf{1}[\Delta_i \geq 0]$，丢弃value不增长的chunk
3. **Filtered BC (top-70%)**：保留 $\Delta_i$ 排名top 70%的chunk

参数：
- $\delta = 2.0$（AWR clip ceiling）
- RoboSuite: $\tau=10$, $H=10$, $\kappa=0.02$（top-70% threshold）
- AgileX: $\tau=2$, $H=50$, $\kappa=0.06$

所有三种都consistently outperform vanilla BC，验证WVM能从noisy data中filter出真progress。

---

## 9. Ablation研究 (Table 4)

| Metric | Ours | w/o $\mathcal{L}_{\text{video}}$ | scratch | frozen | $p=0$ | $p=1$ | HL-Gaussian |
|---|---|---|---|---|---|---|---|
| Hesitation-RMSE↓ | 0.05 | 0.08 | 0.08 | 0.12 | 0.09 | 0.05 | 0.06 |
| Retry-VOC↑ | 0.78 | 0.68 | 0.62 | 0.45 | 0.67 | 0.75 | 0.59 |
| Expert-VOC↑ | 0.95 | 0.95 | 0.96 | 0.92 | 0.98 | 0.91 | 0.87 |

### 9.1 Video co-training的necessity

- **w/o $\mathcal{L}_{\text{video}}$**：Hesitation-RMSE 0.05→0.08，Retry-VOC 0.78→0.68，但Expert-VOC不变（0.95）
- **scratch（video stream从零训练）**：Retry-VOC 0.62
- **frozen（freeze video权重）**：最差，Hesitation-RMSE 0.12，Retry-VOC 0.45

**Intuition**：pretrained video world model的prior是起点，但必须继续co-training才能让feature适合value prediction。完全freeze等于"读取"一个为生成优化的latent，对value任务suboptimal。

### 9.2 Prefix randomization trade-off

- $p=0$：Expert-VOC虚高0.98，但Hesitation-RMSE 0.09，Retry-VOC 0.67 → **prefix shortcut问题**
- $p=1$：Expert-VOC掉到0.91，但suboptimal recovery（Retry-VOC 0.75）→ cross-chunk consistency破坏
- $p=0.5$：完美balance

这个ablation强烈建议future value model evaluation必须用suboptimal data，否则会被shortcut model欺骗。

### 9.3 Flow matching vs HL-Gaussian head

HL-Gaussian用51个bin，soft target是Gaussian-smoothed one-hot：

$$p_k(v) = \frac{\exp(-(v - c_k)^2 / 2\sigma^2)}{\sum_j \exp(-(v - c_j)^2 / 2\sigma^2)}, \quad \sigma = \frac{1}{K-1}$$

- $c_k = (k-1)/(K-1)$：bin中心
- $\sigma$：设为一个bin width
- $K=51$：bin数

Loss是soft cross-entropy（等价KL）：

$$\mathcal{L}_{\text{value}}^{\text{HLG}} = \mathbb{E}_v\left[-\sum_{k=1}^K p_k(v) \log \text{softmax}(z)_k\right]$$

Inference直接expectation：

$$\hat{v} = \sum_{k=1}^K \text{softmax}(z)_k \cdot c_k$$

**结果**：HL-Gaussian的Hesitation-RMSE只略升（0.05→0.06），但Retry-VOC从0.78→0.59大幅下降，Expert-VOC 0.95→0.87。

**Intuition**：categorical head的fixed bin support保留了conditional mean（所以RMSE差不多），但丢了fine-grained density variation，对ordinal metric（VOC）很伤。Flow matching head是continuous return density，不bound support或resolution，能保留local value differential——这对相邻chunk的ranking至关重要。

这个对比其实呼应了Stop Regressing那篇的观察，但flow matching比categorical更进一步——它不是discrete classification，而是continuous distribution learning。

---

## 10. 训练细节

### 10.1 Hardware & 时间

- 32× NVIDIA A100-SXM4-40GB
- ~40小时wall-clock
- 30,000 steps
- Global batch size 1024
- AdamW, $\beta_1=0.9$, $\beta_2=0.95$, weight decay 0
- Peak LR $1\times10^{-4}$, cosine decay到0.1×peak, 500 warmup
- bf16 mixed precision
- Gradient clip max norm 1.0

### 10.2 推理

- Single Euler step（不需要多步denoising）
- Chunk size $h=4$
- Overlapping-window averaging

**为什么1步够**：训练corpus相对model capacity中等规模，velocity field已经足够平滑，1步Euler就接近ground-truth value chunk。多步没有measurable gain。

这跟diffusion policy里常见1步inference的观察一致——当training data相对于capacity不爆炸时，rectified flow的straight path让1步足够。

### 10.3 Training mixture (Table A.2)

总407,086条trajectory，1410.83小时：
- RoboCOIN: 98,171条, 673.80h, 30/50 fps
- EgoDex: 299,100条, 688.56h, 30 fps
- RoboReward: 7,428条, 36.01h, 10 fps
- RoboSuite (self): 1,865条, 11.32h, 10 fps
- AgileX single (self): 160条, 0.39h, 15 fps
- AgileX dual (self): 120条, 0.26h, 15 fps
- ARX (self): 242条, 0.50h, 15 fps

Latent target FPS: 2.0（AgileX/ARX self-collected用3.0）

---

## 11. Baseline实现细节

### GVL
- gpt-5.4 API backbone
- 32 frames/call
- Autoregressive completion-percentage on shuffled frames

### VLAC
- InternRobotics/VLAC checkpoint
- InternVL backbone
- Single-pass 32 frames, temp 0.5, batch 32
- Disabled image branch

### Robometer
- robometer/Robometer-4B checkpoint
- Qwen3-VL-4B
- Multi-anchor: 5 anchors × 8 frames, subsample 5

### RoboReward
- teetone/RoboReward-4B
- Qwen3-VL-4B
- 5 anchors × 32 frames, gen len 128

### TopReward
- Qwen/Qwen3-VL-8B-Instruct
- 15 prefix samples, mean reduction
- 2 fps, max 32 frames

### Robo-Dopamine
- tanhuajie2001/Robo-Dopamine-GRM-3B
- Incremental mode, frame_interval=1
- Up to 32 frames/traj

---

## 12. 局限与未来方向

1. **Scale限制**：训练corpus相对小，zero-shot到完全unseen task/scene能力受限
2. **Task scope**：Suboptimal-Value-Bench主要pick-and-place，dexterous和long-horizon未覆盖
3. **Future**：scale up training mixture，扩展benchmark到dexterous和long-horizon

---

## 13. 我对这篇paper的几点intuition

### 13.1 为什么world model > VLM做value backbone

VLM是image-text对齐的产物，pretrain任务不要求temporal consistency——一张图配一段text就够了。Video world model的pretrain任务是"预测下一帧/下一段video"，这本质上是next-token prediction on temporal sequence，必须capture dynamics。

Value function by definition是对future的期望，所以"对未来敏感"的representation自然更合适。这不是"用更fancy的backbone"，是任务对齐的natural选择。

### 13.2 Chunk-level prediction的深层意义

Scalar value是Markovian的，但实际task progress有temporal structure——plateau、retry、hesitation都是temporal pattern。Chunk-level prediction逼model学local progress profile，这相当于在value层面做"local temporal modeling"，和video prediction的"local frame modeling"对齐。

Flow matching在chunk上做相当于学一个"local value velocity field"，每个chunk内的value变化方向被显式建模。

### 13.3 Asymmetric MoT的"老师-学生"隐喻

Video stream不被value stream干扰是关键——这保持了world model的generation capability。Value stream从video latent借力但反向不传播梯度。这就像有个pretrained好的老师，学生在旁边听讲，学生进步不影响老师。

如果用symmetric attention或者naive concat，video generation会被value loss污染，world model prior退化。

### 13.4 Prefix randomization与Expert-VOC的"自证陷阱"

Ablation最interesting的发现：$p=0$ 时Expert-VOC最高0.98但suboptimal最差。这说明现有社区用Expert-VOC作为唯一metric的训练方法可能学到的是"prefix extrapolation shortcut"而非真正的value understanding。

这给未来value model evaluation一个重要warning：必须用suboptimal data test，否则会reward shortcut learner。

### 13.5 Flow matching > Categorical的Ordinal Sensitivity

HL-Gaussian的conditional mean和flow matching差不多（RMSE相近），但ordinal metric（VOC）差很多。这暗示value model evaluation的metric选择决定了head选择——如果你只care mean，categorical够；care ranking/progress profile，flow matching显著优。

这跟Distributional RL里C51 vs QR-DQN vs IQN的讨论一脉相承，但flow matching走得更远——continuous distribution without bin support限制。

### 13.6 Video co-training as Implicit Regularizer

Frozen video weights最差，scratch次之，co-training最好——这告诉我们pretrained prior是起点而非终点。Value gradient通过MoT反向传播到video stream，让video latent对value task更"友好"。同时 $\mathcal{L}_{\text{video}}$ 防止video stream"漂离"generation任务，保持prior不退化。

这种mutual benefit的co-training pattern在multimodal learning里越来越常见，paper的ablation给了quantitative证据。

---

## 14. 相关工作脉络

### 14.1 World model系
- Ha & Schmidhuber (2018): World Models经典
- Dreamer (Hafner et al., 2019): Latent imagination
- V-JEPA 2 (2025): Self-supervised video model
- Wan2.2: WVM的直接backbone

### 14.2 World Action Model系
- WAM (Wang et al., 2026): Action-conditioned visual dynamics
- Fast-WAM (Yuan et al., 2026): Latent space prediction，无pixel decoding
- Cosmos Policy, GigaWorld-Policy, MotuBrain, Causal World Modeling: 各家WAM

### 14.3 Value Model系
- VIP / R3M / LIV: Visual representation as reward
- GVL (Ma et al., 2024): VLM in-context value learner
- Robometer (2026): Trajectory comparison reward
- RoboReward (2026): VLM reward
- Robo-Dopamine (2025): Process reward model
- VLAC (2025): Vision-language-action-critic
- TopReward (2026): Token probability as zero-shot reward
- ViVa (2026): 最接近WVM，但single-task + action-annotated

### 14.4 Distributional RL
- C51 (Bellemare et al., 2017): Distributional perspective
- Stop Regressing (Farebrother et al., 2024): Value via classification
- HL-Gaussian: WVM的ablation baseline
- Value Flows (Dong et al., 2025): 相邻value的flow

### 14.5 Foundation Model for Robotics
- π0.5 / π0.6* / π0.7*: Physical Intelligence的VLA系列
- X-VLA, Robo-Mutual, Universal Actions: Cross-embodiment
- DROID, OXE, RoboCOIN, EgoDex: 数据集

---

## 15. 我会建议的follow-up

1. **Scale video co-training到更大Wan2.2**：5B video DiT + 0.7B value DiT可能还能push，试14B的Wan
2. **Long-horizon Suboptimal benchmark**：当前都是pick-and-place，扩展到multi-step task with chained suboptimal segments
3. **Action-conditioned WVM**：当前value model是action-free的，加action token可能给online RL用
4. **Test-time flow refinement**：1步Euler够是因为corpus小，scale后多步可能有用
5. **Hierarchical chunk**：当前固定 $h=4$，multi-scale chunk（如h=4,8,16）可能capture不同temporal scale的progress pattern

---

这篇paper的core insight其实很Karpathy-style：**用对任务对齐的pretrain prior（video world model for temporal future）而非off-the-shelf prior（VLM for static image-text）**，配合minimal but principled architectural choices（MoT asymmetric mask + flow matching chunk + prefix randomization），在工程上达到SOTA。整个story非常clean，ablation扎实，benchmark contribution实在。

如果你之后想聊distributional value head的数学细节、MoT的attention实现、或者Suboptimal-Value-Bench的标注pipeline怎么scale到10K trajectory，我都可以深入展开。
