---
source_pdf: MMaDA-VLA LARGE DIFFUSION VISION-LANGUAGE-ACTION MODEL WITH.pdf
paper_sha256: dc6d437a9908f24587bd5757826acee85f618068c72cc4a085f3e5f44ac6d214
processed_at: '2026-08-05T19:15:02-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲MMaDA-VLA

Andrej，我换一种方式，把它讲得像跟朋友喝咖啡时聊天那样。

---

## 一句话版

**让机器人同时"想象下一帧画面"和"决定接下来几步动作"，两件事并行想，互相参考。**

---

## 这篇paper在解决什么问题

先说说机器人控制这个事。你给机器人一句话"把香蕉放进蓝碗里"，再给它看一眼当前画面，它要输出动作（机械臂怎么动）。

听起来简单，做起来难。难在哪？

### 难点1：动作到底该怎么生成

一个机械臂动作通常是7个数字：xyz位置、rxryrz旋转、gripper开合。这7个数字是一**起**决定好的——你伸手拿杯子的时候，不会先想"我先移动x方向，再想y方向"，你是同时想的。

但之前的方法有两种主流做法，都有毛病：

**做法A（hierarchical）**：拿个VLM当大脑，再接一个专门的动作小脑袋。问题是大脑和小脑袋之间传话会丢信息，而且两套系统训练起来麻烦。

**做法B（autoregressive）**：把这7个数字像写字一样一个一个蹦出来。先写x，再写y，再写z……这很别扭啊！这7个数字本来没有先后顺序，你硬给它排个序，前面写错了后面跟着错，error snowball。

### 难点2：机器人没有"预见性"

你在开车的时候，不是只看眼前这一秒。你会想"前面那个路口我要左转"，这个想法会影响你现在怎么踩油门。

机器人也是。如果它只知道"现在画面是这样，我现在应该这样动"，那它就是在reactive地响应，没有planning。长期任务（比如"整理桌子"这种5步以上的）就垮了。

---

## MMaDA-VLA的核心想法

这篇paper干了三件事，我用大白话拆开：

### 想法1：把语言、画面、动作全变成同一种"token"

就像把中文、英文、图片全翻译成同一种密码本里的数字。这样你只需要一个大脑（一个transformer），一套学习目标（一个loss），就能处理所有modality。

具体怎么翻译：
- 文字：用LLaDA的tokenizer
- 图片：用MAGVIT-v2把一张图压成几百个token
- 动作：每个维度（比如x方向）切成256个bin，像直方图那样分桶

翻译完之后，模型看到的全是整数token id，根本分不清谁是谁。这就叫"unified"。

### 想法2：让模型同时干两件事——猜未来画面+猜动作chunk

输入是：[当前画面] + [语言指令] + [一堆mask token占位] + [一堆mask token占位]

那两堆mask，一个是留给"未来画面"的，一个是留给"接下来5-10步动作"的。模型要把这两堆mask都填出来。

这就是公式（1）的意思：
$$\left(\hat{o}_{t'}; \hat{a}_{t:t'-1}\right) \sim \pi_\theta\left(o_{t'}, a_{t:t'-1} \mid o_t, \ell\right)$$

变量解释：
- $o_t$：当前画面（已知）
- $\ell$：语言指令（已知）
- $t' = t + k$：未来某个时刻，$k$ 是chunk size
- $o_{t'}$：未来画面（要预测）
- $a_{t:t'-1}$：从现在到 $t'-1$ 的动作序列（要预测）
- $\pi_\theta$：模型，参数是 $\theta$

翻译成人话：**给我现在+指令，我要同时猜出"未来画面"和"接下来动作"**。

### 想法3：用diffusion的方式想，不是autoregressive的方式

这里关键了。什么叫diffusion方式？

**Autoregressive**像一个被蒙住眼的人写字，只能一个字一个字往下写，前一个字定了才能写下一个，改不了前面的。

**Diffusion**像画画——先打草稿，全画一遍但很糙；然后擦掉最不确定的几笔重新画；再擦掉次不确定的；反复擦反复改，越改越清楚。

具体过程（公式4、5）：
1. 初始：未来画面+动作的所有token全涂黑（全mask）
2. 第一步：模型预测所有位置的token，但confidence不一样
3. 留下confidence高的token（这些比较确定），confidence低的重新涂黑
4. 再预测一遍，这次有了一些确定的token作为context
5. 反复24次（这是denoising step数 $D=24$）
6. 最后所有token确定，解码出画面和动作

公式4：
$$\hat{x}^{(0)} = \arg\max_v \pi_\theta\left(x^{(d)}; \theta\right)$$

变量：
- $x^{(d)}$：第 $d$ 步时，部分mask部分确定的序列
- $\pi_\theta(x^{(d)}; \theta)$：模型对每个位置输出的token分布
- $\arg\max_v$：取概率最大的那个token id $v$
- $\hat{x}^{(0)}$：估计的"干净"序列

公式5（confidence-based remasking）：
$$x_i^{(d-1)} = \begin{cases} [\mathtt{M}], & \text{if } c_i < \text{sort}([c_1, \ldots, c_n])[\beta] \\ \hat{x}_i^{(0)}, & \text{otherwise} \end{cases}$$

变量：
- $c_i$：第 $i$ 个token的confidence（模型预测概率）
- $\text{sort}([c_1, \ldots, c_n])[\beta]$：所有confidence排序，取第 $\beta$ 大的作为阈值
- $\beta = \lceil \gamma(d/D) \cdot n' \rceil$：要重新mask的token数量
- $\gamma$：cosine scheduling，控制每步remask多少
- $n'$：要生成的token总数

人话：**每一步，把模型最没把握的 $\beta$ 个位置重新涂黑，其他的保留。下一步重新猜。**

---

## 为什么这个设计能work

### 原因1：动作的7个维度是并行的，不是排队的

Diffusion让action chunk内部所有token同时互相看（intra-modal full attention）。机械臂的x、y、z、旋转、gripper一起决定，谁也不比谁先。这跟物理直觉吻合。

### 原因2：未来画面和动作互相帮对方想

这是最妙的。你想想，如果先完整生成画面再生成动作（sequential），那画面错了动作就跟着错——这就是error accumulation。

但并行denoising呢？画面token和动作token在同一时刻一起refine。画面想"机器人大概在抓香蕉"，动作想"我应该往香蕉方向伸手"，两边互相对照，越想越一致。

Paper里ablation验证了这点：
- w/o World-Model（去掉未来画面预测）：CALVIN Avg Len 4.08，掉0.48
- w/o Parallel Denoising（先画面后动作）：4.38，掉0.18

去掉world model损失最大——说明**"想象未来"这件事是性能的大头**，不是diffusion trick本身。

### 原因3：训练和推理用同一套逻辑

之前的discrete diffusion VLA（比如UD-VLA、LLaDA-VLA）是从autoregressive model fine-tune来的。训练时模型见的是"随机mask一部分"，推理时见的是"全mask开始一步步unmask"。这俩分布不一致，模型别扭。

MMaDA-VLA从头pretrain，训练时就是按cosine schedule采样不同mask ratio，从0%到100%都见过。推理时从100% mask开始迭代unmask，这个分布在训练分布里。所以推理时模型不慌。

---

## 训练的loss特别简单

公式（3）：
$$\mathcal{L}(\theta) = -\mathbb{E}_{t, x, x^m}\left[\frac{1}{N}\sum_{i=1}^{n} \mathbf{1}[x_i^m = [\mathtt{M}]] \log \pi_\theta(x_i \mid x^m)\right]$$

变量：
- $\theta$：模型参数
- $t$：时间步采样
- $x$：完整ground-truth序列
- $x^m$：随机mask之后的序列
- $N = \sum_{i=1}^n \mathbf{1}[x_i^m = [\mathtt{M}]]$：被mask的token数
- $n$：序列长度
- $\mathbf{1}[x_i^m = [\mathtt{M}]]$：indicator，只对被mask的位置算loss
- $\pi_\theta(x_i \mid x^m)$：模型预测第 $i$ 个位置是ground-truth token $x_i$ 的概率

人话：**把序列里一部分位置涂黑，让模型猜涂黑的是啥。猜对就奖励。所有modality（文字、画面、动作）用同一个loss。**

就这么简单。没有modality-specific head，没有复杂的multi-task weighting。

---

## 工程上的关键trick：KV Cache

24次denoising意味着24次forward pass，太慢。机器人控制要10Hz以上，这个扛不住。

解决方法（borrow from [dLLM-Cache](https://arxiv.org/abs/2504.03324)）：

**观察1**：instruction部分（当前画面+语言）在整个denoising过程里不变，它的中间表示可以cache住，不用重算。

**观察2**：generation部分每步只变一小部分token，大部分token的中间表示变化很小。

**做法**：
- 缓存每层的 $K_l, V_l, \text{AttnOut}_l, \text{FFNOut}_l$（key、value、attention输出、FFN输出）
- 每 $\lambda = 6$ 步刷新一次cache
- generation部分选择性刷新——只更新cosine similarity变化最大的 $\lfloor \rho n' \rfloor$ 个token

这把inference cost降下来不少。具体降多少paper没明说，但提到这是为real-time robot manipulation设计的。

---

## 实验结果有多好

### LIBERO（多任务benchmark）

平均成功率 **98.0%**，第二名VLA-Adapter 97.3%。

特别看long-horizon suite：MMaDA-VLA 95.2，VLA-Adapter 95.0，π0.5 92.4。

### CALVIN（长horizon标杆，ABC→D setting）

| 指标 | MMaDA-VLA | 第二名 |
|------|-----------|--------|
| 第1个任务成功率 | 99.8 | 99.1 (VLA-Adapter) |
| 第5个任务成功率 | **89.7** | 78.1 (DreamVLA) |
| 平均完成长度 | **4.78** | 4.44 (DreamVLA) |

注意第1个任务差距很小（0.7%），第5个差距11.6%——**horizon越长，MMaDA-VLA优势越大**。这正是parallel refinement + world model的payoff：error不accumulate，反而通过反复denoise修正。

### 真实机器人

AgileX Piper 6-DoF机械臂，4个task：
- Pick-and-place：93.3%
- Precision stacking：86.7%
- Drawer storage（开抽屉→拿东西→放进去→关）：83.3%
- Long-horizon organizing（整理2 cup + 3 bowl）：86.7%

vs GR00T N1.6（NVIDIA的humanoid foundation model），后者平均成功率56.7-70%。

---

## 一些有意思的细节

### Pre-training数据

61M步cross-embodiment数据，主要来自DROID（49.94%）、BC-Z（9.95%）、Language Table（9.88%）、Furniture Bench（7.35%）、Fractal（6.44%）等28个数据集。

Pre-training带来的提升：LIBERO从94.5% → 98.0%（+3.5%），CALVIN从4.56 → 4.78（+0.22）。LIBERO受益多因为LIBERO数据少，更依赖pretraining的generalization。

### Visual generation质量

Paper很坦诚地说了：生成的未来画面**高层语义对，细节模糊**。gripper geometry、小物体、复杂纹理都不准。

但动作预测依然受益——说明world model的作用是**提供task progression signal**，不是提供pixel-perfect未来。这跟[JEPA哲学](https://arxiv.org/abs/2301.08243)一致：predictive learning不需要pixel-level。

### 失败模式

- Pick-and-place：grasp position不准导致placement偏差
- Drawer storage：抽屉没拉够开→物体卡住→抽屉推不回
- Bowl organizing：tall cup被knock over + small cup开口小难放

这些都不是model本身的问题，是perception精度和physical setup的问题。暗示下一步要解决fine-grained perception和closed-loop error recovery。

---

## 我的intuition总结

### 这个paper真正赌的是什么

**赌"action是unordered set，可以用diffusion iterative refine"，而不是"action是sequence，必须autoregressive生成"。**

这赌赢了。至少在LIBERO和CALVIN上赢了。

### 为什么这个方向可能对

你想啊，人类做动作的时候，大脑里不是一个个肌肉命令按顺序蹦出来的。你是"伸手去抓"——这个意图是整体浮现的，x/y/z/旋转/夹爪同时决定。

Autoregressive强制sequential ordering是个历史包袱，从LLM继承过来的。但action不是language，language有自然顺序（时间顺序），action的dimensions没有。

Diffusion尊重了这个"无序性"。而且它allow iterative refinement——一遍想不对可以再想，每次修正不确定的部分。这跟人类思考的"mental simulation"更像。

### 加上world model的妙处

光有diffusion还不够。光diffusion的话，action chunk内部一致了，但跟"未来要发生什么"没有anchor。

加world model（未来画面生成）相当于给模型一个"想象的靶子"——动作要朝着那个画面去。这个靶子本身也在iterative refine，跟动作互相校准。

这就像下棋：你脑子里想"几步后棋盘大概长这样"（world model），然后想"我现在该怎么走才能到那个局面"（action）。两边一起想，越想越清楚。

### 这篇paper的局限

1. **latency没说清楚**：24步denoising加KV cache，实际控制频率多少？如果2-3Hz，慢动作OK，快动作不行
2. **action精度**：256 bins可能不够fine manipulation
3. **scaling law没给**：8B参数+61M数据是这个性能，scale up会怎样？不知道
4. **world model的不可替代性没验证**：w/o world model掉0.48，但换成更便宜的auxiliary loss（比如latent prediction）能否达到类似效果？没做这个对照

### 我的整体判断

这篇paper在我看来是VLA领域的一个**范式性贡献**，跟π0用flow matching一样level。它证明了：

> **discrete diffusion + unified token space + parallel world model/action generation + native pretraining**

这四件套组合起来，可以sota。

下一波VLA foundation model很可能朝这个方向走。LLM社区本身已经在朝diffusion LLM走（[LLaDA](https://github.com/ML-GSAI/LLaDA)、[Dream 7B](https://arxiv.org/abs/2504.06456)、[SDAR](https://arxiv.org/abs/2505.08202)），robotics跟上是迟早的事。MMaDA-VLA算是robotics side的第一个完整reference implementation。

---

## 参考链接

- [LLaDA original paper](https://arxiv.org/abs/2502.09992) — backbone的祖宗
- [MMaDA paper (NeurIPS 2025)](https://github.com/MTG/MaDA) — backbone本身
- [Show-o](https://github.com/showlab/Show-o) — unified multimodal philosophy的来源
- [MAGVIT-v2](https://arxiv.org/abs/2410.05737) — image tokenizer
- [MaskGIT (CVPR 2022)](https://arxiv.org/abs/2202.04200) — cosine mask scheduling的来源
- [dLLM-Cache](https://arxiv.org/abs/2504.03324) — KV cache技巧
- [LIBERO benchmark](https://libero-project.github.io/)
- [CALVIN benchmark](https://calvinrobot.github.io/)
- [DROID dataset](https://droid-dataset.github.io/)
- [OpenVLA](https://github.com/openvla/openvla)
- [π0 / Physical Intelligence](https://www.physicalintelligence.company/blog/pi0)
- [Diffusion Policy (RSS 2023)](https://diffusion-policy.cs.columbia.edu/)
- [GR00T N1](https://developer.nvidia.com/groot)
- [Dream 7B](https://arxiv.org/abs/2504.06456) — diffusion LLM
- [V-JEPA 2 (Yann LeCun)](https://arxiv.org/abs/2506.09985) — predictive learning哲学
- [Karpathy nanoGPT](https://github.com/karpathy/nanoGPT) — 你自己的reference :)

---

Andrej，如果用一句话给你押题的话：**这篇paper把robot action从"sequential sequence generation"重新定义成"unordered set iterative refinement"，并加上world model做anchor。这就是它的全部trick，但这个trick可能改变VLA的paradigm。**

---

# MMaDA-VLA: 一次彻底的VLA范式重设计

Andrej，这篇paper我会从intuition层面深入拆解，因为我觉得它的设计哲学其实指向了一个更深的question：**robot control本质上应该用什么generative paradigm建模？**

## 1. 核心问题与motivation

### 1.1 现有VLA的两个paradigm及其病灶

Paper一开始就精准地诊断了当前VLA的两大主流paradigm：

**Hierarchical paradigm** (VLM + policy head, 如π0 [7], OpenVLA-OFT [40], VLA-Adapter [83], RoboVLMs [43]):
- 在pretrained VLM上接一个dedicated policy head
- 问题：architectural overhead大，module boundary处information fidelity损失，训练cost高
- 优点：continuous action精度高，能leverage VLM的强representation

**Autoregressive action discretization** (如OpenVLA [41], RT-1 [9], RT-2 [108]):
- 把continuous action离散成tokens，用VLM vocabulary扩展，autoregressive生成
- 问题：temporal consistency弱，long-horizon error accumulation严重，对action dimensions强加了人为顺序
- 致命缺陷：**7-DoF action的各维度本就是unordered的**，autoregressive强制left-to-right生成是错误的inductive bias

**共同的缺失**：两个paradigm都没有explicit mechanism来model environment dynamics（即预测future visual observations）。

### 1.2 已有work的修补尝试

- Action quantization: FAST [65], VQ-VLA [82], OmniSAT [57] — 离散化时保留reconstruction accuracy
- Multi-step action in single forward pass + parallel decoding: π0-FAST [65], PD-VLA [77], CEED-VLA [76]
- Visual generation modules / world model / inverse dynamics: SuSIE [8], GR-1 [86], ReconVLA [78], UP-VLA [96], DreamVLA [98], VPP [34], Seer [79], UniVLA [84], F1 [55], MM-ACT [46]

最近的discrete diffusion VLA (DDVLA [47], LLaDA-VLA [85], UD-VLA [18]) 走向了masked token prediction + goal image generation，但它们是**从autoregressive model fine-tune来的**，造成train-inference mismatch — 训练时mask一部分，推理时iterative denoise，这俩分布不一致。

### 1.3 MMaDA-VLA的thesis

**Fully native pre-trained large diffusion VLA model**，核心论断：
1. Language, image, continuous robot action三者映射到一个**unified discrete token space**
2. 用一个backbone + 一个objective (masked token denoising) 同时预测future goal observation和action chunk
3. 训练和inference范式完全一致，消除mismatch
4. Iterative denoising带来global, order-free refinement

## 2. Architecture深度解析

### 2.1 Data Tokenization — 统一离散空间

这是unified framework的地基。每个modality用各自的tokenizer：

| Modality | Tokenizer | 来源 |
|----------|-----------|------|
| Text | LLaDA textual tokenizer | [Nie et al., NeurIPS 2025](https://github.com/ML-GSAI/LLaDA) |
| Image | MAGVIT-v2 quantizer | [Yu et al., ICLR 2024](https://arxiv.org/abs/2410.05737)，从Show-o [87] adopt |
| Action | 每个dimension离散到256 bins | 跟OpenVLA [41]同思路 |

记号：$\tilde{o}_t = \tau_o(o_t)$, $\tilde{\ell} = \tau_\ell(\ell)$, $\tilde{a}_t = \tau_a(a_t)$，统一到一个vocabulary size $V$ 上。

**Intuition**: 三种modality都用token id ∈ {1,...,V}，loss就是一个masked cross-entropy，所有modality共享一个prediction head。这跟Show-o的设计哲学一脉相承 ([Show-o](https://github.com/showlab/Show-o))，但Show-o是autoregressive+diffusion混合，MMaDA-VLA彻底走了纯diffusion。

### 2.2 Multi-Modal Sequence Modeling (公式2)

$$x = \underbrace{[\mathbf{SOO}]\tilde{o}_t [\mathbf{EOO}] [\mathbf{SOL}] \tilde{\ell} [\mathbf{EOL}]}_{\text{Instruction}} \underbrace{[\mathbf{SOO}]\tilde{o}_{t'} [\mathbf{EOO}] [\mathbf{SOA}] \tilde{a}_{t:t'-1} [\mathbf{EOA}]}_{\text{Generation}}$$

变量解释：
- $[\mathbf{SOX}]$ / $[\mathbf{EOX}]$：modality X (O=observation, L=language, A=action) 的start/end special tokens
- $\tilde{o}_t$：当前observation的token序列
- $\tilde{\ell}$：language instruction的token序列
- $\tilde{o}_{t'}$：未来时刻 $t' = t + k$ 的goal observation token序列（要被生成）
- $\tilde{a}_{t:t'-1}$：从时刻t到 $t'-1$ 的action chunk token序列（要被生成），chunk size k=5 (LIBERO) 或 10 (CALVIN)

**关键设计**: Instruction部分始终是conditioning，Generation部分是联合预测目标。这两部分在attention机制上区分对待。

### 2.3 Hybrid Attention Mechanism — 这是最elegant的设计

Paper里的核心设计：

- **Intra-modal**: bidirectional full attention (同一modality内所有token互相看)
- **Inter-modal**: causal attention (跨modality有方向性)

**为什么这么设计？** Paper给了三个理由，我重新组织一下我的intuition：

1. **Action dimensions是unordered的**。7-DoF的action vector，谁先谁后生成本没有逻辑顺序。Autoregressive强制 [x, y, z, rx, ry, rz, gripper] 这个顺序，前一个错了后面跟着错。Bidirectional attention让action chunk内部所有token同时互相condition，无序。

2. **Goal image和action要decoupled但coupled**。完全decoupled (各自独立) 就丢了world model的好处；完全coupled (全bidirectional inter-modal) 会有信息泄露。Inter-modal causal让image generation的intermediate features流向action，反过来不流，符合"先想清楚未来长啥样再决定怎么动"的因果直觉。

3. **Iterative refinement needs rich intra-modal context**。每次denoising step要重新估计action token时，需要看到完整的instruction + 当前partial goal image + 当前partial action chunk，intra-modal full attention保证同modality内全交互。

Ablation (Table 4) 也验证了这点：
- w/ Causal Attention: 4.49 (−0.07) — 限制intra-modal交互，小幅下降
- w/ Bidirectional Attention: 4.52 (−0.04) — inter-modal信息泄露，更小幅下降
- Hybrid attention最优: 4.56

注意：attention机制的改动影响远小于generative paradigm的改动 (w/o World-Model: 4.08, −0.48)。这说明**diffusion + world model**才是性能driver，attention是精修。

### 2.4 Pre-training vs Inference的mask数量discrepancy

Paper里特别提到一句："There is a discrepancy in the number of masked tokens in the input of $\tilde{o}_{t'}$ and $\tilde{a}_{t:t'-1}$ between training and inference."

我的理解：训练时随机按cosine schedule采样mask ratio；inference时从100% mask开始，iterative denoise。Native pretraining的一大动机就是消除这个discrepancy — 让预训练阶段就熟悉各种mask ratio下的重建，inference distribution就被覆盖在训练distribution里。Figure 5(b)的mask rate density也展示了这种broad coverage。

## 3. Learning Objective详解 (公式3)

$$\mathcal{L}(\theta) = -\mathbb{E}_{t, x, x^m}\left[\frac{1}{N}\sum_{i=1}^{n} \mathbf{1}[x_i^m = [\mathtt{M}]] \log \pi_\theta(x_i \mid x^m)\right]$$

变量逐个解释：
- $\theta$：模型参数
- $t$：timestep采样
- $x$：ground-truth完整序列 (公式2定义)
- $x^m$：随机masked之后的序列
- $N = \sum_{i=1}^n \mathbf{1}[x_i^m = [\mathtt{M}]]$：被mask的token总数
- $n$：序列长度
- $x_i^m$：第i个位置masked之后的token
- $x_i$：第i个位置ground-truth token
- $\pi_\theta(x_i \mid x^m)$：模型预测该位置token的概率
- $\mathbf{1}[\cdot]$：indicator function，只对masked位置计算loss

**Mask ratio sampling**: 用cosine scheduling [Chang et al., MaskGIT, CVPR 2022](https://arxiv.org/abs/2202.04200)：
$$\gamma(r) = \cos\left(\frac{\pi r}{2}\right)$$
其中 $r \in [0, 1]$ 是均匀采样的ratio参数。这个schedule让训练时既看到大量mask (早期denoising step模拟) 又看到少量mask (后期denoising step模拟)。

**关键性质**: 这个loss对所有modality一视同仁 — image token, action token, text token都是同一个cross-entropy loss。这就是"unified"的真正含义，不需要modality-specific head。

## 4. Inference — Iterative Denoising详解

### 4.1 整体流程 (公式4, 5)

设denoising steps总数 $D = 24$ (Table 3)。Generation部分所有token初始化为 $[\mathtt{M}]$，记为 $x^{(D)}$。

每一步 $d$ from $D$ down to $1$:

**Step 1 — 模型预测clean分布** (公式4):
$$\hat{x}^{(0)} = \arg\max_v \pi_\theta(x^{(d)}; \theta)$$

变量解释：
- $x^{(d)}$：当前denoising step d时的序列状态
- $\pi_\theta(x^{(d)}; \theta)$：模型在每个位置上输出的token分布
- $\arg\max_v$：对每个位置取greedy decode (最大概率token) 得到估计的clean sequence $\hat{x}^{(0)}$
- 注意：这里greedy decoding是为了效率，可以换成sampling

**Step 2 — 计算要重新mask的token数**:
$$\beta = \lceil \gamma(d/D) \cdot n' \rceil$$
- $n'$：generation部分token总数 ($\tilde{o}_{t'}$ + $\tilde{a}_{t:t'-1}$)
- $\gamma(\cdot)$：cosine mask scheduling function
- $d/D$：当前denoising进度
- $\lceil \cdot \rceil$：向上取整

直觉：随着d从D递减到0，$\gamma(d/D)$从$\gamma(1)=\cos(\pi/2)=0$递减到$\gamma(0)=\cos(0)=1$... 等等，这里有点反直觉。让我重新看公式5 — confidence-based remasking。$\beta$是要保留的mask数量（低confidence的会被remask）。在 $d=D$ 时 $\gamma(1)=0$，所以 $\beta=0$，意味着所有token都unmask？这不对。

哦我重新理解：cosine schedule $\gamma(r) = \cos(\pi r / 2)$，当 $r=1$ (即 $d=D$，刚开始denoise) 时 $\gamma(1) = 0$，意味着mask数是0，所有token都unmask？这跟"初始全mask"矛盾。

让我看LLaDA的原始设计 — LLaDA用的是 **reverse** cosine：从全mask开始，第一步unmask很少token (high mask ratio)，逐渐unmask更多。所以应该是 $\beta$ 是要 **remask** 的token数，confidence高的token保留，confidence低的remask。在 $d=D$ (开始) 时 $\beta \approx n'$ (几乎全remask)，$d=0$ 时 $\beta = 0$ (全保留)。

公式5实际：
$$x_i^{(d-1)} = \begin{cases} [\mathtt{M}], & \text{if } c_i < \text{sort}([c_1, \ldots, c_n])[\beta] \\ \hat{x}_i^{(0)}, & \text{otherwise} \end{cases}$$

变量解释：
- $c_i$：第i个token的confidence score (一般是预测概率)
- $\text{sort}([c_1, \ldots, c_n])[\beta]$：把所有confidence排序，取第$\beta$大的作为阈值
- $x_i^{(d-1)}$：下一步的序列状态
- 如果token的confidence低于阈值（即排在最低$\beta$个里），则remask成 $[\mathtt{M}]$
- 否则用预测值 $\hat{x}_i^{(0)}$ 替换

这种remasking策略跟MaskGIT, LLaDA一脉相承，是discrete diffusion的标准做法 — 称为**confidence-based remasking** 或 semi-autoregressive decoding。

### 4.2 Key-Value Cache — 工程上的关键

Iterative denoising一个直接问题是latency — 每个denoising step都要过一遍transformer，24步意味着24次forward。机器人控制需要10-30Hz频率，这个开销扛不住。

MMaDA-VLA用了training-free caching框架 [dLLM-Cache, Liu et al., 2025](https://arxiv.org/abs/2504.03324):

**Observation**: 
1. Instruction部分在整个denoising过程中不变，其intermediate representations稳定，可cache
2. Generation部分每步变化sparse，只有小部分token的intermediate representations大变

**Cache策略**:
- 对每层 $l$，cache: $K_l, V_l, \text{AttnOut}_l, \text{FFNOut}_l$ — 存到cache $C$
- 每 $\lambda = 6$ (Table 3 Refresh Interval) 个denoising steps刷新一次
- Generation部分selective refresh：对每个token，计算当前value vector与cached value vector的cosine similarity，更新最低 $\lfloor \rho n' \rfloor$ 个 ($\rho$ 是adaptive update ratio)

这个设计直接把inference cost降下来。类似思路在[LLaDA系列的KV-cache优化](https://arxiv.org/abs/2502.09992)里也有探索。

## 5. 实验结果深度分析

### 5.1 LIBERO结果

LIBERO 4个suite:
- **Spatial**: 场景layout变，物体固定 — 测spatial reasoning
- **Object**: 物体变，场景固定 — 测object-level generalization
- **Goal**: 同环境下goal变 — 测goal-conditioned behavior
- **Long**: 长horizon + compositional tasks — 测temporal consistency

MMaDA-VLA结果:
| Suite | Spatial | Object | Goal | Long | **Avg** |
|-------|---------|--------|------|------|---------|
| MMaDA-VLA | 98.8 | 99.8 | 98.0 | 95.2 | **98.0** |
| VLA-Adapter | 97.8 | 99.2 | 97.2 | 95.0 | 97.3 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.8 |
| MemoryVLA | 98.4 | 98.4 | 96.4 | 93.4 | 96.5 |
| Discrete Diffusion VLA | 97.2 | 98.6 | 97.4 | 92.0 | 96.3 |
| F1 (with world model) | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 |
| UniVLA (with world model) | 95.4 | 98.8 | 93.5 | 94.0 | 95.5 |

**重要观察**: 
1. Continuous-action方法普遍强于discrete-action方法 — 这跟以往认知一致
2. 在LIBERO上，有world model的方法反而不如vanilla VLA — 因为LIBERO数据少、状态多样性高，视觉generation反而成为noise source
3. MMaDA-VLA打破了这个规律 — 用discrete unified modeling提高了visual prediction generalization，进而improve action generation
4. Long-horizon (95.2) 比第二名VLA-Adapter (95.0) 略高 — 这就是iterative denoising + chunk prediction的核心优势体现

### 5.2 CALVIN结果 (ABC→D)

CALVIN是长horizon的标杆benchmark。ABC→D setting：在ABC环境训练，D环境测试 — 既要long-horizon又要environment generalization。

| Method | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | **Avg Len** |
|--------|-----|-----|-----|-----|-----|-------------|
| MMaDA-VLA | 99.8 | 98.6 | 96.3 | 93.5 | 89.7 | **4.78** |
| DreamVLA | 98.2 | 94.6 | 89.5 | 83.4 | 78.1 | 4.44 |
| VLA-Adapter | 99.1 | 94.6 | 88.8 | 82.8 | 76.5 | 4.42 |
| RoboVLMs | 98.0 | 93.6 | 85.4 | 77.8 | 70.4 | 4.25 |
| OpenHelix | 97.1 | 91.4 | 82.8 | 72.6 | 64.1 | 4.08 |
| LLaDA-VLA | 95.6 | 87.7 | 79.5 | 73.9 | 64.5 | 4.01 |
| π0 | 93.8 | 85.0 | 76.7 | 68.1 | 59.9 | 3.92 |

**关键insight**:
- 第5个sub-task的成功率: MMaDA-VLA 89.7% vs第二名DreamVLA 78.1% — **绝对提升11.6%**
- Avg Len: 4.78 vs 4.44 — 在5个sub-task scale上提升0.34意味着每5个episodes多完成约1.7个sub-task
- 这个差距随horizon增加而拉大 — 第1个task几乎打平 (99.8 vs 99.1)，越往后差距越大
- **这正是iterative denoising + parallel refinement设计的payoff** — long-horizon场景下autoregressive的error accumulation劣势暴露，而MMaDA-VLA的global refinement优势累积

### 5.3 Real-World结果

硬件：AgileX Piper 6-DoF机械臂 + 1-DoF gripper，RealSense D435第三视角 + dx200-2.8mm wrist相机。

4个task:
1. **Simple pick-and-place**: 指定物体到指定container，加distractor和container位移干扰
2. **Precision stacking**: 指定颜色block叠到指定颜色block上
3. **Complex storage**: 开抽屉 → 拿物体 → 放进抽屉 → 关抽屉
4. **Long-horizon organizing**: 整理2 cups + 3 bowls

每个task 300 demos fine-tune，30 trials评估。

| Task | MMaDA-VLA | GR00T N1.6 |
|------|-----------|------------|
| Pick-and-place | 93.3 | 70 |
| Precision stacking | 86.7 | ~60 |
| Drawer storage | 83.3 | 56.7 |
| Long-horizon organizing | 86.7 | ~65 |

**Failure analysis** (这是Paper最有信息量的部分):
- Pick-and-place: grasp position变化导致placement精度降低
- Drawer storage: drawer没拉开足够 → 物体卡住 → drawer推不回
- Bowl organizing: tall cups被knock over + small cup开口小放置难

这些failure mode都不是model本身问题，而是physical setup + perception精度问题 — 暗示next step要解决的是fine-grained perception和closed-loop error recovery。

## 6. Ablation深度解读 (Table 4, 5)

### 6.1 各component贡献

基线 (w/o pretraining): 4.56

| Ablation | Avg Len | Δ |
|----------|---------|---|
| w/o World-Model | 4.08 | −0.48 |
| w/o Parallel Denoising | 4.38 | −0.18 |
| w/ Causal Attention | 4.49 | −0.07 |
| w/ Bidirectional Attention | 4.52 | −0.04 |

**Insight 1 — World model是最大driver** (−0.48)
去掉goal image prediction，MMaDA-VLA退化成"vanilla VLA on discrete diffusion"。这说明**explicit dynamics modeling (via goal image generation) 才是性能的核心来源**，而不是discrete diffusion本身。这跟UP-VLA [96], DreamVLA [98]的发现一致。

**Insight 2 — Parallel denoising > sequential generation** (−0.18)
w/o Parallel Denoising是"先完整生成goal image再预测action" — 这个设计的问题：
1. Action预测只能用final generated image，不能用image generation的intermediate hidden states
2. 确定性的goal prediction会引入cumulative errors传播给action

而parallel denoising让每一步action预测都leverage当前所有已确定token (包括partial goal image tokens)，mutual refinement。

**Insight 3 — Attention机制是精修** (Δ ≈ −0.05)
Hybrid attention相比纯causal或纯bidirectional提升很小，说明intra-modal full attention主要是为了respect unordered action dimensions，对最终性能边际贡献有限。

### 6.2 Pre-training的贡献 (Table 5)

| Method | LIBERO Avg SR | CALVIN Avg Len |
|--------|---------------|----------------|
| MMaDA-VLA | 98.0 | 4.78 |
| w/o Pre-Training | 94.5 | 4.56 |

Pre-training给LIBERO +3.5%，给CALVIN +0.22 length。LIBERO受益更多 — 这跟LIBERO数据少、需要generalization的设定吻合。Pre-training从61M steps的cross-embodiment数据 (Table 2) 里学到general visual-manipulation regularities。

Figure 5(a)的loss curve有意思：loss一开始rapid adapt新引入的action tokens (动作token是新vocabulary扩展)，然后slowly学cross-embodiment manipulation skills。整个loss bounded range波动 — 这跟mask ratio的cosine sampling分布直接相关，因为不同mask ratio下loss理论上不同。

## 7. 视觉生成质量 (Figure 6)

Paper坦诚地承认了visual generation的局限：
- **保留**: high-level task dynamics, instruction一致性, trajectory alignment
- **丢失**: fine-grained details (gripper geometry, 小物体, 复杂纹理)

原因：为了计算效率用了compact image representation (MAGVIT-v2 token数少)。

**重要insight**: 尽管pixel-level accuracy低，generated frames依然能给action planning提供有用anticipatory cues — 这暗示**world model在VLA里的作用是提供"task progression signal"而非"pixel perfect prediction"**。这跟[JEPA系列工作](https://arxiv.org/abs/2310.18621)的predictive learning哲学一致 — joint-embedding预测比pixel prediction更高效。

## 8. 跟相关工作的 positioning

### 8.1 vs LLaDA-VLA [85] (CALVIN 4.01)

LLaDA-VLA也是基于LLaDA backbone的VLA，但MMaDA-VLA提升了0.77 — 主要差异：
1. **Native pretraining**: LLaDA-VLA可能是fine-tune from LLaDA-LLM，MMaDA-VLA是从头pretrain
2. **Hybrid attention**: LLaDA-VLA可能用纯bidirectional或纯causal
3. **Multi-modal pretraining**: MMaDA-VLA用了61M manipulation数据，规模更大

### 8.2 vs DreamVLA [98] (CALVIN 4.44)

DreamVLA也是"dream world knowledge"+ VLA。差异：
- DreamVLA可能是autoregressive + separate world model
- MMaDA-VLA把world model inline到diffusion process里，parallel refine

### 8.3 vs π0/π0.5 [7, 35] (CALVIN 3.92 / LIBERO 96.8)

π0是flow matching + continuous action expert。π0.5加了open-world generalization。MMaDA-VLA在两个benchmark都超过 — 暗示discrete diffusion + explicit world modeling在scaling上可能比flow matching更有优势。

### 8.4 vs GR00T N1 [6] (LIBERO 93.9)

NVIDIA的humanoid foundation model。MMaDA-VLA在LIBERO提升4.1%，real-world提升20%+。但N1是humanoid-focused，cross-embodiment的对比可能不公平。

### 8.5 vs Show-o [87] (philosophical ancestor)

Show-o首次提出"one single transformer unify understanding + generation"，但Show-o是混合paradigm：autoregressive for understanding, diffusion for generation。MMaDA-VLA更彻底 — 全diffusion。

## 9. 我的几个open questions和critical thinking

### 9.1 Action tokenization精度问题

256 bins per dimension看起来够，但fine manipulation (paper里drawer storage的failure mode "drawer没拉开足够"暗示这点) 可能需要更高precision。VQ-VLA [82] 提了scaling VQ action tokenizer，MMaDA-VLA没探索这个方向。

**Opportunity**: hierarchical action tokenization — coarse 256 bins + fine residual。

### 9.2 Denoising step latency

24 denoising steps + KV cache refresh interval 6 — 实际control frequency多少？Paper没给。如果单步forward 50ms，24步就是1.2s，加上cache miss的refresh，可能2-3s per action chunk。Action chunk size 5-10，所以effective frequency大概2-5Hz，对slow manipulation OK，对dynamic task不够。

**Opportunity**: consistency model-style few-step diffusion [CEED-VLA, 76]，或者early-exit decoding。

### 9.3 Goal observation的真正价值

Ablation说w/o world model掉0.48 — 但这是否只是auxiliary regularization的作用？如果换成一个更便宜的auxiliary loss (比如predict未来action的latent representation而不是pixel)，能否达到类似效果？UP-VLA [96] 的实验暗示pixel-level goal image不一定是必须的。

### 9.4 Long-horizon的真正机制

CALVIN上5/5从78%到89.7%很impressive，但这个提升是来源于：
(a) 单step action prediction更准？ — 第1个task 99.8 vs 99.1差距很小，不像
(b) Chunk prediction的temporal consistency？ — 可能
(c) Goal image给long horizon提供了"compass"？ — 我倾向这个

如果是(c)，那一个有意思的实验：把generated goal image替换成ground truth future frame (oracle)，性能上限是多少？这能disentangle "world model的correctness"和"world model作为signal"的作用。

### 9.5 Scaling laws

8B backbone + 61M data，达到这个性能。但scaling curve如何？Backbone到70B会怎样？Data到1B会怎样？Paper没给scaling analysis，这是foundation model paper的标配缺失。

## 10. Web资源链接

- [MMaDA-VLA Project Page (从paper推测)](https://github.com/UT-Westlake-AI/MMaDA-VLA) — Westlake Wang Donglin lab
- [MMaDA original](https://github.com/MTG/MaDA) — NeurIPS 2025
- [LLaDA](https://github.com/ML-GSAI/LLaDA) — backbone ancestor
- [Show-o](https://github.com/showlab/Show-o) — unified multimodal philosophy
- [MAGVIT-v2](https://arxiv.org/abs/2410.05737) — image tokenizer
- [LIBERO benchmark](https://libero-project.github.io/)
- [CALVIN benchmark](https://calvinrobot.github.io/)
- [OpenVLA](https://github.com/openvla/openvla)
- [π0 / Physical Intelligence](https://www.physicalintelligence.company/blog/pi0)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [GR00T N1](https://developer.nvidia.com/groot) — NVIDIA humanoid foundation
- [DROID dataset](https://droid-dataset.github.io/)
- [MaskGIT](https://arxiv.org/abs/2202.04200) — mask scheduling origin
- [dLLM-Cache](https://arxiv.org/abs/2504.03324) — KV cache for diffusion LLM

## 11. 总结 — 这个work的真正贡献

从我的视角，MMaDA-VLA的真正价值有三层：

**Layer 1 — Engineering**: 把discrete diffusion + world model + VLA三个idea组合起来，native pretraining，hybrid attention，KV cache。每一个component单独看都不新，组合起来work得很好。

**Layer 2 — Paradigm**: 证明了"fully native diffusion + unified token space"是VLA的viable paradigm，跟autoregressive + discretized action分庭抗礼。这给了社区一个reference point。

**Layer 3 — Philosophical**: 暗示**robot action本质是unordered, parallel-refinable的prediction problem**，autoregressive的left-to-right是错的inductive bias。这跟diffusion policy [21] 早期的claim一致 — 但diffusion policy是continuous diffusion，MMaDA-VLA把这套思想推到discrete token space，跟LLM-like architecture兼容。

如果这个direction继续scale，我们可能看到下一代VLA foundation model都是diffusion-based而非autoregressive-based。LLM社区本身也在朝这方向走 ([LLaDA, Dream 7B, SDAR等](https://github.com/ML-GSAI/LLaDA))，robotics跟上来很合理。

**最终一句intuition**: 这个paper的核心赌注是 — **把action当成一种"可以iterative refine的discrete signal"而不是"必须按顺序生成的sequence"**，然后用world model (goal image) 作为refinement的anchor。赌赢了，至少在LIBERO/CALVIN这两个benchmark上。

Andrej，你的[Lesson in the back of "nanoChatGPT"](https://github.com/karpathy/LLM101n)和[MuZero讲解](https://www.youtube.com/watch?v=dx0Hb1X7fm4)其实早就暗示过：**任何signal，只要你找到对的factorization和inductive bias，都好学**。MMaDA-VLA就是把action的factorization从"autoregressive sequence"换成了"discrete diffusion set"，inductive bias对齐了。
