---
source_pdf: UniCorn.pdf
paper_sha256: 5476c9cce560becdeecbdd4f1a2dfaa275a61b0f9bef448df07fb990e34e1d75
processed_at: '2026-08-12T19:22:43-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# UniCorn 用人话说: 模型自己当自己的老师

Andrej，我用最直白的话重讲一遍。核心就一句话：**模型心里其实知道什么是好图，但手上画不出来——那就让心去教手。**

## 故事开头: 一个尴尬的病人

想象一个中风病人，医生给他看一张猫的图片，他能准确说"这是只橘猫，坐在桌上，左边有个水杯"。但你让他画一只猫，他画得像坨墨迹。神经科学叫这个 **Conduction Aphasia**——大脑的"理解区"和"表达区"之间的桥断了。信号进得来，出不去。

paper 发现 UMM（比如 BAGEL）就是这种病人。你给它一张图问"这图好不好？哪里有问题？"，它能答得头头是道，甚至接近 GPT-4 水平（Fig. 3）。但让它自己根据 prompt 生成图，却一塌糊涂。**理解能力是 silent passenger，搭了便车，没碰方向盘。**

UniCorn 就是要把这位 passenger 拽到驾驶座上，让"知道"反过来 supervise "会做"。

参考: BAGEL paper https://arxiv.org/abs/2505.14683

## 问题怎么 framed

paper 把 UMM 写成一个 policy $\pi_\theta$，输入输出都是交错的 text 和 image token 序列：

$$X = (x_1, \ldots, x_N), \quad x_n \in T \cup I$$

- $T$: text token 集合
- $I$: image token 集合
- $x_n$: 第 n 个 token，可能是字也可能是图像 patch

"Unified" 的意思就是 text 和 image 走同一套 backbone，权重共享。这个公式没啥 fancy 的，就是个 notation，但它告诉你一件事：**I2T 和 T2I 是同一个模型的两面，参数交叠**。所以理解差了，生成也跟着差；理解强了，理论上能反过来拉生成一把。

## 核心思想: Self-Play + 把心里话练回去

整个 UniCorn 就干两件事：

### 第一步: 模型自己跟自己下棋（Self Multi-Agent Sampling）

同一个 UMM，通过不同 prompt 扮演三个角色：

**Proposer（出题的）**：生成 5000 条多样 T2I prompt，分 10 大类（物体、关系、空间、时间、文字、科学、肖像、风格、计数、常识）。用 in-context learning + 动态 seeding，让 prompt 多样性爆炸。

**Solver（做题的）**：每条 prompt 做 8 次 rollout（不同 random seed、不同 cfg_text_scale），得到 8 张候选图。这跟 DeepSeek-R1 的 rejection sampling 思路一致。

**Judge（打分的）**：针对每张图，按 rubric 给 0-10 分，还带 chain-of-thought reasoning。比如"这是只玻璃雕塑乌龟，红色花纹，黑色大理石底座，光线柔和……评分：10"。Judge 不请外人，就是 UMM 自己。

这里关键 insight：**既然 UMM 自己做 reward model 时接近 GPT-4 水平，那何必再请 GPT-4o 当老师？自己教自己就行**。这也避开了外部 teacher 的 entropy 太高、小模型 fit 不动的问题（Tab. 4 里 UniCorn* 用 Qwen3-VL-235B 反而 UniCycle 掉了 6.5 分，证明了这一点）。

参考: DeepSeek-R1 https://arxiv.org/abs/2501.12948 ; Self-Rewarding LLMs https://arxiv.org/abs/2401.10020

### 第二步: 把"心里想的"也练回去（Cognitive Pattern Reconstruction）

这一步是 paper 最 clever 的部分。传统 rejection sampling 只挑高分图训练，UniCorn 觉得"心里话"也是金矿，于是把 self-play 的中间产物重构成 4 类训练 data：

**G (Generation)**：Best-of-N 选出的高分图，直接 prompt→image 训练。这是基本功。

**C (Caption)**：拿最好的图 $I^*$，让模型反推原 prompt，即 $\pi_\theta(T|I^*)$。这是"反向理解"——你画的图你自己得能看懂、能描述回来。这是双向对齐的关键。

**J (Judgement)**：让模型预测 judge 的打分和 reasoning，即 $\pi_\theta(J|T,I)$。把"什么是好图"这个判据内化进参数。相当于给 generator 装个内置裁判。

**R (Reflection)**：同一 prompt 找一张高分 $I^*$ 和一张低分 $I_{lose}$，让模型从低分图加 judge reasoning "修正"出高分图，即 $\pi_\theta(I^*|T, I_{lose}, J)$。这是显式学"怎么改错"。

数据 mix 是 5k G + 5k C + 3k J + 1k R，总共 14k 样本。8×H800 跑 7 小时，600 steps。**比一顿饭时间还短**。

intuition：这不是普通的 SFT，是把模型的 inner monologue（judge 怎么想的、怎么从坏图改到好图）也蒸馏回模型本身。Reflexion 在 LLM agent 里做过类似的事，UniCorn 把它搬到 image——image 没法显式"思考"，就用 judge 的 CoT 当 verbal thought。

参考: Reflexion https://arxiv.org/abs/2303.11366

## 理论: 为什么这四类 data 必须一起训

paper 在 Appendix C 给了三个数学论证，我用大白话讲：

### 论证 1: 为什么必须加 Caption（Mutual Information）

互信息公式：

$$MI(I; T) = H(I) - H(I|T) = H(T) - H(T|I)$$

- $MI$: image 和 text 共享的信息量
- $H(I|T)$: 给了 text，image 还剩多少不确定性（生成方向）
- $H(T|I)$: 给了 image，text 还剩多少不确定性（理解方向）

如果你只训 generation（最小化 $H(I|T)$），对 $H(T|I)$ **完全没有 signal**。结果就是理解能力萎缩——这就是 Tab. 2 里 w.o. CJR 时 MME-P 从 1685 暴跌到 311 的原因。模型忘了怎么读图，只记得怎么画图。

Caption 数据强制 $p(I|T)p(T) = p(T|I)p(I)$ 这个 Bayes consistency，两边一起锁住。这跟 VAE 的双向 ELBO、CycleGAN 的 cycle consistency loss 一个道理——双向对齐才能学到 shared representation。

参考: CycleGAN https://arxiv.org/abs/1703.10593

### 论证 2: 为什么必须加 Judgement

把目标 distribution 扩展到包含 judgement $J$：

$$\pi_\theta(I, T, J) = \pi_\theta(J|I, T) \cdot \pi_\theta(I|T) \cdot \pi_\theta(T)$$

- $\pi_\theta(T)$: 语言模型 prior
- $\pi_\theta(I|T)$: T2I generator
- $\pi_\theta(J|I, T)$: 内置 reward model

如果你只训前两项，模型只会生成，但不知道自己生成得好不好。加 $\pi_\theta(J|I, T)$ 后，模型内部就有了"自我评价"能力，generator 就能朝"被自己认可"的方向收敛。这相当于把 reward model 蒸馏进 generator，但不像 RLHF 那样分离两个 model。

### 论证 3: 为什么必须加 Reflection

$$\pi_\theta(I^*|T, J) = \pi_\theta(I^*|I, T, J) \cdot \pi_\theta(I|T, J)$$

- $I^*$: 最优图
- $I$: suboptimal 图
- $\pi_\theta(I|T, J)$: 初始生成
- $\pi_\theta(I^*|I, T, J)$: "修正算子"

这个分解把"从坏到好"显式参数化为一个 transition operator。Reflection data 训练的就是这个修正算子。模型不只学"怎么画"，还学"画错了怎么改"。对 mode collapse 和复杂 prompt（计数、空间关系）特别有效。

### 总 loss

$$\mathcal{L}_{Unified} = \mathcal{L}_G + \mathcal{L}_C + \mathcal{L}_J + \mathcal{L}_R$$

四个 NLL 加起来。paper 证明这等价于最小化 joint distribution $\pi_*(T, I, J)$ 的 NLL，所以数学上是 well-grounded 的 multi-task SFT。

## 新 benchmark: UniCycle——逼模型自证清白

paper 还提了一个新 benchmark，叫 **UniCycle**，思路非常巧妙。

传统 T2I benchmark（TIIF、WISE、CompBench）只测"图生成得好不好"，容易 cherry-pick 或 overfit 某个评估器。UniCycle 强制闭环：

1. 给 prompt $T$，模型生成 image
2. 对这张图问一组 question $\{q_k\}$（"图里有几只青蛙？青蛙什么颜色？位置在哪？"）
3. 模型基于自己生成的图回答 $\hat{y}_k$
4. 外部 judge（Qwen3-235B）对比 $\hat{y}_k$ 和 reference answer，给分 $s_k$
5. 计算 Soft / Hard score

评分公式：

$$\operatorname{Soft}(T) = \frac{1}{|\mathscr{Q}(T)|} \sum_{k \in \mathscr{Q}(T)} s_k$$
$$\operatorname{Hard}(T) = \mathbb{1}[\forall k \in \mathscr{Q}(T), s_k = 1]$$

- $\mathscr{Q}(T)$: 针对 prompt $T$ 构造的所有问题
- $s_k$: 第 k 个问题的得分（text 类问题是 keyword 恢复率，非 text 类是 0/1）
- Soft: 平均分，给 partial credit
- Hard: 所有问题全对才算 1，严格

dataset 1401 个 instance，2968 个 question，MCQ / Yes-No / Open-ended 混合。设计细节很讲究：negation 任务用 Yes/No（避免歧义），spatial relation 用 MCQ（"left" vs "front-left" 不打架），color/counting 用 open-ended（保留难度）。

**UniCycle 最 critical 的发现**：Janus-Pro 在传统 T2I benchmark 上看着还行，但在 UniCycle 上 Hard score 只有 9.9（BAGEL 36.6，UniCorn 46.5）。这意味着 Janus-Pro 生成的图连它自己都看不懂——典型的 Conduction Aphasia 量化证据。

| Model | Hard | Soft |
|---|---|---|
| BAGEL | 36.6 | 58.2 |
| Show-o2 | 36.1 | 52.5 |
| Janus-Pro | 9.9 | 25.8 |
| UniCorn* (外部 judge) | 40.0 | 58.6 |
| **UniCorn** | **46.5** | **66.6** |

参考: TIIF https://arxiv.org/abs/2506.02161 ; Qwen3 https://arxiv.org/abs/2505.09388

## 实验结果: 小数据跑出大分数

**主表 (Tab. 1)**：

| Model | TIIF-S | TIIF-L | WISE | OneIG | CompBench | DPG | Geneval |
|---|---|---|---|---|---|---|---|
| FLUX.1-dev | 66.2 | 66.7 | 50.0 | 43.4 | 83.1 | 83.8 | 82 |
| Janus-Pro | 65.4 | 61.1 | 35.0 | 26.7 | 74.0 | 84.3 | 80 |
| BAGEL | 71.0 | 71.8 | 50.0 | 36.1 | 82.2 | 84.0 | 78 |
| **UniCorn** | **74.7** | **72.9** | **55.0** | **42.6** | **88.5** | **86.8** | 82 |
| Δ vs BAGEL | +3.7 | +1.1 | +5.0 | +6.5 | +6.3 | +2.8 | +4.0 |

几个亮点：
- **OneIG-EN Text 子项 +22.4**：文字渲染暴涨，说明 knowledge internalization 真的起作用
- **CompBench Numeracy +13.1**：数数能力大幅提升（"seven frogs on the lake" 那种）
- **DPG 86.8 超过 GPT-4o 86.2**：dense prompt 上反超闭源
- **WISE Physics +10**：世界知识类生成大幅提升

14k 数据、600 steps、7 小时，换来 SOTA。这跟 LLaMA-3 instruct tuning 用几十万条数据比，sample efficiency 高得离谱。

## Ablation: 四类 data 各自的角色

Tab. 2 把四种 data 拆开看：

| Setting | TIIF-S | MME-P | MMVP |
|---|---|---|---|
| Base | 71.0 | 1685.0 | 69.3 |
| Full | 74.7 | 1660.0 | 70.0 |
| w.o. CJR（只剩 G） | 72.3 | **311.0** | 7.10 |
| w.o. R | 73.8 | 1632.0 | 71.3 |
| w.o. J | 74.2 | 1542.0 | 65.3 |
| w.o. C | 74.5 | 1653.0 | 68.0 |
| w.o. G | 73.4 | 1669.0 | 70.0 |

**最 striking 的数字**：w.o. CJR 时 MME-P 从 1685 跌到 311。这就是只训 generation 的灾难——catastrophic forgetting 把理解能力直接抹掉。

- **w.o. R**：TIIF-R 掉 2.5，reflection 学到的"修正"能力对复杂 prompt 关键
- **w.o. J**：MME-C 从 696 → 478，judgement 通过 shared parameter 反向 regularize 理解
- **w.o. C**：MMVP 掉 2，caption 维持 visual grounding
- **w.o. G**：理解全保持但 TIIF 只 73.4，说明光靠 CJR 不能完全替代 generation 训练本身

intuition：四类 data 像蛋白质的不同 amino acid，缺一个都让模型 fold 错方向。G 是"会做"，CJR 是"知道自己做了什么、做得好不好、怎么改"。

## Scaling: 自己生成的 data 比老师蒸馏的香

Fig. 7 的 scaling 曲线：

- 1k 数据就超过 RecA
- 5k 数据超过 IRG（用 30k GPT-4o 蒸馏数据训练）和 DALL-E 3
- 8k / 10k / 20k 继续涨

**5k 自生成 > 30k GPT-4o 蒸馏**。这说明 self-play 的 sample efficiency 比外部 teacher distillation 高约 6 倍。

直觉解释：模型自己生成的 data 落在自己 reachable manifold 上，每条样本都在它的 learning capacity 之内。GPT-4o 蒸馏的 data 可能在 model 当前 capacity 之外，信号被浪费——就像让初中生硬背大学教材，不如让他自己刷题。

参考: IRG https://arxiv.org/abs/2509.06945 ; DALL-E 3 https://cdn.openai.com/papers/dalle-3.pdf

## 跨架构验证: 不挑 base model

Tab. 4 top 把 method 套到 Janus-Pro（pure autoregressive，跟 BAGEL 的 hybrid 不同）：

| Model | TIIF | WISE | OneIG-EN |
|---|---|---|---|
| Janus-Pro | 63.2 | 35.0 | 26.7 |
| +UniCorn | 65.9 (+2.7) | 42.0 (+7.0) | 31.4 (+4.7) |

WISE 涨 7 分最猛，说明 knowledge-driven generation 受益最大。这给我一个直觉：**understanding 是 knowledge 的容器，generation 是 knowledge 的出口。容器满了出口堵着，UniCorn 就是疏通出口**。这个 mechanism 不依赖具体架构。

## Self-Play 必要性: 不用外部强 judge

Tab. 4 bottom 比较了 self-play 和用外部强 judge（Qwen3-VL-235B）：

| Model | TIIF | UniCycle Hard |
|---|---|---|
| UniCorn (self-play) | 73.8 | **46.5** |
| UniCorn* (external 235B judge) | 74.4 (+0.6) | 40.0 (-6.5) |

**反直觉**：用更强外部 judge 只换来微小 generation 提升，但 UniCycle 大跌 6.5 分。

paper 解释：强 teacher entropy 太高，小 model fit 不动。但更深层原因是 self-play 让理解-生成共享同一 distribution——内部一致性更好。**自己的 bias 和自己的 bias 是 matched 的，生成时不需要 over-correct**。这跟 self-distillation 比 teacher distillation 更稳的现象同源。

## 跟其他 self-improving 方法对比

Tab. 15：

| Method | External Model Free | External Data Free |
|---|---|---|
| IRG | ✗ | ✗ (用 GPT-4o + Qwen2.5VL) |
| UniRL | ✓ | ✓ (但用 GPT-4o 做 reward) |
| SRUM | ✗ | ✗ (用 SAM3) |
| RecA | ✓ | ✗ (用 GPT-4o) |
| **UniCorn** | ✓ | ✓ |

UniCorn 是唯一完全自给自足的。这在实际部署上很重要——不需要调外部 API，不需要付费，不依赖第三方 model 的可用性。

参考: UniRL https://arxiv.org/abs/2505.23380 ; SRUM https://arxiv.org/abs/2510.12784 ; RecA https://arxiv.org/abs/2509.07295

## 一些可能的延伸联想

### 跟 AlphaZero 的关系

self-play + best-of-N + 自我评估——这是 AlphaZero 的核心模式。AlphaZero 用 MCTS 做 policy improvement，UniCorn 用 rejection sampling + reflection 做。区别：AlphaZero 的 reward 来自 game outcome（ground truth），UniCorn 的 reward 来自 model 自己 judge（proxy）。

这暗示一个潜在风险：**self-reward 可能 reward hacking**。如果 judge 本身偏好高饱和度，self-play 会把这个 bias 放大。paper 没深入讨论这点，但 Tab. 4 显示 self-play 比 external judge 更好——可能因为 self-play 的 bias 和 generator 的 bias matched，生成时不需要 over-correct，反而更协调。

参考: AlphaZero https://arxiv.org/abs/1712.01815

### 跟 RLHF / Constitutional AI 的关系

UniCorn 的 self-judge 类似 Constitutional AI 的 self-critique，但区别：
- CAI 在 preference pair 上做 DPO，UniCorn 直接 SFT
- CAI 的 critique 是改进 response，UniCorn 的 reflection 是直接生成更好 image
- UniCorn 没显式 reward model，把 reward 蒸馏回 generator + understanding

如果加 DPO 阶段（用 reflection 数据里 $I^*$ vs $I_{lose}$ 做 preference pair），可能进一步提升 sample efficiency。

参考: Constitutional AI https://arxiv.org/abs/2212.08073 ; DPO https://arxiv.org/abs/2305.18290

### 跟 CycleGAN / Dual Learning 的关系

caption = inverse mapping，generation = forward mapping，UniCycle = cycle consistency——这跟 Dual Learning for MT（Sennrich 2016）和 CycleGAN（Zhu 2017）的 cycle consistency loss 是同一思想脉络。区别：这里不 adversarial，而是 SFT + NLL。

参考: CycleGAN https://arxiv.org/abs/1703.10593 ; Dual Learning https://arxiv.org/abs/1610.07151

### 跟 "thinking while generating" 的关系

TwiG 和 T2I-R1 把 chain-of-thought 显式插入 generation。UniCorn 不显式插入，而是隐式通过 judgement / reflection 训练把 reasoning 蒸馏进 parameter。这类似 implicit CoT 方向（e.g. Quiet-STaR）。

参考: T2I-R1 https://arxiv.org/abs/2505.00703 ; TwiG https://arxiv.org/abs/2511.16671 ; Quiet-STaR https://arxiv.org/abs/2403.09629

## 几个 implementation 的 magic number

- 8 rollouts per prompt（DeepSeek-R1 风格）
- 最高分 < 7 的 group 整组丢弃
- LR 1e-5，600 steps（短训练）
- CE:MSE = 0.1:1（text loss 占 10%，image loss 占 90%）
- EMA 0.99
- Diffusion timestep shift 4.0
- Gen resolution (512, 1024)，Und resolution (378, 980)
- Max context 40k

CE:MSE = 0.1:1 这个比例关键。caption + judgement + reflection 主要是 text，weight 过高会 dominate 信号，过低又起不到作用。0.1 是经验最优。

## Limitations 和我的 follow-up 猜测

paper 自己承认：
1. **single-turn**：没有 multi-turn 自我改进循环，understanding metrics（MME/MMMU）没显著提升
2. **compute cost**：self-play 需要 rollout + judge，expensive

我额外看到：
3. **judge bias 未分析**：没实验验证 judge 是否有 systematic bias
4. **reflection 只做一对 $I^*/I_{lose}$**：可以扩展为 multi-step refinement 链
5. **understanding 没涨**：如果用 multi-turn，可能让 understanding 也涨，真正双向 co-evolution

**follow-up 方向猜测**：
- multi-turn 自我博弈让 understanding 也涨
- 把 reflection 扩成 iterative refinement 链（类似 OpenAI o1 style 长思考）
- DPO 阶段在 reflection pair 上做
- 用 UniCycle 做在线 RL 的 reward signal

参考: OpenAI o1 https://openai.com/o1/

## 一句话总结

UniCorn 把 UMM 当成一个"脑裂"大脑——一个 hemisphere 会看（understanding），另一个 hemisphere 会画（generation），但 corpus callosum 断了。Self-play + CPR 就是重新接上 corpus callosum，让两个 hemisphere 自己对话、自己教自己，7 小时学完。

**关键 insight 不是新 architecture，而是"用 model 自己的 hidden capability 作为 supervision signal"**。这个思想在 LLM 时代被反复验证（self-reward、self-play、self-distill），UniCorn 把它带进了 multimodal——而且证明了它比请外部老师还更香。

这给我一个更深的直觉：**pretraining 已经把世界知识塞进参数了，post-training 的任务不是"再教一遍"，而是"让模型把自己的 latent knowledge 显式化、可用化"**。这条路可能比疯狂堆数据、疯狂调参数更 sustainable。

参考: Self-Rewarding LLMs https://arxiv.org/abs/2401.10020 ; Absolute Zero https://arxiv.org/abs/2505.03335

---

# UniCorn: Self-Play 让 UMM 的 Understanding 反过来 Supervise Generation

Andrej, 这篇 paper 我用 build intuition 的方式讲。核心一句话: **current UMMs 自己能 judge 一张图好不好,但 generate 同一张图时却 fail——理解是 silent passenger,UniCorn 通过 self-play 把这个 passenger 拽到驾驶座上**。

## 1. 核心观察: Conduction Aphasia

paper 借用神经科学术语 "Conduction Aphasia" (传导性失语症) 命名这个现象。临床上这类病人能 **理解** 语言,但无法 **复述** 刚才听到的话——Wernicke 区和 Broca 区之间 arcuate fasciculus 断了。UMMs 一模一样: I2T 强、T2I 弱,Fig. 3 显示 BAGEL 在 MMRB2 / Omni-RewardBench 上做 reward model 时接近 GPT-4 水平,但自己生成图像时却差很多。

intuition 上,模型内部 representation 已经 encode 了 "什么是好图、什么是 prompt-image alignment",但 generation head 没拿到这个 signal。这让我想到 LLM 里 pretraining 和 instruction tuning 的 gap——你已经会,但你不知道你会的能被你自己用。

参考: BAGEL (Deng et al. 2025) https://arxiv.org/abs/2505.14683 ; GPT-4o system card https://arxiv.org/abs/2410.21276

## 2. 思想源头: Cognitive Symmetry 与 Plato's Cave

paper §3.1 用了 Blanco 的 "bi-logic" 和认知对称性作为 motivation。一个小孩看到 apple 能说 "apple",听到 "apple" 能想象 apple——双向 mapping 才算真懂。当前 UMM 只完成了 observation→concept (I2T),concept→appearance (T2I) 是断的。这种 framing 让我想到 VAE 的 ELBO 双向约束、以及 contrastive learning 里 bidirectional consistency 的传统。paper 把 AGI 比作 "escape Plato's cave": 不要只看 shadow,还要会反推 shadow 背后的 source。

## 3. 框架总览: Self Multi-Agent + CPR

UniCorn 整个 pipeline 拆成两个 stage:

**Stage 1: Self Multi-Agent Sampling** — 同一个 UMM 用不同 prompt 扮演三个 role:
- **Proposer** $\pi_\theta(T|T)$: 生成 diverse T2I prompts,分 10 个 category (Tab. 7),用 ICL + 动态 seeding 扩展
- **Solver** $\pi_\theta(I|T)$: 每个 prompt 做 8 rollouts (DeepSeek-R1 风格),不同 seed 和 cfg_text_scale
- **Judge** $\pi_\theta(T|T,I)$: 给 (prompt, image) 打 0-10 分,带 CoT rubrics,LLM-as-a-judge

**Stage 2: Cognitive Pattern Reconstruction (CPR)** — 把 raw interaction 重构成 4 类训练 data:
- **Generation (G)**: Best-of-N 高分 image
- **Caption (C)**: 用 best image 反推原 prompt $\pi_\theta(T|I^*)$
- **Judgement (J)**: 让模型预测 judge 的分数和 reasoning $\pi_\theta(J|T,I)$
- **Reflection (R)**: 给低分 image + judge,生成高分 image $\pi_\theta(I^*|T,I_{lose},J)$

最终 5k G + 5k C + 3k J + 1k R 一起 SFT。

### 为什么这个设计 build 我的 intuition

我觉得这相当于把 AlphaGo 的 self-play + ReMCTS 思路搬到 UMM 里,但加了关键一步: **cognitive replay**。不像 STaR / Reflexion 只用 final answer,UniCorn 把 inner monologue (judge 的 reasoning) 也作为训练 signal 蒸馏回模型本身。

参考: Reflexion (Shinn et al. 2023) https://arxiv.org/abs/2303.11366 ; Self-Rewarding LLMs (Yuan et al. 2024) https://arxiv.org/abs/2401.10020 ; DeepSeek-R1 https://arxiv.org/abs/2501.12948 ; Absolute Zero https://arxiv.org/abs/2505.03335

## 4. 形式化定义与公式逐项解释

### 4.1 UMM policy (公式 1)

$$X = (x_1, \ldots, x_N), \quad x_n \in T \cup I$$

- $X$: interleaved multimodal input sequence
- $x_n$: 第 n 个 token,可能来自 text vocabulary $T$ 或 image token set $I$
- $N$: sequence length
- $\pi_\theta$: 参数为 $\theta$ 的 policy,把 $X$ 映射到 output $Y = \pi_\theta(X)$

这一定义覆盖 I2T (understanding) 和 T2I (generation) 两种 case,关键是 $T \cup I$ 共享同一 backbone,这就是 "unified" 的意义。

### 4.2 UniCycle metric (公式 2)

$$\operatorname{Soft}(T) = \frac{1}{|\mathscr{Q}(T)|} \sum_{k \in \mathscr{Q}(T)} s_k$$
$$\operatorname{Hard}(T) = \mathbb{1}[\forall k \in \mathscr{Q}(T), s_k = 1]$$

- $T$: original prompt
- $\mathscr{Q}(T)$: 针对 prompt $T$ 构造的问题集合 (例如 "图里有几只青蛙?")
- $s_k$: judge 对第 $k$ 个问题的 binary score;对于 text 类问题,等于 correctly recovered keywords 的比例
- $\operatorname{Soft}(T)$: 平均 score, soft 指标
- $\operatorname{Hard}(T)$: indicator, 当且仅当所有问题都对才取 1,严格指标
- $\mathbb{1}[\cdot]$: indicator function

intuition: Soft 看 average 复原度,Hard 看严格 all-or-nothing。两个一起报能避免 partial credit 偏差。

### 4.3 Mutual Information 解释 caption 必要性 (公式 3-4)

$$MI(I; T) = H(I) - H(I|T) = H(T) - H(T|I)$$

- $MI(I; T)$: image 和 text 的互信息
- $H(I)$, $H(T)$: 边缘熵
- $H(I|T)$: 给定 text 时 image 的条件熵 (生成方向不确定性)
- $H(T|I)$: 给定 image 时 text 的条件熵 (理解方向不确定性)

单向训 $p(I|T)$ 只最小化 $H(I|T)$ 的上界,对 $H(T|I)$ 没有直接 signal——这就是 Tab. 2 里 **w.o. CJR 时 MME-P 从 1685 暴跌到 311** 的理论原因。

公式 4:
$$p(I, T) = p(I|T)p(T) = p(T|I)p(I), \quad (T,I) \sim \mathcal{D}_C$$

- $\mathcal{D}_C$: caption 数据集
- $p(T)$, $p(I)$: 由 dataset 和 model 共同决定的 prior

caption data 强制双向一致,即 Bayes consistency $p(I|T)p(T) = p(T|I)p(I)$,这给我一种 VAE 双向 ELBO 的感觉。

### 4.4 Joint distribution 分解 (公式 5)

$$\pi_\theta(I, T, J) = \pi_\theta(J|I, T) \cdot \pi_\theta(I|T) \cdot \pi_\theta(T)$$

- $\pi_\theta(T)$: text prior (语言模型)
- $\pi_\theta(I|T)$: text-to-image generation
- $\pi_\theta(J|I, T)$: judgement / reward model

这是 chain rule 的一个特定 ordering,告诉你 UMM = LM + generator + reward 三件事的乘积。**reward model 内置**就是 UniCorn 不需要外部 SAM / GPT-4o 的原因。

### 4.5 Reflection trajectory (公式 6)

$$\pi_\theta(I^*|T, J) = \pi_\theta(I^*|I, T, J) \cdot \pi_\theta(I|T, J)$$

- $I^*$: 最佳 image
- $I$: suboptimal image (exploration 时采的)
- $\pi_\theta(I|T, J)$: 初始生成分布 (受 judgement 引导)
- $\pi_\theta(I^*|I, T, J)$: refinement / correction operator

这个分解把 "改进" 显式参数化为一个 transition operator,Reflexion 在 LLM 里做的就是这个,这里搬到 image。

### 4.6 总 loss (公式 7)

$$\mathcal{L}_{Unified} = \mathcal{L}_G + \mathcal{L}_C + \mathcal{L}_J + \mathcal{L}_R$$

各分量:
$$\mathcal{L}_G = -\mathbb{E}_{(I^*, T) \sim \mathcal{D}_{bon}}[\log \pi_\theta(I^*|T)]$$
$$\mathcal{L}_C = -\mathbb{E}_{(T, I^*) \sim \mathcal{D}_C}[\log \pi_\theta(T|I^*)]$$
$$\mathcal{L}_J = -\mathbb{E}_{(I, T, J) \sim \mathcal{D}_J}[\log \pi_\theta(J|I, T)]$$
$$\mathcal{L}_R = -\mathbb{E}_{(I^*, I, T, J) \sim \mathcal{D}_R}[\log \pi_\theta(I^*|I, T, J)]$$

- $\mathcal{D}_{bon}$: Best-of-N 采样得到的 generation 数据集
- $\mathcal{D}_C, \mathcal{D}_J, \mathcal{D}_R$: caption / judgement / reflection 数据集
- 所有 loss 都是 NLL 形式,统一到 SFT 框架

intuition: 整个 framework 本质是用 NLL 把 4 个不同 conditional 都对齐到同一个 $\theta$,所以是 multi-task SFT 的特殊形式。论文证明 (App. C) 这等价于最小化 joint distribution 的 NLL,因此理论上保证 auxiliary understanding 任务优化 final unified objective。

## 5. UniCycle Benchmark: Text → Image → Text 闭环

paper 提了一个新 benchmark,我觉得这是论文最 clever 的部分之一。传统 benchmark (TIIF, WISE, CompBench) 单独评估 T2I 或 I2T,容易 cherry-pick 或 over-fit。UniCycle 强制 Text → Image → Text 闭环:

1. 给 prompt $T$,model 生成 image
2. 对生成的 image 问一组 question $\{q_k\}$ (covering 10+ categories)
3. model 基于 image 回答 $\hat{y}_k$
4. 外部 judge (Qwen3-235B) 对比 $\hat{y}_k$ 和 reference $a_k$,给 score $s_k$
5. 计算 Soft / Hard score

dataset: 1401 个 TIIF-style instances,2968 个 question,分布如下 (Tab. 16):
- MCQ: 1067 (35.95%)
- Yes/No: 200 (6.74%)
- Open-ended: 1701 (57.31%)

设计细节: negation 任务用 Yes/No (避免歧义), spatial relation 用 MCQ ("left" vs "front-left" 歧义), color/counting 用 open-ended (保留 difficulty)。这种 question-type allocation 显示作者对 evaluation bias 有成熟考量。

结果 (Tab. 3 + Tab. 17):
| Model | Hard | Soft |
|---|---|---|
| BAGEL | 36.6 | 58.2 |
| Show-o2 | 36.1 | 52.5 |
| Janus-Pro | 9.9 | 25.8 |
| UniCorn* (Qwen3-VL-235B judge 外部) | 40.0 | 58.6 |
| **UniCorn** | **46.5** | **66.6** |

Janus-Pro 在 UniCycle 上惨败 (9.9),说明它的 generation 和 self-understanding 是脱节的——它生成的图自己都看不懂。这正是 Conduction Aphasia 的量化证据。

参考: TIIF https://arxiv.org/abs/2506.02161

## 6. 实验设置

**Base model**: BAGEL (understanding 和 generation 部分解耦的 hybrid UMM)
**Training**: 8× H800, 600 steps, LR $1\times10^{-5}$ constant, AdamW ($\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-15}$), warmup 50, gradient clip 1.0, EMA 0.99, CE:MSE = 0.1:1, max context 40k, gen resolution (512,1024), und resolution (378,980), diffusion timestep shift 4.0

**Data mixture**: 5k G + 5k C + 3k J + 1k R = 14k 总样本

**Cost**: 约 7 小时,极轻量级 post-training。这让我想到 LLaMA-3 instruct tuning 也是相对小的数据量级——这里更夸张,14k 样本就 SOTA。

**Judge filtering**: 最高分 < 7 的 sample group 整组丢弃

## 7. Main Results (Tab. 1)

| Model | TIIF-S | TIIF-L | WISE | OneIG | CompBench | DPG | Geneval |
|---|---|---|---|---|---|---|---|
| SD3 Medium | 64.8 | 64.8 | 42.0 | 42.8 | 84.3 | 84.1 | 74 |
| FLUX.1-dev | 66.2 | 66.7 | 50.0 | 43.4 | 83.1 | 83.8 | 82 |
| Janus-Pro | 65.4 | 61.1 | 35.0 | 26.7 | 74.0 | 84.3 | 80 |
| Show-o2 | 62.8 | 63.9 | 61.0 | 30.8 | 82.8 | 86.1 | 76 |
| BLIP3-o | 58.8 | 58.7 | 52.0 | 30.7 | 84.7 | 80.7 | 84 |
| BAGEL | 71.0 | 71.8 | 50.0 | 36.1 | 82.2 | 84.0 | 78 |
| **UniCorn** | **74.7** | **72.9** | **55.0** | **42.6** | **88.5** | **86.8** | 82 |
| Δ vs BAGEL | +3.7 | +1.1 | +5.0 | +6.5 | +6.3 | +2.8 | +4.0 |

关键 takeaway:
- **TIIF-S +3.7**: short prompt 跟随性提升明显
- **OneIG-EN Text +22.4**: 文字渲染暴涨,说明 knowledge internalization 起作用
- **CompBench Numeracy +13.1**: 数数能力大幅提升 (e.g. "seven frogs on the lake")
- **DPG 86.8 > GPT-4o 86.2**: 在 dense prompt 上超过 GPT-4o (虽然 understanding metrics 还差一截)
- **WISE +5.0**: world knowledge 类生成大幅提升 (Chemistry +4.0, Physics +10.0)

参考: WISE https://arxiv.org/abs/2503.07265 ; OneIG https://arxiv.org/abs/2506.07977 ; CompBench https://arxiv.org/abs/2407.08572 ; DPG https://arxiv.org/abs/2404.05999 ; GenEval https://arxiv.org/abs/2310.11513

## 8. Ablation: 三层证据

### 8.1 Data Pattern Ablation (Tab. 2)

| Setting | TIIF-S | TIIF-R | MME-P | MMB | MMVP |
|---|---|---|---|---|---|
| Base | 71.0 | 70.7 | 1685.0 | 84.6 | 69.3 |
| Full (Ours) | 74.7 | 78.4 | 1660.0 | 84.1 | 70.0 |
| w.o. CJR | 72.3 | 74.0 | **311.0** | 24.3 | 7.10 |
| w.o. R | 73.8 | 75.9 | 1632.0 | 84.2 | 71.3 |
| w.o. J | 74.2 | 74.8 | 1542.0 | 82.6 | 65.3 |
| w.o. C | 74.5 | 76.4 | 1653.0 | 84.3 | 68.0 |
| w.o. G | 73.4 | 72.3 | 1669.0 | 84.2 | 70.0 |

**最 striking 的是 w.o. CJR**: MME-P 从 1685 → 311,理解能力直接崩了。这是 catastrophic forgetting 的极端例子——纯 generation SFT 让 understanding head 完全 collapse。

- **w.o. R**: TIIF-R 掉 2.5,说明 reflection 学到 "fix bad image" 能力对复杂 prompt 重要
- **w.o. J**: MME-C 从 696 → 478,说明 judgement 不只是 generation 的事,它通过 shared parameter 反向 regularize understanding
- **w.o. C**: MME-P 还行,但 MMVP (视觉 perception) 从 70.0 → 68.0,说明 caption 维持了 visual grounding
- **w.o. G**: understanding 全保持但 TIIF-S 仅 73.4,说明 caption/judge/reflection 不能完全替代 generation 本身的训练

intuition: 4 种 data 像蛋白质里的不同 amino acid,缺一个都让模型 fold 错方向。Generation 是 "ability to act",CJR 是 "ability to know what you're acting on"。

### 8.2 Base Model Ablation (Tab. 4 top)

把 method 套到 Janus-Pro-7B (pure autoregressive,与 BAGEL 的 hybrid 不同):

| Model | TIIF | WISE | OneIG-EN |
|---|---|---|---|
| Janus-Pro | 63.2 | 35.0 | 26.7 |
| +UniCorn | 65.9 (+2.7) | 42.0 (+7.0) | 31.4 (+4.7) |

WISE 涨 7 点最猛,说明 knowledge-driven generation 受益最大——因为 understanding 内部化才能 unlock knowledge 用于 generation。这告诉我 method 不依赖 BAGEL 的 hybrid 结构,pure AR 上也 work。

参考: Janus-Pro https://arxiv.org/abs/2501.17811

### 8.3 Self-Play Necessity (Tab. 4 bottom + Q1)

UniCorn* = 用 Qwen3-VL-235B-A22B-Instruct 替换 self-play 来 construct data (外部强 judge):

| Model | TIIF | WISE | OneIG-EN | UniCycle Hard |
|---|---|---|---|---|
| UniCorn (self-play) | 73.8 | 55.0 | 42.6 | **46.5** |
| UniCorn* (external strong judge) | 74.4 (+0.6) | 54.0 (-1.0) | 44.9 (+2.3) | 40.0 (-6.5) |

**反直觉但重要**: 用更强外部 judge (Qwen3-VL-235B) 只换来微小 generation 提升,但 UniCycle 大跌 6.5 点。

paper 给的解释: 强 teacher 的 entropy 太大,小 model 难 fit,且 self-play 让理解-生成共享同一 parameter space 的同一 distribution——内部一致性更好。这让我想到 "self-distillation 比 teacher distillation 更稳" 的现象。

## 9. Scaling Law (Fig. 7)

| Train data | TIIF |
|---|---|
| 1k | surpasses RecA |
| 5k | surpasses IRG (trained on 30k GPT-4o distilled data) and DALL-E 3 |
| 8k, 10k, 20k | continues to scale |

关键: **5k self-generated data > 30k GPT-4o distilled data (IRG)**。这说明 self-play 的 sample efficiency 比外部 teacher distillation 高 ~6 倍。也验证 paper 的 "favorable scaling regime" 说法。

直觉: 自己生成的 data 处于 model 自己 reachable manifold 上,而 GPT-4o 的 data 可能在 model 当前 capacity 之外,distillation 信号被浪费。

参考: IRG (Interleaving Reasoning) https://arxiv.org/abs/2509.06945 ; DALL-E 3 https://cdn.openai.com/papers/dalle-3.pdf

## 10. CPR 数据细节 (Tab. 7 + Tab. 8)

10 个 prompt category + rubric,覆盖:
- General Object (形状/颜色/纹理)
- Object Relations (action/interaction/negation)
- General Knowledge (节日/名人/宗教)
- Spatio Reasoning (2D/3D/occlusion)
- Temporal Reasoning (同步/时序变化)
- Text Rendering (海报/手写)
- Natural Science (生物解剖/物理化学)
- Portrait (close-up/half-body/full-body)
- Stylization (anime/oil painting)
- Counting (precise number)

每个 category 有专门的 judgement rubric (Tab. 7 第三列),judge 用 rubric + CoT 出 0-10 分。Reflection 的 trick: 同一 prompt 取最高分 $I^*$ 和低分 $I_{lose}$,训练 $(T, I_{lose}, J) \to I^*$ 让 model 学 "修正"。

Tab. 8 给了具体训练 example:
- Generation: 直接 prompt→image
- Caption: "Type1: Infer the image-generation prompt from the picture" → original prompt
- Judgement: "Judge the image quality based on the generation prompt" → JSON {analysis, score}
- Reflection: 同 prompt + bad image + judge → good image

## 11. 与相关方法对比 (Tab. 15)

| Method | External Model Free | External Data Free | External Model | Hyperparams |
|---|---|---|---|---|
| IRG | ✗ | ✗ | GPT-4o + Qwen2.5VL | 0 |
| UniRL | ✓ | ✓ | GPT-4o | 1 |
| SRUM | ✗ | ✗ | SAM3 | 1 |
| RecA | ✓ | ✗ | GPT-4o | 3 |
| **UniCorn** | ✓ | ✓ | - | 0 |

UniCorn 是唯一同时 **不依赖外部 model** 和 **不依赖外部 data** 的方法,且 hyperparameter tuning 最少。这是 "fully self-contained" 的真正含义。

参考: UniRL https://arxiv.org/abs/2505.23380 ; SRUM https://arxiv.org/abs/2510.12784 ; RecA https://arxiv.org/abs/2509.07295

## 12. 我对 method 的几点深度联想

### 12.1 与 RLHF / RLAIF 的关系

UniCorn 的 self-judge 类似 Constitutional AI 的 self-critique,但区别在于:
- CAI 在 preference pair 上做 DPO,UniCorn 直接 SFT
- CAI 的 critique 是改进 response,UniCorn 的 reflection 是直接生成更好 image
- UniCorn 没显式 reward model,而是把 reward 蒸馏回 generator + understanding

如果加 DPO 阶段 (用 reflection 数据里 $I^*$ vs $I_{lose}$),可能进一步提升。

参考: Constitutional AI https://arxiv.org/abs/2212.08073

### 12.2 与 STaR / V-STaR / Reflexion 的关系

Reflexion 在 LLM agent 上做 verbal RL,UniCorn 把这个搬到 image generation——但 image 是高维 continuous,没有语言那种 discrete "thought" 中间状态。UniCorn 用 judgement 的 CoT reasoning 充当 image 的 "verbal thought",这是关键 insight。

### 12.3 与 CycleGAN / Dual Learning 的关系

caption = inverse mapping,generation = forward mapping,UniCycle = cycle consistency——这跟 Dual Learning for MT (Sennrich 2016) 和 CycleGAN (Zhu 2017) 的 cycle consistency loss 是同一思想脉络。区别: 这里不是 adversarial,而是 SFT + NLL。

参考: CycleGAN https://arxiv.org/abs/1703.10593 ; Dual Learning https://arxiv.org/abs/1610.07151

### 12.4 与 AlphaGo / AlphaZero 的关系

self-play + best-of-N + 自我评估——这是 AlphaZero 的核心模式。AlphaZero 用 MCTS 做 policy improvement,UniCorn 用 rejection sampling (Best-of-N) + reflection 做。区别: AlphaZero 的 reward 来自 game outcome (ground truth),UniCorn 的 reward 来自 model 自己 judge (proxy)。

这暗示一个潜在问题: **self-reward 可能 reward hacking**。如果 judge 本身有 bias (例如偏好高饱和度),self-play 会把这个 bias 放大。Paper 没深入讨论这个,但 Tab. 4 显示 self-play 比 external judge 更好——可能因为 self-play 的 bias 和 generator 的 bias 是 matched 的,生成时不需要 over-correct。

参考: AlphaZero https://arxiv.org/abs/1712.01815

### 12.5 关于 mode collapse 的解决

§3.3.2 提到 "directly optimizing this cross-domain alignment remains stochastic and inefficient, often leading to mode collapse"。w.o. CJR 时 MME-P 311 就是 mode collapse 的征兆。CPR 通过三个 auxiliary task (C, J, R) 作为 regularizer,把 generation 拉回 manifold。这本质是 **multi-task regularization 防止 single-task overfitting**——一个经典技巧。

### 12.6 与 LLM "thinking while generating" 的关系

TwiG (Guo et al. 2025) 和 T2I-R1 (Jiang et al. 2025) 把 chain-of-thought 显式插入 generation。UniCorn 不显式插入,而是隐式通过 judgement / reflection 训练把 reasoning 蒸馏进 parameter。这类似 implicit CoT 的方向 (e.g. Quiet-STaR)。

参考: T2I-R1 https://arxiv.org/abs/2505.00703 ; TwiG https://arxiv.org/abs/2511.16671 ; Quiet-STaR https://arxiv.org/abs/2403.09629

## 13. Limitations

paper 自己承认:
1. **single-turn**: 没有 multi-turn 自我改进循环,understanding metric (MME/MMMU) 没显著提升
2. **compute cost**: self-play 需要做 rollout + judge,expensive

我额外看到:
3. **judge bias 未分析**: 没实验验证 judge 是否有 systematic bias
4. **reflection 只做一对 $I^*/I_{lose}$**: 可以扩展为 multi-step refinement
5. **understanding 没提升**: 这是 limitation 1 的具体表现。如果用 multi-turn,可能让 understanding 也涨,真正双向 co-evolution

## 14. 实现细节的几个 magic number

- 8 rollouts per prompt (DeepSeek-R1 风格)
- 最高分 < 7 的 group 整组丢弃
- LR 1e-5, 600 steps (短训练)
- CE:MSE = 0.1:1 (text loss 占 10%, image loss 占 90%)
- EMA 0.99
- Diffusion timestep shift 4.0
- Gen resolution (512, 1024), Und resolution (378, 980)
- Max context 40k

CE:MSE = 0.1:1 这个比例很关键,因为 caption + judgement + reflection 主要是 text,如果 weight 过高会 dominate 信号,过低又起不到作用。0.1 是经验最优。

## 15. 我整体的评价

paper 的 **核心贡献** 不是 SOTA 数字,而是 3 个 framing:

1. **Conduction Aphasia 的 framing**: 给 UMM 的 understanding-generation gap 起了个好名字,这会让后续研究有共同语言
2. **Self-play on single UMM**: 把 AlphaGo-style self-play + Reflexion-style reflection 嫁接到 unified multimodal,工程上极简 (14k data, 600 steps, 7 hours)
3. **UniCycle benchmark**: T→I→T 闭环评估是真正能区分 "假 SOTA" 和 "真理解" 的工具,Janus-Pro 在上面暴雷就是证据

**理论部分** 用 Mutual Information + Bayes + NLL 分解证明 auxiliary task 等价于优化 unified objective,数学干净但不算 novel——dual learning 早有类似分析。但它把这套理论具体到 UMM 的 4 类 data 上,清晰可操作。

**我对 follow-up 的猜测**:
- multi-turn 自我博弈会让 understanding 也涨
- 把 reflection 扩成 iterative refinement 链 (类似 OpenAI o1 style 长思考)
- DPO 阶段在 reflection pair 上做可能进一步提升 sample efficiency
- 用 UniCycle 做在线 RL 的 reward signal (而非 SFT)

参考: OpenAI o1 https://openai.com/o1/ ; DPO https://arxiv.org/abs/2305.18290

---

总结一句 build intuition 的话: UniCorn 把 UMM 当成 " schizophrenic 大脑"——一个 hemisphere 会看 (understanding),另一个 hemisphere 会画 (generation),但 corpus callosum 断了。Self-play + CPR 就是重新接上 corpus callosum,让两个 hemisphere 自己对话、自己教自己,7 小时学完。**关键不是 "用什么新 architecture",而是 "用 model 自己的 hidden capability 作为 supervision signal"**——这个思想在 LLM 时代被反复验证 (self-reward, self-play, self-distill),UniCorn 把它带进了 multimodal。
