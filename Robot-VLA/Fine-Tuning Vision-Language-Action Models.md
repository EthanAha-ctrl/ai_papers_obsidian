---
source_pdf: Fine-Tuning Vision-Language-Action Models.pdf
paper_sha256: b860aa1206b6cfb0ce8be177f961379dd6a133d52cc74ac346636e0f4952a596
processed_at: '2026-08-04T08:18:46-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# OpenVLA-OFT 用人话版

好嘞 Karpathy，咱换个画风，就像在 Tesla Autopilot 茶水间白板前随便聊那样。

---

## 这篇paper在讲啥？

一句话：**OpenVLA原版太慢太笨，作者研究了一下发现fine-tuning的recipe比想象中重要得多，搞出个OFT配方，把OpenVLA从"3Hz的智障"变成"77Hz的猛男"，还在LIBERO上刷到97.1% SOTA。**

具体点说，OpenVLA有仨毛病：

1. **慢得离谱**：生成一个action要0.24秒，控制频率4Hz。bimanual robot要25Hz，根本玩不转。
2. **动作精度被256-bin discretization削了**：把连续动作硬塞进256个桶里，精细操作时手抖。
3. **只预测单步action**：每一步都重新看图重新想，errors一路累积，长horizon任务直接崩。

作者就琢磨：这仨毛病怎么治？分别对应三个设计选择——decoding方式、action表示、loss函数。

---

## 三个关键改动

### 改动一：从"一个一个说"变成"一口气说完"

原OpenVLA的decoding方式是autoregressive，跟LLM生成text一模一样——先预测position_x，再预测position_y，再position_z... 7个dimension要7次forward pass串行跑。

OFT换成**parallel decoding**：一次性把所有action dimension全部predict出来。怎么做到的？把causal attention mask换成bidirectional，input用一堆"空placeholder embedding"（只有positional encoding不同），让模型一次看全所有位置然后输出。

直觉上，autoregressive应该更强大——能建模action dimension之间的conditional dependence。比如gripper close的时候wrist orientation通常会跟着变。parallel decoding假设各个dimension独立，理论上应该损失表达力。

**但实验发现根本没掉点**。

这个结果挺震撼的。我琢磨着，可能因为action chunking救了场——一次predict 8个timestep × 7 dimension = 56个outputs，模型在chunk内部通过bidirectional attention能学到temporal correlation。单步内的correlation可能确实underfit了，但chunk level够用。

另外parallel decoding配上action chunking简直是绝配。原来做chunking要K×D次串行pass（K=8, D=7就是56次），现在还是1次forward pass。Throughput直接26× speedup。

这个设计让我想到BERT vs GPT的关系。GPT适合生成式任务因为autoregressive天然order-aware，BERT适合理解类任务因为bidirectional能看全context。Robot action prediction更像是"给observation编个码然后map到action空间"，跟生成text那种"一个token一个token往外蹦"的任务本质不同。所以parallel decoding反而更合适。

### 改动二：从"猜桶号"变成"直接输出数字"

原OpenVLA把每个action dimension normalize到[-1, +1]然后均分256个bin。模型相当于做256-way classification。

问题：精度上限就是1/256 ≈ 0.004，对于fine-grained manipulation可能不够。比如你要把gripper精确移动到某个位置，误差0.4%可能就是抓不稳。

OFT换成continuous output：把language model decoder的output embedding layer换掉，接个4-layer MLP直接output连续action value。

实验结果：+5% absolute success rate提升。

这个改动其实挺自然的。VLM原本是设计来predict discrete token的，但action本质是连续物理量。硬把连续量discretize本质上是引入了quantization noise。continuous output让模型能用全部float32精度表达action。

更深层的intuition：discrete token prediction的loss是cross-entropy，对"差一个bin"和"差一百个bin"一视同仁（都是错）。而continuous regression（L1 loss）对距离敏感，差0.01和差0.5惩罚不同。这种"距离感知"的监督信号对manipulation更合理。

### 改动三：从"复杂的diffusion"变成"简单的L1回归"

这是最反直觉的发现。

Diffusion policy是这两年的明星方法，能建模multimodal action distribution（同一个observation可能有多种valid action）。训练时给action加noise让模型学denoise，推理时从纯noise开始一步步denoise出真实action。

L1 regression简单粗暴：模型直接output action value，loss是预测值和ground truth的L1距离（mean absolute error）。

**实验结果：L1和diffusion性能基本一样**（95.3% vs 95.4%）。

这让我重新思考robot learning里的multimodality到底有多重要。

Diffusion的卖点一直是"能采样多模态分布"。但实际robot demo dataset里，同一个state真的有那么多valid action吗？大多数情况下，给定一个observation和language instruction，正确的action是相当确定的——抓杯子就是往杯子方向伸手，没什么多模态可言。

所谓multimodality很多时候是**数据质量问题**的symptom：
- 不同人teleop策略不同 → 但本质上最优策略可能就一种
- 同一task有多个阶段，不同阶段动作分布不同 → 这其实是mixture不是multimodality
- 数据没过滤干净，包含失败尝试 → 这是noise不是mode

OFT用filtered dataset（去掉了unsuccessful demos），分布自然更unimodal。L1回归学到median mode，反而对noise robust。

L1的另一个好处：训练收敛快、推理快。Diffusion要50步denoise，L1一步到位。Table II里diffusion即使配PD&AC，throughput还是4.2Hz——因为50步串行denoise成了bottleneck。

减少denoising steps可以提速，但Table II显示T_test=1时success rate直接崩到0%。因为单步denoising时模型在预测noise而不是action，任务完全不对。

所以"简单方法+大模型"在fine-tuning场景下赢了"复杂方法+大模型"。这让我想到你以前说过的"bitter lesson"——简单-scalable的方法最终会赢。OFT某种程度上是这个lesson在robot fine-tuning上的印证。

---

## FiLM：language grounding的secret weapon

ALOHA实验暴露了一个尴尬现象：OpenVLA fine-tune完，language following只有33% success rate——完全random chance！

原因挺有意思：wrist camera总是拍到gripper，而gripper state和grasp action高度correlated。policy发现"看到gripper close就predict某个action"已经能fit训练数据，根本不学language。这是典型的spurious correlation shortcut。

FiLM（Feature-wise Linear Modulation）是解药。公式：

$$\text{FiLM}(\mathbf{F} | \gamma, \beta) = \hat{\mathbf{F}} = (1 + \gamma) \odot \mathbf{F} + \beta$$

人话翻译：拿language embedding的mean，project成两个vector $\gamma$ 和 $\beta$，然后用它们对visual feature做affine变换——scale一下再加个shift。

$(1 + \gamma)$ 这个设计很巧。初始化时 $\gamma \approx 0$，所以 $(1 + \gamma) \approx 1$，visual feature几乎不变，不破坏pretrained representation。训练过程中 $\gamma$ 慢慢学到一个非零值，开始modulate。

关键implementation detail：$\gamma$ 和 $\beta$ 是per-channel的（$D_{ViT}$维），不是per-patch的。也就是说每个hidden dimension有一个scale/shift，对所有spatial location共享。

为什么per-patch不行？因为per-patch相当于让language控制每个pixel location，太严格，容易overfit。Per-channel相当于让language说"多关注texture还是shape"，让模型自己学where。这模拟了CNN里FiLM对整个feature map做global modulation的behavior。

加了FiLM后language following从33%飙到90%+。Figure 5的ablation直接证明了FiLM是不可或缺的。

更深层的intuition：VLM的language能力在pre-training时是explicit的（LM head直接predict token），但fine-tune成policy后language信息需要通过visual pathway影响action。这个"language→action"的pathway如果没有explicit bridge（如FiLM），模型可能走shortcut绕过language。FiLM相当于强制让language在visual processing早期就inject，让整个visual feature都language-conditioned。

LIBERO上为啥不需要FiLM？可能因为LIBERO的visual input比较clean（单视角、背景简单），spurious correlation少。ALOHA的多视角+wrist camera引入了大量shortcut机会，必须有FiLM救场。

---

## 架构变化总结

原OpenVLA的pipeline：
```
Image → SigLIP+DINOv2 → 256 patches → MLP projector → 
concat with language tokens → Llama-2 7B (causal attention) → 
output embedding → 256-way classification × 7 tokens
```

OFT+改完：
```
Multiple images → shared SigLIP+DINOv2 (with FiLM modulation) → 
proprio state → MLP projector → 
concat all → Llama-2 7B (bidirectional attention) → 
MLP action head → continuous action chunk (K×D values)
```

参数量变化（ALOHA, OFT+）：
- LoRA adapter: 111M
- Action head: 269M  
- Proprio projector: 17M
- FiLM projectors: 456M
- Total trainable: 853M（on 7.5B base model）

FiLM参数量竟然456M，比action head还大。因为每个transformer block都有独立projector，SigLIP和DINOv2各有多个block，projector要project language embedding到每个block的hidden dimension。

---

## LIBERO结果解读

最亮眼的数据点：

| Method | Avg Success Rate |
|--------|------------------|
| 原OpenVLA | 76.5% |
| + PD&AC | 90.2% |
| + PD&AC + continuous | 95.3% |
| + extra inputs (wrist cam, proprio) | 97.1% |

分解一下improvement来源：
- **PD&AC alone: +13.7%**（76.5→90.2）
- **Continuous action: +5.1%**（90.2→95.3）
- **Extra inputs: +1.8%**（95.3→97.1）

最大的win来自action chunking！这印证了temporal modeling比spatial precision更重要。

Long-horizon任务提升最夸张：53.7% → 94.5%（+40.8%）。这完全符合直觉——长horizon任务每一步error累积，chunking直接跳过中间replanning，compounding errors大幅减少。这个现象在DAgger那篇classic paper里有理论分析。

对比π0（94.2%），OFT用更简单的base model（OpenVLA 7B vs π0 3.3B但pretrained更多bimanual data）和更简单的training objective（L1 vs flow matching）竟然赢了。Recipe > raw model capacity/pretraining data。

---

## ALOHA结果：真正的考验

OpenVLA预训练时只见过single-arm data。ALOHA是bimanual + 25Hz + 3 cameras + 14-DOF proprio。Distribution shift巨大。

结果（Figure 4 aggregate score）：
- ACT: 最差
- Diffusion Policy: 中等，scalability有限
- RDT-1B: language following好但closed-loop反馈差（Figure 6显示bowl miss了还继续往空气里倒东西）
- π0: 整体最强baseline，能从failure恢复
- **OpenVLA-OFT+: 最强**

最有意思的是"scoop X into bowl"任务：
- OpenVLA-OFT+ (with FiLM): 100%
- OpenVLA-OFT+ (no FiLM): 35%

FiLM一来直接从35%到100%。这说明base model其实有能力，只是没FiLM时根本不attend language，纯靠visual shortcut。FiLM强制language injection后模型才"认真听指令"。

RDT-1B的失败模式很有教育意义（Figure 6）。它language following很好（知道该scoop raisins），但visual feedback loop坏了——bowl没放对位置它也继续按原计划倒，完全不看实际场景。这是diffusion policy的通病：一旦action chunk生成，open-loop执行，中间环境变化无法响应。

π0能从failure recover，说明flow matching + shorter effective chunk可能有更好的closed-loop特性。

OpenVLA-OFT+能赢可能因为：
1. L1 regression学的median mode比diffusion的sampled mode更"保守"
2. FiLM让language信号在visual processing早期就inject，决策更grounded
3. Action chunking + parallel decoding的throughput足够高（77.9Hz），可以频繁replan

---

## 我的几个intuition和猜测

**1. Parallel decoding为什么不掉点**

Autoregressive的theoretical advantage在conditional generation——predict $a_2$ 时能看到 $a_1$。但robot action prediction本质是mapping $\text{obs} \rightarrow \text{action}$，不是generation。给定obs，action分布是固定conditional distribution $p(a|o)$，各个dimension之间的correlation可以从obs直接infer，不需要通过sequential generation chain。

这跟LLM的text generation本质不同——text生成是"创作"，token间有严格causal order。Action prediction是"翻译"，obs和action之间是synchronous mapping。

**2. L1 = Diffusion的implication**

Robot learning community这两年把multimodality过度强调了。Diffusion的真正价值可能不在multimodality，而在：
- 训练stable（不像GAN有mode collapse问题）
- 能处理高维output（action chunk维度高）
- 用noise augmentation implicit regularize

L1能match diffusion说明：在fine-tuning场景下，数据filtered过、task定义明确、base model capacity充足，multimodality根本不是bottleneck。简单的regression就够。

这跟语言模型scaling的lesson类似——当你model足够大、data足够多，简单objective（next token prediction）就能emerge复杂能力。Diffusion的architectural sophistication可能在capacity不够时才必要。

**3. Pretraining vs Fine-tuning recipe**

Table XV显示去掉OpenVLA pretraining直接fine-tune Prismatic VLM，性能从97.1掉到91.9（-5.2%）。这gap不算大。

我猜未来方向：如果fine-tuning data足够大（比如10K+ demos per task），pretraining的importance会进一步降低。这意味着robot learning可能不需要"pretrain on internet-scale data"这个holy grail，只要有好的fine-tuning recipe + 中等scale task data就够。

但这只是猜测。pretraining带来的5%可能就是"floor effect"——从91到97的5%可能比从50到70的20%更难，因为是长尾。

**4. 为什么wrist camera是double-edged sword**

Wrist camera提供fine-grained spatial info（看到gripper和object的relative pose），对precision manipulation至关重要。但它也引入了"shortcut opportunity"——gripper state直接visible在image里，policy可以绕过language和high-level reasoning直接从gripper visual cue推action。

这就是为什么ALOHA上FiLM成了必需品，而LIBERO（无wrist cam）不需要。FiLM强制language在visual encoding早期就inject，让visual representation本身language-conditioned，policy必须同时用language和visual才能做决策。

这个发现对未来robot learning architecture设计有启发：**多视角input虽然info rich但容易引入shortcut，需要explicit mechanism强制cross-modal grounding**。

---

## 对你可能有启发的延伸思考

1. **Parallel decoding和autoregressive的trade-off**：在action space上parallel赢，但在"reasoning then act"的scenario下autoregressive可能更好（比如先在hidden state里"想"几步再输出action）。你的[Thinking in latent space](https://x.com/karpathy/status/1927457418079582516)想法可能和这个相关——parallel decoding本质上是zero-step thinking。

2. **L1 vs Diffusion的lesson**：在model capacity充足时，simple objective + large model > complex objective + small model。这和LLM的scaling law一脉相承。Diffusion可能更适合capacity-constrained场景或真正multimodal的任务（如autonomous driving的轨迹预测）。

3. **FiLM和attention的分工**：Attention是dynamic routing（每次推理重新计算），FiLM是static gain control（学到的固定modulation pattern）。两者互补。这和你提过的"[SSM vs Attention](https://arxiv.org/abs/2407.04620)"讨论有呼应——不同的information aggregation mechanism有不同的role。

4. **Recipe > Architecture**：这篇paper最强的message是——同一个base model，不同fine-tuning recipe能产生30%+的性能差异。这对你推动的"recipe engineering is the new architecture engineering"观点是个empirical support。

---

## 几个我觉得paper没讲透的问题

1. **Action chunking的开环执行**：K=25在25Hz下意味着1秒open-loop。dynamic环境（如deformable object manipulation）下这可能出问题。Paper没讨论closed-loop chunking（如[Bidirectional Decoding](https://arxiv.org/abs/2408.17355)）。

2. **L1在true multimodal任务上的失败**：Paper自己在Limitations里承认。但没给出"什么程度multimodality开始让L1崩"的empirical threshold。

3. **FiLM vs cross-attention**：FiLM是简单的affine modulation，cross-attention是更flexible的information routing。为什么用FiLM不用cross-attention？Paper没ablation。

4. **Pretraining data的domain gap**：OpenVLA在single-arm上pretrain，fine-tune到bimanual能work，说明transferability不错。但反过来，如果pretrain在bimanual上、fine-tune到single-arm会怎样？没实验。

5. **Chunk size K的选择**：K=8 for LIBERO, K=25 for ALOHA。这个选择是tuned还是arbitrary？K的sweet spot和task dynamics有关吗？没分析。

---

## 最后的take-away

这篇paper给robot learning community几个concrete lesson：

1. **Parallel decoding + action chunking是VLA的enabling技术**，让大模型VLA从"demo only"进入"deployable"。
2. **Simple L1 regression在fine-tuning场景下够用**，不需要diffusion的复杂度。这对实际部署是好消息——simpler stack更容易debug和maintain。
3. **Language grounding需要explicit mechanism**，不能假设VLM的language ability自动transfer到action prediction。FiLM是个简单有效的解。
4. **Fine-tuning recipe比pretraining data coverage更重要**——OpenVLA没见过bimanual数据，用OFT fine-tune后赢了见过bimanual的π0和RDT-1B。

我的整体感觉：这篇paper不是architectural breakthrough，是engineering recipe breakthrough。它把VLA research的焦点从"设计更fancy的architecture"拉回到"systematically研究fine-tuning design space"。这种empirical rigour在robot learning field其实挺稀缺的。

对了，如果你对L1 vs Diffusion的equivalence有进一步intuition，我特别想听。我感觉这背后可能有个更deep的theoretical story关于"when does model capacity substitute for objective complexity"，可能和[Nakkiran et al.](https://arxiv.org/abs/2006.08395)的double descent那套framework有关，但还没想清楚。

随便聊的，欢迎拍砖！

---

# OpenVLA-OFT: Fine-Tuning VLA Models 详解

你好 Karpathy！这篇paper正好戳中你一直在思考的VLA model scaling & efficiency问题。让我深入讲讲。

## Background & Motivation

当前VLA (Vision-Language-Action) models如[OpenVLA](https://openvla.github.io/)、[RT-2](https://robotics-transformer2.github.io/)、[π0](https://www.physicalintelligence.company/blog/pi0)、[RDT-1B](https://rdt-1b.github.io/)已经展示了strong task performance和semantic generalization。但一个practical的问题是：**如何有效fine-tune到新robot setup**？

OpenVLA原始设计有几个致命问题：
1. **Autoregressive action generation太慢**：3-5 Hz，远低于bimanual robot需要的25-50+ Hz
2. **Discrete action tokens有精度损失**：256-bin discretization
3. **Single-timestep prediction**导致compounding errors（参考[DAgger](https://arxiv.org/abs/1011.0686)的classic analysis）

Paper的核心contribution是systematic study这三个设计维度，提出**OFT recipe**。

---

## 三个核心设计决策

### 1. Action Decoding: Autoregressive vs. Parallel

**Autoregressive decoding**（原OpenVLA）：
- 用causal attention mask，类似LLM next-token prediction
- 每个action token依赖前面所有tokens
- For D=7 dimensional action，需要7 sequential forward passes
- Latency: 0.2396 sec per action

**Parallel decoding**（OFT）：
- 用**bidirectional attention**（类似BERT/[MaskGIT](https://arxiv.org/abs/2202.04200)）
- Input是empty action embeddings（只有positional encoding不同，类似[ACT](https://tonyzhaozh.github.io/aloha/)的做法）
- 所有action tokens同时generate，single forward pass
- Latency降到0.0629 sec

**Action chunking的enabling**：
原OpenVLA做action chunking不现实——chunk size K=8意味着8×7=56 sequential passes。但用parallel decoding后，只需1次forward pass生成KD个actions。

Intuition上，parallel decoding放弃了autoregressive的expressivity（不能建模action token间的conditional dependency），但实验显示**没有性能下降**。这暗示robot actions的多模态性主要在不同initial state间，而不是同一state内不同dimension间。

### 2. Action Representation: Discrete vs. Continuous

**Discrete**（原OpenVLA）：
- 每个action dimension normalize到[-1, +1]
- Uniform discretize成256 bins
- VLM的output embedding layer直接复用，类似vocabulary tokens
- Cross-entropy loss

问题：discretization是lossy的，256 bins意味着精度≈1/128（约0.008）。对于需要精确grasping的任务有局限。

**Continuous**（OFT）：
- 用4-layer MLP action head（ReLU activation）替代output embedding layer
- 直接output continuous action values
- 配合L1 regression或diffusion training

实验显示continuous带来**+5% absolute improvement**。

### 3. Learning Objective: Next-Token vs. L1 vs. Diffusion

这是最有意思的部分。

**L1 regression**：
$$\mathcal{L}_{L1} = \frac{1}{KD} \sum_{k=1}^{K} \sum_{d=1}^{D} |a_{k,d}^{pred} - a_{k,d}^{gt}|$$

其中：
- $K$: action chunk size
- $D$: action dimensionality
- $a_{k,d}^{pred}$: predicted action at chunk position $k$, dimension $d$
- $a_{k,d}^{gt}$: ground-truth action

**Diffusion**（参考[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)）：
- DDPM forward process: $a_t = \sqrt{\bar{\alpha}_t} a_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
- Model预测noise $\epsilon$
- Inference时reverse diffusion，50 steps

**实验结论惊人**：L1 regression和diffusion在LIBERO上性能comparable（95.3% vs 95.4%）！

这背后的intuition我推测是：
- OpenVLA的7B backbone容量巨大，能express多模态action distribution
- 在fine-tuning数据规模下（500 demos），multimodality可能不是主要bottleneck
- L1 regression天然给median mode，对noise robust（这是[L1的well-known property](https://en.wikipedia.org/wiki/Least_absolute_deviations)）

但paper在Limitations也讨论：对于truly multimodal demonstrations，L1可能学median而无法sample alternative modes。

---

## FiLM: 关键的Language Grounding Trick

在ALOHA上发现一个有趣现象：原OpenVLA fine-tune后**language following基本是random chance**（33%）。原因是spurious correlations——policy latch onto visual features而非language。

**FiLM (Feature-wise Linear Modulation)** from [Perez et al. 2018](https://arxiv.org/abs/1709.07871)：

$$\text{FiLM}(\mathbf{F} | \gamma, \beta) = \hat{\mathbf{F}} = (1 + \gamma) \odot \mathbf{F} + \beta$$

变量解释：
- $\mathbf{F}$: visual feature map，shape是(num_patches, $D_{ViT}$)
- $\gamma$: scaling vector, $D_{ViT}$-dimensional
- $\beta$: shifting vector, $D_{ViT}$-dimensional  
- $\odot$: element-wise multiplication（broadcasting across patches）
- $(1 + \gamma)$: 关键设计——初始化时$\gamma \approx 0$，所以$(1+\gamma) \approx 1$，保持pretrained visual features不扰动

$\gamma$和$\beta$怎么来：
$$\gamma, \beta = f(\bar{x}_{lang}), h(\bar{x}_{lang})$$

其中$\bar{x}_{lang}$是language embedding的平均。

**最关键的implementation细节**：
$\gamma$和$\beta$是$D_{ViT}$维的vector，**不是per-patch的**。即每个hidden dimension有一个scale/shift，对所有patches共享。这模拟了CNN里FiLM对整个feature map做spatially-agnostic modulation。

如果改成per-patch modulation，language grounding反而变差。这说明FiLM的本质是**channel-wise gating**，让language信息selectively boost/repress某些visual feature channels。

FiLM插在SigLIP和DINOv2 vision transformer的每个block的self-attention之后、FFN之前。每个block有独立的projector。

---

## 架构图解析（Appendix A）

**Base OpenVLA**:
- Vision: SigLIP + DINOv2 dual encoder → 256 patches each → concat → 3-layer MLP projector → language embedding space
- Language: Llama-2 7B
- Output: 7 discrete action tokens (3 position + 3 orientation + 1 gripper)

**OpenVLA-OFT+ modifications**:
1. Multiple input images via shared SigLIP-DINOv2 backbone
2. Robot proprioceptive state → 2-layer MLP → language embedding space
3. Causal → bidirectional attention
4. Output layer → 4-layer MLP (ReLU) for continuous actions
5. Output K actions instead of 1
6. FiLM modules in both SigLIP and DINOv2 transformers

**Trainable parameters breakdown** (ALOHA, OFT+):
- LoRA adapter: 111M
- Action head: 269M
- Proprio projector: 17M
- FiLM projectors: 456M (!)
- Total: 853M trainable on 7.5B model

注意FiLM projectors的参数量很大——因为每个transformer block都有独立projector，且dual vision encoder。

---

## LIBERO实验数据深度分析

Table I的核心数据：

| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|-----|
| OpenVLA (original) | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| + PD&AC | 91.3 | 92.7 | 90.5 | 86.5 | 90.2 |
| + PD&AC, Cont-Diffusion | 96.9 | 98.1 | 95.5 | 91.1 | 95.4 |
| + PD&AC, Cont-L1 (OFT) | 96.2 | 98.3 | 96.2 | 90.7 | 95.3 |
| OFT + extra inputs | 97.6 | 98.4 | 97.9 | 94.5 | **97.1** |

**关键观察**：
1. **Long-horizon提升最大**：53.7 → 94.5 (+40.8%)。这验证了action chunking减少compounding errors的假设（参考[ROSS](https://arxiv.org/abs/1011.0686)理论）
2. **PD&AC本身带来+13.7%**，超过continuous actions的+5.1%。说明**temporal modeling比spatial precision更重要**
3. **L1 vs Diffusion**: 95.3 vs 95.4，基本相同。但L1 inference快100×以上

Table II的efficiency数据：

| Variant | Throughput (Hz) | Latency (sec) |
|---------|-----------------|---------------|
| OpenVLA | 4.2 | 0.2396 |
| + PD | 15.9 | 0.0629 |
| + PD&AC | 108.8 | 0.0735 |
| + PD&AC, Cont-L1 | 109.7 | 0.0729 |
| + PD&AC, Cont-Diffusion (50 steps) | 4.2 | 1.9070 |

**Diffusion的trap**：
Table II里diffusion即使加PD&AC，throughput也只有4.2 Hz——和原OpenVLA一样！原因是50 denoising steps串行执行。

可以减少denoising steps（用[DDIM](https://arxiv.org/abs/2010.02502)）：
- T_test=10: 19.3 Hz, SR=91.0%
- T_test=5: 35.1 Hz, SR=90.0%
- T_test=2: 80.3 Hz, SR=85.7%
- T_test=1: 109.4 Hz, SR=0.0% (!!!)

T_test=1时SR崩到0%——因为单步denoising无法处理diffusion的noise prediction任务，模型预测的是noise而不是action。

---

## ALOHA Real-World实验：真正的test

ALOHA setup极具挑战性：
- Bimanual (14-DOF)
- 25 Hz control
- 3 cameras (top + 2 wrist)
- **OpenVLA预训练时完全没见过bimanual数据**！

Tasks设计巧妙：
1. **fold shorts/shirt**: deformable object manipulation
2. **scoop X into bowl**: tool use + language following (3 ingredients)
3. **put X into pot**: long-horizon + OOD evaluation

Table III的throughput：
- OpenVLA original: 1.8 Hz (完全不可用)
- OpenVLA-OFT+: 77.9 Hz (43× speedup!)
- ACT: 432.8 Hz (但84M params)
- π0: 291.6 Hz (JAX实现，3.3B params)
- RDT-1B: 84.1 Hz (1.2B params)

虽然OpenVLA-OFT+不是最快，但考虑到7.5B参数量，77.9 Hz已经足够real-time。

**Language following ablation**（Figure 5）：
- With FiLM: ~90%+ success
- Without FiLM: 33% = random chance

FiLM的importance在ALOHA上比LIBERO明显得多。原因paper推测是：wrist cameras引入大量spurious correlations（gripper总在视野里，和grasp动作高度correlated），policy容易shortcut而不学language。

---

## Limitations & Open Questions

Paper诚实地指出：

1. **Multimodal demonstrations**: L1 regression学median mode，可能丢失alternative solutions。这对teleop收集的data尤其相关——不同人可能用不同strategy。

2. **Pretraining vs Fine-tuning**: OFT在fine-tuning上work，但pretraining是否也需要diffusion的expressivity？未知。

3. **Inconsistent language grounding**: LIBERO不需要FiLM，但ALOHA必需。这个discrepancy没完全解释清楚。

---

## 我对这篇paper的intuition building

几个值得深究的点：

**1. 为什么parallel decoding不损失performance？**

Autoregressive的theoretical advantage是能建模token间的conditional distribution: $p(a_1, ..., a_D) = \prod p(a_d | a_{<d})$。Parallel decoding假设independence: $p(a_1, ..., a_D) = \prod p(a_d | x)$。

但robot actions的dimensions通常是高度correlated的（比如gripper close和wrist orientation变化是coupled的）。所以理论上parallel应该损失。

我的推测是：**action chunking itself provides the correlation structure**。预测K个timesteps × D dimensions = KD个outputs同时generate，模型可以通过bidirectional attention在chunk内部建模correlations。Single-timestep的D个dimensions可能indeed underfit correlations，但chunk level已经足够。

**2. L1 vs Diffusion的equivalence很反直觉**

Diffusion的优势在multimodal distributions。但实验显示comparable performance。

可能的explanation：
- Fine-tuning data是filtered的（unsuccessful demos removed），分布相对unimodal
- 7B model的capacity足够大，能通过deterministic function approximation fit出complex mappings
- L1的median property实际上是个implicit regularizer，避免overfit到outlier demos

这让我想到一个更深的问题：**robot learning到底需不需要multimodal action distributions**？如果task定义良好、demos质量高，可能unimodal就够。Multimodality更多是suboptimal data的symptom。

**3. FiLM为什么这么有效？**

FiLM本质是conditional computation——language信息gating visual features。这和[Visual Prompting](https://arxiv.org/abs/2210.07393)、[CoCoOp](https://arxiv.org/abs/2203.05557)思路类似。

但FiLM的spatial-agnostic设计（channel-wise modulation）很关键。如果per-patch FiLM，相当于让language控制每个spatial location——这太strict，容易overfit。Channel-wise让language控制"what features to attend to"，让模型自己学习where。

这和attention的role不同：attention是dynamic routing，FiLM是static gain control。两者互补。

**4. 为什么pretrain-finetune gap这么大？**

Table XV显示，去掉OpenVLA pretraining（直接fine-tune Prismatic VLM），性能从97.1%掉到91.9%（-5.2%）。

这说明**vision-language pretraining provides useful inductive bias**。即使是L1 regression这种简单object，pretrained representations也比from-scratch学得好。

但5.2%的gap不算huge。这可能暗示：随着fine-tuning data增大，pretraining的重要性会下降。这是否意味着future work可以简化pretraining stage？

---

## 进一步reading

如果你想深入：

1. **[OpenVLA原paper](https://arxiv.org/abs/2406.09246)**: 理解base model
2. **[ACT](https://tonyzhaozh.github.io/aloha/)**: parallel decoding + chunking的origin
3. **[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)**: diffusion for robot control
4. **[π0](https://www.physicalintelligence.company/blog/pi0)**: flow matching VLA，OFT的主要competitor
5. **[RDT-1B](https://rdt-1b.github.io/)**: diffusion-based bimanual VLA
6. **[FAST](https://arxiv.org/abs/2501.09747)**: 另一个action tokenization改进
7. **[FiLM原paper](https://arxiv.org/abs/1709.07871)**: 理解conditional modulation
8. **[LIBERO benchmark](https://libero-project.github.io/)**: 标准sim评估
9. **[Open X-Embodiment](https://robotics-transformer-x.github.io/)**: VLA pretraining data
10. **[DROID dataset](https://droid-dataset.github.io/)**: large-scale manipulation data

---

## 总结

这篇paper的beauty在于：通过controlled ablation study，揭示VLA fine-tuning的**关键design choices远比fancy algorithms重要**。

三个take-aways：
1. **Parallel decoding + action chunking**是enabling technology，让VLA从"slow thinker"变成"fast reactor"
2. **Simple L1 regression**和complex diffusion相当，前提是base model capacity足够。这对deployment-friendly是好消息
3. **Language grounding需要explicit mechanism**（如FiLM），不能假设VLM的language ability automatically transfer到action prediction

这和你以前提的"Recipe matters more than architecture"观点很resonant——这篇paper本质上是个fine-tuning recipe paper，但insights足够deep。

你的intuition如何？我特别想听你对L1 vs Diffusion的equivalence的看法——这是不是暗示robot learning的multimodality被over-hyped了？
