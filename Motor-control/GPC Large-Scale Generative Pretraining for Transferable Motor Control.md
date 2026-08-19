---
source_pdf: GPC Large-Scale Generative Pretraining for Transferable Motor Control.pdf
paper_sha256: 00dc276b89d4329ee00c6b1e42f8d3b9a3b68a9139467b4d6e56633d186cf2a7
processed_at: '2026-08-19T09:42:07-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GPC 人话版

## 一句话总结

把LLM的"next-token prediction"套路搬到物理仿真人物控制上。先学会把动作编码成离散token，再用GPT-style transformer预测下一个token，最后用少量参数fine-tune到具体任务。

---

## 问题是什么

你有个虚拟人物，想让它像真人一样动。传统方法要么手工写controller（walk controller, run controller...），要么用RL对着mocap数据学。但都遇到一个尴尬：学会的skill不好复用，换个任务就得重训。

后来大家搞generative controller——先学一个motion的"先验分布"，下游任务从这个分布里挑合适的skill。但之前的generative controller都用continuous latent space（VAE、CVAE、diffusion），有三个老问题：

**第一个，mode collapse**。GAN式训练不稳定，policy容易塌缩到几个pattern。

**第二个，off-manifold drift**。连续空间太大，inference时sample出来的latent经常飘到训练数据流形外面。走路这种慢动作还好，但cartwheel、backflip这种高动态动作，latent稍微偏一点，人物就摔了。

**第三个，行为单调**。连续prior的inference往往是deterministic的，同一个任务反复做同一个动作，没有diversity。

paper里Table 9的数据很扎眼：强推一下，CVAE的survival rate只有3.1%，GPC有52.5%。这不是小差距，是数量级的差距。

---

## 核心idea

既然continuous latent有这些问题，那就用discrete token。这恰恰是LLM走过的路——从word2vec的continuous embedding，到BPE的discrete token + transformer。

GPC的三步走：

**第一步**：学一个tracking controller，把reference motion编码成离散token，再用token还原出action。

**第二步**：用GPT-style transformer学这些token的分布。给当前state，预测下一个要执行的skill token。

**第三步**：用CoLA（一种PEFT）把pretrained model adapt到具体任务，只加不到1%的参数。

---

## 第一步：FSQ怎么把动作变成token

这里有个关键选择：用什么方法做离散化？

之前的VQ-VAE要学一个codebook，训练时各种麻烦——codebook collapse、dead code、需要EMA update、需要auxiliary loss、需要定期重初始化没用的code。调参调到头秃。

FSQ的思路特别简单粗暴：不学codebook，直接用数学定义。

encoder输出一个40维vector，每一维经过tanh压到[-1,1]，乘4变[-4,4]，round到最近的整数。每一维有9个可能的值（-4到4），40维就是$9^{40}$个可能的code。这个codebook是隐式的，不需要存储，不需要更新，不会collapse。

梯度怎么回传？straight-through estimator——前向用rounding，反向直接把梯度透传到rounding之前的连续值。

实验结果（Table 1）显示FSQ比VQ-VAE好。Bones 680小时数据上MPJPE是34.90mm vs 37.92mm，AMASS 40小时数据上差距更大，44.43mm vs 59.28mm。原因是大规模数据天然填充了更多grid cell，FSQ的均匀grid反而成了优势。

---

## encoder-decoder的关键设计

这里有个很巧的asymmetry。

encoder只吃reference motion的未来window，不吃当前character state。decoder同时吃当前state和离散token。

为什么不让encoder也看state？因为如果encoder看了state，它会偷懒——把state信息直接塞进token，decoder就不需要学"如何在当前state下执行skill"了。token就变成了state的copy，不是skill的abstraction。

解耦之后，encoder学的是"要做什么动作"——pure skill identity。decoder学的是"在当前身体状态下怎么执行这个动作"。这给后面的generative controller提供了干净接口：transformer只需要预测skill identity，不用操心执行细节。

reference motion window用的是dilated indices（1, 2, 5, 7, 12, 18, 25帧），类似CNN的dilated convolution。compact representation里能塞进long-horizon信息。

---

## 端到端RL训练为什么重要

paper做了个ablation（Table 2）特别说明这点。

一种做法是先用监督学习在kinematic数据上训encoder，然后冻住encoder，只用RL训decoder。另一种是encoder和decoder一起用RL训。

结果差距巨大：冻住encoder的MPJPE是78.26mm，端到端RL的是34.90mm。差了一倍多。

原因很直觉：kinematic reconstruction只关心"token能不能还原reference motion"。但physical control关心的是"token能不能驱动一个有重力、有惯性、有contact physics的人物还原reference motion"。两者之间有巨大gap——PD controller的dynamics、actuation limit、contact、momentum都不是kinematic能capture的。

端到端RL让encoder学到的token representation对physics友好，这很关键。

---

## 第二步：Transformer怎么学token分布

### Token grouping的trick

FSQ输出40个token，每个token有9个值。直接对40个token做autoregressive，sequence length太长，self-attention cost是$O(40^2)$。

solution是把每5个consecutive token打包成一个super-token。40/5=8个grouped token，每个grouped token的vocab是$9^5=59049$。sequence length从40降到8。

这个trick类似LLM里从char-level到subword-level的升级。Table 3的ablation很说明问题：

- G=1（不group）：vocab只有9，几乎没有expressiveness，FPS只有25
- G=8：vocab 4.3×10^7，比GPT-2的50257还大，直接OOM
- G=5：sweet spot，APD最高（行为多样性最好），FPS 92

grouped token carry higher-level semantic information。每个grouped token代表5个原始token的组合，相当于一个motion的"subword unit"。

### Transformer架构

比较modest——6层，1024 dim，4 heads，4096 FFN dim。大约100M参数量级。motion的复杂度远低于自然语言，不需要LLM那么大。

训练就是标准teacher forcing + cross-entropy + label smoothing + EMA。inference用nucleus sampling top-p=0.9。

---

## 自回归怎么工作

每个timestep，transformer接收当前character state作为context token，然后autoregressively预测8个grouped token。

第1个token条件于state。第2个token条件于state + 第1个token。第3个token条件于state + 前2个token。以此类推。causal self-attention保证训练和inference一致。

8个token预测完后，送给frozen FSQ decoder，decode出66个DoF的target joint rotations，PD controller转成torque，驱动人物。

---

## Emergent behavior是最大的亮点

最让人兴奋的是Fig 5和Fig 6展示的emergent behavior。

你给character一个强推力，它摔了。然后它自动roll一下站起来。这个recovery behavior没有任何explicit reward，没有任何specialized training。它完全来自大规模数据的diversity + stochastic sampling。

机制是这样的：large-scale data里有各种fall recovery的motion clip。transformer学到了"摔倒后该做什么"的分布。stochastic sampling让模型在摔倒状态下能sample到recovery skill token。而continuous prior的deterministic inference做不到这点——latent一旦飘到off-manifold就回不来。

Table 9的数据印证了这点：无perturbation时GPC和CVAE的survival rate差不多（82% vs 79%）。但强push下，GPC 52.5%，CVAE 3.1%。差距完全拉开。

---

## 第三步：CoLA怎么高效fine-tune

### 先说LoRA家族的演进

LoRA是PEFT的基础idea：冻结pretrained weight $W_0$，加一个low-rank update $BA$。$B$和$A$是低秩矩阵，rank远小于原始维度。只训练$B$和$A$，参数量很小。

DoRA在LoRA基础上做了改进：把update分解成magnitude和direction。direction做explicit normalization，magnitude单独控制。优化更稳定。

CoLA在DoRA基础上又加了一层：在低秩空间里用FiLM做task conditioning。task observation通过MLP产生$\gamma(c)$和$\beta(c)$，在低秩空间做affine modulation。

公式看起来复杂，核心idea就一句话：task condition注入到低秩空间，而不是full feature space。这极大降低了conditioning的参数量。

### Nucleus sampling当exploration constraint

这是CoLA fine-tuning最关键的设计。

inference时，先取frozen base model的output distribution，取top-p=0.9的mass作为support set。然后adapted model的distribution在这个support set上renormalize、sample。

效果是：adapter想sample一个base model认为低概率的token，这个token直接被clip掉。adapter只能在base model的high-probability manifold里explore。

Fig 17的ablation说明了trade-off：top-p太小（0.5），收敛快但限制了skill set，return上限低。top-p太大（0.99），自由度太大导致jitter。top-p=0.9是sweet spot。

这个设计本质上把pretrained model当作"安全guardrail"——你可以explore，但不能explore到pretrained model认为不合理的区域。

### SFT的作用

有example motion时先做SFT，在example motion对应的token上minimize cross-entropy。这让adapter bias到正确的skill subset，避免RLFT在huge skill space里explore的低效。

但Table 4有个有趣发现：SFT的return反而比w/o SFT低（143 vs 230）。SFT让model更restricted，exploration少了。

那SFT的价值在哪？Fig 10/11说明了：SFT能让model保持crouched walking style。head height维持在0.8-1.3m。没SFT的model会做各种height的motion。

所以SFT适合"我要这个style"的task，不适合"纯粹maximize task reward"的task。

---

## Reward设计

tracking reward是5项的weighted sum：global joint position、global joint rotation、root height、joint velocity、joint angular velocity。

每项都是$\exp(\text{coeff} \times \text{error})$形式，error越小reward越接近1。

权重设计有直觉：global joint position权重最大（0.5）+ coeff最大（-100），因为它直接反映motion视觉相似度。root height单独强调（权重0.2 + coeff -100），因为root height错了整个motion都崩。

---

## 实验数据的几个关键takeaway

### FSQ vs VQ-VAE

大规模数据（Bones 680hr）上差距小，中等规模（AMASS 40hr）上差距大。说明大规模数据能缓解VQ-VAE的codebook collapse问题，但FSQ在任何scale都稳定。

### GPC vs CVAE robustness

无perturbation时差不多。perturbation越大GPC优势越明显。强push下CVAE几乎完全collapse（3.1%），GPC保持52.5%。这是discrete token + stochastic sampling的核心优势。

### Downstream task

GPC略好CVAE（94.2% vs 93.9% success rate），远好MaskedMimic（89.9%）。MaskedMimic的问题是训练分布和inference分布有shift——训练时condition在random masked future joint positions，inference时condition在fixed root height。

---

## Muon optimizer

细节但值得提：actor用Muon optimizer，critic用AdamW。

Muon是geometry-aware optimizer，对matrix-shaped参数做SVD-like分解，更新orthogonal component。比Adam的element-wise更新对矩阵参数更高效。只用Muon训练hidden weight matrices，低维参数和interface layer用AdamW保持稳定。

---

## FSQ latent space的structure

附录B.6做了个skill retrieval实验：用Sentence-BERT-style mean-pooling得到每个motion clip的global embedding，用L2 distance做retrieval。

coarse level（8类动作）P@1=0.72，fine level（20类细分风格）P@1=0.66。证明FSQ latent不是random assignment，相似skill的token在grid上靠近。

更有趣的是Fig 14：interpolate "punching"和"kicking"的latent code，能produce一个同时punch + kick的motion。latent space有linear compositionality。

---

## 跟其他工作的关系

### vs ASE / CALM / MaskedMimic

这些都是continuous latent space的generative controller。ASE用GAN，有mode collapse风险。MaskedMimic用masked inpainting，continuous latent。GPC用discrete token，immune to mode collapse和off-manifold drift。

### vs Decision Transformer / Trajectory Transformer

这些都是transformer + RL，但offline setting。Decision Transformer用continuous embedding，Trajectory Transformer用per-dimension binning。GPC是online RL + simulator，token是skill-level的，不是action-level的。每个grouped token对应一小段未来motion的skill identity，carry higher semantic meaning。

### vs TokenHSI

TokenHSI在固定predefined task set上训练，transfer到新task。GPC是unconditional generative prior，task-agnostic pretraining，PEFT用CoLA adapt。更灵活。

---

## Limitations

paper诚实列了几个：

只测了locomotion和简单HSI（platform、barrier）。没测manipulation、tool use、social interaction。

没有text conditioning。借鉴MotionGPT的设计加text encoder + cross-attention是obvious extension。

human-object interaction只测了静态环境。dynamic object interaction没测。

CoLA的rank=64可能限制expressiveness。

SFT的trade-off没充分讨论——SFT让return降低但improve style control。

---

## 我的直觉解读

### 为什么这个pipeline work

三个因素叠加：

**discrete token约束了distribution**。continuous latent space是$\mathbb{R}^d$，太大，off-manifold区域太多。discrete token把distribution约束在finite grid上，每个grid cell都是training data visit过的区域。inference时sample不会飘到未见过的区域。

**stochastic sampling提供了recovery能力**。deterministic inference一旦state偏离normal就回不来。stochastic sampling让模型在偏离状态下能"重新选择"skill——摔倒后sample到recovery token，而不是执着于继续走。

**大规模数据提供了skill diversity**。Bones 680小时数据覆盖了各种fall recovery、roll、get-up。transformer学到了这些skill的conditional distribution。perturbation来了，模型sample到合适的recovery skill。

### 为什么PEFT能work

nucleus sampling constraint是关键。它把adapter的exploration限制在base model的high-probability manifold内。adapter不能发明新skill，只能在pretrained skill set里做选择和组合。

这好比一个会各种武术的人学新任务——他不需要从头学走路，只需要学会在合适的时机用合适的招式。CoLA的FiLM conditioning就是"合适的时机"的signal。

### 跟LLM的类比

GPC整个pipeline跟LLM几乎一一对应：

- FSQ tokenization ↔ BPE tokenization
- tracking controller pretraining ↔ masked language modeling
- GPT-style transformer ↔ GPT
- nucleus sampling ↔ nucleus sampling
- CoLA PEFT ↔ LoRA/adapter
- SFT before RLFT ↔ instruction tuning before RLHF

这个类比不只是表面相似。它说明discrete token + autoregressive transformer是通用skill modeling范式，不限于language。

### 对未来的启示

text conditioning是obvious next step。加text encoder，通过CoLA的FiLM注入。让用户用自然语言控制人物动作。

real robot transfer可能受益于discrete action space。distributional shift更可控，action永远在data manifold内。

hierarchical control：high-level text → mid-level motor token → low-level PD torque。天然extension。

scale up transformer。目前只有6层1024 dim。scale到LLM size可能enable更复杂skill composition。

lifetime learning。把CoLA的task condition扩展成memory，让agent记住过去任务。接近meta-RL。

---

## 最后的直觉

GPC本质上是证明了：LLM的配方在motor control上work。discrete token约束distribution，transformer学conditional distribution，PEFT高效adapt，stochastic sampling提供robustness。

这不是incremental improvement，是paradigm transfer。就像ResNet把CNN的best practice推广到所有视觉任务一样，GPC把LLM的best practice推广到motor control。

影响的不只是character animation。robotics的motor control、game AI的NPC control、甚至protein conformational control都可能借鉴这个pipeline。

---

# GPC: 把LLM的Next-Token Prediction范式迁移到Physical Motor Control

这篇paper的核心论点相当ambitious：把physics-based character control的generative controller从continuous latent space（VAE、CVAE、diffusion）迁移到discrete token + autoregressive transformer的范式。这恰好是LLM走过的路径（word2vec continuous embedding → BPE discrete token + transformer）。下面我从直觉到技术细节层层拆解。

参考链接：
- Paper: https://arxiv.org/abs/2309.15505 (FSQ)
- https://arxiv.org/abs/2106.09685 (LoRA)
- https://arxiv.org/abs/2402.09353 (DoRA)
- https://arxiv.org/abs/1709.07871 (FiLM)
- https://arxiv.org/abs/1804.02717 (DeepMimic)
- https://arxiv.org/abs/1707.06347 (PPO)
- https://arxiv.org/abs/2106.01345 (Decision Transformer)
- https://arxiv.org/abs/1904.09751 (Nucleus sampling)
- https://arxiv.org/abs/1711.00937 (VQ-VAE)
- https://github.com/NVLabs/ProtoMotions/
- https://bones.studio/datasets
- https://kellerjordan.github.io/posts/muon (Muon optimizer)

---

## 1. Motivation：为什么需要Discrete Token？

Continuous latent space在motor control generative controller里有几个长期failure mode：

**Mode collapse**：GAN-style generative controller（如ASE https://arxiv.org/abs/2205.01906）训练不稳定，discriminator一旦dominate policy，policy会collapse到少数几个motion pattern。

**Off-manifold drift**：VAE/CVAE在inference时sample的latent z经常落在training data流形之外。对于highly dynamic motion（vault, cartwheel, flip），small off-manifold drift = physical failure（character摔倒）。Table 9里CVAE在强perturbation下survival rate只有3.1%就是这个问题的体现。

**Stochasticity缺失**：continuous prior inference往往是deterministic的（取mean），导致downstream task行为diversity低。Fig 8/9显示CVAE在同样task下反复执行相似motion，缺乏behavioral diversity。

Discrete token的好处：
- Distribution被严格约束在fixed grid上，codebook本身就是data manifold的离散化sample
- 天然支持autoregressive modeling，可以复用GPT的全部infra（causal mask, teacher forcing, top-p sampling）
- Stochastic sampling变得自然：每个token是一个categorical distribution，sample一个token序列等价于sample一段motion

---

## 2. 三阶段Pipeline总览

```
Stage 1: Skill Quantization (FSQ tracking controller, end-to-end RL)
    Input:  character state s_t + reference motion window ŝ_{t:t+h}
    Encoder: MLP → R^40
    FSQ:    round to 9 levels per dim → 9^40 implicit codebook
    Decoder: MLP → 66 DoF target joint rotations → PD controller → torque

Stage 2: Generative Controller (GPT-style transformer)
    Input:  character state s_t as context token
    Output: autoregressively predict 8 grouped tokens (vocab 9^5=59049)
    Training: cross-entropy + label smoothing + EMA
    Inference: top-p=0.9 nucleus sampling

Stage 3: Task Adaptation (CoLA)
    Frozen: pretrained generative controller
    Trainable: CoLA adapter layers (<1% params)
    Conditioning: task observation c → FiLM modulation in low-rank space
    Training: SFT (optional) → RLFT (PPO)
```

---

## 3. FSQ的数学细节

公式 (4)：
$$\hat{z}_t = \lfloor \lfloor \frac{L}{2} \rfloor \tanh(z_t) \rceil$$

变量逐个解释：
- $z_t \in \mathbb{R}^d$：encoder output，$d=40$（40个latent channel）
- $L=9$：每个channel的quantization level数
- $\tanh(z_t)$：把每个element压到$[-1, 1]$
- $\lfloor L/2 \rfloor = 4$：scale factor
- $4 \cdot \tanh(z_t) \in [-4, 4]$
- $\lfloor \cdot \rceil$：element-wise rounding to nearest integer，结果取值于$\{-4, -3, -2, -1, 0, 1, 2, 3, 4\}$，共9个level
- 隐式codebook大小 = $L^d = 9^{40} \approx 1.5 \times 10^{38}$

梯度回传：straight-through estimator (STE)。前向用rounding（不可微），反向直接把gradient pass-through到tanh之前的z_t。

**为什么FSQ比VQ-VAE好？**

VQ-VAE (公式2-3)：
$$k^* = \arg\min_k \|z - e_k\|_2$$
$$L_{VQ} = \|x - \hat{x}\|_2^2 + \|\text{sg}[\mathcal{E}(x)] - e_{k^*}\|_2^2 + \beta\|z - \text{sg}[e_{k^*}]\|_2^2$$

VQ-VAE需要：
1. Reconstruction loss $\|x - \hat{x}\|^2$
2. Codebook loss $\|\text{sg}[\mathcal{E}(x)] - e_{k^*}\|^2$（pull codebook entry to encoder output）
3. Commitment loss $\beta\|z - \text{sg}[e_{k^*}]\|^2$（pull encoder output to selected codebook entry）
4. EMA update for codebook
5. Dead code reinitialization

VQ-VAE常见failure mode：
- **Codebook collapse**：大部分code从来不激活
- **Low utilization**：只有少数code承担所有数据

FSQ完全不需要这些。Codebook是数学上预定义的均匀grid，没有任何learnable parameter。代价：FSQ的grid是均匀的，不像VQ-VAE那样适应数据分布。但paper实验显示在large-scale data下FSQ的utilization反而更高（Table 2: 82.15% vs 76.34%），因为large data自然填充了更多grid cell。

---

## 4. Encoder-Decoder架构的关键设计

Encoder：MLP [1024, 1024, 1024, 512, 256] + ReLU → R^40

Decoder：MLP [1024, 1024, 1024, 512, 256] + ReLU → R^{N_DoF}（66 DoF的target joint rotations）

**关键架构选择**：encoder只吃reference motion $\hat{s}_{t:t+h}$，不吃character state $s_t$。decoder同时吃$s_t$和离散code $\hat{z}_t$。

这个asymmetry是intentional的。如果encoder同时看$s_t$和$\hat{s}_{t:t+h}$，encoder会bypass decoder——直接把$s_t$信息塞进code，让decoder变得多余。把encoder和decoder的input解耦后：
- Encoder学的纯粹是"reference motion要执行什么skill"——state-independent的skill identity
- Decoder学的是"在当前character state下如何执行这个skill"——state-conditional execution

这给generative controller提供了干净的接口：transformer只需要预测skill identity token，不需要关心当前character state（state会从context token传入decoder）。

Reference motion window用dilated indices $h = \{1, 2, 5, 7, 12, 18, 25\}$，类似CNN的dilated convolution。这能在compact representation里capture long-horizon信息。

---

## 5. Generative Controller: GPT-style Next-Token Prediction

### 5.1 Token Grouping

直接对40个9-ary token做autoregressive，sequence length = 40。Self-attention cost = $O(40^2) = 1600$ per layer。

Grouping：把每$G$个consecutive token pack成一个super-token。Paper选$G=5$：
- 40 / 5 = 8个grouped tokens
- 每个grouped token vocab = $9^5 = 59049$
- Sequence length从40降到8
- Self-attention cost从1600降到64

这相当于LLM里char-level → subword-level (BPE)的升级。Table 3 ablation展示了trade-off：

| G | Vocab size | N_token | APD↑ | ADE↓ | Accel↓ | N_param | Mem(GB) | FPS |
|---|---|---|---|---|---|---|---|---|
| 8 | 4.3×10^7 | 5 | - | - | - | 4.4×10^10 | 170 | OOM |
| 5 | 59049 | 8 | 0.34 | 0.30 | 3.67 | 6.0×10^7 | 0.27 | 92.85 |
| 4 | 6561 | 10 | 0.29 | 0.29 | 3.21 | 6.7×10^5 | 0.03 | 115.47 |
| 2 | 81 | 20 | 0.26 | 0.27 | 3.04 | 8.3×10^4 | <0.01 | 56.43 |
| 1 | 9 | 40 | 0.27 | 0.24 | 2.88 | 9×10^3 | <0.01 | 25.15 |

**Insight**：
- $G=1$：vocab只有9，模型几乎没表达力。APD低（0.27）。FPS低（25.15）因为40个token的self-attention cost高
- $G=8$：vocab 4.3×10^7，比现代LLM还大（GPT-2 vocab 50257），OOM
- $G=5$：APD最高（0.34），表示行为diversity最大。FPS 92.85，inference cost可接受

为什么$G=5$比$G=1$的APD高？我的理解是，grouped token carry higher-level semantic information。每个grouped token代表5个原始token的组合，相当于一个"motion subword"。Transformer学这些semantic unit之间的关系比学char-level的token-to-token关系更容易。

### 5.2 Autoregressive Factorization

公式 (5)：
$$p_\theta(\tilde{z}_t | s_t) = p_\theta(\tilde{z}_t^0 | s_t) \prod_{j=1}^{d'-1} p_\theta(\tilde{z}_t^j | s_t, \tilde{z}_t^{<j})$$

变量：
- $s_t$：character state，作为context token（通过MLP state encoder投影成embedding）
- $d' = 8$：grouped token的数量
- $\tilde{z}_t^j$：第$j$个grouped token，取值于$\{0, 1, ..., 9^5-1\}$
- $\tilde{z}_t^{<j}$：前$j$个已生成token
- $p_\theta$：transformer decoder的输出categorical distribution

每个token生成条件于：character state + 之前生成的token。Causal self-attention确保$j$位置的token只能attend to $\{0, 1, ..., j\}$位置的token。

Loss (公式6)：
$$\mathcal{L}_{CE} = -\sum_{j=0}^{d'-1} \log p_\theta(\tilde{z}_t^j | s_t, \tilde{z}_t^{<j})$$

标准cross-entropy + label smoothing。EMA with smoothing factor 0.9。

### 5.3 Transformer架构 (Table 10)

- $d_{model} = 1024$
- $N_{heads} = 4$
- $N_{layers} = 6$
- $d_{ff} = 4096$
- GELU activation
- Learned positional encoding
- Vocab size = $9^5 = 59049$

这是个相对小的transformer（约100M params量级），不像LLM那么大。原因是motion的复杂度远低于自然语言。

### 5.4 Inference: Nucleus Sampling

Inference时，每个token位置softmax后得到59049维categorical distribution。Nucleus sampling top-p=0.9：

1. 按probability降序排列所有vocab
2. 累积probability直到≥0.9，截断
3. 在截断后的subset里renormalize + sample

这避免了sample low-probability outlier（导致unrealistic motion），同时保留diversity。

---

## 6. CoLA: Conditional Low-rank Adaptation

### 6.1 公式分解

公式 (7)：
$$Wx = W_0 x + m \frac{B(\text{diag}(\gamma(c))Ax + \beta(c))}{\|B(\text{diag}(\gamma(c))Ax + \beta(c))\|_F}$$

变量逐个解释：
- $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$：frozen pretrained weight matrix
- $x$：layer input
- $A \in \mathbb{R}^{r \times d_{in}}$：down-projection，rank $r=64$
- $B \in \mathbb{R}^{d_{out} \times r}$：up-projection，zero-initialized
- $c$：task condition（e.g., target position, heightmap features from CNN）
- $\gamma(c), \beta(c) \in \mathbb{R}^r$：lightweight MLP从$c$产生的modulation vectors
- $\text{diag}(\gamma(c))$：对角矩阵，做element-wise scaling
- $m \in \mathbb{R}^{d_{out}}$：learnable magnitude vector
- $\|\cdot\|_F$：Frobenius norm（对vector就是L2 norm）

### 6.2 设计哲学的三个层次

**Layer 1: LoRA**。原始LoRA：$Wx = W_0 x + BAx$，$B, A$ low-rank。缺点：update的magnitude和direction耦合，optimizer同时调两者容易oscillate。

**Layer 2: DoRA**。DoRA分解LoRA update为magnitude + direction：
$$Wx = W_0 x + m \cdot \frac{BAx}{\|BAx\|_F}$$
这让low-rank update的direction被explicitly normalized，magnitude由$m$单独控制。优化更稳定，expressiveness更好。

**Layer 3: CoLA = DoRA + FiLM**。CoLA在DoRA的低秩空间$r$里加入task conditioning：
$$BAx \to B(\text{diag}(\gamma(c))Ax + \beta(c))$$
这就是FiLM的affine modulation，但作用在低秩空间$\mathbb{R}^r$而非full feature space $\mathbb{R}^{d_{out}}$。这极大降低了conditioning的parameter count。

### 6.3 为什么B zero-init？

训练开始时$B=0$，整个adapter contribution = 0，模型行为完全等于pretrained model。然后optimizer逐渐让adapter deviate。这避免了random init的adapter在训练初期破坏pretrained model的行为。

### 6.4 Hyperparameters (Table 12)

- Rank $r = 64$
- Scaling $\alpha = 128$（low-rank update的global scale）
- top-p = 0.9
- Temperature $T = 1.0$

总adapter parameter < 1% of pretrained model。

---

## 7. RL Fine-tuning的关键设计

### 7.1 Action space是token序列

Action space是$d'=8$个discrete token序列，每个token是$L^G$-ary categorical。PPO在token-level log-prob上做update，advantages在同一个decision step内的所有tokens之间共享（同一step的8个token共享一个advantage estimate）。

### 7.2 Nucleus sampling as exploration regularization

这是GPC的PEFT最关键设计。Inference时：
1. 取frozen base model的output distribution
2. 取top-p=0.9 mass作为support set $\mathcal{S}$
3. Renormalize adapted model的distribution on $\mathcal{S}$
4. Sample from renormalized distribution

这相当于：adapter想sample一个base model认为低概率的token，这个token会被直接clip掉。adapter只能在base model的high-probability manifold里explore。

Fig 17 ablation：
- top-p=0.5：收敛快但限制了skill set，return上限低
- top-p=0.99：自由度太大导致jitter
- top-p=0.9：sweet spot
- T=1.2：distribution flatten，sample到low-prob code导致persistent jitter

### 7.3 SFT before RLFT

有example motion时先做SFT：把adapter的cross-entropy loss在example motion对应的discrete latent code上minimize。这让adapter bias到正确的skill subset，避免RLFT在huge skill space里explore的低效。

Table 4 ablation：
| Method | Return | Pert.Ret. | Entropy | APD | ADE |
|---|---|---|---|---|---|
| SFT | 143.36 | 125.44 | 2.57 | 0.24 | 0.15 |
| w/o SFT | 230.42 | 176.35 | 5.08 | 0.26 | 0.27 |

**有趣的观察**：w/o SFT的return反而更高（230 vs 143）。这说明SFT让model过于restricted，少了exploration，但return上限反而低。但APD也低（0.24 vs 0.26），ADE反而低（0.15 vs 0.27）。

我的解读：SFT真正的价值是style control，不是max return。Fig 10/11显示SFT能让model保持crouched walking style——head height维持在0.8-1.3m，没SFT的model会做各种height的motion。所以SFT适合"我要这个style"的task，不适合"maximize task reward"的task。

---

## 8. Tracking Reward细节

公式 (8)：
$$r = w_{gp}r_{gp} + w_{gr}r_{gr} + w_{rh}r_{rh} + w_{jv}r_{jv} + w_{jav}r_{jav}$$

每个$r_{(\cdot)} = \exp(c_k \cdot \text{error}_k)$形式，error越小reward越接近1。

权重 (Table 5)：
| Term | Weight $w_k$ | Coeff $c_k$ | Meaning |
|---|---|---|---|
| $r_{gp}$ | 0.5 | -100 | global joint position error |
| $r_{gr}$ | 0.3 | -5 | global joint rotation error |
| $r_{jv}$ | 0.1 | -0.5 | joint linear velocity error |
| $r_{jav}$ | 0.1 | -0.1 | joint angular velocity error |
| $r_{rh}$ | 0.2 | -100 | root height error |

注意weight总和 = 1.2，不是1.0（是weighted sum不是convex combination）。

**设计insight**：
- $r_{gp}$权重最大（0.5）+ coeff最大（-100）：global joint position直接反映motion视觉相似度，是主要signal
- $r_{rh}$权重0.2 + coeff大（-100）：root height错导致整个motion都崩，所以单独强调
- $r_{gr}$权重0.3但coeff小（-5）：joint rotation的sensitivity比position低

---

## 9. 关键实验数据深度解读

### 9.1 FSQ vs VQ-VAE vs MLP (Table 1)

Bones (680 hr)：
| Method | Succ. (%) | MPJPE (mm) |
|---|---|---|
| MLP | 99.98 | 25.56 |
| VQ-VAE | 99.94 | 37.92 |
| FSQ | 99.98 | 34.90 |

AMASS (40 hr)：
| Method | Succ. (%) | MPJPE (mm) |
|---|---|---|
| MLP | 99.59 | 30.26 |
| VQ-VAE | 99.30 | 59.28 |
| FSQ | 99.51 | 44.43 |

**Insight 1**：MLP永远最好，expected——quantization bottleneck会丢精度。
**Insight 2**：FSQ比VQ-VAE好（34.90 vs 37.92mm on Bones, 44.43 vs 59.28mm on AMASS）。FSQ的精度损失比VQ-VAE小，因为VQ-VAE的codebook collapse问题在AMASS这种moderate-scale数据上更严重。
**Insight 3**：Bones上VQ-VAE和FSQ差距小（37.92 vs 34.90），AMASS上差距大（59.28 vs 44.43）。Large-scale data能缓解VQ-VAE的codebook collapse（更多code被activate）。

### 9.2 End-to-end RL vs Kinematic pretrain (Table 2)

| Method | Succ. (%) | MPJPE (mm) | Util. (%) |
|---|---|---|---|
| FSQ-K (kinematic pretrain + frozen encoder) | 99.03 | 78.26 | 76.34 |
| FSQ (end-to-end RL) | 99.98 | 34.90 | 82.15 |

**Critical insight**：End-to-end RL把MPJPE从78mm降到35mm（降一半），codebook util从76%升到82%。这证明discrete latent必须为physical control优化，不能只优化kinematic reconstruction。

为什么？Kinematic reconstruction只关心"latent能不能reconstruct reference motion"。但physical control关心的是"latent能不能drive一个physics-simulated character reproduce reference motion"。两者的gap包括：
- PD controller dynamics
- Character的actuation limit
- Contact physics
- Gravity, momentum

End-to-end RL让encoder学到对physics-friendly的latent representation。

### 9.3 vs Diffusion-based trackers (Table 7)

| Method | Type | End2end | Succ. (%) | MPJPE (mm) |
|---|---|---|---|---|
| PDP | Continuous (DDPM) | No (distilled from MLP) | 98.90 | 37.32 |
| FPO | Continuous (flow matching) | Yes | 96.40 | 41.98 |
| GPC | Discrete (FSQ) | Yes | 99.51 | 44.43 |
| MLP | Continuous | Yes | 99.59 | 30.26 |

**Insight**：Pure imitation objective下continuous MLP永远最好。Diffusion/flow matching的expressiveness在tracking task上是overhead——额外的generative modeling capacity没转化为tracking精度。

那为什么还用FSQ？因为FSQ提供了与next-token prediction paradigm的天然interface。Continuous方法做不到autoregressive token-level modeling。GPC的真正价值不在tracking精度，而在它enable了后续的generative controller + PEFT pipeline。

### 9.4 GPC vs CVAE robustness (Table 9)

无perturbation：
| Metric | GPC | CVAE |
|---|---|---|
| Norm. Jerk ↓ | 657±147 | 1072±38 |
| Mean Accel. (m/s²) ↓ | 4.33±0.46 | 6.14±0.21 |
| APD Root (m) ↑ | 2.66 | 2.40 |
| APD Pose (m) ↑ | 13.55 | 12.22 |
| Survival Rate (%) ↑ | 82.1% | 79.4% |

中等push (2.4 m/s)：
| Metric | GPC | CVAE |
|---|---|---|
| Survival Rate (%) ↑ | 68.1% | 44.4% |

强push (9.8 m/s)：
| Metric | GPC | CVAE |
|---|---|---|
| Survival Rate (%) ↑ | 52.5% | 3.1% |

**Key insight**：无perturbation时两者survival close（82.1% vs 79.4%）。Perturbation越大，GPC优势越明显。强push下CVAE几乎完全collapse（3.1%）。

为什么？GPC的token distribution严格约束在data manifold内。即使push让character偏离normal state，stochastic sampling能让GPC"重新选择"一个recovery skill（e.g., get-up, roll, cartwheel recovery）。CVAE的continuous latent一旦drift off-manifold就回不来——decoder输出是deterministic function of latent，没有"重新选择"的freedom。

Fig 5/6显示GPC的emergent recovery behavior：fall后自动transition到roll recovery。这些behavior没有任何explicit reward，完全来自large-scale data的diversity + stochastic sampling。这是"emergent"的精髓。

### 9.5 Downstream task对比 (Table 13)

| Method | Succ. (%) ↑ | Final Target Dist. (m) ↓ |
|---|---|---|
| MaskedMimic | 89.92 | 0.44 |
| CVAE | 93.89 | 0.41 |
| GPC | 94.20 | 0.37 |

GPC略好CVAE，远好MaskedMimic。MaskedMimic的问题：training时condition在random masked future joint positions，inference时condition在fixed root height（goal-reaching task setup）。这个distribution shift让MaskedMimic produce unnatural behavior。

---

## 10. 与Related Work的脉络联系

### 10.1 与ASE / CALM / MaskedMimic

- ASE (https://arxiv.org/abs/2205.01906)：GAN-based skill embedding，continuous latent。Mode collapse risk。
- CALM (https://research.nvidia.com/labs/toronto-ai/calm/)：ASE + semantic conditioning
- MaskedMimic (https://research.nvidia.com/labs/toronto-ai/masked-mimic/)：masked inpainting，continuous latent

GPC相对优势：mode collapse免疫（FSQ无learned codebook）、off-manifold drift免疫（discrete grid）、stochastic sampling更自然（categorical distribution vs deterministic continuous）。

### 10.2 与Decision Transformer / Trajectory Transformer (附录C.2)

- Decision Transformer (https://arxiv.org/abs/2106.01345)：offline RL，continuous embedding
- Trajectory Transformer：offline RL，per-dimension binning
- GPC：online RL + simulator，FSQ tokenization + skill-level (not action-level) tokens

GPC的关键区别：token是skill-level的，每个grouped token对应一小段未来motion的"skill identity"，不是单个action dimension的离散值。这让token carry higher semantic meaning。

### 10.3 与TokenHSI

TokenHSI：tokenize per-task goals，deterministic policy，AMP-style adversarial reward。在固定predefined task set上训练，transfer到新task。
GPC：unconditional generative prior，task-agnostic pretraining。PEFT用CoLA，不在pretraining时绑定task。

### 10.4 与MoConvQ

MoConvQ：VQ-VAE-based，residual VQ。需要VQ-VAE的所有heuristics（EMA, dead code reinit, auxiliary loss）。
GPC：FSQ，no learned codebook。Simpler training，better utilization。

### 10.5 与PULSE / Perpetual Humanoid Control

PULSE (https://arxiv.org/abs/2305.09455)：continuous CVAE-based prior，universal humanoid motion representations。Downstream task用high-level policy sample latent。
GPC：discrete token prior，autoregressive modeling。Downstream用CoLA adapter modulate frozen transformer。

---

## 11. Optimizer细节：Muon

Table 6显示GPC用Muon optimizer训练actor，AdamW训练critic。

Muon (https://kellerjordan.github.io/posts/muon)是geometry-aware optimizer for matrix-shaped parameters。它对每个weight matrix做SVD-like decomposition，对orthogonal component用Newton-style update。这比Adam对矩阵参数的element-wise更新更高效。

GPC只用Muon训练encoder/decoder的hidden weight matrices（matrix-shaped），低维参数和interface layer（biases, boundary linear layers）用AdamW。这是Muon论文推荐的hybrid策略。

---

## 12. Skill Latent Space的Probing (附录B.6)

Paper设计skill retrieval实验验证FSQ latent space的structure：
- Coarse level：8 classes（walk, jog, jump, kick, punch, crawl, idle, dance）
- Fine level：20 classes（e.g., tired_walk, angry_walk, zombie_walk, happy_jog）

用Sentence-BERT-style mean-pooling得到每个motion clip的global embedding，用L2 distance做retrieval。

Table 8结果：
- Coarse: P@1 = 0.72, MRR = 0.81
- Fine: P@1 = 0.66, MRR = 0.76

这证明FSQ latent不是random assignment——相似skill的latent code在grid上靠近。Fig 15的t-SNE visualization也显示clustering by motion type。

更intriguing的是Fig 14的skill composition：interpolate "punching"和"kicking"的latent code，能produce一个同时punch + kick的motion。这证明latent space有linear compositionality。

---

## 13. Limitations和Future Directions

Paper Section 9诚实讨论了几个limitation：

1. **Locomotion-focused**：只测试了locomotion + HSI（platform, barrier）。没测试manipulation、tool use、social interaction。

2. **No text conditioning**：没法用自然语言指令控制。这是obvious extension——借鉴MotionGPT (https://arxiv.org/abs/2310.16323)的设计，加text encoder + cross-attention。

3. **Human-object interaction limited**：只测试了静态环境（platform, barrier）。Dynamic object interaction（pick up, push, throw）没测。

4. **CoLA的FiLM input dim和layer结构没详细说明**：r=64可能限制expressiveness。Higher rank可能better但更多parameter。

5. **SFT的trade-off没充分讨论**：Table 4显示SFT让return降低（143 vs 230）。Paper说SFT适合style control，但当task需要max return + behavioral diversity时SFT可能harmful。这需要更多分析。

---

## 14. 对未来工作的启示

### 14.1 Multimodal conditioning
Text + motor token。可以借鉴MotionGPT的设计，让text encoder输出condition embedding，通过CoLA的FiLM注入到generative controller。

### 14.2 Human-object interaction
Manipulation skill tokenization。目前FSQ只encode locomotion skill，object manipulation需要更复杂的skill representation（object state + body state joint encoding）。

### 14.3 Real robot transfer
Sim-to-real。FSQ的discrete action space可能对sim-to-real有利——distributional shift更可控，因为action永远在data manifold内。

### 14.4 Hierarchical control
High-level text → mid-level motor token → low-level PD torque。这是GPC的天然extension。

### 14.5 Lifetime learning
把CoLA的task condition $c$扩展成memory（e.g., past task embeddings），让agent记住过去任务。这接近meta-RL。

### 14.6 Larger transformer
目前GPC的transformer只有6层、1024 dim。Scale到LLM size（100+ layers, 10K+ dim）可能enable更复杂skill composition。

---

## 15. 总结：GPC的真正贡献

GPC的真正贡献：证明了discrete token + autoregressive transformer范式在physics-based motor control上可行且effective。

具体来说：
1. **FSQ优于VQ-VAE** for motion tokenization——simpler training, better utilization, no heuristics
2. **End-to-end RL training of discrete latent** is critical——kinematic pretraining不够
3. **GPT-style autoregressive**可以model motor skill distribution，产生emergent recovery behavior
4. **CoLA PEFT** with nucleus sampling constraint可以高效adapt到downstream task while preserving pretrained behavior
5. **Stochastic sampling**是关键robustness source——比continuous prior的deterministic inference更robust to perturbation

整个pipeline可以看作：把LLM的成功配方（discrete token + transformer + PEFT）完整迁移到motor control。这是paradigm transfer的有力证明。

可能影响的不只是character animation领域。Robotics的motor control、protein folding的conformational control、game AI的NPC control都可能借鉴这个pipeline。
