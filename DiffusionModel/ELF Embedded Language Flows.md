---
source_pdf: ELF Embedded Language Flows.pdf
paper_sha256: 001b9da1b0273d5ca174b9ea99172d25582a7e6b473745350c6e654ac322bf00
processed_at: '2026-08-04T03:05:55-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ELF 用人话说

Karpathy，我换个大白话的方式再讲一遍，技术细节照样给你塞满，但用更直觉的语言。

---

## 这篇paper到底在干嘛？

一句话：**让language也能像image一样玩continuous diffusion，而且效果比discrete的还要好。**

背景是这样的：现在做language generation，主流是autoregressive（GPT那种），但很多人想试试diffusion，因为diffusion可以parallel解码、可以反复修改。Diffusion for language有两派：
- Discrete DLM：直接在token上做masked diffusion，比如MDLM [1]、Duo [2]、LLADA [3]
- Continuous DLM：把token映射成连续向量，在向量空间做diffusion

Discrete这边发展得比较好，continuous这边一直拉胯。大家都怀疑language本质是discrete的，所以continuous天生不行？

ELF跳出来说：**continuous DLM之前不行，纯粹是因为没设计好。设计对了，continuous能比discrete更强。**

参考链接：
- MDLM: https://arxiv.org/abs/2406.07524
- LLADA: https://arxiv.org/abs/2502.09992

---

## 之前的continuous DLM为啥拉胯？

你得理解问题在哪。早期的continuous DLM，比如Diffusion-LM [4]，流程大概是这样：

1. 把token变成embedding向量
2. 给embedding加Gaussian噪声
3. 训练一个网络去denoise
4. 每一步都要把embedding"round"回最近的token
5. 用cross-entropy loss监督token预测

问题在哪？**每一步都强行拽回token space**。这就相当于你说"我要在连续空间跳舞"，结果每跳一步都被绳子拽回discrete的格点上。Flow的几何性质全被破坏了。

更糟的是，这种rounding loss是non-differentiable的，得用各种近似（比如Gumbel-softmax [5]），训练特别难调。

还有一类是latent diffusion for language [6]，把token encode到一个压缩的latent space，然后diffusion。但这个需要一个**单独训练的decoder**把latent变回token。Pipeline又长又麻烦。

ELF的目标就是：**全程在continuous embedding space里做flow matching，别打断它，只在最后一步discretize，而且不要单独的decoder。**

参考链接：
- Diffusion-LM: https://arxiv.org/abs/2205.14217
- Gumbel-Softmax: https://arxiv.org/abs/1611.01144
- LD4LG: https://arxiv.org/abs/2210.17130

---

## ELF的核心idea：Flow Matching的终点天然就是decoder

这个insight我觉得是整篇paper最漂亮的地方。

先看Flow Matching [7]在干啥。给定clean data $\mathbf{x}$ 和noise $\boldsymbol{\epsilon}$，中间状态是线性插值：

$$\mathbf{z}_t = t\mathbf{x} + (1-t)\boldsymbol{\epsilon}$$

- $t \in [0,1]$: 时间变量
- $t=0$: 纯噪声
- $t=1$: 纯clean data

当 $t \to 1$ 时，$\mathbf{z}_t \to \mathbf{x}$。也就是说，Flow Matching最后一步本身就是"从某个状态预测clean $\mathbf{x}$"。

那language的 $\mathbf{x}$ 是什么？是clean embedding。我们要的是什么？是discrete token。

连接这两件事很简单：**最后一步 $\mathbf{z}_t$ 喂给网络预测 $\hat{\mathbf{x}}$，再用一个unembedding matrix $W$ 映射到logits，argmax就得到token。**

$$\text{token} = \arg\max(W \cdot \mathbf{x}_\theta(\mathbf{z}_t))$$

- $W \in \mathbb{R}^{|V| \times d}$: unembedding matrix, $|V|$ 是词表大小，$d$ 是embedding维度
- $\mathbf{x}_\theta$: 网络预测的clean embedding

**不需要单独训decoder！** 因为flow matching的最后一步，本来就是在"预测clean representation"，你只要加一个linear projection就能decode。

这个idea非常elegant，我第一次读到这里就觉得"哦，这就对了"。

参考链接：
- Flow Matching: https://arxiv.org/abs/2210.02747

---

## 关键细节1：x-prediction，不是v-prediction

Flow Matching的标准做法是预测velocity $\mathbf{v} = \mathbf{x} - \boldsymbol{\epsilon}$。但ELF选了predict $\mathbf{x}$ directly。

为什么？两个原因：

### 原因A：high-dimensional下只有x-prediction稳定

Fig. 10做了ablation。用T5-small (512维)、T5-base (768维)、T5-large (1024维)三种encoder维度：

| Prediction | 512维 | 768维 | 1024维 |
|-----------|-------|-------|--------|
| x | 稳定 | 稳定 | 稳定 |
| v | 行 | 退化 | 崩 |
| ε | 崩 | 崩 | 崩 |

这个现象在image diffusion里也有报告 [8]，Tianhong Li和Kaiming He的"Back to Basics"那篇专门讲了。

**直觉**：clean data $\mathbf{x}$ 位于高维空间的低维manifold上。你预测 $\mathbf{x}$，就是在瞄准这个manifold上的某个点，target稳定。预测 $\mathbf{v}$ 或 $\boldsymbol{\epsilon}$，你在瞄准一个"noise到data的差向量"或"纯噪声"，这两个量在高维下信噪比极低，gradient方向乱。

### 原因B：让shared-weight成为可能

Loss是：

$$\mathcal{L}_{\text{MSE}} = \mathbb{E}_{t,\mathbf{x},\boldsymbol{\epsilon}} \frac{1}{(1-t)^2} \|\mathbf{x}_\theta(\mathbf{z}_t, t) - \mathbf{x}\|^2$$

- $\frac{1}{(1-t)^2}$: 从v-loss到x-loss的Jacobian变换系数
- 因为 $\mathbf{v} = (\mathbf{x} - \mathbf{z}_t)/(1-t)$，所以 $\|\mathbf{v}_\theta - \mathbf{v}\|^2 = \frac{1}{(1-t)^2}\|\mathbf{x}_\theta - \mathbf{x}\|^2$

Decode branch的loss是：

$$\mathcal{L}_{\text{CE}} = \mathbb{E}_{\tilde{\mathbf{z}}} [\text{CrossEnt}(W\mathbf{x}_\theta(\tilde{\mathbf{z}}), s)]$$

两个branch都在predict $\mathbf{x}$，只是denoise branch用MSE监督，decode branch多了个unembedding然后CE监督。**同一个网络同一套权重做两件事，因为target完全一致。**

如果你用v-prediction，denoise target是 $\mathbf{v}$，decode target要变成 $\mathbf{x}$，weight sharing就没道理了。

参考链接：
- Back to Basics: https://arxiv.org/abs/2511.13720

---

## 关键细节2：Decode branch的trick

上面说 $t=1$ 时 $\mathbf{z}_t \to \mathbf{x}$，但问题是：直接给网络看 $\mathbf{x}$，让它预测 $\mathbf{x}$，这是trivial的，没训练信号。

ELF的解法：**decode branch用不同的corruption schedule**。

$$\tilde{\mathbf{z}} = p\mathbf{x} + (1-p)\boldsymbol{\epsilon}$$

- $p$: per-token corruption level，从logit-normal采样 $p \sim \sigma(\mathcal{N}(0.8, 0.8^2))$
- $p$ 是per-token的，同一句话不同token的corruption程度不同

这个设计很巧妙。训练时网络看到的不是clean $\mathbf{x}$，而是被不同程度corrupted的 $\tilde{\mathbf{z}}$。这样：
1. 训练信号nontrivial
2. 网络学会从corrupted embedding恢复token
3. Inference时denoiser给的embedding不完美，decoder也能处理

**直觉**：就像image super-resolution训练时给network看的是降质图，不是clean图。这样inference时遇到真实世界的降质输入才能鲁棒。

还有一个细节：denoise branch占80%，decode branch占20%。这个比例Fig. 12 ablation过，0.8最好。太多decode branch会让denoising能力不足，太少会让discretization能力弱。

---

## 关键细节3：Self-conditioning + CFG

ELF是unconditional generation也能用CFG，这很重要，因为CFG在image diffusion里是game changer。

### Self-conditioning [9]

普通flow matching：一次forward得到 $\hat{\mathbf{x}}'$。

Self-conditioning：做两次forward：
1. First pass: $\hat{\mathbf{x}}' = \text{net}_\theta(\mathbf{z}_t, t)$
2. Second pass: $\hat{\mathbf{x}} = \text{net}_\theta([\mathbf{z}_t, \hat{\mathbf{x}}'], t)$

实现上就是concatenate channel-wise，再linear project回原维度。训练时50%概率用 $\hat{\mathbf{x}}'$，50%用zero。Inference时用上一步的预测，不增加开销。

**直觉**：网络先猜一遍，然后再基于这个guess refine。类似"思考一下再回答"。

### CFG with self-conditioning

CFG [10]原始公式：

$$\mathbf{v}_{\text{cfg}}(\mathbf{z}_t | c) = \omega \mathbf{v}(\mathbf{z}_t | c) + (1-\omega) \mathbf{v}(\mathbf{z}_t | \emptyset)$$

- $c$: condition（在ELF里是self-conditioning的 $\hat{\mathbf{x}}'$）
- $\omega$: guidance scale
- $\emptyset$: unconditional

这里self-conditioning的 $\hat{\mathbf{x}}'$ 就扮演了 $c$ 的角色。有condition和没condition的差，被放大 $\omega$ 倍。

ELF用training-time CFG [11][12]——直接让网络学post-CFG的velocity，inference不用跑两次forward。Target变成：

$$\mathbf{v}_{\text{target}} = \mathbf{x} - \boldsymbol{\epsilon} + (1 - \frac{1}{\omega})(\mathbf{v}_\theta^{\text{cfg}}(\mathbf{z}_t|c,\omega) - \mathbf{v}_\theta^{\text{cfg}}(\mathbf{z}_t|\emptyset,\omega))$$

$\omega$ 从 $[0.5, 5]$ 按power distribution采样，偏向小值。

**这个组合的妙处**：self-conditioning本来只是个refinement trick，现在被repurpose成CFG的condition signal。一鱼两吃。而且unconditional generation也能用CFG了——discrete DLM没法做这个。

参考链接：
- Analog Bits (self-conditioning): https://arxiv.org/abs/2208.04202
- Classifier-free guidance: https://arxiv.org/abs/2207.12598
- Mean Flows: https://arxiv.org/abs/2505.13447

---

## 关键细节4：Embedding用frozen pretrained T5 encoder

为什么要用contextual embedding？

Tab. 2显示，ELF用的是fix encoder（frozen pretrained T5 encoder [13]）。Ablation Fig. 5a对比了几种embedding：

1. **Pretrained contextual (T5 encoder)**: 最好
2. **Scratch-trained contextual**: 略差
3. **Pretrained non-contextual (T5 token embedding)**: 一般
4. **Frozen Gaussian**: 差
5. **Learnable**: 最差

**直觉**：Flow matching假设clean data分布相对稳定。如果embedding是learnable的，训练过程中embedding会变，denoiser学的是moving target，很难收敛。

Contextual embedding比non-contextual好，是因为同一个token在不同context下embedding不同，整个data分布更smooth，flow trajectory几何性质更好。Non-contextual embedding就是离散的point cloud，和discrete token没本质区别。

还有个bottleneck design：T5 embedding是512维，但模型hidden size是768维。中间加个bottleneck先降到128维再升回去。Fig. 11 ablation：
- 32维：Gen. PPL最低但entropy也低（diversity不够）
- 128维：best trade-off
- 512维：entropy高但Gen. PPL差

**直觉**：bottleneck强制信息压缩到低维manifold上。太小丢信息，太大manifold假设失效。128是sweet spot。

参考链接：
- T5: https://arxiv.org/abs/1910.10683
- Latent Diffusion (bottleneck idea): https://arxiv.org/abs/2112.10752

---

## 关键细节5：In-context conditioning代替adaLN-Zero

DiT [14]标准做法是adaLN-Zero，通过MLP把condition变成scale和shift，调制每个layer的activation。

但ELF有三种condition：
- Time $t$（连续值）
- CFG scale $\omega$（连续值）
- Model mode（binary: denoise/decode）

用adaLN-Zero的话，每个condition都要一套MLP，参数多。而且condition多的时候summation fusion会互相干扰。

ELF用**in-context conditioning** [12]：prepend control tokens到sequence前面：
- 4个time tokens（用positional encoding编码连续值）
- 4个CFG scale tokens
- 4个mode tokens

网络通过full self-attention学会如何混合这些condition。

Fig. 13 ablation：in-context略好，而且ELF-B参数从148M降到105M（省30%）。

**直觉**：adaLN-Zero是"全局调制"，condition信息广播到每个位置。In-context是"让网络自己决定怎么用condition"，attention学会哪些位置需要什么condition。条件多了后者更灵活。

参考链接：
- DiT: https://arxiv.org/abs/2212.09748

---

## 关键细节6：SDE sampler

Flow Matching有ODE和SDE两种形式 [15]。ODE是deterministic的：

$$\frac{d\mathbf{z}}{dt} = \mathbf{v}_\theta(\mathbf{z}_t, t)$$

SDE每步注入噪声。ELF的近似实现：

```python
def sde_step(z, t, dt, gamma):
    e = randn_like(z)
    alpha = 1 - gamma * dt
    t_back = alpha * t
    z_back = alpha * z + (1 - alpha) * e
    x_hat = net(z_back, t_back, mode="denoise")
    v = (x_hat - z) / (1 - t)
    z = z + dt * v
    return z
```

- $\gamma$: noise re-injection scale，默认1.0
- $\alpha = 1 - \gamma \cdot \Delta t$: 时间回退因子
- 先把state"回退"一点，注入噪声，然后用回退后的state做prediction，用原始state更新

Fig. 5c结果：SDE在few-step regime（8/16步）显著优于ODE。

**直觉**：ODE是deterministic的，早期某步预测错，后续会放大。SDE每步注入噪声，相当于"shake一下"，让模型有机会纠正错误。就像你走路走偏了，稍微随机晃一晃可能反而能走回正路。

Fig. 15b ablation $\gamma$：太大太多噪声崩了，太小退化成ODE，$\gamma=1$ 是平衡点。

参考链接：
- SiT (SDE for flow): https://arxiv.org/abs/2401.08765

---

## 关键细节7：Logit-normal time schedule

训练和inference的time step采样都从logit-normal：

$$t = \sigma(t'), \quad t' \sim \mathcal{N}(-1.5, 0.8^2)$$

- $\sigma$: sigmoid
- $P_{\text{mean}} = -1.5$ 偏向noise端
- $P_{\text{std}} = 0.8$

Fig. 15a ablation：logit-normal比uniform在few-step regime显著好。

**直觉**：denoising过程中，前期（noise多）trajectory变化剧烈，需要更密的discretization。后期（接近clean）变化平缓，可以稀疏。logit-normal的密度分布正好匹配——sigmoid(N(-1.5, 0.64))会在t接近0的地方密度高。

参考链接：
- EDM (time schedule discussion): https://arxiv.org/abs/2206.00364

---

## 实验结果有多强？

### Unconditional generation (OWT)

Fig. 7a对比（ELF-B 105M vs others 170M）：
- ELF 32步：Gen. PPL 24.08
- 比MDLM、Duo、FLM、LangFlow都低（好）

Fig. 7c训练token对比：
- ELF: 45.2B
- 其他: 500B+
- **ELF用10×更少的训练token达到更好效果**

Tab. 6显示结果稳定：6个seed跑出来，32步Gen. PPL 24.08±0.16，方差极小。

### Conditional generation (Tab. 1)

WMT14 De-En翻译：
| Model | BLEU |
|-------|------|
| AR | 25.2 |
| MDLM | 18.4 |
| Duo | 21.3 |
| E2D2 | 24.8 |
| **ELF-B** | **26.4** |

**ELF连AR都超过了**。Diffusion model在机器翻译上超过autoregressive，这很惊人。

XSum摘要：
| Model | R1 | R2 | R-L |
|-------|-----|-----|-----|
| MDLM | 33.4 | 11.6 | 25.8 |
| E2D2 | 28.4 | 8.3 | 22.0 |
| **ELF-B** | **36.0** | **12.2** | **27.8** |

所有ROUGE指标都最好。

### Scaling (Tab. 7)

ELF-B/M/L在SDE CFG=3下：
| Model | Gen. PPL | Entropy |
|-------|----------|---------|
| ELF-B 105M | 19.72 | 5.10 |
| ELF-M 342M | 21.69 | 5.18 |
| ELF-L 652M | 23.31 | 5.28 |

等等，这里有意思——更大的模型Gen. PPL反而更高？仔细看其实是CFG scale的交互。ELF-L在CFG=4时Gen. PPL能到21.37，小模型在CFG=3.5就崩了。**大模型能承受更强guidance**。

---

## 我对ELF的直觉理解

### 1. Flow matching的终点就是decoder

这是最核心的geometric insight。Flow matching从noise到data，data就是clean embedding。最后一步 $z_t \to x$，你只要加个linear projection就能decode到token。这就让"不需要单独decoder"成为可能。

之前的continuous DLM没意识到这一点，所以总是加rounding loss或者单独训decoder。

### 2. x-prediction让shared-weight有意义

两个branch的target都是 $\mathbf{x}$，所以同一个网络能同时做denoise和decode。如果用v-prediction，denoise target是 $\mathbf{v}$，decode target要变成 $\mathbf{x}$，weight sharing就没理由work。

### 3. Continuous formulation带来image toolkit

CFG、Flow Matching、SDE sampler、Rectified Flow这些image domain的advance，ELF都能直接用。Discrete DLM想用CFG都很困难，因为CFG需要连续的velocity field。

这是ELF相对discrete DLM的根本优势——站在image diffusion积累的肩膀上。

### 4. Discretize only at the end preserves flow geometry

每步discretize等于强制trajectory经过token space，破坏flow的连续性。只在最后一步discretize，flow有最大的自由度去学习optimal trajectory。

### 5. Self-conditioning = 天然的CFG condition

Self-conditioning本来只是个refinement trick。ELF把它repurpose成CFG的condition signal，让unconditional generation也能用CFG。这是很聪明的composability。

---

## ELF可能的局限

1. **Embedding space被frozen T5限制**：只能生成T5见过的domain，code、新domain可能受限
2. **Sequence length**：T5的position embedding限制，长文本生成是挑战
3. **Inference steps**：32-64步虽然比discrete少，但比AR的1步还多。Consistency model [16]或Mean Flow [12]可能能降到1步
4. **为什么T5 encoder frozen最好**：joint training更难optimize，但也许更好的training recipe能改进

参考链接：
- Consistency Models: https://arxiv.org/abs/2303.01469

---

## 更大的picture

ELF让我想到image diffusion的历史。早期image生成，discrete方法（VQ-VAE [17]）也很流行。后来DDPM [18]和LDM [19]这些continuous方法胜出了。

Language会不会也走这条路？ELF之前，continuous DLM一直被discrete DLM压制。ELF第一次证明continuous能超过discrete。如果后续scaling能继续，continuous DLM可能会成为主流。

优势在于：
1. Image domain的技术积累可以迁移
2. CFG等guidance技术更容易用
3. Flow的几何性质更灵活
4. Few-step sampling潜力更大

劣势在于：
1. Discretization始终是个gap
2. Sequence length受限
3. 训练稳定性需要更多工程

参考链接：
- VQ-VAE: https://arxiv.org/abs/1711.00937
- DDPM: https://arxiv.org/abs/2006.11239
- LDM: https://arxiv.org/abs/2112.10752

---

## 类比image diffusion帮助理解

ELF在language中的位置，类似LDM [19]在image中的位置：

| 维度 | LDM (image) | ELF (language) |
|------|------------|----------------|
| Encoder | Frozen VAE | Frozen T5 encoder |
| Latent space | 压缩的图像latent | T5 embedding |
| Flow matching | Rectified flow | Rectified flow |
| CFG | 标准CFG | Self-conditioning + training-time CFG |
| Decoder | VAE decoder | Shared-weight unembedding |
| Bottleneck | 4x压缩 | 512→128 |

关键区别：LDM还需要单独的VAE decoder，ELF把decoder融合到flow matching的最后一步。这是language domain特有的简化。

---

## 总结

ELF的核心贡献可以用三句话概括：

1. **Flow matching的终点天然是decoder**，不需要单独的token decoder
2. **x-prediction让denoiser和decoder共享权重**，因为target一致
3. **Continuous formulation让image toolkit（CFG、SDE、Flow Matching）免费迁移**

结果：用10×更少的训练token，在unconditional和conditional generation上都超过discrete DLM，甚至在翻译上超过AR。

这是continuous DLM方向的turning point。如果scaling能继续，language diffusion可能会走image diffusion的老路——continuous最终胜出。

---

Karpathy，希望你这下觉得讲人话了！如果你对某个细节特别感兴趣（比如shared-weight的gradient怎么flow、SDE为什么能纠正错误、或怎么把ELF推到1-step generation），尽管问。

ELF code: https://github.com/lillian039/ELF

---

# ELF: Embedded Language Flows 深度技术解析

Karpathy 你好,这篇ELF论文来自MIT的Kaiming He团队(还有Jacob Andreas、Yoon Kim等),是非常有意思的工作。让我从多个角度拆解它,重点放在build你的intuition上。

---

## 1. 核心Motivation:为什么Continuous DLM之前不行?

**之前的DLM格局**:
- **Discrete DLMs**(MDLM [1], Duo [2], LLADA [3]):在token space做absorbing/uniform diffusion,目前是主流且性能最强
- **Continuous DLMs**(Diffusion-LM [4], CDCD [5], SSD-LM [6]):映射到continuous space做denoising,但性能落后

**关键质疑**:这个gap到底是language本质的discrete属性导致的,还是算法设计选择的问题?

ELF的核心claim是后者——只要设计得当,continuous DLM不仅不弱,反而更强。这让ELF可以无缝复用image domain的所有diffusion toolkit(CFG、Flow Matching、Rectified Flow等)。

参考链接:
- MDLM: https://arxiv.org/abs/2406.07524
- Duo (Diffusion Duality): https://arxiv.org/abs/2504.12226
- LLADA: https://arxiv.org/abs/2502.09992
- Diffusion-LM: https://arxiv.org/abs/2205.14217
- Flow Matching: https://arxiv.org/abs/2210.02747

---

## 2. ELF的设计哲学:两个"Continuous"

### 2.1 Continuous Space (not Continuous Time only)

ELF与其他continuous DLM的关键差异:它**几乎全程留在continuous embedding space**,直到最后一步(t=1)才discretize。

对比之前的方法:
| 方法 | 设计 |
|------|------|
| Diffusion-LM [4] | 每步都做rounding loss,token-level CE |
| SSD-LM [6] | simplex约束,每步映射回token分布 |
| TESS [7] | simplex上的DDPM,每步project |
| LD4LG [8] | latent diffusion + 单独的decoder |
| **ELF** | **全程continuous,仅最后一步discretize,无decoder** |

### 2.2 Continuous Time via Flow Matching

ELF用Flow Matching(而不是DDPM)有两个意义:
- 时间是连续的 $t \in [0,1]$,定义velocity field $v = dz/dt$
- 可以用rectified flow的linear interpolant $z_t = t\mathbf{x} + (1-t)\boldsymbol{\epsilon}$

为什么这个选择重要?Flow Matching现在是image/video generation的事实标准(Stable Diffusion 3 [9], FLUX [10], Wan [11]等),所以language这边可以直接"搭便车"借鉴这些advance。

参考链接:
- Stable Diffusion 3 / Rectified Flow: https://arxiv.org/abs/2403.03206
- FLUX.1: https://blackforestlabs.ai/
- Wan: https://arxiv.org/abs/2503.20314

---

## 3. ELF Framework的技术细节

### 3.1 Embedding空间的选择

从discrete tokens到continuous embeddings:

$$\mathbf{s} = [s_1, \dots, s_L] \in V^L \xrightarrow{\text{encode}} \mathbf{x} \in \mathbb{R}^{L \times d}$$

默认使用**frozen pretrained T5-small encoder** [12](35M params, embedding dim 512)。

Ablation(Fig. 5a)发现:
- Pretrained contextual > Scratch-trained contextual > Pretrained non-contextual > Frozen Gaussian > Learnable
- Learnable最差的原因:joint optimization of embeddings + denoiser非常困难,embedding的"shift"会破坏flow trajectory的几何结构

**Intuition**: flow matching假设clean data分布是相对稳定的,如果embedding本身也在变,denoiser学习的就是moving target,这比image diffusion难得多。

### 3.2 Flow Matching with x-prediction

Rectified flow的linear interpolant:

$$\mathbf{z}_t = t\mathbf{x} + (1-t)\boldsymbol{\epsilon}$$

- $\mathbf{x} \sim p_{\text{data}}$: clean embedding
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$: Gaussian noise
- $t \in [0,1]$: time, $z_0 \sim p_{\text{noise}}$, $z_1 \sim p_{\text{data}}$
- 注意:这里flow direction与很多paper相反,ELF中 $t=0$ 是noise,$t=1$ 是clean

Velocity field:

$$\mathbf{v} = \frac{d\mathbf{z}}{dt} = \mathbf{x} - \boldsymbol{\epsilon}$$

**关键选择:x-prediction而不是v-prediction**

ELF的loss(公式1):

$$\mathcal{L}_{\text{MSE}} = \mathbb{E}_{t,\mathbf{x},\boldsymbol{\epsilon}} \left[ \frac{1}{(1-t)^2} \|\mathbf{x}_\theta(\mathbf{z}_t, t) - \mathbf{x}\|^2 \right]$$

- $\mathbf{x}_\theta$: network预测的clean embedding
- $\frac{1}{(1-t)^2}$: 从v-prediction到x-prediction的Jacobian变换系数,因为 $\mathbf{v} = (\mathbf{x} - \mathbf{z}_t)/(1-t)$,所以 $\|\mathbf{v} - \mathbf{v}_\theta\|^2 = \frac{1}{(1-t)^2}\|\mathbf{x} - \mathbf{x}_\theta\|^2$
- 当 $t \to 1$,这个系数会爆炸,但因为 $\mathbf{x}_\theta$ 也在逼近 $\mathbf{x}$,数值上是稳定的

**为什么x-prediction对ELF至关重要?**(参考 [13] Tianhong Li & Kaiming He的"Back to Basics")

Fig. 10的ablation显示:
- x-prediction在dim 512/768/1024都稳定
- v-prediction在512还行,768和1024退化
- ε-prediction在所有维度都崩了

**Intuition**: clean data通常位于高维空间中的低维manifold上。x-prediction直接target manifold上的点,gradient direction稳定;v-prediction target的是noise→data的差,方向变化大;ε-prediction target的是噪声,在高维下几乎没有结构信号。

而且x-prediction让shared-weight denoiser-decoder成为可能——两个任务都是"预测clean representation"。

参考链接:
- Back to Basics (Li & He): https://arxiv.org/abs/2511.13720
- Flow Matching original: https://arxiv.org/abs/2210.02747

### 3.3 Shared-weight Denoiser-Decoder (核心创新)

这个设计是ELF的精髓。标准Flow Matching在 $t=1$ 时 $\mathbf{z}_t \to \mathbf{x}$,没有信号可言。但语言必须输出discrete tokens,怎么办?

**ELF的trick**: 把最后一步(t=1)的discretization当作一个"decoding task",让同一个网络来做。

具体:
1. **Denoise branch (80% training)**: 标准FM,用 $\mathcal{L}_{\text{MSE}}$
2. **Decode branch (20% training)**: 在 $t=1$ 但用不同的corruption schedule,用 $\mathcal{L}_{\text{CE}}$

Decode branch的关键:因为 $\mathbf{z}_t \to \mathbf{x}$ 在 $t \to 1$,直接给网络看 $\mathbf{x}$ 没意义。所以引入per-token corruption:

$$\tilde{\mathbf{z}} = p\mathbf{x} + (1-p)\boldsymbol{\epsilon}$$

- $p$: per-token corruption level,从不同logit-normal采样 $p \sim \sigma(\mathcal{N}(P_{\text{mean}}=0.8, P_{\text{std}}^2=0.8^2))$
- 注意 $p$ 是per-token的,不同token有不同corruption level

然后CE loss:

$$\mathcal{L}_{\text{CE}} = \mathbb{E}_{\tilde{\mathbf{z}}} \left[ \text{CrossEnt}(W\mathbf{x}_\theta(\tilde{\mathbf{z}}), s) \right]$$

- $W$: learnable unembedding matrix, $\mathbb{R}^{|V| \times d}$
- $s$: ground-truth token

网络通过binary "mode" token判断denoise还是decode。

**为什么这个设计巧妙?**

1. **No separate decoder**: 不像latent diffusion [14]需要额外训一个decoder(AR decoder或NAR decoder),ELF把denoiser的最后一步自然延伸为decoder
2. **Robustness**: decode branch训练时看到的是corrupted embedding(不是clean的),这样inference时denoiser给出的imperfect embedding也能被decode
3. **Weight sharing的合理性**: 因为x-prediction下,两个任务都是"从corrupted z预测clean x",只是最后decode branch多了一个unembedding映射

参考链接:
- Latent Diffusion Models: https://arxiv.org/abs/2112.10752
- LDM for language (LD4LG): https://arxiv.org/abs/2210.17130

---

## 4. CFG with Self-conditioning

### 4.1 Self-conditioning [15]

Standard FM做一次forward得到 $\hat{\mathbf{x}}'$。Self-conditioning做两次:
1. First pass: $\hat{\mathbf{x}}' = \text{net}_\theta(\mathbf{z}_t, t)$
2. Second pass: $\hat{\mathbf{x}} = \text{net}_\theta(\mathbf{z}_t | \hat{\mathbf{x}}', t)$

实现:concatenate $[\mathbf{z}_t, \hat{\mathbf{x}}']$ 沿channel dim,然后linear project回原维度。

训练时50%概率用 $\hat{\mathbf{x}}'$,50%用zero;inference时用上一step的预测。

### 4.2 Training-time CFG

Standard CFG [16] inference时需要两次forward:
$$\mathbf{v}_{\text{cfg}}(\mathbf{z}_t | c) = \omega \mathbf{v}(\mathbf{z}_t | c) + (1-\omega) \mathbf{v}(\mathbf{z}_t | \emptyset)$$

- $c$: conditioning signal(在ELF里是self-conditioning的 $\hat{\mathbf{x}}'$)
- $\omega$: guidance scale
- $\emptyset$: unconditional

ELF采用training-time CFG [17, 18, 19]——直接让网络学 $\mathbf{v}_{\text{cfg}}$ 而不是 $\mathbf{v}$。Regression target变为:

$$\mathbf{v}_{\text{target}} = \mathbf{x} - \boldsymbol{\epsilon} + \left(1 - \frac{1}{\omega}\right)\left(\mathbf{v}_\theta^{\text{cfg}}(\mathbf{z}_t | t, c, \omega) - \mathbf{v}_\theta^{\text{cfg}}(\mathbf{z}_t | t, \emptyset, \omega)\right)$$

- $\omega$: sampled from power distribution in $[0.5, 5]$ bias toward smaller
- 当 $\omega=1$: 退化为standard FM
- $\mathbf{v}_\theta^{\text{cfg}}$: 网络直接输出post-combination velocity

这样inference时一次forward就够。

参考链接:
- Self-conditioning (Analog Bits): https://arxiv.org/abs/2208.04202
- Classifier-free guidance: https://arxiv.org/abs/2207.12598
- Mean Flows (training-time CFG): https://arxiv.org/abs/2505.13447
- Improved Mean Flows: https://arxiv.org/abs/2512.02012

---

## 5. Conditioning策略:In-context vs adaLN-Zero

DiT [20]标准做法是adaLN-Zero,把所有condition通过summation融合。但ELF有多个heterogeneous conditions:
- Time $t$ (continuous, [0,1])
- CFG scale $\omega$ (continuous, [0.5, 5])
- Model mode (binary: denoise/decode)

ELF采用**in-context conditioning** [19]: prepend control tokens到sequence前面,通过full self-attention交互。

具体prepend:
- 4 time tokens (用positional encoding编码连续值)
- 4 CFG scale tokens
- 4 mode tokens

**Intuition**: adaLN-Zero用summation融合conditioning,当condition类型多了,各condition的scale和modulation会互相干扰。In-context把condition当作"特殊token",attention可以学会如何混合它们。

Ablation Fig. 13显示in-context略好,而且把ELF-B从148M减到105M(参数减少30%)。

参考链接:
- DiT: https://arxiv.org/abs/2212.09748
- Improved Mean Flows (in-context conditioning): https://arxiv.org/abs/2512.02012

---

## 6. Sampling: ODE vs SDE-inspired

### 6.1 ODE Sampler (Euler)

$$\mathbf{z}_{t+\Delta t} = \mathbf{z}_t + \Delta t \cdot \mathbf{v}_\theta(\mathbf{z}_t, t)$$

其中 $\mathbf{v}_\theta = (\mathbf{x}_\theta - \mathbf{z}_t)/(1-t)$

### 6.2 SDE-inspired Sampler

Flow Matching的SDE形式 [21]可以理解为每步注入infinitesimal noise。ELF的简化近似:

```python
def sde_step(z, t, dt, gamma):
    e = randn_like(z)
    alpha = 1 - gamma * dt
    t_back = alpha * t
    z_back = alpha * z + (1 - alpha) * e
    x_hat = net(z_back, t_back, mode="denoise")
    v = (x_hat - z) / (1 - t)
    z = z + dt * v
    return z
```

- $\gamma$: noise re-injection scale, default 1.0
- $\alpha = 1 - \gamma \cdot \Delta t$: 时间回退因子
- $z_{\text{back}} = \alpha z + (1-\alpha)e$: 在回退后的时间点注入噪声
- 然后用回退后的state做denoiser prediction,但用original state更新

**Intuition**: SDE的noise re-injection能correct早期denoising错误。ODE是deterministic的,一旦某步预测偏差,后续会放大。SDE的随机性能"shake"轨迹,让模型有机会纠正。Fig. 5c显示SDE在few-step regime显著优于ODE。

参考链接:
- SiT (SDE for Flow Matching): https://arxiv.org/abs/2401.08765

---

## 7. Time Schedule: Logit-normal

ELF训练和inference都用logit-normal schedule:

$$t = \sigma(t'), \quad t' \sim \mathcal{N}(P_{\text{mean}}, P_{\text{std}}^2)$$

- $P_{\text{mean}} = -1.5$: 偏向noise端(因为前期denoising更难)
- $P_{\text{std}} = 0.8$
- $\sigma$: sigmoid function

Inference时,从同一分布采样 $T-1$ 个时间点排序形成grid,保证 $t_0=0, t_T=1$。

Ablation Fig. 15a显示logit-normal比uniform在few-step regime显著更好。

**Intuition**: noise → data的过程中,前期(noise dominant)的trajectory变化大,需要更密集的discretization;后期(data dominant)变化小,可以稀疏。logit-normal正好符合这个需求。

参考链接:
- Karras et al. EDM time schedule: https://arxiv.org/abs/2206.00364

---

## 8. 实验数据深度解析

### 8.1 主结果 (Fig. 7)

ELF-B (105M) vs baselines (170M) on OWT:
- 32 steps: Gen. PPL 24.08±0.16
- 比MDLM、Duo、FLM、LangFlow都好
- Training tokens: 45.2B vs 524B (10×更少!)

Tab. 5的token估计:
| Method | Effective tokens | Ratio |
|--------|-----------------|-------|
| MDLM | 524.3B | 11.6× |
| Duo | 524.3B | 11.6× |
| FMLM (distilled) | 576.7B | 12.8× |
| LangFlow | 524.3B | 11.6× |
| **ELF** | **45.2B** | **1.0×** |

### 8.2 条件生成 (Tab. 1)

WMT14 De-En (BLEU) and XSum (ROUGE):
| Model | De-En BLEU | R1 | R2 | R-L |
|-------|-----------|-----|-----|-----|
| AR | 25.2 | 30.5 | 10.2 | 24.4 |
| MDLM | 18.4 | 33.4 | 11.6 | 25.8 |
| Duo | 21.3 | 31.4 | 10.1 | 25.0 |
| E2D2 | 24.8 | 28.4 | 8.3 | 22.0 |
| **ELF-B** | **26.4** | **36.0** | **12.2** | **27.8** |

ELF在translation上甚至超过AR baseline!这很强。

### 8.3 Scaling (Tab. 7)

ELF-B (105M), ELF-M (342M), ELF-L (652M) with SDE/ODE, varying CFG scale:

64-step SDE, CFG=3:
- ELF-B: Gen. PPL 19.72, Entropy 5.10
- ELF-M: Gen. PPL 21.69, Entropy 5.18
- ELF-L: Gen. PPL 23.31, Entropy 5.28

注意ELF-L在CFG=3.5/4.0时还能继续提升(Gen. PPL 21.37),而小模型已经饱和甚至退化。说明大模型能承受更强guidance。

### 8.4 Bottleneck dim ablation (Fig. 11)

bottleneck dim 32/128/512:
- 32: lowest Gen. PPL but low entropy (overfitting manifold?)
- 128: best trade-off
- 512: high entropy but poor Gen. PPL (manifold hypothesis violated)

**Intuition**: 太小的bottleneck丢信息,太大的bottleneck让flow不再"贴着manifold走"。128是sweet spot,符合 [13]的observation。

### 8.5 Prediction target ablation (Fig. 10)

T5-small (512), T5-base (768), T5-large (1024):
- x-pred: 稳定,所有dim都好
- v-pred: 512 OK, 768+退化
- ε-pred: 全崩

### 8.6 Denoising mode probability (Fig. 12)

denoise vs decode比例:0.5/0.8/0.95
- 0.8最好

太多decode branch会让denoising能力不足;太少decode branch让discretization能力弱。

### 8.7 Optimizer (Fig. 14)

Muon [22] vs AdamW [23]:
- Muon learning rate 2e-3
- AdamW learning rate 1e-4

Muon在SDE下提升尤其明显。Muon是对hidden layer用Newton-Schulz iteration的optimizer,对矩阵参数特别有效。

参考链接:
- Muon optimizer: https://kellerjordan.github.io/posts/muon/

---

## 9. 与Concurrent Works的对比

ELF与同期工作对比(Tab. 2):

| Method | Process | State | Train per-step discr. | Sep. dec. |
|--------|---------|-------|----------------------|-----------|
| DFM [24] | FM | simplex | Yes | - |
| CFM [25] | FM | simplex | Yes | - |
| FLM [26] | FM | one-hot | Yes | - |
| LangFlow [27] | Bregman FM | learn emb | Yes | - |
| **ELF** | **FM** | **fix enc** | **No** | **No** |

**ELF的独特定位**:
- 唯一使用frozen encoder + 无per-step discretization + 无separate decoder的方法
- 其他flow-based LMs都引入了token-level CE supervision在中间步骤,这让trajectory被"拽"向token space,失去了continuous space的灵活性

参考链接:
- DFM (Discrete Flow Maps): https://arxiv.org/abs/2604.09784
- FLM (Flow Map Language Models): https://arxiv.org/abs/2602.16813
- LangFlow: https://arxiv.org/abs/2604.11748

---

## 10. 几个值得深挖的设计细节

### 10.1 为什么不直接用T5的token embedding?

T5的token embedding是non-contextual的,每个token一个固定vector。但flow matching需要clean data分布相对smooth,而one-hot token embedding是离散的"point cloud"。

用T5 encoder得到contextual embedding后,sequence中的每个position都有一个"被周围token contextualize"的representation,这在几何上更smooth,更适合continuous flow。

### 10.2 为什么normalize clean embeddings?

在corruption之前normalize:
$$\mathbf{x}_{\text{normalized}} = \frac{\mathbf{x} - \mu}{\sigma}$$

其中 $\mu, \sigma$ 是OWT的统计量。

**Intuition**: T5 embedding的scale可能跟Gaussian noise的scale不匹配。Normalize让clean embedding的scale接近 $\mathcal{N}(0, I)$,这样linear interpolation $z_t = t x + (1-t) \epsilon$ 的几何关系才合理。

### 10.3 Decode branch的noise scale差异

- OWT: noise scale 5.0
- Conditional generation: noise scale 1.0

OWT用更大noise是因为unconditional generation更难,需要decode branch能处理更corrupted的input。Conditional task有context提供信息,decode可以更"温和"。

### 10.4 Mode token的设计

binary mode token通过4个learnable embedding表示(为什么是4个?可能是为了redundancy让attention更容易区分)。

Denoise branch和decode branch在同一个batch里处理,用masking实现selective的corruption和loss计算。

---

## 11. ELF的局限与可能的延伸

### 11.1 Embedding space的"绑死"问题

ELF用frozen T5 encoder,意味着generation只能在T5 embedding space表示的"distribution"内。如果T5没见过的domain(比如code、特殊domain),可能受限。

可能的改进:
- Joint training encoder(论文提到效果稍差)
- 多encoder ensemble
- 重新pretrain一个更强的encoder

### 11.2 Sequence length限制

T5-small的position embedding限制了sequence length。论文用L=1024(OWT), L=128(WMT), L=1088(XSum)。长文本生成仍是挑战。

### 11.3 Inference steps仍多

ELF用32-64 steps,虽然比discrete DLMs少,但比AR LLM的1 token/step还是多。可能的改进:
- Consistency model [28]
- Mean Flow [19]的one-step generation
- Distillation

### 11.4 CFG的更深层利用

论文只用了self-conditioning作为CFG的condition。如果能引入semantic condition(比如topic、style),CFG会更powerful。

参考链接:
- Consistency Models: https://arxiv.org/abs/2303.01469

---

## 12. 我的intuition总结

ELF的精髓可以浓缩为几个insight:

1. **Flow Matching的endpoint是natural decoder**: $t=1$ 时 $z_t \to x$,这一步本身就是"从representation到clean representation"的映射,加上unembedding就是"从representation到token"。这种几何上的自然性让shared-weight成为可能。

2. **x-prediction是high-dimensional flow的key**: 在language embedding这种768+维的空间里,v-prediction和ε-prediction都因为manifold假设失效而崩溃。x-prediction直接target manifold,是唯一稳定的选择。

3. **Discretize only at the end preserves flow flexibility**: 每步discretize(像Diffusion-LM)等于强制trajectory经过token space,破坏了flow的几何性质。只在最后一步discretize让flow有最大的自由度。

4. **Continuous formulation = image toolkit免费可用**: CFG、Flow Matching、SDE sampler、Rectified Flow这些image domain的advance可以直接迁移,这是continuous DLM相对discrete DLM的根本优势。

5. **Self-conditioning + CFG的composability**: Self-conditioning天然提供了一个"signal",可以作为CFG的condition。这种composability让ELF能在unconditional generation也用CFG,而discrete DLM做不到。

ELF我认为是continuous DLM方向的turning point。它证明了只要设计得当,continuous DLM不仅能match discrete DLM,还能超越。这让我联想到image diffusion早期也是discrete(VQ-VAE [29])更强,但最终continuous (DDPM [30], LDM [14])胜出。Language可能也会走这条路。

参考链接:
- VQ-VAE: https://arxiv.org/abs/1711.00937
- DDPM: https://arxiv.org/abs/2006.11239

---

希望这些分析能build你的intuition,Karpathy!如果你想深挖某个具体方面(比如shared-weight的gradient flow、CFG的几何意义、或与MAR [31]/Llada这类masked diffusion的对比),我可以展开。

参考链接:
- ELF code: https://github.com/lillian039/ELF
- MAR (Autoregressive Image Generation without Vector Quantization): https://arxiv.org/abs/2406.11838
