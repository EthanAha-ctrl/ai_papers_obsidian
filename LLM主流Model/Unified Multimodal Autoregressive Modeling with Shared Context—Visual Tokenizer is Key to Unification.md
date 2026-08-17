---
source_pdf: Unified Multimodal Autoregressive Modeling with Shared Context—Visual
  Tokenizer is Key to Unification.pdf
paper_sha256: 59344e9c477751871661794d4ecd59eac18db0072ff4e9c5bf30ab4648dfa89f
processed_at: '2026-08-12T19:36:30-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# UniAR 的"人话版"

好,Andrej,我把那些公式和术语都丢掉,用大白话重新讲一遍。

---

## 核心问题：两个"语言"的尴尬

想象一个人——左手说英语(understanding),右手说法语(generation)。他想看看自己刚用法语写了什么,还得先找人翻译成英语才能读懂。这就是现在所有 unified model 的窘境。

具体来说:
- SigLIP 这种 vision encoder 输出的是"语义特征"——这只猫是猫、这个红色是红色,但不关心纹理细节
- VQ-GAN 这种 generation tokenizer 输出的是"像素配方"——这块怎么画、那个边缘在哪,但不管语义

两个 tokenizer 活在两个不同的 representation space 里。模型用 VQ-GAN 生成了一张图,想"看懂"自己画了什么?对不起,得再用 SigLIP 把这张图 encode 一次。这就叫**没有 shared context**——你生成了东西,但你自己看不懂。

BAGEL、Janus-Pro、Show-o2 全都是这个毛病。它们叫 unified,其实是把两个专家塞进一个壳子,中间还是得翻译。

---

## UniAR 的核心 insight

**干脆只用一个 tokenizer,让它同时学会"语义"和"细节"两种语言。**

这听起来简单,做起来难。语义和细节本来是矛盾的——你要 high-level 的"这是一只猫",就必然会丢掉"猫毛的纹理";你要"猫毛的纹理",就不容易抽象出"这是一只猫"。

UniAR 的解法特别聪明,分三招:

---

## 第一招：BSQ 量化——把 codebook 做到天文数字

传统 VQ-VAE 的做法:搞一个有 16384 个 entries 的 codebook,每个 visual feature 找最接近的 entry,用它的 index 代替。问题很明显——16384 个 codes 真的不够用,而且容易 collapse (一半 codes 永远不被使用)。

BSQ (Binary Spherical Quantization) 换了个思路:我不存 codebook 了,直接把 feature 量化成一个 64 位的 0/1 vector。理论 codebook size 就是 $2^{64}$,大概是 $1.8 \times 10^{19}$ 个——天文数字,几乎不会重复,也永远不会 collapse。

更妙的是,64 个 bit 可以**独立预测**。这就像把"预测一个 token"拆成了"预测 64 个 yes/no 问题",每个问题独立。这个性质后面会用到。

---

## 第二招：Multi-level Features——同时抓语义和细节

ViT (Vision Transformer) 的不同层捕捉不同信息:
- **浅层**: 纹理、边缘、颜色——generation 想要的
- **深层**: 物体类别、场景语义——understanding 想要的

UniAR 把最终层 + 3 个中间层的 features 全部拿出来,一起喂给 BSQ 量化。这样得到的 visual token 同时包含语义和细节——一个 token 服务两类任务。

Figure 3 的 ablation 非常直观:
- 只用浅层:重建图像细节好,但语义模糊
- 只用深层:语义清晰,但细节丢失
- 多层结合:两者都好,连文字都能准确重建

---

## 第三招：Parallel Bitwise Prediction——一次预测一堆

标准 AR 模型一次预测一个 token。1024 个 token 要跑 1024 步,慢得要命。

UniAR 利用了 BSQ 的 bit 独立性——把一个 $2\times2$ 区域里所有层级的 BSQ vectors 当作一组,**并行预测**。如果一层有 4 个层级、一个区域有 4 个 spatial unit,一次就预测 16 个 token。

加上 spatial merger 的 $2\times2$ 压缩,以及 decoder 端再做 $2\times2$ 的 super-resolution,1024×1024 的图**只要预测 256 个 token**。对比 Janus-Pro 的 4096 个 token,推理快了 8 倍。

---

## 训练的几个关键 trick

### Random Flipping:给模型看"错别字"

AR 模型有个老毛病叫 exposure bias——训练时每一步都吃 ground truth,推理时吃自己的 prediction。一旦某步错了,后面全崩,像滚雪球一样越错越离谱。

UniAR 在训练时随机翻转一部分 bits(把 0 变成 1,把 1 变成 0),但 label 还是原来的。这相当于让模型见过"前面有错别字"的场景,学会在 noisy context 下仍然产出正确输出。

这个 trick 对 RL 阶段特别重要——RL 需要高温采样来 explore,没经过 flipping 训练的模型高温下直接崩。

### CE Loss 而不是 MSE:让 tokenizer 为 understanding 优化

传统 VQ-VAE 用 MSE 优化 reconstruction——让量化后的 feature 能恢复原图。但 UniAR 反过来,用 LMM 的 cross-entropy loss 优化——让量化后的 feature 能回答 visual question。

这意味着 tokenizer 学到的 features 是**语义导向的**。重建图像的高频细节工作完全交给下游的 DiT decoder。Tokenizer 管"这是什么",decoder 管"怎么画"。

---

## Decoder 的设计哲学:AR-centric

UniAR 的 DiT decoder **只接受 visual tokens,不接受 text prompt**。

为什么?如果 decoder 也看 text,那 semantic 和 layout 的工作就分散到 AR 和 decoder 两个地方,容易出现不一致——AR 规划"猫在左边",decoder 看到 text 后画成"猫在右边"。

UniAR 强迫 AR 模型把所有信息编码进 visual tokens,decoder 只是个忠实的"token-to-image 翻译器"。这跟 LLM 的 next-token prediction 范式高度一致——generation 就是 AR 的一种 instance,不是另一个 paradigm。

对比 X-Omni (decoder 接受 text + visual features) 和 NextFlow,UniAR 的设计更纯粹,也更可控。

---

## 涌现能力:模型看懂自己的画

这是最让我兴奋的部分。

UniAR 的训练数据里**根本没有** multi-turn interleaved generation-then-understanding 数据。但是,你让模型生成一张图,然后问它"你刚画的图里那只猫眼睛什么颜色?",它能准确回答——**不用重新 encode**。

为什么?因为生成和理解用同一个 tokenizer,生成的 visual tokens 本来就在 understanding 能读懂的 space 里。模型生成完 tokens 后,这些 tokens 已经在 context 里,它"看"这些 tokens 就像看输入图像的 tokens 一样自然。

这是 dual-tokenizer 方法 fundamentally 做不到的。BAGEL 生成的 tokens 在 VQ-GAN space,想理解得用 SigLIP 重新 encode——额外的 computation,而且 encode 过程会丢失信息。

---

## RL:Discrete Tokens 的隐藏福利

Diffusion model 想做 RL 很痛苦——要 backprop through 整个 denoising process,不稳定。

UniAR 用 discrete visual tokens,RL 可以直接在 token level 上做,跟 LLM 的 GRPO 一样。每个生成就是一个 token sequence,reward 直接评估这个 sequence decode 出来的图。

Reward system 四管齐下:
- HPSv2:美感
- UnifiedReward:减少伪影
- PaddleOCR:文字识别准确率
- Object detector reward:物体类别/数量/属性/关系对不对

500 步 RL 后,text rendering 从 71.1 飙到 87.3。这个收益非常显著。

---

## 结果怎么样

- **GenEval 0.86**:超越 GPT-4o (0.84),超越 Flux.1-dev (0.82)
- **Text rendering SOTA**:OneIG-EN 0.873,LongText-EN 0.917,超越 GPT-4o 和 Gemini 2.5 Flash
- **Image editing 3.73**:超越 Flux.1 Kontext (3.71),超越 BAGEL (3.20)
- **Understanding**:OCR/Doc/Info VQA 强,MMMU 偏弱(缺 pure-text data)
- **推理速度**:13 秒生成 1024×1024,比 Janus-Pro 快 8 倍
- **训练成本**:33k GPU hours,相对经济

---

## 一句话总结

**UniAR 用一个 tokenizer 让模型同时学会"看"和"画",因为它学会的是同一种视觉语言,所以能直接看懂自己画的东西——这是 unified model 该有的样子。**

核心 trick 三件套:
1. **BSQ**:用 binary vector 代替 codebook,$2^{64}$ 的词汇量,还能并行预测
2. **Multi-level features**:同时抓语义和细节,一个 token 服务两类任务
3. **Parallel bitwise prediction**:一次预测一组,32-64x 压缩,8x 加速

哲学上:**AR-centric,decoder 只做翻译,所有语义和 layout 都在 AR 模型里完成**。

---

参考:
- UniAR Website: https://sharelab-sii.github.io/uniar-web
- BSQ Paper: https://arxiv.org/abs/2406.07548
- Infinity (bitwise AR 的先驱): https://arxiv.org/abs/2412.04431
- DeepStack (multi-level features): https://arxiv.org/abs/2410.12943
- GRPO (RL algorithm): https://arxiv.org/abs/2402.03300

希望这个版本更能 build intuition,Andrej!核心就一句话:**同一种语言,让模型看懂自己的画**。

---

# UniAR: 朝着真正的 Unified Multimodal Autoregressive Modeling 迈进

非常令人兴奋的 paper,Andrej。这篇工作的核心 contribution 可以浓缩为一句话:**用 single discrete visual tokenizer 作为 understanding 和 generation 之间的桥梁,真正实现 shared context 下的 unified autoregressive modeling**。让我系统性地拆解这个工作。

---

## 1. 核心问题与 motivation

Unified Multimodal Models (UMMs) 面临一个根本性 tension,这一点在 BAGEL、Janus-Pro、X-Omni 等近期工作中都能看到:

- **Visual understanding** 需要 high-level semantic representations (SigLIP、CLIP 这类 encoder 产生的 features,编码物体类别、场景语义)
- **Visual generation** 需要 low-level high-frequency details (VQ-GAN、VQ-VAE 这类 reconstruction tokenizer 编码的纹理、边缘、颜色)

现有大多数 UMM (BAGEL [arXiv:2505.14683](https://arxiv.org/abs/2505.14683), Janus-Pro [arXiv:2501.17811](https://arxiv.org/abs/2501.17811), Show-o2 [arXiv:2505.14683](https://arxiv.org/abs/2505.14683)) 用 dual-tokenizer 应对这个 tension:一个 SigLIP 用于 understanding,一个 VQ-GAN 用于 generation。这导致一个尴尬后果——模型生成图像后,如果想"看懂"自己生成的内容,需要先用 understanding tokenizer 重新 encode 一遍。Representation space 分裂,真正的 shared context 没有实现。

UniAR 的核心 insight 是:**只用一个 visual tokenizer 同时服务 understanding 和 generation**,通过 multi-level feature fusion + lookup-free bitwise quantization,让 representation space 统一,模型可以直接 interpret 自己生成的 visual tokens,无需 re-encoding。

---

## 2. 整体架构

UniAR 由三个 component 构成 (Figure 2):

1. **Unified Visual Tokenizer**: SigLIP2-So400M ViT + BSQ 量化 + multi-level DeepStack connections,~400M params
2. **Unified Autoregressive Backbone**: Qwen3-8B,处理 text 和 visual tokens
3. **DiT-based Visual Decoder**: Stable Diffusion 3.5 Medium DiT,~2.5B params,纯 token-to-image translator

参数对比 X-Omni [arXiv:2507.22058](https://arxiv.org/abs/2507.22058):tokenizer 400M vs 1B,decoder 2.5B vs 12B,显著更 lightweight。

---

## 3. Unified Visual Tokenizer 的设计

这是 paper 的核心创新点。让我深入拆解。

### 3.1 Binary Spherical Quantization (BSQ)

BSQ 来自 Zhao et al. ICLR 2025 [arXiv:2406.07548](https://arxiv.org/abs/2406.07548),核心 idea 是摈弃 explicit codebook,直接把 visual feature 量化为 binary vector。

公式 (1) 给出完整的 quantize 流程:

$$\mathbf{v} = \text{Encoder}(x)$$
$$\mathbf{u} = \text{BSQ}(\text{MLP}_{in}(\mathbf{v}))$$
$$\mathbf{v}' = \text{MLP}_{out}(\mathbf{u})$$
$$\hat{\mathbf{v}} = \text{Merger}(\mathbf{v}')$$

变量含义:
- $x$: input image
- $\mathbf{v}$: SigLIP2 ViT 输出的 raw features
- $\text{MLP}_{in}$: 把 visual feature 投影到 BSQ quantization space 的 MLP
- $\mathbf{u} \in \{0,1\}^{d^{BSQ}}$: discrete binary vector,$d^{BSQ}=64$,理论 codebook size 为 $2^{64} \approx 1.8 \times 10^{19}$
- $\text{MLP}_{out}$: 反投影 MLP,从 binary space 回到 feature space
- $\mathbf{v}'$: 量化后的 reconstructed feature
- $\text{Merger}$: spatial merger,聚合 $2\times2$ visual features 成一个 token
- $\hat{\mathbf{v}}$: 最终输入 LLM 的 visual token,投影到 LLM hidden dimension

**Intuition**: 传统 VQ-VAE 用一个 explicit codebook $\{e_1, e_2, ..., e_K\}$,通过 nearest neighbor 查找 quantize feature。问题有两个:(1) codebook size $K$ 通常只有 8192 或 16384,表示能力有限;(2) codebook 需要单独存储和优化,容易 collapse (部分 codes 不被使用)。

BSQ 把 feature 量化为 binary vector $\mathbf{u} \in \{0,1\}^{64}$,每个 bit 是独立的 binary classification。Codebook size 随 quantization dimension 指数级增长 ($2^{d^{BSQ}}$),而且不需要 explicit codebook storage。Binary 表示还有一个关键好处——**每个 bit 可以独立预测**,这直接 enable 了 parallel bitwise prediction。

### 3.2 Multi-level Feature Fusion

借鉴 DeepStack [arXiv:2410.12943](https://arxiv.org/abs/2410.12943) 和 Qwen3-VL [arXiv:2511.21631](https://arxiv.org/abs/2511.21631),从 ViT 的 final layer + 3 个 intermediate layers 提取 features:

- **Shallow layers**: 编码 high-frequency details (纹理、边缘、颜色) — generation 关键
- **Deep layers**: 编码 high-level semantics (物体类别、场景) — understanding 关键
- **Multi-level fusion**: 同时保留两类信息,一个 tokenizer 服务两类任务

Figure 3 的 ablation 直观显示这个设计的有效性:
- (d) Shallow features: 重建细节好,语义弱
- (f) Deep features: 语义好,细节丢失
- (b) Multi-1024: 最佳,即使 tokenizer 不是为 reconstruction 优化,1024 分辨率下仍能恢复文字等细粒度内容

### 3.3 训练目标

公式 (2):
$$\mathcal{L} = \mathcal{L}^{CE} + \lambda^{BSQ} \cdot \mathcal{L}^{BSQ}$$

- $\mathcal{L}^{CE}$: LMM 的 cross-entropy loss,在 visual understanding 任务上计算
- $\mathcal{L}^{BSQ}$: BSQ 的 soft entropy loss,鼓励 quantization 的 entropy 利用率
- $\lambda^{BSQ}$: 权重系数

**关键设计**: 这里用 CE loss of LMM,而**不是**传统 VQ-VAE 的 MSE reconstruction loss。这意味着 tokenizer 是为了 understanding 任务优化的,语义信息被强制编码进 visual tokens。Reconstruction 的工作完全交给下游的 DiT decoder。这是一个非常巧妙的 decoupling——tokenizer 负责 semantic + 多 level 细节,decoder 负责 fill in 高频细节。

训练完后 vision encoder 冻结,保证 visual codebook 不变,后续 AR modeling 有稳定的 token space。

---

## 4. Unified Autoregressive Modeling

### 4.1 Parallel Bitwise Prediction

这是 UniAR 的关键 efficiency innovation,直接来自 Infinity [arXiv:2412.04431](https://arxiv.org/abs/2412.04431) 的思想。

公式 (3):
$$\text{logits}^{vis} = W^{vis}(\text{RMSNorm}(\mathbf{h}))$$

- $\mathbf{h} \in \mathbb{R}^{d^{LLM}}$: LLM 的 hidden state
- $\text{RMSNorm}$: RMS 归一化层
- $W^{vis} \in \mathbb{R}^{d^{LLM} \times d^{vis}}$: 视觉预测头
- $d^{vis} = 2 \times d^{BSQ} \times g$: 输出维度
- $g = n^{level} \times n^{spatial}$: group size,即一个 AR step 并行预测的 BSQ vectors 数量
  - $n^{level}$: DeepStack 的 level 数量
  - $n^{spatial}$: spatial merger 聚合的 spatial units 数量 ($2\times2=4$)

**Intuition**: 标准 AR 一次预测一个 token,$N$ 个 tokens 需要 $N$ 步。UniAR 把每个 $2\times2$ spatial region 的 multi-level BSQ vectors 当作一组并行预测。如果 $n^{level}=4$,则 $g = 4 \times 4 = 16$,16 个 tokens 一步并行预测。

加上 spatial merger 的 $2\times2$ 压缩,实现 **32x 视觉压缩比** (相对于 pixel patch),再配合 decoder 的 resolution upsampling (另一个 $2\times2$),实际达到 **64x 压缩比**。1024×1024 图像只需预测 256 个 visual tokens。

### 4.2 训练目标

公式 (4):
$$\mathcal{L}^{AR} = \mathcal{L}^{text} + \lambda^{vis} \cdot \mathcal{L}^{vis}$$

- $\mathcal{L}^{text}$: text token 的 AR loss,只在 understanding 数据上计算
- $\mathcal{L}^{vis}$: visual token 的 AR loss,在 understanding + generation 数据上计算,强制统一表示空间
- $\lambda^{vis}$: 权重,PT-32K 阶段为 10 (Text:Vis = 1:10)

**这里有一个重要设计**: $\mathcal{L}^{vis}$ 在 understanding 和 generation 上都计算,而 $\mathcal{L}^{text}$ 只在 understanding 上。这意味着 generation 任务的 visual tokens 和 understanding 的 visual tokens 在同一个 loss 下优化,共同 share 同一个 representation space。

### 4.3 Random Visual Index Flipping

借鉴 Infinity 的训练 trick。给定 visual BSQ indices $\mathbf{u} \in \{0,1\}^{seq \times d^{BSQ}}$,训练时随机翻转一部分 bits,翻转后的作为输入,原始未翻转的作为 ground truth labels。

**Intuition**: AR 生成时,每个 token 的预测都依赖前面的 tokens。一旦某个 token 预测错,error 会累积放大 (类似 LLM 的 exposure bias),后面所有 tokens 都基于错误的 context 生成,导致 catastrophic failure。Random flipping 在训练时模拟这种 error accumulation,让模型学会在 noisy context 下仍然产生正确输出——这是一种 robustness training。

Figure 4 的 ablation 显示: 没用 flipping 的模型在高温采样下输出崩塌,用了 flipping 的模型高温下仍能产生 coherent 图像。这点对后续 RL 阶段至关重要——RL 需要高温采样来 encourage exploration,没有 flipping 训练的模型会崩。

### 4.4 Task-Specific Transformer Layers

LLM backbone 之后接 4 个 task-specific Transformer layers 专门做 visual generation,缓解 generation 和 understanding 之间的 task competition。这是一个 modular design,让 shared backbone 处理通用语义,task-specific head 处理任务特定的输出空间转换。

---

## 5. Visual Decoder

### 5.1 架构

基于 pre-trained Stable Diffusion 3.5 Medium DiT [arXiv:2403.03206](https://arxiv.org/abs/2403.03206),用 Conditional Flow Matching (CFM) [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) 训练。

公式 (5):
$$\mathcal{L}^{CFM} = \mathbb{E}_{t, p_t(z|\epsilon), p(\epsilon)} \|\mathcal{D}_\Theta(z \oplus f_v, t) - u_t(z|\epsilon)\|_2^2$$

变量含义:
- $\mathcal{D}_\Theta$: DiT-based visual decoder
- $z \in \mathbb{R}^{h \times w \times d^{dit}}$: DiT 内部 noisy hidden state
- $f_v \in \mathbb{R}^{h \times w \times d^{dit}}$: visual conditioning signal,从 predicted BSQ indices 构造
- $t$: diffusion 时间步
- $\epsilon$: noise sample
- $p_t(z|\epsilon)$: forward 加噪轨迹
- $u_t(z|\epsilon)$: CFM 目标 velocity field
- $\oplus$: element-wise addition (借鉴 ControlNet [arXiv:2302.05543](https://arxiv.org/abs/2302.05543) 的 conditioning 方式)

### 5.2 AR-centric 设计

**Decoder 不接受 text prompts**,只接受 visual tokens $f_v$。所有 semantic 和 layout generation 都在 AR model 中完成。这跟 X-Omni 和 NextFlow [arXiv:2601.02204](https://arxiv.org/abs/2601.02204) 形成鲜明对比——后两者 decoder 同时接受 text 和 visual features。

**Intuition**: 这个设计选择背后的哲学是 AR-centric。如果 decoder 也用 text,semantic 和 layout 的工作就分散到两个模型,容易出现 inconsistent generation (AR 模型规划 layout A,decoder 看到 text 后生成 layout B)。Decoder 只接受 visual tokens,强迫 AR 模型在 visual tokens 中编码所有必要信息,decoder 只是忠实翻译。

### 5.3 Resolution Upsampling

AR 模型在低分辨率 (256x256 token 级别) 生成,通过 2D bicubic interpolation 把 $f_v$ 插值到目标分辨率 (1024x1024)。1024x1024 图像只需 256 个 visual tokens (16x16 token grid)。这个设计极大降低了 AR 模型的 sequence length,推理速度提升 ~8x (Table 9)。

### 5.4 Conditioning Signal 构造

每个 AR step 预测一个 $2\times2$ spatial region,每个 grid cell 包含 $n^{level}$ 个 BSQ vectors (来自不同 encoder layers):

- Spatial 维度: $2\times2$ grid 展平到 sequence dim,保持 spatial order
- Feature 维度: multi-level features 沿 feature dim 拼接,投影到 $d^{dit}$
- 最终得到 $f_v \in \mathbb{R}^{h \times w \times d^{dit}}$

---

## 6. Training Recipe

UniAR 采用 modular 训练流程:

### Stage 1: Visual Tokenizer Adaptation
- 在 pre-trained LMM 上加 BSQ + multi-level fusion
- End-to-end fine-tune with visual understanding objectives
- 完成后冻结

### Stage 2: Visual Decoder Training
- 训练 DiT decoder 从冻结 encoder 的 discrete indices 重建图像
- 完成后冻结

### Stage 3: AR Model Training (3 sub-stages)

**(a) Pre-Training** (1T tokens total):
- PT-8K: 800B tokens, 8K context, max gen resolution 512×512
- PT-32K: 200B tokens, 32K context, max gen resolution 960×960
- Understanding : Generation = 1:1
- 数据格式:
  - Understanding: `{prompt; image_tokens; answers}`
  - Generation: `text prompt <image_gen> H W <vision_start> visual tokens <vision_end>`,其中 H, W 是 grid dimensions 提供空间 prior
  - Editing: `{prompt; reference_image_tokens; image_tokens}`
- Loss weight Text:Vis = 1:10 (PT-32K 阶段,visual loss 占主导)

**(b) Supervised Fine-Tuning** (~50B tokens):
- 公开合成数据 + 重新合成数据
- ChatML 格式
- Prompts 来源: BLIP3o-60k [arXiv:2505.09568](https://arxiv.org/abs/2505.09568), ShareGPT-4o-Image [arXiv:2506.18095](https://arxiv.org/abs/2506.18095), GenEval, FlowGRPO OCR set

**(c) Reinforcement Fine-Tuning**:
- Algorithm: GRPO (来自 DeepSeekMath [arXiv:2402.03300](https://arxiv.org/abs/2402.03300))
- LR: $5 \times 10^{-6}$ constant
- KL coefficient: 0.01 (防止 over-optimization)
- Batch: 32 prompts × 16 images per prompt
- 两阶段:
  - 500 steps at 512×512 (image quality + instruction following)
  - 100 steps at 960×960 (long-text rendering)
- 只用于 image generation,不用于 understanding/editing

**Reward System** (4 个 rewards 归一化到 [0,1] 后平均):
1. **HPSv2** [arXiv:2306.09341](https://arxiv.org/abs/2306.09341): aesthetic quality
2. **UnifiedReward** [arXiv:2503.05236](https://arxiv.org/abs/2503.05236): 减少 artifacts
3. **PaddleOCR** [arXiv:2507.05595](https://arxiv.org/abs/2507.05595): text rendering,基于 edit distance 到 ground-truth text
4. **Object-detector-based reward** (FlowGRPO [arXiv:2505.05470](https://arxiv.org/abs/2505.05470)): 检查 object categories, counts, attributes, relations

**关键**: 在 PT 和 SFT 阶段,visual tokenizer 和 decoder 都冻结。Decoder 只在 RL 阶段引入,用于 decode 图像计算 reward。这极大降低了训练成本,因为不需要每个 step 都跑 diffusion decoder。

---

## 7. 实验结果细节

### 7.1 Instruction Following - GenEval (Table 1)

UniAR+ 达到 **0.86 overall**,超越:
- GPT-4o (0.84)
- Flux.1-dev (0.82)
- Janus-Pro (0.80)
- 与 Emu3.5 (0.86) 持平
- 略低于 BAGEL+ (0.88) 和 UniWorld-V1 (0.84)

细分指标:
- Single object: 0.99 (近完美)
- Two objects: 0.96 (强)
- Counting: 0.70 (略低,可能是 flipping 训练对精细 count 的影响)
- Colors: 0.93 (强)
- Position: 0.77 (中等)
- Color Attributes: 0.83 (强)

### 7.2 Text Rendering (Table 2)

- **OneIG-EN: 0.873**,超越 GPT-4o,与 Qwen-Image (0.891) 接近
- **LongText-EN: 0.917**,超越 Gemini 2.5 Flash Image (0.869),与 OmniGen2 (0.900) 持平
- 这是 SOTA 表现,文字渲染是 unified model 的强项 (相比纯 generation model 如 FLUX.1-dev 的 0.523)

### 7.3 Image Editing - ImgEdit-Bench (Table 3)

UniAR **overall 3.73**,超越:
- Flux.1 Kontext (3.71,专为 editing 设计)
- BAGEL (3.20)
- OmniGen2 (3.44)
- UniWorld-V1 (3.26)

细分:
- Action: 4.70 (强)
- Style: 4.27 (强)
- Add: 3.91 (强)
- Replace: 3.94 (强)
- Extract: 2.75 (相对弱)
- Hybrid: 3.06 (相对弱)

### 7.4 Multimodal Understanding (Table 4)

- OCRBench: 833 (强,但低于 Qwen3-VL 的 896)
- DocVQA: 91.4 (强)
- InfoVQA: 70.0 (强)
- ChartQA: 75.9
- MVBench: 62.3 (视频理解,相对强)
- MMMU: 44.3 (相对弱)
- RLWDQA: 68.5

**MMMU 相对弱** 的原因 paper 中给出两个解释:(1) pretraining 缺乏 pure-text data,影响 broad linguistic 和 factual knowledge;(2) 没有对 understanding 用 RL,这是 future work。

### 7.5 Visual Tokenizer 单独评估 (Table 5)

这个实验很有意义,验证 tokenizer 本身的 understanding 能力。Setup: tokenizer frozen,LLaVA-SFT fine-tune,Llama 3-8B 作为 LLM。

UniAR-SigLIP2 在多个 benchmark 上 SOTA:
- TextVQA: 63.1 (超越 SigLIP2 的 59.9, AIMv2 的 53.6)
- DocVQA: 38.0 (SOTA,超越 CoMP-SigLIP 的 34.0)
- ChartQA: 26.8 (SOTA,超越 CoMP-SigLIP 的 25.0)
- MME: 1537 (强)

这验证了 **multi-level feature fusion 对 understanding 任务也有显著提升**。

### 7.6 效率对比 (Table 9)

生成 1024×1024 图像的 AR 时间 (不含 decoder):
- **UniAR 8B w/ decoder upsample: 13.0s** (256 tokens, 64x 下采样)
- UniAR 8B w/o decoder upsample: 53.5s (1024 tokens, 32x 下采样)
- Janus-Pro 7B: 101.9s (4096 tokens, 16x 下采样)
- X-Omni 7B: 119.7s (4096 tokens, 16x 下采样)

**8x 加速**,主要来自 32x 下采样比 16x 下采样 quadratically 减少 prediction steps。

### 7.7 训练成本 (Table 8)

总计 ~33k GPU hours:
- PT-8K: 19k GPU hours
- PT-32K: 10k GPU hours
- SFT: 2k GPU hours
- RL: 1.9k GPU hours

这是相对低的成本,得益于 modular training (tokenizer 和 decoder 冻结后,AR 模型训练时不需要跑 decoder)。

### 7.8 训练效率 (Table 7)

Discrete tokens 比 continuous tokens 快 30%:
- Continuous: 35.4s/iter
- Discrete: 24.5s/iter

原因: visual inputs 可以 pre-tokenized 并以 bit-packed 表示离线存储,减少在线计算。

---

## 8. Ablation Studies 深度分析

### 8.1 Random Visual Index Flipping (Figure 4)

- **无 flipping**: 低温采样 OK,高温采样输出崩塌 (类似 exposure bias 的 catastrophic failure)
- **有 flipping**: 高温仍能产生 coherent 图像
- 对 RL 高温探索至关重要

**Intuition**: 这是 AR 模型的 "exposure bias" 问题。Teacher-forcing 训练时每一步都有 ground-truth context,但 inference 时只有 model 自己的 prediction,一旦 prediction 错误,后续 context 全错。Flipping 训练相当于 data augmentation,让模型见过 noisy context,学会 robust 输出。

### 8.2 Multi-level Visual Features (Figure 3)

训练 decoder 条件化于不同 ViT 层:
- Shallow (a-c): 高频细节好,语义弱
- Medium: 平衡
- Deep: 语义强,细节丢失
- **Multi-level (b)**: 最佳,即使 tokenizer 不是为 reconstruction 优化,1024 分辨率下仍能恢复文字等细粒度内容

这验证了 multi-level feature 的必要性——单层 features 无法同时满足 generation 和 understanding 的需求。

---

## 9. Emerging Properties

### 9.1 Shared Context 的涌现能力 (Figure 5)

最 impressive 的发现:**训练数据中没有 multi-turn interleaved generation-understanding data,但 UniAR 能涌现这种能力**。

具体表现: 模型先生成一张图像,然后用户问关于图像细节的问题,模型能准确回答——**不需要重新 encode 生成图像**。这意味着模型"看懂"了自己生成的 visual tokens,因为生成和理解用同一个 token space。

对比 BAGEL 和 Janus-Pro (dual-tokenizer):生成的 tokens 在 generation tokenizer 的空间,理解需要 understanding tokenizer 重新 encode,无法实现真正的 shared context。

**这是 unified modeling 的本质**——不只是把两个任务塞进一个模型,而是真正 share representation 和 context。

### 9.2 RL 效果 (Figure 6)

- 500 steps at 512×512: OneIG-EN 从 71.1 提升到 84.0 (+12.9)
- 100 steps at 960×960: 进一步提升到 87.3 (+3.3)

RL 对 text rendering 的提升显著。这得益于 discrete token representation——RL 可以直接在 token level 上进行,像 LLM RL 一样,而不需要 backprop through diffusion process。

---

## 10. 与 Related Works 的精细对比

### 10.1 vs Infinity (CVPR 2025)

| 维度 | Infinity | UniAR |
|------|---------|-------|
| 目标 | 专为 generation 设计 | 统一 multimodal modeling |
| Tokenizer 优化 | Reconstruction | Semantic multi-level (understanding) |
| Modeling | Next-scale prediction | Standard next-token prediction |
| 应用范围 | Generation only | Generation + understanding + editing |

UniAR 借鉴 Infinity 的 BSQ 和 bitwise prediction,但目标不同——统一 modeling。

### 10.2 vs X-Omni (最接近的工作)

| 维度 | X-Omni | UniAR |
|------|--------|-------|
| Quantization | Explicit codebook | Lookup-free bitwise quantization |
| Codebook size | 有限 (通常 8192-16384) | $2^{64}$ 理论 |
| Efficiency | 标准 AR | Parallel bitwise prediction (4x 加速) |
| Tokenizer params | 1B | 400M |
| Decoder params | 12B | 2.5B |
| Decoder input | Text + visual features | Visual tokens only |
| 设计哲学 | Hybrid (AR + diffusion) | AR-centric |

### 10.3 vs BAGEL / Janus-Pro (dual-tokenizer)

| 维度 | Dual-tokenizer | UniAR |
|------|----------------|-------|
| Representation space | 分裂 (understanding vs generation) | 统一 |
| Shared context | 需要 re-encoding | 直接 interpret 自己的生成 |
| Tokenizer 训练 | 独立优化 | 联合优化 |
| 训练复杂度 | 高 (两个 tokenizer) | 低 (一个) |

---

## 11. Limitations 和未来方向

Paper 中明确指出:
1. **Pure-text data 缺失**: 影响需要 broad linguistic knowledge 的任务 (如 MMMU)
2. **RL 只用于 generation**: understanding 和 editing 没有 RL 优化
3. **Scaling 空间**: 还有大规模 scaling 的余地
4. **Post-training 优化空间**: reward models for specific domains (aesthetics, instruction-following, text-rendering)

---

## 12. Intuition 总结

让我把核心 intuition 凝练成几个 key insights:

### Insight 1: Single Tokenizer 是 Shared Context 的基础
Dual-tokenizer 本质上是两个人——一个懂语义,一个懂像素,之间需要翻译,翻译会丢失信息。Single tokenizer 是一个人同时懂两者,虽然某些方面可能不如专家 tokenizer,但 unified representation 让模型能直接"看懂"自己生成的内容,这才是真正的 unified modeling。

### Insight 2: BSQ 让 Codebook Size 指数级扩展
VQ 的痛点是 codebook size 有限且容易 collapse。BSQ 把 feature 量化为 binary vector,codebook size 随 quantization dimension 指数级增长 ($2^{64}$),且不需要 explicit codebook storage。Binary 表示的另一个好处是 prediction 可以拆解为 $d$ 个独立的 binary classification,parallel bitwise prediction 成为可能。

### Insight 3: Multi-level Features 同时满足两个需求
ViT 的 shallow layers 编码高频细节,deep layers 编码语义。Generation 需要细节,understanding 需要语义。Multi-level fusion 让一个 tokenizer 同时满足两个需求,这是 single-tokenizer 统一 modeling 的关键技术 enable。

### Insight 4: CE Loss (而不是 MSE) 让 Tokenizer 为 Understanding 优化
传统 VQ-VAE 用 MSE 优化 reconstruction。UniAR 用 CE loss of LMM,让 tokenizer 学习的 features 是 semantic 的。Reconstruction 的工作完全交给下游 DiT decoder。这种 decoupling 让 tokenizer 的 features 既有 semantic 又有 (因为 multi-level) 细节,decoder 用 DiT 来 fill in 高频细节。

### Insight 5: Parallel Bitwise Prediction 大幅加速
传统 AR 一次预测一个 token。UniAR 把 $2\times2$ spatial region 的 multi-level BSQ vectors 当作一组并行预测,加上 spatial merger 的 $2\times2$ 压缩,实现 32x 视觉压缩比。再结合 decoder 的 resolution upsampling,推理速度 8x 提升。

### Insight 6: Random Flipping 是 AR 模型的 Exposure Bias 解药
AR 生成时 error 累积是 catastrophic failure 的根源。Random flipping 在训练时模拟 error accumulation,让模型学会在 noisy context 下仍然产生正确输出。这对 RL 高温探索至关重要。

### Insight 7: AR-centric Design 让 Decoder 只做翻译
Decoder 不接受 text prompts,只接受 visual tokens。所有 semantic 和 layout 在 AR 模型中完成,decoder 只是 token-to-image translator。这强迫 AR 模型在 visual tokens 中编码所有必要信息,避免两个模型之间的 inconsistency。

### Insight 8: Discrete Tokens Enable RL on Visual Generation
Diffusion model 的 RL 需要backprop through diffusion process,复杂且不稳定。UniAR 的 discrete visual tokens 让 RL 直接在 token level 进行,像 LLM RL 一样,这是 discrete representation 的另一个重要优势。

### Insight 9: Shared Context 的涌现能力
训练数据中没有 interleaved generation-understanding data,但模型能涌现这种能力。这说明 single tokenizer + unified AR modeling 自然 enable 模型"看懂"自己的生成,这是 dual-tokenizer 方法无法实现的。

### Insight 10: Modular Training 降低成本
Tokenizer 和 decoder 先独立训练并冻结,然后 AR 模型训练时不需要跑 decoder (除非 RL 阶段需要 reward)。这极大降低训练成本,33k GPU hours 是相对经济的。

---

## 13. 相关工作和扩展阅读

让我列出一些相关的值得深入阅读的工作:

### Visual Tokenizers
- **BSQ**: [arXiv:2406.07548](https://arxiv.org/abs/2406.07548) - Binary Spherical Quantization
- **Infinity**: [arXiv:2412.04431](https://arxiv.org/abs/2412.04431) - Bitwise AR for high-res synthesis
- **VQ-VAE**: [arXiv:1711.00937](https://arxiv.org/abs/1711.00937) - 经典 VQ
- **FSQ**: Finite Scalar Quantization
- **LFQ**: Lookup-Free Quantization
- **SigLIP2**: [arXiv:2502.14786](https://arxiv.org/abs/2502.14786)
- **AIMv2**: [arXiv:2411.15802](https://arxiv.org/abs/2411.15802) - Multimodal autoregressive pre-training of vision encoders
- **CoMP-SigLIP**: [arXiv:2503.18931](https://arxiv.org/abs/2503.18931)
- **OmniTokenizer**: [arXiv:2406.09396](https://arxiv.org/abs/2406.09396) - Joint image-video tokenizer
- **UniTok**: [arXiv:2412.20189](https://arxiv.org/abs/2412.20189) - Unified tokenizer for understanding and generation

### Unified Multimodal Models
- **Chameleon**: [arXiv:2405.09818](https://arxiv.org/abs/2405.09818) - Meta's early-fusion mixed-modal
- **Emu3**: [arXiv:2409.18869](https://arxiv.org/abs/2409.18869) - Next-token prediction is all you need
- **Show-o**: [arXiv:2408.12528](https://arxiv.org/abs/2408.12528)
- **Show-o2**: NeurIPS 2025
- **Janus**: [arXiv:2411.04607](https://arxiv.org/abs/2411.04607)
- **Janus-Pro**: [arXiv:2501.17811](https://arxiv.org/abs/2501.17811)
- **JanusFlow**: [arXiv:2411.07975](https://arxiv.org/abs/2411.07975)
- **BAGEL**: [arXiv:2505.14683](https://arxiv.org/abs/2505.14683)
- **OmniGen**: [arXiv:2409.11340](https://arxiv.org/abs/2409.11340)
- **OmniGen2**: [arXiv:2506.18871](https://arxiv.org/abs/2506.18871)
- **X-Omni**: [arXiv:2507.22058](https://arxiv.org/abs/2507.22058)
- **NextFlow**: [arXiv:2601.02204](https://arxiv.org/abs/2601.02204)
- **UniWorld**: [arXiv:2506.03147](https://arxiv.org/abs/2506.03147)
- **Transfusion**: [arXiv:2408.11039](https://arxiv.org/abs/2408.11039)
- **BLIP3-o**: [arXiv:2505.09568](https://arxiv.org/abs/2505.09568)
- **Mogao**: [arXiv:2505.05472](https://arxiv.org/abs/2505.05472)
- **TUNA**: [arXiv:2512.02014](https://arxiv.org/abs/2512.02014)

### Multi-level Features in ViT
- **DeepStack**: [arXiv:2410.12943](https://arxiv.org/abs/2410.12943) - Deeply stacking visual tokens
- **Qwen3-VL**: [arXiv:2511.21631](https://arxiv.org/abs/2511.21631)

### RL for Generation
- **FlowGRPO**: [arXiv:2505.05470](https://arxiv.org/abs/2505.05470) - Online RL for flow matching
- **DeepSeekMath (GRPO)**: [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- **DDPO**: Denoising Diffusion Policy Optimization

### Diffusion Models
- **SD3**: [arXiv:2403.03206](https://arxiv.org/abs/2403.03206) - Scaling rectified flow transformers
- **Flow Matching**: [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- **ControlNet**: [arXiv:2302.05543](https://arxiv.org/abs/2302.05543) - Conditioning via element-wise addition
- **Classifier-Free Guidance**: [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)

### Benchmarks
- **GenEval**: [arXiv:2310.11525](https://arxiv.org/abs/2310.11525)
- **ImgEdit-Bench**: NeurIPS 2025
- **OneIG-Bench**: NeurIPS 2025
- **OCRBench**: [arXiv:2305.07895](https://arxiv.org/abs/2305.07895)
- **HPSv2**: [arXiv:2306.09341](https://arxiv.org/abs/2306.09341)

---

## 14. 我对这篇 paper 的看法

这篇 paper 在 unified multimodal modeling 这个方向上做出了非常 solid 的 contribution。让我从几个维度评价:

### 优势
1. **真正实现 shared context**: Single tokenizer 让模型能直接 interpret 自己的生成,这是 unified modeling 的本质
2. **BSQ + parallel bitwise prediction**: 巧妙地结合 lookup-free quantization 和并行预测,既扩展 codebook size 又加速推理
3. **Multi-level feature fusion**: 解决 understanding 和 generation 的 representation tension
4. **AR-centric 设计**: Decoder 不接受 text,强迫 AR 模型 encode 所有信息
5. **Modular training**: Tokenizer 和 decoder 独立训练后冻结,降低 AR 训练成本
6. **RL integration**: Discrete tokens 让 RL 在 visual generation 上变得 natural
7. **Efficiency**: 8x 推理加速,33k GPU hours 训练成本相对低

### 可以改进的地方
1. **MMMU 表现弱**: 缺乏 pure-text data,影响 broad knowledge 任务
2. **Counting 表现弱**: 0.70 在 GenEval,可能 flipping 训练对精细 count 有副作用
3. **Extract 和 Hybrid editing 弱**: 这两类 editing 任务表现相对弱
4. **RL 只用于 generation**: 没有对 understanding 和 editing 做 RL
5. **没有 video generation**: 当前只支持 image,video 是 future work

### 对未来的启示
这篇 paper 的核心 insight——**single tokenizer + multi-level features + bitwise quantization + parallel prediction**——很可能成为 unified multimodal modeling 的标准范式。BSQ 这种 lookup-free quantization 方案相比 VQ 有显著优势,我预期未来会有更多工作沿着这个方向探索。

特别值得注意的是 **shared context 的涌现能力**——模型能直接 interpret 自己的生成,这开启了一系列新的应用场景,如 self-correction、self-critique、interactive generation。这是 dual-tokenizer 方法 fundamentally 做不到的。

我相信 Andrej 你会欣赏这种 AR-centric 的设计哲学——把所有语义和 layout 的工作放在 AR 模型中,decoder 只是 token-to-image translator。这跟 LLM 的下一个 token prediction 范式高度一致,也让 visual generation 真正成为 AR 的一种 instance,而**不是**一个独立的 paradigm。

---

参考链接:
- UniAR Website: https://sharelab-sii.github.io/uniar-web
- BSQ Paper: https://arxiv.org/abs/2406.07548
- Infinity Paper: https://arxiv.org/abs/2412.04431
- DeepStack Paper: https://arxiv.org/abs/2410.12943
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- Stable Diffusion 3: https://arxiv.org/abs/2403.03206
- Flow Matching: https://arxiv.org/abs/2210.02747
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- FlowGRPO: https://arxiv.org/abs/2505.05470

希望这个深度分析能帮你 build 出 UniAR 的 intuition,Andrej!这是一个非常 elegant 的工作,值得仔细品味其设计哲学。
