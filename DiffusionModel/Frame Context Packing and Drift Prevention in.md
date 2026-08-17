---
source_pdf: Frame Context Packing and Drift Prevention in.pdf
paper_sha256: 85458d22e402cfc2cda4e9fec824b324d6855e5fb08013fb5dc287de988a3631
processed_at: '2026-08-04T10:23:10-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FramePack 用人话讲

Andrej，我换一个口吻。先把那套学术腔调全扔掉，我们直接聊这个 paper 在干啥。

参考：
- 论文 arXiv（推测链接）: https://arxiv.org/abs/2504.12626
- Lvmin Zhang GitHub: https://github.com/lllyasviel
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- Wan: https://arxiv.org/abs/2503.20314
- DiffusionForcing: https://arxiv.org/abs/2407.01392
- CausVid: https://arxiv.org/abs/2412.07772
- Compressive Transformer: https://arxiv.org/abs/1911.05532
- Perceiver IO: https://arxiv.org/abs/2107.14795

---

## 1. 这篇 paper 在干一件啥事

你有一个 video diffusion model，比如 HunyuanVideo 或者 Wan。你训练它做 next-frame prediction——给它前面 $T$ 帧，它生成下一段 $S$ 帧。

问题来了：你想让它生成长视频，比如几千帧。那意味着 model 要"看到"很长的 history。Transformer 的 attention 复杂度是 $O(n^2)$，把所有历史帧全塞进去，显存爆炸。

那能不能不塞所有历史帧？也不能，因为会 **forgetting**——模型忘掉早期内容，比如人脸上纹身消失了、场景墙的颜色变了。

所以核心需求是：**让 model 记住很长的 history，但显存不要爆。**

这就是 FramePack 要解决的工程问题。但 paper 真正有意思的是它同时解决另一个更难的问题——**drifting**。下面分两块讲。

---

## 2. Forgetting 和 Drifting 是一对死敌

这是 paper 最值得 build intuition 的部分。两个 failure mode 看起来是独立的问题，其实互相打架。

**Forgetting**：history 不够长，model 记不住。直觉解法——给更多 history、给更强的 attention memory。

**Drifting**：next-frame prediction 是 chained inference，每一步都有小误差，误差累积起来越走越偏，最后视频变成糊糊的、扭曲的。CausVid 的实验证明 causal generator 中 quality degradation 集中在 video 末尾，且 high-quality 部分长度有上限。

现在矛盾来了——

**你给 model 越强的 memory，drifting 越严重**。为什么？因为更强的 memory 一方面减少 initial error 发生（看得多，生成准），另一方面一旦某帧出错，强 memory 把这个 error 也忠实地记住、传下去，error propagation 加速。

**你想打断 error propagation，必然要 weaken memory**。比如 DiffusionForcing 在 history 上加 noise，让 model 不要太依赖过去帧——drifting 缓解了，forgetting 又严重了。

这就是 paper 里反复讲的 **forgetting-drifting dilemma**。

---

## 3. FramePack 怎么压 history：几何级数

### 3.1 核心观察

邻近的帧最相关，越远的帧越不重要。这听起来 trivial，但关键是怎么利用这个观察。

vanilla 做法：每帧都用同样的 context length $L_f$（480p frame 约 1560 tokens），总 context length $L = L_f \times (T+S)$，$T$ 大就爆。

FramePack：**按时间距离指数压缩**。最近的帧用全精度，越远的帧越狠地压。

### 3.2 公式

第 $i$ 帧的 context length：

$$\phi(F_i) = \frac{L_f}{\lambda^i}$$

- $F_i$：第 $i$ 帧，$i=0$ 是最近的，$i=T-1$ 是最远的
- $L_f$：单帧原本的 context length
- $\lambda > 1$：压缩率，论文用 $\lambda=2$

总 context length 是个 geometric series：

$$L = S \cdot L_f + L_f \cdot \sum_{i=0}^{T-1} \frac{1}{\lambda^i} = S \cdot L_f + L_f \cdot \frac{1 - 1/\lambda^T}{1 - 1/\lambda}$$

当 $T \to \infty$：

$$\lim_{T \to \infty} L = \left(S + \frac{\lambda}{\lambda - 1}\right) \cdot L_f$$

$\lambda=2$ 时，history 部分固定占 $2L_f$——无论 $T$ 是 100 帧还是 10000 帧，history 都只占 2 帧的 token 量。

**这就是 FramePack 的 magic：context length 与 input frame 数解耦。**

### 3.3 怎么实现这个压缩

通过 **patchify kernel** 控制。Transformer 的 input layer 通常有一个 patchify 操作，把 latent pixels 聚合成 token。比如 HunyuanVideo 用 $(2, 4, 4)$ kernel——2 个时间步 × 4 个高 × 4 个宽的 latent pixel 聚成 1 个 token。

FramePack 给不同帧用不同 kernel。最近的 1 帧用 $(1, 2, 2)$——精度最高。往前一点的 4 帧用 $(4, 8, 8)$——压缩 64 倍。再往前 16 帧用 $(8, 16, 16)$——压缩 1024 倍。

物理意义：远期帧的 short-term motion 细节已经无关紧要（camera 已经动了 50 帧，那帧的瞬时像素运动不重要），但 scene、identity、style 这种 long-term context 还要保留——这些信息 low-frequency，pooling 掉细节也丢不掉。

### 3.4 一个小 trick：任意压缩率用 binary 表达

Hardware 喜欢整数倍 2。但有时你想要 2.625 倍。怎么做？

$$\frac{1}{1} + \frac{1}{2} + \frac{1}{2} + \frac{1}{4} + \frac{1}{8} + \frac{1}{8} + \cdots = 2.625$$

把 2.625 写成 binary，每个 bit 对应一项 power-of-2。要 1 就 duplicate 那一项，要 0 就不要。Elegant。

### 3.5 Tail frames 怎么办

$T$ 极大时远端帧会压到 sub-pixel。论文给三个 option：
- `td`（tail delete）：直接删
- `ta`（tail append）：每帧 3D pool 后用 nearest kernel
- `tc`（tail compress）：所有 tail 帧做 global average pool 一起 encode

实验差异 negligible。这本身是个 observation——远期帧的 precise representation 不重要，反正它们已经被压成"低频信号"了。

---

## 4. Drifting 怎么治：三招

光解决 forgetting 不够，drifting 还在。paper 给了三个互相补充的方法。

### 4.1 Endpoint Planning：给未来一个 anchor

Vanilla sampling 是顺序生成：生成 $X_1, X_2, \ldots$。每一步只看到过去，error 单向累积。

Endpoint planning 改成：第一轮同时生成 video 的开头和结尾，后续轮次 fill 中间。这样中间帧的生成是 bidirectional 的——它既看到 past anchor 也看到 future anchor。

数学直觉：单向 causal 的 conditional probability $\mathbb{P}(X_t | X_{t-1})$ 对 estimation error 极其敏感。Bidirectional 的 $\mathbb{P}(X_t | X_{t_1}, X_{t_2})$（$t_1 < t < t_2$）天然把 $X_t$ 拉向两端 anchor 的 consistency。

适合 motion 范围小的场景：跳舞、说话、旋转、火苗、流水。

### 4.2 Inverted Sampling：image-to-video 的绝招

这是 Table 1/2 里最 effective 的方法。

设定：image-to-video 第一帧是用户给的 groundtruth image（高质量），最后一帧是模型生成的 endpoint。

普通 forward sampling：从 first frame 往 last frame 推。每步看到的是上一步的 output，error 往前累积。

Inverted sampling：从 last frame 往 first frame 推。每一步都 "look back to" 第一帧那个 groundtruth。**errors 永远跑不出 first frame 的 anchor influence**——每一步都把生成结果 pull 回 high quality 状态。

这就是为什么 Table 1 里 inverted anti-drifting 在所有 drifting metrics 上都最好（$\Delta_{\text{drift}}^{\text{Clarity}}$ 只有 2.25%，vanilla 是 3.18% 看似差不多但 ELO 差 150 分）。

代价：dynamic degree 较低。因为 endpoint 是 anchor，motion 自由度受限。这是 freedom vs consistency 的 trade-off。

### 4.3 Multiple Endpoints：长视频的 storytelling

更长的视频可以规划多个 endpoint，每个对应一个 prompt。当 endpoint 之间足够远，error 在 endpoint 间 negligible，drifting 几乎不发生。

这是 film "storyboard" 的 computational analogue——人拍电影先画分镜再 fill，FramePack 让 model 也这么干。

### 4.4 RoPE 怎么处理非连续时间

普通 RoPE 在连续 time index 上加 phase。FramePack 的 endpoint sampling 跳过了中间 frame indices。怎么处理？

简单粗暴：**skip blank phases**。RoPE 的 time dimension 只在 anchor frames 上激活 phase，空白处不分配 index。

这里有个数学 subtlety——理论上 RoPE 的 phase 是非线性函数，但 FramePack 在 compression kernel 处直接 average pool RoPE phases。这种 pooling 不严格等价于"中间位置"的 RoPE，但 empirically 工作。Transformer 对 position encoding 形式有 surprising tolerance，类似 Perceiver IO、Nystromformer 的 robustness。

---

## 5. History Discretization：从 LLM 偷来的 robustness

### 5.1 关键 insight

Paper 里这句话很值得琢磨：

> Discrete autoregressive systems (e.g., LLMs) often demonstrate less obvious drifting than continuous autoregressive systems for visual content diffusion.

LLM 几乎不 drift。为什么？因为 token 是 discrete 的。每步生成一个 token，token 来自固定 codebook，error 被 quantization bottleneck 限制——下一步只能从 codebook 里选一个，不会"漂移"出 vocabulary。

Video diffusion 是 continuous 的，每步生成 continuous latent，error 是 unbounded 的，可以漂移出 training distribution。

### 5.2 FramePack 怎么借这个优势

用 K-Mean 给 latent videos 训一个 codebook $\Omega \in \mathbb{R}^{K \times C}$。然后训练时把 history frames quantize：

$$Q(F)_p = \arg\min_k \|F_p - \Omega_k\|_2$$

- $F_p$：frame 的第 $p$ 个 latent pixel
- $\Omega_k$：codebook 里第 $k$ 个 code
- $Q(F)$：每个 pixel 指向的 code index

然后 history frames 替换成 $\Omega_{Q(F)}$——dequantize 后的版本——喂给 diffusion model。

### 5.3 K 的 trade-off

- $K=1$：history 变成单一颜色，drifting 完全消失但 sections 之间 unrelated（彻底放弃 memory）
- $K \to \infty$：等价于 no discretization，drifting 回来
- $K=128$ 或 $K=256$：sweet spot，drift reduction 和 consistency 都不错

这相当于在 continuous diffusion 上"凿出"一个 discrete bottleneck。Training 时 model 就学会 expect quantized history，inference 时也 quantize model output——train/inference distribution gap 被 quantization "swallow"了。

这个 trick 跟 DAgger（Dataset Aggregation）的思路同源——reduce train/inference mismatch。也跟 VQ-VAE、MAGVIT-v2 的 codebook 类似。

---

## 6. 结果到底好不好

### 6.1 几个对比表的关键数字

| Method | Clarity | $\Delta_{\text{drift}}^{\text{Clarity}}$ | ELO |
|---|---|---|---|
| Repeating image-to-video | 56.73% | 9.51% | 1015 |
| Anchor frames (StreamingT2V-like) | 69.58% | 2.85% | 1173 |
| Causal attention (CausVid-like) | 62.88% | 7.45% | 1087 |
| DiffusionForcing ($\sigma_{\text{test}}=0.5$) | 67.41% | 3.55% | 1174 |
| **FramePack Inverted** | **71.15%** | **2.25%** | **1220** |
| **FramePack Vanilla + Discrete** | 70.01% | 3.13% | **1224** |

两个 FramePack variant 并列 Rank 1，把所有 alternative 打下去。

DiffusionForcing 的 $\sigma$ ablation 把 trade-off 摆得很明白：$\sigma_{\text{test}}=0$（clean history）drifting 8.41%，$\sigma_{\text{test}}=0.5$（heavy noise）drifting 3.55% 但 Clarity 也降。这是 forgetting-drifting dilemma 的 explicit instantiation。

### 6.2 Compute 上的颠覆

> FramePack can process thousands of frames with 13B models even on laptops (e.g., 6GB or 8GB GPU memory).

Training：batch size 64 on 8×A100-80G node，13B HunyuanVideo，480p LoRA。传统 video diffusion training batch size 通常 <10。

Inference：fixed context length，GPU memory 与 video 长度 decoupled。这是 streaming RNN 的 memory profile + transformer 的 attention 表达力（在 compressed context 内）。

---

## 7. 一些更深的联想

### 7.1 与 Compressive Transformer 的精神延续

DeepMind 2019 年的 Compressive Transformer（https://arxiv.org/abs/1911.05532）也是把老 memory 压缩存起来。FramePack 的差异：在 input layer 就压，compression rate 随时间指数增长，应用在 diffusion 而非 LM。但精神上——"old memory should be cheaper"——是一脉相承的。

### 7.2 Perceiver IO 的 implicit echo

Perceiver 用 cross-attention 把 variable input 压到 fixed latent array。FramePack 用 patchify kernel 实现 similar 效果但更 implicit——compression 嵌在 input projection 里。两者都是"fixed memory budget"思路。

### 7.3 与 Mamba/SSM 的对比

Mamba 用 linear-complexity 的 hidden state 实现 long context。FramePack 用 transformer + compression 实现 similar memory profile。Mamba 的 weakness 是 in-context retrieval 弱（hidden state 是 fixed-size bottleneck），FramePack 保留 attention 的 selective retrieval——在 compressed tokens 上还能"找"。

### 7.4 与 Token Merging（ToMe）的关联

ToMe（https://arxiv.org/abs/2210.05458）按 similarity merge tokens。FramePack 按时间 index 决定 merge aggressive 程度。feature-similarity version 例外——那才是真的 ToMe 的变体。

### 7.5 Sora 的 speculation

Sora 传闻用 spatiotemporal patches + diffusion transformer。如果 Sora 也是 next-frame-section prediction 形式，必然面对 forgetting-drifting dilemma。FramePack 的方法可能是 Sora 内部 solution 的 public version——progressive compression 解决 forgetting，endpoint planning 解决 drifting。Speculation，但 architecture constraints 决定 solution space 不大。

### 7.6 Memory Networks / NTM 的回声

FramePack 的"memory compression"思想可以追溯到 Memory Networks（Weston 2014）和 NTM（Graves 2014）。但 FramePack 不显式维护 memory bank——memory 就是 compressed input tokens 本身，没有 separate memory module。更"扁平"，但少了 explicit write/read head 的灵活性。

### 7.7 与 Consistency Models 的潜在结合

Consistency Models（https://arxiv.org/abs/2303.01469）用 few-step inference。FramePack 的 next-frame-section prediction 与 consistency models 结合可能实现 real-time streaming video。Paper 没探索，但 CausVid 已经展示 distillation 的潜力。

---

## 8. 我的看法

### 8.1 Paper 真正的贡献

最深的 insight 是把 forgetting-drifting dilemma 显式 frame 出来，然后用三个 orthogonal 的方法一起打破 symmetry：

1. **Asymmetric compression**：让 memory 增长但 compute 不增长
2. **Non-causal sampling**：让 future anchor 过去 frames——打破 causal error propagation
3. **Discrete bottleneck**：让 train/inference distribution gap 被 quantization "swallow"

单独任何一个都不够。三者叠加才能在 13B scale 上跑 thousands of frames on laptop。

### 8.2 Weaknesses

1. **Inverted sampling 的 dynamic degree 低**——endpoint anchor 限制 motion 自由度。Free-form storytelling 可能不理想。
2. **Feature-similarity packing 的 compute**——每步算 cosine similarity，长 video inference 速度如何论文没详谈。
3. **K 的 sensitivity**——K=128 vs 256 论文 ablation 不充分。Codebook 可能 not transfer 到新 domain。
4. **RoPE phase pooling 理论 gap**——empirical work 但没数学分析。
5. **与 2025 年 sparse attention SOTA 对比缺失**——Table 2 没比 Sparse VideoGen、Sliding Tile Attention 等。

### 8.3 真正的 takeaway

Video diffusion 的 drifting 本质是 continuous representation space 的 curse。LLM 用 discrete token 天然避免 drifting。FramePack 用 K-Mean quantization 把这个 advantage "borrow" 到 continuous diffusion 中——是个 elegant 折中。

更深的启示：**memory 和 error propagation 是同一枚硬币的两面**。你想加强一个，另一个必然也加强——除非你打破 symmetry。FramePack 提供了三种打破 symmetry 的方法。这三招可能不止适用于 video，可能适用于所有 sequential generation——audio、agent trajectory、robot control。

希望这个版本够"人话"了。如果你想 drill down 某个具体方面（比如 RoPE pooling 的数学细节、K-Mean codebook 训练、或与 SSM 的对比），可以继续。

---

# FramePack: Next-Frame-Prediction Video Diffusion 的 Forgetting-Drifting Dilemma

Andrej，这篇 Lvmin Zhang（ControlNet 作者）等人在 Stanford 的工作直击 next-frame prediction video diffusion 的一个根本矛盾——**forgetting 与 drifting 之间的不可调和性**。让我深入展开技术细节，并尽可能补充相关联想来 build your intuition。

参考链接：
- FramePack 论文（arXiv 镜像，推测链接）: https://arxiv.org/abs/2504.12626
- Lvmin Zhang 主页: https://lvmin-zhang.ai/
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- Wan: https://arxiv.org/abs/2503.20314
- DiffusionForcing: https://arxiv.org/abs/2407.01392
- CausVid: https://arxiv.org/abs/2412.07772
- HistoryGuidance: https://arxiv.org/abs/2502.06764
- Compressive Transformer (DeepMind, 高度相关的早期工作): https://arxiv.org/abs/1911.05532
- Perceiver IO: https://arxiv.org/abs/2107.14795

---

## 1. 核心矛盾：Forgetting-Drifting Dilemma

论文首先 reframe 了 next-frame prediction 的两个 failure modes：

- **Forgetting**：随着 generation 推进，model 失去对早期 content 的 access，temporal consistency 崩塌。本质是 memory capacity 不足。
- **Drifting**：error accumulation（也叫 observation bias 或 exposure bias）导致 quality degradation，最终 drift out of training distribution。

矛盾的本质是：**stronger memory 既能减少 initial errors 的发生，又会加速 errors 的 propagation**。反之，interrupt error propagation（如 noise history、mask history）必然 weakens temporal dependencies，加剧 forgetting。这与你在"micrograd"talk 中讲的 RNN long-range dependency 问题在 diffusion 设置下被重新激活，但矛盾更尖锐——因为 diffusion 的 multi-step denoising 本身已经是一个 sequential decision process。

CausVid [65] 的实证显示 causal video generator 中 quality degradation 集中在 video 末尾，且 high-quality portion 长度存在 upper bound。DiffusionForcing [6] 把 drifting 归因于 training/inference 的 observation disparity。

---

## 2. FramePack: Progressive Compression Bottleneck

### 2.1 核心公式（Time-Proximity Based）

变量定义：
- $X \in \mathbb{R}^{S \times \sqrt{h} \times w \times c}$：要生成的 next frame section，$S$ 是 frame 数（典型 1 或小数）
- $F \in \mathbb{R}^{T \times h \times w \times c}$：input history frames，$T \gg S$ 是 challenging case
- $L_f$：per-frame context length（Hunyuan/Wan/Flux 480p frame 约 1560 tokens）
- $F_0$：most important frame（最近），$F_{T-1}$：least important（最老）

**公式 (1)：Frame-wise context length**

$$\phi(F_i) = \frac{L_f}{\lambda^i}$$

- $\phi(F_i)$：第 $i$ 个 frame 经 VAE encoding 和 transformer patchify 后的 context length
- $\lambda > 1$：compression parameter
- $i$：frame index，越大越不重要

实现细节：通过 **manipulating transformer patchify kernel size** 来控制 compression rate。例如 $\lambda=2, i=5$ 表示 kernel 的三维 product 等于 $2^5 = 32$，可以是 $(2,4,4)$ 或 $(8,2,2)$ 等任意 3D kernel。

**公式 (2)：Total context length as geometric progression**

$$L = S \cdot L_f + L_f \cdot \sum_{i=0}^{T-1} \frac{1}{\lambda^i} = S \cdot L_f + L_f \cdot \frac{1 - 1/\lambda^T}{1 - 1/\lambda}$$

当 $T \to \infty$：

$$\lim_{T \to \infty} L = \left(S + \frac{\lambda}{\lambda - 1}\right) \cdot L_f$$

对 $\lambda = 2$，收敛到 $(S + 2) \cdot L_f$——即 history 部分固定占 2 个 frame 的 token 量，无论 $T$ 多大。这就是 FramePack 的 **compression bottleneck invariant to input frame number** 的核心 claim。

### 2.2 Binary Rate Covering Trick

由于 hardware 偏好 power-of-2，论文给出一个 elegant trick：把任意 rate 写成 binary bits 后 translate。例如想要 rate 2.625：

$$1 + \frac{1}{2} + \frac{1}{2} + \frac{1}{4} + \frac{1}{8} + \frac{1}{8} + \frac{1}{16} + \cdots = \frac{1}{2} + \frac{1}{8} + \sum_{i=0}^{+\infty} \frac{1}{2^i} = 2.625$$

即 duplicate $1/2$ 和 $1/8$ 两项。这个 trick 让 FramePack 在保持 hardware-friendly 的同时支持 arbitrary compression rate。

### 2.3 Patchify Kernel 设计空间

3D kernel $(p_f, p_h, p_w)$ 表示 frame、height、width 三个维度的 stride。同 compression rate 有多种实现：

| Compression Rate | 可选 Kernel |
|---|---|
| 64 | $(1,8,8)$, $(4,4,4)$, $(16,2,2)$, $(64,1,1)$ |

**Independent patchifying parameters**：empirical evidence 表明为不同 compression rate 用 independent neural network layers 更稳。常用：(2,4,4), (4,8,8), (8,16,16)。更高 compression（如 (16,32,32)）先 2×2×2 downsample 再用最大 kernel (8,16,16)。初始化时从 pretrained patchify projection（如 Hunyuan/Wan 的 (2,4,4)）插值。

### 2.4 Tail Options

当 $T$ 极大时 frames 会低于 minimum unit size（1 latent pixel）。3 个 options：
1. `td` (tail delete)：直接删除
2. `ta` (tail append)：每个 tail frame 用 (1,32,32) 3D pool 再用 nearest kernel encode
3. `tc` (tail compress)：global average pooling 所有 tail frames 用 nearest kernel encode

实测差异 negligible——这本身是一个 interesting observation，说明远期 frames 的 precise representation 不重要。

### 2.5 RoPE Alignment

不同 compression kernel 产生不同 context length，需要 RoPE (Rotary Position Embedding) [43] alignment。RoPE 对每个 token position 生成 complex number（real + imaginary），论文称其为 **"phase"**。Alignment 方法是直接 average pool 这些 phases 来 match compression kernel 的 downsample ratio。这是一个 surprising simple solution——理论上 RoPE 的 phase 是非线性函数，average pool 并不严格等价于"中间位置"的 RoPE，但 empirically 工作。

---

## 3. Feature-Similarity Based Packing

Time proximity 是 baseline，更 advanced 的是用 feature similarity 来决定 importance。

### 3.1 公式 (3)：Cosine Similarity

$$\text{sim}_{\text{cos}}(F_i, \hat{X}) = \sum_p \frac{(F_i)_p \cdot \hat{X}_p^\top}{\|(F_i)_p\| \|\hat{X}_p\|}$$

- $p$：pixel position
- $F_i$：第 $i$ 个 history frame
- $\hat{X}$：estimated next frame section

对每个 pixel 计算 cosine similarity 然后求和，作为 frame-level similarity。排序后 $F_0$ 最相似，$F_{T-1}$ 最不相似，直接替代 time proximity。

### 3.2 公式 (4) + (5)：Hybrid Similarity

Time-only 排序的 issue：在 world model datasets（如 video games）中，agent 可能 revisit 同一 scene，similarity 突变导致排序跳变。论文用 smooth time term + cosine term 的 hybrid：

$$\text{sim}_{\text{time}}(F_i, \hat{X}) = e^{-(\text{time}(F_i) - \text{time}(\hat{X}))^2}$$

$$\text{sim}_{\text{hybrid}}(F_i, \hat{X}) = \text{sim}_{\text{cos}}(F_i, \hat{X}) + \lambda_{\text{time}} \cdot \text{sim}_{\text{time}}(F_i, \hat{X})$$

- $\text{time}(\cdot)$：frame 的 starting time（秒）
- $\lambda_{\text{time}}$：weighting parameter

这个 hybrid 在 Minecraft、GTA 这类 world model 中特别有用——agent 回到之前 visited 的 view，feature similarity 把那些 frames 排到前面。论文也提到可对 facial identity metrics 排序来支持 movie generation 中 actor consistency。

---

## 4. Drift Prevention: Breaking the Causal Chain

### 4.1 数学 Motivation

Vanilla next-frame prediction 的 chained conditional probability：

$$\mathbb{P}(X_t | X_{t-1}) \to \mathbb{P}(X_t) \quad (\text{if estimation perfect})$$

实际中 estimation imperfect，导致 $\mathbb{P}(X_t | X_{t-1})$ 偏离 $\mathbb{P}(X_t)$，errors accumulate。

Bi-directional context：

$$\mathbb{P}(X_t | X_{t_1}, X_{t_2}) \quad \text{where } t_1 < t < t_2$$

提供 "future anchor"，让中间 frames 被 "pull toward" endpoint consistency。

### 4.2 Four Sampling Strategies (Figure 2)

**Vanilla (a)**：顺序生成。最简单但最 vulnerable to drifting。

**Anti-drifting (b)**：first iteration 同时生成 beginning 和 ending sections，后续 iterations fill gap。适合：
- 小范围 motion
- Periodic motion（dancing, talking, spinning）
- Texture patterns（fire flame, water flow）

**Inverted (c)**：image-to-video 专用。第一帧是 groundtruth user input（高质量），最后一帧是 generated endpoint。所有生成步骤都 "approximate the high-quality user input"——每一步都是 iteratively refined。**这是 Table 1/2 中最 effective 的方法**。

**Planned (d)**：multiple endpoints with different prompts。当 prompted sections 足够 distant，error accumulation 几乎 negligible。这是 film "storyboard" 的 computational analogue。

### 4.3 RoPE with Random Access

Non-consecutive sampling 需要修改 RoPE 支持非连续 time indices。方法：**skip blank phases（indices）in time dimension**。这相当于让 RoPE 的 time axis 变成 sparse 的，只在 generated/anchor frames 上激活 phase。

---

## 5. History Discretization: Borrowing from LLMs

### 5.1 公式 (6)：K-Mean Quantization

$$Q(F)_p = \arg\min_k \|F_p - \Omega_k\|_2$$

- $\Phi \in \mathbb{R}^{(B \times T \times H \times W) \times C}$：precomputed latent videos dataset
- $\Omega \in \mathbb{R}^{K \times C}$：codebook via K-Mean
- $K \in \mathbb{Z}^+$：codebook size
- $Q(F) \in [0, K-1]^{T \times H \times W}$：indices matrix
- $\Omega_{Q(F)} \in \mathbb{R}^{T \times H \times W \times C}$：quantized frames

Training 时把所有 history frames $F$ 替换为 $\Omega_{Q(F)}$。

### 5.2 Intuition 的两面

- $K = 1$：history 变成单一 color，drifting 完全消除但 sections 之间 unrelated（彻底放弃 memory）
- $K \to \infty$：等价于 no discretization，drifting 仍然存在
- 中间 $K$（实验中 $K = 128$ 或 $K = 256$ 较好）：在 reduce error propagation 和 maintain plausible consistency 之间取得平衡

### 5.3 深刻的观察

论文有一句关键 insight：

> Discrete autoregressive systems (e.g., LLMs) often demonstrate less obvious drifting than continuous autoregressive systems for visual content diffusion.

这与你在 nanogpt / GPT 教学中反复强调的 autoregressive discrete token model 的 robustness 一致。FramePack 把这个 insight 用 K-Mean quantization "注入"到 continuous diffusion setting 中——把 history 的 representation space 从 continuous $\mathbb{R}^C$ 投影到 discrete $\{0, 1, \ldots, K-1\}$，再 dequantize 回 continuous 喂给 diffusion model。这是 train-time regularization，让 model 在 training 时就 "expect" history 是 quantized 版本，从而 reduce inference-time distribution shift。

类似思路：
- VQ-VAE: https://arxiv.org/abs/1711.00937
- VQGAN: https://arxiv.org/abs/2012.09841
- MAGVIT-v2: https://arxiv.org/abs/2310.05753（video tokenizer）
- FlexTok [2]（论文引用，adjustable token context length）

---

## 6. Naming Convention（极其 compact 的 ablation 表示）

举例 `td_f16k4f4k2f1k1_g9_x_f1k1`：

| Token | Meaning |
|---|---|
| `td` | tail delete |
| `f16k4` | 16 frames 用 kernel (4,8,8) encode |
| `f4k2` | 4 frames 用 kernel (2,4,4) encode |
| `f1k1` | 1 frame 用 kernel (1,2,2) encode |
| `g9` | generate 9 frames |
| `x` | skip frames（endpoint 方法） |
| `+D` | history discretization |

Kernel 简写：`k1` = (1,2,2)，`k2` = (2,4,4)，`k4` = (4,8,8)，`k8` = (8,16,16)，`k16` = (16,32,32)。

Inverted 形式：`f1k1_x_g9_f1k1f2k2f16k4_td`——注意 input frames 在右侧（time reverse）。

---

## 7. Quantitative Results 解读

### 7.1 Drifting Metric 公式 (7)

$$\Delta_{\text{drift}}^M(V) = |M(V_{\text{start}}) - M(V_{\text{end}})|$$

- $V$：tested video
- $V_{\text{start}}$：first 15% of frames
- $V_{\text{end}}$：last 15% of frames
- $M$：任意 quality metric（Clarity, Motion, Semantic, Anatomy 等）

用 absolute difference 是因为 video 可以 forward 或 backward 生成，direction-agnostic。

### 7.2 Table 1 关键发现

**Inverted anti-drifting**（如 `f1k1_x_g9_f1k1f2k2f16k4_td`）：
- 4/7 global metrics 最好
- **所有** drifting metrics 最好（$\Delta_{\text{drift}}^{\text{Clarity}}$ ≈ 2.25%, $\Delta_{\text{drift}}^{\text{Motion}}$ ≈ 1.85%, $\Delta_{\text{drift}}^{\text{Semantic}}$ ≈ 2.68%, $\Delta_{\text{drift}}^{\text{Anatomy}}$ ≈ 8.58%）
- 但 Dynamic 较小（89.29% vs vanilla 92.91%）——因为 endpoint 是 anchor，限制了 motion 自由度
- ELO 1220, Rank 1

**Vanilla + discrete history** (`f16k4f2k2f1k1_g9+D`, K=256)：
- Drifting 表现接近 inverted
- **Dynamic 更大**（91.74%）——保持 motion freedom
- ELO 1224, Rank 1（与 inverted 持平）

**Vanilla sampling**（无 endpoint）：
- Dynamic 最高，但论文指出这可能是 drifting effect 而非真实质量（ELO 1072-1092）
- Drifting metrics 明显更高

### 7.3 K 参数 ablation

$K = 128$ 是 sweet spot：
- 太小：history 信息丢失，sections 之间不连贯
- 太大：discretization 失效，drifting 回来
- 这个 trade-off curve 类似 VQ-VAE 中 codebook size 的选择

### 7.4 vs Alternative Methods (Table 2)

| Method | Clarity | $\Delta_{\text{drift}}^{\text{Clarity}}$ | ELO |
|---|---|---|---|
| Repeating image-to-video | 56.73% | 9.51% | 1015 |
| Anchor frames (StreamingT2V-like) | 69.58% | 2.85% | 1173 |
| Causal attention (CausVid-like) | 62.88% | 7.45% | 1087 |
| DiffusionForcing ($\sigma_{\text{train}}$ random, $\sigma_{\text{test}}=0.5$) | 67.41% | 3.55% | 1174 |
| History guidance (HistoryGuidance-like) | 68.05% | 7.35% | 1152 |
| **FramePack Inverted** | **71.15%** | **2.25%** | **1220** |
| **FramePack Vanilla + Discrete** | 70.01% | 3.13% | **1224** |

DiffusionForcing 的 $\sigma$ ablation 很 informative：
- $\sigma_{\text{test}}=0$（clean history）：drifting 严重（8.41%），ELO 1095
- $\sigma_{\text{test}}=0.5$（heavy noise on history）：drifting 减小（3.55%）但 Clarity 也降（67.41%）
- 这验证了 noise-on-history 是 forgetting-drifting trade-off 的 explicit instantiation

---

## 8. Compute Efficiency 的颠覆性 Claim

> FramePack can process thousands of frames with 13B models even on laptops (e.g., 6GB or 8GB GPU memory).

Training 端：batch size 64 on single 8×A100-80G node with 13B HunyuanVideo 480p LoRA training, window size 2 或 3。这相比 full-video training 是巨大提升——传统 video diffusion training 受限于 quadratic attention complexity，batch size 通常 < 10。

Inference 端：fixed context length 意味着 GPU memory 与 video 长度 **decoupled**。这与 streaming RNN 的 memory profile 类似，但保留了 transformer 的 full attention 表达力（在 compressed context 内）。

---

## 9. 与相关工作的深层联系

### 9.1 Compressive Transformer (DeepMind 2019)

https://arxiv.org/abs/1911.05532

这是 FramePack 的 conceptually 最接近的 prior work。Compressive Transformer 把旧的 activations 通过 1D conv 或 pooling 压缩后存入 compressed memory。FramePack 的差异：
- 在 input layer 就做 compression（patchify kernel），而非 attention 后
- Compression rate 随时间指数增长（geometric），Compressive Transformer 是固定 rate
- 应用到 diffusion 而非 LM

### 9.2 Perceiver IO

https://arxiv.org/abs/2107.14795

Perceiver 用 cross-attention 把 variable-size input 压到 fixed-size latent array。FramePack 用 patchify kernel 实现 similar 效果但更 "implicit"——compression 嵌入在 input projection 中。Perceiver 的 asymmetric cross-attention 在 FramePack 中变成 symmetric self-attention（在 compressed tokens 上）。

### 9.3 DiffusionForcing [6]

https://arxiv.org/abs/2407.01392

DiffusionForcing unifies next-token prediction 和 full-sequence diffusion，核心是 history noise scheduling（$\sigma_{\text{train}}$ 在 training 时随机，$\sigma_{\text{test}}$ 在 inference 时固定）。FramePack 与之的对比是论文重点之一：
- DiffusionForcing 在 history 上加 noise 来 reduce observation bias
- FramePack 用 discretization 来 reduce distribution shift
- FramePack 还额外用 bi-directional sampling 来 break causal chain
- Table 2 显示 FramePack 整体优于 DiffusionForcing 的所有 $\sigma$ 组合

### 9.4 CausVid [65]

https://arxiv.org/abs/2412.07772

CausVid 把 bidirectional video model 蒸馏成 fast causal model，便于 KV cache 加速。论文指出 CausVid 在 video 末尾 quality 下降且存在 upper bound length——这是 causal 系统的 fundamental limitation。FramePack 反其道而行：保留 bidirectional attention（在 compressed context 内）+ bi-directional sampling，避免 causal 系统的 drifting。

### 9.5 FAR (Long-context Autoregressive Video)

https://arxiv.org/abs/2503.19325

FAR 用 multi-level causal attention + KV cache 实现 long-context autoregressive video modeling。与 FramePack 的对比：FAR 用 hierarchical attention pattern，FramePack 用 hierarchical patchify——前者是 attention 层面，后者是 input 层面。

### 9.6 LTXVideo [17]

https://arxiv.org/abs/2501.00103

LTXVideo 用 highly compressed latent space 来 efficient diffusion。FramePack 引用它作为 dataset pipeline 参考。两者的 synergy：LTXVideo 压缩 spatial latent，FramePack 压缩 temporal context，理论上可以叠加。

### 9.7 WorldMem [58]

https://arxiv.org/abs/2504.12369

WorldMem 用 retrieval-based memory——把 visited views 存到 external memory bank，按需 retrieve。FramePack 的 feature-similarity-based packing 是 implicit retrieval：直接用 similarity 排序现有 frames 而非外部 memory。WorldMem 更 explicit 但需要额外 memory management。

### 9.8 Memory Networks / Neural Turing Machines

FramePack 的 "memory compression" 思想可以追溯到 Weston 2014 的 Memory Networks 和 Graves 2014 的 NTM。但 FramePack 不显式维护 memory bank——memory 就是 compressed input tokens 本身。这与 Set Transformer 的 inducing points 思想也有 echo。

### 9.9 Token Merging (ToMe)

https://arxiv.org/abs/2210.05458

ToMe 在 attention 之前 merge similar tokens 来加速。FramePack 的 patchify compression 可以看作 "structured ToMe"——按时间 index 决定 merge 的 aggressive 程度，而非按 similarity（feature-similarity version 例外）。

### 9.10 Mamba / SSM 的对比

State Space Models（如 Mamba, https://arxiv.org/abs/2312.00752）用 linear-complexity 的 hidden state 实现 long context。FramePack 用 transformer + compression 实现 similar memory profile 但保留 attention 的 in-context expressiveness。两者本质都是 "fixed-size state + selective read/write"，FramePack 的 "selective" 体现在 compression rate schedule。

---

## 10. Implementation 细节的 Intuition

### 10.1 为什么 Inverted Sampling 最 effective？

考虑 image-to-video：第一帧是 user-provided groundtruth。Forward sampling：
- Step 1: $X_{1:9} \sim p(X_{1:9} | F_0)$，$F_0$ 是 groundtruth
- Step 2: $X_{10:18} \sim p(X_{10:18} | F_{0:9})$，但 $F_{1:9}$ 是 step 1 的 output（含 error）
- Step 3: errors compound

Inverted sampling：
- Step 1: 生成 endpoint $X_T$ from $F_0$
- Step 2: $X_{T-9:T-1} \sim p(X_{T-9:T-1} | F_0, X_T)$，两端都是 anchor
- Step 3: 继续往 $F_0$ 方向 fill，每一步都 "see" $F_0$（永远 high quality）

这就是为什么 Table 1 中 inverted 的 drifting metrics 全面碾压——**errors 永远不会 escape $F_0$ 的 anchor influence**。

### 10.2 为什么 Discretization 有效？

Training 时：history 是 $\Omega_{Q(F)}$，即 quantized 版本。
Inference 时：history 是 model output，也 quantize 后再 dequantize。

关键：**training 和 inference 的 history 都经过同一 quantization bottleneck**，所以 distribution shift 被消除。这与 DAgger（Dataset Aggregation, Ross et al. 2011）的思想类似——reduce train/inference mismatch。

更深层的 intuition：quantization 是 low-pass filter。LLM 的 token 是 extreme quantization（每个 token 是 vocabulary 中一个 ID），所以 LLM 几乎不 drift。FramePack 的 K=128 是 moderate quantization，在 drift reduction 和 detail preservation 之间取平衡。

### 10.3 Patchify Kernel 的 Physical Meaning

$(p_f, p_h, p_w)$ kernel 把 $p_f \times p_h \times p_w$ 个 latent pixels 聚合成 1 token。这相当于 3D average pooling + linear projection。对时间维度的 compression（$p_f > 1$）会"模糊"短期 motion 细节但保留长期趋势——这是远期 frames 可以被 aggressively compressed 的根本原因：它们的 short-term motion 信息已经 irrelevant，只需要提供 long-term context（场景、identity、style）。

### 10.4 RoPE Phase Pooling 的 Subtlety

RoPE 把 position $n$ 编码为 $e^{in\theta}$ 形式的 complex number。对相邻 positions $n, n+1, \ldots, n+k$ 的 RoPE phases 做 average pool：

$$\text{phase}_{\text{avg}} = \frac{1}{k} \sum_{j=0}^{k-1} e^{i(n+j)\theta} = \frac{e^{in\theta}}{k} \cdot \frac{1 - e^{ik\theta}}{1 - e^{i\theta}}$$

这是一个 **Dirichlet kernel**，magnitude 随 $k$ 和 $\theta$ 振荡。理论上不严格等价于"中间位置"的 RoPE，但论文 empirically 报告它 work。可能的解释：transformer 学会了 interpret pooled phase 作为 "smeared position"，并且 attention pattern 在 compressed tokens 上仍然有意义。这与 Nystromformer、Perceiver 等 "approximate attention" 工作的 robustness 类似——transformer 对 position encoding 的 precise form 有 surprising tolerance。

---

## 11. 潜在的扩展与 Open Questions

### 11.1 与 Consistency Models 的结合

Consistency Models（Song et al. 2023, https://arxiv.org/abs/2303.01469）用 few-step inference。FramePack 的 next-frame-section prediction 与 consistency models 结合可能实现 real-time streaming video generation。论文未探索，但 CausVid 已展示 distillation 的潜力。

### 11.2 与 Mamba/DiM 的结合

FramePack 的 compression bottleneck 是 architecture-agnostic 的。把 transformer 换成 Mamba 可能进一步 reduce memory（Mamba 的 hidden state 是 $O(1)$ per token）。但 Mamba 的 in-context retrieval 能力弱于 attention——FramePack 依赖 attention 来 "selectively attend" 到 compressed frames 的 informative 部分，换成 Mamba 可能 loss 这个能力。

### 11.3 Multi-modal Memory

Feature-similarity-based packing 已经支持 facial identity metrics。可以扩展到：
- Audio-visual sync（用 audio similarity 排序 frames）
- 3D geometry（用 camera pose similarity）
- Semantic scene graphs（用 scene composition similarity）

这与 Gen3C [38]（https://arxiv.org/abs/2503.03751）的 3D-informed video generation 有 synergy。

### 11.4 Test-Time Training (TTT)

TTT [9]（https://arxiv.org/abs/2504.05298）在 inference 时 fine-tune model on long context。FramePack 的 fixed context length 与 TTT 的 "memory in weights" 思路是 complementary——FramePack 提供 explicit memory，TTT 提供 implicit memory via weight updates。

### 11.5 Hierarchical Latent Diffusion

PyramidFlow [25] 在 pyramid 中 diffuse video latents。FramePack 的 progressive compression 与 pyramid structure 有 structural similarity——都是 multi-resolution representation。可能的 hybrid：在 FramePack 的 compressed frames 上做 pyramid diffusion，进一步 reduce compute。

### 11.6 与 Sparse Attention 的对比

Sparse VideoGen [56], SpargeAttn [71], Sliding Tile Attention [72] 都用 sparse attention pattern 加速 video diffusion。FramePack 用 dense attention 但在 compressed tokens 上——所以 sparse attention 方法可以叠加在 FramePack 之上，在 compressed context 内再 sparsify。这是 orthogonal 优化方向。

### 11.7 Drifting 的 Theoretical Analysis

Wang et al. [51]（https://arxiv.org/abs/2503.10704）提供了 autoregressive video diffusion 的 unified error analysis framework。FramePack 的 anti-drifting methods 可以在这个 framework 下分析：
- Endpoint planning：把 causal chain 切成 segments，error bound 变成 per-segment 而非 cumulative
- Discretization：把 error propagation 通过 quantization noise "regularize"——每步 quantization 引入 bounded noise，但阻断了 unbounded error growth

---

## 12. 我的 Critical Thoughts

### 12.1 Strengths

1. **Dilemma 的 explicit articulation**：forgetting-drifting trade-off 在 prior work 中隐式存在但很少被 so cleanly framed。这个 framing 本身就是 contribution。
2. **Architecture-agnostic**：能 finetune HunyuanVideo/Wan 而非从头训练，极大降低 adoption barrier。
3. **Laptop-friendly inference**：13B model on 6-8GB GPU 是 practical breakthrough。
4. **Ablation 严谨**：naming convention 让 ablation 可复现；Table 1 覆盖大量 configurations；Table 2 与 alternative methods 公平对比。
5. **Discretization insight**：连接 LLM robustness 和 video diffusion 是 deep observation。

### 12.2 Weaknesses / Open Concerns

1. **Inverted sampling 的 Dynamic 较小**：endpoint anchor 限制了 motion freedom。对 free-form storytelling 可能不理想。
2. **Feature-similarity packing 的 compute**：每步都要计算 cosine similarity，长 video 可能 expensive。论文未深入讨论 inference speed。
3. **K 的 sensitivity**：K=128 vs K=256 的差异论文未充分 ablation。Codebook 的 offline K-Mean 可能 not transfer 到新 domain。
4. **RoPE phase pooling 的 theoretical gap**：empirical work 但无理论分析为什么 OK。
5. **Identity preservation**：Table 2 显示 Identity score 在 FramePack 中并非最优（82.11% vs anchor frames 79.52% 仅略高）。Long video 中 actor identity 仍是 challenge。
6. **与最新 sparse attention 的对比缺失**：Table 2 未对比 Sparse VideoGen 等 2025 年 SOTA。

### 12.3 真正的 Insight

最深的 insight 在 section 4.2 的那句话——**discrete autoregressive systems demonstrate less obvious drifting than continuous ones**。这暗示了一个更深的问题：**video diffusion 的 drifting 问题本质是 continuous representation space 的 curse**。如果 video 生成走 VQ-VAE/MAGVIT 路线（discrete token + autoregressive transformer），drifting 自然减少——这正是 VideoPoet、Sora 早期讨论中提到的 discrete token 优势。FramePack 用 K-Mean discretization 把这个 advantage "borrow" 到 continuous diffusion 中，是一个 elegant 的折中。

### 12.4 与 Sora 的 Speculative 联系

Sora（OpenAI 2024）传闻用 spatiotemporal patches + diffusion transformer。如果 Sora 也是某种形式的 next-frame-section prediction，那么它必然面对 forgetting-drifting dilemma。FramePack 的方法可能是 Sora 内部 solution 的 public version——progressive compression 解决 forgetting，endpoint planning 解决 drifting。这是 speculation，但 architecture constraints 决定了 solution space 不大。

---

## 13. 总结：FramePack 的 Conceptual Contribution

FramePack 不仅是工程 trick，而是 reframe 了 video generation 的 fundamental tension：

> **Memory and error propagation are two sides of the same coin. You cannot increase one without increasing the other—unless you break the symmetry.**

打破 symmetry 的三个方法：
1. **Asymmetric compression**（FramePack structure）：让 memory 增长但 compute 不增长——通过 importance-weighted compression
2. **Non-causal sampling**（Endpoint planning）：让 future context anchor intermediate generation——打破 causal error propagation
3. **Discrete bottleneck**（History discretization）：让 train/inference distribution gap 被 quantization "swallow"——把 continuous error 变成 bounded discrete error

这三者组合，让 next-frame prediction 在 13B scale 上能跑 thousands of frames on laptop GPU——这是 video diffusion 的 "ChatGPT moment" 之一（如果 analogy 合适的话）。

希望这个深入分析能 build your intuition, Andrej。如果你想 drill down 到某个具体 aspect（比如 RoPE pooling 的数学、K-Mean codebook 的 training details、或与 SSM/Mamba 的对比），我可以继续展开。
