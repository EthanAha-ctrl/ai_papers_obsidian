---
source_pdf: Watermarking for AI-Generated Content.pdf
paper_sha256: a52b5524c87b96d94e57e5b896a532c66cbe38445d50e095a42ee1be3a6a565c
processed_at: '2026-08-13T03:44:09-07:00'
target_folder: AI生态
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Watermarking for AI-Generated Content: 去掉术语讲清楚

好, 我用更口语化的方式把 watermarking 这件事从头讲一遍, 同时保持足够的技术细节。

---

## 1. 这事儿到底在解决什么问题

想象一个场景: 你刷 Twitter 看到一张惊人图片, 说某地发生大事。你怎么知道这图是真实拍摄还是 Midjourney 生成?

传统思路是 post-hoc detection — 训练一个 classifier 看图是不是 AI 生成。早期 AI 生成的手有 6 根手指、眼睛奇怪, classifier 一抓一个准。但现在 Midjourney v6、DALL-E 3 的输出已经几乎无懈可击, classifier 失效了。OpenAI 自己的 AI classifier 就因为这个被 deprecated [https://openai.com/blog/new-ai-classifier-for-indicating-ai-written-text](https://openai.com/blog/new-ai-classifier-for-indicating-ai-written-text)。

**Watermarking 的核心 idea**: 与其等生成完了再去判断差异 (这种差异在缩小), 不如生成时主动 embed 一个我们自己设计的、有结构的信号。这样 detection 从"找微弱统计 gap"变成"检测一个有明确数学结构的 pattern", 难度完全不同。

类比: 假设你要在一片森林里藏一根针, 然后让人找。如果针是随机扔的, 找到很难 (post-hoc detection 的情况, 信号微弱); 如果你在针上绑一个会发出特定频率信号的 beacon, 找到就容易了 (watermarking 的情况, 信号明确)。

---

## 2. Watermark 的本质: Syntax

一个 watermarking scheme 本质上就是一组 algorithms:

- **Generation**: 给 prompt $\pi$, 用 secret key $\mathrm{gk}$ 生成 watermarked response $x$。记作 $\mathsf{Watermark}_{\mathrm{gk}}^{\mathcal{M}}(\pi) \to x$。
- **Detection**: 给 content $x$, 用 detection key $\mathrm{dtk}$ 判断是否 watermarked。返回 true/false。
- **Decoding** (可选): 如果 watermark 里 embed 了 message, 用 decoding key $\mathrm{dck}$ 把 message 解出来。
- **Attribution** (可选): 判断 $x$ 是否精确匹配某次 watermarked generation, 用于追责。

这里关键 insight: **generation key 和 detection key 可以不一样**。这听起来是细节, 后面会看到它解决了 fundamental 的矛盾 (robustness vs. unforgeability)。

---

## 3. Quality: 为什么 watermark 会伤 quality, 以及怎么定义"不伤"

Watermark 要修改生成分布, 修改就可能伤害 quality。论文把 quality 保证分成 4 个层次, 我用比喻讲:

### Level 1: Empirical Validation

"我跑了一堆实验, perplexity、FID 都没明显变化, 所以 quality 没问题。"

问题: 实验只覆盖被测的 prompt 分布。一个 weird prompt 上 quality 退化, 实验看不出来。

### Level 2: Low-distortion

每个 token 上分布偏移很小。公式:

$$
\max_{\mathsf{m}, \pi} \frac{1}{2} \sum_{x \in \mathcal{R}} \left| \Pr[\mathcal{M}(\pi) \to x] - \Pr_{\mathrm{gk}}[\mathsf{Watermark}_{\mathrm{gk}}^{\mathcal{M}}(\mathsf{m}, \pi) \to x] \right|
$$

变量含义:
- $\mathsf{m}$: embed 的 message bits
- $\pi$: prompt
- $\mathcal{R}$: response 集合 (可能是 partial response)
- $\frac{1}{2}\sum|\cdot|$: total variation distance, 衡量两个分布差异

直觉: 单次生成角度, 两个分布接近。但 multiple response joint 分布可能差很远 — 比如 Green-Red watermark 每个 token 上偏移小, 但长文本上 green token 比例系统性偏高, 一看就露馅。

### Level 3: Distortion-free

任何 efficient algorithm 拿到一个 sample, 判断不出它是来自原模型还是 watermarked 模型 (单 sample 角度):

$$
\left| \Pr_{x \gets \mathcal{M}(\pi)}[D^{\mathcal{M}}(1^{\lambda}, x) \to 1] - \Pr_{\mathrm{gk}}\left[ D^{\mathcal{M}}(1^{\lambda}, x) \to 1 \right] \right| \leq \mathsf{negl}(\lambda)
$$

变量含义:
- $D^{\mathcal{M}}$: 区分器 $D$, 可以查询原模型 $\mathcal{M}$
- $1^{\lambda}$: security parameter (unary)
- $\mathsf{negl}(\lambda)$: negligible in $\lambda$

直觉: 单次生成完全保留分布。但跨多次生成仍可能有 bias — 比如水印 scheme 总是把某些 token 推向某些方向, 多次 sample 就能看出。

### Level 4: Undetectable

最强的保证: 即使允许 attacker **多次 adaptive 查询**, 仍然分不清。

$$
\left| \Pr[D^{\mathcal{M}, \mathcal{M}}(1^{\lambda}) \to 1] - \Pr_{\mathrm{gk}}[D^{\mathcal{M}, \mathsf{Watermark}}(1^{\lambda}) \to 1] \right| \leq \mathsf{negl}(\lambda)
$$

变量含义:
- $D^{\mathcal{O}_1, \mathcal{O}_2}$: $D$ 可以 adaptive 查询 oracle $\mathcal{O}_1, \mathcal{O}_2$ (前者总是原模型, 后者可能是原模型或 watermarked 模型)

这意味着 **任何 efficient quality metric** 都看不出差异。如果有 metric 能看出, 它就是一个 distinguisher, 违反 undetectability。所以 undetectability 自动 imply quality 不退化, 不管你怎么定义 quality (FID、Inception Score、人类偏好...都不退化)。

**关键 impossibility**: statistical undetectability (对 unbounded $D$ 成立) 是 impossible 的 [https://arxiv.org/abs/2311.04378](https://arxiv.org/abs/2311.04378)。只能追求 computational version。

---

## 4. Green-Red Watermark: 用具体数字讲

[https://arxiv.org/abs/2301.10226](https://arxiv.org/abs/2301.10226)

这是最经典的 LLM watermark, 具体怎么工作:

1. 用 secret key + hash function, 在每个 token 位置把 vocabulary (假设 50000 个 token) 划分成 red list 和 green list, 各占 50%。
2. 给 green list 的 logits 加上 $\delta$ (比如 2.0):

$$
\tilde{l}(i)_j = l(i)_j + \delta \quad \text{if } v_j \in G
$$

变量含义:
- $l(i)_j$: 第 $i$ 个位置上, 第 $j$ 个 vocab token 的 logit
- $\delta$: 一个 positive bias (超参, 常用值 1.0-4.0)
- $\tilde{l}(i)_j$: watermarked logit

3. softmax 之后 green token 概率被 boost。具体幅度: $\delta = 2$ 时, green token 概率大约变成原来的 $e^2 \approx 7.4$ 倍 (相对于 red token), 经过 softmax 归一化后, 实际 green token 比例从 50% 上升到大约 80%。

**Detection 用 z-test**:

$$
z = \frac{|s|_G - \gamma T}{\sqrt{T \gamma (1 - \gamma)}}
$$

变量含义:
- $T$: text 长度 (token 数)
- $|s|_G$: 实际数到的 green token 数
- $\gamma$: 预期 green 比例 (这里是 0.5)
- 分母 $\sqrt{T\gamma(1-\gamma)}$: 二项分布 $B(T, \gamma)$ 的标准差

具体例子: 一段 200 token 的 text, $\gamma = 0.5$, random text 下 $|s|_G \sim B(200, 0.5)$, 均值 100, 标准差 $\sqrt{200 \times 0.5 \times 0.5} = \sqrt{50} \approx 7.07$。

如果 $z > 4$ 判为 watermarked, 对应 $|s|_G > 100 + 4 \times 7.07 \approx 128$。查正态分布表, $z = 4$ 单侧 p-value $\approx 3.17 \times 10^{-5}$, 即 **FPR $\approx 3 \times 10^{-5}$**。

水印文本下, green 比例约 80%, $|s|_G \approx 160$, 对应 $z \approx 8.5$, 远超阈值, 检测出来。

这就是 Green-Red 的精髓: 用 hash function 把 secret key 散布到每个 token 位置的 red/green 划分上, attacker 不知道 key 就不知道哪个 token 是 green, 无法系统性地避开。但生成时, 我们知道 key, 就 bias 向 green。

**局限**: 
- 对 paraphrase 不 robust (改了 token 但保留语义, green token 比例会被打乱)
- 对 Emoji Attack 不 robust (在词之间插入 emoji 再删掉, k-gram hash 失效)
- 单 token 偏移小, 但跨多 response 能看出系统性 bias (违反 undetectability)

---

## 5. Gumbel Watermark: 一个巧妙的 trick

Aaronson [https://openai.com/blog/scott-aaronson-on-watermarking/](https://openai.com/blog/scott-aaronson-on-watermarking/) 提出的版本, 关键是用 **Gumbel max trick**。

**背景知识 — Gumbel max trick**: 要从 $\exp(l_j)$ 归一化的分布采样, 不需要算 softmax, 可以这样做: 对每个 $j$, sample $g_j \sim \text{Gumbel}(0)$ (i.i.d.), 然后 $\arg\max_j (l_j + g_j)$ 就是正确的采样。这是一个数学上等价但计算上不同的 sampling method。

**Watermark 的 trick**: $g_j$ 不真随机, 而是用 $\text{PRF}(\text{seed}, \text{k-gram})$ 生成。这样:
- 生成时: $\arg\max_j (l_j + g_j)$, 实际上是从原分布采样, distribution 完全保留 (这是 Gumbel max trick 的数学保证)
- 检测时: 重现 $g_j$, 看实际 token 是否等于 $\arg\max_j (l_j + g_j)$

关键 insight: 当 k-gram 不重复 (k 足够大), 每次用的 $g_j$ 序列都是新的, single response distribution 精确等于原模型 — 这就是 **computational distortion-free**。

**为什么这个 trick 聪明**: 它把 watermark randomness 和 sampling randomness 用同一个机制处理。传统 sampling 是先算 softmax 概率再按概率采样, 而 Gumbel 是先加 noise 再取 max, 这两种方式 distribution 等价但 "noise 在哪一步加" 不同。Gumbel 把 noise 移到 max 之前, 让我们可以用 PRF 控制 noise, 同时不破坏 distribution。

Kuditipudi et al. [https://arxiv.org/abs/2307.15593](https://arxiv.org/abs/2307.15593) 把 randomness 改成 fixed key sequence 循环使用, 这样能容忍 constant rate 的 edit (robust), 但 detection 要 brute-force 搜索 $L$ 个 key sequences, 计算开销和 FPR 都乘以 $L$。

---

## 6. Undetectable Watermark: 终极 quality 保证

Christ et al. [https://arxiv.org/abs/2311.04378](https://arxiv.org/abs/2311.04378) 实现了第一个 undetectable watermark。

**核心问题**: 为什么 Gumbel watermark 只能 distortion-free 不能 undetectable?

答案: Gumbel 用 PRF applied to k-gram。如果同一个 k-gram 出现两次, PRF 输出相同, 引入 token 间 correlation。多次查询后, attacker 可能通过统计发现这些 correlation。

**Christ et al. 的 solution**: 只用包含足够 **empirical entropy** 的 token 序列作为 PRF input。

具体流程:
1. 从原模型采样 tokens $v_1, \ldots, v_{\ell}$
2. 累计 empirical entropy 直到达到 threshold $\tau$
3. 用 $(v_1, \ldots, v_{\ell})$ 作为 PRF input, 生成 randomness, bias 后续 tokens
4. 重复

**Empirical entropy** 直觉: 给定 logits 序列, 实际观测到的 token 序列的"惊讶程度"。如果 logits 都很 flat (e.g. 均匀分布), 任何 token 都很惊讶, entropy 高; 如果 logits 都很 sharp (e.g. 99% 概率某个 token), 实际 token 很不惊讶, entropy 低。

为什么这样能实现 undetectable: PRF input 永不重复 (因为高 entropy 序列撞车概率极低), 所以 watermark 不会引入 token 间 correlation。attacker 多次查询, 看到的还是 i.i.d. samples from 原分布。

**Tradeoff**: 一旦 $(v_1, \ldots, v_{\ell})$ 中任何 token 被修改, PRF input 变了, watermark 失效。所以对 substitution 不 robust。可以用 block 结构缓解 cropping, 但 substitution 仍无解。

---

## 7. PRC Watermark: 把 cryptography 拉进来

Christ & Gunn [https://eprint.iacr.org/2024/1086](https://eprint.iacr.org/2024/1086) 的工作是这个领域跟 cryptography 最深融合的。

**问题**: 上面 undetectable watermark 不 robust, 怎么做到既 undetectable 又 robust to substitution?

**Solution: Pseudorandom Error-Correcting Codes (PRC)**。

PRC 是一个 error-correcting code (有纠错能力), 但其 codeword 集合看起来是 random 的 (与 random string 不可区分)。

直觉: 一般的 ECC (如 Reed-Solomon) codeword 有明确 structure (比如 parity check), 一看就是 codeword。PRC 的 codeword 集合是从所有可能 code 中 pseudorandomly 选出来的, attacker 看不出 structure, 但 decoder 知道 structure 可以 decode。

**Watermark 流程**: 采样 LLM response 使其 tokens 与一个 PRC codeword 有显著 correlation。检测时, 把 PRC decoder 应用到 response 上, 如果 response 没被改太多 (substitution 在 ECC 纠错能力内), decoder 成功, watermark 检出。

数学上, PRC 是 $[n, k, d]$ code:
- $n$: codeword 长度
- $k$: message 长度
- $d$: minimum distance (纠错能力 $\lfloor (d-1)/2 \rfloor$)

但 codeword 集合是 pseudorandomly 选择的, pseudorandomness 保证 undetectability, error-correction 保证 robustness。这是第一次同时实现两个 property。

**目前状态**: 还没有 practical implementation 在 LLM 上跑通, 效率是 open question。Image 版本 [https://arxiv.org/abs/2410.07369](https://arxiv.org/abs/2410.07369) 已经实现。

---

## 8. Image Watermark: Latent Space 的 intuition

Image watermark 的 in-processing 方法都围绕 diffusion model 的 latent space $z_T$ (初始 noise)。

### Tree-Ring Watermark [https://arxiv.org/abs/2305.20030](https://arxiv.org/abs/2305.20030)

正常 diffusion: $z_T \sim \mathcal{N}(0, I)$, 然后 DDIM reverse process 生成 image。

Tree-Ring: 把 $z_T$ 在 Fourier domain 中的某些 concentric rings 强制设为 0, 其他位置仍 random Gaussian。生成时用这个 modified $z_T$。

Detection: 拿到 image, 用 DDIM inversion 估计 $z_T$, 看 Fourier domain 中那些 rings 是否接近 0。是 → watermarked。

```
Normal diffusion:
  z_T ~ N(0, I)  →  DDIM reverse  →  image

Tree-Ring:
  z_T with rings=0 in Fourier  →  DDIM reverse  →  image

Detection:
  image  →  DDIM inversion  →  estimated z_T
  check Fourier(estimated z_T) rings ≈ 0?
```

**Quality 问题**: 强制某些 Fourier 系数为 0, 引入 latent 偏离 $\mathcal{N}(0, I)$, image 质量和 diversity 下降。

**Robustness**: 对一般 image edit (resize, compress) 比较鲁棒, 但对 **surrogate attack** 脆弱 — Saberi et al. [https://arxiv.org/abs/2310.11451](https://arxiv.org/abs/2310.11451) 展示可以训一个 neural network 学习 watermark pattern, 然后用 PGD 优化 image 让 watermark 失效。

### Gaussian Shading [https://arxiv.org/abs/2404.04956](https://arxiv.org/abs/2404.04956)

把 $z_T \sim \mathcal{N}(0, I)$ 限制在一个固定 quadrant (由 key 决定), 即 truncated Gaussian。检测: 拿回 latent, 看它是否在那个 quadrant。

**Single-image quality claim**: 论文证明单 image 分布等于 unwatermarked 分布 (因为 quadrant 截断 + 对称性正好抵消)。但 **multi-image statistic 破坏**: 所有 image 都从同一 quadrant 出发, FID 这种跨 image 统计会退化。

### PRC Image Watermark [https://arxiv.org/abs/2410.07369](https://arxiv.org/abs/2410.07369)

把 Gaussian Shading 的 "fixed quadrant" 改成 "PRC-sampled fresh quadrant" — 每次生成用 PRC sample 一个新的、pseudorandom 的 quadrant。

- Single-image quality 保留 (PRC codeword 看起来 random, 不破坏分布)
- Multi-image quality 也保留 (因为每次 quadrant 不同, 跨 image 无 systematic bias)
- Robust (PRC 有纠错能力)
- 支持 multi-bit message

直觉: Gaussian Shading 用 fixed quadrant 是 "明文" 版本, PRC 用 pseudorandom quadrant 是 "密文" 版本 — 看起来 random, 其实有 structure, attacker 看不出来, 但 decoder 能 decode。

---

## 9. Threat Models 用人话说

### Attack 1: Watermark Removal

你想把 watermarked image 改成"看起来不像 AI 生成", 但保留 image 内容和质量。

三种策略:
- **Edit attack**: 小幅修改, 比如删除一些 token, resize image, 加 noise。简单版是 random edit, 高级版是 white-box 用 PGD 优化 adversarial perturbation [https://arxiv.org/abs/2309.10704](https://arxiv.org/abs/2309.10704)。
- **Regeneration attack**: 把 watermarked content 喂给另一个 (non-watermarked) GenAI model 重写。比如 paraphraser 改写 text [https://arxiv.org/abs/2310.03991](https://arxiv.org/abs/2310.03991), 或 diffusion denoiser 处理 image [https://arxiv.org/abs/2406.12527](https://arxiv.org/abs/2406.12527)。这是最强的 attack 之一, 因为 regeneration 同时改变 content 又保留 utility。
- **Downsampling attack**: 让 watermarked output 包含真正想要的内容作为 subset, 然后提取 subset。代表是 **Emoji Attack**: prompt LLM "在每个词之间插入一个 pineapple emoji 🍍", 生成后删除所有 pineapple, 严重破坏基于 k-gram 统计的 watermark。

### Attack 2: Watermark Forgery

反过来, 你想让 non-watermarked content 被 detector 误判为 watermarked。这可以用来诬告模型生成了 problematic content (比如诬告 GPT-4 生成了某段违法 text)。

Gu et al. [https://arxiv.org/abs/2312.04469](https://arxiv.org/abs/2312.04469) 展示: 只要拿到大量 watermarked vs. unwatermarked text corpus, 就能 partially 学到 Green-Red 的 red/green 划分规则, 然后生成 maximize green ratio 的 text 来 forge watermark。Saberi et al. [https://arxiv.org/abs/2310.11451](https://arxiv.org/abs/2310.11451) 在 image 上做了类似 attack。

### Attack 3: Secret Key Extraction

最 advanced — 直接提取 secret key, 然后 removal 和 forgery 都 trivially 可做。Jovanovic et al. [https://arxiv.org/abs/2402.19361](https://arxiv.org/abs/2402.19361) 展示了 watermark stealing attack。

---

## 10. Robustness vs. Unforgeability: 为什么必须分开 Detect 和 Attribute

这是论文最深刻的 insight 之一。

**Robustness** 要求: watermarked content 即使被修改, 仍能被检测出来。这对 misinformation detection 必要 — 用户可能压缩/截图/转发, 修改了很多, watermark 还在。

**Unforgeability** 要求: 不知道 key 的人, 无法制造能被 attributed 到某模型的 content。这对追责必要 — 如果你想说"GPT-4 生成了这段违法 content, OpenAI 要负责", 你需要证明这 content 真的来自 GPT-4, 不是有人伪造的。

**两者矛盾**: 如果 detector robust, attacker 可以拿一个真的 watermarked generation, 稍微改成 problematic content, 然后 attribute 给模型 — "你看, watermark 还在, 所以这是你的模型生成的!" 这显然不合理。

**Solution**: 用不同 algorithm。Detection 用 robust algorithm, attribution 用 unforgeable algorithm。Fairoze et al. [https://arxiv.org/abs/2310.18491](https://arxiv.org/abs/2310.18491) 通过在 LLM text 中 embed digital signature 实现 — 不包含大量 contiguous honestly-watermarked token 的 text 无法通过 attribution。这样 attacker 即使有原 watermarked text, 想伪造 attribution 必须直接 copy 长段 original text, 这就露馅了 (因为 problematic content 一般不会和原 text 完全一致)。

---

## 11. Open Problems 用人话讲

### 11.1 Open-source Model Watermarking

Llama 这种 open-weight model, 用户能任意修改 decoding 算法。你 watermark 设计在 sampling 阶段, 用户用自己 decoding 就绕过了。怎么 enforce?

Idea: 把 watermark 直接 train 进 model parameter, 让模型"天生"输出带 watermark 的 content, 不依赖于 decoding 算法。Zhao et al. [https://arxiv.org/abs/2210.03312](https://arxiv.org/abs/2210.03312), [https://arxiv.org/abs/2302.07511](https://arxiv.org/abs/2302.07511) 探索了 distillation-resistant 的方法 — 即使 model 被 distill 成另一个 model, watermark 还在。但这条路还很远。

### 11.2 Copyright Proof 的问题

假设用户自己画了一张图, 谎称是 DALL-E 生成的, 想 claim copyright。OpenAI 想 reveal watermark key 证明"这不是我们模型生成的"。

但问题: 对很多 scheme, **OpenAI 可以反过来, 给定用户画的那张图, 制造一个 fake watermark key** 让该图在这个 key 下被 detect 为 watermarked。所以 reveal key 不能证明什么。

需要的 property: 给定 content, adversary 难以制造一个 watermark key 使该 content 被 detect。可以通过要求 model owner **预先 commit** watermark key (commitment scheme [https://en.wikipedia.org/wiki/Commitment_scheme](https://en.wikipedia.org/wiki/Commitment_scheme)) 实现 — 这样事后无法换 key。

### 11.3 Privacy Risk: 被低估的 downside

假设 OpenAI 在每个用户生成的 image 里 embed 用户 ID (multi-bit watermark 技术上支持)。用户把这 image 发到 anonymous Twitter account 上。任何有 detector access 的人 (比如 OpenAI 内部, 或公开 detector 的合作伙伴) 都能 link 这 anonymous account 到该用户的真实身份。

这比传统 fingerprint (比如不同 whitespace pattern) 更可怕, 因为 **undetectable watermark 让用户根本不知道自己被 track**。传统 fingerprint 用户仔细看可能发现差异, undetectable watermark 在 distribution level 就完全隐藏了。

---

## 12. Policy 这边在发生什么

- **EU AI Act** Article 50 (2024 年 8 月生效): GenAI provider 必须把 output 标记为 "machine-readable format", Recital 133 提到 watermark 和 cryptographic methods [https://artificialintelligenceact.eu/](https://artificialintelligenceact.eu/)
- **US Executive Order 14110** (2023 年 10 月): 要求 Department of Commerce 240 天内提交 watermarking 技术报告 [https://www.whitehouse.gov/briefing-room/presidential-actions/2023/10/30/](https://www.whitehouse.gov/briefing-room/presidential-actions/2023/10/30/)
- **California SB 942** (2026 年生效): GenAI provider 必须提供 publicly available detection tool 和 latent disclosure [https://leginfo.legislature.ca.gov/](https://leginfo.legislature.ca.gov/)
- **中国**《深度合成管理规定》第 17 条 (2023 年 1 月生效): 对可能引起混淆的 GenAI content 必须加显著标识 [http://www.cac.gov.cn/2022-12/11/c_1672221949354811.htm](http://www.cac.gov.cn/2022-12/11/c_1672221949354811.htm)
- **C2PA** [https://c2pa.org/](https://c2pa.org/): industry 标准, Adobe + Microsoft + NYT + BBC 等共同建立, 定义 content provenance 的规范
- **Google SynthID** [https://deepmind.google/technologies/synthid/](https://deepmind.google/technologies/synthid/): 已经在 Gemini (text)、Lyria (audio)、Imagen (image)、VideoFX (video) 上 production 部署

一个 critical observation: policy 文本往往 vague, 没说清楚什么算 "watermark"、什么 quality 要求、什么 robustness 要求。如果 policy 超出技术能力 (比如要求 watermark 对所有 transformation 都 robust, 但技术上做不到), 法规就成空文。Christodorescu et al. [https://arxiv.org/abs/2404.06009](https://arxiv.org/abs/2404.06009) 专门讨论了这个 gap。

---

## 13. 整个领域的发展 intuition

把时间线串一下, 你能看到 watermarking 从 ad-hoc heuristic 一步步走向 rigorous cryptography:

1. **2023 初 — Green-Red (Kirchenbauer)**: 第一个被广泛采用的 LLM watermark, 简单有效, z-test 给 closed-form FPR bound。但只 low-distortion, 不 distortion-free。

2. **2023 中 — Aaronson Gumbel**: Gumbel max trick 把 watermark randomness 和 sampling randomness 统一, 实现 single-response distortion-free。漂亮但 multi-response 上仍可能露馅。

3. **2023 末 — Kuditipudi fixed key**: 用 fixed key sequence 循环, 换 robustness, 代价是 detection 复杂度和 FPR 都乘 $L$。

4. **2023 末 — Christ et al. Undetectable**: PRF input 永不重复 (empirical entropy threshold), 实现 undetectable — 任何 efficient metric 都看不出差异。但 substitution 一改就失效。

5. **2024 — PRC (Christ & Gunn)**: 把 coding theory 拉进来, PRC 同时具备 pseudorandomness (undetectability) 和 error-correction (robustness), 第一次同时实现两者。

6. **2024 — Image PRC (Gunn et al.)**: PRC idea 推广到 image latent space, 把 Gaussian Shading 的 fixed quadrant 改成 PRC-sampled quadrant, 解决了 multi-image quality 退化问题。

7. **2024 — SynthID production**: Google 在 Gemini 上大规模部署, Nature paper [https://www.nature.com/articles/s41586-024-08041-3](https://www.nature.com/articles/s41586-024-08041-3) 给了 production-scale 数据, 是 empirical 端的 benchmark。

整条线背后一个统一 intuition: **watermarking 本质是"在 distribution level 加 structure, 让 structure 看不见但 detect 得到"**。早期 work 用 hash + bias 加 structure 但破坏 distribution; Gumbel 用 noise reuse 加 structure 且 distribution-preserving (但 single-response only); undetectable 通过 entropy threshold 实现 multi-response preservation; PRC 用 coding theory 的 pseudorandom code 同时实现 distribution-preserving + robust。

---

## 14. Tradeoff 总览

论文 Section 7 提到 "no free lunch" [https://arxiv.org/abs/2402.16187](https://arxiv.org/abs/2402.16187): 不存在所有维度都最优的 watermark。要 trade-off:

| Property | 含义 | 与谁冲突 |
|---|---|---|
| Quality (undetectability) | 任何 metric 看不出差异 | Robustness (robust signal 必然有 structure, 可能被 detect) |
| Low FPR | 人类 content 极少被误判 | Detection power (FPR 越低, TPR 通常越低) |
| Robustness | 经过 transformation 仍能 detect | Unforgeability (robust → 可被 forge attribution) |
| Unforgeability | 不知道 key 无法制造 watermarked content | Robustness |
| Multi-bit | 能 embed message | 单内容 entropy 有限, message 越长 FPR 越高 / quality 越差 |
| Computational efficiency | generation/detection 快 | Robustness (复杂 decoding 慢) / Multi-bit (长 message 慢) |

**Application-driven choice**:
- Misinformation detection: 要 low FPR + robustness, 不要 multi-bit
- 训练数据清洗: 不要 robustness (用户不主动 remove), 要 low FPR + efficiency
- Attribution / 追责: 要 unforgeability, 不要 robustness (用单独 algorithm)
- User tracking (privacy-violating use case): 要 multi-bit + undetectability

---

## 15. 还有什么没讲到

论文里没深入但有意思的方向:

1. **Steganography 的历史联系**: watermarking 和 steganography (信息隐藏) 有共同的数学基础, 但 goal 不同 — steganography 是隐藏"有 message"这个事实, watermarking 是隐藏"有 watermark"这个事实同时保证可检测。Wasserstein distance、channel capacity 等信息论工具两边都用。

2. **Backdoor 的区别**: model backdoor (在 model parameter 里 embed trigger pattern) 看起来像 watermark, 但机制完全不同。Backdoor 是 trigger → output 改变, watermark 是 output → signal detectable。两者都涉及"在 distribution 里藏东西", 但 threat model 不同。

3. **MPC 联系**: PRC 的构造用到了 cryptography primitive, 跟 secure multi-party computation、zero-knowledge proof 等技术同源。未来 watermark verification 可能用 ZKP 实现"我能 detect watermark 但不 reveal key"。

4. **Differential privacy 角度**: distortion-free 和 DP 的定义有形式上的相似 (都是 distribution indistinguishability)。两者能否统一在某个框架下是个有趣的 open question。

5. **Model fingerprinting vs. watermarking**: model fingerprinting 是 model owner 验证 model 身份 (不要求 attacker 合作), watermarking 是检测 content 来源。两者 threat model 和 technical solution 不同。

---

## 总结

这篇 SoK 的价值在于: 把 watermarking 在 GenAI 语境下从 ad-hoc heuristic 提升到有 formal definition、clear threat model、systematic evaluation 的科学状态。技术脉络从 Green-Red 的 z-test 一路演进到 PRC 的 cryptography fusion, 每一步都在解决前一步的局限。Policy 端 (EU AI Act、California SB 942) 推动需求, 但 policy 文本的 vague 和 technical reality 之间还有 gap 要 bridge。

最 fundamental 的 insight: **watermarking 是 distribution level 加 invisible-but-detectable structure 的艺术**, 而 PRC 是目前最好的"既 invisible 又 robust"的方案。Open problem 在 open-source model watermarking、copyright proof、privacy risk 几个方向。

---

# SoK: Watermarking for AI-Generated Content 深度解析

这篇论文是一篇 Systematization of Knowledge (SoK) 文章, 由 UC Berkeley 的 Xuandong Zhao 等人撰写, 对 generative AI watermarking 领域做了系统性的梳理。我下面从 motivation、formal definitions、threat models、具体方案, 一直到 open problems, 把这篇 paper 的核心脉络和背后 intuition 讲清楚, 同时补充一些技术上更细节的推导。

---

## 1. 为什么需要 Watermarking?

### 1.1 Post-hoc Detection 的根本困境

post-hoc detection (事后检测) 指在生成完成后, 通过统计信号或训练一个 binary classifier 来区分 human-created content 与 AI-generated content。代表性工作包括 DetectGPT [https://arxiv.org/abs/2301.11305](https://arxiv.org/abs/2301.11305), Fast-DetectGPT, Binoculars [https://arxiv.org/abs/2401.12070](https://arxiv.org/abs/2401.12070) 等。

这类方法有两个本质缺陷:

1. **error rate floor**: 报告的 error rate 很难低于 $10^{-3}$, 而且没有 theoretical guarantee on false positive rate (FPR)。在 misinformation 这种场景, FPR 必须非常低 (比如 $10^{-6}$), 否则会把大量人类内容误判为 AI 生成。

2. **out-of-distribution 失效**: 当 generative model 升级, 旧的 detector 对新模型 output 失效。OpenAI 自己的 AI classifier 就因为 low accuracy 和 inconsistent performance 被 deprecated [https://openai.com/blog/new-ai-classifier-for-indicating-ai-written-text](https://openai.com/blog/new-ai-classifier-for-indicating-ai-written-text)。

直觉上理解: post-hoc detection 是在追逐一个 moving target — generative model 越来越接近真实数据分布, statistical gap 越来越小, 最终趋于 zero。watermarking 则是主动 embed 一个信号, 把 detection 的难度从"寻找微弱统计差异"转化为"检测一个我们自己设计的、有结构性的信号"。

### 1.2 Watermarking 的应用场景

论文列举了几类 use case:

- **Combating misinformation**: 大规模生成的假信息如果都带 AI flag, 用户至少能区分来源
- **Fraud detection**: GenAI 能产生 tailored scam message, 传统基于模板匹配的 detection 失效
- **Academic integrity**: 检测学生用 ChatGPT 写作业
- **Avoiding training data contamination**: 避免 model collapse (在 AI 生成数据上反复训练导致退化 [https://www.nature.com/articles/s41586-024-07566-y](https://www.nature.com/articles/s41586-024-07566-y))
- **Signature & attribution**: 证明某段 content 是否由特定 model 生成

### 1.3 Policy context

值得注意的几个 policy:

- **EU AI Act** Article 50 要求 GenAI provider 把 output 标记为 "machine-readable format", 明确提到 watermark 和 cryptographic methods
- **US Executive Order 14110** 要求 Department of Commerce 240 天内提交 watermarking 技术报告
- **California SB 942** (California AI Transparency Act) 2026 年生效, 要求 provider 提供 publicly available detection tool 和 latent disclosure
- **China** 的《深度合成管理规定》第 17 条要求对可能引起混淆的 GenAI content 加显著标识

Industry 这边, Google DeepMind 的 **SynthID** [https://deepmind.google/technologies/synthid/](https://deepmind.google/technologies/synthid/) 是 production-scale 部署的代表, 已经在 Gemini (text), Lyria (audio), Imagen (image), VideoFX (video) 上落地。**C2PA** [https://c2pa.org/](https://c2pa.org/) 是 content provenance 的标准, 试图建立 cross-platform 的 provenance chain。

---

## 2. Watermark 的 Formal 定义

### 2.1 Syntax

论文用一套比较通用的 notation 来描述 watermarking scheme:

- $\mathcal{M}(\pi) \to x$: 从 generative model $\mathcal{M}$ 以 prompt $\pi$ 采样得到 response $x$。$x$ 可以是 text sequence、image、audio、video。
- $\mathsf{Watermark}_{\mathrm{gk}}^{\mathcal{M}}(\pi) \to x$: 用 generation key $\mathrm{gk}$ 生成 watermarked response。这是 watermarking scheme 的核心算法。
- 对 multi-bit watermark, 还可以 embed message: $\mathsf{Watermark}_{\mathrm{gk}}^{\mathcal{M}}(\mathsf{m}, \pi) \to x$, 其中 $\mathsf{m} \in \{0,1\}^k$。
- $\mathsf{Detect}_{\mathrm{dtk}}(x) \to \{\text{true}, \text{false}\}$: detection algorithm, 用 detection key $\mathrm{dtk}$
- $\mathsf{Decode}_{\mathrm{dck}}(x) \to \{0,1\}^k$: 解码 embedded message
- $\mathsf{Attribute}_{\mathrm{ak}}(x) \to \{\text{true}, \text{false}\}$: attribution, 判断 content 是否精确匹配某次 watermarked generation

注意这里把 generation key (gk), detection key (dtk), decoding key (dck), attribution key (ak) 分开, 在很多 scheme 里它们是同一个 key, 但 formal 上分开有好处 — 后面会看到 robustness 和 unforgeability 之间的张力需要通过分开 detect / attribute 来解决。

### 2.2 Quality 的层次结构

这是这篇 paper 在 conceptual 上的一个核心贡献: 把 watermark 的 quality 保证分成几个递进的层次, 从弱到强:

#### (a) Empirical Quality Validation

通过 perplexity (PPL)、diversity、MAUVE score [https://arxiv.org/abs/2102.01436](https://arxiv.org/abs/2102.01436)、human eval、LLM-as-a-judge [https://arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685) 等指标比较 watermarked vs. unwatermarked output。

局限: 只覆盖被测试的 prompt 分布, 无法保证 out-of-distribution prompt 上的 quality。

#### (b) Low-distortion

**Definition 3.1 (Distortion)**:

$$
\max_{\mathsf{m}, \pi} \frac{1}{2} \sum_{x \in \mathcal{R}} \left| \Pr[\mathcal{M}(\pi) \to x] - \Pr_{\mathrm{gk}}[\mathsf{Watermark}_{\mathrm{gk}}^{\mathcal{M}}(\mathsf{m}, \pi) \to x] \right|
$$

变量解释:
- $\mathsf{m}$: 要 embed 的 message
- $\pi$: prompt
- $\mathcal{R}$: 可能的 (partial) response 集合
- $\Pr[\mathcal{M}(\pi) \to x]$: 原始模型产生 $x$ 的概率
- $\Pr_{\mathrm{gk}}[\mathsf{Watermark}_{\mathrm{gk}}^{\mathcal{M}}(\mathsf{m}, \pi) \to x]$: watermarked 模型产生 $x$ 的概率
- $\frac{1}{2} \sum |\cdot|$: total variation distance

这是 single-response 概念: 对每个 prompt, 两个分布在 TV distance 意义下接近。但多个 response joint 分布可能差很远 — Green-Red watermark [https://arxiv.org/abs/2301.10226](https://arxiv.org/abs/2301.10226) 就是典型例子, 每个 token 上 distribution 偏移很小, 但长序列上 green token 比例明显偏高。

#### (c) Distortion-free

**Definition 3.2 (Computational Distortion-freeness)**: 对任何 prompt $\pi$、message $\mathsf{m}$、security parameter $\lambda$, 以及任何 polynomial-time algorithm $D$:

$$
\left| \Pr_{x \gets \mathcal{M}(\pi)}[D^{\mathcal{M}}(1^{\lambda}, x) \to 1] - \Pr_{\substack{\mathrm{gk}}} \left[ D^{\mathcal{M}}(1^{\lambda}, x) \to 1 \right] \right| \leq \mathsf{negl}(\lambda)
$$

变量解释:
- $D^{\mathcal{M}}$: 区分器 $D$, 可以向原模型 $\mathcal{M}$ 查询, 同时被 given 一个 sample $x$
- $1^{\lambda}$: security parameter 的 unary 表示
- $\mathsf{negl}(\lambda)$: negligible function in $\lambda$

直觉: $D$ 拿到一个 sample, 它要判断这个 sample 是来自原模型还是 watermarked 模型。distortion-free 要求 $D$ 几乎分不出来 (单次 sample 角度)。统计版本要求这个对 unbounded $D$ 也成立。

代表方案: Kuditipudi et al. [https://arxiv.org/abs/2307.15593](https://arxiv.org/abs/2307.15593) 用 Gumbel sampler 实现, 关键是用 fixed randomness (称为 "key") 而非 PRF on k-gram, 这样 single response distribution 精确等于原模型 distribution。代价是检测时需要 brute-force 搜索 $L$ 个 key sequences, 计算复杂度和 FPR 都上升 factor $L$。

#### (d) Undetectable

**Definition 3.3 (Undetectability)**: 对任何 polynomial-time algorithm $D$:

$$
\left| \Pr[D^{\mathcal{M}, \mathcal{M}}(1^{\lambda}) \to 1] - \Pr_{\mathrm{gk}}[D^{\mathcal{M}, \mathsf{Watermark}}(1^{\lambda}) \to 1] \right| \leq \mathsf{negl}(\lambda)
$$

这里 $D^{\mathcal{O}_1, \mathcal{O}_2}$ 表示 $D$ 有 adaptive query access to 两个 oracles, 可以用任意 prompt 和 watermark message。

直觉: $D$ 可以多次 adaptive 查询, 要判断自己拿到的是只访问原模型, 还是同时访问原模型 + watermarked 模型。这是最强的 quality 保证: 如果任何 efficient quality metric 能区分, 那它就构成一个 distinguisher, 违反 undetectability。所以 undetectability 自动 imply 任何 efficient quality metric 下的 quality 不退化, 包括跨多个 generation 的 metric (FID, Inception Score 等)。

代表方案: Christ et al. [https://arxiv.org/abs/2311.04378](https://arxiv.org/abs/2311.04378) 的核心 idea 是用 PRF applied to preceding tokens, 但只用那些包含足够 empirical entropy 的 token 序列作为 PRF input, 这样 PRF input 永不重复, 不会引入 token 间 correlation。empirical entropy 衡量序列中的随机性, 可以从 logits 计算。

**Statistical undetectability 是不可能的** — 这是 Christ et al. 证明的。所以只能追求 computational version。

### 2.3 False Positive Rate

**Definition 3.4**: detector 的 FPR at most $\varepsilon$ 意味着, 对任何 fixed content $x$:

$$
\Pr_{\mathrm{dtk}}[\mathsf{Detect}_{\mathrm{dtk}}(x) \to \text{true}] \leq \varepsilon
$$

关键: 这个定义 **agnostic to content distribution**。如果 FPR 只对特定分布成立, detector 会系统性地误判某类内容 (例如 non-native English writer, 这正是 GPT detector 被诟病的 bias [https://www.cell.com/patterns/fulltext/S2666-3864(23)00151-5](https://www.cell.com/patterns/fulltext/S2666-3864(23)00151-5))。

### 2.4 Robustness

**Definition 3.5 (Robustness)**: detector 对 channel $\mathcal{E}$ robust with error $\varepsilon$ for property $P$ 意味着, 对任何 prompt $\pi$:

$$
\Pr_{\substack{\mathrm{gk}, \mathrm{dtk} \\ x \gets \mathsf{Watermark}_{\mathrm{gk}}^{\mathcal{M}}(\pi) \\ x' \gets \mathcal{E}(x)}} \left[ \mathsf{Detect}_{\mathrm{dtk}}(x') = \text{false} \text{ and } P(\mathcal{M}, \pi, x) = \text{true} \right] \leq \varepsilon
$$

变量解释:
- $\mathcal{E}$: channel, 代表 environment 或 adversary 的修改操作 (例如 paraphrasing、resize、compression)
- $x'$: 经过 channel 后的 content
- $P(\mathcal{M}, \pi, x)$: 某个关于 $\mathcal{M}, \pi, x$ 的 property, 通常度量 entropy

为什么需要 property $P$? 如果 prompt 要求 deterministic response, 那 watermark 根本没法 embed (不破坏 distortion-freeness 的话), 更谈不上 robust。所以 robustness 只在 response 有足够 randomness 时才有意义。

### 2.5 Unforgeability

**Definition 3.6 (Unforgeability)**: 对任何 polynomial-time adversary $\mathcal{A}$:

$$
\Pr_{\substack{\mathrm{gk}, \mathrm{ak} \\ x \gets \mathcal{A}^{\mathsf{Watermark}_{\mathrm{gk}}^{\mathcal{M}}}(1^{\lambda}, \mathrm{ak})}} \left[ \mathsf{Attribute}_{\mathrm{ak}}(x) = \text{true} \text{ and } x \notin \mathcal{Q} \right] \leq \mathsf{negl}(\lambda)
$$

其中 $\mathcal{Q}$ 是 $\mathcal{A}$ 通过查询 watermarked oracle 得到的 response 集合。

直觉: $\mathcal{A}$ 可以查询 watermarked model 收集 watermarked content, 但它不能基于这些"伪造"出新内容, 使得新内容能被 attribute 到这个 model 上。

**关键张力**: robustness 和 unforgeability fundamentally incompatible。如果 detector robust, attacker 可以拿一个 watermarked generation, 稍微改一点成 problematic content, 然后诬告模型生成了这个 problematic content。解决方案: **detection (要求 robust) 和 attribution (要求 unforgeable) 用不同 algorithm**。Fairoze et al. [https://arxiv.org/abs/2310.18491](https://arxiv.org/abs/2310.18491) 构造了支持 unforgeable public attribution 的 scheme, 通过把 digital signature 嵌入 LLM text, 使得不包含大量 contiguous honestly-watermarked token 的文本无法通过 attribution 测试。

---

## 3. Threat Models

### 3.1 Attack Objectives

1. **Watermark Removal**: 修改 watermarked content 使 detector 判为 false (或 decoder 解出错误 message)。同时要 preserve quality, 否则退化掉 quality 也能 trivially remove watermark。

2. **Watermark Forgery**: 制造 content 让 detector 误判为 watermarked。这可以用来诬告模型生成了 problematic content。注意 forgery 不需要 watermark key — Gu et al. [https://arxiv.org/abs/2312.04469](https://arxiv.org/abs/2312.04469) 和 Jovanovic et al. [https://arxiv.org/abs/2402.19361](https://arxiv.org/abs/2402.19361) 展示了只要有大量 watermarked vs. unwatermarked content corpus, 就能 partially 学到 Green-Red 的 red-green 划分。

3. **Secret Extraction**: 提取 watermark key, 这是更 advanced 的目标。一旦拿到 key, removal 和 forgery 都 trivially 可做。

### 3.2 Adversary Capabilities

论文列了几个维度的 adversary capability, 这是分析 watermark security 时必须明确的:

- **Generator oracle access**: 能否生成更多同 key 的 watermarked output
- **Access to watermarked/non-watermarked content**: 有多少 sample 可用
- **White-box access to model**: 能否修改 model 内部参数
- **Non-watermarked generator access**: 能否查询原模型
- **Chosen key oracle**: 能否用 chosen key 生成 watermarked output
- **Verifier feedback granularity**: verifier 返回 true/false 还是 continuous score
- **Verifier oracle access**: 能否多次查询 verifier (这很重要, 如果可以 iteratively refine, evasion 容易很多)
- **Surrogate model access**: 是否有 surrogate model 可用于 paraphrasing / regeneration

---

## 4. 具体方案详解

### 4.1 Text Watermarks

#### 4.1.1 Green-Red Watermark (Kirchenbauer et al.)

[https://arxiv.org/abs/2301.10226](https://arxiv.org/abs/2301.10226)

这是最被广泛引用的 LLM watermark。

**机制**: 每个 token 位置上, 用 secret key (经过 hash function 或 PRF) 把 vocabulary 划分成 red list $R$ 和 green list $G$。在 Kirchenbauer 版本中, partition 基于前 $k$ 个 token 的 k-gram 通过 hash 伪随机决定。然后对 logits 做 bias:

$$
\tilde{l}(i)_j = \begin{cases} l(i)_j + \delta & \text{if } v_j \in G \\ l(i)_j & \text{if } v_j \in R \end{cases}
$$

变量解释:
- $l(i)_j$: 第 $i$ 个 token 位置, vocabulary 中第 $j$ 个 token 的 logit
- $\delta$: green list bias, 一个正实数 (e.g. 2.0)
- $\tilde{l}(i)_j$: watermarked logit

效果: softmax 之后, green token 的概率被 boost 了。检测时, 用同样 hash function 划分 red/green, 计算 green token 数量 $|s|_G$, 用 z-metric:

$$
z = \frac{|s|_G - \gamma T}{\sqrt{T \gamma (1 - \gamma)}}
$$

变量解释:
- $T$: text 长度 (token 数)
- $|s|_G$: 实际 green token 数量
- $\gamma$: green list 的预期比例 (通常是 0.5)
- 分母 $\sqrt{T\gamma(1-\gamma)}$: 二项分布 $B(T, \gamma)$ 的标准差

直觉: 在 random text 下 $|s|_G \sim B(T, \gamma)$, 均值 $\gamma T$, 标准差 $\sqrt{T\gamma(1-\gamma)}$。watermark 使 $|s|_G$ 偏高, $z$ 值大于阈值 (e.g. 4) 判为 watermarked。$z = 4$ 对应单侧 p-value $\approx 3 \times 10^{-5}$, 即 FPR $\approx 10^{-5}$。

**局限**:
- Low-distortion (单 token 上偏移小), 但 multi-response 上偏移明显 — 长文本中 green token 比例系统性偏高
- 对 paraphrase、Emoji Attack (在词之间插入 emoji 再删除) 不 robust
- δ 越大, robustness 越强, 但 quality 越退化 — tradeoff

#### 4.1.2 Gumbel Watermark (Aaronson / Kuditipudi)

Aaronson [https://openai.com/blog/scott-aaronson-on-watermarking/](https://openai.com/blog/scott-aaronson-on-watermarking/) 提出的方案和 Kirchenbauer 类似, 都基于前 $k$-gram 通过 PRF derive randomness 来 bias next token, 但用 **Gumbel biasing strategy**: 在 logits 上加 Gumbel noise, 然后取 argmax。这个操作有一个漂亮性质 — 当 $k$ 足够大使 k-gram 不重复时, **single response distribution 完全等于原模型**, 即 computational distortion-free。

数学上: 如果 $g_j \sim \text{Gumbel}(0)$ 独立同分布, 则 $\arg\max_j (l_j + g_j)$ 的分布正比于 $\exp(l_j)$, 这正好是 softmax 采样。所以 Gumbel trick 在不破坏 distribution 的情况下实现了 sampling。watermark 的 trick 是: $g_j$ 不真随机, 而是用 PRF(seed, k-gram) 生成的, 这样检测时可以重现 $g_j$, 检验 token 是否匹配。

Kuditipudi et al. [https://arxiv.org/abs/2307.15593](https://arxiv.org/abs/2307.15593) 的版本用 **fixed randomness** (一个预先生成的 key 序列), 而不是 PRF on k-gram, 周期性循环这个 key。这带来 robustness: 能容忍 constant rate of edit。代价是检测时要 brute-force 搜索 $L$ 个 key sequences, 计算 cost 和 FPR 都乘以 $L$。

#### 4.1.3 Undetectable Watermark (Christ et al.)

[https://arxiv.org/abs/2311.04378](https://arxiv.org/abs/2311.04378)

**核心 idea**: 确保 PRF input 永不重复, 这样 watermark 不会引入 token 间 correlation。

具体流程:
1. 从原模型输出 tokens $v_1, \ldots, v_{\ell}$
2. 直到 $v_1, \ldots, v_{\ell}$ 累积足够 empirical entropy (达到某个 threshold)
3. 用 $(v_1, \ldots, v_{\ell})$ 作为 PRF input 生成 randomness, bias 后续 tokens
4. 重复

**Empirical entropy** 是关键概念: 给定当前 logits, 实际观测到的 token 序列的"意外程度"。entropy 高意味着 PRF input 含有足够 randomness, 不会与之前 PRF input 重复。

**Robustness 局限**: 任何一个 $(v_1, \ldots, v_{\ell})$ 中的 token 被修改, 整个 watermark 失效。可以通过"block"结构缓解 cropping — 每个 block 独立 embed, 但对 substitution 仍不 robust。

#### 4.1.4 Pseudorandom Error-correcting Codes (PRC Watermark)

Christ & Gunn [https://eprint.iacr.org/2024/1086](https://eprint.iacr.org/2024/1086)

这是 watermarking 领域跟 cryptography 最深度融合的工作。

**核心 idea**: 用 PRC (pseudorandom error-correcting code) 来构造 watermark。PRC 的 codeword 同时满足两个性质:
1. **看起来 random**: 不能与 random string 区分开 (这就保证了 undetectability)
2. **robust**: codeword 经过一定 fraction 的 bit flip 后, decoder 仍能恢复

Watermark 流程: 采样 LLM response 使其 tokens 与一个 PRC codeword 有显著 correlation。检测时, 把 PRC decoder 应用到 response 上, 如果没被改太多, decoder 成功, watermark 检出。

数学上, PRC 是一个 $[n, k, d]$ code, $n$ 是 codeword 长度, $k$ 是 message 长度, $d$ 是 minimum distance。但 PRC 的 codeword 集合本身是 pseudorandom 选择的, 不能与 random code 区分。

**优势**: 第一个同时实现 undetectable + robust to constant fraction of substitution 的 watermark。**局限**: 目前还没有 practical implementation, PRC 在实际 LLM 上的效率还需要验证。后续工作 [https://arxiv.org/abs/2410.07369](https://arxiv.org/abs/2410.07369) 在 image 上实现了类似 idea。

#### 4.1.5 Semantic Sentence Watermark (SemStamp / k-SemStamp)

[https://arxiv.org/abs/2310.03991](https://arxiv.org/abs/2310.03991), [https://arxiv.org/abs/2402.11399](https://arxiv.org/abs/2402.11399)

**核心 idea**: 在 sentence 级别做 watermark, 利用 semantic 表示空间。SemStamp 用 LSH 把候选 sentence 映射到 semantic watermark space, 通过 rejection sampling 使每个生成 sentence 落在 valid region。检测: 对 valid region sentence 数量做 one-proportion z-test。

**直觉**: token-level watermark 对 paraphrase 不 robust, 因为 paraphrase 会换 token 但保留 sentence semantics。sentence-level watermark 把 watermark 放在 semantics 上, paraphrase 后 watermark 仍能检出 (前提是 semantic 表示相对不变)。

k-SemStamp 改进: 用 k-means clustering 替代 LSH, 把 cluster 信息作为 intrinsic semantic 信息。但这两个方案都缺 theoretical guarantee。

### 4.2 Image Watermarks

#### 4.2.1 Stable Signature (Fernandez et al.)

[https://arxiv.org/abs/2303.15435](https://arxiv.org/abs/2303.15435)

**机制**: fine-tune Latent Diffusion Model (LDM) decoder, 使其所有输出 embed 一个 fixed binary signature $m$。

两步:
1. 预训练 watermark extractor $W$: 从 image 恢复 binary message
2. Fine-tune LDM decoder $D$: 使 $D$ 的所有输出都能让 $W$ 恢复出 fixed signature $m$

**Limitation**: 
- Quality 退化 (因为是 model parameter 层面修改)
- 对 regeneration attack (用另一个 diffusion model 重新生成) 不 robust
- 无 theoretical guarantee
- 类似方案 DiffusionDM [https://arxiv.org/abs/2303.10137](https://arxiv.org/abs/2303.10137) 也面临类似问题

#### 4.2.2 Tree-Ring Watermark (Wen et al.)

[https://arxiv.org/abs/2305.20030](https://arxiv.org/abs/2305.20030)

**机制**: 在 diffusion model 的 latent space 中, 把初始 noise 在 Fourier domain 中某些 concentric rings 强制设为 0。检测: 用 DDIM inversion [https://arxiv.org/abs/2010.02502](https://arxiv.org/abs/2010.02502) 估计 initial latent, 看 watermark rings 区域的值是否异常小。

**架构图解析**:

```
Standard diffusion:
  noise z_T (Gaussian) → DDIM reverse process → image

Tree-Ring:
  z_T with rings=0 in Fourier domain → DDIM reverse process → image
  
Detection:
  image → DDIM forward (inversion) → estimated z_T_0
  check rings in Fourier(estimated z_T_0) ≈ 0?
```

**Limitations**:
- 引入显著 latent 分布偏离, 降低 image quality 和 variability
- Zero-bit scheme, 不能 embed message
- 对 adversarial surrogate attack 脆弱: latent pattern 容易被 neural network 学到 (Saberi et al. [https://arxiv.org/abs/2310.11451](https://arxiv.org/abs/2310.11451))

#### 4.2.3 Gaussian Shading Watermark (Yang et al.)

[https://arxiv.org/abs/2404.04956](https://arxiv.org/abs/2404.04956)

**机制**: 用 watermarking key 把 latent space sampling 限制到一个 fixed quadrant。

数学上: 标准 diffusion 的 $z_T \sim \mathcal{N}(0, I)$。Gaussian Shading 用 key 决定一个 quadrant $Q$, 然后 sample $z_T \sim \mathcal{N}(0, I) |_{z_T \in Q}$ (truncated Gaussian)。

**Quality claim**: 论文声称 "lossless performance", 通过证明单个 watermarked image 分布等于 un-watermarked image 分布。但这个 proof 没考虑跨多个 generation 的 correlation — 所有 image 都从同一 quadrant 出发, 长尾上系统性偏离 $\mathcal{N}(0, I)$, 影响 FID、Inception Score 等多 image 统计。

#### 4.2.4 PRC Watermark for Images (Gunn et al.)

[https://arxiv.org/abs/2410.07369](https://arxiv.org/abs/2410.07369)

这是 Section 4.1.4 PRC watermark 的 image 版本。

**机制**: 类似 Gaussian Shading, 每次 generation 用 PRC sample 一个 fresh quadrant, 而不是 fixed quadrant。

**关键优势**:
- Undetectability imply 不退化任何 quality metric, 包括跨 generation 的 metric (FID, CLIP Score, Inception Score)
- Robustness 来自 PRC 的 error-correcting property
- 支持 multi-bit message

**直觉对比**: Gaussian Shading 用 fixed quadrant 简单但退化 multi-image statistic; PRC 用 pseudorandom quadrant (看起来 random, 但其实有 structure) 既保证 single-image quality 又保证 multi-image quality。这正是 PRC 的精髓: codeword 看起来 random 但有 coding structure。

---

## 5. Empirical Evaluation

### 5.1 Detection Metrics

- **AUROC**: trade-off between TPR 和 FPR
- **Fixed FPR comparison**: 在固定低 FPR (e.g. 0.1%) 下比较 TPR, 因为高 FPR 在 misinformation 场景 untenable

### 5.2 Attack Categories

论文把 evasion attack 分三类:

1. **Edit Attacks**: 小幅 local 修改 (text deletion, image distortion, synonym replacement)。可加 optimization, 如 white-box 攻击 [https://arxiv.org/abs/2309.10704](https://arxiv.org/abs/2309.10704) 通过 PGD 优化 adversarial noise

2. **Regeneration Attacks**: 把 watermarked output 喂给另一个 (non-watermarked) GenAI model。比如 paraphrasing model 改写 watermarked text [https://arxiv.org/abs/2310.03991](https://arxiv.org/abs/2310.03991), 或者 denoising autoencoder 处理 watermarked image [https://arxiv.org/abs/2406.12527](https://arxiv.org/abs/2406.12527)

3. **Downsampling Attacks**: 让 watermarked output 包含真正想要的 output 作为 subset, 然后提取 subset 破坏 watermark。代表: **Emoji Attack** (Pineapple Attack) — prompt LLM 在词之间插入 emoji, 生成后删除 emoji, 严重破坏基于 k-gram 统计的 watermark

**Forgery**: 可以用 surrogate model 在 watermarked + unwatermarked image 上训练, 然后用 PGD 让 unwatermarked image 伪造 watermark signal [https://arxiv.org/abs/2310.11451](https://arxiv.org/abs/2310.11451)。

### 5.3 Quality Metrics

**Text**:
- Perplexity (PPL): fluency
- Diversity: n-gram repetition, Distinct n-grams, Self-BLEU [https://arxiv.org/abs/1804.06438](https://arxiv.org/abs/1804.06438)
- MAUVE [https://arxiv.org/abs/2102.01436](https://arxiv.org/abs/2102.01436): distributional similarity to human text
- LLM-as-a-Judge
- Task-specific: ROUGE, BLEU, BERTScore, BARTScore, InstructScore, Pass@k for code

**Image**:
- PSNR, SSIM: post-generation watermarking quality
- FID [https://arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500), CLIP Score [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020), Inception Score, LPIPS [https://arxiv.org/abs/1801.03924](https://arxiv.org/abs/1801.03924)
- DreamSim, BLIP Score, ImageReward

---

## 6. Open Problems

### 6.1 Robustness + Unforgeable Public Attribution 的张力

公开 attribution key 让所有人都能 detect, 这对 misinformation detection 很有用, 但与 robustness 不兼容。Christodorescu et al. [https://arxiv.org/abs/2404.06009](https://arxiv.org/abs/2404.06009) 强调 policy 应该认知到 watermark 不同 scheme 的能力差异。

### 6.2 Copyright Proof

一个微妙问题: model owner 想 "证明" 某张 image 是他的模型生成的, 可以 reveal watermark key。但问题在于 — 对某些现有 scheme, 恶意 model owner 可以反过来, 给定一张 user-generated image, **制造一个 fake watermark key** 让该 image 在这个 key 下被 detected as watermarked。所以 reveal key 不能证明什么。

需要的 property: adversary 给定 content, 难以制造一个 watermark key 使该 content 被 detect。可以通过要求 model owner 预先 publish commitment to watermark key 来实现。

### 6.3 Open-Source Model Watermarking

Open-source model 用户可以任意修改 decoding, 现有 watermarking 方法难以 enforce。一个 promising 方向: 在 model parameter 中 train-in watermark, 让 model 本身 "天生" embed watermark, 不依赖于 decoding 过程的修改。Zhao et al. [https://arxiv.org/abs/2210.03312](https://arxiv.org/abs/2210.03312), [https://arxiv.org/abs/2302.07511](https://arxiv.org/abs/2302.07511) 探索了 distillation-resistant watermark。

### 6.4 Privacy Risk

一个被低估的 risk: watermark 可以被用来 violate privacy。例如, model provider 在每个 user 生成的 image 中 embed user ID (技术上可行, multi-bit watermark 支持)。如果该 user 把 image 发到 anonymous social media account, 任何有 detector access 的人都能 link account 到 user identity。这种 tracking 比传统 fingerprint (例如不同 whitespace pattern) 更隐蔽且 provably imperceptible — undetectable watermark 让用户根本不知道自己被 track。

---

## 7. 技术脉络与直觉总结

让我把这个领域的发展逻辑梳理一下, 帮助 build intuition:

**第一阶段 — Heuristic Watermark**: Green-Red 简单有效, 但有 low-distortion (单 token) 但 multi-response 上偏移明显。FPR 可以通过 z-test 给出 closed-form bound, 这是其流行的重要原因。

**第二阶段 — Distribution-preserving Watermark**: Gumbel trick (Aaronson, Kuditipudi) 实现 single-response distortion-free。关键 insight 是把 watermark randomness 和 sampling randomness 用同一个机制处理 — Gumbel max trick 在不破坏 distribution 的情况下做 sampling。

**第三阶段 — Undetectable Watermark**: Christ et al. 把 distortion-free 推广到 multi-response, 通过 PRF input 永不重复 (利用 empirical entropy threshold)。这给出了第一个有 quality guarantee 的 watermark。

**第四阶段 — Robust + Undetectable**: PRC (Christ & Gunn) 把 coding theory 拉进来, 第一次同时实现 undetectable 和 robust to constant fraction edit。这是 watermark 与 cryptography 深度融合的开始, PRC 同时具备 pseudorandomness (保证 undetectability) 和 error-correction (保证 robustness)。

**Cross-modal insight**: 同样的 PRC idea 可以从 text 推广到 image (Gunn et al. 的 image PRC watermark)。Gaussian Shading 是 PRC 的特殊退化情况 (fixed quadrant 而非 pseudorandom quadrant), 因此牺牲了 multi-image undetectability。

**Fundamental tradeoff**: 
- robustness vs. unforgeability (通过 detect/attribute 分离解决)
- undetectability vs. statistical undetectability (后者 impossible)
- quality vs. detectability (δ 在 Green-Red 中, quadrant 限制在 Gaussian Shading 中)
- robustness vs. FPR (Kuditipudi 用 fixed randomness 提高 robustness, 但 FPR 乘以 $L$)

**The "no free lunch" theorem** [https://arxiv.org/abs/2402.16187](https://arxiv.org/abs/2402.16187): 不存在同时所有维度最优的 watermark, 必须 trade-off。最优 scheme 取决于 application scenario — 比如用于训练数据清洗的 watermark 不需要 robustness (用户不会主动 remove), 用于 misinformation detection 的 watermark 需要 low FPR + robustness, 用于 attribution 的需要 unforgeability。

---

## 8. 关于 Policy 与 Technology 的 Gap

论文 Section 2.4 详述了各国 policy。一个核心 point: legal documents 通常 vague, 没有明确说明什么算 "watermark"。EU AI Act Article 50 要求 "machine-readable format" 的 marking, Recital 133 提到 watermark 和 cryptographic methods, 但具体实现细节留给 European Commission guidelines。

Christodorescu et al. [https://arxiv.org/abs/2404.06009](https://arxiv.org/abs/2404.06009) 强调, 政策应该 align with watermarking 技术的实际能力和限制, 否则会出台不可执行的法规。例如, 如果法律要求 watermark 在所有 transformation 下都 robust, 而技术上无法实现 (对 regeneration attack 等), 法规就成了空文。

---

## 9. 我对这篇 paper 的看法

这篇 SoK 写得相当 thorough, 把 watermarking 在 GenAI 语境下的 formal foundation (Definition 3.1-3.6)、threat model、evaluation methodology、representative schemes 系统化。几个特别有价值的点:

1. **Quality 定义的层次化**: 把 empirical validation、low-distortion、distortion-free、undetectability 这四个概念清楚地区分开, 并指出它们在 single-response vs. multi-response、heuristic vs. provable 维度上的差异。这对一个新进入者快速建立 mental model 很有帮助。

2. **Detect vs. Attribute 的分离**: 论文明确指出 robustness 和 unforgeability 的 fundamental incompatibility, 并提出用不同 algorithm/key 解决。这个 conceptual 区分在之前的工作里经常被混淆。

3. **Policy 与 Technical Reality 的连接**: Section 2.4 把各国 policy 梳理得很清楚, Section 7 又讨论 policy 挑战。这在 SoK 里比较罕见, 帮助 bridge 研究者和 policy maker。

不足之处:
- Section 6.3 video/audio watermark 部分很薄, 只有 paragraph
- 缺少一个统一的实验 comparison table (各方案在 FPR、robustness、quality、cost 上的 head-to-head)
- Open problem 中 model watermark 和 dataset watermark 只是 briefly mentioned, 没有深入

总体而言, 这是一篇质量很高的 SoK, 适合作为 GenAI watermarking 领域的入门参考。配合 MarkLLM toolkit [https://arxiv.org/abs/2405.10051](https://arxiv.org/abs/2405.10051) 一起读, 能快速上手做实验。

---

## References

主要 reference:

- Kirchenbauer et al. "A Watermark for Large Language Models" [https://arxiv.org/abs/2301.10226](https://arxiv.org/abs/2301.10226)
- Aaronson's talk [https://openai.com/blog/scott-aaronson-on-watermarking/](https://openai.com/blog/scott-aaronson-on-watermarking/)
- Christ, Gunn, Zamir "Undetectable Watermarks for Language Models" [https://arxiv.org/abs/2311.04378](https://arxiv.org/abs/2311.04378)
- Kuditipudi et al. "Robust Distortion-free Watermarks" [https://arxiv.org/abs/2307.15593](https://arxiv.org/abs/2307.15593)
- Christ & Gunn "Pseudorandom Error-Correcting Codes" [https://eprint.iacr.org/2024/1086](https://eprint.iacr.org/2024/1086)
- Fairoze et al. "Publicly Detectable Watermarking" [https://arxiv.org/abs/2310.18491](https://arxiv.org/abs/2310.18491)
- Wen et al. "Tree-Ring Watermarks" [https://arxiv.org/abs/2305.20030](https://arxiv.org/abs/2305.20030)
- Gunn, Zhao, Song "Undetectable Watermark for Generative Image Models" [https://arxiv.org/abs/2410.07369](https://arxiv.org/abs/2410.07369)
- Fernandez et al. "Stable Signature" [https://arxiv.org/abs/2303.15435](https://arxiv.org/abs/2303.15435)
- Yang et al. "Gaussian Shading" [https://arxiv.org/abs/2404.04956](https://arxiv.org/abs/2404.04956)
- SemStamp [https://arxiv.org/abs/2310.03991](https://arxiv.org/abs/2310.03991)
- SynthID [https://deepmind.google/technologies/synthid/](https://deepmind.google/technologies/synthid/)
- SynthID-Text Nature paper [https://www.nature.com/articles/s41586-024-07566-y](https://www.nature.com/articles/s41586-024-07566-y)
- C2PA [https://c2pa.org/](https://c2pa.org/)
- WavMark [https://arxiv.org/abs/2308.12770](https://arxiv.org/abs/2308.12770)
- WAvES benchmark [https://arxiv.org/abs/2406.08633](https://arxiv.org/abs/2406.08633)
- Saberi et al. image detector attacks [https://arxiv.org/abs/2310.11451](https://arxiv.org/abs/2310.11451)
- MarkLLM [https://arxiv.org/abs/2405.10051](https://arxiv.org/abs/2405.10051)
