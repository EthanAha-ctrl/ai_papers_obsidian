---
source_pdf: LLaDA2.0-Uni Unifying Multimodal Understanding and.pdf
paper_sha256: 0682a19c4e6aa49d7609eb9bd39b153fc39fac41934913682faf3902efcb5109
processed_at: '2026-08-05T15:14:30-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 LLaDA2.0-Uni

Andrej，我换个方式讲，尽量让你三分钟抓到 essence。

---

## 一句话总结

**之前用 diffusion 做 unified multimodal 一直做不好，这篇 work 把三个关键短板都补上了，第一次让 diffusion-based unified model 在 understanding 上不输 specialist VLM。**

---

## 背景为什么乱

unified multimodal 这个赛道现在两派人马在打架：

**AR 派**（Janus, BAGEL, OmniGen2）：text 用 next-token prediction，image 也用 next-token prediction（或者 hybrid AR + diffusion decoder）。好处是 LLM 范式直接复用，坏处是 image generation 要 sequential 采样几千个 token，慢得要死。

**Diffusion 派**（MMaDA, Lumina-DiMOO, LLaDA-o）：text 和 image 都用 masked diffusion，block-level 并行 decode。好处是快 + 双向 context；坏处是之前几个工作 performance 全面被 AR 派吊打。

被吊打的原因 paper 在 Section 2.1 点得很清楚，三个：

1. **VQ tokenizer 是 reconstruction 训练的，没语义**——所以 understanding 任务死在 OCR / document 这种需要细粒度语义的地方。Lumina-DiMOO 在 OCRBench 上只有 7.6 分，基本是 0
2. **VQ 压缩太狠**——generation 质量也崩
3. **纯 bidirectional attention 对 text 不可靠**——text 的 "what comes next" 强依赖因果，双向会糊

LLaDA2.0-Uni 就是针对这三点分别开药。

---

## 三个关键工程选择

### 选择一：换 tokenizer

这是灵魂。

之前的 VQ-VAE 是拿 pixel reconstruction loss 训的，codebook 学到的是 "texture patch" 这种低级 stuff。让它去做 OCR 这种需要语义的任务，等于让一个只学过素描的人去做阅读理解。

LLaDA2.0-Uni 用的 SigLIP-VQ 直接拿 SigLIP2-g 这个 ViT 当 encoder，然后在 understanding task 上训 codebook（codebook size 16384, dim 2048）。意思是 codebook 的每个 entry 自带 "dog's ear shape" 这种 semantics，直接对齐 Qwen2.5 的语义空间。

代价是 SigLIP-VQ **没法直接 decode 回 image**——它只会语义不会像素。所以需要第三个组件补上。

参考 X-Omni: https://arxiv.org/abs/2507.22058

### 选择二：block-wise attention

纯 bidirectional attention 对 text 不好这件事，LLaDA / MMaDA / Lumina-DiMOO 之前都验证过。直觉上很好理解——你让模型同时看到 "前面已经 committed 的 token" 和 "应该被预测的 token"，它分不清谁是谁，text 的因果结构就糊了。

LLaDA2.0-Uni 用的是 BDLM (Block Diffusion, https://openreview.net/forum?id=LEYc8MB4tP) 的 scheme：

- sequence 切成 block
- block 内部 full bidirectional（享受 parallel decoding）
- block 之间 causal（保留 "前面 block 已经确定" 的因果结构）

paper 还透露一个我觉得很关键的小细节：因为 SigLIP-VQ 是跟 Qwen2.5 对齐的，它的 token 序列**本身就带 AR bias**。纯双向 attention 会破坏这个 bias。block-wise 相当于 "尊重 SigLIP-VQ 的 inductive bias"。

这个 ablation paper 没做，但我猜如果换成纯 ViT feature（不带 AR bias 的 encoder），block-wise 相对 full bidirectional 的优势就没那么明显了。

### 选择三：独立 diffusion decoder

因为 SigLIP-VQ 不能直接 decode 成 image，需要一个专门的 decoder 把 semantic token 翻译回像素。

他们直接拿 Z-Image-Base（6B 的 text-to-image 模型, https://arxiv.org/abs/2511.22699）改了一下，把原本的 text prompt conditioning 换成 dLLM 输出的 semantic visual tokens。

这里有个**反直觉的工程选择**：他们**只**用 visual tokens 做 conditioning，**不**再喂 text prompt。这跟 NextFlow / X-Omni 不同——那两个是 text + visual tokens 冗余地一起喂。LLaDA2.0-Uni 的论点是：semantic tokens 已经 encode 了 prompt 的所有信息（dLLM 就是基于 prompt 生成它们的），再喂 text 是 redundant 还会让两个 modality 信号打架。

然后用 consistency distillation 把 50 步压到 8 步，11.4× speedup，性能几乎不掉。

---

## 训练 recipe 三阶段

跟主流 unified model 差不多：

- **Stage 0** (100B tokens, 8k ctx)：vision-language alignment，只用 image-caption + text。mask 策略是 generation 任务只 mask image token，understanding 任务只 mask text token——这个 decoupled masking 是个 important detail
- **Stage 1** (210B tokens)：加 OCR / grounding / video / image editing / interleaved generation
- **Stage 2** (80B tokens)：高质量 SFT，8k → 16k context

Decoder 独立训三阶段：warm-up（冻结 semantic processor）→ multi-domain → high-fidelity refinement。

---

## SPRINT：我觉得最 interesting 的工程贡献

DLLM 推理慢这个事大家都知道。BDLM 要 $B \times T$ 次 forward pass（B 个 block × T 个 denoising step）。SPRINT 是 training-free 的加速，两个 axis 各打一拳：

### Axis 1：Sparse Prefix Retention

每个 denoising step 都要 attend 整个 prefix，attention 是 quadratic 的。SPRINT 在每个 block 的第一步做 full forward，拿到 KV cache，然后基于 importance score 剪枝 prefix，后续 step 只 attend 剪枝后的版本。

Importance score 是两个信号的加权平均：
- **key norm**：position $i$ 的 key vector 的 L2 norm，衡量它对 attention distribution 的 pull 强度
- **top-1 confidence**：模型在这个 position 上的预测确定性

直觉：key norm 大 = 这个 position 容易被 attend 到（数学上 dot-product attention 里 key norm 大的项容易胜出）；confidence 高 = 这个 position 已经基本确定，保留它对后续 denoising 有帮助。

然后 modality-aware 剪枝：text token keep ratio 1.0（一个都不剪），image token keep ratio 0.8。理由是 image 在 spatial 上有冗余（neighboring patches 通常相似），剪 20% 没事；text 是 instruction / reasoning chain，剪一个可能丢关键信息。

全局 keep ratio 0.5——prefix 实际 attend 长度砍一半。

### Axis 2：Non-uniform Token Unmasking

标准 schedule 每 step unmask 固定数量的 token，无视 confidence。某些 token 模型一开始就 confident，再 refine 是浪费；某些 token 模型很 uncertain，给一个 step 不够。

SPRINT 改成 confidence-adaptive：只要某个 position 的 top-1 confidence 超过 threshold τ（0.93 或 0.95），就接受。同时强制每 step 至少 unmask $\lceil m/(T-t) \rceil$ 个，保证 termination。

效果：confident token 一个 step 就 unmask，uncertain token 多给几个 step——相当于把 denoising budget 动态分配给难 token。

### SPRINT 实验结果

平均 score 只降 0.6（76.3 → 75.7），speedup 1.6×。DocVQA 因为输出最长，speedup 3.5×。MMMU 和 ChartQA 反而**提升** +2.4 和 +0.9——这是 non-uniform unmasking 的功劳，难 token 多得了 refinement。

OCRBench 掉 2.3 分——因为 OCR 需要精确字符，τ=0.93 太激进，token 还没充分 refine 就被接受。这是 SPRINT 的 boundary case。

---

## 实验结果的人话版

### Understanding

- vs specialist Qwen2.5-VL：general VQA 已经持平（MMStar 64.1 vs 63.9），reasoning 和 OCR 还有 gap（MMMU 50.1 vs 51.3, DocVQA 89.5 vs 94.9）
- vs AR-based unified BAGEL：reasoning 上明显落后（WeMath 29.3 vs 45.8, MMMU 50.1 vs 55.3）。BAGEL 是 hybrid AR+diffusion，text reasoning 仍然 AR backbone 在做，先天优势
- vs D-Diff baseline：**全面碾压**。Lumina-DiMOO 在 OCR 上几乎 0 分（7.6），LLaDA-o 在 MMBench-CN 只有 69.9——这两个数字就是之前 D-Diff unified model 的"理解差"诅咒

### Generation

- GenEval 0.89, DPG 87.76, UniGenBench 79.63——全部 unified model SOTA
- Position 子项 0.90 是**所有 model**（包括 specialist）最高
- vs specialist Z-Image-Turbo：DPG 87.76 vs 84.86，反超 specialist

### Editing

- ImgEdit 3.92 unified 第一
- MICo-Bench 47.1 **所有 model SOTA**，比 Qwen-Image-Edit 35.9 高 31%

MICo-Bench 这个 gap 我觉得很有意思。multi-reference editing 需要同时理解多张图 + 生成融合图。dLLM 的 bidirectional context 天然适合这种 "并行融合多 source" 的任务。AR-based 必须 sequential 处理，融合效率低。

而且 Lumina-DiMOO 同架构只有 23.3，说明光有 dLLM 不够——**SigLIP-VQ 提供的 semantic token 才是关键**。reconstructive VQ 会丢 reference image 的细节，fusion 时就糊了。

### WISE-Bench + Thinking：最 under-discussed 的结果

WISE-Bench 测 reasoning-informed image generation（world knowledge + 物理常识）。

- 不带 thinking：0.68
- 带 thinking：**0.78**，提升 14.7%

这个数字我觉得是 paper 里最 exciting 的 hint。它暗示 unified model + CoT 在 image generation 上的提升远大于 text-only reasoning 单独的提升。visual generation 能从 reasoning 受益，反过来也成立。

Figure 8 给的国际象棋例子就是这种 interleaved reasoning 的 instance——模型边说 "考虑 Kc4 这个 move..." 边输出 image 表示"如果走 Kc4 棋盘会变成这样"。这种 "边 reason 边 generate" 的 capability 只有 diffusion-based unified 才有——AR 必须 reasoning 完 commit 之后才能 generate image。

---

## 我的几个判断

### 1. SigLIP-VQ 是真正的分水岭

之前 D-Diff unified model 卡在 understanding 差，根因是 VQ-VAE codebook 缺语义。SigLIP-VQ 把 codebook 直接训成语义对齐的，相当于把 "理解用 ViT + 生成用 VAE" 这个 decoupled 设计 collapse 回 single encoder，但牺牲了直接 reconstruction。再用 diffusion decoder 补上 reconstruction 这一环。

这是一个很有原则的 trade-off：让 tokenizer 专注一件事（semantic），把 reconstruction 留给专门 decoder。

### 2. Block-wise attention 是为 SigLIP-VQ 量身定做

完全 bidirectional 不行，完全 AR 又丧失 diffusion 优势。Block-wise 是中间道路。但 paper 暗示 SigLIP-VQ token 因为跟 Qwen2.5 对齐，**本身就有 AR bias**。如果用纯 bidirectional，这个 bias 被破坏。Block-wise 让 block 间保留 AR 结构，相当于 "尊重 SigLIP-VQ 的 inductive bias"。

### 3. SPRINT 是 dLLM 真正的工程胜利

DLLM 最大实际问题是 inference 慢。SPRINT 两个 axis 各打一拳，training-free，1.6× speedup 几乎不掉分。这种"把 prefix KV cache 剪枝 + 动态 unmasking schedule"的组合我觉得会变成 dLLM inference 的标配。

但 OCRBench 掉 2.3 说明有 boundary case：精确字符任务对 aggressive unmasking 敏感。可能需要 task-aware τ schedule。

### 4. Reasoning 上短期追不上 AR 派

MMMU / WeMath 上 BAGEL 明显领先。AR backbone 在 reasoning 上的先天优势——text reasoning 是 AR 的主场——diffusion 短期内难追。这是 unified diffusion model 的结构性限制。

### 5. Interleaved reasoning 是真正的 frontier

WISE + thinking 的 14.7% 提升暗示这条路有 promise。AR-based unified 也能做 CoT，但它们 reasoning 是 text-only 模式，reasoning 完再 generate image。Diffusion-based unified 在 principle 上可以做 "边 reason 边 generate"，让 image token 和 reasoning token interleaved 出现。这是 diffusion unified 相对 AR unified 的**结构性优势**，目前还没被充分开发。

---

## 不足的地方

1. **MMMU / WeMath 落后 BAGEL**——reasoning gap 是结构性的
2. **Fine-grained visual detail 弱**——paper 自己承认 SigLIP-VQ 在 detail-sensitive 任务（比如高精度 editing）上会 bottleneck
3. **InterGen benchmark 太小**——150 samples 统计意义有限
4. **SPRINT 在 OCR 上退化**——confidence-based unmasking 对精确字符任务不友好
5. **缺 scaling law 分析**——16B 是个特定 size，dLLM unified 的 scaling behavior 跟 AR 可能很不一样
6. **没有 RL 探索**——Conclusion 提了 "begun exploring RL" 但没数据
7. **Speed 数据不够细**——没拆解 SPRINT 两个 axis 各贡献多少

---

## 一句话 takeaway

> 它不是 "another unified multimodal model"，是 "the first unified dLLM that doesn't suck at understanding"。

之前 D-Diff unified 一直在 "理解差" 的诅咒下挣扎。LLaDA2.0-Uni 通过 SigLIP-VQ + block-wise attention + 独立 diffusion decoder 这个 triple combo，第一次让 D-Diff unified 在 understanding 上接近 specialist VLM 水平，同时在 generation 上保持 SOTA。

验证了一个 thesis：**dLLM 的 unified multimodal 设定可以 work，只要 token representation 选对**。Token representation 是 foundation——它决定了下游所有任务的天花板。VQ-VAE 让 D-Diff 卡了半年，SigLIP-VQ 解开了这个结。

接下来的 frontier 我认为就三个词：**interleaved reasoning / dLLM-specific RL / scaling law**。

---

参考：
- Paper GitHub: https://github.com/inclusionAI/LLaDA2.0-Uni
- HuggingFace: https://huggingface.co/inclusionAI/LLaDA2.0-Uni
- LLaDA2.0 main paper: https://arxiv.org/abs/2512.15745
- LLaDA original: https://arxiv.org/abs/2502.09992
- BDLM: https://openreview.net/forum?id=LEYc8MB4tP
- MMaDA: https://arxiv.org/abs/2505.15809
- Lumina-DiMOO: https://arxiv.org/abs/2510.06308
- BAGEL: https://arxiv.org/abs/2505.14683
- Z-Image: https://arxiv.org/abs/2511.22699
- SigLIP2: https://arxiv.org/abs/2502.14786
- X-Omni: https://arxiv.org/abs/2507.22058
- Flow Matching: https://arxiv.org/abs/2210.02747
- Consistency Models (sCM): https://arxiv.org/abs/2410.11081
- DeepSeek-V3 (load balancing): https://arxiv.org/abs/2412.19437
- SGLang: https://arxiv.org/abs/2312.07104

---

# LLaDA2.0-Uni 深度解读

很高兴跟你聊这篇 paper，Andrej。我读完之后的直觉是：这篇工作的真正贡献不在于"统一了 understanding 和 generation"这个口号——这个口号已经被喊了一年了——而在于它**认真回答了 "为什么 discrete diffusion 适合做 unified multimodal" 这个问题**，并且用一组很扎实的工程选择把之前 MMaDA / Lumina-DiMOO / LLaDA-o 的短板都补上了。下面我尽量 build your intuition。

---

## 1. 这篇 paper 真正在解决什么问题

之前的 unified multimodal 模型分两派：

**AR 派**（Janus, Lumina-mGPT, BAGEL, OmniGen2, Hunyuan Image 3.0）：用 next-token prediction 统一两种模态。优点是 LLM 的训练范式直接复用；缺点是 image generation 上 AR 的 latency 灾难（要 sequential 采样几千个 token）。

**Masked diffusion 派**（MMaDA, Lumina-DiMOO, LLaDA-o）：用 mask-prediction 统一。优点是 parallel decoding + bidirectional context；缺点是之前几个工作 performance 全面落后 AR 派。

落后的根因有三个，paper 在 Section 2.1 把它们点出来了：
1. reconstructive VQ tokenizer 缺语义 → understanding 性能差
2. VQ 压缩率过高 → generation 质量差  
3. full bidirectional attention 对 text 不可靠

LLaDA2.0-Uni 的 thesis 一句话总结：**用完全 semantic 的离散 token（SigLIP-VQ）+ block-wise attention + MoE dLLM + 独立 diffusion decoder**，三个工程选择分别打掉上面三个短板。

GitHub: https://github.com/inclusionAI/LLaDA2.0-Uni  
HuggingFace: https://huggingface.co/inclusionAI/LLaDA2.0-Uni

---

## 2. 架构总览

```
Image ──► SigLIP-VQ ──► discrete semantic tokens
                              │
Text  ─────────────────────►  │
                              ▼
                  16B MoE dLLM (LLaDA2.0-mini)
                  block-wise masked diffusion
                              │
              ┌───────────────┴───────────────┐
              ▼                                ▼
        text tokens                  visual tokens
        (direct output)                  │
                                         ▼
                              Diffusion Decoder (Z-Image-base, 6B)
                              flow matching + 8-step distillation
                                         ▼
                                    High-fidelity image
```

三个组件我分别 build intuition。

### 2.1 SigLIP-VQ Tokenizer

关键创新在这里。传统 VQ-VAE（Esser 2021, Wang 2024b）的 codebook 是用 pixel reconstruction loss 训练的，codebook 的语义信息很弱。Janus / MMaDA 用这种 VQ 来做 understanding 时就死在 OCRBench / DocVQA 这种需要细粒度语义的任务上——Table 2 里 Lumina-DiMOO 的 OCRBench 只有 7.6，而 LLaDA2.0-Uni 是 75.7，差了 10 倍。

SigLIP-VQ 的做法：
- 用 SigLIP2-g ViT（Tschannen 2025, https://arxiv.org/abs/2502.14786）作为 visual feature extractor
- 接一个 vector quantizer，codebook size = 16384，dim = 2048
- **直接在 understanding task 上训练**，让 codebook 自带语义

这背后的 intuition 是：VQ 的离散化是必要的（因为你后面要在 dLLM 里跑 discrete diffusion），但 codebook 学什么不一定要靠 pixel reconstruction loss。让它对齐 Qwen2.5 的语义空间，codebook 的每个 entry 自然就有 "dog's ear shape" 这种 semantics，比 VQ-VAE 那种 "patch of texture" 强多了。

但 SigLIP-VQ 牺牲了什么？**没有 native reconstruction mechanism**。这就引出第三个组件——diffusion decoder。

参考：
- SigLIP2 paper: https://arxiv.org/abs/2502.14786  
- X-Omni (SigLIP-VQ 架构来源): https://arxiv.org/abs/2507.22058

### 2.2 16B MoE dLLM Backbone

用的是 LLaDA2.0-mini（Bie et al. 2025, https://arxiv.org/abs/2512.15745），MoE 架构 16B total params。几个细节：

**Vocab 扩展**：原始 text vocab + SigLIP-VQ codebook (16384) + 一组 special tokens（用于 image generation / understanding 的边界标记）。新加的 visual token embedding 随机初始化，text 部分保留 pretrained 权重。这是个很标准的处理 multimodal extension 的方式，没什么花样。

**Block-wise Attention**：这是关键。完全 bidirectional attention 之前多个工作（Nie 2025 LLaDA, Yang 2025 MMaDA, Xin 2025a Lumina-DiMOO）都验证过会让 text 性能掉。直觉上很好理解：text token 的 "what comes next" 强依赖因果关系，纯双向会混淆 "已经在前面出现过的 token" 和 "应该被预测的 token"。

LLaDA2.0-Uni 用的是 BDLM (Arriola 2025, Block Diffusion, https://openreview.net/forum?id=BDLM) 的 scheme：
- 把 sequence 切成 block，每个 block size = $L_B$
- block 内部用 full bidirectional attention（parallel decoding 友好）
- block 之间用 causal / block-wise attention（保留 "前面 block 已经 committed 的干净 token" 这个因果结构）

这个设计特别适配 SigLIP-VQ token 的特性：因为 SigLIP-VQ 是和 Qwen2.5 对齐的，它的 token 序列**本身就带有 autoregressive bias**（i.e. position k 的语义部分依赖于 position k 之前的 token）。如果你用纯 bidirectional 全 attention，会破坏这种 bias，让 SigLIP-VQ token 的 inductive bias 失效。block-wise 在这里相当于 "保留 within-block 的扩散灵活性，但 block 间仍然让模型学到 SigLIP-VQ 的因果结构"。

**1D RoPE + size tokens**：很反直觉的一点，他们没用 2D RoPE。而是在 flattened 1D visual sequence 前面加 `<height>` 和 `<width>` tokens（例如 `<imgsize 512>`）。理由是简单。Liu 2026 / Xin 2025b / Geng 2025 都验证过这个 trick 有效。我自己感觉这个 trick 能 work 的核心是：dLLM 的 bidirectional attention 本身就会让 size tokens 的信息 broadcast 到所有 visual token，所以即使 1D position encoding，2D 结构信息也通过 size tokens 这条 path 进来了。

**MoE 的 Load Balancing**：公式 (4) 用的是 DeepSeek-V3 的 auxiliary-loss-free 机制（Liu 2024a, https://arxiv.org/abs/2412.19437）+ RMSNorm-style bias update：

$$b_i = b_i + u \cdot \frac{(F_i - Q_i)}{\sqrt{\frac{1}{n}\sum_{j=1}^{n}(F_j - Q_j)^2}}$$

变量解释：
- $b_i$：第 $i$ 个 expert 的 routing bias（加到 gate logits 上影响 token 路由）
- $u$：更新步长
- $F_i = \mathbb{E}(f_i)$：当前由 bias $b$ 诱导的 expert $i$ 的实际负载分布
- $Q_i = 1/n$：均匀分布理想值
- 分母：所有 expert load 偏差的 RMS，归一化作用

这是个 RMSNorm-style 的更新——分子是偏差，分母是偏差的 RMS norm，整体就是"在 deviation space 上做 normalization"。好处是 update 量级自适应，不会因为某次 batch 里 expert load 异常就 bias 跳变。他们还把 routing gate 输出 scale 了 2.5 倍来稳定 RMS magnitude。

---

## 3. 训练 Objective 的数学细节

### 3.1 BDLM Loss（公式 3）

$$\mathcal{L}_{\mathrm{BDLM}}(\theta) = -\mathbb{E}_{t, \boldsymbol{x}_0, \boldsymbol{x}_t} \left[ \frac{\alpha_t'}{1-\alpha_t} \sum_{k=1}^{K} \sum_{i=1}^{L_B} \mathbb{1}[\boldsymbol{x}_{t,k}^i = [\boldsymbol{\mathrm{MASK}}]] \log p_\theta(\boldsymbol{x}_{0,k}^i | \boldsymbol{x}_{0,<k}, \boldsymbol{x}_{t,k}) \right]$$

逐项拆解：
- $t$：扩散 timestep，从 [0,1] 均匀采样
- $\boldsymbol{x}_0$：干净 token 序列（训练时是 ground truth）
- $\boldsymbol{x}_t$：在 timestep $t$ 被部分 mask 的版本，mask 概率是 $1 - \alpha_t$（$\alpha_t$ 是 noise schedule，类似 DDPM 中的 $\bar{\alpha}_t$，从 1 单调降到 0）
- $\alpha_t' = d\alpha_t/dt$：noise schedule 的导数
- $\frac{\alpha_t'}{1-\alpha_t}$：diffusion-derived time weight，来自 ELBO 推导。直觉上，这个权重在 $t \to 1$（high noise）时大，因为高噪声时模型需要做"从大量 mask 中重建"的难任务，给更高 weight 是合理的
- $K = L_{\mathrm{total}} / L_B$：block 数
- $L_B$：block size
- $\boldsymbol{x}_{t,k}^i$：第 $k$ 个 block 中的第 $i$ 个 token
- $\boldsymbol{x}_{0,<k}$：所有**前面**的 block 的**干净**版本（这就是 "block-wise causal" 的体现）
- $\boldsymbol{x}_{t,k}$：当前 block $k$ 的 noisy 版本
- $\mathbb{1}[\cdot]$：indicator，只在 masked 位置计算 loss

**Intuition**：这个 objective 实际上是 "AR over blocks + diffusion within block"。前面 block 的 clean token $\boldsymbol{x}_{0,<k}$ 是 conditioning，让模型在生成 block $k$ 时**看到**前面已经生成的内容（commitment）；block 内部则是 masked diffusion，parallel decode。

这就是为什么 block-wise 设计能保留 SigLIP-VQ 的 autoregressive bias：模型学到的是"看到前面的视觉 token 之后，预测下一个 block 内部的内容"，而 block 内部用扩散做 parallel decode。

### 3.2 SFT Loss + MTRS（公式 5 + 6）

$$\mathcal{L}_{\mathrm{SFT}}(\theta) = -\mathbb{E}_{t, (c, x_0), x_t} \left[ \frac{\alpha_t'}{1-\alpha_t} \sum_{k=1}^{K} \sum_{i=1}^{L_B} \mathbb{1}[x_{t,k}^i = [\mathbf{MASK}]] \log p_\theta(x_{0,k}^i | c, x_{0,<k}, x_{t,k}) \right]$$

差别只在多了 conditioning $c$（input prompt）。

**MTRS (Mask Token Reweighting)**：

$$\mathcal{L}_{\mathrm{MTRS}} = \frac{\sum_j \beta_j \mathcal{L}_{\mathrm{SFT}}^{(j)}}{\sum_j \beta_j}, \quad \beta_j = \frac{1}{\sqrt{\sum_{k=1}^{K} \sum_{i=1}^{L_B} \mathbb{1}[x_{t,k}^{i,(j)} = [\mathbf{MASK}]]}}$$

- $j$：sample index
- $\beta_j$：第 $j$ 个 sample 的 weight，等于该 sample 中被 mask 的 token 数的**平方根倒数**
- 分母求和：所有 sample weight 之和，做归一化

**Intuition**：SFT 时 sample 长度差异极大（image generation 几千个 visual token，text QA 几十个）。两种 naive 处理都有问题：
- token-averaged loss：长 sample 主导 gradient，short sample 几乎学不到
- sample-averaged loss：短 sample 上的每个 token 权重过大，鼓励模型 "answer 简短"

MTRS 用 $\beta_j = 1/\sqrt{N_{\text{mask}}}$，相当于做了一次介于 token-average 和 sample-average 之间的插值。$N_{\text{mask}}$ 越大 $\beta$ 越小，但不是线性减小而是平方根，所以长 sample 还是比短 sample 占略多一点 weight，但不至于 dominate。这是个很 LLaDA 的 trick——diffusion 训练里的样本长度不均衡问题在 AR 训练里不那么突出（每个 sample token 数差距小），在 dLLM 里才真正需要专门处理。

### 3.3 Complementary Masking

Li 2025b (LaViDA) 提出的 trick。对一个 sequence $x_0$，构造两个 antithetical 训练实例：
- $x_t$：按 mask schedule 随机 mask
- $x_t'$：用**互补** mask（$x_t$ 没 mask 的位置在 $x_t'$ 中被 mask，反之亦然）

效果：每个 token 位置在每对 sample 中**恰好一次**未被 corrupted。数据效率翻倍，并且消除 token-level 的 sampling bias（避免某些 token 总是被 mask 的极端情况）。

### 3.4 Diffusion Decoder 训练（公式 7 + 8）

Diffusion decoder 用 flow matching（Lipman 2022, https://arxiv.org/abs/2210.02747），基于 Z-Image-Base（Cai 2025, https://arxiv.org/abs/2511.22699）。

**Flow Matching Loss（公式 7）**：

$$\mathcal{L}_{\mathrm{FM}}(\theta) = \mathbb{E}_{\boldsymbol{x}_0, \boldsymbol{x}_1, \boldsymbol{z}, t} \left[ \| \boldsymbol{v}_{\theta,t}(\boldsymbol{x}_t, \boldsymbol{z}) - \boldsymbol{v}_t \|_2^2 \right]$$

- $\boldsymbol{x}_0$：起点（高斯噪声）
- $\boldsymbol{x}_1$：终点（clean image latent）
- $\boldsymbol{z}$：条件信号（来自 dLLM 的 semantic visual tokens，**替代**传统 text prompt）
- $\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_1$：线性插值
- $\boldsymbol{v}_t = \boldsymbol{x}_1 - \boldsymbol{x}_0$：target velocity field（沿 straight path 的恒定速度）
- $\boldsymbol{v}_{\theta,t}$：网络预测的 velocity

注意，这里 $\boldsymbol{z}$ **只**用 semantic visual tokens，**不**混合 text prompt。这点跟 NextFlow（Zhang 2026b）和 X-Omni 不同——那两个工作把 text prompt 和 visual tokens 冗余地一起喂。LLaDA2.0-Uni 的论点是：semantic tokens 本身已经 encode 了所有 prompt 信息（因为 dLLM 已经基于 prompt 生成它们），再喂 text 就是 redundant，还会让两个 modality 信号打架。

**Few-step Distillation Loss（公式 8）**：

$$\mathcal{L}_{\mathrm{Distil}}(\theta) = \mathbb{E}_{\alpha_0, z, t} \left[ \lVert v_{\theta,t} - v_t \rVert_2^2 + \lVert u_{\theta,t} - v_t + t \cdot \frac{\mathrm{d}u_{\theta^-,t}}{\mathrm{d}t} \rVert_2^2 \right]$$

- $v_{\theta,t}, u_{\theta,t}$：网络的 **dual outputs**（同一 backbone，最后一层分出两个 head）
- $u_{\theta^-,t} = \mathrm{stop.grad}(u_{\theta,t})$：consistency branch 用 stop gradient
- $\mathrm{d}u_{\theta^-,t}/\mathrm{d}t$：Jacobian-vector product，用 UCGM (Sun 2025) 的二阶差分近似
- 第一项：标准 flow matching loss
- 第二项：consistency loss

**Consistency loss 的 intuition**：flow matching 的训练目标是 "给定 $t$，预测 velocity"。但推理时我们走 multi-step Euler：
$$x_{t+\Delta t} = x_t + v_{\theta,t}(x_t) \cdot \Delta t$$
consistency loss 强制 $u_{\theta,t}$（一个等价的"trajectory 中点速度"预测）满足自洽关系：
$$u_{\theta,t}(x_t) \approx v_t + t \cdot \frac{\mathrm{d}u_{\theta^-,t}}{\mathrm{d}t}$$

这相当于说：沿着已经走出来的 trajectory，下一步的 velocity 预测应该和当前的 velocity 一致（在 trajectory 切线方向上）。这就让模型**即使 inference 时用很大 step（8 步而不是 50 步）也能保持 trajectory 一致性**——本质就是 LCM (Lu & Song 2024, https://arxiv.org/abs/2410.11081) 思想在 flow matching 上的实例化。

关键工程 trick：只用一个 auxiliary projection layer 加到 decoder backbone 上做 dual output，distillation 完之后这个 auxiliary layer 丢弃，推理时只用主 head。

---

## 4. SPRINT：Training-free 推理加速

这是 paper 里我自己觉得最 interesting 的部分，因为它揭示了 dLLM 推理效率的两个独立 bottleneck。

### 4.1 问题背景

Block-wise dLLM 推理 cost 是 $B \times T$ forward passes（$B$ 个 block，每个 block $T$ 个 denoising step）。两个独立 bottleneck：
1. **per-step cost**：每个 denoising step 要 attend 整个 prefix，attention 是 quadratic
2. **step count**：固定 schedule 每 step unmask $\lceil m/T \rceil$ 个 token

SPRINT 沿这两条 axis 各做一个优化。

### 4.2 Sparse Prefix Retention（降低 per-step cost）

**核心 idea**：每个 block 第一步做 full forward pass 拿到 full KV cache，然后基于 importance score 剪枝 prefix，后续 steps 都只 attend 剪枝后的 prefix。

Importance score（公式 1）：
$$s_i = \alpha \cdot \bar{I}_i + (1-\alpha) \cdot c_i$$

$$\bar{I}_i = \frac{\|\mathbf{k}_i\|_2}{\frac{1}{L}\sum_{j=1}^{L} \|\mathbf{k}_j\|_2}$$

- $\mathbf{k}_i$：position $i$ 的 key vector
- $\bar{I}_i$：mean-normalized key norm——衡量 "position $i$ 对 attention distribution 的 pull 强度"。key norm 越大，dot-product attention 时越容易被 attend 到
- $c_i = \max_v p_\theta(v | \mathbf{x}_t)$：top-1 softmax confidence——模型在这个 position 上的预测确定性
- $\alpha = 0.5$：两个信号等权混合

**Modality-aware pruning**：text 和 image 分别维护 keep ratio $r_{\text{text}}, r_{\text{img}}$。paper 用了两个 setting：
- Selective: $r_{\text{text}} = 1.0, r_{\text{img}} = 0.8$（text 全留，image 剪 20%）
- Full: $r_{\text{text}} = r_{\text{img}} = 1.0$（不剪，作为对照）

为什么 text 不剪？因为 text token 承载 instruction 和 reasoning chain，剪掉一个可能丢失关键信息。image token 在 spatial 上有高冗余（neighboring patches 通常相似），剪 20% 几乎不影响。

全局 keep ratio $r = 0.5$——这意味着 prefix 实际 attend 的长度是原长度的一半。对长序列这个节省是巨大的。

### 4.3 Non-uniform Token Unmasking（降低 step count）

Standard schedule 每 step unmask $\lceil m/T \rceil$ 个 token，无视 confidence。这很浪费——某些 position 模型一开始就很 confident，再 refinement 也没用；某些 position 模型很 uncertain，给一个 step 还不够。

SPRINT 用 confidence-adaptive：

$$\mathcal{A} = \{ n \in [m] : c_n > \tau \}$$

- $[m]$：所有 $m$ 个 still-masked position 的集合
- $c_n$：position $n$ 的 top-1 confidence
- $\tau$：threshold（paper 用 0.93 或 0.95）
- $\mathcal{A}$：本 step 接受的所有 position

同时强制 minimum $\lceil m/(T-t) \rceil$ 个 acceptance 保证 termination（否则可能某一步所有 position 都不达 threshold 就死锁）。

**Intuition**：这相当于 "动态把 denoising budget 分配给难 token"。confident token 一个 step 就 unmask，uncertain token 多给几个 step。Table 13 的数据验证了这点——ChartQA 和 MMMU 反而**提升** +0.9 和 +2.4，因为难 token 多得了 refinement。

### 4.4 SPRINT 实验结果（Table 13）

| Bench | w/o SPRINT Score | w/ SPRINT Score | w/o TPS | w/ TPS | Speedup |
|-------|------------------|------------------|---------|--------|---------|
| DocVQA | 89.5 | 89.0 | 8.0 | 27.6 | **3.5×** |
| ChartQA | 80.1 | 81.0 | 28.7 | 62.3 | 2.2× |
| AI2D | 82.0 | 80.9 | 19.5 | 42.9 | 2.2× |
| MMMU | 50.1 | **52.5** | 49.4 | 52.2 | ~1.0× |
| OCRBench | 75.7 | 73.4 | 21.2 | 36.0 | 1.7× |
| MMStar | 64.1 | 63.0 | 31.7 | 49.2 | 1.6× |
| GenEval | 89.0 | 87.8 | 2.8 | 5.1 | 1.8× |
| DPG | 87.76 | 86.27 | 2.7 | 7.8 | 2.9× |
| **Avg** | **76.3** | **75.7** | **24.3** | **39.8** | **1.6×** |

观察：
- 平均 score 只降 0.6（76.3 → 75.7），speedup 1.6×——非常 favorable 的 trade-off
- DocVQA speedup 最大 3.5×，因为它输出最长（document transcription 通常几千 token），prefix 剪枝收益最大
- GenEval/DPG 是 image generation，输出 ~1024 个 visual token，speedup 也 2-3×
- MMMU 和 ChartQA 反而提升——这是 non-uniform unmasking 的功劳
- OCRBench 掉 2.3——因为 OCR 是 character-level prediction，τ=0.93 可能太激进，导致 token 还没充分 refine 就被接受

OCRBench 的退化揭示 SPRINT 的一个 limitations：当任务需要**精确字符**时，低 threshold 是危险的。

---

## 5. Diffusion Decoder Distillation 实验（Table 14）

| Method | Speed (s/img) | GenEval | DPG | UniGenBench | OneIG-EN | WISE |
|--------|---------------|---------|-----|-------------|----------|------|
| Decoder 50 steps | 32.95 | 0.89 | 87.76 | 79.63 | 0.505 | 0.68 |
| Decoder Turbo 8 steps | **2.90** | 0.87 | 87.24 | 79.76 | 0.500 | 0.68 |

11.4× speedup，性能几乎不变。GenEval 掉 0.02，DPG 掉 0.52——这些退化基本在 noise level。

跟 SPRINT 的 1.6× 结合，整体 generation pipeline 端到端加速可能在 5× 左右。这是个相当实际的数字。

---

## 6. 实验 Benchmark 深度分析

### 6.1 Multimodal Understanding（Table 2）

跟几个 baseline 比：

| Bench | Qwen2.5-VL-7B (specialist) | BAGEL (AR+Diff unified) | Lumina-DiMOO (D-Diff) | LLaDA-o (D-Diff+Diff) | **LLaDA2.0-Uni** |
|-------|---------------------------|--------------------------|------------------------|------------------------|------------------|
| MMStar | 63.9 | 67.0 | 61.0 | 58.0 | **64.1** |
| MMBench-EN | 83.5 | 85.0 | 84.5 | 71.1 | 81.5 |
| MMMU-val | 51.3 | 55.3 | 58.6 | 44.9 | 50.1 |
| OCRBench | 84.2 | 73.3 | 7.6 | 74.6 | **75.7** |
| DocVQA | 94.9 | 94.3 | 7.2 | 91.5 | 89.5 |
| InfoVQA | 80.3 | 60.7 | 6.2 | 54.7 | 70.1 |

**关键观察**：
1. **vs. specialist VLM (Qwen2.5-VL)**：在 general VQA 上已经持平（MMStar 64.1 vs 63.9）。但在 reasoning（MMMU 50.1 vs 51.3）和 OCR（DocVQA 89.5 vs 94.9）上还有 gap
2. **vs. AR-based unified (BAGEL)**：BAGEL 在 reasoning 上明显强（MMMU 55.3 vs 50.1, WeMath 45.8 vs 29.3）——BAGEL 是 hybrid AR+diffusion，text reasoning 仍然是 AR backbone 在做，这块有先天优势
3. **vs. D-Diff baseline (Lumina-DiMOO, LLaDA-o)**：全面碾压。Lumina-DiMOO 在 OCR 上几乎是 0 分（7.6/7.2），LLaDA-o 在 MMBench-CN 只有 69.9——这都印证了 D-Diff unified model 之前 understanding 性能差的根因

### 6.2 Text-to-Image Generation（Table 3, 4, 5, 6, 7, 8）

**GenEval (Table 3)**:
- LLaDA2.0-Uni 0.89，超过所有 unified model
- Position 子项 0.90，是**所有** model（包括 specialist）中最高
- vs. Lumina-DiMOO 0.88 几乎持平，但 Lumina-DiMOO understanding 性能崩了

**DPG (Table 4)**:
- LLaDA2.0-Uni 87.76，unified model 中 SOTA
- vs. Z-Image-Turbo 84.86——这很 impressive 因为 Z-Image-Turbo 是 specialist

**UniGenBench (Table 6)**:
- LLaDA2.0-Uni 79.63，超过所有 unified model
- Logic 63.99（unified 第一），Layout 90.30（unified 第一）

**CVTG-2K (Table 7)**:
- LLaDA2.0-Uni 0.765，unified model 第一
- **稳定性**是关键优势：BAGEL/Lumina-DiMOO/InternVL-U 在 region 数量增加时分数掉得很厉害，LLaDA2.0-Uni 几乎不掉

**WISE-Bench (Table 8)**:
- LLaDA2.0-Uni 0.68，超过所有 unified
- **+w/ thinking: 0.78**，比不带 thinking 的版本提升 14.7%
- Biology 0.79, Physics 0.87, Space 0.82——这些需要 world knowledge 的任务上特别强

wise-bench 的提升验证了一个非常 exciting 的方向：unified model + CoT 在 image generation 上的提升远大于 text-only reasoning 单独的提升。这意味着 visual generation 可以从 reasoning 受益，反过来也成立。

### 6.3 Image Editing（Table 9, 10, 11）

**ImgEdit (Table 9)**:
- LLaDA2.0-Uni 3.92，unified model 中第一
- Adjust 4.16（unified 第一）, Hybrid 3.97（unified 第一）
- vs. specialist Qwen-Image-Edit 4.35 还有 gap

**GEdit-Bench (Table 10)**:
- EN: 6.61, CN: 6.66
- Perceptual Quality 子项 7.52——这个高意味着编辑后 image fidelity 保留好，没有引入 artifact

**MICo-Bench (Table 11)**:
- LLaDA2.0-Uni 47.1，**SOTA**（所有 model）
- vs. Qwen-Image-Edit 35.9, BAGEL 34.4, OmniGen2 33.8, Lumina-DiMOO 23.3
- HOI (Human-Object Interaction) 46.0，比第二名高一倍多——这个 gap 非常大

MICo-Bench 上的优势我觉得揭示了一个重要的 insight：multi-reference editing 需要模型**同时**理解多张图（每张图各自的语义）+ 生成融合图（图像中组合它们）。dLLM 的 bidirectional context 在这里特别有优势——它能 attend 到所有 reference image 的所有 token 并行做融合推理。AR-based 模型必须 sequential 处理，融合效率低。

Lumina-DiMOO 同架构只有 23.3 这个事实说明，光有 dLLM 不够，**SigLIP-VQ 提供的 semantic token** 才是关键——它能 faithful 保留 reference image 的语义，而 reconstructive VQ 会丢失细节。

### 6.4 Interleaved Generation & Reasoning

Paper 在这里贡献了一个新 benchmark——InterGen（150 samples，3 categories: Story Telling, Explanation, Event Forecasting）。

Table 12 vs Emu3.5:
- Story Telling: 6.42/7.02 vs 6.28/6.83
- Explanation: 6.22/6.35 vs 6.19/6.48
- Event Forecasting: 5.19/5.94 vs 5.08/5.75

数据集太小（150 samples），统计意义有限，但展示了 capability。Figure 8 里的国际象棋例子（paper 给的 Figure 3）很有说服力——模型逐个分析 4 个 candidate move（Ke3, Ke5, Kc4, Kd5），每个都做了 visual reasoning + 语言学评估，最后输出 "Answer: B"。这种 chain-of-thought visual reasoning 是 AR-only unified model 难做的——AR 模型要在 text reasoning 完成 commit 之后才能 generate 新 image，不能像 diffusion 那样在 reasoning 中插入 image generation。

---

## 7. 训练 Pipeline

**Stage 0: Vision-Language Alignment (100B tokens, 8k context)**
- 数据：image-caption + text
- 目标：align visual 和 linguistic representation
- Masking 策略：generation task 只 mask image token；understanding task 只 mask text token——这是个 important detail，让两种 task 的 mask 不互相干扰
- Resolution：generation 256→512，understanding 800×800 arbitrary resolution

**Stage 1: Multi-task Pre-training (210B tokens)**
- 加入 OCR, Grounding, Counting, Video, Image Editing, Interleaved Generation
- Resolution 升级：generation 512, understanding 800

**Stage 2: SFT (80B tokens)**
- 高质量 instruction tuning
- 8k → 16k context length

**Diffusion Decoder 训练**（独立 3 stage）：
- Warm-up：冻结 semantic processor，align 跨模态
- Multi-domain generalization：unfreeze 所有参数
- High-fidelity refinement：在高质量数据上做 aesthetic refinement

**Data Preparation 几个亮点**：
- 140M images for generation，三阶段 filter（metadata, aesthetics, quality）
- 60M SFT samples, text-only:multimodal = 1:5
- 6M refined Koala36M clips for interleaved（5 秒采一帧，2-6 frame per sequence）
- 8M reasoning-augmented samples（Flux-6M, Zebra-CoT, Weave）

---

## 8. 工程细节

### 8.1 Image Token Pre-extraction

训练前把整个 dataset 跑过 frozen VQ tokenizer，把 token index 存到 disk。训练时直接 load index 而非 image——避免每次 forward pass 都跑 ViT encoder。

这是个**很 cost-effective 但很 memory-heavy** 的 trick：存 140M images × 1024 token × 4 bytes ≈ 575 GB。但训练吞吐提升是巨大的。

### 8.2 Data Packing（Figure 5）

Multimodal 训练里 sample 长度差异极大（短 text 几十 token vs image generation 几千 token）。传统 padding 浪费严重。Data packing 把多个 short sample 拼成 fixed-length sequence。

需要注意：在 dLLM 里 packing 比 AR 里 tricky，因为 BDLM 的 block-wise attention 需要清楚知道 sample 边界（不能让 sample A 的 token attend 到 sample B 的 token）。Paper 没细讲，但 dFactory 框架（InclusionAI 2025）应该处理了这个。

### 8.3 dFactory + VeOmni

- dFactory 是 InclusionAI 内部针对 dLLM 优化的训练引擎
- 基于 VeOmni（Ma 2025, https://arxiv.org/abs/2508.02317）分布式 ecosystem
- 支持 flexible parallelization strategy（推测：TP+PP+DP+Expert parallel 混合）

---

## 9. 我的几个直觉判断

### 9.1 SigLIP-VQ 是这个工作的灵魂

读完全文我最 strong 的判断是：SigLIP-VQ 这个选择是 LLaDA2.0-Uni 与所有前作的真正分水岭。

之前 unified D-Diff model 卡在 understanding 性能差，根因是 VQ-VAE 的 codebook 缺语义。SigLIP-VQ 把 codebook 直接训成语义对齐的，相当于把 "理解用 ViT + 生成用 VAE" 这个 decoupled 设计 collapse 回 single encoder，但牺牲了直接 reconstruction 能力。再用 diffusion decoder 补上 reconstruction 这一环。

这是一个**很有原则的 trade-off**：让 tokenizer 专注一件事（semantic），把 reconstruction 留给专门的 decoder。前作 SigLIP-VQ（X-Omni）只是 understanding，LLaDA2.0-Uni 把它 extend 到 unified setting 并证明 work。

### 9.2 Block-wise Attention 是为 SigLIP-VQ 量身定做

完全 bidirectional 不行，完全 AR 又丧失 diffusion 优势。Block-wise 是中间道路。但 paper 暗示了一个细节：SigLIP-VQ token 因为是跟 Qwen2.5 对齐的，**本身就有 autoregressive bias**。如果用 full bidirectional，这个 bias 被破坏。Block-wise 让 block 间保留 AR 结构，相当于 "尊重 SigLIP-VQ 的 inductive bias"。

这给我一个推测：如果换成纯 ViT feature（不带 AR bias 的 encoder），block-wise 可能就比 full bidirectional 没那么明显优势。这个 ablation paper 没做。

### 9.3 SPRINT 是 dLLM 真正的工程胜利

DLLM 最大的实际问题是 inference 慢——BDLM 的 $B \times T$ forward passes 比 AR 的 $L$ 次还多（虽然单次 forward 并行）。SPRINT 把这两条 axis 分别优化，并且是 training-free 的，这个工程价值很高。

但 OCRBench 掉 2.3 分说明 SPRINT 有 boundary case：精确字符任务对 aggressive unmasking 敏感。这可能可以通过 task-specific τ 来 mitigate。

### 9.4 WISE-Bench + Thinking 的 14.7% 提升

这是 paper 里**最 under-discussed** 的结果。带 thinking 的 WISE-Bench 从 0.68 提升到 0.78，10% absolute。这个数字暗示：

> Unified model + CoT 是 image generation 的下一个 frontier。

AR-based unified model 也能做 CoT，但它们 reasoning 是 text-only 模式，reasoning 完再 generate image。Diffusion-based unified 在 principle 上可以做 "边 reason 边 generate"，让 image token 和 reasoning token 在 sequence 里 interleaved 出现。这相当于把 CoT 从"线性文本推理"扩展到"多模态推理"。

Figure 8 的国际象棋例子就是这种 interleaved reasoning 的 instance——模型边说 "考虑 Kc4 这个 move..." 边输出 image 表示"如果走 Kc4 棋盘会变成这样"。这个能力只有 unified diffusion model 才有。

### 9.5 我觉得不足的地方

1. **MMMU / WeMath 上落后 BAGEL**：BAGEL 在 reasoning benchmark 上明显强（WeMath 45.8 vs 29.3，gap 巨大）。这是 AR backbone 在 reasoning 上的先天优势，diffusion 短期内难追
2. **Fine-grained visual detail 弱**：paper 自己在 Conclusion 承认 "SigLIP-VQ struggles to preserve fine-grained image details"。这点在 image editing 的高 precision 任务上会成为 bottleneck
3. **InterGen benchmark 太小**：150 samples，统计意义有限。需要更大 benchmark 验证 interleaved generation 能力
4. **SPRINT 在 OCR 上退化**：说明 confidence-based unmasking 对精确字符任务不友好，可能需要 task-aware τ schedule
5. **缺 scaling law 分析**：16B 是个特定 size，paper 没探讨 dLLM unified model 的 scaling law。dLLM 的 scaling behavior 跟 AR 可能很不一样，这是个 interesting 开放问题
6. **没有 RL 探索**：Conclusion 提到 "we have begun exploring RL"，但 paper 没数据。dLLM 的 RL 训练（reward 设计、gradient flow through mask）是 interesting 方向
7. **Speed 数据不够细**：Table 13 给的 TPS 是整体吞吐，但没拆解 SPRINT 两个 axis 各贡献多少。Sparse Prefix Retention 和 Non-uniform Unmasking 的 ablation 我很想看

---

## 10. 相关工作的联想（可以延伸读的）

读完这篇 paper 我觉得有几条 line 值得继续 follow：

1. **LLaDA main paper**（Bie 2025, https://arxiv.org/abs/2512.15745）：16B dLLM backbone，理解 LLaDA2.0-Uni 必须先理解 LLaDA
2. **BDLM**（Arriola 2025, ICLR）：Block Diffusion 是这个 work 的 objective 基础
3. **LLaDA**（Nie 2025, https://arxiv.org/abs/2502.09992）：原始 LLaDA，8B 参数，验证 dLLM 可行
4. **MMaDA**（Yang 2025, https://arxiv.org/abs/2505.15809）：前作 unified D-Diff，性能差很多，对比能看出 LLaDA2.0-Uni 的改进 magnitude
5. **Lumina-DiMOO**（Xin 2025a, https://arxiv.org/abs/2510.06308）：另一个 unified D-Diff baseline
6. **BAGEL**（Deng 2025, https://arxiv.org/abs/2505.14683）：AR+Diff hybrid，reasoning 上仍然领先
7. **Z-Image**（Cai 2025, https://arxiv.org/abs/2511.22699）：diffusion decoder 的 base
8. **Flow Matching**（Lipman 2022, https://arxiv.org/abs/2210.02747）：decoder 训练的基础
9. **Consistency Models / LCM**（Lu & Song 2024, https://arxiv.org/abs/2410.11081）：few-step distillation 的理论基础
10. **X-Omni**（Geng 2025, https://arxiv.org/abs/2507.22058）：SigLIP-VQ 架构来源
11. **DeepSeek-V3**（Liu 2024a, https://arxiv.org/abs/2412.19437）：auxiliary-loss-free load balancing 来源
12. **SGLang**（Zheng 2024, https://arxiv.org/abs/2312.07104）：他们说要集成 SGLang 进一步加速

---

## 11. 一个 takeaway

读完整篇 paper 我对 LLaDA2.0-Uni 的判断是：

> 它不是 "another unified multimodal model"，而是 "the first unified dLLM that doesn't suck at understanding"。

之前 D-Diff unified model 一直在 "理解差" 的诅咒下挣扎。LLaDA2.0-Uni 通过 SigLIP-VQ + block-wise attention + 独立 diffusion decoder 这个 triple combo，第一次让 D-Diff unified 在 understanding 上接近 specialist VLM 水平，同时在 generation 上保持 SOTA。

从更 long-term 视角看，这个 work 验证了一个 thesis：**dLLM 的 unified multimodal 设定可以 work，只要 token representation 选对**。Token representation 是 foundation——它决定了下游所有任务的天花板。VQ-VAE 让 D-Diff 卡了半年，SigLIP-VQ 解开了这个结。

接下来的 frontier 我认为是：
- **Interleaved reasoning + generation**：WISE-Bench +thinking 的 14.7% 提升暗示这条路有 promise
- **dLLM-specific RL**：reward model 怎么对 mask diffusion 设计？
- **Scaling law for unified dLLM**：跟 AR 的 scaling law 形状很可能不同
- **Better tokenizer**：保留 fine-grained detail 的同时保 semantic——可能 hierarchical VQ 或者 multi-codebook 是答案

我非常期待 Andrej 你对这个 work 的判断。我觉得它 represent 了一个很 healthy trend——把 LLM diffusion 化的研究开始认真撞墙到 multimodal unified 这种 AR 已经先发优势的领域，开始有 competitive 的结果。

---

参考链接：
- Paper GitHub: https://github.com/inclusionAI/LLaDA2.0-Uni  
- HuggingFace Model: https://huggingface.co/inclusionAI/LLaDA2.0-Uni  
- LLaDA2.0 main paper: https://arxiv.org/abs/2512.15745  
- LLaDA original: https://arxiv.org/abs/2502.09992  
- BDLM (Block Diffusion): https://openreview.net/forum?id=LEYc8MB4tP  
- MMaDA: https://arxiv.org/abs/2505.15809  
- Lumina-DiMOO: https://arxiv.org/abs/2510.06308  
- BAGEL: https://arxiv.org/abs/2505.14683  
- Z-Image: https://arxiv.org/abs/2511.22699  
- SigLIP2: https://arxiv.org/abs/2502.14786  
- X-Omni: https://arxiv.org/abs/2507.22058  
- Flow Matching: https://arxiv.org/abs/2210.02747  
- Consistency Models (sCM): https://arxiv.org/abs/2410.11081  
- DeepSeek-V3: https://arxiv.org/abs/2412.19437  
- SGLang: https://arxiv.org/abs/2312.07104  
- VeOmni: https://arxiv.org/abs/2508.02317  
- Janus: https://arxiv.org/abs/2410.13848  
- Emu3: https://arxiv.org/abs/2409.18869  
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
