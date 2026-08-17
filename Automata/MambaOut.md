---
source_pdf: MambaOut.pdf
paper_sha256: 7396537296914e4069f5e61b1b682752f3abc6f3e0c045c23714a67af1e6dfee
processed_at: '2026-08-05T16:14:07-07:00'
target_folder: Automata
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MambaOut 用人话讲：把 Mamba 的"灵魂"抽掉，结果更强了

Andrej，这篇 paper 的故事其实特别简单，我给你讲个比喻你就懂了。

假设你在搬家。Attention 这个机制就像你把所有东西都摊在客厅地上，要用什么直接找，东西绝对不会丢，但是房子越大，找东西越累。SSM（Mamba 的核心）就像你把所有东西压缩进一个固定大小的行李箱，找东西飞快，但是行李箱就那么大，东西多了必然丢一些。

所以 attention 擅长短序列（客厅不大，摊得开），SSM 擅长长序列（客厅爆了，只能用行李箱）。这道理没毛病。

然后问题来了：**vision 社区看到 Mamba 在 NLP 上火了，一窝蜂往图像上搬，结果发现打不过甚至打不过普通的 ConvNet**。这篇 paper 的作者就问了个特别尖锐的问题：你们搬之前有没有想过，vision 任务到底需不需要 Mamba？

Paper link: https://arxiv.org/abs/2405.14174

---

## 1. Mamba 在 NLP 上为什么有道理

Mamba 的核心是 selective SSM，本质就是个现代化的 RNN。它的递归公式长这样：

$$h_t = \overline{A} h_{t-1} + \overline{B} x_t$$

变量含义：
- $t$：timestep，就是序列里第 $t$ 个位置
- $h_t \in \mathbb{R}^N$：hidden state，$N$ 是 state dimension（Mamba 默认 16），这是那个"行李箱"
- $h_{t-1}$：上一步的 hidden state
- $x_t$：当前输入 token
- $\overline{A}$：state transition matrix，控制"上一秒的记忆保留多少"
- $\overline{B}$：input projection，控制"新输入记多少进去"

关键点：**$h_t$ 的维度 $N$ 是固定的**。不管你输入 100 个 token 还是 100 万个 token，行李箱大小不变。所以：
- 记忆是 **lossy** 的（东西塞多了必然挤丢）
- 每步更新的 cost 是 **$O(N)$ 常数**（不管历史多长，就一个矩阵向量乘）

对比 attention：attention 把所有历史 token 的 key 和 value 都存着，**lossless**，但是 memory size 随序列线性涨，检索 cost 随序列线性涨（实际上 attention 是 quadratic，因为每个 token 都要扫一遍所有历史）。

**所以 Mamba 在 NLP 上有道理**，因为：
1. LLM 的 context window 动不动 8K、32K、128K，attention 的 quadratic cost 真的爆了 → **long-sequence 特性满足**
2. LLM 做 next-token prediction，每个 token 只能看前面的 token（你不能偷看未来的词）→ **autoregressive 特性满足**

两个条件都满足，Mamba 的"压缩记忆换效率"才划算。

---

## 2. Vision 的问题：两个条件一个都不满足（对 classification）

作者做了个特别干净的分析。先看"long-sequence"这个条件到底满不满足。

### 2.1 一个超简单的判据：$L$ vs $6D$

Transformer 一个 block 的 FLOPs 可以拆成两项：

$$\text{FLOPs} = \underbrace{24D^2L}_{\text{linear in }L} + \underbrace{4DL^2}_{\text{quadratic in }L}$$

变量：
- $D$：channel dimension（embedding 维度）
- $L$：sequence length（token 数）
- $24D^2L$：所有 projection 和 MLP 的 cost，跟 $L$ 成正比
- $4DL^2$：attention score 计算（$QK^T$ 是 $2DL^2$）和 attention 乘 value（另一个 $2DL^2$），跟 $L^2$ 成正比

两者的比值：

$$r_L = \frac{4DL^2}{24D^2L} = \frac{L}{6D}$$

**直觉**：当 $L > 6D$，quadratic 项主导，attention 成为瓶颈，这时候才算"long-sequence"，SSM 才有用武之地。当 $L < 6D$，linear 项主导，attention 根本不是瓶颈，你用 SSM 纯属给自己找麻烦。

### 2.2 代入实际数字

| Model | $D$ | Threshold $\tau = 6D$ |
|---|---|---|
| ViT-Small | 384 | 2304 |
| ViT-Base | 768 | 4608 |

| Task | Image size | Patch size | Token 数 $L$ | Long-seq? |
|---|---|---|---|---|
| ImageNet classification | $224 \times 224$ | $16 \times 16$ | $14 \times 14 = 196$ | **196 远小于 2304，NO** |
| COCO detection | $800 \times 1280$ | $16 \times 16$ | ~4000 | **4000 > 2304，YES** |
| ADE20K segmentation | $512 \times 2048$ | $16 \times 16$ | ~4096 | **YES** |

**看到没？ImageNet classification 只有 196 个 token，连 ViT-S 的 threshold 都差一个数量级**。attention 在这种序列长度上跑得飞快，SSM 的 efficiency 优势完全用不上。

### 2.3 Autoregressive 这个条件

这个更致命。SSM 的递归公式 $h_t = \overline{A}h_{t-1} + \overline{B}x_t$ 决定了 $h_t$ 只能见到 $x_1, x_2, \ldots, x_t$，**天然是 causal 的**（只能看过去，不能看未来）。

这对 next-token prediction 是 perfect 的——你本来就不该偷看未来的词。

但是 **image classification 是 understanding task**，模型可以一次性看到整张图。你强制让它"从左上到右下逐个 patch 看，每个 patch 只能看前面"，这是在给模型戴眼罩。

作者做了个超直观的实验（Figure 3b）：拿 ViT，给它加 causal mask（让每个 token 只能 attend 前面的 token），结果 ImageNet accuracy 直接掉。**这直接证明 causal constraint 对 understanding task 是有害的**。

Visual Mamba 怎么解决这个？用 bidirectional scan——一个 branch 从左上扫到右下，另一个从右下扫到左上，两个 branch 的输出 merge。但是**每个 branch 内部仍然是 causal 的**！左上角的 token 在前向 branch 里永远看不到右下角的内容，只能靠后向 branch 补。这相当于把一个 fully-visible 的 attention 拆成两个"半瞎"的 RNN 再拼起来，信息融合质量肯定不如原生的 attention。

**所以对 ImageNet classification**：
- Long-sequence：❌（196 token 太短）
- Autoregressive：❌（understanding task 不该 causal）

两个都不满足，Mamba 在这就是 over-engineering。

---

## 3. MambaOut 的做法：把 SSM 抽掉，看会怎样

### 3.1 Mamba block 的结构

Mamba block 其实是 **Gated CNN + SSM** 的组合。看 Figure 1a，Mamba block 就是先做 Gated CNN 的那套（conv + gating），然后再接一个 SSM。

用公式说，meta-architecture 是：

$$Y = (\text{TokenMixer}(X'W_1) \odot \sigma(X'W_2))W_3 + X$$

变量：
- $X \in \mathbb{R}^{N \times D}$：input
- $X'$：normalized input
- $W_1, W_2 \in \mathbb{R}^{D \times rD}$：两个 projection，$r$ 是 expansion ratio
- $W_3 \in \mathbb{R}^{rD \times D}$：output projection
- $\sigma$：activation function（GELU）
- $\odot$：element-wise multiply（gating 操作）
- $+X$：residual

两条 branch：
- $\text{TokenMixer}(X'W_1)$：做 token mixing 的 branch
- $\sigma(X'W_2)$：gate branch，控制 token mixer 输出多少

Gated CNN 和 Mamba 的区别就在 TokenMixer：

$$\text{TokenMixer}_{\text{GatedCNN}}(Z) = \text{Conv}(Z)$$
$$\text{TokenMixer}_{\text{Mamba}}(Z) = \text{SSM}(\sigma(\text{Conv}(Z)))$$

**一目了然**：Mamba 就是 Gated CNN 后面多接了个 SSM。作者说：那我把 SSM 拿掉，用 Gated CNN 代替 Mamba，叫 MambaOut，看效果怎样。

### 3.2 Gated CNN block 的具体实现

```python
class GatedCNNBlock(nn.Module):
    def __init__(self, dim, expension_ratio=8/3, kernel_size=7, 
                 conv_ratio=1.0, ...):
        self.norm = norm_layer(dim)
        hidden = int(expension_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()  # GELU
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels, 
                              kernel_size=7, padding=3, 
                              groups=conv_channels)  # depthwise conv
        self.fc2 = nn.Linear(hidden, dim)
    
    def forward(self, x):
        shortcut = x  # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        # g: gate, i: identity (不 mix), c: conv input
        c = c.permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]
        c = self.conv(c)            # depthwise 7x7 conv
        c = c.permute(0, 2, 3, 1)  # back to [B,H,W,C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        return x + shortcut
```

几个关键设计：
1. **$7 \times 7$ depthwise convolution**：跟 ConvNeXt 一致，local token mixing
2. **Partial conv**：只在一部分 channel 上做 conv（`conv_ratio` 控制比例），其余 channel 直接 pass through（identity branch $i$）。这是跟 InceptionNeXt 学的，为了减少 FLOPs 同时保持 receptive field
3. **Gating**：`act(g) * cat(i, c)`，gate branch $g$ 过 GELU 后控制 identity + conv 的输出
4. **Expansion ratio $8/3$**：跟 ConvNeXt 一样的 ratio

### 3.3 整体架构

MambaOut 用 ResNet-style 的 4-stage hierarchical 结构（Figure 4）：

| Size | Blocks (S1-S4) | Channels (S1-S4) | Params | MACs |
|---|---|---|---|---|
| Femto | (3,3,9,3) | (48,96,192,288) | 7.3M | 1.2G |
| Tiny | (3,3,9,3) | (96,192,384,576) | 26.5M | 4.5G |
| Small | (3,4,27,3) | (96,192,384,576) | 48.5M | 9.0G |
| Base | (3,4,27,3) | (128,256,512,768) | 84.8M | 15.8G |

Stem 是 3 个 stride-2 的 $3 \times 3$ conv，总共 downsample 8×。然后 4 个 stage，每个 stage 之间再 downsample 2×。Stage 3 堆很多 block（27 个），因为 high resolution + moderate channel 的 stage 最需要 depth。

---

## 4. 实验结果：hypothesis 完美验证

### 4.1 ImageNet classification：MambaOut 打所有 visual Mamba

| Model | Token Mixing | Params | MACs | Acc@224 |
|---|---|---|---|---|
| Vim-S | Conv+SSM | 26M | 5.1G | 80.5 |
| VMamba-T | Conv+SSM | 22M | 5.6G | 82.2 |
| LocalVMamba-S | Conv+SSM | 50M | 11.4G | 83.7 |
| VMambaV9-S | Conv+SSM | 50M | 8.7G | 83.6 |
| Vim-B | Conv+SSM | 90M | 18.0G | 83.7 |
| **MambaOut-Tiny** | **Conv only** | **27M** | **4.5G** | **82.7** |
| **MambaOut-Small** | **Conv only** | **48M** | **9.0G** | **84.1** |
| **MambaOut-Base** | **Conv only** | **85M** | **15.8G** | **84.2** |

最 punchy 的对比：**MambaOut-Small 84.1% vs LocalVMamba-S 83.7%**。MambaOut 用更少的 MACs（9.0G vs 11.4G）和差不多的 params（48M vs 50M），反而高了 0.4%。

**你去掉 SSM，模型反而变强了**。这就是 Occam's razor——最简单的解释胜出。

而且作者还加了个"嘲讽"对比：CAFormer-M36 用的是 7 年前发明的 separable conv + vanilla attention，结果 85.2%，比所有 visual Mamba 都高 1%+。等于说 visual Mamba 连老古董都打不过。

### 4.2 COCO detection：SSM 确实有用

| Backbone | Token Mixing | APb | APm |
|---|---|---|---|
| VMamba-T | Conv+SSM | 46.5 | 42.1 |
| LocalVMamba-T | Conv+SSM | 46.7 | 42.2 |
| **MambaOut-Tiny** | **Conv only** | **45.1** | **41.0** |
| VMambaV9-T | Conv+SSM | 47.4 | 42.7 |
| TransNeXt-Tiny (SOTA hybrid) | Conv+Attn | 49.9 | 44.6 |

这里 MambaOut-Tiny 比 VMamba-T 低 1.4 APb。**在 detection 这种 long-sequence task 上，SSM 确实有贡献**——因为 token 数 ~4K，attention 的 quadratic cost 开始成为问题，SSM 的 global receptive field 有优势。

但是 visual Mamba 离 SOTA hybrid（TransNeXt）还差 3+ AP，说明 SSM 的 causal bias 仍然是个负债。

### 4.3 ADE20K segmentation：同样的故事

| Backbone | Token Mixing | mIoU (SS) | mIoU (MS) |
|---|---|---|---|
| VMamba-T | Conv+SSM | 47.3 | 48.3 |
| LocalVMamba-T | Conv+SSM | 47.9 | 49.1 |
| **MambaOut-Tiny** | **Conv only** | **47.4** | **48.6** |
| TransNeXt-Tiny | Conv+Attn | 51.1 | 51.2 |

MambaOut-Tiny 介于 VMamba-T 和 LocalVMamba-T 之间，比 LocalVMamba-T 低 0.5 mIoU。**SSM 在 long-seq task 上有 marginal 优势，但离 SOTA 还差得远**。

---

## 5. 更深层的 intuition

### 5.1 为什么 SSM 在 vision 上"水土不服"

我觉得根本原因是 **SSM 的设计哲学跟 spatial data 不匹配**。

SSM 的 selectivity 机制（$\Delta, A, B, C$ 是 input-dependent 的）在 language 上很自然：模型根据 token 内容决定"这个词重不重要，要不要记住"。比如遇到句号，$\Delta$ 可能变大，让 hidden state "reset"；遇到关键词，$\Delta$ 变小，"hard remember"。

但在 vision 上，一个 patch 的"重要性"取决于 **spatial context**——这个 patch 是物体中心还是背景边缘，要看周围才知道。而 SSM 的 causal scan 让 selectivity 变成"基于扫描前缀的判断"，这跟 spatial reasoning 的需求根本不对路。

你想想，图像左上角的一个 patch，在 causal scan 里它只能看到自己（它是第一个），怎么判断自己重不重要？这就是 causal bias 对 spatial data 的根本伤害。

### 5.2 Bidirectional scan 为什么不够

Visual Mamba 用双向 scan（前向 + 后向）来缓解 causal 问题。但这只是"两个半盲的 RNN 拼起来"：

- 前向 branch：左上角 token 看到 0 个历史，右下角 token 看到所有历史
- 后向 branch：右下角 token 看到 0 个历史，左上角 token 看到所有历史
- Merge 后：每个 token 都能"间接"看到全局，但是经过了一层有损压缩

对比 attention：每个 token 直接 attend 所有其他 token，信息无损，而且 attention score 是 content-based 的（$QK^T$），可以 dynamically 决定看哪里。

**所以 bidirectional SSM 是个"打了折扣的 fully-visible"**，而 attention 是原生的 fully-visible。在 understanding task 上，这个折扣体现为 performance gap。

### 5.3 Gated CNN 为什么够用

MambaOut 的成功说明：**对 vision understanding，local conv + gating 就够了**。

$7 \times 7$ depthwise conv 提供 local receptive field，gating 提供 input-dependent 的 modulation（类似 SE block 或 attention 的简化版）。在 4-stage hierarchical 结构里，经过多次 downsample 和堆叠 block，effective receptive field 已经覆盖大部分图像。而且 conv 是 **fully-visible 的**（每个输出位置看到周围 $7 \times 7$ 的输入，没有 causal 限制），这跟 understanding task 的需求天然匹配。

所以 MambaOut 的成功不是"conv 比 SSM 强"，而是"conv 的 inductive bias 跟 vision understanding task 更匹配，而 SSM 的 causal bias 是个负债"。

### 5.4 跟 LLM 的对比

作者在 conclusion 里暗示要扩展到 LLM/LMM。在 LLM 上：
- Long-sequence：✅（context window 几万到几十万 token）
- Autoregressive：✅（next-token prediction 天然 causal）

两个条件都满足，所以 Mamba/RWKV 在 LLM 上有真实价值。Jamba（AI21 Labs 的 Transformer-Mamba hybrid）就是这条路线。

这也解释了为什么 Mamba 在 NLP 上 work 但搬到 vision 就拉胯——**task characteristic 变了，SSM 的优势用不上，劣势暴露了**。

### 5.5 Gated CNN 的历史

论文追溯到 Dauphin et al. 2017 的 Gated CNN（ref [18]），这是 pre-Transformer 时代的 language model 架构。当时 Gated CNN 想用 conv 替代 RNN 做 language modeling，效果还行但被 Transformer 秒了。

Mamba 其实是 **"Gated CNN + SSM" 的组合**——把 Gated CNN 的 conv 换成 SSM 来获得 long-range + autoregressive 能力。所以 MambaOut 本质上是**退回 Gated CNN**，砍掉 Mamba 加的那个 SSM 层。

这个 ablation 设计非常干净：control 了所有其他变量（gating、partial conv、meta-arch），只 isolate SSM 的贡献。

### 5.6 跟 ConvNeXt 的关系

MambaOut 的 Gated CNN block 跟 ConvNeXt block 非常像（$7 \times 7$ depthwise conv + MLP + residual），区别在于：
- MambaOut 用 gating（$\text{act}(g) \cdot \text{cat}(i, c)$），ConvNeXt 没有 gating
- MambaOut 用 partial conv，ConvNeXt 是 full conv

实际上 MambaOut-Tiny（82.7%）跟 ConvNeXt-T（82.1%）很接近，gating 带来一点提升。这说明 **visual Mamba 的 conv 部分已经足够强，SSM 是个"额外负担"**。

### 5.7 对 visual Mamba 社区的建议

根据 paper 的 logic，未来的 visual Mamba 研究应该：

1. **不要在 ImageNet classification 上死磕** —— SSM 在这是负债，打不过简单的 ConvNet
2. **在 high-resolution dense prediction 上发力** —— detection、segmentation、medical imaging、remote sensing 这些 token 数真的大的场景
3. **解决 causal bias** —— 双向 scan 不够，可能需要更聪明的 non-causal SSM 变体
4. **跟 attention hybrid** —— 像 Jamba 那样，在关键层用 attention 保持 fully-visible，其他层用 SSM 省 cost

---

## 6. 一句话总结

**Mamba 是为"长序列 + 自回归"设计的工具。Image classification 两个条件都不满足，所以 Mamba 在这是 over-engineering。Detection/segmentation 满足 long-sequence 但不满足 autoregressive，所以 SSM 有 marginal 价值但还没发挥出来。**

作者用最简单的实验（把 SSM 抽掉）证明了这个 argument，而且结果干净利落。这种"conceptual clarity + clean ablation"的风格，正是好的 research 该有的样子。

---

## References

- MambaOut paper: https://arxiv.org/abs/2405.14174
- MambaOut code: https://github.com/yuweihao/MambaOut
- Mamba original: https://arxiv.org/abs/2312.00752
- S4 (structured SSM): https://arxiv.org/abs/2111.00396
- Vision Mamba (Vim): https://arxiv.org/abs/2401.09417
- VMamba: https://arxiv.org/abs/2401.10166
- LocalMamba: https://arxiv.org/abs/2403.09338
- PlainMamba: https://arxiv.org/abs/2403.17695
- EfficientVMamba: https://arxiv.org/abs/2403.09977
- Gated CNN (Dauphin 2017): https://arxiv.org/abs/1612.08083
- ConvNeXt: https://arxiv.org/abs/2201.03545
- MetaFormer: https://arxiv.org/abs/2111.11418
- InceptionNeXt: https://arxiv.org/abs/2303.16900
- Jamba (Transformer-Mamba hybrid LLM): https://arxiv.org/abs/2403.19887
- RWKV: https://arxiv.org/abs/2305.13048
- Mamba-2: https://arxiv.org/abs/2405.21060
- Tri Dao's blog: https://tridao.me/
- Albert Gu's Mamba repo: https://github.com/state-spaces/mamba
- Kobe Bryant farewell (paper 致敬): https://www.youtube.com/watch?v=JiZkGCowAZc

---

# MambaOut: Do We Really Need Mamba for Vision? 深度解析

Andrej，这篇 paper 写得相当 sharp，核心 argument 干脆利落：**Mamba 的 SSM token mixer 在 vision 的 understanding tasks 上根本没必要，至少在 image classification 上是 over-engineering**。作者 Weihao Yu（也是 MetaFormer、InceptionNeXt 的作者）用 Occam's razor 的思路，把 Mamba block 里的 SSM 抽掉，剩下 Gated CNN，结果在 ImageNet 上把所有 visual Mamba 都打了。下面我把技术细节拆开讲，帮你 build intuition。

Paper link: https://arxiv.org/abs/2405.14174
Code: https://github.com/yuweihao/MambaOut

---

## 1. 核心论点的逻辑骨架

作者把 Mamba 适合的任务拆成两个 characteristic：

- **Characteristic 1: long-sequence** —— 因为 SSM 的 hidden state 是 fixed-size 的 lossy memory，只有当序列足够长、attention 的 quadratic complexity 真的成为瓶颈时，SSM 的 constant-time memory merge 才有意义。
- **Characteristic 2: autoregressive (causal)** —— 因为 SSM 本质是 recurrent，$h_t$ 只能见到 $h_{t-1}$ 和 $x_t$，天然 causal。这个 inductive bias 对 generation 任务友好，对 understanding 任务是有害的 constraint。

然后作者去 vision tasks 上对号入座：

| Task | Long-seq? | Autoregressive? | SSM 必要性 |
|---|---|---|---|
| ImageNet classification | ❌ (196 tokens << 2304) | ❌ | 不必要 (Hypothesis 1) |
| COCO detection | ✅ (~4K tokens) | ❌ | 值得探索 (Hypothesis 2) |
| ADE20K segmentation | ✅ (~4K tokens) | ❌ | 值得探索 (Hypothesis 2) |

实验结果与 hypothesis 完美对齐：MambaOut（去 SSM）在 classification 上全面胜出 visual Mamba，但在 detection/segmentation 上落后于 SOTA visual Mamba。

---

## 2. SSM 的数学本质：为什么它是 RNN

这是论文 Equation 1-3，是整个 argument 的基石。Mamba 用的是 **selective SSM**（S4 → S6 的演化），参数 $(\Delta, A, B, C)$ 是 input-dependent 的（"selective" 的来源）。

### 2.1 Discretization (Eq. 1)

$$\overline{A} = \exp(\Delta A), \quad \overline{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$$

变量含义：
- $A \in \mathbb{R}^{N \times N}$：continuous-time state transition matrix，控制 hidden state 的衰减/保留。在 Mamba 里初始化为 HiPPO 矩阵或对角负数（让记忆指数衰减）。
- $B \in \mathbb{R}^{N \times 1}$：continuous-time input matrix，把 scalar input $x(t)$ 投影到 state space。
- $\Delta \in \mathbb{R}$：step size / discretization delta。**这是 Mamba "selectivity" 的关键**——$\Delta$ 是由 input $x_t$ 通过一个小 MLP 算出来的，所以不同 token 有不同的"采样步长"，相当于选择性记忆。
- $\overline{A}, \overline{B}$：discretized 版本，用于递归更新。
- $\exp(\cdot)$：matrix exponential，$\exp(M) = I + M + M^2/2! + \cdots$。当 $\Delta$ 很小时，$\overline{A} \approx I + \Delta A$，$\overline{B} \approx \Delta B$（这是 zero-order hold 的近似）。
- $I$：identity matrix，维度同 $A$。

这个 discretization 是从 continuous-time ODE $\dot{h} = Ah + Bx$ 来的，用 ZOH（zero-order hold）离散化。**直觉**：连续动力系统被采样成离散递归，$\Delta$ 大表示"跳过更多信息"，$\Delta$ 小表示"精细记忆"。

### 2.2 Recurrence (Eq. 2-3)

$$h_t = \overline{A} h_{t-1} + \overline{B} x_t$$
$$y_t = C h_t$$

变量含义：
- $t$：timestep（在 vision 里就是 token 的扫描顺序位置）
- $x_t \in \mathbb{R}$：第 $t$ 步的 scalar input（实际上 Mamba 是 per-channel 独立做 SSM，所以 $x_t$ 是某个 channel 的值）
- $h_t \in \mathbb{R}^N$：hidden state，$N$ 是 state dimension（Mamba 默认 $N=16$），**fixed-size memory**
- $h_{t-1}$：上一步的 hidden state
- $y_t$：output
- $C \in \mathbb{R}^{1 \times N}$：output matrix，从 state 读出 output

**这是整个 paper 的核心 insight 来源**：$h_t$ 的维度 $N$ 是固定的，不随序列长度增长。所以：
- **Memory 是 lossy 的**：你把整个历史压进 $N$ 维向量，必然丢信息。
- **Merge complexity 是 $O(N)$ 的常数**：不管前面有多少 token，更新 $h_t$ 的 cost 一样。

对比 causal attention：
- Memory = $\{(k_i, v_i)\}_{i=1}^{t-1}$，**lossless**，但 size 随 $t$ 线性增长。
- Merge complexity = $O(t \cdot d)$，随序列增长。

**直觉**：SSM 用"压缩但有损"换"恒定 cost"，attention 用"无损但膨胀"换"线性 cost"。短序列 attention 完胜（无损 + cost 可接受），长序列 SSM 才有机会（attention 爆了，SSM 还稳）。

### 2.3 为什么 causal 是 SSM 的 inherent property

Equation 2 里 $h_t$ 只依赖 $h_{t-1}$ 和 $x_t$，递归展开：

$$h_t = \overline{A}^t h_0 + \sum_{i=1}^{t} \overline{A}^{t-i} \overline{B} x_i$$

所以 $y_t = C h_t = f(x_1, x_2, \ldots, x_t)$，**只能看到前缀**。这就是 Eq. 4 的 causal mode：

$$y_t = f(x_1, x_2, \ldots, x_t)$$

而 fully-visible mode（Eq. 5）是：

$$y_t = f(x_1, x_2, \ldots, x_t, \ldots, x_T)$$

Attention 默认 fully-visible（BERT/ViT），加 causal mask 变 causal（GPT）。**SSM 没法原生变 fully-visible**，只能靠 bidirectional scan（前向 + 后向两个 branch），但每个 branch 内部还是 causal。这就是 Vision Mamba 用双向 scan 的原因，也是它的根本局限。

作者在 Figure 3(b) 做了个很 punchy 的实验：把 ViT 的 attention 加 causal mask，ImageNet accuracy 掉了。这直接证明 **causal constraint 对 understanding task 是有害的**。

---

## 3. Long-sequence 的量化判据：一个很干净的 ratio

这是论文里我最喜欢的一个小推导（Eq. 6-7），用一个简单的 FLOPs ratio 来定义"什么时候算 long-sequence"。

### 3.1 Transformer block 的 FLOPs (Eq. 6)

$$\text{FLOPs} = 24D^2L + 4DL^2$$

变量：
- $D$：channel / embedding dimension
- $L$：sequence length（token 数）
- $24D^2L$：linear-in-$L$ 项，来自 QKV projection ($3 \times 2D^2L$) + output projection ($2D^2L$) + MLP (ratio 4, so $2 \times 4D^2L = 8D^2L$)，加起来 $6 + 8 + ... = 24$（具体：QKV proj $3 \cdot 2D^2L = 6D^2L$，output proj $2D^2L$，MLP up $2 \cdot 4D^2L = 8D^2L$，MLP down $2 \cdot 4D^2L = 8D^2L$，总 $6+2+8+8 = 24$）
- $4DL^2$：quadratic-in-$L$ 项，来自 attention 的 $QK^T$ ($2DL^2$) 和 $\text{Attn} \cdot V$ ($2DL^2$)

### 3.2 Ratio (Eq. 7)

$$r_L = \frac{4DL^2}{24D^2L} = \frac{L}{6D}$$

**直觉**：当 $r_L > 1$，即 $L > 6D$，quadratic 项主导，attention 成为瓶颈，这才算"long-sequence"。

### 3.3 代入 vision 的实际数字

| Model | $D$ | Threshold $\tau = 6D$ |
|---|---|---|
| ViT-S | 384 | 2304 |
| ViT-B | 768 | 4608 |

| Task | Image size | Patch | $L$ | vs $\tau$ | Long-seq? |
|---|---|---|---|---|---|
| ImageNet cls | $224^2$ | $16^2$ | 196 | $196 \ll 2304$ | ❌ |
| COCO det | $800 \times 1280$ | $16^2$ | ~4000 | $4000 > 2304, \approx 4608$ | ✅ |
| ADE20K seg | $512 \times 2048$ | $16^2$ | ~4096 | 同上 | ✅ |

**这个判据很优雅**：不需要真的去 profile attention 的内存，光看 $L/D$ 比值就能判断。ImageNet 的 196 token 根本不够 long-sequence 的资格，attention 在这上面跑得飞快，SSM 的 efficiency 优势完全用不上。

---

## 4. MambaOut 的架构：Mamba block 减去 SSM

### 4.1 Meta-architecture (Eq. 8-9)

$$X' = \text{Norm}(X)$$
$$Y = (\text{TokenMixer}(X'W_1) \odot \sigma(X'W_2))W_3 + X$$

变量：
- $X \in \mathbb{R}^{N \times D}$：input，$N$ 是 token 数，$D$ 是 channel
- $X'$：normalized input
- $W_1 \in \mathbb{R}^{D \times rD}$：token-mixer branch 的 projection
- $W_2 \in \mathbb{R}^{D \times rD}$：gate branch 的 projection
- $W_3 \in \mathbb{R}^{rD \times D}$：output projection
- $r$：MLP expansion ratio（MambaOut 用 $8/3$，跟 ConvNeXt 一致）
- $\sigma$：activation（GELU）
- $\odot$：element-wise multiplication（**这是 gating 的核心**，gate branch 控制 token-mixer branch 的输出）
- $+X$：residual connection

**直觉**：这个 meta-arch 是 "gated MLP" —— 一条 branch 做 token mixing，另一条 branch 做 gating，两者逐元素相乘。这跟 GLU（Gated Linear Unit）、SwiGLU 的思路同源。Mamba 和 Gated CNN 都用这个骨架，区别只在 TokenMixer 是什么。

### 4.2 Gated CNN vs Mamba (Eq. 10-11)

$$\text{TokenMixer}_{\text{GatedCNN}}(Z) = \text{Conv}(Z)$$
$$\text{TokenMixer}_{\text{Mamba}}(Z) = \text{SSM}(\sigma(\text{Conv}(Z)))$$

**这就是整篇 paper 的实验变量**：Mamba block = Gated CNN block + 额外的 SSM。去掉 SSM 就是 Gated CNN。所以"MambaOut"名字很贴切——把 Mamba 的核心（SSM）拿出去。

MambaOut 具体用：
- Conv = $7 \times 7$ depthwise convolution（跟 ConvNeXt 一致）
- 只在 partial channels 上做 conv（跟 InceptionNeXt 一致，为了 speed）

看 Algorithm 1 的 PyTorch 代码：

```python
class GatedCNNBlock(nn.Module):
    def __init__(self, dim, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0, ...):
        ...
        self.fc1 = nn.Linear(dim, hidden * 2)  # 同时产生 gate 和 input
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=7, 
                              padding=3, groups=conv_channels)  # depthwise
        self.fc2 = nn.Linear(hidden, dim)
    
    def forward(self, x):
        shortcut = x  # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        # g: gate, i: identity input (no mixing), c: conv input
        c = c.permute(0, 3, 1, 2)  # to [B, C, H, W] for conv
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1)  # back to [B, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        return x + shortcut
```

**关键细节**：`fc1` 输出 $2 \times \text{hidden}$ 维，split 成三份：
- $g$：gate branch（过 GELU 后做 gating）
- $i$：identity branch，不做 token mixing，直接 pass through
- $c$：conv branch，过 depthwise conv

这种"partial conv"设计让 conv 只作用在一部分 channel 上，减少 FLOPs 同时保持 receptive field。跟 InceptionNeXt 的思路一模一样。

### 4.3 整体架构（Figure 4）

MambaOut 用 ResNet-style 的 4-stage hierarchical 结构：
- Stem：3 个 $3 \times 3$ stride-2 conv，downsample 8×
- Stage 1-4：每个 stage 堆叠 Gated CNN block，stage 之间 stride-2 downsample
- 不同 size 的配置：

| Size | Blocks (S1-S4) | Channels (S1-S4) | Params | MACs |
|---|---|---|---|---|
| Femto | (3,3,9,3) | (48,96,192,288) | 7.3M | 1.2G |
| Tiny | (3,3,9,3) | (96,192,384,576) | 26.5M | 4.5G |
| Small | (3,4,27,3) | (96,192,384,576) | 48.5M | 9.0G |
| Base | (3,4,27,3) | (128,256,512,768) | 84.8M | 15.8G |

注意 Small 和 Base 在 stage 3 堆了 27 个 block——这是 heavy stage，跟 ConvNeXt 的设计哲学一致（高 resolution 少 channel 的 stage 用更多 block）。

---

## 5. 实验数据：hypothesis 验证

### 5.1 ImageNet classification（Hypothesis 1 验证）

关键对比（Table 1，@224²）：

| Model | Token Mixing | Params | MACs | Acc |
|---|---|---|---|---|
| Vim-S | Conv+SSM | 26M | 5.1G | 80.5 |
| VMamba-T | Conv+SSM | 22M | 5.6G | 82.2 |
| LocalVMamba-S | Conv+SSM | 50M | 11.4G | 83.7 |
| VMambaV9-S | Conv+SSM | 50M | 8.7G | 83.6 |
| **MambaOut-Tiny** | **Conv** | **27M** | **4.5G** | **82.7** |
| **MambaOut-Small** | **Conv** | **48M** | **9.0G** | **84.1** |
| **MambaOut-Base** | **Conv** | **85M** | **15.8G** | **84.2** |

**MambaOut-Small vs LocalVMamba-S**：84.1 vs 83.7，**用更少的 MACs（9.0 vs 11.4G）高出 0.4%**。这是 paper 最 punchy 的结果——去掉 SSM 反而更好。

更有意思的对比：CAFormer-M36（用 7 年前的 separable conv + vanilla attention）85.2%，比所有 visual Mamba 都高 1%+。作者直接说："如果未来要挑战我们的 Hypothesis 1，需要 visual Mamba 在 ImageNet 上达到 SOTA。"——这等于把球踢给了 visual Mamba 社区。

### 5.2 COCO detection（Hypothesis 2 验证）

| Backbone | APb | APm |
|---|---|---|
| VMamba-T | 46.5 | 42.1 |
| LocalVMamba-T | 46.7 | 42.2 |
| **MambaOut-Tiny** | **45.1** | **41.0** |
| VMambaV9-T | 47.4 | 42.7 |
| TransNeXt-Tiny (SOTA hybrid) | 49.9 | 44.6 |

MambaOut-Tiny 比 VMamba-T 低 1.4 APb。**在 long-sequence task 上，SSM 确实有用**。但 visual Mamba 离 SOTA hybrid（TransNeXt）还有 3+ AP 的差距。

### 5.3 ADE20K segmentation（Hypothesis 2 验证）

| Backbone | mIoU (SS) | mIoU (MS) |
|---|---|---|
| VMamba-T | 47.3 | 48.3 |
| LocalVMamba-T | 47.9 | 49.1 |
| **MambaOut-Tiny** | **47.4** | **48.6** |
| TransNeXt-Tiny | 51.1 | 51.2 |

MambaOut-Tiny 介于 VMamba-T 和 LocalVMamba-T 之间，比 LocalVMamba-T 低 0.5 mIoU。同样验证 SSM 在 long-seq 上的价值，但离 SOTA hybrid 还差 3+ mIoU。

---

## 6. 更深层的 intuition 与联想

### 6.1 SSM 的 lossy memory vs attention 的 lossless memory

这是 Figure 2 的核心。可以把 attention 看成"完美但昂贵的记忆"——所有历史 token 的 (k, v) 都存着，检索时全扫一遍。SSM 是"压缩但有损的记忆"——历史被压进 fixed-size $h$，检索是 $O(1)$ 的矩阵向量乘。

**类比**：attention 像把整本书摊在桌上随时翻，SSM 像把书读完后写一段摘要，下次只看摘要。短文档摊桌上更快，长文档只能写摘要。

在 vision understanding 上，图像就 196-4K token，"摊桌上"完全没问题，为什么要"写摘要"？这就是 paper 的核心质问。

### 6.2 Causal constraint 为什么对 understanding 有害

Figure 3(b) 的 ViT causal mask 实验很有说服力。Understanding task 需要每个 token 看到**全局**信息（比如识别一个物体需要看周围 context）。Causal mask 强制 token 只能看前缀，相当于让模型"蒙着眼睛从左上到右下逐步揭开图像"，这显然违背 understanding 的本质。

Visual Mamba 用 bidirectional scan 缓解：一个 branch 从左上到右下，一个从右下到左上，两个 branch 的输出 merge。但**每个 branch 内部仍然是 causal 的**——左上角的 token 在前向 branch 里永远看不到右下角，只能靠后向 branch 补。这相当于把一个 fully-visible 的 attention 拆成两个"半盲"的 RNN，再拼起来。信息融合的 quality 肯定不如原生的 fully-visible attention。

**这解释了为什么 visual Mamba 在 ImageNet 上打不过甚至打不过 Gated CNN**——SSM 带来的 causal bias 是净负债，而它的 efficiency 优势在 196 token 上根本用不上。

### 6.3 为什么 detection/segmentation 上 SSM 有用

这里 token 数 ~4K，attention 的 quadratic 项开始主导（$L \approx 6D$）。而且 dense prediction 任务对**长程依赖**敏感（一个像素的语义依赖远处 context），SSM 的 global receptive field（通过递归扫完整个序列）比局部 conv 有优势。

但注意：visual Mamba 在这些任务上仍然打不过 TransNeXt 这种 hybrid。原因可能是：
1. SSM 的 causal bias 仍然是个负债（detection 也是 understanding task）
2. Bidirectional scan 的信息融合不如 attention 的原生 fully-visible
3. SSM 的 lossy memory 在 fine-grained dense prediction 上丢信息

所以 Hypothesis 2 说的是"值得探索"，不是"SSM 一定行"。

### 6.4 与 LLM 场景的对比

作者在 conclusion 里暗示要扩展到 LLM/LMM。在 LLM 上：
- Long-sequence：✅（context window 8K-128K+）
- Autoregressive：✅（next-token prediction 天然 causal）

两个 characteristic 都满足，所以 Mamba/RWKV 在 LLM 上有真实价值。Jamba（AI21 Labs 的 Transformer-Mamba hybrid）就是这条路线的例子。

这跟 vision 形成鲜明对比：vision understanding 两个都不满足（classification）或只满足一个（detection/seg），所以 SSM 的价值大打折扣。

### 6.5 Gated CNN 的历史脉络

论文追溯到 Dauphin et al. 2017 的 Gated CNN（reference [18]），这是 pre-Transformer 时代的 language model 架构。Mamba 其实是"Gated CNN + SSM"的组合——把 Gated CNN 的 conv 换成 SSM 来获得 long-range + autoregressive 能力。

所以 MambaOut 本质上是**退回 Gated CNN**，砍掉 Mamba 加的那个 SSM 层。这个 ablation 设计非常干净：control 了所有其他变量（gating、partial conv、meta-arch），只 isolate SSM 的贡献。

### 6.6 与 ConvNeXt 的关系

MambaOut 的 Gated CNN block 跟 ConvNeXt block 非常像（$7 \times 7$ depthwise conv + MLP + residual），区别在于：
- MambaOut 用 gating（$\text{act}(g) \cdot \text{cat}(i, c)$），ConvNeXt 没有 gating
- MambaOut 用 partial conv，ConvNeXt 是 full conv
- MambaOut 的 meta-arch 是 MetaFormer 风格（token mixer + MLP 融合）

实际上 MambaOut-Tiny（82.7%）跟 ConvNeXt-T（82.1%）很接近，gating 带来一点提升。这说明 visual Mamba 的 conv 部分已经足够强，SSM 是个"额外负担"。

### 6.7 Selectivity 的 vision 适用性

Mamba 的 "selective" 体现在 $\Delta, A, B, C$ 是 input-dependent 的。在 language 上，这让模型能根据 token 内容决定"记多少"（比如遇到标点就 reset，遇到关键词就 hard remember）。

在 vision 上，token 是 patch embedding，selectivity 的语义没那么清晰——一个 patch 的"重要性"更多取决于 spatial context（这个 patch 是物体还是背景），而不是 patch 本身的内容。而 SSM 的 causal scan 让 selectivity 变成"基于扫描前缀"的判断，这跟 spatial context 的需求不匹配。

这可能也是 visual Mamba 表现不佳的深层原因：SSM 的 selectivity 机制是为 temporal/sequential data 设计的，spatial data 需要的是另一种 attention pattern。

---

## 7. 批判性思考与 open questions

### 7.1 实验设计的 fairness

MambaOut 用 DeiT 的训练 recipe（300 epoch, batch 4096, lr 4e-3, strong augmentation），visual Mamba 各家用的 recipe 不完全一致。作者把训练 hyper-parameter 控制一致了（Table 5），但 visual Mamba 可能没在自己的最优 recipe 下跑。不过作者也指出 visual Mamba 即使用自己的 recipe 也打不过 MambaOut，所以这个 concern 不太致命。

### 7.2 Isotropic vs Hierarchical

Vim（Vision Mamba）是 isotropic（像 ViT），VMamba/LocalMamba 是 hierarchical（像 ResNet）。MambaOut 是 hierarchical。Vim 在 ImageNet 上只有 80.5%（Vim-S），明显弱于 hierarchical 的方案。这说明 visual Mamba 的 isotropic 路线问题更大（没有 inductive bias，又加了 causal constraint）。

### 7.3 未来的 visual Mamba 该往哪走

根据 paper 的 logic：
1. **不要在 ImageNet classification 上死磕** —— SSM 在这是负债
2. **在 high-resolution dense prediction 上发力** —— detection/seg/medical imaging/remote sensing 这些 token 数真的大的场景
3. **解决 causal bias** —— 双向 scan 不够，可能需要更聪明的 non-causal SSM 变体（比如把 SSM 改成可以"看到未来"的形式，但这跟 RNN 本质冲突）
4. **跟 attention hybrid** —— 像 Jamba 那样，在关键层用 attention 保持 fully-visible，其他层用 SSM 省 cost

### 7.4 Mamba-2 的后续

MambaOut 写于 Mamba-1 时代。Mamba-2（Tri Dao 后来的工作）把 SSM 和 attention 统一了（SSM = structured linear attention），在效率上有提升。但 paper 的核心 argument（causal bias + lossy memory 对 understanding 不友好）对 Mamba-2 同样适用，因为 Mamba-2 本质还是 recurrent。

---

## 8. 总结：这篇 paper 的真正贡献

MambaOut 不是在 propose 一个新 SOTA model，而是在做一个**概念澄清 + 实验验证**的 work。它的价值在于：

1. **给 visual Mamba 社区提供了一个 baseline**：如果你做的 visual Mamba 打不过 MambaOut（一个去掉 SSM 的版本），那你的 SSM 没有贡献，只是徒增复杂度。
2. **明确了 SSM 的适用边界**：long-sequence + autoregressive 两个 condition，缺一不可。Vision understanding tasks 至多满足一个。
3. **用 Occam's razor 做了一次"剃刀实验"**：最简单的解释（gating + conv 就够了）胜过复杂的解释（加 SSM）。

**对你（Karpathy）的直觉共鸣**：这篇 paper 的精神跟你在 "State of GPT" / "Let's build GPT" 里强调的"先理解 mechanism 再用 tool"完全一致。Mamba 在 LLM 上有道理（long + autoregressive），但搬到 vision 上需要重新审视 task characteristic，不能盲目 follow 趋势。

---

## References

- MambaOut paper: https://arxiv.org/abs/2405.14174
- Mamba 原始 paper: https://arxiv.org/abs/2312.00752
- S4 (structured SSM): https://arxiv.org/abs/2111.00396
- Vision Mamba (Vim): https://arxiv.org/abs/2401.09417
- VMamba: https://arxiv.org/abs/2401.10166
- LocalMamba: https://arxiv.org/abs/2403.09338
- PlainMamba: https://arxiv.org/abs/2403.17695
- Gated CNN (Dauphin 2017): https://arxiv.org/abs/1612.08083
- ConvNeXt: https://arxiv.org/abs/2201.03545
- MetaFormer: https://arxiv.org/abs/2111.11418
- InceptionNeXt: https://arxiv.org/abs/2303.16900
- Jamba (Transformer-Mamba hybrid LLM): https://arxiv.org/abs/2403.19887
- RWKV: https://arxiv.org/abs/2305.13048
- Mamba-2: https://arxiv.org/abs/2405.21060
- Tri Dao 的博客 (SSM 系列): https://tridao.me/
- Albert Gu 的 SSM 讲解: https://github.com/state-spaces/mamba
- Kobe Bryant farewell speech (paper 致敬来源): https://www.youtube.com/watch?v=JiZkGCowAZc
