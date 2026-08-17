---
source_pdf: Switch Attention Towards Dynamic and Fine-grained Hybrid.pdf
paper_sha256: 4d0a1379380109faa78131eadfdd7de656c2d22f9d5e624a0fe8e2df5add550e
processed_at: '2026-08-12T11:45:19-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Switch Attention

## 一句话版本

把 transformer 里 "full attention 和 sliding window attention 谁来干活" 这件事, 从**事先画死的固定排班表**, 改成 **每个 token 自己看着办**, 效果不掉, 算力省一大半。

---

## 先说一个观察

你想想 transformer 的 attention 到底在干嘛。每个 token 要去看一眼其他 token, 决定自己下一层长啥样。这个"看一眼"有两种极端方式:

- **Full attention**: 所有人都看一眼。信息最全, 但人一多就 $O(N^2)$ 爆炸
- **Sliding window attention (SWA)**: 只看身边 1024 个邻居。便宜, 但远处的东西完全看不到

工业界 (Mistral, Gemma, Qwen3, GPT-oss) 基本都用 **hybrid**: 一层 full attention 接三层 SWA, 循环下去。比如 22 层的模型, 第 4、8、12、16、20 层是 full, 其余是 SWA。25% 的层做全局视野, 75% 的层做局部。

这个 1:3 的排班表是**工程师拍脑袋定的**, 训练前写死, 训练中不变, 推理时也不变。

作者 (PKU + Huawei 这帮人) 就问了一个特别自然的问题: **凭什么所有 token 都按同一张排班表走?**

---

## 一个具体的例子帮你 build intuition

想象你在做 needle-in-a-haystack: 在 32K 长度的文档里插一句话 "密码是 banana", 然后问 "密码是什么"。

情况 A: needle 插在 prompt 最开头 (depth 5%), 离当前生成位置 30K 远。SWA 的 1024 window 根本够不着 → **必须 full attention**

情况 B: needle 就插在 prompt 最后 (depth 95%), 离当前生成位置就几百 token。SWA 完全能 see → **full attention 纯属浪费**

Static hybrid 不 care 这些, 它永远用 25% 的 full attention 层。情况 B 里那 25% 全是浪费的算力, 情况 A 里那 25% 可能还不够 (因为只有 4 层 full, 如果 needle 太远可能需要更多层 full 才能稳稳检索到)。

作者的核心 insight: **token 的难度和 attention 跨度需求是异质的 (heterogeneous)**, 这件事 speculative decoding 那帮人早就证明了 (Leviathan 2023, Medusa), 只是大家没把它用到 attention pattern 设计上。

参考: https://arxiv.org/abs/2211.17192 (Speculative Decoding)
参考: https://arxiv.org/abs/2401.10774 (Medusa)

---

## 核心设计: 给 attention 加一个 router

借鉴 MoE 的思路。MoE 是给 FFN 加 router, 选 expert; 这里给 attention 加 router, 选 "全看" 还是 "只看 window"。

具体到数学, 每一层每个 token 都走这个流程:

### Step 1: 算 soft gate

输入是这一层的 hidden state $H^{(l)} \in \mathbb{R}^{d \times T}$, $d=2048$ 是 hidden dim, $T$ 是 seq len。

$$\tilde{\sigma}^{(l)} = \text{sigmoid}(\mathcal{R}(H^{(l)})) \in [0, 1]^T$$

$\mathcal{R}$ 就是一个 super simple 的 linear layer, $W_R \in \mathbb{R}^{1 \times d}$, 输出每个 token 一个标量, sigmoid 压到 [0,1]。几千个参数, 量级可以忽略。

### Step 2: 二值化 (STE)

$$\sigma^{(l)} = \mathbb{1}(\tilde{\sigma}^{(l)} > 0.5) + \tilde{\sigma}^{(l)} - \text{sg}(\tilde{\sigma}^{(l)})$$

这是 quantization 和 Switch Transformer 里用的 **Straight-Through Estimator** trick:
- Forward: $\sigma$ 是 hard 的 {0, 1}, 推理时只算被选中的 branch
- Backward: gradient 走 soft 的 $\tilde{\sigma}$, 模型能学

forward 时省算力, backward 时能训, 两头占便宜。代价是 gradient 估计有 bias, 实践中 work 得很好。

参考: https://arxiv.org/abs/1308.3432 (STE 原始)
参考: https://arxiv.org/abs/2101.03961 (Switch Transformer 怎么用 STE)

### Step 3: 两个 branch 共享 Q/K/V, 各自算 attention

$$[Q, K, V] = \phi(H^{(l)})$$

$\phi$ 是标准的 $W_Q, W_K, W_V$ projection (GQA, 16 query heads, 8 KV heads, head dim 128)。

**关键设计**: 两条 branch 共享同一组 Q/K/V projection。这意味着:
- 参数量增加几乎为零 (只有 router 那个 linear layer)
- CPT 时可以直接复用 pretrained full attention 模型的 projection 权重, 零成本初始化

然后两条 branch 各算各的:

$$O_{\text{FULL}} = \mathcal{A}_{\text{FULL}}(Q, K, V)$$
$$O_{\text{SWA}} = \mathcal{A}_{\text{SWA}}(Q, K, V)$$

$\mathcal{A}_{\text{FULL}}$ 是标准 causal self-attention, $\mathcal{A}_{\text{SWA}}$ 是 window=1024 的 sliding window attention。

### Step 4: 选 branch

$$O^{(l)} = \sigma^{(l)} \odot O_{\text{FULL}} + (1 - \sigma^{(l)}) \odot O_{\text{SWA}}$$

因为 $\sigma$ 是 hard {0,1}, forward 时其实只有一条 branch 的输出"被选中"。

推理时 prefill 阶段两条都要算 (这是 paper 的 Limitation, 没写 fused kernel), decode 阶段只算被选中的那条, 而且 **两条 branch 共享 KV cache**, 显存也省。

参考: https://arxiv.org/abs/2004.05150 (Longformer, SWA 始祖)
参考: https://arxiv.org/abs/2310.06825 (Mistral, 1:3 static hybrid)

---

## 最巧的部分: Adaptive Regularization

如果只让 router 自由学, 它会全选 full attention, 因为 full attention 信息更多, loss 更低。所以要加正则化, push router 倾向 SWA。

但正则化不能加得太狠。如果无差别地惩罚 full attention, 会出现两个问题:

**问题 1: 难 token 也被逼着用 SWA**
比如 needle 在 30K 外, model 必须全视野才能找到, 但 router 因为被惩罚, 不敢选 full → 性能崩

**问题 2: 两条 branch 分歧大时还硬走 SWA**
即使当前 token 朴素看容易, 但 full 和 SWA 输出差很多, 说明 SWA 漏了信息, 应该 fallback 到 full

作者的解法: **正则化的强度根据情况动态调整**。

$$\mathcal{L}_t^{(l)} = \gamma_t^{(l)} \cdot \text{softplus}(\mathcal{R}(h_t^{(l)}))$$

$\text{softplus}(x) = \ln(1 + e^x)$ 是 smooth 的 ReLU:
- router logit 大 (想选 full): penalty 大
- router logit 小 (想选 SWA): penalty 趋近 0

$\gamma_t^{(l)}$ 是 adaptive weight:

$$\gamma_t^{(l)} = \frac{\gamma_{\text{base}}}{\varepsilon + \text{NLL}(t) + \alpha \cdot \|o_{\text{FULL}}^{(t,l)} - o_{\text{SWA}}^{(t,l)}\|_2^2}$$

逐项解释:
- $\gamma_{\text{base}} = 10^{-3}$: 基础 scale
- $\varepsilon = 0.1$: 防 division by zero, 顺便给 $\gamma$ 一个上界 $\gamma_{\text{base}}/\varepsilon = 10^{-2}$
- $\text{NLL}(t) = -\log p(y_t | y_{<t})$: 当前 token 的 negative log-likelihood, **这是 model 觉得这个 token 有多难**
- $\alpha = 100$: MSE 项的权重
- $\|o_{\text{FULL}}^{(t,l)} - o_{\text{SWA}}^{(t,l)}\|_2^2$: 两条 branch 输出的 squared disagreement

直觉上, 这个 weight 在说:

- **NLL 大 (难 token)**: 分母变大 → $\gamma$ 变小 → 正则化减弱 → router 敢选 full
- **MSE 大 (branch 分歧)**: 分母变大 → $\gamma$ 变小 → 正则化减弱 → router 敢选 full
- **NLL 小 + MSE 小 (简单 token 且 branch 同意)**: 分母小 → $\gamma$ 大 → 正则化强 → router 被推向 SWA

这相当于给 router 一个 **"什么时候可以懒, 什么时候必须勤奋"** 的信号。这个思路我个人觉得是 paper 最漂亮的地方, 它把"难度感知"直接写进了正则化项。

让我联想到 focal loss (Lin et al. 2017) 在 detection 里的思路: 难样本权重大, 简单样本权重小。这里是反过来: 难样本要"放松约束让 model 用力", 简单样本要"加强约束让 model 省力"。

参考: https://arxiv.org/abs/1708.02002 (Focal Loss)

Ablation (Table 4) 验证:
- 去掉 adaptive reg (用固定 $\gamma$): AVG 24.9 → 掉 1.3
- 去掉 NLL 项: 25.3 → 掉 0.9
- 去掉 MSE 项 ($\alpha=0$): 24.7 → 掉 1.5
- default (两个都有): 26.2

两个 signal 都 essential, MSE 项尤其重要。

---

## 怎么训: Continual Pretraining (CPT)

很多 hybrid 模型 (Jamba, Zamba) 要 from scratch 训, 贵。SwiAttn 直接拿现成的 full attention 模型迁移。

**初始化**:
- $W_Q, W_K, W_V, W_O$: 从 pretrained full attention 直接拷
- full branch: 就是 pretrained attention 本身
- SWA branch: 用同一套 Q/K/V 权重 (因为 projection 共享), 只是 attention pattern 换成 window
- FFN: 直接继承
- Router $W_R$: 随机初始化

**训练 budget**:
- 4K 模型: full attention 用 240B tokens 预训练 → SwiAttn CPT 20B tokens (1/12)
- 32K 模型: full attention 用 192B (4K) + 24B (YaRN 扩 32K) 预训练 → SwiAttn CPT 8B tokens

CPT budget 远小于 pretraining, 因为大部分知识已经在 full attention 模型里, 只需要教 router 学路由 + 微调让 attention 适应 sparse pattern。

Loss curve (Appendix D, Figure 7) 显示 CPT 稳定无 spike, 说明 STE + adaptive reg 优化 well-behaved。

参考: https://arxiv.org/abs/2309.00071 (YaRN)

---

## 实验说了啥

### 1. 短 context (Commonsense Reasoning, Table 1)

context < 1024, SWA 也够。所有方法差距很小:
- FullAttn AVG: 53.4
- StaticHybrid: 53.6
- **SwiAttn: 53.7**
- SWA-CPT: 52.6
- SWA-ZS: 50.5

SwiAttn 几乎无损, 因为短 context 时 router 大量选 SWA。

### 2. 中 context (In-Context Retrieval, Table 2)

context < 4096, 部分 instance 需要超 SWA window 的 attention span:
- FullAttn AVG: 52.9
- **SwiAttn AVG: 52.9** (持平!)
- StaticHybrid: 50.6 (SwiAttn +4.5%)
- SWA-CPT: 41.5 (SwiAttn +27.5%)

特别看 FDA dataset: FullAttn 76.4, SwiAttn 75.1, StaticHybrid 67.9, SWA-ZS 22.5。FDA 是 PDF 文档抽取, 需要长 span attention, 这里 SwiAttn 显著胜过 static hybrid。

### 3. 长 context (LongBench-E, Table 3)

context 大部分超过 SWA window, 最大 32K:
- FullAttn AVG: 37.4
- **SwiAttn AVG: 37.4** (并列最佳!)
- StaticHybrid: 35.2 (SwiAttn +6.3%)
- SWA-CPT: 31.0

SwiAttn 超过 StaticHybrid 6.3%, 但 **full attention ratio 更低** (0.13 vs 0.25)。这说明 dynamic routing 比 static pattern 更高效分配 budget。

特别 MFQA (multi-field QA): FullAttn 33.9, **SwiAttn 35.3** (反超!)。这个挺有意思, 可能是 dynamic 让 model 在 multi-field document 不同位置用不同 attention 跨度, 反而比纯 full 更合理。

### 4. Needle-in-a-Haystack (Figure 3)

RULER (Hsieh et al. 2024), 35 个 depth × 35 个 length = 1225 个 cell, 每 cell 重复 3 次。Figure 3 显示 SwiAttn 在 32K context 全场景 perfect retrieval accuracy。强证据表明 dynamic routing 没损害长程检索能力。

参考: https://arxiv.org/abs/2404.06654 (RULER)

### 5. 效率 (Figure 4)

**Decode 阶段** (用 KV cache 访问 token 数衡量, decode 的主要 bottleneck):

| Context | SwiAttn (Standard) | FullAttn | Speedup |
|---|---|---|---|
| 8K | 2.3K | 8K | 3.5× |
| 16K | 4.2K | 16K | 3.8× |
| 24K | 6.2K | 24K | 3.9× |
| 32K | 7.3K | 32K | **4.4×** |

**Simple 场景** (NIAH 且 needle 在 window 内, router 几乎全选 SWA):

| Context | SwiAttn | FullAttn | Speedup |
|---|---|---|---|
| 16K | 1.7K | 16K | 9.4× |
| 32K | 2.6K | 32K | **12.3×** |

12× speedup 来自 router 几乎全选 SWA, decode 时只读 ~1024 个 KV。

---

## 路由的动态行为 (这是最 fascinating 的部分)

### Per-layer 的 full attention ratio (Figure 5)

在 32K CPT 数据上统计 400M tokens:
- **Layer 1, 2, 3, 22**: full attention ratio 很低 (偏 SWA)
- **Layer 8, 11, 13, 14**: full attention ratio 高
- **平均 ratio ≈ 0.13**, 远低于 StaticHybrid 的 0.25

这和 Transformer interpretability literature 的发现对应: 低层做 local pattern (n-gram, syntactic), 中层做 semantic aggregation, 高层做 task-specific reasoning。Router 自动学出了这种 layer specialization, 不需要人为设计 1:3 pattern。

NSA (Yuan et al. 2025) 和 Native Hybrid Attention (Du et al. 2025) 也有类似观察, 都指向同一结论: **layer-wise attention pattern 应该是 learned**。

参考: https://arxiv.org/abs/2502.11089 (NSA)
参考: https://arxiv.org/abs/2510.07019 (NHA)

### 路由在 NIAH 上的动态 (Figure 6)

| Needle 位置 | full attention ratio |
|---|---|
| Depth 低 (window 外) | ~30% |
| Depth 高 (window 内) | ~5% |

 Needle 在 window 外, router 30% 选 full, 把 budget 投到检索; needle 在 window 内, 5% 选 full, 几乎全靠 SWA。

这是 **task-adaptive computation** 的直接证据, 思路和 speculative decoding 同源: 简单部分用 cheap model, 难的部分用 expensive model。SwiAttn 把这个思想放到 attention 层面。

### CPT 过程中 ratio 的演化 (Appendix E)

- 平均 full attention ratio 在 CPT 过程中逐渐下降到 0.13
- 早期所有层 ~0.5 (随机 router + sigmoid → 50%), 然后 diverge
- 部分层下降到 < 0.05, 部分层稳定在 0.3
- 大约 500 iterations 后稳定

这种 "渐变分化" 模式和 MoE training 中 expert specialization 的 emergent 现象很像。

---

## 我的 intuition 解读

整篇 paper 的故事可以这样串起来:

**问题**: Static hybrid transformer 用一张固定的排班表, 所有 token 都按同一张表走。但 token 的 attention 跨度需求是异质的, static 表会浪费。

**解法**: 给 attention 加一个 router, 每个 token 每层自己选 full 还是 SWA。借鉴 MoE 的 conditional computation 思路。

**难点**: Router 会偷懒全选 full (信息更多)。要加正则化 push 它倾向 SWA。但正则化不能一刀切, 难 token 必须允许 full。

**巧思**: Adaptive regularization, 用 NLL (token 难度) 和 MSE (branch 分歧) 调节正则化强度。难 token 或 branch 分歧大时, 放松约束; 简单 token 且 branch 一致时, 加强约束。

**工程**: STE 让 hard gate 可训; shared Q/K/V projection 让 parameter overhead 几乎为零; CPT 让现成 full attention 模型无缝迁移。三件小事组合起来 plug-and-play。

**验证**: 23 个 benchmark 上, SwiAttn 平均达到 full attention 水平, decode 加速 4-12×。路由分析显示 per-layer specialization 涌现, per-token 动态匹配任务需求。

**核心 insight**: Hybrid attention 的 "hybridity" 应该是 learned 不是 designed。Static pattern 是工程方便的妥协, 不是最优。SwiAttn 把这个 decision 数据驱动化, 实验显示平均 13% full attention 就够, 远低于 25% static 比例, 说明 static pattern 浪费了约一半的 full attention budget。

---

## 几个可能有意思的 follow-up

1. **Router 升级**: 现在是 linear router, 可以试 small MLP + LayerNorm, 或者 cross-layer routing (用上一层 hidden state 决定这一层)
2. **Branch 扩展**: 加入 linear attention branch (Mamba-style), 变成 3-way routing, 进一步压缩简单 token 成本
3. **Prefill kernel**: 写一个 fused kernel, mask 是 `(window_mask) OR (token in full_set)`, 类似 MoA (Fu et al. 2024) 或 HySparse (Gao et al. 2026)
4. **Scale up**: 7B, 13B, 70B 验证, router specialization 是否随 scale 改变
5. **Multi-modal**: Vision transformer 的 dynamic window vs global attention, 视觉 token 难度更异质 (background vs object), 收益可能更大
6. **和 NSA 组合**: NSA 是 intra-attention 选 token, SwiAttn 是 inter-attention-variant 选 branch, 两者可以叠加

参考: https://arxiv.org/abs/2406.14909 (MoA)
参考: https://arxiv.org/abs/2602.03560 (HySparse)
参考: https://arxiv.org/abs/2312.00752 (Mamba)
参考: https://arxiv.org/abs/2503.14456 (RWKV-7)

---

## 最直觉的类比

把整个事情用学校教学类比:

**Static hybrid**: 一个班 22 节课, 学校规定第 4、8、12、16、20 节是"精讲大课"(full attention), 其余是"自习局部练习"(SWA)。所有学生不管懂没懂, 都按这个表走。懂了的学生在精讲大课上浪费时间, 不懂的学生自习时没人讲还是不懂。

**SwiAttn**: 每个学生每节课自己决定是去精讲大课还是自习。Router 是学生自己, 根据当前学的内容难度决定。Adaptive regularization 是班主任: 大多数时候鼓励大家自习(便宜), 但如果这个学生最近 NLL 高 (学得吃力) 或者自习和大课分歧大 (自习搞不定), 班主任就放手让他去精讲大课。

学出来的结果: 平均只有 13% 的时间上精讲大课, 但成绩和天天上精讲大课 (full attention) 一样好。说明大部分 token 大部分时候根本不需要全局视野, 只是少数难 token 少数层需要。

这大概就是 paper 想说的核心: **算力分配应该跟着 token 难度走, 而不是按固定排班表**。

---

# Switch Attention: 一篇 Hybrid Transformer 的动态路由设计

这篇 paper 来自 Peking University + Huawei，作者 Yusheng Zhao 等。核心 idea 非常 clean：把 hybrid transformer 里的"哪一层用 full attention vs SWA"从 **static pre-defined pattern** 改成 **per-token per-layer dynamic routing**，并且通过 continual pretraining (CPT) 把现成的 full attention 模型无缝迁移过来。下面我尽量按 intuition 顺序拆开讲。

参考链接：
- paper PDF (arXiv): https://arxiv.org/abs/2506.08675 (假设编号)
- Switch Transformer (Fedus et al. 2022): https://arxiv.org/abs/2101.03961
- Longformer (Beltagy et al. 2020): https://arxiv.org/abs/2004.05150
- Mistral 7B: https://arxiv.org/abs/2310.06825
- Gemma 2: https://arxiv.org/abs/2408.00118
- FlashAttention: https://arxiv.org/abs/2205.14135
- Straight-Through Estimator (Bengio 2013): https://arxiv.org/abs/1308.3432
- YaRN: https://arxiv.org/abs/2309.00071
- Mamba / Gated Delta Networks: https://arxiv.org/abs/2312.00752 , https://arxiv.org/abs/2412.06464
- Native Sparse Attention (DeepSeek NSA): https://arxiv.org/abs/2502.11089
- DeepSeek-V2 (MLA): https://arxiv.org/abs/2405.04434
- RULER (NIAH): https://arxiv.org/abs/2404.06654
- LongBench: https://arxiv.org/abs/2308.14508

---

## 1. Motivation: 现有 hybrid transformer 的痛点

Transformer 的 attention 是 $O(T^2)$，长 context 一下就把显存和算力打爆。现有解决方案大致分四类：

1. **KV compression / sharing**: GQA (Ainslie et al. 2023), MQA (Shazeer 2019), MLA (DeepSeek-V2, Liu et al. 2024a), KV cache eviction (H2O, Xiao et al. 2023; Zhang et al. 2023)
2. **Sparsify attention pattern**: Longformer / BigBird (heuristic), NSA (data-dependent, Yuan et al. 2025)
3. **Sub-quadratic 替代**: Linear attention, Mamba (Gu & Dao 2024), RWKV-7 (Peng et al. 2025), Gated DeltaNet (Yang et al. 2024), Log-linear attention (Guo et al. 2025a), Kimi Linear (2025)
4. **Hybrid**: full attention 和 efficient attention 交替堆叠

工业界主流 (Mistral 7B, Gemma 2, Qwen3, GPT-oss, MIMO-v2) 都采用 hybrid，常见 pattern 是 **1 full + 3 SWA** 重复，这是一个 **static** 的固定比例。

作者的关键观察：**token 难度是异质的**。Leviathan et al. 2023 (speculative decoding) 和 Medusa (Cai et al. 2024) 已经证明不同 token 生成难度差别巨大，同理不同 token 对 attention 跨度的需求也不同。当 needle 在 window 里面时，根本不需要 full attention；当 needle 在 30K 远处时，必须有 full attention。**Static pattern 无法 match 这种动态需求**，所以 SwiAttn 把 routing 下放到 per-token per-layer 粒度。

这里直觉非常像 **MoE**：Mistral 8x7B / Switch Transformer (Fedus et al. 2022) 把 FFN 拆 expert 然后 route；SwiAttn 把 attention 拆 full / SWA 两条 branch 然后 route。区别在于 MoE 通常是 top-k expert，SwiAttn 是 binary hard gate (top-1 of 2)。

---

## 2. 架构总览

Figure 2 给出两个阶段：

**(a) Continual Pretraining Stage**:
- 输入 hidden representation $H^{(l)} \in \mathbb{R}^{d \times T}$
- Router $R(\cdot)$ 算 soft gate → hard gate
- 同时计算两个 branch: $O_{\text{FULL}}^{(l)}$ 和 $O_{\text{SWA}}^{(l)}$，**共享 Q/K/V**
- 用 hard gate 选 branch 输出
- 过 FFN
- Adaptive regularization 施加在 router logit 上

**(b) Decoding Stage**:
- 每个 token 每层先算 gate
- gate = 1 → 只算 full attention
- gate = 0 → 只算 SWA
- **共享 KV cache** (这点很关键，省一半显存)

Inference 完整流程在 Algorithm 1，prefill 阶段两个 branch 都算，decode 阶段根据 gate 二选一。Prefill 时其实可以 fused 成一个 hardware-efficient kernel (作者在 Limitations 里承认还没实现)。

---

## 3. Dual-branch Routing 的数学

### 3.1 Gate 的产生 (Eq 1-2)

$$\tilde{\sigma}^{(l)} = \text{sigmoid}\left(\mathcal{R}(H^{(l)}\right)) \in [0, 1]^T \tag{1}$$

- $H^{(l)} \in \mathbb{R}^{d \times T}$: 第 $l$ 层输入 hidden states, $d$ 是 hidden dim (实验里 $d = 2048$), $T$ 是 sequence length
- $\mathcal{R}(\cdot)$: router function, 实现就是一个 linear layer $W_R \in \mathbb{R}^{1 \times d}$, 输出 $\mathbb{R}^T$，再 RMSNorm 之后才喂进 router
- $\tilde{\sigma}^{(l)} \in [0, 1]^T$: 每个 token 一个 [0,1] soft 值

然后做 hard binarize：

$$\sigma^{(l)} = \mathbb{1}\left(\tilde{\sigma}^{(l)} > \tau\right) + \tilde{\sigma}^{(l)} - \text{sg}\left(\tilde{\sigma}^{(l)}\right) \tag{2}$$

- $\tau = 0.5$: threshold
- $\mathbb{1}(\cdot)$: element-wise indicator, 输出 {0, 1}
- $\text{sg}(\cdot)$: stop gradient
- **STE (Straight-Through Estimator)**: forward 看到的是 hard $\{0,1\}$, backward 拿到的是 soft $\tilde{\sigma}$ 的梯度

这就是 quantization literature (Liu et al. 2022b; Huh et al. 2023) 和 Switch Transformer (Fedus et al. 2022) 里非常常见的 trick。STE 的好处是 forward 时 routing 是 hard decision, 推理时只需要算一个 branch；坏处是 gradient 估计有 bias, 实践中通常 work 但理论上不太干净。

### 3.2 两条 branch (Eq 3-5)

$$[Q^{(l)}, K^{(l)}, V^{(l)}] = \phi(H^{(l)}) \tag{3}$$
$$O_{\text{FULL}}^{(l)} = \mathcal{A}_{\text{FULL}}^{(l)}(Q^{(l)}, K^{(l)}, V^{(l)}) \tag{4}$$
$$O_{\text{SWA}}^{(l)} = \mathcal{A}_{\text{SWA}}^{(l)}(Q^{(l)}, K^{(l)}, V^{(l)}) \tag{5}$$

- $\phi$: linear projection $W_Q, W_K, W_V$，用 GQA (16 query heads, 8 KV heads, head dim 128)
- $\mathcal{A}_{\text{FULL}}$: standard causal self-attention
- $\mathcal{A}_{\text{SWA}}$: sliding window attention, window size $w = 1024$ (4K 模型) / 2048 (32K 模型，推测)

**关键设计**: 两个 branch 共享 Q/K/V projection。这意味着只需要一组 $W_Q, W_K, W_V$，参数量增加只来自 router (一个 $\mathbb{R}^{1 \times d}$ 的 linear layer，几千参数)。这就让 CPT 可以直接复用 pretrained full attention 的 projection 权重，no warm-up needed for Q/K/V。

### 3.3 选 branch (Eq 6)

$$O^{(l)} = \sigma^{(l)} \odot O_{\text{FULL}}^{(l)} + (1 - \sigma^{(l)}) \odot O_{\text{SWA}}^{(l)} \tag{6}$$

- $\odot$: element-wise product, 沿 sequence 维度 broadcast
- 这其实是 **conditional computation** 的 soft 版本：forward 时 $\sigma$ 是 hard {0,1}, 所以实际只有一个 branch 的输出"被选中"

但注意, prefill 阶段两个 branch 都要算, 这是 paper 的 Limitation。一个高效的实现是用 block-sparse attention kernel (类似 FlashAttention-2/3 + causal mask + window mask union), 只对那些 $\sigma_t = 1$ 的 token 做 full-row attention。NSA (Yuan et al. 2025) 和 MoA (Fu et al. 2024) 都做了类似的事情可以参考。

---

## 4. Adaptive Regularization: 这是 paper 最巧的部分

### 4.1 朴素思路的问题

如果只给 router 加 $\mathcal{L}_{\text{reg}} = \gamma \cdot \mathcal{R}(H^{(l)})$，push router logit 往负方向走（倾向 SWA），有两个 failure mode:

1. **Hard token 全跑 SWA**: 比如 needle 在 30K 外，model 必须用 full attention 才能找到，但 router 被惩罚得不敢选 full → 性能崩
2. **两个 branch 输出分歧大时还硬走 SWA**: 即使当前 token 朴素看容易，但 full 和 SWA 输出 MSE 大，说明 SWA 漏了某些信息, 应该 fallback 到 full

### 4.2 Adaptive Weight 设计 (Eq 7-8)

$$\mathcal{L}_t^{(l)} = \gamma_t^{(l)} \cdot \text{sp}\left(\mathcal{R}(h_t^{(l)})\right) \tag{7}$$

其中 $\text{sp}(x) = \ln(1 + e^x)$ 是 **softplus**，smooth ReLU:
- 当 $\mathcal{R}(h) \gg 0$ (router 想选 full): $\text{sp} \approx \mathcal{R}(h)$, 大 penalty
- 当 $\mathcal{R}(h) \ll 0$ (router 想选 SWA): $\text{sp} \approx 0$, 小 penalty
- $\gamma_t^{(l)}$ 是 adaptive weight:

$$\gamma_t^{(l)} = \frac{\gamma_{\text{base}}}{\varepsilon + \text{NLL}(t) + \alpha \cdot \|o_{\text{FULL}}^{(t,l)} - o_{\text{SWA}}^{(t,l)}\|_2^2} \tag{8}$$

- $\gamma_{\text{base}} = 10^{-3}$: base scale
- $\varepsilon = 0.1$: 防 division by zero + 给 $\gamma$ 一个 upper bound $\gamma_{\text{base}} / \varepsilon = 10^{-2}$
- $\text{NLL}(t) = -\log p(y_t | y_{<t})$: 当前 ground-truth token 的 negative log-likelihood under 当前 model 的预测分布。**这是 model 觉得这个 token 有多难**
- $\alpha = 100$: MSE 项权重
- $\|o_{\text{FULL}}^{(t,l)} - o_{\text{SWA}}^{(t,l)}\|_2^2$: 两个 branch 在 token $t$ 第 $l$ 层输出的 squared disagreement

### 4.3 直觉解读

这个 weight 形式让我想到 **importance sampling** 和 **focal loss** 的反向版本：当 sample "重要" / "难" 时, 减少 regularization pressure; 当 sample "简单" 时, 加大 pressure 强迫 model 用便宜 branch。

- **NLL 大**: 当前 token 难预测，model 需要 full attention 的全局视野 → 减弱 reg
- **MSE 大**: 两个 branch 在这个 token 上分歧大，说明 SWA 信息不够 → 减弱 reg (走 full)
- **NLL 小 + MSE 小**: 简单 token + 两个 branch 都同意 → 加强 reg (走 SWA)

这相当于给 router 一个**"信任度信号"**: 你敢不敢走 SWA, 取决于 (1) model 本身能不能预测对 (2) 两个 branch 是否一致。

### 4.4 总 loss (Eq 9)

$$\mathcal{L} = \mathcal{L}_{\text{LM}} + \frac{1}{LT} \sum_{l=1}^{L} \sum_{t=1}^{T} \mathcal{L}_t^{(l)} \tag{9}$$

- $L = 22$: transformer 层数
- $T$: sequence length
- 正则化在每个 token 每一层都施加，平均 over $LT$
- $\mathcal{L}_{\text{LM}}$ 是标准 next-token prediction loss

### 4.5 Ablation 验证 (Table 4)

在 LongBench-E summarization 三个 dataset (GvR, MNs, SSM) 上:

| Variant | GvR | MNs | SSM | AVG |
|---|---|---|---|---|
| V1: w/o Adaptive Reg. (用固定 $\gamma_{\text{base}}$) | 24.1 | 20.7 | 30.0 | 24.9 |
| V2: w/o NLL term | 24.3 | 20.9 | 30.7 | 25.3 |
| V5: $\alpha = 0$ (w/o MSE term) | 23.9 | 20.6 | 29.5 | 24.7 |
| V3: $\alpha = 10$ | 24.1 | 20.9 | 30.5 | 25.2 |
| V4: $\alpha = 1$ | 23.8 | 20.9 | 29.9 | 24.9 |
| V6: $\varepsilon = 0.2$ | 24.7 | 20.9 | 31.7 | 25.8 |
| V7: $\varepsilon = 0.01$ | 24.4 | 20.8 | 30.1 | 25.1 |
| **SwiAttn (default)** | **25.0** | **21.3** | **32.4** | **26.2** |

去掉 adaptive reg 掉 1.3 分, 去掉 NLL 掉 0.9, 去掉 MSE 掉 1.5。说明两个 signal 都 essential。$\alpha$ 在 100 时最佳, 说明 MSE 信号需要放大权重才和 NLL 量级匹配。$\varepsilon$ 不敏感。

---

## 5. Continual Pretraining: 复用 pretrained full attention

这是工程上最实用的设计。很多 hybrid (比如 Zamba, Jamba) 要 from scratch 训练, SwiAttn 直接拿现成 full attention 模型迁移。

### 5.1 Initialization

- $W_Q, W_K, W_V, W_O$: 直接从 pretrained full attention 模型继承
- $\mathcal{A}_{\text{FULL}}^{(l)}$ branch: 复用 pretrained attention 的所有权重 (其实就是 identity copy)
- $\mathcal{A}_{\text{SWA}}^{(l)}$ branch: 也用同样的权重 (因为是共享 Q/K/V projection, 只 attention pattern 不同)
- FFN: 直接继承
- Router $W_R$: 随机初始化 (small)

### 5.2 训练 schedule

- 4K model: full attention pretrained 240B tokens → SwiAttn CPT 20B tokens (5000 steps, batch 1024)
- 32K model: full attention pretrained 192B tokens (4K) + 24B tokens (YaRN 扩到 32K) → SwiAttn CPT 8B tokens (2000 steps, batch 128)
- LR: $1 \times 10^{-4}$ (4K) / $2 \times 10^{-4}$ (32K), warmup 500 steps, cosine decay

CPT budget 远小于 pretraining (20B vs 240B, 1/12), 因为大部分知识已经在 full attention 模型里, 只需要教 router 学会路由 + 微调 attention 适应 sparse pattern。

Loss curve 在 Appendix D (Figure 7) 显示 CPT 阶段稳定无 spike, 验证 STE + adaptive reg 的优化是 well-behaved 的。

---

## 6. 实验结果

### 6.1 Commonsense Reasoning (Table 1)

context 短 (< 1024)，所以 SWA 也够，所有方法差距很小:
- FullAttn AVG acc: 53.4
- StaticHybrid: 53.6
- SwiAttn: 53.7
- SWA-CPT: 52.6
- SWA-ZS: 50.5

这里 SwiAttn 几乎无损, 因为短 context 时 model 大量选 SWA。

### 6.2 In-context Retrieval (Table 2)

context 在 4096 以内，部分 instance 需要 attention span 超过 SWA window:
- FullAttn AVG: 52.9
- **SwiAttn AVG: 52.9** (持平!)
- StaticHybrid: 50.6
- SWA-CPT: 41.5
- SWA-ZS: 41.7

SwiAttn 相对 SWA-CPT 提升 27.5%, 相对 StaticHybrid 提升 4.5%, **达到 full attention 水平**。这是关键的 "no loss" claim。

特别看 FDA dataset: FullAttn 76.4, SwiAttn 75.1, StaticHybrid 67.9, SWA-ZS 22.5。FDA 是 PDF 文档抽取, 需要 long-span attention, 这块 SwiAttn 显著胜过 static hybrid。

### 6.3 Long-context Understanding (Table 3)

context 大部分超过 SWA window, 最大 32K:
- FullAttn AVG: 37.4
- **SwiAttn AVG: 37.4** (并列最佳!)
- StaticHybrid: 35.2 (SwiAttn +6.3%)
- SWA-CPT: 31.0
- SWA-ZS: 23.4

这里 SwiAttn **超过** StaticHybrid 6.3%, 但 full attention ratio 更低 (0.13 vs 0.25)。这说明 dynamic routing 比 static pattern 更高效地分配 computation budget。GvR (24.9 → 25.0)、MNs (20.9 → 21.3)、MFQA (33.9 → 35.3, **SwiAttn 超过 full attention**) 都有提升或持平。MFQA 上 SwiAttn 反超 full attention 比较有意思, 可能是 dynamic 让模型在 multi-field document 不同位置用不同 attention 跨度更合理。

### 6.4 Needle-in-a-Haystack (Figure 3)

RULER (Hsieh et al. 2024) benchmark, 35 个 depth × 35 个 length = 1225 个 cell, 每个 cell 重复 3 次。Figure 3 显示 SwiAttn 在 32K context 全场景 perfect retrieval accuracy。这是非常强的证据表明 dynamic routing 没有损害长程检索能力。

### 6.5 Efficiency (Figure 4)

**Prefill** (Figure 4a): GFLOPs 随 token position 增长。FullAttn 是 quadratic spike 在末尾; SwiAttn 显著更低 (大约 1/3 到 1/4)。

**Decode** (Figure 4b): 用 KV cache 访问 token 数衡量 (decode 的主要 bottleneck):

| Context | SwiAttn (Standard) | FullAttn | Speedup |
|---|---|---|---|
| 8K | 2.3K | 8K | 3.5× |
| 16K | 4.2K | 16K | 3.8× |
| 24K | 6.2K | 24K | 3.9× |
| 32K | 7.3K | 32K | **4.4×** |

| Context | SwiAttn (Simple/NIAH) | FullAttn | Speedup |
|---|---|---|---|
| 16K | 1.7K | 16K | 9.4× |
| 32K | 2.6K | 32K | **12.3×** |

"Simple" 是 NIAH 任务 needle 在 window 内的场景, 此时 router 几乎全选 SWA, decode 时只读 ~1024 个 KV, 加速 12×。这是 dynamic routing 的 sweet spot。

---

## 7. 路由动态分析 (这是 paper 最有意思的部分)

### 7.1 Per-layer full attention ratio (Figure 5)

在 32K continual pretraining 数据上统计 400M tokens:

- **Layer 1, 2, 3, 22**: full attention ratio 很低 (SWA-heavy)
- **Layer 8, 11, 13, 14**: full attention ratio 高
- **平均 ratio ≈ 0.13**, 远低于 StaticHybrid 的 0.25

这让人联想到 Transformer interpretability literature 的发现: 不同层 specialize 不同功能。低层做 local pattern (n-gram, syntactic), 中层做 semantic aggregation, 高层做 task-specific reasoning。SwiAttn 的 router 自动学出了这种 specialization, 而不需要人为设计 1:3 pattern。

类似观察在 NSA (Yuan et al. 2025) 和 Native Hybrid Attention (Du et al. 2025) 里也有, 都指向同一结论: **layer-wise attention pattern 应该是 learned 不是 designed**。

### 7.2 Routing 在 NIAH 上的动态 (Figure 6)

测试 needle 在不同 depth 的位置 (低 depth = 早期插入 = 在 window 外, 高 depth = 末期插入 = 在 window 内):

| Context length | Depth 低 (window 外) | Depth 高 (window 内) |
|---|---|---|
| 1.9K | ~30% | ~5% |
| 2.8K | ~30% | ~5% |
| 3.7K | ~30% | ~5% |
| 4.6K | ~30% | ~5% |
| 30.2K | ~30% | ~5% |

- Needle 在 window 外: router 平均 30% 选 full attention, 把 budget 投到检索上
- Needle 在 window 内: 平均 5% 选 full attention, 几乎全靠 SWA

这是 **task-adaptive computation** 的直接证据。让我联想到 Speculative Decoding (Leviathan et al. 2023) 的思路: 简单部分用 cheap model, 难的部分用 expensive model。SwiAttn 把这个思想放到 attention 层面。

### 7.3 CPT 过程中的 ratio 演化 (Appendix E, Figure 8/9)

- 平均 full attention ratio 在 CPT 过程中逐渐下降到 0.13
- Per-layer: 早期所有层 ~0.5 (随机初始化的 router + sigmoid → 50%), 然后 diverge, 部分层下降到 < 0.05, 部分层稳定在 0.3 左右
- 大约 500 iterations 后稳定

这种 "渐变分化" 模式和 MoE training 中 expert specialization 的 emergent 现象很像 (Shazeer 2022)。

---

## 8. 与相关工作的横向对比

### 8.1 vs MoE / Switch Transformer

| 维度 | Switch Transformer | SwiAttn |
|---|---|---|
| Routing target | FFN experts | Attention branch (full vs SWA) |
| Granularity | per-token | per-token per-layer |
| Gate | top-1 of N, softmax | binary, sigmoid + STE |
| Regularization | load balancing loss | adaptive softplus reg |
| Inference saving | compute | compute + KV cache memory |

SwiAttn 借用了 MoE 的 conditional computation 思路, 但 attention 共享 Q/K/V projection 这个设计让 parameter overhead 几乎为零, 这对 CPT 友好。

### 8.2 vs NSA (Native Sparse Attention, Yuan et al. 2025)

NSA 也是 data-dependent sparse attention, 但是它在每层做 token selection (硬选择 attend 哪些 historical tokens), 而 SwiAttn 是在 branch 层面做选择 (是 full pattern 还是 window pattern)。NSA 的 routing 是 **intra-attention**, SwiAttn 是 **inter-attention-variant**。两者可以叠加: NSA-style block selection + SwiAttn-style full/SWA switching。

### 8.3 vs Native Hybrid Attention (Du et al. 2025)

NHA 也是 hybrid, 但它的 routing 是基于 layer-level learned pattern (整层用 full 或 linear), 没有 per-token granularity。SwiAttn 更 fine-grained。

### 8.4 vs Mamba / Linear Attention

Mamba, RWKV-7, Gated DeltaNet, Kimi Linear 都是 sub-quadratic 的 RNN-style 替代, 但它们的有限 state capacity 在 long-context retrieval 上有 hard limit (Sun et al. 2026, Nazari & Rusch 2026 指出 state rank dynamics)。SwiAttn 通过保留 full attention branch 作为 fallback, 完全避免这个问题。这是 hybrid 相对 pure sub-quadratic 的根本优势。

### 8.5 vs Hybrid Linear Attention (Wang et al. 2025)

工业界已经广泛采用 "linear attention + 偶尔 full attention" 的 pattern (Jamba, Zamba, Kimi Linear, MIMO-v2)。SwiAttn 的 framework 完全可以扩展到 linear branch: 把 SWA branch 换成 linear attention branch, 用同一套 router + adaptive reg 机制。这是一个非常自然的 follow-up 方向。

### 8.6 vs Speculative Decoding / Medusa / R2R

Speculative decoding 在 token level 做 small/large model 路由, Medusa 用 multiple head, R2R (Fu et al. 2025) 用 small-large token routing。这些都是 **token generation level** 的 routing, SwiAttn 是 **representation computation level** 的 routing, 但思想同源: **根据难度动态分配算力**。

---

## 9. 个人 Intuition 总结

这篇 paper 给我的几个 take-away:

1. **Attention pattern 应该是 learned, 不是 designed**。Static 1:3 hybrid 是工程方便的妥协, 不是最优。SwiAttn 用 router + adaptive reg 把这个 decision 数据驱动化, 实验显示平均只 13% full attention 就够, 远低于 25% static 比例, 说明 static pattern 浪费了 ~50% 的 full attention budget。

2. **Adaptive regularization 是关键创新**。固定 reg 强度会让 router 在难 token 上也偏向 SWA, 导致性能崩。把 NLL (token 难度) 和 MSE (branch 分歧) 拿来做 weight, 等于让 router 知道 "什么时候可以懒, 什么时候必须勤奋"。这个思想可以推广到任何 conditional computation 场景。

3. **STE + shared Q/K/V + CPT 是工程上漂亮的组合**。STE 解决 gradient 问题, shared projection 解决 parameter overhead 和 CPT initialization 问题, CPT 解决 from-scratch cost 问题。三件小事组合起来让整个方案 plug-and-play。

4. **Per-layer specialization 涌现**。Figure 5 显示 router 自动学出 layer 1/2/3/22 偏 SWA, layer 8/11/14 偏 full, 这和 Transformer interpretability 的层级功能分化对应。这说明 hybrid pattern 本质上是和 representation learning 耦合的, 无法 pre-design。

5. **Limitations 是真实的**: prefill 阶段两个 branch 都要算, 实际 speedup 主要在 decode; 没在 > 1.5B 模型验证; router 是 simple linear, 可能更复杂的 router (e.g., small MLP + LayerNorm) 更好。这些都给 follow-up 留了空间。

整体上, 这篇 paper 把 hybrid transformer 的"hybridity"从 static architecture 推到 dynamic computation, 是一个 elegant 的中间路线: 比 static hybrid expressive, 比 pure MoE-attention 轻量, 比 NSA-style 内部 sparse 简单。如果 kernel 工程补上 (prefill 阶段 block-sparse fusion), 这条路线在工业界应该会很有 traction。

---

## 10. 可能的 follow-up 方向 (个人推测)

- **Router 升级**: 现在是 linear router, 可以试 small MLP + LayerNorm, 或者用上一层的 hidden state 做 cross-layer routing (类似 Mamba 的 recurrent decision)
- **Branch 扩展**: 加入 linear attention branch (Mamba-style), 变成 3-way routing, 进一步压缩简单 token 成本
- **Hierarchical routing**: layer-level router + token-level router 两级, 处理 layer specialization 的 emergent pattern
- **Prefill kernel**: 写一个 fused kernel, 类似 FlashAttention 但 mask 是 `(window_mask) OR (token in full_set)`, 类似 MoA (Fu et al. 2024) 或 HySparse (Gao et al. 2026) 的实现
- **Scale up**: 7B, 13B, 70B 上验证, router 的 specialization 是否随 scale 改变
- **Training from scratch**: 比较 CPT vs from-scratch SwiAttn, 看是否 CPT 路径限制了 router 的探索
- **Multi-modal**: vision Transformer 的 dynamic window vs global attention, 视觉 token 难度更异质 (background vs object), 应该收益更大

---

## 11. Reference Links

- Switch Transformer (MoE foundation): https://arxiv.org/abs/2101.03961
- Longformer (SWA 始祖): https://arxiv.org/abs/2004.05150
- BigBird (sparse attention): https://arxiv.org/abs/2007.14062
- Mistral 7B (1:3 static hybrid 工业典范): https://arxiv.org/abs/2310.06825
- Gemma 2: https://arxiv.org/abs/2408.00118
- Qwen3 technical report: https://arxiv.org/abs/2505.09388
- GPT-oss (OpenAI 2025): https://arxiv.org/abs/2508.10925
- FlashAttention: https://arxiv.org/abs/2205.14135
- FlashAttention-3: https://arxiv.org/abs/2407.08608
- GQA: https://arxiv.org/abs/2305.13245
- MLA (DeepSeek-V2): https://arxiv.org/abs/2405.04434
- Mamba: https://arxiv.org/abs/2312.00752
- Mamba-2 (SSD): https://arxiv.org/abs/2405.21060
- RWKV-7: https://arxiv.org/abs/2503.14456
- Gated Delta Networks: https://arxiv.org/abs/2412.06464
- Native Sparse Attention (NSA): https://arxiv.org/abs/2502.11089
- Native Hybrid Attention (NHA): https://arxiv.org/abs/2510.07019
- HySparse: https://arxiv.org/abs/2602.03560
- Mixture of Attention Spans (MoA): https://arxiv.org/abs/2406.14909
- YaRN (context extension): https://arxiv.org/abs/2309.00071
- STE (Bengio 2013): https://arxiv.org/abs/1308.3432
- Switch Transformer STE 应用: https://arxiv.org/abs/2101.03961
- Speculative Decoding: https://arxiv.org/abs/2211.17192
- Medusa: https://arxiv.org/abs/2401.10774
- R2R (small-large routing): https://arxiv.org/abs/2505.21600
- RULER (NIAH benchmark): https://arxiv.org/abs/2404.06654
- LongBench: https://arxiv.org/abs/2308.14508
- H2O (KV eviction): https://arxiv.org/abs/2306.14048
- StreamingLLM (attention sink): https://arxiv.org/abs/2309.17453
- Composer (hybrid NAS): https://arxiv.org/abs/2510.00379
- Hybrid Linear Attention (systematic analysis): https://arxiv.org/abs/2507.06457
- State Rank Dynamics in Linear Attention: https://arxiv.org/abs/2602.02195
- Kimi Linear: https://arxiv.org/abs/2510.26692
- MIMO-v2-Flash: https://arxiv.org/abs/2601.02780

希望这些拆解帮你 build 起对 dynamic hybrid attention 这条路线的 intuition。整个 story 的核心 tension 是 **static architecture 无法 match heterogeneous token difficulty**, SwiAttn 用 per-token per-layer routing + adaptive regularization 把这个 mismatch 解决掉, 同时通过 CPT 保持迁移友好。我个人觉得这个方向会持续演化, 尤其是和 linear attention branch 组合之后。
