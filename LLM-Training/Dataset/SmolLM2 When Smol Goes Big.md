---
source_pdf: SmolLM2 When Smol Goes Big.pdf
paper_sha256: a95fbda201d21cd25c89e949636373d36c288bd2d287e3cabd208e2041ea36d3
processed_at: '2026-08-12T07:57:30-07:00'
target_folder: LLM-Training/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SmolLM2 用人话讲

Andrej，咱们这次换种讲法——我把它当一个 story 讲给你听，把那些 paper 里藏着的"为什么这么干"的 intuition 全挖出来，同时保留公式和数据的硬核部分。

---

## 一、为什么这篇 paper 值得你花 30 分钟

小模型这事儿现在是真热，原因就两条：
1. **Inference 是真贵**，超大模型连 Google 都想往边缘推
2. **Overtraining 已经是 industry consensus**——Qwen2.5-1.5B 跑 18T tokens，Llama3.2-1B 跑 9T tokens，Chinchilla 那套 20 tokens/param 早就没人看了

SmolLM2 的故事本质上就一句话：**1.7B 参数的小模型，喂 11T tokens，用 online re-balancing + 三个自己造的 specialized 数据集，把性能推到 Qwen2.5-1.5B / Llama3.2-1B 同级，甚至部分 benchmark 超越**。

听起来简单，里面的工程决策全都是被 \$250K compute budget 逼出来的。这种"在预算约束下做最优决策"的 paper 才有意思——不是 "we scaled to 100B params and it works"，而是 "we have \$250K, how do we spend it"。

---

## 二、整篇 paper 的 mental model

我把它拆成 4 层来理解：

```
Layer 1:  Data-Centric Philosophy
         小模型对 noise 敏感 → 每个 token 都得"值得学"
         
Layer 2:  Online Re-balancing
         跑 → 评估 → 调配比 → 再跑
         (因为 \$250K 不允许 you do many full runs)
         
Layer 3:  WSD Scheduler
         Warmup → Stable → Decay
         (Stable 可以无限延伸，随时决定 decay)
         
Layer 4:  Stage-specific Data Mix
         Stage 1-3: Stable phase, 慢慢引入 specialized data
         Stage 4:   Decay phase, 注入最高质量 data
```

这四层叠起来才是 SmolLM2 的核心方法论。下面一层一层讲。

---

## 三、Online Re-balancing：为什么不能一次跑完

这个问题 Karpathy 你肯定懂，但 paper 里讲得不够直白，我替它讲清楚：

**残酷现实**：SmolLM2-1.7B 训练一次 = 256 H100 × 数周 ≈ **1e23 FLOPs ≈ \$250K USD**。

这意味着什么？意味着你**没法跑 Chinchilla 那种几十次 ablation 决定配方**。你只有 1-2 次完整训练的机会，中间必须做 "online" 调整。

所以作者搞了个 4-stage pipeline：

| Stage | Tokens | 何时切 | 关键操作 |
|-------|--------|--------|---------|
| Stage 1 | 0–6T | 启动 | 用前期 ablation 的 baseline mixture |
| Stage 2 | 6–8T | 评估发现 code/math 弱 | 把 code 从 10% → 20%, 加 5% OWM |
| Stage 3 | 8–10T | math 还是弱 + MMLU MCF 涌现 | 翻转 web 比例到 40/60, 用 Stack-Edu 替 StarCoderData |
| Stage 4 | 10–11T | 准备 annealing | 注入 FineMath-4+ / InfiWebMath-3+ / AugGSM8K |

**Intuition**：你盯着 eval 看，哪里弱就往哪里偏。这是一种 "data curriculum by human-in-the-loop" 的思路，跟 OLMo 2 / MiniCPM 都是一脉相承的。

参考：
- https://arxiv.org/abs/2406.03476 (Blakeney et al. domain upsampling)
- https://allenai.org/blog/olmo2 (OLMo 2)

---

## 四、WSD Scheduler：这篇 paper 的隐形 MVP

公式长这样：

$$\text{LR}(t) = \begin{cases}
\alpha \cdot \frac{t}{T_w} & t < T_w \quad \text{(warmup)} \\
\alpha & T_w \le t < T_s \quad \text{(stable)} \\
\alpha \cdot \left(1 - \frac{t - T_s}{T_d}\right) & T_s \le t \le T_s + T_d \quad \text{(decay)}
\end{cases}$$

变量含义：
- $\alpha = 5.0 \times 10^{-4}$：peak learning rate
- $T_w = 2000$ steps：warmup step 数
- $T_s$：stable phase 结束位置
- $T_d = 0.1 \times (T_s + T_d)$：decay step 数（总 step 的 10%）

**为什么这玩意儿关键**？

Cosine schedule 必须先知道总训练长度。WSD 让 stable phase 可以**无限延伸**，你看着 eval 决定 "差不多了，开始 decay 吧"。

对 \$250K 的训练来说这是 game-changer——你不用提前 commit "我要训 11T tokens"，可以 "我看到 MMLU plateau 了，math/code 还有 headroom，那我在 10T 处触发 decay，剩下的 1T 专门喂高质量 specialized data"。

这就是 paper 里 Stage 4 的核心逻辑。**Decay phase = 把稀缺的高质量数据在这个窗口发挥最大作用**。

Hägele et al. 2024 的 paper 证明 decay phase 在 loss surface 上相当于"沿当前 basin 滑向最低点"，低 learning rate 下高 signal-to-noise 数据的影响被放大。

参考：https://arxiv.org/abs/2405.18392

---

## 五、三个新数据集：SmolLM2 的真正 IP

这篇 paper 最有价值的部分其实是三个数据集，不是模型本身。因为模型是 "用这些数据训出来的"，而数据集是可复用的资产。

### 5.1 FineMath：web math 数据为什么不够用

现有 math 数据集的问题：

| 数据集 | Tokens | 问题 |
|--------|--------|------|
| OpenWebMath (OWM) | 12B | 太小，5 epochs 就过拟合 |
| InfiMM-WebMath | 40B | 内容偏 academic paper，缺 step-by-step |

作者发现这两个数据集 annealing 60B tokens 后 GSM8K 才到 10-14%，远低于 DeepSeekMath 这类 SOTA 小模型。

**FineMath 的 pipeline**：

```
FineWeb 5.8B URLs
  → Resiliparse 提取
  → Llama-3.1-70B-Instruct 3-scale 标注 (1=有 math, 3=step-by-step)
  → 训练 classifier，筛 domain
  → 合并 OWM/InfiMM-WebMath domain URLs（共 7.7B URLs）
  → OWM pipeline 重新提取（保留 LaTeX）
  → 5-scale 二次过滤（4-5=教材级，3-5=教程级）
  → MinHash LSH dedup
  → fastText 语言过滤
  → 13-gram decontamination（vs GSM8K/MATH/MMLU, LCS overlap ≥0.6）
```

**关键 ablation 结果**（Figure 1）：

| Dataset | GSM8K | MATH |
|---------|-------|------|
| OWM | 10% | ~6% |
| InfiMM-WebMath | 14% | ~5% |
| FineMath4+ | **~30%** | **~30%** |

FineMath4+ 相对 InfiMM-WebMath：**GSM8K 2× 提升，MATH 6× 提升**。这就是 paper 里那句"retaining high-quality mathematical content with reasoning"的分量。

**Intuition**：math reasoning 的核心是 chain-of-thought 形式的 step-by-step 解题。web 上 99% 的 math 内容是 "calculator output" / "公式表" / "学术 paper abstract"，这些对 small model 的 reasoning capability 没贡献。Llama-3.1-70B-Instruct 当 judge 筛 score 4-5 的样本，相当于用大模型的 reasoning capability 来 distill 出 "what good math reasoning looks like" 给小模型当 training data。

参考：https://arxiv.org/abs/2310.06786 (OWM), https://arxiv.org/abs/2409.12568 (InfiMM-WebMath)

### 5.2 Stack-Edu：把 FineWeb-Edu 思路搬到代码

代码数据有个被忽视的问题：**直接堆 raw GitHub 对小模型不友好**。GitHub 上大量是 boilerplate（package.json, config files, 自动生成代码），这些对 reasoning 学习没贡献，还会"挤占"小模型的 capacity。

Stack-Edu pipeline：

```
StarCoder2Data (top 15 languages, ~450B tokens)
  → Llama3-70B-Instruct 5-scale 标注 (0-5)
  → per-language StarEncoder classifier (F1 > 0.7)
  → threshold 3（Java 用 2，因为数据更稀疏）
  → ~125B tokens
```

**MultiPL-E 提升对比**：

| Language | Original | Stack-Edu filtered |
|----------|----------|---------------------|
| Python | 20.7 | **25.6** |
| C++ | 16.7 | **24.8** |
| JavaScript | 18.2 | **22.4** |
| Java | 17.6 | **22.7** |

平均 ~5 个百分点提升，几乎免费（classifier filtering 成本远低于训练）。

**Intuition**：小模型学代码的核心是 "well-structured code with educational comments"，不是 "production-grade minified JS"。这个观察对任何训小模型的团队都适用。

参考：https://arxiv.org/abs/2409.02326 (Arctic-Snowcoder)

### 5.3 SmolTalk：SFT 数据集的"组合拳"

SmolTalk 是 1.1M pairs 的组合：

| 组件 | 样本数 | 用途 |
|------|--------|------|
| MagPie-Ultra | 431k | 通用对话（Llama-3.1-405B-FP8 生成） |
| Smol-Rewrite | 56.2k | 文本重写 |
| Smol-Constraints | 36.2k | IFEval 风格约束 |
| Smol-Summarization | 101k | 摘要 |
| NuminaMath-CoT | 112k | 数学推理 |
| MetaMathQA | 50k | GSM8K 增强 |
| Self-OSS-Starcoder2-Instruct | 50.7k | 代码 |
| APIGen-Function-Calling | 87.5k | 函数调用 |
| 其他 | ~200k | 长上下文 / system prompt / 知识等 |

**最有意思的是 Smol-Constraints**：

paper 里说，他们先用 MagPie 方法生成 550k instructions → 过滤冲突约束 / 错误响应 → 56.3k → 10-gram decontamination vs IFEval → 最终 36k。

这 36k 是关键——它让 SmolLM2-Instruct 在 IFEval 上达到 56.7，吊打 Qwen2.5-1.5B-Instruct 的 47.4。

**Intuition**：instruction following 的核心不是"对话能力"，是"约束遵守能力"。模型得学会 "如果 prompt 说 '以 JSON 格式输出'，就真的输出 JSON"。这种能力需要专门的 constraint-rich 训练数据，MagPie-Ultra 的自由对话数据不够。

参考：https://arxiv.org/abs/2406.08464 (MagPie), https://arxiv.org/abs/2311.07911 (IFEval)

---

## 六、4 个训练 Stage 的 Story

这个是最有 teaching value 的部分，我详细讲。

### Stage 1 (0–6T tokens)：建立 baseline

数据配比：
- FineWeb-Edu + DCLM (60/40): 90%
- StarCoder-Data: 10%

**为什么 60/40**？Table 1 的 ablation 显示：
- FineWeb-Edu 在 MMLU / ARC / OpenBookQA 强（educational content）
- DCLM 在 HellaSwag / CommonsenseQA 强（conversational style）
- 60/40 mix 在两边都接近最优

**为什么 code 只给 10%**？因为 11T 总训练量下，10% × 11T = 1.1T tokens，而 StarCoder-Data 总共 250B tokens，相当于 ~4 epochs。这符合 Muennighoff et al. 2023 的 "4-5 epoch repetition threshold"。

**为什么完全没 math**？因为 FineMath 才 10-34B tokens，在 90% web 数据里被稀释到看不见。留给后期。

Stage 1 结束时：MMLU 29.62 (MCF), math 3.21, code 8.87。Knowledge/reasoning 正常，math/code 灾难。

### Stage 2 (6–8T tokens)：第一次干预

数据配比：
- Web (60/40): 75%
- Code: 20%（↑10%）
- OWM: 5%（新增）

**为什么加 OWM**？因为 Stage 1 后 math 太弱，但 OWM 只有 12B tokens，全量加会被秒过 4-epoch 阈值。所以只给 5%，希望"少量多次"。

**结果**：code 提升明显（10.56），math 几乎不变（3.7）。**OWM 没起作用**。

**Bonus discovery**：MMLU MCF 在 6T 后开始超过 25% random baseline。这跟 Gu et al. 2024 / Du et al. 2024 "小模型不能做 MCF" 的论断矛盾。可能解释：长训练让小模型在 distribution 上更靠近 instruction-like text。

### Stage 3 (8–10T tokens)：第二次干预

数据配比：
- Web (40/60): ~66%（**比例翻转**）
- Stack-Edu: ~20%（替代 StarCoderData）
- Math (OWM + InfiMM-WebMath-text): ~10%（↑5%）
- Jupyter Notebooks: 微量新增

**为什么翻转 web 比例**？因为额外 ablation 发现 DCLM 比例高能小幅提升 MMLU MCF。

**为什么换 Stack-Edu**？因为 Stage 2 末 code 表现还不够强，Stack-Edu 的 educational filtering 比 raw StarCoderData signal density 更高。

**结果**：math 终于动起来（7.27），code 继续涨（16.75）。

**Loss spike 谜团**：这个阶段出现了 unexplained loss spike，回退 + 跳过相关数据后仍存在。paper 没深究，但恢复说明模型 capacity 没被破坏。这类 spike 在 PaLM / Chinchilla 训练日志里都见过，可能与 optimizer state 二阶矩估计 + mixture distribution shift 相关。

### Stage 4 (10–11T tokens, Decay)：annealing

数据配比：
- Web (40/60): 58%
- Stack-Edu: 24%（扩展更多语言）
- Math (InfiWebMath-3+ + FineMath-4+ + AugGSM8K): 14%
- Cosmopedia v2: 4%

**结果**：math 从 7.27 → 22.07，code 16.75 → 23.21。

这个 3× math 提升就是 annealing 阶段注入高质量 specialized data 的力量。

### Context Length Extension

取 Stage 4 中间 checkpoint（最后 75B tokens 之前）：
- Seq len: 2k → 8k
- RoPE $\theta$: 10k → 130k
- Mixture: 40% long-context docs + 60% Stage 4 mix

最终得到 SmolLM2-1.7B base model。

---

## 七、架构细节

Table 6 的 Llama2 架构：

| 参数 | 值 | 备注 |
|------|-----|------|
| Layers $L$ | 24 | |
| $d_{model}$ | 2,048 | |
| $d_{ff}$ | 8,192 | SwiGLU 实际 ratio ≈ 2.67 |
| Heads $H$ | 32 | head dim = 64 |
| Vocab $V$ | 49,152 | tokenizer 在 math/code 上 retrain |
| RoPE $\theta$ | 10,000 | 长上下文阶段 → 130k |
| Tied embedding | Yes | 节省参数 |
| Activation | SwiGLU | |

**SwiGLU 公式**：

$$\text{FFN}(x) = (\text{Swish}(W_{\text{gate}} x) \odot W_{\text{up}} x) W_{\text{down}}$$

其中 $\text{Swish}(z) = z \cdot \sigma(z)$，$\sigma$ 是 sigmoid。

**RoPE**：对于位置 $m$、维度对 $i \in [0, d/2)$：

$$\text{Rot}(m, 2i) = \cos(m \theta^{-2i/d}), \quad \text{Rot}(m, 2i+1) = \sin(m \theta^{-2i/d})$$

$\theta = 10{,}000$ 是 default base frequency。长上下文拉到 $1.3 \times 10^5$ 让模型在 8k context 内位置外推更好。

**为什么 1.7B 没用 GQA**？可能因为 quality 优先。135M / 360M 用 GQA 减少 KV cache，方便 edge deployment。

参考：
- https://arxiv.org/abs/2106.09685 (RoPE)
- https://arxiv.org/abs/2305.13245 (GQA)

---

## 八、Post-Training 细节

### SFT

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t})$$

- $x$：prompt
- $y$：target response
- $y_t$：第 $t$ 个 response token
- $\pi_\theta$：模型

超参：2 epochs, batch 128, seq 8192, LR $3.0 \times 10^{-4}$

### DPO

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

变量含义：
- $x$：prompt
- $y_w$：偏好 response（winner）
- $y_l$：不偏好 response（loser）
- $\pi_\theta$：当前策略
- $\pi_{\text{ref}}$：参考策略（SFT checkpoint，冻结）
- $\beta = 0.5$：KL penalty 系数
- $\sigma$：sigmoid

超参：2 epochs, LR $1.0 \times 10^{-6}$, batch 128, seq 1024, UltraFeedback 数据集。

**关于 $\beta = 0.5$**：标准 DPO 用 $\beta = 0.1$。SmolLM2 用 0.5，意味着 implicit reward signal 更强，允许策略偏离 ref 更多。对小模型来说，需要更激进的 preference 信号产生明显行为变化。

参考：https://arxiv.org/abs/2305.18290 (DPO)

---

## 九、评估亮点

### Base Model（Table 4）

| Benchmark | SmolLM2-1.7B | Llama3.2-1B | Qwen2.5-1.5B |
|-----------|-------------|-------------|-------------|
| HellaSwag | **68.7** | 61.2 | 66.4 |
| ARC | **60.5** | 49.2 | 58.5 |
| MMLU-Pro | **19.4** | 11.7 | 13.7 |
| TriviaQA | **36.7** | 28.1 | 20.9 |
| GSM8K | 31.1 | 7.6 | **61.7** |
| MATH | 11.6 | 3.3 | **34.3** |
| HumanEval | 22.6 | 18.9 | **37.2** |

SmolLM2 在 knowledge/reasoning 上明显领先，math/code 落后 Qwen2.5-1.5B（Qwen 用 18T tokens + 强 math/code pipeline）。

### Instruct Model（Table 5）

| Benchmark | SmolLM2-Instruct | Llama3.2-1B-Inst | Qwen2.5-1.5B-Inst |
|-----------|-----------------|------------------|-------------------|
| IFEval | **56.7** | 53.5 | 47.4 |
| MT-Bench | 6.13 | 5.48 | **6.52** |
| GSM8K | 48.8 | 37.4 | **63.3** |
| MATH | **21.0** | 19.5 | 19.6 |
| HumanEval | 28.1 | 33.5 | **30.5** |

SmolLM2-Instruct 在 IFEval 上大幅领先，这是 Smol-Constraints 数据的功劳。

---

## 十、几个值得深挖的点

### 10.1 Chinchilla 比例彻底过时

SmolLM2 用 11T / 1.7B ≈ **6470 tokens/parameter**，约 320× Chinchilla optimal。

Qwen2.5-1.5B 用 18T tokens，Llama3.2-1B 用 9T tokens。所有 industry 小模型都在 overtraining。

**Why**：训练 compute 便宜，inference 贵。多花 training compute 换 inference 便宜，部署成本算下来划算。de Vries 2023 的 "Go smol or go home" 讲的就是这个 tradeoff。

参考：https://www.harmdevries.com/post/model-size-vs-compute-overhead/

### 10.2 小模型为什么单阶段训练更好

135M / 360M 用 single-stage training，1.7B 用 4-stage。

**Intuition**：小模型 capacity 太小，多阶段 mixture shift 反而让模型 confused。1.7B 有足够 capacity "记住"不同阶段的 distribution，135M 没有。

这是个有意思的 scaling 现象——多阶段 curriculum 是大模型才能享受的 luxury。

### 10.3 LLM-as-judge 的核心 IP

三个数据集（FineMath / Stack-Edu / SmolTalk）的共同点：用 Llama-3.1-70B/405B 当 judge，把稀有的高质量样本从海量噪声里筛出来。

**这是 SmolLM2 真正的 IP**。Appendix C.2 / C.3 / D.1 的 annotation prompt 写得非常具体（"Add 1 point if..."），这些 rubric 是数据质量的灵魂。

### 10.4 4-5 Epoch 重复阈值

Muennighoff et al. 2023 的 scaling data-constrained LM 工作建议避免重复超过 4-5 epochs。SmolLM2 全程遵循这个阈值——StarCoder-Data 10% 配比就是为了让 11T total 下保留 ~4 epochs headroom。

但 FineMath4+ 在 annealing 60B tokens（~6 epochs）后没 plateau，InfiWebMath4+ 在 80B（~10 epochs）后 plateau。说明数据质量高时可以承受更多 epochs，但仍有上限。

参考：https://arxiv.org/abs/2305.16264

---

## 十一、资源链接

- 模型 collection: https://hf.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9
- FineMath: https://huggingface.co/datasets/HuggingFaceTB/FineMath
- Stack-Edu: https://huggingface.co/datasets/HuggingFaceTB/Stack-Edu
- SmolTalk: https://huggingface.co/datasets/HuggingFaceTB/smoltalk
- Nanotron: https://github.com/huggingface/nanotron
- Datatrove: https://github.com/huggingface/datatrove
- LightEval: https://github.com/huggingface/lighteval

---

## 十二、整体 Take

**SmolLM2 的核心 thesis**：小模型的瓶颈在 data density，不在架构。1.7B 参数要塞下 world knowledge + reasoning + code + math + instruction following，每个 token 都得"值得学"。

三个新数据集的共同点是 **用大模型当 judge 把稀有高质量样本从海量噪声里筛出来**，然后通过 multi-stage 在线 re-balancing 让这些数据在 annealing 阶段发挥最大效用。

这跟 Phi-3/4 的 "textbooks are all you need" 路线相似，但 SmolLM2 用 web-crawl + classifier filtering（保留真实分布），避免 Phi 系列在某些 held-out benchmark 上的 overfitting 嫌疑。

**下一步可能的方向**：
1. 更激进的 synthetic reasoning data（类似 Orca / WizardMath）
2. Self-distillation / iterative refinement（类似 DeepSeekMath-R1）
3. 多语言扩展（SmolLM2 目前以英语为主）
4. Long-context 阶段更长训练（目前只有 75B tokens 在 8k context 上）

如果你有兴趣深挖，重点看 Appendix C.2 / C.3 / D.1 的 annotation prompt——这些 rubric 是整篇 paper 数据质量的灵魂，可复用的核心 IP。

---

# SmolLM2 深度技术解读

Andrej，这篇 paper 是 Hugging Face TB 团队的 data-centric 小模型工作，核心 takeaway 是**在 $250K compute budget 下，通过 4 阶段在线数据再平衡 + 3 个新数据集，把 1.7B 模型推到 Qwen2.5-1.5B / Llama3.2-1B 同级或更强的水平**。下面我把关键的技术 intuition 拆开讲。

---

## 1. 核心方法论：为什么是 "online" 调配比

完整训练一次 SmolLM2 大约消耗 1e23 FLOPs ≈ \$250K USD GPU compute。这个成本下，没法像 Chinchilla 论文那样跑 dozens of ablations 再决定配方，所以作者采用了**"train → eval → re-balance" 的闭环**：

| 阶段 | 触发条件 | 操作 |
|------|---------|------|
| Stage 1 (0–6T) | 启动 | 用 ablation 结果定 baseline mixture |
| Stage 2 (6–8T) | 看到 code/math 落后 | 加 5% OWM + 把 code 从 10%→20% |
| Stage 3 (8–10T) | math 仍弱 + MMLU MCF 涌现 | 加 InfiMM-WebMath (text-only) + Stack-Edu 替代 StarCoderData + 翻转 FW-Edu/DCLM 比例到 40/60 |
| Stage 4 (10–11T, decay) | annealing window | 注入最高质量数据：FineMath-4+ / InfiWebMath-3+ / AugGSM8K |

**Intuition**：这套方法学和 Blakeney et al. 的 "Does your data spark joy?" 一脉相承——把稀缺的高质量数据放到 annealing 阶段，由于 LR 衰减时模型相当于在低 noise 信号上做 fine-grained 收尾，高熵 token 在此时比 stable phase 影响大得多。这与 MiniCPM、OLMo 2 的观察一致。

参考：
- https://arxiv.org/abs/2406.03476 (Blakeney et al. domain upsampling)
- https://arxiv.org/abs/2405.18392 (Hägele et al. WSD scheduler)

---

## 2. 架构细节

SmolLM2-1.7B 用的是 Llama2 架构（tied embedding）：

| 参数 | 值 | 备注 |
|------|-----|------|
| Layers $L$ | 24 | |
| Model dim $d_{model}$ | 2,048 | |
| FFN dim $d_{ff}$ | 8,192 | ratio = 4× (SwiGLU 实际有效比 ≈ 2/3 · 4 ≈ 2.67) |
| Heads $H$ | 32 | head dim = 64 |
| Vocab $V$ | 49,152 | 比 Llama 的 32k 大，因为 tokenizer 在 math/code 上 retrain |
| Positional emb | RoPE, $\theta=10{,}000$ | 长上下文阶段改为 130k |
| Activation | SwiGLU | $\text{SwiGLU}(x) = \text{Swish}(W_1 x) \odot (W_2 x)$ |
| Tied embedding | Yes | 节省参数，input/output embedding 共享 |

**SwiGLU 公式**：

$$\text{FFN}(x) = (\text{Swish}(W_{\text{gate}} x) \odot W_{\text{up}} x) W_{\text{down}}$$

其中 $\text{Swish}(z) = z \cdot \sigma(z) = z \cdot \frac{1}{1+e^{-z}}$，$\odot$ 是 element-wise 乘。相比 ReLU FFN，SwiGLU 让梯度流更平滑，对 small model 尤其重要。

**RoPE**：对于 head 维度 $d$、位置 $m$、维度对索引 $i \in [0, d/2)$：

$$\text{Rot}(m, 2i) = \cos(m \theta^{-2i/d}), \quad \text{Rot}(m, 2i+1) = \sin(m \theta^{-2i/d})$$

$\theta = 10{,}000$ 是 default base frequency。长上下文阶段拉到 $1.3 \times 10^5$ 让模型在 8k context 内有更好的位置外推。

**对比 360M / 135M 小模型**：使用 GQA（Grouped Query Attention），即 $K$/$V$ head 数 $< Q$ head 数，减少 KV cache。1.7B 没用 GQA 可能是为了 quality 优先。

参考：https://arxiv.org/abs/2106.09685 (RoPE), https://arxiv.org/abs/2305.13245 (GQA)

---

## 3. 学习率调度：WSD

Warmup-Stable-Decay 是这篇 paper 的关键工程选择：

$$\text{LR}(t) = \begin{cases}
\alpha \cdot \frac{t}{T_w} & t < T_w \\
\alpha & T_w \le t < T_s \\
\alpha \cdot \left(1 - \frac{t - T_s}{T_d}\right) & T_s \le t \le T_s + T_d
\end{cases}$$

变量含义：
- $\alpha = 5.0 \times 10^{-4}$：peak learning rate
- $T_w = 2000$ steps：warmup 步数
- $T_s$：stable phase 结束位置（paper 中是 10T tokens 对应的 step）
- $T_d = 0.1 \cdot (T_s + T_d)$：decay 步数 = 总步数 10%

**Intuition**：cosine schedule 必须先知道总训练长度。WSD 让 stable phase 可以无限延伸，直到你看了 eval 觉得"差不多了"再触发 decay。这对 \$250K 的训练是 game-changer——你不用提前 commit 总 token 数，可以"按观察决定何时收尾"。

paper 在 10T tokens 时触发 decay，因为观察到 MMLU MCF 已经 plateau，但 math/code 还有明显 headroom，所以 decay 阶段（1T tokens）专门用来喂高质量 math/code。

---

## 4. 数据集：三个新构建的数据集

### 4.1 FineMath

现有 math 数据集的两个问题：
1. **太小**：OWM 仅 12B tokens，InfiMM-WebMath 40B tokens
2. **内容偏 academic paper**，缺少 step-by-step reasoning

FineMath 构建流水线：

```
FineWeb 5.8B URLs 
  → Resiliparse extract
  → Llama-3.1-70B-Instruct 3-scale 标注（score 1=有 math 内容, 3=step-by-step 解答）
  → 训练 classifier, 筛选 domain (≥10 pages with score≥2)
  → 合并 OWM/InfiMM-WebMath 的 domain URLs (7.7B total)
  → OWM pipeline 重新提取（保留 LaTeX）→ 7.1B pages, 6.5T tokens
  → 5-scale 二次过滤（score 4-5 教材级 / 3-5 教程级）
  → MinHash LSH (10 hashes) dedup
  → fastText 语言过滤 → 英语
  → 13-gram 去污染（vs GSM8K/MATH/MMLU, LCS overlap ratio ≥0.6）
```

最终变体：
- **FineMath4+**：10B tokens, 6.7M docs（只保留 score 4-5）
- **FineMath3+**：34B tokens, 21.4M docs（score 3-5）
- **InfiWebMath4+ / 3+**：把同一 5-scale classifier 应用到 InfiMM-WebMath

**Annealing ablation 结果**（Figure 1）：
- FineMath4+ 在 GSM8K 上达到 **2× improvement** vs InfiMM-WebMath
- MATH 上达到 **6× improvement**
- FineMath4+ 在 60B tokens annealing 后没有 plateau，而 InfiWebMath4+ 在 80B（~10 epochs）后 plateau——说明 FineMath 数据量足够避免重复

### 4.2 Stack-Edu

把 FineWeb-Edu 的 classifier-based filtering 思路用到代码上：

```
StarCoder2Data (top 15 languages, ~450B tokens)
  → Llama3-70B-Instruct 5-scale 标注 (0-5)
  → per-language StarEncoder classifier (500k samples, F1 > 0.7)
  → threshold 3 (Java 用 2)
  → ~125B tokens across 15 languages
```

15 语言中 top 4 的 MultiPL-E 提升明显：

| Language | StarCoder2Data | Stack-Edu | MultiPL-E (orig→filt) |
|----------|---------------|-----------|----------------------|
| Python | 50.6B | 21.8B | 20.7 → **25.6** |
| C++ | 69.7B | 16.0B | 16.7 → **24.8** |
| JavaScript | 45.3B | 11.1B | 18.2 → **22.4** |
| Java | 45.6B | 42.1B | 17.6 → **22.7** |

**Intuition**：直接堆 raw GitHub 代码对小模型不利——大量 boilerplate（package.json, config, 自动生成文件）会污染容量。教育性过滤把信号密度提了一大截。

### 4.3 SmolTalk

SFT 数据集，总 1.1M pairs：

| 组件 | 样本数 | 用途 |
|------|-------|------|
| MagPie-Ultra (Llama-3.1-405B-FP8 生成) | 431k | 通用对话 |
| Smol-Rewrite | 56.2k | 文本重写 |
| Smol-Constraints | 36.2k | IFEval 风格约束 |
| Smol-Summarization | 101k | 摘要 |
| NuminaMath-CoT | 112k | 数学推理 |
| MetaMathQA | 50k | GSM8K 增强 |
| Self-OSS-Starcoder2-Instruct | 50.7k | 代码 |
| APIGen-Function-Calling | 87.5k | 函数调用 |
| SystemChats2.0 | 35.9k | system prompt |
| LongAlign | 3.73k | 长上下文 |
| OpenHermes2.5 | 100k | 知识 |
| 其他 | ~70k | everyday/explore |

**MagPie-Ultra** 用 Llama-3.1-405B-Instruct-FP8 + 系统 prompt 生成三轮对话，再用 Llama-3.1-8B / Llama-Guard-3-8B 过滤质量与安全，ArmoRM 评分过滤，gte-large-en-v1.5 语义去重。

**Smol-Constraints** 流程很有意思：先用 MagPie 方法生成 550k instructions → 过滤冲突约束/错误响应 → 56.3k → 10-gram decontamination vs IFEval → 36k。

参考：https://arxiv.org/abs/2406.08464 (MagPie), https://arxiv.org/abs/2310.06786 (OWM), https://arxiv.org/abs/2409.12568 (InfiMM-WebMath)

---

## 5. 训练阶段详解

### Stage 1 (0–6T tokens)

| Component | Ratio | Rationale |
|-----------|-------|-----------|
| FineWeb-Edu + DCLM (60/40) | 90% | Ablation 确认的 web mix |
| StarCoder-Data | 10% | 限制到 10% 以保留 4 epochs 余量 |

不含 math，因为 FineMath 等数据集太小，早期加入会被稀释。

**Findings**：knowledge/reasoning 表现符合预期，但 code/math 表现差（Table 3 中 math 仅 3.21, code 仅 8.87）。

### Stage 2 (6–8T tokens)

| Component | Ratio | 变化 |
|-----------|-------|------|
| Web (60/40) | 75% | ↓15% |
| Code | 20% | ↑10% |
| OWM | 5% | 新增 |

**Findings**：code 在多语言上提升；math 几乎无变化（OWM 太小，5 epochs 不足以产生大影响）。

**重要发现**：MMLU MCF（multiple-choice format，直接输出 A/B/C/D）开始超过 random (>25%)。这挑战了 Gu et al. 2024 / Du et al. 2024 的小模型不会做 MCF 的说法。论文的额外 ablation 发现：把 FineWeb-Edu/DCLM 比例从 60/40 调到 40/60（更偏 DCLM）能小幅提升 MMLU MCF——DCLM 的对话风格与 MCF 输出更兼容。

### Stage 3 (8–10T tokens)

| Component | Ratio | 变化 |
|-----------|-------|------|
| Web (40/60) | ~66% | 翻转比例 |
| Stack-Edu | ~20% | 替代 StarCoderData |
| Math (OWM + InfiMM-WebMath-text) | ~10% | ↑5% |
| Jupyter Notebooks | 微量 | 新增 |

**Findings**：出现 loss spike，回退 + 跳过相关数据后仍存在——原因未明，但评估指标在阶段末恢复。这种 spike 可能与 mixture 分布突变有关（Stack-Edu vs StarCoderData 分布差异 + 翻转 web 比例）。

### Stage 4 (10–11T tokens, Decay)

| Component | Ratio |
|-----------|-------|
| Web (40/60) | 58% |
| Stack-Edu (扩展更多语言) | 24% |
| Math: InfiWebMath-3+ + FineMath-4+ | 14% |
| AugGSM8K | 0.02% |
| OWM | 0.08% |
| Cosmopedia v2 (合成教材) | 4% |

**Findings**：math 从 7.27 → 22.07（stage 3 末 → stage 4 末），code 16.75 → 23.21。验证了 annealing 阶段注入高质量 specialized data 的策略。

### Context Length Extension

取 stage 4 中间 checkpoint（在最后 75B tokens 之前），继续训练：

- Seq len: 2k → 8k
- RoPE $\theta$: 10k → 130k
- Mixture: 40% long-context docs (DCLM 10% + FineWeb-Edu 10% + Dolma books 20%) + 60% stage 4 mix

最终得到 SmolLM2-1.7B base model。

---

## 6. 评估亮点

### Base Model (Table 4)

| Benchmark | SmolLM2-1.7B | Llama3.2-1B | Qwen2.5-1.5B |
|-----------|-------------|-------------|-------------|
| HellaSwag | **68.7** | 61.2 | 66.4 |
| ARC | **60.5** | 49.2 | 58.5 |
| MMLU-Pro (held-out) | **19.4** | 11.7 | 13.7 |
| TriviaQA (held-out) | **36.7** | 28.1 | 20.9 |
| GSM8K (5-shot) | 31.1 | 7.6 | **61.7** |
| MATH (4-shot) | 11.6 | 3.3 | **34.3** |
| HumanEval | 22.6 | 18.9 | **37.2** |

SmolLM2 在 knowledge/reasoning 上明显领先，math/code 落后 Qwen2.5-1.5B（Qwen 用了 18T tokens + 强 math/code data 体系）但超 Llama3.2-1B。

### Instruct Model (Table 5)

| Benchmark | SmolLM2-Instruct | Llama3.2-1B-Inst | Qwen2.5-1.5B-Inst |
|-----------|-----------------|------------------|-------------------|
| IFEval | **56.7** | 53.5 | 47.4 |
| MT-Bench | 6.13 | 5.48 | **6.52** |
| GSM8K (5-shot) | 48.8 | 37.4 | **63.3** |
| MATH (4-shot) | **21.0** | 19.5 | 19.6 |
| HumanEval | 28.1 | 33.5 | **30.5** |

SmolLM2-Instruct 在 instruction-following（IFEval）上大幅领先，这归功于 SmolTalk 中的 Smol-Constraints 数据。

---

## 7. Post-Training 公式

### SFT

标准 cross-entropy loss，但只在 response tokens 上计算（mask prompt）：

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t})$$

- $x$：prompt
- $y$：target response
- $y_t$：第 $t$ 个 response token
- $\pi_\theta$：模型

超参：2 epochs, batch 128, seq 8192, LR $3.0 \times 10^{-4}$

### DPO

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

变量含义：
- $x$：prompt
- $y_w$：偏好（winner）response
- $y_l$：不偏好（loser）response
- $\pi_\theta$：当前策略（训练中）
- $\pi_{\text{ref}}$：参考策略（SFT checkpoint，冻结）
- $\beta = 0.5$：KL penalty 系数（小于常见的 0.1，意味着允许策略偏离 ref 更多）
- $\sigma$：sigmoid 函数

超参：2 epochs, LR $1.0 \times 10^{-6}$, batch 128, seq 1024, UltraFeedback 数据集。

**Intuition on $\beta=0.5$**：相比标准 DPO 的 $\beta=0.1$，更大的 $\beta$ 让 implicit reward signal 更强，但牺牲了一些 KL 约束。对小模型（容量有限）来说，需要更激进的 preference 信号来产生明显的行为变化。

---

## 8. 关键洞察与未解问题

### 8.1 MMLU MCF 涌现现象

6T tokens 后小模型开始能在 MMLU MCF 上超过 25% random baseline。这与 Gu et al. "小模型不能做 MCF" 的论断相反。可能的解释：长训练让小模型在 distribution 上更靠近 instruction-like text（DCLM 的 ELI5 风格），导致输出格式更可控。

### 8.2 Loss Spike 之谜

Stage 3 出现 unexplained loss spike，即使回退 + 数据 filtering 仍存在。这类 spike 在 T5/PaLM/Chinchilla 训练日志中都有报道，可能与 optimizer state 的二阶矩估计、batch 内 outlier token、或者 mixture distribution shift 相关。SmolLM2 没深究，但恢复说明 model capacity 没被破坏。

### 8.3 小模型 vs 单阶段训练

135M / 360M 用 single-stage training（不像 1.7B 的 4-stage）。原因：小模型容量不足以在不同阶段"切换风格"——多阶段 mixture shift 反而让小模型 confused。这是个很有意思的 scaling 现象。

### 8.4 数据重复的 4-5 epoch 阈值

遵循 Muennighoff et al. 2023 的 scaling data-constrained LM 工作，避免重复超过 4-5 epochs。SmolLM2 的 StarCoder-Data 在 stage 1 用 10% 配比，正是为了 11T total 下保留 ~4 epochs 的 headroom。

### 8.5 Chinchilla 比例被打破

Chinchilla 推荐 ~20 tokens/parameter。SmolLM2 用 11T / 1.7B ≈ **6470 tokens/parameter**，约 320× Chinchilla optimal。这是 de Vries "Go smol or go home" 论证的小模型 inference cost tradeoff：训练多花 compute 换 inference 便宜。

参考：
- https://arxiv.org/abs/2203.15556 (Chinchilla)
- https://arxiv.org/abs/2305.16264 (Muennighoff, data constrained)
- https://www.harmdevries.com/post/model-size-vs-compute-overhead/ (de Vries)

---

## 9. 资源链接

- 模型 collection: https://hf.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9
- FineMath dataset: https://huggingface.co/datasets/HuggingFaceTB/FineMath
- Stack-Edu dataset: https://huggingface.co/datasets/HuggingFaceTB/Stack-Edu
- SmolTalk dataset: https://huggingface.co/datasets/HuggingFaceTB/smoltalk
- Nanotron (training framework): https://github.com/huggingface/nanotron
- Datatrove (data pipeline): https://github.com/huggingface/datatrove
- LightEval (evaluation): https://github.com/huggingface/lighteval
- 135M / 360M model cards: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct, https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct

---

## 10. 整体直觉总结

SmolLM2 的核心 thesis 可以浓缩为：**小模型的瓶颈不在架构而在 data density**。1.7B 参数要塞下 world knowledge + reasoning + code + math + instruction following，每个 token 都得"值得学"。三个新数据集（FineMath, Stack-Edu, SmolTalk）的共同点是**用 Llama-3.1-70B/405B 当 judge 把稀有的高质量样本从海量噪声里筛出来**，然后通过 multi-stage 在线 re-balancing 让这些数据在 annealing 阶段发挥最大效用。

这跟 Phi-3/4 的 "textbooks are all you need" 路线相似，但 SmolLM2 用的是 web-crawl + classifier filtering（保留真实分布），而非纯合成数据，避免了 Phi 系列在某些 held-out benchmark 上的 overfitting 嫌疑。Qwen2.5-1.5B 在 math/code 上仍占优，说明 Hugging Face 的 specialized data pipeline 还有提升空间——下一个迭代点可能在于更激进的 synthetic reasoning data 注入。

如果你有兴趣，可以重点看 Appendix C.2 / C.3 / D.1 的 annotation prompt——这些 prompt 是整篇 paper 数据质量的灵魂，把 LLM-as-judge 的细粒度 rubric 写得非常具体（"Add 1 point if..."），是可复用的核心 IP。
