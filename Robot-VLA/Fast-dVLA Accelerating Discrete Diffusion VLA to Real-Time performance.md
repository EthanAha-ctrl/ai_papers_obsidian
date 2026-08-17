---
source_pdf: Fast-dVLA Accelerating Discrete Diffusion VLA to Real-Time performance.pdf
paper_sha256: 3dbbc9a73aa163031ebbcd4a876eadf57c1809f859075a501118c46b15204acb
processed_at: '2026-08-04T07:09:58-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Fast-dVLA

好，我换个更口语的方式再讲一遍，目标是让你 build intuition。

参考链接（先放这儿方便随时查）：
- Project page: https://chris1220313648.github.io/Fast-dVLA/
- Block Diffusion: https://arxiv.org/abs/2503.09573
- Diffusion Forcing: https://arxiv.org/abs/2407.01392
- Fast-dLLM: https://arxiv.org/abs/2505.22618
- Dream 7B: https://arxiv.org/abs/2508.15487
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054

---

## 一句话概括

现在的 discrete diffusion VLA 模型（dVLA）思路很新，但推理太慢，跑不到 real-time。这篇 paper 发现了一个很 hack 的事情：**虽然模型架构是全 bidirectional attention，但模型自己在解码时其实有 left-to-right 倾向**。于是作者就顺着这个倾向，把 attention 改成 block-causal，让 KV cache 能用起来，再叠加一个叫 diffusion forcing 的 trick 让不同 block 可以异步 denoise，最后拿一个 asymmetric distillation 让你不用从头训练。结果 4.1× 加速，性能不掉，real-world 能稳跑 30 Hz。

---

## 1. 先讲清楚 dVLA 是啥，以及它为什么慢

### 1.1 dVLA 的基本玩法

你有一个 VLM backbone，比如 Dream 7B（https://arxiv.org/abs/2508.15487）。你想让它输出 robot action。

传统做法是加一个 flow-matching head 输出连续 action（π0 这类，https://arxiv.org/abs/2410.24164）。dVLA 换了个思路：**把 action 离散化成 token，直接用 LLM 的 token 预测机制来做**。

具体说，一个 action chunk 被表示成长度 L 的离散 token 序列：

$$\mathbf{a}_0 = (a_0^1, a_0^2, \dots, a_0^L)$$

这里 $a_0^i$ 就是第 $i$ 个 action token，对应一段低层 robot action。词汇表里加一个 special mask token M。

训练时搞 forward diffusion：按 mask ratio $\gamma_t \in (0, 1]$ 随机把一部分 token 替换成 M。模型学习从 corrupted 序列 $\tilde{\mathbf{a}}_t$ 恢复出原始 token，loss 只在 mask 位置算：

$$\mathcal{L}_{\text{act}}(\theta) = -\sum_{i \in \mathcal{M}_{\gamma_t}} \log p_\theta(a_0^i \mid \tilde{\mathbf{a}}_t, \mathbf{c})$$

变量解释：
- $\theta$ 是模型参数
- $\mathcal{M}_{\gamma_t}$ 是被 mask 的位置集合
- $\mathbf{c}$ 是 multimodal context（图像 + 语言指令）
- $\tilde{\mathbf{a}}_t$ 是加了噪声的序列

推理时反复 denoise，大概 10-30 步就能把全 mask 序列恢复成完整 action。

### 1.2 为什么慢

dVLA 用的是 bidirectional attention，所有 token 互相可见。这带来一个要命的问题：**KV cache 完全用不了**。

你想啊，AR 模型（GPT 这种）为啥快？因为前面 token 的 K/V 算完就 cache，下一个 token 只需要 query 这些 cached K/V。但 bidirectional attention 里，token $i$ 的 K/V 依赖整个序列（包括 mask 位置）。每次 denoise 一步，mask 状态变了，整个序列的 K/V 全部要重算。所以每一步 forward 都是 full attention，没有 cache 复用。

Figure 1（paper 里那个图）很直观：

| 范式 | Forward 次数 | 单步速度 | 总速度 |
|---|---|---|---|
| AR VLA | 多（L 次） | 快（KV cache） | 中 |
| dVLA bidirectional | 少（10-30 步） | 慢（每步重算） | 慢 |
| Block Diffusion | 中等 | 中（block 内 bidirectional, block 间 AR） | 中 |
| Fast-dVLA | 少 | 快（KV cache + inter-block 并行） | 快 |

dVLA 的尴尬是：虽然 forward 次数少了，但每次 forward 太贵，总速度反而比 AR 慢。

---

## 2. 关键 observation：dVLA 偷偷在 left-to-right（Figure 3）

这是整篇 paper 的"aha moment"。

作者拿了 Dream-VLA，记录它在 denoise 过程中每个位置被 decode 的频率，画成热图（Figure 3，step 0/5/10）。

按理说 bidirectional attention 应该是"全局同时收敛"，对吧？但热图显示：**earlier temporal positions 早在 early diffusion iterations 就被 decode 了，later positions 晚很多才被 decode**。

也就是说，模型其实表现出了 left-to-right 解码倾向，虽然架构上没有强制这个。

作者给了两个原因：
1. **Backbone bias**：dVLA 的 backbone 通常从 AR VLM 初始化（比如 Dream 7B 是 AR diffusion LLM），保留了 autoregressive characteristic。
2. **Action 时序依赖**：action 之间有时间依赖（trajectory），早 action 决定晚 action，模型自然学到先 decode 早的。

这个 observation 直接启发了一个想法：**既然模型本来就想 left-to-right，那我把 attention 强行改成 block-causal 应该不会掉多少性能**。

---

## 3. 核心 idea 1：Block-wise attention 启用 KV cache

### 3.1 怎么改

把 action token 序列切成 N 个 block，每个 block 大小 k（paper 里 k=7，因为 action chunk 通常是 7 维）。

第 i 个 block 的索引集合：

$$B_i = \{(i-1)k, \dots, ik-1\}$$

对应的 token 子序列记为 $Y_{B_i}$。

Attention 改成 block-causal：
- **Block 内**：仍然 bidirectional，所有 token 互相 attend
- **Block 间**：block $i$ 只能 attend 到 block $1, 2, \dots, i$，不能 attend 到未来 block

### 3.2 为啥这样 KV cache 就能用了

你想 bidirectional attention 为啥 cache 不了？因为 token $i$ 的 K/V 依赖所有 token 的 embedding，包括 mask 位置的 embedding。mask 状态一变，K/V 就变。

改成 block-causal 后，block $i$ 内 token 的 K/V 只依赖 block $1..i$。一旦 block $i$ 内所有 token 都已经 unmask 完成，block $i$ 的 K/V 就**永远固定**了——因为它的输入（block 1..i 的当前状态）不会变了。

Figure 4 很直观：
- (a) bidirectional：每一步 KV 都在变
- (b) block-wise：block 完成后 KV 固定，可以 cache

后续 block 的 denoise 只需要 query 这部分 cached K/V 就行，不用重算。

直觉：这是把 AR 的 KV cache 好处部分搬到了 diffusion 框架里，但保留了 block 内的并行性。

---

## 4. 核心 idea 2：Diffusion Forcing 让 block 间也能并行

### 4.1 Block Diffusion 的问题

光做 block-causal 还不够。已有的 Block Diffusion（https://arxiv.org/abs/2503.09573）就是这个思路：block 内 parallel decoding，block 间严格 AR——前一个 block 完全完成才开始下一个。

这样虽然 KV cache 能用，但 block 间没有 overlap，加速有限（大概 1.8-3.3×，Table 1 里能看到）。

### 4.2 Diffusion Forcing 的核心思路

Diffusion Forcing（Chen et al., NeurIPS 2024, https://arxiv.org/abs/2407.01392）原本是用在 video diffusion 里的 idea。核心 insight：**不同时间步可以用不同 noise level 训练，模型学会异步 denoise**。

应用到 Fast-dVLA：给不同 block 分配不同的 noise level，单调递增：

$$t_1 < t_2 < \cdots < t_N$$

完整的 noisy 序列记为：

$$Y^{t_{1:N}} = \{Y_{B_1}^{t_1}, Y_{B_2}^{t_2}, \dots, Y_{B_N}^{t_N}\}$$

含义：早 block 噪声低（信息保留多，接近 clean），晚 block 噪声高（很乱，需要更多步 denoise）。

### 4.3 Block-wise AR 分解

模型学到的联合分布按 block-wise autoregressive 分解：

$$p_\theta(Y^0 \mid Y^{t_{1:N}}) = \prod_{i=1}^{N} p_\theta(Y_{B_i}^0 \mid Y_{B_1}^{t_1}, Y_{B_2}^{t_2}, \dots, Y_{B_i}^{t_i})$$

变量含义：
- $Y^0$ 是完整 clean 序列
- $Y_{B_i}^0$ 是第 i 个 block 的 clean tokens
- $Y_{B_j}^{t_j}$ 是第 j 个 block 在 noise level $t_j$ 下的 noisy 版本
- 条件路径：block $i$ 的预测只看 block $1..i$ 的当前 noisy 状态（block-causal）

这个 factorization 的妙处：因为不同 block 在不同 noise level，推理时可以让早 block 先 denoise 完成（noise 低，需要步数少），晚 block 后 denoise（noise 高，需要步数多），它们可以**同时**处于 pipeline 的不同阶段。

类比 CPU 的 instruction pipeline：不同指令在不同 stage，整体吞吐量提升。

### 4.4 与 Block Diffusion 的区别

| | Block Diffusion | Fast-dVLA |
|---|---|---|
| Block 内 attention | Bidirectional | Bidirectional |
| Block 间 attention | Causal | Causal |
| 训练时 noise level | 所有 block 同 t | 不同 block 不同 t |
| 推理时 block 间关系 | 严格串行 | 异步并行 |

训练时让模型见过"前 block 部分干净 + 后 block 高噪声"的混合状态，模型就学会了这种异步生成。

---

## 5. 训练 trick：Asymmetric Distillation（关键效率保证）

### 5.1 直接训练的问题

如果你从零训一个 Fast-dVLA，loss 是：

$$\mathcal{L}_{\text{BD}} = \mathbb{E} \sum_{i=1}^{N} \left[-\log p_\theta(Y_{B_i}^0 \mid Y_{B_<i}^{t_<i}, c)\right]$$

这里 $Y_{B_<i}^{t_<i}$ 表示 block $1..i-1$ 在各自 noise level 下的状态。这需要从头训，慢。

### 5.2 Asymmetric Distillation 的核心

Key insight：你手上已经有一个 finetuned 的 bidirectional dVLA（比如 Dream-VLA 已经 finetuned 好了）。拿它当 teacher，用 block-wise causal 的 Fast-dVLA 当 student，做 KL distillation。

但 "asymmetric" 体现在视野不对称：

$$\mathcal{L}_{\text{AD}} = \mathbb{E}\left[\sum_{i=1}^{N} D_{\text{KL}}\left(p_\theta(Y_{B_i}^0 \mid Y_{B_{\leq i}}^{t_{\leq i}}, c) \,\Vert\, p_{\phi^-}(Y_{B_i}^0 \mid Y_{B_{\leq N}}^{t_{\leq N}}, c)\right)\right]$$

变量含义：
- $p_\theta$ 是 student（block-causal，Fast-dVLA）
- $p_{\phi^-}$ 是 teacher（bidirectional，已 finetuned 的原 dVLA）
- Student 只看 block $1..i$（causal 限制）
- Teacher 看全部 block $1..N$（全局视野）
- $D_{\text{KL}}$ 是 KL divergence，在 mask token 位置上聚合

"asymmetric" 就是：teacher 用 global view 预测每个 block，student 用 causally restricted view 去拟合 teacher 的预测。这逼 student 学会：在缺未来 context 时也能逼近 teacher 在 full context 下的输出。

### 5.3 LoRA 实现

工程上用 LoRA（rank=32）做 distillation，但有个关键 trick：
- 计算 teacher logits 时：**关掉 LoRA 分支**（用原 backbone，保证 teacher 是 frozen 原模型）
- 计算 student logits 时：**打开 LoRA 分支**

这样最大化保留 pretrained VLM 的 visual-language prior，让 LoRA 只学 attention pattern 的转换。

### 5.4 训练效率（Figure 8）

四种策略对比（基于 Dream-VLA on LIBERO，纵轴 action MSE，横轴 training steps）：

1. **Asymmetric Distillation from Finetuned Weight**（蓝线）：2000 steps 收敛
2. **Training from Finetuned Weight with $\mathcal{L}_{\text{BD}}$**（橙线）：约 10000 steps
3. **Training from Scratch with $\mathcal{L}_{\text{BD}}$**（绿线）：约 20000 steps
4. **Training from Scratch with $\mathcal{L}_{\text{act}}$**（红线，baseline dVLA）：更慢

asymmetric distillation 比从头训快 **10×**，比从 finetuned weight 继续训快 **5×**。

具体 budget：
- Dream-VLA：4k distill steps（约原 finetune budget 的 1/5）
- DD-VLA：4k distill steps（约 1/8）
- UD-VLA：3k distill steps（约 1/8）

直觉解释：student 和 teacher 同架构，只是 attention pattern 不同。LoRA 只需学"如何在缺未来 context 下逼近 teacher 在 full context 下的预测"，这是个低维映射问题，远比从头学 action 生成容易。

类比：把一个 bidirectional translator 蒸馏成 left-to-right translator，比训一个 left-to-right translator from scratch 快得多——大部分语言知识是共享的。

---

## 6. 推理 trick：Pipelined Parallel Decoding（Algorithm 1）

这是 paper 最工程化的部分。难点：如何让 inter-block parallelism 真正变成 speedup，同时不让性能崩。

### 6.1 双状态机制

每个 active block 有两个状态：
- **Semi-activated**：刚加入 pipeline，谨慎解码
- **Fully-activated**：成熟，激进解码

两个阈值控制状态转换：
- $\tau_{\text{add}} = 2/7$：前一 block 完成 ratio 超过 2/7 时，新 block 加入 pipeline，状态为 semi-activated
- $\tau_{\text{act}} = 4/7$：前一 block 完成 ratio 超过 4/7 时，当前 block 转为 fully-activated

直觉：$\tau_{\text{add}} < \tau_{\text{act}}$ 创造一个 overlap 区间——前 block 还没完成就启动后 block，但后 block 在前 block "基本完成"前都保持 semi-activated 谨慎状态。

### 6.2 Semi-activated 下的 confidence-aware decoding

Semi-activated block 用 confidence-aware decoding（来自 Fast-dLLM 思路）：
- 计算所有 mask 位置的 confidence score
- 只 decode confidence $\geq \tau_{\text{conf}}$ 的位置
- 默认 $\tau_{\text{conf}} = 0.5$

直觉：在信息还不完整时（前 block 还在 denoise），只对那些 model 很有把握的 token 做 commit，避免错误累积。

### 6.3 Fully-activated 下的 radical decoding

Fully-activated block 用对数调度：
- $k \gets \lfloor |\mathcal{R}_i| / n \rfloor$，其中 $\mathcal{R}_i$ 是剩余 mask 位置
- block-specific 阈值：$\tau_i \gets \min(\tau_{\text{conf}}, \min(\text{TopK}(\mathbf{c}_i, k)))$
- 每步保证至少 decode $1/n$ 的剩余 tokens

n 的选择（Table S2）：
- $\log_2$（n=2）：186.67 tokens/s, avg. len. 4.54（最快）
- $\log_3$（n=3）：164.42 tokens/s, avg. len. 4.57
- $\log_4$（n=4）：144.71 tokens/s, avg. len. 4.58

激进 $\log_2$ 性能没掉，速度最快，所以默认用它。

### 6.4 KV cache 更新时机

Algorithm 1 第 22 行："Update the KV cache for completed blocks"——只有 block 完全 unmask 后才 cache 它的 KV，之后所有 step 复用。

### 6.5 整体 pipeline 直觉

想象一个 4-block 序列，pipeline 大概长这样：

```
Step 1: B1 (full mask, t=0.9)
Step 2: B1 (partial, t=0.6)
Step 3: B1 (partial, t=0.4) → 加 B2 (full mask, t=0.9, semi)
Step 4: B1 完成 ratio > 2/7, 仍在 denoise; B2 (semi, decode high-conf tokens)
Step 5: B1 完成 ratio > 4/7, B2 转 fully-activated; 加 B3 (semi)
Step 6: B1 完成 → cache KV; B2 (fully, radical decode); B3 (semi)
...
```

每个时刻 pipeline 里同时有 2-3 个 block 在不同阶段，类似 CPU 的 instruction-level parallelism。

### 6.6 为什么双状态机制必要

如果一加入 pipeline 就激进 decode 后 block，前 block 还在变 → 后 block 基于错误前缀生成 → 错误累积。

如果一直谨慎 decode，pipeline 利用率低 → 没加速。

双状态是折中：semi-activated 谨慎（只 commit high-conf token），等前 block "基本完成"（>4/7）后再激进。这本质是一种 speculative decoding 的 confidence-gating（https://arxiv.org/abs/2211.17192）。

---

## 7. 实验数据（人话版）

### 7.1 RQ1：加速策略对比（Table 1, LIBERO）

| Decoding Method | Avg. SR | Speed |
|---|---|---|
| Dream-VLA baseline | 85.6% | 98.8 tokens/s (×1.0) |
| + Fast-dLLM | 82.8% | 183.2 (×1.9) |
| + Block Diffusion | 85.8% | 181.7 (×1.8) |
| **+ Fast-dVLA** | **87.0%** | **313.1 (×3.2)** |
| DD-VLA baseline | 96.3% | 152.1 (×1.5) |
| + Fast-dLLM | 93.5% | 312.5 (×3.2) |
| + Block Diffusion | 96.7% | 322.1 (×3.3) |
| **+ Fast-dVLA** | **96.6%** | **402.7 (×4.1)** |

人话解读：
- **Fast-dLLM**（training-free 直接 cache KV）：速度还行，但性能掉 2-3%，因为 bidirectional 下 KV cache 是 biased 的
- **Block Diffusion**：性能保持，但 speedup 有限（×1.8-3.3），因为 inter-block 串行
- **Fast-dVLA**：性能保持甚至略升，speedup ×3.2-4.1

为什么 Fast-dVLA 性能略高于 baseline？作者归因于 block-wise attention 在 training 时更稳定（temporal causality 显式约束）。

### 7.2 UD-VLA 上的扩展（Table 2, CALVIN ABCD-D）

UD-VLA 是 unified dVLA，输出 visual foresight + action（625 tokens 长），block size 设为 32 的倍数。

| Method | Avg. Len. | Speed |
|---|---|---|
| UD-VLA | 4.64 | 67.3 (×1.0) |
| + Fast-dLLM | 4.32 | 132.5 (×2.0) |
| + Block Diffusion | 4.50 | 129.5 (×1.9) |
| **+ Fast-dVLA** | **4.54** | **186.7 (×2.8)** |

长 sequence 上 Fast-dVLA 仍然 2.8× 加速 + 性能基本保持（avg len 4.54 vs 4.64）。

### 7.3 与 SOTA flow-matching VLA 对比（Table 3, CALVIN ABCD-D）

| Method | 5/5 完成 | Avg. Len. |
|---|---|---|
| RT-1 | 22.7% | 2.45 |
| GR-1 | 73.1% | 4.21 |
| UP-VLA | 81.2% | 4.42 |
| MDT | 80.1% | 4.52 |
| **UD-VLA + Fast-dVLA** | **81.2%** | **4.54** |

Fast-dVLA 在 long-horizon 上达到 SOTA avg len，5/5 完成率 81.2% 与 UP-VLA 持平。也就是说 dVLA 加速后性能不输 flow-matching SOTA。

### 7.4 Real-world 30 Hz（Figure 7）

真实双臂 AgileX 平台，3 个任务：
1. **Conveyor Picking**：传送带抓取（动态）
2. **Vegetables Stowing**：按文字标签分拣
3. **Vegetables Retrieving**：按指令抓目标放锅里

每任务 100 demos 训练，40 trials 评测。

关键数据：**执行频率稳定 30 Hz**，conveyor 任务每分钟抓取数比 baseline 翻倍。这是 paper 标题"Real-Time Performance"的兑现。

### 7.5 Ablation 关键发现

**Block size 与 action dim 对齐**（Table 5, LIBERO-Long）：
- Multiples of action dim：SR 74.7%, speedup 4.01×
- Random block size：SR 73.3%, speedup 3.95×

block size 选 action dim 倍数（k=7）能更好保持 action 内在 temporal dependency。

**Confidence threshold $\tau_{\text{conf}}$**（Figure 9）：
- 默认 0.5
- 降低 → 速度线性提升，性能线性下降
- 0.5 是 sweet spot：2.8× 加速，性能仅降 2%

**$\tau_{\text{add}}$ / $\tau_{\text{act}}$ 双状态机制**（Table S1）：
- 当 $\tau_{\text{add}} = \tau_{\text{act}}$ 时退化为单状态（block 一加入就 fully-activated）
- 双状态（$\tau_{\text{add}} < \tau_{\text{act}}$）保性能同时速度接近单状态

---

## 8. 串起来直觉：为什么这套设计 work

### 8.1 为什么 block-wise attention 不掉性能

paper 隐含的论点：dVLA 的 bidirectional attention 实际上是 over-engineered——模型本身就有 left-to-right tendency（来自 AR VLM 初始化 + action 时序性）。所以把 global bidirectional 换成 block-causal 几乎不损失信息，反而让训练更稳定（显式 temporal causality 约束）。

类比：BERT bidirectional 在很多任务上并不比 GPT causal 显著好，因为数据本身有方向性。

### 8.2 为什么 diffusion forcing 能并行

传统 block diffusion 卡在"必须等前 block 完成才能开始后 block"，因为训练时所有 block 同 noise level，前 block 没完成 → 后 block 没条件 → 无法开始。

diffusion forcing 改成不同 noise level，后 block 训练时见到的是"前 block 部分干净 + 自己高噪声"的混合状态。模型学到这种异步生成能力，推理时就可以 overlap。

### 8.3 为什么 asymmetric distillation 比 from scratch 快 10×

student 和 teacher 同架构，只是 attention pattern 不同。LoRA 只需学"如何在缺未来 context 下逼近 teacher 在 full context 下的预测"。这是一个低维映射问题，远比从头学 action 生成容易。

### 8.4 整体直觉

你可以这样理解 Fast-dVLA：它把一个 fully bidirectional dVLA 转成了一个"**block-AR + intra-block diffusion + inter-block 异步 diffusion**"的混合体。

- Block-AR 部分：让 KV cache 能用（block 间），获得 AR 的速度优势
- Intra-block diffusion：保留 block 内 parallelism，避免 AR 的逐 token 串行
- Inter-block 异步 diffusion：让 block 间也能 overlap，打破 Block Diffusion 的严格串行限制

三者结合，dVLA 速度追上 flow-matching VLA，real-world 30 Hz 落地。

---

## 9. 在大图谱里的位置

### 9.1 dLLM 加速谱系

- **Fast-dLLM** (Wu et al., 2025, https://arxiv.org/abs/2505.22618)：training-free，强行复用 bidirectional KV cache，引入 bias。Fast-dVLA 用 block-wise attention 从架构上解决这个 bias。
- **Block Diffusion** (Arriola et al., 2025, https://arxiv.org/abs/2503.09573)：NLP 上的 block-AR-diffusion hybrid，但严格 block 间串行。Fast-dVLA 用 diffusion forcing 解开这个限制。
- **Diffusion Forcing** (Chen et al., NeurIPS 2024, https://arxiv.org/abs/2407.01392)：video diffusion 的异步 denoise 思想。Fast-dVLA 把它引入 action 生成。
- **Fast-dLLM via Discrete Diffusion Forcing** (Wang et al., ICLR 2026, https://openreview.net/forum?id=t5uLZSRjhF)：启发 Fast-dVLA 的 asymmetric distillation。

### 9.2 VLA 加速谱系

- **Pruning 类**：MoLe-VLA (Mixture-of-Layers), EfficientVLA, ADP, LightVLA
- **Early-exit 类**：Deer-VLA, CEED-VLA
- **Caching 类**：VLA-Cache, CronusVLA
- **量化类**：BitVLA (1-bit), QVLA
- **轻量架构**：TinyVLA, SmolVLA, Flower (950M)
- **Fast-dVLA 是首个针对 dVLA inference 加速的系统工作**

### 9.3 VLA 范式竞争

- AR VLA（OpenVLA, π0-FAST）：慢但成熟
- Flow-matching VLA（π0, π0.5, GR00T-N1）：当前 SOTA 性能 + 速度平衡
- dVLA：理论优势（统一 multimodal, 保留 VLM prior）但速度瓶颈

Fast-dVLA 让 dVLA 在速度上对齐 flow-matching，同时保留 dVLA 范式优势。这是范式竞争的关键 turn。

---

## 10. 相关联想

这是 build intuition 时可能有用的延伸阅读：

- **LLaDA** (Nie et al., 2025, https://arxiv.org/abs/2502.09992): 大规模 masked diffusion LLM
- **Dream 7B** (Ye et al., 2025, https://arxiv.org/abs/2508.15487): 7B diffusion LLM backbone
- **MDT** (Reuss et al., 2024, https://arxiv.org/abs/2410.10865): Multimodal Diffusion Transformer
- **Consistency Models** (Song et al., 2023, https://arxiv.org/abs/2303.01469): 一步生成思路，可叠加到 Fast-dVLA
- **Speculative Decoding** (Leviathan et al., 2023, https://arxiv.org/abs/2211.17192): draft-verify 范式，与双状态机制有精神关联
- **Jacobi Decoding** (Santilli et al., 2023, https://arxiv.org/abs/2305.13270): PD-VLA 用的并行解码思路
- **UniVLA** (Wang et al., 2025, https://arxiv.org/abs/2506.19850): unified VLA 思路
- **FlowVLA** (Zhong et al., 2025, https://arxiv.org/abs/2508.18269): visual chain-of-thought + flow matching
- **SimplerEnv** (Li et al., 2024, https://arxiv.org/abs/2405.05941): real-to-sim benchmark
- **LIBERO** (Liu et al., 2023, https://arxiv.org/abs/2306.03310): 标准 VLA benchmark
- **CALVIN** (Mees et al., 2022, https://arxiv.org/abs/2112.03227): long-horizon benchmark
- **MODE** (Reuss et al., 2025, https://arxiv.org/abs/2504.09095): mixture of expert denoisers

---

## 11. 复现要点

如果你要复现：

1. **选 base dVLA**：Dream-VLA 或 DD-VLA 都已开源
2. **LoRA distillation**：rank=32，teacher 关 LoRA / student 开 LoRA
3. **Block size**：选 action dimensionality 的整数倍
4. **Diffusion schedule**：单调递增 $t_1 < \cdots < t_N$，参考 Diffusion Forcing 原文
5. **Training steps**：~3-4k 步（base model finetune budget 的 1/5~1/8）
6. **Inference hyperparams**：$\tau_{\text{add}}=2/7$, $\tau_{\text{act}}=4/7$, $\tau_{\text{conf}}=0.5$, $n=2$（log2 radical decoding）
7. **Attention mask 实现**：block-wise causal，需自定义 attention mask
8. **KV cache 管理**：block 完成后 cache K/V，新 block 的 query 复用

工程坑点：
- Attention mask 必须区分 intra-block（bidirectional）和 inter-block（causal）
- KV cache 数据结构需支持 block-level 增量 append
- Pipeline 调度需在 GPU kernel 之外做 dynamic control flow（半 Python 半 CUDA）

---

## 12. 我的一些思考

这篇 paper 体现了一种趋势：**discrete diffusion LLM/VLA 的工程优化空间还很大**。目前 bidirectional 全注意力是 over-engineered，block-causal + asynchronous 是更精细的设计。

后续可能看到更多类似 fast-dLLM, Fast-dVLA, Block Diffusion 这类把"理论 elegant 的 bidirectional diffusion"压缩到"实用 fast"的工作。

另一个值得关注的点：paper 没有充分讨论的：
1. **Block size k=7 是手动调的**，能否自动 search？
2. **Asymmetric distillation 的 teacher-student gap 上界**没理论分析
3. **Real-world 30 Hz 是在什么 hardware 上**？如果需要 H100 才能 30 Hz，对社区仍难复现
4. **与 consistency model 的结合**：能否让每个 block 的 denoise step 也用 consistency distillation 减到 1-4 步？
5. **与 speculative decoding 的关系**：双状态机制本质是 confidence-gated speculative decoding，能否合并传统 spec decoding 的 verify 机制？
6. **Multi-task scaling**：只在 LIBERO/CALVIN/SIMPLER 任务上验证，能否 scale 到 Open-X-Embodiment 级别？

---

## 13. 总结

Fast-dVLA 的核心 contribution 是一个相当 elegant 的工程组合：
1. **Observation**（bidirectional dVLA 的隐式 AR 倾向）
2. **Architecture**（block-wise causal attention 启用 KV cache）
3. **Training objective**（diffusion forcing 启用 inter-block parallel）
4. **Distillation**（asymmetric KD 让训练成本降到 1/10）
5. **Inference scheduling**（双状态 pipelined decoding）

整体类似一个"把 bidirectional 模型蒸馏成 partially autoregressive + 异步 diffusion"的 recipe。结果是 dVLA 速度追上 flow-matching VLA，real-world 30 Hz 落地。

如果你想 build intuition，建议先看 Block Diffusion 原文理解 block-AR-diffusion hybrid，再看 Diffusion Forcing 理解异步 noise schedule，最后回 Fast-dVLA 看如何把两者融合并加 asymmetric distillation。这是一个典型的"组合多个 prior trick 形成新方法"的工程论文，每个组件单独看都不新，组合起来 solve 了一个真实的 bottleneck。

说白了，这篇 paper 就是告诉我们：**你以为 dVLA 是个 bidirectional 模型，其实它心里是个 AR 模型，你只要顺着它的本性改架构，速度就能起飞**。

---

# Fast-dVLA: 把 Discrete Diffusion VLA 推到 Real-Time 的工程艺术

这篇 paper 解决了一个相当具体但很重要的工程瓶颈：discrete diffusion VLA (dVLA) 推理速度太慢，远低于 30 Hz 的 real-time 要求。作者通过一个 key observation 把 dVLA 的 bidirectional attention 偷偷替换成 block-wise causal attention，再叠加 diffusion forcing 实现 inter-block parallelism，最终达到 4.1× 加速且性能不降。下面我从 motivation、method、training、inference、experiments 几个层面深入讲解。

参考链接：
- Project page: https://chris1220313648.github.io/Fast-dVLA/
- Block Diffusion: https://arxiv.org/abs/2503.09573
- Diffusion Forcing (NeurIPS 2024): https://arxiv.org/abs/2407.01392
- Fast-dLLM: https://arxiv.org/abs/2505.22618
- LLaDA: https://arxiv.org/abs/2502.09992
- Dream 7B: https://arxiv.org/abs/2508.15487
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- Dream-VLA: https://arxiv.org/abs/2512.22615
- DD-VLA: https://arxiv.org/abs/2508.20072

---

## 1. 背景：dVLA 范式的崛起与困境

当前 VLA 主流是 flow-matching 架构（π0、GR00T-N1、π0.5），VLM 做多模态理解，flow-matching head 输出连续 action。最近 dVLA 基于 diffusion LLM (dLLM) 思路兴起，代表工作有 Dream-VLA、DD-VLA、LLaDA-VLA、UD-VLA。

dVLA 的核心 idea：action 被离散化为 token 序列 $\mathbf{a}_0 = (a_0^1, \dots, a_0^L)$，其中 $a_0^i$ 是第 $i$ 个 action token（对应一段低层 robot action），词汇表加入特殊 mask token M。

Forward diffusion：按时间相关 mask ratio $\gamma_t \in (0,1]$ 随机把一部分 action token 替换为 M。
Reverse process：模型预测被 mask 位置的原始 token，用 cross-entropy 训练：

$$\mathcal{L}_{\text{act}}(\theta) = -\sum_{i \in \mathcal{M}_{\gamma_t}} \log p_\theta(a_0^i \mid \tilde{\mathbf{a}}_t, \mathbf{c})$$

变量含义：
- $\theta$：模型参数
- $\mathcal{M}_{\gamma_t}$：在 mask ratio $\gamma_t$ 下被 mask 的位置集合
- $\tilde{\mathbf{a}}_t$：corrupted 序列
- $\mathbf{c}$：multimodal context（视觉 + 语言）

dVLA 优势：inherent 在统一 multimodal alignment 上更好，更能保留 VLM pretrained knowledge（不像 flow-matching head 是一个额外模块）。

**核心痛点**：dVLA 推理频率远低于 30 Hz。原因是 bidirectional attention 让 KV cache 完全无法重用——每个 denoise step 整个序列的 K/V 都会变。

---

## 2. Figure 1 的四种解码范式对比（理解全文关键）

| 范式 | Forward per Sequence | Forward Speed | Speed per Sequence |
|---|---|---|---|
| (a) AR VLA | 多（L 次）| 快（KV cache）| 中等 |
| (b) dVLA bidirectional | 少（10-30 次 denoise）| 慢（每步重算全部 KV）| 慢 |
| (c) Block Diffusion | 中等（按 block 数）| 中等（block 内 bidirectional，block 间 AR）| 中等 |
| (d) Fast-dVLA | 少 | 快（KV cache + inter-block parallel）| 快 |

Block Diffusion (Arriola et al., 2025) 是关键 precursor：它把序列切成 block，block 内 bidirectional parallel decoding，block 之间 AR。这样每个 block 完成后 KV 可以缓存，下一个 block 用。但缺点是 inter-block 严格串行，无法 overlap。

Fast-dVLA 的核心突破：通过 diffusion forcing 让不同 block 处于不同 noise level，从而可以异步 denoise——前面 block 还在 denoise 时，后面 block 已经以更高 noise level 开始 denoise。这就是 inter-block parallelism。

---

## 3. Motivation：bidirectional dVLA 的隐式 AR 倾向（Figure 3）

这是 paper 最有意思的 observation。作者把 Dream-VLA 在 denoise 过程中每个位置被 decode 的频率画出来（Figure 3，step 0/5/10 的热图）。

发现：**尽管模型架构是 fully bidirectional attention，但解码顺序呈现明显 left-to-right pattern**——earlier temporal positions 早在 early diffusion iterations 就被 decode 了。

两个原因：
1. **Backbone bias**：dVLA backbone 通常从 AR VLM 初始化（Dream 7B 是 AR diffusion LLM），保留一定 autoregressive characteristic。
2. **Action 时序依赖**：action 之间有时间依赖（chunk 内的 trajectory），早 action 决定晚 action，模型自然学到先 decode 早的。

这个 observation 暗示：你可以强行把 bidirectional dVLA 转成 block-wise causal 的形式，性能损失应该可控——因为模型本来就有这种倾向。

---

## 4. 方法核心 1：Block-wise Attention 与 KV Cache 复用

### 4.1 设计思路

把 action token 序列切成 N 个 block，每个 block 大小 k（论文里 k=7，与 action dimensionality 对齐，因为 action chunk 通常是 7 维）。

第 i 个 block 的索引集合：
$$B_i = \{(i-1)k, \dots, ik-1\}$$
对应 token 子序列 $Y_{B_i}$。

**Block-wise causal attention**：block $i$ 的 token 只能 attend 到 block $1..i$ 的 token，不能 attend 到 block $i+1..N$。在 block 内部，attention 仍是 bidirectional 的（block 内所有 token 互相可见）。

效果（Figure 4b）：一旦 block $i$ 内所有 token 都被 unmask 完成，其 KV 状态就**永远固定**，后续 block 的 denoise 可以直接复用这部分 KV cache。这对比 bidirectional attention（Figure 4a），KV 每个 step 都在变。

### 4.2 为什么这个 trick 在 dVLA 上能 work

正常情况下 bidirectional attention 的 KV 不能 cache 是因为：token $i$ 的 K/V 依赖所有 token 的 embedding，包括 masked 位置（mask token 也是 embedding）。所以 mask 状态一变，K/V 就变。

改成 block-causal 后，block $i$ 内 token 的 K/V 只依赖 block $1..i$。当 block $i$ 内所有 token 都已 unmask（mask 状态固定），block $i$ 的 K/V 就永久固定。后续 step 处理 block $i+1$ 时只需 query block $i+1$ 的 token 对 block $1..i$ 的 cached K/V 做 attention 即可。

直觉：这是把 AR-style KV cache 的好处部分搬到了 diffusion 框架内，但保留了 block 内的 parallelism。

---

## 5. 方法核心 2：Diffusion Forcing 实现 Inter-block Parallelism

### 5.1 单调 noise schedule

这是从 Diffusion Forcing (Chen et al., NeurIPS 2024) 借来的 idea，原文用于 video diffusion。核心 insight：不同时间步可以用不同 noise level 训练，让模型学会异步 denoise。

形式化：给 block $i$ 分配 noise level $t_i$，要求单调递增：
$$t_1 < t_2 < \cdots < t_N$$

完整噪声序列：
$$Y^{t_{1:N}} = \{Y_{B_1}^{t_1}, \dots, Y_{B_N}^{t_N}\}$$

含义：早 block 噪声低（信息保留多），晚 block 噪声高（更不确定）。

### 5.2 Block-wise AR 分解

模型学到的联合分布按 block-wise autoregressive 分解：

$$p_\theta(Y^0 \mid Y^{t_{1:N}}) = \prod_{i=1}^{N} p_\theta(Y_{B_i}^0 \mid Y_{B_1}^{t_1}, \dots, Y_{B_i}^{t_i})$$

变量含义：
- $Y^0$：完整 clean 序列
- $Y_{B_i}^0$：第 i 个 block 的 clean tokens
- $Y_{B_j}^{t_j}$：第 j 个 block 在 noise level $t_j$ 下的 noisy 版本
- 条件路径：block i 的预测只依赖 block 1..i 的当前 noisy 状态（block-wise causal）

这个 factorization 的妙处：因为不同 block 在不同 noise level，所以推理时可以让早 block 先 denoise 完成（noise level 低，需要 step 少），晚 block 后 denoise（noise level 高，需要 step 多），它们可以**同时**处于 pipeline 不同阶段。

类比 video diffusion 的 "from slow bidirectional to fast autoregressive" (Yin et al., CVPR 2025, https://arxiv.org/abs/2503.07978)：把 bidirectional 一次性生成转成多阶段异步生成，每个阶段处理不同清晰度的内容。

### 5.3 与 Block Diffusion 的区别

| | Block Diffusion | Fast-dVLA |
|---|---|---|
| Block 内 attention | Bidirectional | Bidirectional |
| Block 间 attention | Causal | Causal |
| 推理时 block 间关系 | 严格串行（前 block 完成才开始后 block）| 异步并行（前 block 部分完成即可激活后 block）|
| 训练目标 | 单一 noise level（所有 block 同 t）| 不同 block 不同 t（diffusion forcing）|

---

## 6. 训练：Asymmetric Distillation（关键效率 trick）

### 6.1 从头训练目标 L_BD

如果直接训练 Fast-dVLA，loss 是：

$$\mathcal{L}_{\text{BD}} = \mathbb{E} \sum_{i=1}^{N} \left[-\log p_\theta(Y_{B_i}^0 \mid Y_{B_<i}^{t_<i}, c)\right]$$

这里 $Y_{B_<i}^{t_<i}$ 表示 block 1..i-1 在各自 noise level 下的状态。但这个训练需要从头，慢。

### 6.2 Asymmetric Distillation 损失 L_AD

Key idea：用已有的 finetuned bidirectional dVLA 当 teacher，用 block-wise causal dVLA 当 student，做 KL distillation。但"asymmetric"的含义是视野不对称：

$$\mathcal{L}_{\text{AD}} = \mathbb{E}\left[\sum_{i=1}^{N} D_{\text{KL}}\left(p_\theta(Y_{B_i}^0 \mid Y_{B_{\leq i}}^{t_{\leq i}}, c) \,\Vert\, p_{\phi^-}(Y_{B_i}^0 \mid Y_{B_{\leq N}}^{t_{\leq N}}, c)\right)\right]$$

变量含义：
- $p_\theta$：student 模型（block-wise causal，Fast-dVLA）
- $p_{\phi^-}$：teacher 模型（bidirectional，已 finetuned 的原 dVLA）
- $Y_{B_{\leq i}}^{t_{\leq i}}$：student 只看 block 1..i（受 causal 限制）
- $Y_{B_{\leq N}}^{t_{\leq N}}$：teacher 看全部 block 1..N（全局视野）
- $D_{\text{KL}}$：在 mask token 位置上聚合的 KL divergence

"asymmetric" 的 essence：teacher 用 global view 预测每个 block，student 用 causally restricted view 拟合 teacher 的预测。这个不对称让 student 学到在缺乏 future context 时也能逼近 teacher 在 full context 下的输出。

### 6.3 LoRA 实现

工程上用 LoRA（rank=32）做 distillation。关键 trick：
- 计算 teacher logits 时：**关掉 LoRA 分支**（用原 backbone）
- 计算 student logits 时：**打开 LoRA 分支**

这样保证 teacher 是 frozen 的原模型，student 是 LoRA-adapted 的同架构模型。最大保留 pretrained VLM 的 visual-language prior，让 LoRA 只学 attention pattern 的转换。

### 6.4 训练效率（Figure 8）

四种训练策略对比（基于 Dream-VLA on LIBERO，纵轴 action MSE，横轴 training steps）：
1. **Asymmetric Distillation from Finetuned Weight**（蓝）：2000 steps 收敛
2. **Training from Finetuned Weight with $\mathcal{L}_{\text{BD}}$**（橙）：~10000 steps
3. **Training from Scratch with $\mathcal{L}_{\text{BD}}$**（绿）：~20000 steps
4. **Training from Scratch with $\mathcal{L}_{\text{act}}$**（红，baseline dVLA）：更慢

asymmetric distillation 比从头训练快 **10×**，比 finetuned weight 继续训练快 **5×**。这是论文的实用价值：你可以拿一个已 finetuned 的 dVLA，花 1/5 ~ 1/8 原训练预算就转成 Fast-dVLA。

具体配置：
- Dream-VLA：4k distill steps（约 1/5 原 finetune budget）
- DD-VLA：4k distill steps（约 1/8 原 finetune steps）
- UD-VLA：3k distill steps（约 1/8），batch size 12

---

## 7. 推理：Pipelined Parallel Decoding（Algorithm 1）

这是 paper 最工程化的部分。难点在于：如何让 inter-block parallelism 真正变成 speedup，同时不让性能崩。

### 7.1 双状态机制

每个 active block 有两个状态：
- **Semi-activated**：刚加入 pipeline，谨慎解码
- **Fully-activated**：成熟，激进解码

两个阈值控制状态转换：
- $\tau_{\text{add}} = 2/7$：当前一 block 完成 ratio 超过 2/7 时，新 block 加入 pipeline 为 semi-activated
- $\tau_{\text{act}} = 4/7$：当前一 block 完成 ratio 超过 4/7 时，当前 block 转为 fully-activated

直觉：$\tau_{\text{add}} < \tau_{\text{act}}$ 创造一个 overlap 区间——前 block 还没完成就启动后 block，但后 block 在前 block "基本完成"前都保持谨慎（semi-activated）。

### 7.2 Confidence-aware decoding in semi-activated state

Semi-activated block 用 confidence-aware decoding（来自 Fast-dLLM 思路）：
- 计算所有 mask 位置的 confidence score
- 只 decode confidence $\geq \tau_{\text{conf}}$ 的位置
- 默认 $\tau_{\text{conf}} = 0.5$（Figure 9 ablation）

直觉：在信息还不完整时（前 block 还在 denoise），只对那些 model 很有把握的 token 做 commit，避免错误累积。

### 7.3 Radical decoding in fully-activated state

Fully-activated block 用对数调度：
- $k \gets \lfloor |\mathcal{R}_i| / n \rfloor$，其中 $\mathcal{R}_i$ 是剩余 mask 位置
- block-specific 阈值：$\tau_i \gets \min(\tau_{\text{conf}}, \min(\text{TopK}(\mathbf{c}_i, k)))$
- 即每步保证至少 decode $1/n$ 的剩余 tokens

n 的选择（Table S2）：
- $\log_2$（n=2）：186.67 tokens/s, avg. len. 4.54（最快，每步解码半数）
- $\log_3$（n=3）：164.42 tokens/s, avg. len. 4.57
- $\log_4$（n=4）：144.71 tokens/s, avg. len. 4.58

激进 $\log_2$ 性能没掉，速度最快。所以默认用 $\log_2$。

### 7.4 KV cache 更新时机

Algorithm 1 第 22 行："Update the KV cache for completed blocks"——只有 block 完全 unmask 后才 cache 它的 KV，之后所有 step 复用。

### 7.5 整体 pipeline 直觉

想象一个 4-block 序列：

```
Step 1: B1 (full mask, t=0.9)
Step 2: B1 (partial, t=0.6)
Step 3: B1 (partial, t=0.4) → 加 B2 (full mask, t=0.9, semi)
Step 4: B1 完成 ratio > 2/7, still denoising; B2 (semi, decode high-conf tokens)
Step 5: B1 完成 ratio > 4/7, B2 转 fully-activated; 加 B3 (semi)
Step 6: B1 完成 → cache KV; B2 (fully, radical decode); B3 (semi)
...
```

每个时刻 pipeline 里同时有 2-3 个 block 在不同阶段，类似于 CPU pipeline 的 instruction-level parallelism。

---

## 8. 实验数据详解

### 8.1 RQ1：与其它加速策略对比（Table 1, LIBERO）

| Decoding Method | Spatial | Goal | Object | Long | Avg. SR | Speed |
|---|---|---|---|---|---|---|
| Dream-VLA baseline | 90.2 | 92.0 | 88.0 | 72.0 | 85.6 | 98.8 (×1.0) |
| + Fast-dLLM | 88.4 | 89.4 | 83.4 | 70.2 | 82.8 | 183.2 (×1.9) |
| + Block Diffusion | 91.8 | 90.4 | 88.6 | 72.2 | 85.8 | 181.7 (×1.8) |
| **+ Fast-dVLA** | 91.2 | 92.0 | 90.2 | 74.6 | 87.0 | 313.1 (×3.2) |
| DD-VLA baseline | 97.2 | 98.6 | 97.4 | 92.0 | 96.3 | 152.1 (×1.5) |
| + Fast-dLLM | 94.0 | 95.2 | 94.8 | 89.8 | 93.5 | 312.5 (×3.2) |
| + Block Diffusion | 97.6 | 98.6 | 97.2 | 93.2 | 96.7 | 322.1 (×3.3) |
| **+ Fast-dVLA** | 97.0 | 98.8 | 97.6 | 92.8 | 96.6 | 402.7 (×4.1) |

关键结论：
- Fast-dLLM（training-free 直接 cache KV）：速度还行但**性能掉 2-3%**，因为 bidirectional 下 KV cache 是 biased 的
- Block Diffusion：性能保持但 speedup 有限（×1.8-3.3），因为 inter-block 串行
- **Fast-dVLA**：性能保持甚至略升，speedup ×3.2-4.1

为什么 Fast-dVLA 比 baseline 性能略高？作者归因于 block-wise attention 在 training 时更稳定（temporal causality 显式约束）。

### 8.2 UD-VLA 上的扩展（Table 2, CALVIN ABCD-D）

UD-VLA 是 unified dVLA，输出 visual foresight + action（625 tokens 长），block size 设为 32 的倍数。

| Method | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | Avg. Len. | Speed |
|---|---|---|---|---|---|---|---|
| UD-VLA | 99.2 | 96.8 | 93.6 | 90.4 | 84.0 | 4.64 | 67.3 (×1.0) |
| + Fast-dLLM | 97.2 | 92.0 | 85.8 | 80.8 | 76.2 | 4.32 | 132.5 (×2.0) |
| + Block Diffusion | 98.8 | 94.4 | 89.4 | 86.2 | 80.4 | 4.50 | 129.5 (×1.9) |
| **+ Fast-dVLA** | 98.4 | 95.2 | 92.2 | 87.0 | 81.2 | 4.54 | 186.7 (×2.8) |

UD-VLA 长 sequence 上 Fast-dVLA 仍然 2.8× 加速 + 性能基本保持（avg len 4.54 vs 4.64）。

### 8.3 RQ2：与 SOTA flow-matching VLA 对比（Table 3, CALVIN ABCD-D）

| Method | 1/5 | 5/5 | Avg. Len. |
|---|---|---|---|
| RT-1 | 84.4 | 22.7 | 2.45 |
| GR-1 | 94.9 | 73.1 | 4.21 |
| UniVLA* | 94.8 | 69.0 | 4.26 |
| UP-VLA | 96.2 | 81.2 | 4.42 |
| MDT | 98.6 | 80.1 | 4.52 |
| **UD-VLA + Fast-dVLA** | 98.4 | 81.2 | 4.54 |

Fast-dVLA 在 long-horizon 上达到 SOTA avg len，5/5 完成率 81.2% 与 UP-VLA 持平。

### 8.4 RQ3：Real-world 30 Hz 突破（Figure 7）

真实双臂 AgileX 平台，3 个任务：
1. **Conveyor Picking**：传送带抓取（动态任务）
2. **Vegetables Stowing**：按文字标签分拣
3. **Vegetables Retrieving**：按指令抓目标放锅里

每任务 100 demos 训练，40 trials 评测。

关键数据：**执行频率稳定 30 Hz**，conveyor 任务每分钟抓取数比 baseline 翻倍。这是 paper 标题"Real-Time Performance"的兑现。

### 8.5 RQ5：Ablation 关键发现

**Block size 与 action dim 对齐**（Table 5, LIBERO-Long）：
- Multiples of action dim：SR 74.7%, speedup 4.01×
- Random block size：SR 73.3%, speedup 3.95×

block size 选 action dim 倍数（k=7）能更好保持 action 内在 temporal dependency。

**Confidence threshold $\tau_{\text{conf}}$**（Figure 9）：
- 默认 0.5
- 降低 → 速度线性提升，性能线性下降
- 0.5 是 sweet spot：2.8× 加速，性能仅降 2%

**$\tau_{\text{add}}$ / $\tau_{\text{act}}$ 的双状态机制**（Table S1）：
- 当 $\tau_{\text{add}} = \tau_{\text{act}}$ 时退化为单状态（block 一加入就 fully-activated）
- 双状态（$\tau_{\text{add}} < \tau_{\text{act}}$）保性能同时速度接近单状态

---

## 9. 与相关工作位置定位

### 9.1 在 dLLM 加速谱系中

- **Fast-dLLM** (Wu et al., 2025, https://arxiv.org/abs/2505.22618)：training-free，强行复用 bidirectional KV cache，引入 bias。Fast-dVLA 用 block-wise attention 从架构上解决这个 bias。
- **Block Diffusion** (Arriola et al., 2025, https://arxiv.org/abs/2503.09573)：NLP 上的 block-AR-diffusion hybrid，但严格 block 间串行。Fast-dVLA 用 diffusion forcing 解开这个限制。
- **Diffusion Forcing** (Chen et al., NeurIPS 2024, https://arxiv.org/abs/2407.01392)：video diffusion 的异步 denoise 思想。Fast-dVLA 把它引入 action 生成。
- **Fast-dLLM via Discrete Diffusion Forcing** (Wang et al., ICLR 2026, https://openreview.net/forum?id=t5uLZSRjhF)： inspires Fast-dVLA 的 asymmetric distillation。

### 9.2 在 VLA 加速谱系中

- **Pruning 类**：MoLe-VLA (Mixture-of-Layers), EfficientVLA, ADP, LightVLA
- **Early-exit 类**：Deer-VLA, CEED-VLA
- **Caching 类**：VLA-Cache, CronusVLA
- **量化类**：BitVLA (1-bit), QVLA
- **轻量架构**：TinyVLA, SmolVLA, Flower (950M)
- **Fast-dVLA 是首个针对 dVLA inference 加速的系统工作**

### 9.3 在 VLA 范式竞争中

- AR VLA（OpenVLA, $\pi_0$-FAST）：慢但成熟
- Flow-matching VLA（$\pi_0$, $\pi_0.5$, GR00T-N1）：当前 SOTA 性能 + 速度平衡
- dVLA：理论优势（统一 multimodal, 保留 VLM prior）但速度瓶颈

Fast-dVLA 让 dVLA 在速度上对齐 flow-matching，同时保留 dVLA 范式优势。这是范式竞争的关键 turn。

---

## 10. 直觉总结：为什么这套设计能 work

### 10.1 为什么 block-wise attention 不掉性能

paper 隐含的论点：dVLA 的 bidirectional attention 实际上是 over-engineered——模型本身就有 left-to-right tendency（来自 AR VLM 初始化 + action 时序性）。所以把 global bidirectional 换成 block-causal 几乎不损失信息，反而让训练更稳定（显式 temporal causality 约束）。

类比：BERT bidirectional 在很多任务上并不比 GPT causal 显著好，因为数据本身有方向性。

### 10.2 为什么 diffusion forcing 能并行

传统 block diffusion 卡在"必须等前 block 完成才能开始后 block"，因为训练时所有 block 同 noise level，前 block 没完成 → 后 block 没条件 → 无法开始。

diffusion forcing 改成不同 noise level，后 block 训练时见到的是"前 block 部分干净 + 自己高噪声"的混合状态。模型学到这种异步生成能力，推理时就可以 overlap。

### 10.3 为什么 asymmetric distillation 比 from scratch 快 10×

因为 student 和 teacher 同架构，只是 attention pattern 不同。LoRA 只需学"如何在缺未来 context 下逼近 teacher 在 full context 下的预测"。这是一个低维映射问题，远比从头学 action 生成容易。

直觉类比：把一个 bidirectional translator 蒸馏成 left-to-right translator，比训一个 left-to-right translator from scratch 快得多，因为大部分语言知识是共享的。

### 10.4 为什么双状态机制有必要

如果一加入 pipeline 就激进 decode 后 block，前 block 还在变 → 后 block 基于错误前缀生成 → 错误累积。

如果一直谨慎 decode，pipeline 利用率低 → 没加速。

双状态折中：semi-activated 谨慎（只 commit high-conf token），等前 block "基本完成"（>4/7）后再激进。这本质是一种 speculative decoding 的 confidence-gating。

---

## 11. 局限与可能延伸

paper 没有充分讨论的：
1. **Block size k=7 是手动调的**，能否自动 search？类似 Neural Architecture Search。
2. **Asymmetric distillation 的 teacher-student gap 上界**没理论分析。能否证明 KL 收敛性？
3. **Real-world 30 Hz 是在什么 hardware 上**？没说明 GPU/compute。如果需要 H100 才能 30 Hz，对社区仍难复现。
4. **与 consistency model 的结合**：能否让每个 block 的 denoise step 也用 consistency distillation 减到 1-4 步？这会进一步加速。
5. **与 speculative decoding 的关系**：双状态机制本质是 confidence-gated speculative decoding，能否合并传统 spec decoding 的 verify 机制？
6. **Multi-task scaling**：只在 LIBERO/CALVIN/SIMPLER 任务上验证，能否 scale 到 Open-X-Embodiment 级别 multi-task？
7. **Action chunk size 与 block size 关系**：paper 用 action chunk size 8/5/10 但 block size 7，这之间不齐是否带来 suboptimality？

---

## 12. 相关联想（可能有用的延伸阅读）

- **LLaDA** (Nie et al., 2025, https://arxiv.org/abs/2502.09992): 大规模 masked diffusion LLM，paper 之一基础
- **Dream 7B** (Ye et al., 2025, https://arxiv.org/abs/2508.15487): 7B diffusion LLM backbone
- **MDT** (Reuss et al., 2024, https://arxiv.org/abs/2410.10865): Multimodal Diffusion Transformer
- **Diffusion Forcing** (Chen et al., NeurIPS 2024, https://arxiv.org/abs/2407.01392): 异步 denoise 原始论文
- **Consistency Models** (Song et al., 2023, https://arxiv.org/abs/2303.01469): 一步生成思路，可叠加到 Fast-dVLA
- **Speculative Decoding** (Leviathan et al., 2023, https://arxiv.org/abs/2211.17192): draft-verify 范式，与双状态机制有精神关联
- **Jacobi Decoding** (Santilli et al., 2023, https://arxiv.org/abs/2305.13270): PD-VLA 用的并行解码思路
- **π0.5** (Physical Intelligence, https://arxiv.org/abs/2504.16054): SOTA flow-matching VLA 对比基线
- **UniVLA** (Wang et al., 2025, https://arxiv.org/abs/2506.19850): unified VLA 思路
- **FlowVLA** (Zhong et al., 2025, https://arxiv.org/abs/2508.18269): visual chain-of-thought + flow matching
- **SimplerEnv** (Li et al., 2024, https://arxiv.org/abs/2405.05941): real-to-sim benchmark
- **LIBERO** (Liu et al., 2023, https://arxiv.org/abs/2306.03310): 标准 VLA benchmark
- **CALVIN** (Mees et al., 2022, https://arxiv.org/abs/2112.03227): long-horizon benchmark
- **MoE for VLA** (MODE, Reuss et al., https://arxiv.org/abs/2504.09095): mixture of expert denoisers

---

## 13. 实际复现要点

如果有人想复现：

1. **选 base dVLA**：Dream-VLA 或 DD-VLA 都已开源
2. **LoRA distillation**：rank=32，teacher 关 LoRA / student 开 LoRA
3. **Block size**：选 action dimensionality 的整数倍
4. **Diffusion schedule**：单调递增 $t_1 < \cdots < t_N$，可参考 Diffusion Forcing 原文
5. **Training steps**：~3-4k 步（base model finetune budget 的 1/5~1/8）
6. **Inference hyperparams**：$\tau_{\text{add}}=2/7$, $\tau_{\text{act}}=4/7$, $\tau_{\text{conf}}=0.5$, $n=2$（log2 radical decoding）
7. **Attention mask 实现**：block-wise causal，需自定义 attention mask
8. **KV cache 管理**：block 完成后 cache K/V，新 block 的 query 复用

工程坑点：
- Attention mask 必须区分 intra-block（bidirectional）和 inter-block（causal）
- KV cache 数据结构需支持 block-level 增量 append
- Pipeline 调度需在 GPU kernel 之外做 dynamic control flow（半 Python 半 CUDA）

---

## 14. 总结

Fast-dVLA 的核心 contribution 是一个相当 elegant 的工程组合：
1. **Observation**（bidirectional dVLA 的隐式 AR 倾向）
2. **Architecture**（block-wise causal attention 启用 KV cache）
3. **Training objective**（diffusion forcing 启用 inter-block parallel）
4. **Distillation**（asymmetric KD 让训练成本降到 1/10）
5. **Inference scheduling**（双状态 pipelined decoding）

整体类似一个"把 bidirectional 模型蒸馏成 partially autoregressive + 异步 diffusion"的 recipe。结果是 dVLA 速度追上 flow-matching VLA，real-world 30 Hz 落地。

这工作体现了一种趋势：**discrete diffusion LLM/VLA 的工程优化空间还很大**，目前 bidirectional 全注意力是 over-engineered，block-causal + asynchronous 是更精细的设计。后续可能看到更多类似 fast-dLLM, Fast-dVLA, Block Diffusion 这类把"理论 elegant 的 bidirectional diffusion"压缩到"实用 fast"的工作。

如果你想 build intuition 关于这套方法，建议先看 Block Diffusion 原文理解 block-AR-diffusion hybrid，再看 Diffusion Forcing 理解异步 noise schedule，最后回 Fast-dVLA 看如何把两者融合并加 asymmetric distillation。这是一个典型的"组合多个 prior trick 形成新方法"的工程论文，每个组件单独看都不新，组合起来 solve 了一个真实的 bottleneck。
