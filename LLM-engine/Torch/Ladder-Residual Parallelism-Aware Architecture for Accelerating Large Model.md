---
source_pdf: Ladder-Residual Parallelism-Aware Architecture for Accelerating Large
  Model.pdf
paper_sha256: a62852910243572082f432e52fb845aac0dd3f75b6dea28fafbc3ed355b7b0be
processed_at: '2026-08-05T11:39:29-07:00'
target_folder: LLM-engine/Torch
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话讲讲 Ladder Residual

## 一句话版

Transformer 在多 GPU 上跑的时候，GPU 之间要不停地 AllReduce 交换中间结果，这个通信很慢。这篇 paper 说：**我把每层的输入改成用"上一层之前"的 activation，这样当前层的 AllReduce 就可以跟下一层的计算并行跑了**，完全不用改底层 kernel，纯改架构就能快 30%。

---

## 问题是什么

你有一个 70B 的 Llama，单卡放不下，用 8 张卡 Tensor Parallelism 跑。

TP 的逻辑很简单：把每层的 weight 矩阵切成 8 份，每张卡算自己那份，算完之后 **AllReduce 求和**，才能得到完整的输出，再传给下一层。

问题在于：每个 transformer block 有 attention 和 MLP 各一次 AllReduce，一个 70B 模型有 80 层，所以 forward 一次要跑 **160 次 AllReduce**。即使有 NVLink，这些 AllReduce 加起来占了 inference latency 的 **38%**。GPU 在等 AllReduce 的时候，SM 是空转的。

之前的人怎么解决呢？要么写 fused kernel（把 AllReduce 和 matmul 融在一起，用 tile 化的方式 overlap），要么上更贵的硬件（Blackwell 的 72 卡 NVLink domain）。这些都是"系统层"的优化。

这篇 paper 说：**等一下，我们能不能改一下 model architecture，让通信自然地就能 overlap 上？**

---

## 核心 trick：用"旧" activation

### 标准 Transformer 的数据流

看一个 transformer layer 的 forward：

$$
x_i = \text{AllReduce}(h_i(x_{i-1})) + x_{i-1}
$$

变量解释：
- $x_{i-1}$ 是第 $i-1$ 层的 residual stream（就是累积的 activation）
- $h_i$ 是第 $i$ 层的子模块（attention 或 MLP），分片在 8 张卡上
- AllReduce 把 8 张卡的 partial sum 求和回来
- 加回 $x_{i-1}$ 得到 $x_i$

**阻塞点在哪？** 下一层 $h_{i+1}$ 要用 $x_i$ 当输入，但 $x_i$ 要等 AllReduce 跑完才有。所以 AllReduce 期间 GPU 空转。

### Ladder Residual 的改动

只改一个地方：$h_{i+1}$ 的输入从 $x_i$ 变成 $x_{i-1}$：

$$
x_{i+1} = \text{AllReduce}(h_{i+1}(x_{i-1})) + x_i
$$

就这一行。残差加法那一侧没变，仍然加到 $x_i$ 上。

**为什么这样就能 overlap？** 因为 $x_{i-1}$ 在**两层之前就已经 AllReduce 完了**，它已经躺在 GPU 的显存里准备好。所以 $h_{i+1}$ 可以立刻开始计算，根本不用等 $h_i$ 的 AllReduce。同时 $h_i$ 的 AllReduce 在网络上传，跟 $h_{i+1}$ 的 compute 时间上重叠。

打个比方：你在厨房做菜。标准 transformer 是——炒完菜 A，等服务员把菜 A 端给客人，客人吃完反馈，你才开始炒菜 B。Ladder residual 是——炒完菜 A 就直接开始炒菜 B（用上上个客人的反馈），同时服务员在端菜 A，两件事并行。

---

## 为什么用"旧" activation 不会崩

这是整篇 paper 最妙的地方。你可能觉得：用两步前的 activation 当输入，模型不就错了吗？

关键观察来自 Deja Vu 那篇 paper（https://arxiv.org/abs/2310.17157）：**Transformer 每层对 residual stream 的改动，相对于 residual stream 本身来说，非常小**。

具体来说，$\|h_i(x_{i-1})\|$ 比 $\|x_{i-1}\|$ 小 1-2 个数量级。这意味着 $x_i$ 和 $x_{i-1}$ 其实差别很小。所以 $h_{i+1}(x_{i-1})$ 跟 $h_{i+1}(x_i)$ 的差别，大概是 $\|h_{i+1}\| \times \|x_i - x_{i-1}\| / \|x_i\|$，这是一个**二阶小量**——两个都小的东西乘在一起。

从 ODE 角度看更直觉：ResNet 是 $\dot{x} = f(x, t)$ 的离散化，标准 transformer 是 forward Euler step。Ladder 相当于把导数采样点延后了一个 step——类似 Adams-Bashforth 方法。只要 $f$ 变化平缓（残差网络的特点），这个延后几乎无害。

所以本质上，Ladder Residual 用了一个 transformer 自带的归纳偏置（"每层改动小"），换来了通信和计算的 overlap。免费的午餐。

---

## 实现长什么样

Algorithm 1 的伪代码，我翻译成人话：

```
跑一层 Ladder Transformer：
  1. 等上一层的 attention AllReduce 完（wait）
  2. 把上一层的 attention 结果加进 residual
  
  3. 用当前 residual 做 LayerNorm → Attention
  4. 发起 AsyncAllReduce（不等，立刻返回一个 handle）
  
  5. 等上一层的 MLP AllReduce 完（wait）
  6. 把上一层的 MLP 结果加进 residual
  
  7. 用当前 residual 做 LayerNorm → MLP
  8. 发起 AsyncAllReduce（不等，立刻返回 handle）
  
  9. 把所有东西传给下一层
```

关键点：步骤 4 发起 AllReduce 之后，立刻去做步骤 5-7。步骤 4 的 AllReduce 在 NCCL 的独立 CUDA stream 上跑，步骤 5-7 的 compute 在 default stream 上跑，两者硬件层面并行。

每层传递 5 个东西给下一层：residual stream、上一层 attention 的 partial sum、上一层 MLP 的 partial sum、以及这两个的 AllReduce handle。这就是为什么 Figure 1 右图里有那种"阶梯状"的连接——信息流跨了两层。

PyTorch 里实现这个非常简单，就是 `dist.all_reduce(..., async_op=True)` 拿一个 work handle，然后该干嘛干嘛，需要的时候 `work.wait()`。完全不用碰 CUDA kernel。这点跟 Flux（https://arxiv.org/abs/2406.06858）需要手写 fused kernel 形成鲜明对比。

---

## 实验数据怎么说

### Inference 加速（Table 1）

在 Llama 架构的各种 size 上测，TP=8，1024 prompt + 512 generation：

| Model | P2P 开（NVLink） | P2P 关（模拟慢互联） |
|-------|:-:|:-:|
| 1B | 1.56× | 1.39× |
| 8B | 1.46× | 1.40× |
| 70B | 1.29× | 1.59× |
| 405B | 1.31× | 1.57× |

两个 trend：
1. 通信越慢，Ladder 越值。P2P 关掉模拟跨节点，加速比能到 1.6×
2. P2P 开的情况下，模型从 8B → 70B，加速比从 1.46 降到 1.29。因为大模型 compute/communication ratio 更高，AllReduce 占比下降。但 405B 跨两节点又升回 1.31×，因为跨节点通信又变贵了

### 405B 跨节点（Figure 3）

这个实验最有实际意义。405B 单节点 8 卡放不下，必须跨节点。传统做法是节点内 TP + 节点间 PP，但 PP 在 batch=1 时一半 GPU 空转。Ladder 让跨节点 TP 变得可行——30%+ 的加速 across 各种 batch size。

### Prefill vs Decode（Table 2）

70B 模型分解来看：

| 阶段 | Standard | Ladder 加速 |
|------|:-:|:-:|
| Prefill（P2P开） | baseline | 5.78% |
| Decode（P2P开） | baseline | 23.71% |
| Decode（P2P关） | baseline | 37.71% |

Prefill 阶段加速很小，因为 prefill 是 compute-bound（一次处理 1024 个 token，matmul 很大），AllReduce 占比本来就低。Decode 阶段才是 Ladder 的主场——每次只生成 1 个 token，compute 很小，AllReduce 占比高，overlap 省下的都是真金白银。

---

## 从 scratch 训练效果

1.2B 和 3.5B 模型在 FineWeb-edu 上训 100B tokens（https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu）：

| 模型 | 平均准确率 | Wikitext PPL |
|------|:-:|:-:|
| Standard-1.2B | 59.98 | 18.54 |
| Ladder-1.2B | 58.92 | 18.42 |
| Standard-3.5B | 64.11 | 14.48 |
| Ladder-3.5B | 62.91 | 14.90 |

1.2B 几乎完全持平。3.5B 略差一点点（1.2 个百分点准确率，0.42 PPL），但考虑到换来 55%+ 的 inference 加速，这点损失完全可接受。

---

## 最让我兴奋的实验：Llama-3.1-8B 后训练转换

他们把 Llama-3.1-8B-Instruct 的**上半 16 层**转成 Ladder 结构，然后用 Infinity-Instruct 的 3B tokens SFT 2 个 epoch（https://arxiv.org/abs/2410.05944）：

| 模型 | MMLU | GSM8K | HumanEval+ | 平均 |
|------|:-:|:-:|:-:|:-:|
| 原版 Llama-3.1-8B-Instruct | 68.14 | 84.99 | 60.40 | 56.11 |
| 直接转换（零样本） | 63.19 | **10.54** | 30.50 | 41.65 |
| 3B tokens SFT 后 | 67.33 | **86.81** | 60.51 | **57.61** |

注意几个点：

1. 零样本直接转换，GSM8K 从 84.99 崩到 10.54。生成式任务对 representation shift 最敏感，因为需要精确的多步推理
2. 只用 3B tokens 轻量 SFT，性能完全恢复，平均分甚至**超过原版**（57.61 vs 56.11）
3. 对比之下，Mamba-in-the-Llama（https://arxiv.org/abs/2408.15237）把 Llama 转 Mamba 需要 50B tokens 才能恢复。Ladder 的 representation shift 远比架构替换小

**为什么只转上半 16 层？** 因为 transformer 的下层在做 embedding → semantic 的 rapid transformation，activation 变化快，staleness 损失大。上层已经进入 semantic 空间，activation 变化慢，staleness 几乎无害。这本身就是一个很直觉的发现。

这个实验说明 Ladder 可以作为 **drop-in acceleration**——拿一个已经训好的模型，轻量 retraining，就能加速 21%。工程价值极大。

---

## "30% 更大的 Ladder" 对比

这个对比很 clever。70B Ladder 比 Standard 快 30%，那如果让 Ladder 多 30% 参数量，同样 wall-clock 下比呢？

| 对比 | 准确率 | PPL | tokens/sec |
|------|:-:|:-:|:-:|
| Standard-1.2B vs Ladder-1.5B | 60.33 > 59.98 | 17.47 < 18.54 | 1277 > 1008 |
| Standard-3.5B vs Ladder-4.5B | 64.21 > 64.11 | 14.05 < 14.48 | 1217 > 949 |

同等 wall-clock 下，Ladder 模型可以做得更大、效果更好、还更快。这是 inference-aware scaling 的范例——不是单纯"加速"，是"在同等时间预算下做更强的模型"。

---

## 我觉得这个工作为什么重要

### 1. Architecture-hardware co-design 的范例

过去几年，大家解决 LLM serving 瓶颈的思路是：要么更快的硬件（NVLink、HBM、Blackwell），要么更聪明的系统（kernel fusion、vLLM、paged attention）。这篇 paper 指出第三条路：**改架构本身**，让架构天然适配分布式系统的通信模式。

### 2. 利用 transformer 的归纳偏置

"每层改动小"这个现象，Deja Vu 发现了用来做 sparsity，LayerSkip 用来做跳层。Ladder 是第一个把它转化成 communication overlap 的。这说明 transformer 内部还有很多没被 exploit 的结构。

### 3. 工程上极其轻量

不用写 CUDA kernel，不用改 NCCL，不用换硬件。纯 PyTorch 层面改一行 forward 逻辑，用 `async_op=True` 的 AllReduce handle。跟 CUDA Graph、torch.compile、Pipeline Parallel、FSDP 都兼容。这种低侵入性在 LLM infra 复杂度爆炸的今天特别有价值。

### 4. Staleness 的哲学

Ladder 本质上是把 async SGD 的思想——"用 stale gradient 更新参数"——搬到了 forward pass 的 activation 上。staleness = 1 layer，有界，收敛性有保障。这个思想可以推广：能不能 staleness = 2？能不能在 backward 也做？能不能在 attention 的 KV cache 上做？都是 open question。

### 5. 跟 ODE 数值方法的深层连接

标准 transformer = forward Euler。Ladder = 延迟一步的显式格式。如果能引入 Adams-Bashforth 或 Runge-Kutta 的思路，可能设计出既允许更大 staleness 又保持稳定性的架构。这把 numerical analysis 和 deep learning architecture 设计连起来了，很优雅。

---

## 局限和我想看到的方向

1. **staleness > 1 没试**：如果 $h_{i+k}$ 用 $x_{i-1}$，给 overlap 更多空间，能不能在更慢的互联上拿到更大加速？代价是 representation shift 更大，需要更多 retraining。这条曲线值得画出来。

2. **跟 attention kernel 的 overlap**：目前 overlap 主要是 MLP 的 matmul。Flash Attention 内部是 softmax+matmul 交织，跟 AllReduce overlap 工程上更难。如果能做到，decode 阶段加速还能再上一个台阶。

3. **Dynamic batching 场景**：paper 都是固定 prompt+gen 长度。实际 serving 是 dynamic batch，不同 request 的 AllReduce size 不一样，overlap 行为会不会被打乱？

4. **训练 backward 的 overlap**：paper 说训练时 FSDP 已经能 overlap ReduceScatter 所以帮助小。但如果用 TP+SP 训练（Megatron 风格），backward 的 AllReduce 同样阻塞，Ladder 应该能帮。

5. **渐进式 staleness**：下层 staleness=0，上层 staleness=1 或 2。这样下层保持精度，上层多 overlap。需要实验验证。

6. **跟 speculative decoding 的结合**：spec decoding 在 token 维度做异步，Ladder 在 layer 维度做异步。两者正交，应该可以叠加。

---

## 参考资料

- 主 paper: https://arxiv.org/abs/2410.05944
- gpt-fast 实现: https://github.com/pytorch-labs/gpt-fast
- Deja Vu（staleness 先验）: https://arxiv.org/abs/2310.17157
- Flux（kernel overlap baseline）: https://arxiv.org/abs/2406.06858
- Megatron-LM（TP 原始）: https://arxiv.org/abs/1909.08053
- Mamba-in-the-Llama（后训练对比）: https://arxiv.org/abs/2408.15237
- Cross-Layer Attention: https://arxiv.org/abs/2405.12981
- PaLM（parallel attn/mlp）: https://arxiv.org/abs/2204.02311
- Speculative Decoding: https://arxiv.org/abs/2211.17192
- Hogwild!（异步优化）: https://arxiv.org/abs/1109.1033
- Highway Networks: https://arxiv.org/abs/1505.00387
- DenseNet: https://arxiv.org/abs/1608.06993
- FineWeb-edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Llama 3: https://arxiv.org/abs/2407.21783

---

# Ladder-Residual: 详解与直觉构建

## 1. Paper 核心思想一句话总结

通过把 residual stream 的"加法路径"和"模块输入路径"解耦，让 $h_{i+1}$ 吃两步前的 stale activation $x_{i-1}$，从而 $x_i$ 的 AllReduce 可以和 $h_{i+1}$ 的 compute overlap 起来——这是一个 **architecture-hardware co-design** 的思路，在 PyTorch 层面就能拿到 30% 左右的端到端 inference 加速，完全不需要写 NCCL kernel fusion。

Paper link: https://arxiv.org/abs/2410.05944 (Muru Zhang, Tri Dao et al., 2024)
Tri Dao 的 group page: https://tridao.ai/
相关 blog (gpt-fast): https://github.com/pytorch-labs/gpt-fast

---

## 2. 问题背景：为什么 TP 有通信瓶颈

### 2.1 Tensor Parallelism 的基本结构

TP（Shoeybi et al., Megatron-LM, 2020, https://arxiv.org/abs/1909.08053）把一个两层 matmul 序列 $(XA)B$ 分片：

- $A$ 沿 output dim split：$A = [A_1, A_2]$
- $B$ 沿 input dim split：$B = \begin{bmatrix} B_1 \\ B_2 \end{bmatrix}$
- 在每个 GPU 上独立算 $(X A_k) B_k$，最后 AllReduce 求和

一个 Transformer block 包含一个 attention（QKV + proj）和一个 MLP（gate/up + down），每个都是 column-then-row 的两段 matmul，所以**每层产生 2 次 AllReduce**。

### 2.2 通信占比实测

Paper 里测出 70B model 在 TP=8、batch=4 时，AllReduce 占 end-to-end latency 的 **38%**（NVLink 全开）。一旦 P2P 关掉（模拟跨 node 慢互联），AllReduce 占比可以飙到 **>50%**。

这是很 fundamental 的瓶颈，跟 NVLink 带宽上限、PCIe lane 限制、数据中心 power density 都有关，硬件层面解决非常昂贵（比如 Blackwell 的 72-GPU NVLink domain）。

### 2.3 之前方法的局限

- **Flux (Chang et al., 2024)** https://arxiv.org/abs/2406.06858 ：把 matmul tile 化，fused AllGather + matmul kernel，需要手写 CUDA kernel
- **CoCoNet (Jangda et al., 2022)** https://arxiv.org/abs/2105.05720 ：DSL + compiler，不通用，跟 PyTorch/JAX 生态兼容性差
- **Sequence Parallelism (Megatron)** https://arxiv.org/abs/2205.05198 ：把 LayerNorm 之外的 activation 分片，但仍然需要 AllGather
- **Parallel Attention/MLP (PaLM)** https://arxiv.org/abs/2204.02311 ：融合 QKV 和 gate/up，attention 和 MLP 并行算 → 通信减半，但仍然阻塞

这些方法本质上都在"通信这件事"上做文章，没碰 architecture 本身。

---

## 3. Ladder Residual 的核心 trick

### 3.1 标准 Transformer 的递推

$$
\begin{aligned}
x_i^* &= h_i(x_{i-1}) \\
x_i &= \text{AllReduce}(x_i^*) + x_{i-1} \\
x_{i+1}^* &= h_{i+1}(x_i) \\
x_{i+1} &= \text{AllReduce}(x_{i+1}^*) + x_i
\end{aligned} \tag{Eq.1}
$$

变量含义：
- $x_i$：第 $i$ 个 block 之后的 residual stream（完整的、跨所有 GPU 一致的 activation）
- $x_i^*$：partial sum，每个 GPU 上自己算出来的部分结果，需要 AllReduce 才能恢复完整 $x_i$
- $h_i$：第 $i$ 个 block 的子模块（attention 或 MLP），分片在各 GPU 上
- 下标 $i$ 表示 block 的层序号

**关键阻塞点**：$h_{i+1}$ 必须等 AllReduce($x_i^*$) 完成才能开始，因为它需要 $x_i$ 作为输入。这个数据依赖让 AllReduce 成为串行的瓶颈。

### 3.2 Ladder Residual 的递推

$$
\begin{aligned}
x_i^* &= h_i(x_{i-2}) \\
x_i &= \text{AllReduce}(x_i^*) + x_{i-1} \\
x_{i+1}^* &= h_{i+1}(x_{i-1}) \\
x_{i+1} &= \text{AllReduce}(x_{i+1}^*) + x_i
\end{aligned} \tag{Eq.2}
$$

**唯一改动**：$h_{i+1}$ 的输入从 $x_i$ 改成 $x_{i-1}$。残差加法那一行 $\text{AllReduce}(x_i^*) + x_{i-1}$ 完全保持不变。

这意味 $h_{i+1}$ 的输入是"前一个 block 没经过更新前的 residual stream"，是 stale 的，但**这个 stale 数据在 $h_i$ 开始计算时就已經 AllReduce 完成了**（因为它是 $h_{i-1}$ 那一轮的 AllReduce 结果），所以可以立刻拿来用，不需要等当前 $h_i$ 的 AllReduce。

### 3.3 为什么这能 work——三个层次的直觉

#### 直觉 A：残差网络的 ODE 视角

ResNet 可以看成离散化的 ODE $\dot{x}(t) = h(x(t), t)$。标准 Transformer 是显式 Euler step：$x_{i+1} = x_i + h_i(x_i)$。Ladder Residual 相当于把"导数采样点"延迟了一个 timestep，类似 multi-step method 的延迟显式格式。

只要 $h$ 相对于 $x$ 的 norm 很小（这点 Liu et al. 在 Deja Vu https://arxiv.org/abs/2310.17157 里观测到了：每层 update 比起 residual stream 的 norm 平均小 1-2 个数量级），这个延迟就只引入 $O(\|\Delta h\|)$ 的扰动，远比残差本身的 $O(\|h\|)$ 小。

#### 直觉 B：异步优化的视角

这跟 asynchronous SGD / Hogwild! 是亲戚。在 async SGD 里，worker 用 stale gradient 更新参数，只要 staleness 有界，收敛性可证。Ladder Residual 是把同样的思想用到 forward pass 的 activation 上——staleness = 1 layer。

#### 直觉 C：信息流没有断

注意残差加法那一行没变，仍然是 $\text{AllReduce}(x_i^*) + x_{i-1}$。这意味着每个 $h_{i+1}$ 虽然吃的是 $x_{i-1}$，但它**产生的更新 $h_{i+1}(x_{i-1})$ 仍然加回到完整的 $x_i$ 上**，所以信息没有真正丢失，只是延后了一个 step 被混合进 residual stream。从表达能力上看，这跟标准 transformer 几乎等价。

---

## 4. Architecture 图解析

### 4.1 Figure 1 拆解

**左图（Standard Transformer block）**：
- residual 流过来 → AttentionNorm → Attention → AllReduce → 加回 residual → MLPNorm → MLP → AllReduce → 加回 residual → 下一层
- 每次加回前必须等 AllReduce，AllReduce 期间 GPU 空转

**右图（Ladder Residual block）**：
- residual $r$ 流过来，但同时有一个"两步前的 stale residual"分支
- $h_i$ 模块直接吃 stale input（图中蓝色虚线表示 residual 跨越两层）
- $h_i$ 的输出 AllReduce 与 $h_{i+1}$ 的 compute 时间上重叠

图中蓝色 edge 是 residual connection，注意 Ladder 版本里 residual 跳跃了两个 module 而不是一个——这是"ladder"（阶梯）名字的由来：每层残差都踩在前一阶而不是当前一阶上。

### 4.2 Algorithm 1 的逐行解析

```
function LAYER(residual, attn_out, mlp_out, attn_work, mlp_work):
  1. attn_work.wait()       # 等"前一层 attention"的 AAR 完成
  2. residual += attn_out   # 把前一层 attention 加进残差流
  
  3. attn_out = AttentionNorm(residual)   # 用当前 residual（已经吸收了前一层 attn）
  4. attn_out = Attention(attn_out)
  5. attn_out, attn_work = AAR(attn_out)   # 异步发起 AllReduce，立刻返回 handle
  
  6. mlp_work.wait()        # 等"前一层 MLP"的 AAR 完成
  7. residual += mlp_out    # 把前一层 MLP 加进残差流
  
  8. mlp_out = MLPNorm(residual)
  9. mlp_out = MLP(mlp_out)
  10. mlp_out, mlp_work = AAR(mlp_out)
  
  11. return residual, attn_out, mlp_out, attn_work, mlp_work
```

**关键直觉**：每层有 5 个状态在跨层传递：
- `residual`：累积的残差流
- `attn_out`：前一层 attention 的 partial sum（还在等 AllReduce）
- `mlp_out`：前一层 MLP 的 partial sum
- `attn_work`, `mlp_work`：前一层 AAR 的 NCCL handle

这一层在算 attention 时，前一层的 mlp AllReduce 正在网络上飞——这就是 overlap。

### 4.3 与异步 pipeline 的对比

| 方面 | 1F1B Pipeline | Interleaved Pipeline | Ladder Residual |
|------|---------------|---------------------|-----------------|
| 并行维度 | batch | micro-batch | activation layer |
| 重叠对象 | 前后向 compute | 不同 layer 的 compute | compute 和 TP AllReduce |
| staleness | 0 | 0 | 1 layer |
| 适用 | 训练 | 训练 | inference/training |

Ladder Residual 的 staleness=1 是"层级别"的，比 async pipeline 的 micro-batch staleness 更小、更可控。

---

## 5. 实验数据精读

### 5.1 Table 1：各 size 的 inference speedup

| Model | P2P disabled | P2P enabled |
|-------|-------------:|------------:|
| 1B | 1.39× | 1.56× |
| 3B | 1.50× | 1.57× |
| 8B | 1.40× | 1.46× |
| 34B | 1.47× | 1.44× |
| 70B | 1.59× | 1.29× |
| 176B | 1.54× | 1.35× |
| 405B | 1.57× | 1.31× |

**两个 trend**：
1. P2P disabled（模拟跨节点慢互联）下 speedup 更高——通信越慢，Ladder 越值
2. P2P enabled 下，模型从 8B → 70B，speedup 从 1.46× 降到 1.29×——因为大模型的 compute/communication ratio 更高，AllReduce 占比下降

**405B 的 30%+ 加速**意味着跨节点 TP 现在变得可行，而不用依赖 PP+DP 来掩盖跨节点延迟。

### 5.2 Table 2：latency 分解

70B model, batch=1, TP=8：

| Variant | Prefill improvement | Decode improvement | Token/sec improvement |
|---------|---:|---:|---:|
| UpperBound (P2P=1) | 30.54% | 30.00% | 42.90% |
| Parallel (P2P=1) | 5.42% | 18.04% | 21.75% |
| **Ladder (P2P=1)** | **5.78%** | **23.71%** | **30.79%** |
| UpperBound (P2P=0) | 35.84% | 52.71% | 110.7% |
| Parallel (P2P=0) | 14.92% | 28.73% | 40.07% |
| **Ladder (P2P=0)** | **6.94%** | **37.71%** | **59.87%** |

**重要观察**：
- UpperBound 是把所有 AllReduce 直接删掉的极限，Ladder 在 decode 阶段几乎触及 UpperBound 的 80%（P2P=0 情况下）
- Prefill 阶段 Ladder 加速很小（5.78%），因为 prefill 是 compute-bound，通信占比本来就低
- **Decode 阶段**才是 Ladder 的主场——decode 是 memory/communication bound，每个 token 都要全层 AllReduce

### 5.3 训练 from scratch 结果（Table 3）

1.2B 和 3.5B 在 FineWeb-edu 上训 100B tokens，对比 Standard / Parallel / Ladder：

| Variant (1.2B) | Avg acc | Wikitext PPL |
|---|---:|---:|
| Standard | 59.98 | 18.54 |
| Parallel | 58.75 | 18.95 |
| **Ladder** | **58.92** | **18.42** |

1.2B 几乎完全持平。3.5B 略差一点点（62.91 vs 64.11），PPL 也只高 0.42。

### 5.4 Llama-3.1-8B 后训练 adaptation（Table 4）

最 interesting 的实验：把 Llama-3.1-8B-Instruct 的 **上层 16 层** 替换成 Ladder 结构，然后用 Infinity-Instruct 的 3B tokens SFT 2 epochs：

| Variant | MMLU | GSM8K | HumanEval+ | Avg |
|---|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct (原版) | 68.14 | 84.99 | 60.40 | 56.11 |
| Hybrid-16L-zeroshot (零样本转换) | 63.19 | **10.54** | 30.50 | 41.65 |
| **Hybrid-16L-retrained** | **67.33** | **86.81** | **60.51** | **57.61** |

**关键 insight**：
- 直接 zeroshot 转换会让 GSM8K 崩到 10.54（generative task 对 representation shift 最敏感）
- 只用 3B tokens SFT 就能完全恢复，**avg 甚至超过原版**（57.61 vs 56.11）
- 对比 Mamba-in-the-Llama (Wang et al., 2024, https://arxiv.org/abs/2408.15237) 需要 50B tokens 才能恢复——Ladder 的 representation shift 远比 RNN 替换小

### 5.5 Table 5：30% 更大的 Ladder 对比 Standard

这个比较特别有意思。因为 70B Ladder 比 Standard 快 30%，那如果我们让 Ladder 多 30% 参数量，比较公平吗？

| Variant | 1.2B vs 1.5B Ladder | 3.5B vs 4.5B Ladder |
|---|---|---|
| Accuracy | 60.33 > 59.98 | 64.21 > 64.11 |
| PPL | 17.47 < 18.54 | 14.05 < 14.48 |
| Tokens/sec | 1277 > 1008 | 1217 > 949 |

**结论**：同等 wall-clock 下，Ladder 模型可以做得更大、效果更好。这是 inference-aware scaling 的一个范例。

---

## 6. 实现 detail 与 NCCL stream

### 6.1 AsyncAllReduce 的实现机制

PyTorch NCCL collective 默认跑在单独的 CUDA stream 上，所以 `dist.all_reduce(..., async_op=True)` 会立刻返回一个 `Work` handle。GPU 上 compute stream 继续往后跑，AllReduce 在通信 stream 上并行进行——硬件层面 NVLink 和 SM compute 是独立资源。

调用 `work.wait()` 只是 host-side 同步，等 AllReduce 提交到通信 stream；真正的 device-side 等待发生在下次用这个 tensor 做 compute 时。但 Ladder 通过 `wait()` 强制对齐，确保 residual 累加用的是完整的 AllReduce 结果。

### 6.2 与 CUDA Graph 的兼容

Paper 用 `torch.compile(mode="reduce-overhead")` 开 CUDA graph。CUDA graph 对 NCCL stream 的处理略 tricky——需要把 NCCL op 也录进 graph，并保证 replay 时 stream 同步语义一致。gpt-fast（https://github.com/pytorch-labs/gpt-fast）已经做了这块工程，Ladder 直接复用。

### 6.3 Pipeline Parallel 兼容

在 PP 边界，需要等 AllReduce 完成后用 `batch_isend_irecv` 把 `(residual, attn_out, mlp_out)` 三个 tensor 传到下一 stage。注意要传三个，因为下一 stage 的第一层需要"前一层"的 attn_out 和 mlp_out 来做残差累加。

### 6.4 FSDP 兼容

FSDP 的 ReduceScatter 本来就可以 overlap（prefetch weights），所以 Ladder 对 FSDP 帮助小，paper 里只提了 5-7% 的训练 speedup。这是为什么 paper 主要 focus 在 inference。

---

## 7. 我的延伸思考 / 相关联想

### 7.1 跟 Highway Network / DenseNet 的关系

Ladder Residual 本质上是把"残差连接"和"模块输入连接"分开：残差连接保持 1-skip，模块输入连接变成 2-skip。这跟 Highway Network (Srivastava et al., 2015, https://arxiv.org/abs/1505.00387) 里 gating 决定 skip 比例有精神上的相似——都是显式控制信息流的"远近"。

DenseNet (Huang et al., 2017, https://arxiv.org/abs/1608.06993) 把所有前层 concat 进来，是更激进的 cross-layer 信息复用。Ladder 只 cross 1 层，但目的不同——DenseNet 是为 representation reuse，Ladder 是为通信 overlap。

### 7.2 跟 LayerSkip 的关系

LayerSkip (Elhoushi et al., 2024, https://arxiv.org/abs/2404.16710) 在 inference 时动态跳过某些层。Ladder 用"延迟一层的 activation"，跟 layer skip 有一种对偶关系：LayerSkip 是"不要这层"，Ladder 是"这层用旧输入"。

### 7.3 跟 Deja Vu / Contextual Sparsity 的关系

Deja Vu (Liu et al., 2023, https://arxiv.org/abs/2310.17157) 发现 activation 在 transformer 里变化慢，所以可以做 contextual sparsity——只激活部分 head/neuron。Ladder 利用了同样的"activation 慢变"先验，但应用方向是 communication overlap 而非 compute sparsity。两者其实可以叠加。

### 7.4 跟异步优化 / Hogwild! 的关系

Hogwild! (Recht et al., 2011, https://arxiv.org/abs/1109.1033) 证明了无锁异步 SGD 在 sparse gradient 下收敛。Ladder 是 forward pass 的异步化——可以问：能不能 staleness > 1？比如 $h_{i+k}$ 用 $x_{i-1}$，给 overlap 更多空间？Paper 没试，但理论上 staleness 越大，representation shift 越大，需要的 retraining 越多。这是一个值得探索的方向。

### 7.5 跟 Blockwise Parallel Decoding 的对偶

Blockwise parallel decoding (Stern et al., 2018, https://arxiv.org/abs/1811.03115) 和 speculative decoding (Leviathan et al., 2023, https://arxiv.org/abs/2211.17192) 在 token 维度做异步（用 draft model 预测多个 token）。Ladder 在 layer 维度做异步。两者都是用"近似"换"延迟"，且都能容忍有限 staleness。

### 7.6 跟 Mamba / SSM 的对比

Mamba (Gu & Dao, 2023, https://arxiv.org/abs/2312.00752) 是 sequential 的，本身就难做 TP（因为 state 不能简单分片）。Ladder 是 transformer 的优化，但思路——**架构先于系统**——在 SSM 上也成立：未来 SSM 的 distributed serving 也可能需要类似的 architectural 修改来 overlap 通信。

### 7.7 跟 Cross-Layer Attention 的关系

Cross-Layer Attention (Brandon et al., 2024, https://arxiv.org/abs/2405.12981) 让多层共享 K/V cache 来减内存。Ladder 让多层共享 stale activation 来减通信——都是"跨层共享"。如果两者结合，可以同时减 KV cache 内存和 AllReduce 通信。

### 7.8 为什么不从第一层就上 Ladder

在 hybrid adaptation 实验里，paper 只把**上层** 16/20 层转成 Ladder。原因是"下层"的 representation 还在 rapid change 阶段（embedding → semantic），staleness 损失大。这呼应了 ResNet 的低层特征更 sensitive 的经验。一个 follow-up 方向是"渐进式 staleness"——下层 staleness=0，上层 staleness=1 或 2。

### 7.9 跟 Neural ODE 的深层联系

如果把 transformer 看成离散化 Neural ODE $\dot{x} = f(x, t)$，那标准 ResNet 是 forward Euler step：$x_{i+1} = x_i + f(x_i, t_i) \Delta t$。Ladder 是 $x_{i+1} = x_i + f(x_{i-1}, t_i) \Delta t$，等价于 $f$ 用"前一步的 $x$"。这其实是 Adams-Bashforth 方法的最简形式（AB2 用前两步的 $f$）。更激进的 AB 方法或者 RK 方法可能能允许更大 staleness 但更稳定，这是个有意思的 future direction。

### 7.10 训练时的 overlap 潜力

Paper 提到训练时 FSDP 已经能 overlap ReduceScatter，所以 Ladder 对训练帮助小。但是！如果用 TP+SP 训练（Megatron 风格），Ladder 应该能在 backward 也帮助——backward 的 AllReduce 同样是阻塞的。Paper 没深入这快，是一个 open area。

### 7.11 跟 ReKV / CachedAttention 的潜在结合

如果上层共享 stale activation，那 stale activation 实际上可以 cache 起来跨 request 复用（类似 RadixAttention, https://arxiv.org/abs/2312.07104）。Ladder 的 stale $x_{i-1}$ 是层级别的 cache key，这跟 prefix caching 在某种抽象上同构。

### 7.12 Loss landscape 的影响

Ladder 引入的 staleness 等价于在 forward 加了一个扰动 $\delta = h_i(x_{i-1}) - h_i(x_i)$。这个扰动量级 $\|\delta\| \sim \|h_i\| \cdot \|x_i - x_{i-1}\| / \|x_i\|$。因为 $\|h_i\| / \|x_i\| \sim 0.1$，$\|x_i - x_{i-1}\| / \|x_i\| \sim 0.1$，所以 $\|\delta\| / \|x_i\| \sim 0.01$——二阶小量。这解释了为什么从 scratch 训练几乎没差。但如果模型深度增加（比如 200+ layer），staleness 1 也可能累积起来。这给"分层 staleness"提供了理论依据。

---

## 8. 局限与开放问题

1. **Pre-fill 阶段帮助有限**：因为 prefill 是 compute-bound，AllReduce 占比低。Ladder 主要救 decode。
2. **Attention kernel 难以 tile 化**：论文 overlap 的 compute 主要是 MLP 的 matmul，attention 部分（特别是 flash-attn 的 softmax+matmul）跟 AllReduce 的 overlap 工程上更难。
3. **Hybrid adaptation 上限**：只能转上半部分层，转太多 zeroshot 性能崩。需要 distillation 之类的更聪明 adaptation。
4. **staleness > 1 没试**：会不会让 cross-node 慢互联下加速更大？需要实验。
5. **跟 ring-attention 的关系**：sequence parallelism 也有 AllReduce 瓶颈，Ladder 思路能不能套用？
6. **小 batch + KV cache prefill 的混合场景**：paper 都是固定 prompt+gen，实际 serving 是 dynamic batch，overlap 行为会不会被打乱？

---

## 9. 总结性直觉

这篇 paper 给我一个很深的 impression：**很多"系统瓶颈"其实可以靠"架构先验"绕过去**。Ladder 之所以能 work，是因为 transformer 的残差结构本身就隐含了"层间变化慢"这个先验——这个先验被 Deja Vu 发现过，被 layer skip 用过，但 Ladder 是第一个把它转化成"通信 overlap"的工程化方案的。

更深一层的直觉：**任何带残差的 model，只要每层扰动相对残差小，都可以做这个 trick**。Diffusion model 的 UNet、ResNet-152、ViT-Huge 都是候选。Tri Dao 的风格一贯如此——找到数学上的"近似可交换性"，然后用工程手段 exploit 它（看 FlashAttention 也是这个路数）。

Paper 本身的工程落地也很干净：不需要 custom kernel，纯 PyTorch 层面修改，跟 CUDA Graph / compile / PP / FSDP 都兼容，可以直接 plug-in 到现有 Llama serving 栈里。这种"低侵入性"的 architecture modification 在目前这个 LLM infra 复杂度爆炸的时代特别有价值。

reference 链接汇总：
- 主 paper: https://arxiv.org/abs/2410.05944
- gpt-fast (实现基础): https://github.com/pytorch-labs/gpt-fast
- Deja Vu (staleness 先验来源): https://arxiv.org/abs/2310.17157
- Flux (kernel overlap baseline): https://arxiv.org/abs/2406.06858
- Megatron-LM (TP 原始 paper): https://arxiv.org/abs/1909.08053
- Mamba-in-the-Llama (后训练 adaptation 对比): https://arxiv.org/abs/2408.15237
- Cross-Layer Attention: https://arxiv.org/abs/2405.12981
- PaLM (parallel attn/mlp baseline): https://arxiv.org/abs/2204.02311
- Highway Networks: https://arxiv.org/abs/1505.00387
- DenseNet: https://arxiv.org/abs/1608.06993
- Speculative Decoding: https://arxiv.org/abs/2211.17192
- Hogwild! (异步优化视角): https://arxiv.org/abs/1109.1033
- Llama 3 herd: https://arxiv.org/abs/2407.21783
- Mamba: https://arxiv.org/abs/2312.00752
- FineWeb-edu 数据集: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
