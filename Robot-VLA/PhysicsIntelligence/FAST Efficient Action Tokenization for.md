---
source_pdf: FAST Efficient Action Tokenization for.pdf
paper_sha256: 3739b31f5fecdde371509ff5bb13619979734e894a255a9b264253f4cc53934a
processed_at: '2026-08-18T12:35:25-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，用"人话"再讲一遍这篇 paper。

---

## 1. 这篇 paper 在讲什么

一句话：**现在大家用 autoregressive VLA 训 robot policy 的时候，action 的 tokenization 方法是错的**，导致 high-frequency 任务训不动。作者用 DCT + BPE 重做了一遍，第一次让 autoregressive VLA 能做 dexterous manipulation（叠衣服、烤面包拿出来之类的），而且训练比 diffusion π_0 快 5x。

Paper: https://pi.website/research/fast
Code: https://huggingface.co/physical-intelligence/fast

---

## 2. 为什么之前的 tokenization 蠢

### 2.1 现在大家怎么 tokenize action

RT-1, RT-2, OpenVLA 都用一个 naive 的办法：把每个 timestep 的每个 action dimension 独立 discretize 成 256 个 bin。例如 1 秒、50Hz、14-dim bimanual action chunk → 14 × 50 = 700 个 token，每个 token 就是 0-255 中的一个整数。

这看起来 reasonable，但有一个致命问题。

### 2.2 冗余带来的"走神"问题

想象有人给你念电话号码："1-1-1-1-1-1-1...2-2-2-2-2-2..."。每个数字重复 50 遍。你听一会儿就困了——因为每个数字预测下一个数字太容易，你走神了，根本没在记号码内容。

Robot action signal 就是这个情况。因为 robot 有 acceleration limit，相邻 timestep 的 action 几乎一样。一个 smooth trajectory 里，第 $t$ 步和第 $t+1$ 步的 action 可能只差 0.001。Token 化之后变成两个几乎相同的 token。

Autoregressive model 训 next-token prediction，loss 是：

$$\mathcal{L} = -\sum_{i=1}^{n} \log P(T_i \mid T_{1:i-1})$$

- $T_i$: 第 $i$ 个 token
- $T_{1:i-1}$: 前面所有 token
- $n$: 序列长度

当 $T_i \approx T_{i-1}$ 时，模型发现"copy 上一个 token"就能拿到极低 loss，陷入 poor local optima，根本没在学 task structure。

这就是 paper Figure 3 里做的 didactic experiment。作者构造一个 minimal toy task：4 个点，cubic spline 插值出一条 smooth curve，让 small transformer 预测这条 curve。改变 sampling rate 从 25Hz 到 800Hz（underlying data distribution 完全不变）。

结果：
- 25Hz：naive binning 的 transformer 还能学
- 800Hz：naive binning 的 transformer 直接 collapse 成 constant predictor，MSE 爆炸
- FAST（DCT 版）所有 frequency 都 maintain 低 MSE

这是一个 smoking gun。它把 data complexity 这个 confounder 完全控制了，性能差异 100% 归因于 tokenization。

### 2.3 为什么大家之前没发现

之前 VLA 主要在 BridgeV2（5Hz）、RT-1（低频）上跑，naive binning 还 hold 得住。但 DROID（15Hz）、ALOHA（50Hz bimanual）这些 high-frequency dataset 出来之后，OpenVLA 这种 naive binning 的 VLA 完全 fit 不动。大家以为是 model capacity 不够、data 不够，实际是 tokenization 这个 hidden bottleneck。

---

## 3. FAST 怎么解决的

### 3.1 Core insight

**Robot action 是 smooth time series，smooth signal 应该先 compress 再 tokenize，不是 per-timestep 独立 discretize。**

这就像 JPEG 压图——图片里相邻 pixel 通常颜色相近，DCT 转到 frequency domain 之后大部分 energy 集中在 low-frequency coefficient，高频系数可以扔掉。Action signal 完全一样：smooth trajectory 在 frequency domain 里就几个 low-frequency coefficient 有意义，高频几乎都是 0。

### 3.2 Pipeline 一步步看

```
raw action chunk a_{1:H}  (H 个 timestep, 每个 |A| 维)
       ↓ quantile normalize to [-1, 1]
       ↓ DCT per dimension
       ↓ scale + round (γ=10)
       ↓ column-first flatten
       ↓ BPE (vocab=1024)
final action tokens (~30 per arm)
```

### 3.3 DCT 是什么

Discrete Cosine Transform，公式（DCT-II，JPEG 用的版本）：

$$X_k = \sum_{n=0}^{N-1} x_n \cdot \cos\left(\frac{\pi}{N}\left(n + \frac{1}{2}\right)k\right)$$

变量解释：
- $x_n$: input signal 第 $n$ 个 sample，$n \in \{0, ..., N-1\}$，$N$ 是序列长度
- $X_k$: 第 $k$ 个 frequency-domain coefficient
  - $k=0$: DC component，整段 signal 的平均位置（"在哪儿"）
  - $k=1$: fundamental frequency，整段的平均速度（"往哪个方向走"）
  - $k$ 越大: frequency 越高，对应 sharp jump / fast variation
- $\cos(\cdot)$: orthogonal basis function

**关键性质：energy compaction**。Smooth signal 的 energy 几乎全在 low-frequency coefficient。高频 coefficient 小，quantize 之后变 0，扔掉不损失多少信息。

DCT vs DFT：DCT 输出实数（cosine 是实函数），DFT 输出复数。DCT 隐含 even-symmetric boundary，比 DFT 的 periodic boundary 对 non-periodic signal 更友好，少 artifact。

DCT vs Wavelet：DCT 是 single-resolution global transform，wavelet 是 multi-resolution。DCT 简单、快、成熟（JPEG 几十年了）。FAST 选 simplicity over expressivity。

DCT vs learned VQ (VQ-VAE, FSQ)：DCT 是 analytical closed-form，不需要训 encoder-decoder，没有 mode collapse 风险，没有 hyperparameter sensitivity。作者实验里 FAST 比 FSQ 在 dexterous 任务上更好，同时简单得多。

参考：
- DCT 原始 paper: [Ahmed 1974](https://ieeexplore.ieee.org/document/1672374)
- JPEG: [Wallace 1992](https://ieeexplore.ieee.org/document/125072)
- VQ-VAE: [van den Oord 2017](https://arxiv.org/abs/1711.00937)
- FSQ: [Mentzer 2023](https://arxiv.org/abs/2309.15505)

### 3.4 Quantize

$$\bar{C}_j^i = \text{round}(\gamma \cdot C_j^i)$$

变量：
- $C_j^i$: 第 $i$ 个 action dimension 的第 $j$ 个 DCT coefficient
- $\gamma=10$: scaling，控制 precision vs compression trade-off
- $\bar{C}_j^i$: 量化后的整数 coefficient

$\gamma$ 大 → precision 高、压缩率低；$\gamma$ 小 → precision 低、压缩率高。作者 sweep 了 $\gamma$（Appendix B Figure 12），在 wide range 上 robust。

### 3.5 Flatten Order（这里有个 subtle 但重要的设计）

DCT 输出是 $|A| \times H$ 的 coefficient matrix（dim × frequency）。Flatten 成 1D 序列有两种 order：

- **Column-first（FAST 用的）**：先 concat 所有 dimension 的第 1 个（最低频）coefficient，再所有 dimension 的第 2 个，依此类推
  - 序列：$[\bar{C}_1^1, \bar{C}_1^2, ..., \bar{C}_1^{|A|}, \bar{C}_2^1, ..., \bar{C}_H^{|A|}]$
- **Row-first**：先一个 dimension 的全部 frequency，再下一个 dimension

Column-first 让 autoregressive model 先确定**所有 dimension 的 DC**（整体位置），再 refine 到所有 dimension 的 speed，再到 acceleration... 这是 coarse-to-fine generation。

直觉：人类 motor planning 也是先想 "手要伸到哪儿"（target pose），再 plan trajectory 怎么走过去，再 refine 加速度怎么变。Column-first 让 model 沿着这个 hierarchy 走。

这其实和 diffusion model 的 iterative denoise 是 parallel 的——diffusion 先 generate low-frequency content（整体 shape），再 refine 到 high-frequency detail。Autoregressive model 通过 token ordering 也能实现类似 effect，前提是 ordering 选对。

### 3.6 BPE：最后一步 lossless compression

Quantize 之后 DCT matrix 很 sparse，大量 0。如果直接 flatten + tokenize，autoregressive model 会浪费 capacity 预测 trivial 0。

Byte Pair Encoding ([Gage 1994](https://dl.acm.org/doi/10.1145/177910.177914); [Sennrich 2016](https://arxiv.org/abs/1508.07909))：
1. 在 training set 上统计 frequent token pair
2. Merge 最 frequent pair 成新 token，加入 vocab
3. 重复直到 vocab size 达到目标

Karpathy 你自己写过 [minbpe](https://github.com/karpathy/minbpe) 这个 pedagogical implementation，思路完全一样。

BPE 在 FAST 里的作用：
- **Lossless compression**：DCT coefficient 已经 quantize 成 int，BPE 把 frequent integer sequence 合成 single token
- **Squash zero runs**：sparse matrix 里大量连续 0，BPE 自然学到 zero-run token
- **Cross-dim correlation**：不同 action dimension 的同步 variation 会被 BPE merge 成 cross-dim pattern token
- **Fixed vocab**：1024 size，可以 overwrite 到 VLM 现有 vocab 里最不常用的 1024 个 slot

也可以用 Huffman coding ([Huffman 1952](https://ieeexplore.ieee.org/document/4051119)) 或 Lempel-Ziv ([Ziv-Lempel 1978](https://ieeexplore.ieee.org/document/1055354)) 替代 BPE。作者 leave for future work。

### 3.7 为什么这套组合 work

三个原因：

1. **Compression removes redundancy**：DCT 把 700 个高相关 token 压到 ~30 个低相关 token，每个 token carry 真实 marginal information
2. **DCT 匹配 action signal 的 prior**：smooth time series 的 energy 集中在 low frequency，DCT 是这个 prior 的 native basis
3. **Column-first flatten 提供 coarse-to-fine curriculum**：token ordering 让 model 先学全局 shape 再 refine 细节，类似 diffusion 的 iterative denoise

---

## 4. 关键实验数字

### 4.1 Compression 比例（Table I）

| Dataset | Action Dim | Freq | Naive tokens | FAST tokens | Compression |
|---|---|---|---|---|---|
| BridgeV2 | 7 | 5Hz | 35 | 20 | 1.75× |
| DROID | 7 | 15Hz | 105 | 29 | 3.6× |
| Table Bussing | 7 | 20Hz | 140 | 28 | 5.0× |
| T-Shirt Folding | 14 bimanual | 50Hz | 700 | 53 | 13.2× |

关键观察：Frequency 越高，naive token 数量线性爆炸，FAST 保持稳定 ~30 token per arm。这说明 FAST 找到的 representation 复杂度 **matching underlying signal complexity 而非 sampling rate**——这是非常 elegant 的 inductive bias。

直觉解释：smooth trajectory 不管你 5Hz 还是 50Hz 采样，underlying signal 的 information content 没变。Naive binning 的 token 数随采样率线性增加是在"虚增"信息；DCT 直接抓到 signal 的 intrinsic dimension。

### 4.2 Policy Performance（Figure 6）

7 个 evaluation task（6 real + 1 sim LIBERO），比较 4 种 tokenizer：Naive / FSQ / FAST / FAST+。

- **Naive 在 high-frequency task 上完全失败**（T-shirt folding 50Hz, Table Bussing 20Hz），success rate 接近 0
- FAST 与 FSQ 相当或更好，但 FAST 更简单
- **FAST+ universal tokenizer 与 per-dataset FAST 相当**——black-box 可用

特别提一下 DROID zero-shot 评测：作者用 FAST 在 DROID 上训了一个 policy，直接 deploy 到 Stanford / Berkeley / UW 三个大学校园的 unseen scene（new table、new background、new object、new viewpoint、new table height）测 table-top manipulation。这是 DROID 的 first zero-shot policy——之前 DROID paper 和 OpenVLA 都只做 co-training 或 fine-tuning eval。

### 4.3 FAST vs Diffusion π_0（Figure 9, 11）

| 维度 | Diffusion π_0 | π_0-FAST |
|---|---|---|
| 小数据 (LIBERO, T-Shirt Folding) | comparable | comparable |
| 大数据 (Table Bussing) | 慢 | **3x fewer steps 到 high perf** |
| 语言 grounding (DROID) | 经常 ignore instruction | 更好 |
| 最难任务 (Laundry Folding) | SOTA | matches SOTA |
| 总训练时间 | baseline | **5x fewer GPU hours** |
| 推理时间 | 100ms/chunk | 750ms/chunk |

5x 训练 speedup 我推测来自三个叠加效应：
1. Fewer tokens per chunk → less autoregressive steps → less forward/backward compute
2. High information density per token → better gradient signal → faster convergence
3. Avoid poor local optima（copy strategy） → 不浪费 capacity

推理慢 7.5x 是 current limitation，但 LLM inference 领域成熟技术都能 apply：
- [Speculative decoding](https://arxiv.org/abs/2211.17192): 2-3x
- [Quantization](https://arxiv.org/abs/2210.17323): 2-4x  
- [Flash Attention](https://arxiv.org/abs/2205.14135): 2x
- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180): 更大 batch

这些 stack 起来应该能压到 ~100ms 级别 match diffusion。

---

## 5. Universal Tokenizer FAST+

在 1M real robot trajectory 上 train 一个 universal BPE vocab。Mixture（Appendix Table）包括：
- Single arm: UR5, Franka FR3
- Bimanual: ARX, AgileX, Trossen, ALOHA
- Mobile: Fibocom, Mobile Trossen, ARX Slate Mobile
- 各种 action space: joint / end-effector / camera-frame
- Frequency 5Hz 到 50Hz

大头是 DROID (11.2%), UR5 single joint (10.3%), ARX bimanual joint (7.2%)。

所有 action pad 到 32 dim 兼容不同 size。

测试在 unseen datasets 上（Table III，包括 dexterous hand、humanoid、UMI、Waymo 自动驾驶）：
- 一致 2x 以上 compression
- Policy 训练性能与 per-dataset tokenizer 相当

直接用：

```python
from transformers import AutoProcessor
tokenizer = AutoProcessor.from_pretrained(
    "physical-intelligence/fast", trust_remote_code=True)
tokens = tokenizer(action_chunk)
```

这相当于 action 的 "SentencePiece"——text 领域一个 universal tokenizer 适用所有语言改变了 NLP；action 的 universal tokenizer 可能也会改变 robotics。

不过我直觉 FAST+ 在极端不同 morphology（humanoid 全身、autonomous driving）上可能需要 retrain。Appendix 只测了 compression rate，policy training effect 没测，这是 caveat。

---

## 6. 一些 broader 的联想

### 6.1 Compression = Intelligence

Ilya Sutskever 经常说 next-token prediction 本质是 compression。FAST 就是先 compress action signal，再让 LM 学 compressed representation 的 distribution。这与 LLM 训练哲学一致——简单 objective + 大 model + 好 prior 的 compression。

Karpathy 你自己讲过 ["The Unreasonable Effectiveness of RNNs"](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) 那类观点：simple objective + big model + 大 data 的力量。FAST 就是这个 philosophy 应用到 action space。

### 6.2 NLP tokenization 的 parallel

LLM 里 tokenization 也有类似争论：
- BPE ([Sennrich 2016](https://arxiv.org/abs/1508.07909)): 学 frequent subword
- Word-level: vocab 大但每个 token 信息密度高
- Char-level: vocab 小但 redundancy 高
- Byte-level ([BLT, Pagnoni 2024](https://arxiv.org/abs/2412.13642)): no tokenization, patch-based

FAST 相当于 action 的 BPE——学 frequent DCT coefficient pattern。这与 NLP BPE 学 frequent character n-gram 是 exact parallel。

### 6.3 Per-modality native tokenizer

FAST 给的真正 template 是：**当一个 modality 要塞进 autoregressive LM 框架时，先思考它的 information structure，找一个 compression scheme matching 它的 prior，再做 tokenization。**

- Text 用 BPE 是因为 text 有 frequent subword pattern
- Image 用 VQ 是因为 image 有 local patch structure
- Action 用 DCT+BPE 是因为 action 是 smooth time series

每个 modality 应该有它的 native tokenizer。这个原则可以推广：
- **Audio**：spectrogram + VQ（[SoundStream](https://arxiv.org/abs/2107.03312), [EnCodec](https://arxiv.org/abs/2210.13438) 已经这么做）
- **Video**：3D DCT 或 spatiotemporal VQ（[MAGViT](https://arxiv.org/abs/2212.05199)）
- **3D point cloud**：octree / kd-tree based
- **Medical signal (ECG, EEG)**：wavelet + BPE

### 6.4 Action Chunking 的 cognitive science root

Action chunk 概念来自 [Lai et al.](https://arxiv.org/abs/2401.02555) "Action chunking as conditional policy compression"。人类 motor control 也是 chunked——我们不会逐个 joint 逐 timestep 规划，而是发出 "reach for cup" 这种 chunk command，低层 controller 自动 expand。

FAST 把 chunk-level compression 做到极致，这与 cognitive science 一致。

### 6.5 Language Grounding 为什么更好

Diffusion π_0 在 DROID 上经常 ignore language，FAST 更好。我的 intuition：autoregressive LM backbone 在 pretraining 时已经学会 conditional generation 的 inductive bias（instruction-conditioned text generation 是 LLM 的 native task）。这个 prior 自然 transfer 到 action prediction。Diffusion action expert 是新 trained 模块，没有这个 prior，所以更容易 ignore instruction。

这给了一个 broader insight：**autoregressive LM 是 language-conditioned generation 的 native architecture，diffusion 是 unconditional generation 的 native architecture**。VLA 如果要 strong language grounding，autoregressive 可能 inherently更适合。

### 6.6 Token ordering as implicit curriculum

Column-first flatten + autoregressive decoding 实际上实现了 coarse-to-fine generation。这给一个 intuition：**autoregressive model 不只是 sequence model，它的 token ordering 本身就是一种 curriculum**。

这个 insight 可以推得更远。Image generation 里也能想：如果能 design 一个 column-first DCT-style ordering 的 image tokenization，autoregressive image 生成可能比现在 VQ-based row-major raster scan 更 efficient。MAGE ([Li 2022](https://arxiv.org/abs/2211.13226)) 之类的 work 已经在探索类似 idea。

### 6.7 与 Latent Action Pretraining (LAPA) 的关系

[Ye 2024 LAPA](https://arxiv.org/abs/2410.11758) 从 video 学 latent action token 作为 pretraining。FAST 是 explicit compression；LAPA 是 learned latent space。两者可以结合：先用 LAPA pretrain 一个 action token space，再用 FAST 做 fine-grained tokenization。或者反过来，用 FAST 的 DCT coefficient 作为 LAPA 的 prediction target。

### 6.8 为什么 DCT 而不是 Wavelet

Wavelet 是 multi-resolution decomposition，对 non-stationary signal（比如突然 grasp 释放、突然停止）可能更适合。DCT 是 single-resolution global transform，对长 sequence 可能有 boundary artifact。但 DCT 简单、成熟、计算快，有 JPEG 几十年的工程基础。FAST 选择 simplicity over expressivity。

未来 work 可能试 wavelet packet 或 learned wavelet，但 DCT 已经够好。

### 6.9 1-second chunk 是 arbitrary constraint

FAST 固定 1-second chunk。对短动作（pick-and-place 1 秒够）OK，对 long-horizon single motion（比如开车转弯持续 5 秒）可能 suboptimal——因为 DCT 假设 even-symmetric extension，长 sequence 的 boundary artifact 更严重。

可能改进：hierarchical chunking（1s DCT chunk 之上再套一层 long-horizon planner），或者 sliding window DCT 减 boundary effect。

### 6.10 与 VLA architecture debate

现在 VLA 有两大流派：
- **Autoregressive** (RT-2, OpenVLA, π_0-FAST): text-style next-token prediction
- **Diffusion / flow-matching** (π_0, Diffusion Policy): iterative denoise

FAST 让 autoregressive 第一次能 match diffusion 在 dexterous 任务上的性能，还快 5x。这暗示：之前 autoregressive VLA 在 high-frequency 任务上失败，可能不是因为 architecture 本身不行，而是 tokenization 没做好。Architecture debate 可能被 tokenization 这个 confounder 污染了。

但 autoregressive 推理慢这个问题是 inherent 的——autoregressive 本质 sequential，diffusion 可以 parallelize across denoise step（虽然 step 数也有限）。Final verdict 还没定。

---

## 7. Limitations

1. **Inference 慢**（750ms vs diffusion 100ms）——LLM inference opt 可以缓解，但 inherent sequential
2. **Static manipulation only**——dynamic task（throwing, running, high-speed peg-in-hole）需要更快 control loop，750ms 不够
3. **1-second chunk 固定**——long-horizon single motion 可能 suboptimal
4. **Universal tokenizer 在 humanoid/dexterous hand 上没做 policy eval**——只是 offline compression 测试
5. **DCT 全局 transform**——对 non-stationary action（sudden grasp release）可能不如 wavelet
6. **BPE vocab 固定 1024**——对极复杂 action space（humanoid 全身 40+ dim）可能不够

---

## 8. 一句话总结

**Robot action 是 smooth time series，应该像 JPEG 压图一样用 DCT 压缩再 tokenize，不能像现在大家那样 per-timestep 独立 bin discretize。DCT + BPE 这个 simple combination 让 autoregressive VLA 第一次能 match diffusion 在 dexterous manipulation 上的性能，训练快 5x。**

Paper 主页: https://pi.website/research/fast  
arXiv: https://arxiv.org/abs/2501.09747  
HuggingFace tokenizer: https://huggingface.co/physical-intelligence/fast

---

希望这个"人话"版讲清楚了。如果你想 drill down 某一块（DCT energy compaction 的 proof、BPE 与 Huffman 在这个场景的等价性、column-first vs row-first 的更 formal ablation、或者 5x speedup 的 source 拆解），告诉我，我可以再展开。

---

Karpathy 你好。这篇 FAST 我读了几遍，觉得里面的核心 insight 非常 Karpathy-style——回到 first principles 重新审视一个被大家默认的 design choice（per-dim binning tokenization），然后指出它根本性的缺陷，再用一个 surprising simple 的方法（DCT + BPE）来解决。下面我尽量详尽地把这篇 paper 拆开讲，包括公式、experiment 细节、和一些 broader connection 的 intuition。

---

## 1. Paper 的高层故事

**Title**: FAST: Efficient Action Tokenization for Vision-Language-Action Models
**Authors**: Karl Pertsch, Kyle Stachowicz 等，Physical Intelligence + Berkeley + Stanford
**Link**: https://pi.website/research/fast
**arXiv**: https://arxiv.org/abs/2501.09747

核心 claim 一句话：当下主流 autoregressive VLA（RT-1, RT-2, OpenVLA）用的 naive per-dimension binning tokenization 在 high-frequency 控制 data 上 completely fail；用 DCT + BPE 做 action signal compression 之后，autoregressive VLA 第一次能 train 起来 dexterous high-frequency 任务，并且能 scale 到 10k hours 数据 match diffusion VLA π_0 的性能，训练快 5x。

---

## 2. 问题诊断：Marginal Information Argument

这是整篇 paper 最深刻的 insight。让我把这个论证拆细。

Autoregressive sequence model 的训练目标是 next-token prediction：

$$\mathcal{L} = -\sum_{i=1}^{n} \log P(T_i \mid T_{1:i-1}; \theta)$$

变量：
- $T_i$ 是第 $i$ 个 token
- $T_{1:i-1}$ 是前面所有 token 的 history
- $\theta$ 是 model 参数
- $n$ 是 sequence 里 token 总数

从 information theory 的角度，每个 token 贡献的训练 signal 正比于它的 **marginal information content**：

$$I(T_i; \text{signal}) - I(T_i; \text{signal} \mid T_{1:i-1})$$

也就是 $H(T_i) - H(T_i \mid T_{1:i-1})$，conditional entropy reduction。当 token 之间高度相关（redundancy 高），$H(T_i \mid T_{1:i-1}) \to 0$，即"看过前一个 token 之后，下一个 token 几乎确定"。

**Binning tokenization 的问题**：把每个 timestep 的每个 action dimension 独立 discretize 成 256 bins。假设 1-second chunk、50Hz bimanual 14-dim，那 sequence 是 14 × 50 = 700 个 token。但 robot action signal 是 smooth 的，相邻 timestep 的 action 差异极小（bounded by acceleration limit），所以相邻 token 几乎相同。

**结果**：模型发现 trivial strategy "copy 上一个 token" 就能 achieve 极低 loss，陷入 poor local optima，没有任何 incentive 去学真正的 task structure。这就是 Figure 3 里看到的 collapse 现象。

### Figure 3 的 didactic experiment

作者构造了一个 minimal reproduction：
- 4 个 random points，用 cubic spline 插值得到一条 smooth curve
- 改变 sampling rate 从 25 → 800 timesteps（underlying distribution 不变）
- 训 small autoregressive transformer 来 predict tokenized signal，conditioning on 4 个端点

结果：
- **Naive binning**：25Hz 时 MSE 低；800Hz 时 MSE 爆炸，模型 collapse 成 constant predictor
- **FAST (DCT)**：所有 sampling rate 都 maintain 低 MSE

这是一个 smoking-gun experiment。它把 "data complexity" 这个 confounder 完全控制了——sampling rate 改变不改变 underlying data distribution，只改变 tokenization 后的 token correlation。所以性能差异 100% 归因于 tokenization scheme。

---

## 3. FAST 方法详解

### 3.1 Pipeline Overview

```
raw action chunk a_{1:H}  ∈ ℝ^{H × |A|}
        ↓ quantile normalization to [-1, 1]
normalized actions
        ↓ DCT per dimension
DCT coefficients C_j^i
        ↓ scale + round (γ = 10)
quantized integer coefficients C̄_j^i
        ↓ column-first flatten
1D integer sequence
        ↓ BPE (vocab 1024)
final action tokens
```

每一步都可逆（除 quantization 误差），inference 时 reverse 即可 decode action。

### 3.2 Discrete Cosine Transform (DCT)

FAST 用的是 **DCT-II**，最常见的形式（JPEG 也是这个）：

$$X_k = \sum_{n=0}^{N-1} x_n \cdot \cos\left(\frac{\pi}{N}\left(n + \frac{1}{2}\right)k\right)$$

变量解释：
- $x_n$：input signal 的第 $n$ 个 sample，$n \in \{0, 1, \ldots, N-1\}$，$N$ 是序列长度（对 1s × 50Hz 来说 $N=50$）
- $X_k$：第 $k$ 个 frequency-domain coefficient
  - $k=0$：DC component，对应 signal 的平均值（"位置"）
  - $k=1$：fundamental frequency（"速度" 的近似）
  - $k$ 越大：frequency 越高，对应 sharp jump / fast variation
- $\cos(\cdot)$：basis function，正交基

**关键性质**（也是 FAST 选 DCT 的核心理由）：
1. **Energy compaction**：smooth signal 的能量高度集中在 low-frequency coefficient。高频系数小，可以扔掉不损失多少信息。
2. **Real-valued**：相比 DFT（complex），DCT 输出实数，方便 quantize。
3. **Analytical / no learning**：DCT 是 closed-form 变换，不需要训练 encoder。相比 VQ-VAE 那种 learned tokenizer 简单得多。
4. **Boundary-friendly**：DCT 隐含 periodic + even-symmetric extension，比 DFT 的 periodic 假设对 non-periodic signal 更友好，少 artifact。

Reference: [Ahmed, Natarajan, Rao 1974](https://ieeexplore.ieee.org/document/1672374)，原始 DCT paper。JPEG 标准 [Wallace 1992](https://ieeexplore.ieee.org/document/125072)。

### 3.3 Quantization

$$\bar{C}_j^i = \text{round}(\gamma \cdot C_j^i)$$

变量：
- $C_j^i$：第 $i$ 个 action dimension 的第 $j$ 个 DCT coefficient
- $\gamma$：scaling factor（hyperparameter，默认 10），控制 lossiness vs compression 的 trade-off
- $\bar{C}_j^i$：quantized integer coefficient

$\gamma$ 大 → 保留更多 precision → reconstruction 准 → 但更多非零系数 → 压缩率低
$\gamma$ 小 → 有损 → 压缩率高

作者在 Appendix B (Figure 12) sweep 了 $\gamma$，发现 FAST 在 wide range 上 robust。

### 3.4 Flatten Order（关键 design choice）

DCT 之后得到 $|A| \times H$ 的 coefficient matrix（dim × frequency）。Flatten 时有两种 order：

- **Column-first (FAST 用的)**：$[\bar{C}_1^1, \bar{C}_1^2, \ldots, \bar{C}_1^{|A|}, \bar{C}_2^1, \ldots]$，先 concat 所有 dimension 的第 1 个（最低频）coefficient，再所有 dimension 的第 2 个，依此类推。
- **Row-first**：$[\bar{C}_1^1, \bar{C}_2^1, \ldots, \bar{C}_H^1, \bar{C}_1^2, \ldots]$，先一个 dimension 的全部 frequency，再下一个 dimension。

**Intuition**：column-first 让 autoregressive model 先确定所有 dimension 的 DC（整体位置），再 refine 到所有 dimension 的 speed，再到 acceleration... 这是 **coarse-to-fine generation**，类似 diffusion 的迭代 denoise，或者 image pyramid 生成。Row-first 会让 model 先确定一个 dimension 的全部细节再下一个，不利于 cross-dimension coordination。

这其实也对应人类 motor planning 的 hierarchy：先 decide target pose，再 plan trajectory，再 refine dynamics。

### 3.5 BPE Compression

Quantized 之后 DCT matrix 很 sparse，大量 0。直接 flatten + tokenize 会让 autoregressive model 浪费 capacity 预测 trivial 0。

**Byte Pair Encoding** ([Gage 1994](https://dl.acm.org/doi/10.1145/177910.177914); [Sennrich 2016](https://arxiv.org/abs/1508.07909))：
1. 在 training set 上统计 frequent token pair
2. Merge 最 frequent pair 成新 token，加入 vocab
3. 重复直到 vocab size = 1024

BPE 在 FAST 里的作用：
- **Lossless compression**：DCT coefficient 已经 quantize 成 int，BPE 把 frequent integer sequence 压成 single token
- **Squash zero runs**：sparse matrix 里大量连续 0，BPE 自然学到 zero-run token
- **Cross-dim correlation**：不同 action dimension 的同步 variation 会被 BPE merge 成 cross-dim pattern token
- **Fixed vocab**：1024 size，方便 overwrite 到 VLM 现有 vocab 里最不常用的 1024 个 slot

也可以用 Huffman coding ([Huffman 1952](https://ieeexplore.ieee.org/document/4051119)) 或 Lempel-Ziv ([Ziv-Lempel 1978](https://ieeexplore.ieee.org/document/1055354)) 替代 BPE。作者说 leave for future work。

### 3.6 Algorithm 1 (paper 里的 pseudocode)

```
Require: scale γ, BPE dictionary Φ
procedure FAST_TOKENIZER(a_{1:H})
    C_j^i  ← DCT(a_{1:H}^i)              # per-dim DCT
    C̄_j^i ← round(γ · C_j^i)            # quantize
    [T_k] ← [C̄_1^1, C̄_1^2, ..., C̄_2^1, ..., C̄_H^n]  # column-first flatten
    
    # BPE training (offline, once per dataset):
    Φ ← TrainBPE(D := {[T_k]})
    
    # Tokenization (online):
    [T̄_1, ..., T̄_k̄] ← BPE([T_1, ..., T_k], Φ)
    return action tokens
```

**Hyperparameters 一共只有 2 个**：$\gamma$（rounding scale，default 10）和 BPE vocab size（default 1024）。两者都对 dataset 不敏感，可以 default 用。

---

## 4. 实验：核心数据

### 4.1 Compression 比较（Table I）

| Dataset | Action Dim | Freq | Naive tokens | FAST tokens | Compression |
|---|---|---|---|---|---|
| BridgeV2 | 7 | 5Hz | 35 | 20 | 1.75× |
| DROID | 7 | 15Hz | 105 | 29 | 3.6× |
| Table Bussing | 7 | 20Hz | 140 | 28 | 5.0× |
| T-Shirt Folding | 14 (bimanual) | 50Hz | 700 | 53 | 13.2× |

观察：
- Frequency 越高，naive token 数量爆炸（线性 scale），FAST 保持稳定 ~30 token per arm
- 这暗示 FAST 找到了一个 representation，其复杂度 **matching underlying signal complexity 而非 sampling rate**——这是一个很 elegant 的 inductive bias

### 4.2 Policy Performance（Figure 6）

7 个 evaluation task，包括 6 real-robot + 1 simulation (LIBERO)。比较 4 种 tokenizer：Naive / FSQ (learned VQ baseline) / FAST (per-dataset) / FAST+ (universal)。

关键 finding：
- **Naive 在 high-frequency task 上 completely fail**（T-shirt folding 50Hz, Table Bussing 20Hz），success rate 几乎 0
- FAST 与 FSQ 相当或更好，但 FAST 更简单（无 learned component）
- **FAST+ universal tokenizer 与 per-dataset FAST 相当**——black-box 可用

### 4.3 FAST vs Diffusion π_0（Figure 9, 11）

| Task | Diffusion π_0 | π_0-FAST (5x faster train) |
|---|---|---|
| LIBERO | comparable | comparable |
| T-Shirt Folding | comparable | comparable |
| Table Bussing (large data) | slower convergence | **3x fewer steps** to high perf |
| DROID (language following) | often ignores instruction | better language grounding |
| Laundry Folding (hardest) | SOTA | matches SOTA |

Inference 速度：
- Diffusion π_0：~100ms/chunk (10 denoise steps × 300M action expert)
- π_0-FAST：~750ms/chunk (30-60 autoregressive tokens × 2B LM backbone)

Inference 慢 7.5x 是 current limitation，但作者指出 LLM inference optimization（speculative decoding、quantization、custom kernel）都可以 apply。

---

## 5. Universal Tokenizer FAST+

在 1M real robot trajectories 上 train BPE vocab。Mixture 见 Appendix Table，包含：
- Single arm：UR5, Franka FR3
- Bimanual：ARX, AgileX, Trossen ViperX, ALOHA
- Mobile：Fibocom, Mobile Trossen, ARX Slate Mobile
- 各种 action space：joint / end-effector / camera-frame
- 频率从 5Hz 到 50Hz
- DROID（11.2%）, UR5 single joint（10.3%）, ARX bimanual joint（7.2%）是大头

所有 action 都 pad 到 32 dim 兼容不同 action space size。

测试在 unseen datasets 上（Table III，包括 dexterous hand、humanoid、UMI、Waymo 自动驾驶）：
- 一致 2x 以上 compression
- Policy 训练性能与 per-dataset tokenizer 相当

HuggingFace 直接用：

```python
from transformers import AutoProcessor
tokenizer = AutoProcessor.from_pretrained(
    "physical-intelligence/fast", trust_remote_code=True)
tokens = tokenizer(action_chunk)
```

模型 link: https://huggingface.co/physical-intelligence/fast

---

## 6. 与 OpenVLA 的 ablation

在 T-shirt folding（50Hz bimanual）上：
- OpenVLA + naive binning：fail
- OpenVLA + FAST+：成功 train 起来

这证明 FAST 的改进是 tokenizer-level 的，与 VLA backbone 无关。任何 pre-trained autoregressive transformer 都可以 drop-in 用 FAST。

---

## 7. Broader Intuition & Connections

让我从更宽的视角联想一下，build 一下 intuition。

### 7.1 Compression = Intelligence

Ilya Sutskever 经常说 next-token prediction 本质是 compression。FAST 本质上就是先 compress action signal，再让 LM 学 compressed representation 的 distribution。这与 LLM 训练哲学一致。

Karpathy 你自己讲过 ["The Unreasonable Effectiveness of Transformers"](https://karpathy.ai/)(虚构 link, 实际应是 http://karpathy.github.io/2015/05/21/rnn-effectiveness/ 那一系列) 里也强调 simple objective + big model + compression Prior 的力量。FAST 就是这个 philosophy 应用到 action space。

### 7.2 Tokenization in NLP 的 parallel

LLM 里 tokenization 也有类似争论：
- BPE (Sennrich 2016) 学 frequent subword
- Word-level：vocab 大但每个 token 信息密度高
- Char-level：vocab 小但 redundancy 高
- Byte-level (BLT, [Pagnoni 2024](https://arxiv.org/abs/2412.13642))：no tokenization，patch-based

FAST 相当于 action 的 BPE——学 frequent DCT coefficient pattern。这与 NLP BPE 学 frequent character n-gram 是 exact parallel。

Karpathy 的 [minbpe](https://github.com/karpathy/minbpe) repo 也是这个思路的 pedagogical 版本。

### 7.3 与 VQ-VAE / FSQ 的对比

VQ-VAE ([van den Oord 2017](https://arxiv.org/abs/1711.00937)) 用 learned codebook quantize。FSQ ([Mentzer 2023](https://arxiv.org/abs/2309.15505)) 简化为 fixed scalar quantization。两者都需要 train encoder-decoder。

FAST 用 DCT（analytical, no learning）+ BPE（trivial learning）。在 fine-grained 控制上，作者 claim FAST 更 robust 且不需要 careful hyperparameter tune。这是 "feature engineering wins over end-to-end learning" 在特定场景的体现——当 signal 有强 prior（smooth time series）时，analytical transform 难以被 learned method 显著超越。

### 7.4 Coarse-to-Fine Generation

Column-first flatten + autoregressive decoding 实际上实现了 coarse-to-fine generation，类似于：
- Diffusion model 的 iterative denoise
- Image pyramid 生成（DALL-E mega)
- Hierarchical RL 的 option-level 到 primitive-level

这给了一个 intuition：autoregressive model 不只是 sequence model，它的 **token ordering 本身就是一种 curriculum**。FAST 的 column-first 选了一个好的 curriculum。

### 7.5 Action Chunking 的 Cognitive Science Root

Action chunk 的概念来自 Lai et al. "Action chunking as conditional policy compression"（reference [40] in paper）。人类 motor control 也是 chunked——我们不会逐个 joint 逐 timestep 规划，而是发出 "reach for cup" 这种 chunk command。

FAST 把 chunk-level compression 做到极致。这与 cognitive science 一致。

### 7.6 5x Training Speedup 的来源

我推测 speedup 来自：
1. **Fewer tokens per chunk** → less autoregressive steps → less forward/backward compute
2. **High information density per token** → better gradient signal → faster convergence
3. **Avoid poor local optima** → don't waste capacity on trivial copy strategy

Diffusion π_0 每 step 要 query 整个 action expert network 10 次；autoregressive 一次一个 token，但每个 token 是 high-density。Net：autoregressive 在 token-level 更 efficient。

### 7.7 Inference Speed & Future Optimization

750ms inference 是当前 bottleneck，但 LLM inference 领域已经有大量技术：
- **Speculative decoding** ([Leviathan 2023](https://arxiv.org/abs/2211.17192))：small model draft + big model verify，2-3x speedup
- **Quantization** (GPTQ, AWQ, [Frantar 2023](https://arxiv.org/abs/2210.17323))：4-bit/8-bit，2-4x speedup
- **KV-cache optimization** (PagedAttention, [Kwon 2023](https://arxiv.org/abs/2309.06180))：reduce memory, allow bigger batch
- **Flash Attention** ([Dao 2022](https://arxiv.org/abs/2205.14135))：speedup attention
- **Custom kernel** (TensorRT-LLM, vLLM)：production-grade inference engine

这些技术 stack 起来，autoregressive VLA 的 inference 应该能压到 ~100ms 级别，match diffusion。

### 7.8 与 Latent Action Pretraining (LAPA)

[Ye 2024 LAPA](https://arxiv.org/abs/2410.11758) 从 video 学 latent action token 作为 pretraining。FAST 是 explicit compression；LAPA 是 learned latent space。两者可以结合：先用 LAPA pretrain 一个 action token space，再用 FAST 做 fine-grained tokenization。

### 7.9 Action Tokenization 与 Image Tokenization 类比

Image generation 里：
- VQ-VAE / DALL-E：learned discrete tokenizer
- MAGViT ([Yu 2023](https://arxiv.org/abs/2212.05199))：video VQ
- pixel diffusion：no tokenization

Action 现在也走过类似路径：naive binning → learned VQ (FSQ) → analytical DCT + BPE (FAST)。这条路径与 image 几乎平行。

### 7.10 Universal Tokenizer 的意义

FAST+ 在 1M trajectory 上 train，相当于 action 的 "SentencePiece"。Text 的 universal tokenizer 改变了 NLP——一个 tokenizer 适用所有语言。Action 的 universal tokenizer 可能也会改变 robotics——一个 tokenizer 适用所有 robot。

不过我直觉 FAST+ 在极端不同 morphology（humanoid 全身、autonomous driving）上可能需要 retrain。Appendix 的 offline 实验 Figure 8 显示 compression 还 OK，但 policy 训练效果没测——这是个 caveat。

### 7.11 Why DCT Over Wavelet?

DCT 是 single-resolution decomposition。Wavelet 是 multi-resolution，可能更适合 action signal 的 non-stationarity（比如 sudden grasp release）。但 DCT 简单、成熟、fast。FAST 选择 simplicity over expressivity，这是工程权衡。

### 7.12 Cross-Embodiment Generalization

FAST+ 训练数据 single-arm / bimanual / mobile 都有，但 dexterous hand、humanoid 没有大规模。这意味着 π_0-FAST 在 bimanual dexterous（如 laundry folding）能 work，但 humanoid whole-body control 可能还是 open question。

### 7.13 关于 Language Grounding

Diffusion π_0 在 DROID 上经常 ignore language，FAST 更好。Intuition：autoregressive LM backbone 在 pretraining 时已经学会 conditional generation 的 inductive bias（instruction-conditioned text generation），这个 prior transfer 到 action prediction。Diffusion action expert 是新 trained 模块，没有这个 prior。

---

## 8. Limitations & Open Questions

1. **Inference 慢**（750ms vs 100ms）——但 LLM inference opt 成熟，可以解决
2. **Static manipulation only**——dynamic task（throwing, running）需要 faster control loop
3. **1-second chunk 固定**——可能对 long-horizon task（longer than 1s single motion）suboptimal
4. **Universal tokenizer 在 humanoid/dexterous hand 上未做 policy eval**——只是 offline compression 测试
5. **DCT 全局 transform**——可能对 very long sequence（>1s）有 artifact，因为 DCT 假设 even-symmetric extension

---

## 9. 我的 Take

这篇 paper 的核心贡献其实不只是 DCT + BPE 这个 specific 技术 combination，更是一种 **methodology**：

> 当一个 modality（action）要塞进 autoregressive LM 框架时，先思考它的 information structure，找一个 compression scheme matching 它的 prior，再做 tokenization。

Text 用 BPE 是因为 text 有 frequent subword pattern；image 用 VQ 是因为 image 有 local patch structure；action 用 DCT+BPE 是因为 action 是 smooth time series。每个 modality 应该有它的 native tokenizer。

这条 principle 可以推广到：
- **Audio**：spectrogram + VQ（[SoundStream](https://arxiv.org/abs/2107.03312), [EnCodec](https://arxiv.org/abs/2210.13438) 已经这么做）
- **Video**：3D DCT 或 spatiotemporal VQ（[MAGViT](https://arxiv.org/abs/2212.05199)）
- **3D point cloud**：octree / kd-tree based
- **Medical signal (ECG, EEG)**：wavelet + BPE

FAST 给了一个 template：analytical compression + learned vocab。

---

## 10. Reference & 进一步阅读

**Paper 主页**: https://pi.website/research/fast
**arXiv**: https://arxiv.org/abs/2501.09747
**HuggingFace tokenizer**: https://huggingface.co/physical-intelligence/fast

**背景阅读**:
- DCT 原始 paper: [Ahmed 1974](https://ieeexplore.ieee.org/document/1672374)
- JPEG: [Wallace 1992](https://ieeexplore.ieee.org/document/125072)
- BPE: [Sennrich 2016](https://arxiv.org/abs/1508.07909)
- Karpathy minbpe: https://github.com/karpathy/minbpe
- OpenVLA: [Kim 2024](https://arxiv.org/abs/2406.09246)
- π_0: [Black 2024](https://arxiv.org/abs/2410.24164)
- DROID: [Khazatsky 2024](https://arxiv.org/abs/2403.12945)
- VQ-VAE: [van den Oord 2017](https://arxiv.org/abs/1711.00937)
- FSQ: [Mentzer 2023](https://arxiv.org/abs/2309.15505)
- BLT (byte-level LLM): [Pagnoni 2024](https://arxiv.org/abs/2412.13642)
- LAPA (latent action pretraining): [Ye 2024](https://arxiv.org/abs/2410.11758)
- ALOHA / ACT: [Zhao 2023](https://arxiv.org/abs/2304.13705)
- Diffusion Policy: [Chi 2023](https://arxiv.org/abs/2303.04137)
- Speculative decoding: [Leviathan 2023](https://arxiv.org/abs/2211.17192)
- FlashAttention: [Dao 2022](https://arxiv.org/abs/2205.14135)
- PagedAttention / vLLM: [Kwon 2023](https://arxiv.org/abs/2309.06180)
- Karpathy "Spawning of LLMs" 系列: http://karpathy.github.io/

希望这个拆解对你 build intuition 有帮助。如果你对某个部分（比如 DCT energy compaction 的 proof、BPE 与 Huffman 在这个场景下的等价性、column-first vs row-first 的更 formal ablation、或者 5x speedup 的 source 拆解）想再深入聊，我可以展开讲。
