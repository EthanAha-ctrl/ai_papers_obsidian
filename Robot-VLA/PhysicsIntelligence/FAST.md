---
source_pdf: FAST.pdf
paper_sha256: 3739b31f5fecdde371509ff5bb13619979734e894a255a9b264253f4cc53934a
processed_at: '2026-08-04T07:31:06-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FAST 用人话讲

## 一句话说清楚

Robot 的 action signal 就像一段音频，采样率越高相邻点越像。你用 LLM 那套 next-token prediction 去 predict 每个时间点每个 dimension 的 action bin，模型发现 "copy 上一个 token" 就能拿接近 zero loss，于是摆烂不学了。FAST 的解法很粗暴：先把 action chunk 丢进 DCT 变成频域系数（大部分高频系数是 0 或接近 0），再用 BPE 把这堆稀疏整数压成几十个 token，让每个 token 都 carry 实实在在的信息量。

---

## 为什么 naïve binning 会崩

### 直觉

想象你在 50Hz 做 bimanual manipulation，action 是 14 维，1 秒 chunk = 700 个 token。Robot 动作本身是 smooth 的（电机加速度有物理上限），相邻两个 token 几乎一模一样。

Autoregressive model 的训练 objective 是：

$$\mathcal{L} = -\sum_{i=1}^{n} \log P(T_i \mid T_{1:i-1})$$

每个 token 给你的 learning signal 大致是：

$$\text{signal}_i \approx H(T_i) - H(T_i \mid T_{1:i-1})$$

- $H(T_i)$: token $i$ 本身的 entropy
- $H(T_i \mid T_{1:i-1})$: 给定前面所有 token，token $i$ 还剩多少 uncertainty

当 signal smooth 且高频采样时，$H(T_i \mid T_{i-1}) \to 0$，模型发现 "直接抄" 就赢了。于是 gradient 消失，model collapse 到 trivial solution。

### Figure 3 的 toy experiment 把这件事讲绝了

任务：给 4 个 random point，predict 它们的 cubic spline interpolation。固定 data distribution，只改 sampling rate (25 → 800)。

- Naïve binning: sampling rate 一上来 MSE 就爆，800 timesteps 时模型输出完全 flatten 成 copy
- FAST: 所有 sampling rate 都稳定

这个实验漂亮在它 isolate 了 tokenization 这一个变量。data 复杂度不变，model capacity 不变，training steps 不变，唯一变的是你怎么把 continuous signal 切成 discrete tokens。

OpenVLA 在 DROID 上 train 不动这件事就是这个 toy problem 的 real-world 版本。

OpenVLA paper: https://arxiv.org/abs/2406.09246

---

## FAST 的 pipeline

```
Raw action chunk (H 个 timestep, D 个 dimension)
  ↓ Quantile normalize 到 [-1, 1]
  ↓ 每个 dimension 独立做 DCT
  ↓ 乘以 γ=10 然后 round (量化)
  ↓ Column-first flatten 成 1D integer 序列
  ↓ BPE 压缩
Final tokens (大概 30 个 per arm per second)
```

### DCT 这一步

对第 $d$ 个 dimension 的 action sequence $a_{1:H}^{(d)}$ 做 Type-II DCT（和 JPEG 同款）：

$$C_k^{(d)} = \sum_{n=0}^{H-1} a_n^{(d)} \cos\left(\frac{\pi}{H}\left(n + \frac{1}{2}\right) k\right)$$

变量解释：
- $a_n^{(d)}$: 第 $n$ 个 timestep 第 $d$ 个 dimension 的 action value（已 normalize）
- $C_k^{(d)}$: 第 $d$ 个 dimension 的第 $k$ 个 frequency component
- $k=0$: DC component，整个 chunk 的 "平均位置"
- $k$ 大: 高频细节
- $H$: chunk length

Robot action 是 smooth signal，能量高度集中在 low-frequency coefficients。高频系数做完 quantization 基本都变 0。这就是 JPEG 对 natural image 有效的同一道理——natural image pixel 之间 smooth，DCT 后大部分 energy 在左上角几个系数。

DCT 原始 paper (Ahmed, Natarajan, Rao 1974): https://ieeexplore.ieee.org/document/1672320
JPEG: https://ieeexplore.ieee.org/document/125072

### 量化这一步

$$\bar{C}_k^{(d)} = \text{round}(\gamma \cdot C_k^{(d)})$$

- $\gamma = 10$: scale hyperparameter
- 大 $\gamma$ → 更精确重建，但 token 更多
- 小 $\gamma$ → 更激进压缩，但损失 fidelity

这步是 lossy 的，类似 JPEG 的 quantization table。

### Flatten 顺序：column-first 是关键

DCT 系数矩阵 $\bar{C} \in \mathbb{Z}^{D \times H}$，flatten 时有两种选：

**Column-first (论文用的)**: $[\bar{C}_1^{(1)}, \bar{C}_1^{(2)}, \ldots, \bar{C}_1^{(D)}, \bar{C}_2^{(1)}, \ldots]$
先把所有 dimensions 的 lowest frequency 放完，再放所有 dimensions 的 second lowest，依此类推。

**Row-first (没用)**: $[\bar{C}_1^{(1)}, \bar{C}_2^{(1)}, \ldots, \bar{C}_H^{(1)}, \bar{C}_1^{(2)}, \ldots]$
先把第一个 dimension 的所有频率放完。

为什么 column-first 更好？Autoregressive decoding 时模型先 commit 所有 dimensions 的 "overall shape"（low frequency），再 refine high-frequency details。这类似于 coarse-to-fine 生成，避免模型在 global structure 还没确定时就提前 lock in 某个 dimension 的局部细节。

这种 ordering 的直觉和 image generation 里 low-res → high-res 一脉相承。

### BPE 这一步

BPE (Byte Pair Encoding) 本来是 NLP 的 tokenization 方法，Sennrich 2015: https://arxiv.org/abs/1508.07909

FAST 的用法：
1. 初始 vocabulary 是所有 unique integers（从 quantization 来的，比如 -3, -2, -1, 0, 1, 2, 3, ...）
2. 统计哪对相邻 integer 出现最频繁，merge 成 new token，加进 vocabulary
3. 重复直到 vocab size = 1024

效果：
- 大量 repeated zeros 被压缩成单个 token
- 跨 dimension 的常见 coefficient pattern 被 merge
- 这是整个 pipeline 唯一 learned 的部分，但训练只要几分钟

为什么不用 Huffman 或 Lempel-Ziv (gzip 的算法)？论文说 "could be used, leave for future work"。BPE 的优势是能 produce 固定大小的 vocabulary，可以直接 overwrite 进 VLM 的 vocabulary（替换 LLM vocabulary 里最少用的 tokens）。

---

## 只有 2 个 hyperparameter

- $\gamma = 10$ (rounding scale)
- BPE vocab size = 1024

论文在所有 single-dataset 实验中用同一组值，没怎么 tune。对比 VQ-VAE based tokenizer（比如 FSQ, https://arxiv.org/abs/2309.15505）需要 tune codebook size, commitment loss weight, encoder/decoder architecture, EMA decay 等等，FAST 的 simplicity 是很大优势。

Appendix Figure 12 还画了 compression vs reconstruction tradeoff：FAST 在 low-fidelity 区不如 VQ-based（因为 DCT 是固定 basis，VQ 可以 learned basis），但 scaling 到 high-fidelity 区 FAST 明显更好。这解释了为什么 VQ-based 方法在 coarse task 上还行但 high-frequency dexterous task 崩。

---

## FAST+ Universal Tokenizer

把上面 pipeline 在 1M real robot action trajectories 上跑一遍 BPE 训练，得到一个 "universal" vocabulary。数据覆盖：

- Single-arm (Franka, UR5, WidowX)
- Bi-manual (ARX, Trossen, AgileX, ALOHA)
- Mobile (Fibocom, ARX slate mobile)
- Joint space / end-effector world frame / end-effector camera frame
- 5Hz 到 50Hz

Action 全部 pad 到 32 维。

发布在 HuggingFace：

```python
from transformers import AutoProcessor
tokenizer = AutoProcessor.from_pretrained(
    "physical-intelligence/fast",
    trust_remote_code=True
)
tokens = tokenizer(action_chunk)
```

这玩意儿类似 NLP 里的 SentencePiece——一个 pretrained tokenizer，整个 community 直接拿去用，新机器人新任务不用重训。

HuggingFace: https://huggingface.co/physical-intelligence/fast

Figure 8 显示在 14 个 unseen dataset 上测试（包括 NYU DexHand, Berkeley DexHand, UMI, HumanPlus, Waymo autonomous driving），FAST+ 都能达到 ≥2x compression。Figure 6 显示在 policy training 上 FAST+ 和 dataset-specific FAST 表现几乎一致。

---

## 实验结果几个关键 number

### Compression ratio (Table I)

| Dataset | Freq | Naïve tokens | FAST tokens | Ratio |
|---------|------|--------------|-------------|-------|
| BridgeV2 | 5Hz | 35 | 20 | 1.75x |
| DROID | 15Hz | 105 | 29 | 3.6x |
| Table Bussing | 20Hz | 140 | 28 | 5x |
| T-Shirt Folding | 50Hz | 700 | 53 | 13.2x |

观察：FAST 输出 token 数大致与信号 complexity 相关，与 sampling frequency 解耦。50Hz bimanual 的 700 个 naive token 被压成 53 个。

### Policy performance (Figure 6)

在 50Hz T-Shirt Folding 上：
- Naïve binning: ~0% 成功率
- FSQ (learned VQ): 中等
- FAST / FAST+: 高

在 20Hz Table Bussing 上同样 pattern。低频的 Libero 所有方法都 work。

### vs Diffusion π0 (Figure 9, 11)

π0 (Black et al. 2024, https://arxiv.org/abs/2410.24164) 是当前 SOTA VLA，用 flow matching (diffusion)。

- Small dataset (<50h): Libero, T-Shirt Folding 上 FAST 和 diffusion 性能 comparable
- Large dataset: Table Bussing 上 FAST 收敛 **3x 更快**
- DROID 上 FAST 更好地 follow language instruction (diffusion π0 经常 ignore prompt)
- π0-FAST generalist 在 4 个 dexterous task 上匹配 diffusion π0，但训练 **5x 更少 GPU hours**

### Inference speed 的 tradeoff

- Diffusion π0: 100ms per chunk (10 diffusion steps, 300M action expert)
- π0-FAST: 750ms per chunk (30-60 autoregressive steps, 2B LM backbone)

750ms 对 static manipulation 没问题，dynamic task 就不行。作者明确说 speculative decoding, quantization, custom kernel 是 future work。这点很诚实，没回避。

### DROID zero-shot

这是第一个在 DROID 上 train 出来能在完全 unseen environment zero-shot deploy 的 generalist policy。之前 DROID 原始 paper (https://arxiv.org/abs/2403.12945) 和 OpenVLA 都只做 co-training 或 fine-tuning eval。

π0-FAST 在 3 个大学 campus (Berkeley, Stanford, UW) 测试，新 table、新 background、新 viewpoint、新物体。16 个 task 44 个 trial 做 quantitative eval (Table II)，还能定性 demo pick/place, drawer, faucet, wiping 等。

### Ablation 两个关键

1. **Backbone independent**: 把 FAST+ 套到 OpenVLA (Prismatic 7B) 上，原本 OpenVLA 在 50Hz T-Shirt Folding 完全 train 不动，加 FAST 之后能 train 起来。说明 FAST 是 plug-and-play 的 tokenization 改进，不挑 backbone。

2. **BPE 不能少**: 只用 DCT quantization 不做 BPE → 性能下降。原因是有大量 repeated 0 token dilute learning signal，且 inference 要 decode 几百个 token 很慢。BPE 把这些 redundancy 消掉。

---

## Build intuition 的几个点

### 1. Tokenization 决定 learning signal

$$\text{useful gradient} \propto \frac{\text{marginal info per token}}{\text{tokens per chunk}}$$

Naïve binning 在 high-freq data 上分子趋零、分母巨大。FAST 把 information 集中到少数高信息 token。这和 LLM 中 BPE 的 motivation 完全一致——避免 model 把 capacity 浪费在 frequent sub-pattern 上。

### 2. Compression = Information Concentration

Autoregressive modeling $P(T_{1:n}) = \prod P(T_i \mid T_{1:i-1})$ 在 token 之间高度 redundant 时效率极低。Compression 先验地把 mutual information 消掉，让每个 token carry 新信息。

### 3. DCT 是 "free" 的 frequency analysis

对比 VQ-VAE：
- DCT: 解析方法，$O(N \log N)$，无训练，hyperparameter 少
- VQ-VAE: 要 train encoder-decoder, tune codebook, commitment loss, EMA...

DCT 在 high-fidelity 区比 VQ 好（Appendix Figure 12），因为固定 basis 在 fine detail 上 stable，VQ 的 learned codebook 容易 collapse 或 overfit specific dataset。

### 4. Column-first = Coarse-to-fine

Autoregressive decoding 先 commit global shape 再 refine local detail。这和 image generation 的 low-res → high-res 思想一致，避免 early commitment to wrong local detail。

### 5. Universal tokenizer 的意义

FAST+ 是 "SentencePiece for robot actions"。一旦 pretrained，整个 community 黑盒用，降低 robot learning 的 entry barrier。这是 paper 的 long-term impact 可能最大的部分。

---

## 几个我想吐槽或追问的点

1. **750ms inference 真的能用吗？** Table bussing 这种 quasi-static task OK，但 paper title 说 "dexterous"，laundry folding 也有 dynamic flatten 动作。real-time control loop 一般要 10-100Hz，750ms 等于 0.67Hz command update。Paper 没细讲 action chunk 执行时怎么 overlap 推理和执行。

2. **DCT 是 fixed basis**，对某些 action distribution 可能不是最优。比如 humanoid walking 是 periodic signal，DCT 其实挺好；但接触丰富的 dexterous in-hand manipulation 可能 piece-wise smooth + 间断 jump，DCT 这种 global basis 在 discontinuity 处会有 Gibbs 现象。Wavelet 或 learned basis 可能更好。

3. **Universal tokenizer 的 mixture 偏 distribution**。π0 自有 data 占大头，DROID 11.2%，OpenX 3.8%。如果你的 robot 不在这个 distribution 里（比如刚出来的新硬件），FAST+ 的 compression ratio 可能下降。不过 paper 说只需 normalize 到 [-1, 1] 然后 tokenize 1-second chunk 就行。

4. **为什么 column-first 不用实验数据 backup**？论文只说 "leads to more stable policy rollouts"，没给 ablation number。读者得相信作者。

5. **没讲 DCT chunk boundary artifact**。1-second chunk 之间 DCT 是独立做的，相邻 chunk 在 boundary 处可能不连续。这个问题在 JPEG 里 block artifact 众所周知。Robot action 有没有类似问题？Paper 没提。

6. **γ=10 是怎么选的**？只说 "not very sensitive"。Appendix B 应该有 sweep，但你得自己翻。

7. **BPE 的 vocabulary 是 1024**，这个 number 和 LLM 里动辄 32k-128k vocab 比小很多。可能是因为 action signal 比 language 简单，1024 够用。但 universal tokenizer 跨这么多种 robot，1024 会不会 cap performance？

---

## 进一步阅读

- FAST project page: https://pi.website/research/fast
- π0 paper: https://arxiv.org/abs/2410.24164
- OpenVLA: https://arxiv.org/abs/2406.09246
- DROID: https://arxiv.org/abs/2403.12945
- BPE original: https://arxiv.org/abs/1508.07909
- DCT original: https://ieeexplore.ieee.org/document/1672320
- JPEG: https://ieeexplore.ieee.org/document/125072
- FSQ: https://arxiv.org/abs/2309.15505
- VQ-VAE: https://arxiv.org/abs/1711.00937
- BridgeData V2: https://arxiv.org/abs/2308.12952
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- ALOHA: https://arxiv.org/abs/2304.13705
- RT-2: https://arxiv.org/abs/2307.15818
- RT-1: https://arxiv.org/abs/2212.06817
- FAST+ on HuggingFace: https://huggingface.co/physical-intelligence/fast

需要哪部分再展开（DCT 数学细节、BPE 训练 statistics、column-first 的因果分析、或与 diffusion VLA 的更深入对比），尽管问。

---

# FAST: Efficient Action Tokenization for Vision-Language-Action Models 深度解析

这篇 paper 来自 Physical Intelligence (Pi)、UC Berkeley、Stanford 的合作团队（Karl Pertsch、Kyle Stachowicz 等，包括 Sergey Levine 和 Chelsea Finn）。核心 idea 极其优雅：把 robot action trajectory 当作 time-series 信号，用 DCT (Discrete Cosine Transform) + BPE (Byte Pair Encoding) 压缩成离散 token，从而让 autoregressive VLA 能在 high-frequency dexterous tasks 上训练成功。

项目主页: https://pi.website/research/fast

---

## 1. 核心问题诊断：naïve binning 为什么在 high-frequency data 上崩

### 1.1 现象观察

Prior VLA 工作（RT-1 [8], RT-2 [10], OpenVLA [39]）都使用 per-dimension, per-timestep binning：把每个 action dimension 独立地分到 256 个 bins 中。这种方案在 low-frequency data（BridgeV2 5Hz, RT-1 3Hz）上 work，但在 high-frequency dexterous tasks 上完全失败，比如 OpenVLA 在 DROID dataset (15Hz) 上就 struggle。

### 1.2 根本原因：marginal information content → 0

Autoregressive model 的训练目标是 next-token prediction：

$$\mathcal{L} = -\sum_{i=1}^{n} \log P(T_i \mid T_{1:i-1})$$

每个 token $T_i$ 提供的 learning signal 正比于它的 **marginal information content**：

$$I(T_i; \text{future} \mid T_{1:i-1}) = H(T_i) - H(T_i \mid T_{1:i-1})$$

对于 smooth action signal $a(t)$，当 sampling frequency $f \to \infty$ 时，相邻 token 的差异 $\Delta a = a(t + \Delta t) - a(t) \propto 1/f \to 0$。这意味着：

$$H(T_i \mid T_{i-1}) \to 0 \quad \text{as} \quad f \to \infty$$

即模型只需要 "copy 上一个 token" 就能获得接近 zero loss，陷入 trivial local optima。Training 没有有效 gradient signal 去学复杂的 trajectory structure。

### 1.3 Figure 3 的 didactic experiment 验证

Authors 构造了一个 toy task：给定 4 个随机点，predict 插值它们的 cubic spline。在不同 sampling rate (25 → 800 timesteps) 下训练同一个 autoregressive transformer。

- **Naïve binning**: MSE 随 sampling rate 单调上升，800 timesteps 时模型 collapse 成 "copy first action"
- **FAST (DCT-based)**: MSE 在所有 sampling rate 下保持稳定低位

这个实验干净地分离了变量：data distribution 复杂度不变（同样是 4 个点插值），只是 sampling rate 变。Performance 差异完全归因于 tokenization scheme。

参考: Section IV, Figure 3
OpenVLA paper: https://arxiv.org/abs/2406.09246

---

## 2. FAST 算法核心：DCT + Quantization + BPE

### 2.1 Pipeline 总览

```
Raw action chunk a_{1:H} ∈ ℝ^{H × D}
   ↓ Quantile normalization → [-1, 1]
   ↓ Per-dimension DCT
   ↓ Scale & round (quantization)
   ↓ Column-first flattening
   ↓ BPE compression
Action tokens [T_1, ..., T_n]
```

### 2.2 DCT 数学详解

FAST 使用 Type-II DCT（与 JPEG 相同）。对于长度 $H$ 的 action sequence $a_{1:H}^{(d)}$ (第 $d$ 个 dimension):

$$C_k^{(d)} = \sum_{n=0}^{H-1} a_n^{(d)} \cos\left(\frac{\pi}{H}\left(n + \frac{1}{2}\right) k\right), \quad k = 0, 1, \dots, H-1$$

变量含义：
- $C_k^{(d)}$: 第 $d$ 个 action dimension 的第 $k$ 个 frequency component coefficient
- $a_n^{(d)}$: 第 $n$ 个 timestep、第 $d$ 个 dimension 的 action value
- $H$: action chunk length (1 second of control)
- $k$: frequency index；$k=0$ 是 DC component (signal mean)，$k$ 大对应高频
- $n$: time index

Inverse DCT (用于 decoding):

$$a_n^{(d)} = \frac{1}{H}\left[C_0^{(d)} + 2\sum_{k=1}^{H-1} C_k^{(d)} \cos\left(\frac{\pi}{H}\left(n + \frac{1}{2}\right) k\right)\right]$$

**为什么 DCT 适合 robot actions**:
1. Robot trajectories 是 $C^k$ smooth (至少 piece-wise smooth)，因为加速度有界、joint velocity 受电机限制
2. Smooth signal 的 energy 在频域上高度集中在 low-frequency coefficients (类似 JPEG 对 natural images 有效的原因)
3. DCT 是 real-valued，比 DFT 更紧凑（没有 phase 信息）
4. Energy compaction property: 少数 coefficients 捕获大部分 variance

参考 DCT 原始 paper: Ahmed, Natarajan, Rao 1974 https://ieeexplore.ieee.org/document/1672320
JPEG: Wallace 1992 https://ieeexplore.ieee.org/document/125072

### 2.3 Quantization

$$\bar{C}_k^{(d)} = \text{round}(\gamma \cdot C_k^{(d)})$$

- $\gamma$: scale hyperparameter；作者在所有实验中用 $\gamma = 10$
- Larger $\gamma$ → 更高 reconstruction fidelity，但更少 compression
- Rounding 后大多数 high-frequency coefficients 变成 0 (sparse matrix)

这是 **lossy** 压缩步骤，类似 JPEG 中的 quantization step。

### 2.4 Flattening order: column-first 是关键

DCT coefficient matrix $\bar{C} \in \mathbb{Z}^{D \times H}$，需要 flatten 成 1D 序列喂给 BPE。两种选择：

**Column-first (论文使用)**:
$$[\bar{C}_1^{(1)}, \bar{C}_1^{(2)}, \dots, \bar{C}_1^{(D)}, \bar{C}_2^{(1)}, \bar{C}_2^{(2)}, \dots]$$

即先放所有 dimensions 的最低频，再放所有 dimensions 的第二低频，依此类推。

**Row-first (未使用)**:
$$[\bar{C}_1^{(1)}, \bar{C}_2^{(1)}, \dots, \bar{C}_H^{(1)}, \bar{C}_1^{(2)}, \bar{C}_2^{(2)}, \dots]$$

即先放完第一个 dimension 的所有频率，再放第二个 dimension。

**为什么 column-first 更好**: autoregressive decoding 时，模型先预测所有 dimensions 的 low-frequency (overall shape)，再 refine high-frequency details。这类似于 coarse-to-fine 解码，rollout 更稳定。Row-first 会让模型在还没确定 overall trajectory 的情况下提前预测某个 dimension 的高频细节，容易导致局部 inconsistency。

### 2.5 BPE: 把 sparse integer 序列压成 dense tokens

BPE (Byte Pair Encoding) 最初用于 NLP tokenization (Sennrich et al. 2015 https://arxiv.org/abs/1508.07909)。FAST 把它用在 DCT 量化后的 integer 序列上：

1. 初始 vocabulary: 所有 unique integers (从 quantization 来的)
2. Iteratively merge 最频繁的 adjacent pair → new token
3. 重复直到 vocabulary size 达到 |V| = 1024

**BPE 在 FAST 中发挥两个作用**:
- **Compression**: 把大量 repeated zeros "squash" 成单个 token，把频繁出现的 coefficient 组合 merge
- **Cross-dimension pattern capture**: 比如 $\bar{C}_1^{(1)} = 5, \bar{C}_1^{(2)} = 5$ 这种常见组合可被 merge 成 1 个 token

BPE 是整个 pipeline 唯一的 learned component (需要训练 vocabulary)，但训练只需几分钟。

### 2.6 Algorithm 1 完整伪代码

```
Algorithm 1: FAST Tokenizer
Require: scale γ, BPE dictionary Φ

procedure FASTTokenizer(a_{1:H})
  C_j^i ← DCT(a_{1:H}^i)           # Compute DCT coefficients per dimension
  C̄_j^i ← round(γ · C_j^i)         # Quantize
  [T̃_k] ← [C̄_1^1, C̄_1^2, ..., C̄_2^1, ..., C̄_H^n]   # Column-first flatten
  
  BPE Training (offline):
    φ ← TrainBPE(D := {[T̃_k]})
  
  Tokenization (online):
    [T_1, ..., T_n] ← BPE([T̃_1, ..., T̃_k], φ)
  
  return action tokens
```

Hyperparameters 总共只有两个:
- $\gamma$ (rounding scale) = 10
- BPE vocabulary size = 1024

对比 VQ-VAE based tokenizer (如 FSQ [48]) 需要 train encoder-decoder network，tune codebook size、commitment loss weight、reconstruction loss weight 等多个 hyperparams。

---

## 3. FAST+ Universal Tokenizer

### 3.1 动机

每次在新 dataset 上 train BPE vocabulary 虽然快，但增加 friction。Authors 想 train 一个 universal tokenizer，能 black-box 应用到任何 robot。

### 3.2 训练数据

1M real robot action trajectories (1-second chunks)，覆盖:
- **Morphologies**: single-arm (Franka, UR5, WidowX), bi-manual (ARX, Trossen, AgileX), mobile (Fibocom, ARX slate mobile), ALOHA
- **Action spaces**: joint space, end-effector world frame, end-effector camera frame
- **Frequencies**: 5Hz (BridgeV2) to 50Hz (ARX bi-manual)
- **Padding**: all actions padded to 32 dimensions

完整 mixture 见 Appendix Table (paper 末尾)。主要来自 π0 datasets + 11.2% DROID + 5% BridgeV2 + 3.8% OpenX + 5% ALOHA。

### 3.3 使用

```python
from transformers import AutoProcessor
tokenizer = AutoProcessor.from_pretrained(
    "physical-intelligence/fast",
    trust_remote_code=True
)
tokens = tokenizer(action_chunk)
```

HuggingFace 集成让 adoption 极其简单。也可以 fit 新 tokenizer:

```python
new_tokenizer = tokenizer.fit(action_dataset)
```

### 3.4 Universal tokenizer 评估

Figure 8 显示 FAST+ 在 unseen datasets (SOAR, SERL, NYU DexHand, Berkeley DexHand, UMI, HumanPlus, Waymo autonomous driving) 上均能达到 ≥2x compression。Table III 列出 14 个 evaluation datasets 跨越 single-arm, dexterous hands, UMI, humanoid, navigation。

Figure 6 显示 FAST+ 在 policy training 上 matching dataset-specific FAST performance。

---

## 4. 实验结果详解

### 4.1 Compression Ratio (Table I)

| Dataset | Action Dim | Frequency | Naïve tokens | FAST tokens | Compression |
|---------|-----------|-----------|--------------|-------------|-------------|
| BridgeV2 | 7 | 5Hz | 35 | 20 | 1.75x |
| DROID | 7 | 15Hz | 105 | 29 | 3.6x |
| Table Bussing | 14 | 20Hz | 140 | 28 | 5.0x |
| T-Shirt Folding | 14 | 50Hz | 700 | 53 | 13.2x |

**关键观察**: FAST 输出 tokens 数量大致与信号 complexity 相关，与 sampling frequency 几乎解耦。每个 arm ~30 tokens per chunk。Naïve binning 的 token count 与 frequency 线性增长。

### 4.2 Policy Performance (Figure 6)

在 5 个 task 上比较 4 种 tokenizer:
- **Naïve binning** (RT-2/OpenVLA 风格)
- **FSQ** (learned VQ-VAE alternative, Mentzer et al. 2023 https://arxiv.org/abs/2309.15505)
- **FAST** (dataset-specific DCT+BPE)
- **FAST+** (universal)

在 50Hz T-Shirt Folding 上:
- Naïve: 完全失败 (~0% success)
- FSQ: 中等
- FAST & FAST+: 高成功率

在 20Hz Table Bussing 上同样 pattern。在低频 Libero (simulation) 上所有方法 work。

### 4.3 vs Diffusion π0 (Figure 9, 11)

Diffusion π0 是当前 SOTA VLA (Black et al. 2024 https://arxiv.org/abs/2410.24164)，使用 flow matching。

- **Small datasets (<50h)**: Libero, T-Shirt Folding 上 FAST 与 diffusion 性能 comparable
- **Large datasets**: Table Bussing 上 FAST converge **3x faster**
- **DROID**: FAST 更好地 follow language instructions (diffusion π0 经常 ignore language)
- **π0-FAST generalist**: 匹配 diffusion π0 在 laundry folding 等 dexterous task 上的性能，但训练只需 **5x fewer GPU hours**

### 4.4 Inference Speed Tradeoff (Section VI-E)

- Diffusion π0: **100ms** per 1-second chunk
  - 10 diffusion steps
  - 300M parameter action expert
- π0-FAST: **750ms** per 1-second chunk
  - 30-60 autoregressive decoding steps
  - 2B parameter language model backbone

这是 autoregressive VLA 的主要 limitation。在 static manipulation tasks 上 750ms 可接受，但 dynamic tasks (e.g., catching, fast reorientation) 不行。Authors 指出 LLM inference optimization (speculative decoding, quantization, custom kernels) 是 future work。

### 4.5 DROID Zero-Shot Evaluation (Section VI-B, Figure 7)

这是 **第一个** 在 DROID 上 train 成功的 generalist policy，能在完全 unseen environment 中 zero-shot evaluate。Prior DROID 工作 (Khazatsky et al. 2024 https://arxiv.org/abs/2403.12945) 和 OpenVLA 都只做 co-training 或 fine-tuning evaluation。

Quantitative eval: 16 tasks, 44 trials (Table II)，包括 pick/place, drawer, faucet, wiping 等。
Qualitative eval: 跨 3 个大学 campus (UC Berkeley, Stanford, UW)，新场景、新视角、新物体。

### 4.6 Ablation Studies (Section VI-D)

**Backbone independence**: 在 OpenVLA (Prismatic 7B) 上 test FAST。把 OpenVLA 改为 accept multiple images + predict 1-second chunks，用 FAST+ tokenization。FAST 让 OpenVLA 能 train 在 50Hz T-Shirt Folding 上 (原本完全失败)。

**BPE 必要性**: 去掉 BPE (只用 DCT quantization + raw flatten) → 性能下降但仍优于 naïve。原因:
1. 大量 repeated 0-tokens dilute learning signal
2. Inference 需要预测 hundreds of tokens → slow

---

## 5. π0-FAST Scaling to 10k Hours (Section VI-F)

### 5.1 Training data

π0 dataset mixture:
- 903M timesteps 自有 data
- 9.1% open-source: BridgeV2 + DROID + OpenX

### 5.2 Results (Figure 11, 15)

在 4 个 dexterous tasks 上 (Table Bussing, T-Shirt Folding, Grocery Bagging, Toast, Laundry Folding):
- π0-FAST (autoregressive) 匹配 π0 (diffusion) 性能
- 5x fewer GPU hours
- Compute-matched comparison (Figure 15): 同样 compute 下 π0-FAST clearly outperforms diffusion π0

### 5.3 Laundry Folding 案例 (Figure 10)

最难 task: bi-manual ARX 从 laundry basket 取衣服、flatten、fold、stack。需要:
- 精确 grasp
- Dynamic motion (flatten cloth)
- Retry/correction when cloth tangles
- 精确 placement on existing stack

π0-FAST 能完成，是 autoregressive VLA 的 milestone。

---

## 6. 直觉总结与 Insights

### 6.1 Tokenization 决定 learning signal

这篇 paper 最重要的 take-away: **tokenization 不只是 preprocessing，它直接决定了 autoregressive model 的 learning signal strength**。

$$\text{useful gradient} \propto \frac{\text{marginal information per token}}{\text{tokens per chunk}}$$

Naïve binning 在 high-frequency data 上: marginal info → 0, tokens per chunk → large → gradient 极弱。
FAST: marginal info per token 高 (每个 DCT coefficient 是 global signal 的某种 statistic), tokens per chunk 小 → gradient 强。

### 6.2 Compression = Information Concentration

这个 insight 联系到 information theory:
- Autoregressive training 本质是 modeling $P(T_{1:n}) = \prod P(T_i | T_{1:i-1})$
- 如果 $T_i$ 大部分 redundant (high mutual info with $T_{1:i-1}$)，模型 capacity 浪费在 memorize redundancy
- Compression 把 information 集中到少数 high-info tokens

这和 LLM 中 BPE 的 motivation 一致：稀有 token sequence merge 成单 token，避免 character-level model 浪费 capacity 在 frequent sub-patterns。

### 6.3 DCT 是 "cheap" 的 frequency analysis

为何选 DCT 而非 learned VQ-VAE:
1. **Analytical**: 无需训练 encoder-decoder network
2. **Fast**: $O(N \log N)$ via FFT-like algorithms
3. **Hyperparameter-light**: 只有 $\gamma$ 和 vocab size
4. **Robust**: 不需 dataset-specific tuning
5. **Bidirectional**: encode 和 decode 都 $O(N \log N)$

VQ-VAE 需要小心 tune codebook size, commitment loss, EMA decay 等；reconstruction quality 对 architecture sensitive。

### 6.4 Column-first flattening = Coarse-to-Fine Decoding

这个 design choice 类似:
- Image generation 中的 coarse-to-fine (low resolution → high resolution)
- Audio codec 中先传 spectral envelope 再传 fine structure
- Hierarchical RL 中先学 high-level plan 再 refine low-level control

在 autoregressive generation 中，先确定 global structure 再 add details 让 sampling 更稳定，避免 "early commitment to wrong local detail"。

### 6.5 Open Questions

Paper 自己指出:
1. **Dynamic tasks**: 750ms inference latency 限制 dynamic manipulation
2. **Mobile/humanoid/dexterous hands**: 离线 compression 实验显示 FAST+ work，但实际 policy training 未验证
3. **Diffusion + compression**: FAST tokenization 也可与 diffusion decoding 结合 (encode action 到 compressed space 再 diffusion)
4. **Alternative compression**: Huffman coding, Lempel-Ziv (gzip) 等可能替代 BPE

---

## 7. 个人思考 (build your intuition)

Karpathy 你会喜欢这篇 paper 的几个点:

1. **First-principles thinking**: Authors 从 "autoregressive learning signal = marginal information content" 出发，derive 出 compression 必要性，再选 DCT。这种 reasoning chain 非常干净。

2. **Toy experiment 极具说服力**: Figure 3 的 spline interpolation 实验，单一变量控制 (sampling rate)，clean ablation。这种 didactic experiment 在 robot learning paper 中少见。

3. **Connections to LLM tokenization**: BPE 在 NLP 中 well-known 重要，但 robotics 之前没人认真思考 action tokenization。Paper 把这两个 domain 的 insight bridge 起来。

4. **Compression vs Reconstruction tradeoff**: Appendix Figure 12 显示 FAST 在 low-fidelity 区域不如 VQ-based，但 scaling to high-fidelity 更好。这解释了为何 VQ-based methods 在 coarse task work 但 high-frequency dexterous 失败。

5. **Universal tokenizer**: FAST+ 类似 "SentencePiece for robot actions"。一旦 train 好，整个 community 可用，类似 NLP 中 pretrained tokenizer 标准化。

6. **5x training speedup**: 这是实际 impact。Diffusion π0 training 用 thousands of GPU hours，5x saving 巨大。

7. **Inference speed limitation 诚实**: Authors 不回避 750ms vs 100ms 的问题，明确指明 future work direction (speculative decoding, quantization)。

---

## Web Links 参考

- FAST project page: https://pi.website/research/fast
- π0 paper (Black et al. 2024): https://arxiv.org/abs/2410.24164
- OpenVLA paper (Kim et al. 2024): https://arxiv.org/abs/2406.09246
- DROID dataset paper (Khazatsky et al. 2024): https://arxiv.org/abs/2403.12945
- BPE original (Sennrich et al. 2015): https://arxiv.org/abs/1508.07909
- DCT original (Ahmed, Natarajan, Rao 1974): https://ieeexplore.ieee.org/document/1672320
- JPEG standard (Wallace 1992): https://ieeexplore.ieee.org/document/125072
- FSQ (Mentzer et al. 2023): https://arxiv.org/abs/2309.15505
- VQ-VAE (van den Oord et al. 2018): https://arxiv.org/abs/1711.00937
- BridgeData V2 (Walke et al. 2023): https://arxiv.org/abs/2308.12952
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- ALOHA (Zhao et al. 2023): https://arxiv.org/abs/2304.13705
- RT-2 (Brohan et al. 2023): https://arxiv.org/abs/2307.15818
- RT-1 (Brohan et al. 2022): https://arxiv.org/abs/2212.06817
- FAST+ HuggingFace: https://huggingface.co/physical-intelligence/fast

如果你想深入某个 aspect (e.g., 为什么 DCT Type-II 比 Type-I/III/IV 好，或 BPE merge 的具体 statistics，或 column-first flattening 在 autoregressive sampling 中的因果分析)，告诉我可以再展开。
