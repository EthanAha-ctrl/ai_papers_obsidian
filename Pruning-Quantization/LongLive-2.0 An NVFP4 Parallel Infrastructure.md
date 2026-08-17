---
source_pdf: LongLive-2.0 An NVFP4 Parallel Infrastructure.pdf
paper_sha256: 6b15c7a74766bab77643c67a46d3a8df7bf55b8ed7e902b1b0518e4e71f8088f
processed_at: '2026-08-05T15:54:32-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# LongLive-2.0 人话版

## 一句话总结

NVIDIA 的人把 long video generation 从训练到推理全链路压到了 **4-bit 浮点（NVFP4）**，顺手设计了一套配合 AR video training 的 sequence parallelism，最后 5B 模型跑 720p 视频能到 45.7 FPS。

---

## 他们要解决啥问题

生成 long video，比如 60 秒的 720p 视频，autoregressive chunk-by-chunk 地生成，有三个让人头疼的事：

**第一，sequence 太长，单卡装不下。** 视频 latent 不是 text token，几十秒视频 encode 完就是几十万 token，attention activation 直接爆显存。

**第二，越长的视频，GEMM（矩阵乘法）越是大头。** 大家都觉得 attention 是 bottleneck，但 video DiT 里的 linear layer（QKV projection、FFN）总 FLOPs 随长度线性涨，长视频里 GEMM 反而是主战场。

**第三，训练用 BF16、推理才量化的做法有 gap。** 大部分量化工作都是 PTQ（post-training quantization）——模型训完再压到 4-bit。这会有 distribution shift，video generation 对 numerical error 很敏感，压完画质会掉，眼睛会糊，identity 会漂。

LongLive-2.0 的核心赌注：**训练和推理都用 NVFP4，从头到尾一个 precision，消除 mismatch**。

---

## NVFP4 到底是个啥

先说 FP4。4 bit 表示一个浮点数，能表示的就那么几个值：

```
{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
```

就 16 个数，而且分布不均匀——小值附近密（间距 0.5），大值附近稀疏（4 到 6 差了 2）。这比 INT4 的均匀分布好，因为神经网络权重小值多大值少，FP4 给小值更精细的分辨率。

但 16 个数肯定不够用，所以加了 **hierarchical scaling**：

- 每 16 个 element 共享一个 FP8 scale（block-wise）
- 整个 tensor 共享一个 FP32 scale（global）

相当于每个 element 的真实值 = FP4 值 × block scale × global scale。block scale 处理局部 dynamic range，global scale 处理整体 magnitude。

NVFP4 vs MXFP4（OCP 标准）的区别：NVFP4 block 更小（16 vs 32），block scale 用 E4M3（有 mantissa）vs E8M0（纯 exponent），还多一个 global FP32 scale。所以 NVFP4 局部 tracking 更准。

---

## 训练怎么搞：Balanced SP

### 先说背景：AR teacher forcing

Autoregressive video training 用 efficient teacher forcing。对一段 N 个 chunk 的视频，构造两路 latent：

- **clean**：ground truth latent，当 context
- **noisy**：加噪后的 latent，当 prediction target

拼成 `[clean; noisy]` 一个长 sequence，用 block-sparse mask 让每个 noisy chunk attend 前面的 clean chunk。一次 forward 就能监督所有 N 个 noisy chunk，比每次只监督一个 suffix 快 N 倍。

### 传统 SP 哪里不行

Sequence parallelism（SP）就是把长 sequence 切到多卡上。DeepSpeed-Ulysses 的做法是按 head dim 切，All-to-All 通信。

但直接套到 AR teacher forcing 上有两个坑：

**坑一：loss 分布不均。** `[clean; noisy]` 按 sequence 切 4 段，前几段可能全是 clean（没 loss），后几段全是 noisy（loss 密集）。有的 GPU 闲死，有的 GPU 忙死。

**坑二：VAE encoding 重复。** 每个 SP rank 都得 encode 完整视频（或 root encode 完 broadcast）。VAE 这步没享受到 sharding 的好处。

### Balanced SP 的思路

核心一句话：**让每个 rank 拿同一个 temporal chunk 的 clean + noisy pair**。

不是把 `[all clean; all noisy]` 切成 4 段，而是每个 rank 自己构造 `[自己的 clean chunk, 自己的 noisy chunk]`。这样：
- 每个 rank 都有 context 和 target，loss 均匀
- 每个 rank 只 VAE-encode 自己那一段视频（加个左边 halo 覆盖 VAE 的 temporal receptive field），encode 完扔掉 halo，cost 从 $O(F)$ 降到 $O(F/P + h)$

### 还有几个细节

**Halo 构造**：VAE 是 3D causal conv，有 temporal receptive field。rank $p$ 只 encode 自己 chunk 的话，边缘 latent 会缺左边 context。所以 encode 时多拿左边 $h$ 个 frame，encode 完丢掉，只留 local chunk。结果和 full-video encode 完全一致。

**Natural mask**：Ulysses All-to-All 之后 token order 变成 interleaved `[clean_0, noisy_0, clean_1, noisy_1, ...]`。传统做法每层 attention 都要 permute 回 `[all clean; all noisy]` 再算 mask，开销大。Balanced SP 直接在 interleaved order 上算 mask——从 token index 推出它属于哪个 rank、是 clean 还是 noisy、原始 temporal position，用 `flex_attention` compile 成 fused kernel。

**Error recycling**：teacher forcing 有 exposure bias（训练用 clean context，推理用 model rollout）。LongLive v1 用 error recycling buffer 缓解。Balanced SP 下这个 buffer 也按 SP shard，每个 rank 只存自己那部分 position 的 error，warming-up 时只 gather 同 SP rank 的 DP rank。

---

## 算法侧：clean pipeline

### 跳过多阶段

以前的 Self-Forcing、Causal-Forcing 要 ODE initialization → short DMD → streaming long-tuning DMD，三阶段。LongLive v1 加了个 long tuning stage，更复杂。

LongLive-2.0 直接在 long video data 上 AR fine-tune bidirectional diffusion model（Wan2.2-TI2V-5B），一步到位。原因：infrastructure 到位了（SP + NVFP4 让 long video training cheap），不需要先在 short video 上训再 transfer。

### Multi-shot prompting

每个 temporal chunk 绑独立 text prompt，cross-attention 按 chunk 分解。不同 shot 可以不同 prompt，支持 prompt switch，用户编辑 future chunk 时保留 preceding history。

### Few-step distillation via LoRA

DMD（distribution matching distillation）通常要 fine-tune 整个 backbone。LongLive-2.0 只训 LoRA：

$$W \approx \text{Dequant}(Q_{search}(W_0)) + \frac{\alpha}{r} BA$$

$W_0$ 是 frozen 的 quantized backbone，$BA$ 是 trainable 的 low-rank（rank 128）。

关键：LoRA 在 **original diffusion model** 上训（不是 AR model），训完直接 plug 进任何 AR checkpoint。类似 LCM-LoRA 的 universality。好处是 DMD 和 AR 训练可以并行，LoRA 可复用，优化更稳定。

---

## 推理怎么搞

### W4A4 NVFP4 inference

Weight 4-bit + Activation 4-bit。理论 4× throughput（memory traffic 减 4 倍），实际 1.84× 因为有 non-GEMM overhead。

因为 backbone 是 NVFP4-aware trained，PTQ 的 mismatch 消失。Ablation 显示：PTQ 从 85.06 掉到 84.04，pretrained NVFP4 保持 84.51。

### NVFP4 KV cache

AR generation 的 KV cache 随 history 线性增长，很快爆显存。LongLive-2.0 把 KV cache 也量化到 NVFP4：

- Key 先做 μ-smoothing（减均值，SmoothQuant 思路，flat 化 outlier channel）
- 然后 adaptive scale selection（Four Over Six：每个 block 试 max→6 和 max→4 两种 scale，选 reconstruction error 小的）
- 存储从 BF16 的 4 bytes/element 降到 ~1.125 bytes/element，压缩比 3.6×

customized CUDA dequant kernel 让 overhead < 2%。

**附带好处**：SP inference 的 All-to-All 通信交换的是 K/V tensor，量化后通信量也降 3.6×。Table 6 显示 SP=2 时通信从 1.8s 降到 1.1s，SP=4 时从 12.8s 降到 7.8s。

### Async streaming VAE decoding

VAE decode 是最后一步 bottleneck。Baseline 攒齐所有 latent 再 sequential decode，memory $O(C \cdot T_c)$，latency $C \cdot (t_{DiT} + t_{VAE})$。

LongLive-2.0：
1. 3D VAE 改成 chunk-by-chunk streaming decode + CPU offload，memory 降到 $O(T_c)$
2. 专用一个 GPU 跑 VAE，和 N-GPU DiT cluster 异步

DiT denoise chunk $c+1$ 时，VAE node decode chunk $c$。因为 $t_{DiT} \geq t_{VAE}$，decode 被 hide 掉。Latency 从 $C(t_{DiT}+t_{VAE})$ 降到 $C \cdot t_{DiT} + t_{VAE}$。

### Multi-shot attention sink

Sliding window attention 丢掉窗口外的 token 会导致 appearance drift。Standard attention sink（pin 前几帧）在 multi-shot 场景失败：单一 global sink 保不住 shot 内 coherence，moving shot-level sink 会丢 global identity。

解法是两组 anchor：
- **Global sink**：video 前 $S_g$ 帧，永久固定，保 global identity
- **Shot-level sink**：current shot 前 $S_s$ 帧，每次 scene cut 重新 bind，保 local coherence

Effective KV set = global sink ∪ shot sink ∪ sliding window。

Shot-level sink 零 memory 开销，只 track 两个 scalar pointer（START, LEN）。Prompt switch 自动 trigger scene cut，rebind shot sink，global sink 和 history 不动。

---

## 效果数字

### 训练（Table 1）

64s 视频：
- BF16 w/o SP：OOM
- BF16 w/ SP：1372.9s
- BF16 Balanced SP：1196.5s
- NVFP4 Balanced SP：639.5s（2.1× vs BF16+SP）

NVFP4 收益随长度增长：16s 只有 1.3×，64s 有 2.1×。因为长 sequence GEMM 占比大。

### 推理（Table 3, 64s video）

- BF16：112.9s, 36.4 GB
- + NVFP4：96.0s, 29.7 GB
- + NVFP4 KV cache：99.5s, 19.4 GB（memory 大降，latency 微增但后续被 async hide）
- + Async decoding：57.6s, 19.4 GB
- + 2-step distillation：36.3s, 19.4 GB → **45.7 FPS**

### 画质（Table 4, VBench）

- BF16 4-step：85.06 total
- NVFP4 4-step：84.51（掉 0.55）
- NVFP4 2-step：83.14（掉 1.92，换 2× 速度）

对比 Self-Forcing 1.3B 4-step：84.31 total, 21.2 FPS。LongLive-2.0 NVFP4 2-step：83.14, 45.7 FPS——快 2.15×，质量略低但同档。

---

## 局限

1. **NVFP4 inference 只在 Blackwell（GB200）上提速**。H100/A100 没有原生 FP4 Tensor Core，只能 fallback 到 SP inference，收益主要来自 SP + KV cache 压缩，不是量化本身。

2. **2-step distillation 画质掉得多**（85.06 → 83.14）。Few-step 还有空间，可能需要更好的 distillation objective 或更大 LoRA capacity。

3. **Long video benchmark 有限**。VBench-Long 60s 是目前主流，但分钟级、小时级 generation 的 evaluation 还不成熟。

---

## 一句话 intuition

这篇 paper 的核心 insight：**量化不该是事后压缩，该是训练-推理 co-design 的一等公民**。传统 QAT 是"模拟量化噪声"，NVFP4 在 Blackwell 上是"真的在 FP4 算"，这是质变。配合 Balanced SP 把 AR teacher-forcing 的 data structure 编码进 parallelism strategy，让 quantization + parallelism 两者正交地加速 video generation 全流程。

参考：
- GitHub: https://github.com/NVlabs/LongLive
- NVFP4 blog: https://developer.nvidia.com/blog/nvfp4-tensor-core-programming
- DeepSpeed-Ulysses: https://arxiv.org/abs/2309.14509
- Self-Forcing: https://arxiv.org/abs/2506.08009
- Four Over Six: https://arxiv.org/abs/2512.02010
- LCM-LoRA: https://arxiv.org/abs/2311.05556
- StreamingLLM: https://arxiv.org/abs/2309.17453
- SmoothQuant: https://arxiv.org/abs/2211.01808
- QuaRot: https://arxiv.org/abs/2404.14028
- flex_attention: https://arxiv.org/abs/2412.05496

---

# LongLive-2.0: NVFP4 Parallel Infrastructure for Long Video Generation

这篇 paper 来自 NVIDIA（Song Han 团队 + Yukang Chen 等），核心 idea 是把 **NVFP4 quantization** 和 **sequence parallelism** co-design，贯穿 long video generation 的训练和推理全流程。我来一层一层拆解，build your intuition。

---

## 1. 为什么需要这件事：long video AR generation 的瓶颈本质

Long video generation 走 autoregressive chunk-by-chunk 路线（像 Self-Forcing、Causal-Forcing、LongLive v1），瓶颈有三层：

**(a) Sequence length 爆炸**。Video latent 不同于 text token，一个 64s 720p 视频经过 VAE 之后 latent token 数量轻松到几十万级别。单卡装不下 attention 的 activation。

**(b) GEMM 占比随长度上升**。Attention 是 O(L²) 的，但对 video DiT 来说，linear projection（QKV、FFN）这些 GEMM 的总 FLOPs 也随 L 线性甚至超线性增长。Paper 里明确说"GEMM computation during training, the proportion of which increases as video length grows"——意味着 video 越长，量化 GEMM 收益越大。

**(c) Train-inference precision mismatch**。大部分 quantization 工作做的是 PTQ（post-training quantization），训练用 BF16，推理时才压到 4-bit。这会造成 distribution shift，尤其 video generation 对 numerical error 敏感（色偏、texture 模糊、identity drift）。LongLive-2.0 的核心论点是：**training 和 inference 用同一个 NVFP4 precision，消除 mismatch**。

参考链接：
- NVFP4 Blackwell architecture: https://developer.nvidia.com/blog/nvfp4-tensor-core-programming
- DeepSpeed-Ulysses: https://arxiv.org/abs/2309.14509
- Self-Forcing: https://arxiv.org/abs/2506.08009

---

## 2. NVFP4 是什么：E2M1 + hierarchical scaling

这是整个 paper 的 numeric 基石。我详细讲。

### 2.1 E2M1 FP4 format

E2M1 表示：1 bit sign + 2 bits exponent + 1 bit mantissa。可表示的数值集合是 finite 的：

$$\mathbb{F}_{E2M1} = \{0, \pm 0.5, \pm 1, \pm 1.5, \pm 2, \pm 3, \pm 4, \pm 6\}$$

注意是非均匀 spacing：小数值附近 spacing 细（0.5），大数值附近 spacing 粗（4 到 6 之间跳 2）。这比 INT4 uniform quantization 好，因为 neural network weight/activation 通常是 heavy-tailed 分布，小值多、大值少，FP4 给小值更精细 resolution。

最大可表示 magnitude $M^{FP4} = 6$。对比 FP8 E4M3 的 $M^{FP8} = 448$。

### 2.2 Hierarchical scaling 公式

Paper 的公式 (2)(3)：

$$\hat{\mathbf{X}} = \hat{\mathbf{X}}^{FP4} \cdot \alpha^{FP8} \cdot \alpha^{FP32}$$

变量含义：
- $\hat{\mathbf{X}}^{FP4} \in \mathbb{F}_{E2M1}$：每个 element 的 4-bit 浮点值
- $\alpha^{FP8}$：block-wise scale，每 **16 个 element** 共享一个 FP8 E4M3 scale（这就是 NVFP4 vs MXFP4 的关键区别——MXFP4 是 32-element block，NVFP4 是 16-element，更精细）
- $\alpha^{FP32}$：tensor-wise global scale，整个 tensor 一个 FP32 值

为什么 hierarchical？因为单一 global scale 无法同时 cover tensor 里的 outlier 和 small value。Block scale 处理局部 dynamic range，global scale 处理整体 magnitude。

### 2.3 为什么 NVFP4 比 MXFP4 好

- NVFP4 block size 16 vs MXFP4 block size 32 → 局部 dynamic range tracking 更准
- NVFP4 block scale 用 FP8 E4M3（有 8 bits）vs MXFP4 用 FP8 E8M0（只有 exponent，无 mantissa）→ block scale 自身精度更高
- NVFP4 多一个 tensor-level FP32 global scale → 处理跨 block 的整体 magnitude 差异

参考：https://arxiv.org/abs/2512.02010 (Four Over Six paper，scale search 的来源)

### 2.4 RHT (Random Hadamard Transform) for gradient stability

Paper 提到 weight-gradient GEMM 用 RHT。这是关键 trick。Weight gradient 是 $\frac{\partial L}{\partial W} = \text{grad\_output}^T \cdot \text{input}$，这个 GEMM 的 operand 之一是 activation input，里面有 outlier channel。直接 quantize 会爆炸。

RHT 的做法：quantize 之前对 operand 乘一个 random Hadamard matrix $H \in \{\pm 1/\sqrt{n}\}^{n \times n}$。Hadamard 是 orthogonal 的，所以 $\|Hx\| = \|x\|$，但 outlier 被 spread 到所有 dimension，magnitude 分布变 flat。这样 block quantization 的误差分布更均匀。

参考 QuaRot: https://arxiv.org/abs/2404.14028 (类似思路)

---

## 3. Balanced Sequence Parallelism：co-design 的核心创新

这是 paper 最有 system insight 的部分。我详细讲为什么传统 SP 不行。

### 3.1 AR training 的 teacher-forcing layout

Efficient teacher-forcing（来自 Self-Forcing）的 setup：对一个 $N$-chunk 的 raw video window $\mathbf{X}$，encode 成 VAE latent $\mathbf{Z}$，然后构造 paired stream $[\mathbf{z}_{clean}; \mathbf{z}_{noisy}]$。

- $\mathbf{z}_{clean}$：所有 chunk 的 clean latent（ground truth），作为 context
- $\mathbf{z}_{noisy}$：所有 chunk 加噪后的 latent，作为 prediction target

Block-sparse AR mask 让每个 noisy chunk $i$ attend to：preceding clean chunks $\{0, ..., i-1\}$ 和自己的 noisy tokens。这样一次 forward pass 就 supervise 所有 $N$ 个 noisy chunk，比 naive teacher-forcing（每次只 supervise 一个 suffix）快 $N$ 倍。

### 3.2 传统 SP 的问题

公式 (1) 说传统 SP 把 $[\mathbf{z}_{clean}; \mathbf{z}_{noisy}]$ 当普通 sequence，按 head dim 切（DeepSpeed-Ulysses 风格）。问题有二：

**Problem 1: Loss imbalance**。如果 SP group size $P=4$，sequence 切成 4 段。前几段可能是 clean-heavy（只有 context，没 loss），后几段是 noisy-heavy（全是 target，loss 密集）。导致 GPU 间 workload 不均，idle GPU 等忙的 GPU。

**Problem 2: VAE encoding replicated**。每个 SP rank 都要 encode 完整 video（或 root encode 后 broadcast）。VAE 这一步没享受到 SP 的 sharding，是 bottleneck。

### 3.3 Balanced SP 的解法

核心 insight：**让每个 rank 拿同一个 temporal chunk 的 clean + noisy pair**。

公式 (1)：

$$\mathbf{z}^{(p)} = [\mathbf{z}_{clean}^{(p)}, \mathbf{z}_{noisy}^{(p)}] \in \mathbb{R}^{\frac{L}{P} \times H \times d}$$

变量：
- $p$：rank index
- $L$：total clean+noisy token length
- $P$：SP group size
- $H$：attention head 数
- $d$：head dimension

每个 rank 拿 $\frac{L}{P}$ 个 token，其中一半 clean 一半 noisy，来自**同一个 temporal chunk**。这样：
- Loss-bearing token 均匀分布 → workload balanced
- 每个 rank 只 VAE-encode 自己的 local chunk $\mathbf{X}^{(p)}$ + left halo → VAE cost 从 $O(F)$ 降到 $O(F/P + h)$

### 3.4 Halo 构造（Appendix C）

VAE 是 3D causal convolution，有 temporal receptive field。如果 rank $p$ 只 encode 自己的 chunk，边缘的 latent 会缺左边 context。解法：encode 时多拿左边 $h$ 个 frame（halo），encode 完之后 discard halo latents，只保留 local chunk $\mathbf{Z}^{(p)}$。这样 local latent 和 full-video encode 的结果完全一致，但 cost 降低。

### 3.5 Natural teacher-forcing mask on Ulysses order

这是另一个巧妙设计。Ulysses All-to-All 之后，global token order 变成 interleaved（公式 8）：

$$[\mathbf{z}_{clean}^{(0)}, \mathbf{z}_{noisy}^{(0)}, \mathbf{z}_{clean}^{(1)}, \mathbf{z}_{noisy}^{(1)}, ..., \mathbf{z}_{clean}^{(P-1)}, \mathbf{z}_{noisy}^{(P-1)}]$$

传统做法是先 permute 回 $[\text{all clean}; \text{all noisy}]$ 再 apply mask，每层 attention 都要 permute，开销大。

Balanced SP 的做法：直接在 interleaved order 上计算 mask。公式 (9)：

$$p(i) = \lfloor \frac{i}{2L_{loc}} \rfloor, \quad r(i) = i \bmod 2L_{loc}, \quad t(i) = p(i) L_{loc} + (r(i) \bmod L_{loc})$$

- $p(i)$：token $i$ 属于哪个 rank block
- $r(i)$：rank 内 offset
- $t(i)$：原始 temporal position
- $r(i) < L_{loc}$ → clean token；$r(i) \geq L_{loc}$ → noisy token

这样 mask predicate 直接从 token index 算出，用 PyTorch 的 `flex_attention` compile 成 fused kernel，不需要 materialize permutation。

参考 flex_attention: https://arxiv.org/abs/2412.05496

### 3.6 SP-aware error recycling

Teacher forcing 有 exposure bias：训练用 clean context，推理用 model rollout。LongLive v1 用 error recycling buffer（存过去 prediction error，stochastic 注入 clean prefix）。

Balanced SP 下这个 buffer 也要 shard：按 local block position + diffusion timestep 建 2D bucket，每个 rank 只存 $N_{blk}/P$ 个 local position。Warming-up 时只 gather 同 SP rank 的 DP rank，避免 cross-SP 通信（因为 position 对其他 SP rank 无效）。

---

## 4. Training pipeline：clean pipeline 的算法部分

### 4.1 直接 AR fine-tune，跳过 ODE init 和 multi-stage DMD

对比 previous work：
- Self-Forcing：ODE initialization → short DMD → streaming long-tuning DMD，三阶段
- Causal-Forcing：类似复杂 pipeline
- LongLive v1：加 long tuning stage，更复杂

LongLive-2.0：直接在 long video data 上 AR fine-tune bidirectional diffusion model（Wan2.2-TI2V-5B），一步到位得到 long + interactive + multi-shot AR model。原因：strong infrastructure（SP + NVFP4）让 long video training 变 cheap，不需要靠 short video 训练再 transfer。

### 4.2 Multi-shot interactive prompting

每个 temporal chunk $\mathbf{Z}_i$ 绑定独立 text prompt $\mathbf{T}_i$，cross-attention factorized：

$$\text{CrossAttn}(\mathbf{Z}_i, \mathbf{T}_i)$$

而不是整个 video 共享一个 global prompt。这样不同 shot 可以有不同 prompt，支持 prompt switch at chunk boundary，用户编辑 future chunk 时保留 preceding history。

### 4.3 Few-step distillation via standalone LoRA

这是第二个 algorithm insight。DMD（Distribution Matching Distillation）通常要 fine-tune 整个 backbone。LongLive-2.0 只训 LoRA：

公式 (4)：

$$\mathbf{W} \simeq \text{Dequant}(Q_{search}(\mathbf{W}_0)) + \Delta \mathbf{W}, \quad \Delta \mathbf{W} = \frac{\alpha_{LoRA}}{r} \mathbf{B}\mathbf{A}$$

- $\mathbf{W}_0$：pretrained backbone weight（frozen）
- $Q_{search}$：scale-search-based NVFP4 quantization
- $\mathbf{A} \in \mathbb{R}^{r \times d_{in}}, \mathbf{B} \in \mathbb{R}^{d_{out} \times r}$：trainable low-rank matrix，rank $r=128$
- $\alpha_{LoRA} = 128$：LoRA scaling

关键：LoRA 在 **original diffusion model**（不是 AR model）上训 DMD，训完直接 plug 进 AR model。类似 LCM-LoRA 的 universality。好处：
- DMD 训练和 AR 训练可以并行
- 同一个 LoRA 可以插到不同 AR checkpoint
- 优化更稳定（比 fine-tune 整个 quantized backbone）

参考 LCM-LoRA: https://arxiv.org/abs/2311.05556

---

## 5. Inference infrastructure

### 5.1 W4A4 NVFP4 inference

Weight 4-bit + Activation 4-bit。理论 throughput speedup 4×（因为 memory traffic 减 4 倍，GEMM 是 memory-bound）。实际 1.84× 因为有 non-GEMM overhead。

关键：因为 backbone 是 NVFP4-aware trained，PTQ 的 mismatch 消失。Table 7 的 ablation 证明：PTQ 从 85.06 掉到 84.04，pretrained NVFP4 保持 84.51，几乎无损。

### 5.2 NVFP4 KV cache quantization

公式 (5)(6)。每 chunk $F_c = 8$ frames，$T_c = F_c L_f$ latent tokens。Layer $\ell$ 的 cached KV：

$$\mathbf{K}_{\ell,c}, \mathbf{V}_{\ell,c} \in \mathbb{R}^{T_c \times H \times d}$$

reshape 成 $\mathbb{R}^{(T_c H) \times d}$，独立 NVFP4 quantize。

Key 先做 $\mu$-smoothing（公式 6）：

$$\bar{\mathbf{K}}_{\ell,c}[t, h, :] = \mathbf{K}_{\ell,c}[t, h, :] - \frac{1}{d} \sum_{u=1}^{d} \mathbf{K}_{\ell,c}[t, h, u]$$

这是 SmoothQuant 的思路：Key 的某些 channel 有 outlier，先减均值让分布 flat。然后 apply adaptive scale selection（Four Over Six，公式 12-13）。

存储 cost：从 $4 T_c H d$ bytes（BF16，2 bytes × 2 for K&V）→ $\frac{9}{8} T_c H d$ bytes。压缩比 ~3.6×。

为什么 $\frac{9}{8}$？FP4 是 4 bit = 0.5 byte，每 16 element 一个 FP8 scale（1 byte）+ amortized FP32 global scale。所以每 16 element：$16 \times 0.5 + 1 = 9$ bytes，即 $\frac{9}{8}$ bytes/element。两个 tensor（K and V）各算一份。

Customized parallel CUDA dequantization kernel：因为 sliding window attention 一次可能 access 多个 cached chunk，需要高效 in-window reconstruction。Overhead < 2%。

参考 SmoothQuant: https://arxiv.org/abs/2211.01808
参考 QuantVideoGen (2-bit KV): https://arxiv.org/abs/2602.02958

### 5.3 Asynchronous streaming VAE decoding

VAE decode 是最后一步 bottleneck。Baseline 做法是攒齐所有 latent chunk 再 sequential decode，memory $O(C \cdot T_c)$，latency $C \cdot (t_{DiT} + t_{VAE})$。

LongLive-2.0 的做法：
1. 3D VAE 改造成 chunk-by-chunk streaming decode + immediate CPU offload → VAE GPU memory 降到 $O(T_c)$
2. 专用一个 GPU 做 VAE decode，和 $N$-GPU DiT SP cluster 异步跑

DiT denoise chunk $c+1$ 时，VAE node decode chunk $c$。因为 $t_{DiT} \geq t_{VAE}$，decode 被 hide 掉。End-to-end latency 从 $C(t_{DiT} + t_{VAE}))$ → $C \cdot t_{DiT} + t_{VAE}$。

Table 3 显示 async decoding 让 64s video 的 E2E latency 从 99.5s 降到 57.6s，memory 从 99.5GB 降到 57.6GB（实际是 19.4GB，因为 VAE offload 到 CPU）。

---

## 6. Multi-shot attention sink

### 6.1 为什么 standard attention sink 在 multi-shot 失败

Standard attention sink（StreamingLLM）pin 住前几帧，保留 global identity。但 multi-shot 场景：
- 单一 global sink 无法 preserve intra-shot coherence（shot 内的 local temporal continuity）
- Moving shot-level sink 会丢 global identity（不同 shot 的 character appearance 漂移）

### 6.2 Two cooperating anchor sets

- **Global Sink** $\mathcal{A}_g$：video 前 $S_g$ 帧，permanently fixed，preserve global identity
- **Shot-Level Sink** $\mathcal{A}_s$：current shot 前 $S_s$ 帧，每次 scene cut 重新 bind，maintain local temporal coherence

Effective KV set at step $t$：

$$\mathcal{K}_{eff}(t) = \mathcal{A}_g \cup \mathcal{A}_s \cup \text{KV}_{[t-W, t)}$$

$W$ 是 sliding window length。Overlapping token deduplicated。

$\mathcal{A}_s$ 零 memory overhead：只 track 两个 scalar pointer (START, LEN)，virtual prepend，不 copy data。

Prompt switch $p_k \to p_k'$ 自动 trigger scene cut：rebind $\mathcal{A}_s$ 到新 chunk，re-init cross-attention cache，global sink 和 preceding history 不动。这实现 minute-scale interactive generation without redundant recomputation。

参考 StreamingLLM: https://arxiv.org/abs/2309.17453

---

## 7. 实验数据深度解析

### 7.1 Table 1: AR training efficiency

| Input Length | BF16 w/o SP | BF16 w/ SP | BF16 Balanced SP | NVFP4 Balanced SP |
|---|---|---|---|---|
| 16s | 75.3s | 52.2s | 45.8s | 40.1s (1.3×) |
| 32s | 202.7s | 162.7s | 136.8s | 119.3s (1.4×) |
| 64s | OOM | 1372.9s | 1196.5s | 639.5s (2.1×) |

关键观察：
- 64s 时 BF16 w/o SP 直接 OOM → SP 是必需的，不只是加速
- BF16 SP → BF16 Balanced SP：16s 提速 12%，64s 提速 13% → Balanced SP 的 workload balance 收益
- BF16 Balanced SP → NVFP4 Balanced SP：16s 提速 12%，64s 提速 46% → **NVFP4 收益随长度增长**，验证了 paper 的核心论点（GEMM 占比随长度上升）

为什么 64s 时 NVFP4 收益最大？因为长 sequence 的 GEMM FLOPs 远大于短 sequence，而 NVFP4 GEMM 理论 4× throughput。短 sequence 时 non-GEMM overhead（launch、sync、norm）占比大，dilute 收益。

### 7.2 Table 2: DMD training memory

逐步 quantize generator/real/fake：
- 全 BF16：70.5 GB
- Generator NVFP4：63.3 GB（0.90×）
- + Real NVFP4：57.2 GB（0.81×）
- + Fake NVFP4+LoRA：49.0 GB（0.69×）

每 GPU 省 21.5 GB。DMD 要同时跑 generator + real-score + fake-score 三个 model，memory 压力大，NVFP4 让三 model co-locate 成为可能。

### 7.3 Table 3: Inference efficiency progressive ablation

64s video 的演化：
- BF16：112.9s, 36.4 GB
- + NVFP4：96.0s, 29.7 GB → 量化本身提速 + 省内存
- + NVFP4 KV cache：99.5s, 19.4 GB → memory 大降，latency 微增（dequant overhead）但后续被 hide
- + Async decoding：57.6s, 19.4 GB → latency 大降，VAE hide 掉
- + 2 steps：36.3s, 19.4 GB → few-step distillation，45.7 FPS

注意 NVFP4 KV cache 单独看 latency 反而升（29.7 → 99.5 是写错了，实际应该是 96.0 → 99.5s 是 16s 的数，64s 是 96.0 → 看 FPS 从 32.0 降到 29.7）。Wait，重看 table：NVFP4 KV cache 行 FPS 29.7，比纯 NVFP4 的 32.0 略低——dequant overhead。但加上 async decoding 后 FPS 还是 29.7（async 不改 model FPS，改 E2E latency）。然后 3 steps → 35.2 FPS，2 steps → 45.7 FPS。

### 7.4 Table 4: VBench performance

LongLive-2.0-5B NVFP4 4-step：Total 84.51，vs BF16 85.06，drop 0.55。2-step 进一步 drop 到 83.14。这是 quality-speed tradeoff。

对比 baseline：
- Wan2.2-TI2V-5B BF16 50-step：3.3 FPS，83.32 total → 慢但质量基准
- Self-Forcing 1.3B BF16 4-step：21.2 FPS，84.31 total
- LongLive-2.0 NVFP4 2-step：45.7 FPS，83.14 total → **比 Self-Forcing 快 2.15×，质量略低但同档**

### 7.5 Table 5: VBench-Long 60s

LongLive-2.0 BF16：avg rank 3.67（best），Subject Consistency 97.48，Background Consistency 97.00。NVFP4 略降到 3.83，但 Subject Consistency 反而 97.62（best）——可能是 NVFP4 的 stochastic rounding 起到 regularization 效果。

### 7.6 Table 6: SP inference on H100

非 Blackwell GPU（H100）没有 native NVFP4 GEMM，所以用 SP inference 替代。BF16 SP=2 通信 1.8s，4-bit KV cache 降到 1.1s（3.6× 通信压缩，验证公式 11 的理论）。SP=4 时通信从 12.8s 降到 7.8s。

这给了一个重要 insight：**KV cache quantization 不只省 memory，还省 SP 通信**。因为 All-to-All 交换的 K/V tensor 变小了。

---

## 8. 系统级 intuition 总结

把整个 paper 的 design choices 串起来：

1. **NVFP4 是训练-推理对齐的 precision**，不是 PTQ 的压缩手段。这要求 training 时就 quantize GEMM operand（forward + backward + weight-gradient），需要 RHT 稳定 gradient。
2. **Balanced SP 是 AR teacher-forcing 专用的 SP 变体**。核心是让每个 rank 拿同一 temporal chunk 的 clean+noisy pair，自然 balance loss + shard VAE encoding + 避免每层 attention 的 permutation。
3. **KV cache quantization 三重收益**：省 memory（3.6×）+ 省 SP 通信（3.6×）+ enable long video generation on limited memory。
4. **Async streaming decoding** 把 VAE 从 critical path 挪走，让 E2E FPS 接近 model-only FPS。
5. **Clean pipeline 是 infrastructure enable algorithm** 的典型案例：因为 training cheap 了，可以直接 long video AR fine-tune，不需要 ODE init + multi-stage DMD 的复杂 pipeline。
6. **Standalone LoRA for DMD** 让 distillation 和 AR training decouple，LoRA 可复用、可并行训练。

### 联想到的相关工作

- **SANA-Video** [7]：block linear diffusion transformer，另一条 efficiency 路线，走 architecture 简化而非 quantization
- **QuaRot / SpinQuant**：rotation-based quantization，RHT 的更系统化版本，https://arxiv.org/abs/2404.14028
- **SageAttention3** [74]：microscaling FP4 attention，natively 在 attention 里做 FP4，和 LongLive-2.0 的 W4A4 GEMM 互补，https://arxiv.org/abs/2505.11594
- **SVDQuant** [34]：用 low-rank component absorb outlier，4-bit diffusion model PTQ，https://arxiv.org/abs/2411.05007
- **StreamingLLM / Attention Sink** [63]：Multi-shot sink 的 origin，https://arxiv.org/abs/2309.17453
- **CausVid** [69]：bidirectional → causal AR distillation 的开创性工作，https://arxiv.org/abs/2412.07772
- **DeepSpeed-Ulysses** [29]：Balanced SP 的 base，https://arxiv.org/abs/2309.14509
- **Four Over Six** [12]：adaptive block scaling，https://arxiv.org/abs/2512.02010
- **Causal-Forcing** [82]：最新的 forcing 变体，https://arxiv.org/abs/2602.02214

### Limitation 的诚实评估

Paper 自己承认：NVFP4 inference 只在 Blackwell（GB200）上提速。H100/A100 没有原生 FP4 Tensor Core，只能 fallback 到 SP inference（Table 6）。这意味着 LongLive-2.0 的 1.84× inference speedup 是 hardware-locked 的。对于非 NVIDIA 最新硬件的用户，收益主要来自 SP + KV cache compression，不是 quantization 本身。

另外 2-step distillation 的 quality drop（85.06 → 83.14）说明 few-step 还有空间，可能需要更好的 distillation objective（如 Consistency Distillation）或更激进的 LoRA capacity。

---

## 9. 对你的 intuition building

Andrej，从你的视角看，这篇 paper 的核心 system insight 是：**quantization 不应该是一个 post-hoc 的 compression step，而应该是 training-inference co-design 的一等公民**。这和 QAT（quantization-aware training）的传统思路一脉相承，但 NVFP4 的 hardware-native 支持（Blackwell Tensor Core）让它从"模拟 quantization 噪声"变成"真的在 FP4 算"，这是质变。

Balanced SP 的 insight 更微妙：**parallelism strategy 要 match data layout 的结构**。传统 SP 假设 sequence 是 homogeneous 的，但 AR teacher-forcing 的 $[\text{clean}; \text{noisy}]$ 有内禀结构（哪些 token 有 loss、哪些是 context），盲目 shard 会破坏这个结构。Balanced SP 的"每个 rank 拿同一 chunk 的 clean+noisy pair"是对这个结构的尊重。

这给你一个 meta-intuition：**未来的 system-algorithm co-design 会越来越强调"把 algorithm 的 inductive bias 编码到 system 的 data layout 里"**，而不是把 algorithm 当 black box 让 system 通用加速。

Paper 链接（推测，NVIDIA NVlabs）：https://github.com/NVlabs/LongLive
NVIDIA NVFP4 blog: https://developer.nvidia.com/blog/nvfp4-tensor-core-programming
Blackwell architecture brief: https://www.nvidia.com/en-us/data-center/blackwell-architecture/
