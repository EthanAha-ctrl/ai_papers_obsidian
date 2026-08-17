---
source_pdf: Running VLAs at Real-time Speed.pdf
paper_sha256: c9e97418bb7ede18c58a8ad1715060d8af21e0a00e6ffe158feb0a8a81f3571a
processed_at: '2026-08-12T02:29:30-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇paper

## 核心story

这篇paper其实在讲一个很simple的story：**大家觉得big VLA model跑不快，其实是可以跑很快的，关键在于你怎么榨硬件**。

π₀这种3B+参数的model，naive PyTorch实现要100+ms，openpi官方JAX实现也要50-60ms。这篇paper说：单张RTX 4090，我能跑到20ms（单view）到36ms（三view），达到30 FPS real-time。

更cool的是，他们还提出了一个**Full Streaming Inference**的framework，让VLA内部的不同component跑在不同frequency上：480Hz的force control loop + 30Hz的visual loop + <1Hz的text reasoning loop。这个idea其实对应人类神经系统的hierarchy。

## 为什么33ms是magic number

30 FPS camera每33ms来一帧。如果你的inference >33ms，比如34ms，听起来差不多，实际上**每帧都会慢一点，累积起来必须丢帧**。如果关键event（比如pen开始下落）正好发生在丢的那帧，latency直接多一整个frame time。

所以33ms是一个**硬门槛**，过了就是real-time，没过就是"接近real-time"，qualitatively different。

## π₀架构quick recap

π₀是一个MoE架构：

```
[VLM Backbone 3B]                    [Action Expert 300M]
  SigLIP (400M)                         downsized Gemma
  Gemma (2.6B)                     flow matching decoder
       ↑                              ↑
  multi-view images              states + action noises
  task prompt                        ↓
       ↓                        10 denoising steps
  KV cache ──────────────→  action chunk (63 steps)
```

Key insight: VLM处理semantic理解（compute-heavy），AE处理action generation（memory-heavy）。这个区别后面会exploit。

## 优化的三个层次

### Layer 1: 杀掉CPU overhead

PyTorch每次launch CUDA kernel都有Python overhead。π₀一次inference要launch 1000+ kernels，Python成为bottleneck。

**CUDA Graph**: record一次kernel序列，之后replay时完全GPU驱动，Python完全不介入。约束是所有pointers和kernel code必须run-to-run constant。Transformer没有dynamic branches，满足。

效果：**2x speedup**，从100+ms降到50ms级别。

### Layer 2: Graph simplification

这部分像compiler的constant folding，但更aggressive。

**Trick 1: RMS Norm folding**

RMS norm是线性的，后面的matmul也是线性的，两个线性操作可以merge成一个matmul，直接改weights：

$$W \cdot \text{RMSNorm}(x) = W \cdot \text{diag}(\gamma/\sigma) \cdot x = W' \cdot x$$

变量解释：
- $x$: input token vector
- $\gamma$: RMS norm的learnable scale
- $\sigma$: RMS norm的标准差
- $W$: 后接linear layer的weight
- $W'$: 合并后的新weight

直接省掉一个kernel launch。

**Trick 2: Action-time embedding folding**

AE里action value先up-project到1024维，concat timestep encoding，再喂下一个linear layer。

- Action branch：两个连续linear没有nonlinearity，fold成一个
- Time branch：inference时只有10个timestep，直接tabulate所有结果，fuse到SiLU之前的bias

**Trick 3: QKV fusion + RoPE precompute**

Q、K、V三个matmul合成一个大matmul，slice回来。RoPE的rotation matrix precompute好，fuse进matmul。

效果：额外省7-8ms。

### Layer 3: Kernel-level optimization

这是最hardcode的部分。

**GEMM tile tuning**: PyTorch默认走cuBLAS，但不一定最优。用Triton手动调tile size，比如64×64×32 vs 64×64×64，针对每个matmul的shape找最优config。

**Gated FFN fusion**: 
$$\text{FFN}(x) = FC_1(x) \cdot \text{GELU}(FC_2(x))$$

两个matmul并行 + load/store coalescing。load一次input tile，load两个weight tiles，只写回combined result。省1.7ms。

**Partial Split-k**: 遇到512×1152×1152这种shape，64×64 tile产生144 blocks，无法均匀分配到128 SMs。拆成512×1152×1024 + 512×1152×128两部分，不同tile size，写在同一个kernel里。

**Scalar op fusion**: bias、residual、activation全部fuse进GEMM。RMS norm先算stats到buffer，在下个GEMM结束时除以factor。

## Roofline分析

每个BF16 GEMM的theoretical lower bound：

$$t_{\text{roofline}} = \max\left(\frac{2KM}{T_{\text{bandwidth}}}, \frac{NKM}{T_{\text{compute}}}\right)$$

变量：
- $N \times K \times M$: GEMM dimensions (output × input × reduction)
- $T_{\text{bandwidth}}$: HBM带宽，RTX 4090 = 1.01 TB/s
- $T_{\text{compute}}$: BF16 MAC throughput，boost后91.4 TMAC/s
- $2KM$: 只算weight matrix的memory（activation可L2 cache，weight太大不行）

Key observation：
- **Vision encoder + LLM**: compute-limited（tensor core是bottleneck）
- **AE**: bandwidth-limited（HBM带宽是bottleneck）

这个区别是Full Streaming Inference的基础。

加上synchronization overhead（CUDA graph约1.72ms，software barrier可降到0.86ms），2 views的lower bound约20.6ms。他们做到27.3ms，剩余空间约30%。

## Full Streaming Inference - 最有意思的部分

### Core observation

AE是IO-bound，VLM是compute-bound。Concurrent运行可以同时saturate bandwidth和compute。

实测：
| Condition | Time |
|-----------|------|
| Sequential VLM + 10 AE | 27.3 ms |
| Concurrent VLM + 10 AE | 26.3 ms |
| Concurrent VLM + 16 AE | 32.7 ms |

16个AE concurrent跑，总时间还在33ms内。这意味着：**30个VLM + 480个AE per second**。

### 重新思考AE的角色

传统AE做flow matching，10个denoising steps后才输出完整action chunk。Full Streaming模式下，AE要改成gradual generation——每个step生成一部分action，像auto-regressive decoding。

这样AE可以接受high-frequency input（force sensor 2KHz），立即产生reaction。比如检测到force spike，立即emergency stop。

### 三层control loop

```
<1Hz    Text reasoning (VLM piggyback)
  ↓
30Hz   Visual loop (VLM encoding)
  ↓ 
480Hz  Force loop (AE streaming)
  ↓
kHz    Layer-level (future work)
```

这对应人类神经系统hierarchy：conscious thinking → visual perception → reflexive motor → spinal reflex。

### <1Hz Text loop的clever trick

VLM每秒30次pass over weights做visual encoding。Text inference的bottleneck是autoregressive时的weight loading。Piggyback策略：load一次weight，先算visual matmul，再算text inference。

实现上就是special attention matrix，visual tokens多，text部分额外MACs占比小。

结果：额外获得30 tokens/s的text stream，人类说话才3.3 tokens/s，绰绰有余。

## Real World验证

Task：grab falling pen。上面grabber释放笔，下面grabber要在正确时机抓住。End-to-end reaction <200ms，达到人类水平。

Setup：
- 30 FPS USB camera（故意不用RealSense，delay >100ms太慢）
- Camera delay约2帧（ISP + USB）
- Pen下落30cm
- 600 episodes训练数据
- 1D trajectory：0=不动作，1=关闭grabber
- Current + previous frame作为two views（给network速度hint）

Inference system多线程：
- Camera thread → ring buffer
- Inference thread → 取最新frame，输出trajectory到circular buffer
- Output thread → 发送action到grabber

关键细节：**GPU需要warm-up才能达到full speed**，实验前先让inference time稳定。

Result：10次连续实验，100% success rate。

从learning角度trivial（LeNet/SVM也行），但从system角度验证了low latency。人类对30cm下落距离的反应也是lower bound，说明系统达到human-level。

## Build Intuition

这篇paper的intuition有几个层次：

### 1. VLA不是monolithic black box

VLA内部其实有temporal structure。VLM和AE有不同的frequency demand，有不同的resource profile（compute vs bandwidth）。Full Streaming Inference就是把这种内部结构expose出来，map到不同的control loop。

### 2. Roofline不只是分析工具，是scheduling guide

当你有compute-bound和memory-bound两个kernel，concurrent运行是free lunch。这个idea在GPU programming里well-known，但apply到VLA这种complex model上是新的。

### 3. Real-time不是"更快"，是"更匀"

30 FPS要求每帧都<33ms，偶尔快偶尔慢不行。CUDA Graph、static buffers、zero-copy都是为了让inference time **deterministic**，减少jitter。

### 4. Engineering matters

Paper里很多"trivial"的optimization（image resize <60us、pinned memory、static buffer）加起来很重要。Real-time system是全链路optimization，木桶效应。

### 5. Human analogy

三层control loop对应人类神经系统：
- <1Hz text reasoning ≈ conscious thinking
- 30Hz visual ≈ visual perception  
- 480Hz force ≈ reflexive motor control

这种biological analogy给了我很强的intuition：**real-time robot control应该是hierarchical的，不同frequency用不同的algorithm**。VLA恰好天然有这种hierarchy，只是之前被monolithic inference掩盖了。

## Future directions的intuition

- **60-120 FPS**: 需要更多MACs，INT8 quantization或adaptive view selection
- **7B models**: RTX 5090带宽提升大，MACs提升小，scaling AE应该OK，scaling VLM需要quantization
- **kHz control**: AE的layers本身运行在kHz，如果能每层感知最新signal，可能有更高frequency。但demo data收集是challenge

## Reference links

- Paper code: https://github.com/Dexmal/realtime-vla
- π₀: https://arxiv.org/abs/2410.24164
- π₀.5: https://arxiv.org/abs/2504.16054
- openpi: https://github.com/Physical-Intelligence/openpi
- PaliGemma: https://arxiv.org/abs/2407.07726
- SigLIP: https://arxiv.org/abs/2303.15343
- Gemma: https://arxiv.org/abs/2403.08295
- Flow Matching: https://arxiv.org/abs/2210.02747
- FlashAttention: https://arxiv.org/abs/2205.14135
- Triton: https://arxiv.org/abs/2011.04374
- CUDA Graph docs: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs
- Real-time chunking: https://arxiv.org/abs/2506.07339
- RMSNorm: https://arxiv.org/abs/1910.07467
- RoPE: https://arxiv.org/abs/2104.09864
- MoE: https://arxiv.org/abs/1701.06538

---

# Running VLAs at Real-time Speed 深度解析

这篇paper来自Dexmal和StepFun，核心目标非常明确：把π₀级别的multi-view VLA模型在单张消费级GPU（RTX 4090）上跑到30Hz frame rate，最高480Hz trajectory frequency。这是一个非常engineering-heavy的工作，但背后的system design思考很值得build intuition。

## 1. 核心问题与Motivation

VLA模型（Vision-Language-Action）参数量已经达到billion级别，比如π₀有3.3B参数。一个forward pass通常需要几百ms，这对于dynamic task来说是致命的。论文给出的turning point是**33ms**（1/30s），只有低于这个阈值才能完整处理30 FPS相机的每一帧。哪怕34ms，都会导致丢帧，如果关键事件发生在丢的帧上，latency直接增加一整个frame time。

Paper里设计的验证任务很精巧：grasping a falling pen。两个垂直对齐的grabber，上面的释放笔，下面的要在正确时机抓住。end-to-end reaction time < 200ms，达到人类平均水平。

**参考链接**:
- π₀ paper: https://arxiv.org/abs/2410.24164
- openpi repo: https://github.com/Physical-Intelligence/openpi
- 本论文代码: https://github.com/Dexmal/realtime-vla

## 2. π₀ 模型架构回顾

π₀是一个MoE（Mixture of Experts）架构的VLA：

**VLM Backbone (3B参数)**:
- Vision encoder: SigLIP (400M params) - https://arxiv.org/abs/2303.15343
- LLM: Gemma (2.6B params) - https://arxiv.org/abs/2403.08295
- 整体初始化自PaliGemma - https://arxiv.org/abs/2407.07726

**Action Expert (AE, 300M参数)**:
- 从Gemma downsized而来，更小的width和MLP dimension
- 通过flow matching生成action chunking - https://arxiv.org/abs/2210.02747
- 通过MoE routing与VLM backbone耦合

关键设计：multi-view images和task prompt路由到VLM backbone，states和action noises路由到AE。这样AE可以专注处理high-frequency控制信号，而VLM处理semantic理解。

## 3. Eliminating Overheads - 第一阶段优化

### 3.1 CUDA Graph 消除CPU开销

PyTorch的Python层launch CUDA kernel有显著overhead。π₀单次inference要launch超过1000个kernels。CUDA Graph的核心思想：record一次kernel launch序列，之后replay时完全由GPU和driver驱动，消除所有Python执行overhead。

关键约束：所有kernel codes和buffer pointers必须run-to-run constant。对于VLA来说transformer blocks没有dynamic branches，这个条件满足。

效果：约**2倍加速**，榨出了naive implementation的主要inference overhead。

### 3.2 Computational Graph Simplification

这部分类似compiler里的"constant folding"，但在model inference context下有更多可做的。三个transformations：

**(1) RMS Norm Weight Folding**

RMS norm公式：
$$y_i = \frac{x_i}{\sqrt{\frac{1}{N}\sum_{j=1}^{N} x_j^2 + \epsilon}} \cdot \gamma_i$$

其中：
- $x_i$: input token的第i个元素
- $N$: token维度
- $\epsilon$: small constant for numerical stability
- $\gamma_i$: learnable scale parameter

由于RMS norm后面的linear layer $W$也是线性的，可以利用结合律：
$$W \cdot \text{RMSNorm}(x) = W \cdot \text{diag}(\gamma / \sigma) \cdot x = W' \cdot x$$

直接修改$W$的weights，把RMS norm的affine参数absorb进去，省掉一个kernel。

**(2) Action-Time Embedding Folding**

AE里action value先up-project到1024维，concatenate一个projected timestep encoding vector，再喂给下一个linear layer。

- Action value branch：两个连续linear layer之间没有nonlinearity，可以fold成一个
- Time branch：inference时只有10个不同timesteps，可以tabulate linear layer的结果，一直fuse到SiLU operation之前的bias vector

**(3) QKV Fusion**

把Q、K、V三个projection matrices合并成一个大矩阵：
$$W_{QKV} = [W_Q; W_K; W_V] \in \mathbb{R}^{d \times 3d}$$

一次matmul后slice出Q、K、V。同时把RoPE（Rotary Position Embedding）也fuse进matmul，precompute RoPE weights。

效果：减少7-8ms

**参考**: RoPE原论文 https://arxiv.org/abs/2104.09864

### 3.3 其他Overheads

- **Image resizing**: 选择接近224×224的分辨率（如240×320），手写resize代码，desktop x86 CPU上<60us
- **Pinned memory**: CPU-GPU数据传输用pinned memory
- **Static CPU buffers**: 减少jitter
- **Zero-copy camera frames**: 减少latency

## 4. In-Depth Kernel Optimization - 第二阶段

### 4.1 GEMM Tile Parameter Tuning

默认PyTorch的matmul走cuBLAS，dispatch到compiled cutlass kernels。但有些kernels的configuration不是最优的。用Triton手动调优tiling策略。

一个重要observation：LLM的transformer只跑17次而非18次attention和FFN layers，因为只把KV cache传给AE，最后一层的features不需要。这节省约0.7ms。

Triton调优总共节省约1.5ms。

### 4.2 Fusing Gated Linear Layers

Transformer FFN用gated up-projection：
$$\text{FFN}(x) = FC_1(x, w_1) \cdot \text{GELU}(FC_2(x, w_2))$$

其中：
- $FC_1, FC_2$: 两个fully-connected layers
- $w_1, w_2$: 对应的weight matrices
- $\text{GELU}$: Gaussian Error Linear Unit activation

关键insight：这两个matmul可以并行，更重要的是它们的load/store可以coalescing。load一个input tile后，可以load两个weight tiles进行计算。写回时只需要写combined result，减少memory operation。

效果：节省1.7ms

### 4.3 Partial Split-k

一个特殊GEMM size: $512 \times 1152 \times 1152$（M×K×N）

问题：用64×64 tile产生144 blocks，不是128的倍数，无法均匀分配到RTX 4090的128个SMs。

解决方案：split成两部分
- Part 1: $512 \times 1152 \times 1024$ matmul，64×64 tile，均匀分配到SMs
- Part 2: $512 \times 1152 \times 128$，32×32 block，K维度split-2 partition

两部分写在一个kernel里，互相独立。效果<0.1ms但理论意义重要。

### 4.4 Fusing Scalar Operations

- **Bias + residual + activation**: trivially合并进GEMM
- **RMS norm**: 先计算token-level stats到separate buffer，然后在下一个GEMM的所有accumulations完成后除以对应factor

效果：约4ms

## 5. Lower Bound Analysis

### 5.1 Roofline Model

BF16 GEMM的lower bound：
$$t_{\text{roofline}} = \max\left(\frac{2KM}{T_{\text{bandwidth}}}, \frac{NKM}{T_{\text{compute}}}\right)$$

变量解释：
- $N \times K \times M$: GEMM维度（注意paper里写法是N×K×M，对应output×input×reduction）
- $T_{\text{bandwidth}}$: HBM带宽，RTX 4090为1.01 TB/s
- $T_{\text{compute}}$: BF16 MAC/s with FP32 accumulation，boosted frequency 2.79GHz下为91.4 TMAC/s
- $2KM$: 只考虑第二个矩阵的memory operation（因为第一个矩阵和result通常是activations，可在L2 cache；network parameters太大无法全cache）

Lower bound结果：
| Views | Roofline |
|-------|----------|
| 1 view | 12.8 ms |
| 2 views | 19.7 ms |
| 3 views | 26.7 ms |

重要observation：
- Vision encoder和LLM的ops主要是compute-limited
- AE的ops主要是bandwidth-limited
- 因此consecutive operator的overlap不影响大部分情况的roofline argument

### 5.2 Synchronization Overhead

总共有1378个matmuls。用简单A+B kernel测试1378次的sync overhead：

| Setting | Time | Overhead |
|---------|------|----------|
| Pytorch | 13.81 ms | +12.92 ms |
| CUDA graph | 2.61 ms | +1.72 ms |
| Software barrier | 1.75 ms | +0.86 ms |
| Fused no sync | 0.89 ms | baseline |

**Software Barrier实现**（Triton）:
```python
lock_goal += psize
tl.atomic_add(lock_ptr, 1)
while tl.atomic_or(lock_ptr, 0) < lock_goal:
    pass
```

原理：当launch的block数量等于SM数量时，可以确保所有block同时运行，用global memory创建barrier。

**重要constraint**: 传统CUDA programming model认为不能explicitly synchronize across all blocks（因为它们可能不同时存在）。但实践中当block数=SM数时可行。

Updated lower bounds（加上0.86ms sync overhead）:
| Views | Updated LB |
|-------|------------|
| 1 view | 13.7 ms |
| 2 views | 20.6 ms |
| 3 views | 27.6 ms |

对比实际实现，剩余improvement space最多30%，说明已经接近optimal。

## 6. Full Streaming Inference - 核心创新

这是paper最有思想价值的部分。核心observation：AE是IO bounded，VLM是compute bounded，如果concurrent运行，IO和compute resource都能更好利用。

### 6.1 Overlapped Streaming Measurement

| Condition | Time |
|-----------|------|
| Sequential VLM + 10 AE | 27.3 ms |
| Concurrent VLM + 10 AE | 26.3 ms |
| Concurrent VLM + 16 AE | 32.7 ms |

关键发现：只要能concurrent运行，可以afford 30 VLMs + 480 AEs per second。AEs均匀分布在timeline上，开启480Hz control loop的可能性。

### 6.2 480 Hz Action Expert Re-design

重要区分：**trajectory frequency** vs **control frequency**
- Trajectory frequency: 输出trajectory的node密度，trivially通过interpolation增加
- Control frequency: stimuli到reaction的bounded time，要求整个pipeline的高频处理

**High-frequency input signal候选**:
- Force sensors (3D/6D): >2K Hz采样，latency到us级
- Motor current: 通常>1K Hz控制
- Resistance-based tactile signals

**AE角色转变**:
传统AE做flow matching/diffusion integration，10个denoising steps后才输出完整action list。需要rewrite成gradual generation——每个step生成action list的一部分，类似auto-regressive decoding。

参考Real-time chunking (RTC): https://arxiv.org/abs/2506.07339

**实现细节**:
- 新input signal作为新token注入transformer-based AE
- 用separate stream做memcpy更新GPU global memory
- AE连续manipulate 480 Hz trajectory的consecutive timestamped nodes
- Nodes在从GPU取出时"committed"（potentially asynchronous）

**两个feedback loops**:
- Quick loop: 高频signal注入 → 1个AE处理 → reaction trajectory生成，最快2ms
- Slow loop: image-driven，frame capture → VLM处理 → AE使用，最快1/30s

**Persistent Megakernel**可能是最佳实现方式。

### 6.3 <1Hz Textual Loop

VLA的VLM部分不仅处理visual data，还要处理text（multimodal understanding、task planning、CoT reasoning）。

**Piggyback策略**: 已经在30Hz做visual token encoding，意味着每秒30次pass over transformer weights。Text inference的bottleneck是autoregressive inference时的checkpoint loading。可以让text inference搭便车：load一个matrix weight后，先用它计算VLM part的matmul，再计算text data的inference。实现上可以做成special attention matrix。

结果：额外获得30 tokens/s的auto-regressive text stream。人类说话约3.3 tokens/s，30 tokens/s相当充裕。

### 6.4 三层Feedback Loop总结

- **Force loop (480 Hz)**: 高频input/output由AE处理
- **Visual loop (30 Hz)**: image-based reaction由VLM处理
- **Textual loop (<1 Hz)**: text-based interaction和reasoning，带来更多intelligence

## 7. Real World Validation

### 7.1 Setup

- 两个垂直对齐grabbers
- 30 FPS 720P USB camera（不用RealSense，因为delay >100ms）
- Camera delay约2帧（1帧ISP + 1帧USB传输）
- Pen下落30cm
- 第一grabber握笔不同位置创造small variation
- 600 episodes训练数据
- 1D trajectories: 0=不动作，1=关闭grabber
- 使用current + previous frame作为two views（给network速度hint）

### 7.2 Training

- 使用openpi repo
- Empty prompt
- 每episode包含release前几秒和catch后几秒
- 只训练几个epochs（数据充足）

### 7.3 Inference System Design

多线程架构：
- **Camera thread**: 等frame到来，放入ring buffer（带timestamp）
- **Inference thread**: 取最新frame运行network，输出timestamped grabber states trajectory，circular buffer存储
- **Output thread**: 循环发送output buffer对应item到grabber

**GPU warm-up**: 需要warm-up达到full speed（功耗和clock rate），实验前先让inference time稳定。

### 7.4 Result

10次连续实验，**100% success rate**。

从learning角度这结果trivial（LeNet或SVM也能做到），但从system角度，单次成功catch验证了VLA实现的low latency。人类对30cm下落距离的反应也是lower bound。

## 8. Related Work

**Manual Kernel设计**:
- XLA: https://arxiv.org/abs/2102.01926
- TensorRT: https://developer.nvidia.com/tensorrt
- CUTLASS: https://github.com/NVIDIA/cutlass
- Triton: https://arxiv.org/abs/2011.04374
- FlashAttention: https://arxiv.org/abs/2205.14135
- FlashDecoding: https://pytorch.org/blog/flash-decoding/

**Scheduling-based**:
- TVM: https://arxiv.org/abs/1802.04799

**Superoptimization**:
- Mirage: https://www.usenix.org/conference/osdi25
- Neptune: https://arxiv.org/abs/2510.08726

## 9. Future Directions

### 9.1 60-120 FPS Visual Loop

- Encoding stage已经compute-bound，需要更多MACs
- 8-bit multiplication: 如果能用INT8，compute power大幅释放
- Adaptive view selection: bimanual robot有3+ cameras，可以dynamically fuse成fewer tokens
- 人类能区分30 vs 60 FPS，60 FPS是下一阶段，120 FPS是"beyond human"

### 9.2 7B+ Models

- RTX 5090: 1.79 TB/s bandwidth（比4090提升大），但BF16 MACs提升相对小
- Scaling VLM: 增加total MACs
- Scaling AE: 增加bandwidth（应该不是问题）
- 7B是合理next milestone（2x current 3B，LLM literature里well studied）

### 9.3 More Fine-grained Feedback Loop

- AE的layers本身运行在更高frequency（每秒数千层）
- 如果每层都能感知最新signal + 更快的intermediate output，可能有更高control frequency
- 挑战：每秒kilos of samples的demonstration data收集

## 10. 我的Intuition Building

这篇paper的精髓在于**重新思考VLA的temporal structure**。传统观点把VLA当作一个monolithic 30Hz black box，但这篇paper揭示了VLA内部其实有不同tiers的frequency：

1. **Text reasoning** (~1Hz): 人类级别的思考速度
2. **Visual processing** (30Hz): 相机frame rate
3. **Action chunking** (480Hz): force/tactile control
4. **Layer-level** (kHz): 未来方向

这种hierarchical frequency mapping对应人类神经系统的hierarchy：
- Conscious reasoning (~Hz)
- Visual perception (~30Hz)
- Reflexive motor control (~100Hz)
- Spinal reflexes (~kHz)

Full Streaming Inference的真正贡献在于：它不是简单地"加速VLA"，而是**把VLA的内部结构映射到robot control的frequency hierarchy**。AE不再只是VLM的"action head"，而是变成了一个独立的high-frequency controller，VLM为其提供context（通过KV cache）。

从systems perspective，concurrent execution的insight很关键：
- VLM: compute-bound → 利用tensor cores
- AE: memory-bound → 利用HBM bandwidth
- 两者concurrent → 同时saturate两种resource

这让我想到GPU编程里的"roofline model"不仅是分析工具，更是**scheduling的指导原则**。当你有两个kernels，一个compute-bound一个memory-bound，concurrent运行是free lunch。

Software barrier的实现也很有启发。传统CUDA教学强调"不能跨block synchronize"，但实践中当block数=SM数时这个限制不成立。这种"practical wisdom vs textbook theory"的gap，正是high-performance engineering的精髓。

**最终intuition**: VLA的real-time化不是单纯追求更快的matmul，而是要**识别模型内部的temporal structure**，然后**map到hardware的resource hierarchy**（compute vs bandwidth，不同frequency的loops）。这篇paper给出的是一个paradigm，不仅是benchmark数字。

**相关延伸阅读**:
- π₀.5: https://arxiv.org/abs/2504.16054
- MemoryVLA: https://arxiv.org/abs/2508.19236
- GeoVLA: https://arxiv.org/abs/2508.09071
- Flow Matching: https://arxiv.org/abs/2210.02747
- PaliGemma: https://arxiv.org/abs/2407.07726
- SigLIP: https://arxiv.org/abs/2303.15343
- Gemma: https://arxiv.org/abs/2403.08295
- FlashAttention: https://arxiv.org/abs/2205.14135
- Triton: https://arxiv.org/abs/2011.04374
- MoE: https://arxiv.org/abs/1701.06538
- RMSNorm: https://arxiv.org/abs/1910.07467
- RoPE: https://arxiv.org/abs/2104.09864
- SiLU: https://arxiv.org/abs/1702.03118
