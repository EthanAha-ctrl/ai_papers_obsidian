---
source_pdf: Helios Real Real-Time Long Video Generation Model.pdf
paper_sha256: 6533735d41c67ae2e4dd9cce932ae3080fddb5946e0db3fb8f185a25741ce70b
processed_at: '2026-08-04T23:38:06-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 Helios

## 一句话概括

14B 的视频生成模型，单张 H100 跑 19.5 FPS，能生成分钟级长视频，质量不掉。

## 这事为什么难

长视频生成有三个老问题，之前各用各的笨办法凑合：

**问题 1：越生成越崩**
模型生成一段，用这段当历史去生成下一段。历史不完美，误差累积，几十秒后画面就糊了、颜色漂了、或者突然蹦回开头。

**问题 2：14B 太慢**
Wan 14B 生成 5 秒视频要 50 分钟。实时？想都别想。

**问题 3：14B 训练显存炸**
14B 模型训练通常要 4D parallelism + FSDP，工程门槛极高。蒸馏阶段要同时放 4 个 14B 模型，单卡根本塞不下。

之前 community 的共识是：要实时长视频，只能用 1.3B 小模型 + distillation + Self-Forcing rollout。Krea 把 14B 推到 6.7 FPS 就觉得不错了。

Helios 说：14B 也能实时，而且更快更好。

---

## Helios 怎么做的

### Anti-drift：别滚动，直接模拟

Self-Forcing 的思路是"训练时就滚动生成，让模型见到推理时的不完美历史"。问题：滚动越长开销越大，而且超出训练长度照样崩。

Helios 换了个思路：**与其滚动生成不完美历史，不如直接给真实数据加扰动来模拟不完美历史**。

具体三个招：
- **Frame-Aware Corrupt**：每帧独立随机加 blur / noise / exposure 变化。模拟"模型自己生成的历史有各种 artifact"
- **First-Frame Anchor**：第一帧永远留在历史里当全局颜色锚。因为观察到 drift 从不在开头发生
- **Relative RoPE**：不管视频多长，历史的时间索引永远是 0 到 $T_{Hist}$，未来永远是 $T_{Hist}$ 到 $T_{Hist}+T_{Noisy}$。模型永远只在见过的相对位置内工作，不会被推到训练没见过的绝对位置

这三招组合，不用 expensive rollout 就能 robust 到分钟级。

### 实时：双重压缩 + 蒸馏

**压缩历史 token**：近期历史精细保留，远期历史粗粒度压缩。16 帧近期用 4×8×8 压缩，2 帧远期用 1×2×2 压缩。总 token budget 固定，跟视频长度无关。8× 压缩。

**压缩 noisy token**：diffusion 早期步骤决定全局结构，低分辨率够了；后期步骤 refine 细节，才需要高分辨率。Pyramid flow 把单条轨迹拆成 3 个 scale 的轨迹，早期低分辨率跑。2.29× 压缩。

**压缩采样步数**：DMD 蒸馏 50 步到 3 步。加 GAN objective 让 student 能超越 teacher 上限。

三个乘起来，14B 的 effective compute 压到接近 1.3B 水平。

### 训练显存：把 4 个 14B 塞进 80GB

蒸馏阶段要 generator + real-score + fake-score + EMA 四个 14B。Helios 用：
- ZeRO-3 shard EMA
- 异步把不用的模型 offload 到 host memory（利用 TTUR 的不对称更新）
- GAN discriminator 的梯度 cache 住，立即释放 activation

Peak 显存压到训练单 14B 的水平。

---

## 最关键的两个 insight

### Insight 1：别用 causal mask

CausVid / Self-Forcing 把 bidirectional diffusion 改成 autoregressive 用的是 causal mask。这切断了 section 间的双向交互，每个 section 倾向独立生成新场景，质量天花板下降。

Helios 用 **Guidance Attention**：历史主动影响未来，未来不影响历史。看起来是 autoregressive，但 noisy context 内部仍然是 bidirectional。pre-trained model 的表达力全保住。

这是个架构层面的智慧——**统一不一定要通过 masking 实现，可以通过 asymmetric attention pattern 实现**。

### Insight 2：用 data augmentation 替代 expensive rollout

Self-Forcing 范式强依赖训练时 rollout 长度。训练 rollout 5 秒，推理超 5 秒就崩。要 robust 就得长 rollout，长 rollout 限制模型 scale 到 1.3B。

Helios 用 Frame-Aware Corrupt 直接模拟 imperfect history 的分布，避开 expensive rollout。这让 14B 训练成为可能。

**与其让训练模拟推理过程，不如让训练模拟推理过程的产物**。

---

## 结果

- 19.53 FPS on 单张 H100（14B 模型）
- 比 Wan 14B 快 52 倍
- 比 Krea 14B 快 3 倍，质量更好
- 长视频 drifting 各维度优于 1.3B baseline
- 训练不用 parallelism / sharding，batch size 跟 image diffusion 一样

---

## 为什么这篇 paper 重要

之前的共识是：实时长视频 = 小模型 + 蒸馏 + Self-Forcing。14B 被认为太慢太贵。

Helios 证明了：**14B + 实时 + 长视频三个目标不互斥**。关键是算法层面的压缩 + 模拟，而不是堆 infra。

这对 community 的意义：未来的 video generation 研究可以在 14B recipe 上继续推进——更高分辨率、更长 horizon、更复杂 motion——而不用回到 1.3B 的质量瓶颈。

Project page: https://pku-yuangroup.github.io/Helios-Page

---

# Helios: 深入解析 14B 实时长视频生成

非常 excited 来细读这篇 paper。Helios 把 14B 的 video diffusion model 推到 19.5 FPS on 单张 H100，同时支持分钟级长视频——这是 128× speedup，且 quality 不掉。这个 result 直接挑战了"实时长视频生成只能用 1.3B 小模型"这个 community 共识。下面我把整篇 paper 的 mechanics、intuition、和联想都摊开来。

---

## 0. 总体直觉：Helios 在解决什么

长视频生成有三个根本矛盾：

1. **Autoregressive drift**：模型用自己生成的 imperfect history 作为下一节的 condition，error 累积放大，几十秒后画面崩塌。
2. **Real-time 推理**：14B DiT 一次 forward 就慢得要命，更别说 50 步 denoise + 长 history 的 attention。
3. **训练显存**：14B 模型训练通常需要 4D parallelism + FSDP，工程门槛极高。

Helios 的核心 insight 是：**这三个矛盾可以一起用"压缩 + 模拟"来解决**，而不是分别用 self-forcing、KV-cache、pipeline parallelism 这种"堆 infra"的方式。具体说：

- **Anti-drift**：不靠 expensive train-as-infer rollout（Self-Forcing 范式），而是训练时主动注入 realistic perturbation 模拟 drift（Frame-Aware Corrupt）+ 用第一帧做 distribution anchor + 用 Relative RoPE 解决位置外推。
- **Real-time**：不靠 KV-cache / sparse attention / quantization，而是从 token view 和 step view 双重压缩：history 用 hierarchical patchification（8× token 压缩），noisy context 用 pyramid multi-scale flow（2.29× token 压缩），sample step 用 adversarial distillation（50→3 步）。
- **训练显存**：用上面两个压缩把单 GPU batch size 推到 image diffusion scale；distillation 阶段需要 4 个 14B 模型，用 sharded EMA + async VRAM freeing + cache grad for GAN 把 peak 显存压到单 14B 训练水平。

这个组合让 14B 模型从"50 分钟生成 5 秒视频"变成"19.5 FPS 实时分钟级生成"。

Project page: https://pku-yuangroup.github.io/Helios-Page

---

## 1. Unified History Injection：如何把 bidirectional 模型变 autoregressive

### 1.1 Representation Control：统一的 condition interface

之前的工作（CausVid、Self-Forcing）把 bidirectional diffusion model 改造成 autoregressive generator 用的是 **causal masking + diffusion forcing** 组合。Helios 拒绝了这个路径，理由是：

1. Causal masking fundamentally 改变了 bidirectional pretrained model 的 inference regime，限制了可达到的 quality。
2. Frame-wise noise space 极大，优化慢，必须 step distillation，但 distilled model 难以继续 community 开发。

Helios 的方案是把长视频生成 cast 成 **video continuation**：

$$
X_{input} = [X_{Hist}, X_{Noisy}]
$$

其中：
- $X_{Hist} \in \mathbb{R}^{B \times C \times T_{Hist} \times H \times W}$：历史 context，clean
  - $B$ = batch size
  - $C$ = channels
  - $T_{Hist}$ = 历史帧数
  - $H, W$ = height, width
- $X_{Noisy} \in \mathbb{R}^{B \times \tilde{C} \times T_{Noisy} \times H \times W}$：noisy context，待 denoise
  - $\tilde{C}$ 是 noisy latent 的 channel 数
  - $T_{Noisy}$ 是当前 section 帧数
- **关键约束**：$T_{Hist} \gg T_{Noisy}$，且训练和推理都固定这两个值

模型 denoise $X_{Noisy}$ conditioned on $X_{Hist}$，生成 temporally coherent continuation。因为 $T_{Hist}$ 和 $T_{Noisy}$ 固定，模型永远是"看一段长 history + 生成一段短未来"，可以 autoregressive 地无限滚动下去。

任务自动切换靠 $X_{Hist}$ 的 representation：
- 全零 → T2V（无 history）
- 仅最后一帧非零 → I2V（单图 conditioning）
- 多帧非零 → V2V

训练时随机以一定比例 zero out historical context，让模型自然 generalize 到三种任务。这个 unification 很优雅，避免了为每个任务设计独立 architecture。

### 1.2 Guidance Attention：历史与未来的不对称处理

历史 context 和 noisy context 统计不同，应该被区别对待：
- $X_{Hist}$ 已经 clean 且和 prompt 对齐，不应被 denoise，也不应受 $X_{Noisy}$ 影响
- $X_{Hist}$ 的角色是 **guide** $X_{Noisy}$ 的 denoising

Helios 用两个机制 explicit enforce 这个 separation：

**机制 1**：$X_{Hist}$ 的 timestep 固定为 0（始终 clean，no noise injection）

**机制 2**：Guidance Attention，self-attention 层中：

$$
X_{Self} = \text{Attention}([Q_{Noisy}, Q_{Hist}], [K_{Noisy}, K_{Hist} \cdot amp], [V_{Noisy}, V_{Hist}])
$$

- $Q, K, V$ 是 query/key/value
- $[\cdot, \cdot]$ 是 concatenation
- $\cdot$ 是 element-wise multiplication
- **$amp$ 是 head-wise amplification tokens**：每个 attention head 学一个独立的 amplification factor，调制历史 key 的强度。这让每个 head 可以独立"放大重要历史信号 / 衰减冗余或有害信号"

直觉：不同 attention head 关注不同语义维度，对历史的需求也不同。$amp$ 给了模型 head-wise 的选择性记忆能力，类似于一个 learnable attention mask 但更灵活。

cross-attention 层：

$$
X_{Cross} = \text{Attention}(Q_{Noisy}, K_{Text}, V_{Text})
$$

- $K_{Text}, V_{Text}$ 是 text prompt encoder 输出的 key/value

**关键设计**：cross-attention 只作用于 $X_{Noisy}$，因为 $X_{Hist}$ 已经在之前的 step 里吸收了 text 语义，再注入是冗余的。这是 Helios 对 inference cost 的一个细节优化。

**为什么这个设计优于 causal masking**：causal mask 阻止 noisy context 影响 history，但同时切断了 cross-section 的 bidirectional interaction，每个 section 倾向于生成独立的新场景。Guidance Attention 通过 asymmetric attention pattern（history 主动影响 noisy，noisy 不影响 history）保留了 bidirectional 的表达力，同时维持 autoregressive 的因果性。

Ablation 显示加 causal mask 到 Guidance Attention 上会导致 unstable training；移除 Guidance Attention 则会让 semantic content 随时间累积（如 bird crest 越变越大、saturation 持续上升）。

---

## 2. Easy Anti-Drifting：三种 drift 模式与对应解法

Helios 把 drift 归纳为三种 canonical manifestation（Figure 5）：

### 2.1 Position Shift 与 Relative RoPE

**问题**：diffusion model 在 inference horizon 匹配 training horizon 时表现最好。如果训练只见过 5 秒 clip（absolute temporal indices 0:某值），推理 1440 帧时模型遇到 unseen temporal position，质量大幅退化。更糟的是，absolute temporal indices 还会让生成 snap back 到早期位置，造成 **repetitive motion**——scene 周期性 reset。

**根因**：RoPE 本质是 periodic function，在 multi-head attention 中，不同 head 的 frequency 与 attention pattern 相互作用，长序列外推时周期性暴露出来。Reference: RoPE 的周期性问题在 LongRoPE、YaRN 等 long-context LLM 工作里有详细讨论。

**Helios 解法**：**Relative RoPE**
- 无论目标视频多长，$X_{Hist}$ 的 temporal index 范围恒为 $0:T_{Hist}$
- $X_{Noisy}$ 的 temporal index 恒为 $T_{Hist}:T_{Hist}+T_{Noisy}$
- 模型永远只在固定的相对位置范围内工作

直觉：把"绝对时间外推"问题转成"固定窗口的相对位置预测"问题。每个 section 都在 $[0, T_{Hist}+T_{Noisy}]$ 这个固定区间内生成，模型不必外推到 training 时没见过的位置 index。这就从根上消除了 RoPE 周期性 + multi-head attention 的相互作用导致的 repetitive motion。

这与 language model 里 ALiBi vs RoPE 的争论呼应——相对位置编码对长序列外推天然友好。Reference: RoFormer paper https://arxiv.org/abs/2104.09864

### 2.2 Color Shift 与 First-Frame Anchor

**观察**：分析正常视频 vs drifting 视频的 saturation、aesthetic score、RGB mean/variance over time（Figure 6）：
- 正常视频这些统计量稳定
- Drifting 视频前期跟正常轨迹相似，但某个点之后突然 shift 并保持 unstable
- **关键**：drift 几乎从不在生成开始时发生

**Helios 解法**：**First-Frame Anchor**
- 训练和推理时都始终保留第一帧在 $X_{Hist}$ 中
- 第一帧作为 **global visual anchor**，约束后续段的 distribution shift

直觉：第一帧是"原始的、clean 的、与 prompt 对齐"的 visual reference。它充当一个 distributional anchor，把后续生成的 color statistics 拉回原始分布。这类似于 conditional generation 里的 reference image，但用法是"全局颜色锚"。

Ablation 显示移除 First-Frame Anchor 在 frame 720 就出现明显 color drift，且 subject identity 也会逐渐偏离第一帧确立的 identity。

### 2.3 Restoration Shift 与 Frame-Aware Corrupt

**问题**：模型训练时 history 是 clean video，推理时 history 是模型自己生成的 imperfect output。小 error 逐 section 累积，最终放大成 blur、noise 等 restoration artifact。

这是经典的 **exposure bias** / **train-inference gap** 问题，在 RNN 时代叫 scheduled sampling，在 GAN 时代叫 professor forcing。Reference: Professor Forcing https://arxiv.org/abs/1610.00486

**Helios 解法**：**Frame-Aware Corrupt**
- 对每个历史帧**独立**采样一种扰动（关键：独立，不是全局一致）
- 四种扰动：
  - 概率 $p_c$：调整 exposure，幅度 $\sim \mathcal{U}[a_{min}, a_{max}]$
  - 概率 $p_a$：加 noise，水平 $\sim \mathcal{U}[b_{min}, b_{max}]$
  - 概率 $p_b$：下采样再上采样，因子 $\sim \mathcal{U}[c_{min}, c_{max}]$
  - 概率 $p_d$：保持 clean
  - $p_a + p_b + p_c + p_d = 1$
- $T_{Hist}$ 帧 → $T_{Hist}$ 个独立的 corruption decision

实际训练参数（Table 1）：
- Stage 1-2: $p_a=0.0, p_b=0.8, p_c=0.1, p_d=0.1$；$b_{min}=0, b_{max}=0.33$；$c_{min}=0, c_{max}=0.1$
- Stage 3: $p_a=0.4, p_b=0.4, p_c=0.0, p_d=0.2$；$a_{min}=0.3, a_{max}=1.7$；$b_{min}=0, b_{max}=0.33$

注意 stage 1-2 主用 blur（下采样上采样），stage 3 加上 exposure adjustment——这种 stage-specific 配置是因为 distillation 阶段对 history robustness 要求更高。

**为什么 per-frame independent 很重要**：如果整段 history 用同一种 corruption，模型会学到"全局一致 distortion"的 spurious pattern；per-frame independent 模拟了真实 inference 时"不同 frame 有不同 error pattern"的分布。

Ablation 显示移除 Frame-Aware Corrupt 在 240 帧就 severe drift，minute-scale 完全不可用。

### 2.4 三种 anti-drift 的协同

- Relative RoPE 解决位置外推（position drift）
- First-Frame Anchor 解决颜色分布漂移（color drift）
- Frame-Aware Corrupt 解决累积误差（restoration drift）

这三者组合起来，让模型不需要 Self-Forcing 那种 train-as-infer rollout 就能 robust 到分钟级。这是 Helios 最核心的 contribution——**用 data augmentation 的思路解决 train-inference gap，而不用 expensive rollout**。

---

## 3. Deep Compression Flow - Token View：如何把 14B 的 token cost 压到 1.3B 水平

### 3.1 Multi-Term Memory Patchification：分层记忆压缩

**核心观察**：在 autoregressive video generation 中：
- 预测未来帧主要依赖**近期 history**（local motion, short-range continuity）
- **远期 history** 主要贡献粗粒度全局 context

这跟人脑记忆系统很像——短期记忆精细，长期记忆抽象。Helios 把 $X_{Hist}$ 分成 short/mid/long 三段：

- $T_1, T_2, T_3$ 帧分别属于 short/mid/long，$0 < T_1 < T_2 < T_3$
- 每段用独立 Conv kernel $(p_t^{(i)}, p_h^{(i)}, p_w^{(i)})$，$i \in \{1, 2, 3\}$
- 压缩比随时间距离增加：$p_t^{(1)} < p_t^{(2)} < p_t^{(3)}$，空间维度同理

Patchification 后 token 数：

$$
L_{short} = \frac{T_1 H W}{p_t^{(1)} p_h^{(1)} p_w^{(1)}}, \quad L_{mid} = \frac{T_2 H W}{p_t^{(2)} p_h^{(2)} p_w^{(2)}}, \quad L_{long} = \frac{T_3 H W}{p_t^{(3)} p_h^{(3)} p_w^{(3)}}
$$

总 token 数：

$$
L_{total} = HW \left( \frac{T_1}{p_t^{(1)} p_h^{(1)} p_w^{(1)}} + \frac{T_2}{p_t^{(2)} p_h^{(2)} p_w^{(2)}} + \frac{T_3}{p_t^{(3)} p_h^{(3)} p_w^{(3)}} \right)
$$

**关键性质**：$L_{total}$ 与目标视频长度无关！只要 $T_1 + T_2 + T_3$ 配上对应压缩比，总 token budget 固定。这意味着模型可以在固定 token budget 下保留任意长的 history。

实际配置（Section 4.1）：
- $(p_t^{(1)}, p_t^{(2)}, p_t^{(3)}) = (4, 2, 1)$ —— short 用 4× 时间压缩，long 用 1×
- $(p_h^{(1)}, p_h^{(2)}, p_h^{(3)}) = (8, 4, 2)$
- $(p_w^{(1)}, p_w^{(2)}, p_w^{(3)}) = (8, 4, 2)$
- $(T_1, T_2, T_3) = (16, 2, 2)$

算下来 historical-context token 数从 $5HW$ 压到 $\frac{5}{8}HW$（约 8× 压缩）。

直觉：近期 16 帧用 4×8×8=256× 压缩（保留运动细节），远期 2 帧用 1×2×2=4× 压缩（保留全局 context）。这是 token budget 的 "unequal allocation"——给近期多预算，远期少预算。

训练时随机 zero out 一部分 historical context 模拟 T2V/I2V/V2V 推理场景。

### 3.2 Pyramid Unified Predictor Corrector：多尺度 flow matching

**核心观察**：diffusion sampling 的早期 step 主要决定 global structure（layout、color），后期 step 主要 refine fine-grained details（edges、textures）。

**Helios 方案**：coarse-to-fine schedule，早期在低分辨率 latent 空间采样，逐渐过渡到全分辨率。

**训练阶段**：
- 把生成过程分成 K 个 stage，stage k 在分辨率 $(h^k, w^k)$ 操作
- 构造从 scale k-1 到 scale k 的 linear interpolation path：

$$
x_t^k = (1 - \lambda_t) x^k + \lambda_t \text{Up}(x^{k-1})
$$

- $k \in \{1, 2, ..., K\}$：stage index
- $\lambda_t \in [0, 1]$：noise level 控制参数，与 timestep 关联
- $\text{Up}(\cdot)$：上采样操作
- $x^k$：scale k 的 clean target
- $x^{k-1}$：scale k-1 的 clean target

**边界条件**：
- $k=1$：$\text{Up}(x^{k-1}) = \epsilon \sim \mathcal{N}(0, I)$（从纯噪声开始）
- $k=K$：$x^k = x_0$（全分辨率 clean sample）

**Timestep partition**：$T \in [0, 1000]$ 分成 stage 边界 $T_0 = 1000 > T_1 > \cdots > T_K = 0$，stage k 只在 $[T_k, T_{k-1}]$ 操作。

**Ground-truth velocity**（沿 linear path constant）：

$$
v^k = x^k - \text{Up}(x^{k-1})
$$

**Velocity-matching objective**：

$$
\mathcal{L} = \mathbb{E}_{k, \lambda_t, x_t^k, \text{Up}(x^{k-1}), y} \left[ \| u_\theta^k(x_t^k, y, \lambda_t, k) - v^k \|_2^2 \right]
$$

- $u_\theta^k(\cdot)$：参数化 velocity field（第 k stage 的网络预测）
- $y$：conditioning input（text prompt 等）
- 网络输入 $(x_t^k, y, \lambda_t, k)$：当前 noisy state + condition + noise level + stage index
- 实践 $K=3$

**关键**：所有 stage 共享同一个 $\lambda_t$ schedule，保持 flow matching 在不同 scale 间的一致性。stage index $k$ 作为额外 condition 喂给网络，让单个网络处理多尺度。

**推理阶段**：
- 分 K 个 stage，分配 $(N_1, N_2, ..., N_K)$ 步，总步数 $N = \sum_{k=1}^K N_k$
- stage k 在离散时间步 $\{t_k^n\}_{n=0}^{N_k}$ 上 update：

$$
x_{t_k^n}^k = x_{t_k^{n-1}}^k + u_\theta^k(x_{t_k^{n-1}}^k, y, t_k^{n-1}) (t_k^n - t_k^{n-1})
$$

- 这是 Euler method 的标准 ODE 积分
- stage 转换时用 nearest-neighbor 上采样 terminal state，然后 correct 注入噪声和 covariance 维持分布一致性（参考 PyramidFlow https://arxiv.org/abs/2410.05954）
- UniPC 的 state buffer 在 stage 转换时 reset（因为 prediction tensor shape 跨 stage 变化，cached prediction 不能跨 stage 复用）

**计算复杂度对比**：
- 单尺度 N 步：$\mathcal{O}(HW \cdot N)$
- 标准金字塔（每 stage 分辨率减半）：

$$
\left(HW + \frac{H}{2}\frac{W}{2} + \frac{H}{4}\frac{W}{4} + \cdots + \frac{H}{2^{K-1}}\frac{W}{2^{K-1}}\right) \times \frac{N}{K}
$$

这是个等比级数，求和约 $\frac{4}{3} HW \cdot \frac{N}{K}$，相对单尺度的 $\mathcal{O}(HW \cdot N)$ 是 $\frac{4}{3K}$ 倍。

实际 $K=3$ 把 noisy-context token 数从 $NHW$ 降到 $\frac{7}{16}NHW$（约 2.29× 压缩）。

### 3.3 两个压缩的乘积效应

Historical context 8× 压缩 + Noisy context 2.29× 压缩，在 attention FLOPs 上分别转化为 64× 和 5.2× 减少（因为 attention 是 $\ell^2$ 复杂度）。这让 14B 模型的 effective compute 降到接近 1.3B 模型的水平。

---

## 4. Deep Compression Flow - Step View：Adversarial Hierarchical Distillation

### 4.1 DMD 背景回顾

Distribution Matching Distillation (DMD) 是把 multi-step teacher 蒸馏成 few-step student 的主流方法。Reference: https://arxiv.org/abs/2311.18828

DMD pipeline：
1. 采样 noise $\epsilon$ 喂给 few-step generator $G_\theta$
2. 用 $x_0$ prediction + backward simulation 得到 clean sample $x_0$
3. 采样 noise level $\lambda_\tau \sim \mathcal{U}[0, 1]$，扰动 $x_0$ 得 $x_\tau$
4. 用 real-score estimator $p_{real}$ 和 fake-score estimator $p_{fake}$ 评估：
   - $s_{real}$ 用 CFG：$\text{CFG}(s_{real}^{cond}, s_{real}^{uncond})$
   - $s_{fake}$ 只用 conditional：$s_{fake}^{cond}$
5. $s_{real} - s_{fake}$ 定义 distribution-matching gradient 更新 $G_\theta$
6. $p_{fake}$ 用 flow-matching loss $\mathcal{L}_{Flow}$ 训练

### 4.2 Helios 的四个改进

**改进 1：Pure Teacher Forcing with Autoregressive Teacher**

现有方法（Self-Forcing++、LongLive、Rolling Forcing、Reward Forcing）训练时做长 rollout（几十秒到几分钟视频），计算开销巨大，限制模型到 1.3B。

Self-Forcing 的 robustness 强依赖训练时的 rollout 长度：训练只 rollout 5 sections，推理超 5 秒就 drift。这是经典 exposure bias 的另一种体现。

Helios 方案：
- Distillation 阶段**只用 real data** 作为 historical context
- 每步只生成**单个 section**（不 rollout）
- 加上 Easy Anti-Drifting（Section 2）达到长 rollout 的 anti-drift 效果
- 用 **Helios-Base** 作为 teacher（已能生成高质量长视频），而非 Wan（只能短视频）

这是关键 trade-off：用 data augmentation 的 anti-drift 替代 expensive rollout 的 anti-drift，训练开销大幅降低，但 robustness 相当（Ablation Figure 18 证明）。

**改进 2：Staged Backward Simulation**

DMD 在单条 flow trajectory 做 backward simulation 恢复 $x_0$。Helios 把它分解成 K 个 stage：

给定 stage k 的当前 state $x_t^k$ 和预测 velocity field $u_\theta^k(x_t^k, y, \lambda_t, k)$，估计 terminal state：

$$
x_0^k = x_t^k - \lambda_t \cdot u_\theta^k(x_t^k, y, \lambda_t, k)
$$

- $\lambda_t$ 是当前 noise level
- 公式直接从公式 5 的 linear interpolation path 推出（把 $\lambda_t$ 对应的 noisy state 减去 noise 贡献）

然后用公式 5 重建 $x_t^k$，再用公式 10 重新估计 $x_0^k$，重复直到 stage k 收敛。$x_0^k$ 初始化 stage (k+1)。K stage 后 $x_0 = x_0^K$。

Ablation 显示如果直接把 multi-scale $\{x_0^k\}$ 全部喂给 fake-score estimator 会导致不稳定训练（Figure 18）——只在最后用 $x_0^K$ 反而稳定。这是个反直觉但重要的发现。

**改进 3：Coarse-to-Fine Learning**

Helios 通过 K 个 stage + 多个 flow trajectory 传播梯度，优化难度增加。三个 curriculum 策略：

1. **Staged ODE Init**：用 Helios-Mid 生成 ODE solution pairs 构造紧凑数据集用于初始化，跨 K stage 进行。每个 stage 只需生成单 section 而非多 section，autoregressive teacher 引导。

2. **Dynamic Re-noise**：从 Beta 分布采样 timestep，参数按 cosine decay schedule：
   - 早期集中在高 noise timestep（学 coarse structure）
   - 后期更均匀（强调中低 noise timestep 学细节）
   
   直觉：diffusion 不同 noise level 的学习难度不同，curriculum 式采样让模型先学容易的（global structure）再学难的（detail）。

**改进 4：Adversarial Post-Training**

纯 DMD distillation 让 student 继承 teacher 偏差，受 teacher 表达能力上限约束。Helios 加 GAN objective 提供 teacher-independent 监督，让 student 能"超越" teacher。

具体：在 $p_{fake}$ 的 DiT layers 加 multi-granularity classification branches $D$（GAN head），分布在 layers [5, 15, 25, 35, 39]，dim 768。

Non-saturated GAN objective：

$$
\mathcal{L}_D = \mathbb{E}[\log D(x_\tau^{real}, \tau)] + \mathbb{E}[-\log D(x_\tau^K, \tau)]
$$

- $D(\cdot, \tau)$：discriminator 在 noise level $\tau$ 上判断 real/fake
- $x_\tau^{real}$：real data 加 noise 到 level $\tau$
- $x_\tau^K$：generator 输出 $x_0^K$ 加 noise 到 level $\tau$

近似 R1 regularizer（参考 APT https://arxiv.org/abs/2501.08316）：

$$
\mathcal{L}_{aR1} = |D(x_\tau^{real}, \tau) - D(\mathcal{N}(x_\tau^{real}, \sigma_D I), \tau)|_2^2
$$

- $\mathcal{N}(x_\tau^{real}, \sigma_D I)$：对 real sample 加 Gaussian 扰动
- $\sigma_D = 0.1$
- R1 正则化防止 discriminator 过拟合 training data

完整对抗目标：

$$
\mathcal{L}_D = \mathbb{E}[\log D(x_\tau^{real}, \tau)] + \mathbb{E}[-\log D(x_\tau^K, \tau)] + \lambda_D \cdot \mathbb{E}[|D(x_\tau^{real}, \tau) - D(\mathcal{N}(x_\tau^{real}, \sigma_D I), \tau)|_2^2]
$$

$$
\mathcal{L}_G = \mathbb{E}[\log D(x_\tau^K, \tau)]
$$

- $\lambda_D = 100$
- 为节省显存，discriminator 输入用 $H' \times W'$ random crop（$H' = H/2, W' = W/2$）

**最终目标**：

$$
\mathcal{L}_{G_\theta} = \mathcal{L}_{DMD} + w_G \cdot \mathcal{L}_G
$$

$$
\mathcal{L}_{p_{fake}} = \mathcal{L}_{Flow} + w_D \cdot \mathcal{L}_D
$$

- $w_G = 5e-2$，$w_D = 1e-2$
- TTUR：每 5 次 $p_{fake}$ 更新对应 1 次 $G_\theta$ 更新

Ablation 显示移除 Adversarial Post-Training 在 naturalness 和 realism 上明显下降。

---

## 5. 推理时技巧：Training-free Anti-Drift 与 Interactive Editing

### 5.1 Adaptive Sampling

**观察**：drift 伴随 RGB/latent 统计的明显 shift（mean 和 variance）。

**机制**：维护 latent section 的 RGB mean $\mu_t$ 和 variance $\sigma_t^2$ 的全局 EMA 统计：

$$
\bar{\mu}_t = \rho_\mu \bar{\mu}_{t-1} + (1 - \rho_\mu) \mu_t
$$

$$
\bar{\sigma}_t^2 = \rho_\sigma \bar{\sigma}_{t-1}^2 + (1 - \rho_\sigma) \sigma_t^2
$$

- $\rho_\mu, \rho_\sigma \in (0, 1)$：smoothing 系数
- $\bar{\mu}_t, \bar{\sigma}_t^2$：到时间 t 的全局统计

**Drift detection**：

$$
\|\mu_t - \bar{\mu}_t\|_2 > \delta_\mu \quad \text{and} \quad \|\sigma_t^2 - \bar{\sigma}_t^2\|_2 > \delta_\sigma
$$

- $\delta_\mu, \delta_\sigma$：预设阈值

当检测到 drift，下个 section 生成时对 history 应用 Frame-Aware Corrupt。这隐式降低模型对 biased history 的依赖，鼓励使用其内在生成先验。

直觉：这是一个 runtime anomaly detection + disturbance rejection 机制。当历史统计偏离"正常"分布时，主动"扰动"历史以打破模型对 biased history 的过度依赖。类似于 control theory 里的 feedback correction。

### 5.2 Interactive Interpolation

长视频生成支持交互式编辑（用户随时改 prompt）。naive 切换 prompt embedding 会引起瞬时 conditional shift 和视觉 discontinuity。

Helios 用 prompt interpolation：

$$
e^{[j]} = (1 - \lambda_j) e^{(1)} + \lambda_j e^{(2)}, \quad \lambda_j = \frac{j}{M-1}, \quad j = 0, 1, ..., M-1
$$

- $e^{(1)}, e^{(2)} \in \mathbb{R}^{\ell_{Text} \times D}$：当前和目标 prompt embedding
- $\ell_{Text}$：text length
- $D$：hidden dimension
- $M$：插值步数
- $\lambda_j \in [0, 1]$：线性插值系数
- $e^{[0]} = e^{(1)}$，$e^{[M-1]} = e^{(2)}$

生成时按顺序喂这些 embedding，从 $e^{(1)}$ 渐变到 $e^{(2)}$。这是 Krea 的方法在 Helios 上的直接应用，让 world model 式的 interactive generation 成为可能（Figure 23 例子：prompt 切换导致主体渐变——cat→wildcat→fox→wolf→antelope）。

---

## 6. Infrastructure：让 14B 训练落到单 GPU

### 6.1 Workload Analysis

标准 DiT 每层复杂度：

$$
\mathcal{O}(\alpha B \ell D^2 + \beta B \ell^2 D)
$$

- $\alpha$：linear layer 成本系数
- $\beta$：attention 成本系数
- $B$：batch size
- $\ell$：sequence length（token 数）
- $D$：hidden dimension
- $\ell^2$ 项主导 self-attention

整体复杂度 $\mathcal{O}(\hat{L}(\alpha B \ell D^2 + \beta B \ell^2 \hat{D}))$，activation memory $\mathcal{O}(\gamma L B \ell D)$（$\gamma$ 取决于 implementation，$L$ 是层数，$\hat{L}$ 是某种 normalized 层数）。

8× 和 2.29× 的 token 压缩转化为：
- Historical context attention FLOPs：$8^2 = 64\times$ 减少
- Noisy context attention FLOPs：$2.29^2 \approx 5.2\times$ 减少
- Activation memory 线性减少

### 6.2 三阶段训练的显存策略

**Stage 1-2**：只需 VAE + text encoder + DiT。把 VAE latents 和 text embeddings offload 到磁盘，GPU 上只剩 DiT，单 GPU batch size 可比 image diffusion model。

**Stage 3**：需要 4 个 14B 模型（few-step generator、real-score estimator、fake-score estimator、EMA model）+ GAN heads。80GB 显存预算下，naive 装不下。

**策略 1：Sharded EMA**
- 用 ZeRO-3 把 FP32 EMA 参数 shard 到 Z 个 GPU
- 14B 模型在 Z GPU 上每设备存 $\frac{14 \times 4}{Z}$ GiB EMA 参数
- 消除冗余 replica

**策略 2：Asynchronous VRAM Freeing**
- 序列执行多个大模型：noise → few-step staged generator → $x_0^{staged}$ → re-noise → real-score & fake-score estimator → 计算 $\mathcal{L}_{DMD}, \mathcal{L}_{GAN}, \mathcal{L}_{Flow}$
- TTUR 下每次迭代只更新一个模型
- 异步 offload 未用模型到 host memory（pinned memory + non-blocking transfer + CPU-GPU scheduling）
- Peak VRAM 限制在训练单 14B 模型的水平

**策略 3：Cache Grad for GAN**

(a) 更新 generator：标准 autodiff 要保留 $G_\theta$ 和 $p_{fake}$ 所有 activations 直到 backprop 完成，显存不够。
- 解决：forward pass 时立即 cache discriminator 对输入的梯度
- 立即释放 estimator 的中间 activations
- Backprop 时用 cached input gradient，不保留完整 computation graph
- Peak memory 降到单 14B 模型水平

(b) 更新 fake-score estimator：
- Gradient accumulation + batched execution
- 先单独 forward/backward 计算 $\mathcal{L}_{Flow}$，累积梯度，立即释放 activations
- 再合并 real/fake/perturbed samples 一次 forward/backward 算剩余 loss

### 6.3 Flash Normalization 与 Flash RoPE

**Flash Normalization**（Triton kernel fusion for LayerNorm/RMSNorm）：
- 把 mean/variance 计算、归一化、affine 变换合并到单 kernel
- 用 `tl.math.rsqrt` 等 optimized primitives
- 只 cache scalar 统计：row-wise $inv\_var \in \mathbb{R}^{B \times \ell}$ 和 $\mu \in \mathbb{R}^{B \times \ell}$
- 不存完整归一化 tensor $\mathbf{z} \in \mathbb{R}^{B \times \ell \times D}$
- 内存从 $\mathcal{O}(B\ell D)$ 降到 $\mathcal{O}(B\ell)$
- 内部 FP32 计算保证数值稳定，输入输出保留 bfloat16
- Row-wise parallelism（一个 program instance per token）+ coalesced memory access

**Flash RoPE**（Triton kernel fusion for RoPE）：
- 输入 $\mathbf{x} \in \mathbb{R}^{B \times \ell \times H \times D}$ 展平到 $\mathbb{R}^{(B \cdot \ell \cdot H) \times D}$
- $H$ 是 attention head count，$D$ 是 head dimension
- 一个 program instance per attention head
- Interleaved memory access 直接取 real/imaginary component
- Pre-compute $\cos, \sin$，应用旋转：
  - $\text{out}_{real} = x_{real} \cdot \cos - x_{imag} \cdot \sin$
  - $\text{out}_{imag} = x_{real} \cdot \sin + x_{imag} \cdot \cos$
- Backward 复用 forward kernel，把 $\sin$ 取反（$\sin_{neg} = -\sin$）做逆旋转
- 不存完整 intermediate tensor，只需 $\cos, \sin \in \mathbb{R}^{B \times \ell \times (D/2)}$
- 内存从 $\mathcal{O}(B\ell H D)$ 降到 $\mathcal{O}(B\ell D)$

Ablation Table 6：Wan-2.1-T2V-14B 50 步训练时间从 398.03s 降到 340.38s（约 15% 加速），inference 从 98.68s 降到 84.41s。

### 6.4 与 FlashAttention 的协同

因为 Helios 不用 causal masking，可以无缝集成 FlashAttention（bidirectional attention 优化 backend）。这是 Helios 拒绝 causal masking 路径的另一个 bonus——保留了使用最高效 attention kernel 的可能。Reference: https://arxiv.org/abs/2205.14135

---

## 7. 实验：HeliosBench 与定量结果

### 7.1 HeliosBench 设计

- 240 个 LLM-refined prompts from Self-Forcing
- 4 个 duration tier：very short (81 frames), short (240 frames), medium (720 frames), long (1440 frames)
- 5 个 metric：Aesthetic (LAION aesthetic predictor), Dynamic (Farnebäck), Motion Smoothness (RAFT), Semantic (ViCLIP), Naturalness (OpenS2V-Eval)
- 额外 drifting metric：Aesthetic, Motion Smoothness, Semantic, Naturalness 的时间衰减
- 每个指标映射到 10 分制（用 empirical 阈值 $\mathcal{T}_k = [\tau_0, ..., \tau_8]$）
- 短视频 weight：Semantic 0.35, Naturalness 0.35, 其他各 0.10
- 长视频 weight：drifting metrics 各 0.099，Semantic/Naturalness 0.255，其他 0.03

这个 weight 设计反映了一个直觉：短视频主要看 quality，长视频主要看 stability over time。

### 7.2 短视频结果（Table 3）

Helios-Distilled 14B：
- Throughput: 19.53 FPS
- Total: 6.00
- Aesthetic 8, Dynamic 7, Smoothness 10, Semantic 5, Naturalness 5

对比关键 baseline：
- CausVid 1.3B: 24.41 FPS, Total 4.50（更快但 quality 差很多）
- Self-Forcing 1.3B: 21.20 FPS, Total 5.75
- Krea 14B: 6.74 FPS, Total 5.95（同 size 但慢 3 倍）
- Wan 2.1 14B: 0.33 FPS, Total 6.15（quality 相当但慢 60 倍）
- LongCat-Video 13.6B: 0.33 FPS, Total 6.30

**Helios 比 Wan 14B 快 52 倍**，比 FastVideo/TurboDiffusion 14B 快 2-3 倍，且 quality 与同 size base model 相当。

### 7.3 长视频结果（Table 4）

Helios-Distilled：
- Throughput: 19.53 FPS
- Total: 6.94（含 Throughput Score 6）, Total* 6.34
- Drifting 各维度均优于或匹配 baseline

对比 Reward Forcing 1.3B（22.13 FPS, Total 6.88）：Helios Total 更高且 drifting 更低。

### 7.4 Ablation 关键发现

- **Guidance Attention + causal mask**：unstable training
- **移除 Guidance Attention**：semantic accumulation over time（如 bird crest 越变越大）
- **移除 First-Frame Anchor**：frame 720 就 color drift + identity drift
- **移除 Frame-Aware Corrupt**：240 帧就 severe drift
- **Multi-Term Memory Patchification**：解决 naive history context scalability（context length 6 时 OOM，Helios 可到 18）
- **Pyramid Unified Predictor Corrector**：吞吐量近翻倍，少量 quality 下降
- **Pure Teacher Forcing vs Self-Forcing**：anti-drift robustness 相当
- **Autoregressive Teacher vs Bidirectional Teacher**：前者更好
- **Multi-scale $x_0^k$ 喂 fake-score estimator**：unstable training
- **移除 Coarse-to-Fine Learning**：模型无法收敛，第一 section 质量不可接受
- **移除 Adversarial Post-Training**：naturalness/realism 下降
- **Decouple DMD**：收敛慢，grayish tone，grid-like artifact
- **Reinforcement Post-Training（Reward-weighted Regression）**：semantic/aesthetic 下降，severe flickering，所以排除 RL

---

## 8. 直觉与联想

### 8.1 拒绝 causal masking 的深层意义

CausVid / Self-Forcing 路线把 bidirectional diffusion model 改成 causal autoregressive，本质上是把 diffusion 拉回 GPT 范式。这有两个 hidden cost：
1. **表达力损失**：bidirectional attention 是 diffusion model 的核心优势（全局信息流通），causal mask 切断了 cross-section 的双向交互，每个 section 倾向独立生成新场景
2. **工程限制**：causal attention 跟 FlashAttention 等 high-efficiency backend 不兼容

Helios 用 Guidance Attention 实现 asymmetric attention pattern（history → noisy 是单向的，但 noisy 内部是 bidirectional），保留了 bidirectional 的表达力 + 维持 autoregressive 的因果性。这是个**架构层面**的智慧——避免在 paradigm 上做不必要的妥协。

### 8.2 Token budget 视角的记忆系统

Multi-Term Memory Patchification 让我想到人脑的多层记忆系统：
- Short-term memory（海马体）：精细、容量小、快速衰减
- Long-term memory（皮层）：抽象、容量大、稳定

Helios 用 hierarchical compression 把这个思想落到 token budget 分配上。远期历史用大 kernel 大压缩（粗粒度），近期历史用小 kernel（细粒度）——这正是"近期保留细节、远期保留语义"的对应。

更深的联想：这与 Retrieval-Augmented Generation (RAG) 里的 chunk size 选择、long-context LLM 里的 sliding window attention 有相似的 design philosophy——**对 context 的不同部分用不同的粒度处理**。Radial Attention 等工作也在探索类似的 energy decay sparse attention pattern。

### 8.3 Anti-drift 的"模拟 vs 滚动"哲学

Self-Forcing 范式用 train-as-infer rollout 缩小 train-inference gap，思路是"让训练时见到推理时的不完美 history"。但这带来两个问题：
1. Robustness 强依赖 rollout 长度，超出训练 rollout 就 drift
2. 长 rollout 计算开销巨大，限制模型 scale

Helios 的 Frame-Aware Corrupt 是个 elegant alternative：**与其在训练时滚动生成 imperfect history，不如直接用 data augmentation 模拟 imperfect history 的分布**。这避开了 expensive rollout，让 14B 模型训练成为可能。

这让我想到 scheduled sampling（Professor Forcing）vs teacher forcing 的经典争论。在 sequence-to-sequence 任务上，scheduled sampling 是 data augmentation 的方式，而 professor forcing 是 adversarial 的方式。Helios 选择的是 data augmentation 路线，简单但 effective。

### 8.4 Flow matching 的多尺度分解

Pyramid Unified Predictor Corrector 把单条 flow trajectory 分解成多条 multi-scale trajectory。这有两个 efficiency gain：
1. 早期 stage 在低分辨率跑，token 少
2. 后期 stage 在高分辨率跑，refine 细节

直觉：diffusion sampling 的早期 step 决定 global structure，低分辨率足够；后期 step 决定 detail，需要高分辨率。这是 PyramidFlow 的核心思想在 Helios 上的应用 + UniPC 的 predictor-corrector 融合。

更深的联想：这跟 image generation 里的 "coarse-to-fine" progressive generation（如 Progressive GAN、StyleGAN 的 progressive growing）有相通的哲学，但实现上是 flow matching 的多尺度分解，而非 GAN 的 progressive growing。

### 8.5 DMD + GAN 的 hybrid distillation

纯 DMD 受 teacher 上限约束——student 最多和 teacher 一样好。加 GAN objective 提供 teacher-independent 的 real data 监督，student 能"超越" teacher。这跟 DMD2、Spark-Wan 的思路一脉相承。

更深的联想：这是"模仿学习 + 强化学习"的 hybrid 在 diffusion distillation 上的体现。DMD 是 imitation（mimic teacher distribution），GAN 是 RL（optimize against real data reward）。两者结合打破 teacher 上限。

### 8.6 RoPE 周期性 + multi-head attention 的相互作用

Absolute RoPE 在长序列上的周期性问题，在 long-context LLM 里已经被广泛讨论（LongRoPE、YaRN、NTK-aware scaling）。Helios 用 Relative RoPE 通过固定时间索引范围规避了这个问题——模型永远只在"训练见过的相对位置"内工作。

这跟 ALiBi vs RoPE 的争论呼应：相对位置编码对长序列外推更友好。Helios 在 video generation 上验证了这个直觉。

### 8.7 Drift detection + disturbance rejection 的控制论视角

Adaptive Sampling 通过 EMA 监控 RGB/latent 统计的偏离来检测 drift，再对后续 history 主动加扰动。这是个 runtime anomaly detection + disturbance rejection 机制，跟 control theory 里的 feedback correction 相似。

更深的联想：这跟 outlier detection + robust control 在 cyber-physical system 里的应用有共通的 design pattern——**监控关键统计量、检测异常、主动扰动打破 bias**。

### 8.8 4 个 14B 模型的显存工程

Stage 3 训练需要 4 个 14B 模型（generator、real-score、fake-score、EMA），80GB 显存挑战极大。Helios 的三个策略（Sharded EMA、Async VRAM Freeing、Cache Grad for GAN）组合下来把 peak VRAM 压到单 14B 训练水平。

这让我想到 LLM training 里的 ZeRO、FSDP、pipeline parallelism 等 memory optimization 技术，但 Helios 用了更激进的 **model-level scheduling**——利用 TTUR 的不对称更新频率，把不需要的模型 offload 到 host memory。这是 distributed training 思路在 single-GPU 上的极致运用。

### 8.9 Architecture 与 inference regime 的耦合

Helios 反复强调"避免改变 inference regime of bidirectional pre-trained models"。这是个深层 insight：**pre-trained model 的 inference regime 是其表达力的重要部分，改变它会引入难以弥补的 quality loss**。

Causal masking 改变了 bidirectional model 的 inference regime，限制了 quality ceiling。Helios 通过 Guidance Attention 实现"看起来 autoregressive 但实际 bidirectional"的 architecture，保留了 pre-trained model 的全部表达力。

这让我想到 LLM 里 decoder-only vs encoder-decoder 的争论——decoder-only 强行用 causal mask 统一了 architecture，但 encoder-decoder 在某些任务上可能更优。Helios 的 design 提示：**统一不一定要通过 masking 实现，可以通过 asymmetric attention pattern 实现**。

### 8.10 与 world model 的关联

Helios 在 Introduction 提到 "world model" 应用。real-time long video generation 是 world model 的关键 capability——支持 interactive generation、game engine、embodied AI。

Helios 的 Interactive Interpolation 让用户能动态修改 prompt，生成过程平滑过渡。这本质上是 **conditional generation with time-varying condition**，是 world model 式的 interactive control 的 primitive。

更深的联想：Sora、Genie、Worldplay 等 world model 工作都在探索类似的 interactive long video generation。Helios 通过 14B + 19.5 FPS 把这个 capability 推到了实用的 compute budget 内。

---

## 9. 局限与未来方向

Helios 自己列了三个局限：
1. 现有 metric 不够准确，需要 perceptually aligned metric
2. Section stitching boundary 仍有 flickering artifact（所有 autoregressive 方法共通问题，RL 优化 smoothness 可能是未来方向）
3. 实验限于 $384 \times 640$ 分辨率，更高分辨率未探索

我额外想到的几个未来方向：
- **Latent re-encode** 在 Appendix B 提到，把 section-by-section latent decode 再 re-encode 成连续 latent sequence 以消除 multi-first-frame distribution artifact——这个 trick 可能可以推广到其他 autoregressive video model
- **3D attention 的进一步优化**：当前 Helios 不用 sparse/linear attention，未来结合 Radial Attention 或 Sparse VideoGen 可能进一步加速
- **更高分辨率训练**：需要解决 3D attention 在高分辨率下的 quadratic 复杂度
- **Audio-video joint generation**：LTX-2 等工作已经在探索，Helios 框架可能可以扩展
- **更长 horizon**：当前 minute-scale，能否 hour-scale？这需要更激进的 history compression

---

## 10. 总结：Helios 的核心 contribution 与设计哲学

Helios 不是单点突破，而是**一系列 coordinated design choice** 让 14B video diffusion model 达到实时分钟级生成。核心 contribution：

1. **拒绝 causal masking**：用 Guidance Attention 保留 bidirectional inference regime，维持 pre-trained model 的 quality ceiling
2. **用 data augmentation 替代 expensive rollout**：Frame-Aware Corrupt + First-Frame Anchor + Relative RoPE 组合实现 anti-drift，让训练开销可控
3. **Token-level 双重压缩**：Multi-Term Memory Patchification（history）+ Pyramid Unified Predictor Corrector（noisy），把 14B effective compute 压到 1.3B 水平
4. **Step-level 蒸馏**：Adversarial Hierarchical Distillation 把 50 步降到 3 步，加 GAN 突破 teacher 上限
5. **Infrastructure 极致优化**：让 4 个 14B 模型在 80GB 显存内训练

**设计哲学**：Helios 拒绝了"用 infra 堆 performance"的常规路径（KV-cache、sparse attention、quantization、pipeline parallelism），转而用 **algorithmic compression + data augmentation** 解决根本矛盾。这让 14B 实时长视频生成成为可能，且 quality 不掉。

这对社区的意义在于：**14B 模型 + 实时 + 长视频 三个目标不再 mutually exclusive**。未来的 video generation 研究可以在这个 recipe 上继续推进——更高分辨率、更长 horizon、更复杂 motion——而不用回到 1.3B 小模型的 quality 瓶颈。

---

## Reference Links

- **Helios Project Page**: https://pku-yuangroup.github.io/Helios-Page
- **Wan 2.1 (base model)**: https://arxiv.org/abs/2503.20314
- **Self-Forcing**: https://arxiv.org/abs/2506.08009
- **CausVid**: https://arxiv.org/abs/2504.14032 (CVPR 2025)
- **DMD (original)**: https://arxiv.org/abs/2311.18828
- **DMD2**: https://arxiv.org/abs/2405.14867
- **PyramidFlow**: https://arxiv.org/abs/2410.05954
- **Diffusion Forcing**: https://arxiv.org/abs/2407.01392
- **FlashAttention**: https://arxiv.org/abs/2205.14135
- **RoPE (RoFormer)**: https://arxiv.org/abs/2104.09864
- **UniPC**: https://arxiv.org/abs/2302.04867
- **FramePack**: https://arxiv.org/abs/2504.12626
- **APT (Adversarial Post-Training)**: https://arxiv.org/abs/2501.08316
- **Krea RealTime 14B**: https://github.com/krea-ai/realtime-video
- **LongLive**: https://arxiv.org/abs/2509.22622
- **Rolling Forcing**: https://arxiv.org/abs/2509.25161
- **Reward Forcing**: https://arxiv.org/abs/2512.04678
- **CogVideoX**: https://arxiv.org/abs/2408.06072
- **HunyuanVideo**: https://arxiv.org/abs/2412.03603
- **LTX-Video**: https://arxiv.org/abs/2501.00103
- **Mochi**: https://github.com/genmoai/models
- **LongCat-Video**: https://arxiv.org/abs/2510.22200
- **SANA Video**: https://arxiv.org/abs/2509.24695
- **Spark-Wan**: https://github.com/PKU-YuanGroup/Spark-Wan
- **Open-Sora Plan**: https://arxiv.org/abs/2412.00131
- **Self-Forcing++**: https://arxiv.org/abs/2510.02283
- **Causal Forcing**: https://arxiv.org/abs/2602.02214
- **Stable Video Infinity**: https://arxiv.org/abs/2510.09212
- **VBench**: https://arxiv.org/abs/2311.13510
- **OpenS2V-Eval**: https://arxiv.org/abs/2505.20292
- **Triton (OpenAI)**: https://openai.com/research/triton
- **ZeRO**: https://arxiv.org/abs/1910.02054
- **CFG-Zero-Star**: https://arxiv.org/abs/2503.18886
- **VideoAlign (RL reward)**: https://arxiv.org/abs/2501.13918

这篇 paper 我觉得最有启发的是 **architecture choice 与 inference regime 的耦合**这个深层 insight——pre-trained model 的 inference regime 包含了它学到的所有 bidirectional interaction pattern，改变这个 regime（如 causal mask）会引入难以弥补的 quality loss。Guidance Attention 给了一个 elegant 的 alternative，让 architecture 适配 autoregressive generation 而 inference regime 保持 bidirectional。这个思路可能可以推广到其他"改造 pre-trained model 为 autoregressive generator"的场景。
