---
source_pdf: WORLDPACK.pdf
paper_sha256: cb228190d978c9f8d10b44a6492438e9e6199d4651310f8b2f02983996ba1d27
processed_at: '2026-08-13T05:47:59-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# WorldPack 用人话讲

## 一句话版本

Video world model 走远了就忘了之前看过的东西，WorldPack 用两个 trick 解决：**把老帧压扁塞进 context** + **按几何位置捞回重要旧帧**。

## 为什么这事难

先 build intuition 关于问题本身。

想象你在 Minecraft 里走了一圈 A→B→C→A。你回到 A 的时候，一个真正懂世界的 model 应该能画出 A 原来的样子——你之前见过啊。但现有 model 做不到。

原因很 dumb 但很 fundamental：**attention 太贵了**。

Standard DiT 的 self-attention 复杂度是 O(m²n²d)，其中：
- m = context 里的 frame 数
- n = 每个 frame 的 token 数  
- d = token 维度

m 翻倍，cost 翻 4 倍。所以 NWM 只敢用 context = 4，Oasis 用 32。你走 100 步，前面看过的 A 早就被挤出 context window 了，model 就 "失忆" 了。

这就是为什么 LoopNav benchmark (Lian et al., 2025, https://arxiv.org/abs/2505.22976) 专门设计 A→B→A 这种 loop 任务——它精准地戳中这个痛点。

## WorldPack 的两个核心 trick

### Trick 1: Trajectory Packing（把老帧压扁）

**Intuition**: 你不需要每帧都 full resolution。最近的几帧要高清（用于 short-term dynamics），但 10 步前的帧你只需要知道 "大概 spatial layout 是啥样" 就行，可以 aggressively 压缩。

**具体怎么做**:

每帧占的 token 数按 temporal distance 衰减：

$$\ell_{t-i} = \frac{L_f}{\lambda^i}$$

变量解释：
- ℓ_{t-i}: 第 i 步前的 frame 占的有效 context length
- L_f: 最近一帧的 base context length（full resolution）
- λ: compression base，paper 用 λ = 2
- i: temporal distance（i=0 是当前帧，i=1 是上一步，...）

举例：λ = 2
- i = 0: ℓ = L_f（full res）
- i = 2: ℓ = L_f / 4（用 4×4 patchify，压 16 倍）
- i = 4: ℓ = L_f / 16（用 8×8 patchify，压 64 倍）

总 packed context length:

$$L_{\text{pack}} = S \cdot L_f + \sum_{i=S+1}^{N_{\text{con}}} \ell_{t-i} + \sum_{j=1}^{N_{\text{mem}}} \ell_{M_j}$$

变量：
- S: 保留 full resolution 的最近帧数
- 第一项: S 个高清帧
- 第二项: 压缩后的 recent history
- 第三项: 压缩后的 retrieved memory frames

Paper 的实际配置：context = 2.84 frames worth of tokens，但 trajectory = 19 frames。**你看到 19 帧只花 2.84 帧的 token budget**。这就是 packing 的 magic。

这个技术 transfer 自 Zhang & Agrawala 2025 的 "Packing Input Frame Contexts" 工作。

### Trick 2: Memory Retrieval（按几何位置捞旧帧）

**Intuition**: 你转身的时候，之前 forward 看到的东西现在就在 backward 视野里。所以应该 retrieve 那些 "之前 forward 看过、现在在你身后" 的帧。

**几何 setup**:

当前 agent 在 position p = (x_t, y_t, 0)ᵀ，看着方向 d（由 yaw θ_t 和 pitch φ_t 算出）：

$$\mathbf{d} = (\cos\phi_t \cos\theta_t, \cos\phi_t \sin\theta_t, \sin\phi_t)^\top$$

- φ_t: pitch（俯仰角）
- θ_t: yaw（偏航角）
- z = 0: Minecraft 里 agent 大致在同一高度

对每个 past frame i，算三个量：

**1. Forward distance s_i**: past frame 相对 current position 沿当前 view direction 的投影
$$s_i = (\mathbf{p}_i - \mathbf{p})^\top \mathbf{d}$$
- s_i > 0: past frame 在你前方
- s_i < 0: 在你后方
- s_i ≈ 0: 在你正侧面

**2. Lateral distance ℓ_i**: past frame 到当前 viewing ray 的垂直距离
$$\ell_i = \|(\mathbf{p}_i - \mathbf{p}) - s_i \mathbf{d}\|$$
- ℓ_i 小: past frame 几乎在你视线上

**3. Directional similarity**: 
$$\cos\Delta\theta_i = \mathbf{d}_i^\top \mathbf{d}$$
- = 1: past frame 的 view direction 和当前完全同向
- = -1: 完全反向
- = 0: 垂直

**Score function**:

$$\text{score}_i = w_c \cdot \max(\cos\Delta\theta_i, 0) \exp\left(-\frac{s_i^2}{2\sigma_s^2}\right) \exp\left(-\frac{\ell_i^2}{2\sigma_\ell^2}\right)$$
$$+ w_a \cdot \max(-\cos\Delta\theta_i, 0) \exp\left(-\frac{(s_i - \mu_s)^2}{2\sigma_s^2}\right) \exp\left(-\frac{\ell_i^2}{2\sigma_\ell^2}\right)$$

拆解：

**第一项 (w_c)**: 选同方向、spatially close 的帧。场景：你 back-trace 自己走过的路，之前同位置同方向看过的帧现在最 relevant。

**第二项 (w_a)**: 选反方向、在前方 μ_s 距离的帧。场景：你转过身，之前 forward 看过的东西现在在 backward 视野里。

参数：σ_ℓ = 10.0（lateral tolerance 大），σ_s = 0.01（forward distance 要很接近 0 或 μ_s），μ_s = 1.0，w_c = w_a = 1.0。

**Exclusion window**: 20 frames（= 1 秒 at 20 FPS）。retrieve 时跳过最近 1 秒，避免 redundancy，强制 model 去捞更远的帧。

## 架构 backbone: CDiT + RoPE

### CDiT (Conditional Diffusion Transformer)

来自 NWM (Bar et al., 2024, https://arxiv.org/abs/2412.03572)。

Standard DiT: 所有 token 互相 self-attention，O(m²n²d)。

CDiT: 
- Target frame tokens 之间 self-attention
- Past frames 作为 key/value 被 cross-attend

复杂度降到 O(mn²d)，linear in m。这是 WorldPack 能用长 context 的 foundation。

### RoPE (Rotary Position Embedding)

来自 RoFormer (Su et al., 2023, https://arxiv.org/abs/2104.09864)。

**为什么需要**: Memory retrieval 选的帧可能来自 trajectory 早期任意位置。如果用 absolute position encoding，训练时见过的 position 范围和 inference 时 memory 的 position 范围 mismatch。

RoPE 对 position m 的 query/key 应用 rotation：

$$R(m, \theta_i) = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$

- m: position index
- θ_i = 10000^{-2i/d}: 第 i 维的 frequency
- d: token dimension

Attention score q_m · k_n 只依赖 (m - n)，即 relative position。这样 "temporal distance = 100" 和 "temporal distance = 1" 在 representation space 上 comparable。

## 实验结果

### LoopNav benchmark

两个任务：
- **ABA**: A→B 探索，B→A 重建。测 pure retrieval。
- **ABCA**: A→B→C 探索，C→A 重建。测 spatial reasoning across viewpoints。

Navigation range: 5, 15, 30, 50（越大越难）。

### 关键数据 (vs NWM, 同 backbone)

| Nav Range | Metric | NWM (ctx=4) | WorldPack (ctx=2.84, traj=19) | Gain |
|-----------|--------|-------------|-------------------------------|------|
| 5 | LPIPS ABA | 0.64 | 0.52 | -19% |
| 5 | LPIPS ABCA | 0.67 | 0.56 | -16% |
| 50 | DreamSim ABA | 0.47 | 0.42 | -11% |
| 50 | FVD ABCA | 810 | 455 | **-44%** |

FVD 在 ABCA-50 上降 44% 是最 striking 的 result——long-horizon consistency 大幅提升。

### Ablation: 最关键的发现

Figure 5 的 ablation 是 paper 最 convincing 的部分：

- Base model: 标准设置
- Packing only: 压缩但没 retrieval
- Memory only: retrieval 但没压缩
- WorldPack (both): 两个都开

**结果**: 单独用任一个都只有 modest improvement，组合起来 gain 显著放大。

Figure 4 更 striking：
- ABCA-30, last 61 frames:
  - Base: ~17 LPIPS
  - Packing only: ~17 LPIPS（**几乎没改进！**）
  - Packing + retrieval: ~13 LPIPS（大幅改进）

**Intuition**: Trajectory packing 本身几乎没用——它只是让你 fit 更多帧。但如果 fit 的都是 recent frames，对 spatial reasoning（需要早期帧）没帮助。**Memory retrieval 才是 driver，packing 是 enabler**。

Compression 释放了 token budget，retrieval 决定怎么花这个 budget。

### Real-world data (RECON)

RECON dataset (Shah et al., 2021, https://openreview.net/forum?id=d_SWJhyKfVw) 是 real-world robot navigation 数据。

| Model | Context | DreamSim | LPIPS | PSNR | SSIM |
|-------|---------|----------|-------|------|------|
| Baseline | 4 | 0.23 | 0.48 | 12.7 | 0.36 |
| Packing only | 2.84 | 0.18 | 0.45 | 13.4 | 0.40 |
| WorldPack | 2.84 | 0.17 | 0.44 | 13.6 | 0.40 |

Real-world 也 work，说明 method 不只 limited to Minecraft。

### Computational cost

| Model | Context | Trajectory | Inference Time | Memory |
|-------|---------|------------|----------------|--------|
| Baseline | 4 | 4 | 0.430s | 22.08 GB |
| WorldPack | 2.84 | 19 | 0.468s | 21.78 GB |

看 19 帧只比看 4 帧多 9% 时间，memory 反而降了（因为 compression 减少了 tokens）。

## 我的几点 intuition

**1. 这本质是 hierarchical memory for world models**

类比 human memory：working memory（recent high-res）+ episodic memory（retrieved low-res）。WorldPack 实现了 explicit version。Score function hand-crafted 但 geometric grounding 让它 interpretable。

**2. 为什么 Minecraft 是 testbed**

Minecraft 有 discrete blocks、deterministic physics、rich spatial structure。LoopNav 利用这个：你走一圈回来，ground truth 确定，可以精确 measure consistency。

但 limitation: real-world physics 更 noisy，ground truth 难定义。Paper 自己承认这点。

**3. RoPE 的 hidden role**

没有 RoPE，retrieve 任意 temporal distance 的 memory 会有 distribution shift。RoPE 让 "distance = 100" 和 "distance = 1" 在 representation space 上 comparable。这是 WorldPack 能 work 的 hidden enabler，paper 没强调但很重要。

**4. Compression ratios 2^0, 2^2, 2^4 的 design**

Discrete levels 让 separate projection layers tractable。每个 compression level 有独立 input projection，initialized by interpolating from pretrained (4,4) patchify layer。如果 share 一个 projection，不同 compression 的 token statistics 会 confuse model。

**5. 与 RAG 的相似**

Memory retrieval concept 上是 RAG for video world models。Key difference: retrieval score 是 geometric（position/orientation based）而不是 semantic similarity。这 leverage 了 navigation 任务的结构——你知道 agent pose，可以精确算哪些 past views overlap。

**6. 与 Diffusion Forcing 的关系**

Oasis 用 Diffusion Forcing (Chen et al., 2024, https://arxiv.org/abs/2407.01392)，combine next-token prediction with full-sequence diffusion。WorldPack 用 CDiT with autoregressive，不同 approach。Diffusion Forcing stable 但 cost 高，WorldPack 通过 memory 间接 achieve long-term consistency。

**7. 潜在 extension 方向**

- **Learned retrieval**: 现在 score function hand-designed。Could learn neural scorer based on visual feature similarity。
- **Hierarchical retrieval**: Multi-hop: 先 coarse spatial retrieve，再 visual feature refine。
- **Adaptive compression**: 现在 λ fixed。Could learn per-frame adaptive compression。
- **Policy learning**: Paper 只做 observation prediction。Could extend to learn policy from this world model (像 DIAMOND 那样)。

**8. FVD ABCA-50 降 44% 的意义**

FVD measure video distribution quality。45% reduction 意味着 WorldPack 生成的 long rollout 在 distribution level 上 much closer to ground truth。这 validate 了 central claim: 长期 spatial consistency 大幅提升。

**9. SSIM 没显著改进的解释**

Paper 自己承认 SSIM "not decisively superior"。这是 distortion-perception tradeoff (Blau & Michaeli, 2018) 的体现：distortion-based metrics favor blurred predictions that minimize pixel-wise difference，而 WorldPack 生成 sharper、更 perceptually faithful 内容。LPIPS 和 DreamSim 改进更大说明 perceptual quality 确实提升。

**10. Exclusion window 20 frames 的巧妙**

这个设计 force retrieval 跨越更广 temporal range。如果没有 exclusion，score function 会一直选最近的高分帧，model 还是 overemphasize recent。20 frames exclusion 让 model 去捞更远的、spatially relevant 的帧。

## 相关工作链接

- **NWM (Bar et al., 2024)**: https://arxiv.org/abs/2412.03572
- **Oasis (Decart et al., 2024)**: https://oasis-model.github.io/
- **DIAMOND (Alonso et al., 2024)**: https://arxiv.org/abs/2405.12399
- **MineWorld (Guo et al., 2025)**: https://arxiv.org/abs/2504.08388
- **LoopNav (Lian et al., 2025)**: https://arxiv.org/abs/2505.22976
- **RoPE / RoFormer (Su et al., 2023)**: https://arxiv.org/abs/2104.09864
- **Diffusion Forcing (Chen et al., 2024)**: https://arxiv.org/abs/2407.01392
- **RECON dataset (Shah et al., 2021)**: https://openreview.net/forum?id=d_SWJhyKfVw
- **WorldMem (Xiao et al., 2025)**: https://arxiv.org/abs/2504.12369 (concurrent memory-based world simulation)
- **Context as Memory (Yu et al., 2025)**: https://arxiv.org/abs/2506.03141 (related memory retrieval)
- **DreamSim (Fu et al., 2023)**: NeurIPS 2023
- **Packing Input Frame Contexts (Zhang & Agrawala, 2025)**: trajectory packing 的 source

## 最 key 的 takeaway

Ablation 告诉我们：**packing alone 几乎没用，retrieval alone 改进有限，组合起来 huge improvement**。

这 suggest future work 应该 focus on better retrieval mechanisms（learned、hierarchical、multi-modal）而不是 just better compression。Compression 是 enabler，retrieval 是 driver。

整个 story 很 elegant：用 geometric structure（agent pose）guide retrieval，用 hierarchical compression fit more in fixed budget，用 RoPE handle arbitrary temporal distances。三个 component 都不复杂，但组合起来 address 了 video world modeling 的核心痛点。

---

# WorldPack: 深度技术讲解

让我来详细拆解这篇 paper，重点 build 你的 intuition 关于 spatial consistency in video world models 这个 long-standing problem。

## 核心问题：为什么 Video World Models 会 "失忆"？

想象你在 Minecraft 里走了 A→B→C 一圈，然后想回到 A。从直觉上，一个真正的 world simulator 应该能在你回到 A 时还原 A 的样子——你之前见过它，你应该记得。但是现有的 video world models 比如 Oasis、DIAMOND、NWM 都会在这类 loop navigation 任务上失败，原因很 fundamental：

**Computational cost 与 context length 的 trade-off**。

Standard DiT 的 self-attention 复杂度是 O(m²n²d)，其中：
- m = frames in context
- n = tokens per frame  
- d = token dimension

这意味着如果你想 keep 100 frames in context，attention cost 会 quadratic blow up。所以现有模型比如 NWM 只能用 context length = 4，Oasis 用 32，MineWorld 用 15。这就导致一个 critical issue：当你走远了，早期的 A 点 observation 已经被 evict 出了 context window，模型 "forget" 了 A 长什么样。

WorldPack 的核心 insight：**你不需要 keep 所有 frames at full resolution**。Recent frames 需要 high fidelity（用于 short-term dynamics prediction），但 long-term frames 只需要 provide spatial layout cues——你可以 aggressively compress 它们，甚至 retrieve 特定的 past frames that matter。

## 架构总览

WorldPack = CDiT backbone + RoPE temporal embedding + Memory Retrieval + Trajectory Packing

让我一个一个拆。

### 1. CDiT (Conditional Diffusion Transformer)

来自 NWM (Bar et al., 2024) 的工作。Key idea 是将 self-attention 和 cross-attention 分离：

- **Target frame tokens** (the noisy latent being denoised): 只在它们之间做 self-attention
- **Past frames**: 作为 key/value 被 cross-attend

这把复杂度从 O(m²n²d) 降到 O(mn²d)，linear in context length m。

为什么这个 matters：你想要 longer context，但 standard DiT 的 quadratic scaling 阻止了你。CDiT 的 linear scaling 让你 economically 扩展 context。这是 WorldPack 整个 story 的 foundation——没有 CDiT，trajectory packing 都没意义因为 attention cost 会爆炸。

### 2. RoPE (Rotary Position Embedding) for Temporal

RoPE 来自 Su et al. 2023 的 RoFormer。Standard positional encoding 在 variable-length context 上有 distribution shift 问题。RoPE 通过 rotation matrix 编码 relative position，使得无论 memory frame 来自哪里，model 都能 consistently represent 它的 temporal distance。

这点很 subtle 但 important：在 WorldPack 里，memory retrieval 选的帧可能来自 trajectory 早期任意位置。如果你用 absolute position encoding，训练时见过的 position 范围和 inference 时 memory 的 position 范围会 mismatch。RoPE 的 relative encoding 解决了这个问题。

具体公式上，RoPE 对 query/key 在 position m 应用 rotation：

$$R(m, \theta_i) = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$

其中 θ_i = 10000^{-2i/d} 是第 i 维的 frequency。这样 attention score q_m · k_n 只依赖于 (m-n)，即 relative position。

### 3. Memory Retrieval

这是 paper 里最数学化的部分。让我详细解释。

**Setting**: 你在 position p = (x_t, y_t, 0)ᵀ 看着方向 d（由 yaw θ_t 和 pitch φ_t 决定）。你有一堆历史 frames at positions p_i with directions d_i。你想 score 每个 past frame i 关于"现在是否值得 retrieve"。

**Geometry**:

当前 view direction (unit vector):
$$\mathbf{d} = (\cos\phi_t \cos\theta_t, \cos\phi_t \sin\theta_t, \sin\phi_t)^\top$$

φ_t 是 pitch（俯仰角），θ_t 是 yaw（偏航角）。注意 z=0 平面假设——Minecraft 里 agent 大致在同一高度移动。

**Three geometric quantities**:

1. **Forward distance** s_i: past frame i 相对 current position 沿 current view direction 的投影距离
$$s_i = (\mathbf{p}_i - \mathbf{p})^\top \mathbf{d}$$
- s_i > 0: past frame 在你前方
- s_i < 0: past frame 在你后方
- s_i = 0: past frame 在你侧面

2. **Lateral distance** ℓ_i: past frame 相对 current view line 的垂直距离
$$\ell_i = \|(\mathbf{p}_i - \mathbf{p}) - s_i \mathbf{d}\|$$
这个是 past frame 到 current viewing ray 的最近距离。ℓ_i 小意味着 past frame 几乎在你正前方视线上。

3. **Directional similarity** cos Δθ_i:
$$\cos\Delta\theta_i = \mathbf{d}_i^\top \mathbf{d}$$
- = 1: past frame 的 view direction 与当前完全一致（同向）
- = -1: 完全相反方向
- = 0: 垂直

**Score function**:

$$\text{score}_i = w_c \cdot \max(\cos\Delta\theta_i, 0) \exp\left(-\frac{s_i^2}{2\sigma_s^2}\right) \exp\left(-\frac{\ell_i^2}{2\sigma_\ell^2}\right)$$
$$+ w_a \cdot \max(-\cos\Delta\theta_i, 0) \exp\left(-\frac{(s_i - \mu_s)^2}{2\sigma_s^2}\right) \exp\left(-\frac{\ell_i^2}{2\sigma_\ell^2}\right)$$

让我 decompose 这个：

**第一项 (w_c term)**: 选同方向 (cos Δθ > 0)、且 spatially close (s_i ≈ 0, ℓ_i ≈ 0) 的 frames。这些是"你正在看的方向上之前看过的近处 frames"——典型场景是你 back-trace 自己走过的地方。

**第二项 (w_a term)**: 选反方向 (cos Δθ < 0)、且在前方一定距离 (s_i ≈ μ_s = 1.0) 的 frames。这听起来 counterintuitive——为什么要 retrieve 反方向的 frames？

Intuition: 当你转过身看反方向时，你之前 forward-facing 看过的东西现在就在你 backward-facing 的视野里。所以 retrieve 那些 "在 forward 方向 μ_s 距离、你之前 forward-facing 看过的 frames"——它们现在是 backward-facing 视野的内容。

**Parameter intuition**:
- σ_ℓ = 10.0: lateral tolerance 大，acceptable
- σ_s = 0.01 (for w_c term): forward distance 要非常接近 0，i.e., 你几乎在同一位置
- μ_s = 1.0 (for w_a term): 反方向 frame 在前方 1.0 单位处
- w_c = w_a = 1.0: 两种 retrieval mode 同等重要

**Exclusion window**: 20 frames (= 1 sec at 20 FPS)。意思是 retrieve 时跳过最近 1 秒的 frames，避免 redundancy。这 encourage retrieval 跨越更广的 temporal range。

### 4. Trajectory Packing

这是从 Zhang & Agrawala 2025 ( Packing Input Frame Contexts ) transfer 过来的技术。

**核心 idea**: 不同 temporal distance 的 frames 用不同 resolution 编码。

**Formal formulation**:

Recent context frames: z_t, z_{t-1}, ..., z_{t-N_con}，共 N_con + 1 帧。

Memory frames: z_{M_1}, ..., z_{M_{N_mem}}，共 N_mem 帧，从 history 中 retrieved。

每帧的有效 context length (i.e., 占用多少 tokens)：

$$\ell_{t-i} = \frac{L_f}{\lambda^i}, \quad \ell_{M_j} = \frac{L_f}{\lambda^{d_j}}$$

Variables:
- L_f: most recent frame 的 base context length (full resolution)
- λ > 1: compression base (paper 用 λ = 2)
- i: temporal distance from current (i=0 是 most recent, i=1 是 1 step before, etc.)
- d_j: memory frame j 的 "scale" (由 temporal distance 或 retrieval importance 决定)

**Example**: λ = 2, i = 2 → ℓ_{t-2} = L_f / 4 → 4×4 patchify kernel (16x compression in 2D)
λ = 2, i = 4 → ℓ_{t-4} = L_f / 16 → 8×8 patchify kernel (64x compression in 2D)

**Total packed context length**:

$$L_{\text{pack}} = S \cdot L_f + \sum_{i=S+1}^{N_{\text{con}}} \ell_{t-i} + \sum_{j=1}^{N_{\text{mem}}} \ell_{M_j}$$

Variables:
- S: 最 recent 的 S 帧 keep full resolution (不压缩)
- 第一项: S 个 full-resolution frames
- 第二项: 压缩后的 recent history frames
- 第三项: 压缩后的 retrieved memory frames

**Paper's specific setup**:
- Compression ratios: 2^0, 2^2, 2^4 (corresponding to context lengths 1, 1/4, 1/16)
- 训练时 across 19 个 context lengths
- Last 8 frames 被 memory retrieval 替换
- 不同 compression ratio 用独立的 input projection layers（不共享），initialized by interpolating from pretrained (4,4) patchify layer

**Why separate projection layers?** 不同 compression ratio 下的 token statistics 分布不同。如果 share 一个 projection，model 会 confused。单独的 projection 让每个 compression level 学自己的 representation。

**Implementation detail**: WorldPack 最终 context = 2.84 (相比 baseline 的 4)，trajectory length = 19 frames（相比 baseline 的 4 frames）。这是 compression 的 magic——你看到 19 帧但 token count 只有 2.84 frames worth。

## Preliminaries: Diffusion Formulation

让我也讲讲 paper Section 3 的数学 setup，因为这是 foundation。

**Full-sequence formulation**:
$$p_\theta(\mathbf{z}_{0:T}^{k-1} | \mathbf{z}_{0:T}^k) = \mathcal{N}(\mathbf{z}_{0:T}^{k-1}; \mu_\theta(\mathbf{z}_{0:T}^k, k), \sigma_k^2 I)$$

Variables:
- z_{0:T}^k: sequence of latent frames at noise level k
- μ_θ: predicted mean
- σ_k²: noise variance at level k
- T: sequence length

This generates whole sequence jointly，但是 sequence length 被训练时固定。

**Autoregressive formulation**:
$$p_\theta(\mathbf{z}_{t+1} | \mathbf{z}_{t-m+1:t})$$

Condition on recent m frames, predict next 1 frame. 这 allows extending beyond training horizon.

**Action-conditioned**:
$$\mathbf{z}_{t+1} \sim F_\theta(\mathbf{z}_{t+1} | \mathbf{z}_{t-m:t}, \mathbf{a}_t)$$

a_t 是 action at time t。F_θ 是 stochastic transition model。这 approximates environment dynamics p(z_{t+1} | z_{≤t}, a_{≤t})。

## Experimental Setup: LoopNav Benchmark

这是评估的核心。LoopNav (Lian et al., 2025) 专为测 long-horizon consistency 设计 in Minecraft。

**Two tasks**:

1. **ABA (Spatial Memory Retrieval)**:
   - A→B: exploration phase (model sees context)
   - B→A: reconstruction phase (model must reproduce A's scenes)
   - 测 pure retrieval——A 已经 seen 过，能否 reconstruct？

2. **ABCA (Spatial Reasoning)**:
   - A→B→C: exploration
   - C→A: reconstruction via different path
   - 测 spatial reasoning——需要 leverage accumulated memory across viewpoints

**Navigation ranges**: 5, 15, 30, 50 (size of area agent moves in)。越大越难。

**Metrics**:
- **SSIM** (↑): structural similarity, low-level alignment
- **LPIPS** (↓): perceptual fidelity (deep features)
- **PSNR** (↑): pixel-level reconstruction
- **DreamSim** (↓): deep feature similarity (Fu et al., 2023)
- **FVD** (↓): temporal video quality

## 结果分析

### Quantitative Results (Table 1, 2)

让我 highlight key comparisons with NWM (the closest baseline, same CDiT backbone):

| Nav Range | Metric | NWM (ctx=4, traj=4) | WorldPack (ctx=2.84, traj=19) | Gain |
|-----------|--------|---------------------|-------------------------------|------|
| 5 | LPIPS ABA | 0.64 | 0.52 | -19% |
| 5 | LPIPS ABCA | 0.67 | 0.56 | -16% |
| 15 | LPIPS ABA | 0.67 | 0.57 | -15% |
| 50 | LPIPS ABCA | 0.65 | 0.63 | -3% |
| 50 | DreamSim ABA | 0.47 | 0.42 | -11% |
| 50 | FVD ABCA | 810 | 455 | -44% |

Key observations:
- **LPIPS 改进巨大**，especially at small nav range——这表明 perceptual quality 大幅提升
- **FVD 在大 nav range 上改进最显著** (ABCA-50: 810 → 455)，说明 long-horizon consistency 改进最大
- **SSIM 没那么显著**，paper 解释：distortion-based metrics favor blurred predictions，而 WorldPack generates sharper, more perceptually faithful content

注意 paper 自己承认 SSIM 上 "not decisively superior"——这是 distortion-perception tradeoff (Blau & Michaeli, 2018) 的体现。

### Ablation Study (Figure 4, 5)

这是 paper 最 convincing 的部分。

**Component analysis** (Figure 5, ABA-5):
- Base model: 标准设置
- Packing only: ctx=2.84, traj=19, no retrieval
- Memory only: ctx=4, traj=1+3 retrieved memories, no packing
- WorldPack (both): ctx=2.84, traj=19 with retrieval

Result: Both 单独 components 都只 modest improvement，但组合一起 gain 显著放大。这表明 two mechanisms 是 synergistic:
- Packing 让你 fit more frames economically
- Retrieval 让你 select the right frames to fit

**Memory retrieval effect** (Figure 4):
- ABCA-30, last 61 frames: 
  - Base: ~17 LPIPS
  - Packing only: ~17 LPIPS (基本没改进!)
  - Packing + retrieval: ~13 LPIPS (大幅改进)
- ABCA-50, last 101 frames:
  - 同样 pattern——packing alone 几乎没用，加 retrieval 后 huge improvement

这是非常重要的 finding：**trajectory packing 本身几乎没用**——它只是 enable longer context，但如果 longer context 都是 recent frames，对于 spatial reasoning 任务（需要 retrieve 早期 frames）没有帮助。**Memory retrieval 是 essential**。

这 build 我的 intuition: WorldPack 的 secret 不在 compression 本身，而在 compression 释放了 budget 给 retrieval。Compression 是 enabler，retrieval 是 driver。

### Real-world Data (RECON)

RECON dataset (Shah et al., 2021) 是 real-world robot navigation 数据。

| Model | Context | DreamSim | LPIPS | PSNR | SSIM |
|-------|---------|----------|-------|------|------|
| Baseline | 4 | 0.23 | 0.48 | 12.7 | 0.36 |
| Packing only | 2.84 | 0.18 | 0.45 | 13.4 | 0.40 |
| WorldPack | 2.84 | 0.17 | 0.44 | 13.6 | 0.40 |

Real-world data 上也 work，说明 method 不只 limited to Minecraft simulator。

### Computational Efficiency (Table 4)

| Model | Context | Trajectory | Inference Time (1-step) | Memory (GB) |
|-------|---------|------------|--------------------------|-------------|
| Baseline | 4 | 4 | 0.430s | 22.08 |
| WorldPack | 2.84 | 19 | 0.468s | 21.78 |

Inference time 增加 9% (因为 retrieval overhead)，但 memory usage 反而下降（因为 compression 减少了 tokens）。你看 19 frames 的时间代价只比看 4 frames 多 9%——这就是 packing 的 efficiency。

## 我的 Intuition 与 Critique

让我 share 一些更深层的 thoughts：

**1. 这本质上是 "hierarchical memory" for world models**

类比 human memory：working memory (recent high-res) + episodic memory (retrieved low-res)。WorldPack 实现了一个 explicit version of this。Memory retrieval 的 score function 很 hand-crafted，但 geometric grounding 让它 interpretable。

**2. 为什么 Minecraft 是 testbed?**

Minecraft 的 key feature: discrete blocks, deterministic physics, but rich spatial structure。LoopNav benchmark 巧妙地利用了 Minecraft 的 spatial nature——你走一圈回来，ground truth 是确定的，可以 measure consistency。

但 paper 自己 admit limitation: 评估限于 simulator。Real-world physics 更 noisy，ground truth 难定义。

**3. RoPE 在这里的关键作用**

如果没有 RoPE，retrieve 任意 temporal distance 的 memory 会有 distribution shift。RoPE 让 "temporal distance = 100" 和 "temporal distance = 1" 在 representation space 上 comparable。这是 WorldPack 能 work 的 hidden enabler。

**4. Compression ratios 2^0, 2^2, 2^4 的 design choice**

为什么不连续？我猜是因为 discrete levels 让 separate projection layers 更 tractable。Continuous compression 会需要 more complex interpolation。这有点像 multi-scale representations in vision。

**5. 与 RAG (Retrieval-Augmented Generation) 的相似**

Memory retrieval 在 concept 上就是 RAG for video world models。但 key difference: retrieval score 是 geometric (position/orientation based) 而不是 semantic similarity。这 leverage 了 navigation 任务的 structure——你知道 agent pose，可以精确 compute 哪些 past views overlap。

**6. 潜在 extension**

- **Learned retrieval**: 现在 score function hand-designed。Could learn a neural scorer, e.g., based on visual features similarity。
- **Hierarchical retrieval**: 现在 single-level retrieval。Could do multi-hop: retrieve based on coarse spatial, then refine based on visual features。
- **Compression schedule learning**: 现在 λ fixed。Could learn adaptive compression per frame。
- **Action prediction**: Paper 只做 observation prediction。Could extend to learn policy from this world model。

**7. 与 Diffusion Forcing (Chen et al., 2024) 的关系**

Oasis 用 Diffusion Forcing，combines next-token prediction with full-sequence diffusion。WorldPack 用 CDiT with autoregressive，是不同 approach。Diffusion Forcing 的 advantage 是 stable long generation，但 cost 高。WorldPack 通过 memory 间接 achieve long-term consistency。

**8. FVD 的 ABCA-50 improvement (810 → 455)**

This 是 paper 最 striking result。FVD measures video distribution quality。45% reduction 意味着 WorldPack 生成的 long rollout 在 distribution level 上 much closer to ground truth。这 validate 了 paper 的 central claim: 长期 spatial consistency 大幅提升。

## 参考链接

- **WorldPack paper**: arXiv 链接 (paper 没给 explicit link, but 基于作者和标题可查)
- **NWM (Bar et al., 2024)**: https://arxiv.org/abs/2412.03572
- **Oasis (Decart et al., 2024)**: https://oasis-model.github.io/
- **DIAMOND (Alonso et al., 2024)**: https://arxiv.org/abs/2405.12399
- **MineWorld (Guo et al., 2025)**: https://arxiv.org/abs/2504.08388
- **LoopNav (Lian et al., 2025)**: https://arxiv.org/abs/2505.22976
- **Packing Input Frame Contexts (Zhang & Agrawala, 2025)**: 这个是 trajectory packing 的 source
- **RoPE / RoFormer (Su et al., 2023)**: https://arxiv.org/abs/2104.09864
- **Diffusion Forcing (Chen et al., 2024)**: https://arxiv.org/abs/2407.01392
- **DreamSim (Fu et al., 2023)**: NeurIPS 2023
- **RECON dataset (Shah et al., 2021)**: https://openreview.net/forum?id=d_SWJhyKfVw
- **WorldMem (Xiao et al., 2025)**: https://arxiv.org/abs/2504.12369 (related concurrent work on memory-based world simulation)
- **Context as Memory (Yu et al., 2025)**: https://arxiv.org/abs/2506.03141 (related memory retrieval work)

## 总结

WorldPack 的 elegance 在于：它用 geometric structure (agent pose) 来 guide memory retrieval，用 hierarchical compression 来 fit more in fixed budget，用 RoPE 来 handle arbitrary temporal distances。三个 components 都不复杂，但组合起来 address 了 video world modeling 的核心痛点——long-term spatial consistency。

最 key 的 takeaway 是 ablation 的发现：**packing alone 几乎没用，retrieval alone 改进有限，组合起来 huge improvement**。这 suggest future work 应该 focus on better retrieval mechanisms (learned, hierarchical, multi-modal) 而不是 just better compression。

希望这 build 了你的 intuition about 这篇 paper！如果你对某个 specific component 想深挖，告诉 me。
