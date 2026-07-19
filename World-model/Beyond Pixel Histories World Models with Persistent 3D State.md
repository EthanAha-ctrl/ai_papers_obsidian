---
source_pdf: Beyond Pixel Histories World Models with Persistent 3D State.pdf
paper_sha256: e90622e9d705093ca9d19bce7c106657acd05f1c3586bc31c90cfee14ff5067e
processed_at: '2026-07-18T17:12:11-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# PERSIST: Persistent 3D State World Models 深度解析

## 1. 核心Motivation与问题定位

### 1.1 现有AR video diffusion world model的瓶颈

当前 frontier interactive world model (Oasis, GameGen-X, Diamond, Matrix-Game, Genie 3) 的主流范式是 **AR video diffusion**：以 causal DiT 为 backbone，将 history of past pixel observations $O_{t-K}^{t-1}$ 与 actions $A_{t-K}^t$ 作为 condition，去denoise 当前 frame $o_t$。这种 formulation存在几个根本性缺陷：

1. **Context window有限**：即使把observations压缩到latent space（e.g. Oasis用spatiotemporal token compression），AR模型能condition的history实际只有几秒
2. **Pixel information redundancy**：每个pixel observation是partial、viewpoint-dependent、time-fixed的快照，对同一个3D location会反复从不同angle观测，redundancy极高
3. **Retrieval scaling差**：WorldMem这类key-frame retrieval方法，随history bank增长，identifying relevant past evidence变得越来越难，retrieval cost与episode长度耦合
4. **Out-of-view dynamics丢失**：pixel history无法建模occluded regions的演化，例如agent转过身时背后的水在流

### 1.2 PERSIST的核心insight

PERSIST 的核心 insight 可以用一句话概括：**把记忆载体从pixel observations替换为主动生成的3D latent state**。这是把传统game engine的"persistent world state + render pipeline"思想移植到learned world model中。

具体地，将 hidden state $s$ 的proxy定义为 $\tilde{s} = \langle \mathbf{w}, \mathbf{c} \rangle$：
- $\mathbf{w} \in \mathbb{R}^{12\times 12\times 12\times 48}$：world-frame，agent周围的固定cuboid region的3D latent
- $\mathbf{c} \in \mathbb{R}^{10}$：camera state，编码agent在$\mathbf{w}$中的viewpoint

这个proxy的crucial property：**memory cost是fixed的**，与episode长度无关。Camera $\mathbf{c}$ 本质上是一个spatial lookup key，去query $\mathbf{w}$中relevant的subset。

## 2. Pipeline架构详解

### 2.1 整体数据流

完整pipeline（Algorithm 1）在每个timestep $t$ 执行：

```
1. Agent根据 o_{t-1} 产生 action a_t
2. W_θ 生成 world-frame w_t（3D latent scene演化）
3. 3D-VAE decode w̄_t → w_t
4. C_θ 预测 camera c_t
5. R(c_t, w_t) → w_{2D,t}（depth-ordered feature stack）
6. P_θ denoise pixel latent ō_t，conditioned on w_{2D,t} + past pixels
7. 2D-VAE decode → o_t
```

### 2.2 World-Frame Prediction $\mathcal{W}_\theta$

**输入condition**：
$$\bar{\mathbf{w}}_t \sim \mathcal{W}_\theta(\bar{\mathbf{w}}_t \mid \bar{W}_{t-K}^{t-1}, A_{t-K}^t, C_{t-K-1}^{t-1}, \bar{O}_{t-K-1}^{t-1})$$
其中 $K$ 是temporal context window（3D-S/XL都是 $K=8$）。

**关键设计**：
- **3D-VAE**：把 $48^3 \times 2138$ voxel grid（one-hot encoding of voxel semantic labels）压到 $\bar{\mathbf{w}} \in \mathbb{R}^{12^3 \times 48}$，138M参数
- **DiT backbone**：spatial module从2D扩展到3D，处理XYZ三个维度
- **Position embedding**：弃用RoPE（Su et al. 2024, https://arxiv.org/abs/2104.09864），改用每个voxel token centroid的absolute XYZ position embeddings。这是个很有意思的选择——RoPE的rotary mechanism在3D下不一定有rotational equivariance的interpretation，absolute embedding更适合discrete voxel grid
- **Plücker embeddings**（Sitzmann et al. 2021, https://proceedings.neurips.cc/paper/2021/hash/a11ce019e96a4c60832eadd755a17a58-Abstract.html）：从camera $C_{t-K-1}^{t-1}$ 计算pixel-to-3D projection信息，channel-wise concat到pixel patches，作为cross-attention的输入
- **Action & camera embedding**：joint MLP，inject via AdaLN（Peebles & Xie 2023, https://arxiv.org/abs/2212.09748）
- 3D-XL用patch size=1（1728 spatial tokens），3D-S用patch size=2（216 spatial tokens），8倍spatial token ratio

**关键能力**：$\mathcal{W}_\theta$支持conditioning on $\bar{W}=\varnothing$，意味着可以从initial condition $(o_0, c_0)$ 推断 $\mathbf{w}_0$。这就让PERSIST能在inference时不需要ground truth 3D supervision。这一点非常重要——training用GT 3D，inference不用，本质上是把3D-VAE作为隐式的"image-to-3D inverse graphics"先验。

### 2.3 Camera Model $\mathcal{C}_\theta$

**Camera参数化** $\mathbf{c} = \langle \text{pos}, \text{rot}, \text{fov} \rangle$：
- $\text{pos} \in \mathbb{R}^3$：camera在world-frame中的position
- $\text{rot} \in \mathbb{R}^6$：6D continuous rotation representation（Zhou et al. 2019, https://arxiv.org/abs/1812.07035）——比quaternion或Euler更continuous，更适合neural net预测
- $\text{fov} \in \mathbb{R}$：field of view

**Residual prediction**：输出 $\bar{\mathbf{c}} = \langle \text{pos}, \Delta\text{pitch}, \Delta\text{yaw}, \Delta\text{fov} \rangle$，其中
$$\Delta\text{pitch} = \text{pitch}_t - \text{pitch}_{t-1}, \quad \Delta\text{yaw} = \text{yaw}_t - \text{yaw}_{t-1}$$

这是个很重要的设计选择——residual prediction让camera model学到dynamics而非绝对position，pitch/yaw的small delta更接近unimodal distribution，MSE loss能更好工作。

**Architecture**：1D causal transformer with RoPE temporal embeddings，pos和rot走separate positional embedders，concatenate fov，linear projection成token。$\mathbf{W}$裁到inner-most $4^3$ voxels（agent的immediate surroundings）与A联合embed后AdaLN注入。Context window $K_\mathcal{C}=8$。

**Loss**：per-component MSE on $\bar{\mathbf{c}}$。234M参数，单A100上16小时训完。

### 2.4 World-to-Pixel Projection $\mathcal{R}$

这是把3D latent state"渲染"成2D observation guidance的关键模块：

$$\mathcal{R}(\mathbf{c}, \mathbf{w}) = (\tilde{\mathbf{w}}_{2D}, \mathbf{d})$$

- $\tilde{\mathbf{w}}_{2D} \in \mathbb{R}^{h\times w\times l\times m}$：每个pixel位置上depth-ordered的 $l$ 个voxel features，每个feature有 $m$ channels
- $\mathbf{d} \in \mathbb{R}^{h\times w\times l}$：每个voxel feature在camera space的linear depth
- 最终 $\mathbf{w}_{2D} \in \mathbb{R}^{h\times w\times z}$，$z = l \times (m+1)$

**Implementation**：GPU-native triangle rasterization，把voxel features assign到一个static voxel-grid mesh的faces上，然后用 **depth-peeling**（Bavoil & Myers 2008, https://developer.download.nvidia.com/SDK/10/opengl/src/glhlsl/examples/doc/dual_depth_peeling.txt；Nagy & Klein 2003）生成per-pixel depth-ordered feature stack。

Depth-peeling的关键作用：rendering多个depth layer，避免visibility culling丢失被occlude的几何。每个pixel位置上保留 $l$ 个depth-sorted features，类似multiplane image（MPI, Zhou et al. 2018, https://arxiv.org/abs/1805.09817）的思想。

因为mesh topology在fixed resolution下不变，可以构造一次后只更新vertex attributes——很高效的实现选择。

### 2.5 Pixel Frame Prediction $\mathcal{P}_\theta$

$$\bar{\mathbf{o}}_t \sim \mathcal{P}_\theta(\bar{\mathbf{o}}_t \mid W_{2D_{t-K}}^t, A_{t-K}^t, \bar{O}_{t-K}^{t-1})$$

**核心insight**：$\mathcal{P}_\theta$本质上是一个**learned deferred shader**（Thies et al. 2019, https://arxiv.org/abs/1904.12356）。它需要预测3D latents没提供的信息：texture、lighting、particle effects、screen-space overlays。

**关键channel分配**：$\mathbf{w}_{2D}$的channel数被刻意分配得比 $\bar{\mathbf{o}}$ 多——$w_{2D}$ embedder output 752 channels，$\bar{\mathbf{o}}$ embedder 16 channels。这是个很重要的inductive bias，**强迫模型把3D latent作为primary information source**，pixel history只作为refinement。

**Architecture**：causal DiT with interleaved spatial & temporal modules（Decart 2024的Oasis backbone，https://oasis-model.github.io/）。$\mathbf{w}_{2D}$通过channel-wise 1D-conv压到latent space，与 $\bar{\mathbf{o}}$ channel-wise concatenate。Pixel context window $K_\mathcal{P}=16$（比3D的K=8更长，因为pixel space dynamics更高频）。460M参数。

## 3. Flow Matching Formulation

### 3.1 Conditional Flow Matching Objective

Rectified flow（Lipman et al. 2023, https://arxiv.org/abs/2210.02747；Albergo & Vanden-Eijnden 2023, https://openreview.net/pdf?id=li7qeBbCR1t）的noising process：
$$x^\tau = (1-\tau)x^0 + \tau x^1, \quad x^1 \sim \mathcal{N}(0, \mathbb{I})$$

$\tau \in [0,1]$，$\tau=0$是clean data，$\tau=1$是pure noise。Linear interpolation保证velocity field在ODE意义上rectified（straight trajectories）。

**Training objective**：
$$\mathcal{L}(\theta) = \left\| \mathcal{V}_\theta(x^\tau, \tau) - (x^0 - x^1) \right\|^2, \quad \tau \sim p(\tau)$$

其中 $\mathcal{V}_\theta$ 预测的velocity vector $v = x^0 - x^1$ 指向clean data。这与diffusion model的noise prediction $\epsilon$-parameterization等价但geometric interpretation更清晰：模型直接预测从noise到data的displacement vector。

**Inference**：从 $x^1$ 迭代denoise到 $x^0$：
$$x^{\tau - d^k} = x^\tau + \mathcal{V}_\theta(x^\tau, \tau) d^k, \quad \tau \leftarrow \tau - d^k$$

PERSIST用20步denoising，scheduling function:
$$\tau^k = \frac{\eta k}{1 + (\eta-1)k}, \quad \eta=3$$

这个scheduling的intuition：$\eta>1$时前期step小后期step大，early阶段精细denoise高频细节，late阶段大step加速收敛。与DDIM的uniform schedule相比，对image这种low-frequency dominant data更有效。

### 3.2 Exposure Bias Mitigation

**Diffusion Forcing**（Chen et al. 2024, https://arxiv.org/abs/2411.18276）：训练时给每个frame independent sample noise level。Inference时current frame从random noise开始denoise，past context frames用fixed small noise $\tau_{\text{ctx}}$：
- $\mathcal{W}_\theta$：$\tau_{\text{ctx}} = 0.02$
- $\mathcal{P}_\theta$：$\tau_{\text{ctx}} = 0.1$

**Cross-component noise augmentation**：因为 $\mathcal{W}_\theta$ inference时condition在 $\mathcal{P}_\theta$ 的predictions上，反之亦然，有distributional shift。训练时给 $\bar{O}$ 加10% noise训 $\mathcal{W}_\theta$，给 $\bar{W}$ 加10% noise训 $\mathcal{P}_\theta$。这让各组件可以**separately trained，无需end-to-end fine-tune**就组合inference——这点很巧妙，借鉴了Self-Forcing（Huang et al. 2025c, https://arxiv.org/abs/2506.08009）的思想但更轻量。

## 4. 环境定义与问题形式化

形式化定义interactive environment：
$$\mathcal{E} = \langle \mathbb{S}, \mathbb{O}, \mathbb{A}, \Omega, p \rangle$$

- $\mathbb{S}$：latent state space
- $\mathbb{O}$：observation space
- $\mathbb{A}$：action space
- $\Omega: \mathbb{S} \to \mathbb{O}$：partial projection function
- $p(s' \mid a, s)$：transition probability

Learning world simulation objective：minimize $\mathbb{E}[D(O^n, \tilde{O}^n)]$，其中 $D$ 是observation sequence的distance metric。

**Key insight from Section 4**：直接condition在hidden state $s$上理论上更好，但$s$可能arbitrarily complex或unmeasurable（如video game的program memory），且学习$\Omega$恢复observations是comparably hard problem。Proxy $\tilde{s} = \langle \mathbf{w}, \mathbf{c} \rangle$ 是个**操作上tractable的中间表示**——既有3D structure的geometric prior，又能避免直接建模不可观测的true state。

## 5. 实验设置

### 5.1 Dataset (Craftium / Luanti)

- **Engine**：Luanti (开源voxel engine, formerly Minetest, https://github.com/luanti-org/luanti)
- **Platform**：Craftium（Malagon et al. 2025, https://arxiv.org/abs/2503.05331）
- **规模**：~40M interactions, ~100K trajectories, 460 hours @ 24Hz
- **3D observations**：$48^3$ voxel grid centered on agent，每个voxel是integer label（thousands of possible configs: water, stone, air等）
- **Actions**：23维multi-hot encoding（key presses + discretized mouse movements）

**关键设计选择**：训练在**procedurally generated worlds**而非single fixed map（与Diamond、Oasis不同），增加spatial & temporal consistency的难度——model无法overfit到fixed layout。

### 5.2 训练compute breakdown

| Component | GPU | Days | 参数量 |
|-----------|-----|------|--------|
| 2D-VAE | 8×A100 | 4 | 227M |
| 3D-VAE | 8×A100 | 12 | 138M |
| 3D-S denoiser | 8×H100 | 3 | 686M |
| 3D-XL denoiser | 8×H100 | 10 | 686M |
| Pixel denoiser $\mathcal{P}_\theta$ | 8×H100 | 10 | 460M |
| Camera model $\mathcal{C}_\theta$ | 1×A100 | 0.67 | 234M |

Optimizer: AdamW, lr=1e-4，batch size 64（camera model 256）。

### 5.3 Evaluation Modes

四种action policy专门设计来stress-test不同方面：
- **Free Play**：random action sampling
- **Move Forward / Backward**：translation + 周期性spinning（观察四周）
- **Orbit**：circular trajectory + 始终朝向圆心（extreme spatial revisiting test）

Orbit mode是检验spatial memory最强的test——agent反复从不同angle看同一片region。

## 6. 实验结果

### 6.1 主表 (Table 1)

| Method | FVD↓ | Visual Fidelity↑ | 3D Consistency↑ | Temporal Consistency↑ | Overall↑ |
|--------|------|---|----|----|---|
| Oasis | 706 | 2.1±0.1 | 1.9±0.1 | 1.8±0.1 | 1.9±0.1 |
| WorldMem | 596 | 1.7±0.09 | 1.7±0.09 | 1.5±0.08 | 1.5±0.07 |
| PERSIST-S | 209 | **2.8±0.1** | **2.7±0.1** | 2.5±0.1 | 2.6±0.09 |
| PERSIST-XL | 181 | 2.8±0.09 | 2.5±0.09 | 2.5±0.09 | 2.6±0.08 |
| PERSIST-XL+w₀ | **116** | **3.2±0.1** | **2.8±0.1** | **2.8±0.1** | **3.0±0.1** |

**关键观察**：
1. **FVD改进3.4×**（Oasis 706 → PERSIST-XL+w₀ 116）——绝对huge gain
2. **3D consistency从1.9 → 2.8**：human rater评分提升47%，这是persistent 3D state最直接的payoff
3. **3D-S vs 3D-XL几乎tied**：spatial resolution从216 tokens到1728 tokens（8×）提升有限。这说明**3D representation的存在性比resolution更重要**——只要有一个persistent 3D anchor，就能lock住大部分spatial consistency
4. **WorldMem反而比Oasis差**：这个很反直觉。原因可能是retrieval的key-frame viewpoint matching有noise，反而引入inconsistency；而且WorldMem需要600 GT frames warm-start，仍然表现最差
5. **PERSIST-XL+w₀提升**：提供GT $\mathbf{w}_0$让FVD从181→116，说明 $\mathcal{W}_\theta$ 对initial 3D state的inference还不完美，仍有改进空间

### 6.2 Head-to-Head Comparisons (Appendix C.1)

从Figure 9-12的score delta矩阵能看出：
- PERSIST-S在3D Spatial Consistency上微胜PERSIST-XL (+0.21)——可能是lower resolution的3D latent有更强的spatial smoothing prior
- PERSIST-XL在Overall Score上微胜 (+0.14)
- Ground truth在所有metric上压倒所有method

### 6.3 Per-set Performance (Tables 9-12)

不同evaluation mode下表现：
- **Circle around** (orbit): PERSIST-XL+w₀优势最大（overall 3.1 vs PERSIST-S 2.5）
- **Free play**: PERSIST-XL+w₀ 3.4 vs PERSIST-S 3.0
- **Forward/backward look around**: PERSIST-S反而更好——可能是low-frequency motion下lower-res 3D prior更稳定

## 7. Novel Capabilities

### 7.1 Single-Image 3D Outpainting

$\mathcal{W}_\theta(\varnothing \mid o_0, c_0) \to \mathbf{w}_0$ 能从单张RGB推断整个 $48^3$ region的3D structure。Figure 13显示同一input frame可以sample出diverse但coherent的initial world frames——agent背后、侧面、地下的structure都是imagined的。这是image-to-3D inverse graphics的emergent ability。

### 7.2 Mid-Episode 3D Editing

任何timestep $t$ 可以pause generation，把 $\mathbf{w}_t$ 替换为 $\tilde{\mathbf{w}}_t$（手动编辑或程序化修改），再resume。Figure 6展示：
- 全局terrain/biome edit
- 精细asset placement（种树）

这种fine-grained control在pixel-space world model中几乎不可能实现——必须从pixel反推3D再编辑再render，而PERSIST直接在3D space操作。

### 7.3 Off-Screen Persistent Dynamics

Figure 7展示了emergent property：agent视野外的水会持续flow，最终overflow onto agent。这是persistent 3D state的直接结果——$\mathcal{W}_\theta$在每一步都evolve整个world-frame，不只是agent visible的部分。

Pixel history-based model无法做到这点，因为occluded region在pixel observation中根本没出现过。

## 8. Limitations与Future Work

### 8.1 现有限制

1. **GT 3D supervision依赖**：training需要GT $\mathbf{w}$，限制applicability到能提供3D annotation的simulators
2. **Exposure bias still存在**：Figure 14显示2000步时仍有glitches——tree trunk消失、wood blocks错预测、texture color drift。但3D state的grounding作用能让 $\mathcal{P}_\theta$ recovery from这些artifacts
3. **有限空间memory**：$48^3$ voxel grid随agent移动会discard远距离信息，与Minecraft chunk loading类似但更受限

### 8.2 Future Roadmap (Appendix D)

1. **Wild training via 2D-to-3D foundation models**：用VGGT（Wang et al. 2025, https://arxiv.org/abs/2503.11651）或SAM 3D（Team et al. 2025, https://arxiv.org/abs/2511.16624）从pixels生成synthetic 3D annotations作为pre-processing
2. **End-to-end post-training**：因为pipeline fully differentiable，可以在generated rollouts上做端到端RL-style post-training来align train/inference distribution（参考Self-Forcing Huang et al. 2025c）
3. **3D memory bank**：取代fixed $48^3$ grid，用spatial chunk loading的mechanism。3D store是inherently deduplicated和spatially organized，比pixel memory bank高效很多——相同3D location不会被多次store

## 9. 关键Insight总结

### 9.1 Why 3D State Works

核心机制：把memory从**temporal extent**转移到**spatial extent**。

- Pixel history-based：信息以time-ordered sequence存储，retrieval随history length增长而退化
- 3D state-based：信息以spatial coordinate indexed存储，retrieval cost固定，与episode length无关

Camera $\mathbf{c}$ 作为spatial lookup key：把temporal retrieval problem转化为spatial projection problem（$\mathcal{R}$），后者是well-defined的geometric operation而非学习task。

### 9.2 Why "Learned Deferred Shader" Matters

$\mathcal{P}_\theta$作为learned deferred shader的formulation（Thies et al. 2019）让model能学到任意rendering function，包括non-physical effects：particle、lighting、screen-space overlays。这避免了NeRF ray-marching的cost（Mildenhall et al. 2021, https://arxiv.org/abs/2003.08934；Muller et al. 2022, https://arxiv.org/abs/2201.05989），同时保留geometric grounding。

### 9.3 与现有方法的关系

- 与Genie 3（Ball et al. 2025）的关系：Genie 3是latent dynamics world model，但latent space没有explicit 3D structure。PERSIST可以视为"structured latent"版本
- 与Voyager（Huang et al. 2025b, https://arxiv.org/abs/2503.19981）的关系：Voyager做long-range video diffusion生成explorable 3D scene，但是static environment。PERSIST是dynamic
- 与Marble（World Labs 2025, https://www.worldlabs.ai/blog/marble-world-model）：multimodal world model，static 3D。PERSIST的dynamic 3D state是显著differentiation

### 9.4 VAE Architecture Choices

- **2D-VAE**：ViT-based（Dosovitskiy et al. 2021, https://arxiv.org/abs/2010.11929），借鉴Oasis实现。Input $360\times 640\times 3$ → patch size 10 → $\bar{\mathbf{o}} \in \mathbb{R}^{36\times 64\times 16}$。227M参数，KL coefficient $1e-6$（接近deterministic autoencoder）
- **3D-VAE**：3D-ResNet based on Trellis（Xiang et al. 2025, https://arxiv.org/abs/2412.01298）。Input $48^3 \times 2138$（2138是voxel class数one-hot）→ $\bar{\mathbf{w}} \in \mathbb{R}^{12^3 \times 48}$。Cross-entropy loss on voxel class labels（不是MSE）——因为voxel是discrete semantic label而非continuous field。138M参数

3D-VAE用cross-entropy而2D-VAE用MSE是个重要细节——反映3D voxel representation的discrete nature vs 2D pixel的continuous nature。

## 10. 实现细节中的Engineering Trade-offs

### 10.1 Token Count Analysis (Table 8)

| Model | Temporal Tokens | Spatial Tokens |
|-------|----------------|----------------|
| $\mathcal{W}_\theta$-S | 8 | 216 |
| $\mathcal{W}_\theta$-XL | 8 | 1728 |
| $\mathcal{P}_\theta$ | 16 | 576 |
| $\mathcal{C}_\theta$ | 8 | n/a |

3D-XL的1728 spatial tokens + 8 temporal = 13,824 tokens，每个token hidden size 1024。这是相当大的attention cost。但PERSIST-XL vs PERSIST-S的结果说明这个overhead的marginal benefit有限——值得反思的cost/benefit tradeoff。

### 10.2 Inference Cost

每frame需要：
1. $\mathcal{W}_\theta$ denoising：20 steps × (1728 + 8 + cross-attention to pixels) tokens
2. $\mathcal{C}_\theta$ forward pass：cheap
3. $\mathcal{R}$ rasterization：GPU-native，cheap
4. $\mathcal{P}_\theta$ denoising：20 steps × (576 + 16) tokens

总compute大概2-3× single-frame video diffusion，但带来3D consistency的huge quality gain。

### 10.3 VAE Pre-encoding Optimization

Training时先train VAE，再pre-encode所有latents到disk，避免每个batch重复encode。3D-VAE 12 days训练完后，pre-encode加速后续denoiser训练。这是diffusion model训练的standard optimization但容易忽略。

## 11. 对World Model社区的影响

### 11.1 概念性贡献

PERSIST证明了一个hypothesis：**pixel observations并非world state的好载体**。这个观察对world model社区有几个implications：

1. **Memory mechanism设计**：从temporal retrieval转向spatial indexing
2. **Latent space structure**：inductive bias from 3D geometry > generic latent
3. **Decoupling of perception & dynamics**：$\mathcal{P}_\theta$专注perception/rendering，$\mathcal{W}_\theta$专注dynamics，各司其职

### 11.2 与Embodied AI的联系

对RL agent training的意义：persistent 3D state让world model可以作为RL的simulator，agent可以在其中long-horizon train without spatial drift。Pixel-based world model在1000+ step后quality collapse，根本不能作为RL training environment。PERSIST的2000 step还能保持coherent是重要里程碑。

### 11.3 与Genie系列的对比

Genie 1是frame-prediction world model，Genie 2是latent dynamics video diffusion，Genie 3（Ball et al. 2025, Nature https://www.nature.com/articles/s41586-024-07848-0）是scalable latent world model。PERSIST与Genie系列的key difference是**explicit 3D structure in latent**——Genie系列的latent是unstructured，靠model自己emerge 3D understanding。PERSIST的hypothesis是这种emergence太难，不如直接build in。

## 12. 我的Critique与延伸思考

### 12.1 Strengths

- Conceptually clean decomposition：world/camera/renderer三模块clearly motivated
- Strong empirical results：3.4× FVD improvement是substantial
- Emergent capabilities rich：off-screen dynamics、3D editing、single-image initialization
- Exposure bias mitigation strategies well-designed（diffusion forcing + cross-component noise）

### 12.2 Weaknesses

- **GT 3D supervision是major limitation**：限制applicability。Future work用2D-to-3D foundation models generate synthetic annotations是个path，但foundation model本身的3D error会propagate到world model
- **Voxel discretization的inductive bias**：48³ voxel grid + cross-entropy loss是strong prior，对non-voxel环境（real-world video）迁移需要重新设计3D representation
- **Long-horizon still drift**：Figure 14的2000-step glitches说明exposure bias仍存在，即使mitigation策略帮了大忙
- **Camera model simplicity**：MSE loss + residual prediction对aggressive camera motion（rapid yaw）可能困难
- **3D-S vs 3D-XL marginal gain**：8× spatial token cost换marginal FVD improvement，说明3D representation的**existence比resolution重要**——这反而是个positive finding但也质疑XL的necessity

### 12.3 Open Questions for Future Research

1. **Continuous 3D representation**：能否用Gaussian splatting或neural implicit surface替代discrete voxel grid？会失去3D-VAE的semantic supervision优势但gain resolution
2. **Hierarchical 3D memory**：48³ grid + chunk loading机制如何处理large open-world？类似Minecraft的chunk system
3. **Object-centric decomposition**：把world-frame分解为object slots，每个object有own dynamics？类似object-centric world model
4. **Action-conditioned vs action-free**：能否做action-free version，让world model自主evolve？这对video generation有用
5. **Multi-agent generalization**：current formulation只有一个agent，multi-agent场景下camera model需要扩展
6. **Differentiable rendering的局限**：depth-peeling的 $l$ layers是fixed upper bound，对dense occlusion scene可能不够
7. **Self-supervised 3D**：能否在 $\mathcal{W}_\theta$ training中用contrastive loss between多viewpoint rendering of same $\mathbf{w}$，完全绕开GT 3D supervision？

## References

- PERSIST paper: https://francelico.github.io/persist.github.io (原文链接)
- Rectified Flow: https://arxiv.org/abs/2209.03003
- Diffusion Forcing: https://arxiv.org/abs/2411.18276
- DiT: https://arxiv.org/abs/2212.09748
- 6D Rotation Representation: https://arxiv.org/abs/1812.07035
- Plücker Embeddings / Light Field Networks: https://proceedings.neurips.cc/paper/2021/hash/a11ce019e96a4c60832eadd755a17a58-Abstract.html
- Luanti (Minetest): https://github.com/luanti-org/luanti
- Craftium: https://arxiv.org/abs/2503.05331
- Oasis: https://oasis-model.github.io/
- WorldMem: https://arxiv.org/abs/2504.12369
- Self-Forcing: https://arxiv.org/abs/2506.08009
- VGGT (2D-to-3D): https://arxiv.org/abs/2503.11651
- SAM 3D: https://arxiv.org/abs/2511.16624
- Deferred Neural Rendering: https://arxiv.org/abs/1904.12356
- Depth Peeling: https://developer.download.nvidia.com/SDK/10/opengl/src/glhlsl/examples/doc/dual_depth_peeling.txt
- Trellis (3D-VAE): https://arxiv.org/abs/2412.01298
- NeRF: https://arxiv.org/abs/2003.08934
- Instant NGP: https://arxiv.org/abs/2201.05989
- Voyager: https://arxiv.org/abs/2503.19981
- Genie 3 / WHAM: https://www.nature.com/articles/s41586-024-07848-0
- Marble (World Labs): https://www.worldlabs.ai/blog/marble-world-model
- Minedojo: https://arxiv.org/abs/2206.11395
- FVD metric: https://arxiv.org/abs/1812.01717
- ViT: https://arxiv.org/abs/2010.11929
- RoPE: https://arxiv.org/abs/2104.09864
- Matrix-Game: https://arxiv.org/abs/2506.18701
- GameFactory: https://arxiv.org/abs/2501.08325

这篇工作的核心贡献在我看来是把"persistent state"这个game engine的核心concept引入learned world model。它不是简单地加3D representation，而是把整个world simulation objective重新decompose成符合传统引擎架构的三个模块。这种structural prior与learned components的结合，可能是让world model从几秒rollout扩展到分钟级coherent experience的关键。后续如果能解决GT 3D supervision依赖，这条路大有可为。
