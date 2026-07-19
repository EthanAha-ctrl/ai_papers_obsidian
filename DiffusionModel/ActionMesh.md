---
source_pdf: ActionMesh.pdf
paper_sha256: 37e911c6beee8d4e7b7214c96e8677cdf76c2a1bd2d075e37363aff09e6c2e0f
processed_at: '2026-07-18T01:07:38-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# ActionMesh 深度解读: 把 3D diffusion "撑" 出一个时间轴

Andrej, 这篇 paper 我读了之后直觉上觉得很干净, 核心贡献就一句话: **拿一个已经训好的 image-to-3D latent diffusion 模型 (TripoSG), 用最小的改动给它加上 temporal axis, 再用一个 temporal autoencoder 把"独立的 4D 形状序列"折叠成一个"单一 topology 的 animated mesh"**. 这两步分离得很漂亮, 因为它们各自承担了不同的责任 — Stage I 负责"几何随时间演化且跨帧一致", Stage II 负责"绑定到同一个 mesh 拓扑上以支持 texture/retargeting".

下面我尽量按 build-intuition 的方式展开.

---

## 1. 为什么 naive 方法会失败 — Figure 3 的现象

最直觉的 baseline 是: 拿 image-to-3D model (比如 TripoSG) 逐帧跑, 每帧都从相同 Gaussian noise 开始. 结果会出现两类不一致:
- **全局朝向漂移**: 同一只马, 在 frame 1 是面向 +x, frame 5 突然面朝 +z. 因为每帧 latent diffusion 的 conditioning (DINOv2 feature) 是独立的, 没有跨帧信号.
- **高频细节 flicker**: 表面小尖刺、毛发等细节在帧间"闪". 这是 latent diffusion 本身的 multimodality — 同一个 image 可以解出多个合理 3D, 每帧独立 sample 就会跳到不同 mode.

**直觉 takeaway**: image-to-3D 模型的 latent space 是"shape 的 manifold", 但它没有"motion 的 manifold". 你需要让 N 个 latent 在同一个 denoising trajectory 里互相看到对方, 才能把它们锁在同一个 motion mode 上. 这就是 inflated attention 的全部动机.

---

## 2. 背景: 3DShape2VecSet / TripoSG 的 latent space

先回顾下 backbone 的结构 (因为 ActionMesh 的所有改动都是对它的"极小扰动"):

- **Encoder $\mathcal{E}_{3D}$**: 在 surface 上采样 dense points $P \in \mathbb{R}^{N_p \times 6}$ (XYZ + normal), 通过 cross-attention 把 $P$ 作为 context, 一组 learned query vectors (VecSet, 数量 $T$) attend 到 $P$ 上, 再过若干 self-attention layers, 得到 latent $z \in \mathbb{R}^{T \times D}$.
- **Decoder $\mathcal{D}_{3D}$**: 拿 $z$, 过 self-attention, 再用一个 cross-attention, 输出任意 query 3D point 的 occupancy / SDF, 然后 marching cubes [Lorensen 1987] 出 mesh.
- **Generative model $\mathcal{G}_{3D}$**: 一个 DiT (diffusion transformer, [Peebles 2023]), decoder-only + cross-attention 注入 image conditioning (DINOv2 [Oquab 2023]). TripoSG 用的是 **rectified flow** [Lipman 2023, Liu 2023], flow timestep $s \in [0, 1000]$ 做 Fourier embedding, 拼成一个 extra token.

关键点: 这个 latent $z$ 是"shape 的紧凑表达", 每个维度都承载了几何信息. ActionMesh 的所有 temporal 改动都在这个 latent space 里做, 不动 image condition, 不动 decoder 结构 (除了 Stage II 的改动).

参考: 
- 3DShape2VecSet: https://arxiv.org/abs/2301.11411
- TripoSG: https://arxiv.org/abs/2505.06535
- Rectified Flow: https://arxiv.org/abs/2209.03003
- DiT: https://arxiv.org/abs/2212.09748
- DINOv2: https://arxiv.org/abs/2304.07193

---

## 3. Stage I: Temporal 3D Diffusion — 把 attention "inflated"

### 3.1 Inflated self-attention 的公式

设输入 tensor:
$$\mathbf{X} \in \mathbb{R}^{N \times T \times D}$$
- $N$: 帧数 (paper 用 16)
- $T$: tokens per shape (训练 1024, 推理 2048)
- $D$: feature dim

inflated attention (Eq. 1):
$$\text{infattn}(\mathbf{X}) = \text{reshape}^{-1}\big(\text{selfattn}\big(\text{reshape}(\mathbf{X})\big)\big)$$

其中 $\text{reshape}(\mathbf{X}) \in \mathbb{R}^{1 \times (NT) \times D}$ — 把帧维度和 token 维度合并成单个 sequence. 这样 self-attention 在长度 $NT$ 的 token 序列上算, **任意帧的任意 token 都能 attend 到任意帧的任意 token**, 实现跨帧同步.

**直觉**: 这就等价于把 N 个独立的 self-attention 拼成一个大的 self-attention. Pretrained 的 $W_Q, W_K, W_V$ 权重直接复用 — 因为 attention 本身是 permutation-invariant 的, 加一个维度进来等价于 sequence length 变长, pretrained 权重天然兼容. 这是 "inflation" 思想 (from video models, e.g., [Singer 2023 Make-A-Video], [Ho 2022 Imagen Video]) 的精髓: **不要重新学一个新结构, 只把已有结构的"作用域"扩大**.

复杂度从 $O(N \cdot T^2 \cdot D)$ 变成 $O((NT)^2 \cdot D)$, 所以用 FlashAttention2 [Dao 2024] 来扛.

参考:
- Make-A-Video: https://arxiv.org/abs/2209.10692
- Imagen Video: https://arxiv.org/2210.02303
- MVDream (analogous idea for multi-view): https://arxiv.org/abs/2308.16512
- FlashAttention-2: https://arxiv.org/abs/2307.08691

### 3.2 为什么还要 rotary positional embedding (RoPE)

光 inflation 之后还有一个微妙问题: attention 是 permutation-invariant 的, inflate 之后模型并不知道 token 的"frame index". 这会导致 **相邻帧之间的 latent 出现 sub-frame jitter** — 因为 attention weights 看不出"这个 token 来自第 3 帧, 那个来自第 4 帧, 它们应该比较像".

paper 的解法是 **在 inflated self-attention 内部注入 relative frame index 的 RoPE** [Su 2023]. RoPE 的好处是它对相对位置编码, 而不是绝对位置 — 所以"frame 3 ↔ frame 4"和"frame 12 ↔ frame 13"会用同样的相对偏置, 这正符合"相邻帧应该平滑过渡"的归纳偏置.

Table 3 的 ablation 证实了这一点: 去掉 rotary embedding, CD-3D 从 0.050 → 0.054, CD-4D 从 0.069 → 0.084. CD-4D 退化得更狠 (21% 相对退化), 因为 4D 一致性正是依赖于"相邻帧 token 之间的相对位置信号".

参考 RoPE: https://arxiv.org/abs/2104.09864

### 3.3 Masked generation — 让模型能"锚定"已知的 source mesh

这一步是为了解锁 {3D+text}-to-animation 等下游应用. 核心思路: 在训练时, 随机挑 $N_S \in \{1, 2, 3\}$ 个 latent 当作 "source" (即已知, noise-free), 剩下 $N_T = N - N_S$ 个当作 "target" (要 denoise).

具体做法:
1. 对 source latent, **不加噪声** (即 flow matching step $s=0$).
2. 把这个 $s=0$ 信号通过 flow timestep embedding 告诉模型 — 比 CAT3D [Gao 2024] 那种 binary mask 注入更"自然", 因为 rectified flow 的 timestep embedding 已经是个连续信号通道.
3. 训练 loss 只在 target latents 上算, source 不算.

推理时 (video-to-4D):
- 先用 image-to-3D model 在某一帧上跑一个 mesh $\mathcal{M}_1$, 编码成 $z_1^* = \mathcal{E}_{3D}(\mathcal{M}_1)$.
- 在每个 denoising step 前把 $z_1^*$ (clean) 复制到 latent sequence 的对应位置, 让 noisy target tokens 能 attend 到这个 clean "anchor".

**直觉**: 这等价于一个"半监督"的 diffusion — 一部分 latent 是 observation, 一部分是要 infer 的. 模型本质上学到的是一个"given partial shape sequence, predict the rest" 的条件分布. 这个建模方式使得 {3D+text}-to-animation / motion transfer / autoregressive extrapolation 都是同一种 pattern: 把已知 mesh 当作 source, 把要预测的当 target.

参考:
- CAT3D: https://arxiv.org/abs/2405.01412
- MaskGit (masked generation 思想): https://arxiv.org/abs/2112.01527
- MAGE: https://arxiv.org/abs/2211.13150

Ablation (Table 3): 去掉 masked modeling, CD-4D 从 0.069 → 0.116, 退化 68%. 说明这个机制不仅是"功能扩展", 对 video-to-4D 本身也有 strong inductive bias — 因为 anchor mesh 给了模型一个明确的几何 prior, 防止它在 latent space 里乱跑.

---

## 4. Stage II: Temporal 3D Autoencoder — 把独立形状"折叠"成同一个 mesh

### 4.1 问题定义

Stage I 输出的是 4D mesh $\{(\mathbf{V}_k, \mathbf{F}_k)\}_{k=1}^{N}$, 每帧 topology 都不同. 这对下游 (texturing, retargeting) 没用. 我们需要把它转成 animated mesh $\{(\mathbf{V}_k, \mathbf{F})\}_{k=1}^{N}$ — 同一个 face connectivity $\mathbf{F}$, 只是 vertex 位置随时间变.

数学上: 给定 reference mesh $(\mathbf{V}, \mathbf{F})$, 学一组 displacement $\delta_k$, 使得 $(\mathbf{V} + \delta_k, \mathbf{F})$ 逼近 $(\mathbf{V}_k, \mathbf{F}_k)$ 的表面. 这是一个 deformation field regression 问题.

### 4.2 架构

输入:
- 一组 shape latents $\mathbf{Z} = \{z_k\}_{k=1}^N$, 每个 $z_k = \mathcal{E}_{3D}(P_k)$, 其中 $P_k$ 是第 $k$ 帧的 surface point cloud (XYZ + normal).
- 一对 source/target timesteps $(t_{src}, t_{tgt})$.
- 一组 query points (推理时 = reference mesh vertices, 训练时 = 在 source mesh 上随机采样).

Decoder $\mathcal{D}_{4D}$ 处理:
1. 全部 $z_k$ 过 self-attention layers (同样用 inflated attention + RoPE).
2. $(t_{src}, t_{tgt})$ 做 Fourier embedding, 拼起来当作 extra token, 和 shape tokens 一起过 self-attention.
3. 最后一个 cross-attention: query 是 query 3D points (position + normal), context 是 self-attention 处理后的 tokens, 输出 query point 从 $t_{src}$ 到 $t_{tgt}$ 的 displacement.

**关键设计决策**: query point 不仅用 position, 还用 **normal**. 为什么? 因为 mesh 上存在"空间邻近但拓扑距离远"的点 — 比如手指相握时, 两个手指的某些 vertex 在 3D 空间上几乎重合, 但拓扑上属于不同 finger. 只用 position, 网络会 ambiguous. 加上 normal, 两根相对的手指 normal 方向相反, 就能 disambiguate.

Table 4 ablation: 去掉 normals, CD-M 从 0.137 → 0.148, 退化 8%. 去 normal 主要影响 motion 一致性 (CD-M), 因为 deformation field 的歧义会直接导致 vertex correspondence 错乱.

### 4.3 为什么 $(t_{src}, t_{tgt})$ 当 token 比 append 到 query 好

Paper 试了两种:
- (A) 把 $(t_{src}, t_{tgt})$ Fourier 嵌入后拼成一个 token, 和 shape tokens 一起进 self-attention.
- (B) 把 $(t_{src}, t_{tgt})$ 拼到 query point feature (position+normal) 后面.

Table 4: A 比 B 好 (CD-M 0.137 vs 0.151). paper 指出 B 有工程优势 — 同一组 shape tokens 处理后, 缓存 context, 不同 $(t_{src}, t_{tgt})$ 的 query 可以共享, 加速. 但精度差.

**直觉**: 把 timestep 放进 self-attention 意味着 shape tokens 本身被 "time-aware" refine. 整个 shape 表达根据当前要预测的 (src, tgt) 对做不同处理. 而 B 只在最后的 cross-attention 才注入 time, shape tokens 还是 time-agnostic 的, 信息融合得晚.

---

## 5. 评估指标 — CD-3D / CD-4D / CD-M 的几何含义

Paper 用了三个 metric, 互补:

### 5.1 Chamfer distance (Eq. 2)
$$
\text{CD}(\mathbf{S}_k, \hat{\mathbf{S}}_k) = \frac{1}{P} \sum_{i=1}^{P} \Big[ \min_{\hat{\mathbf{x}} \in \hat{\mathbf{S}}_k} \|\mathbf{x}_{k,i} - \hat{\mathbf{x}}\|_2^2 + \min_{\mathbf{x} \in \mathbf{S}_k} \|\mathbf{x} - \hat{\mathbf{x}}_{k,i}\|_2^2 \Big]
$$
- $\mathbf{S}_k$: GT mesh 上采样的 $P=100{,}000$ points
- $\hat{\mathbf{S}}_k$: predicted mesh 上采样的 $P$ points
- 第一项: 每个 GT 点到 prediction 最近点的距离平方 (recall)
- 第二项: 每个 pred 点到 GT 最近点的距离平方 (precision)
- 对称求和 → bidirectional chamfer

### 5.2 CD-3D (Eq. 3)
$$
\text{CD-3D} = \frac{1}{K} \sum_{k=1}^{N} \text{CD}(\mathbf{S}_k, \bar{\mathbf{S}}_k^{3D})
$$
- $\bar{\mathbf{S}}_k^{3D}$: 每帧独立做 ICP [Besl 1992] 把 pred 对齐到 GT (估计 $(\mathbf{r}_k, \mathbf{t}_k) \in SO(3) \times \mathbb{R}^3$)
- 含义: 每帧 geometry 的独立质量. 不评估 motion.
- $K$ 等于 $N$ (paper 中 N=16, K 应该是 typo, 应该是 N)

### 5.3 CD-4D
- 用 ICP4D: 只在第一帧算一次 $(\mathbf{r}_1, \mathbf{t}_1)$, 应用到所有帧. 这就强制要求 pred 的全局坐标系和 GT 的全局坐标系一致, 任何 frame-to-frame 的旋转 / 平移漂移都会被惩罚.
- 含义: 4D 一致性, 即"不仅每帧几何对, 整体在时间维度上的姿态也对".

### 5.4 CD-M (Motion Chamfer, Eq. 4-6)
$$
\sigma_i = \arg\min_j \|\mathbf{x}_{1,i} - \hat{\mathbf{x}}_{1,j}\|_2^2
$$
$$
\tau_i = \arg\min_j \|\mathbf{x}_{1,j} - \hat{\mathbf{x}}_{1,i}\|_2^2
$$
- $\sigma_i$: 第 $i$ 个 GT 点在 pred 上最近的那个点 index (在第 1 帧建立, 之后所有帧都用同一映射)
- $\tau_i$: 第 $i$ 个 pred 点在 GT 上最近的那个点 index
- 注意: 这个对应关系 **只在第 1 帧建立, 然后固定**, 不重新做 nearest neighbor.

$$
\text{CD-M} = \frac{1}{NP} \sum_{k=1}^{N} \sum_{i=1}^{P} \|\mathbf{x}_{k,i} - \hat{\mathbf{x}}_{k,\sigma_i}\|_2^2 + \|\mathbf{x}_{k,\tau_i} - \hat{\mathbf{x}}_{k,i}\|_2^2
$$
- 项 1: GT 点 $i$ 在第 $k$ 帧的位置 vs 它对应的 pred 点 $\sigma_i$ 在第 $k$ 帧的位置.
- 项 2: pred 点 $i$ 在第 $k$ 帧的位置 vs 它对应的 GT 点 $\tau_i$ 在第 $k$ 帧的位置.

**直觉**: CD-M 测的是"如果我用第 1 帧的 correspondence 把 GT 和 pred 的 vertex 绑起来, 之后每帧它们是否一起动". 这就直接衡量了 motion fidelity — topology consistent mesh 应该让 vertex $i$ 在 GT 和 pred 里走同一条轨迹.

CD-M 是 ActionMesh 真正核心优势的地方 — 因为它有 topology-consistent 输出, 别的方法 (LIM / DreamMesh4D / V2M4) 没有显式 vertex correspondence, 在 CD-M 上自然吃亏.

---

## 6. 主结果 (Table 1) 解读

| Method | Time | CD-3D | CD-4D | CD-M |
|---|---|---|---|---|
| LIM [31] | 15min | 0.095 | 0.127 | 0.258 |
| DM4D [18] | 35min | 0.095 | 0.140 | 0.247 |
| V2M4 [4] | 35min | 0.063 | 0.223 | 0.500 |
| **ActionMesh** | **3min** | **0.050** | **0.069** | **0.137** |

- **CD-3D 提升 21%** (vs V2M4 0.063): geometry 质量. 这主要归功于 TripoSG backbone 强 (Stage I 的 ablation 显示如果直接 per-frame 跑 TripoSG, CD-3D 也就 0.050, 说明 Stage II 不损害 geometry).
- **CD-4D 提升 46%** (vs LIM 0.127, 提升 46%, paper 写的是相对 best baseline): temporal consistency. 这就是 inflated attention + rotary embedding + masked modeling 的功劳.
- **CD-M 提升 45%** (vs DM4D 0.247): motion fidelity. 这是 topology consistent + temporal 3D autoencoder 联合的成果.
- **速度快 10×**: 3min vs 15–35min. 因为完全是 feed-forward, 没有 test-time optimization. 这是 paper 强调的"production-ready"关键.

### Ablation 关键观察 (Table 2)
- Full: 0.050 / 0.069 / 0.137
- w/o Stage II: 0.050 / 0.069 / — (motion 没法算, 因为没 topology consistent mesh)
- w/o Stage I & II (直接 per-frame TripoSG): 0.050 / 0.187 / — (CD-4D 大幅退化, 说明 Stage I 是 4D 一致性的来源)
- w/ Craftsman backbone (替代 TripoSG): 0.072 / 0.117 / 0.216 (method 仍然 work, 说明 framework 对 backbone choice 鲁棒)

**重要直觉**: Stage I 不显著改善 CD-3D (0.050 → 0.050), 但显著改善 CD-4D (0.187 → 0.069). 这说明 Stage I 的作用不是"让每帧 geometry 更好", 而是"让每帧 geometry 互相一致". 完美符合 paper 的设计意图 — backbone 已经负责 per-frame quality, temporal 3D diffusion 只负责"synchronization".

---

## 7. 训练细节 (Supplementary C)

- **数据**: 13,200 个 animated sequences (来自 Objaverse / Objaverse-XL + internal). 每个序列 16–128 keyframes, 渲染 16 个视角 / keyframe.
- **训练序列长度**: 16 frames.
- **优化**: AdamW, lr=1e-4, weight decay=1e-2, bfloat16 mixed precision, batch=96, 170K steps.
- **关键 trick**: **deform a single canonical point cloud over time** (而不是每帧重新采样). 这保证了训练时 latents 之间的 vertex correspondence 是"真"的. 如果每帧独立采样 point cloud, 模型根本学不到 motion.
- **训练 tokens**: $T=1024$, 推理时 $T=2048$ (更多 tokens 精度更高, 训练受 memory 限制).
- **Stage II 训练**: 用同一份数据, 但**不需要 canonical point cloud 假设**, 因为 surface points 可以每帧独立采样. 这是一个 nice 解耦 — Stage I 需要"真"correspondence (来自 Objaverse rig), Stage II 只需要 shape sequence.
- **Autoencoder loss**: $\ell_2$ 在 vertex positions 上.

---

## 8. 应用层面 — masked modeling 的真正威力

Section 3.4 列了 5 个应用, 全部复用同一个 trained model:

1. **{3D+text}-to-animation**: 渲染 mesh 一张图 → video model 文生视频 → ActionMesh (把已知 3D 当 source).
2. **{Image+text}-to-4D**: image-to-3D → 上一步.
3. **Text-to-4D**: text-to-image → 上一步.
4. **Motion transfer / retargeting**: 把 video A 当 source, mesh B 当 reference, 直接跑 (虽然训练时没见过 mismatched 对象, 但模型 generalize 得很好 — Figure 4).
5. **Animation extrapolation**: autoregressive, 把上一 chunk 最后一帧当 source, 处理下一 chunk. Table 5/6 显示了 context window $c_w$ 的影响 — $c_w=1$ 是 sweet spot (efficiency vs accuracy).

**直觉**: 因为 masked generation 的本质是"conditional shape sequence completion", 任何"已知一部分, 预测其余"的任务都能套进去. 这就是为什么 paper 强调 "minimal architectural changes" — 你不需要为每个任务改结构, 只需要改"哪些 token 是 source".

---

## 9. 局限性 (Figure 7)

1. **Topological changes**: 不能处理 (一只手松开 → 握拳 这种细节 topology 变化, 或者章鱼触手缠绕). 因为假设了 fixed $\mathbf{F}$.
2. **Strong occlusions**: 如果 reference frame 的某部分被遮挡, 或者 motion 过程中某 part 消失, 模型会 hallucinate 失败.

Paper 在 conclusion 里 hint 了未来方向: **topology-aware latent updates** — 在 latent space 里做局部的 instantiate / fuse / remove parts, 而不是显式 mesh surgery. 这听起来像是要结合 recent works on part-based generation (e.g., [RigAnything 2025] https://arxiv.org/abs/2503.01117, [PartPacker] 等).

---

## 10. 我的几个 intuition takeaways

1. **"Inflate, don't reinvent"**: 这篇 paper 的核心哲学. 所有改动都是对 pretrained 3D model 的 minimal surgery (inflated attention + RoPE + masked flow step), 让 4D 数据稀缺的问题被 3D pretrained prior 缓解. 这和 Make-A-Video 当年对 image diffusion 的做法 [Singer 2023] 同构 — temporal 3D = spatiotemporal (2D image → video) 的 3D 版本.

2. **Stage I / Stage II 分工很美**: Stage I 只负责"latent 空间里的 4D 一致性", 不用管 topology. Stage II 只负责"topology binding", 不用管 motion generation. 两步可以独立训练 / 推理 / 替换 backbone (Table 2 显示 Craftsman 也能 work). 这比端到端训一个 4D model 要 modular 得多.

3. **CD-M 这个指标设计很好**: 在第 1 帧建 correspondence, 之后固定 — 这就强制评估"vertex trajectories" 而不是 "frame geometry". 这正是 topology-consistent mesh 真正的卖点. 未来 4D benchmark 应该都加上这个 metric.

4. **为什么 masked generation 也能提升 video-to-4D 本身**: 这是一个不太显然的发现 (Table 3 ablation). 直觉解释: image-to-3D 在单帧上比 video-to-4D 强很多 (前者有海量数据, 后者数据稀缺). 通过 masked modeling, 你把 image-to-3D 的强 prior "注入" video-to-4D 的生成 — 一个"已知好 frame + 推其他 frames"的任务比"无 anchor 推所有 frames"容易得多. 这是一种"distillation via conditioning".

5. **Canonical point cloud trick**: supplementary 里那句 "deforming a single canonical point cloud over time—rather than re-sampling points at each timestep—is critical for stable training" 是个很实用的工程点. 任何做 4D shape reconstruction 的人都该注意 — point cloud correspondence 是训练信号的根本.

6. **和 ShapeGen4D [Yenphraphai 2025, https://arxiv.org/abs/2505.11490] 的对比**: 两者都是 feed-forward video-to-4D mesh, 但 ShapeGen4D 不显式 enforce global topology consistency, 所以 CD-M 上 ActionMesh 应该明显领先 (paper 没给 ShapeGen4D 的 CD-M 数字, 因为没开源).

7. **可能的下一步**: 把 Stage I 的 masked latent idea 推广到 **arbitrary number of source meshes** (paper 现在 $N_S \in \{1,2,3\}$), 这就能支持"多视角 anchor" — 类似 CAT3D 的多图条件, 但是是 3D mesh 条件. 这可能让模型处理更复杂 motion 的重建.

8. **另一个直觉**: paper 没强调但我觉得很关键的是 — temporal 3D autoencoder 的 decoder 输出的是 **deformation field**, 不是 occupancy. 这意味着 reference mesh 的 vertex count / topology 可以**任意** (推理时把任意 mesh 的 vertex 当 query). 这就解锁了 motion transfer — 你可以把马的 motion 迁移到牛 / 龙 / 任何同 semantic class 的 mesh, 因为 deformation field 是 continuous 的, 不依赖特定 topology.

9. **和 L4GM [Ren 2024, https://arxiv.org/abs/2404.01322] / 4DGT [Xu 2025] 的本质区别**: L4GM 输出 4D Gaussians, 4DGT 也是 Gaussians, 都不是 mesh. Gaussians 渲染好, 但**没有 vertex-to-vertex correspondence**, 没法做 texture transfer / retargeting. ActionMesh 的 topology consistency 是 production pipeline 的硬需求 — 游戏引擎 / 动画软件都需要 mesh.

10. **潜在的 scaling 路径**: 现在 13K sequences. 如果上到 100K+ (Objaverse-XL 有 10M+ objects, 假设 10% 有 rig 就是 1M sequences), 且训练 sequence length 从 16 推到 64+, temporal 3D diffusion 的 motion prior 应该会显著变强. 这条路线和 video diffusion 的 scaling story 完全平行.

---

## 11. 相关论文链接 (方便进一步阅读)

- **3DShape2VecSet**: https://arxiv.org/abs/2301.11411
- **TripoSG**: https://arxiv.org/abs/2505.06535
- **Trellis**: https://arxiv.org/abs/2412.01506
- **CLAY**: https://arxiv.org/abs/2406.13797
- **Craftsman3D**: https://arxiv.org/abs/2405.14974
- **Hunyuan3D 2.5**: https://arxiv.org/abs/2506.05465
- **Dora-VAE**: https://arxiv.org/abs/2412.17812
- **LRM**: https://arxiv.org/abs/2311.04400
- **MVDream**: https://arxiv.org/abs/2308.16512
- **CAT3D**: https://arxiv.org/abs/2405.01412
- **CAT4D**: https://arxiv.org/abs/2410.06485
- **SV4D**: https://arxiv.org/abs/2407.17085
- **L4GM**: https://arxiv.org/abs/2404.01322
- **DreamMesh4D**: https://arxiv.org/abs/2410.06974
- **Consistent4D**: https://arxiv.org/abs/2402.05959
- **ShapeGen4D**: https://arxiv.org/abs/2505.11490
- **V2M4**: https://arxiv.org/abs/2506.04978
- **LIM**: https://arxiv.org/abs/2410.05717
- **Motion2VecSets**: https://arxiv.org/abs/2405.20341
- **Make-It-Animatable**: https://arxiv.org/abs/2411.12149
- **MagicArticulate**: https://arxiv.org/abs/2503.09212
- **RigAnything**: https://arxiv.org/abs/2503.01117
- **DriveAnyMesh**: https://arxiv.org/abs/2503.13942
- **DiT**: https://arxiv.org/abs/2212.09748
- **Rectified Flow**: https://arxiv.org/abs/2209.03003
- **FlashAttention-2**: https://arxiv.org/abs/2307.08691
- **RoPE**: https://arxiv.org/abs/2104.09864
- **DINOv2**: https://arxiv.org/abs/2304.07193
- **Perceiver IO**: https://arxiv.org/abs/2107.14795
- **Objaverse**: https://arxiv.org/abs/2212.08013
- **Objaverse-XL**: https://arxiv.org/abs/2307.05663
- **DAVIS**: https://arxiv.org/abs/1605.01778
- **Marching Cubes** (Lorensen 1987): 原始 SIGGRAPH paper, 经典算法.
- **ICP** (Besl 1992): https://ieeexplore.ieee.org/document/121791
- **Project page**: https://remysabathier.github.io/actionmesh/

---

## 12. 一句话总结 (build intuition)

ActionMesh = **3DShape2VecSet 的 latent space + 一个 inflated attention 把 N 个 shape latents 在同一个 denoising trajectory 里 lock 住 + 一个把 arbitrary shape sequence fold 成单一 topology mesh 的 deformation autoencoder**. 三个组件都用 minimal modification 复用 pretrained 3D prior, 因此在 13K sequences 上就能跑出 SOTA, 速度比 per-scene optimization 快 10×, 且输出真正能直接喂给游戏引擎 / 动画 pipeline 的 animated mesh. 整套设计的精髓是 **把 4D 生成拆成"shape manifold 上的时间演化"和"topology 绑定"两个正交子问题**, 然后各自用最适合的工具解决.
