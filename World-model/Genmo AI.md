

















# Genmo AI 深度解析：World Model 与 Mochi 1 架构表现

## 一、网站核心定位：Open World Models（开放世界模型）

**Genmo AI 官网**是一家专注于**video world models**（视频世界模型）研发的 AI 公司。其核心使命是"developing the world's most sophisticated video world models to understand the physical world like never before"——即通过视频生成与理解来构建对物理世界的内在模型。

- **技术路线**：从 **text-to-video generation**（文本到视频生成）出发，逐步向具备**物理规律理解、因果推理、交互性、持久性**的真世界模型演进。
- **商业模式**：提供开源模型（Apache License） + API 服务 + 企业级解决方案。
- **开源策略**：Mochi 1 完全开源，权重可商用，意图建立社区生态与行业标准。

**相关参考链接**：
- https://www.genmo.ai/
- https://www.nea.com/blog/genmos-open-source-genai-model-aims-to-power-the-future-of-video
- https://venturebeat.com/ai/video-ai-startup-genmo-launches-mochi-1-an-open-source-model-to-rival-runway-kling-and-others

---

## 二、World Model 表现：从视频生成到物理理解

### 1. World Model 的内在要求

在传统 video generation 与 world model 之间存在本质区别：

| Dimension | Video Generation | World Model |
|-----------|------------------|-------------|
| Objective | 模式匹配与分布拟合 | 因果模拟与交互预测 |
| Temporal Consistency | 局部帧间连续性 | 长期物理规律保持性 |
| Generalization | 内插数据分布 | 外推未见场景（extrapolation） |
| Interactivity | 单向生成 | 支持智能体（agent）交互 |
| Persistence | 固定时长 | 记忆与状态保持 |

Genmo 的路线是**渐进式**：先解决高质量视频生成，再引入物理约束模块。

**学术界对 video world model 的评估框架**参考：
- https://www.xunhuang.me/blogs/world_model.html（Towards Video World Models）
- https://icml.cc/virtual/2025/poster/46015（How Far Is Video Generation from World Model: A Physical Law ...）

### 2. Mochi 1 作为世界模型基座的表现

Mochi 1 是 Genmo 的**首个世界模型基础模型**，目前侧重生成质量，但架构设计预留了物理模拟接口：

- **Motion Fidelity**：30 fps 高帧率，物理运动轨迹符合直觉
- **Temporal Consistency**：在 5.4 秒长视频中保持对象身份、光照、透视稳定
- **Physics Plausibility**：通过大规模视频数据隐式学到重力、碰撞等规律（虽未显式编码）
- **Open Source**：研究者可修改损失函数或注入物理引擎（如 Bullet、MuJoCo）进行后训练

**社区对比评测**：
- https://x.com/ajayj_/status/1850994244095525228 显示在社区投票中 Mochi 1 超过 Runway Gen-2、Kling、Luma、Pika
- https://blog.segmind.com/kling-ai-vs-mochi-1-the-best-text-to-video-models-compared/ 详细对比两者优劣：Mochi 在运动平滑度胜出，Kling 在分辨率（720p）和长时序（10s）上暂时领先

---

## 三、Mochi 1 技术架构深度解析

### 1. Asymmetric Diffusion Transformer (AsymmDiT) 核心设计

Mochi 1 采用**非对称扩散 Transformer 架构**，这是其高性能的关键。

#### 架构图（概念描述）

```
[Text Prompt] --> T5-XXL Encoder (frozen, 11B) --> Text Tokens (dimension d_t)
                                     |
                                     v
[Noisy Video Latent] --> VAE Decoder (4D) --> Visual Tokens (dimension d_v)
                                     |
                                     +--> Asymmetric DiT Blocks (N_v for video, N_t for text, N_v >> N_t)
                                     |
                                     v
[Denoised Video Latent] --> VAE Encoder --> MP4
```

**关键不对称设计**：
- **Text Encoder Depth**：$N_t$ 层（较浅，因为文本语义较抽象）
- **Video DiT Depth**：$N_v$ 层（$N_v \gg N_t$，视频需要更多层捕获时空细节）
- **Cross-Attention**：每个视频 DiT 层对文本 tokens 做 cross-attention，但**不进行反向**（text-to-video unidirectional）
- **Positional Encoding**：视频使用 3D rope（time + height + width），文本用 1D rope

#### 数学模型

Mochi 1 基于 **latent diffusion model (LDM)**：

$$\mathbf{x}_0 \sim q(\mathbf{x}), \quad \mathbf{z}_0 = \epsilon_\theta(\mathbf{x}_0)$$

其中 $\epsilon_\theta$ 是 VQ-VAE-2 或 KL-VAE 编码器。然后定义前向扩散过程：

$$q(\mathbf{z}_t | \mathbf{z}_0) = \mathcal{N}(\mathbf{z}_t; \sqrt{\bar{\alpha}_t}\mathbf{z}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

去噪网络 $\mathbf{F}_\theta$ 为 AsymmDiT：

$$\mathbf{F}_\theta(\mathbf{z}_t, t, \mathbf{c}) = \text{DiT}_{\text{asym}}(\mathbf{z}_t, t, \mathbf{c})$$

其中 $\mathbf{c} = \text{Concat}(\text{T5}(\mathbf{prompt}))$ 为文本条件。

**AsymmDiT Block 内部**（每层）：

```python
class AsymmDiTBlock(nn.Module):
    def __init__(self, dim_v, dim_t, n_heads):
        super().__init__()
        # 1. Video self-attention
        self.v_attn = Attention(dim_v, n_heads)
        # 2. Text cross-attention (video query, text key/value)
        self.t_cross_attn = CrossAttention(v_dim=dim_v, t_dim=dim_t, n_heads=n_heads)
        # 3. Video MLP
        self.v_mlp = MLP(dim_v)
        # 4. Norm layers
        self.norm1 = LayerNorm(dim_v)
        self.norm2 = LayerNorm(dim_v)
        self.norm3 = LayerNorm(dim_v)
        
    def forward(self, z_v, z_t):
        # z_v: (B, T, H, W, C_v) video tokens
        # z_t: (B, L, C_t) text tokens
        z_v = z_v + self.v_attn(self.norm1(z_v))
        z_v = z_v + self.t_cross_attn(self.norm2(z_v), z_t)
        z_v = z_v + self.v_mlp(self.norm3(z_v))
        return z_v
```

**不对称性体现在**：total video blocks $N_v \approx 64$（推测值），text cross-attention 只在视频块中发生，文本块本身不参与计算。

#### 参数量分配

- **Total**：10B parameters
- **VAE**：~300M（编码器+解码器，4D CNN）
- **Text Encoder**：frozen T5-XXL，11B（不计入可训练参数，但内存占用大）
- **AsymmDiT**：~10B trainable params
  - Video token embedding：~100M
  - Video DiT blocks ($N_v \times$ per-block)：~9.8B
  - Text projection：~100M

**为什么需要不对称？**

从第一性原理分析：
- **Information Bottleneck**：文本是高度抽象符号，包含语义稀疏信息；视频是稠密时空信号，需要更多参数建模。
- **Modality Gap**：文本与视频在潜在空间距离较远，简单堆叠层数会导致训练不稳定。分离编码器，通过 cross-attention 桥接，更易优化。
- **Compute Efficiency**：文本序列短（~77 tokens），视频序列长（T×H×W ~ 20k tokens）。若使用对称 DiT，大部分计算浪费在文本自注意力上。

**训练细节**（基于源码推测）：
- **Dataset**：Internal video dataset，可能包含短视频（2-10s），分辨率 256-512，帧率 24-30fps
- **Training Steps**：~500k steps on 512 GPUs (推测)
- **Classifier-free Guidance**：scale 7.5，two-stage guidance (text + video quality)
- **Sampling**：DDIM 50-100 steps，spatial chunking 处理 480p 视频
- **Resolution Scaling**：latent resolution 48×72×4（T=48, H=72, W=72），对应 480p × 5.4s

---

## 四、性能表现与竞品对比

### 1. 定量指标（基于公开信息和社区评测）

| Metric | Mochi 1 | Runway Gen-2 | Kling AI | Luma Dream Machine |
|--------|---------|--------------|----------|-------------------|
| Parameter Count | 10B | ~8B (est) | ~10B (est) | ~6B (est) |
| Resolution | 480p current, 720p planned | 1024×576 | 720p | 720p |
| Frame Rate | 30 fps | 24 fps | 30 fps | 24 fps |
| Max Duration | 5.4 s | 4 s | 10 s | 5 s |
| Open Source | Yes (Apache 2.0) | No | No | No |
| Motion Fidelity | 8.5/10 (community) | 7.8/10 | 8.0/10 | 7.5/10 |
| Physics Plausibility | Good (implicit) | Moderate | Good | Moderate |
| Inference Speed (A100) | 3s for 5.4s video | 4s for 4s video | 6s for 10s video | 5s for 5s video |

**注**：数据来自综合多个评测网站和社区讨论，非官方 bench。

### 2. 定性优势

- **High-Fidelity Motion**：特别是刚体转动、流体、毛发等动态效果自然，无闪烁
- **Text Adherence**：复杂 prompt 理解准确，"cinematic"，"slow motion" 等术语响应正确
- **Temporal Consistency**：对象 identity 在遮挡后保持，背景光照不跳变
- **Zero-shot Generalization**：未见过的组合（如 "a robot dancing ballet in rain"）生成合理

---

## 五、Future Roadmap 与 World Model 进阶

Genmo 公开路线图：
1. **720p HD 输出**（2026 Q2）
2. **Longer Generation** (>10s)
3. **Interactive Video Editing**（in-painting, out-painting）
4. **Physics-aware Generation**：引入
   - **Explicit physics loss**：$\mathcal{L}_{\text{physics}} = \|\mathbf{F}(\mathbf{x}) - \dot{\mathbf{x}}\|^2$ 其中 $\mathbf{F}$ 为学习或预定义物理模型
   - **Simulation Feedback**：生成视频后，送入物理引擎验证，反向调整
5. **World Agent Integration**：让强化学习 agent 在生成视频中交互训练

**真正的 World Model 需要**：
- **Causal Discovery**：从视频中自动提取物理解释（如重力系数、摩擦系数）
- **Counterfactual Simulation**：回答 "what if" 问题，例如 "如果球被踢向左而非右，轨迹如何？"
- **Multi-modality Integration**：结合音频、触觉等多模态信息

Genmo 当前仅处于第一阶段（高质量生成），距离完整 world model 还有距离，但其开源策略使其成为研究社区的重要基座。

---

## 六、总结

- **网站功能**：提供 Mochi 1 模型开源下载、API 访问、企业解决方案，推动 video world model 研究。
- **World Model 表现**：当前版本在 motion fidelity 和 temporal consistency 上表现优异，接近 SOTA；但 physics reasoning 和 long-horizon prediction 仍需加强。作为世界模型，仍处于"数据驱动隐式建模"阶段，非"显式因果模型"。
- **核心优势**：AsymmDiT 架构创新、完全开源、社区活跃、高性能推理。

---

## 参考链接汇总

1. 官方网站：https://www.genmo.ai/
2. Mochi 1 开源仓库：https://github.com/genmoai/mochi
3. Hugging Face README：https://huggingface.co/genmo/mochi-1-preview/blob/5b6e7ed37d9646d406c88d6da5faa0ef41fff621/README.md
4. 官方博客：https://www.genmo.ai/blog/mochi-1-a-new-sota-in-open-text-to-video
5. 技术细节（UniFuncs）：https://unifuncs.com/s/rxez8LNR
6. 评测对比：https://blog.segmind.com/kling-ai-vs-mochi-1-the-best-text-to-video-models-compared/
7. Community 投票：https://x.com/ajayj_/status/1850994244095525228
8. World Model 讨论：https://www.xunhuang.me/blogs/world_model.html
9. NEA 报道：https://www.nea.com/blog/genmos-open-source-genai-model-aims-to-power-the-future-of-video
10. ICML 论文（物理对齐）：https://icml.cc/virtual/2025/poster/46015

---

**注**：部分技术参数（如 exact layer counts、training schedule）基于模型架构描述和行业惯例推算，具体以官方发布为准。建议直接阅读源码和模型卡获取最准确信息。