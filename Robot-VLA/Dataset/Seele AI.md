

Let me search for information about this company and extract content from their website.















好的，我已经收集了大量信息。下面为你详细拆解 **Seele AI** (seeles.ai) 这家公司。

---

# 🎮 Seele AI — 全球首个 Text-to-3D-Game 的 AI Platform 深度解析

## 一、公司概况

| 维度 | 详情 |
|------|------|
| **全称** | Seele Inc. |
| **成立时间** | 2022 年 |
| **总部** | Shenzhen, China (深圳) |
| **定位** | No-Code AI Game Maker — 通过 Natural Language Prompt 生成可玩的 2D/3D Game |
| **核心投资方** | **Baidu** (百度) 领投，以及其他中国 Tech Investors |
| **最新融资** | 2025 年 5 月完成 Pre-A Round (multi-million dollar)，正在寻求新一轮融资，估值达到 **数亿美元** 级别 |
| **产品形态** | Web-based Studio (Seele Studio) + Export Pipeline |

> 参考: [Bloomberg](https://www.bloomberg.com/news/articles/2025-05-28/baidu-backed-ai-3d-game-creator-seele-seeks-new-funding) | [TechInAsia](https://www.techinasia.com/news/baidubacked-ai-3d-game-maker-seele-seeks-funding) | [PitchBook](https://pitchbook.com/profiles/company/711280-00)

---

## 二、核心产品：Text-to-Game Pipeline

Seele 自称是 **"世界首个 3D Multimodal Model"**，能够将 Text Prompt 转化为完整可玩的 3D Game。其 Pipeline 可以分解为以下阶段：

### 2.1 Pipeline 架构（推测性技术解析）

```
[Text Prompt] 
    ↓ NLU (Natural Language Understanding)
    ↓
┌──────────────────────────────────────────┐
│  Multimodal Orchestrator (LLM-based)     │
│  - 解析 Game Genre, Mechanics, Style     │
│  - 生成 Scene Graph & Game Logic Plan    │
└──────────────┬───────────────────────────┘
               ↓
  ┌────────────┼────────────┐
  ↓            ↓            ↓
[3D Asset    [Game Code   [Audio/Music
 Generator]   Generator]   Generator]
  ↓            ↓            ↓
  │  ┌─────────┘            │
  ↓  ↓                      ↓
[Scene Assembly & Rendering Engine]
  ↓
[Playable Game Output]
  ↓
[Export: Unity (.unitypackage) / Three.js (WebGL)]
```

### 2.2 各模块技术分解

#### (A) Text-to-3D Asset Generation

这是 Seele 的核心竞争力。基于当前 SOTA 的 Text-to-3D 技术栈，Seele 很可能采用了以下技术组合：

**方法一：Score Distillation Sampling (SDS)**

最初由 DreamFusion (Google, 2022) 提出的核心公式：

$$\nabla_\theta \mathcal{L}_{SDS} = \mathbb{E}_{t, \epsilon} \left[ w(t) \left( \hat{\epsilon}_\phi(z_t; y, t) - \epsilon \right) \frac{\partial g(\theta)}{\partial \theta} \right]$$

其中：
- $\theta$ = 3D representation 的参数（如 NeRF 的 MLP weights 或 3D Gaussian Splatting 的 parameters）
- $\hat{\epsilon}_\phi$ = pretrained 2D Diffusion Model 的 noise prediction network
- $z_t$ = 在 timestep $t$ 对 rendered image 加噪后的 latent
- $y$ = text prompt (conditioning signal)
- $w(t)$ = weighting function，控制不同 noise level 的贡献
- $g(\theta)$ = differentiable renderer，将 3D scene render 成 2D image
- $\epsilon$ = 随机采样的 Gaussian noise

**直觉理解**：不需要 3D training data！而是利用 pretrained 2D Diffusion Model 作为 "critic"，通过 gradient descent 优化 3D representation，使其从任意视角 render 出来的 2D image 都符合 text prompt 的语义。

**方法二：3D Gaussian Splatting (3DGS) 表示**

相比 NeRF，3DGS 更适合 real-time rendering：

每个 Gaussian primitive 由以下参数定义：
- $\mu \in \mathbb{R}^3$ — 中心位置 (mean position)
- $\Sigma \in \mathbb{R}^{3 \times 3}$ — 3D covariance matrix（用 scale vector $s$ 和 rotation quaternion $q$ 参数化）
- $\alpha \in [0,1]$ — opacity
- $c$ — Spherical Harmonics (SH) coefficients 表示的 view-dependent color

Rendering 公式 (alpha compositing)：

$$C(p) = \sum_{i \in \mathcal{N}} c_i \cdot \alpha_i \cdot \prod_{j=1}^{i-1}(1 - \alpha_j)$$

其中 $C(p)$ 是 pixel $p$ 的最终颜色，$\mathcal{N}$ 是影响该 pixel 的 Gaussians（按深度排序）。

**方法三：直接 3D Latent Diffusion Transformer**

如 Direct3D 等方法，训练一个直接在 3D latent space 中做 diffusion 的 Transformer：

$$p_\theta(x_0) = \int p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} | x_t) \, dx_{1:T}$$

其中 $x_0$ 是 3D latent representation，$x_T$ 是 pure noise，$p_\theta$ 是 learned denoising distribution。

> Seele 的 blog 中提到使用了 **Diffusion Models, GANs, Transformers** 等组合来生成 game assets。
>
> 参考: [seeles.ai/resources/blogs/ai-asset-generator](https://www.seeles.ai/resources/blogs/ai-asset-generator)

#### (B) Game Logic / Code Generation

Seele 使用所谓的 **"Vibe Coding"** 概念：

- 用户用 natural language 描述游戏逻辑（如 "player can double jump and collect coins"）
- 系统通过 **Code-generating LLM**（类似 GPT-4 / Claude 级别的模型）自动生成：
  - **C# scripts**（for Unity export）
  - **JavaScript / TypeScript**（for Three.js WebGL export）
- 包含 physics, collision detection, UI layout, scoring systems 等

#### (C) Audio Generation

- 自动生成 background music 和 sound effects
- 很可能使用类似 MusicGen / AudioLDM 的 text-to-audio model

#### (D) Scene Assembly

- 将所有 generated assets (3D models, textures, animations) 组装到 unified scene graph 中
- 自动处理 **PBR (Physically Based Rendering)** materials：
  - Albedo map
  - Normal map
  - Roughness/Metallic map
  - Ambient Occlusion

---

## 三、核心特性与产品差异化

### 3.1 与竞品对比

| 特性 | **Seele AI** | **Rosebud AI** | **Scenario** | **Game Worlds** |
|------|-------------|---------------|-------------|----------------|
| **2D Game 生成** | ✅ | ✅ | ❌ | ✅ |
| **3D Game 生成** | ✅ (核心优势) | ❌ (仅 web 2D) | ❌ | 有限 |
| **Text-to-Game** | ✅ | ✅ | ❌ | ✅ |
| **Export Unity** | ✅ (完整 .unitypackage) | ❌ | ❌ | ❌ |
| **Export Three.js/WebGL** | ✅ | ✅ (browser only) | ❌ | ✅ |
| **Multiplayer-Ready** | ✅ | ❌ | ❌ | 有限 |
| **Mobile "Pinch" UI** | ✅ (独创) | ❌ | ❌ | ❌ |
| **3D Asset 生成** | ✅ (内置) | 需外部工具 | ✅ (2D only) | 有限 |
| **Prototype 速度** | **2-10 min** | 5-15 min | N/A | 10-20 min |

> Seele 的**关键差异化**在于它是唯一同时支持 **2D 和 3D** 并且可以 **Export 到 Unity** 的 AI Game Maker。
>
> 参考: [Slashdot Comparison](https://slashdot.org/software/comparison/Rosebud.ai-vs-SEELE-AI/) | [Reddit Review](https://www.reddit.com/r/VibeCodeDevs/comments/1qw6k54/i_tried_every_ai_vibecoding_platform_in_2026/)

### 3.2 开发效率对比数据

根据 Seele 官方 blog 提供的数据：

| 任务 | **AI-Assisted (Seele)** | **Traditional** |
|------|------------------------|----------------|
| Prototype Creation | **2-10 min** | 40+ hours |
| 3D Asset Generation | **秒级 ~ 分钟级** | 数小时 ~ 数天 |
| Complete Playable Demo | **< 30 min** | 数周 ~ 数月 |

---

## 四、商业模式与定价

| Plan | 价格 | 特性 |
|------|------|------|
| **Free** | $0 | 基础功能，有限 generation credits |
| **Pro** | **$16+/month** | 商业使用许可，更多 credits，高优先级 |
| **Enterprise** | 定制 | 批量 API 访问，专属支持 |
| **7-Day Free Trial** | 免费 | Pro 功能试用，随时取消 |

> 参考: [seeles.ai/upgrade](https://www.seeles.ai/upgrade)

---

## 五、第一性原理分析：为什么 Seele 能做到 Text-to-3D-Game？

从**第一性原理**拆解，Text-to-3D-Game 本质上需要解决 **四个子问题**：

### 问题 1：从 Language 到 Structured Representation
- **核心技术**: LLM + Structured Output (JSON Scene Graph)
- **直觉**: 就像 compiler 把 high-level language 编译成 AST (Abstract Syntax Tree)，这里是把 natural language 编译成 **Game Scene Description**

### 问题 2：从 Description 到 3D Geometry
- **核心技术**: Text-to-3D Diffusion Models
- **直觉**: 2D Diffusion Models 已经能生成照片级 images。通过 SDS 或 direct 3D latent diffusion，把 2D 的 "审美能力" 蒸馏到 3D representations 中

### 问题 3：从 Static Assets 到 Interactive Logic
- **核心技术**: Code-Generating LLM
- **直觉**: LLM 已经能写出高质量代码。Game Logic 本质上是一组 **event-driven state machines**，对 LLM 来说是非常 structured 且 well-defined 的任务

### 问题 4：从 Components 到 Unified Playable Experience
- **核心技术**: Game Engine Integration Layer (Unity / Three.js runtime)
- **直觉**: 这是 **engineering problem** 而非 **research problem** — 把 AI-generated outputs 映射到现有 game engine 的 data formats

**Seele 的核心 insight**：这四个子问题在 2024-2025 年都已经有了 individually mature 的解决方案（LLMs, Diffusion Models, 3DGS, Game Engines）。Seele 的价值在于 **orchestration** — 把它们 seamlessly 组合成 end-to-end pipeline。

---

## 六、融资与市场信号

- **2025 年 5 月** Bloomberg 报道 Seele 正在寻求新一轮融资，估值 **数亿美元**
- **Baidu** 作为 strategic investor 的意义：可以接入百度的 ERNIE LLM 能力、百度 Cloud GPU infra
- 这反映了**中国 AI Gaming Sector** 的整体崛起趋势

> 参考: [Bloomberg](https://www.bloomberg.com/news/articles/2025-05-28/baidu-backed-ai-3d-game-creator-seele-seeks-new-funding) | [OpenTools](https://opentools.ai/news/seele-inc-eyes-fresh-funding-to-revolutionize-ai-3d-game-design)

---

## 七、潜在风险与挑战

1. **生成质量天花板**: Text-to-3D 目前的质量与 AAA Game 的手工资产仍有巨大差距（topology 不干净、texture 有 artifacts）
2. **Game Logic 复杂性**: 简单 game 可以自动生成，但 RPG 级别的复杂逻辑仍然需要 human intervention
3. **IP / Copyright 风险**: AI-generated assets 的版权归属在全球范围内仍有法律不确定性
4. **竞争加剧**: Epic Games (Unreal + AI), Unity (AI Muse), Roblox (AI Studio) 等巨头也在布局
5. **变现路径**: $16/month 的 Pro plan 在 indie developer 市场的 willingness-to-pay 需要验证

---

## 八、联想与延伸

- **名字来源**: "Seele" 在 German 语中意为 "灵魂 (Soul)"，同时也是 *Neon Genesis Evangelion (新世纪福音战士)* 中的神秘组织名称——这暗示了创始团队可能有 anime/gaming culture 背景
- **技术趋势**: Seele 代表了 **"AI-native Game Engine"** 的方向——未来的 Game Engine 不再是 Unity/Unreal 这样需要手动操作的工具，而是 **intent-driven**（用自然语言描述意图，AI 自动构建）
- **类比**: 就像 **Canva** 之于 **Photoshop**，Seele 之于 **Unity/Unreal** — 极大降低创作门槛，但牺牲了部分精细控制

---

**总结**: Seele AI 是一家位于深圳、由百度投资的 AI Gaming Startup，其核心产品是一个 **Text-to-Playable-3D-Game** 的 no-code platform。它将 LLM (game logic), Diffusion Models (3D assets), 以及 Audio Generation 整合成 end-to-end pipeline，支持 export 到 Unity 和 Three.js。目前估值数亿美元级别，代表了 **AI-native Game Development** 的前沿方向。