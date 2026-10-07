---
source_pdf: VideoGen-of-Thought A Collaborative Framework for Multi-Shot Video Generation.pdf
paper_sha256: c8440f4731de060588deb8882d9486a3c416fc432d61fde1372e49c9d61806b6
processed_at: '2026-08-13T00:57:11-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VGoT 用人话讲

## 一句话先说清楚

你想让 AI 生成一部短片, 现在的 video model 画 5 秒钟没问题, 但你说"给我生成一个 30 shot 的 Mary 一生故事", 它就废了。原因是 model 不知道什么叫"故事结构", 它只会画一个连续的长镜头, 画着画着人就变样了, 逻辑也飞了。

VGoT 干的事情特别简单: **别让一个 model 啥都干, 拆成 4 步, 每步用一个专门的小模型, 像 pipeline 一样串起来**。就像拍电影不是一个人又写剧本又当演员又剪片子, 而是导演、编剧、摄影、后期各司其职。

参考: [VGoT Project Page](https://cheliosoops.github.io/VGoT/)

---

## 痛点到底在哪儿

你拿 Sora、CogVideoX、VideoCrafter2 这种 SOTA model 去生成 multi-shot, 会遇到三个坑:

**第一个坑**: 现在的 video model 训练时看的都是 single clip, 就是"一句话配 5 秒视频"这种数据。你让它画 30 个 shot, 它根本不知道 shot 和 shot 之间该咋过渡, 画出来就是 30 个无关的片段拼在一起。

**第二个坑**: 角色 identity 漂移。Mary 在 shot 1 是金发蓝眼, shot 5 变成黑发棕眼, shot 10 连脸型都变了。这是 diffusion model 的 fundamental 问题 — 它每次 sample 都是独立随机的, 没有任何机制保证"这是同一个人"。

**第三个坑**: 叙事逻辑。你说"Mary 从小到老", model 不会自动 plan 出 child Mary 怎样、teenage Mary 怎样、old Mary 怎样。它只会根据你给的 prompt 闷头画, 画到哪算哪。

scaling up model 只能解决"画得更精细", 解决不了"画得有故事"。这是 paper 的核心 motivation。

参考: [Sora Technical Report](https://openai.com/sora) | [CogVideoX](https://arxiv.org/abs/2408.06072)

---

## 4 个 Module 到底在干啥

### Module 1: Script Module — LLM 当编剧

你想, 拍电影第一步是啥? 写剧本。VGoT 就是这么干的。

用户给一句话: *"30 shots, describe Mary's life from birth to death"*

然后 GPT-4o 把这一句话展开成 30 个 short description: 
- $s_1$: "Baby Mary's first cry in a hospital room"
- $s_2$: "Toddler Mary taking first steps in a garden"
- ...

接着, 每个 $s_i$ 再被扩展成 5 个 domain 的 detailed prompt:

$$p_i = \mathcal{M}_{\text{LLM}}(s_i, i)$$

这 5 个 domain 是: character ($p_{cha}$), background ($p_b$), relation ($p_r$), camera ($p_{cam}$), HDR lighting ($p_h$)。

举例 shot 1 的完整 prompt:
- **Character**: Baby Mary, newborn, wrapped in white blanket
- **Background**: Hospital room, soft medical lighting
- **Relation**: Mary held by mother, doctor standing nearby
- **Camera**: Close-up, eye-level
- **HDR**: Soft warm light from window, cool clinical light overhead

这就是电影的 shot list。LLM 是编剧, 把用户的 logline 变成可执行的 shooting script。

**为啥是 5 个 domain 不是 3 个或 10 个?** 我觉得这是 author 对电影语言的 discretization。电影学院教镜头分析, 就是 character + setting + relation + camera + lighting 这五件事。author 借用了这个 prior。

参考: [GPT-4o](https://arxiv.org/abs/2303.08774) | [Chain-of-Thought](https://arxiv.org/abs/2201.11903)

---

### Module 2: Keyframe Module — IP-Adapter 当摄影师

剧本有了, 下一步是画 keyframe (每个 shot 的代表画面)。这里有个大问题: 怎么保证 Mary 在不同 shot 里长得一样?

VGoT 的 solution 是 **Avatar-driven IP-Adapter**, 这个 idea 我觉得很 elegant:

**Step A**: LLM 在写剧本的时候, 已经预先生成了多个 avatar 的 description, 比如:
- Child Mary (age 5)
- Teenage Mary (age 16)
- Mid-aged Mary (age 35)
- Elder Mary (age 60)
- Old Mary (age 80)

每个 avatar 都用 5-domain prompt 描述。

**Step B**: 用 Kolors (text-to-image model) 给每个 avatar 画一张 portrait:
$$I_a = \mathcal{M}_I(P_a)$$

- $I_a$: avatar image set
- $P_a$: avatar prompt set

**Step C**: 每个 shot 根据 time stage 选对应的 avatar image, 用 CLIP vision encoder 提取 image embedding:
$$e_j^I = \text{CLIP}_{\text{vision}}(I_{a,j})$$

**Step D**: 用 IP-Adapter 把 image embedding 注入到 text-to-image diffusion 的 cross-attention 里, 生成 keyframe:
$$I_i = \mathcal{M}_I(e_i^T, e_j^I)$$

- $e_i^T$: shot $i$ 的 text embedding
- $e_j^I$: 对应 avatar 的 image embedding

**IP-Adapter 的核心机制**:

原始 cross-attention:
$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

- $Q$: from noisy latent (query, 当前要生成的图像特征)
- $K, V$: from text embedding (key/value, 文本条件)
- $d_k$: key 的维度, $\sqrt{d_k}$ 是 scaling factor

IP-Adapter 改成:
$$\text{Attn}_{\text{new}} = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V + \lambda \cdot \text{softmax}\left(\frac{Q(K')^\top}{\sqrt{d_k}}\right)V'$$

- $K', V'$: from image embedding (avatar 的 identity 特征)
- $\lambda$: 权重, 控制 identity 强度 (通常 0.5-1.0)

直观理解: 第一个 attention term 让 text 控制"干什么", 第二个 term 让 image 控制"长什么样"。两个 signal 叠加, 既匹配剧本又保持 identity。

**为啥不直接用前一个 shot 的 keyframe 做 image condition?** 因为 Mary 是 time-varying 的。如果用 shot 1 的 baby Mary 当 reference, shot 30 就会画成一个老 baby, 悖论。所以 author 让 LLM 先把"人生阶段"这个 semantic 信息显式 plan 出来, 每个阶段一个 avatar, 这叫 **semantic-aware identity management**。

这个 design 我觉得是 paper 最聪明的点之一。

参考: [IP-Adapter](https://arxiv.org/abs/2308.06721) | [Kolors](https://kolors.kuaishou.com/) | [CLIP](https://arxiv.org/abs/2103.00020)

---

### Module 3: Shot-Level Video Module — DynamiCrafter 当摄像师

每个 shot 的 keyframe 画好了, 下一步是让画面动起来。VGoT 用 DynamiCrafter, 一个 image+text conditioned video diffusion model。

公式:
$$Z_i = \mathcal{M}_V(e_i^T, e_i^I, \epsilon)$$

- $Z_i \in \mathbb{R}^{f \times h \times w \times d}$: shot $i$ 的 latent code, $f$ 帧
- $\mathcal{M}_V$: DynamiCrafter
- $e_i^T$: text embedding
- $e_i^I$: keyframe image embedding  
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$: 初始 Gaussian noise

这里有个 paper 里非常 subtle 但很重要的 finding:

**Asymmetric Prompt Design**: keyframe 生成用 detailed 5-domain prompt $p_i$, video 生成用 short prompt $s_i$。

为啥? author 在 4.3 节直接说: video model 训练时看到的 caption 都很简单, 你给它一个 5-domain 长 prompt, 它反而"被约束住", motion 变得不自然、不 dynamic。给它一个短句子反而能发挥得更好。

这个 observation 其实揭示了一个重要现象: **text-to-video model 的训练数据分布和 text-to-image model 不一样**。image model (LAION-5B 之类) 见过很多 detailed caption, video model (WebVid 之类) 见的多是"YouTube 标题"那种 short caption。

所以 VGoT 做了一个 trick: **keyframe 用 rich prompt 拿到信息密度高的静态画面, video 用 short prompt 让 motion 自由发挥**。这是个很 Karpathy-style 的 engineering insight — 不强求一个 model 干所有事, 而是匹配每个 model 的训练分布。

参考: [DynamiCrafter](https://arxiv.org/abs/2310.12190) | [Latent Diffusion](https://arxiv.org/abs/2112.10752)

---

### Module 4: Smooth Module — FIFO Reset Boundary 当剪辑师

这是 paper 最 technical 的部分, 但 intuition 很清楚。

**问题**: shot 1 是 baby Mary 在医院, shot 2 是 toddler Mary 在花园。直接把两个 latent 拼起来, 画面会有 visual jump — 角色突然变样、背景突然切换, 没有任何过渡。

VGoT 的 solution: 借鉴 FIFO-Diffusion 的 queue 思路, 在 shot 边界插入一段"reset noise"。

**FIFO-Diffusion 背景**: 

FIFO 的核心 idea 是把 "长时间视频" 转成 "latent queue 的滚动更新"。维护一个 queue $Q_k = \{z^f\}_{f_k}^{f_k + n}$, queue 里不同 frame 处于不同 noise level (前面 noise 多, 后面 noise 少), 用 DDIM sampler 逐步更新:

$$Q_k \gets \Phi(Q_k, \tau_k, c; \epsilon_\theta)$$

- $Q_k$: latent queue
- $\Phi$: DDIM sampler (确定性采样)
- $\tau_k$: timestep (当前 noise level)
- $c$: condition (text/image)
- $\epsilon_\theta$: noise prediction network

这相当于一个 waterfall: 高 noise latent 从前面进, 低 noise (干净) latent 从后面出, 源源不断生成 infinite video。

**VGoT 的改造**:

在 shot $i$ 和 shot $i+1$ 之间, 设置一个 reset boundary:
- Boundary 长度 = $k$ 帧 (一个 shot 的 frame 数)
- Boundary 内的 latent **重新 sample 成 pure noise** $\epsilon \sim \mathcal{N}(0, \mathbf{I})$
- 但 condition 是 **transition condition** — 包含前 shot 的 condition 残留 + 后 shot 的 condition 接管

paper 原文: "Each reset boundary contains the conditional embedding $e_i^T$ and $e_i^I$, and a series of noise $\epsilon$ set with time step scheduler $\beta_i$."

**为什么这能 smooth?**

你想, FIFO queue 里前段是高 noise, 后段是低 noise。VGoT 在 boundary 处插入一段新的高 noise latent, 但用的是 transition context (前 shot 的 embedding)。这段高 noise latent 在 queue 中被 denoise 的过程中, 后 shot 的 condition 会逐渐接管, 两个 shot 的 visual feature 就在这段 boundary 里 **blend** 了, 形成自然过渡。

这就是用 **noise-level manipulation** 实现 transition, 不需要额外训练一个 transition model, 完全 training-free。

**人话总结**: 想象你在剪辑视频, 不要硬切, 而是在两个片段之间加一段"渐变帧", 这段帧的内容一半像前一个 shot 一半像后一个 shot, 自然就过渡过去了。VGoT 是在 latent space 干这件事, 用 noise level 控制混合比例。

参考: [FIFO-Diffusion](https://arxiv.org/abs/2405.11473) | [DDIM](https://arxiv.org/abs/2010.02502)

---

## 实验结果说人话

### Quantitative (Table 1)

VGoT 在 Cross-Shot Face Consistency 上 **0.2688**, 第二名 VideoCrafter2 只有 **0.0686**, 高了一个数量级。这就是 IP-Adapter avatar 机制的效果 — baseline 都没有 identity preservation, VGoT 有。

但 VGoT 的 CLIP Score 反而比 baseline 低一点 (e.g., character domain 0.4086 vs VideoCrafter1 0.4365)。author 解释: IP-Adapter 强行注入 identity, 生成内容偏离 text prompt, 但 narrative depth 更高。这是 **prompt fidelity vs narrative richness** 的 trade-off。

### Human Evaluation (Table 2)

10 个用户评 50 个 video, VGoT 在 Cross-Shot Consistency 上 **66.67%** 用户选 "Good", baseline 最好的 CogVideo 只有 23.17%。用户感知到的"故事连贯性"碾压 baseline。

### Ablation (Table 3)

| Config | FC (Cross) | SC (Cross) | PSNR |
|---|---|---|---|
| w/o EP w/o IP | 0.113 | 0.365 | 24.33 |
| w EP w/o IP | 0.117 | 0.366 | 24.33 |
| w/o EP w IP | 0.329 | 0.419 | 23.92 |
| **Full** | 0.274 | 0.386 | **25.79** |

**两个关键 insight**:

1. **IP 是 cross-shot consistency 的核心**: 加 IP 后 FC 从 0.11 跳到 0.33, 3 倍提升。
2. **EP 让 consistency 略降但 narrative 丰富**: w/o EP 时 FC 0.329, Full Model 0.274。这是因为 EP 让每个 shot 内容更多样, identity consistency 稍微牺牲, 换取故事多样性。

这就是 paper 里那个 trade-off: **consistency vs narrative richness**。Full Model 选择了一个平衡点。

---

## 我 Karpathy 视角的 Intuition

### 1. VGoT 的哲学是 Modular System, 不是 End-to-End

Sora 路线: 大模型 + 大数据, 让 model 自己学会 multi-shot。
VGoT 路线: 小模型 + explicit structure + planning, 用 system design 实现 multi-shot。

这两条路很像 LLM 早期的争论: end-to-end neural (GPT 路线) vs modular symbolic (老 AI 路线)。但现在 LLM Agent 又把 modular 思路带回来了 — ReAct, Toolformer, Chain-of-Thought 都是 modular。

VGoT 之于 video, 就像 ReAct 之于 LLM: 把 generation 拆成 planning + execution pipeline。

### 2. 两个最聪明的 Engineering Trick

**Trick 1: Asymmetric Prompt** — keyframe 用长 prompt, video 用短 prompt。这个 insight 来自对 model 训练数据分布的理解, 很有 engineering taste。

**Trick 2: FIFO Reset Boundary** — 用 noise level 实现 cross-shot transition。不训练新 model, 只用 noise manipulation, 在 latent space 做 blending。这是 diffusion model 特性的一种 "abuse", 但 abuse 得很漂亮。

### 3. Limitation 和未来方向

Paper Appendix D 提到: 每个 shot 只能有一个 IP embedding, 多角色场景搞不定。这是 immediate next step。

另外一个 gap 是 evaluation: 现有 metric (CLIP, FC, SC) 都不能 capture narrative coherence。我们需要一个 "narrative metric", 评估视频的故事连贯性、角色发展弧、情节逻辑性。这个 metric 一旦有, multi-shot video generation 领域会迎来一次 leap。

### 4. VGoT 是 Video Agent 的雏形

往大了想, VGoT 其实是一个 video agent:
- LLM = planner (规划 shot list)
- T2I + IP-Adapter = perception + memory (生成 keyframe, 维持 identity)
- Video model = actor (执行 motion)
- Smooth module = orchestrator (协调 shot 间过渡)

这个架构可以继续扩展: 加 retrieval (查 reference video), 加 critique (LLM 评估生成结果), 加 retry (失败重生成)。这就是个完整的 agentic video generation pipeline。

参考: [ReAct](https://arxiv.org/abs/2210.03629) | [Toolformer](https://arxiv.org/abs/2302.04761)

---

## 最后总结

VGoT 的价值在两点:

1. **证明 multi-shot video generation 可以 decompose**, 不一定要 end-to-end 训一个大模型。这是一个重要的 paradigm shift, 降低门槛, 让更多人能玩 multi-shot video。

2. **两个 engineering trick (asymmetric prompt + FIFO reset boundary) 可以独立迁移**到任何 video generation pipeline, 不局限于这个 paper 的具体实现。

如果你想自己做 multi-shot video, 不用复现 VGoT 全部, 但 borrow 这两个 trick 就够你搞一个能用的 pipeline 出来。

这就是 paper 的核心, 用人话讲完了。

参考链接汇总:
- [VGoT Project Page](https://cheliosoops.github.io/VGoT/)
- [IP-Adapter Paper](https://arxiv.org/abs/2308.06721)
- [DynamiCrafter](https://arxiv.org/abs/2310.12190)
- [FIFO-Diffusion](https://arxiv.org/abs/2405.11473)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Kolors Model](https://kolors.kuaishou.com/)
- [GPT-4o Technical Report](https://arxiv.org/abs/2303.08774)
- [DDIM Sampler](https://arxiv.org/abs/2010.02502)
- [CogVideoX](https://arxiv.org/abs/2408.06072)
- [ReAct: Reasoning + Acting](https://arxiv.org/abs/2210.03629)

---

# VideoGen-of-Thought (VGoT): Multi-Shot Video Generation 的 Modular 化思路

## 1. Motivation: 为什么 Single-Shot Scaling 不够用

当前 video generation 领域 (Sora, Kling, Veo, CogVideoX 等) 大量努力都投入到 scale up model + 扩大 latent shape + 增加帧数上, 这条路线本质上是做 **long single-shot**。但 paper 的核心 insight 是: **multi-shot video 的难点不在于"画得久", 而在于"画得有叙事逻辑"**。

三个 key challenges:
- **Multi-Shot**: minute-level video 包含多个 shot
- **Reasonability**: narrative 和 storyline 的逻辑连贯性
- **Consistency**: 跨 shot 的 temporal + identity consistency

这三者本质上是一个 **compositional + structured** 的问题, 而不是一个 pure generation problem。所以作者选择 **modular decomposition** 而不是 **monolithic scaling**, 这是整个 paper 的哲学内核, 类似 Chain-of-Thought 在 LLM reasoning 中的应用。

参考: [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) | [FIFO-Diffusion](https://arxiv.org/abs/2405.11473)

---

## 2. 整体架构: 4 个 Collaborative Modules

VGoT 把 multi-shot video 生成切成 4 个 module, 每 module 专责一件事, 是一个 **training-free** 的 pipeline:

```
User Input (one-sentence, N shots)
        ↓
[1] Script Module (LLM) ─────→ {p_i}^N_{i=0}  (5-domain prompts)
        ↓
[2] Keyframe Module (T2I + IP-Adapter) ─────→ {I_i}^N_{i=0}  (consistent keyframes)
        ↓
[3] Shot-Level Video Module (DynamiCrafter) ──→ {Z_i}^N_{i=0}  (latent per shot)
        ↓
[4] Smooth Module (FIFO-like reset boundary) ──→ Final video V
```

关键设计哲学: **decouple narrative planning, visual identity, motion generation, and temporal stitching**, 每个 module 都用一个已经训练好的 SOTA model, 不需要 joint training。

---

## 3. Module 1: Script Generation — Cinematic Prompt 工程

### 3.1 设计直觉

电影剧本的工业化流程是: logline → outline → treatment → screenplay → shot list。VGoT 借鉴这个 pipeline, 用 LLM 把用户一句话扩展为结构化的 shot-by-shot script。

### 3.2 公式

$$p_i(p_{cha}, p_b, p_r, p_{cam}, p_h) = \mathcal{M}_{\text{LLM}}(s_i, i)$$

变量解释:
- $s_i$: 第 $i$ 个 shot 的 short description (e.g., "Mike's Discovery: Mike examines an ancient map...")
- $i$: shot index, 用于 LLM 维持 temporal context
- $p_{cha}$: character description (谁)
- $p_b$: background description (在哪)
- $p_r$: relation (谁和谁、谁和物体关系)
- $p_{cam}$: camera pose (镜头语言: medium shot, close-up, dolly in...)
- $p_h$: HDR lighting description (光照氛围)

### 3.3 为什么是 5 个 domain?

这是 paper 中一个 subtle 但 important 的设计。CLIP score 在 Table 1 上显示, 这 5 个 domain 的 prompt 让生成结果在 character/background/relation/camera/lighting 5 个维度都有可控信号。**直接 reason 是: text-to-video diffusion model 训练时往往只看到 coarse caption, 给它 structured prompt 反而会让 motion 变弱** (paper 4.3 提到这个 finding)。所以 author 做了一个 **asymmetric design**: 
- Keyframe 用 detailed 5-domain prompt $p_i$ → 静态画面信息密度高
- Video generation 用 short prompt $s_i$ → motion dynamics 不被 over-spec 约束

这个 trade-off 很有意思, 体现 author 对现有 video diffusion model 训练数据分布的理解。

### 3.4 Algorithm 1 分析

```
for each s_i in S':
    p_i = M_LLM(s_i, i)  # 当前 shot + 上下文
    P.append(p_i)
```

iterative generation, LLM conditioned on current $s_i$ + previous $p_{i-1}$ (隐式 context), 维持 narrative coherence。这里其实可以更 explicit 地把 $p_{<i}$ 全部 feed 进去, 但 author 选择了简单的 Markov-like 方式, 可能是 cost 考虑。

参考: [GPT-4o Technical Report](https://arxiv.org/abs/2303.08774)

---

## 4. Module 2: Keyframe Generation — IP-Adapter 保 Identity

### 4.1 核心问题

multi-shot video 最痛的失败 mode 是: 同一个角色在 shot 1 是金发, shot 5 变成黑发, shot 10 长相都变了。这是 single-shot diffusion model 的 fundamental 局限 — 它不知道"这是同一个人"。

### 4.2 解决方案: Avatar-Driven IP Embedding

**Step 1**: LLM 基于 story $S'$ 生成 avatar descriptions $P_a = [P_{a1}, P_{a2}, ...]$。例如 Mary's life story, 生成 5 个 avatar: Child Mary, Teenager Mary, Mid-aged Mary, Elder-aged Mary, Old Mary。每个 avatar 都用 5-domain prompt 描述。

**Step 2**: 用 text-to-image model $\mathcal{M}_I$ (Kolors) 生成 avatar portraits $I_a = [\mathcal{M}_I(P_{a1}), \mathcal{M}_I(P_{a2}), ...]$

**Step 3**: 用 CLIP vision encoder 提取 image embedding $e_j^I$

**Step 4**: 用 IP-Adapter 注入到 cross-attention:

$$I_i = \mathcal{M}_I(e_i^T, e_j^I)$$

变量:
- $e_i^T = E_{\text{text}}(p_i)$: 第 $i$ shot 的 text embedding (from CLIP text encoder)
- $e_j^I$: 第 $j$ 个 avatar 的 image embedding (from CLIP vision encoder)
- IP-Adapter 的 mechanism: 在 cross-attention 中 decouple $K, V$, 引入额外的 $K', V'$ from image embedding

### 4.3 IP-Adapter Cross-Attention 详解

原 cross-attention (来自 Stable Diffusion):
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

其中 $Q$ from noisy latent embedding $e_i$, $K, V$ from text embedding $e_t$.

IP-Adapter 修改后:
$$\text{Attention}_{\text{new}} = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V + \lambda \cdot \text{Softmax}\left(\frac{Q(K')^\top}{\sqrt{d_k}}\right)V'$$

其中 $K', V'$ from image embedding $e_j^I$, $\lambda$ 是 weight (通常 0.5-1.0). 这相当于让 text prompt 控制内容/动作, 让 image embedding 控制 identity/外观。

### 4.4 直觉: 为什么 Avatar 而不是直接用前一个 shot 的 keyframe?

因为故事是 time-varying 的。Mary 从 child 到 old, 外观在变化。如果直接用前一个 shot 的 frame 做 IP, 会 forced 她永远是 child。所以 author 让 LLM 主动 plan 出不同阶段的 avatar, 然后每个 shot 选择对应阶段的 avatar embedding — 这是一种 **semantic-aware identity management**。

参考: [IP-Adapter](https://arxiv.org/abs/2308.06721) | [Kolors](https://kolors.kuaishou.com/)

---

## 5. Module 3: Shot-Level Video Generation — DynamiCrafter

### 5.1 公式

$$Z_i = \mathcal{M}_V(e_i^T, e_i^I, \epsilon)$$

变量:
- $\mathcal{M}_V$: DynamiCrafter (image+text to video diffusion model)
- $e_i^T$: text embedding (这里用 $s_i$ 而非 $p_i$, 见 4.3 节的解释)
- $e_i^I$: keyframe $I_i$ 的 image embedding
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$: initial noise
- $Z_i \in \mathbb{R}^{f \times h \times w \times d}$: latent code, $f$ 帧, $h \times w$ 空间分辨率, $d$ channels

### 5.2 为什么用 DynamiCrafter 而不是 SVD 之类?

DynamiCrafter 接受 **dual condition** (image + text), 而 SVD 主要 image-driven, 文本控制弱。VGoT 需要 keyframe 作为 visual anchor + text 作为 motion hint, DynamiCrafter 正好 fit。

### 5.3 Forward + Reverse Diffusion 回顾

Paper Section 3.1 给的 background:

**Forward**: 
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})$$

- $\beta_t$: variance schedule at timestep $t$
- 累积加噪 $T$ 步, 得到 $x_T \sim \mathcal{N}(0, \mathbf{I})$

**Reverse**:
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

- $\mu_\theta, \Sigma_\theta$: 神经网络预测的 mean 和 variance
- $\theta$: model parameters

**Loss** (simplified):
$$\mathcal{L}_{\text{unconddiff}} = \mathbb{E}_{x_0, \epsilon \sim \mathcal{N}(0, \mathbf{I}), t}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

- $\epsilon$: 真实 noise
- $\epsilon_\theta(x_t, t)$: 网络预测的 noise
- $\|\cdot\|^2$: L2 距离

### 5.4 Latent Diffusion 为什么重要

Latent Diffusion (LDM) 用 VAE 把 pixel space $x \in \mathbb{R}^{H \times W \times 3}$ 压到 latent space $z \in \mathbb{R}^{h \times w \times d}$ (通常 $h = H/8, w = W/8$)。所有 diffusion 操作在 latent 上做, 大幅降低 compute cost, 同时 cross-attention 注入 condition 也更稳定。

参考: [Latent Diffusion Models](https://arxiv.org/abs/2112.10752) | [DynamiCrafter](https://arxiv.org/abs/2310.12190)

---

## 6. Module 4: Smooth Mechanism — FIFO-like Reset Boundary

### 6.1 这是 paper 最 technical 的核心

**问题**: shot $i$ 和 shot $i+1$ 主题不同, 直接 concatenate 会有 visual jump。如果让 diffusion 跨 shot 连续 denoise, noise level 不匹配会导致 artifact。

### 6.2 FIFO-Diffusion 背景

FIFO 的核心 idea: 维护一个 latent queue $Q_k = \{z^f\}_{f_k}^{f_k + n}$, 不同 frame 在 queue 中处于不同 noise level, 通过 DDIM sampler $\Phi$ 逐步 update:

$$Q_k \gets \Phi(Q_k, \tau_k, c; \epsilon_\theta)$$

- $Q_k$: latent queue
- $\tau_k$: timestep
- $c$: condition
- $\epsilon_\theta$: noise predictor
- $\Phi$: DDIM sampler (deterministic sampling)

**FIFO 的精髓**: 把 "时间维度上的长 video" 转化为 "noise level queue 的滚动更新", 这样可以用 fixed-size model 生成 infinite-length video。

### 6.3 VGoT 的修改: Reset Boundary

VGoT 借鉴 FIFO 的 queue 思路, 但针对 **shot transition** 做了改造:

在 shot $i$ 结束、shot $i+1$ 开始的位置, 设置一个 **reset boundary**:

- Boundary 长度 = $k$ 帧 (一个 shot 的 frame 数)
- Boundary 内 latent 重新 sample $\epsilon \sim \mathcal{N}(0, \mathbf{I})$
- 但 condition embedding 用 $e_i^T$ 和 $e_i^I$ (即 shot $i$ 的 condition, 而非 shot $i+1$)

等等, 这里需要仔细读 paper。paper 说: "we consider it necessary to reset the noise $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ for $\{z^f\}_{f=k \times i}^{k \times (i+1)}$ generated with Eq 6 at shot $i$, and the length of the reset boundary should be equal to the frame number $k$ for each shot. Each reset boundary contains the conditional embedding $e_i^T$ and $e_i^I$, and a series of noise $\epsilon$ set with time step scheduler $\beta_i$."

我理解是: 在 shot $i$ 结束时, **为即将到来的 shot $i+1$ 准备一段 noise + 当前 shot 的 condition**, 让这段 latent 在 queue 中被 denoise, 同时 shot $i+1$ 的 condition 在 queue 后段才 fully take over。这样实现 soft transition, 而不是 hard cut。

### 6.4 直觉: 为什么这能 smooth?

FIFO 的本质是: 队列中前段 noise level 高, 后段 noise level 低, 像一个 waterfall, 噪声从前面进来, 干净的 frame 从后面出去。

VGoT 在 shot 边界做 reset = 在 queue 中插入一段新的 high-noise latent, 但 condition 是 transition context (前 shot 的 condition 残留 + 后 shot 的 condition 接管), 这样 denoise 出来的 boundary frames 会 **blend** 两个 shot 的 visual feature, 实现 cross-shot smoothness。

这是 paper 中最聪明的设计 — 用 noise-level manipulation 实现 transition, 不需要额外训练 transition model。

参考: [FIFO-Diffusion](https://arxiv.org/abs/2405.11473) | [DDIM](https://arxiv.org/abs/2010.02502)

---

## 7. 实验分析

### 7.1 Dataset 设置

Paper 自己用 VGoT 生成 10 个 stories, 每个 30 shots, 共 300 shots。这是因为现有 dataset 没有合适的 multi-shot story benchmark, 所以 author 自建 evaluation set。

### 7.2 Quantitative Metrics

- **CLIP Score** (5 domains): text-image alignment
- **PSNR**: pixel-level quality
- **IS (Inception Score)**: 生成质量 + 多样性
- **Face Consistency (FC)**: InsightFace 提取 face feature, 算 cosine similarity
  - Within-Shot: 同 shot 内 face 一致性
  - Cross-Shot: 跨 shot face 一致性 (key metric!)
- **Style Consistency (SC)**: VGG19 提取 style feature, 算 similarity

### 7.3 Table 1 解读

| Model | FC (Cross-Shot) | SC (Cross-Shot) |
|---|---|---|
| EasyAnimate | 0.0268 | 0.2037 |
| CogVideo | 0.0222 | 0.2069 |
| VideoCrafter1 | 0.0350 | 0.1867 |
| VideoCrafter2 | 0.0686 | 0.1798 |
| **VGoT** | **0.2688** | **0.4276** |

**VGoT 在 Cross-Shot FC 和 SC 上比 baseline 高一个数量级**, 这是 IP-Adapter avatar 机制的直接效果。

CLIP Score 上 VGoT 反而低一些 (e.g., CLIP(p_cha): VGoT 0.4086 vs VideoCrafter1 0.4365). Author 解释: **VGoT 生成更丰富多样的内容, 不严格 match prompt, 但 narrative 更深**。这其实是 IP-Adapter 把 identity 强行注入后, generation 偏离纯 text prompt 的正常现象。

### 7.4 Table 2: Human Evaluation

10 个用户评 50 个 video, 三维度: Within-Shot / Cross-Shot Consistency / Visual Quality。

VGoT 在 Cross-Shot Consistency 上 **66.67%** 用户评为 "Good", 远超 CogVideo (23.17%) 和 VideoCrafter2 (27.18%). 这印证了 quantitative 结果。

### 7.5 Table 3: Ablation Studies

| Config | CLIP avg | PSNR | IS | FC (Cross) | SC (Cross) |
|---|---|---|---|---|---|
| w/o EP w/o IP | 0.1146 | 24.33 | 7.46 | 0.1129 | 0.3650 |
| w EP w/o IP | 0.1146 | 24.33 | 7.58 | 0.1174 | 0.3663 |
| w/o EP w IP | 0.1223 | 23.92 | 7.45 | 0.3291 | 0.4186 |
| **Full Model** | 0.1111 | **25.79** | 7.52 | 0.2738 | 0.3859 |

关键 insight:
1. **IP module 是 cross-shot consistency 的核心**, w/o IP 时 FC cross 只有 0.11, w IP 时跃升到 0.33
2. **Full Model PSNR 最高 (25.79)**, 整体生成质量最佳
3. **Full Model CLIP avg 最低 (0.1111)**, 因为 EP+IP 让生成偏离 prompt, 但 narrative 更丰富
4. **w/o EP w IP 的 FC (0.3291) 比 Full (0.2738) 高**, 因为 EP 让 shot 内容更多样, 牺牲了一点 identity consistency 换取 narrative depth

这是 paper 中最有意思的 trade-off: **consistency vs narrative richness**, 二者通过 EP 和 IP 的 balance 实现。

---

## 8. 我的 Intuition & Insight

### 8.1 VGoT 的真正贡献是什么?

VGoT 不是一个新的 generative model, 而是 **一个 system-level orchestration framework**。它的贡献是: 把 multi-shot video generation 这个看似 monolithic 的难题, **证明可以 decompose 成 4 个子问题, 每个用现成 SOTA model 解决, 通过巧妙的 condition design 和 noise manipulation 实现 cross-shot coherence**。

### 8.2 对未来工作的暗示

- **Module 替换性**: VGoT 的 4 个 module 都是 plug-and-play 的。可以换更强的 T2I (e.g., Flux), 更强的 video model (e.g., Sora-class), 更强的 LLM (e.g., o1 for planning). 这是 architecture 的优雅之处。
- **Limitation: Single IP per shot**: Paper 在 Appendix D 提到这个 limitation, 多角色场景需要 multi-IP, 这是 immediate next step。
- **Evaluation gap**: 现有 metric (CLIP, FC, SC) 都不能 fully capture narrative coherence, 需要 narrative-aware metric。

### 8.3 与 Sora 等端到端 model 的对比

Sora 路线是: 大模型 + 大数据 + long context, 让 model 自己学会 multi-shot。
VGoT 路线是: small modular models + explicit structure + planning, 通过 system design 实现 multi-shot。

两条路线代表两种哲学:
- **End-to-end learning**: scale wins, structure emerges
- **Modular system**: structure is designed, scale is partitioned

VGoT 的优势: 可解释、可控、低成本 (training-free)。劣势: 上限受制于 component model 的能力。
Sora 的优势: ceiling 高, seamless。劣势: 黑盒、难控、compute prohibitive。

### 8.4 类比到 LLM Agent

VGoT 之于 video generation, 类似 ReAct/Tool-Use 之于 LLM: 把 generation 任务变成 planning + execution pipeline。这个类比 paper 中已经提到 (Chain-of-Thought), 但更深层的类比是: VGoT 是一个 **video agent**, LLM 是 planner, T2I 是 tool, video model 是 actor, smooth module 是 orchestrator。

参考: [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)

---

## 9. 总结

VGoT 的核心价值不在于 SOTA numbers, 而在于 **把 multi-shot video generation 从一个 generative problem 重新 frame 为一个 structured generation problem**。它的 4-module design 是一个 **practical recipe**, 可以 immediate apply 到现有 model stack 上, 产出比纯端到端 model 更 coherent 的 multi-shot video。

最值得学习的两个 design:
1. **Asymmetric prompt**: keyframe 用 detailed 5-domain prompt, video 用 short prompt, 利用不同 model 的训练分布特性
2. **FIFO-like reset boundary**: 用 noise-level manipulation 实现 cross-shot transition, 不需要额外训练

这两个 idea 都可以独立迁移到其他 video generation pipeline 中。

参考链接:
- [VGoT Project Page](https://cheliosoops.github.io/VGoT/)
- [VGoT Paper (arXiv)](https://arxiv.org/abs/2506.05328)
- [IP-Adapter](https://arxiv.org/abs/2308.06721)
- [DynamiCrafter](https://arxiv.org/abs/2310.12190)
- [FIFO-Diffusion](https://arxiv.org/abs/2405.11473)
- [Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [Kolors](https://kolors.kuaishou.com/)
- [GPT-4o Technical Report](https://arxiv.org/abs/2303.08774)
