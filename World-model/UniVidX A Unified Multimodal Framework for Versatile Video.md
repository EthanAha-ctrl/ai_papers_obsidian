---
source_pdf: UniVidX A Unified Multimodal Framework for Versatile Video.pdf
paper_sha256: c30dfd859ced478f62d4f412c9d260cba84c8a81f2f5c5a93a3e2ed7072a2580
processed_at: '2026-08-12T20:21:40-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话再讲一遍 UniVidX

Karpathy，前面那个版本太"论文味"了，我换个口气, 像在咖啡店白板上跟你画着聊。

---

## 这帮人到底想干嘛

一句话：**让一个已经训练好的 text-to-video 大模型, 不重训, 只加几个小模块, 就能同时干 15 种活儿**。

这 15 种活儿分两类:
- **Intrinsic 那堆**: 给你 RGB 视频, 能拆出 albedo (物体本身的颜色)、irradiance (光照)、normal (表面朝向); 反过来也行——给 albedo + normal, 能合成 RGB; 甚至只给一句文字 "一个机器人在厨房做饭", 能同时吐出 RGB + albedo + normal 四个对齐的视频
- **Alpha 那堆**: 给 RGB 视频能抠出 foreground / alpha matte / background; 反过来给文字能生成带透明通道的 RGBA 视频; 还能做 video inpainting、换背景、换前景

传统做法是每个任务训一个专用模型。你想做 normal 估计? 训一个 NormalCrafter。想做 matting? 训一个 RVM。想做 relighting? 再串一个 inverse rendering + forward rendering 的 pipeline。结果就是——每个模型只会一招, 而且 pipeline 串起来的时候各模态之间对不齐 (albedo 和 normal 可能错位几个 pixel)。

UniVidX 说：**凭啥不能一个模型全包了?** 你给我哪些模态作为输入, 我就把剩下的模态给你生成出来。输入组合可以任意——纯文字、纯视觉、文字+视觉混合都行。

Project page: https://houyuan111.github.io/UniVidX.github.io/

---

## 为啥这事难

你想啊, 一个预训练的 T2V 模型 (这里用的是 Wan2.1-T2V-14B, https://arxiv.org/abs/2503.20314), 它这辈子只见过一种输入：文字。它的 input conv layer 是按 RGB latent 的 channel 数设计的, 它的 attention 是按单流视频设计的。你突然塞给它 albedo + normal + irradiance 四路视频让它处理, 它就懵了。

最直觉的解决办法是 **channel concatenation**——把四个模态的 latent 沿 channel 维拼起来, 改一下 input conv 的 in_channels, 加几个新的 output head。Diffusion Renderer (https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/)、CtrlVDiff (https://arxiv.org/abs/2511.21129) 都是这么干的。

但问题是：**改了 input conv 就毁了 priors**。input conv 是 VDM 跟外界接触的"第一层皮", 你把它拆了重训, 等于让模型从头学怎么"看"。大佬们 (Diffusion Renderer) 这么干能成, 是因为他们有几十万视频可以训。UniVidX 这帮人手里只有 924 个室内视频 (Intrinsic) 和 484 个 matting 视频 (Alpha), 这点数据训 channel-concat 模型, 直接 collapse——Fig. 9 里那个 channel-concat variant 的输出根本就是噪声糊一片, 你能肉眼看到模型完全没学会。

所以核心矛盾是：**怎么让 VDM 同时处理多模态, 又不动它的 input/output 结构?**

---

## Trick 1: SCM —— 把 condition/target 塞进 timestep

这是我觉得最漂亮的一个 idea。

正常 flow matching 训练是这样: 你有一个 clean latent $\mathbf{x}$, 采样一个 noise $\boldsymbol{\epsilon}$, 在 timestep $t \in [0,1]$ 上线性插值:
$$\mathbf{z}_t = (1-t)\boldsymbol{\epsilon} + t\mathbf{x}$$
模型要预测 velocity $\mathbf{v} = \mathbf{x} - \boldsymbol{\epsilon}$。

UniVidX 说：我有四个模态的 latent, 训练时我随机挑几个当 target (要被生成), 剩下的当 condition (作为已知信息)。怎么实现这个"挑"的动作?

**给 condition 模态强制 $t=1$, 给 target 模态正常采 $t \in [0,1]$**。

就这么简单。$t=1$ 意味着 $\mathbf{z}_1 = \mathbf{x}$, 也就是 clean latent, 没加噪声——它就是 condition。$t<1$ 的那些是 noisy latent——它们是 target。模型看到的输入就是一组 mixed timestep 的 batch, 沿 batch 维拼起来, input 结构完全没变。

公式长这样:
$$\mathcal{L}_{\text{uni}} = \mathbb{E}_{t, \mathbf{x}^{\mathcal{T}}, \boldsymbol{\epsilon}} \left\| \mathbf{v}_\theta(\mathbf{z}_t^{\mathcal{T}} \mid \mathbf{z}_1^{C}, c_{\text{txt}}) - \mathbf{v} \right\|_2^2$$

变量解释:
- $\mathbf{z}_t^{\mathcal{T}}$: target 模态在 timestep $t$ 的 noisy latent
- $\mathbf{z}_1^{C}$: condition 模态的 clean latent (强制 $t=1$)
- $c_{\text{txt}}$: 文字 prompt
- $\mathbf{v} = \mathbf{x}^{\mathcal{T}} - \boldsymbol{\epsilon}$: ground truth velocity
- $\mathbf{v}_\theta$: 模型预测的 velocity

**Intuition**: 想象 VDM 是一个会"去噪"的厨师, 平时只见过一种食材 (文字描述对应的 RGB)。SCM 干的事是——每次训练时随机告诉他 "今天这批食材里, 这几样是你已经做好的菜 (condition, 别动), 那几样是半成品 (target, 你给我继续做)"。厨师见多了各种组合, 就学会了"不管给我什么当原料, 我都能把剩下的补齐"。这就是 omni-directional generation。

推理时就按任务来设定 partition。比如 video relighting:
- Condition: albedo + normal (原视频的) → $t=1$, clean
- Target: RGB + irradiance (新光照下的) → 从 $t=0$ 开始 denoise
- Text: "warm sunset lighting from the left"

模型就给你生成新光照下的 RGB 和 irradiance, 而 albedo/normal 保持不变。一个模型, 不用改任何架构。

参考 flow matching 原始 paper: https://arxiv.org/abs/2210.02747

---

## Trick 2: DGL —— 每个模态有自己的小脑, 且只在"干活"时才用

SCM 解决了"怎么训", 但冒出新问题: albedo、normal、irradiance 这哥几个的分布天差地别。
- albedo 在 $[0,1]$ 紧凑分布, 颜色为主
- normal 是 unit vector, 三个分量平方和为 1
- irradiance 是 HDR, 动态范围可以跨好几个数量级

如果让一个 LoRA (低秩适配器, https://arxiv.org/abs/2106.09685) 同时适配这四种分布, 它的参数会被拉扯得四分五裂——这叫 **catastrophic interference**。

DGL 给出的方案分两步:

**Step 1: 每个模态一个独立 LoRA**

$$\Delta W_k = B_k A_k, \quad B_k \in \mathbb{R}^{d \times r}, A_k \in \mathbb{R}^{r \times d}$$

- $W$: 冻住的预训练权重 (14B backbone)
- $k$: 模态编号 (RGB=1, albedo=2, ...)
- $r$: LoRA rank, 设 32
- $B_k, A_k$: 第 $k$ 个模态专属的可训练低秩矩阵

总可训练参数 385M, 是 backbone 的 ~2.7%。

四个模态四组 LoRA, 互不干涉, 各学各的分布。

**Step 2: Gating —— 该用的时候才用**

$$W_k' = W + m_k \cdot \Delta W_k$$

- $m_k = 1$: 第 $k$ 个模态是 target 时, 激活它的 LoRA
- $m_k = 0$: 第 $k$ 个模态是 condition 时, 关掉 LoRA, 只用原始 $W$

**这个 asymmetry 是点睛之笔**。为啥 condition 要关掉 LoRA?

因为 condition 是 clean 输入, 它的任务是"提供 context 给 target 参考"。VDM 原本就极擅长从 clean 视觉输入提取 semantic feature——这是它在几亿视频上学来的看家本领。你如果在 condition 上也加 LoRA, 等于硬把 LoRA 训成"既能编码 clean condition 又能生成 noisy target", 这俩任务的 gradient 方向是打架的。Ablation (Tab. 6) 验证了这点: 把 gating 关掉 (LoRA 始终 active), albedo PSNR 从 16.89 掉到 15.02, normal MAE 从 11.09 恶化到 13.01。

**Intuition**: 想象 VDM 是一个熟练的厨师长, 会做中餐。LoRA 是给厨师长配的几个小学徒, 每个学徒专攻一种菜系 (川菜、粤菜、西餐)。当某道菜是"今天要做的菜" (target), 对应学徒上灶台帮忙; 当某道菜是"已经做好的参考菜" (condition), 学徒别瞎掺和, 让厨师长自己品——因为厨师长品菜的能力是几十年的功底, 学徒上去反而添乱。

Ablation 还做了另一个对比: 用一个 shared LoRA (rank 加倍到 64 保持参数量) 而非 decouple。Fig. 10 的 attention map 显示, shared variant 的 FG/BG attention 严重泄漏, 根本分不开——即使加了 distinct RoPE positional encoding 也救不回来。这说明**参数空间本身必须物理隔离**, 光靠 positional encoding 这种"软标签"不够。

---

## Trick 3: CMSA —— 让模态之间能"看见"彼此

SCM 让四个模态沿 batch 维拼起来, DGL 让它们参数隔离。但这带来新问题: vanilla self-attention 把 batch 里每个 sample 当独立的处理, 四个模态各算各的 attention, **彼此完全不知道对方存在**。

结果就是 cross-modal misalignment。Fig. 12 里 'w/ Van.' (vanilla attention) variant 在 astronaut 那个 prompt 上, albedo 和 RGB 错位明显——astronaut 的 suit 在 albedo 上偏左, 在 RGB 上偏右。

CMSA 的修改:
$$k_{\text{shared}} = [k_1, k_2, \ldots, k_n], \quad v_{\text{shared}} = [v_1, v_2, \ldots, v_n]$$
$$\text{Attention}(q_i, k_{\text{shared}}, v_{\text{shared}}) = \text{Softmax}\left(\frac{q_i k_{\text{shared}}^\top}{\sqrt{d_k}}\right) v_{\text{shared}}$$

变量:
- $q_i, k_i, v_i$: 第 $i$ 个模态的 query/key/value
- $d_k$: key 维度, 用于 scaling 防止 softmax 饱和
- $n$: 模态数 (Intrinsic 是 4, Alpha 是 4)

**所有模态的 K/V 拼成一个 shared pool, 但每个模态用自己的 Q 去 attend 这个 pool**。

**Intuition**: 这是 "shared library, private reading list" pattern。想象一个图书馆, 四个模态都把各自的书 (K/V) 放到公共书架上。但每个模态有自己的"借书证" (Q), 从书架上取自己需要的部分。RGB 的 Q 可能主要去看 normal 的 K/V (为了知道几何形状), normal 的 Q 可能主要去看 albedo 的 K/V (为了知道表面材质)。这种共享 context 让它们天然对齐, 但各自的 Q 让它们保持自己的"视角"。

这跟 Wonder3D (https://arxiv.org/abs/2310.15008)、CAT3D (https://arxiv.org/abs/2410.06685)、ViewDiff (https://arxiv.org/abs/2407.12222) 这些 cross-domain diffusion 的思路一脉相承, 但 UniVidX 把它用在了 video 多模态对齐上。

---

## 三个 trick 怎么协同

这是我觉得这篇 paper 最值得品的地方。单独看每个 trick 都不算太新:
- SCM 本质是 conditional flow matching 的一个变体
- DGL 是 LoRA + Mixture of Experts 的简化版
- CMSA 是 cross-attention 的对称化改写

但组合起来产生了一个化学反应:

1. **SCM 让 omni-directional training 可行**: 不需要为每个 task 设计不同的 condition injection 机制, 一个 timestep 标量统一了所有
2. **DGL 让 small-data fine-tuning 不毁 priors**: condition 走原 backbone 保留 VDM 的看家本领, target 走 LoRA 学 modality-specific 生成
3. **CMSA 让 cross-modal alignment 在 generation 中就被 enforce**: 不需要后处理对齐

三个加起来, 才有那个让人惊讶的数字: **19K 训练帧达到 NormalCrafter 860K 帧的精度** (Tab. 4), **45× 数据效率**。作为 Auxiliary-Free 方法 (不需要 trimap 输入) 在 video matting 上甚至超过所有 Mask-Guided 方法 (Tab. 5)。

---

## 为啥数据效率这么夸张

作者在 Limitations 里说了一句很诚实的话: fine-tuning 过程"does not learn representations from scratch but rather steers these powerful priors toward the task-specific manifold"。

翻译成人话: VDM 已经在几亿视频上学会了"世界长什么样"——什么是物体、什么是表面、什么是光照、什么是透明。这些知识都在它的权重里。fine-tune LoRA 不是让它重新学这些概念, 而是微调它的"输出分布"——从"生成 RGB 视频"偏移到"生成 RGB + albedo + normal 视频"。因为偏移的量不大 (latent space 上的小扰动), 所以需要的数据极少。

这跟 Aghajanyan 2020 (https://arxiv.org/abs/2012.13255) 关于"intrinsic dimensionality"的观察一致——预训练模型的 fine-tuning 实际发生在一个低维子空间里, 不需要太多数据。

SCM + DGL + CMSA 三个设计共同确保了这种"manifold steering"不会因为 architectural mismatch (输入结构不对、参数干扰、模态不对齐) 而失败。任何一个 trick 缺了, small-data fine-tuning 就会 collapse——ablation 都验证了。

---

## 最让我"哦原来如此"的几个细节

### 1. 不包含 roughness/metallic

Disney BRDF 模型里有 roughness 和 metallic 两个材质参数, UniVid-Intrinsic 偏偏不用。理由: (1) ground truth 噪声大; (2) VDM 自己能从 context 推断材质——它见过几亿视频, 知道金属该怎么反射、塑料该怎么反光, 不需要你显式告诉它。

这是个很 insight 的观察: **VDM 的 priors 已经覆盖了材质属性, 显式参数化反而是冗余的, 还引入噪声**。

### 2. 不包含 depth

depth 是 macro-geometric, 不是 photometric component, 而 normal 已经 capture 了 fine local geometry。从 shading equation 角度, depth 不直接参与, normal 才参与。这种"从物理方程反推该选哪些模态"的思路很干净。

### 3. Alpha 的 VAE 处理

Alpha 本来是单 channel (灰度, 表示透明度), 但 VAE 要求 3-channel RGB 输入。作者直接把 alpha 复制三份变成三通道灰度图, 塞进 VAE。简单粗暴但有效——alpha 和 RGB 共享同一个 latent space, SCM/CMSA 天然适用。

### 4. Background 的训练目标

UniVid-Alpha 的 BG 不是简单"前景背后的原背景", 而是"假设前景从未存在过的完整场景"。VDM 的 generative inpainting 能力被用来填补前景遮挡的区域。这意味着 BG 生成质量不取决于原图背景是否完整, 而取决于 VDM 对"这种场景该长什么样"的 priors。

### 5. 玻璃失败案例的诚实

Fig. 14 里, UniVid-Intrinsic 在 claw machine 侧边的玻璃上 normal 估计正确, 但在中心玻璃罩上完全失败——normal "穿透"表面反射了内部物体。作者分析这是 InteriorVid 的分布偏置: 训练集里 peripheral 区域通常是平面墙, 中心区域是 complex object。模型学到 "中心 = 复杂几何"的 prior, 在中心玻璃上错误地应用了这个 prior。

这种 failure 很有意义——它说明 **VDM priors 既能帮你 (大部分场景), 也能害你 (分布外 corner case)**。但作者强调这是 data-dependent 而非 architectural defect, 补几个玻璃场景的数据就能解决。

类似的, UniVid-Alpha 在冰块上能正确生成折射的 BL (blended RGB), 但 alpha matte 饱和到 1.0 (完全不透明), 因为 VideoMatte240K 没有半透明物体的 alpha 标签。VDM 知道冰是透明的 (BL 里有折射), 但没学过"透明物体的 alpha 该是多少"。

---

## 几个我脑子里冒出来的联想

### 1. 跟 ControlNet (https://arxiv.org/abs/2302.05543) 的对比

ControlNet 也是给 T2I 模型加 visual condition, 但它是"加一个 side branch", 主干不动。UniVidX 不加 branch, 而是**把 condition 塞进 batch 维当 "already denoised sample"**。思路完全不同——ControlNet 是"旁路注入", SCM 是"主流参与"。SCM 的好处是天然支持 omni-directional (任意模态都可以是 condition 或 target), ControlNet 的 condition 和 target 是固定角色。

### 2. 跟 MoE (Mixture of Experts) 的类比

DGL 的 per-modality LoRA + gating 很像一个极简的 MoE: 每个模态是一个 "expert", gating 根据"谁是 target"激活对应 expert。但跟传统 MoE 不同的是, 这里 expert 数 = 模态数 (固定 4 个), 而且每个 expert 只在 target 时激活, condition 时 bypass。这种 asymmetric routing 在 MoE 文献里不常见, 但对 "preserve backbone priors" 这个目标很关键。

### 3. 跟 InstructPix2Pix (https://arxiv.org/abs/2210.09276) 的对比

InstructPix2Pix 也想做"一个模型处理多种 image editing 任务", 但它是通过 instruction text 来区分任务。UniVidX 不需要 instruction, 任务由"哪些模态是 condition 哪些是 target"隐式定义。这意味着 UniVidX 的任务空间是 $\mathcal{P}(\text{modalities}) \times \text{tasks}$ 的组合空间, 比 InstructPix2Pix 的 instruction 空间更结构化。

### 4. 未来方向

作者在 Conclusion 里说 "broader V2V settings left for future work"。我猜他们想说的是: 现在 UniVidX 只验证了 pixel-aligned 的模态 (所有模态空间分辨率相同)。如果模态不对齐呢? 比如 text→skeleton→video, skeleton 是稀疏关键点, 跟 RGB video 不是 pixel-aligned。SCM 的 timestep trick 还能用吗? CMSA 的 shared K/V 在跨分辨率下怎么处理? 这些都是 open question。

还有, 现在 UniVid-Intrinsic 和 UniVid-Alpha 是两个分开的模型, 因为没有 joint annotated data。如果未来有个 dataset 同时标了 intrinsic 和 alpha, 理论上可以一个模型处理 7+ 模态。但 14B backbone + 7 个 LoRA set 的 VRAM 会很吓人——作者提到现在已经被限制在 21 帧、480p、4 模态了。

---

## 一句话总结

**UniVidX = 用 timestep 编码 condition/target partition + 用 per-modality LoRA 隔离模态分布 + 用 shared K/V attention 对齐模态, 三招合起来让预训练 VDM 在 <1k 视频上学会 15 个 pixel-aligned 多模态任务**。

核心 insight 就一句: **别动 VDM 的 input/output 结构, 让它用最原生的方式处理所有模态, 你的小模块只负责"路由"和"对齐"**。这个思路我觉得可以迁移到很多其他 aligned multimodal domain——深度、光流、语义分割, 甚至 medical imaging 里的 CT/MRI/PET 多模态融合。只要是 pixel-aligned 的, SCM + DGL + CMSA 这套 recipe 应该都能用。

参考链接汇总:
- UniVidX: https://houyuan111.github.io/UniVidX.github.io/
- Wan2.1-T2V: https://arxiv.org/abs/2503.20314
- Flow Matching: https://arxiv.org/abs/2210.02747
- LoRA: https://arxiv.org/abs/2106.09685
- ControlNet: https://arxiv.org/abs/2302.05543
- InstructPix2Pix: https://arxiv.org/abs/2210.09276
- Wonder3D: https://arxiv.org/abs/2310.15008
- IntrinsiX: https://research.nvidia.com/labs/toronto-ai/intrinsix/
- Diffusion Renderer: https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/
- NormalCrafter: https://arxiv.org/abs/2503.03301
- RobustVideoMatting: https://arxiv.org/abs/2108.11579
- Aghajanyan 2020 (intrinsic dimensionality): https://arxiv.org/abs/2012.13255

---

# UniVidX 深度解读

Karpathy，这篇 paper 我读了三遍，挺有意思的——它本质上是把 VDM 当成一个 "universal multimodal engine"，通过三个巧妙的 architectural 设计让一个 pre-trained T2V 模型同时承担十几个 pixel-aligned 的多模态视频任务。下面我尽量 build 你的 intuition。

## 1. 高层直觉：它在解决什么问题

Pre-trained VDM（这里是 Wan2.1-T2V-14B，https://arxiv.org/abs/2503.20314）已经蕴含了大量 real-world dynamics priors。但下游任务通常要么 fine-tune 一个专用模型（NormalCrafter 只估 normal），要么串行 pipeline（Ouroboros：先 inverse 再 forward rendering）。这两种方式的问题是：(1) 模型被锁死在 fixed input-output mapping；(2) cross-modal 一致性靠后处理拼凑。

UniVidX 想做的事是：给定一组 pixel-aligned 的 modalities $\mathcal{Z} = \{z_1, z_2, \ldots, z_n\}$（比如 RGB + albedo + irradiance + normal），让模型**任意指定哪些是 condition、哪些是 target**，用一个 flow matching 目标端到端训练，推理时按 task 自由组合。这覆盖了三种 paradigm：Text→X、X→X、Text&X→X，共 15 个 task。

Project page: https://houyuan111.github.io/UniVidX.github.io/

---

## 2. 三个核心设计

### 2.1 Stochastic Condition Masking (SCM)

**核心 trick**：在训练时，每个 mini-batch 随机把 $\mathcal{Z}$ 切成两个 disjoint subset：
- Target subset $\mathcal{Z}_{\text{tgt}}$：被 corrupt 成 noisy latent，模型要预测 velocity field 去 denoise 它
- Condition subset $\mathcal{Z}_{\text{cond}}$：保持 clean，作为 condition 注入

**实现上极其优雅——通过 timestep manipulation**：

对 target subset，标准 flow matching 插值：
$$\mathbf{z}_t^{\mathcal{T}} = (1-t)\,\boldsymbol{\epsilon} + t\,\mathbf{x}^{\mathcal{T}}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$
- $t \in [0,1]$ 是 flow matching timestep
- $\mathbf{x}^{\mathcal{T}}$ 是 target modality 的 clean latent
- $\boldsymbol{\epsilon}$ 是 Gaussian noise
- $t=0$ 全噪声，$t=1$ 全 clean

对 condition subset，**强制 $t=1$**，记作 $\mathbf{z}_1^{C}$，相当于"已经在 clean state 的 latent"。

训练目标：
$$\mathcal{L}_{\text{uni}} = \mathbb{E}_{t, \mathbf{x}^{\mathcal{T}}, \boldsymbol{\epsilon}} \left\| \mathbf{v}_\theta(\mathbf{z}_t^{\mathcal{T}} \mid \mathbf{z}_1^{C}, c_{\text{txt}}) - \mathbf{v} \right\|_2^2$$
- $\mathbf{v}_\theta$ 是模型预测的 velocity field
- $\mathbf{v} = \mathbf{x}^{\mathcal{T}} - \boldsymbol{\epsilon}$ 是 ground truth velocity（flow matching 中沿 straight path 的恒定速度向量）
- $c_{\text{txt}}$ 是 text condition（T2V backbone 原生支持）

**Intuition**：这个设计的精髓在于，它把"哪个是 condition、哪个是 target"完全从 architecture 里解耦出来，全部塞到 timestep 这一个标量上。Backbone 看到的就是一个 mixed timestep batch——某些 sample 是 $t=0.3$ 的 noisy latent（要 denoise），某些 sample 是 $t=1$ 的 clean latent（直接作为 context）。模型通过反复见到各种 partitioning，学到的是 **omni-directional conditional distribution** $p(\mathcal{Z}_{\text{tgt}} \mid \mathcal{Z}_{\text{cond}}, c_{\text{txt}})$，而不是单向 mapping。

**推理时**：根据具体 task 设定 partition。比如 video relighting 任务中，albedo+normal 作为 condition（$t=1$），RGB+irradiance 作为 target（从 $t=0$ 开始 denoise），text prompt 描述目标 lighting。这就是 Text&X→X paradigm。

为什么这个比 channel concatenation 好？作者在 Sec. 4.3 做了关键 ablation：channel concatenation 需要改 input conv layer 的 channel 数，引入新的 randomly initialized parameters，破坏了 VDM 输入分布。在 <1k videos 的小数据上直接 collapse（Fig. 9 触目惊心）。而 SCM 沿 batch dimension 拼接，input/output 结构完全不变，VDM priors 完整保留。

参考 flow matching: https://arxiv.org/abs/2210.02747

### 2.2 Decoupled Gated LoRA (DGL)

SCM 解决了"怎么训"，但留下一个隐患：不同 modalities 的 latent distribution 差异巨大（albedo 在 [0,1] 紧凑分布，normal 是 unit vector，irradiance HDR 动态范围很大）。如果让一个 monolithic LoRA 同时适配所有 modalities，参数会互相干扰。

DGL 的两个设计点：

**(1) Per-modality LoRA decoupling**：
$$\Delta W_k = B_k A_k, \quad B_k \in \mathbb{R}^{d \times r}, \quad A_k \in \mathbb{R}^{r \times d}, \quad r \ll d$$
- $W \in \mathbb{R}^{d \times d}$ 是 frozen pre-trained weight
- $k$ 是 modality index
- $r$ 是 LoRA rank（paper 设 32）
- 每个 modality 有自己的 $B_k, A_k$，互不共享

总参数：385M（14B backbone 上加 LoRA，rank=32）。

**(2) Gating 机制**：
$$W_k' = W + m_k \cdot \Delta W_k$$
- $m_k \in \{0, 1\}$ 是 gate 信号
- 当 modality $k$ 是 **target**（noisy input，需要被生成）时 $m_k = 1$，LoRA active
- 当 modality $k$ 是 **condition**（clean input）时 $m_k = 0$，LoRA bypass，只用原始 $W$

**Intuition**：这个 gating 逻辑非常关键，因为它 asymmetric 地处理了 condition vs target。Condition 走原 backbone，最大化复用 VDM native encoding 能力——这是它能 <1k 数据泛化的根本原因。Target 走 LoRA-adapted path，吸收 modality-specific 的生成分布。如果对所有 input 都激活 LoRA（ablation 中的 'w/o Gating' variant，Tab. 6），albedo PSNR 从 16.89 跌到 15.02，normal MAE 从 11.09 恶化到 13.01——因为 LoRA 在 clean condition 上"加了不该加的扰动"，破坏了 VDM 提取 robust semantic feature 的能力。

**为什么不用一个 shared LoRA（'w/o Dec.' variant）**：作者做了公平 ablation（rank 加倍到 64 保持参数量一致），结果 Fig. 10 的 attention map 显示，shared variant 的 FG/BG attention 严重泄漏，根本分不开模态。作者还在 shared variant 上加了 distinct RoPE positional encoding 试图区分模态，仍然失败。这说明**参数空间本身必须 decouple**，单靠 positional encoding 不足以分离 modalities。

参考 LoRA: https://arxiv.org/abs/2106.09685

### 2.3 Cross-Modal Self-Attention (CMSA)

SCM 让 modalities 沿 batch 维拼接，DGL 让它们参数 decouple，但 vanilla self-attention 会把每个 modality 当独立 batch 处理——没有任何 information exchange。这导致 cross-modal misalignment（Fig. 12 中 'w/ Van.' variant 在 astronaut 那个 prompt 上 albedo 和 RGB 严重错位）。

CMSA 的设计：
$$k_{\text{shared}} = [k_1, k_2, \ldots, k_n], \quad v_{\text{shared}} = [v_1, v_2, \ldots, v_n]$$
$$\text{Attention}(q_i, k_{\text{shared}}, v_{\text{shared}}) = \text{Softmax}\left(\frac{q_i \, k_{\text{shared}}^\top}{\sqrt{d_k}}\right) v_{\text{shared}}$$
- $q_i, k_i, v_i$ 分别是 modality $i$ 的 query/key/value
- $d_k$ 是 key dimension，用于 scaling
- $n$ 是 modality 数量

**Intuition**：这是把每个 modality 的 attention 都"看到"所有 modalities 的 K/V，但保持 Q 是 modality-specific 的。换句话说，**所有模态共享一个 attention context pool，但每个模态从 pool 里 retrieve 的内容不同**。这个不对称设计很关键——共享 K/V 提供 cross-modal alignment 的 grounding，独立 Q 让每个 modality 保持自己的"语义视角"。

这与 cross-domain diffusion 的 prior work 思路一致（Wonder3D https://arxiv.org/abs/2310.15008, CAT3D https://arxiv.org/abs/2410.06685, ViewDiff https://arxiv.org/abs/2407.12222），但 UniVidX 把它用在 video 的多模态对齐上，并配合 SCM 实现了 omni-directional generation。

---

## 3. 两个 Instantiation

### 3.1 UniVid-Intrinsic

处理 4 个 modalities：
- $R \in \mathbb{R}^{T \times H \times W \times 3}$：RGB video
- $A \in \mathbb{R}^{T \times H \times W \times 3}$：albedo（diffuse reflectance，与光照/视角无关）
- $I \in \mathbb{R}^{T \times H \times W \times 3}$：irradiance（incoming light intensity，含 shadow）
- $N \in \mathbb{R}^{T \times H \times W \times 3}$：normal（per-pixel 表面朝向）

**关键 decision：不包含 roughness/metallic**——因为：(1) ground truth 噪声大；(2) VDM 自身能从 context 推断材质响应，生成 realistic reflection 不需要显式参数化。这个观察很 insight，说明作者对 VDM priors 的能力边界有清晰认知。

**不包含 depth**——depth 是 macro-geometric，不是直接 photometric component，而 normal 已经包含 fine local geometry。

数据集 InteriorVid：924 个室内视频（21 帧，480×640），由 Blender Cycles 渲染，OpenEXR 16-bit float 保留线性空间全动态范围。167 个 SuperhiveMarket 3D 场景 + random walk camera trajectory + randomized FOV/focal length。

### 3.2 UniVid-Alpha

处理 4 个 modalities：
- $R \in \mathbb{R}^{T \times H \times W \times 3}$：blended RGB (BL)
- $F \in \mathbb{R}^{T \times H \times W \times 3}$：foreground (FG)
- $P \in \mathbb{R}^{T \times H \times W \times 3}$：alpha matte（原本单 channel，复制 3 次适配 VAE）
- $B \in \mathbb{R}^{T \times H \times W \times 3}$：background (BG)

BG 的训练目标：恢复"前景从未存在过"的干净场景。VDM 的 generative inpainting 能力被用来填补 occluded regions，避免 "holes"。

训练数据：VideoMatte240K (https://github.com/zhanglongxin/RobustVideoMatting)，484 个视频，432×768 分辨率。Text caption 用 Qwen3-VL (https://arxiv.org/abs/2511.21631) 生成。

---

## 4. 实验亮点

### 4.1 Text→X Generation (Tab. 1)

| Task | Method | Temporal Flickering ↑ | User Study (Visual/TA/MC) |
|---|---|---|---|
| Text-to-Intrinsic | IntrinsiX | N/A (image) | 8.44 / 8.65 / 7.02 |
| | **UniVid-Intrinsic** | 0.9876-0.9885 | 9.23 / 9.04 / 9.29 |
| Text-to-RGBA | LayerDiffuse | N/A (image) | 9.12 / 8.89 / 8.61 |
| | **UniVid-Alpha** | 0.9891-0.9954 | 9.12 / 9.04 / 9.35 |

Temporal Flickering 接近 1.0，意味着时序极度稳定——这是 image-based baseline 物理上无法做到的。User study 在 text alignment (TA) 和 modality consistency (MC) 上优势尤其明显。

参考 IntrinsiX: https://research.nvidia.com/labs/toronto-ai/intrinsix/ (项目页)
参考 LayerDiffuse: https://arxiv.org/abs/2402.17113

### 4.2 Inverse/Forward Rendering (Tab. 2)

在 InteriorVid-Test 上对比 RGB↔X, Diffusion Renderer, Ouroboros, Stable Normal, Lotus, NormalCrafter：

| Metric | Best baseline | UniVid-Intrinsic |
|---|---|---|
| Albedo PSNR ↑ | 14.21 (Ouroboros) | **16.89** |
| Albedo SSIM ↑ | 0.7063 (Ouroboros) | **0.7812** |
| Normal MAE ↓ | 13.68 (Stable Normal) | **11.09** |
| Normal 11.25° ↑ | 64.13 (NormalCrafter) | **70.52** |
| Forward Rendering PSNR ↑ | 13.48 (RGB↔X) | **15.31** |

特别值得注意的是 normal 估计——UniVid-Intrinsic 居然超过了专门做 normal 的 Stable Normal / Lotus / NormalCrafter。作者把这归功于：(1) multi-modal joint training 让 normal 从 albedo/irradiance 中获得额外约束；(2) VDM priors 提供强结构先验。

参考 Ouroboros: https://arxiv.org/abs/2508.14461
参考 Diffusion Renderer: https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/
参考 NormalCrafter: https://arxiv.org/abs/2503.03301
参考 Stable Normal: https://arxiv.org/abs/2506.18344

### 4.3 Normal Estimation 数据效率 (Tab. 4)

在 Sintel benchmark (https://sintel.is.tue.mpg.de/) 上：

| Method | Training Frames | Mean ↓ | 11.25° ↑ | Rank |
|---|---|---|---|---|
| NormalCrafter | 860K | 30.7 | 23.5 | 1.0 |
| Lotus | 59K | 32.3 | 22.4 | 2.2 |
| **Ours** | **19K** | 33.5 | 21.6 | 3.1 |

UniVid-Intrinsic 用 19K 帧达到接近 NormalCrafter (860K 帧) 的精度——**45× 数据效率提升**。这是 VDM priors + multi-modal joint training 的威力。

### 4.4 Video Matting (Tab. 5)

VideoMatte benchmark 上对比 MG (Mask-Guided) 和 AF (Auxiliary-Free) 方法：

| Category | Best Method | MAD ↓ | MSE ↓ | Grad ↓ | dtSSD ↓ |
|---|---|---|---|---|---|
| MG | MatAnyone | 4.37 | 0.74 | 2.57 | 1.42 |
| AF | RVM | 5.47 | 0.78 | 2.64 | 1.61 |
| **Ours (AF)** | UniVid-Alpha | **4.24** | **0.69** | **1.86** | **1.39** |

作为 AF 方法（不需要 trimap/mask 输入），它甚至超过了所有 MG 方法——这在传统方法里几乎是不可能的。作者归因于 VDM 提供的 robust semantic segmentation capability，能从 complex background 中区分 subject。

唯一弱项是 Conn (Connectivity) 0.52——比 MG 方法的 0.26-0.31 差一些，说明在 thin structure / hole 区域细节上还有提升空间。

参考 RVM: https://arxiv.org/abs/2108.11579
参考 MatAnyone: https://arxiv.org/abs/2503.22448
参考 MaGGIe: https://arxiv.org/abs/2404.16035

### 4.5 Ablation 验证 (Tab. 6, Fig. 9-12)

四个关键 ablation：
1. **Channel-concat vs batch-concat**: channel-concat 在小数据上完全 collapse（Fig. 9），验证了"保留 VDM priors"的核心论点
2. **Decoupled vs Shared LoRA**: shared LoRA 即使 rank 加倍也无法分离 modalities（Fig. 10 attention map）
3. **Gated vs Always-on**: gating 在 condition input 上保留 native encoding，PSNR 提升 1.87dB
4. **CMSA vs Vanilla attention**: vanilla 在 cross-modal alignment 上失败（Fig. 12）

---

## 5. 我对这篇 paper 的几点 reflection

**1. "Timestep as partition" 是个 beautiful trick**。传统条件生成要么用 cross-attention 注入 condition，要么用 channel concat。SCM 把 condition/target 的区分完全压缩到一个 scalar $t$ 上，让 backbone 看到的是统一的 (noisy_or_clean_latent, t) tuple。这种统一性是 omni-directional generation 的前提。

**2. Gating 的 asymmetry**：condition 走原 backbone、target 走 LoRA-adapted path——这个 asymmetry 是数据效率的根源。如果 condition 也走 LoRA，相当于强迫 LoRA 同时学"如何编码 clean condition" 和 "如何生成 noisy target"，这两个任务在 latent space 上的 distribution shift 太大，必然互相干扰。Gating 通过 spatial routing 把这两个任务在参数空间分开。

**3. CMSA 的不对称设计**：共享 K/V 但独立 Q——这是 "shared memory, private query" pattern。从 neuroscience 角度类比，类似 hippocampus 的 shared representation + task-dependent readout。从信息论角度，K/V 是 cross-modal alignment 的 grounding，Q 是 task-specific retrieval。

**4. 关于数据效率的根源**：作者在 Limitations 里坦白说，他们的 fine-tuning "steers priors toward task-specific manifold" 而不是 from-scratch learning。这个 framing 和 Liang et al. 2025 "Diffusion Renderer" 的 observation 一致——diffusion model 在小数据上 fine-tune 时，更像是在 pre-trained manifold 附近做局部调整，不需要学习新的 representation。SCM + DGL + CMSA 三个设计共同确保了这种 "manifold steering" 不会因为 architectural mismatch 而失败。

**5. Potential limitation 的 honesty**：作者明确指出玻璃场景失败（Fig. 14）——中心玻璃罩的 normal 错误地"穿透"表面反射内部细节。这是训练集分布偏置（InteriorVid 中 peripheral 通常是平面墙，中心是 complex object），不是 architectural defect。同样的 honesty 在 UniVid-Alpha 的冰块 alpha matte 失败上——VDM 知道冰是透明的（生成 BL 时正确折射），但因为 VideoMatte240K 没有半透明 alpha 标签，无法预测 fractional alpha。这些 failure 都是 data-dependent，理论上可以通过补充 targeted data 解决。

**6. Missing piece**: 目前 UniVid-Intrinsic 和 UniVid-Alpha 是分开的两个模型，因为没有 joint annotated data。如果能有一个 dataset 同时包含 intrinsic labels 和 alpha labels，理论上可以 unify 进一个模型处理 7+ modalities。这会面临 VRAM 挑战（14B backbone + 7 LoRA sets），但是 framework 本身没有 architectural 限制。

---

## 6. 关键 references 汇总

- **UniVidX project**: https://houyuan111.github.io/UniVidX.github.io/
- **Wan2.1-T2V**: https://arxiv.org/abs/2503.20314
- **Flow Matching**: https://arxiv.org/abs/2210.02747
- **LoRA**: https://arxiv.org/abs/2106.09685
- **IntrinsiX**: https://research.nvidia.com/labs/toronto-ai/intrinsix/
- **LayerDiffuse**: https://arxiv.org/abs/2402.17113
- **Ouroboros**: https://arxiv.org/abs/2508.14461
- **Diffusion Renderer**: https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/
- **NormalCrafter**: https://arxiv.org/abs/2503.03301
- **Stable Normal**: https://arxiv.org/abs/2506.18344
- **Lotus**: https://arxiv.org/abs/2409.18124
- **RobustVideoMatting (VideoMatte240K)**: https://arxiv.org/abs/2108.11579
- **MatAnyone**: https://arxiv.org/abs/2503.22448
- **MaGGIe**: https://arxiv.org/abs/2404.16035
- **Wonder3D (cross-domain diffusion prior art)**: https://arxiv.org/abs/2310.15008
- **CAT3D**: https://arxiv.org/abs/2410.06685
- **ViewDiff**: https://arxiv.org/abs/2407.12222
- **Sintel benchmark**: https://sintel.is.tue.mpg.de/
- **VBench (Temporal Flickering metric)**: https://vchensu.github.io/vbench/
- **Qwen3-VL (captioning)**: https://arxiv.org/abs/2511.21631
- **InteriorVerse (synthetic indoor data)**: https://sinabig.github.io/2021/12/02/2021-12-02-interiorverse/

---

## 7. TL;DR for Karpathy

如果你只有 30 秒：
- **What**: 一个 unified framework，让 pre-trained T2V model 同时支持 15 个 pixel-aligned multimodal video task
- **How**: 三个 design——SCM 用 timestep 编码 condition/target partition，DGL 给每个 modality 独立 LoRA 且只在 target 时 activate，CMSA 共享 K/V 但独立 Q 实现 cross-modal alignment
- **Why it works**: 三个 design 共同确保 pre-trained VDM priors 最大化保留，fine-tuning 只是 manifold steering 而非 from-scratch learning
- **Surprising result**: 19K 训练帧达到 860K 帧的 NormalCrafter 精度；作为 AF 方法在 video matting 上超过所有 MG 方法
- **Limitation**: data-dependent failure（玻璃、半透明物体），architecturally 没有 blocker

这篇 paper 的核心 contribution 不在任何一个 trick 单独，而在三个 trick 的协同——SCM 让 omni-directional training 成为可能，DGL 让 small-data fine-tuning 不破坏 priors，CMSA 让 cross-modal consistency 在 generation 时就被 enforce 而非后处理拼凑。这个组合 recipe 很可能可以迁移到其他 aligned multimodal domain（深度/光流/语义分割等），作者在 Conclusion 里也明确提到 broader V2V settings 是 future work。
