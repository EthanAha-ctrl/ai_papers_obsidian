---
source_pdf: GenEx.pdf
paper_sha256: 5f89036ba1547be9f40504a2620fec2eae5a1997a2d3cac9e5980d689dfa5846
processed_at: '2026-08-04T14:26:07-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GenEx 用人话说

Andrej，咱换个讲法——不讲 paper 结构，讲"这玩意儿到底在搞啥、为啥这么搞、搞成了啥样、哪儿还别扭"。

---

## 一句话版本

你给它一张照片，它给你"脑补"出整个 360° 的世界，然后你可以在里头走来走去，每走一步它生成一段视频给你看前方长啥样。**本质就是把 video diffusion model 当成一个 egocentric world simulator 用。**

---

## 为什么这事有意思

人看一眼房间，脑子里就建了个 3D 模型，能想象"拐角后面是啥"、"我走过去会看到什么"。AI 想干这事一直很难。已有的几条路都有毛病：

| 路 | 代表 | 毛病 |
|---|---|---|
| 显式 3D reconstruct | [NeRF](https://arxiv.org/abs/2003.08934)、[3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) | 需要 multi-view、受 render distance 限制、不能 "boundless" 走很远 |
| Single-image 3D | [Stable Zero123](https://stability.ai/news/stable-zero123-3d-consistent-image-generation)、[SV3D](https://arxiv.org/abs/2403.12008)、[Wonderworld](https://arxiv.org/abs/2406.09394) | FoV 太窄、render distance 短、背景容易崩 |
| 纯 video generation | [Sora](https://openai.com/research/video-generation-models-as-world-simulators)、[Stable Video Diffusion](https://arxiv.org/abs/2311.15127) | 没有 physical grounding，生成的视频跟 "world" 没关系，只是好看 |
| Game engine | [UE5](https://www.unrealengine.com/)、[Unity](https://unity.com/) | 需要人工建场景，不能从一张照片自动生成 |

GenEx 想要的 sweet spot：**用 generative model（能从单图生成）+ physics engine 数据训练（有 grounding）+ panoramic 表示（能 boundless 走）+ video diffusion 当 transition（能动态演化）**。

---

## 核心机制：两步走

### 第一步：World Initialization——从单图脑补 360°

输入一张普通 RGB 照片 $i_0$ + 一句 text 描述 $l_0$（比如 "a city street at dusk"），输出一张 360° equirectangular panorama $x_0$。

底层是 [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) 加了个 [panorama LoRA](https://huggingface.co/jbilcke-hf/flux-dev-panorama-lora-2)。原始 LoRA 只吃 text：

$$
x_0 \sim p_{\text{flux}}(x \mid l_0)
$$

但这样跟你的输入照片对不上。论文把它改成 image+text conditioned：

$$
x_0 \sim p_{\theta_1}(x \mid i_0, l_0)
$$

变量解释：
- $i_0$：输入的单张 RGB image
- $l_0$：language description，引导生成的场景类型
- $x_0$：输出的 360° equirectangular panorama
- $\theta_1$：image-to-panorama generator 的参数

**人话**：原本的模型你给它一句话它生成一张全景图，但跟你给的照片没啥关系。GenEx 改成 "看这张照片 + 这句话，生成跟照片一致的全景图"。

具体怎么改的——是 ControlNet 还是 IP-Adapter 还是 ReferenceNet——论文没说。这是个工程 gap，想复现得自己试。

---

### 第二步：World Transition——往前走，生成新视角

这是核心。agent 在脑补的世界里走一步，它看到的 panorama 就要更新。

输入：上一帧的 panorama $x_{t-1}^{S}$ + action $a_t = (\alpha_t, d_t)$
输出：一段新的 panoramic video $\mathbf{x}_t = (x_t^0, x_t^1, \ldots, x_t^S)$

变量解释：
- $x_{t-1}^{S}$：上一步 video 的最后一帧 panorama（上标 $S$ 表示 sequence 的最后一帧）
- $a_t$：第 $t$ 步的 action
- $\alpha_t$：旋转角度（左右转头多少度）
- $d_t$：向前走多远
- $\mathbf{x}_t$：生成的 panoramic video，有 $S$ 帧
- $x_t^s$：第 $t$ 步 video 的第 $s$ 帧

三步 pipeline：

**Step A：Action sampling**
$$
a_t \sim \mathcal{A}
$$
从 action space 里采一个 $(\alpha_t, d_t)$。$\mathcal{A}$ 是 UE5/Unity 里定义的连续动作空间。

**Step B：Sphere rotation——把 panorama 在球面上转 $\alpha_t$ 度**
$$
{x_{t-1}^{S}}' = \mathcal{T}(x_{t-1}^{S}, \alpha_t)
$$

这是**纯几何变换，不经过神经网络**。具体在 spherical polar coordinate 上做：

$$
\mathcal{T}(u, \nu, \Delta\phi, \Delta\theta) = f_{S \to \mathcal{P}}\bigl(\mathcal{R}(f_{\mathcal{P} \to S}(u, \nu), \Delta\phi, \Delta\theta)\bigr)
$$

$$
\mathcal{R}(\phi, \theta, \Delta\phi, \Delta\theta) = (\phi + \Delta\phi \pmod{2\pi},\ \theta + \Delta\theta \pmod{\pi})
$$

变量：
- $(u, \nu)$：panorama 像素坐标，$u$ 是 column（0 到 $W-1$），$\nu$ 是 row（0 到 $H-1$）
- $(\phi, \theta)$：球面坐标，$\phi$ 是 longitude（经度，$[-\pi, \pi)$），$\theta$ 是 latitude（纬度，$[-\pi/2, \pi/2]$）
- $\Delta\phi, \Delta\theta$：旋转量
- $f_{\mathcal{P} \to S}$：pixel → spherical 的映射
- $f_{S \to \mathcal{P}}$：spherical → pixel 的映射

**人话**：panorama 本质是球面展开成 2D 图。agent "转头" 就是把球面转一下再重新展开成 2D。这事纯几何，不需要 AI 学。

**Step C：Panoramic video generation——往前走 $d_t$，生成视频**
$$
\mathbf{x}_t \sim p_{\theta_2}(\mathbf{x} \mid {x_{t-1}^{S}}', \epsilon_t), \quad \epsilon_t \sim \mathcal{N}(0, I)
$$

变量：
- ${x_{t-1}^{S}}'$：旋转后的 panorama，作为 condition
- $\epsilon_t$：Gaussian noise，注入随机性让世界 "explorable"（同一步可以生成不同的前方场景）
- $\theta_2$：panoramic video generator 参数

**这里有个我没读明白的点**：公式里没显式 condition on $d_t$。走 1 米和走 10 米，condition 一样？我猜 $d_t$ 隐式编码在 video 长度 $S$ 里——走得远 video 帧数多。但论文没明说。

---

## 关键 inductive bias：把 rotation 和 forward motion 拆开

这是我读完最大的 takeaway。

**旋转是确定性几何，forward motion 才需要 generative model 学。**

如果让 diffusion model 同时学 "转头" 和 "往前走"，它得同时学几何变换 + 场景演化，任务太难。GenEx 把转头疼用纯数学解决，diffusion 只管 "往前走看到啥"。

这个拆分带来的好处：
1. 学习难度降低——diffusion 只学一个相对简单的 forward motion 分布
2. Drift 被压住——旋转不引入 generative error，只有 forward motion 才有
3. Loop consistency 好做——走一圈回来，旋转部分完全可逆，只有 forward 部分有误差

实验里 IELC metric（走一圈回来 latent MSE < 0.1，即使 20m loop）验证了这个设计有效。

---

## Spherical-Consistency Learning (SCL)

Equirectangular panorama 有个毛病：左右边缘（$\phi = -\pi$ 和 $\phi = \pi$）在球面上是同一条经线，但展开成 2D 图后是两个边缘。普通 convolution 不知道这件事，生成的内容在接缝处不连续。

SCL（在他们[前作](https://arxiv.org/abs/2411.11844)里详细讲）就是在 loss 里加约束，让 panorama 在球面上 smooth。Table 1 显示加了 SCL 后：

| Metric | w/o SCL | w/ SCL | 提升 |
|---|---|---|---|
| FVD ↓ | 81.9 | 69.5 | -12.4 |
| PSNR ↑ | 29.4 | 30.2 | +0.8 |
| SSIM ↑ | 0.91 | 0.94 | +0.03 |

**人话**：把 panorama 想成一张纸卷成圆筒，左右边缘要粘起来。SCL 就是逼着模型在接缝处生成连续内容。

---

## 三种 Exploration Mode

agent 怎么决定走哪步？policy 是：

$$
a_t = \arg\max_a \pi_{\text{explore}}(a \mid x_{t-1}^{S}, \mathcal{I})
$$

变量：
- $\pi_{\text{explore}}$：exploration policy
- $x_{t-1}^{S}$：上一步最后看到的 panorama
- $\mathcal{I}$：instruction

三种 mode：

### (a) Interactive：人手动控制
人直接给 $(\alpha_t, d_t)$。问题：人可能让 agent 撞墙，generation 质量雪崩。

### (b) GPT-assisted free exploration：GPT 当 pilot
[GPT-4o](https://arxiv.org/abs/2303.08774) 来选 action，目标是**避免 generative model 崩**。这其实是个 meta-policy——LLM 不规划 task 目标，它规划 "怎么走才不让生成质量 degrade"。

**人话**：GPT 当司机，它知道 "别太靠近墙，别转太猛"，让生成的视频保持高质量。

### (c) Goal-driven navigation：GPT 拆解 goal
给个 goal "走到蓝车位置"，GPT 把它拆成 low-level action sequence，迭代执行。

---

## 最有思想的部分：Imagination-Augmented Policy

常规 embodied policy：

$$
A = \arg\max_A \pi_{\theta_3}(A \mid i_0, g)
$$

变量：
- $A$：embodied action（比如回答 "stop or go"）
- $i_0$：初始单图 observation
- $g$：goal / question
- $\theta_3$：policy model 参数（GPT-based）

问题：只看一张图，agent 不知道拐角后面是啥，没法回答 "Danger ahead—stop or go ahead?"。

GenEx 的方案：**先在脑里走一遍，收集 imagined observations，再决策**。

**Step 1**：用 GenEx 跑一遍 imaginative exploration：
$$
\mathbf{x}_{0:T} \sim p(\mathbf{x}_{0:T} \mid i_0, l_0, \mathcal{I})
$$

**Step 2**：用 real + imagined observations 一起决策：
$$
A = \arg\max_A \pi_{\theta_3}(A \mid i_0, \mathbf{x}_{0:T}, g)
$$

**人话**：与其在真实世界里 risk 去 explore（可能贵、可能危险），先在脑里 simulate 一遍看看拐角后面是啥，再决定要不要往前走。这就是人类 "mental simulation" 的 AI 版本。

---

### Multi-agent 版本：Theory of Mind 的 generative 实现

更野心的版本：agent 不仅想象自己走，还想象 "如果我是 agent-k，我会看到什么"。

给 K 个 instruction $\mathcal{I}_k$ = "navigate to agent-k's position"，跑 K 次 GenEx：

$$
\{\mathbf{x}_{1:T}^{(k)}\}_{k=1}^K = (\mathbf{x}_{1:T}^{(1)}, \mathbf{x}_{1:T}^{(2)}, \ldots, \mathbf{x}_{1:T}^{(K)})
$$

然后决策：

$$
A = \arg\max_A \pi_{\theta_3}(A \mid i_0, \{\mathbf{x}_{1:T}^{(k)}\}_{k=1}^K, g)
$$

**人话**：我想象自己走到你的位置，看看你看到的世界，从而推断你会怎么行动，再调整我自己的决策。这就是 theory of mind——用 video generation 来实现。

这个方向我觉得潜力巨大。跟 [Fan et al. Evidential Active Recognition (CVPR 2024)](https://arxiv.org/abs/2402.17139) 的 prudent open-world perception 有呼应，跟 cognitive science 里的 mental simulation / theory of mind 文献也接得上。

---

## 实验结果人话版

### Generation Quality（Table 1）

| Model | FVD ↓ | PSNR ↑ | SSIM ↑ |
|---|---|---|---|
| Baseline (cubemap) | 196.7 | 26.1 | 0.88 |
| GenEx w/o SCL | 81.9 | 29.4 | 0.91 |
| GenEx | **69.5** | **30.2** | **0.94** |

- [FVD](https://arxiv.org/abs/1812.01717)：视频分布距离，越低越好
- [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)：像素级重建质量
- [SSIM](https://en.wikipedia.org/wiki/Structural_similarity_index_measure)：结构相似度

**结论**：panorama 表示比 cubemap 好很多，SCL 又在 panorama 基础上提升一截。

### Loop Consistency（Figure 9）

走 20m 闭环回到原点，latent MSE < 0.1。1000 个 loop 平均。**说明 Markov 假设下的 drift 被压住了**，主要归功于把 rotation 拆成确定性几何。

### Embodied Decision Making（Table 2）

| Method | Acc (%) | Logic Acc (%) |
|---|---|---|
| Multimodal GPT-4o（只看单图） | 46.10 | 12.51 |
| GPT-4o + GenEx | **85.22** | **83.88** |
| Human with Image | 91.50 | 70.93 |
| Human with GenEx | **94.00** | **86.19** |

两个 finding：

**Finding 1：Vision without imagination 对 GPT 是误导。**
Multimodal GPT-4o 只看一张图，acc 46%，logic acc 只有 12.5%——它在猜，没做真正 spatial reasoning。加 GenEx 后跳到 85%。说明单张 egocentric image 给 LLM 的信息不够推断 "拐角后面啥情况"。

**Finding 2：GenEx 也能增强人类。**
人类看图 91.5%，看了 GenEx 生成的 imagined video 后 94%。Multi-agent 场景更明显：55% → 77%。**Generative world explorer 当人类的 "外脑"。**

---

## 我的 critique

1. **$d_t$ 去哪了？** Transition 公式只 condition on rotated panorama + noise，distance 没显式出现。复现关键，论文 skip 了。
2. **World initialization 的 image conditioning 架构没讲。** ControlNet？IP-Adapter？这是复现的另一个关键。
3. **Long-range 只测了 20m。** "Boundless" 这个 claim 需要 100m、500m 的 stress test。closed loop 回到原点 MSE 低，不意味着中间 trajectory 上物体没 morph。
4. **Sim-to-real gap 完全没讨论。** 训练全是 UE5/Unity，inference 时 $i_0$ 是真实照片会怎样？
5. **GPT pilot 的 prompt 没给。** "避免 collapse" 怎么 prompt 的？这是工程核心。
6. **Inference latency 没讨论。** 一次 imagination-augmented decision 要跑 $T$ 次 video diffusion，multi-agent 还要 $\times K$。real-time embodied AI 这可能是致命瓶颈。

---

## 跟其他工作的关系——更广的图景

### 跟 model-based RL 的 world model 对比

GenEx 本质是 model-based RL 里的 world model，只不过：
- 用 diffusion 替代显式 dynamics function（适合高维 visual observation）
- 用 panoramic egocentric observation 替代 global state（规避全局一致性）
- 用 LLM 当 policy（zero-shot 适配新 task，替代 learned value function）

跟 [DreamerV3](https://arxiv.org/abs/2301.04104) 对比：Dreamer 在 latent space 学 dynamics，GenEx 在 pixel space 直接生成。Dreamer 快但需要 reconstruction，GenEx 直观但 compute 贵。

### 跟 Tesla world model 对比

你在 [Tesla AI Day](https://www.youtube.com/watch?v=j0z4Fx-CDEg) 讲的 world model 核心是 "predict future camera frames conditioned on past frames + action"。GenEx 是同一个 idea 的 academic 版，多了 360° panoramic + GPT planning 两层。Tesla 用真实 driving data，GenEx 用 UE5/Unity simulation data——这其实是 sim-to-real 的两个不同切入点。

### 跟 Sora 对比

[Sora](https://openai.com/research/video-generation-models-as-world-simulators) positioning 自己为 world simulator，但 OpenAI 没给技术细节，也没展示 embodied agent 在 Sora-generated world 里做 decision making。GenEx 是这个 vision 的 academic 对应版，有完整技术报告 + Imagination-Augmented Policy 实验。

### 跟 DeepMind Genie 2 对比

[Genie 2](https://deepmind.google/discover-blog/genie-2-a-large-scale-foundation-world-model/) 是同期工业进展，能从单图生成可交互 world。GenEx 强调自己有完整技术细节公开 + Imagination-Augmented Policy 概念。两者方向一致，GenEx 更 academic transparent。

### 跟 WorldLabs 对比

[WorldLabs](https://www.worldlabs.ai/blog)（Fei-Fei Li 的新公司）发布了 anime-world generation demo。GenEx 说自己 complementary。

---

## 未来有意思的方向

1. **用 [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) + dynamic deformation 替代 panoramic video diffusion**——可能解决 drift 问题，因为 Gaussian 有显式 3D structure
2. **用 [VLM-based reward](https://arxiv.org/abs/2310.08588) 给 GPT pilot 做 RL fine-tuning**——让它学会避免 collapse 的 action distribution
3. **Sim-to-real**：用 [domain randomization](https://arxiv.org/abs/1903.07288) + real panoramic data（[Matterport3D](https://arxiv.org/abs/2001.08557)）做 joint training
4. **Multi-agent theory-of-mind 扩展到 adversarial setting**——想象对手视角做 game-theoretic planning，跟 [multi-agent RL](https://arxiv.org/abs/2010.02692) 结合
5. **Latent space version**：把 panoramic video diffusion 压缩成 latent dynamics model，像 [GAIA-1](https://arxiv.org/abs/2303.17130) 那样，解决 inference latency 问题
6. **跟 [DUSt3R](https://arxiv.org/abs/2311.11844) 这类 geometry-aware 3D vision 结合**——GenEx 已经用 DUSt3R 做 active 3D mapping 的 backend，可以进一步把 geometric prior 注入 diffusion training

---

## Reference Links

- [GenEx 项目主页](https://genex.world)
- [GenEx 前作 arXiv](https://arxiv.org/abs/2411.11844)
- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [FLUX panorama LoRA v2](https://huggingface.co/jbilcke-hf/flux-dev-panorama-lora-2)
- [Sora technical report](https://openai.com/research/video-generation-models-as-world-simulators)
- [DeepMind Genie 2](https://deepmind.google/discover-blog/genie-2-a-large-scale-foundation-world-model/)
- [WorldLabs blog](https://www.worldlabs.ai/blog)
- [DUSt3R (CVPR 2024)](https://arxiv.org/abs/2311.11844)
- [Wonderworld](https://arxiv.org/abs/2406.09394)
- [Stable Video Diffusion](https://arxiv.org/abs/2311.15127)
- [SV3D](https://arxiv.org/abs/2403.12008)
- [Stable Zero123](https://stability.ai/news/stable-zero123-3d-consistent-image-generation)
- [Video Language Planning](https://arxiv.org/abs/2402.17139)
- [GPT-4 technical report](https://arxiv.org/abs/2303.08774)
- [FVD metric](https://arxiv.org/abs/1812.01717)
- [LPIPS](https://arxiv.org/abs/1801.03924)
- [DreamerV3](https://arxiv.org/abs/2301.04104)
- [GAIA-1](https://arxiv.org/abs/2303.17130)
- [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079)
- [NeRF](https://arxiv.org/abs/2003.08934)
- [Matterport3D](https://arxiv.org/abs/2001.08557)
- [Unreal Engine 5](https://www.unrealengine.com/)
- [Unity](https://unity.com/)
- [Tesla AI Day on world models](https://www.youtube.com/watch?v=j0z4Fx-CDEg)

---

如果你想我再 drill down 某一块——比如 SCL 的具体 loss form、或者 multi-agent theory-of-mind 跟 cognitive science 的联系、或者把 GenEx 跟 DreamerV3 做个详细架构对比——说一声。

---

# GenEx: Generating an Explorable World — 深度技术解析

Andrej，这篇 paper 我觉得非常对你的胃口——它把 generative video model、panoramic geometry、embodied reasoning 三件事缝合在一起，核心 ambition 是把 Sora-style video diffusion 重新 cast 成一个 **egocentric world model**，并且让 GPT 在这个 "imagined world" 里做 lookahead planning。下面我从 intuition → formulation → architecture → experiments → 与同期工作的关系，逐层拆解。

---

## 1. 核心 Intuition：把 World Model 拆成 "Initialize + Transition"

人类认知科学里有个老概念叫 **mental model / mental simulation**：你瞥一眼房间，脑里就建了个内部 3D 表示，可以 "在脑子里走过去看看拐角后面是什么"。GenEx 想复刻这件事，关键 insight 是——

**不要去 reconstruct 一个 globally consistent 3D scene（NeRF/Gaussian Splatting 那条路），而是 maintain 一个 egocentric panoramic observation stream，用 video diffusion 当 transition function。**

这避开了几个 hard problems：
- 不需要显式 depth / 3D supervision（depth estimator 在 occlusion、textureless 区域经常崩）
- 不需要 globally bounded scene（single-image 3D 方法如 Stable Zero123、SV3D、Wonderworld 都受 render distance / FoV 限制）
- 允许 "boundless" exploration（只要 transition model 不 drift）

代价是：你要解决 **loop consistency**（走一圈回来看到的应该和出发时一样）和 **spherical seam artifacts**（equirectangular panorama 左右边缘的 discontinuity）。

---

## 2. Problem Formulation 详解

### 2.1 整体 joint distribution

论文把整件事写成：

$$
p(\mathbf{x}_{0:T} \mid i_0, l_0) = \underbrace{p_{\theta_1}(x_0 \mid i_0, l_0)}_{\text{world init}} \cdot \underbrace{\prod_{t=1}^{T} p_{\theta_2}(\mathbf{x}_t \mid x_{t-1}^{S}, a_t)}_{\text{world transition}}
$$

变量含义：
- $i_0$：单张 input RGB image（agent 第一眼看到的）
- $l_0$：language description，e.g. "a city street at dusk"
- $x_0$：初始化的 360° panorama（agent 当前位置的完整视野）
- $\mathbf{x}_t = (x_t^0, x_t^1, \ldots, x_t^S)$：第 $t$ 步生成的 panoramic **video**，长度 $S$ 帧，对应 agent 向前走 $d_t$ 距离的过程中看到的连续 panorama
- $x_t^S$：第 $t$ 步 video 的**最后一帧**，作为下一步的 conditioning（论文里记号有点 abused，$x_0^S := x_0$）
- $a_t = (\alpha_t, d_t)$：action，包含 rotation angle $\alpha_t$ 和 forward distance $d_t$
- $\theta_1$：image-to-panorama generator
- $\theta_2$：panoramic video generator

这个 factorization 的关键 trick 是 **Markov 假设**：下一步只依赖上一步的最后一帧 + 当前 action。这是为了让 inference tractable，但也埋下了 drift 的隐患（后面 IELC metric 就是为这个设计的）。

### 2.2 World Initialization

基础模型是 [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) + [jbilcke-hf 的 panorama LoRA v2](https://huggingface.co/jbilcke-hf/flux-dev-panorama-lora-2)。原始 panorama LoRA 只接受 text：

$$
x_0 \sim p_{\text{flux}}(x \mid l_0)
$$

但这样没法保证和 input image $i_0$ consistent。论文把它 extend 成 image+text conditioned：

$$
x_0 \sim p_{\theta_1}(x \mid i_0, l_0)
$$

具体怎么 extend 的（ControlNet？ReferenceNet？IP-Adapter？）论文没细说，但从 Figure 4 看，应该是把 $i_0$ 作为 visual condition 注入。这里其实是个工程 gap——reader 想复现得自己猜。

### 2.3 World Transition

这是核心模块，pipeline 有三步：

**Step A: Action sampling**
$$
a_t \sim \mathcal{A}, \quad |\mathcal{A}| = \infty
$$
$\mathcal{A}$ 是 Unreal Engine / Unity 里定义的连续 action space（旋转角 + 距离），离散化到训练时用的 grid。

**Step B: Sphere rotation**
$$
{x_{t-1}^{S}}' = \mathcal{T}(x_{t-1}^{S}, \alpha_t)
$$
这里 $\mathcal{T}$ 是在 spherical polar coordinate 上的旋转，**不经过 neural network**，纯几何变换。这点很聪明：旋转是确定性几何操作，让 diffusion model 只学 "forward movement" 这件事，降低学习难度。

具体公式（Appendix D.4）：

$$
\mathcal{T}(u, \nu, \Delta\phi, \Delta\theta) = f_{S \to \mathcal{P}}\bigl(\mathcal{R}(f_{\mathcal{P} \to S}(u, \nu), \Delta\phi, \Delta\theta)\bigr)
$$

$$
\mathcal{R}(\phi, \theta, \Delta\phi, \Delta\theta) = (\phi + \Delta\phi \pmod{2\pi},\ \theta + \Delta\theta \pmod{\pi})
$$

其中 $(\phi, \theta)$ 是 spherical polar coordinates（longitude / latitude），$(u, \nu)$ 是 panorama pixel grid coordinates。$f_{S \to \mathcal{P}}$ 和 $f_{\mathcal{P} \to S}$ 是球面↔2D 网格的双射：

$$
f_{S \to \mathcal{P}}(\phi, \theta) = \left(\frac{W}{2\pi}(\phi + \pi),\ \frac{H}{\pi}\left(\frac{\pi}{2} - \theta\right)\right)
$$

$$
f_{\mathcal{P} \to S}(u, \nu) = \left(\frac{2\pi u}{W} - \pi,\ \frac{\pi}{2} - \frac{\pi \nu}{H}\right)
$$

变量：
- $W, H$：panorama 的宽、高
- $\phi \in [-\pi, \pi)$：longitude（经度，水平方向）
- $\theta \in [-\pi/2, \pi/2]$：latitude（纬度，垂直方向）
- $u \in [0, W-1]$：pixel column
- $\nu \in [0, H-1]$：pixel row

注意 equirectangular projection 在两极会严重 stretch（top/bottom 行对应极点附近一小块球面区域），所以 panorama 上下边缘通常不存有用内容。

**Step C: Panoramic video generation**
$$
\mathbf{x}_t \sim p_{\theta_2}(\mathbf{x} \mid {x_{t-1}^{S}}', \epsilon_t), \quad \epsilon_t \sim \mathcal{N}(0, I)
$$

$\epsilon_t$ 是 Gaussian noise，注入 stochasticity 让世界 "explorable"（同一 action 可以生成不同 forward exploration 的 video）。这里 condition 只用了 rotated panorama，没显式 condition on $d_t$——这个我读着有点疑惑，可能 $d_t$ 隐含在训练数据的 video 长度 $S$ 里（走多远→video 多长）。

### 2.4 Spherical-Consistency Learning (SCL)

如果不加约束，在 equirectangular panorama 上训 diffusion 会出 seam：左右边缘（$\phi = -\pi$ 和 $\phi = \pi$ 其实是同一条经线）生成的内容不连续。SCL（参考他们前作 [Generative World Explorer, Lu et al. 2024](https://arxiv.org/abs/2411.11844)）的做法是在 training loss 里加一个约束，强制 panorama 在球面上 smooth。Table 1 显示加了 SCL 后 FVD 从 81.9 降到 69.5，PSNR 从 29.4 升到 30.2，SSIM 从 0.91 升到 0.94——确实有用。

直觉上：把 equirectangular image 想成一张纸卷成圆筒粘起来，左右边缘必须 match；SCL 就是让 convolution 跨越这条 seam。

---

## 3. 数据与 Representation

### 3.1 三种 360° 表示

论文用了三种 representation 互转（Figure 3）：
- **Cubemap**：6 个 90° perspective face，UE5/Unity 原生输出格式，渲染友好
- **Equirectangular panorama**：2D 矩形图，video diffusion model 的训练 / inference 格式
- **Sphere**：3D 球面，做 rotation 时用

训练数据从 [UE5](https://www.unrealengine.com/) 的 realistic city assets 和 [Unity](https://unity.com/) 的 animated world assets 里采 cubemap sequence，然后投影成 equirectangular 训 diffusion。这保证了 **physical grounding**——生成的世界有真实物理结构，不是 random AI hallucination。

### 3.2 为什么不用真实世界数据？

论文明说：在真实世界收集 dense panoramic exploration trajectory 太贵、变异性太大。用 physics engine 可以批量采样 arbitrary trajectory。这其实呼应了 [Sora technical report](https://openai.com/research/video-generation-models-as-world-simulators) 的思路——world simulator 不一定需要 real video，simulation data 也能学出 world dynamics。

---

## 4. Exploration Policy 与三种 Mode

$$
a_t = \arg\max_a \pi_{\text{explore}}(a \mid x_{t-1}^{S}, \mathcal{I})
$$

$\mathcal{I}$ 是 instruction。三种 mode：

1. **Interactive**：人类直接给 $(\alpha_t, d_t)$。问题：人可能让 agent 撞墙，generation quality 雪崩。
2. **GPT-assisted free exploration**：[GPT-4o](https://arxiv.org/abs/2303.08774) 当 pilot，选 action 来**最大化 generation fidelity**（避免 collapse）。这是个有意思的 meta-policy——LLM 不规划 task，它规划 "怎么走才不让 generative model 崩"。
3. **Goal-driven navigation**：GPT 把 high-level goal（"走到蓝车位置"）拆成 low-level action sequence，迭代执行。

---

## 5. Imagination-Augmented Policy（核心 contribution）

这部分我觉得是 paper 最有思想的地方。常规 embodied policy：

$$
A = \arg\max_A \pi_{\theta_3}(A \mid o=i_0, g)
$$

只看初始单图，对未观察区域一无所知。论文提出：先用 GenEx 在脑里 "走一遍" 收集 imagined observations $\mathbf{x}_{0:T}$，再决策：

$$
A = \arg\max_A \pi_{\theta_3}(A \mid i_0, \mathbf{x}_{0:T}, g)
$$

Algorithm 2 把这写清楚了。直觉是：与其在真实世界里 risk 去 explore（可能 expensive / dangerous），不如在 imagined world 里 simulate 一遍再 commit decision。这和 [Du et al. Video Language Planning](https://arxiv.org/abs/2402.17139)、[Yang et al. "Video as the new language for real-world decision making"](https://arxiv.org/abs/2402.17139)、[Bu et al. closed-loop visuomotor control with generative expectation](https://arxiv.org/abs/2409.09016) 是同一波思路，但 GenEx 多了 360° panoramic + spherical consistency 这一层。

### Multi-agent 版本

更野心：agent 不仅想象自己走，还想象 "如果我是 agent-k，我会看到什么"。给 K 个 instruction $\mathcal{I}_k$ "navigate to agent-k's position"，跑 K 次 GenEx，得到 $\{\mathbf{x}_{1:T}^{(k)}\}_{k=1}^K$，然后：

$$
A = \arg\max_A \pi_{\theta_3}(A \mid i_0, \{\mathbf{x}_{1:T}^{(k)}\}_{k=1}^K, g)
$$

这是 theory of mind 的 generative 版本——用 video generation 模拟 "他者视角"。我觉得这个方向潜力巨大，跟 [Fan et al. Evidential Active Recognition (CVPR 2024)](https://arxiv.org/abs/2402.17139) 的 open-world prudent perception 有点呼应。

---

## 6. 实验数据解读

### 6.1 Generation Quality（Table 1）

| Model | Representation | FVD ↓ | MSE | LPIPS ↓ | PSNR ↑ | SSIM ↑ |
|---|---|---|---|---|---|---|
| Baseline | 6-view cubemaps | 196.7 | 0.10 | 0.09 | 26.1 | 0.88 |
| GenEx w/o SCL | panorama | 81.9 | 0.05 | 0.05 | 29.4 | 0.91 |
| GenEx | panorama | **69.5** | **0.04** | **0.03** | **30.2** | **0.94** |

- [FVD (Frechet Video Distance)](https://arxiv.org/abs/1812.01717)：衡量 generated video distribution 和 real video distribution 的距离，越低越好
- [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)：pixel-level 重建质量
- [SSIM](https://en.wikipedia.org/wiki/Structural_similarity_index_measure)：structural similarity
- [LPIPS](https://arxiv.org/abs/1801.03924)：perceptual distance，用 deep features 算
- Baseline 用 cubemap 表示直接训，明显比 panorama 差——验证了 panoramic representation 的优势
- SCL 全面提升所有 metric，尤其 FVD 降了 12.4

### 6.2 Imaginative Exploration Loop Consistency (IELC)

这是论文新提的 metric，专门测 long-range drift。做法：随机采样 closed-loop path（走一圈回到原点），算初始 real image 和最终 generated image 的 latent MSE，1000 个 loop 平均。

Figure 9 显示：即使 20m loop + 多个连续 video，latent MSE 仍 < 0.1。这个数我觉得 impressive——意味着 Markov 假设下，spherical rotation 的 deterministic 部分吸收了大部分视角变化，diffusion 只学 forward motion，drift 被压住了。

### 6.3 Embodied Decision Making（Table 2 & 3）

Table 2（单 agent EQA）：

| Method | Acc (%) | Confidence (%) | Logic Acc (%) |
|---|---|---|---|
| Random | 25.00 | 25.00 | = |
| Human Text-only | 44.82 | 52.19 | 46.82 |
| Human with Image | 91.50 | 80.22 | 70.93 |
| Human with GenEx | **94.00** | **90.77** | **86.19** |
| Unimodal Gemini-1.5 | 30.56 | 29.46 | 13.89 |
| Unimodal GPT-4o | 27.71 | 26.38 | 20.22 |
| Multimodal Gemini-1.5 | 46.73 | 36.70 | 0.0 |
| Multimodal GPT-4o | 46.10 | 44.10 | 12.51 |
| GPT-4o with GenEx | **85.22** | **77.68** | **83.88** |

Table 3（multi-agent）类似趋势，GPT-4o + GenEx 达到 94.87% acc。

两个关键 finding：

**Finding 1: Vision without imagination can mislead GPTs.**
看 Multimodal GPT-4o (46.10) vs Unimodal GPT-4o (27.71)——多模态确实比纯文本好，但远不如加 GenEx 的 85.22。更微妙的是 Multimodal Gemini-1.5 的 Logic Acc 是 0.0——它根本没在做 spatial reasoning，只是在猜。这说明**单张 egocentric image 给 LLM 的信息不足以推断 "拐角后面是否危险"**，硬猜会 hallucinate。

**Finding 2: GenEx 也能增强人类认知。**
Human with Image (91.50) → Human with GenEx (94.00)，multi-agent 场景 55.24 → 77.41。这其实是个 cognitive augmentation 的 demo——generative world explorer 当人类的 "外脑"，帮他们想象看不到的区域。

---

## 7. 与同期工业进展的关系

论文 Discussion 部分主动 compare 了几个 concurrent work：

- **[WorldLabs](https://www.worldlabs.ai/blog)**（Fei-Fei Li 的新公司，2024-12）：发布了从单图生成 anime world 的 demo。GenEx 说自己 complementary，并强调了技术细节公开。
- **[DeepMind Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)**（2024-12）：interactive world model blog。GenEx 强调自己有完整技术报告 + Imagination-Augmented Policy 概念。
- **[Sora](https://openai.com/research/video-generation-models-as-world-simulators)**：OpenAI 把 video model 当 world simulator 的 positioning paper。GenEx 是这个 vision 的 academic 对应版，但加了 panoramic + embodied policy 这两层。

更广的 related work 谱系：
- Single-image 3D：[Wonderworld (Yu et al.)](https://arxiv.org/abs/2406.09394)、[Stable Zero123](https://stability.ai/news/stable-zero123-3d-consistent-image-generation)、[TripoSR](https://arxiv.org/abs/2403.02151)、[SV3D](https://arxiv.org/abs/2403.12008)——都受 render distance / FoV 限制
- Video generation：[Stable Video Diffusion](https://arxiv.org/abs/2311.15127)、[VideoPoet](https://arxiv.org/abs/2311.15127)、Sora——缺 physical grounding
- 3D mapping backbone：[DUSt3R (CVPR 2024)](https://arxiv.org/abs/2311.11844)——GenEx 用它做 active 3D mapping 的 backend
- Video-as-policy：[Video Language Planning](https://arxiv.org/abs/2402.17139)、[This&That (Wang et al.)](https://arxiv.org/abs/2407.05530)——visual plan for decision making

---

## 8. 我会怎么批判性看这篇 paper

**优点：**
1. Formulation 干净：world = init + transition 的 Markov 分解很优雅
2. Geometry-aware design：把 rotation 拆出来用确定性 spherical transform，让 diffusion 只学 forward motion，是正确的 inductive bias
3. IELC metric 设计得好，直接 measure drift
4. Imagination-Augmented Policy 是个有思想的概念，把 world model 和 policy 的关系讲清楚了
5. Multi-agent version 用 generative simulation 实现 theory of mind，方向新颖

**我会追问的问题：**
1. **$d_t$ 在 video generation 里怎么 condition 的？** 公式里只看到 $\epsilon_t$ 和 rotated panorama，distance 信息去哪了？是隐式编码在 video length $S$ 里，还是被 discretize 成几档？这对复现很关键。
2. **World initialization 的 image conditioning 具体架构？** ControlNet / IP-Adapter / ReferenceNet？论文完全 skip 了。
3. **Long-range 一致性的根本限制。** IELC 测的是 closed loop 回到原点的 latent MSE，但中间 trajectory 上物体可能已经 morph 了。20m loop < 0.1 很好，但 100m 呢？500m 呢？boundless 这个 claim 需要更激进的 stress test。
4. **Sim-to-real gap。** 训练数据全是 UE5/Unity，inference 时如果 $i_0$ 是真实照片，domain gap 会让 generation 质量下降多少？论文承认这是 future work，但没给任何 quantitative hint。
5. **GPT-assisted exploration 的 "avoid collapse" 是怎么 prompt 的？** 这是工程关键，论文只说 GPT-4o 当 pilot，没给 prompt template。
6. **计算成本。** 一次 Imagination-Augmented Policy 要跑 $T$ 次 video diffusion（multi-agent 还要 $\times K$），inference latency 多少？这对 embodied real-time decision making 是致命问题，论文完全没讨论。

---

## 9. 给你的 intuition 总结

如果让我一句话概括 GenEx 的 contribution：**它把 video diffusion 从 "生成好看的视频" 重新 cast 成 "egocentric world dynamics 的 transition model"，然后用 GPT 当 imagination-augmented planner，在 imagined trajectory 上做 lookahead 决策。**

更深一层的 intuition：这其实是在重新发明 model-based RL 的 world model，只不过——
- 用 diffusion 替代显式 dynamics 函数（更适合高维 visual observation）
- 用 panoramic egocentric observation 替代 global state（更接近 agent 视角，规避全局一致性）
- 用 LLM 当 policy + planner（替代 learned value function，zero-shot 适配新 task）

你之前在 [Tesla AI Day 讲 world model](https://www.youtube.com/watch?v=j0z4Fx-CDEg) 的时候，核心是 "predict future camera frames conditioned on past frames + action"。GenEx 是同一个 idea 的 academic 版，但多了 360° panoramic + GPT planning 这两层。如果把这个跟 neurally-rendered latent space（比如 [GAIA-1](https://arxiv.org/abs/2303.17130) 那种）对比，GenEx 显然更 "literal"——直接在 pixel space 生成，不学 latent dynamics。这有利有弊：pixel space 直观但 compute 贵，latent space 快但需要 reconstruction。

未来有意思的延伸方向：
- 把 GenEx 的 panoramic video diffusion 替换成 [3D Gaussian Splatting](https://arxiv.org/abs/2303.17130) + dynamic deformation，可能解决 drift 问题
- 用 [VLM-based reward](https://arxiv.org/abs/2310.08588) 给 GPT pilot 做 RL fine-tuning，让它学会避免 collapse 的 action distribution
- sim-to-real：用 [domain randomization](https://arxiv.org/abs/2303.17130) + real panoramic data（比如 [Matterport3D](https://arxiv.org/abs/2303.17130)）做 joint training
- multi-agent theory-of-mind 可以扩展到 adversarial setting——想象对手视角做 game-theoretic planning

---

## Reference Links

- [GenEx 项目主页](https://genex.world)
- [GenEx 前作 arXiv (Lu et al. 2024)](https://arxiv.org/abs/2411.11844)
- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [FLUX panorama LoRA v2](https://huggingface.co/jbilcke-hf/flux-dev-panorama-lora-2)
- [Sora technical report](https://openai.com/research/video-generation-models-as-world-simulators)
- [DeepMind Genie 2 blog](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)
- [WorldLabs blog](https://www.worldlabs.ai/blog)
- [DUSt3R (CVPR 2024)](https://arxiv.org/abs/2311.11844)
- [Wonderworld](https://arxiv.org/abs/2406.09394)
- [Stable Video Diffusion](https://arxiv.org/abs/2311.15127)
- [SV3D](https://arxiv.org/abs/2403.12008)
- [TripoSR](https://arxiv.org/abs/2403.02151)
- [Video Language Planning](https://arxiv.org/abs/2402.17139)
- [GPT-4 technical report](https://arxiv.org/abs/2303.08774)
- [FVD metric paper](https://arxiv.org/abs/1812.01717)
- [LPIPS](https://arxiv.org/abs/1801.03924)
- [Evidential Active Recognition](https://arxiv.org/abs/2402.17139)
- [Unreal Engine 5](https://www.unrealengine.com/)
- [Unity](https://unity.com/)

如果你想我再展开某一块——比如 SCL 的具体 loss form，或者 Imagination-Augmented Policy 跟 DreamerV3 的 world model 范式对比，或者 multi-agent theory-of-mind 那块更深的 cognitive science 联系——告诉我，我可以接着 drill down。
