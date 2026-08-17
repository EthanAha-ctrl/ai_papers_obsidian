---
source_pdf: Pelican-Unify 1.0.pdf
paper_sha256: f5ab200f893c536c84a81ab04a0634475b2819a160549690d7fc70af058c13f9
processed_at: '2026-08-06T02:38:24-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Pelican-Unify 1.0

## 一句话版本

embodied AI 圈子现在大家各做各的——有人做 VLM 看图说话，有人做 VLA 把语言映射到 action，有人做 world model imagine 未来像素。Pelican-Unify 说：这四件事其实是一件事，应该塞进同一个 model，用同一套 representation，同一个 training loop 一起训。他们真这么做了，并且实验证明joint training没让任何一项能力变弱，反而让三项都拿到第一或第二。

## 为什么这件事值得关注

过去几年embodied AI的story是specialization——你做policy就专心做policy，我做world model就专心imagine未来。每个specialist单独看都很强，但它们之间接起来的那个interface是个semantic断崖。

举个具体场景：VLM看完场景说"红色插头应该拔出来"，这句话传给VLA policy，policy只收到一个text embedding，它完全不知道"红色插头"在物理上意味着什么consequence，也不知道拔出来之后scene会变成什么样。World model能imagine未来像素，但它imagine的方向完全没被task logic steer，可能imagine出一个视觉流畅但task-irrelevant的未来。

Pelican-Unify的claim是：这种断崖不是engineering问题，是architecture问题。你只要把四件事split开train，interface就一定是断的。真正的解法是让它们share同一个representation，从training第一天就互相see对方、互相shape。

## 架构怎么做到的

我把架构画成人话版：

```
你给model看：过去几帧video + action history + 语言指令 + 机器人状态
        ↓
   【Qwen3-VL 把这些都吃进去】
        ↓
   【VLM 边想边说一段 CoT：先讲"会发生什么"，再讲"该做什么"】
        ↓
   【把 CoT 最后的 hidden state 压成一个 dense vector z】
        ↓
   z 是唯一通往下层的桥
        ↓
   【Wan2.2 DiT 同时 denoise 两样东西】
        ↓
   ← 左手出 future video    右手出 action chunk →
```

三个细节最聪明，我一个个讲。

### 细节一：tied embedder

VLM 的 input 端有个 video encoder $\mathcal{E}_v$ 把帧变成 token，有个 action encoder $\mathcal{E}_a$ 把 action 变成 token。下游的 DiT 要 denoise 的 video latent 和 action latent，**用的就是这两个同一个encoder**。

Eq. 6 写得很清楚：
$$x^v = \mathcal{E}_v(v_{t:t+H}), \quad x^a = \mathcal{E}_a(a_{t:t+H})$$

变量：$x^v$ 是 future video 经 $\mathcal{E}_v$ 编码后的 clean latent，$x^a$ 是 future action chunk 经 $\mathcal{E}_a$ 编码后的 clean latent。$v_{t:t+H}$ 是 $t$ 到 $t+H$ 的未来帧，$H$ 是 chunk horizon。

这意味着什么？VLM 学会"读"的坐标系，和 DiT 学会"写"的坐标系，是同一个。没有alignment adapter，没有projection layer，没有contrastive loss来对齐两个space。Gradient从video loss直接流回 $\mathcal{E}_v$，VLM的video representation就被action task和video prediction task同时塑造。

类比一下：LLM 里 input embedding 和 output embedding tying，reader和writer共享token space，gradient天然双向流动。Pelican-Unify把这招用到video和action上。

### 细节二：z 是 bottleneck

Eq. 5：$z = P_\phi(h_{\tau_t})$

变量：$h_{\tau_t}$ 是 VLM 在 CoT trace 最后一个 token 的 hidden state，$P_\phi$ 是一个 learned projection，$z$ 是投影后的 dense latent。

关键：下游的 DiT **只看到 z**，看不到原始 observation、看不到 raw language、看不到action history。所有信息必须经过 z 这个瓶颈。

为什么要这么设计？如果 DiT 能直接拿 raw observation embedding，那它就有两条通路——一条走 z (reasoning)，一条走 raw obs (perception shortcut)。Model 会偷懒走 shortcut，reasoning 就被绕过了。

把 z 设成唯一通路，就逼着 reasoning 必须encode所有generation需要的东西。video head 要预测未来帧，z 就得包含 object motion、contact formation；action head 要预测 joint trajectory，z 就得包含 end-effector waypoint、subskill 分解。三个 loss（text、video、action）都对 z 求梯度，z 被迫成为"semantic + predictive + actionable"的交集。

这是 information bottleneck 思路：压缩bandwidth，反而逼出abstraction。

### 细节三：CoT 是interleaved的

CoT trace $\tau_t$ 里包含两种 language：
- Video CoT：描述场景应该怎么 evolve（哪个 object 会 move，contact 怎么 form）
- Action CoT：描述 motor program（调用哪个 subskill，end-effector 去 哪个 waypoint）

两种 language 放进同一个 sequence，同一个 causal pass。这其实是在用 autoregressive language modeling 的 causal mask 强制一个思考顺序：先 imagine 再 plan 再 act。

这招比 ECoT (Zawalski et al. 2025) 更进一步——ECoT 把 CoT 当 reasoning scaffold，最后还是 flat action head 输出。Pelican-Unify 把 CoT 的 terminal state 直接 condition diffusion，CoT 真的 shape 了 generation。
- ECoT paper: https://arxiv.org/abs/2503.00038

## Diffusion 部分讲清楚

### Action stream：标准 flow matching

Eq. 8：
$$x_s^a = (1-s)x^a + s\epsilon^a, \quad u_s^a = \epsilon^a - x^a$$

变量解释：
- $s \sim \mathcal{U}(0,1)$：diffusion time，从0到1均匀采样
- $x^a$：clean action latent
- $\epsilon^a \sim \mathcal{N}(0, I)$：标准 Gaussian noise
- $x_s^a$：在 time $s$ 处的 noised latent
- $u_s^a$：target velocity field，方向从 clean 指向 noise

$(1-s)x + s\epsilon$ 是从 clean 到 noise 的线性插值，对应 probability flow ODE $dx_s = (\epsilon - x) ds$，velocity field 就是 $\epsilon - x$。这是 flow matching (Lipman et al. 2023) 标准形式，比 DDPM 的 $\sqrt{\bar\alpha}x + \sqrt{1-\bar\alpha}\epsilon$ 简单——没用 variance schedule，就是线性 path。

DiT 预测 $\hat{u}_s^a$，loss 在 Eq. 12 用 Smooth L1：
$$\mathcal{L}_{\text{action}} = \mathbb{E}_{s, \epsilon^a}[M_a \odot \text{SmoothL1}(\hat{u}_s^a, u_s^a)]$$

$M_a$ 是 valid action dimension mask。比如 7-DOF arm + 2-DOF gripper = 9维，但有些 task 里 gripper 维度无效（pour task 不需要预测 gripper），mask 让 loss 只在 valid 维度累加。

### Video stream：prefix-conditioned flow matching

这部分有创意。video 要 predict 未来，还要 condition 在已观察到的 prefix 上。

Eq. 9：
$$x_s^v = M_{\text{cond}} \odot x^v + M_{\text{fut}} \odot ((1-s)x^v + s\epsilon^v)$$

Eq. 10：
$$u_s^v = M_{\text{fut}} \odot (\epsilon^v - x^v)$$

变量：
- $x^v$：包含 prefix 和 future 两段的 combined video latent
- $M_{\text{cond}}$：binary mask，1 的位置对应 prefix frames
- $M_{\text{fut}}$：binary mask，1 的位置对应 future frames
- $M_{\text{cond}} + M_{\text{fut}} = \mathbf{1}$
- $\epsilon^v$：仅在 future 区域采样的 Gaussian noise

人话：**prefix frames 保持 clean，future frames 被 noise**，DiT 只 denoise future 部分。这相当于把 video generation 变成 inpainting——已知过去，预测未来。

妙处还是 tied embedder：$o_{\leq t}$ 用 $\mathcal{E}_v$ encode，future 也用 $\mathcal{E}_v$ encode，prefix latent 和 future latent 在同一个 token space，DiT 天然能看懂 prefix，不需要额外 condition encoder。

Video loss (Eq. 11) 也只在 future 区域算：
$$\mathcal{L}_{\text{video}} = \mathbb{E}_{s, \epsilon^v}[\|M_{\text{fut}} \odot (\hat{u}_s^v - u_s^v)\|_2^2]$$

### Joint denoising：video 和 action 互相看

Eq. 7：$(\hat{u}_s^v, \hat{u}_s^a) = f_\theta(x_s^v, x_s^a, z, s)$

$f_\theta$ 是 Wan2.2 初始化的 DiT + 两个轻量 head $d_v, d_a$。Paper 里讲：
- video tokens 和 action tokens 通过 **shared self-attention** 交互——action token 能 attend 到 video token，video token 能 attend 到 action token
- $z$ 通过 **cross-attention** 注入——reasoning summary 持续 shape 整个 denoising trajectory
- modality-specific 参数只在 input boundary ($\mathcal{E}_v, \mathcal{E}_a$) 和 output boundary ($d_v, d_a$)，中间几十层 transformer block 是 share 的

shared self-attention 是真正的"unification"机制。每一层都让 action token 和 video token 互相看，action 被 imagine 的 consequence shape，imagine 被 action 的执行 shape。这跟 π0.5 那种"action 是 VLM 最后接一个 flow head"有本质区别。
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054

## 实验的关键 story

### Story 一：unification 不牺牲 specialist

Table 1 关键数字：

| Model | 8 benchmark Avg |
|---|---|
| Qwen3-VL-4B-Instruct (base) | 58.2 |
| **Pelican-Unify 1.0** | **64.7** |

从 58.2 提到 64.7，+6.5。但细看：
- General benchmark（MMMU, MMBench, MMStar, InfoVQA, ChartQA）：基本持平或微涨，MMMU 从 52.6→53.0
- Embodied benchmark：Where2Place 17.0→45.2 (**+28.2**)，PhyX 41.1→61.7 (**+20.6**)

这个分布说明：unification training 学到的是**物理 grounding**，不是通用 reasoning。Action history embedding 和 action loss 让 VLM 被迫理解"这个动作会让 scene 怎么变"，这种 causal 理解 transfer 到 Where2Place 和 PhyX 这类物理直觉任务。General reasoning 没退化也没大涨——加 action/video loss 没"污染"VLM 的核心能力。

### Story 二：RoboTwin 第二名

Table 2：

| Model | Clean | Randomized | Avg |
|---|---|---|---|
| π0 | 65.9 | 58.4 | 62.2 |
| π0.5 | 82.7 | 76.8 | 79.8 |
| starVLA | 88.2 | 88.3 | 88.3 |
| LingBot-VA | 92.9 | 91.6 | 92.3 |
| AIM | 94.0 | 92.1 | 93.1 |
| **MotuBrain** | **95.8** | **96.1** | **95.9** |
| **Pelican-Unify** | 93.6 | 93.3 | **93.5** |

Pelican-Unify 第二，落后 MotuBrain 2.4 个点。Paper 自己标注"second-best average success rate among compared methods"——他们承认不是第一。

但我注意到：MotuBrain 也是 unified world action model（Motus 团队出品），是 Pelican-Unify 最直接竞品。MotuBrain 没在 VLM benchmark 上比，所以 Pelican-Unify 的优势是**同时** VLM 第 1 + WorldArena 第 1 + RoboTwin 第 2，而 MotuBrain 只有 action 强。

Clean 和 Randomized 几乎无差（93.6 vs 93.3），对扰动鲁棒。Failure 集中在 hanging mugs、dustbin insertion 这种 tight 几何对齐 task——这种任务本质是几何精度问题，reasoning 救不了。

### Story 三：WorldArena 第 1

Table 3：

| Model | EWM Score | Rank |
|---|---|---|
| **Pelican-Unify** | **66.03** | **1** |
| WorldScape v0.2 | 64.24 | 2 |
| FlowWAM-FiveAges | 64.12 | 3 |
| MotuBrain | 64.07 | 4 |
| Wan2.6 | 59.80 | 12 |
| Veo3.1 | 57.77 | 15 |

比第 2 高 1.79 分。但细项更有意思：
- **3D Accuracy: 98.13（第 1）**
- **Motion Quality: 62.69（第 1）**
- Visual Quality 63.43（中等）
- Content Consistency 60.33（中等）
- Physics Adherence 61.51（中等）

3D 几何和运动学遥遥领先，pixel-level visual quality 和 physics fidelity 只是中等。这合理——DiT 从 Wan2.2 初始化但被 robot data heavily fine-tune，pixel generation 能力比纯 video model 弱；但 robot data 里有真实 3D 结构和 kinematic 约束，所以 3D accuracy 和 motion quality 强。

Human eval (Table 4) 更说明问题：

| Model | Task Success | Controllability | Temporal Consistency | Physical Plausibility | Avg |
|---|---|---|---|---|---|
| **Pelican-Unify** | **1.81** | **2.00** | 2.00 | 1.23 | **1.76** |
| Seedance2.0 | 1.21 | 1.87 | 1.98 | 1.15 | 1.55 |
| Happyhorse-1.0 | 1.65 | 1.81 | 2.00 | 0.13 | 1.40 |
| **EnerVerse-AC** | **0.00** | 1.84 | 2.00 | 1.64 | 1.37 |
| Wan2.7 | 1.19 | 1.68 | 2.00 | 0.29 | 1.29 |

**Task Success 1.81 是最关键的**。其他 video diffusion model（Happyhorse, Wan2.7, Cosmos-Predict2）temporal consistency 都满分 2.0，但 Task Success 很低——生成的 video 视觉流畅但**任务上跑题了**。EnerVerse-AC 甚至 Task Success=0.00 但 Temporal Consistency=2.00，这是"漂亮的废话"。

这恰好印证 paper 核心论点：**纯 video model optimize 的是 pixel fidelity，但 embodied 需要的是 task-relevant future imagination**。Pelican-Unify 让 action loss 和 text loss 同时 flow 到 z，generated video 必须既 imagination-plausible 又 action-consistent 又 task-relevant。

Physical Plausibility 1.23 是相对弱项——不如 EnerVerse-AC 的 1.64。Robot data 里物理约束编码还不够强，未来可能需要 physics engine supervision 或 contact-aware loss。

## Real robot 上的 compositional generalization

UR5e 上：atomic task A=plug RJ45，B=waterproof，分别训练，**训练数据里完全没有 A+B 的 chained demonstration**。Test time 给 "plug RJ45 cable into port 3 and apply waterproofing"，model 要在一个 episode 里完成 A→B。

Paper 说 failures 集中在 A→B 的 transition——"the moment where the just-completed A-state must be re-perceived as the new initial condition for B"。VLA baselines 失败原因不是 re-perception，而是 "action distributions carry no representation of what should happen after A is done"。

这是个很 clean 的 test，直接验证了 imagination face——model 在 transition 处先 imagine A 完成后 scene 应该长什么样，然后把这个 imagined state 当作 B 的 initial condition。

**但只有 2 个 atomic task 的 1 种 combination，规模太小**。A+B 只有一种组合。真正 compositional generalization 应该测 A+B、A+C、B+C、A+B+C 等多种组合。Paper 这里证据单薄。

## 我读这篇 paper 的几个疑问

### 疑问一：仍然是 pipeline

虽然 paper 反复强调"not a pipeline"，但架构上仍是：
$$\text{VLM} \xrightarrow{H_t} \text{CoT} \xrightarrow{z} \text{DiT} \xrightarrow{\hat{v}, \hat{a}}$$

这本质上还是 sequential pipeline，只是 z 是 bottleneck、embedder tying 让 coordinate 统一、joint loss 让 gradient 双向 flow。真正的"unified"可能需要 VLM 和 DiT 的 attention 完全 shared、CoT 和 video/action token interleaved 在同一个 sequence 里联合生成。Pelican-Unify 1.0 算"tightly coupled pipeline"。

话说回来，工程上能 work 的往往就是这种 tight coupling，完全 unified 在 optimization 上 unstable。

### 疑问二：关键工程细节缺失

Paper 没说：
- z 的 dimension 是多少
- $P_\phi$ 是 linear projection 还是 MLP
- VLM 和 DiT 之间是否有 stop-gradient（如果 z 同时受 text 和 video/action gradient，VLM 的 representation 会被 action loss heavily modify，可能伤害 language 能力）
- VLM 在生成 CoT 时是否 stop-gradient 到 DiT
- DiT 的 Wan2.2 初始化是用了 pretrain weight 还是只借架构

这些细节决定实际可复现性。

### 疑问三：loss weight λ 的 ablation 完全缺失

Eq. 14：$\mathcal{L} = \lambda_{\text{text}}\mathcal{L}_{\text{text}} + \lambda_{\text{video}}\mathcal{L}_{\text{video}} + \lambda_{\text{action}}\mathcal{L}_{\text{action}}$

Paper 没给 $\lambda_{\text{text}}, \lambda_{\text{video}}, \lambda_{\text{action}}$ 具体值，也没说怎么调。三个 loss scale 差异巨大：
- Text NLL per token ~ 1-5
- Video MSE per pixel ~ 0.1-1
- Action SmoothL1 ~ 0.01-0.1（action dim 通常 7-30）

不调 balance 很可能某个 loss dominate。这是 paper 最大工程细节缺失。

### 疑问四：z 是否真有 bottleneck 效果

Paper 说 z 是 "only interface"，但没做 ablation：
- 如果让 DiT 直接拿 VLM 的 final hidden state（高 dim，无 bottleneck），performance 会怎样
- 如果 z dim 非常小（比如 32），会 collapse 吗
- z 里 encode 了什么信息（用 probing 实验）

### 疑问五：三能力"互相帮助"的 causal 证据缺失

Paper 说 "shared reasoning and predictive representations improve both generalizability and robustness"，但没做 ablation：
- 只 train text+action（无 video）→ action performance 会怎样
- 只 train text+video（无 action）→ video quality 会怎样
- 只 train action（无 text 无 video）→ 还能达到 93.5% 吗

如果第三种情况 action performance 也接近 93.5%，那 "unification 带来 integrated behavior" 的 claim 就弱了——说明 action 能力本来就来自 action data，video 和 text 只是锦上添花。

## 相关联想

### Predictive coding / Free energy principle

Paper 引用 Clark [9], Friston [16], Hesslow [19], Jeannerod [21] 这些 cognitive science 工作。Friston 的 free energy principle 说大脑本质在做 predictive coding——大脑不断 predict 感官输入，最小化 prediction error。Hesslow 的 simulation theory 说我们通过模拟行为和感知来思考。Jeannerod 说 motor planning 招募了运动模拟系统。

Pelican-Unify 把这些理论落到了工程上：reasoning + imagination + action 在同一个 model 里 co-evolve，正是 embodied cognition 的核心思想。
- Friston free energy: https://www.nature.com/articles/nrn2787
- Clark predictive brain: https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/whatever-next-predictive-brains-situated-agents-and-the-future-of-cognitive-science/

### 和 LLM embedding tying 的类比

Pelican-Unify 的 tied embedder $\mathcal{E}_v, \mathcal{E}_a$ 让 reader (VLM) 和 writer (DiT) 共享 token space，这跟 LLM 里 input embedding 和 output embedding tying 是同一个 idea。LLM 里这招让 gradient 双向流动，让 representation 更 efficient。Pelican-Unify 把这招用到 video 和 action embedding 上，效果类似——representation 被 read 和 write 两个 task 同时 shape。

### Information bottleneck 思想

z 作为唯一 interface，强迫 reasoning 信息必须通过低 bandwidth 通道传给 generation。这跟 Tishby 的 information bottleneck 思想一致——压缩 bandwidth 反而逼出 abstraction。
- Tishby IB paper: https://arxiv.org/abs/1703.00310

### Multi-task learning 的scale-up

Pelican-Unify 在三个 benchmark 上同时第 1 或第 2，证明 joint training 让每个能力都强。这背后的 deep reason 是：当 model 必须用同一个 z 去 condition 三个 head 时，z 被迫学到"对所有三个任务都有用"的 feature，这种 feature 比 single-task expert 学到的更 abstract、更 transferable、更 robust。

这是 multi-task learning 的 classic observation (Caruana 1997) 在 embodied foundation model 上的复现，但 scale 和 capability range 是前所未有的。
- Caruana multitask learning: https://link.springer.com/chapter/10.1007/3-540-44674-6_7

### 对比 MotuBrain / Motus

Motus [4] 提出 unified latent action world model，MotuBrain [39] 是 scale-up 版。这是 Pelican-Unify 最直接竞品。Motus 系列没显式 language reasoning（无 CoT），所以"unification"定义不同——Motus 的 unification 是 video+action，Pelican-Unify 的 unification 是 text+video+action。
- Motus paper: https://arxiv.org/abs/2506.21485 (估计号)

### 对比 Gemini Robotics ER

DeepMind 的 Gemini Robotics ER [37] 强调 reasoning。Pelican-Unify 在 VLM benchmark 上和它比较合适但 paper 里没直接比。Gemini Robotics 的技术细节公开较少。
- https://arxiv.org/abs/2503.20020
- https://deepmind.google/technologies/gemini/robotics/

### 对比 Helix

Figure AI 的 Helix [15] 把 VLM 和 action policy 完全分离但 joint train。Pelican-Unify 的 benchmark 里没放 Helix 数字（可能没公开）。
- https://www.figure.ai/news/helix

## 这篇 paper 真正的 contribute 了什么

最重要的不是 architecture——架构是 existing components 的巧妙组合（Qwen3-VL + CoT + z projection + Wan2.2 DiT + tied embedder + joint flow matching）。**最重要的是实证证明了 unification 不牺牲 specialist**。

历史上大家一直怀疑"joint training 会让每个能力都变弱"——VLM 加 action head 后 VLM benchmark 会掉，加 video head 后 pixel quality 会掉，加 CoT 后 action accuracy 会掉。Pelican-Unify 在三个 benchmark 上同时第 1 或第 2，证明这个 intuition 错了：**joint training 反而让每个能力都强**。

下一个值得期待的方向：
- 更激进 unification（VLM 和 DiT 完全 share attention）
- z 的 interpretability（probing z 里 encode 了什么）
- Cross-embodiment unification（同一个 model 同时 control UR5e、Tienkung、不同 gripper）
- Physics-aware loss（解决 Physical Plausibility 1.23 这个相对弱点）
- Long-horizon composition 的 systematic evaluation
- Counterfactual imagination（imagine "如果 action 是 B 而不是 A 会怎样"）

Paper 结尾的 coda 写得很好——"We do not claim general embodied intelligence. We claim something more specific"。2026 年这个 embodied AI hype 的语境下，这种克制很清醒。Pelican-Unify 1.0 不是终点，但它给了 unification paradigm 第一个 strong empirical foothold。

---

## 相关链接汇总

- X-Humanoid: http://www.x-humanoid.com/
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- RT-2: https://robotics-transformer2.github.io/
- OpenVLA: https://openvla.github.io/
- Helix: https://www.figure.ai/news/helix
- Gemini Robotics: https://deepmind.google/technologies/gemini/robotics/
- Wan: https://github.com/Wan-Video/Wan2.1
- Cosmos: https://www.nvidia.com/en-us/ai/cosmos/
- WorldArena: https://arxiv.org/abs/2602.08971
- RoboTwin: https://github.com/TianxingChen/RoboTwin
- ECoT: https://arxiv.org/abs/2503.00038
- Motus: https://arxiv.org/abs/2506.21485
- Friston free energy: https://www.nature.com/articles/nrn2787
- Tishby information bottleneck: https://arxiv.org/abs/1703.00310
- Caruana multitask learning: https://link.springer.com/chapter/10.1007/3-540-44674-6_7
- Lipman flow matching: https://arxiv.org/abs/2210.02747
- Qwen3-VL (估计): https://arxiv.org/abs/2511.21631

---

# Pelican-Unify 1.0：把Understanding、Reasoning、Imagination、Action压进同一个training loop

## 1. 这篇paper在吵什么

整个embodied AI社区现在有四种specialist：VLM (Gemini Robotics ER, Pelican-VL)只负责"看懂"，VLA (RT-2, π0, π0.5, OpenVLA, Helix)只负责"映射到action"，World Model (Cosmos-Predict, LeWorldModel)只负责"imagine future pixels"，World Action Model (WAM类)只负责"想象+action但无language reasoning"。每个specialist都很强，但interface处的representation是断的：VLM说"我应该拿起那个红色插头"，VLA却不知道红色插头意味着什么物理后果；World Model会imagine一个像素级future，但这个future没被task logic steer。

Pelican-Unify的claim是：**这四个能力应该在一个shared representation里co-evolve**。所谓"unification"他们反复强调三点——structurally shared representations、mutually constrained conditions、co-evolution through a common training process。注意这是paper的原文反复强调的，作者特别怕读者把"unified"理解成把几个expert的output拼起来或者stitching一个pipeline。

更精确地说，他们定义了三个property：

- **Unified understanding**：scene、instruction、action history、visual context全部embed进同一个semantic space。这就避免了"perception module输出embedding，policy module又重新encode一遍"的semantic break。
- **Unified reasoning**：reasoning不是脱离action的language monologue，而是language-grounded, supervisable process，它的final state直接condition下游要generate什么future、要execute什么action。也就是说CoT的最后一个hidden state就是downstream的condition。
- **Unified generation**：future video和action从**同一个denoising process**、**同一个condition z**联合生成。这意味着你imagine的future和你execute的action是同一个stochastic process的两面。

这个claim本身不新（embodied cognition、predictive coding、free energy principle都说过类似的话），但把它做成一个end-to-end trainable、并且实验上证明"unification不牺牲specialist能力"是新的工程实证。

## 2. 整体架构

整个model是composite map (Eq. 1):

$$(\tau_t, \hat{v}_{t:t+H}, \hat{a}_{t:t+H}) = \mathcal{M}_\Theta(a_{<t}, l, o_{\leq t}, s_{\leq t})$$

变量含义：
- $\tau_t$：time step $t$ 的chain-of-thought trace（一段natural language reasoning）
- $\hat{v}_{t:t+H}$：从 $t$ 到 $t+H$ 的imagined future video，$H$ 是chunk horizon
- $\hat{a}_{t:t+H}$：同期要执行的action chunk
- 输入端：$a_{<t}$ action history、$l$ language instruction、$o_{\leq t}$ observation sequence、$s_{\leq t}$ robot proprioceptive state

这个map由三段组成，但参数 $\Theta$ 是共享的：

```
[obs, action hist, language, state] 
      ↓ shared embedders E_v, E_a
      ↓
[VLM: Qwen3-VL backbone]  → H_t (Eq.3)
      ↓ autoregressive decode
[CoT trace τ_t] (Eq.4) interleaves "video CoT" + "action CoT"
      ↓ take last hidden state h_{τ_t}
[projector P_φ] (Eq.5)
      ↓
[dense latent z]  ← 这是唯一的"reasoning→generation"接口
      ↓
[Unified Future Generator: Wan2.2 DiT + two heads]
      ↓ joint denoising with shared diffusion time s
[future video v̂]  +  [action chunk â]
```

这里有几个非常聪明的engineering选择我先讲intuition再讲公式。

### 2.1 Tied embedders的妙处

注意paper里专门命名了 $\mathcal{E}_v$ (3D video VAE) 和 $\mathcal{E}_a$ (轻量MLP)，它们在VLM的input side被用来把video frames和action history lift进token space，**同时**在UFG的input side被reuse去embed diffusion的clean targets $x^v$ 和 $x^a$ (Eq. 6)：

$$x^v = \mathcal{E}_v(v_{t:t+H}), \quad x^a = \mathcal{E}_a(a_{t:t+H})$$

这意味着"VLM学会read的坐标系"和"DiT学会denoise的坐标系"是同一个。这里没有alignment adapter、没有projection layer、没有contrastive alignment loss。Gradients从video loss和action loss会直接flow back到 $\mathcal{E}_v, \mathcal{E}_a$，再flow到VLM。这就是paper反复说的"shared representation"，它具体落实在embedder tying这一招上。

我个人觉得这是整个architecture里最巧的一招。可以类比LLM里input embedding和output embedding tying——同样的token space既被reader使用也被writer使用，gradient天然双向流动。

### 2.2 z 是唯一接口

paper强调 $z = P_\phi(h_{\tau_t})$ (Eq.5) 是"the only interface through which downstream future generation accesses the model's understanding and reasoning"。这是个很关键的设计：downstream generation看不到原始的 $o_{\leq t}$、$a_{<t}$、$l$，只看到经过VLM full processing + CoT projection之后的 $z$。

为什么要这样设计？因为如果DiT直接拿到raw observation embedding，那observation和reasoning就是两条平行通路，无法保证reasoning真的shape了generation。把z变成瓶颈，就强迫reasoning必须encode所有generation需要的信息——video head需要预测未来frame，那z必须包含object motion、contact formation；action head需要预测joint trajectory，那z必须包含end-effector waypoint、subskill decomposition。语言loss、video loss、action loss三方都对z求梯度，z被迫成为"semantic + predictive + actionable"的intersection。

这有点像information bottleneck的思路：你压缩bandwidth，反而逼出abstraction。

### 2.3 CoT是interleaved的

paper里讲CoT trace $\tau_t$ 包含两种language：
- **Video CoT**：描述场景应该怎么evolve（"哪个object会move，contact怎么形成，workspace怎么reorganize"）
- **Action CoT**：描述motor program（"调用哪个subskill，end-effector应该target哪个waypoint"）

把两种放进同一个sequence、同一个causal pass里，意味着model必须先想"会发生什么"再想"该做什么"——这其实是在用autoregressive language modeling这个工具来强制一个causal order：先imagine再plan再act。

这一步让我想起ECoT (Embodied Chain-of-Thought, Zawalski et al. 2025) [51] 和MolmoAct [26] 的工作，但Pelican-Unify更进一步把CoT的terminal state直接condition diffusion，而ECoT只把CoT当reasoning scaffold，最后还是用一个flat action head。

## 3. Conditional Diffusion的细节

这是paper技术含量最高的部分，我详细拆。

### 3.1 Action stream：标准flow matching

Eq. 8:
$$x_s^a = (1-s)x^a + s\epsilon^a, \quad u_s^a = \epsilon^a - x^a$$

变量：
- $s \sim \mathcal{U}(0,1)$：diffusion time，均匀采样。$s=0$对应clean latent $x^a$，$s=1$对应pure noise $\epsilon^a$。
- $x^a = \mathcal{E}_a(a_{t:t+H})$：clean action latent（已经过MLP encoder）
- $\epsilon^a \sim \mathcal{N}(0, I)$：标准Gaussian noise
- $x_s^a$：noised latent at time $s$
- $u_s^a$：target velocity field，方向是从clean $x^a$ 指向noise $\epsilon^a$

注意 $(1-s)x + s\epsilon$ 是从clean到noise的**linear interpolation**，对应probability flow ODE $dx_s = (\epsilon - x) ds$，velocity field就是 $\epsilon - x$。这是flow matching（Lipman et al. 2023）的标准形式，和DDPM的 $\sqrt{\bar{\alpha}}x + \sqrt{1-\bar{\alpha}}\epsilon$ 不同——这里没用variance schedule，就是最简单的线性path。

DiT预测 $\hat{u}_s^a = f_\theta(\cdot)_a$，loss在Eq. 12是Smooth L1：
$$\mathcal{L}_{\text{action}} = \mathbb{E}_{s, \epsilon^a}[M_a \odot \text{SmoothL1}(\hat{u}_s^a, u_s^a)]$$

$M_a$ 是valid action dimension mask——比如7-DOF arm + 2-DOF gripper = 9维，但有些维度在某些task里无效（比如gripper在pour task里不需要预测）。这个mask让loss只在valid dimension上累加。

### 3.2 Video stream：prefix-conditioned flow matching

这是paper比较有创意的部分。video不仅要预测future，还要condition在已观察到的prefix $o_{\leq t}$ 上。

Eq. 9:
$$x_s^v = M_{\text{cond}} \odot x^v + M_{\text{fut}} \odot ((1-s)x^v + s\epsilon^v)$$

Eq. 10:
$$u_s^v = M_{\text{fut}} \odot (\epsilon^v - x^v)$$

变量：
- $x^v$：包含prefix和future两段的combined video latent，每一段都已经被 $\mathcal{E}_v$ encode过
- $M_{\text{cond}}$：binary mask，1的位置对应prefix frames
- $M_{\text{fut}}$：binary mask，1的位置对应future frames，且 $M_{\text{cond}} + M_{\text{fut}} = \mathbf{1}$
- $\epsilon^v$：仅在future区域采样的Gaussian noise

直观理解：**prefix frames保持clean，future frames被noise，DiT要denoise的是future部分**。这相当于把video generation变成了inpainting任务——已知过去，预测未来。

这里一个非常聪明的设计是：因为 $o_{\leq t}$ 也是用 $\mathcal{E}_v$ encode的（同一个tied embedder），prefix latent和future latent在同一个token space，DiT天然就能看懂prefix、不需要额外的condition encoder。这跟RT-1早期那种"observation encoder + action decoder"完全分离的设计形成鲜明对比。

Eq. 11的video loss也只在future区域计算：
$$\mathcal{L}_{\text{video}} = \mathbb{E}_{s, \epsilon^v}[\|M_{\text{fut}} \odot (\hat{u}_s^v - u_s^v)\|_2^2]$$

### 3.3 Joint denoising backbone

Eq. 7:
$$(\hat{u}_s^v, \hat{u}_s^a) = f_\theta(x_s^v, x_s^a, z, s)$$

$f_\theta$ 是DiT (Wan2.2初始化) + 两个lightweight heads $d_v, d_a$。关键点paper里讲了：

- video tokens和action tokens通过**shared self-attention**交互——这意味着action token能attend到video token、video token能attend到action token，反过来也是。这就实现了"action shape想象到的consequence、imagination shape action的执行"。
- $z$ 通过**cross-attention**注入——reasoning summary持续shape整个denoising trajectory的每一步。
- modality-specific参数只在input boundary（$\mathcal{E}_v, \mathcal{E}_a$）和output boundary（$d_v, d_a$），中间的heavy computation（几十层transformer block）是shared的。

这个shared self-attention是真正的"unification"机制——action token和video token在每一层都互相看到。这是和MolmoAct、π0.5那种"action是VLM最后接一个flow head"的本质区别。

### 3.4 三loss联合

Eq. 13: $\mathcal{L}_{\text{text}} = -\sum_i \log p_\phi(\tau_{t,i} | c_t, \tau_{t,<i})$ ——标准autoregressive NLL。

Eq. 14:
$$\mathcal{L} = \lambda_{\text{text}}\mathcal{L}_{\text{text}} + \lambda_{\text{video}}\mathcal{L}_{\text{video}} + \lambda_{\text{action}}\mathcal{L}_{\text{action}}$$

三个loss都flow back through z（video和action通过cross-attention gradient、text通过NLL gradient），也都flow back through $\mathcal{E}_v, \mathcal{E}_a$（tied embedder）。这就是paper说的"single training loop"——不是一个pipeline里串三个trainer，是三个loss同时优化同一个 $\Theta$。

paper没给 $\lambda$ 具体值，这是一个open question——三个loss scale差异很大（text NLL per token大概是1量级，video MSE per pixel也是1量级但token数远少于text token数，action SmoothL1非常小），不调好balance很容易dominate。这块的ablation完全缺失。

## 4. 实验：拆开看specialist能力

paper的设计哲学是：先证明"unification不牺牲specialist"，再证明"unification带来integrated behavior"。所以分了三组evaluation。

### 4.1 Understanding：8个VLM benchmark

Table 1的关键数字：

| Model | Avg (8 benchmarks) |
|---|---|
| OpenVLA | 3.3 |
| MolmoAct | 27.5 |
| π0.5 | 10.2 |
| Gemma3-4B-IT | 32.9 |
| **Qwen3-VL-4B-Instruct** (base) | 58.2 |
| **Pelican-Unify 1.0** | **64.7** |

从Qwen3-VL-4B的58.2提升到64.7，**+6.5**。但是细看：

- General benchmark（MMMU, MMBench, MMStar, InfoVQA, ChartQA）：基本持平或微涨。MMMU从52.6→53.0，MMBench 84.5→84.9，没有退化——这是关键，说明加action/video loss没"污染"VLM的核心能力。
- Embodied benchmark（Where2Place, PhyX, RefSpatial）：大幅提升。Where2Place 17.0→45.2 (**+28.2**)，PhyX 41.1→61.7 (**+20.6**)，RefSpatial 48.0→49.3。

这个分布很有意思：general benchmark没动，embodied benchmark暴涨。说明unification training学到的是**物理grounding**而非通用reasoning。这个gain的来源我推测是action history embedding和action loss——VLM被迫理解"这个动作会让scene怎么变"，这种causal理解transfer到了Where2Place（哪里可以放东西）和PhyX（物理直觉）这类任务。

paper里的Figure 1 visualization也支持这个解读：standard VLA training会让attention map发散、grounding能力下降；Pelican-Unify保留了grounding同时还能predict action。

### 4.2 Action：RoboTwin 50-task

Table 2关键数字：

| Model | Clean | Randomized | Avg |
|---|---|---|---|
| π0 | 65.9 | 58.4 | 62.2 |
| π0.5 | 82.7 | 76.8 | 79.8 |
| starVLA | 88.2 | 88.3 | 88.3 |
| LingBot-VA | 92.9 | 91.6 | 92.3 |
| AIM | 94.0 | 92.1 | 93.1 |
| **MotuBrain** | **95.8** | **96.1** | **95.9** |
| **Pelican-Unify 1.0** | 93.6 | 93.3 | **93.5** |

Pelican-Unify排第二，**落后MotuBrain 2.4个百分点**。Clean和Randomized之间几乎无差（93.6 vs 93.3），说明对随机扰动鲁棒。但需要注意的是：

- MotuBrain [39] 本身也是unified world action model (来自Motus团队)，是Pelican-Unify最直接的竞品。
- Pelican-Unify的优势是**同时**还是VLM第1和WorldArena第1，而MotuBrain没在VLM benchmark上比较。
- paper特意标注"second-best average success rate among compared methods"——他们承认不是第一。

我注意到一个细节：RoboTwin的failure集中在"long-horizon或geometry-sensitive task"如hanging mugs和dustbin insertion。这暗示unification在long-horizon上确实有提升（这是paper后面real robot实验要证明的），但在tight几何对齐上并没有比specialist VLA强——因为这种任务本质是几何精度问题，不是reasoning能救的。

### 4.3 Imagination：WorldArena

Table 3关键数字：

| Model | EWM Score | Rank |
|---|---|---|
| **Pelican-Unify** | **66.03** | **1** |
| WorldScape v0.2 | 64.24 | 2 |
| FlowWAM-FiveAges | 64.12 | 3 |
| MotuBrain | 64.07 | 4 |
| Wan2.6 | 59.80 | 12 |
| Veo3.1 | 57.77 | 15 |

比第2名WorldScape高1.79分。但更interesting的是细项：

- **3D Accuracy: 98.13**（第1）——空间几何一致性最强
- **Motion Quality: 62.69**（第1）——运动学合理性最强
- Visual Quality 63.43（中等）
- Content Consistency 60.33（中等）
- Physics Adherence 61.51（中等）
- Controllability 59.28（中等）

这个pattern很说明问题：Pelican-Unify在**3D几何和运动学**上遥遥领先，但在**pixel-level visual quality和physics fidelity**上只是中等。这是合理的——它的DiT从Wan2.2初始化，但被robot data heavily fine-tune，所以pixel generation能力比纯video model弱；但robot data里包含大量真实3D结构和kinematic约束，所以3D accuracy和motion quality强。

Human eval (Table 4)更说明问题：

| Model | Task Success | Controllability | Temporal Consistency | Physical Plausibility | Avg |
|---|---|---|---|---|---|
| **Pelican-Unify** | **1.81** | **2.00** | 2.00 | 1.23 | **1.76** |
| Seedance2.0 (API) | 1.21 | 1.87 | 1.98 | 1.15 | 1.55 |
| Happyhorse-1.0 | 1.65 | 1.81 | 2.00 | 0.13 | 1.40 |
| EnerVerse-AC | **0.00** | 1.84 | 2.00 | 1.64 | 1.37 |
| Wan2.7 | 1.19 | 1.68 | 2.00 | 0.29 | 1.29 |

**Task Success 1.81**（满分2.0）是最关键的——其他video diffusion model（Happyhorse, Wan2.7, Cosmos-Predict2）的temporal consistency都是满分2.0，但Task Success很低，说明它们生成的video视觉上流畅但**任务上跑题了**。EnerVerse-AC甚至Task Success=0.00但Temporal Consistency=2.00，这是"漂亮的废话"的典型。

这恰好印证了paper的核心论点：**纯video model optimize的是pixel fidelity，但embodied需要的是task-relevant future imagination**。Pelican-Unify通过让action loss和text loss同时flow到z，让generated video必须既imagination-plausible又action-consistent又task-relevant。

Physical Plausibility 1.23 是Pelican-Unify相对弱的一项——还是不如EnerVerse-AC的1.64。这说明robot data里物理约束编码还不够强，未来可能需要physics engine supervision或者contact-aware loss。

## 5. Real Robot：Compositional Generalization

这个实验设计很巧妙。在UR5e上，atomic task A=plug RJ45、B=waterproof，分别训练，**训练数据里完全没有A+B的chained demonstration**。Test time给一个instruction "plug RJ45 cable into port 3 and apply waterproofing"，model要在同一episode里完成A→B。

paper说failures集中在A→B的transition——"the moment where the just-completed A-state must be re-perceived as the new initial condition for B"。VLA baselines失败的原因不是re-perception，而是"action distributions carry no representation of what should happen after A is done"。

这是个非常clean的test，因为它直接验证了"imagination face的render post-A scene state and re-condition on it"——也就是说model在transition处会先imagine A完成后的scene应该长什么样，然后把这个imagined state当作B的initial condition。

但是，**只有2个atomic task的composition，规模太小**。A+B只有一种combination。真正的compositional generalization应该测A+B、A+C、B+C、A+B+C等更多组合。paper这里给的证据单薄了。

Figure 4展示了real execution video和imagined video的对比——这是个qualitative claim，paper说"the model does not merely hallucinate plausible scenes, but instead conditions its predictions on actual environment dynamics"。我觉得这个claim需要更quantitative的evaluation，比如imagined frame和real frame之间的feature similarity。

## 6. Zero-shot Transfer

Tienkung humanoid上，5个seen task（每个~300 video-action episode）+3个unseen task（每个只有50 video sequences）joint training。Table 4证明zero-shot generalization在human eval上1.76分（最高）。

但这里有个需要警惕的细节：unseen task只用了50个video sequence，没有action label（"3 unseen tasks, which were provided with only 50 video sequences per task"）。这意味着zero-shot transfer是**video-only supervision transfer to action**——也就是unseen task的action是靠imagination face从seen task学到的action policy transfer过来的。这是unification paradigm的天然能力：你看到video，z就能infer合理的action，因为video和action在同一个DiT里co-trained。

这其实是一个很强的claim，但paper没有给ablation——如果只用VLA baseline在unseen task上做in-context demo会怎么样？如果用MotuBrain做同样实验呢？

## 7. 我对这篇paper的critical reading

### 7.1 "Unification"的定义仍然是个pipeline

虽然paper反复强调"not a pipeline"，但实际架构上仍然是：

$$\text{VLM} \xrightarrow{H_t} \text{CoT} \xrightarrow{z} \text{DiT} \xrightarrow{\hat{v}, \hat{a}}$$

这本质上还是个sequential pipeline，只是：
- z是bottleneck而非full hidden state（所以reasoning被compress）
- embedder tying让coordinate system统一
- joint loss让gradient双向flow

真正的"unified"可能需要更激进的设计——比如VLM和DiT的attention完全shared、CoT和video/action token interleaved在同一个sequence里联合autoregressive+diffusion生成。Pelican-Unify 1.0只能算"tightly coupled pipeline"。但话说回来，工程上能work的往往就是这种tight coupling，完全unified可能在optimization上unstable。

### 7.2 z的dimension、architecture细节缺失

paper没说：
- z的dimension是多少
- P_φ是linear projection还是MLP
- VLM和DiT之间是否有stop-gradient（如果z同时受text和video/action gradient，VLM的representation会被action loss heavily modify，可能伤害language能力）
- VLM在生成CoT时是否stop-gradient到DiT
- DiT的Wan2.2初始化是用了pretrain weight还是只借了架构

这些细节决定了实际可复现性。

### 7.3 Loss weight λ的ablation完全缺失

paper没给 λ_text, λ_video, λ_action 的具体值，也没说怎么调的。三个loss的scale差异巨大：
- Text NLL per token ~ 1-5
- Video MSE per pixel ~ 0.1-1（但pixel数远多于text token）
- Action SmoothL1 ~ 0.01-0.1（action dim通常7-30）

不调balance很可能某个loss dominate。这是paper最大的工程细节缺失。

### 7.4 z是否真的有information bottleneck效果

paper说z是"only interface"，但没做ablation：如果让DiT直接拿VLM的final hidden state（高dim，无bottleneck），performance会怎么样？如果z dim非常小（比如32），会collapse吗？z里encode了什么信息（用probing实验）？

### 7.5 Compositional generalization的规模

只有2个atomic task的1种combination。我期待看到：
- N个atomic task → N(N-1)/2种pair组合的systematic测试
- Long-horizon chain (A+B+C)
- Counterfactual composition (训练A+B和C+D，测试A+C)

### 7.6 三个能力"互相帮助"的causal证据缺失

paper说"shared reasoning and predictive representations improve both generalizability and robustness"，但没做ablation：
- 只train text+action（无video）→ action performance会怎样？
- 只train text+video（无action）→ video quality会怎样？
- 只train action（无text无video）→ 还能达到93.5%吗？

如果第三种情况action performance也接近93.5%，那"unification带来integrated behavior"的claim就弱了——说明action能力本来就来自action data，video和text只是锦上添花。

## 8. 联想到的相关工作

### 8.1 ECoT和reasoning supervision

ECoT [51] (Zawalski et al. 2025) 是第一个systematic study embodied CoT的工作。Pelican-Unify可以看作ECoT的"next step"——ECoT只在action supervision上加CoT，Pelican-Unify把CoT变成z的来源condition下游generation。可以读：
- https://arxiv.org/abs/2503.00038 (ECoT paper)

### 8.2 π0 / π0.5：flow matching for action

π0 [5] 用flow matching做action chunk生成，π0.5 [20] 扩展到open-world。Pelican-Unify的action stream基本是π0的flow matching，但加了video joint denoising和z conditioning。可以读：
- https://arxiv.org/abs/2410.24164 (π0)
- https://arxiv.org/abs/2504.16054 (π0.5)
- https://www.physicalintelligence.company/blog/pi0

### 8.3 Helix (Figure AI)

Helix [15] 是Figure AI的VLA，把VLM和action policy完全分离但joint train。Pelican-Unify的对比benchmark里没放Helix的数字（可能没公开）。Helix blog：
- https://www.figure.ai/news/helix

### 8.4 Gemini Robotics ER

Gemini Robotics ER [37] 是DeepMind的embodied VLM，强调reasoning能力。Pelican-Unify在VLM benchmark上和它比较合适但paper里没直接比。可以读：
- https://arxiv.org/abs/2503.20020
- https://deepmind.google/technologies/gemini/robotics/

### 8.5 Cosmos-Predict

NVIDIA的Cosmos [1, 2] 是world foundation model。Pelican-Unify的UFG基本是Cosmos-style video diffusion + action head。可以读：
- https://arxiv.org/abs/2511.00062
- https://www.nvidia.com/en-us/ai/cosmos/

### 8.6 MotuBrain / Motus

Motus [4] 提出unified latent action world model，MotuBrain [39] 是其scale-up版本。这是Pelican-Unify最直接的竞品，但Motus系列没有显式language reasoning（无CoT），所以"unification"定义不同。可以读：
- https://arxiv.org/abs/2506.21485 (Motus, 大概这个号)

### 8.7 World Arena

WorldArena [35] 是paper里imagination能力evaluation用的benchmark：
- https://arxiv.org/abs/2602.08971

### 8.8 RoboTwin

RoboTwin是dual-arm benchmark：
- https://github.com/TianxingChen/RoboTwin

### 8.9 Qwen3-VL

Pelican-Unify的VLM backbone：
- https://arxiv.org/abs/2511.21631 (估计的号)

### 8.10 Wan2.2 / Wan2.6 / Wan2.7

Wan是video generation model系列，Pelican-Unify的UFG初始化自Wan2.2：
- https://github.com/Wan-Video/Wan2.1
- https://arxiv.org/abs/2503.20314 (Wan technical report)

### 8.11 RT-2

RT-2 [55] 是最早的VLA之一，把VLM直接co-finetune到action token：
- https://robotics-transformer2.github.io/
- https://arxiv.org/abs/2307.15818

### 8.12 OpenVLA

OpenVLA [25] 是开源VLA：
- https://openvla.github.io/
- https://arxiv.org/abs/2406.09246

### 8.13 Predictive coding / Free energy principle

paper引用了Clark [9], Friston [16], Hesslow [19], Jeannerod [21] 等cognitive science工作，这是"unification"的theoretical foundation：
- Friston free energy: https://www.nature.com/articles/nrn2787
- Clark predictive brain: https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/whatever-next-predictive-brains-situated-agents-and-the-future-of-cognitive-science/

## 9. 总结：这篇paper真正contribute了什么

我觉得这篇paper最重要的不是architecture（架构是existing components的巧妙组合：Qwen3-VL + CoT + z projection + Wan2.2 DiT + tied embedder + joint flow matching），而是**实证证明了unification不牺牲specialist**。

历史上，大家一直怀疑"joint training会让每个能力都变弱"——VLM加了action head后VLM benchmark会掉，加了video head后pixel quality会掉，加了CoT后action accuracy会掉。Pelican-Unify在三个benchmark上同时第1或第2，证明了这个intuition是错的：**joint training反而让每个能力都强**，因为shared representation强制了mutual information maximization。

这背后的deep reason我推测是：当model必须用同一个z去condition text、video、action三个head时，z被迫学到"对所有三个任务都有用"的feature，这种feature比任何一个single-task expert学到的feature更abstract、更transferable、更robust。这是multi-task learning的classic observation在embodied foundation model上的复现，但scale和capability range是前所未有的。

下一个值得期待的direction：
- 更激进的unification（VLM和DiT完全share attention）
- z的interpretability（probing z里encode了什么）
- Cross-embodiment unification（同一个model同时control UR5e、Tienkung、不同的gripper）
- Physics-aware loss（解决Physical Plausibility 1.23这个相对弱点）
- Long-horizon composition的systematic evaluation
- Counterfactual imagination（imagine "如果action是B而不是A会怎样"）

paper结尾的coda写得很好——"We do not claim general embodied intelligence. We claim something more specific"。这种克制在2026年这个embodied AI hype的语境下显得很清醒。Pelican-Unify 1.0不是终点，但它给了unification paradigm第一个strong empirical foothold。

---

相关链接汇总：
- Paper PDF (估计): https://arxiv.org/abs/2026.xxxxx (paper里没给arxiv号)
- X-Humanoid: http://www.x-humanoid.com/
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- RT-2: https://robotics-transformer2.github.io/
- OpenVLA: https://openvla.github.io/
- Helix: https://www.figure.ai/news/helix
- Gemini Robotics: https://deepmind.google/technologies/gemini/robotics/
- Wan: https://github.com/Wan-Video/Wan2.1
- Cosmos: https://www.nvidia.com/en-us/ai/cosmos/
- WorldArena: https://arxiv.org/abs/2602.08971
- RoboTwin: https://github.com/TianxingChen/RoboTwin
- ECoT: https://arxiv.org/abs/2503.00038
- Motus: https://arxiv.org/abs/2506.21485
- Friston free energy: https://www.nature.com/articles/nrn2787
