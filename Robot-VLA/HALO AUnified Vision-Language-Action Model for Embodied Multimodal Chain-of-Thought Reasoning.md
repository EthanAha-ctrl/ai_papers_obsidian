---
source_pdf: HALO AUnified Vision-Language-Action Model for Embodied Multimodal Chain-of-Thought
  Reasoning.pdf
paper_sha256: 89a5c5cd60690439d49531bf16ae0ce32270e997bb35570dd52578076153a06d
processed_at: '2026-08-04T23:23:14-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

用更接地气的方式给你 build 一下这篇 paper 的 intuition。

我们目前做机器人的 VLA models，比如 $\pi_0$ 或者 OpenVLA，本质上都在做条件反射。模型看到一张图、听到一句话，直接输出机械臂该怎么动。这种 stimulus-response 的范式在实验室里很好用，一旦遇到 long-horizon 任务（比如“打开抽屉，把柠檬放进去，关上抽屉”）或者 out-of-distribution 的场景（比如桌子背景突然变了，或者杯子换了个形状），模型直接懵逼。因为它没有“想一想”的机制，完全靠 pattern matching 硬套。

人类做事的逻辑完全不同。你看到柠檬和抽屉，脑子里会先过一遍步骤：“我得先抓住柠檬，然后移动到抽屉上方，松手”。接着你会在脑子里“想象”一下：手抓着柠檬悬在抽屉上方的画面。最后你的手臂才执行这个动作。HALO 这篇 paper 就是把人类这种“think $\rightarrow$ imagine $\rightarrow$ act”的认知过程，硬塞进了一个 unified VLA model 里面，起名叫 EM-CoT (Embodied Multimodal Chain-of-Thought)。

---

### 1. Problem Formulation 的直觉：把大问题拆成三步

传统 VLA 学的是一个端到端的 mapping $\pi_\theta(\mathbf{a}_{t:t+m} \mid \mathbf{l}, \mathbf{o}_{t-k:t})$。输入是语言指令 $\mathbf{l}$ 和过去几帧图像 $\mathbf{o}_{t-k:t}$，输出是 action chunk $\mathbf{a}_{t:t+m}$。

HALO 把这个 joint policy 强行拆成了三个有前后依赖关系的 conditional distributions（公式 1-3）：

$$\mathbf{r} \sim P_\theta(\cdot \mid \mathbf{l}, \mathbf{o}_{t-k:t})$$
$$\hat{\mathbf{o}}_{t+h} \sim P_\theta(\cdot \mid \mathbf{l}, \mathbf{o}_{t-k:t}, \mathbf{r})$$
$$\mathbf{a}_{t:t+m} \sim \pi_\theta(\cdot \mid \mathbf{l}, \mathbf{o}_{t-k:t}, \mathbf{r}, \hat{\mathbf{o}}_{t+h})$$

**变量解释**：
- $\mathbf{l}$: language instruction，比如 "put the lemon into the drawer"
- $\mathbf{o}_{t-k:t}$: 观察到的历史图像帧，$t-k$ 到 $t$ 时刻，论文里 $k=3$
- $\mathbf{r}$: textual chain-of-thought，第一人称的内心独白，比如 "I see a lemon. I need to grasp it first..."
- $\hat{\mathbf{o}}_{t+h}$: 预测的未来第 $t+h$ 帧的视觉 subgoal image
- $\mathbf{a}_{t:t+m}$: 最终要执行的 action chunk，长度为 $m$（sim 环境下 $m=16$）

**Intuition**: 这本质上是一个 Bayesian factorization。直接学 $P(\mathbf{a} \mid \mathbf{o})$ 是极度 ill-posed 的，因为同一个画面可以对应无数种合理动作（从左边抓或者从右边抓）。加上中间的 $\mathbf{r}$ 和 $\hat{\mathbf{o}}$，相当于给 action 提供了极强的 semantic 和 spatial 约束，大大缩小了 action space 的搜索范围。

---

### 2. Architecture 细节：Mixture-of-Transformers 为什么要这么搞

把 text reasoning、image generation、action prediction 塞进一个模型里，最容易踩的坑就是 ManualVLA 那种做法：把图像变成离散 token，跟文字一起 autoregressive 生成。这会严重破坏 VLM 原本的 language reasoning 能力，因为图像 token 的分布和文字 token 差太远了。

HALO 借鉴了 BAGEL 和 Mixture-of-Transformers (MoT) 的思路，搞了三个独立的 expert：

1.  **Multimodal Understanding Expert**: 负责生成文字 $\mathbf{r}$，保持 autoregressive 的天然属性。
2.  **Visual Generation Expert**: 负责生成 subgoal image $\hat{\mathbf{o}}_{t+h}$，用 flow matching / diffusion，处理 continuous latent space。
3.  **Action Prediction Expert**: 负责输出 action $\mathbf{a}$，同样用 flow matching。

这三个 expert 的 FFN 参数完全独立（每个都用 Qwen2.5-1.5B 初始化，总共约 4.5B params），但它们**共享同一套 Self-Attention 层**。这就像三个专家坐在同一个会议室里开会，各自有专业背景，但能互相看到对方在黑板上写什么。

#### Attention Mask 的玄机（Figure 3 解析）
这里最能体现工程细节。因为三种模态的生成逻辑不同，必须设计极其精细的 attention mask：
- **Text tokens**: 严格 causal mask，只能看前面的字。
- **Visual tokens (同一帧内)**: **Bidirectional mask**！因为 diffusion 生成图像时，每个 patch 的 noise 都需要看周围 patch 的 clean 版本来决定怎么去噪。
- **Visual tokens (跨帧或跨模态)**: Causal mask，未来的图像不能泄漏给过去的文字。
- **Noise tokens**: **绝对隔离**。Noise tokens 既不能看 ground truth target，别的 tokens 也不能看 noise tokens。如果不隔离，diffusion 训练时模型直接抄答案，或者在 reasoning 阶段被无意义的 noise 污染上下文。

#### 模态切换
模型默认在 autoregressive 跑文字。当 decoder 吐出 `<vision start>` 这个 special token 时，routing 机制把后续 hidden states 丢给 Visual Generation Expert 处理；遇到 `<action start>` 就切给 Action Expert。

#### 多模态 Encoder 配置
为了把异构数据映射到统一的 hidden dimension：
- **Text**: Qwen2.5 tokenizer。
- **Visual Understanding**: ViT (SigLIP2-so400m/14) + NaViT，处理任意长宽比图像，提取 semantic feature。
- **Visual Generation**: 用 FLUX 的 pre-trained VAE，downsample 8x，latent channel 16。把 2x2 spatial patches flatten 后 linear project 到 LLM 空间。VAE 全程 frozen。
- **Action**: 简单 Linear projection，因为 action 是低维物理量。

---

### 3. EM-CoT Data Pipeline：怎么自动生成 CoT 数据

这是 paper 最 scable 的贡献。你想让模型有 Chain-of-Thought，就得有几百万条带 reasoning 的轨迹数据，人工标注根本不可能。HALO 搞了个全自动流水线（见 Algorithm 1 & 2）：

**Step 1: Action Primitives Extraction (Rule-based)**
把连续的 low-level action（6D pose, gripper state）通过规则切分成离散动作原语。
核心公式函数：`IsIdle(P^a[t], P^a[t-1], G^a[t], G^a[t-1]; θ_vel, θ_dg)`
- $P^a[t]$: 时刻 $t$ 机械臂末端位置
- $G^a[t]$: 时刻 $t$ 夹爪开合状态
- $\theta_{\text{vel}}$: 速度阈值，$\theta_{\text{dg}}$: 夹爪变化阈值
如果位移速度小于 $\theta_{\text{vel}}$ 且夹爪没动，判定为 idle。用 idle 时间是否超过 $\theta_{\text{min.idle}}$ 来切分 motion segment。在 segment 内部，看 $\Delta G$ 判断是 grasp 还是 release，看位移方向 $\Delta P$ 判断是 move_left 还是 move_right。

**Step 2: VLM Annotator (Qwen3-VL)**
如果你直接把一堆 raw coordinates 喂给 VLM，它会 hallucinate。但因为有了 Step 1 的 symbolic primitives，VLM 就能看懂了。
分为三个 prompting 阶段：
1.  **Task Narrative**: 让 VLM 把动作序列写成一段连贯的故事。
2.  **Subtask Decomposition**: 让 VLM 把故事拆成 2-5 个高层 subtask（比如 "Pick up lemon", "Put in drawer"）。
3.  **Subtask Alignment & First-person Reasoning**: 让 VLM 对每段 frame 输出第一人称内心独白，包含 visual observation, goal-driven inference, movement logic，限制在 50 词以内。

**Step 3: Visual Subgoal Extraction (Algorithm 2)**
找每个 subtask 结束的最后一帧 $I_{t_g}$，把这张图作为该 subtask 的 visual subgoal，并 broadcast 给属于这个 subtask 的所有 frames。这提供了一个 sparse 的视觉监督信号，模型不需要预测每一帧怎么变，只需要预测“干完这步世界长什么样”。

---

### 4. Training Recipe 细节与公式

训练分两个 stage。

**Stage 1: Versatile Pre-training**
目标：打基础。混合三类数据：VQA (LLaVA-NeXT-779k), Visual Generation (OXE + SSv2 egocentric videos), Action Prediction (OXE)。
Loss 函数（公式 4）：
$$\mathcal{L}_{\text{pt}} = 0.25\mathcal{L}_{\text{CE}} + 0.5\mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{L1}}$$
- $\mathcal{L}_{\text{CE}}$: VQA 的 cross-entropy loss
- $\mathcal{L}_{\text{MSE}}$: Visual Generation 的 flow-matching MSE loss
- $\mathcal{L}_{\text{L1}}$: Action Prediction 的 $L_1$ flow-matching loss

**Intuition**: 权重比例 CE:MSE:L1 = 1:2:4。Action prediction 最难，因为 continuous 且物理结构敏感，给最高权重；VQA 最简单，给最低权重。SSv2 这种第一人称视频对学习 physical common sense 极其关键，它让模型知道推杯子上部杯子会倒。

**Stage 2: EM-CoT-Augmented Fine-tuning**
目标：注入 CoT 推理能力。使用 Stage 1 产出的 $\mathcal{D}_{\text{ft}}$ 数据集。
Loss 函数（公式 5）：
$$\mathcal{L}_{\text{ft}} = \mathcal{L}_{\mathbf{r}} + \mathcal{L}_{\hat{\mathbf{o}}} + \mathcal{L}_{\mathbf{a}}$$
此时权重 1:1:1，因为三件事同等重要。
**防遗忘 Trick**: Fine-tune 时必须混入 general VQA 数据，否则模型会 catastrophic forgetting，把 general world knowledge 忘光，导致 reasoning 退化为表面 pattern。

**Hyperparameters (Table 4)**: Pre-training 跑 90k steps，seq len 40k/rank；Fine-tuning 跑 110k steps (sim) / 80k steps (real)，seq len 27k/rank。Optimizer 用 AdamW，Constant LR schedule (Pre-training $1 \times 10^{-4}$, Fine-tuning $5 \times 10^{-5}$)。

---

### 5. 实验数据表深度解析

**Table 1 (RoboTwin 2.0 主结果)**:

| Model | Easy | Hard |
| :--- | :--- | :--- |
| Diffusion Policy | 28.0% | 0.6% |
| RDT-1B | 34.5% | 13.7% |
| $\pi_0$ | 46.4% | 16.3% |
| Halo w/o EM-CoT | 75.3% | 21.2% |
| **Halo** | **80.5%** | **26.4%** |

从数据里能读出几个极其关键的 insight：
1.  Diffusion Policy 在 Hard 设置下直接跌到 0.6%。Hard 设置就是 domain randomization。说明 pure reactive policy 一旦遇到没见过的背景或物体颜色，直接抓瞎。
2.  Halo w/o EM-CoT 就已经能碾压 $\pi_0$（75.3% vs 46.4%）。这说明 versatile pre-training 给的 foundation 极其强大。
3.  EM-CoT 在 Easy 上提升 5.2%，在 Hard 上提升 5.2%（相对提升 24.5%）。**CoT 的最大价值在 OOD 场景**。当视觉特征失效时，模型靠 semantic reasoning 续命。

**Table 2 (Ablation Study)**:
Panel A (去掉 Pre-training 数据):
去掉 Visual Generation 数据，Hard 从 21.2 暴跌到 10.5。证明了 world model 能力（预测未来画面）是 OOD 泛化的核心基石。
去掉所有 Pre-training，Hard 跌到 0.0。证明 pre-training 是绝对刚需。

Panel B (去掉 EM-CoT 组件):
去掉 Text reasoning，Hard 从 26.4 跌到 18.3。
去掉 Visual subgoal，Hard 跌到 22.5。
两者都有用，且互补。Text 提供逻辑链，Vision 提供 spatial grounding。

---

### 6. 技术联想与延展

沿着这个工作，我能联想到几个非常有潜力的方向：

1.  **Test-time Scaling for VLA**: HALO 目前在 inference 时只 sample 一条 reasoning chain。完全可以借鉴 OpenAI o1 的思路，inference 时生成多条 $\mathbf{r}$ 和 $\hat{\mathbf{o}}$，然后用一个 verifier 模型评估哪个 subgoal 最 physically plausible，选最好的那条去 execute。这在 OOD 下大概率能再拉一波成功率。

2.  **RL for EM-CoT**: 目前 $\mathbf{r}$ 的生成是纯 supervised learning，模仿 VLM 写的 sentence。如果引入 RL，用 task success 作为 reward，让模型自己去探索“怎么想”最容易导致成功，可能会涌现出比人类 demo 更高效的 reasoning strategy。比如模型可能会发现“先想受力分析”比“先想目标位置”成功率更高。

3.  **Active Perception 结合**: 现在只输入 3 帧 head camera 图像。如果模型在 $\mathbf{r}$ 阶段发现“我看不清那个柠檬在哪”，能不能主动输出一个“移动相机看一眼”的 action，然后再继续 EM-CoT？这把闭环拉到了感知层。

4.  **与 LeCun JEPA 的融合**: HALO 的 visual foresight 是在 pixel space 生成 $\hat{\mathbf{o}}_{t+h}$。JEPA 哲学认为应该 predict latent representation 而避免预测 pixel 细节。如果 HALO 的 Visual Generation Expert 换成 latent predictor，可能计算成本大幅下降，而且可能更 abstract、更利于高层 reasoning。

Reference:
- BAGEL (MoT for multimodal): https://arxiv.org/abs/2505.14683
- Mixture-of-Transformers (MoT): https://arxiv.org/abs/2411.04996
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- Flex Attention (Mask 实现): https://arxiv.org/abs/2412.05496
- FLUX VAE: https://arxiv.org/abs/2506.15742
- JEPA: https://arxiv.org/abs/2301.08243

---

# HALO: Embodied Multimodal Chain-of-Thought Reasoning for VLA

非常精彩的paper，我从几个层次帮你 build intuition。

---

## 1. 高层 Intuition：HALO 在解决什么问题

当前主流 VLA 模型（如 π0、OpenVLA、FAST）大多是 **reactive policy**：把 `(image, instruction)` 直接映射到 `action chunk`。这种 mapping 在 distribution 内、short-horizon、与训练数据相似的场景下 work，但有两个根本性缺陷：

- **没有 reasoning**：模型没有显式的 "先想想再行动" 步骤，遇到 long-horizon 任务（比如 "打开抽屉→放入柠檬→关上抽屉"）就难以分解。
- **没有 world model / foresight**：模型没有 "predict 行动之后世界会变成什么样" 的能力，在 contact-rich 或 novel object 的场景下，pure pattern matching 会失败。

人类做事显然不一样：你拿起杯子之前会先在脑子里 "想象" 杯子被拿起来后的位置和姿态，再用文字性的 sub-goal 驱动 motor plan。HALO 就是把这种 **"think → imagine → act"** 的 cognitive pathway 显式建模成一个 unified policy。

关键 insight：text-only CoT（ECoT paper, Zawalski et al.）和 image-only subgoal（CoT-VLA, UP-VLA）都被证明不够——纯文本缺乏 fine-grained spatial grounding，纯图像缺乏 semantic abstraction。HALO 的核心贡献是 **同时做两者，并且 decouple 到不同的 expert 里**，避免 ManualVLA 那种 "强行把 image generation 塞进 autoregressive VLM" 导致 reasoning 能力退化的 trap。

Reference: 
- ECoT: https://arxiv.org/abs/2407.08693
- CoT-VLA: https://arxiv.org/abs/2504.10576
- UP-VLA: https://arxiv.org/abs/2501.18867
- ManualVLA: https://arxiv.org/abs/2512.02013

---

## 2. Problem Formulation 的 Intuition

HALO 把传统 monolithic policy:

$$\pi_\theta(\mathbf{a}_{t:t+m} \mid \mathbf{l}, \mathbf{o}_{t-k:t})$$

decompose 成三个 conditional distribution（Eq. 1-3）：

$$\mathbf{r} \sim P_\theta(\cdot \mid \mathbf{l}, \mathbf{o}_{t-k:t}) \quad \text{(textual reasoning)}$$

$$\hat{\mathbf{o}}_{t+h} \sim P_\theta(\cdot \mid \mathbf{l}, \mathbf{o}_{t-k:t}, \mathbf{r}) \quad \text{(visual subgoal)}$$

$$\mathbf{a}_{t:t+m} \sim \pi_\theta(\cdot \mid \mathbf{l}, \mathbf{o}_{t-k:t}, \mathbf{r}, \hat{\mathbf{o}}_{t+h}) \quad \text{(action chunk)}$$

**变量解释**：
- $\mathbf{l} \in \mathcal{L}$：language instruction（比如 "put the lemon into the drawer"）
- $\mathbf{o}_{t-k:t} \in \mathcal{O}$：过去 $k$ 帧的 visual observation history（论文里 $k = 3$）
- $\mathbf{r}$：textual chain-of-thought，是一段 first-person reasoning，比如 "I see a yellow lemon on a white plate. To put it into the drawer, I first need to grasp it firmly with my left gripper..."
- $\hat{\mathbf{o}}_{t+h}$：预测的第 $t+h$ 帧的 subgoal image（$h$ 是 lookahead horizon）
- $\mathbf{a}_{t:t+m} \in \mathcal{A}$：长度为 $m$ 的 action chunk（simulation $m=16$，real-world $m=50$）

**Intuition**：这个 formulation 本质上是一个 **Bayesian factorization**。Joint policy $P(\mathbf{r}, \hat{\mathbf{o}}, \mathbf{a} \mid \mathbf{l}, \mathbf{o}) = P(\mathbf{r} \mid \mathbf{l}, \mathbf{o}) \cdot P(\hat{\mathbf{o}} \mid \mathbf{l}, \mathbf{o}, \mathbf{r}) \cdot P(\mathbf{a} \mid \mathbf{l}, \mathbf{o}, \mathbf{r}, \hat{\mathbf{o}})$。这种顺序 condition 是符合人类 cognitive pathway 的：先 reason 出 intention，再 imagine outcome，最后 ground 到 motor command。它也避免了直接学 $P(\mathbf{a} \mid \mathbf{l}, \mathbf{o})$ 这种 dense mapping 的 ill-posedness（同一个 observation 可以对应多个合理 action，比如从不同角度抓同一个杯子）。

---

## 3. Architecture Intuition：为什么是 Mixture-of-Transformers

### 3.1 核心设计

HALO 用 **Mixture-of-Transformers (MoT)**，3 个 expert 共享 self-attention，但 FFN/参数独立：

| Expert | 功能 | 数据模态 | 生成方式 |
|--------|------|----------|----------|
| Multimodal Understanding | textual reasoning $\mathbf{r}$ | text + image tokens | autoregressive |
| Visual Generation | subgoal $\hat{\mathbf{o}}_{t+h}$ | VAE latents + noise tokens | diffusion / flow matching |
| Action Prediction | action chunk $\mathbf{a}_{t:t+m}$ | continuous action tokens | flow matching |

每个 expert 都用 Qwen2.5-1.5B 初始化（独立 init，不 share weights），cumulative ~4.5B params。三者通过 **shared self-attention** 交互——意思是 attention 的 Q/K/V 投影是共享的 attention layer 上跑所有 expert 的 tokens，但每个 expert 的 hidden state 维持在自己的 subspace 里。

**Intuition**：为什么这样做？textual reasoning 是 discrete autoregressive，visual generation 是 continuous diffusion，action 是 continuous flow matching。这三种 sampling paradigm **本质不同**：autoregressive 用 next-token prediction loss，diffusion 用 noise schedule + denoising score matching。强行把它们塞到一个 monolithic transformer 里（像 ManualVLA 那样把 image 当 discrete token 生成）会牺牲 VLM 的 reasoning 能力，因为 VAE discretize 后的 visual tokens 与 language tokens 的 distribution 差异太大，会让 attention 的 distribution collapse。

MoT 的解法：保持 expert 内部的 natural generative workflow，只在 self-attention 这一层 cross-pollinate。可以把它想象成 **三种 specialist 在一个会议室里讨论，但各自有专长**——理解 expert 提供 semantic context（"我在拿杯子"），视觉 expert 在这个 context 上做 imagination（"杯子会被举到这里"），action expert 在两者基础上输出 motor plan。

### 3.2 Modality Switching 用 Special Tokens

模型默认在 autoregressive text mode 运行。当 decoder 生成 `⟨vision start⟩` token 时，routing 把后续 hidden states 送到 visual generation expert；生成 `⟨action start⟩` 时切到 action expert。这种 "explicit control token" 设计让 inference 的 workflow 是 deterministic state machine：

```
[text reasoning: ⟨think start⟩ ... ⟨think end⟩] 
  → [visual foresight: ⟨vision start⟩ ... ⟨vision end⟩]
  → [action: ⟨action start⟩ ... ⟨action end⟩]
```

### 3.3 Attention Mask 的玄机

这个 paper 最 underrated 的细节是 attention masking 策略（Figure 3）。Mask 分四种情况：

1. **Text tokens**：标准 causal mask（autoregressive）。
2. **Visual tokens within same frame**：**bidirectional**（让 frame 内的 spatial patches 互相 attend，capture global spatial dependency）。
3. **Cross-frame / cross-modality visual tokens**：causal mask（不允许未来帧 leak 到过去）。
4. **Noise tokens**：**绝对隔离**——不能 attend 到对应的 ground truth target（否则 diffusion 训练时直接 cheat），其它 token 也不能 attend 到 noise（防止 noise 污染 reasoning context）。

Pre-training 的 mask（Figure 8）和 fine-tuning 略有不同，因为 pre-training 阶段还没引入 EM-CoT 的 sequential structure。

**Intuition**：这种混合 mask 本质上是一个 **"locally bidirectional, globally causal"** 的结构。Diffusion model 训练时每个 patch 的 noise 需要看其他 patch 的 clean 版本来确定 denoising direction，所以必须 bidirectional。但 action prediction 又依赖前面产生的 reasoning 和 subgoal，必须 causal。Mask 设计是这两种需求的最小冲突 union。

### 3.4 多模态 Encoders

- **Text**：标准 Qwen2.5 tokenizer，vocab 扩展加入 special control tokens。
- **Visual understanding**：ViT + SigLIP2-so400m/14（pre-trained 384×384，interpolate 到 980×980）+ NaViT 处理任意 aspect ratio + 2-layer MLP projector。
- **Visual generation**：FLUX 的 pre-trained VAE，downsample 8x，latent channel 16。VAE latents 被 flatten 成 2×2 spatial patches 后再 linear project 到 LLM hidden dim。VAE frozen。
- **Action**：simple linear projection（action 是低维物理量，没必要复杂 encoder）。

**Intuition**：双 visual pathway（ViT for semantic + VAE for spatial latent）是关键 trick。Semantic reasoning 需要 "这是杯子" 这种 high-level concept，VAE latent 给不了；visual generation 需要 "杯子的 precise 位置和形状"，ViT feature 给不了。所以 Halo 对 image 用两种 encoder 同时编码，输入到不同的 expert。

Reference:
- MoT: https://arxiv.org/abs/2411.04996
- BAGEL (灵感来源): https://arxiv.org/abs/2505.14683
- SigLIP2: https://arxiv.org/abs/2502.14786
- NaViT: https://arxiv.org/abs/2307.06304
- FLUX.1 Kontext: https://arxiv.org/abs/2506.15742
- Qwen2.5: https://arxiv.org/abs/2412.15115

---

## 4. EM-CoT Data Pipeline：怎么自动生成 CoT 监督

这是 paper 的第二个核心贡献。CoT 监督数据稀缺，人工标注 long-horizon trajectory 的 reasoning 极其昂贵。HALO 用三阶段自动 pipeline：

### Stage 1: Action Primitives Extraction（Algorithm 1）

Rule-based matching 把连续 low-level action 转成 discrete motion primitives。

**核心函数**：
```
IsIdle(P^a[t], P^a[t-1], G^a[t], G^a[t-1]; θ_vel, θ_dg)
```
- $P^a[t]$：arm $a \in \{l, r\}$ 在时刻 $t$ 的 end-effector pose
- $G^a[t]$：gripper 开合状态
- $\theta_{\text{vel}}$：速度阈值
- $\theta_{\text{dg}}$：gripper 变化阈值
- $\theta_{\text{min.idle}}$：idle 持续多少帧才算 segment 边界

Algorithm 流程：
1. 对每帧计算 idle flag（位移速度 < $\theta_{\text{vel}}$ 且 gripper 变化 < $\theta_{\text{dg}}$）
2. 用 `SegmentActions()` 按 idle period ≥ $\theta_{\text{min.idle}}$ 切分 motion segments
3. 在每个 segment 内 label：
   - $\Delta G^a[t] \le -\theta_{\text{dg}}$ → `grasp`
   - $\Delta G^a[t] \ge \theta_{\text{dg}}$ → `release`
   - 否则计算 $\Delta P = \sum_{t=s'+1}^{e'} (P^a[t] - P^a[t-1])$，用 `GetDirection(ΔP; θ_dir)` 给 move 加方向 label

**Intuition**：这一步把 high-dimensional continuous trajectory 压缩成 discrete symbolic sequence（`{grasp, release, move_left, move_right, idle}` 等）。这是 RT-H 的思路（hierarchical action via language）。Symbolic representation 给后续 VLM annotator 提供 structured input，否则直接把 raw 6D pose 喂给 VLM 它没法理解。

### Stage 2: VLM Annotator（Qwen3-VL）

三阶段 prompting：

**Stage 2a - Task Narrative Generation**：把 symbolic action 序列 + instruction 喂给 VLM，要求生成 single coherent paragraph 描述任务时序，强调 bimanual coordination。

**Stage 2b - Subtask Decomposition**：要求 VLM 把 narrative 拆成 2-5 个 goal-oriented subtasks（如 "Pick up red cup", "Pour water into cup"），强调不要拆成低层 motion（不要 "move", "grab"）。

**Stage 2c - Subtask Alignment + First-person Reasoning**：把 subtask list 和 frame-wise action label 给 VLM，要求对每个 frame segment 输出 first-person reasoning（"I see...", "I need to...", "I am moving..."），每段 <50 词，且必须包含 (a) visual observation (b) goal-driven inference (c) movement logic。

**Intuition**：三阶段 prompting 是 key。直接 one-shot prompt VLM 让它输出完整 CoT 会 hallucinate，因为它不知道哪些 frame 属于哪个 subtask。先 narrative → 再 decompose → 再 align 是 **coarse-to-fine** 的 hierarchy，每一步都给下一步提供 structured context。First-person 视角是为了让模型在 inference 时 self-consistent（"I" 在训练和推理时一致）。

### Stage 3: Visual Subgoal Extraction（Algorithm 2）

```
Input: Frame annotations A = {a_0, ..., a_{T-1}}, Image sequence T = [I_0, ..., I_{T-1}]
Output: Goal image sequence G = [G_0, ..., G_{T-1}]
```

对每个 subtask $s$，找到 subtask 结束的最后一帧 $t_g$（即下一个 subtask 开始的前一帧），把 $I_{t_g}$ 作为该 subtask 的 visual subgoal，broadcast 到所有属于 $s$ 的 frame。

**Intuition**：terminal frame 的选择很巧妙。Subgoal image 应该是 "完成这个 subtask 后世界长什么样"，所以用 terminal frame 而不是中间帧。这个 sparse supervision 降低了学习难度——模型不需要预测 trajectory 中每一帧的 evolution（这个 ill-posed），只需要 predict "下一个 milestone 长什么样"。

Reference:
- RT-H (action hierarchy via language): https://arxiv.org/abs/2403.01823
- Qwen3-VL: https://arxiv.org/abs/2511.21631

---

## 5. Training Recipe：两阶段策略

### Stage 1: Versatile Pre-training

数据混合：
- **VQA**：LLaVA-NeXT-779k（cross-entropy loss $\mathcal{L}_{\text{CE}}$）
- **VG (Visual Generation)**：OXE robotic trajectories + SSv2 egocentric videos，做 future frame prediction $(l, I_{t-k:t}) \to I_{t+h}$，flow-matching MSE loss $\mathcal{L}_{\text{MSE}}$
- **AP (Action Prediction)**：OXE 做 imitation learning，$L_1$ flow-matching loss $\mathcal{L}_{\text{L1}}$

总 loss（Eq. 4）：
$$\mathcal{L}_{\text{pt}} = 0.25\mathcal{L}_{\text{CE}} + 0.5\mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{L1}}$$

**Intuition**：权重 1:2:4（CE:MSE:L1）反映 optimization difficulty——action prediction 最难（continuous + 物理结构敏感），所以 weight 最高；VQA 最容易（已经 pre-trained 的 capability），weight 最低。这类似于 multi-task learning 里给难任务更高 weight 的常见 trick。

**为什么需要 VG 数据？**：SSv2 是 egocentric manipulation video，给模型提供 **physical common sense**——比如 "推杯子上部会倒，推下部会滑动"。OXE 给模型 robotic-specific dynamics。这两个数据源互补：SSv2 是 human demonstration（丰富但无 action label），OXE 是 robot demonstration（有 action label 但 scenario 有限）。

### Stage 2: EM-CoT-Augmented Fine-tuning

数据：RoboTwin 2.0（2500 demos，50/task × 50 tasks）+ real-world（320 demos，80/task × 4 tasks），都过 EM-CoT pipeline 处理。

Loss（Eq. 5）：
$$\mathcal{L}_{\text{ft}} = \mathcal{L}_\mathbf{r} + \mathcal{L}_{\hat{\mathbf{o}}} + \mathcal{L}_\mathbf{a}$$

权重 1:1:1，因为这时三件事同等重要。

**Key trick**：fine-tune 时混入 general VQA 数据（Cheang et al. 2025 的 GR-3 思路），防止 catastrophic forgetting。这点很重要——纯 fine-tune EM-CoT 会让模型忘记 general world knowledge，导致 reasoning 退化为 surface pattern。

### Training Hyperparameters (Table 4)

| Config | Pre-training | Fine-tuning |
|--------|--------------|-------------|
| LR | $1 \times 10^{-4}$ | $5 \times 10^{-5}$ |
| Schedule | Constant | Constant |
| Seq len/rank | 40k | 27k |
| Steps | 90k | 110k (sim) / 80k (real) |
| GPUs | 32× H100 | 32× H100 |
| Loss weight | CE:MSE:L1 = 1:2:4 | 1:1:1 |

**Intuition**：40k 的超长 sequence 是必要的，因为要把 VQA + VG + AP 样本打包成一个 sequence 训练（提高 GPU 利用率，Flex Attention 加速）。Constant LR + 没有 decay 说明这个 scale 下模型还没到 convergence plateau，不需要 decay 来 refine。

Reference:
- GR-3: https://arxiv.org/abs/2507.15493
- LLaVA-NeXT: https://llava-vl.github.io/blog/2024-01-30-llava-next/
- SSv2: https://arxiv.org/abs/1706.04230
- OXE / OpenVLA: https://openx-embodiment.github.io/
- Flex Attention: https://arxiv.org/abs/2412.05496

---

## 6. 实验数据 Intuition

### 6.1 RoboTwin 2.0 主结果（Table 1）

| Model | Easy | Hard |
|-------|------|------|
| Diffusion Policy | 28.0% | 0.6% |
| RDT-1B | 34.5% | 13.7% |
| π0 | 46.4% | 16.3% |
| Halo w/o EM-CoT | 75.3% | 21.2% |
| **Halo** | **80.5%** | **26.4%** |

几个 insight：

1. **Diffusion Policy 在 Hard 设置下几乎归零**（0.6%）。Hard 设置是 domain randomization，object appearance + background 大变。这说明 reactive policy 完全依赖 visual pattern matching，OOD 即崩。
2. **Halo-w/o-CoT 已经比 π0 高 28.9 points**（Easy），这说明 versatile pre-training 的 foundation 极强。EM-CoT 只贡献额外 5.2 points 在 Easy 上，但在 Hard 上贡献 5.2 points（相对 +24.5%）。
3. **EM-CoT 在 Hard 设置的相对提升更显著**。这印证了核心假设：reasoning 在 OOD 场景下作用最大——因为 visual pattern 失效时，model 必须靠 semantic reasoning 来 ground。

### 6.2 Ablation 1: Training Recipe (Table 2 Panel A)

| Setting | Easy | Hard |
|---------|------|------|
| Full (V+T+A) | 75.3 | 21.2 |
| w/o V (visual gen) | 58.2 | 10.5 |
| w/o V+T | 42.9 | 3.9 |
| w/o V+T+A | 32.4 | 0.0 |

- 去掉 VG 数据：Hard 降一半（21.2 → 10.5）。**Visual generation 数据是 OOD robustness 的关键**——它教模型 "世界会怎么 evolve"，在 Hard setting 下视觉变化大，world model 比 pattern matching 更鲁棒。
- 去掉所有 pre-training：Hard 直接归零。证明 pre-training 是 foundation requirement。
- VQA 数据（T）单独去掉：Hard 从 10.5 降到 3.9，说明 VQA 提供的 semantic grounding 对 long-horizon task decomposition 至关重要。

### 6.3 Ablation 2: EM-CoT Components (Table 2 Panel B)

| Setting | Easy | Hard |
|---------|------|------|
| Halo (V+T) | 80.5 | 26.4 |
| w/o T (no text reasoning) | 77.8 | 18.3 |
| w/o V (no visual subgoal) | 76.1 | 22.5 |
| w/o V&T (no EM-CoT) | 75.3 | 21.2 |

- 去掉 text reasoning：Hard 从 26.4 降到 18.3（相对 -30.7%）。Textual reasoning 对 OOD 最关键。
- 去掉 visual subgoal：Hard 降到 22.5（相对 -14.8%）。Visual foresight 也有用但不如 text。
- 两者互补，加在一起最强。

### 6.4 Per-task 分析

看几个极端 case：
- **Blocks Ranking Size**：π0 = 7%, Halo = 58%。这是精细 size discrimination 任务，需要 reasoning "哪个 block 更大"。
- **Stamp Seal**：π0 = 3%, Halo = 60%。需要 fine-grained spatial reasoning。
- **Stack Blocks Three**：π0 = 17%, Halo = 96%。Long-horizon 任务，需要 task decomposition。

这些 task 都是 "对 reasoning 敏感" 的，HALO 的优势最大。在简单 reactive 任务（如 Grab Roller 96% vs 95%）上 advantage 很小，因为那种任务不需要 reasoning。

### 6.5 Real-World 结果（Figure 7）

四个任务：sweeping, cup nesting, screwdriver handover, lemon into drawer。HALO 在 basic 和 generalization setting 都超过 π0 和 π0.5。

Generalization axes：
- Visual distraction（桌上放干扰物）
- Lighting variation（desk lamp 亮度变化）
- Background variation（桌布颜色变化）
- Novel object（broom → sponge）

HALO 在 generalization setting 下 degradation 比 baseline 小得多。这印证了 simulation 实验：EM-CoT 让模型在 OOD 下 "想清楚该做什么"，而不是靠 pattern matching。

Reference:
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- RDT-1B: https://arxiv.org/abs/2410.07864
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- Mobile ALOHA: https://arxiv.org/abs/2401.02117

---

## 7. 与相关工作的关系：我帮你 build 一张 mental map

### 7.1 VLA 演化谱系

```
Reactive VLA (π0, OpenVLA, FAST)
    ↓ +text reasoning
ECoT-VLA (Zawalski 2025): textual CoT before action
    ↓ +visual reasoning
CoT-VLA (Zhao 2025): generate subgoal images
UP-VLA (Zhang 2025): unified understanding + prediction
    ↓ +unified multimodal
ManualVLA (Gu 2025): shared expert for text+image gen
    ↓ +decoupled experts
HALO: MoT with 3 specialized experts, EM-CoT
```

### 7.2 MoT 架构的位置

MoT (Liang et al. 2024) 最初是为 multimodal foundation model 设计的 sparse 架构，把 text 和 image/video token 分到不同 expert，shared self-attention。BAGEL (Deng et al. 2025) 把 MoT 用于 unified understanding + generation，证明这个架构能 scale。HALO 是把 MoT 扩展到 VLA 场景，加 action expert。

### 7.3 World Model 路线

HALO 的 visual foresight 可以看作 implicit world model——预测 "action 之后世界长什么样"。对比：
- **Dreamer 系列**（RL world model）：explicit dynamics model + reward model + policy
- **GAIA-1, Genie**：video generation as world model
- **UniSim**：universal simulator via video generation
- **HALO**：world model 作为 reasoning 的一部分，不是独立 component

HALO 的 subgoal prediction 是 "sparse" world model（只 predict milestone，不 predict 每一帧），这比 dense frame prediction 更 tractable。

### 7.4 Reasoning in Robotics

- **Code as Policies** (Liang et al.)：LLM 生成 code 执行
- **VoxPoser** (Huang et al.)：LLM 输出 3D value map
- **SayCan** (Ahn et al.)：LLM scoring + affordance
- **RT-2** (Google)：VLM 直接输出 action
- **RT-H** (Belkhale et al.)：action hierarchy via language
- **ECoT**：textual reasoning for robot control
- **HALO**：multimodal reasoning in unified architecture

HALO 与 ECoT 的区别：ECoT 只有 text reasoning，没有 visual foresight；HALO 有 visual expert 做 imagination，更接近 human cognition。

Reference:
- RT-2: https://arxiv.org/abs/2307.15818
- SayCan: https://arxiv.org/abs/2204.01691
- VoxPoser: https://voxposer.github.io/
- Code as Policies: https://code-as-policies.github.io/
- Chain-of-Thought (Wei et al.): https://arxiv.org/abs/2201.11903

---

## 8. 一些 Critical Observations 和 Open Questions

### 8.1 Strengths

1. **Decoupled experts 是 elegant design**：避免了 ManualVLA 把 image token prediction 塞进 autoregressive VLM 的退化。Diffusion expert 可以 native 用 flow matching，text expert 保持 autoregressive，action expert 保持 continuous flow——三者各得其所。
2. **EM-CoT data pipeline 是 scalable**：rule-based primitive + VLM annotator 是巧妙的两阶段 abstraction，比纯 GPT-4V 标注便宜得多。
3. **Pre-training 数据混合策略**：VQA + VG + AP 三类数据互补，ablation 证明每一类都有 measurable benefit。

### 8.2 潜在 Limitations

1. **Latency**：think → imagine → act 三阶段 sequential generation，inference 时比 reactive policy 慢。Paper 没报 latency，但估计 textual reasoning 要生成几十 token，visual generation 要 multi-step diffusion。Real-time control 可能 challenge。
2. **Subgoal image 质量**：paper 显示的 subgoal image 看起来 reasonable，但没量化评估（FID, 人类评估等）。Subgoal 不准的话会误导 action。
3. **Action expert 用 flow matching**：和 π0 一样，但 π0 有 VLM condition action expert，HALO 是 action expert 直接 condition 在 reasoning + subgoal 上。Abation 没比较这两种 conditioning。
4. **EM-CoT 数据 bias**：Rule-based primitive 提取依赖 threshold（$\theta_{\text{vel}}, \theta_{\text{dg}}, \theta_{\text{min.idle}}$），不同 robot morphology 可能需要不同 threshold。Pipeline 的 robustness 在跨 robot 时可能有问题。
5. **Hard setting 还是只有 26.4%**：虽然比 baseline 强，但离实用还远。Domain randomization 太 aggressive 时 visual generation expert 可能也 OOD。

### 8.3 我联想到的几个方向

1. **Hierarchical EM-CoT**：当前 EM-CoT 是 single-level，可以做 multi-level（task → subtask → motion primitive → action），类似 RT-H 的 hierarchy + HALO 的 multimodal reasoning。
2. **Reinforcement Learning on EM-CoT**：现在 EM-CoT 是 supervised，可以加 RL fine-tune（用 task success 作为 reward），让 model 学会 generate "better" reasoning 而不只是模仿 demo。
3. **Test-time scaling**：像 OpenAI o1 那样，inference 时 sample 多个 EM-CoT 候选，用 world model verify 哪个 subgoal 最 plausible。这能显著提升 OOD 性能。
4. **Active perception**：当前只用 head camera 的 3 帧 history。可以加主动控制相机视角的能力，让 model 在 reasoning 阶段 "探头看看"。
5. **Diffusion Policy × EM-CoT**：HALO 用 flow matching 做 action，但只在 expert 内部。可以探索把整个 think-imagine-act 作为一个 hierarchical diffusion process。
6. **与 LeCun JEPA 的关系**：JEPA 提倡 predict latent representation 而非 pixel。HALO 的 visual foresight 是 pixel-space，可以加一个 latent-space counterpart 做 dual prediction，可能更 sample efficient。

Reference:
- JEPA (LeCun): https://arxiv.org/abs/2301.08243
- GR00T N1 (NVIDIA): https://arxiv.org/abs/2503.14734

---

## 9. 总结：HALO 的 Key Takeaways

1. **EM-CoT formulation**：把 VLA policy 拆成 think → imagine → act 三阶段 conditional distribution，对应 human cognitive pathway。
2. **MoT architecture**：3 个 specialized expert（understanding / visual generation / action），shared self-attention + modality-specific FFN，每个 expert 保持 native generative workflow。
3. **Attention masking**：text causal，visual within-frame bidirectional，noise token 隔离，是混合 sampling paradigm 的关键。
4. **EM-CoT data pipeline**：rule-based primitive extraction + 三阶段 VLM prompting + terminal frame subgoal，完全自动化的 CoT supervision。
5. **Two-stage training**：versatile pre-training (VQA+VG+AP, 1:2:4 weight) → EM-CoT fine-tuning (1:1:1 weight, 混 VQA 防 forgetting)。
6. **Results**：RoboTwin 80.5% Easy / 26.4% Hard（+34.1% / +10.1% over π0），real-world 4 tasks 在 generalization setting 下显著超 π0 / π0.5。

**Final intuition**：HALO 给出了一个 clean 的 "robot as a deliberative agent" 范式——不是 stimulus-response，而是 stimulus-deliberation-imagination-response。这个 paradigm 在 long-horizon + OOD 场景下展示了明显优势。下一步的关键问题是 latency、subgoal 质量评估、以及 EM-CoT 本身能否通过 RL 进一步优化。

如果你想 deep dive 某一块（比如 attention mask 的实现细节、EM-CoT prompt 的具体格式、或者 real-world 实验 setup），告诉我，我可以再展开。
