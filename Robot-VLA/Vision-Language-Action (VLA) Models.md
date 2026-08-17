---
source_pdf: Vision-Language-Action (VLA) Models.pdf
paper_sha256: 15c95fa481686f6b2589eb2298268f6e39672f6cd0572b1c373a0a29ec0ca069
processed_at: '2026-08-13T01:39:29-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 VLA

好，我们换一种方式聊。你刚看完那篇 1122 行的 review，脑子里可能塞满了 token、diffusion、LoRA 这些词。我试着用更直觉的方式把整个故事串起来，顺便补一些论文里没展开的技术细节。

---

## 故事的起点：为什么 robot 一直这么"笨"

你想一下，过去十年 AI 发生了什么。Vision 那边，ResNet 到 ViT 到 DINOv2，模型能认出图片里有一只猫、一个苹果、一把椅子。Language 那边，从 GPT-1 到 GPT-4，模型能写代码、做数学题、跟你聊天。Robotics 那边呢？还在用 MPC、hand-crafted reward function、reinforcement learning 训一个 pick-and-place 训三个月。

三个领域各自猛涨，但 robot 还是那个 robot。你在 ChatGPT 里问"怎么煮意大利面"，它能给你写一个完美的菜谱。但你要把这个菜谱交给一个 robot arm 去执行，对不起，得先派一个 PhD student 去手动标 5000 条 demonstration trajectory，再训一个 policy network，换个稍微不同的厨房就全崩了。

这个 gap 的本质是什么？是 **action**。Vision 有 image tokens，language 有 text tokens，但 action 没有一个统一的、可扩展的、能从 internet data 里学到东西的表示方式。你不可能从网上爬 10 亿条"机械臂 joint angle 序列"的数据。

VLA 的核心 insight 就一句话：**把 action 也变成 token，让 Transformer 用预测下一个词的方式预测下一个动作**。

就这么简单。但这个 idea 的连锁反应是巨大的。

---

## 从 CLIPort 到 RT-2：两步关键的跳跃

### 第一步：CLIPort — 先把眼睛和耳朵接上

2022 年，Shridhar 等人做了 CLIPort。逻辑很直白：

CLIP 已经学会了把 "red block" 这个 text 和图片里红色方块的 visual patch 对齐到同一个 embedding space。那我就直接用 CLIP 的 embedding 当 input，后面接一个 TransporterNet 预测 pick-and-place 的位置。

$$\text{action} = f(\text{CLIP}_{\text{image}}(\mathbf{I}), \text{CLIP}_{\text{text}}(\mathbf{T}))$$

你给一张桌面照片和一句 "put the blue block on the red square"，模型就知道要拿蓝色的、放到红色方块上面。这个模型在小规模 task 上 work 得很好，但你给它一个没见过的物体组合，或者稍微复杂一点的指令，就挂了。

为什么？因为 CLIP 的 alignment 是 image-level 的，不是 action-level 的。CLIP 知道 "blue block" 长什么样，但不知道"拿起来"需要什么样的 motor command。Vision-language 和 action 之间还是隔着一层 hand-crafted 的 interface。

### 第二步：RT-2 — 把 action 真正变成语言的一部分

2023 年 7 月，Google DeepMind 发了 RT-2。这是真正的 paradigm shift。

核心 idea：既然 LLM 能预测下一个 token，那我让 action 也变成 token，LLM 不就能预测下一个 action 了吗？

具体怎么做？一个 7-DoF 机械臂 + 1 个 gripper，每一步 action 是 8 个数字。把每个数字离散化成 256 个 bin，每个 bin 对应一个 token。这样一步 action 就是 8 个 token，和一句话里的 8 个 word 没有本质区别。

然后你拿一个 PaLI-X（55B 参数的 vision-language model），在两个数据集上同时训练：

- **互联网数据**：image-caption pairs, VQA, web text（几十亿条）
- **robot 数据**：RT-1 收集的 130k 条真实 robot demonstration

训练 loss 就是标准的 next-token prediction：

$$\mathcal{L} = -\sum_t \log P(a_t | a_{<t}, \mathbf{I}, \mathbf{T}; \theta)$$

只不过 $a_t$ 可能是 language token，也可能是 action token。模型分不清，也不需要分清。

结果是：RT-2 可以执行 "pick the red cup left of the bowl" 这种组合指令，即使它从没在 robot 数据里见过这个具体的 cup+bowl 组合。为什么？因为它在 internet 数据上学过 "left of" 是什么意思，学过 "red cup" 是什么样子。这些知识通过 co-fine-tuning "流"到了 action prediction 里。

这就是论文里说的 **emergent generalization**。你不需要为每个新场景重新训 robot，model 把 web knowledge 迁移过来了。

参考：https://robotics-transformer2.github.io/

---

## Tokenization：VLA 的"秘密武器"

我前面说 VLA 的核心是"把 action 变成 token"，但这里面的门道很多。让我把三种主流方案掰开讲。

### 方案一：Discrete Binning（RT-1, OpenVLA）

最直觉的方法。你有 7 个 joint angles + 1 个 gripper command，每个都是一个连续值。把每个维度独立地 normalize 到 $[-1, 1]$，然后均匀分成 256 个 bin：

$$a_{\text{disc}}^{(j)} = \text{round}\left(\frac{a^{(j)} + 1}{2} \times 255\right) \in \{0, 1, \ldots, 255\}$$

一步 action 就变成 8 个整数，每个是一个 token。vocab 大小是 256（每个维度独立），或者 $256^8$（如果当作一个联合 token，但这太大了）。

这个方案的好处是简单、直接用 LLM 的 cross-entropy loss 就能训。坏处是精度有限：256 bins 意味着如果你的 joint angle range 是 $[-\pi, \pi]$，精度大概是 $\frac{2\pi}{256} \approx 0.025$ 弧度 $\approx 1.4°$。对大部分 manipulation 够用，但对 surgical robotics 这种需要 sub-millimeter 精度的场景就不够了。

OpenVLA（Stanford, 2024）就是用这个方案。7B 参数，基于 Prismatic-7B（Llama-2 backbone + DINOv2 + SigLIP 双 vision encoder），在 Open X-Embodiment 的 970k 条 real-world demonstration 上训练。开源，可以用 LoRA fine-tune。

参考：https://openvla.github.io/

### 方案二：Diffusion Head（Octo, Pi-0, RDT-1B）

如果你觉得 discrete binning 太粗糙，那能不能直接在连续空间生成 action？

Diffusion model 的思路是：先给 action 加噪声，然后学一个网络预测噪声是什么，inference 时一步步去噪。

Forward process（加噪声）：
$$\mathbf{a}_t = \sqrt{\bar{\alpha}_t} \mathbf{a}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

其中 $\mathbf{a}_0$ 是 clean action，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ 是 noise schedule，$t$ 是 timestep（$t=0$ 最干净，$t=T$ 最 noisy）。

Training loss：
$$\mathcal{L}_{\text{diff}} = \mathbb{E}_{t, \mathbf{a}_0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{a}_t, t, \mathbf{c}) \|^2 \right]$$

其中 $\mathbf{c}$ 是 conditioning（vision + language + state 的 fused embedding），$\boldsymbol{\epsilon}_\theta$ 是要学的 noise prediction network。

Inference（去噪）：
$$\mathbf{a}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{a}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{a}_t, t, \mathbf{c}) \right) + \sigma_t \mathbf{z}$$

从 $\mathbf{a}_T \sim \mathcal{N}(0, \mathbf{I})$ 开始，迭代 $T$ 步得到 $\mathbf{a}_0$。

**为什么 diffusion 比 MLP head 好？** 因为 MLP head 训练时会遇到 mode collapse：如果对一个状态有多个合理的 action（比如从左边抓和从右边抓都可以），MSE loss 会让模型预测它们的平均值，结果是一个两个方向都不沾边的 garbage action。Diffusion 天然支持 multimodal distribution，因为它学的是 score function $\nabla_{\mathbf{a}} \log p(\mathbf{a} | \mathbf{c})$，可以在多个 mode 之间分配概率。

**但 diffusion 的代价是什么？** Inference 慢。每生成一个 action 需要 50-100 步 denoising，每步都要 forward pass 一次 network。相比之下 autoregressive 只需要 8 步（8 个 token）。

Pi-0（Physical Intelligence, 2024）用的是 flow matching，是 diffusion 的变体，用 straight-line path 代替 DDPM 的 curved path，sampling 可以少到 10 步。但仍然比 autoregressive 慢。

参考：https://arxiv.org/abs/2410.24164

### 方案三：FAST Tokenization（Pi-0 Fast, 2025）

这是目前最巧妙的方案，来自 Chelsea Finn 组。

核心 idea：为什么不直接压缩 action trajectory，而不是一步一步生成？

具体做法分两步：

**Step 1: DCT 压缩**

你有一段 1000ms 的 action trajectory $\mathbf{a}_{1:T}$，$T$ 可能是 100 步（10ms 一步）。做 Discrete Cosine Transform：

$$X_k = \sum_{n=0}^{T-1} a_n \cos\left(\frac{\pi}{T}\left(n + \frac{1}{2}\right) k\right), \quad k = 0, 1, \ldots, T-1$$

DCT 把时域信号转到频域。物理意义是：大部分 action trajectory 是平滑的，能量集中在低频。高频成分（比如微小的抖动）对 task success 不重要。所以你保留前 $K$ 个低频系数（$K = 16$），就能很好地 reconstruct trajectory。

$$\hat{\mathbf{a}}_{1:T} \approx \sum_{k=0}^{K-1} X_k \phi_k$$

其中 $\phi_k$ 是 DCT basis function（余弦波）。

**Step 2: BPE 聚合**

然后对频域系数做 Byte-Pair Encoding，把常见的系数组合聚合成 discrete token。类似 LLM 把 "the", "ing" 这种高频组合变成单个 token。

结果是：1000ms 的 action window 压缩成 16 个 token。一次 autoregressive forward pass 就能预测整段 trajectory。

**效果**：Pi-0 Fast 在桌面 GPU 上达到 **200 Hz** 控制频率，比标准 autoregressive 快 15×。精度损失可忽略。

这个方案的美妙之处在于：它利用了 action trajectory 的时域冗余性。相邻的 action steps 高度相关，DCT 把这种冗余性 explicit 地提取出来。这和视频压缩（H.264 用 DCT）是同一个原理。

参考：https://arxiv.org/abs/2501.09747

---

## Vision Encoder 的选择：为什么 OpenVLA 用两个？

你可能会问，CLIP 的 vision encoder 不就够了吗？为什么要 DINOv2 + SigLIP 两个？

答案在于两种 encoder 学到的东西不同：

**CLIP / SigLIP**：通过 image-text contrastive learning 训练。学到的 visual feature 高度 aligned with language semantics。你问它 "red cup 在哪"，它能准确告诉你。但它的 feature 丢失了很多 spatial detail，因为 contrastive loss 只关心 global alignment。

**DINOv2**：通过 self-supervised learning（student-teacher distillation + masked patch prediction）训练。不依赖 text，纯视觉。学到的 feature 保留了 dense spatial information，对 object localization 和 geometry understanding 很强。但它不直接和 language aligned。

OpenVLA 把两个 feature 加起来：

$$\mathbf{V}_{\text{fused}} = \text{LayerNorm}\left(\text{DINOv2}(\mathbf{I}) + \text{SigLIP}(\mathbf{I})\right)$$

这样既有 semantic alignment（SigLIP），又有 spatial precision（DINOv2）。实验显示比单用任何一个都好。

这个 idea 也可以从 multi-view geometry 的角度理解：两个 encoder 提供了同一 scene 的两个 "view"，一个偏向 semantics，一个偏向 geometry。Fusion 相当于在 feature space 做了 multi-view fusion。

参考：https://arxiv.org/abs/2304.07193（DINOv2）

---

## Dual-System：为什么一个模型搞不定所有事？

RT-2 证明了 end-to-end VLA 可以 work。但很快人们发现它在 long-horizon task 上 struggle。比如 "clean the kitchen"——你需要打开柜子、拿海绵、拧水龙头、擦桌子、放回海绵。这是 20+ 步的 task，每步又有几十个 motor command。一个 autoregressive VLA 要生成几百个 token，error 会累积，中间某步错了后面全崩。

NVIDIA 的 GR00T N1（2025）给了一个解法：用两个 system。

```
System 2 (LLM-based, 慢但聪明)
  Input: "clean the kitchen" + visual scene
  Output: [open cabinet, grab sponge, turn on faucet, wipe table, put back sponge]
  Latency: ~800ms (每 5-10 秒调用一次)

System 1 (Diffusion policy, 快但只会执行)
  Input: "grab sponge" + current visual + joint state
  Output: 50Hz motor commands
  Latency: 10ms (持续运行)
```

这两个 system 通过一个 queue 解耦。System 2 每隔几秒 push 一个 subtask，System 1 持续 pop subtask 执行。

为什么这样设计？因为 long-horizon planning 和 real-time control 对模型的要求完全不同：

- **Planning** 需要 reasoning, generalization, commonsense。LLM 擅长这个。但 LLM 慢，800ms 出一个 decision，没法做 50Hz 控制。
- **Control** 需要 precision, reactivity, smoothness。Diffusion policy 擅长这个。但它不擅长 high-level reasoning，你问它"接下来该干嘛"它答不上来。

这和人的认知系统很像。Kahneman 的 System 1 / System 2 理论：System 1 是快速、直觉的（对应 reactive control），System 2 是慢速、理性的（对应 planning）。人脑就是这么分工的。

**但 dual-system 有一个很难的问题：temporal synchronization。**

System 2 出一个 plan 要 800ms，System 1 每 20ms 要一个 action。在 System 2 thinking 的这 800ms 里，System 1 怎么办？它还在执行上一个 subtask，但环境可能已经变了。

GR00T N1 的解法是让 System 1 有一定的 autonomy：它收到一个 subtask 后，可以自主执行几秒，不需要等 System 2 每步指令。但遇到 unexpected situation（比如物体突然移动），System 1 需要 interrupt System 2 重新 plan。这个 interrupt 机制是 engineering 难点。

Helix（Figure AI, 2025）走得更远：System 2 用 7-9 Hz 的 VLM，System 1 用 200 Hz 的 transformer 控制 humanoid 的全身上肢。这意味着 System 1 在 System 2 的两次决策之间要自主跑 20-30 步。

参考：https://arxiv.org/abs/2503.14734（GR00T N1）

---

## Co-fine-tuning 为什么 work？

这是 VLA 最 magic 的地方，我觉得值得仔细讲讲。

你有一个 55B 的 VLM（PaLI-X），在 internet 上训了几年。它见过几十亿张图片，理解 "red cup"、"left of"、"pick up" 这些概念。但它不知道怎么控制机械臂。

然后你拿来 130k 条 robot demonstration。每条是 (image, instruction, action trajectory)。你把这些 action 也变成 token，和 VQA data 混在一起训。

```python
# 训练 batch 大概长这样
batch = [
    {"image": img1, "text": "What color is the car?", "answer": "red"},
    {"image": img2, "text": "pick the blue block", "answer": "<action_token_1><action_token_2>..."},
    {"image": img3, "text": "Describe the scene", "answer": "A kitchen with..."},
    {"image": img4, "text": "place on red square", "answer": "<action_token_1><action_token_2>..."},
    ...
]
```

模型不知道哪条是 VQA，哪条是 robot control。它只是学一个统一的 next-token prediction。

**为什么这能 work？** 因为 language 和 action 共享同一套 compositional structure。

"pick the red cup left of the bowl" 这个指令，模型在 VQA 数据上学过怎么理解。它知道 "pick" 是一个 grabbing action，"red cup" 是一个 object，"left of the bowl" 是一个 spatial relation。在 robot 数据上，它学会了 "pick" 对应什么样的 motor primitive，"left of" 在视觉上对应什么 spatial location。

co-fine-tuning 让这两套知识在同一个 parameter space 里 align。当你给一个新指令 "grab the yellow mug right of the plate"，即使 robot 数据里没见过这个组合，模型也能用 VQA 学到的 compositional understanding 来 infer 正确的 action。

这就像小孩学语言：先听大人说话（web data），再自己动手试（robot data），两者互相 reinforce。

OpenVLA 的实验证实了这一点：7B 参数的 OpenVLA（co-fine-tuned）比 55B 的 RT-2-X（只在 robot data 上训）success rate 高 16.5%。co-fine-tuning 比 scaling 参数量更重要。

---

## 为什么 Inference 这么慢？怎么加速？

VLA 的 inference bottleneck 在哪？

一个典型的 forward pass：

1. **Vision encoding**: 400 个 vision tokens，每个 1024 维。ViT-L/14 的 forward 大约 50ms。
2. **Language encoding**: 12 个 language tokens。BERT/T5 forward 大约 5ms。
3. **Cross-attention fusion**: $O(N_v \times N_l \times d)$，大约 10ms。
4. **Autoregressive decoding**: 8 个 action tokens，每个一次 forward pass。如果 backbone 是 7B，每次 forward 大约 50ms，总共 400ms。

加起来大概 465ms 一步。控制频率约 2 Hz。这比 robot 需要的 50-200 Hz 差了一个数量级。

**加速的几种思路：**

### 量化

把 FP32 权重变成 INT8：

$$\mathbf{W}_{\text{int8}} = \text{round}\left(\frac{\mathbf{W}_{\text{fp32}}}{s}\right), \quad s = \frac{\max(|\mathbf{W}_{\text{fp32}}|)}{127}$$

精度损失大约 2-3%，但 inference 快 2-4 倍，memory 减半。OpenVLA 在 Jetson Orin 上 INT8 量化后保持 97% task success，控制频率达到 30 Hz。

更激进的 INT4（GPTQ, AWQ）可以再快 2 倍，但 VLA 对 precision 敏感，精细 manipulation 会有明显 degradation。

### LoRA 推理时合并

LoRA 训练时 $\mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}$，推理时可以 merge：

$$\mathbf{W}_{\text{merged}} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}$$

merge 后的矩阵和原矩阵 shape 一样，inference 零开销。但好处是你可以在同一个 base model 上挂多个 LoRA adapter，每个对应一个 task，切换 task 只需换 adapter。

### Token 缩减

400 个 vision tokens 太多了。很多 token 对应背景区域，对 action prediction 没用。

VLA-Cache 的思路：如果连续几帧视觉没怎么变，就复用之前的 vision tokens，不重新 encode。这对固定 camera 的 manipulation task 很有效，inference 快 40-50%。

Token merging（类似 ToMe）把相似的 vision token 合并，也可以减少 sequence length。

### FAST + Parallel Decoding

前面讲过 FAST 把 1000ms action 压成 16 token。再加上 parallel decoding（一次 forward 预测多个 token），Pi-0 Fast 达到 200 Hz。

但 parallel decoding 有个 subtle issue：autoregressive 的每步 conditioning on 之前所有 token，parallel decoding 打破了这个 causal chain，可能引入 inconsistency。所以 fast 但 trajectory smoothness 会稍差。

---

## Safety：VLA 怎么保证不伤人？

这是 deployment 的最后一公里。一个 7B VLA 在实验室 success rate 95%，但部署到医院、家庭、工厂，5% 的 failure 可能导致严重后果。

SafeVLA（2025）把问题形式化为 Constrained MDP：

$$\max_\pi \mathbb{E}_\pi \left[ \sum_t \gamma^t r(s_t, a_t) \right] \quad \text{s.t.} \quad \mathbb{E}_\pi \left[ \sum_t \gamma^t c_i(s_t, a_t) \right] \leq d_i$$

其中 $c_i$ 是 cost function（比如 collision risk, force exceedance），$d_i$ 是 threshold。用 Lagrangian 方法转成 unconstrained：

$$\mathcal{L}(\pi, \boldsymbol{\lambda}) = \mathbb{E}_\pi \left[ \sum_t \gamma^t \left( r(s_t, a_t) - \sum_i \lambda_i c_i(s_t, a_t) \right) \right]$$

$\lambda_i$ 是 Lagrange multiplier，动态调整：如果 cost 超过 threshold，增大 $\lambda_i$ 让 policy 更保守。

实验显示 unsafe behavior 减少 80%。但论文也承认，定义"什么是 unsafe"本身就是 open problem。碰撞是 unsafe，但太保守导致 task failure 也是一种 "unsafe"（比如手术 robot 因为太谨慎而无法完成手术）。

**更 practical 的 safety 方案是 hierarchical shield：**

```
VLA policy → proposed action
    ↓
Safety shield (CBF / MPC / rule-based)
    ↓
if safe: execute
if unsafe: modify or reject
```

Control Barrier Function（CBF）定义一个 safety set $\mathcal{C} = \{x : h(x) \geq 0\}$，保证 state 永远不离开 $\mathcal{C}$：

$$\dot{h}(x) \geq -\alpha h(x)$$

这个方案的好处是：VLA 可以大胆 propose action，shield 负责兜底。VLA 不需要自己学 safety，降低了训练难度。

---

## World Model：让 VLA 会"想象"

当前 VLA 的一个大问题：它是 reactive 的，不是 predictive 的。它看到当前 frame，直接 predict action，不会"想一下如果我这么做会怎样"。

World model 就是让 VLA 有了 "imagination"：

$$\hat{s}_{t+1} = f_\phi(s_t, a_t)$$

给定当前 state 和 action，预测下一步 state。有了这个，VLA 可以做：

1. **Model-predictive control**: 在 world model 里 rollout 多个候选 action，选最好的。
2. **Counterfactual reasoning**: "如果我从左边抓会怎样？从右边呢？"
3. **Failure anticipation**: 预测 action 会不会导致 failure，提前调整。
4. **Data augmentation**: 在 world model 里生成 synthetic experience。

3D-VLA（2024）就是 VLA + world model 的尝试，用 generative model 预测 3D scene 的变化。

但 world model 本身很难学。物理世界太复杂了：摩擦、接触、变形、碰撞，每一个都是 nonlinear dynamics。Current world model 在简单 task（push block）上 work，复杂 task（fold cloth）还很远。

Differentiable physics simulator（Brax, MuJoCo MJX）提供了一条路：把物理引擎嵌入 training loop，让 world model 和 physics laws 一致。但这还处于早期阶段。

---

## Cross-Embodiment：一个 model 控制所有 robot

现在每个 robot 形态都要训一个 VLA。Franka arm 一个，UR5 一个，humanoid 一个，quadruped 一个。这不 scale。

理想状态：一个 VLA 可以控制任何 robot，只需要告诉它 robot 的 morphology spec。

$$\pi_\theta(\mathbf{a} | \mathbf{o}, \mathbf{T}, \text{embodiment\_spec})$$

怎么做？关键是找到一个 embodiment-agnostic 的 action representation。

不用 joint angles（每个 robot 的 joint 定义不同），改用 task-space representation：

$$\text{action} = \{\text{end-effector pose}, \text{contact force}, \text{grasp configuration}\}$$

这样不管你是 7-DoF arm 还是 5-finger hand，action 都是在 task space 里定义的。Inverse kinematics 把 task-space action 转成具体 robot 的 joint command。

Open X-Embodiment 数据集就是往这个方向努力：22 种不同 robot 的数据统一格式，让一个 model 可以 cross-embodiment 学习。Octo 在这个数据集上训练，可以 generalize 到没见过的 robot。

但 cross-embodiment 的难点是：不同 robot 的 dynamics 差异很大。一个 KUKA arm 的惯量和 ABB arm 完全不同，同样的 task-space command 产生不同的 motion。要让 VLA 理解这些 dynamics 差异，可能需要 meta-learning 或 system identification。

参考：https://robotics-transformer-x.github.io/

---

## Evaluation：我们到底该怎么衡量 VLA？

当前 VLA paper 基本只报 task success rate。"pick apple" 成功了就算 1，失败算 0。但这掩盖了很多问题：

**一个"成功"的 pick apple 可能是：**
- 路径最优，一次成功
- 路径歪歪扭扭，碰了 3 次桌子但最终抓到了
- 差点掉下来，最后侥幸稳住了

这三种"成功"在 deployment 中的风险完全不同。

论文里提到的新 evaluation 维度：

1. **Safety violation rate**: 执行过程中触发 safety boundary 的比例
2. **Near-miss rate**: 差点出事但侥幸没出事的比例
3. **Recovery success rate**: failure 后能否 recover
4. **Trajectory smoothness**: action 的 jerk（加加速度）指标
5. **Latency distribution**: 不是平均 latency，而是 P99 latency（最慢 1% 的 case）
6. **Energy consumption**: 每个 task 消耗多少 J
7. **OOD robustness curve**: 随 distribution shift 增大，performance 怎么衰减

VLA-Arena（2025）开始往这个方向走，但还不够。一个好的 VLA benchmark 应该像自动驾驶的 NuScenes 一样标准化，让不同 model 在同一条件下比较。

---

## 我对 VLA 未来发展的几个直觉

### 直觉一：VLA 的"GPT-3 时刻"还没到

RT-2 像 GPT-2：证明了 idea work，但规模和性能都还不够。OpenVLA 像 GPT-3：开源、可复现、7B 参数已经能做很多事。但 VLA 的 GPT-4——一个在 web-scale + robot-scale 数据上训练、能 zero-shot 适应任何 robot task 的 model——还没出现。

这个 model 可能需要：
- 100B+ 参数的 backbone
- 10B+ robot demonstration（跨 embodiment）
- 1T+ web VQA data
- 高效的 co-fine-tuning recipe

### 直觉二：Tokenization 会越来越 important

现在 action tokenization 还很粗糙（256 bins 或 DCT+BPE）。未来可能会有 learned tokenizer，类似 LLM 的 SentencePiece，但针对 action trajectory 优化。

一个可能的 direction：hierarchical tokenization。High-level token 表示 "grasp object"（一个 skill），mid-level 表示 "move to pose (x,y,z)"，low-level 表示 joint angles。不同 level 的 token 用不同 granularity，VLA 可以在不同 level 上 reasoning。

### 直觉三：VLA + World Model 是 long-horizon 的 key

纯 reactive VLA 做不了 long-horizon task。Dual-system 缓解了这个问题，但 System 2 的 LLM 还是基于 language reasoning，没有 physical dynamics 的 grounding。

World model 让 VLA 有了 "physical imagination"：可以在脑中 simulate "如果我做 X，物体会怎么动"。这对 long-horizon planning 是 game-changer。

想象一个 VLA 做 "pack groceries into bag"：
1. World model simulate：先放牛奶会怎样？包会歪。
2. World model simulate：先放苹果会怎样？苹果会滚。
3. World model simulate：先放面包会怎样？面包会被压。
4. Decision：先放重的东西在底部。

这种 reasoning 需要 world model 提供 "what-if" 的 ability。当前 world model 还太弱，但 3D-VLA、DynaPoint 等 work 在往这个方向走。

### 直觉四：Agentic learning 是 deployment 的关键

当前 VLA 是 "train once, deploy forever"。但 real world 是 non-stationary 的：新物体会出现，环境会变化，user preference 会变。

未来 VLA 需要 agentic learning loop：
1. Deploy → collect data
2. Self-assess → identify failure cases
3. Self-improve → 用 RL 或 self-supervised learning 更新
4. Safety verify → 确保更新不破坏已有能力
5. Re-deploy

这个 loop 的难点是 catastrophic forgetting：学了新东西忘了旧的。Replay buffer、EWC、modular adapter 是可能的 solution。

### 直觉五：VLA 会让 robot 像 LLM 一样"promptable"

想象未来你买了一个 home robot，不需要编程，只需要对它说话：

"帮我把厨房收拾一下，碗放 dishwasher，杯子放柜子，桌子擦一下。"

VLA 会：
1. LLM 分解 task：[find dishes, put in dishwasher, find cups, put in cabinet, wipe table]
2. Vision 识别场景：定位碗、杯子、桌子
3. Diffusion policy 执行每一步
4. 遇到没见过的杯子也能 generalize（因为 web pretraining）
5. 遇到困难（杯子在很高的架子上）会自主 re-plan

这就是 VLA 的终极愿景：**让 robot 像使用 ChatGPT 一样简单**。

---

## 一句话总结

VLA 的核心 insight 是 **把 robot control 变成 sequence prediction**。通过把 vision、language、state、action 全部 tokenize，用 Transformer 的 next-token prediction 统一训练，VLA 让 LLM 的 scale 和 generalization 能力第一次真正 flow 到 robotics。

当前 VLA 的发展阶段类似 2018 年的 LLM（GPT-2 刚出来）：idea 已经被验证，但 scale、data、efficiency 都还在早期。未来 5 年如果能在 data scale、world model、cross-embodiment 三个方向突破，VLA 有可能成为 robotics 的 "GPT-3 时刻"。

参考：
- VLA Review GitHub: https://github.com/Real-World-System-Identification/VLA-Review
- RT-2: https://robotics-transformer2.github.io/
- OpenVLA: https://openvla.github.io/
- Pi-0: https://arxiv.org/abs/2410.24164
- GR00T N1: https://arxiv.org/abs/2503.14734
- FAST: https://arxiv.org/abs/2501.09747
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- Octo: https://octo-models.github.io/
- CogACT: https://arxiv.org/abs/2411.19650
- SafeVLA: https://arxiv.org/abs/2503.03480
- DINOv2: https://arxiv.org/abs/2304.07193
- LoRA: https://arxiv.org/abs/2106.09685
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

---

# Vision-Language-Action (VLA) Models Review Paper - 深度解析

这是一篇系统性的 review paper，覆盖 2022-2025 年间超过 80 个 VLA 模型，由 Cornell University、HKUST 和 University of Peloponnese 联合发表。整体叙事从 isolated modalities 演化到 unified embodied agents，构建了一个从 concepts → progress → applications → challenges → future roadmap 的完整逻辑闭环。

参考链接：
- GitHub Repo: https://github.com/Real-World-System-Identification/VLA-Review
- RT-2 Blog: https://robotics-transformer2.github.io/
- OpenVLA: https://openvla.github.io/
- GR00T N1: https://arxiv.org/abs/2503.14734
- Pi-0: https://arxiv.org/abs/2410.24164
- Octo: https://octo-models.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/

---

## 1. Conceptual Foundations: 从 modality 隔离到 unified token space

### 1.1 三个演化阶段的核心 intuition

**Stage 1: Foundational Integration (2022-2023)**

早期 VLA 的核心 insight 在于把 CLIP-style 的 semantic embedding 直接和 motion primitives 拼接。CLIPort (2022) 是典型代表，其本质可以写成：

$$\pi_\theta(\mathbf{a} | \mathbf{I}, \mathbf{T}) = \text{TransporterNet}\left(\text{CLIP}_{\text{visual}}(\mathbf{I}), \text{CLIP}_{\text{text}}(\mathbf{T})\right)$$

其中 $\mathbf{I}$ 是 RGB image，$\mathbf{T}$ 是 language instruction，$\mathbf{a}$ 是 SE(2) 的 pick-and-place action。这里的 intuition 是：CLIP 预训练已经把"red block"和对应的 visual patch 对齐到同一个 embedding space，policy network 只需要在这个空间里做 spatial transport。

RT-1 (2022) 把 action 离散化成 256 个 bins，每个 joint 维度独立 tokenize，然后用 EfficientNet + Universal Sentence Encoder + Transformer 的 stack。RT-1 的 130k demonstrations 让 model 在 kitchen manipulation 上达到 97% success rate，但缺乏 compositional reasoning。

**Stage 2: Specialization and Embodied Reasoning (2024)**

这个阶段的核心突破是把 visual chain-of-thought 引入到 action prediction。RT-2 (2023, Google DeepMind) 把 action tokens 直接作为 language tokens 的扩展，用 Discrete Cosine Transform (DCT) 压缩 action trajectory，用 Byte-Pair Encoding (BPE) 离散化：

$$\mathbf{a}_{tokenized} = \text{BPE}\left(\text{DCT}\left(\mathbf{a}_{1:T}\right)\right)$$

其中 $\mathbf{a}_{1:T}$ 是 T 步的连续 action trajectory，DCT 把时域信号转换到频域后低频系数集中大部分能量，BPE 把频域系数聚合成离散 token。这个设计让 RT-2 可以用 PaLI-X (55B) 或 PaLM-E (562B) 的 language backbone 直接 autoregressive 地预测 action，把 robot control 变成 next-token prediction。

VoxPoser (2023) 走了完全不同的路线：用 LLM (GPT-4) + VLM (ViLD, MDETR) 组合产生 3D voxel value map，再调 classical MPC 生成 trajectory。这种 neuro-symbolic 的设计实现了 zero-shot manipulation，但牺牲了 end-to-end 的 gradient flow。

Octo (2024) 引入 diffusion head 作为 action decoder，在 Open X-Embodiment 的 800k episodes 上训练 93M 参数的 generalist policy。Diffusion policy 的核心公式：

$$\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{t, \mathbf{a}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{a}_t, t, \mathbf{o})\|^2\right]$$

其中 $\mathbf{a}_t = \sqrt{\bar{\alpha}_t}\mathbf{a}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$ 是 forward process 加噪后的 action，$\boldsymbol{\epsilon}_\theta$ 是要学习的 noise prediction network，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ 是 cumulative noise schedule。Diffusion 的优势在于可以表达 multimodal action distribution，避免 MLP head 的 mode collapse。

**Stage 3: Generalization and Safety-Critical Deployment (2025)**

最新一波工作集中在 robustness、safety、cross-embodiment 三个方向。SafeVLA (2025) 把问题形式化为 Constrained MDP：

$$\max_\pi \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t)\right] \quad \text{s.t.} \quad \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t c_i(s_t, a_t)\right] \leq d_i, \forall i$$

其中 $c_i$ 是第 $i$ 个 cost function（比如 collision risk），$d_i$ 是对应的 threshold，用 Lagrangian relaxation 转换为 unconstrained 问题：

$$\mathcal{L}(\pi, \boldsymbol{\lambda}) = \mathbb{E}_\pi\left[\sum_t \gamma^t (r(s_t,a_t) - \sum_i \lambda_i c_i(s_t, a_t))\right]$$

实验显示 unsafe behavior 减少超过 80%。

GR00T N1 (NVIDIA, 2025) 引入 dual-system 设计：System 1 是 diffusion-based 10ms latency 的 low-level controller，System 2 是 LLM-based 的高层 planner。这是直接借鉴 Kahneman 的 System 1 / System 2 框架。

### 1.2 Token 三件套：Prefix / State / Action

这是 VLA 区别于普通 VLM 的关键 design choice，本质上是把 robot 的整个"存在状态"都用 token 表示，让 Transformer 可以用统一的 attention 机制处理。

**Prefix Tokens** 编码 (vision, language) 的 joint context。以 Figure 7 的 "stack the green blocks" 为例：

$$\mathbf{V} = \text{ViT}(\mathbf{I}) \in \mathbb{R}^{N_v \times d_v}, \quad N_v=400, d_v=1024 \text{ (typical)}$$
$$\mathbf{L} = \text{BERT}(\mathbf{T}) \in \mathbb{R}^{N_l \times d_l}, \quad N_l=12, d_l=768$$

这里 $N_v=400$ 对应 ViT-L/14 的 patch tokens (224×224 输入，14×14 patch + CLS token)，$N_l=12$ 是 BERT-base 的 12 层 hidden states 投影后的 token 数。

**State Tokens** 编码 robot 的 proprioceptive state：joint angles $\boldsymbol{\theta}$, gripper status, force-torque readings。Algorithm 1 中：

$$\mathbf{S} = \text{MLP}(\boldsymbol{\theta}) \in \mathbb{R}^{d_s}, \quad d_s=64$$

注意 state 是被压缩成单个 token（或少数几个 token）的，这和 NaVILA 等模型把每个 joint 维度都做成独立 token 的设计形成对比。压缩比是个 trade-off：太紧凑会丢失精度，太冗余会拖慢 inference。

**Action Tokens** 是 autoregressive 生成的。论文中的 Algorithm 1:

```
V ← ViT(I)              ▷ 400 vision tokens
L ← BERT(T)             ▷ 12 language tokens  
S ← MLP(θ)              ▷ 64-dim state encoding
F ← CrossAttention(V, L, S)  ▷ 512-dim fused token
A ← FAST(F)             ▷ 50 action tokens
Output: τ_{1:N}          ▷ motor commands
```

这里关键的是 FAST tokenizer 把连续的 motor command trajectory $\boldsymbol{\tau}_{1:N}$ 压缩成 50 个离散 token。FAST 的核心是 DCT + bit-pair encoding，把 1000ms 的 action window 压缩成 16 个 token，让 Pi-0 Fast 达到 200 Hz 控制频率。

**Cross-attention fusion 的数学**：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

其中 $\mathbf{Q} \in \mathbb{R}^{N_q \times d_k}$ 来自 query modality（比如 state），$\mathbf{K}, \mathbf{V} \in \mathbb{R}^{N_{kv} \times d_k}$ 来自 key-value modality（比如 vision+language）。Multi-head 版本：

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}^O$$
$$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$$

### 1.3 Autoregressive Action Generation 的核心公式

VLA 把 robot control 变成 sequence modeling，action prediction 的 likelihood 是：

$$P(\mathbf{a}_{1:N} | \mathbf{I}, \mathbf{T}, \boldsymbol{\theta}) = \prod_{i=1}^{N} P(\mathbf{a}_i | \mathbf{a}_{<i}, \mathbf{I}, \mathbf{T}, \boldsymbol{\theta}; \phi)$$

其中 $\phi$ 是 model parameters，$\mathbf{a}_{<i} = (\mathbf{a}_1, \ldots, \mathbf{a}_{i-1})$ 是已经生成的 action tokens。训练时用 teacher forcing：

$$\mathcal{L}_{\text{AR}} = -\sum_{i=1}^{N} \log P(\mathbf{a}_i^* | \mathbf{a}_{<i}^*, \mathbf{I}, \mathbf{T}, \boldsymbol{\theta}; \phi)$$

其中 $\mathbf{a}_i^*$ 是 ground truth action token。

这个 formulation 的漂亮之处在于：你可以直接用 LLM 的训练 pipeline（cross-entropy + autoregressive decoding）来训练 robot policy。RT-2, OpenVLA, GR00T N1 都是这个范式。

---

## 2. Architectural Innovations: 三大范式深度对比

### 2.1 Early Fusion Models (EF-VLA, ICLR 2025)

**核心 insight**：在 transformer backbone 之前就把 vision 和 language 融合，保持 CLIP 学到的 representational alignment。

EF-VLA 用 frozen CLIP encoders 处理 image-text pairs，然后在 transformer 早期层注入 fused embedding：

$$\mathbf{h}_0 = \text{Concat}\left(\text{CLIP}_{\text{vis}}(\mathbf{I}), \text{CLIP}_{\text{txt}}(\mathbf{T})\right)$$
$$\mathbf{h}_{l+1} = \text{TransformerLayer}_l(\mathbf{h}_l), \quad l = 0, \ldots, L-1$$
$$\mathbf{a} = \text{PolicyHead}(\mathbf{h}_L)$$

关键 trade-off：frozen backbone 让 CLIP 的 alignment 不被 robot data 稀释，但代价是无法 task-specific 微调 visual representation。EF-VLA 在 compositional manipulation 上比 baseline 提升 20%，unseen goal descriptions 上达到 85% success。

这种设计呼应了 Flamingo (Alayrac et al., 2022) 的 gated cross-attention：keep foundation model frozen，只在 interface 层学习。

### 2.2 Dual-System Architectures (GR00T N1)

GR00T N1 的架构可以画成：

```
[Language: "clean the table"]
        ↓
    System 2 (LLM)
    - Task decomposition
    - Skill composition
        ↓
    [subtask_1, subtask_2, ..., subtask_k]
        ↓
    System 1 (Diffusion Policy, 10ms)
    - Low-level motor control
    - Reactive grasping
        ↓
    [motor commands at 100 Hz]
```

System 2 用 LLM 做高层 planning，把 long-horizon goal 分解成 atomic subtasks；System 1 是 diffusion policy，以 10ms latency 做精细控制。两者通过 queue/buffer 解耦：

$$\text{subtask}_t = \text{System2}(\text{goal}, \text{history}), \quad \text{if } t \mod T_{\text{plan}} = 0$$
$$\mathbf{a}_t = \text{System1}(\text{subtask}_{\lfloor t/T_{\text{plan}} \rfloor}, \mathbf{o}_t, \boldsymbol{\theta}_t)$$

其中 $T_{\text{plan}}$ 是 planning 周期。实验显示在 multi-stage household manipulation 上比 RT-1, RT-2, OpenVLA 提升 17% success rate，collision failure 减少 28%。

这个设计也呼应了 Helix (Figure AI) 的 200Hz 全身控制：System 2 用 7-9 Hz 的 VLM 做 reasoning，System 1 用 200 Hz 的 transformer 做 motor control。

### 2.3 Self-Correcting Frameworks (SC-VLA)

SC-VLA 引入 dual-path inference：

- **Fast path**：lightweight transformer 直接 predict action
- **Slow path**：检测到 failure 时触发 chain-of-thought reasoning

Failure detection 可以基于：
1. **Action confidence**：$H(\mathbf{a}) = -\sum_i P(\mathbf{a}_i) \log P(\mathbf{a}_i) > \tau_H$
2. **State prediction error**：$\|\mathbf{s}_{t+1} - \hat{\mathbf{s}}_{t+1}\| > \tau_s$
3. **External signal**：force-torque 异常、collision detection

SC-VLA 在 cluttered environment 上把 failure rate 降低 35%。这个范式和 LLM 的 self-correction (Constitutional AI, Self-Refine) 思路一致。

### 2.4 三大范式的对比 Table

| 范式 | 代表模型 | Training cost | Inference latency | Generalization | Safety |
|------|----------|---------------|-------------------|-----------------|--------|
| Early Fusion | EF-VLA | Low (frozen backbone) | Medium | High on compositional tasks | Medium |
| Dual-System | GR00T N1, Helix | High (two systems) | Layered (10ms + LLM) | High on long-horizon | High (planning-level) |
| Self-Correcting | SC-VLA | Medium | Adaptive | High on edge cases | High (failure recovery) |

---

## 3. Tokenization & Encoding 深度解析

### 3.1 Vision Tokenization

主流 VLA 用 ViT-L/14 或 ConvNeXt 把 224×224 RGB 转成 patch tokens：

$$\mathbf{V} = \text{ViT}\left(\text{PatchEmbed}(\mathbf{I}) + \text{PosEmbed}\right) \in \mathbb{R}^{(14\times 14 + 1) \times 1024}$$

其中 14×14 = 196 个 spatial patches + 1 个 CLS token = 197 tokens。但 RT-2 用 ViT-22B 时是 400+ tokens，因为用了更高分辨率。

DINOv2 (Oquab et al., 2023) 是 self-supervised pretraining，提供 dense visual features，比 CLIP 更适合 dense prediction tasks。OpenVLA 同时用 DINOv2 + SigLIP 双 vision encoder，融合公式：

$$\mathbf{V}_{\text{fused}} = \text{LN}\left(\text{DINOv2}(\mathbf{I}) + \text{SigLIP}(\mathbf{I})\right)$$

这种 ensemble 设计让 model 既有 self-supervised 的 dense feature（DINOv2），又有 language-aligned 的 semantic feature（SigLIP）。

### 3.2 Action Tokenization 的三种方案

**方案 A: Discrete binning (RT-1, OpenVLA)**

每个 action dimension 独立离散化成 256 bins：

$$\mathbf{a}_{\text{disc}}^{(j)} = \text{round}\left(\frac{\mathbf{a}^{(j)} - \mu_j}{\sigma_j} \times 127 + 128\right) \in \{0, 1, \ldots, 255\}$$

其中 $\mu_j, \sigma_j$ 是 dataset 的 per-dimension mean/std。优点是简单，缺点是 7-DoF arm + gripper 需要 8 个 token，每个 256 bins，表达精度有限（~1/256 = 0.4%）。

**方案 B: Diffusion head (Octo, Pi-0, RDT-1B)**

直接在 continuous space 用 diffusion 生成：

$$\mathbf{a}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{a}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{a}_t, t, \mathbf{c})\right) + \sigma_t \mathbf{z}$$

其中 $\mathbf{c}$ 是 conditioning（vision+language+state），$\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$，$\sigma_t$ 是 noise scale。优点：multimodal action distribution，缺点：inference 慢（需要 50-100 步 denoising）。

**方案 C: FAST (Pi-0 Fast)**

用 DCT 把 action trajectory 转到频域，保留低频系数，再 BPE 编码：

$$\mathbf{A}_{\text{freq}} = \text{DCT-II}\left(\mathbf{a}_{1:T}\right), \quad A_k = \sum_{n=0}^{T-1} a_n \cos\left(\frac{\pi}{T}\left(n + \frac{1}{2}\right)k\right)$$

保留前 $K$ 个低频系数（$K \ll T$），然后用 BPE 把频域系数聚合成离散 token。1000ms action window → 16 tokens，inference 速度提升 15×，达到 200 Hz。

### 3.3 State Tokenization 的设计空间

**Single token (Algorithm 1)**: $\mathbf{S} = \text{MLP}(\boldsymbol{\theta}) \in \mathbb{R}^{64}$

**Per-joint tokens (某些 models)**: 7-DoF arm = 7 tokens，每个 joint 一个 token。优点：attention 可以 per-joint reasoning，缺点：序列长，inference 慢。

**Hierarchical tokens (Helix)**: 把 state 分成 body frame + arm frame + hand frame 三层，每层一个 token，用 structural prior 减少 sequence length。

---

## 4. Training Paradigms: 从 Behavior Cloning 到 Agentic Learning

### 4.1 Co-fine-tuning 的核心 insight

RT-2 的关键创新是 co-fine-tuning：同时在 web-scale vision-language data 和 robot trajectory data 上训练：

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{VQA}}(\text{web data}) + \lambda_2 \mathcal{L}_{\text{AR}}(\text{robot data})$$

其中 $\mathcal{L}_{\text{VQA}}$ 是 visual question answering loss，$\mathcal{L}_{\text{AR}}$ 是 action token prediction loss。这种 mixing 让 language reasoning 能力 "flow" 到 action prediction，实现 emergent generalization：模型可以理解 "pick the red cup left of the bowl" 这种 combinatorial instruction，即使没见过这个具体组合。

OpenVLA (7B) 用同样的策略在 OXE + DROID 上训练，比 RT-2-X (55B) 提升 16.5% success rate，证明 co-fine-tuning 比 pure scaling 更重要。

### 4.2 Parameter-Efficient Adaptation (LoRA)

LoRA (Hu et al., 2022) 把 weight update 分解成低秩矩阵：

$$\mathbf{W} = \mathbf{W}_0 + \Delta\mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}$$

其中 $\mathbf{W}_0 \in \mathbb{R}^{d \times d}$ 是 frozen pre-trained weight，$\mathbf{B} \in \mathbb{R}^{d \times r}$, $\mathbf{A} \in \mathbb{R}^{r \times d}$, $r \ll d$ 是 trainable。参数量从 $d^2$ 降到 $2dr$。

OpenVLA 用 LoRA 后可训练参数减少 70%，GPU hours 减少 70%。Pi-0 Fast 只用 10M adapter 参数 + frozen backbone，达到 200 Hz control。

LoRA 的 forward pass：

$$\mathbf{y} = \mathbf{W}_0 \mathbf{x} + \mathbf{B}\mathbf{A}\mathbf{x} = (\mathbf{W}_0 + \mathbf{B}\mathbf{A})\mathbf{x}$$

可以 merge 成 single matrix 实现 zero-overhead inference：

$$\mathbf{W}_{\text{merged}} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}$$

### 4.3 Reinforcement Learning Fine-tuning (ConRFT, GRPO)

ConRFT (2025) 结合 behavior cloning 和 Q-learning：

$$\mathcal{L}_{\text{ConRFT}} = \mathcal{L}_{\text{BC}} + \lambda Q_{\text{target}}(\mathbf{s}, \mathbf{a})$$

其中 Q-target 用 human-in-the-loop fine-tuning 提供 reward signal，在 8 个 contact-rich tasks 上达到 96.3% success。

GRPO (Group Relative Policy Optimization) 是 PPO 的简化版，去掉 value network：

$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^G \sum_t \log \pi_\theta(a_{i,t}|s_{i,t}) \cdot A_{i,t}$$

其中 $A_{i,t} = \frac{R_i - \text{mean}(R)}{\text{std}(R)}$ 是 group-normalized advantage。这种 design 在 VLA fine-tuning 上比 PPO 更 stable。

### 4.4 Sim-to-Real Transfer

UniSim (Yang et al., 2023) 用 neural closed-loop sensor simulator 生成 photorealistic scenes：

$$\mathbf{I}_{\text{aug}} = \text{UniSim}(\text{scene description}, \text{lighting}, \text{occlusion level})$$

Domain randomization 公式：

$$\theta_{\text{train}} \sim p(\theta), \quad p(\theta) = \mathcal{U}(\theta_{\min}, \theta_{\max})$$

覆盖 friction, mass, lighting, texture 等参数。GraspVLA 在 1B synthetic action data 上 pretrain，实现 zero-shot sim-to-real transfer。

---

## 5. Inference Acceleration: 让 7B VLA 跑在 edge 上

### 5.1 Quantization

INT8 quantization：

$$\mathbf{W}_{\text{int8}} = \text{round}\left(\frac{\mathbf{W}_{\text{fp32}}}{s}\right), \quad s = \frac{\max(|\mathbf{W}_{\text{fp32}}|)}{127}$$

OpenVLA 在 Jetson Orin 上 INT8 quantization 保留 97% task success，控制频率达到 30 Hz。

更激进的 INT4 quantization (GPTQ, AWQ) 可以进一步压缩，但 VLA 对 precision 敏感，sub-millimeter manipulation 会明显 degradation。

### 5.2 Pruning

Structured pruning 移除冗余 attention heads：

$$\text{importance}_h = \frac{1}{N}\sum_{i=1}^N \|\text{head}_h(\mathbf{x}_i)\|_2$$

移除 importance 最低的 20% heads。在 diffusion-based policy 上 grasp stability 几乎无影响。RDT-1B pruning 25% 可以 sub-4GB deployment，task success drop <2%。

### 5.3 Parallel Decoding

Standard autoregressive decoding：

$$\mathbf{a}_i = \text{Decode}(\mathbf{a}_{<i}, \mathbf{c})$$

每步需要 forward pass，N 个 token 需要 N 次 forward。

Parallel decoding (GR00T N1) 一次性预测多个 tokens：

$$\mathbf{a}_{i:i+k} = \text{ParallelDecode}(\mathbf{a}_{<i}, \mathbf{c})$$

通过 prefix-tuning 或 masked prediction 实现。Pi-0 Fast 通过 FAST tokenization 把 1000ms 压缩到 16 tokens，加上 parallel decoding，达到 200 Hz。

Action chunking 把 multi-step routine 抽象成 single token：

$$\text{pick-and-place}(\text{obj}, \text{pos}) \to \text{single token}$$

减少 inference steps 40%。

### 5.4 Hardware-Aware Compilation

TensorRT-LLM 的优化：
1. Kernel fusion: $\text{QK}^\top \mathbf{V}$ 合并成 single kernel
2. Quantization-aware scheduling
3. Memory-optimized attention (FlashAttention-3)

OpenVLA-OFT 在 RTX GPU 上 inference latency 减少 30%，能耗减少 25%。

### 5.5 Edge VLA 的极限设计

TinyVLA (50M params) 设计：

```
FastViT backbone (compact 128-d language)
    ↓
Diffusion decoder (50M)
    ↓
5× faster inference
```

MoManipVLA、Edge VLA 都在 Jetson-class GPU 上达到 30-50 Hz inference，逼近 OpenVLA 性能。这是部署到 mobile robot 的关键。

---

## 6. Applications Deep Dive

### 6.1 Humanoid Robotics: Helix 案例解析

Figure AI 的 Helix 是 dual-system 设计的 SOTA：

**System 2 (7-9 Hz reasoning)**:
- Input: RGB-D streams + language command
- VLM: SigLIP / DINOv2
- LLM: LLaMA-4
- Output: task plan + sub-goal specification

**System 1 (200 Hz control)**:
- Input: sub-goal + current state
- Transformer visuomotor policy
- Output: dense action vector for 全身上肢

应用场景（Figure 12）: "Please take the water bottle from the fridge"
1. SigLIP/DINOv2 segment 视觉 scene，识别 fridge, handle, bottle
2. LLaMA-4 tokenize instruction，fuse with visual context
3. High-level planner: [locate handle, pull door, identify bottle, grasp]
4. Mid-level planner: grasp type, joint trajectories
5. Low-level VLA controller (diffusion policy): sub-second latency 执行
6. Agentic AI module: 实时 grip 调整（应对 tilted bottle / slippery grip）

这种 hierarchical 设计让 Helix 可以 generalize 到 unseen objects 和 tasks。

### 6.2 Autonomous Vehicles: OpenDriveVLA & ORION

OpenDriveVLA 的架构：

$$\mathbf{F}_{\text{hierarchical}} = \text{CrossAttn}\left(\text{2D tokens}, \text{3D tokens}, \text{language tokens}\right)$$

用 multi-view camera inputs (6 个 cameras 典型) + BEV (Bird's Eye View) representation。Autoregressive decoder 同时输出：
- Action plan (steering angle, acceleration)
- Trajectory visualization（人可解释）

在 nuScenes 和 Waymo Open Motion dataset 上 SOTA。

ORION 的核心组件：
- QT-Former: long-horizon visual context aggregation
- LLM: traffic narrative reasoning
- Generative trajectory planner

ORION 的优势在于处理 ambiguous instructions: "take the exit behind the red truck" 需要 multi-step reasoning + occlusion reasoning。

### 6.3 Healthcare: RoboNurse-VLA

RoboNurse-VLA 的 pipeline：

```
Voice input → STT → LLaMA-2 → semantic command
    ↓
RGB-D + SAM2 → scene segmentation
    ↓
Pose regression + gripper classifier
    ↓
Real-time instrument handover
```

在手术场景中实现 90%+ success rate，robust to novel tools 和 dynamic OR (operating room) scenes。LoRA-based fine-tuning 让 model 可以快速 adapt 到不同 hospital workflow。

### 6.4 Industrial Robotics: CogACT

CogACT 是 modular VLA 设计：

```
DINOv2 ViT-L/14 + SigLIP ViT-So400M/14
    ↓
Prismatic-7B (vision-language encoder)
    ↓
DiT-Base (300M diffusion action transformer)
```

关键创新：用 diffusion transformer 而不是 autoregressive head 做 action decoding，可以表达 multimodal action distribution。在 real-world manipulation 上比 OpenVLA 提升 28%+，可以 rapid adapt 到不同 robot embodiments (6-DoF arm vs bimanual)。

### 6.5 Agriculture: 苹果采摘场景

VLA 在 orchard 的应用（Figure 15）：

```
RGB-D camera + multispectral sensor
    ↓
ConvNeXt / DINOv2 (ripeness detection)
    ↓
T5 / LLaMA parse "pick only Grade A fruits"
    ↓
Action tokens → end-effector 控制
    ↓
LoRA fine-tune to new crop variety
```

Synthetic data (UniSim) 生成 photorealistic orchard scenes 处理 occlusion / lighting variability。Closed-loop feedback 让 model 在 deployment 中持续 improve。

### 6.6 AR Navigation

VLA + AR glasses 的场景（Figure 16）：

```
Smart glasses camera + voice query
"how do I reach Gate 22 without stairs?"
    ↓
ViT scene understanding (escalator detection)
    ↓
LLM route planning
    ↓
AR overlay with directional cues
```

可以迭代 refine: "navigate to pharmacy" → "avoid busy areas" → "take scenic route"。对 visually impaired 用户特别有价值。

---

## 7. Challenges: 五大瓶颈

### 7.1 Real-Time Inference Constraints

Standard autoregressive VLA 在 single GPU 上只能 3-5 Hz，远低于 100+ Hz control 需求。

**计算瓶颈**：
- Vision tokens: 400 tokens × 512 dim = 205k floats = 820 KB
- Cross-attention: $O(N_v^2 \cdot d) = 400^2 \times 512 = 82M$ FLOPs
- Memory bandwidth: 1.2 GB/s for high-dim visual embeddings

**解决方案对比**：

| 技术 | Speedup | Accuracy drop | Hardware |
|------|---------|----------------|----------|
| INT8 quant | 2-4× | <3% | Edge GPU |
| LoRA | 1.5× | <2% | Any |
| FAST tokenization | 15× | <1% | Any |
| Parallel decoding | 2.5× | 5-10% | High-end GPU |
| Pruning (20%) | 1.3× | <2% | Any |

### 7.2 Multimodal Action Representation + Safety

**Discrete tokenization 的精度问题**：256 bins 在 sub-millimeter manipulation 会有 0.4% 误差，对 surgical robotics 是 unacceptable 的。

**MLP head 的 mode collapse**：当多个 action trajectory 都合理时，MLP 会 average 它们，产生无效的中间 action。

**Diffusion head 的计算开销**：50-100 步 denoising ≈ 3× transformer decoder 的 compute。

**Safety 的延迟问题**：Emergency stop 通常需要 200-500ms 安全验证，对高速场景（autonomous driving）是危险的。

**Collision prediction accuracy**: 仅 82% 在 cluttered dynamic scenes。

### 7.3 Dataset Bias + Generalization Gap

**Bias 的具体数据**：
- 17% 的 dataset associations 倾向 stereotypical interpretations ("doctor" → male)
- OpenVLA 在 novel settings 漏掉 23% object references
- Compositional generalization 失败率高（"yellow horse" 之类罕见组合）

**Generalization 退化**：novel tasks 上 performance 退化 40%+。Household-trained VLA 在 industrial/agricultural 场景表现差。

**Curse of distribution shift**: 
$$D_{\text{train}} \neq D_{\text{test}}, \quad \text{KL}(D_{\text{train}} \| D_{\text{test}}) \gg 0$$

解决方案：hard-negative sampling, contrastive fine-tuning, sim-to-real with domain randomization。

### 7.4 System Integration Complexity

**Temporal mismatch**：
- System 2 (LLM): ~800ms latency
- System 1 (controller): ~5ms update
- Ratio: 160×, 需要 careful synchronization

**Feature space misalignment**：
- Vision encoder: 1024-dim tokens
- Action decoder: 7-dim (arm) + 1-dim (gripper)
- Dimension reduction 1024→8 需要保留 semantic content

**Sim-to-real gap**:
$$\text{sim error}: \epsilon_{\text{sim}}, \quad \text{real error}: \epsilon_{\text{real}} = \epsilon_{\text{sim}} + \epsilon_{\text{domain\_shift}}$$

**Compute budget**: 7B+ params 需要 28GB+ VRAM，远超 edge device (Jetson Orin 32GB 紧张)。

### 7.5 Robustness + Ethics

**Environmental robustness**：
- Low-contrast scenes: 20-30% accuracy drop (OpenDriveVLA)
- Acoustic noise: language understanding deterioration (CoVLA)
- Occlusion: pose estimation errors (RoboMamba)

**Ethical concerns**：
- Privacy: AR navigation 记录 visual data
- Bias: 在 medical/hiring applications 放大社会偏见
- Accountability: autonomous decision 的 liability 归属
- Workforce displacement: 机器人取代 human labor 的社会影响

---

## 8. Future Roadmap: 6 大方向

### 8.1 Multi-modal Foundation Model 作为 "Cortex"

构想：unified multimodal foundation model 同时编码 semantics + dynamics + contact priors + common sense physics。

$$\mathbf{h}_{\text{cortex}} = f_\theta(\mathbf{I}, \mathbf{T}, \text{affordance traces}, \text{interaction history})$$

Foundation model 提供 stable semantic anchors，downstream planner/controller 调用。

### 8.2 Agentic Lifelong Learning

**核心 insight**: deployed VLA 应该 self-supervised 持续 improve：

$$\pi_{t+1} = \text{Update}(\pi_t, \text{rollout data}, \text{safety constraints})$$

**Catastrophic forgetting 的解决**：
- Replay buffer: 保留 old task data
- Regularization: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{new}} + \lambda \|\theta - \theta_{\text{old}}\|^2$
- Modular adapters: 不同 task 用不同 LoRA module

**Safety-aware updates**: 每次更新前 verify policy 不违反 safety invariant。

### 8.3 Hierarchical Neuro-Symbolic Planning

**Long-horizon decomposition**:

```
Goal: "make breakfast"
    ↓ LLM decomposition
[get eggs, get pan, crack eggs, cook, serve]
    ↓ mid-level skill policy
[grasp egg skill, crack skill, ...]
    ↓ low-level controller
[motor commands at 100 Hz]
```

**Symbolic verification**: high-level plan 可以用 formal methods 验证 safety constraints：

$$\varphi_{\text{plan}} \models \text{safety spec}$$

**Control Barrier Functions** (CBF) 做 low-level shield：
$$h(\mathbf{x}) \geq 0 \Rightarrow \text{safe}, \quad \dot{h}(\mathbf{x}) \geq -\alpha h(\mathbf{x})$$

### 8.4 World Models + Causal Reasoning

World model 预测 state transition：

$$\hat{\mathbf{s}}_{t+1} = f_\phi(\mathbf{s}_t, \mathbf{a}_t)$$

Counterfactual evaluation: "if I push here, what collides?"

**Differentiable physics**: 把物理 simulator 嵌入 training loop：

$$\mathcal{L}_{\text{physics}} = \|f_\phi(\mathbf{s}_t, \mathbf{a}_t) - \text{Sim}(\mathbf{s}_t, \mathbf{a}_t)\|^2$$

### 8.5 Cross-Embodiment Transfer

**Embodiment-agnostic skill representation**:

不用 joint angles，改用 task-space constraints：

$$\text{skill} = \{\text{contact points}, \text{force profile}, \text{trajectory constraints}\}$$

Wheeled / quadruped / humanoid 共享 skill library，通过 inverse kinematics 转换到具体 embodiment。

Meta-learning 实现 few-shot calibration：

$$\pi_{\text{new robot}} = \text{MetaLearn}(\pi_{\text{base}}, \text{few demos})$$

### 8.6 Evaluation Beyond Task Success

新 metrics:
- **Safety violation rate**: 不安全 action 比例
- **Uncertainty calibration**: $\text{ECE} = \sum_i \frac{|B_i|}{N}|\text{acc}(B_i) - \text{conf}(B_i)|$
- **Recovery behavior**: failure 后的恢复成功率
- **Temporal coherence**: trajectory smoothness
- **Energy consumption**: J/task
- **OOD robustness**: 在 distribution shift 下的 performance curve

---

## 9. 一些容易忽略的技术细节

### 9.1 VLA Tokenization Pipeline 的细节

Algorithm 1 中的 64-dim state encoding 是个 design choice。论文用 single MLP token，但实际 production 系统（如 Helix）会用 hierarchical state tokens:

```
Body frame token (16-dim)
Arm frame token (16-dim)  
Hand frame token (16-dim)
Gripper status token (16-dim)
→ Total: 4 tokens × 16-dim = 64-dim
```

这种 structural prior 让 attention 可以 per-body-part reasoning。

### 9.2 Vision Token 数量的 trade-off

ViT-L/14 224×224 输入 → 196+1 = 197 tokens
ViT-L/14 336×336 输入 → 576+1 = 577 tokens
ViT-22B (RT-2) 高分辨率 → 400+ tokens

Token 数量 $N_v$ 对 inference 的影响：

$$\text{Attention complexity} = O(N_v^2 \cdot d)$$

$N_v$ 从 200 增加到 400，attention 计算 4×。但更高分辨率 vision tokens 提供 fine-grained spatial info，对 manipulation 重要。

### 9.3 Diffusion vs Autoregressive 的深层对比

| 维度 | Autoregressive | Diffusion |
|------|----------------|-----------|
| Sampling | 1-pass (with KV cache) | Iterative (50-100 steps) |
| Multimodal | 困难 (mode collapse) | 天然支持 |
| Mode coverage | 单 mode | 多 mode |
| Latency | $O(N)$ | $O(N \cdot T_{\text{diff}})$ |
| Best for | High-frequency control | Multi-solution tasks |

HybridVLA 把两者 combine：autoregressive 做高层规划，diffusion 做 low-level 生成。

### 9.4 FAST Tokenization 的数学

DCT-II 公式：

$$X_k = \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi}{N}\left(n + \frac{1}{2}\right)k\right), \quad k = 0, 1, \ldots, N-1$$

Action trajectory $\mathbf{a}_{1:T}$ 经过 DCT 后，能量集中在低频系数。保留前 $K$ 个系数（$K \ll T$）：

$$\hat{\mathbf{a}}_{1:T} \approx \sum_{k=0}^{K-1} X_k \phi_k$$

其中 $\phi_k$ 是 DCT basis function。$K=16$ 对 1000ms window 已经足够 reconstruct 主要 trajectory shape。

然后 BPE 把频域系数聚合成 discrete tokens，类似 LLM 的 subword tokenization。

### 9.5 LoRA 在 VLA 上的特殊考量

Standard LoRA 假设 weight update 是低秩的，但 VLA 的 action prediction 可能在某些 layer 上需要 high-rank updates。Adaptive LoRA (AdaLoRA) 用 SVD-based importance score 动态调整 rank：

$$\text{importance}_i = \sigma_i(\Delta \mathbf{W})$$

DoRA (Weight-Decomposed LoRA) 把 weight 分解成 magnitude + direction:

$$\mathbf{W} = m \cdot \frac{\mathbf{W}}{\|\mathbf{W}\|}$$

分别 LoRA-adapt magnitude 和 direction，比 vanilla LoRA 在 VLA 上效果更好。

### 9.6 Constrained RL 在 SafeVLA 中的应用

SafeVLA 用 Primal-Dual algorithm 优化 CMDP:

$$\lambda_i^{t+1} = \max\left(0, \lambda_i^t + \eta \left(\mathbb{E}_\pi[c_i] - d_i\right)\right)$$

Policy gradient with Lagrangian:

$$\nabla_\theta \mathcal{L} = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \left(R - \sum_i \lambda_i c_i\right)\right]$$

这种 design 在 deployment 中可以动态 adjust risk tolerance，比固定 threshold 更 adaptive。

---

## 10. 我的几个 critical observations

### 10.1 Tokenization 是 VLA 的 "secret sauce"

VLA 的核心创新其实是 tokenization：把视觉、语言、state、action 全部统一到 token space。这让 Transformer 可以用统一的 self-attention 处理所有 modality，避免了 hand-crafted interface。但 tokenization 也是 bottleneck：

- **Vision token 太多**: 400 tokens 让 inference 慢
- **Action token 精度有限**: 256 bins 不够 surgical precision
- **State token 信息压缩**: 单个 64-dim token 可能丢失重要 state info

Future work 在 token compression 上还有大量空间（VLA-Cache, token merging）。

### 10.2 Dual-System 是当前 VLA 的 sweet spot

纯 end-to-end VLA (RT-2) 在 long-horizon tasks 上 struggle。纯 modular pipeline 失去 end-to-end gradient。Dual-system (GR00T N1, Helix) 是当前最佳折中：

- System 2 (LLM) 处理 reasoning, generalization
- System 1 (diffusion policy) 处理 reactive control, precision

但 temporal synchronization 是 open challenge。8 Hz LLM + 200 Hz controller 的 ratio 是 25×，需要 careful design。

### 10.3 Evaluation 标准需要重构

当前 VLA paper 主要报 task success rate，但这掩盖了：
- **Latency-variance trade-off**: 某些 model 高 success 但 latency 不稳定
- **Safety near-miss**: success 但 dangerously close to failure
- **Recovery ability**: failure 后能否 recover
- **Sample efficiency**: 多少 demos 才能达到 SOTA

VLA-Arena (Zhang et al., 2025) 是好的开始，但需要更多 standardized benchmark。

### 10.4 Cross-Embodiment 是下一个 frontier

当前每个 robot 都要 train 一个 VLA，不可持续。未来需要 embodiment-agnostic policy:

$$\pi_\theta(\mathbf{a} | \mathbf{o}, \mathbf{T}, \text{embodiment spec})$$

其中 embodiment spec 描述 robot 的 morphology（DOF, link lengths, actuator limits）。Meta-learning 在 few-shot 适应新 robot 上有潜力。

### 10.5 VLA + World Model 是 long-horizon 的关键

纯 VLA 缺乏对 environment dynamics 的理解。World model 提供 "what-if" reasoning:

$$\hat{\mathbf{s}}_{t+k} = f_\phi(\mathbf{s}_t, \mathbf{a}_{t:t+k})$$

可以 imagine 不同 action 的 consequence，选择最优。VLA + world model 是 long-horizon planning 的 promising direction。

### 10.6 安全 + 伦理是 deployment 的最后一公里

技术上 VLA 已经接近 deployable，但 safety 和 ethics 是社会层面的 challenge:
- **Audit trail**: 谁对 VLA 决策负责
- **Bias mitigation**: 训练数据中的社会偏见
- **Privacy**: AR VLA 收集的 visual data 归属
- **Workforce**: 机器人取代人类 labor 的 transition

这些不是纯技术问题，需要跨学科 collaboration。

---

## 11. 与相关工作的关联

### 11.1 与 LLM Agent 的关系

VLA 可以看作 LLM Agent 的 embodied 版本。LLM Agent 用 text reasoning + tool use；VLA 用 visual perception + motor action。两者都基于 token prediction，但 VLA 的 grounding 在 physical world。

### 11.2 与 RLHF 的关系

VLA 的 safety alignment 借鉴 RLHF：
$$\mathcal{L}_{\text{RLHF}} = -\mathbb{E}\left[\log \pi_\theta(a|s) \cdot R(s,a)\right]$$

SafeVLA 的 constrained RL 是 multi-objective RLHF 的扩展。

### 11.3 与 Diffusion Models 的关系

Diffusion policy 是 score-based generative model 的应用：

$$\nabla \log p(\mathbf{a}|\mathbf{o}) = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{a}_t, t, \mathbf{o})}{\sqrt{1-\bar{\alpha}_t}}$$

Pi-0, RDT-1B, CogACT 都用 diffusion head，受益于 multimodal action distribution。

### 11.4 与 Neuro-Symbolic AI 的关系

VLA 的 hierarchical design (System 1 + System 2) 是 neuro-symbolic 的具体实现。High-level 用 LLM 做 symbolic reasoning，low-level 用 neural policy 做 continuous control。这和 AlphaGo 的 MCTS + neural net 思路一致。

### 11.5 与 World Models 的关系

World model 是 model-based RL 的核心。VLA + world model 可以做：
- **Model-predictive control**: $\mathbf{a}^* = \arg\max \sum_t r(\hat{\mathbf{s}}_t, \mathbf{a}_t)$
- **Counterfactual reasoning**: "if I had done X, what would happen?"
- **Imagination-based learning**: 在 world model 里 try 新策略

### 11.6 与 Continual Learning 的关系

VLA 的 lifelong learning 是 continual learning 在 robotics 的应用：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{new task}} + \lambda_1 \mathcal{L}_{\text{replay}} + \lambda_2 \mathcal{L}_{\text{regularization}}$$

EWC, SI, LwF 等方法都可以迁移到 VLA。

---

## 12. 总结

这篇 review paper 的核心贡献：

1. **Unified framework**: 把 80+ VLA model organize 在 token-based representation 框架下
2. **三大 architectural paradigm**: Early Fusion, Dual-System, Self-Correcting
3. **六大 application domain**: Humanoid, AV, Industrial, Healthcare, Agriculture, AR
4. **五大 challenge**: Real-time, Multi-modal action, Bias, Integration, Robustness/Ethics
5. **Future roadmap**: Cortex foundation model, agentic learning, neuro-symbolic planning, world model, cross-embodiment, evaluation

VLA 的核心 insight 在于：**把 robot control 变成 sequence modeling**。通过 tokenization 把 vision, language, state, action 统一到一个序列空间，用 Transformer 的 next-token prediction 训练。这个 paradigm shift 让 LLM 的 scale 和 generalization 能力可以 flow 到 robotics。

但 VLA 还在早期阶段。当前 SOTA 在：
- Real-time: 200 Hz (Pi-0 Fast)
- Generalization: 85% on unseen tasks (EF-VLA)
- Long-horizon: dual-system 让 multi-stage task success +17% (GR00T N1)
- Safety: unsafe behavior -80% (SafeVLA)
- Edge deployment: 30-50 Hz on Jetson (Edge VLA, TinyVLA)

未来 5-10 年的关键方向：
- **Cortex foundation model**: web-scale + interaction-scale pretraining
- **Agentic lifelong learning**: self-improve without catastrophic forgetting
- **Hierarchical neuro-symbolic planning**: LLM + symbolic + neural
- **World models**: counterfactual reasoning + imagination
- **Cross-embodiment**: 一个 policy 跨 robot
- **Evaluation beyond success**: safety, recovery, energy, OOD

VLA 的 trajectory 像极了 LLM 的早期：从 narrow task 到 generalist agent，从 research demo 到 production deployment。如果 LLM 用 5 年从 GPT-1 到 GPT-4，VLA 可能用同样时间从 RT-1 到 embodied AGI。

---

## References

1. RT-2: https://robotics-transformer2.github.io/
2. OpenVLA: https://openvla.github.io/
3. GR00T N1: https://arxiv.org/abs/2503.14734
4. Pi-0: https://arxiv.org/abs/2410.24164
5. Octo: https://octo-models.github.io/
6. Open X-Embodiment: https://robotics-transformer-x.github.io/
7. DINOv2: https://arxiv.org/abs/2304.07193
8. LoRA: https://arxiv.org/abs/2106.09685
9. Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
10. FAST: https://arxiv.org/abs/2501.09747
11. Helix (Figure AI): https://www.figure.ai/
12. CogACT: https://arxiv.org/abs/2411.19650
13. SafeVLA: https://arxiv.org/abs/2503.03480
14. ConRFT: https://arxiv.org/abs/2502.05450
15. VoxPoser: https://arxiv.org/abs/2307.05973
16. CLIPort: https://cliport.github.io/
17. VIMA: https://vimalabs.github.io/
18. Gato: https://arxiv.org/abs/2205.06175
19. PaLM-E: https://palm-e.github.io/
20. UniSim: https://universal-simulator.github.io/unisim/
21. Flamingo: https://arxiv.org/abs/2204.14198
22. CLIP: https://arxiv.org/abs/2103.00020
23. RT-1: https://arxiv.org/abs/2212.06817
24. LAION-5B: https://arxiv.org/abs/2210.08414
25. OpenDriveVLA: https://arxiv.org/abs/2503.23463
26. ORION: https://arxiv.org/abs/2503.19755
27. RoboNurse-VLA: https://arxiv.org/abs/2409.19590
28. RDT-1B: https://arxiv.org/abs/2410.07864
29. SC-VLA: https://arxiv.org/abs/2407.08693 (Embodied CoT)
30. EVA-VLA: https://arxiv.org/abs/2410.21230
31. TinyVLA: https://arxiv.org/abs/2501.23718
32. SpatialVLA: https://arxiv.org/abs/2501.15830
33. Pi-0.5: https://arxiv.org/abs/2504.16054
34. SmolVLA: https://arxiv.org/abs/2506.01844
35. VLA Review Repo: https://github.com/Real-World-System-Identification/VLA-Review
