---
source_pdf: DriveWorld-VLA.pdf
paper_sha256: 047ebabbce3437a5722302de5ac813c3e3fb664c276099f92d444f581ab579c8
processed_at: '2026-08-03T23:47:55-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 DriveWorld-VLA

参考: 
- Paper: https://arxiv.org/abs/2506.23955 (推测位置)
- InternVL3: https://arxiv.org/abs/2504.10479
- NAVSIM benchmark: https://github.com/autonomousvision/navsim
- DriveVLA-W0 对比: https://arxiv.org/abs/2510.12796

---

## 一句话讲清这 paper 在干嘛

让一个 VLM 同时学会两件事：**看现在的路** 和 **脑补走了某条路之后路会变成什么样**，然后挑那条脑补出来最舒服的路去开。

---

## 1. 用老司机 vs 新手讲清楚 motivation

新手开车只看眼前，前面车刹车了就跟着踩，反应式驾驶。

老司机开车脑子里一直在演小电影："我现在变道的话，左边那辆白车会不会挤我？我提前 50 米减速的话，后车会不会追尾？" 这些 "what-if" 演完，老司机才动手。这就叫 **counterfactual imagination**。

之前的 end-to-end driving model 大多是新手——给它 image 它输出 trajectory，中间没有 "演未来" 这一步。

DriveWorld-VLA 想造一个老司机。关键是：这个 "演未来" 不能让模型画一张 future RGB image 再分析，那样太贵也太慢。它要在 **latent space 里偷偷演**，模型自己在脑子里把未来 BEV 想一遍，然后根据想得舒不舒服决定怎么开。

---

## 2. 之前方法的毛病，用比喻讲

作者 Figure 1 给了三种 coupling，我用 "学车" 类比：

### (a) Disentangled Interaction —— "请教练在旁边喊"

学员自己开，教练在副驾喊 "前面要撞了"。学员没有内化规则，教练一走就完蛋。代表：ADR1、ReSim，world model 当外部 simulator 调用。

### (b) Feature Sharing —— "共用一个脑子但各想各的"

两个人共享一个大脑做感知，但一个人只管看路（perception），另一个人只管踩油门（planning），俩人不交流 "如果我踩油门会发生什么"。代表：Epona、HERMES、WoTE。

### (c) DriveWorld-VLA —— "一个人同时看路 + 演未来 + 决策"

同一个 latent space 里完成 perceive → imagine → act 三件事，imagination 的结果直接喂回去调 action。这才是真正的 "internalize"。

---

## 3. 三个训练阶段，用学开车类比

这是 paper 最核心的工程 insight，也是 Table 5 ablation 里 7.7 PDMS 的来源。

### Stage 1: 学会 "看路 + 预测"

只让模型学：**给我现在的画面，告诉我下一秒 BEV 会变成什么样，顺便预测 expert 会怎么开**。

这一步模型完全不知道 "action 会改变未来"。它只是死记硬背：场景 A 之后通常变成场景 B。

公式 5 的 loss 就两项：
- $\mathcal{L}_{seg}$: BEV segmentation 监督，逼 latent 能解出语义 map
- $\mathcal{L}_{act}$: 行为克隆，逼 action head 输出 expert trajectory

类比：小孩看爸爸开车，先学会 "前面有路口爸爸通常减速"，纯观察，不懂因果关系。

### Stage 2: 学会 "我踩油门的话会发生什么"

这时模型开始接受一个新输入：**future action**。给它一个 hypothetical action，让它生成对应的 future BEV。

监督信号用了一个巧妙 trick（公式 6-7）：把 GT future frame 喂回 Stage 1 frozen 的 VLM，蒸馏出 "GT future BEV latent" $\mathcal{B}'_{t+\Delta t}$，当成 regression target。Stage 1 的 encoder 当 teacher，DiT 当 student。

公式 8 的 flow-matching loss 本质是学一个 velocity field：
- $x_0 \sim \mathcal{N}(0, I)$: 起点是纯噪声
- $x_k = \frac{k}{N}\mathcal{B}'_{t+\Delta t} + (1-\frac{k}{N})x_0$: 噪声和 target 之间的线性插值
- $k \in [1, N]$: flow-matching 时间步
- $\mathrm{DiT}_\theta$ 学的 target: $\mathcal{B}'_{t+\Delta t} - x_0$, 就是从噪声指向 target 的向量

直觉：模型在 BEV latent 空间里学一条从 "乱" 到 "对" 的轨迹，这条轨迹的方向受 (current BEV, future action) 双重 conditioning 控制。

类比：小孩坐副驾拿方向盘模拟器练手，爸爸说 "如果你现在往左打方向盘，车会偏成这样"，反复练。

### Stage 3: 学会 "先想再做 + 反思"

真正闭环。流程变成：

1. 模型先预测一个 action $\mathcal{A}'_{t+\Delta t}$
2. 用这个 action 当条件，让 DiT rollout 25 步 Euler sampling，生成对应的 future BEV imagination $\mathcal{B}'_{t+\Delta t}$
3. 一个 learned reward model $\mathcal{R}$ 给这个 (action, imagined future) 打分 $\hat{r}_{t+\Delta t}$
4. 拿 $\hat{r}_{t+\Delta t}$ 当权重去加权 BC loss（公式 11）

公式 11 的精妙之处：
$$\mathcal{L}'_{act} = \hat{r}_{t+\Delta t} \cdot \|\mathcal{A}'_{t+\Delta t} - \mathcal{A}_{t+\Delta t}\|^2$$

- $\hat{r}_{t+\Delta t}$: 模型给自己 action 打的分数
- $\mathcal{A}_{t+\Delta t}$: GT expert action
- 高 reward 的样本权重大，低 reward 的样本权重小

这是 **reward-weighted regression** 的变体，类似 ACT、Diffusion Policy 里的 implicit Q-learning 思路。避免 explicit RL 的 variance，又能利用 reward signal 引导。

类比：小孩开始自己开车了，开完一段自己打分 "这次变道挺顺，下次还这么干；刚才刹车太急，下次轻点"。

---

## 4. 推理时模型在干嘛

deployment 时只需要一次 forward：

1. 多 view image + text + history action + BEV → VLM → hidden state $\mathcal{H}_t$
2. action head 直接出 trajectory（不做 imagination 也行，相当于 Stage 1 模式）
3. 想用 Stage 3 的 reflective mode：先出 action，再 25 步 Euler sample 出 future BEV，再过 reward head，可选 top-k trajectory 里挑 reward 最高的

这里有个 **huge question**：25 步 Euler sampling 在 H20 上要多久？paper 没给 latency。如果是 50ms 一帧，real-time 难；如果 5ms 一帧，能用。这是 deployment 的关键 missing piece。我猜他们在 closed-loop NAVSIM 上是 offline 评估，online 跑可能要 distill DiT 到 1-4 步 consistency model。

---

## 5. 结果好在哪

NAVSIMv1: **91.3 PDMS** vs human 94.8。已经接近人类水平。

最亮眼的不是总分，是 **EP (Ego Progress) 85.9** vs DiffusionDrive 82.2、DriveVLA-W0 83.3。EP 高 3 个点意味着车开得更猛、跑得更远，但 NC (No Collision) 还是最高的 99.1。

直觉解读：之前的方法保守不敢开，因为它们不知道 "如果我往前冲会发生什么"，所以宁愿停车。DriveWorld-VLA 因为能脑补 future，知道 "往前冲也不会撞"，所以敢开。这是 proactive 和 reactive 的本质区别。

nuScenes 上 CR (Collision Rate) **0.16%**，比 HERMES-p 的 0.32% 低一半。但 L2 0.61m 比 HERMES-p 0.36m 差。说明 DriveWorld-VLA 路径不一定贴合 expert，但更安全。这也跟 closed-loop 训练目标一致——它优化的是 "不撞"，不是 "像 expert"。

---

## 6. 坑和疑问

1. **Reward model 怎么 generalize？** Stage 3 的 reward 是在 NAVSIM simulator 里跑出来的真 reward 当监督。real world 没有 simulator，reward model 在 OOD 场景会不会瞎打分？这是 deployment 的最大隐患。

2. **$\Delta t$ 到底多大？** Paper 没明说 future BEV 是 0.5s 后还是 2s 后。如果是 0.5s，long-horizon planning 根本没体现；如果是 2s，BEV 预测本身就极难，loss 容易塌。

3. **NAVSIM 是 non-reactive**。其他车不会因为你的 action 改变行为。real world 是 reactive 的，你变道别人会让路或者挤你。DriveWorld-VLA 的 imagination 假设环境是 passive 的，这在 real driving 是大问题。

4. **Latency 没报告**。25 步 Euler sampling 在 LLM forward 上加多少 ms？这是 commercial deployment 的硬指标。

5. **VLM 同时承担太多**。一个 hidden sequence 要编码 perception、language、history、future imagination、reward。scale 到 70B 时候 BEV token 和 image token 的 capacity 抢占会很严重。可能要做 routing 或 MoE。

6. **Progressive training 为什么要这么死板？** Table 5 显示 non-progressive 掉 7.7 PDMS。但这只是工程结论，没有理论解释。是 curriculum 的影响，还是 optimization landscape 的问题？理论上能不能用更好的 initialization 让 joint training 也 work？这是 follow-up 可以挖的。

---

## 7. 我会想到的联想

- 这套 "先 perception-only pretrain → 再加 action-conditioned imagination → 再加 reward-weighted refinement" 三段式，跟 LLM 的 **pretrain → SFT → RLHF** 几乎是同构的。World model 的 rollout 当成了 RLHF 里 RM + PPO 的合成体。这个 paradigm 估计会被复制到 robot manipulation、具身智能等所有 long-horizon decision making 场景。

- Latent space rollout 替代 pixel rollout，跟 Sora、Genie、DreamerV3 的思路一脉相承。区别是 DriveWorld-VLA 把 rollout 直接 plug 进 policy 优化 loop，不是单独当数据增广。这是把 world model 从 "data augmenter" 升级为 "reasoning engine" 的关键一步。

- Reward-weighted regression（公式 11）跟 ACT (Action Chunking Transformer) 的 KL control、Diffusion Policy 的 implicit Q-learning、IQL 都是同源思路。避免 explicit policy gradient 的方差爆炸，用 BC 当 anchor。这套 trick 在 robot learning 已经成熟，driving 这边刚跟进。

- 25 步 Euler sampling 在推理时太慢。后续工作大概率会用 **Consistency Model** 或 **Shortcut Model** 把它蒸馏到 1-4 步，甚至 1 步。这是接下来 6-12 个月必看的方向。Taylor et al. 2024 的 Consistency Trajectory Model、Boyi Li 2024 的 Flow Matching Distillation 都可以直接套。

- 真正的 reactive closed-loop 验证缺位。NAVSIM non-reactive 偏乐观。需要上 CARLA、Wolf-dc、或者 Nvidia Constellation 这种 reactive simulator 才能验证 imagination 的真实价值。我赌 reactive 场景下 DriveWorld-VLA 优势会更大，因为 reactive 场景下 "想未来" 更重要。

- 如果把 reward model 换成真正的 RL signal（PPO / GRPO），把 BC anchor 拿掉，整个 framework 能不能 work？这是从 imitation 到 true RL 的跨越。风险大但天花板高。

---

## 8. 一句话再总结

DriveWorld-VLA 让 VLM 在 BEV latent space 里学会 "做梦"，然后用 "梦到的好坏" 反过来指导 action，最终在 NAVSIM 上把 PDMS 推到 91.3，接近人类水平 94.8。三阶段 progressive training 是工程关键，feature-level BEV 监督是性能来源，reward-weighted refinement 是 closed-loop 优势的来源。

剩下的问题全是 deployment 相关的：latency、reactive 场景、reward model 的 OOD generalization。这些不解决，paper 仍然停在 benchmark SOTA，离上车还差一截。

---

# DriveWorld-VLA 深度解析

Andrej, 这篇 paper 来自地平线 + 复旦 + 华东师大 + 北航联合团队，核心思想是把 VLA 和 World Model 在 latent space 中 "硬耦合" 在一起，让一个 LLM 同时承担 reasoning engine 和 imagination engine 的职责。下面我按 architecture、training paradigm、math、experiments 的顺序拆解，尽量 build up 你的 intuition。

参考链接:
- Paper arXiv (推测): https://arxiv.org/abs/2506.23955 附近
- InternVL3 backbone: https://arxiv.org/abs/2504.10479
- BEVFormer: https://arxiv.org/abs/2203.13970
- NAVSIM: https://arxiv.org/abs/2406.13349
- DiffusionDrive (CVPR 2025): https://arxiv.org/abs/2412.15207
- DriveVLA-W0: https://arxiv.org/abs/2510.12796
- HERMES: https://arxiv.org/abs/2501.14729

---

## 1. Motivation: 为什么之前的 VLA+WM 耦合都不对劲

作者在 Figure 1 里给出了三种 coupling strategy 的对比，这是理解整篇 paper 的钥匙：

- **(a) Disentangled Interaction**: WM 当外部 simulator 用，VLA 单独训练，再 call WM 做 rollout。问题：知识 transfer 不进来，VLA 内部没有 internalize 物理规律。典型代表 ADR1, ReSim。
- **(b) Feature Sharing**: 共享 encoder backbone，但 WM 和 policy head 之间没有 action-conditioned causal link。问题：缺少 counterfactual imagination，模型仍然是 reactive 而非 proactive planning。典型代表 Epona, HERMES, WoTE。
- **(c) DriveWorld-VLA**: 把 WM 的 latent state 当作 "decision variable"，在 shared latent space 里做 action-conditioned controllable imagination。这是从 "post-hoc verification" 到 "in-the-loop reasoning" 的范式转变。

直觉上，作者认为之前的方法都把 world model 看成一个 "预言机"，policy 生成完动作后再问预言机 "这步好不好"。DriveWorld-VLA 反过来：policy 在生成 action 之前先在 latent space 里 "脑补" 多个 what-if future，然后选那个 future 最好的 action。这非常像 AlphaGo 的 MCTS 思想，但是用 latent diffusion 替代了显式树搜索。

---

## 2. 整体架构

输入四类 token:
1. $\mathcal{I}_t$: multi-view images，按 InternVL 方式 tokenize
2. $\mathcal{T}_t$: textual prompts（navigation commands 等）
3. $\mathcal{A}_{t-1}$: historical ego actions，序列化为自然语言
4. $\mathcal{B}_t$: BEV feature map，由 BEVFormer (Li et al., 2024c) 提取，shape 为 $\mathbb{R}^{H\times W\times C}$，spatially flatten 后投影到 VLM embedding space

VLM backbone 用的是 InternVL3，输出最后层 hidden states 作为 shared latent representation:

$$\mathcal{H}_t = \mathrm{VLM}_\theta^\circ(\mathcal{I}_t, \mathcal{B}_t, \mathcal{A}_{t-1}, \mathcal{T}_t) \tag{1}$$

- $\mathcal{H}_t$: 时间步 $t$ 的 shared latent representation，下游同时喂给 imagination branch 和 action branch
- $\circ$ 上标：表示 VLM 在 Stage 1 处于 trainable 状态

关键 insight: 这里的 $\mathcal{H}_t$ 不是单 token，而是 sequence-level hidden states $\in \mathbb{R}^{B\times L\times D}$，其中 $D=1536$ 是 InternVL3 的 hidden dim。作者在 Appendix A.3 中描述了一个 latent pooling 机制：投影到 $d=256$，用 $N_L=700$ 个 learnable latent queries $Z_0$ 做 cross-attention，得到 fixed-length compact representation。这是一个类似 Perceiver / Q-Former 的设计，目的是把 variable-length vision-language sequence 压成 fixed-length，方便下游 diffusion 用。

下游有两个 head：

### 2.1 Imagination branch (DENOISER)

DENOISER 内部分两路：

**History-conditioned branch** (公式 2-3):
$$\mathcal{B}'_t = \mathrm{CrossATTN}_\theta^\diamond(\mathcal{B}_t, \mathcal{H}_t, \mathcal{H}_t) \tag{2a}$$
$$\mathcal{B}_{t+\Delta t} = \mathrm{DENOISER}_\theta^1(\mathcal{H}_t, \mathcal{B}'_t, \mathcal{A}_{t-1}) \tag{2b}$$

- $\mathcal{B}'_t$: 当前 BEV 状态经过 cross-attention 融合 $\mathcal{H}_t$ 后的 "contextualized BEV latent"
- $\mathrm{CrossATTN}^\diamond$: BEV queries 用 $\mathcal{B}_t$，keys/values 用 $\mathcal{H}_t$，把 LLM 的语义信息注入 BEV 几何空间
- $\mathrm{DENOISER}^1$: 第一支 denoiser，仅用 history 做 forward prediction
- $\mathcal{A}_{t-1}$: 历史 action，作为 motion prior

**Future action-conditioned branch** (Stage 2 启用):
$$\mathcal{B}_{t+\Delta t} = \mathrm{DiT}_\theta(\mathcal{B}'_t, \mathcal{A}_{t+\Delta t}, x_k, \tfrac{k}{N}) \tag{8}$$

- $\mathcal{A}_{t+\Delta t}$: GT 未来动作，作为 conditioning signal
- $x_k$: flow-matching 中第 $k$ 步的 noisy latent
- $k$: timestep, uniformly sampled from $[1, N]$
- $\mathrm{DiT}$: Diffusion Transformer，架构见 Figure 3

解码 segmentation head：
$$S_{t+\Delta t} = \mathrm{SEG}_\theta(\mathcal{B}_{t+\Delta t}), \quad S_t = \mathrm{SEG}_\theta(\mathcal{B}'_t) \tag{3}$$

- $S_{t+\Delta t}$: semantic BEV map at future time $t+\Delta t$
- $S_t$: 当前时刻的 BEV segmentation，用于 "图像到 BEV tokenizer" 的训练

这里有个细节，作者强调 "explicit feature-level supervision" 而不是 downstream-task supervision。也就是，他们不直接用 RGB future images 算 reconstruction loss，而是用 GT future BEV latent $\mathcal{B}'_{t+\Delta t}$ 作为 regression target。这避开了 pixel-level rollout 的昂贵代价，同时让 latent space 保持可解释性。

### 2.2 Action branch

$$\mathcal{A}'_{t+\Delta t} = \mathrm{ACT}_\theta(\mathcal{H}_t, \mathcal{B}_t, \mathcal{A}_{t-1}) \tag{4}$$

- $\mathrm{ACT}$: lightweight action decoder (MLP-based)，输出未来 trajectory waypoints
- 注意输入：直接用 $\mathcal{H}_t$ 和 raw $\mathcal{B}_t$，绕过了 imagination branch。这是设计上分离 prediction 和 imagination 的关键。

---

## 3. Three-Stage Progressive Training

这是 paper 最精彩的部分。作者明确指出 (Table 5): 如果同时训 Stage 2 和 Stage 3，会掉 7.7 PDMS。这证明了 progressive 解耦训练的必要性。

### Stage 1: VLA & WM Joint Training

目标：学共享 latent space $\mathcal{H}_t$，让 $\mathcal{H}_t$ 同时能解码出 future BEV semantic map 和 future action。

Loss:
$$\mathcal{L}_{s_1} = \mathcal{L}_{seg} + \mathcal{L}_{act} \tag{5}$$

- $\mathcal{L}_{seg}$: BEV segmentation loss (cross-entropy over semantic classes)
- $\mathcal{L}_{act}$: imitation learning loss on expert trajectory

注意 Stage 1 **不** condition on future action，只用 history。这是 "reactive imagination" 阶段：模型先学会 "基于观察到的场景，未来大概会变成什么样"。这一步让 BEV tokenizer 和 VLM hidden state 对齐，建立 latent foundation。

### Stage 2: Action Controllability Fine-Tuning

目标：赋予 WM "如果 ego 执行某个 action，未来会怎样" 的能力。

关键问题：BEV latent 是抽象的，没有直接的 sensor observation，无法用 pixel reconstruction。作者用了一个聪明的 trick：

**Stage 2 监督信号构造** (公式 6-7):
$$\mathcal{H}_{t+\Delta t} = \mathrm{VLM}_\theta^*(\mathcal{I}_{t+\Delta t}, \mathcal{B}_{t+\Delta t}, \mathcal{A}_{t+\Delta t}, \mathcal{T}_{t+\Delta t}) \tag{6}$$
$$\mathcal{B}'_{t+\Delta t} = \mathrm{CrossATTN}_\theta^*(\mathcal{B}_{t+\Delta t}, \mathcal{H}_{t+\Delta t}, \mathcal{H}_{t+\Delta t}) \tag{7}$$

- $*$ 上标：表示 VLM frozen，仅做 forward，不反传
- $\mathcal{I}_{t+\Delta t}, \mathcal{B}_{t+\Delta t}, \mathcal{A}_{t+\Delta t}$: 都来自 GT future frame

也就是说：把 GT 未来帧塞回 Stage 1 已经训好的 pipeline，得到 GT future BEV latent $\mathcal{B}'_{t+\Delta t}$，作为 Stage 2 的 regression target。这是一个 self-distillation 的味道——Stage 1 的 frozen encoder 当成 teacher，Stage 2 的 DiT 当 student。

**Flow-matching loss** (公式 8):
$$\mathcal{L}_{FM} = \|\mathrm{DiT}_\theta(\mathcal{B}'_t, \mathcal{A}_{t+\Delta t}, x_k, \tfrac{k}{N}) - (\mathcal{B}'_{t+\Delta t} - x_0)\|^2 \tag{8}$$

- $x_0 \sim \mathcal{N}(0, I)$: 初始 noise sample
- $x_k$: 第 $k$ 步插值后的 noisy latent，$x_k = \tfrac{k}{N} \mathcal{B}'_{t+\Delta t} + (1-\tfrac{k}{N}) x_0$（隐式构造）
- $\mathrm{DiT}_\theta$: 学的是 velocity field $\tfrac{d x_t}{d t}$，target 是 $\mathcal{B}'_{t+\Delta t} - x_0$
- $k/N$: normalized timestep embedding

这个 formulation 是 Rectified Flow / Stochastic Flow Matching 的标准形式，比 DDPM 的 noise prediction 更线性、更易采样。直觉上，模型在学习 "从随机起点到目标 BEV future 的向量场"，conditioned on (current BEV, future action)。

$\mathcal{L}_{s_2} = \mathcal{L}_{FM}$，单独训，只更新 DiT 参数。

### Stage 3: Future-Guided Evaluation & Refinement

目标：闭环。给定当前 observation，先预测 action，再用该 action 做条件生成 future imagination，最后用 imagination 的质量反馈 refine action。

**Inference-time rollout** (公式 9):
$$\mathcal{B}_{t+\Delta t}^{k+1} = \mathcal{B}_{t+\Delta t}^k + \tfrac{1}{N} \cdot \mathrm{DiT}_\theta(\mathcal{B}'_t, \mathcal{A}'_{t+\Delta t}, x_k, \tfrac{k}{N}) \tag{9}$$

- $k$: sampling step index, $k \in [0, N-1]$
- $N=25$: 总采样步数
- 初始化 $\mathcal{B}_{t+\Delta t}^0 = x_0 \sim \mathcal{N}(0, I)$
- Euler method: $\mathcal{B}_{t+\Delta t}^{k+1} \leftarrow \mathcal{B}_{t+\Delta t}^k + \Delta t \cdot v(x_t, t)$, 其中 $\Delta t = 1/N$
- 最终 $\mathcal{B}'_{t+\Delta t} = \mathcal{B}_{t+\Delta t}^{N}$

**Reward function** (公式 10):
$$\hat{r}_{t+\Delta t} = \mathcal{R}(\mathcal{B}'_{t+\Delta t}, \mathcal{B}_{t+\Delta t}, \mathcal{A}'_{t+\Delta t}) \tag{10}$$

- $\mathcal{B}'_{t+\Delta t}$: imagination 出来的 future BEV latent
- $\mathcal{B}_{t+\Delta t}$: GT future BEV latent (由 simulator 提供 online evaluation)
- $\mathcal{A}'_{t+\Delta t}$: predicted action
- $\hat{r}_{t+\Delta t} \in \mathbb{R}$: scalar reward，估计 trajectory 质量

**Reward-weighted action loss** (公式 11):
$$\mathcal{L}'_{act} = \hat{r}_{t+\Delta t} \cdot \|\mathcal{A}'_{t+\Delta t} - \mathcal{A}_{t+\Delta t}\|^2 \tag{11}$$

- $\hat{r}_{t+\Delta t}$: predicted reward scalar，作为 weighting
- $\mathcal{A}_{t+\Delta t}$: GT expert action
- 直觉：reward 高的 trajectory，BC loss 权重大；reward 低的 trajectory，loss 权重小
- 这避免了 high-reward 和 low-reward trajectory 之间互相干扰，是 reward-conditioned imitation learning 的一种

总 loss:
$$\mathcal{L}_{s_3} = \mathcal{L}'_{act} + \mathcal{L}_{seg} + \mathcal{L}_{rew} \tag{12}$$

- $\mathcal{L}_{rew}$: reward model 的监督 loss，用 simulator 中执行 trajectory 得到的真实 reward 作为 target
- Stage 3 时 DENOISER 和 VLM 都 frozen，只训 reward model 和 action head

---

## 4. 训练细节

| 项目 | NAVSIM | nuScenes |
|---|---|---|
| Input view | 左前+正前+右前拼接 → 256×1024 | 6 views resized to 640×384 |
| BEV encoder | ResNet-34 | Swin-T (BEV-Planner 风格) |
| Optimizer | AdamW, lr=1e-4, bs=16 | AdamW, lr=7e-5, bs=1 |
| GPU | 8× H20 | 8× H20 |
| Stage epochs | 20 each | 24 each |
| Total time | ~120h | ~93h |
| Ego state | disabled | disabled |

image tokenization 细节：每个 patch 448×448，分配 K=256 个 `<IMG_CONTEXT>` placeholder token。多个 patch 时再加一个 thumbnail patch 做全局 context。这跟 InternVL 的多分辨率策略一致。

---

## 5. 实验结果分析

### 5.1 NAVSIMv1 (Closed-loop)

DriveWorld-VLA: **91.3 PDMS** vs Human **94.8 PDMS**

逐项对比 DriveVLA-W0 (90.2) 和 DiffusionDrive (88.1):
- NC: 99.1 (vs 98.7 / 98.2) — 安全性最高
- DAC: 98.2 (vs 99.1 / 96.2) — drivable area compliance 略低于 DriveVLA-W0
- TTC: 96.1 (vs 95.3 / 94.7) — time-to-collision 最强
- EP: 85.9 (vs 83.3 / 82.2) — ego progress 明显最高

直觉解读：DriveWorld-VLA 在 **EP** 和 **TTC** 上的领先反映了 proactive planning 的优势——它敢于推进更多，但又因为 future imagination 预判了 collision 风险，所以 NC 反而更高。这是 "想得远 + 走得快 + 不撞" 的 Pareto frontier 突破。

### 5.2 NAVSIMv2

DriveWorld-VLA: **86.8 EPDMS** vs DriveVLA-W0 86.1

值得注意的细节指标：
- DDC (Driving Direction Compliance): 99.6 vs 98.0 — 大幅领先，说明 path-following 更稳
- LK (Lane Keeping): 97.0 vs 93.2 — 横向控制明显更平滑
- EC (Extended Comfort): 78.6 vs 58.9 — 显著优势，反映 proactive planning 对 jerk 的预测更准

但 HC (History Comfort) 略低 (97.8 vs 97.9)，这是个微弱短板，可能因为 reward model 更看重 EC 而非 HC。

### 5.3 nuScenes (Open-loop)

L2 avg: **0.61m**, CR avg: **0.16%**

对比 HERMES-p (0.36m / 0.32%): L2 上 DriveWorld-VLA 略差，但 CR 上 DriveWorld-VLA 更优。这说明 DriveWorld-VLA 在 open-loop 上虽然 path deviation 略大，但 collision 更少——这是 long-horizon planning 的体现。

Ablation Table 4 给了一个很有意思的现象：Stage 3 对 nuScenes 的 2nd/3rd second 提升几乎为 0（-0.01, -0.02）。作者解释：
1. open-loop 不依赖 generative supervision
2. Reward model 对 closed-loop 更有效
3. open-loop 训练和 closed-loop 训练的 feedback mechanism 不一样

这暗示了一个更深的 insight: world model 的核心价值在 closed-loop 里体现得更充分，因为 closed-loop 需要 counterfactual reasoning（"如果我做 X，环境会怎样"），而 open-loop 只需要 single-step best match to expert。

### 5.4 关键 Ablation

**Progressive vs Non-progressive** (Table 5): 91.3 vs 83.6，差 7.7 PDMS。这是 paper 最强的论据之一。解释：必须先让模型学会 "从 observation 推 latent"，再学 "用 action condition latent"，最后才学 "用 latent 反过来 refine action"。三步是 hierarchical 的，不能并行。这跟 LLM 的 pretraining→SFT→RLHF 三阶段很像，本质都是 curriculum learning。

**Feature-level vs Task-level supervision** (Table 7): 91.3 vs 87.9。作者通过在 inference 阶段注入 $\mathcal{N}(0, 5)$ 噪声来弱化 feature-level supervision，结果显著掉点。这说明 BEV latent 的精细化监督才是性能来源，仅靠 trajectory 和 segmentation 的 task-level 监督不够。

**VLM freeze strategy** (Table 6): 最佳是 "unfreeze + pretrain"。完全 freeze 限制到 87.6 PDMS。说明 VLM 在 Stage 1 必须更新参数来 align latent space with BEV semantics。

---

## 6. 与 Related Work 的对比直觉

- vs **DriveVLA-W0** (Li et al., 2025b): DriveVLA-W0 用 future image prediction 作 self-supervised signal refine VLA。它是 image-space world model，cost 高、controllability 弱。DriveWorld-VLA 用 BEV latent space，cost 低且能 action-condition。
- vs **HERMES** (Zhou et al., 2025b): HERMES 同时做 3D scene understanding 和 generation。它是 representation-level unification，但缺少 action-conditioned counterfactual rollout。
- vs **Epona** (Zhang et al., 2025): autoregressive diffusion world model，pixel/BEV 生成质量高，但没和 policy 联合优化。DriveWorld-VLA 把 imagination 直接喂回 action refinement。
- vs **WoTE** (Li et al., 2025c): WoTE 用 BEV world model 做 online trajectory evaluation，类似 DriveWorld-VLA Stage 3 的 reward model。区别在于 WoTE 评估是 single-branch，DriveWorld-VLA 是 dual-branch (history + action conditioned) + reward，耦合更紧。
- vs **DiffusionDrive** (Liao et al., 2025): DiffusionDrive 是 truncated diffusion 做轨迹分布采样，不做 future scene imagination。DriveWorld-VLA 同时做 scene evolution 和 trajectory sampling。

---

## 7. 局限性 / 我会问作者的问题

1. **Reward model 的 simulator 依赖**：Stage 3 需要在 simulator 里执行 predicted trajectory 得到 online reward。这本质上是 RL with simulator。在 closed-loop NAVSIM 上 simulator 可得，但真实部署时没有 ground-truth simulator，reward model 的 generalization 风险大。
2. **$\Delta t$ 的尺度**：未来 BEV 用 $\mathcal{B}_{t+\Delta t}$ 监督，但 $\Delta t$ 究竟是 0.5s, 1s, 还是 2s？paper 里没明说，对 long-horizon planning 的实际 horizon 影响很大。
3. **$\mathcal{H}_t$ 作为 shared representation 的 capacity 问题**：一个 latent sequence 同时承载 perception、language、history、future imagination，是否会在 multi-task interference？特别是当 VLA scale 到 70B 这种规模，BEV token 和 LLM token 之间的 dim mismatch 会不会成为 bottleneck？
4. **NAVSIM 是 non-reactive closed-loop**：NAVSIMv1 是 non-reactive，NAVSIMv2 是 pseudo-simulation，nuScenes 是 open-loop。真正的 reactive closed-loop（如 CARLA closed-loop, NvidiaConstellation）上效果如何？这是 deployment-ready 的关键 missing piece。
5. **Flow-matching N=25 采样**：推理时 25 步 Euler sampling，对 latency 影响多大？作者没报告 FPS。在 H20 上单帧 rollout 时间是多少？这决定能否 deploy。
6. **DiT 的 conditioning 方式**：action 是怎么嵌入到 DiT 的？AdaLN? cross-attention? paper Figure 3 只画了大概流程，没说细节。我猜是 timestep embedding + action embedding 拼接后做 FiLM，但这是 implementation 关键。

---

## 8. Intuition 总结

把 DriveWorld-VLA 想成三件事叠在一起：

1. **A VLM that speaks BEV**: 把 BEV feature 当成一种"语言"，让 VLM 在 multi-modal token 序列里同时理解图像、文字、动作、BEV。
2. **A conditional flow-matching generator in BEV latent space**: 用 DiT 学一个 velocity field，从 random noise 到 future BEV latent，conditioned on (current BEV, future action)。这是 "世界模型" 的数学实体。
3. **A reward-weighted imitation learner**: 用 future imagination 的 quality 作 reward，对 BC loss 做 per-sample weighting，让模型偏好 "好未来" 的 action。

这三件事通过 progressive training 解耦：先学 1，再加 2，最后加 3。每一步都 unlock 一个 capability，避免 joint optimization 的塌缩。

直觉上，这是把 AlphaZero 的 "policy network + value network + MCTS" 思想搬到 autonomous driving：DiT rollout ≈ MCTS expansion，reward model ≈ value head，action head ≈ policy head。区别是不做显式树搜索，而是用 diffusion 的 multi-step sampling 当成 implicit tree expansion。这是 latent world model + RL 思路在 driving 领域的一次重要工程验证。

对未来工作的 hint: 如果把 reward model 换成真正的 RL signal（PPO-style），把 DiT 换成更高效的一步蒸馏 generator（Consistency Model / Shortcut Model），整个 framework 的 latency 和 sample efficiency 都会有数量级提升。这是接下来 12-18 个月我会密切关注的方向。
