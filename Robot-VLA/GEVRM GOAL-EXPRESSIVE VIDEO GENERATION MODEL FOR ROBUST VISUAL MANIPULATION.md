---
source_pdf: GEVRM GOAL-EXPRESSIVE VIDEO GENERATION MODEL FOR ROBUST VISUAL MANIPULATION.pdf
paper_sha256: df74289b4da82e43afc65908e147860895539fc9a24fb8fdd63fadf9f85d09e4
processed_at: '2026-08-19T09:37:52-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GEVRM 用人话讲

## 一句话版本

机器人学会先在脑子里"放一段未来应该发生什么的小电影"，然后一边执行一边对比"我现在看到的样子跟我脑子里想的不一样在哪"，再根据这个偏差调整动作。遇到光线变了、画面抖了、被挡了一下，它不会被带偏，因为它一直在跟自己想象的那个 goal 对齐。

---

## 为什么现在 VLA 机器人一出门就废

你训了个 VLA 模型，在 lab 里表现贼好，抓杯子开抽屉都很顺。然后你把它部署到真实世界，发现光线稍微暗一点它就懵了，camera 稍微歪一点它就抓不到东西，画面有点 noise 它就乱动。

原因很简单：这个模型只在"干净数据"上学过，它学到的是"看到这个特定 pixel pattern 就输出这个 action"。pixel pattern 一变，它就不知道自己在哪了。

以前的解法是 data augmentation——训练时把图片随便翻转、旋转、加 noise，让模型见过各种变体。但这有个上限：你能 augment 的 distribution 就那么宽，真实世界的 perturbation 是无限的。你 aug 不到的，它就崩。

GEVRM 的作者从控制论里翻出一个 1982 年的老原理叫 **Internal Model Control (IMC)**，核心思想是：**如果你在控制器内部装一个能模拟系统行为的 model，外部扰动来了，你内部 model 一对比就知道"哎不对，external 肯定有什么东西干扰我了"，然后主动去抵消它。**

这原理在工业控制里用了几十年，在机器人运动控制里也验证过，但没人把它塞到 visual-language manipulation 这种 general task 里。GEVRM 就是来干这个的。

---

## GEVRM 三个部件，各干一件事

### 部件 1：Video Planner —— "想象未来"

给机器人一句话 "把红色方块推到右边"，再加一段当前看到的画面，让它 generate 一段未来会发生的 video。

这段 video 不是给 人 看的，是给机器人自己当 goal 用的——"我接下来应该到达这个 visual state"。

技术上就是一个 text-to-video 的 diffusion model，跟 Sora 同款架构：DiT + spatio-temporal VAE 压缩 + Rectified Flow 训练。但有两个 trick：

**Trick A：Random Mask。** 训练时随机盖住 video 的不同 frame 组合——有时候盖前面让它 predict 后面，有时候盖中间让它补全，有时候盖首尾让它填中间。测试时只用到"前面 known 预测后面"这一种 case，但训练时让模型把所有 case 都练过，它就被迫真的理解"物体怎么运动、时间怎么推进"，而不是死记"开头到结尾的像素对应关系"。

**Trick B：Rectified Flow。** 传统 DDPM 采样要 1000 步，Rectified Flow 让 noise 到 data 的路径走直线，10 步就能采样完。对 robot real-time control 来说这很关键——你不可能等 10 秒才出下一个 action。

### 部件 2：State Alignment —— "对比当下跟想象的差多少"

这是 GEVRM 最 core 的部件，也是 IMC 原理在这里 instantiation 的地方。

你有两个东西：
- **Current state**：机器人现在看到的画面
- **Goal state**：video planner 刚想象出来的未来画面

用两个 ResNet-34 分别 encode 成 512 维向量，L2 normalize 一下。然后用 **prototypical contrastive learning**（SwAV 那套）把它们对齐到同一组 cluster 上。

具体来说：定义 3000 个 prototype 向量，每个 current/goal embedding 算一下它跟哪个 prototype 最像（softmax over cosine similarity）。用 Sinkhorn-Knopp 算法强制 batch 内每个 cluster 被均匀分配（防止 collapse），然后 cross-predict：让 current encoder 去 predict goal 应该属于哪个 cluster，反之亦然。

**为什么这能模拟"系统对扰动的 response"？**

你想啊，没有扰动的时候，current state 的 cluster assignment 跟 goal state 的 cluster assignment 应该比较接近（因为 goal 就是 current 的未来）。一旦有 perturbation——比如光线突然暗了——current state 的 embedding 会偏移，cluster assignment 就变了。但 goal state 是 video planner 在 clean imagination 下生成的，没受扰动影响，assignment 保持稳定。

**这个 assignment 上的偏差，就是 implicit 的扰动 estimate。** 你不需要显式建模"光线变化怎么影响 pixel"，contrastive learning 自动把这个 signal 学进 embedding 里了。

这个 embedding 叫 "internal embedding"，作为 condition 喂给下面的 policy。

### 部件 3：Goal-guided Diffusion Policy —— "根据偏差输出 action"

就是一个 Diffusion Policy（Chi et al. 2023 那篇），condition 不是单纯的 observation，而是 (current internal embedding $z$, goal internal embedding $\hat{z}$) 一起。

训练时 jointly optimize 两个 loss：
- Diffusion policy 的 BC loss：让输出的 action 跟 expert demo 一样
- State alignment 的 contrastive loss：让 encoder 学到能区分 perturbation 的 representation

推理时从 Gaussian noise 开始，20 步 reverse diffusion 采样出一个 7-DoF action（6D end-effector pose + 1D gripper 开关）。

---

## 测试时怎么跑

```
给定一句话 + 初始画面
↓
Video planner 生成 51 个 future goal frames
↓
接下来的 20 步，每步：
  - 用 current encoder encode 当前画面 → z
  - 用 goal encoder encode 对应的 goal frame → ẑ
  - Diffusion policy sample action → 执行
↓
20 步后，重新 generate goal（因为环境变了，老 goal 不准了）
↓
循环直到任务完成
```

每 20 步 refresh 一次 goal，这像 MPC 的 receding horizon 但没那么贵——MPC 每步都要解 optimization，GEVRM 每 20 步才重新 generate 一次 video。

---

## 实验告诉我们什么

### Goal 生成质量（Table 1）

在 BridgeData 和 CALVIN 上，GEVRM 的 FID（生成图与真实图 feature 分布距离）比 baseline 好 3-5 倍，FVD（video 版本的 FID）好 5 倍。这主要归功于 spatio-temporal VAE 压缩 + random mask + Rectified Flow 三件套。

### 标准 CALVIN 泛化（Table 2）

在 A、B、C 三个 environment 训练，D environment 测试（table texture、furniture 位置、color patch 全不同）。GEVRM 单 task 成功率 92%，连续 5 个 task 成功率 26%，跟 SuSIE 持平但 1-4 chain 都更好。

### 扰动 CALVIN（Table 3、4）—— **这是核心结果**

五种 perturbation 下 GEVRM 大幅领先：

| 扰动类型 | GEVRM | GR-1 | SuSIE |
|---|---|---|---|
| Image Shift | 1.00 | 1.00 | 0.96 |
| Image Rotation | 1.16 | 1.07 | 0.72 |
| Color Jitter | 1.64 | 1.35 | 1.44 |
| Occlusion | 2.52 | 2.39 | 2.08 |
| **Noise** | **1.76** | 1.57 | **0.36** |

Noise 这个 case 最 dramatic：SuSIE 直接崩到 0.36（几乎完全失败），GEVRM 还有 1.76。因为 noise 直接破坏了 image editing model 的 input，SuSIE 没法 generate 合理 sub-goal。GEVRM 的 contrastive state alignment 把 noise-induced 偏离 encode 进 internal embedding，policy 学过如何在这种偏离下做 correction。

### Ablation（Figure 5）

去掉 VAE fine-tune 或去掉 state alignment，性能都显著下降。$\lambda$（balance contrastive loss 和 BC loss 的温度参数）在 0.5 到 1.5 之间都差不多，1 最优，说明这套 design 比较鲁棒。

### T-SNE（Figure 6）

加了 state alignment 后，latent representation 的 cluster 边界清晰，temporal 相邻 frame 在 latent space 也相邻（trajectory smooth）。没加的话 cluster 模糊、trajectory 散乱。

---

## 为什么这个思路有意思

### 把控制论老原理"翻译"到 deep learning

IMC 在工业控制用了几十年，但都是 explicit model（比如系统的 transfer function）。GEVRM 用 contrastive learning implicit 地学这个 model——你不用告诉它"光线变化怎么影响 pixel"，它自己从 data 里学到"current state 在 latent space 应该映射到哪个 cluster，扰动来了会偏移到哪"。

这种 cross-disciplinary translation 本身有价值。control theory 几十年积累的 robust design 原理，通过 deep learning 工具 instantiate 出来，可能启发更多类似工作。

### Hierarchy 让 data 规模解耦

- High-level video planner：在 internet-scale 的 text-video pairs 上预训练，不需要 robot action label
- Low-level policy：只需要少量 play data（400 条 trajectory 就能 real-world 部署），不需要 language label

这比 OpenVLA、RT-2 那种 end-to-end 需要海量 (image, instruction, action) triples 的方案 sample efficient 得多。

### Video generation 作为 world model

Sora 证明了 video generation model 可以是 world simulator。GEVRM 把这个思路用到 robot：用 video model 想象未来，然后执行。这比 Dreamer 那种 latent dynamics rollout 简单——你不用学复杂的 latent transition，直接 generate 未来 frame 就行。

### Video planner 的 random mask 是 key

这个 trick 看起来简单但效果大。它让 video model 学到 object permanence（物体被挡住后还能想象它还在）和 temporal causality（任意时刻都能 predict 任意未来）。这在 occlusion perturbation 下特别有用——GEVRM 在 occlusion case 上 Avg Length 2.52 是所有方法里最高的。

---

## 几个 Failure Mode 推测

1. **极端扰动**：完全遮挡目标物体很久，video planner 想象不出合理 goal
2. **OOD object**：train 时没见过的物体形状，video generation 会 hallucinate
3. **Long-horizon**：超过 5 个 chained task，goal generation 累积误差
4. **Real-time 要求高**：如果需要几百 Hz control，50-step sampling 不够（不过 Table 7 显示 open-loop 4 步能到 50Hz，大多数 manipulation 够了）
5. **Language ambiguity**：instruction 不够 specific 时，video generation 可能 sample 到错误 goal

---

## 可能的 Extension

1. 加 force/torque sensing 进 internal model，让 embedding 不只视觉
2. 检测到大扰动时立刻 refresh goal，不等 20 步
3. Hierarchical prototype（3000 不够细可以用 hierarchical k-means）
4. 生成 goal 后用 VLM 验证是否符合 instruction
5. 3D video generation（NeRF-style world model）
6. Active perception——robot 主动调整 camera 视角减少扰动
7. Online prototype update 遇到新 task 时 continual learning

---

## 我的 takeaway

GEVRM 给我最深的 intuition 是：**robustness 不一定要靠"见过更多 perturbation"的 data augmentation，可以靠"内部有一个能对比想象的 reference"的 closed-loop 结构。**

这跟人怎么应对扰动很像。你伸手抓杯子，灯突然暗了，你不会傻住——你脑子里有个"杯子应该在哪、手应该往哪移"的预期，灯暗了你的 visual feedback 跟预期对不上，你靠这个偏差继续调整。GEVRM 就是把这套"预期 + 偏差 + 纠正"的机制用 video generation + contrastive learning + diffusion policy 实现了。

这种"control theory 原理 + deep learning 工具"的组合可能还会继续出活儿：Lyapunov stability 注入 policy training、LQR/MPC 用 diffusion 实现、adaptive control 用 online prototype update 之类的。值得关注的 direction。

---

## 关键 References

- **GEVRM 本身**：本 paper
- **IMC 原理**：Garcia & Morari 1982, https://www.semanticscholar.org/paper/Internal-model-control-a-unifying-review-and-new-Garcia-Morari/
- **UniPi**：Du et al. 2024, https://arxiv.org/abs/2310.07117
- **SuSIE**：Black et al. 2023, https://arxiv.org/abs/2310.10639
- **GR-1**：Wu et al. 2023, https://arxiv.org/abs/2312.13139
- **Diffusion Policy**：Chi et al. 2023, https://arxiv.org/abs/2303.04137
- **SwAV**：Caron et al. 2020, https://arxiv.org/abs/2006.09882
- **Sinkhorn**：Cuturi 2013, https://arxiv.org/abs/1306.2597
- **Rectified Flow**：Liu et al. 2022, https://arxiv.org/abs/2209.03003
- **Stable Diffusion 3**：Esser et al. 2024, https://arxiv.org/abs/2403.03206
- **Open-Sora**：Zheng et al. 2024, https://github.com/hpcaitech/Open-Sora
- **CALVIN**：Mees et al. 2022, https://arxiv.org/abs/2112.03227
- **Sora**：Brooks et al. 2024, https://openai.com/research/video-generation-models-as-world-simulators
- **Dreamer**：Hafner et al. 2019, https://arxiv.org/abs/1912.01603
- **OpenVLA**：Kim et al. 2024, https://arxiv.org/abs/2406.09246
- **HiP**：Ajay et al. 2024, https://arxiv.org/abs/2310.10639
- **T5**：Raffel et al. 2020, https://arxiv.org/abs/1910.10683

---

# GEVRM: 把 Internal Model Control 注入 VLA 框架的 Robust Visual Manipulation

## 0. 一句话直觉

这篇论文的核心 trick 是：把 1982 年控制论里经典的 **Internal Model Control (IMC)** 原理，通过 video generation model + prototypical contrastive learning 这两个 modern deep learning 工具，"翻译"进 VLA (Vision-Language-Action) 框架。高层用 text-guided video diffusion 生成 expressive goal frames 作为 reference input；中层用 SwAV-style 的 cluster assignment 把 current state 和 goal state 对齐到同一个 latent prototype space，模拟系统 response；低层用一个 goal-conditioned diffusion policy 输出 7-DoF action。整套闭环在 perturbation 到来时，靠 internal embedding 的偏差隐式 infer 扰动并抵消。

---

## 1. Motivation：为什么 VLA 在真实部署会崩

当前 VLA 模型（RT-1, RT-2, OpenVLA, GR-1, RoboFlamingo 等）在 lab environment 表现很好，但 deployment 时遇到：

- **Lighting fluctuation**：自然光变化、阴影漂移
- **Video stream noise**：信号传输引入的 pixel noise
- **Camera shift/rotation**：mount 松动、震动
- **Occlusion**：人手遮挡、物体临时被挡
- **Color jitter**：白平衡漂移

这些 perturbation 给 VLA 喂入了 "unforeseen state information"，policy 在错误 state estimate 下输出 fragile action。传统解法是 **image augmentation**（flip/rotate/color jitter），但 augmentation 只能覆盖 narrow distribution，无法应对 deployment-time 的 unknown perturbation。

作者从控制论借来 IMC：**closed-loop 系统里如果有一个 internal model 能模拟 external input + reference input，就能精确 track reference 并抵消 disturbance**。这个思想在机器人运动控制里已被验证（Emken & Reinkensmeyer 2005），但还没人把它 instantiation 到 visual-language manipulation 这种 general task 里。GEVRM 就是来填这个坑。

参考：
- IMC 经典 paper: https://www.semanticscholar.org/paper/Internal-model-control-a-unifying-review-and-new-Garcia-Morari/
- Kawato neural internal model: https://pubmed.ncbi.nlm.nih.gov/10607637/

---

## 2. 整体架构：Figure 2 拆解

GEVRM 的 dataflow 是 hierarchical 的，分三层：

```
Language g ──→ T5 encoder (frozen) ──→ text embedding
                                           │
Image seq τ_{0:t} ──→ 2D VAE (8×8 spatial) ──→ 3D VAE (4× temporal) ──→ latent
                                           │
                                           ↓
                                    DiT-XL/2 + Random Mask
                                           │
                                           ↓ (Rectified Flow sampling, 50 steps)
                                    Goal frames τ_{t:T}
                                           │
                ┌──────────────────────────┴──────────────────────────┐
                ↓                                                     ↓
        Current state x_t                                    Goal state x_goal
                │                                                     │
        ResNet-34 (ψ)                                     ResNet-34 (ψ')
                │                                                     │
            L2 norm → z                                          L2 norm → ẑ
                │                                                     │
                └──────────────► Prototypes E = {e_n}, n=1..N ───────┘
                                           │
                                  SwAP-style cross assignment
                                           │
                                           ↓
                                Internal embedding (z, ẑ)
                                           │
                                           ↓
                            Goal-guided Diffusion Policy π_φ(a | z, ẑ)
                                           │
                                           ↓
                                  7-DoF action a_t
```

关键直觉：video planner 输出的不是"应该做什么 action"，而是"应该到达什么 visual state"。这个 decoupling 让 video planner 可以在 internet-scale 的 text-video pairs 上预训练，而 policy 只需要少量 play data（无需 language label）。

---

## 3. Problem Formulation：Eq. (1) 的 Hierarchical 分解

$$
p_{\Theta}(a_t, \tau_{t:T} \mid g, \tau_{0:t}) = p_{\phi}(\tau_{t:T} \mid g, \tau_{0:t}) \cdot p_{\varphi}(a_t \mid \tau_{0:T})
$$

变量含义：
- $\Theta = \{\phi, \varphi, \psi, \psi'\}$: 全部参数集合
- $\phi$: video generation model (DiT) 参数
- $\varphi$: goal-guided diffusion policy 参数
- $\tau_{0:t}$: 历史 image observation 序列（已观测）
- $\tau_{t:T}$: 未来 image goal frames（要 generate）
- $g$: language instruction
- $a_t$: 当前 step 要执行的 7D action

这个 factorization 是 **UniPi (Du et al. 2024)** 提出的思路，但 GEVRM 在两边都做了升级：
- 左侧 $p_\phi$：从 image-level 升级到 video-level，从 DDPM 升级到 Rectified Flow
- 右侧 $p_\varphi$：从 inverse dynamics model 升级到 goal-conditioned diffusion policy，并加入 state alignment

训练数据需求因此变成两份独立 dataset：
- $\mathcal{D}_{\tau, g} = \{(\tau^i, g^i)\}_{i=0}^I$：text-video pairs，可从 internet 抓
- $\mathcal{D}_{\tau, a} = \{(\tau^j, a^j)\}_{j=0}^J$：play data，少量机器人 teleoperation 数据，**不需要 language label**

参考 UniPi: https://arxiv.org/abs/2310.07117

---

## 4. Component 1: Robot Behavior Planner

### 4.1 Video Spatio-temporal Compression

直接在 pixel space 跑 DiT 计算量爆炸。作者用两级 VAE 级联压缩：

**Stage 1 (2D VAE)**：在单帧上做 $8 \times 8$ spatial downsample。来自 Stable Diffusion 的 latent diffusion 思路（Rombach et al. 2022）。
$$
\text{256×256×3} \xrightarrow{\text{2D VAE}} \text{32×32×4}
$$

**Stage 2 (3D VAE)**：在 time 维度做 $4\times$ downsample。来自 Open-Sora 的 MAGVIT-v2 思路（Yu et al. 2023）。关键是用 **Causal 3D Conv**，保证每帧的 output 只依赖 antecedent frames，避免 future leakage。

$$
\text{T×32×32×4} \xrightarrow{\text{3D VAE}} \text{T/4×32×32×4}
$$

总压缩比：$8 \times 8 \times 4 = 256\times$。一个 51-frame、256×256 的 video，从 $51 \times 256 \times 256 \times 3 \approx 10M$ tokens 压到 $\approx 40K$ tokens，DiT 可以跑得动。

直觉：2D VAE 先压 spatial redundancy（自然图像 spatial correlation 强），3D VAE 再压 temporal redundancy（相邻帧 90% 像素相似）。这个顺序很重要——如果先用 3D VAE 在 pixel space 算，卷积核要扫过 H×W×T，FLOPs 极高；先 2D 压完后再 3D，每个 3D conv 的输入维度小很多。

参考：
- Latent Diffusion: https://arxiv.org/abs/2112.10752
- Open-Sora: https://github.com/hpcaitech/Open-Sora
- MAGVIT-v2: https://arxiv.org/abs/2310.05737

### 4.2 Random Mask Mechanism

这是从 UL2 (Tay et al. 2022) 借来的 mixture-of-denoisers 思想，移植到 video generation。训练时随机 unmask 不同 frame 组合：

| Mask 策略 | 含义 | 权重 |
|---|---|---|
| `quarter_head` | unmask 前 $h$ 帧（前 $h$ 已知，predict 后续） | **0.75** |
| `quarter_tail` | unmask 后 $h$ 帧 | 0.025 |
| `quarter_head_tail` | unmask 前后 $h$ 帧，中间被 mask | 0.025 |
| `image_head` | 只 unmask 第一帧 | 0.025 |
| `image_tail` | 只 unmask 最后一帧 | 0.025 |
| `image_random` | 随机 unmask 一帧 | 0.025 |
| `image_head_tail` | unmask 首尾两帧 | 0.05 |
| `interpolate` | unmask 间隔均匀的几帧 | 0.025 |
| `quarter_random` | 随机 unmask 一个 quarter | 0.025 |
| `image_random` | 随机 unmask 一帧 | 0.0 |

**测试时**只有"前 $h$ 帧 known，预测后续"这一种情形，所以 `quarter_head` 占 75% 权重。其余 25% 是 auxiliary task，作用是 forcing 模型理解"任意时刻 snapshot → 任意未来"的映射，而不只是死记"开头 → 结尾"。

直觉：这像 data augmentation 的一种"task augmentation"——同一段 video 被当作多种 different denoising subtask 来学，模型被迫学到 object dynamics 和 temporal causality 的 abstract representation，而不是简单 memorize frame-to-frame correlation。论文 Figure 3 显示，在 perturbed environment 下，baselines (AVDC, GR-1, SuSIE) 会 hallucinate 出扭曲的物体甚至完全破坏场景，GEVRM 仍能保持 3D consistency。

参考 UL2: https://arxiv.org/abs/2205.05131

### 4.3 Model Backbone & Rectified Flow

DiT 用的是 **STDiT3-XL/2**（Open-Sora 系列的 spatio-temporal transformer），从预训练 text-to-video model 初始化（Zheng et al. 2024 Open-Sora）。Language encoder 用 **frozen T5**（Raffel et al. 2020），max length 300 tokens。

训练用 **Rectified Flow**（Liu et al. 2022；Esser et al. 2024 Stable Diffusion 3 也用这个），替代传统 DDPM。

Rectified Flow 的核心：学一个 ODE $\dot{x}_t = v_\theta(x_t, t)$，让 trajectory 从 noise $x_0 \sim \mathcal{N}(0, I)$ 到 data $x_1$ 走**直线路径**：
$$
x_t = (1-t) \cdot \text{noise} + t \cdot \text{data}
$$
$$
\mathcal{L}_{RF} = \mathbb{E}_{t \sim U(0,1), \epsilon} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]
$$

直线 ODE 的好处：可以少步采样而不损失质量。Table 6 的 ablation 显示：

| Sampling steps | Infer. time [s] | Avg. Length |
|---|---|---|
| 50 | 0.598 | 1.76 |
| 40 | 0.501 | 1.67 |
| 30 | 0.379 | 1.63 |
| 20 | 0.260 | 1.60 |
| 10 | 0.135 | 1.67 |

steps 从 50 减到 10，inference time 减少 4.4×，但 Avg Length 几乎不变。这对 robot 的 real-time control 至关重要。

参考：
- Rectified Flow: https://arxiv.org/abs/2209.03003
- Stable Diffusion 3: https://arxiv.org/abs/2403.03206
- T5: https://arxiv.org/abs/1910.10683

---

## 5. Component 2: State Alignment via Prototypical Contrastive Learning

这是 IMC 原理在 latent space 的 instantiation，也是这篇论文最核心的"灵魂"。

### 5.1 编码 current 和 goal state

用两个 ResNet-34（参数分别为 $\psi, \psi'$）分别 encode：
- $f_\psi(x)$: current visual state
- $f_{\psi'}(x_{goal})$: goal visual state

然后 L2 normalize 到 unit sphere：
$$
z = \frac{f_\psi(x)}{\|f_\psi(x)\|_2}, \quad \hat{z} = \frac{f_{\psi'}(x_{goal})}{\|f_{\psi'}(x_{goal})\|_2}
$$
$z \in \mathbb{R}^{512}$, $\hat{z} \in \mathbb{R}^{512}$（ResNet-34 最后一层 feature dim）。

### 5.2 Prototypes 和 Cluster Assignment

定义 $N=3000$ 个 trainable prototypes $\mathbf{E} = \{e_n\}_{n=1}^N$，每个 $e_n \in \mathbb{R}^{512}$，L2 normalized。

对 source (current) 和 target (goal) 各算 cluster assignment probability：
$$
p_n^{\text{source}} = \frac{\exp\left(\frac{1}{\delta} z \cdot e_n\right)}{\sum_{n'} \exp\left(\frac{1}{\delta} z \cdot e_{n'}\right)}, \quad p_n^{\text{target}} = \frac{\exp\left(\frac{1}{\delta} \hat{z} \cdot e_n\right)}{\sum_{n'} \exp\left(\frac{1}{\delta} \hat{z} \cdot e_{n'}\right)}
$$

变量：
- $\delta = 0.1$: temperature parameter，控制 softmax 的 sharpness。$\delta$ 小 → distribution 更 peaky
- $z \cdot e_n$: source vector 和第 $n$ 个 prototype 的 cosine similarity（因为都 L2 normalized）
- $p_n^{\text{source}}$: 当前 state 被分配到 cluster $n$ 的概率

### 5.3 Sinkhorn-Knopp 避免 Collapse

如果直接用 $p^{\text{source}}$ 当 target 训练自己，模型会 collapse——所有样本都分给某个 prototype。SwAV (Caron et al. 2020) 的解法是用 **Sinkhorn-Knopp algorithm** 算一个 balanced assignment $q$ 作为 supervision target：

$$
q^{\text{source}}, q^{\text{target}} = \text{Sinkhorn-Knopp}(Z^{\text{source}}, Z^{\text{target}})
$$

Sinkhorn-Knopp 本质是 optimal transport 的一种特例：在 batch 内所有 sample 上，强制每个 cluster 的总分配质量等于 $1/N$。这等价于一个 entropic-regularized OT 问题：
$$
\min_Q \langle Q, -\log P \rangle + \epsilon \sum_{i,n} Q_{i,n} \log Q_{i,n}
$$
$$
\text{s.t.} \quad Q \mathbf{1} = 1/B, \quad Q^\top \mathbf{1} = 1/N
$$
其中 $B$ 是 batch size, $N$ 是 prototype 数。

### 5.4 SwAV-style Cross-assignment Loss

核心 trick：**用 source 的 target 监督 target 的 prediction，反过来也一样**：
$$
\mathcal{I}_\psi = -\mathbb{E}_{x, x_{goal} \sim \mathcal{D}_{a,x}} \left( q^{\text{source}} \ln p^{\text{target}} + q^{\text{target}} \ln p^{\text{source}} \right)
$$

变量：
- $q^{\text{source}}$: 由 Sinkhorn 算出的 source 应该被分配到的 cluster 分布（stop-gradient）
- $p^{\text{target}}$: target encoder 实际预测的 cluster 分布
- 这个 cross-entropy 让 target encoder 去预测 source 的 cluster assignment，反之亦然

直觉：positive pair $(x, x_{goal})$ 来自同一 trajectory，应该被分配到相近的 cluster。通过 swap prediction，模型被迫让 current encoder 和 goal encoder 学到一致的 semantic clustering。当 perturbation 到来时，$z$（current）的 cluster assignment 会偏离，但 $\hat{z}$（goal）保持不变——这个偏离就是 implicit perturbation estimate，作为 condition 喂给 policy。

参考：
- SwAV: https://arxiv.org/abs/2006.09882
- Sinkhorn distances: https://arxiv.org/abs/1306.2597
- DreamerPro (prototype in RL): https://arxiv.org/abs/2110.14523

---

## 6. Component 3: Goal-guided Diffusion Policy

### 6.1 Action Space

7-DoF：
- $a_{EE} \in \mathbb{R}^6$: end-effector 6D pose（3 translation + 3 rotation，用 axis-angle 或 quaternion 截断）
- $a_{gripper} \in \{-1, 1\}$: gripper binary state（开/关）

输出 $a = [a_{EE}, a_{gripper}] \in \mathbb{R}^7$。

注意：作者**只用 third-view static camera**，不用 robot proprioception 也不用 gripper view，故意让任务更 challenging 以凸显 robustness。

### 6.2 DDPM Forward Process

定义 Markov noise chain $\{a_k\}_{k=0}^K$，$K=20$ steps：
$$
q(a_k \mid a_{k-1}) = \mathcal{N}\left(\sqrt{1 - \beta_k} a_{k-1}, \beta_k I\right)
$$
变量：
- $\beta_k$: 第 $k$ 步的 noise variance（cosine schedule）
- $a_0$: ground-truth action
- $a_k$: 第 $k$ 步加噪后的 action

累积：
$$
\bar{\alpha}_k = \prod_{i=1}^k (1 - \beta_i), \quad q(a_k \mid a_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_k} a_0, (1 - \bar{\alpha}_k) I\right)
$$

### 6.3 Training Objective (Eq. 5)

简化版的 score matching：
$$
\mathcal{I}_\varphi = \mathbb{E}_{k \sim U(1,K), \epsilon \sim \mathcal{N}(0, I), x, x_{goal}, a \sim \mathcal{D}_{x,a}} \left[ \left\| \epsilon - \pi_\varphi\left(\sqrt{\hat{\alpha}_k} a + \sqrt{1 - \hat{\alpha}_k} \epsilon, z, \hat{z}, k\right) \right\|_2 \right]
$$

变量：
- $k$: 随机采样的 diffusion step
- $\epsilon$: 标准 Gaussian noise
- $\hat{\alpha}_k$: 累积 noise schedule
- $z, \hat{z}$: state alignment 输出的 internal embedding
- $\pi_\varphi$: score network (一个 MLP with FiLM conditioning on $z, \hat{z}, k$)

网络输入：noised action $a_k$ + condition $(z, \hat{z}, k)$；输出：预测 noise $\epsilon$。

### 6.4 Reverse Sampling (Eq. 7)

$$
a_{k-1} = \frac{1}{\sqrt{\alpha_k}} \left( a_k - \frac{\beta_k}{\sqrt{1 - \bar{\alpha}_k}} \pi_\varphi(a_k \mid z, \hat{z}, k) \right) + \sqrt{\beta_k} \epsilon
$$

从 $a_K \sim \mathcal{N}(0, I)$ 开始，迭代 $K$ 步得到 $a_0$。

### 6.5 Joint Objective (Eq. 6)

$$
\mathcal{I} = \mathcal{I}_\varphi + \lambda \mathcal{I}_\psi
$$

$\lambda = 1$ (最优, 见 Figure 5b)。两项 jointly optimize：

- $\mathcal{I}_\varphi$ 让 policy 输出正确的 action（behavior cloning）
- $\mathcal{I}_\psi$ 让 state encoder 学到能区分 perturbation 的 representation

这两项的 synergy：policy 训练时，每个 iteration 都更新 state encoder，encoder 学到的 "internal embedding" 会越来越好地反映"current state 偏离 goal 多少"，policy 因此能 condition on 这个偏离做出 correction action。

参考 Diffusion Policy: https://arxiv.org/abs/2303.04137

---

## 7. Test-time Pipeline (Algorithm 1)

```
输入: x_{0,test}, g_test, time limit T, goal refresh interval L_test=20, 
      goal generation number M=51
输出: action sequence

t ← 0
while t ≤ T:
    1. 采样 M 个 future goals: {x_{m,goal}}_{m=0}^{M} ~ P_φ(· | x_{t,test}, g_test)
    2. for l = 1 to L_test:
        a. z_t ← L2norm(f_ψ(x_{t,test}))
        b. ẑ_l ← L2norm(f_ψ'(x_{l,goal}))
        c. a_t ~ π_φ(· | z_t, ẑ_l)   # 20 steps DDPM sampling
        d. x_{t+1,test} ← Env.Step(a_t)
        e. t ← t + 1
    # 每 L_test 步重新 sample goal，因为环境变了
```

关键设计：**goal 不是一次性 generate 到底，而是每 20 步 refresh**。这模仿了 closed-loop control 的 receding horizon 思想——MPC 也是每步 re-plan。但 GEVRM 不是每步重 plan（计算太贵），而是每 20 步重 plan，中间 20 步用同一个 goal 但 current state 实时更新。

控制频率：Table 7 显示 open-loop control steps = 1 时 infer time 0.077s（13Hz），steps = 4 时 0.020s（50Hz）。对大多数 manipulation 够用了。

---

## 8. 实验数据深度解析

### 8.1 Goal Generation Quality (Table 1)

| Benchmark | Method | FID ↓ | FVD ↓ | LPIPS ↓ | SSIM ↑ | PSNR ↑ |
|---|---|---|---|---|---|---|
| BridgeData | AVDC | 246.45 | 22.89 | 0.23 | 0.73 | 18.22 |
| BridgeData | SuSIE | 114.79 | - | 0.22 | 0.71 | 16.39 |
| BridgeData | **GEVRM** | **35.70** | **4.16** | **0.06** | **0.89** | **22.36** |
| CALVIN | GR-1 | 236.75 | 12.83 | 0.20 | 0.65 | 18.59 |
| CALVIN | SuSIE | 214.14 | - | 0.15 | 0.75 | 18.12 |
| CALVIN | **GEVRM** | **94.47** | **3.80** | **0.09** | **0.80** | **21.10** |

- **FID (Fréchet Inception Distance)**: 用 Inception-V3 提 feature，算生成图与真实图 feature 分布的 Fréchet distance。越小越好。
- **FVD (Fréchet Video Distance)**: 同理但用 video feature（通常用 Inflated 3D I3D）。
- **LPIPS**: 用 deep feature 算 perceptual distance，越小越相似。
- **SSIM**: 结构相似度，越大越好。
- **PSNR**: peak signal-to-noise ratio，越大越好。

GEVRM 在 BridgeData 上 FID 比 SuSIE 好 3.2×，FVD 比 AVDC 好 5.5×。这个 gap 主要来自 video spatio-temporal compression + random mask + Rectified Flow 三件套。video compression 让模型在低维 latent space 学到更 abstract 的 dynamics；random mask 让模型对任意 frame 的 conditional generation 都 robust；Rectified Flow 让训练更 stable。

### 8.2 Standard CALVIN Generalization (Table 2)

Train on A, B, C → Test on D（不同 table texture, furniture 位置, color patch）：

| Method | 1-chain | 2-chain | 3-chain | 4-chain | 5-chain |
|---|---|---|---|---|---|
| HiP | 0.08 | 0.04 | 0.00 | 0.00 | 0.00 |
| UniPi | 0.56 | 0.16 | 0.08 | 0.08 | 0.04 |
| GR-1* | 0.75 | 0.45 | 0.20 | 0.15 | 0.10 |
| SuSIE | 0.87 | 0.69 | 0.49 | 0.38 | 0.26 |
| **GEVRM** | **0.92** | **0.70** | **0.54** | **0.41** | 0.26 |

5-chain (连续完成 5 个 chained instruction) GEVRM 与 SuSIE 持平，但 1-4 chain GEVRM 都更好。1-chain 92% 比 SuSIE 87% 高 5 个点，这显示 video-level planning 比 image-level sub-goal planning 在 single task 上更准。

### 8.3 Perturbed CALVIN (Table 3 & 4) - 核心结果

5 种 perturbation 的 average：

| Method | 1 | 2 | 3 | 4 | 5 | Avg. Length |
|---|---|---|---|---|---|---|
| SuSIE | 0.56 | 0.26 | 0.13 | 0.10 | 0.06 | 1.11 |
| RoboFlamingo | 0.63 | 0.35 | 0.18 | 0.09 | 0.05 | 1.31 |
| GR-1 | 0.67 | 0.38 | 0.22 | 0.11 | 0.06 | 1.44 |
| **GEVRM** | **0.70** | **0.47** | **0.26** | 0.11 | 0.07 | **1.62** |

GEVRM 的 Avg Length 1.62 vs GR-1 的 1.44，提升 12.5%。这个 gap 在 5 种 perturbation 分别看：

| Perturbation | GEVRM | GR-1 | SuSIE |
|---|---|---|---|
| Image Shift | 1.00 | 1.00 | 0.96 |
| Image Rotation | 1.16 | 1.07 | 0.72 |
| Color Jitter | 1.64 | 1.35 | 1.44 |
| Image Occlusions | 2.52 | 2.39 | 2.08 |
| Noise Interference | 1.76 | 1.57 | 0.36 |

关键观察：
- **Noise Interference** 是 GEVRM 优势最大的场景（1.76 vs SuSIE 0.36，提升 4.9×）。SuSIE 几乎完全崩溃，因为 noise 破坏了 image editing model 的 input。GEVRM 的 state alignment 把 noise-induced 偏离 encode 进 internal embedding，policy 学过如何"修正"这种偏离。
- **Image Occlusions** GEVRM 2.52 是所有方法里最高的。video generation model 在 random mask training 下学到 object permanence，即使部分被遮挡也能想象完整 goal。
- **Image Rotation** SuSIE 0.72 严重低于 GEVRM 1.16。SuSIE 的 augmentation 大概没有 rotation augmentation，或者有但不够。

### 8.4 Ablation (Figure 5)

(a) Training paradigm ablation：
- Full GEVRM: 最优
- w/o VAE fine-tune: 显著下降。VAE 在 internet video 上 pretrain 拿到 general spatio-temporal prior，fine-tune 到 robot data 后这个 prior 帮助 generalize。
- w/o state alignment: 也显著下降。说明 IMC 的 internal model 这部分确实贡献了 robustness。

(b) $\lambda$ sweep：$\lambda \in \{0.5, 0.8, 1, 1.2, 1.5\}$ 都差不多，$\lambda=1$ 最优。说明 contrastive loss 和 BC loss 的 scale 比较均衡。

### 8.5 T-SNE 可视化 (Figure 6)

对比有/无 state alignment 的 latent representation：
- **有 SA**：clear cluster centers，distinct class boundaries，temporal 相邻 frame 的 representation 在 latent space 也相邻（smooth trajectory）。
- **无 SA**：cluster 边界模糊，相邻 frame 的 representation 散乱。

直觉：SA 让 representation 同时具有 (1) semantic discriminability（不同 task 不同 cluster）和 (2) temporal smoothness（同 trajectory 内的 frame 接近）。这两点对 robust policy 都关键：前者让 policy 知道"现在在哪个 task 阶段"，后者让 policy 在小扰动下不会被甩到完全不同的 cluster。

### 8.6 Real-world (Figure 7)

UR5 robot，3 个 task：
- Pick red cup: SR = 0.8
- Put smaller blue bowl into red bowl: SR = 0.8
- Take tiger out of red bowl: SR = 0.6（soft deformable object 更难）

400 条 teleoperation trajectory，20-120 steps each，5Hz control。这个数据量比 RT-2 (130k episodes) 小 300×，但 GEVRM 通过 internet video pretrain 弥补。

---

## 9. 与相关工作的 deeper 联系

### 9.1 与 UniPi / SuSIE / GR-1 的 evolution

- **UniPi** (Du et al. 2024, NeurIPS 2023): 第一个把 decision-making 重新 cast 为 text-conditioned video generation。用 internet video pretrain，inverse dynamics model 提取 action。没有 robustness 设计。
- **SuSIE** (Black et al. 2023): 用 image editing diffusion model（InstructPix2Pix 风格）生成 sub-goal image，再 low-level controller 执行。引入 data augmentation 应对 perturbation。
- **GR-1** (Wu et al. 2023): autoregressive video prediction + action prediction end-to-end。预训练 video model 帮助 action generation。
- **GEVRM**: video generation + prototypical contrastive state alignment + diffusion policy。三者中唯一显式 instantiate IMC 原理的。

### 9.2 与 Closed-loop Visuomotor Control (Bu et al. 2024) 的区别

Bu et al. 2024 (arXiv:2409.09016) 也提出 closed-loop visuomotor control with generative expectation，但它的 generative expectation 是 reconstruct current observation，没有显式的 goal state。GEVRM 有 explicit goal，且用 contrastive learning 而非 reconstruction 来学 internal model。

### 9.3 与 Diffusion Policy (Chi et al. 2023) 的关系

GEVRM 的 low-level policy 是 Diffusion Policy 的 goal-conditioned 版本。原 Diffusion Policy 的 condition 是 observation history，GEVRM 加上 $\hat{z}$（goal embedding）。这让 policy 能做"forward-looking"的 action 而非 reactive。

### 9.4 与 Dreamer / World Models 的联系

Dreamer (Hafner et al.) 学一个 latent dynamics model 并在里面 rollout。GEVRM 的 video generation model 本质也是个 world model（predict future state given current + action），但 GEVRM 不在 world model 里 rollout 优化，而是用它生成 single horizon 的 goal，然后 policy 执行。这是 "world model as planner" 而非 "world model as imagination rollouts"。

参考 Dreamer: https://arxiv.org/abs/1912.01603

### 9.5 与 Active Inference (Friston) 的哲学联系

Friston 的 Free Energy Principle 里，agent 维护 internal generative model，通过 minimize prediction error (surprise) 来行动。GEVRM 的 state alignment loss $\mathcal{I}_\psi$ 本质上是在 minimize "current state 的 cluster assignment" 和 "goal state 的 cluster assignment" 之间的 divergence——这可以视为一种 prediction error。policy 生成 action 来 reduce 这个 error，符合 active inference 的逻辑。

参考 Free Energy Principle: https://www.sciencedirect.com/science/article/pii/S0079610710001249

### 9.6 与 Sora 的关系

Sora (Brooks et al. 2024) 证明 video generation model 可以作为 world simulator，能模拟物理规律。GEVRM 用类似的 DiT 架构 + spatio-temporal VAE，但加入 random mask 让模型适应 robot manipulation 的 specific dynamics（gripper 抓取、drawer 推拉等）。可以视为 Sora 在 robot domain 的 specialization。

参考 Sora: https://openai.com/research/video-generation-models-as-world-simulators

### 9.7 与 MPC 的对比

经典 MPC 每 step solve 一个 optimization:
$$
\min_{a_{0:H}} \sum_{t=0}^H \|x_t - x_{goal}\|^2 + \|a_t\|_R^2
$$
$$
\text{s.t.} \quad x_{t+1} = f(x_t, a_t)
$$
计算量大，需要已知 dynamics $f$。

GEVRM 用 learned video model 一次性 generate 整段 future，用 diffusion policy 解码 action，是 "generate-then-execute" 而非 "optimize-then-execute"。优势：不需要 explicit dynamics model；劣势：缺乏 hard constraint handling（如 obstacle avoidance）。

### 9.8 与 OpenVLA / RT-2 的根本差异

OpenVLA, RT-2 是 end-to-end VLM → action。GEVRM 是 hierarchical：
- High-level: text → video goal (互联网规模 pretrain)
- Low-level: (current, goal) → action (少量 robot data)

OpenVLA 需要海量 (image, instruction, action) triples，GEVRM 把 language supervision 推到 high-level，low-level 不需要 language。这让 GEVRM 在 low-data robot scenario 下更 sample efficient。

参考 OpenVLA: https://arxiv.org/abs/2406.09246

### 9.9 与 HULC / RoboFlamingo 的对比

HULC (Mees et al. 2022a) 用 multi-modal transformer + contrastive learning align video 和 language。RoboFlamingo 用 pre-trained VLM 做 single-step 理解 + 显式 policy head。它们都是 flat policy（无 hierarchy），且没有显式 future generation。GEVRM 的 hierarchy + future generation 让它能 chain 5 个 instruction（Avg Length 1.62 vs RoboFlamingo 1.31）。

参考 HULC: https://arxiv.org/abs/2112.03227

### 9.10 与 VLA robustness literature 的关系

机器人 VLA robustness 这条线主要有几支：
1. **Data augmentation**: SuSIE, AugLC, DrQ (Kostrikov) — 简单但 narrow
2. **Domain randomization**: Pashevich et al. 2019, OpenAI ERT — sim2real
3. **Domain adaptation**: 但需要 target domain data
4. **Closed-loop with feedback**: GEVRM, Bu et al. 2024

GEVRM 是 (4) 的代表，把 control theory 的 IMC 原理 deep learning 化。

参考 DrQ: https://arxiv.org/abs/2011.12750

### 9.11 Sinkhorn-Knopp 的更深层直觉

为什么必须用 Sinkhorn？如果只用 $p^{\text{source}}$ 自己当 target，模型 collapse 到"所有 sample 都分给某个 prototype"。这是 contrastive learning 的常见 failure mode（比如 SimCLR 不 collapse 是因为负样本对，BYOL 不 collapse 是因为 stop-gradient + EMA）。

SwAV 的 Sinkhorn-Knopp 强制 batch 内每个 cluster 收到 $B/N$ 的总 mass。这是一个 transport polytope constraint，让 cluster assignment 必须均匀分布。从 optimal transport 角度看：
$$
\min_Q \langle Q, C \rangle - \epsilon H(Q)
$$
where $C_{ij} = -\log p_{ij}$, $H(Q)$ 是 entropy, $\epsilon$ 是 regularization。

Sinkhorn 迭代解这个：
$$
Q^{(t+1)} = \text{diag}(u) Q^{(t)} \text{diag}(v), \quad u \leftarrow 1/(Q \mathbf{1}), \quad v \leftarrow 1/(Q^\top \mathbf{1})
$$
反复迭代直到 row 和 column marginal 都达到要求。

### 9.12 与 Model-Based RL 的 latent dynamics learning

Dreamer, PlaNet 学 latent dynamics $z_{t+1} = g(z_t, a_t)$。GEVRM 不显式学 dynamics，而是学"current latent → goal latent"的 alignment。这避开了 dynamics learning 的 difficulty（long-horizon rollout 误差累积），但代价是没有 long-horizon imagination。可以视为 model-based RL 的"轻量版"。

### 9.13 与 Hierarchical RL 的关系

HiP (Ajay et al. 2024) 是 hierarchical 的：text → high-level plan → sub-goals → actions。GEVRM 也是 hierarchy 但只有两层，且 high-level 是 video generation 而非 language planning。HiP 的层级更多但每层都 simpler。

参考 HiP: https://arxiv.org/abs/2310.10639 (Compositional Foundation Models for Hierarchical Planning)

### 9.14 与 Diffusion for Decision Making 的潮流

最近几个工作：
- **Diffuser** (Janner et al. 2022): diffusion over state-action trajectory
- **Decision Diffuser** (Ajay et al. 2022)
- **Diffusion Policy** (Chi et al. 2023): diffusion over action conditioned on observation
- **GEVRM**: diffusion over action conditioned on (current, goal) embeddings

GEVRM 的 contribution 是把 goal conditioning 通过 contrastive prototype embedding 实现，让 conditioning signal 本身就包含 perturbation estimate。

### 9.15 Potential Failure Modes 推测

基于论文结果和架构推测，GEVRM 可能在以下场景失败：
1. **Extreme perturbation**（完全遮挡目标物体 long time）：video planner 无法 generate 合理 goal
2. **Out-of-distribution object**：train 时没见过的 object 形状
3. **Long-horizon task** (>5 chain)：goal generation 累积误差
4. **Real-time requirement**：如果 control frequency 需要数百 Hz，50-step sampling 不够
5. **Multi-modal goal ambiguity**：language instruction 不够 specific 时，video generation 可能 sample 到错误 goal

### 9.16 可能的 Extension 想法

1. **Force/torque sensing** 加入 internal model：当前只用 vision，加 tactile 让 internal embedding 更丰富
2. **Closed-loop re-planning trigger**：检测到 large perturbation 时立刻 refresh goal，不等 L_test=20
3. **Hierarchical prototype**：3000 个 prototype 可能不够细，可以 hierarchical k-means
4. **Goal verification**：生成 goal 后用 VLM 验证是否符合 instruction
5. **3D video generation**：当前是 2D + time，可以扩展到 3D voxel grid + time（NeRF-style world model）
6. **Active perception**：robot 主动调整 camera 视角来 reduce perturbation
7. **Multi-robot collaboration**：goal generation model 可以生成多 robot 协作 scene
8. **Continual learning**：online update prototype set 遇到新 task 时

---

## 10. 关于 Implementation 的细节推测

### 10.1 训练数据规模推测

CALVIN ABC 三个 environment 共有多少 play data？CALVIN 论文说每个 environment 大约 6h play data，~10k trajectories。所以 ABC 加起来 ~30k trajectories。

BridgeData V2 是 internet robot data，~7k trajectories，24 environments。

VAE + DiT 在 BridgeData 上预训练，然后 fine-tune 到 CALVIN/real-world。这是 transfer learning 的 standard recipe。

### 10.2 训练 compute 推测

Table 8 显示 batch_size=6, bf16, ZeRO-2。STDiT3-XL/2 大约 1B 参数。30000 iterations × 6 batch = 180k samples。在 8×A100 估计 1-2 天。

### 10.3 Hyperparameter 的 design choice

- $\delta = 0.1$ (temperature): SwAV 原文用 0.1，沿用
- $N = 3000$ prototypes: 比 SwAV 的 3000 一致，比 DeepCluster 的 10000 小，比 DreamerPro 的 2048 大
- $L_{test} = 20$: 20 步大概对应 4 秒（5Hz）或 0.67 秒（30Hz），是 reasonable 的 goal refresh rate
- $M = 51$ goals: 一次性 generate 51 个 future frames，对应约 1-2 秒 video

### 10.4 为什么用 ResNet-34 而不是 ViT 或 CLIP encoder

推测原因：
1. ResNet-34 输出 spatial pooled feature，维度 512，适合 prototype dot product
2. ResNet 结构让 contrastive learning 训练更 stable（CNN inductive bias 帮助 early training）
3. CLIP encoder frozen 后可能与 robot domain 不 match
4. 轻量化：DiT 已经很重，state encoder 不能太重

---

## 11. 总结：GEVRM 给我的 intuition

把这篇论文 read 完，我建立的 mental model 是：

**GEVRM = Video Planner (想象未来) + State Alignment (评估当下偏离) + Diffusion Policy (纠正动作)**

这三个 component 正好对应 IMC 的三要素：
- Reference input → Video Planner 输出的 goal frames
- Internal model → State Alignment 学到的 prototype assignment
- Feedback controller → Diffusion Policy 在 condition 上做出 robust action

最 clever 的地方是：**用 contrastive prototype learning 来"implicitly simulate 系统对扰动的 response"**。传统 IMC 需要显式建模系统 dynamics，GEVRM 用 contrastive learning 让模型自己学到"current state 在 latent space 应该映射到哪个 cluster"，扰动到来时映射偏离，policy 用这个偏离作为信号。这种 implicit modeling 比 explicit dynamics learning 更 scalable。

video generation model 在这里扮演双重角色：既是 "imagination engine"（generate future goal），又是 "physical law prior"（在 internet video 上预训练，知道物体怎么运动、怎么被抓取）。这呼应了 Sora 作为 world simulator 的论点。

这篇 paper 的 contribution 主要是 architectural/philosophical（把 IMC 思想引入 VLA），而非 algorithmic novelty（每个 component 都有 prior work）。但这种 cross-disciplinary translation 本身有价值——control theory 几十年积累的 robust design 原理，通过 deep learning 工具 instantiate 出来，可能启发更多类似工作。

---

## 12. Key References 汇总

- **GEVRM 本身**：本文
- **IMC 原理**: Garcia & Morari 1982, https://www.semanticscholar.org/paper/Internal-model-control-a-unifying-review-and-new-Garcia-Morari/
- **UniPi**: Du et al. 2024, https://arxiv.org/abs/2310.07117
- **SuSIE**: Black et al. 2023, https://arxiv.org/abs/2310.10639
- **GR-1**: Wu et al. 2023, https://arxiv.org/abs/2312.13139
- **Diffusion Policy**: Chi et al. 2023, https://arxiv.org/abs/2303.04137
- **SwAV**: Caron et al. 2020, https://arxiv.org/abs/2006.09882
- **Sinkhorn**: Cuturi 2013, https://arxiv.org/abs/1306.2597
- **Rectified Flow**: Liu et al. 2022, https://arxiv.org/abs/2209.03003
- **Stable Diffusion 3**: Esser et al. 2024, https://arxiv.org/abs/2403.03206
- **Open-Sora**: Zheng et al. 2024, https://github.com/hpcaitech/Open-Sora
- **CALVIN**: Mees et al. 2022, https://arxiv.org/abs/2112.03227
- **BridgeData V2**: Walke et al. 2023, https://arxiv.org/abs/2308.12952
- **Sora**: Brooks et al. 2024, https://openai.com/research/video-generation-models-as-world-simulators
- **Dreamer**: Hafner et al. 2019, https://arxiv.org/abs/1912.01603
- **Active Inference**: Friston, https://www.sciencedirect.com/science/article/pii/S0079610710001249
- **OpenVLA**: Kim et al. 2024, https://arxiv.org/abs/2406.09246
- **Diffuser**: Janner et al. 2022, https://arxiv.org/abs/2205.09991
- **T5**: Raffel et al. 2020, https://arxiv.org/abs/1910.10683
- **Latent Diffusion**: Rombach et al. 2022, https://arxiv.org/abs/2112.10752
- **ResNet**: He et al. 2016, https://arxiv.org/abs/1512.03385
- **Closed-loop visuomotor**: Bu et al. 2024, https://arxiv.org/abs/2409.09016
- **HiP**: Ajay et al. 2024, https://arxiv.org/abs/2310.10639
- **HULC**: Mees et al. 2022, https://arxiv.org/abs/2112.03227
- **DreamerPro**: Deng et al. 2022, https://arxiv.org/abs/2110.14523
- **MAGVIT-v2**: Yu et al. 2023, https://arxiv.org/abs/2310.05737
- **UL2**: Tay et al. 2022, https://arxiv.org/abs/2205.05131
- **RoboFlamingo**: Li et al. 2023, https://arxiv.org/abs/2311.01378
- **AVDC**: Ko et al. 2023, https://arxiv.org/abs/2310.08576

---

## 13. 最后的思考

GEVRM 让我最 excited 的点是它证明了 control theory 的 classical 原理可以"翻译"到 deep learning framework 而不失其本质。这种 translation 是双向的：
- Deep learning 帮助 control theory 处理 high-dim perception (vision, language)
- Control theory 帮助 deep learning 处理 robustness, stability, closed-loop feedback

未来这条线可能继续延展：
- **Lyapunov stability** 注入 neural policy training（确保 closed-loop stability guarantee）
- **Optimal control** 的 LQR/MPC 思想用 diffusion policy 实现
- **System identification** 用 contrastive learning 做 implicit system model learning
- **Adaptive control** 用 online prototype update 实现

希望这些直觉和细节对你的理解有帮助。如果你对某个具体 component 想更深入（比如 Sinkhorn 的实现细节、Rectified Flow 的数学推导、或 video VAE 的 causal convolution 设计），可以继续问。
