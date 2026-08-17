---
source_pdf: Video Generators are Robot Policies.pdf
paper_sha256: b816cc80f0368c3ab09df906de831b47c93d7f78b9a19af1ff698ae26eaa8165
processed_at: '2026-08-13T00:33:03-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

我们用最直白的语言，把这个 paper 的核心逻辑和直觉重新梳理一遍。为了 build your intuition，我会用几个 mental model 来拆解。

---

## 1. 核心直觉：从"照猫画虎"到"脑内模拟"

传统的 Behavior Cloning (BC) 逻辑是：给神经网络看一张当前环境的图片，让它直接输出机械臂下一步该怎么动。这就像让一个人闭着眼睛，只凭刚才瞥了一眼的状态，直接手把手地操作。一旦桌上的杯子换了个位置，或者换个颜色，模型就傻眼了，因为它只记住了特定的 pixel 到 action 的映射。

这篇 paper 的 insight 非常优雅：**如果你能"想象"出完成任务的全过程视频，那么推导出具体的机械臂动作就极其简单了。**

你可以这样理解：高水平的运动员在跳水或者打高尔夫之前，脑子里会先过一遍完美的动作画面。这个 Video Generation model (比如 SVD) 就扮演了"脑内模拟器"的角色。它在 internet 海量视频上预训练过，已经懂得了物理世界的规律（杯子被推会滑动、抽屉拉了会出来）。当你给它一个任务描述，它先在脑内"幻想"出机械臂完成任务的完整视频。既然视频里机械臂的轨迹都画出来了，后面的 Action decoder 只需要做一件事：把这个视觉轨迹"翻译"成电机指令。

---

## 2. 架构拆解：Master 与 Slave 的双人舞

架构上，paper 设计了两个并行的 Diffusion U-Net。

**Video U-Net ($\mu_\theta$) —— The Master (脑内模拟器)**
基于 Stable Video Diffusion (SVD)。输入是初始观测图片 $v_0$ 和任务文本 $c$。它的工作是在 latent space 里一步步 denoise，生成未来的视频帧。

**Action U-Net ($\alpha_\theta$) —— The Slave (翻译官)**
基于 Diffusion Policy。它自己不去理解世界，它的工作是读取 Master 的脑电波。在每个 denoising step $i$，从 Video U-Net 的 decoder 层（第 9, 14, 17, 20, 23 层）抽取 spatiotemporal features，经过 CNN adapter 压缩成一个 vector $h_i$。Action U-Net 就以这个 $h_i$ 为条件，同时进行 denoise，输出机械臂的 7-DoF action sequence。

### 公式背后的直觉

Video 的训练 loss：
$$L_{video} = \mathbb{E}_{z_0, \epsilon, i}[\|\epsilon - \mu_\theta(z_i, i, \phi(c), z_{i,0})\|^2]$$
- $z_0$: 干净的 ground truth latent video
- $\epsilon$: 注入的 Gaussian noise
- $i$: diffusion timestep
- $\phi(c)$: CLIP text embedding
- $z_{i,0}$: 第一帧的 noisy latent
这个 loss 就是在逼着 Video U-Net 学会"想象"机械臂做事的物理过程。

Action 的训练 loss：
$$L_{action} = \mathbb{E}_{a_0, \epsilon, i}[\|\epsilon - \alpha_\theta(a_i, i, h_i)\|^2]$$
- $a_0$: ground truth action sequence
- $h_i$: 从 Video U-Net 抽取的 feature vector
这个 loss 让 Action U-Net 学会把 Video U-Net 的脑电波翻译成 action。

### 最关键的 Stop-Gradient

这里有个反直觉但极具启发性的 design choice：**Action loss 产生的 gradient 绝对不允许流回 Video U-Net**。

直觉解释：Video U-Net 见过千千万万的 internet 视频，它的世界知识非常丰富。你只有 50 条 demonstration 数据，如果让 action loss 的梯度流回去，这 50 条数据微弱的信号会把 Video U-Net 原本丰富的 prior 给带偏、给 overfit 污染掉。所以必须 freeze 住 Video U-Net 的知识，让 Action decoder 去单方面适应它。Table 3 的数据直接印证了这一点：Joint training（允许梯度流回）成功率只有 0.57，而 2-Stage（stop gradient，分开训练）成功率升到了 0.63。

---

## 3. Ablation 实验：为什么"想象"能带来 Generalization？

这篇 paper 最 enlightening 的部分是它对 prediction horizon 的分析（Figure 3）。

他们做了一个实验：action prediction 固定预测 1.6 秒，但让 video generation 预测的时间长度从 0 秒（只看当前帧）一直拉长。结果发现，对于有 distribution shift（测试时物体位置变化）的任务，预测时间越长，成功率飙升；对于没 distribution shift 的任务，提升不明显。

**直觉解释**：如果你只看眼前这一帧，模型只能靠死记硬背的 reflex 反应。如果模型被迫"想象"未来 32 步视频，它就必须在内部构建出"杯子被推了会往哪滑"、"抽屉拉开是什么角度"的 environment dynamics。一旦它学会了这种 dynamics，换一个新物体、新位置，它的 dynamics engine 依然能正确推演，这就带来了 generalization。

---

## 4. 颠覆性的发现：Zero-Shot Action 生成

Figure 4 的实验堪称神来之笔。
研究人员做了一个极端测试：
1. 让 Video U-Net 看全部 24 个任务的视频（没有 action label）。
2. 只让 Action U-Net 在其中 12 个任务上学习（有 action label）。
3. 测试时，让模型去做那没见过的 12 个任务。

结果：Video Policy 在从没见过 action label 的任务上，依然有 0.21-0.28 的成功率。而传统的 Diffusion Policy (DP-ResNet) 在这些任务上成功率全是 0.00。

**直觉解释**：Action decoder 并没有学习具体的"任务策略"，它只学会了"如何从视频中读取机械臂轨迹"。真正的 policy 已经在 Video U-Net 里形成了。只要 Video U-Net 能"想象"出这个新任务该怎么做，Action decoder 就能照葫芦画瓢把 action 输出来。这彻底证明了标题的 claim：Video Generators are Robot Policies。

---

## 5. 实验数据：50 条 Demo 干翻 3000 条

在 RoboCasa benchmark 上（Table 1），数据对比极其震撼：
- GR00T (NVIDIA) 用了 300 条 MimicGen demos，平均成功率 0.50。
- DP-VLA 用了 3000 条 demos，平均成功率 0.57。
- Video Policy 只用了 50 条 human demos，平均成功率 0.63。

为什么数据效率这么高？因为 Video U-Net 已经预训练好了世界模型，那 50 条 demo 的作用仅仅是用来训练一个轻量级的"翻译官"，把视觉想象映射到机械臂关节上。翻译一种语言可比从头学习物理世界容易太多了。

在真实世界实验中（Table 4），用 handheld gripper 采集数据，测试换新物体、换新背景。Open Drawer 和 Pick and Place 的成功率都在 0.8-1.0 之间，极其 robust。

---

## 6. 局限与未来：通往 AGI 的跳板？

paper 老实承认了瓶颈：9 秒生成 25 帧视频，这速度完全没法做 real-time closed-loop control。而且 SVD 的物理 prior 还不够强，遇到把杯子扶正这种精细的物理交互，video generation 会产生幻觉，导致 Stack Cups 任务成功率只有 0.2-0.3。

但这里有一个巨大的联想空间：这篇 paradigm 的威力直接绑定于背后的 Video Generation model。SVD 相对原始，如果换成 OpenAI 的 Sora 2 或者更高级的世界模型，物理一致性会指数级提升。当 world simulator 足够强，这套架构的 bottleneck 就消失了。后续如果结合 Consistency Models 或者 Flow Matching 把推理速度压到实时，这极可能成为未来 Robot Foundation Model 的主流架构之一。

**相关参考链接：**
- Video Policy 项目主页: https://videopolicy.cs.columbia.edu
- SVD (Stable Video Diffusion): https://stability.ai/news/stable-video-diffusion-open-ai-video-model
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- RoboCasa Benchmark: https://robocasa.ai/
- Sora 技术报告 (Video generation models as world simulators): https://openai.com/research/video-generation-models-as-world-simulators

---

# Video Generators are Robot Policies — 一篇用 Video Diffusion 当 Policy 的 paper

这是一篇来自 Columbia University (Carl Vondrick lab) 和 Toyota Research Institute (TRI) 的工作,第一作者 Junbang Liang 也是 Dreamitate 的作者。整个故事的核心 insight 极其简洁: **如果 video generation model 能够"想象"出 robot 执行任务的过程, 那么从这些"想象"中 decode 出 robot action 只需要极少量的 demonstration 数据**。Video generation model 实际上承担了 policy 的角色, action decoder 只是一个"接口"。

下面我从底层直觉开始, 把这篇 paper 拆解到位。

---

## 1. 核心动机: BC 的 generalization 困境

传统的 Behavior Cloning (BC) 方法, 比如 Diffusion Policy, 在精确任务上很强, 但面临两个根本性问题:

1. **Perceptual distribution shift**: 测试时物体颜色、位置、背景变了, end-to-end policy 直接崩。
2. **Behavioral distribution shift**: 切换到新任务时, 完全 retrain, 数据需求大。

Computer Vision 和 NLP 解决这类问题的范式是 scale up data — internet 上的 image/text 数据几乎是无限的。但 robot demonstration 数据采集极其昂贵 (teleoperation、handheld gripper、precision tracking), 无法 scale。

**关键 insight**: internet 上的 video 是近乎无限的, 并且已经隐式编码了 "物体如何运动、动作如何影响世界" 的物理 prior。如果能把 video generation model 的 prior "蒸馏" 到 robot policy, 就能突破 demonstration data 的瓶颈。

paper 的标题本身就是结论 — **video generators 已经是 robot policies**。

---

## 2. 整体架构: 双 U-Net 的 modular design

让我把架构图 (Figure 2) 拆开讲。

### 2.1 输入与 conditioning

- 输入: 初始观测 $v_0 \in \mathbb{R}^{t \times c \times h \times w}$, 任务描述文本 $c$
- 视频部分: 用 Stable Video Diffusion (SVD) 作为 backbone。SVD 是 latent diffusion model, 所以 $v_0$ 先通过 frozen VAE encoder 得到 latent $z_0 = \mathrm{VAE}(v_0)$
- 文本 $c$ 通过 CLIP text encoder 得到 embedding $\phi(c)$, 通过 cross-attention 注入 video U-Net $\mu_\theta$
- Noisy future frames $z_1, ..., z_t$ 通过 channel-wise concat 与 $z_0$ 拼接 (image-to-video conditioning)

### 2.2 Video U-Net $\mu_\theta$ 与 Action U-Net $\alpha_\theta$ 的耦合

这是这篇 paper 最有意思的设计。两个 U-Net **不是简单的级联**, 而是**在每个 denoising step 紧耦合**:

在 video U-Net 的 decoder 中, 取出 5 个间隔均匀的中间 hidden embedding (layer 9, 14, 17, 20, 23), 这些是 spatiotemporal features。然后通过一个 CNN adapter, 把这些 spatiotemporal latent 压成一个 global vector $h_i$ (其中 $i$ 是 denoising step), 作为 action U-Net 的 global conditioning。

公式 (1) 和 (2) 形式化了这个过程:

$$\{\hat{v_t}\} = f(v_0, c) \tag{1}$$

$$\{a_t\} = g(\psi_0, ..., \psi_i) \quad \text{where} \quad \psi_i = f_i(v_0, c) \tag{2}$$

- $v_0$: 初始 RGB 观测
- $c$: 任务文本描述
- $f$: video generator (SVD-based)
- $g$: action predictor
- $\psi_i = f_i(v_0, c)$: video generator 第 $i$ 层 hidden feature
- $a_t \in \mathbb{R}^k$: 第 $t$ 步的 robot action, $k$ 是 end-effector 的 action 维度 (实验中是 7-dim: 6-DoF pose + gripper open/close scalar)

action U-Net 本身是 Diffusion Policy 改造的 1D CNN U-Net, 接收 $h_i$ 作为 conditioning:

$$\{a_t\} = \alpha_\theta(a_i, i, h_i) \tag{3}$$

- $a_i$: 第 $i$ 个 denoising step 的 noisy action sequence
- $i$: diffusion timestep embedding
- $h_i$: video U-Net 在第 $i$ 个 denoising step 提取的 global feature
- $\alpha_\theta$: action U-Net

注意: action U-Net 和 video U-Net 的 denoising step 是**同步**的 — 每个 denoising step 都做一次 video denoise 和 action denoise, 两者的 noise level 对齐。这是 joint denoising 的核心机制, 让 pixel 和 action 在 representation 上深度耦合。

### 2.3 为什么这个设计 elegant

关键点在于: action U-Net 仅仅把 video U-Net 的 intermediate features 作为 **conditional input**, 不参与 video generation 的优化。这意味着 video U-Net 的训练目标是**纯 video generation**, 它可以在 action-free 的 video data 上预训练 (这是这个工作后续 ablation 的核心 trick), 而 action head 只需要少量带 action label 的 demonstration 数据来 fine-tune。

这正好对应了 paper 的标题和摘要中说的 "modular design enables learning from action-free video data"。

---

## 3. 训练: 为什么 stop-gradient 是关键

### 3.1 两个 loss

Video diffusion loss 是标准 DDPM 形式:

$$L_{video} = \mathbb{E}_{z_0, \epsilon, i}[\|\epsilon - \mu_\theta(z_i, i, \phi(c), z_{i,0})\|^2] \tag{4}$$

- $z_0$: 干净的 latent video (ground truth VAE-encoded)
- $\epsilon \sim \mathcal{N}(0, I)$: 注入的 Gaussian noise
- $z_i$: 第 $i$ 个 diffusion step 的 noisy latent
- $i$: diffusion timestep
- $\phi(c)$: CLIP text embedding
- $z_{i,0}$: 第一帧的 noisy latent (image-to-video conditioning)
- $\mu_\theta$: video U-Net 预测的 noise (或 $x_0$)

Action diffusion loss 类似:

$$L_{action} = \mathbb{E}_{a_0, \epsilon, i}[\|\epsilon - \alpha_\theta(a_i, i, h_i)\|^2] \tag{5}$$

- $a_0$: ground truth action sequence
- $h_i$: CNN adapter 在 noise level $i$ 输出的 video feature vector

### 3.2 stop-gradient 的关键性

paper 在 3.3 节明确说: **从 $L_{action}$ 到 $\mu_\theta$ 的 gradient 被截断** (stop gradient), 即 action loss 不会更新 video U-Net 的权重。

这个 design choice 的理由是: video generation model 已经经过 internet-scale 预训练, 它应该 "drive the policy"; action head 只是 "decode" 它的中间表征。如果允许 action loss 反传回 $\mu_\theta$, 那么 video U-Net 会被少量 demonstration 数据 overfit 污染, 丢失它的 generalization prior。

这个 design choice 在 Table 3 的 ablation 中得到验证:
- Joint training (允许 gradient 流回 video U-Net): 0.57 平均成功率
- 2-Stage training (video 先训练, 然后 freeze, 单独训 action head): **0.63** 平均成功率
- No Video Tuning (直接用 vanilla SVD, 不在 RoboCasa 上 fine-tune): 0.09 (基本废了)

**0.09 vs 0.63 的对比极具说服力** — 这说明仅仅用 internet 上预训练的 SVD 是不够的, 必须在 RoboCasa 的 robot execution video 上 fine-tune video generation; 而 action head 反而不需要从 action loss 学到"任务知识", 它只是在 decode。

这是这篇 paper 最 profound 的发现: **在 robot-rollout video 上学到的 video generation, 是 policy 的本体; action decoder 只是个 adapter**。

### 3.3 为什么 video generation 是"更通用"的目标

直觉上, 这是因为: video generation 必须建模**环境 dynamics + robot motion + object interaction**; 而 action prediction 只建模 end-effector 的轨迹。前者包含的信息量严格大于后者, 且 video data 可以 action-free 采集 (相机随便录)。

---

## 4. 实验: 数据极其漂亮

### 4.1 RoboCasa benchmark

RoboCasa 是 24 个 manipulation task 的大规模仿真 benchmark, 包含 Pick-and-Place、Doors、Drawers、Knobs、Levers、Buttons、Insertion 等 7 大类任务。

Table 1 比较了 9 个 baseline:
- 3DA, DP3 (3D representation-based)
- DP-ResNet, DP-CLIP (Diffusion Policy 不同 visual encoder)
- GR00T (NVIDIA 的 foundation model, 300 demos)
- FPV (point cloud + visual fusion)
- DP-VLA (3000 MimicGen demos)
- UVA (Unified Video Action Model, 并发工作)
- Ours 50 demos / Ours 300 demos

Video Policy 用 50 demos 就达到 0.63 平均成功率, 而 GR00T 用 300 demos 是 0.50, DP-VLA 用 3000 demos 是 0.57。**在数据效率上 6x 到 60x 的优势**。

特别值得注意的是 Pick-and-Place 类任务 (训练和测试环境有显著 distribution shift): Video Policy 在 PnPCounterToStove 上 0.58, PnPSinkToCounter 上 0.64, PnPStoveToCounter 上 0.64, 全面碾压 baseline (大多在 0.0-0.3)。这说明 video prior 对 distribution shift 极其 robust。

### 4.2 Libero10 benchmark

Libero10 是更短 horizon 的任务集, 10 个任务, Table 2 显示 Video Policy 达到 0.94 平均成功率, 超过 $\pi_0$-FAST (0.60) 和 $\pi_0$ (0.85), 接近 UVA (0.90)。

Table 9 (附录) 显示 per-task, 多个任务达到 1.00 (100% 成功率)。

### 4.3 为什么 UVA 不行

UVA (Unified Video Action Model, [49]) 是并发工作, 也做 joint video + action diffusion, 但它在 RoboCasa 上表现不佳 (0.50)。paper 在 4.3 节指出原因: UVA 过度依赖 single-camera setup, 而 RoboCasa 是 multi-view。Video Policy 的架构 (SVD 直接接受 multi-view 输入) 可以轻松适应任意 camera 配置。

这其实暗示了一个更深的点: **从 SVD 这种已经训练好的 video foundation model 出发, 比从头训 joint video-action model 更有优势**, 因为前者的 visual prior 已经足够丰富。

---

## 5. Ablation: prediction horizon 和 action-free data

### 5.1 Video prediction horizon 的影响 (Figure 3)

这是我认为最 enlightening 的 ablation 之一。Action prediction horizon 固定为 1.6 秒, 但 video prediction horizon 从 0 (只重建 conditioning frame) 一直变到更长 rollout。

结果:
- 对**有 distribution shift 的任务** (Pick-and-Place 类): video horizon 从 0 增加到 32 步, 成功率从 ~0.15 飙升到 ~0.55, **巨大提升**
- 对**没有 distribution shift 的任务** (Doors、Drawers 等): 提升温和, 从 ~0.45 到 ~0.65

这个对比揭示了: **video generation 学的是 environment dynamics**。当任务需要 generalization 到新物体/新位置时, 模型必须"理解"环境会怎么演化; 而对 in-distribution 任务, 即使 video horizon 短也能凑合 (因为环境熟悉)。

Table 8 给出 per-task 细节, 最 striking 的是 PnPCounterToStove: 32 steps 是 0.82, 0 steps 是 0.02 — 这是 40x 的差距, 完美印证 hypothesis。

### 5.2 Action-free video 的威力 (Figure 4)

这个 ablation 直接验证了 paper 的核心 claim: action-free video data 能帮助 generalize 到没见过 action label 的任务。

实验设计:
- Video U-Net 在全部 24 个任务的 video data 上 fine-tune (Stage 1, 无 action)
- Action U-Net 只在 12 个随机采样的任务上训练 (有 action label)
- 测试在全部 24 个任务上

对比 baseline: DP-ResNet 也只在这 12 个任务上训练 (没有 video generation 阶段)。

结果 (Table 6):
- Ours (Half Tasks): 在训练过的 12 个任务上 0.41, 在没训练过的 12 个任务上 0.21-0.28 (每个 task 都有一定成功率!)
- DP-ResNet (Half Tasks): 在没训练的任务上几乎全部 0.00

**这是 paper 最强的论证**: 即使 action head 从来没见过这些任务的 action label, 只要 video generation 学过这些任务的视频, 就能 zero-shot 生成 action。这彻底证明 video generation 是 policy 的本质。

---

## 6. 真实世界实验: handheld gripper 采集 + Franka 执行

### 6.1 数据采集

Real-world 数据采集方式很有意思 — 用一个 handheld gripper (配备 Intel RealSense T265 tracking 相机 + 单轴 load cell 测量 grasp force + ArUco marker 追踪 jaw opening), 人在做任务, 同时记录 RGB video (左、右、gripper 三个视角, 30Hz) 和 action state (6D pose + jaw width + force)。

每个任务采集 200 demos。这种方式比 teleoperation 采集更快、更自然, 因为它脱离 robot 本体 (类似 UMI 的思路)。

### 6.2 5 个任务的 generalization 测试

Table 4 评估三个维度的 generalization:
- **Object location**: 物体位置变化 (最小 distribution shift)
- **Unseen objects**: 全新物体 (中等)
- **Unseen background**: 训练白色桌面, 测试黑/红/蓝桌布

结果分析:
- **Open Drawer**: 三个维度都 0.8-1.0, 极其 robust
- **Pick and Place**: 三个维度都 0.8-1.0, 即使是透明杯这种 hard case
- **M&Ms to Cup**: 在 unseen background 上跌到 0.2, 因为 M&M 很小, 背景颜色影响 gripper 定位
- **Upright Object / Stack Cups**: 0.2-0.8 之间, paper 解释是 SVD 的 physical prior 不够强, video 会生成不现实的 upright placement 或 gripper toppling

### 6.3 Figure 5 的 qualitative examples

最 impressive 的例子:
- 第一行: Pick and Place, 物体放进一个透明杯 (透明物体对 BC 极难, 因为 visual feature 弱), Video Policy 成功完成
- 第二行: unseen object (形状不规则), 成功 grasp 并放置
- 第四行: 蓝色背景, 成功

Figure 14、15 展示 video prediction 和真实 rollout 的对齐, 三视角的预测视频和真实执行在 gripper motion、object interaction 上高度一致 — 这印证了 paper 的核心 claim, "task success is closely tied to the generated video"。

---

## 7. 局限性和未解决问题

paper 在 Section 6 诚实承认:
1. 只验证了 SVD 一个 video backbone (没试 Sora、Kling、Veo 等)
2. 只一个 real-world embodiment (handheld gripper + Franka)
3. **Computational cost 是最大瓶颈**: 25 帧 video 在 A100 上要 9 秒, 无法实时控制; 论文寄希望于 consistency models、rectified flow 等 diffusion 加速技术
4. 物理 prior 不足: Upright Object、Stack Cups 等需要精细物理理解的任务, video generation 会幻觉

### 7.1 我自己的几个延伸思考

- **Sora-class model 应该会大幅提升**: Sora 的物理一致性远超 SVD, 如果换成 Sora 作为 backbone, Upright Object 这类失败案例可能直接解决。这是这个 paradigm 最 exciting 的点 — 随着视频模型 scale up, robot policy 自动受益。
- **Action label 采集 vs 视频采集**: 这套范式彻底切断了 "robot 必须自己采集数据" 的枷锁。任何 YouTube 上的 cooking、装配视频都可能成为 action-free supervision。这是迈向 internet-scale robot learning 的真正路径。
- **World model 范式统一**: 这篇 paper 和 Sora blog "Video generation models as world simulators" 在精神上完全契合 — 都是 "video generation is a world model, 从中可以提取 control signal"。Sora 是 simulator, Video Policy 是 policy, 二者本质同源。
- **VLA 范式的另一种实现**: 与 $\pi_0$、GR00T、OpenVLA 这类 VLA (Vision-Language-Action) 不同, Video Policy 没有显式 token 化 action, 而是让 action 通过 video generation 的中间 representation 被"提取"出来。这条路线可能比 VLA 的 action tokenization 更优雅, 因为它不需要 quantize action。
- **Inference speed 是 deployment 的死结**: 9 秒生成 25 帧, 完全无法 close-loop 控制。要么用 diffusion 加速 (consistency distillation, Flow Matching), 要么用 amortized policy distillation (把 video policy 蒸馏成 fast non-diffusion policy)。这是这套范式能落地的前提。

---

## 8. 关键参考文献与链接

paper 网站: https://videopolicy.cs.columbia.edu

- Stable Video Diffusion (SVD): https://stability.ai/news/stable-video-diffusion-open-ai-video-model / arXiv: https://arxiv.org/abs/2311.15127
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/ / arXiv: https://arxiv.org/abs/2303.04137
- RoboCasa: https://robocasa.ai/ / arXiv: https://arxiv.org/abs/2406.02523
- Libero: https://libero-project.github.io/
- GR00T N1 (NVIDIA): https://arxiv.org/abs/2503.14734
- UVA (Unified Video Action Model): https://unified-video-action-model.github.io/
- Dreamitate (前作, 同作者): https://dreamitate.cs.columbia.edu/
- Sora (OpenAI blog "Video generation models as world simulators"): https://openai.com/research/video-generation-models-as-world-simulators
- $\pi_0$ (Physical Intelligence): https://www.physicalintelligence.company/blog/pi0
- UMI (Universal Manipulation Interface, handheld gripper 思路来源): https://universal-manipulation-interface.github.io/
- MimicGen (自动生成 demonstration): https://mimicgen.github.io/
- Consistency Models (加速 diffusion 推理): https://arxiv.org/abs/2303.01969
- Rectified Flow (SD3): https://arxiv.org/abs/2403.03206
- UniPi (Du et al., 视频生成当 policy 的早期工作): https://universal-policy.github.io/

---

## 9. 一句话总结

这篇 paper 的核心 thesis, 用 Karpathy 式的语言来说: **policy 不是从 action label 学出来的, 而是从 "想象任务的执行过程" 中浮现出来的**。Video diffusion model 用 internet-scale 数据学会了 "想象", action decoder 只是把这种想象翻译成 motor command。当 video generation 学得足够好, action label 的需求就指数级下降, 而 generalization 反而上升。这是把 internet 视频数据真正接入 robot control 的最干净的工作之一。

如果 Sora 级别的 video model 后续接入这套 paradigm, 我们可能正在见证 robot learning 从 "teleop data scarcity" 走向 "internet video abundance" 的范式切换。
