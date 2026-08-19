---
source_pdf: GR-2 A Generative Video-Language-Action Model with.pdf
paper_sha256: 8b9bf7c3de182ce263ca65d103fc3a84aa0c70d13aa88f6b8dd5cdf5df2f5f3a
processed_at: '2026-08-19T09:45:15-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的，Andrej，我们抛开学术腔调，用最直白的话来聊聊 GR-2 到底干了什么，以及它为什么能 work。

如果给一个婴儿看几千万个小时人类做饭、收拾屋子、活动的视频，然后再递给他一套机械臂让他模仿，他大概率会比从零开始学要快得多。GR-2 的核心 intuition 就是这个。它是一个 GPT-style 的 transformer，但是它“吐”出来的不光是文字，而是未来的图像和机械臂的动作轨迹。它本质上把 robot manipulation 变成了一个 sequence modeling 问题：理解过去，预测未来，并且把动作当成是“播放”这个未来的媒介。

下面我依然为你拆解里面的技术细节，构建更深的 intuition。

### 1. 核心直觉：Video Generation as Implicit World Model

Robot learning 最大的痛点是 data 太少。你搞不到几百万条真实机械臂抓取的轨迹。但是 YouTube 上有无限多的人类操作视频。GR-2 的策略就是：先在 38 million 个 internet video clips 上做 autoregressive pre-training，目标是给定 text 和当前 frame，预测接下来的 frames。

这里面的 intuition 是：如果模型能准确预测下一帧会发生什么，它就必须理解物理世界的 dynamics——杯子被打翻水会流出来，手抓取物体时物体会有位移。这种对物理世界和因果关系的理解，被压缩成了 transformer 里的 weights。当它后续去学机械臂动作时，它不需要再从零学习这些 commonsense，它只需要学习怎么把它内部映射的“未来视觉世界”，通过机械臂的动作在现实里“演”出来。

### 2. 架构与公式解析：一切都是 Token

为了能用 GPT 架构，GR-2 把所有的输入都变成了 discrete tokens。

来看 paper 里的核心公式 (1) 和 (2)：

$$ \mathbf{a}_{t:t+k} = \pi (l, \mathbf{o}_{t-h:t}, \mathbf{s}_{t-h:t}) \tag{1} $$

$$ \pi (l, \mathbf{o}_{t-h:t}, \mathbf{s}_{t-h:t}) \to \mathbf{o}_{t+1}, \mathbf{a}_{t:t+k} \tag{2} $$

*   **$\pi$**: 代表整个 GR-2 模型。
*   **$l$**: Language instruction，比如 "pick up the yellow bottle"。
*   **$\mathbf{o}_{t-h:t}$**: Observation history。$h$ 是 history length。这是用 frozen VQGAN 转成 image tokens 的历史画面序列。
*   **$\mathbf{s}_{t-h:t}$**: Robot state history。机械臂的 end-effector position, rotation 和 gripper state，通过 linear layer 映射成 tokens。
*   **$\mathbf{o}_{t+1}$**: 预测出的 future image token。这是公式 (2) 相比 (1) 多出来的东西，也是 joint training 的核心。
*   **$\mathbf{a}_{t:t+k}$**: 预测出的 action trajectory。$k$ 是预测的未来时间步长度。这通过 conditional VAE (cVAE) 生成。

在 architecture 层面，输入序列就是 `[Text Tokens] + [Image View 1 Tokens] + [Image View 2 Tokens] + [Robot State Tokens]`。Transformer 预测出未来每个视角的 image tokens，以及 action tokens。用大白话说，模型同时干两件事：想象下一步看到的画面，同时算出接下来一段时间的连续动作。用 cVAE 生成一段长为 $k$ 的轨迹，而不是一步一步生成单点 action，这极大地保证了轨迹的平滑性，避免了 jittering。

### 3. Fine-tuning 阶段的数据魔法

Pre-training 阶段是在互联网视频上跑的，Fine-tuning 阶段才真正用到了 robot data。为了让模型泛化到没见过的场景，他们用了一招很赛博朋克的 data augmentation：

1.  **Object Insertion**: 训练一个 diffusion model，往现有的训练视频里“塞”入新的物体。
2.  **Background Changing**: 用 SAM (Segment Anything Model) 把背景抠出来，然后用 Latte (一个 video generation model) 把背景换掉，同时保持前景的机械臂和物体的运动完全不变。

这种做法生成了海量的“新环境”数据，直接解决了机器人数据多样性不足的死穴。

### 4. 部署细节：高层想象，底层控制

GR-2 网络输出的是 Cartesian space 的 end-effector trajectory。但是真实的 Kinova Gen3 机械臂需要 joint 级别的控制指令。

他们用了一个叫 Whole-Body Control (WBC) 的 algorithm。WBC 做的事情是：拿到 GR-2 生成的 Cartesian trajectory，先用 trajectory optimization 把它变得极度平滑，然后考虑 collision constraints 和 manipulability，解算出 joint actions，最后以 200Hz 的频率发给电机。高层网络负责天马行空地想象和规划，底层控制算法负责把物理世界的摩擦、碰撞、重力抹平，这种分层设计非常实用。

### 5. 实验数据与 Scaling 的胜利

数据非常有说服力：
*   **Multi-task Learning**: 在 105 个真实任务上，GR-2 达到了 97.7% 的 success rate。在换了背景和厨房环境的 Unseen Environments 中，达到 71.7%；加上前面说的 data augmentation，直接飙到 87.0%。
*   **Bin Picking**: 在工业级的抓取任务上，有 122 个物体（67个没见过）。GR-2 把 average success rate 从 GR-1 的 33.3% 拉升到了 79.0%。对于 transparent (透明) 和 deformable (软体) 物体也表现很好，这说明 VQGAN 提取的 visual feature 极其鲁棒。
*   **Scaling Law**: 从 30M 到 719M 参数，模型越大，video prediction 的 validation loss 越低，real robot 的 success rate 越高。这证明了这条路还没撞到天花板，更多算力和数据可以继续推高性能。

总结一下，GR-2 极其优雅地证明了：通过看视频建立一个 internal world model，然后把 robot action 当作对这个世界模型的 querying 和 replay，是通往 generalist robot agent 的一条黄金大道。

### Reference Links

*   **GR-2 Project Page**: [https://gr2-manipulation.github.io](https://gr2-manipulation.github.io)
*   **GR-1 Paper (Predecessor)**: [Unleashing Large-Scale Video Generative Pre-Training for Visual Robot Manipulation](https://arxiv.org/abs/2312.13139)
*   **Sora (Video Generation as World Simulators)**: [Video generation models as world simulators](https://openai.com/research/video-generation-models-as-world-simulators)
*   **VPT (Video Pre-Training in Minecraft)**: [VPT: Learning to Act by Watching Unlabeled Online Videos](https://openai.com/index/vpt/)
*   **VQGAN (Image Tokenizer)**: [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841)
*   **ACT (Action Chunking using cVAE)**: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
*   **SAM (Segment Anything)**: [Segment Anything](https://arxiv.org/abs/2304.02643)
*   **RT-2 (Vision-Language-Action Models)**: [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)
*   **Latte (Video Generation Model for Data Aug)**: [Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/abs/2401.03048)

---

Andrej，这篇 ByteDance Research 的 GR-2 paper 非常精彩，它本质上沿着你一直以来推崇的 "sequence modeling is all you need" 哲学，将 robot manipulation 统一到了 autoregressive next-token prediction 的框架下。GR-2 的核心 intuition 在于：通过大规模 internet video pre-training，模型学习到了一个强大的 internal world model，而 robot action prediction 可以被视为对这个预测出的未来视觉世界的 "replay"。

下面我为你进行深度的技术拆解，构建对这个 system 的 intuition。

### 1. 核心架构与公式解析

GR-2 是一个 GPT-style 的 transformer。它的输入是 language instruction、video frames 以及 robot states，输出则是 future video frames 和 action trajectories。

**核心公式解析：**

在 paper 的公式 (1) 和 (2) 中，定义了 policy $\pi$ 的行为：

$$ \mathbf{a}_{t:t+k} = \pi (l, \mathbf{o}_{t-h:t}, \mathbf{s}_{t-h:t}) \tag{1} $$

$$ \pi (l, \mathbf{o}_{t-h:t}, \mathbf{s}_{t-h:t}) \to \mathbf{o}_{t+1}, \mathbf{a}_{t:t+k} \tag{2} $$

*   **$\pi$**: 代表 robot policy network，即 GR-2 模型本身。
*   **$l$**: Language instruction，比如 "pick up the yellow mustard bottle"。
*   **$\mathbf{o}_{t-h:t}$**: Observation history。$\mathbf{o}$ 是 image observation，下标 $t-h:t$ 表示从时间步 $t-h$ 到 $t$ 的图像序列，$h$ 是 history length。
*   **$\mathbf{s}_{t-h:t}$**: Robot state history。$\mathbf{s}$ 包含 end-effector 的 position, rotation 以及 binary gripper state (开/关)。
*   **$\mathbf{a}_{t:t+k}$**: Action trajectory。$\mathbf{a}$ 是 action，下标 $t:t+k$ 表示从当前时刻 $t$ 预测到未来 $t+k$ 时刻的轨迹，$k$ 是 action trajectory length。
*   **$\mathbf{o}_{t+1}$**: 在公式 (2) 中，模型不仅输出 action，还输出 future image token $\mathbf{o}_{t+1}$。这是 joint training 的关键。

**Tokenization 策略：**
为了把这个连续的 control 问题变成 GPT 能处理的 discrete sequence，GR-2 做了如下处理：
*   **Text**: 使用 frozen CLIP text encoder 进行 tokenize。
*   **Image**: 使用 frozen VQGAN 将每帧 image 转换为 discrete tokens。VQGAN 的 codebook 会在 internet data 和 in-domain robot data 上联合训练，这保证了 visual representation 的通用性。
*   **Robot State**: 通过 trainable linear layers 编码。
*   **Action Trajectory**: 使用 conditional VAE (cVAE) 生成。这点非常关键，action chunking 通过 cVAE 生成，能够处理多模态的 action distribution，同时保证 trajectory 的平滑性，这对 real-time deployment 至关重要。

### 2. Pre-training 阶段的 Intuition

GR-2 的 pre-training 数据达到了惊人的 38 million video clips，超过 50 billion tokens，涵盖了 Howto100M, Ego4D, Something-Something V2 等数据集。

这个阶段的目标极其纯粹：给定 text 和当前 frame，autoregressive 地预测 future frames。

**Intuition 构建**：为什么 video generation pre-training 对 robot manipulation 有效？
当你训练一个巨大的 GPT 去预测 internet 视频的下一帧时，它必须学习到物体之间的物理交互规律、手如何抓取物体、东西如何掉落等 dynamics。这个 learned world dynamics 是跨越 modality 的。当 GR-2 之后在 robot data 上 fine-tune 时，它不需要从零学习 "杯子被打翻水会流出" 这种 commonsense，它只是需要学习如何把它内部映射的未来视觉世界，通过 Cartesian action 轨迹 "画" 到现实世界中。

### 3. Fine-tuning 阶段的多视角与数据增强

在 fine-tuning 时，GR-2 优雅地处理了 multi-view 输入。Real robot 通常有多个相机（static head camera + end-effector camera）。GR-2 直接将多视角的 image tokens 和 robot state tokens 拼接进 GPT 的 context window，同时输出多个视角的 future frames 和 action trajectory。

为了提升 generalization，他们用了一种非常聪明的 data augmentation 策略：
*   **Object Insertion**: 训练一个 diffusion model，结合自收集数据集和 Open Images，将特定物体 inpaint 到场景中。
*   **Background Changing**: 使用 SAM (Segment Anything Model) 分割出背景，然后使用 video generation model (Latte) 条件化原视频生成新背景的视频，同时保持 robot motion 不变。

这种基于 generative models 的 data augmentation 直接解决了 robot learning 中 data scarcity 和 scene diversity 不足的痛点。

### 4. Real-Robot 部署：Whole-Body Control (WBC)

GR-2 在网络层面输出的是 Cartesian space 的 trajectory。但 Kinova Gen3 (7-DoF) 需要的是 joint torque 或者 joint position。Paper 引入了 WBC (Whole-Body Control) 算法来作为桥梁。

WBC 算法的核心是 trajectory optimization。它将 GR-2 生成的 Cartesian trajectory 进行平滑性和连续性优化，然后将其转换为 low-level joint actions，并在 200Hz 的频率下执行。优化框架中集成了 collision constraints 和 manipulability 指标。这意味着高层的 policy 关注 "做什么" (semantic 和 high-level planning)，底层的 WBC 关注 "怎么做" (kinematics 和 dynamics)，这种分层架构能最大化利用 200Hz 的控制频率来抵抗 disturbances。

### 5. 实验数据与 Scaling Law

实验结果非常亮眼：
*   **Multi-task Learning**: 在 105 个 tasks (涵盖 picking, placing, uncapping 等 8 种 skills) 上，GR-2 在 Simple setting 达到了 97.7% 的 success rate。即使在 Unseen Environments (换了厨房背景和干扰物)，也达到了 71.7% (使用 Data Aug 后达到 87.0%)。
*   **End-to-End Bin Picking**: 在 122 个物体 (55 seen, 67 unseen) 上测试，GR-2 将 average success rate 从 GR-1 的 33.3% 直接拉升到 79.0%。对于 transparent, deformable, reflective 物体也表现优异，这暗示了 VQGAN 强大的 visual representation 能力。
*   **Scaling**: Paper 测试了 30M, 95M, 312M, 719M 四个规模。结果显示，随着模型增大，video prediction 的 validation loss 线性下降，同时 real-robot 的 success rate 也随之上升。这证明了 GR-2 架构的 scalability，表明 more parameters + more video data = better robot policy。

### 6. 广度联想与技术对比

从你的视角来看，GR-2 可以被视作多个前沿思想的集大成者：

1.  **Video Pre-Training (VPT) 的延续与突破**: OpenAI 的 VPT 在 Minecraft 里证明了看无标签视频可以学 policy。GR-2 则在真实的 continuous control 领域证明了，internet-scale 的 human activity video 可以直接 transfer 到 robot manipulation。
2.  **Sora 作为 World Simulator 的具身化**: Sora 证明了 DiT 能生成高质量的物理世界 video。GR-2 用 GPT-style (而非 Diffusion) 做同样的事，但更进一步，它把生成的 video 和 action 绑定。Paper 中提到 "predicted action is trying to replay the trajectory in the predicted video"，这是一个极度优美的直觉。Future image prediction 成为了 action 的隐式 planner。
3.  **Action Chunking 与 cVAE**: 使用 single-step action prediction 容易导致 jittering 且无法处理 multi-modality。GR-2 引入 ACT (Action Chunking with Transformers) 的思想，用 cVAE 预测 $t:t+k$ 的轨迹，这让 policy 具备了 anticipation 能力。
4.  **与 RT-2 / OpenVLA 的对比**: RT-2 和 OpenVLA 依赖 large pre-trained VLMs (如 PaLI 或 LLaVA)，它们把 robot action 变成 text tokens 吐出来。GR-2 没有走 VLM->Action Token 的路线，而是走了 Video Generation -> Action 的路线。前者更依赖语言逻辑，后者更依赖视觉物理 dynamics。对于 manipulation 这种重空间和物理的任务，GR-2 的路线直觉上更合理。

总结来说，GR-2 的成功在于它极致地践行了 "Pre-train a world model, then act" 的理念。通过 50 billion tokens 的 video 预训练，它把 commonsense 刻进了 weights；然后通过几千条轨迹的 fine-tuning，它学会了如何将自己想象出的未来用机械臂演绎出来。200Hz 的 WBC 保证了这种演绎足够丝滑和鲁棒。

### Reference Links

*   **GR-2 Project Page**: [https://gr2-manipulation.github.io](https://gr2-manipulation.github.io)
*   **GR-1 Paper (Predecessor)**: [Unleashing Large-Scale Video Generative Pre-Training for Visual Robot Manipulation](https://arxiv.org/abs/2312.13139)
*   **Sora (Video Generation as World Simulators)**: [Video generation models as world simulators](https://openai.com/research/video-generation-models-as-world-simulators)
*   **VPT (Video Pre-Training in Minecraft)**: [VPT: Learning to Act by Watching Unlabeled Online Videos](https://openai.com/index/vpt/)
*   **VQGAN (Image Tokenizer)**: [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841)
*   **ACT (Action Chunking using cVAE)**: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
*   **SAM (Segment Anything)**: [Segment Anything](https://arxiv.org/abs/2304.02643)
*   **RT-2 (Vision-Language-Action Models)**: [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)
*   **Latte (Video Generation Model for Data Aug)**: [Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/abs/2401.03048)
