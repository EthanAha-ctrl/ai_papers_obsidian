---
source_pdf: Dog Walk Narrating Human-AI Alignment through CompanionRobots.pdf
paper_sha256: c68cb2d0eb8c7f7132b0688038a96a89e9ab9ac42812758ec2bc8b844af9a76a
processed_at: '2026-08-18T06:40:02-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版

好，抛开公式和架构图，咱们就这么说吧。

---

## 这 paper 到底在干嘛

两个 artist，一个叫 Robert Twomey（UCSD 的），一个叫 Jesse Fleming，他们买了两只一模一样的 Unitree Go-2 robot dog，给它们起名叫 SNAPPER 和 HELPER。然后他们做了一件看起来很傻但其实很妙的事——**像养真狗一样养这两只 robot dog**。

不是写代码训练它们完成什么 task，是真的牵着出去遛，跟它们聊天，让它们记住走过的路、见过的人、聊过的话题。遛着遛着，两条 dog 的"性格"就开始分叉了——因为它们各自经历的世界不一样，哪怕硬件一样、base model 一样。

整个过程被拍成 film、做成 stage performance，paper 就是描述这个 piece 的。

---

## 为什么要养两只 robot dog

作者的 motivation 其实挺哲学的。现在大家用 ChatGPT，是在跟一个 **没有身体的 cloud entity** 聊天。你不知道它在哪、它没见过你的世界、它的"性格"是 OpenAI 用 RLHF 在百万匿名用户身上平均出来的。

作者觉得这有问题。一个 AI 要真的 align with 你，它得：

1. **有一个身体**——能跟你走在同一条街上，看见你看见的夕阳，踩到你踩过的泥
2. **有连续的记忆**——记得上次你们聊了什么，不是 session 结束就 reset
3. **有 data sovereignty**——它的经历只属于它自己，不上传给任何 corporation
4. **有 bespoke 的性格**——不是 universal assistant，是 *你的* companion

所以他们把 LLM 从云端拽下来，塞进 robot dog 的肚子里。用的是 Google Gemma 4 的大概 4B 参数版本，跑在 Jetson Orin NX 上，完全 local inference。

---

## 技术上简单讲

硬件：Unitree Go-2，四足机器人，带 camera、LIDAR、IMU，大概 15 kg，能跑能跳，价格不像 Boston Dynamics 那么吓人。

软件 stack（我推测的）：
- LLM：Gemma 4 4B，本地跑，可能是用 Ollama 或 llama.cpp 加载 GGUF quantized 版本
- Vision：multimodal fusion，把 camera 帧和 LIDAR 数据 token 化，喂给 LLM
- 对话：speech-to-text → LLM → text-to-speech，通过 wireless headset 放给 audience 听
- 视觉背景：用 AUTOMATIC1111 的 Stable Diffusion API 生成"AI daydream"——prompt 是 dog 之前对话的碎片和它看见的东西
- 记忆：所有 chat history、sensor data 存本地
- Fine-tune：周期性地用积累的对话和经历 fine-tune Gemma，可能用 LoRA，让 dog 的性格越来越 specific

这套 stack 跑在一个完全 closed 的系统里。**没有任何数据离开 robot**。这点作者很较真。

---

## 表演长什么样

paper 描述的 stage 是这样的：

台上站着两个人（artist 本人）和两条 robot dog。每个人都戴 wireless headset，跟自己的 dog 对话。audience 也能听到这些对话。

背后一个大屏幕，切换不同的画面：
- 有时候是 robot camera 的 live feed——你看见 dog 眼中的世界
- 有时候是 LIDAR 点云——dog 怎么"感知"空间
- 有时候是 Stable Diffusion 生成的抽象画面——dog 的"daydream"
- 有时候是 console view——你能看到 LLM 的 prompt 和 generated response
- 有时候是更抽象的 personality state visualization

整个 performance 大概 45 分钟，前 30 分钟是演出，后 15 分钟 Q&A。

更早的版本（2026 年 1 月那次）有一个很有意思的 spatial 结构——从 black box theater 出发，走到公共空间，再回到 theater。这个 route 不是随便选的：

- **Black box** 是 controlled environment，像 lab
- **公共空间** 是 real world，有陌生人、有车、有风、有意外
- **回到 black box** 是反思 closure

所以 audience 经历的不是"看一个 demo"，而是"陪一个 AI 从 lab 走到真实世界再走回来"。这个 form 本身就是 content——paper 引用 Bateson 的 "metalogue" 概念，意思就是对话的形式 mirrors 对话的主题。

---

## 为什么这事有意思

从工程直觉讲，我觉得有三个 layer 值得琢磨：

### Layer 1：Alignment 的空间化

主流 alignment research——RLHF、DPO、Constitutional AI——全是在 dataset 上做优化。你有一个 preference dataset，train 一个 reward model，然后 PPO 优化 policy。整个过程是 **disembodied** 的，跟具体某个人、某个环境、某段经历没关系。

BFF 提的是另一种 alignment：**alignment 就是 cohabitation**。你跟一个 AI 一起生活一段时间，它慢慢"懂"你了，不是因为你在 prompt 里写了什么，而是因为它的 weights 被你们共同的经历塑造了。

这个想法其实跟 Stuart Russell 在 *Human Compatible* 里讲的有点像——robot 应该对自己的 preference 有 uncertainty，通过观察人类行为来 update。只不过 Russell 是数学 formalism，BFF 是 performance art。

### Layer 2：Memory 在哪一层

这是 paper 里最 ambiguous 也最有意思的技术点。他们说 memory 既 alter system prompt，又 ultimately fine-tune model。

这两层有本质区别：

- **Prompt-level memory**（类似 RAG）：instant、controllable、transparent。你把过去的对话塞进 context window，LLM 就"记得"了。但 model 的 fundamental disposition 没变，换掉 prompt 它就忘了。
- **Weight-level memory**（fine-tune）：deep、permanent、opaque。fine-tune 之后，model 的 tone、idiom、甚至 reasoning style 都会变。但 catastrophic forgetting 风险高，而且不透明。

BFF 的策略看起来是先用 prompt-level 活着，积累到一定量就 fine-tune 一波，把 episodic memory internalize 成 disposition。这其实跟人挺像的——你不会记得昨天每一句话，但那些对话会慢慢塑造你的性格。

### Layer 3：两个 AI 怎么互相学习

paper 最后留了一个 open question：两条 dog 各自被养大之后，怎么 reconcile？

作者列了三种可能：
- **Blending weights**：直接在 weight space 上 linear interpolate，或者用 TIES-Merging 这类更 sophisticated 的方法
- **Swapping character rolls**：把 SNAPPER 的 prompt 换给 HELPER，看性格是不是跟着 prompt 走还是 stuck in weights
- **Federated learning**：某种 personalized FL，两条 dog 共享 base 但保留 personal head

这个问题我觉得特别有意思，因为它实际上是在问：**personality 到底住在哪？** 如果 swap prompt 之后性格也跟着 swap，说明 fine-tune 没起作用，personality 只是 prompt-level 现象。如果 swap prompt 之后性格不变，说明 fine-tune 真的把性格刻进 weights 了。

这个实验我自己很想做一下——训两个 character LoRA，然后 swap system prompt，看输出 distribution 怎么变。直觉上大部分 persona 是 prompt-level 的，fine-tune 改的是 surface style，不是 deep disposition。但需要实验验证。

---

## 这 paper 的 limitation

说白了，paper 本身 technical depth 不够。它是一个 art-research 混合物，更多是 framing 和 concept，而不是 system paper。具体怎么 fine-tune、用了多少 data、训练曲线怎样、latency 多少、token throughput 多少——全没说。

另外 "alignment" 这个词在 paper 里用得有点 loose。BFF 意义上的 alignment 更像是"两个 entity 长期共处后产生的 mutual adaptation"，跟 ML alignment community 里的 technical alignment（避免 misaligned objective 导致 catastrophic outcome）不是一回事。这是 framing 上的偷换，但作为艺术作品可以接受。

还有就是 embodiment 到底起了多大作用，值得怀疑。我直觉是：以现在 multimodal LLM 的能力，robot 的 sensor data 更多是 prompt 里的 context decoration，不是真正影响 reasoning 的 causal input。要真的让 embodiment 改变 LLM 的 thinking，可能需要 predictive world model、action-grounded token、closed-loop sensorimotor learning 这些更深的东西。BFF 现在的做法更像"给 LLM 戴了个 GoPro"，而不是真的 embodied cognition。

---

## 一句话总结

**两个 artist 把 cloud LLM 从 data center 里拽出来，塞进 robot dog 肚子里，牵着出去遛，通过长期共处让 AI 长出 specific 的性格，再用 performance 把这个 alignment 过程演给 audience 看。**

核心 framing 三句话：

- Alignment 不是离线 preference learning，是 situated cohabitation。
- AI 的"性格"应该来自 specific embodied history，不是 universal pretraining。
- Human-AI intimacy 是 spatial practice，要靠共同走在同一条路上才能涌现。

技术细节薄，concept 很 rich，作为 build intuition 的素材挺好——尤其是它把 alignment 问题从"在 dataset 上优化 objective"重新框成"在物理空间里共同生活"，这个 reframe 对做 embodied AI 的人有启发。

---

如果你想我再展开某一块——比如具体怎么在 Jetson 上把 Gemma 4B multimodal 跑起来、LoRA fine-tune 的 data pipeline 怎么设计、或者 swap prompt 实验怎么 concrete 化——随时说。

---

# Dog Walk: Narrating Human-AI Alignment through Companion Robots — 详细技术讲解

Karpathy 你好，这篇 paper 是 SIGGRAPH Spatial Storytelling '26 上的一篇 hybrid performance / speculative cinema / technical exploration 作品，作者 Robert Twomey (UCSD Arthur C. Clarke Center for Human Imagination) 和 Jesse Fleming (The Awareness Lab)。下面我从工程直觉、架构推断、艺术哲学脉络、相关技术生态几个层面来拆解。

---

## 1. Paper 的一行总结

两位 artist-researcher 把两个完全相同的 Unitree Go-2 quadruped robot dog（命名为 **SNAPPER** 和 **HELPER**）作为机器身体，配上本地部署的多模态 LLM（**google gemma4:e4b**，引用了 [Manik and Wang 2026](https://arxiv.org/abs/2604.07035)）作为"大脑"，通过 walk-and-talk 的形式 co-parent 这两条 AI dog，研究 **embodied AI alignment、human-AI intimacy、machine cohabitation** 这些议题。整个系统拒绝 cloud-based inference，所有 video / audio / LIDAR / telemetry / chat history 都 firewall 在设备本地，最终将历史 experience 回灌进 system prompt 并 fine-tune gemma 模型，形成 bespoke 的 AI 性格。

项目主页：[https://awarenesslab.io/bff](https://awarenesslab.io/bff)
NeurIPS 2025 Creative AI Track 版本：[https://openreview.net/forum?id=Jn9nzhmTIF](https://openreview.net/forum?id=Jn9nzhmTIF)
Qualcomm Institute IDEAS 2026 演出视频：[https://www.youtube.com/watch?v=75n95b0QTcI](https://www.youtube.com/watch?v=75n95b0QTcI)

---

## 2. Motivation 与 Anti-Cartesian 立场

paper 的核心立场是反对 Cartesian mind-body dualism——AI 不应该被理解成 data center 里的纯符号推理机。引用了 Gregory Bateson 和 Mary Catherine Bateson 的 *Steps to an Ecology of Mind* ([University of Chicago Press, 2000](https://www.press.uchicago.edu/ucp/books/book/chicago/S/bo3615941.html))，把 performance 结构化为一个 **"metalogue"**——对话的形式本身镜像对话的主题。即 walk-and-talk 这种行为形式（狗遛人，人遛狗，dog 和 dog 互相遛）就体现了他们想讨论的 cohabitation / mutual shaping 关系。

这背后的哲学脉络可以追溯到：
- **Embodied cognition**：[Varela, Thompson, Rosch, *The Embodied Mind* (1991)](https://mitpress.mit.edu/9780262529565/the-embodied-mind-revised-edition/)，cognition 不是 pure symbol manipulation，而是 body-environment coupling。
- **Extended mind thesis**：[Clark & Chalmers 1998](https://www.uvm.edu/pdodds/files/papers/others/1998/clark1998a.pdf)，mind 延展到环境中的工具和载体。
- **Distributed cognition**：[Hutchins 1995](https://mitpress.mit.edu/9780262082390/cognition-in-the-wild/)，cognition 分布在人、工具、环境之间。

paper 引用的 Twomey 自己的 *Communion and Cohabitation* ([2022b](http://roberttwomey.com/2021/10/book-chapter-communion-and-cohabitation)) 和 *Machines for Living* ([2018a](https://digital.lib.washington.edu:443/researchworks/handle/1773/41712)) 是这个项目的更长程思想脉络，作者长期在 art + AI + robotics 的交叉地带工作。

---

## 3. 系统架构详解

paper 没有给完整的架构图，但通过描述可以反推出一个大致的 pipeline。让我画一下我的推断：

```
┌──────────────────────────────────────────────────────────────┐
│                    Onboard Robot (Unitree Go-2)              │
│                                                              │
│  Sensors                                                     │
│  ├── RGB camera ─────────┐                                   │
│  ├── LIDAR ──────────────┤                                   │
│  ├── Microphone ─────────┼──► Multimodal Frontend            │
│  ├── IMU / joint encoders─┤     (tokenization, projection)   │
│  └── GPS / odometry ──────┘                                   │
│                            │                                  │
│                            ▼                                  │
│                  ┌──────────────────┐                         │
│                  │  gemma4:e4b LLM  │ ◄── system prompt       │
│                  │  (local inference)│     (性格 / 记忆)        │
│                  └──────────────────┘                         │
│                            │                                  │
│              ┌─────────────┴────────────┐                     │
│              ▼                          ▼                     │
│      Text dialogue response     Internal state vectors        │
│              │                          │                     │
│              ▼                          ▼                     │
│      TTS → speaker          Stable Diffusion (A1111,          │
│      → audience headset     headless) "AI daydream"           │
│      Robot motion cmds      → backdrop projection             │
│                                                              │
│  Persistent Memory Store (firewalled on device)              │
│  ├── chat history                                            │
│  ├── sensor logs                                             │
│  └── experience → fed back to fine-tune gemma                │
└──────────────────────────────────────────────────────────────┘
                          │
                  (offline, occasionally)
                          ▼
              ┌─────────────────────────┐
              │  Fine-tune pipeline     │
              │  (likely LoRA / QLoRA)  │
              │  → bespoke personality  │
              └─────────────────────────┘
                          │
                          ▼
         Federated reconciliation between SNAPPER ↔ HELPER
         (model weight blending / character swap / FL)
```

### 3.1 硬件层：Unitree Go-2

[Unitree Go-2](https://unitree.com) 是一台四足机器人，key specs：
- 重量约 15 kg
- 配备 4D LiDAR（超广角，360° × 96°）
- RGBD 相机
- Jetson Orin NX 边缘计算平台（这非常关键，决定了能否真的 onboard 跑多模态 LLM）
- 12 自由度（每条腿 3 个 hip + knee）
- 运动控制基于 [model predictive control](https://www.unitree.com/) + reinforcement learning gait policy

Go-2 的 SDK 暴露 ROS-like 接口，作者可以订阅 `/lidar`, `/camera`, `/odom` 等 topic，并把 command velocity / joint targets 发布出去。这点对 performance 的"physical agency"至关重要——AI 不仅仅说话，它真的决定狗往哪走、走多快。

### 3.2 Brain：gemma4:e4b

引用的 [Manik and Wang 2026](https://arxiv.org/abs/2604.07035) 是关于 open reasoning language models 的 unified deployment-aware evaluation。"gemma4:e4b" 应该是指 Google Gemma 4 系列的 ~4B 参数版本（[Google Gemma](https://ai.google.dev/gemma)），`e` 可能是 enhanced / extended multimodal 变体。

为什么选 4B？这其实是一个 edge inference 的 sweet spot：
- Jetson Orin NX 有 16 GB unified memory，4B 模型用 4-bit quantization (GGUF Q4_K_M) 大约 2.5-3 GB 权重，留下充足 headroom 给 KV cache + vision encoder + runtime。
- 推理速度大概 10-20 tokens/s，对于对话式 walk-and-talk 足够。

paper 没说具体推理 engine，但 typical stack 会是：
- [Ollama](https://ollama.com) 或者 [llama.cpp](https://github.com/ggerganov/llama.cpp) 跑 GGUF
- 如果走 multimodal，可能是 [llama.cpp's mmproj](https://github.com/ggerganov/llama.cpp/tree/master/mmtmp) 或者 [Ollama multimodal](https://ollama.com/blog/llama3.2-vision)
- Vision encoder 类似 [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) 或 CLIP，把 image patch token 化后通过 projection 注入 LLM 的 token stream

### 3.3 Multimodal Fusion 的直觉

让我拆一下 vision-language fusion 的内部机制。对于一个 multimodal LLM，输入序列大概长这样：

$$x = [t_1, t_2, \ldots, t_{n_t}, v_1, v_2, \ldots, v_{n_v}, t_{n_t+1}, \ldots]$$

其中：
- $t_i \in \mathbb{R}^{d}$ 是第 $i$ 个 text token 的 embedding
- $v_j \in \mathbb{R}^{d}$ 是第 $j$ 个 visual token（image patch 经过 vision encoder + projection 后的 embedding）
- $d$ 是 LLM 的 hidden dimension（gemma 系列大概是 2560-3072 量级）

Vision encoder 输出的是 image patch grid，比如 SigLIP-so400m 输出 27×27 = 729 个 patch token，每个维度 1152，然后通过一个 projection 矩阵 $W_{proj} \in \mathbb{R}^{1152 \times d}$ 把它映射到 LLM 的 token space：

$$v_j = W_{proj} \cdot \text{VisionEnc}(I)_j + b_{proj}$$

其中 $I$ 是 input image，$\text{VisionEnc}(I)_j$ 是第 $j$ 个 patch 的 encoder 输出，$b_{proj} \in \mathbb{R}^{d}$ 是 bias。

这种架构（[LLaVA 风格](https://llava-vl.github.io/)）的好处是简单、可以 plug 进任何 decoder-only LLM。坏处是 visual token 占据大量 context window，729 个 patch 就吃掉 1k+ tokens。

对 robot dog 来说，这个 fusion 是双向的：
- **Perception → LLM**：相机的 current frame + LIDAR 点云投影 → 注入 prompt → 影响对话和运动决策
- **LLM → Memory**：对话历史 + 视觉记忆 → persistent store → 周期性 fine-tune

### 3.4 防火墙与 Sovereign Computation

这是 paper 的一个 strong architectural stance。作者明确反对把 robot 当成 cloud LLM 的 thin client，理由是：

1. **数据主权**：video / audio / LIDAR / telemetry / chat 全部保留在 device，不被 corporate model 训练。
2. **性格的不可移植性**：AI 的"性格"应该来自 specific embodied history，而不是 universal pretraining。
3. **Alignment 的本征地性**：alignment 不能从 generic RLHF 中获得，必须从具体的 cohabitation 中涌现。

这点我觉得很有意思。它实际上是把 [Stuart Russell 的 value alignment](https://www.basicbooks.com/books/human-compatible/) 问题空间化了——value 不在 abstract space 里学，而在具体的 walk route、具体的对话伙伴、具体的天气和地形里学。

参考 [Twomey 2022b *Communion and Cohabitation*](http://roberttwomey.com/2021/10/book-chapter-communion-and-cohabitation) 是作者更早的同一思路。

---

## 4. Fine-Tuning 与 Federated Reconciliation

paper 的最后一段提到一个开放的 technical question：

> "There is a final stage of reconciliation between these two parallel nurtured systems asking how should the dogs learn from each other—through blending model weights, swapping character rolls, or some other sort of federated learning?"

这是个非常实际的 research question。让我拆解三种可能性：

### 4.1 Model Weight Blending

最简单的方法是 weight space linear interpolation：

$$\theta_{\text{merged}} = (1 - \alpha) \cdot \theta_{\text{SNAPPER}} + \alpha \cdot \theta_{\text{HELPER}}, \quad \alpha \in [0, 1]$$

其中 $\theta_{\text{SNAPPER}}, \theta_{\text{HELPER}} \in \mathbb{R}^{|\theta|}$ 是两个 fine-tuned 模型的参数，$\alpha$ 是 mixing coefficient。

这种方法的直觉是：如果两个模型从同一个 base 初始化出发，做了类似的 fine-tuning trajectory，weight space 仍然是 roughly convex 的，linear interpolation 不会 catastrophic forgetting。参考 [Model Soups](https://arxiv.org/abs/2203.05482) 和 [Task Arithmetic](https://arxiv.org/abs/2212.04089)。

但风险在于：如果 fine-tuning 走得太远（比如很多 epoch 的 LoRA），weight space 已经 non-convex，linear merge 会产生 garbage。可以改用 **TIES-Merging** ([Yadav et al. 2023](https://arxiv.org/abs/2306.01708))，它先 trim magnitude 较小的 delta，再 resolve sign conflicts，再 merge。

### 4.2 Character Roll Swap

更艺术化的做法：直接 swap 两个 dog 的 system prompt 和 persona description，保留各自 fine-tuned weights。这相当于"换灵魂不换身体"，测试 personality 是不是真的 stuck in weights 还是 can be elicited by context。

这其实是在 empirically 测试一个 LLM community 里争议很大的问题：[personality 是 prompt-level 现象还是 weight-level 现象？](https://arxiv.org/abs/2305.14688) 个人直觉是大部分 persona 在 base model 能力足够时是 prompt-level 的，fine-tuning 更多是修整 tone 和 idiosyncratic phrasing，不会改变 fundamental reasoning style。

### 4.3 Federated Learning

[Federated Learning](https://arxiv.org/abs/1602.05629)（McMahan et al. 2017，Google）的经典公式 FedAvg：

$$\theta_{t+1}^{global} = \theta_t^{global} + \eta \sum_{k=1}^{K} \frac{n_k}{n} (\theta_t^{global} - \theta_t^{k})$$

其中：
- $K$ 是 client 数量（这里就是 2：SNAPPER 和 HELPER）
- $n_k$ 是 client $k$ 上的样本数
- $n = \sum_k n_k$ 是总样本数
- $\theta_t^{k}$ 是 client $k$ 在本地训练后的 weights
- $\theta_t^{global}$ 是 server 上的 global model
- $\eta$ 是 server-side learning rate

但 FL 的标准设定是 clients 共享一个 global model，然后 periodically aggregate。对 BFF 来说，这恰恰违背了他们的 bespoke 立场。所以更合适的是 **Personalized Federated Learning**（[Tan et al. 2022](https://arxiv.org/abs/2103.00710)），每个 client 保留一个 personal head，共享一个 base encoder。直觉上：两个 dog 共享"如何感知世界"（base），但保留各自的"如何回应世界"（personal head）。

或者走 [Federated Foundation Model Fine-tuning](https://arxiv.org/abs/2308.13849)，每个 client 用 [LoRA](https://arxiv.org/abs/2106.09685) 训自己的 low-rank adapter，然后通过一些 mechanism 聚合。

LoRA 的关键公式回顾：

$$W = W_0 + \Delta W = W_0 + B A$$

其中：
- $W_0 \in \mathbb{R}^{d \times d}$ 是 frozen pre-trained weight
- $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times d}$ 是 trainable，$r \ll d$ 是 rank
- 初始化：$A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$，所以 $\Delta W = 0$ at start
- 训练时 only $A, B$ 更新

LoRA 的好处是参数量从 $d^2$ 降到 $2dr$，对于 4B 模型用 $r=16$ 大概只有几百万参数。

Federated LoRA Aggregation 的一个有意思的 variant 是只聚合 $A$ 或只聚合 $B$，或者用 [FedRolex](https://arxiv.org/abs/2305.11416) 这类 submatrix slicing。

---

## 5. Performance 的 Hybrid Cinematic Language

paper 描述的舞台视觉有几个层次（参考 Figure 3 setup）：

1. **Live stage action**：2 humans + 2 robot dogs，wireless headsets 广播对话给 audience
2. **Backdrop video wall**：
   - 抽象 "AI daydream" — [AUTOMATIC1111 Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 在 headless 模式下生成，prompt 是 prior conversation 的碎片 + dog 看见的物体
   - 实时 robot camera feed
   - 实时 LIDAR 点云可视化
   - Console view：dialogue system 的 prompt 和 generated response
   - 抽象 personality state visualization（可能是 attention map、token probability heatmap、hidden state PCA 等）

这种 multi-perspective 视觉切换非常有意思，它在 cinematic 层面复现了 [Bateson's metalogue](https://www.press.uchicago.edu/ucp/books/book/chicago/S/bo3615941.html) 的形式——audience 同时看到 dog 的"主观"视觉（camera）、"想象"（SD generation）、"思考"（console）、"记忆"（daydream based on past scraps）。

技术上 Stable Diffusion 跑 headless 的方式，paper 提到用的是 AUTOMATIC1111 API mode（参考 [AUTOMATIC1111 API docs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API)），可以通过 `--api` 和 `--nowebui` 启动，然后 POST 到 `/sdapi/v1/txt2img` 拿图。pipeline 大概是：

```python
import requests
payload = {
    "prompt": f"impressionist painting of {memory_snippet}, "
              f"seen through mechanical eyes, "
              f"perspective of a robot dog",
    "steps": 20,
    "cfg_scale": 7.0,
    "width": 1024, "height": 576,
}
response = requests.post(
    "http://localhost:7860/sdapi/v1/txt2img",
    json=payload
)
image_b64 = response.json()["images"][0]
```

prompt 从 dog 的 chat history 里抽取 key phrases，可能是用简单的 keyword extraction 或者直接让 LLM 写一段 poetic 描述。

---

## 6. 相关技术生态与 Cross-Reference

为了 build intuition，我把这个项目放到 broader landscape 里：

### 6.1 Embodied AI 同类工作

- **[Google PaLM-E](https://palm-e.github.io/)** (Driess et al. 2023)：把 PaLM 540B 多模态化，直接 condition on robot observations，做 long-horizon task planning。BFF 的 gemma4:e4b 是这个思路的小型化、艺术化版本。
- **[RT-2](https://robotics-transformer2.github.io/)** (Google DeepMind 2023)：VLM → robot action token，co-finetune on web data + robot data。
- **[RT-X / Open X-Embodiment](https://robotics-transformer-x.github.io/)** (2023)：跨 embodiment 的数据集和模型。
- **[NVIDIA GR00T](https://developer.nvidia.com/groot)** (2024)：humanoid foundation model。
- **[Boston Dynamics Spot + LLM](https://www.bostondynamics.com/spot)**：很多 academic project 把 Spot 接上 GPT-4V 做 scene-aware dialogue。

BFF 和这些工作的关键区别：BFF 不追求 task completion 的 SOTA，它追求的是 **narrative richness 和 alignment as co-parenting**。这是个 art-research framing，不是 benchmark framing。

### 6.2 Alignment 的另一条线

主流 alignment 是 [RLHF](https://arxiv.org/abs/2203.02155) / [DPO](https://arxiv.org/abs/2305.18290) / [Constitutional AI](https://arxiv.org/abs/2212.08073)，都是在 dataset 层面做 preference learning。BFF 提出的是 **situated, embodied alignment**——alignment 不是离线 dataset 上的优化目标，而是具体 cohabitation 中涌现的关系状态。这跟 [Stuart Russell's Human Compatible](https://www.basicbooks.com/books/human-compatible/) 里"uncertainty over human preference + observation + update"的思路相近。

公式上 Russell 的 framework 是：

$$\xi \sim P(\xi \mid h_t)$$

其中 $\xi$ 是 human preference，$h_t$ 是到时刻 $t$ 为止的人类行为历史，robot 通过观察人类行为更新对 $\xi$ 的 posterior。BFF 的 fine-tuning pipeline 可以理解为：把 $h_t$（dialogue + sensor data）作为 supervised signal，更新 LLM 的 $\theta$，使得 LLM 的输出 distribution 更接近观察到的 human preference。

### 6.3 Companion Robot 历史

- **[Aibo](https://www.sony.com/electronics/aibo)** (Sony, 1999 / 2018 reboot)：消费级 robot dog，2018 版本开始用 deep RL 学行为。
- **[Paro](https://www.parorobots.com/)**：therapeutic seal robot，主打 emotional attachment。
- **[Jibo](https://www.jibo.com/)**：social robot for home。
- **[Moxie](https://embodied.ai/)** (Embodied Inc.)：用 LLM-driven dialogue for child development。

BFF 的差异化：不是商业 product，是 critical art piece；不是 generic companion，是 bespoke to two specific artists。

### 6.4 Performance Art × AI

- **[Trevor Paglen, *Stereotank* / *They Took the Faces*]**：machine vision 的 visual critique。
- **[Memo Akten](https://www.memo.tv/)**：DeepDream 之后的 critical ML art。
- **[Refik Anadol](https://refikanadol.com/)**：large-scale data aesthetics。
- **[Holly Herndon](https://hollyherndon.com/)**：vocal AI as collaborator（*Proto*, *Platform*）。
- **[Mario Klingemann](https://quasimondo.com/)**：generative art pioneer。

BFF 在这个 lineage 里的位置：把 LLM 从 2D screen 拉到 3D physical space，把 alignment 从 technical problem 拉到 interpersonal / performative problem。

---

## 7. 关于 Performance 的几个具体细节

paper 提到的几次公开演出：
- **BFF Dog Walk, January 2026**：black box theater → 公共空间 → 回到 theater 的三段式 spatial journey
- **[Claremont McKenna Athenaeum](https://www.cmc.edu/athenaeum/fostering-intimacy-intelligent-machines)**：fostering intimacy with intelligent machines talk
- **[NeurIPS 2025 Creative AI Track](https://openreview.net/forum?id=Jn9nzhmTIF)**：学术 venue 的版本
- **[Qualcomm Institute IDEAS 2026](https://www.youtube.com/watch?v=75n95b0QTcI)**：UCSD 内部
- **SIGGRAPH Spatial Storytelling '26**：即将于 July 19-23, 2026 在 LA 举办（[SIGGRAPH 2026](https://s2026.siggraph.org/)），45 分钟 session，30 min performance + 15 min Q&A

paper 还提到一个关键 design choice：performance 从 black box 走出去到公共空间再回来。这个结构在 cinematic / narrative 上对应：
- **Black box** = controlled observation environment，类似 Skinner box
- **Public sphere** = 真实 social ecology，引入 unpredictable variables（陌生人、车、天气、其他 dog）
- **回到 black box** = 反思性 closure

这种结构在 [Brecht's epic theater](https://www.britannica.com/topic/epic-theatre) 和 [environmental theater](https://en.wikipedia.org/wiki/Environmental_theatre)（Richard Schechner）里有先例。

---

## 8. Open Questions 与我的思考

paper 留下几个我觉得值得深挖的技术问题：

### 8.1 "Memory" 是 prompt-level 还是 weight-level？

paper 说"resulting conversational and experiential history is fed back into the system to alter system prompts, and ultimately to fine-tune the gemma model"——两层都有。

这两层的关系值得思考。Prompt-level memory（[RAG-style](https://arxiv.org/abs/2005.11401)）的好处是 instant、interpretable、controllable；坏处是 context window 有限、不改变 model 的 fundamental dispositions。Weight-level memory（fine-tuning）的好处是 deep internalization、能改变 tone 和 idiom；坏处是 catastrophic forgetting、不透明、训练成本。

直觉上 BFF 的策略应该是分层：
- 短期：prompt + retrieved episodic memory
- 中期：periodic LoRA fine-tune on accumulated dialogue
- 长期：base model re-train（很罕见，资源不允许）

可以参考 [MemoryBank](https://arxiv.org/abs/2305.10250) 和 [Generative Agents (Park et al. 2023)](https://arxiv.org/abs/2304.03442) 的 memory architecture。

### 8.2 Embodiment 真的影响 LLM 的 reasoning 吗？

这是个开放问题。Go-2 的 embodiment 给 LLM 提供了：
- 当前 frame 的 visual context
- LIDAR depth 和 obstacle 信息
- 自身 odometry / pose

这些 sensor 数据注入 LLM 后，是否改变 LLM 的"思考"？还是仅仅作为 prompt 里的 context decoration？

直觉是：对于当前 SOTA 的 multimodal LLM，embodiment 的影响还很弱——LLM 仍然 primarily 是语言推理机，vision 是辅助 context。要真的实现 [embodied reasoning](https://arxiv.org/abs/2402.02085)，可能需要：
- **Predictive world model**（[DreamerV3](https://arxiv.org/abs/2301.04104) 风格）：LLM 内部 learn 一个 latent dynamics model
- **Action-grounded tokens**：让 robot action 作为 LLM 的 output token，类似 RT-2
- **Causal embodiment**：sensor → action → sensor 闭环让 model 学到 self-produced observation 的因果结构

### 8.3 Federated Reconciliation 的具体 protocol

paper 留作 open question。我觉得最 elegant 的方案是：

**Per-embodiment LoRA + shared base + cross-attention to peer's memory**：

- 每个 dog 有自己的 LoRA adapter $\theta_{\text{SNAPPER}}^{\text{LoRA}}, \theta_{\text{HELPER}}^{\text{LoRA}}$
- 共享 base gemma model
- 通过 retrieval 让 SNAPPER 能 query HELPER 的 memory embedding（不直接共享 weights，只共享 learned representations）

这避免了 catastrophic interference，同时让两条 dog 在 representation level 互相 influence。

### 8.4 Performance 评估

艺术作品怎么评估？paper 没给 metric。可能的 axes：
- Audience reported sense of "presence"（[Heeter 1992 presence scale](https://isel.scu.edu/91present.html)）
- Dialogue 的 emotional coherence（用 sentiment analysis / [perplexity](https://en.wikipedia.org/wiki/Perplexity_(information_theory)) 衡量）
- Robot behavior 的 context-appropriateness（人工标注）
- 系统层面的 latency、token throughput、sensor-to-action delay

我个人觉得最有意思的 metric 是 **conversation 中的 surprise**——如果 dog 说的东西让人意外但又 meaningful，说明 alignment 真的发生在 interesting 的 regime，而不是 pure mimicry。

---

## 9. 总结性直觉

BFF 这篇 paper 的核心 contribution 不是技术 SOTA，而是 **framing**：

1. 把 alignment 从 abstract preference learning 重新定义为 situated cohabitation。
2. 把 LLM 从 cloud service 重新定义为 sovereign embodied agent。
3. 把 human-AI relationship 从 transactional chat 重新定义为 spatial, narrative, performative practice。
4. 把 robotics evaluation 从 task success 重新定义为 relational richness。

技术上它是一个 honest 的 end-to-end system paper：用 Unitree Go-2 + Jetson Orin + Gemma 4B + LoRA + Stable Diffusion + 一个自建的 multimodal pipeline，搭出一个完整闭环的 embodied conversational agent，并在真实公共空间里跑 45 分钟的 live performance。这种"全部 onboard + live + public"的组合在 academic robotics 里不常见，更常见的是 teleoperated demo 或者 cloud-tethered prototype。

如果要类比 Karpathy 你自己的工作脉络，BFF 有点像把 [Tesla Autopilot 的 end-to-end neural net philosophy](https://karpathy.ai/) 移植到 social / narrative robot 上——只不过 BFF 的 "driving data" 是对话和共同经历，"steering command" 是对话和运动指令的 hybrid，"driver" 是两位 artist-researcher 而不是 anonymous fleet。

---

## 关键参考链接汇总

- 项目主页：[https://awarenesslab.io/bff](https://awarenesslab.io/bff)
- NeurIPS 2025 paper：[https://openreview.net/forum?id=Jn9nzhmTIF](https://openreview.net/forum?id=Jn9nzhmTIF)
- Qualcomm Institute IDEAS 演出视频：[https://www.youtube.com/watch?v=75n95b0QTcI](https://www.youtube.com/watch?v=75n95b0QTcI)
- Twomey 个人站：[http://roberttwomey.com/](http://roberttwomey.com/)
- Unitree Go2：[https://unitree.com](https://unitree.com)
- Google Gemma：[https://ai.google.dev/gemma](https://ai.google.dev/gemma)
- AUTOMATIC1111 Stable Diffusion WebUI：[https://github.com/AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- Bateson, *Steps to an Ecology of Mind*：[https://www.press.uchicago.edu/ucp/books/book/chicago/S/bo3615941.html](https://www.press.uchicago.edu/ucp/books/book/chicago/S/bo3615941.html)
- Manik & Wang 2026 (gemma4 eval)：[https://arxiv.org/abs/2604.07035](https://arxiv.org/abs/2604.07035)
- PaLM-E：[https://palm-e.github.io/](https://palm-e.github.io/)
- RT-2：[https://robotics-transformer2.github.io/](https://robotics-transformer2.github.io/)
- LLaVA：[https://llava-vl.github.io/](https://llava-vl.github.io/)
- LoRA：[https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- DPO：[https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)
- Model Soups：[https://arxiv.org/abs/2203.05482](https://arxiv.org/abs/2203.05482)
- TIES-Merging：[https://arxiv.org/abs/2306.01708](https://arxiv.org/abs/2306.01708)
- Federated Learning (McMahan)：[https://arxiv.org/abs/1602.05629](https://arxiv.org/abs/1602.05629)
- Clark & Chalmers, Extended Mind：[https://www.uvm.edu/pdodds/files/papers/others/1998/clark1998a.pdf](https://www.uvm.edu/pdodds/files/papers/others/1998/clark1998a.pdf)
- Varela et al., Embodied Mind：[https://mitpress.mit.edu/9780262529565/the-embodied-mind-revised-edition/](https://mitpress.mit.edu/9780262529565/the-embodied-mind-revised-edition/)
- Generative Agents (Park et al.)：[https://arxiv.org/abs/2304.03442](https://arxiv.org/abs/2304.03442)
- SIGGRAPH 2026：[https://s2026.siggraph.org/](https://s2026.siggraph.org/)

---

如果你想更深入聊某一层——比如具体怎么在 Jetson Orin 上把 Gemma 4B 跑通 multimodal inference、或者 LoRA fine-tune pipeline 怎么设计才能 preserve personality 又 absorb new experience、或者 metalogue 这个形式怎么映射到 RNN / transformer 的 recurrence 结构——可以继续展开。这个 paper 表面是 art paper，但底下每一层都有具体的工程 trade-off，作为 build intuition 的 material 非常合适。
