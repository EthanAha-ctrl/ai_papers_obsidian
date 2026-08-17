---
source_pdf: MLA A Multisensory Language-Action Model for Multimodal Understanding
  and Forecasting in Robotic Manipulation.pdf
paper_sha256: 96547849ddfb62f30d5eec9e46e63562bd5d75ed1f1c941dc4cb658eaa6a8949
processed_at: '2026-08-05T19:04:20-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MLA 用人话讲

好，Karpathy，我换个频道，就当咱俩在 whiteboard 前面聊天，把这篇 paper 的核心 idea 啃干净。

---

## 一句话概括

**这篇 paper 干的事儿：把 LLM 自己变成一个"多感官大脑"，让它直接看 image、point cloud、tactile，还能脑补出未来会发生啥，最后输出 robot action。**

---

## 这帮人到底在解决什么问题？

你想想现在 VLA 的状况。大家都在用 SigLIP 这种 vision encoder 把 image 压成 tokens，喂给 LLM，LLM 再吐 action。这条 path 在 pick-and-place 这种简单任务上还行，但碰到 contact-rich 的任务——比如盖章、擦白板、用铲子铲鸡蛋——就拉胯了。

为啥拉胯？三个 reason：

**第一，2D image 不够用。** 你拿个 camera 拍张照，物体离 camera 多远、3D shape 啥样、gripper 夹住没夹住，这些信息 2D pixel 根本 capture 不全。Robotics 是 spatial-physical world，你缺 depth、缺 geometry、缺 contact feedback，action 就不准。

**第二，加新 modality 就得加新 encoder，很烦。** 你想用 point cloud？好，加个 PointNet++ encoder。想用 tactile？再加个 MLP encoder。这些 encoder 都是在 internet data 上 pretrain 的，根本没见过 robot 场景。结果就是 encoder 输出的 representation 跟 LLM 的 embedding space 对不齐，LLM 看了一堆 token 但"读不懂"。而且 inference 时多个 encoder 跑起来慢。

**第三，现有 VLA 只会"看当下"，不会"想未来"。** Robot 要 interact with dynamic environment，你得知道"我下一个动作会导致物体怎么动"。现有的一些 future prediction 工作（如 [DreamVLA](https://arxiv.org/abs/2507.04447)）只预测 future image，但 future point cloud 呢？future tactile 呢？没人做。

MLA 就是在这三个痛点上同时发力。

---

## 核心创新 1：Encoder-Free，让 LLM 自己当 perception module

### 大白话讲

传统做法：`raw image → SigLIP encoder → 256 tokens → LLM`

MLA 做法：`raw image → patchify + linear projection → 256 tokens → LLM 前几层自己学着当 encoder`

这跟 [Fuyu](https://arxiv.org/abs/2310.03771)、[Emu3](https://arxiv.org/abs/2409.18869) 的思路一样：**干嘛非要 separate encoder？LLM 的 transformer block 本身就是强大的 feature extractor，让它自己学。**

但问题来了：point cloud 和 tactile 这种 robotics-specific 的 modality，LLM 从来没见过，你直接把 raw tokens 塞进去，LLM 怎么知道"这个 point cloud token 对应 image 里的哪个 region"？

### 关键 trick：用物理 projection 做 supervision

这是整篇 paper 最 clever 的地方。

你想想：point cloud 里的每个 3D point，通过 camera 参数，可以投影到 2D image 上的一个 pixel 位置。这是纯几何操作，不需要任何 label。

所以 MLA 做了这么件事：

1. 把 point cloud 的 256 个 token，每个 token 取它对应的 3D points 的中心点
2. 用 camera intrinsics/extrinsics 把这个 3D 中心点投影到 2D image plane
3. 看它落在哪个 14×14 image patch 上
4. 那个 image patch 的 token 和这个 point cloud token 就是 **positive pair**
5. 其他所有 token 是 **negative pair**
6. Tactile token 同理：gripper fingertip 的 3D 位置投影到 image plane，找到对应的 patch

然后用 InfoNCE contrastive loss 把 positive pair 拉近、negative pair 推开。

### 公式讲人话

Image 和 Point Cloud 之间的 contrastive loss：

$$\mathcal{L}_{\mathrm{img.pc}} = -\frac{1}{256} \sum_{i=1}^{256} \log \frac{\exp\left(\langle f_j^{\mathrm{img}}, f_i^{\mathrm{pc}} \rangle / \tau\right)}{\sum_{j=1}^{256} \exp\left(\langle f_j^{\mathrm{img}}, f_i^{\mathrm{pc}} \rangle / \tau\right)}$$

- $f_j^{\mathrm{img}}$ = 第 $j$ 个 image token 的 feature
- $f_i^{\mathrm{pc}}$ = 第 $i$ 个 point cloud token 的 feature
- $\langle \cdot, \cdot \rangle$ = 内积，衡量两个 feature 有多像
- $\tau$ = temperature，控制 softmax 有多 sharp。$\tau$ 小的时候 model 对 positive 和 negative 的区分更 aggressive
- 分子 = positive pair 的 similarity（应该大）
- 分母 = 所有 pair 的 similarity 之和（包括 positive 和 256 个 negative）
- 整个 loss 的意思：**让 positive pair 的 similarity 在所有 candidate 里占的 proportion 尽量大**

Tactile 只有一两个 token，所以 contrastive loss 是 unidirectional 的：一个 tactile token 从 256 个 image/point cloud token 里找它的 positive。

### 为什么在第 8 层做 alignment？

Ablation 显示第 8 层最好。Intuition 很简单：

- 第 4 层太浅，feature 还没融合好，alignment signal 没什么可 align 的
- 第 32 层（最后一层）太深，final hidden state 已经被 action generation 和 future prediction 两个 task 占满了 representation capacity，你再加 contrastive loss 跟抢资源
- 第 8 层刚好：浅层做 perception alignment，深层做 reasoning 和 generation，各司其职

这个 insight 让我想到 transformer 的 hierarchical feature：浅层 low-level pattern，深层 high-level semantics。MLA 在第 8 层插一刀，等于告诉 model "前 8 层你给我把 multisensory 搞对齐，后面 24 层你去做 reasoning"。

---

## 核心创新 2：Future Multisensory Generation

### 大白话讲

Alignment 解决了"看懂当下"的问题。但 robot 还需要"想象未来"——我执行这个 action 之后，image 会怎么变？point cloud 会怎么变？tactile 会怎么变？

MLA 在 LLM 的 final hidden state 后面接了三个 lightweight transformer decoder，分别预测 future image、future point cloud、future tactile。

### 三个 prediction head 细节

**Image prediction**：
- Input: LLM 最后一层 hidden state
- Output: future keyframe image
- Loss: MSE
- Trick: 用 depth map 把 background 抠掉，只预测 foreground。这样 model 专注学 object dynamics，不用浪费 capacity 学 background pixel interpolation

**Point cloud prediction**：
- Input: 同上
- Output: future keyframe point cloud
- Loss: Chamfer Distance
- Trick: 借鉴 [PointMAE](https://arxiv.org/abs/2203.06604)，把 point cloud 分成 G 个 local patch，每个 patch 有 M 个 neighbor points，先 align coarse structure 再 refine local detail

Chamfer Distance 的直觉：

$$CD(P, \hat{P}) = \frac{1}{|P|}\sum_{p \in P} \min_{\hat{p} \in \hat{P}} \|p - \hat{p}\|^2 + \frac{1}{|\hat{P}|}\sum_{\hat{p} \in \hat{P}} \min_{p \in P} \|p - \hat{p}\|^2$$

- $P$ = ground truth point cloud，$\hat{P}$ = predicted point cloud
- 第一项：每个 GT point 找最近的 predicted point，距离越小越好
- 第二项：反过来，每个 predicted point 找最近的 GT point
- 两项对称，防止 predicted points 全 collapse 到一个 mode（比如全挤在 center）

**Tactile prediction**：
- Input: 同上
- Output: future tactile embedding（低维）
- Loss: MSE

### Keyframe vs Adjacent Frame

这个 ablation 很有启发：预测 **keyframe**（70%）比预测 **adjacent frame**（64%）好。

道理：adjacent frame 跟 current frame 差别太小，大部分 pixel 没变。Model 学到的就是"复制粘贴"，supervision signal 几乎为零。Keyframe 是 action transition 的节点，物体状态有实质变化，model 被迫学真正的 dynamics。

这跟 [RVT](https://arxiv.org/abs/2305.18109)、[Lift3D](https://arxiv.org/abs/2411.17432) 等 subgoal prediction 工作的观察完全一致：**预测有意义的未来，不要预测 trivial 的未来。**

### 为什么 future prediction 只在 training 用？

三个 decoder 只在 training 时跑，inference 时完全 bypass。所以 inference latency 跟 vanilla VLA 一样。

Future prediction 的作用是：**training 时逼迫 LLM 的 internal representation 编码 dynamics 信息，这些信息间接帮助 action head 生成更好的 action。** 跟 LLM 里用 auxiliary loss 做 representation learning 一个道理。

---

## 三阶段 Training：抄 LLM 的作业

这帮人很明显在 mimic LLM 的 pretrain → SFT → RLHF 三阶段范式：

### Stage 1: Pretrain（大规模 robot data）

- 570K trajectories，36M frames
- 从 [Open-X-Embodiment](https://arxiv.org/abs/2310.08864)、[DROID](https://arxiv.org/abs/2403.12945)、RoboMIND 等 28 个 dataset 里挑
- **只用 image + language + robot state**（这些 dataset 没 point cloud 和 tactile）
- 但保留 point cloud 和 tactile 的 token position（空 token），保证 sequence structure 一致
- 10 epochs
- Loss: DDPM action loss（预测 noise）

### Stage 2: SFT（task-specific + alignment）

- 引入全部 modality：image + point cloud + tactile + language
- 300 epochs
- Loss: $\mathcal{L}_{\mathrm{diff}} + \mathcal{L}_{\mathrm{contrastive}}$
- 用前面讲的 token-level contrastive alignment

### Stage 3: Post-Training（future prediction）

- 同样的 data 和 modality
- 100 epochs
- Loss: $\mathcal{L}_{\mathrm{diff}} + \mathcal{L}_{\mathrm{contrastive}} + \mathcal{L}_{\mathrm{future}}$
- 加 future image + point cloud + tactile prediction

Post-training 对应 LLM 的 RLHF 阶段，只不过 MLA 用的是 supervised auxiliary prediction 而非 RL。这很 pragmatic——robotics 上 RL 的 sample efficiency 太差，future prediction 能直接从 demonstration 学到 dynamics signal。

---

## 实验结果讲人话

### Real-World（6 个 contact-rich task）

MLA 比 [π0](https://arxiv.org/abs/2410.24164)（SOTA 2D VLA）平均高 12%，比 [SpatialVLA](https://arxiv.org/abs/2501.15830)（SOTA 3D VLA）平均高 24%。

特别是在 "Wiping a Whiteboard" 这种需要 tactile feedback 的任务上，MLA 优势巨大。因为擦白板时，visual feedback 告诉你"擦没擦干净"，但 tactile feedback 告诉你"接触力度够不够、有没有贴着板面"。π0 和 SpatialVLA 没有 tactile，只能盲擦。

### Simulation（RLBench 10 个 task）

| Model | Mean Success Rate |
|-------|------------------|
| OpenVLA | 40% |
| SpatialVLA | 46% |
| UP-VLA | 42% |
| π0 | 65% |
| HybridVLA | 66% |
| **MLA** | **81%** |

注意：simulation 里 MLA **没用 tactile**（仿真 tactile 不真实），只用 image + point cloud。但还是比所有 baseline 高 15+ 个点。这证明 encoder-free alignment 和 future prediction 这两个核心 design 不依赖 tactile，本身就有 value。

### Generalization

在最难的 "placing egg on bread with spatula" 任务上：

| Setting | π0 | MLA |
|---------|-----|------|
| Original | 47% | 53% |
| Unseen Object | 35% (-26%) | 45% (-15%) |
| Unseen Background | 25% (-47%) | 40% (-25%) |

MLA 在 unseen object 和 unseen background 上 degradation 明显小于 π0。这说明 multisensory alignment + future prediction 学到的是更 robust 的 representation，不是靠 memorize specific visual pattern。

---

## 几个我自己的 Intuition

### 1. Encoder-free 在 domain gap 大时是优势

SigLIP 在 internet image 上 pretrain，它见过猫狗汽车风景，但没见过 robot gripper 夹鸡蛋。它的 representation space 里"鸡蛋"可能跟"圆形物体"聚类，但 robot 需要的是"鸡蛋的 fragility 和 spatial position"。

你用 SigLIP encoder，等于强制把 robot observation 塞进一个为 internet image 设计的 representation space。这个 space 可能是 bottleneck。

MLA 让 LLM 直接处理 raw token，通过 contrastive learning 自己学 representation，flexibility 更高。Ablation 也证实了：引入 SigLIP + PointMAE encoder 反而降 7%。

### 2. Physical projection 是天然的 supervision signal

这跟 [NeRF](https://arxiv.org/abs/2003.08934) 用 camera projection 做 volume rendering、[DETR](https://arxiv.org/abs/2005.12872) 用 bipartite matching 做 detection assignment 一样，都是**用数学约束代替人工 label**。

MLA 用 3D-to-2D projection 构建 contrastive positive pair，zero label cost，但 supervision signal 极其 fine-grained（token-level）。这比 CLIP 的 image-level alignment 精确得多。

### 3. Future prediction 是 implicit world model

[Dreamer](https://arxiv.org/abs/2012.09353) 系列 explicitly 建 latent dynamics model，在 latent space rollout。MLA 没有 explicit world model，但 future prediction 起到类似作用：逼迫 representation 编码 "if I do action A, the world will change to state B" 的因果信息。

Explicit world model 的优势是可以 rollout 长序列做 planning；MLA 的 future prediction 只预测 next keyframe，planning 能力有限。但 MLA 的优势是简单、可监督、inference 时零开销。

### 4. Keyframe prediction 的 redundancy 问题

预测 adjacent frame 等于让 model 学 identity function，supervision signal 被 redundant pixel 淹没。Keyframe 是 dynamics change 的 tipping point，supervision signal 最 dense。

这跟 LLM 里 next-token prediction 类似：你 predict "the" 后面的 "the" 学不到啥，predict "The capital of France is" 后面的 "Paris" 才学到 knowledge。**Supervision signal 的 information density 比 data quantity 重要。**

### 5. Tactile 一个 token 够用吗？

每个 tactile sensor 只 6 个数值（normal force、tangential force magnitude、tangential force direction 2D vector），两个 sensor 共 12 维。压成 1 个 token 看起来太激进。

但实验证明够用。原因：tactile 信息本身就是 low-dimensional 的（contact force + 方向），一个 4096-dim 的 token 足以 encode。而且 LLM 通过 cross-modal contrastive 学会了"这个 tactile token 对应 image 里的 gripper region"，contextual information 补足了 single token 的 expressiveness 限制。

### 6. 跟 scaling 的关系

MLA 用 LLaMA-2 7B。如果 scale up 到 30B+，encoder-free 的优势会更大——你不需要 pretrain 一个 30B 级别的 vision encoder，LLM 自己就是 encoder。Alignment 也更 scalable，因为 contrastive learning 在大 model 上效果通常更好。

### 7. 可能的 failure mode

Paper 没讨论，但我猜：
- Tactile sensor noise 导致 alignment signal 不稳定
- Point cloud 投影到 image 时，occluded region 找不到对应 patch（paper 没说怎么处理 occlusion）
- Future image prediction 在 lighting 变化大的场景可能不稳定
- 第 8 层 alignment 在更大数据上是否仍最优？可能需要 adaptive layer selection

---

## 跟相关工作的关系图

```
VLA 主线:
  PaLM-E → RT-2 → OpenVLA → π0 → HybridVLA
                                    ↓
                                 MLA (加 multisensory + future prediction)

Multisensory VLA:
  3D-VLA (point cloud) ─┐
  TactileVLA (tactile) ─┤→ MLA (统一 image + point cloud + tactile)
  OmniVTLA              ┘

Future Prediction VLA:
  DreamVLA (image only) ─→ MLA (image + point cloud + tactile)

Encoder-Free VLM:
  Fuyu → Emu3 ─→ MLA (用到 robotics sensors 上)

Contrastive Alignment:
  CLIP → SigLIP → ImageBind ─→ MLA (token-level + physical projection)
```

---

## 最后的 Takeaway

MLA 的核心 message 我觉得就一句：**Stop treating perception and reasoning as separate modules. Let the LLM do everything, and use physically-grounded supervision signals to teach it.**

这跟 LLM 的发展趋势一致：从 BERT 的 pretrain-finetune，到 GPT-3 的 in-context learning，到现在的 encoder-free multimodal——**统一架构 > 模块化架构**，只要你有好的 supervision signal。

MLA 给的 supervision signal 就是两个：physical projection（spatial correspondence）和 future prediction（temporal dynamics）。都是 zero-label、physically-grounded 的。这个思路我觉得可以 extend 到更多 modality（audio、proprioception、force-torque）和更长的 temporal horizon。

---

## References

- [MLA Project Page](https://sites.google.com/view/open-mla)
- [π0: A vision-language-action flow model](https://arxiv.org/abs/2410.24164)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [HybridVLA](https://arxiv.org/abs/2503.10631)
- [SpatialVLA](https://arxiv.org/abs/2501.15830)
- [DreamVLA](https://arxiv.org/abs/2507.04447)
- [UP-VLA](https://arxiv.org/abs/2501.18867)
- [CoT-VLA](https://arxiv.org/abs/2505.02293)
- [3D-VLA](https://arxiv.org/abs/2403.09631)
- [Emu3](https://arxiv.org/abs/2409.18869)
- [Fuyu](https://arxiv.org/abs/2310.03771)
- [Prismatic VLMs](https://arxiv.org/abs/2407.21219)
- [PointNet](https://arxiv.org/abs/1612.00593)
- [PointMAE](https://arxiv.org/abs/2203.06604)
- [Any2Point](https://arxiv.org/abs/2405.03922)
- [Lift3D Policy](https://arxiv.org/abs/2411.17432)
- [CLIP](https://arxiv.org/abs/2103.00020)
- [SigLIP](https://arxiv.org/abs/2303.15343)
- [SimCLR](https://arxiv.org/abs/2002.05709)
- [ImageBind](https://arxiv.org/abs/2302.01813)
- [DDPM](https://arxiv.org/abs/2006.11239)
- [DDIM](https://arxiv.org/abs/2010.02502)
- [NeRF](https://arxiv.org/abs/2003.08934)
- [DETR](https://arxiv.org/abs/2005.12872)
- [DreamerV3](https://arxiv.org/abs/2301.04104)
- [Reactive Diffusion Policy](https://arxiv.org/abs/2503.14213)
- [RVT](https://arxiv.org/abs/2305.18109)
- [Open-X-Embodiment](https://arxiv.org/abs/2310.08864)
- [DROID](https://arxiv.org/abs/2403.12945)
- [GELLO](https://arxiv.org/abs/2309.13037)
- [RLBenchmark](https://arxiv.org/abs/1909.12271)
- [DINOv2](https://arxiv.org/abs/2304.07193)

---

# MLA: Multisensory Language-Action Model 深度解析

很高兴能和 Karpathy 您探讨这篇工作。下面我会从一个 researcher 的视角，把这 paper 的 motivation、architecture、key formulations、training paradigm、experimental results 都拆解一遍，并穿插我对设计选择的理解和相关联想。

---

## 1. 问题动机：VLA 的 perception gap

当前主流 VLA (Vision-Language-Action) 模型，比如 [OpenVLA](https://arxiv.org/abs/2406.09246)、[π0](https://arxiv.org/abs/2410.24164)、[HybridVLA](https://arxiv.org/abs/2503.10631) 等，大多沿用 VLM 的范式：用预训练好的 vision encoder (通常是 [SigLIP](https://arxiv.org/abs/2303.15343)) 把 image 编码成 tokens，再喂给 LLM 生成 action。这条路径有两个根本问题：

1. **2D image 无法捕捉 spatial dependencies**：robotics 是 spatial-physical world，单纯 2D pixel 缺乏 depth、几何结构信息。
2. **Modality-specific encoders 没有 robotics 预训练**：哪怕引入 3D point cloud encoder（如 [PointNet++](https://arxiv.org/abs/1706.02413) 系列）或 tactile encoder，这些 encoder 在 internet-scale data 上预训练，representation 与 robot domain 有 gap，并且 inference 时增加额外开销。

MLA 想回答的核心 question 是：**如何把 multisensory modalities 整合进 unified representation，并预测它们的 future states，从而协同增强 VLA 对 physical world 的理解和 action generation？**

我的 intuition 是：作者认为问题不在于"加更多 modalities"，而在于"如何让 LLM 本身成为 perception module"。这跟 [Emu3](https://arxiv.org/abs/2409.18869)、[Fuyu](https://arxiv.org/abs/2309.17141) 在 general VLM 里去掉 vision encoder 的思路一脉相承，但 MLA 更进一步——通过 positional correspondence 做 contrastive alignment 来"教" LLM 看懂 robotics-specific sensors。

---

## 2. Architecture Overview

### 2.1 整体范式

MLA 建立在 [Prismatic VLM](https://arxiv.org/abs/2407.21219) 的 LLM backbone 上（实际上是 LLaMA-2 7B），不同于传统 VLA 的 vision encoder + projector 模式，MLA 设计了三个 lightweight tokenizers，直接把 raw sensory signals 转成 shared token sequence：

```
Raw Multisensory Inputs
   ├── Image I ∈ R^(H×W×3)        → Image Tokenizer   → f^img ∈ R^(B×256×4096)
   ├── Point Cloud P ∈ R^(B×1024×3) → 3D Tokenizer     → f^pc  ∈ R^(B×256×4096)
   ├── Tactile T (12-dim signal)  → Tactile Tokenizer → f^tac ∈ R^(B×1×4096)
   ├── Robot State S_t            → MLP              → state tokens
   └── Language L                 → LLaMA tokenizer  → text tokens
                                                    ↓
                              Concatenated Sequence f ∈ R^(B×N_t×4096)
                                                    ↓
                          LLaMA-2 7B (repurposed as unified policy)
                                                    ↓
                ┌───────────────────────┬──────────────────────────┐
                ↓                       ↓                          ↓
        Diffusion Action Head    Future Image Decoder    Future PC Decoder
        (DDPM, MSE on noise)     (MSE loss)              (Chamfer Distance)
                                                          Tactile Decoder (MSE)
```

### 2.2 三个 Tokenizer 细节

**Image Tokenizer**：标准 ViT-style patchify，14×14 patch size，得到 N_img = 256 个 tokens。这里没有用 SigLIP 这种预训练 vision encoder，而是直接 patch + linear projection，让 LLM 的前几层 transformer block 来做"encoder"的工作。

**3D Point Cloud Tokenizer**：基于 PointNet 系列的 FPS (Farthest Point Sampling) + KNN (k-Nearest Neighbors) + linear layer 的三 block 结构。原始 1024 个 3D points 下采样成 N_pc = 256 个 tokens。这个设计借鉴了 [PointNet](https://arxiv.org/abs/1612.00593)、[PointMLP](https://arxiv.org/abs/2112.03013)、[Any2Point](https://arxiv.org/abs/2405.03922) 等工作。

**Tactile Tokenizer**：这是最 lightweight 的设计。每个 gripper finger 装一个 tactile sensor (Tashan TS-E-A)，从每个 sensor 提取 6 个值：
- normal force (1 个)
- tangential force magnitude (1 个)
- tangential force direction (2 个分量，2D vector)

两个 sensor 共 12 维，经过一个 MLP 映射成 1 个 token f^tac ∈ R^(B×1×d_h)。这跟 [Reactive Diffusion Policy](https://arxiv.org/abs/2503.14213) 这种 visual-tactile policy 的设计类似，但 MLA 把 tactile 压缩到极简的 1 token，依赖 LLM 自己来 interpret。

### 2.3 Action Representation

Single-arm: 7-DoF vector $a_t = (\Delta x, \Delta y, \Delta z, R_r, R_p, R_y, g)$
- $\Delta x, \Delta y, \Delta z$: Cartesian position delta (end-effector 位移)
- $R_r, R_p, R_y$: Euler angles for rotation (roll, pitch, yaw)
- $g$: gripper width (夹爪开合度)

Dual-arm: 两个 7-DoF 拼接成 14-DoF，简单但需要 model 学到 inter-arm coordination。

Action head 用的是 DDPM (Denoising Diffusion Probabilistic Model)，跟 π0 的 flow matching 思路类似但实现不同。Diffusion noise tokens 和 timestep 通过 MLP projector 嵌入，append 到 token sequence 末尾，让 LLM 在 self-attention 里融合它们。Inference 时用 [DDIM](https://arxiv.org/abs/2010.02502) 加速，n=4 步采样。

---

## 3. Encoder-Free Multimodal Alignment (核心创新 1)

这是 paper 最有意思的设计。让我把数学拆开讲。

### 3.1 核心思路

传统 contrastive learning（如 [CLIP](https://arxiv.org/abs/2103.00020)、[SigLIP](https://arxiv.org/abs/2303.15343)）是 image-level alignment：一张图对应一段文字。但 robotics 场景下，multimodal tokens 在 spatial 位置上有明确 correspondence，做 image-level alignment 太粗。

MLA 的 insight 是：**用 3D-to-2D 的物理 projection 构建 token-level positive pairs**。

### 3.2 Positive/Negative Pair Construction

具体步骤：

1. **3D point cloud token → 2D image coordinates**: 每个 point cloud token $f_i^{pc}$ 是由一组 3D points 聚合而成，取这组 points 的 center point $(X_c, Y_c, Z_c)$（在 world coordinate 下）。
2. **Camera projection**: 用 camera intrinsics/extrinsics 把 $(X_c, Y_c, Z_c)$ 投影到 2D image plane，得到 pixel coordinate $(u, v)$。
3. **找对应的 image patch**: $(u, v)$ 落在哪个 14×14 patch 上，那个 patch 对应的 image token $f_j^{img}$ 就是 $f_i^{pc}$ 的 positive pair。
4. **Tactile token → 2D image**: 直接读 robot state，把 gripper fingertip 的 3D 位置投影到 2D image plane，找到对应的 image patch。
5. **Positive triplet**: $(f_j^{img}, f_i^{pc}, f^{tac})$ 在 spatial 上 correspondence。其他 unmatched tokens作为 negatives。

这个设计的妙处在于：它利用了 **physical spatial correspondence** 作为 supervision signal，而不是依赖语义标签。比 image-level contrastive 更 fine-grained，比 simple concatenation 又有 explicit alignment objective。

### 3.3 三个 Contrastive Losses

**Loss 1: Image-Point Cloud Alignment (InfoNCE)**

$$\mathcal{L}_{\mathrm{img.pc}} = -\frac{1}{256} \sum_{i=1}^{256} \log \frac{\exp\left(\langle f_j^{\mathrm{img}}, f_i^{\mathrm{pc}} \rangle / \tau\right)}{\sum_{j=1}^{256} \exp\left(\langle f_j^{\mathrm{img}}, f_i^{\mathrm{pc}} \rangle / \tau\right)}$$

变量含义：
- $f_j^{\mathrm{img}}$: 第 $j$ 个 image token 的 embedding feature（上标 img 表示 image 模态）
- $f_i^{\mathrm{pc}}$: 第 $i$ 个 point cloud token 的 embedding feature（上标 pc 表示 point cloud 模态）
- $\langle \cdot, \cdot \rangle$: 内积，作为 similarity measure
- $\tau$: temperature parameter，控制 softmax 分布的尖锐程度。$\tau$ 越小分布越尖锐（harder negative mining），$\tau$ 越大越平滑。[SimCLR](https://arxiv.org/abs/2002.05709) 用 0.5，CLAP 类工作用 0.05-0.1。
- 上标 256 是 token sequence 长度，由于 image 和 point cloud 都被 tokenize 成 256 tokens，所以是 bidirectional InfoNCE。
- 分子是 positive pair 的 similarity，分母是所有 candidate（包括 positive）的 similarity 之和。

这个 loss 的 intuition：让 spatially 对应的 image token 和 point cloud token 在 embedding space 里靠近，spatially 不对应的 token 推开。

**Loss 2 & 3: Tactile-Image and Tactile-Point Cloud Alignment**

$$\mathcal{L}_{\mathrm{tac.img/pc}} = -\log \frac{\exp\left(\langle f^{\mathrm{tac}}, f_{j/i}^{\mathrm{img/pc}} \rangle / \tau\right)}{\sum_{j/i=1}^{256} \exp\left(\langle f^{\mathrm{tac}}, f_{j/i}^{\mathrm{img/pc}} \rangle / \tau\right)}$$

变量含义：
- $f^{\mathrm{tac}}$: 单个 tactile token embedding
- $f_{j/i}^{\mathrm{img/pc}}$: 对应的 image 或 point cloud token
- 由于 tactile 只有 1 个 token（在 single-arm setting 下），所以是 **unidirectional** contrastive loss：1 个 positive vs 256 个 candidates。
- 在 dual-arm setting 下，tactile 有 2 个 tokens（每 arm 一个），可以做 bidirectional。

**Total Contrastive Objective**：

$$\mathcal{L}_{\mathrm{contrastive}} = \mathcal{L}_{\mathrm{img.pc}} + \mathcal{L}_{\mathrm{tac.img}} + \mathcal{L}_{\mathrm{tac.pc}}$$

### 3.4 Alignment 位置：第 8 层 Transformer Block

Ablation 显示，在第 8 层 transformer block 的输出做 contrastive alignment 效果最好。原因：
- **太浅**（如第 4 层）：feature 还没充分融合 multimodal context，alignment signal 太弱。
- **太深**（如第 32 层 final layer）：final hidden states 已经被 action generation 和 future prediction 多个 objective 优化，"占用"了 representation capacity，再加 contrastive 信号会被稀释。
- **第 8 层**：在 shallow 层做 alignment，留给后面 24 层 transformer block 做 high-level reasoning。

这个 insight 让我想到 transformer 的 hierarchical representation：浅层做 local pattern matching，深层做 abstract reasoning。MLA 把 alignment 摆在浅层是合理的 division of labor。

### 3.5 跟相关工作的对比

- [CLIP](https://arxiv.org/abs/2103.00020) / [SigLIP](https://arxiv.org/abs/2303.15343): image-text alignment, image-level, 需要单独 encoder
- [ImageBind](https://arxiv.org/abs/2302.01813): 多模态 alignment, 但也是 image-level
- MLA: token-level, physical spatial correspondence, encoder-free

---

## 4. Future Multisensory Generation (核心创新 2)

### 4.1 为什么做 future prediction？

单纯 contrastive alignment 让 model "理解" multimodal inputs，但 model 还缺一个关键能力——**reasoning about physical dynamics**。预测未来状态是 implicitly 学 dynamics 的好方法，相关工作如 [DreamVLA](https://arxiv.org/abs/2507.04447)、[UP-VLA](https://arxiv.org/abs/2501.18867)、[CoT-VLA](https://arxiv.org/abs/2505.02293) 都做 future image prediction。但 MLA 把它扩展到 image + point cloud + tactile 三个 modality 联合预测。

### 4.2 三个 Future Prediction Head

**Image Prediction**:
- Decoder: 4 层 transformer (self-attention + feed-forward)
- Input: LLM final-layer hidden states
- Supervision: MSE loss against next keyframe image
- **Key trick**: 用 depth map 去除 background pixels，只预测 foreground regions。这降低了 prediction 难度，让 model 专注在 foreground object 的 dynamics。
- Keyframe 定义: 基于 robotic joint velocity 和 action transitions 识别（借鉴 [RVT](https://arxiv.org/abs/2305.18109) 和 [Lift3D Policy](https://arxiv.org/abs/2411.17432)）

**Point Cloud Prediction**:
- Decoder: 4 层 transformer
- 借鉴 [PointMAE](https://arxiv.org/abs/2203.06604) (Masked Autoencoders for Point Cloud)
- 把 ground-truth point cloud 分成 G 个 local patches: 用 FPS 采 G 个 center points，每个 center 用 KNN 取 M 个 neighbor points
- Decoder 输出 $\hat{P} \in \mathbb{R}^{G \times M \times 3}$
- Supervision: Chamfer Distance

**Chamfer Distance 公式**（paper 没写但标准定义）：

$$CD(P, \hat{P}) = \frac{1}{|P|}\sum_{p \in P} \min_{\hat{p} \in \hat{P}} \|p - \hat{p}\|^2 + \frac{1}{|\hat{P}|}\sum_{\hat{p} \in \hat{P}} \min_{p \in P} \|p - \hat{p}\|^2$$

变量含义：
- $P$: ground-truth point cloud
- $\hat{P}$: predicted point cloud
- $p, \hat{p}$: 单个 3D point
- $\|\cdot\|^2$: squared Euclidean distance
- 两项对称：第一项是每个 GT point 找最近 pred point，第二项反过来。这样双向监督，避免 pred point 都 collapse 到一个 mode。

**Tactile Prediction**:
- Decoder: 4 层 transformer
- 输出 low-dimensional tactile embedding
- Supervision: MSE loss against ground-truth tactile signal

### 4.3 Keyframe vs Adjacent Frame

Ablation 有个很有意思的发现：predicting future **keyframes** (70%) > predicting **adjacent frames** (64%)。

我的解读：adjacent frame 包含大量 redundant information（大部分 pixel 没变），supervision signal 实际是"复制当前帧"，model 学到的是 trivial identity mapping。Keyframe 是 action transition 点，包含真正的 dynamics change，forcing model 学 underlying physical reasoning 而非 surface-level interpolation。

这跟 [SuSIE](https://arxiv.org/abs/2310.08440)、[Lift3D](https://arxiv.org/abs/2411.17432) 等 subgoal prediction 工作的观察一致。

---

## 5. Three-Stage Training Paradigm

MLA 的 training pipeline 显著模仿 LLM 的训练范式：

### Stage 1: Large-Scale Pretraining

- 数据：570K trajectories，36M frames，从 Open-X-Embodiment、[DROID](https://arxiv.org/abs/2403.12945)、RoboMIND 等 28 个 datasets 中精选
- 10 epochs
- **只用 image + language + robot state**（因为这些 dataset 没有 point cloud 和 tactile）
- 但**保留** point cloud 和 tactile 的 token positions（作为 empty placeholder），保证 sequence structure 一致性，便于后续 SFT 接入
- Action loss: $\mathcal{L}_{\mathrm{diff}}$ = DDPM MSE loss (预测 noise)

Sampling ratio 关键 dataset:
- BridgeV2: 20.93% (in-the-wild manipulation)
- Kuka: 20.22%
- Fractal: 13.67%
- Robo-Net: 11.53%
- BC-Z: 7.54%
- Language Table: 7.70%
- ManiSkill: 5.26%
- DROID: 4.82%

### Stage 2: Supervised Fine-Tuning (SFT)

- 引入所有 multisensory modalities (image, point cloud, tactile, language)
- 300 epochs
- Loss: $\mathcal{L}_{\mathrm{sft}} = \mathcal{L}_{\mathrm{diff}} + \mathcal{L}_{\mathrm{contrastive}}$
- 这里 contrastive loss 通过 positional correspondence 实现 token-level alignment

### Stage 3: Post-Training

- 数据和 modalities 跟 SFT 一致
- 100 epochs
- Loss: $\mathcal{L}_{\mathrm{post}} = \mathcal{L}_{\mathrm{diff}} + \mathcal{L}_{\mathrm{contrastive}} + \mathcal{L}_{\mathrm{future}}$
- 关键：**future prediction loss 只在 training 时用，inference 时完全不影响效率**

这个三阶段对应 LLM 的 pretrain → SFT → RLHF/DPO 范式，post-training 里的 future prediction 起到了 RLHF 中 reward signal 的角色——它不直接生成 action，但通过 auxiliary supervision 让 representation 更 robust。

---

## 6. 实验结果

### 6.1 Real-World Tasks (6 个 contact-rich tasks)

**Single-arm (Franka FR3 + ROBOTIQ gripper)**:
1. Pressing a stamp onto paper
2. Wiping a whiteboard with an eraser
3. Placing a dish on a rack
4. Placing an egg on bread with a spatula

**Dual-arm (两个 Franka FR3)**:
5. Scooping popcorn into a bowl
6. Opening a pot lid and picking corn

**Results** (每个 task 15 rollouts):
- MLA 平均比 π0 (SOTA 2D VLA) 高 **12%**
- MLA 平均比 SpatialVLA (SOTA 3D VLA) 高 **24%**

特别地，在 "Wiping a Whiteboard" 任务上 MLA 优势明显，因为它能用 tactile sensing 调节 downward 和 lateral movement。

### 6.2 RLBench Simulation (10 tasks)

| Model | Mean S.R. |
|-------|-----------|
| OpenVLA | 40±0.02 |
| SpatialVLA | 46±0.03 |
| UP-VLA | 42±0.04 |
| DreamVLA* | 65±0.05 |
| π0 | 65±0.04 |
| HybridVLA | 66±0.05 |
| **MLA** | **81±0.03** |

MLA 在 simulation 上甚至没用 tactile（仿真 tactile 不真实），只用 image + point cloud，但还是显著超 baselines。这证明 encoder-free alignment + future generation 这两个核心设计的普适性。

### 6.3 Generalization

在 "Placing an egg on bread with a spatula"（最难任务）上：

| Setting | π0 | MLA |
|---------|-----|------|
| Original | 47 | 53 |
| Unseen Object | 35 (-26%) | 45 (-15%) |
| Unseen Background | 25 (-47%) | 40 (-25%) |

MLA 在 unseen object 和 unseen background 上的 degradation 明显小于 π0。这归功于：
1. Multimodal alignment 学到更 robust 的 representation（不依赖单一模态的 surface patterns）
2. Future prediction 让 model 理解 underlying dynamics 而非 memorize specific configurations

---

## 7. Ablation Study 关键 Insights

### 7.1 Input Modalities & Alignment Strategy

| Config | Result |
|--------|--------|
| Ex1: 2D image only | baseline |
| Ex2: 2D + 3D (concat) | +X |
| Ex3: 2D + 3D + tactile (concat) | +Y |
| Ex4: all + image-level contrastive | +Z |
| Ex5: all + token-level contrastive (proposed) | +Z+7% |

Token-level contrastive 比 image-level contrastive 高 7%，证明 fine-grained spatial alignment 重要。

### 7.2 Extra Encoders 反而降低性能

引入预训练 SigLIP (2D) 和 PointMAE (3D) encoder，性能反而 -7%，且降低 inference efficiency。这印证了 paper 的核心论点：**robotics-specific sensors 跟 internet-pretrained encoder 有 domain gap，与其强行 align，不如让 LLM 直接 learn**。

### 7.3 Contrastive Loss Layer Position

| Layer | Result |
|-------|--------|
| 4 | 中等 |
| **8** | **best** |
| 12 | 较好 |
| 32 | limited gain |

我的理解：第 8 层是浅层 perception 和深层 reasoning 的 sweet spot。

### 7.4 Future Prediction Modality Removal

去掉任何一个 modality 的 future prediction 都会降 accuracy。三者互补：
- Image prediction: semantic information (物体状态、场景 context)
- Point cloud prediction: geometric structure (3D 位置、shape)
- Tactile prediction: interaction dynamics (contact force、摩擦)

---

## 8. 我对设计的几个 Intuition Building 联想

### 8.1 跟 LLM Pretrain-SFT-RLHF 范式的对应

MLA 的三阶段训练几乎是 LLM 训练范式的 robotics 版本：
- Pretrain: 大规模 robot data 上学 general visuomotor policy
- SFT: task-specific data + alignment (类似 instruction tuning)
- Post-training: future prediction as "RL" signal (类似 RLHF)

但 MLA 的 post-training 不是 RL，而是 supervised auxiliary prediction。这是个 pragmatic 选择——RL 在 robotics 上 sample efficiency 差，而 future prediction 能直接从 demonstration 学到 dynamics signal。

### 8.2 Positional Correspondence as Supervision Signal

MLA 用 3D-to-2D projection 构建 positive pairs，这本质上是用 **physical geometry 作为 supervision**。这让我想到：
- [NeRF](https://arxiv.org/abs/2003.08934): 也用 camera projection 但 for rendering
- [DETR](https://arxiv.org/abs/2005.12872): bipartite matching 但 for object detection
- [DINOv2](https://arxiv.org/abs/2304.07193): self-supervised visual features

MLA 是把 physical projection 用作 cross-modal contrastive supervision 的 signal，这在我看来是个 novel 的 cross-domain connection。

### 8.3 Encoder-Free 的深层含义

跟 [Fuyu](https://arxiv.org/abs/2310.03771)、[Emu3](https://arxiv.org/abs/2409.18869) 思路一脉相承，但 MLA 把它用在了 robotics-specific sensors 上。这背后有个深层 insight：

**Pretrained encoder 的 representation 可能是 bottleneck，不是 asset**。当 target domain (robotics with tactile, point cloud) 跟 pretrain domain (internet images) gap 大时，encoder 的 inductive bias 反而阻碍 alignment。让 LLM 直接处理 raw tokens，通过 contrastive learning 自己学 representation，反而更灵活。

这个观察在 LLM 社区已经有类似讨论（如 [CapabILITIES of LLMs as Encoders](https://arxiv.org/abs/2402.12788)）。

### 8.4 Future Prediction 的"World Model" 角色虽然 implicit

MLA 没显式叫 world model，但 future prediction 起到的作用其实是 implicit world modeling。跟 [Dreamer](https://arxiv.org/abs/2012.09353)、[DreamerV3](https://arxiv.org/abs/2301.04104) 不同的是：
- Dreamer: latent dynamics model, 在 latent space roll out
- MLA: explicit multimodal prediction, 直接生成 future observations

Explicit prediction 的优势是可解释、可监督，劣势是 prediction 本身可能 noise 大。MLA 用 keyframe prediction + background removal 来 mitigate noise 问题。

### 8.5 Tactile 的 Low-Dimensionality 是 Feature 不是 Bug

每个 tactile sensor 只 6 个数值，两个 sensor 共 12 维，比 image 的 256×4096 维小几个数量级。这看起来是 limitation，但实际是 design choice：
- Tactile 信息天然 low-dimensional（contact force、方向）
- 一个 token 足以 represent
- LLM 通过 cross-modal contrastive 学会 interpret 这个 token

这跟 [Reactive Diffusion Policy](https://arxiv.org/abs/2503.14213) 的 visual-tactile policy 形成对比：RDP 用更复杂的 tactile representation，MLA 极简但靠 LLM reasoning 补足。

### 8.6 跟 VLA Scaling 的关系

MLA 用 LLaMA-2 7B，相对 OpenVLA 的 7B 差不多 scale，比 PaLI-Gemma (π0 backbone) 大。如果 scale up 到 30B+，encoder-free 设计的优势会更明显：
- 不需要预训练 30B 级别的 vision encoder
- Inference efficiency 更好
- Alignment 通过 LLM 自身完成，更 scalable

### 8.7 关于 Dual-Arm Coordination

Dual-arm 14-DoF action 看起来简单（两个 7-DoF 拼接），但 coordination learning 是难点。Paper 没深入讨论 dual-arm 的 cross-arm attention 机制，但既然 LLM 用 full self-attention over all tokens，两 arm 的 tokens 自然能 cross-attend。这可能是个值得未来探索的方向。

### 8.8 Failure Modes 推测

Paper 没讨论 failure cases，但可以推测：
- Tactile sensor noise 可能导致 alignment signal 不稳定
- Point cloud 投影到 image 时，occluded region 找不到对应 patch
- Future image prediction 在 lighting 变化大的场景可能不稳定
- 第 8 层 alignment 在更大数据规模上是否仍最优？

### 8.9 跟 Sim2Real 的关系

MLA 在 RLBench 上 81%，real-world 也 SOTA。但 sim 上没用 tactile，real-world 用了。这说明 tactile 主要在 real-world contact-rich 任务上贡献，sim 的提升来自 image+point cloud 的 alignment + future prediction。这跟 [RialTo](https://arxiv.org/abs/2404.03935) 等 sim2real 工作的 robustness 来源不同。

---

## 9. 总结：MLA 给我们的几个 takeaways

1. **Encoder-free 是 viable 甚至 superior 的选择**，当 target domain 跟 pretrain domain gap 大时。
2. **Token-level cross-modal alignment** 比 image-level alignment 更 fine-grained，physical spatial correspondence 提供天然的 supervision signal。
3. **Future prediction as post-training auxiliary objective** 是 implicitly 学 world dynamics 的好方法，不影响 inference 效率。
4. **Three-stage training paradigm** 借鉴 LLM 范式，progressive capability building。
5. **Keyframe prediction > adjacent frame prediction**，redundant supervision 反而有害。
6. **Shallow layer alignment + deep layer reasoning** 是 transformer 内部的 division of labor。

---

## References

- [π0: A vision-language-action flow model](https://arxiv.org/abs/2410.24164)
- [OpenVLA: An open-source VLA model](https://arxiv.org/abs/2406.09246)
- [HybridVLA: Collaborative diffusion and autoregression](https://arxiv.org/abs/2503.10631)
- [SpatialVLA: Exploring spatial representations for VLA](https://arxiv.org/abs/2501.15830)
- [DreamVLA: VLA with comprehensive world knowledge](https://arxiv.org/abs/2507.04447)
- [UP-VLA: Unified understanding and prediction model](https://arxiv.org/abs/2501.18867)
- [CoT-VLA: Visual chain-of-thought reasoning for VLA](https://arxiv.org/abs/2505.02293)
- [Open-X-Embodiment](https://arxiv.org/abs/2310.08864)
- [DROID dataset](https://arxiv.org/abs/2403.12945)
- [Prismatic VLMs](https://arxiv.org/abs/2407.21219)
- [PointNet: Deep learning on point sets](https://arxiv.org/abs/1612.00593)
- [PointNet++](https://arxiv.org/abs/1706.02413)
- [PointMAE: Masked autoencoders for point cloud](https://arxiv.org/abs/2203.06604)
- [Any2Point](https://arxiv.org/abs/2405.03922)
- [Lift3D Policy](https://arxiv.org/abs/2411.17432)
- [CLIP](https://arxiv.org/abs/2103.00020)
- [SigLIP](https://arxiv.org/abs/2303.15343)
- [SimCLR](https://arxiv.org/abs/2002.05709)
- [DDIM](https://arxiv.org/abs/2010.02502)
- [DDPM](https://arxiv.org/abs/2006.11239)
- [Emu3](https://arxiv.org/abs/2409.18869)
- [Fuyu](https://arxiv.org/abs/2310.03771)
- [NeRF](https://arxiv.org/abs/2003.08934)
- [DETR](https://arxiv.org/abs/2005.12872)
- [DreamerV3](https://arxiv.org/abs/2301.04104)
- [Reactive Diffusion Policy](https://arxiv.org/abs/2503.14213)
- [RVT: Robotic View Transformer](https://arxiv.org/abs/2305.18109)
- [GELLO teleoperation framework](https://arxiv.org/abs/2309.13037)
- [RLBenchmark](https://arxiv.org/abs/1909.12271)
- [MLA Project Page](https://sites.google.com/view/open-mla)

希望这个深度拆解对您有所帮助，Karpathy。如果您对某个部分感兴趣，我可以再深挖——比如 contrastive loss 的 temperature schedule、dual-arm coordination 的 cross-attention pattern、或者 future prediction decoder 的具体 architecture diagram。
