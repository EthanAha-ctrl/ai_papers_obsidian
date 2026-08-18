---
source_pdf: DeVA Decoupled Video-Action Model with physical guidance for robot policy
  learning.pdf
paper_sha256: 3110217b21baacf62563f1bd43ebbd9d68861d0f41e9088284f2a9f0edb66dc4
processed_at: '2026-08-18T05:22:38-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DeVA 用人话讲

## 一句话版本

让 video model 和 action model 各干各的活，中间用 cross-attention 传纸条，还额外请两个小工（affordance 和 depth decoder）在旁边喊 "这边这边" 帮 action model 找方向。

---

## 一、这论文到底想解决啥问题

想象你在训练一个 robot policy。现在市面上两条路：

**路线 A：VLA 路线**。就是拿个 image-text 预训练的 vision-language model，加个 action head，直接 fine-tune。OpenVLA、π0、GR00T 都是这路子。问题是这些 model 预训练的时候学的是 "这张图配这段话"，**静态的**。它根本不知道 "如果手往左推，杯子会怎么动"。这种 dynamics 知识得全靠下游那点 robot demos 来补，50 个 demo 学不出多少 dynamics。

**路线 B：VAM 路线**。就是拿 video generation model 当 backbone，因为它预训练时就学了 "未来几帧会怎么演化"，天然编码了 dynamics。然后在这个 backbone 上接 action。听起来美好，但实现起来有两种搞法：

- **搞法 B1（unified）**：video 和 action 用同一个 backbone、同一个 feature space。UWM、Cosmos-Policy 这路。问题是你一个 feature 既要重建高清视频像素，又要预测 robot action，这两个目标在 representation space 上互相打架。video generation 想 feature 保留高频纹理细节，action prediction 想 feature 抽象成 "往左推 3 厘米"。一个 feature 干两件事，优化很难收敛。

- **搞法 B2（dual）**：video 和 action 用两个 transformer，action attend 一下 video 的某层 feature。Mimic-Video、DiT4DiT 这路。问题有两个：一是它们一般只从 video backbone 的**某一层**、**某一个 denoising step** 抽 feature，浪费了其他层的信息；二是 video-action 之间的知识是**隐式**学的，没有显式告诉 video model "你关注一下哪里能抓、哪里有深度结构"。

DeVA 就是路线 B 的改良版，解决 B1 和 B2 的所有痛点。

---

## 二、DeVA 的三个核心 idea

### Idea 1：分家（Decoupled Experts）

别让 video 和 action 挤在一个 backbone 里吵架。给它们各自一个 transformer：

- **Video expert**：Cosmos-Predict2 改的，28 层 DiT，2B 参数。专门学 "未来视频会怎么演化"
- **Action expert**：GR00T 风格的 action DiT，16 层，500M 参数。专门学 "robot 该怎么动"

两个 expert 参数完全独立，各算各的 loss。video expert 管 video diffusion loss，action expert 管 flow-matching loss。

但分家不是断联。它们通过 cross-attention 保持通信。

### Idea 2：多层传纸条（Multi-Level Feature Transfer）

这是这论文最 key 的贡献。

Diffusion model 的不同层、不同 denoising step 编码的东西完全不一样：
- **浅层 / 高噪声 step**：学的是 coarse layout、global structure（"杯子大概在左边"）
- **深层 / 低噪声 step**：学的是 fine appearance、texture（"杯子把手长这样"）

Action prediction 同时需要两种粒度。粗粒度决定 "往哪个区域靠近"，细粒度决定 "精确抓哪个点"。

以前的方法只从一层抽 feature，相当于只给 action model 看一种分辨率的照片。

DeVA 的做法：从 video backbone 的 28 层里**均匀采样 8 层**，每层 feature 通过 cross-attention 注入 action expert 对应的层。同时还加了 12 个 learnable bridge tokens，在 action self-attention 里当 "情报汇总员"，把全局 video context 聚合一下。

这样 action expert 可以 attend 粗粒度也可以 attend 细粒度，想看啥看啥。

### Idea 3：物理小工喊话（Physical Guidance）

Video expert 学的是 "重建像素"，它不 care 哪些 pixel 是 "能抓的地方"，哪些是 "不能碰的"。但 action prediction 关心的恰好是这些。

DeVA 在 video backbone 旁边挂了两个轻量 decoder：

**Affordance decoder**（17.6M 参数，很小）：
- 输出一张和 image 同分辨率的 heatmap
- 每个像素的值是 "end effector 落在这里的概率"
- 用语言 condition（通过 FiLM），所以不同指令下 affordance 不同
- 仿真里用 MuJoCo 的 contact point 做 ground truth
- 真实世界里用 UAD 这个 off-the-shelf model 生成 pseudo-label

**Depth decoder**（13.4M 参数，也很小）：
- 输出 relative depth map
- 用 Video Depth Anything 离线生成 ground truth
- 近的地方值接近 0，远的地方接近 1

这两个 decoder 做两件事：
1. **反向监督 video features**：让 video backbone 的中间特征编码 "哪里能抓" 和 "几何结构"，而不只是 "像素怎么重建"
2. **正向指导 action expert**：decoder 的中间层 feature 也 concat 到 action cross-attention 的 KV 里，action expert 直接看到物理提示

---

## 三、训练怎么搞：两阶段

### Stage 1：热身（10K steps）

只训 video DiT + 两个 decoder。让 video backbone 先学会 "预测未来视频" + "中间特征编码物理结构"。

Loss 是三个加起来：

$$\mathcal{L}_v = \mathcal{L}_{\mathrm{video}} + \lambda_{\mathrm{aff}} \mathcal{L}_{\mathrm{aff}} + \lambda_{\mathrm{depth}} \mathcal{L}_{\mathrm{depth}}$$

这里的 video loss 是标准 EDM denoising。affordance loss 是 noise-weighted 的 BCE：

$$\mathcal{L}_{\mathrm{aff}} = \mathbb{E}_t\left[w(t) \cdot \ell_{\mathrm{BCE}}\left(\hat{A}_t, \mathcal{A}_t\right)\right], \quad w(t) = \frac{1}{(1 + \sigma_t)^2}$$

**变量解释**：
- $t$：diffusion 的 noise step（从 0 到 T）
- $\sigma_t$：step $t$ 对应的噪声标准差
- $w(t)$：权重函数。噪声大时 $\sigma_t$ 大，$w(t)$ 小，loss 权重低。因为高噪声时 video feature 还很乱，不能指望 decoder 从垃圾里预测出 clean affordance
- $\hat{A}_t$：模型预测的 affordance map
- $\mathcal{A}_t$：ground truth affordance map
- $\ell_{\mathrm{BCE}}$：binary cross-entropy，对 positive class 加权解决前景背景不平衡

Depth loss 类似，多了个 gradient term 保持边缘锐利：

$$\mathcal{L}_{\mathrm{depth}} = \mathbb{E}_t\left[w(t) \left(\ell_1^{\mathrm{mask}}(\hat{D}_t, D_t) + \lambda_{\mathrm{grad}} \cdot \ell_{\mathrm{grad}}^{\mathrm{mask}}(\hat{D}_t, D_t)\right)\right]$$

**变量解释**：
- $\hat{D}_t$：预测的 depth map
- $D_t$：ground truth depth（Video Depth Anything 生成的）
- $\ell_1^{\mathrm{mask}}$：只在有效 depth 像素上算 L1（有些像素可能没有 depth 值，用 mask 掉）
- $\ell_{\mathrm{grad}}^{\mathrm{mask}}$：masked image gradient loss，惩罚 depth map 的空间梯度的不一致，鼓励局部几何光滑
- $\lambda_{\mathrm{grad}}$：gradient loss 的权重

### Stage 2：联合训练（~45K steps）

加入 action expert，两个 expert 一起训。**关键：decoders 冻结但 loss 保留**。

为什么冻 decoder？如果 decoder 参数还在动，它和 action expert 互相扯皮，action loss 会通过 decoder 拉扯 decoder 参数，decoder 参数变了又影响 video feature 的物理 grounding，不稳定。

为什么不关 loss？video backbone 还需要 affordance/depth loss 继续塑形，不能让它退回纯 pixel reconstruction。

Action loss 是 flow-matching：

$$\mathcal{L}_{\mathrm{act}} = \mathbb{E}_t\left[\left\|v_\theta(\tilde{A}_t, t, c) - u_t\right\|_2^2\right]$$

**变量解释**：
- $v_\theta$：action DiT 学的 velocity field（神经网络）
- $\tilde{A}_t$：在 flow-matching time $t$ 的 noised action（通过线性插值 noise 和 clean action 得到）
- $t$：flow-matching 的时间变量，从 0（纯 noise）到 1（clean action）
- $u_t$：target flow，即从 noise 指向 clean action 的向量
- $c$：conditioning，包括 observation、language、multi-level video features、physical guidance features

总 loss 简单相加：$\mathcal{L} = \mathcal{L}_v + \mathcal{L}_{\mathrm{act}}$

---

## 四、架构细节再过一遍

### Video Expert

- Cosmos-Predict2 预训练 checkpoint，480p/16fps
- 28 层 DiT，2B 参数
- T5-XXL（4.86B）做 language encoding，**全程 frozen**
- 多 view 通过 tiling 拼成一帧处理（省计算）
  - RoboCasa：3 view 拼成 $2\times2$（第四格空着）或 $1\times3$
  - LIBERO：2 view 水平拼接
  - YAM 真实机器人：3 view 水平拼接
- 第一帧作为 condition，其余帧是 prediction target

### Action Expert

- 16 层 DiT，500M 参数
- self-attention 和 cross-attention 交错
- 8 个 cross-attention 层对应 video backbone 均匀采样的 8 层
- 12 个 learnable bridge tokens（dim 1024）在 self-attention 里聚合全局 video context
- input MLP 投影 noisy action，output MLP 投影回 action space
- **没有单独的 visual encoder**，直接吃 video features + T5 embeddings + physical guidance

### Physical Guidance Decoder

- 两个 DPT-style decoder
- 每个 4 个 decoding stage，加 temporal attention
- 从 video backbone 抽 multi-scale feature：
  - Affordance 用 blocks $\{6, 13, 20, 27\}$
  - Depth 用 blocks $\{3, 10, 17, 24\}$（故意错开，不抢同一层的 feature）
- Affordance decoder 通过 FiLM 注入 T5 language embedding
- 最细输出分辨率 $H/4 \times W/4$
- 第三 stage 的 feature resample 后 concat 到 action conditioning

### 参数账本

| 部件 | 参数量 | 状态 |
|---|---|---|
| T5-XXL | 4.86B | frozen |
| VAE tokenizer | 127M | frozen |
| Video DiT | 1.96B | 训 |
| Action head | 564M | 训 |
| Affordance head | 17.6M | Stage 1 训 / Stage 2 冻 |
| Depth head | 13.4M | Stage 1 训 / Stage 2 冻 |
| Fusion projection | 5.2M | 训 |
| **可训总计** | **2.56B (34%)** | |
| **总计** | **7.55B** | |

注意 decoder 只占可训参数的 1.4%，非常轻量。大头还是 video DiT 和 action expert。

---

## 五、实验结果讲人话

### RoboCasa（厨房任务，24 个 task）

- DeVA 用 50 demos/task 拿 72.0%
- GR00T-N1.5 用 300 demos 拿 64.1%
- Cosmos-Policy 用 50 demos 拿 67.1%（最近 baseline）

DeVA 用**六分之一的 data** 打过了用 300 demos 的方法，比同 data budget 的 Cosmos-Policy 高 5 个点。

### LIBERO（40 个桌面 task）

- DeVA 平均 99.0%
- 几乎所有 baseline 都 97-98%，已经接近 ceiling
- DeVA 在 Long horizon suite 上 98.4%，领先最明显

Long horizon 最考验 temporal dynamics，正好是 video prior 发威的地方。

### LIBERO-Plus（鲁棒性测试，10030 个扰动 variant）

- DeVA 80.8%
- OpenVLA-OFT 69.6%（第二）

DeVA 领先 11 个点。这很 make sense：affordance 和 depth 是 geometric invariants，appearance 怎么变（打光、背景纹理、camera 角度），"哪里能抓" 和 "近的近的远的远" 这些几何结构不会变。

### 真实机器人（双臂，3 个 task，30 demos/task）

| Method | Handover | Lift Pot | Pick Bottles | Avg |
|---|---|---|---|---|
| GR00T-N1.6 | 0.88 | 0.45 | 0.10 | 0.48 |
| π0.5 | 0.70 | 0.55 | **0.75** | 0.67 |
| Cosmos-Policy | 0.45 | 0.23 | 0.35 | 0.34 |
| **DeVA** | **0.90** | **0.68** | 0.65 | **0.74** |

DeVA 在 Handover 和 Lift Pot 上最强。Pick Up Bottles 上 π0.5 略胜，作者分析是因为 π0.5 的 action horizon 是 50 步（DeVA 是 25 步），长 horizon 在 grasp-lift-place 多阶段任务里保持 action coherence 更好。

Cosmos-Policy 才 0.34，unified 架构在有限 data + 有限 step 下收敛确实慢。

---

## 六、Ablation 讲人话

### 1. 各组件贡献（Fig 8a）

| 配置 | Success Rate | 增量 |
|---|---|---|
| 只训 action（无 video） | 19.8 | baseline |
| + 预测 goal image | 25.8 | +6.0 |
| + 预测 future video | 36.8 | +11.0 |
| + decoupled 多层 transfer | 66.0 | **+29.2** |
| + physical guidance | 72.0 | +6.0 |

**大头是 +29.2 那个**。从 36.8 暴涨到 66.0，全是 decoupling + multi-level transfer 的功劳。这说明：
- video prior 有用（+11），但光有 video prior 不够
- 关键是怎么把 video 信息**高效**传给 action
- 单层抽 feature 或 unified backbone 都不行，多层 + decoupled 才是正解

### 2. Physical Guidance 的两个组件（Fig 8b）

- Base：66.0
- + Affordance：涨
- + Depth：涨
- + 两个都加：72.0

两个互补：affordance 说 "去哪"，depth 说 "怎么过去"。

### 3. Feature 交互方式（Table 7）

| 方式 | Success Rate |
|---|---|
| Self-attention 直接融合 | 66.79 |
| Cross-attention 直接融合 | 66.50 |
| Cross-attention + 额外 transformation 层 | 63.75 |
| 聚合后再 cross-attention | 62.75 |

**越折腾越差**。video backbone 的 feature 已经很 rich，简单直接地 attend 就好，加额外 processing 层反而破坏信息。这和 [Diffusion Hyperfeatures](https://arxiv.org/abs/2305.16843) 的发现一致。

### 4. 要不要单独编码 initial observation（Table 8）

加了单独 initial frame encoding 只涨 0.25 个点，基本无用。因为 video backbone 本身就 conditioned on initial frame，feature 里已经有了。DeVA 的设计是 self-consistent 的。

### 5. 用哪个 denoising step 的 feature（Table 9）

| 策略 | Success Rate |
|---|---|
| 只用最后一步 denoising feature | 49.6 |
| 只用最后一步 + 冻 video backbone | 22.4 |
| 多步 denoising feature 聚合 | 53.2 |

两个发现：
1. **多步比单步好 3.6 点**：不同 denoising step 编码不同 abstraction，多步更丰富
2. **冻 backbone 暴跌到 22.4**：把 video model 当 frozen feature extractor 根本不行，必须 joint adapt

第二个发现特别重要。它说明 pretrained video feature 虽然有 dynamics prior，但这个 prior 不直接对齐 action prediction。必须让 video backbone 在 action loss 的拉扯下调整 feature 分布，才能 serve action。

### 6. Decoupled vs Unified 收敛速度（Fig 7b）

- Unified 在 45K step 收敛到 34%
- Decoupled base 同 budget 收敛到 66%
- Decoupled + physical 71%

Decoupled 不仅天花板高，**收敛快一倍**。Unified 的问题就是 video loss 和 action loss 在同一个 feature space 里互相拉扯，gradient 方向冲突，优化 landscape 坎坷。

---

## 七、直觉总结

### 为什么 Decoupling 比 Unified 好

想象你要训一个学生同时写诗和做数学题。你让他用同一个脑回路干两件事，他得在 "诗歌感" 和 "数学感" 之间反复横跳，两个都学不好。

DeVA 让两个 student 各自专精，然后定期交流笔记。Video student 专注 "未来画面怎么变"，action student 专注 "手怎么动"。cross-attention 是他们交流的渠道。

### 为什么 Multi-Level Transfer 重要

想象你要描述一个城market。只看卫星图知道大概布局，只看街景图知道细节，两个都不够。你两个都要，根据需要切换。

Video backbone 的不同层就像不同分辨率的地图。单层 transfer 只给 action student 一种分辨率，多层 transfer 给一整套 from satellite to street view，action student 自己选。

### 为什么 Physical Guidance 有用

Video student 天生学的是 "像素怎么重建"，它对 "哪里能抓" 这种 task-relevant 信息没感觉。你额外给它两个 homework：画 affordance heatmap 和画 depth map。它为了完成这两个作业，中间 feature 就得 encode "哪里能交互" 和 "几何结构"。这些 feature 又传给 action student，action student 就直接拿到了 "去哪抓、怎么抓" 的提示。

而且这两个 decoder 的 feature 直接 concat 到 action cross-attention 的 KV 里，相当于 action student 不仅看 video student 的笔记，还看两个 physical 小工的笔记。

### 为什么 Stage 2 冻 Decoder 但保留 Loss

想象 physical 小工学会画 affordance map 后，你让他保持画风不变（冻参数），但还继续交作业（loss 保留）。这样：
- 小工的"判断标准"稳定，action student 看到的 guidance 不会乱跳
- Video student 还得继续为了满足小工的判断标准而调整 feature

如果连小工的判断标准也跟着变（不冻），action student 和小工互相扯皮，不稳定。如果连作业都不交了（关 loss），video student 就偷懒退回纯 pixel reconstruction。

---

## 八、我觉得还可以改进的地方

1. **Inference 慢**：作者自己承认，joint denoising video + action 比纯 action policy 贵。未来可以用 latent video prediction 或 accelerated sampler 解决。

2. **Affordance 还是 2D image-plane 的**：只在 image 上说 "去这里"，没有 6-DoF SE(3) 的 affordance。ReKep 那种 3D keypoint constraint 可能更强。

3. **Action horizon 固定 25 步**：Pick Up Bottles 上输给 π0.5（50 步）说明 horizon 选择 matters。Adaptive horizon 或 hierarchical action chunk 可能更好。

4. **Pseudo-label 质量**：真实场景 affordance 用 UAD 生成 pseudo-label，depth 用 Video Depth Anything 生成。这两个 teacher model 的 error 会 propagate 进来。没看到对 pseudo-label 质量的 sensitivity analysis。

5. **Bridge tokens 的作用没单独 ablate**：12 个 bridge tokens 多少贡献不清楚。

参考：
- [DeVA Project Page](https://deva-model.github.io)
- [Cosmos-Predict2](https://github.com/nvidia-cosmos/cosmos-predict2)
- [π0 Paper](https://arxiv.org/abs/2410.24164)
- [OpenVLA](https://openvla.github.io/)
- [GR00T-N1](https://arxiv.org/abs/2503.14734)
- [Cosmos-Policy](https://arxiv.org/abs/2412.04252)
- [Diffusion Hyperfeatures](https://arxiv.org/abs/2305.16843)
- [Video Depth Anything](https://arxiv.org/abs/2501.12375)
- [UAD Affordance](https://arxiv.org/abs/2410.01844)
- [ReKep 3D Keypoint](https://arxiv.org/abs/2409.01652)
- [LIBERO Benchmark](https://libero-project.github.io/)
- [RoboCasa](https://robocasa.ai/)

---

# DeVA 深度讲解：Decoupled Video-Action Model with Physical Guidance

## 一、核心 motivation：为什么需要 DeVA

先 build 一下这个工作的 intuition。当前的 robot policy learning 主要有三个流派：

1. **VLA (Vision-Language-Action)**：代表是 OpenVLA [1]、π0 [2]、π0.5 [3]、GR00T-N1 [22]。它们继承 image-text pretraining 的强 semantic priors，但 pretraining objective 是**静态的** image-text matching，对 physical dynamics、temporal causality、action consequence 的 supervision 很弱。所以 control-relevant knowledge 都得从下游 imitation learning 里学，对 task-specific demonstrations 需求大。

2. **Unified VAM**：例如 UWM [5]、Cosmos-Policy [6]、WAM [7]。把 video prediction 和 action prediction 放在**同一个** backbone 或 latent space 里。优点是 cross-modal exchange 直接，缺点是单一 feature space 要同时 support visual generation 和 control prediction，模态特异性被挤压，policy optimization 难收敛。

3. **Dual-DiT VAM**：例如 Mimic-Video [9]、DiT4DiT [10]。video transformer 和 action transformer 分开，action 可以 attend video features。但它们一般只从**选定**的一层、**固定**的 denoising stage 抽 feature，没有充分利用 video backbone 各层各 stage 分布开的 complementary abstractions。而且 video-action objective 都是 implicit 学，没有显式引入 scene geometry、interaction region 这类 task-relevant signal。

DeVA 想要兼顾三件事：
- **Specialized experts**：video 和 action 各自保留模态特异 capacity
- **Multi-level feature transfer**：跨 video backbone 多层 transfer，让 action expert 接触到分布式的 spatiotemporal representations
- **Physically salient guidance**：用 affordance 和 relative-depth 显式 shape video features 并直接 condition action stream

参考：[Cosmos-Predict2 GitHub](https://github.com/nvidia-cosmos/cosmos-predict2), [π0 arXiv](https://arxiv.org/abs/2410.24164), [OpenVLA](https://openvla.github.io/), [GR00T-N1](https://arxiv.org/abs/2503.14734)

---

## 二、架构总览（Figure 2 解析）

DeVA 由四块组成：

### 1. Video Expert
- 基于 **Cosmos-Predict2** [60] 的 latent video diffusion transformer
- 初始化自 480p/16fps 预训练 checkpoint
- 包含 **28 个 DiT blocks**，约 **2B parameters**
- 使用 spatiotemporal VAE 把 video 压成 compact latent tokens
- T5-XXL text encoder（4.86B params）做 language conditioning，**全程 frozen**
- 训练用 standard EDM denoising objective

输入：当前 observation $O_t$ + language instruction $T$ + 多 view tiled 成一帧（RoboCasa 是 $2\times2$ 或 $1\times3$，LIBERO 是水平 tile 2 view，YAM 是水平 tile 3 view）
输出：future observations $\{O_{t+1}, \dots, O_{t+h}\}$

### 2. Action Expert
- 基于 GR00T-N1.5 [22] 的 action head 风格
- **16 层** DiT，约 **500M parameters**
- self-attention 和 cross-attention 交错
- **12 个 learnable bridge tokens**，dimension 1024，batch-expanded 成 (n, 12, 1024)
- input MLP 把 noisy action trajectory 投影到 model dim
- output MLP 把 denoised tokens 投影回 action space

输出：action sequence $\{A_t, \dots, A_{t+h-1}\}$

### 3. Multi-Level Feature Interaction
- 从 video backbone 的 28 层里**均匀采样 8 层**（early-to-late）
- 这 8 层 feature 通过 **layer-wise cross-attention** 注入 action expert 对应的 8 个 cross-attention 层
- 同时 **bridge tokens** 在 action self-attention 中聚合 video context，提供 compact global interface

### 4. Physical Guidance Decoder（Figure 3）
- 两个 **DPT-style decoder**：affordance + relative-depth
- 各 4 个 decoding stages，加 interleaved temporal attention
- Affordance decoder：17.6M params
- Depth decoder：13.4M params
- 从 video backbone 不同 block 抽 multi-scale feature
  - Affordance 用 blocks $\{6, 13, 20, 27\}$
  - Depth 用 blocks $\{3, 10, 17, 24\}$
- Affordance decoder 通过 **FiLM** [50] 注入 T5 language embedding（task-conditioned）
- 输出 finest resolution 为 $H/4 \times W/4$
- 第三 stage 的 feature 经 resample + concat channel dim 后注入 action expert 的 conditioning

### 总参数量（Table 2）

| Component | Parameters | Status |
|---|---|---|
| T5-XXL | 4.86B | frozen |
| VAE tokenizer | 126.9M | frozen |
| Video2World DiT | 1.96B | trained |
| Action head | 564M | trained |
| Affordance head | 17.6M | trained |
| Depth head | 13.4M | trained |
| Joint-fusion projection | 5.2M | trained |
| **Trainable total** | **2.56B (33.9%)** | |
| **Total** | **7.55B** | |

注意：decoders 只占 trainable parameters 的 **1.4%**，非常轻量。

---

## 三、关键公式深度解析

### 公式 1：Affordance Map 定义

$$\mathcal{A}_t(u, v) = P\left(p_t^{ee} = (u, v) \mid O_t, T\right)$$

**变量解释**：
- $\mathcal{A}_t(u, v)$：时刻 $t$ 在图像坐标 $(u, v)$ 处的 affordance 值，是一个概率
- $p_t^{ee}$：end effector 在 image plane 上的投影位置
- $O_t$：当前 observation
- $T$：language instruction（task-conditioning）

**直觉**：每个 pixel 表示 "如果当前要执行 instruction $T$，end effector 落在 $(u,v)$ 的 likelihood"。这是 task-conditioned 的 affordance，不是 generic 的。

**Target 构造**：
- 仿真：从 MuJoCo 拿 contact point 投影到 image，用 3×3 Gaussian kernel ($\sigma=1.0$) 平滑
- 真实：用 UAD [50] 这个 off-the-shelf affordance model 生成 pseudo-label

### 公式 2：Affordance Loss

$$\mathcal{L}_{\mathrm{aff}} = \mathbb{E}_t\left[w(t) \cdot \ell_{\mathrm{BCE}}\left(\hat{A}_t, \mathcal{A}_t\right)\right], \quad w(t) = \frac{1}{(1 + \sigma_t)^2}$$

**变量解释**：
- $w(t)$：noise-weighted 函数，$t$ 是 diffusion noise step
- $\sigma_t$：diffusion 时刻 $t$ 对应的 noise level
- $\hat{A}_t$：模型预测的 affordance map
- $\mathcal{A}_t$：ground truth affordance map
- $\ell_{\mathrm{BCE}}$：pixel-wise binary cross-entropy with positive-class reweighting

**为什么这么设计**：
- 当 $\sigma_t$ 大（噪声大），$w(t)$ 小，loss 权重小；当 $\sigma_t$ 小（噪声小、接近 clean），$w(t)$ 大
- 这避免了高噪声阶段强迫 decoder 输出 clean affordance（因为 video feature 此时还很混乱）
- 重新加权 positive class 是因为 affordance map 中前景/背景极不平衡

### 公式 3：Depth Loss

$$\mathcal{L}_{\mathrm{depth}} = \mathbb{E}_t\left[w(t) \left(\ell_1^{\mathrm{mask}}(\hat{D}_t, D_t) + \lambda_{\mathrm{grad}} \cdot \ell_{\mathrm{grad}}^{\mathrm{mask}}(\hat{D}_t, D_t)\right)\right]$$

**变量解释**：
- $\hat{D}_t, D_t$：预测深度和 ground truth depth
- $\ell_1^{\mathrm{mask}}$：只在 valid depth pixels 上算 L1（masked）
- $\ell_{\mathrm{grad}}^{\mathrm{mask}}$：masked image gradient loss，鼓励局部几何一致性
- $\lambda_{\mathrm{grad}}$：gradient loss 的权重

**Target 构造**：
- 用 **Video Depth Anything** [57] 离线生成 relative depth
- 反转并归一化到 $[0,1]$，近 = 0，远 = 1
- 每个 camera view 独立估计再 tile，避免 boundary artifacts
- 不需要 metric accuracy，只要 relative 几何 structure

### 公式 4：Video Diffusion Loss

$$\mathcal{L}_{\mathrm{video}} = \mathbb{E}_t\left[\lambda(t) \cdot \ell_{\mathrm{EDM}}(\hat{V}_t, V_t)\right]$$

**变量解释**：
- $\hat{V}_t$：denoised video latent prediction 在 diffusion time $t$
- $V_t$：ground truth target latent
- $\lambda(t)$：EDM 标准加权 term
- $\ell_{\mathrm{EDM}}$：EDM [Karras et al.] 的 denoising objective

### 公式 5：Action Flow-Matching Loss

$$\mathcal{L}_{\mathrm{act}} = \mathbb{E}_t\left[\left\|v_\theta(\tilde{A}_t, t, c) - u_t\right\|_2^2\right]$$

**变量解释**：
- $v_\theta$：velocity prediction network（action DiT）
- $\tilde{A}_t$：在 flow-matching time $t$ 的 interpolated noisy action trajectory
- $u_t$：target flow（从 noise 到 clean action 的向量场）
- $c$：conditioning inputs，包括 observation、language、multi-level video features、physical guidance
- $t$：flow-matching timestep（注意：和上面的 diffusion $t$ 是不同的）

**为什么用 flow-matching 而不是 diffusion**：action 是低维连续向量，flow-matching 训练更稳定、采样更高效，是 π0、GR00T 系列采用的范式。

### 公式 6 & 7：两阶段总 Loss

**Stage 1（warmup 10K steps）**：
$$\mathcal{L}_v = \mathcal{L}_{\mathrm{video}} + \lambda_{\mathrm{aff}} \mathcal{L}_{\mathrm{aff}} + \lambda_{\mathrm{depth}} \mathcal{L}_{\mathrm{depth}}$$

**Stage 2（联合训练）**：
$$\mathcal{L} = \mathcal{L}_v + \mathcal{L}_{\mathrm{act}}$$

Stage 2 中 **decoders freeze**，但 gradient 仍能通过 frozen decoder 回流到 video backbone，让 video features 持续被 physical structure 拉扯。这是一个很微妙的设计：decoder 参数不动，但它定义的"目标方向"还在塑形 video features。

---

## 四、训练 Pipeline 细节

### Stage 1: Video and Decoder Warmup（10K steps）
- 只训 video DiT + affordance decoder + depth decoder
- 让 video backbone 学会：(a) 预测 future video，(b) 中间 feature 编码物理结构

### Stage 2: Joint Video-Action Training（~45K steps）
- 加入 action expert
- 两个 expert 联合优化，但保持各自参数空间独立
- physical decoders frozen 但仍 active（输出仍作为 guidance，loss 仍计算）
- 学习率 schedule：
  - 1K steps linear warmup 到 $10^{-4}$
  - 1k-30k linear decay 到 $3\times10^{-5}$（0.3× peak）
  - 之后降到 $6\times10^{-6}$（0.06× peak）保持
- batch size 64，8 GPU FSDP，bfloat16
- gradient clip global norm 1.0

### Inference
- video latent 和 action trajectory 都从 Gaussian noise 初始化
- 通过各自的 generative process 联合采样
- physical decoders 保持 active 提供实时 guidance

### Action Representation 细节（Table 4）

| Setting | YAM | RoboCasa | LIBERO |
|---|---|---|---|
| Clip length | 25 | 33 | 25 |
| Resolution | 128×384 | 256×256 | 128×256 |
| Action dim | 14 | 7 | 7 |
| State dim | 14 | — | — |
| Delta mask | (6, -1, 6, -1) | no | no |
| Norm. | quantile 1/99 pct | min-max | min-max |

YAM 的 14-DoF bimanual action：6 维 delta pose（每臂）+ 1 维 absolute gripper。Delta mask (6, -1, 6, -1) 表示前 6 维用 delta，第 7 维（gripper）用 absolute，对两臂都这样。

---

## 五、实验结果深度分析

### 1. RoboCasa (Table 1)

| Method | Demos/Task | Success Rate |
|---|---|---|
| GR00T-N1 | 300 | 49.6 |
| DP-VLA | 3000 | 57.3 |
| π0 | 300 | 62.5 |
| GR00T-N1.5 | 300 | 64.1 |
| FLARE | 300 | 66.4 |
| **Cosmos-Policy** | **50** | **67.1** |
| **DeVA (ours)** | **50** | **72.0** |

**关键 insight**：DeVA 用 50 demos 比用 300-3000 demos 的方法都高。相比同样 50 demos 的 Cosmos-Policy（同属 VAM），DeVA 高 4.9 个点，说明 decoupling + physical guidance 比 unified 架构有实质优势。

### 2. LIBERO (Fig 5a)

| Method | Spatial | Object | Goal | Long | Short-Avg | Avg |
|---|---|---|---|---|---|---|
| Diffusion Policy | 78.3 | 92.5 | 68.3 | 50.5 | 79.7 | 72.4 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 98.3 | 96.9 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 98.0 | 97.1 |
| CogVLA | 98.6 | 98.8 | 96.6 | 95.4 | 98.0 | 97.4 |
| Cosmos-Policy | 98.1 | 100.0 | 98.2 | 97.6 | 98.8 | 98.5 |
| DiT4DiT | 98.4 | 99.6 | 98.6 | 97.6 | 98.9 | 98.6 |
| **DeVA** | **99.2** | **99.6** | **98.8** | **98.4** | **99.2** | **99.0** |

LIBERO 上很多方法都接近 ceiling，DeVA 在 Long horizon 上 98.4 是最显著的领先点（long-horizon 最考验 temporal dynamics）。

### 3. LIBERO-Plus 鲁棒性 (Fig 5b)

| Method | Success Rate |
|---|---|
| OpenVLA | 15.6 |
| WorldVLA | 25.0 |
| NORA | 39.0 |
| UniVLA | 42.9 |
| π0 | 53.6 |
| π0-fast | 61.6 |
| OpenVLA-OFT | 69.6 |
| RIPT-VLA | 68.4 |
| **DeVA** | **80.8** |

LIBERO-Plus 有 10,030 个 perturbation variants，覆盖 7 个维度：object layout、camera viewpoint、robot init、language、lighting、background texture、sensor noise。DeVA 比 OpenVLA-OFT 高 11.2 个点。从 LIBERO 99.0% 掉到 80.8% 的 18.2 个点也说明 robustness 仍有 gap。

**为什么 DeVA 鲁棒性好**：physical guidance（affordance + depth）提供了 geometric invariants。即使 appearance 变了，affordance map 仍指向 "where to act"，depth 仍描述 "scene geometry"，这种 inductive bias 对 distribution shift 鲁棒。

### 4. 真实世界 Bimanual Manipulation (Table 6)

| Method | Handover Marker | Lift Pot | Pick Up Bottles | Avg |
|---|---|---|---|---|
| GR00T-N1.6 | 0.88 | 0.45 | 0.10 | 0.48 |
| π0.5 | 0.70 | 0.55 | 0.75 | 0.67 |
| Cosmos-Policy | 0.45 | 0.23 | 0.35 | 0.34 |
| **DeVA** | **0.90** | **0.68** | 0.65 | **0.74** |

平台：I2RT YAM 双臂，每臂 6-DoF + 1-DoF gripper = 14-DoF。3 个 RGB-D camera（2 wrist + 1 head）。30 demos/task via Meta Quest Pro 遥操作。

- Handover Marker：DeVA 0.90，比 GR00T-N1.6 略高，比 π0.5 高 20 点
- Lift Pot：DeVA 0.68，需要双臂协同抓 handle 同步抬起，最考验 coordination
- Pick Up Bottles：π0.5 0.75 反而最高，作者分析是因为 π0.5 有 50-step action horizon（vs DeVA 的 25-step），长 horizon 在 grasp-lift-place 间保持 action coherence 更好

---

## 六、关键 Ablation 详解

### Ablation 1：Backbone Design（Fig 8a）

| Variant | Success Rate |
|---|---|
| Action-only | 19.8 |
| + Goal image prediction | 25.8 (+6.0) |
| + Future video prediction | 36.8 (+11.0) |
| + Decoupled multi-level transfer | 66.0 (+29.2) |
| + Physical guidance (full DeVA) | 72.0 (+6.0) |

**关键 insight**：
- 从 action-only 到 +goal image 只涨 6 个点，说明 single goal frame 提供的 dynamics 信息有限
- +future video 大涨 11 点，证明 video generation 学到的 spatiotemporal prior 有用
- +decoupled multi-level transfer 暴涨 29 点！这是论文最核心的贡献。说明 video backbone 的不同层编码了不同的 abstraction，多层 transfer 比单层或共享 backbone 好得多
- +physical guidance 再涨 6 点，是 nice add-on

### Ablation 2：Physical Guidance（Fig 8b）

- Base (decoupled): 66.0
- + Affordance: 涨
- + Depth: 涨
- + Both: 72.0（最高）

两者 complementary：affordance 提供 "where"（interaction region），depth 提供 "how"（geometry），共同 ground action prediction。

### Ablation 3：Feature Interaction Mechanism（Table 7）

| Mechanism | Success Rate |
|---|---|
| Self-attention in action head | 66.79 |
| Direct cross-attention | 66.50 |
| Specific layers for each cross-attention | 63.75 |
| Aggregated attention before cross-attention | 62.75 |

**核心 insight**：
- 简单直接的 self-attention 或 cross-attention 效果最好
- 加额外 transformation 或 aggregation 反而**降性能**！这暗示 pretrained video backbone 的 feature 已经 encode 了足够信息，**过度 processing 反而破坏**信息
- DeVA 最终选 cross-attention + bridge tokens，既支持 layer-wise transfer，又能做 global aggregation

### Ablation 4：Conditioning Sources（Table 8）

| Conditioning | Success Rate |
|---|---|
| T5 + Initial Obs + Video | 66.75 |
| T5 + Video only | 66.50 |

**Insight**：单独编码 initial observation 几乎无用（+0.25）。因为 video backbone 本身就 conditioned on initial frame，其 features 已包含 visual context。这是 DeVA 设计 self-consistent 的证据。

### Ablation 5：Denoising Step Conditioning（Table 9）

| Strategy | Success Rate |
|---|---|
| Final denoising step | 49.6 |
| Final step + frozen backbone | 22.4 |
| Multi-step denoising features | 53.2 |

**两个关键 insight**：
1. Multi-step 比 final-step 高 3.6 点。这印证了 [Diffusion Hyperfeatures](https://arxiv.org/abs/2305.16843) 的观察：不同 denoising step 的 feature 编码不同 abstraction，多步聚合更丰富
2. Frozen backbone 暴跌到 22.4。说明**不能把 video backbone 当 frozen feature extractor**，必须 joint adaptation 让 video features 朝 action-relevant 方向 move

### Ablation 6：Decoupled vs Unified（Fig 7b）

- Unified 架构在 ~45K step 收敛到 34%
- Decoupled base 在同 budget 收敛到 66%
- Decoupled + physical guidance 收敛到 71%

Decoupled 不仅 final 更高，**收敛速度也快**。Unified 架构的 optimization 难度源自 single feature space 同时被 video generation 和 action prediction 两个 loss 拉扯，gradient 方向冲突。

### Ablation 7：Data Efficiency（Fig 7a, 7c）

- 同样 demo budget 下 DeVA 都高于 π0.5 和 GR00T-N1.6
- 完整 budget 下 DeVA 66% vs π0.5 48% vs GR00T-N1.6 29%
- DeVA 处理的 training examples 比 Cosmos-Policy 少 20×

---

## 七、直觉与 Why It Works

### 1. 为什么 Decoupling 比 Unified 好

考虑一个 unified backbone 同时输出 video latent 和 action token。Video generation loss 鼓励 feature 保留 high-freq visual detail，action prediction loss 鼓励 feature 抽象出 control-relevant signal。这两个目标在 representation space 上是**正交甚至冲突**的。

Decouple 后：
- Video expert 自由学 visual dynamics
- Action expert 自由学 control abstraction
- 两者通过 cross-attention 通信，gradient 通过这个接口自然平衡

### 2. 为什么 Multi-Level Transfer 重要

Diffusion model 的不同层/不同 denoising step 编码了 hierarchical abstraction（参考 [Diffusion Hyperfeatures](https://arxiv.org/abs/2305.16843), [Revelio](https://arxiv.org/abs/2412.04252)）：
- 浅层 / 高噪声 step：粗粒度 layout、global structure
- 深层 / 低噪声 step：细粒度 appearance、texture

Action prediction 需要多种粒度的信息：粗 layout 决定 "approach which region"，细 appearance 决定 "precise grasp point"。单层 transfer 只覆盖一个 abstraction level，多层 transfer 让 action expert 能自由 attend 任意粒度。

### 3. 为什么 Physical Guidance 是 Game Changer

Video generation 的 objective 是 reconstruct pixels，它不 care 哪些 pixel 是 interaction region。Action prediction 关心的恰好是 interaction region（gripper 要去哪里）和 geometry（怎么 avoid collision、怎么 align）。

Affordance supervision 把 video features 在 interaction region 处"激活"，相当于 attention bias。Depth supervision 把 video features 在 geometric 边界处"锐化"。这两个 supervisory signal 把 video backbone 从"general future predictor"塑造成"task-relevant future predictor"。

而且这两个 signal 的 decoder feature 直接作为 KV 注入 action cross-attention，相当于 action expert 同时看到 "predictive dynamics" 和 "physical affordance cues"。

### 4. 为什么 Frozen Decoder 但 Active Supervision

Stage 2 冻 decoder 但保留 loss 是个 elegant trick：
- 冻 decoder：保证 action expert 看到的 physical guidance 是稳定的，不会随 action loss 而漂移
- Active loss：video backbone 仍受 affordance/depth loss pull，保持物理结构 encoded
- 这避免了 decoder 和 action expert 互相扯皮，又保证了 video feature 的物理 grounding 不丢

---

## 八、Limitations 与未来方向

作者自承认：
1. **Computational cost**：joint denoising video + action 引入 inference overhead
2. **Spatiotemporal attention** 比纯 action policy 贵
3. 未来方向：latent-space prediction + accelerated sampling

我还想补充几个潜在问题：
- **Calibration**：affordance pseudo-label 在真实场景用 UAD [50] 生成，质量如何 calibrate？
- **Horizon mismatch**：YAM 实验中 π0.5 用 50-step horizon 反而在 Pick Up Bottles 上超过 DeVA 的 25-step，说明长 horizon 对某些 task 重要
- **3D reasoning**：虽然 depth 提供几何，但仍是 2D image-plane affordance。如果做 6-DoF affordance 在 SE(3) 上可能更强（参考 [ReKep](https://arxiv.org/abs/2409.01652)）

---

## 九、Related Work 速览

- **VLA 主线**：RT-1 [16], RT-2 [4], PaLM-E [17], OpenVLA [1], Octo [19], CogACT [20], RDT-1B [21], π0 [2], π0.5 [3], OpenVLA-OFT [24], CogVLA [29], UniVLA [28]
- **Video-Action 主线**：Video Policy [38], UVA [37], UWM [5], Cosmos-Policy [6], WAM [7], Mimic-Video [9], DiT4DiT [10], ViPRA [39], FLARE [36]
- **Physical Grounding**：Where2Act [41], 3D-AffordanceNet [42], Affordances from Human Videos [43], General Flow [44], RoboPoint [45], ReKep [46], RT-Affordance [47], UAD [50], 3D-Diffusion-Policy [54], Depth Anything V2 [56], Video Depth Anything [57]
- **Diffusion Feature Interpretation**：Diffusion Hyperfeatures [11], Revelio [12]

参考链接：
- [DeVA Project Page](https://deva-model.github.io)
- [Cosmos-Predict2](https://github.com/nvidia-cosmos/cosmos-predict2)
- [LIBERO Benchmark](https://libero-project.github.io/)
- [RoboCasa](https://robocasa.ai/)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [Depth Anything V2](https://arxiv.org/abs/2406.09414)
- [Video Depth Anything](https://arxiv.org/abs/2501.12375)
- [π0.5](https://arxiv.org/abs/2504.16054)
- [Diffusion Hyperfeatures](https://arxiv.org/abs/2305.16843)

---

## 十、总结

DeVA 的核心贡献可以浓缩成一句话：**让 video generative backbone 和 action policy 各自专业化，通过 multi-level feature transfer + physical guidance 让两者高效协作**。

它不是追求更大的 unified model，也不是简单把 video model 当 feature extractor，而是探索**结构化的 expert 分工 + 物理监督**这一中间路线。从 ablation 看，decoupling 带来的 29 点提升是最大头的贡献，physical guidance 的 6 点是 cherry on top。

对你（Karpathy）这种关注 modular design 和 inductive bias 的视角，DeVA 的 design 暗合 "modularity beats monolith when modalities need different inductive biases" 的直觉。它也再次验证：**pretrained feature 不能 frozen 用，必须 joint adapt**（Table 9 的 22.4% vs 49.6% 是 best evidence）。

如果想 build deeper intuition，我建议关注两个延伸方向：
1. **ReKep-style 3D keypoint constraints** + DeVA 的 physical guidance：把 affordance 从 image plane 升级到 SE(3) keypoint
2. **Latent video prediction** + accelerated sampling：解决 DeVA 自承的 inference cost 问题，类似 [LAPO](https://arxiv.org/abs/2410.11758) 的 latent action pretraining 思路
