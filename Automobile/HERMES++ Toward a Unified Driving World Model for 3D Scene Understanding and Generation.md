---
source_pdf: HERMES++ Toward a Unified Driving World Model for 3D Scene Understanding
  and Generation.pdf
paper_sha256: d263a24f321adbdff5086da5e0c75a019352f1802ec7d28e07597063c31a6a5f
processed_at: '2026-08-19T10:58:26-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HERMES++ 人话版：让 AI 既能"看懂"又能"预测未来"的 driving brain

Karpathy，我把这篇 paper 拆成几个"吃饭聊天"能讲明白的层次，从最顶层的问题到最底层的 trick 都过一遍。

---

## 一、这篇 paper 到底在解决什么 human problem？

想象你坐上一辆 autonomous driving 的车，开到一个复杂 intersection。一个好的 driving AI 应该能干两件事：

**第一件 —— 看懂现在**：你问它"前面那个红 car 是不是要 cut in？"，它能答出来。这叫 scene understanding。现在有一堆 VLM-based 方法（OmniDrive、DriveLM、DriveGPT4）能做这个。

**第二件 —— 预测未来**：它能在脑子里 "simulate" 接下来 3 秒整个 scene 会怎么变 —— 前车会往前开、左边 pedestrian 会过马路、traffic light 会变红。这叫 future generation。GAIA-1、ViDAR、OccWorld、DriveX 这一批 world model 能做这个。

**问题是**：这两拨人各干各的，互相不说话。

- 做 generation 的那帮人：能 simulate 未来，但是 mute —— 你问它"为什么前面那辆车要左转？"，它答不上来。它只会画出未来 scene 的样子，不懂语义。
- 做 understanding 的那帮人：能滔滔不绝解释 scene，但是 frozen 在 current frame —— 你问它"3 秒后那个 pedestrian 在哪？"，它只能瞎猜，没有真正的 temporal dynamics modeling。

**HERMES++ 的核心 mission**：把这俩能力塞进一个 brain 里，而且要让他们互相 help —— 语义理解指导未来预测，几何预测反过来 ground 语义理解。

这个 motivation 其实就是 LeCun 喊了好几年的 JEPA 思想在 driving 上的落地，也是 NVIDIA GR00T、World Labs、DeepMind Genie 2 这些 foundation world model 的统一愿景 —— 一个 model 同时 perceive、understand、predict、act。

---

## 二、核心 insight：用 BEV 当 LLM 和 3D 世界的"翻译官"

最大的 design challenge 是：**LLM 是吃 token 的（1D sequence），但 driving 是 3D 空间问题**。怎么把 multi-view camera 的 2D 图片"喂"给 LLM，同时让 LLM 处理后还能还原回 3D 几何？

主流方案有两种，都有问题：

**方案 A：直接把 6 个 camera 的图各切成 patch，flatten 成 token 塞给 LLM**
- 问题：6 个 view × 每张图几百个 patch = token 数量爆炸，超 LLM context window
- 问题：LLM 不知道哪些 token 属于哪个 view、view 之间什么 spatial 关系、怎么 fuse
- 结果：spatial topology collapse。LLM 会把"路上的白线"误以为是"左转标志"（paper 里 Fig. 4 的真实失败案例）

**方案 B：用 sparse query（像 Q-Former）从多 view 里 extract 几百个 token**
- 问题：query 太 sparse，几何信息丢得太多
- 问题：需要额外 supervision（3D box、lane detection）才能学好

**HERMES++ 的方案：BEV tokenizer**

这是整个 paper 最关键的 insight。简单讲就是：

1. 用 ConvNeXt-L encode 6 个 camera 的图（standard 操作）
2. 用 BEVFormer 的 spatial cross-attention，把 6 个 view 的 feature "lift" 到一个 top-down 的 BEV 平面（180×180 grid，每个 cell 对应真实世界约 0.57m × 0.57m 的区域）
3. 下采样 4×（变成 45×45 = 2025 个 token），同时 channel expand 4×（保信息）
4. Flatten 成 2025 个 token，linear project 到 LLM hidden dim，喂给 LLM

**为什么 BEV 是 key**：
- **统一坐标系**：6 个 view 的信息在同一个 metric 空间里 fuse 了，每 token 有明确 spatial location（第 3 行第 5 列的 token 就是 BEV 里那个 cell）
- **token 数合理**：2025 个 token，正好处在 LLM context window 的 sweet spot
- **几何信息保住了**：因为 BEV 是 top-down view，天然的 geometric structure，不会像 flatten perspective view 那样 collapse
- **和 LLM 兼容**：flatten 后就是 1D token sequence，LLM 直接处理

paper 里 Fig. 4 的 ablation 证明这个 point：用等量 token（2532），multi-view 输入 3s 预测 CD=1.91，BEV 输入 CD=1.44，差 32%。而且 multi-view baseline 会 hallucinate "左转"，BEV 正确预测直行。**spatial structure 是不能丢的**。

这个 idea 跟 BEVWorld、OmniDrive 的思路一脉相承，但 HERMES++ 是 first 把 BEV token 同时用于 understanding 和 future generation。

---

## 三、World Queries：让"未来"能读懂"现在"的小 trick

好，现在我们能让 LLM 看 BEV 了，它能答 VQA 了。但 future generation 怎么办？

最 naive 的想法：LLM 处理 BEV token，输出一个 latent，然后 decoder 这个 latent 成 future point cloud。但这样 understanding 和 generation 是 separated 的，没有 synergy。

HERMES++ 的 trick：**搞一组 special token 叫 world queries，让它们跟 BEV token、text token 一起进 LLM，借 LLM 的 reasoning 能力"读"语义信息，然后再拿出来用于 future generation**。

具体怎么搞：

1. 从下采样后的 BEV feature 用 adaptive max pooling 提取 $n=4$ 个 base spatial query（选最 salient 的 spatial 信息，相当于 free foreground detection）
2. 对每个 future time step（1s, 2s, 3s），把 ego-motion embedding 加进去（告诉 query "我们 1 秒后会往前走 5m"）
3. 再加 learnable frame embedding（告诉 query "我是第 1 秒的 query" 还是 "第 2 秒的"）
4. Concat + project 到 LLM dim，得到 $(\Delta t \times n) = (3 \times 4) = 12$ 个 world queries
5. 这些 queries 和 BEV tokens、text tokens 一起 concat 喂给 LLM

**关键 magic**：LLM 是 causal attention，这些 world queries 放在 sequence 末尾，能 attend 到前面所有 BEV token 和 text token。所以 LLM 处理这些 query 的过程，就是把 semantic context 和 world knowledge "写" 进 query representation。

比如 LLM 在 answer "前面那辆红 car 要 cut in" 时，world queries 就"读到"了这个 context。然后这些被 enrich 的 query 拿出来，去 condition future BEV generation。

**paper Table VII 的 ablation 是 smoking gun**：

| 设定 | 3s CD ↓ |
|------|---------|
| 不用 world queries，只用 text injection | 1.634 |
| World queries bypass LLM（不经过 LLM reasoning） | 1.526 |
| World queries 走过 LLM（proposed） | **1.436** |

走过 LLM 的 version 显著好于 bypass 的。**LLM 的 reasoning 和 world knowledge 是关键**，queries 不能直接和 BEV feature 做 cross-attention 了事，必须让 LLM "思考"一下。

更妙的是：generation loss 的梯度会通过 world queries 反传回 LLM，**反向提升 LLM 的 understanding 能力**。这是真正的 bidirectional synergy —— semantic 帮 geometric，geometric 也帮 semantic。

这个设计让我想到你（Karpathy）常说的 "autoregressive transformer 是 universal computation substrate"。这里 LLM 不只 generate text，它还在"思考"如何预测未来几何，把 world queries 当作 working memory。

---

## 四、Current-to-Future Link：把"现在" propagate 到"未来"

World queries 是 sparse prior（每 future frame 才 4 个 token），但要生成 dense future BEV feature（45×45 = 2025 个 cell），需要把 current BEV feature propagate 到未来。

这就是 Current-to-Future Link 的活。它是一个 stack of 6 个 transformer block，每个 block 有 cross-attention、self-attention、feed-forward 三层。

**Cross-attention：future BEV 怎么"查询"现在和语义**

公式（简化版）：
$$\text{Future BEV} = \text{Current BEV} + \text{CrossAttn}(\text{Current BEV}, [\text{World Queries}; \text{Text Embedding}])$$

意思是：future BEV 的每个位置都去"查询" world queries 携带的 semantic prior 和 text embedding 携带的 reasoning guidance。这就实现了 "语义指导几何演化" 的 mechanism。

**Ego Modulation：解决"车动了背景也动"的混淆**

这是非常 tricky 的问题。车往前开 5m，整个 background 也会相对往后流 5m。但如果一个 pedestrian 站着不动，她在 BEV 里的位置也应该不变。怎么让 model 分清"是我在动所以背景流"和"object 自己在动"？

HERMES++ 用 DiT 的 adaLN-Zero trick：
1. 把 ego-motion 通过 MLP + Tanh 编码成两个 vector $\gamma$ 和 $\beta$
2. 在 self-attention 和 feed-forward 的 LayerNorm 后做 affine transform：$\text{output} = (\gamma + 1) \odot \text{LN}(x) + \beta$
3. $\gamma$ 和 $\beta$ 在训练初期 zero-initialized（保证初期是 identity transform，训练稳定）

**关键**：ego modulation 只 applied 到 self-attention 和 feed-forward，不 applied 到 cross-attention。理由是 cross-attention 应该专注 semantic aggregation，不应该被 ego-motion 干扰。

这个设计本质就是 video diffusion 里 condition timestep 的方法，这里 condition ego-motion。效果：让 model 内部把 ego-motion 和 scene dynamics 解耦。

**Table V 的 ablation 一目了然**：

| 设定 | 3s CD ↓ |
|------|---------|
| 去掉 Link（直接 copy current BEV + ego-motion） | 2.377 |
| Simple Link（3 层 vanilla attention） | 1.542 |
| + Textual Injection（加 text embedding condition） | 1.506 |
| + Ego Modulation（加 ego-motion condition） | 1.442 |
| + More blocks（3→6 层） | **1.436** |

w/o Link 的 CD 2.377 灾难性 —— 直接 copy current BEV 完全不能 model scene evolution，即使加了 ego-motion。Simple Link 就把 CD 砍到 1.542。Textual Injection 降 0.036，Ego Modulation 再降 0.064。每个 module 都在 contribute。

---

## 五、BEV-to-Point Render：把 2D feature 变 3D cloud

现在有了 future BEV feature，怎么变成 point cloud？这里用 differentiable neural rendering。

两步：

**第一步：BEV（2D）lift 到 volumetric（3D）**

BEV feature 是 $180 \times 180$ 的 2D grid，没有 height 信息。要变 3D，就沿 height 维 expand 成 32 个 level，得到 $180 \times 180 \times 32$ 的 volumetric feature，再用 3D conv refine。

**第二步：SDF neural rendering**

对每条 LiDAR ray（从 LiDAR 中心出发的射线），在 ray 上 sample $n$ 个点。每个点 trilinear interpolate 出 local feature，过一个 shallow MLP 预测 SDF（signed distance function）值。然后 volume rendering 把 SDF 转成 depth：

$$\text{depth} = \sum_i w_i \cdot d_i$$

权重 $w_i$ 由 SDF gradient 决定 —— SDF 在 surface 处 zero-crossing，gradient 大的地方就是 surface，weight 就高。

这套技术是 NeuS（NeurIPS'21）的经典方法，在 driving 领域 UniPAD（CVPR'24）和 PonderV2 已经用过。HERMES++ 直接复用。

**为什么用 SDF 而不是直接 Chamfer Distance**：
- Chamfer Distance 是 discrete point cloud 之间的距离，对 noise 敏感，gradient 不稳定
- SDF rendering 是 continuous、differentiable 的，optimization landscape 平滑
- 可以处理 partial observation（某些 ray 没打中任何 object）

---

## 六、Joint Geometric Optimization：训练时的隐藏 regularizer

这是 HERMES++ 相对 conference version HERMES 的主要新增 contribution。

**问题 motivation**：只用 explicit point cloud loss（render loss）训，latent feature 会学"捷径" —— 沿 camera projection rays 形成 ray-shaped artifacts，center 处响应异常高（paper Fig. 5a）。因为这个 representation 在 perspective view 下渲染时已经能 minimize loss，但 latent 本身没真正学到 intrinsic 3D geometry，是"伪几何"。

**解决思路**：训练时引入一个 frozen geometry-aware encoder，提供 target，让 predicted feature 对齐它。

具体：

1. **Pretrain** 一个 sparse 3D conv encoder，用 self-supervised point cloud reconstruction（voxelize GT point cloud → encoder → render → reconstruct）。这个 encoder 学到 "geometry-aware representation"。
2. **Main training** 时，把 encoder frozen，用它处理 GT point cloud 得到 target feature $\mathbf{V}_t$
3. 强制 predicted volumetric feature $\hat{\mathbf{V}}_t$ 对齐 $\mathbf{V}_t$，用两个 loss：

**Cosine loss（voxel-wise 局部对齐）**：
每个 voxel 的 feature vector 方向要对齐，不管 magnitude。给 latent 一定自由度。

**Gram loss（global structural pattern 对齐）**：
把 volumetric feature 沿三个正交轴（XY, XZ, YZ）pool，算 Gram matrix（feature 之间的 pairwise correlation），让 predicted 和 target 的 Gram matrix 一致。

Gram matrix loss 是 Neural Style Transfer（Gatys et al. 2015）的经典技巧，在那里它捕捉"style"（texture、correlation pattern）。这里 reinterpret 成"global structural pattern" —— 哪些 voxel 之间应该 correlate，比 voxel-wise 局部对齐更全局、更 robust。

**Table IV ablation**：

| Cosine | Gram | 3s CD ↓ |
|--------|------|---------|
| ✗ | ✗ | 1.637 |
| ✓ | ✗ | 1.441 |
| ✗ | ✓ | 1.544 |
| ✓ | ✓ | **1.436** |

只 explicit loss 时 CD=1.637，加 implicit regularization 降到 1.436。Fig. 5 可视化：只用 explicit loss 时 feature 有明显 ray-shaped artifacts 和 center bias；加 implicit regularization 后 feature 变 spatially compact、符合 intrinsic geometry。

**Inference 时 frozen encoder 不用**，所以零额外 inference cost。这是 training-time-only regularizer。

---

## 七、实验数据到底说明什么？

### 7.1 Generation（3s CD ↓）

- ViDAR (CVPR'24): 1.73
- DriveX (ICCV'25): 1.10
- HERMES (conference, 1.8B): 1.17
- **HERMES++ (1.8B): 1.01**
- **HERMES++ (3.8B): 0.97**

相比 DriveX（当时的 SOTA）降 0.13（11.8%），相比 conference version 降 13.7%。

### 7.2 Understanding（CIDEr ↑，OmniDrive-nuScenes）

- LLaVA-OneVision (7B): 0.732
- Omni-Q (7B, +3D Box/Lane supervision): 0.732
- OmniDrive-BEV (7B, +supervision): 0.595（差！BEV 直接喂 7B LLM 没好 alignment 反而差）
- HERMES (1.8B, no supervision): 0.741
- **HERMES++ (3.8B, no supervision): 0.772**

HERMES++ 用更小 LLM + 不用 auxiliary supervision，超过 7B baseline。BEV representation 的 geometric inductive bias 起到关键作用。

注意 OmniDrive-BEV（直接用 BEV 喂 7B LLM）效果反而差（0.595），说明光有 BEV 不够，还要有好的 alignment 机制（world queries、Current-to-Future Link 这些 trick）。

### 7.3 Emergent planning ability

HERMES++ 没用 motion planning supervision 训练，只在 world queries 上 attach 一个轻量 MLP head 做 trajectory regression。结果（nuScenes validation）：

- UniAD: L2=0.46m, Collision=0.37%
- ORION (ICCV'25): L2=0.34m, Collision=0.37%
- OmniDrive++: L2=0.33m, Collision=0.30%
- **HERMES++: L2=0.37m, Collision=0.29%（最低 collision rate）**

通过优化 future scene generation，模型 inherently 学到了 driving dynamics 和 collision avoidance。这是 emergent capability 的例子 —— 你不显式训 planner，但 planner 能力"涌现"出来了。

### 7.4 LLM scaling 持续收益

- 0.8B: 3s CD 1.434
- 1.8B: 3s CD 1.436
- 3.8B: 3s CD **1.255**

3.8B vs 0.8B 在 generation error 降 12.5%。这说明 LLM 内在的 world knowledge 是 scalable inductive bias。Driving scene 高度 compositional（car + pedestrian + road + traffic light + causal interaction），更大 LLM 携带的 prior 更丰富。

---

## 八、几个 ablation 背后的 intuition

### Max Pooling 胜过 Cross-Attention

World queries 的初始化方式，max pooling 比 learnable cross-attention 还好（CD 1.436 vs 1.442）。这违反"learnable 一定比 hand-crafted 好"的直觉。

解释：BEV 是 sparse representation（绝大部分 voxel 是空背景），max pooling 自然 select 最 salient feature（object occupancy），相当于 "free" foreground detection。Avg pool 把背景噪声稀释信号，attention pool 在 limited data 下难训。这是 strong inductive bias 在低 data regime 的胜利。

### n=1 在 generation 最好，n=4 在 understanding 最好

World queries 数量 $n$：$n=1$ 在 generation CD 最好（1.419），但 $n=4$ 在 understanding CIDEr 最好（0.720）。

解释：generation 不需要太多 query token（1 个就够，因为是 sparse prior），understanding 需要 more representational capacity（要 capture 丰富 semantic）。作者选 $n=4$ 做 trade-off。Future work 机会：decoupled query allocation per task。

### Discontinuous horizon 是杀手

预测 horizon 实验：跳过 1s 和 2s 直接预测 0s+3s，CD 1.677；连续预测 0-3s，CD 1.436。差 17%。

解释：driving dynamics 是 Markovian，中间 frame 是 future prediction 的 inductive bridge。直接外推 3 秒，noise 放大，optimization landscape 恶化。这印证 long-horizon prediction 需要 intermediate state。

---

## 九、跟 broader world model literature 的 connection

### 跟 LeCun JEPA 的关系

JEPA（Joint Embedding Predictive Architecture）思想：在 latent space 做 prediction，不在 pixel space。HERMES++ 走的正是 latent prediction 路线 —— predict future BEV feature，再 render 出 point cloud。这避免了 pixel-level generation 的 high-frequency noise 问题，专注 high-level structure。

不过 HERMES++ 用 explicit point cloud supervision，不是真正 self-supervised joint embedding。这跟 V-JEPA 2 的 fully self-supervised 还有 gap。

### 跟 VLA（Vision-Language-Action）的关系

VLA model（GR00T、OpenVLA、ImpromptuVLA）目标是 perception → reasoning → action 全打通。HERMES++ 不是 VLA（没有 action 输出），但 motion planning 的 emergent ability 暗示 VLA potential。如果加 action token 输出，可能变成 driving VLA。

### 跟 Diffusion-based world model 的关系

DriveDreamer、Vista、GAIA-1 用 diffusion model 做 future video generation。HERMES++ 不用 diffusion，用 transformer + neural rendering。区别是：
- Diffusion：generation quality 高，但是 inference 慢（要 denoise 多步）
- Transformer + rendering：inference 快（single forward pass），generation quality 取决于 latent 表达能力

HERMES++ 选择后者，因为 driving 需要低 latency。但 diffusion 路线在 visual fidelity 上更强。

### 跟 Token-level unified world model 的关系

真正 unified 的 GPT-style world model（像 GPT-4o 那种 unified multimodal token）是 Epona、DrivingGPT、Doe-1 在探索的方向。HERMES++ 还没到那个程度 —— 它 BEV、text、world queries 是 separate input slot，function 上 specialized。但这是 first step。

---

## 十、Potential limitations 和 future work

### 3s horizon 太短

3s 在 real autonomous driving 是短 horizon（高速 3 秒 ≈ 100m）。实际需要 5-10s。HERMES++ 没探索 >3s。Long horizon 的 error accumulation 是 world model 的 fundamental challenge。

### Single-shot prediction 而非 autoregressive

HERMES++ 一次预测所有 3 个 future frame，不基于已生成 frame 继续外推。这样 inference 快、error 不会 exponentially accumulate，但 extend horizon 困难。Autoregressive 路线理论上能 extend 更长 horizon，但容易 drift。

### Camera-only 输入但 LiDAR supervision

Pros：inference 不需要 LiDAR，sensor 成本低。
Cons：监督信号依赖 LiDAR 数据，仍是 supervised。未来方向是纯 video + text supervision，类似 Vista 的路线。

### Open-loop evaluation 的 gap

Motion planning 是 open-loop（ego trajectory 已知）。Open-loop 和 closed-loop gap 很大（参考 BEV-Planner 的 critique）。HERMES++ 的 emergent planning 在 closed-loop 是否成立是 open question。

### Unified 程度

虽然 claim "unified framework"，实际架构仍有 specialization（BEV tokens 给 understanding，world queries 给 generation）。这更像 "shared backbone with task-specific interface"，而非完全 unified token space。

---

## 十一、为什么我觉得这 paper 重要

回到本质，HERMES++ 的核心 insight 浓缩成几条：

1. **BEV 是 LLM 和 3D 物理世界之间的 natural bridge**。把 multi-view information consolidate 到统一 metric 空间，preserve geometric topology，compress 到 LLM-compatible token 数量。

2. **World queries 是 knowledge transfer 的关键载体**。让 queries 走过 LLM，pretrained world knowledge 和 reasoning capability 自然注入；让 queries 参与 forward pass，generation gradient 反向 improve understanding。这是 bidirectional synergy。

3. **Implicit geometric regularization 解决 latent "伪几何" 问题**。Frozen geometry-aware encoder 提供 target，cosine + Gram loss 在 voxel-wise 和 global pattern 两个 level 对齐，避免 explicit supervision 的 shortcut 问题。

4. **Ego Modulation 通过 adaLN 解耦 ego-motion 和 scene dynamics**。借 DiT 的 conditioning mechanism，让 model 分清"背景因为我前进而流"和"object 因为自己而动"。

5. **Model scaling 在 unified framework 下持续收益**。3.8B vs 0.8B generation error 降 12.5%，说明 LLM 的 world knowledge 是 scalable inductive bias。

这条路走下去，driving world model 离你（Karpathy）常说的 "general world model" 圣杯又近了一步 —— 一个 model 同时 perceive、understand、predict、act，all grounded in physical reality。

---

## External References

核心 method 参考：
- World Models (Ha & Schmidhuber 2018): https://arxiv.org/abs/1802.02281
- DreamerV3: https://arxiv.org/abs/2301.04105
- V-JEPA 2 (Meta AI): https://ai.meta.com/blog/v-jepa-2-world-model-analysis-robot-learning/
- I-JEPA: https://arxiv.org/abs/2301.08243

Driving world model:
- HERMES original (ICCV'25): https://arxiv.org/abs/2507.04598
- DriveX (ICCV'25): https://arxiv.org/abs/2509.12314
- ViDAR (CVPR'24): https://arxiv.org/abs/2404.07991
- OccWorld (ECCV'24): https://arxiv.org/abs/2401.14162
- Vista (NeurIPS'24): https://arxiv.org/abs/2410.01476
- GAIA-1 (Wayve): https://arxiv.org/abs/2309.17080

BEV 和 VLM:
- BEVFormer (ECCV'22): https://arxiv.org/abs/2203.17070
- UniAD (CVPR'23 best paper): https://arxiv.org/abs/2212.10156
- InternVL2: https://arxiv.org/abs/2404.16821
- LLaVA: https://arxiv.org/abs/2304.08485
- OmniDrive (CVPR'25): https://arxiv.org/abs/2406.05422
- DriveLM (ECCV'24): https://arxiv.org/abs/2402.04832

Neural rendering:
- NeuS: https://arxiv.org/abs/2106.10689
- UniPAD: https://arxiv.org/abs/2312.13724
- PonderV2: https://arxiv.org/abs/2402.00806

Conditioning mechanism:
- DiT (Peebles & Xie, ICCV'23): https://arxiv.org/abs/2212.09748
- Neural Style Transfer (Gatys et al.): https://arxiv.org/abs/1508.06576

VLA 和 broader:
- GR00T (NVIDIA): https://developer.nvidia.com/groot
- Doe-1: https://arxiv.org/abs/2412.09627
- Epona: https://arxiv.org/abs/2503.06624
- ImpromptuVLA (NeurIPS'25): https://arxiv.org/abs/2502.20132

你自己的 work 也有 connection：
- nanoGPT (Karpathy): https://github.com/karpathy/nanoGPT
- "State of GPT" talk 里你强调 latent prediction 和 autoregressive 的 universal power，这 paper 在 driving domain 印证

希望这版"人话"帮你 build intuition。如果你想 dive deeper 某个 specific module（比如 SDF rendering 的 numerical 细节、Gram loss 的 spectral interpretation、world queries 的 attention pattern 可视化），告诉我，我再展开。

---

# HERMES++：Driving World Model 的统一理解与生成框架深度解读

Karpathy 你好，这篇 paper 在 driving world model 这个 niche 里算是一个比较扎实的尝试，把 LLM-based scene understanding 和 future point cloud generation 这两个 traditionally 分裂的 paradigm 在一个 unified framework 里拼起来。我会把所有重要的 technical 细节、公式变量含义、ablation 背后的 intuition，以及和 broader world model literature 的 connection 都讲一遍。

---

## 1. Motivation 与 problem framing

Driving world model 这个领域目前有两条平行的 line of work：

**Generation-centric 路线**：GAIA-1 [5], DriveDreamer [21], ViDAR [7], OccWorld [6], DriveX [18] 等，主要任务是预测未来 scene 的 evolution（2D video 或 3D point cloud / occupancy）。它们能 forecast 但是 mute on semantics —— 不会解释为什么这么演化、不能回答 VQA、不能给出 scene description。

**Understanding-centric 路线**：DriveLM [16], OmniDrive [15], DriveGPT4 [17], ORION [66] 等 VLM-based 方法，能做 VQA、scene description、graph reasoning，但是 frozen 在当前 frame，没有 future geometry prediction 能力。

HERMES++ 的核心 claim：一个真正的 driving world model 应该同时具备这两个能力，并且两者要有 **deep interaction**（不止是 multi-task feature sharing）。语义 reasoning 要 guide 几何 evolution，几何约束要 ground 语言 generation。

这个 framing 其实暗合 LeCun 的 JEPA 思想（H-World, V-JEPA 2），还有 NVIDIA GR00T、World Labs、DeepMind Genie 2 这些 foundation world model 的方向 —— joint embedding + predictive + grounded。Driving 只是它的一个 concrete domain。

Reference: 
- World Models (Ha & Schmidhuber 2018): https://arxiv.org/abs/1802.02281
- V-JEPA 2: https://ai.meta.com/blog/v-jepa-2-world-model-analysis-robot-learning/
- DriveX (ICCV 2025): https://arxiv.org/abs/2509.12314

---

## 2. 整体架构（Fig. 2 解读）

整个 pipeline 可以拆成 6 个 stage：

```
[Multi-view Images {I_t^i}] 
        ↓ (Vision Encoder: OpenCLIP ConvNeXt-L)
[Multi-scale Perspective Features]
        ↓ (BEVFormer-style Spatial Cross-Attention)
[BEV Feature F_t^bev ∈ R^(w×h×c)]  w=h=180, c=256
        ↓ (Strided Conv + Pooling, ×4 downsample, ×4 channel)
[F_t^down ∈ R^(w/4 × h/4 × 4c)]
        ↓ (Flatten + Linear)
[BEV Tokens F_t ∈ R^(L_BEV × C)]  L_BEV = 45×45 = 2025, C = LLM dim
        ↓
   ┌────────────────────────────────────────┐
   │  Concat: [BEV tokens, Text tokens,     │
   │           World Queries Q^w]            │
   └────────────────────────────────────────┘
        ↓ (LLM: InternVL2 1.8B / 3.8B)
   ┌──────────────┬──────────────────────────┐
   │              │                          │
   ▼              ▼                          ▼
[Text Output]  [B_t (LLM-processed BEV)]  [Q^w_ε (enriched queries)]
                                                  │
                                                  ▼
                          ┌──────────────────────────────────┐
                          │ Current-to-Future Link           │
                          │  (Cross-Attn + Self-Attn + FFN)  │
                          │  with Textual Injection          │
                          │  and Ego Modulation              │
                          └──────────────────────────────────┘
                                                  │
                                                  ▼
                              [Future BEV {B_{t+i}}_{i=1..Δt}]  Δt=3
                                                  │
                                                  ▼
                          ┌──────────────────────────────────┐
                          │ Shared BEV-to-Point Render R       │
                          │  (3D Conv → SDF Neural Rendering)  │
                          └──────────────────────────────────┘
                                                  │
                                                  ▼
                          [Future Point Clouds {P_{t+i}}]
                          
Training-time:
                          ┌──────────────────────────────────┐
                          │ Frozen Geometric Feature Extractor│
                          │ (Sparse 3D Conv, self-supervised) │
                          └──────────────────────────────────┘
                                ↓ provides target V_t
                          [Implicit Geometric Regularization]
                          L_cos + L_gram on latent manifold
```

这个架构有几个值得品的设计：

**(a) BEV 作为 universal interface**：multi-view 6 个 camera 的 perspective view 在 BEV 空间统一，自然 preserve geometric topology 和 metric scale，又因为 flattened 成 1D token sequence 和 LLM 兼容。

**(b) Shared Render**：current frame 和 future frames 共用同一个 decoder R，确保 latent space 的 geometry-aware 表示一致性。

**(c) Frozen Geometric Feature Extractor**：在 training-time 提供一个 geometry-aware prior target $\mathbf{V}_t$，inference-time 不参与，零额外开销。

---

## 3. 关键模块详解

### 3.1 BEV Tokenizer（公式 2、3）

**BEVFormer-style spatial cross-attention**（公式 2）：

$$\mathbf{B}(x, y) = \sum_{i=1}^{N} \sum_{z \in \mathcal{H}} \mathrm{DA}\left(\mathbf{Q}(x, y), \mathbf{F}_i, \mathcal{P}_i(x, y, z)\right)$$

变量含义：
- $N$：相机数量（nuScenes 是 6 个 camera）
- $\mathbf{Q}(x, y) \in \mathbb{R}^C$：BEV grid 上位置 $(x, y)$ 处的 learnable query，对应 BEV 空间中一个 metric cell
- $\mathbf{F}_i$：第 $i$ 个 camera 的 perspective feature map（来自 ConvNeXt-L）
- $\mathcal{P}_i(x, y, z) = \pi_i(x, y, z)$：把 BEV 上的 3D 位置 $(x, y, z)$ 通过 camera intrinsics + extrinsics 投影到第 $i$ 个 camera image plane 上的 2D 位置
- $\mathcal{H}$：predefined height anchors 集合（在 $z$ 方向离散采样，比如 12 个 height level）
- $\mathrm{DA}(\cdot)$：multi-scale deformable cross-attention（参考 Deformable DETR），在每个 reference point 周围 learnable offsets 处采样

**Intuition**：每个 BEV cell 通过沿 height 维 sweep 一组 anchor，把每个 anchor 投影到所有 $N$ 个 camera 上，从对应 perspective location 采样特征再聚合。这就是把 perspective view "lift" 到 3D BEV 的 standard 操作。这种 lift 比直接用 depth estimation 更稳健，因为 deformable attention 学到的 offsets 能补偿 depth 不确定性。

**Downsampling + Flatten**（公式 3）：

$$\mathbf{F}_t = \phi\left(\mathrm{Flatten}(\mathbf{F}_t^{\mathrm{down}})\right) \in \mathbb{R}^{L_{\mathrm{BEV}} \times C}$$

其中 $\mathbf{F}_t^{\mathrm{down}} \in \mathbb{R}^{\frac{w}{4} \times \frac{h}{4} \times 4c}$，$L_{\mathrm{BEV}} = \frac{w}{4} \times \frac{h}{4} = 45 \times 45 = 2025$。

设计要点：
- $w = h = 180$，BEV 物理范围 $[-51.2\text{m}, 51.2\text{m}]$，每 cell ≈ 0.57m resolution
- 下采样 4× 把 token 数从 32400 砍到 2025，正好在 LLM context window 的 sweet spot
- 4× channel expansion（$c \to 4c$）保信息不丢，类似 MobileNet 的 inverted residual 思想

Table III 的 ablation 显示：×8 下采样 CD=1.781，×4 下采样 CD=1.436，Direct Query (一开始就 query 稀疏 BEV) CD=2.012。**先 dense 再下采样，远胜一开始就 sparse query** —— 这是 information-preserving compression 的胜利。

### 3.2 BEV-to-Point Render R（公式 4、5）

这是把 2D BEV feature 还原成 3D point cloud 的核心 module。两步：

**Volumetric lifting**：把 BEV feature $\mathbf{F}_t^{\mathrm{down}}$ 或 $\mathbf{B}_t$ 通过 nearest neighbor upsample 回 $180 \times 180$，再 reshape 成 $\hat{\mathbf{V}}_t \in \mathbb{R}^{w \times h \times z \times c'}$（$z = c' = 32$），即沿 height 维 expand 成 32 个 level。然后 3D conv refine。

**SDF-based neural rendering**：对每条 LiDAR ray $\mathbf{r}_k$（从 LiDAR 中心 $\mathbf{o}$ 沿方向 $\mathbf{t}_k$），discretize 成 $n$ 个 sample point $\mathbf{p}_i = \mathbf{o} + d_i \mathbf{t}_k$，深度 $0 \leq d_1 < d_2 < \cdots < d_n$。

对每个 sample point trilinear interpolate 出 local feature $\mathbf{f}_i$，shallow MLP $\phi_{\mathrm{SDF}}$ 预测 SDF 值：
$$s_i = \phi_{\mathrm{SDF}}(\mathbf{p}_i, \mathbf{f}_i)$$

Rendered depth（公式 4）：
$$\tilde{d}(\mathbf{r}_k) = \sum_{i=1}^{n} w_i d_i, \quad w_i = T_i \alpha_i$$

其中 transmittance $T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)$，opacity $\alpha_i$ 由 SDF 的 gradient 计算（公式 5）：
$$\alpha_i = \max\left(\frac{\sigma_\tau(s_i) - \sigma_\tau(s_{i+1})}{\sigma_\tau(s_i)}, 0\right)$$

这里 $\sigma_\tau(x) = (1 + e^{-\tau x})^{-1}$ 是带 learnable parameter $\tau$ 的 sigmoid。

**Intuition**：这是 NeuS [78] / VolSDF 风格的 volume rendering。SDF 在 surface 处 zero-crossing，通过 sigmoid 把 SDF 转成 density-like 量。$\alpha_i$ 本质是 "ray 在第 $i$ 个 sample point 处碰到 surface 的概率"，由相邻点 SDF 差分除以当前点 SDF 给出 —— 类似 signed distance to occupancy conversion。

这套 differentiable rendering 让 LLM 的 latent 输出可以直接被 supervise 到 point cloud level，端到端可微。

Reference:
- NeuS: https://arxiv.org/abs/2106.10689
- UniPAD (CVPR 2024): https://arxiv.org/abs/2312.13724
- PonderV2 (TPAMI 2025): https://arxiv.org/abs/2402.00806

### 3.3 World Queries（公式 6）—— 知识 transfer 的载体

这是整篇 paper 最有意思的设计。它要做的事情是：**让 generation branch 能"读取" LLM 在 understanding 过程中积累的 semantic context 和 world knowledge**。

形式化：
$$\mathbf{Q}^w = \phi\left(\mathrm{Concat}_{i=1}^{\Delta t}(\mathbf{Q} \oplus \mathbf{e}_{t+i}) \oplus \mathbf{F}\mathbf{E}\right) \in \mathbb{R}^{(\Delta t \times n) \times C}$$

变量：
- $\Delta t = 3$：预测 horizon 是 3 个 future frame
- $n = 4$：每个 future time step 用 4 个 query token
- $\mathbf{Q} \in \mathbb{R}^{n \times 4c}$：base spatial queries，通过 **adaptive max pooling** over $\mathbf{F}_t^{\mathrm{down}}$ 得到（保留最 salient 的 spatial 信息）
- $\mathbf{e}_{t+i} \in \mathbb{R}^{1 \times 4c}$：ego-motion embedding，由 MLP 处理 ego-motion 参数得到
- $\mathbf{F}\mathbf{E} \in \mathbb{R}^{\Delta t \times 4c}$：learnable frame embeddings，编码 temporal order
- $\oplus$：element-wise addition with broadcasting（spatial 维和 temporal 维分别 broadcast）
- $\phi$：shared linear layer，把 $4c$ 维投到 LLM hidden dim $C$

这些 queries 和 BEV tokens、text tokens 一起 concat 进 LLM 的 input sequence。**关键 trick**：queries 是 causal attention 的一部分，所以它们能 attend 到所有前面的 BEV tokens 和 text tokens，自然 aggregate semantic context。同时，由于 LLM 在大规模 pretraining 中积累了 world knowledge（交通规则、物体行为、因果 reasoning），处理这些 queries 的过程就把 world knowledge "写" 进 query representation 里，得到 enriched queries $\mathbf{Q}_\epsilon^w$。

**为什么 world queries 必须经过 LLM？** Table VII 的 ablation 给出答案：

| Setting | 3s CD ↓ | CIDEr ↑ |
|---------|---------|---------|
| (a) 不用 Q^w，只用 Textual Injection | 1.634 | 0.703 |
| (b) Q^w bypass LLM | 1.526 | 0.709 |
| (c) Q^w through LLM (proposed) | **1.436** | **0.720** |

Setting (b) 把 Q^w 直接和 BEV feature 交互，不经过 LLM reasoning。虽然比 (a) 好（说明 BEV-aware query initialization 有用），但远不如 (c)。这证明 LLM 的 reasoning 和 world knowledge 是关键的。

更妙的是：**generation loss 的梯度通过 Q^w 反传到 LLM，反过来提升 understanding 能力**。这是真正 bidirectional 的 task synergy，而不是简单的 shared backbone。

### 3.4 Current-to-Future Link（公式 7、8）

World queries 提供 sparse semantic prior（每 future frame 仅 4 个 token），但要生成 dense future BEV feature，需要把 current BEV $\mathbf{B}_t$ propagate 到未来。这就是 Current-to-Future Link 的作用。

**Textual Injection**：从 LLM-processed text tokens 通过 average pooling + linear projection 得到 $\hat{\mathbf{T}} \in \mathbb{R}^{k \times C}$（$k$ 是 pooled token 数）。

**Cross-attention**（公式 7）：
$$\mathbf{X}_{\mathrm{cross}}^{(l)} = \mathbf{X}^{(l)} + \mathbf{CrossAttn}(\mathbf{LN}(\mathbf{X}^{(l)}), [\mathbf{Q}_{\epsilon, i}^w; \hat{\mathbf{T}}])$$

变量：
- $\mathbf{X}^{(l)}$：第 $l$ 个 block 的输入 feature，初始化为 $\mathbf{B}_t$（current BEV）
- $\mathbf{LN}$：LayerNorm
- $[\mathbf{Q}_{\epsilon, i}^w; \hat{\mathbf{T}}]$：第 $i$ 个 future time step 的 enriched world queries 和 text embedding 的 concatenation，作为 cross-attention 的 Key 和 Value

这意味着：**future BEV 的每一步预测都被当前 BEV 的几何 context、world queries 的 semantic prior、text embedding 的 reasoning guidance 三者共同 condition**。

**Ego Modulation**（公式 8）：
$$\mathrm{EM}(\mathbf{x}) = (\gamma + 1) \odot \mathrm{LN}(\mathbf{x}) + \beta$$

ego-motion for time $t + i$ 通过 MLP + Tanh 得到 $\gamma$ 和 $\beta$。两个 modulation vector 在训练初期 zero-initialized，保证初期是 identity transform，训练稳定。

设计上 EM **只 applied 到 self-attention 和 feed-forward，不 applied 到 cross-attention**。理由：cross-attention 应该专注 semantic aggregation，不应该被 ego-motion 干扰。

**这个设计本质是 DiT 的 adaLN（Adaptive Layer Norm）[1] 在 driving world model 的应用**。在 video diffusion 里 adaLN 用来 condition timestep，这里用来 condition ego-motion。效果是：把 ego-motion 从 scene dynamics 解耦，static background 的 motion（车前进时背景"向后"流）和 dynamic object 的 motion（车自己开动）分开建模。

Table V 的 ablation 印证了这些设计的有效性：

| Configuration | 3s CD ↓ |
|---------------|---------|
| w/o Link | 2.377 |
| + Simple Link (3 vanilla attn layers) | 1.542 |
| + Textual Injection | 1.506 |
| + Ego Modulation | 1.442 |
| + More blocks (3→6 layers) | **1.436** |

注意：w/o Link 是直接 copy $\mathbf{B}_t$ + add ego-motion，CD 2.377 —— 完全无法 model scene evolution，说明简单的 motion compensation 远远不够。

Reference:
- DiT (Peebles & Xie): https://arxiv.org/abs/2212.09748
- adaLN Zero init: 上面 paper 的 Section 3.2

### 3.5 Joint Geometric Optimization（公式 9、10、11、12）—— 隐藏的 latent regularizer

这是 HERMES++ 相对 conference version HERMES 的主要新增 contribution 之一。核心 motivation：仅靠 explicit point cloud supervision（render loss）会让 latent representation 走 "捷径" —— 沿 camera projection rays 形成 ray-shaped artifacts，center 处响应异常高（Fig. 5a），因为这个 representation 在 perspective view 下渲染时已经能 minimize loss，但 latent 本身没有真正学到 intrinsic 3D geometry。

**Explicit constraint**（公式 9）：
$$\mathcal{L}_{\mathrm{render}} = \sum_{i=0}^{\Delta t} \lambda_i \frac{1}{N_i} \sum_{k=1}^{N_i} |d(\mathbf{r}_k) - \tilde{d}(\mathbf{r}_k)|$$

变量：
- $i$：frame index（0 表示当前 frame，1~3 表示未来 frame）
- $\lambda_i = 1 + 0.5i$：empirically 设的，长期预测权重更高（强调 long-horizon accuracy）
- $N_i$：第 $i$ 个 frame 的 LiDAR ray 数量
- $d(\mathbf{r}_k), \tilde{d}(\mathbf{r}_k)$：第 $k$ 条 ray 的 ground-truth 和 predicted depth

L1 loss 比 L2 对 outlier 更鲁棒，这是 point cloud 渲染的 standard 选择。

**Implicit regularization via frozen geometry extractor**：

Pretrain 阶段：用 sparse 3D convolution encoder + Render R 做 self-supervised point cloud reconstruction（voxelize ground-truth $P_t$ → encoder → render → reconstruct）。这个 pretrained encoder 学习到 geometry-aware representation。

Main training 阶段：把 encoder **frozen**，用它处理 voxelized ground-truth $P_t$ 得到 target geometry-aware feature $\mathbf{V}_t \in \mathbb{R}^{w \times h \times z \times c'}$。然后强制 predicted volumetric feature $\hat{\mathbf{V}}_t$ 对齐 $\mathbf{V}_t$，通过两个 loss：

**Cosine similarity loss**（公式 10）—— voxel-wise 局部一致性：
$$\mathcal{L}_{\mathrm{cos}} = 1 - \frac{1}{whz} \sum_{i,j,k} \frac{\hat{\mathbf{V}}_t(i, j, k) \cdot \mathbf{V}_t(i, j, k)}{\|\hat{\mathbf{V}}_t(i, j, k)\|_2 \|\mathbf{V}_t(i, j, k)\|_2}$$

$(i, j, k)$ 是 voxel grid 三维 index。这相当于每个 voxel 的 feature vector 方向对齐。注意是 cosine 不是 L2，所以只约束方向不约束 magnitude，给 latent 一定自由度。

**Gram loss**（公式 11、12）—— global structural pattern consistency：

对 $\mathbf{V}_t$ 沿三个正交轴 pool 得到 $\mathbf{V}_t^{HW}, \mathbf{V}_t^{HZ}, \mathbf{V}_t^{WZ}$，分别代表三个 perspective 的 projection。每个 perspective 的 Gram matrix：
$$\mathbf{G}_t^d = \mathbf{V}_t^d {\mathbf{V}_t^d}^T \in \mathbb{R}^{N_d \times N_d}$$

$N_d$ 是 perspective $d$ 上的 spatial token 数。Gram matrix 捕捉 feature 之间的 pairwise correlation。Loss：
$$\mathcal{L}_{\mathrm{gram}} = \frac{1}{3} \sum_d \|\mathbf{G}_t^d - \hat{\mathbf{G}}_t^d\|_F^2, \quad d \in \{HW, HZ, WZ\}$$

$\|\cdot\|_F$ 是 Frobenius norm。

**Intuition**：Gram matrix loss 是 Neural Style Transfer（Gatys et al. 2015 [2]）的经典技巧 —— 在那里它捕捉 "style"（texture、color correlation），与 "content"（spatial arrangement）互补。这里被 reinterpret 成 "global structural pattern"：哪些 voxel 之间应该 correlate，相比 voxel-wise 局部对齐更全局、更 robust。

Table IV ablation 印证：

| $\mathcal{L}_{\cos}$ | $\mathcal{L}_{\gram}$ | 3s CD ↓ |
|---|---|---|
| ✗ | ✗ | 1.637 |
| ✓ | ✗ | 1.441 |
| ✗ | ✓ | 1.544 |
| ✓ | ✓ | **1.436** |

两者结合效果最好，cosine 单独效果最显著，gram 是补充。Fig. 5 的可视化很直观：只用 explicit loss 时 BEV feature 出现明显的 ray-shaped artifacts 和 center bias；加上 implicit regularization 后，feature 变得 spatially compact 且符合 intrinsic geometry。

Reference:
- Neural Style Transfer (Gatys et al. 2015): https://arxiv.org/abs/1508.06576
- perceptual loss (Johnson et al. 2016): https://arxiv.org/abs/1603.08155

### 3.6 Total Loss（公式 13、14、15）

Language modeling loss（next-token prediction）：
$$\mathcal{L}_{\mathrm{lang}} = -\sum_{i=1}^{L_{\mathrm{text}}} \log P(\mathbf{T}_i | \mathbf{F}_t, \mathbf{T}_1, \dots, \mathbf{T}_{i-1}; \Theta)$$

标准 autoregressive cross-entropy。$\Theta$ 是 LLM 参数（实际训练时用 LoRA）。

Generation loss：
$$\mathcal{L}_{\mathrm{gen}} = 10\mathcal{L}_{\mathrm{render}} + \mathcal{L}_{\mathrm{cos}} + \mathcal{L}_{\mathrm{gram}}$$

注意 $\mathcal{L}_{\mathrm{render}}$ 的权重是 10，这表明 render loss 在数值 scale 上比 cosine/gram 小很多，需要放大才能 dominate optimization。

Total：
$$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{lang}} + \mathcal{L}_{\mathrm{gen}}$$

简单相加，没有用 uncertainty-based dynamic weighting（如 Kendall et al. 2018 [3]），略显粗糙，但实证 work。

---

## 4. Three-stage Training Pipeline

Table I 给出 training schedule，是典型的 progressive multi-stage tuning：

**Stage 1: Geometry-aware pretraining**（12+6 epochs，lr=2e-4）
- Sparse 3D encoder self-supervised on point cloud reconstruction（voxelized GT → encoder → render → reconstruct）
- 这个 pretrained encoder 之后 frozen 用作 implicit regularizer 的 target source

**Stage 2: Vision-language alignment + refinement**（3+6 epochs，lr=2e-4 → 4e-4，batch=128）
- **Sub-phase 2a**：Pretrain tokenizer + Render 用 current point cloud reconstruction（supervised by $\mathcal{L}_{\mathrm{render}} + \mathcal{L}_{\mathrm{cos}}$）
- **Sub-phase 2b**：Vision-language alignment，只训 LLM projector，用 masking-based augmentation：把 captions stitch 到 unmasked views 上，扩充 image-text pairs 从 70K 到 200K
- **Sub-phase 2c**：Refinement，全部参数 unfrozen，LLM 用 LoRA，数据是 NuInteract [84]（dense captions）+ OmniDrive-nuScenes [15]（scene descriptions + QA）

**Stage 3: Unified training**（36 epochs，lr=4e-4，batch=128）
- 加入 Current-to-Future Link
- 数据：nuScenes keyframes + descriptions + conversation annotations
- Supervision: $\mathcal{L}_{\mathrm{gram}} + \mathcal{L}_{\mathrm{cos}} + \mathcal{L}_{\mathrm{render}} + \mathcal{L}_{\mathrm{lang}}$
- 测试 unified 能力

这个 schedule 的 insight：
- Geometry prior 必须 first 建立（Stage 1），因为它是后续所有 geometry-aware supervision 的 anchor
- Vision-language alignment 在 Stage 2 解决 modality gap
- Stage 3 才让两个 task 真正联合优化，让 semantic 和 geometric 互相 guide

---

## 5. 关键实验结果深度解读

### 5.1 Main comparison（Table II）

最 striking 的数字：

**3s future point cloud generation (CD ↓)**：
- ViDAR (CVPR'24): 1.73
- 4D-Occ (CVPR'23): 2.11
- DriveX (ICCV'25): 1.10
- HERMES (conference, 1.8B): 1.17
- **HERMES++ (1.8B): 1.01**
- **HERMES++ (3.8B): 0.97**

相比 DriveX，HERMES++ 3.8B 降 0.13 CD（11.8%），相比 conference version 降 13.7%。

**Scene understanding (CIDEr ↑)**：
- LLaVA-OneVision (7B): 0.732
- Omni-L (7B, aux sup: 3D Box, Lane): 0.686
- Omni-Q (7B, aux sup): 0.732
- OmniDrive-2D (7B, aux sup): 0.671
- OmniDrive-BEV (7B, aux sup): 0.595
- ORION (7B, aux sup): 0.635
- HERMES (1.8B, no aux sup): 0.741
- **HERMES++ (1.8B, no aux sup): 0.749**
- **HERMES++ (3.8B, no aux sup): 0.772**

注意：HERMES++ 用更小的 LLM（1.8B vs 7B）+ 不用 auxiliary supervision（OmniDrive 系列用 3D Box + Lane detection 监督）就能超越。BEV representation 的几何 inductive bias 起到关键作用。

### 5.2 BEV vs Multi-view input（Fig. 4）

非常 illustrative 的 case study。等量 token 输入（2532 tokens），两种方法：
- Multi-view: 直接 CLIP encode 每个 camera view，resize 到 2532 tokens
- BEV: 用 BEV tokenizer + 下采样，得到 2532 tokens

Quantitative：3s CD 差距 32%（multi-view 显著差），但 METEOR 几乎一样（0.001 差距）。

Qualitative：multi-view baseline 把 road marking 误读为左转（hallucinate left turn），BEV 正确预测直行。

**Intuition**：multi-view token 在 LLM 处理时被当作 sequence，spatial topology collapse。LLM 不知道哪些 token 对应哪些 view、view 之间什么关系、什么是 road marking 什么是 lane boundary。BEV 把所有 view 在统一 metric 空间 fuse，每 token 有明确 spatial location，LLM 能 leverage 这种 structure。

### 5.3 Generalization across tasks

**NuScenes-QA**（Table IX）：
- LiDAR-based CenterPoint+MCAN: 59.5%
- Camera-based Omni-Q (CVPR'25): 59.2%
- **HERMES++ (camera-only): 61.3%**

超过 LiDAR-based 方法 1.8% —— 这是个 impressive result，证明 BEV representation 学到的 geometry 信息和 LiDAR feature 相当甚至更好（因为融合了 semantic）。

**DriveLM leaderboard**（Table X）：
- DriveLM: FS=0.50
- Team NVIDIA (CVPRW'24 winner): FS=0.59
- Omni-Q: FS=0.58
- FSDrive: FS=0.57
- **HERMES++: FS=0.59** (Match=0.43，超过所有 baseline)

特别是 Match metric（detection-and-reasoning alignment）0.43，超过 Omni-Q 0.37 大截。这说明模型真正在做 grounded reasoning，不是用 language prior 蒙。

**Motion planning**（Table XI）：
- UniAD: L2=0.46m, Collision=0.37%
- OmniDrive++: L2=0.33m, Collision=0.30%
- ORION (ICCV'25): L2=0.34m, Collision=0.37%
- **HERMES++: L2=0.37m, Collision=0.29%**

HERMES++ 的 collision rate 是最低的（0.29%），甚至比 ORION 还低 0.08%。注意：HERMES++ 并未用 motion planning supervision 训练，只是 attach 一个 lightweight MLP head 到 world queries 上做 trajectory regression。这说明：**通过优化 future scene generation，模型 inherently 学到了 driving dynamics 和 actionable priors**。这是一个 emergent capability 的例子。

### 5.4 LLM 架构和 scale 的影响（Table XII）

**Architecture**（用 25% training data 测试）：
- Llama-3.2: 3s CD 1.533, CIDEr 0.700
- Qwen3: 3s CD 1.521, CIDEr 0.696
- **InternVL2: 3s CD 1.436, CIDEr 0.720**

InternVL2 最好，可能因为它的 vision-language alignment 设计（large vision encoder + connector）对 BEV-based tokenization 更友好。

**Scale**：
- 0.8B: 3s CD 1.434, CIDEr 0.708
- 1.8B: 3s CD 1.436, CIDEr 0.720
- 3.8B: 3s CD **1.255**, CIDEr **0.742**

3.8B vs 0.8B：generation error 降 12.5%。这说明 **world knowledge 和 reasoning capability 是 scalable 的 inductive bias**。Driving scene 高度 compositional（vehicle + pedestrian + road + traffic light + causal interaction），更大 LLM 携带的 prior 更丰富。

---

## 6. 关键 Ablation 背后的 Intuition

### 6.1 World Queries 的初始化策略（Table VIIIa）

| Method | 3s CD ↓ |
|--------|---------|
| Random Init | 1.448 |
| Attention Pool | 1.438 |
| Cross Attn | 1.442 |
| Avg Pool | 1.444 |
| **Max Pool** | **1.436** |

**Max Pool 胜出 cross-attention** —— 这违反"learnable 一定比 hand-crafted 好"的直觉。但 BEV 是 sparse representation（绝大部分 voxel 是空背景），max pooling 自然 select 最 salient feature（object occupancy），相当于 "free" foreground detection。Avg pool 把背景噪声稀释信号，attention pool 在 limited data 下难训。这是 strong inductive bias 在低 data regime 的胜利。

### 6.2 World queries 数量 n（Table VIIIb）

| n | 3s CD ↓ | CIDEr ↑ |
|---|---------|---------|
| 0 | 1.478 | 0.716 |
| 1 | **1.419** | 0.712 |
| 2 | 1.431 | 0.717 |
| 4 | 1.436 | 0.720 |
| 8 | 1.430 | 0.719 |

有趣：n=1 在 generation 上最好，但 n=4 在 understanding 上最好（CIDEr 0.720）。作者选 n=4 做 trade-off。这暗示 **generation 不需要太多 query token**（1 个就够），understanding 需要 more representational capacity（因为要 capture 丰富 semantic）。这是个 future work 机会：decoupled query allocation per task。

### 6.3 Generation horizon（Table VIIIc）

| Horizon | 3s CD ↓ |
|---------|---------|
| 0-1s | 0.550 (1s) |
| 0-2s | 1.147 (2s) |
| 1-3s (no current) | 1.476 (3s) |
| 0+3s (discontinuous) | 1.677 (3s) |
| 0-3s (continuous) | 1.436 (3s) |

**Discontinuity 是杀手**：0+3s（跳过 1s, 2s）CD 1.677，远不如连续 0-3s 的 1.436。这印证 driving dynamics 是 Markovian，中间 frame 是 future prediction 的 inductive bridge。直接外推 3 秒，noise 放大，optimization landscape 恶化。

### 6.4 Task unification 的 synergy（Table VI）

| Setting | 3s CD ↓ | CIDEr ↑ |
|---------|---------|---------|
| Separated unification | 1.634 | 0.703 |
| Joint unification | **1.436** | **0.720** |

"Separated unification" 指：share visual tokenizer，但 understanding 和 generation 各走各的 branch，无 interaction。结果 generation CD 差 0.198，understanding CIDEr 差 0.017。

**双向收益**：
- Semantic guides geometry：CD 降 0.198
- Geometry grounds semantic：CIDEr 升 0.017（smaller gain 但 still positive）

Geometry 益处更大，可能因为 generation 是更难、更 under-determined 的任务，更需要 semantic prior 约束。Understanding 是相对 well-defined 的任务（VQA 有明确 ground truth），geometric grounding 的边际增益较小。

---

## 7. Cross-reference 与 broader context

### 7.1 World Model 谱系

HERMES++ 的定位：
- **Latent world model**（Ha & Schmidhuber, Dreamer 系列）：HERMES++ 也是 latent-based，但 latent 不是 RSSM 而是 BEV feature
- **Generative world model**（GAIA-1, DriveDreamer, Vista）：HERMES++ 不做 image generation，做 point cloud generation，保留 explicit geometry
- **Predictive + Joint Embedding**（V-JEPA 2, I-JEPA）：HERMES++ 的 implicit geometric regularization 思想类似 JEPA 的 latent prediction，但 supervise 不同
- **VLA world model**（GR00T, OpenVLA, ImpromptuVLA [59]）：HERMES++ 不是 VLA，但 motion planning 的 emergent ability 暗示 VLA potential

Reference:
- DreamerV3: https://arxiv.org/abs/2301.04105
- I-JEPA: https://arxiv.org/abs/2301.08243
- GR00T: https://developer.nvidia.com/groot

### 7.2 BEV representation 的角色

BEV 在 autonomous driving 已经是 de facto standard（BEVFormer [70, 71], BEVFusion [68], UniAD [97], VAD [98]）。HERMES++ 把 BEV 用作 LLM 的 visual token，相当于在 LLM 和 3D 世界之间建了一座 bridge。这个 idea 之前在 BEVWorld [28]、OmniDrive [15] 也有探索，但 HERMES++ 是 first 把 BEV 同时用于 understanding 和 future generation 的 unified interface。

Reference:
- BEVFormer: https://arxiv.org/abs/2203.17070
- UniAD: https://arxiv.org/abs/2212.10156
- BEVWorld: https://arxiv.org/abs/2407.05679

### 7.3 LLM + 3D 的相关 work

- 3D-LLM (NeurIPS'23): 用 3D features 喂 LLM
- LL3DA (ICCV'23): 3D scenes + LLM
- LiDAR-LLM [90]: LiDAR + LLM，用 range image
- OmniDrive [15]: 用 Q-Former3D 把 BEV compress 给 LLM
- DriveLM [16]: graph-based VQA
- ImpromptuVLA [59]: open-weights VLA

HERMES++ 区别于这些 work 的核心点：**它让 LLM 直接参与 future generation 的 conditioning，而不只是 static scene understanding**。World queries 通过 LLM 是关键。

Reference:
- 3D-LLM: https://arxiv.org/abs/2307.12981
- LiDAR-LLM: https://arxiv.org/abs/2403.09501

### 7.4 Differentiable Rendering 的灵感来源

HERMES++ 的 SDF-based rendering借鉴自：
- NeuS [78] (NeurIPS'21): SDF + volume rendering
- UniPAD [76] (CVPR'24): autonomous driving 的 universal pre-training with rendering
- PonderV2 [77] (TPAMI'25): UniPAD 的 extend version

这套 differentiable rendering 让 point cloud supervision 端到端 backprop 到 latent space，避免了离散 point cloud 操作（如 Chamfer distance 直接优化）的不稳定性。

### 7.5 adaLN-Zero 的应用

Ego Modulation 公式 $\mathrm{EM}(\mathbf{x}) = (\gamma + 1) \odot \mathrm{LN}(\mathbf{x}) + \beta$ 完全复刻 DiT [1] 的 adaLN-Zero。在 DiT 里 conditioning signal 是 timestep，在 HERMES++ 里是 ego-motion。这种 affine transformation 在 LayerNorm 之后做，相当于 condition the feature 的 scale 和 shift，是 high-capacity 的 conditioning 方式。

---

## 8. 一些 Critical Observations 与 potential limitations

### 8.1 关于 "Unified" 的程度

虽然 paper claim "unified framework"，实际架构仍有 specialization：
- BEV tokens 给 LLM 做 understanding
- World queries 给 generation 做 conditioning
- 两者 share LLM 但 function 上是 separate input slot

这其实更像 "shared backbone with task-specific interface"，而非完全 unified token space（像 GPT-4o 那种真正 unified multimodal token）。但这不影响 method 的 effectiveness。

### 8.2 3s prediction horizon 的局限

3s 在 real autonomous driving 是一个相对短的 horizon（高速行驶 3s ≈ 100m）。实际 deployment 需要 5-10s 甚至更长。Long horizon 预测的 error accumulation 是 world model 的 fundamental challenge。HERMES++ 没在 paper 里探索 >3s 的表现。

### 8.3 Camera-only 假设

HERMES++ 是 camera-only（不用 LiDAR 输入），但 supervision 是 LiDAR point cloud。这意味着：
- Pros: inference 时不需要 LiDAR，sensor suite 成本低
- Cons: 监督信号依赖 LiDAR 数据，仍是 supervised 而非 truly self-supervised

未来的方向可能是 video + text supervision（无 LiDAR），类似 Vista [2] 的路线。

### 8.4 Single-step vs Autoregressive

HERMES++ 的 future generation 是 single-shot（一次预测所有 3 个 future frame），不是 autoregressive（不基于已生成 frame 继续外推）。前者快但 error 不会 exponentially accumulate；后者理论上能 extend horizon 但实际容易 drift。在 long horizon scenario，autoregressive 可能更优，但需要 careful design 防止 drift。

### 8.5 Open-loop evaluation 的局限

Table XI 的 motion planning 是 open-loop（ego-vehicle trajectory 已知，只预测 self trajectory）。Open-loop 和 closed-loop 的 gap 很大（参考 BEV-Planner [95] 的 critique）。HERMES++ 的 emergent planning ability 在 closed-loop 是否仍然成立是 open question。

---

## 9. 关联到你（Karpathy）的 work 和 view

你自己在多次 talks 里强调过：
1. **World model 应该是 "predict the next token in latent space" 而非 pixel space**（参考你的 "State of GPT" talk）。HERMES++ 走的正是 latent prediction 路线 —— predict future BEV feature，再 render 出 point cloud。
2. **Autoregressive transformer 是 universal computation substrate**（"Software 2.0" 的延伸）。HERMES++ 用 LLM 处理 BEV token，相当于把 driving scene 也 tokenize 进 token stream。
3. **End-to-end 是方向**。HERMES++ 的 motion planning emergent ability 印证了 e2e 的 power —— 不显式 design planner，让 model 自己 internalize driving dynamics。

你的 nanoGPT 教程（https://github.com/karpathy/nanoGPT）强调 simplicity。HERMES++ 的 LLM 用的是 InternVL2，没有 architecture 改动，只在 input/output 层面 adapt，这符合你的 minimalist 哲学。

---

## 10. Future direction 思考

基于 HERMES++ 的发现，未来几个有潜力方向：

**(a) Token-level unified world model**：把 BEV、text、action、future frame 全部 unified 到 single token stream，真正 GPT-style autoregressive。Epona [37]、DrivingGPT [35] 已经开始这个方向。

**(b) Self-supervised geometry learning**：去掉 LiDAR supervision，用 multi-view photometric consistency + temporal consistency 学 geometry。Vista [2] 已经探索。

**(c) Long-horizon with hierarchical temporal abstraction**：用 hierarchy of timescale（秒级、分钟级）做 long-horizon prediction，避免单 step 的 error accumulation。

**(d) Closed-loop integration with planner**：把 world model 放到 closed-loop control pipeline，让 prediction 真正 affect action。Doe-1 [23] 探索 closed-loop 但只 2D。

**(e) Multi-modal generation**：扩展到 RGB + LiDAR + depth 多模态 future generation，做 comprehensive scene simulation。这是 paper Section VII 提到的 limitation。

**(f) Test-time scaling**：用 chain-of-thought / tree-search 在 inference time enhance prediction。FSDrive [54] 已开始 explore spatio-temporal CoT。

---

## 11. 总结：HERMES++ 的真正贡献

回到本质，HERMES++ 的核心 insight 可以浓缩成几条：

1. **BEV 是 LLM 和 3D 物理世界之间的 natural bridge**。它把 multi-view perspective information consolidate 到统一 metric 空间，既 preserve geometric topology 又 compress 到 LLM-compatible token 数量。

2. **World queries 是 knowledge transfer 的关键载体**。让 queries 走过 LLM，pretrained world knowledge 和 reasoning capability 自然注入；让 queries 参与 forward pass，generation gradient 反向 improve understanding。这是 bidirectional synergy。

3. **Implicit geometric regularization 是 latent geometry alignment 的有效手段**。Frozen geometry-aware encoder 提供 target，cosine + Gram loss 在 voxel-wise 和 global pattern 两个 level 对齐，避免 explicit supervision 的 "shortcut" 问题。

4. **Ego Modulation 通过 adaLN 解耦 ego-motion 和 scene dynamics**。借鉴 DiT 的 conditioning mechanism，让 model 区分 "background 因为我前进而流" 和 "object 因为自己而动"。

5. **Model scaling 在 unified framework 下持续收益**。3.8B vs 0.8B 在 generation error 上降 12.5%，说明 LLM 内在的 world knowledge 是 scalable inductive bias。

这条路走下去，driving world model 离你常说的 "general world model" 的圣杯又近了一步 —— 一个 model 同时 perceive、understand、predict、act，all grounded in physical reality。

---

## References (主要 external links)

- World Models (Ha & Schmidhuber 2018): https://arxiv.org/abs/1802.02281
- DreamerV3: https://arxiv.org/abs/2301.04105
- V-JEPA 2 (Meta AI): https://ai.meta.com/blog/v-jepa-2-world-model-analysis-robot-learning/
- I-JEPA: https://arxiv.org/abs/2301.08243
- HERMES original (ICCV'25): https://arxiv.org/abs/2507.04598 (估计 link)
- DriveX (ICCV'25): https://arxiv.org/abs/2509.12314
- ViDAR (CVPR'24): https://arxiv.org/abs/2404.07991
- OccWorld (ECCV'24): https://arxiv.org/abs/2401.14162
- Vista (NeurIPS'24): https://arxiv.org/abs/2410.01476
- GAIA-1 (Wayve): https://arxiv.org/abs/2309.17080
- DriveDreamer (ECCV'24): https://arxiv.org/abs/2309.09777
- BEVFormer (ECCV'22): https://arxiv.org/abs/2203.17070
- BEVFormer v2: https://arxiv.org/abs/2211.17119
- UniAD (CVPR'23, best paper): https://arxiv.org/abs/2212.10156
- BEVFusion: https://arxiv.org/abs/2205.13542
- InternVL2: https://arxiv.org/abs/2404.16821
- LLaVA: https://arxiv.org/abs/2304.08485
- OmniDrive (CVPR'25): https://arxiv.org/abs/2406.05422
- DriveLM (ECCV'24): https://arxiv.org/abs/2402.04832
- NeuS: https://arxiv.org/abs/2106.10689
- UniPAD: https://arxiv.org/abs/2312.13724
- PonderV2: https://arxiv.org/abs/2402.00806
- DiT (Peebles & Xie, ICCV'23): https://arxiv.org/abs/2212.09748
- Neural Style Transfer (Gatys et al.): https://arxiv.org/abs/1508.06576
- GR00T (NVIDIA): https://developer.nvidia.com/groot
- Doe-1: https://arxiv.org/abs/2412.09627
- Epona: https://arxiv.org/abs/2503.06624
- FSDrive (NeurIPS'25): https://arxiv.org/abs/2502.03621
- ORION (ICCV'25): https://arxiv.org/abs/2503.19025
- ImpromptuVLA (NeurIPS'25): https://arxiv.org/abs/2502.20132
- BEVWorld: https://arxiv.org/abs/2407.05679
- DriveGPT4: https://arxiv.org/abs/2310.01415
- DriveVLM: https://arxiv.org/abs/2402.14168
- nanoGPT (Karpathy): https://github.com/karpathy/nanoGPT

希望这个详尽讲解能帮你 build intuition。如果对某个 module（比如 SDF rendering 的细节、Gram loss 的 spectral interpretation、或者 world queries 的 attention pattern）想 dive deeper，告诉我，我可以进一步展开。
