---
source_pdf: Cosmos 3 Omnimodal World Models for Physical AI.pdf
paper_sha256: 22d7b487878936a7bd49a11d2b714afb69d27759051d88763aea8643244944c5
processed_at: '2026-08-03T17:29:21-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，那我用大白话重新讲一遍，重点是让你听懂**为什么**要这么设计，而不只是列技术细节。

---

## 这篇 paper 到底在解决什么问题

想象你要造一个 home robot，让它收拾餐桌。现在的做法是给它拼一套"积木"：

- 一个 **VLM**（vision-language model）负责看场景、理解"餐桌上有盘子"
- 一个 **video generator** 负责想象"如果我抓这个盘子，未来会发生什么"
- 一个 **policy model** 负责输出"机器人应该怎么动关节"
- 一个 **world model** 负责模拟"这个动作之后世界会变成什么样"

这四五个模型互相之间**不共享 representation**，要把它们的输出拼来拼去，既慢又笨。NVIDIA 的核心论点是：**这玩意儿本来就该是一个模型**。Understanding 和 generation 是一枚硬币的两面——你要理解"盘子会被抓起来"，本质上你脑子里已经在 simulate 这个过程了；你要 generate "抓盘子的视频"，你脑子里得先 understand 盘子是什么、怎么动。

所以 Cosmos 3 的赌注是：**一个网络，同时干 understanding 和 generation，五种 modality（language / image / video / audio / action）全包**。这不是简单的 model ensemble，是 architectural level 的统一。

---

## 核心架构：一个大脑，两个子系统

Cosmos 3 用了个叫 **Mixture-of-Transformers (MoT)** 的设计。你可以把它想成一个 transformer 里有**两个独立的"小脑"**，但共享一个"注意力系统"。

- **Reasoner tower**：负责"想"。走 autoregressive（像 GPT 那样），吃 language tokens 和 ViT-encoded vision tokens，输出 next-token prediction。它干的是 VLM 的活——理解场景、回答问题、做 spatial reasoning。
- **Generator tower**：负责"画"。走 diffusion（像 Stable Diffusion 那样），吃 VAE-encoded vision tokens + audio tokens + action tokens，输出 denoised pixels/sounds/actions。它干的是 video generator / world model / policy model 的活。

**关键 trick 是它们怎么交互**：

$$O_{AR} = \text{Attn}_{causal}(Q_{AR}, K_{AR}, V_{AR})$$

$$O_{DM} = \text{Attn}_{full}\big(Q_{DM}, [K_{AR}; K_{DM}], [V_{AR}; V_{DM}]\big)$$

翻译成人话：

- **Reasoner 只看自己**——它做 causal attention，永远不往后看 generator 的 noisy tokens。这保证了 reasoner 的"思考"不会被 generator 的"草稿"污染。Reasoner 的 KV cache 永远是干净的 conditioning context。
- **Generator 看 reasoner + 自己**——它做 bidirectional attention，能看到 reasoner 的所有 KV（也就是 text prompt 和 conditioning image 的语义），也能看到自己 DM 段里的所有 tokens（包括 clean conditioning 和 noisy target）。这让它 denoise 的时候拿得到全部上下文。

**直觉上**：reasoner 是个"思考者"，它把世界状态压进 KV cache；generator 是个"想象者"，从 reasoner 的 KV 里抽信息来 denoise。两个子系统参数完全独立（各自的 LayerNorm、FFN、QKV projection），但共享一个 attention operator。这比 Transfusion 那种 FFN 也共享的设计更彻底分离，绝对质量更好，代价是参数 ×2。

为什么这么设计能 work？因为 understanding 和 generation 本质上需要**不同的 computation pattern**——understanding 是 discrete token prediction，要 sharp decision boundary；generation 是 continuous denoising，要 smooth manifold walk。硬塞进同一套 FFN 容易互相干扰。但它们又**需要共享 semantic representation**——所以 attention 共享是合理的，KV cache 是共享的语义接口。

---

## 五种 modality 怎么塞进一个序列

输入序列被切成两段，永远按这个顺序：

```
[AR 段: text + ViT vision] [DM 段: VAE vision + audio + action]
```

- AR 段在前，DM 段在后
- DM 段内部：clean conditioning 在前，noisy target 在后
- 每段内按 vision → audio → action 排

不同 task 只改 DM 段放什么：

- **Text2Image**：DM 段只有一张 noisy image
- **Text2Video**：DM 段是 noisy video frames（可选加 noisy audio）
- **Image2Video**：DM 段先放 clean first frame，再放 noisy future frames
- **Video Transfer**：DM 段先放 clean control video（edge/depth/seg map），再放 noisy RGB target
- **Action**：三种 mode（forward dynamics / inverse dynamics / policy）通过改变 clean/noisy 组合实现

**这个统一格式的威力**：你不用为每个 task 写不同的 forward pass，模型架构完全不变，只改 sequence 里塞什么。一个 checkpoint 就能干 T2I、T2V、I2V、V2V、transfer、policy、FD、ID 全部任务。

---

## Action 怎么统一四种完全不同的 embodiment

这是我觉得最巧妙的部分。Robot action 是 7-DoF joint position，camera action 是 6-DoF pose，autonomous vehicle action 是 steering + throttle，egocentric human action 是 head pose + 21-keypoint hand——**维度、语义、控制频率全不一样**，怎么塞进一个模型？

Cosmos 3 设计了一个**三元组 action representation**：

1. **Ego pose**：agent 主观察帧的相对位姿变化 $\Delta T_t = T_{t-1}^{-1} T_t$（SE(3) 相对变换）
2. **Effector pose**：end-effector 的相对位姿变化
3. **Grasp state**：当前 manipulation state（fingertip positions 或 gripper open/close）

每个 embodiment 用自己需要的子集：
- Camera / AV：只有 ego pose
- Robot：ego + effector + grasp
- Egocentric human：head pose（ego）+ wrist pose（effector）+ fingertip（grasp）

**旋转用 6D 表示**（Zhou et al. 2019），不用四元数或欧拉角。为什么？四元数有 double cover 问题（$q$ 和 $-q$ 表示同一旋转），欧拉角有 gimbal lock，3D rotation matrix 有 9 个数但只有 3 个 DoF——神经网络学起来都有 continuity 问题。6D 表示是取 rotation matrix 的前两列（6 个数），通过 SVD 投影回 SO(3)。虽然 over-parameterized，但**在 neural network 里学起来最稳定**。

然后每个 domain k 有自己的 input/output projection：

$$z = W_{in}^{(k)} x + b_{in}^{(k)} \quad \text{(encode)}$$
$$x = W_{out}^{(k)} z + b_{out}^{(k)} \quad \text{(decode)}$$

其中 $x \in \mathbb{R}^{d_{in}^{(k)}}$ 是 normalized action vector（不同 domain 维度不同），$z \in \mathbb{R}^{d_{model}}$ 是 latent action token。$W_{in}^{(k)} \in \mathbb{R}^{d_{model} \times d_{in}^{(k)}}$ 是 domain-specific 的投影矩阵。

**直觉**：MoT backbone 学一个**共享的 action 语义空间**（"往前移动"、"抓住物体"、"松开"这种 abstract action concept），每个 domain 的小 projection layer 负责把 embodiment-specific 的 raw action（joint angle / steering / hand pose）映射到这个共享空间。这就像给所有机器人设计一个**通用摇杆接口**——摇杆输出的"前进/后退/左/右"是统一的，但底下连接的是 wheel motor 还是 leg servo 还是 manipulator 是 embodiment-specific 的。

---

## 时间怎么对齐：3D MRoPE + Absolute Temporal Modulation

Transformer 要知道每个 token 的 position。Language 用 1D position 就够了，video 要 3D（time × height × width），audio 和 action 只要 1D（time）。Cosmos 3 继承 Qwen3-VL 的 **3D MRoPE**：每个 token 拿一个 $(t, h, w)$ 三元组。

- Language tokens：$t = h = w$，退化成 1D RoPE
- Vision tokens：$t$ 按时间，$h, w$ 按空间
- Audio / Action：只变 $t$，$h = w = 0$

但这里有个**多采样率问题**。Video 可能是 24/30/60 FPS，audio 是 25 TPS（48000/1920），robot action 是 10-15 Hz。如果 RoPE 的 temporal index 增量都设成 1，那 60 FPS video 的"时间流速"会被模型看成 24 FPS 的 2.5 倍快——物理时间完全错乱。

解决方法是 **Absolute Temporal Modulation**：定义 TPS（temporal steps per second），base TPS = 6（即 24/4，因为 VAE temporal compress ×4）。当 temporal index 增 1 unit step 时，实际 temporal increment：

$$\delta t = \frac{TPS_{base}}{TPS}$$

所以：
- 24 FPS video（TPS=6）→ $\delta t = 1$
- 30 FPS video（TPS=7.5）→ $\delta t = 0.8$
- 10 FPS video（TPS=2.5）→ $\delta t = 2.4$
- Audio（TPS=25）→ $\delta t = 0.24$
- Robot action @ 15Hz（TPS=15）→ $\delta t = 0.4$

**30 FPS video 的 1 秒 = 30 tokens × 0.8 = 24 positional units，跟 24 FPS video 的 1 秒 = 24 tokens × 1 = 24 positional units 完全对齐**。Audio 的 1 秒 = 25 tokens × 0.24 = 6 positional units，也对齐了。

直觉上：这是把 RoPE 从 "token index space" 升级到 "physical time space"。所有模态都映射到同一个**绝对物理时间轴**，video/audio/action 在同一时刻发生的事情在 positional space 里就是同一位置。这让 audio-visual synchronization 和 video-action alignment 能被 model 自然学到。

还有个小 trick：**AR-DM margin = 15000**。如果 DM 段的 temporal index 直接接在 AR 段最后一个 language token 后面，会出现 first-frame over-saturation 和 checkerboard artifacts（在 Super 模型上尤其明显）。推测是最后一个 language token 和 first frame vision tokens 占相邻 temporal positions，temporal embedding 几乎相同导致 attention 混乱。解决：在 AR 和 DM 之间插一个固定 15000 的 temporal gap，把后面所有 DM tokens 的 temporal index 整体 shift。**这是免费的 architectural hack，纯靠 positional spacing 解决 first-frame 失真问题**。

---

## 训练三阶段：先通才，再专才

### Reasoner（understanding 侧）

- **Pre-training**：22M samples，next-token prediction。所有参数联合训练，不做 projector-only alignment stage（发现没必要）。Square-root normalized per-token loss weighting 平衡短长序列贡献。
- **SFT**：2.2M samples，importance-aware sampling。混 1:4 pre-training 数据防 specialization 破坏 general capability。覆盖 general spatial understanding + temporal understanding + AV + robotics + smart infrastructure。

Data curation 用了 **AI-judge quality filtering**：Gemma-4-31B-it 给每个 sample 从三个维度打 1-5 分——Faithfulness（response 是否 grounded in context，防 hallucination）、Completeness（是否完整回答 instruction）、Correctness（factual/logical 准确性）。三维度都 ≥ threshold 才保留。Pre-training 用 threshold=2（保 78%），SFT 用 threshold=5（保 46%，只要最高质量）。

### Generator（generation 侧）

三阶段 curriculum：

1. **Pre-training**：31T tokens（Nano）/ 17.86T tokens（Super）。大量 image + video + audio，学 general generative prior。Multi-resolution training（256p/480p/720p 同时训，比例 1:1:2:1）。Training modes 比例 T2I 20% / T2V 56% / I2V 16% / V2V 8%。
   
   Training objective 是 **rectified flow matching**：对 target latent $x_0$ 构造 noisy latent $x_\sigma = \sigma \epsilon + (1-\sigma) x_0$，模型预测 velocity $v^* = \epsilon - x_0$。Video 用 mode sampling（质量更好），image/audio/action 用 logit-normal。不同分辨率用不同 shift value（256p s=1, 480p s=3, 720p s=5）——高分辨率对 low-noise 细节更敏感，shift 让更多 signal 落在 low-noise 区域。

2. **Mid-training**：2.4T tokens（Nano）。引入 action + video transfer 这两个新模态。Data 比例：image 10% / video 32% / video+audio 8% / action 25% / general transfer 20% / driving transfer 5%。Action loss scale ×10 补偿 normalized action vector 的 MSE 偏小。

3. **Post-training**：专精到具体 task。
   - **Cosmos3-Super-Text2Image**：两阶段 SFT，Stage 1 用 45% real / 40% synthetic / 15% text-rendering，Stage 2 用 470K ultra-high-quality image-caption pairs。UniGenBench 拿 91.36（开源 SOTA）。
   - **Cosmos3-Super-Image2Video**：480p，189 frames（≈8 秒 @ 24fps），10K iter。Artificial Analysis I2V leaderboard #1 open-weight。
   - **Cosmos3-Nano-Policy-DROID**：3-view canvas（wrist 360×640 在上 + 2 external 180×320 在下），predict 32 future absolute joint-position actions @ 15Hz + auxiliary RGB frames。Inference 优化叠满——4 diffusion steps + CFG parallelism + skip video latent decode → 能部署在 2 张 RTX Pro 6000 GPU 上。

---

## 合成数据：补 long-tail 的关键

Web-scale pre-training 数据虽然海量，但分布是 **long-tailed** 的。Robotics 场景、autonomous driving 的 corner case、warehouse 安全事件这些 Physical AI 关键 domain 在 web data 里严重欠采样。Cosmos 3 造了五个合成数据集来补：

| Dataset | 内容 | 规模 |
|---|---|---|
| SDG-PhyxSim | PhysX 刚体物理仿真（dominoes, bowling, billiards, wrecking_ball 等 10 scene families） | 76K clips, 4 相机视角, per-frame physics state |
| SDG-RobotSim | Isaac Sim + MimicGen + DreamZero + SOMA humanoid | 386K clips, mobile + quadruped + humanoid + manipulator |
| SDG-DriveSim | Omniverse 驾驶仿真，long-tail scenarios（cut-in, jaywalking, weather） | 264K clips, 4K/24fps, 4-cam 或 7-cam surround |
| SDG-SynHuman | 数字人，metric depth + camera params | 237K clips, 5841 小时, 4050 digital human assets |
| SDG-Warehouse | 仓库安全事件（forklift-human near-miss, fire evacuation 等） | 123K clips, depth+seg+BBox+3D oriented BBox |

**SDG-PhyxSim 的物理 annotation 设计特别值得讲**：每个 object 的颜色直接 encode 物理状态——红=X 轴, 绿=Y 轴, 蓝=Z 轴；静止物体灰；saturation 随 magnitude 增长。所以"红色越饱和"="X 方向速度越快"。这种 visual physics encoding 让模型从 RGB 直接学到物理状态，不用单独的 physics state input。

**Ablation 结果**（Tab. 26）：SDG-All 混合训练几乎全面提升 PAIBench domain scores——Overall +0.10, Quality +0.10，9 个指标中 8 个正向。唯一普遍降的是 Human domain（-0.38 到 -0.69），即使是 SDG-SynHuman 也降。这说明 **sim-to-real gap 在 human 上最明显**——当前 simulator 还不能完美 replicate 人的微妙 appearance 和 motion nuance。但 Industrial / Physics / Robot 等 Physical AI 域大幅提升。

---

## 工程上的几个关键决策

### Joint Data Loader：处理 modality 异构性

传统 LLM training loader 假设每个 sample 同样多 token。Cosmos 3 跨模态 per-sample token count 差 100 倍（720p 2 秒视频 > 几十条短文本）。如果按 sample count 平分给 rank，会导致 padding waste + rank 间严重 workload 不平衡 + 大规模下 NCCL timeout。

四个机制：
1. **Token-budgeted packed sequences**：按 token budget 而非 sample count 拼 sequence，无 padding
2. **Rank-synchronous stream selection**：所有 rank 同一 step 处理同一 stream（global seed keyed on iteration index），防 cross-rank compute 不平衡。**端到端 throughput +54%**
3. **Look-ahead packing**：下一个 sample 超剩余 budget 就丢进 look-aside buffer，继续扫后面更小的 sample 填空隙。**Effective sequence length +8%**

### Ulysses Context Parallelism

长 sequence 怎么跨 GPU？Ulysses scheme：sequence 维度切到 CP rank，进 attention 前 all-to-all 转 head 维度（每个 rank 拿自己负责 head 的完整 sequence），attention 本地跑，再 all-to-all 转回。两次 all-to-all per attention layer。

为什么不用 Ring Attention？Ring 需要先把 interleaved AR/DM 拼成单 sequence 才能切，破坏 AR/DM 独立 shard 的便利性。NVLink 上 all-to-all 带宽很香，Ulysses 更简单。

### Async Checkpointing

专门 spawn child process + Gloo process group（不在 NCCL 上跑），CPU 算 save plan，GPU 不阻塞。Save plan memorization（第一次算完后续复用，-60% overhead）。对比同步 checkpoint：Nano -4% training time, Super -9% training time。

### Selective Activation Checkpointing

默认 activation checkpointing 只存每 block input，backward 时全 block 重算 → +33% FLOPs。SAC 按 FLOPs-to-memory ratio 排序，优先 materialize 高 ratio 的 op。**Attention output 是最大赢家**：quadratic cost 但 linear-size output。Cosmos3-Nano @ 74K token budget：**+13% throughput，数值无变化**。

---

## 结果一句话总结

- **Reasoner**：在 Physical AI domain（robotics 57.8, smart infra 62.6, driving 79.3）都是 SOTA，general benchmark 跟 Gemini 3.1 Pro 接近
- **Image generation**：Cosmos3-Super-Text2Image 在 Artificial Analysis leaderboard #1 open-weight（#4 overall），UniGenBench 91.36 开源第一
- **Video generation**：Cosmos3-Super 在 PAIBench-G T2V 和 I2V 都 SOTA（含 closed-source），Physics-IQ 物理 consistency SOTA，Cosmos HUE 人类评估开源第一
- **Audio-visual**：Semantic audio-visual grounding 和 alignment SOTA（SAV 8.35, AVAlign 8.16），audio fidelity（PQ）还有 headroom
- **Transfer generation**：unified backbone 在每个 control modality 上击败 Cosmos-Transfer2.5（用 dedicated ControlNet）
- **Robot policy**：Cosmos3-Nano-Policy-DROID 在 RoboLab-120 拿 39.7% success rate（specific instructions），远超 π0.5 28.1% 和 DreamZero 25.2%。RoboArena real-world leaderboard #1

---

## 我的几个直觉性 takeaway

1. **Dual-tower MoT 是个稳定的 unified recipe**。比 Transfusion（共享 FFN）更彻底分离，比 Show-o / Janus 那种 shared backbone 更明确——reasoner 永远不被 noisy token 污染。代价是参数 ×2，但绝对质量更优。这就像是给"思考"和"想象"各配一套独立的神经回路，但共享一个注意力系统——符合人脑的 intuition（prefrontal cortex 负责 reasoning，visual cortex 负责 imagination，但它们通过 attention mechanism 交互）。

2. **Absolute Temporal Modulation 是处理多模态多采样率的关键 insight**。这是把 RoPE 从 "token index space" 升级到 "physical time space" 的 elegant trick。就像给所有时钟调到同一时区——video 24fps、audio 25tps、action 15Hz 在模型眼里都是同一个绝对时间轴。

3. **Unified action representation 让一个 backbone 跨 4 个 embodiment**。Robot + AV + camera + egocentric human 共享一套参数，每 domain 只加 small projection layer。这证明了 Physical AI 不需要 per-embodiment model——只要 action representation 设计得够 abstract，"前进/抓取/松开"这种 abstract action concept 是跨 embodiment 可迁移的。

4. **Reasoner benefits Generator**——understanding 和 generation 共享 backbone 不是巧合。Physical AI reasoner 给的 conditioning embedding 比 generic VLM 给的更 physically grounded，T2V Robot domain +4.8 PSNR。这就像是"先理解世界才能画好世界"——reasoner 学到的物理常识通过 KV cache 传给 generator，让 generator 画出来的东西更符合物理规律。

5. **Synthetic data 是 long-tail coverage 的关键**。Web data 拍不到的 Physical AI 场景（collision, corner case driving, warehouse safety），用 simulator 造。Sim-to-real gap 在 human 上最明显，但 Industrial/Physics/Robot 大幅提升。这就像在现实世界拍不到的场景里用游戏引擎"拍电影"——Isaac Sim / Omniverse 就是 Physical AI 的"摄影棚"。

6. **Inference 优化叠 7 个 trick**（Cache-DiT, Ulysses CP, CFG-Parallel, HSDP, CPU offload, VAE-Patch-Parallel, FP8）才能把 64B Super 模型部署到合理 latency。这告诉我们：model architecture 之外，inference engineering 同等重要。特别是 **Reasoner tower caching**——T2I/T2V/I2V 任务里 reasoner 输入不变，所以 reasoner forward 只跑一次，output cache 给后续所有 denoising step 复用，per-step latency 大幅降低。

---

## 参考链接

- Paper 主页: https://research.nvidia.com/labs/cosmos-lab/cosmos3
- Code: https://github.com/nvidia/cosmos
- Cosmos3-Super checkpoint: https://huggingface.co/nvidia/Cosmos3-Super
- Cosmos3-Nano checkpoint: https://huggingface.co/nvidia/Cosmos3-Nano
- Cosmos3-Super-Text2Image: https://huggingface.co/nvidia/Cosmos3-Super-Text2Image
- Cosmos3-Super-Image2Video: https://huggingface.co/nvidia/Cosmos3-Super-Image2Video
- Cosmos3-Nano-Policy-DROID: https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID
- SDG-PhyxSim: https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes
- SDG-RobotSim: https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes
- SDG-DriveSim: https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios
- SDG-SynHuman: https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Digital-Human-Scenes
- SDG-Warehouse: https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes
- Cosmos-HUE benchmark: https://huggingface.co/datasets/nvidia/Cosmos-HumanEval-v1
- Mixture-of-Transformers 原始 paper: https://arxiv.org/abs/2403.09869
- Transfusion: https://arxiv.org/abs/2408.11039
- 6D rotation representation (Zhou et al. CVPR 2019): https://arxiv.org/abs/1812.07035
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- Wan 2.2 video tokenizer: https://arxiv.org/abs/2503.20314
- FlashAttention-3: https://arxiv.org/abs/2407.08609
- DeepSpeed Ulysses: https://arxiv.org/abs/2309.14509
- DROID robot dataset: https://arxiv.org/abs/2403.12945
- Physics-IQ benchmark: https://arxiv.org/abs/2506.09639
- PAI-Bench: https://arxiv.org/abs/2512.01989
- Artificial Analysis T2I arena: https://artificialanalysis.ai/text-to-image/arena
- Artificial Analysis I2V arena: https://artificialanalysis.ai/image-to-video/arena

如果你想我深入展开某一块——比如 rectified flow matching 为什么用 mode sampling 比 logit-normal 好、6D rotation 为什么比四元数稳定、reasoner KV cache 怎么在 multi-step diffusion 里复用、或者 15000 这个 AR-DM margin 数值是怎么调出来的——我可以再钻进去讲。

---

这篇 Cosmos 3 是 NVIDIA 出的一个 omnimodal world model family，核心论点是：**Physical AI 不应该把 understanding 和 generation 拆成两套独立模型**（VLM + video generator + world model + action model 拼起来的 fragment pipeline 是次优的），而应该用一个 unified backbone 同时处理 language, image, video, audio, action 五种模态的理解和生成。下面我尽可能详细拆给你看，重点 build intuition。

---

## 1. 核心架构：Mixture-of-Transformers (MoT)

Cosmos 3 的核心 trick 是 **dual-tower** 设计——同一个 transformer stack 里有两个独立的参数集，一个走 autoregressive（reasoner），一个走 diffusion（generator），它们只在一个 shared self-attention operator 里交互。这跟 Transfusion / BAGEL / Janus-Pro 这一类 unified model 的最大区别在于：Cosmos 3 **不共享 FFN 和 LayerNorm**，而是给两路各一份独立参数，但都从同一个 pre-trained VLM 初始化。

### 1.1 Token arrangement 的统一格式

输入序列被切成两段：

- **AR subsequence**（在前）：language tokens + ViT-encoded vision tokens（用于 understanding）
- **DM subsequence**（在后）：VAE-encoded vision tokens + audio tokens + action tokens（用于 generation，会被 noise-perturb）

格式约定：(1) AR 在前 DM 在后；(2) DM 内部 clean conditioning 在前、noisy 在后；(3) 每段内按 vision → audio → action 排。

各种 generation mode 都用这套统一格式表达：

```
Text2Image:    S_T2I = [S_AR, ṽ_1]
Text2Video:   S_T2V+A = [S_AR, ṽ_{1:N}, s̃]
Image2Video:  S_V2V = [S_AR, v_{1:P}, ṽ_{P+1:N}]    # P=1 是 I2V, P>1 是 V2V
Transfer:     S_Transfer = [S_AR, v_{1:N}^ctrl, ṽ_{1:N}]
Action:       Forward / Inverse / Policy (三种 clean/noisy 组合)
```

其中 $S_{AR} \triangleq [l_1, \ldots, l_n, \langle EOS \rangle, \langle BOG \rangle]$，$\langle BOG \rangle$ 是 "begin of generation" 特殊 token，告诉模型接下来 DM 段要开始了。$\tilde{v}$ 是 noisy vision tokens，$v$ 是 clean vision tokens，$\tilde{s}$ 是 noisy audio。这种设计的 intuition 是：**AR 段就是条件 context，DM 段就是被生成的 content**，所以 AR 永远在前、DM 永远在后，不同 task 只改 DM 里放什么。

### 1.2 Dual-stream joint attention 的核心公式

这里是整篇 paper 最关键的架构选择。设 $Q_{AR}, K_{AR}, V_{AR}$ 是 reasoner 的 query/key/value，$Q_{DM}, K_{DM}, V_{DM}$ 是 generator 的。

**AR 路只用 causal self-attention**，永远只看自己：

$$O_{AR} = \text{Attn}_{causal}(Q_{AR}, K_{AR}, V_{AR})$$

**DM 路 用 bidirectional attention，但 key/value 是 AR+DM 的拼接**：

$$O_{DM} = \text{Attn}_{full}\big(Q_{DM}, \Theta[K_{AR}; K_{DM}], \Theta[V_{AR}; V_{DM}]\big)$$

这里 $\Theta[\cdot;\cdot]$ 是 sequence 维度的 concatenation，$[\cdot;\cdot]$ 表示拼接。**关键 invariant：AR tokens 永远不会被 DM tokens 更新**，这保住了 AR 路的 causal integrity——reasoner 看到的 context 永远是 clean 的 conditioning，不会被 generator 的 noise 污染。而 DM 路可以自由地 attend 到 text prompt 和 conditioning image/video，所以 diffusion generation 拿得到全部上下文。

Intuition 上你可以把它想成：reasoner 是一个 "thinking" 的子系统，它把世界状态压缩进 KV cache；generator 是一个 "imagining" 的子系统，它从 reasoner 的 KV 里抽信息来 denoise。它们共享一个 attention operator 但参数完全独立。

### 1.3 Two-way flat attention 的 kernel 实现

工程上这个 dual-mask attention 怎么高效实现？他们没用 FlexAttention（因为 mask 是 opaque 的，会浪费 tensor core），而是**拆成两次 variable-length SDPA call**：

1. 第一次：causal varlen SDPA，只处理 packed 的 AR tokens，输出 block-diagonal causal mask
2. 第二次：bidirectional varlen SDPA，query 是 DM tokens，但 K/V 是按 sample 粒度 interleave 的 $[R_0, G_0, R_1, G_1, \ldots, R_n, G_n]$，每个 generator query 只能 bidirectionally attend 到自己 sample 的 $[R_i, G_i]$ 块

这样能消除 padding overhead，**end-to-end throughput +22%**。在 Hopper 上用 FlashAttention-3，在 Blackwell 上用 NATTEN（基于 CUTLASS，针对第五代 tensor core 优化）。

---

## 2. Modality encoders：四个模态四套 tokenizer

### 2.1 Vision（双 encoder）

- **Understanding 路用 ViT**：16×16 patch → 2×2 token merge → MLP 投到 transformer hidden dim。借鉴 Qwen3-VL 的 DeepStack（多层级 feature 聚合）+ video timestamps interleaved 在 frames 之间。这个 encoder 跟 backbone 一起训练。
- **Generation 路用 Wan2.2 VAE**：temporal compress ×4，spatial 32×32（实际是 16×16 spatial + 2×2 patch merge）。Frozen。每个 VAE token 用 linear 投到 hidden dim。

注意：**understanding 和 generation 用不同 encoder**，因为 ViT 给的是 semantic token（discrete-ish），VAE 给的是 continuous latent（适合 flow matching）。这其实是个不对称设计——reasoner 看 semantic，generator 看 pixel-level latent。

### 2.2 Audio

audio VAE from Lee et al. 2025，stereo 48 kHz，hop size 1920 samples → **25 tokens/sec**。Frozen。linear projection 到 hidden dim。

### 2.3 Action（这是我觉得最巧妙的部分）

action 要支持 camera、autonomous vehicle、robot、egocentric human 这四种异构 embodiment。Cosmos 3 的设计是：

**统一 action 三元组**：ego pose + effector pose + grasp state

- **ego pose**：agent 主观察帧的相对位姿变化 $\Delta T_t = T_{t-1}^{-1} T_t$（SE(3) 相对变换）
- **effector pose**：end-effector 的相对位姿变化
- **grasp state**：当前 manipulation state（fingertip positions 或 gripper open/close value）

旋转用 **6D 表示**（Zhou et al. 2019，CVPR），因为 3D rotation 虽然 DoF=3，但用 6D over-parameterized 表示能避免 discontinuity 问题（四元数和欧拉角都有拓扑问题）。OpenCV convention：z 轴沿手指/gripper，x 轴向右。

**Domain-aware projection**：每个 domain k 有自己的 input/output projection：

$$z = W_{in}^{(k)} x + b_{in}^{(k)} \quad \text{(Eq.1, input projection)}$$

$$x = W_{out}^{(k)} z + b_{out}^{(k)} \quad \text{(Eq.2, output projection)}$$

其中 $x \in \mathbb{R}^{d_{in}^{(k)}}$ 是 normalized action vector（不同 domain 维度不同，比如 egocentric 是 head-pose delta + 左右手 wrist-pose delta + fingertip coords 拼起来的），$z \in \mathbb{R}^{d_{model}}$ 是 latent action token。$W_{in}^{(k)} \in \mathbb{R}^{d_{model} \times d_{in}^{(k)}}$ 和 $b_{in}^{(k)} \in \mathbb{R}^{d_{model}}$ 是 domain-specific 参数。

Intuition：**MoT backbone 学共享 action 语义空间，domain-specific projection 负责 embodiment-specific 维度映射**。这避免了 robot action（7-DoF joint position）和 camera action（6-DoF pose）硬塞进同一个 vector 的问题。预测出来 6D rotation 用 SVD 转回 $3 \times 3$ SO(3) matrix。

---

## 3. Position embedding：3D MRoPE + Absolute Temporal Modulation

### 3.1 Position index 怎么分配

继承 Qwen3-VL 的 3D MRoPE：每个 token 拿 $(t, h, w)$ 三元组。

- **Language tokens**：$t = h = w$，单调递减 → 退化成 1D RoPE
- **ViT vision tokens**：同帧共享 $t$，$h, w$ 按 spatial location 独立变化
- **VAE video tokens**：$t$ 按时间 frame index，$h, w$ 按 spatial grid
- **Image tokens**：单帧 video，只变 $(h, w)$
- **Audio / Action tokens**：只变 $t$，$h = w = 0$

**关键细节：AR-DM margin = 15000**。直接让 DM tokens 从 AR 最后一个 token 的 temporal offset 接下去，会导致 first frame over-saturation + checkerboard artifacts（在 Super 模型上尤其明显）。推测原因：最后一个 language token 和 first frame vision tokens 占相邻 temporal positions → 几乎相同的 temporal embedding。解决：在 AR 和 DM 之间插一个固定 15000 的 temporal gap，把后面所有 vision/audio/action tokens 的 temporal index 整体 shift。**这是个免费的 architectural trick，纯靠 positional spacing 解决了 first-frame 失真问题**。

### 3.2 Absolute Temporal Modulation（这是处理不同 FPS 的核心）

问题：60 FPS 视频和 24 FPS 视频的 temporal index 增量对应的物理时间不同——同样增 1，24 FPS 是 1/24 秒，60 FPS 是 1/60 秒。如果让 RoPE 增量都一样，模型会把 24 FPS 视频的"时间流速"看作 60 FPS 的 2.5 倍快——明显错。

定义 **TPS = temporal steps per second**（物理时间分辨率）：
- video: TPS = FPS / 4（因为 VAE temporal compress ×4）
- audio: TPS = 48000 / 1920 = 25
- action: TPS = sampling frequency

定义 base TPS = 6（即 24/4，因为 24 FPS 是训练数据最常见的）。

当 temporal index 增 1 unit step 时，**实际 temporal increment**：

$$\delta t = \frac{TPS_{base}}{TPS} \quad \text{(Eq.9)}$$

所以 24 FPS video（TPS=6）→ $\delta t = 1$；30 FPS video（TPS=7.5）→ $\delta t = 0.8$；10 FPS video（TPS=2.5）→ $\delta t = 2.4$。**这让不同 FPS 的等物理时长在 positional space 里占相同距离**——30 FPS 视频的 1 秒 = 30 个 token × 0.8 = 24 positional units，跟 24 FPS 视频的 1 秒 = 24 个 token × 1 = 24 positional units 完全对齐。

Intuition：这是把 RoPE 从 "token index space" 升级成 "physical time space" 的 trick，让多模态多采样率在同一个时间轴上对齐。这个对 audio 和 action 尤其重要，因为它们的 TPS 跟 video 完全不同（audio 是 25，robot action 通常是 10-15）。

---

## 4. Model variants

| Variant | Backbone | LLM Layers | Hidden | Heads | KV Heads | FFN | 总参 |
|---|---|---|---|---|---|---|---|
| Edge | Nemotron-3 2B (from scratch) | 28 | 2048 | 16 | 8 | 9216 | 4B |
| Nano | Qwen3-VL 8B | 36 | 4096 | 32 | 8 | 12288 | 16B |
| Super | Qwen3-VL 32B | 64 | 5120 | 64 | 8 | 25600 | 64B |

注意 KV heads 都压到 8（GQA），head dim 都是 128。**两个 tower 各自独立参数**，所以参数量大致是 backbone ×2。

Edge 用 ReLU-squared FFN activation + 移除 QK normalization（跟 Qwen3-1.7B 不一样），跟 NVIDIA Nemotron-3 一致。Nano 和 Super 直接用 Qwen3-VL 权重初始化两个 tower。

---

## 5. Training objective：Rectified Flow Matching

整个 generator 用 rectified flow matching 训练。对任意 modality 的 target latent $x_0$，构造 noisy latent：

$$x_\sigma = \sigma \cdot \epsilon + (1 - \sigma) \cdot x_0$$

其中 $\epsilon \sim \mathcal{N}(0, I)$，$\sigma \in [0, 1]$ 是 noise level。denoiser $v_\theta(x_\sigma, \sigma, c)$ 学预测 **constant velocity**：

$$v^* = \epsilon - x_0$$

用 masked MSE loss（conditioning tokens 比如 I2V 的第一帧 gate 掉不参与 loss）。

**Per-modality time sampling**：image/audio/action 用 **logit-normal** 噪声分布，video 用 **mode sampling**（实验发现 mode sampling 对 video 质量更好）。

**Rectified-flow shift reparameterization**：

$$\sigma = \frac{s \cdot \bar{t}}{1 + (s-1) \cdot \bar{t}}, \quad \bar{t} = 1 - t$$

$s \geq 1$ 把 marginal distribution 往高噪声偏。不同分辨率用不同 shift：
- 256p: s=1
- 480p: s=3
- 720p: s=5
- mid-training 提升：256p/480p/720p → 3/5/10

Intuition：高分辨率视频对 low-noise stage 的细节重建更敏感，shift 让更多 training signal 落在 low-noise 区域，提升高频细节。

**Multi-resolution training**：固定 74K token budget，256p/480p/720p 按 1:2:1 比例（image-only : 256p video : 480p video : 720p video = 1:1:2:1）。Sequence packing，无 padding。

**Training modes 比例**：T2I 20%, T2V 56%, I2V 16%, V2V 8%。

**CFG dropout**：text 条件 10% dropout（用 classifier-free guidance）。

---

## 6. 三阶段 training curriculum

### 6.1 Reasoner：pre-training + SFT

- **Pre-training**：22M samples（19.7M from Nemotron Nano 2 + 2.3M curated）。所有参数联合训练（不做 projector 单独 alignment stage，发现没必要）。Sequence ≤ 16K tokens，per-sample limit 2048 image / 8192 video tokens。Square-root normalized per-token loss weighting。AdamW，peak lr 5e-5（LLM）/ 5e-6（ViT），cosine decay 到 0.1，10% warmup。$\beta = (0.9, 0.999)$，weight decay 0.05，gradient clip 1.0。
- **SFT**：2.2M samples，importance-aware sampling。混 1:4 pre-training-to-SFT 比例（防 specialization 破坏 general capability）。8200 iter，global batch 512，AdamW lr 1e-5（LLM）/ 1e-6（ViT），cosine decay 到 0.1，1000 warmup。$\beta = (0.9, 0.95)$，weight decay 0.1，gradient clip 1.0。

**Data curation pipeline**：
1. **Semantic deduplication**：用 Qwen3-VL-Embedding-8B（image/text）和 PE-Core-G14-448（video）算 joint embedding，K-means 聚类后 cos sim > 0.95 砍掉。砍掉 4.23%。
2. **AI-judge quality filtering**：用 Gemma-4-31B-it 当 judge，从 1-5 给三个维度打分：
   - **Faithfulness**：response claims 是否 grounded in image/video/text context（防止 hallucination）
   - **Completeness**：是否完整回答 instruction
   - **Correctness**：factual/logical/task-level 准确性
   
   规则：三维度都 ≥ threshold 才保留。Pre-training 用 threshold=2（保留 78%），SFT 用 threshold=5（保留 46%，只要最高质量）。

### 6.2 Generator：pre-training → mid-training → post-training

| Stage | Nano tokens | Super tokens | GPU |
|---|---|---|---|
| Pre-training | 31.05T | 17.86T | 1024 / 2048 GB200 |
| Mid-training | 2.4T | 1.9T | 1024 / 2048 GB200 |

Mid-training 引入 action + transfer 这两个新模态，data 比例：

| Stream | Modes | Share |
|---|---|---|
| Image | T2I | 10% |
| Video | T2V/I2V/V2V | 32% |
| Video+Audio | T2(V+A)/I2(V+A)/V2(V+A) | 8% |
| Action | FD/ID/Policy | 25% |
| General Transfer | edge/blur/depth/seg | 20% |
| Driving Transfer | world-scenario-map | 5% |

Action loss scale ×10 补偿 normalized action vector 的 MSE 偏小。

### 6.3 三个 post-trained 变体

**Cosmos3-Super-Text2Image**：
- Stage 1：20K steps SFT，45% real / 40% synthetic / 15% text-rendering，lr 1e-4
- Stage 2：2K steps refinement，470K ultra-high-quality image-caption pairs
- Context 70K tokens，只用 >720p 图片
- UniGenBench 拿到 91.36（开源 SOTA）

**Cosmos3-Super-Image2Video**：
- 10K iter，lr 1e-5，约 50B tokens
- 480p，189 frames（≈8 秒 @ 24fps）
- 混 20% T2I image tokens 保 semantic alignment

**Cosmos3-Nano-Policy-DROID**：
- Input：3-view canvas（wrist 360×640 在上 + 2 external 180×320 在下左右），canvas 总尺寸 540×640
- Predict 32 future absolute joint-position actions @ 15Hz + auxiliary RGB frames
- Action encoder/decoder/MLP 全部 fresh init，5× lr multiplier 加速适应
- lr 2e-4，其他跟 mid-training 一致
- **Inference 优化**：4 diffusion steps + shift 5 + CFG scale 3 + CFG parallelism + skip video latent decode → 能部署在 2 张 RTX Pro 6000 GPU 上

---

## 7. Data 规模和合成数据

### 7.1 Pre-training 视觉数据

767M images + 347.7M video clips（从 7.8B raw images + 3B raw videos 处理）。
- 720p: image 26.8%, video 36.4%
- 480p: image 26.0%, video 30.8%
- ≥1080p: image 25.2%, video 12.2%
- 16:9 是最常见 aspect ratio（image 52%, video 97.3%）

Curation pipeline：scene-change detection (TransNetV2) → black border 移除 → embedding + KMeans dedup（147M image + 400M video, 20000 clusters each）→ semantic tagging (47 hierarchical categories) → DOVER aesthetic/technical quality + VTSS training suitability 三连续分 + 100 binary artifact tags。

### 7.2 Audio 数据：speech-synchronized vs non-speech

138.9M pre-training clips（保留 broad coverage）。Mid-training 筛到 18.8M：12.8M non-speech + 6M speech-synchronized。

筛选 pipeline（这里设计很精巧）：
1. **Source separation**：SAM-Audio 分离 speech stem + remaining stem
2. **Lip-sync scoring**：SyncNet 算 has_face + lip_sync_confidence，speech_synced = has_face ∧ lip_sync_confidence > 3.0
3. **Audio event detection**：FireRedASR2S 估 speech_ratio + music_ratio，high_music = music_ratio > 0.1
4. **Instrument detection**：Qwen3-VL 判 is_music_instrument（保护乐器演奏视频，music 是 visible event 不是 BGM）
5. **Speech branch**：保留 speech_synced clips；如果 high_music 且非 instrument，SAM-Audio 去音乐，再验证 speech_ratio ≥ 0.05 & music_ratio = 0
6. **Non-speech branch**：speech_ratio ≥ 0.05 用 remaining stem 去人声，否则保原始 audio；如果 high_music 且非 instrument，去音乐

### 7.3 Action 数据：8.4M episodes, 61.3K hours

| Domain | Hours | Share |
|---|---|---|
| Egocentric motion (bimanual hand, head-mounted cam, 21-keypoint 3D hand pose) | 41.3K | 67.4% |
| Autonomous vehicle (NVIDIA Hyperion, in-house logs) | 10.0K | 16.3% |
| Robotics (AgiBot / Franka / Google Robot / WidowX / UMI / UR, 90.4K tasks, 516.7K episodes) | 5.4K | 8.7% |
| Camera motion (ViPE + DepthAnything3 估 camera pose) | 4.6K | 7.5% |

### 7.4 五个 SDG 合成数据集

| Dataset | Clips | Resolution/FPS | 关键内容 |
|---|---|---|---|
| SDG-PhyxSim | 76,489 | 1920×1080/30 | PhysX 刚体物理仿真，10 scene families（dominoes, ball_mixer, bowling, billiards, towers, wrecking_ball 等），4 相机视角，per-frame physics state annotation（linear/angular velocity, COM displacement, cumulative rotation）|
| SDG-RobotSim | 208,022 | varies | 386K clips 含 humanoid，Isaac Sim + MimicGen + DreamZero + Simulario + SOMA |
| SDG-DriveSim | 264,000 | 4K/24 | 1,467 小时，long-tail scenarios（cut-in, jaywalking, weather degradation 等），4-cam 或 7-cam surround |
| SDG-SynHuman | 236,937 | 1080p/30 | 5,841 小时，4050 数字人资产 + 198 indoor + 200 outdoor 环境，metric depth + camera params |
| SDG-Warehouse | 122,952 | 1920×1080/30 | 412 小时，4 safety scenarios（forklift-human near-miss, fire evacuation, forklift-shelf collision, box-pickup），深度+分割+BBox+3D oriented BBox |

**SDG-PhyxSim 物理 annotation 设计特别值得看**：每个 object 的颜色直接 encode 物理状态（红=X, 绿=Y, 蓝=Z 轴；静止物体灰；saturation 随 magnitude 增长），normalized 到全 clip 范围。所以同一颜色在同一 clip 内对应同一物理量级。这种 visual physics encoding 让模型可以从 RGB 直接学到物理状态。

SDG-All 混合训练效果最好（Tab.26）：Overall 79.77（+0.10），Quality 72.56（+0.10），9 个指标中 8 个正向，只有 Human domain 因 sim-to-real gap 普遍降（-0.38 到 -0.69）。

---

## 8. Infrastructure（这部分的工程量很惊人）

### 8.1 SILA 数据平台

Lance columnar storage（不是 Postgres table-per-pipeline），每个 sample 是一行，每个 curation signal 是一个 typed column。Fragment-level coordination：worker 拿 time-limited lease，heartbeat 维持，掉了别人接管。Staged Ray execution：load → preprocess → compute → postprocess → write → commit 分 stage 跑，backpressure 防 fast stage 淹没 slow stage。Node-local model endpoints（vLLM 跑 captioner/scorer，worker 直接调本地 endpoint）。

Job startup 从 30-60 分钟降到 5 分钟，throughput ×10，daily 可处理 billions of row-level annotations。

### 8.2 Joint Data Loader（处理 modality 异构性的关键）

传统 LLM training loader 假设每个 sample 同样多 token。Cosmos 3 跨模态 per-sample token count 差 100 倍（720p 2 秒视频 > 几十条短文本）。如果按 sample count 平分给 rank，会导致：
- 大量 padding waste
- rank 间严重 workload 不平衡
- 大规模下 NCCL collective timeout

四个机制：
1. **Token-budgeted packed sequences**：每个 rank 按 token budget $T_{max}$ 而非 sample count 拼 sequence，无 padding
2. **Joint data loader**：每个 stream 自己一个 loader + prefetch buffer
3. **Rank-synchronous stream selection**：global seed keyed on iteration index，所有 rank 同一 step 处理同一 stream（防 cross-rank compute 不平衡）。**端到端 throughput +54%**
4. **Look-ahead packing**：下一个 sample 超剩余 budget 就丢进 look-aside buffer，继续扫后面更小的 sample 填空隙，最后把 look-aside 还回 stream 头部。**Effective sequence length +8%**

冷启动处理：构造时每个 stream prefetch 一次，然后 distributed barrier，第一个 forward pass 之前所有 worker 都 warm 完。

### 8.3 Distributed training: HSDP + Ulysses CP

- **HSDP**：Hybrid Sharded Data Parallelism，optimizer state/gradient/parameter 在 replica group 内 shard，跨 group replicate
- **Ulysses CP**：sequence 维度切到 CP rank，进 attention 前 all-to-all 转 head 维度（每个 rank 拿自己负责的 head 的完整 sequence），attention 本地跑，再 all-to-all 转回。两次 all-to-all per attention layer。
- 为什么不用 Ring Attention：Ring 需要先把 interleaved AR/DM 拼成单 sequence 才能切，破坏了 AR/DM 独立 shard 的便利性；mask 构造在 rotating ring schedule 下也复杂。NVLink 上 all-to-all 带宽很香。

### 8.4 Selective Activation Checkpointing (SAC)

默认 activation checkpointing 只存每 block 的 input，backward 时全 block 重算 → +33% FLOPs，-33% throughput。

SAC 按 **FLOPs-to-memory ratio** 排序候选 op，优先 materialize 高 ratio 的（attention output 是最大赢家：quadratic cost 但 linear-size output）。Cosmos3-Nano @ 74K token budget：**+13% throughput，数值无变化**。

### 8.5 Video Tokenizer 加速

Wan2.2 causal VAE 默认每次 encode 一个 latent chunk（4 frames after prime）。GPU 严重 underutilized。

优化：
- **Chunked encoding**：每 call 处理多帧。最优点：256p 68 帧/call，480p 24 帧/call，720p 12 帧/call。让 encoder 跑到 compute-bound 侧。
- **AOT torch.compile**：分 45 个静态 shape graph（3 resolution × 5 aspect ratio × 3 call mode），每个 rank 编译一个 graph 写到共享 FS，全部 rank 加载完整 45 graphs。Warm-up 从 15 min → <1 min。
- **Known frame count specialization**：robot action dataset 固定帧数，直接 specialize 到 runtime shape，跳过 padding-crop。

### 8.6 Async checkpointing

专门的 spawn child process + Gloo process group（不在 NCCL 上跑），CPU-side 算 save plan，GPU 不阻塞。

- Save plan memorization：第一次算完后续复用，-60% overhead
- dedup_to_lowest_rank=True：replicated tensor 只存 rank 0，load 时其他 rank 只读自己 shard + rank-0 shard
- 对比同步 checkpoint：Nano -4% training time, Super -9% training time

### 8.7 Throughput

| Model | Iter (s) | TFLOPS | MFU | Iter/hr | Img Tok/hr/GPU (M) | Vid Tok/hr/GPU (M) |
|---|---|---|---|---|---|---|
| Cosmos3-Nano | 7.1 | 520 | 0.23 | 507 | 4.56 | 16.23 |
| Cosmos3-Super | 19.5 | 673 | 0.30 | 185 | 1.66 | 5.91 |

Super 的 MFU 0.30 比 Nano 0.23 高，因为大 model computation/token 多，更饱和 GPU。

### 8.8 Serving：vLLM-Omni 集成

Generator 集成进 vLLM-Omni，支持：
- **Cache-DiT**：跨 denoising step 复用 transformer block output（training-free）
- **Ulysses CP**：长 sequence 跨 GPU shard
- **CFG-Parallel**：conditional + unconditional forward 并行到两个 GPU rank
- **HSDP**：FSDP2 shard 权重，按需 gather
- **CPU offload**：layer-wise weight swap
- **VAE-Patch-Parallel**：latent 切 spatial tile 跨 rank encode/decode
- **FP8 quantization**

Plain PyTorch path 用 torch.compile + CUDA graph replay，transformer block granularity capture，**T2I 30%-60% 加速**。

**Reasoner tower caching**：T2I/T2V/I2V/V2V 任务里，reasoner 输入（text + 可选 conditioning image/video）在 sampling trajectory 里不变，所以 reasoner forward 只跑一次，output cache 给后续所有 denoising step 复用。**Per-step latency 大幅降低，质量零影响**。

---

## 9. Results 摘要

### 9.1 Reasoner (48 benchmarks)

- General: Cosmos3-Super 73.7，跟 Gemini 3.1 Pro (77.5) 接近，跟 Qwen3-VL-32B (72.8) 持平
- Robotics: Cosmos3-Super 57.8，**SOTA**（超 Gemini 3.1 Pro 58.2 接近，超 Qwen3-VL-32B 52.6）
- Smart Infra: Cosmos3-Super 62.6，**SOTA**
- Driving: Cosmos3-Super 79.3，**SOTA**（远超 Gemini 3.1 Pro 47.2）

### 9.2 Image generation

UniGenBench（Cosmos3-Super-Text2Image 拿 91.36，开源第一，超 Gemini 3 Pro Image 90.69）。
Artificial Analysis Text-to-Image Leaderboard：**#1 open-weight, #4 overall**（仅次于 GPT Image 2, GPT Image 1.5, Nano Banana 2）。

CVTG（视觉文字生成）：GNED 80.88, PNED 89.08，跟 Qwen-Image-2512 (79.68/90.86) 和 Hunyuan 3.0 (71.40/87.68) 同级或更好。

### 9.3 Video generation

PAIBench-G（1044 prompts，6 Physical AI domains）：

| Model | T2V Overall | T2V Domain | T2V Quality | I2V Overall |
|---|---|---|---|---|
| Cosmos3-Super | **80.0** | **86.8** | 73.1 | **82.8** |
| Cosmos3-Nano | 79.4 | 85.8 | 73.0 | 82.7 |
| Veo-3.1 | 79.1 | 85.2 | 72.9 | 82.6 |
| Wan2.2-A14B | 78.0 | 83.2 | 72.8 | 81.3 |

Cosmos3-Super 在 T2V 和 I2V overall 都拿到 SOTA（含 closed-source）。

RBench（650 embodied task cases）：Cosmos3-Nano 58.4% / Cosmos3-Super 58.1%（Veo-3.1 56.3%, Wan2.6 60.7%）。

Physics-IQ（专门测物理一致性）：Cosmos3-Super I2V 43.8（+WMReward+BoN 48.9），V2V 59.7（+WMReward+BoN 63.4），**两个都是 SOTA**。

### 9.4 Cosmos HUE（人类评估）

T2V: Veo-3.1 91.3, Seedance 90.0, **Cosmos3-Super 89.3（开源第一）**, Cosmos3-Nano 87.6, Wan2.2-A14B 88.2。
I2V: Veo-3.1 89.7, **Cosmos3-Super 89.6（差 0.1）**, Cosmos3-Nano 88.6。
HUE 拆 4 个维度：Semantic Alignment / Physical Laws / Geometric Reasoning / Visual Integrity。Cosmos3-Super 在 12 axis 中 9 个开源第一，且 AV (87.7) 和 Physics (91.5) 上超过所有 closed-source。

**Human World Bench**（egocentric human motion, 180 samples）：Cosmos3-Super **71.9**（SOTA，超 Veo-3.1 67.8 +4.1，超 Wan2.2-A14B 60.7 +11.2）。

Artificial Analysis Image-to-Video Leaderboard：**Cosmos3-Super-Image2Video #1 open-weight**。

### 9.5 Audio-Visual

Cosmos-SoundBench（144 FoleyBench prompts），AVQ = 0.5×SAV + 0.5×PQ：

| Model | AVQ | SAV | SA | AVAlign | Visual Sup. | PQ |
|---|---|---|---|---|---|---|
| Seedance-1.5-Pro | **7.64** | 8.21 | 8.22 | 8.06 | 8.61 | **7.06** |
| Veo-3.1 | 7.45 | 8.21 | 8.21 | 8.01 | 8.85 | 6.68 |
| Cosmos3-Super | 7.31 | 8.34 | 8.30 | 8.14 | **9.18** | 6.28 |
| Cosmos3-Nano | 7.34 | **8.35** | **8.33** | **8.16** | 9.10 | 6.32 |

Cosmos 在 semantic audio-visual grounding 和 alignment 上 SOTA，但 audio fidelity（PQ）还有 headroom——这是 audio VAE 的限制，不是 architecture 限制。

### 9.6 Transfer generation（Video-to-Video with control signals）

PAIBench-C（600 clips，4 controls：blur/edge/seg/depth）。Cosmos 3 不用 ControlNet，而是把 control video 直接塞进 input sequence 当 conditioning。结果：

| Model | DOVER ↑ | Seg mIoU ↑ | Blur SSIM ↑ | Edge F1 ↑ | Depth si-RMSE ↓ |
|---|---|---|---|---|---|
| Cosmos3-Super | 10.14 | 0.71 | 0.91 | **0.50** | **0.58** |
| Cosmos3-Nano | 10.39 | **0.72** | 0.91 | 0.49 | 0.62 |
| Cosmos-Transfer2.5 | 9.49 | 0.68 | 0.90 | 0.45 | 0.68 |

Cosmos 3 unified backbone 在每个 metric 上通过两个变体之一击败 Cosmos-Transfer2.5（用 dedicated ControlNet per modality）。Intuition：**ControlNet 这种 per-modality adapter 不再是 high-fidelity control 的前提**，unified sequence model + control as conditioning 就够了。

**Two-weight CFG** for transfer：text 和 control 各自一个 guidance weight，control weight 从 prompt-only 预测外推到 fully conditional（加强结构控制），text weight 从 negative-prompt 预测外推远离（加强 caption fidelity）。

### 9.7 Action generation

四个 domain × {FD, ID, Policy} 的总览：

**AV ID（ego-trajectory from video）**：Cosmos3-Nano MT-init RRE 0.211°, RTE 0.014m, ATE 0.98m，**远超 VGGT (0.596°, 0.768m, 23.46m) 和 DepthAnything3 (0.312°, 0.354m, 9.29m)**。Cosmos 的 metric-scale 估计比 general domain baseline 强很多。

**Camera motion FD**：Cosmos3-Super MT-init 0.142°, 0.026m, 0.99m，**超 Lingbot-World (0.299°, 0.057m, 2.88m) 和 HY-World 1.5 (0.377°, 0.042m, 1.39m)**。

**Egocentric motion FD**（HWB）：Cosmos3-Super MT-init PSNR 16.19dB，**远超 LOME 9.36dB**。

**Robotics FD**（DROID）：Cosmos3-Super MT-init PSNR **26.04dB**，超 Ctrl-World 22.99dB。

**Robot policy**（RoboLab-120，120 language-conditioned tasks）：

| Model | Vague | Default | Specific |
|---|---|---|---|
| **Cosmos3-Nano-Policy-DROID** | **20.6** | **36.8** | **39.7** |
| Cosmos3-Nano (PT-init) | 16.7 | 28.1 | 30.2 |
| π0.5 | 15.2 | 28.0 | 28.1 |
| DreamZero | 14.9 | 25.7 | 23.9 |
| π0-FAST | 9.2 | 15.5 | 14.9 |
| GR0OT N1.6 | 5.4 | 7.2 | 5.3 |

Specific instructions 下 Cosmos3-Nano-Policy-DROID 39.7% vs π0.5 28.1% vs DreamZero 25.2%——**显著 SOTA**。RoboArena real-world leaderboard 也是 #1。

**MT-init vs PT-init**：mid-trained checkpoint（见过多 domain action 数据）比 pre-trained 初始化收敛快、绝对分数高。LIBERO-10 适应新 embodiment：MT-init 500 iter 就 24.6% success，PT-init 还是 0%。2000 iter MT-init 97.4% vs PT-init 95.2%。

---

## 10. 关键 ablations 给的 intuition

### 10.1 Reasoner 怎么 benefit Generator

把 understanding tower 从 Cosmos3-Nano Reasoner 换成 Qwen3-VL-8B（同样大小），generator tower 从头训。PAIBench T2V Domain score：Cosmos3 Reasoner 75.7 vs Qwen3-VL 73.7，**Robot domain +4.8（66.5 → 71.3）**。Intuition：**Physical AI 专门的 reasoner 给 generator 的 conditioning embedding 更 physically grounded**，quality score 持平说明 gain 不是来自 visual quality。

### 10.2 FPS control

四种配置：Base / Text Control / MRoPE FPS Modulation / Both。Composite score = E[VQ × MF]：

| Setting | VQ | MF | Composite |
|---|---|---|---|
| Base | 12.89 | 0.6626 | 8.51 |
| Text only | 12.99 | 0.7169 | 9.28 |
| MRoPE only | 13.03 | 0.7409 | 9.63 |
| **Both** | 12.84 | **0.7649** | **9.81** |

VQ 几乎不变，所以 gain 全在 motion fidelity。Text + MRoPE 互补：text 显式告诉模型 FPS，MRoPE 在 positional space 对齐物理时间。

### 10.3 Action mode synergy（PushT 数据集）

joint FD/ID/policy 训 6K steps vs 三个 single-mode checkpoint 各 2K steps（per-mode budget 相同）：

| Setting | FD PSNR ↑ | ID MSE ↓ | Policy Coverage ↑ |
|---|---|---|---|
| 2K single-mode | 27.13 | 1.11e-3 | 74.1% |
| 6K joint | 26.22 | **3.09e-4** | **77.3%** |

ID MSE 降 72%，policy coverage +3.2%，FD PSNR 微降。Intuition：**三个 action mode 共享 underlying structure**，joint training 让 action representation 学得更通用。

### 10.4 Action domain synergy matrix

用 PSNR（FD）/ MSE（ID）/ coverage（Policy）做 cross-domain transfer matrix。关键发现：
- Camera motion 从几乎所有 co-training domain 都受益（AV +0.86, Google Robot +0.79 FD PSNR）
- Robot manipulation domains 互相广泛正迁移（WidowX 从 Google Robot +1.39 FD PSNR）
- Egocentric 给 AgiBot warmup 加速：5K step +0.94 PSNR，late training +1.3-1.6 PSNR

Intuition：**action domain 之间有可迁移的 visual-action prior**（object persistence, scene geometry, motion-correspondence），即使 embodiment 不同。

---

## 11. 我的几个 takeaway

1. **Dual-tower MoT 是个稳定的 unified understanding+generation recipe**。比 Transfusion（同 FFN 同 LayerNorm）更彻底分离，比 Show-o / Janus 那种 shared backbone 更明确——reasoner 永远不被 noisy token 污染。代价是参数 ×2，但绝对质量更优。

2. **3D MRoPE + Absolute Temporal Modulation 是处理多模态多采样率的关键 insight**。这是把 RoPE 从 token index 升级到 physical time 的 elegant trick，让 audio (25 TPS) / video (6 TPS @ 24fps) / action (10-30 TPS) 在同一时间轴对齐。15000 AR-DM margin 这种小 hack 解决 first-frame artifact 也很 pragmatic。

3. **Unified action representation（ego + effector + grasp, 6D rotation, domain-specific projection）让一个 backbone 跨 4 个 embodiment**。Robot policy + AV ID + camera FD + egocentric FD 共享一套参数，每 domain 只加 small projection 层。这是 large-scale 的 generalization——证明了 Physical AI 不需要 per-embodiment model。

4. **Synthetic data 是 long-tail coverage 的关键**。SDG-All 混合训练几乎全面提升 PAIBench domain scores。sim-to-real gap 在 Human domain 最明显（即使 SDG-SynHuman 也降 Human score），但 Industrial / Physics / Robot 等 Physical AI 域大幅提升。

5. **Reasoner benefits Generator**——understanding 和 generation 共享 backbone 不是 coincidence。Physical AI reasoner 提供的 conditioning embedding 比 generic VLM 给的更 physically grounded，T2V Robot domain +4.8 PSNR。

6. **Inference 优化叠 7 个 trick**（Cache-DiT, Ulysses CP, CFG-Parallel, HSDP, CPU offload, VAE-Patch-Parallel, FP8）才能把 64B Super 模型部署到合理 latency。这是 model architecture 之外同等重要的工程。

---

## References / 链接

- Paper: https://research.nvidia.com/labs/cosmos-lab/cosmos3
- Code: https://github.com/nvidia/cosmos
- Cosmos3-Super: https://huggingface.co/nvidia/Cosmos3-Super
- Cosmos3-Nano: https://huggingface.co/nvidia/Cosmos3-Nano
- Cosmos3-Super-Text2Image: https://huggingface.co/nvidia/Cosmos3-Super-Text2Image
- Cosmos3-Super-Image2Video: https://huggingface.co/nvidia/Cosmos3-Super-Image2Video
- Cosmos3-Nano-Policy-DROID: https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID
- SDG datasets: https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes (及其余 4 个)
- Cosmos-HUE benchmark: https://huggingface.co/datasets/nvidia/Cosmos-HumanEval-v1
- Mixture-of-Transformers (Liang et al.): https://arxiv.org/abs/2403.09869 (TMLR 2025)
- Transfusion (Zhou et al.): https://arxiv.org/abs/2408.11039 (ICLR 2025)
- 6D rotation representation (Zhou et al. 2019): https://arxiv.org/abs/1812.07035
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- Wan 2.2 video tokenizer: https://arxiv.org/abs/2503.20314
- FlashAttention-3: https://arxiv.org/abs/2407.08609
- DeepSpeed Ulysses: https://arxiv.org/abs/2309.14509
- DROID robot dataset: https://arxiv.org/abs/2403.12945
- RoboLab benchmark: https://arxiv.org/abs/2602.xxxxx (Yang et al. 2026, RSS)
- RoboArena: https://arxiv.org/abs/2506.xxxxx (Atreya et al. 2025, CoRL)
- Physics-IQ: https://arxiv.org/abs/2506.xxxxx (Motamed et al. 2026, WACV)
- PAI-Bench: https://arxiv.org/abs/2512.01989
- Artificial Analysis T2I leaderboard: https://artificialanalysis.ai/text-to-image/arena
- Artificial Analysis I2V leaderboard: https://artificialanalysis.ai/image-to-video/arena
- Lance columnar format: https://arxiv.org/abs/2504.15247
- vLLM-Omni: https://arxiv.org/abs/2603.xxxxx (Yin et al. 2026)
- BAGEL (Deng et al. 2025): https://arxiv.org/abs/2505.14683
- LatentSync (Li et al. 2024): https://arxiv.org/abs/2412.09262
- SAM-Audio (Shi et al. 2025): https://arxiv.org/abs/2512.18099
- FireRedASR2S (Xu et al. 2026): https://arxiv.org/abs/2603.10420
- ViPE (Huang et al. 2025): https://arxiv.org/abs/2508.10934
- DepthAnything3 (Lin et al. 2025): https://arxiv.org/abs/2511.10647

如果你想我深入展开某一块——比如 two-way flat attention 的 kernel 细节、rectified flow matching 的 mode sampling vs logit-normal sampling 数学推导、policy mode 在 15Hz 下怎么和 32 action chunk 对齐 camera frame、或者 ego pose 的 SE(3) relative transform 在 numerical stability 上为什么用 6D 而不是四元数——我可以再钻进去讲。
