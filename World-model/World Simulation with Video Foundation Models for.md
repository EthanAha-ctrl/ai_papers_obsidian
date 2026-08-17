---
source_pdf: World Simulation with Video Foundation Models for.pdf
paper_sha256: 2597170ef9e8128c1ec3c87aa1eaf321cd599268451da1f61aead325eca576fe
processed_at: '2026-08-13T05:07:37-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 Cosmos-Predict2.5

Andrej，我换个方式讲，少堆术语，多讲intuition。

---

## 这玩意儿到底在干嘛

NVIDIA做了一个"**世界做梦机器**"。

给它一句话、一张图、或者一段视频，它能"想象"出接下来5-6秒会发生什么，生成一段physical上合理的视频。

比如你说"机器人把苹果放进碗里"，它就能生成一段机器人真的在这么干的视频。

**干嘛用呢？** 训练机器人和自动驾驶的时候，真机实验太贵太危险。如果在video里能"跑"policy，成本几乎为零，还能随便造edge case。类似于AlphaGo自己跟自己下棋，但这里是机器人自己在"梦境"里练习抓东西、开车。

GitHub: https://github.com/nvidia-cosmos/cosmos-predict2.5

---

## 为什么这事难

你可能会想，Sora、Veo这些不都能生成视频吗？直接拿来用不就行了？

问题在于general video model追求"好看"，physical AI追求"**对**"。比如：

- 机器人抓手抓杯子，杯子不能凭空消失（object permanence）
- 汽车转弯，7个相机的视角必须geometrically一致
- 玻璃碎了，碎片不能往天上飞

Sora生成的视频里，物理常识经常翻车。NVIDIA的目标就是做一个**physical上对**的video model。

---

## 数据：3500万小时怎么洗成200M clips

### 大海捞针

从35M hours raw video里，经过7道工序，最后只留4%——200M clips。

这7道工序我可以"人话"翻译一下：

1. **切镜头** - 长视频按镜头边界切成短clip
2. **转码** - 统一格式
3. **裁黑边** - 去掉画面边缘的black bar
4. **多级过滤** - 6层filter，从cheap到expensive
5. **写caption** - 用VLM给每个clip写描述
6. **去重** - 语义相似的留分辨率最高的
7. **分片** - 按内容类型分桶，方便采样

### 过滤的顺序有讲究

```
便宜 → 贵
Aesthetic → Motion → OCR → Perceptual → Semantic → VLM
```

VLM filter最贵，放最后，只过剩下的4%。工程上的常识——先用cheap filter砍掉大部分，再上expensive的精修。

### 最关键的决策

**排除所有"假物理"内容**：video games, animations, cartoons, synthetic patterns全部踢掉。

为什么？因为这些内容不符合physical world的规律。如果model学了Mario跳跃的物理，让它模拟真机器人抓东西就完蛋了。

### 五个domain的定制数据

| Domain | 数据规模 | 特点 |
|--------|---------|------|
| Robotics | AgiBot/Bridge/DROID/GR00T等 | 按camera视角分，强制统一视角描述 |
| Driving | 3.1M clips, 7 cameras | 20秒一段，覆盖天气/路况/光照 |
| Smart Spaces | 40K clips | 工厂、仓库场景 |
| Human Dynamics | YOLOX+RTMPose筛选 | 人体运动，提升robot对人的模拟 |
| Physics | 按力学taxonomy curate | 碎玻璃、滚球、流水等physical phenomena |

Driving captioning特别讲究，要描述6个aspect：agents、环境、meta actions、速度、状态变化、interactions。

---

## Flow Matching：教AI学会"从噪声走到真实"

### 核心比喻

想象你站在一片纯噪声的"迷雾"里，要走到"真实视频"那个点。你不知道路怎么走，但有个老师告诉你每一步的**速度向量**（velocity）该往哪指。

训练就是让DiT network学会预测这个velocity。推理时，从纯噪声出发，按照学到的velocity一步步走，最终走到一个真实视频。

### 数学上

给定干净数据 $x$ 和噪声 $\epsilon$，在时间 $t$ 上的interpolated point：

$$x_t = (1-t)x + t\epsilon$$

- $t=0$: 完全是clean data
- $t=1$: 完全是noise
- 中间值: clean和noise的线性混合

ground-truth velocity就是这条直线对 $t$ 的导数：

$$v_t = \epsilon - x$$

training loss就是让network预测的velocity $\mathbf{u}(x_t, t, c; \theta)$ 逼近 $v_t$：

$$\mathcal{L} = \mathbb{E}\left[\|\mathbf{u} - v_t\|^2\right]$$

### 为什么选FM不选EDM

Predict1用的EDM ([Karras 2022](https://arxiv.org/abs/2206.05029))，Predict2.5换成FM ([Lipman 2022](https://arxiv.org/abs/2210.02727))。

两者forward/backward process数学上等价，区别在network预测什么：
- **EDM**: 预测denoised signal，input/output都被precondition成standardized Gaussian，好处是stable
- **FM**: 直接预测velocity，training target更直接，optimization更smooth

NVIDIA的实验发现FM在实践里sample quality更好。这跟Stable Diffusion 3、Flux等用FM是同一个趋势。

### Shifted Logit-Normal：高分辨率的"冗余陷阱"

这是个很subtle但重要的点。

**问题**：高分辨率video里，相邻pixel高度correlated（天空一大片蓝、桌面一大片木纹）。如果加的噪声太少，model看到的是"几乎干净"的input，无法"打散"这种correlation，学不到meaningful structure。

**解法**：故意让训练时high-noise region出现得更频繁。

具体做法——先从logit-normal分布采 $t$，然后做变换：

$$t_s = \frac{\beta t}{1 + (\beta - 1)t}$$

- $\beta = 1$: no shift，$t_s = t$
- $\beta > 1$: $t_s > t$，把分布推向更高noise
- 训练时progressive增加：256p用 $\beta=1$，720p用 $\beta=5$

**Intuition**: 高分辨率内容冗余多，得"狠狠加噪"才能让model见到足够diversity的input。

### 5% explicit high-noise sampling

即使用了shifted distribution，还观察到abrupt frame transitions的artifact。Hypothesis是high-noise region样本还是太少。

解法很暴力——**5%的训练样本直接从最高2%的noise distribution里采**。这种targeted sampling显著减少了transition artifacts。

这是工程经验，paper没给ablation说为什么是5%和2%，但效果显著。

---

## 架构：DiT + 去掉绝对位置编码

### 核心改动

相比Predict1，架构上最大的变化是**去掉absolute positional embedding，只保留3D RoPE**。

为什么？Absolute embedding给每个位置一个fixed reference，但这就限制了model——训练时见过的resolution/length就定死了，换一个新的就generalize不好。

去掉之后，model只靠relative position（RoPE）工作，对unseen resolution和length的flexibility大增。这个insight来自long-context LLM的progress（[YaRN](https://arxiv.org/abs/2309.00071), [NTK-aware scaling](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/)）。

### 视觉tokenizer

用WAN2.1 VAE，compression rate $4 \times 8 \times 8$（time × H × W）：
- 时间压缩4x：93帧 → 24 latent frames
- 空间各压缩8x
- 再加 $1 \times 2 \times 2$ patchification

16fps下93帧 ≈ 5.8秒video，这是model一次能生成的长度。

### 文字编码器：Cosmos-Reason1

Predict1用T5，Predict2.5换成[Cosmos-Reason1](https://arxiv.org/abs/2503.15558)——NVIDIA自家的Physical AI专用VLM。

关键设计：不是取最后一层output，而是**concatenate多个block的activations**，project到1024维。

Intuition: 不同层capture不同granularity的信息，concat起来更rich。这跟DALL-E 3、SD3用CLIP+T5多encoder的思路类似，但这里是单model多layer。

Cosmos-Reason1的vision encoder还能接visual input做style control，paper说留作future work——这是个值得关注的hint。

### 三种模式一个模型

一个model同时支持：
- **Text2World**: 只有文字 → 视频
- **Image2World**: 文字 + 一张图 → 视频  
- **Video2World**: 文字 + 前几帧 → 后续帧

怎么做到的？**Frame-replacement strategy**：生成序列的initial frames直接被conditioning frames替换。mask机制告诉model哪些frame是给定的，loss只算在要生成的frames上。

好处是early frames保证faithful to conditioning input，后续frames的visual cue能更smooth地propagate。

---

## 训练：渐进式 + 分domain + Model Soup + RL

### 5个stage的渐进pre-training

```
Stage 1: 256p, Text2Image, 1帧        ← 先学单帧质量
Stage 2: 256p, + Video2World, 93帧    ← 引入时序
Stage 3: 480p                          ← 提分辨率
Stage 4: 720p                          ← 最终分辨率
Stage 5: + Text2World                  ← 加入纯text生成
```

每个stage收敛后才进入下一个。Stage 5引入Text2World时，condition frames采样概率是0.5/0.25/0.25（0帧/1帧/2帧）。

### Domain-specific SFT

Pre-training完，给5个domain（object permanence, high motion, complex scenes, driving, robotic manipulation）分别fine-tune一个model。

**关键决策：不joint train，而是分别train**。

为什么？Joint train要balance mixture ratio，很麻烦。Separate train能充分利用每个domain的数据，性能更好。General domain的slight degradation可以通过model merging缓解。

### Model Soup：简单平均就够好

每个domain的SFT model + 一个4K cooldown model，怎么merge成一个？

试了4种方法：
1. **Model Soup** - 简单averaging weights ([Wortsman 2022](https://arxiv.org/abs/2203.08283))
2. TIES - 解决parameter interference ([Yadav 2023](https://arxiv.org/abs/2306.01708))
3. DARE-Linear ([Yu 2024](https://arxiv.org/abs/2311.03099))
4. DARE-TIES

跑了20+个merged model的hyperparameter sweep，发现**Model Soup（最简单的）效果最好**。

更有意思的发现：grid search over hyperparameters consistently outperforms heuristic selection based on individual models' win rates。意思是，不要根据single model的表现来猜怎么merge，老老实实grid search。

### RL Post-training：把RLHF思路搬到diffusion

这part我个人觉得最有意思。

**思路**: 把conditioning当成state，denoising trajectory当成action，用RL让model生成更符合人类偏好的视频。

用[VideoAlign](https://arxiv.org/abs/2501.13918)做reward model，评估text alignment / motion quality / visual quality三个维度。

算法是GRPO-style ([DeepSeek-R1](https://arxiv.org/abs/2501.12948)同款)：
- 每个condition生成8个outputs
- 在group内normalize reward算advantage
- Trajectory probability分解成每步conditional probability的乘积
- 受GPU memory限制，每2步accumulate一次gradient，整个trajectory（10步）一次update

**关键trick: diffusion loss regularization**。在fine-tuning dataset上同时算diffusion loss，防止reward hacking——否则model可能找到一个让reward model高分但实际很烂的"捷径"。

结果：Text2World的reward sum从1.08涨到1.69，human voting也confirm RL model更preferred。

这个方向我预测会越来越火，类比RLHF for LLM的trajectory。

### Timestep Distillation：4步生成

用rCM ([Zheng 2025](https://arxiv.org/abs/2510.08431))做distillation，把推理步数从几十步压到4步，质量几乎不降。

为此专门搭了infrastructure：fused flash attention with JVP support, FSDP2 adaptation, context parallelism adaptation。

---

## Infrastructure的工程细节

4096张H100的training setup：

- **FSDP2**: per-parameter sharding，比FSDP1的bucket-based更细粒度，memory管理更好
- **Ulysses-style Context Parallelism**: 比ring-attention更communication-efficient，intra-node all-to-all
- **Selective Activation Checkpointing**: 轻量op（element-wise, normalization）优先recompute
- **Elastic Reward Service**: RL post-training时用，producer-consumer pipeline，CUDA IPC zero-copy

MFU：2B 36.49%，14B 33.08%（14B因为context parallelism=8，communication cost更高）。

---

## Cosmos-Transfer2.5：加"控制手柄"

Predict是"自由生成"，Transfer是"受控生成"——你给edge/depth/segmentation/blur map，它生成对应的真实视频。类似ControlNet但更multi-modal。

### 架构小改动

Transfer1把4个control block全堆在main branch开头。Transfer2.5改成**每7个block后插1个control block**，更均匀地逐渐注入conditioning。

### 数据规模

| Modality | Videos | Tool |
|----------|--------|------|
| Depth | 10M | [Video Depth Anything](https://arxiv.org/abs/2503.23554) |
| Segmentation | 3M | [SAMv2](https://arxiv.org/abs/2408.00714) |
| Edge | 14M | - |
| Blur | 14M | - |

### 效果：3.5x更小但更好

Transfer2.5-2B Uniform Weights vs Transfer1-7B：
- Quality Score: 9.31 vs 9.24
- Edge F1: 0.41 vs 0.26（提升明显）
- Blur SSIM: 0.87 vs 0.82

小model赢大model，归功于：更强的Predict2.5 base + 更curated的Physics AI data。

### Long Video：RNDS指标

新提出**averaged Relative Normalized Dover Score (RNDS)**：

$$\mathsf{RNDS}[i] = \frac{\mathrm{DOVER}[i] / \mathrm{DOVER}_{GT}[i]}{\mathrm{DOVER}[1] / \mathrm{DOVER}_{GT}[1]}$$

- $i$: chunk index
- $\mathrm{DOVER}[i]$: 第$i$个生成chunk的质量分数
- $\mathrm{DOVER}_{GT}[i]$: ground truth的分数
- Normalization让曲线从(1,1)开始，方便对比degradation趋势

结果：Transfer2.5的RNDS曲线降幅远小于Transfer1，说明长视频error accumulation更少。这对closed-loop simulation至关重要。

---

## 应用：五个direction的实战

### 1. Robot Policy Learning - 最impressive的结果

**Setup**: 双臂机器人（Kinova Gen3 + Robotiq gripper）+ head camera，100个teleoperation demos训练diffusion policy。

**Augmentation策略**：
- Global edge control across整个image
- Blur control只apply到robot pixels（用Grounding DINO + SAMv2 isolate robot）
- 用LLM生成prompt variations：换苹果颜色、换碗颜色、换桌面、换光照、换背景

**9个unseen test scenarios + 1 combo**：山竹代替苹果、橙色碗、米色桌布、黑色桌布、加spotlight、加distractor、黑色柜子、打开抽屉、前三者组合。

**结果**：
- Base policy: 1/30
- Standard image augmentation: 5/30
- **Cosmos-Transfer2.5 augmentation: 24/30**

Key insight: Standard image augmentation（brightness/contrast/saturation）做不了semantic edit——改不了物体颜色、环境appearance、lighting。Transfer2.5可以。这个gap在real-world deployment中是质的差别。

### 2. Driving Multi-view Simulation

**架构trick**: 把7个view的latents concatenate到temporal dimension——相当于把views当成sequential frames。为此把latent temporal dim从24减到8，才能fit 7 views。

每个view独立encode，concatenate一个size 7的learnt embedding到channel dim，3D RoPE per view单独construct。

**Control input**: "World Scenario Map"——HD map elements（lane lines, poles, traffic lights）+ dynamic 3D bounding boxes（color-coded by class, shaded by heading）。

**结果**：
- FVD/FID提升2.3x
- 3D cuboid和lane detection on生成视频的metrics提升60%
- 接近real video reference

这意味着生成的driving video已经"真实"到能训BEVFormer、LATR这种perception model了。

### 3. Camera-controllable Multi-view

给定source video + N个target camera轨迹，生成N个target videos。

**Camera representation**: Plücker raymaps——每个pixel映射到6D ray，patchify后通过projection layer加到video tokens上。

**两个variant**：
- **agibot**: head-view → left/right gripper views
- **multiview**: third-view → 两个under camera transformation的view

Multiview比single-view baseline的Sampson Error从26.61降到19.73 px，cross-view consistency更好。

### 4. Synthetic Data for VLA Training

生成"机器人执行unseen指令"的视频 → 用inverse dynamics model提取pseudo-action → 得到(vision, language, action)三元组训VLA。

在[DreamGen benchmark](https://arxiv.org/abs/2505.12705)上，Cosmos-Predict2.5-14B的instruction following score：
- Object GPT: 91.8（Hunyuan 38.0, CogVideoX 72.0, WAN2.1 72.0）
- Behavior GPT: 70.2
- Env GPT: 69.0

这个paradigm——用world model生成synthetic VLA data——我觉得是scaling embodied intelligence的关键enabler之一。

### 5. Action-Conditioned World Generation

输入：1张图 + 一串robot actions → 输出：follow action的future frames。

**Action注入方式ablation**（Bridge dataset, 7D action: $\Delta x, \Delta y, \Delta z, \Delta\theta_r, \Delta\theta_p, \Delta\theta_y$, GripperWidth）：

| Method | PSNR | FVD |
|--------|------|-----|
| **TimeEmbedding** (proposed) | 24.95 | 146 |
| CrossAtten | 24.41 | 159 |
| ChannelConcat | 23.11 | 267 |

TimeEmbedding最好——action信息通过timestep modulation注入DiT。这个结论跟我预期一致，modulation-based conditioning在diffusion里通常比cross-attention或concat更efficient。

---

## 效果：小模型打大模型

### PAI-Bench ([Zhou 2025](https://arxiv.org/abs/2512.01989))

**Text2World Overall**:
- Cosmos-Predict2.5-2B post: 0.768
- Cosmos-Predict2.5-14B post: 0.768
- Wan2.2-27B-A14B: 0.769 ← 打平

**Image2World Overall**:
- Cosmos-Predict2.5-2B post: **0.810** ← best
- Cosmos-Predict2.5-14B post: **0.810** ← best
- Wan2.2-27B-A14B: 0.806

### Human Evaluation

2B model（比Wan2.2 5B小60%，比Wan2.1 14B小85.7%）：
- vs Wan2.2 5B: 30.0% vs 26.2% ← 更受欢迎
- vs Wan2.1 14B: 33.0% vs 34.8% ← 打平

14B model：
- vs Wan2.1 14B: 48.6% vs 31.8% ← 大幅胜出
- vs Wan2.2 27B-A14B（参数一半）: 38.1% vs 35.9% ← 打平

**Key insight**: 2B→14B的scaling benefit在human evaluation里比自动metric更明显（vs Wan2.1 14B从33%涨到48.6%）。说明大model的improvement在细微quality维度上，自动metric可能catch不到。

---

## 我的核心takeaways

1. **Video generation → Physical engine的paradigm shift**: 这paper本质是在说，video model可以当作physical world的simulator。跟NEES、IsaacSim这种rule-based simulator不同，这里是learned from data的。Trade-off：更realistic但less controllable。

2. **Data curation决定上限**: 4% survival rate + 排除fake physics + domain定制captioning。这跟LLM的"garbage in garbage out"同理，但在video domain更severe，因为physical plausibility很难从loss里学出来。

3. **Flow Matching + shifted noise schedule**: high-resolution的redundancy问题用bias toward high noise解决，这是个universal insight，不只video，image diffusion也适用。

4. **Domain SFT + Model Soup > Joint training**: 简单averaging就能beat TIES/DARE这种复杂方法，再次印证"simple wins"。

5. **RL for diffusion**: GRPO-style + diffusion loss regularization的recipe，类比RLHF for LLM。这条路会越来越重要，尤其是reward model变强之后。

6. **Multi-view的temporal dimension hack**: 把views当sequential frames塞到temporal dim，是个memory-efficient的工程方案，但可能不是最elegant。未来可能需要native multi-view architecture。

7. **Action conditioning via time embedding**: modulation > cross-attention > channel concat，这个结论对robotics world model设计有参考价值。

8. **开源的意义**: 在Sora/Veo/Kling closed-source主导的领域，NVIDIA open source 2B/14B models + code + benchmarks，给academic和startup提供了infrastructure。这跟Meta开源LLaMA的战略类似。

9. **Closed-loop是next frontier**: Paper提到Transfer2.5支持closed-loop simulation，但没给细节。我猜是generate→perceive→act→generate的loop，但这要求long-horizon stability，RNDS metric就是为了量化这个。

10. **VLA synthetic data**: 用world model生成video再反推action的paradigm，可能解决robotics data scarcity的瓶颈。跟[DreamGen](https://arxiv.org/abs/2505.12705)、[GR00T-Dream](https://arxiv.org/abs/2503.14734)是同一个方向。

---

如果某个点你想deep dive，比如：
- Flow Matching的probability path ODE推导
- RL post-training的gradient decomposition数学
- Multi-view RoPE的具体实现
- Plücker raymap的camera conditioning细节

我可以再展开。这篇paper工程密度很高，每个section背后都有不少没写进paper的trick。

---

# Cosmos-Predict2.5: NVIDIA的Physical AI World Foundation Model深度解析

Andrej你好，这篇paper是NVIDIA继Cosmos-Predict1之后推出的第二代video world foundation model，针对Physical AI场景做了大量优化。我从技术细节层面给你拆解。

## 1. 核心动机与定位

Physical AI（robotics, autonomous driving等embodied agents）的训练面临real-world data采集成本高、风险大的问题。World simulator可以作为physical world的safe proxy，让agent在silicon里学perception和control。Cosmos-Predict2.5的目标就是做一个high-fidelity的video world simulator。

相比Cosmos-Predict1的三个关键改进：
- **数据**：stricter filtering pipeline，survival rate从30%降到4%
- **架构**：去掉absolute positional embedding，unify Text2World/Image2World/Video2World到一个model
- **训练**：model merging + RL post-training，用Cosmos-Reason1替代T5

GitHub repos:
- https://github.com/nvidia-cosmos/cosmos-predict2.5
- https://github.com/nvidia-cosmos/cosmos-transfer2.5

---

## 2. Data Pipeline - 200M curated clips的炼成

### 2.1 整体pipeline架构

7个stage的pipeline，从35M hours raw video（vs Predict1的20M hours）中提炼出200M trainable clips：

1. **Shot-aware splitting** - 用boundary detection model切分long-form video
2. **GPU-based transcoding** - 加速转码
3. **Video cropping** - 去black borders
4. **Multi-stage filtering** - 多级过滤
5. **Captioning** - VLM生成caption
6. **Semantic deduplication** - 嵌入相似度去重
7. **Sharding** - 沿content type/resolution/aspect ratio/length分片

### 2.2 关键filter设计

Filtering pipeline的顺序很关键（从cheap到expensive）：
1. Aesthetic scoring filter
2. Motion filter - 量化运动程度
3. OCR filter - 去文字overlay
4. Perceptual quality filter (DOVER-like, [Wu et al., 2023](https://arxiv.org/abs/2306.11894))
5. Semantic artifacts filter (VTSS-like)
6. VLM filter (Qwen2.5-VL-7B) - 最expensive放最后

特别有意思的是：**explicitly排除physically unrealistic content**（video games, synthetic visual patterns, animations, cartoons），保持与physical world distribution对齐。这对于Physical AI至关重要。

### 2.3 Domain-specific data curation

5个target domain的定制化pipeline：
- **Robotics**: AgiBot-Beta, Bridge, DROID, GR00T, 1X, OpenX, RoboMIND
- **Autonomous Driving**: 3.1M 20-second 7-camera surround-view clips
- **Smart Spaces**: 40K warehouse/factory clips
- **Human Dynamics**: YOLOX + RTMPose筛选有人类运动的clips
- **Physics**: 专门针对classical mechanics和fluid mechanics的taxonomy

Driving captioning的6个aspect值得关注：
1. Agents and traffic elements
2. Global environmental factors  
3. Meta actions (longitudinal/lateral)
4. Speed
5. Dynamic state transitions
6. Interactions

---

## 3. Flow Matching数学详解

### 3.1 FM vs EDM的本质区别

Cosmos-Predict2.5从EDM ([Karras et al., 2022](https://arxiv.org/abs/2206.05029))切换到Flow Matching ([Lipman et al., 2022](https://arxiv.org/abs/2210.02727))。两者在forward/backward diffusion process上数学等价，区别在denoising network的parameterization：

- **EDM**: preconditioning使得network的input/output都是approximately standardized Gaussians，简化training，提升stability
- **FM**: network直接predict diffusion trajectory的velocity，更直接的training target，smoother optimization

### 3.2 核心公式

给定data sample $x$（image或video），noise vector $\epsilon \sim \mathcal{N}(0, I)$，timestep $t \in [0, 1]$：

**Interpolated latent**:
$$\mathbf{x}_t = (1-t)\mathbf{x} + t\boldsymbol{\epsilon}$$

其中：
- $\mathbf{x}$: 原始数据sample（clean signal）
- $\boldsymbol{\epsilon}$: 从标准高斯分布采样的noise
- $t$: timestep，$t=0$时完全是clean data，$t=1$时完全是noise
- $\mathbf{x}_t$: 在trajectory上的interpolated point

**Ground-truth velocity**:
$$\mathbf{v}_t = \boldsymbol{\epsilon} - \mathbf{x}$$

这是从$\mathbf{x}$到$\boldsymbol{\epsilon}$的线性插值的velocity（对$t$求导即得）。

**Training objective**:
$$\mathcal{L}(\boldsymbol{\theta}) = \mathbb{E}_{\mathbf{x}, \boldsymbol{\epsilon}, \mathbf{c}, t} \left\| \mathbf{u}(\mathbf{x}_t, t, \mathbf{c}; \boldsymbol{\theta}) - \mathbf{v}_t \right\|^2$$

变量说明：
- $\boldsymbol{\theta}$: model parameters
- $\mathbf{c}$: conditioning information（text embeddings, reference frames等）
- $\mathbf{u}(\cdot; \boldsymbol{\theta})$: predicted velocity function（DiT network）
- $\mathbf{v}_t$: ground-truth velocity
- 优化目标是MSE between predicted和ground-truth velocity

### 3.3 Shifted Logit-Normal Distribution - 关键创新

**问题**：High-resolution content包含大量redundancy（相邻pixel高度correlated）。如果noise level太小，model无法"break apart"这种correlation，难以学到meaningful structure。

**解决方案**：bias training toward higher noise levels，采用shifted logit-normal distribution ([Esser et al., 2024](https://arxiv.org/abs/2403.03206))。

首先从logit-normal distribution采样$t$，然后apply monotone transformation：

$$t_s = \frac{\beta t}{1 + (\beta - 1)t}$$

变量说明：
- $t$: 原始logit-normal采样值，$\in [0,1]$
- $\beta$: shift hyper-parameter，控制noise level偏移程度
- $t_s$: shifted timestep，skewed toward higher noise

**Intuition**: 
- $\beta = 1$时，$t_s = t$，no shift
- $\beta > 1$时，$t_s > t$，push model encounter noisier inputs更frequently
- 帮助model学会在correlation被heavily disrupted时reconstruct signal

**Progressive shift schedule**: 
- 256p: $\beta = 1$
- 480p: 中间值
- 720p: $\beta = 5$

### 3.4 高noise region的额外采样

观察到abrupt transitions between frames的artifact，hypothesis是high-noise region训练样本太少。修改scheduler使得**5%训练样本explicitly从highest 2% noise distribution采样**。这个targeted sampling显著减少了transition artifacts。

---

## 4. Network Architecture详解

### 4.1 整体DiT架构

基于latent diffusion model，denoising network沿用Cosmos-Predict1的DiT，但有关键改动。

**Configuration对比**:

| Config | 2B | 14B |
|--------|-----|------|
| Layers | 32 | 36 |
| Model Dim | 2,048 | 5,120 |
| FFN Hidden | 8,192 | 20,480 |
| AdaLN-LoRA Dim | 256 | 256 |
| Attention Heads | 16 | 40 |
| Head Dim | 128 | 128 |
| Activation | GELU | GELU |
| Positional Embed | 3D RoPE | 3D RoPE |

### 4.2 关键架构改动

**去掉absolute positional embeddings**，只保留relative positional embeddings (3D RoPE)。

**为什么**：absolute embeddings提供fixed spatial/temporal reference，限制model对unseen resolution/sequence length的generalization。去掉后获得更大flexibility，可以handle higher-resolution content和longer video sequences during post-training。

这个设计借鉴了long-context LLMs的progress（[YaRN](https://arxiv.org/abs/2309.00071), [NTK-aware scaling](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/))。

### 4.3 Visual Tokenizer - WAN2.1 VAE

使用[WAN2.1 VAE](https://arxiv.org/abs/2503.20314)，一个causal VAE，compression rate $4 \times 8 \times 8$（time × height × width）。

- Time维度压缩4x
- 空间维度各压缩8x
- 在此之上再apply $1 \times 2 \times 2$ patchification

**训练规模**：93 pixel frames对应24 latent frames，16 fps，约5.8秒video。

### 4.4 Text Encoder - Cosmos-Reason1

用[Cosmos-Reason1](https://arxiv.org/abs/2503.15558)替代T5（Predict1用的）。

**关键设计**：
- 不是用single transformer layer的output，而是concatenate多个block的activations for each token
- Project到1024-dimensional space（受[Wang et al., 2025](https://arxiv.org/abs/2502.20739)启发）
- 通过cross-attention layers integrate到denoising process
- Cosmos-Reason1的vision encoder还支持visual conditional inputs for style control（留作future work）

**为什么更好**：Cosmos-Reason1是specialized for Physical AI的decoder-only VLM，provides richer text representations和finer-grained control over world generation。

### 4.5 三种模式统一

一个model支持三种模式：
- **Text2World**: 只用text prompt引导
- **Image2World**: text + reference image
- **Video2World**: text + video sequence（temporally coherent continuation）

**Frame-replacement strategy**: 对Image2World和Video2World，generated sequence的initial frames被consistently substituted with conditioned frames。两个purpose：
1. Flexibility - conditioned frames数量可调
2. Strengthen temporal consistency - early frames保持faithful to conditioning

### 4.6 Conditioning机制

DiT被conditioned by concatenating ground-truth frames with noisy frames。用masking scheme指定哪些是conditional input：每个input token由original token + mask token拼接，mask是binary flag。Denoising loss只apply到designated frames。

---

## 5. Training Recipe详解

### 5.1 Progressive Pre-training

5个stage的渐进训练（Tab. 4）：

| Stage | Task | Resolution | Frames |
|-------|------|-----------|--------|
| 1 | Text2Image | 256p (320×192) | 1 |
| 2 | Text2Image \| Video2World | 256p | 1 \| 93 |
| 3 | Text2Image \| Video2World | 480p (832×480) | 1 \| 93 |
| 4 | Text2Image \| Video2World | 720p (1280×704) | 1 \| 93 |
| 5 | Text2Image \| Video2World \| Text2World | 720p | 1 \| 93 \| 93 |

**关键设计**：
- 从Text2Image开始学高质量单帧生成
- 引入Image2World/Video2World时，randomly sample 1或5 conditioning frames
- Stage 5加入Text2World时，sample 0/1/2 condition frames概率为0.5/0.25/0.25

**Optimizer**: AdamW, $\beta_1=0.9$, $\beta_2=0.999$, weight decay 0.001
- 2B: LR = $3 \times 10^{-5}$
- 14B: LR = $1.3 \times 10^{-5}$
- Linear decay + 2000 iter warmup

### 5.2 Supervised Fine-tuning (SFT)

**Domain分类**（基于InternVideo2 embeddings训练multi-head classifier）：

| Domain | Videos |
|--------|--------|
| Object Permanence | 10.4M |
| High Motion | 1.0M |
| Complex Scenes | 1.6M |
| Driving | 3.1M |
| Robotic Manipulation | 730K |
| 4K | 388K |

**关键策略**：为每个domain单独fine-tune一个model，而不是joint training。好处：
- 充分利用data，不需要balance mixture ratios
- Domain-specific SFT大幅提升specialized domain性能
- General domain的slight degradation可以通过model merging缓解

每个specialized model训练30k iterations，batch size 256。

### 5.3 Model Merging

尝试4种方法：
1. **Model Soup** ([Wortsman et al., 2022](https://arxiv.org/abs/2203.08283)) - 简单averaging
2. **TIES** ([Yadav et al., 2023](https://arxiv.org/abs/2306.01708)) - 解决interference
3. **DARE-Linear** ([Yu et al., 2024](https://arxiv.org/abs/2311.03099))
4. **DARE-TIES**

**有趣发现**：简单grid search over hyperparameters consistently outperforms heuristic selection based on individual fine-tuned models' win rates。最终选择**Model Soup** variant（effectiveness + simplicity）。

还有**cooldown stage**：用4K videos，LR linearly decay to zero，增强fine-grained visual detail和smoother motion。

### 5.4 Reinforcement Learning Post-training

借鉴RLHF思路，但applied to flow-based world generation。

**核心思想**：把conditions视为states，整个denoising trajectories视为actions。

**Reward model**: [VideoAlign](https://arxiv.org/abs/2501.13918)（VLM-based），评估：
- Text alignment
- Motion quality  
- Visual quality

**算法**: GRPO-style ([Guo et al., 2025](https://arxiv.org/abs/2501.12948))
- 每个input condition生成8个outputs，20 diffusion steps
- 在rollout group内normalize reward计算advantage
- 由于GPU memory constraint，trajectory probability分解为每步conditional probability的sum
- 实际中每两个conditional probabilities计算gradient，accumulate整个trajectory（10步）的gradient做一次parameter update
- 训练256 steps，batch size 32

**Regularization**: diffusion loss on fine-tuning dataset，alleviate reward hacking（详见[Ye et al., 2025](https://arxiv.org/abs/2512.04332)）

**结果**（Tab. 6）：
- Text2World: pre-train model的sum reward从1.08提升到1.69
- Image2World: 从0.23提升到0.42
- Human voting也confirm RL models更preferred

### 5.5 Timestep Distillation

用rCM ([Zheng et al., 2025](https://arxiv.org/abs/2510.08431)) - hybrid forward-reverse joint distillation framework，integrates:
- Continuous-time consistency distillation
- Distribution matching distillation

**Infrastructure支持**：
- Fused flash attention with Jacobian-vector product (JVP) support
- FSDP2 adaptation
- Context parallelism adaptation

**结果**：4 steps即可produce high-fidelity samples，quantitative结果与teacher model类似（Tab. 7, 8）。

### 5.6 Infrastructure

**FSDP2** - per-parameter sharding（vs FSDP1的bucket-based），finer-grained memory management。结合[TorchTitan](https://arxiv.org/abs/2510.00571)优化：
- Asynchronous distributed checkpointing
- Meta-device initialization

**Context Parallelism** - Ulysses-style ([Rasley et al., 2020](https://arxiv.org/abs/2002.06695))，比ring-attention更communication-efficient，支持NATTEN sparse attention和fused flash attention with JVP。

**Selective Activation Checkpointing** - 轻量operator（element-wise, normalization）优先recomputation。

**MFU**（Tab. 9）：
- 2B: 36.49%（4096 H100, context parallelism=2）
- 14B: 33.08%（context parallelism=8，communication cost增加）

---

## 6. Cosmos-Transfer2.5 - ControlNet风格框架

### 6.1 架构差异

对比[Cosmos-Transfer1-7B](https://arxiv.org/abs/2503.14492)：
- **Transfer1**: 4个control blocks sequentially在main branch开始
- **Transfer2.5**: 4个control blocks均匀分布，每7个blocks后插入1个control block

**好处**：preserves total control blocks数量，但更gradually integrate conditioning information throughout network。

### 6.2 Control Modalities

| Modality | 数据规模 | 工具 |
|----------|---------|------|
| Depth | 10M videos | [Video Depth Anything](https://arxiv.org/abs/2503.23554) |
| Segmentation | 3M videos | [SAMv2](https://arxiv.org/abs/2408.00714) |
| Edge | 14M videos | - |
| Blur | 14M videos | - |

每个control branch独立训练100k iterations，batch size 64。

### 6.3 量化结果（Tab. 12）

Cosmos-Transfer2.5-2B **Uniform Weights** vs Transfer1-7B Uniform Weights：
- Blur SSIM: 0.87 vs 0.82
- Edge F1: 0.41 vs 0.26
- Depth si-RMSE: 0.67 vs 0.70
- Seg mIoU: 0.76 vs 0.74
- Quality Score: 9.31 vs 9.24

**3.5x更小但性能更好**，归因于：
1. 更强的Cosmos-Predict2.5-2B base model
2. 更精心curated的Physics-AI-focused training data

### 6.4 Long Video Generation - RNDS指标

新提出**averaged Relative Normalized Dover Score (RNDS)** 评估error accumulation：

$$\mathsf{RNDS}[i] = \left(\frac{\mathrm{DOVER}[i]}{\mathrm{DOVER}_{\mathrm{GT}}[i]}\right) / \left(\frac{\mathrm{DOVER}[1]}{\mathrm{DOVER}_{\mathrm{GT}}[1]}\right)$$

变量说明：
- $i = 1, \dots, T$: chunk index
- $\mathrm{DOVER}[i]$: 第$i$个generated chunk的[Dover score](https://arxiv.org/abs/2306.11894)
- $\mathrm{DOVER}_{\mathrm{GT}}[i]$: 对应ground-truth video的Dover score
- Normalization确保RNDS曲线总是从$(1,1)$开始

**结果**：Cosmos-Transfer2.5-2B的RNDS曲线降幅远小于Transfer1-7B，说明less hallucination、less error accumulation、higher long-video fidelity。

---

## 7. 应用场景

### 7.1 Robot Policy Learning - Real2Real Augmentation

**Setup**: Bimanual robot (两个Kinova Gen3 arm + Robotiq gripper) + head-mounted RealSense D455，用Meta Quest 2 teleoperation收集100 demonstrations，训练UNet-based [Diffusion Policy](https://arxiv.org/abs/2303.04137)。

**Augmentation策略**：
- Global edge control across entire image
- Blur control restricted to robot pixels（用Grounding DINO + SAMv2 isolate robot）
- Edge threshold: medium, Blur threshold: very low, CFG scale: 3

**Prompt template** (用LLM generate variations):
```
The scene depicts a bright, modern kitchen... a robot faces [TABLE]. 
On the table rest [COLOR_APPLE] apple and [COLOR_BOWL] bowl. 
[SENTENCE_LIGHT] In the background are... [SENTENCE_BACKGROUND]...
```

**结果**（Tab. 13） - 9个unseen test scenarios + 1 combo：
- Base policy: 1/30
- Standard image augmentation: 5/30  
- **Cosmos-Transfer2.5 augmentation: 24/30**

关键insight：Standard image augmentation（brightness, contrast等）无法做semantic edits（改object color, environment appearance, lighting），而Transfer2.5可以自然处理。

### 7.2 Driving Simulation - Multi-view Generation

**架构trick**: 把multiple views concatenate along temporal dimension（views as sequential frames），latent temporal dimension减到8以fit 7 views。

每个view：
- 独立encode/decode
- Concatenate per-view learnt embedding (size 7) in latent channel dimension
- 3D-factorized RoPE（per view separately construct）
- Cross-attention with text embeddings

**World Scenario Map**作为control input：HD maps + dynamic 3D bounding boxes，color-coded by object type，shaded by motion direction。

**训练数据**：
- Predict2.5-2B/auto/multiview: 1.5M clips, 7 cameras, 30FPS
- Transfer2.5-2B/auto/multiview: [RDS-HQ](https://arxiv.org/abs/2506.09042), 140K 20-second scenes

**结果**（Tab. 14, 15）：
- FVD/FID提升up to 2.3x
- 3D cuboid和lane detection metrics提升up to 60%
- 生成video上的检测性能接近real video reference

### 7.3 Multi-view Generation with Camera Control

**目标**: 给定source video + $N$个target camera轨迹，synthesize $N$个target videos。

**Camera representation**: Plücker raymaps ([Sitzmann et al., 2021](https://arxiv.org/abs/2106.05507))，pixels映射到6D ray representations，patchify后通过camera projection layer align到video latents维度，加到video tokens上prior to self-attention。

**训练数据**:
- [AgiBot](https://arxiv.org/abs/2506.11894): 145,820 episodes, 3 views
- MultiCamVideo: 136K episodes, dynamic camera
- SynCamVideo: 34K episodes, static novel viewpoints

**两个variant**：
- **Transfer2.5-2B/robot/multiview-agibot**: head-view → left/right gripper views
- **Transfer2.5-2B/robot/multiview**: third-view → two synchronized views under camera transformations

**结果**（Tab. 17）：Multi-view比single-view baseline的Sampson Error从26.61降到19.73 px，camera accuracy相当。

### 7.4 Synthetic Data for VLA Training

用[Cosmos-Predict2.5-14B/robot/gr00tdream-gr1](https://arxiv.org/abs/2503.14734)在[DreamGen benchmark](https://arxiv.org/abs/2505.12705)评估。

**Pipeline**: 生成robot执行unseen commands的video → 用latent action model或inverse-dynamics model提取pseudo-action → 得到(vision, language, action)标注的VLA training data。

**结果**（Tab. 18） - Instruction Following Scores:
- Object GPT: 91.8 (vs Hunyuan 38.0, CogVideoX 72.0, WAN2.1 72.0)
- Behavior GPT: 70.2
- Env GPT: 69.0

### 7.5 Action-Conditioned World Generation

**Cosmos-Predict2.5-2B/robot/action-cond**: input是single conditional image + robot action sequence → output是follow action的future frames chunk。

**Action conditioning设计**: 
- Action embedder MLP maps each action to tensor
- 加到timestamp embeddings of DiT modules

**Bridge dataset**: 7D action vector $(\Delta x, \Delta y, \Delta z, \Delta\theta_r, \Delta\theta_p, \Delta\theta_y, \text{GripperWidth})$

**Ablation**（Tab. 20）:
- TimeEmbedding: PSNR 24.95, FVD 146 ← **proposed**
- CrossAtten: PSNR 24.41, FVD 159
- ChannelConcat: PSNR 23.11, FVD 267

TimeEmbedding方式最好，验证了action信息通过timestep modulation注入的有效性。

---

## 8. Benchmark结果

### 8.1 PAI-Bench ([Zhou et al., 2025](https://arxiv.org/abs/2512.01989))

**Text2World**（Tab. 10）:
- Cosmos-Predict2.5-2B post-train: 0.768
- Cosmos-Predict2.5-14B post-train: 0.768  
- Wan2.2-27B-A14B: 0.769

**Image2World**（Tab. 11）:
- Cosmos-Predict2.5-2B post-train: **0.810** ← best
- Cosmos-Predict2.5-14B post-train: **0.810** ← best
- Wan2.2-27B-A14B: 0.806

### 8.2 Human Evaluation

**2B comparison**（Fig. 6）:
- vs Wan2.2 5B (60% smaller): 30.0% vs 26.2%
- vs Wan2.1 14B (85.7% smaller): 33.0% vs 34.8% (comparable)

**14B comparison**（Fig. 7）:
- vs Wan2.1 14B: 48.6% vs 31.8%
- vs Wan2.2 27B-A14B (half params): 38.1% vs 35.9%

**Key insight**: 14B比2B的benefit在human evaluation中更明显（vs Wan2.1 14B从33%到48.6%）。

---

## 9. 核心Insights总结

1. **Flow Matching > EDM for video**: velocity-based formulation提供更直接training target，smoother optimization

2. **Shifted logit-normal distribution**: high-resolution content的redundancy问题需要bias toward high noise levels，progressive shift schedule（$\beta$从1到5）配合5% explicit high-noise sampling解决transition artifacts

3. **去掉absolute positional embedding**: 借鉴long-context LLM，获得resolution/length generalization flexibility

4. **Cosmos-Reason1作为text encoder**: 多layer activation concatenation比single layer更rich，specialized for Physical AI

5. **Domain-specific SFT + Model Soup**: 比joint training更effective，simple averaging就能得到best of all worlds

6. **RL post-training with regularization**: GRPO-style + diffusion loss regularization有效avoid reward hacking

7. **Frame-replacement strategy**: 简单但effective的temporal consistency增强方法

8. **Action conditioning via time embeddings**: 比cross-attention和channel concat都好，验证了modulation-based conditioning的优势

9. **World scenario map for driving**: HD map + color-coded/shaded cuboids提供比纯spatial control更rich的driving场景control

10. **Long video RNDS metric**: Normalized relative Dover score量化error accumulation，Transfer2.5表现远优于Transfer1

## 10. 我的思考

这篇paper体现了NVIDIA在Physical AI领域的系统化工程能力。几个值得关注的direction：

1. **Sim2Real的video foundation model路径**: 直接在pixel space建模world dynamics，相比latent world model（如V-JEPA 2）保留更多high-fidelity信息，但computational cost更高

2. **Unified model的trade-off**: 一个model支持Text2World/Image2World/Video2World确实方便，但每种模式的optimal architecture可能不同

3. **RL for diffusion models**: GRPO-style的post-training在video generation上有效，但trajectory probability的decomposition（每两步accumulate gradient）是工程妥协，理论上可能不是optimal

4. **Multi-view的temporal dimension reuse trick**: 把views concatenate到temporal dimension是memory-efficient的hack，但可能限制view数量和temporal length的trade-off

5. **Open source的意义**: 在Sora/Kling/Veo等closed-source主导的领域，NVIDIA open source 2B/14B models对academic research和Physical AI adoption有重要价值

Paper的GitHub: https://github.com/nvidia-cosmos/cosmos-predict2.5

希望能build你的intuition！如果想深入某个具体aspect（比如flow matching的数学derivation、RL post-training的gradient computation细节、或multi-view的RoPE设计），我可以进一步展开。
