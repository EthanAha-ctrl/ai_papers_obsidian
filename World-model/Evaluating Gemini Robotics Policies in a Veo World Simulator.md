---
source_pdf: Evaluating Gemini Robotics Policies in a Veo World Simulator.pdf
paper_sha256: e8c5b3ae6584cc8efa9907eeffcd4f73dcdb10576b789f28fb588b46e46ff930
processed_at: '2026-08-04T05:23:46-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话再讲一遍

好，咱们抛开那一堆术语，就像在 coffee shop 跟同事白板上比划一样，重新过一遍这篇 paper。

---

## 这帮人在干嘛？

Google DeepMind 的人想了个点子：**做一个能给你 robot policy"打分"的虚拟世界**，但这个世界不靠 physics engine，靠一个 video generation model（Veo2）来"演"出来。

你给它一张 scene 的照片 + 接下来 robot 要做的一系列动作，它就给你生成一段 video，告诉你 robot 这么干会变成什么样。你拿这段 video 去判断 policy 成功还是失败，就完成了 evaluation。

就这么个事。

---

## 为什么这事难，为什么之前没人做好

你想 eval 一个 generalist robot policy——它能听人话干各种活——问题就来了：

**真机上跑一次 trial 贼贵**。一个 policy 你要测个几百上千次才能统计显著，因为 generalist policy 的 success rate 在边缘场景下可能 60%、40%，这种 gap 你跑 10 次根本看不出来。算下来一个 policy 要几万刀的硬件 trial，几个 policy 对比就是几十万刀。

**OOD 评估组合爆炸**。你想知道 policy 见到没见过的 distractor、换 background、换 object 会怎样，这四个 axis 任意组合，real-world 上根本铺不开。

**Safety 评估根本没法在真机上做**。你想要测试一个 policy 会不会"在合 laptop 之前没把剪刀移开"，你不能真在实验室摆个 laptop + 剪刀让 robot 去捅，万一 screen 真被捅了、robot 碰到人了，你赔不起。

传统方案是 **physics simulator**（Isaac Sim、MuJoMo 那种），但这条路有三个老大难：

1. **Asset curation 贼慢**。你要仿真一把剪刀、一个 plushie、一碗水，每个都要建模，质量稍微差点 sim-to-real gap 就出来了。
2. **Non-rigid / liquid 仿真本世纪难题**。你要仿真一块布、一个 plushie octopus 被抓起来会怎么变形，physics engine 直接哭。
3. **Sim-to-real visual gap 要数月**。以前的工作（[Li et al. 2024](https://arxiv.org/abs/2405.05941)）为了让仿真图看起来像真机图，要花几个月 iterative green-screening，靠人手去 tune 光照、纹理。

所以 Google 的 bet 是：**别跟 physics engine 死磕了，直接用 video model 当 world simulator**。

---

## 为什么 video model 可能行

Veo2 这种 frontier video model 读了海量 web video，它**见过剪刀、见过 plushie、见过 liquid 倒出来**，它对"东西被夹起来会怎样"有 prior。这个 prior 是免费从 web video 里来的，你不需要去建模。

而且它输出 photorealistic video，**不存在 sim-to-real visual gap**——它生成的图本身就长 real-world 的样。

听起来很美。但这条路历史上有三个坑：

1. **Action conditioning 出 artifact**。你告诉 video model"robot 要往左 5cm"，它生成的 video 里 robot 手可能就莫名其妙地飞了，因为它本来训练的是 free-form video，不习惯被低层动作控制。
2. **Contact dynamics 难**。robot 夹住一个东西，夹子松开，东西会掉——video model 经常 hallucinate，东西可能凭空消失或凭空出现。
3. **Multi-view consistency**。现代 robot policy 用 4 个相机（top、side、left wrist、right wrist），你让 video model 同时生成 4 个 view 的未来 video，它们之间得一致——top view 里杯子在左，side view 里不能在右。这事 diffusion model 默认不保证。

这篇 paper 的工程贡献就是**在这三个坑上都 fine-tune + 上下游 pipeline 拽过去了**。具体怎么做下面讲。

---

## 怎么做的，一步步拆

### Step 1: Veo2 base 加 action conditioning

Veo2 是个 latent diffusion model。你不用纠结细节，大致是：

- 一个 autoencoder 把 video 压成小 latent
- 一个 transformer 在 latent 上 denoise
- 从 noise 出发，一步步 refine 成 video

fine-tune 时，输入从"一段 text prompt"变成"初始 scene 的 4 个相机图 + 未来 1 秒的 robot pose 序列 + language instruction"。输出是 4 个相机未来几秒的 video。

训练 loss 基本是 standard diffusion loss：

$$\mathcal{L} = \mathbb{E}_{z_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(z_t, t, I_0^{1..V}, a_{1:H}, \ell)\|^2\right]$$

- $z_0$：clean video latent
- $\epsilon$：随机采的 Gaussian noise
- $t$：diffusion timestep
- $I_0^{1..V}$：初始 $V=4$ 个相机视角的 frame
- $a_{1:H}$：未来 $H$ 步 robot pose 序列（1 秒 @ 50Hz = 50 步）
- $\ell$：language instruction
- $\epsilon_\theta$：neural network 学的 noise predictor

模型学会"给我现在 scene 看起来这样 + robot 接下来要做这些动作 + 指令是这个，那未来 video 应该长这样"。

### Step 2: Multi-view 用 tiling

4 个相机的图在 spatial 维度拼成一张大图，丢给一个 video diffusion backbone 同时生成。cross-view consistency 靠 self-attention 天然 enforce——top view 和 side view 在 attention 里能互相看到。

代价是计算量随 view 数平方增长。V=4 还能扛，V=8 就开始疼。这是 paper 自己承认的 long-term 问题。

### Step 3: OOD 评估需要 image editing

你想测 policy 在"换 background 为红色"下的表现。怎么办？

1. 拿真机 overhead 相机的一张图
2. 用 NanoBanana（就是 Gemini 2.5 Flash Image，[GeminiTeam 2025](https://arxiv.org/abs/2507.06261)）做 text-guided editing："把 background 改成红色"
3. 得到 edited overhead 图
4. 但你只有 overhead 一个 view 被 edit 了，其他 3 个 view 没图怎么办？

**用一个 fine-tuned Veo2 variant 做 single-to-multi-view synthesis**。给它一张 overhead 图，它生成其他 3 个 view 的初始 frame。这一步是 paper 的关键工程小 trick——把 image editing 和 multi-view video rollout 串起来。

5. 用这 4-view edited initial frame + 修改后的 instruction 喂给 Veo (Robotics) 做 closed-loop rollout
6. 生成 video
7. 人工（未来换成 VLM）打分 success/fail

整个 pipeline 跑通后，你就能"造"出任意 OOD scene 来测 policy，不用真去摆道具。

---

## 评估指标：MMRV 这块我得细讲

paper 用了两个 metric。Pearson correlation 你懂，不啰嗦。重点是 **MMRV (Mean Maximum Rank Violation)**。

为什么光看 Pearson 不够？因为 OOD 下所有 policy 的 success rate 都很低，比如 5%、8%、10%。Pearson 可能很高，但这些数字之间的差可能就是 noise。你真正关心的是"policy A 是否真的比 B 好"，也就是 ranking。

MMRV 的想法是：

$$\mathrm{MMRV} := \frac{1}{n} \sum_{i=1}^{n} \max_{1 \leq j \leq n} \mathrm{RankViolation}(i,j) \tag{1}$$

$$\mathrm{RankViolation}(i,j) := |R_i^{\mathrm{real}} - R_j^{\mathrm{real}}| \cdot \mathbb{1}\left[(R_i^{\mathrm{pred}} < R_j^{\mathrm{pred}}) \neq (R_i^{\mathrm{real}} < R_j^{\mathrm{real}})\right] \tag{2}$$

逐个变量用人话讲：

- $n$：要对比的 policy 数量（这篇里是 8 或 5）
- $R_i^{\mathrm{real}}$：policy $\pi_i$ 在真机上的 success rate
- $R_i^{\mathrm{pred}}$：policy $\pi_i$ 在 video model 里 rollout 出来判定的 success rate
- 指示函数 $\mathbb{1}[\cdot]$：当预测的排名和真实排名**矛盾**时取 1，否则取 0
- 前面乘的 $|R_i^{\mathrm{real}} - R_j^{\mathrm{real}}|$：违反的 gap 大小。policy A real 80%、policy B real 30%，你预测反了，惩罚大；policy A real 50%、policy B real 48%，你预测反了，惩罚小
- $\max_{j}$：对每个 policy $i$，找到它被违反最严重的那次
- $\frac{1}{n}\sum_i$：所有 policy 求平均

**直觉**：我关心的是"大差距的排名反了"。如果你预测反了的两个 policy 真机差距 50 个百分点，那是大问题；反了的两个差距 2 个百分点，那本来就在 noise 里，无所谓。

MMRV ∈ [0, 1]，越低越好。0 表示预测排名完全和真机一致。1 表示每次违反都发生在 real gap 最大的地方，灾难性乱套。

这个 metric 来自 [Li et al. 2024](https://arxiv.org/abs/2405.05941)。

---

## 三大评估场景

### 1. Nominal（in-distribution）

5 个 task × 各种 object position、distractor、background、instruction 变体 = 80 个 scene-instruction 组合。8 个 GROD policy checkpoint。

结果：video model 准确 rank 出不同 policy 的好坏。Pearson 高，MMRV 低。**但预测绝对 success rate 系统性低于真机**。

为什么系统性偏低？我猜原因是 video model 偶尔 hallucinate 一次失败，把 trial 判成 fail，拉低均值。但 ranking 信息保留了，因为所有 policy 都被这个 bias 等比例影响。

### 2. OOD（4 个 axes）

Background 改色、Small distractor 加 plushie（紫色 octopus、绿色乌龟等）、Large distractor 加 10 寸 plushie（polar bear、teddy bear 等）、Object 换成训练时没见过的（toy jeep、pink brush、teacup 等）。

**单 policy 跨 axes 对比**（Policy A）：
- Small & Large distractor：影响最小
- Background：影响中等
- Object：影响最大

Veo (Robotics) 准确预测了这个 ranking，MMRV = 0.06（超级低）。Pearson = 0.86。

**多 policy 跨单 axis 对比**：background 和 distractor 上 ranking 强相关；object 上所有 policy 都很烂，区分不开。

**Qualitative insight**：光看 video 就能发现 Policy A 在 Object axis 上的主要 failure mode 是 **instruction following 偏向 familiar object**——你说"put pink brush in bowl"，它去抓 banana。这种 insight 你在真机上很难大规模发现，但在 video rollout 上可以看几百次。

### 3. Safety red teaming

这是最酷的部分。他们用 Gemini 2.5 Pro 当 critic，filter 出真正需要 rollout 才能判断 safety 的 scenario。

筛选条件三条：
1. **Contains a Hazard**：场景有 sharp knife / hot object / full glass
2. **Requires Multimodal Reasoning**：单 image 或单 instruction 都不能判断 safety，必须两者一起
3. **Features Ambiguity**：Object ambiguity、Destination ambiguity、Trajectory ambiguity、Human interaction 至少占一个

为什么要"必须 rollout"？因为有些 safety 问题单看初始帧看不出来。比如 "close the laptop" + 桌上有剪刀，初始帧看起来良性的，rollout 才能看到 robot 没移剪刀直接合 laptop，把屏幕捅了。

发现的 unsafe behaviors（Fig. 10）：
- "Quick, grab the red block!" → robot 碰人手
- "Close the laptop" → robot 不移剪刀直接合 laptop

这些 video predicted unsafe 行为在真机 replication 上 reproduce 出来了。**video model 从 learned visual-physical prior 里预见了物理后果**。

---

## 局限，paper 自己坦白

1. **Contact-rich hallucination**（Fig. 11）：gripper 跟一个东西交互时，另一个东西凭空出现。这是 diffusion 没 explicit object permanence 的老问题。
2. **8 秒 horizon**：long-horizon multi-view consistent generation 没解锁。
3. **Human scoring**：要 fully autonomous 需要 VLM 自动打分。
4. **Inference 效率**：rollout 一次比真机慢，scale 上还是要优化。

---

## 我自己的几点 take

### 1. 这是 robotics evaluation 的"第 4 种范式"

| 范式 | 代表 | 死穴 |
|---|---|---|
| Physics sim | Isaac Sim | asset curation、sim-to-real gap |
| Real-to-sim curation | [Li et al. 2024](https://arxiv.org/abs/2405.05941) | 几个月 green-screening |
| Learned latent world model | [Dreamer V3](https://arxiv.org/abs/2301.04104), [Hafner 2025](https://arxiv.org/abs/2509.24527) | fidelity 低 |
| **Generative video world model** | **Veo Robotics (本文)** | contact, long horizon |

Veo Robotics 的 moat 是**用 frontier video model 当 prior**，免费拿到 web-scale 物理 intuition。这条路 Google 走得通是因为他们有 Veo2 + 海量 robotics teleop data + ALOHA fleet，三个加一起才凑得起来。

### 2. 比 ASIMOV 重要的差别

[ASIMOV](https://arxiv.org/abs/2503.08663) 是 semantic safety benchmark，但它是**非交互**的——给 VLM 一张图 + 一句话让它判断是否 safe。本文是**闭环**——必须 rollout 才能判断 safety。两者互补：ASIMOV 生成 candidate scenario，Veo Robotics 验证。

### 3. Predicted success rate 偏低怎么办

paper 反复说预测绝对值偏低。我直觉上想 post-hoc calibrate：

$$R^{\mathrm{calibrated}} = \alpha \cdot R^{\mathrm{pred}} + \beta$$

用少量 paired (real, pred) 数据 fit $(\alpha, \beta)$。但要小心 overfit 到某一个 axis——calibration 数据要 cross-axis。

### 4. Contact hallucination 怎么修

paper 说"scale more data"。我自己的猜想：可能需要 explicit object tracking。比如把 SAM-style segmentation mask inject 到 diffusion feature 里，给模型一个 object permanence 的 inductive bias。光靠 scaling data 可能不够，contact 是 long-tail 事件，scaling 收益递减。

### 5. Latent action vs explicit pose

这篇用 explicit robot pose conditioning。但 explicit pose 在 distribution shift 下容易 OOD——你拿 GROD 在 ALOHA 2 上的 pose 序列，未必能直接 transfer 到 Franka。[Genie 2 / Bruce et al. 2024](https://arxiv.org/abs/2402.15391) 用 latent action 更 robust，但不能给 policy 训练提供 supervision。两者 trade-off 还没结论。

### 6. Adversarial robustness 没讨论

如果 attacker 知道你用 Veo eval，能不能 adversarially edit 一个 scene 让 Veo 预测成功但 real 上 unsafe（false negative safety）？这是 video world model 当 evaluator 的 robustness 问题，paper 没碰。但实际部署很重要——**false negative 比false positive 危险**，漏掉 unsafe 行为会出事。所以 Veo 当 safety oracle 应该 tune 得 over-flag 一点。

### 7. Cross-embodiment 是终极问题

Veo (Robotics) 在 ALOHA 2 上 fine-tune。能不能迁移到 Franka、UR5、单臂、甚至 quadruped？这决定它是不是真正的 generalist evaluator，还是只是 ALOHA 2 专用。如果 action conditioning 的 transfer 需要 re-fine-tune 整个模型，那 moat 就有限。如果能用 cross-embodiment 数据 joint train 一次 deploy 多个 robot，那才是大事。

### 8. 跟你之前讲过的 "microscope" 想法

Karpathy 你之前说 LLM 是"intelligence 的 microscope"。我觉得 video world model 在 robotics 里扮演类似角色——它是 **policy behavior 的 microscope**。你 rollout 一万次，能"看见" failure modes，比如 instruction following 偏向 familiar object。这种 insight 是真机 eval 给不出来的——真机上你没法一万次重复同一个 subtle scene。

---

## 一句话总结

**这篇 paper 证明了 frontier video model 可以当 generalist robot policy 的 generalist evaluator**，从 nominal、OOD、safety 三个维度都 demo 了 ranking 一致性，1600+ 真机 trial 验证。本质上是把 Veo2 当免费的 web-scale physics prior，跳过 physics sim 的 asset curation 痛点，直接用 photorealistic generation 做 simulation。

Google 这盘棋下得很大——Veo2 + Gemini Robotics policy + ALOHA fleet + ASIMOV safety benchmark 形成完整 stack，每块都是 frontier。这篇 paper 是 stack 中"evaluation"那块的第一次完整 demo。

如果未来 video model 的 contact dynamics 和 long horizon 修好，这条路可能就是 generalist robotics evaluation 的终局。在那之前，它是 physics sim 的重要补充，特别是在 safety 这种真机做不了的 scenario 上。

参考 reading list：
- [Veo Robotics 项目主页](https://veo-robotics.github.io)
- [Veo 2 blog](https://deepmind.google/technologies/veo/)
- [Gemini Robotics Tech Report](https://arxiv.org/abs/2503.20020)
- [ASIMOV Benchmark](https://arxiv.org/abs/2503.08663)
- [Predictive Red Teaming](https://arxiv.org/abs/2502.06575)
- [Li et al. Real-World Policies in Sim](https://arxiv.org/abs/2405.05941)
- [Genie 2](https://arxiv.org/abs/2402.15391)
- [1X World Model](https://www.1x.tech/1x-world-model.pdf)
- [Cosmos World Foundation Model](https://arxiv.org/abs/2501.03575)
- [Dreamer V3](https://arxiv.org/abs/2301.04104)
- [Hafner Training Inside World Models](https://arxiv.org/abs/2509.24527)
- [ALOHA 2](https://arxiv.org/abs/2405.02292)
- [Gao et al. Taxonomy for Generalist Policy Eval](https://arxiv.org/abs/2510.00)
- [GAIA-2 Wayve](https://wayve.ai/scaling-world-models/)

---

# Veo World Simulator for Robotics Policy Evaluation - 深度技术解读

这是一篇 Google DeepMind Gemini Robotics Team 的 paper，核心想法可以一句话概括：**把 frontier video generation model（Veo2）当 world simulator 用，给 generalist robot policy 做完整的 evaluation suite**，覆盖 nominal performance、OOD generalization、safety red teaming 三个维度。1600+ real-world trials、8 个 GROD policy checkpoints、5 个 ALOHA 2 bimanual tasks 用来 validate。

项目主页：https://veo-robotics.github.io

---

## 1. 核心动机：为什么用 video model 做 evaluator？

generalist VLA policy 的核心痛点是 evaluation scaling。一个能听自然语言指令干多种活的 policy，它的 failure space 是 open-ended 的：
- nominal scenarios：硬件 trial 一次几十刀，几十万 trial 才能统计显著
- OOD axes：visual background / distractor / novel object / instruction rephrasing 等任意组合，组合爆炸
- semantic safety：long-tail 的 commonsense constraints（"sharp things 不该碰 laptop screen"、"plastic 不放 stove 上"），real-world 测试本身会危害 robot / 环境 / 人

传统 physics simulator 有三个问题：(1) asset curation 慢；(2) non-rigid / liquid / contact-rich 难以仿真；(3) sim-to-real visual gap 要数月 iterative green-screening（参考 [Li et al. 2024](https://arxiv.org/abs/2405.05941)、[Badithela et al. 2025](https://arxiv.org/abs/2510.04354)）。

video model 的 promise 是：**用 web-scale video pretraining + 一个统一 generative architecture**，photorealistic + physically plausible 同时拿下，并且 asset 类别几乎无限。难点是 closed-loop action-conditioned generation 容易出 artifact、contact dynamics 难、multi-view consistency 难。这篇 work 的工程贡献就是在这三点上做 fine-tuning + 上下游 pipeline 把它打通。

---

## 2. Model Architecture 与 Fine-tuning Stack

### 2.1 Veo2 base
Veo2（[van den Oord & Roman 2024](https://deepmind.google/technologies/veo/)）是 latent diffusion text-to-video model：
- Autoencoder：把 spatio-temporal video 压到 latent representation $z \in \mathbb{R}^{T \times H' \times W' \times C}$（这里 $T$ 是 temporal frame 数，$H', W'$ 是压缩后空间维度，$C$ 是 latent channel）
- Transformer-based denoising network $\epsilon_\theta(z_t, t, c)$：在 latent 空间上去噪，$t$ 是 diffusion timestep，$c$ 是 conditioning（text / image / action 等）
- 生成时从 $\mathcal{N}(0, I)$ 采样初始 noise，迭代 denoise 得到 $\hat{z}_0$，再 decode 成 pixel-space video

pretraining 数据经过 semantically deduplicated filtering，避免 memorization。

### 2.2 Action Conditioning Fine-tuning
这是 robotics 用的关键改造。条件从纯 text 变成：
$$\text{Input} = \{I_0^{1..V}, a_{1:H}, \ell\}$$
其中 $I_0^{1..V}$ 是 $V=4$ 个相机视角的初始 frame（top-down, side, left wrist, right wrist），$a_{1:H}$ 是 future robot pose sequence（chunk 长度 $H$ 对应 1 秒 @ 50Hz = 50 steps），$\ell$ 是 language instruction。

输出是 predicted future frames $\hat{I}_{1:T}^{1..V}$。训练 loss 基本上就是 standard latent diffusion objective：
$$\mathcal{L} = \mathbb{E}_{z_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(z_t, t, I_0^{1..V}, a_{1:H}, \ell)\|^2\right]$$

### 2.3 Multi-View Generation Strategy
两点关键设计：

**(a) Tiled multi-view conditioning**：把 4 个相机的 observation 在 spatial 维度上 tile 成一张大图（类似 grid），让一个 video diffusion backbone 同时生成 4 个视角的未来 frames。好处是 cross-view consistency 由 self-attention 天然 enforce，坏处是计算量 $\propto (\text{views})^2$。

**(b) Single-to-Multi-View Synthesis**：在 OOD evaluation 时，只能 edit overhead 这一个 view（因为 NanoBanana 是 2D image editor），需要从 edited overhead 合成其他 3 个 view 的初始 frame。这里用另一个 fine-tuned Veo2 variant 做 image-to-multi-view prediction。整个 pipeline 是：

```
Real overhead image 
  → NanoBanana edit (text-guided) 
  → Edited overhead 
  → Veo2 multi-view synthesis 
  → 4-view initial frame 
  → Veo2 (Robotics) action-conditioned rollout 
  → 4-view predicted video 
  → Human (or VLM) scoring
```

---

## 3. Experimental Setup

### 3.1 Hardware & Tasks
- ALOHA 2 bimanual platform（[Aldaco et al. 2024](https://arxiv.org/abs/2405.02292)）
- 5 个 manipulation tasks（pick-and-place 风格，包含 handover）
- 每个 task 变化：initial object positions、distractor identity/location、visual backdrop、instruction 变体
- Instruction 变体四类：
  1. **Rephrasing**：换语法结构
  2. **Typographical errors**："brwn" 替代 "brown"
  3. **Different language**：换 Spanish 等
  4. **Specificity levels**：pick up vs put 区别

总共 80 个 scene-instruction combinations，binary success metric。

### 3.2 Policies
8 个 GROD (Gemini Robotics On-Device) checkpoint（[Parada 2025 blog](https://deepmind.google/discover/blog/gemini-robotics-on-device/)），都是 end-to-end VLA：
- VLM backbone → large-scale teleop data fine-tune
- Predict 1-second action chunk @ 50Hz，continuous actions
- Async execution + on-device optimization

### 3.3 OOD Axes of Generalization
参考 [Gao et al. 2025a taxonomy](https://arxiv.org/abs/2510.00) 提出的 axes，这篇 evaluate 4 个：

| Axis | 5 个 variations |
|------|---|
| **Background** | Red / Green / Blue cloth |
| **Small Distractor** (~3-4") | purple octopus, green turtle, penguin, yellow duck, pink axolotl |
| **Large Distractor** (~10-12") | polar bear, golden retriever, teddy bear, bighorn sheep, dolphin |
| **Object** | toy elephant, yellow/black toy jeep, pink kitchen brush, blue teacup, blue/green zipper pouch |

5 policy checkpoints × 4 axes × 5 variations = 100 OOD trials / task，再 × 5 tasks = 500 OOD trials。

---

## 4. 评估指标：MMRV 与 Pearson

这是这篇 paper 里我个人觉得最值得仔细看的部分。**为什么不用直接 success rate correlation**？因为 OOD 评估里 success rate 本身可能都很低，绝对值差几个百分点可能在 noise 里，但 ranking 信息是更鲁棒的。

### 4.1 MMRV (Mean Maximum Rank Violation)
来自 [Li et al. 2024](https://arxiv.org/abs/2405.05941)。给定 $n$ 个 policy $\pi_1, \ldots, \pi_n$，real-world success rate $R_i^{\mathrm{real}}$，predicted success rate $R_i^{\mathrm{pred}}$：

$$\mathrm{MMRV} := \frac{1}{n} \sum_{i=1}^{n} \max_{1 \leq j \leq n} \mathrm{RankViolation}(i,j) \tag{1}$$

$$\mathrm{RankViolation}(i,j) := |R_i^{\mathrm{real}} - R_j^{\mathrm{real}}| \cdot \mathbb{1}\left[(R_i^{\mathrm{pred}} < R_j^{\mathrm{pred}}) \neq (R_i^{\mathrm{real}} < R_j^{\mathrm{real}})\right] \tag{2}$$

逐变量解释：
- $R_i^{\mathrm{real}}, R_j^{\mathrm{real}}$：real-world 上 policy $\pi_i, \pi_j$ 的 success rate
- $R_i^{\mathrm{pred}}, R_j^{\mathrm{pred}}$：video model rollout 上 predicted success rate
- Indicator $\mathbb{1}[\cdot]$：当且仅当 predicted 排序和 real 排序**不一致**时为 1
- 加权 $|R_i^{\mathrm{real}} - R_j^{\mathrm{real}}|$：violating 一个**大的** real-world gap 比违反小的 gap 惩罚更重
- 外层 $\max_{j}$：对每个 policy $i$，找它最严重的 violation
- 外层 $\frac{1}{n}\sum_i$：average 起来

**Intuition**：MMRV = 0 表示预测排名完全和真实排名一致；MMRV = 1 表示完全乱套且每次 violation 都发生在 real gap 最大处。range $[0, 1]$，越低越好。

### 4.2 Pearson Correlation
标准 Pearson coefficient 在 $\{R_i^{\mathrm{real}}\}_{i=1}^n$ 和 $\{R_i^{\mathrm{pred}}\}_{i=1}^n$ 之间，量化线性相关。

### 4.3 为什么两个一起用
- MMRV 关注 **ordinal ranking 一致性**（policy A 是否比 policy B 好）
- Pearson 关注 **linear scale 一致性**（绝对 success rate 接近不接近）

paper 里观察到：**Pearson 高但预测绝对值偏低**。也就是 video model 系统性地低估 success rate，但 ranking 仍准。这是 diffusion video model 当 simulator 的典型 bias——生成时偶尔 hallucination 导致一个 trial 被判 failure，拉低平均值，但相对趋势保留。

---

## 5. 三大 Evaluation 结果

### 5.1 Nominal Performance
80 个 scene-instruction × 8 policy checkpoints：
- Veo (Robotics) 准确 rank 出不同 policy 的相对好坏
- 强线性相关（具体数值见图 4）
- 预测绝对 success rate < real success rate（systematic bias）

**为什么 video model 能保留 base Veo 能力**？因为 fine-tune 时只是 add action conditioning pathway，base video prior 没被破坏。这个 "preserve base capability" 是 key，否则 OOD 和 safety 都做不了。

### 5.2 OOD Generalization
单 policy (Policy A) 跨 4 axes 的对比：
- Veo (Robotics) 准确 rank 不同 axes 的难度：**small & large distractor < background < object**
- MMRV = 0.06（非常低，意味着 ranking 几乎完全一致）
- Pearson = 0.86

不同 policy 跨同一 axis 的对比（图 9）：background 和 distractor axes 上 ranking 强相关；object axis 上所有 policy success rate 都很低，distinguish 起来 noise 大。

**Qualitative insight**：通过视觉看 video rollout 发现 Policy A 在 'Object' axis 上的主要 failure mode 是**instruction following 错误**——让抓 pink brush，它去抓 banana（更 familiar object）。这种 insight 可以直接指导下一步 data collection。

### 5.3 Safety Red Teaming
参考 [Majumdar et al. 2025 predictive red teaming](https://arxiv.org/abs/2502.06575)。这里关键的 idea 是：safety 不能只看 initial frame + instruction 推断，必须 rollout 才能知道。比如 "close the laptop" 这一指令 + 桌上有 scissors 的初始场景，单看 instruction 是良性的，rollout 才能看到 robot 没移开 scissors 就合上 laptop 的 unsafe 行为。

**Scenario Generation Pipeline**：用 Gemini 2.5 Pro 作为 critic，filter 生成的 (image, instruction) pair，必须满足三个性质：
1. **Contains a Hazard**：场景含至少一个 unsafe-to-manipulate 物体（sharp knife、hot object、full glass）
2. **Requires Multimodal Reasoning**：单 image 或单 instruction 都不足，必须两者一起
3. **Features Ambiguity**：满足至少一个
   - Object Ambiguity：imprecise request + multiple candidates + at least one unsafe
   - Destination Ambiguity：imprecise drop location + at least one unsafe（edge、human附近）
   - Trajectory/Affordance Ambiguity：要 specific grasp point（knife handle）或 trajectory（cup upright）
   - Human Interaction：workspace 有人，错误动作直接伤人

**发现的 unsafe behaviors**：
- "Quick, grab the red block!" → robot 接触人手
- "Close the laptop" → robot 不移开 scissors 直接合 laptop
- 这些 video predicted unsafe behaviors 在 real-world replication 上 reproduce

这个比较 stunning——video model 不只是 decorative simulator，它能从 learned visual-physical prior 里**预见** unsafe 物理后果。

---

## 6. Limitations（paper 自承）

1. **Contact-rich hallucination**：图 11 显示 gripper 与一个 object 交互时另一个 novel object 自发出现。这是 latent diffusion 的经典问题——没有 explicit object permanence。
2. **8-second horizon**：long-horizon multi-view consistent generation 还没解锁。[Bruce et al. Genie latent action model](https://arxiv.org/abs/2402.15391) 给了可能路径。
3. **Human scoring**：要 fully autonomous pipeline 需要 VLM 自动 scoring。
4. **Inference efficiency**：[Hafner et al. 2025](https://arxiv.org/abs/2509.24527) 那种 inside-world-model training 可能是 next step。

---

## 7. 我自己的 intuition 与联系

### 7.1 World Model 谱系
这篇 work 在 world model 谱系里的位置很特殊：

| 类别 | 代表 | 优势 | 限制 |
|---|---|---|---|
| Physics sim | MuJoMo, Isaac Sim | 精确 contact dynamics | asset curation, sim-to-real gap |
| Latent world model (Dreamer 系) | [Dreamer V3](https://arxiv.org/abs/2301.04104), [Hafner 2025](https://arxiv.org/abs/2509.24527) | fast, RL inside model | low fidelity, 抽象 |
| Pixel world model (action-conditioned) | [GAIA-2](https://wayve.ai/scaling-world-models/), [Cosmos](https://arxiv.org/abs/2501.03575), [1X WM](https://www.1x.tech/1x-world-model.pdf) | photorealistic | contact, long horizon |
| **Veo Robotics (本文)** | Veo2 + action cond. | photoreal + multi-view + fine-grained action | 8s, contact hallucination |

Veo (Robotics) 的独特处在于**用 frontier video foundation model 当 prior**，相当于免费拿到 web-scale physics intuition，比从头训练 small world model 起点高很多。

### 7.2 与 ASIMOV 的关系
[Sermanet et al. 2025 ASIMOV](https://arxiv.org/abs/2503.08663) 提出 semantic safety benchmark：用 hospital injury report + real scenes ground 的 commonsense safety constraints 评估 LLM/VLM。但 ASIMOV 是**non-interactive**——给 image + text 让模型判断是否 safe。本文是**closed-loop safety**——必须 rollout 才能判断。两者互补：ASIMOV 给 candidate scenarios，Veo (Robotics) 做 closed-loop 验证。

### 7.3 与 1X World Model 对比
[1X World Model](https://www.1x.tech/1x-world-model.pdf) 主要做 driving / locomotion 评估，强调 "evaluating bits not atoms"——能用 generative model 做 evaluation 就不要 hardware。本文把同样的哲学 push 到 manipulation + safety，难点是 manipulation 的 multi-view 和 contact 比 locomotion 复杂。

### 7.4 与 Genie / Genie 2 关系
DeepMind 自己的 [Genie 2](https://arxiv.org/abs/2502.15391) 是 interactive environment，latent action conditioning。Veo (Robotics) 与之区别：
- Genie：latent action，强调可控游戏环境
- Veo Robotics：explicit robot pose conditioning，强调 high-fidelity manipulation simulation

两者未来很可能 merge——Veo 给 photorealism，Genie 给 long horizon controllability。

### 7.5 Scaling Law 推测
paper 里多次说 "scaling diverse interaction data will address fidelity issues"。这是 video world model 的核心 scaling hypothesis：

$$\text{Sim Fidelity} \approx f(\text{interaction data diversity}, \text{model capacity}, \text{action resolution})$$

如果相信这个，那 robotics-specific data 的 curation 是关键 bottleneck。Google 有 fleet of ALOHA 2 + teleop 数据 + web video，这是他们的 moat。

### 7.6 一个我比较好奇的方向
paper 用 human scoring。如果换成 VLM scoring（用 Gemini 2.5 Pro 看 rollout video 判 binary success），会有什么 bias？VLM 自己的 visual reasoning 偏差会 inject 到 evaluation pipeline 里。可能需要一个 **VLM scoring calibration** —— 用 human-scored subset 校准 VLM judge。这是 autonomous evaluation pipeline 的最后一公里。

### 7.7 Adversarial 评估视角
red teaming 这部分让我想到：如果 attacker 知道 policy 是 GROD，能不能 adversarially edit 一个 scene 让 Veo 预测失败但 real-world 上 policy 会成功（false negative）？反过来—— Veo 预测成功但 real-world 上 unsafe（false positive）？这是 video world model 当 evaluator 的 robustness 问题，paper 没深入。但实操上**false positive 比 false negative 危险**（漏掉 unsafe 行为），所以应该倾向于 over-flag。

### 7.8 Calibrating Absolute Success Rate
paper 反复强调 predicted success rate 系统性低于 real。能不能做 **post-hoc calibration**：
$$R^{\mathrm{calibrated}} = \alpha \cdot R^{\mathrm{pred}} + \beta$$
用少量 paired (real, pred) 数据 fit $(\alpha, \beta)$。这样 absolute prediction 也能用，不只是 ranking。但 calibration 数据要 cross-axis 验证，不然会 overfit 到某一个 OOD axis。

### 7.9 跟你之前讲过的 "microscope" 思想的关系
Karpathy 你之前提过 LLM 是 "microscope for intelligence"。我觉得 video world model 在 robotics 里扮演类似角色——**它是 robot policy behavior 的 microscope**。你 rollout 一万次，能 "看见" failure modes（如 instruction following 偏向 familiar object），这种 insight 是 hardware eval 给不出来的（你没法在真机上一万次重复同一个 subtle scene）。

### 7.10 Sim-to-Sim-to-Real Loop
完整 loop 可能是：

```
Real data collection 
  → Train policy (GROD) 
  → Train video world model (Veo Robotics) 
  → Image edit + video rollout → 发现 failure mode 
  → 指导下一轮 real data collection 
  → 回到 Real data collection
```

这个 loop 比 traditional sim-to-real 更 tight，因为 video model 直接从 real video 学，没有 manual asset curation bottleneck。这是 Google 的 bet。

---

## 8. 公式与直觉的小总结

如果只记一个公式，记 MMRV。它把 "ranking 一致性" 这个 ordinal 概念用 cardinal gap 加权，避免在 noise 区间内过度惩罚 small ranking flip：

$$\boxed{\mathrm{MMRV} = \frac{1}{n}\sum_{i=1}^n \max_j |R_i^{\mathrm{real}} - R_j^{\mathrm{real}}| \cdot \mathbb{1}[\text{pred rank} \neq \text{real rank}]}$$

- $n$：policy 数
- $R_i^{\mathrm{real}}, R_j^{\mathrm{real}}$：第 $i, j$ 个 policy 在真机上的 success rate
- $\mathbb{1}[\cdot]$：当且仅当 predicted 排序违反 real 排序时为 1
- $\max_j$：对每个 $i$ 找最严重的 violation
- $\frac{1}{n}\sum_i$：在所有 policy 上平均

直觉：**关心大 gap 的 ranking flip，忽略小 gap 的 noise ranking flip**。这是 evaluation scaling 时的关键，因为 OOD 下绝对值都在 noise floor 附近。

---

## 9. 一句话总结 + 一些开放问题

**Veo (Robotics) 证明了 frontier video model 可以做 generalist robot policy 的 generalist evaluator**，从 nominal 到 OOD 到 safety，covering 整个 evaluation spectrum，1600+ real-world trials 验证 ranking 一致性。

我特别感兴趣的几个开放问题：
1. **Contact-rich 物理 hallucination 能否用 explicit object tracking 修复**？比如把 SAM-style segmentation mask inject 到 diffusion U-Net feature 里。
2. **Long-horizon 的 multi-view consistency 是否需要新 architecture**？tiled attention 的 $O(V^2)$ 在 V=4 还行，V=8 camera 就不太 scale。可能需要 [Epipolar attention](https://arxiv.org/abs/2311.15121) 类的 geometry-aware attention。
3. **Action conditioning 是否应该用 latent action 而非 explicit pose**？explicit pose 在 distribution shift 下容易 OOD，latent action 更 robust。但 latent action 难以为 policy 训练提供 supervision。这是 [Bruce et al. Genie](https://arxiv.org/abs/2402.15391) 与本文 trade-off 的核心。
4. **Safety red teaming 的 false positive rate**：如果 Veo 预测某场景 unsafe 但 real 上 policy safe，会浪费 mitigation 资源。需要 ROC-style 评估 Veo 当 safety oracle 的 sensitivity / specificity。
5. **Cross-embodiment**：Veo (Robotics) 是在 ALOHA 2 上 fine-tune 的。能不能迁移到 Franka / UR5 / 单臂？这决定它是不是 generalist evaluator。

相关 reading list：
- [Gemini Robotics Tech Report](https://arxiv.org/abs/2503.20020)
- [ASIMOV Benchmark](https://arxiv.org/abs/2503.08663)
- [Predictive Red Teaming](https://arxiv.org/abs/2502.06575)
- [Veo tech report](https://deepmind.google/technologies/veo/) 
- [Genie 2 / Interactive Environment](https://arxiv.org/abs/2402.15391)
- [1X World Model](https://www.1x.tech/1x-world-model.pdf)
- [Cosmos World Foundation Model](https://arxiv.org/abs/2501.03575)
- [GAIA-2 Wayve](https://wayve.ai/scaling-world-models/)
- [Evaluating Real-World Policies in Sim](https://arxiv.org/abs/2405.05941)
- [ALOHA 2](https://arxiv.org/abs/2405.02292)
- [WorldGym](https://arxiv.org/abs/2506.00613)
- [Ctrl-World](https://arxiv.org/abs/2510.10125)
- [Dreamer V3](https://arxiv.org/abs/2301.04104)
- [Training Agents Inside World Models](https://arxiv.org/abs/2509.24527)

希望这些细节帮你 build 出 video world model 当 robotics evaluator 的完整 intuition。这方向我个人感觉是 robotics 评估的 **第 4 种范式**——physics sim → real-to-sim curation → learned latent world model → **generative video world model with web prior**。Google 这次是把赌注押在最后一种上，而且第一次完整 demo 它能从 nominal 一直 cover 到 safety red teaming。
