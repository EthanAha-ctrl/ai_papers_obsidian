---
source_pdf: Current World Models Lack a Persistent State Core.pdf
paper_sha256: 0462d55767de2471336e74116c27c2551429a1d9509c0942666676842f18b395
processed_at: '2026-08-18T04:20:20-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：这篇 paper 到底在说什么

## 一句话版本

现在的 video generation model 说自己是 world model，但其实它们只是 **会画画的相机**，不是 **会演化的世界**。你转开相机的瞬间，世界就冻住了。

---

## 用一个画面讲清楚

想象你在用 Sora 或者类似的 model 生成一段视频。prompt 是："卧室里，一只猫从地上跳上床。"

然后你让相机往左转一下，看一会儿墙，再转回来看床。

**应该发生什么**：猫已经在床上了，因为它在你没看的时候跳上去了。月亮不需要你看着才转。

**实际发生什么**（这是 paper 测了 23 个 model、9600 段视频后的结论）：

- 猫还在地上（世界冻住了）
- 猫消失了（object 被删了）
- 猫出现在奇怪的地方（hallucinate）
- 猫变成两只（duplicate）
- 猫根本没离开画面，相机被猫"绑架"了

这就是 Einstein 那句话的 deep 含义："I like to think the moon is there even if I am not looking at it." 一个真正的 world model，世界的 state 应该在你不看的时候继续演化。

---

## 为什么这个问题以前没人测？

因为以前的 benchmark 都在测 **你看得见的时候** 画得像不像：

- VBench (https://vchitect.github.io/VBench-project/) 测 quality、motion、text alignment
- WorldScore (https://worldscore.github.io/) 测 camera controllability
- VideoPhy (https://sites.google.com/view/videophy/) 测 physical commonsense

**没有一个 benchmark 问过**：当你转开相机的时候，世界还在跑吗？

这就好比你去考驾照，考官只看你直道开得稳不稳，从来不考你"闭上眼睛三秒钟再睁开，车还在路上吗"。

---

## WRBench 怎么测这件事

paper 的核心实验设计非常聪明。它把 camera motion 重新定义成 **对 observability 的 intervention**，而不是"rendering 命令"。

### 实验流程

1. 给 model 一个 prompt："卧室里，猫从地上跳上床"
2. 让 camera 转走（target 离开视野）
3. 让 camera 转回来（target 回到视野）
4. 检查：猫在不在床上？

**关键设计**：prompt **不告诉** model 猫最后应该在床上。video 本身必须 carry 这个 evidence。这样 model 不能靠"读题"作弊。

### 6 层诊断链

paper 不是给一个分数，而是给 6 个维度的诊断：

1. **相机有没有按指令转**（Requested-camera precision）
2. **prompt-only model 的相机有没有按意图转**（Prompt-camera alignment）
3. **画面有没有崩**（Visual integrity）
4. **看得见的时候 spatial 关系和 action state 对不对**（Visible consistency）
5. **target 有没有真的离开又回来**（Re-observation support）—— 这是个 gate
6. **回来的时候 state 对不对**（Re-observed consistency）—— 只有 5 通过才算

第 5 步是 gate，第 6 步是真正的 test。这避免了 model 作弊——如果你永远把猫留在画面里，re-observed test 根本不会被触发，你不能算 pass。

---

## 最核心的发现

### 发现 1：画面好 ≠ 世界对

这是 paper 最反直觉的发现。你可能会想：画面越干净的 model，应该 world state 也越对吧？

**错**。数据显示：

- 画面最干净的 model（Hailuo, visible score 0.891/0.759）re-observation support 只有 6.3%——它太"乖"了，永远把 subject 留在 frame 里，所以测试根本没被触发
- 画面一般的 model（Gen3C, visible 0.723/0.558）support 高达 73%——它会真的让 target 离开
- 但 Gen3C 的 re-observed state 只有 0.640——回来的时候世界还是错的

**画面质量、access、re-observed correctness 是三个独立的 capability**。scale 不能 bridge 这个 gap。Wan 从 1.3B scale 到 14B，re-observed state 从 0.657 **下降**到 0.621。

这就像一个学生，考试成绩提高了，但不是因为真的学会了，而是因为更会"抄"——画面更精致，但 underlying world model 没变。

### 发现 2：in-place change 是 universal hard case

paper 设计了一个 $2 \times 2$ 的 event factor：

- **Move**（猫从地上跳到床上）：给 model 一个新坐标去 track，static scene 提供 anchor
- **In-place change**（毯子被折叠、杯子被碰倒、人坐下）：object 不移动，只是 state 变了

结果显示，move 的事件 re-observed state 普遍还行（~0.68），in-place change 的 re-observed state 普遍掉 0.10-0.15（~0.55）。

**为什么**？move 给 model 一个新 location 去 anchor，回来时可以"哦猫应该在床上"；in-place change 没有新 location，model 回来时只能 replay 最后看到的 intact configuration——所以杯子回来是 upright 的，**好像它从来没被碰倒过**。

这让我想到 LeCun 的 JEPA (https://openreview.net/forum?id=BZ5a1r-kVsf) 哲学：predictive world model 应该在 abstract representation space 里做 prediction，而不是 pixel space。in-place change 的失败恰恰说明现在的 model 没有 abstract state representation，只有 pixel-level memory。

### 发现 3：paradigm 决定 access，不决定 correctness

paper 把 23 个 model 分成 4 类 control paradigm：

1. **Prompt-only**（Hailuo, Kling）：只给一句话。support 3-6%（最低，因为相机太保守）
2. **Model-inferred**（Wan-Fun, Lingbot, LiveWorld）：内部 control。support 6-40%
3. **Source-video**（ReCamMaster, InSpatio, HyDRA）：给 reference video。support 33-62%
4. **Geometry-cache**（Gen3C, Spatia, VerseCrafter）：给 point cloud / 3D cache。support 26-73%

access 单调随"externalize 的已见 footage 量"上升。但 re-observed correctness 在所有 paradigm 上都卡在 0.58-0.66。

**这告诉我们**：给 model 更多外部 memory（point cloud、reference video）能让它"把 target 放回原位"，但不能让它"知道 target 在 unobserved 期间发生了什么"。所有这些 memory 都是 **where-memory**，不是 **what-memory**。

---

## 最戏剧性的几个 case

### HyDRA：相机控制最好，世界最差

HyDRA 的 CamPrec 是全场最高的 0.822——相机轨迹执行得极精确。但 re-observed state 是全场最低的 0.445。

为什么？HyDRA 在一个完全合成的 corpus (HM-World) 上训练 reappearance memory。当相机转到 training data 没覆盖的 region 时，它直接 fallback 到 synthetic prior——后院变成摩天大楼城市广场。

**这就是 overfit 到 training domain 在 unobserved region 的暴露**。你的相机控制再准，world model 是 hallucinate 出来的，没用。

### Lingbot：visible state 最好，但测试根本没触发

Lingbot World Act 的 visible spatial/state 是 model-inferred 类里最强的（0.874/0.719），re-observed state 看起来也不错（0.725）。但 support 只有 6.4%，n=32。

Lingbot 把 camera pose 和 object dynamics 绑死了——相机基本不动，object 永远在 frame。所以 re-observed test 几乎从未被 pose。那 32 个样本的 0.725 是 sparse reading，不能 generalize。

**这就像一个学生，每次考试都只做他会的那几道题，正确率 90%，但实际能力很弱**。

### Gen3C：access 最高，但第一帧有鬼影

Gen3C support 73% 全场第一，因为 3D cache re-projection 能把 target 精确放回原位。但前 0.4 秒有 translucent doubled overlay（cache 初始化），camera 离开 cached frustum 后画面 progressive degradation。

**access 是用 cache 换来的，但 cache 的 generative infill 还没 photometrically closed**。

---

## 为什么 scaling 救不了这件事

这是 paper 对 scaling thesis 最严肃的 empirical challenge。

Wan-Fun 1.3B → 14B：
- re-observation support: 13.8% → 18.2% ✓（更多样本进入测试）
- re-observed state: 0.657 → 0.621 **下降**

Wan-Fun 5B → A14B：
- support: 12.0% → 17.6% ✓
- re-observed state: 0.664 → 0.649 **下降**

Wan 2.1 → 2.2 升级：
- visible spatial/state 大涨（0.725/0.513 → 0.810/0.625）✓
- re-observed state 留在 0.621-0.664 band 不动

Wan 2.2 的 release notes 明确说要 target "cinematic aesthetics 和 complex motion" (https://arxiv.org/abs/2503.20314)。VBench total 从 83.96% 涨到 86.22%。但这些 metric **全部是从 generated pixels 和 text alignment 测的，从未测过 hidden-state persistence after occlusion**。

**scaling 在 observable axes 上有效，但在 hidden-state persistence 上失效**。这不是 scaling law 的 failure，是 **training objective 的缺失**。现代 video scaling 的 supervision 从未触达 unobserved 期间的 state evolution。

---

## 用 RL 文献类比

Karpathy 你肯定熟悉 Dreamer (https://danijar.com/project/dreamer/) 这条线。Dreamer 的核心是 RSSM (Recurrent State-Space Model)：

- latent state $s_t$ 在 time step 之间持续 roll-forward
- 即使没有 observation，dynamics 也在 latent space 里 evolve
- policy 可以在"imagined rollout"上训练

video diffusion model **没有这种 module**。它们的 latent state 在 unobserved 期间是 frozen 的。camera 回来时，model 要么：
- replay 最后看到的 intact configuration（Silent erasure，knock case）
- 从 training prior hallucinate（HyDRA case）
- 把 target 和 camera 耦合在一起拖走（Wan-Fun case）

这让我想到你在 Tesla 时讲过的 "system 2 thinking"——需要 world model 在 latent space 里 roll-forward planning。video generation 社区现在做的还是 "system 1"：reflexive frame generation from observed context。

---

## paper 对未来 work 的 proposal

paper 提了一个 **long-to-short recipe**：

### Step 1: 在长 horizon 上学 persistence

让 model 在 long sequence 上训练，learn "state 在 unobserved 期间 evolve"。这一步可以用 synthetic data 大量喂——比如用 game engine 生成"object 离开视野后发生 change，再回来"的 paired data。

### Step 2: 加 explicit camera-execution supervision

在 short horizon 上加 camera control 训练，但要 **explicit supervision on requested displacement**。Lingbot 的失败证明：如果 camera 和 object dynamics 绑死，visible state 好但 re-observation test 永远不被 pose。需要 disentanglement signal——类似 ReCamMaster MultiCamVideo 的"same event under 10 cameras"。

### Step 3: Endpoint-directed reward/policy training（只 proposed）

paper 在 Appendix H 说，可以用 WRBench records export 成 preference pairs，做 reward model 或 policy training。但 paper 明确说这只是 **proposed**，不是 demonstrated。

关键警告：**rewarding visible plausibility alone 会 reinforce failure mode**——model 会学会"keep salient subject in frame"，这正好是 Hailuo/Kling 现在的 bottleneck。

---

## 更深的 intuition：什么是 world model？

这篇 paper 逼着你想一个 fundamental question：**world model 到底应该 model 什么？**

### Pixel-level view

如果 world model = frame predictor，那 re-observed state 的 failure 就是合理的——你只负责 render pixels，没看到的 region 没有 pixel evidence，当然没法 reconstruct。

### Latent-level view

如果 world model = latent state roll-forward（Dreamer / JEPA 哲学），那 re-observed state 的 failure 是 fatal bug——你的 latent state 应该在 unobserved 期间继续 evolve，回来时直接 decode 出 endpoint state。

paper 的立场很明确：**world model 应该是后者**。Einstein 的 moon quote 不是诗意，是认识论要求——physical world 的 reality 独立于 observation。

这跟 LeCun 的 JEPA (https://openreview.net/forum?id=BZ5a1r-kVsf) 哲学高度一致：prediction 应该在 abstract representation space 做，不在 pixel space。in-place change 的失败恰恰说明现在的 model 没有 abstract state representation，只有 pixel-level memory。

也跟 Ha & Schmidhuber 的 World Models (https://worldmodels.github.io/) 一脉相承：compress spatial + temporal dynamics into compact latent, then roll forward in latent space。video generation 社区拿到了 spatial compression（DiT latent space），但丢了 temporal dynamics compression。

---

## 这篇 paper 的 meta-level 价值

撇开 technical contribution，这篇 paper 做了一件很有价值的事：**给 "world model" 这个被滥用的 term 一个可证伪的定义**。

现在 Sora、Wan、Kling 都说自己是 world model。但这个 label 没有可证伪的 criterion。WRBench 提供了一个：

> A world model must maintain an internal world state that evolves continuously over time, decoupled from observation, so that objects endure and events run to their conclusions whether or not a camera is watching.

这个定义是 **measurable** 的。你可以拿 WRBench 跑一下，看你的 model 在 re-observed state 上是 0.45 还是 0.80。

这让我想到 ImageNet 之前，CV 领域也是一堆"object recognition"工作，但没有统一 benchmark，每个人都说自己 SOTA。ImageNet 给了一个 measurable criterion，整个领域就 calibration 了。WRBench 有可能对 world model 社区起类似作用——把"world model"从一个 marketing term 变成一个 engineering target。

---

## 我的一些联想和延伸

### 跟 object permanence 的关联

这跟 developmental psychology 的 object permanence (https://en.wikipedia.org/wiki/Object_permanence) 概念高度相关。Piaget 说婴儿在 8 个月大时获得这个能力——知道物体在被遮挡时仍然存在。

现在的 video model 在 re-observed state 上 0.62，相当于**还没到婴儿 8 个月的水准**。这个 benchmark 本质上是在测 computational object permanence with state evolution。

### 跟 LLM 的 reasoning 的类比

LLM 社区也遇到过类似问题。GPT-3 能 generate 流利的 text，但 CoT (Chain-of-Thought) 出现前，reasoning 能力被 surface fluency 掩盖。WRBench 做的事类似于把 reasoning 从 fluency 里 disentangle 出来——re-observed state 就是 world model 的 "CoT test"。

### 跟 autonomous driving 的关联

你在 Tesla 时的 work 就是 build world model for driving。当你的 ego vehicle 被卡车遮挡时，后面的行人是不是还在原位？这是一个 production-critical 的 object permanence 问题。WRBench 的 diagnostic framework 可以直接迁移到 driving world model 的 evaluation——occlusion reasoning 是自动驾驶的核心 capability。

Tesla FSD 的 occupancy network (https://www.youtube.com/watch?v=j0z4FweC448) 就是某种 what-memory：在 voxel space 里 track occupancy state，即使被遮挡也 maintain。这跟 paper 提的 what-memory 方向一致。

### 跟 3D GS / NeRF 的关联

Gen3C 用 3D cache re-projection 实现 73% access，但 re-observed state 只有 0.640。这告诉我们：**3D reconstruction ≠ world simulation**。你可以把 scene geometry 重建得很准，但如果你没有 temporal dynamics 的 model，回来时还是只能 replay static scene。

这跟 NeRF (https://www.matthewtancik.com/nerf) 的局限一致：NeRF 是 static scene reconstruction，不是 world model。4D NeRF / Dynamic 3D GS 加了 temporal axis，但仍然是 observed trajectory 的 interpolation，不是 unobserved state 的 extrapolation。

### 跟 Minecraft / game engine world model 的关联

Genie (https://sites.google.com/view/genie-2024/home) 这种 interactive environment generation 已经有 world model 的雏形——game engine 在 unobserved 区域也 maintain state。但 Genie 的 world 是 discrete action space，state transition 是 rule-based 的。video diffusion model 要学到这种 persistent state，可能需要某种 neuro-symbolic hybrid，或者至少是 discrete latent state code（VQ-VAE 路线）。

---

## 一句话收尾

**WRBench 告诉我们：现在的 video "world model" 其实是 view-conditioned renderer，不是 world simulator。它们把 world 当 tracking shot 拍，camera 回来时 target 还在它被 abandon 时的 state，而非 event 已经演化到的 endpoint。scaling 救不了这件事，因为 training objective 从未监督 hidden-state persistence。要 build 真正的 world model，需要一个显式的 what-memory module + endpoint-persistence training objective + camera-dynamics disentanglement。**

这就像 ImageNet 之前的 CV——大家都说做 object recognition，但没有 measurable criterion。WRBench 给 world model 这个 term 一个可证伪的定义，这可能是它最大的长期价值。

---

# WRBench：当相机转开时，世界还在演化吗？

## 1. 论文的核心诘问

开篇用 Einstein 的那句"I like to think the moon is there even if I am not looking at it"做了一个非常锋利的 framing。这把整个 evaluation 的关注点从"rendering convincing frames"推到了"maintaining an internal world state that keeps evolving over time, decoupled from observation"。

这跟 RL 文献里 Dreamer / PlaNet /世界模型的传统定义一脉相承（latent state roll-forward in time），但 paper 把这个要求**显式地投影到 video generation 社区**——你们说自己是 world model，那你们的 latent representation 在 unobserved 期间到底有没有 dynamic evolution？ WRBench 的答案是：**几乎没有**。

Project page: https://jinplu.github.io/WRBench  
Code: https://github.com/JinPLu/WRBench

---

## 2. 关键 concept：viewpoint intervention 作为 observability 的探针

paper 把 camera motion 重新定义为 **intervention on observability**，而不是 rendering command。这是一个非常聪明的实验设计 move：

- prompt 指定 initial scene + event（"in the bedroom, a cat jumps onto the bed"）
- 然后 camera 转走，target 离开视野
- camera 转回来，target 应该已经在 bed 上（event 的 endpoint）
- 但实际看到的：cat 还在 floor 上、消失了、出现在错位置、变成两个副本

这种 failure 的诊断困难在于 **attribution problem**：你看到 cat 回来时是错的，但这是因为 (a) camera 根本没转走，还是 (b) camera 转走了但 world 停止演化？单一分数无法区分。 WRBench 的核心设计就是把这个 attribution chain 显式拆开。

---

## 3. Natural-25 prompt suite 与 event-view record

### 3.1 基本评估单元：event-view record

paper 在公式 (1) 定义了 evaluation unit：

$$r_i = (x_i^0, e_i, \tau_i, \nu_i, \pi_i)$$

各符号含义：
- $r_i$：第 $i$ 个 event-view record（**不是** 一个 prompt 或一段 clip，而是 prompt + event + camera condition + visibility regime + interface variant 的整体）
- $x_i^0$：initial observation，第一帧
- $e_i$：specified event，比如"a cat jumps onto the bed"
- $\tau_i$：intended viewpoint intervention，相机怎么转
- $\nu_i$：visibility regime（visible / temporarily hidden / returned）
- $\pi_i$：prompt 或 interface variant（不同 model 接受不同输入形式）

这个 tuple 设计的关键是 **prompt 不暴露 returned endpoint state**——生成视频本身必须 carry 任何 re-observed-state 的 evidence，这样 model 不能作弊。

### 3.2 4 级 event design（$2 \times 2$ factorial）

Natural-25 把每个 scene family × 四个 event level，沿两个独立 axis 切：

| Factor | Levels |
|---|---|
| Spatial displacement | move / no-move |
| State change | change / no-change |

得到 4 cell：none、spatial-only、state-only、full。这个 design 后面在 Finding 2 里被用来分离两种失败模式，是非常 clean 的 ablation 设计。

25 个 scene family 从 19 个 venue 采，覆盖 indoor / outdoor / human / animal actors。Domestic indoor 是最大组（8/25），但每个 family 等量贡献，不奖励对高频场景的 memorization。

---

## 4. WRBenchLib：异构 generator 的公平化层

### 4.1 generation-provenance map（公式 2）

$$z_{i,m} = \Phi_m(r_i) = (u_{i,m}, d_{i,m}, v_{i,m}, \eta_{i,m})$$

- $z_{i,m}$：model $m$ 在 record $i$ 上的完整 provenance record
- $\Phi_m$：model $m$ 的映射函数（WRBenchLib 提供）
- $u_{i,m}$：model-specific input。可能是 trajectory、source video、geometric condition、或纯 prompt
- $d_{i,m}$：condition **actually delivered**——这点很重要，因为有些 model 会忽略部分输入
- $v_{i,m}$：generated video
- $\eta_{i,m}$：proprovenance log，记录完整 pipeline 以复现

### 4.2 4 种 viewpoint condition type

paper 把 23 个 model 划分为 4 种 control paradigm：

| Paradigm | 代表 model | 它 externalize 了什么 |
|---|---|---|
| Source-video | ReCamMaster, HyDRA, InSpatio | appearance + layout + 部分 dynamics 的 reference stream |
| Geometry-cache | Gen3C, Spatia, VerseCrafter | point cloud / 3D cache / 4D control |
| Model-inferred | Wan-Fun, Lingbot, LiveWorld, Hunyuan* | 只有内部 camera/action/state control，没有外部 view-state reference |
| Prompt-only | Hailuo, Kling, Wan API, HappyHorse | 只有自然语言 camera intent |

这个分类的设计哲学是 **比较的公平性**：让一个接受 dense trajectory 的 model 和一个只接受 sentence 的 model 比"camera precision"是不公平的。所以 WRBench 对 prompt-only model 用 CamAlign（common-yaw diagnostic）替代 CamPrec（strict trajectory match）。

---

## 5. 6 个 evaluation dimensions 的诊断链

paper 最重要的贡献是这条 hierarchical diagnostic chain。每个 dimension 单独报告，**不 collapse 成一个 leaderboard score**：

| # | Dimension | Denominator | 测什么 |
|---|---|---|---|
| (i) | Requested-camera precision | 7,500 local requested-control rows | 给了 trajectory 的 model 是否 follow |
| (ii) | Prompt-camera alignment | 7,800 yaw rows | prompt-only model 是否 follow common yaw intent |
| (iii) | Visual integrity | 9,600 outputs | frame evidence 是否 structurally intact（cuts / disappearance / identity drift / structural collapse） |
| (iv) | Visible spatial & state consistency | 9,600 outputs | target 在视野内时 spatial relation 和 action state 是否正确 |
| (v) | Re-observation support | 9,600 outputs | target 是否真的 leave 然后 return 到 judgeable 形态 |
| (vi) | Re-observed spatial & state consistency | 2,073 judgeable rows | returned target 是否 preserve 了 event endpoint |

(vi) 是 **conditional on (v)**——这是整个 benchmark 设计的精髓。如果 target 从没离开过视野，或回来时无法辨认，那就 NA，不算 pass 也不算 fail。这避免了"model 把 target 永远留在屏幕里"作弊通过 re-observed test。

---

## 6. 评估方法的技术细节

### 6.1 Visual integrity（公式 3, 11, 12, 13）

公式 (3)：

$$I_{\text{vis}}(v) = \min(s_{\text{global}}(v), s_{\text{local}}(v))$$

- $I_{\text{vis}}$：video $v$ 的 visual integrity score
- $\min$ 操作：global 和 local 都要通过，avoid 一个 high-global 但局部崩坏的视频被算 pass

Global component（公式 11）：

$$s_{\text{global}}(v) = [g_1^\top g_T]_0^1$$

- $g_t$：第 $t$ 帧的 $\ell_2$-normalized DINOv2 CLS token（参考 https://dinov2.metamindresearch.com/）
- $g_1^\top g_T$：第一帧与最后一帧 CLS token 的内积，即 cosine similarity
- $[\cdot]_0^1$：clip 到 $[0,1]$

Local component（公式 12, 13）：

$$\mathcal{B}_t = \{\max_{j \in \Omega_{t+1}} m_{ij}^{(t)}\}_{i \in \Omega_t} \cup \{\max_{i \in \Omega_t} m_{ij}^{(t)}\}_{j \in \Omega_{t+1}}$$

$$s_{\text{local}}(v) = P_{20}\left(\{P_{20}(\mathcal{B}_t)\}_{t=1}^{T-1}\right)$$

- $m_{ij}^{(t)}$：相邻帧 patch-token cosine similarity
- $\Omega_t$：第 $t$ 帧的非 padding patch 索引集
- $\mathcal{B}_t$：bidirectional best-match set，避免固定坐标匹配对 object/camera motion 过敏感
- $P_{20}$：20th percentile。用低 tail 而非 mean，让分数对 **localized collapse / ghosting / disappearance / hard cuts / identity drift** 敏感，即便 median frame-pair 看起来还行

注意：DINOv2 不是 object detector / tracker / prompt-grounded mask。它只是 visual-evidence floor，告诉你"这帧能不能信"，告诉你"这帧能不能信"，告诉你"这帧能不能信"——不是 state correctness。

实现细节（Appendix D.1）：
- 3 fps 时基采样
- 保留首末帧
- 序列 cap 24 帧
- resize 不 center crop（WRBench 场景经常把关键 subject 放在边缘）
- patch size 14，padding mask 掉

### 6.2 Visible / re-observed consistency probes（公式 4, 5, 6, 7, 8）

paper 用 prompt-conditioned yes/no probes scored by Qwen-3.5-9B。

公式 (4)：

$$p_{\text{yes}}(q \mid v, c) = \frac{\exp L_{\text{yes}}(q, v, c)}{\exp L_{\text{yes}}(q, v, c) + \exp L_{\text{no}}(q, v, c)}$$

- $q$：probe question（fixed template，measuring 之前就 freeze）
- $v$：video
- $c$：context bundle，包含 scene / event / target object / camera condition / visibility regime / dimension-specific rubric
- $L_{\text{yes}}, L_{\text{no}}$：log-sum-exp over allowed yes/no answer tokens

公式 (5) — polarity adjustment：

$$e_a(q, v, c) = \begin{cases} p_{\text{yes}}(q \mid v, c), & q \in \mathcal{P}_a^+ \\ 1 - p_{\text{yes}}(q \mid v, c), & q \in \mathcal{P}_a^- \end{cases}$$

- $a$：metric（spatial 或 state）
- $\mathcal{P}_a^+$：positive probes，问"intended evidence 是否 present"
- $\mathcal{P}_a^-$：negative probes，问"counter-evidence 是否 absent"
- 对 negative probe 取 $1 - p_{\text{yes}}$：因为"counter-evidence absent"是好事

公式 (6)：

$$M_a(v, c) = \frac{1}{|\mathcal{P}_a^+| + |\mathcal{P}_a^-|} \sum_{q \in \mathcal{P}_a^+ \cup \mathcal{P}_a^-} e_a(q, v, c)$$

这是 visible measurement 的形式。

公式 (7) — re-observation 的 judgeability predicate：

$$S_{\text{return}} = \{a_{\text{rsp}}, a_{\text{rst}}\} \quad \text{and} \quad R_a(v, c) = H(v, c) \land U(v, c) \land J_a(v, c)$$

- $S_{\text{return}}$：return metrics 集合，$a_{\text{rsp}}$ = re-observed spatial, $a_{\text{rst}}$ = re-observed state
- $R_a$：judgeability predicate（boolean）
- $H$：存在 nontrivial hidden/unjudgeable interval
- $U$：target returns to observable field
- $J_a$：returned evidence 足够 identifiable for metric $a$ to be scored
- $\land$：logical AND，三者都必须满足

**关键设计**：$J_a$ 由**单独的 VLM gate**（Qwen-3-VL-Instruct-8B）评估，和 Qwen-3.5-9B scoring evaluator 隔离开。这样 re-observation applicability 在 re-observed-consistency score 计算之前就被独立 establish。

公式 (8)：

$$M_a(v, c) = \begin{cases} \frac{1}{|\mathcal{P}_a^+| + |\mathcal{P}_a^-|} \sum_q e_a(q, v, c), & R_a(v, c) = 1 \\ \text{NA}, & R_a(v, c) = 0 \end{cases}$$

当 judgeability 失败，metric 是 NA，不是 0 也不是 1。

### 6.3 Aggregation（公式 9）

$$\mathcal{D}_{m,a} = \{i : S_a(v_{i,m}, c_i) = 1\}$$
$$\rho_{m,a} = \frac{|\mathcal{D}_{m,a}|}{|\mathcal{Z}_a|}$$
$$\bar{M}_{m,a} = \begin{cases} \frac{1}{|\mathcal{D}_{m,a}|} \sum_{i \in \mathcal{D}_{m,a}} M_a(v_{i,m}, c_i), & |\mathcal{D}_{m,a}| > 0 \\ \text{NA}, & |\mathcal{D}_{m,a}| = 0 \end{cases}$$

- $\mathcal{D}_{m,a}$：model $m$ 在 metric $a$ 上的 applicable denominator set
- $\rho_{m,a}$：support rate（access 的量化）
- $\bar{M}_{m,a}$：conditional mean，只在 judgeable subset 上算

报告时 $\rho$ 和 $\bar{M}$ 同时给出。这就是 paper 反复强调的 preservation-access-re-observed-consistency 三层分离。

### 6.4 Human calibration（公式 10）

$$\Delta_{p,a} = M_a(v_p^A, c_p) - M_a(v_p^B, c_p)$$
$$\rho_a = \text{Spearman}(\{y_{p,a}\}, \{\Delta_{p,a}\})$$
$$\hat{y}_{p,a}(\tau_a) = \begin{cases} 1, & \Delta_{p,a} > \tau_a \\ 0, & |\Delta_{p,a}| \leq \tau_a \\ -1, & \Delta_{p,a} < -\tau_a \end{cases}$$

- $\Delta_{p,a}$：metric 差值
- $y_{p,a} \in \{-1, 0, 1\}$：ordered human label for pair $p$ on axis $a$
- $\rho_a$：Spearman rank correlation
- $\hat{y}_{p,a}(\tau_a)$：thresholded decision；$\tau_a$ 在 reporting前 fixed，per-model/per-table 不调
- $\mathcal{C}_a$：human 和 automatic evidence 都 defined 的 pair 子集

2,547 deduplicated human annotator verdicts，每个 dimension 独立 judge。

---

## 7. 实验：23 个 model 的 diagnostic profile

### 7.1 Table 2 关键数字速读

**Source-video**：
| Model | CamPrec | CamAlign | Integ | Reobs supp | Vis sp | Vis st | Reobs sp | Reobs st |
|---|---|---|---|---|---|---|---|---|
| ReCamMaster | 0.717 | 0.729 | 0.740 | 58.5% | 0.715 | 0.535 | 0.665 | 0.616 |
| InSpatio 14B | 0.693 | 0.661 | 0.824 | 62.3% | 0.821 | 0.668 | 0.734 | 0.664 |
| HyDRA | 0.822 | 0.855 | 0.691 | 33.2% | 0.648 | 0.500 | 0.509 | 0.445 |

**Geometry-cache**：
| Model | CamPrec | CamAlign | Integ | Reobs supp | Vis sp | Vis st | Reobs sp | Reobs st |
|---|---|---|---|---|---|---|---|---|
| Gen3C | 0.699 | 0.764 | 0.749 | 73.0% | 0.723 | 0.558 | 0.681 | 0.640 |
| Spatia | 0.704 | 0.482 | 0.763 | 25.8% | 0.731 | 0.541 | 0.600 | 0.586 |
| VerseCrafter | 0.781 | 0.667 | 0.846 | 28.0% | 0.707 | 0.508 | 0.607 | 0.584 |

**Model-inferred**：
| Model | CamPrec | CamAlign | Integ | Reobs supp | Vis sp | Vis st | Reobs sp | Reobs st |
|---|---|---|---|---|---|---|---|---|
| Wan-Fun 2.1-1.3B | 0.771 | 0.729 | 0.842 | 13.8% | 0.725 | 0.513 | 0.709 | **0.657** |
| Wan-Fun 2.1-14B | 0.757 | 0.526 | 0.846 | 18.2% | 0.733 | 0.530 | 0.659 | 0.621 |
| Wan-Fun 2.2-5B | 0.724 | 0.335 | 0.812 | 12.0% | 0.805 | 0.607 | 0.709 | **0.664** |
| Wan-Fun 2.2-A14B | 0.758 | 0.553 | 0.848 | 17.6% | 0.810 | 0.625 | 0.698 | 0.649 |
| Lingbot World Cam | 0.513 | 0.175 | 0.870 | 6.0% | 0.876 | 0.735 | 0.717† | 0.663† |
| Lingbot World Act | 0.468 | 0.168 | 0.856 | 6.4% | 0.874 | 0.719 | 0.771† | 0.725† |
| LiveWorld | 0.812 | 0.856 | 0.775 | 39.6% | 0.703 | 0.541 | 0.661 | 0.600 |

**Prompt-only**：
| Model | CamPrec | CamAlign | Integ | Reobs supp | Vis sp | Vis st | Reobs sp | Reobs st |
|---|---|---|---|---|---|---|---|---|
| Hailuo 2.3 | – | 0.075 | 0.829 | 6.3% | 0.891 | 0.759 | 0.719† | 0.642† |
| Kling v2.6 | – | 0.094 | 0.864 | 3.3% | 0.854 | 0.674 | 0.711† | 0.617† |

† = sparse support，n 很小，要谨慎读。

### 7.2 几个戏剧性反差

1. **HyDRA**：CamPrec 最高（0.822，全场第一），但 re-observed state 最低（0.445，全场垫底）。paper 在 Appendix E.5 解释：HyDRA 在 HM-World 合成 corpus 训练 reappearance memory，camera 执行得极精确，但 yaw 转到没见过的 region 时直接 fallback 到 synthetic training prior——backyard 变成 generic urban plaza with skyscrapers。这就是 overfit to training domain 在 unobserved region 暴露的典型 case。

2. **Lingbot World Act**：visible spatial/state 是 model-inferred 里最强（0.874/0.719），re-observed state 也看起来高（0.725），但 support 只有 6.4%，n=32。也就是说 Lingbot 把 camera 绑死在 object dynamics 上，相机基本不动，target 永远在屏幕里——re-observed test 几乎根本没被 pose。CamAlign 0.168 是 controllable model 里最低的。

3. **Gen3C**：support 73% 全场第一，因为 3D cache re-projection 能把 target 放回原位。但 same re-projection pipeline 在 camera 离开 cached frustum 后 progressive degradation，第一帧还有 ~0.4s translucent doubled overlay。

4. **Prompt-only**（Hailuo, Kling）：visible quality 全场最高（0.891/0.759），但 support 3-6%。它们的 bottleneck 是 **从不创建 test**——camera 太 timid，target 永远在 frame。

---

## 8. 六大 Findings 详解

### Finding 1：Visual quality ≠ re-observed state correctness

Figure 4 的 Pearson correlation 结构：

- visible spatial ↔ visible state：$r = 0.97$（强相关，自己成一个 block）
- re-observed spatial ↔ re-observed state：$r = 0.94$（自己成 block）
- visible → re-observed：$r = 0.60 - 0.79$（中等）
- **re-observation support ↔ visible spatial：$r = -0.42$（负相关！）**
- re-observation support ↔ visual integrity / re-observed state：$r = -0.15$

负相关是这篇 paper 最漂亮的发现之一：**画面越干净的 model，越倾向于把 subject 锁在 frame 里**，于是 re-observation test 反而越少被创建。

Figure 5 的 grouped bars：
- static camera：support ≈ 0
- yaw camera：support ≈ 40%
- 但两种 yaw 之间 re-observed state 差距 < 0.01

camera motion **decides whether the test can be run, not its result**。这是 access vs. correctness 的彻底解耦。

### Finding 2：in-place state change 是 universal hard case

$2 \times 2$ factorial 的 paired Wilcoxon 结果：

| Contrast | Visible spatial | Visible state | Re-observed spatial | Re-observed state |
|---|---|---|---|---|
| Add move | +0.008 (n.s.) | +0.070 (p<0.01) | – | +0.038 (p<0.01) |
| Add in-place change | **−0.114 (p<0.001)** | −0.031 (n.s.) | −0.075 (p<0.01) | −0.068 (p<0.001) |
| Per-model interaction | – | – | – | mean −0.009, p=0.45 |

**机制解读**：
- 一个 move 给 model 一个 **新坐标** 去跟踪，static scene 提供 anchor
- 一个 in-place change（fold, knock, sit）**不提供新 location**，altered object 在原位 drift 和 smear
- 两个 axis 没有交互——move 在 in-place 之上的 benefit 是加性的，不 rescue

Appendix E.2 给出三种 distinct failure signature（同一 in-place cause）：
- fold：blanket 回来但 fold 状态错
- knock：cup 回来 upright，**好像 event 从没发生**——carrier replay 最后观察到的 intact configuration，silently erase change
- sit：seated person 在 occlusion 中 dissolve，scene 短暂崩塌，再 displaced 出现

这三种 surface artifact 共享一个 root cause：**no anchor for an unobserved change**。

### Finding 3：paradigm 决定 access，不决定 correctness

Table 3 access ladder（按 re-observation support 排）：

| Model | Subtype | Reobs supp | CamAlign | Vis avg |
|---|---|---|---|---|
| Gen3C | Geometry-cache | 73.0% | 0.764 | 0.641 |
| InSpatio 14B | Source-video | 62.3% | 0.661 | 0.745 |
| ReCamMaster | Source-video | 58.5% | 0.729 | 0.625 |
| VerseCrafter | Geometry-cache | 28.0% | 0.667 | 0.608 |
| Wan-Fun A14B | Model-inferred | 17.6% | 0.553 | 0.718 |
| Hailuo 2.3 | Prompt-only | 6.3% | 0.075 | 0.825 |
| Kling v2.6 | Prompt-only | 3.3% | 0.094 | 0.764 |

access 单调随 "externalize 的 already-seen footage 量" 下降。但 Table 4 显示 re-observed state 在 relocation vs. in-place 上**每个 paradigm 都掉 ≈ 0.10-0.15**：

| Model | Spatial event | State event | Gap |
|---|---|---|---|
| Gen3C | 0.711 | 0.559 | +0.152 |
| InSpatio 14B | 0.720 | 0.591 | +0.129 |
| Spatia | 0.633 | 0.512 | +0.121 |
| VerseCrafter | 0.638 | 0.527 | +0.111 |
| ReCamMaster | 0.686 | 0.589 | +0.097 |
| Wan-Fun A14B | 0.682 | 0.611 | +0.071 |

Gen3C gap 最大（0.152），因为它最擅长 re-projecting static scene——但 stored record of where surfaces were **cannot reconstruct a change it never observed**。这就是 **what-memory** 缺失的实证证据。

model-inferred → geometry-cache：access 17.6% → 73%（4 倍），但 relocation re-observed state 0.682 → 0.711（几乎不动），in-place 0.611 → 0.559（**反而下降**）。richer paradigm 只把 bottleneck 从"object 是否回来"移到"回来是否反映 unobserved change"。

### Finding 4：Scaling 不 deliver endpoint persistence

Wan-Fun 在 family 内 scale 的两个 clean pair：

| Pair | Reobs supp | Re-observed state |
|---|---|---|
| Wan 2.1 1.3B → 14B | 13.8% → 18.2% ↑ | 0.657 → 0.621 ↓ |
| Wan 2.2 5B → A14B | 12.0% → 17.6% ↑ | 0.664 → 0.649 ↓ |

scaling 提升 access，**降低** conditional re-observed state。Wan 2.2 升级 explicit target "cinematic aesthetics 和 complex motion"（参考 https://arxiv.org/abs/2503.20314），VBench total 83.96% → 86.22% 涨，但 re-observed state 留在 0.621-0.664 band 不动。

paper 的解读：modern video scaling **tuned 到 observable axes**（pixels + text alignment），这些 supervision 从未触达 hidden-state persistence after occlusion。所以这是一个 **compositional gap 而非 scaling law**。

### Finding 5：缺的是 what-memory，不是 camera encoding

paper 把所有 controllable model 按 **state carrier**（不是 camera-encoding format）排序成 ladder：

| Carrier | 代表 | Reobs supp | Re-observed state |
|---|---|---|---|
| Camera-lens control only | Wan-Fun 2.1/2.2 | 12-18% | 0.62-0.66 |
| External geometry adapter | VerseCrafter GeoAdapter, Spatia point cloud | 26-28% | ~0.58 |
| Source-video carrier | ReCamMaster, InSpatio | 58-62% | 0.62-0.66 |
| Explicit reappearance memory | HyDRA Memory Tokenizer + Dynamic Retrieval Attention | 33.2% | **0.445** |
| Dedicated state adapter | LiveWorld out-of-sight channel | 39.6% | 0.600 |
| Pose/action conditioning | Lingbot World/Act | 6% | sparse |

camera-encoding format（Plücker ray embeddings vs. one-hot pose tokens vs. C2W matrix injection）是 **second-order**。CameraCtrl (https://hehao13.github.io/projects/CameraCtrl/) 用 Plücker，MotionCtrl (https://wzhouxia.github.io/MotionCtrl/) 直接 inject C2W，within-method ablation 只显示 small representation effect。

**关键 insight**：每个 carrier 都 store "where to look back"，**没有一个 store "what changed while hidden"**。

LiveWorld 的 w/o Event Evo ablation 是关键证据：移除 generative evolution engine，foreground fidelity 崩，background geometry 完好——证实 **spatial memory alone never writes hidden dynamics**。

### Finding 6：Endpoint persistence 是 unwritten training objective

paper 读 public training ladder，发现 **no public loss 或 documented post-training stage supervises the unobserved outcome**：

- Wan-Fun control-camera fine-tuning：scope + data 未披露，只 supervises lens conditioning
- ReCamMaster MultiCamVideo（13,600 scenes × 10 synchronized cameras）：supervises camera/content disentanglement，但 paired distribution 上训练，re-observed state 卡在 0.616
- VerseControl4D（35,000 clips，GeoAdapter only，backbone frozen）：supervises external layout，re-observed state 0.584
- HyDRA HM-World（59,000 clips with exit-entry events）：supervises reappearance retrieval，re-observed state 0.445（最弱，因为 synthetic-domain fallback）
- LiveWorld VACE state adapter：5,000 steps，supervise out-of-sight channel latent dynamics，re-observed state 0.600
- Lingbot 长轨迹 curriculum：supervise "objects evolving and reappearing within a sequence"——这是最接近的，但 camera 绑死在 object dynamics 上，support 6%

**关键 observation**：没有一个 public stage 是 **reinforcement learning on the endpoint**。所有 post-training 都是 distribution-matching distillation（ReCamMaster/InSpatio JDMD, Hunyuan WorldPlay Context-Forcing, Lingbot causal-plus-DMD）或 reward-weighted DMD（MagicWorld）。"reward" 在这个 literature 里 shape distillation batch，不是 environment-return objective。

paper 提议 **long-to-short recipe**：
1. 先在长 horizon 上 learn persistence（unobserved 期间 state 持续 evolve）
2. 再加 explicit camera-execution supervision with requested-displacement 标签
3. optional：endpoint-directed reward or policy training（Appendix H，只 proposed）

---

## 9. Human calibration 数据

Table 5：AC1 agreement
- View execution: 0.898
- Visual integrity: 0.877 (Spearman ρ = 0.709)
- Visible spatial: 0.788
- Visible event-state: 0.790
- Re-observed spatial: 0.875 (1 reversal out of 136)
- Re-observed event-state: 0.937 (8 reversals out of 136)

re-observed 维度的 agreement 比 visible 更高，这有点反直觉但讲得通：visible 评分要 judge 细节，re-observed 是 endpoint match/no-match 的 binary 判断，更容易 align。

Table 16 的 judgeability agreement：
- Re-observed spatial：J = 0.941, NJ = 0.966, Bal. = 0.954
- Re-observed event-state：J = 0.941, NJ = 0.946, Bal. = 0.943

judgeability gate 本身高度可靠。

---

## 10. 对 world model 设计的 implication

把所有发现压成一个 intuition：**video diffusion models 现在做的是 view-conditioned rendering，不是 world simulation**。它们的 latent state 在 unobserved 期间是 frozen 的，camera 回来时 replay 最后看到的 intact configuration，或者从 training prior hallucinate。

要真正 build persistent world model，需要三个 component：

### 10.1 一个显式的 state writer

需要在 latent space 里有 module 持续写入 **what changed**，即使该 region 没被 render。RL 文献里 Dreamer (https://danijar.com/project/dreamer/) 的 RSSM 就是这种 role——roll-forward latent dynamics in time。video model 现在没有 analog。

### 10.2 Endpoint-persistence training objective

需要监督 unobserved outcome。Long-to-short recipe 的 intuition：
- 长 horizon 让 model 学会 state 在 unobserved 期间 evolve（这一步可以 synthetic data 大量喂）
- 短 horizon 加 camera control，但要 explicit supervision on requested displacement（让 camera 不绑死在 object 上，Lingbot 的失败 case）

### 10.3 Disentangle camera motion 从 object dynamics

Lingbot 的 case 证明：pose/action conditioning 能 preserve visible world state，但 camera 被绑死导致 re-observation test 永远不被 pose。所以训练时需要 explicit disentanglement signal，类似 ReCamMaster MultiCamVideo 的"same event under 10 cameras"。

### 10.4 Avoid synthetic-domain fallback

HyDRA 在 HM-World 训练，unobserved region 直接 fallback 到合成 prior。需要训练 data 覆盖足够多样的 unobserved dynamics，或者用某种 domain-randomization 让 model 不会 overfit 到单一 synthetic style。

---

## 11. 与相关 benchmark 的 positioning

Table 1 总结：

| Benchmark | World Dynamics | Unified Control | Visual Quality | State Robustness | Evolution Consistency | Pathway Diagnostics |
|---|---|---|---|---|---|---|
| VBench (https://vchitect.github.io/VBench-project/) | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| VBench-2.0 (https://arxiv.org/abs/2503.21755) | ○ | ○ | ○ | ✗ | ✗ | ✗ |
| WorldModelBench | ✓ | ✗ | ○ | ✗ | ✗ | ✗ |
| WorldScore (https://worldscore.github.io/) | ○ | ○ | ✓ | ✗ | ✗ | ✗ |
| WorldMark | ✓ | ○ | ○ | ✗ | ✗ | ✗ |
| WBench | ○ | ✓ | ○ | ✗ | ✗ | ✗ |
| iWorld-Bench | ○ | ○ | ✓ | ✗ | ✗ | ✗ |
| MIND | ○ | ○ | ○ | ✗ | ✗ | ✗ |
| STEVO-Bench | ✓ | ✗ | ○ | ○ | ○ | ✗ |
| LiveBench (LiveWorld) | ○ | ○ | ○ | ○ | ○ | ✗ |
| MBench | ✓ | ○ | ✗ | ○ | ○ | ✗ |
| **WRBench** | ✓ | ✓ | ○ | ✓ | ✓ | ✓ |

WRBench 的独特之处是把 camera control 重新框为 **evidence intervention** 而非 rendering target。最接近的邻居是 STEVO-Bench、LiveWorld、MBench——它们都关心 out-of-sight dynamics，但 WRBench **separate visible evolution / re-observation access / re-observed-state consistency** 三个层，避免把"missed return"和"wrong return"混成一个 score。

---

## 12. 局限与未来方向

### 12.1 Dense-control leakage risk（Appendix H.2）

paper 警告：如果未来加入 dense control（target-view depth, segmentation, edge maps, LiDAR, HD maps），这些 control **可能 encode post-event endpoint**，turning world-state test into control-following。三种 setting 分离 claim：

1. Source-only control：control 只从 source view 提，target-view endpoint 不 supply
2. Endpoint-masked target control：target-view control 允许，但 object/contact endpoint region 被 mask
3. Full target control：upper bound，不算 hidden-state inference 证据

这跟 Cosmos (https://github.com/nvidia-cosmos) 这类 dense-control 系统 future 加入 benchmark 时要带 explicit input-policy label。

### 12.2 VLM-labeled masks / boxes extension

现在 probes 是 yes/no，未来可以加 VLM-labeled masks, boxes, dense control settings 做 finer-grained 评估。

### 12.3 Preference-pair export（Appendix H）

WRBench records 可以 export 成 preference pairs 给 reward model 或 policy training。Table 17：2,680 input → 2,506 kept pairs after missing-asset removal。可以 mine pairs over：
- target-relative camera displacement
- visual integrity
- judgeable re-observation
- re-observed consistency

paper 明确说：rewarding visible plausibility alone 会 reinforce WRBench 暴露的失败——**keep salient subject in frame, fail to create judgeable hidden-and-returned test**。

---

## 13. 一个直觉性的总结

把 paper 提炼成一句：**current video "world models" 把 world 当 tracking shot 拍，camera 回来时 target 还在它 abandoned 时的 state，而非 event 已经演化到的 endpoint**。

用 RL 文献类比：Dreamer 类 world model 有 RSSM 在 latent space 持续 roll-forward，unobserved 期间 dynamics 继续 evolve；video diffusion 没有这种 module，latent state 在 unobserved 期间冻结，回来时 replay 或 hallucinate。 WRBench 把这个 gap 用 9,600 个 video、23 个 model、6 个 dimension 的 diagnostic chain 系统化暴露，并指出未来 work 的方向是 **what-memory + endpoint-persistence objective + camera-dynamics disentanglement**。

Scaling law 在 visible axes 上有效（pixels, motion quality, instruction adherence），但在 hidden-state persistence 上失效——0.62 这个数从 1.3B 到 14B 不动甚至下降。这是 paper 对 scaling thesis 的最严肃 empirical challenge 之一。

最后一句 Einstein 的话其实点出了一个非常 deep 的认识论问题：**physical world 的 reality 独立于 observation**。一个真正的 world model 必须在 representation level 满足这一点，否则它只是 conditional frame generator。WRBench 给了这个 property 第一个 measurable form。

---

## 14. 参考链接

- WRBench project: https://jinplu.github.io/WRBench
- WRBench code: https://github.com/JinPLu/WRBench
- VBench: https://vchitect.github.io/VBench-project/
- WorldScore: https://worldscore.github.io/
- Wan: https://github.com/Wan-Video/Wan2.1
- Gen3C: https://research.nvidia.com/labs/toronto-ai/Gen3C/
- ReCamMaster: https://jabulong.github.io/Recamaster/
- LiveWorld: https://arxiv.org/abs/2603.07145
- Lingbot: https://arxiv.org/abs/2601.20540
- HyDRA: https://arxiv.org/abs/2603.25716
- InSpatio World: https://arxiv.org/abs/2604.07209
- MotionCtrl: https://wzhouxia.github.io/MotionCtrl/
- CameraCtrl: https://hehao13.github.io/projects/CameraCtrl/
- VGGT: https://vgg-t.github.io/
- DINOv2: https://dinov2.metamindresearch.com/
- Cosmos: https://github.com/nvidia-cosmos
- Dreamer: https://danijar.com/project/dreamer/
- Hunyuan GameCraft: https://arxiv.org/abs/2506.17201
- MagicWorld: https://arxiv.org/abs/2511.18886
- WorldPlay: https://arxiv.org/abs/2512.14614
