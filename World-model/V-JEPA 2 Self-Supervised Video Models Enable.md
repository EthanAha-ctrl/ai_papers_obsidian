---
source_pdf: V-JEPA 2 Self-Supervised Video Models Enable.pdf
paper_sha256: 9cfcfde5fb0d9730637da5b9e7317825c3f3d09e91f3553e22eeba42c74d2226
processed_at: '2026-08-13T00:02:12-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# V-JEPA 2 用人话讲讲

Andrej，那我换个讲法。想象咱俩在办公室白板前，喝着咖啡，我给你白话一遍这篇 paper 到底干了啥，为啥它有意思。

---

## 这群人想解决什么问题

LeCun 这几年一直念叨一个想法：人怎么学世界的？小婴儿坐在那儿看几个小时，啥也不干就观察，慢慢脑子里建了个"世界模型"——知道杯子从桌子掉下来会砸地上，知道推门把手门会开，知道自己伸手过去手会到哪儿。这种从**纯观察**学到的世界模型，让人能在新环境里做没见过的事。

现在的 AI 机器人学习方法大概两条路：

**第一条路是 VLA**（Vision-Language-Action），代表是 RT-2、π0、OpenVLA、Gr00t N1。思路是：拿一个在互联网视频和文本上预训练好的大模型，再用一堆人遥操作的机器人轨迹做 behavior cloning，直接学一个从"看到啥 + 任务描述"到"输出 action"的 policy。效果好，但问题是它本质是模仿，没显式学物理世界规律，新情况容易翻车，而且需要成功示范。

**第二条路是 video generation world model**，代表是 NVIDIA 的 Cosmos。思路是：学一个能生成未来视频的模型，给它当前帧 + action，它"画出"未来帧。听起来很美，但实际用起来两个问题：(1) 生成一帧要跑几十步 diffusion，慢得要命；(2) 它把 capacity 浪费在"草地第几片叶子往哪飘"这种不可预测的细节上，对控制机器人没帮助。

V-JEPA 2 走第三条路：**在学到的 representation 空间里预测，不碰像素**。用一个 self-supervised 方式在 100 万小时网络视频上预训练一个 encoder，让它学会"视频接下来会变成什么样"的抽象表示；然后再用 62 小时的机器人视频（不需要成功标记，不需要 reward），训一个小 predictor 把 action 接进来；最后用这个 predictor 做 planning，控制真实机器人。整个过程没有语言监督，没有 reward，没有 expert 示范，纯靠观察 + 一点点交互。

结果：Franka 机械臂两个不同实验室 zero-shot 抓放物体，pick-and-place 80% 成功率。Cosmos 同样任务 0%。

---

## JEPA 这个思路到底妙在哪儿

LeCun 的核心论点其实一句话：**别在像素空间预测，太浪费**。

想象你看一段视频：一个人把杯子推到桌子边。下一帧杯子会到哪儿？如果你是 pixel prediction model，你得预测杯子上每个像素、桌子上每个像素、背景墙每个像素。但背景墙怎么变根本不重要，杯子的纹理怎么变也不重要，重要的就是杯子**这个物体**位置变了。

JEPA 干的事：先把视频通过一个 encoder 压成一组 representation token（每个 token 大致对应一个时空区域），然后让 predictor 在这个 representation 空间里预测被 mask 掉的 token。等于让 model 学"世界在抽象层面怎么演化"，把不可预测的细节直接忽略掉。

这跟 MAE（Masked Autoencoder）的差别就在这：MAE 让你重建像素，JEPA 让你重建 representation。听起来好像差不多，实际差远了——representation 是 model 自己学的，它可以选择性编码"重要"的信息，把"不重要"的直接丢掉。这就给 model 一个 freedom：学对下游有用的抽象，而不是忠实重建一切。

代价是：怎么防止 representation collapse？如果 encoder 把所有东西都映射成常数向量，那 predictor 随便猜都能对，loss 一直是 0。LeCun 团队的解法是 BYOL/MoCo 那套：encoder 有个 EMA 版本当 teacher，target 用 teacher 的输出但 stop-gradient，让 student encoder 追 teacher。这个 trick 工作得出奇好，I-JEPA、V-JEPA 1 已经证明了，V-JEPA 2 把它 scale 上去。

---

## Pretraining 具体怎么训

具体公式是这样：

$$
\min_{\theta, \phi, \Delta_y} \| P_\phi(\Delta_y, E_\theta(x)) - \text{sg}(E_{\bar{\theta}}(y)) \|_1
$$

挨个说人话：

- $y$ 是原始视频片段
- $x$ 是 $y$ 把一堆 tubelet（时空小块，大小 $2 \times 16 \times 16$）随机 mask 掉之后的版本
- $E_\theta$ 是 encoder，吃 $x$，吐一组 feature token
- $E_{\bar{\theta}}$ 是 encoder 的 EMA 版本，吃完整 $y$，吐 target feature token
- $\bar{\theta}$ 就是 $\theta$ 的 EMA，每个 step 慢慢追
- $\Delta_y$ 是一个可学习的 mask token，告诉 predictor "这些位置需要你填"
- $P_\phi$ 是 predictor，吃 encoder 输出 + mask token，吐对 mask 位置的预测
- $\text{sg}(\cdot)$ 是 stop-gradient，target 那条路不回传梯度
- $\|\cdot\|_1$ 是 L1 loss

直觉就是：encoder 看到被打马赛克的视频，predictor 试图补全被马赛克盖住的部分，但补全的目标是"teacher encoder 对完整视频提取的 feature"，不是原始像素。Student encoder 和 predictor 一起努力追上 teacher 的输出。Teacher 又在慢慢追 student，形成 bootstrap。

这套机制下，encoder 学到的 feature 自动就是"对预测未来有用的抽象"，因为它只被训练去预测那些被 mask 掉的部分，剩下的细节它管不着。

---

## Scaling 这件事怎么做对

V-JEPA 1 当时用 2M 视频、ViT-H (600M)、16 帧 256×256、训 90K 步。V-JEPA 2 四个维度同时 scale：

**数据**：2M → 22M (VM22M)。混合了 SSv2、Kinetics、HowTo100M、YT-Temporal-1B 和 ImageNet。ImageNet 被复制成 16 帧重复视频混进去，占 25% weight——保证 model 不丢静态外观能力。YT1B 用了基于 DINOv2 embedding 的 cluster-based retrieval 做 curation，从 1.5M cluster 筛到 210K，去掉噪声内容。

**模型**：ViT-L (300M) → ViT-g (1B)。ViT-g 用 1408 width, 40 layers, 22 heads, 6144 MLP dim。Predictor 一直是 ViT-small (22M)，不 scale。

**训练步数**：90K → 252K。学习率 schedule 改成 warmup-constant-cooldown，简化超参调节，而且可以中途 checkpoint 启动多个 cooldown 实验。

**分辨率和时长**：16 帧 256×256 → 64 帧 384×384。这个 scale 最贵——直接训 64×384×384 要 60 GPU-years。他们的 trick 是 **progressive resolution**：前 240K 步用 16 帧 256×256 训，最后 12K 步 cooldown 时把分辨率和帧数拉满，学习率线性衰减。这样高分辨率训练的开销只在最后付，省 8.4 倍 GPU time，下游性能几乎不损失。

四个 scaling 累计 +4.0 points average accuracy across 6 tasks (SSv2, Diving-48, Jester, K400, COIN, IN1K)。

这里有个工程 insight 我想强调：**constant-cooldown schedule 比 cosine schedule 更灵活**。Cosine 你一旦定下来总步数就锁死了，constant 你可以一直训到下游性能 plateau 才开始 cooldown，cooldown 还能从多个 checkpoint 各启一个试不同分辨率。这种工程上的可操作 性对大模型训练特别重要。

---

## 关键跳跃：从 video model 到 robot controller

预训练完的 V-JEPA 2 能预测被 mask 掉的视频片段，但 predictor 没有 action 输入，不知道"如果我做某个动作会怎样"。要变成可用于控制的 world model，得加 action conditioning。

他们 freeze 住 V-JEPA 2 encoder，上面接一个新 predictor，叫 V-JEPA 2-AC。训练数据是 Droid dataset 的 raw 视频，Franka 机械臂遥操作采集的，他们用了不到 62 小时。**关键点：不用标注哪些是成功 trajectory，不用 reward，不用 task label，只用原始视频 + end-effector state 信号**。

输入长这样：
- 视频 $(x_k)_{k \in [16]}$，256×256，4 fps，共 4 秒
- End-effector state $s_k \in \mathbb{R}^7$，前 3 维是 position (xyz)，中 3 维是 orientation (Euler 角)，最后 1 维是 gripper 开合
- Action $a_k \in \mathbb{R}^7$ 定义为 $\Delta s_k = s_{k+1} - s_k$，即相邻帧 end-effector state 的变化量

Action 用 delta 而非 absolute 是个重要设计——planning 时你直接优化"该往哪个方向挪"，不用管当前 absolute pose。

Encoder 把每帧独立编码成 $z_k = E(x_k) \in \mathbb{R}^{16 \times 16 \times 1408}$（空间 16×16 token，每个 1408 维）。然后 $(a_k, s_k, z_k)$ 交错喂给 predictor，predictor 预测 $\hat{z}_{k+1}$。

Loss 两个加起来：

Teacher-forcing loss：
$$
\mathcal{L}_{\text{tf}} = \frac{1}{T} \sum_{k=1}^{T} \| P_\phi((a_t, s_t, E(x_t))_{t \le k}) - E(x_{k+1}) \|_1
$$

每步都用 ground-truth 上一帧作为输入，预测下一帧。

Rollout loss：
$$
\mathcal{L}_{\text{rollout}} = \| P_\phi(a_{1:T}, s_1, z_1) - z_{T+1} \|_1
$$

把 predictor 自己的输出再喂回去，展开 T 步预测远方。实践中 T=2，只展开一步，避免长链梯度不稳定。

总 loss = teacher-forcing + rollout。

**为什么两个都要**：只用 teacher-forcing，inference 时 model 没见过自己的输出当输入，会有 distribution shift。只用 rollout，训练不稳定且太贵。T=2 的 rollout 是个便宜又有效的 scheduled sampling，让 model 适应自己的预测误差累积。

Predictor 架构：300M 参数 transformer，24 层，16 头，1024 hidden，GELU。**Block-causal attention**——某时刻的 token 可以 attend 同时刻所有 token（action、state、patches 互相看），以及之前所有时刻的 token，但不能看未来。这尊重时间因果性。

---

## Planning 怎么做：把 world model 当 oracle 搜动作

这部分是整个 paper 最美的设计，也是 LeCun energy-based model 思想的具体落地。

设当前时刻 $k$，机器人状态 $s_k$，当前帧 $x_k$，目标帧 $x_g$。把 $x_k$ 和 $x_g$ 都过 frozen encoder 得到 $z_k$ 和 $z_g$。要找一个 action sequence $\hat{a}_{1:T}$ 让"想象中"的未来状态尽量接近 goal：

$$
\mathcal{E}(\hat{a}_{1:T}; z_k, s_k, z_g) = \| P(\hat{a}_{1:T}; s_k, z_k) - z_g \|_1
$$

$$
(a_i^\star)_{i \in [T]} = \arg\min_{\hat{a}_{1:T}} \mathcal{E}(\hat{a}_{1:T}; z_k, s_k, z_g)
$$

直觉：给一个 candidate action sequence，让 world model 想象执行之后 representation 长啥样，跟 goal 的 representation 算 L1 距离，距离越小越好。最优 action sequence 就是让想象未来最接近 goal 的那个。

**怎么搜**：用 Cross-Entropy Method (CEM)，一种经典的 zeroth-order 优化方法。流程：

1. 初始化一个高斯分布 $\mathcal{N}(0, I)$ over action sequence
2. 从这个分布采样 N=800 个 candidate trajectories
3. 每个 trajectory 过 world model 算 energy
4. 选 top-10 (elite set)
5. 用 elite set 的均值和方差更新高斯分布参数
6. 重复 10 次 refinement
7. 返回最终高斯分布的均值作为最优 action sequence
8. 只执行第一个 action，然后重新观察、重新 plan（receding horizon）

整套在 RTX 4090 上 16 秒一步。Horizon=1（只搜下一步），因为他们任务都是 relatively greedy 的。

我之所以觉得这个设计漂亮，是因为它把"控制机器人"问题完全转化成"在 representation 空间最小化 energy"问题。没有 policy network，没有 RL，没有 reward function engineering，没有 imitation learning。World model 学好后就是个 oracle，planner 就是搜 action 让 oracle 满意。

---

## 实验结果讲讲故事

### 单目标 reaching

给一张 goal 图，机器人要把 end-effector 挪到那个位置。注意是单目 RGB，没有 depth sensor，model 得隐式学深度。

Figure 9 那个 energy landscape 特别有说服力：固定 $\Delta z = 0$，扫 $\Delta x$ 和 $\Delta y$，energy 在 ground-truth action 附近达到最小，而且 landscape 局部 convex。这说明 model 真的学到了"哪个动作会让我接近 goal"的几何结构，不是死记硬背。CEM 在这种 convex landscape 上效率很高。

三个 reaching 任务，end-effector 都单调逼近 goal，最终误差 < 4cm。这其实是个视觉伺服（visual servoing）问题，但经典视觉伺服要标定相机、要 depth、要 explicit 几何，V-JEPA 2-AC 全是 implicit 学的。

### Pick-and-Place

这个任务更难，需要先抓、再移、再放。他们用三个 sub-goal 分阶段：
- Sub-goal 1（4 步）：抓住物体
- Sub-goal 2（10 步）：移到目标附近
- Sub-goal 3（4 步）：精确放置

结果对比 Octo（VLA baseline）：
- Octo Pick&Place Cup: 10%
- V-JEPA 2-AC Pick&Place Cup: 80%

**8 倍提升**。Octo 在 Open-X Embodiment 1M+ 轨迹上预训练过，还在整个 Droid 上做了 hindsight relabeling fine-tune，参数量也不小。但行为克隆学到的 policy 在新环境就是不如 model-based planning 鲁棒。

对比 Cosmos（NVIDIA latent diffusion video generation world model）：
- Cosmos: 4 分钟/action，Pick&Place Cup 0%, Box 0%
- V-JEPA 2-AC: 16 秒/action，Pick&Place Cup 80%, Box 50%

Cosmos 慢 15 倍，效果还差。原因我前面说了：diffusion 每次采样要跑完整 denoising chain，而且 stochasticity 让 energy landscape 不平滑，CEM 不好搜。JEPA 一次 forward pass 出 representation，deterministic，landscape 平滑。

### Action anticipation on Epic-Kitchens-100

这个不是机器人任务，是预测 1 秒后厨房里人会做啥动作。V-JEPA 2 ViT-g_384 拿 39.7 recall-at-5，比之前 SOTA PlausiVL（8B 参数，专门为这个任务设计的 LLM-based 模型）高 12.1 points，相对提升 44%。

V-JEPA 2 只有 1B 参数，没用语言监督，pure self-supervised video pretraining，就击败了 8B 的 LLM-augmented 模型。这说明对于 short-horizon 动作预测，视觉信号本身已经足够强，language prior 反而可能是干扰。

### Video Question Answering

把 V-JEPA 2 跟 LLM 对齐做视频问答，用 LLaVA 那套 visual instruction tuning。结果在 8B LLM class 上拿下多个 SOTA：

- PerceptionTest: 84.0
- MVP: 44.5  
- TempCompass: 76.9
- TemporalBench: 36.7
- TOMATO: 40.3

特别值得注意：**V-JEPA 2 预训练阶段完全没碰语言**，纯视觉 SSL。传统 wisdom（Tarsier2、LLaVA-NeXT-Video 这类工作）都认为 video encoder 必须在预训练时就接触 image-text pair 才能对齐 LLM。V-JEPA 2 证明这个 wisdom 错了：足够好的 visual representation，加上后期 alignment，照样 SOTA。

这给我一个 intuition：**video encoder 学到的 representation 已经足够丰富，LLM alignment 只是把它们"翻译"成语言可用的形式**。语言监督在 pretraining 阶段是个 shortcut，不是必需品。

---

## 一些没说清的弱点

我给你列几个我看到的真实问题，别被 hype 冲昏头脑：

**Camera position sensitivity 是个 fundamental issue**。Appendix B.4 Figure 16 显示，把相机绕机器人基座转一圈，model 推断出的 action 坐标轴会跟着旋转，误差接近线性。说明 model 没 truly 学到 3D scene geometry，而是 overfit 到训练数据里 camera-robot 的相对 configuration。Droid 数据集里相机位置相对固定，model 学了个 implicit calibration。换 lab 还能 work 一部分是因为相机位置碰巧接近。这个不解决，scale 到更多 environment 会有问题。

**Action representation 绑死 morphology**。Action 定义为 end-effector 7D state delta，换机器人（比如换成 bimanual、换成 mobile manipulator、换成 legged robot）就得重新设计 action space，重训 V-JEPA 2-AC。这跟 RT-2 那种 language-action 对齐的 generality 没法比。

**实验范围窄**。只有 Franka + table-top + rigid object (cup, box)。没测 deformable object、tool use、bimanual、locomotion、in-hand manipulation。80% pick-and-place 是不错，但离"general world model"还远。

**Compute 门槛高**。1M 小时视频训 1B ViT-g，这是 FAIR 级别的资源。学术界难复现。code 开源了 (https://github.com/facebookresearch/vjepa2)，但训练成本是个 barrier。

**Planning 慢**。16 秒/action 对真实机器人控制算慢了，要做 reactive task（比如接抛过来的球）根本来不及。论文里任务都是 quasi-static 的。

**Long horizon 弱**。Autoregressive rollout 误差累积，超过 16 秒就不行。Pick-and-place 已经需要 sub-goal 分解，更复杂任务（比如"做个三明治"）没法直接做。

---

## 我个人的联想和延伸

**Hybrid VLA + World Model**。我觉得未来最可能的路径是 VLA 和 world model 结合：VLA 做快速初始化提供 prior action distribution，world model 做 look-ahead refinement 和 sanity check。π0 已经在 diffusion policy 里加了 action diffusion，加个 world model loss 应该不冲突。OpenVLA + V-JEPA 2-AC 这种组合我觉得一两年内会出现。

**Object-centric JEPA**。当前 V-JEPA 2 的 representation 是 spatial token grid，没有显式 object 分解。Long-horizon planning 难做部分原因就是 object 之间的关系没显式表示。加 slot attention 或 object discovery module，让 representation compositional，应该能 extend horizon。

**Equivariant world model**。Camera sensitivity 问题本质是 model 没 SE(3) equivariance。如果 encoder 学的是 SE(3) equivariant feature，world model 对 camera 变换自然鲁棒。这个方向已经有早期工作（ClimaX、Equivariant JEPA 之类），跟 V-JEPA 2 结合应该有意思。

**Active perception**。当前 model 是 passive observer。如果让 robot 主动调整相机位置获得更好 observability，相当于 world model + active sensing 联合优化。这跟 active learning in vision 有联系。

**Counterfactual reasoning**。JEPA 这种 predictive architecture 天然支持 "what if" reasoning：给定当前状态，比较两个 action 的预测后果。这可以用于 safety reasoning（"如果我执行 A，会不会撞到人？"）。给 V-JEPA 2-AC 加个 contrastive action 评估 head，应该能做 safety filter。

**Hierarchical world model**。作者 future work 提到了。我的想法是：用 V-JEPA 2 当 low-level controller（秒级 prediction），上面叠一个 abstract world model 做 minute-level planning，再上面叠 LLM 做 task-level reasoning。三层不同时间尺度的 world model，类似 Schmidhuber 1990s 的 hierarchical RL 思想。

**Language-conditioned V-JEPA 2-AC**。Section 7 已经把 V-JEPA 2 对齐了 LLM 做 VidQA，自然延伸是对齐 LLM 给 V-JEPA 2-AC 提供 language goal。"把那个红色杯子放到左上角"这种 instruction，LLM 把它翻译成 sub-goal images 或 sub-goal representation，喂给 planner。这就打通了 language to action 的链路，而且保留 world model 的 planning 能力。

**JEPA for audio / multimodal**。V-JEPA 2 证明 JEPA 在 video 上 scale work。自然延伸是 A-JEPA (audio)、T-JEPA (tactile)、multimodal JEPA。LeCun 路线如果走通，下一轮 multimodal pretraining 范式可能就长这样，而不是 contrastive。

**JEPA 跟 Diffusion 的关系**。Diffusion 在 pixel/latent space 工作，JEPA 在 representation space 工作。最近有些工作（Latent Consistency Models、Consistency Trajectory Models）尝试让 diffusion 一步生成，缩小跟 JEPA 的速度差距。但本质上 diffusion 的 stochasticity 对 control 不友好。我赌 JEPA 路线在 robot 上会赢，diffusion 在 content generation 上会赢，两者井水不犯河水。

---

## 一句话总结

V-JEPA 2 是 LeCun "world model via observation" 哲学的**第一个**真正 convincing demonstration：纯 self-supervised video pretraining + 极少量无标签 robot data，zero-shot 真实机器人控制。它没打败所有 baselines，没解决所有问题，但证明了这条路线可行且 scalable。如果 future work 把 horizon 拉长、把 language goal 加进来、把 morphology generalization 解决，这可能是通向 general robot AI 的一条主干道。

我个人觉得这篇 paper 的历史地位可能类似 BERT for NLP 或者 ViT for vision：一个 paradigm shift 的开头，后续会有大量 follow-up 把它推到更大规模、更强能力。两三年后回头看，V-JEPA 2 可能是 "JEPA 时代" 的起点。

---

参考链接汇总：
- V-JEPA 2 GitHub: https://github.com/facebookresearch/vjepa2  
- V-JEPA 2 Blog: https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks  
- LeCun 2022 position paper: https://openreview.net/pdf?id=BZ5a1r-kVsf  
- V-JEPA 1: https://arxiv.org/abs/2404.08471  
- I-JEPA: https://arxiv.org/abs/2301.08243  
- Droid dataset: https://arxiv.org/abs/2403.12945  
- Cosmos: https://arxiv.org/abs/2501.03575  
- Octo: https://arxiv.org/abs/2405.12213  
- DINO-WM: https://arxiv.org/abs/2411.04983  
- PerceptionLM: https://arxiv.org/abs/2504.13180  
- RT-2: https://arxiv.org/abs/2307.15818  
- OpenVLA: https://arxiv.org/abs/2406.09246  
- π0: https://arxiv.org/abs/2410.24164  
- Gr00t N1: https://arxiv.org/abs/2503.14734  
- TD-MPC2: https://arxiv.org/abs/2310.16828  
- Dreamer: https://arxiv.org/abs/1912.01603  
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122  
- VideoMAE v2: https://arxiv.org/abs/2303.12028  
- DINOv2: https://arxiv.org/abs/2304.07193  
- SigLIP2: https://arxiv.org/abs/2502.14786  
- Perception Encoder: https://arxiv.org/abs/2504.13181  
- Epic-Kitchens-100: https://doi.org/10.1007/s11263-021-01531-2

---

# V-JEPA 2 深度解析

Andrej，这篇 paper 是 LeCun 团队 (FAIR at Meta + Mila) 沿着 JEPA (Joint-Embedding Predictive Architecture) 路线的最新里程碑，把 LeCun 多年来鼓吹的 "world model via observation" 思想真正落地到 robot manipulation。我把核心 intuition 和技术细节拆给你看。

---

## 1. Big Picture: 为什么这件事重要

LeCun 2022 年的 position paper "A Path Towards Autonomous Machine Intelligence" 提出 JEPA 范式作为 alternative to generative models 和 contrastive methods。核心论点: world model 应该在 **learned representation space** 做预测，NOT in pixel space，因为 pixel-level prediction 会浪费 capacity 在 unpredictable details 上 (e.g., 草地每片叶子的精确位置)。

V-JEPA 2 的贡献在于: 把这个哲学思想 scale 到 1M+ hours internet video + 62 hours robot data，并**第一次**展示一个 self-supervised video model 能 zero-shot control Franka arm 做 pick-and-place。这跟 RT-2、π0、OpenVLA、Gr00t N1 这类 VLA (Vision-Language-Action) 路线有根本分歧——那些方法靠 behavior cloning + internet-scale VLM pretraining，V-JEPA 2 走的是 **model-based planning** 路线。

参考:
- V-JEPA 2 GitHub: https://github.com/facebookresearch/vjepa2
- V-JEPA 2 Blog: https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks
- LeCun 2022 position paper: https://openreview.net/pdf?id=BZ5a1r-kVsf
- I-JEPA (前置工作): https://arxiv.org/abs/2301.08243
- V-JEPA (前置工作): https://arxiv.org/abs/2404.08471

---

## 2. V-JEPA 2 Pretraining: Mask-Denoising in Representation Space

### 2.1 核心 Objective

公式 (1) 是整个 paper 的灵魂:

$$
\underset{\theta, \phi, \Delta_y}{\text{minimize}} \;\; \| P_\phi(\Delta_y, E_\theta(x)) - \text{sg}(E_{\bar{\theta}}(y)) \|_1
$$

变量逐一解释:
- $E_\theta(\cdot)$: encoder, 参数 $\theta$, ViT 架构 (input: masked video)
- $P_\phi(\cdot)$: predictor, 参数 $\phi$, ViT-small 架构
- $x$: masked view of video (一些 tubelets 被 drop 掉)
- $y$: 原始完整 video
- $\Delta_y$: learnable mask token, 告诉 predictor 哪些位置需要预测
- $\bar{\theta}$: encoder 权重的 EMA (exponential moving average), 即 teacher network
- $\text{sg}(\cdot)$: stop-gradient, 阻止梯度流过 target branch
- $\|\cdot\|_1$: L1 loss (L1 比 L2 在 representation space 更 robust to outliers)

**关键 intuition**: 这是 non-contrastive SSL，防止 representation collapse 靠两个机制:
1. **EMA teacher**: target branch 用 encoder 的 EMA，类似 BYOL/MoCo 的思路
2. **Stop-gradient**: 不让 predictor 学到的信息"反噬" encoder，迫使 encoder 自己学有用 features

### 2.2 架构细节

- **3D-RoPE** (Rotary Position Embedding): 把 feature dimension 切成 3 段 (~相等), 分别对应 temporal, height, width 三个 axis。比 absolute sincos position embedding 在 ViT-g 这种大模型上 training 更稳定。这是从 V-JEPA 1 到 V-JEPA 2 的一个关键改动。3D-RoPE 论文: https://arxiv.org/abs/2104.09864 (RoFormer)
- **Tubelet size**: $2 \times 16 \times 16$ (T×H×W), 即每个 token 代表 2 帧 × 16×16 像素
- **Multi-block masking**: 沿用 V-JEPA 1 的策略, 用多个不同尺度的 rectangular masks

### 2.3 Scaling Ingredients (Figure 3 的核心数据)

四个 scaling 维度的 ablation (在 ViT-L/16 上做):
| Intervention | Avg Accuracy Gain |
|---|---|
| Data: VM2M → VM22M | +1.0 |
| Model: ViT-L (300M) → ViT-g (1B) | +1.5 |
| Training: 90K → 252K iters | +0.8 |
| Resolution: 256 → 384, 16 → 64 frames | +0.7 (训练时) +更多 eval 时 |

累计 +4.0 points over baseline。

### 2.4 Pretraining Dataset (VM22M)

Table 1 数据:
| Source | Samples | Type | Hours | Weight |
|---|---|---|---|---|
| SSv2 | 168K | EgoVideo | 168 | 0.056 |
| Kinetics | 733K | ExoVideo | 614 | 0.188 |
| HowTo100M | 1.1M | ExoVideo | 134K | 0.318 |
| YT-Temporal-1B | 19M | ExoVideo | 1.6M | 0.188 |
| ImageNet | 1M | Images | n/a | 0.250 |

注意 ImageNet 占 0.25 weight 是个重要设计——images 被复制成 16 帧重复视频。这保证 model 不丢失 static appearance 能力 (在 ImageNet probe 上重要)。

**Data Curation** 对 YT1B 用 cluster-based retrieval: 用 DINOv2 ViT-L 抽 YT1B 每个场景中间帧的 embedding, 聚成 1.5M 个 clusters, 再用 Kinetics/SSv2/COIN/EpicKitchen 训练集做 target distribution 选 210K 个 clusters, 最终保留 115M scenes。Reference: DINOv2 curation pipeline https://arxiv.org/abs/2304.07193

### 2.5 Progressive-Resolution Training (Figure 5)

这是 engineering 亮点。直接在 ViT-g 上训 64×384×384 需要 ~60 GPU-years。Solution:

```
Phase 1 (warmup):    12K iters,  16 frames, 256×256, LR linear warmup
Phase 2 (constant): 228K iters,  16 frames, 256×256, LR constant
Phase 3 (cooldown):  12K iters,  64 frames, 384×384, LR linear decay
```

8.4× speedup, 几乎不损失 downstream 性能。这思路来自 Touvron et al. 2019 (Fixing train-test resolution discrepancy) https://arxiv.org/abs/1906.06423 和 DINOv2 的 progressive training。

---

## 3. V-JEPA 2-AC: Action-Conditioned World Model

这是 paper 最 exciting 的部分。Stage 1 学到的 representation 怎么变成可用于 control 的 world model?

### 3.1 Inputs 格式

从 Droid dataset (https://arxiv.org/abs/2403.12945) sample 4-second clips:
- **Video**: $(x_k)_{k \in [16]}$, 256×256, 4 fps
- **End-effector state**: $s_k \in \mathbb{R}^7$ — 3 dim position (xyz) + 3 dim orientation (Euler angles extrinsic) + 1 dim gripper state
- **Actions**: $a_k \in \mathbb{R}^7$ defined as $\Delta s_k = s_{k+1} - s_k$ — 即相邻帧 end-effector state 的变化量

**关键设计**: action 是 relative (delta), 不是 absolute。这让 model 学到的是 "given current state + delta command, what's next state representation"——这种形式更便于 planning 时的 energy minimization。

### 3.2 Loss Function

公式 (2): Teacher-forcing loss
$$
\mathcal{L}_{\text{teacher-forcing}}(\phi) := \frac{1}{T} \sum_{k=1}^{T} \| \hat{z}_{k+1} - z_{k+1} \|_1 = \frac{1}{T} \sum_{k=1}^{T} \| P_\phi((a_t, s_t, E(x_t))_{t \le k}) - E(x_{k+1}) \|_1
$$

变量:
- $T = 15$: 时间步数
- $\hat{z}_{k+1}$: predictor 预测的下一帧 representation
- $z_{k+1} = E(x_{k+1})$: ground-truth 下一帧的 frozen encoder output
- $P_\phi$: 300M 参数 transformer predictor

公式 (3): Rollout loss
$$
\mathcal{L}_{\text{rollout}}(\phi) := \| P_\phi(a_{1:T}, s_1, z_1) - z_{T+1} \|_1
$$

变量:
- $a_{1:T}$: 完整 action sequence
- $s_1, z_1$: 初始状态
- $z_{T+1}$: T+1 时刻 ground-truth representation
- 实践中 $T = 2$, 只展开一步 (避免高 computational cost 和 gradient instability)

公式 (4): Total loss = teacher-forcing + rollout。

**为什么需要 rollout loss**: Teacher-forcing 让 model 学 step-by-step 预测, 但 inference 时需要 autoregressive rollout。Rollout loss 用 predictor 自己的输出作为下一步 input, 减少训练/inference mismatch (类似 scheduled sampling)。T=2 是 trade-off: 太大会让 gradient path 太长 + error accumulation, 太小不能缓解 distribution shift。

### 3.3 Predictor 架构

- 300M params, 24 layers, 16 heads, 1024 hidden dim, GELU
- **Block-causal attention**: 每个时刻的 patch token 可以 attend 同时刻的 action/state/patches, 以及所有之前时刻的内容。这设计是为了尊重时间因果性。
- Action 和 state token 只施加 temporal RoPE (没有 spatial component, 因为它们不是空间 patch)
- 输入: action, state, frame features 各自经过 learnable affine transform 映射到 hidden dim

### 3.4 Planning via Energy Minimization

公式 (5):
$$
\mathcal{E}(\hat{a}_{1:T}; z_k, s_k, z_g) := \| P(\hat{a}_{1:T}; s_k, z_k) - z_g \|_1
$$
$$
(a_i^\star)_{i \in [T]} := \underset{\hat{a}_{1:T}}{\text{argmin}} \; \mathcal{E}(\hat{a}_{1:T}; z_k, s_k, z_g)
$$

变量:
- $\hat{a}_{1:T}$: candidate action trajectory
- $z_k$: 当前帧的 representation
- $s_k$: 当前 end-effector state
- $z_g$: goal image 的 representation
- $a_i^\star$: 最优 action sequence

**Intuition**: 在 representation space 中, 让 model "想象"执行某 action sequence 后的 future state, 跟 goal state 算 L1 distance, 选 distance 最小的 action。这就是 LeCun 2022 提出的 energy-based model 思路的具体实现。

**Optimization**: Cross-Entropy Method (CEM, Rubinstein 1997):
1. Sample action trajectories from $\mathcal{N}(0, I)$
2. 用 energy function 评估每个 trajectory
3. 选 top-k (实践中 top 10) 用来更新 mean 和 variance
4. Iterative refinement, 最后返回 mean
- 参数: 800 samples, 10 refinement steps, horizon=1
- 每个 action 16 秒 on RTX 4090

**Receding Horizon Control**: 只执行 first action, 然后重新观察 + 重新 plan。

---

## 4. Zero-Shot Robot Experiments (Tables 2, 3)

部署到 Franka Emika Panda + RobotiQ gripper, 两个不同 lab, 没有从这些 lab 收集任何数据。

### 4.1 Single-Goal Reaching (Figure 8, 9)
Figure 9 的 energy landscape 是 beautiful intuition builder: 在 $\Delta x, \Delta y$ 平面上扫, 能量函数在 ground-truth action 附近达到 minimum, 而且 landscape 局部 convex——这意味着 CEM 可以高效找到最优 action, 即使没有精确 calibration。

Figure 8 三个 reaching tasks 中, end-effector 都能单调逼近 goal, 最终误差 <4cm。注意: 这是 **monocular RGB**, 没有 depth sensor, model 必须隐式学 depth。

### 4.2 Pick-and-Place (Table 2, 3)

| Method | Lab | Reach | Grasp Cup | Grasp Box | Reach w/ Cup | Reach w/ Box | P&P Cup | P&P Box |
|---|---|---|---|---|---|---|---|---|
| Octo | 1 | 100% | 20% | 0% | 20% | 70% | 20% | 10% |
| Octo | 2 | 100% | 10% | 0% | 10% | 70% | 10% | 10% |
| **V-JEPA 2-AC** | 1 | 100% | 70% | 30% | 90% | 80% | 80% | 80% |
| **V-JEPA 2-AC** | 2 | 100% | 60% | 20% | 60% | 70% | 80% | 50% |

V-JEPA 2-AC 在 P&P Cup 上达 80% (avg), Octo 只有 10%, **8 倍**提升。

Table 3 Planning Performance 对比 Cosmos (NVIDIA 的 latent diffusion video generation model, https://arxiv.org/abs/2501.03575):
- Cosmos: 80 samples, 10 iter, horizon=1, **4 min/action**, Pick&Place Cup 0%, Box 0%
- V-JEPA 2-AC: 800 samples, 10 iter, horizon=1, **16 sec/action**, Pick&Place Cup 80%, Box 50%

**关键 insight**: Video generation models (Cosmos) 在 latent space 内生成 pixel-level consistent 未来帧, 但 (1) 太慢, (2) energy landscape 不平滑 (diffusion 的 stochasticity 导致), (3) 训练 objective 优化 perceptual quality 而非 planning usefulness。JEPA 完全避开这些问题。

参考 Cosmos: https://arxiv.org/abs/2501.03575
参考 Octo: https://arxiv.org/abs/2405.12213

### 4.3 Sub-goal 分解

Pick-and-Place 用 3 个 sub-goals:
- Sub-goal 1 (4 timesteps): 抓住物体
- Sub-goal 2 (10 timesteps): 移到目标位置附近
- Sub-goal 3 (4 timesteps): 精确放置

这是 hierarchical planning 的简单 version。作者在 future work 里明确说要扩展到更长的 horizon, 不需要 sub-goals。

### 4.4 Limitations

1. **Camera position sensitivity** (Appendix B.4, Figure 16): 由于训练时 action 是相对于 robot base 的, 但 camera 看不到 robot base, model 必须 implicit infer action coordinate axis from RGB。误差几乎是 systematic rotation, 可以 unsupervised 校准: 让 robot 做随机动作, 比较 inferred action 和实际 action, 解出 2×2 旋转矩阵, 之后 planning 时乘上这个矩阵。

2. **Long-horizon error accumulation**: Autoregressive rollout 误差累积, 限制 plan horizon。

3. **Image goals**: 需要 visual goal, 不能用 language。作者明确说未来要 align with LLM (Section 7 已经做了 VidQA 的 alignment, 是 starting point)。

---

## 5. Understanding: Probe-based Classification (Table 4)

Frozen encoder + 4-layer attentive probe (3 self-attention + 1 cross-attention with learnable query token)。

**关键结果**: V-JEPA 2 ViT-g_384 average 88.2 across 6 tasks (SSv2, Diving-48, Jester, K400, COIN, IN1K), 在 motion understanding 上 SOTA:
- SSv2: 77.3 (vs InternVideo2-6B 67.7, vs PE_core_G 55.4)
- Diving-48: 90.2
- Jester: 97.8

但 appearance 上还是弱于 image-text contrastive 模型:
- ImageNet: 85.1 (vs PE_core_G 87.6)
- K400: 87.3 (vs SigLIP2 88.0)

**Intuition**: Self-supervised video pretraining 没有语言监督, 但能学到强 motion representation。Appearance 上不如 contrastive 是因为缺乏 semantic supervision。但 Table 6/8 显示, 跟 LLM 对齐后, 这个 gap 几乎消失。

参考 SigLIP2: https://arxiv.org/abs/2502.14786
参考 Perception Encoder (PE): https://arxiv.org/abs/2504.13181

---

## 6. Prediction: Action Anticipation (Table 5)

Epic-Kitchens-100 action anticipation, 用 attentive probe + predictor output 联合作为 input (Appendix D.1, Table 20)。

| Method | Params | Verb | Noun | Action |
|---|---|---|---|---|
| PlausiVL | 8B | 55.6 | 54.2 | 27.6 |
| V-JEPA 2 ViT-L | 300M | 57.8 | 53.8 | 32.7 |
| V-JEPA 2 ViT-g_384 | 1B | 63.6 | 57.1 | **39.7** |

V-JEPA 2 ViT-g_384 比 PlausiVL (8B params, 专门为这个 task 设计) 提升 12.1 points (+44% relative), 用更小 8 倍的 model。

**Scaling 表现**: Linear scaling w.r.t. model size (300M → 600M → 1B)。这与 LLM scaling laws 的精神一致。

Appendix D.2 Figure 18 显示 longer anticipation time (1s → 10s) 性能急剧下降, 这是 expected 因为 EK100 是 non-deterministic。

参考 EK100: https://doi.org/10.1007/s11263-021-01531-2

---

## 7. Video Question Answering: Aligning with LLM (Tables 6, 7, 8)

这是 paper 中另一个反 conventional wisdom 的发现: **video encoder pretrained WITHOUT language supervision** 可以被 align to LLM 后达到 SOTA。

### 7.1 Controlled Comparison (Table 6)

Frozen encoder + Qwen2-7B-Instruct + 18M alignment samples:

| Encoder | Avg | PerTest | MVP | TempCompass | TemporalBench | TVBench |
|---|---|---|---|---|---|---|
| DINOv2 ViT-g518 | 45.7 | 67.1 | 22.4 | 62.3 | 26.8 | 47.6 |
| SigLIP2 ViT-g384 | 48.1 | 72.4 | 26.2 | 66.8 | 25.7 | 48.7 |
| PE ViT-G/14 448 | 49.1 | 72.3 | 26.7 | 67.0 | 27.5 | 51.6 |
| **V-JEPA 2 ViT-g512** | **52.3** | 72.0 | **31.1** | **69.2** | **33.3** | **55.9** |

V-JEPA 2 在 temporal understanding (MVP, TemporalBench, TVBench) 上明显胜出, 在 PerceptionTest (更多 semantic) 上略输 PE 和 SigLIP。

### 7.2 Scaling Vision Encoder (Table 7)

End-to-end (unfrozen encoder) 训练, scaling 300M → 1B, 256 → 512 resolution:
- PerTest: 74.6 → 77.7 (+3.1)
- TVBench: 50.9 → 57.5 (+6.6)

确认 vision encoder scaling + resolution scaling 都有效。

### 7.3 Scaling Data (Table 8) — SOTA on 8B LLM class

V-JEPA 2 ViT-g384 + Llama 3.1 8B, 用 88.5M alignment samples:

| Benchmark | PLM 8B (prev SOTA) | V-JEPA 2 | Gain |
|---|---|---|---|
| PerceptionTest | 82.7 | **84.0** | +1.3 |
| MVP | 39.7 | **44.5** | +4.8 |
| TempCompass | 72.7 | **76.9** | +4.2 |
| TemporalBench | 28.3 | **36.7** | +8.4 |
| TOMATO | 33.2 | **40.3** | +7.1 |

参考 PerceptionLM (PLM 8B): https://arxiv.org/abs/2504.13180
参考 Qwen2.5-VL: https://arxiv.org/abs/2502.13923
参考 InternVL 2.5: https://arxiv.org/abs/2412.05271

---

## 8. 与 Related Work 的关系: 我的 Intuition

### 8.1 vs Dreamer (Hafner et al.)
Dreamer (https://arxiv.org/abs/1912.01603) 和 TD-MPC2 (https://arxiv.org/abs/2310.16828) 也是 latent world model + planning, 但关键差异:
- Dreamer 在 pixel space reconstruct (RSSM), V-JEPA 2 完全抛弃 pixel prediction
- Dreamer 从 interaction 学, V-JEPA 2 主要从 observation 学, interaction 只占 62 小时
- Dreamer 在 single environment, V-JEPA 2 zero-shot 到新 lab

### 8.2 vs VLA Models (RT-2, OpenVLA, π0, Gr00t N1)
这些方法 (https://arxiv.org/abs/2307.15818 RT-2, https://arxiv.org/abs/2406.09246 OpenVLA, https://arxiv.org/abs/2410.24164 π0, https://arxiv.org/abs/2503.14734 Gr00t N1) 都是 behavior cloning + VLM pretraining, 直接学 policy。V-JEPA 2 走 model-based planning:
- 优点: 可以用失败 trajectory (no reward needed), 可以处理新任务不用 fine-tune, 推理时可以"思考"
- 缺点: planning 慢 (16s/action), short horizon

这两条路线未来很可能 hybrid: 用 VLA 做 fast initialization + world model 做 refinement / look-ahead。

### 8.3 vs DINO-WM (Zhou et al. 2024)
DINO-WM (https://arxiv.org/abs/2411.04983) 思想非常类似: 用 DINO pretrained features 做 latent world model, 然后用于 planning。但 V-JEPA 2:
- 用 self-supervised video pretraining 而非 image SSL (DINO)
- 训练 1B 参数 model, DINO-WM 用更小 model
- 实证 zero-shot 真实 robot pick-and-place

DINO-WM 是这条思想的小规模 precursor, V-JEPA 2 是 scaled-up version。

### 8.4 vs Cosmos (NVIDIA)
Cosmos (https://arxiv.org/abs/2501.03575) 是 latent diffusion video generation model, 也包含 action-conditioned 版本用于 robot。但 Table 3 显示:
- Cosmos 4 min/action vs V-JEPA 2-AC 16 sec/action (15× slower)
- Cosmos Pick&Place Cup 0% vs 80%

**根本原因**: Diffusion 的 stochasticity 让 energy landscape 不平滑, 而且每 sample 都要 run full denoising 过程。JEPA 一次 forward pass 出 representation, 而且是 deterministic。

### 8.5 vs World Models (Ha & Schmidhuber 2018)
经典 World Models paper (https://arxiv.org/abs/1803.10122) 用 VAE + MDN-RNN + Controller, 在 RL setting 学习。V-JEPA 2 是 observation-based, 不需要 reward, 不需要 RL agent 在 environment 中探索。

### 8.6 vs VideoMAE v2
VideoMAE v2 (https://arxiv.org/abs/2303.12028) 也是 video SSL, 但用 pixel reconstruction。V-JEPA 2 在 representation space reconstruct, 抛弃 pixel。SSv2 上 V-JEPA 2 (77.3) vs VideoMAE v2 (56.1), 提升 21 points。

---

## 9. Personal Take: 这篇 Paper 的 Historical Significance

我认为这是 LeCun 路线的**第一个**真正 convincing demonstration。之前的 I-JEPA / V-JEPA 1 只在 classification/probe 上有结果, 跟 MAE/contrastive 差距不足以说服 community。V-JEPA 2 同时在三个层面 push:
1. **Scale**: 1M hours video + 1B params
2. **Capability**: zero-shot real robot manipulation
3. **Cross-modal**: 与 LLM 对齐做 VidQA 也 SOTA

特别是第 3 点很有意思。传统 wisdom (e.g., Yuan et al. 2025 "Tarsier2", Li et al. 2024 LLaVA-NeXT-Video) 认为 video encoder 需要在 pretraining 阶段就接触 language 才能用于 VQA。V-JEPA 2 证明 pure SSL video pretraining + 后期 alignment 也能 SOTA, 这暗示: **video encoder 学到的 representation 已经足够丰富, LLM alignment 只是把它们"翻译"成 language**。

另外, 我注意到一个值得玩味的细节: **V-JEPA 2 没用 language supervision, 但在 EK100 action anticipation 上击败了 PlausiVL (8B, 用 LLM)**。这暗示: 对于 short-horizon 动作预测, 视觉 signal 比 language prior 更 informative。

---

## 10. Open Questions / Future Work 作者提到

1. **Long-horizon planning**: 16 秒上限。需要 hierarchical world model (multi-scale spatial + temporal abstraction)。这让人想起 Schmidhuber 1990s 的 hierarchical RL work。

2. **Language goal**: 现在 plan target 是 image, 未来要 language → V-JEPA 2-AC representation space。Section 7 的 V-JEPA 2 + LLM alignment 是 starting point, 但需要更紧密耦合。

3. **Scaling to 20B**: 当前 1B 还远未 saturate。需要 engineering breakthrough 让 ViT-g → ViT-20B feasible for video。

我额外想到的方向 (hallucination 范围):
- **Action regularization**: 当前 predictor 学的是 free-form, 可以加 action smoothness constraint (沿运动学轨迹正则化) 来减少 planning 时的 jittery actions。
- **3D world model**: 加 depth 估计 / multi-view consistency, 可能解决 camera position sensitivity 问题。
- **Active perception**: 让 robot 主动调整 camera position 来获得 better observability。
- **Counterfactual planning**: 用 JEPA 做 "what if" reasoning, "如果我执行 action A 而非 B, 会怎样?"
- **Object-centric representations**: Slot attention 类方法 + V-JEPA, 让 representation 更 compositional, 可能改善 long-horizon reasoning。
- **Symmetry-based world model**: 用 equivariant representation (e.g., SE(3) equivariant) 让 model 通用化到不同 camera 角度。

---

## 11. 一些 Critical 评估

最后我说几个我看到的 weaknesses, 帮你建立 balanced intuition:

1. **Robot experiments scope**: 只有 Franka + table-top manipulation, 没测 locomotion, 没测 bimanual, 没测 deformable object, 没测 tool use。Pick-and-place 80% 是不错, 但远非 "general world model" 水平。

2. **Action representation**: action 定义为 end-effector delta, 限制到特定 robot morphology。换 robot 需要重新设计 action space 或重新 collect Droid-like data。

3. **Camera sensitivity** 是个 fundamental issue, 而非 engineering detail。这说明 model 没 truly 学到 3D scene geometry, 而是 overfit 到训练数据中 camera-robot 的相对 configuration。

4. **VidQA results 不是全方位 dominate**: TVBench 和 MVBench 上 PLM 8B 仍胜出 (Table 8)。这说明对于某些需要 fine-grained semantic reasoning 的任务, 语言监督在 pretraining 阶段还是 helpful。

5. **Compute**: 用 1M hours video 训 1B ViT-g, 这是 FAIR-level compute, 学术界难以 reproduce。开源 code 在 https://github.com/facebookresearch/vjepa2, 但训练成本是个 barrier。

6. **Baselines 不完全公平**: Octo 在 Open-X Embodiment (1M+ trajectories) 上 pretrain, V-JEPA 2-AC 在 Droid (62 hours) 上 fine-tune。Droid 在 Octo 训练集内, 但 Octo fine-tune 时也用了 hindsight relabeling, 这算 reasonable。Cosmos 也是用 action-conditioned fine-tune。Comparison 还算 fair, 但 V-JEPA 2 用了 1M hours internet video pretrain, 这个 prior knowledge 量级是其他 baseline 没有的。

---

希望这给你 build 起完整的 intuition, Andrej。如果想深入某个 part (e.g., progressive training 的数学, CEM 的具体 algorithm, block-causal attention 的实现), 我可以再展开。
