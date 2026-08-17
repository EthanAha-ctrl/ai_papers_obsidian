---
source_pdf: Embodied AI Agents Modeling the World.pdf
paper_sha256: 8462537aec222d9038396034b116223511ef8b4e39d1a0e61a0187f5e09dd190
processed_at: '2026-08-04T03:28:29-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇 paper

## 一句话版本

Meta 在说：** 现在的 LLM/VLM 用来做 embodied agent 是不对路的，它们是 "会编故事" 的模型，不是 "会思考" 的模型；真正该做的是在 abstract latent space 里学一个 world model，只预测关键信息、不重建无关细节，这样 agent 才能真正 plan 而不是 hallucinate。**

---

## 为什么要写这篇 paper

Meta 在 LLM 这波落后了，但他们在 ** 别的东西上堆了很多料 **：眼镜、avatar、机器人、touch sensor、4D video 数据集、实时语音对话模型... 这些东西散落在十几个 project 里，外界看不清 Meta 到底在干嘛。

这篇 paper 就是把所有这些料用 ** "world model" ** 这根线串起来，告诉外界：** 我们不是在追 LLM，我们在搭下一波——embodied AI 的地基。**

作者名单里有 Pascale Fung（对话 AI 老炮）、Yann LeCun（JEPA 那条线的灵魂人物）、Jitendra Malik（vision 界祖师爷）——这阵容本身就是一个信号：Meta 觉得 embodied + world model 是个 ** 足够大的赌注 **。

---

## 核心观点：为什么 LLM 当 agent 不行

你拿 GPT-4 去做 agent，让它规划 "帮我做顿饭"，它会编出一套看起来合理的步骤。** 但它是 "编" 出来的，不是 "想" 出来的。**

LLM 的训练目标是 next-token prediction。这个目标让模型变得 ** 极擅长 "续写" **——给它一个开头，它能顺着语料的统计分布接下去。做饭步骤的语料网上很多，所以它接得很顺。但这只是 ** 表面 pattern match **，它并没有在脑子里真的 "模拟" 这个厨房、这个锅、这把刀。

打个比方：你让一个背过无数菜谱但 ** 从没进过厨房 ** 的人指导你做菜，他能背出一堆步骤，但遇到 "锅有点糊了怎么办" 这种 ** 状态偏离 ** 的情况就抓瞎，因为他脑子里没有 "world dynamics"——他只会背菜谱，不会 simulate 厨房。

Diffusion video model 也类似。你让它生成 "切洋葱" 的视频，它能生成很逼真的画面，但它是 ** 在 pixel 层面重建 **，把光照、纹理、背景全建模了，** 唯独没有在 abstract 层面建模 "刀 → 洋葱 → 两半" 这个 causal transition**。pixel-level 细节对 planning 毫无用处，反而淹没了真正重要的东西。

所以 Meta 的主张是：** 别重建 pixel，别生成 text，在 latent space 里只预测 "下一步 world state 的 abstract representation"。** 这就是 JEPA。

---

## JEPA 到底是什么，用大白话

传统生成模型（GPT、Sora）做的是：** 输入 → 重建输出 **。

JEPA 做的是：** 输入 → encoder → latent → 在 latent 里预测下一步的 latent **。

它 ** 根本不生成 pixel 或 text**，只在 "压缩后的 abstract 空间" 里 forward roll。

打个比方：
- ** 生成模型 ** 像画家，你问他明天会发生什么，他给你画一张明天的画——画得很细，但画里 99% 的细节（墙的颜色、地上的阴影）对决策没用，还可能画错。
- ** JEPA ** 像一个只记要点的人，你问他明天会发生什么，他说 "刀会切下去、洋葱会变两半、眼泪会流"——不画细节，只记 ** 因果骨架 **。

这个 latent space 是被训练出来的，** 只保留对 action causal 的信息 **，扔掉所有与 action 无关的 nuisance（光照、视角、纹理）。

为什么这对 planning 重要？因为 plan 就是 "我在脑子里 rollout 一串 action，看看 world 会变成什么样，挑 cost 最小的那串执行"。如果你在 pixel 空间 rollout，每一帧几百万维，算不动；如果在 latent 空间 rollout，几百到几千维，** 算得动，而且 rollout 出来的 trajectory 是 "因果正确" 的 **，因为 latent space 只编码 causal 信息。

---

## 论文给的 planning 公式，说人话

给定当前观察、goal 图、一个 encoder、一个 predictor：

```
在 latent 空间里 rollout T 步：
    s_0 = encoder(current observation)
    s_1 = predictor(s_0, a_1)
    s_2 = predictor(s_1, a_2)
    ...
    s_T = predictor(s_{T-1}, a_T)

目标：选 a_1, ..., a_T 让 s_T 离 s_goal 尽量近
    min sum_k || s_k - s_goal ||_1
```

就是 ** 在 latent 空间里算 "我走 T 步之后离 goal 还有多远"，挑最近的 action 序列 **。

为啥用 $L_1$ 距离不用 $L_2$？$L_1$ 对 outlier 鲁棒，latent space 里某些维度可能是 sparse 的，$L_2$ 会被少数大值 dominate。

为啥用 CEM（cross-entropy method）优化？因为 action space 是连续的，predictor 在长 horizon 上可能不稳，CEM 是 ** sample-based 黑盒优化 **，不需要 gradient，对 predictor 不光滑也 robust。

---

## 三类 agent，用大白话

### Virtual Embodied Agent（虚拟人）

就是 VR 里的 avatar、metaverse 里的 NPC、AI studio 里的数字人。

它的 "body" 是 mesh + motion model，它的 "action" 是 ** 脸上的表情、手上的手势、身体的姿态 **。

关键难题：** 别人说话时你怎么办？** 现在的 avatar 都只会 "自己说"，不会 "听别人说时做反应"。但真人对话里，listener 会点头、会 "嗯嗯"、会笑——这些 listener behavior 占了对话的一半。

Meta 的 Seamless 项目搞了 4000 小时的双人对谈数据，训练了一个 diffusion transformer，输入双方语音 + 可选用户视觉，输出脸上的 Imitator feature + 身体 SMPL-H 参数。它能 ** 模仿笑、同步点头、turn-taking 时切换姿态 **。

为啥这事难？因为 listener behavior 不是 "无动作"，是 ** 极其 subtle 的 micro-response**。你要同步、要 contextual、要 natural，但又不能过度——点头点太频繁就 creepy 了。

### Wearable Agent（眼镜这类）

眼镜戴在你头上，** 它看到的就是你看到的 **，听到的是你听到的。这是 ** shared perception **——你和 AI 共享感官。

它要做的不是替你做事，是 ** 提示你下一步该干啥 **。比如你在做饭，它看了一眼说 "该放盐了"；你在装家具，它说 "下一步拧那颗螺丝"。

这里最关键的能力是 ** goal inference **：你没明说要干啥，但 AI 要从你当前行为 + 环境 + 数字 context 推断你的 goal。论文的 benchmark 显示，多选题场景下 VLM 能 84%，但让它 free-form 生成你的 goal，只有 55%。** VLM 能在选项里挑，但不会自己想出来 **——这是 LLM 当 agent 的根本局限。

Meta 在这里的方案是 VLWM：在 video 上 self-supervised 学 "动作的文本描述 + 状态的文本描述" 的 interleaved sequence，让它能 rollout future。比 prompt VLM 高 20% 成功率。

### Robotic Agent（机器人）

机器人是 embodied AI 的终极形态。** 它有真身体，能真做事 **。

现在机器人有两条路：

** 路线 1：VLA 模型 **。拿 VLM 当 backbone，用 teleoperation 数据（人遥控机器人，记录动作）训练。代表是 π0.5、GR00T N1、Gemini Robotics、OpenVLA。问题：** teleop 数据又贵又少 **，而且模型只学了 "见过的场景"。

** 路线 2：RL in simulation + sim2real **。在仿真里训强化学习，再迁移到真机。问题：** reward 不好设计、sim 和 real 之间有 gap **（仿真里衣服不会真皱，真机上手抓物体摩擦系数不一样）。

Meta 的 vision：** 把 VLA policy 当 System 1（快反应），world model + scoring model 当 System 2（慢思考）**。VLA 直接出 action，但 world model 先在 latent 里 rollout 看看这 action 会不会 fail，如果会 fail 就 re-plan。

这就像人：** 大部分时候你抓杯子靠肌肉记忆（System 1），但遇到滑的杯子你会先 "想一下" 用多大劲（System 2）**。VLA 单独用就是只有 System 1，OOD 就摔杯子；加了 world model 就有了 System 2 的 "preview and check" 能力。

---

## Mental World Model：最有意思的新概念

agent 要 ** 不光 model 物理 world，还要 model 你脑子里在想啥 **。

这叫 Theory of Mind。人是天生会这个的：你看一个人伸手够杯子，你自动推断 "他想喝水"；你看到一个人皱眉，你推断 "他不爽"。

agent 的 mental world model 要建模：
- 你 ** 注意力在哪儿 **
- 你的 ** 短期记忆、长期记忆 **
- 你的 ** 短期意图、长期 goal **
- 你的 ** 情绪 **
- 你的 ** belief **（你觉得某个东西在哪）
- 你对 ** 别人的 belief **（nested belief，二阶 Theory of Mind）

最后这个 ** nested belief 最有意思 **。例子：
- A 觉得 B 觉得钥匙在桌上（但 B 其实没觉得，是 A 误会了）
- agent 要推理出 "A 对 B 的 belief 有误判"，才能预判 A 的下一步行为

现在的 LLM 在这上面很差。ExploreToM 这个 benchmark 用程序化方式对抗性地生成 belief reasoning 任务，** 把现有模型按在地上摩擦 **。

但论文诚实地承认：** 这还只是 evaluation benchmark，learning signal 还没人做出来 **。怎么把 belief tracking 放进 training loop 是 open problem。

---

## Memory：KV cache 会爆这件事

transformer 的 working memory 是 KV-cache，** 长度随对话线性增长 **。

```
memory_cost = O(N * d)
read_FLOPs = O(N * d)
```

你聊 1 小时还好，聊 1000 小时呢？agent 要陪伴你一辈子，KV-cache 就爆了。

论文指出三类现有 memory 都不行：
- ** Model weights**：训练完就 frozen，加新知识要 fine-tune，还会 catastrophic forgetting
- ** KV-cache**：线性增长，不可持续
- ** External memory (RAG)**：uncompressed，要存所有 raw interaction，且需要额外检索处理

论文提出要搞 ** episodic memory **：能 ** sublinear 增长 **、能 ** 在 inference 时 forward-read-write 局部更新 **、能 ** 个人化压缩存储 **。

说白了：** agent 要有一个 "会自己整理、压缩、遗忘" 的长期记忆系统 **，像人脑——你不记得昨天中午每一秒，但你记得 "昨天和谁吃了饭、聊了啥大事"。

这个方向 Meta 还在概念阶段，没给具体 architecture，但点出了 ** 长期 agent 的核心瓶颈 **。可能的方向是 test-time training（TTT）、Mamba 这类 sublinear memory RNN、或者 memory compression + retrieval 混合架构。

---

## Benchmarks 的设计哲学：反作弊

四个 benchmark 都在 ** 防模型走捷径 **，这点很统一，也很关键：

** Minimal Video Pairs**：两个视频几乎一样、问题一样、答案不同。模型不能用 "视觉表面相似性" 蒙混，必须真的理解物理 dynamics。结果：SOTA 40%，人类 93%。

** IntPhys 2**：用婴儿认知实验的 violation-of-expectation 范式。放一个物理上可能的视频和一个不可能的（球穿墙），看模型能不能识别。结果：模型 chance level，人类接近满分。

** CausalVQA**：反事实、假设、预期、规划类问题。"如果当时没用那把刀会怎样？" 模型在反事实上大幅落后人类。

** WorldPrediction**：用 "action equivalent"——多个不同的 low-level 动作能实现同一 high-level 状态转移。强迫模型在 abstract 层推理，不能靠 pixel continuity 蒙。人类满分，SOTA 在 planning task 上只 38%。

** 这四个 benchmark 加起来传达一个信号：现有模型在 "表面理解" 上很强，在 "深层因果理解" 上几乎是零。** 这正是 world model 要补的洞。

---

## System A + System B：最深的思考

这篇 paper 最有意思的是 section 7，它把学习分成两类：

** System A（观察学习）**：看大量数据，自己提取结构。包括 contrastive learning、next-token prediction、MAE。优点：能 scale，能发现 abstract representation。缺点：** 分不清因果和相关 **，需要被动喂数据，没主动性。

** System B（行动学习）**：通过交互学。RL 是典型。优点：grounded in control、能发现新解。缺点：sample inefficient、reward 难定、action space 一大就崩。

** 这俩单独用都有死穴。合起来用才对。**

- System A 给 System B 提供 ** compressed representation 和 prior **，让 RL 在 latent imagination 里 sample-efficient 训练（这就是 Dreamer 和 V-JEPA 2 在做的）。
- System B 给 System A 提供 ** 主动收集的 informative data **，关键是它产生 ** (action, sensory consequence) 的配对数据 **——action 是 agent 自己做的，所以是低 variance 的 "supervision"；sensory consequence 是 noisy 的。用低 variance 信号去 denoise 高 variance 信号，这是 ** 自监督里混进半监督 ** 的好办法。

** 北极星愿景 **：让机器人像婴儿一样，** 看大量 video（System A）+ 自己 motor babbling 乱动观察后果（System B）**，自主学出 world model。不需要 teleop 数据，不需要 reward engineering，纯 self-supervised embodied learning。

这路径如果走通，** 就是 AGI 的另一条路 **——不靠文字语料堆 LLM，靠 embodied interaction 学 world understanding。

---

## 伦理：隐私 + 拟人化

** 隐私 **：眼镜 24/7 在你身边，听到看到所有。比 web agent 难一个量级，因为物理世界的 metadata（你在哪、和谁、什么时候）本身就极度敏感。

技术方案：on-device 加密存储 + federated learning + differential privacy + data minimization。但论文承认 federated learning 单独不够，gradient 能被 inversion；DP 加了会掉精度；data minimization 在 web agent 上都还做不好（AgentDAM 显示 agent 经常 leak 不必要信息）。

** 拟人化 **：avatar 太像人会让人 ** 高估它的能力 **，导致失望、错误信任、甚至安全事故。这叫 illusory agency。

方案：** 显式 communicate 局限、transparent decision、必要时降级 anthropomorphism **。对 therapy 适度拟人有用，对 safety-critical 场景要 de-emphasize。

---

## 我读完的整体感受

这篇 paper ** 不是一个技术 paper，是一份战略宣言 **。

Meta 在 LLM 这波没拿到 top spot，但他们手里有别人没有的 asset：** 眼镜的 egocentric perception、avatar 的 dyadic interaction 数据、机器人的 teleop 平台、touch sensor、4D video 数据集、实时语音模型 **。

把这些 asset 用 "world model" 串起来，等于说：** 下一波 AI 不在 language model，在 embodied world model；不在 next-token prediction，在 latent dynamics prediction；不在单 agent，在 human-agent collaboration；不在 GPU 集群，在 physical interaction data。**

这是一个 ** 把 LLM 降维、把 embodiment 升维 ** 的战略 reframe。

如果他们成功，下一波 foundation model 不是 LLM，是 ** World Foundation Model (WFM) **——同时编码物理动力学、社会认知、个人记忆。

如果失败，瓶颈大概率在：
- ** Episodic memory 还只是概念，没 architecture **
- ** Mental world model 还没 training signal **
- ** Edge inference 算力跟不上 world model rollout **
- ** VLA + world model + scoring model 三件套的系统集成复杂度 **

但整体方向我觉得 ** 是对的 **——LLM 走到 GPT-5、Claude 4 这一代，"语言能力" 已经不是瓶颈，瓶颈在 ** grounding、planning、long-horizon interaction **，而这些恰恰是 embodied AI + world model 要解决的。

---

如果你想我展开某个点——比如 V-JEPA 2 的具体 VICReg loss、Seamless 的 flow matching 细节、VLWM 的 tokenization、或者 nested belief 的具体形式化——直说，我再 drill down。

---

# Embodied AI Agents: Modeling the World 深度解读

这是 Meta AI Research 在 2025 年 7 月发布的一篇 position paper，作者阵容非常豪华（Pascale Fung、Yoram Bachrach、Asli Celikyilmaz、Kamalika Chaudhuri，加上 Yann LeCun 通过 JEPA/V-JEPA 2 等工作深度参与，以及 Jitendra Malik 等资深研究员）。这篇文章本质上 ** 是 Meta 对 "embodied AI" 这条路线的一份蓝图 **，把它过去几年散落在 Seamless、V-JEPA 2、Perception Encoder、Sparsh touch encoder、EgoExo4D、ExploreToM、Moshi 等工作中的碎片，用 "world model" 这根线串起来。

下面我按 "build intuition" 的方式拆给你看。

---

## 1. 论文的整体 intuition

这篇 paper 的核心 thesis 可以浓缩成一句话：** embodied agent 的本质是 "perceive → model world (physical + mental) → plan → act → remember "，而 LLM/VLM 的 next-token / next-pixel prediction 范式不适合做这件事的 planning 核心，需要换成 JEPA-style 的 latent predictive world model。**

论文把 embodied AI agents 分成三类，这三类共享一个统一的 world modeling 框架：

| Agent Type | Embodiment 形式 | Action space | 主要应用 |
|---|---|---|---|
| Virtual Embodied Agents (VEA) | 2D/3D avatar、VR/MR 中的虚拟角色 | facial expression、gesture、body motion (SMPL-H, Imitator features) | therapy、metaverse NPC、studio avatar |
| Wearable Agents | Meta AI Glasses 等 first-person 设备 | "show and tell" 给 user 看；机器 initiative 提示下一步 | coaching (cooking、assembly)、tutoring |
| Robotic Agents | 人形 / 机械臂 + 轮式平台 | joint torques、end-effector Δ(x,y,z)、whole-body control | household、disaster relief、elderly care |

这三类 agent 都需要 world model，但 ** 作用的 "粒度" 不同 **：
- VEA 主要做 ** dyadic interaction 的 motion model **（低频，~30Hz 视觉生成）。
- Wearable agent 做 ** high-level procedural planning **（秒到分钟尺度，"下一个动作是什么"）。
- Robotic agent 同时要做 ** low-level motion planning **（<20Hz，centroid / end-effector 层）和 ** high-level task planning **。

---

## 2. World Model：论文的真正核心

### 2.1 为什么 generative model 不够用

论文反复强调 LLM 和 diffusion model 在 embodied planning 上有 "fundamental flaw"：

> Generative models trained to predict the next token or pixel are excellent for creative tasks but they include too many textual or visual details while missing the essential information for reasoning and planning tasks.

直觉上，next-token prediction ** 是一个 "对 world state 做 full reconstruction" 的目标 **。当你预测下一帧 pixel 时，你把大量与任务无关的细节（光照、纹理、背景）都建模了，但 "insert the battery" 这种 high-level transition 反而被淹没在像素噪声里。而且 generative objective 是 likelihood，** 不保证 latent space 是 "action-causal" 的 **，所以模型经常 hallucinate 出 spurious correlation 的 plan。

论文给出的方案是 ** JEPA (Joint-Embedding Predictive Architecture) **：不重建 observation，只在 abstract latent space 里预测未来。这是 LeCun 2022 那篇 "A Path Towards Autonomous Machine Intelligence" 的延续。

### 2.2 LeCun AMI 架构（论文 Figure 2）

[LeCun 2022 paper](https://openreview.net/pdf?id=BZ5a1r-kVsf) 提出的 modular architecture 包括：

```
        ┌─────────────┐
        │ Perception  │ ← observations o_t
        └──────┬──────┘
               │  (latent state s_t)
               ▼
   ┌────────────────────────┐
   │   World Model WM(·)     │ ← predicts s_{t+1} = WM(s_t, a_t, h_t)
   └──────┬────────┬─────────┘
          │        │
          ▼        ▼
   ┌─────────┐  ┌──────────────┐
   │  Actor  │  │ Cost Module  │  (intrinsic + critic)
   │ (plan)  │  │  C(s_t,a_t)  │
   └─────────┘  └──────────────┘
          ▲
          │
   ┌──────┴──────┐
   │ Short-term  │
   │   Memory    │
   └─────────────┘
```

关键点：
- ** Cost module** 是 intrinsic 的（hard-wired drives，比如 "不要摔"），加上 learned critic（估计 long-term cost-to-go）。
- ** Actor** 通过 MPC：在 latent space 里 rollout 多个 candidate action 序列，挑 cost 最小的执行第一步，然后 re-plan（receding horizon）。
- ** Configurator** 控制 perception 和 world model 的 "注意力/分辨率"，根据当前 task 决定要精细建模什么。

论文把这套架构和 embodied agent 结合，** 关键扩展是加了 "user" 这个维度 **——agent 既要 model 物理 world，又要 model 用户的 mental world，所以 Figure 1 的 interaction loop 是 user ↔ world ↔ agent 三角形。

### 2.3 JEPA Planning 的具体公式（论文 Figure 3）

V-JEPA 2-AC（Action-Conditioned）这篇里给的 planning formulation，从论文 Figure 3 可以提取出来。给定：

- Context frames $\{o_1, o_2, ..., o_n\}$（observed）
- Goal observation $o_{goal}$
- Encoder $E_\theta: o \mapsto \hat{s}$（输出 latent embedding）
- Predictor $P_\phi(\hat{s}_t, a_t) \mapsto \hat{s}_{t+1}$（在 latent space rollout）
- Action sequence $(a_k)_{k \in [T]}$，$T$ 是 planning horizon
- Cost function 这里取 $L_1$ distance 到 goal embedding

Planning objective：

$$\min_{(a_k)_{k \in [T]}} \;\sum_{k=1}^{T} \big\| P_\phi(\hat{s}_{k-1}, a_k) - \hat{s}_{goal} \big\|_1$$

其中：
- $\hat{s}_0 = E_\theta(o_n)$ 是当前 observed state 的 embedding
- $\hat{s}_k = P_\phi(\hat{s}_{k-1}, a_k)$ 是递归预测出的第 $k$ 步 latent state
- $\hat{s}_{goal} = E_\theta(o_{goal})$ 是 goal 的 embedding
- $a_k \in \mathbb{R}^d$ 是 action（在 V-JEPA 2-AC 里是 end-effector 的 $\Delta(x, y, z)$ 加 orientation 的 delta）
- $T$ 是 rollout 长度

变量含义：
- 下标 $k$：planning step index，$k \in [T] = \{1, 2, ..., T\}$
- 下标 $goal$：标记 goal state 的 embedding
- $\theta, \phi$：encoder 和 predictor 的参数

优化可以用 gradient-based（如果 $P_\phi$ 可微）或者 ** Cross-Entropy Method (CEM) **（[Rubinstein 1997](https://www.sciencedirect.com/science/article/pii/S0377221797002222)），论文里特别提到 CEM，因为 action space 在机器人上是连续的、且 JEPA predictor 在 long horizon 上可能不稳。

这个公式背后 ** 一个关键 intuition **：我们 ** 不在 pixel space 规划 **，所以在 latent space 里 "靠近 goal" 就够了，不需要细节。这相当于把 planning 从 $\sim 10^6$ 维像素空间降到 $\sim 10^3$ 维 latent 空间，** 计算量降低 3 个数量级 **，而且不会被纹理/光照噪声 distract。

### 2.4 JEPA vs Generative vs VLM 三条路（论文 Figure 4）

论文 Figure 4 给出了三种 action-conditioned world model 的对比：

| 架构 | 预测空间 | 计算成本 | Plan 可解释性 | Hallucination 风险 |
|---|---|---|---|---|
| Generative video model | pixel space | 高 | 低（生成视频） | 高 |
| VLM-as-planner (text) | text token | 中 | 高（自然语言） | 高（spurious correlation） |
| ** JEPA / VLWM (latent) ** | ** latent embedding ** | ** 低 ** | ** 中 ** | ** 低 ** |

VLWM 是论文重点推的一个变体：** 在语言 token 空间预测 future state description **，相当于把 JEPA 的 latent space 换成 "interleaved natural language"。VLWM 在 VPA (Visual Planning for Assistance) benchmark 上拿到：
- ** +20% SR** (success rate)
- ** +10% mAcc** (mean action accuracy)
- ** +4% mIoU** (mean intersection-over-union)

为什么 VLWM 比 pure VLM prompting 好？因为 VLM 是 next-token 训练的，没有显式被 "rollout future" 训练，所以 hallucinate；VLWM 是 self-supervised 在 video 上学 "下一个动作的文本描述 + 下一个 state 的文本描述" 的 interleaved sequence，** plan 的因果结构被 explicitly 训练进去 **。

---

## 3. Multimodal Perception 的细节

### 3.1 Perception Encoder (PE) + Perception Language Model (PLM)

论文里 PE 是 Meta 的 vision encoder，** 纯 contrastive (CLIP-style) 训练 **，不做 task-specific pretraining。PLM 在 PE 之上接 LLM，用 synthetic + human-annotated image/video 数据训练。结果：

- Image perception tasks: ** +9.1** average points
- Video captioning: ** +39.8 CIDEr**
- Fine-grained video QA: ** +3.8** average points

直觉上，PE 的关键贡献是 ** video data integration **——很多 CLIP-style 模型在 video 上掉链子是因为 contrastive loss 在 frame-level，没建模 temporal。PE 通过 robust video data augmentation 把 contrastive 扩展到时序。

### 3.2 Audio LLM（Moshi）

[Moshi](https://arxiv.org/abs/2410.00037) 是 Meta 的 speech-text foundation model，支持实时对话。架构上：
- Input：audio encoder（semantic + paralinguistic representation）
- Core：LLM（text token + audio token 联合）
- Output：audio decoder（synthesis with long-range context）

embodied agent 在 audio 上的 ** 关键挑战 **：
1. ** Noise robustness**：眼镜场景 bystander speech vs wearer-intended speech 区分极难
2. ** Speech variability**：accent、dialect、whispered speech
3. ** 有限算力**：edge AI
4. ** Tool use + RAG + factuality**：在 speech LLM 上集成检索

未来方向论文提到 ** 多语言扩展到 non-written language **，这是个很 interesting 的点——纯 oral language 用户怎么用 agent，这本质上是把 literacy assumption 拆掉。

### 3.3 Touch (Sparsh)

[Sparsh](https://openreview.net/forum?id=xYJn2e1uu8) 是 Meta 的 general-purpose touch encoder。Touch 的 ** 关键 uniqueness **：
- ** 频率比 vision 高 **（tactile 信号 ~1kHz vs vision ~30Hz）
- ** 当 occlusion 发生时唯一信息源 **（unpacking bag 时手挡住物体）
- ** 提供 force feedback**，对 elderly care 这种需要 gentle handling 的场景必要

Sparsh 能预测：force estimation、slip detection、texture recognition、object pose、grasp stability。

直觉上，touch 是 ** 闭环控制 ** 的关键信号，vision 是开环预测，touch 才能真正 close the loop on manipulation。

---

## 4. Mental World Model（论文最有 originality 的部分）

### 4.1 概念定义

Physical world model 是 agent 对外部物理世界的 representation。** Mental world model 是 agent 对 user / 其他 agent 的 mental state 的 representation **。这是 Theory of Mind (ToM) 的 explicit modeling。

Mental state $X_{m,s}$ 对 subject $s$ 在 perceptual input $X_p$ 下定义为 textual description 的集合：

$$X_{m,s} = \{ \text{attention}_s, \text{STM}_s, \text{LTM}_s, \text{intention}_s^{\text{short}}, \text{goal}_s^{\text{long}}, \text{emotion}_s, B_s^{\text{phys}}, B_s^{\text{other}} \}$$

变量解释：
- $\text{attention}_s$：subject $s$ 当前注意的 perceptual field
- $\text{STM}_s$：short-term memory
- $\text{LTM}_s$：long-term memory + world knowledge
- $\text{intention}_s^{\text{short}}$：短期意图
- $\text{goal}_s^{\text{long}}$：长期目标
- $\text{emotion}_s$：情感状态
- $B_s^{\text{phys}}$：$s$ 对物理世界状态的 belief
- ** $B_s^{\text{other}}$**：$s$ 对 ** 其他 subject 的 mental state 的 belief ** ← 这是 nested belief / higher-order ToM

注意 $B_s^{\text{other}}$ 本身又是一个 mental state，所以这是 ** 递归结构 **。

### 4.2 ExploreToM

[ExploreToM (Sclar et al., 2024)](https://arxiv.org/abs/2412.12175) 用 ** programmatic scene construction ** 对抗性地生成 belief reasoning task。Figure 5 展示了 nested belief 的 probing：

```
Subject A believes (Subject B believes (object X is at location L))
                                ↑
                        2nd-order belief
```

现有 benchmark 的局限：
- ** ToMI** (Le et al., 2019)：只支持 restricted action set
- ** Hi-ToM** (Wu et al., 2023)：稍微扩展但仍然 restriction 重
- ** ExploreToM**：通过程序化场景生成，能 stress test higher-order belief tracking

但论文承认：** ExploreToM 只做 evaluation，不做 learning **。未来要做的是把 latent belief tracking + social feedback 放到 training loop 里，让 agent 在 extended collaboration 中动态更新对他人 mental state 的估计。

### 4.3 Mental world model 的应用

- ** Anticipate goals**：提前 offer assistance
- ** Infer belief discrepancies**：检测 false-belief 场景
- ** Predict emotional responses**：调整 message 风格

直觉上，mental world model 是 ** 让 agent 从 "reactive" 变成 "proactive" 的关键 **。没有它，agent 只能等 user 明确指令；有了它，agent 能 pre-empt user need。

---

## 5. Memory 的细节（论文 3.5 节）

论文把 memory 分成三类，并指出第四类 "episodic memory" 是未来研究方向。

### 5.1 三类现有 memory

| Memory Type | 写入速度 | 容量 | 可变性 | 例子 |
|---|---|---|---|---|
| ** Fixed memory ** | 慢（backprop） | 固定 | 训练后 frozen | model weights |
| ** Working memory ** | 快（forward pass） | 随时间线性增长 | mutable/immutable | RNN hidden state、KV-cache |
| ** External memory ** | 快（写入 DB） | 可扩展 | uncompressed | RAG |

### 5.2 KV-cache 的问题

KV-cache 是 transformer 的 working memory，** 长度 $O(N)$ ** 增长（$N$ 是 token 数），所以：

$$\text{Memory cost} = O(N \cdot d) \quad \text{FLOPs for read} = O(N \cdot d)$$

其中 $d$ 是 head dim * num heads。对话一长就爆。

### 5.3 Episodic Memory 的设计目标

论文提出 episodic memory 需要：

1. ** Personalization**：每个 user 一个 compressed memory slot（像 LoRA adapter 但 for memory）
2. ** Life-long training**：sublinear growth in capacity
3. ** Test-time training**：forward-read-write 的 local update rule

直觉上，这是 ** "transformer 的 KV-cache 不能无限长，但 agent 要 remember 永久" ** 这个矛盾的解决方案。可能的实现方向：
- Test-time training（[Sun et al. 的 TTT 系列](https://arxiv.org/abs/2407.04620)）
- Episodic memory compression（[Pink et al. 2025](https://arxiv.org/abs/2502.06975)）
- Memory-augmented RNN（Mamba、Linear Attention 等 sublinear memory）

---

## 6. World Model Benchmarks 的解读

论文给了四个 benchmark，从不同角度压力测试 world modeling。这些 benchmark ** 设计哲学高度一致 **：用 minimal pair 或 violation-of-expectation 来防止 shortcut。

### 6.1 Minimal Video Pairs (MVP)

[MVP (Krojer et al., 2025)](https://arxiv.org/abs/2506.09987)

设计：** 两个几乎一样的 video + 同一个问题 + 不同答案 **。

```
Video A: 球从桌上滚下来，碰到地板 → Q: 球会停在哪？ A: 地板
Video B: 球从桌上滚下来，碰到墙   → Q: 球会停在哪？ A: 墙
                  (视觉几乎相同)
```

数据规模：** 55,000 video-QA pairs **

结果：
| Model | Accuracy |
|---|---|
| Random guess | 25% |
| SOTA video-language model | ** 40.2%** |
| ** Human ** | ** 92.9%** |

直觉：模型在 ** 视觉相似性 ** 上 fall back to surface pattern，没有真正 model 物理 dynamics。

### 6.2 IntPhys 2

[IntPhys 2 (Bordes et al., 2025)](https://arxiv.org/abs/2506.09849)

设计：** violation-of-expectation paradigm **，类似 infant cognitive science 实验。

四个 principle：
1. ** Permanence**：物体不能凭空消失
2. ** Immutability**：物体属性不变
3. ** Spatio-Temporal Continuity**：物体轨迹连续
4. ** Solidity**：物体不能穿透

结果：** 模型 chance level，human 接近 ceiling **。

直觉：这就像把 Wason selection task 拿来测 LLM 一样——表面看 LLM 能 talk about physics，但真正的 intuitive physics 能力几乎是 0。

### 6.3 CausalVQA

[CausalVQA (Foss et al., 2025)](https://arxiv.org/abs/2506.09943)

五种问题类型：
1. ** Counterfactual**："如果当时没碰那个杯子，会怎样？"
2. ** Hypothetical**："如果用更重的球砸，会怎样？"
3. ** Anticipation**："接下来会发生什么？"
4. ** Planning**："要达到 X，下一步该做什么？"
5. ** Descriptive**（baseline）

结果：模型在 anticipation 和 hypothetical 上 ** 显著 underperform human **。

### 6.4 WorldPrediction

[WorldPrediction (Chen et al., 2025)](https://arxiv.org/abs/2506.04363)

两个 task：
- ** WorldPrediction-WM**：给定 initial state image 和 final state image，从 4 个 video 里选正确的 action video
- ** WorldPrediction-PP** (Procedural Planning)：给定 initial 和 final，从 distractors 里选正确的 action sequence

关键设计：** "action equivalents" **——多个不同的 low-level action 可以实现同样的 high-level state transition，所以模型必须 ** 在 abstract level 推理 **，不能依赖 pixel-level continuity。

数据基于 ** EgoExo4D**（[Grauman et al., 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Grauman_Ego-Exo4D_Understanding_Skilled_Human_Activity_From_First-_and_Third-Person_CVPR_2024_paper.pdf)）。

结果：
| Task | Human | SOTA |
|---|---|---|
| WM | ~100% | ** 57%** |
| PP | ~100% | ** 38%** |

PP 比 WM 难，因为需要 ** sequence-level reasoning ** 而不是 single-step。

** 半 MDP formulation **：WorldPrediction 形式化为 partially observable semi-MDP，state 是高 level 的（"drawer is open"），action 是 high-level option，这种 abstraction 让它和 POMDP 不同——option 内部细节被 abstract 掉了。

---

## 7. 三类 Agent 的架构细节

### 7.1 Virtual Embodied Agent（VEA）

** 核心是 Dyadic Motion Model **（论文 Figure 11）。

架构基于 ** Diffusion Transformer + Flow Matching **：

```
Input: dyadic audio (双方 speech) ──┐
       user visual features (opt) ─┤
                                    ▼
                          Diffusion Transformer
                                    │
                ┌───────────────────┴───────────────────┐
                ▼                                          ▼
        Imitator features (face)                SMPL-H (body)
                │                                          │
                └─────── 2D / 3D renderer ─────────────────┘
```

两种 integration 策略（Figure 10）：

**(a) Cascaded integration**：
```
Speech LLM → speech tokens → Motion Model → motion
```

**(b) Codebook integration**（更先进）：
```
Speech LLM hidden states → Adapter → emotion code + gesture code
                                                ↓
                                          Motion Model
```

Codebook 的好处：** speech LLM 已经 grasp conversational context **，通过 adapter 把 context-aware 的 emotion/gesture code 喂给 motion model，所以手势和当前对话的 semantic content 对齐（说 "fly" 时手势模仿飞行）。

** Seamless Interaction Dataset **：4000+ 小时、4000+ 参与者、1300+ prompts，包含 Naturalistic 和 Improvised 两类。这是目前最大的 dyadic interaction dataset。

评估指标：
- ** Subjective**：pairwise comparison，10 维度（lifelikeness、clarity of intent、turn-taking、listening、speaking...）
- ** Objective face**：Sync-C、Sync-D（lip-sync）、FID
- ** Objective body**：FGD (Fréchet Gesture Distance)、Diversity

直觉：** VEA 的核心难题是 turn-taking 和 listener behavior **。当前大多数 avatar 只会 "speak"，不会 "listen"——但 realistic conversation 里 listener 也有 micro-expression、nodding 等。Seamless 模型显式建模 listening gestures，这是它优于以前工作的点。

### 7.2 Wearable Agent

** 架构（Figure 12）**：
```
Egocentric video + audio + digital context + longitudinal
              ↓
   Contextual Goal Prediction Module
              ↓
       Predicted goal
              ↓
         Planner (VLWM / JEPA)
              ↓
      Next action step (show / tell)
```

** VLWM (Vision-Language World Model) ** 是核心模型：
- 输入：visual context
- 输出：interleaved natural language sequences 描述 (action, world state) 的 future trajectory
- 训练：self-supervised 在 unlabelled video 上（kitchen、workshop、clinical recordings）
- 用途：simulate future trajectories → evaluate candidate plans → reason causal dependencies

VLWM 的优势（论文给出）：
- 比 VLM prompting 高 ** +20% SR, +10% mAcc, +4% mIoU**
- PlannerArena human evaluation：VLWM 的 system-2 reasoning 被人类 prefer

** JEPA-based planner 是另一种路径 **：
- 预测在 latent video representation 空间
- 优势：differentiable planning objective + test-time efficiency + 与 MPC 集成
- 劣势：less interpretable

VLWM vs JEPA planner 的 trade-off：
| 维度 | VLWM | JEPA Planner |
|---|---|---|
| 可解释性 | 高（自然语言） | 低 |
| Optimization | discrete search | gradient-based |
| Latency | 中 | 低（适合 MPC） |
| 用途 | high-level plan | real-time control |

** Goal Inference Benchmark **：
- 348 参与者，3477 recordings
- multimodal：visual、audio、digital、longitudinal
- Human ** 93%** vs best VLM ** 84%**（multiple-choice）
- 但 generative setting：best VLM 只 ** 55%** relevant goals

直觉：wearable agent 的最大挑战是 ** "user 没说但要猜" **。当前 VLM 在多选题上能 84%，但让它 free-form generate user 的 goal，就掉到 55%——这说明 VLM ** 能 discriminate 但不能 well-generate goals **，可能是因为训练目标里没有 "infer goal" 这个 task。

### 7.3 Robotic Agent

** 物理能力分类**：
1. ** Locomotion + Navigation**：legged robot 在 uneven terrain
2. ** Manipulation**：grasping、placing
3. ** Dexterous manipulation**：multi-finger hands，"put key in lock"

** Brain 能力**：
- ** Generalization**：new tasks、new embodiments、new environments
- ** Efficient & lifelong adaptation**：personalization without catastrophic forgetting
- ** Spatial & temporal memory**：semantic mapping + sub-task tracking + user interaction memory
- ** Language instructions & planning**：multi-level abstraction
- ** Human/agent interaction**：receiving instructions、asking clarifications、physical assistance

** 两条主要技术路线**：

** (1) Vision-Language-Action Models (VLAs)**
- 代表：[π0.5 (Physical Intelligence)](https://arxiv.org/abs/2504.16054)、[GR00T N1 (NVIDIA)](https://arxiv.org/abs/2503.14734)、[Gemini Robotics](https://arxiv.org/abs/2503.20020)、[OpenVLA](https://arxiv.org/abs/2406.09246)、[SmolVLA](https://arxiv.org/abs/2506.01844)
- 训练：teleoperation data + VLM backbone
- 优势：open-set task specification via language
- 瓶颈：** clean teleoperation data 生成 **

** (2) RL in simulation + sim2real**
- 优势：full-body controller for locomotion/navigation
- 瓶颈：** reward specification + sim2real gap **（deformable object、precise interaction physics）

论文的 vision（Figure 15 left）：** 把 VLA policy + world model + cost/scoring model 组合 **：

```
       observation
           ↓
   ┌───────────────────┐
   │   World Model      │ ← predicts future state under candidate actions
   └─────────┬─────────┘
             │
   ┌─────────┴─────────┐
   │  Scoring Model    │ ← evaluates: will VLA's action succeed?
   └─────────┬─────────┘
             │
             ▼ (if predicted fail → re-plan)
        VLA Policy
             │
             ▼
         action
```

直觉：** VLA 是 System 1（fast reactive），World Model + Scoring 是 System 2（slow deliberative）**。VLA 单独用会 fail on OOD，但 World Model 能 detect "我预测执行这个 action 会 fail"，触发 re-plan。

### 7.4 Classical Robotics 仍有价值（论文 6.2.1）

论文专门讨论 classical robotics 的 ** analytical models + MPC **。MIT Humanoid 做 backflip（[Chignoli et al.](https://ieeexplore.ieee.org/document/9555782)）就是 hierarchical control：

```
High-level (Hz~1):    trajectory planning
Mid-level (Hz~100):   MPC with analytical dynamics
Low-level (Hz~1000):  joint torque control
```

直觉：** learned world model 和 analytical model 不是替代关系，是互补 **。低频高层用 learned model（semantic、abstract），高频低层用 analytical model（precise dynamics）。这种 hierarchical 思路在 humanoid 上非常重要，因为 learned model 推理速度跟不上 1kHz 的 torque control。

---

## 8. System A + System B：未来学习范式

这是论文里我认为 ** 最有思想深度的部分 **（Section 7）。

### 8.1 两个 paradigm

** System A**（observation-based learning）：
- Self-supervised / unsupervised learning on passive data
- 例子：contrastive learning、next-token prediction、MAE
- 优势：** scales with data **，发现 abstract hierarchical representations
- 劣势：** 没有主动性 **，分不清 correlation vs causation，需要 curated data

** System B**（action-based learning）：
- Reinforcement learning、active perception
- 优势：grounded、sparse reward、real-time adaptive、能 discover novel solutions
- 劣势：** sample inefficient**，high-dim action space 难，需要 reward specification

### 8.2 互相帮助

** System A → System B**：提供 structure、priors、compressed representations
- 例：Dreamer、[V-JEPA 2](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) 学 world model，让 RL 在 latent imagination 里 sample-efficient 训练

** System B → System A**：主动收集 informative data
- 直接：optimize System A 的 predictive objective（active learning）
- 间接：explore 产生 task-relevant trajectories
- 关键概念：** parallel corpora of (action, sensory consequence) **，这相当于 self-supervised 的 "supervised" 信号，因为 action 是低 variance 的（agent 知道自己做了什么），但 sensory consequence 是 noisy 的

### 8.3 Table 1 的四个 integration 例子

| Domain | System A | System B | Learning Loop |
|---|---|---|---|
| Motor Control | Dreamer / Video-JEPA | RL with imagination | World model enables sample-efficient control via latent imagination |
| Language Learning | Pretraining (next-token) | RLHF | System A 学 linguistic representation，System B 通过 interaction 对齐人类 preference |
| Complex Skills | SSL from play data | Goal-conditioned BC | Self-supervised embedding of latent plans and affordances |
| ** Social Learning ** | Observe multi-agent interaction | Interactive probing for intent/belief | ** "Not done to our knowledge" ** ← 论文 explicitly 说没做过 |

最后一行 ** social learning 是空白 **——这正是 mental world model 的 future direction。让 agent 一边观察 multi-agent 互动，一边主动 probe 来推断 belief，这是个未探索的领域。

### 8.4 North Star：infant-like learning

> A possible North Star for this would be a robot that would learn, like infants do, useful action / vision world models from observation of its video stream plus motor babbling.

[Motor babbling](https://en.wikipedia.org/wiki/Motor_babbling) 是婴儿自发地挥动手脚，通过观察 consequence 学 body schema 和 world dynamics。这相当于 ** System A (observation) + System B (random action) ** 的最小组合。

直觉：当前机器人训练是 "teleoperation → imitation"，相当于 supervised learning；infant 路径是 "self-supervised exploration"，sample inefficient 但 general。论文的 vision 是 ** 让两者结合 **：用大量 passive video 预训练 + 少量 aligned (video, action) 微调 + motor babbling 自我探索。

---

## 9. Multi-Agent Interaction

### 9.1 三种 multi-agent 场景

1. ** Multi-robot**：disaster relief、autonomous vehicle fleet
2. ** Multi-wearable**：眼镜 + 手表 + haptic vest 协作
3. ** Virtual + physical hybrid**：plan dinner（wearable 推荐 menu + robotic cook + wearable guidance）

### 9.2 三个核心挑战

- ** Communication**：emergent communication（[Foerster et al., 2016](https://arxiv.org/abs/1605.06621)）
- ** Coordination**：decentralized、partial view
- ** Conflict resolution**：[negotiation-based approaches](https://arxiv.org/abs/1903.09946)

直觉：** multi-agent 在 embodied 场景比 web agent 难得多 **，因为 embodied agent 有 spatial overlap、physical interference、real-time constraint。这领域 fundamental 研究还不够。

---

## 10. Ethical Considerations

### 10.1 Privacy & Security

embodied agent ** 24/7 在用户身边 **，能看到、听到所有。论文给的技术方案：

1. ** On-device encrypted storage**：model weights 和 personal data 加密存本地
2. ** Federated Learning**（[Kairouz et al., 2021](https://arxiv.org/abs/1911.02117)）：只传 gradient 不传 data
3. ** Differential Privacy**（[Dwork et al., 2006](https://link.springer.com/chapter/10.1007/11681878_14)）：gradient 仍可 inversion，需要 DP
4. ** Data minimization**：agent 只 access 任务必要信息——但 [AgentDAM (Zharmagambetov et al., 2025)](https://arxiv.org/abs/2503.09780) 显示 web agent 经常 leak 不必要 sensitive info

直觉：embodied agent 的 privacy 比 web agent 难 ** 一个量级 **，因为它感知的是物理世界，物理世界的 "metadata"（在哪、和谁、什么时候）本身就极度敏感。

### 10.2 Anthropomorphism

** Illusory agency**：用户高估 agent 能力，导致 unrealistic expectation、disappointment、甚至 safety issue。

论文的 proposal：** responsible design patterns **，包括：
- 显式 communicate capability limitation
- Transparency about decision-making
- 多模态替代设计（voice / gesture interface，不一定 face）

直觉：** anthropomorphism 是双刃剑 **——提升 engagement 但模糊了 "machine vs human" 的边界。对 therapy、教育场景，适度 anthropomorphism 有用；对 safety-critical 场景（医疗、驾驶），应该 de-emphasize。

---

## 11. 我的整体 intuition & 批评

### 11.1 这篇 paper 真正在说什么

表面上这是 embodied AI 的综述，** 实际上这是 Meta 对 "post-LLM" 范式的一份路线图 **。关键信号：

1. ** 反复强调 generative model 在 planning 上的 inefficiency **——这等于明示 LLM 路径走不通
2. ** 主推 JEPA / VLWM 这类 latent predictive model**——LeCun 的长期主张
3. ** Mental world model 概念把 "Theory of Mind" 提升到 first-class component**——这是 RL 之外的 social AI 路径
4. ** System A + System B 整合 ** 把 SSL 和 RL 统一在一个 architecture 里

### 11.2 几个我觉得最强的 idea

1. ** "Action equivalent" 在 WorldPrediction 里的设计 **：强迫模型在 abstract level 推理，这是反 shortcut 的好方法
2. ** Mental world model 的 nested belief 递归 **：$B_s^{\text{other}}$ 是 belief about others' belief，这是 cognitive science 接入 AI 的 clean 接口
3. ** Episodic memory with sublinear growth**：当前 KV-cache 的 $O(N)$ 增长是 agent 长程互动的死穴，提出 forward-read-write local update 是关键
4. ** System A + B 整合的 parallel corpora (action, sensory consequence) **：把 action 当成低 variance 监督信号去 denoise sensory learning，这个 idea 很 powerful
5. ** Mental world model + Social Learning 在 Table 1 标 "not done" **：诚实承认空白

### 11.3 几个我觉得有问题的地方

1. ** "Generative model inefficient" 的论断有点 over-claimed**：π0.5、Gemini Robotics 这些 VLA 用 generative 范式做得相当不错，论文倾向于贬低
2. ** VLWM 的 "在 language space 预测" 看起来 elegant，但 token 速率限制**——high-frequency control 用 language 当 latent 速度跟不上
3. ** Mental world model 离 practical 还很远**：ExploreToM 还只是 evaluation，没有 learning signal
4. ** Memory section 比较浅**：只提了概念，没给出具体 architecture 或实验
5. ** Multi-agent 部分过于 high-level**，没给 Meta 自己的具体研究
6. ** 整篇 paper 缺少 quantitative system-level comparison**：VLWM vs JEPA planner vs VLM prompting 的 head-to-head 数据没有完整 table

### 11.4 这篇 paper 的 strategic 意义

Meta 在 LLM 时代相对落后于 OpenAI、Anthropic、Google，但 Meta 在 ** embodied AI（眼镜、avatar、机器人、touch、Seamless）** 上的积累是独一无二的。这篇 paper 把这些 asset 用 "world model" 串起来，等于告诉外界：** 下一波 AI 不在 language model，在 embodied world model **。这是一个 strategic reframe。

如果他们成功，下一波 foundation model 不是 LLM，是 ** World Foundation Model (WFM) **——同时编码 physical dynamics、social cognition、personal memory。

如果他们失败，瓶颈大概率在：
- Memory architecture（episodic memory 还只是概念）
- Mental world model 的 training signal（怎么 ground belief tracking）
- Real-time inference on edge（眼镜、机器人算力限制）

---

## Reference Links

- [LeCun, A Path Towards Autonomous Machine Intelligence (2022)](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)
- [WorldPrediction Benchmark (Chen et al., 2025)](https://arxiv.org/abs/2506.04363)
- [Seamless Interaction: Dyadic Audiovisual Motion Modeling](https://ai.meta.com/research/publications/seamless-interaction-dyadic-audiovisual-motion-modeling-and-large-scale-dataset/)
- [IntPhys 2 (Bordes et al., 2025)](https://arxiv.org/abs/2506.09849)
- [CausalVQA (Foss et al., 2025)](https://arxiv.org/abs/2506.09943)
- [Minimal Video Pairs (Krojer et al., 2025)](https://arxiv.org/abs/2506.09987)
- [ExploreToM (Sclar et al., 2024)](https://arxiv.org/abs/2412.12175)
- [DINO-WM (Zhou et al., 2024)](https://arxiv.org/abs/2411.04983)
- [Sparsh: Self-supervised Touch Representations](https://openreview.net/forum?id=xYJn2e1uu8)
- [Ego-Exo4D (Grauman et al., CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Grauman_Ego-Exo4D_Understanding_Skilled_Human_Activity_From_First-_and_Third-Person_CVPR_2024_paper.pdf)
- [Moshi: Real-time Speech-Text Dialogue Model](https://arxiv.org/abs/2410.00037)
- [π0.5 (Physical Intelligence)](https://arxiv.org/abs/2504.16054)
- [GR00T N1 (NVIDIA)](https://arxiv.org/abs/2503.14734)
- [Gemini Robotics](https://arxiv.org/abs/2503.20020)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [SmolVLA](https://arxiv.org/abs/2506.01844)
- [MIT Humanoid Backflip (Chignoli et al.)](https://ieeexplore.ieee.org/document/9555782)
- [Federated Learning Advances (Kairouz et al.)](https://arxiv.org/abs/1911.02117)
- [AgentDAM: Privacy Leakage for Web Agents](https://arxiv.org/abs/2503.09780)
- [Episodic Memory Position Paper (Pink et al., 2025)](https://arxiv.org/abs/2502.06975)
- [Emergent Communication (Foerster et al., 2016)](https://arxiv.org/abs/1605.06621)
- [Hi-ToM: Higher-Order Theory of Mind](https://aclanthology.org/2023.findings-emnlp.715/)
- [Rubinstein CEM (1997)](https://www.sciencedirect.com/science/article/pii/S0377221797002222)
- [Meta AI Glasses](https://www.meta.com/ai-glasses/)

---

如果你想深入某一个部分，我可以再展开——比如 V-JEPA 2 的具体 training objective（VICReg + action-conditioned prediction）、Seamless 的 diffusion transformer 细节、或者 VLWM 的具体 tokenization 策略。哪个方向你最想 build intuition？
