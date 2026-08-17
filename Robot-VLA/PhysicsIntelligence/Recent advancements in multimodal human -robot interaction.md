---
source_pdf: Recent advancements in multimodal human -robot interaction.pdf
paper_sha256: 869989cb8553c1374e6a78c8dbe20a92549422d4afd5fe6e7c1456cec25837db
processed_at: '2026-08-11T21:42:44-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇 Multimodal HRI Review

## 一句话总结

这篇 paper 干了一件事:把 "robot 怎么跟人互动" 这个大杂烩领域,按 **input → output → application** 三条线梳理了一遍,告诉你 2023 年以前大家都在玩什么。

---

## 这个领域到底在干嘛

你想想,你跟一个人聊天,你会同时用到眼睛看他表情、耳朵听他说话、手做手势、身体姿态传达情绪。robot 也想这么干,这就是 multimodal HRI。

核心矛盾很简单:robot 只有一个摄像头一个麦克风的时候,跟人互动很笨。给它多几个感官,互动就自然了。这篇 review 就是在盘点 "大家给 robot 加了哪些感官,这些感官怎么协作"。

---

## 五种感官的通俗版

paper 把 robot 的感官分成五类,我给你用人话翻译:

### 1. Audio — 耳朵
robot 用麦克风收声,做 ASR (speech-to-text),再做 NLU 理解意图。你跟 Pepper robot 说 "帮我拿杯水",它得先听清你说啥,再知道你想干嘛。

难点:方言、口音、背景噪音。工厂车间里 ASR 基本废掉,所以大家转向 gesture。

### 2. Visual — 眼睛
robot 用摄像头看世界。子任务包括:
- 你是谁 (face recognition)
- 你在指哪里 (gesture recognition)
- 你在看哪里 (gaze tracking)
- 你脸上是啥表情 (affective computing)

难点:光照一变就崩,人一挡就瞎。

### 3. Haptic — 触觉
robot 摸东西能感觉到力、压力、纹理。比如做手术的 robot 得知道捏 organ 用多大劲,不然捏碎了。

传感器:ATI force/torque sensor, BioTac 指尖,或者 [robotic skin](https://doi.org/10.1002/aisy.202100047) 整个身体铺 tactile sensor。

### 4. Kinesthetic — 运动觉
robot 知道自己关节在什么位置、转多快。本质上是 "robot 自己的身体感知"。工业 arm 撞到东西,关节 torque 突变,立刻停。

### 5. Proprioceptive — 本体感觉
这个跟 kinesthetic 很像,但更强调 "robot 对自己身体状态的内部估计"。论文里提到一个 cool work,[Malinovská et al., 2022](https://doi.org/10.1109/ICDL53763.2022.9962195),让 humanoid robot 从自己关节位置预测 "哪里被摸了",相当于学了一个 internal body model。

**直觉**:这五个 modality 其实就是人类自己的五感在 robot 上的映射。你想想你自己倒水喝,你用眼睛看杯子位置 (visual),手感觉杯子重量 (haptic),手臂肌肉感知关节角度 (kinesthetic + proprioceptive),耳朵听水流声判断满了没 (audio)。robot 要做同样的事,就得把这几个 channel 融起来。

---

## Input 端:robot 怎么"读"人

paper 把 input 拆成四大块:

### Computer Vision
就是 robot 的眼睛。你给它一张图 $I \in \mathbb{R}^{H \times W \times 3}$,它要做 detection (YOLO)、recognition (FaceNet)、tracking (SORT/DeepSORT)、pose estimation (OpenPose)。

2012 年 AlexNet 之后这块基本被 CNN 统治,2020 年后 ViT 开始抢地盘。但在 HRI 场景有个特殊点:robot 不止要识别物体,还要理解人的 **意图**。比如你伸手去拿杯子,robot 得预判 "这人要拿杯子" 而不是 "这人在挥挥手"。这就是 [gaze + action anticipation](https://doi.org/10.1109/ICRA.2017.798912) 的工作。

### Natural Language Processing
你跟 robot 说话,经过这条 pipeline:

$$\text{speech} \xrightarrow{\text{ASR}} \text{text} \xrightarrow{\text{NLU}} \text{intent + slots} \xrightarrow{\text{dialogue mgr}} \text{response} \xrightarrow{\text{NLG}} \text{text} \xrightarrow{\text{TTS}} \text{speech}$$

举个具体例子:你说 "把温度调到 20 度"。
- ASR: "把温度调到 20 度"
- NLU: intent = `SET_TEMPERATURE`, slot = `{target: 20, unit: celsius}`
- Dialogue manager: 查当前温度,决定回复
- NLG: "好的,已为您调到 20 度"
- TTS: 生成语音输出

[Khurana et al., 2022](https://doi.org/10.1007/s11042-022-13428-4) 是 NLP 综述,强调 OOV (out-of-vocabulary) 词是大问题。现在 LLM 时代这个问题缓解了很多,因为 LLM 的 vocabulary 大得多。

### Gesture Recognition
手势分两种:
- **Static**: 摆个 pose,比如点赞 👍、比心 🫶。你截一帧就能识别。数学上就是 $f: \mathbb{R}^{J \times 3} \to \mathcal{G}$,$J$ 是手部关键点数,MediaPipe 给 21 个点。
- **Dynamic**: 一连串动作,比如 "过来过来" 的招手。你需要时序模型,$f: \mathbb{R}^{T \times J \times 3} \to \mathcal{G}$。典型用 [ST-GCN](https://doi.org/10.1109/CVPR.2018.00942) (spatial-temporal graph convolutional network) 或 3D-CNN + LSTM。

**直觉**: skeleton-based 方法比 raw image 好太多,因为你去掉了 texture/lighting/clothing 这些 nuisance variable,只保留 "人形结构" 这个本质信号。这跟 mesh-based CG 的思路一样 — 你只关心关节角度,不关心穿啥衣服。

### Emotion Recognition
这个 paper 讲得最有意思。它说 emotion 在 HRI 里有三个视角:

1. **Robot 自己有情感** — 给 robot 一个内部 emotional state $e_r(t)$,通常用 PAD 三维空间 (Pleasure-Arousal-Dominance)。比如 robot 被 yelled at,$e_r$ 的 arousal 维度上升。
2. **Robot 表达情感** — 通过 facial expression、voice prosody、body posture 让人看出来 robot "高兴/困惑/抱歉"。
3. **Robot 识别人情感** — 从 face + voice + physiological signal (EEG, GSR, ECG) 反推人的 emotion。

第三点里有个很关键的区分:
- **Internal signals** (EEG, GSR, ECG): 不容易伪装,但要戴设备,很 invasive
- **External signals** (face, voice, body): 容易采集,但人可以装 (poker face)

**多模态的 motivation 就在这里**:你 face 上装笑,但 GSR (皮肤电导)  betray 你紧张。单一 modality 容易被骗,fusion 起来 robust。

---

## Output 端:robot 怎么"说"回人

input 是"robot 读人",output 是"robot 跟人沟通"。也是四大块:

### Speech Synthesis (TTS)
robot 把文本转语音。现代 neural TTS (VITS, FastSpeech2, Tacotron2) 的 pipeline:

$$\text{text} \to \text{phoneme} \to \text{mel-spectrogram} \to \text{waveform}$$

VITS 的 loss 是个 combo:
$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_1 \mathcal{L}_{\text{adv}} + \lambda_2 \mathcal{L}_{\text{FM}} + \lambda_3 \mathcal{L}_{\text{dur}} + \lambda_4 \mathcal{L}_{\text{pitch}} + \lambda_5 \mathcal{L}_{\text{energy}}$$

各项意思:
- $\mathcal{L}_{\text{recon}}$: mel-spectrogram 重建误差,管音质
- $\mathcal{L}_{\text{adv}}$: GAN 对抗损失,让声音更自然
- $\mathcal{L}_{\text{FM}}$: flow matching,管 phoneme-to-mel 对齐
- $\mathcal{L}_{\text{dur}}, \mathcal{L}_{\text{pitch}}, \mathcal{L}_{\text{energy}}$: 三个超音段特征预测

HRI 场景特别需要 **expressive TTS** — 同一句 "好的" 用开心/抱歉/疑惑语气说,效果完全不同。

### Visual Feedback
robot 通过屏幕、灯光、表情展示状态。比如:
- Pepper 胸前 LED 灯蓝色 = 正常待命
- 闪红 = 出错了
- 滚动箭头 = 指路
- 屏幕显示对话气泡 = 给听障用户看 ASR 结果

这块本质是 UX design,跟 robotics 算法关系不大,但 paper 强调它对 HRI 体验影响巨大。

### Gesture Generation — 最有意思的一块
robot 怎么生成"边说边比划"的 gesture。这里有个核心 insight:**gesture 跟 speech 必须时间对齐**。

McNeill 的语言学理论说,人说话时 gesture 分四个阶段:
- **preparation**: 手抬起来准备
- **stroke**: 最强调的那个瞬间,跟语音重音对齐
- **hold**: 停顿一下
- **retraction**: 手放下

[ACE framework](https://doi.org/10.1002/cav.6) 实现这套 timing,关键约束是 stroke 的 onset 要和 speech 的 pitch accent 在 ±100ms 内对齐,不然就感觉"假"。

[Chae et al., 2022](https://doi.org/10.1016/j.robot.2022.104154) 用 morphemic analysis 把句子切成语义单元,再从 motion primitives 库检索 gesture,report 83% 准确率。这跟现在 [BEAT](https://doi.org/10.1145/3478513.3480492) 的端到端 transformer 思路形成对比 — 后者直接从 audio+text 生成 joint trajectory:

$$\hat{Q}_{1:T} = \text{TransformerDec}(z_{1:T}^{audio}, z_{1:T}^{text}, q_0)$$

**直觉**: gesture generation 难就难在它不是 "翻译" 问题,是 "诠释" 问题。你说 "那个杯子很大",手势可以比划大小、可以指向杯子、可以两个都做 — 没有 ground truth,只有"自然不自然"的主观判断。

### Emotional Expression Generation
paper 引 [Stock-Homburg, 2022](https://doi.org/10.1007/s12369-021-00778-6) 强调一个点:**emotion 之间的 transition 比单点 emotion 更重要**。robot 从 "开心" 切到 "疑惑" 应该平滑过渡,不能突变,不然人觉得很诡异。

可以建模成 Markov process:
$$P(e_t | e_{t-1}) = \text{softmax}(W_e e_{t-1} + b_e)$$

或更复杂用 ODE: $\dot{e} = f(e, u, t)$,$u$ 是外部刺激。

### Multimodal Feedback Fusion
最后,robot 把 speech + visual + haptic 组合输出。比如医院导诊 robot:
- 语音说 "请跟我来"
- 屏幕 display 路线图
- 头部朝目标方向转
- 如果用户没跟上,加个 beep

每个 channel 互补,降低单一 channel 过载。

---

## 四大应用场景

paper Section 4.3 + 5.3 讲了四个主要落地场景:

### 1. Industrial Cobots
工厂里的协作 robot。比如 KUKA iiwa、Franka Panda 这种 7-DOF arm,跟工人一起装配。安全是核心 — 撞到人得立刻停。

传感器组合:
- Vision sensor (Kinect Azure, RealSense): 看人在哪
- Tactile sensor: 检测碰撞
- Joint torque sensor: 内置,从 torque 异常推断 collision
- Proximity sensor: 超声波/电容,提前感知人靠近

[Popov et al., 2017](https://doi.org/10.1109/ROMAN.2017.8172400) 用 joint torque 分类碰撞类型。

### 2. Assistive Mobile Robots
智能轮椅 + 智能助行器。全球 65M 轮椅用户,很多人手部力量不够用不了 joystick,所以多模态输入很关键:
- Speech: "去厨房"
- Eye gaze: 看哪个方向就往哪走
- Face expression: 皱眉 = 停,笑 = 继续
- EMG: 肌肉电信号驱动
- EEG: 脑电接口,最 invasive 但给高位截瘫用户用

[Sharifuddin et al., 2019](https://doi.org/10.1109/AiDAS47888.2019.8970865) 用 CNN 做语音控制轮椅。

### 3. Robotic Exoskeletons
外骨骼,辅助瘫痪患者行走,或工业工人负重。核心是 **intent decoding** — robot 怎么知道人想迈腿了?

paper 区分两种 HRI:
- **cHRI (cognitive)**: 用 EEG/EMG 解码意图,在肌肉实际收缩前预测
- **pHRI (physical)**: 用 force/position 作为控制输入,robot 检测人施加的力,跟随运动

pHRI 的核心公式是 [Hogan 1984 impedance control](https://doi.org/10.23919/ACC.1984.4788393):

$$F_{ext} = M_d \ddot{e} + B_d \dot{e} + K_d e$$

其中:
- $e = q - q_d$ (position error)
- $M_d, B_d, K_d$ 是 desired inertia/damping/stiffness 矩阵
- $F_{ext}$ 是人施加的力

直觉:把 robot 模拟成 "弹簧+阻尼+质量" 系统,人推它就像推一个可调阻尼的弹簧。$K_d$ 小 = 软 (compliant),大 = 硬 (stiff)。$B_d$ 大 = 抑制震荡。这是 exoskeleton 控制的基石。

### 4. Robotic Prosthesis
假肢控制。给定 $N$ 通道 sEMG 信号 $X \in \mathbb{R}^{T \times N}$,分类 grasp pattern (power grasp, pinch, tripod, etc.)。

经典方法提 5 个时域 feature:
1. **RMS**: $\sqrt{\frac{1}{T}\sum_t x_t^2}$ — 信号能量
2. **Waveform Length**: $\sum_t |x_t - x_{t-1}|$ — 信号复杂度
3. **Zero Crossings**: 信号过零次数 — 频率指标
4. **Mean Absolute Value**: $\frac{1}{T}\sum_t |x_t|$ — 平均幅度
5. **Max-Min**: $\max(x) - \min(x)$ — 动态范围

[Ameri et al., 2018](https://doi.org/10.1371/journal.pone.0203835) 直接用 CNN 端到端从 raw EMG 分类,跳过手工 feature。

---

## 几个最 interesting 的 recent works

### [Hou et al., 2022 — STM3I](https://doi.org/10.1155/2022/3952758)
Self-tuning multimodal fusion。核心思想:fusion 权重不是固定的,是动态的。

$$\hat{y} = \sum_i w_i(t) f_i(x_i), \quad w_i(t) = \text{softmax}(g_\phi(\text{context}_t))$$

**直觉**:工厂噪音大时,语音 channel 不可信,权重 $w_{audio} \downarrow$,gesture 权重 $w_{gesture} \uparrow$。安静办公室里反过来。一个 model 适应多个场景。

### [Bucker et al., 2022 — LLM Trajectory Reshaping](https://doi.org/10.1109/IROS.2022.9982180)
用 LLM 改 robot trajectory。你说 "稍微往左点",LLM 解析成 trajectory edit,再用 transformer 跟原 trajectory 对齐:

$$\tau' = \tau + \text{CrossAttn}(\tau, \text{LLM}(u))$$

这是 LLM × robotics 早期工作之一,跟 [Google RT-2](https://arxiv.org/abs/2307.15818) 同源思路。

### [Wang et al., 2022 — Husformer](https://arxiv.org/abs/2209.15182)
Cross-modal transformer 做 human state recognition。核心是让一个 modality 的 latent 强化另一个:

$$z_i' = \text{SelfAttn}(z_i) + \text{CrossAttn}(z_i, z_j)$$

类似 [ViLBERT](https://doi.org/10.1109/CVPR.2019.00374) 在 vision-language 上的 co-attention 设计。

### [He et al., 2022 — M2NN](https://doi.org/10.1109/JSEN.2022.3205956)
EEG + fNIRS 融合做 motor imagery 分类。用 multi-task learning 强制两个 modality 在 shared latent 空间一致:

$$\mathcal{L} = \mathcal{L}_{\text{EEG}} + \mathcal{L}_{\text{fNIRS}} + \lambda \|W_{\text{shared}}\|_2^2$$

**直觉**:EEG 时间分辨率高但空间分辨率低,fNIRS 反过来。两个一起用,既有 "快" 又有 "准"。

---

## 我的几个 intuition

读完这篇 review,我 build 了几个 mental model:

### 1. HRI 本质是个 bandwidth matching 问题
人有五感,robot 也有对应的五感。每种 modality 有不同的:
- **Bandwidth**: visual > audio > haptic
- **Latency budget**: haptic < 1ms < audio < 100ms < visual < 33ms < dialogue < 500ms
- **Noise robustness**: haptic > audio > visual (一般情况)
- **Privacy invasiveness**: EEG > EMG > camera > mic

设计 HRI 系统就是在这些 trade-off 间找平衡。

### 2. Symmetry principle
input 和 output 是镜像的:
- Speech recognition ↔ Speech synthesis
- Gesture recognition ↔ Gesture generation
- Emotion recognition ↔ Emotion expression

这暗示 robot 应该用同一套 representation 处理双向 — 你识别 emotion 用的 latent space,应该也能 generate emotion。这跟 [VQ-VAE](https://doi.org/10.48550/arXiv.1711.00937) 的 encoder-decoder symmetry 思路一致。

### 3. Timing is everything
gesture generation 里 stroke-pitch alignment 误差 <100ms,人才能感觉"自然"。TTS 里 phoneme duration 误差大了听起来像机器人。Haptic 控制延迟 >1ms 人就觉得"卡"。

HRI 系统的 latency budget 比纯 ML 系统严格得多 — 后者慢 100ms 只是"慢",前者慢 100ms 就是"违和"。

### 4. Closed-loop affect 是被低估的金矿
大部分 paper 做 open-loop:robot recognize emotion → display emotion。少有人做 closed-loop:robot 显示 emotion → 影响 human emotion → robot 再 recognize → 调整自己的 emotion。这是真正"社交"的精髓。

[Shao et al., 2020](https://doi.org/10.3390/robotics9020044) 是少数尝试 — robot 表达 non-verbal emotion → EEG 测人 → 闭环调整。

### 5. Evaluation 是最大瓶颈
paper 里 200+ work 各自用不同 metric,没有统一 benchmark。这是 [RoboCup@Home](https://athome.robocup.org/)、[Stanford HRI Benchmark](https://doi.org/10.1145/3434073) 存在的意义,但还不够。

### 6. LLM/VLM 浪潮会重塑这个领域
这篇 review 写于 2023 中,完全没覆盖 [RT-2](https://arxiv.org/abs/2307.15818)、[Open X-Embodiment](https://arxiv.org/abs/2310.08864)、[Gemini Robotics](https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/) 这波。

2024 年后 multimodal HRI 越来越像 "VLM 当 brain + robot policy 当 body"。这篇 review 描述的 modular pipeline (ASR + NLU + dialogue mgr + NLG + TTS) 会被 end-to-end VLM 替代。但底层 sensing (force/tactile/proprioception) 依然是 modular 的,因为 VLM 主要处理 visual+language。

---

## 这篇 paper 的局限

1. **错失 LLM 浪潮**:2023 中发表,没赶上 RT-2、Gemini Robotics 这波
2. **缺定量 meta-analysis**:只统计 topic 分布,没有 "modality X 在 task Y 上平均 accuracy 多少" 这种数据
3. **跨学科 depth 不够**:neuroscience 部分偏浅,EEG 工作只引了少量
4. **real-world deployment 案例少**:大部分 paper 是 lab study,真实部署的 long-term evaluation 罕见

---

## 给你的 takeaway

如果你只记一件事:**multimodal HRI 的核心不是 "加更多 sensor",是 "让 sensor 之间对话"**。

每个 modality 单独工作都有盲区,但只要它们的 latent representation 能 cross-attend 互相校验,系统整体就比单一 modality robust 得多。这就是 cross-modal transformer 的哲学,也是 [STM3I](https://doi.org/10.1155/2022/3952758) self-tuning fusion 的动机。

进一步阅读推荐:
- [Goodrich & Schultz 2007 — HRI survey](https://doi.org/10.1561/1100000005)
- [Brohan et al., 2023 — RT-2](https://arxiv.org/abs/2307.15818)
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864)
- [Hogan 1984 — Impedance Control](https://doi.org/10.23919/ACC.1984.4788393)
- [BEAT — Gesture Generation](https://doi.org/10.1145/3478513.3480492)
- [Amershi et al., 2019 — Human-AI Interaction Guidelines](https://doi.org/10.1145/3290605.3300233)
- [RoboCup@Home](https://athome.robocup.org/)
- [VITS TTS](https://arxiv.org/abs/2106.06103)
- [FaceNet](https://doi.org/10.1109/CVPR.2015.7298682)
- [YOLO](https://doi.org/10.1109/CVPR.2016.91)
- [ST-GCN for gesture](https://doi.org/10.1109/CVPR.2018.00942)
- [PRISMA 2020](https://www.prisma-statement.org/)
- [Stock-Homburg 2022 — Robot Emotion Survey](https://doi.org/10.1007/s12369-021-00778-6)

---

# Multimodal Human–Robot Interaction: 一篇 systematic review 的深度解读

## 1. Paper meta 信息

- **Title**: *Recent advancements in multimodal human–robot interaction*
- **Authors**: Hang Su (Politecnico di Milano), Wen Qi (SCUT, 通讯), Jiahao Chen (CASIA), Chenguang Yang (Bristol Robotics Lab), Juan Sandoval & Med Amine Laribi (Pprime Institute, Poitiers)
- **Venue**: *Frontiers in Neurorobotics* 17:1084000, 2023
- **DOI**: [10.3389/fnbot.2023.1084000](https://doi.org/10.3389/fnbot.2023.1084000)
- **Type**: Systematic review, 227 篇实证论文被最终纳入分析

这是一篇典型的 PRISMA-style systematic review,核心 contribution 在于把 multimodal HRI 这个非常 fragmented 的研究领域按 **input → output → application** 三轴切分,提供一张全景图。对我而言,这篇 paper 的价值主要是 taxonomic —— 它帮我们把"robot 怎么看、怎么听、怎么说、怎么动、怎么感受人"这件事拆成可以单独优化的子模块。

---

## 2. PRISMA Methodology — 为什么这件事重要

作者遵循 [PRISMA 2020](https://www.prisma-statement.org/) 指南在 Web of Science、Scopus、ProQuest 上用 7 个关键词组合检索 2008–2022 的文献,初始 359 篇,经筛选最终 227 篇。

| Stage | 论文数 |
|---|---|
| Identified | 359 |
| Screened | 313 (剔除 editorial / conference / book) |
| Eligible | ~245 (剔除数据不足/重复标题) |
| Included | 227 |

这个 filtering 流程对 HRI 这种横跨 robotics、NLP、affective computing、neuroscience、 HCI 的学科尤其关键,因为不同社区对同一个术语(比如 "gesture"、"intention"、"engagement")定义并不一致。如果不做严格的 inclusion criteria,review 就会变成"什么都被认为相关"的杂烩。可以参照 [Page et al., 2021 (BMJ)](https://www.bmj.com/content/372/bmj.n71)。

---

## 3. 五种 Modalities 的精确定义

这是 paper 最值得 memorize 的部分。作者把 HRI 用的 modality 分成五类,我额外补上每个 modality 在 robotics 里的传感器/数学描述:

### 3.1 Audio modality
- **物理量**: 声压波形 $p(t)$, 采样后 $x[n] = p(n/f_s)$
- **pipeline**: ASR (e.g., Whisper, wav2vec2) → NLU (intent + slots) → dialogue manager → NLG → TTS (e.g., Tacotron2, FastSpeech2, VITS)
- 关键 challenge: accent, dialect, OOV words; paper 引用 [Khurana et al., 2022](https://doi.org/10.1007/s11042-022-13428-4)

### 3.2 Visual modality
- **输入**: RGB(-D) image $I \in \mathbb{R}^{H \times W \times 3}$,或 3D point cloud $\mathcal{P} \subset \mathbb{R}^3$
- **任务**: object detection (YOLO, [Redmon et al., 2016](https://doi.org/10.1109/CVPR.2016.91)), face recognition (FaceNet, [Schroff et al., 2015](https://doi.org/10.1109/CVPR.2015.7298682)), gesture recognition, gaze tracking
- **失败模式**: 光照变化、occlusion、viewpoint 变化

### 3.3 Haptic modality
- **物理量**: 力 $F \in \mathbb{R}^3$,力矩 $\tau \in \mathbb{R}^3$,触觉压力分布 $P(u,v) \in \mathbb{R}^{M \times N}$
- 通常通过 [F/T sensor ( ATI Mini40 )](https://www.ati-ia.com/)、[BioTac](https://www.syntouchinc.com/)、或 robotic skin (e.g., [Armleder et al., 2022](https://doi.org/10.1002/aisy.202100047)) 获取

### 3.4 Kinesthetic modality
- **物理量**: 关节位置 $q \in \mathbb{R}^n$、速度 $\dot{q}$、加速度 $\ddot{q}$
- 通常来自 encoder 或 IMU;反馈通过 motor command $\tau_{cmd}$

### 3.5 Proprioceptive modality
- **物理量**: 关节内部 state $(q, \dot{q}, \tau_{joint})$ 加上本体对外力的 internal estimate
- [Malinovská et al., 2022](https://doi.org/10.1109/ICDL53763.2022.9962195) 训练了一个 NN 从 proprioception 预测 touch location,即 $f_\theta: (q, \dot{q}) \mapsto \hat{t} \in \text{SE}(3)$,相当于学一个 forward sensorimotor model。

> **Intuition**: 这五种 modality 实际对应了人类感觉系统(听觉 + 视觉 + 触觉 + 本体感觉 + 运动觉),paper 暗示一个好的 HRI 系统应该 mimick 人类自己的感知-动作 coupling,而不是单一 channel。

---

## 4. Input 端: Multimodal Signal Processing

### 4.1 Computer Vision 详解

paper 列了 6 个子任务:object tracking、face recognition、gesture recognition、affective computing、gaze tracking、multi-camera。其底层都是 CNN/ViT family。可以补充一段架构直觉:

对 2D 任务,通用 backbone 是 ResNet/EfficientNet/ConvNeXt,vision tower 输出 $F \in \mathbb{R}^{H' \times W' \times C}$;检测头加 anchor-free (FCOS, CenterNet) 或 anchor-based (RetinaNet);姿态估计走 HRNet / ViTPose;gaze 走 [GazeCapture](https://doi.org/10.1145/2858036.2858119) 风格的 multi-task CNN。

对 3D 场景感知,robot 一般用 [PointNet++](https://doi.org/10.1109/CVPR.2017.16) 或 [VoxelNet](https://doi.org/10.1109/CVPR.2018.00157) 处理 LiDAR point cloud。

### 4.2 NLP pipeline

paper 把 NLP 在 HRI 中分解为 ASR、NLU、NLG、QA、dialogue management、MT。其中 NLU 通常形式化为:

$$\hat{y} = \arg\max_y P(y | x) = \arg\max_y \frac{P(x|y) P(y)}{P(x)}$$

其中 $x$ 是 utterance token sequence,$y \in \mathcal{Y}$ 是 intent label。Slot filling 则是 sequence labeling,输出 $s_{1:T}$,典型 BIO tagging:

$$P(s_{1:T} | x_{1:T}) = \prod_t P(s_t | s_{t-1}, x_t)$$

现代做法是直接用一个 transformer encoder (BERT/RoBERTa) 做 $P(y, s_{1:T} | x)$ 的 joint decoding,例如 [Bastianelli et al., 2014](https://doi.org/10.1109/TRO.2007.907484) 描述的 SLU 思路。最新趋势是 LLM in-context 解析,例如 [Bucker et al., 2022](https://doi.org/10.1109/IROS.2022.9982180) 用 LLM 把自然语言指令 reshape 成 robot trajectory。

### 4.3 Gesture recognition

paper 把手势分成 **static** vs **dynamic**。数学表达上:
- Static: $g_s \in \mathcal{G}_s$,$f_\theta: \mathbb{R}^{J \times 3} \to \mathcal{G}_s$,$J$ 是 hand joint 数 (MediaPipe 21 个,MANO 15 个)
- Dynamic: $g_d \in \mathcal{G}_d$,$f_\theta: \mathbb{R}^{T \times J \times 3} \to \mathcal{G}_d$,典型用 [ST-GCN](https://doi.org/10.1109/CVPR.2018.00942) 或 3D-CNN + LSTM ([Ur Rehman et al., 2022](https://doi.org/10.32604/cmc.2022.019586))

[skeleton model approach](https://doi.org/10.1109/TSMCC.2007.893280) 的好处是 dimensionality 低(只关心 joint 而非像素),且对 texture/lighting 不变。

### 4.4 Emotion recognition — 一个三视角分类法

paper 引用 [Salovey & Mayer, 2004](https://doi.org/10.1017/CBO9780511806582.019) 的 emotional intelligence 概念,把 HRI 情绪研究分成:

1. **Robot internal psychological state** — 给 robot 一个内部情感 state $e_r(t) \in \mathcal{E}$,通常用 [OCC model](https://doi.org/10.1145/77739.77740) 或 [PAD](https://doi.org/10.1109/TAF.2005.18) 三维 pleasure-arousal-dominance 空间
2. **Robot emotional expression** — 输出端,生成 facial expression / voice prosody / body posture
3. **Human emotion recognition** — 输入端,从 multimodal signals $x = (x_{face}, x_{voice}, x_{physio})$ 估计 $y_{emo}$

paper 还区分 **internal signals** (EEG, GSR, BVP/ECG) 和 **external signals** (face, body, voice)。Internal 通常被认为 more objective 但更 noisy 且可穿戴性差;external 容易采集但易被伪装。这正是 multimodal fusion 的动机:用 $\{(x_i, \sigma_i)\}$ 各自的 reliability 加权。

---

## 5. Output 端: Multimodal Feedback

### 5.1 Speech Synthesis (TTS)

paper 描述的 pipeline 是:
$$\text{text} \xrightarrow{\text{NLG}} \text{normalized text} \xrightarrow{\text{frontend}} \text{phoneme + prosody} \xrightarrow{\text{vocoder}} \text{waveform}$$

现代 neural TTS 的损失 (e.g., [VITS](https://doi.org/10.48550/arXiv.2106.06103)) 大致是:

$$\mathcal{L}_{\text{VITS}} = \mathcal{L}_{\text{recon}} + \lambda_1 \mathcal{L}_{\text{adv}} + \lambda_2 \mathcal{L}_{\text{FM}} + \lambda_3 \mathcal{L}_{\text{dur}} + \lambda_4 \mathcal{L}_{\text{pitch}} + \lambda_5 \mathcal{L}_{\text{energy}}$$

其中:
- $\mathcal{L}_{\text{recon}}$: mel-spectrogram reconstruction
- $\mathcal{L}_{\text{adv}}$: GAN 判别器损失
- $\mathcal{L}_{\text{FM}}$: flow matching / monotonic alignment
- $\mathcal{L}_{\text{dur}}, \mathcal{L}_{\text{pitch}}, \mathcal{L}_{\text{energy}}$ 三个 suprasegmental feature 的预测 loss

HRI 场景下还需 expressive TTS,即给一句话加上"疑问/安慰/兴奋"等 emotional style。

### 5.2 Visual feedback

paper 把视觉反馈拆成 7 类:status、error、wayfinding、object highlight、emotion、multi-camera、conversation display。这本质上是 robot → human 的 **visual communication design** 问题,跟 UX 设计耦合很强。

### 5.3 Gesture generation — 重点章节

paper 提到 **Articulated Communicator Engine (ACE)** framework ([Kopp & Wachsmuth, 2004](https://doi.org/10.1002/cav.6))。关键 insight 是 gesture 的 **timing** 必须和 speech 的 prosody 对齐:

$$\text{align}(g, s) = \{(\tau^{g}_{\text{prep}}, \tau^{g}_{\text{stroke}}, \tau^{g}_{\text{hold}}, \tau^{g}_{\text{retract}}), (\tau^{s}_{\text{onset}}, \tau^{s}_{\text{peak}}, \tau^{s}_{\text{offset}})\}$$

其中:
- $\tau^{g}_{\text{stroke}}$: gesture 最强调的瞬间(语义 peak)
- $\tau^{s}_{\text{peak}}$: 语音中重音或 pitch accent 的瞬间

[McNeill 的 Kita model](https://press.uchicago.edu/ucp/books/book/chicago/G/bo3623559.html) 指出 stroke 通常和 pitch accent 同步,误差 < 100ms。

[Chae et al., 2022](https://doi.org/10.1016/j.robot.2022.104154) 提出了一个 morpheme-based 方法,先用 morphemic analysis 把句子切成语义单元,再从 motion primitives 库里检索对应 gesture;report 了 83% 的 expression unit / gesture type 准确率。这跟 modern LLM-driven gesture generation (e.g., [BEAT](https://doi.org/10.1145/3478513.3480492)) 形成对比,后者用 transformer 端到端从 audio + text 生成 joint trajectory:

$$\hat{Q}_{1:T} = \text{TransformerDec}(z_{1:T}^{audio}, z_{1:T}^{text}, q_0)$$

### 5.4 Emotional expression generation

paper 引用 [Stock-Homburg, 2022](https://doi.org/10.1007/s12369-021-00778-6) 强调 **smooth mood transition** 比 one-to-one emotion→action mapping 更重要。这可以用 Markov 过程建模:

$$P(e_t | e_{t-1}) = \text{softmax}(W_e e_{t-1} + b_e)$$

或更复杂的 ODE: $\dot{e} = f(e, u, t)$,其中 $u$ 是外部刺激。([Rincon et al., 2019](https://doi.org/10.1007/s10115-018-1231-9) 在葡萄牙 day-care center 做了真实部署。)

[Shao et al., 2020](https://doi.org/10.3390/robotics9020044) 的 setup 很有意思:用 robot 表达 non-verbal emotion 来 elicit user 的 affect,再用 EEG 直接测量 — 用 MLP 和 SVM 对比,这是 closed-loop emotion interaction 的雏形。

### 5.5 Multimodal feedback fusion

paper 列了 6 类融合输出。从信息论角度,这其实是 channel capacity 利用率最大化的问题。设 $I(X; Y_i)$ 是第 $i$ 个 channel 的互信息,理想情况下:

$$I(X; Y_1, Y_2, \ldots, Y_n) = H(X) - H(X | Y_1, \ldots, Y_n) \geq \max_i I(X; Y_i)$$

所以只要 channels 是 conditionally independent 给定 $X$,多 modality 严格 better。

---

## 6. Applications — 4 大落地场景

### 6.1 Industrial cobots

paper 提到 [Industry 4.0](https://www.iso.org/standard/74579.html) 框架下 cobot 需要 sensing 多样化。常见的传感器分类:

| 传感器 | 用途 | 典型型号 |
|---|---|---|
| Vision sensor | object recognition, hand tracking | [Kinect Azure](https://azure.microsoft.com/en-us/products/kinect-dk), RealSense D435 |
| Tactile sensor | collision detection, grip force | [Robotiq 2F-85](https://robotiq.com/products/2f85-140) |
| Torque sensor | collision classification | 内置 (KUKA iiwa, Franka Panda) |
| Proximity sensor | safety envelope | Ultrasonic, capacitive |
| Encoder | joint state | 绝对/增量式 magnetic/optical |

[Tang et al., 2015](https://doi.org/10.1108/IR-03-2015-0059) 在 noisy 工厂环境下用 vision-based static pose recognition 替代 ASR,这是个很 pragmatic 的工程选择。

### 6.2 Assistive mobile robots

全球 ~65M wheelchair user([Desai et al., 2017](https://doi.org/10.1109/ICNTE.2017.7947914))。智能 wheelchair 用多 modality 输入:
- joystick (default)
- speech ([Sharifuddin et al., 2019](https://doi.org/10.1109/AiDAS47888.2019.8970865) — CNN-based)
- face/eye gaze ([Rabhi et al., 2018a](https://doi.org/10.1016/j.cmpb.2018.08.013))
- EMG ([Kumar et al., 2019](https://doi.org/10.1007/978-981-15-0111-1_8))
- EEG ([Zgallai et al., 2019](https://doi.org/10.1109/ICASET.2019.8714373))

### 6.3 Robotic exoskeletons

paper 区分 **cHRI** (cognitive HRI, 用 EEG/EMG 解码 intent) 和 **pHRI** (physical HRI, 用 force/position 作为控制输入)。

[Hogan 1984 impedance control](https://doi.org/10.23919/ACC.1984.4788393) 的核心公式:

$$F_{ext} = M_d (\ddot{q} - \ddot{q}_d) + B_d (\dot{q} - \dot{q}_d) + K_d (q - q_d)$$

其中:
- $M_d, B_d, K_d \in \mathbb{R}^{n \times n}$ 是 desired inertia, damping, stiffness
- $q, \dot{q}, \ddot{q}$ 是 actual joint position / velocity / accel
- $q_d, \dot{q}_d, \ddot{q}_d$ 是 desired trajectory
- $F_{ext}$ 是 robot 作用于环境的力

这在 exoskeleton 设计里是基础公式 — 调小 $K_d$ 让 robot 更 compliant,调大 $B_d$ 抑制 oscillation。[Klauer et al., 2014](https://doi.org/10.3389/fnins.2014.00262) 用 NMES + lockable passive exoskeleton 做重力补偿。

### 6.4 Robotic prosthesis

控制核心是 **pattern recognition on EMG**。给定 $N$ 通道 sEMG 信号 $X \in \mathbb{R}^{T \times N}$,目标分类 grasp pattern $y \in \{1, \ldots, K\}$:

$$\hat{y} = \arg\max_k P(k | X) = \arg\max_k \text{softmax}(W_o \text{CNN}(X))$$

[Ameri et al., 2018](https://doi.org/10.1371/journal.pone.0203835) 用 CNN 直接从 raw EMG 端到端做 simultaneous myoelectric control,跳过经典 feature engineering (Mav, WAV, ZC, SSC, WL)。

[Zhang et al., 2022](https://doi.org/10.1145/3523286.3524576) 用 MYO wristband 提取 active element,然后做 5 个时域特征:
1. RMS (Root Mean Square): $\sqrt{\frac{1}{T}\sum_t x_t^2}$
2. Waveform length: $\sum_t |x_t - x_{t-1}|$
3. Zero crossings: $\#\{t : (x_t - x_{t-1})(x_{t+1} - x_t) < 0\}$
4. Mean Absolute Value: $\frac{1}{T}\sum_t |x_t|$
5. Max-Min diff: $\max x - \min x$

---

## 7. 一些值得注意的最新工作

paper Section 5 提到一堆 2022 年的最新工作,我挑几个对 intuition 建设有帮助的:

### 7.1 [Hou et al., 2022 — STM3I](https://doi.org/10.1155/2022/3952758)
**Self-tuning Multimodal Fusion** 的思想是让 fusion 权重 $w_i(t)$ 随时间/上下文自适应,即:

$$\hat{y} = \sum_i w_i(t) f_i(x_i)$$

其中 $w_i(t) = \text{softmax}(g_\phi(\text{context}_t))$。这解决了 "在 noisy factory speech 没用、gesture 主导;在医院 speech 是主、gesture 辅助" 这种 dynamic 权重问题。

### 7.2 [Bucker et al., 2022 — LLM Trajectory Reshaping](https://doi.org/10.1109/IROS.2022.9982180)
用 LLM 把 "move the cup a bit to the left" 这种自然语言解析成 trajectory edit,再用 multimodal focus transformer 把它和原 trajectory 对齐:

$$\tau' = \tau + \text{CrossAttn}(\tau, \text{LLM}(u))$$

其中 $u$ 是语言指令,$\tau$ 是原 trajectory。这是 LLM × robotics 早期工作之一,和 [Google RT-2](https://doi.org/10.48550/arXiv.2307.15818)、[NOIR](https://doi.org/10.48550/arXiv.2310.03186) 思路同源。

### 7.3 [Wang R. et al., 2022 — Husformer](https://doi.org/10.48550/arXiv.2209.15182)
**Cross-modal transformer for human state recognition**,核心是用一个 modality 的 latent 强化另一个:

$$z_i' = \text{SelfAttn}(z_i) + \text{CrossAttn}(z_i, z_j)$$

类似于 [ViLBERT](https://doi.org/10.1109/CVPR.2019.00374) 在 vision-language 上的设计,但用 human state。

### 7.4 [He et al., 2022 — M2NN](https://doi.org/10.1109/JSEN.2022.3205956)
EEG + fNIRS multimodal motor imagery 分类。用 multi-task learning:
$$\mathcal{L} = \mathcal{L}_{\text{EEG}}(W_{\text{shared}}, W_{\text{EEG}}) + \mathcal{L}_{\text{fNIRS}}(W_{\text{shared}}, W_{\text{fNIRS}}) + \lambda \|W_{\text{shared}}\|_2^2$$

通过 shared representation 强制两个 modality 在 latent 空间一致,提升 generalization。

### 7.5 [Armleder et al., 2022 — Robotic Skin](https://doi.org/10.1002/aisy.202100047)
大规模 robotic skin 提供 dense tactile feedback $T \in \mathbb{R}^{M \times N \times C}$,$M \times N$ 是 taxel 网格,$C$ 是 channel (force, temp, etc)。这种 skin 让 pHRI 可以做 whole-body collision-aware control。

### 7.6 [Strazdas et al., 2022 — ROSA](https://doi.org/10.3390/s22030923)
融合 voice + face + gesture 的 non-contact HMI,大样本 user study 是亮点 — 因为很多 paper 只在 lab 里跑几个 subject,ROSA 做了大规模 evaluation。

---

## 8. Discussion & Future directions

paper 最后讨论了几个 core challenge:

1. **NLP 的训练数据 bias** — 罕见语言/方言性能差
2. **CV 的环境敏感性** — 光照、occlusion、appearance 变化
3. **ML 的 generalization** — domain shift
4. **Haptic 反馈精度** — 仍达不到 human-touch 级别

paper 提到的未来方向:
- Emotion recognition + generation 的 tighter integration (closed-loop affective HRI)
- Multi-robot systems(多 robot 协作 HRI)
- Personalized learning — 让 robot 适应 individual communication style
- 伦理考量 — [IEEE Ethically Aligned Design](https://standards.ieee.org/industry-connections/ec/autonomous-systems/) 是个好起点

paper 没说但我觉得重要:**LLM-as-HRI-brain** 在 2023 之后开始 dominate,如 [Open X-Embodiment](https://doi.org/10.48550/arXiv.2310.08864)、[RT-X](https://doi.org/10.48550/arXiv.2310.08864),已经把 multimodal HRI 从 "modular pipeline" 推向 "VLM end-to-end policy"。这篇 2023 年中发表的 review 还没来得及反映这一波浪潮。

---

## 9. 我的 takeaways (Intuition building)

读这篇 review 后,我会这样 frame multimodal HRI:

1. **Modality 是 interface 不是本质**。Audio/visual/haptic 只是 robot 与 human 之间"带宽"和"延迟"不同的 channel。真正的工作量在 **representation fusion**。

2. **Symmetry 是设计原则**。Paper 强调 input (recognition) 和 output (generation) 是镜像的:gesture recognition 对 gesture generation,emotion recognition 对 emotion generation,speech recognition 对 TTS。一个完整 HRI 系统必须同时具备两端。

3. **Closed-loop affect** 是被严重低估的方向。Robot 不止 recognize human emotion,还要在内部维护自己的 affective state,再生成 expression。这跟 RL 中的 reward shaping 和 intrinsic motivation 有强 connection ([Oudeyer & Kaplan, 2009](https://doi.org/10.1155/2009/615398))。

4. **Latency budget** 是隐藏约束。HRI 中各 modality 的 acceptable latency 不同:
   - Haptic: < 1 ms (mechanoreceptor 频率上限 1 kHz)
   - Audio: < 100 ms (phoneme boundary)
   - Gesture: < 200 ms (gesture stroke alignment)
   - Visual: < 33 ms (30 FPS)
   - Dialogue: < 500 ms (turn-taking)

5. **Evaluation metric 是 HRI 的阿喀琉斯之踵**。Paper 引用的 200+ paper 大多各自有不同的 metric,缺乏 unified benchmark。这跟 [Stanford HRI Benchmark](https://doi.org/10.1145/3434073) 或 [RoboCup@Home](https://athome.robocup.org/) 这类 community benchmark 是未来方向。

---

## 10. 进一步阅读建议

如果想要 build 更深的 intuition,推荐补这几篇:

- [Goodrich & Schultz, 2007 — HRI survey (Foundations & Trends)](https://doi.org/10.1561/1100000005) — 经典综述,定义了 HRI 的 taxonomy
- [Lemaignan et al., 2017 — Psychology of HRI](https://doi.org/10.1145/2909824) — cognitive 视角
- [Mavridis, 2015 — HRI taxonomy review](https://doi.org/10.1016/j.robot.2014.08.017)
- [Brohan et al., 2023 — RT-2](https://doi.org/10.48550/arXiv.2307.15818) — VLM-driven HRI 新范式
- [Padmakumar et al., 2024 — HumanoidBench](https://doi.org/10.48550/arXiv.2405.17069) — humanoid 仿真 benchmark
- [Amershi et al., 2019 — Guidelines for Human-AI Interaction (CHI)](https://doi.org/10.1145/3290605.3300233) — 18 条 HAI 设计原则
- [Breazeal, 2003 — Sociable Machines](https://direct.mit.edu/books/monograph/2339/Sociable-Machines) — Kismet 鼻祖工作

---

## 11. paper 的局限

老实说这篇 review 有几个短板:

1. **2023 中年发布,错过了 LLM/VLM 浪潮** — 大量 multimodal HRI 工作现在直接用 GPT-4V / Gemini / open VLM 做 policy backbone,这篇 review 完全没碰这块。
2. **Evaluation 数据匮乏** — 没有定量 meta-analysis,只有 topic 分布 table,缺乏 "modality X 在 task Y 上的平均 accuracy" 这种 quantitative aggregation。
3. **跨学科 depth 不够** — neuroscience 和 HCI 的部分偏浅,可以补 [Pfurtscheller](https://doi.org/10.1016/S1388-2457(01)00497-5) 的 EEG work、[Pan et al., 2020](https://doi.org/10.1038/s42256-020-0181-5) 的 affective BCMI 综述。
4. **No real-world deployment data** — paper 自己也承认 "limited number of real-world deployments" 是局限。

---

总结一句:**这篇 paper 是一个 2023 年中视角下的"multimodal HRI 地图",值得作为 entry point,但需要叠加 2023 年底以后的 VLM-driven robotics 工作来补全**。对你 build intuition 而言,我建议聚焦在 (a) impedance control 在 pHRI 的角色、(b) cross-modal transformer fusion 机制、(c) gesture-audio alignment timing,这三个点都是把 multimodal HRI 工程化的"first-class citizen"。

参考链接:
- Paper 本体: [Frontiers Neurorobotics 17:1084000](https://doi.org/10.3389/fnbot.2023.1084000)
- PRISMA: [prisma-statement.org](https://www.prisma-statement.org/)
- Hogan impedance: [ACC 1984](https://doi.org/10.23919/ACC.1984.4788393)
- ACE engine: [Kopp & Wachsmuth 2004](https://doi.org/10.1002/cav.6)
- RT-2: [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)
- Open X-Embodiment: [arXiv:2310.08864](https://arxiv.org/abs/2310.08864)
- FaceNet: [CVPR 2015](https://doi.org/10.1109/CVPR.2015.7298682)
- YOLO: [CVPR 2016](https://doi.org/10.1109/CVPR.2016.91)
- VITS: [arXiv:2106.06103](https://arxiv.org/abs/2106.06103)
- BEAT (gesture generation): [ACM MM 2021](https://doi.org/10.1145/3478513.3480492)
- IEEE Ethically Aligned Design: [standards.ieee.org](https://standards.ieee.org/industry-connections/ec/autonomous-systems/)
- Stanford HRI Benchmark: [dlr/rm/SHR](https://doi.org/10.1145/3434073)
- RoboCup@Home: [athome.robocup.org](https://athome.robocup.org/)
