---
source_pdf: Ego-Exo4D.pdf
paper_sha256: 15e31586e111eb32fe54131840b7733b095fb3b20bebb14be9610d11967e26a8
processed_at: '2026-08-04T02:25:48-07:00'
target_folder: World-model/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Ego-Exo4D 的人话版

Andrej，咱就跳过 paper 八股，直接讲这玩意儿到底在干嘛、为啥重要、哪些设计聪明、哪些地方暴露了问题的本质难度。

## 一句话说清这 dataset 是干嘛的

你戴一副 Aria 眼镜做菜、弹琴、爬墙、修自行车，旁边架 4 台 GoPro 拍你。5 个视角全部同步，录 1-42 分钟一段。740 个人、13 个城市、43 种任务、1286 小时。然后请 52 个真专家（教练、主厨、攀岩指导）看视频给口头点评："这脚法不错但重心偏了"、"胡椒应该现磨不是用粉"。再加上你自己的边做边讲、第三方标注员的逐动作描述。三种文字 + 5 个视角 + Aria 全套传感器（IMU、眼动、7 麦克风、SLAM 点云、6DoF）。

就这么个东西。它不是 Ego4D 的 v2，它根本是另一个物种——Ego4D 是"生活的随机切片"，Ego-Exo4D 是"有技能的人认真干活"。

## 为啥要做这件事 —— 真正的 motivation

paper 里写得很学术，但底层 motivation 其实特别朴素：**人是怎么从"看别人做"学到"自己会做"的？**

你小时候看爸爸切洋葱，你脑子里的 representation 是第三人称的——他的手、他的刀、洋葱在砧板上。然后你自己拿起刀，视角变成第一人称——你的手、刀在你眼前。这中间有一个非常厉害的 cross-view translation，人类婴儿期就会，但 AI 完全不会。

现有的 video dataset 全都是单视角的。Ego4D 只有 ego，HowTo100M 只有 exo（YouTube 教学视频）。这就好比你只让模型看左边或右边一只眼睛，它永远学不会深度知觉。Ego-Exo4D 第一次大规模把两个视角同步录下来，让模型有机会学那个 translation。

更深一层：技能这个东西，在视频里是"how"而不是"what"。你说"他切洋葱"——这是 what，谁都能标注。但"他切洋葱时刀刃角度不对、手腕没放松、节奏不稳"——这是 how，只有同行专家看得出来。这就是为啥要请 52 个真专家点评，而不是像 Ego4D 那样雇众包标注员。这是这 dataset 最稀有的资产。

## 摄像头 setup 的几个聪明设计

**Aria 眼镜 75g**。这个数字很关键。一个 GoPro 150g+，戴头上跳舞或攀岩动作就变形了，数据就不自然。75g 让你几乎忘掉它存在，能录到真实技能表现。

**时间同步用 QR code 玩帧率差**。手机放 29fps 的 QR 序列给所有相机看，Aria 30fps，GoPro 59fps。因为帧率不一样，两个连续帧上看到同一个 QR 的概率很低，一旦发生就能把曝光中心时刻定位到 ±0.574ms。这个精度是后续做 3D triangulation 的前提——sync 错 16ms，人体关节就能差好几厘米。70% 的 take 自动跑通，剩下 30% 靠人工对"打火机第一次火花"、"球鞋触球瞬间"这类可定位事件。这种混合方案很实用——纯硬件 timecode 要接 LTC 到 audio 通道，会丢掉 stereo audio 这个模态。

**GoPro 也做 6DoF localization**。Aria 自己跑 SLAM 出 6DoF + 点云，但 GoPro 是静态的，没运动没法直接 SLAM。他们在 Aria 建的 map 上做 P4P + RANSAC 给 GoPro 定位，91.4% 成功率。失败的是 GoPro 对着白墙白桌——没纹理就没 feature。这是个很有意思的失败模式，说明 future rig 应该在 exo camera 选位上考虑纹理丰富度。

## 三种文字标注，是这 dataset 的灵魂

这点我想多讲点，因为 paper 里写得比较散，但直觉上这里藏着宝藏。

**Expert commentary**：52 个专家，平均每人 10+ 年经验 + 有教学经历。看视频，每分钟暂停 7 次，每次口述 4 句话，说"这动作好/不好在哪、为什么、怎么改"。还配 1-10 分 proficiency 评分 + 在视频上画红圈红箭头标身体部位。总共 117,812 段时间戳文字，6000 小时人工。

这件事以前没人做过。为啥？因为贵——你得找真专家，给真钱，花真时间。Ego4D 雇大学生标 "C picks up cup"，那便宜得多，但标不出"他拿杯子的手势暗示他不是厨师"这种细节。

expert commentary 的本质是把**技能的隐式知识**转成显式语言。一个篮球教练看一眼投篮就知道"手肘外翻 5 度"，你问他怎么知道的他也说不清，就是看出来的。现在我们有了这种"看出来"的语言化版本，可以训练模型也学会这种感知。

**Narrate-and-act**：你自己边做边讲，像 YouTube 教程但没后期。10% 的 take 才有，因为大部分要无打扰自然做。这是第一人称的"what + 为什么"。

**Atomic action descriptions**：第三方标注员写的简短动作描述，每句一个动词，时间戳 ±1 秒，432K 句。"C picks up wrench" 这种。Ego4D 的传统做法，但这里多了个关键标签——每条标注还标"这个动作在 ego 可见吗"、"哪个 exo 视角看得最清楚"。这个 visibility 标签后来被其他所有 benchmark 复用作 exo 视角选择依据，是个小而关键的设计。

三种文字放一起你就能比较：
- Expert：大词表、长句、低密度（7/分钟），讲 how
- Narrate-and-act：中词表、中句、中密度，讲 what + why
- Atomic：小词表、短句、高密度（31/分钟），讲 what

这三种文字密度和风格的差异本身就是研究素材。你可以用 atomic 做 video-language pretraining（量大、干净），用 expert 做 proficiency estimation（有评判），用 narrate-and-act 做 how-to generation（第一人称教程风格）。

## 四大 benchmark，挨个说人话

### 1. Correspondence —— 在两个视角里认出同一个东西

你切洋葱，ego 视角里洋葱在你手边 200 像素大。exo 视角里洋葱在桌子上 10 像素大、还可能被你身体挡住。给模型 ego 里的洋葱 mask，让它在 exo 每一帧找出同一个洋葱的 mask。

这个任务为什么难：
- 同一个物体在两个视角下外观差距巨大（近场 vs 远场）
- exo 里物体可能就几个像素，几乎看不见
- 你身体经常挡住 exo 视线
- 物体状态会变（整洋葱 → 切开的洋葱 → 切碎的洋葱，算 3 个 instance）

baseline 用 XSegTx（基于 SegSwap 的 cross-view segmentation transformer）和 XView-XMem（基于 XMem 的跨视角跟踪，喂 interleaved ego-exo 帧序列）。最好的 ST baseline IoU 也就 34.9%（Ego→Exo），25%（Exo→Ego）。任务基本没解决。

我的直觉：**这个问题本质上需要物体级别的 3D 理解**。你看到 ego 里洋葱的纹理 + 你知道 ego 相机的 6DoF（Aria 给了）+ 你看到 exo 的场景，应该能反推出洋葱在 exo 像面上的投影位置。但 benchmark 故意把 camera pose 排除在输入之外，逼模型从纯视觉学。这其实有点矛盾——实际 AR 眼镜是有 SLAM pose 的，不用白不用。我猜未来 winning method 一定会偷偷用几何信号，或者从视频里隐式估计出几何信号。

### 2. Translation —— 给 exo 生成 ego

更激进的版本：给 5 帧 exo clip + 物体 mask，要模型生成 ego 视角下那个物体区域的 RGB。

拆成两个子任务：
- Track prediction：先预测物体在 ego 帧里的 mask（在哪、什么形状）
- Clip generation：给定 GT ego mask，生成那个区域的 RGB 像素

第二阶段用 DiT（diffusion transformer）压倒 pix2pix（GAN）。SSIM 0.59 vs 0.42，LPIPS 0.46 vs 0.50，CLIP 相似度 81.9 vs 79.8。Diffusion 在这任务上完胜。

直觉：**这任务的瓶颈不在生成模型，在 condition**。物体在 exo 里太小，DiT 要"想象"出 ego 里手指尖接触洋葱皮的纹理，这几乎不可能从 exo 单帧推出来。必须用 exo object crop（放大物体区域）和 ego crop mask（告诉形状朝向）作为强 condition。Ablation 证实：缺 exo object crop → 颜色纹理全错；缺 ego crop mask → 朝向全错。

这个任务让我想起 NeRF 的极端情况——视角差 180 度、物体小、要补全完全看不见的部分。纯几何方法完全失败，必须靠语义先验。未来可能要走 diffusion + 3D-aware condition 的路。

### 3. Keystep recognition —— 你在做哪一步

17 个 procedural activity、664 个 keystep、143K segments。Trimmed clip 分类。训练有 ego+exo，测试只有 ego RGB。

这个任务最值得讲的是 baseline 比较给出的 insight：

| 方法 | Test Acc |
|---|---|
| TimeSformer (K600) ego-only | 35.93 |
| TimeSformer (K600) ego+exo train | 31.04（掉了！） |
| EgoVLPv2 ego-only | 37.72 |
| EgoVLPv2 ego+exo train | 38.76（涨了） |
| VI Encoder（显式对比学习 view-invariant） | **41.53** |
| Viewpoint distillation | 39.49 |

这里有个非常深刻的反直觉：**直接加 exo classification loss 反而掉点**。因为 exo 视角下 keystep 太好认了，分类 loss 一拉，模型表征就往 exo-friendly 方向偏，反而离 ego 任务更远。

但 EgoVLPv2 用 contrastive video-language pretraining 加 exo 反而涨点，VI Encoder 用 clip-level contrastive loss（同步 ego-exo 对为正，非同步为负）效果最好。这说明：**exo 视角的正确用法是做 representation learning 的辅助 signal，不是做 task supervision 的额外 head**。

这个 insight 我觉得很值得推广到其他 multi-view training 场景：训练时多视角的价值在于约束 representation space 的几何结构（什么视角不变量是任务相关的），而不是增加 label 信号密度。

### 4. Procedure understanding —— 任务结构推断

给定当前 keystep 和历史，预测：之前的 keystep、是否 optional、是否 mistake、缺了哪些、接下来可以做什么。

用 task graph（有向图 + OR/XOR + optional/repeatable）表示 procedure 结构。从训练集统计 transition probability P(B|A) 自动建图，再人工修正。

测试结果让我惊讶的是 future keystep 预测最难。用 GT 标签的 graph-based：prev 82.32 cAP、missing 82.63、mistake 73.06，但 future 只有 62.82。原因显然：future 是发散的——做完"打蛋"后可以"加盐"也可以"加牛奶"也可以"倒油"，多模态分布，单纯从共现统计算不出来。要做对 future 必须理解物体的当前状态（碗里有没有油、锅热没热），这是当前 graph-based method 完全忽略的。

直觉：**future prediction 需要 state estimation，不只是 transition 统计**。这是 procedural AI 的下一个前沿。Ego-Exo4D 给了 object mask + 多视角，理论上可以做 object state 识别（碗的纹理变化、锅的烟雾），但 baseline 没人试。这是低垂的果实。

### 5. Proficiency estimation —— 你做得好不好

两个变体：
- Demonstrator：整个 demo 的水平，4 类分类
- Demonstration：视频内时间点定位，找出"做得好"和"需改进"的时刻

proficiency 的 label 怎么来？很巧妙：用 participant 自填问卷（多少年经验、教过别人没、有证书没）+ expert 评分 1-10，scenario-specific 转换函数映射到 4 类。比如 basketball 按年经验分（<1 novice, 1-3 early, 3-10 intermediate, ≥10 late），bouldering 按最高完成难度（V3 以下 novice ... V6+ late），cooking 按 expert 平均分（<3.5 novice ... ≥8 late）。

这个 label pipeline 暴露了个本质困难：**技能水平是多维的**。一个厨师可能刀工 9 分但调味 6 分。简单 4 类根本不够。但作为 benchmark 起点，4 类至少让模型先学会区分新手和老手。

baseline 用 TimeSformer。Ego 视角在 cooking 上好（手-物近场信息重要），Exo 视角在 bouldering 上好（body pose 信息重要）。EgoVLPv2 预训练 + ego 视角达 50.4% test acc，比 random 26.4% 高不少。Late fusion 不提升——简单平均 softmax 太弱。

Demonstration 那个时间点定位 task 用 ActionFormer 改造做 timestamp regression，mAP 只有 3-4%。这个数字意味着任务基本刚起步。原因：expert commentary 的时间戳很稀疏（每分钟 7 次），定位到 0.25 秒内极难。

直觉：**proficiency estimation 的真正数据是 expert commentary 文本本身**。当前 baseline 只用 video 特征，完全没用语言。如果训练一个 video-conditioned LLM 读 expert 文字、看视频片段，学"什么样的视觉特征对应什么样的 critique"，应该能大幅提升。这相当于让模型从专家那里学审美直觉。

### 6. Ego pose —— 从 ego 视频恢复全身 3D 姿态

这任务特别有意思，因为 ego 视频里你自己身体大部分根本看不见——你看不到自己的腿、看不到自己后背、连胳膊都经常出 FoV。要从这种极不完整的信息推断全身 17 个关节的 3D 坐标。

Annotation pipeline 很聪明：
1. 在 4-5 个 exo 视角用 MMPose 检测 2D bbox（用 Aria MPS 的 3D 头戴位置投影选 camerawearer 的 bbox，解决多人场景问题）
2. 检测 2D keypoint
3. 多视角 RANSAC triangulation 得 3D keypoint
4. 自动生成 9.2M 帧（pseudo-GT），人工修正 376K 帧（GT）

auto GT 与 manual GT 的 MPJPE 只差 3.33cm，远小于 baseline 误差，说明 auto GT 可信，能把训练数据从 376K 扩到 9.2M。

baseline 结果：
- Static pose（用平均 pose 平移）：215 cm 误差（baseline）
- Kinpoly（物理 simulator humanoid 跟踪 head pose）：24.36 cm
- EgoEgo（DROID-SLAM 给 head pose + diffusion 生成 body motion，训练在 AMASS）：26.38 cm
- Location-based（transformer 从 3D head 位置序列直接回归 joint 位置）：18.51 cm

Location-based 最简单但最好，这个有点反直觉。我的解读：**当前的 diffusion-based 方法（EgoEgo）受限于 AMASS 训练分布**，AMASS 是 mocap 数据，主要是行走、跑步等日常动作，没有 cooking、bike repair 这些技能动作的分布。所以 EgoEgo 在 cooking 上失败（33.81 cm），Location-based 用 task-specific 数据训练反而占优（15.00 cm）。

CVPR 2024 challenge 冠军用 multi-scale model fusion ensemble，把 Location-based 误差压到 15.32 cm。但这是工程 ensemble，没本质突破。

hand pose 任务类似，但每只手 21 个 joint。POTTER（Pooling Attention Transformer）在 manual+auto 数据上 PA-MPJPE 11.07mm，是当前 SOTA。可见视角越多 GT 越准（Table 15b：3 views 14mm, 6 views 10mm）。拇指和指尖误差最大——最常被物体挡。

直觉：**ego pose 的根本瓶颈是 FoV**。Aria 110° FoV 看不到脚下、看不到大腿。要解决要么加 camera（背部、腰部），要么用 strong prior 从 head pose + IMU 推断下半身。后者在 physics-based model（Kinpoly）里有雏形，但当前 physics model 的 realism 不够。

## 一些我自己看完 paper 的联想

**1. Expert commentary 是 LLM 时代的金矿**

117K 段时间戳文字 + 视频片段，每段是 expert 对 skill 的精细评判。这相当于一个大规模的"video-grounded skill critique"语料库。可以训练一个 model：给视频片段 → 输出 critique 文字。这是 video-to-text 的精细化版本，比 "describe what happens" 难得多，需要"看出"动作的微妙差异。

更进一步：用 Llama 3 或 GPT-4V 做 zero-shot critique，跟 human expert 对比，看模型能不能看出 "手肘外翻" 这种细节。我赌当前 VLM 完全做不到，因为这种 fine-grained skill perception 不在它们的训练分布里。Ego-Exo4D 给了评测这个的 benchmark。

**2. Proficiency estimation 的真正应用是 life-long learning**

想象一个人戴 AR 眼镜学吉他，每天都练 1 小时。系统能持续追踪他的 proficiency score 随时间变化，给出"你这周的换和弦速度提升了 20%、但节奏稳定性退步了"这种反馈。这需要 temporal span 远超 Ego-Exo4D 当前 take 的 1-42 分钟。未来的 longitudinal skill tracking dataset 会是下一步。

**3. Ego-exo 的不对称性**

paper 强调 ego+exo 互补，但实际应用里 exo 通常没有——你不会在自家厨房架 4 个 GoPro。所以"训练用 exo、测试只用 ego"是正确范式，VI Encoder 的成功印证了这一点。但更进一步：**能不能用 internet 上的 exo instructional video 作为训练时的 exo 来源，让它和 ego 训练数据 unsupervised 配对？** 这需要 cross-video retrieval + temporal alignment，Xue & Grauman 2023 在 NeurIPS 已经开始探索。

**4. Energy 公式应该成为标准**

$$
E(O^t) = \alpha C + \beta M + \sum \gamma_j \mathbb{1}(S_j)
$$

这个 $\alpha C + \beta M + \gamma S$ 三项式应该成为所有 always-on 视觉模型的报告标准。当前 paper 只算 FLOPs 是耍流氓——memory transfer 在实际硬件上能占 >50% 能耗。Ego-Exo4D 第一次把这个搬到主流 benchmark 里，希望后续 dataset 都跟上。

**5. Hand pose 6 个视角 vs 3 个视角差 4mm**

这个数字看着小，实际意义重大。它说明 Ego-Exo4D 的 5 视角 setup 在 hand pose GT 上已经接近饱和——再加视角收益递减。这意味着未来 hand pose 数据集不需要更多视角，需要的是更多场景多样性、更复杂的手-物交互。Ego-Exo4D 已经做到了。

**6. Skill 的连续性 vs 离散性**

当前 proficiency 用 4 类分类，但 expert 评分是 1-10 连续的。技能到底是连续进步还是阶段性跃迁？从 expert commentary 看，有时候一个细节改了（"手腕放松"）就让整体水平上一个台阶。这暗示 skill space 可能有 manifold 结构，某些方向是 fast axis。用 continuous score 训 regression 而不是分类，可能学到更精细的 skill embedding。

**7. Translation 任务的真正意义是 AR coaching**

paper 里轻描淡写说"可能用于 AR coaching"，但实际上这是 AR 眼镜的杀手级应用。想象你戴眼镜学修自行车，眼镜在你视角里叠加一个 expert 手部动作的 3D 渲染，告诉你"手应该放这"。要实现这个，需要：(a) 知道当前 keystep（keystep recognition）+ (b) 从 exo expert demonstration 生成 ego 视角的动作渲染（translation）+ (c) 评估你做得对不对（proficiency）+ (d) 决定下一步（procedure understanding）。**Ego-Exo4D 把这整个 pipeline 的所有组件都做成了 benchmark**。这才是它的真正价值——它不是单个任务的数据集，是一个完整 AR coaching 系统的 decomposed benchmark。

**8. Aria 的 SLAM + IMU 在 baseline 里被严重 underused**

Location-based 用了 head 3D 位置序列，EgoEgo 用了 DROID-SLAM head pose。但 Aria MPS 的 1 kHz 高频 IMU、eye gaze、点云都没在 baseline 里用。这些信号对 body pose 应该非常有用——尤其 IMU 能直接感知下肢加速度，gaze 能暗示 body orientation。我猜 CVPR 2025 的冠军一定会把这些信号塞进 transformer。

**9. Dataset 的 long-tail 问题**

paper 自己提了：cooking 9 倍于 soccer 时长。这反映现实——做顿饭 30 分钟，踢个球 drill 1 分钟。但训练时这会让 model 偏向 cooking。未来需要采样策略或 task-balanced loss。Ego4D 也有类似问题，Ego-Exo4D 继承了。

**10. Cross-domain skill transfer**

paper 里 8 个 domain 是独立的，但人类技能有 transfer——会弹钢琴的人学吉他比从零学快。能不能从 Ego-Exo4D 训练一个"skill embedding"，让钢琴 proficiency 预测吉他学习曲线？这需要跨 domain 的 skill representation，当前 dataset 没有"同一人学多种技能"的纵向数据，但可以做 cross-sectional 研究。

## 总结一句

Ego-Exo4D 把 ego-centric vision 从"生活记录"推进到"技能学习"，从单视角推进到 ego-exo 协同，从"是什么"推进到"怎么做+做得好不好"。它不只是数据集，是一套完整的 AR coaching 系统 decomposition。接下来的事就是社区把它推到 saturated——我估计 keystep recognition 会先饱和（像 ImageNet 一样），proficiency estimation 和 future prediction 最难也最有趣，ego body pose 还差一个数量级才能用。

arXiv 链接：https://arxiv.org/abs/2311.18297
数据申请：https://ego-exo4d-data.org/

---

# Ego-Exo4D: 一个 ego-exo 双视角视频理解的基础数据集

Andrej,这篇 paper 是 Meta FAIR 牵头、15 个机构合作两年的产物,核心 motivation 来自认知科学里 actor-observer translation 的能力 —— 婴儿在很早期就能从 exo 视角观察他人行为,映射到自己的 ego 视角,这是人类视觉学习的根基。Ego-Exo4D 把这种"观察专家然后内化为自己的身体动作"的过程数据化,这是 Ego4D(纯 ego daily life)和 HowTo100M(纯 exo instructional)都没有触及的关键 missing piece。

参考链接：
- 项目主页: https://ego-exo4d-data.org/
- Project Aria: https://projectaria.com/
- 前作 Ego4D: https://ego4d-data.org/
- 论文 arXiv: https://arxiv.org/abs/2311.18297

---

## 1. 为什么 ego-exo 双视角是关键

ego 视角捕获近场手-物交互和 camera wearer 的 attention(gaze),exo 视角捕获全身 pose 和环境 context。一个 how-to 视频往往在 exo(展示厨师整体动作)和 ego-like(切到手上特写)之间切换,这其实是人类 instructional video 的天然 grammar。

更深层的 intuition:要把"看到别人怎么做"转化为"我自己怎么做",需要在两个视角间做 semantic correspondence。这对 AR coaching(虚拟教练实时指导)、robot learning(机器人通过观察人学习 dexterous manipulation)都是核心能力。

---

## 2. 数据集核心数字

| 维度 | 数字 |
|---|---|
| 总视频时长 | 1,286 小时 |
| Takes(独立任务实例) | 5,035 |
| Camera wearers | 740 |
| Scenes | 123 |
| Cities | 13(横跨日本、哥伦比亚、加拿大、印度、新加坡、美国 7 州)|
| Activities | 43(分 8 个 domain)|
| Keysteps | 689 unique |
| Ego+exo segments | 143,442(454h)|
| 3D body pose frames | ~14M(含 auto GT)|
| Manual 3D body pose | 376K 3D / 2M 2D |
| Manual 3D hand pose | 68K 3D / 340K 2D |
| Object masks | 1.8M(ego 742K + exo 1.1M)|
| Expert commentary 片段 | 117,812(52 位专家,~6000 小时工作)|

8 个 domain 分两类：
- **Physical skills**(身体技能): soccer, basketball, dance, bouldering, music —— 强调 body pose 和与对象互动
- **Procedural skills**(程序性技能): cooking, bike repair, health care —— 强调 keystep 序列和 hand-object manipulation

这个分类很重要,因为 procedural 任务用 task graph 建模,physical 任务用 body pose + proficiency 建模,benchmark task 是按这个分界设计的。

---

## 3. Camera rig 和时间同步细节

每个采集点用统一 rig：1 个 Aria + 4 个 GoPro + 三脚架 + GoPro Remote,总成本 < $3000(不含 Aria)。

**Aria 传感器配置**:
- 1× rolling-shutter RGB camera: 30 fps, 1408×1408, FoV 110°
- 2× global-shutter monochrome SLAM cameras: 30 fps, 640×480, FoV 150° 每个
- 2× monochrome eye-tracking cameras: 10 fps, 320×240
- 7-microphone array(spatial audio)
- 2× IMU(800 Hz, 1000 Hz)
- barometer(50 Hz), magnetometer(10 Hz)
- 整机 75g(对比单 GoPro > 150g)—— 这点对 dance、bouldering 这种高动态活动至关重要,重了就影响自然动作

**时间同步**:用 smartphone 播放 29 fps 的 QR code 序列编码 wall-clock time,展示给所有相机。利用帧率差做亚帧精度同步:
- Aria(30 fps):2 帧同 QR → ±0.574 ms
- GoPro(59 fps):3 帧同 QR → ±0.558 ms

手动验证确保 GoPro 与 Aria RGB 在 ±1 帧(16.66ms)内。实践里 ~70% 的 take 自动 pipeline 完成,剩下 30% 用 lighter sparking、球鞋触球、手开始下滑吉他等可定位事件手动 sync。

这个精度对 ego pose 三角测量是必需的 —— 如果帧不同步,3D triangulation 会因为时间错位引入几厘米误差,直接污染 GT。

参考:
- Project Aria paper: https://arxiv.org/abs/2306.15518
- Aria MPS 文档: https://projectaria.com/docs/mps/

---

## 4. Aria MPS 的 3D 信号

Aria MPS(Machine Perception Services)输出几个关键信号：

**6DoF localization**:用 state-of-the-art VIO + SLAM,在统一的 metric、gravity-aligned 坐标系下,给每帧毫米级 6DoF pose,以及帧间 1 kHz 高频运动。783 个 Aria recordings 中 95.9% 全程 localization 成功,3.5% 部分失败(短 gap),0.6% 完全失败(主要是 glasses 摔落等物理冲击)。

**Eye gaze**:单一外向射线,锚定在两眼之间。可选 calibration 让 wearer 看手机屏幕图案做头部动作,生成个性化 gaze 方向。

**Point clouds**:用 photometric stereo 在连续帧或左右 SLAM 相机之间三角化静态场景元素。输出包含 3D 点云和每个点在相机图像中的 causal 2D observation。

**GoPro 6DoF localization**(Ego-Exo4D 额外开发):在 Aria SLAM 建的 map 上做 GoPro 帧 localization。用 P4P 算法(Kukelova et al., 2016)+ RANSAC 估计 6DoF 和焦距。3,724 个 GoPro recordings 中 91.4% 成功。失败主因是 GoPro 对准无纹理区域(白墙、白桌)。

这些 6DoF pose 是 ego-exo correspondence、translation、pose triangulation 的几何基础。

---

## 5. 三种语言标注 —— 这是数据集的灵魂

三种标注在 viewpoint、purpose、temporal density 上根本不同：

### 5.1 Expert commentary(专家评论)
52 位 domain experts(均 10+ 年经验,90% 有教练/教师经历),每人给每个视频看完整遍,再倒回去每隔大约 7 次/分钟暂停一次给口头评论,平均 4 句/段。117,812 段时间戳文本,~6000 小时工作量。

核心特征:聚焦 **how**(执行质量)而非 **what**(动作内容),会 surface 非专家看不出的小动作差异。比如:
- "The dancer's hand is rotated inwardly a bit. Her palm should be facing to the ground..."
- "Great footwork. He's using dribble to set up his footwork and his shot..."

还提供 1-10 的 proficiency 评分、空间 telestrator 手绘(红箭头/红圈标注身体部位)。每视频 2-5 位 expert 评论,提供 multi-perspective。

这是**第一次**在大规模视频数据集里有 expert critique,直接支持 proficiency estimation benchmark。Whisper 自动转录。

### 5.2 Narrate-and-act(边做边讲)
参与者自己用第一人称边做边讲,类似 how-to tutorial 但无后期制作。仅 ~10% 的 take 有,因为大部分 take 要让参与者无打扰自然执行。第一人称反思,语义上是 "what + why"。

### 5.3 Atomic action descriptions(原子动作描述)
第三方非专家 annotator 写,时间戳对齐每个原子动作,432K 句。规则:每句尽量只含 1 个 verb,时间戳误差 ±1 秒。Camera wearer 统一称 "C"(如 "C picks up a wrench"),其他人用其他字母。

**关键附加标签 —— visibility**:对每条 narration,annotator 标注(1)是否在 ego camera 可见,(2)哪个 exo camera 视角最好。这个标签后来被 correspondence、expert commentary、pose annotation 都用作 exo 视角选择依据。

### 5.4 三种标注的统计对比

| 指标 | Atomic | Narrate-and-act | Expert commentary |
|---|---|---|---|
| Vocabulary size | 最小 | 中 | 最大 |
| Captions/video 密度 | 最高(~31/min) | 中 | 低(~7/min) |
| 句长 | 短 | 中 | 长(平均 4 句) |
| Viewpoint | 第三方客观 | 第一人称主观 | 第三方专家 |

Cooking 和 soccer 的 atomic 密度特别高 —— cooking 是因为 procedural step 多,soccer 是因为 drill 反复执行。

---

## 6. 四大 Benchmark Task Family

### 6.1 Ego-exo Relation

#### 6.1.1 Correspondence

**任务**:给定同步 ego-exo 视频对 + 一个视角里的 object mask 序列(查询),在另一视角每同步帧输出同一 object 的 mask(如可见)。两个方向都做(Ego→Exo 和 Exo→Ego)。

**挑战**:极端视角差、重度遮挡、大量小 object(cooking utensils、bike repair tools,几个像素大小)、长视频平均 3 分钟。

**输入排除**:semantic labels、camera pose、IMU/深度 —— 鼓励 open-world correspondence 方法。

**Annotations**:1.8M masks @ 1fps,5,566 objects,1,335 takes,平均每个 take 5.5 个 object,每 object 平均跟踪 173 帧(扣除遮挡)。

**Metrics**:
- Location Error(LE):预测与 GT mask 质心的归一化距离
- IoU
- Contour Accuracy(CA):质心对齐后的 shape 相似度
- Visibility Accuracy(VA):balanced accuracy,衡量能否判断 object 在 target view 是否可见

**Baselines**:

1. **XSegTx**(spatial):基于 SegSwap(Shen et al., 2022)image co-segmentation 改造。输入 ego frame + exo frame + query mask(三通道),经 ResNet50 backbone → flatten → cross-image transformer(self-attention + cross-attention 交替)→ decoder 预测两视角 mask + visibility 分类头。损失:BCE + Dice,只对两视角都可见的帧算。

2. **XView-XMem**(spatio-temporal):基于 XMem(Cheng & Schwing, 2022)。训练时喂 interleaved ego-exo 帧序列(ego→exo→ego...),让模型学会跨视角跟踪。把 XSegTx 的 embedding 融合进 XMem working memory 以减轻 track drift。

**关键结果**（Table 4, Test set）:

| Query | Method | Type | VA | IoU | LE↓ | CA |
|---|---|---|---|---|---|---|
| Ego | XSegTx | S | 66.31 | 18.99 | 0.070 | 0.386 |
| Ego | XView-XMem(+XSegTx) | ST | **66.79** | **34.90** | **0.038** | **0.559** |
| Exo | XSegTx | S | 82.01 | 27.14 | 0.104 | 0.358 |
| Exo | XView-XMem(+XSegTx) | ST | 59.71 | 25.00 | 0.117 | 0.327 |

**直觉 takeaway**:
- 时序信息显著帮助 Ego→Exo(IoU 从 18.99 → 34.90,几乎翻倍)
- Ego→Exo 比 Exo→Ego 更难(因为 exo 里 object 太小,几个像素)
- 但即使最好的 ST baseline,IoU 也才 ~25-35% —— 任务远未解决

#### 6.1.2 Translation(exo→ego 生成)

**任务拆成两子任务**:
- **Ego Track Prediction**:给定 exo clip(5 帧,5 秒跨度)+ exo mask,预测 ego 帧中 object 的 segmentation mask
- **Ego Clip Generation**:给定 exo clip + exo mask + **GT ego mask**,生成 ego 视角下 object 区域的 RGB 像素

**输入**:5 个 exo frame + exo mask,可选 ego camera pose(作为 upper bound 变体)
**排除**:depth、3D point cloud、IMU、SLAM —— 这些在 wild video 中通常没有

**Track Prediction Baselines**:
1. **pix2pix-mask**:4 通道输入(exo frame + exo mask),4 通道输出(ego frame + ego mask)。BCE + Dice loss 监督 mask。
2. **GNT-mask**:Generalizable NeRF Transformer(Varma et al., 2023)改造。对每个点 x 和 viewing direction d,ray transformer 预测 RGB color c 和 object existence score e(e 是点 x 处有 object 的概率)。用 ego camera pose 作为额外输入。

**Clip Generation Baselines**:
6 个输入图:exo frame, exo mask, exo object crop, cropped exo mask, **ego mask**(GT), cropped ego mask。全部 resize 到 256×256。
1. **DiT-pix**(Peebles & Xie, 2022):Transformer-based diffusion。6 图沿 channel 拼接 + 与 noisy ego object crop 拼接 → DiT;另用 2 个 ResNet-50 编码 6 图特征,通过 AdaLN 注入 DiT 每层。
2. **pix2pix-pix**:6 图沿 channel 拼接输入 pix2pix。
3. Clip-to-clip 变体:用 3D-Conv 替 2D-Conv,space-time divided attention(TimeSformer 风格)。

**关键结果**:
- Track prediction:GNT-mask 略优于 pix2pix-mask,但用了 ego pose 作为 cheating 输入
- Clip generation(Table 5b):

| Method | SSIM↑ | PSNR↑ | DISTS↓ | LPIPS↓ | CLIP↑ |
|---|---|---|---|---|---|
| pix2pix-pix | 0.42 | 16.4 | 0.36 | 0.50 | 79.8 |
| **DiT-pix** | **0.59** | 16.1 | **0.31** | **0.46** | **81.9** |

**直觉 takeaway**:DiT 的 diffusion 显著优于 GAN,在 photorealism 和 CLIP alignment 上都好。Ablation 表明:
- 缺 exo object crop → 无法推断颜色/纹理(因为 object 在整张 exo frame 中太小)
- 缺 ego crop mask → object 朝向预测错误
- Clip-to-clip 不一定提升定量指标,但**视觉一致性**更好,尤其在 exo 严重遮挡帧

参考:
- SegSwap: https://arxiv.org/abs/2203.06612
- XMem: https://arxiv.org/abs/2207.03080
- DiT: https://arxiv.org/abs/2212.09748
- pix2pix: https://arxiv.org/abs/1611.07004
- GNT: https://openreview.net/forum?id=xE-LtsE-xx

---

### 6.2 Ego(-exo) Keystep Recognition

#### 6.2.1 Fine-grained keystep recognition

**任务**:trimmed clip 分类。训练时有 ego+exo 多视角,测试时**只有 ego RGB**(排除 exo、narration、audio、gaze、3D、IMU)。

**特点 vs 普通 action recognition**:
1. Fine-grained:同 activity 的不同 keystep 可能涉及同样 object(如 "fold the bedsheet" vs "smooth out the bedsheet")
2. 时间跨度差异巨大:"kneading dough" 平均 87.3 秒,"get salt" 平均 3.6 秒 —— 需要多时间尺度
3. 训练时可借 exo 的 contextual cue 学习 view-invariant 表征,蒸馏到 ego-only 测试模型

**Annotations**:17 个 procedural activities(cooking 11、bike repair 4、health 2),664 个 keystep,143,442 segments,平均 11.34 秒/segment。cutoff ≥20 samples/keystep → 278 个 keystep。

**Baselines**:
1. **TimeSformer**(K600 预训练):经典 action classification
2. **EgoVLPv2**(Ego4D 或 EgoExo4D 预训练):video-language pretraining
3. **VI Encoder**:两阶段。Stage 1 在所有 ego-exo 对上做 clip-level contrastive loss(InfoNCE 风格),正样本是同步 ego-exo 对,负样本是非同步对。Stage 2 加 classification loss 微调
4. **Viewpoint Distillation**:Stage 1 训 multi-view teacher(ego+exo 都输入);Stage 2 训 ego-only student,从 teacher 蒸馏(Hinton et al., 2015)
5. **Ego-Exo Transfer**(Li et al., 2021):MAE backbone + Ego-Exo 伪标签预训练 + Object-Score/Interaction-Map 辅助头 + 分类微调

**关键结果**（Table 6, Test Top-1 Acc）:

| Method | Train data | Test Acc |
|---|---|---|
| TimeSformer (K600) | ego | 35.93 |
| TimeSformer (K600) | ego,exo | 31.04(下降!) |
| EgoVLPv2 (EgoExo4D) | ego | 37.72 |
| EgoVLPv2 (EgoExo4D) | ego,exo | 38.76 |
| **VI Encoder** | ego,exo | **41.53** |
| Viewpoint Distillation | ego,exo | 39.49 |
| Ego-Exo Transfer MAE | ego,exo | 36.58 |

**直觉 takeaway**:
- TimeSformer 加 exo 分类 loss 反而掉点(35.93→31.04) —— 因为分类 loss 在 exo 视角下太容易,反而把表征拉离了 ego 任务所需
- EgoVLPv2 加 exo 提升了(37.72→38.76) —— 因为它的对比学习目标能从 exo 抽取互补语义
- **VI Encoder 最好**(41.53) —— 显式 view-invariant 约束最有效
- Per-keystep 分析:exo 在 "have a conversation asking different questions" 这种需要环境 context 的步骤上占优;ego 在 "cut carrots"、"unpack the new tube" 这种小 object manipulation 上占优

#### 6.2.2 Energy-efficient multimodal keystep recognition

**任务**:online 流式 keystep 预测,在 energy budget B 内。模型 F = (sensor policy F^P, keystep predictor F^K)。每步 t,policy 决定激活哪些 sensor → observation O^t ⊆ {S_1^t, ..., S_K^t}。

**Energy 公式**(Eqn. 1, 这是 paper 里最值得讲的公式):

$$
E(O^t) = \alpha \cdot C(O^t) + \beta \cdot M(O^t) + \sum_{j=1}^{K} \gamma_j \cdot \mathbb{1}(S_j \in O^t)
$$

变量含义:
- $O^t$:时间步 t 的观测(被激活的 sensor 子集)
- $C(O^t)$:前向 pass 的总 MAC 操作数(MAC/s)
- $M(O^t)$:总 DRAM 读写量(MB/s)
- $S_j \in O^t$:第 j 个 sensor 是否激活,indicator function
- $\alpha$:compute energy 系数 = **4.6 pJ/MAC**(来自 Sze et al., 2020 的 ASIC 估算)
- $\beta$:memory transfer 系数 = **80 pJ/byte**(Horowitz 2014 ISSCC 经典数字)
- $\gamma_j$:sensor j 持续运行的功率。$\gamma_{rgb} = 15$ mW,$\gamma_{audio} = 0.5$ mW(Liu et al., 2020 IEDM)

**为什么这个公式重要**:它把"模型 efficiency"从 FLOPs/MACs 单一指标扩展到真实 AR/VR 硬件上的 total power。memory transfer 在很多模型里占 >50% 总能耗(Yang et al., 2022),忽略它就低估了真实成本。sensor 本身的功耗(camera 15mW 持续开启)在 always-on 场景里更是主导。

**Tiers**:
- High-efficiency:20 mW budget
- High-performance:2.8W budget

**Baselines**(每个有 backbone + sampling policy):
- X3D-XS:vision-only,最轻
- LaViLa:vision-only,TimeSformer-Base + CLIP-style video-language pretraining
- Light-ASDNet:audio-only,1D 卷积沿 spectrogram 时间维
- AV-LF:audio-visual late fusion
- Sampling policy:fixed stride s、greedy、random、AV-cascade(audio 不置信才切 vision)

**关键结果**:
- High-efficiency(Table 7a):

| Method | Modality | mcAP | Power |
|---|---|---|---|
| Light-ASDNet + s=5 | A | 65.18 | 19.67 mW |
| X3D-XS + s=10 | V | 76.85 | 19.14 mW |
| **AV-LF w/ X3D-XS + s=15** | AV | **77.89** | 19.70 mW |

- High-performance(Table 7b):

| Method | Modality | mcAP | Power |
|---|---|---|---|
| LaViLa + s=5 | V | 93.24 | 2245.66 mW |
| AV-LF w/ LaViLa + s=5 | AV | 92.18 | 2274.40 mW |

**直觉 takeaway**:
- High-efficiency 下 AV 优于单模态 → 有互补信息
- High-performance 下 AV 反而比纯 vision 略差 → LaViLa 视觉特征已经足够强,简单线性 late fusion 反而损失表达力
- 音频特别擅长 "stir fry egg mixture"、"cut butter" 等声音 distinctive 的 keystep;视觉擅长 "add green chillies"、"get celeries" 等无声动作
- Cascade 策略差,因为 audio backbone 常给出错误但 over-confident 预测,阻止切到 vision

参考:
- TimeSformer: https://arxiv.org/abs/2102.05095
- EgoVLPv2: https://arxiv.org/abs/2307.05421
- LaViLa: https://arxiv.org/abs/2212.04501
- X3D: https://arxiv.org/abs/2004.04730
- Sze et al. energy: https://ieeexplore.ieee.org/document/9000795

#### 6.2.3 Procedure understanding

**任务**:给定 keystep segment $s_i$ 和 history $S_{:i-1}$,做 5 件事:
1. 确定 previous keysteps($s_i$ 之前应做的)
2. 判断 $s_i$ 是否 optional
3. 判断 $s_i$ 是否 procedural mistake(前置条件不满足)
4. 预测 missing keysteps(应做但未做的)
5. 预测 next keysteps(依赖满足,可执行的)

两种 supervision:
- Instance-level:训练/测试都有 keystep 标签
- Procedure-level:训练/测试都无标签,只有 keystep 名 taxonomy(弱监督)

**Task graph**：有向图,node=keystep,edge=dependency。含 OR/XOR 结构、optional、repeatable 属性。例如 "Mix Eggs" 可在 "Add Water" 或 "Add Milk"(任一)后执行;"Pour Mixture" 要求 "Melt Butter" XOR "Heat Oil"。

**Transition probability**:

$$
P(B|A) = \frac{\#\text{keystep B follows keystep A}}{\#\text{occurrences of keystep A}}
$$

$P(B|A)$ 即给定 A 出现后 B 紧跟的条件概率,通过训练集统计得到。这个 graph 在训练集上从 keystep 共现统计自动初始化,再人工修正。

**Metrics**:calibrated Average Precision(cAP,De Geest et al., 2016)。random baseline ~50%。

**Baselines**:
- Graph-based:Keystep Assignment(EgoVLPv2 cosine 匹配 keystep 名)+ Procedural Reasoning(transition graph)
- End-to-end:3 个 MLP 直接从 EgoVLPv2 视频特征预测 prev/optional/next

**关键结果**（Table 8, Test cAP %）:

| Supervision | Baseline | Prev. | Opt. | Mistake | Miss. | Fut. |
|---|---|---|---|---|---|---|
| - | Uniform | 59.13 | 56.73 | 60.66 | 65.64 | 65.65 |
| Instance | Graph-Based(GT) | **82.32** | 62.10 | **73.06** | **82.63** | 62.82 |
| Instance | End-to-End | 62.05 | 61.39 | 52.07 | 61.77 | 59.25 |
| Procedure | Graph-Based(assignment) | 53.43 | 52.36 | 57.81 | 53.92 | 53.54 |
| Procedure | Graph-Based(prediction) | 66.22 | 49.00 | 58.59 | 64.18 | 58.34 |

**直觉 takeaway**:
- 用 GT 标签的 graph-based 在 prev/miss/mistake 上远超 uniform(82/82/73 vs 59/65/60),说明简单共现统计已能捕捉 procedure 结构
- **future keystep 预测最难** —— 因为有多个合法 next,分布发散
- End-to-end 低于 graph-based,因为没显式 graph 先验
- Procedure-level 整体低,因为弱监督下连 keystep 识别都不准 —— 这是真正的开放问题

参考:
- cAP (De Geest et al.): https://link.springer.com/chapter/10.1007/978-3-319-46478-7_17
- ActionFormer: https://arxiv.org/abs/2207.07925

---

### 6.3 Proficiency Estimation

#### 6.3.1 Demonstrator proficiency(整个 demo 的技能水平)

**任务**:4 类分类(novice / early expert / intermediate expert / late expert),一个 participant 一个 label。

**Label 来源**:结合 participant 问卷(years experience、是否教过别人、是否有专业 training)和 expert commentary 的 1-10 评分,通过 scenario-specific 转换函数映射到 4 类(见 Table E7)。例如:
- Basketball/Soccer:year of experience X ∈ [0,1) → novice;[1,3) → early;[3,10) → intermediate;≥10 → late
- Bouldering:最高完成难度 H,V3 以下 novice,V4 early,V5 intermediate,V6+ late
- Cooking:expert 平均评分 P,<3.5 novice,[3.5,5) early,[5,8) intermediate,≥8 late

**Baselines**:TimeSformer 在 ego、在 4 个 exo、ego+exo late fusion(softmax 平均)。

**关键结果**（Table 10, Test Acc）:

| Pretraining | Ego | Exos | Ego+Exos |
|---|---|---|---|
| Random init | 33.9 | 47.5 | 45.7 |
| K400 | 44.1 | 47.0 | 46.1 |
| HowTo100M | 36.7 | **46.6** | **47.0** |
| EgoVLP | 43.8 | 44.5 | 43.5 |
| **EgoVLPv2** | **50.4** | 47.0 | 48.7 |

**直觉 takeaway**:
- Ego 在 cooking 上更好(需要手-物近场);Exo 在 bouldering 上更好(需要 body pose) —— 互补
- Late fusion 不提升 —— 简单平均 softmax 太弱
- 多个 demonstration 推断同一人(48.3 → 51.0 in val)有帮助

#### 6.3.2 Demonstration proficiency(视频内时间点定位)

**任务**:在 video 中 localize "good execution" 和 "needs improvement" 的时间戳集合。

**形式化**:$\hat{G}, \hat{I} = h(\mathcal{V})$,$\hat{G} = \{t_1^g, t_2^g, ...\}$ 是 good execution 时间戳,$\hat{I} = \{t_1^i, t_2^i, ...\}$ 是 improvement tip 时间戳。

**Annotation 来源**:把 expert commentary 的时间戳分类 —— 每段 comment 标记是否描述 good execution 和/或 提建议(同一段可两者都是,见 Table E8)。

**Metric**:基于 $L_1$-distance 的 mAP(不是 temporal IoU,因为是单点而非 segment),threshold k ∈ {0.25, 0.5, 1.0} 秒。

**Baseline**:ActionFormer 改造,做 timestamp regression 而非 segment localization。NMS 用 $L_1$-distance。Omnivore features。

**关键结果**（Table 11, Test Avg mAP % across k ∈ {0.25, 0.5, 1.0}）:

| Method | Ego | Exos | Ego+Exos |
|---|---|---|---|
| Random | 2.20 | 2.20 | 2.20 |
| Uniform tips | 2.39 | 2.39 | 2.39 |
| **ActionFormer** | **3.87** | **3.87** | **4.04** |

mAP ~3-4% —— **极其困难**。任务还很初级,有很大的改进空间。

参考:
- TimeSformer: https://arxiv.org/abs/2102.05095
- ActionFormer: https://arxiv.org/abs/2207.07925
- Omnivore: https://arxiv.org/abs/2201.08310

---

### 6.4 Ego Pose

#### 6.4.1 Body pose

**任务**:从 ego RGB 视频、IMU 序列,或两者,预测 3D body pose 序列 $\mathcal{P} = \{\mathcal{P}_1, ..., \mathcal{P}_T\}$,其中 $\mathcal{P}_t \in \mathbb{R}^{17 \times 3}$ 是 MS COCO 17 joints 的 3D 坐标。

**Annotation pipeline**(auto + manual):
1. **Auto GT**:用 MMPose 在 4-5 个 exo 视角检测 2D bbox(用 Aria MPS 的 3D headset 位置投影选 camerawearer 的 bbox)→ 2D keypoint → 多视角 RANSAC triangulation 得 3D keypoint。~9.2M 帧。
2. **Manual**:多视角 annotation UI,人工修正 2D 投影后再 triangulate。~376K 3D / 2M 2D。

Auto GT vs Manual GT 的 MPJPE 仅 3.33 cm —— 远小于 baseline 方法误差,验证 auto GT 可用作训练扩充。

**Baselines**:
1. **Static pose**:固定为训练集平均 pose,根据 IMU 平移。地板 baseline
2. **Kinpoly**(Luo et al., 2021):物理 simulator 里用 humanoid 跟踪 head pose + action type,输出 joint torque 而非 kinematic 位置
3. **EgoEgo**(Li et al., 2023):两步。Step 1:DROID-SLAM 给初始 head pose 轨迹 + GravityNet 修旋转 + HeadNet 用光流修缩放。Step 2:DDPM 扩散模型条件在 head pose 序列,生成 full body motion,训练在 AMASS
4. **Location-based**:transformer 直接从 3D 头部位置序列预测 3D joint 位置(不用 parametric body model)。40k 迭代,Adam,lr 1e-4,window 40 帧,MSE loss

**关键结果**（Table 12, Test）:

| Method | MPJPE(cm)↓ | MPJVE(m/s)↓ |
|---|---|---|
| Static pose | 215.87 | - |
| EgoEgo | 26.38 | 0.66 |
| Kinpoly | 24.36 | 0.65 |
| **Location-based** | **18.51** | **0.64** |

**Per-scenario 分析**(Table 13):Location-based 在 cooking(12.65)、health(11.63)、music(15.00)上最好;在 dance(21.15)和 basketball(19.89)上误差大 —— 高动态、肢体大幅度的活动最难。

**CVPR 2024 challenge**(Table E10):
- Multi-Scale Model Fusion(ensemble 不同 transformer 层数):MPJPE 15.32 cm
- UCB Ego(conditional diffusion on SLAM pose + AdapterNet):17.19 cm
- Levelwise attention ViT:18.09 cm

#### 6.4.2 Hand pose

**任务**:从 ego RGB 帧,预测至少部分可见手的 3D joints,每手 21 个,MS COCO 约定。

**Annotation**:类似 body,但额外检测 ego 视角的 hand keypoint,用全身 pose 推断多人场景下的手位置。

**Baselines**(都改造成只监督 2D/3D joint,不监督 mesh):
1. **THOR-net**(Aboukhadra et al., 2023):Keypoint-RCNN 提 2D → GraFormer lift 到 3D
2. **HandOccNet**(Park et al., 2022):ResNet50-FPN 提 2D 特征 + Feature Injecting Transformer(注入 hand 信息到遮挡区) + Self-Enhancing Transformer
3. **POTTER**(Zheng et al., 2023):Pooling Attention Transformer(降内存)+ HybrIK mesh regression head
4. **METRO**(Lin et al., 2021):CNN global feature + transformer encoder 联合建模 vertex-vertex 和 vertex-joint,只在 FreiHand 上预训练,直接 inference

**关键结果**（Table 14, mm）:

| Method | Manual MPJPE | Manual PA-MPJPE | Manual+Auto MPJPE | Manual+Auto PA-MPJPE |
|---|---|---|---|---|
| METRO* | - | 20.61 | - | 20.61 |
| THOR-net | 51.24 | 17.99 | 47.64 | 17.61 |
| HandOccNet | - | 17.22 | - | **13.56** |
| POTTER | 30.57 | 11.14 | 28.94 | 11.07 |

**直觉 takeaway**:
- Manual+Auto 训练显著好于 Manual only —— 数据规模很关键
- POTTER 效率最高(5.2 GMACs vs THOR-net 123.6 GMACs),精度也最好
- **可见视角数 vs PA-MPJPE**(Table 15b):3 views → 14.01mm,4 → 12.15,5 → 11.03,6 → 10.02 —— 视角越多 GT 越准,这也是为什么所有实验只在至少 3 个视角可见的 joint 上做

**Per-joint 误差**(Fig. 26):拇指和指尖误差最大 —— 因为最常被 object 或自身遮挡。

**CVPR 2024 challenge**(Table E11):
- PCIE EgoHandPose(ViT-H + RLE regression loss + ensemble):MPJPE 25.51, PA-MPJPE 8.49
- Hand3D(直接 HaMeR 出 box,ViT-H backbone):PA-MPJPE 9.30
- POTTER-ensemble:28.68 / 10.24

参考:
- EgoEgo: https://arxiv.org/abs/2312.04205
- Kinpoly: https://arxiv.org/abs/2104.07119
- DROID-SLAM: https://arxiv.org/abs/2106.02020
- AMASS: https://amass.is.tue.mpg.de/
- THOR-net: https://arxiv.org/abs/2303.12131
- HandOccNet: https://arxiv.org/abs/2203.08253
- POTTER: https://arxiv.org/abs/2303.16525
- METRO: https://arxiv.org/abs/2012.09760
- HaMeR: https://hamer.is.tue.mpg.de/
- GraFormer: https://arxiv.org/abs/2204.02567

---

## 7. Dataset split

统一 split(stratified by activity + proficiency,participant 独占一个 split 防止 leakage):
- Train: 3,082 takes
- Val: 842 takes
- Test: 1,121 takes

所有 benchmark 共用,方便 cross-benchmark 研究。

---

## 8. 与现有数据集对比(Table 1)

Ego-Exo4D vs 主要相关数据集:
- vs **Ego4D**(Grauman et al., 2022):931 subjects,74 scenes,110 actions,纯 ego;Ego-Exo4D 740 subjects,123 scenes,689 keysteps,ego+exo,更多模态,新增 expert commentary
- vs **Assembly101**(Sener et al., 2022):53 subjects,1 scene,513h,tabletop 玩具车组装,有 ego+exo;Ego-Exo4D 规模和多样性大得多
- vs **H2O**(Kwon et al., 2021):4 subjects,3 scenes,5h;小规模实验室
- vs **EgoExoLearn**(Huang et al., 2024):136 subjects,7 scenes,120h,ego 模仿 exo demonstration;Ego-Exo4D 同步捕获真实 ego-exo

---

## 9. 关键直觉总结

1. **Ego-exo 是同一物理过程的两个 semantic 侧面,不是简单两个视角**。ego 给 attention 和 close-up manipulation,exo 给 body 和 context。任何只看一个视角的方法都丢了一半信息。

2. **Skill 在视频里是非平凡的**。它是 "how" 而非 "what",expert commentary 第一次把这个维度数据化。Proficiency estimation mAP 只有 3-4% 说明问题远未解决。

3. **Procedure structure 是可挖掘的,但 future 预测极难**。Graph-based 在 prev/missing/mistake 上能达 80+ cAP,但 future 只有 62,因为 future 是发散的多模态分布。

4. **Energy efficiency 必须建模 sensor 功耗**。单看 FLOPs/MACs 严重低估 always-on 场景的实际能耗,Eqn. 1 的 $\alpha C + \beta M + \sum \gamma_j$ 形式应该成为 AR/VR 视觉模型的标配。

5. **Ego pose 仍有大量改进空间**。最好的 body 方法 MPJPE 15-18 cm,hand PA-MPJPE 8-11 mm。dance、basketball 等高动态场景最差。Auto GT 与 manual 仅差 3.33 cm,可信,且把训练数据从 376K 扩到 9.2M,效果显著。

6. **Multi-view 训练,单视角测试** 是 AR 应用的现实约束,但也是 representation learning 的好工具。VI Encoder(41.53% keystep acc)说明显式 view-invariant 对比学习是当前最佳策略。

7. **Time sync 是 hidden infrastructure**。QR + frame rate 差的 ±0.574 ms 精度是所有多视角 triangulation 的基础。30% take 需要手动 sync,说明自动化仍有提升空间。

8. **Cross-view translation 不只是 geometry,需要 semantic priors**。DiT 显著优于 pix2pix,因为 object 在 exo 中太小,模型必须从语义先验"幻觉"出 ego 视角下的细节(如手指接触点)。缺 exo object crop 就完全失败,说明语义信号是核心。

---

## 10. 未来方向（个人联想）

- **Ego-exo joint foundation model**:把 Aria 的 IMU+audio+eye gaze+point cloud+6DoF 全部模态和三种语言一起预训练一个多视角多模态 backbone,可能产生类似 CLIP 在图像上的 zero-shot 能力,但 for skilled activity。

- **Skill-conditioned generation**:用 expert commentary 的 proficiency score 作 condition,训练 diffusion 生成"同一个 step 的 novice vs expert 版本",这能用于 AR coaching 实时给反馈。

- **Cross-embodiment translation**:把 exo 的 expert demonstration 转成 ego 视角还不够,真正有价值的是转成"我自己的手"的视角 —— 这需要 hand shape 个性化 + 视角合成。

- **Procedure graph 自动发现**:从 1286 小时未标签视频里用 contrastive learning 挖 keystep dependency,可能发现人类没意识到的 sub-step 结构。

- **Audio-first keystep detection**:从 Table 7a 看,Light-ASDNet 在 19.67 mW 下能达 65.18 mcAP,远超 X3D-XS 的能耗效率 —— audio 在 always-on 场景可能比 vision 更优先。可以设计 audio-trigger-vision 的 cascade policy。

- **Ego-pose 中的 SLAM signal is underexploited**:EgoEgo 用了 DROID-SLAM 的 head pose,但 Aria MPS 的 1 kHz 高频 6DoF + eye gaze 没充分利用。用 gaze direction 作为 body orientation prior,应该能大幅降 MPJVE。

- **Hand-object interaction 的 3D**:Ego-Exo4D 提供了 hand pose + object mask + camera 6DoF,理论上可以 reconstruct 出 hand-manipulating-object 的 4D sequence。这对 robot imitation learning 是金矿。

- **Skill 的细粒度 taxonomy**:当前 4 级 proficiency 粗,可以用 expert commentary 的 1-10 分训练回归模型,产出 continuous skill embedding,可能揭示不同 domain 的 skill 有共享结构(如 musician 的"音色控制"和 chef 的"火候控制"在某种抽象层面同构)。

- **Self-supervised ego-exo**:用 1286h 未标注的同步 ego-exo 对做 contrastive learning,可能学到 view-invariant 的 activity primitive 表示,无需 keystep 标注。这是 EgoVLPv2 在 Ego-Exo4D 上预训练的扩展。

- **AR Coach 端到端 demo**:把 proficiency estimation + procedure understanding + ego pose 三个 benchmark 串起来,可以构建一个 demo:AR 眼镜实时识别你在做哪步、做得如何、下一步该做什么,并用 ego-exo translation 把 expert 的手部动作叠加到你的 ego 视角。这是 Ego-Exo4D 的终极应用愿景。

Ego-Exo4D 是继 Ego4D 之后 ego-centric vision 的下一个里程碑,把视角从单一 ego 扩展到 ego-exo 协同,把目标从 daily activity 扩展到 skilled activity,把标注从 narration 扩展到 expert critique。它给社区提供了 1286h 的同步多视角多模态数据 + 4 个 benchmark 家族 + 14M 帧 pose 标注 + 117K 段 expert commentary,基本上是接下来 3-5 年 ego-centric 研究的主战场。
