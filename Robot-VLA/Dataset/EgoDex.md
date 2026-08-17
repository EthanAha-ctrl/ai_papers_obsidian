---
source_pdf: EgoDex.pdf
paper_sha256: eff045367034cfb7837e1b9f9f9ee1ebeb9f14c04ad90e8883d9190bdd04cb9f
processed_at: '2026-08-04T02:30:00-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# EgoDex 用人话讲

## 一句话概括

Apple 团队用 Apple Vision Pro 采集了 829 小时的第一人称视频，配同步的 3D 手部关节追踪，规模甩开同类 dataset 一个数量级，目的就一个:把 Sutton 的 "bitter lesson" 硬塞进 robot manipulation 这个一直吃不到大数据红利的领域。论文链接:[EgoDex GitHub](https://github.com/apple/ml-egodex)。

## 为什么这件事值得说

robot manipulation 这十年一直有个尴尬:LLM 有 internet text，vision 有 ImageNet ([Russakovsky et al. 2015](https://arxiv.org/abs/1409.0575))，唯独 robot 拿不到 internet-scale 数据。主流的 teleoperation 路线 ([DROID](https://droid-dataset.github.io/)、[Open X-Embodiment](https://robotics-transformer-x.github.io/)) 听起来美好，实际操作时一个 human 站在 robot 旁边手动摇 76k 次，这玩意儿再怎么 community-wide 也扩不动。

EgoDex 的核心 insight 很简单:戴上 Vision Pro 做日常动作就完事了，没有 robot、没有 teleop、没有 reset 成本。视频是 1080p 30FPS，同时 hand tracking 是 production-grade，25 joints/hand 的 3D pose 在采集时就自动生成。这跟 [DexCap](https://arxiv.org/abs/2403.07788) 戴 mocap 手套、[UMI](https://arxiv.org/abs/2402.10329) 拿手持 gripper 的"主动采集"路线完全不在一个 paradigm 上 — 后者再便宜也激励不了人做 829 小时日常操作。

这跟当年 ImageNet 之前 AlexNet 诞生的逻辑一模一样:先得有大规模、annotation 干净的数据，scaling 才能跑起来。

## 数据规模到底多大

扔几个数字直观感受:

| 指标 | EgoDex | 第二名 |
|---|---|---|
| Trajectory 数 | 338k | 162k (RoboNet) |
| Task 数 | 194 | 125 (EPIC-KITCHENS) |
| Frame 数 | 90M | 21M (Ego4D HOI subset) |
| 总存储 | 2.0 TB | — |

跟最接近的同类工作 [EgoMimic](https://arxiv.org/abs/2410.24221) 比:EgoMimic 只有 4 小时、3 个 task、只跟踪 wrist。EgoDex 是 829 小时、194 个 task、双手 25 joints/hand + upper body。差了 200× scale。

数据压缩比 250×(raw 500TB → 2TB)，靠现代 video codec 实现。训练时用 [torchcodec](https://github.com/pytorch/torchcodec) 做 lazy decoding，只解 batch 里的 frame，否则 I/O 会爆。

## Action representation 的选择:48 维怎么来的

这是 build intuition 的关键。EgoDex 的 "action" 不是 robot joint torque，是 **human hand 的 3D keypoint trajectory**。具体每只手:

- wrist 的 3D position:3 维
- wrist 的 6D orientation:6 维 (用 [Zhou et al. 2018](https://arxiv.org/abs/1812.07035) 的连续 rotation representation，避免 quaternion discontinuity)
- 5 个 fingertip 的 3D position:5 × 3 = 15 维

一只手 24 维，双手 48 维。公式:

$$
\mathbf{a}_t = \left[ \mathbf{p}_L^{\text{w}}, \mathbf{r}_L^{\text{w}}, \mathbf{p}_L^{\text{t}_1}, \dots, \mathbf{p}_L^{\text{t}_5}, \mathbf{p}_R^{\text{w}}, \mathbf{r}_R^{\text{w}}, \mathbf{p}_R^{\text{t}_1}, \dots, \mathbf{p}_R^{\text{t}_5} \right] \in \mathbb{R}^{48}
$$

变量说明:
- $\mathbf{p}_L^{\text{w}}, \mathbf{p}_R^{\text{w}} \in \mathbb{R}^3$:左右 wrist 的 3D position
- $\mathbf{r}_L^{\text{w}}, \mathbf{r}_R^{\text{w}} \in \mathbb{R}^6$:wrist 的 6D rotation representation (用 6D 是因为 4D quaternion 在神经网络里有拓扑不连续问题)
- $\mathbf{p}_L^{\text{t}_i}, \mathbf{p}_R^{\text{t}_i} \in \mathbb{R}^3$:第 $i$ 个 fingertip 的 3D position

pose 都在当前 camera frame 里表达，action chunk 是 relative trajectory (类似 UMI 的设计，避免绝对坐标漂移)。

这个 representation 看起来简陋(没 contact force、没 finger 之间的相对 pose)，但是有几个好处:humanoid-friendly(可以直接 transfer 给未来的人形 robot)、与 camera frame 解耦、维度适中(48D 在 diffusion/flow matching 里训练很顺手)。

## 两个 Benchmark:Forward vs Inverse

EgoDex 的两个 benchmark 对应两种不同的 policy 形式，这里 build intuition 很重要。

### Benchmark 1:Trajectory Prediction(forward)

$$
f_\theta\left( \mathbf{o}_{0..t}, \mathbf{s}_{0..t}, l \right) = \hat{\mathbf{a}}_{t:t+H}
$$

- $\mathbf{o}_{0..t}$:从时刻 0 到 $t$ 的 egocentric image 序列
- $\mathbf{s}_{0..t}$:从时刻 0 到 $t$ 的 skeletal pose 序列(就是 hand joint 3D 位置)
- $l$:自然语言描述(GPT-4 整合过的)
- $H$:prediction horizon，$H=30$ 是 1 秒，$H=60$ 是 2 秒
- $\hat{\mathbf{a}}_{t:t+H}$:预测的 48 维 action chunk

这个就是经典的 imitation learning setup:看过去，预测未来。

### Benchmark 2:Inverse Dynamics(goal-conditioned)

$$
f_\theta\left( \mathbf{o}_{0..t}, \mathbf{s}_{0..t}, \mathbf{o}_{t+H}, l \right) = \hat{\mathbf{a}}_{t:t+H}
$$

多了一个 $\mathbf{o}_{t+H}$:horizon 结束时刻的 goal image。这个改动在 Table 4 里效果炸裂:final distance 从 0.062m 砍到 0.029m，降 53%。

直觉很清楚:人手 motion 高度 multimodal。你让模型"放杯子到桌上"，杯子可以放桌上任意位置、任意速度、任意路径。forward prediction 模型只能学 conditional distribution 的 mean，或者 sample 一个 mode，但都可能跟 ground truth 对不上。但是给一个 goal image "这就是终点长啥样"，model 知道终点确定，只需要 prediction 怎么从起点走到终点，multimodality 大幅收缩。

这个 insight 跟 [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)、[SuSIE](https://arxiv.org/abs/2304.14156)、[HierDiff](https://arxiv.org/abs/2305.03136) 这类工作的逻辑一脉相承:goal image 是廉价的 multimodality solver。

## Best-of-K Metric:为什么要这么评

human motion 的 multimodality 让传统 L2 metric 失效。EgoDex 的做法:每个 test sample 让 model sample $K$ 次，取离 ground truth 最近的那次算分。

公式:

$$
\text{Err} = \frac{1}{N} \sum_{n=1}^{N} \min_{k \in \{1,\dots,K\}} \left[ \frac{1}{H} \sum_{t=1}^{H} \frac{1}{12} \sum_{j=1}^{12} \left\| \mathbf{a}_{t,j}^{(k)} - \mathbf{a}_{t,j}^{\text{GT}} \right\|_2 \right]
$$

- $N$:test set 样本数
- $K$:每个样本采样次数(1, 5, 10)
- $H$:prediction horizon
- 12 个 keypoint:2 个 wrist + 10 个 fingertip(每手 5 finger)
- $\mathbf{a}_{t,j}^{(k)}$:第 $k$ 次采样在时刻 $t$ 第 $j$ 个 keypoint 的 3D position
- $\mathbf{a}_{t,j}^{\text{GT}}$:ground truth

单位是 meters。0.04m 大约就是 4cm，跟手指精度差不多。

deterministic model(BC)的 Err 与 $K$ 无关，因为每次 sample 都一样。stochastic model(DDPM、FM)的 Err 随 $K$ 增大而改善，因为有更多机会命中 ground truth 的 mode。

这个 metric 在 deploy 时其实有个微妙问题:真实 robot 只能执行一次 action，相当于 $K=1$。所以严格说 deploy 性能看 $K=1$ 列，但 paper 把 K=1/5/10 都列出来，方便看 distribution learning 的质量。

## 实验数据:几个关键 take-away

### Take-away 1:EncDec 一致性优于 Decoder-only，但差距很小

Table 2 里 EncDec+FM 在 K=10 下 0.038m，Dec+FM 0.040m，差 2mm。Encoder-decoder 把 image 编码进 latent 再 decoder 出 action，比 decoder-only 直接 token 化多了一层 inductive bias，但数据规模上来后这个优势缩小。

### Take-away 2:BC 学 mean，FM/DDPM 学 distribution

Table 2 三组对比:

| K | EncDec+BC | EncDec+DDPM | EncDec+FM |
|---|---|---|---|
| 1 | **0.044** | 0.052 | 0.051 |
| 5 | 0.044 | 0.042 | 0.041 |
| 10 | 0.044 | 0.039 | **0.038** |

K=1 时 BC 最好，因为 BC 的 L2 loss 学的是 conditional expectation，average 出来的轨迹在 multimodal 下虽然 mode-averaged，但平均距离 ground truth 近。DDPM/FM 单次 sample 命中某个 mode，可能跟 ground truth mode 对不上，距离反而大。

K=10 时 FM 最好，因为多次 sample 把 conditional distribution 覆盖了，总能 sample 到 ground truth mode 附近。0.038m vs 0.044m，差 14%。

直觉:BC 是"保守的平均先生"，FM 是"有创意的多次尝试者"。前者稳妥，后者有上限。

### Take-away 3:Prediction Horizon 越长越难

Table 3 用 Dec+BC 测:

| Horizon | Avg (m) | Final (m) |
|---|---|---|
| 1s (H=30) | 0.031 | 0.049 |
| 2s (H=60) | 0.045 | 0.062 |
| 3s (H=90) | 0.053 | — |

1s→2s 恶化 45%，2s→3s 再恶化 18%。48 维 dexterous action 的 long-horizon prediction 本质上是 high-dimensional sequence prediction，error 累积很快。这跟 LLM 里 next-token prediction 的误差累积有类似味道，只不过这里 dim=48 远高于 token vocab dimension 的 effective 信息密度。

### Take-away 4:Visual Goal-Conditioning 是最强 trick

Table 4:

| Model | Avg (m) | Final (m) |
|---|---|---|
| Dec+BC | 0.045 | 0.062 |
| Dec+BC + goal image | 0.035 | 0.029 |

Avg distance 降 22%，**Final distance 降 53%**。Final distance 从 0.062m 直接掉到 0.029m，几乎跟 Avg distance 平了。

直觉:goal image 把"未来终点的视觉样子"告诉了 model。原本 forward prediction 里终点是 latent 的 multimodal distribution，现在被 image 一锤定音。这跟 [UniPi](https://arxiv.org/abs/2302.05646)、[SuSIE](https://arxiv.org/abs/2304.14156) 的 video-conditioned policy 思路一致。

### Take-away 5:Data Scaling 单调有效

Figure 5 用 log-scale 横轴画 dataset size vs metric。曲线单调下降，没 saturate。这印证了 paper 的核心论点:数据是 bottleneck，不是 model。

### Take-away 6:Model Size 已饱和

200M 参数 vs 500M 参数的 Dec+BC，metric 完全一样(0.045m avg, 0.062m final)。意味着当前 829 小时数据已经喂饱了 200M 模型。这是 community 友好信号:单卡 RTX 4090 都能跑实验。

## 194 个 Task 的分布逻辑

EgoDex 把 task 分三类，这个分类本身是个聪明的数据采集工程:

**Reversible(76 对 = 152 个 task)**:两个 task 互为 inverse。比如 charge/uncharge device、screw/unscrew、stack/unstack。一类的 final state 落在另一类的 initial state 里，采集时正反两个方向都算有效 demonstration，yield 翻倍。

**Reset-free(28 个 task)**:final state 落在自身 initial state distribution 里。比如 throw and catch ball，重力就是 reset。typing keyboard、flip pages 也是，动作循环本身就是 reset。

**Reset(14 个 task)**:必须显式 reset 的，比如 basic_pick_place、pour、sweep_dustpan、make_sandwich。

Reversible + Reset-free 占 180 个 task，93%，意味着大部分采集不需要 reset 成本。这是数据 scale 起来的关键工程细节。

具体 task 从 tie shoelace、deal cards、flip pages、tighten screw、knit scarf、play piano 到 use chopsticks，dexterity 远超 pick-and-place。Figure 2 的 verb 分布显示 EgoDex 大部分 verb 都有 $10^3$ 以上 demonstration，而 DROID 大部分 verb 在 $10^1$ 以下。

## 我对 paper 的几个直觉判断

### 1. Action representation 是 "pretraining 用的 representation"，不是 deploy 用的

48D keypoint 没有 contact force、没有 finger 间 relative pose、没有 object pose。这种 representation 直接 deploy 到真 robot 上几乎不可能 — 即使是 humanoid，joint 角度、torque 都需要从 keypoint 反推 inverse kinematics。

EgoDex 的定位更像 "robot manipulation 的 ImageNet":用来预训练 visual encoder 或 manipulation prior，downstream 用小规模 robot 数据 fine-tune。Section 6 也提到了 4 条 transfer 路径(co-training、pretrain-SFT、visual encoder、manipulation prior + RL/IL)。这些路径里 EgoDex 扮演"通用语义先验"的角色，跟 [R3M](https://arxiv.org/abs/2203.12601)、[VIP](https://arxiv.org/abs/2210.00030) 的逻辑同源。

### 2. Egocentric 是"未来的数据形态"

EgoDex 的 long-term bet 在 Section 1 里很隐晦但很清楚:未来 AR headset / smart glasses 普及后，egocentric video 会像 internet text 一样 passively 产生。Vision Pro 现在卖得贵，但假设 5 年后 Apple Glasses 量产，每个戴上的人都在被动贡献 manipulation 数据。这跟 LLM 的 internet text scaling 路径完全同构。

### 3. Inverse Dynamics 路线是"video foundation model → robot policy"的关键桥梁

Table 4 的 53% 提升是这篇 paper 最 actionable 的 result。Internet video 的 natural 形式是"有 video、没 action label"，但是有起点 image、终点 image、中间 video。如果训练 inverse dynamics model $f(\mathbf{o}_{start}, \mathbf{o}_{end}) \to \mathbf{a}_{middle}$，internet video 就能转化为 robot policy 数据。EgoDex 用 paired hand pose 把这条路验证了，未来如果 [HaMeR](https://hamer.is.tue.mpg.de/) 这种 3D hand prediction 精度上来，internet video 都能这么用。

### 4. Benchmark 设计有点保守

EgoDex 的 benchmark 评估是 open-loop trajectory prediction，没有 closed-loop roll-out。closed-loop 才是真实 deploy 的场景，open-loop 的 0.04m 误差在 closed-loop 里可能因为 compounding error 爆炸。这是 paper 的 limitation。不过考虑到 EgoDex 主打 dataset 而非 policy，benchmark 保守可以理解。

### 5. Scene Diversity 是最大的 Achilles' Heel

全部 tabletop 环境、全部 Vision Pro 单一设备、全部Apple 的采集者。这跟 ImageNet 当年的"internet image 多样性"还差很远。Section 7 自己承认这点，并提到未来用 [RoboAug](https://arxiv.org/abs/2503.18738) 这类 procedural background randomization 来缓解。但是 augmentation 跟真实 distribution diversity 还是有质的不同。

### 6. 为什么不用 history 是个谜

Appendix A.3 明确写"only the current image observation and proprioceptive state are passed as input (i.e., no history)"。这跟 [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) 的 ablation(history 有用)矛盾。可能是 829h 数据量太大，单 frame 已经够 model 学到 strong prior；也可能是作者为了公平比较 BC/DDPM/FM 都用相同 input，留 history 作 future work。这块值得 community follow-up。

### 7. Foundation Model 路线的想象空间

如果 GPT-5、Gemini 3、Claude 4 这种 multimodal LLM 直接吃 EgoDex 的 video 做 next-frame prediction(或者更激进,next-keyframe prediction)，会 emergently 学到 manipulation policy 吗?这正是 [Cosmos](https://arxiv.org/abs/2501.03575)、[Genie](https://arxiv.org/abs/2401.09024) 这类 world model 路线的 hypothesis。EgoDex 的 video + 3D pose + language 三模态对齐，是为这种 foundation model 准备的"理想训练数据"。

## 总结一句

EgoDex 是 robot manipulation 领域少见的"敢于直接 bet on scale"的工作。它赌的是 Sutton 的 bitter lesson 在 manipulation 上也成立，赌具是 Apple Vision Pro + 829 小时数据 + 194 个 dexterous task + 48D action representation + inverse dynamics benchmark。实验数据支持这个 bet:scale 单调有效、medium model 已 saturate、goal-conditioning 把 error 砍半。这是 robot manipulation 的 ImageNet moment 候选。

## 参考链接

- [EgoDex GitHub](https://github.com/apple/ml-egodex)
- [EgoDex 数据下载](https://ml-site.cdn-apple.com/datasets/egodex/)
- [Bitter Lesson (Sutton)](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
- [DROID](https://droid-dataset.github.io/)
- [Open X-Embodiment](https://robotics-transformer-x.github.io/)
- [Ego4D](https://ego4d-data.org/)
- [EPIC-KITCHENS](https://epic-kitchens.github.io/2024/)
- [EgoMimic](https://arxiv.org/abs/2410.24221)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [UMI](https://arxiv.org/abs/2402.10329)
- [DexCap](https://arxiv.org/abs/2403.07788)
- [HaMeR](https://hamer.is.tue.mpg.de/)
- [Zhou et al. 6D rotation](https://arxiv.org/abs/1812.07035)
- [torchcodec](https://github.com/pytorch/torchcodec)
- [FurnitureBench](https://furniture-bench.github.io/)
- [CLIP](https://arxiv.org/abs/2103.00020)
- [R3M](https://arxiv.org/abs/2203.12601)
- [VIP](https://arxiv.org/abs/2210.00030)
- [MAPLE](https://arxiv.org/abs/2504.06084)
- [Humanoid Policy ~ Human Policy](https://arxiv.org/abs/2503.13441)
- [Cosmos](https://arxiv.org/abs/2501.03575)
- [Genie](https://arxiv.org/abs/2401.09024)
- [Motion Tracks](https://arxiv.org/abs/2501.06994)
- [X-IL](https://github.com/Intelligent-Computing-Lab-Yale/X-IL)
- [RoboAug](https://arxiv.org/abs/2503.18738)
- [SuSIE](https://arxiv.org/abs/2304.14156)
- [UniPi](https://arxiv.org/abs/2302.05646)

---

# EgoDex:大规模 Egocentric Video 的 Dexterous Manipulation 学习数据集

## 1. Paper 的核心动机与"Bitter Lesson"

这篇 Apple 的 paper 直接对标 Rich Sutton 在 2019 年提出的 "bitter lesson" ([Sutton 2019](http://www.incompleteideas.net/IncIdeas/BitterLesson.html))。Sutton 的核心观点:简单的 supervised learning + 海量数据 + 充足算力,远胜过人类手工设计的 heuristic 方法。在 LLM (GPT-4) 和 large vision models (AlexNet→ImageNet 范式) 都被验证了,但是 robot manipulation 一直没法套用这个 recipe,根本原因有两个:

1. **数据不确定**:robot 操作的"正确数据"是什么尚未明朗 — 是 joint torques?end-effector poses?wrist trajectories?还是某种 latent action?
2. **scale 受限**:teleoperation 是当前主流 ([Open X-Embodiment](https://robotics-transformer-x.github.io/), [DROID](https://droid-dataset.github.io/)),但是一个 human 必须站在 robot 旁边手动操作,物理上不可扩展。

EgoDex 提出第三条路:**egocentric human video + 同步采集的 3D hand pose**,这是一种"passively scalable"的数据源,跟 internet 上的 text/image 一样可以无限增长 — 因为未来 wearable headset (Apple Vision Pro, smart glasses) 普及后,数据可以"被动产生"。

## 2. 数据收集:Apple Vision Pro 是关键

数据完全用 Apple Vision Pro (运行 visionOS 2) 收集。选这个硬件有几个关键理由:

- **高分辨率 passthrough** (1080p, 30 FPS, wide FOV),采集者几乎像裸眼一样观察环境。
- **没有 pose offset**:不像 head-mounted camera,Vision Pro 的相机和眼睛几乎完全对齐,记录的就是人看到的。
- **production-grade hand tracking**:用 bare hands 操作,不需要 mocap gloves 等外置硬件。
- **on-device SLAM + 多 camera calibration**:实时精确跟踪每个手指的每个 joint。

数据规模:
- 829 hours 视频
- 90 million frames
- 338,000 episodes
- 194 tabletop manipulation tasks
- 总存储 2.0 TB (压缩后);未压缩会超过 500 TB
- 加载用 [torchcodec](https://github.com/pytorch/torchcodec) 按需解码 frame

## 3. 与现有 dataset 的对比 (Table 1 解析)

| Dataset | # Traj. | # Tasks | # Frames | Lang | Cam Ext | Dexterous | Method |
|---|---|---|---|---|---|---|---|
| RoboTurk | 2k | 3 | 12M | ✗ | ✗ | ✗ | teleop |
| RoboNet | 162k | n/a | 15M | ✗ | ✗ | ✗ | scripted |
| BridgeData V2 | 60k | 13 | 2M | ✓ | ✗ | ✗ | teleop+scripted |
| DROID | 76k | 86 | 19M | ✓ | ✓ | ✗ | teleop |
| EgoMimic | 2k | 3 | 0.4M | ✗ | ✓ | ✗ | egocentric |
| EPIC-KITCHENS | 40k | 125 | 12M | ✓ | ✗ | ✗ | egocentric |
| HOI4D | 4k | 54 | 2M | ✗ | ✗ | ✓ | egocentric |
| Ego4D (HOI) | 89k | n/a | 21M | ✓ | ✗ | ✗ | egocentric |
| **EgoDex** | **338k** | **194** | **90M** | ✓ | ✓ | ✓ | egocentric |

EgoDex 在 trajectory 数量、task 数量、frame 数量上都远超第二名。"Dexterous annotation"这里特指 multi-finger hand pose annotation,不包括 low-fidelity 的 parallel-jaw gripper 或 wrist-only tracking。

## 4. Action Representation:48 维的 dexterous action

因为关注的是 dexterous manipulation,action 必须能表达足够的 bimanual dexterity。EgoDex 选择了以下 representation:

- 每只 wrist 的 3D position (3维)
- 每只 wrist 的 6D orientation (采用 [Zhou et al. 2018](https://arxiv.org/abs/1812.07035) 的连续 rotation representation,避免 quaternion 的 discontinuity 问题)
- 5 个 fingertip 的 3D position (5 × 3 = 15维)

每只手: $3 + 6 + 15 = 24$ 维。双手: $24 \times 2 = 48$ 维。

公式表达:

$$
\mathbf{a}_t \in \mathbb{R}^{48}, \quad \mathbf{a}_t = \left[ \mathbf{p}_{L}^{\text{wrist}}, \mathbf{r}_{L}^{\text{wrist}}, \mathbf{p}_{L}^{\text{tip}_1}, \dots, \mathbf{p}_{L}^{\text{tip}_5}, \mathbf{p}_{R}^{\text{wrist}}, \mathbf{r}_{R}^{\text{wrist}}, \mathbf{p}_{R}^{\text{tip}_1}, \dots, \mathbf{p}_{R}^{\text{tip}_5} \right]
$$

其中:
- $\mathbf{p}_{L}^{\text{wrist}}, \mathbf{p}_{R}^{\text{wrist}} \in \mathbb{R}^3$ 是左右 wrist 的 3D position
- $\mathbf{r}_{L}^{\text{wrist}}, \mathbf{r}_{R}^{\text{wrist}} \in \mathbb{R}^6$ 是 6D rotation representation (连续表示)
- $\mathbf{p}_{L}^{\text{tip}_i}, \mathbf{p}_{R}^{\text{tip}_i} \in \mathbb{R}^3$ 是第 $i$ 个 fingertip 的 3D position

注意:pose 表达在当前 camera frame 里,action chunk 是 relative trajectory (类似 [UMI](https://arxiv.org/abs/2402.10329) 的设计)。

## 5. Benchmark 任务的两个公式

### Benchmark 1:Dexterous Trajectory Prediction (forward prediction)

$$
f_\theta \left( \mathbf{o}_{0..t}, \mathbf{s}_{0..t}, l \right) = \hat{\mathbf{a}}_{t:t+H}
$$

变量含义:
- $f_\theta$:参数为 $\theta$ 的神经网络 (policy)
- $\mathbf{o}_{0..t}$:从时间 0 到 $t$ 的 egocentric image observations (图像序列)
- $\mathbf{s}_{0..t}$:从时间 0 到 $t$ 的 skeletal pose observations (3D 关节 pose 序列)
- $l$:自然语言任务描述 (经过 GPT-4 整合)
- $\hat{\mathbf{a}}_{t:t+H}$:预测的 action chunk,从 $t$ 到 $t+H$
- $H$:prediction horizon (例如 $H=30$ 对应 1 秒,$H=60$ 对应 2 秒)

### Benchmark 2:Inverse Dynamics (visually goal-conditioned)

$$
f_\theta \left( \mathbf{o}_{0..t}, \mathbf{s}_{0..t}, \mathbf{o}_{t+H}, l \right) = \hat{\mathbf{a}}_{t:t+H}
$$

新增变量:
- $\mathbf{o}_{t+H}$:goal image observation at horizon 结束时刻

这个公式可以理解为 visually goal-conditioned policy:给定起始状态 + 终止状态的图像,prediction 中间的 action sequence。

## 6. "Best of K" Evaluation Metric

由于人类 motion 高度 multimodal (比如放水果到篮子里,可以放不同位置、不同速度、不同轨迹),单 sample 评估不够。论文采用 "best of K":

$$
\text{Err} = \frac{1}{N} \sum_{n=1}^{N} \min_{k \in \{1, \dots, K\}} \left[ \frac{1}{H} \sum_{t=1}^{H} \frac{1}{12} \sum_{j=1}^{12} \left\| \mathbf{a}_{t,j}^{(k)} - \mathbf{a}_{t,j}^{\text{GT}} \right\|_2 \right]
$$

变量解释:
- $N$:test set 样本数
- $K$:每个样本采样次数
- $H$:prediction horizon
- 12:12 个 keypoint (2 wrist + 10 fingertip,每只手 1 wrist + 5 fingertip,双手共 12)
- $\mathbf{a}_{t,j}^{(k)}$:第 $k$ 次采样在时刻 $t$ 的第 $j$ 个 keypoint 的 3D position
- $\mathbf{a}_{t,j}^{\text{GT}}$:ground truth

最终值的单位是 meters (3D 空间平均误差)。Deterministic model (BC) 的 metric 与 $K$ 无关;stochastic model (DDPM, FM) 的 metric 随 $K$ 增大而改善,因为更多采样机会。

## 7. 模型架构 (Figure 7 解析)

Paper 训练了 14 个 model,2 种架构 × 3 种 policy representation:

### 架构:
- **Encoder-Decoder (EncDec)**:image + proprioceptive state 输入 encoder,latent 再 decoder 输出 action chunk
- **Decoder-only (Dec)**:类似 GPT/LLaMA,token 化输入后 causal attention

### Policy representations:
- **BC (Behavior Cloning)**:直接回归 $\hat{\mathbf{a}}_{t:t+H}$,deterministic
- **DDPM (Denoising Diffusion Probabilistic Models)**:迭代去噪采样,stochastic
- **FM (Flow Matching)**:连续时间的 stochastic 生成模型,类似 rectified flow

输入侧:
- Image observation → ResNet encoder (224×224 输入)
- Language annotation → frozen [CLIP](https://arxiv.org/abs/2103.00020) encoder
- 仅当前 frame + proprioceptive state,**no history**

训练设置:
- 50,000 gradient steps
- batch size 2048 (256 per GPU × 8 A100)
- Adam, lr = 1e-4
- DDPM/FM: 16 sampling steps
- 72 小时训练

## 8. 实验数据详解

### Table 2:Trajectory Prediction (2 秒 horizon)

| Model | K=1 Avg | K=5 Avg | K=10 Avg | K=1 Final | K=5 Final | K=10 Final |
|---|---|---|---|---|---|---|
| Dec+BC | 0.045 | 0.045 | 0.045 | 0.062 | 0.062 | 0.062 |
| Dec+DDPM | 0.053 | 0.044 | 0.041 | 0.071 | 0.050 | 0.044 |
| Dec+FM | 0.052 | 0.042 | 0.040 | 0.071 | 0.049 | 0.043 |
| EncDec+BC | 0.044 | 0.044 | 0.044 | 0.060 | 0.060 | 0.060 |
| EncDec+DDPM | 0.052 | 0.042 | 0.039 | 0.071 | 0.048 | 0.043 |
| **EncDec+FM** | 0.051 | 0.041 | **0.038** | 0.070 | 0.047 | **0.041** |

关键观察:
- **EncDec > Dec**:encoder-decoder 一致性优于 decoder-only,但差距很小 (1-3 mm)
- **K=1 时 BC 最好**:EncDec+BC 0.044m vs EncDec+FM 0.051m,BC 平均预测质量更好 (BC 学的是 conditional mean)
- **K=5/10 时 FM/DDPM 最好**:EncDec+FM 在 K=10 时 0.038m,比 BC 的 0.044m 好 14%。Stochastic model 通过多次采样能命中 ground truth 的 mode
- **Final distance > Avg distance**:越往后预测越难,合理

### Table 3:Prediction Horizon 影响

| Horizon | Avg (m) | Final (m) |
|---|---|---|
| H=30 (1s) | 0.031 | 0.049 |
| H=60 (2s) | 0.045 | 0.062 |
| H=90 (3s) | 0.053 | — |

- 从 2s→1s:Avg 提升 31%,Final 提升 21%
- 从 2s→3s:Avg 恶化 18%

48 维 dexterous action 的长程 prediction 是非常 hard 的 task。

### Table 4:Visual Goal-Conditioning

| Model | Avg (m) | Final (m) |
|---|---|---|
| Dec+BC | 0.045 | 0.062 |
| Dec+BC + goal image | 0.035 | 0.029 |

- Avg distance 降低 22%
- **Final distance 降低 53%** ← 非常显著

直觉:goal image 充当"visual anchor",grounding 了 trajectory 的 endpoint,极大缓解 multimodality 问题。Final distance 几乎和 avg distance 相等 (0.035 vs 0.029),说明 goal image 直接告诉了"终点长什么样"。

### Figure 5:Dataset Size Scaling

性能随 dataset size 单调改善 (log-scale 横轴)。这印证了 paper 的核心论点:scale matters。

### Model size 实验:
- 200M 参数 vs 500M 参数的 Dec+BC:Avg distance 都是 0.045m,Final distance 都是 0.062m
- 当前数据量级下,medium-size 模型已 saturate,这对 community 是好消息:commodity GPU (如 RTX 4090) 即可参与研究

## 9. Task 类型分类与 194 个 Task 全列表

### 三类 task 定义 (Section 3.4):

1. **Reversible (76 对,共 152 个 task)**:两个 task 互为 inverse。比如 charge/uncharge device,open/close box。一类的 final state 落在另一类的 initial state distribution 里。无 reset 成本。
2. **Reset-free (28 个 task)**:final state 落在自身 initial state distribution 内。比如 throw and catch ball (重力就是 reset)。
3. **Reset (14 个 task)**:每次 demonstration 后必须 reset 到初始状态。比如 basic_pick_place, pour, sweep_dustpan。

这种分类很巧妙:reversible + reset-free task 共 180 个,占 93%,意味着大部分采集不需要 reset,极大提升了 yield。

### 194 个 task 包含 (摘录):
- 日常物品操作:zip/unzip bag, open/close case, fold/unfold paper
- Furniture assembly/disassembly (来自 [FurnitureBench](https://arxiv.org/abs/2305.12821))
- 精细动作:tie/untie shoelace, thread bead necklace, deal/gather cards
- 工具使用:use chopsticks, paint/clean brush, point and click remote, type keyboard
- 烹饪:make sandwich, fry egg, boil/serve egg
- 物体堆叠/排列:stack/unstack cups, jenga, dominoes, tetra board
- 还有趣味:use Rubik's cube, play piano, play mancala, flip coin

## 10. Skeletal Joints 完整列表

每个 hand 25 个 joint (1 wrist + 4 finger × 5 joint + thumb 5 joint):

**Upper Body** (20 joints):hip, spine1-7, neck1-4, leftShoulder/Arm/Forearm/Hand, rightShoulder/Arm/Forearm/Hand

**每只手 25 joints**(以 left hand 为例):
- 4 个 finger 各有:IntermediateBase, IntermediateTip, Knuckle, Metacarpal, Tip (共 4 × 5 = 20)
- Thumb:IntermediateBase, IntermediateTip, Knuckle, Tip (4)
- Wrist (leftHand):1
- 共 25 joints

Confidence values 0-1:0 表示 fully occluded。注意 wrist confidence 表示整只 hand 是否被检测,而 finger joint confidence 是相对于 wrist 的,所以"wrist 低 + finger 高" 不可信。

## 11. 关键洞察与 Intuition Building

### 洞察 1:BC 学的是"average",FM/DDPM 学的是"distribution"

从 Table 2 可以看到 BC 在 K=1 时最好,在 K=10 时最差。这印证了经典生成模型 vs deterministic regression 的权衡:
- BC 的 L2 loss → 学 conditional expectation (multimodal 时 mode averaging)
- FM/DDPM → 显式建模 conditional distribution,可以 sample 出某个 specific mode

当 multimodality 严重时,FM 的"命中 mode"能力 (K=10 0.038m) 比 BC 的"平均" (0.044m) 更优。

### 洞察 2:Visual Goal Conditioning 是"廉价的 multimodality 解决方案"

Table 4 显示加 goal image 让 Final distance 从 0.062m 直接降到 0.029m,这是数量级的改善。直觉:
- Forward prediction 必须从 conditional distribution $p(\mathbf{a}_{t:t+H} | \mathbf{o}_{0..t})$ 里 sample
- Goal-conditioned 从 $p(\mathbf{a}_{t:t+H} | \mathbf{o}_{0..t}, \mathbf{o}_{t+H})$ 里 sample,条件更具体,distribution 更 sharp

这让人联想到 [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)、[Universal Manipulation Interface (UMI)](https://arxiv.org/abs/2402.10329) 等工作。inverse dynamics 这条路径可能是从 internet video 学习 robot policy 的 key insight。

### 洞察 3:Passively Scalable 的"哲学"

EgoDex 的核心 thesis:robot data 必须像 internet text/image 一样"passively 产生"。Apple Vision Pro 这种 AR headset 让数据采集变成了"穿上设备、做日常动作"。这跟 [DexCap](https://arxiv.org/abs/2403.07788) (mocap glove)、[UMI](https://arxiv.org/abs/2402.10329) (手持 gripper) 等"主动采集"方法形成鲜明对比 — 后者即使再便宜,人也不会为了"被采集"而做 1000 次 pick-and-place。

### 洞察 4:Scale > Embodiment

EgoDex 用 human hand 作为 universal embodiment,paper Section 6 讨论 transfer 到 humanoid robot 的几种路径:
1. 直接 zero-shot transfer (需要 humanoid 形态/dynamics 完全匹配)
2. Co-training with small robot dataset ([EgoMimic](https://arxiv.org/abs/2410.24221), [Humanoid Policy ~ Human Policy](https://arxiv.org/abs/2503.13441))
3. Pretraining + SFT (类似 LLM 的训练 recipe)
4. Train visual encoder for downstream imitation learning ([R3M](https://arxiv.org/abs/2203.12601), [VIP](https://arxiv.org/abs/2210.00030))
5. Learn manipulation priors + RL/IL fine-tuning ([MAPLE](https://arxiv.org/abs/2504.06084))

这五种路径的"前提"都是 EgoDex 这种 scale 的数据存在。

### 洞察 5:Video Compression 的"工程层"考量

Raw data 500 TB,压缩后 2 TB,压缩比 250×。这是一个经常被忽视的工程现实:internet-scale video data 必须用现代 video codec (H.264/H.265/AV1) 压缩。torchcodec (PyTorch 官方) 解决了 GPU 上随机 frame access 的 decoding 效率问题,这对训练效率至关重要。

## 12. 与相关工作的对比

### vs [EgoMimic](https://arxiv.org/abs/2410.24221) (2024, ICLR'25)
- EgoMimic:4 小时数据,只跟踪 wrist
- EgoDex:829 小时数据,跟踪 25 joints/hand + upper body
- 200× scale-up,500× annotation complexity

### vs [HOT3D](https://arxiv.org/abs/2410.19344) (Apple CVPR 2025)
- HOT3D 也是 egocentric multi-view hand-object tracking,但是更聚焦于 hand pose estimation benchmark,而非 manipulation learning

### vs [Ego4D](https://ego4d-data.org/)
- Ego4D:3700 小时 egocentric video,但是无 paired 3D hand annotation,且包含大量非 manipulation 场景
- EgoDex:更短但是 focused,且 manipulation 行为丰富

### vs [HaMeR](https://arxiv.org/abs/2307.16789) 路线
- Ren et al. [Motion Tracks](https://arxiv.org/abs/2501.06994) 用 HaMeR 后处理 internet video 得到 hand pose
- EgoDex 直接在采集时获得 3D pose,精度远高于 post-hoc 单目预测

## 13. Limitations 与未来方向

Paper 自承:
- **Scene diversity 有限**:全部 tabletop 环境
- **Occlusion 时 annotation 不准**:heavy occlusion (towel folding) 或 high speed motion 时,ARKit 预测本身有误差
- **未来方向**:procedural background randomization (类似 [RoboAug](https://arxiv.org/abs/2503.18738)) + 多环境采集

我的额外联想:
- 当前数据只有 Apple Vision Pro 一种设备,如果未来 Meta Quest, Snap Spectacles 等也加入,需要 camera intrinsic calibration 协议
- Action representation 是 48D keypoints,缺少 contact force 信息 — 这是 limitation:dexterous manipulation 的核心是 contact-rich,光有 keypoint position 不够
- 没有 object pose annotation,只有 hand pose — 如果 object pose 也标注,可以学习 HOI (hand-object interaction) 更精细的模型
- prediction horizon 最大 90 帧 (3 秒),对于 long-horizon manipulation (比如组装宜家家具) 不够

## 14. 我的开放问题 / 直觉

1. **从 keypoint 到 motor control 的鸿沟**:EgoDex 只给 keypoint trajectory,如何 transfer 到真实 robot?关节空间、force control、contact sensing这些 robot-specific 问题怎么解?
2. **Best-of-K metric 在 deploy 时有用吗?** Deploy 时 robot 只能执行一次 action,K=10 metric 反映的是"如果 model 重试 10 次,最好的那次有多好"。真实 deployment 时 K=1,所以应该看 K=1 metric。Paper 没强调这点。
3. **48D action 是否过简?** Finger 之间的 contact、slip、force 都没建模。但是从 paper 的视角,EgoDex 是"pretraining"用途,downstream 可以加 force/tactile sensing fine-tune。
4. **为什么 EgoDex 不在 encoder 里加 history?** Appendix A.3 提到"only current image observation and proprioceptive state are passed as input"。这跟 Diffusion Policy 的实验结论 (history 有用) 矛盾。可能是 EgoDex 数据量太大,单 frame 已足;或者作者保守,留下 future work 空间。
5. **与 foundation model 的关系**:如果 GPT-5、Gemini 3、Claude 4 等多模态 LLM 直接吃 EgoDex 视频做 next-frame prediction,会不会 emergently 学到 manipulation policy?这正是 world model 路线 ([Cosmos](https://arxiv.org/abs/2501.03575), [Genie](https://arxiv.org/abs/2401.09024)) 的 hypothesis。

## 15. 参考链接

- [EgoDex GitHub](https://github.com/apple/ml-egodex)
- [EgoDex 数据下载](https://ml-site.cdn-apple.com/datasets/egodex/) (part1-5.zip, test.zip, extra.zip)
- [Bitter Lesson by Rich Sutton](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
- [DROID Dataset](https://droid-dataset.github.io/)
- [Open X-Embodiment / RT-X](https://robotics-transformer-x.github.io/)
- [Ego4D](https://ego4d-data.org/)
- [EPIC-KITCHENS](https://epic-kitchens.github.io/2024/)
- [Diffusion Policy (Chi et al.)](https://diffusion-policy.cs.columbia.edu/)
- [Universal Manipulation Interface (UMI)](https://arxiv.org/abs/2402.10329)
- [DexCap](https://arxiv.org/abs/2403.07788)
- [HaMeR](https://hamer.is.tue.mpg.de/)
- [6D Rotation Representation (Zhou et al.)](https://arxiv.org/abs/1812.07035)
- [torchcodec](https://github.com/pytorch/torchcodec)
- [FurnitureBench](https://furniture-bench.github.io/)
- [CLIP](https://arxiv.org/abs/2103.00020)
- [EgoMimic](https://arxiv.org/abs/2410.24221)
- [Humanoid Policy ~ Human Policy](https://arxiv.org/abs/2503.13441)
- [HOT3D (Apple, CVPR 2025)](https://arxiv.org/abs/2410.19344)
- [Motion Tracks (Ren et al.)](https://arxiv.org/abs/2501.06994)
- [MAPLE](https://arxiv.org/abs/2504.06084)
- [X-IL Framework](https://github.com/Intelligent-Computing-Lab-Yale/X-IL)
- [Cosmos World Foundation Model (NVIDIA)](https://arxiv.org/abs/2501.03575)
- [Genie (DeepMind)](https://arxiv.org/abs/2401.09024)
- [R3M](https://arxiv.org/abs/2203.12601)
- [VIP](https://arxiv.org/abs/2210.00030)

## 16. 总结

EgoDex 是 robot manipulation 领域少见的、敢于直接贯彻 Sutton "bitter lesson" 的工作。核心 bet:egocentric video + paired 3D hand pose 是 robot manipulation 的 "ImageNet moment"。829 小时数据、194 个 task、90M frame 的规模,让简单的 supervised learning 方法 (BC, DDPM, FM) 也能学到 48D dexterous trajectory prediction。EncDec+FM 在 K=10 下达到 0.038m 平均误差,visual goal-conditioning 把 final error 砍到 0.029m。Performance 随数据 scale 单调改善,medium-size model 已 saturate — 意味着 bottleneck 是数据不是算力。这正是 ImageNet 时刻的特征。
