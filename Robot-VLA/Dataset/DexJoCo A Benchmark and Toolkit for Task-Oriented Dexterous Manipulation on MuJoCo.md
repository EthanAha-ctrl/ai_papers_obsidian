---
source_pdf: DexJoCo A Benchmark and Toolkit for Task-Oriented Dexterous Manipulation
  on MuJoCo.pdf
paper_sha256: 28441f5042ecfb2f52480dce0db6fba3fdf575f411139e87251244a5f6d39101
processed_at: '2026-08-03T20:32:57-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 DexJoCo

## 一句话总结

现在 robot learning 圈子几乎都在玩 gripper（就那种两指一开一合的夹爪），但这帮人偏要搞 dexterous hand（就是那种像人手一样有很多 finger joint 的灵巧手），然后发现现有所有 policy 都不太 work，于是做了一个 benchmark 把这个尴尬暴露出来。

## 为什么要做这个 benchmark

**背景**: 你看 [RT-2](https://arxiv.org/abs/2307.15818)、[OpenVLA](https://arxiv.org/abs/2406.09246)、[π0.5](https://arxiv.org/abs/2504.16054)、[GR00T N1.5](https://arxiv.org/abs/2503.14734) 这些 VLA model，action space 基本都是 7-DoF arm + 1-DoF gripper = 8 维。但 dexterous hand 比如 Allegro Hand 是 16-DoF，bimanual 双手场景就是 7+16+7+16 = 46 维 action。维度爆炸是表面问题，真正的问题是 joint 之间有 coupling（腱耦合、kinematic chain 约束），现有 action head 根本 capture 不了这种结构。

**现有 dexterous benchmark 的四个毛病**:
1. 很多 benchmark 比如 [UniDexGrasp](https://arxiv.org/abs/2304.00810) 只建模 hand 不建模 arm，workspace 完全不现实，真机没法 deploy
2. 要么搞 in-hand manipulation（手里转魔方那种），functional 太单一；要么搞 pick-and-place，体现不出 dexterous hand 比 gripper 强在哪
3. 数据收集难，motion planning 生成不出自然的 dexterous trajectory，RL 生成的又不像人
4. 没有 unified language instruction 和数据格式给现代 VLA 用

所以 DexJoCo 的 motivation 就是: **做一个真正能暴露 dexterous hand 独特价值、且数据收集 low-cost 的 benchmark**。

## 11 个 task 长啥样

**Tool-Use 类**（6 个）:
- Hammer Nail（锤钉子）
- Click Mouse（按鼠标）
- Pick Bucket（提水桶）
- Pinch Tongs（夹夹子）
- Fold Glasses（折眼镜）
- Water Plant（浇水）

**Bimanual + Reasoning / Long-horizon 类**（5 个，标 /B）:
- Unlock iPad /B（双手解锁 iPad 输密码）
- Hanoi /B（双手玩汉诺塔）
- Assembly /B（双手插 peg 进 socket）
- Microwave /B（双手用微波炉热热狗）
- Photograph /B（双手拿相机拍 logo）

这些 task 的设计哲学是: **每个 task 都必须依赖 dexterous hand 的 fine-grained finger coordination 才能完成，gripper 搞不定**。比如 Pinch Tongs 需要 squeeze + release，Water Plant 需要 press handle 到特定 joint threshold 才出水。

## Task 的形式化定义

$$\mathcal{T} = (\mathcal{O}, \mathcal{G})$$

- $\mathcal{O} = \{o_1, ..., o_m\}$: scene 里 m 个可交互物体
- $\mathcal{G} = \{g_{\text{seq}}, g_{\text{pose}}, g_{\text{joint}}, g_{\text{contact}}\}$: 4 类 success constraint

**4 个 constraint 分别是**:
- $g_{\text{seq}}$: 时序约束。比如 Hanoi 必须先移最上面的 disk
- $g_{\text{pose}}$: 物体 6D pose 条件。比如 nail 要被砸到特定深度
- $g_{\text{joint}}$: 铰接物体 joint state。比如 watering can handle 要按到 threshold
- $g_{\text{contact}}$: 接触条件。比如 finger 必须碰到 button

只有所有相关 constraint 同时满足才算成功。这种 compositional design 让 long-horizon task 的 evaluation 非常严格。

## 数据收集硬件: 2300 美金搞定

- **Rokoko Smartgloves**: 抓 finger pose，避免 camera occlusion
- **2× HTC Vive Tracker + 2× Base Station**: 抓 wrist 6D pose 控制 Franka end-effector
- 3D 打印一个 connector 把它们组装起来

对比 [DexCap](https://arxiv.org/abs/2403.07788) 用 expensive mocap、[Bunny-VisionPro](https://arxiv.org/abs/2407.03162) 用 Apple Vision Pro，这个组合 cost 和 usability 平衡得很好。

## GeoRT Retargeting 公式详解

这是技术含量最高的部分。Human hand 和 Allegro Hand 的 kinematic structure 完全不同，直接线性映射不可行。所以用 [GeoRT](https://arxiv.org/abs/2406.11468) 这个 self-supervised retargeting 方法，loss 是 5 项的加权和:

$$\mathcal{L} = \mathcal{L}_{\text{dir}} + \lambda_1 \mathcal{L}_{\text{cover}} + \lambda_2 \mathcal{L}_{\text{flat}} + \lambda_3 \mathcal{L}_{\text{pinch}} + \lambda_4 \mathcal{L}_{\text{col}}$$

**每一项的 intuition**:

- $\mathcal{L}_{\text{dir}}$ (direction preservation): human index finger 往上动，robot index finger 也得往上动，不能跑偏。这保证 teleoperation 的 intuitiveness
- $\mathcal{L}_{\text{cover}}$ (workspace coverage): human 能做出的所有 hand pose 都得能 map 到 robot reachable workspace，不能某些 human pose map 到 robot joint limit
- $\mathcal{L}_{\text{flat}}$ (sensitivity flatness): mapping sensitivity 要均匀，不能某些区域 human 动 1cm robot 动 10cm、其他区域 human 动 5cm robot 几乎不动。避免"dead zone"
- $\mathcal{L}_{\text{pinch}}$ (pinch preservation): thumb-index pinch 是 dexterous manipulation 最 fundamental 的 primitive，retargeting 绝对不能破坏 pinch 对齐
- $\mathcal{L}_{\text{col}}$ (self-collision avoidance): 高 DoF hand 容易 finger 互相穿透，得避免

$\lambda_1, \lambda_2, \lambda_3, \lambda_4$ 是平衡超参。这种 multi-objective design 的理念是: 单一 forward kinematics loss 不够，必须显式分解成多个 interpretable sub-objectives。

**关键 insight**: "Only fingertip workspaces are recorded during data collection and used for training" — retargeting model 是 task-agnostic 的，不用每个 task 重训。这是 pipeline 可 scale 的关键。

## Domain Randomization 的两个 level

**rand-obj** (轻度):
- Object 在桌面上 (x, y) 平面 randomize
- Table height: $\Delta h \sim U(0, 0.05)$ m

**rand-full** (重度，在 rand-obj 基础上加):
- 50 个 preset camera pose (球面 dense 采样后选 occlusion 最小的 50 个)
- 光照 randomize: 位置扰动 $U(-0.3, 0.3)$，方向扰动 $U(-0.4, 0.4)$，diffuse RGB $U(0.3, 0.8)$，ambient RGB $U(0.3, 0.7)$
- 桌面纹理从 texture library 采样

**最 smart 的 trick**: visual randomization 不需要重新 teleop。同一条 trajectory 在不同 rendering setting 下 replay，得到视觉上不同但 action identical 的 training data。相当于免费的 data augmentation。这本质上是 imitation learning 特有的优势——action 来自 human demo，所以 visual appearance 变化完全不影响 action quality，把 perception 和 action 解耦了。

## 实验: 5 个 policy 大乱斗

比的是: [ACT](https://arxiv.org/abs/2304.13705)、[Diffusion Policy](https://arxiv.org/abs/2303.04137) (DP-T Transformer 和 DP-C CNN 两个变体)、[π0.5](https://arxiv.org/abs/2504.16054)、[GR00T N1.5](https://arxiv.org/abs/2503.14734)。

所有 policy 都用 action chunking formulation:

$$\mathcal{P}(a_{t:t+k-1}) = \pi_\theta(a_{t:t+k-1} \mid s_{t-h+1:t}, l)$$

变量含义:
- $a_{t:t+k-1}$: time step $t$ 开始的 $k$ 步 action chunk
- $s_{t-h+1:t}$: $h$ 帧 historical observation window
- $l$: optional language instruction
- $\pi_\theta$: 参数 $\theta$ 的 policy
- $\mathcal{P}$: 条件概率分布

四个 baseline 的区别仅在 $\mathcal{P}$ 怎么参数化: ACT 用 C-VAE，DP 用 diffusion，π0.5 和 GR00T N1.5 用 flow-matching。

**bimanual task 的 action head 问题**: π0.5 和 GR00T N1.5 默认 32 维 action head，但 bimanual 需要 46 维。作者用 **partial pretrain-AH** 策略: 保留前 32 维 pretrained 权重，额外 14 维 random init。

### 核心实验结果 (Table 2)

| Model | rand-obj Avg | rand-full Avg | Drop |
|-------|--------------|---------------|------|
| DP-T | 50.4% | 20.0% | -30.4% |
| DP-C | 47.6% | 28.4% | -19.2% |
| ACT | 35.5% | 22.7% | -12.8% |
| π0.5 | 52.5% | 34.1% | -18.4% |
| GR00T N1.5 | 40.2% | 30.5% | -9.7% |

**三个 critical observation**:

1. **Visual randomization 让所有 policy 大幅 drop**，说明现有 VLA 的 visual robustness 严重不足。pretrain on web data 并不能直接 transfer 到 diverse visual conditions。

2. **π0.5 在单臂 task 上 dominant，但 bimanual 优势消失**。Hammer Nail 84.7% vs DP-T 81.3%，但 Hanoi、Assembly 这种 bimanual long-horizon task 上和 DP-T 差不多。这就是 action head random init 的代价——额外 14 维从 scratch 学，抵消了 large-scale pretraining 的优势。

3. **DP-C 在 Unlock iPad (52.0% vs π0.5 12.0%) 和 Pinch Tongs (57.3% vs π0.5 24.0%) 上意外最强**。作者 hypothesis 是因为 DP-C 唯一用 [FiLM](https://arxiv.org/abs/1709.07871) injection 而非 self/cross attention。

**我对 FiLM 优势的 alternative hypothesis**: FiLM conditioning 是 multiplicative 的，$y = \gamma \odot x + \beta$，这种 element-wise modulation 对 precise spatial localization (button 位置、tongs hinge) 比 attention 的 soft aggregation 更适合。Attention tends to smooth over fine details，而 FiLM preserves spatial resolution。

## 失败模式分析 (Figure 5, 6)

这个 breakdown 极其 informative:

**Button-pressing tasks**: Policy 能 perceive object (能 pick up tablet，能 push mouse onto mousepad)，但 fail to click intended button。说明 perception 能识别 object 存在，但无法 ground 到 affordance——"哪里是可交互的"。

**Insertion tasks (Assembly, Hanoi)**: Insertion 步骤失败率极高。这是典型 contact-rich manipulation，vision-only policy 无法感知 insertion 过程中的 force feedback，判断不了 peg 是否对准 socket。

**Pinch Tongs**: Policy 能 grasp tongs 但 fail to squeeze and release。作者认为是 "insufficient temporal memory"——policy 需要记住"已经 grasp 了"这个 state 才能 execute 后续 squeeze。暗示现有 action chunking 的 history window $h$ 可能不够长。

**Microwave**: Policy 能 place hot dog 进 microwave，但随后 withdraw hand 时把 hot dog 也带出来了。这是非常 interesting 的 failure mode——policy 学到了 "place" 的 action primitive，但没学到 "release" 的 temporal structure，place 和 retrieve 纠缠在一起。

## Multi-task training 暴露的 negative transfer (Table 3)

Multi-task training 结果:
- DP-T 在所有 task 上都 degrade (50.4% → 33.2%)
- π0.5 在 Click Mouse 和 Pinch Tongs 上有提升，但 avg 下降 (52.5% → 45.5%)

这与 general expectation 相悖——foundation model 应该 multi-task 更 robust 才对。

**我的 interpretation**: dexterous task 之间存在 **action space conflict**。Hammer Nail 需要 forceful swing，Fold Glasses 需要 gentle precision。这些 task 的 action distribution 差异巨大，shared backbone 难以同时 accommodate。这是 dexterous manipulation 特有的 challenge——gripper-based task 的 action distribution 相对 homogeneous (都是 approach-grasp-lift)，但 dexterous task 的 action manifold 是高度 multi-modal 的。

## 语言 generalization 的彻底失败 (Figure 7)

这是 paper 最 sobering 的实验。训 π0.5 在 Unlock iPad 上用 single-digit password 1-5，测试:
- Seen digits: "1", "2", "4"
- Arithmetic: "1+1", "2+2"
- English words: "two", "one plus one"

结果 (average precision %):
- "1": 15.3%
- "2": 30.7%
- "4": 4.0%
- "1+1": 24.7%
- "2+2": 1.3%
- "two": 30.0%
- "one plus one": 20.7%

表面看 "two" 和 "1+1" 30%+ precision 似乎不错，但 Figure 7 heatmap 揭示真相: **model 有一个固定的 output bias 倾向于输出 "2"**，无论 instruction 是什么。当 correct answer 恰好是 "2" 时 bias 碰巧命中；当 correct answer 是 "4" 时 precision 降到 4%。

Quantitative analysis:
- Chi-square test rejects independence ($p = 2.15 \times 10^{-4}$)，说明 model 确实对 language 有 response
- 但 Normalized Mutual Information 只有 0.018，几乎可忽略
- Average JS divergence across instruction pairs: 0.026 (max 0.057)

**核心结论**: VLA model 的 "language understanding" 在 compositional generalization 上完全失效。Model 学到的是 instruction marginal distribution 上的 **mode collapse**，而非真正的 language-conditioned action generation。这与最近 [OpenVLA critique](https://arxiv.org/abs/2412.03163) 等工作的观察一致——当前 VLA 的 language grounding 是 fragile 的。

## Action head reinitialization 实验 (Table 3 rand-AH)

对比 partial pretrain-AH (Table 2) 和 fully random reinitialization (Table 3 rand-AH)，发现 **retaining pretrained weights 在 most task 上更好**。

这个结果有点反直觉——可能 expect random init 让 action head 可以从 scratch 学 bimanual-specific representation。但实验显示 pretrained 的低维 action representation 对 bimanual 也有 transfer value。

**我的 interpretation**: 这暗示 manipulation 的 low-dimensional structure 存在某种 universal property (比如 end-effector pose 控制)，即使 bimanual 场景，单臂的 action prior 依然 useful。这与 [EigenPose](https://arxiv.org/abs/2406.11468) 等工作中 "action can be decomposed into task-agnostic 和 task-specific components" 的观察一致。

## Asynchronous inference 的隐藏 implication

借鉴 [SmolVLA](https://arxiv.org/abs/2506.01844): 下一个 action chunk 在当前 chunk 执行时生成，消除 idle waiting。Overlapping chunks 用 temporal ensembling 平滑。

这个 design 有 profound 的 implication: **inference frequency 直接影响 performance**。轻量 policy (DP-T ~100M) 推理快，能用更 recent observation，reactivity 更好。重量级 policy (π0.5) 推理慢，可能用 stale observation。这解释了为什么 DP-T 在某些 task 上能 competitive π0.5: 原因是 DP-T 的 inference latency 更小，而非 representation 更强。

## Discussion 部分的两个深层 insight

### 1. Lack of dexterous hand-centric foundation models

当前 VLA model 几乎都 pretrain 在 gripper-based data 上，导致 action space mismatch。这不仅仅是 dimension mismatch，更深层是 **joint coupling**。Gripper 1-DoF 是 trivial 的 (open/close)，但 dexterous hand 16 个 joint 之间存在 strong biomechanical coupling (index finger 的 MCP/PIP/DIP joint 腱耦合)。现有 action head 把每个 joint 独立建模，无法 capture 这种 coupling structure。

这 motivates **embodiment-aware representation**: 未来 foundation model pretrain 时需要考虑 hand kinematic structure，比如用 GNN 编码 joint connectivity，或用 hand-specific tokenizer。

### 2. Vision-only policy 的 fundamental limitation

Paper 最 provocative 的 claim: **vision-only (即使加 proprioception) 不足以解决 contact-rich manipulation**。原因: visual observation 无法 capture:
- Contact force magnitude and direction
- Object slip onset
- Insertion 过程中的 micro-deformation

这 motivates multi-modal policy incorporating tactile sensing。相关工作 [ViTACFormer](https://arxiv.org/abs/2506.15953)、[Glovity](https://arxiv.org/abs/2510.09229) 都在探索 visuo-tactile fusion。但 paper 没做 tactile 实验，这是一个 limitation。

## 我的 critical reflections

### 1. Dataset scale 偏小

1.1K trajectories 对 11 个 task，平均每个 task 只有 100 条。对比 [Open-X Embodiment](https://arxiv.org/abs/2310.08864) 的 million-scale、[DexMimicGen](https://arxiv.org/abs/2410.24185) 的 auto-generation，DexJoCo data scale 明显偏小。可能所有 policy struggle 不是 policy capacity 不够，是 data insufficient。

### 2. Evaluation 不够 complete

Paper 没 report:
- Sample efficiency curve (data size vs success rate)
- Training compute cost
- Inference latency breakdown
- Tactile/force-based baseline 对比

这些 missing 让 "vision-only insufficient" 的结论稍 premature——没 compare with tactile-augmented policy，怎么知道 vision-only 是 bottleneck？

### 3. 与 DexMimicGen 的 positioning

[DexMimicGen](https://arxiv.org/abs/2410.24185) 用 few human demos + MimicGen auto-generation 可以 scale 到更多 data。DexJoCo 坚持 pure human demonstration 追求 naturalness 但牺牲 scale。这两个 paradigm 的 systematic comparison 缺失。未来可能需要 hybrid: human demo 提供 seed，auto-generation 提供 scale。

### 4. Allegro Hand 的选择

Allegro Hand 16-DoF，biomimetic 程度不如 [LEAP Hand](https://arxiv.org/abs/2309.08444) 或 [EyeSight Hand](https://arxiv.org/abs/2410.03138)。选 Allegro 可能因为 availability 和 MuJoCo Menagerie 的 mature model。但限制了 biomechanical plausibility 的 study。

## 与 broader landscape 的 connection

### Teleoperation hardware evolution

DexJoCo 的 Rokoko+Vive 组合代表 cost-efficiency frontier。但更 future-looking 的工作:
- [DexUMI](https://arxiv.org/abs/2505.21864): 用 human hand 作为 universal interface
- [DexCap](https://arxiv.org/abs/2403.07788): 用更 precise mocap
- [GR-Dexter](https://arxiv.org/abs/2512.24210): 集成 force feedback

### Foundation model for dexterous manipulation

最近 [EgoScale](https://arxiv.org/abs/2602.16710) 等工作在 scale egocentric human data pretrain dexterous representation。DexJoCo benchmark 为这类 model 提供 evaluation framework。期待 hand-centric foundation model 在 DexJoCo 上的 systematic evaluation。

### Bimanual coordination 的 open challenge

DexJoCo 5 个 bimanual task 上，所有 policy success rate 都很低 (Hanoi 24.7% best, Assembly 5.3% best)。对比 [Bi-DexHands](https://arxiv.org/abs/2211.01926) 的 RL 结果——RL 在 pass-the-ball 等 task 上能 achieve high success rate。这暗示 **imitation learning 在 bimanual coordination 上可能存在 fundamental challenge**: human bimanual coordination 的 variability 太高，单条 demonstration 难以 capture coordination pattern。

## 最核心的 takeaway

这篇 paper 的 contribution 主要是 **evaluation infrastructure** 而非 novel algorithm。它的 value 在于:
1. 系统性揭示了 current VLA 在 dexterous manipulation 上的 multiple failure modes
2. 提供 low-cost data collection 的 reference implementation
3. 为 community 提供 standardized benchmark 来 measure future progress

语言 generalization 的 negative result 尤其有价值——它 deconstruct 了 "VLA 具备 language understanding" 的 naive assumption，为 future work 指明了 compositional grounding 的 challenge。

## Useful references

- [DexJoCo Project Page](https://dexjoco.github.io)
- [GeoRT: Geometric Retargeting](https://arxiv.org/abs/2406.11468)
- [π0.5: Vision-Language-Action Model](https://arxiv.org/abs/2504.16054)
- [GR00T N1.5: NVIDIA Foundation Model](https://arxiv.org/abs/2503.14734)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [ACT: Action Chunking with Transformers](https://arxiv.org/abs/2304.13705)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [RoboSuite](https://arxiv.org/abs/2009.12293)
- [RoboCasa](https://arxiv.org/abs/2406.02524)
- [Bi-DexHands](https://arxiv.org/abs/2211.01926)
- [DexMimicGen](https://arxiv.org/abs/2410.24185)
- [DexCap](https://arxiv.org/abs/2403.07788)
- [FiLM: Visual Reasoning Conditioning](https://arxiv.org/abs/1709.07871)
- [SmolVLA](https://arxiv.org/abs/2506.01844)
- [Mobile ALOHA](https://arxiv.org/abs/2401.02126)
- [UMI: Universal Manipulation Interface](https://arxiv.org/abs/2406.11468)
- [LEAP Hand](https://arxiv.org/abs/2309.08444)
- [ViTACFormer: Visuo-Tactile Fusion](https://arxiv.org/abs/2506.15953)
- [SAPIEN/PartNet-Mobility](https://arxiv.org/abs/2003.08515)
- [Hunyuan3D](https://arxiv.org/abs/2411.02293)
- [OpenVLA Critique](https://arxiv.org/abs/2412.03163)

---

# DexJoCo: Task-Oriented Dexterous Manipulation Benchmark 深度解析

## 1. Motivation 与核心问题

这篇paper要解决的核心矛盾非常清晰: 当前robotics community的VLA (Vision-Language-Action) model生态几乎完全围绕parallel gripper构建, 从data collection pipeline [UMI](https://arxiv.org/abs/2406.11468), [Mobile ALOHA](https://arxiv.org/abs/2401.02126) 到 [Open-X Embodiment](https://arxiv.org/abs/2310.08864) dataset, 再到 [RT-2](https://arxiv.org/abs/2307.15818), [OpenVLA](https://arxiv.org/abs/2406.09246), [π0.5](https://arxiv.org/abs/2504.16054), [GR00T N1.5](https://arxiv.org/abs/2503.14734) 等foundation model, action space几乎都是7-DoF arm + 1-DoF gripper = 8维左右. 而dexterous hand (比如Allegro Hand 16-DoF) 加上双臂bimanual场景, action space维度会膨胀到40+维, 这对policy的expressivity、pretraining transfer、data efficiency都提出了完全不同的要求.

DexJoCo作者identification出的现有dexterous benchmark的4个核心缺陷:
1. **Hand-only setup**: 很多benchmark [UniDexGrasp](https://arxiv.org/abs/2304.00810) 省略了manipulator, 导致workspace不真实
2. **Task类型单一**: in-hand manipulation (如cube rotation) 缺乏functional diversity; pick-and-place又无法体现dexterous hand相比gripper的优势
3. **Data collection困难**: dexterous hand无法用motion planning生成合理trajectory, 现有方法过度依赖RL, 产生的轨迹不natural
4. **缺乏standardization**: 没有unified language instruction和数据格式给modern VLA使用

---

## 2. Benchmark Task Design 深度解析

### 2.1 Task formalization

每个task被formalize为 $\mathcal{T} = (\mathcal{O}, \mathcal{G})$:
- $\mathcal{O} = \{o_1, o_2, ..., o_m\}$: scene中m个interactive objects的集合
- $\mathcal{G} = \{g_{\text{seq}}, g_{\text{pose}}, g_{\text{joint}}, g_{\text{contact}}\}$: 4类functional success constraints

让我详细拆解这4个constraint types, 这是理解benchmark设计哲学的关键:

- **$g_{\text{seq}}$ (temporal/sequential constraints)**: 强制sub-goal执行的时序依赖. 比如Hanoi task中必须先移动最上面的disk
- **$g_{\text{pose}}$ (object pose conditions)**: 物体6D pose的目标条件. 比如Hammer Nail中nail需要被hammered到特定深度
- **$g_{\text{joint}}$ (articulated joint-state requirements)**: 铰接物体的joint state. 比如Water Plant中watering can的handle joint达到threshold才出水
- **$g_{\text{contact}}$ (collision/contact conditions)**: 接触条件. 比如Click Mouse中finger需要与button产生特定contact

一个task成功当且仅当所有相关constraints同时满足. 这种compositional design使得long-horizon task的evaluation非常严格.

### 2.2 11个Task的capability-oriented分类

| Category | Tasks | 关键challenge |
|----------|-------|---------------|
| **Tool-Use** | Hammer Nail, Click Mouse, Pick Bucket, Pinch Tongs, Fold Glasses, Water Plant | Fine-grained finger coordination, articulated object interaction |
| **Reasoning** | Unlock iPad (/B), Photograph (/B) | Language grounding, multi-step planning |
| **Bimanual Coordination** | Unlock iPad, Hanoi, Assembly, Microwave, Photograph (标记/B) | Asymmetric role allocation |
| **Long-Horizon** | Hanoi, Assembly | Temporal dependency, memory |

注意`/B`标记的5个task是bimanual, 这使得action space维度double: Franka Panda (7-DoF × 2) + Allegro Hand (16-DoF × 2) = 46维. 这是为什么paper后面提到π0.5和GR00T N1.5默认的32维action head不够用的根本原因.

### 2.3 Asset construction pipeline

值得注意的engineering细节:
- Base scene基于 [RoboSuite](https://arxiv.org/abs/2009.12293)
- Robot assets来自 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- Object assets来自 [RoboCasa](https://arxiv.org/abs/2406.02524) 和 [PartNet-Mobility/SAPIEN](https://arxiv.org/abs/2003.08515)
- 缺失物理参数的asset用 [Hunyuan3D](https://arxiv.org/abs/2411.02293) 生成后手动赋值

**Visual state change design** 是一个很clever的设计: 比如Water Plant中handle joint达到threshold就显示water, iPad Unlock中button被touch时highlight, Click Mouse中按下button激活display. 这种design让success condition同时是perceptually salient的, 便于人类和policy都判断progress.

---

## 3. Teleoperation System 与 Retargeting Algorithm

### 3.1 Hardware design

整套hardware成本约$2,300 USD, 极其cheap:
- **Rokoko Smartgloves**: 捕获finger pose, 避免camera-based方法的occlusion问题
- **2× HTC Vive Trackers + 2× Base Stations**: 跟踪wrist 6D pose控制Franka end-effector
- 3D-printed connector整合trackers和gloves

这个设计的关键trade-off: 相比 [DexCap](https://arxiv.org/abs/2403.07788) 使用更expensive的mocap系统, 或 [Bunny-VisionPro](https://arxiv.org/abs/2407.03162) 使用Apple Vision Pro, Rokoko+Vive的组合在cost和usability之间取得了非常好的平衡.

### 3.2 GeoRT retargeting loss function 详解

这是技术含量最高的部分. 由于human hand和Allegro Hand的kinematic structure完全不同, 直接linear mapping不可行. 作者采用 [GeoRT](https://arxiv.org/abs/2406.11468) 的self-supervised retargeting方法:

$$\mathcal{L} = \mathcal{L}_{\text{dir}} + \lambda_1 \mathcal{L}_{\text{cover}} + \lambda_2 \mathcal{L}_{\text{flat}} + \lambda_3 \mathcal{L}_{\text{pinch}} + \lambda_4 \mathcal{L}_{\text{col}}$$

让我逐一解释每个loss term的physical meaning:

**$\mathcal{L}_{\text{dir}}$ (direction preservation loss)**: 
Retargeting model $f$ 将human fingertip keypoints $x_H$ 映射到robot joint positions $q_R = f(x_H)$. $\mathcal{L}_{\text{dir}}$ 约束human fingertip的运动方向在robot fingertip空间中得到保持. 直觉上: 如果human index finger向上移动, robot index finger也应该向上移动, 而不是横向. 这保证了teleoperation的intuitiveness.

**$\mathcal{L}_{\text{cover}}$ (workspace coverage loss)**:
最大化robot fingertip reachable workspace对human fingertip workspace的覆盖. 直觉: 我们希望human能做出的所有hand pose都能被robot reproduce, 而不是某些human pose会map到robot joint limit.

**$\mathcal{L}_{\text{flat}}$ (sensitivity flatness loss)**:
保持mapping sensitivity的均匀性. 直觉: 如果human移动1cm, robot应该相应移动一个proportional的距离, 而不是在某些区域非常sensitive、某些区域几乎不响应. 这避免了teleoperation中的"dead zone"问题.

**$\mathcal{L}_{\text{pinch}}$ (pinch behavior preservation loss)**:
特殊保护pinch grasp这种critical manipulation primitive. Pinch是dexterous manipulation最基本的skill, 如果retargeting破坏了thumb-index pinch的对齐, 大量precision task都会失败.

**$\mathcal{L}_{\text{col}}$ (self-collision avoidance loss)**:
避免robot hand自碰撞. 高DoF hand很容易在retargeting过程中产生finger互相穿透的情况.

$\lambda_1, \lambda_2, \lambda_3, \lambda_4$ 是balance hyperparameters. 这种multi-objective optimization的设计理念是: 单一的forward kinematics loss无法捕捉retargeting的所有desiderata, 必须显式decompose成多个interpretable sub-objectives.

**关键insight**: "Only fingertip workspaces are recorded during data collection and used for training" —— 这意味着retargeting model是task-agnostic的, 不需要为每个task重新训练. 这是整个pipeline可scalable的关键.

### 3.3 Wrist tracking的relative pose设计

Wrist tracking有一个很elegant的设计: 初始wrist pose被记录为reference, 后续action都表示为relative pose change (delta action). Robot执行这些delta来reproduce motion. 这个设计的好处是:
- 消除了human-robot workspace offset
- 使得同一套系统可以用于不同base position的robot setup
- Delta action formulation与modern VLA的training paradigm更兼容

---

## 4. Domain Randomization Protocol

### 4.1 两个randomization level

Paper定义了两个evaluation regime, 这是理解Table 2结果的关键:

**rand-obj**: 仅randomize
- Object placement on table plane (x, y范围见Table 6)
- Table height: $\Delta h \sim U(0, 0.05)$ m

**rand-full**: 在rand-obj基础上增加
- 50个preset third-person camera poses (在spherical surface上dense sample后选择occlusion最小的50个)
- Lighting randomization: position perturb $U(-0.3, 0.3)$, direction perturb $U(-0.4, 0.4)$, diffuse RGB $U(0.3, 0.8)$, ambient RGB $U(0.3, 0.7)$
- Tabletop texture从texture library采样

### 4.2 Replay-based visual augmentation的trick

这是paper中一个很smart的engineering insight: **visual randomization不需要重新teleoperate**. 同一条trajectory可以在不同rendering settings下replay, 产生视觉上不同但action identical的training data. 这相当于免费的data augmentation, 成本几乎为零.

从RL角度类比: 这类似于RL中的domain randomization, 但因为我们的action来自human demonstration而不是RL policy, 所以visual appearance的变化完全不会影响action的quality. 这种"disentangle perception from action"的设计是imitation learning特有的优势.

### 4.3 Dynamics randomization (Table 3)

Table 6的dynamics randomization细节非常informative. 以Pinch Tongs为例:
- Tongs joint friction loss $\sim U(0, 0.05)$
- Joint stiffness multiplier $\sim U(0.75, 1.25)$
- Tongs mass multiplier $\sim U(0.75, 1.25)$

这种dynamics randomization是sim-to-real transfer的关键. 注意joint friction和stiffness同时被randomize, 这模拟了real-world中articulated object的manufacturing variance.

---

## 5. Policy Evaluation 深度分析

### 5.1 Action chunking formulation

$$\mathcal{P}(a_{t:t+k-1}) = \pi_\theta(a_{t:t+k-1} \mid s_{t-h+1:t}, l)$$

变量解释:
- $a_{t:t+k-1}$: 从time step $t$开始的$k$步action chunk
- $s_{t-h+1:t}$: $h$帧historical observation (window size $h$)
- $l$: optional language instruction
- $\pi_\theta$: 参数为$\theta$的policy
- $\mathcal{P}$: 条件概率分布

这个formulation统一了所有4个baseline: ACT (C-VAE), DP (diffusion), π0.5 (flow-matching), GR00T N1.5 (flow-matching). 区别仅在conditional distribution的参数化方式.

### 5.2 Asynchronous inference机制

借鉴 [SmolVLA](https://arxiv.org/abs/2506.01844) 的设计: 下一个action chunk在当前chunk执行时生成, 消除idle waiting. Overlapping chunks通过temporal ensembling平滑.

这个design有profound的implication for evaluation: **inference frequency直接影响performance**. 轻量级policy (如DP-T ~100M) 推理快, 能用更recent的observation, reactivity更好. 重量级policy (如π0.5) 推理慢, 可能用stale observation. 这解释了为什么DP-T在某些task上能competitive π0.5: 不是因为DP-T representation更强, 而是因为它的inference latency更小.

### 5.3 Action head adaptation for bimanual tasks

π0.5和GR00T N1.5的default action head是32维, 但bimanual task需要46维. 作者采用**partial pretrain-AH** strategy: 保留pretrained的前32维权重, 额外14维random initialize.

Table 3的rand-AH实验对比了完全random reinitialization, 发现partial pretrain-AH在most task上更好. 这个结果有点反直觉: 我们可能expect random init让action head可以从scratch学习bimanual-specific representation, 但实验显示pretrained的低维action representation对bimanual也有transfer value.

**我的interpretation**: 这暗示了manipulation的low-dimensional structure存在某种universal property (比如end-effector pose控制), 即使bimanual场景, 单臂的action prior依然useful. 这与 [EigenPose](https://arxiv.org/abs/2406.11468) 等工作中"action can be decomposed into task-agnostic和task-specific components"的观察一致.

---

## 6. 实验结果深度解读

### 6.1 Table 2的核心findings

| Model | rand-obj Avg | rand-full Avg | Δ (drop) |
|-------|--------------|---------------|----------|
| DP-T | 50.4% | 20.0% | -30.4% |
| DP-C | 47.6% | 28.4% | -19.2% |
| ACT | 35.5% | 22.7% | -12.8% |
| π0.5 | 52.5% | 34.1% | -18.4% |
| GR00T N1.5 | 40.2% | 30.5% | -9.7% |

几个critical observations:

1. **Visual randomization导致sharp success rate drop**: 几乎所有policy在rand-full下都大幅下降, 说明现有VLA的visual robustness严重不足. 这与 [OpenVLA](https://arxiv.org/abs/2406.09246) 的观察一致 —— pretrain on web data并不能直接transfer到diverse visual conditions.

2. **π0.5在单臂task上dominant, 但bimanual task上优势消失**: Hammer Nail (84.7% vs DP-T 81.3%), Water Plant (88.7% vs 84.0%), Click Mouse (64.7% vs 62.0%). 但在Hanoi/Assembly这种bimanual long-horizon task上, π0.5几乎与DP-T持平. 这正是action head random init的代价: 额外14维从scratch学习, 抵消了large-scale pretraining的优势.

3. **DP-C在Unlock iPad和Pinch Tongs上unexpectedly最强**: Unlock iPad DP-C 52.0% vs π0.5 12.0%; Pinch Tongs DP-C 57.3% vs π0.5 24.0%. 作者hypothesize这是因为DP-C是唯一使用 [FiLM](https://arxiv.org/abs/1709.07871) injection的policy, 而FiLM相比self/cross attention在fine-grained visual perception上更effective.

**我的alternative hypothesis**: FiLM的优势在于它的conditioning是multiplicative的, $y = \gamma \odot x + \beta$, 这种element-wise modulation对precise spatial localization (button位置, tongs hinge) 比attention的soft aggregation更suitable. Attention tends to smooth over fine details, 而FiLM preserves spatial resolution.

### 6.2 Failure mode analysis (Figure 5, 6)

Figure 5的failure mode breakdown极其informative:

**Button-pressing tasks (Unlock iPad, Click Mouse, Photograph)**:
Policy能perceive object (能pick up tablet, 能push mouse onto mousepad), 但fail to click intended button. 这说明perception能识别object的存在, 但无法ground到object的affordance — 即"哪里是可交互的".

**Insertion tasks (Assembly, Hanoi)**:
Insertion步骤失败率极高. 这是典型的contact-rich manipulation, vision-only policy无法感知insertion过程中的force feedback, 导致无法判断peg是否对准socket.

**Pinch Tongs**:
Policy能grasp tongs但fail to squeeze and release. 作者认为这是"insufficient temporal memory" —— policy需要记住"已经grasp了"这个state, 才能execute后续的squeeze动作. 这暗示了现有action chunking的history window $h$ 可能不够长.

**Microwave**:
Policy能place hot dog into microwave, 但随后withdraw hand alongside the hot dog. 这是一个极其interesting的failure mode —— policy学到了"place"的action primitive, 但没有学到"release"的temporal structure, 导致place和retrieve纠缠在一起.

### 6.3 Multi-task training degradation (Table 3)

Multi-task training结果揭示了一个negative transfer现象:
- DP-T在所有task上都degrade (avg 50.4% → 33.2%)
- π0.5在Click Mouse和Pinch Tongs上有提升, 但avg下降 (52.5% → 45.5%)

这与general expectation相悖: foundation model应该在multi-task上更robust. 我的interpretation: **dexterous task之间存在action space conflict**. 比如Hammer Nail需要forceful swing, 而Fold Glasses需要gentle precision. 这些task的action distribution差异巨大, shared backbone难以同时accommodate. 这是dexterous manipulation特有的challenge —— gripper-based task的action distribution相对homogeneous (都是approach-grasp-lift), 但dexterous task的action manifold是高度multi-modal的.

### 6.4 VLA language generalization failure (Figure 7, Appendix A)

这是paper最sobering的实验之一. 训练π0.5在Unlock iPad task上用single-digit password 1-5, 测试:
- Seen digits: "1", "2", "4"
- Arithmetic: "1+1", "2+2"
- English words: "two", "one plus one"

结果 (average precision %):
- "1": 15.3%
- "2": 30.7%
- "4": 4.0%
- "1+1": 24.7%
- "2+2": 1.3%
- "two": 30.0%
- "one plus one": 20.7%

表面看"two"和"1+1"的30%+ precision似乎不错, 但Figure 7的heatmap揭示了真相: **model有一个固定的output bias倾向于输出"2"**, 无论instruction是什么. 当correct answer恰好是"2"时, 这个bias碰巧命中; 当correct answer是"4"时, precision降到4%.

Quantitative analysis:
- Chi-square test rejects independence ($p = 2.15 \times 10^{-4}$), 说明model确实对language有response
- 但Normalized Mutual Information只有0.018, 几乎可忽略
- Average JS divergence across instruction pairs: 0.026 (max 0.057)

**核心结论**: VLA model的"language understanding"在compositional generalization上完全失效. Model学到的是instruction marginal distribution上的mode collapse, 而非真正的language-conditioned action generation. 这与最近 [OpenVLA critique](https://arxiv.org/abs/2412.03163) 等工作的观察一致: 当前VLA的language grounding是fragile的.

---

## 7. Discussion部分的深层insights

### 7.1 Lack of dexterous hand-centric foundation models

Paper指出: 当前VLA model几乎都pretrain在gripper-based data上, 导致action space mismatch. 这不仅仅是dimension mismatch, 更深层的问题是**joint coupling**. Gripper的1-DoF是trivial的 (open/close), 但dexterous hand的16个joint之间存在strong biomechanical coupling (比如index finger的MCP/PIP/DIP joint的tendon coupling). 现有action head将每个joint独立建模, 无法capture这种coupling structure.

这motivates了**embodiment-aware representation**: 未来的foundation model需要pretrain时考虑hand kinematic structure, 比如用graph neural network编码joint connectivity, 或用hand-specific tokenizer.

### 7.2 Vision-only policy的fundamental limitation

这是paper最provocative的claim: **vision-only (即使加proprioception) 不足以解决contact-rich manipulation**. 原因是visual observation无法capture:
- Contact force magnitude and direction
- Object slip onset
- Micro-deformation during insertion

这motivates multi-modal policy incorporating tactile sensing. 相关工作 [ViTACFormer](https://arxiv.org/abs/2506.15953), [Glovity](https://arxiv.org/abs/2510.09229) 都在探索visuo-tactile fusion. 但paper没有experiment with tactile, 这是一个limitation.

### 7.3 Sim-to-real transfer的open problem

Paper明确承认这是未解决的问题. Domain randomization是一个patch, 但根本性的fidelity gap依然存在:
- Object properties (friction, mass distribution)的systematic mismatch
- Rendering的photorealism gap
- Sensor signal (tactile, proprioception)的noise model差异

这需要beyond domain randomization的systematic sim-real alignment, 比如differentiable simulation, real-to-sim distillation, 或sim-real co-training.

---

## 8. 我的critical reflections

### 8.1 Dataset scale的concern

1.1K trajectories对于11个task, 平均每个task只有100条. 对比 [Open-X Embodiment](https://arxiv.org/abs/2310.08864) 的million-scale, [DexMimicGen](https://arxiv.org/abs/2410.24185) 的auto-generation, DexJoCo的data scale明显偏小. 这可能解释了为什么所有policy都struggle在difficult task上 —— 可能不是policy capacity的问题, 而是data insufficient.

### 8.2 Evaluation的completeness

Paper没有report:
- Sample efficiency curve (data size vs success rate)
- Training compute cost
- Inference latency breakdown
- Tactile/force-based baseline comparison

这些missing evaluation使得"vision-only insufficient"的结论稍显premature —— 如果没有compare with tactile-augmented policy, 怎么知道vision-only是bottleneck?

### 8.3 与 [DexMimicGen](https://arxiv.org/abs/2410.24185) 的positioning

DexMimicGen使用few human demos + MimicGen auto-generation, 可以scale到更多data. DexJoCo坚持pure human demonstration, 追求naturalness但牺牲了scale. 这两个paradigm的systematic comparison缺失. 未来工作可能需要hybrid approach: human demo提供seed, auto-generation提供scale.

### 8.4 Allegro Hand的选择

Allegro Hand是16-DoF, 但biomimetic程度不如 [LEAP Hand](https://arxiv.org/abs/2309.08444) 或 [EyeSight Hand](https://arxiv.org/abs/2410.03138). 选择Allegro可能是因为availability和MuJoCo Menagerie的mature model. 但这限制了biomechanical plausibility的study.

---

## 9. 与broader research landscape的connection

### 9.1 Teleoperation hardware的evolution

DexJoCo的Rokoko+Vive组合代表了cost-efficiency frontier. 但方向上, 更future-looking的工作在探索:
- [DexUMI](https://arxiv.org/abs/2505.21864): 使用human hand作为universal interface
- [DexCap](https://arxiv.org/abs/2403.07788): 使用更precise的mocap
- [GR-Dexter](https://arxiv.org/abs/2512.24210): 集成force feedback

### 9.2 Foundation model for dexterous manipulation

最近 [EgoScale](https://arxiv.org/abs/2602.16710) 等工作在scale egocentric human data来pretrain dexterous representation. DexJoCo benchmark为这类model提供了evaluation framework. 未来期待看到hand-centric foundation model在DexJoCo上的systematic evaluation.

### 9.3 Bimanual coordination的open challenge

DexJoCo的5个bimanual task上, 所有policy的success rate都很低 (Hanoi 24.7% best, Assembly 5.3% best). 这与 [Bi-DexHands](https://arxiv.org/abs/2211.01926) 的RL结果对比鲜明 —— RL在Bi-DexHands的pass-the-ball等task上能achieve high success rate. 这暗示了 imitation learning在bimanual coordination上可能存在fundamental challenge: human bimanual coordination的variability太高, 单条demonstration难以capture coordination pattern.

---

## 10. Useful references

- [DexJoCo Project Page](https://dexjoco.github.io)
- [GeoRT: Geometric Retargeting](https://arxiv.org/abs/2406.11468)
- [π0.5: Vision-Language-Action Model](https://arxiv.org/abs/2504.16054)
- [GR00T N1.5: NVIDIA Foundation Model](https://arxiv.org/abs/2503.14734)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [ACT: Action Chunking with Transformers](https://arxiv.org/abs/2304.13705)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [RoboSuite](https://arxiv.org/abs/2009.12293)
- [RoboCasa](https://arxiv.org/abs/2406.02524)
- [Bi-DexHands](https://arxiv.org/abs/2211.01926)
- [DexMimicGen](https://arxiv.org/abs/2410.24185)
- [DexCap](https://arxiv.org/abs/2403.07788)
- [FiLM: Visual Reasoning Conditioning](https://arxiv.org/abs/1709.07871)
- [SmolVLA](https://arxiv.org/abs/2506.01844)
- [Mobile ALOHA](https://arxiv.org/abs/2401.02126)
- [UMI: Universal Manipulation Interface](https://arxiv.org/abs/2406.11468)
- [LEAP Hand](https://arxiv.org/abs/2309.08444)
- [ViTACFormer: Visuo-Tactile Fusion](https://arxiv.org/abs/2506.15953)
- [SAPIEN/PartNet-Mobility](https://arxiv.org/abs/2003.08515)
- [Hunyuan3D](https://arxiv.org/abs/2411.02293)

---

这篇paper的contribution主要在于**evaluation infrastructure**而非novel algorithm. 它value在于: (1) 系统性揭示了current VLA在dexterous manipulation上的multiple failure modes; (2) 提供了low-cost data collection的reference implementation; (3) 为community提供了standardized benchmark来measure future progress. 语言generalization的negative result尤其有价值 —— 它deconstructs了"VLA具备language understanding"的naive assumption, 为future work指明了compositional grounding的challenge.
