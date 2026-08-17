---
source_pdf: ManipArena Comprehensive Real-world Evaluation.pdf
paper_sha256: 7f4b17fb5eac0b15f0a388b48f997951f7c97ade9ab26d86b09f5bcf8e9f979e
processed_at: '2026-08-05T16:21:02-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲一遍 ManipArena

好,我换个讲法。前面那篇太"论文导读"了,这次我用我自己脑子里怎么理解这篇 paper 的方式讲。

---

## 这帮人到底想干嘛

你想想现在的 robot learning 圈子有个尴尬局面:

- 大家都在 simulator 里刷榜,RLBench、LIBERO、CALVIN 分数越刷越高
- 但真把 model 搬到 real robot 上,**完全不是一回事**
- 为什么?sim 里没有阳光从窗户照进来导致的 shadow drift,没有 joint backlash,没有 gripper 打滑,没有 50ms 的 inference latency
- 然后你想做 real-world benchmark,结果发现每个 lab 的 setup 不一样——我家 lab 灯是暖色的,你家是冷色的,他家背景有张海报——**同一个 model 在三个 lab 跑分差 30 分,你都不知道是 model 不行还是灯不行**

所以 ManipArena 想解决的就一件事:**让 real-world evaluation 变成可控实验,而不是玄学排行榜**。

参考:https://maniparena.x2robot.com/

---

## 他们怎么做的:五个 trick

### Trick 1:Green-screen booth

整个 evaluation 在一个绿幕棚里跑。墙是绿的,顶是绿的,灯是固定 LED panel + softbox。

为什么这是神来之笔?在开放 lab 里 model 失败了,你不知道是因为:
- object 变了
- 还是 layout 变了
- 还是阳光从窗户斜进来了
- 还是隔壁 PhD 把他的 coffee mug 放在我桌子边上了

这四个 confound 在 real world 里**永远同时变**。

绿幕棚直接把背景和光照锁死。剩下的变量就只有你设计进去的——换了什么 object、换了什么位置。**失败归因**突然变得可能了。

这招其实很妙。CV 圈子里 chroma key 用了几十年了,但 robot evaluation 里没人这么干。它本质上就是把 NVIDIA Omniverse Replicator 那种 "在 sim 里 randomize 一切" 的哲学反过来用——**在 real 里 isolate 一切**,然后只把你想测的变量暴露出来。

### Trick 2:Diversity 不是自然收集来的,是设计出来的

大部分 robot dataset 的多样性是 "我随便录了几百条 demo,自然就有多样性了"。ManipArena 说不行,每个 task 配一个 **diversity guide**,规定:

- Level 1:object 的 material / color / size 必须覆盖哪些组合(比如 `arrange_cup` 要训 ceramic mug 和 paper cup,各两种颜色)
- Level 2:object 的 position / orientation 必须覆盖哪些 spatial config(`insert_wireline` 要 socket L/R/C × wire U/M/D = 9 种)
- Level 3(只 semantic task 有):任务本身的组合排列要变(`press_button_in_order` 的颜色序列要从 4 选 3 的 24 种排列里采)

而且每个 dimension 的分布要保持近似 uniform(±10-15%),防止 model 学到 frequency bias。

**为什么这个重要?** 因为如果没有这个,你测 OOD 的时候根本不知道是 model 真的不行,还是你训练分布太窄,所谓的 OOD 其实只是"没见过的某个角落的 in-distribution"。

diversity guide 的本质是:**训练分布足够宽,OOD 才有意义**。

### Trick 3:T1-T10 分层,一次跑完一条 generalization 曲线

每个 task 10 个 trial,但 trial 顺序是设计过的:

| Trial | 内容 | 测什么 |
|---|---|---|
| T1-T4 | 训练分布内 object + 位置变 | In-domain competence |
| T5-T8 | object appearance 变(shape/material),仍在训练语义内 | Visual shift |
| T9-T10 | 训练里完全没见过的 object | Semantic OOD |

举例 `put_spoon_to_bowl`:
- T1-T4:不锈钢勺(训练见过的)
- T5-T8:儿童勺(形状不同,但是训练里有的)
- T9-T10:黑塑料勺(训练里完全没出现)

一次 evaluation 跑完,你直接得到一条 degradation curve:
$$\text{Score}(T_1\text{-}T_4) \to \text{Score}(T_5\text{-}T_8) \to \text{Score}(T_9\text{-}T_{10})$$

这条曲线的斜率就是 generalization 能力的量化。OneModel 在 `sort_headphone` 上从 10.0 掉到 0.5(−95%),DreamZero 从 6.7 掉到 4.5(−33%)——**一次实验,你就知道 VLA 和 WAM 的 OOD robustness 完全不是一个量级**。

而且他们还玩了 compound OOD:有些 task 在 T9-T10 同时改两个 factor。`sort_headphone` 的 T9-T10 同时换 neckband 类型 + 换白色,直接测"两个 OOD factor 叠加是 graceful degrade 还是 catastrophic collapse"。

### Trick 4:Sub-task partial credit,不要 binary success

每个 task 拆成有序 sub-task,每个 trial 给 0-10 分,完成几个 sub-task 给几分。

`pour_water` 拆 5 步:
1. Grasp bottle (2 pts)
2. Lift bottle (2 pts)
3. Move to pouring position (2 pts)
4. Tilt & pour (3 pts)
5. Return upright (1 pt)

抓不起来的 model 0 分;抓起来抬起来但倒不进水的 model 3 分;倒进去但没还原的 model 9 分。

为什么这是大事?binary success rate 把三种 qualitatively 不同的失败模式都打成 "失败":
- 完全不能开始(cable 都没抓起来)
- 完成一半(sub-task 3 之后崩了)
- 99% 完成就差最后 retract

partial credit 让你能说"DreamZero 在 `insert_wireline` 上 SR=0%,但稳定完成第一个 sub-step(cable pickup),只是后续失败"——这种诊断信号在 binary 下完全看不见。

### Trick 5:Server-side inference + One-model-for-all

参赛者不碰机器人。你 expose 一个 HTTP endpoint,接受 obs(相机图 + proprioception),返回 action。Organizers 的服务器调你的 endpoint 控制机器人。

而且强制 **one model for all tasks**——一个 endpoint 服务所有 20 个 task,不能 per-task 切。

这个设计的精髓在于:**逼着大家训 generalist,而不是 specialist**。你没法训 15 个 π0.5 各过拟合一个 task。这是在测 manipulation foundation model 是否真的存在,而不是测工程打包能力。

附带好处:参与者不需要硬件(降低门槛)、所有人在 identical hardware 上跑(公平)、weight 不上交(护 IP)。

---

## 实验告诉我们什么(这部分最有意思)

他们测了三个 baseline:

- **π0.5-Single**:每个 task 单独 fine-tune 一个 π0.5(VLA),15 个 specialist
- **π0.5-OneModel**:一个 π0.5 联合训 15 个 task(VLA,generalist)
- **DreamZero**:World Action Model,先 dream 出未来 video frame,再从 frame 里 extract action

总分:
| Model | Total /1500 |
|---|---|
| π0.5-Single | 626.3 |
| **π0.5-OneModel** | **640.5** |
| DreamZero | 500.3 |

但总分不重要,有意思的是**拆开看**。

### 发现 1:Multi-task 训练有得有失

OneModel 在 semantic task 上赢 Single 一大截:
- `sort_headphone`:73 vs 35(**+109%**)
- `pair_up_items`:51.5 vs 21(**+145%**)

直觉:多 task 训练让 model 见过更多 object,共享 visual representation,semantic classification 变强。

但 OneModel 在 instruction-conditioned sequential task 上崩盘:
- `press_button_in_order`:13 vs 48(**−73%**)
- `put_items_into_drawer`:7 vs 34(**−79%**)

这两个 task 的共同点:必须**先看场景里的 instruction**(打印的颜色卡 / 抽屉分类映射),再按 sequence 执行。multi-task 训练把这种 task-specific 的 procedural memory 给洗掉了。

OneModel 在 `press_button` 上 8 个 trial 都是 1 分——它**完全丢失了这个 task 的 policy**,只能 default 到 minimal behavior。

直觉上这是个 capacity 分配问题。同一个 transformer 容量,semantic recognition 的 gradient 和 procedural memory 的 gradient 是冲突方向。这跟 LLM 里 catastrophic forgetting 文献完全呼应,但在 manipulation 上第一次被这么干净地测出来。

### 发现 2:VLA 和 World Model 是互补的,不是谁替代谁

这个发现可能是整篇 paper 最有科学价值的。

| 能力维度 | OneModel | DreamZero |
|---|---|---|
| Coarse pick-place (`pick_items`) | 37 | **97.8** |
| Precision bimanual (`put_glasses`) | **87** | 37 |
| Semantic understanding (avg) | **58.9** | 32.7 |
| Spatial robustness (basket shift) | −44% | **−8%** |
| Compound OOD degradation | −95% | **−33%** |

解读:

**DreamZero 强 coarse 弱 precision**——视频预测对"物体大幅运动"有 visual signal,对"末端毫米级调整"几乎无 signal。`insert_wireline` 它稳定打 3 分(只完成 cable pickup,后续全失败)。`sort_headphone` 它 6 个 trial 都打 9/10,**稳定 missing retraction**——因为 retraction 几乎不改变画面,视频预测目标学不到这个动作。

**DreamZero 强 OOD/spatial**——这点特别深刻。它学的是物理 affordance(形状、可抓性),不是 appearance-action mapping。所以 neckband 白耳机对它只是"另一个可抓物体",对 OneModel 是"训练分布外"导致 catastrophic shift。basket 从右边挪到左边,对 DreamZero 几乎无影响(−8%),因为视频预测天然 spatially equivariant,对 OneModel 则是 −44%,因为 action mapping 是 absolute-coordinate-sensitive。

**OneModel 强 precision 弱 coarse**——直接 action supervision 提供了 visual signal 微弱但 motor 精度高的 supervision。retraction 这种动作也能学。但失去了 WAM 那种"物理动力学中间表示",对 coarse planning 反而不如 DreamZero。

**结论**:VLA 和 WAM 学的是**互补能力**。下一个前沿的 architecture 应该是 hybrid——用 WAM 做 coarse planning + spatial reasoning,用 VLA 做 fine-grained motor execution。这跟 NVIDIA GR00T + Cosmos、Google SayCan + RT-2 的思路都 echo 了。

参考:
- GR00T: https://developer.nvidia.com/groot
- Cosmos: https://www.nvidia.com/en-us/ai/cosmos/
- SayCan: https://say-can.github.io/
- RT-2: https://robotics-transformer2.github.io/

### 发现 3:VLA 的 OOD "泛化" 可能是 spatial overfitting 的副产物

这个 case 特别 sharp。`put_spoon_to_bowl` 的 T9 和 T10 用**同一个** OOD object(黑塑料勺),只是位置不同:
- OneModel:T9=0,T10=4
- π0.5-Single:T9=10,T10=0

两个 VLA 学到了**不同的 spatial bias**。OneModel 在 T9 那个位置崩了,Single 在 T10 那个位置崩了。它们的 "OOD 泛化" 本质上是**某 spatial config 的 accidental alignment**,不是真正的 object understanding。

这意味着:**当前 VLA 在 OOD 上的表现,可能有一大半是 spatial overfitting 在 disguise**。要测真 OOD,必须把 spatial config 也 hold 出去,否则你测的是 spatial memorization,不是 object generalization。

### 发现 4:Force-sensitive fine manipulation 是 open frontier

4 个 task 全部 model 都低于 30 分:
- `pour_water`:19
- `insert_wireline`:24
- `arrange_cup`:25
- `put_stationery`:26

这些 task 都要求 force feedback:液体重量变化、insertion contact 突变、笔袋拉链阻力、杯子堆叠压力控制。

**当前 VLA 和 WAM 都只 condition on vision + end-effector pose,缺 force 这个 channel**。

而 ManipArena 开放了 motor current 和 joint velocity——motor current 是 torque proxy:
$$\tau_m = K_t \cdot I_m$$

其中 $K_t$ 是 motor torque constant(N·m/A),$I_m$ 是 motor current(A),$\tau_m$ 是输出 torque(N·m)。

如果 policy 能 condition on $\tau_m$,就能做 closed-loop impedance control:
$$\tau_{\text{cmd}} = K_p(q_d - q) + K_d(\dot{q}_d - \dot{q}) + \tau_{\text{ff}}(\hat{F}_{\text{ext}})$$

变量含义:
- $q, q_d$ — 实际 / 目标 joint angle
- $\dot{q}, \dot{q}_d$ — 实际 / 目标 joint velocity
- $K_p, K_d$ — 位置 / 速度增益
- $\tau_{\text{ff}}$ — 基于估计外部力 $\hat{F}_{\text{ext}}$ 的前馈 torque
- $\tau_{\text{cmd}}$ — 最终 commanded torque

paper 把 motor current 开放出来,等于在 force-aware policy 这个方向上**铺好路等别人来挖**。这跟 MIT MiniCheetah、Stanford PUPPY 的 proprioceptive policy 思路一致。

参考:
- MiniCheetah: https://www.csail.mit.edu/publications/mini-cheetah

### 发现 5:Latency 是 hidden variable

DreamZero 在 A800 上 7-8 s/step,π0.5 是 110 ms/step——慢 50-70×。

这个差距在 coarse pick-place 上可能不致命,但在 force control 上是**致命的**。4 秒延迟下你根本来不及做 closed-loop 反应,等于开环。

paper 没深挖这个,但 latency × capability 才是真正的 deployment capability。一个 7s/step 的 model 哪怕分数高,也部署不了。

---

## 还有什么有意思的

### Real2Sim 那一套
他们用 3D Gaussian Splatting 重建 booth(桌子、shelf、fixture 的 geometry + appearance),用 Hunyuan3D 生成 task object 的 mesh + texture,在 IsaacLab 里跑。然后**把 real 录的 joint trajectory 在 sim 里 replay**,得到 paired real-sim observation sequence。

这等于做了 **trajectory-level Real2Sim alignment**——同一 trajectory 在两个 domain 里跑,model 在 sim 成功 real 失败,可以直接对照 observation 找差异源。

参考:
- 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Hunyuan3D: https://3d.hunyuan.tencent.com/
- IsaacLab: https://github.com/isaac-sim/IsaacLab

### Long-horizon 对 VLA 架构的挑战
Mobile task 平均 2878 frames @ 20fps,远超当前 VLA 的 context window(ViT-22B 一般 8k token,折合 ~400 frame)。这暗示 **hierarchical policy 是必须的**——高层做 navigation + sub-goal generation,低层做 local manipulation,中间用 language/sub-goal 作为"压缩 token"。

这跟 SayCan / LLM-as-planner 的层级结构呼应,但 ManipArena 第一次给了严格的 long-horizon real-world 测试场。

### Green-screen 的更激进用法
paper 提到 chroma-key 可以在 post-processing 合成任意背景——kitchen、工业场景、adversarial texture。这其实可以单独开一篇 paper:**在绿幕棚里训的 policy,在合成 kitchen 背景下测,直接量化 background robustness**。这是 domain randomization 的 mirror image——一个 randomize sim texture,一个 real 背景在 pixel level randomize。

### 跟 Open-X-Embodiment 的对称关系
Open-X-Embodiment [https://robotics-transformer-x.github.io/] 是把 **data 跨 embodiment** 标准化,ManipArena 是把 **evaluation 跨 embodiment** 标准化。一个在 input 端,一个在 output 端,刚好对称。两者结合才有可能形成真正的 manipulation foundation model 生态。

### 跟 LeRobot 的关系
数据格式用 LeRobot v2.1 [https://github.com/huggingface/lerobot],这跟 Hugging Face 把 robot data "Huggingfacification" 的策略高度一致。**统一数据格式 = 统一 evaluation 协议 = 可比较的 leaderboard**。ManipArena 的 server-side HTTP 推理协议其实是 LeRobot 数据格式在 evaluation 端的对应物。

---

## 一句话总结

如果让我用一句话讲 ManipArena 在干嘛:

**Real-world robot benchmark 之所以没意义,是因为环境噪声混淆了 model 能力;ManipArena 用绿幕棚 + 设计过的 diversity + 分层 OOD trial + sub-task partial credit + server-side one-model-for-all,把 evaluation 从 leaderboard 变成 controlled experiment,从而第一次让 VLA 和 world model 的能力边界能被清晰归因——结果发现它们是互补的,不是谁替代谁,而 force-sensitive fine manipulation 是下一个真正开放的前沿。**

用最朴素的直觉说:**这帮人把 robot eval 当作物理学实验来做,不是当作 Kaggle 竞赛来做**。这是 manipulation 评估方法学上的一次升级,可能比任何一个具体的 model 结果都更有长期价值。

---

参考链接汇总:
- ManipArena 项目页:https://maniparena.x2robot.com/
- ManipArena 代码:https://github.com/maniparena/maniparena-repo
- π0.5 paper:https://arxiv.org/abs/2504.16054
- DreamZero paper:https://arxiv.org/abs/2602.15922
- Flow Matching (Lipman et al.):https://arxiv.org/abs/2210.02747
- LeRobot:https://github.com/huggingface/lerobot
- Open-X-Embodiment:https://robotics-transformer-x.github.io/
- RT-2:https://robotics-transformer2.github.io/
- SayCan:https://say-can.github.io/
- NVIDIA GR00T:https://developer.nvidia.com/groot
- NVIDIA Cosmos:https://www.nvidia.com/en-us/ai/cosmos/
- 3D Gaussian Splatting:https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Hunyuan3D:https://3d.hunyuan.tencent.com/
- IsaacLab:https://github.com/isaac-sim/IsaacLab
- RLBench:https://github.com/stephenjames21/RLBench
- LIBERO:https://libero-project.github.io/
- CALVIN:https://calvinrobot.github.io/
- ManiSkill2:https://github.com/haosulab/ManiSkill2
- VLABench:https://github.com/VLABench/VLABench
- RoboArena:https://arxiv.org/abs/2506.18123
- RoboChallenge:https://arxiv.org/abs/2510.17950
- ManipulationNet:https://arxiv.org/abs/2603.04363
- BEHAVIOR-1K:https://behavior.stanford.edu/
- MiniCheetah:https://www.csail.mit.edu/publications/mini-cheetah

---

# ManipArena: 一篇试图把 Real-world Manipulation Evaluation 从"黑箱排行榜"变成"受控实验台"的 Paper

这篇 paper 在 2026 年 3 月放出来,作者来自 SYSU、X SQUARE ROBOT、MBZUAI、Tsinghua。核心问题其实非常简单一句话:**当前 VLA / world model 的 evaluation 大量在 simulator 里做,但 simulator 测出来的 capability 跟 real-world deployment 之间隔着一道 "reality gap",而且 real-world evaluation 又因为机器人平台不统一、lighting 不固定、背景混乱而无法横向比较**。ManipArena 给的答案是:把 real-world evaluation 本身做成一个 "controlled experiment",通过 green-screen + 固定照明 + 设计过的 diversity guide + 分层 OOD trial,让 "model 失败" 这件事可以归因到具体的 generalization axis 上,而不是一锅粥的环境噪声。

下面我按 build intuition 的顺序拆解,从 motivation → design → experiment → insight。

---

## 1. 为什么需要这个 benchmark:Reality Gap 的具体来源

Simulator-centric benchmark(RLBench [6]、LIBERO [8]、CALVIN [9]、ManiSkill2 [4]、VLABench [12])的吸引力来自 controllability + reproducibility,但它们屏蔽掉了四类物理因素:

- **Perception noise**:真实相机有 vignetting、rolling shutter、自动白平衡漂移、specular highlight 在 metal/refractive surface 上的剧烈变化;
- **Complex contact dynamics**: articulated object(抽屉铰链、笔袋拉链)、deformable object(衣服)、liquid(pouring water)的 contact 在 sim 里几乎不可能高保真建模,RLBench 这些 benchmark 干脆把这些 task 排除掉;
- **Hardware constraints**:关节 backlash、gripper compliance、joint torque limit、base wheel slip;
- **System latency**:从 inference 端到 actuator 的端到端延迟在 sim 里是 0,在 real 上是几十到几百 ms,对 high-frequency contact 任务(pouring、insertion)影响极大。

而 real-world benchmark(RoboArena [1]、RoboChallenge [10]、ManipulationNet [3])则存在另一个问题:每个 lab 自己的 setup 都不一样,lighting 受天气、季节、窗户位置影响,背景里有不可控的 clutter,导致同一个 model 在不同 lab 的 "成绩" 无法直接比较。

Table 1 把这个矛盾清楚地摆出来:
| Benchmark | Env | Reasoning | Generalization | Mobile | Sensory | Real-to-Sim |
|---|---|---|---|---|---|---|
| RLBench / LIBERO | Sim | Low | Limited–Mod | ✗ | ✗ | ✓ |
| ManiSkill2 | Sim | Low | Strong | ✓ | ✗ | ✓ |
| VLABench | Sim | High | Strong | ✗ | ✗ | ✓ |
| RoboArena | Real | Medium | Weak | ✓ | ✗ | ✗ |
| ManipulationNet | Real | Medium | Weak | ✗ | ✗ | ✓ |
| **ManipArena** | **Real** | **High** | **Systematic** | **✓** | **✓** | **✓** |

注意 ManipArena 是唯一一个把 Reasoning=High、Generalization=Systematic、Sensory=✓(提供 motor current / joint velocity)和 Mobile=✓ 同时打钩的。

参考:
- https://github.com/maniparena/maniparena-repo
- https://maniparena.x2robot.com/

---

## 2. 五条设计原则的内在逻辑

paper 里给的五条原则不是平行的,而是一个层次结构:

1. **Reasoning-Oriented**:task 设计上,**没有 trivial task**,每个 task 都要求 reasoning,只是 reasoning 的瓶颈位置不同。这与 pick-and-place 类 baseline 区分开。
2. **Multi-Level Generalization**:用 diversity guide 主动注入多样性,而不是被动收集自然数据。
3. **Mobile Manipulation**:拓展到 tabletop 之外,要求 navigation + spatial memory + 长时间 whole-body control。
4. **Rich Sensory Diagnostics**:开放 motor current 和 joint velocity,鼓励 force-aware policy。
5. **Real2Sim Synchronization**:用 3DGS 重建场景,把 sim 变成 real 的"几何+视觉"孪生,从而可以做 sim-to-real gap 分析。

前两条决定 "测什么",第三条决定 "测得多大",第四条决定 "能诊断多深",第五条决定 "能不能 scale"。这五条放在一起,本质上是把 benchmark 从 leaderboard 转化为 controlled experiment。

---

## 3. Competition Format:Server-Side Inference + One-Model-For-All

这个设计可能是整篇 paper 在工程上最聪明的一步。

**机制**:参与者不需要带任何硬件,只需要暴露一个 HTTP endpoint:
- Request payload:camera images (3 views) + proprioception
- Response:action vector
- Organizers 的 infrastructure 负责机器人控制、数据采集、scoring。

**One-model-for-all-tasks 规则**:每个参与者只能提交一个 model endpoint,这个 endpoint 在所有 20 个 task 上都被 evaluate,不能 per-task 切换。这是刻意的——它把 optimization pressure 从 "训 15 个 specialist 各自过拟合一个 task" 改成 "训 1 个 generalist 处理 pouring、sorting、inserting、button-press 全部"。

直觉上,这是在直接测试 "manipulation foundation model" 是否真的存在,而不是测试 "task-specific engineering"。

附带优势:
- **Low barrier**:参与者不需要硬件;
- **Reproducibility**:所有 trial 在 identical hardware 上跑;
- **Fair**:没有 latency/hardware 优化的红利;
- **IP protection**:weights 和 code 永远不上交。

Scoring:每个 task 10 个 trial,每个 trial 0–10 分(partial credit),单 task 满分 100,15 个 tabletop task 满分 1500。

---

## 4. Task 设计:三类 reasoning 瓶颈

20 个 task 分三类:

### 4.1 Execution Reasoning(10 task)
瓶颈在于 **怎么执行**:grasp strategy、force control、bimanual coordination。视觉目标明确,难的是 motor execution。

代表 task:
- `arrange_cup_inverted_triangle`:3 个杯子摆倒三角,涉及 multi-object spatial planning;
- `put_ring_onto_rod`:sub-cm 插入精度;
- `pour_water_from_bottle`:force control + 液体动力学;
- `insert_wireline`:bimanual hand-off,mm-level 接触对准;
- `put_items_into_drawer`:10 个 sub-task 的长链;
- `put_blocks_to_color`:7 个 diversity dimension,configuration space 最大。

### 4.2 Semantic Reasoning(5 task)
瓶颈在于 **做什么**:必须先解决 semantic ambiguity 才能行动。

代表 task:
- `sort_headphone`:在 clutter 里找到耳机并分类;
- `classify_items_as_shape`:按 cube/sphere/cylinder 分类,ordering 跨 trial 随机;
- `press_button_in_order`:从打印的颜色卡片读出 sequence,再按顺序按键;
- `pair_up_items`:配对 glove/sock;
- `pick_fruits_into_basket`:fruit vs. non-fruit 区分(bread / lettuce 是 distractor)。

注意 `press_button_in_order` 是一个特别精巧的 task——它要求 model 既要 **视觉读取 instruction** 又要 **按 sequence 执行**,procedural memory 和 semantic recognition 同时被考验。后面 baseline 实验里 OneModel 在这个 task 上崩溃 (13/100),正是这个 task 的诊断价值体现。

### 4.3 Mobile Manipulation(5 task)
瓶颈在于 **长时间、大空间**:navigation、spatial memory、2-3 分钟的 sustained whole-body control。

代表 task:
- `put_clothes_in_hamper`:14 个 sub-task,avg 193.6 s,3 次往返;
- `hang_up_picture`:bimanual pick painting + navigate + hang;
- `organize_shoes`:左右脚配对;
- `put_bottle_on_woodshelf`;
- `take_and_set_tableware`:avg ~211 s,摆盘 + 刀叉左右分置。

关键统计:
| Category | Tasks | Traj | Avg Frames | Avg Duration |
|---|---|---|---|---|
| Execution | 10 | 5,157 | 784 | 39.2 s |
| Semantic | 5 | 2,783 | 499 | 25.0 s |
| Mobile | 5 | 2,872 | 2,878 | 143.9 s |
| Total | 20 | 10,812 | — | — |

Mobile task 占 26.7% 的 trajectory 但 60.6% 的 frame,这个 imbalance 对固定 context window 的 VLA 架构是结构性挑战。

---

## 5. Green-Screen Enclosure:把 confound 变成 controlled variable

这可能是 paper 里我最喜欢的一个工程细节。

**问题描述**:在 open lab 里,model 在一次 trial 上失败了,你不知道是:
(a) object 变了;
(b) layout 变了;
(c) 光照变了;
(d) 背景里多了一团 clutter。

这些因素在 open environment 里同时变,**你无法 attribute**。

**解决方案**:整个 evaluation 在一个 green-screen booth 里进行,uniform chroma-key 墙 + 顶,固定 LED panel + softbox 照明。

三个 property:
1. **Variable isolation**:背景 uniform,performance 差异只能来自设计过的 object/spatial 变化;
2. **Controlled illumination**:constant color temperature + intensity,屏蔽掉窗户、季节、天气、overhead light 的干扰;
3. **Reproducibility & portability**:booth 是 self-contained 的,可以搬到别处,不同 site 的 booth 视觉条件一致,从而**支持去中心化 evaluation**。

**Future extensibility**(这点很妙):green-screen 的 chroma-key 可以在 post-processing 或 real-time projection 里被合成成任意自然背景——kitchen、工业场景、adversarial texture,从而把 "background robustness" 作为一个独立的实验变量。

直觉上,green-screen 是把 CV 里很久以前就有的 chroma-key trick 用在 robot evaluation 上,效果是把 "环境 confound" 这个一直在暗中搞乱 real-robot benchmark 的变量变成可控的。这一点其实跟 NVIDIA 的 Omniverse replicator 思路是 mirror 的,只不过一个在 sim 里 randomize,一个在 real 里 isolate。

---

## 6. Systematic Diversity Design:三层 hierarchy

paper 强调,real-world benchmark 的多样性不能 "incidentally arise from natural collection",必须当作 designed experimental variable。每个 task 配一个 diversity guide,让每个 diversity dimension 的分布近似 uniform(10-15% 容差),防止 model 利用 frequency bias。

三层 hierarchy:

### Level 1 — Physical Attribute Diversity(外观层)
- material × color × size
- 例:`arrange_cup` = ceramic mug vs. paper cup × red vs. white = 4 种;
- `pour_water` = glass / paper / ceramic × blue/red × handle L/R = 12 种;
- 目标:**perceptual generalization**(model 能处理 different-looking object 吗?)。

### Level 2 — Spatial Configuration Diversity(布局层)
- position + orientation randomization
- 例:`insert_wireline` = socket (L/R/C) × wire (U/M/D) = 9 种;
- `pour_water` = handle (L/R) × cup pos (6 zones) = 12 种;
- `put_blocks_to_color` = 7 个 dimension;
- 目标:**spatial generalization**(能处理新 layout 吗?)。

### Level 3 — Semantic Composition Diversity(任务层,仅 semantic reasoning task)
- object combination、ordering、category assignment 都跨 trial 变
- 例:`press_button_in_order` = 4 色 (pink/yellow/blue/green) 中选 3 排序 = 24 种;
- `pick_fruits_into_basket` = {apple, pear, grape, banana, bread, lettuce} 子集;
- 目标:**semantic generalization**(能处理新 task configuration 吗?)。

直觉上,这三层分别对应 perception、spatial reasoning、semantic reasoning 三种不同的 failure mode,paper 在 evaluation 里又用 stratified T1-T10 来 probe 每一层的 generalization profile,等于在 measurement 端做了 attribute decomposition。

---

## 7. Stratified OOD Evaluation:T1–T10 不是随机

每个 task 10 trial,但 trial 顺序按 OOD 难度分层:
- **T1–T4 (In-domain)**:训练分布内的 object + 位置变化;
- **T5–T8 (Visual shift)**:appearance 变化(material/shape),但仍在训练语义内;
- **T9–T10 (Semantic OOD)**:从未见过的 object。

举例 `put_spoon_to_bowl`:
| Trial | Spoon Type | Color | OOD Level |
|---|---|---|---|
| T1–T4 | Stainless steel | White | In-domain |
| T5–T8 | Children's spoon | White | Visual shift |
| T9–T10 | Plastic | Black | Semantic OOD |

这种分层让一次 evaluation 就能生成一条 **generalization degradation curve**,不需要做额外实验。

### Semantic distance 设计
OOD 不是二元的。paper 还故意把 OOD object 按 semantic distance 分档:
- **Near-OOD**:`goggles` for `put_glasses`——形状相似,grasp 策略可复用;
- **Medium-OOD**:`sunglasses`——形状相似但折叠机制不同;
- **Far-OOD**:`neckband sport headphone (white)`——form factor + color 同时变 = compound shift。

直觉上,这是在测试 "OOD 难度到底是 binary in/out,还是由 semantic distance 决定?"。后面的实验结果显示 DreamZero 在 far-OOD 上 degrade -33%,OneModel degrade -95%,说明 semantic distance 假设是成立的,而且两个 paradigm 对 distance 的 sensitivity 截然不同。

### Compound OOD
有些 task 在 OOD trial 上同时改变多个 factor,从而测试 generalization 是否 graceful degrade。例:`sort_headphone` 的 T7-T8 只改 type(bluetooth,仍黑),T9-T10 同时改 type(neckband)和 color(白)。

### Configuration-level generalization
对剩下 8 个没有真正 OOD object 的 task,T9-T10 改 spatial/compositional configuration。例:`pick_items_into_basket` 改 basket 位置(R→C→L),`press_button_in_order` 改颜色 sequence。

---

## 8. Robot Platform:X2Robot Bimanual + Quanta X1 Mobile

**Single embodiment** 是关键设计:所有 20 个 task 用同一套硬件,从而 performance 差异反映 policy 能力,而不是 hardware 差异。

### Tabletop
- 4 个 robot unit,每个:
  - Bimanual 6-DOF follower arm;
  - Master-follower teleoperation interface;
  - 3 个相机:face(全局)+ 左/右 wrist(特写);
  - 固定 LED panel + softbox 照明;
- 每个 unit 收 ~25% 的训练数据;
- Evaluation 也分布在这 4 个 unit 上,保证 train/test platform 一致。

### Mobile
- Quanta X1:mecanum-wheel omnidirectional base + 可调 lifting column + bimanual ARX arm + 头部相机 + 双腕相机;
- 工作区 ~3m × 3m。

这种 4+1 设计让 task 之间的 embodiment 完全统一,这对 "multi-task 训练是否有效" 这种科学问题非常关键——embodiment mismatch 通常是 hidden confound。

---

## 9. Sensor Data:56D / 62D,关键是 motor current

Tabletop 每帧 56D:
| Modality | Dim | Description |
|---|---|---|
| End-effector pos | 3×2=6 | 左右臂 XYZ |
| End-effector rot | 3×2=6 | Euler 角 |
| Gripper state | 1×2=2 | 开/合 |
| Joint position | 7×2=14 | 关节角 |
| Joint velocity | 7×2=14 | 关节角速度 |
| Joint current | 7×2=14 | 电机电流(末维 = gripper current)|
| Total | 56 | |

Mobile 额外 6D:
| Modality | Dim |
|---|---|
| Head rotation (pan/tilt) | 2 |
| Lifting column height | 1 |
| Base velocity (v_x, v_y, ω_θ) | 3 |

**Motor current 是 torque proxy**,paper 明确说是 "deliberate design choice"——它是 force-aware policy 的关键信号:
- `pour_water`:通过电流变化感知 liquid weight 增加;
- `insert_wireline`:通过电流突变检测 insertion contact;
- `arrange_cup`:通过电流过载避免 excessive force。

直觉上,这相当于给 policy 开了"本体感觉"通道。当前 VLA baseline 没用这些信号,只用了 14D 的 end-effector pose,paper 把它开放出来是在 force-aware manipulation 这个 frontier 上"挖坑"——后面会看到,force-sensitive fine manipulation 正是 baseline 集体塌陷的地方。

数据格式:LeRobot v2.1,parquet 存 proprioception,mp4 存 video,info.json / episodes.jsonl / tasks.jsonl 描述 metadata。

---

## 10. Real2Sim:3DGS + Hunyuan3D + IsaacLab

为了让 offline evaluation scalable + sim-to-real 可诊断,他们做了 digital twin:

1. **场景重建**:用 3D Gaussian Splatting (3DGS) [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/] 重建 booth、table、shelf、fixture 的 geometry + appearance;
2. **物体生成**:task object 用 Hunyuan3D [https://3d.hunyuan.tencent.com/] 生成高质量 mesh + texture,带物理属性(mass、friction、collision mesh);
3. **Simulation**:基于 IsaacLab [https://github.com/isaac-sim/IsaacLab] + IsaacLab-Arena,GPU 并行 env;
4. **Trajectory alignment**:把真实录的 joint trajectory 在 sim 里 replay,得到 paired real-sim observation sequence。

直觉上,这种 "trajectory-level Real2Sim" 的关键价值在于 **sim-to-real gap 的 isolation**:同一 trajectory 在两个 domain 里跑,model 在 sim 里成功 / real 里失败,可以直接对照 observation 找差异源,而不是凭空猜。

这与 NVIDIA 的 GR00T 项目、Google 的 RT-X Real2Sim、物理 AB 项目的思路都有重合,但 ManipArena 把它跟 stratified OOD 一起做,等于在 sim-to-real 上叠加了 generalization 维度。

---

## 11. Sub-Task Partial Credit Scoring:把 binary success 变成结构化诊断

每个 task 拆成有序 sub-task,partial credit。例:`pour_water_from_bottle`(5 sub-task):
1. Grasp bottle → 2 pts
2. Lift bottle → 2 pts
3. Move to pouring position → 2 pts
4. Tilt & pour → 3 pts
5. Return upright → 1 pt

完全抓不起来的 model = 0 分;能抓能抬但倒不进水的 model = 3/10;partial credit 直接告诉你 **failure 发生在哪个 sub-stage**。

`put_items_into_drawer` 有 10 个 sub-task(open drawer 1 → place stationery → close → drawer 2 → ... → retract),partial credit 揭示是 drawer manipulation 阶段失败还是 object grasping 阶段失败。

直觉上,这是把 RL 里 sparse reward 的痛点用 task decomposition 换成 dense reward 信号——但不改变 task,只改变 measurement。这种诊断在 binary success rate 下完全看不见:例如 DreamZero 在 `insert_wireline` 上 SR=0%,但平均每 trial 2.4 分,说明它稳定完成第一个 sub-step(cable pickup),只是后续失败。

---

## 12. Baseline 深度分析:π0.5 vs DreamZero

paper 选了 3 个 baseline,代表两个 paradigm:

### 12.1 π0.5-Single:task-specific VLA
π0.5 [5] 是 Physical Intelligence 的 SOTA VLA,backbone 是 pretrained VLM + flow-matching action head。Single 配置下,每个 task 单独 fine-tune 一个 π0.5,共 15 个 specialist。

**Flow matching 公式** [Lipman et al., 2023]:

给定起点分布 p_0(噪声)和目标分布 p_1(action),构造时间 t∈[0,1] 上的 conditional flow:
$$x_t = (1-t)\,x_0 + t\,x_1$$

其中:
- $x_0 \sim p_0$ — 噪声 sample(通常 Gaussian)
- $x_1 \sim p_1$ — target action sample
- $t \in [0,1]$ — flow time
- $x_t$ — 在时间 t 的中间状态

对应的 conditional vector field:
$$u_t(x \mid x_0, x_1) = x_1 - x_0$$

Flow matching 训练目标是最小化参数化 vector field $v_\theta(x_t, t, c)$ 与目标 vector field 的差异:
$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, x_0, x_1, c}\left[\big\| v_\theta(x_t, t, c) - (x_1 - x_0) \big\|^2\right]$$

变量:
- $v_\theta$ — 参数化 vector field(神经网络,backbone 是 VLM)
- $c$ — conditioning(视觉 obs + 语言 instruction + proprioception)
- $\theta$ — 网络参数

推理时从 $x_0 \sim \mathcal{N}(0, I)$ 出发,用 ODE $\dot{x}_t = v_\theta(x_t, t, c)$ 从 $t=0$ 积分到 $t=1$ 得到 action。

为什么用 flow matching 而不是 diffusion:flow matching 的 ODE 路径是直线,采样步数少,对 robot control 的低 latency 友好。π0.5 在 A800 上 ~110 ms/step,~9 Hz。

### 12.2 π0.5-OneModel:unified multi-task VLA
一个 π0.5 在全部 15 个 tabletop task 上联合训练。这是 ManipArena "one-model-for-all" 范式的 baseline 版本。

### 12.3 DreamZero:World Action Model(WAM)
DreamZero [11] 不直接预测 action,而是 **先 dream 出未来 video frame,再从生成的 frame 里 extract action**。

视频生成用 autoregressive video diffusion。给定 conditioning c(当前 obs + language),生成未来 frame sequence $\hat{x}_{1:K}$。视频 diffusion 的训练目标:
$$\mathcal{L}_{\text{video}}(\phi) = \mathbb{E}_{t, \epsilon, x_{0:K}, c}\left[\big\| \epsilon - \epsilon_\phi(x_t, t, c) \big\|^2\right]$$

其中:
- $x_{0:K}$ — 真实未来 K 帧
- $\epsilon$ — Gaussian noise
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ — forward 加噪
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ — cumulative noise schedule
- $\epsilon_\phi$ — 参数化 denoising network

生成完未来 frame 后,action 通过一个 inverse dynamics module 从相邻帧推出。这意味着 **WAM 的 objective 是 visual prediction,不是 action supervision**,action 只是副产品。

Latency 代价巨大:A800 单卡 7-8 s/step,双卡 4-5 s/step——比 VLA 慢 50-70×,这对 closed-loop control 几乎不可行,但它揭示了 WAM 的本质:**它学的是 dynamics,不是 policy**。

---

## 13. 实验:几个 high-level 数字

Table 4 完整结果:

| Model | Total /1500 | Best task | Worst task |
|---|---|---|---|
| π0.5-Single | 626.3 | put_glasses 80 | pour_water 8 |
| π0.5-OneModel | **640.5** | pick_fruits 94 | press_button 13 |
| DreamZero | 500.3 | pick_items 97.8 | press_button 3 |

### 几个关键观察:

**(1) 远未饱和**。最高 42.7%,中位数 per-task best 仅 51.5。4 个 task 全模型都低于 30(pour_water 19,insert_wireline 24,arrange_cup 25,put_stationery 26)——这些是 force-sensitive fine manipulation,paper 把它定为 open frontier。

**(2) 没有单一 model 统治**。OneModel 赢 7/15,Single 赢 3/15,DreamZero 赢 4/15(1 tie)。pairwise correlation:
- π0.5 vs DreamZero: r = 0.34
- π0.5 vs OneModel: r = 0.74
- OneModel vs DreamZero: r = 0.43

低相关说明 **performance 在一个 task 上无法预测另一个 task**,各 model 在不同 task type 上各有优势。这其实是"互补能力"的强证据。

**(3) Multi-task 训练的 trade-off**:
- **Gain(semantic transfer)**:OneModel 在 sort_headphone 上 +109%(73 vs 35),pair_up_items +145%(51.5 vs 21),pick_fruits +4(94 vs 90)。multi-task 让 model 看到更广的 object,共享 visual representation。
- **Cost(procedural forgetting)**:OneModel 在 press_button 上 -73%(13 vs 48),put_items_into_drawer -79%(7 vs 34)。这两个 task 共同特征是 **必须从场景里读取 instruction** 再执行 sequence,multi-task 训练稀释了 task-specific 的 instruction→action mapping。

直觉上,这是一个 capacity 分配问题:同一个 transformer 容量,要么分配给 generic visual recognition(被 semantic task 正则化),要么分配给 task-specific procedural memory(被 instruction-conditioned sequence 任务需要)。这两个 task family 的 gradient 方向是冲突的。这个发现实际上呼应了 LLM 里 catastrophic forgetting 与 multi-task finetuning 的 trade-off 文献,只不过在 manipulation domain 第一次这么清晰地被测量出来。

**(4) VLA vs WAM 的能力互补**:

| 维度 | OneModel 优势 | DreamZero 优势 |
|---|---|---|
| Coarse manipulation | pick_items 37 | **pick_items 97.8** |
| Precision manipulation | **put_glasses 87** | put_glasses 37 |
| Sequential reasoning | press_button 13 | press_button 3(都低,但 Single=48)|
| Semantic understanding | **avg 58.9** | avg 32.7 |
| Spatial robustness | basket shift -44% | **basket shift -8%** |
| Compound OOD | sort_headphone -95% | **sort_headphone -33%** |

直觉解读:
- **DreamZero 强 coarse / 弱 precision**:视频预测对"物体大幅运动"信号大,对"末端毫米级调整"几乎无 visual signal,retraction 这种"画面不变"的 sub-step 直接丢分;
- **DreamZero 强 OOD robust**:因为它学的是物理 affordance(形状、可抓性),而非 appearance-action mapping,所以 neckband 白耳机对它只是"另一个可抓物体",对 OneModel 则是"训练分布外"导致 catastrophic shift;
- **DreamZero 强 spatial invariance**:视频预测天然要求 spatial equivariance——"抓东西放篮子里"的 future frame 与绝对位置无关,而 OneModel 的 action mapping 是 absolute-coordinate-sensitive;
- **OneModel 强 semantic**:语言 supervision + cross-task visual co-training,让 semantic classification 的 representation 被共享;
- **OneModel 强 precision**:直接 action supervision 提供了 visual 信号微弱但 motor 精度高的 supervision,retraction 这种动作也能学。

**(5) Brittle OOD:spatial configuration 比 object identity 更决定成败**。

`put_spoon_to_bowl` 的 T9 和 T10 用 **同一个** OOD object(黑塑料勺)只在不同 spatial configuration 上:
- OneModel:T9=0,T10=4
- π0.5-Single:T9=10,T10=0

两个 VLA 学到了 **不同的 spatial bias**,它们的 "OOD 泛化" 其实是某 spatial config 的 accidental alignment,不是真正的 object understanding。这是个挺深刻的发现:**当前 VLA 的 OOD 表现本质上可能是 spatial overfitting 的副产物**。

---

## 14. Capability Boundary 五维分析

paper 把 sub-task 类型归到 5 个核心 dimension,每个 model 有自己的 "Strong" dimension,无 model 全胜:

```
                Coarse  Precision  Seq-Reason  Semantic  Spatial-Robust
π0.5-Single     ✗       ✓          ✓           ◐         ✗
π0.5-OneModel   ◐       ✓          ✗           ✓         ✗
DreamZero       ✓       ✗          ✗           ◐         ✓
```

这个矩阵给出了未来 manipulation foundation model 的明确 architectural 目标:**同时拥有 VLA 的 precision action supervision 和 WAM 的 physical reasoning / spatial equivariance**。这个 insight 其实指向一种 "hybrid architecture":用 WAM 做 coarse planning + spatial reasoning,用 VLA 做 fine-grained motor execution。这跟 Google 的 SayCan + RT-2、NVIDIA 的 GR00T + Cosmos WAM、Tesla Optimus 的世界模型 + control head 思路都 echo 了。

参考:
- SayCan: https://say-can.github.io/
- RT-2: https://robotics-transformer2.github.io/
- GR00T: https://developer.nvidia.com/groot
- Cosmos: https://www.nvidia.com/en-us/ai/cosmos/

---

## 15. Diagnostic Value of Partial Credit

paper 4.5 节给了几个特别有说服力的 case:

- DreamZero `insert_wireline`:SR=0%,avg 2.4/trial → **稳定完成第一步 cable pickup,后续失败**;
- DreamZero `sort_headphone`:SR=60%,6 个 trial 都是 9/10 → **稳定 missing retraction**;
- OneModel `press_button`:SR=0%,avg 1.3/trial → **几乎完全丢失 task policy**,只能 default 到 minimal behavior;
- π0.5-Single `press_button`:SR=20%,avg 4.8/trial → **还在 active attempt sequence**,偶尔成功。

二元的 success rate 把这 4 种 case 都打成 "失败",partial credit 揭示的是 "完全不能开始" / "完成部分" / "间歇成功" 三种 qualitative 不同的 failure mode。这等于把 manipulation benchmark 从 RL 里的 sparse binary signal 升级成 dense diagnostic signal。

---

## 16. 一些延伸的直觉和联想

### 16.1 与 Open-X-Embodiment / RT-X 的关系
Open-X-Embodiment [https://robotics-transformer-x.github.io/] 是把 data 跨 embodiment 标准化的努力,ManipArena 是把 **evaluation 跨 embodiment 标准化**——一个在 input 端,一个在 output 端,刚好对称。

### 16.2 与 BEHAVIOR-1K、Habitat 的关系
BEHAVIOR-1K [https://behavior.stanford.edu/] 用 atomic action decomposition 把 task 拆成可测量的 sub-goal,ManipArena 的 sub-task partial credit 是同思路在 manipulation 上的版本。

### 16.3 Green-screen 的更激进用法
paper 提到 chroma-key 可以在 post-processing 合成任意背景。这其实可以做一个非常有意思的实验:**在 green-screen booth 里训的 policy,在合成 kitchen / industrial / adversarial 背景下测,直接量化 background robustness**。这跟 domain randomization 是同一哲学,只不过一个 randomize sim texture,一个 real 背景在 pixel level randomize。这是个可以单独开一篇 paper 的方向。

### 16.4 Motor Current 与 Force-aware Policy 的 frontier
4 个 force-sensitive task 全 model 都低于 30 分,这强烈暗示 **直接 action supervision 缺了 force 这个 channel**。Motor current 是 torque proxy,有公式
$$\tau_m = K_t \cdot I_m$$
其中 $K_t$ 是 motor torque constant(N·m/A),$I_m$ 是 motor current(A),$\tau_m$ 是输出 torque(N·m)。

policy 如果能 condition on $\tau_m$,就能做 closed-loop force control:
$$\tau_{\text{cmd}} = K_p (q_d - q) + K_d (\dot{q}_d - \dot{q}) + \tau_{\text{ff}}(\hat{F}_{\text{ext}})$$

其中 $\tau_{\text{ff}}$ 是基于估计外部力 $\hat{F}_{\text{ext}}$ 的前馈,可以用 current-based impedance control。这跟 MIT MiniCheetah、Stanford PUPPY 的 proprioceptive policy 思路一致 [https://www.csail.mit.edu/publications/mini-cheetah]。

### 16.5 Latency 作为 hidden variable
DreamZero 4-7 s/step,π0.5 110ms/step。在 contact-rich 任务上,这个 latency 差距对 force control 是致命的——4 秒延迟下,你根本来不及做 closed-loop 反应,等于开环。paper 没深挖这个,但 latency 与 capability 的乘积才是真正的 deployment capability。这跟 NVIDIA 的 GR00T N1 强调 inference throughput 的思路一致 [https://developer.nvidia.com/groot]。

### 16.6 Stratified OOD 与 "generalization 的科学"
ManipArena 的 T1-T10 分层其实是一个 measurement 协议创新。把 model performance 表述成
$$\text{Score}_{\text{task}}(M) = \sum_{i=1}^{10} s_i(M, \text{trial}_i) / 10$$
其中 $s_i \in [0, 10]$,并按 layer 分组,可以定义 generalization degradation:
$$\Delta_{\text{OOD}}(M) = \frac{\mathbb{E}[s_{T9-T10}] - \mathbb{E}[s_{T1-T4}]}{\mathbb{E}[s_{T1-T4}]}$$

paper 测出 OneModel $\Delta_{\text{OOD}} \approx -95\%$,DreamZero $\approx -33\%$。这种 normalized degradation 系数第一次让"泛化能力"成为可比较的标量。

### 16.7 Long-horizon mobile manipulation 与 context window
Mobile task avg 2878 frames @ 20fps,远超当前 VLA 的 context window(VIT-22B 一般 8k token,折合 ~400 frame)。这其实暗示了 **hierarchical policy 的必要性**:高层做 navigation + sub-goal generation,低层做 local manipulation,中间用 language/sub-goal 作为 "压缩 token"。这跟 SayCan / LLM-as-planner 的层级结构呼应,但 ManipArena 第一次给了严格的 long-horizon real-world 测试场。

### 16.8 与 LeRobot 的标准化策略
ManipArena 用 LeRobot v2.1 format 发布 [https://github.com/huggingface/lerobot],这跟 Hugging Face 把 robot data "Huggingfacification" 的策略高度一致。**统一数据格式 = 统一 evaluation 协议 = 可比较的 leaderboard**。ManipArena 的 server-side HTTP 推理协议其实是 LeRobot 数据格式在 evaluation 端的对应物。

---

## 17. 总结:这篇 paper 在科学上的真实贡献

去掉工程包装,这篇 paper 在科学上做了三件值得认真对待的事:

1. **把 generalization 从"binary in/out"变成"distance-graded, multi-axis measurement"**。T1-T10 stratified design + 三层 diversity hierarchy + compound OOD 设计,首次让 real-world generalization 可以被 attribute 到具体的 axis(perceptual / spatial / semantic)和具体的 distance(visual shift / near-OOD / far-OOD / compound)。

2. **第一次在严格 controlled 环境下直接对比 VLA 与 WAM**,揭示两者的能力是 **complementary 而非 dominance** 关系:DreamZero 强 coarse + spatial + OOD,弱 precision + sequential;OneModel 强 semantic + precision,弱 spatial + procedural。这指向 hybrid architecture 是下一个前沿。

3. **Identify force-sensitive fine manipulation 为 open frontier**(4 个 task 全部低于 30 分),并 open-source 提供 motor current + joint velocity 信号,等于把 force-aware policy 这个方向直接铺好路。

如果用一句话压缩这篇 paper 的 message: **real-world robot benchmark 之所以没意义,是因为环境噪声混淆了 model 能力;ManipArena 用 green-screen + designed diversity + stratified OOD + sub-task partial credit,把 evaluation 变成可控实验,从而第一次让 VLA 与 world model 的能力边界可以被清晰归因。**

---

## 关键链接

- Project: https://maniparena.x2robot.com/
- Code: https://github.com/maniparena/maniparena-repo
- π0.5 paper: https://arxiv.org/abs/2504.16054
- DreamZero paper: https://arxiv.org/abs/2602.15922
- 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Hunyuan3D: https://3d.hunyuan.tencent.com/
- IsaacLab: https://github.com/isaac-sim/IsaacLab
- LeRobot: https://github.com/huggingface/lerobot
- Flow Matching (Lipman et al.): https://arxiv.org/abs/2210.02747
- Open-X-Embodiment: https://robotics-transformer-x.github.io/
- RT-2: https://robotics-transformer2.github.io/
- SayCan: https://say-can.github.io/
- NVIDIA GR00T: https://developer.nvidia.com/groot
- Cosmos World Foundation Model: https://www.nvidia.com/en-us/ai/cosmos/
- BEHAVIOR-1K: https://behavior.stanford.edu/
- RLBench: https://github.com/stephenjames21/RLBench
- LIBERO: https://libero-project.github.io/
- CALVIN: https://calvinrobot.github.io/
- ManiSkill2: https://github.com/haosulab/ManiSkill2
- VLABench: https://github.com/VLABench/VLABench
- RoboArena: https://arxiv.org/abs/2506.18123
- RoboChallenge: https://arxiv.org/abs/2510.17950
- ManipulationNet: https://arxiv.org/abs/2603.04363
