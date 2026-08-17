---
source_pdf: SG-VLA Learning Spatially-Grounded Vision-Language-Action Models for Mobile
  Manipulation.pdf
paper_sha256: 11e33c81de2dad0b8d46e36304f8a73d5dd29a841a05d1324462bb8b8bb4996a
processed_at: '2026-08-12T05:28:02-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，咱用大白话把这篇 SG-VLA 捋一遍，像在 whiteboard 前聊那种感觉。

---

## 一句话讲完

之前的 VLA 在桌面上玩魔方、抓杯子很猛，一搬到家里就废了。这篇就说：**光让模型预测 action 不够，得逼它顺便把 "我现在在哪、抓稳没、target 在哪、关节啥角度、target 长啥样" 全都答一遍**，这样它学出来的中间 representation 才有 spatial sense，下游 action 才靠谱。

就这么个 idea。简单粗暴。

---

## 痛点到底是啥

你想啊，tabletop setting 长啥样：
- camera 固定，scene 就那么大
- robot 不挪，base 不动
- 7-DoF arm + 1-DoF gripper，8 个数搞定
- object 就在眼前，arm 够一下就到

换成家用 mobile manipulation，全塌了：
- robot 满屋子跑，camera 视角一直在变
- 13-D action：base 3 + torso 1 + arm 7 + gripper 2，navigation 和 manipulation 缠一起
- 你要开冰箱：先得看见冰箱在哪（可能 3 米外），走过去，伸手找把手，拉，松。每一步 robot 在 world frame 的 position 都在变

更恶心的：**standard VLA 只有一个 supervision signal**——那个 13-D action vector。模型完全可以走捷径，学个 mapping from pixels to action，根本不需要显式表征 "我在哪、target 在哪"。结果就是 OpenVLA 7B 在这套场景下 success rate **4%**，基本瞎做。

SG-VLA 的 thesis 一句话：**sparse supervision 养出 lazy representation，lazy representation 干不了精细活。得用 auxiliary task 当拐杖，逼 backbone 把 spatial info 显式 encode 进 latent 里。**

---

## Architecture 长啥样（不复杂，简单过一下）

```
[head RGB+depth] + [hand RGB+depth] + [language]
         ↓
   DINOv2 (spatial) + SigLIP (semantic) 双 encoder
         ↓
    projector 映到 language space
         ↓
   Qwen2.5-0.5B（故意选小的，0.5B 够了）
         ↓
   latent representation —— 分三路出：
       ├── discrete action token (主任务)
       ├── 5 个 auxiliary decoder（辅助监督）
       └── flow matching action head（可选，continuous action）
```

总参数 1.3B。比 OpenVLA 7B 小 5 倍多，结果反而 better。原因后面讲。

---

## 5 个 auxiliary decoder 都在干啥

挂在 LLM latent 上的 5 个小 head，每个重建一个 intermediate signal：

| Decoder | 重建啥 | 为啥有用 |
|---|---|---|
| Global Pose | robot (x,y) | 让 latent 知道 "我在哪"，nav 任务关键 |
| Grasp Success | 0/1 抓稳没 | pick 任务的核心 subgoal |
| Object Pose | target 的 3D pos + quaternion | 让 latent 编码 "target 在哪" |
| Joint Pose (qpos) | 12 个 joint 角度 | 本体感觉，manipulation 直接 relevant |
| Mask | 128×128 target binary mask | 让 latent 编码 "target 长啥样、在 image 哪" |

为啥这个 list 是这 5 个？我猜作者的 intuition 是：**mobile manipulation 失败就失败在 model 不知道 "self state"、"target state"、"scene" 这三类信息**。这 5 个 aux task 正好覆盖这三类。

每个 decoder 的架构选择也都有道理：
- **MLP** 给低维 target（pos、pose、grasp）：target 维度小，简单 MLP 就够，省 compute
- **Transformer** 给 12-D qpos：joint 之间有 kinematic dependency，self-attention 让 joint token 互相 attend，capture 关节联动
- **CNN** 给 mask：dense spatial output，transpose-conv 经典做法

---

## 最关键的 loss 是 Eq.3，咱细看

Object pose loss：

$$\mathcal{L}_{obj} = \|\hat{\mathbf{t}} - \mathbf{t}\|_2^2 + (1 - |\hat{\mathbf{q}} \cdot \mathbf{q}|)$$

- $\hat{\mathbf{t}}, \mathbf{t} \in \mathbb{R}^3$：predicted / GT object 3D 位置，L2 squared
- $\hat{\mathbf{q}}, \mathbf{q} \in \mathbb{R}^4$：predicted / GT object orientation，**unit quaternion**
- $\hat{\mathbf{q}} \cdot \mathbf{q}$：4D dot product，等于两个 quaternion 在 4D 球面上夹角的 cosine
- 为啥加 **abs**？因为 quaternion 有 double cover：$\mathbf{q}$ 和 $-\mathbf{q}$ 表示同一个 rotation。不加 abs 模型可能瞎学个 sign flip，loss 看起来大其实几何上对。加 abs 把 loss 压到 [0, 1]，0 = 完美对齐。
- 为啥用 $1 - |\cdot|$ 而不是 $\arccos(|\cdot|)$？前者是后者的一阶近似，gradient 更 smooth、好优化。真 geodesic loss 在 $|\cdot| \approx 0$ 附近 gradient 爆炸，训不稳。

这个 loss 看着不起眼，其实是 rotation learning 的标准 trick，在 [Ur5Net / Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) 这些 manipulation 工作里到处都是。

---

## 训练 scheme 是真正的核心 contribution

直接把 5 个 aux decoder 跟 VLM 一起 co-train，结果**变差**：0.60 → 0.51。15% 退化。

为啥？随机 init 的 decoder 输出大梯度，反传回 VLM，把 pre-train 的 web-scale knowledge 全搅乱了。这就是经典的 **negative transfer**。

作者的 fix 是三阶段 progressive：

**Stage 1（3 epochs）：Decoder warm-up**
- detach：aux loss 不能反传到 VLM
- VLM 只接收 action token loss 梯度，继续学主任务
- 5 个 decoder 用 frozen VLM latent 学自己 task
- 直觉：让 decoder 先 "学会读" VLM latent，建立 reasonable baseline

**Stage 2（7 epochs）：Joint refinement**
- 全开 gradient
- 5 个 aux loss + action loss 一起 backprop
- 这时 decoder 已经稳定，产生的 gradient 是 "有意义的 supervised signal"
- 推 VLM latent 去显式编码 spatial 信息，但不会 destroy 已有 representation

**Stage 3（optional）：Action head 单训**
- VLM 全 frozen
- 只训 Flow Matching action expert
- 为啥单独训？因为 denoising loss 跟 next-token CE loss 一起训，denoising 死活不收敛（paper 报告的）

直觉上整个流程像：先让 VLM 当 "老师" 教 decoder 读 latent，再让 decoder 当 "监督员" 推 VLM latent 学得更 structured，最后让 action head 在稳定的 latent 上学连续 action 生成。这种 **curriculum** 思路在 multi-task + pre-trained model 场景几乎必须。

---

## 实验里最 interesting 的几个发现

### 发现 1：multi-view + depth 就能 6× 提升

OpenVLA 7B 单 view RGB：avg 0.04
加 multi-view：0.24（6×）
再加 depth：0.32（再 1.3×）
换 Qwen0.5B + multi-view + depth：**0.60**

**smaller LLM 反而 better**。直觉：tabletop 的 semantic generalization 需要 LLM 大，mobile manipulation 更吃 spatial precision + action coordination，LLM 大不直接帮这些。7B 训练慢、容易过 head、还没收敛。

### 发现 2：history 反而 hurt

加 past 4-step history：0.60 → 0.49。

paper 说 "task 足够 reactive，不需要 history"。我觉得这个解释 weak。pick-and-place 明显有 phase（approach / grasp / lift / transit / place），phase 之间有强 temporal dependency。

更可能的原因：raw history 直接拼到 input，让 VLM attention 被分散。如果用 causal transformer 在 latent level integrate，或者把 history 编码成 phase tag，结果可能不一样。这一块 paper 没深挖，留 future。

### 发现 3：aux task 之间 complementary，不 redundant

| 加哪个 aux | avg success rate |
|---|---|
| baseline | 0.60 |
| + is_grasped | 0.66 |
| + qpos | **0.71** |
| + global pos | 0.65 |
| + all | **0.73** |

"+ all" 接近最优，说明每个 aux 都贡献了 unique aspect，没谁完全 redundant。

但有意思的 trade-off：
- `qpos` 是最稳的，几乎所有 task 都 gain（manipulation 直接需要的本体感觉）
- `is_grasped` 对 pick 帮助最大（pick 的核心 subgoal 就是抓稳）
- `global pos` 对 nav-heavy（open/close fridge/drawer）帮助巨大，但对 pick/place **负贡献**（pick 0.16→0.07）

为啥 global pos 伤 pick/place？直觉：global pos decoder 强迫 VLM 在 pick 时也 focus on 全局位置，把 latent 推向 navigation-friendly sub-space，扰乱了 fine manipulation 的优化方向。这暗示 **conditional auxiliary weighting**（按 subtask type 调 λ）是自然的下一步。

### 发现 4：seg + object pose 是大杀器

只在 pick+place 上训 seg + object pose，结果：

| | avg |
|---|---|
| baseline | 0.27 |
| + seg + obj pos | **0.47** |

**+74% relative**。place 提升尤其猛（最高 +81%）。

直觉：place 任务的瓶颈就是 "target 在哪 + 长啥样"，seg + pose 直接显式监督这两个信息进 latent。这比让 VLM 从 action loss 自己 figure out 容易太多。

### 发现 5：Flow Matching action head 是双刃剑

| | avg | pick | fridge-open |
|---|---|---|---|
| discrete token | **0.73** | 0.13 | 0.87 |
| + flow match | 0.69 | **0.27** | 0.76 |

pick 大涨（+107%），place 涨，但 open/close fridge/drawer 全跌。**clear dichotomy**：

- Fine manipulation（pick/place）：discrete bucketing 损失精度，continuous action 赢
- Navigation-heavy（open/close）：需要 "明确决策" 走哪、转多少，discrete token 提供 commit，flow matching 反而引入 sampling noise

这暗示 future 方向是 **task-adaptive routing**：根据 subtask type 自动选 discrete vs continuous。Paper 没实现，留了 clear direction。

---

## 几个我觉得 subtle / 可疑 / 值得继续做的点

### 1. Depth normalization 有盲区
$$p_{obs} = 1 - \tanh\left(\frac{\text{depth}}{1000}\right)$$

1m 之外基本 saturate 到 0。家用 nav 经常要看 3-5m 外的 fridge 位置。这个 normalize 把远距离 scene 全丢了。改进：multi-scale depth encoding，近/中/远 3 个不同 scale 的 tanh，让远距离也保留 info。

### 2. Pre-train VLM 的 knowledge 保留
Stage 2 全开 aux gradient 进 VLM，会不会侵蚀 VLM 的 web-scale pre-train 知识（语言理解、semantic reasoning）？Paper 没 measure。一个 sanity check：让 SG-VLA 在 novel object name 上测 generalization，看是否退化。这 ablation 缺了。

### 3. Sim-to-real gap 完全没碰
全部在 ManiSkill-HAB sim 上。Real-world depth sensor noise、光照变化、texture gap 都没 test。Aux label 在 sim 里免费，real 里 mask 要靠 SAM2、object pose 要靠 FoundationPose / AR-tag、qpos 是 robot proprio（免费）。理论能 transfer，但 paper 没做。

### 4. Action head isolation 的 trade-off
Stage 3 完全 freeze VLM 训 Flow Match。好处：VLM 不被 denoising loss 干扰。坏处：VLM latent 是为 discrete token 优化的，对 flow match decoding 不一定最优。更好方案：Stage 3 解冻 VLM last layer 或加 LoRA 微调。

### 5. Conditional aux weighting
global pos 在 pick 上 hurt，在 nav 上 help。最自然的 fix：训一个 task classifier，gate aux loss 权重。pick 阶段 λ_pos = 0.1，nav 阶段 λ_pos = 1.0。这个 paper 没做，是 obvious next step。

### 6. 跟其他工作的关系
- [SpatialVLA](https://arxiv.org/abs/2501.15830)：input 端做 spatial encoding，SG-VLA 是 output 端做 spatial supervision，互补
- [DreamVLA](https://arxiv.org/abs/2507.04447)：用 future frame prediction 当 aux，思路类似但是 future frame，SG-VLA 是 current frame 的 intermediate signal
- [π₀](https://arxiv.org/abs/2410.24164)：Flow Match VLA 的老祖宗
- [OpenVLA](https://openvla.github.io/)：tabletop VLA baseline
- [ManiSkill-HAB](https://arxiv.org/abs/2410.19084)：benchmark

合体方向：SpatialVLA 的 input encoding + SG-VLA 的 output aux supervision + DreamVLA 的 future prediction + conditional aux weighting + task-adaptive action routing。理论上能 push 到 80%+。

---

## 最后的 takeaway

如果让我用一句话总结这篇 paper 的 intuition：

**Pre-trained VLM 是个懒学生，光给它 final action 监督它就学个 shortcut mapping。你得给它一套 "mid-term exam"——逼它把 self state、target state、scene structure 全答一遍——它学出来的 intermediate representation 才有 spatial sense，final action 才能精准。**

而且这套 "mid-term exam" 不能一开始就上，否则懒学生还没学会读 VLM latent，瞎给答案，反而把 VLM 的 web-scale 知识搅乱。得先让 decoder 学会读，再让它们反过来监督 VLM。这就是 progressive training 的本质。

engineering 层面最有用的 takeaways：
1. multi-view + depth 在 mobile manipulation 上是 must，单 view RGB 直接废
2. smaller LLM 在 mobile manipulation 上反而 better，别盲目 scale
3. aux task 是 representation shaping 的杠杆，但要 progressive train
4. discrete vs continuous action 是 task-dependent，混合 system 是方向
5. 5 个 aux 不是拍脑袋，覆盖 self state / target state / scene 三类，缺一不可

paper 的 limitation 也明显：sim only、conditional weighting 没做、knowledge retention 没 measure、real-world 没 test。但作为 VLA 扩展到 mobile manipulation 的 early exploration，方法学贡献清晰，ablation 详实，是个 solid work。

---

## References

- [SG-VLA paper (arXiv 待发布)](https://arxiv.org)
- [ManiSkill-HAB benchmark](https://github.com/haosulab/ManiSkill-HAB)
- [OpenVLA](https://openvla.github.io/)
- [π₀ - Flow Matching VLA](https://arxiv.org/abs/2410.24164)
- [SpatialVLA](https://arxiv.org/abs/2501.15830)
- [DreamVLA](https://arxiv.org/abs/2507.04447)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [SigLIP](https://arxiv.org/abs/2303.15343)
- [Qwen2.5](https://arxiv.org/abs/2412.15115)
- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [Mobile ALOHA](https://mobile-aloha.github.io/)
- [Prismatic VLMs](https://arxiv.org/abs/2402.07865)
- [PCGrad - Gradient Surgery for multi-task](https://arxiv.org/abs/2001.06782)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [Habitat 2.0](https://arxiv.org/abs/2106.14405)
- [BEHAVIOR Robot Suite](https://arxiv.org/abs/2503.05652)
- [NaVILA](https://arxiv.org/abs/2410.04423)

---

# SG-VLA: Learning Spatially-Grounded Vision-Language-Action Models for Mobile Manipulation — 深度讲解

## 1. Paper 一句话定位

SG-VLA 来自 UC San Diego (Hao Su & Zhuowen Tu 组) + Lambda Inc.，做的是把 VLA 从 tabletop manipulation 推广到 mobile manipulation (家用机器人)。核心 idea：**与其让 VLM backbone 只学一个 sparse 的 action prediction 监督信号，不如在 latent representation 上挂一组 auxiliary decoders，强制 backbone 把 spatial / manipulation-aware 的信息显式编码进 feature 里**。配合 multi-view + depth 输入，在 ManiSkill-HAB benchmark 上把 average success rate 从 60% (直接 IL) 推到 73%，而 vanilla OpenVLA 在这套设置下只有 4%。

Paper link (arXiv should be searchable): https://arxiv.org/abs/2506.00000 (占位，作者列表与 Ruisen Tu, Arth Shukla 等对得上)；GitHub repo 通常会在 https://github.com/sg-vla 跟进。ManiSkill-HAB benchmark: https://github.com/haosulab/ManiSkill-HAB

---

## 2. 为什么 mobile manipulation 难（与 tabletop 的本质差异）

Tabletop VLA (RT-2, OpenVLA, π₀, CogACT, RDT-1B) 之所以 work，是因为 camera 视角基本固定、scene scale 固定、robot 自由度有限（一般 7-DoF arm + 1-DoF gripper）、base 不动。一旦换到家用 mobile manipulation，几个东西同时塌了：

- **Partial observability**：head camera 看不全房间，必须靠 base motion 主动探索，hand camera 提供 local 细节。
- **13-D action space**：base pose ΔX (3D, position+orientation) + torso height Δz (1D) + arm Δq (7D) + gripper ΔG (2D, 每根 finger 独立)。navigation 和 manipulation 耦合在一个 action vector 里。
- **Spatial reasoning 变成 first-class**：要把 "走到冰箱前 → 抓把手 → 拉 → 松" 串起来，每步 robot 自身 pose 在 world frame 里都在变，光看 egocentric RGB 完全不够。

所以 paper 的 thesis 是：**直接 imitation learning 给的 supervision 太 sparse** — 只有最终 13-D action vector 是 ground truth，模型完全没必要把 "robot 在哪、object 在哪、joint 怎么样" 这些中间信号显式表征出来。Auxiliary tasks 就是把这些 implicit 中间信号变 explicit。

---

## 3. Architecture 总览（Figure 2 解析）

```
[Head RGB + Depth] ┐
                   │──► DINOv2 + SigLIP dual encoder ─► Projector ─┐
[Hand RGB + Depth] ┘                                            │
                                                                ├─► Qwen2.5-0.5B LLM ─► latent ─┬─► discrete action token (training Stage 1)
[Language instruction] ─────────────────────────────────────────┘                                ├─► 5 auxiliary decoders (Stage 2)
                                                                                                  └─► Flow Matching action expert 100M (Stage 3, frozen VLM)
```

具体 sub-components:

### 3.1 Visual encoder
Dual encoder 借鉴 OpenVLA / MiniVLA 的思路：
- **DINOv2** ([Oquab et al. 2023](https://arxiv.org/abs/2304.07193))：自监督 ViT，强 spatial correspondence / patch-level 几何理解，对 depth 也好使。
- **SigLIP** ([Zhai et al. 2023](https://arxiv.org/abs/2303.15343))：sigmoid loss 训出来的 image-text alignment，强 semantic。
- 两者 feature concat 或 fuse 后过 projector 映到 language embedding space。

为什么 dual 而不是 single？DINOv2 不懂 "杯子" 是什么，SigLIP 不懂 patch 哪个对应哪个，合起来才有 semantic + spatial 双 buff。这一招在 OpenVLA、SpatialVLA 里都见过。

### 3.2 LLM backbone
**Qwen2.5-0.5B** ([Yang et al. 2024](https://arxiv.org/abs/2412.15115)) — 故意选小的，paper 说 1.3B 总参数。对比 OpenVLA 7B，smaller 但 mobile manipulation 上反而 higher。直觉：tabletop 的 semantic generalization 需要 LLM 大，mobile manipulation 更需要的是 spatial precision + action coordination，LLM 大不直接帮这些，反而训练慢、容易过 head。

### 3.3 Action head
两条 path：
- **Discrete action token**：直接让 LLM autoregressive 输出 action tokens（RT-2 风格，OpenVLA 风格）。
- **Flow Matching action expert**：100M param，借鉴 [π₀ (Black et al. 2024)](https://arxiv.org/abs/2410.24164)。从 LLM latent 出发 denoise 出连续 13-D action chunk。

Flow Matching 简短回顾（[Lipman et al. 2023](https://arxiv.org/abs/2210.02747)）：diffusion 是从 noise 到 data 的 SDE/ODE，flow matching 直接学一个 vector field v(x, t) 把简单 prior (Gaussian) transport 到 data distribution，比 DDPM 训练更稳、采样更快。Loss 是：

$$\mathcal{L}_{FM} = \mathbb{E}_{t, x_0, \epsilon}\|v_\theta(x_t, t) - u_t(x_t \mid x_0)\|^2$$

其中 t ∈ [0,1] 是 time，x_0 是真实 action chunk，x_t = (1-t)·x_0 + t·ε 是线性插值 path（条件 flow matching 常用），u_t = x_0 - ε 是 target vector field。Paper 这里设 chunk size = 8、denoising step = 10、execute 前 2 步（receding horizon）。

---

## 4. Input Modalities (Section 4.2 + Eq.1)

### 4.1 Multi-view RGB
Head camera 给 global scene layout，hand camera 给 manipulation zone 的 fine-grained 几何。两个 view 互补，避免单 view 的 occlusion + perspective 退化。

### 4.2 Depth normalization — Eq.1 公式拆解

$$p_{obs} = 1 - \tanh\left(\frac{\text{depth value}}{1000}\right) \tag{1}$$

变量含义：
- `depth value`：raw depth sensor 读数，单位 mm（典型 RGB-D / LiDAR in simulation）。值越大 = 离 camera 越远。
- `1000`：scale factor，把 mm 数值压到适合 tanh 的 range（约 0 ~ 几之间）。这是一个 "1 meter" 的 reference scale。
- `tanh(·)`：把任意正 input 映到 (0, 1)，平滑 saturate。
- `1 - tanh(·)`：flip 方向，让 **近 → 1**、**远 → 0**。

直觉：raw depth 大数不利于神经网络学（数值 scale 大、unbounded），tanh 压缩后类似 occupancy / proximity 表示。近的物体激活强，远的物体激活弱。比起直接 /255 normalize 或 log-depth，tanh 有 saturation，对远距离 noise 鲁棒。

可能的小 issue：1m 之外就基本 saturate 到 0，意味着 1m 外的 scene structure 几乎丢了。在 mobile manipulation 里，远距离的 scene layout 也是 important 的（要 navigate），paper 这里没讨论这个 trade-off，是个潜在 ablation 漏洞。

### 4.3 Temporal history
Past 4 timesteps 的 observations/actions。Table 2 显示加上 history 反而 degrade（0.60 → 0.49）。Paper 给的解释是 "model 不好 integrate temporal info" 或者 "task 足够 reactive"。我另一个直觉：4-step history 里如果直接塞 raw actions / images，会引入大量 irrelevant noise；如果用更 structured 的 state-history encoding（比如 latent RNN / causal transformer），结果可能不一样。这一点上 paper 没深挖，留 future work。

---

## 5. Auxiliary Decoders (Section 4.3 + Table 1)

5 个 decoder，每个盯一个 "intermediate signal"，全部从 LLM latent 出发：

| Decoder | Type | Target | Loss |
|---|---|---|---|
| Global Pose | MLP | 2D (x, y) robot position | MSE |
| Grasp Success | MLP | binary | CrossEntropy |
| Object Pose | MLP | 3D pos + quaternion (7D) | Eq.3 |
| Joint Pose (qpos) | Transformer + 12 mask tokens | 12D joint angles | MSE |
| Mask | CNN transpose-conv | 128×128 binary mask | BCE |

### 5.1 为什么 MLP vs Transformer vs CNN 分工

- **MLP** for low-D target (2D pos, 7D object pose, binary grasp)：target dimension 小，latent 已经是高度浓缩的 896-D feature，简单 MLP Proj(896→512) + 3-layer MLP + AvgPooling 足够。
- **Transformer** for 12D qpos：joint angles 之间有强 kinematic dependency（一个 joint 动了下游全动），self-attention 让每个 joint token 都能 attend 到其他 joint，capture kinematic chain。12 个 learnable mask tokens + 2-layer Transformer + sine-cos positional encoding（让 token 知道自己是 joint 1 / 2 / ... / 12）。
- **CNN** for mask：dense 128×128 spatial output，transpose conv 4-stage + BatchNorm + GELU。CNN 对 dense spatial prediction 仍是最经济选择。

### 5.2 Loss 函数全解析

**Eq.2** — Position + Grasp：
$$\mathcal{L}_{pos} = \text{MSE}(\hat{\mathbf{p}}, \mathbf{p}), \quad \mathcal{L}_{grasp} = \text{CrossEntropy}(\hat{y}, y)$$
- $\hat{\mathbf{p}}, \mathbf{p} \in \mathbb{R}^2$：predicted / ground truth robot (x, y) global position。
- $\hat{y}, y \in \{0, 1\}$：predicted / GT grasp state (binary，是否抓稳)。

**Eq.3** — Object pose：
$$\mathcal{L}_{obj} = \|\hat{\mathbf{t}} - \mathbf{t}\|_2^2 + (1 - |\hat{\mathbf{q}} \cdot \mathbf{q}|)$$
- $\hat{\mathbf{t}}, \mathbf{t} \in \mathbb{R}^3$：predicted / GT object 3D position (translation)，L2-squared loss。
- $\hat{\mathbf{q}}, \mathbf{q} \in \mathbb{R}^4$：predicted / GT object orientation，单位 quaternion。
- $\hat{\mathbf{q}} \cdot \mathbf{q}$：quaternion dot product，等于 cosine of 4D angle。当两个 quaternion 表示同一 rotation 时 $|\hat{\mathbf{q}} \cdot \mathbf{q}| = 1$。
- 为什么要 **abs**？因为 quaternion 有 double cover：$\mathbf{q}$ 和 $-\mathbf{q}$ 表示同一 rotation，没 abs 的话 model 可能学到 flip sign，loss 看起来很大其实几何上对。1 - abs 把 loss 范围压到 [0, 1]，0 = 完美。
- 这一项其实是 geodesic loss 的一阶近似（真正 geodesic 是 $\arccos(|\hat{\mathbf{q}} \cdot \mathbf{q}|)$），paper 用线性近似更 smooth。

**Eq.4** — qpos + segmentation：
$$\mathcal{L}_{qpos} = \text{MSE}(\hat{\mathbf{J}}, \mathbf{J}), \quad \mathcal{L}_{seg} = \text{CrossEntropy}(\hat{\mathbf{M}}, \mathbf{M})$$
- $\hat{\mathbf{J}}, \mathbf{J} \in \mathbb{R}^{12}$：12D joint configurations（含 arm 7 + torso 1 + base？paper 没完全说清，从架构图猜是 12 个 actuated joint）。
- $\hat{\mathbf{M}}, \mathbf{M} \in \{0,1\}^{128 \times 128}$：binary mask for target object。
- 注意：seg 用 "CrossEntropy" 而非 "Binary CrossEntropy" 在 paper 里写法略不严谨，实际 binary mask 用 BCE with logits 更常见。

**Eq.5** — Multi-task loss：
$$\mathcal{L}_{auxiliary} = \lambda_{pos}\mathcal{L}_{pos} + \lambda_{grasp}\mathcal{L}_{grasp} + \lambda_{qpos}\mathcal{L}_{qpos} + \lambda_{obj}\mathcal{L}_{obj} + \lambda_{seg}\mathcal{L}_{seg}$$

权重 paper 给的是：$\lambda_{pos}=1.0, \lambda_{grasp}=5.0, \lambda_{qpos}=1.0, \lambda_{obj}=1.0, \lambda_{seg}=1.0$。

**为什么 grasp 权重 5.0**？Binary classification loss 数值量级本来就比 MSE / BCE-dense 小，加大权重避免被淹没。这是一个实用工程 trick，paper 没特别 justify。

---

## 6. Three-Stage Progressive Training (Section 5)

这是 paper 最 critical 的方法学贡献，关系到 auxiliary tasks 是否真帮忙。

### 6.1 问题诊断（Section 5.1）
直接 co-train 随机 init 的 auxiliary decoder + pretrained VLM → 性能 0.60 → 0.51，**变差**。原因：随机 decoder 输出大梯度 → 反传回 VLM → 把 VLM 的预训练 representation 搅乱。

这是 multi-task learning 的经典现象，叫 **negative transfer** 或 **gradient interference**。在 [Multi-task Learning in NLP / vision](https://arxiv.org/abs/2009.09796) 里被广泛研究。

### 6.2 Stage 1 — Decoder Adaptation
- 冻 decoder → VLM 的 gradient flow（detach 操作）。
- VLM 只接收 **discrete action token** loss 的梯度，继续学 action prediction。
- Decoder 用 frozen VLM 的 latent 学自己的 task。
- 持续 3 epochs（10-epoch 总训练的 30%）。

直觉：让 decoder 先 "读懂" VLM latent 里的信息，建立 reasonable baseline。这一阶段 decoder 还不能反过来影响 VLM。

### 6.3 Stage 2 — Joint Refinement
- 解冻，所有 gradient flow 全开。
- 5 个 auxiliary loss + 主 action loss 一起 backprop。
- 持续 7 epochs。

直觉：decoder 已经稳定 → 它们产生的 gradient 现在是 "有意义的 supervised signal"，可以 push VLM latent 去显式编码 spatial / manipulation 信息。这时 VLM 不会被打散，反而被 sharpen。

### 6.4 Stage 3 — Isolated Action Head Training
- 冻 VLM 所有参数。
- 只训 Flow Matching action expert。
- 这个 stage 是必要的，因为 Flow Matching denoising loss 和 discrete token CE loss 一起训时，denoising loss 不收敛（paper 报告）。

直觉：Flow Matching 目标 landscape 和 LM next-token prediction 不同，混训互相干扰。让 VLM 先稳定 latent，再让 action head 学从 latent decode action chunk。

### 6.5 整体训练 timeline

```
Stage 1 (3 ep): decoder warm-up + VLM action-token training
   ↓
Stage 2 (7 ep): joint refinement, VLM + 5 aux decoders + action token
   ↓
[optional] Stage 3: VLM frozen, Flow Matching action expert 训练
```

---

## 7. Dataset (Section 3)

- 基于 [ManiSkill-HAB](https://arxiv.org/abs/2410.19084)（Arth Shukla, Stone Tao, Hao Su, ICLR 2025），simulation-based home rearrangement。
- 3 个 long-horizon task：TidyHouse / PrepareGroceries / SetTable。
- 总计 44K episodes, 1.4M transitions。
- 4 个 atomic subtask：Pick / Place / Open / Close。

**Auxiliary task data 分配策略**（很 subtle 的工程决定）：
- Global position / grasp / qpos prediction：只在 SetTable 上训（8K ep, 240K transitions）— 因为这些 task 需要 robot state info，所有 subtask 都 relevant。
- Segmentation + object pose：在 3 个 task 的 pick+place data 上训（40K ep, 1.2M transitions）— 因为 seg/pose 只在有 target object manipulation 时有意义，open/close drawer 没 target object。

这种 **task-conditioned data allocation** 是工程上的聪明选择，避免 seg decoder 在 open drawer data 上被迫瞎训一通无意义 mask。

---

## 8. Experimental Results 详解

### 8.1 Table 2 — Input Modality Ablation

| Method | Pick | Place | Fridge-Open | Drawer-Open | Fridge-Close | Drawer-Close | Avg |
|---|---|---|---|---|---|---|---|
| OpenVLA 7B (single-view RGB) | 0.00 | 0.19 | 0.02 | 0.00 | 0.00 | 0.04 | **0.04** |
| OpenVLA + Multiview | 0.06 | 0.35 | 0.14 | 0.38 | 0.00 | 0.53 | 0.24 |
| OpenVLA + Multiview + Depth | 0.12 | 0.41 | 0.43 | 0.30 | 0.00 | 0.67 | 0.32 |
| Base VLM (Qwen 0.5B) + Multiview | 0.06 | 0.53 | 0.60 | 0.30 | 0.63 | 0.93 | 0.52 |
| Base VLM + Multiview + Depth | 0.16 | 0.56 | 0.67 | 0.36 | 0.83 | 1.00 | **0.60** |
| + History | 0.00 | 0.47 | 0.57 | 0.40 | 0.47 | 1.00 | 0.49 |

观察：
1. OpenVLA 7B 在 mobile manipulation 几乎全崩（4% avg）。这是 paper 设的 baseline 来 motivate 后续。
2. 仅加 multi-view 就 6×（0.04→0.24）。
3. 再加 depth 再 1.3×（0.24→0.32）。
4. 换 smaller Qwen0.5B 反而更好（0.32→0.60），说明 7B LLM 在这 setting 是 overkill + 训练不够。
5. Depth 对 fridge/drawer 这种需要距离感的 task 提升最大（fridge close 0→0.67），因为要逼近冰箱门必须知道距离。
6. History 反而 hurt，paper 解释为 reactive task 不需 history。但我倾向于另一个解读：history 维度增加引入 representation 负担，model 还没学好怎么用。后续工作 [DreamVLA](https://arxiv.org/abs/2507.04447) 等会用更 structured history。

### 8.2 Table 3 — Progressive Training + 各 Auxiliary Task

| Progressive | Method | Pick | Place | Fridge-Open | Drawer-Open | Fridge-Close | Drawer-Close | Avg |
|---|---|---|---|---|---|---|---|---|
| No | SG-VLA (best from T2) | 0.16 | 0.56 | 0.67 | 0.36 | 0.83 | 1.00 | 0.60 |
| No | + all aux | 0.03 | 0.50 | 0.60 | 0.23 | 0.67 | 1.00 | 0.51 |
| Yes | + is_grasped | 0.30 | 0.53 | 0.83 | 0.57 | 0.80 | 0.93 | 0.66 |
| Yes | + qpos | 0.23 | 0.67 | 0.87 | 0.70 | 0.90 | 0.90 | **0.71** |
| Yes | + global pos | 0.07 | 0.27 | 0.90 | 0.70 | 0.97 | 1.00 | 0.65 |
| Yes | + all | 0.13 | 0.70 | 0.87 | 0.77 | 0.90 | 1.00 | **0.73** |

关键 insights：
1. **Naive co-training 显式失败**：0.60 → 0.51，drop 15%。直接证明 random decoder 干扰 VLM 的猜想。
2. **Progressive training 后 +all**：0.60 → 0.73，提升 22%。证明方法学的价值。
3. **各 aux task 贡献不一**：
   - `qpos` 单独最稳，0.71，几乎全面 gain。直觉：joint 状态对 manipulation 是直接相关的本体感觉 signal。
   - `is_grasped` 对 pick 提升最大（0.16→0.30），因为它直接告诉 model "抓稳没"，pick 任务的核心 subgoal。
   - `global pos` 对 navigation-heavy（open/close fridge/drawer）提升最大，对 pick/place 反而 hurt（0.16→0.07, 0.56→0.27）。直觉：pick/place 不太需要 global pos，反而把 model 推向 navigation-friendly 的 sub-space，扰乱了 manipulation 优化方向。这个 trade-off 暗示 global pos decoder 应该只在 nav-heavy subtask 上启用。
4. **+ all 0.73** 高于任何单一 aux，证明 task 之间 **complementary 而非 redundant**。multi-task 的 grail。

### 8.3 Table 4 — Seg + Object Pose

只在 pick+place 上训，覆盖 3 个 task。

| Method | SetTable Pick | SetTable Place | PrepareGrocery Pick | PrepareGrocery Place | TidyHouse Pick | TidyHouse Place | Avg |
|---|---|---|---|---|---|---|---|
| SG-VLA | 0.16 | 0.56 | 0.10 | 0.33 | 0.07 | 0.40 | 0.27 |
| + seg + obj-pos | **0.26** | **0.78** | **0.13** | **0.60** | **0.33** | **0.73** | **0.47** |

巨大提升（0.27→0.47，**+74%** relative）。直觉：seg + object pose 直接给 "target 在哪 + 长什么样" 这两个 manipulation 最 critical 的 spatial signal，对 place 提升尤其猛（最高 81% gain）。

### 8.4 Table 5 — Flow Matching Action Head

| Method | Pick | Place | Fridge-Open | Drawer-Open | Fridge-Close | Drawer-Close | Avg |
|---|---|---|---|---|---|---|---|
| SG-VLA (discrete token) | 0.13 | 0.70 | 0.87 | 0.77 | 0.90 | 1.00 | **0.73** |
| + action head (Flow Match) | **0.27** | **0.80** | 0.76 | 0.60 | 0.76 | 0.97 | 0.69 |

**Mixed effect**：
- Pick +107%（0.13→0.27）：fine-grained grasp 受益于连续 action 精度。
- Place +14%：放置精度受益。
- Fridge-Open -13%、Drawer-Open -22%、Fridge-Close -16%：navigation-heavy 退化。
- Drawer-Close 基本持平（1.00→0.97）。

直觉：open/close fridge 这种需要 base motion + arm coord 的 task，discrete action token 提供 "明确决策"，flow matching 反而引入 sampling noise。Pick/place 这种 fine manipulation，discrete bucketing 损失精度，continuous 更好。

这暗示 SG-VLA 可以做 **task-adaptive routing**：根据 subtask type 自动选用 discrete vs continuous。Paper 没实现这个，但留了 clear direction。

---

## 9. 几个我想深挖的细节 / 潜在 issue

### 9.1 Depth normalization 的盲区
Eq.1 把 1m 外的 depth 全部 saturate 到 0。家用 mobile manipulation 经常需要看 3-5m 外的 fridge 位置来 navigate，这个 normalization 把远距离 scene structure 都丢了。改进方向：用 `1 - tanh(depth / D)` with adaptive D，或者用 multi-scale depth encoding（近 / 中 / 远 3 个 tanh with 不同 scale），让远距离也保留 information。

### 9.2 History 的负贡献
Paper 给的 "reactive task 不需要 history" 解释不够 convincing。Pick-and-place 其实有明显的 phase（approach / grasp / lift / transit / place），phase 之间有强 temporal dependency。可能 issue 是：raw history feature 直接拼到 input，让 VLM 的 attention 被分散。更 structured 的方案：用 causal transformer 在 latent level integrate history，或者把 history 编码为 phase tag。

### 9.3 Global pos 的 negative transfer 到 pick/place
Table 3 里 global pos 对 pick/place 严重 hurt（pick 0.16→0.07）。这暗示 global pos decoder 在 pick 阶段也在 active，强迫 VLM 在 pick 时也 focus on global position，反而扰乱了 fine manipulation。一个 fix：让 aux decoder 在不同 subtask 上 weighted differently（比如 pick 阶段 λ_pos = 0.1，nav 阶段 λ_pos = 1.0）。Conditional auxiliary weighting 是个自然的 future direction。

### 9.4 1.3B 模型的 generalization
Paper 完全在 simulation 上做 (ManiSkill-HAB)。Sim-to-real gap 没讨论。Auxiliary tasks 在 sim 里 supervised 充足，real-world 里 mask / object pose / qpos 都能获取（mask 用 SAM2，qpos 是 robot proprio，obj pose 用 AR-tag 或者 foundation pose estimator），理论上能 transfer。但 depth sensor noise、光照变化、texture gap 这些没在 paper 里 test。

### 9.5 Flow Matching vs Diffusion
π₀ 用 Flow Matching 替代 DDPM 是因为采样快 + 训练稳。但 SG-VLA Table 5 显示 Flow Matching 在 nav-heavy 退化，说明它的优势 domain-specific。可能 future：用 mixture-of-experts，manipulation head 用 Flow Match，nav head 用 discrete token。

### 9.6 与 SpatialVLA / DreamVLA 的关系
- [SpatialVLA (Qu et al. 2025)](https://arxiv.org/abs/2501.15830)：也探索 spatial representation for VLA，用 3D scene latent + spatial-aware action。SG-VLA 的 aux task 思路和 SpatialVLA 的显式 spatial encoding 是互补的 — SpatialVLA 在 input 端做 spatial encoding，SG-VLA 在 output 端做 spatial supervision。合起来可能更强。
- [DreamVLA (Zhang et al. 2025)](https://arxiv.org/abs/2507.04447)：用 world model dream 做 latent prediction 作为 auxiliary，思路类似 SG-VLA 的 aux decoder，但 dream 是未来 frame prediction，SG-VLA 是当前 frame 的 intermediate signal reconstruction。

### 9.7 Pre-training VLM 的 "知识保留" 问题
Stage 1 frozen VLM for decoder path 只允许 action token gradient 进 VLM。但 Stage 2 全开，所有 aux loss 都 backprop 进 VLM。这时 VLM 的 web-scale pre-train knowledge（语言理解、semantic reasoning）会不会被 manipulation-specific 的 objective 侵蚀？Paper 没 measure 这个。一个 sanity check 是让 SG-VLA 在 language-conditioned task with novel object name 上 generalization 测一下，看是否退化。这个 ablation 缺了。

### 9.8 Action head 训练的 isolation trade-off
Stage 3 完全 freeze VLM 训 Flow Matching。好处是 VLM 不被 denoising loss 干扰。坏处是 VLM 的 latent 可能不是最优的 for Flow Matching decoding — 因为 VLM 在 Stage 1/2 是为 discrete token 优化的 latent。更好的方案可能是 Stage 3 解冻 VLM 的 last layer 或加 LoRA 微调。

---

## 10. 给直觉 build 的核心 takeaways

1. **Auxiliary supervision 是 representation shaping 的杠杆**：让 backbone 同时学多个 task，等于在每个 layer 都加 explicit constraint，强迫 feature 编码 multi-faceted 信息。这不是新发现（multi-task learning 几十年了），但 VLA 之前没人系统地 explore。
2. **Progressive training 在 multi-task + pre-trained model 场景几乎是必须的**：随机 init 的 head 直接 co-train 会 destroy pre-train representation。Stage 1 让 head 先 "学读" feature，Stage 2 再 "推" feature 去适配 head。这个 stage 划分本质上是一种 **curriculum**。
3. **Input modality 的边际收益**：multi-view 是 6×、depth 再 1.3×、smaller LLM 再 2×、aux training 再 1.2×。最大的 gain 不是 model size，而是 perception 和 supervision design。这是 mobile manipulation 与 tabletop 最大的区别。
4. **Discrete vs Continuous action 的 task-dependence**：fine manipulation 用 continuous、locomotion-heavy 用 discrete。混合 system 是方向。
5. **失败模式是分 task 的**：naive co-training 全 task 退化；global pos 只 pick/place 退化；Flow Match 只 nav 退化。说明 multi-task 系统的 failure mode 不是 uniform，需要 per-task analysis。

---

## 11. 我会怎么 extend 这工作

如果让我（hypothetically）接着做：
1. **Conditional auxiliary weighting**：根据 subtask type 自动调 λ。需要先训一个 task classifier head，再 gate aux loss。
2. **Latent-level auxiliary**：现在 aux decoder 直接从 VLM latent decode 到 target。可以加 contrastive loss 让 latent 在 same-task episode 里更 cluster，cross-task 更 separable。
3. **Real-world transfer**：用 SAM2 / FoundationPose 提供 mask + object pose GT，验证 sim-to-real。
4. **Longer horizon**：现在 4 atomic subtask。家用任务往往几十步。引入 hierarchical planning：VLM 高层 subgoal + SG-VLA 低层 execution。
5. **Cross-embodiment**：13-D action 是特定 robot。能不能 mask 掉不同 DoF 来支持不同 robot？Open X-Embodiment 思路。
6. **Action chunk size ablation**：Flow Match 用 chunk=8 execute 2。chunk 大小 vs reactivity 是 trade-off，没 explore。
7. **Self-supervised aux**：现在 aux 需要 GT。能不能从 video 自监督 reconstruct？比如 future frame prediction 作为 auxiliary。

---

## 12. 相关参考 links

VLA foundations:
- [OpenVLA (Kim et al. ICML 2024)](https://openvla.github.io/) — 7B baseline
- [π₀ (Black et al. 2024)](https://arxiv.org/abs/2410.24164) — Flow Matching VLA
- [π₀.5 (Physical Intelligence 2025)](https://arxiv.org/abs/2504.16054) — open-world VLA
- [RT-2 (Brohan et al. CoRL 2023)](https://robotics-transformer2.github.io/)
- [CogACT (Li et al. 2024)](https://arxiv.org/abs/2411.19650)
- [RDT-1B (Liu et al. 2024)](https://arxiv.org/abs/2410.07864)
- [MiniVLA (Belkhale & Sadigh 2024)](https://arxiv.org/abs/2410.11195)
- [VQ-VLA (Wang et al. ICCV 2025)](https://arxiv.org/abs/2412.18009)
- [SpatialVLA (Qu et al. 2025)](https://arxiv.org/abs/2501.15830)
- [DreamVLA (Zhang et al. 2025)](https://arxiv.org/abs/2507.04447)

Mobile manipulation:
- [Mobile ALOHA (Fu et al. CoRL 2024)](https://mobile-aloha.github.io/)
- [ManiSkill-HAB (Shukla et al. ICLR 2025)](https://arxiv.org/abs/2410.19084) — SG-VLA 的 benchmark
- [Habitat 2.0 (Szot et al. NeurIPS 2021)](https://arxiv.org/abs/2106.14405)
- [BEHAVIOR Robot Suite (Jiang et al. 2025)](https://arxiv.org/abs/2503.05652)
- [NaVILA (Cheng et al. RSS 2025)](https://arxiv.org/abs/2410.04423)

Vision-language backbones:
- [DINOv2 (Oquab et al. 2023)](https://arxiv.org/abs/2304.07193)
- [SigLIP (Zhai et al. ICCV 2023)](https://arxiv.org/abs/2303.15343)
- [Prismatic VLMs (Karamcheti et al. ICML 2024)](https://arxiv.org/abs/2402.07865)
- [Qwen2.5 (Yang et al. 2024)](https://arxiv.org/abs/2412.15115)

Generative action models:
- [Flow Matching (Lipman et al. ICML 2023)](https://arxiv.org/abs/2210.02747)

Datasets:
- [Open X-Embodiment](https://robotics-transformer-x.github.io/)

Multi-task learning theory:
- [Multi-Task Learning survey](https://arxiv.org/abs/2009.09796)
- [Gradient Surgery (PCGrad)](https://arxiv.org/abs/2001.06782) — 解决 gradient interference，SG-VLA 没用但相关

---

## 13. 总结

SG-VLA 的核心 contribution 不在 architecture（dual encoder + Qwen + 5 个 head 都是标准件），而在 **训练方法学**：把 multi-task auxiliary supervision + progressive stage training 应用到 VLA 上，证明 dense supervision 能显式 shape backbone 的 latent representation，让 VLA 在 mobile manipulation 上从 60% 推到 73%。Paper 的工程价值在于 **"什么 aux task 对什么 subtask 有用"** 的精细 ablation，以及 **progressive training 必要性** 的实验证明。

未来方向我更看好：conditional aux weighting、sim-to-real、cross-embodiment、和 latent-level contrastive aux。SG-VLA 是把 "VLA 也要 representation learning" 这件事 formalize 了，下一步是把这个 representation learning 推到 self-supervised + real-world scale。
