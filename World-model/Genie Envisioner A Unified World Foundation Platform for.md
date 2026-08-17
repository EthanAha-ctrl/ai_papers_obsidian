---
source_pdf: Genie Envisioner A Unified World Foundation Platform for.pdf
paper_sha256: d8802a311ff9db058bf8f25cb5ccc03711e32b11cd5f02a1759324ddb3cfbd71
processed_at: '2026-08-04T21:05:53-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Genie Envisioner 人话版

好，我换个讲法，假装我们 coffee chat，我给你把这 paper 捋一遍。

---

## 一句话讲完

这篇 paper 做的事：**教一个 video diffusion 模型去"想象"机器人干活的过程，然后从这种"想象"里直接读出 action 来控制机器人，顺便还能当 simulator 用来评测 policy**。

就这么个事。听起来简单，做起来一堆坑，他们把这些坑都填了一遍。

---

## 为什么要搞这个？

现在做 robot manipulation 主流路线是 VLA：拿一个 VLM（vision-language model），给它看图片 + 指令，让它直接输出 action。RT-2、OpenVLA、π0、GR00T N1、UniVLA 都这条路线。

GE 团队觉得这路线有个根本问题：**VLM 把视觉压成 language token 的过程，丢掉了 spatial-temporal 细节**。

举个例子，"把水倒进杯子"这个 task，VLM 看一眼场景，encode 成一段语言式的 representation，然后从这段 representation 里 decode 出 action。但倒水这个动作的关键细节——壶口倾斜角度、水流速度、杯子快满了要减速——这些是 spatiotemporal dynamics，语言 token 表达不了。

GE 的 thesis：**与其把视觉压成语言，不如直接在 visual latent space 里做 generative modeling，让 model 学会"在脑子里播放机器人干活的视频"，然后从这种播放里把 action 读出来**。

这其实跟人挺像的。你倒水之前，脑子里会先"预演"一遍倒水的过程（壶怎么抬、水怎么流、杯子什么时候满），然后手照着这个预演去动。GE 就是把这个过程 modeling 出来。

参考文献类比：
- Ha & Schmidhuber 的 World Models（2018）：https://arxiv.org/abs/1803.10122 —— 最早提"在 latent space 里做梦"的思路
- Dreamer 系列：https://arxiv.org/abs/2304.10573 —— 在 learned world model 里做 planning
- DeepMind 的 Genie：https://arxiv.org/abs/2402.15391 —— interactive 环境，概念上很像

---

## 三个组件，各管一摊

GE 平台就三层，一层套一层：

### GE-Base：会"做梦"的底层 model

核心是个 video diffusion transformer（DiT），base architecture 用了 LTX-Video 2B（轻量、快）或 COSMOS2 2B（高质量）。给它三个 view 的摄像头画面 + 一句 language instruction，它生成接下来机器人干活的视频。

输入包括：
- 当前观察 $\mathbf{x}_0$（三个 view：head cam + 两个 wrist cam）
- 历史稀疏采样帧 $\hat{\mathbf{x}}_{0:t-1}$（从过去所有 chunk 里挑 4 帧）
- Language instruction $q$（用 T5-XXL 编码）

公式：
$$\mathbf{x}_{1:N}^{(t)} = \mathcal{W}(\hat{\mathbf{x}}_{0:t-1}, \mathbf{x}_0, \tau(q))$$

意思就是：**第 $t$ 步要生成的 video chunk（$N$ 帧）= 一个 world model $\mathcal{W}$，吃进去历史稀疏帧 + 当前观察 + 指令文本 embedding**。

那个 **sparse memory** 是关键 trick。意思是模型不需要记住每一帧历史，只挑 4 帧稀疏采样作为"记忆点"。这相当于人记事——你不会记得倒水的每一毫秒，但你记得"壶抬起来""水开始流""杯子快满了"这几个 key moment。

训练的时候 sparse memory 是随机挑的（数据增强），推理的时候是固定间隔均匀挑。这样 model 被迫学会从远期历史里挑 relevant cue，对各种 frame rate 都 robust。

**Multi-view cross-attention** 也值得一提。三个 view 共同生成，部分 DiT block 做跨 view attention（三个摄像头互相交换信息），部分 block 各 view 独立处理。Hidden state shape $(B, N, T, H, W, C)$，$N$=3 是 view 数。这是为了 efficiency——全部 block 都跨 view 太贵。

### GE-Act：从"做梦"里读出 action 的 plug-and-play module

GE-Base 会生成视频了，但生成视频不能直接驱动 robot。需要一个模块把 visual latent 翻译成 motor command。

GE-Act 就干这个。它是个 160M 参数的轻量 action decoder，跟 GE-Base 的 visual backbone 并行跑，用 flow-matching（diffusion 的变种）做 action denoising。

公式：
$$\mathbf{v}_i = B_i^{\text{vis}}(\mathbf{v}_{\text{in}}, \mathcal{T}(q))$$

$$\mathbf{a}_i = \mathcal{B}_i^{\text{act}}\left(\mathbf{z}_{\text{act}}, \mathbf{CrossAttn}(\mathbf{z}_{\text{act}}, \mathbf{v}_i)\right)$$

意思就是：**GE-Base 的第 $i$ 层 visual block 输出 visual feature $\mathbf{v}_i$，GE-Act 的第 $i$ 层 action block 把 noise-initialized action token $\mathbf{z}_{\text{act}}$ 拿过来，通过 cross-attention 从 $\mathbf{v}_i$ 里"取"出 visual context，然后 refine 出 action representation $\mathbf{a}_i$**。

整个 forward 跑完，最后输出 54 步 torque trajectory @ 30 Hz，在 RTX 4090 上 **200ms 内跑完**，real-time 够用。

这里有个超级聪明的 trick 叫 **slow-fast asynchronous inference**：
- Video DiT 每次只跑 **1 步 denoising**（生成 visual latent，cache 起来）
- Action model 跑 **5 步 denoising**，都用同一份 cached visual latent
- Video @ 5 Hz, Action @ 30 Hz（比例 1:6）

直觉类比：**大脑每秒"想"5 次大概要干嘛（low-frequency planner），小脑每秒执行 30 次精细动作（high-frequency controller）**。这跟人骑车、弹琴的视觉-运动 hierarchy 是一回事。

技术上为什么能这么干：因为 video latent 不需要 pixel-perfect，只需要"概念上对"就够 action model 用；action trajectory 需要精细，所以多 refine 几步。这跟 LCM（Latent Consistency Model）的 1-step inference 思路类似：https://arxiv.org/abs/2310.04360

### GE-Sim：把 model 当 simulator 用

GE-Base 加点东西就能当 simulator 用。给它初始 observation + 一段 action trajectory，它生成 action 执行后的视频。这就替代了 MuJoCo / Isaac Gym 这种 physics engine，省去手动建模环境的成本。

技术上 trick 叫 **hierarchical action-conditioning**。问题：action 是 14D vector（双 arm 各 7D：xyz + rpy + gripper open），但 video model 是 token-based 的，怎么把 14D vector 喂进去？

两条 path：

**Path 1（空间维度）**：把 pose 渲染成 image
- Position $(x,y,z)$ 用相机内外参投到 pixel 坐标
- Orientation $(r,p,y)$ 转 rotation matrix 的正交轴投影
- Gripper openness 渲染成 unit circle（open 浅、closed 深）
- Left/right arm 用不同颜色

得到一张 "pose image" $\mathbf{P}_i$，跟历史 frame $\mathbf{I}_i$ 一起用 shared encoder 编码，element-wise add：
$$\mathbf{v}_i = \mathcal{E}(\mathbf{I}_i) + \mathcal{E}(\mathbf{P}_i)$$

意思是：**visual context token = 历史 frame 编码 + pose 渲染图编码**，两者加在一起喂给 video model。

直觉类比：把 action "翻译"成 image 让 video model 看得懂。跟 ControlNet 用 canny edge / depth map 当 condition 是同一个 idea：https://arxiv.org/abs/2302.05543

**Path 2（时间维度）**：算 motion delta
$$\Delta \mathbf{a}_i = \mathbf{a}_i - \mathbf{a}_{i-1}$$

意思：**第 $i$ 步 pose 减第 $i-1$ 步 pose = 这一步的运动增量**。Encode 成 motion token 通过 cross-attention 注入。

训练 GE-Sim 的时候他们还专门加了 **failure cases**（erroneous executions, incomplete behaviors, suboptimal trajectories）。这点很关键——大部分 generative model 只见过成功 trajectory，但 deployment 时 policy 会 generate 失败 action，simulator 必须能 simulate 失败，否则 sim-to-real gap 巨大。

Closed-loop simulation workflow：
1. 给 instruction + 初始观察
2. Policy 输出 action
3. GE-Sim 用 action + 初始观察生成 next video chunk
4. 生成的 video 反馈给 policy 出下一步 action
5. 迭代

GE-Sim 还能当 **data engine**——同一 action 在不同初始环境跑，生成 diverse 数据。

---

## 训练 pipeline：三阶段 curriculum

GE-Base 的 pretraining 分两个 stage：

**Stage I (GE-Base-MR)**：domain adaptation
- 57-frame clips，frame rate 随机 3-30 Hz
- 4 帧 sparse memory
- 32 A100 × 7 天

类比：先让模型"见多识广"，各种速度都见过，部署时 sensor latency / frame drop 不会让它崩。

**Stage II (GE-Base-LF)**：policy alignment
- 9-frame clips @ 5 Hz fixed
- 2 latent frames
- 32 A100 × 3 天

类比：把 temporal granularity 校准到跟 downstream action policy 对齐（GE-Act 用 5 Hz visual feature 来 predict 30 Hz action）。

然后 GE-Act 的训练：
- **Stage 0**：action pretraining on AgiBot-World-Beta（~1M episodes, 2967 hours），GE-Base frozen，只训 action head，video generation 关掉（用 cached latent），16 A100 × 3 天
- **Stage 1**：task-specific video adaptation（AgiBot-World + task subset ×10 upweight），8 A100 × 12 小时
- **Stage 2**：task-specific action specialization（full model fine-tune），8 A100 × 36 小时

跨 embodiment adaptation（Agilex Cobot Magic / Dual Franka）只用了 1 小时 teleoperation 数据（250 demos），但还是 two-stage：
1. 先微调 video DiT 让它学会生成新 robot 的视频
2. Action head 从 scratch 训（action space semantics 不同，没法 transfer）

---

## 实验里最值得记住的几个数

### Table 1 的关键数据

| Pretraining | Adaptation | E2E (w/ State) | SR (w/ State) |
|-------------|------------|----------------|---------------|
| 无 | 无 | 0.15 | 0.05 |
| General video (LTX) | Task VidAda | 0 | 0 |
| GE-Base (in-domain) | 无 | 0.81 | 0.64 |
| GE-Base (in-domain) | Task VidAda | 0.89 | 0.76 |

**这个表讲了个大事**：general video pretraining 完全帮不上忙（success rate 0%），in-domain embodied pretraining 才是真正起作用的。

为什么？因为 general video 学的是 human-centric motion prior，但 robotic manipulation 需要的是 action-conditioned dynamics，这两个 distribution 是 categorical 不同的。Visual appearance 容易 transfer（光照、物体、场景），但 dynamics transfer 不了。

这跟 NLP 里 "general LM pretraining → medical fine-tune" 不一样，language distribution shift 比较小，但 visual dynamics distribution shift 是 categorical 的。

### Figure 11 的 cross-embodiment 数据

在 Agilex Cobot Magic 上做 cloth folding / box folding：
- π0、UniVLA、GR00T N1 几乎全 0% success
- 只有 GE-Act 能完成

在 Dual Franka 上做 cloth folding：
- GE-Act 用 1 小时 adaptation 数据，超过了 extensively trained 的 π0 和 GR00T N1

直觉解释：**GE-Base 的 visual dynamics 是 embodiment-agnostic 的（"物体怎么动"），所以 visual backbone 能 transfer；action space 是 embodiment-specific 的（关节 angle vs EEF pose vs torque），所以 action head 必须重训**。

### Figure 2 那个 candy/stamp demo

Task："Yellow candy needs blue stamp, white candy needs red stamp. Fold a box, put appropriate candy inside, seal the box, apply correct stamp based on candy type."

这个 task 难点：
1. Deformable object（fold a box）
2. **Memory-based decision**：candy 放进盒子后看不见，要根据 memory 选 stamp
3. Cross-embodiment：Agilex Cobot Magic，pretraining 没见过
4. 只 1 小时 adaptation data

GE-Act 完成了。这说明 sparse memory 机制能把 long-horizon 信息保留在 visual latent space 里，model 能"记住"candy 类型。这是 VLA 路线难做到的——language token 里很难塞这种 "看不见但记得" 的信息。

---

## EWMBench：robotic 专用 video benchmark

他们还专门做了个 benchmark，因为 general video metric（FVD、MSE、VBench）跟 robotic task success 弱相关。

EWMBench 测三个维度：

**Scene Consistency**：用 fine-tune 过的 DINOv2 提 patch embedding，算连续帧之间 cosine similarity。高了说明场景稳定。

**Action Trajectory Quality**：
- Spatial alignment (SA)：用 symmetric Hausdorff distance 测两条 trajectory 的最大偏离
  $$\mathrm{SA}_{\text{score}} = \frac{1}{d_{\text{symH}}(G, P) + \epsilon}$$
  其中 $G$ 是 GT trajectory，$P$ 是生成的，$\epsilon$ 防除零

- Temporal alignment (TA)：用 Normalized Dynamic Time Warping，允许时间轴弹性对齐
  $$\mathrm{TA}_{\text{score}} = \frac{1}{d_{\text{NDTW}}(G, P) + \epsilon}$$

- Dynamic consistency (DYN)：比较 velocity / acceleration 分布的 Wasserstein distance
  $$\mathrm{DYN}_{\text{score}} = \alpha \cdot \frac{\min(\Delta v^{\text{gt}}, \Delta v^{\text{pred}}) + \epsilon}{\max(\Delta v^{\text{gt}}, \Delta v^{\text{pred}}) + \epsilon} \cdot \frac{1}{W(v)} + \beta \cdot \frac{\min(\Delta a^{\text{gt}}, \Delta a^{\text{pred}}) + \epsilon}{\max(\Delta a^{\text{gt}}, \Delta a^{\text{pred}}) + \epsilon} \cdot \frac{1}{W(a)}$$
  其中 $\Delta v$ 是 velocity amplitude（max - min），$W(\cdot)$ 是 Wasserstein distance，$\alpha=0.007, \beta=0.003$（velocity 权重比 acceleration 高，因为 accel 噪声大）

**Motion Semantics**：
- Global alignment：VLM 给 video 生成 summary caption，跟 instruction 比 BLEU
- Key-step consistency：VLM 给 GT 和 generated video 都生成 step-by-step description，对应步骤算 CLIP similarity
- Logical correctness：GPT 定义错误 taxonomy（hallucinated actions, object disappearances, physically implausible motions），VLM 检测
- Diversity：$1 - \text{CLIP}_{\text{sim}}$ between generations under same instruction

EWMBench 结果（Figure 17a）：

| Model | Scene | Motion | Semantics | Total |
|-------|-------|--------|-----------|-------|
| GE-Base | 0.9427 | **1.6676** | 2.0907 | **4.7010** |
| Kling | 0.8888 | 0.9440 | 2.0370 | 3.8698 |
| Hailuo | 0.8577 | 0.5362 | 2.0186 | 3.4125 |
| COSMOS | 0.7963 | 0.7085 | 1.7824 | 3.2872 |
| OpenSora | 0.9210 | 0.3442 | 1.8739 | 3.1392 |
| LTX | 0.9156 | 0.4002 | 1.6518 | 2.9676 |

GE-Base 在 Motion 维度几乎翻倍第二名（1.6676 vs Kling 0.9440），说明 embodied pretraining 主要帮助 motion dynamics。Scene 和 Semantics 各家差距小，因为这是通用 video model 本来就擅长的。

他们还做了 human preference correlation（Figure 18）：EWMBench ranking 跟人类评分强 concordance，VBench 在 embodied 场景下 misaligned。这证明这套 metric 设计是 task-relevant 的。

---

## 我觉得这 paper 真正的 contribution

1. **统一 framework**：policy learning + evaluation + simulation 全在一个 video generative model 里。这避免了 VLA 路线里 VLM training 和 simulator 互相 decoupled 的尴尬。

2. **Sparse memory 机制**：把 long-horizon temporal reasoning 从 dense sequence 问题转成 sparse retrieval 问题。这个 idea 可能比这篇 paper 本身更有价值。

3. **Slow-fast asynchronous inference**：1-step video denoising + 5-step action denoising + 1:6 frequency mismatch。这个 deployment trick 让 video diffusion 能 real-time 跑在 robot 上。

4. **Cross-embodiment 只需 1 小时**：visual backbone transferable + action head 重训这个范式，证明了 embodied foundation model 的可行性。

5. **EWMBench**：第一个认真的 robotic world model 评估体系，跟 human preference 对齐。

---

## 几个我没在 paper 里看到、但值得思考的点

1. **Video diffusion 是非 causal 的**（chunk 内部 bidirectional attention），但 deployment action 是 strict causal 的。这个 train/inference distribution mismatch paper 没讨论，可能 long-horizon 会出问题。

2. **Closed-loop error accumulation**：GE-Sim 多步 rollout noise 累积，visual state 可能 drift 到 OOD。Long-horizon closed-loop robustness 没数据。

3. **本质还是 behavior cloning**：supervised on teleoperated trajectories，不能 recover from errors 或 explore alternative strategies。World model 框架本来可以支持 model-based RL，paper 没走这条——留给了 follow-up。

4. **没整合 sim 数据或 internet video**：只用了 AgiBot-World-Beta。如果想做 "one foundation, hundreds of embodiments"，可能需要 explicit embodiment token 让 model 知道是哪个 robot。

5. **1 小时 cross-embodiment 很 impressive，但还不到 AGI manipulation**：dexterous hand、locomotion、whole-body coordination 都没测。

---

## 整体直觉总结

如果让我用一句话 build 你的 intuition：

**GE = 把 "在脑子里想象机器人干活" 这个过程 explicitly 建模成 video diffusion，然后从这种想象里把 action 读出来，顺便把想象本身当 simulator 用**。

它跟 VLA 的本质区别：VLA 把 visual 压成 language 再 decode action，丢了 spatiotemporal 细节；GE 直接在 visual latent space 里走完整条 perception → imagination → action 链路，保留了所有 fine-grained dynamics。

它跟 Dreamer 的本质区别：Dreamer 用 RSSM 这种 explicit latent dynamics model，capacity 有限；GE 用 video diffusion 作为 implicit dynamics model，容量大得多，scaling 也更顺畅。

它跟 Genie（DeepMind）的本质区别：Genie 做 interactive game environment，没考虑 action grounding 和 real robot deployment；GE 把这个 idea 拉到 real-world manipulation 上，加了 multi-view、cross-embodiment、real-time inference 一整套工程。

我直觉这是 "video diffusion as foundation for embodied AI" 路线目前最完整的工作。架构、训练、inference、benchmark、real-world 实验都到位。剩下的就是 scale up + open up（他们承诺 code/model/benchmark 全 release）。如果后续做 model-based RL、加 dexterous hand、整合 internet-scale data，这条路线还能继续走很远。

主要 references：
- Paper project page: https://genie-envisioner.github.io
- AgiBot-World dataset: https://arxiv.org/abs/2503.06669
- π0: https://arxiv.org/abs/2410.24164
- GR00T N1: https://arxiv.org/abs/2503.14734
- UniVLA: https://arxiv.org/abs/2505.06111
- LTX-Video: https://arxiv.org/abs/2501.00103
- COSMOS: https://arxiv.org/abs/2501.03575
- EnerVerse (前作): https://arxiv.org/abs/2501.01895
- EWMBench: https://arxiv.org/abs/2505.09694
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- DeepMind Genie: https://arxiv.org/abs/2402.15391
- DreamerV3: https://arxiv.org/abs/2304.10573
- ControlNet: https://arxiv.org/abs/2302.05543
- LCM: https://arxiv.org/abs/2310.04360

---

# Genie Envisioner (GE) 深度技术讲解

Andrej，这篇 paper 我读了三遍，下面我从 architecture、training、inference、benchmark、limitations 几个维度把所有关键的技术细节都拆解给你，并附上相关的联想和 references。

---

## 1. 核心动机与整体定位

GE 想解决的核心问题是：现有的 robot manipulation pipeline 由 disjoint 的 data collection、policy training、evaluation 三个 stage 组成，每个 stage 都需要 bespoke infrastructure，friction 严重。GE 的 thesis 是：**把 sensing、policy learning、evaluation 全部 collapse 进一个 video-generative world model**，形成 closed-loop。

这跟主流 VLA 路线（RT-2, OpenVLA, GR00T N1, π0, UniVLA）的根本差异在于 representation space：
- 主流 VLA：vision → VLM 把视觉 encode 成 language-centric semantic representation → 在这个 representation 上学 policy
- GE：vision → 直接在 visual latent space 里 generative modeling，保留 spatial-temporal cues → 从这个 vision-centric space 里 decode action

直觉上，robotic manipulation 本质是一个 spatiotemporal dynamics 问题，把它强制压成 language token 会丢掉 fine-grained geometry。GE 走的路线更接近 "world model as policy backbone"，类似 Dreamer 系列的思路，但用 video diffusion 而非 RSSM 作为 dynamics backbone。

References:
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- Dreamer / TD-MPC2 / DayDreamer: https://arxiv.org/abs/1912.01603, https://arxiv.org/abs/2310.16828
- Genie (DeepMind): https://arxiv.org/abs/2402.15391

---

## 2. GE-Base: World Foundation Model

### 2.1 基本建模思路

GE-Base 把 robotic video world modeling 形式化为 **text-and-image-to-video** 生成问题。关键设计是 **sparse memory mechanism** —— 这是 long-horizon tasks 的关键。

公式：
$$\mathbf{x}_{1:N}^{(t)} = \mathcal{W}(\hat{\mathbf{x}}_{0:t-1}, \mathbf{x}_0, \tau(q))$$

变量含义：
- $\mathbf{x}_{1:N}^{(t)}$：第 $t$ 步要生成的 video chunk，包含 $N$ 帧（N=9 in Stage II）
- $\hat{\mathbf{x}}_{0:t-1}$：从过去所有 chunk 里稀疏采样得到的历史帧（4 帧）
- $\mathbf{x}_0$：初始 visual observation
- $\tau(q)$：language instruction $q$ 经 T5-XXL 编码后的 text embedding
- $\mathcal{W}$：world model（DiT backbone）

**直觉**：sparse memory 是一种 data augmentation + temporal reasoning 机制。Dense memory (所有过去帧) 在 autoregressive 长 horizon 下会导致 latent 序列爆炸；纯 causal mask 又会让早期 context 被 forget。Sparse sampling 强迫模型从远期历史中"挑出" relevant cues，类似 Transformer-XL 的 segment recurrence 但更激进。

### 2.2 Architecture 细节

GE-Base 用了两个 backbone：
- **LTX-Video 2B**：轻量、快，用于 GE-Act 的 real-time control
- **COSMOS2 2B**：高保真，用于 GE-Sim 的高质量仿真

Multi-view 设置（dual-arm egocentric）：
- Head view: $v^h$
- Left wrist: $v^l$
- Right wrist: $v^r$

每个 frame 都被 video encoder $\mathcal{E}$ 编码成 latent token：
- $\mathcal{E}(v_0^{(i)})$：第 $i$ 个 view 的初始观测 token
- $\mathcal{E}(v_{t-1}^{(i)})$：第 $i$ 个 view 的历史帧 token

Position embedding 设计很有意思：
- **2D rotary positional embedding** $e_{\text{pos}}$：保留空间对齐
- **View-specific learnable embedding** $e_{\text{view}}$：区分 viewpoint-specific 信息
- **Timestep encoding** $e_t$：denoising timestep

这三者叠加在每个 token 上。

### 2.3 Cross-view Attention 的 Hybrid Scheme

这是 architecture 里最 tricky 的部分。Hidden state shape 是 $(B, N, T, H, W, C)$，其中 $N$ 是 view 数量（=3）。

Cross-view self-attention 在 $(N, H, W)$ 维度上做，让三个 view 互相 attend。但全部 block 都这样做计算量太大。所以 GE 用 hybrid：
- **Selected DiT blocks**：cross-view attention，shape 为 $(B, N, T, H, W, C)$
- **其他 blocks**：把 $N$ 折叠进 batch，变成 $(B \cdot N, T, H, W, C)$，各 view 独立处理

这个设计非常像 ViT 的 mixture-of-depths，也让人联想到 multi-view NeRF 里的 epipolar attention。

完整预测公式：
$$\hat{x}_t = \mathcal{W}\left(\{v_0^{(i)}, v_{\hat{t}}^{(i)}, z^{(i)}\}_{i \in \{h,l,r\}}, \mathcal{T}(q)\right)$$

变量：
- $v_0^{(i)}, v_{\hat{t}}^{(i)}$：第 $i$ 个 view 编码后的初始 & 历史视觉 token
- $z^{(i)}$：第 $i$ 个 view 的 noise map
- $\mathcal{T}(q)$：language instruction 的编码

References:
- LTX-Video: https://arxiv.org/abs/2501.00103
- COSMOS (NVIDIA): https://arxiv.org/abs/2501.03575
- RoPE ( rotary positional embedding): https://arxiv.org/abs/2104.09864
- EnerVerse (同 lab 前作): https://arxiv.org/abs/2501.01895

---

## 3. GE-Base 的两阶段 Pre-training

这是 paper 里非常关键、但又容易被 skim 过去的部分。两个 stage 不是简单的 progressive training，而是在做 **temporal abstraction 的 calibration**。

### Stage I: GE-Base-MR (Multi-Resolution Temporal Adaptation)

- 57-frame clips, frame rate 随机从 3 Hz 到 30 Hz
- 4 sparse memory frames
- 编码进 8-frame latent space via VAE
- 32 A100 GPUs, ~7 days

**直觉**：这一阶段是 domain adaptation。General video model（LTX-Video 在 generic video 上训过）学的是 "human-centric motion priors"，没见过 robotic arm、gripper、control dynamics。随机 frame rate 3-30 Hz 是关键 trick —— 让模型对 sampling rate 不敏感，这样部署时 sensor latency、frame drops 不会 catastrophic failure。

### Stage II: GE-Base-LF (Low-Frequency Policy Alignment)

- 9-frame clips @ 5 Hz fixed
- 4 sparse memory frames
- 2 latent frames
- Only video generation components updated (VAE frozen)
- 32 A100, ~3 days

**直觉**：这一阶段是 **与 downstream action policy 的 temporal granularity 对齐**。GE-Act 后续会用 5 Hz visual features 来 predict 30 Hz action sequences。如果 visual feature 的 frame rate 太高（如 Stage I 的 30 Hz），latency 太大；太低（如 1 Hz），又会丢掉 sub-second dynamics。5 Hz 是 sweet spot。

这两阶段合起来就是一个 "broad → narrow" 的 curriculum：先学 invariance，再学 task-specific granularity。

---

## 4. GE-Act: World Action Model

### 4.1 Architecture

GE-Act 是一个 **plug-and-play parallel action branch**，核心思路：

- 与 GE-Base 的 visual backbone **并行**（而非串联）
- 保持 GE-Base 的 DiT block depth，但 **reduce hidden dimension**
- **160M 参数**（vs GE-Base 的 2B）
- 用 flow-matching 做 action denoising

每个 timestep 的 forward：
$$\mathbf{v}_i = B_i^{\text{vis}}(\mathbf{v}_{\text{in}}, \mathcal{T}(q))$$

$$\mathbf{a}_i = \mathcal{B}_i^{\text{act}}\left(\mathbf{z}_{\text{act}}, \mathbf{CrossAttn}(\mathbf{z}_{\text{act}}, \mathbf{v}_i)\right)$$

变量：
- $\mathbf{v}_i$：GE-Base 的第 $i$ 个 visual DiT block 的输出
- $B_i^{\text{vis}}$：GE-Base 的第 $i$ 个 visual block
- $\mathbf{v}_{\text{in}}$：输入视觉 token
- $\mathbf{z}_{\text{act}}$：noise-initialized action tokens（diffusion/flow-matching 的 noise prior）
- $\mathcal{B}_i^{\text{act}}$：GE-Act 的第 $i$ 个 action DiT block
- $\mathbf{a}_i$：第 $i$ 层的 action representation
- $\mathbf{CrossAttn}(\mathbf{z}_{\text{act}}, \mathbf{v}_i)$：action token 对 visual feature 做 cross-attention，把 visual context 注入 action pathway

**直觉**：这相当于把 GE-Base 当成一个 "frozen visual encoder" 用，但用 cross-attention 而不是 pooling 把 visual features 取出来。这跟 π0 把 VLM 的 token hidden state 喂给 flow-matching action expert 是同样的 philosophy，但 GE 的 visual backbone 是 video diffusion 而不是 VLM，所以输出 token 里自然带 spatiotemporal dynamics，而不是 language abstraction。

### 4.2 三阶段训练

**Stage 0: Action-space pre-training**
- AgiBot-World-Beta（~1M episodes, 2967 hours）
- GE-Base-LF **fixed**
- Only action decoder updated
- Video generation **disabled**（关键 efficiency trick）
- 4-frame visual memory @ 5 Hz → predict 54-step action @ 30 Hz
- 16 A100, ~3 days

这里 **video generation disabled** 是个重要细节：因为 visual memory 是已经 frozen 的 GE-Base-LF 输出的 latent，不需要重新跑 video diffusion，直接用 cached latent 作为 condition。这把 training cost 从 "每步都跑 video diffusion" 降到 "每步只跑 action diffusion"。

**Stage 1: Video adaptation**
- Update only video generation components of $\mathcal{W}$
- Dataset: AgiBot-World corpus + task-specific subset（upweight ×10）
- 8 A100, 12 hours

**Stage 2: Action specialization**
- Full model fine-tuned on task-specific data
- 8 A100, 36 hours

两阶段 adaptation 的逻辑：先用 task-specific visual data 把 visual backbone 微调到能 generate 这个 task 的视频（不需要 action 监督），再用 task-specific action data 训 action head。这是 visual → action 的解耦。

### 4.3 Slow-Fast Asynchronous Inference

这是 deployment 的核心 trick，我觉得是 paper 里最聪明的部分之一。

**Asymmetric denoising**：
- Video DiT: **1 denoising step** per inference → 生成 visual latent，cache 起来
- Action model: **5 denoising steps**, all conditioned on cached visual latent
- 54 action steps in **200ms** on NVIDIA RTX 4090

**Frequency mismatch**：
- Video DiT @ 5 Hz
- Action model @ 30 Hz
- 比例 1:6

这意味着：每跑 1 次视频 forward，跑 6 次 action forward，但 video forward 只做 1 步 denoising，所以总 cost 大致是 $1 \times \text{cost}_{\text{video}} + 6 \times 5 \times \text{cost}_{\text{action-step}}$。

**直觉**：这本质上是把 video generation 当成一个 "low-frequency world state estimator"，把 action generation 当成 "high-frequency motor controller"。这跟人的视觉-运动控制层级很像：视觉感知是 coarse 更新的，motor control 是 fine-grained 高频的。也类似 LLM agent 框架里 "System 2 planner" + "System 1 executor" 的 dual-process architecture。

更深一层：这种 design 让 video latent space 的 dimensionality 下降到只需要 sparse future frames，不需要 dense frame sequence。这避免了 "为 predict 高频 action 而必须 generate 高频 video" 的浪费。

---

## 5. Cross-Embodiment Generalization

实验设计的精髓在于 **few-shot adaptation protocol**：
- AgiBot G1: in-domain（训练用平台）
- Agilex Cobot Magic: 250 demos (~1 hour) 
- Dual Franka: 250 episodes (~1 hour), 用 space-mouse 采集
- RoboTwin (simulator): 200 demos (50/task × 4 tasks), **all-in-one** training

**Two-stage adaptation for new embodiment**：
1. Video DiT 微调（CLIP + video encoder frozen），让 model 学会生成新 embodiment 的视频
2. Action DiT **从 scratch 训练**（保留 GE-Base visual backbone）

注意：action head 是从 scratch 训的，因为不同 embodiment 的 action space semantics 完全不同。但 visual backbone 不需要重训 —— 这印证了 GE-Base 学到的是 embodiment-agnostic 的 visual dynamics。

实验结果（Figure 11）：
- 在 Agilex Cobot Magic 上做 cloth folding / box folding：π0、UniVLA、GR00T N1 几乎全部 0% success，只有 GE-Act 能完成
- 在 Dual Franka 上做 cloth folding：GE-Act 用 1 小时 adaptation 超过了 extensively trained π0 和 GR00T N1
- RoboTwin all-in-one：4 个 task 中 3 个超过 task-specific baselines，第 4 个（lift pot）略低，可能是 task interference

**Table 1 的关键发现**：

| VidAW | VidAda | E2E (w/ S) | E2E (w/o S) | SR (w/ S) | SR (w/o S) |
|-------|--------|------------|-------------|-----------|------------|
| ✗ | ✗ | 0.15 | 0.30 | 0.05 | 0.11 |
| ✗ | ✓ | 0 | 0.05 | 0 | 0 |
| ✓ | ✗ | 0.81 | 0.49 | 0.64 | 0.26 |
| ✓ | ✓ | 0.89 | 0.37 | 0.76 | 0.37 |

关键 insight：
- General video pretraining (LTX-Video) + task VidAda 几乎是 0% success → general video pretraining 帮助有限
- In-domain embodied pretraining (GE-Base) → 64-81%
- In-domain + VidAda → 76-89%
- Robot state (S) 只有在 VidAW 时才 help；直接加到 general-video-pretrained 模型上反而 hurt，因为 shortcut learning

**这个结果非常有意思**：它说明 robotic manipulation 不是 "transfer learning free lunch" 的领域。Generic video pretraining 学到的是 visual appearance prior，但 manipulation 需要的是 **action-conditioned dynamics**，这个 distribution shift 在 generic video 里完全没体现。GE-Base 之所以有效，是因为它在 AgiBot-World-Beta 上学到了 robotic-specific dynamics。

References:
- GR00T N1: https://arxiv.org/abs/2503.14734
- π0: https://arxiv.org/abs/2410.24164
- UniVLA: https://arxiv.org/abs/2505.06111
- AgiBot-World: https://arxiv.org/abs/2503.06669

---

## 6. GE-Sim: World Simulator

GE-Sim 把 GE-Base 转成 **action-conditioned video generator**，可以用来做 closed-loop policy evaluation 和 controllable data generation。

### 6.1 Action Representation

每个 control step：
- 单 arm: 7D vector $[x, y, z, \text{roll}, \text{pitch}, \text{yaw}, o]$
  - $(x, y, z)$：end-effector position
  - (roll, pitch, yaw)：orientation
  - $o$：gripper openness
- Dual-arm: 14D（两 arm concat）
- K-step horizon: $\mathbf{A} \in \mathbb{R}^{K \times 14}$

### 6.2 Hierarchical Action-conditioning

这是 GE-Sim 最 tricky 的部分。问题：GE-Base 是 token-based video model，但 action 是低层 control command，**两个 modality 的 semantic gap 巨大**。GE 用了两条路径：

**Path 1: Pose2Image Conditioning（空间维度）**

把 pose vector $a_i = [x_i, y_i, z_i, r_i, p_i, y_i, o_i]$ 投影到 image space：
- Position $(x_i, y_i, z_i)$ 用 calibrated camera intrinsics/extrinsics 投到 pixel 坐标
- Orientation $(r_i, p_i, y_i)$ 转 rotation matrix，正交轴投影到 image plane 表示方向
- Gripper openness $o_i$ 渲染成 unit circle，open=light, closed=dark
- Left/right arm 用不同 color 编码

得到 pose image $\mathbf{P}_i$，与对应 history frame $\mathbf{I}_i$ 一起用 shared encoder $\mathcal{E}$ 编码，element-wise add：

$$\mathbf{v}_i = \mathcal{E}(\mathbf{I}_i) + \mathcal{E}(\mathbf{P}_i)$$

变量：
- $\mathbf{v}_i$：fused token，包含 visual context + explicit pose
- $\mathbf{I}_i$：第 $i$ 步历史 frame
- $\mathbf{P}_i$：第 $i$ 步的 pose image

**直觉**：这个设计本质是 "把 action 转成 image" 来桥接 semantic gap。Action 是抽象的 vector，video model 看不懂；pose 渲染成 image 后，video model 就能用 visual reasoning 处理它。类似的想法在 ControlNet 的 canny edge / depth map conditioning 里见过。

**Path 2: Motion Vector Conditioning（时间维度）**

计算 motion delta：
$$\Delta \mathbf{a}_i = \mathbf{a}_i - \mathbf{a}_{i-1} = [\Delta \mathbf{p}_i, \Delta \mathbf{r}_i]$$

变量：
- $\mathbf{a}_i = [\mathbf{p}_i, \mathbf{r}_i]$：第 $i$ 步 6-DoF pose
- $\mathbf{p}_i \in \mathbb{R}^3$：position
- $\mathbf{r}_i \in \mathbb{R}^3$：orientation
- $\Delta \mathbf{a}_i$：motion delta（encode 位置+朝向的变化）

这些 delta 用 learnable encoder 编码成 motion tokens，concat 到 reference image style token，通过 cross-attention 注入每个 DiT block。

**Reference image**：用 frozen CLIP image encoder 编码，作为 style anchor 通过 cross-attention 注入，保持 visual consistency。

**直觉**：Pose2Image 给 spatial conditioning（这一步手在哪、指向哪），Motion Vector 给 temporal conditioning（手怎么动）。两条 path 分工明确，类似 diffusion model 里的 spatial condition + temporal condition 分离设计。

### 6.3 Training

- Init from GE-Base-MR（高 temporal resolution 版本）
- 训练用 AgiBot-World-Beta 全量
- **加入 failure cases**（erroneous executions, incomplete behaviors, suboptimal trajectories）
- VAE 和 CLIP frozen
- Flow-matching loss

**加入 failure cases** 是个很 critical 的设计 —— 大部分 generative model 只见过成功 trajectory，但 deployment 时 policy 会 generate 失败 action，simulator 必须能 simulate 这些情况，否则 sim-to-real gap 巨大。

### 6.4 Closed-loop Simulation Workflow

1. 给 language instruction + 初始 visual observation
2. Policy model 输出 action trajectory
3. GE-Sim 用初始 observation + action policy 生成 video chunk（action outcome）
4. 生成的 video 反馈给 policy model + 原 instruction → 下一步 action
5. 迭代直到 instruction 完成

GE-Sim 也能作为 **data engine**：同一 action trajectory 在不同初始 visual environments 下执行，生成 diverse manipulation sequences。

---

## 7. EWMBench: 评估体系

EWMBench 是这个 paper 真正的 contribution 之一。General video generation 的 metric（FVD、MSE、VBench）跟 robotic task success 弱相关，需要一套专门 metric。

### 7.1 Benchmark Dataset 构造

- 从 AgiBot-World-Beta test set 选 10 个 task
- 每个 task 4-10 个 atomic sub-actions
- 每个 sub-action 标 step-level caption
- 每 task 采样 100 个 video instance
- **Trajectory selection**：dual-arm EEF trajectory voxelized 成 3D grid，pairwise 3D IoU similarity，greedy 选 least-overlapping trajectories → 保证 motion diversity

**直觉**：robotic manipulation 评估里，trajectory diversity 比场景多样性更关键。同一 task 不同 trajectory 体现不同策略，避免 benchmark 只测一种策略。

### 7.2 Metrics

**Scene Consistency**：
- DINOv2 fine-tune 在 robotic data 上
- 提取 patch-wise embedding
- Consecutive frames 之间 cosine similarity

**Action Trajectory Quality**：

Spatial Alignment (SA)：
$$\mathrm{SA}_{\mathrm{score}} = \frac{1}{d_{\mathrm{symH}}(G, P) + \epsilon}$$

变量：
- $G$：ground-truth trajectory
- $P$：generated trajectory
- $d_{\mathrm{symH}}$：symmetric Hausdorff distance（双向最大点距离）
- $\epsilon$：small constant 防除零

**直觉**：Hausdorff 测两条 trajectory 的最大偏离，symmetric 保证双向 worst case。

Temporal Alignment (TA)：
$$\mathrm{TA}_{\mathrm{score}} = \frac{1}{d_{\mathrm{NDTW}}(G, P) + \epsilon}$$

变量：
- $d_{\mathrm{NDTW}}$：Normalized Dynamic Time Warping distance

**直觉**：DTW 允许时间轴的弹性对齐，适合 "动作相似但速度不同" 的情况。NDTW 是 normalized 版本，便于跨 task 比较。

Dynamic Consistency (DYN)：
$$\mathrm{DYN}_{\mathrm{score}} = \alpha \cdot \frac{\min(\Delta v^{\text{gt}}, \Delta v^{\text{pred}}) + \epsilon}{\max(\Delta v^{\text{gt}}, \Delta v^{\text{pred}}) + \epsilon} \cdot \frac{1}{W(v)} + \beta \cdot \frac{\min(\Delta a^{\text{gt}}, \Delta a^{\text{pred}}) + \epsilon}{\max(\Delta a^{\text{gt}}, \Delta a^{\text{pred}}) + \epsilon} \cdot \frac{1}{W(a)}$$

变量：
- $\Delta v = \max(v) - \min(v)$：velocity amplitude
- $\Delta a = \max(a) - \min(a)$：acceleration amplitude
- $W(\cdot)$：Wasserstein distance between velocity/acceleration distributions
- $\alpha = 0.007, \beta = 0.003$：weighting
- $\epsilon = 10^{-8}$

**直觉**：这个 metric 有两个 component：
1. **Amplitude ratio** $\frac{\min}{\max}$：测 amplitude 一致性，防止 low-dynamic case 不稳定
2. **Wasserstein distance**：测 velocity/acceleration 分布的 similarity，不要求 strict temporal correspondence

$\alpha > \beta$ 说明 velocity 比 acceleration 更重要 —— 合理，因为 acceleration 噪声更大。

**Motion Semantics**：

Three-level evaluation:
- **Global alignment**：VLM 生成 summary caption → BLEU 对比 instruction
- **Key-step consistency**：VLM 对生成视频和 GT 视频都生成 step-by-step description → CLIP similarity 对应步骤
- **Logical correctness**：GPT 定义 logical errors taxonomy (hallucinated actions, object disappearances, physically implausible motions) → VLM detect 是否出现

**Diversity**：
$$\text{Diversity} = 1 - \text{CLIP}_{\text{sim}}$$

Pairwise CLIP similarity 同一 instruction 下不同 generation 之间，取 1 - sim。

### 7.3 EWMBench 结果

**Figure 17a**：

| Model | Scene | Motion | Semantics | Score |
|-------|-------|--------|-----------|-------|
| GE-Base | **0.9427** | **1.6676** | **2.0907** | **4.7010** |
| Kling | 0.8888 | 0.9440 | 2.0370 | 3.8698 |
| Hailuo | 0.8577 | 0.5362 | 2.0186 | 3.4125 |
| COSMOS | 0.7963 | 0.7085 | 1.7824 | 3.2872 |
| OpenSora | 0.9210 | 0.3442 | 1.8739 | 3.1392 |
| LTX | 0.9156 | 0.4002 | 1.6518 | 2.9676 |

**关键观察**：
- GE-Base 在 Motion 上几乎翻倍第二（1.6676 vs Kling 0.9440）→ embodied pretraining 主要帮助 motion dynamics
- Scene Consistency GE-Base 最高（0.9427），LTX 也高（0.9156），说明 architecture 本身 OK，关键是 domain pretraining
- Semantics 各家差不多，因为通用 VLM 能力都过关，分不出高下

**Table 2 (GE-Sim evaluation)**：

| Model | BLEU | CLIP | DYN | Div. | PSNR | SA | Log. | TA | Scn. |
|-------|------|------|-----|------|------|----|------|----|----|
| LTX | 0.33 | 90.8 | 0.78 | 0.011 | 19.9 | **0.94** | 0.97 | **0.98** | 0.90 |
| COSMOS | 0.31 | 90.2 | **0.85** | 0.010 | **20.7** | 0.87 | 0.97 | 0.97 | **0.91** |

**关键观察**：
- COSMOS2-based 在 DYN（0.85 vs 0.78）和 PSNR（20.7 vs 19.9）上更好 → 高保真 simulator
- LTX-based 在 SA（0.94 vs 0.87）和 TA（0.98 vs 0.97）上更好 → spatial/temporal alignment 精确
- Diversity 都很低（0.010-0.011）→ fixed action 应该产生 deterministic video，diversity 低说明 action-to-video mapping 准确

**Metric-Human Consistency** (Figure 18)：EWMBench ranking 与 human preference 强 concordance，VBench 在 embodied consistency / goal-conditioned reasoning 场景下 misaligned。

References:
- EWMBench paper: https://arxiv.org/abs/2505.09694
- VBench: https://arxiv.org/abs/2311.13522
- DINOv2: https://arxiv.org/abs/2304.07193
- PhyGenBench: https://arxiv.org/abs/2410.05363

---

## 8. Real-World 实验细节

### AgiBot G1 上的 5 个 task：
1. **Make a sandwich**: 多 object 协调、spatial reasoning、procedural execution
2. **Pour a cup of tea**: 精确 pouring、fluid manipulation
3. **Clean the table**: wiping motion、compliant force
4. **Heat food in microwave**: articulated object + multi-stage interface
5. **Pack laundry detergent**: dynamic perception、conveyor 追踪

### Evaluation Metrics：
- **Step-wise Success Rate (SR)**: 每步独立评估，成功步 / 总步
- **End-to-End Success Rate (E2E)**: 只看最终 outcome，允许中间失败 retry

**直觉**：SR 严格反映 partial completion 能力，E2E 反映 deployment 真实情况（允许 recovery）。两个 metric 配合可以诊断 "哪步最容易 fail"。

### Figure 2 的 demo 很 impressive

Task: "Yellow candy needs blue stamp, white candy needs red stamp. Fold a box, put appropriate candy inside, seal the box, apply correct stamp based on candy type."

这个 task 有几个特别难点：
1. **Deformable object**（fold a box）
2. **Memory-based decision**：candy 放进盒子后看不见，要根据 memory 选 stamp
3. **Cross-embodiment**：Agilex Cobot Magic，pretraining 没见过
4. **Only 1 hour adaptation data**

GE-Act 成功完成，说明 visual world modeling 对 long-horizon memory-intensive sequence 有优势 —— 因为 GE-Base 的 sparse memory 机制把历史信息保留在 visual latent space 里。

---

## 9. 一些 inferred 的技术点

### 9.1 为什么 general video pretraining 帮不上？

Table 1 显示 LTX-Video (general video pretrained) + VidAda 几乎 0% success。我推断原因是：
- General video 里 robotic motion 的 distribution 极其稀疏
- Visual appearance prior 容易 transfer（场景、光照、物体），但 action-conditioned dynamics transfer 不了
- LTX-Video 见过的是 human-centric motion，mechanical motion 的 prior 是 wrong 的

这跟 NLP 里 "general LM pretraining → medical fine-tune" 不同，因为 language distribution shift 比较小，但 visual dynamics distribution shift 是 categorical 的。

### 9.2 Video Diffusion as World Model 的本质

GE 实际上把 video diffusion 当成了一个 implicit dynamics model。每个 video chunk 的生成过程等价于：
$$p(\mathbf{x}_{1:N}^{(t)} | \hat{\mathbf{x}}_{0:t-1}, \mathbf{x}_0, q) = \int p(\mathbf{x}_{1:N}^{(t)} | \mathbf{z}, \text{conditions}) p(\mathbf{z}) d\mathbf{z}$$

其中 $\mathbf{z}$ 是 noise。这跟 standard world model $p(s_{t+1} | s_t, a_t)$ 的形式上区别是 action condition 被 implicit 化了 —— 通过 video latent space 里的 dynamics 来表达。

GE-Act 把 action 从 video latent 中 decode 出来，相当于在 latent dynamics model 上加 action readout。这跟 Dreamer 在 RSSM latent 上加 action head 是同构的，只是 backbone 从 RSSM 换成 video diffusion。

### 9.3 Sparse Memory 与 KV cache 的关系

Sparse memory 4 帧 vs autoregressive 9 帧 chunk，加起来每步 input 是 13 帧 latent。这跟 LLM 的 KV cache + sliding window 思路类似，但是 vision 版本。论文里没明说，但 training 时 sparse memory 是 random sample（data augmentation），inference 时是 uniform sample at fixed interval —— 这个 train/inference mismatch 应该是 acceptable 的，因为 model 见过各种 sampling rate。

### 9.4 为什么 video DiT 只 1 step denoising 就够？

这是 asymmetric inference 的 magic。直觉：
- Flow matching 的 1-step ODE 求解已经能给出 "good enough" visual latent
- Action model 用 5 步 refine，是因为 action 需要高精度（54 步 trajectory）
- Video 不需要 pixel-perfect，只需要 "conceptually correct" 的 latent 给 action model 做 cross-attention

类似 LCM-LoRA 的 1-step inference 思路，但用在 robotic context。

### 9.5 Cross-embodiment transfer 的真正机制

为什么 GE-Base 视觉 backbone transferable，但 action head 必须 scratch？

我推断：
- Visual dynamics（物体如何移动、gripper 如何接触物体）是 embodiment-agnostic 物理
- Action space（关节 angle vs EEF pose vs torque）是 embodiment-specific
- Visual latent 学的是 "what happens"，action head 学的是 "how to make it happen"

所以 transfer 时只 transfer "what happens" 的 knowledge，"how to" 必须重新学。这跟 LLM finetune 时只训 head 不动 backbone 的逻辑同构。

---

## 10. Limitations 和我的思考

Paper 自己列了三个 limitations：
1. **Data coverage**: 只用 AgiBot-World-Beta，没整合 internet-scale 或 sim 数据
2. **Embodiment scope**: 只测 upper-body tabletop + parallel-jaw gripper，没 dexterous hand、locomotion
3. **Evaluation methodology**: EWMBench 还是 proxy metric，自动评估 task success 仍 open

我觉得还有几个没提到的：

4. **Causal confusion risk**：Video diffusion 不是 causal model，它生成的 video 在 chunk 内部是非 causal 的（bidirectional attention）。但 deployment 时 action 是 strict causal 的。这个 train/inference distribution mismatch paper 没讨论。

5. **Closed-loop error accumulation**：GE-Sim 是 video-based simulator，每次生成都有 noise，closed-loop 多步后 noise 累积，最终 visual state 可能 drift 到 OOD。Paper 没给 long-horizon closed-loop 的 robustness 数据。

6. **No exploration**：GE 是 behavior cloning 本质（supervised on teleoperated trajectories），不能 recover from errors 或 explore alternative strategies。这是 VLA 共性问题，但 world model 框架本来可以支持 model-based RL，paper 没走这条路。

7. **Compute cost**：32 A100 × 7 days (Stage I) + 32 A100 × 3 days (Stage II) + 16 A100 × 3 days (action pretraining) ≈ 5,000+ A100-hours，加上 task-specific adaptation。这个 cost 不算小，但相比 GR00T N1 / π0 的训练 cost 应该 comparable。

References:
- DreamerV3: https://arxiv.org/abs/2304.10573
- LCM (Latent Consistency Model): https://arxiv.org/abs/2310.04360

---

## 11. 整体评价与研究方向联想

GE 是一个相当完整的 "vision-centric embodied foundation" 工作，整合度比单独的 VLA 或 video generation model 高。几个值得 follow 的方向：

1. **Action-conditioned video model as unified policy + simulator**：GE-Sim + GE-Act 共享 backbone 这件事，理论上可以做 model-based RL（用 GE-Sim rollout 来 improve GE-Act）。Paper 没做这个，留给后续。

2. **Long-horizon memory**：sparse memory + visual latent memory（Figure 2 candy/stamp task）这个范式很 promising，但 4 帧 sparse memory 对 ultra-long-horizon（100+ steps）够不够存疑。可能需要 retrieval-augmented memory。

3. **Cross-embodiment at scale**：1 小时 adaptation 已经 impressive，但要真正做到 "one foundation, hundreds of embodiments"，可能需要 embodiment token / embodiment embedding 让模型显式知道是哪个 robot。

4. **Visual latent vs language latent debate**：GE 这条路线（visual-centric）和 VLA 路线（language-centric）的最终胜负还要看 scaling。我倾向于 hybrid —— GE 的 visual backbone + language planner 可能是下一步。

5. **World model benchmarking**：EWMBench 是 first serious attempt at evaluating robotic world models，但 human preference correlation 的论证还弱，需要更大规模 human study。

总的来说，GE 这个 paper 是 "video diffusion as foundation for embodied AI" 路线目前最完整的工作之一。架构设计、训练 curriculum、inference efficiency、benchmark 体系都到位，real-world 实验覆盖 in-domain + cross-embodiment + simulator。剩下的就是 scale up + open up（代码、模型、benchmark 都说会 release）。

希望这些细节和联想对你 build intuition 有帮助，Andrej。

主要 references：
- Paper project page: https://genie-envisioner.github.io
- AgiBot-World: https://arxiv.org/abs/2503.06669
- π0: https://arxiv.org/abs/2410.24164
- GR00T N1: https://arxiv.org/abs/2503.14734
- UniVLA: https://arxiv.org/abs/2505.06111
- LTX-Video: https://arxiv.org/abs/2501.00103
- COSMOS: https://arxiv.org/abs/2501.03575
- EnerVerse: https://arxiv.org/abs/2501.01895
- EnerVerse-AC: https://arxiv.org/abs/2505.09723
- EWMBench: https://arxiv.org/abs/2505.09694
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- Genie (DeepMind): https://arxiv.org/abs/2402.15391
- DreamerV3: https://arxiv.org/abs/2304.10573
