---
source_pdf: CTRL-WORLD A CONTROLLABLE GENERATIVE.pdf
paper_sha256: 5acee75221001b53171078e8b02f85151147b6dc629b36cf086d91d1c921ba0f
processed_at: '2026-08-03T17:55:51-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Ctrl-World 人话版

## 一句话讲完

他们造了一台"做梦机器"，让robot policy可以在梦里反复练习和考试，练习完真机成绩从38分飙到83分。

## 为什么要搞这个

训练一个robot policy有个特别尴尬的死循环：

- 你想测试它行不行 → 要在真机上跑几百次
- 你想让它在某个场景上变强 → 要找专家再示范几百次
- 两个都要花时间花钱花人力

Chelsea Finn组在Stanford，他们天天面对这个问题。π0.5这种SOTA policy在没见过的物体上还是会跪，你想知道"跪在哪些case上"就得一遍遍rollout。

所以他们就问了一个很Karpathy的问题：**能不能让policy在"想象空间"里自己跑自己练？**

这就需要一台做梦机器——给定当前画面和一个动作，预测接下来画面会变成啥样。听起来就是video prediction嘛，但事情没那么简单。

## 三个关键技术，用类比讲

### 1. 多视角同时预测——为什么单视角是个坑

之前的world model只生成一个第三人称摄像头视角的画面。听起来没毛病，但实战会出大事：

机械臂去抓一个杯子。第三人称摄像头离得远，夹爪和杯子接触那一瞬间，画面上其实就是一小块像素overlap了。模型根本看不清"到底接没接触上"。结果就是模型经常产生"物体凭空瞬移进夹爪"的幻觉——你没碰到，它认为碰到了。

类比一下：你站在房间角落看一个人缝衣服，针和线离你太远，你以为针已经穿过去了，其实没有。

解决方法：加上手腕摄像头。手腕摄像头离工作区只有十几厘米，contact时刻能拍到高分辨率的近景。模型一看到"哦确实接触上了"，就不会再瞎编。

而且这个wrist camera是π0、π0.5这些VLA policy的标配输入。world model必须同时出third-view和wrist-view才能跟policy对接上。所以他们在token维度把三个视角concat起来一起预测，让attention能在view之间交互，建立3D一致性。

### 2. Frame-level动作注入——为什么sequence-level不行

这是controllability的核心。

想象两种情况：
- 你告诉模型"接下来1秒内动作是[左移、上抬、夹紧、放下、右移]"
- vs. 你告诉模型"接下来1秒大概就这些动作"

第一种是frame-level，第二种是sequence-level。区别在哪？

Sequence-level把整个action chunk当一个全局token喂进去，模型知道"整体意图"但不知道哪个动作对应哪一帧画面变化。结果就是模型平均掉了动作的影响力，动作之间的差异被模糊化。

Frame-level是每一帧的画面tokens都cross-attend到它对应的那一个action embedding。第3帧的画面attend到第3个动作。这样动作的因果效应就被精确对齐到时间轴上了。

消融实验说话最响：去掉frame-level conditioning，FVD从97.4飙到122.7，PSNR掉2个点。动作的厘米级精度也是这么来的——他们实验里给三个不同action chunks（Z轴±6cm，X轴±6cm），生成的画面差异清晰可见。

### 3. Pose作为记忆的索引——长horizon不漂移的秘诀

这个我觉得是paper最巧妙的设计。

问题：world model要连续生成10秒甚至20秒的rollout。但每生成一帧，模型就有微小误差，误差累积下来10秒后画面就degenerate了。怎么让模型"记住"之前的样子？

naive做法：把所有过去帧塞进context。问题：context爆炸，且远处帧不一定relevant。

他们的做法：sparse sampling——往前每隔1-2秒采一帧，总共采7帧。但这7帧怎么用？关键insight：**用机械臂的pose作为"记忆的key"**。

为什么pose好用？因为robot manipulation的运动有准周期性。机械臂在某个pose时，看到的场景大概率跟历史上某个相似pose时看到的场景高度相似。所以预测未来某帧时，模型通过attention机制自动去历史里找pose匹配的帧，从那里retrieve视觉细节。

这就像你写代码时遇到一个bug，你不会从头回忆整个项目，你会去git history里找类似的commit。Pose就是这里的commit hash。

Figure 4的attention可视化直接打脸怀疑者：预测t=4s帧时，attention权重强集中在t=0s的帧——而t=0s和t=4s恰好是机械臂pose相近的两个时刻。

## 两个用途

### 用途一：在梦里考试（Policy Evaluation）

让π0、π0-FAST、π0.5三个policy在world model里跑7个任务（pick-place、fold-towel、drawer、wipe-table、close-laptop、pull-tissue、stack），看每个policy的"成功率"。

结果：world model给的排名和真机排名高度一致。Drawer任务真机成绩π0 (5%) < π0-FAST (20%) < π0.5 (80%)，world model也给π0 (0%) < π0-FAST (20%) < π0.5 (65%)同样的排序。

但有一个gap：world model的**意图识别**（instruction following）很准，但**底层执行成功率**会低估。比如close-laptop π0.5真机能关上70%，world model里只有5%。

原因：复杂physics（笔记本盖子合上的碰撞、滑动物体）建模不精确；DROID的failure data不够覆盖所有policy的失败模式。所以world model有时会把"本来能成的"预测成"失败了"。

启示：要做一个真正可靠的policy evaluator，需要持续把policy的real rollout data喂回world model训练（这正是1X World Model的思路）。

### 用途二：在梦里练习（Policy Improvement）

这是真正impressive的结果。

流程：
1. 准备一些policy没见过的downstream任务（novel objects、rephrased instructions）
2. 在world model里rollout 400条trajectory per task
3. 用instruction rephrasing（LLM改写指令）+ random initial state perturbation增加diversity
4. 用human preference筛选出25-50条"成功"的trajectory
5. 拿这些synthetic trajectory去SFT π0.5-DROID 2k steps

结果数字：
- Spatial Understanding：28.75% → 87.5%
- Shape Understanding：43.75% → 91.25%
- Towel folding with direction：57.5% → 80%
- Novel objects：25% → 75%
- 平均：38.7% → 83.4%（提升44.7%）

这相当于zero-shot真机数据，全靠做梦做出来的提升。

为什么这么有效？我的intuition：π0.5本身有strong prior，它"会"做这些事，只是对特定的instruction format或object appearance没align好。Synthetic trajectory起的作用类似LLM的instruction tuning——不是教新技能，是把已有的capability shape到target distribution上。少量"成功示范"就够了。

## 我个人的几个takeaway

**1. 这个东西能work的本质是SVD prior强**

SVD在互联网视频上学到了texture、motion、object affordance的强prior。Ctrl-World只新初始化了一个3-layer MLP来project action，其他参数全部继承SVD。这说明video model确实是个"weak world model"，你只需要轻量adaptation就能给它装上action control。

**2. Failure data是world model能做evaluator的前提**

DROID有95k trajectories，其中19k是失败的。这点很关键。如果你只用success data训，world model会变成"乐观模拟器"——所有action都预测成成功，根本区分不了policy好坏。Failure data让模型学到"什么action会导致失败"。

**3. Reward model是下一个bottleneck**

paper里筛successful trajectory用的还是human preference。这scalability受限。下一步应该是VLM-based reward model（Du et al. 2023做过相关工作 https://arxiv.org/abs/2303.07280），让整个pipeline真正自动化。

**4. 真正的下一步是MCTS in imagination**

Ctrl-World现在只把world model当data generation engine——sample一堆trajectory然后筛选。但更powerful的用法是把world model当search space，在里面做MCTS或者planning。policy提一个候选action，world model预测结果，根据结果调整action。这才是"在imagination中plan"的真正实现。

**5. 跟Genie 3、Cosmos的关系**

Cosmos（https://arxiv.org/abs/2501.03575）和Genie 3（https://arxiv.org/abs/2504.07098）走的是"通用物理世界模拟器"路线，scope大但controllability弱。Ctrl-World走的是"specialized to robot manipulation with action conditioning"路线，scope窄但能真正close the loop with policy。

未来很可能merge：用Cosmos当backbone，再用Ctrl-World式的方法adapt到robot control。如果Cosmos的physics fidelity上来了，Ctrl-World的low-level execution gap问题就解决了。

**6. 潜在的scaling law**

paper里没明说但我猜：world model quality会随着video backbone scaling提升。现在用1.5B的SVD，如果换成Wan 14B（https://arxiv.org/abs/2503.20314）或者更大模型，FVD会进一步降，low-level execution accuracy会提升，policy improvement的数字可能更夸张。

## 相关链接汇总

- **Project page**: https://ctrl-world.github.io
- **DROID dataset**: https://droid-dataset.github.io
- **π0**: https://arxiv.org/abs/2410.24164
- **π0.5**: https://arxiv.org/abs/2504.16054
- **π0-FAST**: https://arxiv.org/abs/2501.09747
- **SVD**: https://arxiv.org/abs/2311.15127
- **IRASim**: https://arxiv.org/abs/2406.14540
- **WPE**: https://arxiv.org/abs/2506.00613
- **iVideoGPT**: https://arxiv.org/abs/2410.08909
- **Cosmos**: https://arxiv.org/abs/2501.03575
- **Genie 3**: https://arxiv.org/abs/2504.07098
- **Wan**: https://arxiv.org/abs/2503.20314
- **1X World Model**: https://www.1x.tech/1x-world-model.pdf
- **VGGT (multi-view transformer)**: https://arxiv.org/abs/2403.19551
- **EDM (diffusion formulation)**: https://arxiv.org/abs/2206.00364
- **VLM as success detector**: https://arxiv.org/abs/2303.07280
- **Unified World Models**: https://arxiv.org/abs/2504.02792

总结一句：这篇paper把"world model + policy closed loop"从理论可能性变成了实测44.7%提升的工程现实。我赌一年内我们会看到第一篇用world model做MCTS planning的robot论文。

---

# Ctrl-World: 面向Robot Manipulation的可控生成式World Model深度解析

这篇paper来自Stanford（Chelsea Finn组）和Tsinghua的合作工作，核心贡献是把一个passive的video diffusion model改造成一个可以与state-of-the-art VLA policy进行closed-loop交互的controllable world model，并能同时用于policy evaluation和policy improvement。Project page: https://ctrl-world.github.io

## 1. 动机：为什么需要Controllable World Model

VLA policy（如π0, π0.5, OpenVLA）在open-world场景下依然fragile。要让它们持续进步，需要两个反馈循环：

- **Evaluation feedback**：评估policy在unfamiliar objects/instructions下的表现。传统做法要大量real-world rollouts才能拿到statistically significant的结论（参考RoboArena: https://arxiv.org/abs/2506.18123）
- **Improvement feedback**：识别failure cases后需要corrective expert data来fine-tune

两者都很慢、很贵。World model的思路：把policy放到"imagination space"里rollout，从而把这两个反馈循环都搬进simulation。

但prior action-conditioned world models（IRASim: https://arxiv.org/abs/2406.14540, WPE: https://arxiv.org/abs/2506.00613, iVideoGPT等）有几个关键缺陷，导致它们无法真正与modern VLA policy闭环：

| 缺陷 | 具体表现 | 后果 |
|---|---|---|
| Single-view prediction | 只predict一个third-person camera | partial observability，导致物体"瞬移"进gripper的hallucination；且wrist view缺失，无法喂给需要multi-view输入的policy |
| Coarse action conditioning | 通常只在sequence level condition action | 无法捕捉高频action的causal effect，cm级精度丢失 |
| Short-horizon coherence | 没有显式memory机制 | 长horizon rollout中drift严重，10秒后画面degenerate |

Ctrl-World针对这三点提出三个对应的设计：multi-view joint prediction、frame-level action conditioning、pose-conditioned memory retrieval。

## 2. 问题形式化

先看formulation，这对理解整个pipeline至关重要。

**Policy**：modern VLA policy接收multi-view observation和language instruction，输出一个action chunk：
$$a_{t+1}, a_{t+2}, ..., a_{t+H} \sim \pi(\cdot | o_t, l) \quad (1)$$

其中observation $o_t = [I_t^1, ..., I_t^n, q_t]$，$I_t^i$是第i个camera view的image，$q_t$是robot pose（这里指joint configuration），$l$是language instruction，$H$是action horizon（DROID中$H=15$，对应1秒@15Hz）。

**World Model**：接收当前observation和action chunk，预测未来observation sequence：
$$o_{t+1}, ..., o_{t+H} \sim W(\cdot | o_t, A_t) \quad (2)$$

其中$A_t = [a_{t+1}, ..., a_{t+H}]$。

关键在于这是一个**autoregressive closed loop**：world model预测的$O_{t+H}$回喂给policy产生下一个$A_{t+H}$，如此循环。每个loop step对应1秒的action chunk，10个loop就是10秒rollout。这个closed loop特性是Ctrl-World与纯video generation最大的区别——视频生成是一次性的，world model必须是interactive的。

## 3. 方法详解

### 3.1 Architecture Overview

Backbone：Stable Video Diffusion (SVD, https://arxiv.org/abs/2311.15127) 1.5B参数，spatial-temporal transformer结构。

输入token shape（这是理解整个架构的关键）：
- VAE下采样率：$8\times 8$空间
- 每个camera: $192 \times 320$ → latent $24 \times 40$
- 3个views并行concatenate到token dimension: $3 \times 24 \times 40 = 2880$ tokens per frame
- 7个history frames + 5个future frames（15 actions下采样到5步）
- 总token shape: $B \times 12 \times 2880$

```
[history_0] [history_1] ... [history_6] [noised_future_0] ... [noised_future_4]
   ↓pose       ↓pose         ↓pose          ↓action^cart       ↓action^cart
   cross-attn  cross-attn    cross-attn     cross-attn         cross-attn
```

每个frame的pose/action embedding通过frame-wise cross-attention注入。这是和传统video diffusion（只在text level condition）的根本区别。

### 3.2 Multi-View Joint Prediction

设计选择：把N个camera views的latent在token维度concatenate（不是在spatial维度stack，也不是separate prediction）。

为什么concatenate而不是separate？因为joint prediction让attention能cross-view交互，建立3D一致性。引用的VGGT工作（https://arxiv.org/abs/2403.19551）证明feed-forward transformer能scalable地捕捉multi-view几何关系。

**关键insight**：wrist view的引入不只是为了兼容policy输入，更本质的作用是**消除contact时刻的hallucination**。原因：third-person view从外部看，gripper和object接触那一帧的视觉信息其实非常ambiguous（occlusion + 小contact area）；而wrist view提供了contact的高分辨率近景，让模型知道"是否真的接触上了"。Table 1显示wrist view加入后FVD从127.5降到97.4。

### 3.3 Pose-Conditioned Memory Retrieval

这是long-horizon consistency的核心机制。

**朴素做法的问题**：直接把所有past frames塞进context会导致context爆炸，且远处的frame对当前预测不一定relevant。

**Ctrl-World做法**：sparse sampling + pose anchoring。
- 在时间轴上以stride $m$采样$k=7$个history frames（实际$m$对应1-2秒间隔）
- 每个history frame $o_{t-km}, ..., o_t$都伴随对应的pose $q_{t-km}, ..., q_t$
- Pose通过frame-wise cross-attention注入到对应frame的visual tokens

为什么pose是好的anchor？因为robot manipulation中很多状态是周期性或可重复的——gripper在某个pose附近的视觉场景高度相似。当模型预测t=4s帧时，它可以通过attention机制找到历史上pose相近的frame（如t=0s），从那里retrieve视觉细节。Figure 4的attention visualization直接证实了这一点：预测t=4s时attention强集中在t=0s（pose相同的frame）。

这本质上是一种"内容寻址的memory"（content-addressable memory），类似Neural Turing Machine的read head，但是用pose作为key。

### 3.4 Frame-Level Action Conditioning

这是fine-grained controllability的关键。

**输入action的处理流程**（这是论文里写得比较省略，我从appendix和图推断的）：

1. Policy输出joint velocities $a_{t+1:t+H}^{jv}$
2. Adapter（2-layer MLP）将current joint config $q_t^{joint}$和joint velocities映射到future joint configurations $q_{t+1:t+H}^{joint}$
3. Forward Kinematics (FK)将joint configurations转换为Cartesian-space end-effector poses $q_{t+1:t+H}^{cartesian}$
4. 15个Cartesian poses下采样到5个（time stride）
5. 每个pose（7维：xyz + quaternion）通过3-layer MLP映射到1024维embedding
6. 通过frame-wise cross-attention注入到对应future frame

**为什么用Cartesian而不是joint space**？我推测：因为SVD backbone预训练时见过的是视觉信号，Cartesian pose的语义更接近"末端执行器在3D空间的位置"，和visual content的correlation更直接；而joint angles需要模型隐式学习FK，难度更大。

**为什么frame-level而不是sequence-level**？消融实验Table 2显示去掉frame-level conditioning后PSNR从23.56降到21.20，FVD从97.4涨到122.7。原因是sequence-level conditioning（如把action序列作为一个global token）无法让模型区分action chunk内不同时间步的causal effect。Frame-level让每个frame的visual tokens attend到它对应的那个action，建立精确的temporal alignment。

### 3.5 Training Objective

Diffusion loss，预测$x_0$而非$\epsilon$（v-prediction / x0-prediction style，参考EDM: https://arxiv.org/abs/2206.00364）：
$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t'} \|\hat{x}_0(x_{t'}, t', c) - x_0\|^2 \quad (3)$$

其中：
- $x_0 = o_{t+1:t+H}$是clean future observation
- $\epsilon \sim \mathcal{N}(0, I)$是Gaussian noise
- $t' \in [0, T']$是diffusion timestep
- $x_{t'} = \sqrt{\bar{\alpha}_{t'}} x_0 + \sqrt{1 - \bar{\alpha}_{t'}} \epsilon_{t'}$是noised version（$\bar{\alpha}_{t'}$是累积noise schedule）
- $c = [q_{t-km}, ..., q_t, a_{t+1:t+H}', o_{t-km}, ..., o_t]$是所有conditioning：history poses + future action-derived poses + history frames
- $\hat{x}_0$是模型对clean future的预测

**关键训练细节**：
- 7个history frames各自加独立random noise（不是同一个noise level），这是为了增强robustness——推理时history frames本身是model生成的，已经带distribution shift
- 只新初始化action projection MLP，其他参数从SVD继承。这是为了保留预训练video prior
- 95k trajectories, 564 scenes, 76k success + 19k failure trajectories
- 2×8 H100, batch size 64, 100k steps, 2-3天

**为什么要包含failure trajectories**？这是与一般video generation的重要区别。World model需要能模拟policy失败的场景，否则evaluation时无法区分success和failure。Failure数据让模型学到"action → 不好的outcome"的mapping。

## 4. 实验解析

### 4.1 World Model Quality (Table 1)

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | FVD↓ |
|---|---|---|---|---|---|
| WPE-Single-View | 20.33 | 0.772 | 0.131 | 25.50 | 156.4 |
| IRASim-Single-View | 21.36 | 0.774 | 0.117 | 26.46 | 138.1 |
| Ctrl-World-Single-View | 21.27 | 0.793 | 0.110 | 23.47 | 127.5 |
| **Ctrl-World (full)** | **23.56** | **0.828** | **0.091** | **25.00** | **97.4** |

256 clips，每clip 10秒（10轮autoregressive rollout）。FVD从IRASim的138.1降到97.4，是约30%的相对改进。Single-View版本就已经胜过baselines，multi-view进一步大幅提升。

值得注意FID反而略升（25.00 vs Ctrl-World-SV的23.47），这暗示multi-view joint prediction主要提升的是temporal consistency和cross-view consistency，而不是单帧画质——可能是attention capacity被分散到多个views。

### 4.2 Ablation (Table 2)

| 去掉的组件 | Third-view FVD | Wrist-view FVD |
|---|---|---|
| Full Ctrl-World | 97.4 | 127.1 |
| w/o memory | 105.5 | 133.1 |
| w/o frame-level cond | 122.7 | 179.1 |
| w/o joint pred | - | 158.1 |

Frame-level conditioning的影响最大（FVD +25.3）。这证实了fine-grained action control是controllability的bottleneck。

### 4.3 Controllability Visualization (Figure 4)

论文做了非常漂亮的controllability实验：给三个不同action chunks（Z轴±6cm, X轴±6cm），观察生成trajectory的差异。结果显示Ctrl-World能区分cm-level的action差异。Attention map显示预测t=4s帧时强attention到t=0s pose-similar的frame，直接证实memory retrieval的工作机制。

### 4.4 Policy Evaluation (Table 3)

跨7个tasks（Pick-Place, Fold-Towel, Drawer, Wipe-table, Close-laptop, Pull-tissue, Stack）评估3个policy（π0, π0-FAST, π0.5）。

**Key finding**：World model的instruction following ranking与real world高度correlated。例如Drawer任务：π0 (0.05) < π0-FAST (0.20) < π0.5 (0.80)，world model也给出π0 (0.00) < π0-FAST (0.20) < π0.5 (0.65)的相同排序。

**Gap分析**：World model在instruction following上略underestimate（high-level意图capture得准），但在low-level execution success rate上明显underestimate（如Close-laptop π0.5: real 0.70 vs WM 0.05）。论文诊断原因是：
1. 复杂physics（碰撞、滑动、旋转）建模不精确
2. Policy在real world失败后会retry，但world model rollout中这种retry behavior没被capture
3. DROID中failure distribution不覆盖所有policy failure modes

**这给了我们一个重要intuition**：world model的fidelity上限受训练数据中failure mode coverage的限制。要让world model成为可靠的evaluator，需要注入更多policy rollout data（这正是1X World Model的思路：https://www.1x.tech/1x-world-model.pdf）。

### 4.5 Policy Improvement (Table 4-7)

这是paper最impressive的结果。

**Pipeline**：
1. 准备downstream tasks with novel instructions/objects
2. 在world model中rollout 400 trajectories per task
3. 通过instruction rephrasing + initial state perturbation增加diversity
4. Human preference判断保留25-50个successful trajectories
5. SFT π0.5-DROID 2k steps

**Results**：
- Spatial Understanding: 28.75% → 87.5%（+58.75%！）
- Shape Understanding: 43.75% → 91.25%
- Towel folding with direction: 57.5% → 80%
- Novel objects (glove, stapler): 25% → 75%

平均提升**44.7%**（38.7% → 83.4%）。

**为什么synthetic data能这么有效**？我的理解：
- π0.5本身有strong prior，只是对specific instruction format或object appearance没align
- Synthetic trajectories提供的是"对齐数据"而非"新技能数据"——把high-level capability unlock到specific downstream distribution
- 这类似于LLM的instruction tuning：base model已经有能力，只是需要少量示范来shape output distribution

## 5. 核心Intuition总结

把这篇paper放在更大的图景中理解：

### 5.1 World Model作为Policy的"Imagination"

人类可以在脑海中"simulate"动作的后果来plan。Ctrl-World本质上是给robot policy装上了一个imagination engine。关键不是这个imagination有多physical accurate，而是它能否：
1. 保持visual consistency（multi-view + memory解决）
2. 尊重action的causal effect（frame-level conditioning解决）
3. 与policy的I/O format兼容（multi-view + wrist camera解决）

### 5.2 Pretrained Video Prior的价值

SVD提供了强大的visual prior（textures, motions, object affordances），Ctrl-World只需轻量adaptation就能加上controllability。这呼应了"video model = weak world model"的thesis（参考Sora, Cosmos: https://arxiv.org/abs/2501.03575）。但paper也证实了纯video prior不够——需要explicit action conditioning和memory才能成为interactive world model。

### 5.3 Memory = Temporal Credit Assignment

Pose-conditioned memory retrieval本质上解决的是"temporal credit assignment"问题：长horizon rollout中，当前帧的细节应该从哪个历史frame retrieve？用pose作为key是一种structured prior——manipulation中pose是scene state的强surrogate。这比纯temporal attention（让模型自己学会哪frame重要）更sample efficient。

### 5.4 Failure Mode Coverage Matters

DROID的19k failure trajectories让model学到"什么action会导致失败"。这是world model能用于evaluation的前提。如果只用success data训练，model会变成"乐观模拟器"——预测所有action都成功，无法区分policy好坏。

### 5.5 与Genie 3、Cosmos等Foundation World Model的关系

Genie 3 (https://arxiv.org/abs/2504.07098) 和 Cosmos 等大型foundation world model走的是"通用物理世界模拟"路线。Ctrl-World走的是"specialized to robot manipulation with action conditioning"路线。前者ambition更大但controllability弱，后者scope窄但能直接close the loop with policy。两者未来可能merge：用Cosmos做预训练backbone，再用Ctrl-World式的方法adapt到robot control。

## 6. 局限与未来方向

Paper诚实承认的局限：
1. 精确physics（碰撞、滑动）建模不足
2. 对initial observation敏感
3. 长horizon reasoning仍fail
4. 只improve instruction following，不improve low-level success on seen tasks

我会补充几点：
- **Reward model bottleneck**：当前用human preference，scalability受限。需要VLM-based reward model（参考https://arxiv.org/abs/2303.07280）
- **Distribution shift compounding**：world model生成的frame回喂policy，policy再生成action喂回world model——这是两个model相互compound error的过程。10秒能维持，20秒、30秒呢？
- **Exploration**：当前用instruction rephrasing和random init增加diversity，但这是heuristic。理论上应该用world model做planning（MCTS in imagination），而不只是random sample + filter。

## 7. 相关工作链接

- **DROID dataset**: https://droid-dataset.github.io
- **π0**: https://arxiv.org/abs/2410.24164
- **π0.5**: https://arxiv.org/abs/2504.16054
- **π0-FAST**: https://arxiv.org/abs/2501.09747
- **SVD**: https://arxiv.org/abs/2311.15127
- **IRASim**: https://arxiv.org/abs/2406.14540
- **WPE**: https://arxiv.org/abs/2506.00613
- **iVideoGPT**: https://arxiv.org/abs/2410.08909
- **Cosmos**: https://arxiv.org/abs/2501.03575
- **Genie 3**: https://arxiv.org/abs/2504.07098
- **OpenVLA**: https://arxiv.org/abs/2406.09246
- **Unified World Models (Zhu et al.)**: https://arxiv.org/abs/2504.02792
- **VGGT (multi-view transformer)**: https://arxiv.org/abs/2403.19551
- **EDM (diffusion formulation)**: https://arxiv.org/abs/2206.00364
- **Dreamiate (video generation for policy)**: https://arxiv.org/abs/2406.16862
- **Gen2Act**: https://arxiv.org/abs/2409.16283
- **1X World Model**: https://www.1x.tech/1x-world-model.pdf
- **RoboArena**: https://arxiv.org/abs/2506.18123
- **VLM as success detector**: https://arxiv.org/abs/2303.07280
- **Project page**: https://ctrl-world.github.io

整体而言，Ctrl-World是robot world model领域一个重要的milestone：它第一次完整展示了"world model + policy"的closed loop可以同时用于evaluation和improvement，并且用44.7%的improvement数字证明了synthetic data from world model的实际价值。下一步最有想象力的方向是把world model当作planning的search space（如MCTS in imagination），而不只是data generation engine——那样才是真正实现"在imagination中学习"的vision。
