---
source_pdf: RynnVLA-001.pdf
paper_sha256: 5f5090fc4b9db015fc5eea7f3fa9c5f71cd4f2dc471150396cbe27105d93e78d
processed_at: '2026-08-12T02:32:48-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 RynnVLA-001

---

## 问题就一句话

**Robot data 太贵，采几百万条 teleoperation trajectory 要把人累死。但 LLM 有 trillion tokens，VLM 有 billions image-text pairs，VLA 只能喝汤。**

那怎么办？看看天上有没有馅饼掉下来。

---

## 天上真的掉馅饼了

你打开 YouTube/抖音，满屏都是"第一视角拍手在做事"的视频——切菜、开抽屉、拼乐高、用牙刷牙清理通风口（paper 里 Fig.4 真的放了这个例子）。

**这些 ego-centric hand manipulation video，在视觉上跟机械臂 gripper 操作几乎一模一样**：都是看到一只"手"在镜头前面摆弄物体。而且网上这种视频有海量，Ego4D、EPIC-KITCHENS、HowTo100M 加起来随便筛筛就 12M 条。

这就是免费的 manipulation prior，不用白不用。

---

## 但有个尴尬的 gap

视频只有画面，没有 action label。你没法直接拿视频训 VLA，因为 VLA 要的是 `(image, instruction) → action`，而视频给不了 action。

RynnVLA 的核心 contribution 就是**怎么把这个 gap 桥接过去**，用三步走：

---

## 三步走，每一步 narrow 一点 gap

### Step 1：先让模型学会"脑补"

给它一张图 + 一句话（"用牙刷清理通风口"），让它预测接下来 7 帧画面。

这一步模型学到的是：**物体怎么动、手怎么动、东西被抓起来会怎样**。说白了就是物理世界的 "manipulation dynamics"，只不过全部编码在 visual representation 里。

这一步用 12M ego-centric 视频训，backbone 是 Chameleon（Meta 的 AR 多模态生成模型）改成 I2V。

**比喻**：就像让一个人看几万小时做菜视频，他虽然没下过厨，但脑子里的"画面感"已经知道刀怎么切、油怎么溅了。

### Step 2：让模型知道"手要怎么动"

光会脑补画面还不够——VLA 要的是 action，不是 pixel。中间还差一层。

于是加了 EgoDex 数据（Apple Vision Pro 抓的人手腕轨迹），让模型**同时预测画面 + 手腕关键点的运动轨迹**。

手腕轨迹 ≈ end-effector trajectory，这就是从"纯视觉"到"带动作"的中间踏板。模型开始理解"画面变化 ↔ 手怎么动"的因果关系。

**比喻**：光看做菜视频不够，还得加上"厨师的手在哪个位置"的标注，这样你不仅知道画面怎么变，还知道是什么动作导致的画面变。

### Step 3：换上真机械臂数据

前两步学的都是"人手"的 prior，这一步用 SO100 机械臂的 800 条 teleoperation 数据 fine-tune。

关键发现：**前面学的 representation 直接 transfer 过来超有用**。从 scratch 训只有 4% 成功率，经过 Step1+Step2 pretrain 后直接到 90%。

**比喻**：一个看过几万小时做菜视频、还研究过厨师手部轨迹的人，现在给他换一套不同的工具（机械臂 vs 人手），他只需要花几百次练习就能上手，因为"操作感"已经内化在脑子里了。

---

## ActionVAE：动作的"ZIP 文件"

还有一个关键 trick。VLA 预测一段动作（chunk）比预测单步好——单步容易"卡住"，模型反复输出同一个 action 不动。

但直接预测一段高维 action sequence 不好训。于是搞了个 ActionVAE：

- **Encoder**：把一段动作序列压成一个低维 latent vector（比如 32 维）
- **Decoder**：从 latent 还原回完整动作序列

Policy 只需要预测这个 32 维向量，decoder 再展开成可执行的动作。

**比喻**：就像 ZIP 文件。你不用记住 100 个动作的细节，只需要决定"要哪个 ZIP"，解压后自动是完整的一套动作。而且 latent space 是平滑的 Gaussian，policy 在上面 regression 特别稳。

---

## 最让我 excited 的发现

**视频生成的画面其实很 subtle**——Fig.4 里生成的未来帧跟输入帧差别很小，肉眼几乎看不出明显运动。

但这个"看似啥也没生成"的 backbone，给 VLA 用之后成功率从 50% 飙到 84%。

**这说明 video pretrain 的价值根本不在"生成质量"，而在 latent representation 里编码的物理直觉。** 模型不用生成多好看的画面，它只要"懂"画面应该怎么变，这个"懂"就是 transferable 的 manipulation prior。

这跟 LLM 里 next-token prediction 训完做 downstream task 一个道理——生成质量好不好不重要，representation 学到了才重要。

---

## 其他几个有意思的小发现

1. **Action head 越简单越好**：single linear layer 比 5 层 MLP 好。Transformer 输出已经够 expressive，加复杂 head 反而过拟合。

2. **Inference 时不生成视频**：训练时同时预测画面和 action，但部署时**只预测 action，丢弃画面生成**。因为生成视频慢且对 control 没用。砍掉后 inference 快很多，closed-loop control 才可行。

3. **Front camera 管"粗定位"，wrist camera 管"微调"**：关掉 front camera 后，目标在 wrist 视野外就直接失败，在视野内还能凑合。说明 multi-view 不是冗余备份，是功能分工。

4. **Distractor 训练必须有**：只在干净场景训的模型，遇到 distractor 直接 0% 成功（会抓错东西）。必须在训练数据里就放 distractor，模型才能 follow language instruction 而不是被视觉显著性带跑。

---

## 一句话总结

**网上有海量人手操作视频，视觉上跟机械臂操作高度相似，把它当成免费的 manipulation prior 蒸馏进 VLA，通过"视觉预测→手部轨迹预测→机械臂 action"的 curriculum 桥接 gap，用 ActionVAE 把动作序列压成低维 latent 让 policy 好学，inference 时砍掉视频生成只出 action——结果比 Pi0 和 GR00T N1.5 高出 20-35 个百分点。**

核心 insight：**robot data 贵，但 human video 便宜且视觉相似，关键是设计好从"看人做"到"自己做"的桥接 curriculum。**

---

# RynnVLA-001: 用 Human Demonstrations 改进 Robot Manipulation 的 VLA Model 详解

Andrej, 这篇 paper 我看了之后挺 excited 的，思路非常清晰，是 VLA 领域里 video pretraining 路线的一篇 representative 工作。下面我尽量把它的核心 idea、架构 trick、和实验逻辑给你梳理出来。

---

## 1. Core Motivation: 为什么 VLA 需要 Video Pretraining

VLA 模型最大的 bottleneck 是 paired (vision, language, action) data 太少。LLM 能 web-crawl 几 T tokens，VLM 也能找到 billions 的 image-text pairs，但 robot manipulation data 几乎只能 teleoperation 一条一条采，Open X-Embodiment 这种 large-scale dataset 也才百万级 trajectory（[Open X-Embodiment paper](https://arxiv.org/abs/2310.08864)）。

RynnVLA-001 的核心 insight 是：**ego-centric human hand manipulation video** 在视觉层面和 robot gripper manipulation 高度相似（first-person 视角 + 手部/gripper 操作），可以从 web 上 crawl 到 12M 级别的量，作为 pretraining 数据把 "manipulation dynamics prior" 蒸馏到模型里。这个 idea 跟 GR-2（[GR-2 paper](https://arxiv.org/abs/2410.06158)）、VPP（[VPP paper](https://arxiv.org/abs/2412.14803)）思路一脉相承，但 RynnVLA 多加了一个**中间 stage** 把 visual prediction 显式接到 action prediction，这是它的主要 contribution。

---

## 2. 三阶段训练 Pipeline 总览

```
Stage 1: Ego-Centric Video Generative Pretraining
   ↓ (predict future frames)
Stage 2: Human-Centric Trajectory-Aware Video Modeling
   ↓ (predict future frames + human keypoint trajectories)
Stage 3: Robot-Centric Vision-Language-Action Modeling
   ↓ (predict future frames + robot action embeddings via ActionVAE)
```

Stage 1 学 "视觉动态"，Stage 2 引入 "动作信号"（human wrist keypoints 作为 proxy action），Stage 3 再迁移到真实 robot action space。这种 curriculum 设计的核心 intuition：**纯 visual prediction 离 action 太远，直接跳过去 gap 太大，需要一个中间媒介**，而 human wrist trajectory 在 ego-centric 视角下接近 end-effector trajectory，是天然的桥梁。

---

## 3. Stage 1: Ego-Centric I2V Pretraining 细节

### 3.1 数据 Curation Pipeline

从 Ego4D（[Ego4D](https://arxiv.org/abs/2110.07058)）、EPIC-KITCHENS（[EPIC-KITCHENS](https://arxiv.org/abs/2008.01176)）、HowTo100M（[HowTo100M](https://arxiv.org/abs/1906.03327)）、Something-Something（[SSv2](https://arxiv.org/abs/1706.04230)）等数据集出发，pipeline 三步：

1. **Keypoint Detection**: 用 pose estimation model（应该是基于 DWPose / WholeBody，[DWPose](https://arxiv.org/abs/2305.17106)）抽 face/torso/hand keypoints。
2. **Ego-centric Filtering**:
   - **No facial keypoints** → 丢弃（出现脸说明是 third-person 视角）
   - **Presence of hand keypoints** → 保留（手在镜头附近 = ego-centric）
3. **Text Annotation**: 用 Qwen2-VL-7B（[Qwen2-VL](https://arxiv.org/abs/2409.12191)）生成短描述，模拟 robot task instruction（如 "put the bottle in the box"）。

最终得到 12M ego-centric manipulation videos + 244K robotic manipulation videos（来自 Open X-Embodiment / DROID / BridgeData V2 / OXE）。

### 3.2 I2V 架构

基于 **Chameleon**（[Chameleon paper](https://arxiv.org/abs/2405.09818)）这个 AR multimodal foundation model 扩展。Chameleon 本来是 mixed-modal early-fusion 的 image generator，RynnVLA 把它扩展成 I2V。

输入序列结构：
```
[language tokens, visual tokens_t, language tokens, visual tokens_{t+1}, ...]
```

注意一个细节：language tokens 是 interleaved 在每个 timestep 的 visual tokens 之前，而不是只在 sequence 开头给一次。这是为了 mirror VLA inference 时 "每个 action prediction 都 condition on 当前 visual + language" 的 pattern。

Loss 是 cross-entropy over discrete visual tokens（VQGAN tokenizer）+ language tokens：
$$
\mathcal{L}_{\text{Stage1}} = -\sum_t \Big[ \log p(\mathbf{v}_t \mid \mathbf{l}_{<t}, \mathbf{v}_{<t}) + \log p(\mathbf{l}_t \mid \mathbf{l}_{<t}, \mathbf{v}_{<t}) \Big]
$$
其中 $\mathbf{v}_t$ 是 timestep $t$ 的 visual token sequence，$\mathbf{l}_t$ 是对应的 language tokens。

视觉 tokenizer 用的是 VQGAN，pretrain 在 512×512 上，所以 paper 里 ablation 显示降到 256×256 性能崩（reconstruction fidelity 不够），最后用 384×384 折中。

---

## 4. Stage 2: Human-Centric Trajectory-Aware Modeling

### 4.1 关键设计

这一 stage 用 **EgoDex**（[EgoDex paper](https://arxiv.org/abs/2505.11709)）数据集，它是用 Apple Vision Pro 抓的 upper-body joints 轨迹。RynnVLA 只取 wrist keypoints（近似 end-effector position）。

输入序列变成：
$$
[\text{language}, \mathbf{v}_t, \mathbf{s}_t, \langle\text{ACTION\_PLACEHOLDER}\rangle, \dots]
$$
其中 $\mathbf{s}_t$ 是 current wrist keypoint position 的 state embedding（linear projection 到 transformer hidden dim），$\langle\text{ACTION\_PLACEHOLDER}\rangle$ 是 action generation 的 signal token。

### 4.2 Action Head 与 Loss

Transformer 主干仍然输出 discrete visual tokens（cross-entropy），但加一个 **lightweight action head**（single linear layer）把 $\langle\text{ACTION\_PLACEHOLDER}\rangle$ 位置的 last hidden state 映射到 continuous action embedding space。

Loss:
$$
\mathcal{L}_{\text{Stage2}} = \mathcal{L}_{\text{visual CE}} + \lambda \cdot \|\hat{\mathbf{z}}_a - \mathbf{z}_a\|_1
$$
其中 $\hat{\mathbf{z}}_a$ 是 predicted action embedding，$\mathbf{z}_a$ 是 ActionVAE encoder 给的 ground truth latent，L1 loss 只在 $\langle\text{ACTION\_PLACEHOLDER}\rangle$ 位置算。

注意一个重要 ablation：**deeper action head (5-layer MLP) 反而比 single linear layer 差**（Avg.Len 从 4.019 掉到 3.323）。这说明 transformer 输出 representation 已经足够 expressive，加 deep head 反而 overfit / 引入 noise。这跟 LLM 里 "single linear head 最 robust" 的经验一致。

---

## 5. ActionVAE: Action Representation 的核心模块

### 5.1 为什么用 VAE 而不是直接 predict raw actions

VLA 里 predict action chunk（短动作序列）比 predict single-step action 好的两个原因（这跟 ACT / Diffusion Policy 一致，[ACT paper](https://arxiv.org/abs/2304.13705)）：
1. Single-step 容易导致 "stuck"——视觉变化太小，模型重复输出同一个 action。
2. Chunk prediction 一次 forward 出多个 action，减少 inference latency。

但直接 predict 原始 action chunk 的问题：output space 高维，temporal coherence 难保证。ActionVAE 用 VAE 把 action chunk 压缩成一个 compact latent vector，policy 只需 predict 这个 vector，由 decoder 重建出完整 action sequence。

### 5.2 VAE 公式（paper 没明写，但标准 VAE 推断）

设 action chunk $\mathbf{a}_{1:T} = (a_1, a_2, \dots, a_T)$，T 是 chunk length。

Encoder:
$$
\mathbf{z}_a = \mu_\phi(\mathbf{a}_{1:T}) + \sigma_\phi(\mathbf{a}_{1:T}) \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)
$$
其中 $\mu_\phi, \sigma_\phi$ 是 encoder 输出的均值和方差，$\odot$ 是 element-wise product，$\boldsymbol{\epsilon}$ 是 reparameterization trick 用的标准 Gaussian noise。

Decoder: $\hat{\mathbf{a}}_{1:T} = D_\theta(\mathbf{z}_a)$

Training loss:
$$
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi} \Big[ \|\mathbf{a}_{1:T} - \hat{\mathbf{a}}_{1:T}\|_1 \Big] + \beta \cdot D_{\text{KL}}\Big( q_\phi(\mathbf{z}_a \mid \mathbf{a}_{1:T}) \,\|\, \mathcal{N}(0, I) \Big)
$$
$\beta$ 是 KL weight（β-VAE 风格），KL 项约束 latent space 接近标准 Gaussian，这让 policy 在 latent space 上的 regression 更平滑。

### 5.3 两个 domain-specific ActionVAE

因为 human wrist trajectory 和 robot arm kinematics 空间不一样，paper 训练了**两个独立**的 ActionVAE：
- Human trajectory ActionVAE（Stage 2 用）
- Robot action ActionVAE（Stage 3 用）

这个设计挺 important——VAE 是 embodiment-specific 的，但一旦训好就直接 freeze，可以对新数据 extract latent，不用 retrain。

Ablation Table 5 里 "Raw Actions Prediction" vs "Full Model"（用 VAE latent）在 Calvin ABC→D 上：
- Task 5 success rate: 67.0% vs 72.1%
- Avg.Len: 4.019 vs 4.161

VAE 带来 ~5% 的提升，主要来自 latent space 的 temporal consistency 和 compression。

---

## 6. Stage 3: Robot-Centric VLA Modeling

### 6.1 架构改动

- 继承 Stage 2 的 transformer weights（visual prediction 部分）
- **丢弃** Stage 2 的 action head（human 和 robot kinematics 差太多，重头训）
- 新 init 一个 single linear layer 作为 robot action head
- 视觉输入从 single-view 变成 **two-view**（front camera + wrist camera）

输入序列：
$$
[\text{language}, \mathbf{v}_t^{\text{front}}, \mathbf{v}_t^{\text{wrist}}, \mathbf{s}_t^{\text{robot}}, \langle\text{ACTION\_PLACEHOLDER}\rangle, \dots]
$$

### 6.2 训练 Loss

两个 concurrent objective：
1. **Robot Action Prediction**: L1 loss between predicted embedding 和 ActionVAE 编码的 ground truth embedding
2. **Future Visual Prediction**: 继续做 cross-entropy visual token prediction（auxiliary regularization）

$$
\mathcal{L}_{\text{Stage3}} = \mathcal{L}_{\text{visual CE}} + \lambda_a \cdot \|\hat{\mathbf{z}}_a^{\text{robot}} - \mathbf{z}_a^{\text{robot}}\|_1
$$

Visual prediction 在这里**只作 training 时的 regularizer**，inference 时直接 discard（下一节讲）。

### 6.3 Inference 时的关键 trick

训练时同时 predict visual tokens 和 action embedding，但 **inference 时只 predict action embedding，跳过 visual token generation**。这是个非常 practical 的工程优化——visual AR generation 慢且对 control 没必要，去掉后 inference speed 大幅提升，closed-loop control 才 feasible。

Inference loop：
1. 模型接收 language + 当前 RGB（front + wrist）+ robot state
2. 只生成 action embedding（一次 forward）
3. ActionVAE decoder 解出 action chunk
4. Robot 执行整个 chunk
5. 新 observation 进来，回到 step 1

---

## 7. 实验分析

### 7.1 Dataset & Tasks

自己采的数据集，用 **LeRobot SO100** 机械臂（[LeRobot repo](https://github.com/huggingface/lerobot)），3 个 task：

| Task | Demos 数 | 关键 challenge |
|---|---|---|
| Pick up and place green blocks | 248 | 基础 recognition + grasping |
| Pick up and place strawberries | 249 | precise localization + grasping point |
| Grab pen and put into holder | 301 | 3D spatial reasoning + orientation 推断 |

数据用 3 台不同的 SO100 在不同环境、不同光照下采，有 distractor objects 增加复杂度。

### 7.2 Main Result（Table 1）

| Method | Green Blocks | Strawberries | Pen Holder | Average | SR@1 |
|---|---|---|---|---|---|
| GR00T N1.5 | 65.0 | 53.3 | 48.3 | 55.6 | 37.2 |
| Pi0 | 75.6 | 71.1 | 64.4 | 70.4 | 56.3 |
| **RynnVLA-001** | **90.0** | **91.7** | **90.0** | **90.6** | 56.7 |

RynnVLA-001 平均 90.6%，比 Pi0 高 20 个百分点，比 GR00T N1.5 高 35 个点，这个 gap 非常大。SR@1（一次 trial 成功率）跟 Pi0 持平，说明 single-trial precision 还有提升空间。

### 7.3 Distractor Robustness（Table 2）

| Method | Single-Target | Multi-Target | w/ Distractors |
|---|---|---|---|
| GR00T N1.5 | 63.3 | 46.7 | 56.7 |
| Pi0 | 80.0 | 71.1 | 60.0 |
| RynnVLA-001 | 93.3 | 86.7 | 91.7 |

Pi0 在 distractor 场景下从 80% 掉到 60%，而 RynnVLA-001 几乎不掉（93.3 → 91.7）。这非常说明问题——video pretraining 给模型更强的 visual grounding，让它能 follow language instruction 而不是被 visual saliency 带跑。

### 7.4 Pretraining Ablation（Table 3, 4）

这是最 important 的 ablation，验证两 stage pretraining 各自贡献：

| Variant | Average | SR@1 |
|---|---|---|
| Scratch (随机初始化) | 4.4 | 0 |
| Chameleon (T2I 预训练) | 50.0 | 22.8 |
| Video (Stage 1) | 84.4 | 49.4 |
| **Full (Stage 1+2)** | **90.6** | **56.7** |

几个关键观察：
1. **Scratch 基本废了**——说明 small robot dataset（~800 demos）完全不够 from-scratch VLA 训练
2. **Chameleon T2I pretrain 已经能拿到 50%**——纯图像生成 prior 对 grasping 有帮助
3. **Stage 1 video pretrain 把 50% 拉到 84.4%**——ego-centric video 的 manipulation dynamics prior 极其有效
4. **Stage 2 trajectory-aware 再加 6.2%**——human trajectory 作为 visual→action 的中间桥梁确实有 marginal 但 consistent 的提升

这个 ablation 给的 intuition：**video generation pretraining 的核心价值不在 "生成质量"，而在 "manipulation dynamics prior"**。即使生成的视频 subtle visual change（Fig. 4 显示），这个 latent representation 已经编码了大量 manipulation 知识。

### 7.5 Camera Functionality 分析（Section 5.5）

这部分非常 insightful，做了两个 controlled experiment：

**实验 1: Front camera 的 coarse localization 作用**
- 关掉 front camera，只用 wrist camera
- 目标在 wrist camera 初始视野内 → 还能完成（成功率 4/5）
- 目标在 wrist camera 视野外 → 完全失败（0%）

这说明 front camera 提供 "coarse approach" 信号，wrist camera 负责 "fine local adjustment"。

**实验 2: Front camera 的 3D projective 信息**
- 把 front camera 抬高，改变 projective geometry
- 抓笔任务从成功 → 失败

说明模型确实依赖特定视角的 3D 几何信息做 spatial reasoning，不是纯 2D 视觉 policy。

这两个实验给 future VLA 设计的启示：**multi-view 不是冗余，每个 view 都有 specific functional role**。

---

## 8. 跟相关工作的关系梳理

### 8.1 VLA 谱系

VLA 大致两条路线：
1. **Discretize action 成 token，用 LLM 直接 AR predict**：RT-2（[RT-2](https://arxiv.org/abs/2307.15818)）、OpenVLA（[OpenVLA](https://arxiv.org/abs/2406.09246)）、FAST（[FAST](https://arxiv.org/abs/2501.09747)）、VQ-VLA（[VQ-VLA](https://arxiv.org/abs/2507.01016)）。优点：复用 LLM 生态；缺点：precision loss。
2. **Dual-system：VLM 做 reasoning + 专门 policy head 做 continuous action**：LCB（[LCB](https://arxiv.org/abs/2410.11788)）、Pi0（[Pi0](https://arxiv.org/abs/2410.24164)）、GR00T N1.5（[GR00T N1.5](https://research.nvidia.com/labs/gear/gr00t-n1_5/)）、CogACT（[CogACT](https://arxiv.org/abs/2411.19650)）。优点：continuous action 表达力强。

RynnVLA 走第二条但更激进——backbone 本身就是 video generator（不是 VLM），action head 是 simple linear regressor 到 VAE latent。

### 8.2 Future Prediction in Robot Learning

三 paradigm：
1. **Explicit future state as goal**: SuSIE（[SuSIE](https://arxiv.org/abs/2310.10639)）、UniPi（[UniPi](https://arxiv.org/abs/2302.00448)）、GEVRM、DREAMGEN
2. **Joint future + action**: PAD、WorldVLA（[WorldVLA](https://arxiv.org/abs/2506.21539)）、CoT-VLA
3. **Future prediction as pretraining only**: GR-2、VPP、**RynnVLA**

RynnVLA 跟 GR-2 最大的区别：GR-2 直接 video pretrain 后跳到 action，RynnVLA 多了 Stage 2 的 trajectory-aware 中间步，explicitly bridge visual 和 action。

### 8.3 Ego-Centric 数据思路

跟 EgoVid-5M（[EgoVid-5M](https://arxiv.org/abs/2411.08380)）思路类似，都是从 web ego-centric video 蒸 robot 知识。EgoDex（[EgoDex](https://arxiv.org/abs/2505.11709)）提供 trajectory annotation 是关键 enabler。

---

## 9. 我的 Intuition 总结

读完这篇 paper 我有几个 takeaway：

**1. Pretraining curriculum 比 pretraining data 量更 important**
12M video 不算特别大，但通过 Stage 1（visual）→ Stage 2（trajectory）→ Stage 3（action）的 curriculum，每一步都 narrow the gap，最终效果远超 Pi0 / GR00T。这跟 LLM 里 "pretraining task 必须和 downstream task 接近" 的经验一致。

**2. ActionVAE 是关键 enabler**
把高维 action chunk 压成 low-dim latent，让 transformer 在 latent space 上做 regression。这跟 VQ-VLA 的 VQ-VAE 思路互补——VQ 是 discrete，VAE 是 continuous。continuous 在 dexterous task 上应该更有优势。

**3. Simplicity wins**
- Action head 用 single linear layer 比 5-layer MLP 好
- Inference 时 discard visual prediction，只保留 action prediction
这些都是 "less is more" 的体现。Transformer backbone 已经学了足够 representation，加 complex head 反而 hurt。

**4. Video pretraining 的价值在 representation，不在 generation**
生成的视频视觉上变化 subtle（Fig. 4），但作为 backbone 给 VLA 用，性能从 50% 飙到 84%。这说明模型学到的是 manipulation 的 "latent dynamics"，不是 "pixel-level rendering"。

**5. Multi-view 有 functional 分工**
Front camera = coarse localization + 3D geometry，Wrist camera = fine adjustment。设计 camera setup 时不能只看 coverage，要看 functional role。

**6. Limitation 坦诚**
Paper 自己承认：只在 SO100 上测，环境跟 training 接近，camera 固定。Generalization 到其他 embodiment / 环境 / 视角是 future work。这是 VLA 领域普遍 limitation，OpenHelix（[OpenHelix](https://arxiv.org/abs/2505.03912)）、SmolVLA（[SmolVLA](https://arxiv.org/abs/2506.01844)）也面临同样问题。

---

## 10. 可能的 Extension 方向

基于这篇 paper 的思路，我想到几个可能的方向：

1. **Cross-embodiment ActionVAE**: 现在是 embodiment-specific，如果训一个 universal ActionVAE 能 encode 不同 robot 的 action，可以做 zero-shot embodiment transfer。这跟 Open X-Embodiment 的 RT-X 思路结合。

2. **Hierarchical ActionVAE**: 现在 chunk 是 flat sequence，如果加 hierarchical structure（如 "approach → grasp → place" 三阶段），可能 better long-horizon task。

3. **3D-aware visual token**: 现在 visual token 是 2D VQGAN，如果换成 3D Gaussian Splatting token 或 NeRF feature，可能更强 3D spatial reasoning（对应 Section 5.5 的 camera 实验观察）。

4. **Reinforcement fine-tuning on ActionVAE latent**: 现在 policy 是 imitation learning，如果在 ActionVAE latent space 上做 RL（类似 RLHF 的 PPO），可能突破 imitation 的 performance ceiling。

5. **Web-scale data scaling**: 12M video 已经不少，但如果能 scale 到 100M+（用 SAM 2 [SAM 2](https://arxiv.org/abs/2408.00714) 做 segmentation 辅助 filter），可能 emergent ability。

---

## Key References

- [Chameleon (Meta)](https://arxiv.org/abs/2405.09818) - AR multimodal foundation
- [Pi0 (Physical Intelligence)](https://arxiv.org/abs/2410.24164) - flow matching VLA
- [GR00T N1.5 (NVIDIA)](https://research.nvidia.com/labs/gear/gr00t-n1_5/) - humanoid VLA
- [GR-2](https://arxiv.org/abs/2410.06158) - 早期 video pretrain VLA
- [EgoDex](https://arxiv.org/abs/2505.11709) - Apple Vision Pro hand trajectory
- [Ego4D](https://arxiv.org/abs/2110.07058) - ego-centric video benchmark
- [EgoVid-5M](https://arxiv.org/abs/2411.08380) - ego-centric video-action dataset
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864) - cross-embodiment dataset
- [OpenVLA](https://arxiv.org/abs/2406.09246) - open-source VLA
- [FAST](https://arxiv.org/abs/2501.09747) - action tokenizer
- [VQ-VLA](https://arxiv.org/abs/2507.01016) - VQ action tokenizer
- [WorldVLA](https://arxiv.org/abs/2506.21539) - AR world model VLA
- [CogACT](https://arxiv.org/abs/2411.19650) - foundational VLA
- [LeRobot](https://github.com/huggingface/lerobot) - SO100 arm platform
- [Qwen2-VL](https://arxiv.org/abs/2409.12191) - text annotation
- [ACT](https://arxiv.org/abs/2304.13705) - action chunk transformer 原始 paper
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) - action chunk 概念
- [Calvin Benchmark](https://arxiv.org/abs/2112.03227) - long-horizon manipulation benchmark
- [OpenHelix](https://arxiv.org/abs/2505.03912) - dual-system VLA survey
- [SmolVLA](https://arxiv.org/abs/2506.01844) - affordable VLA
- [SAM 2](https://arxiv.org/abs/2408.00714) - segmentation
- [Lumina-mGPT](https://arxiv.org/abs/2408.02657) - 代码基础
- [SuSIE](https://arxiv.org/abs/2310.10639) / [UniPi](https://arxiv.org/abs/2302.00448) - future state as goal
- [VPP](https://arxiv.org/abs/2412.14803) - video prediction policy
- [DROID](https://arxiv.org/abs/2403.12945) - large-scale robot dataset
- [RT-2](https://arxiv.org/abs/2307.15818) - VLA 开山之作
- [VAR (Visual Autoregressive)](https://arxiv.org/abs/2404.02905) - next-scale prediction

---

整体上这篇 paper 是 VLA + video pretraining 路线一个很 solid 的 milestone，curriculum 设计 + ActionVAE + inference-time 简化这三个 design choice 配合得很好。如果想 repro，代码在 [GitHub](https://github.com/alibaba-damo-academy/RynnVLA-001)。期待 Alibaba 后续在更多 embodiment 和 long-horizon task 上的 follow-up。
