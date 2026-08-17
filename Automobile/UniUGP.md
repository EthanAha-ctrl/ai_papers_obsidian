---
source_pdf: UniUGP.pdf
paper_sha256: 614f4274f1f445f836765770e30ff3cbbf27bf4606b0222495668ceb2c36663a
processed_at: '2026-08-12T20:13:15-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# UniUGP 用人话讲：到底在干啥

## 一句话概括

**把"看懂路况"+"想象未来"+"开车"三件事塞进一个模型里，让它们互相教对方。**

---

## 为啥要干这事

先讲个背景。现在自动驾驶有两条路线，各玩各的：

**路线 A：VLA（Vision-Language-Action）**
就是拿 GPT-4o 这种大模型看图说话，看完摄像头画面直接输出方向盘怎么打。优点是它有"世界知识"——见过雪知道雪滑，见过施工知道要减速。缺点是它**光会说不练**，从没真正"想象"过未来画面长啥样，所以遇到没见过的奇葩场景就懵。

**路线 B：World Model**
就是让模型预测下一帧视频长啥样，通过"脑补未来"来学物理规律。优点是它真懂 dynamics（车往前开，前方物体会变大）。缺点是它**没文化**——没有 LLM 的 reasoning 能力，不会用语言解释为什么这么开，也不会听人话。

UniUGP 的 insight 就一句话：**这俩天生应该在一起**。World model 给 VLA 补上"想象能力"，VLA 给 world model 补上"推理能力"和"听人话能力"。

---

## 怎么塞进一个模型的：Hybrid Expert 架构

这里有个设计选择。两种方案：

**方案 1：全共享（像 [Transfusion](https://arxiv.org/abs/2408.11039)）**
所有参数都共用，text token 和 image token 在同一个 transformer 里一起跑。

**方案 2：混合专家（MoT，UniUGP 选的）**
attention 层共享（让 text 和 action 能"互相看"），但 FFN 分开（各算各的）。

为啥选方案 2？因为 **next-token prediction 和 flow matching 是两种完全不同的学习任务**。一个是学离散概率分布，一个是回归连续向量场。硬塞一个 FFN 里会打架，gradient 互相干扰。分开 FFN 就像给两个学生不同的作业本，但让他们坐同一张桌子（attention 共享）能互相抄答案。

类比一下：就像你学钢琴和学画画，老师是同一个（attention 层提供的跨模态信息），但练琴和练画的手部肌肉训练（FFN）得分开。

---

## 三个 Expert 各干啥

### Understanding Expert（Qwen2.5-VL-3B 干的活）
- **输入**：摄像头画面 + 语言指令
- **输出**：CoT reasoning（"前方有施工，需减速"）+ LM logits
- **核心**：next-token prediction，跟普通 VLM 一样

### Planning Expert
- **输入**：历史轨迹 $s$ + 加了噪声的 action $a_\tau$
- **输出**：去噪的 velocity field $u_\tau^{plan}$
- **核心**：flow matching，从噪声"流"向真实 trajectory

公式核心：
$$a_\tau = \tau a + (1-\tau)\epsilon$$

这里 $\tau$ 是 timestep（0 到 1），$\tau=1$ 是干净 action，$\tau=0$ 是纯噪声。训练时模型学一个 vector field 把 $a_\tau$ 拉回 $a$。

为啥用 flow matching 不用 DDPM？因为 flow matching 是 deterministic ODE，几步就能生成，AD 推理要快。

### Generation Expert（Wan2.1 DiT 干的活）
- **输入**：历史画面 + 噪声未来画面 + Understanding 的 hidden states + Planning 的 action embedding
- **输出**：未来几秒视频
- **核心**：DiT blocks 做 video diffusion

这个 expert 是 **cascaded** 在前两个后面的，不是并行。好处是可以单独关掉——部署在手机上资源不够就把 video generation 关掉，understanding + planning 照常跑。

---

## 关键 Insight：Generation 反过来教 Understanding 看

这是 paper 最有意思的发现。看 Table 3 的 ablation：

- 完整模型：Small object 识别 89.3%
- 去掉 generation expert：Small object 识别 83.7%（掉 5.6 个点）

**为啥去掉"想象未来"会掉识别精度？**

Intuition 是这样的：模型要生成未来几秒视频，就得想清楚"远处那个小点未来会变成什么"。如果远处是个施工锥，未来画面里它会变大、逼近。Generation loss 会惩罚"画错未来"，这个 loss 反向 propagate 回 understanding expert，**逼它关注远处的小物体**。

这就是 paper 反复强调的 "world model forces VLA to learn visual causal inference, particularly focusing on distant objects"。

类比：你光看一张照片可能忽略远处的小点，但让你"想象 3 秒后画面"，你被迫去想"那小点是啥、会不会撞过来"。Generation 就是逼模型做这种想象。

---

## 4 阶段训练为啥这么设计

这个 curriculum 设计挺讲究：

**Stage 1**：只训 Understanding，1M steps
用 long-tail 数据 + ImpromptuVLA 80K clips。先让 VLM 把 driving domain 的 perception 基础打好。Planning 和 Gen 先别来捣乱。

**Stage 2**：冻结 Understanding，训 Planning + Generation，4M steps
用 nuScenes、Waymo 等带 trajectory 的数据。Why 冻结 understanding？因为它要作为 stable condition 让 plan/gen 去适应。如果一起训，understanding 的 representation 一直在变，plan/gen 像在追移动靶。

**Stage 3**：只训 Understanding 学 CoT，1M steps
用自己标的 CoT 数据（GPT-4o 生成 + 人工校正）。这步把"为什么这么开"的语言解释能力加进去。CoT prompt 强制模型只从 current image + history 推理，不准偷看 future trajectory。

**Stage 4**：三个一起训，4M steps，数据比例 0.1:0.4:0.5
前面三步各专家都 aligned 了，但跨专家的 alignment 还没建。这步用混合数据 fine-tune，loss 权重 $\alpha=0.3, \beta=0.5, \gamma=0.2$——planning 占大头因为最终目标还是开车开得准。

为啥不一开始就 joint 训？因为 **multi-task 早期会互相干扰**。VLM 预训练的 reasoning 能力会被 noisy 的 planning/generation signal 破坏。先分别 build up，再 fuse，类似 [STaR](https://arxiv.org/abs/2203.14465) 的 curriculum 思路。

---

## 数据集的巧思

Paper 自己构造了 long-tail benchmark，这是个 contribution。用了 6 个数据集：
- [DADA2000](https://ieeexplore.ieee.org/document/9666805)：事故场景司机注意力
- [Lost and Found](https://arxiv.org/abs/1609.04653)：小障碍物
- [StreetHazards](https://arxiv.org/abs/2107.05009)：OOD 物体
- [Waymo-E2E](https://arxiv.org/abs/2510.26125)：4021 个长尾 segments
- 等等

设计 4 类 QA：
1. **Perception**：有没有小物体？会不会出事？
2. **CoT Reasoning**：用 4 步推理（场景分析→关键物体→意图推断→行动原因）
3. **Planning**：预测未来轨迹
4. **Instruction Following**：给定指令（左转/直行），生成对应轨迹

Prompt 设计有个 trick：用 **多种 paraphrase** 问同一个问题（"Any small long-tailed objects?" / "Are there tiny long-tailed items?" / "Does the video have small long-tailed things?"）。避免模型记住单一 question template，逼它真理解语义。

---

## 实验结果的人话解读

### 长尾 benchmark（Table 3）

3B 参数的 UniUGP 把 GPT-4o 和 Qwen2.5-VL-72B 都打不过它。GPT-4o 在 small object recognition 只有 64.2%，UniUGP 是 89.3%。说明**专门训 driving domain + generation supervision 比单纯堆参数有效**。

Ablation 两个关键发现：
- 去掉 CoT：Small 从 89.3%→86.5%。CoT 不只是给人看的，**它反过来帮 perception**——因为 reasoning 让模型显式思考"为什么这是 small object"。
- 去掉 Generation：Small 从 89.3%→83.7%。印证前面的 insight，generation 是 perception 的 supervisor。

### nuScenes Planning（Table 4）

UniUGP 只用**前视单摄像头**，avg L2 = 1.23m，collision rate = 0.33%。

对比 [UniAD](https://arxiv.org/abs/2212.10156) 用**全车摄像头**才 L2=1.03m, collision=0.31%。UniUGP 用 1/6 的摄像头数量基本追平。Collision rate 0.33% 比 [Doe-1](https://arxiv.org/abs/2412.09627) 的 0.53% 降低 37%，说明模型真懂了 lane keeping 和 gap maintenance 这些基本规则。

### 视频生成（Table 5）

FID=7.4，FVD=75.9，都是 SOTA。关键是**继承 Wan2.1 预训练权重**——之前方法都从头训 video generation，UniUGP 直接用几十亿参数的 Wan2.1 做 generation expert，photorealistic 先验白嫖过来。

Figure 4 展示了 trajectory-controllable generation：改 trajectory 输入，生成的视频里车就按新轨迹走。证明 generation expert 真的 condition 在 action 上了，不是瞎生成。

---

## Figure 3 的 Ablation 最直观

对比 UniUGP vs UniUGP w/o Gen 的 CoT 输出：

**UniUGP（完整）**：
"前方施工区，有工人和锥桶。这是潜在危险区域，需减速确保安全。过去轨迹显示直线行驶，当前画面明确显示施工区。因此车辆可能减速避免碰撞，遵守道路施工安全规定。"

**UniUGP w/o Gen（去掉想象）**：
"这是较宽的城市道路，车少。驾驶时应观察路况，确保没有行人或车辆突然闯入。保持在车道内，注意交通标线，按规定速度平稳行驶..."

差别一目了然：完整模型**具体指向远处的施工区**，w/o Gen 就泛泛而谈"注意路况"。这就是 generation expert 教 understanding expert "看远处"的直接证据。

---

## 这篇 paper 真正的贡献

1. **架构上**：MoT 让 VLA 和 World Model 在每层都 cross-modal aligned，不是简单 cascade。参考 [π0](https://arxiv.org/abs/2410.24164) 和 [F1](https://arxiv.org/abs/2509.06951) 的 manipulation 工作思路迁移到 AD。

2. **Insight 上**：用 generation loss 反向 regularize perception，让 VLM 关注 future causal relationship。这是 [World Models](https://arxiv.org/abs/1803.10122)（Ha & Schmidhuber 2018）"generation is perception"思想在 VLA 时代的 revival。

3. **工程上**：Cascade 架构让 generation expert 可关闭，mobile deployment 友好。4-stage curriculum 避免 multi-task interference。

4. **Benchmark 上**：构造了 long-tail driving benchmark，填补了之前只测 structured scene 的空白。

---

## 我觉得还能往哪走

Paper 在 Appendix C 诚实列了 limitations，我补充我的思考：

1. **CoT 和 action 的 alignment 还不够紧**。Paper 自己承认"interpretability-action inconsistency"在复杂交互场景存在。意思是有时 CoT 说得头头是道，action 却没按说的做。这是 LLM-style reasoning 的根本问题——语言空间和物理空间没完全对齐。可能需要 [contrastive learning](https://arxiv.org/abs/2103.00020) 或 RL 把 CoT 和 action 拉紧。

2. **没用 closed-loop 评估**。当前是 open-loop L2 + collision rate，真正考验是 [nuPlan closed-loop](https://arxiv.org/abs/2106.11810) 或 [CARLA](https://carla.org/)。Open-loop L2 好不一定 closed-loop 好，这是 [Tian et al. 2024](https://arxiv.org/abs/2406.16842) 指出的经典问题。

3. **Generation expert 太贵**。Paper 说 mobile 要关掉。那能不能用 [Latent Diffusion](https://arxiv.org/abs/2112.10752) 或 [JEPA](https://openreview.net/forum?id=4PCBOO5IIi) 在 latent space 做 generation，省 pixel-level cost？LeCun 的路线在 efficiency 上确实有优势。

4. **只用 camera，没 LiDAR**。[HERMES](https://arxiv.org/abs/2501.14729) 用 LiDAR unified model，UniUGP 缺这个 modality。3D geometry 信息对 planning 很关键，纯 camera 在夜间/雨天会退化。

5. **CoT 数据靠 GPT-4o 生成**。能不能用 large-scale unlabeled video 自动 mine causal structure？类似 [VideoMAE](https://arxiv.org/abs/2203.12602) + causal discovery，scale up reasoning 数据不用人工标。

6. **Multi-agent reasoning 缺位**。当前是 ego-centric，但长尾场景往往是多车交互（如无信号灯路口）。可以参考 [SMARTS](https://arxiv.org/abs/2110.05914) 风格的 multi-agent reasoning。

---

## 最核心的 Takeaway

如果只记一件事，记这个：

**让模型"想象未来"（generation）反过来逼它"看懂现在"（perception），同时"推理为什么"（CoT）又反过来强化"看懂"和"想象"。三者形成正反馈三角，在长尾场景上比任何单一范式都 robust。**

这就是 UniUGP 名字的含义：Understanding, Generation, Planning 三合一。Generation 是粘合剂——它既是 World Model 的输出，又是 Understanding 的 supervisor，还是 Planning 的 visual validator。

架构上的 MoT 让三者 dense coupled（每层都交互），cascade design 让 generation 可关（部署友好），4-stage curriculum 让训练稳定（避免 early interference）。

工程细节上，flow matching 而非 DDPM 是为了推理速度，Wan2.1 预训练是为了 photorealistic 先验，CoT prompt 的 anti-cheat 设计是为了逼真推理而非复述答案。

这些细节加起来，让一个 3B 模型在长尾 driving 上打败了 72B 的通用 VLM。

Reference:
- [UniUGP Project Page](https://seed-uniugp.github.io/)
- [π0 VLA Flow Model](https://arxiv.org/abs/2410.24164)
- [F1 Robot VLA](https://arxiv.org/abs/2509.06951)
- [Epona World Model](https://arxiv.org/abs/2506.24113)
- [Flow Matching Paper](https://arxiv.org/abs/2210.02747)
- [Wan2.1 Video Model](https://arxiv.org/abs/2503.20314)
- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923)
- [Impromptu VLA](https://arxiv.org/abs/2505.23757)
- [Waymo-E2E Long-tail](https://arxiv.org/abs/2510.26125)
- [Transfusion Unified Model](https://arxiv.org/abs/2408.11039)
- [World Models (Ha & Schmidhuber)](https://arxiv.org/abs/1803.10122)
- [V-JEPA / I-JEPA](https://openreview.net/forum?id=4PCBOO5IIi)
- [DriveLM Benchmark](https://arxiv.org/abs/2305.06147)
- [nuScenes Dataset](https://arxiv.org/abs/1903.11027)
- [nuPlan Closed-loop Benchmark](https://arxiv.org/abs/2106.11810)
- [Doe-1 Unified Model](https://arxiv.org/abs/2412.09627)
- [ReCogDrive](https://arxiv.org/abs/2506.08052)
- [AutoVLA](https://arxiv.org/abs/2506.13757)
- [AlphaDrive RL Reasoning](https://arxiv.org/abs/2503.07608)
- [HERMES LiDAR Unified](https://arxiv.org/abs/2501.14729)
- [DADA2000](https://ieeexplore.ieee.org/document/9666805)
- [Lost and Found](https://arxiv.org/abs/1609.04653)
- [StreetHazards](https://arxiv.org/abs/2107.05009)
- [VideoMAE](https://arxiv.org/abs/2203.12602)
- [CARLA Simulator](https://carla.org/)
- [SMARTS Multi-agent](https://arxiv.org/abs/2110.05914)
- [Latent Diffusion](https://arxiv.org/abs/2112.10752)
- [Switch Transformer](https://arxiv.org/abs/2101.03961)
- [STaR Curriculum](https://arxiv.org/abs/2203.14465)
- [DriveDreamer](https://arxiv.org/abs/2309.09777)
- [OccWorld](https://arxiv.org/abs/2410.10559)
- [GAIA-1](https://arxiv.org/abs/2309.17080)

---

# UniUGP 深度技术解读：Unifying Understanding, Generation, Planning for End-to-End Autonomous Driving

## 1. Big Picture 与 Core Motivation

这篇 ByteDance Seed 的工作核心 insight 在于：**现有的 AD 范式存在二分法的割裂**。VLA 模型（如 [Impromptu VLA](https://arxiv.org/abs/2505.23757)、[AutoVLA](https://arxiv.org/abs/2506.13757)、[ReCogDrive](https://arxiv.org/abs/2506.08052)）能利用预训练 VLM 的世界知识和 reasoning，但无法吃下海量 unlabeled driving video 来学 visual causal reasoning；World Model（如 [Epona](https://arxiv.org/abs/2506.24113)、[OccWorld](https://arxiv.org/abs/2410.10559)、[GAIA-1](https://arxiv.org/abs/2309.17080)）能从 next-frame prediction 学 dynamics，但缺乏 LLM 的 reasoning 和 interaction 能力。

UniUGP 想做的事情是**把这两条路线通过 hybrid expert 架构统一**，让三个能力在一个模型里相互 regularize：
- **Understanding Expert**（VLM 路线）：next-token prediction，输出 CoT reasoning
- **Planning Expert**（VLA 路线）：flow matching，输出 continuous trajectory
- **Generation Expert**（World Model 路线）：DiT-based video generation，输出 future video

关键的 intuition 在于：generation expert 提供 visual causal supervision（逼迫 understanding expert 关注远距离物体和未来因果关系），而 understanding expert 提供语义 reasoning 给 generation expert 做 condition，planning expert 提供 action embedding 给 generation expert 做 controllability。三者形成一个**三角形的相互正则化结构**。

参考 web links:
- [UniUGP Project Page](https://seed-uniugp.github.io/)
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
- [Wan2.1 Video Generation Model](https://arxiv.org/abs/2503.20314)

---

## 2. Hybrid Expert 架构深度解析

### 2.1 为什么是 Hybrid 而非 Fully Unified

这里要区分两种 unified 架构：
- **Joint/early-fusion unified**（如 [Transfusion](https://arxiv.org/abs/2408.11039)、[Janus](https://arxiv.org/abs/2504.06256)、[Show-o](https://arxiv.org/abs/2408.12528)）：把 next-token 和 diffusion 装进同一个 transformer，共享所有参数
- **Hybrid/MoT**（如 [π0](https://arxiv.org/abs/2410.24164)、[F1](https://arxiv.org/abs/2509.06951)、UniUGP）：不同模态走不同 attention 流和 FFN，共享 attention 计算但解耦表示

UniUGP 选择 MoT 是因为：autoregressive text token 和 flow matching 的 continuous action token **学习动态完全不同**。Next-token 是离散概率分布学习，flow matching 是 vector field regression，把它们塞进同一个 FFN 会引起 gradient conflict。MoT 让 attention 共享（捕获跨模态 alignment）但 FFN 分离（保留模态特异性表示）。

### 2.2 MoT Layer 数学拆解

公式 (2) 描述了 MoT layer 的核心：

$$h_o^{und}, h_o^{plan} = \text{MSHA}([\text{QKV}_{und}(x^{und}), \text{QKV}_{plan}(x^{plan})])$$

变量解释：
- $x^{und}$：understanding tokens，来自 text tokenizer + ViT encoder 的 cross-modal aligned representation
- $x^{plan}$：planning tokens，由历史状态 $s$ 和 noised action $a_\tau$ 通过 `Proj.` 投影得到
- $\text{QKV}_{und}$ 和 $\text{QKV}_{plan}$：模态独立的 linear projection，分别将各自 token 映射到 Q、K、V
- $\text{MSHA}$：Multi-Head Self-Attention，关键在于**两个模态的 QKV 是 concatenation 进同一个 attention**，这意味着 understanding token 的 query 可以 attend 到 planning token 的 key/value，反之亦然——这就是跨模态信息流的核心。

Intuition：想象 attention matrix 是一个 block matrix：
```
         |  und K  | plan K |
---------|---------|--------|
und Q    |  A_uu   |  A_up  |  ← understanding 关注自己的语义 + planning 的动作
---------|---------|--------|
plan Q   |  A_pu   |  A_pp  |  ← planning 关注 understanding 的语义 + 自己的动作
```
$A_{up}$ 和 $A_{pu}$ 这两个 off-diagonal block 是关键——它们让 action token 能"看到"语义理解，让 text token 能"感知到"action 的 noise level。这就是为什么 MoT 比 dual-tower 好：cross-modal alignment 在每一层都被 refresh。

然后公式 (3)(4) 是**模态独立的 FFN**：
$$h_{ffn}^{und} = \text{FFN}_{und}(h_o^{und})$$
$$h_{ffn}^{plan} = \text{FFN}_{plan}(h_o^{plan})$$

FFN 是参数大头，分开后两个模态的参数效率更高，类似 [Switch Transformer](https://arxiv.org/abs/2101.03961) 的思路但更细分。

### 2.3 Generation Expert 为什么是 Cascaded 而非 Joint

公式 (9) 是关键：
$$u_\tau^{gen} = \mathcal{W}([v^{hist}, v_\tau^{fut}], [h^{und}, A], \tau)$$

变量：
- $v^{hist}$：历史图像 VAE-encoded tokens
- $v_\tau^{fut}$：noised future image tokens
- $h^{und}$：understanding expert 输出的 hidden states（语义 condition）
- $A$：action embedding（物理 condition）
- $\tau$：timestep embedding
- $\mathcal{W}$：Wan2.1 DiT blocks

注意 generation expert 是**串行级联**在 understanding/planning experts 后面的，不是 MoT 的并行结构。这个设计选择有几个原因：

1. **计算效率**：DiT 的参数量（Wan2.1 是几十亿级别）远大于 Qwen2.5-VL-3B 的 MoT 部分，让它们 joint 训练会很贵
2. **信息流向是单向的**：generation 需要 semantic + action 作为 condition，但 understanding 不需要 future video 作为输入（inference 时倒过来，但训练时已经用 ground truth）
3. **Mobile deployment 友好**：paper 明确说"On mobile devices, the generation expert can be disabled"——这是 cascade 架构的 critical advantage

公式 (10) 是一个有意思的训练 trick：
$$A \sim \begin{cases} \text{Proj.}(a) & \text{if rand}(0\sim1) > 0.5 \\ \text{Proj.}(a_\tau - (1-\tau)u_\tau^{plan}) & \text{else} \end{cases}$$

意思是：50% 概率用 ground truth action 作为 condition，50% 概率用 planning expert 的 single-step denoised action。这是 scheduled sampling 的一种形式，解决 **exposure bias**——inference 时 planning expert 不可能输出 ground truth，所以 generation expert 必须适应 planning expert 的实际输出分布。

---

## 3. Flow Matching 数学拆解

公式 (1) 是 flow matching 的 forward process：
$$a_\tau = \tau a + (1-\tau)\epsilon$$

变量：
- $a$：ground truth action chunk（未来 K 步的 trajectory points）
- $\epsilon \sim \mathcal{N}(0, I)$：随机高斯噪声
- $\tau \in [0, 1]$：timestep，$\tau=1$ 时是 clean action，$\tau=0$ 时是 pure noise
- $a_\tau$：noised action

注意这里 flow matching 的 formulation 跟 [Lipman et al. 2022](https://arxiv.org/abs/2210.02723) 的标准 form 一致：线性插值路径。这跟 DDPM 的马尔可夫噪声过程不同，是 deterministic flow，推理时用 Euler ODE solver 而非 ancestral sampling，少步数就能生成。

公式 (8) 是 planning expert 的训练目标：
$$\mathcal{L}_{plan} = \mathbb{E}_{u_\tau^{plan}}[||u_\tau^{plan} - (\epsilon - a)||_2]$$

变量：
- $u_\tau^{plan}$：模型预测的 vector field
- $\epsilon - a$：target vector field（从 $a_\tau$ 指向 $a$ 的方向）

为什么 target 是 $\epsilon - a$ 而不是 $a - \epsilon$？因为 flow matching 学的是 velocity field $da_\tau/d\tau = a - \epsilon$（从 noise 流向 data），但这里训练的是 reverse process 用的 vector field，方向取反。这个 sign convention 跟 [π0](https://arxiv.org/abs/2410.24164) 是一致的。

公式 (11) 是 generation expert 的 loss，结构完全一样：
$$\mathcal{L}_{gen} = \mathbb{E}_{u_\tau^{gen}}[||u_\tau^{gen} - (\epsilon - v^{fut})||_2]$$

只是 action 换成 future video tokens。

---

## 4. 四阶段训练策略深度剖析

Table 2 给出了训练 hyperparameter。让我把这个 4-stage 训练策略的**因果链条**讲清楚：

### Stage 1: Understanding Expert 单独训练
- Trained Components: 只 Und. Expert
- Dataset: Custom long-tail + ImpromptuVLA（80K clips）
- 1M steps, batch 64, lr 1e-4
- Resolution: 224×224

**Motivation**：Qwen2.5-VL-3B 虽然预训练了，但 driving domain 的长尾场景分布外，需要先建立 baseline perception。这一步不让 planning/generation 参与，是为了避免**多任务干扰**——早期 multitask 训练会让 VLM 因为 noisy 的 planning/generation 信号而破坏预训练的 reasoning 能力。

### Stage 2: Planning + Generation 联合训练
- Trained Components: Gen. Expert + Plan. Expert（Und. Expert 冻结）
- Dataset: nuScenes, NuPlan, Waymo, Lyft, Cosmos
- 4M steps
- Generation Resolution: 512×512

**Motivation**：Und. Expert 冻结意味着它的 hidden states $h^{und}$ 作为 stable condition，让 Plan/Gen 专家去适应它。这里 Plan 和 Gen 一起训是因为它们都依赖 visual dynamics——trajectory prediction 和 video generation 本质上都在学 future state distribution，可以相互 transfer（multi-task positive transfer）。

### Stage 3: CoT Reasoning 训练
- 只训 Und. Expert
- 1M steps
- Custom CoT dataset

**Motivation**：这一步把 Stage 1 学到的 perception + Stage 2 间接获得的 dynamics（通过 frozen hidden states）注入 CoT 推理能力。CoT dataset 是用 advanced VLM + future planning results + manual calibration 构造的，确保 reasoning 跟未来行为 causal aligned。

### Stage 4: 三专家联合训练
- 三个都训
- 数据比例 0.1 : 0.4 : 0.5（Stage1数据 : Stage2数据 : Stage3数据）
- 4M steps

**Motivation**：前面三个 stage 让每个 expert 都独立 aligned 了，但**cross-expert alignment 还没建立**。Stage 4 用 mixed data 联合 fine-tune，解决 modality gap。loss 权重 $\alpha=0.3, \beta=0.5, \gamma=0.2$ 偏重 planning（β最大）说明最终目标是 planning 性能，understanding 和 generation 都是辅助。

公式 (12)：
$$\mathcal{L}_{total} = 0.3 \cdot \mathcal{L}_{und} + 0.5 \cdot \mathcal{L}_{plan} + 0.2 \cdot \mathcal{L}_{gen}$$

这个权重设计跟 [F1](https://arxiv.org/abs/2509.06951) 的 MoT 训练有类似 philosophy——manipulation 任务中 action loss 也占大头。

---

## 5. Dataset Construction Pipeline 细节

### 5.1 Long-tail 数据来源

Paper 收集了 6 个数据集做 long-tail：
- [Waymo-E2E](https://arxiv.org/abs/2510.26125)：4021 segments，专门 curated 长尾事件（<0.003% daily driving）
- [DADA2000](https://ieeexplore.ieee.org/document/9666805)：driver attention in accident scenarios
- [Lost and Found](https://arxiv.org/abs/1609.04653)：small road hazards detection
- [StreetHazards](https://arxiv.org/abs/2107.05009)：out-of-distribution object detection
- [SOM](https://ieeexplore.ieee.org/document/9345367)：LiDAR-guided small obstacle segmentation
- [AADV](https://link.springer.com/chapter/10.1007/978-3-319-54190-7_9)：dashcam accident anticipation

### 5.2 四类 QA 任务设计

Listing 1-3 展示了 prompt 工程，关键的 trick 是**用多种 paraphrase** 增强 generalization：
- "Any small long-tailed objects in the driving video?"
- "Are there tiny long-tailed items in the driving clip?"
- "Does the driving video have small long-tailed things?"

这种 prompt augmentation 避免模型记住单一 question template。

Listing 4 的 CoT prompt 是 4-step structured reasoning：
1. Scene Analysis（traffic lights, road geometry, lane markings）
2. Key Object Identification（最多 3 个关键/rare object）
3. Intention Inference（ego + 其他 agent 的未来行为推断）
4. Action Reason（brief action command + 详细 reasoning chain）

注意 prompt 中明确说 "your reasoning process should not rely on the future trajectory as the basis for your reasoning"——这是 anti-cheat mechanism，防止 VLM 直接从 future trajectory 复述答案，强迫它从 current image + historical trajectory 推理。

---

## 6. 实验结果深度分析

### 6.1 Table 3: 长尾 Benchmark

| Model | Small | Relationship | Abnor.Pred. | GPT | Blue | Planning L2(3s) | Following L2(3s) |
|-------|-------|--------------|-------------|-----|------|-----------------|-------------------|
| GPT-4o | 64.2% | 63.5% | 72.8% | 0.55 | 0.125 | 2.63 | 2.58 |
| Qwen2.5-VL-72B | 75.8% | 74.9% | 81.5% | 0.72 | 0.188 | 1.94 | 1.89 |
| Our w/o CoT | 86.5% | 85.7% | 93.2% | 0.83 | 0.218 | 1.58 | 1.53 |
| Our w/o Gen. | 83.7% | 82.9% | 90.6% | 0.80 | 0.203 | 1.72 | 1.67 |
| **Our** | **89.3%** | **88.6%** | **95.8%** | **0.88** | **0.240** | **1.45** | **1.40** |

关键 ablation insight：
- **w/o CoT**：去掉 CoT 后 Small 从 89.3%→86.5%，证明 CoT reasoning 反过来帮助 perception（reasoning 让模型显式思考"为什么这是 small object"）
- **w/o Gen.**：去掉 generation expert 后所有指标都掉，特别是 Small 从 89.3%→83.7%（掉 5.6 个点），证明 **generation expert 作为 visual causal supervision 对 perception 有显著贡献**——这印证了 paper 的核心 claim

3B 模型打败 GPT-4o 和 Qwen2.5-VL-72B 是有意思的 result。GPT-4o 在 Small object recognition 上只有 64.2%，说明通用 VLM 对 driving 长尾确实 weak。

### 6.2 Table 4: nuScenes Planning

UniUGP: avg L2 = 1.23m, avg Collision = 0.33%

跟 [Doe-1](https://arxiv.org/abs/2412.09627)（Camera* + QA, L2=1.26m, Collision=0.53%）对比，Collision rate 从 0.53% → 0.33% 是相对降低 37.7%——这是显著的安全提升。Collision rate 低说明模型理解了 basic traffic rules（如 lane keeping, gap maintenance）。

跟 [Epona](https://arxiv.org/abs/2506.24113)（Camera* + None, L2=1.25m, Collision=0.36%）对比，说明即使没有 explicit QA supervision，UniUGP 的统一架构也能 match 专门做 world model 的方法。

注意 [UniAD](https://arxiv.org/abs/2212.10156) 用 full camera suite 才达到 L2=1.03m, Collision=0.31%，UniUGP 只用 front camera（Camera*）能到 1.23m/0.33%，gap 不大。

### 6.3 Table 5: 视频生成质量

| Method | FID↓ | FVD↓ |
|--------|------|------|
| DriveDreamer | 52.6 | 452.0 |
| Drive-WM | 15.8 | 122.7 |
| GenAD | 15.4 | 184.0 |
| GEM | 10.5 | - |
| Doe-1 | 15.9 | - |
| Epona | 7.5 | 82.8 |
| FSDrive | 10.1 | - |
| **UniUGP** | **7.4** | **75.9** |

FID 7.4 是 SOTA，FVD 75.9 也最好。**用预训练 Wan2.1 的 generation expert 是 key**——之前方法大多从头训 video generation，UniUGP 继承了 Wan2.1 的 photorealistic 先验。

### 6.4 Table 6: DriveLM GVQA

Final Score 0.59，超过 FSDrive (0.57) 和 OmniDrive (0.56)。BLEU=0.78, ROUGE=0.76, Match=0.41 都是 SOTA。

---

## 7. Figure 3 的 Ablation 启示

Figure 3 对比了 UniUGP vs UniUGP w/o Gen 的 CoT 输出：
- **UniUGP**：聚焦于"construction area, workers, cone markers, vehicle speed needs to be reduced"——注意到远处的施工区，因果预测未来需要减速
- **w/o Gen**：泛泛地谈"observe road conditions, stay within lane, drive at prescribed speed"——没有具体因果指向

这印证 paper 的核心论断：**world model 强制 VLA 关注 future causal relationship，特别是 distant objects 的语义**。这是为什么 ablation 后 Small object recognition 掉 5.6 个点的原因——generation expert 通过"如果远处有 small object，未来 video 会出现它，从而影响 visual generation loss"的链路，反向 propagate 到 understanding expert，逼迫它 attend to 远处。

这其实是 **visual representation learning by generation** 的延伸思想，类似 [World Models](https://arxiv.org/abs/1803.10122)（Ha & Schmidhuber 2018）的原始 motivation：generation is perception。

---

## 8. 我的直觉性思考与联想

### 8.1 跟 LeCun 的 H-JEPA 的对比

UniUGP 的"generation as perception supervision" 跟 [LeCun 的 V-JEPA / I-JEPA](https://openreview.net/forum?id=4PCBOO5IIi) 思想是 sibling，但有关键差异：
- **JEPA**：在 latent space 做 predictive learning，避免 pixel-level generation 的 cost
- **UniUGP**：直接做 pixel-level video generation，因为 AD 需要可解释性（未来 video 是 interpretable evidence）

JEPA 是 efficiency-first，UniUGP 是 interpretability-first。Paper 在 limitation 里也承认"generation expert demands excessive resources"——这其实是 pixel-level generation 的 inherent cost。

### 8.2 跟 F1 Robot Manipulation 的对比

[F1](https://arxiv.org/abs/2509.06951) 用三个 expert（perception + action + generation）做 manipulation，思路高度类似。两者都用 generation 作为 inverse dynamics model 的 visual foresight。差异：
- **F1**：goal-conditioned video generation（未来有 goal image 锚定）
- **UniUGP**：action-conditioned video generation（action 是 trajectory waypoints）

AD 没有 explicit goal image，只能用 action 作为 controllability signal，所以 UniUGP 的 generation expert 必须跟 planning expert 紧密耦合（通过公式 10 的 action embedding）。

### 8.3 为什么不直接做 VLM + Diffusion Policy 的 Two-Tower

[ReCogDrive](https://arxiv.org/abs/2506.08052) 和 [DiffVLA](https://arxiv.org/abs/2505.19381) 是 VLM 输出 semantic features 给 diffusion policy 的 two-tower 设计。UniUGP 用 MoT 把它们 weave 在每一层，这种 dense coupling 的好处：
- VLM 的 hidden states 在每层都被 action token "grounded"
- Diffusion policy 的 noise level 信息能流回 VLM，让 VLM 知道当前推理的 action 处在哪个 denoising stage

Two-tower 是单向 cascade，MoT 是 bidirectional cross-attention，在 long-tail 场景的 reasoning 上更 robust。

### 8.4 Action Token 的 Continuous vs Discrete 取舍

Table 1 显示 UniUGP 用 continuous action（flow matching），而 [AutoVLA](https://arxiv.org/abs/2506.13757) 用 codebook 离散化。Continuous 的优势是 trajectory 本身是 continuous signal，离散化会引入 quantization error（特别是低速精细控制）。Discrete 的优势是可以复用 LLM 的 next-token prediction machinery，不需要单独的 flow matching head。

UniUGP 选 continuous 是因为它的 target 是 long-tail 场景下需要精细 trajectory 调整（如绕开 cone），discrete codebook 的 resolution 会限制。

### 8.5 跟 WALL-OSS 的对比

[WALL-OSS](https://arxiv.org/abs/2509.11766) 用 tightly coupled MoE 做 manipulation，跟 UniUGP 的 MoT 思路一致。但 WALL-OSS 强调"discrete action priors + continuous control"，UniUGP 完全用 continuous（flow matching）。这意味着 UniUGP 放弃了 LLM 的 discrete action token prior，赌的是 continuous flow matching 在 AD 这种 high-frequency control 上更优。

### 8.6 Stage 4 数据比例 0.1:0.4:0.5 的 intuition

这个比例意味着 Stage 1 (long-tail perception) 只占 10%，Stage 2 (planning+gen) 占 40%，Stage 3 (CoT reasoning) 占 50%。

- CoT 数据占比最大（50%）说明 **reasoning 是最终目标**，更多 reasoning data 让模型在 long-tail 上更 robust
- Planning+gen 占 40% 保证 physical dynamics 不被遗忘
- Long-tail perception 只占 10% 是因为 Stage 1 已经 1M steps 充分训练过，这里只是 anti-forgetting

这种 curriculum ratio 跟 [AlphaDrive](https://arxiv.org/abs/2503.07608) 的 RL+reasoning 思路类似——后期 reasoning 主导。

### 8.7 跟 Epona 的核心对比

[Epona](https://arxiv.org/abs/2506.24113) 是最接近的 baseline，也是 autoregressive diffusion world model + planning unified。UniUGP 比 Epona 的 advantage：
1. Epona 没有 CoT reasoning（Table 1 中 Reason.=X）
2. Epona 没有 instruction following（Inter.=X）
3. UniUGP 有更强的预训练 VLM backbone（Qwen2.5-VL-3B）

Epona 的 advantage 是更纯粹的 world model formulation。UniUGP 是 "world model + VLA" 的 hybrid，Epona 是 "world model alone"。

### 8.8 Limitations 部分的诚实

Paper 在 Appendix C 列了 4 个 limitations，值得注意：
1. **Extreme rare event generalization 受限于 training data coverage**——承认 long-tail 数据集还是不够
2. **Generation expert 计算成本高**，mobile 必须关闭——这是 cascade 架构的 inherent trade-off
3. **CoT 和 physical dynamics 的 alignment 仍然 suboptimal**——admit "interpretability-action inconsistency" 在 complex interaction 场景存在
4. **Stage 4 的固定数据比例**——没做 dynamic curriculum

第 3 点最 critical——这意味着 paper 承认 CoT 可能输出"听起来合理但跟实际 action 不完全 aligned"的 reasoning，这是 LLM-style reasoning 的根本性 limitation。

---

## 9. Future Directions 联想

Paper 提了几个方向，我补充我的联想：

1. **Self-supervised visual causal learning**：当前 CoT 还依赖 GPT-4o + manual calibration 构造，如果能用 large-scale unlabeled video 自动 mining causal structure（类似 [VideoMAE](https://arxiv.org/abs/2203.12602) + causal discovery）就能 scale up reasoning data
2. **Closed-loop training**：当前是 open-loop planning 评估（L2 + collision rate），真正的 test 应该是 [nuPlan closed-loop](https://arxiv.org/abs/2106.11810) 或 [CARLA](https://carla.org/) closed-loop，让 generation expert 的 future video 跟 actual rollout 对齐做 RL signal
3. **Multi-agent reasoning**：当前主要 ego-centric，扩展到 [SMARTS](https://arxiv.org/abs/2110.05914) 风格的 multi-agent interaction reasoning
4. **3D-aware generation**：当前 Wan2.1 是 2D video generation，可以升级到 [4D Gaussian Splatting](https://arxiv.org/abs/2309.11127) 或 [DrivingDreamer4D](https://arxiv.org/abs/2505.08627) 的 4D 生成，提供更强的 geometric supervision
5. **End-to-end with LiDAR**：当前只用 camera（Camera*），跟 [HERMES](https://arxiv.org/abs/2501.14729) 的 LiDAR unified model 对比还缺一个 modality

---

## 10. 关键 takeaway 总结

1. **Hybrid Expert > Joint Unified**：MoT 让 attention 共享但 FFN 分离，avoid gradient conflict between next-token prediction 和 flow matching
2. **Generation as Perception Supervision**：world model 不是为了 planning alone，而是反向 regularize VLM 关注 future causal relationship
3. **Cascade Architecture Enables Mobile Deployment**：generation expert 可关闭是 deployment 友好的关键设计
4. **4-stage Curriculum Avoids Multi-task Interference**：先 perception，再 dynamics+planning，再 CoT，最后 joint fusion
5. **CoT Reasoning 反过来帮助 Perception**：ablation 证明 CoT 让 Small object recognition 从 86.5%→89.3%

整体看 UniUGP 是一个 well-engineered system paper，技术上把 VLA 和 World Model 两条路线**优雅地 weave 在 MoT 架构里**，实验数据充分，ablation 严谨。最 intriguing 的发现是 **w/o Gen 的 ablation**——证明了 World Model 路线对 VLA 路线的 cross-modal regularization 价值，这是论文最有 contribution 的 insight。

Reference links:
- [UniUGP Project Page](https://seed-uniugp.github.io/)
- [Flow Matching Paper](https://arxiv.org/abs/2210.02747)
- [π0 VLA Flow Model](https://arxiv.org/abs/2410.24164)
- [Epona World Model](https://arxiv.org/abs/2506.24113)
- [F1 Robot VLA](https://arxiv.org/abs/2509.06951)
- [Wan2.1 Video Model](https://arxiv.org/abs/2503.20314)
- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923)
- [Impromptu VLA](https://arxiv.org/abs/2505.23757)
- [Waymo-E2E Long-tail](https://arxiv.org/abs/2510.26125)
- [DADA2000](https://ieeexplore.ieee.org/document/9666805)
- [Lost and Found](https://arxiv.org/abs/1609.04653)
- [DriveLM Benchmark](https://arxiv.org/abs/2305.06147)
- [nuScenes Dataset](https://arxiv.org/abs/1903.11027)
- [DriveDreamer World Model](https://arxiv.org/abs/2309.09777)
- [OccWorld](https://arxiv.org/abs/2406.09823)
- [Doe-1 Unified Model](https://arxiv.org/abs/2412.09627)
- [Transfusion](https://arxiv.org/abs/2408.11039)
- [Janus Unified Multimodal](https://arxiv.org/abs/2504.06256)
- [AlphaDrive RL Reasoning](https://arxiv.org/abs/2503.07608)
- [ReCogDrive](https://arxiv.org/abs/2506.08052)
