---
source_pdf: Igniting VLMs toward the Embodied Space.pdf
paper_sha256: 16310df8f960b0864a1025a763e0b8e8e0515f81c17836d9e31250930948846d
processed_at: '2026-08-19T12:08:53-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，我换个说法，像跟朋友聊天那样讲。

---

## 这篇 paper 到底在干啥

想象你有个很聪明的朋友，读书万卷，能聊哲学聊艺术，但从来没下过厨房。你现在让他去做一道菜——他看得懂菜谱，但手不会动。

现在的问题就是：**怎么让这个"读书人"学会做饭**。

VLM（视觉语言模型）就是这个读书人。它能看图、能聊天、能推理，但它不会输出"动作"——不会控制机器人的手臂去抓东西、放东西。这篇 paper 就是想办法把这个读书人改造成一个又能聊又能干活的机器人。

---

## 难在哪？三个坑

**第一个坑：动作和文字天生不一样**

文字是压缩过的——"猫"一个字就代表一整只猫。图片也是压缩过的——visual encoder 已经帮你把像素浓缩成有意义的特征了。但动作呢？机器人的手臂每秒钟要发几十个指令，每个指令是连续的数字（位置、角度、速度），没有"压缩器"帮它提炼语义。你跟它说"把杯子放到盘子上"，这句话很简单，但对应的动作是一长串连续的高频信号——**抽象层次对不上**。

**第二个坑：VLM 没见过机器人视角的照片**

VLM 训练用的是网上的图片——构图漂亮、第三人称、广角镜头。但机器人看到的世界是第一人称、鱼眼、画面里还伸着一只自己的机械臂。VLM 一看到机械臂就懵了，会 hallucinate，会认错东西。**训练数据的分布对不上**。

**第三个坑：训练目标冲突**

VLM 的训练方式是"预测下一个 token"——离散的、一个一个蹦出来。但动作是连续的、高频的，更适合用 diffusion 或 flow matching 这种"从噪声里慢慢 refine 出来"的方式。你硬把这两套目标接在一起，模型会精神分裂——**要么忘了怎么说话，要么学不好动作**。

---

## 他们的解法：三招

### 第一招：分两个房间存知识

前面的人试过两种极端。一种是直接把动作接到 VLM 后面一起训——结果 VLM 的语言能力被动作训练搞崩了。另一种是单独搞个动作模块，从 VLM 那边"借"信息——结果动作和语言没绑定紧，你说"拿红色的球"它可能拿蓝色的。

WALL-OSS 的做法很巧妙：**把模型的 FFN（前馈网络）劈成两半**。一半专门存视觉语言知识，一半专门存动作知识。但 self-attention 不劈——三种模态在 attention 层面充分交流。

类比一下：**同一个办公室里，大家在同一个会议室讨论（attention 共享），但回各自工位存文件（FFN 分开）**。这样动作训练的梯度不会直接冲进语言知识的存储里搞破坏，但语言和动作之间的信息交流还是畅通的。

这个叫 "tightly coupled MoE"——紧耦合的混合专家模型。用 static routing（硬编码路由），动作 token 永远去动作 FFN，语言 token 永远去语言 FFN，不用学路由，简单稳定。

### 第二招：先离散再连续，逐步过渡

不能一上来就让 VLM 学连续动作（flow matching），太难了，模型会崩。

他们的做法是分两步：

**Step 1（Inspiration 阶段）**：先把连续动作通过 FAST tokenizer 压成离散 token——就像把一首连续的歌压缩成一段简谱。离散 token 是 VLM 的舒适区，它本来就是干这个的。同时训练 embodied VQA（问模型"画面里机器人在干嘛""下一步该做什么"），让 VLM 先理解机器人视角的世界。

这一步相当于让读书人先看图认字——"哦，这是机械臂，那是杯子，现在在抓东西"。

**Step 2（Integration 阶段）**：把离散动作换成连续的 flow matching。先冻住 VLM 只训练动作头，让动作头有个合理起点。然后解冻 VLM 一起训练。

这一步是让读书人开始动手——但先让他从简单的开始，别一上来就颠勺。

### 第三招：一条链打通，想跳就跳

传统做法是把任务拆成两步：先用 LLM 规划"第一步拿盘子、第二步拿刀、第三步拿叉"，再用另一个模型执行。问题是两步之间的接口不连续，错误会累积——规划错了执行就全错。

WALL-OSS 的做法是**一个模型全包**：指令 → 推理（CoT）→ 子任务分解 → 连续动作。全在一个模型里、端到端可微。

而且训练时用 **path-drop**：随机把中间的推理步骤丢掉，让模型同时学会两条路——
- 复杂任务：指令 → 推理一下 → 拆子任务 → 执行
- 简单任务：指令 → 直接执行

推理时模型自己判断要不要想。简单活儿直接干，复杂活儿先想想再干。

---

## 数据怎么搞的

三个来源：

1. **自己采的机器人数据**：多种机器人平台（桌面臂、移动支架、轮式双臂、轮式人形），多种场景（厨房、卧室、装配），多视角相机。用多模型 pipeline 自动标注步骤，人工抽查。

2. **开源动作数据**：DROID、BC-Z、BRIDGE 等一堆数据集，统一坐标系、统一单位、统一控制频率、用 DoF 模板 + mask 处理不同机器人形态。

3. **VQA 数据**：一部分是通用 VQA（保持 VLM 基本功不退化），一部分是专门针对 embodied 场景的 VQA（空间理解、时序推理、affordance）。

总共超过 10000 小时。

---

## 效果怎么样

几个关键数字：

**VLM 的 embodied 理解力**：
- Object Grounding（找东西在哪）：46% → 92%，接近翻倍
- Scene Captioning（描述场景）：58% → 88%，不再 hallucinate

**动作精度和泛化**：
- 数据充足时（1000 demos）：WALL-OSS 和 π0 都 100%，从零训的 DP 只有 80%
- 数据少时（500 demos）：WALL-OSS 和 π0 90%+，DP 直接崩到 20% 以下
- 换新环境（OOD）：DP 直接 0%，WALL-OSS 和 π0 保持 80%+

**长时序任务**（3-5 分钟的多步骤任务）：
- baseline 会 stage confusion——重复放餐具、不知道下一步干啥、卡在无效循环里
- WALL-OSS 自己生成 subtask 提示来保持进度感知

**需要推理的任务**（Block-Spell 拼字母）：
- 多模态共训练的 WALL-OSS：87% 正确率
- 只训动作的 WALL-OSS：26%
- π0 只训动作：9%（接近瞎猜）

最后这个对比最说明问题——**紧耦合架构 + 多模态共训练让指令跟随能力有本质提升**，不是调参能调出来的。

---

## 一句话总结

WALL-OSS 干了三件事：

1. **FFN 分家、attention 共享** → 动作训练不破坏语言知识，但两者还是紧密绑定
2. **先离散后连续** → 让 VLM 先建立动作意识，再升级到精确连续控制
3. **一条链打通语义到动作** → 一个模型包揽推理、规划、执行，还能根据任务难度灵活跳步

本质上就是在说：**VLM 和 action 不是零和博弈，不需要牺牲一个保另一个，关键是架构上隔离好、训练上分好阶段、数据上补齐短板**。

代码开了：https://github.com/X-Square-Robot/wall-x

---

# WALL-OSS 深度技术讲解

Andrej, 这篇 paper 我读了之后挺兴奋的——它实际上是把当前 VLA 领域几个最棘手的问题用一种特别"工程化"的方式串起来解。让我从底层逻辑开始 build intuition，把每个设计选择背后的"为什么"讲清楚。

---

## 1. 核心问题诊断：三个 Gap

paper 一开篇就诊断了把 VLM 迁移到 embodied space 的三个 fundamental gaps。这个 diagnosis 本身就值得仔细看，因为它直接决定了后面的 architecture。

### Gap 1: Modality & Data Scale Gap

Vision encoder 经过 CLIP/MAE 等多年发展，已经学会了"压缩高频噪声、保留高层语义"——visual token 是高度 semantic-compressed 的。Text modality 同样 compressed。所以 CLIP 能在 web-scale text-image pairs 上把两者对齐。

**Action 不一样**。Action 是 3D 空间 + 时间维度的**连续信号**，频率高（几十 Hz），没有像 vision 那样的"语义压缩器"。同时，embodied 场景里一个 high-level instruction（比如"收拾卧室"）往往抽象掉了大量 subtask 和 scene-level action description——这种**抽象不对称**让 cross-modal association 极其困难。

> Reference: 这种"action 缺乏 representation learning 历史积累"的论点在 π0 paper (https://arxiv.org/abs/2410.24164) 里也有类似讨论，但 WALL-OSS 把它显式列为第一 gap。

### Gap 2: Pretraining Distribution Gap

这个我觉得是最直觉的一个。Internet image 是第三人称、广角、构图良好的。Embodied data 是第一人称、fisheye、self-occlusion、robot arm 出现在画面里。VLM 即使经过大规模预训练，对 embodied scene 的 spatial reasoning、progress tracking 仍然不足（Kamath et al. 的 "What's up with VLMs" 实验 https://arxiv.org/abs/2310.19785 有量化结果）。

这里有个有意思的细节：paper 提到 base VLM 在 embodied scene captioning 时会 hallucinate，在 object grounding 时被 robot arm 误导。这是 VLM 在分布外数据上的典型 failure mode。

### Gap 3: Training Objective Gap

这个 gap 最技术性，也最关键：
- LLM/VLM 的 objective 是 next-token likelihood on **discrete** sequences：$\mathcal{L} = -\log p_\theta(x_t | x_{<t})$
- Action trajectory 是 **continuous** 高频信号，更适合 conditional generative objective，比如 diffusion (https://arxiv.org/abs/2006.11239) 或 flow matching (https://arxiv.org/abs/2210.02747)

直接把 flow matching graft 到 VLM 上会"放大 tokenization gulf 和 independence assumption"——意思是：VLM 的 self-attention 假设 token 之间是 conditionally independent 给定 context，但 flow matching 的 velocity field 是连续依赖，两者假设冲突。

π0 的折中方案是：discretize action at high level（autoregressive），然后让 action noise 通过 self-attention 和 VLM representation 交互。WALL-OSS 认为这还是 too loosely coupled。

---

## 2. Architecture：Tightly Coupled MoE

这是 paper 的核心 architectural contribution。Figure 2 展示了三种 paradigm 对比：

### (a) Mixed Design (RT-2, OpenVLA)
直接把 action 建模（discrete 或 continuous）拼到 VLM 输出端，用 next-token prediction 范式。问题：action supervision 严重扰动 VLM 权重分布，导致 overfit 到 action，丢失 VL prior。

### (b) Decoupled Design (π0)
单独的 action branch，从 VLM 中 extract 信息。Vision/language 只是 auxiliary signal。问题：loosely coupled，instruction following 能力弱——action 和 language 没有学到 tight binding。

### (c) WALL-OSS：Tightly Coupled MoE
关键设计：
- 用 **Mixture-of-Experts**，把 FFN 分成 **Action FFN** 和 **Vision-Language FFN**
- 用 **static routing**（不是 learned softmax/top-k router，参考 Shazeer et al. https://arxiv.org/abs/1701.06538）
- 在 self-attention 层面 vision/language/action 三者**共享** representation，但在 FFN 层面**分离** weight update

这个设计背后的 intuition 我觉得是：
- **Self-attention 是 representation mixing 的地方**——三种 modality 在这里 cross-condition，实现 tight binding
- **FFN 是 knowledge storage 的地方**——分开 FFN 让 action 的 gradient 不会直接污染 VL 的 key-value memory
- **Static routing** 避免了 learned router 在训练初期的不稳定——action token 总是路由到 Action FFN，VL token 总是路由到 VL FFN，这是 hard-coded 的 inductive bias

数学上，对于 action token，前向传播是：
$$\mathbf{h}_{\text{action}}^{(l+1)} = \text{SA}(\mathbf{h}^{(l)}) + \text{FFN}_{\text{action}}(\text{SA}(\mathbf{h}^{(l)}))$$
对于 VL token：
$$\mathbf{h}_{\text{VL}}^{(l+1)} = \text{SA}(\mathbf{h}^{(l)}) + \text{FFN}_{\text{VL}}(\text{SA}(\mathbf{h}^{(l)}))$$
SA 步骤是共享的（同一个 attention layer 看所有 token），FFN 步骤是分离的。

> 这让我想起 MoE 在 language model 里的设计哲学（Switch Transformer, GShard），但这里 MoE 的"专家"是按 modality 分的，不是按 task 分的。这是一个非常巧妙的 repurposing。

---

## 3. Two-Stage Training Curriculum

Figure 4 展示了完整 training pipeline。两个 stage 的设计 logic 是逐步引入 action supervision，避免 catastrophic weight drift。

### Stage 1: Inspiration（启发 VLM 的 embodied 能力）

这个 stage 的目标是给 VLM 注入 **coarse action awareness** 和 **embodied spatial reasoning**，但还停留在 VLM 舒适区（discrete token 预测）。

**Training objective**：

$$\mathcal{L}_{\text{Inspiration}} = \lambda_{\text{VQA}} \sum_t -\log p_\theta(\tau_t | \tau_{<t}, \mathbf{c}) + \lambda_D \sum_k -\log p_\theta(z_k | z_{<k}, \mathbf{c})$$

变量含义：
- $\mathbf{c} = (\text{vision}, \text{instruction})$：multimodal context
- $\tau_t$：第 $t$ 个 **text token**（VQA 的回答，包括 CoT reasoning）
- $z_{1:K} = \text{FAST}(\mathbf{a})$：把连续 action trajectory $\mathbf{a}$ 通过 **FAST tokenization** 离散成 $K$ 个 action token
- $\lambda_{\text{VQA}}, \lambda_D$：两个 loss 的权重 hyperparameter

**FAST tokenization** 来自 Pertsch et al. (https://arxiv.org/abs/2501.09747)，是一种高效的 action tokenization 方法，把连续 action 压缩成离散 codebook token。这里用它是因为：
1. Discrete token 和 VLM 的 native representation 对齐
2. FAST 是 compression-based，token 数量少（K 不大），训练效率高
3. 它保留了 action 的"语义骨架"，是 coarse supervision

同时，**embodied VQA** 任务（Action Planning VQA、Spatial & Temporal QA、Perception VQA、Cognition & Affordance）一起训练，弥补 VLM 对 embodied scene 的 spatial reasoning 不足。这是直接打 Gap 2。

### Stage 2: Integration（融合三模态）

把 discrete action 升级成 continuous action via **flow matching**。分两 phase：

**Phase 1**：freeze VLM，只训练 flow head
**Phase 2**：unfreeze VLM，joint training

Flow matching 的 objective：

$$x_t = (1 - \rho(t)) x_0 + \rho(t) \epsilon$$
$$\mathcal{L}_{\text{Integration}} = \lambda_C \mathbb{E}\left[w(t) \| v_\phi(x_t, \mathbf{h}, t) - (\epsilon - x_0)\|_2^2\right]$$

变量含义：
- $x_0$：clean action sample（ground truth action）
- $x_t$：time $t$ 的 noisy sample（前向 diffusion process 的中间状态）
- $\epsilon$：Gaussian noise $\sim \mathcal{N}(0, I)$
- $\rho(t)$：noise schedule function，控制 $t$ 时刻 noise 占比
- $v_\phi(x_t, \mathbf{h}, t)$：velocity field network，参数 $\phi$，输入是 noisy action $x_t$、VLM encoding $\mathbf{h} = F_\theta(\mathbf{c})$、时间 $t$
- $w(t)$：weighting function，控制不同时间步的 loss 权重
- $\lambda_C$：continuous action loss 的总权重

**velocity target** $(\epsilon - x_0)$ 是 flow matching 的标准形式（Lipman et al. 2022），它学习把 noise distribution transport 到 action distribution 的 velocity field。

为什么两 phase？Phase 1 让 flow head 先有一个合理的 initialization，避免 random flow head 通过 gradient 严重扰动 VLM。Phase 2 再让 VLM 和 flow head joint adapt。这和 LoRA / adapter 训练里常见的 "freeze-then-unfreeze" 思路一致，但用在了 cross-modal 对齐上。

> 这种 staged curriculum 让我想到 LLaVA 的两阶段训练（projector pretrain → joint instruction tuning），但 WALL-OSS 的 stage 划分是按 **action 表示形式**（discrete → continuous）来的，这是更深层的设计。

---

## 4. Unified Cross-Level CoT (Uni-CoT)

这个我觉得是 paper 里最有意思的 conceptual contribution。

**Traditional CoT**（narrow-sense）：step-by-step textual reasoning in LLM
**Uni-CoT**（broad-sense）：instruction → reasoning (CoT) → subtask plan → continuous actions

整个链条在一个 model 里、end-to-end differentiable、可以 forward arbitrary mapping。

**Objective**：

$$\min_\theta \mathbb{E}_{(v, x, c, a)}\left[\ell_{\text{act}}(F_\theta(v, x, c), a_{1:T}) + \lambda \ell_{\text{VQA}}(H_\theta(v, x), y)\right]$$

变量：
- $v$：visual input
- $x$：language instruction
- $c$：**optional** chain-of-thought（注意是 optional！）
- $a_{1:T} \in \mathbb{R}^{T \times d}$：target action trajectory，长度 $T$，维度 $d$
- $F_\theta$：unified predictor，输出 action
- $H_\theta$：embodied-aware VQA head
- $y$：VQA supervision
- $\ell_{\text{act}}$：action prediction loss（flow matching loss）
- $\ell_{\text{VQA}}$：VQA loss
- $\lambda$：balancing hyperparameter

**关键 trick：path-drop**。训练时随机 drop 中间 reasoning step $c$，让 model 同时学会：
- Full chain: $(v, x) \to c \to \text{subtask} \to a$（复杂任务用）
- Direct: $(v, x) \to a$（简单任务用，省推理时间）

这避免了传统 hierarchical system（SayCan https://arxiv.org/abs/2204.01691, Code-as-Policies https://arxiv.org/abs/2209.07753, Hi Robot https://arxiv.org/abs/2502.19417, GR00T N1 https://arxiv.org/abs/2503.14725）的 non-differentiable interface 和 error accumulation。

> Path-drop 让我想到 dropout 的精神——训练时引入随机性，推理时模型能灵活处理不同条件。但这里 drop 的是整个 reasoning chain，是 structural dropout。这个 idea 在 CoT-VLA (https://arxiv.org/abs/2503.02093) 和 ECoT (https://arxiv.org/abs/2407.08693) 里也有类似探索。

---

## 5. Data Composition

数据超过 10,000 小时，三部分：

### Self-collected Robot Action Data
- 平台：desktop arms、mobile stands、wheeled bi-arm、wheeled humanoids
- 相机：egocentric / exocentric / arm-mounted
- 任务：short-horizon manipulation + long-horizon reasoning
- 多模型 pipeline 做细粒度 step annotation + 人工 spot check

### Open-Source Action Data
- DROID (https://arxiv.org/abs/2403.12945), BC-Z (https://arxiv.org/abs/2202.02005), BRIDGE V2 (https://arxiv.org/abs/2308.12952), FurnitureBench (https://arxiv.org/abs/2310.03579), RH20T (https://arxiv.org/abs/2307.00595), UMI-biarm (https://arxiv.org/abs/2409.19499), AgibotWorld (https://arxiv.org/abs/2503.06669) 等
- **Standardization**：coordinate frame、unit、morphology template（maximally expressive DoF + masking）、perception alignment、action time-base normalization

这个 normalization 协议是工程上很重要的一块。跨 morphology 训练要把单臂、双臂、轮式、人形统一到一个 DoF template，missing joint 用 mask/placeholder 处理。

### Multimodal VQA
- General VQA（CapsFusion, COCO, VQAv2, RoboPoint, Robo2VLM 等）：maintain VLM 能力
- Embodied VQA（自建 pipeline）：spatial-temporal reasoning + task reasoning

Table 1 列了完整 dataset。

---

## 6. Experiments：定量分析

### Embodied VQA Benchmark（Table 2）

| Model | Object Grounding | Scene Captioning | Action Planning |
|---|---|---|---|
| Qwen2.5-VL-3B | 46.1% | 57.7% | 59.8% |
| WALL-OSS | **91.6%** | **87.6%** | **69.0%** |

Object Grounding 提升最大（46.1 → 91.6，接近翻倍）。这说明 embodied VQA 数据对 spatial reasoning 的提升非常显著。Scene Captioning 从 57.7 → 87.6 也很大，说明 model 不再 hallucinate 不相关内容。

> 这个数字给 build intuition 提供 reference：在 embodied scene 上，base VLM 的 grounding 能力大概在 50% 量级，经过专门 VQA 训练能到 90% 量级。这是 pretraining stage 的直接回报。

### Zero-Shot Instruction Following（Figure 7）
- Seen object instructions：85% task progress
- Novel object instructions：61% task progress
- 失败案例主要是 pose 不准（grasping/placement 位置），不是 semantic 误解

### Action Accuracy（Collect-Waste, Pick-Place-Cup）

| Setting | WALL-OSS | π0 | DP |
|---|---|---|---|
| Collect-Waste ID (1000 demos) | 100% | 100% | 80% |
| Pick-Place-Cup ID (500 demos) | >90% | >90% | <20% |
| Collect-Waste OOD | >80% | >80% | **0%** |

**Key takeaway**: pretraining 让 model 在少样本和 OOD 上有显著优势。DP（Diffusion Policy，from scratch）在 500 demos 时崩盘，OOD 直接 0%。

### Long-Horizon（Set-Table, Tidy-Bedroom）
- 平均执行时间 > 3 分钟 / 5 分钟
- WALL-OSS 显著优于 π0 和 DP
- Baseline 的主要 failure mode：stage confusion（重复放餐具、不知道下一步该干啥）
- WALL-OSS 通过 self-generated subtask cue 维持 progress awareness

**有意思的细节**：fine-tuning 时只有 **1% 训练数据**带 subtask label，但 model 学会了 generate high-quality subtask instruction。这是 in-context 的一种 generalization。

### CoT 任务（Place-by-Color, Block-Spell）

Place-by-Color 两个 condition：
- 直接视觉匹配（red ball on red paper）：CoT 帮助不大
- 文字推理（red ball on paper printed "red"）：WALL-OSS 显著更好

Block-Spell 的 instruction following 准确率（Table 3）：

| Block Type | WALL-OSS (Co-training) | WALL-OSS (Action-only) | π0 (Action-only) |
|---|---|---|---|
| Letter | **87%** | 26% | 9% |
| Number | **95%** | 80% | 35% |

这个 ablation 非常有信息量：
- **Multi-modal co-training** 让 WALL-OSS 从 26% → 87%（letter）
- 即使 Action-only，WALL-OSS 26% 也比 π0 9% 高——说明 pretraining 阶段留下的 multimodal alignment 有持续 benefit
- π0 在 letter 上只有 9%（接近 random chance）——decoupled architecture 的 instruction following 弱

---

## 7. 我的 Intuition 总结

读完这篇 paper，我脑子里形成了几个 mental model：

1. **MoE-as-modality-insulator**：把 FFN 当成"知识容器"，self-attention 当成"信息交换机"。Static routing 让 action gradient 不污染 VL memory，但 attention 让 binding 还能发生。这是一个非常清晰的 separation of concerns。

2. **Discrete-to-continuous curriculum**：FAST discrete token 是 VLM 和 action 之间的"语义桥梁"——它把 action 拉到 VLM 舒适区附近，让 model 先建立 "action awareness"，然后再扩展到 continuous flow matching。这避免了 cold-start 的 weight drift。

3. **CoT as flexible routing**：path-drop 让 model 学到"什么时候需要思考、什么时候直接行动"。这是一个非常实用的 inductive bias——不同任务需要的 reasoning depth 不同。

4. **1% subtask label → strong generalization**：这说明 VLM 的 reasoning 能力是 pre-existing 的，subtask generation 不是从 0 学，而是被"激活"。和 in-context learning 的精神一致。

---

## 8. 一些 critical observation 和联想

1. **π0 vs WALL-OSS 的本质区别**：π0 是 decoupled（action head 是 separate module，通过 cross-attention 从 VLM 取信息），WALL-OSS 是 coupled MoE（共享 attention layer，分离 FFN）。Paper 的 ablation（Table 3）显示 coupled 设计在 instruction following 上有本质优势。

2. **没有 3D modality**：paper Discussion 部分提到 3D-vision foundation model（VGGT https://arxiv.org/abs/2503.02073, π³ https://arxiv.org/abs/2507.13347）还达不到 fine-grained action prediction 的精度，所以 WALL-OSS 没用 3D bridge。这是个 open question。

3. **Video prediction as intermediate**：Discussion 里提到 video prediction 可能是 VL→A 的"intermediate modality"，但承认会引入 redundant supervision。这是 world model vs policy 的经典 debate。

4. **Static routing 的局限**：static routing 是 hard-coded inductive bias，可能在 morphology 扩展时需要手动调整。Learned router（Top-2 in Mixtral）更灵活但训练不稳定。WALL-OSS 选了稳定性。

5. **Code 已经开源**：https://github.com/X-Square-Robot/wall-x，project page https://x2robot.com/en/research/68bc2cde8497d7f238dde690。这对复现很有帮助。

---

## 9. 我会想做的 follow-up

如果让我 follow 这条线，我会想：

1. **Scaling law of tightly-coupled MoE**：专家数量怎么 scale？Action FFN 和 VL FFN 的 capacity ratio 怎么定？
2. **Multi-embodiment transfer**：static routing 在新 morphology 上是否需要重新设计？能不能 learn 一个 morphology-conditional router？
3. **CoT length 的自适应**：path-drop 是 uniform 随机，能不能做成 task-difficulty-conditional？
4. **Inspiration stage 的 action tokenizer**：FAST 是 fixed codebook，能不能 learn 一个 task-conditional tokenizer？
5. **结合 RL**：目前是 pure imitation，能不能在 Integration Phase 2 后加 RL fine-tuning（PPO / RLHF on action）？
6. **Memory augmentation**：long-horizon task 是否需要 external memory（比如 retrieval）？Uni-CoT 是 in-context，超过 5 分钟可能 bottleneck。

---

## 10. Related work 的 mental map

我把 paper 里提到的 related work 整理成一个 taxonomy：

**Discrete VLA**: RT-2 (https://arxiv.org/abs/2307.15818), OpenVLA (https://arxiv.org/abs/2406.09246), FAST (https://arxiv.org/abs/2501.09747), ACT (https://arxiv.org/abs/2304.13705)
**Continuous VLA**: DP (https://arxiv.org/abs/2303.04137), Octo (https://arxiv.org/abs/2405.12213), π0, π0.5 (https://arxiv.org/abs/2504.16054)
**3D bridge**: 3D-VLA (https://arxiv.org/abs/2403.09631), PointVLA (https://arxiv.org/abs/2503.07511)
**Hierarchical**: SayCan, Code-as-Policies, Hi Robot, GR00T N1, RT-H (https://arxiv.org/abs/2403.01823), HAMSTER (https://arxiv.org/abs/2502.05485)
**CoT for control**: CoT-VLA (https://arxiv.org/abs/2503.02093), ECoT (https://arxiv.org/abs/2407.08693), LLaVA-A: Vision-Action (https://arxiv.org/abs/2406.11815), PIVOT (https://arxiv.org/abs/2402.07872)
**VLM backbones**: Qwen2.5-VL (https://arxiv.org/abs/2502.13923), InternVL series, Gemini 2.5, GPT-5

WALL-OSS 的定位是 **discrete prior → continuous control + tightly coupled MoE + unified CoT**，融合了几个不同 line 的工作。

---

总结一句：这篇 paper 的 contribution 不是单个 trick，而是一整套**针对三个 gap 的对应解药**：tightly coupled MoE（modality gap）+ staged curriculum（objective gap）+ embodied VQA + Uni-CoT（distribution gap）。三者环环相扣，build intuition 时最好作为一个 system 来理解，单看任何一个组件都不完整。
