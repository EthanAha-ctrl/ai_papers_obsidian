---
source_pdf: TacVLA Contact-Aware Tactile Fusion.pdf
paper_sha256: 15c8dc1a218f2f819c9997e39c5c925457a9f11fb42ee09cc464fd4090401eba
processed_at: '2026-08-12T12:38:45-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 TacVLA

好,我用大白话讲一遍。

---

## 机器人现在的问题

想象你让一个机器人去"把抽屉里的小东西拿出来放到碗里"。机器人怎么干这事?

它就靠**两个眼睛(摄像头)**和**一句指令(文字)**。

眼睛告诉它"东西大概在哪",指令告诉它"干啥"。

但问题是:

1. **抽屉里面黑乎乎的**,摄像头根本看不清。机器人在里面瞎抓,抓了半天啥也抓不到。
2. **有些活儿眼睛根本看不出来**。比如你让它"把两个卡在一起的零件掰开",它能看到零件外形,但卡得多紧、该往哪个方向使劲、有没有滑动 — 这种"手感"信息,摄像头给不了。
3. **碰到意外就懵了**。比如人突然把东西拿走了,机器人还在那空抓,因为它看不见"接触没了"这个事件。

人类怎么干这种活儿?**靠手感**。你摸黑找钥匙,手指碰到钥匙那个瞬间你就知道了;你拧螺丝拧不动,手会感觉"卡住了"然后加力或者换个方向。

所以核心想法很简单: **给机器人也装上"手指头上的触觉"**。

---

## 但触觉有个麻烦

你可能会说,那简单啊,装个触觉传感器不就完了?一直把触觉信号传给机器人不就好了?

没那么简单。这里有个**很反直觉的坑**:

**机器人没碰到东西的时候,触觉传感器也在输出信号**。那些是噪声、是基线漂移、是传感器的"自言自语"。

如果你一股脑把这些噪声也塞给机器人,它就会犯迷糊:

- 机器人正在空中飞,准备去抓杯子,结果手指传来一堆乱七八糟的触觉信号
- 机器人就开始怀疑: "我是不是碰到了什么?我要不要调整?我到底在哪?"
- 然后就**抓不准了,动作乱了,反复重新抓**

论文里专门做了实验验证这个: 如果不区分"有没有接触"就一直把触觉塞进去,有一个任务的成绩从 65% 掉到 60%,**比完全不用触觉还差**。

这就是关键 insight: **触觉这个东西,碰到了才有用,没碰到就是添乱**。

---

## TacVLA 的聪明做法

TacVLA 的设计就一个核心思路,非常朴素:

**碰到东西,触觉才接入;没碰到,触觉直接关掉。**

具体怎么"关掉"?两个动作:

1. **捂住耳朵**: 告诉模型"现在别听触觉这一路" — 在注意力机制里把触觉那部分屏蔽掉,其他信号(vision, language)不能去看触觉,触觉也不能去看别人。
2. **把触觉信号清零**: 直接把触觉的那串数字全置零,连位置编码的偏置都消掉,不让传感器的底噪累积。

怎么判断"有没有接触"?简单粗暴 — **数一数触觉传感器上有几个点压力超过阈值**,超过一定数量就认为"接触了"。二元判断,有接触就是 1,没接触就是 0。

就这么一个简单开关,效果巨大。

---

## 效果有多猛

论文测了三类任务,我挑最直观的说:

**任务一: 从盒子里摸东西出来**

这个场景摄像头基本废了 — 前面摄像头看不到盒子里,手腕摄像头光照又差又被挡。

- 只靠视觉的机器人: 20 次只成功 2 次 (10%)。机器人经常伸进去空抓一通,没抓到就以为抓到了,开始往上提,结果啥也没提上来。
- TacVLA: 20 次成功 14 次 (70%)。机器人进去摸,摸到了才抓,没摸到就换个位置继续摸。

**这就是触觉的价值 — 视觉失效的时候,手感是你唯一的确认信号。**

**任务二: 把卡住的零件拆下来**

有四种不同的卡法: 紧配合的轴、按压卡扣、旋转 90 度再拔、往里滑再拔。

最难的是第四种 (往里滑再拔),因为动作过程中有遮挡,视觉看不清。

- 只靠视觉: 20 次成功 6 次 (30%)。机器人经常卡在中间状态,反复重新抓,显得很犹豫。
- TacVLA: 20 次成功 15 次 (75%)。机器人接触转换时更稳,知道"现在卡住了,该换方向了"。

**任务三: 遮住摄像头测试**

直接把前面的摄像头挡住,看机器人还能不能干。

- 只靠视觉: 平均成功率掉到 30% 左右。
- TacVLA: 保持在 60% 以上,**翻了一倍多**。

特别是"旋转 90 度再拔"这个任务,视觉本来就很难判断旋转了多少 (RGB 图像看旋转不直观),触觉直接感知扭矩和正压力,反而更直接,提升最大。

---

## 为什么比 Diffusion Policy 好这么多

论文还对比了两个 diffusion policy (另一种机器人控制方法),它们也加了触觉输入,但效果很差: 拆零件平均 30-49%,摸盒子 0-5%。

原因其实不在触觉本身,而在**底子和训练方式**:

- Diffusion policy 是**从零开始学**这个任务的,250 个示范数据根本不够它学会复杂的接触判断。
- TacVLA 背后是个**已经预训练好的大模型 (Pi0.5)**,它已经从海量数据里学过"物体长啥样、语言怎么理解、动作怎么连续"这些通用知识。新的触觉模态只是锦上添花,用 LoRA 微调一点点参数就能接上。

这就像: **让一个没上过学的人从头学修手表,和让一个钟表匠学一种新工具的区别。** 钟表匠只学新工具就行,底子全在。

---

## 整个流程串起来说

1. **看**: 两个摄像头拍图,SigLIP 编码成视觉 token
2. **读**: 文字指令 + 机器人自己的关节状态,用 PaliGemma 编码成 token
3. **摸**: 触觉阵列 (15×8 个点) 采到压力分布,小 MLP 编成 36 个 token
4. **判断有没有接触**: 数一数有多少点压力超阈值,超了就是"有接触"
5. **如果有接触**: 触觉 token 正常参与,和视觉、语言一起送进大模型推理
6. **如果没接触**: 触觉 token 清零,attention 屏蔽掉,就当触觉不存在
7. **大模型 (Pi0.5) 吃下这些 token**: 前半段双向 attention 让各模态充分交流
8. **Action expert 生成动作**: 用 flow matching (一种比 diffusion 更高效的生成方法) 一次性预测未来一段动作序列
9. **机器人执行,循环**

---

## 这事为什么有意思

抛开技术细节,我觉得这篇 paper 传达了一个更深的思路:

**不是所有信息都该一直参与决策。该上场的时候上场,该退场的时候退场。**

这个道理放到人身上也对:
- 你走路的时候不需要一直盯着脚看 (视觉),但踩到石头了脚底会立刻告诉你 (触觉)
- 你说话的时候不需要一直听自己声音 (听觉),但跑调了会立刻发现 (反馈)

机器人的多模态融合一直是"把所有信号一股脑堆一起"的做法。TacVLA 说: **慢着,触觉是事件型的,碰了才有意义。给它装个开关,该开的时候开,该关的时候关。**

这个开关虽然是个简单的二元判断,但效果比"一直全开"好得多。说明在多模态融合里,**什么时候用哪个模态** 比 **用多少模态** 更重要。

---

## 未来的想象

论文自己承认这个开关太"硬"了 — 只能开或关,不能"半开"。想象一下未来:

- 触觉可以"半激活",刚碰到一点点 vs 死死卡住,激活强度不一样
- 视觉也可以类似处理: 光线好的时候全信视觉,光线差的时候半信视觉半信触觉
- 甚至让模型自己学什么时候该信哪个模态,不用人手设计规则

长远看,如果机器人全身铺满触觉 (像人皮肤一样),这套"按需激活"的思路就更重要了 — 不可能全身十万个触觉点一直全开,脑子处理不过来,必须只激活当前接触区域的那部分。

这篇 paper 算是把这个方向蹚出来了一条小路。

---

# TacVLA: Contact-Aware Tactile Fusion for Robust VLA Manipulation

Andrej, 这篇 paper 来自 Physical Intelligence 相关团队 (包括 Arash Ajoudani @ IIT, Yu She @ Ark Robotics),核心 idea 非常 clean: 在 Pi0.5 这种 VLA backbone 上挂一个触觉模态,但用一个 contact-aware gating 让 tactile tokens 只在 physical contact 发生时才参与 attention,避免 non-contact phase 的 noise 干扰 visual-language grounding。我下面拆开讲,试图帮你 build intuition。

---

## 1. Problem Motivation: Why Tactile for VLA?

当前 VLA 主流 (Pi0.5, OpenVLA, RT-2) 几乎都是 vision + language only。这在 contact-rich 任务上有个 fundamental bottleneck:

- **Visual occlusion**: 末端执行器或物体本身挡住 camera view。In-box picking 这种场景里,front camera 完全看不到 box 内部,wrist camera 也只有 limited illumination + 频繁 occlusion。
- **Fine-grained contact control**: Disassembly/insertion/assembly 任务需要感知 contact force, friction, surface texture, slip。这些 physical properties 单靠 RGB 像素根本 infer 不出来。
- **Dynamic adjustment**: 遇到 unexpected resistance 或 slippage,没有 contact feedback 就无法 closed-loop correct。

论文里举的 4 个 disassembly task 特别有说服力:
- Task 1: tight shaft (沿轴拉出)
- Task 2: press clip (按 + 拉)
- Task 3: shaft rotation (扭 90° + 拉)
- Task 4: slide pull (slide inward + pull out, partial occlusion)

这些任务共同特点: 没有触觉,robot 就只能"盲推",要么 stuck in intermediate state,要么 repeated re-grasping。Vision 告诉你"where",touch 告诉你"how hard, how stuck, how slip"。

---

## 2. Architecture Deep Dive

参考 Fig.1 的整体架构。我把它拆成 5 个模块:

### 2.1 Modality Tokenizers

**(a) Visual Tokenizer**
- 两个 RGB camera: front-facing (global scene) + wrist-mounted (close-range)
- 每个图像过 **SigLIP** visual encoder [Zhai et al., 2023]
- SigLIP 相比 CLIP 用的是 sigmoid loss 而不是 InfoNCE contrastive loss,在小 batch size 下也 work,这是为啥近期很多 VLA 选它
- Output: per-image 的 visual token sequence

**(b) Language + Proprioception Tokenizer**
- 用 **PaliGemma** tokenizer
- Language instruction 和 robot proprioception (joint pos/vel, gripper state) 一起 tokenize 成 token sequence
- Proprioception 用 text token 表达这个挺有意思 — Pi0.5 系列的做法,把 continuous state 用 PaliGemma 词表空间投影进去

**(c) Tactile Tokenizer**
- 硬件: 15×8 = 120 taxels 的 tactile array sensor (adapted from 3D-ViTac [Huang et al., 2024])
- MLP-based encoder 把 120 维的 tactile map 投影成 **36 个 tactile tokens**
- 加 fixed 2D sine-cosine positional embeddings 保留 spatial structure
- 关键设计选择: 没有用 image-like dense representation (像 GelSight 那种高分辨率 contact image),而是低维 token。这避免了 transformer 处理 dense tactile image 的 token 长度膨胀问题

参考公式理解这个 tokenization:

$$
\tilde{\mathbf{z}}_t = [\mathbf{z}_t^{vis}, \mathbf{z}_t^{lan+pro}, \tilde{\mathbf{z}}_t^{tac}]
$$

变量含义:
- $\tilde{\mathbf{z}}_t$: 时间 $t$ 的完整 multimodal token sequence (顶部加 tilde 表示这是 gating 后的 fused version)
- $\mathbf{z}_t^{vis}$: 视觉 tokens (SigLIP 编码 front + wrist camera)
- $\mathbf{z}_t^{lan+pro}$: language + proprioception tokens (PaliGemma tokenizer 输出)
- $\tilde{\mathbf{z}}_t^{tac}$: 经过 contact-aware gating 处理后的 tactile tokens (注意有 tilde)

中括号表示 concatenation,后续送进 VLM backbone。

### 2.2 Pretrained VLM Backbone (Pi0.5)

- Backbone 是 **OpenPI pi05_base** checkpoint
- 整个 multimodal token sequence 当作 transformer 的 prefix
- **Non-causal (bidirectional) attention** 应用在这个 prefix 上,允许 vision/language/tactile tokens 自由 cross-attend
- 这跟 prefix-LM 的设计一致 — prefix 部分 bidirectional,generation 部分 causal
- Fine-tune 用 LoRA,只动 low-rank adapters, tactile encoder 参数 frozen

### 2.3 Action Expert (Flow Matching)

参考 Pi0.5 [Physical Intelligence, 2025] 的 action expert 设计:

$$
\mathbf{a}_{t:t+H} \sim \pi_\theta(\mathbf{a}_{t:t+H} \mid \tilde{\mathbf{z}}_t)
$$

变量含义:
- $\mathbf{a}_{t:t+H}$: 从当前 timestep $t$ 到未来 horizon $H$ 的连续 action chunk (7-DoF arm + gripper)
- $\pi_\theta$: 参数为 $\theta$ 的 policy
- $H$: action prediction horizon (Pi0.5 默认是 50 steps)
- $\tilde{\mathbf{z}}_t$: 条件输入,即 gating 后的 multimodal tokens

Training 用 **flow-matching objective** (Lipman et al. 2023 提出的),相比 diffusion policy 的 stochastic differential equation,flow matching 学一个 vector field 直接 transport noise distribution 到 action distribution,训练更稳定,sample 路径更直,sampling step 可以更少。这是为啥 Pi 系列选它而不是 DDPM。

### 2.4 Contact-Aware Gating Module (核心创新)

这个模块是 paper 的核心 contribution,参考 Fig.1(c)。

**直觉**: Tactile 信号本质上是 *contact-dependent* 的。Robot 没碰到东西的时候,tactile array 输出的就是 sensor noise + baseline drift,这些噪声如果一直参与 attention,会污染 visual-language grounding。比如在 free-space reach 阶段,robot 应该完全靠 vision 去定位,触觉信号这个时候是 uninformative 的。

**实现**: 

定义 binary contact flag:

$$
c_t \in \{0, 1\}
$$

$c_t = 1$ 当且仅当 taxels 中超过预设压力阈值的数量超过一个固定 count。这是一个 simple threshold-based heuristic。

应用 attention mask:

$$
M_t^{tac} = c_t \cdot \mathbf{1}
$$

变量含义:
- $M_t^{tac}$: 时间 $t$ 对 tactile tokens 的 attention mask
- $c_t$: contact flag (0 or 1)
- $\mathbf{1}$: 长度等于 tactile token sequence 的全 1 向量 (即 36 维)

应用 embedding gating:

$$
\tilde{\mathbf{z}}_t^{tac} = c_t \cdot \mathbf{z}_t^{tac}
$$

变量含义:
- $\tilde{\mathbf{z}}_t^{tac}$: gating 后的 tactile tokens (送进 backbone)
- $c_t$: contact flag
- $\mathbf{z}_t^{tac}$: 原始 tactile tokens (MLP encoder + pos embed 输出)

**双重机制**:
1. **Attention mask**: 当 $c_t = 0$ 时,其他 tokens 不能 attend 到 tactile tokens,tactile tokens 也不能 attend 出去。这相当于在 attention matrix 里把这些位置 mask 掉。
2. **Embedding gating**: $\tilde{\mathbf{z}}_t^{tac} = 0$,直接 zero out 触觉 embedding,连 positional encoding 和 embedding layer 的 offset 都 suppress 掉,避免 sensor baseline noise 在 non-contact phase 累积偏置。

**为什么用 fixed token topology 而不是动态插删 token?**
保持固定 token 结构有一个 engineering 上的好处: batch 内 sequence length 一致,不需要处理 variable-length sequence 的 attention masking 复杂度,FlashAttention 也能正常 work。同时概念上,这个 gating 类似 **Mixture-of-Experts** 的 sparse activation 思路 — token 位置一直存在,但只有特定条件下才"激活"参与计算。这跟 Switch Transformer、Mixture-of-Depths 中的条件计算 philosophy 一致。

---

## 3. Training Procedure

- **Data**: 50 demos per task × 5 tasks = 250 demos,10 Hz 录制,visual + language + tactile + proprioception 全 temporal aligned
- **Backbone**: OpenPI pi05_base checkpoint
- **Fine-tune method**: LoRA (Low-Rank Adaptation)
- **Steps**: 10,000 gradient steps
- **Frozen**: tactile encoder parameters
- **Optimizer**: 一致设置(具体没写,可能是 AdamW + cosine schedule)

这个设置有意思的地方: tactile encoder 是 frozen 的,意味着他们要么 (a) 用预训练的 tactile encoder weights,(b) 或者干脆从 scratch 训练但只训练 encoder 不动 backbone,这暗示 tactile modality 通过 LoRA bottleneck 学习到的 cross-modal alignment 才是关键。

---

## 4. Experimental Results Detailed Analysis

### 4.1 Main Results (Table II)

| Method | Disassembly Avg | In-Box Picking |
|--------|-----------------|----------------|
| 3D Diffusion Policy + Tactile | 31.25% | 5% (1/20) |
| Diffusion Policy + Tactile | 48.75% | 0% (0/20) |
| Finetuned Pi0.5 (vision+lang only) | 63.75% | 10% (2/20) |
| **TacVLA (Ours)** | **83.75%** | **70% (14/20)** |

逐 Task 看 disassembly:
- Task 1 (tight shaft): Pi0.5 80% → TacVLA 100% (+20%)
- Task 2 (press clip): Pi0.5 80% → TacVLA 90% (+10%)
- Task 3 (shaft rotation): Pi0.5 65% → TacVLA 70% (+5%)
- Task 4 (slide pull, partial occlusion): Pi0.5 30% → TacVLA 75% (**+45%**)

Task 4 的 +45% 提升最显著,正是因为这个任务有 partial occlusion,vision 信号本来就 degraded,tactile 反而成了主导信号。这就是论文反复强调的"contact-aware fusion 在 vision occlusion 下 value 最大"。

In-box picking: Pi0.5 10% → TacVLA 70%,绝对提升 60%,这是 paper claim 的 60% improvement 来源。这个场景下 vision 完全失效 (front camera 无 visibility,wrist camera 严重 occlusion + 限光),触觉是唯一可靠的 contact verification 信号。

### 4.2 Why Diffusion Policy Baselines Fail So Badly?

3D Diffusion Policy + Tactile 在 disassembly 上只有 31.25%,in-box 是 5%。原因分析:

1. **From-scratch training**: Diffusion policy 没有 pretrained visual-language prior,250 个 demo 不足以学到 robust 的 contact reasoning
2. **Naive modality concatenation**: Diffusion policy 把 visual features 和 tactile features 直接 concat,没有 cross-modal interaction 机制
3. **Action representation**: DDPM 的 action generation 比 flow-matching 更难训练到 smooth trajectory
4. **No language grounding**: Diffusion policy 不吃 language instruction,只能 condition on observation

这跟近期 VLA 领域一个 broader insight 一致: **pretrained multimodal backbone + small fine-tune > from-scratch task-specific training**。

### 4.3 Robustness Evaluation (Fig. 6)

**Block front camera** (severe visual occlusion):

| Method | Task 1 | Task 2 | Task 3 | Task 4 | Avg |
|--------|--------|--------|--------|--------|-----|
| Finetuned Pi0.5 (vision-only) | 40% | 40% | 5% | 35% | 30% |
| Pi0.5 + Tactile (w/o gating) | - | - | - | - | 中等 |
| **TacVLA (with gating)** | 70% | 65% | 45% | 70% | ~62.5% |

平均从 30% 提升到 62.5%,2.1x improvement。论文 claim 的 2.1x 来自这里。

Task 3 提升最大 (+40%),因为 shaft rotation 任务 vision 本来就难看 (rotation 在 RGB 上 ambiguous),触觉直接感知 torque/normal force 反而更直接。

**Human disturbance**: 把已经抓起的物体放回 box,TacVLA 能 detect state change + return to box + re-grasp + continue。Pi0.5 baseline fails to recover。这说明 tactile feedback 提供了 *event detection* 能力 — contact 突然消失是一个明确 signal。

### 4.4 Ablation Study (Table III)

| Method | Disassembly Avg | In-Box Picking |
|--------|-----------------|----------------|
| Pi0.5 + Tactile (w/o Gating) | 71.25% | 40% |
| **TacVLA (with Gating)** | **83.75%** | **70%** |

Gating 贡献: disassembly +12.5%,in-box +30%。

特别 interesting: Task 3 上 w/o gating 是 60%,比 Pi0.5 baseline (vision only) 的 65% 还低!这说明 **naive tactile fusion 不仅没帮助,反而会 hurt performance**。这就是论文反复强调的 motivation — tactile tokens 在 non-contact phase 是 noise,直接 concat 进去会 dilute visual grounding。

Failure cases (Fig. 7):
- Misalignment during object approach
- Repeated re-grasp attempts
- Stalled intermediate states
- Failed lift attempts

这些症状都指向一个原因: 在 reach phase (没接触时),tactile noise 干扰了 visual localization,robot 找不到 object 的正确位置。

---

## 5. Architecture Intuition: Why This Design Works

我把几个设计决策的 reasoning 整理一下:

### 5.1 Why Low-Dimensional Tactile Tokens (36 tokens) vs Image-Like Tactile?

很多 prior work [VLA-Touch, OmniVTLA, TLA, VTLA] 把 tactile 当 image 处理 (GelSight 的 RGB contact image),但这篇 paper 用 15×8 = 120 taxel 数组,MLP 投影到 36 tokens。理由:

1. **Token efficiency**: GelSight image 经 ViT 编码动辄几百 tokens,加上 vision tokens 已经很贵。低维 tactile array 天然 compact。
2. **Information bottleneck**: Tactile 信号的物理意义是局部的 contact force/pressure,本身信息量有限,过度参数化反而过拟合 noise。
3. **Spatial structure preserved**: 加 2D sine-cosine positional embedding 保留了 15×8 grid 的拓扑,模型仍能学 contact geometry。

### 5.2 Why Binary Gating vs Soft Gating?

论文 limitation 部分自己也承认: binary threshold heuristic 不允许 gradual modality weighting。但 binary 设计有几个好处:

1. **Simplicity**: 不需要训练额外的 gating network,zero extra parameters
2. **Interpretability**: $c_t$ 直接对应物理事件 (contact / no contact),可监控可调试
3. **Stability**: Soft gating 在 training 中可能 collapse 到 0 或 1,binary 反而稳定

未来方向可能是 **learnable gating with sigmoid + temperature annealing**,或者借鉴 **Mixture-of-Experts routing** 的思路做 soft routing。

### 5.3 Why Non-Causal Attention on Prefix?

Pi0.5 系列的设计: prefix (multimodal tokens) 用 bidirectional attention,generation (action tokens) 用 causal attention。这跟 PaLM,Prefix-LM 一脉相承。好处:

- Bidirectional 让 vision/language/tactile 互相 attend,获得 deep cross-modal grounding
- Causal 在 action expert 部分保证 autoregressive generation 的因果性

如果 prefix 也 causal,会出现 vision token 不能 attend language token 这种问题,损失 grounding 能力。

### 5.4 Why Flow Matching vs Diffusion for Action?

Flow matching (Lipman 2023) vs Diffusion (DDPM/DDIM) 在 action generation 上的差异:

- **Diffusion**: 学 reverse SDE,sampling 路径是弯的,需要 many steps
- **Flow matching**: 学 vector field 把 noise → action distribution,sampling 路径直,少 steps 就能 generate

对 robotic control 这个 matters: action generation 频率高,少 step 直接 translate 到 inference latency 低。Pi 系列选 flow matching 是有道理的。

---

## 6. Related Work Landscape

我帮你把 paper 引用和相关工作梳理成几个 cluster:

### 6.1 VLA Models
- **Pi0 / Pi0.5** [Physical Intelligence, 2025] - backbone of TacVLA
  - https://www.physicalintelligence.company/blog/pi0
  - https://arxiv.org/abs/2504.16054
- **OpenVLA** [Kim et al., 2024] - 开源 VLA,基于 Prismatic VLM
  - https://openvla.github.io/
  - https://arxiv.org/abs/2406.09246
- **RT-2** [Brohan et al., 2023] - Google 的 VLA,PaLI-X backbone
  - https://robotics-transformer2.github.io/
  - https://arxiv.org/abs/2307.15818

### 6.2 Tactile-Enhanced VLA (最近 burst)
- **TLA** [Hao et al., 2025] - Tactile-Language-Action
  - https://arxiv.org/abs/2503.08548
- **VTLA** [Zhang et al., 2025] - Vision-Tactile-Language-Action with preference learning
  - https://arxiv.org/abs/2505.09577
- **Octopi-1.5** [Yu et al., 2025] - visual-tactile-language model
  - https://arxiv.org/abs/2507.09985
- **OmniVTLA** [Cheng et al., 2025] - semantic-aligned tactile sensing
  - https://arxiv.org/abs/2508.08706
- **VLA-Touch** [Bi et al., 2025] - dual-level tactile feedback
  - https://arxiv.org/abs/2507.17294
- **MLA** [Liu et al., 2025] - multisensory language-action model
  - https://arxiv.org/abs/2509.26642
- **Tactile-VLA** [Huang et al., 2025] - tactile generalization
  - https://arxiv.org/abs/2507.09160

这个领域 2025 年突然爆发,核心驱动是 VLA 主流化 + tactile sensor 硬件成熟。

### 6.3 Tactile Sensors
- **GelSight** [Yuan et al., 2017] - vision-based high-resolution tactile
  - https://arxiv.org/abs/1707.05136
- **GelSight Wedge** [Wang et al., 2021] - compact version
- **3D-ViTac** [Huang et al., 2024] - TacVLA 的 tactile sensor 来源
  - https://arxiv.org/abs/2410.24091
- **VibeCheck** [Zhang et al., 2025] - active acoustic tactile sensing
  - https://arxiv.org/abs/2504.15535
- **GelFlow** [Zhang et al., 2023] - optical flow for tactile
- **GelRoller** [Zhang et al., 2024] - rolling tactile sensor

### 6.4 Diffusion / Flow Policies
- **Diffusion Policy** [Chi et al., 2023] - 经典 visuomotor diffusion
  - https://diffusion-policy.cs.columbia.edu/
  - https://arxiv.org/abs/2303.04137
- **3D Diffusion Policy** [Ze et al., 2024] - 点云版本
  - https://arxiv.org/abs/2403.03954
- **Flow Matching** [Lipman et al., 2023] - generative model 基础
  - https://arxiv.org/abs/2210.02747

### 6.5 VLM Backbones Used
- **SigLIP** [Zhai et al., 2023] - sigmoid loss image pretraining
  - https://arxiv.org/abs/2303.15343
- **PaliGemma** [Google] - VLM based on Gemma
  - https://ai.google.dev/gemma/docs/paligemma
- **PaLI** 系列 - Google 的 VLM
- **Prismatic VLM** - OpenVLA 用的 backbone

### 6.6 Multimodal Fusion Methods (Related Ideas)
- **ImageBind** [Girdhar et al., 2023] - 绑定多模态到同一 embedding space
  - https://arxiv.org/abs/2305.05665
- **MoE / Switch Transformer** [Fedus et al., 2021] - sparse conditional computation
  - https://arxiv.org/abs/2101.03961
- **Mixture-of-Depths** [Raposo et al., 2024] - conditional computation per token
  - https://arxiv.org/abs/2404.02258
- **Modality Dropout** - 训练时随机 drop 某个 modality (类似 multimodal Co-training)
- **Highway Networks** [Srivastava et al., 2015] - gating mechanism 早期工作
  - https://arxiv.org/abs/1505.00387

### 6.7 Contact-Rich Manipulation Broader
- **Peg-in-hole insertion** 经典 task
- **Compliant control / Impedance control** [Hogan, 1985] - 经典 force control
- **CompliantVLA-Adaptor** [Zhang et al., 2026] - VLM-guided variable impedance
  - https://arxiv.org/abs/2601.15541
- **ForceVLA** [Yu et al., 2025] - force-aware MoE for VLA
  - https://arxiv.org/abs/2505.22159
- **Safe learning for contact-rich** [Zhang et al., 2025] - survey
  - https://arxiv.org/abs/2512.11908

---

## 7. Limitations & Future Directions (我自己加的延伸)

Paper 自己列了 3 个 limitation,我加几个延伸思考:

1. **Binary gating too rigid**: 应该考虑 learnable soft gating,参考 Mixture-of-Experts router 或 sigmoid gating。比如:

$$
\tilde{\mathbf{z}}_t^{tac} = \sigma(g(\mathbf{z}_t^{vis}, \mathbf{z}_t^{tac})) \cdot \mathbf{z}_t^{tac}
$$

其中 $g$ 是一个 small MLP 学一个 gating scalar。或者借鉴 **MoE routing** 的思路,让每个 tactile token 自己决定激活强度。

2. **Tactile spatial resolution limit**: 15×8 = 120 taxels 对 fine-grained contact geometry 不够。可以考虑 multi-scale tactile (coarse array + fine GelSight) 的 hierarchical encoding。

3. **Long-horizon tasks**: 当前只测 short-horizon disassembly。Long-horizon 任务里 tactile 信号的 temporal dynamics (slip detection, contact transition) 怎么建模?可能需要 **state-space models (Mamba, S4)** 或 **RNN-style memory** 来 capture temporal tactile evolution。

4. **Cross-embodiment generalization**: Franka 上的 tactile array layout 不通用,换 robot/sensor 怎么 transfer?需要 tactile representation 的 embodiment-agnostic design。

5. **Active sensing**: 当前 tactile 是被动感知,robot 应该 active 探索 contact (press, slide, probe) 来 gain information。这跟 **active perception** literature 接轨。

6. **Tactile-language grounding**: tactile signals 怎么映射到 language concepts (soft/hard, sticky/slippery, rough/smooth)?这是 **tactile-language pretraining** 的 open problem,类似 CLIP 但 for tactile。参考 **Touch-and-Go** [Yang et al., 2022]。

7. **Sim-to-real for tactile**: Tactile sensor 的仿真很难 (contact mechanics 复杂),这限制了大规模训练。**TACTO** simulator, **Taxim** 等是当前主流方向。

8. **Closed-loop force control**: 当前 TacVLA 是 open-loop action chunk prediction。真正 closed-loop force control 需要 high-frequency (1kHz+) tactile feedback loop,这跟 VLA 的 10Hz 频率 mismatch。可能需要 hybrid architecture: VLA 做高层 plan,low-level impedance controller 做高频 force tracking。这是 **CompliantVLA-Adaptor** [Zhang et al., 2026] 的思路。

9. **Causal modeling of contact events**: Tactile 信号本质是 event-based (contact onset, slip, release),应该用 **event-based representation** 而不是 dense frame-based。类似 **event camera** 的思路。

10. **Multi-finger tactile**: 当前只一个 finger 装 tactile array。Multi-finger dexterous manipulation (参考 Dex-Net, AnyTeleop) 需要全手 tactile coverage,token 数量会爆炸,需要 sparse attention 或 hierarchical pooling。

---

## 8. My Personal Take (Personal Speculation)

这篇 paper 整体非常 solid engineering work,把 tactile 模态以 minimal friction 加进 Pi0.5 backbone。几个 highlights:

1. **Contact-aware gating 是关键 insight**: 这个 idea 本质上是说 "modality 应该 condition on task state 激活"。这个思想可以推广: vision 在 occlusion 时应该 down-weight,language 在 ambiguous instruction 时应该 ask for clarification,proprioception 在 stuck 时应该 re-plan。整个 multimodal fusion 应该是 **state-conditional routing**,而不是 static concatenation。

2. **Pi0.5 backbone 的 flexibility**: LoRA fine-tune + frozen tactile encoder 这个 setup 让新模态接入变得 cheap。未来我们可以想象一个 "modality zoo" — vision, language, tactile, audio, force/torque, EMG, 等等 — 每个都有自己的 tokenizer 和 gating,通过 LoRA plug-and-play 进 VLM backbone。

3. **Training data efficiency**: 50 demos × 5 tasks = 250 demos 就能达到 70-100% success rate,这跟 Pi0.5 的 strong prior + LoRA fine-tune 的 few-shot 能力分不开。对比 diffusion policy 用同样数据 only 30-50%,差距巨大。这进一步确认 **pretrained VLM 是 robot learning 的 game changer**。

4. **Open question on gating signal**: 当前 $c_t$ 是 threshold-based heuristic,需要人工设计 threshold。能不能让模型自己 learn threshold? 或者用 **self-supervised contact detection** 从 tactile signal 自己学? 这跟 **anomaly detection** literature 接轨。

5. **Comparison to multimodal LLM in NLP**: 这篇 paper 的 contact-aware gating 跟 **multimodal LLM 中 audio-visual fusion** 的思路类似 — audio 不是一直 informative,要 condition on visual context (e.g., speaker active) 决定 audio 信号的重要性。Cross-modal gating 是 universal idea。

6. **Connection to robot skin**: 长远来看,robot 全身覆盖 tactile sensor (robot skin) 是趋势。这时候 token 数量会爆炸,这篇 paper 的 low-dim token + gating 设计正好 scalable。想象一个 humanoid robot 全身 10000 taxels,gating 把 active contact 区域以外的 tokens 都 mask 掉,computational cost 就 control 住了。

---

## 9. Key Formulas Recap

为了让你 build intuition,我把核心公式再总结一遍,带变量解读:

### Formula 1: Multimodal Token Sequence
$$
\tilde{\mathbf{z}}_t = [\mathbf{z}_t^{vis}, \mathbf{z}_t^{lan+pro}, \tilde{\mathbf{z}}_t^{tac}]
$$

- $\tilde{\mathbf{z}}_t$: 时间 $t$ 的完整 multimodal token sequence (作为 VLM 输入 prefix)
- $\mathbf{z}_t^{vis}$: 视觉 tokens (SigLIP 编码 front + wrist camera 图像)
- $\mathbf{z}_t^{lan+pro}$: language + proprioception tokens (PaliGemma tokenizer)
- $\tilde{\mathbf{z}}_t^{tac}$: gating 后的 tactile tokens (36 维)
- $[\cdot, \cdot, \cdot]$: token sequence concatenation

### Formula 2: Policy / Action Generation
$$
\mathbf{a}_{t:t+H} \sim \pi_\theta(\mathbf{a}_{t:t+H} \mid \tilde{\mathbf{z}}_t)
$$

- $\mathbf{a}_{t:t+H}$: 从 time $t$ 到 $t+H$ 的 action chunk (continuous,7-DoF + gripper)
- $H$: prediction horizon (Pi0.5 默认 50 steps)
- $\pi_\theta$: 参数为 $\theta$ 的 policy (VLM + action expert 组成)
- $\tilde{\mathbf{z}}_t$: 条件输入 (Formula 1 输出)
- $\sim$: 概率采样 (来自 flow matching 的 stochastic generation)

### Formula 3: Contact-Aware Attention Mask
$$
M_t^{tac} = c_t \cdot \mathbf{1}
$$

- $M_t^{tac}$: tactile tokens 的 attention mask (binary,长度等于 tactile token 数 = 36)
- $c_t \in \{0, 1\}$: contact flag (1 = 有 contact,0 = 无 contact)
- $\mathbf{1}$: 全 1 向量 (长度 36)
- 效果: $c_t=0$ 时 mask 全 0,tactile tokens 在 attention matrix 里完全 masked out

### Formula 4: Tactile Embedding Gating
$$
\tilde{\mathbf{z}}_t^{tac} = c_t \cdot \mathbf{z}_t^{tac}
$$

- $\tilde{\mathbf{z}}_t^{tac}$: gating 后送进 backbone 的 tactile tokens
- $c_t$: contact flag
- $\mathbf{z}_t^{tac}$: 原始 tactile tokens (MLP encoder + positional embedding)
- 效果: $c_t=0$ 时直接 zero out,避免 sensor noise 和 positional encoding 的 baseline offset

---

## 10. Web Links Reference

### Paper & Project
- TacVLA 项目页: https://sites.google.com/view/tacvla
- Pi0.5 paper: https://arxiv.org/abs/2504.16054
- Physical Intelligence blog: https://www.physicalintelligence.company/blog/pi0

### Code & Open Source
- OpenPI (Pi0.5 开源): https://github.com/Physical-Intelligence/openpi
- OpenVLA: https://github.com/openvla/openvla
- Diffusion Policy: https://github.com/real-stanford/diffusion_policy
- 3D Diffusion Policy: https://github.com/YanjieZe/3D Diffusion Policy

### VLM Backbones
- SigLIP: https://arxiv.org/abs/2303.15343
- PaliGemma: https://ai.google.dev/gemma/docs/paligemma
- Gemma family: https://ai.google.dev/gemma

### Tactile Sensors
- 3D-ViTac: https://arxiv.org/abs/2410.24091
- GelSight: http://gelsight.mit.edu/
- GelSight Wedge: https://arxiv.org/abs/2107.07206
- VibeCheck: https://arxiv.org/abs/2504.15535

### Related VLA-Tactile Papers
- TLA: https://arxiv.org/abs/2503.08548
- VTLA: https://arxiv.org/abs/2505.09577
- VLA-Touch: https://arxiv.org/abs/2507.17294
- Tactile-VLA: https://arxiv.org/abs/2507.09160
- OmniVTLA: https://arxiv.org/abs/2508.08706
- MLA: https://arxiv.org/abs/2509.26642
- Octopi-1.5: https://arxiv.org/abs/2507.09985

### Generative Models
- Flow Matching: https://arxiv.org/abs/2210.02747
- Stochastic Interpolants: https://arxiv.org/abs/2303.08797

### Multimodal Fusion & MoE
- ImageBind: https://arxiv.org/abs/2305.05665
- Switch Transformer: https://arxiv.org/abs/2101.03961
- Mixture-of-Depths: https://arxiv.org/abs/2404.02258
- LoRA: https://arxiv.org/abs/2106.09685

### Tactile Sim-to-Real
- TACTO: https://arxiv.org/abs/2012.08456
- Taxim: https://arxiv.org/abs/2104.03127

### Contact-Rich Manipulation Surveys
- Safe learning for contact-rich: https://arxiv.org/abs/2512.11908
- Imitation learning for contact-rich: https://arxiv.org/abs/2506.13498
- Multimodal fusion VLA survey: https://www.sciencedirect.com/science/article/pii/S1566253525001834

---

## 11. Summary Intuition

一句话总结: **TacVLA 把触觉当成 VLA 的"条件激活"模态,只在 contact 发生时让 tactile tokens 参与 cross-modal attention,避免 non-contact phase 的 noise 干扰,从而在 contact-rich + occlusion scenario 下显著优于 vision-only VLA 和 from-scratch diffusion policy。**

核心 insight 是: **modalities should be conditionally activated based on physical state, not statically fused.** 这个思想可以推广到很多其他场景 — audio 在 speaker active 时才重要,force/torque 在 contact 时才重要,EMG 在 human 合作时才重要 — 每个 modality 都有自己的 "physical context" 决定它的 relevance,模型应该 learn 这个 relevance function,而不是 naive concatenation。

希望这个拆解对你 build intuition 有帮助。如果对某个具体模块 (flow matching action expert, SigLIP encoding, contact detection threshold 设计, etc.) 想深入聊,可以继续问。
