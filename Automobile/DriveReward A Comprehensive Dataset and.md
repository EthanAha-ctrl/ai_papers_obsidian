---
source_pdf: DriveReward A Comprehensive Dataset and.pdf
paper_sha256: 16fdf0c22d51dd65c0b9cd2e44c29a519167ec47278e466c6b1c319d5a8b96c7
processed_at: '2026-08-18T07:01:04-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话讲讲 DriveReward

好嘞 Andrej，咱抛开学术八股，用最直白的话聊聊这篇paper到底干了啥、为啥这么干、哪里聪明哪里拉胯。

## 一句话总结

**他们把 driving 里 "给trajectory打分" 这件事，从 rule-based 计算器升级成了一个会先推理再打分的 VLM，并且用 counterfactual data 和 VGGT 蒸馏解决两个老大难问题，最后发现 1B 小模型在 narrow task 上能吊打 8B 大模型。**

## 问题在哪：为啥需要这么个东西

end-to-end driving 现在流行一个范式：生成 N 条候选 trajectory，然后挑一条最好的执行。这条 "挑" 的逻辑，老办法基本两种：

**第一种是写死的规则**。比如 NAVSIM 的 PDMS，它本质就是个计算器：用 perception GT 算一下有没有撞车、有没有越界、开了多远、comfortable 不，然后乘起来。问题显而易见——它需要昂贵的标注，perception 一有 noise 就崩，而且它只懂几何不懂语义。你闯红灯它不一定能识别，因为它不知道"红灯"这个概念。

**第二种是learned scoring head**。比如 HydraMDP 给 planner 挂个 score head，SparseDrive 用 BEV feature 打分。问题是这玩意儿和 planner 一起训，容易 overfit 到 dataset-specific 的 proxy metric 上，也没有 high-level reasoning。

所以 motivation 很清楚：**trajectory evaluation 需要 semantic reasoning，而 VLM 天生会这个**。你给 VLM 看张图，问它 "这条轨迹会不会闯红灯"，它能看懂红绿灯、车道线、车辆位置，然后给你答案。这跟 LLM 里用 GPT-4 当 reward model 是一回事。

https://arxiv.org/abs/2401.10020

## 他们怎么干：三个关键 idea

### Idea 1: Counterfactual data augmentation——给模型 "见世面"

driving dataset 几乎全是 "正常安全驾驶"。你拿这种数据训 reward model，它压根没见过 "坏 trajectory" 长啥样，怎么区分好坏？

他们的招数：从三个 source 拼候选 trajectory pool——GT、retrieval bank、planner 输出，然后用 retrieval-based 注入把 sub-optimal trajectory 灌进去。Fig. 5 显示 sub-optimal trajectory 比例显著提升，CF=0 的占 30%。

这个思路和 LLM 里 self-instruct、constitutional AI 一脉相承：让模型主动 "制造坏例子" 来扩充训练分布。OmniDrive、VILTA 也干过类似事，是同一拨人的套路。

https://arxiv.org/abs/2411.10150

### Idea 2: 新增两个 reward 维度——补 PDMS 的盲区

PDMS 有个明显漏洞：一条 trajectory 安全合法但完全走错方向，它照样给高分。比如 GT 要你右转进小路，你直行没变道——collision-free、legal，PDMS 照样 100 分，但其实你导航彻底失败了。

所以他们加了两个维度：

**CF (Command Following)**：看 trajectory endpoint 和 GT endpoint 的横向偏差，threshold 内给 1，超出给 0。简单粗暴但管用。

**LG (Legality)**：让 Qwen3.5 判断有没有闯红灯 / 压实线，给 hierarchical score：闯红灯=0，压实线=0.5，合规=1。这个 hierarchical 设计很聪明——压实线确实比闯红灯轻一档。

公式（Eq. 2）就是把这两个加进 PDMS：

$$\text{PDMS}_{\text{CFLG}} = NC \times DAC \times \frac{5 EP + 5 TTC + 2 C + 2 CF_{\text{pred}} + 2 LG_{\text{pred}}}{16}$$

变量：$NC$ 是无碰撞，$DAC$ 是可行驶区域合规，$EP$ 是行驶进度，$TTC$ 是碰撞时间，$C$ 是舒适度，$CF_{\text{pred}}$ 和 $LG_{\text{pred}}$ 是 reward model 预测的 CF 和 LG 分数。分母 16 是因为加了 2+2 的权重。

### Idea 3: VGGT 蒸馏——给 VLM 补几何脑子

VLM 的老毛病：semantic reasoning 强，spatial / geometric reasoning 弱。你让它判断 "trajectory 是不是压车道线"，它经常看走眼。

他们的解法很 hacky 但很实用：让 reward model 在输出 reasoning token 的同时，额外吐一组 geometry latent token $f_{\text{geo}} \in \mathbb{R}^{k \times d}$，通过 MLP adapter 投影到 VGGT 的 latent space，和 VGGT 真实输出做 L2 loss 对齐：

$$\mathcal{L}_{\text{geo}} = \| \text{Adapter}(f_{\text{geo}}) - \text{VGGT}(I_t) \|_2^2$$

VGGT 是 Oxford + Meta 的 geometry foundation model，能从单目图输出 dense 3D feature。本质上就是把 VGGT 当 teacher，把它对 3D 场景的理解蒸到 reward model 的 latent 里。

这招很关键，因为 reward model 推理时不需要 VGGT（只用学到的 latent），但训练时 VGGT 提供了 geometry supervision。inference 时 overhead 为零，很 deploy-friendly。

https://arxiv.org/abs/2503.08163

## 训练 pipeline：两段式很标准

**Stage 1**: 在 800k driving QA 上 full fine-tune，6 epochs，lr 4e-5。数据是 10 个 dataset 拼的（DriveLM、ReCogDrive、NuScenes-QA 等），目的是让 VLM 从 general-purpose 适配到 driving domain。

**Stage 2**: 在 300k DriveReward dataset 上 SFT，8 epochs，lr 4e-4（比 stage 1 大 10 倍）。freeze vision backbone，只调 LLM。

输出格式是 generative 的 CoT + answer：

```
<reasoning> ... 语义推理 ... </reasoning>
<answer> NC, DAC, TTC, EP, C, CF, LG </answer>
```

这和 DeepSeek-R1、Self-Rewarding LM 的格式一模一样。好处是 interpretable，坏处是 inference 慢，但 1B 模型在 H20 上没问题。

## 实验结果：几个关键数字

### Benchmark 上 1B 打 8B（Table 1）

zero-shot 的 InternVL3-8B 在 Safety 上是 64.2，SFT 后的 InternVL3-1B 跳到 72.2，完整 DriveReward-1B 到 80.6。这说明 reward modeling 是 narrow task，针对性 data + architecture 比 scale 管用。

### Closed-loop zero-shot transfer（Table 4）

这是最有说服力的实验。reward model 只在 nuPlan 上训，直接拿去 CARLA 的 Bench2Drive 做 RL reward，Driving Score 从 40.2 干到 51.4。hybrid reward（rule PDMS + RM CF/LG）能到 57.1，success rate 从 14.5% 到 27.2%。

zero-shot 跨 simulator 还能涨这么多，说明 reward model 学到的是 semantic understanding，不是 dataset-specific shortcut。

### Best-of-N selection 的 marginal gain（Table 5）

这是 paper 最诚实的弱点。oracle Best-of-4 能涨 2.7，DriveReward 只涨 0.2，capture 率 7.4%。作者自己都承认 "marginal"。

直觉上：在强 base policy 附近，几条 trajectory 的 score 差异很小，reward model 的 ranking 精度不够。这点 LLM 里也是——强模型附近 reward model 区分度会下降。可能要专门训 pairwise preference，或者用 contrastive objective。

https://arxiv.org/abs/2504.02160

## 和 LLM RLHF 的对应关系

我觉得这篇 paper 对你 Andrej 最有共鸣的角度是这个对应表：

| LLM RLHF | DriveReward |
|----------|-------------|
| Scalar reward head | Generative CoT + scores |
| Preference pair (DPO) | Multi-dim supervised scores |
| Reward hacking | PDMS 可被游戏但 CF/LG 补 |
| RLAIF (GPT-4 当 reward) | VLM 当 driving reward |
| Self-rewarding / Constitutional AI | Counterfactual augmentation |
| Best-of-N with reward model | Test-time trajectory selection |
| Knowledge distillation | VGGT geometry grounding |

LLM 社区里 reward model 已经从 scalar head 演进到 generative critic（LLaVA-Critic、Critique-out-Loud），driving 这篇走的也是同一条路。Gen-Drive 之前用 DPO 是 scalar / preference 路线，DriveReward 是 generative / CoT 路线，更 future-proof。

但 driving 比 LLM 难的点：

1. **Spatial reasoning**：LLM 输入是 token，driving 输入是 image + trajectory，VLM 几何能力弱，必须靠 VGGT 这种 external teacher 补
2. **Safety critical**：LLM 错了生成 nonsense，driving 错了撞车
3. **Real-time**：LLM 推理可以慢几秒，driving 必须 onboard 实时，所以选 1B 模型

## 我觉得最聪明的几个设计

1. **Visually-grounded annotation**：给 Qwen3.5 当 labeler 时，把 trajectory、碰撞点、越界点都画到图上再喂给它。这相当于把 "判断这条 trajectory 好不好" 这个开放问题，转化成 "看这些 annotation 告诉我 reasoning + score" 这个 grounded QA。没有这个 trick，VLM labeler 质量肯定崩。

2. **Hybrid reward**：不强行用 RM 替代 rule-based，而是让 RM 补 rule-based 的盲区（CF、LG）。这是对 deployment 现实的妥协——能算 rule-based 的环境就用 rule-based + RM addon，不能算的就直接用 RM predicted PDMS。

3. **CF 和 LG 的 hierarchical 设计**：CF 是 binary（走对走错），LG 是 ternary（闯红灯 / 压实线 / 合规）。这种 granularity 区分反映了真实世界的 severity 差异，比一刀切的 binary 好太多。

## 我觉得可以更好的地方

1. **Best-of-N 的 marginal gain**：7.4% capture 率太低。应该专门训 ranking objective，或者 pairwise preference，或者 contrastive。
2. **没有 uncertainty estimation**：reward model over-confidence 会 hurt RL，[Uncertainty-Aware Reward Model](https://arxiv.org/abs/2410.00847) 已经证明了。DriveReward 没考虑这点。
3. **没做 online RM update**：RL 过程中 policy 会 shift，frozen RM 容易 stale。LLM 里已经有人做 iterative DPO。
4. **单 front-view**：没用 multi-view / LiDAR。SparseDrive、VAD 都证明 multi-view 对 geometry 帮助大，未来扩展应该直接受益。
5. **权重 choice 缺 sensitivity analysis**：CF/LG 权重设 2 是 heuristic，paper 没扫这个超参。

## 给你的 intuition

如果让我给你一句话 build intuition：

**这篇 paper 本质上是在说——driving trajectory evaluation 这个 task，与其继续纠结 rule-based 的精度，不如把它 reformulate 成一个 VLM 擅长的 grounded visual reasoning 问题，然后用 LLM RLHF 社区已经验证过的三招（counterfactual data、CoT generation、foundation model distillation）把 1B 小模型调到能打 8B 大模型，最后发现这个 reward signal 既能反哺 RL 又能 zero-shot transfer 到陌生 simulator。**

核心 insight：**reward model 不一定要 "更精确"，而要 "更理解"。VLM 给的 reward 虽然 noise 比 rule-based 大，但它的 semantic 维度是 rule-based 永远学不到的，所以 hybrid 才是王道**。

对你最可能有启发的是 VGGT distillation 那块——把 foundation model 的 geometric prior 蒸到 VLM latent 里，inference 时 teacher 不参与。这个 pattern 应该能推广到 robotics manipulation、embodied AI 一堆场景。比如把 Depth Anything 蒸到 VLA 的 latent 里做 depth-aware action generation。

https://arxiv.org/abs/2401.10020
https://arxiv.org/abs/2504.02160
https://arxiv.org/abs/2503.08163
https://arxiv.org/abs/2411.10150
https://arxiv.org/abs/2504.10479
https://arxiv.org/abs/2411.03765
https://arxiv.org/abs/2406.03844
https://arxiv.org/abs/2406.06978
https://arxiv.org/abs/2503.23129
https://arxiv.org/abs/2506.08052
https://arxiv.org/abs/2509.13769
https://arxiv.org/abs/2312.14130
https://arxiv.org/abs/2410.00847

---

# DriveReward 深度解析

非常开心和你聊这篇paper，Andrej。这篇工作本质上是把 RLHF 里 reward model 的范式迁移到 autonomous driving 的 trajectory evaluation 上，但做了很多 driving-specific 的工程化改造。让我从 motivation、data、architecture、training、experiments 几个角度展开。

## 1. 核心问题与 motivation

End-to-end driving 最近流行 multi-modal trajectory generation（如 DiffusionDrive、GoalFlow、HydraMDP），即生成 N 条候选 trajectory 再选一条。selection 的好坏直接决定 safety 和 efficiency。同时 VLA 模型（如 AdaThinkDrive、AutoVLA、ReCogDrive）需要 RL fine-tuning，这又需要可靠的 reward signal。

现有 evaluation 范式有两大类，paper 在 Fig. 1 里画得很清楚：

**(a) Model-free / rule-based**: 比如 NAVSIM 的 PDMS，用 perception GT 算 NC、DAC、TTC、EP、Comfort，再乘起来。问题：依赖昂贵的 dense annotations，cascading perception noise，无法 scale，也 capture 不了 high-level semantic rules（比如闯红灯、压实线）。

**(b) Model-based**: 比如 SparseDrive 用 BEV feature scoring，HydraMDP 学一个 evaluator head，WOTE / MindDrive 引入 world model 预测未来状态。问题：还是 dataset-specific proxy metric（MSE、PDMS），缺 semantic reasoning，且 jointly train with planner 容易 overfit。

**(c) DriveReward 的思路**: 用 VLM 直接看图 + 读 trajectory + 读 ego state，输出 7 维 reward。VLM 既有 visual understanding 又有 semantic reasoning，能 bypass perception labels。

这个思路和 LLM 里 self-rewarding / critique-as-reward（如 [Self-Rewarding LM](https://arxiv.org/abs/2401.10020)、[LLaVA-Critic](https://arxiv.org/abs/2504.02160)）很接近，但 driving 有两个独特挑战：第一，failure case 稀缺；第二，spatial geometry reasoning 难。paper 针对这两点都给了工程方案。

Reference: 
- [NAVSIM](https://arxiv.org/abs/2411.03765)
- [Bench2Drive](https://arxiv.org/abs/2406.03844)
- [LLaVA-Critic](https://arxiv.org/abs/2504.02160)
- [Self-Rewarding LM](https://arxiv.org/abs/2401.10020)

## 2. DriveReward Dataset 的构造

### 2.1 数据来源与 candidate trajectory pool

base data 是 NAVSIM train set（nuPlan 子集）。对每一帧，从三个 source 构造 trajectory candidates：

1. **GT trajectory**（original dataset）
2. **Retrieval from offline kinematic bank**（ego-state-based retrieval，提供 diversity）
3. **Planner outputs**（mitigate human-algorithm domain gap，避免 distribution shift）

这种 multi-source candidate 构造和 [HydraMDP](https://arxiv.org/abs/2406.06978) 的 hydra-distillation 思路很像——单一 source 会让 reward model 学到 spurious correlation。

### 2.2 Counterfactual augmentation

这是这篇 paper 的关键一招。NAVSIM/Bench2Drive 这类 dataset 都是 safe nominal driving，failure case 是 long-tail。reward model 如果只在 "safe trajectory" 上训，discriminative 能力会很弱。paper 通过 retrieval-based trajectory injection 系统性地引入 sub-optimal / infraction 场景。从 Fig. 5 的分布看，sub-optimal trajectory 的比例显著提升，CF=0 的 trajectory 大约占 30%，LG violation 相对少（因为闯红灯 / 压实线本来就是 long-tail）。

这个思路和 [OmniDrive](https://arxiv.org/abs/2411.10150) 的 counterfactual reasoning dataset、[VILTA](https://arxiv.org/abs/2601.12672)（顺便这篇也是同作者群）的 "learning from failures" 是一脉相承的。和 RLHF 里用 LLM 生成 preference pair 的 RLAIF 思路也相通。

### 2.3 7 个 reward dimensions

paper 定义了 7 个评估维度，其中 5 个沿用 NAVSIM protocol，2 个是新增：

- **NC** (No Collision)
- **DAC** (Drivable Area Compliance)
- **TTC** (Time-to-Collision)
- **EP** (Ego Progress)
- **C** (Comfort)
- **CF** (Command Following) —— 新增
- **LG** (Legality) —— 新增

**CF 的 motivation**：考虑一个分岔路口，GT intent 是右转进 side road，但某条 trajectory 安全地直行没变道。在传统 PDMS 下这条 trajectory 拿高分（collision-free、legal），但完全违背导航意图。CF 用 predicted trajectory endpoint 和 GT trajectory endpoint 的 lateral deviation 做二值判断（threshold 内为 1，否则 0）。这是个非常合理的补丁——PDMS 只看 "安全但不看对不对"。

**LG 的 motivation**：评估红灯 / 压实线。不用 rigid geometric heuristics，直接用 Qwen3.5-35B-A3B 做 semantic reasoning，输出 hierarchical score：闯红灯=0，压实线=0.5，合规=1。这种 hierarchical scoring 反映了 severity difference，是个很好的设计 choice。

### 2.4 Automated labeling pipeline（Fig. 2）

这是 paper 工程上最有意思的部分。他们用 Qwen3.5-35B 做 labeler，但直接喂图给 VLM 让它判断 trajectory 好 bad 是 unreliable 的（VLM 对 spatial / temporal reasoning 弱）。于是他们做 **visually-grounded annotation**：

- 把 planned trajectory 投影到 front-view image 上（绿线）
- collision 位置标红点
- drivable area 偏离点标品红点
- 涉事车辆 bounding box 蓝框
- 偏离路区域黄框

这些 visual prompt 是 rule-based 确定性计算出来的，给 VLM 提供 reliable spatial-temporal grounding。这相当于把 "你判断这条 trajectory 好不好" 这个开放问题，转化成 "你看图上这些 annotation，告诉我 reasoning + score"。

这个 trick 很关键，我怀疑如果没有 visual prompt，Qwen3.5 的 label 质量会崩。本质上是用 deterministic geometric reasoning 来 anchor VLM 的 semantic reasoning。类似的思路在 [DriveLM](https://arxiv.org/abs/2312.14130)、[OmniDrive](https://arxiv.org/abs/2411.10150) 里也见过，叫 "graph visual QA" 或 "counterfactual visual grounding"。

Reference:
- [OmniDrive](https://arxiv.org/abs/2411.10150)
- [DriveLM](https://arxiv.org/abs/2312.14130)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)

## 3. DriveReward Model 架构

### 3.1 Base architecture

base 是 InternVL3-1B（轻量，便于 onboard deployment）。输入 tuple：

$$\mathbf{Q}_t = (I_t, L)$$

其中：
- $I_t \in \mathbb{R}^{H \times W \times 3}$：当前帧 front-view image
- $L$：text input，包含 navigation instruction $l$（如 "Turn left"）、ego state $s_{\text{ego}} \in \mathbb{R}^{d_s}$（velocity, acceleration）、待评估 trajectory $h_{\text{fut}} \in \mathbb{R}^{T \times 3}$

这里 $T$ 是 trajectory horizon，$3$ 是 $(x, y, \theta)$ 或 $(x, y, t)$ 维度（paper 没明确，但 NAVSIM 标准是后者）。

### 3.2 Geometry Grounding module

这是 architecture 上最 novel 的设计。问题：VLM 对 spatial / geometric reasoning 弱（"这条 trajectory 是否压车道线" 这类判断 VLM 容易错）。

paper 的方案：让 reward model 显式输出一个 auxiliary geometric feature $f_{\text{geo}} \in \mathbb{R}^{k \times d}$，然后通过一个 MLP-based Geometry Adapter 投影到 VGGT 的 latent space，和 VGGT 的 dense representation 做 $L_2$ loss 对齐：

$$\mathcal{L}_{\text{geo}} = \| \text{Adapter}(f_{\text{geo}}) - \text{VGGT}(I_t) \|_2^2$$

这里 VGGT 是 [Visual Geometry Grounded Transformer](https://arxiv.org/abs/2503.08163)（Oxford & Meta，CVPR 2025），它是个 geometry foundation model，能从单目 image 输出 dense 3D feature / depth / pose。

这是个 "feature distillation from foundation model" 的思路：把 VGGT 的几何 prior 蒸到 reward model 的 latent 里。不需要 dense 3D GT annotation，但让 reward model 内化 spatial constraint。

直觉上：reward model 看图的时候，不仅要回答 "trajectory 安不安全" 这个 semantic 问题，还要在 latent 里把 3D scene geometry 重建出来——后者由 VGGT 提供 supervision。这让 reward model 对 "trajectory 是否越界" 这种几何判断更可靠。

类似的思路在 [LFD (Learning from Demonstration)](https://arxiv.org/abs/1810.02917)、[DistillBERT](https://arxiv.org/abs/1910.01108)、还有 various VLM distillation works 里都见过。这里把 geometry foundation model 当 teacher 是个很自然的选择。

### 3.3 Reward generation：纯生成式

paper 明确说 "completely bypassing the need for auxiliary MLP regression heads"。输出是 autoregressive next-token prediction：

```
<reasoning> ...semantic CoT... </reasoning>
<answer> NC, DAC, TTC, EP, C, CF, LG </answer>
```

这种 "CoT + answer" 的输出格式和 DeepSeek-R1、[Self-Rewarding LM](https://arxiv.org/abs/2401.10020) 完全一致。好处：
1. **Interpretability**：能 debug 为什么这么打分
2. **Compositionality**：7 个 score 共享一个 reasoning context，避免 7 个 head 各自为政
3. **Scalability**：加新维度不用改 architecture，只改 prompt / dataset

代价：inference 慢（要生成 CoT），但 1B model 在 H20 上是 OK 的。

### 3.4 Two-stage training（Fig. 3）

**Stage 1: Domain-specific pre-training**

在 800k QA 上 full-parameter fine-tune，6 epochs，lr 4e-5。pre-train data 是 10 个 dataset 聚合：

- Spatial/multi-modal scene understanding：DriveLM, ReCogDrive, NuScenes-QA, SUTD, Talk2Car
- Temporal：LingoQA, NuInstruct
- Vision-action alignment：Senna, OmniDrive, DriveGPT4

这个 pretrain 阶段目的：让 VLM 从 general-purpose 适配到 driving domain，掌握 driving-specific reasoning。

**Stage 2: Task-specific SFT**

在 300k DriveReward dataset 上 fine-tune，8 epochs，lr 4e-4（比 stage 1 大 10x），gradient accumulation = 1。

注意：stage 2 只用 original image（无 visual prompt），因为 reward model 推理时是不带 visual prompt 的。pretrain 阶段的 visual prompt 只是给 labeler Qwen3.5 用的。

Training 时 freeze vision backbone，只 fine-tune LLM layers。这是常见做法，保留 pre-trained visual representation。

Reference:
- [InternVL3](https://arxiv.org/abs/2504.10479)
- [VGGT](https://arxiv.org/abs/2503.08163)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

## 4. Experiments 解析

### 4.1 DriveReward-Bench（Table 1）

benchmark 是 5k NAVSIM test samples，评估 5 个 dimension（去掉 TTC 和 Comfort，理由：TTC 被 NC proxied，Comfort 几乎全 1）。

关键观察：
- **Zero-shot 模型 scale correlation 强**：InternVL3 2B → 8B，Safety 从 29.6 → 64.2；Qwen3.5 2B → 4B，Legality 从 80.6 → 78.8（4B 反而略低，可能 overfitting on instruction tuning data）
- **SFT + task-specific data 是 game changer**：InternVL3-1B SFT 后 Safety 从 ~30（zero-shot 2B 的 29.6）跳到 72.2，1B 模型反超 8B zero-shot
- **DriveReward-1B 全面最优**：Safety 80.6, DAC 81.2, CF 99.9, LG 97.6, EP-MAE 0.23

这说明：reward modeling 是个 narrow task，1B 模型 + 针对性 data + architecture 改造（geometry adapter）能 beat 大模型 zero-shot。这点对实用部署很重要。

### 4.2 RL fine-tuning - Open-loop（Table 2）

base policy: InternVL3-2B, InternVL3-8B, DiffusionDriveV2。RL 用 DriveReward-1B 预测的 PDMS 作为 reward。结果：

| Base | Method | PDMS |
|------|--------|------|
| InternVL3-2B | SFT | 80.8 |
| InternVL3-2B | RL-RM (1B) | 82.2 (+1.4) |
| InternVL3-8B | SFT | 85.6 |
| InternVL3-8B | RL-RM (1B) | 86.8 (+1.2) |
| DiffusionDriveV2 | SFT | 89.1 |
| DiffusionDriveV2 | RL-RM (1B) | 90.7 (+1.6) |

RL 用 model-predicted reward 也能带来稳定提升。这相当于 RLHF 里用 reward model 替代 human preference——只不过这里 reward model 也是 VLM，policy 也是 VLM，形成 "VLM RL VLM" 的闭环。

### 4.3 RL fine-tuning - Closed-loop Bench2Drive（Table 4）

这里更激动人心，因为 Bench2Drive 是 CARLA-based closed-loop benchmark，和 NAVSIM 的 distribution 完全不同——所以这是 **zero-shot transfer** 测试。

| Method | Driving Score | Success Rate |
|--------|---------------|--------------|
| Base SFT | 40.2 | 14.5 |
| RL (RM-Predicted PDMS) | 51.4 | 20.8 |
| RL (Rule-Based PDMS) | 55.0 | 25.0 |
| RL (Rule-PDMS + RM CF&LG) | 57.1 | 27.2 |

几个观察：

1. **只用 RM-predicted PDMS**，从 40.2 → 51.4，说明 reward model 在完全 OOD 的 simulator 上还能给出有效信号。这是 generalization 的强证据。

2. **Rule-based PDMS 仍然 better**（55.0 vs 51.4），因为 rule-based 是 ground truth，model prediction 有 noise。

3. **Hybrid（rule PDMS + RM CF/LG）最好**（57.1）。这里 RM 不是替代 rule，而是补 rule 的盲区（CF 和 LG rule-based 难算）。这是 paper 最 practical 的结论：在能算 rule-based PDMS 的环境里，RM 作为 "addon" 提升 semantic dimensions；在不能算的环境里，RM 直接作为 reward。

复合 reward 的公式（Eq. 2）：

$$\text{PDMS}_{\text{CFLG}} = NC \times DAC \times \left( \frac{5 \cdot EP + 5 \cdot TTC + 2 \cdot C + 2 \cdot CF_{\text{pred}} + 2 \cdot LG_{\text{pred}}}{16} \right)$$

变量含义：
- $NC$：No Collision 二值
- $DAC$：Drivable Area Compliance 二值
- $EP$：Ego Progress（连续）
- $TTC$：Time-to-Collision
- $C$：Comfort
- $CF_{\text{pred}}$：reward model 预测的 Command Following（0 或 1）
- $LG_{\text{pred}}$：reward model 预测的 Legality（0 / 0.5 / 1）

分母从 12（原 PDMS）变 16，因为新增了 $2 \cdot CF_{\text{pred}} + 2 \cdot LG_{\text{pred}}$，权重 2 和 Comfort 同级。这个权重 choice 是 heuristic，paper 没做 sensitivity analysis，是个可以再 explore 的点。

### 4.4 Test-time trajectory selection（Table 5）

在 AdaThinkDrive 上做 Best-of-4 selection：
- Original: PDMS 90.3
- Best-of-N oracle: 93.0 (+2.7)
- DriveReward: 90.5 (+0.2)

老实说这个结果 marginal 得厉害——只 capture 了 oracle 上界的 0.2/2.7 ≈ 7.4%。paper 自己也承认 "the performance gain achieved by the reward model is relatively marginal"。

这说明：在强 base policy 上，multi-modal trajectory 之间的 score 差异很小，reward model 的 ranking 能力不够精细。对比 LLM 里 Best-of-N with reward model 通常能 capture 大部分上界，driving 这里的 reward landscape 更崎岖。

可能的改进方向：
- 用 pairwise preference data 训（Gen-Drive 思路 + DPO）
- 加 contrastive learning objective，专门训 ranking
- 更细粒度的 trajectory representation（不只是 $(x,y,\theta)$ sequence）

### 4.5 Ablations（Table 6）

| Ablation | NC | DAC | CF | LG | EP-MAE |
|----------|----|-----|----|----|--------|
| DriveReward-1B | 80.58 | 81.24 | 99.86 | 97.64 | 0.2308 |
| w/o pre-training | 80.52 | 81.06 | 99.84 | 97.48 | 0.2355 |
| w/o reasoning CoT | 71.26 | 73.60 | 100.00 | - | 0.4161 |
| w/o 3D adapter | 79.04 | 80.88 | 99.86 | 97.20 | 0.2518 |

关键发现：

1. **Reasoning CoT 影响最大**：去掉 CoT 后 NC 掉 9 分，EP-MAE 从 0.23 飙到 0.42。说明 CoT 是 reward model 性能的主要 driver。这点和 [Chain-of-Thought Hub](https://arxiv.org/abs/2305.15005)、reasoning model 的实证一致——explicit reasoning 让 model 把 intermediate computation 卸载到 token space。

2. **3D adapter（geometry grounding）影响中等**：去掉后 NC 掉 1.5，DAC 掉 0.36。geometry grounding 确实帮 spatial judgment，但不是 magic bullet。

3. **Pre-training 影响小**：只掉了 0.06-0.16。但 paper 说 "consistent further gains"，可能是 1B 模型本身已经 fit，pretrain 的 gain 主要在更大 base 上体现。

Reference:
- [Chain-of-Thought Hub](https://arxiv.org/abs/2305.15005)
- [Gen-Drive](https://arxiv.org/abs/2503.23129)

## 5. 限定性与可改进方向

paper 自己列了两个 limitation：

1. **单 front-view camera**：没用 LiDAR / multi-view。从 [SparseDrive](https://arxiv.org/abs/2503.11158)、[VAD](https://arxiv.org/abs/2303.12077) 的经验看，multi-view 对 geometry reasoning 帮助大，未来扩展应该直接收益。

2. **Domain gap with real deployment**：dataset 缺 fine-grained routing instruction（GPS-guided lane change、highway exit）。NAVSIM 只有 coarse maneuver command，CF metric 也是 coarse-grained。real deployment 需要更细的 navigation hierarchy。

我能想到的额外 limitation 和探索方向：

- **Reward model uncertainty estimation**：[Uncertainty-Aware Reward Model](https://arxiv.org/abs/2410.00847) 显示 reward model 的 over-confidence 会 hurt RL。DriveReward 没有不确定性估计，RL training 可能被 noisy reward 误导。
- **Online reward model update**：目前是 offline 训好就 frozen。RL 过程中 policy 会 shift，reward model 应该跟着 update（类似 [Iterative DPO](https://arxiv.org/abs/2305.18290)）。
- **Trajectory representation richness**：现在是 $(x, y, \theta)$ sequence。如果用 tokenized trajectory（如 [Wayformer](https://arxiv.org/abs/2310.14643) 的 motion token）可能让 reward model 更好理解。
- **Best-of-N selection 的 marginal gain**：这点最值得深挖。可能需要 reward model 专门训 pairwise preference（[Reward Model as Critic](https://arxiv.org/abs/2504.02160) 思路）。
- **VGGT 计算开销**：inference 时 reward model 不需要 VGGT（只用 latent），但 training 时 VGGT forward 是个大开销。考虑用更轻的 geometry teacher（如 [DUSt3R](https://arxiv.org/abs/2312.14132)）。

## 6. 与 LLM RLHF 的类比

我觉得这篇 paper 最有启发的是它把 RLHF 的范式迁移到了 driving：

| LLM RLHF | DriveReward |
|----------|-------------|
| Preference data (preference pair) | Multi-dimensional reward annotation |
| Reward model (BERT-like scalar head) | VLM generative CoT + scores |
| PPO with KL penalty | PPO with rule-based reward + RM CF/LG |
| Reward hacking | Trajectory that games PDMS but violates CF/LG |
| Best-of-N sampling | Test-time trajectory selection |

LLM 社区里 reward model 已经从 scalar head 演进到 generative critic（[Critique-out-Loud](https://arxiv.org/abs/2406.04102)、[LLaVA-Critic](https://arxiv.org/abs/2504.02160)），driving 这里也走了同样的路。Gen-Drive 之前用 DPO 是 scalar / preference 路线，DriveReward 是 generative / CoT 路线，是更 future-proof 的选择。

但 driving 比 LLM 难的点：
1. **Spatial / geometric reasoning**：LLM 输入是 token，driving 输入是 image + trajectory，spatial grounding 难
2. **Safety critical**：LLM 错了生成 nonsense，driving 错了出事故
3. **Real-time constraint**：LLM 推理可以慢，driving 必须 onboard 实时

paper 的 1B model + VGGT distillation 是对 (1) 和 (3) 的回答。对 (2) 还没完全回答——reward model 本身的 reliability 没有被证明到 safety-grade。

## 7. 和其他 driving VLA / reward 工作的关系

- **Gen-Drive** ([Huang et al., ICRA 2025](https://arxiv.org/abs/2503.23129))：用 DPO 训 reward model。DriveReward 用 supervised CoT，更 explicit、更 interpretable。
- **ReCogDrive** ([Li et al., 2025](https://arxiv.org/abs/2506.08052))：同作者群的 prior work，用 RL fine-tune VLA。DriveReward 可以看作 ReCogDrive 的 reward model 部分。
- **AdaThinkDrive** ([Luo et al., 2025](https://arxiv.org/abs/2509.13769))：同作者群，adaptive thinking for VLA。DriveReward 的 test-time selection 实验就是基于它。
- **HydraMDP** ([Li et al., NeurIPS 2024](https://arxiv.org/abs/2406.06978))：multi-target hydra-distillation。DriveReward 的 multi-dimensional reward 思路类似，但用 VLM 替代 learned head。
- **VILTA** ([Chen et al., 2026](https://arxiv.org/abs/2601.12672))：同作者群，VLM-in-the-loop adversary。reward model 和 adversary model 是两面：一个 reward good behavior，一个 attack weak behavior。

可以看出 Xiaomi EV + Tsinghua 这群人沿着 "VLM as X for driving" 这条线做了 system-level 的布局：reward、adversary、planner、thinker，形成完整 stack。

## 8. 总结

DriveReward 的核心贡献：

1. **Concept**: 把 VLM generative reward model 范式带到 driving，bypass perception annotation dependency
2. **Data**: counterfactual augmentation 解决 long-tail failure case 稀缺，新增 CF/LG 两个 dimensions 补 PDMS 盲区
3. **Architecture**: geometry grounding via VGGT distillation + generative CoT 输出
4. **Empirical**: 1B 模型 beat 8B zero-shot，且 zero-shot transfer 到 CARLA closed-loop 有效

它的 marginal contribution 在 test-time selection 上确实小，但作为 RL reward signal 的价值更清晰——尤其在 hybrid reward（rule-based + RM CF/LG）的 setup 下。

对你 Andrej 可能最 interesting 的 angle：
- 这是 "LLM as reward model" 在 physical-world embodied task 的一次认真尝试
- VGGT distillation 是把 foundation model prior 注入 VLM 的一种新方式，可能可以推广到 robotics、manipulation
- Counterfactual augmentation 和 RLAID / Constitutional AI 思路是相通的
- Best-of-N 的 marginal gain 说明 reward landscape 在 strong policy 附近很 flat，这是个 open problem

如果你想要 intuition 上的一个一句话总结：**DriveReward 把 driving trajectory evaluation 重构成一个 VLM 能回答的 grounded visual QA 问题，并通过 counterfactual data + geometry distillation 让 1B 模型在 narrow reward task 上超越 8B 大模型，再把这个 reward signal 反哺 RL 和 test-time selection**。

希望这个 deep dive 对你 build intuition 有帮助！如果需要我对某个 module 再深入（比如 VGGT 怎么做 feature projection，或者 NAVSIM 的 PDMS 完整 formula derivation），可以告诉我。

Reference 总览：
- [NAVSIM](https://arxiv.org/abs/2411.03765)
- [Bench2Drive](https://arxiv.org/abs/2406.03844)
- [InternVL3](https://arxiv.org/abs/2504.10479)
- [VGGT](https://arxiv.org/abs/2503.08163)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [HydraMDP](https://arxiv.org/abs/2406.06978)
- [SparseDrive](https://arxiv.org/abs/2503.11158)
- [DiffusionDrive](https://arxiv.org/abs/2411.15249)
- [DiffusionDriveV2](https://arxiv.org/abs/2512.07745)
- [ReCogDrive](https://arxiv.org/abs/2506.08052)
- [AdaThinkDrive](https://arxiv.org/abs/2509.13769)
- [OmniDrive](https://arxiv.org/abs/2411.10150)
- [DriveLM](https://arxiv.org/abs/2312.14130)
- [Gen-Drive](https://arxiv.org/abs/2503.23129)
- [LLaVA-Critic](https://arxiv.org/abs/2504.02160)
- [Self-Rewarding LM](https://arxiv.org/abs/2401.10020)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [VILTA](https://arxiv.org/abs/2601.12672)
- [AutoVLA](https://arxiv.org/abs/2506.13757)
- [VAD](https://arxiv.org/abs/2303.12077)
- [GoalFlow](https://arxiv.org/abs/2503.07295)
- [DUSt3R](https://arxiv.org/abs/2312.14132)
- [Uncertainty-Aware Reward Model](https://arxiv.org/abs/2410.00847)
