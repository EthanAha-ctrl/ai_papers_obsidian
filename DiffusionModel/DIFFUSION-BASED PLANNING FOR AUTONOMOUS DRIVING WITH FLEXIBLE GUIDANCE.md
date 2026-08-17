---
source_pdf: DIFFUSION-BASED PLANNING FOR AUTONOMOUS DRIVING WITH FLEXIBLE GUIDANCE.pdf
paper_sha256: e723259989e123cee3803af4b95f00e59ce271bf3c47082ad804b2d08921aa95
processed_at: '2026-08-03T21:51:41-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Diffusion Planner

好，我换一种讲法。前面那篇太"论文解读"了，这次像我们在 white board 前面聊天那样讲。

---

## 这篇 paper 一句话在干嘛

**用 diffusion model 来做自动驾驶的轨迹规划，而且不需要任何 rule-based 的后处理兜底，在 nuPlan 上跑出了 SOTA。**

就这一句话。剩下全是细节。

---

## 先说背景：learning-based planning 为什么一直很拉胯

你想，behavior cloning 这件事，本质上就是：给我一个场景，我回归出一条人类会开的轨迹。

听起来简单，但它有三个根本性的坑，这些年一直没填上。

**第一个坑：multi-modality。**

你在路口，左边可以左转，右边可以右转，前面可以直行。三条都合理。但你用 MSE loss 去回归，模型学到的是三条的平均值——往中间斜着开。这条"平均轨迹"哪条都不是，物理上还不一定可行。

这个问题在 motion prediction 领域早就被讨论烂了，大家用 multi-head prediction、用 winner-takes-all、用 GMM 来搞。但 planning 领域一直很尴尬，因为你 closed-loop 跑的时候只能输出一条，不能像 prediction 那样给你六条让你自己选。

**第二个坑：OOD 一旦犯错，没有 recover 能力。**

IL 模型见过什么场景就会开什么场景。一旦遇到训练分布外的输入，它输出的轨迹可能乱七八糟。然后 error 累积，越开越偏，最后撞墙。

所以现有方法几乎都有个 rule-based 的 fallback：模型输出多条轨迹，用 rule 来打分、过滤、refine。比如 [PlanTF](https://arxiv.org/abs/2310.16924)、[PLUTO](https://arxiv.org/abs/2404.14327)、[GameFormer](https://arxiv.org/abs/2303.05760)，全都是这样。GameFormer 你去掉 refinement 直接只剩 13 分，惨不忍睹。

这就很尴尬——你号称是 learning-based，结果你的 safety 全靠 rule-based 兜底。那 learning 到底干了什么？

**第三个坑：多目标打架。**

你想让模型又安全又舒适又高效，就加 safety loss、comfort loss、efficiency loss。这三个梯度方向经常冲突。模型夹在中间，哪个都学不好。更糟糕的是，IL 学的是"模仿人类"，不是"从错误中纠正"——模型根本不知道犯了错之后该怎么救回来。

**第四个坑（其实是个隐藏问题）：训练完就定型了。**

你想让一辆车开保守一点，或者跟车近一点，或者高速路上快点，市区慢点——对不起，重新训练。没有 runtime 的可调性。

这四个坑，diffusion model 恰好每一个都能填上。这就是这篇 paper 的 motivation。

---

## 为什么 diffusion 能解决这四个坑

我们一个一个说。

### Multi-modality：这是 diffusion 的天然属性

Diffusion model 学的是 score function $\nabla_x \log q(x)$，本质上是学整个数据分布，而不是 regression 到均值。

参考 [Diffusion Policy (Chi et al., 2023)](https://arxiv.org/abs/2303.04137) 在 robot manipulation 上早就验证过这件事——diffusion 能 naturally 输出多模态 action。Paper 里 Figure 5 做了个很直观的实验：同一位置多次采样，没有 navigation 信息时，模型真的能输出左转、右转、直行三种轨迹，且各自清晰分离。给上 navigation 之后，又准准地按指示走。

这是 MSE-based IL 根本做不到的。

### OOD 与 recovery：靠 data augmentation 教会"从 perturbed state 回归"

这个其实不是 diffusion 本身的功劳，是 training trick 的功劳，但配合 diffusion 效果特别好。

做法借鉴 [ChauffeurNet (Bansal et al., 2018)](https://arxiv.org/abs/1812.03079)：把 ego 当前状态加随机扰动，然后用 quintic polynomial 插值到 ground truth future，生成一条物理可行的"恢复轨迹"作为新 label。这样模型学到的就是"从各种奇怪状态回到正常轨迹"的能力。

Ablation 里这个 augmentation 去掉之后直接掉 12 分（76.53 vs 89.19），是所有 ablation 里最狠的。说明 OOD 问题是 closed-loop planning 的头号杀手。

### 多目标打架：用 classifier guidance 而不是 auxiliary loss

这是整个 paper 最漂亮的地方，我下面单独讲。

### Runtime 可调：也是 classifier guidance 的功劳

同上，下面展开。

---

## Classifier guidance——这才是 paper 的灵魂

我先讲直觉，再讲数学。

**直觉版本**：

你已经训练好一个 diffusion model，它能采样出"人类会开的轨迹"。现在你想要的是"人类会开的轨迹，但是不要撞车"。

怎么改？两种思路：

**思路 A**：训练的时候就把 collision loss 加进去，让模型学一个"不会撞车的分布"。问题是你想加 comfort、想加 speed preference，都得重训。每加一个 preference 就重训一次，根本不实用。

**思路 B**：训练时不动，让模型保持"人类驾驶分布"。在 inference 时，每次 denoise 一步，检查当前轨迹的能量（比如 collision 能量、comfort 能量），算梯度，把轨迹往"能量低"的方向推一下。这就叫 classifier guidance。

思路 B 的厉害之处：**所有偏好都在 runtime 加，不重训，随便组合**。

**数学版本**：

目标分布写成：

$$
p_0(\mathbf{x}^{(0)}) \propto q_0(\mathbf{x}^{(0)}) \cdot e^{-\mathcal{E}(\mathbf{x}^{(0)})}
$$

$q_0$ 是训练时学到的"人类驾驶分布"，$e^{-\mathcal{E}}$ 是额外的偏好因子。$\mathcal{E}$ 越大表示越不想要，概率被压低。

Diffusion 采样时，每一步都要算 score $\nabla \log p_t$。用 Bayes 公式展开：

$$
\nabla_{\mathbf{x}^{(t)}} \log p_t(\mathbf{x}^{(t)}) = \nabla_{\mathbf{x}^{(t)}} \log q_t(\mathbf{x}^{(t)}) - \nabla_{\mathbf{x}^{(t)}} \mathcal{E}(\hat{\mathbf{x}}^{(0)})
$$

右边第一项是模型已经学好的 score。第二项是 guidance gradient，关键 trick 在——$\hat{\mathbf{x}}^{(0)}$ 直接用 diffusion model 当前步输出的 $\mu_\theta$ 来近似（这叫 [Tweedie's formula](https://arxiv.org/abs/2209.14687)，[Diffusion Posterior Sampling](https://arxiv.org/abs/2209.14687) 的核心）。

这样 guidance 的梯度完全 training-free，只需要写个可微的 $\mathcal{E}$ 就行。

**具体 energy function 长什么样？**

举两个例子：

**Collision**：
$$
\mathcal{E}_{\text{collision}} = \sum_{M, \tau} \Psi\left(\omega_c \cdot \max\left(1 - \frac{D_M^\tau}{r}, 0\right)\right)
$$

$D_M^\tau$ 是 ego 和第 M 个邻居在第 τ 步的 signed distance，$r$ 是 sensitive distance。距离小于 $r$ 时能量开始升高，距离 0 时能量很大。$\Psi(x) = e^x - x - 1$ 是个平滑的 hinge，避免硬阈值导致梯度不连续。

**Target speed**：
$$
\mathcal{E}_{\text{target\_speed}} = \max(\bar{v} - v_{\text{low}}, 0)^2 + \max(v_{\text{high}} - \bar{v}, 0)^2
$$

只在速度超出 $[v_{\text{low}}, v_{\text{high}}]$ 区间时才有惩罚，区间内梯度为 0。

**Comfort**：
$$
\mathcal{E}_{\text{comfort}} = \mathbb{E}\left[\max\left((j_{\max} - |\text{jerk}|)\Delta\tau^3, 0\right)^2\right]
$$

jerk 超过 $j_{\max}$ 才有惩罚。

**Drivable area**：
用 Euclidean Signed Distance Field 构造 cost map $\mathbf{M}$，ego 越出 lane 时 $\mathbf{M}(x_{\text{ego}}^\tau)$ 越大，能量越高。

**为什么这些 energy function 设计得很讲究？**

Paper 在 Appendix C.3 给了四条经验法则，我翻译一下：

1. **梯度要平滑**：用 $\Psi$ 这种 smooth surrogate，不要 hard hinge，否则采样不稳定。
2. **梯度要稀疏**：只在出问题时才有梯度。Comfort 没超阈值时梯度 0，collision 距离够远时梯度 0。否则梯度一直 pull，反而让轨迹变形。
3. **高阶量要间接 guide**：想控速度，guide 轨迹长度；想控加速度，guide position 曲率。不要直接对 $\mathrm{d}^2 x / \mathrm{d}t^2$ 求梯度——数值不稳。
4. **梯度 magnitude 要一致**：不同 energy 之间梯度尺度可能差 100 倍，要 normalize。Paper 在分母加 count + $\epsilon$ 来拉平。

这四条本质上是 multi-objective optimization 的梯度尺度协调问题，跟 RL 里的 reward shaping 是同一个套路。

**组合性是最爽的**：

因为 $e^{-\mathcal{E}_1} \cdot e^{-\mathcal{E}_2} = e^{-(\mathcal{E}_1 + \mathcal{E}_2)}$，多个 guidance 直接把 energy 相加就行。Collision + drivable + comfort 同时上，不需要调权重。

Figure 2 给了一个特别好的 case：后方来车要追尾。
- 只加 collision guidance：ego 冲出 lane 逃命，违反 drivable。
- 加 collision + drivable guidance：ego 在 lane 内避让，既安全又合规。

这就是 guidance 的可组合性。

参考：[Classifier Guidance (Dhariwal & Nichol, 2021)](https://arxiv.org/abs/2105.05233) 最初是 image generation 上的工作，[Diffusion Posterior Sampling (Chung et al., 2022)](https://arxiv.org/abs/2209.14687) 把它扩展成 training-free 的形式，让任何 inverse problem 都能用。

---

## 架构上有什么聪明的地方

主体是 [DiT (Diffusion Transformer, Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748) 的变体。聪明点在 condition 怎么注入。

**聪明点 1：joint modeling ego + neighbors**

公式 (4) 把 ego 和 M 个邻居的未来轨迹 stack 成一个 tensor，一起做 diffusion。这样模型学的是联合分布 $q(\mathbf{x}_{\text{ego}}, \mathbf{x}_{\text{neighbor}_1}, \ldots | C)$。

直觉上：planning 和 prediction 在 closed-loop 里是耦合的。你 plan 了变道，邻居会反应；邻居要并线，你得让。分开两个 head 训练，interaction 信息被割裂。Joint 之后，cooperative behavior 自然 emerge。

**聪明点 2：丢掉 ego velocity / acceleration**

Ablation 里显示：把 ego 当前 velocity、acceleration 喂进 decoder，性能从 89 掉到 78。为什么？因为 IL 模型会走 shortcut——直接拿 current velocity × dt 外推 future position，根本不去学真正的 planning。

这是 closed-loop planning 的著名 failure mode，[PlanTF](https://arxiv.org/abs/2310.16924) 和 [Is Ego Status All You Need (Li et al., 2024)](https://arxiv.org/abs/2403.04595) 都讨论过。Solution 就是只保留 position + heading，把 velocity 和 acceleration 直接砍掉。

**聪明点 3：MLP-Mixer 处理 vectorized map**

Lane 和 neighbor history 都是 vectorized 表示，比如 lane 是 20 个点 × 12 维 feature。这种数据本征 sparse——20 个点很多是冗余采样。如果直接用 attention，浪费容量。

[MLP-Mixer (Tolstikhin et al., 2021)](https://arxiv.org/abs/2105.01601) 是个 all-MLP 架构，在 point 维度 和 feature 维度上各 mix 一次。先稠密化，再让 transformer 在稠密表示上做 cross-attention。Paper 说这比 [GameFormer](https://arxiv.org/abs/2303.05760) 那种复杂结构设计更简洁。

**聪明点 4：Navigation 用 AdaLN，context 用 cross-attention**

Navigation 是"全局意图"（要走哪条路），应该在每一层都全局调制 feature 的 normalization 参数——用 [AdaLN (Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748)。

Context（neighbor、lane）是"局部约束"（怎么走），用 cross-attention 注入更合适。

两种 condition 走不同路径，符合它们在语义上的角色。这是架构上很 intuitive 的设计。

---

## 实验上发生了什么有意思的事

### 主表（Table 1）

几个关键数字：

- **Diffusion Planner 不加 refine**：Test14 NR 89.19，Test14-hard NR 75.99。这是 pure learning-based SOTA，超过 PlanTF（85.62 / 69.70）和 PLUTO w/o refine（89.90 / 70.03）。

- **Diffusion Planner 加 refine**：Test14 NR 94.80，**超过 Log-replay expert 的 94.03**。Learning 方法超过 human demonstration，在 nuPlan 上还是头一次。

- **GameFormer 不加 refine 只有 13 分**，惨烈。说明 game-theoretic 显式建模在 closed-loop 上未必比 diffusion 的隐式 joint modeling 强。

- **Reactive mode 退化**：Diffusion Planner 从 89.87 掉到 82.80，掉 7 分。PLUTO 从 88.89 掉到 78.11，掉 11 分。Diffusion 的 reactive 鲁棒性更好。我推测是 joint modeling 让 model 学到了邻居的 reactive 行为分布。

### Delivery-vehicle 数据集（Table 2）

这是 paper 的 bonus——Haomo.AI 收集了 200 小时配送车数据。配送车尺寸 1.03m × 2.34m，比 nuPlan 车（2.30m × 5.18m）小一半，开在 bike lane 上，行人 / 自行车交互密集。

观察：Diffusion Planner 92.08 > PlanTF 90.89，差距比 nuPlan 上更大。我推测是配送场景 multi-modal 决策更多（避让行人可能左绕可能右绕），diffusion 优势更明显。

PDM-Hybrid 在 nuPlan 上 92.77，到 delivery 上掉到 80.72。因为 PDM 的 reference line 强依赖 nuPlan 道路结构，场景一换就废。说明 Diffusion Planner 的 transferability 来自架构本身，没有 nuPlan-specific 的 inductive bias。

### Inference 速度（Table 4）

跟其他 diffusion-based planning 比：

| 方法 | Val14 | Inference time |
|---|---|---|
| [Diffusion-ES](https://arxiv.org/abs/2402.06559) w/o LLM | 50 | - |
| Diffusion-ES w/ LLM | 92 | 0.5s |
| [STR2-CPKS-800M](https://arxiv.org/abs/2410.15774) w/o refine | 65.16 | >11s |
| **Diffusion Planner** | **89.87** | **0.04s** |

Diffusion-ES 离开 LLM 就废，STR2 用 800M 参数 + 11s inference 工程上不可用。Diffusion Planner 用 0.04s 达到 89 分。这才是能上车的算法。

---

## 我个人的几点直觉和联想

### 1. Diffusion 在 decision-making 上的真正优势不是 multi-modality，是 guidance

很多人讲 diffusion 在 planning 上的优势，都强调 multi-modality。我觉得这只是表层。

真正的杀手锏是 **classifier guidance 的 training-free composability**。

传统 IL 想加一个新 preference，要么改 loss 重训，要么用 rule-based 后处理。Diffusion 让你写个可微 energy function，runtime 注入，多 preference 任意组合，梯度自动算好。这是 RL 想做但做不到的——RL 改 reward 要重训，diffusion 改 energy 不用重训。

参考 [Diffusion Policy](https://arxiv.org/abs/2303.04137) 在 robot 上其实没怎么用 guidance，纯靠 diffusion 的 multi-modality。Diffusion Planner 把 guidance 用起来，这才是 diffusion 在 decision-making 上的真正 power。

### 2. Joint modeling 是不是真学到了 game-theoretic equilibrium？

这其实是个挺深的问题。

Diffusion 学的是 $q(\mathbf{x}_{\text{ego}}, \mathbf{x}_{\text{neighbor}} | C)$，即联合轨迹分布。在 reactive simulation 中，neighbor 的行为由 simulator 给定（基于 ego 当前 plan 反应）。Diffusion 采样出的 joint trajectory 中，neighbor 部分其实是 model 自己的"预测"，跟 simulator 的 reactive neighbor 不一定一致。

但实验表明这样仍然 work。原因可能是 diffusion 每 0.1s 重新采样，会自适应 simulator 的 reactive behavior。closed-loop 的本质就是每步重新 condition，diffusion 的 iterative refinement 跟这个天然契合。

[GameFormer](https://arxiv.org/abs/2303.05760) 显式做 game-theoretic 交互，但 w/o refine 只有 13 分。说明显式 game theory 未必比 diffusion 的隐式 joint modeling 强。这点值得思考。

### 3. 跟 Decision Transformer 的对比

[Decision Transformer (Chen et al., 2021)](https://arxiv.org/abs/2106.01345) 把 RL 转成 sequence modeling，用 return-to-go 条件化生成。Diffusion Planner 把 planning 转成 trajectory generation，用 classifier guidance 条件化生成。两者都是"用生成模型解决决策问题"，但：

- DT 用 autoregressive，error 累积；diffusion 用 parallel denoising，每步都是 global refinement。
- DT 的 condition 是 scalar return，diffusion 的 condition 是任意可微 energy function——表达能力强得多。

### 4. Sample efficiency 的下一步

10 步 denoise × 0.04s 已经很快，但传统 planner 是单次 forward。要把 diffusion 真正上量产车，下一步肯定是 [Consistency Model (Song et al., 2023)](https://arxiv.org/abs/2303.01469)，一步采样。或者 [guided distillation (Meng et al., 2023)](https://arxiv.org/abs/2302.04855)，把 guided diffusion 蒸馏成单步模型，同时保留 guidance 能力。

这是把 diffusion planning 从研究推向 product 的关键路径。

### 5. Lateral flexibility 的 dataset bias 问题

Author 提到 model 不太会大幅度 lateral movement（变道、避让）。原因是 nuPlan 训练数据中 lane change 场景稀疏。这跟 [PlanTF](https://arxiv.org/abs/2310.16924) 的 observation 一致——dataset bias 是 IL-based planning 的根本痛点。

可能的解法：
- 用 lateral-movement guidance 强制 pull 模型做变道。
- RL fine-tuning，用 reward signal 教 lane change。
- 合成 lane change 场景做 data augmentation。

我觉得 guidance 这条路最 promising，因为它最符合这篇 paper 的 philosophy——用 energy function 注入 capability，而不是改训练。

---

## 最后总结

如果让我用三句话讲这篇 paper 的精髓：

1. **把 planning + prediction 统一成一个 multi-agent trajectory diffusion 任务**，让 cooperative behavior 自然 emerge，避免多 head 之间的 loss 冲突。

2. **把 safety / comfort / speed preference 写成可微 energy function，在 inference 时通过 classifier guidance 注入**。Training-free，可组合，可调，这是 diffusion 在 decision-making 上独有的能力。

3. **架构上四个小心思**——丢 ego velocity 防 shortcut、MLP-Mixer 稠密化 vectorized map、AdaLN 注入 navigation、DPM-Solver + low-temperature 让 inference 0.04s。每个都不复杂，但合起来让 diffusion 第一次在 closed-loop planning 上不依赖 rule-based refinement 就达到 SOTA。

后续如果 consistency model 把 10 步压到 1 步，加上 vision encoder 替换 vectorized input，我觉得这是真正能上车的 architecture。

---

## 几个关键 reference

- [Diffusion Planner 项目主页](https://zhengyinan-air.github.io/Diffusion-Planner/)
- [nuPlan benchmark](https://arxiv.org/abs/2106.11810)
- [DDPM 原始 paper](https://arxiv.org/abs/2006.11239)
- [Score-based SDE (Song et al., 2021)](https://arxiv.org/abs/2011.13456)
- [Classifier Guidance (Dhariwal & Nichol, 2021)](https://arxiv.org/abs/2105.05233)
- [Diffusion Posterior Sampling (Chung et al., 2022)](https://arxiv.org/abs/2209.14687)
- [DPM-Solver (Lu et al., 2022)](https://arxiv.org/abs/2206.00927)
- [DiT (Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748)
- [MLP-Mixer (Tolstikhin et al., 2021)](https://arxiv.org/abs/2105.01601)
- [Diffusion Policy (Chi et al., 2023)](https://arxiv.org/abs/2303.04137)
- [PlanTF](https://arxiv.org/abs/2310.16924)
- [PLUTO](https://arxiv.org/abs/2404.14327)
- [GameFormer](https://arxiv.org/abs/2303.05760)
- [ChauffeurNet (augmentation trick)](https://arxiv.org/abs/1812.03079)
- [Consistency Models](https://arxiv.org/abs/2303.01469)
- [Decision Transformer](https://arxiv.org/abs/2106.01345)
- [Diffusion-ES](https://arxiv.org/abs/2402.06559)
- [STR2 / MoE Planner](https://arxiv.org/abs/2410.15774)
- [Is Ego Status All You Need](https://arxiv.org/abs/2403.04595)

Andrej，这次尽量讲人话了。核心 idea 其实特别 elegant——把 safety 和 personalization 都变成 inference-time 的 gradient surgery，避开 IL + auxiliary loss 那一整套"训练时把所有东西塞进 loss"的老路。我觉得这是 diffusion 进入 production-grade decision-making 的标志性工作之一。

---

# Diffusion Planner 深度讲解

Andrej 你好，这篇 paper 是 Tsinghua + Haomo.AI 的工作，核心理念非常 clean：**把 closed-loop planning 重新定义为一个 joint multi-agent trajectory generation 任务，用 diffusion score matching 学习多模态轨迹分布，并通过 classifier guidance 在 inference 时灵活调整 driving behavior**。整个方法几乎没有 rule-based refinement，这一点在 nuPlan 上是相当罕见的。我尽量从底层原理、架构细节、实验直觉三个层面把直觉 build 起来。

---

## 1. Motivation 的本质问题

Learning-based planner 在 closed-loop 上有几个老毛病，这篇 paper 精准对准：

1. **Multi-modality 问题**：human driving 在同一场景下有多个合理解（左转 / 直行 / 右转）。Behavior cloning 用 MSE / L1 loss 拟合 $\mathbb{E}[y|x]$，会 collapse 到 mean，产生"均值轨迹"——这在 multimodal 分布下既不属于任何 mode，也不 physically feasible。
2. **OOD fragility**：IL 一旦遇到 OOD 输入，误差累积，必须有 rule-based fallback 兜底。但 rule-based fallback 又违背了用 learning 取代规则的初衷。
3. **Multi-objective conflict**：safety、comfort、efficiency 三个目标用 auxiliary loss 写在一起，梯度互相打架。模型学不会"如何从 mistake 中 recover"。
4. **Post-training 不可调**：训练完了，你想让它"开保守一点"或者"跟车近一点"，只能重新训练。

Diffusion model 在这四点上都给出优雅的答案：score function 天然 capture multimodal；classifier guidance 在 inference 时按需注入偏好，无需重训；diffusion 的 score-based 视角与 Energy-Based Model 同构，让 safety / comfort 这类 cost 直接变成能量函数注入。

---

## 2. Diffusion 数学背景回顾（为了 build intuition）

### 2.1 Forward process 与 score function

公式 (1) 定义前向加噪：

$$
q_{t0}(\mathbf{x}^{(t)} | \mathbf{x}^{(0)}) = \mathcal{N}(\mathbf{x}^{(t)} | \alpha_t \mathbf{x}^{(0)}, \sigma_t^2 \mathbf{I})
$$

变量解释：
- $\mathbf{x}^{(0)} \in \mathbb{R}^{(M+1) \times \tau \times d}$：clean trajectory tensor，包含 ego + M 个 neighbor 的 τ 步未来轨迹，每步 d 维状态（这里是 4 维：x, y, sin θ, cos θ）。
- $\mathbf{x}^{(t)}$：在 diffusion time $t \in [0,1]$ 上的 noisy 版本。$t=0$ 是 clean，$t=1$ 是纯噪声。
- $\alpha_t = \sqrt{1 - \sigma_t^2}$：signal retention coefficient，确保 $t \to 1$ 时分布收敛到 $\mathcal{N}(0, \mathbf{I})$。
- $\sigma_t$：noise scale schedule（这里用 linear VP schedule，$\beta_{min}=0.1, \beta_{max}=20.0$）。

### 2.2 Diffusion ODE 视角（这是整个方法的核心视角）

公式 (2) 的 Probability Flow ODE：

$$
\mathrm{d}\mathbf{x}^{(t)} = \left[ f(t)\mathbf{x}^{(t)} - \frac{1}{2}g^2(t) \nabla_{\mathbf{x}^{(t)}} \log q_t(\mathbf{x}^{(t)}) \right] \mathrm{d}t
$$

其中：
- $f(t) = \frac{\mathrm{d}\log\alpha_t}{\mathrm{d}t}$：drift coefficient，控制 deterministic signal decay。
- $g^2(t) = \frac{\mathrm{d}\sigma_t^2}{\mathrm{d}t} - 2 \frac{\mathrm{d}\log\alpha_t}{\mathrm{d}t} \sigma_t^2$：diffusion coefficient，控制噪声注入速率。
- $\nabla_{\mathbf{x}^{(t)}} \log q_t(\mathbf{x}^{(t)})$：**score function**——这才是 diffusion model 真正学的东西。

**直觉**：score function 指向"概率密度增加最快的方向"。在轨迹空间里，score 告诉你"如何把当前噪声轨迹推回 high-density 区域（即合理的人类驾驶轨迹）"。

训练目标（公式 5）实际拟合的是 $\mu_\theta$，而 score 可以从 $\mu_\theta$ 反算：

$$
\mathbf{s}_\theta = \frac{\alpha_t \mu_\theta - \mathbf{x}^{(t)}}{\sigma_t^2}
$$

这是 DDPM/DDIM 中标准的 $\epsilon$-prediction 到 score 的转换。$\alpha_t \mu_\theta$ 是去噪后的 clean 估计，$\mathbf{x}^{(t)} - \alpha_t \mu_\theta$ 是估计的噪声，除以 $\sigma_t^2$ 得到归一化的 score。

参考：[Score-Based Generative Modeling through SDEs (Song et al., 2021)](https://arxiv.org/abs/2011.13456)

---

## 3. Task Redefinition——这是最关键的 insight

公式 (4) 重新定义了 planning：

$$
\mathbf{x}^{(0)} = \begin{bmatrix} x_{\text{ego}}^{(0)} \\ x_{\text{neighbor}_1}^{(0)} \\ \vdots \\ x_{\text{neighbor}_M}^{(0)} \end{bmatrix} = \begin{bmatrix} x_{\text{ego}}^1 & x_{\text{ego}}^2 & \cdots & x_{\text{ego}}^\tau \\ x_{\text{neighbor}_1}^1 & x_{\text{neighbor}_1}^2 & \cdots & x_{\text{neighbor}_1}^\tau \\ \vdots & \vdots & \ddots & \vdots \\ x_{\text{neighbor}_M}^1 & x_{\text{neighbor}_M}^2 & \cdots & x_{\text{neighbor}_M}^\tau \end{bmatrix}
$$

注意符号约定：
- **带括号的上标** $(0), (t)$：diffusion denoising time（即 noise level）。
- **不带括号的上标** $1, 2, \ldots, \tau$：future trajectory time step（这里 τ=8s × 10Hz = 80 步）。
- 每行是一个 agent 的整段 future，每列是一个 time step。

**为什么要 joint modeling？**

直觉上：planning 和 prediction 在 closed-loop 中是耦合的——ego 的 plan 影响 neighbor 的 future，neighbor 的 prediction 反过来约束 ego 的可行域。如果分开用两个 head 训练，interaction 信息被割裂。把它们 stack 成一个 tensor，让 diffusion 在联合分布 $q(\mathbf{x}_{\text{ego}}, \mathbf{x}_{\text{neighbor}_1}, \ldots, \mathbf{x}_{\text{neighbor}_M} | C)$ 上采样，cooperative behavior 自然 emerge。

这让我联想到 [Scene Transformer (Ngiam et al., 2021)](https://arxiv.org/abs/2106.08417) 的 joint prediction 思路，但 Scene Transformer 是用 autoregressive + 聚类的方式建模 multi-modality，diffusion 的优势是不需要显式指定 mode 数量。

---

## 4. 架构详解（Figure 1 解析）

整个架构是 **DiT (Diffusion Transformer)** 的变体，关键在于"如何把 condition $C$ 注入 noise trajectory $\mathbf{x}^{(t)}$"。我把它拆成四个融合模块：

### 4.1 Vehicle Information Integration

第一步：把 noisy future trajectory $\mathbf{x}^{(t)}$ 与**当前 state** $x^0 = [x_{\text{ego}}^0, x_{\text{neighbor}_1}^0, \ldots, x_{\text{neighbor}_M}^0]^T$ concat 起来，作为 decoder 的初始 token。

**关键细节**：ego vehicle 的 velocity 和 acceleration 被显式排除。这点有讲究——之前 [PlanTF (Cheng et al., 2023)](https://arxiv.org/abs/2310.16924) 和 [Is Ego Status All You Need (Li et al., 2024)](https://arxiv.org/abs/2403.04595) 都发现 IL 模型会走 shortcut：直接用 current velocity × dt 外推 future position，不去学真正的 planning。这是 closed-loop planning 的一个著名 failure mode。

Ablation（Table 3）佐证：
- w/ ego state（包含 velocity, accel, yaw rate）：78.65（严重退化）
- w/ SDE (state dropout encoder)：82.90（mitigate 但不够）
- 不加 ego state，只加 position + heading：**89.19**（最好）

### 4.2 Historical Status & Lane Information Fusion (MLP-Mixer)

每个 neighbor 表示为 $\dot{S}_{\text{neighbor}} \in \mathbb{R}^{L \times D_{\text{neighbor}}}$，每条 lane 表示为 $S_{\text{lane}} \in \mathbb{R}^{P \times D_{\text{lane}}}$。

- $L = 21$：过去 2 秒 × 10Hz + 当前 = 21 timestamps。
- $D_{\text{neighbor}} = 11$：x, y, heading sin, heading cos, vx, vy, length, width, category, ...
- $P = 20$：每条 lane polyline 20 个点。
- $D_{\text{lane}} = 12$：x, y, heading, speed limit, traffic light status, ...

**为什么用 MLP-Mixer 而不是 attention？**

公式 (6)：
$$
S = S + \text{MLP}(S^T)^T, \quad S = S + \text{MLP}(S)
$$

第一项在 vector dimension（point 数量维）上 mix，第二项在 feature dimension 上 mix。MLP-Mixer 的优势是**信息稠密化**：vectorized map 输入本征 sparse（很多 polyline point 都是冗余采样），attention 容易浪费容量。Mixer 先把 sparse 的多个 point 融合到一个稠密 vector embedding，再让 transformer 在稠密表示上做 cross-attention，效率更高。

参考：[MLP-Mixer (Tolstikhin et al., 2021)](https://arxiv.org/abs/2105.01601) | [VectorNet (Gao et al., 2020)](https://arxiv.org/abs/2005.04259)

### 4.3 Cross-Attention Fusion

公式 (7)：
$$
\mathbf{x} = \mathbf{x} + \text{MHCA}(\mathbf{x}, Q_f), \quad \mathbf{x} = \mathbf{x} + \text{FFN}(\mathbf{x})
$$

$Q_f$ 是 encoder 输出（neighbor + lane + static object 的聚合表示），作为 cross-attention 的 K, V；$\mathbf{x}$（noisy trajectory token）作为 Q。这是 standard DiT 的 condition injection 模式，但注意 $Q_f$ 不包含 navigation——navigation 单独走另一路。

### 4.4 Navigation Information Fusion via AdaLN

Navigation $S_{\text{route}} \in \mathbb{R}^{(K \times P) \times D_{\text{route}}}$ 通过另一个 MLP-Mixer 提取 $Q_n$。$Q_n$ 与 diffusion timestep embedding $Q_t$ 相加，送入 **Adaptive Layer Norm (AdaLN)**。

AdaLN 的核心（来自 [DiT (Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748)）：

$$
\text{AdaLN}(h, c) = \gamma(c) \cdot \text{LayerNorm}(h) + \beta(c)
$$

其中 $\gamma, \beta$ 是由 condition $c = Q_n + Q_t$ 通过 MLP 产生的 scale 和 shift。这样 navigation 和 diffusion timestep 全局调制每一层的 normalization 参数——**这是 diffusion model 中最有效的 condition 注入方式之一**。

**直觉**：navigation 是"全局意图"（去哪条路），它应该在每一层都影响 feature 的尺度；而 neighbor / lane 是"局部 context"（怎么走），用 cross-attention 注入更合适。两种 condition 走不同路径，符合它们在语义上的角色。

---

## 5. Classifier Guidance——这是整个方法的灵魂

### 5.1 数学推导

目标分布：

$$
p_0(\mathbf{x}^{(0)}) \propto q_0(\mathbf{x}^{(0)}) \cdot e^{-\mathcal{E}(\mathbf{x}^{(0)})}
$$

直觉：$q_0$ 是训练时学到的"人类驾驶分布"，$e^{-\mathcal{E}}$ 是一个额外的偏好因子，把不想要的轨迹能量推高（概率压低）。最终采样分布是两者乘积。

在 diffusion 过程的任意时间 $t$，应用 Bayes：

$$
\nabla_{\mathbf{x}^{(t)}} \log p_t(\mathbf{x}^{(t)}) = \nabla_{\mathbf{x}^{(t)}} \log q_t(\mathbf{x}^{(t)}) - \nabla_{\mathbf{x}^{(t)}} \mathcal{E}\left(\mathbb{E}_{q_{0t}(\mathbf{x}^{(0)}|\mathbf{x}^{(t)})}[\mathbf{x}^{(0)}]\right)
$$

公式 (8) 关键的简化：用 $\mu_\theta(\mathbf{x}^{(t)}, t, C)$ 作为 $\mathbb{E}[\mathbf{x}^{(0)}|\mathbf{x}^{(t)}]$ 的近似（即 Tweedie's formula）。

$$
\nabla_{\mathbf{x}^{(t)}} \log p_t(\mathbf{x}^{(t)}) \approx \nabla_{\mathbf{x}^{(t)}} \log q_t(\mathbf{x}^{(t)}) - \nabla_{\mathbf{x}^{(t)}} \mathcal{E}(\mu_\theta(\mathbf{x}^{(t)}, t, C))
$$

这就是 **Diffusion Posterior Sampling (DPS)** 的核心 trick。参考 [Chung et al., 2022](https://arxiv.org/abs/2209.14687) 和 [Xu et al., 2025](https://openreview.net/forum?id=GcvLoqOoXL)。

**为什么这是 training-free？** 因为 $\mu_\theta$ 是已经训练好的 diffusion model 的输出，对 $\mathbf{x}^{(t)}$ 自动有梯度（只要 forward pass 可微）。我们只需要写一个可微的 energy function $\mathcal{E}$，就能在 inference 时做 gradient surgery。

### 5.2 四种 energy function 的具体形式

#### (a) Collision Avoidance (公式 9)

$$
\mathcal{E}_{\text{collision}} = \frac{1}{\omega_c} \cdot \frac{\sum_{M,\tau} \mathbb{1}_{\mathbf{D}_M^\tau > 0} \cdot \Psi(\omega_c \cdot \max(1 - \frac{\mathbf{D}_M^\tau}{r}, 0))}{\sum_{M,\tau} \mathbb{1}_{\mathbf{D}_M^\tau > 0} + \epsilon} + \frac{1}{\omega_c} \cdot \frac{\sum_{M,\tau} \mathbb{1}_{\mathbf{D}_M^\tau < 0} \cdot \Psi(\omega_c \cdot \max(1 - \frac{\mathbf{D}_M^\tau}{r}, 0))}{\sum_{M,\tau} \mathbb{1}_{\mathbf{D}_M^\tau < 0} + \epsilon}
$$

变量：
- $\mathbf{D}_M^\tau$：ego 与第 M 个 neighbor 在第 τ 个 future step 的 signed distance（正值=分离，负值=重叠）。
- $r$：collision-sensitive distance（梯度只在距离 < r 时非零，sparsity 设计）。
- $\omega_c$：gain 系数，控制梯度 magnitude。
- $\Psi(x) := e^x - x - 1$：smoothed hinge-like function，确保梯度连续可微。
- 两项分别处理"接近"和"重叠"两种情况，分母归一化保证梯度 magnitude 一致。

#### (b) Target Speed (公式 10)

$$
\mathcal{E}_{\text{target\_speed}} = \max\left(\overline{\frac{\mathrm{d}x_{\text{ego}}^\tau}{\mathrm{d}\tau}} - v_{\text{low}}, 0\right)^2 + \max\left(v_{\text{high}} - \overline{\frac{\mathrm{d}x_{\text{ego}}^\tau}{\mathrm{d}\tau}}, 0\right)^2
$$

- $v_{\text{low}}, v_{\text{high}}$：目标速度区间。
- $\overline{\frac{\mathrm{d}x_{\text{ego}}^\tau}{\mathrm{d}\tau}}$：planned trajectory 的平均纵向速度。
- 两侧 max + squared 形成区间外的 quadratic penalty，区间内 gradient 为 0（sparsity）。

**直觉**：这是 piecewise penalty，只在越界时激活。注意它 guide 的是 $\mathrm{d}x/\mathrm{d}\tau$ 而不是直接显式速度——通过 position 间接影响速度，避免对 high-order derivative 直接求导导致数值不稳。

#### (c) Comfort (公式 11)

$$
\mathcal{E}_{\text{comfort}} = \mathbb{E}\left[\max\left((j_{\max} - |\frac{\mathrm{d}^3 x_{\text{ego}}^\tau}{\mathrm{d}\tau^3}|)\Delta\tau^3, 0\right)^2\right]
$$

- $j_{\max}$：longitudinal jerk 上限（如 5 m/s³）。
- $\frac{\mathrm{d}^3 x_{\text{ego}}^\tau}{\mathrm{d}\tau^3}$：jerk = position 的三阶时间导数。
- $\Delta\tau^3$：discrete time step cube，做数值归一化。

#### (d) Drivable Area (公式 12)

$$
\mathcal{E}_{\text{drivable}} = \frac{1}{\omega_d} \cdot \frac{\sum_\tau \Psi(\omega_d \cdot \mathbf{M}(x_{\text{ego}}^\tau))}{\sum_\tau \mathbb{1}_{\mathbf{M}(x_{\text{ego}}^\tau) > 0} + \epsilon}
$$

- $\mathbf{M}$：Euclidean Signed Distance Field（ESDF），离 lane 越远值越大，在 lane 内为 0 或负。
- 通过并行计算构造，参考 [PLUTO (Cheng et al., 2024)](https://arxiv.org/abs/2404.14327)。

### 5.3 Energy function 设计的 4 条经验法则（Appendix C.3）

作者给出非常实用的设计原则，我特别 highlight 一下：

1. **Smooth and continuous gradients**：避免 hard hinge，用 $\Psi(x) = e^x - x - 1$ 这类 smooth surrogate。
2. **Gradient sparsity**：只在出问题时产生梯度。Collision 只在距离 < r 时激活；comfort 只在超阈值时激活。否则梯度会一直 pull，导致 over-correction。
3. **Indirect guidance for higher-order derivatives**：想控速度，guide trajectory length；想控加速度，guide position curvature。直接对 $\mathrm{d}^2 x/\mathrm{d}t^2$ 求梯度容易 numerically unstable。
4. **Consistent gradient magnitude**：用 normalization term（分母的 count + $\epsilon$）把梯度 magnitude 拉到一致 scale。否则不同 energy 之间会"打架"——collision 梯度可能比 comfort 梯度大 100 倍，comfort guidance 就失效了。

这四条本质上是**多目标优化的梯度尺度协调**，跟 RL 中 reward shaping 是同一类问题。

### 5.4 Guidance 的可组合性

Figure 2 给出最直观的例子：
- 仅 collision guidance：ego 为了躲避后方来车，会冲出 lane。
- 加上 drivable guidance：ego 在 lane 内避让，安全 + 合规。

这是 classifier guidance 的杀手锏——**能量函数是相加的**（因为 $e^{-\mathcal{E}_1} \cdot e^{-\mathcal{E}_2} = e^{-(\mathcal{E}_1 + \mathcal{E}_2)}$），所以多个 guidance 可以任意组合，无需调权重训练。

---

## 6. 实验结果深入分析

### 6.1 主表（Table 1）解读

nuPlan 上有四个 benchmark：Val14, Test14, Test14-hard，每个都在 NR (non-reactive) 和 R (reactive) 两种 simulator 模式下测。

几个关键观察：

1. **Diffusion Planner (无 refine)** 在 Test14-hard NR 上达到 75.99，超过 PlanTF (69.70) 和 PLUTO w/o refine (70.03)，逼近 PLUTO w/ refine (80.08)。这是 pure learning-based 方法的 SOTA。

2. **Diffusion Planner w/ refine** 在 Test14 NR 上达到 94.80，**超过 Log-replay expert (94.03)**。这非常震撼——learning 方法超过了 human demonstration 的上限。可能原因：diffusion 生成的轨迹平滑度高，再叠加一个 search-based refinement (PDM-like)，search 空间质量大幅提升。

3. **Reactive mode 退化分析**：Diffusion Planner 在 reactive 模式下从 89.87 降到 82.80（降 7 分），而 PlanTF 从 84.27 降到 76.95（降 7 分），PLUTO w/o refine 从 88.89 降到 78.11（降 11 分）。**Diffusion Planner 的 reactive 鲁棒性更好**。我推测是 joint modeling 让 model 学到了 neighbor 的 reactive behavior distribution。

4. **GameFormer w/o refine 只有 13.32**——game-theoretic 在没有 rule-based 兜底时几乎不可用。这印证了 paper 的核心论点：现有 learning method 严重依赖 fallback。

### 6.2 Delivery-vehicle 数据集（Table 2）

这是 paper 的一个亮点——收集了 200 小时 Haomo.AI 配送车数据，车辆尺寸（1.03m × 2.34m）远小于 nuPlan 车辆（2.30m × 5.18m），行驶在 bike lane 上，行人 / 自行车交互密集。

观察：
- PlanTF 在 delivery-vehicle 上 90.89，Diffusion Planner 92.08，差距比 nuPlan 上更大。我推测是因为 delivery 场景有更多 multi-modal 决策（避让行人可能左绕可能右绕），diffusion 的 multi-modal 建模优势更明显。
- PDM-Hybrid 从 nuPlan 上的 92.77 掉到 80.72——因为 PDM 的 reference line 设计强依赖 nuPlan 的道路结构，delivery 场景下失效。
- 这说明 **Diffusion Planner 的 transferability 来自架构本身的 generality**，没有 nuPlan-specific 的 inductive bias。

### 6.3 与其他 diffusion-based planning 方法对比（Table 4）

| Planner | Test14 | Test14-hard | Val14 | Inference Time (s) |
|---|---|---|---|---|
| Diffusion-es w/o LLM | - | - | 50 | - |
| Diffusion-es w/ LLM | - | - | 92 | 0.5 |
| STR-16M | - | 27.59 | 45.06 | - |
| STR2-CPKS-800M w/o refine | 68.74 | 52.57 | 65.16 | >11 |
| **Diffusion Planner** | **89.19** | **75.99** | **89.87** | **0.04** |

Diffusion-es ([Yang et al., 2024](https://arxiv.org/abs/2402.06559)) 用 LLM 做 trajectory filter，去掉 LLM 后只剩 50 分。STR2-CPKS-800M ([Sun et al., 2024](https://arxiv.org/abs/2410.15774)) 800M 参数 + 11s inference，工程上不可用。**Diffusion Planner 用 0.04s 达到 89 分**——这是工程化的关键。

### 6.4 Ablation Studies（Table 3）

我整理成更易读的形式：

| 变体 | Score | 变化 | 解读 |
|---|---|---|---|
| Base | 89.19 | - | reference |
| w/o z-score norm | 85.02 | -4.17 | 纵向 50m、横向 3m 的尺度差让 attention 难学 |
| w/o interpolation | 83.78 | -5.41 | 只扰动 current state，future 仍 ground truth，模型学不到 recovery |
| w/o augmentation | 76.53 | -12.66 | OOD 严重，必须做 data aug |
| w/ SDE | 82.90 | -6.29 | state dropout 不够，直接删除 ego state 更好 |
| w/ ego state | 78.65 | -10.54 | velocity shortcut 严重损害 planning 能力 |
| w/o current state | 81.11 | -8.08 | current state 还是需要的，但只要 position + heading |

**最重要的 insight**：w/o augmentation 掉 12 分，说明 IL 模型在 closed-loop 下的 OOD 问题是头号杀手。Augmentation 方式（公式 C.1）：扰动 current state → 用 quintic polynomial 插值到 future → 生成一条动态可行的 recovery 轨迹作为新 label。这等于教模型"从 perturbed state 回到正常 trajectory"的能力，是 closed-loop 鲁棒性的关键。

参考：[ChauffeurNet (Bansal et al., 2018)](https://arxiv.org/abs/1812.03079) 首次提出这种 perturbation + recovery 训练方式。

### 6.5 Predicted Neighbor 数量 M 的 ablation（Figure 6）

M ∈ {5, 10, 15, 20, 25, 30}，最优在 M=10 附近。M 太少（< 5）丢失关键 agent；M 太多（> 20）引入远处无关 vehicle 的噪声，干扰 ego planning。**这跟 motion prediction 任务不同——prediction 关心 recall，planning 关心 ego 周边局部影响**。

### 6.6 Inference 超参 sweep（Figure 7）

- Denoise steps: 5 / 10 / 20 / 50。10 步已经稳定（用 DPM-Solver++）。
- Low-temperature: 0.1 ~ 1.0。Temperature = 1 是标准 diffusion 采样，< 1 是把 noise scale 缩小（[Ajay et al., 2022](https://arxiv.org/abs/2208.03674) 的 low-temperature sampling），让输出更 deterministic。0.5 在无 refine 时最好，0.1 在有 refine 时更好（因为 deterministic 轨迹让 refinement 的 scoring 更可靠）。

参考：[DPM-Solver (Lu et al., 2022)](https://arxiv.org/abs/2206.00927) | [Classifier-Free Guidance Revealed (Karras et al.)](https://arxiv.org/abs/2310.17811)

---

## 7. 一些更深层的联想与直觉

### 7.1 Diffusion vs. Decision Transformer

Decision Transformer ([Chen et al., 2021](https://arxiv.org/abs/2106.01345)) 把 RL 转成 sequence modeling，用 return-to-go 条件化生成。Diffusion Planner 把 planning 转成 trajectory generation，用 classifier guidance 条件化生成。两者都是"用生成模型解决决策问题"，但：

- DT 用 autoregressive，error 累积；diffusion 用 parallel denoising，每步都是 global refinement。
- DT 的 condition 是 scalar return，diffusion 的 condition 是任意可微 energy function——表达能力强得多。

### 7.2 Diffusion Policy 的延伸

[Diffusion Policy (Chi et al., 2023)](https://arxiv.org/abs/2303.04137) 在 robot manipulation 上首次大规模验证 diffusion 用于 action generation。Diffusion Planner 是这个范式在 autonomous driving 上的 specialized 版本，关键差异：

- **Action dim**：robot 是 7-DOF arm，driving 是 (M+1) × τ × 4 的 multi-agent tensor。
- **Condition**：robot 用 visual + proprioceptive；driving 用 vectorized map + history，更适合 transformer。
- **Guidance**：robot 一般不用 guidance；driving 强依赖 safety guidance，这是 paper 的核心贡献。

### 7.3 Energy-Based Model 视角

Diffusion model 与 EBM 同构：$q_\theta(\mathbf{x}) \propto e^{-E_\theta(\mathbf{x})}$，score 就是 $\nabla \log q = -\nabla E$。Classifier guidance 等价于修改能量函数 $E \to E + \mathcal{E}_{\text{guidance}}$。

这让我联想到 [Implicit Behavioral Cloning (Florence et al., 2021)](https://arxiv.org/abs/2104.04750)，用 EBM 做 IL，能量函数直接学习。Diffusion 的优势是 sample 效率高（不用 MCMC），而且训练更稳定（score matching 不需要 partition function 估计）。

### 7.4 LQR Controller 下游的耦合

Paper 提到 state 只用 (x, y, sin θ, cos θ) 因为"足够 downstream LQR controller 使用"。这里有个隐含的设计哲学：**planning 输出应该是 controller 容易跟踪的轨迹**。如果输出包含 velocity、acceleration，但 controller 实际只 position track，多余的输出反而引入 inconsistency。

这与 [Planning-oriented Driving (UniAD, Hu et al., 2023)](https://arxiv.org/abs/2212.10156) 的 philosophy 一致：planning 是整个 stack 的最终目标，上游 perception / prediction 都应服务于 planning。

### 7.5 Reactive mode 与 Game Theory

Diffusion Planner 在 reactive mode 下表现好，源于 joint modeling。但更深的问题是：**joint modeling 是否真正学到了 game-theoretic equilibrium？**

我觉得这里有一个微妙的点。Diffusion 学的是 $q(\mathbf{x}_{\text{ego}}, \mathbf{x}_{\text{neighbor}} | C)$，即联合轨迹分布。在 reactive simulation 中，neighbor 的行为由 simulator 给定（基于 ego 当前 plan 反应）。Diffusion 采样出的 joint trajectory 中，neighbor 部分其实是 model 自己的"预测"，不一定跟 simulator 的 reactive neighbor 一致。

但实验表明这样仍然 work，原因可能是：**diffusion 在每个 step 重新采样，会自适应 simulator 的 reactive behavior**。这是 closed-loop 的本质优势——每 0.1s 都重新 condition 在最新 observation 上。

参考：[GameFormer (Huang et al., 2023)](https://arxiv.org/abs/2303.05760) 显式做 game-theoretic 交互建模，但 w/o refine 只有 13 分。说明显式 game theory 在 closed-loop 上未必比 diffusion 的隐式 joint modeling 更好。

---

## 8. Limitations & Future Work 联想

### 8.1 Vectorized Input 的信息损失

Paper 用 vectorized map，而非 end-to-end from image。这避免了 perception 误差，但也限制了模型能用的信息（看不到 lane marking 之外的视觉 cue，比如施工锥、临时障碍）。

未来方向：用 image encoder + BEV feature 作为 condition。参考 [VAD (Jiang et al., 2023)](https://arxiv.org/abs/2303.12077) 或 [Sparse Drive](https://arxiv.org/abs/2407.04356)。

### 8.2 Lateral Flexibility 不足

Author 提到 model 不太会大幅度 lateral movement（变道、避让）。原因是 nuPlan 训练数据中 lane change 场景稀疏。

这个 observation 跟 [PlanTF](https://arxiv.org/abs/2310.16924) 一致——**dataset bias 是 IL-based planning 的根本痛点**。可能的解法：
- **Guidance 增强**：用 lateral-movement guidance（如"距离 target lane 越近 energy 越低"）强制 pull 模型做变道。
- **RL fine-tuning**：在 diffusion policy 上做 reward-based finetuning，参考 [Diffusion Q-Learning (Wang et al., 2022)](https://arxiv.org/abs/2301.12013) 或 [Decision Diffuser (Ajay et al., 2022)](https://arxiv.org/abs/2208.03674)。
- **Data augmentation**：合成 lane change 场景。

### 8.3 Sample Efficiency

10 步 denoise × 0.04s 已经很快，但传统 planner 是单次 forward。Consistency model ([Song et al., 2023](https://arxiv.org/abs/2303.01469)) 可以一步采样，是把 diffusion planning 真正部署到量产车的关键路径。Guided distillation ([Meng et al., 2023](https://arxiv.org/abs/2302.04855)) 可以把 guided diffusion 蒸馏成单步模型，同时保留 guidance 能力。

---

## 9. 总结：方法的精髓与 takeaway

如果让我提炼这篇 paper 的三个核心 insight：

1. **Joint multi-agent trajectory diffusion**：把 planning + prediction 统一成一个生成任务，让 cooperative behavior 自然 emerge，避免多 head 间的 loss 冲突。

2. **Training-free classifier guidance**：把 safety / comfort / speed preference 写成可微 energy function，在 inference 时通过 gradient 修改 score。这是 diffusion 在 decision-making 上独有的能力——RL 改 reward 要重训，IL 改 loss 要重训，diffusion 不需要。

3. **架构层面的小心思**：
   - 丢弃 ego velocity / acceleration 防 shortcut learning
   - MLP-Mixer 处理 sparse vectorized map
   - AdaLN 注入 navigation + timestep
   - DPM-Solver + low-temperature 让 inference 0.04s 完成

这三点合起来，让 diffusion 第一次在 closed-loop planning 上不依赖 rule-based refinement 就达到 SOTA。

---

## References (Web Links)

- Paper: https://zhengyinan-air.github.io/Diffusion-Planner/
- nuPlan benchmark: https://arxiv.org/abs/2106.11810
- Diffusion model (DDPM): https://arxiv.org/abs/2006.11239
- Score SDE: https://arxiv.org/abs/2011.13456
- Classifier Guidance: https://arxiv.org/abs/2105.05233
- Diffusion Posterior Sampling: https://arxiv.org/abs/2209.14687
- DPM-Solver: https://arxiv.org/abs/2206.00927
- DiT (Diffusion Transformer): https://arxiv.org/abs/2212.09748
- MLP-Mixer: https://arxiv.org/abs/2105.01601
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- PlanTF: https://arxiv.org/abs/2310.16924
- PLUTO: https://arxiv.org/abs/2404.14327
- GameFormer: https://arxiv.org/abs/2303.05760
- ChauffeurNet (augmentation trick): https://arxiv.org/abs/1812.03079
- UniAD: https://arxiv.org/abs/2212.10156
- Consistency Models: https://arxiv.org/abs/2303.01469
- Decision Transformer: https://arxiv.org/abs/2106.01345
- VectorNet: https://arxiv.org/abs/2005.04259
- Low-temperature sampling (Ajay et al.): https://arxiv.org/abs/2208.03674
- STR2 / MoE planner: https://arxiv.org/abs/2410.15774
- Diffusion-ES: https://arxiv.org/abs/2402.06559
- Implicit BCloning (EBM IL): https://arxiv.org/abs/2104.04750
- Is Ego Status All You Need: https://arxiv.org/abs/2403.04595
- Guided distillation: https://arxiv.org/abs/2302.04855

---

Andrej，这篇 paper 在我看来是 diffusion 进入 production-grade driving stack 的一个标志性工作。它把 multi-modality、safety constraint、personalization 三个长期困扰 learning-based planning 的问题，用 diffusion score + guidance 这个 unified framework 一起解决了。后续如果 consistency model 能把 10 步压到 1 步，加上 vision encoder 替换 vectorized input，我觉得这是真正能上车的 architecture。如果你最近在思考 diffusion 在 decision-making 上的应用，这篇绝对是必读 baseline。
