---
source_pdf: LIBERO-Plus.pdf
paper_sha256: 661de08e0ed1bb4bb4774177cac9fafa94f06b5385fa2f4107a1927b9a4aada7
processed_at: '2026-08-05T14:40:45-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版

Andrej，那我换个讲法，像在咖啡馆白板上跟你聊一样。

---

## 一句话总结

VLA 模型在 LIBERO 上刷到 95%+ 这件事，**根本不代表它会做机器人任务**。这篇 paper 干的事就是：把场景稍微动一动，95% 立刻掉到 30%；把语言指令抽空，success rate 几乎不变；把两个干扰一起上，比单独加还糟。然后作者把这堆 perturbation 固化成一个新的 benchmark 叫 LIBERO-Plus，10,030 个 task 分 5 个难度等级，自己又做了一个简单的 mixed fine-tuning recipe，camera robustness 从 55.6% 干到 92.8%。

project 链接先放这：
- https://sylvestf.github.io/LIBERO-plus/
- https://github.com/sylvestf/LIBERO-plus
- https://huggingface.co/collections/Sylvest/libero-plus

---

## 整篇 paper 真正想说的就三件事

### 1. "95% success rate" 是个 illusion

你看 Table 1，OpenVLA-OFT 在原版 LIBERO 上 97.1%。然后你把摄像头角度挪一下——掉到 59.7%。你把机械臂初始关节角度抖一下——掉到 37.2%。

π0 更夸张，原版 94.2%，camera 一动掉到 15.8%，robot init 一动掉到 6.6%。Physical Intelligence 那帮人写出 π0 的时候肯定知道这玩意儿脆，但 paper 里没人会主动暴露这件事，这篇 paper 替他们暴露了。

我用一句最直白的话讲：**这些模型不是"会做任务"，是"记住了在某个特定 camera 角度 + 特定 robot 姿态下该怎么动"**。你把这俩 anchor 一改，它就 out-of-distribution 了，跟 ImageNet classifier 看到一张旋转 30 度的猫就分错是一回事。

这里有 7 个 perturbation 维度，按"杀伤力"排序大致是：

| 维度 | 平均 drop | 本质 |
|---|---|---|
| Camera viewpoint | ~60+ | geometric frame shift，最致命 |
| Robot initial state | ~60+ | proprioceptive shift，最致命 |
| Sensor noise | 30~50 | feature corruption |
| Object layout | 30~40 | positional memorization 被打破 |
| Language | 25 | 看似 robust，其实是因为根本没用 |
| Light | 10~30 | wrist camera 救了一半 |
| Background | 5~30 | wrist camera 救了一大半 |

注意 Light 和 Background 看着 drop 小，**这不是因为模型 robust，是因为 wrist camera**。作者做了个超干净的 ablation：把第三视角图换成全黑，只保留 wrist camera，OpenVLA-OFT 还有 43.6% 的成功率。但 OpenVLA-OFT_w 这个去掉 wrist camera 的 variant，light perturbation 一来直接崩。所以 wrist camera 是个作弊器，它在近距离提供 illumination-invariant 的 geometric cue，third-person view 的光照变化它根本不在乎。

这件事说明：**所谓的"light robustness"是 architecture trick，不是 representation learning 的胜利**。

---

### 2. Language 这个 modality 基本被忽略——这是最 punchy 的发现

这一点我觉得是整篇 paper 最有意思的地方。

你想想 VLA 这个 acronym——Vision-Language-Action，语言是核心 selling point 之一，不然干嘛不直接叫 VA。但是作者做了两个实验，把这件事彻底证伪。

**Experiment 1: Blank instruction**

把 language input 整个换成空字符串。结果在 `object` suite 上 success rate 几乎不变。只在 `long` suite 上有下降，因为 long-horizon 任务确实需要 language 来 disambiguate subgoal。

**Experiment 2: Goal replacement**

这个更狠。原任务是 "pick up the alphabet soup and place it in the basket"，作者把 instruction 改成 "pick up the butter and place it in the basket"，butter 跟 alphabet soup 在同一个场景里都有。如果模型真的在 follow language，它应该去抓 butter。但 rollout 显示模型**仍然去抓 alphabet soup**，图 10 里 5 个 case 全是这样。

我把这件事翻译成机器学习黑话：**SFT 训出来的 VLA 学到的是 $p(a_t \mid o_t, \text{episode\_id})$，不是 $p(a_t \mid o_t, l_t)$**。language 在 training data 里跟 episode ID 高度相关（每个 episode 用一句固定 instruction），所以模型完全可以 bypass language，直接从 visual scene 识别出 "这是 episode #37"，然后 emit 出 episode #37 的 memorized motor program。

这个诊断跟你在 nanoGPT / makemore 教学里反复讲的"model 走最懒的 shortcut"完全一个套路。Language 看起来是个 input，但 gradient 没有真正 *flowing through* 它，因为视觉信号已经 sufficient 了。

$$
\mathcal{L}_{\text{SFT}} = -\sum_t \log p(a_t \mid o_t, l)
$$

这个 loss 最小化的时候，如果 $o_t$ 已经 determine $a_t$，那 $l$ 对 $\nabla \mathcal{L}$ 的贡献接近 0，模型自然学不到 *用* $l$。要 force 它用，得在 training 里制造 $o_t$ 不能单独 determine $a_t$ 的 situation——也就是让同一个 visual scene 配不同 language 给出不同 action。LIBERO 没有这种 data。

RIPT-VLA 加了 RL post-training，language drop 稍大一点（↓17.4 vs OpenVLA-OFT 的 ↓15.6），但 goal replacement 上还是 fail。我赌 RL 这块要真正 work，得让 reward signal 直接 depend on *是否完成了 language 指定的事*，而不只是"任务成功"。现在的 RL reward 还是 episode-level binary success，跟 language 没挂钩。

相关阅读：
- INT-ACT 也讲了 VLA generalization boundary: https://arxiv.org/abs/2506.09930
- RT-1 最早讨论 language-conditioned policy: https://arxiv.org/abs/2212.06817

---

### 3. 多个 perturbation 一起上，比单独加还糟——representation 是 entangled 的

这是 Section 5 的 statistical analysis，公式看着吓人但 idea 很简单。

作者想问：如果 camera perturbation 让 success rate 掉 40%，robot perturbation 让它掉 50%，那两个一起上应该掉多少？如果是 independent，应该差不多是 $1 - (1-0.4)(1-0.5) = 0.7$，掉 70%。但实际测下来掉得比 70% 还多。

形式化一下。定义 $D_i \in \{0,1\}$ 表示第 $i$ 种 perturbation 是否施加，$Y \in \{0,1\}$ 表示任务是否成功。conditional probability：

$$
s(D_i = d_i, D_j = d_j) = P(Y=1 \mid D_i = d_i, D_j = d_j)
$$

这里 $d_i, d_j \in \{0,1\}$，$s$ 就是四象限里每一象限的 success rate。

然后定义 compositionality gap：

$$
\Delta_{ij} = \text{Cov}(D_i, D_j \mid Y=1) = \mathbb{E}[D_i D_j \mid Y=1] - \mathbb{E}[D_i \mid Y=1]\,\mathbb{E}[D_j \mid Y=1]
$$

下标 $i, j$ 是 perturbation 类型的 index。$\mathbb{E}[D_i \mid Y=1]$ 是"在所有成功 trial 里，perturbation $i$ 出现的频率"，高就说明模型对这个 perturbation robust。$\mathbb{E}[D_i D_j \mid Y=1]$ 是"两个 perturbation 同时出现还成功的频率"。

$\Delta_{ij} > 0$ 意味着两个 perturbation 一起出现反而比独立预期更容易成功（synergy）。$\Delta_{ij} < 0$ 意味着两个一起出现比独立预期更难（interference）。

实验结果：**绝大多数 $\Delta_{ij} < 0$**。chi-square test p-value 都在 0.05 以下，显著。

最惨的几对：
- Camera × Robot init: joint 19.05%，远低于独立预期
- Camera × Layout: joint 35.95%
- Robot × Noise: joint 22.15%

人话：**模型内部把这些 perturbation 编码到了同一组脆弱的 features 里**。Camera shift 在 physics 上跟 robot init shift 是独立的（你换摄像机角度不影响机械臂关节角度），但 model 看到这两个 shift 都把它们 encode 成"feature distribution 偏离 training manifold"，一个 shift 把 feature 推出 manifold，第二个 shift 再推一把，直接 OOD。

这跟人类感知完全不一样。人类在 camera 移动 + distractor 出现时不会指数级变笨，因为人类有 *disentangled representation*——viewpoint 是一个 axis，object identity 是另一个 axis，互不干扰。VLA 没学到这种解耦。

相关：
- Compositional generalization 的经典框架 COG: https://arxiv.org/abs/2007.02779
- Neural net entanglement: https://arxiv.org/abs/1811.02733

---

## LIBERO-Plus benchmark 本身的设计

作者把诊断结果固化成 benchmark。10,030 个 task，7 个 perturbation × 4 个 LIBERO suite (Spatial/Object/Goal/Long)。最聪明的设计是 **difficulty level 用 model-as-jury 来定**：

- L1: 4 个 reference model 全都成功
- L2: 3 个成功
- L3: 2 个成功
- L4: 1 个成功
- L5: 0 个成功

Reference model 是 OpenVLA-OFT, π0, π0-fast, UniVLA。这等于用当前 model population 的能力 frontier 自动校准 task 难度。好处是 ordinal scale 比 absolute success rate 更 robust，坏处是三年后所有 L5 都被解了 benchmark 就饱和了——这是所有 static benchmark 的宿命。

Table 7 给了 task 分布：

| | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|---|---|---|---|---|---|---|---|---|
| Spatial | 376 | 350 | 354 | 292 | 258 | 351 | 312 | 2293 |
| Object | 396 | 398 | 390 | 297 | 248 | 422 | 425 | 2576 |
| Goal | 408 | 409 | 410 | 279 | 281 | 379 | 403 | 2569 |
| Long | 419 | 393 | 383 | 274 | 289 | 449 | 385 | 2592 |
| Total | 1599 | 1550 | 1537 | 1142 | 1076 | 1601 | 1525 | 10030 |

---

## 作者自己的 post-training recipe

作者用 22,400 条 generalized trajectory (覆盖 6 种 perturbation) 在 OpenVLA-OFT_m 上做 mixed fine-tuning，100k steps，lr $5 \times 10^{-4}$，8×A100，batch size 16，AdamW + cosine schedule。

Table 2 结果：

| | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|---|---|---|---|---|---|---|---|---|
| OpenVLA-OFT_m (base) | 55.6 | 21.7 | 81.0 | 92.7 | 91.0 | 78.6 | 68.7 | 67.9 |
| Ours (+ PT) | **92.8** | 30.3 | 85.8 | 94.9 | 93.9 | **89.3** | 77.6 | **79.5** |
| Δ | +37.2 | +8.6 | +4.8 | +2.2 | +2.9 | +10.7 | +8.9 | +11.6 |

最大 gain 在 camera (+37.2) 和 noise (+10.7)。这告诉你：**appearance-level robustness 基本是 data coverage 问题**，加数据就能解决。

最小 gain 在 robot init (+8.6)。这告诉你：**kinematic-level robustness 是 architecture 问题**，光加数据不够。Robot init 改的是机械臂的 joint configuration，这要求模型理解 forward kinematics，理解 "我现在关节是这个角度，所以 end-effector 在空间这个位置"。2D visual feature 根本 encode 不了这件事，你需要要么 explicit kinematic model，要么 3D representation，要么 equivariant architecture。

我赌 70% camera/light/noise 这种 robustness 是 data 问题，30% 是 architecture 问题；robot init 和 compositional robustness 基本纯 architecture 问题。

---

## 跟其他 benchmark 的对比

Table 3 给了对比：

| Method | Automation | Sim | Fine-grained | 7 个维度覆盖 |
|---|---|---|---|---|
| AGNOSTOS | ✗ | RLBench | ✗ | 只有 layout |
| RL4VLA | ✗ | ManiSkill | ✗ | layout + robot |
| INT-ACT | ✗ | ManiSkill | ✗ | layout + background |
| GemBench | ✗ | RLBench | ✗ | layout + background + robot |
| VLATest | ✓ | ManiSkill | ✗ | 部分 |
| COLOSSEUM | ✓ | RLBench | ✗ | 多维度但没分层 |
| **LIBERO-Plus** | ✓ | LIBERO | ✓ (L1-L5) | 全部 7 个 |

LIBERO-Plus 的卖点：automated + fine-grained difficulty level + 全部 7 维度覆盖。COLOSSEUM 之前是最接近的，但没有 difficulty stratification。

---

## Architecture 对比里几个有意思的点

**OpenVLA-OFT 的 FiLM 机制没 work**：

OpenVLA-OFT 用 Feature-wise Linear Modulation 来 "enhance language grounding"：

$$
h' = \gamma(l) \odot h + \beta(l)
$$

$h$ 是 visual feature，$l$ 是 language embedding，$\gamma, \beta$ 是从 $l$ 预测的 affine 参数。理论上有这个 modulation，visual feature 就会被 language condition。但 blank instruction 实验显示 FiLM 几乎没起作用。原因很可能是模型学到的 $\gamma \approx 1, \beta \approx 0$，退化成 identity modulation。SFT loss 不惩罚这种退化，因为 language 在 training data 里 redundant。

**π0 (flow-matching) vs π0-fast (discrete FAST tokens)**：

π0 在 camera perturbation 下崩到 15.8%，π0-fast 能 hold 在 66.4%。差了 50 个点。这暗示 **discrete action tokenization 强制模型学更 compositional 的 action representation**。FAST 用 DCT 把 action trajectory 变成频域稀疏表示，再 BPE 压成 token。每个 token 对应一段 trajectory 的 frequency component，这种 representation 比 continuous flow matching 的 trajectory manifold 更 robust to input shift。

- FAST paper: https://arxiv.org/abs/2501.09747
- π0 paper: https://arxiv.org/abs/2410.24164

**Wrist camera 是最大的 architecture trick**：

OpenVLA-OFT (有 wrist): camera perturbation ↓37.4  
OpenVLA-OFT_w (无 wrist): camera perturbation ↓78.5  

差 41 个百分点。Wrist camera 提供了 illumination-invariant + close-range geometric cue，是当前 VLA robustness 最大的 single contributor。问题在于：这种 robustness 是 *cheating* 的，因为它依赖一个额外 sensor modality，而不是 representation 本身的 invariance。真实场景里你不一定有 wrist camera，或者 wrist camera 也会被光照影响（手部阴影、反光）。

---

## 我读完之后的几个 open question

这部分是我自己的 speculation，可能 hallucination，你 salt-taking。

**Q1: action space 设计是不是 robot init 脆弱的真正原因？**

LIBERO 里大部分 model 用的是 absolute action（绝对 end-effector pose 或绝对 joint angle）。如果你用 *relative action*（delta end-effector pose），那 policy 天然 invariant to robot init——因为 "往前移动 5cm" 不依赖初始位置在哪。我赌如果有人重做这个实验用 relative action，robot init 这一栏的 drop 会从 60% 降到 20% 以内。

**Q2: 3D / equivariant representation 是不是 camera 脆弱的解药？**

Camera perturbation 改的是 viewpoint，在 SE(3) 群里是个 group action。如果你的 feature 是 SE(3)-equivariant 的，camera shift 就是个 gauge transformation，feature 跟着 transform，policy output 也跟着 transform，success rate 不变。PerAct / Act3D 这类工作已经证明了这点。问题是 SE(3)-equivariant network 通常 expensive 且不好 scale。

- PerAct: https://arxiv.org/abs/2209.05451
- Equivariant RL: https://arxiv.org/abs/2102.07398

**Q3: 怎么真正 force 模型用 language？**

我想到几个办法：
1. **Conflicting instruction training**: 同一个 visual scene，配 N 种不同 language，对应 N 种不同 action。强制模型必须读 language 才能 disambiguate。
2. **Language reconstruction auxiliary loss**: 让模型在 emit action 之前先 reconstruct 一遍 language（或 predict language 的 masked token）。Force information flow through language pathway。
3. **Adversarial language dropout**: 训练时随机 drop language，但同时有个 discriminator 判断 "这个 action 是在有 language 还是无 language 下产生的"。如果 discriminator 分不出来，说明 language 没被用，给个 penalty。

**Q4: World model 的真正用法**

WorldVLA 表现很差（camera 0.3%，几乎全崩）。但 world model 的 *idea* 是对的——预测 next frame 来 force representation 编码 physics。问题在于 next-frame prediction 的 objective 太 pixel-level，学到的是 appearance 不是 dynamics。如果换成 latent dynamics model（JEPA-style），可能能学到更 abstract 的 state。

- JEPA: https://arxiv.org/abs/2301.08243
- DreamerV3: https://arxiv.org/abs/2301.04104

**Q5: RL post-training 的真正用法**

RIPT-VLA 用 RL 在 fixed env 上 post-train，OOD 帮助有限。但 RL 真正的 power 应该是 *adversarial perturbation training*——让 adversary 网络 选择 perturbation 参数来 maximize failure probability，policy 去 resist。类似 RARL / ARP。

- Adversarial RL: https://arxiv.org/abs/2103.00336

**Q6: Benchmark 三年后饱和怎么办？**

L1-L5 这种 static stratification 三年后所有人都能解 L5。要可持续，得做 *continual benchmark*——perturbation parameter 是 continuous 的，用 Elo rating 或 IRT (Item Response Theory) 动态校准 task difficulty，根据当前 model population 自适应。像 TOEFL 那种自适应考试。

- IRT in ML: https://arxiv.org/abs/2005.12724

---

## 最后给你一个直觉判断

这篇 paper 真正的价值，我觉得不在 LIBERO-Plus benchmark 本身（benchmark 会饱和），而在它用 clean ablation 把两件事钉死了：

1. **当前 VLA 名义上是 V-L-A，实际跑起来是 V-A**。Language 是 decorative input。
2. **当前 VLA 学到的是 episode-conditioned motor program memorization，不是 task-level generalization**。Position 一动就崩。

这两件事 community 之前都 *suspect* 但没人这么系统地 *prove*。下一个十年的 VLA 研究议程，我觉得会被这篇 paper 显著塑形——大家会开始问："我的 model 真的在用 language 吗？真的在 generalize 吗？还是又在 memorize？"

如果你想 triangulate，建议把 COLOSSEUM、INT-ACT、AGNOSTOS 这几篇一起读：
- COLOSSEUM: https://arxiv.org/abs/2402.08191
- INT-ACT: https://arxiv.org/abs/2506.09930
- AGNOSTOS: https://arxiv.org/abs/2505.15660

COLOSSEUM 是 coarse robustness audit，LIBERO-Plus 把它升级成 fine-grained + level-stratified + statistical-interaction-aware。INT-ACT 更聚焦 cross-task。AGNOSTOS 是 cross-task generalization 的另一个角度。

希望这次讲得更像人话了。

---

# LIBERO-Plus: VLA Models 的 Robustness 深度剖析

Andrej，这篇 paper 我读完之后第一反应是——它做的事情其实非常像你在 *"Software 2.0"* 和后来的 *"Deep Learning: My 32-year Path"* 里反复强调的那种"benchmark 看起来 saturated 了，但真实 generalization 还差得远"的诊断工作。LIBERO 上动辄 95%+ 的 success rate 把 community 麻痹了，而这篇工作做的事情就是把表面那层 paint 刮掉，露出底下朽木。我下面尽量按你的直觉方式来拆。

paper links:
- Project: https://sylvestf.github.io/LIBERO-plus/
- Code: https://github.com/sylvestf/LIBERO-plus
- Models: https://huggingface.co/collections/Sylvest/libero-plus
- LIBERO benchmark (原版): https://libero-project.github.io/
- OpenVLA: https://github.com/openvla/openvla
- π0: https://arxiv.org/abs/2410.24164
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- RIPT-VLA / RLOO: https://arxiv.org/abs/2505.17016
- FAST tokenizer: https://arxiv.org/abs/2501.09747
- COLOSSEUM (相关的 RLBench robustness benchmark): https://arxiv.org/abs/2402.08191
- AGNOSTOS: https://arxiv.org/abs/2505.15660

---

## 1. 论文在做什么——一句话定位

它做了三件事：

(1) **Diagnostic study**：把 7 个 perturbation 维度 (objects layout, camera viewpoints, robot initial states, language instructions, light conditions, background textures, sensor noise) 系统地打到 10 个 SOTA VLA models 上，看 success rate 怎么崩。

(2) **Mechanistic probing**：表面现象之下的 mechanism，包括 "language 是否真的被用"、"visual 是否在 attend 正确的 object"、"perturbation 之间是否有 combinatorial interaction"。

(3) **Benchmark + recipe**：把诊断结果固化成 LIBERO-Plus (10,030 个 tasks，分 5 个 difficulty levels)，并展示在 generalized 数据集上做 mixed fine-tuning，能把 camera robustness 从 55.6% 拉到 92.8%。

---

## 2. 核心发现总览（Table 1 的解读）

我把 Table 1 的关键 trend 抽出来重新组织一下，因为它信息密度很高。

| Model | Original | Camera | Robot | Language | Light | Background | Noise | Layout |
|---|---|---|---|---|---|---|---|---|
| OpenVLA | 76.5 | 1.1 (↓75.4) | 4.1 (↓72.4) | 26.8 (↓49.7) | 4.4 (↓72.1) | 25.3 (↓51.2) | 19.3 (↓57.2) | 31.6 (↓44.9) |
| OpenVLA-OFT | 97.1 | 59.7 (↓37.4) | 37.2 (↓59.9) | 81.5 (↓15.6) | 85.8 (↓11.3) | 92.4 (↓4.7) | 76.7 (↓20.4) | 77.1 (↓20.0) |
| π0 | 94.2 | 15.8 (↓78.4) | 6.6 (↓87.6) | 61.0 (↓33.2) | 79.6 (↓14.6) | 78.5 (↓15.7) | 79.4 (↓14.8) | 70.4 (↓23.8) |
| π0-fast | 85.5 | 66.4 (↓19.1) | 24.8 (↓60.7) | 63.3 (↓22.2) | 73.0 (↓12.5) | 67.7 (↓17.8) | 75.8 (↓9.7) | 70.3 (↓15.2) |
| WorldVLA | 79.1 | 0.3 (↓78.8) | 30.2 (↓48.9) | 44.2 (↓34.9) | 29.4 (↓49.7) | 14.5 (↓64.6) | 12.2 (↓66.9) | 39.4 (↓39.7) |
| UniVLA | 95.2 | 4.3 (↓90.9) | 50.3 (↓44.9) | 71.8 (↓23.4) | 59.1 (↓36.1) | 80.0 (↓15.2) | 25.3 (↓69.9) | 34.3 (↓60.9) |
| RIPT-VLA | 97.5 | 58.3 (↓39.2) | 36.7 (↓60.8) | 80.1 (↓17.4) | 87.9 (↓9.6) | 90.4 (↓7.1) | 73.8 (↓23.7) | 76.5 (↓21.0) |

**Intuition to build**：

- **Camera + Robot init 是两大杀手**。这两个的共同点是：它们改变的是 *几何/本体感觉* (geometry / proprioception)，而不是 *appearance* (photometric stuff)。这告诉我们 VLA 学到的 representation 严重 *anchored to a specific viewpoint-pose manifold*，而不是真正的 3D / kinematic understanding。这跟你经常吐槽的 "CNNs are lazy, they shortcut to texture" 是一个家族问题，只不过这里 shortcut 到的是 camera extrinsic。
- **Light / Background 影响小**，看似 robust，但 Section 3 拆穿了这个 illusion：那是因为 wrist camera 提供了 illumination-invariant geometric cues。一旦把 wrist camera 抠掉 (OpenVLA-OFT_w)，light perturbation 立刻造成 ↓27.1 的 drop。也就是说 robustness 不是从 representation 的不变性里来的，而是从一个"备用摄像头"里来的。这是一个非常 cheating 的 robustness。
- **Language drop 最小** (平均 ↓25.3)。但 Section 4 用两个 ablation (blank instruction + goal replacement) 证明这并不是因为模型懂 language，而是因为它根本 *没在用* language。这下面专门讲。

---

## 3. 第一个 punchline：models 根本没在 follow language

这个发现我认为是整篇 paper 最有意思的部分，因为它直接戳破了 "VLA = V + L + A" 这个 acronym 本身。

### 3.1 三种 hypothesis 与证伪

作者列了三个 hypothesis 来解释 "language perturbation 几乎不影响 success rate"：

- **H1**: 模型 language generalization 强，所以 paraphrase 不影响。
- **H2**: 模型只抽 keywords 做 matching。
- **H3**: 模型根本没用 language，把它当 noise。

### 3.2 Blank-instruction experiment (Figure 3a)

把 language input 整个替换成空字符串。

结果：在 `object` suite 上，**success rate 几乎不变**；只在 `long` suite 上有显著下降 (因为 long-horizon task 需要语言来 disambiguate subgoals)。

这是 H3 的强证据。换句话说，名义上是 Vision-**Language**-Action，实际跑起来是 Vision-Action。

### 3.3 Goal-replacement experiment (Figure 3b, Figure 10)

这个实验设计得很巧妙：把 instruction 里的 target object 替换成 *同场景里的另一个 object*。例如：

> 原: "pick up the alphabet soup and place it in the basket"  
> 改: "pick up the butter and place it in the basket"

如果模型真的在 follow language，它应该去抓 butter。但 rollout 显示模型**仍然去抓 alphabet soup**。

这就排除了 H1 和 H2，因为：
- H1 (强 generalization) 不可能让它选错 object。
- H2 (keyword matching) 在 keyword 被替换的情况下应该会改变 behavior，但没有。

**Intuition**：这告诉我们 SFT-trained VLA 学到的本质上是 $p(a_t | o_t)$，language 只是 *spurious correlator* 被打包进了 $p(a_t | o_t, l)$。在 LIBERO 这种 *task-episode-aligned* 的数据里，language 和 episode ID 是高度相关的，所以模型完全可以 bypass language，直接从 visual 识别 episode 类型，再 emit 对应的 *memorized action trajectory*。

这其实跟你在 nanoGPT / makemore 教学里讲的 "model takes the easy shortcut if you let it" 完全一致。这里 language 是个 easy feature 没错，但 visual episode identity 是个 *easier* feature，所以模型走了更懒的路。

### 3.4 为什么这件事重要

这件事的杀伤力在于：

1. 它意味着当前 VLA evaluation 里的 "language-conditioned" 是个 *misnomer*。
2. 它意味着你想做 *open-vocabulary instruction following* 这种 selling point，根本没被 benchmark 覆盖到——LIBERO 的 success rate 衡量的是 "在这个固定任务里把动作做对"，而不是 "做对任务"。
3. 它和 embodied AI 里一个长期争论连上了——*到底是 policy learning 还是 trajectory memorization*。这篇 paper 给的证据强烈支持后者，至少对 SFT-only 的 model 是这样。RIPT-VLA (用了 RL post-training) 的语言 drop 稍大一些 (↓17.4)，暗示 RL 可能在 *forcing the model to actually use language* 上有点帮助，但 goal-replacement 那边 RIPT-VLA 也还是 fail 的。

相关参考：
- RT-1 (Brohan et al., 2022): https://arxiv.org/abs/2212.06817
- 关于 VLA 是否真正 language-conditioned 的近期争论，INT-ACT: https://arxiv.org/abs/2506.09930

---

## 4. 第二个 punchline：positional bias, 不是 semantic understanding

Section 3 把 `Object Layout` 这个 perturbation 拆成两个子项：

- **O1: confounding objects** — 在场景里乱加 distractor。
- **O2: target object pose** — 平移/旋转 target object 本身。

结果 (Figure 1)：

- O1 几乎不影响 success rate (π0, π0-fast, RIPT-VLA, UniVLA, WorldVLA 都只小幅下降)。
- O2 让 success rate 大幅跳水。

**Intuition**：这告诉我们模型 *会* ignore distractor（这部分 attention 是 working 的，可能是 DINOv2 features 带的 inductive bias），但**不会** generalize 到 target 位置变化。也就是说它学到的是 "在这个屏幕坐标附近抓一下"，而不是 "找到语义上叫 alphabet soup 的那个物体然后抓它"。

这是一种典型的 **egocentric-frame memorization**：模型把视觉输入当作一个 lookup key，去 query 一个 *precomputed motor program*。Motor program 是 hardcoded 到特定 (object_position_in_image, robot_init_qpos) 的。

这跟 RL agent 在 Atari 上经常表现出的 "position-locked policy" 是一类现象。你在 CS231n 里讲过 convolutional nets 的 translation invariance 是 *architectural* 的，但 policy 的 translation invariance 需要 *data-driven*，这里的数据没给到。

---

## 5. 第三个 punchline：compositional generalization gap 是 *负的*

这个 section 的 statistical formulation 我觉得写得挺漂亮的，值得仔细讲。

### 5.1 数学设定

定义 indicator variables：

$$
D_i = \begin{cases} 1, & \text{if the } i\text{-th perturbation is applied} \\ 0, & \text{otherwise} \end{cases}
$$

这里下标 $i \in \{1, 2, ..., 6\}$，对应 6 种 perturbation (layout / env / light / camera / robot / noise)。语言被排除了，因为前面已经证明 language 信号根本没在用，加进去会"deceptive"。

$$
Y = \begin{cases} 1, & \text{if task succeeds} \\ 0, & \text{otherwise} \end{cases}
$$

成功率的条件概率形式：

$$
s(D_i = d_i, D_j = d_j) = P(Y = 1 \mid D_i = d_i, D_j = d_j), \quad d_i, d_j \in \{0, 1\}
$$

然后定义 *在成功条件下* 的联合概率：

$$
p(D_i = d_i, D_j = d_j \mid Y = 1) = \frac{s(D_i = d_i, D_j = d_j)}{\sum_{a, b \in \{0, 1\}} s(D_i = a, D_j = b)}
$$

这个分母 $\sum_{a,b} s(\cdot)$ 是把 4 种组合 (00, 01, 10, 11) 的成功率加起来做 normalization。这里有个 subtle 的点：这不是严格意义上的 $P(D_i | Y=1)$，因为 $s$ 是 *success rate* 而不是 *trial frequency*——但作者把 trial frequency 假设成均匀的 (每种组合跑同样多 trial)，所以等价。这点在 reading 的时候要心里有数。

### 5.2 Marginal 和 Joint

Marginal：

$$
p(D_i = 1 \mid Y = 1) = \frac{s(D_i = 1, D_j = 0) + s(D_i = 1, D_j = 1)}{\sum_{a,b} s(D_i = a, D_j = b)}
$$

直觉：在所有 *成功* 的 trial 里，第 $i$ 种 perturbation 出现的频率。**高** → 模型对这种 perturbation 鲁棒；**低** → 敏感。

Joint：

$$
p(D_i = 1, D_j = 1 \mid Y = 1) = \frac{s(D_i = 1, D_j = 1)}{\sum_{a,b} s(D_i = a, D_j = b)}
$$

直觉：在所有 *成功* 的 trial 里，两种 perturbation 同时出现的频率。

### 5.3 Compositionality Gap

这是核心定义：

$$
\Delta_{ij} \triangleq \text{Cov}(D_i, D_j \mid Y = 1) = \mathbb{E}[D_i D_j \mid Y=1] - \mathbb{E}[D_i \mid Y=1]\,\mathbb{E}[D_j \mid Y=1]
$$

也就是：

$$
\Delta_{ij} = p(D_i=1, D_j=1 \mid Y=1) - p(D_i=1 \mid Y=1)\, p(D_j=1 \mid Y=1)
$$

**Intuition**：

- $\Delta_{ij} > 0$：两种 perturbation 同时出现时，成功比例 *高于* 它们独立贡献的乘积。意思是模型对这两个 perturbation 的组合 *格外 robust*，可能它们触发的是同一个 robust feature。
- $\Delta_{ij} < 0$：组合 perturbation 比独立 perturbation 的乘积 *更糟*。两个 perturbation *interact negatively*，feature space 里它们是 coupled noise sources。
- $\Delta_{ij} = 0$：独立，可加性成立。

### 5.4 实验结果

2000 次重复实验，OpenVLA-OFT 上。结果：**绝大多数 $\Delta_{ij} < 0$**。Table 8 + Table 9 (chi-square test) 给了显著性验证。

最有意思的几对：

- `Camera × Robot init`: 19.05% joint，独立预期会更高 → 强负 interaction。这个 makes sense：camera 移了 *plus* robot 初始姿态变了，模型完全 lost 掉 spatial frame。
- `Camera × Layout`: 35.95% joint，远低于各自单独时 (57.30% / 71.75%) 的几何平均。
- `Robot × Noise`: 22.15% joint，也强负。
- `Layout × Noise`: 44.55% joint vs 71.75% × 71.50% / norm ≈ 51%，仍然负。

**Intuition**：这告诉我们 VLA 学到的 representation 是 *entangled* 的。Camera 和 Robot init 在 *physics* 上是独立的 (你换摄像机角度不影响机械臂关节角度)，但模型内部把它们 encode 进了同一组脆弱的 features，所以一个 perturbation 把 feature 拉出 manifold，另一个 perturbation 再拉一次，模型直接 out-of-distribution。

这跟 ImageNet classifier 上的 "benchmark robustness 不传递" 是一回事：单独的 Gaussian noise robust 和单独的 rotation robust，加起来 *不会* 给你 Gaussian-noise-on-rotated-images 的 robust。 representation 没有把 invariances 解耦。

相关阅读：
- Compositional generalization 的经典讨论：https://arxiv.org/abs/2007.02779 (COG)
- Neural networks entanglement: https://arxiv.org/abs/1811.02733

---

## 6. Architecture 对比里的 insight

paper 评了 10 个 model，architecture 维度上有几个对比值得单独拎出来：

### 6.1 Autoregressive vs Diffusion

- **Autoregressive**: OpenVLA, OpenVLA-OFT, Nora, WorldVLA, UniVLA, π0-fast
- **Diffusion / Flow-matching**: π0
- **Hybrid**: RIPT-VLA (autoregressive backbone + RL post-training), WorldVLA (action + world model joint)

从 Table 1 看，π0 (flow-matching) 在 *appearance* perturbation (light/background/noise) 上表现不错 (↓14~15)，但在 *geometric* perturbation (camera ↓78.4, robot ↓87.6) 上崩得最惨。这暗示 **flow-matching 的 continuous action representation 对 visual geometry shift 更敏感**，可能是因为 flow matching 学到的是 *trajectory manifold* 而不是 *stepwise corrective policy*，一旦起始 state 偏离，整个 trajectory 直接 fail。

π0-fast 用 FAST tokenizer (DCT + BPE)，把 action sequence 压成 sparse frequency tokens，camera drop 只有 ↓19.1，比 π0 好很多。这暗示 **discrete action token 强制模型学习更 compositional 的 action representation**，每个 token 对应一段 trajectory 的 frequency component，相比 continuous flow 更 robust to input shift。

### 6.2 Wrist camera 的作用

这是 paper 里最干净的一个 ablation：

- `OpenVLA-OFT` (with wrist camera): camera perturbation ↓37.4
- `OpenVLA-OFT_w` (third-view only): camera perturbation ↓78.5

差了 41 个百分点。这说明 wrist camera 提供的 *local, close-range, illumination-invariant* geometric cue 是 robustness 的主要来源。

3rd-black / all-black 实验 (Figure 2) 进一步证实：
- All-black (两路都遮): success rate ≈ 0
- 3rd-black (只遮第三视角，保留 wrist): 仍然能到 43.6 / 43.0 / 67.3 (取决于 model)

这告诉我们：**模型对第三视角的依赖其实没我们想象那么大**。这又引出一个问题——既然 wrist 这么 powerful，那第三视角到底在 contribute 什么？Paper 没有完全回答，但 hint 是：第三视角提供 *scene-level context* (object 在哪、layout 是什么)，wrist 提供 *contact-level precision*。如果 scene context 被扰动，wrist 接管 fine motor，所以 light perturbation 不致命；但 camera perturbation 改的是 *scene-level geometric frame*，wrist 救不了。

### 6.3 Co-training 的作用

π0 / π0-fast 的 pre-training mixture 包含 web data 和 diverse robot data，相比 OpenVLA (OpenX-embodiment only) 在 light/background/noise 上明显更 robust。这印证了一个朴素直觉：**data diversity > model capacity**，对于 robustness 来说。

RIPT-VLA 用了 RL post-training (RLOO + PPO)，整体 robustness 几乎和 OpenVLA-OFT 持平甚至略好。这暗示 RL 在 *已知的 failure mode* 上有 improvement，但不会 magically 产生 OOD robustness，因为 RL 的 reward 仍是在 fixed environment 上算的。

### 6.4 OpenVLA-OFT 的 FiLM 机制

OpenVLA-OFT 用了 Feature-wise Linear Modulation 来 *enhance language grounding*：

$$
h' = \gamma(l) \odot h + \beta(l)
$$

其中 $h$ 是 visual feature，$l$ 是 language embedding，$\gamma, \beta$ 是从 $l$ 预测的 affine 参数。理论上这应该让 visual feature *conditioned on* language。但 Section 4 的 blank-instruction 实验显示 FiLM 在 OpenVLA-OFT 上 *没起作用*。

**Intuition**：FiLM 是 *additive modulation*，如果模型学到的 $\gamma, \beta$ 接近常数 (与 $l$ 无关)，那它就 degenerate 成了 identity modulation。SFT loss 不会惩罚这种 degenerate solution，因为 language 在训练数据里是 redundant signal。这是经典 "auxiliary signal gets ignored if main signal is sufficient" 的 case。

---

## 7. LIBERO-Plus benchmark 本身

### 7.1 Construction

- 起点：LIBERO 的 40 个 task。
- 7 个 perturbation × 4 个 suite (Spatial/Object/Goal/Long) × 500 instances = 14,000 candidate tasks。
- 用 baseline models 跑一遍，去掉所有 model 都能解的 (ceiling effect)，剩下的平衡一下 sub-dimension。
- 最终：**10,030 tasks**，分布见 Table 7。

### 7.2 Difficulty level 设计

非常聪明的设计：用 4 个 reference model (OpenVLA-OFT, π0, π0-fast, UniVLA) 在每个 task 上的成功与否做 4-bit 编码：

- L1: 全部 4 个 model 都成功
- L2: 3 个成功
- L3: 2 个成功
- L4: 1 个成功
- L5: 0 个成功

这给你一个 *ordinal difficulty scale*，比单纯用 success rate 平均要 robust。Figure 7 给了每个 perturbation 维度下 L1-L5 的分布比例。

**Intuition**：这种 "model-as-jury" 的难度定义和 ImageNet-C / ImageNet-P 里用 multiple model 一致性来定义 difficulty 是一个思路。它的好处是自动校准到 *current model population* 的能力 frontier，坏处是 benchmark 会随着 model 进步而 *老化*——三年后所有 L5 都被解了，benchmark 就饱和了。这其实是所有 static benchmark 的宿命。

### 7.3 Difficulty stratification 透露的趋势

从 Figure 5 / Figure 8 可以看到，几乎所有 model 在 L1→L5 上都是单调下降的，但 *斜率* 不一样：

- π0-fast / RIPT-VLA: 下降相对平缓，说明它们在 "稍微难一点" 的 task 上还能 hold。
- WorldVLA: 下降陡峭，说明它的 representation 更 *narrow*。
- OpenVLA (没有 OFT): 几乎所有 level 都低，说明它根本没 generalize。

---

## 8. Post-training recipe 的结果

Table 2 里 "Ours" 这一行是 paper 自己的 recipe：用 generalized dataset (22,400 trajectories，覆盖 6 种 perturbation) 在 OpenVLA-OFT_m 上做 mixed fine-tuning。

| | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|---|---|---|---|---|---|---|---|---|
| OpenVLA-OFT_m (base) | 55.6 | 21.7 | 81.0 | 92.7 | 91.0 | 78.6 | 68.7 | 67.9 |
| Ours (+ PT) | **92.8** | 30.3 | 85.8 | 94.9 | 93.9 | **89.3** | 77.6 | **79.5** |
| Δ | +37.2 | +8.6 | +4.8 | +2.2 | +2.9 | +10.7 | +8.9 | +11.6 |

**Intuition**：

- Camera robustness +37.2 是最大的 gain。这告诉我们 *camera robustness 主要是 data coverage 问题*，不是 architecture 问题。只要 training data 里有足够的 viewpoint variation，模型就能学。
- Robot init 只 +8.6。这是个 hint：robot init perturbation 可能需要 *更本质的 architectural inductive bias* (比如 explicit kinematic model 或 3D representation)，单靠 data augmentation 不够。
- Language +4.8 微弱提升，再次印证 language 不是主战场。
- Noise +10.7 也不小，说明 sensor noise 也可以 largely 用 data augmentation 解决。

这个结果让我想起你在 "A Recipe for Training Neural Networks" 那篇 blog 里讲的——*data augmentation is the most reliable regularizer*。这里又一次被验证。

---

## 9. 我的 critique / 思考方向

这部分不是 paper 内容，是我读完之后想到的，可能对你 build intuition 有帮助，也可能有 hallucination，请 salt-taking。

### 9.1 Language 的 diagnostic 还可以更狠

Paper 用 blank instruction 和 goal replacement 证明 language 没被用。但其实还有一个更狠的实验：*conflicting instruction*——给一个视觉上看起来是 task A 的 setup，但 language 说做 task B。如果模型 follow visual，那它做 A；如果 follow language，做 B。这种 *conflict trial* 比 goal replacement 更干净，因为 goal replacement 至少改了 task 设定。我猜测 conflict trial 会显示模型 100% follow visual。

### 9.2 3D / equivariant representation 是不是解药

Camera 和 robot init 的脆弱性强烈暗示 *implicit 2D feature* 不够。如果换成 *explicit 3D representation* (比如 NeRF feature, 3D Gaussian Splatting feature, 或者 SE(3)-equivariant network)，camera perturbation 应该是个 *gauge transformation* 而不是 OOD shift。

相关工作：
- Equivariant RL: https://arxiv.org/abs/2102.07398
- 3D scene representation for manipulation: https://arxiv.org/abs/2401.05512 (GNFactor)
- PerAct / Act3D: https://arxiv.org/abs/2209.05451

### 9.3 World model 是不是能帮上忙

WorldVLA 在这个 benchmark 上表现很差，但它的 *idea* 是对的——预测 next frame 来强迫 representation 编码 physics。问题是它 *next-frame prediction 的 objective 太 pixel-level*，所以 world model 学到的是 appearance 而不是 dynamics。如果换成 *latent dynamics model* (joint-embedding predictive, JEPA-style)，可能能学到更 abstract 的 state，从而 robustness 更好。

- JEPA: https://arxiv.org/abs/2301.08243
- DreamerV3: https://arxiv.org/abs/2301.04104

### 9.4 RL post-training 的真正用法

RIPT-VLA 用 RL 在 fixed env 上 post-train，对 OOD 帮助有限。但 RL 真正的 power 应该是 *adversarial perturbation training*——让 adversary 选择 perturbation 来 maximize failure probability，policy 去 resist。这类似 robust RL 里的 RARL / ARP。如果 LIBERO-Plus 的 perturbation engine 做成 differentiable 或者可以被 adversary 控制的，可以做 *curriculum adversarial training*。

- Adversarial RL: https://arxiv.org/abs/2103.00336

### 9.5 Compositional gap 的负号意味着什么

负的 $\Delta_{ij}$ 告诉我们 perturbation 之间是 *super-linear* destructive 的。这其实在 *human perception* 上不太成立——人类在 camera + distractor 同时存在时不会比单独时指数级差。这暗示 model 的 representation 没有把 *nuisance variables* (viewpoint, distractor) 和 *task variables* (target identity, goal) 解耦。

如果引入 *disentangled representation learning* (比如 beta-VAE, FactorVAE, 或者更 modern 的 contrastive disentanglement)，是否能把这个 gap 拉向 0？这是个 open question。

- beta-VAE: https://openreview.net/forum?id=Sy2fzU9gl
- Disentanglement library: https://github.com/google-research/disentanglement_lib

### 9.6 Benchmark 的 dynamic 化

L1-L5 这种 static stratification 三年后会 saturated。一个更可持续的设计是 *continual benchmark*——perturbation parameter 是 continuous 的，用 Elo rating 或者 IRT (Item Response Theory) 来动态标定 task difficulty，根据当前 model population 自适应。这和人语言能力测试 (TOEFL, GRE) 的难度校准是一个思路。

- IRT in ML: https://arxiv.org/abs/2005.12724

---

## 10. 关键 takeaways 总结

如果你只记 5 件事：

1. **VLA 的 95% success rate 是 illusion**。Camera 和 robot init 一动就崩到 30% 以下。
2. **Language 基本被忽略**。Blank instruction / goal replacement 都证明这点。"VLA" 实际是 "VA"。
3. **Wrist camera 是 robustness 的作弊器**。抠掉它，light robustness 立刻消失。
4. **Compositional generalization gap 是负的**。Perturbation 之间 super-linear destructive，representation entangled。
5. **Data diversity 能解决大部分 appearance-level robustness** (camera +37.2 通过 post-training)，但 robot init 这种 kinematic-level robustness 需要 architectural innovation。

---

## 11. 我会推荐怎么 follow up

如果你想在脑子里继续 build intuition，我建议你思考下面几个问题：

**Q1**: 如果把 LIBERO 的所有 task 都重写，让 language 和 episode ID *解相关* (每个 episode 用 N 种不同 paraphrase，且不同 episode 共享部分 paraphrase)，model 还能学吗？还是说现在的 VLA 完全是 episode-ID-classifier？

**Q2**: 如果在 OpenVLA-OFT 的 FiLM 上加一个 *bottleneck* (强制 $\gamma, \beta$ 必须从 language 里 reconstruct 出 language 本身，auxiliary loss)，能不能 force 模型真的去读 language？

**Q3**: Compositional gap $\Delta_{ij}$ 如果换成 $\Delta_{ijk}$ (三阶)，会变得更负还是开始 saturated？这能告诉我们 representation 的 entanglement 是 pairwise 还是 higher-order？

**Q4**: Robustness 的 *极限* 在哪？人类在这 7 个 perturbation 下基本 0 drop。VLA 能不能达到？是 data 问题还是 architecture 问题？我赌 70% 是 architecture (需要 3D / equivariant)，30% 是 data。

**Q5**: 如果把 action space 从 absolute action 换成 *relative action* (delta end-effector pose)，robot init robustness 会不会好很多？直觉上应该会，因为 relative action 天然 invariant to init pose。但 paper 里测的 model 大部分是 absolute action token，这可能是 robot init 那一栏这么惨的真正原因。

---

## 12. 相关的 paper 你应该一起读

为了 triangulate 这篇 paper 的发现，我建议把下面几篇一起看：

- **COLOSSEUM** (Pumacay et al., 2024): RLBench 上的 robustness benchmark，6 个 perturbation 维度，但没有 fine-grained 难度分层。https://arxiv.org/abs/2402.08191
- **INT-ACT** (Fang et al., 2025): 也是 probing VLA generalization boundary，更聚焦 cross-task。https://arxiv.org/abs/2506.09930
- **AGNOSTOS** (Zhou et al., 2025): Cross-task generalization。https://arxiv.org/abs/2505.15660
- **VLABench** (Zhang et al., 2024): Long-horizon reasoning。https://arxiv.org/abs/2412.18194
- **RL4VLA** (Liu et al., 2025): RL 在 VLA 上的 empirical study。https://arxiv.org/abs/2505.19789
- **VLATest** (Wang et al., 2025): 软件工程角度的 VLA testing。https://arxiv.org/abs/2501.10335 (proc of ACM FSE)
- **π0 paper**: https://arxiv.org/abs/2410.24164
- **FAST**: https://arxiv.org/abs/2501.09747
- **OpenVLA-OFT**: https://arxiv.org/abs/2502.19645
- **UniVLA**: https://arxiv.org/abs/2505.06111
- **WorldVLA**: https://arxiv.org/abs/2506.21539
- **RIPT-VLA / Interactive post-training**: https://arxiv.org/abs/2505.17016

如果你只读一篇先看 COLOSSEUM，再回头看这篇，会发现 LIBERO-Plus 的核心 contribution 是把 COLOSSEUM 的 *coarse robustness audit* 升级成 *fine-grained, level-stratified, statistical-interaction-aware* 的诊断。

---

希望这个拆解对你 build intuition 有帮助。我个人觉得这篇 paper 的真正价值不在于 LIBERO-Plus benchmark 本身 (benchmark 三年后会饱和)，而在于它把 *"VLA 没在用 language"* 和 *"VLA 是 positional memorization"* 这两件事用 *clean ablation* 钉死了。这两件事 community 之前都 *suspect* 但没人这么系统地 *prove*。下一个十年的 VLA 研究议程，我觉得会被这篇 paper 显著塑形。
