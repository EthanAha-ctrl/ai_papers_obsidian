---
source_pdf: RL-VLM-F.pdf
paper_sha256: 6529e49400abf73387cfde32264731e5421d2d3acdb6bfd6db71dc026b3e0de5
processed_at: '2026-08-11T23:57:11-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RL-VLM-F 用人话说

## 一句话版本

**让 GPT-4V / Gemini 当你的 "RLHF 标注员"**——它看两张 robot 拍的照片，告诉你哪张更接近 task goal，然后你拿这些 preference label 去训一个 reward model，再用这个 reward model 跑 SAC 学 policy。整个过程不用人写一行 reward code。

就这么简单。paper 剩下的部分都是把这个 idea 跑通的各种工程细节。

---

## 为什么这件事是 non-trivial 的？

你可能觉得："这不就是把 RLHF 的 human annotator 换成 VLM 吗？有什么大不了的？"

Trick 在于：**直接问 VLM "这张图几分" 非常 noisy**。你试一下就知道，GPT-4V 给你打 0.7 分，下一张打 0.8 分，但这 0.1 的差异根本不可信——它下次问同样的图可能就给 0.6 了。CLIP-based 方法（[Rocamonde et al. 2023](https://openreview.net/forum?id=JUwczEJY8I)）也是这个问题，cosine similarity 的绝对数值跳动很大。

**但是**，如果你问 VLM "A 和 B 哪张更好？"，它其实相当 stable。这跟人类类似——你让我给两道菜各打 0-10 分我给不准，但你让我说哪道更好吃我能说。

所以 paper 的核心 insight 就是：

> **绝对评分难，相对比较容易。那就只问相对比较。**

这其实是 RLHF（[Christiano et al. 2017](https://arxiv.org/abs/1706.03741)）和 [PEBBLE](https://arxiv.org/abs/2106.05091) 早就发现的事情。RL-VLM-F 的贡献就是把这个 insight 推广到 VLM 上，把整个 pipeline 自动化了。

---

## 整个 pipeline 用大白话讲

想象你在教一个 robot arm 把 drawer 拉开。你只有一个文字描述 "open the drawer"，没有任何 reward function。

### Step 1: Policy 乱动

Robot 先瞎动一阵（SAC 探索），过程中每一步都存下：
- State $s_t$（low-dim，比如 joint angles + drawer position）→ 给 policy 用
- Image $I_t$（RGB render）→ 给 reward model 用

注意这里有个 split：**policy 吃 state，reward model 吃 image**。为什么不都吃 image？因为 image-based policy 训起来 sample efficiency 差很多。让 policy 吃干净的 state vector，把 "看图理解语义" 这个 dirty work 完全外包给 reward model。

### Step 2: 隔一阵子，从 image buffer 里抽两张图问 VLM

比如抽了一张 drawer 开了一半的图 A，一张 drawer 几乎没动的图 B。丢给 Gemini-Pro：

```
Task goal: to open the drawer
Image A: [drawer half open]
Image B: [drawer closed]

Which image better achieves the task goal?
0 = first image, 1 = second image, -1 = no preference
```

VLM 返回 `0`（意思是 A 更好）。存进 preference buffer $\mathcal{D}$。

### Step 3: 用这些 preference 训一个 reward model

Reward model 是个小 CNN（MetaWorld 用 4-layer CNN，SoftGym 用 ResNet-18），吃 image 输出一个 scalar $r_\psi(I)$。

训练 loss 用 Bradley-Terry model，本质上就是：

> "如果 VLM 说 A 比 B 好，就 push $r_\psi(A) > r_\psi(B)$"

具体怎么 push？用 softmax / sigmoid cross-entropy：

$$P_\psi[\sigma^1 \succ \sigma^0] = \frac{\exp(r_\psi(s^1))}{\exp(r_\psi(s^0)) + \exp(r_\psi(s^1))}$$

- $\sigma^0, \sigma^1$：要比较的两个 segment（这里就是两张 image，因为 $H=1$）
- $\succ$：preference 关系，$\sigma^1 \succ \sigma^0$ 读作 "segment 1 preferred over segment 0"
- $r_\psi(s^i)$：reward model 对第 $i$ 个 image 输出的 scalar
- $\psi$：reward network 的 weights

直觉：reward 差值越大，preference 概率越接近 1；reward 相等时概率 = 0.5。这等价于 $\text{sigmoid}(r_\psi(s^1) - r_\psi(s^0))$。

注意这里 reward 的 **绝对值** 无所谓，只有 **差值** 进了 sigmoid。这正对应了 VLM 给的也是 ordinal（序数）信息——只知道谁好谁坏，不知道好多少。

### Step 4: 用更新后的 reward model relabel 整个 replay buffer

这步很关键。SAC 的 replay buffer 里存了一堆旧 transition，当初打 reward 用的是旧版 $r_\psi$。现在 $r_\psi$ 更新了，得把所有旧 transition 的 reward 重新算一遍。

为什么必须这么做？因为 reward model 在持续学习，早期它的输出可能完全是垃圾。如果不 relabel，policy gradient 会基于过时的 reward 估计做更新，造成 reward drift 导致 instability。这个 trick 来自 [PEBBLE](https://arxiv.org/abs/2106.05091)。

### Step 5: 回到 Step 1 继续跑 SAC

循环往复。每隔 $K$（4000-5000 步）做一次 VLM query + reward update，每次 query $M$ 对（40-100 对），整个 training 总共 query $N$ 次（5K-20K 对）。

---

## 两阶段 prompting——这个 trick 很重要

作者发现直接问 VLM "哪张图更好？" 效果一般。他们改成了两步：

**Stage 1 (Analysis)**：让 VLM 先用文字描述两张图分别达到 task goal 的程度。自由生成，类似 Chain-of-Thought。

```
Task goal: to open the drawer
Image A: [...]
Image B: [...]

Describe how well each image achieves the task goal.
```

VLM 可能输出："Image A shows the drawer partially pulled out, approximately halfway open. Image B shows the drawer fully closed, no progress. Image A demonstrates better progress toward opening the drawer."

**Stage 2 (Labeling)**：把 Stage 1 的 response 塞回去，再让它输出最终 label。

```
[Stage 1 response]

Based on the above analysis, output a preference label:
0 = first image preferred
1 = second image preferred
-1 = no preference
```

这个两步走的核心 idea：**让 VLM 先 "想" 再 "答"**。直接让它输出 label，它容易 snap judgment；先让它 reason 一段，再让它 commit，accuracy 明显更高。这跟 [Chain-of-Thought prompting (Wei et al. 2022)](https://arxiv.org/abs/2201.11903) 是一脉相承的思路。

Ablation（Figure 8）证明了这个设计：4 个 task 里 3 个 two-stage 比 single-stage 好。

---

## 实验里最 interesting 的几个发现

### Finding 1: 全 7 个 task 都 outperform baseline

| Task | RL-VLM-F | CLIP Score | BLIP-2 | RoboCLIP | VLM Score | GT Pref |
|------|----------|------------|--------|----------|-----------|---------|
| CartPole | ✓ | ✓ | ✓ | - | 接近 | ✓ |
| Open Drawer | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Soccer | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Sweep Into | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Fold Cloth | ✓ | ✗ | ✗ | - | ✗ | ✓ |
| Straighten Rope | ✓ | ✗ | ✗ | - | ✗ | ✓ |
| Pass Water | ✓ | ✗ | ✗ | - | ✗ | ✓ |

CLIP / BLIP-2 / RoboCLIP 这些 contrastive 方法基本只能解 CartPole 这种最简单的 task。一碰到 deformable object 或 articulated object 全部歇菜。这印证了 contrastive embedding 在 robotics domain 的 OOD 问题——CLIP 训练数据是 web image + caption，对 simulator 渲染的 cloth / rope 根本没见过的样子。

### Finding 2: VLM Score 这个 baseline 在 Open Drawer 上 mode collapse

VLM Score 是让 VLM 直接给图打 0-1 分，然后用 MSE loss 训 reward model 去回归这个分。结果在 Open Drawer 上 reward model 直接学成 "永远输出 0"——因为 VLM 在 training 中给大部分图打 0 分，model 学到一个 degenerate solution：全输出 0 就能 minimize MSE。

这是 scalar reward 的一个经典 failure mode。Preference loss 不会这样，因为 preference loss 只关心 relative ordering，不存在 "全输出 0 就行" 的退化解。

### Finding 3: Sweep Into 上 RL-VLM-F 超过 GT preference

这个很有意思。Sweep Into 的 ground-truth reward 是 MetaWorld 作者写的，包含了 "grasping the cube" 这一项。但这个 task 实际上是 "push cube into hole"，根本不需要 grasp。结果 GT reward 让 agent 先去 grasp cube，反而干扰了学习。

RL-VLM-F 只用 "minimize the distance between the green cube and the hole" 这个 text 描述，VLM 看图判断 "cube 离 hole 近不近"，没有 spurious reward term。结果学出来的 policy 比 GT reward 还好。

**Takeaway：human 写的 dense shaped reward 不一定是最优的**，自然语言描述有时候反而更 "pure"，更能反映 task 的真实目标。这个 insight 对未来的 reward design 研究很有启发性。

### Finding 4: VLM accuracy 随 image pair 差异增大而提高（Figure 6）

这个直觉上很对：

- 比较两张几乎一样的图 → VLM 经常答 "no preference"（被丢弃）或答错
- 比较一张几乎 done 的图和一张初始状态的图 → VLM 几乎全对

这说明 random sampling pair 不是最优的。如果用 active learning，专门挑差异适中的 pair 来问，既能保证 VLM accuracy 高，又能提供 informative 的 supervision signal。作者在 future work 里提到了这个方向。

### Finding 5: Fold Cloth 上 GPT-4V 比 Gemini-Pro 好

Fold Cloth 任务视觉复杂（cloth 形状变化大），Gemini-Pro 在这上面表现差，GPT-4V 明显好（Figure 9）。其他 task 上 Gemini-Pro 够用。

这说明 **不同 VLM 在不同 task 上有 heterogeneous capability**。未来的 pipeline 可以根据 task 性质选合适的 VLM，甚至 ensemble 多个 VLM。

---

## 几个我个人觉得 "聪明" 的设计

### Smart Design 1: Reward model 和 policy 解耦 observation space

Policy 吃 state vector（39-dim for MetaWorld，4-dim for CartPole），sample efficiency 高。
Reward model 吃 image，需要 visual reasoning。

这俩不需要用同一种 observation。Reward model 训好后，把 image → scalar reward 这个映射学到了，policy 通过 reward gradient 间接利用 image 信息，但本身不需要处理 image。**等于把 "看图" 这个 dirty work 完全外包给 reward model，policy 享受 state-based learning 的高效性。**

### Smart Design 2: 把 robot 从 image 里抹掉

MetaWorld task 的 image 本来带 robot arm。作者用 simulator 让 robot transparent，image 里只有 target object（drawer, ball, cube）。

直觉：VLM 看 image 时会 focus 在 object 上，不被 robot pose 干扰。你想，如果 image 里有 robot arm 在某个位置，VLM 可能会被 arm 的位置误导，以为 "arm 在 goal 附近 = task done"，但其实 object 没动。

Real world 应用可以用 inpainting（[Bahl et al. 2022](https://arxiv.org/abs/2207.09450)）把 robot 抹掉。

### Smart Design 3: Reward model 是 small ensemble

用 3 个 reward model 的 ensemble（来自 PEBBLE），output 用 tanh 激活 bounded。Ensemble 帮助量化 reward uncertainty，减少单个 model 的 overfit。

### Smart Design 4: 不 fine-tune VLM

整个 pipeline 里 VLM 是 frozen oracle，只通过 prompt 来调用。这跟 [Ma et al. 2023a (LIV)](https://arxiv.org/abs/2306.00958) 和 [Mahmoudieh et al. 2022](https://arxiv.org/abs/2202.04333) 需要针对 task fine-tune CLIP 形成对比。

好处：
- 不需要 task-specific 训练数据
- VLM 升级（GPT-5, Gemini 2）自动受益
- Zero deployment cost for VLM

### Smart Design 5: GT reward 不一定是 upper bound

这个上面讲过。Sweep Into 的例子告诉我们，human 写的 shaped reward 有时候引入 spurious bias。自然语言 description 更贴近真实 task intent。

---

## 这篇 paper 在大图里的位置

我觉得这个工作处在几个 trend 的交汇处：

1. **Foundation model as oracle for RL** —— 从 [Eureka](https://arxiv.org/abs/2310.12931)（LLM 写 reward code）到 [Motif](https://arxiv.org/abs/2310.00166)（LLM 给 intrinsic reward）到 RL-VLM-F（VLM 给 preference），趋势是用 foundation model 替代 human supervision。

2. **Preference-based RL 的 revival** —— [Christiano 2017](https://arxiv.org/abs/1706.03741) 那篇之后，preference-based RL 一直因为 query 效率低没大规模铺开。现在有了 VLM 这个 "cheap annotator"，整个方向重新 viable 了。

3. **Vision-language model for robot control** —— 从 [CLIP-as-reward](https://openreview.net/forum?id=JUwczEJY8I) 到 [RoboCLIP](https://openreview.net/forum?id=DVlawv2rSI) 到 RL-VLM-F，趋势是用更强的 VLM（GPT-4V, Gemini）替代 CLIP-style contrastive model。CLIP 是 embedding-level alignment，VLM 是 reasoning-level alignment，后者明显更强。

4. **Decoupling perception from control** —— 传统的 end-to-end image-to-action policy sample efficiency 差。这篇 paper 把 perception 完全交给 reward model + VLM，policy 只处理 state。这个 decoupling 思路在 robotics 里越来越流行。

---

## Limitations 和我自己的思考

### Limitation 1: API cost

GPT-4V / Gemini API 不便宜。Fold Cloth 上因为 GPT-4V quota 限制只能做 500 queries。Scale 到 real robot continuous learning 的话，API cost 是大问题。

可能的解法：distill 一个 local VLM 出来当 annotator；或者用 active learning 大幅减少 query 数。

### Limitation 2: Single-image comparison

$H=1$ 意味着 VLM 只看单张图做比较。但很多 task 的 progress 是 dynamic 的——比如 "water spilling over time" 这种信息单张图 capture 不到。

未来工作可以扩展到 video segment comparison，让 VLM 比较两段短视频。这跟 [RoboCLIP](https://openreview.net/forum?id=DVlawv2rSI) 用 video-language model 的思路类似，但用 preference 而非 similarity。

### Limitation 3: 没有真实 robot 实验

全部在 simulation。Real world 的 image noise、lighting variation、occlusion 都没验证。不过作者提到可以用 inpainting 处理 robot removal，这说明他们在朝这个方向想。

### Limitation 4: Random sampling pair

当前从 buffer 随机抽 pair。如果能用 active learning 挑 "differ by a moderate amount" 的 pair（Figure 6 显示这种 pair accuracy 最高），query efficiency 能大幅提升。

可能的 active learning 策略：
- 用 reward model ensemble 的 disagreement 来选 pair（uncertainty sampling）
- 专门挑 reward difference 在某个 sweet spot 的 pair
- 主动探索 reward model 不熟悉的 image region

### Limitation 5: VLM bias inheritance

Impact Statement 里作者自己提到。如果 VLM 对某些 visual pattern 有 bias（比如对颜色、形状的 stereotype），这个 bias 会传到 reward model 再传到 policy。Safety-critical 应用需要 audit VLM 的决策依据。

### 我自己的思考：reward model 会不会成为 bottleneck？

整个 pipeline 里 reward model 是个 ResNet-18，相对简单。如果 task 变得更复杂（比如 long-horizon manipulation，或者 visual appearance 变化更大），这个小 reward model 可能学不动。是否可以用更大的 backbone（ViT, DINOv2 feature + linear head）值得探索。

另外，reward model 的 capacity 决定了它能从 VLM label 里榨取多少信息。如果 VLM 给的 label 蕴含了 fine-grained 的视觉 reasoning（比如 "drawer 开了 50% vs 70%"），但 reward model 太简单拟合不了，信息就浪费了。

---

## 一句话总结

**RL-VLM-F = RLHF + VLM replaces human annotator + decoupled state/image observation**。

核心贡献就这一句。剩下都是 implementation details，但 implementation details 决定了它 work 不 work。这篇 paper 的价值在于把一个 idea（VLM as preference oracle）做成了一个 robust 的 pipeline，并在一系列 challenging task（deformable object manipulation）上验证了 effectiveness。

---

## 参考

- [RL-VLM-F Project Page](https://rlvlmf2024.github.io/)
- [Christiano et al. 2017 - RLHF](https://arxiv.org/abs/1706.03741)
- [PEBBLE](https://arxiv.org/abs/2106.05091)
- [Eureka](https://arxiv.org/abs/2310.12931)
- [Text2Reward](https://arxiv.org/abs/2309.11489)
- [Motif](https://arxiv.org/abs/2310.00166)
- [Constitutional AI / RLAIF](https://arxiv.org/abs/2212.08073)
- [CLIP Score baseline](https://openreview.net/forum?id=JUwczEJY8I)
- [RoboCLIP](https://openreview.net/forum?id=DVlawv2rSI)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [SAC](https://arxiv.org/abs/1801.01290)
- [MetaWorld](https://arxiv.org/abs/1910.10897)
- [SoftGym](https://arxiv.org/abs/2011.07254)
- [GPT-4V System Card](https://openai.com/research/gpt-4v-system-card)
- [Gemini Technical Report](https://arxiv.org/abs/2312.11805)
- [BLIP-2](https://arxiv.org/abs/2301.12597)
- [CLIP](https://arxiv.org/abs/2103.00020)
- [ResNet](https://arxiv.org/abs/1512.03385)
- [Bradley-Terry Model](https://www.jstor.org/stable/2334029)
- [Bahl et al. 2022 - Inpainting for Robot Removal](https://arxiv.org/abs/2207.09450)

---

# RL-VLM-F 详细技术讲解

## 1. 核心 Intuition：把 VLM 当作 "Pairwise Preference Oracle"

这篇 paper 的核心 idea 是：**VLM 直接输出 scalar reward 非常 noisy**，但 VLM 做 "这两张图哪张更接近 task goal" 这种 **pairwise comparison 相对稳定**。利用这一点，作者把 RLHF（Reinforcement Learning from Human Preferences, [Christiano et al. 2017](https://arxiv.org/abs/1706.03741)）的 human annotator 直接替换成 VLM，自动化整个 preference-based RL pipeline。

这等于把 [PEBBLE (Lee et al. 2021)](https://arxiv.org/abs/2106.05091) 里的 human query 部分替换为 VLM query。VLM 不需要任何 RL 训练，只是被当作冻结的 oracle 反复 prompt。

---

## 2. 整体架构解析

整个 pipeline 是一个 alternating cycle，有三个组件相互交互：

```
┌─────────────────────────────────────────────────────────────┐
│  Policy π_θ (SAC, state-based)  ←→  Environment (render I)   │
└─────────────────────────────────────────────────────────────┘
              ↓ image obs stored in T           ↑ relabel rewards
┌─────────────────────────────────────────────────────────────┐
│  Sample pair (σ^0, σ^1) from T  ──→  VLM  ──→ label y       │
│  (Gemini-Pro / GPT-4V, frozen)                                │
└─────────────────────────────────────────────────────────────┘
              ↓ preference buffer D
┌─────────────────────────────────────────────────────────────┐
│  Reward model r_ψ (CNN / ResNet-18, image-based)             │
│  Trained with Bradley-Terry cross-entropy loss               │
└─────────────────────────────────────────────────────────────┘
              ↓ reward
   全部 SAC replay buffer transitions 用新 r_ψ relabel
```

**关键设计选择**：
- **Policy 用 state-based obs**，而 **reward model 用 image-based obs**。这避免了 high-dim image 直接进 policy 时的 sample complexity 问题，把 image reasoning 完全交给 reward model。
- **Reward model 是独立的小网络**（4-layer CNN for MetaWorld, ResNet-18 for SoftGym），不是 VLM 本身。这避免了每次 reward 计算 call API 的延迟，并把 VLM 的 noisy label "smoothing" 进一个 dense reward function。
- **VLM 只 query 稀疏的 image pair**，频率 $K$（4000-5000 env steps per query batch），每次 $M$ 对（40-100 对）。这是 cost-effective 的关键。

---

## 3. 公式深度解析

### 3.1 Return 定义

$$R = \sum_{k=0}^{\infty} \gamma^{k} r(s_k, a_k)$$

- $k$：timestep index
- $\gamma \in [0,1)$：discount factor，越早的 reward 权重越大
- $s_k$：state at timestep $k$
- $a_k$：action at timestep $k$
- $r(s_k, a_k)$：reward function（这里其实论文里用的是 $r_\psi(s_t)$，state-only，不依赖 action）

### 3.2 Bradley-Terry Preference Model（[Bradley & Terry 1952](https://www.jstor.org/stable/2334029)）

$$P_\psi[\sigma^1 \succ \sigma^0] = \frac{\exp\left(\sum_{t=1}^{H} r_\psi(s_t^1)\right)}{\sum_{i \in \{0,1\}} \exp\left(\sum_{t=1}^{H} r_\psi(s_t^i)\right)}$$

逐项解读：
- $\sigma^i$：第 $i$ 个 segment。这里 $i \in \{0, 1\}$ 表示一对比较的 segment
- $\sigma^1 \succ \sigma^0$：segment 1 preferred over segment 0
- $s_t^i$：第 $i$ 个 segment 的第 $t$ 个 state
- $H$：segment 长度。本 paper 取 $H = 1$，即 segment 就是单张 image，简化为：

$$P_\psi[\sigma^1 \succ \sigma^0] = \frac{\exp(r_\psi(s^1))}{\exp(r_\psi(s^0)) + \exp(r_\psi(s^1))} = \text{sigmoid}(r_\psi(s^1) - r_\psi(s^0))$$

- $r_\psi$：参数为 $\psi$ 的 reward network
- $\psi$：reward network 的 weights，需要通过 loss 学出来

这种 Boltzmann / softmax 形式的好处：reward 之间的 **差值** 决定 preference 概率，因此 reward 的绝对 scale 不重要，只有 **相对排序** 重要。这恰好对应了 VLM 给的也是 ordinal preference。

### 3.3 Cross-Entropy Loss

$$\mathcal{L}_{\text{Reward}} = -\mathbb{E}_{(\sigma^0, \sigma^1, y) \sim \mathcal{D}} \left[ \mathbb{I}\{y = (\sigma^0 \succ \sigma^1)\} \log P_\psi[\sigma^0 \succ \sigma^1] + \mathbb{I}\{y = (\sigma^1 \succ \sigma^0)\} \log P_\psi[\sigma^1 \succ \sigma^0] \right]$$

- $\mathcal{D}$：preference dataset，存了所有 VLM 给过的 $(\sigma^0, \sigma^1, y)$ 三元组
- $y \in \{-1, 0, 1\}$：label，-1 表示 incomparable（被丢弃，不进 loss），0 表示 segment 0 preferred，1 表示 segment 1 preferred
- $\mathbb{I}\{\cdot\}$：indicator function，条件为真取 1，否则 0
- $\log P_\psi$：取 log 是为了把 softmax 概率转成 log-likelihood，cross-entropy 标准做法

直觉：如果 label 说 $\sigma^0 \succ \sigma^1$，那我们就 push $r_\psi(s^0)$ 比 $r_\psi(s^1)$ 大；反之亦然。整个 loss 等价于 binary classification with sigmoid cross-entropy on reward difference。

---

## 4. 两阶段 Prompting 策略（核心技术贡献之一）

这是 paper 里最 "trick-y" 的设计。作者 query VLM 两次：

**Stage 1 - Analysis Stage**：
- Input：image pair + task goal text
- Output：VLM 自由生成文字，描述并比较两张 image 完成任务的程度
- 类似 "Chain-of-Thought" reasoning，让 VLM 先 think out loud

**Stage 2 - Labeling Stage**：
- Input：Stage 1 的 free-form response + 重复的提问
- Output：单 token label $y \in \{-1, 0, 1\}$

为什么这么做？因为直接让 VLM 输出 label，它容易 "snap judgment"；先让它 reasoning，再让它 commit label，accuracy 更高。这在 LLM/VLM 社区里类似 [Chain-of-Thought](https://arxiv.org/abs/2201.11903) 的思路，但用在了 preference labeling 上。

Ablation（Figure 8）：在 4 个 task 里 3 个 two-stage 比 single-stage 好。Sweep Into 上两者差不多，其他三个 task 显著好。

---

## 5. Algorithm 1 流程逐行解读

| Line | 操作 | 直觉 |
|------|------|------|
| 5-9 | Policy rollout，存 transition 到 $\mathcal{B}$，存 image 到 $\mathcal{T}$ | SAC standard rollout，但额外 store image |
| 10-13 | 用 $r_\psi$ 给 $\mathcal{B}$ 的 transitions 打 reward，做 SAC gradient step | 标准 off-policy RL |
| 15 | 每 $K$ steps 触发一次 VLM query + reward update | $K$=4000-5000，控制 API cost |
| 16-20 | 从 $\mathcal{T}$ 随机采 $M$ 对 image，query VLM，存到 $\mathcal{D}$ | 关键：random pairs（非 active） |
| 21-24 | 用 $\mathcal{D}$ 训 $r_\psi$，$\mathcal{N}_r$ 个 gradient step | Cross-entropy loss |
| 25 | Relabel 整个 $\mathcal{B}$ 用 updated $r_\psi$ | 关键 trick from PEBBLE：让旧 transition 用新 reward，避免 reward drift 问题 |

**为什么 relabel 整个 buffer**：reward model 在持续学，旧 transitions 上用旧 reward 标的数据变成 stale。如果不 relabel，policy gradient 会基于过时的 reward 估计，造成 instability。这是 PEBBLE 的核心贡献之一。

**为什么 image pair 是随机采的**：随机采会让 VLM 经常比较 "差异很小" 的 image pair，accuracy 低（见 Figure 6）。作者在 future work 里提到用 active learning 选 informative pairs 可能更好。

---

## 6. 实验设计与 Baselines

### 6.1 7 个 Task 的分布

| Domain | Task | 关键挑战 |
|--------|------|----------|
| Classic control | CartPole | 简单，sanity check |
| MetaWorld (rigid) | Open Drawer | Articulated object |
| MetaWorld (rigid) | Soccer | Push ball to goal |
| MetaWorld (rigid) | Sweep Into | Sweep cube into hole |
| SoftGym (deformable) | Fold Cloth | Diagonal fold，cloth keypoints tracking |
| SoftGym (deformable) | Straighten Rope | Rope shape |
| SoftGym (deformable) | Pass Water | Liquid，spill 风险 |

Deformable object 是关键 motivation——这类 task 用 text 描述 state 很难（cloth 怎么 fold 用语言描述？），但用 image VLM 直接看就行。

### 6.2 Baselines 对比

| Baseline | 假设 | 弱点 |
|----------|------|------|
| **VLM Score** | 同样用 VLM，但直接 output raw scalar | Reward noisy，degenerate（Open Drawer 上全输出 0） |
| **CLIP Score** ([Rocamonde et al. 2023](https://openreview.net/forum?id=JUwczEJY8I)) | 用 CLIP cosine sim between image & text | 只能解 CartPole，其他全 fail |
| **BLIP-2 Score** ([Li et al. 2023](https://arxiv.org/abs/2301.12597)) | 用 BLIP-2 替代 CLIP | 同上，noisy |
| **RoboCLIP** ([Sontakke et al. 2023](https://openreview.net/forum?id=DVlawv2rSI)) | 用 S3D video-language model | text-only 变体表现差 |
| **GT Preference** | 用 GT reward function 给 preference | Oracle upper bound |

### 6.3 主要实验结果（Figure 4 + Figure 5）

RL-VLM-F：
- 在 **全部 7 个 task** 上 outperform 所有 non-oracle baseline
- 在 **6/7 个 task** 上 match 或 surpass GT preference
- 在 **Sweep Into** 上 **超过** GT preference——这是因为 author-defined reward 有 "grasping cube" 的 term，对 push into hole 任务是 spurious reward；RL-VLM-F 只用 "minimize distance" 的 text，避免了 reward shaping 的 bias

这是 paper 里一个非常 insightful 的 observation：**GT reward ≠ optimal reward**，特别是作者写的 dense shaped reward 有时反而引入 spurious correlation。用 VLM + 自然语言描述反而更 "pure"。

### 6.4 Preference Accuracy 分析（Figure 6）

x 轴：image pair 之间 ground-truth task progress 差异的 10 个 bin
y 轴：VLM label correct / incorrect / no-preference 的比例

观察：
- 差异越大，accuracy 越高，no-preference 越少
- 差异很小时，VLM 大量 output "no preference"（被丢弃）
- CartPole, Open Drawer, Soccer 上 trend 最清晰
- 这验证了 paper 的核心 intuition：**VLM 做 comparison 比做 absolute scoring 更稳**

### 6.5 Learned Reward 对齐分析（Figure 7）

把 learned reward 沿 expert trajectory 画出来，对比 ground-truth task progress。
- RL-VLM-F：reward 随 task progress 单调上升（虽然有 noise），最终达到峰值
- VLM Score：Open Drawer 上 reward 恒为 0（mode collapse）
- CLIP / BLIP-2 score：noisy，没有 monotonic trend

这解释了为什么 RL-VLM-F 学得到 policy，而 baseline 学不到。

---

## 7. 实现细节中的 "Tricks"

### 7.1 Robot Removal from Image

MetaWorld task 里，作者把 robot 从 image 里 **抹掉**（用 simulator 让 robot transparent）。理由：task 是 object-centric，VLM 应该 focus 在 object 上而不是 robot pose。Real world 应用可以用 inpainting（参考 [Bahl et al. 2022](https://arxiv.org/abs/2207.09450)）。

这是个很 practical 的 insight——**让 VLM 看 task-relevant 的部分**，不要让它分心。

### 7.2 VLM 选择

- 默认 **Gemini-Pro**（[Team et al. 2023](https://arxiv.org/abs/2312.11805)）
- **Fold Cloth** 上 GPT-4V 更好（[OpenAI 2023](https://openai.com/research/gpt-4v-system-card)），因为 cloth 视觉更复杂
- 不全用 GPT-4V 是因为 **API quota 限制**

这暗示了 VLM 能力的 heterogeneous distribution——不同 VLM 在不同 task 上强项不同。未来 VLM 更强了，整个 pipeline 直接受益（modularity）。

### 7.3 Reward Model 架构

- MetaWorld + CartPole：4-layer CNN
- SoftGym：ResNet-18（[He et al. 2016](https://arxiv.org/abs/1512.03385)）
- 3 个 reward model 的 ensemble（来自 PEBBLE）
- 输出用 tanh 激活，bounded reward

### 7.4 Feedback Schedule（Table 1）

| Task | M (queries/session) | K (env steps/session) | N (total budget) |
|------|---------------------|------------------------|------------------|
| Open Drawer / Soccer / Sweep Into | 40 | 4000 | 20000 |
| CartPole | 50 | 5000 | 10000 |
| Cloth Fold | 50 | 1000 | 500（quota 限制）|
| Straighten Rope / Pass Water | 100 | 5000 | 12000 |

Fold Cloth 因为 GPT-4V quota 限制只 query 500 次，居然还能 work，说明 algorithm 对 label 数量不极度敏感。

---

## 8. 与 Related Work 的对比 intuition

### 8.1 vs Eureka / Text2Reward（[Ma et al. 2023](https://arxiv.org/abs/2310.12931), [Xie et al. 2023](https://arxiv.org/abs/2309.11489)）

这些方法让 LLM **写 Python reward code**。问题：
- 需要 **environment source code**（access to ground-truth state variables）
- 对 deformable object（cloth, rope, water）写 code 很难——怎么写 "folded diagonally" 的 code？

RL-VLM-F 不需要代码，直接用 image，所以对 deformable object 友好。

### 8.2 vs Motif / RLAIF（[Klissarov et al. 2023](https://arxiv.org/abs/2310.00166), [Bai et al. 2022](https://arxiv.org/abs/2212.08073)）

- Motif：LLM 比较 **text description of state**，生成 intrinsic reward。需要 ground-truth state→text 的 mapping。
- RLAIF：LLM 直接生成 preference label 用于 alignment。

RL-VLM-F 的区别：用 **VLM + image**，不需要 state→text 的 mapping。这对 deformable object 关键——cloth 怎么用 text 描述？

### 8.3 vs CLIP-based 方法（[Rocamonde et al. 2023](https://openreview.net/forum?id=JUwczEJY8I), [Sontakke et al. 2023](https://openreview.net/forum?id=DVlawv2rSI)）

CLIP reward = cosine similarity between image embedding and text embedding。

问题：
- CLIP 训练数据是 natural image + caption，**robotics 视域外**，similarity score noisy
- Score 是 absolute scalar，不经过 "比较" smoothing
- 实验上只能解 CartPole，其他 task 全 fail

RL-VLM-F 用的是 general-purpose VLM（GPT-4V / Gemini），这些模型在 web-scale data 上训练，reasoning 能力远超 CLIP 的 contrastive embedding。同时 preference 的 ordinal nature 比 scalar 更鲁棒。

---

## 9. Limitations 与 Future Work

作者自己提到的：
1. **Active learning**：当前 random sampling 浪费 query，可以选 informative pairs
2. **Long horizon task**：目前 episode 不长，长 horizon 需要分解成 subtask
3. **Real-world transfer**：目前只在 simulation 验证

我觉得还有几个 implicit 的 limitation：
- **VLM API cost**：Fold Cloth 上 500 queries 都受限，scale 到复杂 task 是问题
- **VLM bias inheritance**：Impact Statement 里提到 VLM 的 bias 可能传到 policy
- **Single-image assumption**：H=1，不能 capture dynamics（比如 "water spilling over time" 的视频 reasoning）
- **Reward model 还是 image-based，需要 separate rendering pipeline**——real robot 上需要 camera + 可能 inpainting
- **No active querying of "where to explore"**——只 active 在 reward labeling 层面，不在 exploration 层面

---

## 10. 关键 Takeaways for Building Intuition

1. **Ordinal > Scalar**：让模型比较两张图，比让它打分更稳定。这是 RLHF 的核心思路被迁移到 VLM-as-annotator 的场景。
2. **VLM 是 frozen oracle**：不需要 fine-tune VLM，直接 prompt。这跟 CLIP-based 方法需要 fine-tune 形成对比。
3. **Decouple policy and reward model 的 observation space**：policy 用 state，reward model 用 image。这样既享受 image 的语义丰富性，又享受 state-based policy learning 的 sample efficiency。
4. **Two-stage prompting = CoT for preference**：让 VLM 先 reasoning 再 commit label，显著提 accuracy。
5. **Reward relabeling 是关键 stability trick**：reward model 在变，旧数据要 relabel。
6. **GT reward 不一定是 best reward**：Sweep Into 的案例说明 human reward shaping 可能引入 spurious correlation，自然语言描述反而更 "pure"。

---

## References

- [RL-VLM-F Project Page](https://rlvlmf2024.github.io/)
- [Christiano et al. 2017 - Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741)
- [PEBBLE (Lee et al. 2021)](https://arxiv.org/abs/2106.05091)
- [Eureka (Ma et al. 2023)](https://arxiv.org/abs/2310.12931)
- [Text2Reward (Xie et al. 2023)](https://arxiv.org/abs/2309.11489)
- [CLIP (Radford et al. 2021)](https://arxiv.org/abs/2103.00020)
- [BLIP-2 (Li et al. 2023)](https://arxiv.org/abs/2301.12597)
- [RoboCLIP (Sontakke et al. 2023)](https://openreview.net/forum?id=DVlawv2rSI)
- [VLMs as Zero-shot Reward Models (Rocamonde et al. 2023)](https://openreview.net/forum?id=JUwczEJY8I)
- [Motif (Klissarov et al. 2023)](https://arxiv.org/abs/2310.00166)
- [Constitutional AI / RLAIF (Bai et al. 2022)](https://arxiv.org/abs/2212.08073)
- [SAC (Haarnoja et al. 2018)](https://arxiv.org/abs/1801.01290)
- [MetaWorld (Yu et al. 2020)](https://arxiv.org/abs/1910.10897)
- [SoftGym (Lin et al. 2021)](https://arxiv.org/abs/2011.07254)
- [Bradley-Terry Model (1952)](https://www.jstor.org/stable/2334029)
- [Chain-of-Thought Prompting (Wei et al. 2022)](https://arxiv.org/abs/2201.11903)
- [ResNet (He et al. 2016)](https://arxiv.org/abs/1512.03385)
- [GPT-4V System Card](https://openai.com/research/gpt-4v-system-card)
- [Gemini Technical Report](https://arxiv.org/abs/2312.11805)
- [Human-to-Robot Imitation (Bahl et al. 2022)](https://arxiv.org/abs/2207.09450)
