---
source_pdf: LIV Language-Image Representations and Rewards for Robotic Control.pdf
paper_sha256: a4cba376ba47cb01610a0995c71b61a6dc21781c50b1c0fe6650ddc408392284
processed_at: '2026-08-05T15:10:28-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# LIV 用人话讲

## 问题是什么

你有一个机器人，你想跟它说"把苹果放进黑锅里"，它就得去干。

这件事难在哪？你需要两样东西：

**第一样**：得让机器人的"脑子"里，"苹果"这个画面和"apple"这个词指向同一个地方。这叫 language grounding。CLIP 干的就是这个——把图像和文本拉到同一个 embedding space 里，让匹配的 pair 距离近，不匹配的远。

**第二样**：得让机器人知道"我现在离完成这个任务还有多远"。比如它刚伸出手抓苹果，离目标还挺远；已经把苹果拎到锅上方了，离目标很近了。这个"离目标多远"的度量，在 RL 里叫 value function。VIP 这篇 paper 干的就是这个——从人类视频里学一个隐式的 value function，不用 action label。

问题是：**这两个东西一直是分开做的**。

CLIP 不懂时序，给它一段视频，它只看最后一帧和文字配不配，中间那些帧全扔了。所以 CLIP 学出来的 representation 没法告诉你"任务进行到哪一步了"。

VIP 懂时序，能学 value function，但它只处理图像，不碰语言。你想让它听人话干活，还得外接一个 language encoder，但那个 encoder 在 pre-training 阶段根本没被优化过，只是个被动的 feature extractor。

所以现实里大家怎么干呢？拿 CLIP 当 visual backbone，外接一个 DistilBERT 处理语言，两个 encoder 各学各的，训练目标也各是各的。这种 ad-hoc 拼接方式效果就一般。

---

## LIV 的核心发现

LIV 的作者发现了一个特别简单但很关键的事实：

**CLIP 的目标函数，其实是 LIV 目标函数的一个特例。**

怎么说呢。VIP 有一个语言版本的变体叫 VIP-L，就是把 image goal 换成 language goal。你拿 VIP-L 去优化，正常视频数据上学到的是时序 value 结构。

但你考虑一个极端退化的情况：视频不是正常视频，而是同一个 goal frame 重复两次——就是一个"假视频"，从头到尾就一帧，配上对应的文字。

你把 VIP-L 套到这种退化数据上，你会发现它**正好**变成 CLIP 的 InfoNCE loss。

数学上就这么回事：退化分布下，初始帧等于 goal 帧，相邻帧也等于 goal 帧，代进去一化简，所有时序项都塌缩掉，剩下的就是 image-text 对的对比学习目标。差一个常数。

这个发现的 implications 很大：

1. CLIP 是 LIV 在"没有时序"数据上的特例
2. LIV 是 CLIP 在"有时序"数据上的自然推广
3. 任何带文字标注的视频，都可以通过"重复最后一帧"这种 trivial augmentation 同时提供时序信号和 CLIP 信号
4. 所以你不需要 separately 设计 visual SSL loss 和 CLIP loss 再加权，它们本来就是一个东西的两个 instance

---

## LIV 怎么实现

基于上面那个发现，实际实现非常简单：

**LIV = VIP-I（图像那边的 value learning）+ InfoNCE（图像-语言对齐）**

就这么两个 loss 加起来，1:1 配，不用调权重。

- VIP-I 那部分：拿视频里的中间帧和 goal frame 算 similarity，强制相邻帧之间满足 Bellman consistency。这让 visual encoder 学到"离 goal 多远"的时序结构。
- InfoNCE 那部分：拿 goal frame 和对应文字做对比学习。这让 image 和 language 拉到同一个空间里。

两个 loss 通过共享的 visual encoder 自动桥接：VIP-I 让图像 embedding 有 temporal value 结构，InfoNCE 让图像 embedding 和语言 embedding 对齐，合起来就是"一个既有 value 结构又能听懂语言的 multi-modal embedding"。

为什么是 1:1 不用调？因为理论推导里 CLIP 就是 VIP-L 在退化分布下的形式，所以两个 loss 天然是同一个东西的两个 instance，权重应该相等。这点在实验里也验证了——不像 TCN+CLIP 那种 ad-hoc 组合需要调 α 调到头秃。

---

## 同一个 objective 干三件事

LIV 最优雅的地方是：**同一个 objective 可以用于 pre-training、fine-tuning、和 reward specification**。

**Pre-training**：在 EpicKitchen 上跑，90k 段人类第一视角做饭视频，20M 帧，20k 句文字标注。机器人数据完全没参与。

**Fine-tuning**：拿到机器人领域的小数据集（比如 FrankaKitchen 250 个 demo），用同一个 LIV objective 再跑 10k 步。不用换 loss，不用调超参，直接跑。

**Reward specification**：学完的 similarity function 直接可以当 reward 用。给定一个 language goal，每一帧算一个 similarity 到 goal 的距离，相邻帧之间的 similarity 差就是 reward。这叫 potential-based reward，理论上不改变 optimal policy。

你拿这个 dense reward 喂给 CEM 或 MPPI 这种 model-based planner，就能在没有 policy 训练的情况下直接规划出动作序列完成任务。

---

## 实验里最 striking 的结果

### 1. Zero-shot reward 能 detect sub-optimal action

LIV 在 EpicKitchen 上 pre-train 完，直接拿去给**没见过的机器人视频**打分。结果发现 similarity 曲线能跟踪任务进度——随着机器人推进任务，距离 goal 越来越近。

更厉害的是，有个视频里机器人去"把帽子放瓶子上"，中间把帽子举得老高，属于没必要的多余动作。LIV 的 cost 曲线上中间冒出一个 bump，正好对应那段多余动作。也就是说 LIV 能 detect 出"这一段是绕远了"。

对比 CLIP 在同样的视频上画曲线，基本是噪声，什么也看不出来。CLIP 根本不懂"进度"这个概念。

### 2. CLIP fine-tune 越调越烂

拿 CLIP 在机器人数据上 fine-tune，FrankaKitchen 上性能从 22 掉到 14。为什么？因为 CLIP fine-tune 只对齐 goal frame 和文字，中间帧它不管，导致中间帧的 representation 崩掉。LIV fine-tune 不会这样，因为 VIP-I 那部分强制了 Bellman consistency，所有中间帧都得好好表现。

### 3. R3M 的"假"语言理解

R3M 是另一个 pre-trained visual representation，号称也支持 language-conditioned reward。但作者做了个狠实验：把语言 goal 换成**随机文字**，R3M 的"成功率"居然没掉，甚至略微上升。

这说明什么？说明 R3M 根本没在理解语言，它只是在检测"画面有没有变化"。它训练时学到的是"时间隔得远的帧 similarity 低"，这跟"任务完成了没"完全是两回事。FrankaKitchen 上随便动一下就能"完成 8.8% 任务"是因为 kitchen 任务都涉及移动物体，但 MetaWorld 上需要精细的 language grounding（比如区分"左转"和"右转"），R3M 就只剩 18.3%。

LIV 在 random language goal 上直接归零，说明它是真听懂了。

### 4. Fine-tune 10 个 demo 顶 50 个

FrankaKitchen 上，拿 CLIP 当 base model，10 个 demo LIV fine-tune 后性能 22.0，跟 CLIP 不 fine-tune 用 50 个 demo 的性能（22.0）持平。等于说 LIV fine-tune 把数据效率提升了 5 倍。

---

## 为什么 LIV 比 ad-hoc 组合强

你可能会问：那我拿 CLIP loss + 任意一个 visual SSL loss（比如 TCN）加权组合不也行吗？为什么非得是 LIV？

试了。TCN+CLIP 这个组合对权重 α 极度敏感。FrankaKitchen 上 α=0.5 最好，但同样 α=0.5 在 MetaWorld 上直接 diverge 训不动。α=0.1 在 MetaWorld 上不 diverge 但性能烂。你没有 oracle 来调这个 α，因为 offline 设置下没法 rollout 评估。

LIV 不存在这个问题。VIP 和 CLIP 不是两个 unrelated loss 拼一起，它们在退化分布下是同一个东西，所以 1:1 配理论上就对了，所有环境都用同一组超参。

---

## 一句话总结

LIV 发现 CLIP 是 value-based video learning 在退化数据上的特例，于是用同一个 loss 同时学 language grounding 和 temporal value，一个 objective 走通 pre-training、fine-tuning、reward 三件事，不用调权重，比所有 ad-hoc 拼接方案都强。

这个工作对你做 VLA 的启示可能是：纯 contrastive image-text learning 学不出 control 需要的"任务进度"概念，必须有显式的时序结构参与。光靠大数据 co-training 让模型自己"悟出来"是低效的，结构化 prior 还是有用的。

---

# LIV: Language-Image Value Learning 深度讲解

## 1. 论文核心 intuition

LIV 解决的是 robot learning 里一个非常根本的张力：**language grounding** 与 **temporal decision-making** 这两件事在 representation learning 里通常被割裂处理。CLIP 只看静态 image-text 对齐，丢掉了视频里的时序结构；VIP/R3M 学了时序 value 但语言模态靠 ad-hoc 拼接。LIV 的核心 insight 来自一个看似简单但很深刻的数学事实——当你把 VIP 的 language goal 版本（VIP-L）应用到一个"退化"的视频分布（每帧都重复 goal frame）上，它就 exactly 退化成 CLIP 的 InfoNCE。

这个事实的 implication 很大：**CLIP 是 LIV 在静态数据上的特例**，所以 LIV 是 CLIP 在时序决策数据上的 natural generalization。这就给出了一个非常优雅的设计——pre-training 用 LIV（=VIP-I+InfoNCE，1:1 配比，无需调权重），fine-tuning 也用 LIV。两个阶段共用同一个 objective，避免了像 TCN+CLIP 那样需要调 α 的尴尬。

参考链接：
- 项目主页：https://penn-pal-lab.github.io/LIV
- 代码：https://github.com/penn-pal-lab/LIV
- arXiv：https://arxiv.org/abs/2302.12766（同期相关工作 Karamcheti et al.）
- VIP 前作：https://arxiv.org/abs/2210.00030
- CLIP：https://arxiv.org/abs/2103.00020
- R3M：https://arxiv.org/abs/2203.12601
- EpicKitchen：https://epic-kitchens.github.io/

---

## 2. 数学层面的拆解

### 2.1 VIP 的 dual RL 起源（Eq.1）

先理解 VIP。VIP 的目标来自 goal-conditioned RL 的 Fenchel dual 形式。最原始的 goal-conditioned value function 满足 Bellman equation：

$$V^*(o;g) = -1 + \gamma \mathbb{E}_{o' \sim P(\cdot|o)}[V^*(o';g)]$$

这里 reward 恒为 -1（每个时间步扣 1 分），V 的取值范围是 $[-\frac{1}{1-\gamma}, 0]$。VIP 不直接学 V，而是用 similarity 来 implicit 参数化：

$$V(o;g) := S(\phi(o), \phi(g))$$

其中 $\phi$ 是 visual encoder，$S$ 是某种 similarity metric（VIP 用负 L2）。Eq.1 的两项分别对应：
- **第一项** $(1-\gamma)\mathbb{E}_{\mu_0(o;g)}[-S(\phi(o);\phi(g))]$：约束初始帧到 goal 的 value。系数 $(1-\gamma)$ 是为了把 $V \in [-\frac{1}{1-\gamma}, 0]$ 归一化到 $[-1, 0]$ 范围。
- **第二项** $\log \mathbb{E}_{(o,o';g) \sim D}[\exp(S(\phi(o);\phi(g)) + 1 - \gamma S(\phi(o');\phi(g)))]$：这是 Bellman consistency 的 log-sum-exp / dual 形式。直觉是：对相邻帧对 $(o, o')$，要求 $S(\phi(o);\phi(g)) \approx -1 + \gamma S(\phi(o');\phi(g))$，即当前 value 等于即时 reward (-1) 加上 discounted 下一帧 value。用 log-sum-exp 是为了 robust 化 dual formulation，避免 hard constraint。

变量说明：
- $\phi$：visual encoder，$\phi: \mathbb{R}^{H \times W \times 3} \to \mathbb{R}^K$
- $o, o'$：相邻两帧 observation
- $g$：goal frame（视频最后一帧）
- $\mu_0(o;g)$：以 goal g 为条件的初始帧分布
- $D(o, o'; g)$：以 goal g 为条件的相邻帧对分布
- $\gamma$：discount factor，论文里 pre-training 用 0.98，fine-tuning 用 0.98 或 0.96

### 2.2 LIV 的多模态扩展（Eq.4）

直接把 image goal $\phi(g)$ 换成 language goal $\psi(l)$，就得到 VIP-L。完整的 LIV 目标 Eq.4 是 VIP-I + VIP-L 的简单并列：

$$\mathcal{L}(\phi,\psi) = \mathcal{L}_{\text{VIP-I}} + \mathcal{L}_{\text{VIP-L}}$$

乍一看这个目标没有任何 cross-modal alignment 项——image goal 和 language goal 各自独立地驱动 visual encoder，二者之间没有显式约束。这看起来"不对"，因为如果没有 alignment，语义相同的 image goal 和 language goal 可能在 embedding space 里隔得很远，无法 grounding。

### 2.3 Proposition 1：退化分布下的等价性

这是论文最关键的理论 insight。考虑一个退化视频分布 $D = \{v := ((g, g); l)\}$——每个"视频"就是同一个 goal frame $g$ 重复两次，配上 text $l$。把这种数据代入 VIP-L：

$$\mathcal{L}_{\text{VIP-L}}(\phi,\psi) = \mathbb{E}_{p(g,l)}\left[-\log \frac{e^{(1-\gamma) S(\phi(g);\psi(l))}}{\mathbb{E}_{D(g')}[e^{(1-\gamma) S(\phi(g');\psi(l))}]}\right] + 1$$

**推导直觉**：在退化分布下，初始帧 $o = g$，相邻帧 $o = o' = g$。代入 VIP-L 的第二项：

$$\log \mathbb{E}[\exp(S(\phi(g);\psi(l)) + 1 - \gamma S(\phi(g);\psi(l)))]$$

$$= \log \mathbb{E}[\exp(1 + (1-\gamma) S(\phi(g);\psi(l)))]$$

常数 $e^1$ 提出来后就是 +1 常数项，剩下的就是分子分母都带 $(1-\gamma) S$ 因子的 InfoNCE 形式。

变量说明：
- $p(g, l)$：goal frame 与 text 的联合分布
- $D(g')$：负样本 goal frame 分布
- 常数 +1：来自 Bellman 里的即时 reward -1 经 log-sum-exp 提取后变成 $e^1$

**这个结果的深意**：
1. CLIP 的 InfoNCE 是 LIV 在"无时序"数据上的特例
2. VIP-L 是 InfoNCE 在"有时序结构"数据上的 generalization
3. 任何 text-annotated 视频都可以通过"重复最后一帧"的 trivial augmentation 变成退化分布，从而在 VIP-L 视角下既包含时序信号又包含 InfoNCE 信号

### 2.4 实际实现的简化（Eq.6）

基于上面这个理论，作者指出既然 InfoNCE 已经隐含在 VIP-L 里，实际实现可以更简洁：

$$\mathcal{L}_{\text{LIV}} = \mathcal{L}_{\text{VIP-I}} + \mathcal{L}_{\text{InfoNCE}}$$

注意这里**只用 image goal 跑 VIP**（VIP-I），而 language 那边只用 InfoNCE 对齐 goal frame 和 text。两边通过 InfoNCE 形成的 mutual information 通道自动桥接。这就是论文 Figure 1 的设计。

Similarity metric 的选择也很关键。CLIP 用 cosine similarity，range 是 $[-1, 1]$。但 value function 需要覆盖 $[-\frac{1}{1-\gamma}, 0]$。所以 LIV 把 similarity 重定义为：

$$S(\phi(\cdot), \psi(\cdot)) := \frac{1}{1-\gamma} \text{CosineSim}(\phi(\cdot), \psi(\cdot))$$

这样 S 的 range 是 $[-\frac{1}{1-\gamma}, \frac{1}{1-\gamma}]$，能覆盖 value function 所需范围。同时，代入 Eq.5 后 InfoNCE 项的 $(1-\gamma)$ 因子刚好 cancel 掉，恢复成标准 CLIP InfoNCE，方便直接复用 CLIP 权重初始化。

---

## 3. 算法实现细节（Algorithm 1）

```
Input: 视频数据集 D = {(o_1^i, ..., g^i; l^i)}
       视觉-语言架构 (φ, ψ)

for each iteration:
    1. Sample minibatch of sub-trajectories: 
       {o_t^i, ..., o_k^i, o_{k+1}^i, ..., g^i; l^i}
       t ∈ [1, h_i - 1], t ≤ k < h_i
    
    2. L_VIP-I(φ) = (1-γ)/B Σ_i [-S(φ(o_t^i); φ(g^i))]
                    + log (1/B) Σ_i exp[S(φ(o_k^i); φ(g^i)) 
                                     + 1 - γS(φ(o_{k+1}^i); φ(g^i))]
    
    3. L_InfoNCE(φ,ψ) = (1-γ)/B Σ_i [-log 
                        exp((1-γ)S(φ(g^i);ψ(l^i))) 
                        / (1/B)Σ_j exp((1-γ)S(φ(g^j);ψ(l^i)))]
    
    4. SGD update: (φ,ψ) ← (φ,ψ) - α∇(L_VIP-I + L_InfoNCE)
```

关键实现细节（来自 Appendix B 的 Table 2）：
- **Pre-training**：CLIP 初始化，Adam optimizer，lr=1e-5，weight decay=1e-3，batch=512，γ=0.98，200k steps，8×V100，加上 VIP-L 项（pre-training 阶段）
- **Fine-tuning**：lr=1e-5，batch=64，10k steps，γ=0.98 或 0.96，**不加** VIP-L 项
- Pre-training 数据：EpicKitchen-100，90k video segments，20M frames，20k unique text annotations

注意一个 subtle 的细节：pre-training 实际用了 Eq.4 的完整形式（VIP-I + VIP-L），而 fine-tuning 用的是 Eq.6 的简化形式（VIP-I + InfoNCE）。作者解释这是 preliminary 实验发现——pre-training 早期 VIP-L 帮助语义结构形成，fine-tuning 阶段 InfoNCE 就够了。

---

## 4. Reward specification 机制

LIV 的 implicit value function 可以直接当 reward 用。给定 image 或 language goal $g$，每一步的 potential-based reward：

$$R(o_t, o_{t+1}; g) := S(\phi(o_{t+1}); \phi(g)) - S(\phi(o_t); \phi(g))$$

这是 potential-based reward shaping 的标准形式，理论上不改变 optimal policy。直觉上：如果下一帧离 goal 更近（similarity 更高），就给正 reward；离得更远，给负 reward。

这个 dense reward 可以直接喂给 model-based planner（CEM 或 MPPI），让 LIV 在没有 policy 训练的情况下也能做 trajectory optimization。这是 LIV 区别于纯 representation learning 工作的关键能力。

---

## 5. 实验设计

### 5.1 三个环境（Figure 3 + Appendix C）

| Environment | Train Tasks | Test Tasks | Horizon | Dataset Size | Data Type |
|---|---|---|---|---|---|
| MetaWorld | 1000 | 6 | 20 | 1M transitions | Random policy |
| FrankaKitchen | 5 | 5 | 50 | 12.5K | Machine demos |
| RealRobot | 9 | 9 | 100 | 90k | Human teleoperation |

RealRobot 的细节特别值得注意：
- Franka 机器人，6-DOF end-effector displacement action space
- 15Hz 控制（比 RT-1、BC-Z 等用降低控制频率的方案更挑战）
- 3 fruits × 3 containers = 9 tasks，每 task 100 demos
- 双相机：3rd-person view + wrist view（Figure 7）
- 任务需要 fine-grained spatial grounding（"apple in black pot" 要区分 apple/pear/pineapple 和 black/green pot/tray）

### 5.2 三个评估 axis

1. **Zero-shot reward**：pre-trained LIV 直接给 unseen human/robot 视频打分（Section 5.1）
2. **Pre-trained representation**：freeze representation，做 LCBC（Section 5.3）
3. **Fine-tuning**：用 in-domain 数据 fine-tune 现成 VLM（Section 5.4）
4. **Reward-based planning**：用 LIV reward 驱动 CEM/MPPI planner（Section 5.5）

---

## 6. 关键实验结果深入分析

### 6.1 Zero-shot reward curves（Figure 2）

这是最能 build intuition 的结果。论文把每帧的 negative cosine similarity 到 goal 的距离画成曲线：
- **Figure 2(a) open cabinet (Human)**：image 和 language goal 曲线都单调下降，说明 LIV 学到了"距离 goal 越来越近"
- **Figure 2(b) open microwave (Human)**：同上
- **Figure 2(c) close the fridge (Robot)**：**关键**——这是 unseen robot domain，但 LIV pre-trained 只见过 human video，仍然能给出单调下降曲线
- **Figure 2(d) put the hat on the bottle (Robot)**：曲线中间有个 bump！作者分析这是机器人在中间不必要地把 hat 抬得太高，属于 sub-optimal action。LIV 能 detect 到这种 sub-optimality

对比 CLIP 的同样曲线（Appendix G.1 Figure 12-13）几乎是无规律的 noisy 曲线——CLIP 完全没有 zero-shot reward 能力。这个对比极其强烈地说明：**仅靠 image-text 对齐学到的表示，没有时序 value 结构，无法做 reward**。

### 6.2 Pre-trained representation for LCBC（Figure 4）

| Model | MetaWorld | FrankaKitchen | RealRobot |
|---|---|---|---|
| LIV | **30.6±5.0** | **29.3±4.6** | **~45%** (best) |
| CLIP | 19.4 | 22 | ~10% |
| R3M+DistilBERT | 12.7 | 18.7 | ~5% |
| VIP+DistilBERT | 24.2 | 18.0 | ~15% |

RealRobot 上 LIV 的 gain 最大（绝对提升 ~35%）。作者的解释很合理：real-world 任务没有"shortcut"视觉线索（不像 sim 里物体总在画面中央），需要 language-grounded hand-eye coordination，而 LIV 是唯一真正 jointly 训练 vision-language 的。

Appendix D.1 Table 5 还做了一个有意思的 ablation：把 language task encoding 换成 one-hot。结果 LIV 从 language 编码获益最大（FrankaKitchen 29.3 vs 17.6，提升 11.7 个点），而 CLIP 在 MetaWorld 上用 language 反而比 one-hot 差（19.4 vs 28.6）。原因：MetaWorld 的 annotation 是 atomic instructions 拼接（如"close drawer turn faucet right push black mug right"），CLIP 的 language encoder 没在 control 数据上训练过，无法 disambiguate 这种长 instruction，造成 task aliasing。

### 6.3 Fine-tuning（Figure 5）

相对提升幅度（vs base model）：

| Method | MetaWorld | FrankaKitchen | RealRobot |
|---|---|---|---|
| LIV FT | +75% | +21% | +40%+ |
| CLIP FT | -16% | -36% | N/A |
| VIP-I FT | +54% | -33% | N/A |
| TCN+CLIP FT | mixed (sensitive to α) | +15% | N/A |

关键 findings：
1. **CLIP FT 在 FrankaKitchen 上让性能下降**（22 → 14），原因是 CLIP FT 过度对齐 goal frame 和 text，破坏了中间帧表示
2. **VIP-I FT 在 FrankaKitchen 上也下降**（22 → 14.8），因为数据集小，纯 temporal SSL overfit
3. **TCN+CLIP 对 α 极度敏感**（Table 9）：α=0.5 在 FrankaKitchen 上最好（25.3），但在 MetaWorld 上 diverge；α=0.1 在 MetaWorld 不 diverge 但性能差（14.4）
4. **LIV FT 在所有 setting 都稳定提升**，因为 VIP 和 InfoNCE 在理论上有 1:1 的 natural 配比，无需调权重

### 6.4 Fine-tuning 的 qualitative 分析（Figure 6）

这个图是理解 LIV FT 为什么 work 的核心。把 image goal 距离和 language goal 距离叠在一张图上：
- **Pre-trained LIV**：两条曲线隔得远（domain gap）
- **LIV FT**：两条曲线接近、单调、平滑收敛到相似位置——既有 temporal coherence 又有 semantic alignment
- **CLIP FT**：goal frame 和 text 几乎完美 overlap，但中间帧的 similarity 抖动剧烈，没 temporal coherence

这就是 LIV 的 magic：**VIP 的 recursive Bellman structure 强制了 temporal smoothness**，而 InfoNCE 提供了 cross-modal alignment，二者互相 regularize。

### 6.5 Reward-based planning（Table 1）

| Model | FrankaKitchen | MetaWorld |
|---|---|---|
| LIV (Pre-Trained) | 1.3±0.8 | 29.7±4.7 |
| LIV (LIV FT) | **20.0±4.5** | **55.2±5.5** |
| CLIP | 0±0 | 18.2±4.4 |
| CLIP (LIV FT) | 15.2±4.6 | 45.3±2.5 |
| CLIP (CLIP FT) | 3.2±0.9 | 30.7±3.3 |
| LOREL | 9.6±3.0 | 47.9±3.2 |
| LOREL (R3M init) | 16.8±3.8 | 47.5±12.7 |
| R3M | 8.8±2.7 | 18.3±7.7 |
| R3M (R3M FT) | 16.1±4.2 | 43.9±3.2 |

**最 striking 的对比**：CLIP (LIV FT) > CLIP (CLIP FT) > CLIP。同样 base model，用 LIV objective fine-tune 比用 CLIP 自己的 objective fine-tune 效果好得多。这直接验证了 Section 4.2 的理论——LIV 是 CLIP 在时序数据上的正确 generalization。

### 6.6 R3M 的"作弊"现象（Appendix F.2 Table 12）

这是 paper 里最 expose 别人问题的一个实验。把 language goal 换成 random goal：

| Model | Correct Goal | Random Goal |
|---|---|---|
| LIV-EPIC | 1.3 | 1.0 |
| LIV-EPIC (LIV FT) | 20.0 | 0.0 |
| LOREL | 9.6 | 0.0 |
| LOREL (R3M init) | 16.8 | 0.0 |
| R3M | 8.8 | **12.1** |
| R3M (R3M FT) | 16.1 | 0.0 |

R3M 在 random goal 上居然比 correct goal 上得分还高！这说明 R3M 的 "language reward" 其实是在检测"视觉有没有变化"，根本没真正 grounding language。它的 pre-training objective 是让 time-apart 的 frame 有 higher score，这跟"是否完成任务"无关，只是"画面有没有动"。在 FrankaKitchen 上随便动一下就能 8.8% success 因为 kitchen 任务都涉及物体运动，但在 MetaWorld 上 R3M zero-shot 只有 18.3%——MetaWorld 需要更精细的 language grounding（比如"turn left" vs "turn right"），R3M 的"视觉变化 detector"撑不住。

### 6.7 Long-horizon generalization（Appendix D.2 Table 6）

RealRobot 上把 3 个 atomic task 串起来，要求 policy 处理 unseen scene configuration（前面 task 完成后桌上多了物体）：

| Task Sequence | LIV | R3M |
|---|---|---|
| Pineapple in Tray, Apple in Tray, Pear in Tray | 5/10 | 0/10 |
| Pineapple in Black Pot, Apple in Tray, Pear in Green Pot | 3/10 | 1/10 |
| Pineapple in Tray, Apple in Green Pot, Pear in Black Pot | 5/10 | 1/10 |

LIV 能 zero-shot 处理 long-horizon composite task，部分 trial 完成全部 3 个 task。这是 representation 质量的真正考验。

### 6.8 Few-shot fine-tuning（Appendix E.2 Table 8）

| Demos/Task | CLIP | CLIP (LIV FT) | CLIP (TCN+CLIP FT) |
|---|---|---|---|
| 10 | 7.3 | 22.0 | 13.3 |
| 20 | 12.7 | **30.7** | 23.3 |
| 50 | 22.0 | **33.0** | 25.3 |

10 个 demo 的 LIV FT 性能（22.0）≈ CLIP 50 个 demo 的性能（22.0）。这意味着 LIV FT 在 FrankaKitchen 上相当于把 demo 需求量减少到 1/5。

---

## 7. 设计哲学与对比工作的 deeper intuition

### 7.1 vs CLIP

CLIP 的 objective 是 static 的——每对 (image, text) 独立对齐。但视频是时序的，CLIP 在 fine-tuning 时只用最后一帧（goal frame）+ text，丢掉了前面所有帧。这就解释了 Figure 6(c) 的失败：中间帧的 similarity 完全乱掉。LIV 通过 VIP-I 强制所有中间帧满足 Bellman consistency，保证 temporal coherence。

### 7.2 vs VIP

VIP 是 LIV 的前作，只有 image goal，没 language。要处理 language task，需要外接 DistilBERT 之类，但 language encoder 在 pre-training 阶段不被 VIP objective 优化，只是被动的 feature extractor。LIV 把 language 也纳入同一 value framework，让 language encoder 也学习 task-progress structure。

### 7.3 vs R3M

R3M 用 language alignment loss（TCN-like contrastive）+ 时间距离预测。它的 language encoder 也是 frozen DistilBERT。更关键的是 R3M 的"reward"信号本质是时间距离，而非 task completion（Table 12 暴露这点）。

### 7.4 vs LOREL

LOREL 学一个 explicit classifier $f_\theta(o_0, o_t, l)$ 判断从 $o_0$ 到 $o_t$ 是否完成了 task $l$。这是 discriminative reward model。问题：
1. 需要 (start, end, label) 三元组训练，标注成本高
2. 在小数据集上容易 overfit（Table 1 里 LOREL 在 FrankaKitchen 上只有 9.6，比 R3M init 版本 16.8 差很多）
3. 不能直接做 representation，只是 reward

LIV 把 representation 和 reward 统一在一个 implicit value function 里，避免 explicit classifier 的 overfit。

---

## 8. Limitations 与后续方向

论文没有 explicit limitations section，但从实验和 Appendix G 的 failure cases（Figure 17）能看出：

1. **Zero-shot language reward 在 unseen robot domain 上 noisy**（Figure 2(c)(d) 的 language 曲线比 image 曲线抖）——因为 language grounding gap，pre-training 数据是 human ego-centric video，robot 视角差距大
2. **Failure cases 存在**（Figure 17）：camera viewpoint、embodiment、language command 的 distribution shift 都可能导致 LIV 失效
3. **Pre-training 加了 VIP-L 项**——这其实是 Eq.4 而非 Eq.6 的简化形式，说明理论上的"InfoNCE 就够"在 pre-training 早期不一定成立，需要 explicit VIP-L 帮 semantic 结构 bootstrapping
4. **Architecture 还是 ResNet50 + CLIP transformer**，没尝试更现代的 ViT backbone 或更大规模 language model
5. **没在 manipulation 之外的 task 上验证**（如 navigation、locomotion）

---

## 9. 总结：LIV 真正的创新点

如果让我一句话概括 LIV 的 contribution：**它发现 CLIP 的 InfoNCE 是 goal-conditioned value function learning 在退化数据分布上的特例，从而用同一个 objective 同时解决 vision-language representation learning、reward learning 和 fine-tuning**。

这个 unification 的 power 在于：
- 理论上 elegant（Proposition 1 是干净的理论结果）
- 实践上 stable（VIP 和 InfoNCE 的 1:1 配比是 free 的，无需调 α）
- 工程上 simple（Algorithm 1 只有 6 行 pseudocode）
- 实验上 strong（三个 axis 都 beat baselines，尤其在 real robot 上 gain 显著）

后续工作如 Voltron (Karamcheti et al., 2023)、MVP 等都在不同方向上推进了 vision-language pre-training for control，但 LIV 的"VIP ⊃ InfoNCE"这个理论 connection 仍然是这个 line of work 里最 clean 的 insight。

参考后续工作：
- Voltron: https://arxiv.org/abs/2302.12766
- MVP / Where are we in search for artificial visual cortex: https://arxiv.org/abs/2303.18240
- RT-2 (后续 vision-language-action model): https://robotics-transformer2.github.io/

如果你 (Karpathy) 在思考这条 line 对 VLA 的启示——LIV 的核心 message 是：**control-aware representation 不能从 pure contrastive image-text learning 里 magically emerge，必须有 temporal value structure 参与**。这与 ViT/CLIP-based VLA 模型（如 RT-2, OpenVLA）的某些 failure mode 可能有关联——这些模型在大规模 co-training 时虽然看到了 video，但 objective 里没有显式的 Bellman structure，可能解释了某些 long-horizon planning 上的弱点。这是个值得 deep dive 的方向。
