---
source_pdf: ATOD Annealed Turn-aware On-policy Distillation for.pdf
paper_sha256: 97161b136c9929c3b79a2f7201f5668867de56dcbe620c1dd38845314f1c5337
processed_at: '2026-08-18T01:28:37-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ATOD 用人话讲

## 1. 这论文到底在干嘛

假设你有一个特别聪明的老师傅（大模型 agent），会操作网页、会搜资料、会在虚拟家里干活。你想要把他的本事教给一个刚毕业的小徒弟（小模型），让小徒弟能独立上岗。这就是 **agent distillation**。

听起来简单，做起来发现有两个现成路子都有各自的问题。

---

## 2. 两个现成路子，各有各的坑

### 路子 A：让小徒弟照着老师傅抄（OPD）

**怎么做**：小徒弟自己试着干活，每说一个字，老师傅在旁边点评"这个字你该说成什么"。

**好处**：每一笔都有人在纠错，进度飞快。小徒弟几分钟就能照猫画虎完成基本任务。

**坑**：一旦小徒弟学到 80% 像，就卡住了。老师傅说啥他改啥，老师傅自己也会犯的错他也照抄。更糟的是，越往后学，徒弟和师傅越像，师傅点评的信号越弱（你俩说的都一样了，还有啥好改的？）。

用公式看：
$$
A_t^{\text{OPD}} = \log \pi_T(a_t\mid s_t) - \log \pi_\theta(a_t\mid s_t)
$$
左边是老师傅给当前 token 的概率，右边是小徒弟自己的概率。两个数越接近，这个 advantage 越接近 0，梯度就消失了。这就是论文里反复说的 **teacher ceiling**。

### 路子 B：让小徒弟自己试错，只看最后成败（GRPO）

**怎么做**：小徒弟每干完一个任务，告诉他是成了还是败了。成了给小红花，败了啥也不给。徒弟自己琢磨哪步做对了哪步做错了。

**好处**：没有天花板——只要环境能给 reward，你就能一直突破，甚至超过老师傅。

**坑**：小徒弟刚入门时，100 次干 100 次都失败，group 里 8 次全是 0 分。看公式：

$$
\hat{A}_i = \frac{r_i - \text{mean}(r_j)}{\text{std}(r_j) + \varepsilon}
$$

8 个 reward 全是 0，分子分母都是 0，advantage 就是 0 或噪声。policy gradient 完全没方向。论文 Figure 2a 里 GRPO 曲线前 30 步基本平的，就是这个原因。

而且 GRPO 是 trajectory-level reward——一个完整 trajectory 干了 50 turn，最后一关失败，前 49 turn 即使全对也拿不到信号。这就是 **sparse credit assignment**。

### 两个坑互补

把 Figure 2a 看成两条曲线：
- OPD 曲线：陡升 + 早期 plateau
- GRPO 曲线：缓慢爬升 + 后期继续涨

ATOD 的洞察非常朴素——**取上包络**。早期用 OPD 快速启动，后期切到 GRPO 继续突破。

---

## 3. ATOD 怎么把两个信号揉一起

核心公式就一个：

$$
A_t = \kappa(s)\, A_t^{\text{OPD}} + \rho(s)\, A_t^{\text{GRPO}}
$$

把 OPD advantage 和 GRPO advantage **线性组合**成一个 token-level advantage，然后用标准 PPO clipped surrogate 更新。

- $s$：训练步数
- $\kappa(s)$：OPD 权重，随训练衰减
- $\rho(s)$：RL 权重，随训练增长

直觉版：早期 $\kappa$ 大 $\rho$ 小，徒弟主要抄师傅；晚期 $\kappa$ 小 $\rho$ 大，徒弟主要靠环境 reward 自己摸索。

退火函数简单到不能再简单：

$$
\kappa(s) = \max\Big(0.1,\; 1.0 - 0.9 \cdot \frac{s}{80}\Big)
$$

$$
\rho(s) = 1.0 + 1.0 \cdot \frac{s}{80}
$$

80 步内 $\kappa$ 从 1.0 线性降到 0.1（floor 0.1 防完全脱锚），$\rho$ 从 1.0 涨到 2.0。80 步后固定。

**关键细节**：$\kappa$ 不是降到 0 而是 0.1。这 0.1 的 floor 让徒弟永远保留一丝"师傅的影子"，防止最后为了刷 reward 跑偏去 reward hacking。这个设计跟 [InstructGPT](https://arxiv.org/abs/2203.02155) 加 KL penalty 防 reward hacking 是一个道理，只是 ATOD 用的是带方向的 anchor（push toward teacher）而不是单向的惩罚（push toward base model）。

---

## 4. T-DUR：把"师傅点评"的能量集中到关键 turn

到这里 hybrid advantage 已经能解决时间维度的信号分配。但还有空间维度的问题：**一个 trajectory 有 50 个 turn，哪些 turn 值得师傅点评？**

### 朴素方案的问题

如果给所有 turn 同样权重（uniform），会发现 80% 的 turn 都是 routine 操作——"open fridge"、"go to sinkbasin"，这种 turn 师傅点评纯属浪费 token budget。真正难的是那 20% 的决策点——"我该往左还是往右走"、"该选 size 2x 还是 x-small"。

### 两个信号：disagreement 和 uncertainty

对每个 turn $k$，统计该 turn 内 sampled token 的两个量：

**(1) Disagreement**：师傅和徒弟在这个 turn 上的分歧

$$
d_k = \frac{1}{N_k}\sum_t \big| \log \pi_\theta(a_t^{(k)}\mid s_t^{(k)}) - \log \pi_T(a_t^{(k)}\mid s_t^{(k)}) \Big|
$$

$d_k$ 大 → 师徒在这个 turn 上意见相左。

**(2) Uncertainty**：徒弟自己对这个 turn 没把握

$$
h_k = \frac{1}{N_k}\sum_t \Big(-\log \pi_\theta(a_t^{(k)}\mid s_t^{(k)})\Big)
$$

$h_k$ 大 → 徒弟自己说话都不自信。

注意 $h_k$ 是 token entropy 的**无偏估计**（Appendix A.3 用 martingale-difference 证了）——意思是徒弟这个 turn 的真实熵约等于 $h_k$。这个 trick 让你不算全 vocab softmax 就能估熵，省了巨大计算量。

### Soft-OR 把两个信号揉一起

每个 turn 内先把 $d_k, h_k$ 在 trajectory 内 min-max normalize 到 $[0,1]$，然后做概率论里的 OR 运算：

$$
w_k = 1 - (1 - \tilde{d}_k)(1 - \tilde{h}_k)
$$

这个公式可以这么读："这个 turn 重要，要么因为师徒分歧大，要么因为徒弟自己没把握，或者两者都有"。

为什么不用 $\max(\tilde{d}_k, \tilde{h}_k)$？因为 Soft-OR 更平滑（max 不可导会让训练不稳定），且符合"概率论 OR"的语义。

### Soft-OR 抓住的两种 critical turn

这里要特别强调，Soft-OR 设计抓住的是两类 turn：

**类型 1：高 disagreement + 高 uncertainty**
徒弟没把握 + 师傅不同意 → 显然要重点监督。

**类型 2：低 uncertainty + 高 disagreement**
徒弟很自信 + 师傅不同意 → 这是最危险的 case。徒弟说"我就是要 click x-small"，自信满满，但师傅说"应该 click 2x"。这种 turn 单纯用 entropy weighting 会漏掉（因为 $h_k$ 很小），但 Soft-OR 通过 disagreement 救回来。

论文 Figure 11 的 ALFWorld case study 就是活生生的例子——最后一步 "place lettuce in fridge"，徒弟 $h_6 = 0.008$（极度自信）但 $d_6 = 0.138$（和师傅不一致），Soft-OR 给 $w_6 = 0.60$。如果只看 entropy，这一步会被忽略，但恰恰是这种"自信地犯错"的 turn 最需要监督。

### 为什么 turn-level 而不是 token-level

论文 ablation Figure 4 证明了：token-level reweighting 反而比 uniform 还差。直觉是 token 太碎，单个 token 的 log-prob 是高度 noisy 的（句号 vs 逗号、空格 vs 字符的 disagreement 都不一样），但一个 turn 是一个完整的语义决策单元，把 turn 内 $N_k$ 个 token 的 disagreement 平均一下信号稳定得多。

这其实是 [Wisdom of the crowd](https://en.wikipedia.org/wiki/Wisdom_of_the_crowd) 的统计版——单点噪声大，平均后信号浮出来。

### 为什么 T-DUR 只 reweight OPD 项

公式里 $w_{k(t)}$ 只乘在 $A_t^{\text{OPD}}$ 上，不动 $A_t^{\text{GRPO}}$。

直觉：GRPO 的 advantage 是 trajectory-level（整个 trajectory 8 个采样里的相对排名），根本无法归因到具体 turn。如果硬给 RL advantage 按 turn 重加权，等于瞎猜哪个 turn 贡献大。而 OPD 是 dense 信号（每个 token 都有 $\Delta \log p_t$），按 turn 重加权是把监督 budget 重新分配，不影响信号正确性。

这是个很务实的工程选择——**只在能 reweight 的地方 reweight**。

---

## 5. 实验告诉我们什么

### 5.1 主结果（Table 1）

最让人惊讶的是 ATOD **超过 teacher**：
- 0.6B student：70.62 vs teacher 68.93
- 1.7B student：71.58 vs teacher 68.93
- 4B student：71.06 vs teacher 68.91

这是 annealing 的功劳——前期 OPD 把 student 推到 teacher 水平，后期 GRPO 让 student 突破。如果纯 OPD，student 最多到 teacher 67-68 分就 plateau。

更猛的数据：0.6B on ALFWorld，vanilla 模型 0.78% 成功率，ATOD 干到 82.81%，**100 倍提升**。GRPO 只能到 30.47%。这个数字说明对弱小模型，pure RL 完全没法启动，OPD 提供的 dense supervision 是救命稻草。

### 5.2 Ablation 给出的因果证据

Figure 4 的 ablation 三个结论：

**第一，去掉 annealing 掉点最猛**（0.6B 从 82.8 掉到 75.8）。固定 $\kappa=\rho=1$ 等于全程把两个信号混在一起。早期 OPD 被 RL 的噪声 advantage 稀释（GRPO 早期 advantage 全 0），晚期 RL 被 OPD 的弱信号稀释（OPD 后期 $\Delta \log p \approx 0$）。两边都不讨好。

**第二，token-level reweighting 反而比 uniform 差**。token 粒度太细，信号噪声盖过信号本身。turn 是 agent 决策的自然单元，强行更细反而引入不稳定。

**第三，去掉 T-DUR（uniform turn weight）也掉点**。监督 budget 有限，平均分给所有 turn 等于把 80% budget 浪费在 routine turn 上。

### 5.3 训练动态（Figure 5, 6）

Figure 5c 是个很有趣的图——平均 trajectory 长度。

- GRPO 的 trajectory 越训越长（从 ~40 涨到 ~50）。这是 sparse RL 的典型病——agent 不知道哪步错了，只好反复试错延长 trajectory。
- ATOD 训练几步就收敛到短 trajectory（~20 turn）。因为 OPD 给的 dense 信号让 agent 早期就学会高效路径。

Figure 6 的 OPD vs RL 信号 magnitude 也很说明问题：
- OPD signal 早期大、迅速衰减（徒弟迅速追上师傅）
- RL signal 早期就上来且保持高位（reward 压力一直在）

这印证了 annealing schedule 的合理性——OPD 信号自然衰退时正好 $\kappa$ 也降，两者同步。

---

## 6. 我对这论文的几点吐槽

### 6.1 Teacher 必须是 GRPO-trained 的没强调

论文里所有 teacher 都是 GRPO 后的 checkpoint（Qwen3-4B GRPO / Qwen3-30B-A3B GRPO）。这意味着 OPD 项的天花板不是 base model 的天花板，而是 **GRPO-trained model 的天花板**。

如果 teacher 是个纯 SFT 模型，OPD 项的 ceiling 会低很多，annealing 退火后 RL 要"突破"的 gap 更大。这个细节论文没强调，但实际上是 ATOD 能超过 teacher 的隐藏前提之一——能突破的天花板本身就是已经被 RL 提过的天花板。

### 6.2 Teacher 推理开销没报告

每一步训练都要 teacher 对 student-sampled trajectory 算 log-prob。虽然只算 sampled token 的 log-prob（不算全 vocab softmax），但 teacher forward pass 仍然是训练瓶颈。论文完全没报告 teacher inference time 占训练总时间的比例。

实操上，如果 teacher 是 30B-A3B，student 是 4B，teacher forward 占 70-80% 时间很正常。改进方向是 [speculative decoding](https://arxiv.org/abs/2211.17192) 风格的 teacher cache、或者周期性 teacher update 让 teacher 自己也跟 student 一起 finetune（[Born-again networks](https://arxiv.org/abs/1805.04751) 思路）。

### 6.3 Linear annealing 是不是最优

Linear schedule 简单粗暴。更 sophisticated 的选择：

- **Cosine annealing**（[SGDR](https://arxiv.org/abs/1608.03983) 风格）：开始和结束都平缓，中间陡降。比 linear 更平滑。
- **Adaptive annealing**：监控 teacher-student gap $\bar{d}$，gap 小到阈值自动加速 $\kappa$ 衰减。类似 [ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)。
- **Bandit-based schedule**：把 annealing schedule 选择当 multi-armed bandit（[Population-based training](https://arxiv.org/abs/1711.09846)）。

Linear 对 hyperparameter $T=80$ 很敏感——如果训练时长变成 300 步，$T=80$ 显然不够；50 步又太晚。Adaptive schedule 能自动适应不同任务难度。

### 6.4 Per-trajectory min-max normalization 的 edge case

如果 trajectory 只有 1-2 个 turn（Search-QA 经常出现），min = max，归一化退化。论文用 $\epsilon = 10^{-8}$ 防除零并 fallback 到 0.5。但这个 fallback 等于 uniform，T-DUR 退化成 no reweighting。

改进方案是 cross-trajectory normalization（用整个 batch 的统计量），或者用 z-score + sigmoid 映射到 [0,1]，避免 min-max 的退化。

### 6.5 没和最强 RL baseline 比

[Archer](https://arxiv.org/abs/2402.16727)、[Group-in-group](https://arxiv.org/abs/2505.10978) 这种 turn-level RL 方法都没进 baseline。这些方法在 multi-turn RL 上是直接竞争关系。论文只比 GRPO，没比 SOTA 的 multi-turn RL，留下"ATOD 是不是真的比纯 turn-level RL 强"的疑问。

---

## 7. 一些更远的联想

### 7.1 ATOD 和 DAgger 的关系

[DAgger (Dataset Aggregation)](https://arxiv.org/abs/1011.0686) 通过 student 自身 rollout 收集数据避免 covariate shift，但纯 imitation 受 expert ceiling 限制。ATOD 的 annealing 可以看作"DAgger 早期 + RL 晚期"。这和 [HG-DAgger](https://arxiv.org/abs/1709.09575) 用 human intervention 突破 ceiling 的精神类似，只是把 human 换成 environment reward。

### 7.2 T-DUR 作为"穷人 PRM"

[OpenAI Let's verify step by step](https://arxiv.org/abs/2305.20050) 和 [Math-Shepherd](https://arxiv.org/abs/2312.08935) 训练专门的 Process Reward Model 给每个 reasoning step 打分。T-DUR 只用 student/teacher log-prob 就能识别关键 turn，**不需要训练 PRM**。

这是个很有价值的方向——把 T-DUR 推广到：
- DPO 训练中按 turn reweight preference
- Tool-use agent 中按 turn reweight reward
- Long-CoT reasoning 中按 step reweight

### 7.3 Information-theoretic view

T-DUR 的 $d_k$ 估计 forward KL on sampled tokens，$h_k$ 估计 student entropy。Soft-OR 的组合相当于找"teacher 知道而 student 不知道"或"student 自己都不确定"的 turn——这接近 mutual information $I(\text{teacher}; \text{correct}) - I(\text{student}; \text{correct})$ 的高 utility 区域。这个视角可以参考 [Information-theoretic RL (Still & Precup)](https://arxiv.org/abs/1202.3562)。

### 7.4 和 Reflexion 的结合

[Reflexion (Shinn et al.)](https://arxiv.org/abs/2303.11366) 让 agent 从失败 trajectory 中 verbalize 经验写进 memory。T-DUR 的 $w_k$ 信号天然在识别"失败关键 turn"。两者结合：用 $w_k$ 挑选哪些 turn 写进 Reflexion memory，而不是全 trajectory 都 verbalize。这能让 Reflexion 的 memory 更聚焦。

### 7.5 推广到 code agent

[SWE-bench](https://arxiv.org/abs/2310.06770) 这种 code agent 任务，trajectory 长、reward 极 sparse。ATOD 的 annealing + turn-level reweighting 理论上能直接套用——teacher 是 GPT-4 / Claude 这种强模型，student 是 7B 开源模型。这个 setting 比 ALFWorld 实际得多，值得做 follow-up。

### 7.6 和 Model Merging 的哲学相通

$\kappa A^{\text{OPD}} + \rho A^{\text{GRPO}}$ 在 advantage 空间做线性插值，这和 [Task vectors](https://arxiv.org/abs/2212.04089) / [TIES merging](https://arxiv.org/abs/2311.03006) 在 weight 空间做插值是不同空间但同哲学。一个潜在改进：用 [slerp (spherical interpolation)](https://en.wikipedia.org/wiki/Slerp) 在 advantage 空间做球面插值，避免线性组合的 scale 问题。

---

## 8. 一句话总结

ATOD 的核心 insight 用一句话说：**给小模型 agent 训练设计一个"先抄后超"的 curriculum**——早期抄师傅避开 RL 冷启动的稀疏 reward，后期靠环境 reward 突破师傅天花板，中间用 turn-level 的 disagreement + uncertainty 信号把师傅的注意力集中到关键决策点。

这个 insight 朴素到几乎人人都想到过，但论文把它做对了——hybrid advantage 的简洁公式、annealing schedule 的工程考量、T-DUR 的 Soft-OR 设计、turn-level 而非 token-level 的粒度选择，每一步都是 careful design choice。

它的 beauty 在于用很简单的公式解决了看起来很复杂的 multi-turn agent 训练问题。这种"简单但精准"的设计哲学，和 [GRPO 用 group-relative 替代 value model](https://arxiv.org/abs/2402.03300) 的思路一脉相承——都是用更 cheap 的信号 estimator 解决 expensive 的问题。

---

## 主要参考链接

**OPD & distillation**：
- [On-Policy Distillation (Agarwal et al.)](https://arxiv.org/abs/2306.13649)
- [MiniLLM (Gu et al.)](https://arxiv.org/abs/2306.08543)
- [Distilling Step-by-Step](https://arxiv.org/abs/2305.02340)
- [Born-again networks](https://arxiv.org/abs/1805.04751)

**RL & GRPO**：
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1](https://www.nature.com/articles/s41586-025-09077-2)
- [DAPO](https://arxiv.org/abs/2503.14476)
- [Archer](https://arxiv.org/abs/2402.16727)
- [Group-in-group policy opt](https://arxiv.org/abs/2505.10978)

**Multi-turn agent**：
- [ALFWorld](https://arxiv.org/abs/2010.03768)
- [WebShop](https://arxiv.org/abs/2207.01206)
- [Search-R1](https://arxiv.org/abs/2503.09516)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [Reflexion](https://arxiv.org/abs/2303.11366)

**Process supervision**：
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- [Math-Shepherd](https://arxiv.org/abs/2312.08935)

**Imitation learning**：
- [DAgger](https://arxiv.org/abs/1011.0686)
- [HG-DAgger](https://arxiv.org/abs/1709.09575)

**Annealing & schedule**：
- [SGDR](https://arxiv.org/abs/1608.03983)
- [Population-based training](https://arxiv.org/abs/1711.09846)

**Model merging（哲学联想）**：
- [Task vectors](https://arxiv.org/abs/2212.04089)
- [TIES merging](https://arxiv.org/abs/2311.03006)

**RLHF anchor**：
- [InstructGPT](https://arxiv.org/abs/2203.02155)

想往哪个方向深挖，告诉我继续展开。

---

# ATOD: Annealed Turn-aware On-policy Distillation 论文精读

这篇 Tencent + THU 的工作瞄准一个很实际的问题：**如何把大模型 agent 的多轮交互能力，蒸到小模型上，同时还能让小模型突破 teacher 的天花板**。下面我会按 motivation → method → experiments → critique 的顺序拆解，所有公式都会解释变量含义，并附上我能联想到的相关工作链接。

---

## 1. 问题的本质：OPD 与 RL 在 multi-turn agent 上的互补性

考虑一个 multi-turn agent trajectory：

$$
\tau = (x, a_1, o_1, \dots, a_K, o_K, a_{K+1})
$$

其中 $x$ 是初始 prompt，$a_k$ 是 model 在第 $k$ turn 生成的 response（可能含 reasoning / action / final answer），$o_k$ 是 environment 返回的 observation，$a_{K+1}$ 是 final response。模型只生成 $a$ tokens，$o$ 由 environment 给出。

### 1.1 GRPO 的稀疏信用分配瓶颈

GRPO 是 DeepSeek-R1 / DeepSeekMath 主推的 RL 算法，核心是**用一组 rollout 内部的相对 reward 替代 value model**。对 prompt $x$，采样一组 trajectory $\{\tau_i\}_{i=1}^G$，每个 trajectory 拿到 outcome reward $r_i = R(\tau_i)$，group-relative advantage 为：

$$
\hat{A}_i^{\text{GRPO}} = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\}) + \varepsilon_A}
$$

- $G$：group size（论文里是 8）
- $\varepsilon_A$：数值稳定项

token 级 importance ratio：

$$
\eta_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t} \mid s_{i,t})}{\pi_{\theta_{\text{old}}}(a_{i,t} \mid s_{i,t})}
$$

- $\pi_\theta$：当前 student policy
- $\pi_{\theta_{\text{old}}}$：上一轮 rollout 的 policy（用于 importance sampling）
- $a_{i,t}$：trajectory $i$ 第 $t$ 个 generated token
- $s_{i,t}$：该 token 的 prefix context（含历史 observation）

GRPO 的 clipped surrogate loss：

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}_x \left[ \frac{1}{G}\sum_{i=1}^G \frac{1}{|\mathcal{T}_i|}\sum_{t \in \mathcal{T}_i} \min\Big(\eta_{i,t}\hat{A}_i^{\text{GRPO}},\; \text{clip}(\eta_{i,t}, 1-\varepsilon, 1+\varepsilon)\hat{A}_i^{\text{GRPO}}\Big) \right]
$$

- $\mathcal{T}_i$：trajectory $i$ 中所有 model-generated token 的位置集合（不含 observation tokens）
- $\varepsilon$：clip 范围（典型 PPO 用 0.1~0.2）

**问题**：在 multi-turn agent task 上，reward 通常是 trajectory-level（成功 / 失败），所有 token 共享同一个 $\hat{A}_i^{\text{GRPO}}$。对于 Qwen3-0.6B 这种弱 student，早期 99% trajectory 都失败，group 内 reward 全是 0，advantage 就是 0，policy 完全没梯度信号。这就是 sparse reward 的冷启动失败模式。可以参考 [DeepSeek-R1 Nature paper](https://www.nature.com/articles/s41586-025-09077-2) 和 [DeepSeekMath](https://arxiv.org/abs/2402.03300) 对 GRPO 细节。

### 1.2 OPD 的 teacher-ceiling 问题

On-policy distillation 让 student 在自己 rollout 出来的 trajectory 上，对齐 frozen teacher 的 token distribution。OPD loss（reverse KL 采样估计）：

$$
\mathcal{L}_{\text{OPD}}(\theta) = \mathbb{E}_x \left[ \frac{1}{G}\sum_i \frac{1}{|\mathcal{T}_i|}\sum_t \eta_{i,t}(\theta) \Big( \log \pi_\theta(a_{i,t}\mid s_{i,t}) - \log \pi_T(a_{i,t}\mid s_{i,t}) \Big) \right]
$$

- $\pi_T$：frozen teacher policy

token-level distillation signal：

$$
\Delta \log p_t = \log \pi_T(a_t \mid s_t) - \log \pi_\theta(a_t \mid s_t)
$$

- 当 teacher 给 sampled token 更高概率时 $\Delta \log p_t > 0$，student 被推高这个 token
- 这本质是对 self-sampled token 做 advantage-weighted likelihood ascent，advantage = $A_t^{\text{OPD}} = \Delta \log p_t$

**问题**：一旦 student 接近 teacher，$\Delta \log p_t \to 0$，梯度消失，学习 plateau。更糟的是，teacher 本身可能有 systematic bias（在 ALFWorld 上走某条 suboptimal path），student 通过模仿永远学不到 reward-improving 的 deviation。这个观察和 [Agarwal et al. "On-Policy Distillation of Language Models"](https://arxiv.org/abs/2306.13649) 以及 [MiniLLM (Gu et al.)](https://arxiv.org/abs/2306.08543) 的结论一致。

### 1.3 为什么需要 hybrid + annealing

从 Figure 2a 训练曲线看得很清楚：
- OPD/SOD：早期陡升，~30 step 后 plateau
- GRPO：起步低且爬升慢，但 100+ step 后还在涨
- ATOD：早期跟 OPD 一样快，后期跟 RL 一样继续涨

这就是 annealing schedule 的物理直觉——把两条互补的曲线"取上包络"。

---

## 2. ATOD 方法详解

### 2.1 Hybrid advantage（核心公式 Eq 8）

把 OPD 与 GRPO 信号线性组合进同一个 token-level advantage：

$$
\boxed{A_t = \kappa(s)\, A_t^{\text{OPD}} + \rho(s)\, A_t^{\text{GRPO}}}
$$

- $s$：global training step
- $\kappa(s)$：OPD 系数，随训练衰减
- $\rho(s)$：RL 系数，随训练增长
- $A_t^{\text{GRPO}} = \hat{A}_i^{\text{GRPO}}$（trajectory 级，所有 token 共享）
- $A_t^{\text{OPD}}$ 见下式

$$
A_t^{\text{OPD}} = \Delta \log p_t \cdot w_{k(t)} = \big(\log \pi_T(a_t\mid s_t) - \log \pi_\theta(a_t\mid s_t)\big) \cdot w_{k(t)}
$$

- $k(t)$：把 token 位置 $t$ 映射到它所属的 turn $k$
- $w_{k(t)} \in [0,1]$：该 turn 的 T-DUR 权重，**只 reweight OPD 项**，不动 RL 项

这个设计很关键：RL 信号本来已经够稀疏，再按 turn 重加权反而会丢信号；而 OPD 信号是 dense 的（每个 token 都有 $\Delta \log p_t$），按 turn 重加权是把监督 budget 花在刀刃上。

actor loss 仍是标准 GRPO clipped surrogate，只是 advantage 换成 $A_t$。**没有额外 explicit KL penalty**，因为 $A_t^{\text{OPD}}$ 本身就提供 anchor 作用——只要 $\kappa_{\min} > 0$ 就有弱 teacher anchor 防止严重 drift 或 reward hacking。

### 2.2 Annealing schedule（Eq 10-12）

定义 progress variable：

$$
p(s) = \min\left(\frac{s}{T}, 1\right), \quad T = \text{coef.anneal.steps}
$$

- $T$：退火窗口长度，论文 hyperparam = **80 step**
- $p(s)$ 从 0 线性增到 1，之后 clamp 在 1

OPD 系数线性衰减带 floor：

$$
\kappa(s) = \max\Big(\kappa_{\min},\; \kappa_{\text{init}} - (\kappa_{\text{init}} - \kappa_{\min})\cdot p(s)\Big)
$$

- $\kappa_{\text{init}} = 1.0$，$\kappa_{\min} = 0.1$（hyperparam）

RL 系数线性增长：

$$
\rho(s) = \rho_{\text{init}} + (\rho_{\max} - \rho_{\text{init}})\cdot p(s)
$$

- $\rho_{\text{init}} = 1.0$，$\rho_{\max} = 2.0$（hyperparam）

注意 $\kappa_{\min}=0.1 > 0$ 这个 floor 设计——保留弱 teacher anchor，防止 student 完全脱锚后 reward hacking。这是借鉴了 RLHF 中 KL penalty 的精神（[InstructGPT](https://arxiv.org/abs/2203.02155) 的 KL anchor）。

Lipschitz 性质：$\kappa(s)$ 和 $\rho(s)$ 的 Lipschitz 常数分别是 $|\kappa_{\text{init}} - \kappa_{\min}|/T$ 和 $|\rho_{\max} - \rho_{\text{init}}|/T$，避免 advantage scale 突变。这其实是个工程细节但很重要——RL 训练对 advantage 尺度变化极其敏感，突然的 scale 跳变会触发 PPO clip 进入饱和区。

### 2.3 T-DUR: Turn-level Disagreement-Uncertainty Reweighting

这是论文最核心的创新，把 token-level 的重要性信号**上抬到 turn 粒度**。

#### 2.3.1 两个信号

对每个 turn $k$，统计该 turn 内 $N_k$ 个 generated token 的：

**(1) Disagreement proxy（Eq 13）**

$$
d_k = \frac{1}{N_k}\sum_{t=1}^{N_k} \Big| \log \pi_\theta(a_t^{(k)}\mid s_t^{(k)}) - \log \pi_T(a_t^{(k)}\mid s_t^{(k)}) \Big|
$$

- $a_t^{(k)}$：turn $k$ 内第 $t$ 个 sampled token
- $d_k$ 大 → student 和 teacher 在该 turn 上分歧大

**(2) Uncertainty proxy（Eq 14）**

$$
h_k = \frac{1}{N_k}\sum_{t=1}^{N_k} \Big(-\log \pi_\theta(a_t^{(k)}\mid s_t^{(k)})\Big)
$$

- 这是 sampled token 的 average negative log-prob
- 它是 turn-level entropy $\bar{H}_k = \frac{1}{N_k}\sum_t H(\pi_\theta(\cdot\mid s_t^{(k)}))$ 的**无偏估计**

#### 2.3.2 为什么 $h_k$ 是 entropy 无偏估计（Appendix A.3）

这是一个 martingale-difference argument。设 $X_t = -\log \pi_\theta(a_t \mid s_t)$，$\mathcal{F}_{t-1}$ 是历史 token 生成的 filtration。因为 $a_t \sim \pi_\theta(\cdot \mid s_t)$：

$$
\mathbb{E}[X_t \mid \mathcal{F}_{t-1}] = \sum_a \pi_\theta(a\mid s_t)(-\log \pi_\theta(a\mid s_t)) = H(\pi_\theta(\cdot\mid s_t))
$$

所以 $D_t = X_t - H_t$ 是 martingale difference：$\mathbb{E}[D_t \mid \mathcal{F}_{t-1}] = 0$。对于 $i < j$：

$$
\mathbb{E}[D_i D_j] = \mathbb{E}[D_i \mathbb{E}[D_j \mid \mathcal{F}_{j-1}]] = 0
$$

turn 平均的方差：

$$
\text{Var}\Big(\frac{1}{N_k}\sum_t X_t\Big) = \frac{1}{N_k^2}\sum_t \mathbb{E}[D_t^2] = O\Big(\frac{1}{N_k}\Big)
$$

这个 trick 让 T-DUR **不需要算 full-vocabulary softmax**（避免 teacher 在整个 vocab 上 forward），只用 sampled token 的 log-prob 就行——这是和 [GKD (Generalized Knowledge Distillation)](https://arxiv.org/abs/2306.13649) 类似的工程优化，但推导更严谨。

#### 2.3.3 Per-trajectory normalization（Eq 15）

$$
\tilde{d}_k = \frac{d_k - \min_{j\in\tau} d_j}{\max_{j\in\tau} d_j - \min_{j\in\tau} d_j}, \quad \tilde{h}_k = \frac{h_k - \min_{j\in\tau} h_j}{\max_{j\in\tau} h_j - \min_{j\in\tau} h_j}
$$

- 分母 < $10^{-8}$ 时归一化值设为 0.5（避免除零）

为什么用 min-max 而不是 z-score？因为后续 Soft-OR 需要 $[0,1]$ 输入。在 trajectory 内 normalize 是为了避免不同 task / trajectory length / environment state 之间 scale 混淆——ALFWorld 的 $d_k$ 和 WebShop 的 $d_k$ 量级完全不可比。

#### 2.3.4 Soft-OR fusion（Eq 16）

$$
w_k = 1 - (1 - \tilde{d}_k)(1 - \tilde{h}_k), \quad w_k \in [0,1]
$$

这是概率论中标准的 OR t-conorm（[Frank t-conorm family](https://en.wikipedia.org/wiki/T-norm)）。性质：
- 单调递增：$\tilde{d}_k$ 或 $\tilde{h}_k$ 任一变大 → $w_k$ 变大
- 对称：$f(d,h) = f(h,d)$
- 边界：$f(d,0) = d$，$f(0,h) = h$，$f(1,h) = 1$，$f(d,1) = 1$
- 满足 $f(d,h) \ge \max(d,h)$

**关键洞察**：这个 fusion 能抓住两类高 utility turn：
1. **高 uncertainty + 高 disagreement**：student 不确定且 teacher 不同意 → 显然要重监督
2. **低 uncertainty + 高 disagreement**：student 自信但和 teacher 不一致 → 这是最危险的 turn，单纯 entropy-based weighting 会漏掉。在 ALFWorld case study 的 S6（最终 place lettuce 动作）就是这种 case：$h_6 = 0.008$ 但 $d_6 = 0.138$，Soft-OR 给 $w_6 = 0.60$，单纯用 entropy 会漏掉

#### 2.3.5 Gradient view（Appendix A.2）

在 unclipped region（$\eta_t \approx 1$）：

$$
\nabla_\theta \mathcal{L}_{\text{OPD}} \approx -\mathbb{E}_t\Big[\kappa(s)\, \Delta\log p_t\, w_{k(t)}\, \nabla_\theta \log \pi_\theta(a_t\mid s_t)\Big]
$$

- 这是 REINFORCE 形式的 advantage-weighted likelihood ascent
- effective advantage = $\kappa(s) \cdot \Delta\log p_t \cdot w_{k(t)}$
- 三个乘子分别控制：(时间衰减) × (teacher-student gap) × (turn utility)

可以对比 [Hsieh et al. "Distilling Step-by-Step"](https://arxiv.org/abs/2305.02340) 的 rationale-level distillation——ATOD 的 turn-level reweighting 在某种意义上是 rationale-level 的更细粒度版本。

### 2.4 完整算法流程

Algorithm 1 三个 stage：

| Stage | 操作 |
|---|---|
| I. Rollout & reward advantage | 用 $\pi_{\theta_{\text{old}}}$ 采样 $G=8$ 个 trajectory，算 environment reward，group-relative normalize 得 $\hat{A}^{\text{GRPO}}$ |
| II. Turn-aware reweighting | 把 model-generated tokens 按 turn 分组，算 $d_k, h_k$，min-max normalize，Soft-OR 融合得 $w_k$，构造 $A_t^{\text{OPD}} = \Delta\log p_t \cdot w_{k(t)}$ |
| III. Annealed hybrid update | 按 $\kappa(s), \rho(s)$ 混合 advantage，clipped surrogate 更新 $\pi_\theta$，同步 $\pi_{\theta_{\text{old}}}$ |

---

## 3. 实验结果深度解读

### 3.1 主实验（Table 1）

三个 benchmark：
- **ALFWorld**：[embodied instruction following](https://arxiv.org/abs/2010.03768)，text-based 家务
- **WebShop**：[web navigation + product selection](https://arxiv.org/abs/2207.01206)
- **Search-QA**：[Search-R1 风格](https://arxiv.org/abs/2503.09516)，覆盖 NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle

学生模型：Qwen3-0.6B / 1.7B / 4B
Teacher：0.6B/1.7B 用 Qwen3-4B GRPO 训练后的 checkpoint；4B 用 Qwen3-30B-A3B GRPO（150 step）

**Aggregate 对比**（Avg SR）：

| Student | Vanilla | GRPO | SDAR | OPD | SOD | TCOD | ATOD | Teacher |
|---|---|---|---|---|---|---|---|---|
| 0.6B | 6.66 | 33.17 | 25.44 | 67.73 | 65.74 | 66.88 | **70.62** | 68.93 |
| 1.7B | 15.59 | 40.30 | 41.94 | 64.91 | 65.59 | 66.05 | **71.58** | 68.93 |
| 4B | 19.24 | 68.93 | 64.37 | 68.13 | 68.28 | 66.91 | **71.06** | 68.91 |

注意几个 critical 现象：

1. **ATOD 超过 teacher**（0.6B: 70.62 vs 68.93，1.7B: 71.58 vs 68.93，4B: 71.06 vs 68.91）——证明 annealing 把 RL signal 真正激活了，超越了 imitation ceiling。

2. **0.6B on ALFWorld**：Vanilla 0.78% → ATOD 82.81%，>100× 相对提升。GRPO 只能到 30.47%，说明 pure RL 在冷启动完全失效。

3. **4B 上 GRPO 反而非常强**（68.93 ≈ teacher），因为 4B 本身就够强，sparse reward 也能学会。但 ATOD 仍然再涨 2.13%。

4. **SDAR 表现平平**——它用 self-distillation，没有外部 teacher，对 0.6B 这种弱 student 没用（自身都还没学会）。这反过来说明 external teacher 在小模型场景无可替代。

### 3.2 Ablation（Figure 4）

三个 ablation 在 ALFWorld 三种 student size 上：

| Ablation | 0.6B | 1.7B | 4B |
|---|---|---|---|
| ATOD (full) | 82.8 | 80.5 | 85.2 |
| w/o Annealing | 75.8 | 73.0 | 82.0 |
| Token-level reweight | ~80 | ~76 | ~83 |
| w/o T-DUR (uniform) | ~79 | ~77 | ~84 |

关键发现：
- **w/o Annealing 掉得最猛**——0.6B 掉 7 个点。固定 $\kappa, \rho$ 比例无法同时兼顾早期 bootstrapping 和后期 exploration。
- **Token-level reweighting 反而比 uniform 还差**——token-level 信号太 noisy，逐 token 重加权引入不稳定。这是 turn-level 设计的强证据。
- **w/o T-DUR（uniform turn weight）也有掉点**——说明监督 budget 必须集中。

### 3.3 训练动态（Figure 5, 6）

Figure 5（ALFWorld Qwen3-1.7B）：
- (a) Training reward：ATOD 全程高于 OPD/SOD/GRPO
- (b) Validation SR：ATOD 早期跟 OPD 几乎重合（前 40 step），后期分叉向上
- (c) Avg turns：GRPO 的 trajectory 越训越长（exploration 失控，agent 反复试错），ATOD 很快收敛到短 trajectory（学习更高效的 policy）

Figure 6 三个 diagnostic：
- (a) OPD/RL signal magnitude：OPD 信号早期大、迅速衰减；RL 信号早期就上来了且全程高位
- (b) Mean turn weight：保持在中等范围（~0.5-0.6），没 collapse 也没爆炸——T-DUR 持续 redistribute
- (c) Teacher-student gap：稳步缩小

### 3.4 Case study 深度分析

#### ALFWorld case（Figure 11，task: clean lettuce + put in fridge）

| Step | 类型 | $d_k$ | $h_k$ | $w_k$ |
|---|---|---|---|---|
| S0 | 开放式规划 "check countertops" | 0.101 | **0.037** | **1.00** |
| S2 | 路径决策 "go to sinkbasin" | **0.228** | 0.025 | **1.00** |
| S3 | routine clean | 0.022 | 0.010 | 0.15 |
| S5 | routine open fridge | 0.019 | 0.006 | 0.00 |
| S6 | 最终 place lettuce（低 entropy 高 disagreement） | **0.138** | 0.008 | 0.60 |

**这是 Soft-OR 设计的教科书级证据**：S6 这种"student 自信但和 teacher 不一致"的 turn，单纯 entropy weighting 会跳过它（$h_6=0.008$ 几乎最低），但 Soft-OR 通过 disagreement 救回了 $w_6 = 0.60$。

#### WebShop case（Figure 12，task: 找 men's dress shirt）

| Step | 行为 | $d_k$ | $h_k$ | $w_k$ |
|---|---|---|---|---|
| S2 | **错点 size x-small** | 0.913 | 0.195 | **0.92** |
| S3 | 改回 size 2x | 0.876 | 0.158 | 0.76 |
| S6 | 反复调整 size/color（同时高 d 高 h） | **1.214** | **0.219** | **1.00** |
| S7 | click buy now（commit） | 0.744 | 0.158 | 0.67 |

WebShop 的 disagreement 普遍高于 ALFWorld——因为 free-text search + 多个 clickable attribute，candidate space 大，teacher-student 在每个决策点都容易不一致。这暗示 T-DUR 在高维 action space task 上更有用武之地。

### 3.5 T-DUR 训练前后的诊断（Figure 13）

这个分析很有意思——**训练前** vs **训练后** student 的 turn-level 信号分布：

- **训练前**：trajectory 平均 47.91 turn（接近 50 上限），全是错的。前几 turn $d_k$ 大（teacher 强烈反对），后面 $d_k$ 衰减（因为 error compounding 让 prefix 偏离 teacher reliable support，teacher 也不靠谱了）。$h_k$ 全程 ~0.1 不变。T-DUR 集中权重在前几 turn。
  
- **训练后**：$d_k$ 整体下降（学到了）。$h_k$ 有两 phase：前 ~20 turn 低（successful trajectory，student 自信且正确），>20 turn 升高（这些是出错的 trajectory，student 后期开始迷茫）。T-DUR 把权重 shift 到后期 turn——因为前期已经掌握了，后期才是薄弱点。

**这个动态 reweighting 是 uniform weighting 永远做不到的**——uniform 永远平均分配，而 T-DUR 会跟着 student 的学习状态调整。

---

## 4. Hyperparameter 与实现细节

Table 2 / Table 3 的关键配置：

| 项 | 值 | 评论 |
|---|---|---|
| Actor LR | 1e-6 | 比一般 RLHF 低一个数量级，因为 advantage 已经很 dense（OPD 提供） |
| Group size $G$ | 8 | 比 GRPO 常见的 16-64 小，可能因 OPD 主导时不需要那么多 reward 估计 |
| Max response | 512 token | 不算长，agent task 单 turn 通常足够 |
| GPU | 8 卡 single node | 工程上 friendly |
| Training steps | 150 | 短训练，契合 OPD 的 fast bootstrap |
| Sampling temp | 1.0 | rollout 高温保证 exploration |
| Val temp | 0.4 | eval 偏 greedy |
| Annealing steps $T$ | 80 | 占总训练 53%，合理 |
| $\kappa_{\min}$ | 0.1 | 弱 anchor floor |
| $\rho_{\max}$ | 2.0 | RL 最终 scale 是 OPD floor 的 20 倍 |

任务级差异（Table 3）：
- ALFWorld: env max steps = 50（长 horizon）
- Search-QA: env max steps = 4（短 horizon，但每个 turn 是 free-text reasoning + search call）
- WebShop: env max steps = 15

---

## 5. 联想与批判性思考

### 5.1 与其他 line of work 的关系

**(1) GRPO 系**：[DeepSeekMath](https://arxiv.org/abs/2402.03300) / [DeepSeek-R1](https://arxiv.org/abs/2501.12948) / [DAPO](https://arxiv.org/abs/2503.14476) — ATOD 把 GRPO 作为 RL backbone，但解决了 GRPO 冷启动痛点。可以想象把 GRPO 换成 [PPO](https://arxiv.org/abs/1707.06347) 或 [REINFORCE++](https://arxiv.org/abs/2506.07598) 也行，核心是 hybrid advantage 那个公式。

**(2) OPD 系**：
- [On-Policy Distillation (Agarwal et al.)](https://arxiv.org/abs/2306.13649) — 奠基性工作
- [MiniLLM (Gu et al.)](https://arxiv.org/abs/2306.08543) — reverse KL，student 自己 sample
- [GKD](https://arxiv.org/abs/2306.13649) — sampled-token 蒸馏，T-DUR 借鉴了它"只看 sampled token log-prob"的工程优化
- [Entropy-aware OPD (Jin et al.)](https://arxiv.org/abs/2603.07079) — 单纯用 entropy 加权，ATOD 论文明确指出这会漏掉"低 entropy 高 disagreement"的 turn
- [TIP (Token Importance in OPD)](https://arxiv.org/abs/2604.14084) — token 级 importance，ATOD 把它上抬到 turn 级

**(3) Multi-turn agent RL**：
- [Archer (Zhou et al.)](https://arxiv.org/abs/2402.16727) — hierarchical multi-turn RL
- [WebRL (Qi et al.)](https://arxiv.org/abs/2502.09583) — self-evolving curriculum
- [Group-in-group (Feng et al.)](https://arxiv.org/abs/2505.10978) — turn-level group relative advantage
- [AgenticRL (Dong et al.)](https://arxiv.org/abs/2507.19849) — agent 专用 policy opt

ATOD 与这些工作的核心区别：它**不修改 RL 算法本身**，而是把 turn-level 信号用在 OPD 项的 reweighting 上，RL 项保持原样。这是个非常巧妙的解耦——RL 信号已经够 sparse 了，再 reweight 风险太高。

**(4) Process Reward Model (PRM)**：[OpenAI Let's verify step by step](https://arxiv.org/abs/2305.20050) / [Math-Shepherd](https://arxiv.org/abs/2312.08935) — PRM 给每个 reasoning step 一个 reward，是另一种 dense supervision 方案。T-DUR 的 turn-level weighting 在哲学上接近 PRM，但**不需要训练额外 reward model**，只用 student/teacher log-prob 信号。优点是 zero extra cost，缺点是没有 explicit correctness signal。

**(5) Self-play / self-distillation**：
- [SDAR (Lu et al.)](https://arxiv.org/abs/2605.15155) — self-distillation baseline，ATOD 实验里效果差
- [Self-distilled reasoner](https://arxiv.org/abs/2601.18734) — 类似思想
- [Born-again networks](https://arxiv.org/abs/1805.04751) — 经典 self-distillation
- 这些方法对 weak student 不友好（自身就没东西可学），ATOD 实验数据印证了这点

**(6) Distilling step-by-step / rationale distillation**：
- [Hsieh et al.](https://arxiv.org/abs/2305.02340) — 多任务蒸馏
- [STaR (Zelikman et al.)](https://arxiv.org/abs/2203.14465) — self-taught reasoner with rationalization
- [ReasonFlux](https://arxiv.org/abs/2502.06763) — 思维链库蒸馏

### 5.2 设计选择上的几个 critical point

**(1) 为什么 T-DUR 不 reweight RL 项？**

论文没明说，但直觉是：RL 项是 trajectory-level advantage，已经是所有 token 共享。如果按 turn 重加权 RL advantage，相当于每个 turn 自己定义一个 "turn-level advantage"，这会破坏 GRPO 的 group-relative 语义。更糟的是，trajectory-level reward 无法归因到具体 turn，重加权等于瞎猜。

但这里有改进空间——可以想象结合 [Group-in-group (Feng et al.)](https://arxiv.org/abs/2505.10978) 的 turn-level group relative advantage，给 RL 项也做 turn-level credit assignment。这是 ATOD 的 natural extension。

**(2) 为什么 Soft-OR 而不是 max 或 weighted sum？**

作者在 Appendix A.4 论证：Soft-OR 是 t-conorm，满足单调 + 对称 + 边界 + 至少和单个信号一样大。比 max 更平滑（max 不可导），比简单相加更符合"任一信号触发即 upweight"的语义。但理论上 weighted sum $w_k = \alpha \tilde{d}_k + (1-\alpha)\tilde{h}_k$ 也是一种选择，论文没做这个 ablation。一个潜在的实验是看 different fusers 在不同 benchmark 上的效果。

**(3) Linear annealing 是否最优？**

Linear schedule 简单但粗糙。更 sophisticated 的选项：
- Cosine annealing（[SGDR](https://arxiv.org/abs/1608.03983) 风格）
- Adaptive annealing：根据 teacher-student gap 自动决定何时切换（gap 小到某阈值就开始降 $\kappa$）
- Bandit-based schedule：把 schedule 选择本身当 multi-armed bandit

Linear 在工程上 robust，但 adaptive schedule 是值得探索的方向。参考 [Population-based training (Jaderberg et al.)](https://arxiv.org/abs/1711.09846)。

**(4) Per-trajectory min-max normalization 的潜在问题**

如果 trajectory 极短（只有 1-2 个 turn），min=max，归一化会退化。论文用 $\epsilon = 10^{-8}$ 防除零，并 fallback 到 0.5。但 Search-QA 的 max steps = 4，实际 trajectory 经常 2-3 turn，这个 edge case 会很常见。一个改进是 cross-trajectory normalization，或者用 z-score 加 sigmoid 映射到 [0,1]。

**(5) Teacher 必须是 GRPO-trained 的**

论文里所有 teacher 都是 GRPO 后的 checkpoint（Qwen3-4B GRPO / Qwen3-30B-A3B GRPO）。这意味着 teacher 本身已经 reward-improved，OPD 项的"ceiling"不是 base model ceiling，而是 GRPO ceiling。这是 ATOD 能超越 teacher 的潜在原因之一——如果 teacher 是纯 SFT 模型，OPD 项的 ceiling 会低很多，annealing 退火后 RL 项要"突破"的难度更大。这点论文没强调，但实际上很重要。

**(6) KL 惩罚的缺席**

ATOD 没加 explicit KL penalty（Actor KL loss = Off），完全依赖 $\kappa_{\min} A_t^{\text{OPD}}$ 提供 anchor。这是个有争议的设计——RLHF 文献普遍认为 KL penalty 防止 reward hacking 必不可少。但 ATOD 的论点是：$A_t^{\text{OPD}}$ 是**带方向**的 anchor（不是单纯惩罚 drift，而是把 student 推向 teacher），所以比单向 KL penalty 更 informative。这个 claim 在 0.6B student 上成立（reward 一直在涨没崩），但在更大模型 / 更复杂 reward 上是否还成立需要更多实验。

### 5.3 局限性（论文未讨论）

**(1) Teacher 推理开销**：每一步都需要 teacher 对 student-sampled trajectory 算 log-prob。虽然只算 sampled token log-prob（不全 vocab softmax），但 teacher forward pass 仍然占训练开销大头。论文没报告 teacher inference time。改进方向：[speculative decoding](https://arxiv.org/abs/2211.17192) 风格的 teacher cache、或者周期性 teacher update（[Born-again networks](https://arxiv.org/abs/1805.04751)）。

**(2) 三个 benchmark 都是 text-based**：没测真正 embodied（VLA / robot manipulation）或 code execution（[SWE-bench](https://arxiv.org/abs/2310.06770)）。reward 在那些 setting 下更 sparse 且更长 horizon，ATOD 是否还 work 未知。

**(3) 没和最强 RL baseline 比**：比如 [Archer](https://arxiv.org/abs/2402.16727)、[Group-in-group](https://arxiv.org/abs/2505.10978) 这种 turn-level RL 方法都没进 baseline。这可能是因为这些方法实现复杂，但理论上它们和 ATOD 是直接竞争关系。

**(4) Annealing window $T=80$ 是固定值**：没 ablation 不同的 $T$。如果训练时长变化（比如 300 step 或 50 step），$T$ 怎么定？这个 hyperparameter 敏感性论文没讨论。

### 5.4 一些更深的联想

**(1) Connection to imitation learning theory**

ATOD 本质是 [DAgger (Dataset Aggregation)](https://arxiv.org/abs/1011.0686) 的 RL-enhanced 版本。DAgger 通过 student 自身 rollout 收集数据避免 covariate shift，但纯 imitation 受 expert ceiling 限制。ATOD 的 annealing 可以看作"DAgger 早期 + RL 晚期"——这和 [HG-DAgger (Ross et al.)](https://arxiv.org/abs/1709.09575) 的混合思想类似，但用 RL 而非 human intervention 来突破 ceiling。

**(2) Information-theoretic view**

T-DUR 的 disagreement + uncertainty 融合，可以解释为 **pointwise mutual information estimation**。$d_k$ 估计的是 $\mathbb{E}[\log \frac{\pi_T}{\pi_\theta}]$（forward KL on sampled tokens），$h_k$ 估计的是 student entropy $H(\pi_\theta)$。Soft-OR 的组合相当于找"teacher 知道而 student 不知道"或"student 自己都不确定"的 turn——这是 mutual information $I(\text{teacher}; \text{correct action}) - I(\text{student}; \text{correct action})$ 的高 utility 区域。这个角度可以参考 [Information-theoretic RL (Still & Precup)](https://arxiv.org/abs/1202.3562)。

**(3) Curriculum learning connection**

ATOD 的 annealing 其实是 **task-difficulty curriculum**——早期"简单任务"是"模仿 teacher"，晚期"困难任务"是"超越 teacher"。这和 [Self-paced learning (Kumar et al.)](https://arxiv.org/abs/1004.1468) / [Actor-critic with curriculum](https://arxiv.org/abs/2007.00346) 的精神相通。但 ATOD 的 curriculum 是在 *signal type* 上做的，不是在 *task instance* 上。

**(4) Connection to model merging / weight interpolation**

$\kappa A^{\text{OPD}} + \rho A^{\text{GRPO}}$ 在 advantage 空间做线性插值，这和 [Model merging via task vectors](https://arxiv.org/abs/2212.04089) / [TIES merging](https://arxiv.org/abs/2311.03006) 在 weight 空间做插值是不同空间但同哲学——都是"两个互补信号 convex combination"。改进方向：用 [slerp (spherical interpolation)](https://en.wikipedia.org/wiki/Slerp) 在 advantage 空间做球面插值，避免线性组合的 scale 问题。

**(5) Process supervision 的低成本替代**

T-DUR 不需要训练 PRM，只用 log-prob 信号。这是个很好的"穷人 PRM"思路。可以推广到：
- DPO 训练中按 turn reweight preference
- Tool-use agent 中按 turn reweight reward
- Long-CoT reasoning 中按 step reweight

**(6) 与 [Reflexion (Shinn et al.)](https://arxiv.org/abs/2303.11366) 的关系**

Reflexion 让 agent 从失败 trajectory 中 verbalize 经验。ATOD 的 turn-level weighting 也在识别"失败关键 turn"——但 ATOD 是把这些 turn 标记出来加强 distillation，Reflexion 是把它们 verbalize 进 memory。两者可以结合：用 T-DUR 的 $w_k$ 信号挑选哪些 turn 写进 Reflexion memory。

### 5.5 一个值得做的 follow-up

我能想到的最直接的 follow-up 是**adaptive annealing based on teacher-student gap**：

- 监控 $\bar{d} = \mathbb{E}_\tau[\frac{1}{K}\sum_k d_k]$ 的滑动平均
- 当 $\bar{d}$ 低于某阈值时（student 已经接近 teacher），自动加速 $\kappa$ 衰减
- 类似 [ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) 的思想

这能让 annealing schedule 自适应不同 student size / task difficulty，而不是 fixed $T=80$。

---

## 6. 总结：为什么 ATOD work

回到 first principles，ATOD work 的核心原因是它**正确识别了两种 supervision signal 的互补时序结构**：

1. **OPD 的边际效用递减**：早期 student 远离 teacher，每个 token 都有 $\Delta \log p_t \ne 0$，梯度信号 dense 且 informative。晚期 student 接近 teacher，$\Delta \log p_t \to 0$，OPD 变成无效 anchor。

2. **RL 的边际效用递增**：早期 student rollout 全错，group 内 reward 全 0，advantage = 0，无梯度。晚期 student 开始有部分成功 trajectory，group 内出现 reward variance，advantage 才 informative。

3. **Turn-level 信号匹配 agent 的决策结构**：agent 的"决策点"是 turn，不是 token。token-level reweighting 把语义相邻的 token 信号切碎，uniform weighting 把所有 turn 平均化，turn-level 是 sweet spot。

4. **Soft-OR 抓住两类 critical turn**：高 entropy（探索）+ 高 disagreement（修正），尤其是不被 entropy-only weighting 抓住的"自信但错误"turn。

把 (1)+(2) 用 annealing 调度组合，(3)+(4) 用 T-DUR 实现，就得到了 ATOD。它的 beauty 在于用很简单的公式（一个 hybrid advantage + 一个 Soft-OR fusion）解决了看起来很复杂的 multi-turn agent 训练问题。这种"简单但精准"的设计哲学，和 GRPO 用 group-relative 替代 value model 的思路一脉相承——都是用更 cheap 的信号 estimator 解决 expensive 的问题。

---

## 参考链接汇总

主论文（无 arXiv 链接，文件内未提供）。

**RL & GRPO 系**：
- [DeepSeekMath (GRPO 原始)](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1 Nature paper](https://www.nature.com/articles/s41586-025-09077-2)
- [DAPO](https://arxiv.org/abs/2503.14476)
- [PPO](https://arxiv.org/abs/1707.06347)
- [Archer (hierarchical multi-turn RL)](https://arxiv.org/abs/2402.16727)
- [WebRL](https://arxiv.org/abs/2502.09583)
- [Group-in-group policy opt](https://arxiv.org/abs/2505.10978)
- [AgenticRL](https://arxiv.org/abs/2507.19849)

**OPD & distillation 系**：
- [On-Policy Distillation (Agarwal et al., GKD)](https://arxiv.org/abs/2306.13649)
- [MiniLLM (Gu et al.)](https://arxiv.org/abs/2306.08543)
- [Distilling Step-by-Step (Hsieh et al.)](https://arxiv.org/abs/2305.02340)
- [STaR (Zelikman et al.)](https://arxiv.org/abs/2203.14465)
- [Born-again networks](https://arxiv.org/abs/1805.04751)
- [SDAR (Self-distilled Agentic RL)](https://arxiv.org/abs/2605.15155)

**Multi-turn agent benchmark**：
- [ALFWorld](https://arxiv.org/abs/2010.03768)
- [WebShop](https://arxiv.org/abs/2207.01206)
- [Search-R1](https://arxiv.org/abs/2503.09516)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [Toolformer](https://arxiv.org/abs/2302.04761)
- [Reflexion](https://arxiv.org/abs/2303.11366)

**Process supervision**：
- [Let's Verify Step by Step (OpenAI)](https://arxiv.org/abs/2305.20050)
- [Math-Shepherd](https://arxiv.org/abs/2312.08935)

**Imitation learning theory**：
- [DAgger](https://arxiv.org/abs/1011.0686)
- [HG-DAgger](https://arxiv.org/abs/1709.09575)

**Annealing & curriculum**：
- [SGDR (cosine annealing)](https://arxiv.org/abs/1608.03983)
- [Population-based training](https://arxiv.org/abs/1711.09846)
- [Self-paced learning](https://arxiv.org/abs/1004.1468)

**Model merging（philosophical connection）**：
- [Task vectors](https://arxiv.org/abs/2212.04089)
- [TIES merging](https://arxiv.org/abs/2311.03006)

**RLHF anchor**：
- [InstructGPT](https://arxiv.org/abs/2203.02155)

**Speculative decoding（teacher inference 优化联想）**：
- [Speculative decoding](https://arxiv.org/abs/2211.17192)

如果你想往哪个方向深挖（比如把 T-DUR 推广到 DPO 训练，或者实现一个 adaptive annealing 版本），可以告诉我，我可以继续展开。
