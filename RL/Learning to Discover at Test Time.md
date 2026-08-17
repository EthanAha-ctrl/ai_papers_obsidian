---
source_pdf: Learning to Discover at Test Time.pdf
paper_sha256: 4656ec28d800002d9a3e3f83a71c7d9032ef5f1873cd953b8808a8391ce06ec7
processed_at: '2026-08-05T13:50:25-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 TTT-Discover

Andrej，我换个风格，把这篇 paper 像跟朋友聊天那样讲一遍。

---

## 一句话总结

之前大家用 LLM 做"科学发现"靠 prompt + 反复搜索——你问 model 一万次，挑最好的答案。这篇 paper 说：让 model 在这个具体问题上自己 learn 起来，用 RL 在 test time 更新 weights。结果：用 open model 在一堆硬核 problem 上刷出 SOTA，每个 problem 只花几百美元。

---

## 背景类比

想象你刚学完一本算法书，遇到一个特别难的编程作业。两种做法：

1. **不用脑子的做法**：照着书上例题反复猜，猜一万次，总有一次运气好猜对。这就是 **Best-of-N**。
2. **认真学的做法**：猜几次，发现不对，反思一下，把成功经验内化进脑子，下次猜得更聪明。这就是 **TTT-Discover**。

AlphaEvolve 是第一种做法的 "高级版本"——它有 buffer 存历史，有 mutation/crossover 的 evolutionary search heuristic，但 model 本身是 frozen 的，每次猜完脑子还是那个脑子。TTT-Discover 让脑子真的变。

---

## 核心方法：两件事

### 第一件：objective 要 favor max，不是 average

Standard RL 优化平均奖励 $\mathbb{E}[R]$。但"科学发现"这个游戏里，你只要有一次 sample 拿到 SOTA 就赢，剩下 9999 次多烂都无所谓。所以应该优化 max，不是 mean。

paper 写了个聪明的 objective：
$$
J_\beta(\theta) = \mathbb{E}_s\left[\log \mathbb{E}_a\left[e^{\beta(s) R(s,a)}\right]\right]
$$

人话翻译：里面那个 $\log \mathbb{E}[e^{\beta R}]$ 是个数学 trick。当 $\beta \to \infty$，它逼近 $\max R$；$\beta$ 小的时候它接近 $\mathbb{E}[R]$。所以这是个 "soft max"——你调 $\beta$ 在"取平均"和"取最大"之间滑动。Discovery 想要 max，所以让 $\beta$ 偏大。

但 $\beta$ 太大会爆炸（gradient 被 outlier 主导），太小又没用。paper 用了一个聪明 trick：**让 $\beta$ 自适应**，每一步算完，看 tilted distribution 跟原 policy 的 KL 有多大，如果超过 budget $\gamma = \ln 2$ 就把 $\beta$ 往下调。这样早期 reward 分布 wide、outlier 多时自动用小 $\beta$，晚期 reward 都挤在一起、小改进也值得放大时自动用大 $\beta$。

直觉：这就像 simulated annealing 的 temperature schedule，但 schedule 不是手工设的，而是根据当前 reward distribution 形状自动调的。

### 第二件：reuse 策略要 favor 探索，不是贪心

每次 sample 一个新 solution，从哪个起点出发？三个选项：

1. **从零开始**（`<empty>`）：探索空间大，但走不远。
2. **从 SOTA 出发**：太贪，锁死在 local region。
3. **从 buffer 里挑一个 previous solution**：这是 reuse。

paper 用 PUCT（AlphaZero 那一套）来选。每个 buffer 里的 state 被打分：
$$
\text{score}(s) = Q(s) + c \cdot P(s) \cdot \frac{\sqrt{1+T}}{1+n(s)}
$$

变量说明：
- $Q(s)$：从 $s$ 出发能到的最好结果（注意是 max 不是 mean！）
- $P(s)$：按 reward 排名的 prior，高的优先
- $n(s)$：被访问次数（包括所有 ancestor）
- $T$：总共扩展了多少次

后半项 $\frac{\sqrt{1+T}}{1+n(s)}$ 是 exploration bonus——访问次数越少，bonus 越大；总访问越多，所有 state 的 bonus 都涨。这保证被忽视的好 state 始终有机会被选中，而不是 forever 被几个 early winner 霸占。

跟 AlphaZero 的区别：AlphaZero 用 mean value + learned prior，paper 用 max value + rank-based prior。这反映 "我只关心 best case" 的哲学。

---

## 为什么这两件事一起做才 work

Ablation 给的数据特别直观（TriMul kernel competition）：

| 配置 | runtime |
|---|---|
| Full TTT-Discover | **1203 µs** |
| Expected reward + PUCT（不要 entropic）| 1986 µs |
| Adaptive entropic + 不 reuse | 5274 µs |
| 完全不做 TTT（只 PUCT reuse）| 2061 µs |
| Naive RL（expected + 不 reuse）| 5329 µs |
| Best-of-N | 5352 µs |
| 人类 SOTA | 1371 µs |

读这张表的人话：

- **不做 TTT、只靠 PUCT reuse**：2061 µs，比人类 SOTA 还差，但比 Best-of-N 强多了。说明 search 有用，但光 search 不够。
- **做 TTT 但用 expected reward**：1986 µs，比 no-TTT 好一点。说明学一下有用，但学错了方向（优化 mean）。
- **做 TTT 用 entropic + 不 reuse**：5274 µs，几乎等于不学。说明 reuse 是命脉——没有 reuse 等于每次从零开始，effective horizon = 1，走不远。
- **全套**：1203 µs，比人类 SOTA 快 14%。

所以两个 component 缺一不可。Entropic 决定"往哪学"，PUCT 决定"从哪出发"。一个解决 objective，一个解决 horizon。

---

## 为什么用 open model 就能 beat closed model

AlphaEvolve 用 Gemini 2.0 Pro + Flash；ShinkaEvolve 用 gpt-5/Gemini-2.5 Pro/Claude Sonnet 4/o4-mini 的 ensemble。TTT-Discover 用 gpt-oss-120b（open model）+ LoRA rank 32。

为什么 open model 能赢？因为 **test-time training 让 model 专门 adapt 到这个 problem**。Frozen Gemini 是"什么都懂一点的通才"，TTT-trained gpt-oss 是"专门为这个 kernel 问题优化过的专家"。专家 beats 通才，这跟 fine-tuning beats zero-shot 是同一个道理，只不过 fine-tuning 发生在 test time，目标是 discovery 而不是 generalization。

Cost 对比也很 dramatic：每个 problem 大约 \$500（Tinker API），50 步 × 512 rollouts，平均 prompt 3k tokens + 16k sample tokens。这比 closed frontier model 反复 prompt 的 cost 可能低一两个数量级。

---

## 实验亮点

### Math：Erdős 最小重叠问题

1955 年 Erdős 提出来的 combinatorics 问题。AlphaEvolve 把 upper bound 砍到 0.380924（从 human SOTA 0.380927 微改进）。TTT-Discover 拿到 **0.380876**，改进幅度是 AlphaEvolve 的 16 倍。

更有意思的：AlphaEvolve 找到的是 95-piece symmetric step function，TTT-Discover 找到的是 **600-piece asymmetric** 的。Asymmetric 本身就是个数学发现——之前数学家隐式认为对称最优，AI 说不对，asymmetric 更好。

### GPU Kernel：TriMul

GPUMode 比赛，AlphaFold3 的核心 op。TTT-Discover 在 A100 上 2198 µs vs 人类 SOTA 4531 µs，**2 倍加速**。在 H100/B200/MI300X 上也都 SOTA，虽然训练只用 H100 timing——cross-architecture generalization 是"意外收获"，说明学到的是 algorithmic insight（fuse elementwise ops, delegate matmul to cuBLAS）而不是 hardware-specific trick。

GPUMode organizer 的 review："executed better than current best human solutions."

### AtCoder 算法竞赛

两个 past AHC 比赛，TTT-Discover 拿了"如果在比赛时提交就是 1st place"的成绩。AHC058 上甚至 beat 了 Sakana AI 的 ALE-Agent（那是第一个 AI 在 AHC 拿 1st 的工作，用 Gemini-3 Pro + gpt-5.2-high），而 TTT 用的是 open model。

### Biology：单细胞 RNA-seq denoising

OpenProblems benchmark，TTT-Discover 从 MAGIC 代码出发，加上 gene-adaptive transform ensembling + low-rank SVD + log-space polishing，在 PBMC 和 Tabula 上都 beat prior SOTA。

但 MIT 专家 Eric Sun 提醒：benchmark metric 改进不一定 transfer 到 biological insight。这是个 caveat——TTT-Discover 学的是"在 benchmark 上拿高分"，不一定是"生物学上更好的 denoiser"。

---

## 这套方法的 limitation

paper §6 说自己只能做 **continuous reward**。如果 reward 是 binary（对/错）或者 sparse（很少给信号），entropic objective 的 exponential tilting 没东西可 tilt，整个方法失效。这是 next direction。

具体实验上，**second autocorrelation inequality 没破 SOTA**（0.959 vs AlphaEvolve V2 的 0.961）。可能因为 AlphaEvolve V2 用了 50000-piece step function，TTT-Discover 受 construction size 限制。这也说明方法不是万能的——在需要"暴力堆 size"的 problem 上，可能 search budget 比 learning 更重要。

---

## 跟其他工作的关系

paper 把自己放在 TTT (Test-Time Training) 这个 framework 里，引用 Yu Sun 2020 [https://arxiv.org/abs/2007.02591] 和 2023 [https://arxiv.org/abs/2310.13807]。TTT 之前三种形式：

1. **Nearest-neighbor TTT**：找 training set 里的邻居，fine-tune。增加 effective capacity。
2. **Novel-instance TTT**：test instance OOD，用 self-supervision 生成 aux task。改善 generalization。
3. **TTT-Discover**（这篇）：single hard problem 上做 RL，目标是 discovery。

三个 concurrent work 做类似的事：EvoTune [https://arxiv.org/abs/2504.05108]、MiGrATe [https://arxiv.org/abs/2508.08641]、ThetaEvolve [https://arxiv.org/abs/2511.23473]。ThetaEvolve 最像，但 TTT-Discover 在公平对比（都用 Qwen3-8B）下 beat 它，因为 entropic objective 和 PUCT 更针对 discovery 的 max-objective 设计。

---

## 我的几个 take

1. **"Policy 是 means 不是 end" 这句话点醒我**。Standard RL 把 policy 当成 deploy 的目标，所以 average reward 重要。Discovery 把 policy 当工具，用完就扔，只要它能 sample 出一个 outlier。这个 framing 让 entropic objective 变得很自然——你不是在 train 一个好 policy，是在用 policy 当 "sample generator" 来找 max。

2. **Adaptive $\beta$ 是真聪明**。很多 paper 用固定 temperature schedule，这篇让 schedule 由数据自己决定。KL budget $\gamma = \ln 2$ 这个值也很有讲究——1 nat 的 KL 对应"从 uniform 到 determinism"的信息量，是个合理的"一步走多远"的 budget。

3. **Asymmetric construction 是真正的发现**。AI 不只是数值上更好，而是结构上不同。这暗示 AI 发现了人类 mathematician 没想到的 solution structure。这种"结构创新"比"数值优化"更有科学意义。

4. **Cost \$500/problem 意味着可以 scale**。如果每个 problem \$5 万，只能做几个 showcase；\$500 可以做几百个。这把"AI 做科学发现"从 demo 推到 viable workflow。

5. **Cross-arch generalization 暗示 model 学到了 algorithmic insight**。TriMul kernel 在 H100 训练，A100 上 2× 加速。这跟"model 学到了 hardware-specific trick"的 hypothesis 矛盾，说明它学到了"fuse elementwise ops + delegate matmul to BLAS"这种 architecture-agnostic 的策略。这跟人类 expert kernel engineer 的 intuition 一致。

6. **Open model + LoRA 是 economics 上的关键**。如果只能用 closed frontier model，这方法没法 scale。gpt-oss-120b 是 open 的，LoRA rank 32 让 update cost 很低，Tinker API 提供 test-time training infra。整个 stack 是 reproducible 的，跟 AlphaEvolve 用 closed Gemini 形成对比。

---

## 一句话再总结

把 AlphaEvolve 的 "frozen LLM + evolutionary search" 升级成 "learning LLM + PUCT reuse + entropic objective"，让 model 在单个 hard problem 上 test-time train，用 open model 在 math/kernel/algorithm/biology 四个 domain 几乎全刷 SOTA，每个 problem 几百美元搞定。核心 insight：discovery 是 max game 不是 mean game，objective 和 search 都要为 max 设计。

参考：[paper arXiv (待发布)](https://arxiv.org) | [AlphaEvolve](https://arxiv.org/abs/2506.13131) | [ThetaEvolve](https://arxiv.org/abs/2511.23473) | [TTRL](https://arxiv.org/abs/2504.16084) | [Yu Sun TTT](https://arxiv.org/abs/2310.13807) | [gpt-oss-120b](https://arxiv.org/abs/2508.10925) | [Tinker](https://www.tinker.systems/) | [GPUMode](https://github.com/gpumode) | [AtCoder AHC](https://atcoder.jp) | [OpenProblems](https://openproblems.bio/) | [PUCT/AlphaZero](https://www.nature.com/articles/nature24270) | [GRPO/DeepSeek-R1](https://arxiv.org/abs/2501.12948) | [REPS](https://aaai.org/ocs/index.php/AAAI/AAAI10/paper/view/1709)

---

# TTT-Discover: Test-Time Training 来做科学发现

Andrej，这篇 paper 的核心 idea 其实很 elegant——之前 test-time scaling 的主流做法（AlphaEvolve 那一支）是用 frozen LLM 反复 prompt、靠 evolutionary search 在 solution space 里搜，本质上 model 一直是同一个；而这篇工作让 model 在 test time 自己学起来，用 RL 在 single test problem 上更新 weights，从而把它从 "一个聪明的 student 反复猜答案" 变成 "一个 student 在这个具体 problem 上反复 learn from mistakes"。

下面我按 motivation → method → experiments → ablation → 实现细节 → 相关工作的脉络，把直觉和数学都讲透。

参考链接先列出来：

- AlphaEvolve (DeepMind, 2025): https://arxiv.org/abs/2506.13131
- AlphaEvolve V2 (Georgiev, Tao et al.): https://arxiv.org/abs/2511.02864
- ThetaEvolve (concurrent): https://arxiv.org/abs/2511.23473
- MiGrATe (concurrent): https://arxiv.org/abs/2508.08641
- EvoTune (concurrent): https://arxiv.org/abs/2504.05108
- TTRL (RL on test set): https://arxiv.org/abs/2504.16084
- One-Example RL: https://arxiv.org/abs/2504.20571
- Yu Sun TTT 原始论文: https://arxiv.org/abs/2310.13807
- TTT for generalization under distribution shifts: https://arxiv.org/abs/2007.02591
- AlphaZero / PUCT (Silver et al.): https://www.nature.com/articles/nature24270
- GRPO (DeepSeek-R1): https://arxiv.org/abs/2501.12948 (DeepSeek-R1)
- gpt-oss-120b model card: https://arxiv.org/abs/2508.10925
- Tinker (Thinking Machines Lab): https://www.tinker.systems/
- GPUMode / KernelBot: https://github.com/gpumode
- AtCoder Heuristic Contest: https://atcoder.jp
- OpenProblems single-cell benchmark: https://openproblems.bio/
- Risk-sensitive RL with entropic objective (Jiang et al.): https://arxiv.org/abs/2509.24261
- Relative Entropy Policy Search (Peters et al.): https://aaai.org/ocs/index.php/AAAI/AAAI10/paper/view/1709

---

## 1. Motivation：discovery problem 跟标准 RL 的三个本质区别

paper 在 §3.1 把这件事讲得非常清楚。一般你拿 RL 算法（PPO、GRPO）直接套，会失败，原因是 discovery problem 和 standard RL 有三点根本不同：

**(1) Objective 是 max，不是 average。** Standard RL 优化 expected reward $\mathbb{E}_{a \sim \pi}[R(s,a)]$，它关心 average 表现。但 discovery 只需要一个 outlier：只要某一次 sample 出来的 state $s$ 满足 $R(s) > r_{\text{sota}}$，你就赢了。一个 expected reward 极低的 policy，只要它能 sample 出一个 max reward，就比 expected reward 高的 policy 强。

paper 给了个 kernel engineering 的例子：SOTA 是 2000µs，做到 1900µs 已经很难，但 expected-reward objective 把它们当成差不多大的 reward，差异几乎被抹平。

**(2) Effective horizon 太短。** 每次 rollout 从头开始，policy 能 "走" 的距离有限。Discovery 经常需要从一个 partial solution 出发，再 refine 好几步——这其实就是 multi-timestep trajectory。State reuse 等于隐式地把 trajectory 拉长。

**(3) Exploration 容易 collapse。** 在 policy 层面，优化 expected reward 会 collapse 到 "safe high-reward action"，避开 risk。在 reuse 层面，naive prioritization 会反复 over-exploit 几个 promising state。

这三点决定了：你不能直接拿 PPO/GRPO 套，要专门设计 objective 和 reuse 规则。这就是 TTT-Discover 的两个组件：**Entropic objective** + **PUCT reuse**。

---

## 2. 整体框架：discovery problem 作为 MDP

§2.1 把 scientific problem 形式化成 MDP。一张表（Table 1）就讲完了：

| Problem | State $s$ | Action $a$ | Transition | Reward $R(s)$ |
|---|---|---|---|---|
| Erdős Min. Overlap | step function certificate | thinking tokens + code | $s' = \text{Python}(\text{Parse}(a))$ | $1/\text{Upper bound}$ |
| Autocorr. Inequalities | step function | thinking tokens + code | $s' = \text{Python}(\text{Parse}(a))$ | $1/\text{Upper bound}$ or lower bound |
| Kernel Engineering | kernel code | thinking + Triton code | $s' = \text{Parse}(a)$ | $1/\text{Runtime}$ |
| Algorithm Competition | algorithm code (C++) | thinking + C++ | $s' = \text{Parse}(a)$ | test score |
| Single Cell Analysis | analysis code | thinking + code | $s' = \text{Parse}(a)$ | $1/\text{MSE}$ |

几个关键 notation：

- $s_{\text{sota}}$ = 目前 best-known solution（比如 leaderboard 顶上的 kernel）
- $r_{\text{sota}} = R(s_{\text{sota}})$
- **Discovery 定义**：找到一个 state $s$ 使得 $R(s) > r_{\text{sota}}$，差距越大越 significant。

Action 总是 "thinking tokens + 一段 code"，环境把 code parse 出来，可能还 run 一下（数学问题要 execute Python 来构造 step function），然后给出 continuous reward。Reward 是连续的，0 表示 invalid（fail validity check 或 timeout）。

---

## 3. Search baselines：Best-of-N → State Reuse → State-Action Reuse

§2.2 把 prior work 的 search 方法都列出来，逐层加东西：

**Best-of-N：**
$$
s = s_{\text{sota}} \text{ or } \langle \text{empty} \rangle, \quad a_i \sim \pi_\theta(\cdot | d, s), \quad i = 1, \ldots, N
$$
就是 i.i.d. sample N 个 rollout，取最好的。这里用 $i$ 不用 $t$ 是强调 rollouts 独立、没有时间顺序。

Initial state $s$ 怎么选？设成 $s_{\text{sota}}$ 太偏 exploitation，会让 policy 不敢往完全不同的方向探索；设成 $\langle \text{empty} \rangle$ 又怕 explore 了一个 promising direction 但没 exploit 充分。State reuse 就是为了解决后者。

**State Reuse：**
$$
s_i \sim \text{reuse}(\mathcal{H}_i), \quad a_i \sim \pi_\theta(\cdot | d, s_i), \quad \mathcal{H}_{i+1} = \mathcal{H}_i \cup \{(s'_i, r_i)\}
$$
$\mathcal{H}_i$ 是 buffer，存之前的 (state, reward)。reuse 是一个 search heuristic，偏 high-reward state 但也给 low-reward state 一定概率。当你 reuse 一个 previous solution $s'_i$，相当于给它 trajectory 加了一步——effective horizon 拉长。

**State-Action Reuse（AlphaEvolve 风格）：**
$$
s_i, c_i \sim \text{reuse}(\mathcal{H}_i), \quad a_i \sim \pi_\theta(\cdot | d, s_i, c_i), \quad \mathcal{H}_{i+1} = \mathcal{H}_i \cup \{(s_i, a_i, s'_i, r_i)\}
$$
不仅 reuse state，还 reuse action 里的 thinking tokens、intermediate code，把这些信息转成自然语言 context $c_i$ 再喂给 LLM。这就是 AlphaEvolve 那一套 evolutionary search：手工设计的 mutation、cross-over、fitness、diversity 测量。

到这一步为止，policy $\pi_\theta$ 的 weights $\theta$ 还是 frozen 的，经验只能改善下一个 prompt，不能改善 policy 自己。TTT-Discover 的下一步就是让 $\theta$ 也 update。

---

## 4. Method 的核心：Entropic Objective + PUCT

### 4.1 Naive RL baseline 和它的失败模式

最直接的 baseline 是标准 RL：
$$
\theta_{i+1} = \theta_i + \eta \nabla_\theta \mathbb{E}_{a \sim \pi_{\theta_i}(\cdot|s)}[R(s,a)], \quad \text{reuse}(\mathcal{H}_i) = \delta_{\langle \text{empty} \rangle}
$$
直接拿 PPO/GRPO 在 single problem 的 environment 里跑。Paper 在 §3.1 解释为什么这会 fail（前面 motivation 那三条）。

Ablation Table 8 给了数据：naive RL（expected reward + no reuse）在 TriMul H100 上拿到 5328.73µs，跟 Best-of-N 的 5352.36µs 几乎一样烂，远不如 TTT-Discover 的 1203.10µs。差距 4 倍以上。

### 4.2 Entropic Objective：从 average 切到 max

这是 paper 的第一个核心设计。目标函数：
$$
J_\beta(\theta) = \mathbb{E}_{s \sim \text{reuse}(\mathcal{H})}\left[\log \mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\left[e^{\beta(s) R(s,a)}\right]\right]
$$

直觉：内层 $\log \mathbb{E}[e^{\beta R}]$ 就是 cumulant generating function / log-moment-generating-function，它是 $\max$ 的 smooth approximation。当 $\beta \to \infty$，$\log \mathbb{E}[e^{\beta R}] / \beta \to \max_a R(s,a)$。所以这个 objective 直接优化 "best reachable reward"，正是 discovery 要的。

它跟 expected reward 的区别在于：expected reward 是 linearity in $R$，对所有 reward 线性加权；而 entropic objective 是 exponential in $R$，high-reward action 被指数级放大。一个 reward 1900µs 和 2000µs 的差，在 expected reward 里是线性 100µs 的差，在 entropic 里（$\beta$ 大时）就是 $e^{\beta \cdot 100\Delta}$ 倍的差。

**梯度推导：**
$$
\nabla_\theta J_\beta(\theta) = \mathbb{E}_{s \sim \text{reuse}(\mathcal{H})}\left[w_{\beta(s)}(a) \nabla_\theta \log \pi_\theta(a|s)\right]
$$
其中
$$
w_{\beta(s)}(a) = \frac{e^{\beta(s) R(s,a)}}{\mathbb{E}_{\pi_\theta(\cdot|s)}[e^{\beta(s) R(s,a)}]}
$$

含义：$w_{\beta(s)}(a)$ 是一个 reweighting factor，把 high-reward action 的 policy gradient 放大 $\propto e^{\beta R}$ 倍。它满足 $\mathbb{E}[w_{\beta(s)}] = 1$，所以 $w_{\beta(s)}(a) - 1$ 是一个 mean-baselined advantage。

**Advantage with KL penalty：**
$$
A(a; s) = w_{\beta(s)}(a) - 1 - \lambda \log \frac{\pi_\theta(a|s)}{\pi_{\theta_0}(a|s)}
$$

变量含义：
- $w_{\beta(s)}(a)$：entropic weight，做 exponential tilting
- $-1$：baseline（因为 $\mathbb{E}[w]=1$）
- $\lambda \log \frac{\pi_\theta(a|s)}{\pi_{\theta_0}(a|s)}$：KL penalty，防止 $\pi_\theta$ 跑离 initial policy $\pi_{\theta_0}$ 太远。$\pi_{\theta_0}$ 是 test-time training 开始前的 frozen model，相当于 reference policy。$\lambda$ 在大部分实验中是 0.1，algorithm engineering 是 0.01。

这种 KL-regularized policy gradient 的形式来自 REPS (Relative Entropy Policy Search, Peters 2010) 和最近 KL-regularized PG for LLM reasoning 的一系列工作 [https://arxiv.org/abs/2505.17508]。

### 4.3 Adaptive $\beta$：通过 KL budget 自动调温度

paper §A.1 讲了一个关键 trick：固定 $\beta$ 很难调。早期训练 $\beta$ 太大会不稳定，后期 $\beta$ 太小会让 advantage 消失（因为小的 improvement 在 large $\beta$ 下 advantage 已经 vanishing 了）。Concurrent work [29] 直接用 $\beta = 2$，但 paper 发现这个值跨 task 很难通用。

他们的做法：定义一个 tilted distribution
$$
q_\beta(\tau|s) = \frac{\pi_\theta(\tau|s) \exp(\beta r(\tau; s))}{\mathbb{E}_{\pi_\theta}[\exp(\beta r(\tau; s))]}
$$
$q_\beta$ 就是 policy 被 reward exponentially tilt 之后的"目标分布"。$w_\beta(\tau|s) = q_\beta(\tau|s) / \pi_\theta(\tau|s)$ 就是 density ratio，控制 update 的有效 step size。

$\beta(s)$ 的选择规则：约束 KL divergence of tilted distribution 到原 policy 等于一个 budget $\gamma$：
$$
\text{KL}(q_{\beta(s)}(\cdot|s) \| \pi_\theta(\cdot|s)) = \gamma
$$
paper 全实验固定 $\gamma = \ln 2$。$\ln 2$ 是 1 nat 的 KL，对应 binary distribution 上"完全确定 vs 完全均匀"的差。

直觉：当 reward 改进空间大（早期、low-value state），$q_\beta$ 会被几个 outlier 迅速 dominate，KL 容易超 budget，于是 $\beta(s)$ 自动变小，防止 update 被几个 outlier 抓走。当 reward 改进空间小（晚期、near-goal state），$q_\beta$ 对给定 $\beta$ 不那么 peaked，KL 还在 budget 内，于是 $\beta(s)$ 可以变大，把微弱 advantage 放大。

**Batch estimator：** 给定 N 个 rollouts from 同一 $s$，rewards $\{r_n\}_{n=1}^N$。Empirical distribution $u(n) = 1/N$。Tilted batch distribution：
$$
q_\beta(n) = \frac{e^{\beta r_n}}{\sum_{m=1}^N e^{\beta r_m}}
$$
通过 bisection search 解
$$
\text{KL}(q_\beta \| u) = \sum_{n=1}^N q_\beta(n) \log(N q_\beta(n)) = \gamma
$$
求出 $\hat\beta(s)$。

然后用 leave-one-out (LOO) 算 advantage，减去 $r_{\max} = \max_n r_n$ 提高 numerical stability：
$$
\hat Z_{-n} = \frac{1}{N-1} \sum_{m \neq n} \exp(\hat\beta(s) (r_m - r_{\max})), \quad A_n = \frac{\exp(\hat\beta(s) (r_n - r_{\max}))}{\hat Z_{-n} + \varepsilon} - 1
$$

减 $r_{\max}$ 是经典 trick：$\exp(\beta r_n) / \sum \exp(\beta r_m) = \exp(\beta(r_n - r_{\max})) / \sum \exp(\beta(r_m - r_{\max}))$，分子分母同除 $\exp(\beta r_{\max})$，避免大 $\beta$ 大 $r$ 时的 overflow。

**Invariant 性质：** advantage 对 reward 的 affine 变换不变——$r'(\tau) = w \cdot r(\tau) + b$ ($w > 0, b \in \mathbb{R}$) 得到完全相同的 advantage。这很重要，因为不同 problem 的 reward scale 千差万别（kernel 是 microseconds，math 是 inequality bound，biology 是 MSE），不用为每个 problem 单独 scale reward。

### 4.4 PUCT Reuse：从 AlphaZero 借来的 tree search

paper §A.2 详细写了 PUCT 规则。每个 archive $\mathcal{H}_t$ 里的 state $s$ 被打分：
$$
\text{score}(s) = Q(s) + c \cdot \text{scale} \cdot P(s) \cdot \frac{\sqrt{1+T}}{1+n(s)}
$$

变量含义：
- $Q(s)$：从 $s$ 出发的 best one-step reachable reward。如果 $s$ 还没被 expand 过，$Q(s) = R(s)$；否则 $Q(s) = m(s) = \max$ child reward。注意 paper 用的是 max 不是 mean——和 AlphaZero 的 mean 完全相反，paper 强调 "we care about the best outcome starting from a state, not the average"。这呼应 discovery 的 max-objective 哲学。
- $P(s)$：rank-based prior。$\text{rank}(s) \in \{0, \ldots, |\mathcal{H}_t|-1\}$ 按 reward 降序排（rank 0 最好）。
$$
P(s) = \frac{|\mathcal{H}_t| - \text{rank}(s)}{\sum_{s' \in \mathcal{H}_t} (|\mathcal{H}_t| - \text{rank}(s'))}
$$
reward 越高的 state 拿到越大的 prior，但仍是线性递减，不是 winner-take-all。
- $n(s)$：visitation count，s 和它的 descendants 被扩展的总次数。
- $T$：目前为止 expanded parents 的总数。
- $c$：exploration coefficient，固定 1.0。
- $\text{scale} = R_{\max} - R_{\min}$：reward range，让 exploration term 跟 reward scale 对齐。

$\frac{\sqrt{1+T}}{1+n(s)}$ 是经典 UCT exploration bonus：随着总扩展数 $T$ 增加，所有 state 的 exploration bonus 都增加；但被访问多次的 state ($n(s)$ 大)，它的 bonus 衰减。这保证 under-visited state 始终保持 candidate 资格，防止 over-exploit 几个 early winner。

**更新规则：** 扩展 parent $p$ 后，观察到 best child reward $y = \max_{s' \in \text{Child}(p)} R(s')$，更新：
- $m(p) \leftarrow \max(m(p), y)$
- $n(a) \leftarrow n(a) + 1$ for all $a \in \{p\} \cup \text{Anc}(p)$（ancestors 也加 1）
- $T \leftarrow T + 1$

Visitation 反向传播到所有 ancestors——扩展任何 descendant 都会减少 ancestor 的 exploration bonus，避免某条 lineage 被反复深挖。

**Archive 管理：** 每个 expanded parent 保留 top-2 children（按 R 排），然后全局保留 top-1000 states by R，同时始终保留 initial seed states。

**跟 AlphaZero PUCT 的四个区别** (§A.2 最后一段)：
1. AlphaZero $Q(s,a)$ 是 mean，paper 是 max（optimistic expansion）
2. AlphaZero $P(s,a)$ 是 learned action prior，paper 是 rank-based prior over archived states
3. Visitation 反向传播到 all ancestors，paper 是"整条 lineage 减 bonus"
4. AlphaZero 用 virtual loss 防止并行扩展同一 branch，paper 直接 block 整条 lineage（ancestors + descendants）不进当前 batch，鼓励 diversity

直觉：PUCT 是 AlphaZero 的"乐观探索"思想嫁接到 discovery——不要 mean value，要 max value；不要 learned prior，要 rank-based prior；exploration bonus 让 under-visited 但 high-reward 的 state 始终是 candidate。

### 4.5 Algorithm 1 串起来

```
Input: problem description d, policy π_{θ_0}
R, T = get_env(d)  # d 诱导 reward 和 transition
H_0 = {(<empty>, R(<empty>), {})}
for i = 0, ..., N-1:
    s_i, c_i ~ reuse(H_i)            # PUCT 选 initial state
    a_i ~ π_{θ_i}(· | d, s_i, c_i)   # 采样 action
    s'_i = T(a_i)                    # transition
    r_i = R(s'_i)                    # reward
    H_{i+1} = H_i ∪ {(s_i, a_i, s'_i, r_i)}
    θ_{i+1} = train(θ_i, (d, s_i, c_i, a_i, r_i))  # entropic objective + KL
return s_{i*}, i* = argmax r_i
```

每步 512 rollouts，50 步，总 sampling budget = 25600。这就是为什么 baseline 叫 Best-of-25600。

---

## 5. 实验结果：四个 domain 几乎全 SOTA

### 5.1 Mathematics

**Erdős Minimum Overlap Problem** (§4.1.1)：

问题：partition $\{1, 2, \ldots, 2n\}$ 成两个 size $n$ 的 set $A, B$，$M_k$ 是 $a_i - b_j = k$ 的解的个数，$M(n) = \min_{A,B} \max_k M_k$，bound $c = \lim_{n\to\infty} M(n)/n$。

Prior bound：$0.379005 < c < 0.380927$ (Haugland 2016 [https://arxiv.org/abs/1609.08000], White 2023 [https://arxiv.org/abs/2305.09028])。AlphaEvolve 把 upper bound 砍到 0.380924。

TTT-Discover 拿到 **0.380876**，比 AlphaEvolve 改进 $0.380924 - 0.380876 = 48 \times 10^{-6}$，而 AlphaEvolve 比 prior human SOTA 改进 $0.380927 - 0.380924 = 3 \times 10^{-6}$，所以 TTT-Discover 的改进是 AlphaEvolve 的 16 倍。

Construction 细节很有意思：AlphaEvolve 是 95-piece symmetric step function，TTT-Discover 发现了一个 **600-piece asymmetric** step function（Figure 2）。对称性被打破这件事本身可能就有数学意义——之前数学家可能直觉上认为对称最优，但 TTT 发现 asymmetry 更好。

算法层面：discovered code 用 FFT-accelerated gradient descent + random hill climbing + simulated annealing，靠 projection 维持可行性 $f(x) \in [0,1], \int f = 1$。

**First Autocorrelation Inequality** (§4.1.2)：

$C_1 = \max$ constant such that $\max_{|t| \leq 1/2} (f * f)(t) \geq C_1 (\int f)^2$ for all nonnegative $f$ supported on $[-1/4, 1/4]$。任何 valid $f$ certify $C_1 \leq \|f * f\|_\infty / \|f\|_1^2$。

Best human: $C_1 \leq 1.50973$ (Matolcsi & Vinuesa 2010 [https://www.sciencedirect.com/science/article/pii/S0022247X09007414])。AlphaEvolve: 1.50530。AlphaEvolve V2: 1.50317。ThetaEvolve: 1.50314 (refine AlphaEvolve construction)。

TTT-Discover: **$C_1 \leq 1.50287$**，用一个 30000-piece step function（AlphaEvolve 和 ThetaEvolve 是 1319-piece）。Figure 3 可视化对比，TTT-Discover 是从 scratch 找到的新 construction，不是 refine 别人的。

Algorithm evolution：早期用 Adam + softmax parameterization 跑到 1.510，然后用 LP（linear programming）按 [46] 的 insight 砍到 1.504，最后关键 insight 是 "只在 near-tight constraints 上 optimize"——只取 convolution 最大的 top-K 位置放进 LP，gradient 也从所有 near-maximum 位置算（而不是只 max 位置）。

**Second Autocorrelation Inequality：** $C_2 = \sup_{f \geq 0} \|f*f\|_2^2 / (\|f*f\|_1 \|f*f\|_\infty)$。Best human: 0.8892。AlphaEvolve: 0.8962。AlphaEvolve V2: 0.9610。TTT-Discover: 0.9591，没有 discovery（没 beat SOTA）。这是 paper 唯一一个没拿下的 math task。

**Circle Packing** (§4.1.3, Table 3)：n=26, n=32。TTT-Discover match 了 best known constructions，没 improve，用 SLSQP + simple geometric initialization，比 ShinkaEvolve [https://arxiv.org/abs/2509.19349] 的 simulated annealing initialization 简单很多。

**和 ThetaEvolve 公平对比** (Table 2 下半部分)：TTT-Discover with Qwen3-8B 在 AC1 拿 1.50525，ThetaEvolve with R1-Qwen3-8B 拿 1.50681。TTT 用 worse model + smaller budget 还 beat ThetaEvolve。

### 5.2 GPU Kernel Engineering (§4.2)

GPUMode competition [https://github.com/gpumode]，两个 task：TriMul (AlphaFold3 的核心 op) 和 DeepSeek MLA Decode。

**TriMul 结果** (Table 4)：

| GPU | 1st Human | TTT-Discover | Speedup |
|---|---|---|---|
| A100 | 4531.5 µs | 2198.2 µs | 2.06× |
| H100 | 1371.1 µs | 1161.2 µs | 1.18× |
| B200 | 1038.9 µs | 914.2 µs | 1.14× |
| MI300X | 2515.8 µs | 1555.7 µs | 1.62× |

注意 A100 上拿到 2× 加速，但训练时 reward function 只用 H100 timing——A100 上的 generalization 是 "意外收获"，说明 discovered kernel 学到了 architecture-agnostic 的优化策略。

Best-of-25600 在 A100 上是 9219.7µs，比 1st human 慢一倍——证明 frozen model 反复 sample 完全不够，必须 train。

**Discovered kernel 策略** (§4.2, §C.2)：identify 到主要 bottleneck 是 memory I/O（一系列 elementwise op），于是 kernel fuse 了：
1. Input LayerNorm
2. Sigmoid + elementwise multiplication for input gating
3. Output LayerNorm + output gating

对 compute-heavy $O(N^3)$ matmul，转 FP16 然后 delegate 给 cuBLAS/rocBLAS 充分利用 TensorCores/MatrixCores。这跟 1st human kernel 策略类似但 fuse 得更彻底——human 没 fuse output LayerNorm 和 gating，TTT-Discover fuse 了。

paper §C.2 给了完整的 TriMul H100 Triton kernel 代码（~470 行），非常值得读。三个 kernel：
- `_row_ln_fp16_kernel`：row-wise LayerNorm, FP32 reduce, FP16 输出
- `_proj_gate_mask_kernel`：fused projection + gating + mask
- `_ln_gate_out_linear_fused_kernel`：hidden-dim LayerNorm → out-gate → final linear 全 fuse

**MLA Decode 结果** (Table 5)：在 AMD MI300X 上三个 instance 测试，TTT-Discover 跟 1st human 在统计意义上没显著差异（top human 1653.8 vs TTT 1669.1）。Discovered kernel 主要靠 `torch.compile()`，没用 Triton 做 fine-grained optimization，paper 承认这限制了进一步改进。Table 10 给了 filter Triton kernel 的版本，runtime 反而更慢（1740.6µs）。

**Expert review** (§4.2.1)：GPUMode organizer Matej Sirovatka, Alex Zhang, Mark Saroufim 评价 "executed better than current best human solutions"。

### 5.3 Algorithm Engineering (§4.3)

AtCoder Heuristic Contest (AHC) [https://atcoder.jp]，两个 past contest：

**AHC039 "Purse Seine Fishing"** (Table 6)：

| Method | Score |
|---|---|
| 1st human | 566,997 |
| ShinkaEvolve (ensemble of frontier models) | 558,026 |
| ALE-Agent (Gemini-2.5 Pro) | 550,647 |
| Best-of-25600 (gpt-oss-120b) | 554,171 |
| TTT-Discover (gpt-oss-120b) | **567,062** |

TTT-Discover 从 ALE-Agent 的 5th place solution 出发（跟 ShinkaEvolve 一样从 5th 开始），refine 到 1st place，beat ShinkaEvolve 的 2nd place。ShinkaEvolve 用 ensemble of gpt-5, Gemini-2.5 Pro/Flash, Claude Sonnet 4, o4-mini，TTT 只用 gpt-oss-120b 单 model。

Discovered algorithm: prefix-sum scoring 生成大量 axis-aligned rectangles candidate → greedy seed 一个 connected union → simulated annealing with add/remove/replace/expand/shrink/slide moves → cleanup + final greedy refinement。

**AHC058 "Apple Incremental Game"** (Table 6)：

| Method | Score |
|---|---|
| 1st human | 847,674,723 |
| ALE-Agent (Gemini-3 Pro Preview + gpt-5.2-high) | 848,373,282 |
| Best-of-25600 | 772,429,752 |
| TTT-Discover | **848,414,228** |

TTT-Discover 从 scratch 出发，beat 所有 human submission 和 ALE-Agent。ALE-Agent 是 Sakana AI [https://sakana.ai/ahc058/] 第一个 AI 拿 1st 的 AHC。

Discovered algorithm: 多个 greedy 规则 + 不同 bias + 短 beam search 生成 initial plan → simulated annealing 做 random edits/swaps/partial rebuilds → local cleanup。用简单公式 estimate upgrade 的 future production 价值，guide greedy + pruning。Caching intermediate state 增量 recompute。

### 5.4 Biology: Single-Cell Denoising (§4.4)

OpenProblems benchmark [https://openproblems.bio]，denoising task。三个 dataset：PBMC, Pancreas, Tabula Muris Senis Lung。训练用 Pancreas，evaluation 在 PBMC 和 Tabula 上测 generalization。

Metric: MSE in log-normalized space + Poisson negative log-likelihood。Reward = MSE score（Poisson 必须 < 0.97，否则 reject）。

**Results** (Table 7)：

| Method | PBMC Score | Tabula Score |
|---|---|---|
| MAGIC (A, R) [prior SOTA] | 0.64 | 0.64 |
| ALRA | 0.50 | 0.47 |
| OpenEvolve | 0.70 | 0.71 |
| Best-of-25600 | 0.62 | 0.65 |
| TTT-Discover | **0.71** | **0.73** |

TTT-Discover 从 MAGIC 代码出发，加上：
1. Gene-adaptive transform ensembling（多个 VST: anscombe, freeman-tukey, sqrt）
2. Low-rank SVD refinement
3. Log-space polishing step 直接 optimize benchmark metric

完整代码在 §E，~720 行 Python。MSE 从 0.19 降到 0.15 (PBMC) / 0.18→0.14 (Tabula)。

**Expert review** (§4.4.1)：MIT 的 Prof. Eric Sun 评价 "simple, aligns with underlying smoothing-based approach of MAGIC, yields empirical improvements on key metrics"，但提醒 benchmark metric improvement 不必然 transfer 到 biological insight。

---

## 6. Ablation：每个组件都重要 (§4.5, Table 8, Figure 4)

TriMul H100 task 上做完整 ablation：

| Config | Best Runtime (µs) |
|---|---|
| Full TTT-Discover (adaptive entropic + PUCT) | **1203.10** |
| Constant $\beta = 2$ entropic + PUCT | 1483.83 |
| Expected reward (no entropic) + PUCT | 1985.67 |
| No TTT + PUCT | 2060.70 |
| Adaptive entropic + ε-greedy (ε=0.1) | 1328.89 |
| Adaptive entropic + no reuse | 5274.03 |
| Naive RL (expected + no reuse) | 5328.73 |
| Best-of-N | 5352.36 |
| Best Human | 1371.1 |

**关键 takeaway**：

1. **Entropic vs Expected reward**：1203 vs 1986，差 ~65%。Entropic objective 的"偏向 max"对 discovery 至关重要。
2. **Adaptive vs Constant $\beta$**：1203 vs 1484。Constant $\beta=2$ 在 late training 时 advantage 消失，improvement 减缓。Adaptive 让 $\beta$ 在 small-improvement regime 自动放大。
3. **PUCT vs ε-greedy**：1203 vs 1329。ε-greedy "work reasonably well, especially with an early lucky kernel"。PUCT 的 tree-based exploration 更稳定。
4. **PUCT vs No reuse**：1203 vs 5274。No reuse 等于把 effective horizon 砍到 1，policy 没法站在 previous solution 肩膀上。
5. **No TTT (only PUCT reuse)**：2060。Pure search 不够，需要 train model。
6. **Naive RL ≈ Best-of-N**：5328 vs 5352，几乎一样。Standard RL 在 single problem 上没什么改进——expected reward objective 让它 collapse 到 safe action。

Figure 4 是 reward distribution 随 step 变化的 violin plot。Full TTT-Discover 的 distribution 整体右移且 max 不断推高；constant $\beta$ 后期停滞；expected reward 整体慢；no TTT 分布不变；no reuse 几乎不动。

---

## 7. Implementation Details (§3.3, Table 9, Appendix A)

- **Model**: gpt-oss-120b [https://arxiv.org/abs/2508.10925] (open model)
- **Training**: LoRA rank 32, Adam (lr=4e-5, β1=0.9, β2=0.95, ε=1e-8)
- **Steps**: 50 training steps
- **Batch**: 512 rollouts/step，分 8 groups × 64 rollouts。每组用同一 context + initial state（保证 entropic estimator 的 batch 是 from same $s$）
- **Context**: 32k tokens，prompt+thinking 限制 26k tokens，剩 6k 给 final response
- **Reasoning effort**: high
- **KL coef $\lambda$**: 0.01 (algorithm engineering) 或 0.1 (其他)
- **$\beta$**: adaptive，KL budget $\gamma = \ln 2$
- **PUCT $c$**: 1.0
- **Sampler/learner mismatch correction**: importance sampling ratio correction (因为 RL infra 里 sampler 和 learner 不同步) [https://fengyao.notion.site/offpolicy-rl]
- **No off-policy step**: 一个 gradient step on 整个 batch

**Cost**: 平均 prompt 3000 tokens，sample 16000 tokens，50 steps × 512 rollouts ≈ \$500/problem on Tinker [https://www.tinker.systems/]。这是非常低的 cost，远低于 closed frontier model 训练。

**Compute budget 对比**：
- TTT-Discover: 50 × 512 = 25600 rollouts (gpt-oss-120b)
- Best-of-25600: 同 budget
- OpenEvolve: 同 25600 budget (但 prompt 增长导致很多 rollout 被 truncate)
- AlphaEvolve / ThetaEvolve: 用 Gemini 2.0 Pro + Flash / R1-Qwen3-8B，budget 不同

**Reward shaping**: minimization problem (upper bound) → reward = 1/bound；maximization problem (lower bound) → reward = bound。Invalid (fail check / timeout) → reward = 0。

**Time limits**: math action 10 min execute；kernel 没超时（remote eval）；algorithm 2 sec + 1GB；denoising 400 sec + 3GB。

---

## 8. Related Work 和 TTT 的历史脉络 (§5)

paper §5 把 TTT 放在 continual learning 的历史里讲，这部分很值得读。

### 8.1 Continual Learning vs TTT

Continual learning（§5.1）传统关注 distribution drift over time——chatbot 每小时用新数据 fine-tune，但所有 user 共享同一 model。Test-time training (§5.2) 强调两个特殊点：

1. **Personalization**: 每个人有自己的 brain，learning 在 individual life context 里
2. **No train-test boundary**: commute 既是 testing (今天要上班) 又是 training (积累未来经验)

Yu Sun et al. 2020 [https://arxiv.org/abs/2007.02591] 提出 TTT for generalization under distribution shift；2023 [https://arxiv.org/abs/2310.13807] 提出 TTT formal framework。

### 8.2 TTT 的三种形式

**TTT on Nearest Neighbors** (§5.2.1)：locally weighted regression (1970s, Cleveland), local learning (Bottou & Vapnik 1990s), KNN-SVM (Zhang et al. 2006)。给 test instance，找 training set 中的 nearest neighbors，在 neighbors 上 fine-tune，再 predict。这能增加 effective capacity——linear model 也能 fit nonlinear ground truth。

最近 Hardt & Sun 2023 [https://arxiv.org/abs/2305.18466] 把这 idea 搬到 LLM；Hübotter et al. 2024 [https://arxiv.org/abs/2410.08020] 改进 neighbor selection；Hübotter et al. 2025 [https://arxiv.org/abs/2510.04786] 把 TTT-on-neighbors 用到 RL reasoning tasks。

**TTT for Novel Instances** (§5.2.2)：test instance OOD，model 能力不是瓶颈，缺的是 data。Test instance 本身 unlabeled，用 self-supervision 生成 aux task（BERT, MAE 风格）做训练。Gandelsman et al. 2022 [https://arxiv.org/abs/2209.14217] MAE-TTT。

最近：
- **AlphaProof** (DeepMind, IMO silver) [https://www.nature.com/articles/s41586-025-04071-21]：test 时 prompt LLM 生成 targeted curriculum of easier problems，然后 RL
- **Akyurek et al.** 2024 [https://arxiv.org/abs/2411.07279]：TTT for few-shot reasoning，augment few-shot demos，然后 supervised learning，在 ARC-AGI 上有效
- **Seek in the Dark** [https://arxiv.org/abs/2505.13308]：policy gradient at test time，policy 自己当 judge，optimize token representation

### 8.3 三个 concurrent work

paper §5.2 末尾提到三个最相关的 concurrent work：

1. **EvoTune** (Surina et al.) [https://arxiv.org/abs/2504.05108]：per-instance RL + replay，PPO/GRPO/DPO 风格 update
2. **MiGrATe** (Phan et al.) [https://arxiv.org/abs/2508.08641]：Mixed-policy GRPO for adaptation at test time
3. **ThetaEvolve** (Wang et al.) [https://arxiv.org/abs/2511.23473]：最相关，test-time learning on open problems，用 R1-Qwen3-8B

paper 的差异点：tailor 了 learning objective (entropic) 和 reuse rule (PUCT) 到 discovery goal，而不是用 standard RL/evolutionary baseline。在 fair comparison (Table 2) 中，TTT-Discover with Qwen3-8B 比 ThetaEvolve with R1-Qwen3-8B 在 AC1 上更好 (1.50525 vs 1.50681)，尽管用了 worse model 和 smaller budget。

### 8.4 两个 tangential formulation

**RL on One Example** (§5.3) [https://arxiv.org/abs/2504.20571]：train on one MATH training example，show 泛化到同 dataset 其他 problem。区别：他们 generalize，TTT-Discover 不 generalize，只 solve this specific problem。

**TTRL** (§5.4) [https://arxiv.org/abs/2504.16084]：train on entire test set with majority voting 做 pseudo-label。区别：TTT-Discover train on 单个 test problem with continuous verifiable reward，找 one exceptional solution，不是 improve average。

---

## 9. 我对这篇 paper 的几点直觉观察

**为什么 entropic objective 比 expected reward 强这么多？** 本质上是因为 discovery 是 **extreme value** 问题，不是 central tendency 问题。$E[R]$ 是 first moment，$\log E[e^{\beta R}]$ 的 $\beta \to \infty$ limit 是 essential supremum——这就是 max。当你用 entropic objective，你 implicitly 在做 "soft maximum"，把所有 sampling budget 都花在 push max 上，而不是 push mean。这在 ablation 里 1203 vs 1986 的 60%+ 差距里体现得淋漓尽致。

**为什么 adaptive $\beta$ 关键？** 因为 reward landscape 在 training 不同阶段 shape 不一样。早期 reward 分布 wide，几个 outlier 主导，固定 $\beta$ 大就爆炸，$\beta$ 小又没 signal。Adaptive $\beta$ 通过 KL budget 自动调节 "tilt 程度"——early 时 tilt 弱（小 $\beta$），晚期 tilt 强（大 $\beta$）。这跟 simulated annealing 的 temperature schedule 是同一种 wisdom。

**为什么 PUCT 比 ε-greedy 强？** ε-greedy 是 memoryless 的——它随机 explore，不在乎 explore 哪里。PUCT 通过 visitation count 反向传播到 ancestors，让整条 lineage 的 bonus 衰减，相当于 "这附近我看过了，去看看别处"。这是 tree search 的 wisdom 嫁接到 solution archive。在 ablation 里差距 1203 vs 1329 不算巨大，但 paper 说 "in early experiments with other applications, lack of exploration is bigger problem"——可能在更难的 task 上 PUCT 优势更明显。

**为什么 no reuse 这么惨（5274 vs 2060 no-TTT）？** 因为 single rollout 的 effective horizon 太短。Discovery problem 经常需要 "partial solution → refine → refine again"，这本质上是 multi-step trajectory。Reuse 等于把 trajectory 拼起来——你 reuse 一个之前 5 step 走到的 state，再走 1 step，effective 是 6 step。所以 reuse 不是 "偷懒复用"，是 implicit 的 trajectory stitching。

**discovery 和 standard RL 的本质区别，paper 总结成一句话**：policy 是 means，不是 end。Standard RL 的 end 是 policy；discovery 的 end 是一个 state，policy 只是个工具，用完就扔。这解释了为什么 average reward 没意义——你 deploy 一次拿到 max，剩下的都不重要。

**Cost issue：\$500/problem 非常便宜。** 对比 AlphaEvolve 用 Gemini 2.0 Pro + Flash 的成本，TTT-Discover 用 open model + LoRA + Tinker API，可能 1-2 个数量级便宜。这暗示 test-time training 的 economics 已经 viable，可以 scale 到很多 problem。

**Generalization 在 kernel engineering 上的意外表现**：训练只用 H100 timing，A100/B200/MI300X 上都 improve。这暗示 model 学到的不是 "针对 H100 micro-arch 的 trick"，而是 architecture-agnostic 的 algorithmic insight（fuse elementwise ops, FP16 matmul delegate）。这种 cross-arch generalization 在人类 kernel engineer 看来也很 nontrivial。

**Math 发现的 asymmetry**：Erdős problem 上 TTT-Discover 找到 asymmetric 600-piece construction，beat AlphaEvolve 的 symmetric 95-piece。这本身有数学价值——它告诉我们对称最优这个 assumption 可能是错的。AI 发现的 not just 更好的数值，更是 "结构上不同" 的解。

**Limitation**：Second autocorrelation inequality 没破 SOTA（0.959 vs 0.961）。可能是因为 AlphaEvolve V2 用了 50000-piece step function，TTT-Discover 限制了 construction size。paper §6 也指出 current form 只能做 continuous reward，sparse/binary reward 和 non-verifiable domain 是 future work。

---

## 10. 公式总览 (cheat sheet)

为了 build intuition，把所有公式集中一遍：

**Entropic objective** (核心)：
$$
J_\beta(\theta) = \mathbb{E}_s [\log \mathbb{E}_a [e^{\beta(s) R(s,a)}]]
$$
- $s$：initial state，from reuse($\mathcal{H}$)
- $a$：action，from $\pi_\theta(\cdot|s)$
- $\beta(s)$：inverse temperature，adaptive via KL budget
- $R(s,a)$：reward
- $\log E[e^{\beta R}]$：cumulant generating function，$\beta \to \infty$ 时 $\to \max$

**Entropic gradient**：
$$
\nabla_\theta J_\beta = \mathbb{E}_s [w_{\beta(s)}(a) \nabla_\theta \log \pi_\theta(a|s)]
$$
- $w_{\beta(s)}(a) = e^{\beta R(s,a)} / E[e^{\beta R}]$：exponential tilting weight，$\mathbb{E}[w]=1$

**Advantage with KL penalty**：
$$
A(a;s) = w_{\beta(s)}(a) - 1 - \lambda \log \frac{\pi_\theta(a|s)}{\pi_{\theta_0}(a|s)}
$$
- $-1$：baseline ($\mathbb{E}[w]=1$)
- $\lambda$：KL coef (0.01 或 0.1)
- $\pi_{\theta_0}$：reference policy (test-time training 前的 frozen model)

**Adaptive $\beta$ via KL budget**：
$$
\text{KL}(q_{\beta(s)}(\cdot|s) \| \pi_\theta(\cdot|s)) = \gamma
$$
- $q_\beta(\tau|s) = \pi_\theta(\tau|s) e^{\beta r} / \mathbb{E}_{\pi_\theta}[e^{\beta r}]$：tilted distribution
- $\gamma = \ln 2$：fixed KL budget
- 解 via bisection search

**LOO advantage estimator**：
$$
\hat Z_{-n} = \frac{1}{N-1} \sum_{m \neq n} \exp(\hat\beta(s)(r_m - r_{\max})), \quad A_n = \frac{\exp(\hat\beta(s)(r_n - r_{\max}))}{\hat Z_{-n} + \varepsilon} - 1
$$
- $r_{\max} = \max_n r_n$：max reward for numerical stability
- $N$：batch size (per group, 64)
- invariant to reward 的 affine 变换

**PUCT score**：
$$
\text{score}(s) = Q(s) + c \cdot \text{scale} \cdot P(s) \cdot \frac{\sqrt{1+T}}{1+n(s)}
$$
- $Q(s) = m(s)$ if $n(s) > 0$ else $R(s)$：max child reward
- $P(s) = (|\mathcal{H}_t| - \text{rank}(s)) / \sum(\ldots)$：rank-based prior
- $n(s)$：visitation count (含 ancestors)
- $T$：total expansions
- $c = 1.0$：exploration coef
- $\text{scale} = R_{\max} - R_{\min}$

**TTT-Discover update**：
$$
\theta_{i+1} = \theta_i + \eta \nabla_\theta J_{\beta(s_i)}(\theta_i), \quad s_i \sim \text{PUCT}(\mathcal{H}_i)
$$

---

## 11. 总结

TTT-Discover 的贡献可以浓缩成一句话：**让 test-time training 从 "在 neighbors 上 fine-tune" 或 "self-supervised aux task" 升级成 "在 single hard problem 上做 RL，用 entropic objective + PUCT 专门优化 extreme value 而非 average"**。

它跟 AlphaEvolve 系列的根本区别在于 model 是 frozen 还是 learning。Frozen model 只能靠 prompt engineering，model 自己不能 "internalize" discovery 的 insight；learning model 可以。这呼应 Rich Sutton 的 "Bitter Lesson" [http://www.incompleteideas.net/IncIdeas/BitterLesson.html]——search 和 learning 都 scale with compute，但历史上 learning 经常 supersede search（Go、protein folding）。

实验上几乎全 SOTA，唯一没拿下的是 second autocorrelation inequality。Cost \$500/problem 非常 viable，意味着这套方法可以 scale 到很多 scientific problem。Open model + LoRA + Tinker API 让它 reproducible，跟 AlphaEvolve 用 closed Gemini 形成鲜明对比。

Limitation 也明显：只做 continuous reward，sparse/binary reward 是 future work；MLA Decode 上没显著 beat human；discovered algorithm 的 biological validity 在 single-cell 那个 task 上未验证（benchmark metric improvement 不等于 biological insight）。

整体看，这篇 paper 把 TTT 这个领域从 "适配 test distribution" 推到 "在 test time 做科学发现"，是 TTT framework 的一个重要扩展。concurrent work 有三个（EvoTune, MiGrATe, ThetaEvolve），说明这个方向是 hot 的，但 TTT-Discover 在 entropic objective + PUCT 的设计上更针对 discovery 的 max-objective 本质，所以在 fair comparison 中 beat ThetaEvolve。
