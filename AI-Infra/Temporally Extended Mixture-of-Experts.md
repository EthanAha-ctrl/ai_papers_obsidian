---
source_pdf: Temporally Extended Mixture-of-Experts.pdf
paper_sha256: a333296e176597db482e43ff8ff2fbfb5adb13d9292c5f04e36ba1428165f9d1
processed_at: '2026-08-12T13:29:41-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---
这个图书馆有 32 个专业书架(MoE experts),每个书架专注于不同领域:数学、历史、编程、法语、物理……你每写一个字,都要决定去哪个书架查资料。
写"今天"去书架 A,写"天气"又跑去书架 B,写"很好"又换到书架 C
这论文说:既然你在做一道数学题,接下来几十个字大概率都还是数学,你就在数学书架前蹲着别动,真要换领域了再换.
这就是所谓的 "temporally extended"

## 核心想法:把 MoE 路由当成 RL 里的 option

RL 里有个老概念叫 **options framework**(Sutton 他们 1999 年就提了),说的就是这种"一次决策管一段时间"的事。

option 就像一种"行为模式",比如"我现在进入解题模式",这个模式会持续一段时间,只有某种条件触发才退出,然后选下一个模式。

论文的核心 reformulation 就一句话:

> **MoE 的 expert mask = option;切换 mask = option 终止;expert 加载延迟 = deliberation cost**

这个对应关系非常自然,作者抓住了一个被大家忽略的 analog。

## 怎么让模型学会"少换专家"

光有概念不够,得让模型真的学会什么时候该换、什么时候忍着别换。作者用了一个叫 **option-critic with deliberation cost** 的 RL 框架,核心就是三件事:

### 1. 加一个 controller 当"交通指挥"

每个 MoE 层配一个小型 controller 网络。它看两个输入:当前 hidden state(模型在想啥)和当前在用哪几个专家。然后输出一个概率:"要不要换专家?"

如果决定换,就用 Plackett-Luce 分布重新采样一组专家;如果不换,就继续用旧的。

### 2. 给"换专家"定个价

每次换专家要付一个 cost η(叫 deliberation cost)。η 越大,模型越懒得换;η 越小,模型越频繁换。

这个 η 是超参,调它就能在"性能"和"切换频率"之间 trade-off。实验里 η ∈ {0.02, 0.03, 0.04},η=0.02 时 switch rate 还有 4% 左右,η=0.04 时压到 1.2%。

直觉就是:**只有换专家带来的质量提升 > η 这个门槛,才值得换**。像超市购物,只有省的钱超过你跑一趟的油钱,才值得开车去另一家。

### 3. 用 self-distillation 当 reward

reward 怎么定义?让 student(被训练的模型)模仿 teacher(冻结的原版 gpt-oss-20b)。每个 token 的 reward 是:

$$r_t = \log p_{\text{teacher}}(a_t) - \log p_{\text{student}}(a_t)$$

student 越像 teacher,reward 越高。这是个 on-policy self-distillation,借了 MiniLLM 的 teacher-mixing 技巧防止 reward hacking(模型学会输出低 entropy 的废话刷分)。

## 训练的时候到底在更新啥

三个东西同时更新:

**(a) LLM 本身的参数**(用 LoRA,省 memory)
让 LLM 在被限制只能用 k̂ 个专家的前提下,生成质量尽量接近 teacher。这就是 intra-option policy update。

**(b) Termination head**
学什么时候该换专家。梯度方向是:"当前 option 的价值 < 切换的期望价值 + η" 时,增大换的概率;反之减小。

**(c) Option selection head**
当真的决定换时,学应该换到哪组专家。

加上两个 critic(V_Ω 和 Q_Ω)用 GAE(λ)估价值,这是标准 actor-critic 套路。

## 实验结果有多惊艳

gpt-oss-20b,32 experts, top-4。作者把它限制到只用 16 个或 8 个专家。

**k̂=16, η=0.02 的结果**:

| Benchmark | 原版 | 静态 pruning 最好成绩 | 这论文 |
|---|---|---|---|
| MATH | 71.5 | 53.5(frequency pruning) | **64.0** |
| MMLU | 79.5 | 55.5 | **72.5** |
| MMMLU | 67.5 | 48.0 | **59.5** |
| Switch rate | ~58% | - | **4.1%** |

性能保留 ~90%,切换率从 58% 降到 4.1%。所有静态 pruning 基线都崩得很难看,Wanda 在 MATH 上只剩 3.5 分。

**k̂=8**(更极端)性能掉得更多,但 switch rate 也只有 9.2%。说明 k̂ 不能无限小,expert 数量本身的表达能力还是硬约束。

Figure 6、7 那张热力图对比特别直观:原版是满天星,这篇 paper 训出来的模型是清晰的横向色带——同一组专家在长段 token 上保持不变。

## 为什么静态 pruning 不行

静态 pruning 就是"挑 k̂ 个最有用的专家,其他永远砍掉"。问题是:**不同 token 需要不同专家**。

数学题需要数学专家,法语题需要法语专家。静态选 16 个,总有 16 个之外的需求被牺牲。Reconstruction loss minimization(Lu et al. 2024)挑出"平均 reconstruction 最好"的 16 个,但这只覆盖平均场景,长尾直接失效。

这篇 paper 的关键区别是:**动态切换**。虽然每时刻也只有 k̂ 个,但不同段落可以切到不同 k̂ 个。用 RL 的语言说,就是 option space 提供了 temporal expressiveness,这是静态方法结构上做不到的。

## 一个有意思的 insight

作者自己指出,这个方法其实有两个 gain 来源混在一起:

1. **动态路由**:可以切换 expert mask,覆盖更多场景
2. **Self-distillation**:LoRA + RL 把 LLM 参数往 teacher 方向调,补偿 capacity 损失

作者承认没做 ablation 隔离这两个。但理论上,即使没有动态切换,单 self-distillation 也能救回一点静态 pruning 的性能。这是 future work 要厘清的。

不过从实验数据看,动态切换的贡献应该占主导——静态 pruning + self-distillation 不可能把 MATH 从 53.5 拉到 64.0 这么多,因为根本没见过那些被砍掉的 expert 的能力。

## 三个被 unlock 的应用场景

作者提出这套思路能打开三扇门:

### 1. Memory-efficient serving

GPU 只需常驻 k̂ 个专家,其余 offload。两次 swap 之间用 k̂/N 的内存跑推理。gpt-oss-20b 上 k̂=8 能省 55% VRAM。

现有系统如 MoE-Infinity、ProMoE、eMoE 都是 reactive——你模型乱切,我拼命预测 prefetch。这篇 paper 是 proactive——直接让模型别乱切,系统端就轻松多了。

### 2. Memory-efficient training

训练时把 response 切成 chunk,每个 chunk 绑定一组专家,只把当前 chunk 用的专家放 GPU,其余 offload。chunk 之间换 GPU 上的专家。降低峰值显存。

### 3. Continual learning

加新专家不影响 active compute(k̂ 固定)。新 domain 来时加新专家,让 controller 学会路由过去。这就是 "Mixture of a Million Experts"(He 2024)的可行 serving 路径——百万专家全驻 GPU 不现实,但 1% 切换率下,prefetching 完全跟得上。

## 一些直觉性的 takeaway

**1. Temporal structure is everywhere in language**

自然语言天然有 temporal structure:一段话里的话题、论点、推理链条,都会持续好几十个 token。当前 MoE 完全没利用这种 structure,每 token 重新决策,是巨大的浪费。

**2. "Cost" 是 RL 的语言,但本质是工程约束**

deliberation cost η 在 RL 里是个抽象标量,但语义上就是"切换专家的 latency"。如果未来有人测出真实 PCIe 带宽下的 expert load time,把 η 设成实测值,policy 学到的切换率就直接对齐实际硬件需求。

**3. Post-training 是 patch,pre-training 才是根治**

这论文是给已经训练好的 gpt-oss-20b 打补丁。理想情况下,应该在预训练阶段就把 temporal continuity 作为 objective 的一部分,让模型一开始就学会"长时间用一组专家"。这样连 controller 都不用加,router 本身就具备 temporal awareness。

**4. Per-layer 独立是工程妥协**

理论上所有层同步切换最省内存(一次 swap 全搞定),但联合 option 空间组合爆炸,没法训。作者用 per-layer 独立 controller,每层自己决定何时切,层与层之间不同步。这是个 tractability vs. optimality 的 trade-off。

Figure 6、7 看得出来,不同 layer 的 temporal continuity 模式不一样,层 0 可能切得频繁些,层 2 可能切得更稀疏。这也暗示不同 layer 处理的"概念粒度"不同。

**5. Self-distillation 不是 ad-hoc trick,是 option-critic 的天然组件**

option-critic 的 intra-option policy update,在 LLM context 下就是"让 LLM 在当前 mask 下表现好"。怎么定义"好"?模仿 teacher 是最自然的选择。所以 self-distillation 不是额外加的,是这个框架的内在要求。作者特别强调这点,我觉得这是 paper 的一个深层 insight。

## 一个有点野的联想

这论文让我想到 **Adaptive Computation Time**(Graves 2016 那篇)。ACT 是每 token 决定"想多久才输出下一个 token",PonderNet 是它的现代版。MoE option termination 是每 token 决定"要不要换一组专家"。

两者本质都是 **在 transformer 内部引入 temporal asymmetry**:不是每个 token 都平等地使用计算资源,而是根据需要动态分配。ACT 动态调深度,这论文动态调宽度(expert 集合)。

如果两者结合,可能能做出"该浅则浅、该窄则窄"的 transformer——既省 compute 又省 memory。这是 adaptive computation 的完整版。

## 另一个联想:和 residual stream 的 temporal abstraction

Kobayashi et al. 2025(https://arxiv.org/abs/2512.20605)那篇 paper 发现 autoregressive model 的 residual stream 里会自发 emerge temporal abstraction,可以直接用于 hierarchical RL。

这论文的 controller 看的 h_t^(ℓ) 就是 residual stream 的状态。理论上,residual stream 里已经 encode 了"我现在在哪个 reasoning 阶段"的信号,controller 只需要学会读这个信号来决定何时换专家。

也许未来可以做得更激进:**不用单独训 controller,直接从 residual stream 里读 emergent 的 temporal abstraction 信号,映射成 expert switching decision**。这样 controller 几乎零参数,完全 leveraging 模型内部的 emergent structure。

## 总结一句话

**MoE 的 routing 一直在偷偷交"切换税",但大家假装没看见。这论文用 RL 的 options framework 把这个税显式化,加个轻量 controller 学会"能不换就不换",在几乎不损失性能的前提下把切换率从 95% 压到 4%,为未来超大专家规模的 MoE 打开了一条实际可部署的路。**

核心参考:
- 论文本体:Princeton, Zeyu Shen & Peter Henderson
- Option-Critic: https://arxiv.org/abs/1609.05140
- Options framework 原始论文: https://www.sciencedirect.com/science/article/pii/S0004370299000521
- Deliberation cost: https://arxiv.org/abs/1709.04571
- gpt-oss: https://arxiv.org/abs/2508.10925
- MiniLLM(teacher mixing 灵感): https://arxiv.org/abs/2306.08543
- On-policy distillation blog: https://thinkingmachines.ai/blog/on-policy-distillation/
- Mixture of a Million Experts: https://arxiv.org/abs/2407.04153
- Emergent temporal abstractions: https://arxiv.org/abs/2512.20605
- Adaptive Computation Time(联想): Alex Graves 2016, https://arxiv.org/abs/1603.08983

---

# Temporally Extended Mixture-of-Experts: 用 Options Framework 重塑 MoE 路由

## 一、核心问题与直觉构建

这篇 paper 直击一个被广泛忽略的问题:**现代 MoE LLM 的 expert routing 几乎在每个 token 上都在切换**。作者测了三个 frontier open-source MoE,gpt-oss-20b、gpt-oss-120b、Qwen3-Next-80B-A3B,switch rate 都在 0.94-1.00 之间(见 Table 1)。

为什么这是问题?当模型大到 GPU memory 装不下所有 expert 时,必须 offload 到 host memory 或 disk,按需 fetch。每 token 切换 expert 意味着每 token 都要 fetch,这会让 prefetching/offloading 优化失效,带来不可接受的 latency。Figure 2 可视化了 layer 0 在一个 trajectory 上的 expert activation pattern,几乎是一个 noisy 的散点图,毫无 temporal continuity。

作者的洞察:**这其实就是 RL 中"temporally extended actions" 问题**。RL 中 options framework(Sutton, Precup, Singh 1999, https://www.sciencedirect.com/science/article/pii/S0004370299000521)正是用来处理这种"在一段时间内保持一个高层决策、只在必要时切换"的 tradeoff。把 expert mask 当作 option,把 expert load latency 当作 deliberation cost,问题就被 beautifully formalized。

## 二、Options Framework 与 s-MDP 的形式化

### 2.1 s-MDP 和 Options 的本质

标准 MDP 是 (S, A, P, r, γ):状态 s、动作 a、转移核 P、奖励 r、折扣 γ。policy π(a|s) 诱导 trajectory τ 和 return G(τ) = Σ_{t=0}^{T-1} γ^t r(s_t, a_t)。

s-MDP 是 MDP 的泛化:agent 在 decision time t_k 选一个 high-level action,environment 演化一个随机时长 κ_k,agent 收到这段时间累计 reward 后再做下一个决策。

Option ω ∈ Ω 是一个三元组 (T_ω, π_ω(a|s), β_ω(s)):
- **T_ω ⊆ S**:initiation set,可启动 option 的状态集合
- **π_ω(a|s)**:intra-option policy,option 活跃时用的 primitive action 分布
- **β_ω(s) ∈ [0,1]**:termination function,到达状态 s 后终止的概率

policy over options π_Ω(ω|s) 决定在状态 s 启动哪个 option。采用 **call-and-return 执行模型**:开始采样 ω_0 ~ π_Ω(·|s_0);option 活跃时,a_t ~ π_ω(·|s_t);每次转移到 s_{t+1} 后以概率 β_ω(s_{t+1}) 终止,如果终止则采样新 option,否则继续。

直觉:option 就像"一个行为模式",一旦进入会持续一段时间,只有明确"该结束"才退出。

### 2.2 MoE Routing 形式化为 Options

每个 MoE layer ℓ 有 N 个 expert,token position t 在 layer ℓ 上 router 产生 logits g_t^(ℓ) ∈ R^N,p_t^(ℓ) = softmax(g_t^(ℓ))。

核心 reformulation:**binary expert mask ω_t^(ℓ) ∈ {0,1}^N 就是 option**。ω_{t,i}^(ℓ) = 1 表示 expert i 在时间 t 在 layer ℓ 被允许。router 的 top-k 操作被限制在 allowed expert 集合内。

记号约定(很关键,容易混):
- **k̂**(k-hat):被 mask 允许的 expert 数量(option 的"大小")
- **k̃**(k-tilde):router 实际激活的 expert 数量(top-k 中的 k)

switch 发生在 ω_t^(ℓ) ≠ ω_{t-1}^(ℓ)。switch rate:
$$\frac{1}{L}\sum_{\ell=1}^{L}\frac{1}{T-1}\sum_{t=1}^{T-1}\mathbf{1}[\omega_t^{(\ell)} \neq \omega_{t-1}^{(\ell)}]$$

对 base model,把 router 自己当作 option-selection policy:用 router logits 取 top-k̂ 作为初始 mask,一直保持,直到某时刻 router 想激活的 k̃ 个 expert 不全在这个 mask 中,触发 switch。这定义了一个 reference switch rate,显示 base model 有多"想"切换。

## 三、Option-Critic Architecture 与梯度推导

### 3.1 核心 Q-value 定义

Option-Critic(Bacon, Harb, Precup 2016, https://arxiv.org/abs/1609.05140)把 policy gradient theorem 扩展到 options framework,允许同时优化 π_ω 和 β_ω。

**Q_U(s, ω, a)**:在状态 s、option ω 活跃时执行 action a 的价值
$$Q_U(s, \omega, a) = r(s,a) + \gamma \sum_{s'} P(s'|s,a) U(\omega, s')$$

- s: 当前状态(MoE 中即 hidden state h_t^(ℓ))
- ω: 当前 option(当前 expert mask)
- a: primitive action(下一 token)
- γ: 折扣因子
- U(ω, s'):在状态 s' 且 option ω 仍活跃时的价值

**U(ω, s')**:option 继续或终止的期望
$$U(\omega, s') = (1 - \beta_\omega(s')) Q_\Omega(s', \omega) + \beta_\omega(s') V_\Omega(s')$$

- 第一项:option 不终止,继续执行 ω 的价值
- 第二项:option 终止,重新选择 option 的价值

**Q_Ω(s, ω)**:在状态 s 执行 option ω 的价值
$$Q_\Omega(s, \omega) = \sum_a \pi_\omega(a|s) Q_U(s, \omega, a)$$

**V_Ω(s)**:在状态 s 的价值(对所有 option 求期望)
$$V_\Omega(s) = \sum_\omega \pi_\Omega(\omega|s) Q_\Omega(s, \omega)$$

### 3.2 Intra-Option Policy Gradient Theorem

**Theorem 1**(优化 intra-option policy π_ω 参数 θ):
$$\frac{\partial Q_\Omega(s_0, \omega_0)}{\partial \theta} = \sum_{s,\omega} \mu(s, \omega) \sum_a \frac{\partial \pi_\omega(a|s)}{\partial \theta} Q_U(s, \omega, a)$$

- μ(s, ω):从初始 (s_0, ω_0) 出发的 discounted state-option visitation distribution
- 这个 gradient 说:在经常访问的 (s, ω) 处,把 π_ω 朝能获得高 Q_U 的 action a 方向推

用 log-derivative trick:
$$\mathbb{E}_{(s,\omega)\sim\mu, a\sim\pi_{\omega,\theta}}\left[\frac{\partial \log \pi_\omega(a|s)}{\partial \theta} Q_U(s, \omega, a)\right]$$

### 3.3 Termination Gradient Theorem

**Theorem 2**(优化 termination function β_ω 参数 ν):
$$\frac{\partial Q_\Omega(s_0, \omega_0)}{\partial \nu} = -\sum_{s,\omega} \mu(s, \omega) \frac{\partial \beta_\omega(s)}{\partial \nu}(Q_\Omega(s, \omega) - V_\Omega(s))$$

直觉:**Q_Ω(s,ω) - V_Ω(s) 是"坚持当前 option 相对于切换"的 advantage**。如果当前 option 价值高于平均,advantage 为正,gradient 推 β_ω 减小(降低终止概率,延长 option);反之增大 β_ω(倾向切换)。

### 3.4 加 Deliberation Cost

Harb et al. 2017(https://arxiv.org/abs/1709.04571)指出 options framework 真正发挥作用是在有 **deliberation cost** 时:每次切换要付 η 的代价。termination gradient 变为:
$$-\sum_{s,\omega} \mu(s, \omega) \frac{\partial \beta_\omega(s)}{\partial \nu}(Q_\Omega(s, \omega) - V_\Omega(s) + \eta)$$

η 是一个 margin:只有当前 option 价值比"切换的期望价值 + η"还要差时,才偏好切换。这把 switching rate 和 quality 显式绑成一个可调 trade-off,作者实验中 η ∈ {0.02, 0.03, 0.04}。

## 四、Controller 架构详解

### 4.1 选项空间与分层结构

每个 layer ℓ 的 option 空间是 N 个 expert 中选 k̂ 个 mask:
$$\Omega^{(\ell)} = \{\omega \in \{0,1\}^N : \|\omega\|_1 = \hat{k}\}$$

理论上可以把所有 layer 的 mask 联合作为一个 option(在 L 层联合 s-MDP 上学习),但 option 空间组合爆炸。作者采用 **per-layer 独立 controller** 的 factorization:每层 controller 看自己层的 hidden state 和当前 mask,把网络其他部分当 environment。这是对 joint s-MDP 的近似,但稳定且实际有效。

### 4.2 状态、嵌入与各 head

**State**:LLM 在 layer ℓ 进入 MLP 前的 hidden representation h_t^(ℓ)。

**Expert set embedding(DeepSets)**:
$$z^{(\ell)}(\omega) = \frac{1}{\hat{k}} \sum_{i \in \omega} \varphi(e_i)$$

- e_i ∈ R^{d_e}(d_e=128):expert i 的 learned embedding
- φ: R^{d_e} → R^{d_c}:两层 GELU MLP,hidden dim 1024
- DeepSets(Zaheer et al. 2017, https://arxiv.org/abs/1703.06114)保证 mask 是 permutation-invariant 的(因为 expert 集合是无序的)

**Termination head**:
$$\beta_t^{(\ell)} = \sigma\left(\text{MLP}_\beta\left(\text{concat}\left(\overline{h}_t^{(\ell)}, \overline{z}^{(\ell)}(\omega_{t-1}^{(\ell)})\right)\right)\right)$$

- h̄_t^(ℓ) = RMSNorm(h_t^(ℓ)),z̄^(ℓ) = RMSNorm(z^(ℓ)):用 RMSNorm 平衡两个表示的 scale
- MLP_β:两层 ReLU MLP
- σ: sigmoid,得到 [0,1] 概率
- 采样 d_t^(ℓ) ~ Bernoulli(β_t^(ℓ))
- **bias 初始化为 -3**,对应初始 switch prob σ(-3) ≈ 0.05,从一开始就鼓励 temporal continuity

**State-value head V_Ω**(linear,初始化自 router weights):
$$V_\Omega(h_t^{(\ell)}) = w_V^\top h_t^{(\ell)} + b_V$$

**Option-value head Q_Ω**:
$$Q_\Omega(h_t^{(\ell)}, \omega) = \text{MLP}_Q\left(\text{concat}\left(\overline{h}_t^{(\ell)}, \overline{z}^{(\ell)}(\omega)\right)\right)$$

MLP_Q 也是两层 ReLU MLP。

### 4.3 Option Selection: Plackett-Luce 与 Gumbel-top-k

当 d_t^(ℓ) = 1 时,要采样新 option。selection head f_sel^(ℓ): R^d → R^N 是从 router weights 初始化的 linear layer,产生 logits c_t^(ℓ) = f_sel(h_t^(ℓ))。

通过 **Plackett-Luce(PL)分布** 采样 k̂ 个 expert(无放回的顺序采样):
$$P_{PL}(i_1, \dots, i_{\hat{k}} | c) = \prod_{j=1}^{\hat{k}} \frac{\exp(c_{i_j})}{\sum_{m \notin \{i_1, \dots, i_{j-1}\}} \exp(c_m)}$$

- (i_1, ..., i_k̂):一个 ordered tuple
- 第 j 步:在还没被选过的 expert 里,按 softmax(c) 采一个

实际实现用 **Gumbel-top-k trick**:给每个 logit 加 i.i.d. Gumbel(0,1) 噪声,然后取 top-k̂ 索引。数学上等价于 PL 采样,但完全 vectorized。

注意:虽然 mask 是 unordered set,但 policy gradient 用 ordered tuple 的 PL log-prob 计算。新 option ω_t^(ℓ) = {i_1, ..., i_k̂}。

### 4.4 Controller 在 MoE 层中的集成

Figure 3、4 的架构:
1. Controller 观察 h_t^(ℓ) 和上一时刻 option ω_{t-1}^(ℓ)
2. 计算 termination probability β_t^(ℓ),采样 d_t^(ℓ)
3. 若 d_t^(ℓ) = 1,通过 PL 采样新 option;否则 ω_t^(ℓ) ← ω_{t-1}^(ℓ)
4. 用 ω_t^(ℓ) 作为 mask,把不在 mask 中的 expert logit 设 -∞,再做 top-k̃ 路由
5. 被灰色化(mask out)的 expert 不参与 forward

t=0 时强制 d_0^(ℓ) = 0,初始 option 用 router logits 的 top-k̂。

## 五、训练:Reward 设计与梯度更新

### 5.1 Self-Distillation Reward

目标是把预训练 MoE 转成 temporally extended MoE 同时保持质量。采用 per-token reverse KL(Kevin Lu 2025, https://thinkingmachines.ai/blog/on-policy-distillation/):
$$r_t = \log p_{\text{teacher}}(a_t | x, a_{<t}) - \log p_{\text{student}}(a_t | x, a_{<t})$$

- teacher:冻结的原始 gpt-oss-20b(无 controller 无 weight update)
- student:正在训练的 model
- 期望意义下,-r_t 是 reverse KL(p_student || p_teacher) 的无偏估计

**为什么 reverse KL 而不是 forward KL**:reverse KL 在 p_teacher 接近 0 的地方没有惩罚,鼓励 student 集中在 teacher 高概率区域,mode-seeking。在这里我们想 student 模仿 teacher 的 distribution,reverse KL 配 on-policy sampling 是合适的。

### 5.2 防 Reward Hacking:Teacher Mixing

直接优化 reverse KL 会出现 degenerate:student 学会输出极低 entropy 的 degenerate 分布(比如重复同一 token),这时 r_t 在该 token 上很高。借鉴 MiniLLM(Gu et al. 2026, https://arxiv.org/abs/2306.08543):
$$p_{\text{mix}} = (1-\tau) p_{\text{student}} + \tau p_{\text{teacher}}$$

τ = 0.2。token 从 p_mix 采样,然后用 importance weight w_t = p_student(a_t) / p_mix(a_t) 校正 off-policy。这是一个 biased 近似(严格 importance weight 应该是 ∏_{t'≤t} 比值的连乘),但作者引用 MiniLLM 发现这降低 variance、效果更好。

### 5.3 Critic 学习:GAE(λ) TD targets

V_Ω 和 Q_Ω 通过 squared TD error with GAE(λ)(Schulman et al. 2018, https://arxiv.org/abs/1506.02438)学习。

V_Ω 的 TD error:
$$\delta_t^V = r_t + \gamma V_\Omega(h_{t+1}^{(\ell)}) - V_\Omega(h_t^{(\ell)})$$

Q_Ω 的 TD error(bootstrap U):
$$\delta_t^Q = r_t + \gamma U(\omega_t^{(\ell)}, h_{t+1}^{(\ell)}) - Q_\Omega(h_t^{(\ell)}, \omega_t^{(\ell)})$$

GAE advantages:
$$\hat{A}_t^V = \sum_{j=0}^\infty (\gamma\lambda)^j \delta_{t+j}^V, \quad \hat{A}_t^Q = \sum_{j=0}^\infty (\gamma\lambda)^j \delta_{t+j}^Q$$

Targets:
$$\hat{V}_t^{\text{targ}} = V_\Omega(h_t^{(\ell)}) + \hat{A}_t^V, \quad \hat{Q}_t^{\text{targ}} = Q_\Omega(h_t^{(\ell)}, \omega_t^{(\ell)}) + \hat{A}_t^Q$$

Joint critic loss:$(V_\Omega - \hat{V}_t^{\text{targ}})^2 + (Q_\Omega - \hat{Q}_t^{\text{targ}})^2$。

### 5.4 三个梯度更新

**(1) Intra-option policy update**(用 Monte Carlo return 估 Q_U):
$$\bar{G}_t = \sum_{j \geq 0} \gamma^j r_{t+j}$$
$$d\theta \mathrel{+}= w_t \nabla_\theta \log \pi_{\omega,\theta}(a_t) \cdot \bar{G}_t$$

θ 包括 expert 和 attention 参数(LoRA r=16, α=16)和 router weights。这里 π_ω 是 LLM next-token distribution。

**(2) Termination gradient with deliberation cost**:
$$d\nu \mathrel{-}= w_t \nabla_\nu \beta_t^{(\ell)} \left(Q_\Omega(h_t^{(\ell)}, \omega_{t-1}^{(\ell)}) - V_\Omega(h_t^{(\ell)}) + \eta\right)$$

注意 η 加在 advantage 上:只有当切换 advantage > η 时,gradient 才推 β 增大。η 越大,switching 越被惩罚。

**(3) Option selection gradient(只在 switch 时)**:
$$d\phi \mathrel{+}= w_t \nabla_\phi \log \pi_{\text{sel}}(\omega_t^{(\ell)} | h_t^{(\ell)}) \left(Q_\Omega(h_t^{(\ell)}, \omega_t^{(\ell)}) - V_\Omega(h_t^{(\ell)})\right)$$

只在 d_t^(ℓ) = 1 时更新。优势是"新 option 相对 state 平均"的价值。

### 5.5 Advantage Normalization(关键稳定性技巧)

Termination advantage:
$$A_t^{\text{term},(\ell)} = Q_\Omega(h_t^{(\ell)}, \omega_{t-1}^{(\ell)}) - V_\Omega(h_t^{(\ell)}) + \eta$$

每层独立做 RMS normalization(无 mean centering,保留 sign):
$$\hat{A}_t^{\text{term},(\ell)} = \frac{A_t^{\text{term},(\ell)}}{\text{RMS}(A^{\text{term},(\ell)})}, \quad \text{RMS}(A^{\text{term},(\ell)}) = \sqrt{\frac{1}{T-1}\sum_{t=1}^{T-1}(A_t^{\text{term},(\ell)})^2}$$

Selection advantage 类似,但只在 switch positions S^(ℓ) = {t : d_t^(ℓ) = 1, t > 0} 上算 RMS。

Intra-option advantage 用 standardized return:
$$\hat{A}_t^{\text{intra}} = \frac{\bar{G}_t - \mu}{\sigma}, \quad \mu = \frac{1}{T}\sum_{t=1}^T \bar{G}_t, \quad \sigma = \sqrt{\frac{1}{T}\sum_{t=1}^T (\bar{G}_t - \mu)^2}$$

这个 advantage 在所有 layer 间共享。

## 六、实验设置全貌

### 6.1 模型与硬件

- Model: **gpt-oss-20b**(https://arxiv.org/abs/2508.10925),24 transformer layers, 32 experts/layer, top-4 routing(k̃=4)
- Native MXFP4 quantization,dequantize 到 bf16 训练
- 硬件:4× NVIDIA H200 140GB GPU
- 框架:基于 TRL(https://github.com/huggingface/trl)的修改版

### 6.2 Hyperparameters

- γ = 0.95(折扣因子)
- λ = 0.95(GAE)
- value loss coef = 0.01
- α_controller = 1e-4(AdamW)
- α_intra = 2e-4
- LoRA: r=16, α=16(同时加在 expert 和 attention 参数上)
- batch: 16 prompts, max prompt len 512, max response len 512
- sampling: temp=1.0, top-p=0.95
- teacher mixing ratio τ=0.2
- deliberation cost η ∈ {0.02, 0.03, 0.04}
- expert budget k̂ ∈ {8, 16}

### 6.3 数据集

- Training: **Nemotron Post-Training Dataset v2**(https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2),10 类(chat/code/math/STEM/multilingual)
- Eval: MATH(https://arxiv.org/abs/2103.03874)、MMLU(https://arxiv.org/abs/2009.03300)、MMMLU,各 200 题
- Eval 温度 0.5,top-p 0.95,max response 2048

### 6.4 Baselines

四个 pruning baselines,128 个 calibration prompts:
- **Frequency-based**:保留 calibration set 上最频繁使用的 k̂ 个 expert
- **Reconstruction loss minimization**(Lu et al. 2024, https://arxiv.org/abs/2402.14800):保留使 reconstruction loss 最小的 k̂ 个 expert。原作者在 N=8 上做 exhaustive search,N=32 不可行,作者用 greedy forward selection 替代
- **Random**:每 token 随机选 k̂ 个 expert
- **Wanda structured**(https://arxiv.org/abs/2306.11695):structured weight pruning,prune (N-k̂)/N 比例

## 七、实验结果深度解读

### 7.1 Switch Rate(关键发现)

Table 1 显示 base model 的 switch rate:
| Model | 平均 switch rate |
|---|---|
| gpt-oss-20b | 0.94-0.95 |
| gpt-oss-120b | 0.98-0.99 |
| Qwen3-Next-80B-A3B | 1.00 |

**所有现代 MoE 模型几乎每 token 都在切换 expert**。这意味着任何 offloading-based serving 都不可避免地付出 latency。

### 7.2 k̂ = 16 结果(Table 2)

| Benchmark | Base | Freq | Recon | Random | Wanda | Ours η=0.02 | η=0.03 | η=0.04 |
|---|---|---|---|---|---|---|---|---|
| MATH | 71.5±5.9 | 53.5 | 51.5 | 15.0 | 3.5 | **64.0±6.7** | 58.5 | 55.0 |
| MMLU | 79.5±5.7 | 55.5 | 35.0 | 33.5 | 9.0 | **72.5±6.3** | 67.5 | 63.0 |
| MMMLU | 67.5±6.5 | 42.0 | 48.0 | 24.0 | 7.0 | **59.5±6.9** | 56.5 | 49.5 |
| switch % | ~58% | - | - | - | - | **4.1%** | 1.3% | 1.2% |

η=0.02 时 switch rate 从 ~58% 降到 4.1%(MATH 上),性能保留 64/71.5 ≈ 90%。所有 pruning baselines 都崩了,Reconstruction 最好也才到 51.5%(MATH)。

### 7.3 k̂ = 8 结果(Table 3,更极端的 budget)

| Benchmark | Base | Ours η=0.02 |
|---|---|---|
| MATH | 71.5±5.9 | 27.5±6.1 |
| MMLU | 79.5±5.7 | 48.5±6.9 |
| MMMLU | 67.5±6.5 | 39.0±6.5 |
| switch % | ~79% | **9.2%** |

k̂ 减半到 8,switch rate 略升到 9.2%,但性能大幅下降。这揭示一个 trade-off:temporal extension 能"挽救"一部分容量损失,但 k̂ 太小没有足够 experts 表达能力,RL controller 也救不回来。

### 7.4 训练动态(Figure 5)

- **Reward**:稳定上升,k̂=8 时提升更明显
- **Switch rate**:初期下降(critic 还在学),然后稳定到 η 决定的水平。η 越大,converged switch rate 越低
- **Perplexity**(teacher 评估 student 输出):持续下降,说明 student 输出越来越对齐 teacher,**没有 collapse**

Figure A2 显示:虽然平均 switch rate 先降后微升,但 **95 percentile switch rate 和 std 持续上升**,说明 termination head 在学"distinguish when to switch vs not",而不是均匀低 switch。

### 7.5 Temporal Continuity 可视化(Figure 6, 7, A4-A7)

对比 Figure 2(base model)的散点图,作者训练的 controller 在 layer 0/1/2 上 expert mask 显示**强烈的 horizontal banding**:同一组 expert 在长段连续 token 上保持不变,只在必要处切换。

不同 layer 有不同的 temporal continuity pattern,这暗示不同 layer 的"概念粒度"不同。

### 7.6 训练稳定性(Figure A3)

- Repetition rate(1 - fraction of unique tokens):稳定在健康范围,没出现 MoE 约束 routing 常见的 catastrophic repetition
- Teacher perplexity on student outputs:持续下降,student 与 teacher 对齐而非发散

### 7.7 定性示例(Section A5)

MATH 题目:求 1!+2!, ..., 8!+9! 的 LCM。
- **Ours**:完整、正确的推理过程
- **Reconstruction**:开始正确,快速退化到 gibberish
- **Random**:部分正确,然后陷入"2592·25 = 2592·25 = ..." 的 repetition loop
- **Wanda**:直接陷入"40320 + 362880 = ..."的无限 repetition

这直观展示:静态 pruning 破坏 capacity 严重,即使 LLM 还能输出 token,推理能力已经崩溃。动态 controller 即使在受限 expert budget 下,也能保持 multi-step reasoning。

## 八、Three Missed Opportunities(第 3.2 节)

作者识别 temporal extension 能 unlock 的三个场景:

### 8.1 Memory-efficient Inference Serving

标准 MoE serving 必须把 N 个 expert 都驻 GPU,或随时准备 fetch 任意一个。有 temporal continuity,只需 k̂ 个 active expert 在 GPU,偶尔 swap。两次 swap 之间,推理以 k̂/N 的 expert memory footprint 运行。

gpt-oss-20b 中 expert 参数占 96%+,k̂=16 可省 ~4.7 GiB(37%) VRAM;k̂=8 可省 ~7.1 GiB(55%)。这直接转化为 serving 上的 GPU 需求降低。

相关系统工作:
- MoE-Infinity(https://arxiv.org/abs/2401.14361):offload 到 host,用 expert activations 预测 cache
- ProMoE(https://arxiv.org/abs/2410.22134):用 activations prefetch
- eMoE(https://arxiv.org/abs/2503.06823):跨层、跨 prompt 相关性
- DuoServe-MoE(https://arxiv.org/abs/2509.07379):CPU offload + 不同 scheduling

但这些都是 reactive 的预测,本质还是被 base model 的高 switch rate 困住。作者的方法是 **proactive 修改模型本身**,让 switch rate 低下来。

### 8.2 Memory-efficient Training via Temporal Chunking

训练时所有 expert 都要在 forward/backward 中可访问。有 temporal continuity,可以把 response 分成 chunk,每个 chunk 内固定 mask,只 k̂ 个 expert 参与计算。Inactive expert 在 chunk 的 forward-backward 期间 offload。降低 peak GPU memory。

### 8.3 Continual Learning with Expandable Experts

只有 k̂/N 个 expert active,可以不停加新 expert 而不增加 per-token compute 或 active memory。新 domain/任务来时初始化新 expert,让 controller 学路由到它们。k̂ 固定保证 inference cost 不变。这呼应 He 2024(https://arxiv.org/abs/2407.04153, Mixture of a Million Experts)的 vision,但作者补上了"如何可控路由"的机制。

## 九、Method 与 Option-Critic 的对应关系

作者的方法本质上把 option-critic with deliberation cost(Harb et al. 2017, https://arxiv.org/abs/1709.04571)适配到 MoE 这个 setting。对应关系:

| Option-Critic 概念 | MoE 设置 |
|---|---|
| State s | h_t^(ℓ)(LLM hidden state) |
| Option ω | expert mask ω_t^(ℓ) |
| Intra-option policy π_ω | LLM next-token distribution |
| Primitive action a | 生成的 token |
| Termination β_ω | controller termination head |
| Policy over options π_Ω | option selection head(PL 分布) |
| Reward r_t | per-token reverse KL |
| Deliberation cost η | 超参(语义上是 expert loading latency 的代理) |
| Option duration | 两次 switch 之间的 token 数 |

Klissarov & Precup 2021(https://arxiv.org/abs/2112.03097)的 flexible option learning 思路也对得上:在多个 option 与当前 primitive action 一致时,同时更新所有这些 option 的 intra-option policy。这里因为 option 是 mask,直接看 mask 一致即可,作者没显式用这个 trick 但理论上等价。

## 十、Limitations 与 Future Directions(Section A4)

作者诚实地列出了几个限制:

1. **From philosophy to deployment**:目前只是验证 temporally extended routing 可学,未做 end-to-end memory-saving 系统;η 是 hyperparameter 不是测量的 hardware latency
2. **Pre-training 时间就引入**:post-training 加 controller 是 patch,理论上应该在 pre-training 时就把 temporal continuity 编入 routing objective
3. **Per-layer vs. cross-layer options**:per-layer 独立是 tractability 妥协,真正的内存收益要 cross-layer 同步切换才好。但 joint option 空间组合爆炸,学习困难
4. **Evaluation scope**:只测 MATH/MMLU/MMMLU 三项,各 200 题。代码、长指令、开放对话没测
5. **Disentangling temporal extension from self-distillation**:gain 来自两个源:(a) 动态切换 mask,(b) on-policy distillation 适配 weights。作者未做 ablation 把静态 pruned model + distillation 隔离出来,这是个未来工作

## 十一、Intuition 总结

我会这样总结这篇 paper 的核心 insight:

1. **MoE 的 routing churn 不是 bug,是被忽视的 cost**。一旦 expert 数量超过 GPU memory,这个 cost 才暴露出来。现有 MoE 是"在 memory 充足假设下"训练的,没人体会 switching cost。

2. **Options framework 是天然 formalism**。Option 就是"持续一段时间的 commitment",termination 决策正是"什么时候切换 expert set"。deliberation cost η 一句话搞定 quality/switch-rate 的 trade-off。

3. **Post-training 也能 fix**。即使模型预训练时没考虑 temporal extension,通过加 controller + LoRA + self-distillation,可以在保留 ~90% 准确度下把 switch rate 从 ~95% 降到 ~4%。

4. **Self-distillation 是 option-critic 的天然组件**。Intra-option policy update 在 MoE 里就是"调 LLM weights 让它在自己 mask 下表现得像 teacher"。Option-critic 不是 RL trick,是把 LLM post-training 重写成 RL 语言的视角。

5. **Per-layer factorization 是工程妥协**。理论上 joint s-MDP 更纯,但 per-layer 独立训练稳定且有效,这是 RL 中常见的"分而治之" pattern。

6. **自然语言本身有 temporal structure**:topic、argument、reasoning chain 都会在长 span 上持续。如果在预训练时就强制 expert routing 反映这种 structure,可能根本不需要 post-training patch。

7. **可扩展到 continual learning**:加 expert 不增加 active compute,这是"百万 expert"vision 的关键 enabler。

## 十二、可能的延伸联想(稍微 hallucinate)

- **与 Emergent Temporal Abstractions(Kobayashi et al. 2025, https://arxiv.org/abs/2512.20605)的联系**:那篇 paper 提出残差流里 emerge temporal abstraction,可以用于 hierarchical RL。这篇 MoE paper 的 controller 其实就是在 LLM "高层"做 temporal abstraction 决策,两个工作可能可以结合:用 LLM 内部 emergent 的 temporal abstraction 信号来引导 expert switching。
- **与 MA-RLHF(Chai et al. 2025, https://arxiv.org/abs/2410.02743)的联系**:把 token 序列作为 macro-action。MoE option 的 duration 本质就是一种 macro-action。可以把 expert switching decision 当作 macro-action boundary 的指示器,反馈给上层 RLHF。
- **与 Adaptive Computation Time(Graves 2016)的联系**:PonderNet、ACT 都是在每 token 决定"何时停思考"。Option termination 也是在每 token 决定"何时换 expert set"。一种可能:用 ACT-style 的 halting probability 替代 Bernoulli termination,加 budget penalty,可能更稳定。
- **与 Burst Attention、Block-sparse Attention 的联系**:如果 expert mask 是 temporal chunked 的,attention 也可以在 chunk 边界处 block-sparse 化。整个 transformer 推理路径都 chunk-aware,极致省 memory。
- **与 Diffusion MoE(eDIFF-I, ERNIE-ViLG 2.0, Wan 2.2)的联系**:这些 model 在不同 denoising stage 用不同 expert。stage 本身就是天然的 temporal chunk,option framework 直接适用。
- **Hardware-aware η tuning**:目前 η 是抽象超参。如果测出每个 expert load 的实际 latency(比如 PCIe 带宽 ÷ expert size),把 η 设成 latency 的实测值,policy 学到的 switch rate 就直接对齐实际 serving 需求。
- **Mixture of a Million Experts(He 2024, https://arxiv.org/abs/2407.04153)的 viable serving**:百万 expert 全部驻 GPU 不现实,但如果 switch rate < 1%,prefetching 是可工作的。这篇 paper 给百万 expert 提供 serving 路径。
- **与 Network Pruning 的关系**:Wanda 等结构化 pruning 是"静态选择重要专家"。这篇 paper 是"动态选择当前需要的专家集合",保留更多 capacity 的代价是加 controller。两者可以组合:先 Wanda 粗筛到 100 个 expert,再 option controller 在其中动态切。
- **Connection to Hierarchical RL in LLMs**:Kobayashi 2025、Xinhan Di 2025(https://arxiv.org/abs/2508.01604)、MA-RLHF 都在探索 LLM 内部 hierarchy。Expert switching 是 hierarchy 的另一个维度:"什么时候换计算路径",这与"什么时候换 reasoning strategy"是同构的。

## 十三、关键公式速查表

| 公式 | 位置 | 作用 |
|---|---|---|
| p_t^(ℓ) = softmax(g_t^(ℓ)) | §2.2 | Router 分布 |
| Switch rate = (1/L)Σ(1/(T-1))Σ 1[ω_t ≠ ω_{t-1}] | §2.2 | 切换频率 |
| Q_U = r + γ Σ P·U | §2.3 | Action-value under option |
| U = (1-β)Q_Ω + β·V_Ω | §2.3 | Option 终止价值 |
| Theorem 1 gradient | §2.3 | Intra-option policy update |
| Theorem 2 gradient | §2.3 | Termination update |
| z^(ℓ)(ω) = (1/k̂) Σ φ(e_i) | §4.2, Eq.(2) | DeepSets mask embedding |
| β_t^(ℓ) = σ(MLP_β(concat(h̄, z̄))) | §4.2, Eq.(3) | Termination head |
| Q_Ω = MLP_Q(concat(h̄, z̄)) | §4.2, Eq.(4) | Option-value head |
| P_PL = ∏ exp(c_i)/Σ exp(c_m) | §4.2, Eq.(5) | Plackett-Luce |
| r_t = log p_teacher - log p_student | §4.3, Eq.(6) | Reverse KL reward |
| p_mix = (1-τ)p_student + τp_teacher | §4.3, Eq.(7) | Teacher mixing |
| Termination grad with η | §4.3, Eq.(9) | Deliberation cost |
| Selection grad | §4.3, Eq.(10) | Option selection update |

## 十四、一句话提炼

**用 RL 中的 options framework 把 MoE expert routing 重新表述为 temporally extended decision problem,加一个轻量 per-layer controller + deliberation cost η,通过 on-policy self-distillation 训练,把 switch rate 从 ~95% 降到 ~4% 同时保留 ~90% 准确度,为 MoE 在 memory-constrained 环境下的 serving、training、continual learning 打开大门。**

主参考链接:
- Paper: Temporally Extended MoE(Princeton, Zeyu Shen & Peter Henderson)
- Option-Critic: https://arxiv.org/abs/1609.05140
- Options framework (Sutton, Precup, Singh 1999): https://www.sciencedirect.com/science/article/pii/S0004370299000521
- Deliberation cost (Harb et al. 2017): https://arxiv.org/abs/1709.04571
- gpt-oss model card: https://arxiv.org/abs/2508.10925
- DeepSets: https://arxiv.org/abs/1703.06114
- GAE: https://arxiv.org/abs/1506.02438
- LoRA: https://arxiv.org/abs/2106.09685
- MiniLLM: https://arxiv.org/abs/2306.08543
- On-policy distillation blog: https://thinkingmachines.ai/blog/on-policy-distillation/
- MATH dataset: https://arxiv.org/abs/2103.03874
- MMLU: https://arxiv.org/abs/2009.03300
- Nemotron Post-Training Dataset v2: https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2
- Reconstruction pruning (Lu et al.): https://arxiv.org/abs/2402.14800
- Wanda: https://arxiv.org/abs/2306.11695
- MoE-Infinity: https://arxiv.org/abs/2401.14361
- ProMoE: https://arxiv.org/abs/2410.22134
- TRL: https://github.com/huggingface/trl
- Mixture of a Million Experts: https://arxiv.org/abs/2407.04153
