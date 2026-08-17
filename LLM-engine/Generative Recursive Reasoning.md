---
source_pdf: Generative Recursive Reasoning.pdf
paper_sha256: c53be1192d945358fc0fa31cab4c9d8ffc09fbd686340161de504ebeb24de0f7
processed_at: '2026-08-04T14:14:34-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GRAM 用人话讲

## 一句话版本

让神经网络"在脑子里反复想一个问题"的时候，**每次想的方式都不一样**，这样就能探索不同的思路，找到不同的答案。

## 展开讲讲

### 先说 problem：怎么让模型"想得更久"

现在 LLM 的做法是 CoT——生成更多 token。这就像考试时把每一步推导都写出来。直观但贵：你想得越久，生成的字就越多，又慢又花钱。

另一条路是 RRMs (Recursive Reasoning Models)。思路是：别写出来了，在 latent state 里反复 refine。用同一套 weights 反复 apply 一个 transition function，像你在脑子里转来转去想一个问题。HRM、TRM、Looped Transformer、Universal Transformer 都是这个路线。

这条路 appealing 的地方：reasoning depth 和 parameter scale / output length 解耦了。一个小模型也能想很多步。

### 但 RRMs 有个致命问题：它们是 deterministic 的

给定同样 input + 同样 init，每次跑出来的思考路径一模一样，最后 converge 到同一个答案。

这听起来好像没什么问题？但想想两个场景：

**场景一：N-Queens 有多个合法解**

8×8 的 N-Queens，去掉几个 queen 让你补全，可能有 3 个合法解。Deterministic 模型跑 100 次，100 次都 converge 到同一个解。其他两个解它永远找不到。这就是 Table 1 里 TRM coverage 只有 36.1% 的原因——不是它解不出来，是它**只会一种解法**。

**场景二：思考路径走偏了**

Deterministic 模型如果某一步走错了方向，后面就只能硬着头皮往下走，没法重来。就像你做数学题，一开始思路错了，再怎么推导都是错的，但 deterministic 模型没有"重来"的能力。

### GRAM 的 insight：把思考变成 stochastic 的

核心 idea 非常简单。原来 deterministic 的 update：

```
h_t = f_H(h_{t-1}, l_t)   # u_t
```

改成：

```
h_t = u_t + ε_t
```

其中 `ε_t` 从一个 learned Gaussian 采样：

```
ε_t ~ N(μ_θ(u_t), σ_θ²(u_t) I)
```

就这么简单。在 deterministic proposal `u_t` 基础上加一个**有方向的随机扰动**。

这里几个变量讲清楚：
- `u_t`：deterministic 算出来的下一步思考方向，和 prior RRMs 一样
- `μ_θ(u_t)`：一个 neural network 算出来的 mean，学的是"给定当前思考状态，朝哪个方向扰动最有用"
- `σ_θ²(u_t)`：另一个 neural network 算出来的 variance，学的是"当前该 explore 多少"
- `ε_t`：实际采样出来的扰动，叫 **stochastic guidance**

跑同一个问题 100 次，每次 `ε_t` 不同，思考路径就不同，到达的答案也不同。

### 为什么是 additive 而不是 VAE 风格的 reparam？

这个设计有讲究。`u_t + ε_t` 的形式让 deterministic part 和 stochastic part 解耦：

- `u_t` 的 gradient 直接 backprop，没有 sampling 的 noise 干扰
- `ε_t` 通过 reparameterized Gaussian `μ_θ(u_t) + σ_θ(u_t) * ξ` (其中 `ξ ~ N(0,I)`) 采样，gradient 也能传

这和 diffusion model 的 forward process 有结构相似性，但本质不同。Diffusion 是 fixed schedule 的加噪去噪，GRAM 的 noise 是 **state-dependent 且 learned** 的。每一步的 noise 分布都根据当前思考状态动态调整。

### Hierarchical 结构：slow thinking 和 fast thinking

GRAM 的 latent state `z = (h, l)` 是分层的：

- **h (high-level)**：每 outer step 更新一次，carry abstract reasoning state。这是"慢思考"，做 strategic decisions
- **l (low-level)**：每次 inner refinement 更新 K 次，carry fine-grained computation。这是"快思考"，polish details

关键：**stochasticity 只在 high-level 注入**。Low-level 是 fully deterministic 的。

为什么这么设计？直觉是：你做数学题的时候，高层战略决策（"用什么方法？"）应该有多个选择，可以探索不同路径；但底层执行（"把这一步算出来"）应该是确定的，不需要随机性干扰。

这让人想到 Kahneman 的 System 1 / System 2：
- System 1 (fast)：low-level `f_L`，快速 deterministic refinement
- System 2 (slow)：high-level `f_H + ε_t`，维护多个 hypotheses，strategic exploration

### 训练：怎么训一个"随机思考"的模型？

这是 paper 里技术含量最高的部分。

GRAM 是个 latent variable model。完整 trajectory `τ = (z_0 → ... → z_T)` 是 latent variable，要 marginalize 掉：

```
p(y|x) = ∫ p(y|τ, x) p(τ|x) dτ
```

这个 integral intractable，所以用 variational inference，引入 posterior `q_φ(τ|x,y)`。

**训练时**用 posterior——能看到 target y，知道"正确方向"是什么：
```
ε_t ~ q_φ(ε_t | u_t, y) = N(μ_φ(u_t, y), σ_φ²(u_t, y) I)
```

**Inference 时**用 prior——target 不可见，只能根据当前 state 猜方向：
```
ε_t ~ p_θ(ε_t | u_t) = N(μ_θ(u_t), σ_θ²(u_t) I)
```

训练目标就是标准 ELBO，但有工程实现：

1. **Truncated BPTT**：gradient 只通过每个 supervision step 的最后一次 transition 反传。这是 biased but memory-efficient 的 approximation。Appendix A.3 验证了 full ELBO 和 surrogate 都单调下降，说明这个 approximation 是有效的。

2. **KL balance** (coefficient 0.8)：Dreamer 系列的 trick，把 KL 项的 weight 分配到 prior 和 posterior 两边，防止 posterior 过早 collapse 到 prior。

3. **Deep supervision**：每个 supervision step 末尾都 decode 并施加 loss，不是只在最后 supervise。

### Width scaling：parallel 采样是新的 test-time compute axis

这是 paper 的一个重要 contribution。

现有 RRMs 只能 depth scaling：跑更多步。这是 sequential 的，有 latency bottleneck。

GRAM 打开了 width scaling：跑 N 次，每次 sample 不同的 noise，得到 N 条 trajectory，N 个候选答案。这是 parallel 的，compute 多了但 latency 不变。

选答案的两种方式：
1. **Majority voting**：选最 frequent 的
2. **LPRM (Latent Process Reward Model)**：训一个 value head `v_ψ(z_t)` 预测 trajectory 的 final accuracy，选 `v_ψ` 最高的

Figure 4 (left) 的数据很 striking：GRAM with N=20 samples at 16 iterations = 97.0% accuracy，TRM at 320 iterations = 90.5%。Compute budget 差不多，但 GRAM 通过 parallel 胜出。

### 这和 test-time compute scaling 的关系

Karpathy 你最近很关注 test-time compute。GRAM 的 width scaling 本质上是一种 latent space 的 search，和 o1/o3 的 autoregressive search 是不同 axis：

- **o1/o3**：autoregressive token-level search，在 token 空间探索不同 reasoning paths
- **GRAM**：latent recursive search，在 latent trajectory 空间探索不同 reasoning paths

GRAM 的优势是 compact：latent state 比 token sequence 紧凑得多，同样 compute 能探索更多 paths。劣势是 expressive power 受限于 latent dimension，不如 token space flexible。

### 实验结果讲讲

**Sudoku-Extreme**：
- GRAM 10M params: 97.0%
- TRM 7M params: 87.4%
- HRM 27M params: 55.0%
- Deepseek-R1 671B params: 0.0%

671B 的 LRM 在这个 task 上完全失败，因为这是 constraint propagation reasoning，pretrained capacity 不 transfer。这证明了 RRMs 路线的价值——不是所有 reasoning 都能靠 scale 解决。

**N-Queens 多解问题**：
- GRAM: 99.7% accuracy, 90.3% coverage (20 samples 找到 90.3% 的合法解)
- TRM: 66.8% accuracy, 36.1% coverage
- AR: 96.3% accuracy, 84.8% coverage

Deterministic recursive 模型 mode collapse 严重。Generative 模型 (AR, MDLM) coverage 高但 accuracy 不如 GRAM。GRAM 把 recursive refinement 的 precision 和 stochastic sampling 的 diversity 结合起来了。

**Unconditional generation**：
- MNIST：GRAM 256 步 FID 73.34，D3PM 1000 步 FID 74.03。GRAM 用 1/4 的步数超过。
- Sudoku：GRAM 16 步 99.05% validity，D3PM 1000 步 91.33%。GRAM 用 1/62 的步数超过。

这里有个很 elegant 的发现：把 input 换成空的，这套 conditional reasoning 框架自然变成 unconditional generative model。Reasoning 和 generation 在 GRAM 里是 unified 的。

### Ablation 的关键 insight

Table 3b 特别 informative：

1. **去掉 stochasticity (`N(μ, 0)`)**：Sudoku 和 N-Queens 都 0%。Deterministic guidance conditioned on target 会导致 severe overfitting，posterior 完全 collapse。

2. **去掉 guidance (`N(0, σ²I)`)**：Sudoku 保持 94.88%，但 N-Queens 崩到 50.27%。纯随机扰动在单解问题上够用（提供 exploration 防 overfitting），但多解问题上需要 structured guidance 来 navigate solution space。

3. **TRM 加 stochastic decoding 或 random init**：没用。说明 gain 不是来自 mere randomness，是来自 variational framework 的结构化 exploration。

### 可视化：GRAM 在 latent space 里怎么探索的

Appendix D.6 用 PCA 把 latent state 降到 2D，构造 loss landscape：

- **TRM**：一条 deterministic path，如果走进 suboptimal region (亮黄) 就出不来了
- **GRAM**：50 条 stochastic trajectories，有些 trapped 在 local minima，有些 reach global optimum (深蓝)

这直观展示了 parallel sampling 的价值：不是每条 trajectory 都成功，但 best-of-N 能把成功的选出来。

### 我的一些联想和疑问

**1. 和 MCTS 的关系**

GRAM 的 width sampling 某种意义上是 latent space 的 search。和 AlphaGo 的 MCTS 比较：
- MCTS：tree structure，有 branching factor，选择性 expand
- GRAM：parallel rollouts，没有 tree structure，每条 trajectory 独立

GRAM 没有 tree structure 是个 limitation。Tree search 能利用 partial trajectory 的 value 来 prune，GRAM 只能等整条 trajectory 跑完再 evaluate。能否给 GRAM 加 tree structure？比如在每个 transition 后用 LPRM 评估，prune 掉低 value 的 partial trajectory。

**2. 和 Continuous CoT 的关系**

Quiet-STaR、Coconut、Continuous CoT 这些 work 都是把 reasoning 移到 latent space。但它们仍然是 autoregressive 的——latent thought 是 sequence 的 extension。GRAM 是 recursive 的，shared transition function 反复 apply。

Recursive 的优势是 parameter efficient；autoregressive 的优势是 expressive。哪个更适合 reasoning？这可能取决于 task structure。Constraint satisfaction (Sudoku) 适合 recursive，因为可以反复 refine；open-ended generation 可能更适合 autoregressive。

**3. Stochastic guidance 学的是什么**

`μ_θ(u_t)` 学的是"在当前思考状态下，什么样的扰动最有可能导向正确答案"。这很像 RL 中的 policy——给定 state，输出 action。但 GRAM 是 amortized 的（一次 forward 算出来），不是 iterative planning。

这和 prompt engineering 里的 "let's think step by step" 有趣的对比：CoT 是让模型显式 explore，GRAM 是让模型 implicit explore。前者 interpretability 好，后者 efficiency 好。

**4. Training efficiency 是真 bottleneck**

Paper 自己承认 deep supervision 的 sequential nature 限制了 training efficiency。每个 supervision step 都要等前一个跑完，不能 parallelize。这是 RRMs 路线相对 Transformers 的 fundamental disadvantage。

如果要让 GRAM scale 到 LLM-scale，可能需要：
- 某种 form 的 parallel supervision（比如 multiple trajectory segments 并行训练）
- Distillation from autoregressive CoT（用 CoT 的 reasoning trace 来 supervise GRAM 的 latent trajectory）
- Curriculum learning（先训 short trajectory，再逐渐 extend）

**5. Posterior collapse 的风险**

训练时 posterior 能看到 target，这很 powerful 但也很 dangerous。如果 posterior 过度依赖 target，inference 时用 prior 就会 degrade。KL balance 是缓解，但根本问题还在。

能否设计一个完全 self-supervised 的训练方式？比如让 model 自己 generate trajectory，然后用 verifier 判断对错，再用 RL 更新？这就把 GRAM 和 RL 结合起来了，类似 Dreamer 的 imagination-based training。

### 总结：GRAM 的 contribution 在哪

1. **Probabilistic multi-trajectory recursion**：把 deterministic RRMs 推广到 stochastic，是自然且重要的 generalization
2. **Width-based inference scaling**：新的 test-time compute axis，和 depth scaling 互补
3. **Unified reasoning + generation**：conditional reasoning 和 unconditional generation 用同一个 framework
4. **Stochastic guidance 的设计**：additive residual + learned Gaussian，简洁有效

但这还是一个 direction-setting 的工作，离 LLM-scale reasoning 有距离。实验都在 small-scale synthetic tasks，training efficiency 是 bottleneck。不过它指出了一条有意思的路：reasoning 的未来可能不只是"生成更多 token"，而是"在 latent space 里 probabilistic 地探索"。

---

**Reference links**:
- GRAM website: https://ahn-ml.github.io/gram-website
- HRM paper: https://arxiv.org/abs/2506.21734
- TRM paper: https://arxiv.org/abs/2510.04871
- Looped Transformers: https://arxiv.org/abs/2311.12424
- Universal Transformers: https://arxiv.org/abs/1807.03819
- Dreamer V3: https://arxiv.org/abs/2301.04104
- VRNN: https://arxiv.org/abs/1506.02216
- Deep Kalman Filters: https://arxiv.org/abs/1511.05121
- ARC-AGI: https://github.com/fchollet/ARC-AGI
- Continuous CoT (Coconut): https://arxiv.org/abs/2412.06769
- Reasoning by Superposition: https://arxiv.org/abs/2505.12514

---

# GRAM (Generative Recursive reAsoning Models) 论文详解

## 1. 核心问题与 Motivation

这篇 paper 来自 KAIST、Mila、NYU 的团队 (Sungjin Ahn, Yoshua Bengio 等)，针对的是一个根本性问题：**future neural reasoning systems 应该如何实现 extended computation？**

当前的 reasoning scaling 主要走两条路：
- **Autoregressive sequence extension**：CoT [1]、ToT [2]、GoT [3] 等通过生成更多 token 来增加计算
- **Recursive Reasoning Models (RRMs)**：Universal Transformers [10]、Looped Transformers [7]、HRM [8]、TRM [9]，通过 shared transition function 反复 refine persistent latent state

RRMs 的 appeal 在于它 decouples reasoning depth from parameter scale 和 output length——一个 compact model 可以通过反复 apply shared transition functions 实现很多步 internal computation。

**但现有 RRMs 有一个致命问题：它们都是 deterministic 的**。给定相同 input 和 initialization，模型 follow 单条 latent trajectory，converge 到单个 prediction。这在 multi-solution 场景下会 mode collapse，也无法 explore alternatives。

GRAM 的核心 insight：把 recursive latent reasoning 变成 **stochastic latent trajectory** 的 probabilistic 多轨迹计算。

## 2. Architecture 详细解析

### 2.1 两层嵌套递归结构

GRAM 用一个 hierarchical latent state `z = (h, l)`，其中：
- **h (high-level)**：每 outer step 更新一次，carry abstract reasoning state，缓慢演化
- **l (low-level)**：每次 inner refinement 更新 K 次，carry fine-grained intermediate computation，快速 refine

公式 (6) 定义 low-level refinement：
$$l_{t,k} = f_L(h_{t-1}, l_{t,k-1}, e_x; \theta), \quad k=1,...,K$$

变量含义：
- `l_{t,k}`：第 t 个 transition、第 k 次 low-level refinement 后的 low-level state
- `l_{t,0} := l_{t-1}`：从上一个 transition 的 low-level state 出发
- `h_{t-1}`：high-level state 在整个 inner refinement 期间保持固定
- `e_x = f_enc(x; θ)`：input embedding，在每步都注入
- `f_L`：low-level transition function，是 [Attention + SwiGLU] × 2 layers 的 stack

公式 (7)-(9) 定义 high-level update：
$$u_t = f_H(h_{t-1}, l_t; \theta)$$
$$\epsilon_t \sim \mathcal{N}(\mu_\theta(u_t), \sigma_\theta^2(u_t) I)$$
$$h_t = u_t + \epsilon_t$$

变量含义：
- `u_t`：deterministic high-level proposal，由 `f_H` 计算
- `l_t := l_{t,K}`：refined low-level component
- `μ_θ(u_t)`：state-dependent mean，由 SwiGLU MLP 参数化，encodes 朝哪个方向 steer trajectory
- `σ_θ²(u_t)`：state-dependent variance，控制 exploration amount
- `ε_t`：**learnable stochastic guidance**，是 GRAM 区别于 prior RRMs 的关键

**注意 stochasticity 只在 high-level 注入**，low-level fully deterministic。设计直觉：高层的抽象 state 应该是 "decision point"，stochasticity 在这里 steer 整条 trajectory；low-level 是细节 refinement，不应该被 noise 干扰。

### 2.2 Supervision Step 与递归组织

整体 computation 按两层组织：
- **Supervision step**：从 `z_0` 出发，跑 T 次 transition 到 `z_T`，这是 decoder 被调用的最小单元
- **Outer loop**：`N_sup` 个 supervision steps 串联，前一个的 terminal state 作为下一个的 initial state

公式 (3) 的递归链：
$$z_0^{(1)} \to z_T^{(1)} = z_0^{(2)} \to \cdots \to z_T^{(N_{sup})}$$

变量含义：
- `z_t^{(n)}`：第 n 个 supervision step 的第 t 次 transition 的 latent state
- `z_0^{(1)}`：固定的初始 state (从 N(0,I) 采样一次，存到 checkpoint，永久固定)
- `z_0^{(n+1)} := z_T^{(n)}`：supervision steps 串联

**Intuition**：supervision step 是 deep supervision 的应用点，每个 step 末尾 decoder 都会 decode 并施加 loss，gradient 密集。这避免了只有 trajectory 末端才有 supervision signal 的稀疏问题。

### 2.3 Architecture Schematic 解析 (Figure 2)

Figure 2 展示了 single stochastic latent transition 的 hierarchical instantiation：

1. Low-level refinement block：`f_L` 被应用 K 次，每次都注入 `h + e_x`，refine `l`
2. High-level deterministic proposal：`f_H(h_{t-1}, l_t)` 输出 `u_t`
3. Stochastic guidance：从 Gaussian `N(μ_θ(u_t), σ_θ²(u_t) I)` 采样 `ε_t`
4. Residual perturbation：`h_t = u_t + ε_t`
5. 新的 `z_t = (h_t, l_t)`

这本质上是 **learned stochastic residual perturbation around deterministic update**。deterministic part 保留了 prior RRMs 的 refinement 能力，stochastic part 注入了 multi-hypothesis exploration。

### 2.4 为什么是 ε_t = u_t + ε_t 而不是 vae 风格的 reparam trick？

这种 additive 形式有一个 deep 的设计意图：
- `u_t` 是 deterministic refinement，gradients 直接 backprop，无阻碍
- `ε_t` 通过 reparameterized Gaussian 采样：`ε_t = μ_θ(u_t) + σ_θ(u_t) * ξ`，其中 `ξ ~ N(0, I)`
- 这种形式与 diffusion model 的 forward process 有结构相似性，但这里 noise 是 state-dependent 且 learned

## 3. Training: Amortized Variational Inference

### 3.1 Latent Variable Formulation

GRAM 作为 latent variable model，conditional likelihood 为：
$$p_\theta(y|x) = \int p_\theta(y|\tau, x) p_\theta(\tau|x) d\tau$$

变量：
- `τ = (z_0 → ... → z_{T_Total})`：完整 latent trajectory，`T_Total = T × N_sup`
- `p_θ(τ|x)`：prior，由 stochastic transitions 定义
- `p_θ(y|τ, x) = p_θ(y|z_{T_Total}, x)`：likelihood，只在 terminal state decode

直接 MLE intractable (marginalize over trajectories)，所以引入 variational posterior `q_φ(τ|x,y)`，优化 ELBO：

$$\log p_\theta(y|x) \geq \mathbb{E}_{q_\phi(\tau|x,y)}[\log p_\theta(y|\tau,x)] - KL(q_\phi(\tau|x,y) \| p_\theta(\tau|x))$$

### 3.2 Prior vs Posterior 的 Markov 结构

公式 (12) 把 prior 和 posterior 都写成 Markov chain：
$$p_\theta(\tau|x) = p(z_0) \prod_{t=1}^{T_{Total}} p_\theta(z_t|z_{t-1}, x)$$
$$q_\phi(\tau|x,y) = p(z_0) \prod_{t=1}^{T_{Total}} q_\phi(z_t|z_{t-1}, x, y)$$

**关键设计**：prior 和 posterior share 同一个 transition module！区别只在 noise distribution：
- Prior: `ε_t ~ p_θ(ε_t|u_t) = N(μ_θ(u_t), σ_θ²(u_t) I)`
- Posterior: `ε_t ~ q_φ(ε_t|u_t, y) = N(μ_φ(u_t, y), σ_φ²(u_t, y) I)`

Posterior 额外 condition on target y，相当于 inference 时"作弊"看到答案的方向。Train 时用 posterior sample，inference 时用 prior sample。这是 **amortized inference** 的经典设计，类似 VRNN [33]、Dreamer [37,38]。

### 3.3 Trajectory-level ELBO 的等价形式

由于 prior 和 posterior 共享 Markov 结构且 stochasticity 全在 `ε_{1:T_Total}`，可以把 trajectory distribution 等价表示在 noise space。又因为 decoder 只读 terminal state，完整 ELBO 简化为：

$$\mathcal{L}_{ELBO} = \mathbb{E}_{q_\phi}[\log p_\theta(y|z_{T_{Total}}, x)] - \sum_{t=1}^{T_{Total}} \mathbb{E}_{q_\phi(\epsilon_{<t}|x,y)} [KL(q_\phi(\epsilon_t|u_t,y) \| p_\theta(\epsilon_t|u_t))]$$

变量：
- 第一项：reconstruction log-likelihood
- 第二项：sum over all transitions 的 KL，每个 KL 的 expectation 要对 ancestral noise `ε_{<t}` 求，因为 `u_t = f_H(h_{t-1}, l_t)` 依赖 `h_{t-1}`，而 `h_{t-1}` 由历史 noise `ε_{<t}` 决定

**Intuition**：每个 transition step 都有一个 KL regularization，鼓励 posterior 不要偏离 prior 太远，同时 information bottleneck 防止 posterior collapse 到 deterministic path。

### 3.4 Truncated Surrogate Objective

实际训练中，gradient 只通过每个 supervision step 的**最后一次 transition** `z_{T-1}^{(n)} → z_T^{(n)}` 反传。这是 truncated BPTT [16,17] 在 recursive reasoning 上的应用。

公式 (14) 的 per-supervision-step surrogate：
$$\mathcal{L}_{GRAM}^{(n)} = \mathbb{E}_{q_\phi}[\log p_\theta(y|z_T^{(n)}, x)] - KL(q_\phi(\epsilon_T^{(n)}|u_T^{(n)}, y) \| p_\theta(\epsilon_T^{(n)}|u_T^{(n)}))$$

这是一个 **biased but memory-efficient** approximation。论文 Appendix A.3 做了 empirical validation (Figure 8)：在 Sudoku 和 N-Queens 上，full ELBO 和 surrogate 都单调下降，说明 surrogate 的 gradient update 确实驱动了 full variational bound 的改善。两条曲线的 gap 反映了 earlier transitions 的累积 KL，不是优化失败。

**关键 trick**：KL balance coefficient 0.8 [37,38] 防止 posterior collapse。这是 Dreamer 系列 world model 的标准技巧——把 KL 项的 weight 分配到 prior 和 posterior 两边，避免 posterior 过早 collapse 到 prior。

## 4. Inference-Time Scaling: Width vs Depth

### 4.1 Depth Scaling: ACT (Adaptive Computation Time)

Appendix A.1 描述了 ACT 的 Q-learning 形式（沿用 HRM [8]）。Halt head `q_ψ: R^D → R^2` 输出 `(q^halt, q^continue)`，作为 binary action 的 Q-values。

Training targets：
- `q̂_n^halt = 1[ŷ^(n) = y]`：当前 state decode 出来对不对
- `q̂_n^continue = max(q_{n+1}^halt, q_{n+1}^continue)`：bootstrap 的 continue value

Loss：
$$\mathcal{L}_{ACT} = \sum_{n=1}^{N_{sup}} [(q_n^{halt} - \hat{q}_n^{halt})^2 + (q_n^{continue} - \hat{q}_n^{continue})^2]$$

Inference：每个 supervision step 后 evaluate `q_ψ(h^(n))`，若 `q^halt > q^continue` 就 halt 返回 `ŷ^(n)`，否则继续，最多 `N_sup^max` 步。

**重要**：这个 loss 只更新 halt head，不反传到 recursive core。不同 parallel trajectories 可能 halt 在不同 depth，与 width scaling 互补。

### 4.2 Width Scaling: Parallel Trajectory Sampling

从 prior `p_θ(τ|x)` 采样 N 条 trajectories `{τ^(i)}_{i=1}^N`，每条 decode 出 `ŷ^(i) = f_dec(z_T^(i))`，然后用两种 selection：

1. **Majority voting**：选最 frequent prediction
2. **LPRM-guided best-of-N**：用 Latent Process Reward Model `v_ψ(z_t)` 预测 trajectory 的 final quality

LPRM 训练 (Appendix A.2)：
$$\mathcal{L}_{LPRM} = \sum_{t=1}^T (v_ψ(z_t) - r)^2$$

其中 `r ∈ [0,1]` 是 final prediction accuracy。Inference 时选 `v_ψ` 最高的 candidate。

**Intuition**：depth scaling 是 sequential 的，有 latency bottleneck；width scaling 是 parallel 的，可同时 explore 多条 stochastic paths。GRAM 把这两个 axis 都打开了。

## 5. Experiments 详细分析

### 5.1 Challenging Puzzle Tasks (Sudoku-Extreme, ARC-AGI)

Table 8 关键数据：

| Method | #Params | Sudoku | ARC-1 | ARC-2 |
|---|---|---|---|---|
| Deepseek-R1 | 671B | 0.0 | 15.8 | 1.3 |
| Gemini 3 Pro | N/A | - | 75.0 | 31.1 |
| Direct Pred | 27M | 0.0 | 21.0 | 0.0 |
| Looped TF | 7M | 61.3 | - | - |
| HRM | 27M | 55.0 | 40.3 | 5.0 |
| TRM | 7M | 87.4 | 44.6 | 7.8 |
| **GRAM** | **10M** | **97.0** | **52.0** | **11.1** |

**关键观察**：
1. 所有 LRM (包括 671B 的 Deepseek-R1) 在 Sudoku-Extreme 上都是 0%——这是 constraint propagation reasoning，pretrained capacity 不 transfer
2. GRAM 用 10M params 打败 27M 的 HRM，且在 ARC-2 上接近 2x 提升
3. Direct prediction 完全失败 (0%)，证明 recursive computation 是 essential 的

Figure 4 (left) 展示 inference-time scaling：GRAM with N=20 samples at 16 iterations 达到 97.0%，而 TRM at 320 iterations 只有 90.5%，comparable compute budget 但 GRAM 通过 parallel sampling 胜出。

### 5.2 Multi-Solution Tasks (N-Queens, Graph Coloring)

Table 1 关键数据 (N-Queens 8×8)：

| Method | Rec | Gen | Accuracy | Coverage |
|---|---|---|---|---|
| TRM | √ | × | 66.8 | 36.1 |
| AR | × | √ | 96.3 | 84.8 |
| MDLM | × | √ | 96.1 | 87.2 |
| **GRAM** | √ | √ | **99.7** | **90.3** |

Figure 4 (right) 揭示关键现象：随着 valid solutions 数量增加，deterministic recursive models (HRM, TRM) 的 accuracy 急剧下降；GRAM 保持 consistent performance。这是 **mode collapse in multi-solution landscapes** 的直接证据。

Graph Coloring 上更明显：GRAM 的 conflict edges 是 2.7 (8-vertex) 和 3.3 (10-vertex)，而 AR 是 19.0 和 61.3。Recursive refinement enables sharper constraint satisfaction——生成模型 sampling 拿到 diversity，但递归 refinement 拿到 precision，GRAM 把两者结合了。

### 5.3 Unconditional Generation (MNIST, Sudoku)

Table 2 (binarized MNIST)：

| Method | IS (↑) | FID (↓) |
|---|---|---|
| VAE | 1.70 | 86.28 |
| D3PM (1000 steps) | 1.86 | 74.03 |
| TRM (16 steps) | 1.00 | 303.29 |
| **GRAM 256 steps** | **2.04** | **73.34** |

**关键发现**：
1. TRM (deterministic) 的 FID 303.29，mode collapse 极其严重
2. GRAM 用 256 步就超过 D3PM 的 1000 步，且 inference-time 增加步骤单调改善 (training 只用 16 步，inference 用 256 步仍 improve)
3. 这是 recursive refinement 在 generative regime 的 transfer

Table 9 (Sudoku unconditional generation)：GRAM 99.05% validity，10.9M params，16 步；D3PM-Big 91.33%，55.1M params，1000 步。GRAM 用 1/5 的参数、1/62 的步数超越。

### 5.4 Ablation Studies (Table 3)

**(a) Architecture ablation** (cumulative addition)：

| Model variant | Sudoku | N-Queens |
|---|---|---|
| base (Looped TF) | 61.25 | 71.30 |
| + DS + HR (=HRM, TRM) | 55.00 / 87.40 | 80.70 / 72.90 |
| + SG | 65.64 | 86.30 |
| + DS + SG | 73.90 | 100.00 |
| + DS + HR + SG (=GRAM) | 93.96 | 99.69 |

**SG (stochastic guidance) 在每个 configuration 都有 consistent gain**，是 GRAM 的核心 contribution。Hierarchical recursion 的效果 task-dependent，但 SG 始终有效。

**(b) Mechanism ablation** (modify ε_t distribution)：

| Model variant | Sudoku | N-Queens |
|---|---|---|
| GRAM (ours) | 93.96 | 99.69 |
| w/o stochastic guidance (N(0, σ²I)) | 82.87 | 72.91 |
| stochasticity only (N(μ, 0)) | 0.00 | 0.00 |
| guide only | 0.00 | 0.00 |
| TRM w/ stochastic decoder | 82.87 | 71.66 |
| TRM w/ random init | 78.53 | 71.82 |

**关键 insight**：
1. 去掉 guidance (`N(0, σ²I)`) Sudoku 保持 94.88% 但 N-Queens 崩到 50.27%——stochasticity alone 能 enable diverse paths 但 structured guidance 在 multi-solution space 必需
2. 去掉 stochasticity (`N(μ, 0)`) 直接 0%——deterministic guidance conditioned on target 导致 severe overfitting
3. TRM 加 stochastic decoding 或 random init 都没改善——**gain 来自 variational framework，不是 mere randomness**

## 6. Latent Trajectory 可视化 (Appendix D.6)

Figure 18-19 用 PCA + K-D tree 构造 loss landscape，可视化 TRM vs GRAM 的 trajectory：
- **TRM**：单条 deterministic path，如果进入 suboptimal region 无法 escape
- **GRAM (50 samples)**：stochastic guidance 产生 diverse trajectories，有些 trapped 在 local minima (亮黄)，有些 reach global optimum (深蓝)

这是 parallel sampling 的可视化论证——不是所有 trajectory 都成功，但只要有一条 reach optimum，best-of-N 或 majority voting 就能选出来。

## 7. 与相关工作对比

### 7.1 vs Latent Reasoning (CoCoSo [4], Continuous CoT [5,6])

这些方法把 reasoning shift 到 continuous representation 但仍 organized around autoregressive sequence generation。GRAM 的区别在于：computation 的 organization 是 recursive（shared transition functions），不是 sequential token generation。

### 7.2 vs Probabilistic Latent State-Space Models (VRNN [33], SRNN [34], Dreamer [37,38])

GRAM shares latent state-space view 但 **reinterpret stochastic dynamics as computation rather than temporal observation modeling**。Dreamer 系列 model temporal dynamics of observations，GRAM model reasoning trajectories。ELBO、truncated BPTT、KL balance 这些技术是 borrowed 的，但应用对象完全不同。

### 7.3 vs Diffusion Models

Structural similarity：forward process 加 noise，reverse process 去 noise。但 GRAM 的 noise 是 state-dependent 且 learned，不是 fixed schedule；且 GRAM 的"denoising"是 recursive refinement，不是 Markov reverse process。GRAM 在 Sudoku generation 上用 16 步超过 D3PM 1000 步，因为递归 refinement 比 diffusion 的 Markov 假设更适合 structured constraint satisfaction。

## 8. Limitations 与 Future Directions

论文自己指出的 limitation：**sequential nature of deep supervision limits training efficiency compared to Transformers**。这阻碍了 GRAM scaling 到更大 foundation model。

但我认为还有几个值得思考的问题：
1. **Stochastic guidance 的 sample efficiency**：posterior 需要 condition on target，训练时如何避免过度依赖 posterior？KL balance 是一种缓解，但根本问题还在
2. **Multi-solution vs single-solution 的 trade-off**：很多 reasoning task (math, code) 只有一个正确答案，stochastic exploration 的价值在哪？可能在于"探索不同 solution strategy"而非"不同 solution"
3. **LPRM 的 calibration**：value head 预测 accuracy，但 reasoning task 中"看起来对"和"实际对"可能差距很大，特别是 hallucination 场景
4. **Hierarchical structure 的 necessity**：ablation 显示 hierarchical recursion 效果 task-dependent，是否 flat stochastic recursion 就够？可能取决于 task 的内在 multi-scale structure

## 9. Build Intuition: GRAM 在 cognitive science 视角下的位置

论文引用了 Kahneman [11] 的 "Thinking, fast and slow" 和 Bengio [12] 的 "consciousness prior"。可以这样理解：
- **System 1 (fast)**：low-level deterministic refinement `f_L`，快速 polish details
- **System 2 (slow)**：high-level stochastic transition `f_H + ε_t`，维护多个 hypotheses，做 strategic exploration
- **Consciousness prior**：high-level state `h` 是 conscious representation，应该 sparse 和 abstract，stochasticity 在这层允许"thought switching"

GRAM 的 hierarchical + stochastic design 在 cognitive architecture 上是有 grounding 的，不纯是 engineering trick。

## 10. 我的评价

这篇 paper 的 contribution 是 solid 的：
1. 把 deterministic RRMs 推广到 probabilistic multi-trajectory，是自然且重要的 generalization
2. Stochastic guidance 的设计 (additive residual + learned Gaussian) 简洁有效
3. Width-based inference scaling 是新的 axis，与 depth scaling 互补
4. 实验设计 controlled，ablation 透彻，与 deterministic baselines 公平比较

但有几个 caveat：
1. 实验都在 small-scale synthetic tasks，离 LLM-scale reasoning 还有距离
2. Training efficiency 是 paper 自己承认的 limitation
3. Stochastic guidance 在 single-solution task (Sudoku) 上 gain 来自 robustness，在 multi-solution task (N-Queens) 上 gain 来自 diversity，两个机制其实不同，paper 没有完全分开讨论

总体来看，这是一个 direction-setting 的工作，为 recursive reasoning 引入 probabilistic 视角，值得 follow。

**Reference links**:
- Paper website: https://ahn-ml.github.io/gram-website
- HRM repository: https://github.com/sapientinc/HRM
- TRM repository: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- Looped Transformers: https://arxiv.org/abs/2311.12424
- Universal Transformers: https://arxiv.org/abs/1807.03819
- Dreamer V3: https://arxiv.org/abs/2301.04104
- VRNN: https://arxiv.org/abs/1506.02216
- ARC-AGI: https://github.com/fchollet/ARC-AGI
