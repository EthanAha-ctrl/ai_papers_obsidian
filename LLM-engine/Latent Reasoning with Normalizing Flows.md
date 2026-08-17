---
source_pdf: Latent Reasoning with Normalizing Flows.pdf
paper_sha256: 23c5ac8dc4a61c0627e4532b21e907e321e0deace99356239e3c84e67a02c625
processed_at: '2026-08-05T12:14:03-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲NF-CoT

Andrej，好，我换个频道，像我们coffee chat那样聊这篇paper。

---

## 这篇paper到底在干啥

一句话：**让LLM在脑子里想事情，不一定要用文字把每一步说出来。**

现在的CoT（chain-of-thought）逼着模型把reasoning过程用自然语言写出来。你想啊，你做一道数学题，脑子里其实先闪过一些模糊的intuition、一些partially formed的想法，然后才组织成文字写下来。现在的LLM没有这个"模糊思考"的阶段，必须每个step都verbalize成token才能往下走。

这篇paper说：我给模型一个**continuous thinking space**，让它在64个continuous vectors里"想"，想完了再输出答案文字。这64个vectors相当于模型的"inner monologue"，但不low bandwidth、不用被natural language的syntax绑架。

参考CoT原始paper: https://arxiv.org/abs/2201.11903

---

## 为什么之前的方法不够好

这个latent reasoning的想法不新。之前有几条路：

### Coconut（Hao et al. 2024）
直接把hidden states feed back给自己。问题：**deterministic**。每次想的东西一模一样，没有"多种思路"的概念。你做数学题可能脑子里try多条路径，Coconut做不到。

### LaDiR（Kang et al. 2025）
用diffusion model在latent space里denoise。好处是stochastic了，但问题：**慢**（要30步denoising），**likelihood算不出来**（diffusion是implicit density），所以没法直接做policy gradient RL。

### NF-CoT（本文）
用normalizing flow代替diffusion。Flow的好处：
1. **一次forward就能sample**（不用30步迭代）
2. **likelihood能精确计算**（可以算 $\log p$ 的exact value）
3. **跟LLM的causal mask天然兼容**（autoregressive flow的Jacobian是triangular）

这三条让latent reasoning第一次拥有了跟explicit CoT完全一样的"sampling、scoring、RL"interface。

参考:
- Coconut: https://arxiv.org/abs/2412.06769
- LaDiR: https://arxiv.org/abs/2510.04573

---

## Normalizing Flow是啥，为啥适合这个场景

### 人话版flow

想象你有一个简单的分布（standard Gaussian，像个球形的noise cloud），你想把它变成一个复杂的分布（比如"所有合理的reasoning trajectories"）。

Flow就是一堆**可逆的变换**，像捏橡皮泥一样，把Gaussian球捏成你想要的shape。因为每一步都可逆，所以：
- **正向**：从Gaussian采样 → 捏 → 得到一个reasoning trajectory（generation）
- **反向**：把一个reasoning trajectory → 反捏 → 回到Gaussian → 算出它的概率（likelihood）

### 数学版

$$p_\theta(y | c) = p(z) \left| \det J_{f_\theta^{-1}}(y; c) \right|$$

变量：
- $y$: 你要建模的东西（这里是一个continuous thought vector）
- $z$: standard Gaussian noise
- $f_\theta^{-1}$: 把 $y$ 映射回 $z$ 的函数
- $J$: 这个函数的Jacobian矩阵
- $c$: conditioning context（prompt + 之前的thoughts）
- $\det$: determinant，衡量变换对volume的缩放程度

Jacobian是triangular的话， $\det$ 就是diagonal元素的乘积，巨好算。Autoregressive flow天然是triangular的，因为每个position只看之前的positions，跟causal Transformer完美对齐。

参考flow基础:
- NICE: https://arxiv.org/abs/1410.8516
- Real NVP: https://arxiv.org/abs/1605.08803
- MAF: https://arxiv.org/abs/1705.07057

---

## 这篇paper最subtle的设计：两个空间

这是我觉得最clever的部分。

### 问题

之前LaDiR的做法：先用VAE encoder把text CoT压缩成continuous vectors $e_{1:K}$，然后用diffusion学 $p(e|q)$。但 $e$ 空间是**为text compression优化的**，不是为autoregressive generation优化的。直接在这个空间里做AR generation很别扭。

### NF-CoT的解法

引入两个空间：

**Space $e$**: VAE encoder输出的，good for compression, bad for AR generation
**Space $u$**: 专门为AR generation设计的，good for sampling和scoring

两个空间通过一个**可逆变换** $F_\theta$ 连接：

$$u_{1:K} = F_\theta(e_{1:K}; q)$$

因为 $F_\theta$ 可逆，$u$ 和 $e$ **信息完全等价**，只是representation不同。

### 为什么这个trick important

你可以这样理解：VAE encoder给你的是"压缩格式"，比如ZIP文件。你很难直接在ZIP上做autoregressive generation。NF-CoT相当于把ZIP解压成一个"虽然格式不同但信息一样"的representation，在这个representation上做AR generation就很自然。

### Eq 3.1: $u$空间的density

$$p_\theta(u_{1:K} | q) = \prod_{i=1}^{K} \mathcal{N}\big(u_i; \mu_\theta(q, u_{<i}), \text{diag}(\sigma_\theta^2(q, u_{<i}))\big)$$

变量：
- $u_i$: 第$i$个continuous thought（dimension $D=2560$）
- $K=64$: thought sequence长度
- $\mu_\theta, \sigma_\theta$: 由LLM backbone + NF head预测的mean和std
- $u_{<i}$: 之前的所有thoughts（causal conditioning）

每个 $u_i$ 都是一个Gaussian，mean和std由prompt和之前的thoughts决定。这就是一个**autoregressive Gaussian model**，跟LLM的token prediction本质上一样，只是输出continuous vector而不是discrete token。

### Eq 3.2: 回到 $e$ 空间的likelihood

$$\log p_\theta(e_{1:K} | q) = \log p_\theta(u_{1:K} | q) + \log \left| \det J_{F_\theta}(e_{1:K}; q) \right|$$

变量：
- 第一项：$u$空间的AR Gaussian likelihood
- 第二项：shallow flow blocks的Jacobian determinant（triangular所以好算）

这个公式说的是：因为 $F_\theta$ 可逆，对 $u$ 建模等价于对 $e$ 建模，只需要加上一个Jacobian correction term。

---

## 架构长啥样

### 整体结构

```
Input: [ prompt ; <BOT> ; u_1, u_2, ..., u_64 ; answer_tokens ]
                                ↑                  ↑
                           NF head输出         LM head输出
                          (μ, σ) for each     token logits
                          continuous thought
```

一个causal sequence，一个backbone（Qwen3-8B），两个head：
- **NF head**: 在continuous-thought positions输出Gaussian的 $(\mu, \sigma)$
- **LM head**: 在answer positions输出token logits

这就是paper说的"unified" design。整个sequence一次forward跑完，两个loss同时算。

### Shallow flow blocks

5个MetaBlocks，实现 $e \to u$ 的变换。每个block是causal affine flow，Jacobian triangular，log-det好算。

Identity initialization：一开始 $F_\theta \approx I$，所以 $u \approx e$。随着训练慢慢学到有意义的变换。

**Key**: 这些shallow blocks只在训练时用！Inference时直接在 $u$-space采样，bypass掉shallow blocks。因为inference时不需要从 $e$ 出发，直接从Gaussian采样 → AR生成 $u$ → decode answer。

### 跟Dual-path的对比

Dual-path是论文的ablation variant：NF和CE用两条不同的path，跑两次backbone。Unified只用一次。Table 1的结果：

- Dual-path: avg 65.2
- Unified: avg 68.8

Unified好3.6个点。原因：dual-path的latent distribution、conditioning representation、inference trajectory三者mismatch。Unified把它们统一到一个causal stream里。

参考STARFlow的MetaBlock设计: https://arxiv.org/abs/2511.20462

---

## 训练怎么做

### Loss function

$$\mathcal{L} = \lambda_{\text{flow}} \mathcal{L}_{\text{flow}} + \lambda_{\text{text}} \mathcal{L}_{\text{text}}$$

两个term：

1. **Flow loss**: $-\log p_\theta(e_{1:K} | q)$，让flow学会生成正确的continuous thoughts
2. **Text loss**: $-\sum_j \log p_\theta(x_j | q, u_{1:K}, x_{<j})$，让LM head学会从thoughts decode出正确答案

$\lambda_{\text{flow}} = \lambda_{\text{text}} = 1.0$，简单weighted sum。

### Two-stage training

**Stage 1**: Freeze backbone，只训练shallow blocks和projector
- 100K samples, 1 epoch, LR $10^{-4}$
- 目的：让NF branch先学会"说话"，别给backbone发garbage gradient

**Stage 2**: Unfreeze全部，joint training
- 2 epochs, LR $5 \times 10^{-5}$
- 全参数更新

### Warm-up为什么重要

Table 4的ablation：去掉Stage 1
- HumanEval: 84.4 → 81.5 (-2.9)
- HumanEval+: 78.7 → 75.5 (-3.2)

Appendix C.1给了diagnostic：
- 有warm-up：Stage 2开始时 $L_{NF} \approx -0.42$，shallow blocks已经学到东西
- 没warm-up：Stage 2开始时 $L_{NF} \approx 0.47$（≈random），gradient norm大2倍（1.96 vs 0.96）

**人话**：pretrained LLM的representation很fragile。如果你直接把一个random init的NF branch接上去做joint training，那个branch会发出乱七八糟的gradient，把backbone的pretrained knowledge搞坏。先freeze backbone训练branch，让branch calibrate好，再unfreeze，backbone收到的就是meaningful signal了。

这个insight我觉得对任何"在pretrained model上加新module"的工作都适用。

### Dequantization noise

训练时给 $e$ 加小noise：$\tilde{e} = e + \epsilon, \epsilon \sim \mathcal{N}(0, 0.3^2 I)$

因为 $e$ 是VAE posterior mean，deterministic target。Flow对deterministic target训练容易overfit到degenerate solution。加noise变成stochastic target，让flow学smooth density。TarFlow的standard practice。

---

## 推理怎么做

### 流程

1. 给prompt $q$
2. 从Gaussian采样 $z_1 \sim \mathcal{N}(0, T_z^2 I)$
3. LLM forward到position 1，NF head输出 $(\mu_1, \sigma_1)$
4. $u_1 = \mu_1 + \sigma_1 \cdot z_1$
5. 把 $u_1$ project到embedding dim，append到sequence
6. 重复2-5，采64个thoughts
7. 切换到LM head，继续生成answer tokens
8. **KV-cache全程复用**

### 为什么比LaDiR快

LaDiR：30步denoising，每步要full backbone forward
NF-CoT：64步AR sampling，每步incremental forward（KV-cache reuse）

Table 2数字：
- LaDiR latent generation: 468.2s
- NF-CoT latent generation: 173.5s
- **2.7× faster**

FLOPs：
- LaDiR: 49.3T/sample
- NF-CoT: 19.9T/sample
- **2.48× cheaper**

### Compression rate

64个latent tokens平均encode 385个text tokens的CoT。**6× compression**。

人话：text CoT里大量token是syntactic sugar、verbose explanation、"let me think step by step"这种filler。Continuous representation把这些压掉，只留essence。当然这个compression是lossy的（VAE encoder学的），但paper证明这个lossiness不影响reasoning quality。

---

## RL怎么搞

### 为什么能做RL

因为flow有exact likelihood，能算 $\log \pi_\theta(\text{thoughts, answer} | q)$。Policy gradient直接可用。

LaDiR做不到，因为diffusion的likelihood intractable。

### Policy factorization

$$\log \pi_\theta(\tilde{u}, \hat{x} | q) = \underbrace{\log p_\theta(\tilde{u}_{1:K} | q)}_{\text{thought policy}} + \underbrace{\log p_\theta(\hat{x}_{1:\hat{N}} | q, \tilde{u}_{1:K})}_{\text{answer policy}}$$

变量：
- $\tilde{u}$: sampled thoughts
- $\hat{x}$: sampled answer
- 第一项：NF head输出的Gaussian likelihood
- 第二项：LM head输出的token likelihood

**两个都是policy action**，一起被RL优化。

### GRPO setup

Group size 8，每个prompt采8个rollout，用execution reward（unit test pass/fail）算advantage：

$$\hat{A}_i = \frac{R_i - \mu}{\sigma + \varepsilon}$$

变量：
- $R_i$: 第$i$个rollout的reward（0或1）
- $\mu, \sigma$: group内的mean和std
- $\varepsilon$: small constant for numerical stability

PPO surrogate + KL penalty，150 steps，LR $3 \times 10^{-6}$。非常lightweight。

### 结果

Table 1：
- Supervised NF-CoT: 68.8
- + RL: 70.1 (+1.3)

所有5个benchmark都有提升，modest but consistent。

### Figure 4: 最important的empirical finding

**Token-space GRPO**: 提升pass@1，但pass@k在large k时saturate甚至低于base model。RL把probability mass集中到少数solution modes，**diversity collapse**。

**NF-CoT latent RL**: 提升pass@1，**同时pass@k的scaling trend保留**。在所有k范围内都高于supervised checkpoint。

### 为什么latent RL不collapse diversity

我的理解：

Token RL更新discrete action distribution。每个token的categorical distribution被sharpen，容易collapse到greedy decoding。

Latent RL更新Gaussian的 $(\mu, \sigma)$。即使 $\sigma$ 被narrow了，sampling时还是 $u = \mu + \sigma \cdot z, z \sim \mathcal{N}(0, I)$。这个 $z$ 是**exogenous noise**，policy gradient没法eliminate它。所以diversity被structural地preserve了。

这给了我一个deep insight：**continuous relaxation for RL不只是optimization trick，它改变了diversity-preserving的mechanism**。

参考DeepSeek-R1 GRPO: https://nature.com/articles/s41586-025-08808-9

---

## 实验结果说了啥

### Table 1主结果

Qwen3-8B-Base（avg 55.8）：

| Method | Avg | Δ |
|--------|-----|---|
| Standard SFT | 59.9 | +4.1 |
| LaDiR (diffusion) | 61.6 | +5.8 |
| NF-CoT (Unified) | 68.8 | +13.0 |
| NF-CoT + RL | 70.1 | +14.3 |

几个takeaway：

1. **LaVAE崩了**（avg 32.7, -23.0）：用 $L_2$ VAE代替flow，完全崩盘。说明exact likelihood不是decorative，是essential。$L_2$ regression到mean target丢失distribution information。

2. **NF-CoT vs LaDiR**: +7.1。同样的continuous CoT target，flow比diffusion好。Exact likelihood training让flow学到更precise的distribution shape。

3. **vs OlympicCoder**: 68.8 vs 68.5。OlympicCoder是strong open-source coding model，NF-CoT match它。说明latent reasoning的gain不是来自更多CoT data，而是来自modeling reasoning trajectory distribution。

### Figure 3: Pass@k scaling

MBPP+：
- Base model: pass@1 ≈ 60, pass@128 = 72
- NF-CoT: pass@1 = 72（已match base的pass@128！）, pass@128 = 87.5

**人话**：NF-CoT采样1次的coverage等于base model采样128次的coverage。64个latent thoughts的stochastic sampling提供了极强的diversity。

### Appendix C.2: Structural diversity

用AST 2-gram算structural similarity：

$$s(a, b) = \frac{1}{2} \cos(\phi_{2g}(a), \phi_{2g}(b)) + \frac{1}{2} \mathbb{I}[h(a) = h(b)]$$

变量：
- $\phi_{2g}$: AST node-type 2-gram的count vector
- $h$: canonicalized AST的hash

结果：
- Base model: mean intra-prompt similarity = 0.548
- NF-CoT: 0.469（**14%相对降低**）

Base model在temperature sampling下lexical diverse，但structurally peaked。NF-CoT的latent sampling在token generation之前就inject了diversity，所以能explore不同的algorithmic strategy。

---

## 最美的实验：Latent Perturbation

### Setup

采样一个base trajectory $\tilde{u}$，加Gaussian noise：

$$u_\sigma = \tilde{u} + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

$\sigma$ 从0扫到3.0，看生成的code怎么变。

### Figure 5的结果

| $\sigma$ | Cosine sim | Pass@1 | Exact match | Pass agreement |
|-----------|-----------|--------|-------------|----------------|
| 0 | 1.000 | 86.0 | 0.973 | 1.000 |
| 3.0 | 0.116 | 83.6 | 0.278 | 0.888 |

**人话**：你把latent trajectory扰到几乎orthogonal（cosine sim 0.116），生成的text几乎完全不同（exact match从97%跌到28%），但**functional correctness几乎不变**（pass@1只跌2.4%）。

### 这个实验说明了什么

1. **Latent space是smooth manifold**：correct solutions在latent space里形成一个smooth manifold，perturbation在manifold内移动但不出manifold

2. **Latent variables控制"how"不控制"whether"**：perturbation改变implementation style（rolling DP vs tabulation vs memoization），改变"怎么solve"，不改变"是否solve"

3. **Form vs Function decoupling**：text form是brittle的，underlying function是robust的。这跟discrete token space很不同——token space里换一个token可能catastrophic

这个实验让我想到deep learning里的manifold hypothesis：real-world data在high-dimensional space里集中在low-dimensional manifold上。这里的是：correct reasoning trajectories在latent space里也form manifold。

---

## Appendix A的例子：同一题多种解法

### HumanEval/63: fibfib

同一个problem，不同latent sample：

1. **Sample 49**: rolling three-state DP
   ```python
   a, b, c = 0, 0, 1
   for _ in range(3, n+1):
       next_value = a + b + c
       a, b, c = b, c, next_value
   ```

2. **Sample 3**: explicit tabulation
   ```python
   fibfib_values = [0, 0, 1]
   for i in range(3, n+1):
       fibfib_values.append(fibfib_values[-1] + fibfib_values[-2] + fibfib_values[-3])
   ```

3. **Sample 35**: recursive memoization
   ```python
   memo = {0: 0, 1: 0, 2: 1}
   def helper(x):
       if x in memo: return memo[x]
       memo[x] = helper(x-1) + helper(x-2) + helper(x-3)
       return memo[x]
   ```

Decoded latent CoT对应地讨论"compressing to fixed state"、"tabulation/list storage"、"cache and recursion"。

**这是什么的evidence**：NF-CoT的latent space不是存储单个hidden solution，而是**distribution over reasoning trajectories**。不同sample steer到不同的algorithmic region。

### HumanEval/89: encrypt

- Sample 1: alphabet string + `.index()` lookup
- Sample 23: `ord`/`chr` + modulo arithmetic

同一个Caesar cipher，两种完全不同的implementation strategy。

---

## 跟其他方法的关系

### vs Coconut

Coconut是deterministic hidden state recycling。NF-CoT是stochastic continuous thought sampling。NF-CoT多了：trajectory distribution、probabilistic sampling、RL compatibility。

### vs LaDiR

LaDiR用diffusion，NF-CoT用flow。
- LaDiR: 30步denoising，慢，likelihood intractable
- NF-CoT: 1-pass AR sampling，快，exact likelihood

同样的continuous CoT target，不同的modeling approach。NF-CoT好7.1个点。

### vs Soft Thinking

Soft Thinking用Gumbel-Softmax把token选择relax成embedding mixture。还是在embedding space里。NF-CoT在独立的continuous thought space建模，有explicit density。

### vs TarFlow/STARFlow

TarFlow和STARFlow是scalable flow的foundational work，用于image/video generation。NF-CoT把这些技术adapt到LLM内部的reasoning。这是"老技术新应用"的nice example。

参考:
- Soft Thinking: https://arxiv.org/abs/2505.15778
- TarFlow: https://arxiv.org/abs/2412.06329
- STARFlow: https://arxiv.org/abs/2511.20462

---

## 我的personal take

### 让我excited的

1. **Likelihood-based latent reasoning是clean interface**。Sampling、scoring、RL都有。这可能是未来reasoning research的foundation。

2. **Autoregressive flow + LLM的unification**。一个backbone，不同head，处理continuous和discrete。这个design pattern可能extend到其他modality。

3. **Latent RL preserves diversity**。Token RL collapse diversity是known problem。Latent RL的noise是structural的，policy gradient没法eliminate。这可能是解决RL diversity problem的一条路。

4. **Perturbation实验的beauty**。Smooth manifold of correct solutions。Reasoning space有topological structure，不是brittle symbolic manipulation。

### 让我skeptic的

1. **Code generation是verifier-friendly的domain**。Unit test pass/fail是clean reward。Math reasoning、commonsense reasoning没有这么clean的verifier。这个方法能generalize吗？

2. **VAE encoder是frozen的**。Continuous CoT target的quality决定上限。如果VAE的compression是lossy的，flow学到的distribution也是distorted的。能不能end-to-end train VAE encoder？

3. **64 latent tokens是magic number**。Paper没有systematic study。这个数字的sensitivity如何？能不能adaptive？

4. **Decoded latent CoT是qualitative的**。我们看到了correspondence between latent和algorithmic strategy，但这不是causal evidence。需要更rigorous的interpretability work。

5. **Fixed latent budget**。简单问题64个tokens浪费，复杂问题可能不够。能否学习variable-length latent？或者hierarchical latent（高层abstraction + 低层implementation）？

### 更deep的思考

这篇paper让我思考一个更大的问题：**reasoning到底应该用什么representation**？

Text是human-readable但verbose、low-bandwidth。Continuous vectors是compact但opaque。理想的representation可能介于两者之间：有structure（能composition）、compact（高bandwidth）、interpretable（能probe）。

NF-CoT选择的是continuous vectors + flow density。这是一个reasonable choice，但可能不是最终的。未来的direction可能包括：
- **Structured continuous representation**: 比如graph-valued latent，能capture reasoning的compositional structure
- **Hierarchical latent**: 不同abstraction level的thought
- **Mixed discrete-continuous**: 高层abstraction用discrete（选择strategy），低层execution用continuous

参考NF-CoT project page: https://nf-cot.vercel.app

---

## 一句话总结

**NF-CoT用normalizing flow给LLM装了一个"continuous thinking module"，这个module能像explicit CoT一样sample和score，但在compact continuous space里做，更快更省，还能做RL不collapse diversity。**

核心insight：**exact likelihood是latent reasoning的"灵魂"**，它把sampling、scoring、RL三个interface统一到一起，让continuous thoughts获得跟text tokens一样的first-class citizen status。

希望这个"人话版"build了你的intuition，Andrej。如果你想dive deeper into哪个specific part（比如STARFlow的MetaBlock internals、GRPO的implementation细节、或者VAE encoder的设计选择），我可以再展开。

---

# NF-CoT: Latent Reasoning with Normalizing Flows 深度讲解

Andrej，这篇paper解决的核心问题是：**如何让continuous CoT保留explicit CoT的所有"好性质"——autoregressive sampling、exact likelihood、KV-cache compatibility——同时在compact continuous space中做reasoning**。作者的选择是用normalizing flow代替diffusion来建模latent thoughts，这个选择背后有深刻的数学动机。我会从几个层次build你的intuition。

---

## 1. 问题动机：为什么Explicit CoT不完美，但Latent CoT很难

### Explicit CoT的概率视角

Paper在Eq 2.1给了一个key formulation：

$$p_\theta(x_{1:N} | q) = \sum_{d_{1:L}} p_\theta(d_{1:L} | q) \, p_\theta(x_{1:N} | q, d_{1:L})$$

变量含义：
- $q$: prompt（问题）
- $x_{1:N}$: answer序列，长度$N$
- $d_{1:L}$: discrete CoT trace，长度$L$，作为中间reasoning变量
- $\theta$: 模型参数

这个formulation的insight是：CoT本质上是在input-output之间引入一个latent variable $d$，把direct prediction $p(x|q)$变成marginalized prediction $p(x|q) = \sum_d p(d|q) p(x|q,d)$。这个latent variable framework让CoT的effectiveness可以解释为：reasoning path是sampled的，不是deterministic的。

但explicit CoT的问题：text是verbose、low-information-density的medium。每个reasoning step必须verbalize成discrete token才能继续，即使underlying update是semantic、uncertain或者partially formed的。这是bottleneck。

### Latent CoT的四个paradigms（Figure 1）

Paper把现有方法分成四类：

1. **Explicit CoT**: discrete text tokens，verbose但有完整LM interface
2. **Coconut** (Hao et al., 2024): deterministic hidden states fed back，但没有trajectory distribution
3. **LaDiR** (Kang et al., 2025): diffusion-based iterative denoising，stochastic但需要多步denoising，likelihood intractable
4. **NF-CoT (本文)**: AR-sampled continuous thoughts，既有stochasticity又有tractable likelihood

参考：
- Coconut paper: https://arxiv.org/abs/2412.06769
- LaDiR paper: https://arxiv.org/abs/2510.04573

---

## 2. Normalizing Flow的数学基础：为什么选NF

### Flow的核心公式

Normalizing flow通过invertible network把data $y$映射到simple base variable $z \sim \mathcal{N}(0, I)$。对于conditional flow（context $c$）：

$$z = f_\theta^{-1}(y; c), \quad p_\theta(y | c) = p(z) \left| \det J_{f_\theta^{-1}}(y; c) \right|$$

变量含义：
- $y$: data point（这里是被建模的continuous thought）
- $z$: base variable，standard Gaussian
- $f_\theta$: invertible transformation，参数化by $\theta$
- $f_\theta^{-1}$: inverse transformation（forward pass of flow是inverse direction）
- $J_{f_\theta^{-1}}$: Jacobian of inverse transformation
- $c$: conditioning context（这里是prompt $q$ 和previous thoughts）

这个公式的核心beauty：**exact likelihood可以直接计算**。只要Jacobian容易计算，就能得到 $\log p(y|c)$ 的精确值。这跟VAE（lower bound）和diffusion（approximate via ELBO或score matching）本质不同。

### Autoregressive Flow的特殊性

Autoregressive flow (Kingma et al., 2016; Papamakarios et al., 2017)的关键性质：每个position只使用context和previous positions来transform，这导致Jacobian是**triangular**。

对triangular matrix，$\det J = \prod_i J_{ii}$，所以：

$$\log |\det J| = \sum_i \log |J_{ii}|$$

这跟causal Transformer的left-to-right mask天然兼容。这是为什么TarFlow (Zhai et al., 2024)和STARFlow (Gu et al., 2026a)能用Transformer blocks实现high-dimensional flow。

参考：
- TarFlow: https://arxiv.org/abs/2412.06329
- MAF: https://arxiv.org/abs/1705.07057
- IAF: https://arxiv.org/abs/1606.04934
- Real NVP: https://arxiv.org/abs/1605.08803
- NICE: https://arxiv.org/abs/1410.8516
- Glow: https://arxiv.org/abs/1807.03039

---

## 3. NF-CoT的核心Formulation：两个空间的设计

### 为什么需要两个空间？

这是paper最subtle的设计。作者引入两个continuous thought space：

**Space 1: $e_{1:K}$（VAE encoder space）**
- 由frozen VAE encoder产生：$q_\phi(e_{1:K} | d_{1:L})$ 的posterior mean
- 优化目标：text compression
- 问题：prompt-conditioned distribution难以autoregressive modeling

**Space 2: $u_{1:K}$（LLM-facing thought space）**
- 通过invertible transformation连接：$u_{1:K} = F_\theta(e_{1:K}; q)$
- 设计为easy to sample和score autoregressively
- 与 $e$ 信息等价

### Eq 3.1: $u$空间的causal Gaussian density

$$p_\theta(u_{1:K} | q) = \prod_{i=1}^{K} \mathcal{N}\big(u_i; \mu_\theta(q, u_{<i}), \text{diag}(\sigma_\theta^2(q, u_{<i}))\big)$$

变量含义：
- $u_i$: 第$i$个continuous thought token，dimension $D$
- $K$: latent sequence length（paper用$K=64$）
- $\mu_\theta(q, u_{<i})$: causal mean function，由LLM backbone + NF head计算
- $\sigma_\theta(q, u_{<i})$: causal std function，diagonal covariance
- $u_{<i}$: 之前所有的thought tokens（causal conditioning）

这个factorization是autoregressive的，每个$u_i$只依赖prompt和之前的thoughts，与LLM的causal mask完全对齐。

### Eq 3.2: likelihood through reparameterization

由于 $F_\theta$ 是invertible的，$u$ 和 $e$ 信息等价。original continuous CoT target的likelihood可以通过 $u$-space写出：

$$\log p_\theta(e_{1:K} | q) = \log p_\theta(u_{1:K} | q) + \log \left| \det J_{F_\theta}(e_{1:K}; q) \right|$$

变量含义：
- $J_{F_\theta}$: Jacobian of $F_\theta$（shallow flow blocks）
- 第一项：$u$-space的autoregressive Gaussian likelihood
- 第二项：Jacobian determinant，由shallow flow blocks的triangular structure容易计算

**Key insight**: 因为 $F_\theta$ 是invertible的，modeling $u$ 等价于modeling $e$，但 $u$ 更容易autoregressive采样。这是"换空间但保信息"的trick。

---

## 4. 架构详解：Shallow + Deep Flow的双层结构

### Figure 2的架构解析

NF-CoT包含两个flow component：

**Component 1: Shallow flow blocks（$F_\theta$）**
- 实现 $e_{1:K} \mapsto u_{1:K}$ 的invertible map
- 5个MetaBlocks（alternating Identity/Flip permutations）
- Identity initialization: $F_\theta \approx I$（初始时 $u \approx e$）
- Causal affine flow → triangular Jacobian → tractable log-det
- 在训练时使用，inference时bypass

**Component 2: Deep autoregressive flow（LLM backbone）**
- Continuous thoughts $u_{1:K}$ 投影到token embedding dimension
- 在同一个causal stream中与answer tokens一起处理
- 在continuous-thought positions: NF head输出 $(\mu_\theta, \sigma_\theta)$
- 在answer positions: standard LM head输出token logits

### 关键的Unified设计

```
[ prompt ; <BOT> ; flow_proj(u_1:K) ; answer_tokens ]
```

整个sequence是单一causal stream：
- Position 0 到 $T_p$: prompt
- Position $T_p$ 到 $T_p + K$: continuous thoughts（NF head负责）
- Position $T_p + K$ 到 $T_p + K + T_a$: answer tokens（LM head负责）

**为什么Unified比Dual-path好**：Dual-path需要跑两次backbone（一次NF，一次CE），且latent distribution、latent representation for conditioning、inference trajectory三者mismatch。Unified把所有放在一个causal sequence里，消除mismatch。

参考STARFlow架构: https://arxiv.org/abs/2511.20462

---

## 5. 训练：Unified Likelihood Objective

### Eq 3.3: 联合训练目标

$$\mathcal{L}_{\text{sup}} = \lambda_{\text{flow}} \mathcal{L}_{\text{flow}} + \lambda_{\text{text}} \mathcal{L}_{\text{text}}$$

其中：

**Flow term**:
$$\mathcal{L}_{\text{flow}} = -\log p_\theta(e_{1:K} | q) = -\log p_\theta(u_{1:K} | q) - \log|\det J_{F_\theta}(e_{1:K}; q)|$$

**Text term**:
$$\mathcal{L}_{\text{text}} = -\sum_{j=1}^{N} \log p_\theta(x_j | q, u_{1:K}, x_{<j})$$

变量含义：
- $\lambda_{\text{flow}}, \lambda_{\text{text}}$: loss weights（paper中都是1.0）
- $N$: answer sequence length
- $x_j$: 第$j$个answer token

两个term都是同一个causal LLM下的likelihood，只是不同positions用不同head。

### Two-stage Curriculum

**Stage 1**: Freeze LLM backbone，只训练shallow flow blocks和continuous-thought projection layers
- 100K samples, 1 epoch
- LR = $1 \times 10^{-4}$, 100 warmup steps
- 目的：align $e \to u$ reparameterization with frozen LLM space

**Stage 2**: Unfreeze全部参数，joint training
- 2 epochs
- LR = $5 \times 10^{-5}$, 200 warmup steps
- 完整objective

### Table 4的ablation insight

去掉Stage 1（warm-up）：
- HumanEval: 84.4 → 81.5 (-2.9)
- HumanEval+: 78.7 → 75.5 (-3.2)
- LiveCodeBench v6: 23.1 → 21.4 (-1.7)

**为什么warm-up有用**？从Appendix C.1看：
- Stage 2 only的初始 $L_{NF} \approx 0.47$（接近random Gaussian baseline）
- Default curriculum的初始 $L_{NF} \approx -0.42$（shallow blocks已经学到nontrivial density）
- Stage 2 only的初始gradient norm 1.96 vs 0.96
- 没有warm-up，pretrained backbone立即被poorly calibrated的NF branch污染

从Table 7的backbone drift看：
- Stage 2 only在final transformer layer的rel-L2 drift大18%
- 这层直接feed NF head和LM head，最敏感

这个ablation给我一个deep insight：**pretrained LLM的representation很fragile，random initialization的auxiliary branch会破坏它**。Warm-up本质上是"先让NF branch学会说话，再让backbone听它"。

### Dequantization noise的trick

$$\tilde{e}_{1:K} = e_{1:K} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_{dq}^2 I), \quad \sigma_{dq} = 0.3$$

为什么有用？因为 $e_{1:K}$ 是VAE encoder的posterior mean，是deterministic target。Flow对deterministic target训练时容易overfit到degenerate solution。加noise变成stochastic target，让flow学到smooth density。这是TarFlow的standard practice。

---

## 6. 推理：Single-pass Autoregressive Sampling

### Inference流程

**Key difference from training**: inference时**不跑shallow flow blocks** $F_\theta$。直接在 $u$-space采样：

1. 给定prompt $q$
2. 从Eq 3.1的autoregressive density采样 $\tilde{u}_{1:K}$:
   - 对每个 $i$ 从 $1$ 到 $K$:
     - LLM forward到position $i$，hidden state过NF head得到 $(\mu_i, \sigma_i)$
     - 采样 $z_i \sim \mathcal{N}(0, T_z^2 I)$
     - $\tilde{u}_i = \mu_i + \sigma_i \cdot z_i$
     - 投影 $\tilde{u}_i$ 到token embedding dim，append到sequence
3. 切换到LM head，继续生成answer tokens
4. **KV-cache全程复用**

变量含义：
- $T_z$: NF sampling temperature（unified model用0.9）
- $z_i$: 第$i$个latent position的Gaussian noise

### 为什么这比LaDiR快很多？

LaDiR需要30个denoising steps在latent trajectory上iterative refinement。每个step都要跑一次backbone forward。

NF-CoT是single left-to-right pass，64个continuous thoughts顺序采样，每个position只需要一次forward（带KV-cache）。

从Table 2看：
- LaDiR latent generation: 468.2s
- NF-CoT (Unified) latent generation: 173.5s
- 加速 2.70×

从Table 3的training throughput：
- LaDiR: 6.45 samples/s, 1.03K tokens/s, 1.50e20 FLOPs
- NF-CoT (Unified): 18.4 samples/s, 5.88K tokens/s, 2.25e19 FLOPs
- Sample throughput 2.85×, FLOPs少6.66×

### Latent compression rate

Paper提到：64个latent tokens encode平均385个text tokens的CoT，compression rate约6.0×。这个数字给了intuition：continuous representation比text dense约6倍。

---

## 7. RL Refinement in Latent Space

### Eq 3.4: Policy likelihood factorization

$$\log \pi_\theta(\tilde{u}, \hat{x} | q) = \log p_\theta(\tilde{u}_{1:K} | q) + \log p_\theta(\hat{x}_{1:\hat{N}} | q, \tilde{u}_{1:K})$$

变量含义：
- $\tilde{u}_{1:K}$: sampled continuous thought trajectory
- $\hat{x}_{1:\hat{N}}$: sampled answer tokens
- $\pi_\theta$: joint policy

这个factorization的关键：**continuous thoughts和answer tokens都是policy actions**，可以一起优化。

### Eq 3.5: GRPO policy gradient

$$\nabla_\theta J_{RL} = \mathbb{E}\left[ A(q, \tilde{u}, \hat{x}) \nabla_\theta \log \pi_\theta(\tilde{u}, \hat{x} | q) \right]$$

变量含义：
- $A(q, \tilde{u}, \hat{x})$: group-normalized advantage
- $R(q, \hat{x})$: reward（这里用unit-test pass/fail）

### Appendix B.5的细节

Latent log-prob的具体计算（Eq B.2）：

$$\log p_\theta(\tilde{x}_t | p, \tilde{x}_{<t}) = -\frac{1}{2} \|z_t\|_2^2 - \sum_{d=1}^{D} \log |x_{a,d}^t|$$

其中：
$$z_t = \frac{\tilde{x}_t - x_b^t}{x_a^t}$$

变量含义：
- $x_a^t, x_b^t$: NF head预测的affine参数（scale, shift）
- $z_t$: recovered Gaussian noise
- $D$: latent dimension（2560）

GRPO advantage（Eq B.3）：

$$\hat{A}_i = \frac{R_i - \mu}{\sigma + \varepsilon_\sigma}$$

PPO surrogate（Eq B.8）：

$$\mathcal{L}_\theta^{lat} = -\mathbb{E}_i \left[ \min(r_i^{lat} \hat{A}_i, \bar{r}_i^{lat} \hat{A}_i) \right] + \beta_{KL} K_i^{lat}$$

其中 $r_i^{lat} = \exp(\text{clip}(\Delta_i^{lat}, -20, 20))$ 是likelihood ratio。

### Figure 4的key insight: RL不collapse diversity

这是paper最重要的empirical finding之一：

**Token-space GRPO**: 
- 提升pass@1
- 但pass@k在large k时saturate，甚至低于base model
- 说明RL把probability mass集中到少数solution modes

**NF-CoT + latent RL**:
- 提升pass@1
- pass@k的upward scaling trend保留
- 在所有k范围内都高于supervised NF-CoT

为什么？我的interpretation：token-space的RL在离散空间更新，容易collapse到greedy solution。但latent space的RL更新的是continuous trajectory distribution，stochasticity来自 $z \sim \mathcal{N}(0, \tau_z^2 I)$ 的noise source。这个noise是structural的，policy gradient可以refine $(\mu, \sigma)$ 但不会eliminate noise source。

参考GRPO/DeepSeek-R1: https://nature.com/articles/s41586-025-08808-9

---

## 8. 实验结果深度分析

### Table 1: Main Results

最关键的数字（Qwen3-8B-Base backbone，base avg 55.8）：

| Method | Avg | Δ |
|--------|-----|---|
| Standard SFT | 59.9 | +4.1 |
| Soft Thinking | 61.6 | +5.8 |
| TaH+ | 61.6 | +5.8 |
| LaVAE | 32.7 | -23.0 |
| LaDiR | 61.6 | +5.9 |
| NF-CoT (Dual-Path) | 65.2 | +9.4 |
| NF-CoT (Unified) | 68.8 | +13.0 |
| NF-CoT + RL | 70.1 | +14.3 |

几个观察：

1. **LaVAE崩了**：用 $L_2$ objective的VAE代替flow，avg从55.8暴跌到32.7。这说明flow的exact likelihood不是decorative，是essential。$L_2$ regression到mean target丢失了distribution information。

2. **NF-CoT (Unified) vs LaDiR**: 68.8 vs 61.6, +7.1。这个gap的来源？LaDiR用diffusion（iterative denoising, implicit density），NF-CoT用flow（exact likelihood, single-pass）。同样的continuous CoT target，但modeling approach不同。Exact likelihood training让flow学到更precise的trajectory distribution。

3. **vs OlympicCoder**: 68.8 vs 68.5, +0.3。OlympicCoder是strong open-source coding model，NF-CoT用Qwen3-8B-Base能匹配它。

4. **RL的增益**: 68.8 → 70.1, +1.3。Modest但consistent across所有benchmarks。这个增益来自150 steps的GRPO，very lightweight。

### Figure 3: Pass@k Scaling

MBPP+:
- Base model: pass@1 = 60.5（不是表1的数字，这个是scaling run）, pass@128 = 72.0
- NF-CoT: pass@1 = 72.1（已match base的pass@128!）, pass@128 = 87.5

HumanEval+:
- Base model: 60+到~80
- NF-CoT: 78.3 → 97.5 (+19.2)

**Key insight**: NF-CoT的pass@1已经等于base的pass@128。这意味着64个latent tokens的stochastic sampling提供了非常强的diversity，单次采样就覆盖了base model需要128次采样才能达到的coverage。

### Table 2: Inference Efficiency

HumanEval, 16 candidates per problem:

| Method | Latent (s) | Decode (s) | Total (s) | Samples/s | FLOPs/sample |
|--------|-----------|-----------|----------|-----------|--------------|
| NF-CoT (Unified) | 173.5 | 152.1 | 325.6 | 8.06 | 19.9T |
| NF-CoT (Dual-Path) | 232.3 | 147.7 | 380.0 | 6.90 | 21.6T |
| LaDiR | 468.2 | 157.1 | 625.3 | 4.20 | 49.3T |

Decode时间差不多（147-157s），差异在latent generation。LaDiR的30步denoising每个step都要full backbone forward，NF-CoT的64步AR sampling每个step只需要incremental forward（带KV-cache）。

FLOPs/sample: NF-CoT (Unified) 19.9T vs LaDiR 49.3T，2.48× cheaper。这个数字给了compute-wise的intuition。

---

## 9. Latent Perturbation Robustness: 最美的实验

### Eq 5.1: Perturbation protocol

$$u_\sigma = \tilde{u} + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

变量含义：
- $\tilde{u}$: base continuous-CoT trajectory
- $\sigma$: perturbation strength，从0到3.0
- $\epsilon$: standard Gaussian noise

### Figure 5的发现

随着 $\sigma$ 从0到3.0：

| Metric | σ=0 | σ=3.0 | 趋势 |
|--------|-----|-------|------|
| Cosine similarity | 1.000 | 0.116 | 急剧下降 |
| Pass@1 | 86.0 | 83.6 | 几乎不变 |
| Pairwise pass agreement | 1.000 | 0.888 | 缓慢下降 |
| Exact-text match | 0.973 | 0.278 | 急剧下降 |

**Deep insight**: perturbation让generated text几乎完全不同（exact match从97%降到28%），但functional correctness几乎不变（pass@1只降2.4%）。这说明：

1. **Latent space是locally smooth的**：小的perturbation不破坏solution的correctness
2. **Latent space是distributed control**：latent variables控制的是"哪个solution strategy"而不是"是否solve problem"
3. **Form vs Function decoupling**：text form是brittle的，但underlying function是robust的

这个实验让我联想到deep learning里的manifold hypothesis：correct solutions form a low-dimensional manifold in latent space，perturbation在manifold内移动但不出manifold。

### Canonical-solution log-prob的非单调性

Paper提到canonical-solution log-probability有非单调trend：moderate noise raises it, large noise degrades it。作者解释：original trajectory commit到一个valid solution mode，可能differ from canonical implementation。Moderate noise让model explore到更接近canonical的mode。这给了我intuition：**latent space中存在multiple modes，对应不同的implementation strategies**。

---

## 10. Appendix A的Qualitative Examples: 多种implementation strategy

### HumanEval/63: fibfib

同一个problem，不同的latent sample产生三种不同的implementation：

1. **Sample 49**: rolling three-state DP（只用 a, b, c 三个变量）
2. **Sample 3**: explicit tabulation（用 list append）
3. **Sample 35**: recursive memoization（用 nested helper + memo dict）

Decoded latent CoT对应地讨论"compressing to fixed state"、"tabulation/list storage"、"cache and recursion"。

**这是什么的evidence**？NF-CoT的latent space不是存储单个hidden solution，而是distribution over reasoning trajectories。不同sample steer到不同的algorithmic region。

### HumanEval/154: cycpattern_check

- Sample 0: direct rotation loop with early return
- Sample 16: precompute rotations + `any()` function

### HumanEval/89: encrypt

- Sample 1: alphabet-index lookup with `.index()`
- Sample 23: ASCII modulo arithmetic with `ord`/`chr`

### Appendix C.2: Structural Diversity

定义structure-aware similarity：

$$s(a, b) = \frac{1}{2} \cos(\phi_{2g}(a), \phi_{2g}(b)) + \frac{1}{2} \mathbb{I}[h(a) = h(b)]$$

变量含义：
- $\phi_{2g}(\cdot)$: parent-child AST-node-type 2-gram的count vector
- $h(\cdot)$: canonicalized AST的hash（α-renamed, docstring stripped）

结果：
- Qwen3-8B-Base: mean intra-prompt similarity = 0.548
- NF-CoT: mean intra-prompt similarity = 0.469
- 相对降低约14%

**这个度量的insight**：token-level diversity metric会被paraphrase、variable name、formatting迷惑。AST 2-gram捕捉control flow structure（loop、conditional、call等），canonical hash捕捉true structural identity。

NF-CoT在HumanEval/124 (is_valid_date)上improvement最大（0.156）：base model collapse到split-and-branch implementation，NF-CoT sample到regex-based、exception-handling-based、date-library-based等多种strategy。

---

## 11. 与Related Work的对比

### vs Coconut (Hao et al., 2024)

Coconut把hidden states fed back autoregressively，但：
- Deterministic（没有trajectory distribution）
- 无法支持RL（没有stochastic policy）
- 无法支持probabilistic sampling

NF-CoT通过NF提供explicit distribution over trajectories，解决了这些限制。

### vs LaDiR (Kang et al., 2025)

LaDiR用diffusion在VAE latent space上denoise：
- Iterative denoising（30 steps）→ 慢
- Implicit density（无exact likelihood）→ 难以做policy gradient
- 与LLM的causal interface不兼容

NF-CoT用autoregressive flow：
- Single pass → 快
- Exact likelihood → 直接policy gradient
- 与LLM causal stream完全兼容

### vs Soft Thinking (Zhang et al., 2026b)

Soft Thinking用Gumbel-Softmax把token选择relax成soft embedding mixture：
- 仍是embedding space的mixture
- 没有explicit trajectory density

NF-CoT在独立的continuous thought space建模，有explicit likelihood。

### vs TarFlow/STARFlow

TarFlow (Zhai et al., 2024)和STARFlow (Gu et al., 2026a)是scalable normalizing flow的foundational work，用于image和video generation。NF-CoT把这些技术adapt到LLM内部的reasoning modeling。

参考：
- Soft Thinking: https://arxiv.org/abs/2505.15778
- TaH+: https://arxiv.org/abs/2511.08577
- Dream (Diffusion LM): https://arxiv.org/abs/2508.15487
- LLaDA: https://arxiv.org/abs/2502.09992
- Ouro: https://arxiv.org/abs/2510.25741

---

## 12. 给你的Intuition总结

Andrej，让我build几个core intuition：

### Intuition 1: Likelihood是 latent reasoning的"灵魂"

LaVAE（用 $L_2$ VAE代替flow）的崩盘（-23%）说明：latent reasoning不是简单的regression to mean thought，而是需要modeling **distribution over reasoning trajectories**。Exact likelihood training让flow学到这个distribution的shape，而不只是中心点。

### Intuition 2: Invertible space transformation的power

$e \to u$ 的invertible transformation是这篇paper最subtle的trick。$e$空间optimized for compression，但difficult to autoregress。$u$空间designed for autoregressive sampling，但通过invertible map保信息。这种"换空间不丢信息"的设计让flow能end-to-end训练exact likelihood。

### Intuition 3: Shared causal stream的unification

Unified design把continuous thoughts和answer tokens放在同一个causal sequence里，用同一个backbone，不同的head。这消除了dual-path的mismatch。这个design philosophy让我想到multimodal models：不同modality share backbone，不同head处理不同output。

### Intuition 4: Latent RL preserves diversity的机制

为什么latent RL不collapse diversity而token RL会？我的理解：token RL更新的是discrete action distribution，容易sharpen到greedy。Latent RL更新的是 $(\mu, \sigma)$ 的Gaussian参数，stochasticity来自 $z \sim \mathcal{N}(0, I)$ 的exogenous noise。Policy gradient可以shift mean和narrow std，但noise source是structural的。这给了latent space的"continuous relaxation"一个deep的justification。

### Intuition 5: Smooth manifold of correct solutions

Latent perturbation实验告诉我：correct solutions form a smooth manifold in latent space。Perturbation在manifold内移动，改变form但保持function。这跟传统的discrete token space很不同——token space的perturbation通常是catastrophic的（换一个token可能改变整个semantic）。

### Intuition 6: Compression rate = 6×的含义

64个latent tokens encode平均385个text tokens的CoT。这个6×compression rate让我思考：text reasoning的information density到底有多低？如果explicit CoT中大部分token是syntactic sugar和verbose explanation，那latent representation确实可以更efficient。但这个compression不是lossless的——是VAE encoder学的lossy compression。

### Intuition 7: Warm-up作为representation alignment

Stage 1的frozen-backbone warm-up本质上是"先align NF branch的语言，再让backbone学习新语言"。没有warm-up，random init的NF branch会给backbone garbage gradient，破坏pretrained representation。这个insight可能对任何"在pretrained LLM上加新branch"的工作都有意义：先冻结backbone训练新branch，再unfreeze全部。

### Intuition 8: NF vs Diffusion的本质差异

Diffusion是implicit density model，需要iterative denoising。NF是explicit density model，single pass采样。在reasoning这个context下，NF的优势：
- Exact likelihood → 直接policy gradient
- Single pass → 推理快
- Causal structure → 与LLM兼容

但NF的disadvantage：
- Invertibility constraint限制architecture expressiveness
- Memory成本（需要存invertible transformation的中间状态）

这篇paper的contribution之一是证明：在reasoning这个特定domain，NF的优势outweigh它的限制。

---

## 13. Limitations和Open Questions

Paper的limitations section提到：
- 只在code generation验证
- Fixed-length VAE-encoded trajectories继承explicit-CoT的biases
- Fixed latent budget可能不适合所有问题
- RL依赖unit-test reward，generalization到没有verifier的domain是open
- Latent不是human-readable，decoded CoTs只是qualitative probe

我想补充几个open questions：

1. **Latent length adaptivity**: 64个tokens对simple problem可能浪费，对complex problem可能不够。能否学习variable-length latent？

2. **Cross-domain transfer**: Code generation有unit-test verifier。Math reasoning、commonsense reasoning怎么定义reward？能否用process reward model？

3. **Latent interpretation**: 我们看到了decoded latent CoTs对应algorithmic strategy，但这是VAE decoder的artifact还是latent space真的organized by algorithm？能否做更系统的interpretability analysis？

4. **Compositionality**: 不同problem的latent space是否share structure？能否做latent arithmetic（类似word embedding的king - man + woman = queen）？

5. **Scaling laws**: 64 latent tokens是hand-tuned还是scaling law指导的？如果scale到256、1024，性能怎么变？compute trade-off如何？

6. **Hierarchical reasoning**: 64个flat latent tokens是否capture hierarchical reasoning structure？能否设计hierarchical latent space（高层abstraction + 低层implementation）？

参考NF-CoT project page: https://nf-cot.vercel.app

---

## 14. 我的personal take

这篇paper让我excited的几点：

1. **Likelihood-based latent reasoning是一个clean interface**：它把explicit CoT的所有好处（sampling、scoring、RL）带到continuous space。这可能是未来reasoning research的foundation。

2. **Autoregressive flow + LLM的unification是elegant的**：同一个backbone，不同的head，处理continuous和discrete variables。这个design pattern可能extend到其他modality（vision、audio）。

3. **Latent RL preserve diversity是empirically important的**：token RL collapse diversity是known problem。这篇paper提供了partial solution——在latent space做RL，noise source是structural的。

4. **Perturbation实验的beauty**：smooth manifold of correct solutions是deep concept。这暗示reasoning space有topological structure，不是brittle的symbolic manipulation。

但同时我也有些skepticism：

1. **Code generation是verifier-friendly的domain**：unit-test pass/fail是clean reward。这个方法在open-ended reasoning上的效果需要验证。

2. **VAE encoder是frozen的，可能limiting**：continuous CoT target的quality决定上限。如果VAE encoder的compression是lossy的，那flow学到的distribution也是distorted的。

3. **64 latent tokens是magic number**：paper没有systematic study of latent length effect。这个design choice的sensitivity如何？

4. **Decoded latent CoT是qualitative的**：我们看到了correspondence between latent和algorithmic strategy，但这不是causal evidence。需要更rigorous的interpretability work。

总的来说，这篇paper是一个strong step toward likelihood-based latent reasoning。它把normalizing flow这个"old"技术带进LLM reasoning这个"hot"领域，demonstrate了exact likelihood的power。我认为这个direction很有promise，特别是latent RL preserve diversity的finding，可能影响未来的reasoning RL research。

---

## 参考链接汇总

**核心paper**:
- NF-CoT project page: https://nf-cot.vercel.app
- TarFlow: https://arxiv.org/abs/2412.06329
- STARFlow: https://arxiv.org/abs/2511.20462
- Coconut: https://arxiv.org/abs/2412.06769
- LaDiR: https://arxiv.org/abs/2510.04573

**Normalizing Flow foundations**:
- NICE: https://arxiv.org/abs/1410.8516
- Real NVP: https://arxiv.org/abs/1605.08803
- Glow: https://arxiv.org/abs/1807.03039
- MAF: https://arxiv.org/abs/1705.07057
- IAF: https://arxiv.org/abs/1606.04934
- Rezende & Mohamed: https://arxiv.org/abs/1505.05770

**Latent reasoning related**:
- Soft Thinking: https://arxiv.org/abs/2505.15778
- TaH+: https://arxiv.org/abs/2511.08577
- Ouro: https://arxiv.org/abs/2510.25741
- Token Assorted: https://arxiv.org/abs/2502.03275

**Diffusion LMs (baselines)**:
- Dream: https://arxiv.org/abs/2508.15487
- LLaDA: https://arxiv.org/abs/2502.09992
- Diffu-Coder: https://arxiv.org/abs/2506.20639

**Benchmarks**:
- HumanEval: https://arxiv.org/abs/2107.03374
- MBPP: https://arxiv.org/abs/2108.07732
- EvalPlus: https://arxiv.org/abs/2304.07274
- LiveCodeBench: https://arxiv.org/abs/2403.07974

**RL**:
- DeepSeek-R1 (GRPO): https://nature.com/articles/s41586-025-08808-9
- AceCoder: https://arxiv.org/abs/2412.04203

希望这个深入讲解build了你的intuition，Andrej。如果你对哪个specific part想更深入讨论（比如STARFlow的MetaBlock细节、GRPO的implementation、或者VAE encoder的设计），我可以再展开。
