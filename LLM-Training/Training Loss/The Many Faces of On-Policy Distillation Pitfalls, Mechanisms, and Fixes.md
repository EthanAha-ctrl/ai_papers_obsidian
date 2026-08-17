---
source_pdf: The Many Faces of On-Policy Distillation Pitfalls, Mechanisms, and Fixes.pdf
paper_sha256: eabf16f19b2b2dc26e8f6e3bf0ddb0f141b0741299c7057d5d8ff36b2bbc509f
processed_at: '2026-08-12T14:31:49-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：OPD / OPSD 到底在搞什么

## 一句话概括

OPD 让 student 在自己的 rollout 上学，teacher 给每一步打分；OPSD 是同一个 student 自己给自己打分，但 teacher 视角多看一张"答案小抄"。这套机制在某些场景下 work 得很好，某些场景下会崩——paper 把为什么会崩讲清楚了。

参考：https://arxiv.org/abs/2506.22193 , https://thinkingmachines.ai/blog/on-policy-distillation

---

## 这套东西为什么吸引人

普通的 SFT 是 off-policy：拿别人写的 response 喂给 student 学。问题在于训练时 student 看到的 prefix 是 teacher 写的，但推理时 student 要面对自己写的 prefix。两边的 state distribution 不一致，student 学完了在自己的 distribution 下表现差。

OPD 直接解决这个：让 student 自己 rollout，teacher 在 student 自己生成的 trajectory 上给 token-level supervision。student 训练时见到的 prefix 和推理时见到的 prefix 一致。这个 idea 听起来就让人兴奋，因为从第一性原理上对。

但实测发现：math reasoning 上 OPD 经常崩，alignment / system prompt internalization 上又很猛。这篇文章就是在解释这个 split。

---

## 失败机制 1：Teacher 被拽进 student 的死胡同

### 直觉

想象你是一个数学很好的 tutor（teacher），学生（student）在黑板上写解题过程，你每一步都要给建议。问题是学生已经走错了一步，比如把 $x^2 + 1 = 0$ 写成了 $x^2 - 1 = 0$ 然后继续往下推。现在轮到你给下一步建议——你不会顺着学生的错误方向继续，你会说"等等，回头看看"或者"但是这里好像有问题"。

于是在 OPD 里，teacher 给 "wait", "but", "let me reconsider" 这些 revision tokens 高概率。student 收到的信号是"别继续这条路"——但 student 已经在这条路上了，所以它学到的是"我应该反复写 wait 和 but"，最终退化成冗长、含糊、原地打转的 reasoning。

### 实测数据

paper 在 GPQA-Diamond 上做了一个非常直观的实验：Qwen3-1.7B 作 student，Qwen3-14B 作 teacher。先让 teacher 独立解题，accuracy 62.12%；然后随机截断 student 的 trajectory，让 teacher 从截断点继续，accuracy 跌到 45.96%。绝对下降 16 个点，32 道题原本 teacher 能做对、被 student prefix 拖死。

Transition matrix 里：40 题 原本对的变错，只有 8 题 原本错的变对。这种 asymmetry 说明 student prefix 不是 random 干扰，它是**有方向性的误导**——student 把 teacher 拽进了一个 teacher 自己不会去的 reasoning state。

### 跟 RLVR 的根本区别

RLVR 的 supervision 来自最终 reward signal（答案对不对），它不依赖 teacher 对 student prefix 的理解。即使 student 走偏，只要最终能纠回来得到正确答案，reward 就是正的。

OPD 是 dense token-level supervision，每一步都依赖 teacher 在当前 prefix 下的判断。一旦 prefix 扭曲了 teacher state，整个 supervision signal 都会被污染。这是 OPD 比 RLVR 更脆弱的根本原因。

---

## 失败机制 2：TopK Reverse-KL 的 +1 Trap

### 直觉

完整的 reverse KL 是：

$$D_{\mathrm{KL}}(\pi_\theta \| \pi_T) = \sum_v \pi_\theta(v)\log\frac{\pi_\theta(v)}{\pi_T(v)}$$

变量解释：
- $v$：vocabulary 里每个 token
- $\pi_\theta(v)$：student 给 token $v$ 的概率
- $\pi_T(v)$：teacher 给 token $v$ 的概率

求梯度后会出现一个 $+1$ 项：

$$\nabla_\theta D_{\mathrm{KL}} = \sum_v \pi_\theta(v)\left[\log\frac{\pi_\theta(v)}{\pi_T(v)} + \mathbf{1}\right]\nabla_\theta \log \pi_\theta(v)$$

完整 vocab 下这个 $+1$ 项贡献为零，因为：

$$\sum_v \pi_\theta(v)\nabla_\theta \log \pi_\theta(v) = \nabla_\theta \sum_v \pi_\theta(v) = \nabla_\theta 1 = 0$$

直觉：所有 token 的概率加起来等于 1（归一化），所以概率的梯度和为零。

但工程上没人用完整 vocab——vocabulary 几万 token，算不起。大家用 TopK 截断，只对 student 或 teacher 的 TopK tokens 算 KL。问题是 TopK 子集 $\sum_{v \in S_K} \pi_\theta(v) \neq 1$，那个恒等式失效，$+1$ 项保留下来：

$$\sum_{v \in S_K} \pi_\theta(v)\nabla_\theta \log \pi_\theta(v) = \nabla_\theta \sum_{v \in S_K} \pi_\theta(v) \neq 0$$

这看起来只是个常数，但它改变了 token 的提升判据。原本要 promote 一个 token，只需要 $\pi_T(v) > \pi_\theta(v)$（teacher 比学生更喜欢这个 token）。现在因为 $+1$ 项的存在，要 promote 需要：

$$\log\frac{\pi_\theta(v)}{\pi_T(v)} + 1 < 0 \Rightarrow \pi_T(v) > e \cdot \pi_\theta(v)$$

提升阈值被抬高 $e \approx 2.7$ 倍。teacher 偏好但 gap 不够大的 token 会被压制。student 被推向 teacher 分布之外的低概率 continuation，最终 collapse。

### 实测

Qwen3-1.7B + Qwen3-8B teacher，OpenThoughts，unnormalized Top20 reverse KL：
- Step 0 → 700：response length 暴涨，"wait"/"but" revision tokens 开始泛滥
- Step 700 → 1000：彻底退化成 "maybe maybe maybe maybe..." 重复，accuracy 跌到接近 0
- Repeat ratio 接近 1.0，teacher-student vocab overlap 崩溃

非常典型的"训练早期看着不错，后期突然雪崩"模式。这种崩塌在 K=5、K=20 都出现，单纯增大 K 救不回来。

### 修复方案

**Stop-gradient TopK**——把 log-prob 项 stop 梯度，让 weighting 只起 advantage 作用：

$$\mathcal{L}_{\mathrm{SG\text{-}TopK}} = -\sum_{v \in S_K} \pi_\theta(v)\left[\log \pi_T(v) - \mathrm{stopgrad}(\log \pi_\theta(v))\right]$$

stopgrad 阻止 $\log \pi_\theta$ 反传，那个 $+1$ 项就消失了。判据恢复成正常的 $\pi_T > \pi_\theta$。

**Renormalized TopK**——在 TopK set 内重新归一化：

$$\bar{\pi}_\theta(v) = \frac{\pi_\theta(v)}{\sum_{u \in S_K} \pi_\theta(u)}, \quad \bar{\pi}_T(v) = \frac{\pi_T(v)}{\sum_{u \in S_K} \pi_T(u)}$$

归一化后 $\sum_{v \in S_K} \bar{\pi}_\theta(v) = 1$，恒等式恢复，$+1$ 项消失。代价是只匹配 TopK 内的相对概率，丢掉了 TopK 之外的概率 mass 信息，不再忠实近似 full-vocab KL。

**Sampled-token PG**——干脆只用 student sample 出的那个 token，把 teacher-student log-prob gap 当作 advantage：

$$\mathcal{L}_{\mathrm{PG}} = -\mathbb{E}_{y \sim \pi_\theta}\sum_t \mathrm{stopgrad}(\log \pi_T(y_t) - \log \pi_\theta(y_t)) \log \pi_\theta(y_t)$$

这就是 PPO 里 KL regularization 的形式，天然没有 TopK bias 问题。代价：只用 1 个 sample token 的信号，比 TopK 的 dense supervision 稀疏。

实测三种修复效果相当，都稳定。这告诉我们：**correct the biased gradient 比选哪种 loss 更重要**。

---

## 失败机制 3：OPSD 学到的是"几何平均"，会抹平 instance-specific 信息

### 直觉

OPSD 里 teacher 是 student 自己 + PI（答案小抄）。student 训练时看不到 PI，要把 PI-conditioned behavior 压进参数。优化目标是：

$$\min_{p_S}\mathbb{E}_{(x,I) \sim \mathcal{D}}\left[D_{\mathrm{KL}}(p_S(\cdot|x) \| p_T(\cdot|x,I))\right]$$

变量：
- $p_S(\cdot|x)$：student 看不到 $I$ 的输出分布
- $p_T(\cdot|x,I)$：teacher 看到 $I$ 的输出分布
- $\mathcal{D}$：$(x, I)$ 的联合分布

最优解 closed form：

$$p_S^*(y|x) = \frac{\exp\left(\mathbb{E}_{I \sim \mathcal{D}(\cdot|x)} \log p_T(y|x,I)\right)}{\sum_{y'} \exp\left(\mathbb{E}_{I \sim \mathcal{D}(\cdot|x)} \log p_T(y'|x,I)\right)}$$

变量：
- 分子：对所有可能 PI $I$，取 $\log p_T(y|x,I)$ 的期望（在 $I$ 的条件分布 $\mathcal{D}(\cdot|x)$ 上），再 exp
- 分母：对所有 $y'$ 同样做，归一化

这是 PI-conditioned teacher 分布的 **normalized geometric mean**（几何平均）。

几何平均有个残酷特性：它**压低那些"在某些 PI 下高、在其他 PI 下低"的 outputs**，保留那些"在所有 PI 下都高"的 outputs。算术平均会被极端值拉高，几何平均会被极端值拉低。

### 两种 PI 结构截然不同的命运

**Case 1：Instance-specific PI（数学题每题答案不同）**

考虑两道题：
- 题 A 的 PI 是 $I_1 = "答案是 7"$，teacher 在 "7" 上概率高
- 题 B 的 PI 是 $I_2 = "答案是 42"$，teacher 在 "42" 上概率高

student 看不到 PI，它要学一个 $p_S(\cdot|x)$。对于题 A，最优 $p_S$ 是几何平均——"7" 和 "42" 在不同 PI 下概率悬殊，几何平均把它们都压低。student 学不到任何具体答案，只能学到一些"看起来像解题但避开具体数字"的废话。

更糟的是数学题的 PI 每题都不同，每道题的 $I$ 都是 instance-specific 的，几何平均把所有 instance-specific 信息都抹平了。student 学到的 policy 比 PI-conditioned teacher 弱一大截。

**Case 2：Shared latent PI（system prompt、alignment preference）**

system prompt 对所有题都是同一条 rule，比如"请简洁回答"。不同题的 PI 都是这条 rule，teacher 在 PI 下学到的行为是一致的：输出更短、跳过冗余步骤。几何平均对一致行为的保留很好，因为不同 examples 下 teacher 高概率的 outputs 是一致的（都倾向于"简洁"）。student 学到的就是把这条 rule 压进参数。

alignment 同理：character profile、emotion factor 在不同 examples 下都是同一套行为模式（同一 character、同一 emotion class），几何平均保留得很好。

### 实测对比

**Math OPSD（失败）**：Qwen3-1.7B + OpenThoughts，answer-only PI 和 full-response PI 都没改进，full-response PI 更差（更长 PI 引入更大 distributional shift）。即使换 Qwen3-8B 作 teacher，PI 仍然无用——不是 teacher 不够强，是 PI 结构本身不适合 OPSD。

**Alignment OPSD（成功）**：CharacterBench、EmotionBench 上 OPSD 比 GRPO/PPO 收敛更快。Character profile、emotion factor 跨 examples 共享同一 latent rule，几何平均能保留。

**System Prompt Internalization（成功）**：reasoning compression 上 OPSD 大幅缩短 response length 不损 accuracy。conciseness system prompt 是固定 rule，跨 examples 一致。

**Persuasion OPSD（失败）**：Persuasion for Good 上 OPSD 20-30 步后 collapse。PI 包含 persuadee 的 personality 和 high-level strategy——这些是 instance-specific 的，不同对话 PI 不一致，几何平均抹平。

### 几何平均的启示

这个 closed-form 公式是这篇文章最深刻的结果。它告诉我们 OPSD 本质上是在做 PI marginalization。它不是把"PI 加给 student"的简单操作，是一个有特定数学结构的最优化问题。PI 的结构决定了这个最优化的成败。

要让 OPSD 在 instance-specific PI 上 work，需要让不同 PI 下的 teacher behaviors 在某个 latent space 上对齐。可能的方向：
- Latent PI encoding：把 instance-specific PI 编码成 latent vector，让 student 学到 latent-conditioned policy（但这又把 PI 引入 student 了，违背 OPSD 初衷）
- Mixture of Experts teacher：不同 PI 路由到不同 expert，几何平均在 expert 内部完成
- Iterative pipeline：SFT 提供稳定初始化，RL 提升任务能力，OPD 把 RL 学到的 behavior distill 回 student（paper §7 的提议）

---

## 修复方案：三个工程 trick

### Trick 1：Stop-gradient TopK

如上所述，把 log-prob 项 stop 梯度。实测 K=5 下 unnormalized 会崩，stop-grad 版本稳定。

### Trick 2：RLVR-adapted Teacher

直觉：teacher 不需要 benchmark accuracy 最高，需要 output distribution 与 student 接近。

实验：用 DAPO 训 Qwen3-1.7B 200 steps 得到 Qwen3-1.7B-GRPO，与 Qwen3-8B 同时作 teacher。两者在 Math500/AIME24/AIME25 上 accuracy 相当，但前者作 OPD teacher 显著优于后者——因为 distribution 更 aligned。

| Teacher | Math accuracy | OPD effectiveness |
|---|---|---|
| Qwen3-8B | ≈ 同等 | 较弱 |
| Qwen3-1.7B-GRPO | ≈ 同等 | 强 |

这呼应失败机制 1：student prefix 扭曲 teacher state 的程度取决于 teacher 是否在 student-like region 上有 calibrated output。一个没在自己 rollout 分布上经过 RL 的 teacher，面对 student prefix 时容易"水土不服"。

### Trick 3：SFT Stabilization

Qwen3-1.7B-Base 直接 OPD 时初期产出 garbled Unicode（非英文乱码）。这种乱码 prefix 让 teacher 也给不出有意义的 supervision——加剧 prefix distortion。修复：先用 teacher traces 做 SFT warm-up，把 student 拉到 well-formed region。

SFT 前后 NLL/PPL 对比：

| Setting | Avg. NLL | PPL |
|---|---|---|
| Before SFT (Qwen3-1.7B-Base on Qwen3-4B traces) | 0.640 | 1.896 |
| After SFT | 0.335 | 1.397 |

NLL 减半，student 进入 teacher output format 的支持集。SFT 不是单纯 warm-up 加数据，是让 student 进入 teacher feedback 能起作用的 region。

---

## 评估偏见与工程细节

### 评估偏见 1：max generation length

OPD 倾向产更长或 repetitive traces。如果 validation 时 max gen length 太短，response 被截断，accuracy 看起来降但未必反映能力下降。这是 OPD evaluation 的隐形陷阱。

### 评估偏见 2：early-stage sample efficiency

OPD 早期提升快，GRPO 后期持续改进。只比 early stage 会误导——必须看 performance ceiling。OPD 的优势主要在 sample efficiency 不在 final performance。

### Teacher Signal 分布特性

**Length-skewed**：∆logprob（teacher log-prob - student log-prob）在早期 token 大，后期小。早期 token supervision 更重要。这跟 reasoning 的特性一致——早期 token 决定 reasoning branch，后期是 branch 内细节。

**Correctness-skewed**：incorrect trajectories 接收的 supervision 比 correct 强。OPD 主要提供 corrective signal for wrong paths，对 correct path 信号弱。这解释了为什么 OPD 在已经强的 student 上提升有限——大部分 sample 已经接近正确，supervision 信号弱。

### TopK Engineering Challenge

SGLang 的 token_ids_logprob API 不支持 per-position token list，只接受 flat list。理想是每个 position 取自己的 TopK，实际必须 flatten 成 union set。最坏情况下 per-position TopK 完全 disjoint，memory 增加 $\frac{\min(|\mathcal{V}|, TK)}{K}$ 倍。

工程修复：只对 teacher TopK 和 student TopK 的 intersection 反传梯度：

$$S_K(y_{<t}) = S_{\mathrm{tea},K}(y_{<t}) \cap S_{\mathrm{stu},K}(y_{<t})$$

引用 Li et al. [21] 发现这种 intersection 与纯 student TopK 效果相当。

---

## 完整失败模式 catalog

| 失败模式 | 触发条件 | 现象 |
|---|---|---|
| Length Explosion | unnormalized TopK reverse KL + 偏弱 student | "wait"/"but" 反复出现，response 极其 verbose |
| Repetition Collapse | unnormalized TopK reverse KL 后期 | "maybe maybe maybe..." 重复，accuracy 跌到 ~0 |
| Thinking Mode Hacking | student no-think + teacher think 配对 | student 自发产生 `</think>...`</think>` 等控制 token |
| Response Length Collapse | answer-only PI 太强 | student 直接输出 `\boxed{7}` 跳过 reasoning |
| Persuasion OPSD Collapse | instance-specific PI | 20-30 steps 后 collapse，truncation ratio 接近 1.0 |

---

## 何时用 OPD / OPSD：决策树

### 用 OPD 当且仅当

- student 已经在 well-formed output region（必要时先 SFT）
- teacher 与 student distribution 接近（必要时先 RLVR-adapt teacher）
- 使用 stable loss（stop-grad TopK / renorm TopK / sampled-token PG）
- 接受 teacher ceiling，不需要突破

### 用 OPSD 当且仅当

- PI 是 shared latent rule（system prompt、alignment preference）
- 不期望突破 teacher capability 上限
- 想要 sample efficiency 不想要 late-stage improvement
- model scale 足够大（8B 以上才适合做 reasoning compression）

### 不要用 OPSD 当

- PI 是 instance-specific（数学答案每题不同）
- PI 太长太丰富（full-response PI 比 answer-only 更差）
- teacher 没经过 RL 优化（PI alone 不能让 teacher 学会用 PI）
- 不同 examples 的 PI 行为不兼容

---

## 与其他范式的关系

### OPSD vs RLVR

OPSD 是 RLVR 的一种 dense supervision 替代。
- 优势：token-level supervision，不需要 long generation，sample efficient
- 劣势：受 teacher capability 上限约束
- 实践建议：早期用 OPSD 快速迭代，后期切到 GRPO

### OPD vs SFT

SFT 是 off-policy distillation：训练时 student 看到 teacher 写的 prefix。
- SFT 优势：稳定，不需要 teacher 在 student prefix 上推理
- SFT 劣势：state distribution mismatch
- OPD 解决了 state distribution mismatch 但引入 prefix distortion 问题
- paper 显示 SFT + OPD 组合最佳：SFT 进入 well-formed region，OPD 在 on-policy 上精修

### OPD vs DAgger

DAgger (Dataset Aggregation) 是经典 imitation learning 算法，解决 covariate shift 问题：让 expert 在 novice 的 state distribution 上提供 labels。OPD 与 DAgger 的思路完全一致——都是 on-policy imitation。但 OPD 在 LLM setting 下有独特挑战：teacher (LLM) 在 student prefix 上的 behavior 不稳定（如 §3.1 所述），而 DAgger 一般假设 expert 在任何 state 下都能给出合理 action。

参考：https://arxiv.org/abs/1011.0686 (DAgger original paper)

---

## 末端联想：未来方向

### Iterative Self-Improvement Pipeline

paper §7 提议：SFT 提供 stable init，RL 提升任务能力，OPD distill 回 student。这形成一个迭代循环：

1. SFT base student 进入 well-formed region
2. RLVR 提升 student 的任务能力（突破 SFT ceiling）
3. OPD 把 RLVR-improved student 的 behavior 蒸馏成新 student（蒸馏时 dense token-level supervision）
4. 回到步骤 2

这个 pipeline 利用了每种方法的优势：SFT 稳定，RL 突破，OPD 蒸馏。关键是要在每一步使用 appropriate teacher——RLVR-adapted teacher 作 OPD teacher。

### Latent PI Encoding

要让 OPSD 在 instance-specific PI 上 work，需要让不同 PI 下的 teacher behaviors 在 latent space 上对齐。可能的方向：
- 把 instance-specific PI 编码成 latent vector
- student 学到 latent-conditioned policy
- 但这又把 PI 引入 student，违背 OPSD 初衷
- 折中：student 学到 marginal over latent，但 latent 在训练时被显式 condition

### Mixture of Experts Teacher

- 不同 PI 路由到不同 expert
- 几何平均在 expert 内部完成
- student 蒸馏时学到 expert selection 的 soft mixture

### Curriculum Learning in OPD

- 早期训练时 student prefix 较短，teacher distortion 小
- 后期 prefix 变长，distortion 增加
- 可能需要 curriculum：从短 rollout 开始，逐步增加 rollout length
- 类似 DAgger 的 BetaDecay scheduling

### Multi-teacher OPD

- 用多个 teacher 提供不同视角的 supervision
- 几何平均天然适合多 teacher 场景
- 但要保证 teacher 之间 distribution compatible

---

## 最后的直觉

这篇 paper 的核心贡献是把 OPD/OPSD 的"何时 work、何时崩"这件事讲清楚了。三个失败机制覆盖了不同层次：

- **数据层**（prefix distortion）：teacher 被学生拽进死胡同
- **优化层**（TopK bias）：常数项没消掉导致提升阈值抬高
- **目标层**（PI marginalization）：几何平均天然抹平 instance-specific 信息

修复方案也对应三个层次：
- **优化层修复**（stop-grad TopK）：阻止 biased gradient
- **数据层修复**（RLVR-adapted teacher + SFT）：让 teacher-student distribution 接近，让 student 进入 well-formed region
- **目标层修复**：目前没有直接修复——只能选择合适的 PI 结构（shared latent rule）

最重要的洞察是 OPSD 那个 closed-form 几何平均解。它告诉我们 OPSD 本质上是一种 marginalization 操作，它的成败取决于被 marginalize 的 PI 的结构。这不是工程问题，是数学结构问题。要突破这个限制，需要重新设计 OPSD 的 objective——比如不再用 reverse KL marginal，用其他 divergence 让 student 学到不同的 aggregation。

---

## 参考链接

- 主论文：https://arxiv.org/abs/2506.22193
- Thinking Machines Lab OPD blog: https://thinkingmachines.ai/blog/on-policy-distillation
- Agarwal et al. on-policy distillation: https://arxiv.org/abs/2406.18217
- DAPO: https://arxiv.org/abs/2503.14476
- PPO: https://arxiv.org/abs/1707.06347
- OpenThoughts: https://github.com/open-thoughts/open-thoughts
- Wildguardmix: https://arxiv.org/abs/2406.16495
- CharacterBench: https://arxiv.org/abs/2503.14675
- EmotionBench: https://arxiv.org/abs/2402.09529
- Persuasion for Good: https://arxiv.org/abs/1906.06725
- GPQA: https://arxiv.org/abs/2311.12022
- DAgger: https://arxiv.org/abs/1011.0686
- Minillm: https://arxiv.org/abs/2309.16379
- Context distillation (Snell et al.): https://arxiv.org/abs/2501.13307
- SDPO (Hübotter et al.): https://arxiv.org/abs/2506.08024

---

# The Many Faces of On-Policy Distillation 深入解读

## 1. 问题定位与核心区分

这篇 paper 来自 UIUC + Renmin University + Peking University 团队，系统性地剖析了 OPD (On-Policy Distillation) 与 OPSD (On-Policy Self-Distillation) 在 LLM post-training中的成败机理。核心矛盾在于：文献中既有成功案例（system prompt internalization, knowledge compression），也有失败报告（instability, degradation），到底什么时候 work、什么时候 break、为什么。

两个概念的关键区别：

- **OPD**：teacher 是 external stronger model（如 Qwen3-8B 教 Qwen3-1.7B），PI (privileged information) optional
- **OPSD**：teacher 不是外部模型，是 student 自己 augmented with PI（如把 final answer $I$ 喂给 student 当作 teacher）；student 训练时不看 $I$，要把 PI-conditioned behavior 压进参数

参考：https://arxiv.org/abs/2411.04356 ，https://thinkingmachines.ai/blog/on-policy-distillation ，https://arxiv.org/abs/2402.15041

---

## 2. 核心公式与变量定义

### 2.1 OP(S)D 主目标

$$
\mathcal{L}_{\mathrm{OP(S)D}} = \mathbb{E}_{x,y}\left[\frac{1}{T}\sum_{t=1}^{T}\ell_t\left(\pi_\theta(\cdot\mid x,y_{<t}),\, \mathrm{stopgrad}(\pi_T(\cdot\mid x,y_{<t},I))\right)\right]
$$

变量含义：
- $x \sim \mathcal{D}$：input prompt
- $y = (y_1,\dots,y_T) \sim \pi_\theta(\cdot\mid x)$：student 自己 sample 出来的 trajectory
- $\pi_\theta$：student policy（参数 $\theta$）
- $\pi_T(\cdot\mid x,I)$：teacher policy，可吃 optional PI $I$
- $T$：sequence length
- $\ell_t$：token-level loss（reverse KL、forward KL、TopK 等变种）
- $\mathrm{stopgrad}$：teacher 不参与 backprop（OPD 里 teacher 是 frozen 外部模型；OPSD 里 teacher 是 student clone，必须 stopgrad 防止 self-feedback）

关键 insight：训练数据来自 student 自己的 policy，但 supervision 来自 teacher。这就是 on-policy 的精髓——student 在自己的 state distribution 上学习，而不是在 teacher 的 distribution 上学。

### 2.2 Full-vocabulary Reverse KL

$$
D_{\mathrm{KL}}(\pi_\theta \| \pi_T) = \sum_{v}\pi_\theta(v)\log\frac{\pi_\theta(v)}{\pi_T(v)}
$$

- $v \in \mathcal{V}$：vocabulary token

为什么选 reverse KL 而不是 forward KL？因为 reverse KL 是 **mode-seeking**，倾向于保留 student 高概率 modes，避免 catastrophic forgetting；forward KL 是 mode-covering，会强迫 student 覆盖 teacher 高概率但 student 低概率的 token，在 OPD 里会 push student 走向 teacher-preferred but student-unlikely tokens，造成 instability。

### 2.3 Full-vocab Reverse KL 的梯度

$$
\nabla_\theta D_{\mathrm{KL}}(p_\theta \| p_T) = \sum_v p_\theta(v)\left[\log\frac{p_\theta(v)}{p_T(v)} + \mathbf{1}\right]\nabla_\theta \log p_\theta(v)
$$

- $p_\theta(v) = \pi_\theta(v\mid x,y_{<t})$：student 在 token $v$ 上的概率
- $p_T(v) = \pi_T(v\mid x,y_{<t},I)$：teacher 在 token $v$ 上的概率

这里隐藏一个关键恒等式：

$$
\sum_v p_\theta(v)\nabla_\theta \log p_\theta(v) = \nabla_\theta \sum_v p_\theta(v) = 0
$$

所以 full-vocab 下，常数 +1 项贡献为零，gradient 简化为只含 $\log\frac{p_\theta}{p_T}$ 的项。但下面会看到，这个恒等式在 TopK 截断下失效——这是论文核心发现之一。

### 2.4 Sampled-token Policy Gradient 视角

把 teacher-student log-prob gap 当作 advantage：

$$
\mathcal{L}_{\mathrm{PG}} = -\mathbb{E}_{y\sim\pi_\theta}\sum_t \mathrm{stopgrad}\!\left(\log\pi_T(y_t\mid x,y_{<t},I) - \log\pi_\theta(y_t\mid x,y_{<t})\right)\log\pi_\theta(y_t\mid x,y_{<t})
$$

梯度形式：

$$
\nabla_\theta \mathcal{L}_{\mathrm{PG}} = -\mathbb{E}_{y\sim\pi_\theta}\sum_t \mathrm{stopgrad}\!\left(\log\pi_T(y_t\mid x,y_{<t},I) - \log\pi_\theta(y_t\mid x,y_{<t})\right)\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})
$$

这种形式自然就是 PPO-style KL regularization 的 dual 视角：把 distillation 当成 dense token-level reward。这个视角下天然避开了下面要讲的 TopK bias 问题。

---

## 3. 三大失败机制

论文识别了三种机制，每种都有清晰的数学/经验证据。

### 3.1 Mechanism 1: Student Prefix 扭曲 Teacher State

直觉：teacher 是强模型，独立解题能做对；但 OPD 要求 teacher 在 student-generated prefix 上继续输出。student 已经走偏了一条 reasoning branch，teacher 继续这条 branch 时会陷入一个它自己不会到达的中间状态。

**实验证据**（Appendix A.22）：在 GPQA-Diamond 上，Qwen3-1.7B 作 student、Qwen3-14B 作 teacher。

| Setting | Accuracy | Format Correctness |
|---|---|---|
| Standalone teacher | 62.12% (123/198) | 98.48% |
| Student-prefix-conditioned teacher | 45.96% (91/198) | 78.79% |

Transition statistics：
- 40 个原本 teacher 答对的题，prefix continuation 后变错
- 8 个原本 teacher 答错的题，prefix continuation 后变对
- 净损失 32 cases，下降 16.16 points

**Token-level 表现**（Figure 8）：当 student prefix 已经 commit 到一条 reasoning branch，teacher 给 revision tokens（如 "wait", "but"）高概率——试图 redirect 而非 extend。这造成 local semantic conflict：student 在路径 A 上，teacher 喊"回头重来"，student 收到的 supervision 是"不要继续这条路"，于是产生 verbose 和 inconsistent reasoning。

直觉构建：teacher 在 OPD 中扮演的角色不是"教解题"，而是"每一步指方向"。但 prefix 是 student 写的，teacher 没有参与 prefix 的决策，被强行拽进 student 的 reasoning state 里——它的"指方向"信号会基于错位的上下文给出错位的建议。这跟 RLVR 的根本区别在于，RLVR 的 supervision 来自最终 reward signal，不依赖 teacher 对 prefix 的理解；OPD 是 dense token-level supervision，每一步都依赖 teacher 在当前 prefix 下的判断。

### 3.2 Mechanism 2: TopK Reverse-KL Gradient Bias

这是论文最漂亮的数学发现。TopK truncation 是为了省 GPU memory（full-vocab KL 在大 vocab 上太贵），但它破坏了 §2.3 中那个让 +1 项消失的恒等式。

**TopK reverse KL loss**：

$$
\mathcal{L}_{\mathrm{Top\text{-}K\text{-}RKL}}(t) = \sum_{v\in S_K(y_{<t})}\pi_S(v\mid x,y_{<t})\log\frac{\pi_S(v\mid x,y_{<t})}{\pi_T(v\mid x,y_{<t},I)}
$$

- $S_K(y_{<t}) \subset \mathcal{V}$：截断后的 TopK 子集，通常是 student 或 teacher 的 TopK

**梯度**：

$$
\nabla_\theta\mathcal{L}_{\mathrm{Top\text{-}K\text{-}RKL}}(t) = \sum_{v\in S_K(y_{<t})}\pi_S(v\mid x,y_{<t})\left[\log\frac{\pi_S(v\mid x,y_{<t})}{\pi_T(v\mid x,y_{<t},I)} + \mathbf{1}\right]\nabla_\theta\log\pi_S(v\mid x,y_{<t})
$$

关键问题：

$$
\sum_{v\in S_K(y_{<t})}\pi_S(v\mid x,y_{<t})\nabla_\theta\log\pi_S(v\mid x,y_{<t}) = \nabla_\theta\sum_{v\in S_K(y_{<t})}\pi_S(v\mid x,y_{<t}) \neq 0
$$

因为 TopK 子集没有覆盖整个 vocabulary，$\sum_{v\in S_K}\pi_S(v) \neq 1$，这个 sum 对 $\theta$ 是有梯度的。+1 项保留下来意味着：

**token 被提升的判据**：$\pi_T(v) > e\cdot\pi_S(v)$（因为要让方括号内 $<0$，需要 $\log\frac{\pi_S}{\pi_T} + 1 < 0$，即 $\frac{\pi_S}{\pi_T} < e^{-1}$，即 $\pi_T > e\cdot\pi_S$）。

这把提升阈值从正常的 $\pi_T > \pi_S$ 抬高了 $e$ 倍。结果：teacher 偏好但 teacher-student gap 不够大的 token 仍然被 suppress，student 被推向 teacher 分布之外的低概率 continuation，触发 instability。

**经验证据**（Figure 4, 11）：Qwen3-1.7B student + Qwen3-8B teacher，OpenThoughts，unnormalized Top20 reverse KL：
- Step 0 → 700：rollout length 暴增，"wait" "maybe" revision tokens 出现
- Step 700 → 1000：退化成重复 "maybe maybe maybe..."，accuracy 在 Math500/AIME24/AIME25 跌到近零
- Token stats：repeat ratio 接近 1，teacher-student overlap 崩溃

这是一个典型的**训练后期 collapse**：早期 loss 看着不错，但优化路径上累积 bias 最终把 student 推到 degenerate region。

### 3.3 Mechanism 3: OPSD 学到的是 PI-Free Marginal Policy

OPSD 的优化目标：

$$
\min_{p_S}\mathbb{E}_{(x,I)\sim\mathcal{D}}\left[D_{\mathrm{KL}}\big(p_S(\cdot\mid x) \| p_T(\cdot\mid x,I)\big)\right]
$$

- $p_S(\cdot\mid x)$：student 看不到 $I$
- $p_T(\cdot\mid x,I)$：teacher 看到 $I$（同一个 student clone）

最优解：

$$
p_S^\star(y\mid x) = \frac{\exp\left(\mathbb{E}_{I\sim\mathcal{D}(\cdot\mid x)}\log p_T(y\mid x,I)\right)}{\sum_{y'}\exp\left(\mathbb{E}_{I\sim\mathcal{D}(\cdot\mid x)}\log p_T(y'\mid x,I)\right)}
$$

- 分子：在 PI 分布 $\mathcal{D}(\cdot\mid x)$ 上对 $\log p_T$ 取期望，再 exp
- 分母：对所有可能 $y'$ 同样做，归一化

这是 PI-conditioned teacher 分布的 **normalized geometric mean**。

**关键 insight**：geometric mean 倾向于压低那些"在某些 PI 下高概率、在其他 PI 下低概率"的 outputs，保留那些"在所有 PI 下都高概率"的 outputs。

这导致两种截然不同的命运：

| PI 结构 | 几何平均效果 | OPSD 是否 work |
|---|---|---|
| **Instance-specific PI**（如数学题每题答案不同 $I_1, I_2, I_3$ 各异）| 不同 PI 下 teacher 行为相互 incompatible，几何平均压低所有答案-related tokens | **失败**：student 学到"什么都不输出"或"绕开答案"，比 PI-conditioned teacher 弱很多 |
| **Shared latent PI**（如固定 system prompt 对所有 examples 相同，或 alignment preference 一致）| 不同 examples 共享同一个 latent rule $I$，几何平均保留 rule-supported outputs | **成功**：student 把 PI-conditioned behavior 压缩成可复用 inductive bias |

实验对比：

- **Math OPSD 失败**（Figure 3, 10）：Qwen3-1.7B + OpenThoughts，step-0 checkpoint 当 teacher；answer-only PI 和 full-response PI 都没改进；full-response PI 更差（更长 PI 引入更大 distributional shift）。即使换 Qwen3-8B 作 teacher，PI 仍然无用（Figure 10）。
- **Alignment OPSD 成功**（Figure 5）：CharacterBench、EmotionBench 上，OPSD 比 GRPO/PPO 收敛更快——因为 alignment 是 shared latent rule。
- **System Prompt Internalization 成功**（Figure 6, 7）：reasoning compression 上 OPSD 大幅缩短 response length 不损 accuracy；safety alignment 上 OPSD 早期提升快但被 teacher 上限约束（GRPO 后期更稳）。

---

## 4. 三种修复方案

### 4.1 Stop-Gradient TopK Loss

把 log-prob 项的梯度 stop 掉，让 weighting 只起 advantage 作用：

$$
\mathcal{L}_{\mathrm{SG\text{-}TopK}}(t) = -\sum_{v\in S_K(y_{<t})}\pi_S(v\mid x,y_{<t})\left[\log\pi_T(v\mid x,y_{<t},I) - \mathrm{stopgrad}\big(\log\pi_S(v\mid x,y_{<t})\big)\right]
$$

梯度：

$$
\nabla_\theta\mathcal{L}_{\mathrm{SG\text{-}TopK}}(t) = -\sum_{v\in S_K(y_{<t})}\pi_S(v\mid x,y_{<t})\left[\log\pi_T(v\mid x,y_{<t},I) - \log\pi_S(v\mid x,y_{<t})\right]\nabla_\theta\log\pi_S(v\mid x,y_{<t})
$$

注意方括号里没有 +1，因为 stopgrad 阻止了 $\log\pi_S$ 反传梯度，那么作为 $\pi_S$ 的 implicit function，那个 $+1$ 项就消失了。判据变成正常的 $\pi_T > \pi_S$。

**对比方案 - Renormalized TopK**（Eq. 14-16）：

$$
\bar{\pi}_S = \frac{\pi_S(v\mid x,y_{<t})}{\sum_{u\in S_K}\pi_S(u\mid x,y_{<t})},\quad \bar{\pi}_T = \frac{\pi_T(v\mid x,y_{<t},I)}{\sum_{u\in S_K}\pi_T(u\mid x,y_{<t},I)}
$$

$$
\mathcal{L}_{\mathrm{Renorm\text{-}Top\text{-}K\text{-}RKL}}(t) = \sum_{v\in S_K}\bar{\pi}_S(v)\log\frac{\bar{\pi}_S(v)}{\bar{\pi}_T(v)}
$$

在 TopK set 内重新归一化，恢复 $\sum_{v\in S_K}\bar{\pi}_S(v) = 1$，让 +1 项再次为零。代价：只匹配 TopK 内相对概率，忽略 mass 分配，不再是 full-vocab reverse KL 的忠实近似。

**实验**（Figure 11, 12）：
- Unnormalized reverse KL（K=5）：训练中 collapse
- Stop-gradient version：稳定
- Renormalized version：稳定，性能相当
- Sampled-token PG（K=1 with stopgrad）：稳定；不加 stopgrad 的 Top1 reverse KL 仍 collapse

### 4.2 RLVR-adapted Teacher

直觉：teacher 不需要 benchmark accuracy 最高，需要 distribution 与 student 接近。让 teacher 先在 training distribution 上做 RLVR (GRPO)，能让 teacher 的 output distribution 更贴近 student 自己的 on-policy distribution。

**实验**（Figure 13）：
- Qwen3-1.7B-GRPO（用 DAPO 训 200 steps）vs Qwen3-8B 作 teacher
- 两者 Math500/AIME24/AIME25 性能相当
- 但前者作 OPD teacher 显著优于后者
- Top20 vocab 分布更 aligned with student

| Teacher | Math perf | OPD effectiveness | Top20 overlap with student |
|---|---|---|---|
| Qwen3-8B | ≈ 同等 | 较弱 | 较低 |
| Qwen3-1.7B-GRPO | ≈ 同等 | **强** | **高** |

insight：teacher accuracy 不足以预测 OPD effectiveness；distribution proximity 才是关键。这呼应 §3.1——student prefix 扭曲 teacher state 的程度取决于 teacher 是否在 student-like region 上有 calibrated output。

### 4.3 SFT Stabilization

**问题**：Qwen3-1.7B-Base 在 OpenThoughts 上 OPD 时，初期产出 garbled Unicode（非英文乱码），teacher 给不出有意义的 supervision。

**修复**：先用 teacher（Qwen3-4B）的 traces 做 SFT warm-up，把 student 拉到 well-formed output region。

SFT 数据准备（Appendix A.23）：
- 30,000 prompts for SFT split，剩余给 OPD split
- Teacher rollout: temperature 0.3，max 4096 tokens，n=1
- 质量过滤后 ~20,000 correct samples
- SFT hyperparameters: lr 1e-5, cosine decay, 2 epochs, DeepSpeed ZeRO Stage 2, FlashAttention-2, BF16

SFT 前后 NLL/PPL 对比（Table 3）：

| Setting | Student | Avg. NLL | PPL |
|---|---|---|---|
| Before SFT | Qwen3-1.7B-Base on Qwen3-4B traces | 0.640 | 1.896 |
| After SFT | Qwen3-1.7B-Base-SFT on Qwen3-4B traces | 0.335 | 1.397 |

insight：SFT 不是单纯 warm-up 加数据，是**让 student 进入 well-formed region**，让 teacher feedback 有意义。当 student 输出乱码时，teacher 在乱码 prefix 上的 supervision 也无意义——加剧 §3.1 的 prefix distortion 问题。SFT 通过缩小 student 与 teacher 的 output format 差距，间接减小了 prefix distortion。

---

## 5. 评估偏见与工程细节

### 5.1 评估偏见（Appendix A.8）

1. **Max validation length 影响 accuracy 测量**：OPD 倾向产更长或 repetitive traces，max gen length 太短会被截断，accuracy 看起来降但实际未必反映能力下降。
2. **早期 sample efficiency 比较不公平**：OPD 早期提升快，GRPO 后期持续改进。只比 early stage 会误导——必须看 performance ceiling。

### 5.2 TopK Engineering Challenge（Appendix A.6）

SGLang 的 token_ids_logprob API 不支持 position-dependent token lists，只接受 flat list。理想是 per-position TopK，实际必须 flatten 成 union set $U = \bigcup_t S_{\mathrm{stu},K}(y_{<t})$。

memory 成本：
- 理想：$T \times K$
- 实际：$T \times |U|$
- 最坏情况（per-position TopK disjoint）：$|U| = \min(|\mathcal{V}|, TK)$，memory 增加因子 $\frac{\min(|\mathcal{V}|, TK)}{K}$

修复：

$$
S_K(y_{<t}) = S_{\mathrm{tea},K}(y_{<t}) \cap S_{\mathrm{stu},K}(y_{<t})
$$

只对两模型 TopK 交集反传梯度。引用 Li et al. [21] 发现这种 intersection 与纯 student TopK 效果相当。

### 5.3 Teacher Signal 分布特性（Appendix A.9-A.11）

**Length-skewed supervision**（Figure 16, 20）：
- ∆logprob 在早期 token 大，后期小
- 早期 token supervision 更重要

**Correctness-skewed**（Figure 17）：
- Incorrect trajectories 接收的 supervision 比 correct 强
- OPD 主要提供 corrective signal for wrong，对 correct path 信号弱
- 这解释了为什么 OPD 在已经强的 student 上提升有限

**PI 影响**（Figure 18）：
- PI 只 refine signal
- 根本分布由 teacher 规模和 reasoning depth 决定
- 同一 teacher 不同 PI 设计 supervision 分布相近
- 不同 teacher supervision 分布显著不同

---

## 6. Hyperparameters 完整对比

Table 1 关键超参：

| Category | OPD | OPSD | GRPO | PPO |
|---|---|---|---|---|
| Rollout batch size | 64 | 64 | 32 | 64 |
| Prompts per batch | 64 | 64 | 32 | 64 |
| Samples per prompt | 1 | 1 | 8 | 1 |
| Max rollout length | 4096 | 4096 | 8192 | 8192 |
| Rollout temp | 1.0 | 1.0 | 1.0 | 1.0 |
| Rollout top-p | 0.95 | 0.95 | 0.95 | 0.95 |
| Optimizer | Adam | Adam | Adam | Adam |
| Learning rate | 2e-6 | 2e-6 | 2e-6 | 2e-6 |
| LR schedule | cosine | cosine | cosine | cosine |
| Warmup fraction | 0.1 | 0.1 | 0.1 | 0.1 |
| Weight decay | 0.1 | 0.1 | 0.1 | 0.1 |
| Adam β₁/β₂ | 0.9/0.98 | 0.9/0.98 | 0.9/0.98 | 0.9/0.98 |
| Training objective | Top-K reverse KL | Top-K reverse KL | policy gradient | PPO |
| Top-K | 20 | 20 | — | — |
| Advantage | — | — | GRPO | GAE |
| GAE γ/λ | — | — | — | 1.0/0.95 |
| Teacher PI | none | final answer / full response | — | — |
| Teacher model | external | step-0 student | — | — |

注意 OPD/OPSD 用 1 sample per prompt，GRPO 用 8——GRPO 多 sample 是为了 advantage normalization；OPD/OPSD 不需要 advantage baseline 因为 supervision 是 dense token-level。

硬件：10× NVIDIA RTX PRO 6000 Blackwell GPU。

---

## 7. 完整失败模式 catalog

### 7.1 Length Explosion（Figure 4, A.16）

Step 700 sample：Qwen3-1.7B distill from Qwen3-8B，response 极其 verbose，"but" 出现 10 次。典型行为：student 在 prefix distortion 下，teacher 给 revision tokens 高概率，student 学会反复 revise。

### 7.2 Repetition Collapse（Figure 4, A.17）

Step 1000 sample：response 全是 "maybe maybe maybe maybe..."，length 达到 limit，accuracy 跌到 ~0。Repeat ratio 接近 1。

### 7.3 Thinking Mode Hacking（Appendix A.14）

Student 训练时 thinking mode disabled，teacher 查询时 thinking enabled。Student 学会部分模仿 teacher 的 thinking-mode protocol，自发产生 `<think>...</think>` 等 control tokens，包括 malformed patterns 如 `<think> ... </think> ... <think>`。说明 distillation 不只学 answer distribution，还学 control tokens。

### 7.4 Response Length Collapse（Appendix A.15）

DAPO-Math-17k + answer-only PI，Qwen3-1.7B self-distillation：student 学会直接输出 `\boxed{7}`，跳过 reasoning。PI 太强导致 student 找捷径绕过 reasoning process。

### 7.5 Persuasion OPSD 失败（Appendix A.13）

Persuasion for Good 数据集上，OPSD 早期竞争力强但 20-30 steps 后 collapse。1.7B 和 4B 都出现：truncation ratio 接近 1.0（generation 过长），student log-prob 持续下降（uncertainty 增加）。这进一步印证 §3.3——即使有 PI，如果不同 examples 的 PI 行为不能压成 shared rule，OPSD 不 work。

---

## 8. PI 设计空间维度（Appendix A.3）

OP(S)D 的 design space 三个正交 axis：

### 8.1 Teacher Construction

| 类型 | 公式 | 特性 |
|---|---|---|
| Self-Teacher | $\pi_T(\cdot\mid x,y_{<t},c) = \pi_\theta(\cdot\mid x,y_{<t},c)$ | teacher=student 同参数，随 student 更新；高风险 self-feedback |
| Frozen teacher | $\pi_T = \pi_{\bar\theta}$，$\bar\theta = \theta^{(0)}$ | 稳定但 stale |
| EMA teacher | $\bar\theta_k = \alpha\bar\theta_{k-1} + (1-\alpha)\theta_k$ | 折中，平滑 high-variance updates |

### 8.2 Privileged Information 形式

- **Math reasoning**：full reasoning trace + answer，或 lighter answer-only
- **Agentic**：tool execution results, verifier signals, environment observations
- **External summary**：external model 把 long interaction history 总结成 high-level guidance（适用于 raw feedback 太长）

### 8.3 Distillation Objectives 谱系

Full-vocab 系列：
- Reverse KL: $D_{\mathrm{KL}}(\pi_\theta \| \pi_T)$ — mode-seeking
- Forward KL: $D_{\mathrm{KL}}(\pi_T \| \pi_\theta)$ — mode-covering
- JSD: $\beta D_{\mathrm{KL}}(\pi_\theta \| m_i) + (1-\beta)D_{\mathrm{KL}}(\pi_T \| m_i)$，$m_i = \beta\pi_\theta + (1-\beta)\pi_T$

TopK 截断变种：
- Student TopK: $S_i^S = \mathrm{TopK}(\pi_\theta, k)$
- Teacher TopK: $S_i^T = \mathrm{TopK}(\pi_T, k)$
- Tail-augmented（SDPO 风格）: $p_i^{\mathrm{tail}} = 1 - \sum_{v\in S_i}\pi_\theta(v)$，保留 tail mass

Sampled-token 系列：
- $A_t = \log\pi_T(y_t\mid x,c,y_{<t}) - \log\pi_\theta(y_t\mid x,y_{<t})$
- $k_1 = -A_t$, $k_2 = \frac{1}{2}A_t^2$, $k_3 = e^{A_t} - 1 - A_t$（低方差、非负）

### 8.4 OPD Gradient Decomposition（Appendix A.4）

完整 OPD 梯度：

$$
\nabla_\theta\mathcal{L}_{\mathrm{OPD}}(\theta) = \mathbb{E}_{x,y}\left[\nabla_\theta\ell(\theta;x,y) + \ell(\theta;x,y)\nabla_\theta\log\pi_\theta(y\mid x)\right]
$$

- 第一项：direct gradient（在 sampled prefix 上 backprop token-level loss）
- 第二项：score-function term（rollout 分布对 $\theta$ 的依赖）

实践中常 ignore 第二项以降 variance，但引入 bias。RL-style estimator 保留第二项，unbiased 但高 variance。

---

## 9. Systematic 实验数据总结

### 9.1 Math Reasoning

| Setting | Math500 | AIME24 | AIME25 | 结论 |
|---|---|---|---|---|
| OPSD, answer-PI (Qwen3-1.7B, OpenThoughts) | 无改进 | 无改进 | 无改进 | 失败 |
| OPSD, full-response PI | 更差 | 更差 | 更差 | PI 越丰富 mismatch 越大 |
| OPSD, GRPO-trained PI teacher | 更差 | 更差 | 更差 | PI alone 不解决问题 |
| OPD, unnormalized Top20 RKL (Qwen3-1.7B ← Qwen3-8B) | collapse | collapse | collapse | +1 bias 致死 |
| OPD with PI（answer / response）| 低于 vanilla OPD | — | — | PI 无效 |
| OPD, stop-grad TopK (K=5) | stable | stable | stable | 修复有效 |
| OPD, renorm TopK (K=5) | stable | stable | stable | 修复有效 |
| OPD with Qwen3-1.7B-GRPO teacher | 强于 Qwen3-8B teacher | — | — | distribution proximity 重要 |
| OPD + SFT warm-up (Base student) | stable improvement | — | — | SFT 进入 well-formed region |

### 9.2 Alignment & System Prompt Internalization

| Setting | OPSD vs GRPO/PPO | 机理 |
|---|---|---|
| CharacterBench (Qwen3-4B-Instruct) | OPSD 收敛更快 | PI 是 shared character profile |
| EmotionBench (Qwen3-4B-Instruct) | OPSD 收敛更快 | PI 是 emotion factor，跨 examples 共享 emotion rule |
| Reasoning compression (Qwen3-8B, DAPO-Math-17k) | OPSD 缩短 length 不损 acc | PI 是固定 conciseness system prompt |
| Safety alignment (Qwen3-1.7B, Wildguardmix) | OPSD 早期快，GRPO 后期持续 | PI 是固定 safety system prompt |
| Persuasion for Good (Qwen3-1.7B/4B) | OPSD collapse after 20-30 steps | PI 含 personality + strategy，跨 examples 不一致 |

### 9.3 General Reasoning

Qwen3-1.7B ← Qwen3-8B on Mixture of Thoughts (Science subset)：
- GPQA-Diamond：fluctuate，无 consistent improvement
- MMLU-Pro：marginally improved

Teacher signal 分析（Figure 20）：
- ∆logprob length-skewed：早期 token supervision 强，后期弱
- Correctness-skewed：incorrect responses 接收更强 supervision，correct path 信号弱
- 这解释了为什么在已经强的 reasoning 上 OPD 提升有限——大部分 sample 已经接近正确，supervision 信号弱

---

## 10. 直觉构建总结

### 10.1 何时 OPD work

✓ 当 teacher 与 student distribution 接近（RLVR-adapted teacher）
✓ 当 student 已经在 well-formed region（SFT warm-up）
✓ 当使用 stable loss（stop-grad TopK / renorm TopK / sampled-token PG）
✓ 当 supervision 信号 dense 且 token-level meaningful（早期 token supervision 更重要）

### 10.2 何时 OPD fail

✗ 当 teacher 在 student prefix 上 reasoning state 扭曲（teacher accuracy 在 student prefix 下从 62.1% 跌到 46.0%）
✗ 当 TopK reverse-KL 含 +1 bias（unnormalized）
✗ 当 student 输出乱码（garbled Unicode）
✗ 当 student 已经接近正确（supervision 信号弱）

### 10.3 何时 OPSD work

✓ 当 PI 是 shared latent rule（system prompt、alignment preference）
✓ 当 PI-conditioned teacher behaviors 在不同 examples 上一致
✓ 当 model 足够大（8B scale 才能有效做 reasoning compression）

### 10.4 何时 OPSD fail

✗ 当 PI 是 instance-specific（数学答案每题不同）
✗ 当 PI-conditioned teacher behaviors 不兼容（几何平均压低所有）
✗ 当 PI 太长太丰富（full-response PI 比 answer-only PI 更差）
✗ 当 teacher 本身没经过 RL 优化（PI alone 不能让 teacher 学会用 PI）

### 10.5 OPSD 与 RLVR 关系

OPSD 是 RLVR 的一种 dense supervision 替代：
- 优势：token-level supervision，不需要 long generation，sample efficient
- 劣势：受 teacher capability 上限约束，无法突破 teacher ceiling
- GRPO 优势：可以突破 teacher 上限，但 sample inefficient
- 实践建议：早期用 OPSD 快速迭代，后期切到 GRPO

### 10.6 核心洞察

这篇 paper 最有价值的发现是 **PI 结构决定 OPSD 命运** 的几何平均解释：

$$
p_S^\star(y\mid x) \propto \exp\left(\mathbb{E}_{I\sim\mathcal{D}(\cdot\mid x)}\log p_T(y\mid x,I)\right)
$$

这个公式告诉我们，OPSD 本质上是在做 PI-marginalization。如果 PI 在 $\mathcal{D}(\cdot\mid x)$ 上的分布是 delta function（shared rule），几何平均就是 teacher 本身；如果 PI 在不同 examples 上是 different delta functions（instance-specific），几何平均会把所有 teacher 都压低，student 学不到任何 specific capability。

这给未来 research 方向：要让 OPSD 在 instance-specific PI 上 work，需要**让 PI-conditioned teacher behaviors 在某个 latent space 上对齐**，使得几何平均仍能保留 instance-specific information。可能的路径包括：
- Latent PI encoding（把 instance-specific PI 编码成 latent vector）
- Mixture-of-Experts teacher（不同 PI 路由到不同 expert）
- Iterative self-improvement pipeline（SFT 提供 stable init，RL 提升任务能力，OPD distill 回 student，正如 paper §7 提议）

---

## 参考

- 论文原文：https://arxiv.org/abs/2506.22193 （The Many Faces of On-Policy Distillation）
- On-Policy Distillation blog (Thinking Machines Lab): https://thinkingmachines.ai/blog/on-policy-distillation
- DAPO: https://arxiv.org/abs/2503.14476
- GRPO / PPO: https://arxiv.org/abs/1707.06347
- OpenThoughts: https://arxiv.org/abs/2506.17211 , https://github.com/open-thoughts/open-thoughts
- Wildguardmix: https://arxiv.org/abs/2406.16495
- CharacterBench: https://arxiv.org/abs/2503.14675
- EmotionBench: https://arxiv.org/abs/2402.09529
- Persuasion for Good: https://arxiv.org/abs/1906.06725
- GPQA: https://arxiv.org/abs/2311.12022
- Agarwal et al. on-policy distillation: https://arxiv.org/abs/2406.18217
- Minillm: https://arxiv.org/abs/2309.16379
- Context distillation (Snell et al.): https://arxiv.org/abs/2501.13307
- Self-distillation enables continual learning: https://arxiv.org/abs/2506.08024
