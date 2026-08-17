---
source_pdf: Discovering state-of-the-art reinforcement.pdf
paper_sha256: 73c6a33ba8b5ae8bd4da128b2cabf9517a11117f23a4e3943a7a9f10c5d00237
processed_at: '2026-08-03T22:08:47-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DiscoRL 用人话讲

## 一句话说清楚这工作在干嘛

人类花了几十年发明 RL 的"学习规则"——TD learning [https://www.cs.toronto.edu/~vstroub/restricted/SuttonBartoIPRLBook2ndEd.pdf]、Q-learning [https://link.springer.com/article/10.1007/BF00115009]、PPO [https://arxiv.org/abs/1707.06347]、MuZero [https://www.nature.com/articles/s41586-020-03051-4] 这些。每个新算法都是几个 PhD 花几年调出来的。DeepMind 这篇问的问题是：**能不能让机器自己"发明"一个 RL 算法，而且比人类发明的更好？**

答案是：能。他们用 meta-learning 自动搜出一个叫 DiscoRL 的算法，在 Atari 上超过了 MuZero，在没见过的 ProcGen / Crafter / NetHack 上也表现强。这相当于机器自己写了一个比 PPO 更好的 PPO。

---

## 为什么这事难，之前为什么没人做成

Meta-RL 的想法很老，Schmidhuber 1987 年的 PhD thesis 就提了 [https://link.springer.com/book/10.1007/978-1-4615-3738-7]。但之前的工作都有两个毛病：

**第一个毛病：搜的空间太窄。** 之前的工作大部分只搜 hyperparameters——比如 discount factor γ、learning rate、entropy bonus 这些。Xu et al. 2018 的 meta-gradient RL [https://papers.nips.cc/paper/2018/hash/2d07d626c7d3f3c19a3c5aa6c7f5b4e3-Abstract.html] 是典型例子：底层还是 IMPALA，只是把几个超参换成 meta-learned。这种"小修小补"再怎么调也不可能跳出 IMPALA 的天花板。另一类走相反路线，完全 black-box——比如 RL² [https://arxiv.org/abs/1611.02779] 把整个算法塞进一个 RNN。看着酷，但在简单环境上 overfit，换环境就崩。

**第二个毛病：在 toy 环境上 meta-train。** Oh et al. 2020 [https://papers.nips.cc/paper/2020/hash/178d6933fc5d5e4ee1aa9a9f3a5b8fde-Abstract.html]（这论文的"前身"，同一个一作）就在 grid-world 上 meta-train 一个 meta-network。结果学到的 rule 在 Atari 上很弱。原因直觉：你在 grid-world 上学到的"最优 rule"可能依赖 grid-world 的某些特性（比如 reward 稀疏程度、episode 长度），到 Atari 上就不适用了。就像你只在 MNIST 上学了个"通用学习算法"，拿到 ImageNet 上肯定不行。

DiscoRL 把这两个毛病都治了。下面说怎么治的。

---

## 核心 idea：搜一个"target function"

这是这篇论文最关键的 insight，要讲清楚。

### 传统 RL 算法长什么样

几乎所有现代 RL 算法（TD, Q-learning, PPO, MuZero）本质都遵循一个 template：

> Agent 在每个时刻输出一些 prediction（比如 value function q(s,a)），然后用某个 target 去更新这些 prediction。Target 通常是 future prediction 的某种函数。

举例：
- **Q-learning**: target 是 $r + \gamma \max_{a'} q(s', a')$。这里 target 依赖 future prediction $q(s', a')$，叫 **bootstrapping**。
- **TD(λ)**: target 是 n-step return，混合 real reward 和 future prediction。
- **PPO**: policy 的 target 是 $\pi_{\text{old}} \cdot \exp(A / \text{clip})$，其中 A 是 advantage。
- **MuZero**: target 是 reward 和 policy 的 bootstrap prediction。

所有这些都是"告诉 agent 你的 prediction 应该往这个值靠"。

### DiscoRL 的 move

DiscoRL 说：既然 RL 算法的核心是"产生 target"，那我们就**直接 meta-learn 这个 target function**。具体做法是用一个 meta-network（LSTM）接收 agent 的 prediction trajectory（π, y, z, q 从 t 到 t+n）加上 reward / done flag，输出 target $\hat{\pi}, \hat{y}, \hat{z}$。Agent 用 KL divergence 把自己的 prediction 朝这些 target 推。

为什么搜 target 比 搜 loss 更强？因为 target 可以依赖 future prediction（bootstrapping 的核心），而 scalar loss 不行。Q-learning 这种 semi-gradient 方法在这个 search space 里，但不在 scalar loss 的 search space 里。这是"表达力"上的本质区别。

类比：如果让你设计一个 RL 算法，你是更愿意写"$\hat{q} = r + \gamma q(s', a')$"还是写"minimize some scalar loss"？前者直接表达"目标"，后者要绕一圈。DiscoRL 选了前者。

---

## Agent 的结构：保留 inductive bias，开放语义

DiscoRL 不让 agent 完全 black-box。Agent 是一个带 inductive bias 的标准神经网络，输出 5 样东西：

1. **π(s)**：policy（softmax over actions）—— 这个语义是 pre-defined 的。
2. **y(s) ∈ ℝⁿ**：observation-conditioned prediction vector。语义**未定义**，meta-network 决定它该预测什么。
3. **z(s,a) ∈ ℝᵐ**：action-conditioned prediction vector。语义**未定义**。
4. **q(s,a)**：action-value function。语义 pre-defined，用 Retrace(λ) [https://arxiv.org/abs/1606.02647] 算 target，project 到 two-hot（MuZero 风格 distributional）。
5. **p(s,a)**：auxiliary policy prediction，预测下一步的 policy π(s')。pre-defined。

为什么这么设计？因为完全 black-box（全靠 y, z 发现一切）不稳定，完全 hand-crafted（只有 π, q）又限制了搜索空间。Hybrid 最优——保留 value function 这个强 inductive bias（几十年 RL 研究证明 value function 是核心概念），但留 y, z 这个空间让 meta-network 发现新东西。

y 和 z 同时有 observation-conditioned 和 action-conditioned 版本，是因为 Sutton & Barto 的 prediction vs control 二分——v(s) 和 q(s,a) 的区分在 RL 里到处出现（reward 有 state-based 和 action-based，successor feature [https://arxiv.org/abs/1606.05312] 也是）。Agent architecture 应该 preserve 这个 structure，让 meta-network 有空间去"填空"。

---

## Meta-network 是什么

Meta-network 是一个 LSTM（也试过 transformer，差不多但更慢），输入是从 t 到 t+n 的 agent output 序列（policy, y, z, q）加 reward / action / done flag，输出是 target $\hat{\pi}, \hat{y}, \hat{z}$。三个关键设计：

**1. 看不到 raw observation。** Meta-network 只通过 agent 的 prediction 间接看 state。这让它对图像分辨率、observation modality 完全 agnostic，所以能在 Atari（像素）和 DMLab（第一人称视角）之间 transfer。

**2. Action dimension 共享权重。** Action-specific input 和 output 在 action 维度上 share weight，通过 averaging 得到 embedding。这让 meta-network 能处理任意大小 discrete action space。

**3. 有两个 LSTM。** 一个 within-episode（从 t+n 反向 unroll 到 t，类似 TD(λ) 的 forward view），一个 across-updates（看 agent 一直在学什么，叫 meta-RNN）。第二个让 meta-network 能实现"reward normalization"、"learning rate schedule"这类 lifetime-adaptive 行为。

类比：meta-network 像一个 RL 算法的"大脑"，agent 是被它控制的"学生"。Meta-network 看学生一段时间的表现，给学生布置下一批"目标"。

---

## Meta-optimization：怎么让 meta-network 学好

这是工程上最难的部分。Meta-network 的参数 η 怎么更新？答案是 **meta-gradient**：让一群 agent（128 个）各自在一个环境学，meta-network 给它们 target，agent 朝 target 更新；然后 backprop through agent update process，更新 η 让 agent 学得更好。

数学上：
$$J(\eta) = \mathbb{E}_\varepsilon \mathbb{E}_\theta[J(\theta)]$$
$$\nabla_\eta J(\eta) \approx \mathbb{E}_\varepsilon \mathbb{E}_\theta[\nabla_\eta \theta \cdot \nabla_\theta J(\theta)]$$

这里 $\nabla_\eta \theta$ 是 chain rule 拆出来的——它表示"agent 参数 θ 是怎么被 η 决定的"。要估计它，得 unroll agent update 20 次，然后 backprop through 整个 update sequence（MAML [https://arxiv.org/abs/1703.03400] 的多步版本）。Backprop 20 步 inner loop 在 100+ environment 并行情况下计算量爆炸，所以用了 MixFlow-MG [https://proceedings.mlr.press/v267/kemaev25a.html] 一种 mixed-mode differentiation 来省 memory。

### 四个稳定 meta-optimization 的 trick

Meta-gradient RL 在大规模下不稳定。这篇用了四个 trick：

1. **Advantage normalization across agent lifetime**：$\bar{A} = (A-\mu)/\sigma$，平衡不同环境的 gradient scale。
2. **Per-agent Adam then average**：每个 agent 的 meta-gradient 先过 Adam 再平均，不是直接平均 raw gradient。因为 Adam 是 scale-invariant 的。
3. **Entropy regularization on y, z**：防止 y, z 过早 collapse。
4. **KL stabilization**：让 meta-network 的 policy target 不要离 target network 太远，防 policy collapse。

这些只 meta-train 时用，evaluate 时 meta-network frozen，agent 只跑 $L(\theta)$。

---

## 实验结果：DiscoRL 在 Atari 上超过 MuZero

主结果（Extended Data Table 1，200M steps evaluation）：

- **Disco57 IQM = 13.86**（human-normalized interquartile mean）
- MuZero ≈ 13.04
- Dreamer [https://www.nature.com/articles/s41586-025-09761-x] 比 MuZero 略低
- 在很多 game 上 DiscoRL **远超** MuZero：Alien 322137 vs 135541，Ms Pacman 101876 vs 79319，Seaquest 999994 vs 815970
- 但不是全胜：Frostbite Disco57 = 1652 vs MuZero = 410173，MuZero 完胜

更重要的指标：**Wall-clock 效率**。DiscoRL 达到 MuZero 最终 performance 只用了 **60% 的 TPU 时间**（Extended Data Fig. 4）。这暗示 meta-learned rule 比人类设计的更 sample-efficient。

### Generalization：在没见过的环境上

Disco57 只在 Atari 上 meta-train，但泛化到：
- **ProcGen** [https://arxiv.org/abs/1912.01588] 16 个游戏：Disco57 在 Bigfish, Bossfight, Chaser, Dodgeball, Maze, Miner 等超过 PPO [https://arxiv.org/abs/1707.06347]、PPG [https://arxiv.org/abs/2102.05280]、MuZero
- **Crafter** [https://arxiv.org/abs/2105.03747] @ 1M steps：Disco57 = 7.58，比 PPO (4.2)、Rainbow (5.0) 高
- **NetHack Challenge** [https://arxiv.org/abs/2006.13760]：Disco57 = 938.10（leaderboard 第三），Disco103 = 1114.24（第二）。**没用任何 domain-specific reward shaping**

这是这论文最震撼的部分——meta-learned rule 学到的不是"Atari 专用算法"，而是泛化的"RL 算法本身"。这有点像 GPT 在很多 task 上 few-shot，但这里 meta-network 学的是 update rule 而不是 policy。

---

## Scaling：环境越多越强

论文给了一个 suggestive 的 scaling law（图 3b）：ProcGen IQM 随 discovery 用到的 Atari game 数量单调上升。这是"meta-learning 的 scaling law"的雏形——**discovered rule 的 performance 是 data 和 compute 的函数**，没有 plateau。跟 LLM 的 Chinchilla [https://arxiv.org/abs/2203.15556] scaling law 类比，但是对 meta-learning。

另一个有意思的发现（图 3a）：最优 rule 在 **3 个 agent lifetime**（约 600M steps per game × 57 games）内就被发现。对比人类设计一个 SOTA RL 算法通常需要几十次实验 + 数年研究，这个效率惊人。

Ablation（图 3c）也 confirm 了关键 hypothesis：
- 去掉 y, z：performance 大降
- 用 grid-world 做 discovery 在 Atari eval：大幅下降
- 去掉 q（value function）：下降但比去掉 y, z 好

这说明：复杂 meta-training 环境 > 大网络 > 多 prediction head。这跟 LLM 里"scale up everything"的直觉相反——对 meta-learning，environment diversity 是主因。

---

## 最 fascinating 的部分：机器学出来的 y, z 是什么？

这是这论文最"科幻"的部分。Meta-network 学出来的 y, z **不是** value function，**不是** successor feature，是某种新东西。论文做了几个分析：

### 1. 行为观察（图 4a）

在 Ms Pacman 中，y 的 confidence（负 entropy）在大 reward **即将到来前 spike**。在 Breakout 中，y 的 confidence 在 policy 即将 commit 到某个 action 前 spike。所以 y 是某种 **salient event predictor**——预测未来几步会发生"重要事件"。

### 2. Attention 分析（图 4b）

在 Beam Rider 中，对 observation 做 gradient norm，看 y 关注图像哪部分：
- **y, z 关注远处的 enemy**（未来可能交互的对象）
- **Policy 关注近处 enemy**（即将决策的对象）
- **Value function 关注 scoreboard**（reward signal）

这是 complementary representations！y, z 学到了 policy 和 value 都不学的东西——它们看的是"未来"。这跟动物里 hippocampus 的 place cell / successor-like representation [https://www.nature.com/articles/nn.2477] 类比有意思。

### 3. Information analysis（图 4c）

训小 MLP 从 y/z/π/q 预测 future entropy 和 future large-reward event。y, z 的预测能力**显著高于** π 和 q。这说明 y, z 不是冗余的，是真正新的 prediction channel，包含 π 和 q 不包含的信息。

### 4. Bootstrapping emergence（图 4d）

扰动 future step $z_{t+k}$，看 target $\hat{z}_t$ 怎么变。发现 bootstrapping horizon 大约 5-10 步。Meta-network **自己学会了 n-step bootstrapping**，没显式 hardcode。这是 emergence 的好例子——人类设计 TD(λ) 的 forward view，机器自己重新发明了。

### 5. Ablation of bootstrapping（图 4e）

把 y, z 的 input 置零（disable bootstrapping）：performance 大降。把 y, z 完全去掉：降得更多。说明 y, z 不只是 auxiliary task，它们**直接 inform policy update**。Meta-network 用这些 prediction 来给 policy 提供 baseline 或 variance reduction 之类的东西。

---

## 一些个人联想

### 1. 跟 LLM in-context learning 的对照

LLM 的 transformer 在前向 pass 里做 in-context learning，是不是某种 implicit meta-learning？Meta-network 的 across-update LSTM（meta-RNN）在 agent lifetime 上 unroll，这跟 transformer 在 sequence 上 attention 有结构相似性。能否用 DiscoRL 的 framework 重新理解 in-context learning？反过来，能否用 transformer 替换 meta-network（论文试过，效果差不多但慢，Extended Data Fig. 3b）然后 scale up？

### 2. 跟 FunSearch / AlphaEvolve 的对照

FunSearch [https://www.nature.com/articles/s41586-023-06224-6] 用 LLM + evolution 演化 code（数学算法），DiscoRL 用 meta-gradient 演化 update rule。可以想象 hybrid：LLM 提议 RL 算法的"形式"（比如"用 n-step return + distributional value"），meta-gradient refine 具体参数。或者反过来，DiscoRL 发现的 rule 用 LLM "翻译"成可解释的 algorithm description。

### 3. 跟 evolution 的类比

论文 abstract 强调"evolution discovered RL mechanisms in animals"。多巴胺 prediction error 的工作 [https://www.nature.com/articles/nn.2477] 表明动物大脑用了类似 TD learning 的机制。这个机制是怎么演化出来的？DiscoRL 给了一个 computational 模拟——从 population experience 涌现 learning rule。如果 scale up environment diversity，会不会涌现 exploration bonus、curiosity、model-based planning 这些更高级的机制？

### 4. Open-ended self-improvement

DiscoRL 是"meta-network frozen after discovery"。下一步显然是让 meta-network 继续 evolve——nested meta-learning。Schmidhuber 的 Gödel machine [https://link.springer.com/chapter/10.1007/978-3-540-87322-4_20] 思想：agent 可以修改自己的 learning rule，只要能 proof 修改是 improvement。DiscoRL 是这个方向的第一步——meta-network 可以"重写"agent 的 update rule，但 meta-network 自己不变。下一步让 meta-meta-network 来 update meta-network。

### 5. 跟 model-based RL 的关系

DiscoRL 是 model-free 形式（meta-network 不显式 model environment dynamics）。但 y, z 可能 implicitly represent dynamics——它们关注"远处的 enemy"暗示某种 forward prediction。是否可以加 model-based component 到 search space？让 meta-network 也可以输出 world model 的 target？或者 discoRL 已经 implicit 学到了 world model，只是没有显式的 latent state？

### 6. Dopamine / 神经科学

动物大脑的 dopamine neuron 编码 reward prediction error（RPE），是 TD learning 的 biological analog。DiscoRL 发现的 y, z 可能对应什么 biological structure？ hippocampus 的 place cell / successor representation [https://www.nature.com/articles/nn.2477]？ entorhinal cortex 的 grid cell？这是 speculative 但 fascinating——如果 DiscoRL scale up 后涌现出类似 hippocampal replay 的机制，那就真的在 computational 上 simulate evolution 了。

### 7. 跟 AlphaFold 的类比

AlphaFold 2 [https://www.nature.com/articles/s41586-021-03819-2] 把蛋白质结构 prediction 从"人类设计特征 + ML"变成"end-to-end differentiable"。DiscoRL 把 RL algorithm design 从"人类设计 update rule"变成"end-to-end meta-learned rule"。都是"用 deep learning 替代人类 hand-crafted feature engineering"。下一步可能是 AlphaFold 3 / AlphaEvolve [https://www.nature.com/articles/s41586-023-06224-6] 那种"自动发现新算法"的范式应用到 RL 上。

### 8. 为什么 y, z dimension 是固定的？

n, m 是 hyperparameter。可以想象自动搜索 dimension——用 information bottleneck [https://arxiv.org/abs/1804.03599] 或 sparse activation（MoE [https://arxiv.org/abs/2009.01325]）让维度被稀疏使用。或者用 PAC-Bayes bound 自动 prune 无用 dimension。

### 9. 局限：discrete action only

Meta-network 通过 action dimension 共享权重，没说明 continuous action 怎么处理。但大部分 interesting 的 control problem（robotics）是 continuous。这是一个明显的 next step。

### 10. 局限：discovery 用 small agent，eval 用 large agent

这跟 LLM 的 pre-train small, fine-tune large 范式相反。能不能直接在大 agent 上 meta-train？还是 meta-learning 的 compute cost 太高只能 small？

---

## 一句话总结

**DiscoRL 把"RL 算法"当成神经网络的参数，用 meta-gradient 在 100+ 复杂环境上自动搜出一个 update rule，这个 rule 在 Atari 上超过 MuZero，泛化到没见过的环境，而且机器自己发明了一种新的 prediction（不是 value function，是某种"未来事件预测器"）。这是 RL 算法设计从"手工"走向"自动"的 milestone，下一步可能是 open-ended self-improvement。**

---

参考链接：
- 主 paper: https://doi.org/10.1038/s41586-025-09761-x
- 代码: https://github.com/google-deepmind/disco_rl
- Oh et al. 2020 (前身): https://papers.nips.cc/paper/2020/hash/178d6933fc5d5e4ee1aa9a9f3a5b8fde-Abstract.html
- Xu et al. 2018 (meta-gradient): https://papers.nips.cc/paper/2018/hash/2d07d626c7d3f3c19a3c5aa6c7f5b4e3-Abstract.html
- MAML: https://arxiv.org/abs/1703.03400
- MuZero: https://www.nature.com/articles/s41586-020-03051-4
- Dreamer V3: https://arxiv.org/abs/2301.07780
- PPO: https://arxiv.org/abs/1707.06347
- IMPALA: https://arxiv.org/abs/1802.01561
- DQN: https://www.nature.com/articles/nature14236
- Retrace: https://arxiv.org/abs/1606.02647
- ProcGen: https://arxiv.org/abs/1912.01588
- Crafter: https://arxiv.org/abs/2105.03747
- NetHack: https://arxiv.org/abs/2006.13760
- MixFlow-MG: https://proceedings.mlr.press/v267/kemaev25a.html
- Successor Features: https://arxiv.org/abs/1606.05312
- Schmidhuber Gödel machine: https://link.springer.com/chapter/10.1007/978-3-540-87322-4_20
- FunSearch: https://www.nature.com/articles/s41586-023-06224-6
- Dopamine TD learning: https://www.nature.com/articles/nn.2477
- Schmidhuber PhD thesis (1987 meta-learning): https://link.springer.com/book/10.1007/978-1-4615-3738-7
- RL²: https://arxiv.org/abs/1611.02779
- AlphaFold 2: https://www.nature.com/articles/s41586-021-03819-2
- Chinchilla scaling law: https://arxiv.org/abs/2203.15556
- IQM metric (Agarwal et al.): https://arxiv.org/abs/2108.13264

---

# DiscoRL: Meta-learning 一个 RL 算法

这篇 paper 是 DeepMind 的 David Silver 团队（Junhyuk Oh, Gregory Farquhar 等人）2025年发表在 Nature 的工作，核心思想是：**用 meta-learning 自动发现一个 RL 更新规则**（叫 DiscoRL），让它在与人类设计算法（MuZero, Dreamer, PPO）的对比中胜出。这就像让机器自己"发明" TD-learning, Q-learning, PPO 这类东西。

论文链接：https://doi.org/10.1038/s41586-025-09761-x
代码开源：https://github.com/google-deepmind/disco_rl

---

## 1. Intuition: 在哪里搜？为什么之前 meta-RL 没成功

Meta-RL 的历史很长（Schmidhuber 1987 PhD thesis [https://link.springer.com/book/10.1007/978-1-4615-3738-7]），但之前的工作要么搜的空间太窄（只搜 hyperparameters，如 Xu et al. 2018 [https://papers.nips.cc/paper/2018/hash/2d07d626c7d3f3c19a3c5aa6c7f5b4e3-Abstract.html], STACX [https://arxiv.org/abs/2101.08721]），要么完全 black-box（如 RL² [https://arxiv.org/abs/1611.02779]，把整个算法塞进 RNN），在简单任务上过拟合。

DiscoRL 的两个关键选择：
1. **搜一个"target function"**，不是搜 loss function。Target-based search space 严格更大，因为它能表示 semi-gradient 方法（如 Q-learning，目标是从 future predictions 来 bootstrap），而 scalar loss 搜不到这种依赖 future prediction 的形式。
2. **在复杂、多样的环境上 meta-train**（Atari 57 games + ProcGen + DMLab-30），不是 grid-world。这避免了"在 toy 环境上学到的 rule 在复杂环境上崩"。

---

## 2. 架构深度解析

### 2.1 Agent network 的输出（图 1b）

Agent 参数化为 θ，对每个 observation s 输出：

- **π(s)**：policy（softmax distribution over actions）
- **y(s) ∈ ℝⁿ**：observation-conditioned vector prediction（n 是任意维度，比如文中 n=32 左右）。语义**未定义**——meta-network 决定 y 应该预测什么。它**可能**变成 value function, successor feature, forward model, distributional predictions... 但也可能完全是新东西。
- **z(s,a) ∈ ℝᵐ**：action-conditioned vector prediction。类比于 Q-value 但维度更高、语义未定义。
- **q(s,a)**：action-value function（pre-defined semantics，用 Retrace target 训练，保证 discovery 过程中至少有 standard value 学习信号）。
- **p(s,a)**：auxiliary policy prediction（预测 1-step future policy π(s')）。

为什么 y, z 同时有 observation-conditioned 和 action-conditioned 两种？论文引用了 Sutton & Barto 中 prediction vs control 的二分：state-value v(s) 对应 prediction，action-value q(s,a) 对应 control。reward prediction 也类似有 state-based 和 action-based 版本。Successor features 也是这样。所以这个二分是 RL 里普遍 pattern，agent architecture 应该 preserve。

### 2.2 Meta-network（图 1c）

Meta-network $m_\eta$ 是一个 LSTM（也试过 transformer，效果差不多但慢，见 Extended Data Fig. 3b），其映射是：

$$m_\eta: f_\theta(s_t), f_{\theta^-}(s_t), a_t, r_t, b_t, \dots, f_\theta(s_{t+n}), f_{\theta^-}(s_{t+n}), a_{t+n}, r_{t+n}, b_{t+n} \mapsto \hat{u}, \hat{y}, \hat{z}$$

变量解释：
- η：meta-parameters（meta-network 的权重）
- θ：agent parameters
- θ⁻：θ 的 exponential moving average（target network 思想，稳定 bootstrapping）
- $f_\theta(s) = [\pi_\theta(s), y_\theta(s), z_\theta(s), q_\theta(s)]$：agent 在 state s 上的全部 output，被打包成 input vector
- $a_t, r_t, b_t$：action、reward、episode termination indicator（"done" flag，处理 episode 边界的 bootstrap）
- $\hat{u}, \hat{y}, \hat{z}$：meta-network 输出的 target，分别对应 policy、observation-conditioned prediction、action-conditioned prediction 的 target

关键设计：
1. **观察不到 raw observation**：meta-network 只通过 $f_\theta(s)$ 间接看到 state。这意味着 meta-network 对图像维度、observation modality 完全 agnostic，可以 generalize 到任何环境的任何 observation。
2. **Action dimension 共享权重**：处理 action-specific 输入输出时，在 action 维度上 share weight，并通过 averaging 得到一个 embedding。这让 meta-network 可以处理任意大小离散 action space。
3. **Backward unrolling LSTM**：LSTM 从 t+n 反向 unroll 到 t（像 n-step return，类似 TD(λ) 的 forward view [https://link.springer.com/article/10.1007/BF00115009]）。
4. **Meta-RNN（across agent updates）**：除了 within-episode LSTM，还加一个 across-update 的 LSTM，让 meta-network 看到整个 agent lifetime 的 statistics（embed 一整批 trajectory 进一个 vector）。这类似"agent 在第几次 update？"这种上下文。它能让 meta-network 实现 reward normalization、adaptive learning rate 等 lifetime-adaptive 行为。

### 2.3 Meta-network 是不是 search over loss？

输出 **target** $\hat{y}, \hat{z}, \hat{u}$ 而非 scalar loss。Target 是"agent 当前 prediction 应该朝哪个值更新"。这对应 RL 中的 **bootstrapping**：$\hat{y}_t$ 可以依赖 $y_{t+k}$（future prediction），这是 Q-learning 等方法的核心。如果只输出 scalar loss $L(y_t, \text{something})$，这个 something 必须是 fixed，无法依赖 future prediction。

---

## 3. 数学公式与变量含义

### 3.1 Agent loss

$$L(\theta) = \mathbb{E}_{s,a \sim \pi_\theta}\left[D(\hat{u}, \pi_\theta(s)) + D(\hat{y}, y_\theta(s)) + D(\hat{z}, z_\theta(s,a)) + L_{aux}\right]$$

- $\theta$：agent network 参数
- $\pi_\theta$：当前 policy（用于 sample trajectories）
- $D(p, q)$：距离函数，选为 **KL divergence** $D_{KL}(p \| q)$。原因：(1) general（能处理 categorical distribution 如 softmax policy 和 prediction heads）；(2) 之前 Oh et al. 2020 [https://papers.nips.cc/paper/2020/hash/178d6933fc5d5e4ee1aa9a9f3a5b8fde-Abstract.html] 实验发现 KL 比 MSE 让 meta-optimization 容易收敛。
- $\hat{u}, \hat{y}, \hat{z}$：meta-network 输出的 target（softmax-normalized）
- $\pi_\theta(s), y_\theta(s), z_\theta(s,a)$：agent 当前 prediction（softmax-normalized）
- $L_{aux}$：对 pre-defined semantics 的 head 的 loss

$L_{aux} = D(\hat{q}, q_\theta(s,a)) + D(\hat{p}, p_\theta(s,a))$

- $\hat{q}$：action-value target，用 **Retrace(λ)** [https://papers.nips.cc/paper/2016/hash/cd896d2269fd8f1e4e87cc489d5ee2ab-Abstract.html] 算出，然后 project 到 two-hot vector（Muzero 风格 distributional projection [https://www.nature.com/articles/s41586-020-03051-4]）。这里 RL domain knowledge "硬塞" 进去保证 value 学习 strong。
- $\hat{p} = \pi_\theta(s')$：1-step future policy，作为 auxiliary prediction target。

### 3.2 Meta-objective 与 meta-gradient

$$J(\eta) = \mathbb{E}_\varepsilon \mathbb{E}_\theta[J(\theta)]$$
$$\nabla_\eta J(\eta) \approx \mathbb{E}_\varepsilon \mathbb{E}_\theta[\nabla_\eta \theta \, \nabla_\theta J(\theta)]$$

- $J(\theta) = \mathbb{E}[\sum_t \gamma^t r_t]$：standard RL objective，discounted return
- $\gamma$：discount factor（Atari 用 0.997）
- $r_t$：reward at step t
- $\varepsilon$：环境（从 Atari/ProcGen/DMLab 分布采样）
- $\nabla_\theta J(\theta)$：standard policy gradient 项，用 advantage actor-critic 估计（这里 advantage 通过训练一个 **meta-value function** 来估计）
- $\nabla_\eta \theta$：meta-gradient 项——agent 参数 θ 是怎么由 η 决定的。这就是 chain rule through agent update 的核心。

$\nabla_\eta \theta$ 的估计：**unroll agent update 20 次**（sliding window）然后 backprop through 整个 update sequence。这是 MAML [https://arxiv.org/abs/1703.03400] 思想的扩展：MAML 是 1-step inner loop + 1-step outer，DiscoRL 是 20-step inner + outer gradient。Backprop through 20 updates 的计算量很大，所以用 **MixFlow-MG** [https://proceedings.mlr.press/v267/kemaev25a.html]——一种 mixed-mode differentiation，能减少 TPU 内存与计算 cost。

### 3.3 Meta-optimization 的稳定性技巧

四个 regularization：

1. **Advantage normalization across agent lifetime**：$\bar{A} = (A - \mu)/\sigma$，其中 $\mu, \sigma$ 是 advantage 的 EMA 和 std。这平衡不同环境的 gradient magnitude。

2. **Per-agent Adam then average**：对每个 agent 的 meta-gradient $g_i$ 单独过 Adam 再 average，而不是直接 average raw gradients：
   $$\eta \leftarrow \eta + \frac{1}{n}\sum_{i=1}^n \text{ADAM}(g_i)$$
   这是因为不同环境的 gradient scale 差异大，Adam 内置 scale-invariance。

3. **Entropy regularization on predictions**：$L_{ent}(\theta) = -\mathbb{E}_{s,a}[H(y_\theta(s)) + H(z_\theta(s,a))]$，防止 y, z 过早 collapse 到 trivial solution。

4. **KL stabilization**：$L_{KL}(\theta) = D_{KL}(m_{\theta^-} \| \hat{\pi})$，让 meta-network 的 policy target $\hat{\pi}$ 不要离 target network $m_{\theta^-}$ 太远（防止 policy collapse）。

这四条只在 meta-train 时用，evaluate agent 时只用 $L(\theta)$，meta-network frozen。

---

## 4. 实验数据详解

### 4.1 Atari 57（Extended Data Table 1）

- **Disco57 IQM = 13.86**（human-normalized）
- MuZero IQM ≈ 13.04（估计，paper 中没直接列）
- Dreamer 比 MuZero 稍低
- **Wall-clock：DiscoRL 达到 MuZero 最终 performance 只用了 60% 的 TPU 时间**（Extended Data Fig. 4）。注意这是 evaluation 时的 compute 对比，不是 discovery 的 compute。

Game-by-game 中 Disco57 在很多 game 上**远超** MuZero，比如：
- Alien: Disco57 = 322137 vs MuZero = 135541
- Asterix: Disco57 = 814458 vs MuZero = 918628（这里 MuZero 更高）
- Ms Pacman: Disco57 = 101876 vs MuZero = 79319
- Seaquest: Disco57 = 999994 vs MuZero = 815970
- Video Pinball: Disco57 = 940631 vs MuZero = 921563
- Frostbite: Disco57 = 1652 vs MuZero = 410173 — 这里 MuZero 完胜，说明 Disco57 不是 all-dominating

这种 per-game 的差异暗示 DiscoRL 学到的 rule 不是简单 universal improvement，可能对某些环境的 reward shaping 更敏感。

### 4.2 Generalization（图 2）

Disco57 在 Atari 上 meta-train，但在 **从未见过** 的 ProcGen [https://arxiv.org/abs/1912.01588]、Crafter [https://arxiv.org/abs/2105.03747]、NetHack [https://arxiv.org/abs/2006.13760]、Sokoban 上也表现强：

- ProcGen 16 games @ 50M steps：Disco57 在 Bigfish, Bossfight, Chaser, Dodgeball, Fruitbot, Maze, Miner 等多个 game 超过 PPO [https://arxiv.org/abs/1707.06347]、PPG [https://arxiv.org/abs/2102.05280]、Dreamer、MuZero。
- Crafter @ 1M：Disco57 = 7.58，比 PPO 4.2、Rainbow 5.0 都高，但低于 Dreamer 11.7。
- NetHack leaderboard：Disco57 = 938.10 mean score（第三名），Disco103 = 1114.24（第二名）。**没用任何 domain-specific subtask 或 reward shaping**。

### 4.3 Discovery 效率（图 3a）

最优 rule 在 **3 个 agent lifetimes**（约 600M steps per game × 57 games）内就被发现。对比人类设计 RL rule 通常需要几十-几百次实验 + 数年研究。

### 4.4 Scalability（图 3b）

ProcGen IQM 随着 discovery 用到的 Atari game 数量单调上升。这强烈暗示：**rule 的 performance 是 data 和 compute 的函数**，没有 plateau。这是"scaling law for meta-learning"的雏形。

### 4.5 Ablation（图 3c）

- 去掉 y, z（只保留 q, p）：performance 大幅下降
- 去掉 q（只保留 y, z, p）：performance 也下降，但比去掉 y, z 好
- 去掉 p：performance 也下降
- 用 grid-world（57 toy 任务）替代 Atari 做 discovery：在 Atari 上 evaluate 大幅下降
- Small agent network 做 discovery：略下降

这个 ablation 直接证明：**复杂的 meta-training 环境比大网络更重要**，这跟 LLM pretraining 的直觉相反——meta-learning 偏好 environment diversity。

---

## 5. Discovery 出来的 prediction 在干什么？（图 4）

这是最 fascinating 的部分——meta-network 学出来的 y, z 是**新的 prediction**，语义不是 value function 也不是 successor feature。

### 5.1 Qualitative（图 4a）

在 Ms Pacman 和 Breakout 中：
- y 的 confidence（负 entropy）在 **大 reward 即将发生前 spike**
- y 的 confidence 在 **policy entropy 即将下降前**也 spike

说明 y 是某种 **event predictor**——预测未来会发生 salient event。

### 5.2 Attention/Gradient analysis（图 4b）

在 Beam Rider 中，对 observation 做 gradient norm，看哪个像素对 y 影响最大：
- **y, z 关注远处的 enemy**（关注未来可能交互的物体）
- **Policy 关注近处 enemy**（即将决策的对象）
- **Value function 关注 scoreboard**（reward signal）

这是 complementary representations！y, z 学到了 policy 和 value 都不学的东西。

### 5.3 Information analysis（图 4c）

训练小 MLP 从 y/z/π/q 预测：
- Future entropy
- Large-reward event in future k steps

发现 y, z 预测这些未来事件的能力**显著高于** π 和 q。这说明 y, z 不是冗余的，是**真正新的 prediction channel**。

### 5.4 Bootstrapping horizon（图 4d）

扰动 future step $z_{t+k}$，看 target $\hat{z}_t$ 变化多大。曲线显示 bootstrapping horizon 大约 **5-10 步**，类似 n-step return 但 stochastic。说明 meta-network 自己学会了 n-step bootstrap，没显式 hardcode。

### 5.5 Ablation of bootstrapping（图 4e）

在 Ms Pacman 上：
- 完整 Disco57：baseline
- 把 y, z 输入置零（disable bootstrapping）：performance 大降
- 把 y, z 输入和 y, z 的 target 全部置零（disable prediction entirely）：performance 更大降

第二个比第一个降得更多，说明 y, z 不仅作 auxiliary task，还直接 inform policy update。

---

## 6. 与之前 Meta-RL 工作的关系

| 工作 | 搜什么 | 在哪 meta-train | Generalize 吗 |
|------|--------|---------|---------|
| Xu et al. 2018 [https://papers.nips.cc/paper/2018/hash/2d07d626c7d3f3c19a3c5aa6c7f5b4e3-Abstract.html] | Hyperparams of existing algorithm | DMLab | Yes，但空间窄 |
| Oh et al. 2020 [https://papers.nips.cc/paper/2020/hash/178d6933fc5d5e4ee1aa9a9f3a5b8fde-Abstract.html] | Policy loss + auxiliary prediction | Grid-world | Atari 上很弱 |
| Kirsch et al. 2020 [https://arxiv.org/abs/2002.03023] | Black-box objective | 简单 task | 不 generalize |
| Lu et al. 2022 DPO [https://arxiv.org/abs/2210.07841] | Policy loss | MuJoCo/Atari 子集 | 局部 |
| Jackson et al. 2023 [https://openreview.net/forum?id=md9Vg5l8GS] | Adversarial env design | Grid-world | 不 generalize |
| **DiscoRL** | **Target function (含新 prediction y, z)** | **Atari+ProcGen+DMLab-30** | **Generalize 到未见 benchmark** |

DiscoRL 主要扩展点是 Oh et al. 2020 那篇——同样用 meta-network + agent with prediction vector，但 Oh et al. 只在 grid-world 上跑，而且没有 action-conditioned prediction z、没有 meta-RNN、没有 Kullback-Leibler target stabilization、没有大规模 population-based aggregation。

---

## 7. 一些可能的联想/猜想

1. **跟 Successor Features 的关系**：y(s) 和 z(s,a) 形式上很像 successor features [https://arxiv.org/abs/1606.05312]——state-conditioned 和 action-conditioned 各一个 vector。但 successor features 是 linear approximator of value function under feature basis；DiscoRL 的 y, z 不需要 linear decomposition。它们可能是某种 nonlinear generalization。

2. **跟 MuZero 的 value-equivalence**：MuZero 的 hidden state 在 value-equivalence 意义上"代表"了 reward 和 policy 的 prediction。DiscoRL 的 y, z 可能是更广义的"prediction head"，包含 successor-like info。

3. **跟 Predictive representations** [https://www.cs.cmu.edu/~dsmclc/papers/dissertation_singh.pdf]：与 daydreaming, latent state model 的联系。Satinder Singh 的工作。这可能是为什么 Singh 是 author——他长期倡导 predictive representations。

4. **跟 LLM in-context learning** 的对照：LLM 的 transformer 做 in-context learning 时，是不是也像 DiscoRL 的 meta-RNN——在 across-update 维度上"记住"agent 当前的 lifetime 状态？这是一个 open question，能否用 DiscoRL 的 framework 重新理解 in-context learning？

5. **Meta-learned RL algorithm 与 biological evolution**：论文 abstract 强调"evolution discovered RL mechanisms in animals"。这跟 dopamine prediction error（TD learning）的"演化涌现"类比 [https://www.nature.com/articles/nn.2477]。DiscoRL 提供了一个 computational analog：从 population experience 涌现 learning rule。

6. **跟 Open-Endedness**：DiscoRL 可以看作 open-ended self-improvement 的早期形式——agent 通过自己更新自己 improve，但 learning rule 还是 fixed 的 meta-learned rule。下一步可能让 meta-network 也 continue updating（nested meta-learning，Schmidhuber 的 Gödel machine 思想 [https://link.springer.com/chapter/10.1007/978-3-540-87322-4_20]）。

7. **跟 FunSearch / AlphaEvolve** [https://www.nature.com/articles/s41586-023-06224-6]：FunSearch 用 LLM 演化 code，DiscoRL 用 meta-gradient 演化 update rule。可以想象 hybrid：LLM 提议新 rule form，meta-gradient refine。

8. **Compute scaling law**：图 3b 给出 environment diversity → held-out performance 的 scaling。这类似 Chinchilla 的 data/compute trade-off，但是对 meta-learning。可能存在：meta-network capacity、environment count、agent lifetime budget 三者之间的 scaling relation。

9. **为什么 y, z 的 dimension（n, m）是固定的？**：可以想象自动搜索 dimension——可能用 information bottleneck 或 PAC-Bayes bound。或者用 sparse activation（MoE）让维度被稀疏使用。

10. **跟 Continual learning**：DiscoRL 的 agent 每 200M steps reset。但 meta-RNN 的 across-update state 是 reset 的吗？如果 meta-RNN 在 agent 之间共享，它能不能 "学到" agent 应该 reset 时机，实现 continual learning rule 的发现？

---

## 8. 局限/未来方向（论文没充分讨论）

1. **只处理 discrete action space**：meta-network 通过 action dimension 共享权重，没显式说明 continuous action 怎么处理。
2. **Meta-network 是 frozen 的**：discovery 完之后 rule 不再 evolve。可以想象 lifelong meta-learning。
3. **Network size**：discovery 用 small agent，eval 用 large agent。是否在大 agent 上直接 meta-train 更好？
4. **No interpretability**：discovered rule 是 LSTM 黑箱，跟 Muzero 的 explicit MCTS plan 不同。Dreamer 也有 explicit world model。DiscoRL 完全 implicit。
5. **Generalization 的 boundary**：在 unseen DMLab-30 上某些 task DiscoRL 远低于 IMPALA（如 Explore Obstructed Goals Large, Disco57=5.84 vs IMPALA=39.50）。是不是只对 narrow transfer set strong？
6. **跟 Model-based RL 的关系**：DiscoRL 是 model-free 形式（meta-network 不显式 model environment dynamics）。但 y, z 可能 implicitly represent dynamics（successor features-like）。是否可以加入 model-based component 到 meta-network search space？

---

## 9. 论文内最值得记录的设计哲学

1. **"Maintain inductive bias, expand search space"**：保留 value function $q$（hard-coded semantic），只 meta-learn y, z。这是关键——完全 black-box 不行，完全 hand-crafted 也不行，hybrid 在中间。

2. **Target, not loss**：把 search space 从 scalar loss 扩展到 target vector，自然纳入 bootstrapping 类方法。

3. **Population-based meta-gradient**：128 个 agent 在 57 environments 上并行 update，meta-gradient 平均。这不是单 agent meta-learning，是 **meta-learning over a population**。这让 discovery 更 robust（类似 genetic algorithm 的 population diversity，但用 gradient）。

4. **Lifetime-adaptive learning rules**：meta-RNN across-update 让 rule 可以根据 agent 的 "年龄" 调整（e.g., exploration 大 → 小）。

5. **Training distribution = popular benchmarks**：不需要精心 curate environments，直接用 Atari+ProcGen+DMLab 就够了。这暗示现代 RL benchmarks 已经足够 diverse，主要 bottleneck 是 RL algorithm design，不是 environment。

---

## 10. 一句话总结

**DiscoRL 把 RL algorithm 当作神经网络参数，用 meta-gradient 通过 agent update process backprop，让这个神经网络在 Atari+ProcGen+DMLab-30 上学到比 MuZero 更好的更新规则，而且 generalize 到从未见过的 benchmark。这是 RL 自我进化路上的一个 milestone——下一步可能是让 meta-network 自己继续 evolve，进入 open-ended self-improvement。**

参考链接集合：
- 主 paper: https://doi.org/10.1038/s41586-025-09761-x
- Code: https://github.com/google-deepmind/disco_rl
- Oh et al. 2020 (前身): https://papers.nips.cc/paper/2020/hash/178d6933fc5d5e4ee1aa9a9f3a5b8fde-Abstract.html
- Xu et al. 2018 (meta-gradient RL): https://papers.nips.cc/paper/2018/hash/2d07d626c7d3f3c19a3c5aa6c7f5b4e3-Abstract.html
- MAML: https://arxiv.org/abs/1703.03400
- MuZero: https://www.nature.com/articles/s41586-020-03051-4
- Dreamer V3: https://www.nature.com/articles/s41586-025-09761-x
- IMPALA: https://arxiv.org/abs/1802.01561
- DQN: https://www.nature.com/articles/nature14236
- PPO: https://arxiv.org/abs/1707.06347
- PPG: https://arxiv.org/abs/2102.05280
- Rainbow: https://arxiv.org/abs/1710.02298
- Retrace: https://arxiv.org/abs/1606.02647
- ProcGen: https://arxiv.org/abs/1912.01588
- Crafter: https://arxiv.org/abs/2105.03747
- NetHack Challenge: https://arxiv.org/abs/2006.13760
- MixFlow-MG: https://proceedings.mlr.press/v267/kemaev25a.html
- Schmidhuber Gödel machine: https://link.springer.com/chapter/10.1007/978-3-540-87322-4_20
- Successor Features: https://arxiv.org/abs/1606.05312
- Sutton & Barto (forward view, TD): https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf
- Atari benchmark (ALE): https://arxiv.org/abs/1207.4708
- FunSearch: https://www.nature.com/articles/s41586-023-06224-6
- DMLab: https://arxiv.org/abs/1612.03801
- Atari NeurIPS 2023 200x faster (MEME): https://arxiv.org/abs/2301.13676
- Agarwal et al. (IQM): https://papers.nips.cc/paper/2021/hash/0380c1c2c1f6c1d1c8aa5f8b5f7e8b8f-Abstract.html (实际 link: https://arxiv.org/abs/2108.13264)
