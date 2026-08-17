---
source_pdf: Logic-RL.pdf
paper_sha256: 8a8a3994c51bc80969eddc5297a7b8620002aaae1e2601e0ad6a9e85195e5d0a
processed_at: '2026-08-05T15:46:50-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Logic-RL 大白话版

## 一句话讲清楚

DeepSeek-R1 说"我们靠 RL 让模型自己学会推理了"，但没放代码没放数据，大家都在猜他们到底怎么搞的。这篇 paper 的做法是：**找 5000 道逻辑题，让 7B 小模型在上面做 RL，结果小模型也自己学会了"等等我再想想"这种行为，还能跨到数学题上。** 本质上就是 R1 的 reduced reproduction，把黑盒变白盒。

paper: https://arxiv.org/abs/2502.14768
code: https://github.com/Unakar/Logic-RL

---

## 为什么用 Knights and Knaves 谜题

想象岛上只有两种人：骑士永远说真话，无赖永远说假话。给你几句话，让你推断谁是谁。这玩意儿好就好在：

- 可以程序生成，要多少有多少
- 难度可以精确控制（几个人 + 多少个布尔操作符）
- 答案只有一个，不用搞 subjective scoring

数学题为啥不行？GSM8K 那种题，有的简单有的难，难度不可控，做实验就像在噪声里找信号。K&K 就像物理实验里的 Ising model——简化到极致，但能暴露现象。

举个最简单的例子：

> Zoey 说："Oliver 不是骑士"
> Oliver 说："Oliver 是骑士 当且仅当 Zoey 是无赖"

这个题，如果 Zoey 是骑士（说真话），那 Oliver 是无赖。那 Oliver 那句话就是假的。"Oliver 是骑士 iff Zoey 是无赖"——Oliver 不是骑士（True iff False = False），自洽。所以答案：Zoey 骑士，Oliver 无赖。

---

## RL 的核心：Reward 怎么设计

RL 就是"做对了给糖，做错了挨打"。这里的 reward 就两部分：

### Format Reward（格式分）

你必须在 `...` 里写思考过程，在 `<answer></answer>` 里写答案。格式对给 +1，错给 -1。

听起来很 trivial，但这是防止 reward hacking 的第一道防线。作者列了一堆 hack 行为：
- 直接跳过 `

---

# Logic-RL: 用 Rule-Based RL 在 7B 模型上重现 R1 风格推理

## 0. 论文的 Core Thesis

这篇 paper 来自 Microsoft Research Asia 等团队，核心论点：**只用 5,000 条程序化生成的 Knights & Knaves (K&K) 逻辑谜题，通过严格的 rule-based RL，一个 7B 模型就能自发涌现 reflection、verification、summarization 等 reasoning 行为，并且 cross-domain generalize 到 AIME (+125%) 与 AMC (+38%)。** 这是对 DeepSeek-R1 的一个 controlled reproduction study，把"R1-zero 风格训练"从黑盒拉回到可复现的实验框架里。

arXiv 链接（推断）: https://arxiv.org/abs/2502.14768
GitHub (Logic-RL): https://github.com/Unakar/Logic-RL
相关 K&K dataset: https://github.com/RealismLastLab/Knights-and-Knaves-puzzles
DeepSeek-R1 paper: https://arxiv.org/abs/2501.12948
DeepSeekMath (GRPO 原始): https://arxiv.org/abs/2402.03300
REINFORCE++: https://arxiv.org/abs/2501.03262

---

## 1. 为什么是 K&K Puzzles，不是 GSM8K？

Karpathy 你应该会 appreciate 这一点——作者明确指出 GSM8K / Omni-MATH 这类数学数据集"uncontrolled variance in problem complexity"，无法做严格的 ablation。K&K 谜题有三个 property：

1. **Procedural generation**：模板化生成，无穷多样性，对 base model 而言必然是 unseen。
2. **Controllable difficulty**：两个 dial —— 角色数 N ∈ [2, 8]，boolean operator 复杂度 ∈ [1, 4]。这让 curriculum learning 和 OOD 评估都可量化。
3. **Verifiable**：唯一 ground truth，reward 不会被 hack 到"看起来对"的答案。

**Intuition**：作者想要的是一个"reduced-form lab setting"。K&K 像物理实验里的 Ising model——足够简单以便分析，足够丰富以暴露现象。GSM8K 是"real world"，但它 noise 太多，做不了 R1 复现实验这种 science。

### Example (from paper Section 2.1)

```
Problem: Zoey remarked, "Oliver is not a knight". 
         Oliver stated, "Oliver is a knight iff Zoey is a knave".
Solution: (1) Zoey is a knave (2) Oliver is a knight
```

形式化上，每个角色 i 有 latent variable $z_i \in \{0,1\}$（knave / knight），每条 statement $s$ 是一个 boolean 表达式 $f_s(z_1, ..., z_n)$。约束：若说话者 $i$ 是 knight，则 $f_s$ must be true；若为 knave，则 $f_s$ must be false。求解 = SAT assignment。难度 = 变量数 + operator depth。

---

## 2. Reward 设计：防 Reward Hacking 是 First-Class Concern

这是 paper 里最有"工程血泪味"的部分。作者列出了一长串观察到的 hack 行为：

- 跳过 ` `
