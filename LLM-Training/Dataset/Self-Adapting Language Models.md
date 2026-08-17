---
source_pdf: Self-Adapting Language Models.pdf
paper_sha256: 3abf83bcf05862d5e6e72078dc0e47ebfcf2e72866e9fdfe1dbc3663c72a9dc6
processed_at: '2026-08-12T04:34:36-07:00'
target_folder: LLM-Training/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，用"人话"再讲一遍，抛开公式，直接讲 intuition。

## 一句话 version

让 LLM 自己学会"给自己写学习笔记"，然后用 RL 告诉它"哪种笔记对你考试最有用"。

## 1. 为什么这个 paper 存在

现在 LLM 有个尴尬的问题: 你给它一段新材料, 比如一篇 Wikipedia 文章, 它能**读懂**, 能在 context 里回答问题, 但一旦把 context 拿走, 它就忘了。这就像学生看了课本能应付开卷考试, 闭卷就歇菜。

传统解法是 finetuning —— 直接把原文拿去 SFT。但原文不是好的训练材料, 它信息密度不均匀, 有冗余, 有废话, 格式也不一定适合模型吸收。这就像让学生把教科书一字不落背下来, 效率极低。

那人类怎么办? 人类会做笔记。把一篇文章读完后, 提炼出关键 facts, 改写成自己的话, 画成图, 列成 bullet points。然后复习这些 notes 而不是原文。效果出奇的好。

**SEAL 就是让 LLM 学会做这件事。**

## 2. 整个 loop 长什么样

想象一个学生(LLM)坐在书桌前:

1. 老师给一篇文章 (passage $C$)
2. 学生写一份"学习笔记" (self-edit $\mathrm{SE}$) —— 可能是 implications 列表, 可能是 QA pairs, 可能是改写
3. 学生把这份笔记拿去自己 SFT 一遍 (inner loop, 用 LoRA, 便宜)
4. SFT 完之后, 给学生闭卷考一下 (task $\tau$, 比如 SQuAD 问题)
5. 考得好, reward = 1; 考砸, reward = 0
6. 用 RL outer loop 强化"写出好笔记"的能力

跑很多轮之后, 学生就学会了"什么样的笔记对我自己最有效"。这个"对我自己"是关键 —— 每个模型的弱点不一样, 适合的笔记也不一样。

## 3. 为什么不用 PPO / GRPO

paper 试过 PPO 和 GRPO, 都不稳定。原因挺微妙的:

普通 RL 里, reward 是环境给的, 跟 policy 参数无关。比如下棋, 你赢了就是赢了, 跟你神经网络权重多少没关系。

但 SEAL 里, reward 是"拿 SE 去 SFT 后考多少分", 这个 SFT 是在当前 $\theta$ 上做的。$\theta$ 变了, 同样的 SE 效果就变了。这就像"笔记好不好"取决于"你已经掌握了什么", 而你已经掌握的东西一直在变。

这种 reward 依赖 policy state 的情况, PPO/GRPO 用旧数据算的 advantage 是 stale 的。就像你 100 分时写的笔记, 等你 120 分时再看就不一定有用了。

ReST^EM 简单粗暴: 每次都用当前 model 重新 sample, 考好的留下当 SFT data, 考差扔掉。等价于 "best-of-N + SFT"。虽然 sample 效率不如 PPO, 但胜在 stable。

## 4. Knowledge Incorporation 实验

用 Qwen-7B, SQuAD 数据集, 任务是"读一段 passage, 然后不带 passage 回答问题"。

几个 baseline 对比:
- **直接 finetune 原文**: 33.5% (基本没用)
- **让 Qwen 自己 generate implications, 再 SFT**: 39.7% (有点用)
- **让 GPT-4.1 generate implications, 再 SFT**: 46.3% (很好)
- **SEAL (RL 训练后)**: 47.0% (最好)

最 surprising 的是 SEAL 超过 GPT-4.1。Qwen-7B 比 GPT-4.1 小一个量级, 但它生成的 synthetic data 反而更好用。Intuition 是: **GPT-4.1 写的是"人类觉得合理的笔记", Qwen-7B 通过 RL 学会写的是"Qwen-7B 自己学起来最舒服的笔记"。** 这俩其实可以是完全不同的东西。就像你自己的错题本, 比学霸的错题本对你更有用, 因为学霸不懂你哪里不会。

## 5. ARC Few-shot 实验

ARC 是抽象推理任务, 给几个 input-output 例子, 预测新 input 的 output。

这里 self-edit 变成了"训练 config", 一段 JSON:
```json
{
  "data_generation": {
    "use_basic_augmentations": true,
    "use_size_augmentations": true,
    ...
  },
  "training": {
    "learning_rate": 1e-4,
    "num_train_epochs": 2
  }
}
```

Model 决定: 用哪些 augmentation, 用什么 lr, 训几个 epoch, loss 算 all tokens 还是只算 output tokens。

结果:
| Method | 成功率 |
|---|---|
| ICL (直接 in-context 推理) | 0% |
| TTT 但不 RL (随便选 config) | 20% |
| SEAL | 72.5% |
| Oracle (人类最优 config) | 100% |

从 20% 到 72.5% 说明 model 真的学会了"怎么训自己"。但离 Oracle 100% 还有差距, 说明还有很多 config 它没探索到。

## 6. CPT 那个反直觉的结果

在 single-passage (一次只学一段) 上 SEAL 很强。换成 CPT (200 篇文章一起 SFT), SEAL 也涨到 58.2%。按理说应该更难啊?

原因: CPT 时每篇文章 sample 5 个 self-edits, aggregate 起来训。5 次采样等于 5 个不同角度的笔记, 覆盖更全。这就像 5 个同学一起做笔记, 比一个人做的笔记更完整。

但 n=2067 时 SEAL 又掉到 46.4%, 被 GPT-4.1 反超。这暗示 single-passage policy 在大规模时 generalization 受限, 不同文章需要的 self-edit 风格可能不一样。

## 7. Catastrophic Forgetting

连续给 8 篇文章, 每篇学一次, 测前面学过的还记不记得。结果是典型的 forgetting curve —— 早期文章的准确率随时间下滑。

这说明 SEAL 目前没解决 continual learning 问题。Paper 提了几个方向: reward shaping 惩罚遗忘, null-space constrained update (AlphaEdit), 或者 inner loop 用 RL 替代 SFT (RL 比 SFT 忘得少, 参考 RL's razor)。

## 8. 几个有意思的设计选择

**为什么用 LoRA**: 750 次 inner loop SFT 一轮, 全量 finetune 根本跑不动。LoRA rank 32~64 够了。

**为什么用 binary reward**: ReST^EM 只支持 binary (SFT on good = SFT on r=1)。如果用 soft reward, 就得回 PPO 那一套, 又回到 instability 问题。这是 simplicity vs signal density 的 trade-off。

**Proxy reward 实验**: 用 GPT-4.1 按 rubric 打分当 reward, 5 分钟就能训完 (vs 6 小时), 效果 45.6% vs 47.0%。只差 1.4 个点但快 72 倍。这暗示 scaling 到更大 model 时 proxy reward 可能是唯一可行 path。

## 9. Prompt format 的影响

测试了 7 种 prompt, 结论:
- **"Rewrite" prompt 最强 (55.6%)**, 比 "implications" 好
- **"Implications-long" baseline 就 49.3%**, 已经超过 implications 经过 RL 后的 47.0%, 说明 prompt engineering 还有巨大空间
- **No-prompt 失败 (18.9%)**, model 还不能自由 form self-edit, 需要 scaffold
- **CoT 在生成 synthetic data 之前没帮助**, 有点反直觉

## 10. 跟 Generative Adapter 的哲学差异

Generative Adapter 直接从 context 生成 LoRA weights。SEAL 生成 data 再 SFT 出 weights。

Single-passage 上 Generative Adapter 66.8% 吊打 SEAL 47.0% —— 直接 predict weights 更 efficient。

但 CPT (200 篇) 上 Generative Adapter 崩到 28.0%, SEAL 还有 58.2%。

直觉解释: **weights 是 instance-specific, 不可组合; data 是 model-agnostic, 可叠加。** 一个学生可以直接抄答案(single instance), 但长期学习还是要靠做笔记(CPT)。

## 11. 大 picture 和未来

Paper §6 的 vision 我觉得是真正的价值:

1. **数据墙**: 人类 text 2028 年用完, 之后靠 synthetic data。SEAL 提供 meta-learned synthetic data generation 路径。
2. **CoT + SEAL synergy**: CoT 是 in-context reasoning, SEAL 是 weight-persistent learning。两者结合 = 推理 + 记忆。
3. **Agentic loop**: Agent 长期跑任务, 每隔一段时间把 experience 压缩成 weight update, 实现 "memory consolidation"。
4. **Nested RL**: inner loop 用 RL 替代 SFT, 可能既缓解 forgetting 又提升 sample efficiency。

## 12. 我的 take

这个 paper 的 contribution 主要是 conceptual 而非 empirical。实验规模有限 (SQuAD, ARC subset), ARC 只有 11+8 个 tasks, 统计意义不强。但 concept 上它打开了一个我觉得 long-term important 的方向:

**让 LLM 从"被动学习"转向"主动学习"。** 现在所有 finetuning 都是人类 design data format, 人类 tune lr, 人类 select augmentation。SEAL 让 model 自己决定这些。这是 meta-learning 的自然延伸, 但 parameterize 在 token space 里, 复用了 LLM 已有的 generative 能力, 极其优雅。

短期看 compute 太贵 (inner loop SFT 每次 30-45 秒), long-context + CoT 可能更便宜。但 long-term, weight update 是唯一能 persist experience 的机制, SEAL 提供了"如何自动生成 update signal"的 framework。

跟 [TTRL](https://arxiv.org/abs/2504.16084), [RL's razor](https://arxiv.org/abs/2509.04259), [Transformer²](https://arxiv.org/abs/2501.06252), [New News](https://arxiv.org/abs/2505.01812) 这些同期工作放一起看, 2025 年这个方向明显在 converge —— 都是 test-time / self- adaptation / continual learning 的不同切面。SEAL 在这个 family 里是个 nice addition, 不是 game changer, 但 concept 很 clean。

一句话: **SEAL = LLM 学会给自己写错题本, 并通过 RL 知道哪种错题本最有用。**

---

Karpathy 你好, 这篇 SEAL paper 我觉得挺有意思, 它从"模型应该学会自己生成 finetuning data"这个 angle 切入, 把 meta-learning 的 outer loop 用 RL 包起来, 让 LLM 变成一个"会给自己写学习材料的学生"。我来拆解一下技术细节, 顺便 build 一些 intuition。

## 1. 核心 idea 的 mental model

先把整个 framework 在脑子里建立一个 mental model:

```
   ┌──────────────────────────────────────────────────────────┐
   │  Outer RL Loop (ReST^EM, on-policy)                       │
   │   θ_t ──► generate SE ──► SFT inner ──► θ'_t ──► reward  │
   │   θ_{t+1} = RL_update(θ_t, SE, r)                        │
   └──────────────────────────────────────────────────────────┘
              ↓
   ┌──────────────────────────────────────────────────────────┐
   │  Inner SFT Loop (LoRA, cheap)                             │
   │   θ'_t = SFT(θ_t, SE)   via gradient descent              │
   └──────────────────────────────────────────────────────────┘
              ↓
   ┌──────────────────────────────────────────────────────────┐
   │  Evaluation on τ  ──►  r ∈ {0, 1}                         │
   └──────────────────────────────────────────────────────────┘
```

关键 insight 在于: model 不直接 output answer, 它 output 的是一份"训练说明书", 说明书告诉 inner loop "该训练什么 data、用什么 lr、几个 epoch、用哪些 augmentation"。这等于把 adaptation strategy 完全 parameterize 在 token space 里, 复用了 LLM 已有的 generative 能力。

## 2. RL objective 的仔细解读

公式 (1):

$$\mathcal{L}_{\mathrm{RL}}(\theta_t) := -\mathbb{E}_{(C, \tau) \sim \mathcal{D}} \left[ \mathbb{E}_{\mathrm{SE} \sim \mathrm{LM}_{\theta_t}(\cdot \mid C)} \left[ r(\mathrm{SE}, \tau, \theta_t) \right] \right]$$

变量拆解:
- $\theta_t$: outer loop 第 $t$ 步的 model 参数 (LLM weights, 这里是 RL policy 的参数)
- $C$: context, 比如 passage (knowledge incorporation) 或 few-shot demos (ARC)
- $\tau$: downstream task的 evaluation, 比如 QA 集或 test input
- $\mathrm{SE}$: self-edit, 是一段 token 序列, 表示"训练指令"
- $\mathrm{LM}_{\theta_t}(\cdot \mid C)$: 以 $C$ 为 prompt 的 autoregressive 分布
- $r(\mathrm{SE}, \tau, \theta_t)$: reward, binary 0/1, 取决于 $\theta'_t = \mathrm{SFT}(\theta_t, \mathrm{SE})$ 在 $\tau$ 上的表现

注意这里 reward 显式依赖 $\theta_t$。这是个 critical 的设计点 —— 它导致整个 RL 问题的 state 是 $(C, \theta_t)$, 而不只是 $C$。意思是: 同样的 $C$ 在不同训练阶段下, 同样的 $\mathrm{SE}$ 可能给出不同 reward, 因为模型 baseline 在变。这就让 off-policy 的 PPO/GRPO 变得不稳定 —— 旧 trajectory 的 reward signal 在新 $\theta$ 下是 stale 的。

paper 里正是因为这个原因, 放弃 PPO/GRPO 改用 ReST^EM。这给了我一个 useful intuition: **当 reward function 本身依赖于 policy parameter 时, on-policy 是必需的, 而且越简单的 on-policy 越稳, ReST^EM 这种 rejection sampling + SFT 是最 robust 的选择。**

## 3. 为什么 ReST^EM 等价于 REINFORCE + stop-gradient

公式 (3)(4):

$$\nabla_{\theta_t} \mathcal{L}_{\mathrm{RL}} \approx -\frac{1}{NM} \sum_{i=1}^{N} \sum_{j=1}^{M} r_{ij} \nabla_{\theta_t} \log p_{\theta_t}(\mathrm{SE}_{ij} \mid C_i)$$

$$= -\frac{1}{NM} \sum_{i=1}^{N} \sum_{j=1}^{M} r_{ij} \sum_{s=1}^{T} \nabla_{\theta_t} \log p_{\theta_t}(y_s^{(i,j)} \mid y_{<s}^{(i,j)}, C_i)$$

- $N$: minibatch 中 context 数量
- $M$: 每个 context 采样的 self-edit 数量
- $T$: self-edit 序列长度
- $y_s^{(i,j)}$: 第 $i$ 个 context 的第 $j$ 个 sample 的第 $s$ 个 token
- $r_{ij} \in \{0, 1\}$: binary reward

当 $r_{ij} = 0$ 时, 整项消失。所以梯度变成: 只对那些取得 positive reward 的 self-edits 做 maximum likelihood (SFT)。这就是 ReST^EM 的"rejection sampling + SFT"。

关键的 stop-gradient trick: reward $r(\mathrm{SE}, \tau, \theta_t)$ 本来是 $\theta_t$ 的函数 (因为 $\theta'_t = \mathrm{SFT}(\theta_t, \mathrm{SE})$), 求梯度时需要穿过整个 SFT inner loop, 这在计算上 infeasible (二阶梯度)。所以 paper 把 reward 当成与 $\theta_t$ 无关的常数处理, 只对 log-likelihood 求梯度。这是个 important approximation, 类似于 "REINFORCE treat reward as constant w.r.t. policy" 的精神, 但这里是因为 inner loop 的 non-differentiability 强制 stop-gradient。

**Intuition**: ReST^EM 在 binary reward 下 = "把 successful trajectories 当 supervised data 训练" = "implicit 的 best-of-N with SFT"。简单粗暴但 on-policy 稳定性高。

## 4. Knowledge Incorporation 的具体 instantiation

Figure 2 的 pipeline:

1. Input: 一段 passage $C$
2. Prompt: "Let's read the following passage and produce a list of implications derived directly or indirectly from the content."
3. Model 生成 implications (即 self-edit $\mathrm{SE}$)
4. 把 $\mathrm{SE}$ 切分成多条 sentences, 作为 SFT 训练 data
5. 用 LoRA (rank 32~64, alpha 32~64, lr 1e-4 ~ 2e-3, epochs 1~20) 做 SFT
6. 得到 $\theta'_t$
7. 在 no-passage-in-context 的 SQuAD QA 上 evaluate
8. 用 GPT-4.1 做 grader (yes/no 判断 correctness), 得到 $r$

这里有几个微妙点:

**LoRA 的角色**: 因为每个 passage 要单独 SFT 一次, 而且 outer loop iteration 极多 (50 passages × 5 samples × 3 seeds = 750 inner loops/round), 必须用 LoRA 这种 lightweight adapter 保证 compute 可行。在 §B.7 scaling 实验里能看出, 即使是 7B model, LoRA 也工作得不错。

**Implication format 的优势**: 相比直接 rewrite, implication 是 deductive closure, 把 passage 转化为多个 atomic facts。这种"原子化"使得 SFT 的 loss 更 dense, 每个 token 都承载可学习信号, 而不是把 model capacity 浪费在"原 passage 的语法复述"上。这其实是 [Akyürek et al. 2024 (Deductive Closure Training)](https://aclanthology.org/2024.findings-acl.584/) 的思路延续。

**为什么 SEAL > GPT-4.1 synthetic data**: 这是个 surprising result。Qwen2.5-7B 自生成的 synthetic data 在 SQuAD no-context 上达到 47.0%, 比 GPT-4.1 的 46.3% 还高。这暗示: **"知道哪些信息对 SFT 自己最有用"是 model-specific 的, 越了解自己的弱点, 生成的 data 越针对性。** GPT-4.1 生成的是"通用合理的 implications", SEAL 生成的是"我 (Qwen-7B) 学起来最高效的 implications"。

## 5. Few-shot Learning (ARC) 的 instantiation

这个 setup 更有意思, 它把 self-edit 扩展成"tool selection + hyperparameter config":

Self-edit 在 ARC 场景下是一段 JSON:
```json
{
  "data_generation": {
    "use_basic_augmentations": true,
    "use_size_augmentations": true,
    "use_chain_augmentations": false,
    "use_repeat_augmentations": false
  },
  "training": {
    "strategy": "train_using_output_tokens",
    "learning_rate": 1e-4,
    "num_train_epochs": 2
  }
}
```

这里 self-edit 是一个离散 + 连续混合的 action space:
- 4 个 boolean flag (16 种 augmentation 组合)
- 1 个 discrete choice (all_tokens vs output_tokens)
- 2 个 continuous values (lr, epochs)

Model 通过 token generation 自然地表达这个混合 action space, 这是 LLM-based RL 的一个天然优势 —— 不需要手工 design action representation。

augmentations 来自 [Akyürek et al. 2025 (TTT for few-shot)](https://arxiv.org/abs/2411.07279), 包括 rotations, flips, reflections, transpositions, resizing, chained transforms。

实验结果 (Table 1):

| Method | Success Rate (%) |
|---|---|
| ICL | 0 |
| TTT + Self-Edit (w/o RL) | 20 |
| SEAL | 72.5 |
| Oracle TTT | 100 |

从 20% → 72.5% 这个 jump 很大, 但还远低于 Oracle 100%。这告诉我: **当前 SEAL 还没完全学到 optimal TTT configuration, 它学到了一些"有用的 heuristic", 但还有 improvement 空间。**

## 6. Continued Pretraining (CPT) 实验 —— 一个重要的 generalization test

Table 2 显示: 在 n=200 passages 的 CPT 设置下, SEAL 达到 58.2%, 比 single-passage 的 47.0% 还高。这个反直觉结果的原因: 在 CPT 里, 每个 passage 采样 5 个 self-edits, aggregate 后训练。**aggregate 起到了"集思广益"的作用, 5 次生成覆盖了 passage 的不同 aspect, 训练信号更丰富。**

但 n=2067 时, SEAL 反而降到 46.4%, 被 GPT-4.1 synthetic (49.2%) 略胜。这暗示: **当 passages 太多时, "single-passage 上学到的 self-edit policy"的边际效益递减**, 因为不同 passage 的 heterogeneity 增加, 单一 policy 难以全覆盖。这是 SEAL 当前局限的一个 evidence。

## 7. 与 Generative Adapter 的对比 (§B.8) —— 设计哲学的差异

Generative Adapter ([Chen et al. 2025](https://openreview.net/forum?id=bc3sUsS6ck)) 是个 hypernetwork, 直接从 context 生成 LoRA weights。结果:

| Model | Single-passage (n=1) | CPT (n=200) |
|---|---|---|
| SEAL | 47.0 | 58.2 |
| Generative Adapter | 66.8 | 28.0 |

Generative Adapter 在 n=1 时很强 (66.8%), 但 CPT 时崩盘 (28.0%)。这个对比很有启发性:
- Generative Adapter 是"per-instance LoRA", 每个 context 独立生成一个 adapter, 无法 batch/reuse
- SEAL 是"per-instance synthetic data", data 可以 aggregate, 可以复用, 可以迁移到任意 base model

**Intuition**: 在 single-instance adaptation 上, 直接预测 weights 比 "data → SFT → weights" 更 efficient (省了 inner loop); 但在 multi-instance / continual 场景, "data as the parameterization of update" 远更 scalable, 因为 data 是 model-agnostic 且 composable 的。

## 8. Catastrophic Forgetting 实验 (§5)

Figure 6 的实验设计: 顺序喂入 8 个 passage, 每个都触发一次 self-edit + SFT, 然后在每个 step 后回头测试所有已学 passage。结果是个三角矩阵, 早期 passage 的 accuracy 随更新数增加而 decay。这是经典 catastrophic forgetting 现象 ([McCloskey & Cohen 1989](https://www.sciencedirect.com/science/article/pii/S0079742108605368))。

paper 提了几个 mitigation 方向:
- **Reward shaping** ([Hu et al. 2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/b710915795b9e9c02cf10d6d2bdb688c-Paper.pdf)): 把"regression on prior tasks"加入 negative reward
- **Null-space constrained edits** ([AlphaEdit, Fang et al. 2025](https://openreview.net/forum?id=HvSytvg3Jh)): 把 update 投影到 prior knowledge 的 null space
- **Representational superposition** ([Cheung et al. 2019](https://proceedings.neurips.cc/paper_files/paper/2019/file/4c7a167bb329bd92580a99ce422d6fa6-Paper.pdf))
- **RL in inner loop** ([Shenfeld et al. 2025 "RL's razor"](https://arxiv.org/abs/2509.04259)): RL 比 SFT 忘得更少

最后这个我觉得特别有意思 —— 如果 inner loop 也用 RL 而非 SFT, forgetting 可能自然缓解。这给我一个延伸思考: **SEAL 的 outer/inner loop 都可以是 RL, 形成 nested RL 结构, 这种 self-referential RL 可能是未来 continual learning 的关键。**

## 9. Self-referential / Meta-learning 血统

SEAL 在思想上有几条 ancestry:

1. **Meta-learning**: [MAML (Finn et al. 2017)](https://proceedings.mlr.press/v70/finn17a.html) 学的是 init weights 让 fast adaptation 容易; SEAL 学的是"生成什么 data"让 SFT 有效。MAML 在 weight space meta-learn, SEAL 在 data space meta-learn。这是 parameter space 的 shift。

2. **Self-referential networks**: [Schmidhuber 1992](https://people.idsia.ch/~juergen/selfref1992.pdf), [Irie et al. 2022](https://proceedings.mlr.press/v162/irie22b.html) —— 模型修改自己的 weights。SEAL 也是这个思路, 但 modification 通过"生成 data + SFT"间接实现, 而非直接 predict weights。

3. **RL² / Learning to learn**: [Duan et al. 2016](https://arxiv.org/abs/1611.02779), [Wang et al. 2017](https://arxiv.org/abs/1611.05763) —— RNN 在 episode 内学习。SEAL 的 outer loop 在 episodes (contexts) 间学"如何学"。

4. **STaR / Self-training**: [Zelikman et al. 2022 STaR](https://proceedings.neurips.cc/paper_files/paper/2022/file/639a9a172c044fbb64175b5fad42e9a5-Paper.pdf), [ReST^EM Singh et al. 2024](https://openreview.net/forum?id=lNAyUngGFK) —— 都是 rejection sampling + SFT 思路。

5. **Test-Time Training**: [Sun et al. 2020](http://proceedings.mlr.press/v119/sun20b.html), [TTT-LM (Sun et al. 2024)](https://arxiv.org/abs/2310.13807), [Akyürek et al. 2025](https://arxiv.org/abs/2411.07279) —— 在 test 时根据 input 更新 weights。SEAL 内嵌了 TTT inner loop, 加上 RL outer loop 学如何 config TTT。

## 10. Prompting 的 ablation (§B.11) —— 几个有趣的发现

Table 10 测试了 7 种 self-edit prompt format:

| Prompt | Original | Round 1 | Round 2 | GPT-4.1 synthetic |
|---|---|---|---|---|
| No self-edit | 33.5 | - | - | - |
| Implications | 39.7 | 43.7 | 47.0 | 46.3 |
| Implications-long | 49.3 | 52.4 | 54.4 | 54.1 |
| Implications-very-long | 45.0 | 51.5 | 52.1 | 40.9 |
| Rewrite | 49.4 | 55.3 | 55.6 | 54.4 |
| Self-QA | 37.3 | 42.8 | 48.7 | 39.2 |
| No-Prompt | 13.8 | 12.7 | 18.9 | 28.6 |
| Implications-CoT | 38.7 | - | - | - |
| CoT-eval | 37.8 | - | - | - |

几个直觉:
- **"Implications-long"的 baseline (49.3%) 已经超过 "Implications" 经过 2 轮 RL 的结果 (47.0%)**。这说明 prompt engineering 还是非常 powerful, RL 训练的 marginal gain 在不同 base prompt 间不可直接比较。
- **"Rewrite" prompt 表现最好 (55.6%)**, 跟 [Park et al. 2025 (New News)](https://arxiv.org/abs/2505.01812) 发现 QA pair 比 implication 强一致。
- **No-prompt 几乎失败 (18.9%)**, 说明 model 还没学会"自由 form"地表达 self-edit, 需要一定的 prompt scaffolding。
- **CoT 在 synthetic data generation 之前没明显帮助**, 这有点反直觉 —— 我以为推理之后再 generate implications 会更精准, 但 paper 说"substantial difference 没观察到"。

## 11. Compute overhead 分析 (§B.5, §B.10)

每 self-edit evaluation 30-45 秒。一轮 ReST^EM = 50 contexts × 5 samples × 3 seeds = 750 inner loops, 大约 6 小时在 2×H100 上。

§B.10 测试了 **proxy reward** 替代方案: 用 GPT-4.1 按 rubric (length, diversity, quality, correctness) 1-5 评分。结果: SEAL full reward 47.0% (6hr), proxy reward 45.6% (5min)。**proxy 几乎追平, 速度快 72 倍**。

这是个非常重要的 scaling insight: 真正的 inner-loop reward 太贵, 但 proxy reward 用得好可能 95% 替代。未来 SEAL scaling 到更大 model 时, proxy reward 可能是必经之路。

## 12. 几个我能想到的延伸方向

### 12.1 Self-edit 作为 "internal curriculum"
SEAL 学到的 self-edit policy 本质上是一种 self-curriculum —— "对我现在的能力, 什么样的 data 训练最有效"。这跟 [DataRater (Calian et al. 2025)](https://arxiv.org/abs/2505.17895) 思路相通, 都是 meta-learn data curation, 但 SEAL 更进一步, data 是 generated 而非 selected。

### 12.2 SEAL + CoT synergy
paper §6 提到一个 vision: CoT 是 in-context reasoning, SEAL 是 weight-persistent learning。两者 synergy:
- Mid-reasoning: CoT 走到某个 milestone, 触发 self-edit, weight update, 然后继续 reasoning
- Post-reasoning: CoT 结束后, distill 整个 reasoning trajectory 为 self-edit, 持久化

这让我联想到 [TTRL (Zuo et al. 2025)](https://arxiv.org/abs/2504.16084) 和 [Can large reasoning models self-train? (Shafayat et al. 2025)](https://arxiv.org/abs/2505.21444) —— 在 test-time 用 RL 自训练。SEAL 提供了"如何把 reasoning 转化为 weight update"的机制。

### 12.3 Agentic loop
Agent 跑 long-horizon task 时, 每 K 步触发一次 self-edit, 把 "experience" 转化为 weight update。这解决 long-context 的 limitation (context 不能无限长, 但 weights 可以)。本质上是 "memory consolidation" 过程, 类似 sleep 在人类学习中的角色。

### 12.4 Meta-SEAL: train a dedicated SEAL generator
paper §6 提到: 训一个专门的 SEAL synthetic-data generator model。这跟 [Snell et al. 的 speculator model](https://arxiv.org/abs/2404.01267) 或 speculative decoding 的 idea 有点像 —— 把"为某个 student 生成训练 data"作为 separate model 的 job。这可能是 scaling SEAL 的关键。

### 12.5 数据墙与 self-play
[Villalobos et al. 2024](https://arxiv.org/abs/2211.04325) 预测 2028 年人类 text 用尽。SEAL 提供了一种 "model-generated training data" 的 meta-learned 路径。结合 [α-LLM self-play](https://arxiv.org/abs/2505.21444) 思路, 可能形成 "model 训 model" 的 self-sustaining loop。

## 13. 与一些近期工作的关联

- **[Transformer² (Sun et al. 2025)](https://arxiv.org/abs/2501.06252)**: 学 task-specific weight modulation。比 SEAL 更 direct, 但少了一些 flexibility (只能调 LoRA weights, 不能生成任意 data)。
- **[Meta-learning online adaptation (Hu et al. 2023)](https://aclanthology.org/2023.emnlp-main.268/)**: 训 small model 输出 token-specific weights。跟 SEAL 思路类似但 parameterize 方式不同。
- **[New News (Park et al. 2025)](https://arxiv.org/abs/2505.01812)**: System-2 fine-tuning, 强调 QA pair 形式。SEAL §B.11 的 self-qa prompt 印证了这个方向。
- **[Entigraph (Yang et al. 2025)](https://openreview.net/forum?id=07yvxWDSla)**: graph-based synthetic data。SEAL §B.9 跟它对比, 显示 RL-based 和 heuristic-based 各有优势。

## 14. 一些 critical thoughts

1. **Single-passage 结果 vs CPT 结果的反差**: 单 passage 上 SEAL 反超 GPT-4.1, CPT (n=2067) 上反而略输。这暗示 SEAL policy 在大规模场景的 generalization 还需验证。

2. **ARC 实验规模太小**: 11 个 training tasks, 8 个 eval tasks, success rate 直接是"% of 5 self-edits that work"。统计意义有限。Oracle 100% 说明 tasks 本身被筛过 (solvable under optimal TTT), 这是为了 controlled experiment, 但也限制了说服力。

3. **Reward 的 binary nature**: $r \in \{0, 1\}$ 很 coarse。如果用 soft reward (e.g., QA accuracy 而非 correctness), signal 会更 dense, 但 ReST^EM 的"SFT on good"的 simplicity 就丢了。这是 design trade-off。

4. **Inner loop 是 SFT, 不是 RL**: 这导致 catastrophic forgetting。如果 inner loop 改成 RL (用 [RL's razor](https://arxiv.org/abs/2509.04259) 思路), forgetting 可能自然缓解, 但 compute 会爆炸。

5. **Self-edit 是 discrete token sequence**: 这意味着 exploration 是 token-level 的, 在长 self-edit 上 exploration efficiency 低。如果 self-edit 是 continuous latent, 可以用 gradient-based exploration, 但失去了 LLM 复用 generative prior 的优势。

## 15. 总体评价

SEAL 是个很有"美学统一"的工作 —— 把"训练 data 生成"、"超参数 tuning"、"weight update" 三件事统一在 LLM 的 token generation 里, 用 RL outer loop 端到端优化。它真正抓住了"模型应该学会自己学习"这个长期 vision。

技术上的核心贡献:
- 证明了 reward 依赖 $\theta_t$ 时 ReST^EM 比 PPO/GRPO 更稳定
- 证明了 small model 自生成的 synthetic data 能超过 GPT-4.1 (在 self-beneficial 设置下)
- 证明了 self-edit policy 能从 single-passage 迁移到 CPT
- 提供了"用 data parameterize update" vs "用 weights parameterize update"的实证对比

Limitations 也很清晰:
- Compute 贵 (inner loop SFT + eval)
- Catastrophic forgetting 未处理
- 需要 explicit downstream task, 不能处理 unlabeled corpora
- 实验规模有限 (SQuAD, ARC subset)

但作为"self-adapting LLM"方向的 initial step, 我觉得它打开了几个有意思的研究 path: nested RL, agentic learning loop, meta-learned synthetic data generator, mid-reasoning weight update。这些都是我长期看好的方向。

主要参考:
- [SEAL website](https://jyopari.github.io/posts/seal)
- [ReST^EM (Singh et al. 2024)](https://openreview.net/forum?id=lNAyUngGFK)
- [TTT for few-shot (Akyürek et al. 2025)](https://arxiv.org/abs/2411.07279)
- [Deductive Closure Training (Akyürek et al. 2024)](https://aclanthology.org/2024.findings-acl.584/)
- [Generative Adapter (Chen et al. 2025)](https://openreview.net/forum?id=bc3sUsS6ck)
- [MAML (Finn et al. 2017)](https://proceedings.mlr.press/v70/finn17a.html)
- [RL² (Duan et al. 2016)](https://arxiv.org/abs/1611.02779)
- [Self-referential networks (Schmidhuber 1992)](https://people.idsia.ch/~juergen/selfref1992.pdf)
- [AlphaEdit (Fang et al. 2025)](https://openreview.net/forum?id=HvSytvg3Jh)
- [RL's razor (Shenfeld et al. 2025)](https://arxiv.org/abs/2509.04259)
- [Data wall (Villalobos et al. 2024)](https://arxiv.org/abs/2211.04325)
- [Synthetic continued pretraining (Yang et al. 2025)](https://openreview.net/forum?id=07yvxWDSla)
- [New News (Park et al. 2025)](https://arxiv.org/abs/2505.01812)
- [Transformer² (Sun et al. 2025)](https://arxiv.org/abs/2501.06252)
- [TTRL (Zuo et al. 2025)](https://arxiv.org/abs/2504.16084)

如果你想深入聊某个 sub-direction (比如 SEAL inner loop 换成 RL 的设计, 或者 mid-reasoning weight update 怎么具体实现), 我可以继续展开。
