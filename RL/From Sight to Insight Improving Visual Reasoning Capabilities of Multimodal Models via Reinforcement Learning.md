---
source_pdf: From Sight to Insight Improving Visual Reasoning Capabilities of Multimodal
  Models via Reinforcement Learning.pdf
paper_sha256: 9fd993aec610e5e84a2b57abefd2325e79aeecd15eeb8b1168ac4543890fd5ab
processed_at: '2026-08-04T11:02:23-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲这篇 paper

Andrej，我换一种讲法，尽量让你能 "feel" 到这篇 paper 在干什么、为什么这么干、哪里巧妙、哪里可疑。

---

## 一句话总结

**Claude 看不懂时钟图片，但你把时钟写成 "2:43" 这几个字，它突然就会做题了。** 所以 MLLM 的瓶颈不在 "脑子笨"，而在 "眼睛瞎"。作者用纯 RL（不靠任何 SFT 或人工标注的 CoT）逼一个 7B 模型自己学会 "先描述图、再想、再复查、再答" 的长链推理，涨了 5.56 个点。

---

## The "aha" experiment：一张表说服你

最 kill shot 的实验在 Table 3。他们拿 Claude-3.5 和 Claude-3.7（thinking mode）做同一套 visual puzzle 题，跑了两种输入：

- **MM（multimodal）**：给图 + 问题
- **TO（text-only）**：把图用 rule 转成文字 + 同一个问题

转法很 dumb，就是 character grid。比如 Clock 直接写成 `Current Time: 2:43`，N-Queens 写成 `Queen positions: [0,6], [1,3], [3,0], [4,4], [5,8], [7,5], [8,2]`，Maze 写成 0/1 矩阵。

结果：

| | Claude-3.5 MM | Claude-3.5 TO | Claude-3.7 MM | Claude-3.7 TO |
|---|---|---|---|---|
| Clock | 3 | 84 | 7 | 83 |
| N-Queens | 25 | 95 | 25 | 97 |
| **Avg** | **25.9** | **52.6** | **42.4** | **66.0** |

Clock 类目从 7% 跳到 83%——一个 76 pp 的 jump。这意味着什么？**LLM 的 reasoning 模块完全有能力算 "2:43 + 1h10m = 3:53"，但 vision encoder + connector 把时钟图片转成 token 这一步崩了。** Claude 没法可靠地数出 hour hand 指向哪、minute hand 指向哪。

这个诊断非常干净。以前大家说 "MLLM reasoning 差"，分不清是 perception 差还是 reasoning 差。他们用 "把图换成等价文字" 这个 controlled experiment 把两者分开了。结论：**是 perception bottleneck，不是 reasoning bottleneck。**

这与 Shojaee et al. 2025 "The Illusion of Thinking" [arxiv 2506.06941](https://arxiv.org/abs/2506.06941) 的发现是 complementary 的：reasoning model 在长 horizon 组合任务上确实脆，但在 visual puzzle 这种 "perception-heavy + 中等 reasoning" 的任务上，瓶颈在前面那一截。

---

## 那怎么修？三种选项被排掉了

**选项 A：SFT with long CoT**——要人标，每个 puzzle 的 reasoning 是 multi-hop、brittle、domain-specific，标一个 Clock 题的 CoT 至少十几步，9000 个样本就是天文数字的成本。

**选项 B：tool calling / bounding box**——比如 GRIT [arxiv 2505.15879](https://arxiv.org/abs/2505.15879) 让模型在 reasoning 里 reference image region，DeepEyes [arxiv 2505.14362](https://arxiv.org/abs/2505.14362) 让模型 crop+zoom。但这些需要 intermediate supervision，要么是 executable tool outputs，要么是 bbox 标注，对于 visual puzzle 这种 "图本身就是个 puzzle" 的场景，工具帮不上。

**选项 C：纯 reward shaping**——只设计 reward，让模型自己摸出 reasoning structure。这就是这篇 paper 的路。思路跟 R1-zero 一样：reward 是 sparse 的，format 是 minimal 的，但 reasoning behavior 自己 emergent。

---

## RL 怎么 setup：GRPO + 6 种 reward

### GRPO 回顾

GRPO 是 DeepSeek-R1 [arxiv 2501.12948](https://arxiv.org/abs/2501.12948) 用的，比 PPO 省一个 critic。核心 idea：对每个 prompt 采 G 条 completion，reward 在 group 内 normalize 当 advantage，不用训 value function。

公式（来自 DeepSeekMath [arxiv 2402.03300](https://arxiv.org/abs/2402.03300)）：

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q,\, \{o_i\}_{i=1}^G}
\left[
\frac{1}{G}\sum_{i=1}^{G}
\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\min\!\Big(
r_{i,t}\,\hat{A}_i,\;
\mathrm{clip}(r_{i,t},\,1-\epsilon,\,1+\epsilon)\,\hat{A}_i
\Big)
\right]
$$

变量：
- $q$：prompt
- $o_i$：第 $i$ 条 group 内 sample，$G$ 通常 8–16
- $r_{i,t} = \pi_\theta(o_{i,t}|q,o_{i,<t}) / \pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})$：importance ratio
- $\hat{A}_i = (r_i - \bar{r}) / \mathrm{std}(r)$：group-relative advantage，$r_i$ 是第 $i$ 条的 reward
- $\epsilon$：clip range，标准 PPO 0.2

作者做了一个非常关键的决定：**KL coefficient = 0**。也就是完全不放 reference policy 锚。这等于让模型 free explore，跟 R1-zero 的 "let it run wild" 思路一致。代价是可能 collapse 到某种 degenerate 解，但他们用 structured tag reward + repetition penalty 来防止这个。

### 6 种 reward 设计

每个 reward 都对应一个 prompt template，让模型输出特定的 tag。Reward 计算就是 parse 这些 tag，按权重加起来。这是 reward shaping 的核心手艺——**用 tag 结构当 "soft supervision"，告诉模型 reasoning 应该长什么样，但不告诉它 reasoning 应该说什么内容**。

Reward 通用形式：

$$
R = \sum_k \alpha_k \cdot \mathrm{score}_k
$$

每个 non-answer tag 的 score 是：

$$
\mathrm{score}_{\text{tag}} = \tanh(\text{unique sentences in tag}) - \tanh(\text{duplicate sentences in tag})
$$

$\tanh$ 是为了 saturate，防止模型 "灌水刷长度"。unique 减 duplicate 是 anti-repetition。

下面是 6 种 reward，权重见 Table 6：

| Reward | Tags | Weights | 想测什么 |
|---|---|---|---|
| **Vanilla** | think, answer | $\alpha_t=0.1,\, \alpha_a=0.9$ | R1-zero baseline |
| **Only-Accuracy** | answer | $\alpha_a=1.0$ | DAPO 路线，光给 answer reward，模型自己学 format |
| **Mixture** | img_desc, think, rethink, answer | $\alpha_i=\alpha_t=0.06,\, \alpha_r=0.08,\, \alpha_a=0.80$ | 把 reward 摊到多个 stage |
| **Continuous** | img_desc, think, rethink, answer(partial) | 同 Mixture，但 answer 给 partial credit | soft supervision 能不能引导更稳的 reasoning |
| **Visual-Fusion** | think(with `<visual>` inside), answer | $\alpha_t=\alpha_v=0.1,\, \alpha_a=0.8$ | 强制 visual grounding 在 think 里 |
| **No-Accuracy** | img_desc, think, rethink, answer-tag | $\alpha_i=\alpha_t=\alpha_r=0.3,\, \alpha_a=0.1$ | 没有 verifiable reward 也能 emergent reasoning 吗 |

### Continuous reward 的细节

这个最有意思。它对 Clock 类目给 partial credit：

```python
HourScore(pred, true):
    diff = circular_distance_on_12_hour_clock(pred, true)
    if diff > 2: return 0.0
    else: return 1 - diff/5

MinuteScore(pred, true):
    diff = circular_distance_on_60_minute_clock(pred, true)
    if diff > 10: return 0.0
    else: return 1 - diff/20

ClockReward = 0.5*HourScore + 0.5*MinuteScore
```

对 numeric 题：

```python
NumericReward(pred, true):
    diff = abs(pred - true)
    if diff > 5.0: return 0.0
    else: return 1.0/(1.0 + diff)
```

这是 LLM-as-judge 之外的另一种 dense supervision——**用 task structure 当 partial reward**。Clock 的 circular distance、数字题的 absolute deviation，都是把 "接近正确" 编码成 scalar。这种 dense reward 的好处是让 reward landscape 更平滑，坏处是 model 可能学 "差不多就行" 而不学 "精确"。

---

## 训练数据怎么造

AlgoPuzzleVQA [arxiv 2503.02871](https://arxiv.org/abs/2503.02871) 每类只有 100 样本，不够 RL。作者改了 source code 造了 9000 个新样本：

- **Clock-3k**：3000 个 Clock 样本。hypothesis：在 hard category 上训能逼出 long reasoning
- **Diverse-8k**：8000 个样本，Clock/Maze/Move-Box/N-Queens 各 2000。hypothesis：多样化训能 generalize

这两套训练集测的是两个不同的 prior——"hard sample unlock long reasoning" vs "diverse sample unlock generalizable reasoning"。

---

## 结果：哪些 work 哪些不 work

### 主表（Table 2，open-ended setting）

Diverse-8k 训出的 6 个 reward，平均准确率：

| Reward | Avg Acc |
|---|---|
| Baseline | 10.0 |
| Vanilla | 13.22 |
| Only-Accuracy | 15.11 |
| **Mixture** | **15.56** |
| Continuous | 14.67 |
| Visual-Fusion | 13.33 |
| No-Accuracy | 11.22 |

几个 takeaway：

1. **所有 RL 都比 baseline 涨**，包括 No-Accuracy（没有任何 answer supervision 居然也涨了 1.22 pp）。这与 Shao et al. 2025 "Spurious Rewards" [arxiv 2506.10947](https://arxiv.org/abs/2506.10947) 一致——"格式 reward alone 就能 eliciting reasoning"。
2. **Mixture 最好**，说明把 reward 摊到多个 reasoning stage 比 R1-zero 的 "只奖 think+answer" 更好。
3. **Only-Accuracy 接近 Mixture**。这个很重要——它意味着 model 完全从 prompt template 学到了 format，不需要 reward 显式奖 format，只要 answer reward 就够。这是 DAPO [arxiv 2503.14476](https://arxiv.org/abs/2503.14476) 的发现，在 multimodal 场景也成立。
4. **Visual-Fusion 反而比 Mixture 差**。这是反直觉的——明明想强化 visual grounding，却跌了。一个 hypothesis：`<visual>` tag 强行要求 think 里塞 visual description，可能打断 reasoning flow，模型把精力放在 "凑 visual tag" 而不是 "用 visual 信息推"。

### Generalization（Table 4）

| Training set | In-domain | Out-of-domain |
|---|---|---|
| Clock-3k (Mixture) | 13 | 12.12 |
| Diverse-8k (Mixture) | 21.25 | 11.0 |

Clock-3k 几乎不 generalize 到 OOD（4.25 for Vanilla），Diverse-8k 好很多。但都存在 ID-OOD trade-off——训得越专，OOD 越差。这符合 RL overfitting 的常识。

### Per-class best reward（Figure 4）

5 个不同 reward 在 9 个 class 里都拿过 best。没有 "one reward fits all"。Number Slide 这个类目 RL 之后反而比 baseline 跌了——这个 class 的 base 13% 已经不算太差，RL 训坏了，可能是 reward shaping 跟这个 task 的结构不兼容。

---

## 涌现的 reasoning 长什么样（Figure 3）

Base Qwen-2.5-VL-7B 给 Clock 题通常直接猜一个答案，比如 `<answer>3:55</answer>`，没有 reasoning。

Mixture reward 训出来的模型会输出：

```
<image description>
The image shows an analog clock with a circular white face, 
black numerals from 1 to 12, and two hands. The hour hand 
(green) is between 2 and 3, closer to 3. The minute hand 
(brown) is pointing at the 9, indicating 45 minutes.
</image description>



<rethink>
Verify: 2:45 + 1:10 = 3:55. 
Check hour hand position: between 2 and 3 means hour is 2.
Check minute hand at 9: 9 * 5 = 45. ✓
Answer 3:55 seems correct.
</rethink>

<answer>3:55</answer>
```

模型自己摸出了 "describe → reason → verify → answer" 这个 4-stage structure，没人教它，纯靠 reward shaping emergent。这就是 R1-zero 的 "aha moment" 在 visual reasoning 上的复现——**format reward + accuracy reward → structured long reasoning emergent**。

---

## 4 类错误（Section 5.4）

即便最好的模型，平均准确率也只有 15.56%。作者分类了 4 种 failure mode：

1. **Visual perception error**：reasoning 长但 perception 还是错的。比如 hour hand 在 8 和 9 之间，模型硬说 "closer to 8"，导致时间读错。**这是最 fundamental 的**——RL 让 model 学会了 "怎么 talk about visual"，但没让它 "看得更准"。
2. **Algorithmic reasoning error**：算错算术。比如 "subtracting 2 hours from 1 o'clock results in 9 o'clock"（少了一步 mod 12 的考虑）。
3. **Self-contradiction**：reasoning 得出 12:35，answer tag 写 12:37。
4. **Never stops**：一直生成不结束。

第 1 类是核心问题——**RL 教会了 model "structured talk about vision"，但没改 vision encoder 本身的 perception 能力**。要让模型真的 "看得准"，可能需要把 reward 渗到 perception 层（比如 image description tag 的 reward 用一个 vision judge 来评估是否准确，而不是只数 unique sentences）。

---

## 我觉得这篇 paper 的贡献与局限

### Contribution
1. **诊断实验太干净了**。把 image 转 text 这个 controlled experiment 直接证明 perception bottleneck，这种诊断方式以前在 visual reasoning 社区没这么 explicit 地做过。
2. **6 种 reward 系统对比**。给出 "format reward alone 够不够"、"visual grounding reward 是不是反作用"、"partial credit 好不好" 这些问题的实证答案。
3. **Long visual reasoning emergent**。在 7B open-source model 上复现 R1-zero 的 emergence 现象，但扩展到 multimodal。

### Limitations
1. **Response length 2048 token 截断了 long reasoning**。作者承认。真正的 long CoT 在 Clock + 复杂 Maze 上肯定需要 4k+。
2. **Perception 没真正修好**。Mixture 让模型学会 "describe image"，但模型描述的图象内容可能是错的（Figure 3 例子就出现了 misread hour hand）。Reward 没奖励 "描述得对"，只奖励 "描述得多"。
3. **15.56% 绝对值仍很低**。这意味着 RL 的 "unlock" 远没达到上限。我猜测瓶颈在 vision encoder 侧，而不是 RL 侧——Qwen-2.5-VL 的 ViT 在 fine-grained spatial reasoning 上本身就不强，RL 调不动它。
4. **Task 太 algorithmic**。AlgoPuzzleVQA 的 puzzle 都是 well-defined rule-based 任务，跟真实 visual reasoning（看图说话、reasoning about scene）差距很大。Reward shaping 这套方法能不能 transfer 到 "open-ended visual reasoning" 还是 open question。

---

## 跟相关工作的关系

- **DeepSeek-R1 [arxiv 2501.12948](https://arxiv.org/abs/2501.12948)**：这篇是 R1-zero 的 multimodal 版。Vanilla reward 就是 R1-zero 的 setup。
- **DAPO [arxiv 2503.14476](https://arxiv.org/abs/2503.14476)**：Only-Accuracy reward 对应 DAPO 的 "single verifiable reward" 哲学。
- **Spurious Rewards [arxiv 2506.10947](https://arxiv.org/abs/2506.10947)**：No-Accuracy reward 居然涨，跟这篇 "format reward alone 就能提升 reasoning" 一致。
- **GRIT [arxiv 2505.15879](https://arxiv.org/abs/2505.15879) 和 DeepEyes [arxiv 2505.14362](https://arxiv.org/abs/2505.14362)**：visual grounding via tool/bbox，依赖 intermediate supervision，跟本文 "纯 reward" 的路线对比鲜明。
- **VisualPRM [arxiv 2503.10291](https://arxiv.org/abs/2503.10291)**：用 process reward model 给 reasoning step 打分，相当于 dense supervision 的另一条路。作者在 future work 里提了 "用 LLM judge 评 intermediate reasoning"，这其实就是 VisualPRM 路线。

---

## 我的几个 intuition

1. **Reward shaping 的本质是 "告诉模型 reasoning 的 shape，但不告诉 content"**。Tag structure（image_desc / think / rethink / answer）是 shape，tag 内容由模型自己生成。这种 "structural supervision without content supervision" 可能是 RL unlock reasoning 的核心机制——它给了模型一个 "scaffold"，让模型的探索空间从 "all possible token sequences" 缩到 "structured reasoning sequences"，exploration efficiency 大大提高。

2. **Vision encoder 的 perception limit 是 RL 调不动的**。RL 调的是 LLM backbone 的 token distribution，但 vision encoder + connector 把 image 编码成 visual token 这一步是 RL 不可触达的（除非用 visual representation reward 反传到 vision encoder，但 GRPO 的 reward signal 是 token-level）。所以这篇 paper 实际上是 "在 perception bottleneck 下用 LLM reasoning 来 compensate"——让 LLM 多想多查多 verify，弥补 perception 信号不够精确。

3. **"Visual puzzle" 是测试 visual reasoning 的好 testbed**。因为它 decouples "domain knowledge" 和 "visual perception + algorithmic reasoning"。Clock / N-Queens / Maze 这些 puzzle 不需要任何 world knowledge，纯粹是 "看你能不能读出图 + 能不能算"。这种 task 跟 LLM benchmark 上常见的 "VQA with knowledge" 很不同，更适合 isolate visual reasoning ability。

4. **"Tag-counting reward" 是 lazy 但 effective 的 reward design**。它没评估 reasoning 内容质量，只数 unique sentences。这等价于一个 length prior + diversity prior。这种 reward 之所以 work，可能是因为 "long + diverse + structured" 的 token sequence 在训练分布上跟 "reasoning-like" 重叠度高，模型一旦被推向这个区域，就会自然 land 在 reasoning behavior 上。这是 reward hacking 的 "good kind"——hacking 出 long structured output，正好就是 reasoning。

5. **下一步应该是 perception-aware reward**。现在 reward 只奖励 "describe the image" 的长度，不奖励 "describe it correctly"。如果用一个 strong VLM（比如 GPT-4o）当 judge 来评 image description tag 的准确性，把 reward 渗到 perception 层，可能会突破 15.56% 这个 ceiling。这就是 VisualPRM 路线，但用在 image description stage 而不是 reasoning step stage。

---

## 一句话带走

这篇 paper 干净地证明了 "MLLM 在 visual puzzle 上的瓶颈是 perception 而不是 reasoning"，然后用纯 reward shaping（无 SFT、无 tool、无 dense annotation）让 7B 模型自发学会 structured long visual reasoning。15.56% 的绝对值仍低，但 emergence 本身是真的——format reward + accuracy reward 让模型自己摸出 "describe → think → verify → answer" 的四阶段结构。下一步要突破 ceiling，可能得让 reward signal 渗到 perception 层。

参考：
- Paper: [arxiv 2505.10862](https://arxiv.org/abs/2505.10862) (HAVE MLLMS LEARNED TO TELL TIME)
- DeepSeek-R1: [arxiv 2501.12948](https://arxiv.org/abs/2501.12948)
- DeepSeekMath/GRPO: [arxiv 2402.03300](https://arxiv.org/abs/2402.03300)
- DAPO: [arxiv 2503.14476](https://arxiv.org/abs/2503.14476)
- Spurious Rewards: [arxiv 2506.10947](https://arxiv.org/abs/2506.10947)
- AlgoPuzzleVQA: [arxiv 2503.02871](https://arxiv.org/abs/2503.02871)
- GRIT: [arxiv 2505.15879](https://arxiv.org/abs/2505.15879)
- DeepEyes: [arxiv 2505.14362](https://arxiv.org/abs/2505.14362)
- VisualPRM: [arxiv 2503.10291](https://arxiv.org/abs/2503.10291)
- Illusion of Thinking: [arxiv 2506.06941](https://arxiv.org/abs/2506.06941)
- EasyR1 framework: [github hiyouga/EasyR1](https://github.com/hiyouga/EasyR1)

---

# From Sight to Insight: 深度技术讲解

Andrej 你好！这篇来自 Dartmouth 与 UCF 的工作（Sharif, Hossain, Ng）非常有意思，因为它的 motivation 极其干净——一个 controlled experiment 把 "visual perception bottleneck" 从 "algorithmic reasoning bottleneck" 中剥离出来，然后只通过 reward shaping 就让一个 7B 的 open-source MLLM 自发地涌现出 long-form structured visual reasoning，没有任何 SFT，没有 tool call，没有 bounding box。这正好踩在你最近关注的 "RL emerges reasoning behaviors without supervision" 这条线上，所以值得拆细看。

---

## 1. 核心论点与 motivation

作者的核心 claim 由两个独立的实验串起来：

**Claim 1 (诊断)**：MLLMs 在 visual puzzles 上失败，bottleneck 在 visual perception 而不在 algorithmic reasoning。

**Claim 2 (修复)**：纯 reward-driven RL（不靠 SFT、不靠 dense CoT annotation）可以解锁 long visual reasoning capability。

Claim 1 的证据极其简洁——他们做了一件事：把 image 通过 rule-based mapping 转成 character-based text，然后在 Claude-3.5 / Claude-3.7 上重测，得到如下表格（论文 Table 3）：

| Category | Claude-3.5 MM | Claude-3.5 TO | Claude-3.7 MM | Claude-3.7 TO |
|---|---|---|---|---|
| Clock | 3 | **84** | 7 | **83** |
| N-queens | 25 | **95** | 25 | **97** |
| Move box | 33 | 58 | 34 | 70 |
| Number slide | 36 | 60 | 74 | 77 |
| Tower of Hanoi | 44 | 60 | 85 | 85 |
| Water Jugs | 47 | 55 | 72 | 62 |
| **Avg.** | **25.9** | **52.6** | **42.4** | **66.0** |

Claude-3.7 平均 +23.6 pp，Clock 类目甚至从 7% 跳到 83%——直接 76 pp 的 jump。这种 "把 image 编码成 'G-RR' 这种 character grid" 就能让 frontier model 突然会做题的现象，强烈暗示 vision encoder + connector 的 grounding 失败，不是 LLM backbone 推理能力不够。这与 Shojaee et al. (2025) "The Illusion of Thinking" 的诊断一致：reasoning model 在长 horizon 组合任务上脆，但这里 bottleneck 出现在 perception stage，是视觉信号还没被"压缩"进可推理的 token 空间。

---

## 2. GRPO 的技术细节与作者的选择

GRPO 来自 DeepSeekMath / DeepSeek-R1（[arxiv 2402.03300](https://arxiv.org/abs/2402.03300), [arxiv 2501.12948](https://arxiv.org/abs/2501.12948)）。本论文的关键 hyperparameter 选择都偏向 "let the model explore"：

| Setting | Value | 我的解读 |
|---|---|---|
| Training Steps | 500 | 极短，符合 R1-zero 的 "reasoning emerges in early steps" 现象 |
| Batch Size | 128 | |
| Max Prompt / Response Length | 2048 / 2048 | 限制 response 长度可能截断了真正 long CoT，作者承认这是 limitation |
| Learning Rate | 1e-6 | 相对保守 |
| Temperature | 1.0 | 高 T 鼓励 exploration |
| **KL Coefficient** | **0.00** | 关键决定！这等于完全放弃 reference policy 锚定 |

GRPO 的目标函数（参考 [DeepSeekMath](https://arxiv.org/abs/2402.03300)）：

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q),\, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q)}
\left[
\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}
\min\!\left(
r_{i,t}(\theta)\, \hat{A}_{i,t},\,
\mathrm{clip}(r_{i,t}(\theta),\, 1-\epsilon,\, 1+\epsilon)\, \hat{A}_{i,t}
\right)
\right]
$$

其中：

- $q$：prompt（图片 + question + system prompt with tags）
- $o_i$：第 $i$ 条 group 内 sample 的 completion，group size $G$ 通常取 8–16
- $r_{i,t}(\theta) = \dfrac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$：importance sampling ratio，PPO-style clip
- $\hat{A}_{i,t}$：**group-relative advantage**，不需要 critic，直接用 group 内 reward 归一化：

$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(\{r_1, \dots, r_G\})}{\mathrm{std}(\{r_1, \dots, r_G\})}
$$

- $\epsilon$：clip 范围（标准 PPO 中 0.2）
- 本论文 KL 项 $\beta\, D_{\mathrm{KL}}(\pi_\theta \| \pi_{\text{ref}})$ 系数 $\beta = 0$，所以完全退化成 "no-reference PPO with group baseline"

**为什么 KL=0 在这里能 work**：作者想强制 model 自己探索出 `<image description> 、`
