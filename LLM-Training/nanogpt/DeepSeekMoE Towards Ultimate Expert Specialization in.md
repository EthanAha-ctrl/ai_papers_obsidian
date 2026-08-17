---
source_pdf: DeepSeekMoE Towards Ultimate Expert Specialization in.pdf
paper_sha256: a64b509e1410d09cf5b70570788d27790985fb9808ce436b9da8de73f2eed651
processed_at: '2026-08-03T18:58:52-07:00'
target_folder: LLM-Training/nanogpt
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DeepSeekMoE 用人话讲

Andrej，你要人话版，那我就用咱们训练模型时那种"白板上画给同事看"的方式来讲。

参考链接：
- Paper: https://arxiv.org/abs/2401.06066
- Code: https://github.com/deepseek-ai/DeepSeek-MoE
- Model: https://huggingface.co/deepseek-ai/deepseek-moe-16b-base

---

## 一、先说这 paper 到底干了啥

一句话：**把 MoE 里的 expert 切得更碎，再单独留几个"公共服务员" expert，这样每个 expert 学的东西更专一，参数不浪费。**

结果就是：16B 总参、2.8B 激活参的 MoE，打平了 7B 的 dense model，只用 40% 的计算量。

---

## 二、为啥要搞这个（motivation）

传统 MoE（GShard、Switch Transformer）长这样：

```
Token 进来 → router 选 2 个 expert → 这 2 个 expert 处理 → 输出
```

问题在哪儿？假设你有 16 个 expert，每个 expert 要接住一大堆五花八门的 token。比如 expert #3 今天处理"Python 代码"，明天处理"唐诗解析"，后天处理"数学公式"。这个 expert 的参数里就得同时塞进这三种风马牛不相及的知识。

Paper 给这现象起名叫 **Knowledge Hybridity**（知识混合）。你可以理解为：一个打工人被迫同时当程序员、翻译、会计，哪样都干不精。

还有一个毛病叫 **Knowledge Redundancy**（知识冗余）。expert #3 学了一遍基础语法，expert #7 也学了一遍，expert #15 又学了一遍。每个 expert 都要花一部分参数去学那些大家都需要的公共知识，纯属浪费。

---

## 三、DeepSeekMoE 的两个招

### 第一招：Fine-Grained Expert Segmentation（把 expert 切碎）

**做法**：原来 16 个大 expert，每个切 4 份 → 变成 64 个小 expert。原来激活 2 个，现在激活 8 个。总参数和总 FLOPs 不变。

**为啥有用**：组合数爆炸。

用 paper 里的数字：
- 原来选 2 个：$\binom{16}{2} = 120$ 种组合
- 现在选 8 个：$\binom{64}{8} = 4{,}426{,}165{,}368$ 种组合

从 120 种组合跳到 44 亿种组合。Router 能给每个 token 调配的"配方"空间大了 8 个数量级。

Intuition：想象你去餐厅。原来只有 16 道菜，每次只能点 2 道，搭配很受限。现在有 64 道小菜，每次点 8 道，你能组合出各种精致套餐，每道小菜也更容易做得专精（比如专门做凉拌黄瓜，不用兼顾红烧肉）。

公式上，标准 MoE 是：

$$
\mathbf{h}_t^l = \sum_{i=1}^{N} g_{i,t} \, \mathrm{FFN}_i(\mathbf{u}_t^l) + \mathbf{u}_t^l
$$

- $\mathbf{u}_t^l$：第 $l$ 层第 $t$ 个 token 的输入 hidden state，维度 $d$
- $\mathbf{h}_t^l$：MoE 输出
- $N$：expert 总数
- $g_{i,t}$：第 $t$ 个 token 对第 $i$ 个 expert 的 gate value（稀疏，只有 top-K 非零）
- $\mathrm{FFN}_i$：第 $i$ 个 expert 的 FFN

切细之后变成：

$$
\mathbf{h}_t^l = \sum_{i=1}^{mN} g_{i,t} \, \mathrm{FFN}_i(\mathbf{u}_t^l) + \mathbf{u}_t^l
$$

- $m$：切分因子（每个原 expert 切 $m$ 份）
- $mN$：小 expert 总数
- 激活数从 $K$ 变 $mK$，保持计算量不变

每个小 expert 的 intermediate hidden dimension 是原来的 $1/m$，所以单个 expert 更窄，更容易收敛到一个窄语义子空间。

### 第二招：Shared Expert Isolation（留几个公共服务员）

**做法**：从 routed expert 里拎出 $K_s$ 个，作为 shared expert。这 $K_s$ 个对每个 token 都无条件激活，不经过 router。为了保持 FLOPs 不变，routed expert 少激活 $K_s$ 个。

公式：

$$
\mathbf{h}_t^l = \underbrace{\sum_{i=1}^{K_s} \mathrm{FFN}_i(\mathbf{u}_t^l)}_{\text{shared，永远开}} + \underbrace{\sum_{i=K_s+1}^{mN} g_{i,t} \, \mathrm{FFN}_i(\mathbf{u}_t^l)}_{\text{routed，top-}(mK-K_s)} + \mathbf{u}_t^l
$$

- $K_s$：shared expert 数
- 前半截无 gate，无条件求和
- 后半截正常 top-K routing，但 $K$ 减少了 $K_s$

Intuition：公司里的 HR、IT、财务这些公共职能部门，不用每个业务部门都自己养一套，统一放 shared expert 里。业务部门（routed expert）只管自己的核心业务，效率拉满。

---

## 四、Figure 2 画的是啥

三张子图对比，总参数和 FLOPs 完全一样：

```
(a) 传统 MoE:     [E1] [E2] [E3] [E4]    激活 2 个
                   ↑ router 选

(b) Fine-grained: 16 个小 expert          激活 8 个
                   ↑ router 选

(c) DeepSeekMoE:  [S] + 15 个小 expert    S 永远开 + router 选 7 个
                   ↑ shared 永远亮
```

直观就这意思。

---

## 五、Load Balance 两个 loss

Router 自由学习容易出问题：少数 expert 被反复选中，其他 expert 训不出来（routing collapse）。

### Expert-Level Balance Loss（防 collapse）

$$
\mathcal{L}_{\mathrm{ExpBal}} = \alpha_1 \sum_{i=1}^{N'} f_i P_i
$$

- $N' = mN - K_s$：routed expert 数
- $f_i$：expert $i$ 被选中的 token 比例（hard count，归一化到 $\sum f_i = N'$）
- $P_i$：expert $i$ 的平均 softmax affinity score（soft score）
- $\alpha_1$：weight，2B 实验设 0.01，16B 设 0.001，越小越好（太大会伤性能）

$f_i \cdot P_i$ 同时惩罚"被选中次数多"和"平均亲和度高"，双保险。

### Device-Level Balance Loss（防 compute bottleneck）

多卡训练时 expert 分到不同 GPU，要保证每张卡计算量均衡：

$$
\mathcal{L}_{\mathrm{DevBal}} = \alpha_2 \sum_{i=1}^{D} f_i' P_i'
$$

- $D$：device 数
- $f_i'$：device $i$ 上所有 expert 的平均 $f_i$
- $P_i'$：device $i$ 上所有 expert 的 $P_i$ 之和
- $\alpha_2$：145B 设 0.05，比 $\alpha_1=0.003$ 大十几倍

**哲学**：expert-level 平衡只是手段（防 collapse），device-level 平衡才是目的（防计算瓶颈）。所以 $\alpha_2 \gg \alpha_1$。过强的 expert-level balance 会强迫 router 把 token 均匀撒到所有 expert，反而损害 specialization，性能下降。

---

## 六、2B 规模验证（核心实验）

### 配置

- 9 层，hidden 1280，10 heads × 128 dim
- 所有 FFN 替换成 MoE
- expert 总参 = 16 × 标准 FFN，激活 = 2 × 标准 FFN
- 总参 ~2B，激活 ~0.3B
- 100B tokens 训练
- AdamW: $\beta_1=0.9, \beta_2=0.95$, weight_decay=0.1
- LR: warmup 2K → peak $1.08 \times 10^{-3}$，80%/90% 处各 ×0.316
- Batch 2K × seq 2K = 4M tokens/batch，25K steps

### 主结果（Table 1）

| Model | Total | Activated | Pile Loss | HellaSwag | TriviaQA | HumanEval |
|---|---|---|---|---|---|---|
| Dense | 0.2B | 0.2B | 2.060 | 38.8 | 4.9 | 0.0 |
| Hash Layer | 2.0B | 0.2B | 1.932 | 46.2 | 6.5 | 1.2 |
| Switch | 2.0B | 0.2B | 1.881 | 49.1 | 8.9 | 2.4 |
| GShard | 2.0B | 0.3B | 1.867 | 50.5 | 10.2 | 3.7 |
| **DeepSeekMoE** | **2.0B** | **0.3B** | **1.808** | **54.8** | **16.6** | **4.9** |

同样 2B 总参、0.3B 激活参，DeepSeekMoE 全面碾压 GShard。Pile loss 1.867 → 1.808，TriviaQA 10.2 → 16.6（+63% 相对）。

### 接近理论上界（Table 2）—— 最震撼的表

| Model | Expert Params | FLOPs/2K | Pile Loss |
|---|---|---|---|
| GShard ×1.5 | 2.83B | 5.8T | 1.808 |
| Dense ×16 | 1.89B | 24.6T | 1.806 |
| **DeepSeekMoE** | **1.89B** | **4.3T** | **1.808** |

两个解读：

1. **DeepSeekMoE 2B = GShard 2.9B**：GShard 要用 1.5× 参数和 1.5× FLOPs 才追平 DeepSeekMoE。参数效率差 50%。

2. **DeepSeekMoE 2B ≈ Dense×16**：一个 16× FFN 大小的 dense model（所有 expert 都激活）Pile loss 1.806，DeepSeekMoE 1.808，几乎一样。

第二点意味着：MoE 的理论上界就是同总参 dense model。DeepSeekMoE 在 2B/100B token 这个 scale 已经基本触顶了。稀疏 routing 的信息损失被 fine-grained + shared 完全补回来了。

Intuition：你有一个 1.89B 参数的 dense FFN，理论上它能记住 1.89B 参数量的所有知识。DeepSeekMoE 只激活其中 0.24B 参数，就几乎完整利用了这 1.89B 的 capacity。这就是 "ultimate expert specialization" 的实证含义。

### Ablation（Figure 3）

逐步加料看效果：
1. GShard baseline
2. +1 shared expert → 小幅提升
3. +segmentation m=2（32 expert）→ 继续提升
4. +segmentation m=4（64 expert）→ 再提升

两个 trick 都有效，且叠加效果累加。

Shared expert ratio 实验：1/2/4 个 shared，Pile loss 1.808/1.806/1.811，差别不大。最终选 shared : activated routed = 1:3。

### Expert Specialization 分析（Figure 4-6）—— paper 最有营养

**Redundancy 测试**（Figure 4）：禁用每个 token 的 top routed expert（mask 掉概率最高的，从剩下的重选）。

- DeepSeekMoE 的 Pile loss 退化曲线比 GShard×1.5 陡峭得多
- 说明 DeepSeekMoE 的每个 routed expert 更不可替代
- GShard 的 expert 间冗余高，禁掉一个还有别的能顶上

Intuition：一个公司里每个人技能独特，走一个就崩；另一个公司人人都会点皮毛，走谁都无所谓。前者 specialization 高，后者冗余高。

**Shared expert 不可替代性**：禁掉 shared expert，多激活 1 个 routed expert。Pile loss 1.808 → 2.414（+0.606，崩了）。说明 shared expert 学的 foundational knowledge，routed expert 完全没学，两者形成了 clean 分工。

**Half-activated 重训**（Figure 6）：从 scratch 训练 1 shared + 3 activated routed（原来 7 的一半激活），仍 beat GShard。

这说明 DeepSeekMoE 的 activated 参数"含金量"高——同样激活量，DeepSeekMoE 能 beat GShard，因为 DeepSeekMoE 激活的那部分参数全在干实事，没有冗余。

---

## 七、16B 规模（真正能用的 model）

### 配置

- 28 层，hidden 2048，16 heads
- 第一层保留 dense FFN（load balance 收敛慢），其余 27 层全 MoE
- 2 shared + 64 routed，每个 expert 0.25× FFN
- 激活 2 shared + 6 routed = 8 个
- 总参 16.4B，激活 2.8B
- 2T tokens，vocab 100K
- LR peak $4.2 \times 10^{-4}$
- Batch 4.5K × seq 4K = 18M tokens，106K steps
- 单 GPU 40GB 可部署，推理速度 ~2.5× 7B dense

### vs DeepSeek 7B（同 corpus 同 token，Table 3）

| 指标 | DeepSeek 7B | DeepSeekMoE 16B |
|---|---|---|
| FLOPs/4K | 183.5T | 74.4T（40.5%）|
| Pile BPB | 0.75 | 0.74 |
| HellaSwag | 75.4 | 77.1 |
| TriviaQA | 59.7 | 64.8 |
| HumanEval | 26.2 | 26.8 |
| MMLU | 48.2 | 45.0 |

40% 计算量，大部分 task 持平或更好。

**MMLU 落后**的原因 paper 讲得很诚实：DeepSeekMoE 16B 的 attention 参数只有 ~0.5B，DeepSeek 7B 有 2.5B。MoE 把 FFN 稀疏化后，attention 相对比例缩水了。MMLU 这种 multiple-choice task 依赖 attention capacity。这个 observation 直接催生了 DeepSeek-V2 的 MLA（Multi-head Latent Attention）。

### vs LLaMA2 7B（Table 4）

| 指标 | LLaMA2 7B | DeepSeekMoE 16B |
|---|---|---|
| FLOPs/4K | 187.9T | 74.4T（39.6%）|
| HumanEval | 14.6 | 26.8 |
| MBPP | 21.8 | 39.2 |
| GSM8K | 15.5 | 18.8 |
| CHID（中文）| 37.9 | 89.4 |

代码生成接近 2× 优势，中文任务碾压。

### SFT 后（Table 5）

DeepSeekMoE Chat 16B vs LLaMA2 SFT 7B：
- HumanEval: 45.7 vs 35.4
- MBPP: 46.2 vs 27.8
- MATH: 15.2 vs 13.5

历史经验说 MoE 不爱从 SFT 获益（Fedus 2021, Artetxe 2022），但 DeepSeekMoE 16B SFT 后效果很好。说明 fine-grained + shared 的架构既利于 pretraining 也利于 SFT。

---

## 八、145B 初步实验（Table 6）

- 62 层，hidden 4096，32 heads
- 4 shared + 128 routed，每个 expert 0.125× FFN
- 激活 4 shared + 12 routed
- 总参 144.6B，激活 22.2B
- 245B tokens（initial study，未充分训练）

对比 GShard 137B（同 hidden 同 layers，传统 top-2）：

| 指标 | GShard 137B | DeepSeekMoE 145B |
|---|---|---|
| Pile | 1.961 | 1.876 |
| TriviaQA | 52.5 | 61.1 |
| MMLU | 26.3 | 39.4 |

完全碾压。

对比 DeepSeek 67B Dense：

| 指标 | DeepSeek 67B | DeepSeekMoE 145B |
|---|---|---|
| FLOPs/4K | 2057.5T | 585.6T（28.5%）|
| Pile | 1.905 | 1.876 |
| TriviaQA | 57.2 | 61.1 |
| MMLU | 45.1 | 39.4（attention 不够）|

28.5% 计算量打平。

**Half-Activated 变体**：2 shared + 6 activated，FLOPs 374.6T（18.2% of DeepSeek 67B），仍 beat GShard 137B，match DeepSeek 67B。

这条数据很有说服力：激活参数减半，性能没掉多少，说明 activated 参数的有效利用率极高。

---

## 九、Hyper-Parameter 总览（Table 7）

| Params | Layers | Hidden | Shared | Routed (activated) | Expert Size |
|---|---|---|---|---|---|
| 2.0B | 9 | 1280 | 1 | 63 (7) | 0.25× |
| 16.4B | 28 | 2048 | 2 | 64 (6) | 0.25× |
| 144.6B | 62 | 4096 | 4 | 128 (12) | 0.125× |

规律：scale 越大，shared 数和 routed 数都涨，但 expert 相对 size 减小。145B 用 0.125× 而不是更小，因为 expert 太小会损害 GPU kernel 效率。架构创新要和 systems co-design。

---

## 十、Build Intuition 的核心 takeaway

1. **Expert 数量和参数量是两码事**：fixed FLOPs 预算下，切细 expert 让组合学爆炸，表达力增益远超线性。类似 ensemble 里"多弱学习器 > 少强学习器"。

2. **Shared expert 解决 information bottleneck**：传统 MoE 每个 expert 重复学公共知识，相当于每个 expert 交"冗余税"。Shared expert 集中化，routed expert 全力聚焦 distinctive knowledge。类似 LoRA 里 shared base + task adapter 的思路。

3. **MoE 上界 = 同 total params dense model**：DeepSeekMoE 2B 几乎触顶 Dense×16，sparse routing 的信息损失被 fine-grained + shared 完全补回。这是 MoE 设计的圣杯。

4. **Attention : FFN 比例会失衡**：FFN 稀疏化后 attention 相对缩水，MMLU 类任务变弱。这是 DeepSeekMoE 的 known limitation，也是 V2 引入 MLA 的 motivation。

5. **Load balance 是 multi-objective**：expert-level 防 collapse，device-level 防 bottleneck，$\alpha_1$ 小 $\alpha_2$ 大是合理 default。

6. **别过早 fine-grain**：145B 用 0.125× 而非更小，expert 太小伤 GPU kernel efficiency。Architecture 和 systems 要 co-design。

7. **Activated 参数的有效利用率是核心 metric**：DeepSeekMoE 142B Half-Activated 用 18.2% FLOPs 匹配 DeepSeek 67B，说明 activated 参数含金量远高于 dense。这比单纯看总参或激活参更本质。

8. **DeepSeekMoE 的 SFT adaptability 好**：打破了 "MoE 不爱 SFT" 的历史经验，因为 expert 高度专一，SFT 时能精准 update 对应 expert 而不干扰其他。

---

## 十一、后续演进（帮你建立 lineage）

- **DeepSeek-V2** (https://arxiv.org/abs/2405.04434)：加了 MLA 解决 attention capacity 不足，shared/routed 比例进一步调优
- **DeepSeek-V3** (https://arxiv.org/abs/2412.19437)：scale 到 671B（37B activated），引入 auxiliary-loss-free load balancing，完全摆脱了 balance loss 对 routing 的干扰
- **TraceMoE / QMoE** 等后续工作都在 DeepSeekMoE 的 fine-grained + shared 框架里做工程优化

这条技术线从 DeepSeekMoE 开始，基本定义了 2024 年之后 open-source MoE 的 design space。

---

核心就这些。两个 trick，一个切细，一个留公共底座，简单粗暴但 effective。后续 V2/V3 在这个骨架上换 attention、调 balance、scale up，一脉相承。

---

# DeepSeekMoE: Towards Ultimate Expert Specialization in MoE Language Models — 深入解读

Andrej，这篇 paper 是 DeepSeek-AI 在 2024 年 1 月放出的工作，核心目标非常清晰：通过两个互补的 architectural trick 把 MoE 的 expert specialization 推到极致，让稀疏激活的参数效率逼近理论上界（即同总参数量的 dense model）。我会把 motivation、math、architecture、experiments 全部铺开来 build 你的 intuition。

参考链接：
- arXiv: https://arxiv.org/abs/2401.06066
- GitHub: https://github.com/deepseek-ai/DeepSeek-MoE
- HuggingFace: https://huggingface.co/deepseek-ai/deepseek-moe-16b-base
- Open LLM Leaderboard: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

---

## 1. Motivation: 传统 MoE 的两个病灶

GShard / Switch Transformer 这类 top-K routing 的 MoE 有两个结构性缺陷，paper 用了两个非常直观的词来描述：

### 1.1 Knowledge Hybridity（知识混合）
传统做法 expert 数量很少（典型 8 或 16），每个 token 只能进 1–2 个 expert。一个 expert 会被迫承接五花八门的 token，于是在它的 parameters 里要同时拟合"代码 + 历史 + 数学 + 对话"这种风马牛不相及的知识。这些知识在参数空间里互相干扰（interference），无法被 simultaneously utilized。

### 1.2 Knowledge Redundancy（知识冗余）
不同 expert 处理的 token 之间有大量 common knowledge（比如基本的语法、token embedding 层面的 transformation）。每个 routed expert 都会在自己的 parameters 里重复学一遍这部分公共知识，造成参数冗余。

Intuition：想象一个公司只有 8 个部门，每个部门既要处理自己的业务，又要重复维护 HR、IT、财务这些公共职能，效率自然低下。DeepSeekMoE 的两个策略对应两种解法。

---

## 2. Architecture: 两个核心策略

### 2.1 Baseline MoE 公式回顾（公式 3–5）

标准 MoE layer 把 Transformer 里的 FFN 替换成：

$$
\mathbf{h}_t^l = \sum_{i=1}^{N} g_{i,t} \, \mathrm{FFN}_i(\mathbf{u}_t^l) + \mathbf{u}_t^l
$$

变量含义：
- $\mathbf{u}_t^l \in \mathbb{R}^d$：第 $l$ 层 attention + residual 后第 $t$ 个 token 的 hidden state
- $\mathbf{h}_t^l$：经过 MoE 后的输出 hidden state
- $N$：expert 总数
- $\mathrm{FFN}_i(\cdot)$：第 $i$ 个 expert（结构等同标准 FFN）
- $g_{i,t}$：gate value，sparse，仅 top-$K$ 非零

Gate 计算：

$$
g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \mathrm{Topk}(\{s_{j,t}\}_{j=1}^N, K) \\ 0, & \text{otherwise} \end{cases}
$$

$$
s_{i,t} = \mathrm{Softmax}_i\!\left( \mathbf{u}_t^{l\,T} \mathbf{e}_i^l \right)
$$

- $s_{i,t}$：token $t$ 对 expert $i$ 的 affinity score
- $\mathbf{e}_i^l$：第 $l$ 层第 $i$ 个 expert 的 centroid vector（也就是 router 的第 $i$ 行 weight）
- $\mathrm{Topk}(\cdot, K)$：取最大的 $K$ 个 score 的集合

### 2.2 Strategy 1: Fine-Grained Expert Segmentation（公式 6–8）

把每个 expert 沿 intermediate hidden dimension 切成 $m$ 份，每个 mini-expert 的 FFN intermediate size 是原来的 $1/m$。同时把激活数也乘以 $m$，保持总 FLOPs 不变。

$$
\mathbf{h}_t^l = \sum_{i=1}^{mN} g_{i,t} \, \mathrm{FFN}_i(\mathbf{u}_t^l) + \mathbf{u}_t^l
$$

$$
g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \mathrm{Topk}(\{s_{j,t}\}_{j=1}^{mN}, mK) \\ 0, & \text{otherwise} \end{cases}
$$

关键直觉：组合学爆炸。paper 给的例子：
- $N=16, K=2$：$\binom{16}{2} = 120$ 种组合
- $m=4$：$N'=64, K'=8$：$\binom{64}{8} = 4{,}426{,}165{,}368$ 种组合

组合数从 $O(10^2)$ 跳到 $O(10^9)$，相当于 router 可以在指数级更大的"专家配方空间"里给每个 token 配药。每个 mini-expert 也更容易收敛到一个细窄的语义子空间（比如"处理 Python 函数定义"或"处理中文成语"），而 forced 去学一个粗粒度大杂烩。

Intuition 升级：从信息论角度，这相当于增加了 router 的 entropy capacity —— 同样 FLOPs 预算下，模型能在 inference 时区分更多 token 类别。从 representation learning 角度，相当于把一个胖 FFN 的 bottleneck 拆开，让每个 sub-FFN 的 rank-1 子空间更专一。

### 2.3 Strategy 2: Shared Expert Isolation（公式 9–11）

在 fine-grained segmentation 基础上，再额外划出 $K_s$ 个 shared expert，它们对每个 token 都无条件激活，不经过 router。为了保持总 FLOPs 不变，从 routed experts 里少激活 $K_s$ 个：

$$
\mathbf{h}_t^l = \underbrace{\sum_{i=1}^{K_s} \mathrm{FFN}_i(\mathbf{u}_t^l)}_{\text{shared, always on}} + \underbrace{\sum_{i=K_s+1}^{mN} g_{i,t} \, \mathrm{FFN}_i(\mathbf{u}_t^l)}_{\text{routed, top-}(mK - K_s)} + \mathbf{u}_t^l
$$

$$
g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \mathrm{Topk}(\{s_{j,t}\}_{j=K_s+1}^{mN},\, mK - K_s) \\ 0, & \text{otherwise} \end{cases}
$$

变量补充：
- $K_s$：shared expert 数量（2B 模型为 1，16B 为 2，145B 为 4）
- 总 expert 数：$mN$（其中前 $K_s$ 个是 shared，剩下 $mN - K_s$ 个是 routed）
- 非 zero gate 数：$mK - K_s$

直觉：shared expert 像"公共底座"，专门吸收跨 token 的 invariant 知识（语法骨架、common sense、token-level normalization），把这部分从 routed expert 的负担里剥离出来。这样 routed expert 可以 100% 聚焦在 distinctive knowledge 上，参数利用率拉满。

Paper 特别 credit 了 DeepSpeed-MoE (Rajbhandari et al., 2022) 的类似 idea，但强调 DeepSeek 是从 algorithmic 角度推导，DeepSpeed-MoE 是 engineering 角度。
- DeepSpeed-MoE 论文: https://proceedings.mlr.press/v162/rajbhandari22a.html

### 2.4 路由可视化（Figure 2）

三张子图对照：
- (a) 传统 top-2 MoE：4 个大 expert，激活 2 个
- (b) Fine-grained segmentation：16 个小 expert，激活 8 个，参数总量和 FLOPs 不变
- (c) DeepSeekMoE 完整版：在 (b) 基础上把其中 1 个变成 shared（始终亮），剩下 15 个里 top-7

三张图的总参数和 FLOPs 完全一致，公平对比。

---

## 3. Load Balance: 两层 loss

自动学出来的 router 容易 routing collapse（少数 expert 被反复选中，其他 expert 训不出来）和 device 间计算不均。

### 3.1 Expert-Level Balance Loss（公式 12–14）

$$
\mathcal{L}_{\mathrm{ExpBal}} = \alpha_1 \sum_{i=1}^{N'} f_i P_i
$$

$$
f_i = \frac{N'}{K' T} \sum_{t=1}^{T} \mathbb{1}(\text{Token } t \text{ selects Expert } i)
$$

$$
P_i = \frac{1}{T} \sum_{t=1}^{T} s_{i,t}
$$

- $N' = mN - K_s$：routed expert 数
- $K' = mK - K_s$：激活 routed expert 数
- $f_i$：每个 expert 被选中的 token 比例（normalize 到 $\sum f_i = N'$）
- $P_i$：每个 expert 的平均 affinity score（softmax 后的均值）
- $\alpha_1$：超参，2B 实验设 0.01，16B 设 0.001，145B 设 0.003

为什么用 $f_i \cdot P_i$ 而非 $f_i$ 单项？因为 $f_i$ 是 hard count（top-K 决策），$P_i$ 是 soft score，两者乘积能同时惩罚"被选中次数多"和"平均亲和度高"两种倾向，引导 router 把 token 均匀分散。这其实是 GShard 原始设计的延续。

### 3.2 Device-Level Balance Loss（公式 15–17）

把 routed expert 分成 $D$ 组，每组部署在一个 device 上：

$$
\mathcal{L}_{\mathrm{DevBal}} = \alpha_2 \sum_{i=1}^{D} f_i' P_i'
$$

$$
f_i' = \frac{1}{|\mathcal{E}_i|} \sum_{j \in \mathcal{E}_i} f_j
$$

$$
P_i' = \sum_{j \in \mathcal{E}_i} P_j
$$

- $\mathcal{E}_i$：第 $i$ 个 device 上的 expert 集合
- $f_i'$：device $i$ 上的平均选中比例
- $P_i'$：device $i$ 上的总 affinity

设计哲学：expert-level 平衡只是手段（防 collapse），device-level 平衡才是目的（防 compute bottleneck）。所以 $\alpha_2 > \alpha_1$（145B 设 $\alpha_2 = 0.05$，$\alpha_1 = 0.003$）。过强的 expert-level balance 反而损害性能。

---

## 4. Validation Experiments (2B 规模)

### 4.1 Setup

- 100B tokens 训练
- 9 layers, hidden 1280, 10 heads × 128 dim
- 所有 FFN 替换为 MoE
- expert 总参数 = 16 × 标准 FFN
- activated expert 参数 = 2 × 标准 FFN
- 总参 ~2B，激活 ~0.3B
- Vocab 8K
- AdamW: $\beta_1=0.9, \beta_2=0.95$, weight_decay=0.1
- LR: warmup 2K steps → peak $1.08 \times 10^{-3}$，80% 和 90% 处各 ×0.316 decay
- Batch 2K × seq 2K = 4M tokens/batch
- 25K steps = 100B tokens

### 4.2 主结果（Table 1）

| Model | Total | Activated | Pile Loss | HellaSwag | TriviaQA | HumanEval |
|---|---|---|---|---|---|---|
| Dense | 0.2B | 0.2B | 2.060 | 38.8 | 4.9 | 0.0 |
| Hash Layer | 2.0B | 0.2B | 1.932 | 46.2 | 6.5 | 1.2 |
| Switch | 2.0B | 0.2B | 1.881 | 49.1 | 8.9 | 2.4 |
| GShard | 2.0B | 0.3B | 1.867 | 50.5 | 10.2 | 3.7 |
| **DeepSeekMoE** | **2.0B** | **0.3B** | **1.808** | **54.8** | **16.6** | **4.9** |

DeepSeekMoE 在所有 12 个 benchmark 上压制 GShard，Pile loss 从 1.867 拉到 1.808（每 token 节省 ~6% 的 nats），TriviaQA EM 从 10.2 → 16.6（+63% 相对提升）。

### 4.3 接近理论上界（Table 2）

这个表是最有说服力的：

| Model | Expert Params | FLOPs/2K | Pile Loss |
|---|---|---|---|
| GShard ×1.5 | 2.83B | 5.8T | 1.808 |
| Dense ×16 | 1.89B | 24.6T | 1.806 |
| **DeepSeekMoE** | **1.89B** | **4.3T** | **1.808** |

- DeepSeekMoE 2B ≈ GShard 2.9B（GShard 用 1.5× 参数和 FLOPs 才追平）
- DeepSeekMoE 2B ≈ Dense×16（一个 16× FFN 大小的 dense model），这是 MoE 的 strict upper bound

第二点意味着：在 2B/100B token 这个 scale，DeepSeekMoE 已经把 expert specialization 做到接近极限 —— 激活的稀疏 expert 几乎完整利用了 1.89B 的 dense FFN capacity。

### 4.4 Ablation（Figure 3）

逐步加料：
1. GShard（baseline）
2. GShard + 1 shared expert → 小幅提升
3. + segmentation m=2（32 experts）→ 进一步提升
4. + segmentation m=4（64 experts）→ 再提升

Shared expert ratio 实验：1, 2, 4 个 shared expert 对应 Pile loss 1.808 / 1.806 / 1.811，差异不大，最终选 1:3（shared : activated routed）。

### 4.5 Specialization 分析（Figure 4–6）—— 这是 paper 最有营养的部分

**Redundancy 测试**（Figure 4）：禁用每个 token 的 top routed experts（mask 掉 routing probability 最高的若干 expert，从剩下的里重选）。DeepSeekMoE 的 Pile loss 退化曲线比 GShard×1.5 陡峭得多 → 每个 routed expert 更不可替代 → redundancy 更低。

**Shared expert 不可替代性**：禁用 shared expert 并多激活 1 个 routed expert，Pile loss 从 1.808 → 2.414（+0.606，灾难性退化）。说明 shared expert 学到的是 routed expert 完全没学的 foundational knowledge，shared 和 routed 之间形成了 clean 分工。

**精准知识获取**（Figure 5）：把 activated routed expert 数从 7 减到 4，DeepSeekMoE 的 Pile loss 仍能匹配 GShard（用满 2 个大 expert）。

**Half-activated 重训**（Figure 6）：从 scratch 训练 1 shared + 3 activated routed（原来 7 的一半），仍 beat GShard。这证明 DeepSeekMoE 的 activated 参数 effective ratio 远高于 GShard。

Intuition 总结：DeepSeekMoE 让每个 activated 参数都在"干实事"，没有参数在重复劳动，也没有参数在跨域混学。这就是 "ultimate expert specialization" 的实证含义。

---

## 5. Scaling to 16B

### 5.1 Setup

- 28 layers, hidden 2048, 16 heads × 128
- 第一层保留 dense FFN（load balance 收敛慢），其余 27 层全 MoE
- 2 shared + 64 routed, 每个 expert 0.25× FFN
- 激活 2 shared + 6 routed = 8 个
- 总参 16.4B，激活 2.8B
- 2T tokens, vocab 100K
- LR peak $4.2 \times 10^{-4}$，batch 4.5K × seq 4K = 18M tokens
- 106,449 steps
- 单 GPU 40GB 可部署（无量化），推理速度 ~2.5× 7B dense

### 5.2 vs DeepSeek 7B（Table 3，同 corpus 同 token）

- FLOPs: 74.4T vs 183.5T（40.5%）
- Pile BPB: 0.74 vs 0.75（更好）
- HellaSwag: 77.1 vs 75.4
- TriviaQA: 64.8 vs 59.7
- HumanEval: 26.8 vs 26.2
- MMLU: 45.0 vs 48.2（弱）

MMLU/CEval/CMMLU 这种 multiple-choice task 落后，paper 给的诊断很 honest：DeepSeekMoE 16B attention 参数只有 ~0.5B，DeepSeek 7B 有 2.5B。multiple-choice 依赖 attention capacity（参见 DeepSeek 7B MQA 变体同样在 MMLU 上挣扎）。这是 MoE 把 FFN 替换为 sparse 后，相对 dense 的 attention 比例缩水的副作用。

### 5.3 vs LLaMA2 7B（Table 4）

- FLOPs: 74.4T vs 187.9T（39.6%）
- HumanEval: 26.8 vs 14.6（接近 2×）
- MBPP: 39.2 vs 21.8
- GSM8K: 18.8 vs 15.5
- 中文 benchmark 全面碾压（CHID 89.4 vs 37.9）

### 5.4 Open LLM Leaderboard（Figure 1）

DeepSeekMoE 16B 在 activated params ~2.8B 的 cohort 里远超同类，和 ~7B activated 的 LLaMA2 7B 持平。

---

## 6. SFT Alignment（Table 5）

历史经验：MoE 通常从 SFT 中获益有限（Fedus 2021, Artetxe 2022）。但 Shen 2023 的 Flan-MoE 显示 instruction tuning 可以 work。DeepSeek 在 1.4M SFT examples 上 fine-tune 8 epochs，constant LR $10^{-5}$。

DeepSeekMoE Chat 16B vs LLaMA2 SFT 7B / DeepSeek Chat 7B：
- BBH: 42.2 vs 39.3 vs 43.1
- MATH: 15.2 vs 13.5 vs 14.7
- HumanEval: 45.7 vs 35.4 vs 45.1
- MBPP: 46.2 vs 27.8 vs 39.0

代码生成上 SFT 后大幅领先，数学推理也 OK，证明 DeepSeekMoE 16B 的 SFT adaptability 良好。MMLU 类任务仍略弱，符合 base model 的 attention capacity 限制。

Flan-MoE 参考: https://arxiv.org/abs/2305.14705

---

## 7. 145B 初步探索（Table 6）

- 62 layers, hidden 4096, 32 heads
- 4 shared + 128 routed, 每个 expert 0.125× FFN
- 激活 4 shared + 12 routed
- 总参 144.6B，激活 22.2B
- 245B tokens（initial study，未充分训练）
- Expert parallelism: routed experts 均匀分到 4 device
- $\alpha_1 = 0.003, \alpha_2 = 0.05$

对比 GShard 137B（同 hidden 同 layers，但传统 top-2）：
- Pile: 1.876 vs 1.961（GShard 严重落后）
- TriviaQA: 61.1 vs 52.5
- 几乎所有 benchmark 都领先

对比 DeepSeek 67B Dense（同 corpus）：
- FLOPs: 585.6T vs 2057.5T（28.5%）
- Pile: 1.876 vs 1.905（更好）
- HellaSwag: 75.8 vs 74.8
- 但 MMLU: 39.4 vs 45.1（attention capacity 限制依然存在）

**Half Activated 变体**：2 shared + 6 activated routed，FLOPs 仅 374.6T（18.2% of DeepSeek 67B），仍 beat GShard 137B 并 match DeepSeek 67B。这进一步验证 2B 实验里 "DeepSeekMoE activated 参数 effective ratio 高" 的结论在 145B 仍成立。

---

## 8. Hyper-Parameter 总览（Table 7）

| Params | Layers | Hidden | Heads | Shared | Routed (activated) | Expert Size | Seq | Batch | LR |
|---|---|---|---|---|---|---|---|---|---|
| 2.0B | 9 | 1280 | 10 | 1 | 63 (7) | 0.25× | 2048 | 2048 | 1.08e-3 |
| 16.4B | 28 | 2048 | 16 | 2 | 64 (6) | 0.25× | 4096 | 4608 | 4.2e-4 |
| 144.6B | 62 | 4096 | 32 | 4 | 128 (12) | 0.125× | 4096 | 4608 | 3.0e-4 |

Pattern：scale 越大，shared expert 数和 routed expert 数都增加，expert 相对 size 减小（145B 用 0.125×，因为 expert 太小会损害 GPU kernel 效率，paper 明确说"finer granularity 会让 computational efficiency 下降"）。

---

## 9. 与同期 / 后续工作的关联

1. **GShard (Lepikhin 2021)**: https://arxiv.org/abs/2006.16668 — top-2 routing baseline
2. **Switch Transformer (Fedus 2021)**: https://arxiv.org/abs/2101.03961 — top-1 routing
3. **Hash Layer (Roller 2021)**: https://arxiv.org/abs/2106.04426 — fixed hash routing
4. **Expert Choice Routing (Zhou 2022)**: https://papers.nips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html — 反向 routing
5. **DeepSpeed-MoE (Rajbhandari 2022)**: https://proceedings.mlr.press/v162/rajbhandari22a.html — shared expert 的 engineering 原型
6. **GLaM (Du 2022)**: https://proceedings.mlr.press/v162/du22c.html
7. **Knowledge Neurons (Dai 2022a)**: https://aclanthology.org/2022.acl-long.581/ — 解释 FFN 是知识存储，支撑 DeepSeekMoE 在 knowledge-intensive task 强的 observation
8. **StableMoE (Dai 2022b)**: https://aclanthology.org/2022.acl-long.489/ — 同一作者群的固定 routing 工作
9. **DeepSeek LLM**: https://arxiv.org/abs/2401.02954 — dense baseline
10. **Chinchilla (Hoffmann 2022)**: https://arxiv.org/abs/2203.15556 — compute-optimal scaling law

后续 DeepSeek-V2 (https://arxiv.org/abs/2405.04434) 在此基础上加了 MLA (Multi-head Latent Attention) 解决你注意到的 attention capacity 不足问题，并把 shared/routed expert 比例进一步调优；DeepSeek-V3 (https://arxiv.org/abs/2412.19437) 把这套架构 scale 到 671B（37B activated），并引入 auxiliary-loss-free load balancing。可以视作 DeepSeekMoE 这套 idea 的成熟延伸。

---

## 10. Build Intuition 的几个关键 takeaway

1. **Expert 数量 ≠ 参数量**：在 fixed FLOPs 预算下，把 expert 切细，组合学爆炸带来的表达力增益远超线性。这点和 ensemble learning 里"多弱学习器 > 少强学习器"的直觉是相通的。

2. **Shared expert 是 information bottleneck 的解药**：传统 MoE 每个 expert 都要重学 common knowledge，相当于每个 expert 都有一个"重复 tax"。Shared expert 把这部分集中化，routed expert 的参数全部用于 distinctive knowledge。这和 LoRA 里 shared base + task-specific adapter 的哲学类似。

3. **MoE 的上界是同 total params 的 dense model**：DeepSeekMoE 2B 几乎触顶 Dense×16，意味着 sparse routing 的"信息丢失"被 fine-grained + shared 完全补回来了。这是 MoE 设计的圣杯。

4. **Attention : FFN 比例在 MoE 里会失衡**：因为 FFN 被 sparse 化（activated 部分 ~2× FFN），但 attention 还是 dense，相对比例就被稀释。这是 DeepSeekMoE 在 MMLU 类任务弱的根因，也是 V2 引入 MLA 的 motivation。

5. **Load balance 是 multi-objective 问题**：expert-level balance 防 collapse，device-level balance 防 bottleneck，两者要分别调 weight。$\alpha_1$ 小 $\alpha_2$ 大是合理 default。

6. **不要过早 fine-grain**：145B 用 0.125× 而非 0.25×，因为 expert 太小会损害 GPU kernel efficiency。架构创新要和 systems co-design。

7. **Activated 参数的有效利用率是 MoE 的核心 metric**：DeepSeekMoE 142B Half-Activated 用 18.2% FLOPs 匹配 DeepSeek 67B，说明 activated 参数的"含金量"远高于 dense。这是评估 MoE 架构优劣的本质指标，比单纯看总参或激活参更本质。

希望这些拆解能帮你 build 起对 MoE design space 的清晰直觉。如果还想深入某一块（比如 router 的 training dynamics、expert collapse 的 theoretical analysis、或 V2/V3 的演进），可以继续追问。
