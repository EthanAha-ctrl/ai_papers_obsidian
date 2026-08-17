---
source_pdf: MoRE.pdf
paper_sha256: 971615891701a0fc87c71ba2730dcbae012dede85adcc4d54e8e35cd760658cf
processed_at: '2026-08-05T20:28:38-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MoRE 用人话讲

## 一句话总结

教四足机器人干各种活儿(走、爬、钻、卸货),以前的方法只会"抄作业"——照着人类示范一遍遍学,学不会就拉倒。MoRE 说:**我让大模型当"裁判"打分,不光抄好作业,连差作业也能从里头学到"哪种做法能成",所以更聪明、更能扛活。**

---

## 这玩意儿到底在解决啥问题

假设你训一只机器狗,要它会六种技能:识别物体、走过去、绕障碍、穿过窄缝、蹲下爬行、卸货。

老办法是:人遥控机器狗做一遍,录下来好几千条"成功轨迹",然后让模型模仿。问题来了:

1. **人遥控累且贵**,采集成功的 demo 成本高,失败的只能扔掉——太浪费
2. **多任务一起训会打架**,爬行要低头矮身,导航要快跑,两个梯度互相拉扯,模型学得拧巴
3. **一旦遇到没见过的情况就懵了**,因为它只会"专家怎么做我就怎么做",没见过这种场面就不知道咋办

MoRE 的回答:换个训法。别光让它抄,让它学会**判断哪个动作能成事**。

---

## 三个关键 trick 拆开讲

### Trick 1: 让大模型当裁判,而不是当复读机

用 Fuyu 8B 这个多模态大模型当大脑。输入一张相机图 + 一句话("去把黄球卸下来"),它一个个吐字儿一样吐出 12 个数字:前进速度、转向、步态、身体高度、俯仰角……这些是喂给底层 walk-these-ways 控制器的高级指令,控制器再算出 12 个关节怎么动。

关键在**怎么训**。老办法是 next-token prediction 抄专家的 token。MoRE 改成 Q-learning:让模型对每个 token 输出"这个动作能成的概率",用 sigmoid 压到 0~1,当 Q 值。

直觉:Q 值就是"我从这个状态选这个动作,后续能拿到多少累积奖励"。奖励只在任务成功时给 +1,其他时候都 0。模型通过反复 Bellman backup,慢慢学到"爬到障碍前该蹲下了,蹲下能成事,这地方 Q 值高"。

为啥这比抄作业强?抄作业只知道"专家在这蹲下了",不知道"蹲下是为啥,不蹲会咋样"。Q-learning 让模型理解**动作的后果**,所以遇到没见过的场景也能推理。

### Trick 2: 多个"小专家"代替一个大模型硬抗

8B 大模型直接 fine-tune 六个任务,梯度打架。MoRE 在每个 transformer 层的 FFN 上挂了几个 LoRA adapter,每个 LoRA 当一个 expert。原始 FFN 参数共享且冻住,只有 LoRA 在动。

每个 token 进来时,一个 router 小网络打分,选 Top-K 个 expert 激活,其他睡觉。这样:
- 走路的 token 找走路 expert
- 爬行的 token 找爬行 expert
- 互不打架,各学各的

参数效率高:总参数 9.82B(比 8B 只多 1.82B),但每个 token 实际激活的参数和 dense model 差不多,算力没爆。

类比:不是一个人同时学钢琴+画画+编程会互相干扰,而是招三个老师,你每节课只请一个,各学各的,但你(共享 backbone)的脑子是共用的。

### Trick 3: 失败数据也喂,学会"凑合也行"

这是最反直觉的点。MoRE 故意混入一堆 sub-optimal 数据(自动采集的、经常失败的轨迹)一起训。结果显示成功率反而涨了 4 个点。

为啥?因为作者看透了 quadruped 任务的结构:

1. **奖励是稀疏的、跟轨迹长度无关**:只有成功那一刻给 +1,跑 5 秒成和跑 10 秒成都拿一样分,所以 Q 值信号不会因轨迹长而衰减
2. **关键点很少**:一条轨迹里真正"生死攸关"的状态就几个(俯身那一刻、倾倒那一刻),大部分中间状态你偏离最优轨迹也能找回来——因为"够好"的动作集合很大
3. **轨迹长**:长轨迹 + 复合误差让 IL 越走越歪,但 RL 的 bootstrapping 反而稳

所以失败的轨迹里,RL 也能学到"哦,走到这儿不该左转,左转 Q 值低,右转 Q 值高"。它把每条轨迹都当成 Q 值标注的样本,而不是只看成功失败标签。

IL 看失败轨迹:这玩意儿失败了,扔掉。
RL 看失败轨迹:中间那几步其实挺好的,我用 Bellman 备份把好步骤的信号传回去。

---

## 训练目标长啥样(不展开公式)

两个 loss 加起来:

**主 loss**:让 Q 值逼近 Bellman target(就是"当前 Q 应该等于 reward + 下一状态最大 Q")。这是 Q-learning 的核心,让模型学会估值。

**保守 loss**:对那些在数据集里很少出现的 action,把它们的 Q 值拉向 0。为啥要这样?防止模型瞎猜"我没见过的动作 Q 值肯定特别高",然后选了离谱动作。这叫 conservative Q-learning,offline RL 的标配。

**MoE 平衡 loss**:让 router 别老把 token 都送到一个 expert,逼着负载均衡,不然其他 expert 训不更新就废了。权重很小(0.002),起辅助作用。

---

## 实验结果有啥看点

主表对比 CLIP(86M)、VC-1(307M)、QUART(8B)、MoRE(9.82B):

- 小模型(CLiP/VC-1)在爬行和卸货上直接 0%——这些任务太复杂,小模型脑子不够
- QUART 是同组前一版工作(纯 IL,无 MoE),平均 44%
- MoRE 60%,硬任务提升尤其大:爬行 32%→49%,卸货 12%→33%

消融实验最有意思:
- 把 RL loss 换回 IL loss:平均从 60 掉到 51,**即使只用专家数据,RL 也比 IL 强**。这印证了 paper 核心论点
- 去掉 MoE:从 60 掉到 48,多任务冲突确实存在
- 去掉 sub-optimal data:从 60 掉到 56,失败数据确实有用

---

## 这篇 paper 真正的 contribution

我觉得有三个层面:

**工程层面**:第一次把"MLLM + MoE + RL fine-tune"三件套焊在一起跑通,而且参数只多 1.82B、A100 训 100 多小时,成本可控。

**方法层面**:给 quadruped VLA 找到了一个对的任务结构假设——"horizon-independent returns + limited critical points"。这个假设让 RL 在 sub-optimal data 上能 work,是整个方法成立的根基。如果任务是稠密奖励、每一步都 critical,这套就玩不转。

**思想层面**:挑战了"VLA = IL on expert demos"的默认范式。Google RT-2、OpenVLA 都是 IL 路线,MoRE 说"大模型时代 RL fine-tune 不仅可行,而且即使纯专家数据下 RL > IL"。这对未来 robot foundation model 怎么训有指向意义。

---

## 我觉得有问题的点

1. Top-K 中的 K、expert 数 N 都没写清楚,关键超参缺失
2. 爬行 49%、卸货 33% 绝对值仍低,hard task 远未解决
3. Sub-optimal data 只占 24%,比例偏小,纯 sub-optimal vs 纯 expert 的对比缺
4. Sim2real 量化分析缺,只放了几个轨迹图
5. Q-Transformer 原文有 invalid action masking 的关键 trick,这里改用 conservative loss 间接处理,效果是否等价没讨论

---

## 给你的 intuition 一句话

**MoRE 让大模型从"抄作业的复读机"变成"会判断后果的裁判",并且用稀疏专家分工避免多任务打架,所以连差生的作业本也能从中提炼出考点。**

想再深挖哪块(Q-Transformer 的 Bellman backup 推导、MixLoRA 的梯度流、walk-these-ways 怎么把 12 维命令翻译成关节动作、或 conservative Q-learning 为啥能防 OOD 高估),你点哪我讲哪。

---

# MoRE: Mixture of Robotic Experts 深度讲解

## 一、高层 Intuition

MoRE 这篇论文把三件事拧到了一起,形成了 quadruped robot 的 end-to-end VLA 系统:

1. **MLLM backbone**: 用 Fuyu 8B 这种 decoder-only transformer 当 backbone,直接吃 RGB image + language instruction,自回归吐出 12 维 discretized action token,完全 end-to-end。
2. **Sparse-activated MoE via LoRA**: 在 FFN 上插入多个 LoRA adapter 形成 Mixture of LoRA Experts,让一个 dense model 在 fine-tune 时变成 sparse MoE,适配 multi-task 而不爆显存。
3. **Offline RL as Q-function**: 把 transformer 的 next-token logits 通过 sigmoid 当 Q-value,用 autoregressive discrete Q-learning (借鉴 Q-Transformer) 训练,从而能消化 sub-optimal data,这是 IL 做不到的。

整篇文章的 core insight 是:quadruped VLA task 的 MDP 具有"horizon-independent returns + limited critical points"的结构,这种结构使得 RL 在 non-critical states 上能"自由探索"找到 good-enough action,而 IL 只能死磕 demonstrator。这正好是 offline RL vs BC 的经典比较 (Kumar et al. 2022) 的应用延伸。

参考:
- Fuyu 8B: https://www.adept.ai/blog/fuyu-8b
- Q-Transformer: https://arxiv.org/abs/2309.10150
- When should we prefer offline RL over BC: https://arxiv.org/abs/2204.05618

---

## 二、Architecture 细节

### 2.1 Backbone: Fuyu 8B 的选择

Fuyu 8B 是 Adept 提出的 multimodal model,它的特点是:
- **Decoder-only**,无 vision encoder,直接把 image patches 当 token 输入 transformer,绕开了 ViT + projector 的两阶段结构
- 支持 arbitrary resolution 和 multiple images
- 32 层 transformer block

对 quadruped robot 来说,这种设计很自然:camera 的 480×640 RGB 图像直接 patchify 后和 text token 一起进 transformer,避免 vision encoder 与 LLM 之间的 alignment gap。

### 2.2 Action Tokenization

paper 中 action a_t 被离散化为 12 维 command,这是输出空间:

$$a_t = [v_x, v_y, \omega_z, \theta_1, \theta_2, \theta_3, f, h_z, \phi, s_y, h_z^f, T]$$

变量解读:
- $v_x, v_y$: x, y 方向线速度
- $\omega_z$: z 轴角速度
- $\theta_1, \theta_2, \theta_3$: gait pattern 参数 (来自 walk-these-ways)
- $f$: 步频 frequency
- $h_z$: 机器人身体高度
- $\phi$: pitch 角
- $s_y$: 足宽
- $h_z^f$: 足抬起高度
- $T$: 终止信号

这 12 维命令会喂给 walk-these-ways 训练好的 low-level RL policy,生成 12 个关节指令。所以 MoRE 是 high-level controller,不是直接的 joint controller。

### 2.3 自回归 Action 生成

公式 (4):

$$P_{LM}(a_t | s_t) = \prod_{i=0}^{d_A} P_{LM}(a_t^i | I_{RGB}; T_{Inst}; a_t^{1:i-1})$$

变量:
- $s_t$: 当前 state (image + instruction + history)
- $a_t^i$: 第 $i$ 维 action token
- $a_t^{1:i-1}$: 已经生成的第 1 到 $i-1$ 维 token,作为上下文
- $d_A$: action 维度 (这里 = 12)

这个自回归分解是 Q-Transformer 思想的关键:把多维 continuous action 离散化为 token 序列,就能用 transformer 的 next-token prediction 机制实现 Q-learning。**注意,后面的 Q-function 也按这种 autoregressive 方式展开**,而不是直接对 12 维联合动作做 Q 估计(否则动作空间太大没法做 max)。

---

## 三、Mixture of LoRA Experts 详解

### 3.1 MixLoRA 结构

paper 用的是 MixLoRA (Li et al. 2024) 思路,核心公式 (6):

$$E_k(x) = (W_{down} + W_{down}^{LoRA_k}) \, f\big( (W_{up} + W_{up}^{LoRA_k}) \, x \big)$$

变量:
- $E_k(x)$: 第 $k$ 个 expert 的输出
- $W_{up}, W_{down}$: FFN 的 up-projection 和 down-projection,**所有 expert 共享**
- $W_{up}^{LoRA_k}, W_{down}^{LoRA_k}$: 第 $k$ 个 expert 独有的 LoRA adapter (低秩矩阵 $BA$,其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$,$r \ll d$)
- $f(\cdot)$: 激活函数 (通常 SwiGLU 内部的 SiLU)

这意味着每个 expert 只在 LoRA 参数上有差异,base FFN 参数是共享的。这种设计相比传统 MoE (每个 expert 一个完整 FFN) 极大节省参数量,而且让 sparse activation 的"激活参数"和 dense model 量级相当。

### 3.2 Router 机制

公式 (2) 和 (3):

$$y = \sum_{k=1}^{N} G(x)_k \, E_k(x)$$

$$G(x) = \text{Softmax}(\text{TopK}(W_g x))$$

变量:
- $N$: expert 总数
- $G(x)_k$: 第 $k$ 个 expert 的 gate weight
- $W_g$: router 的线性投影, $W_g \in \mathbb{R}^{N \times d}$
- TopK: 只保留最大 K 个值,其他设 $-\infty$ 然后 softmax 归零

paper 没明确说 K,从图 3 看应该 K=2 或 K=1 这种稀疏配置。

### 3.3 Attention 上的 LoRA

paper 还在 self-attention 的 Q, K, V, O projection 上各加了一个 LoRA adapter (单一,不是 MoE),只在每个 decoder layer 用一套。整个 backbone (32 层 transformer 的 base 参数) 全程 frozen,只有 LoRA 在动。

### 3.4 Load Balancing Loss

公式 (9) 借鉴 Switch Transformer:

$$\mathcal{L}_{MoE} = \frac{1}{N} \sum_{k=1}^{N} f_k P_k$$

$$f_k = \frac{1}{T} \sum_{x \in \mathcal{D}} \mathbb{1}\{\arg\max p(x) = k\}$$

$$P_k = \frac{1}{T} \sum_{x \in \mathcal{D}} p_k(x)$$

变量解读:
- $f_k$: 实际被分到 expert $k$ 的 token 比例 (硬分配)
- $P_k$: router 给 expert $k$ 的平均概率 (软分配)
- $T$: batch 内总 token 数

这个 loss 的直觉:让 $f_k$ 和 $P_k$ 都均匀 (即都接近 $1/N$),乘积最小化时 $f_k = P_k = 1/N$,所以 loss 推动负载均衡。$\beta = 0.002$ 是个小权重。

参考:
- MixLoRA: https://arxiv.org/abs/2404.15159
- MoE-LLaVA: https://arxiv.org/abs/2401.15947
- LLaVA-MoLE: https://arxiv.org/abs/2401.16160
- Switch Transformers: https://arxiv.org/abs/2101.03961
- LoRA 原文: https://arxiv.org/abs/2106.09685

---

## 四、Task Structure 分析 (Section III-C 的核心 Insight)

paper 识别了 4 个 MDP structural properties,这才是整个方法的真正 motivation:

### 4.1 Horizon-independent Returns

reward 只在任务成功时给 +1,其他时刻都是 0。这意味着 return 不依赖 trajectory 长度,所以 bootstrapping 时 $\gamma \max Q(s_{t+1}, a_{t+1})$ 的传播非常直接,不会因为 horizon 长导致信号衰减得太厉害(只要成功附近 Q 大)。

### 4.2 Limited Critical Points

一条 trajectory 中只有少数几个 state 是 critical (例如 "crawl" 任务里俯身的瞬间,"unload" 任务里倾倒的瞬间),大部分 state 偏离最优轨迹都能恢复,因为:
- near-optimal trajectory 体积大
- 或者 "good-enough" action 的集合在大部分 state 上都很宽

这点很关键。它意味着 RL 在 non-critical state 上可以"随便选"一个好的就行,而 IL 必须模仿 expert 在该 state 的具体动作。如果 expert data 不够 cover 各种 non-critical state,IL 在 OOD 时就崩。RL 因为有 reward 信号,知道哪些"随便选"的 action 也是 good-enough。

### 4.3 Long-horizon Data

trajectory 很长,意味着 BC 的 covariate shift 问题严重(compounding error):一旦偏离 expert 轨迹,后续 state 都没见过,错误累积。RL 的 bootstrapping 在这种场景下反而稳健。

### 4.4 Distribution Shifts between Dataset and Evaluation

collect data 用 random policy 或 scripted policy,evaluation 用学习到的 policy,state 分布不一致,这正是 offline RL 要解决的核心问题,IL 在这里会失败。

---

## 五、Auto-regressive Discrete Q-Learning

### 5.1 Bellman Operator 修改

公式 (7) 是 Q-Transformer 的核心 trick,我详细拆开:

对于 $i < d_A$ (即还没生成到最后一维):

$$Q(s_t, a_t^{1:i-1}, a_t^i) = \max_{a_t^{i+1}} Q(s_t, a_t^{1:i}, a_t^{i+1})$$

意思:第 $i$ 维 token 的 Q-value = 下一维 token 选择最优时的 Q-value。这是"内部" max,在同一 time-step 内沿 action 维度链式展开 Q 值,把 high-dim action 的 max 拆成 1D max 的级联。

对于 $i = d_A$ (生成到最后一维,要跳到下一 time-step):

$$Q(s_t, a_t^{1:d_A-1}, a_t^{d_A}) = R(s_t, a_t) + \gamma \max_{a_{t+1}^1} Q(s_{t+1}, a_{t+1}^1)$$

这里 $\max_{a_{t+1}^1}$ 对下一 time-step 的第一维 token 做 max,然后会沿链传播到下一 time-step 的所有 $d_A$ 维。

整体直觉:把 12 维联合 action 的 max 拆成 12 次 1D max(对 vocabulary 做 max),这样既保持了 transformer 自回归生成的能力,又能 tractable 地做 Q-learning 的 Bellman backup。

### 5.2 Conservative Q-Loss

公式 (8):

$$\mathcal{L}_{RL} = \frac{1}{2} \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\beta(a|s)} \big[ \big( Q(s,a) - \mathcal{B}^* Q^k(s,a) \big)^2 \big] + \alpha \cdot \frac{1}{2} \mathbb{E}_{s \sim \mathcal{D}, a \sim \tilde{\pi}_\beta(a|s)} \big[ \big( Q(s,a) - 0 \big)^2 \big]$$

变量:
- $\mathcal{D}$: offline dataset
- $\pi_\beta(a|s)$: behavior policy (采集数据时用的策略) 的 action 分布
- $\mathcal{B}^* Q^k$: Bellman target,用上一次迭代的 Q-network $Q^k$ 算
- $\alpha = 0.5$: conservative 正则系数
- $\tilde{\pi}_\beta(a|s) = \frac{1}{Z(s)} (1.0 - \pi_\beta(a|s))$: **反行为分布**,在 dataset 中低密度的 action 上权重高
- $Z(s)$: 归一化常数 $\sum_a (1 - \pi_\beta(a|s))$

第一项:标准 Bellman error,在 in-distribution action 上拟合 target。
第二项:conservative 项,把 out-of-distribution (低密度) action 的 Q 值拉向 0,防止 Q 函数对没见过的 action 过分高估。这是 CQL 思想的简化版,只不过这里是按 token 级别做。

### 5.3 Q-Value 怎么从 LM logit 来

$$Q(s, a) = \sigma(P_{LM}(a | s))$$

- $P_{LM}(a|s)$: LM 对 action token 的预测概率(在 vocabulary 上的 softmax 输出)
- $\sigma$: sigmoid,把概率压到 (0, 1) 区间,让 Q 值有界,这有利于 training stability

这是 Q-Transformer 的 trick:LM 的 next-token logits 直接当 Q-value,因为 action 已经被 tokenize 成 vocab 里的 token,Q-learning 的 max over action 变成 max over vocab tokens,自然。

### 5.4 Total Loss

公式 (10):

$$\mathcal{L} = \mathcal{L}_{RL} + \beta \mathcal{L}_{MoE}$$

$\beta = 0.002$,MoE balance loss 权重很小。

参考:
- Q-Transformer 原文: https://arxiv.org/abs/2309.10150
- GeRM (该 group 之前工作): https://arxiv.org/abs/2403.13358
- CQL 原文: https://arxiv.org/abs/2006.04779

---

## 六、实验数据深入解读

### 6.1 Main Results (Table I)

| Method | Params | Distinguish | Go to | Go avoid | Go through | Crawl | Unload | Avg |
|--------|--------|-------------|-------|----------|------------|-------|--------|-----|
| CLIP | 86M | 0.44 | 0.43 | 0.45 | 0.19 | 0 | 0 | 0.25 |
| VC-1 | 307M | 0.46 | 0.43 | 0.45 | 0.31 | 0 | 0 | 0.28 |
| QUART | 8B | 0.66 | 0.60 | 0.53 | 0.41 | 0.32 | 0.12 | 0.44 |
| MoRE | 9.82B | **0.82** | **0.80** | **0.59** | **0.57** | **0.49** | **0.33** | **0.60** |

几个观察:
1. CLIP/VC-1 这种小 vision encoder baseline 在 Crawl/Unload 上完全失败 (0.00),说明 whole-body manipulation 任务对小 model 来说太难
2. QUART (同组前一版工作) 已经用 Fuyu 8B,平均 44%,MoRE 在它基础上 +16% 平均提升
3. MoRE 参数 9.82B vs QUART 8B,只多 1.82B 但提升巨大,说明 MoE 的稀疏激活是"花小钱办大事"
4. "Crawl" 和 "Unload" 这种 hard task 的提升尤其大 (0.32→0.49, 0.12→0.33),这两类任务最依赖 whole-body coordination,正是 MoE 多专家分化 + RL 训练目标能发挥的地方

### 6.2 Ablation Study (Table II)

| Method | S-Data | Distinguish | Go to | Go avoid | Go through | Crawl | Unload | Avg |
|--------|--------|-------------|-------|----------|------------|-------|--------|-----|
| QUART | N | 0.66 | 0.60 | 0.53 | 0.41 | 0.32 | 0.12 | 0.44 |
| w/o RL | N | 0.73 | 0.67 | 0.58 | 0.47 | 0.34 | 0.24 | 0.51 |
| w/o MoE | Y | 0.70 | 0.63 | 0.55 | 0.45 | 0.37 | 0.18 | 0.48 |
| w/o S-Data | N | 0.78 | 0.69 | 0.61 | 0.53 | 0.45 | 0.28 | 0.56 |
| MoRE | Y | 0.82 | 0.80 | 0.59 | 0.57 | 0.49 | 0.33 | 0.60 |

关键 ablation 结论:

**(1) w/o RL (44 → 51)**: 即使纯 expert data,用 RL loss 也比 IL loss 好 7 个点。这印证了 paper 反复强调的 "RL > IL" 论点,即使在 expert-only setting 下。原因是 RL 的 Bellman backup 让模型学到"在 critical state 上选好动作,non-critical state 上选 good-enough",而 IL 只学动作模仿。

**(2) w/o MoE (51 → 48)**: 去掉 MoE 但保留 S-Data,从 51 掉到 48。说明 MoE 的多专家分化对 multi-task 的冲突缓解很重要。

**(3) w/o S-Data (60 → 56)**: 去掉 sub-optimal data 掉 4 个点,证明 RL 利用 sub-optimal data 确实有效。值得注意的副作用:Go avoid 任务 (0.59 → 0.53) 略微下降,paper 解释是 sparse reward 让 RL 学起来困难。

**(4) QUART vs w/o RL**: QUART 44, w/o RL 51,差距 7 个点完全来自 RL loss 替代 IL loss,这部分是 MoE 之外的主要 contribution。

### 6.3 Training Cost

- 8× A100 GPU
- Expert data: 3 epochs, ~100 hours
- Mixed data: ~125 hours (略多 25 小时,因为 sub-optimal data 多了 24%)

考虑到 9.82B 模型,这个训练成本算合理。对比 RT-2 (Google's 55B PaLI-X),这里小一个数量级。

---

## 七、关键 Reference 联想

### 7.1 与 RT-2 / OpenVLA 的对比

RT-2 (https://arxiv.org/abs/2307.15818) 是 Google 的 VLA 开山之作,把 PaLM-E / PaLI 当 backbone,在 robot 数据上 co-finetune。它纯 IL,且用的 PaLI 55B 巨大。OpenVLA (https://arxiv.org/abs/2406.09246) 是开源版,7B Llama 2 + DINO/SigLIP vision encoder,也是 IL。

MoRE 与它们的关键差别:
1. 用 RL 而非 IL
2. 用 MoE 而非 dense fine-tune
3. 应用在 quadruped (4 足) 而非 manipulator (机械臂)
4. 输出是 high-level 12 维 command,不是 joint angle,有 low-level controller (walk-these-ways) 垫底

### 7.2 与 Q-Transformer 的关系

Q-Transformer (https://arxiv.org/abs/2309.10150, Google 2023) 是 RT-2 同期工作,在 robot 上首次把 autoregressive Q-learning 做到 transformer。MoRE 的 RL loss 几乎直接搬 Q-Transformer,但创新点在于:
- 套到 MLLM (Fuyu 8B) 而非专门的小 transformer
- 结合 MoE
- 用在 quadruped long-horizon 任务而非 manipulation short-horizon

### 7.3 与 QUART / GeRM 同系列工作

QUART (https://arxiv.org/abs/2312.14457): 同 group 前一版,纯 IL,无 MoE,是 MoRE 的 baseline。
GeRM (https://arxiv.org/abs/2403.13358): 同 group 工作,首次把 RL Q-function 引入 quadruped VLA,但用的是 small model,MoRE 把它扩展到 large MLLM 上。

所以 MoRE 是 QUART (scale up via MLLM) + GeRM (RL training) + MoE (multi-task) 的合体。

### 7.4 walk-these-ways 作为 low-level controller

walk-these-ways (https://proceedings.mlr.press/v205/margolis23a.html) 是 MIT 的 Igor Margolis 工作,训了一个能处理多种 gait pattern 的 RL low-level policy,接受 (步态参数, 速度, 高度, etc.) 输入输出 12 关节 PD target。MoRE 复用它作为 low-level,这样 MoRE 不需要学习 leg control,只需要学 high-level command generation。这是一个 hierarchical RL 的典型 pattern。

参考:
- RT-2: https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246
- QUART: https://arxiv.org/abs/2312.14457
- GeRM: https://arxiv.org/abs/2403.13358
- walk-these-ways: https://proceedings.mlr.press/v205/margolis23a.html
- RT-X Open Embodiment: https://arxiv.org/abs/2310.08864
- PaLM-E: https://arxiv.org/abs/2303.03378
- SayCan: https://arxiv.org/abs/2204.01691
- RoboVQA: https://arxiv.org/abs/2312.11575
- AutoRT: https://arxiv.org/abs/2401.12963
- LISA: https://arxiv.org/abs/2308.00692
- Cobra (Mamba MLLM): https://arxiv.org/abs/2403.14520
- Parallel Decoding VLA (Song et al. 2025): https://arxiv.org/abs/2503.02310

---

## 八、Intuition 总结

我给你串一下整个方法的 mental model:

**起点问题**: quadruped robot 要做 multi-task VLA,数据来源混杂 (有人类 demo,有自动采集的失败轨迹),如何训一个 large model?

**为什么 MoE**: 多任务 dense fine-tune 会产生 interference (例如 crawl 任务的低姿态参数会和 navigation 的高速参数打架)。LoRA-based MoE 用稀疏 expert 分化,每个 expert 学习特定 action 维度/技能子集,通过 router 动态调度。共享 base FFN 保留通用能力,LoRA-only expert 保证参数效率。

**为什么 RL**: IL 只能复制 expert 行为,在 long-horizon + distribution shift + sparse reward 的场景下,covariate shift 让 IL 在 OOD state 上完全失效。RL 通过 Q-value 的 Bellman backup,可以从 reward 信号中学到 "哪些 non-critical state 上随便选就行",而 expert data 没法覆盖所有 non-critical state。

**为什么 RL + 这种 task 结构特别配**: paper 最有价值的是识别出 "horizon-independent returns + limited critical points" 这两个 property。前者让 Bellman backup 信号不衰减,后者让 RL 在大量 non-critical state 上"轻松"找到 good-enough action,所以即使是 sub-optimal data 也能榨取价值。

**为什么 auto-regressive discrete Q-learning**: 因为 LLM 的输出本质是 token 序列,把 action 拆成 12 维 token 后,Q-learning 的 max over joint action 变成 max over each token,可行。Q-value 直接从 next-token logits sigmoid 出来,模型结构和生成时一样,无缝衔接。

**为什么 Q(s,a) = σ(P_LM(a|s))** 而不是 raw logits: sigmoid 把 Q 压到 (0,1),配合 reward ∈ {0, 1} 和 γ<1,Q 值保持在合理范围内,数值稳定。

**conservative 项的作用**: offline RL 的核心痛点是 OOD action 上 Q 估计爆炸。$\tilde{\pi}_\beta$ 显式地把"在 dataset 中低密度"的 action 的 Q 拉向 0,让 max 不会找到瞎高估的 action。

**load balance loss 的作用**: MoE 训练经典问题——router collapse (所有 token 都路由到一个 expert)。$\mathcal{L}_{MoE}$ 让 expert 被选频率和路由概率都均匀,避免退化。

---

## 九、Potential Weaknesses & Open Questions

读完 paper 我有几个疑问:

1. **K (TopK 中的 K) 值没明确**: 这是 MoE 关键超参,影响 sparsity 程度和表达能力。paper 没说具体值,从图 3 推测可能是 K=2。

2. **Expert 数量 N 没明确**: 9.82B - 8B (Fuyu) = 1.82B 的 LoRA 参数,假设每个 LoRA rank=16, hidden=4096, 一个 LoRA ≈ 2×16×4096 ≈ 130K params per layer per projection。32 layer × 2 projection (up+down) × N experts × attention LoRA = 1.82B / 130K ≈ 14K,这数不对,可能 expert 数在 N=4 或 8 左右。

3. **Crawl/Unload 成功率仍低**: 0.49/0.33 虽然比 baseline 强,但绝对值不高。paper 解释 OOD 评估严苛,但说明 whole-body manipulation 这种 hard task 还远未解决。

4. **Sub-optimal data 占比小**: 440K vs 1.82M expert,只占 24%。RL 在这种"专家数据为主"的混合数据上能发挥多大价值,值得更深 ablation (比如纯 sub-optimal 数据集 vs 纯 expert)。

5. **Real-world 数据微调量没说**: paper 提到 "further fine-tuned on a small real-world dataset",但具体多少数据、训多久、是否模拟器到真实有 sim2real gap 的量化分析都没有。

6. **Q-Transformer 的 leave-one-out token masking 没提**: Q-Transformer 原文有关键的"对 invalid action mask 掉"的 trick,MoRE 这里通过 conservative loss 间接处理 invalid token (拉向 0),但效果是否等价不明。

7. **没有 vision encoder 与 Fuyu 的图像 tokenization 效率分析**: Fuyu 直接 patchify,对 robot 第一视角的 480×640 图像,token 数会很多,attention 计算成本高,paper 没讨论。

---

## 十、Final Thoughts

这篇 paper 的核心贡献是把"RL fine-tune 大 MLLM 当 Q-function + LoRA MoE + quadruped multi-task"这三件事缝合起来。单独看每件事都有 prior work (Q-Transformer for RL+transformer, MoE-LLaVA for MoE+MLLM, GeRM for RL+quadruped),但合体确实第一次。

从研究范式上看,这是 VLA 领域从"IL on expert demos"向"RL on mixed data"过渡的关键一步。Google 的 RT-2 走 IL 路线,OpenVLA 也是 IL,MoRE 提出大模型时代 RL fine-tune 的可行性,且证明即使纯 expert data 下 RL > IL,这对未来 robot foundation model 训练范式有指向意义。

如果你想深入复现/扩展,我会建议:
1. 复现 Q-Transformer 在 manipulator 上的结果,熟悉 auto-regressive discrete Q-learning
2. 复现 MixLoRA 在 multi-task VQA 上,熟悉 LoRA MoE 的 training dynamics
3. 把 walk-these-ways 的 low-level controller 拉下来,理解 high-level command 的语义
4. 在 Isaac Gym 里跑 QUARD 或 QUART-Open 框架,看 MoRE 在他们的 codebase 上做 adapter

希望这个讲解 build 你的 intuition。如果哪个部分想更深入(例如 Q-Transformer 的 Bellman backup 推导、MixLoRA 的 router gradient flow、或者 walk-these-ways 的 reward shaping),可以继续问我。
