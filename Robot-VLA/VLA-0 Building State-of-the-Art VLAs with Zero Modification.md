---
source_pdf: VLA-0 Building State-of-the-Art VLAs with Zero Modification.pdf
paper_sha256: 13517e323b3c661c7d89de05375ac04d0f904c9eda05f6fea4b2b845a55c7fa3
processed_at: '2026-08-13T02:41:21-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 VLA-0

Andrej, 我用最直觉的方式再讲一遍，省掉学术包装，只留 insight。

---

## 这个领域一直在干什么

过去两年大家做 VLA，基本思路是：拿一个 VLM，想办法让它输出 robot action。问题就在"想办法"这三个字上——大家绞尽脑汁搞了三条路：

**第一条路（RT-2, OpenVLA）**：把 action 离散成 256 个 bin，每个 bin 占词表里一个 token。听起来简单，但你想想，"抓香蕉"和"抓苹果"的 gripper position 可能差 0.3mm，256 个 bin 根本分辨不出来。而且你把 VLM 词表里 256 个 token 抢过来当 action bin，VLM 本来学到 "apple" 这个 token 的 semantic embedding 全被污染了。

**第二条路（π₀, SmolVLA）**：VLM 输出一个 latent vector，再接一个 diffusion head 把它 decode 成 action。等于在 VLM 屁股后面又挂了一个小网络。问题是这个小网络是 from scratch 训的，它和 VLM 的 representation 对不齐，训着训着 VLM 的 language 能力就被 drag 下去了。Physical Intelligence 自己在 π₀.₅ paper [9] 里都承认这个现象，专门搞了个 "knowledge insulation" 来缓解。

**第三条路（OpenVLA-OFT, π₀-FAST）**：改架构、改 tokenizer、搞 DCT 压缩。效果不错但 pipeline 贼复杂。

---

## VLA-0 干了什么

啥也没干。真的。

让 VLM 直接把 action 当 text 输出。比如 action 是 `(0.847, 0.523, 0.911)`，就让它输出字符串 `"847 523 911"`。完事了。

- 没加新 token
- 没改词表
- 没加 action head
- 没改架构
- base model 是现成的 Qwen-VL-2.5-3B [18]

整个 "方法" 就是写了个 system prompt：

```
Analyze the input image and predict robot actions for the next H timesteps.
Each action has D dimensions. Output a single sequence of H × D integers (0 - B each),
representing the H timesteps sequentially. Provide only space-separated numbers.
Nothing else.
```

然后 fine-tune Qwen-VL 让它学会吐这串数字。Loss 就是普通的 cross-entropy，和你训 GPT 写莎士比亚一模一样。

---

## 为什么这居然能 work — 我的直觉

这是整篇 paper 最值得思考的部分。我觉得有三层原因：

### 第一层：VLM 的 decoder 本来就是超强的 sequence model

Diffusion policy [4] 干的事是什么？在 action space 上做 sequence modeling，学一个 $p(a_{1:H} \mid \text{observation})$。但 LLM decoder 干的就是 sequence modeling 啊！而且它已经在几万亿 token 上训过了，比任何 from-scratch diffusion head 都强。你加 action head 反而是在用一个弱模型替换一个强模型。

### 第二层：Text representation 自带结构

这点很 subtle。OpenVLA 用 `<act_847>` 这种 reserved token 表示 action bin 847。这个 token 在 VLM 词表里是个完全陌生的 symbol，它的 embedding 是随机初始化的，模型对它没有任何 prior。

VLA-0 用 `"847"` 这个字符串，它是三个 character token：`"8"`, `"4"`, `"7"`。这三个 token 在 VLM 里已经被训练了几万亿次——VLM 见过无数 coordinate、无数 phone number、无数 measurement。它对十进制数字的 compositionality 是有理解的。`"847"` 对 VLM 来说不是 arbitrary symbol，而是 "8 个百 + 4 个十 + 7"，这个 structure 是 free 的。

这就像你教小孩认数字，用阿拉伯数字比用一套全新符号系统容易得多，因为阿拉伯数字的 syntax 已经在小孩脑子里了。

### 第三层：不污染词表 = 保留 grounding

OpenVLA 把 256 个 action bin 塞进词表，等于强行告诉 VLM "以后看到这个 token 别再想到 cat/dog/banana 了，想到的是 gripper position"。VLA-0 完全不动词表，VLM 原本的 vision-language grounding 一点没丢。这或许就是它 generalization 好的根本原因——它真的是在"看图说话"，只不过说的话恰好是数字。

---

## 三个让 VLA-0 真正 SOTA 的小把戏

光靠"action as text"还不够，作者加了三个 tricks：

### Trick 1: Integer normalization

把连续 action $a \in [a_{\min}, a_{\max}]$ 映射到 $[0, B]$ 的整数：

$$
a_{\text{int}} = \text{round}\left( \frac{a - a_{\min}}{a_{\max} - a_{\min}} \cdot B \right)
$$

- $a_{\min}, a_{\max}$：该 action 维度在数据集里的 min/max
- $B$：resolution，LIBERO 上用 $B = 1000$

Ablation 显示 $B=250$ 不够（-1.5），$B=4000$ 没必要（-0.5），$B=1000$ 是 sweet spot。1000 意味着每个数字 3-4 个 character，token 开销适中。

这里有个 elegant 的点：和 discrete token VLA 比，VLA-0 的 resolution 是"免费的"。Discrete token VLA 想要 10000 bin 就要占词表 10000 个位置，根本不可能；VLA-0 想要 10000 resolution 只是每个数字多一个 character，词表纹丝不动。

### Trick 2: Temporal ensemble（最重要的一个）

borrow 自 ACT [23]。每个 time step $t$，模型预测未来 $n$ 步的 action chunk $\hat{a}_{t:t+n}$。当你要 execute time $t$ 的 action 时，你不只用当前预测，而是把过去 $n$ 步里所有对 $t$ 的预测平均：

$$
\tilde{a}_t = \frac{1}{n} \sum_{i=0}^{n-1} \hat{a}_t^{(t-i)}
$$

- $\hat{a}_t^{(t-i)}$：在 time step $t-i$ 时模型预测的、关于未来第 $i$ 个位置的 action
- $n$：chunk size

这个 trick 贡献 +2.0 success rate，是三个里最大的。Intuition：VLM 单次预测有 noise（autoregressive sampling 的 stochasticity），但 noise 在 time 上是独立的，averaging 就能抹掉。Diffusion policy 里的 multi-step denoising 其实也起到类似作用——让 output 在 time 上 smooth。

但这里有个 deployment 问题：real-world 实验里作者没用 ensemble，因为要 8 个并行 model instance，5090 跑不动。所以 real-world 结果可能没达到 VLA-0 的 full potential。

### Trick 3: Masked action augmentation（最 clever 的一个）

这个我觉得是 paper 里最 underappreciated 的 trick。

问题：VLM 是 autoregressive 的，用 teacher forcing 训练，每个 token 看前面所有 token。Action string 长这样：`"847 523 911 402 ..."`。模型很容易学到 shortcut——"前面是 847 523，那下一个数字大概率和前两个有某种 numerical pattern"，而不是真的去看图像。

这就像你让小孩做数学题，他学会了看答案的 pattern 猜下一个数字，而不是真的算。

解法：训练时随机把 target string 里的 character 替换成 mask：

$$
\tilde{s}_i = \begin{cases} \texttt{<mask>} & \text{w.p. } p \\ s_i & \text{otherwise} \end{cases}
$$

然后照常算 CE loss。这样模型不能依赖前面已生成的内容做 auto-completion，必须真的从 image + instruction 推理。

贡献 +1.2 success rate。数字不大，但 concept 很对。这等于把 BERT-style masking 思想搬进 autoregressive training，破坏 spurious correlation，强制 cross-modal grounding。

---

## 实验结果有多 striking

LIBERO benchmark [14]，四个 suite（Spatial / Object / Goal / Long），每个 10 个 task。

### 公平比较（都没有 large-scale action pretraining）

| Model | Avg Success | Avg Rank |
|---|---|---|
| Diffusion Policy | 72.4 | 6.5 |
| π₀-FAST (PaliGemma) | 71.8 | 6.0 |
| SmolVLA 0.24B | 82.8 | 5.3 |
| SmolVLA 2.25B | 88.8 | 4.0 |
| OpenVLA-OFT | 91.9 | 2.8 |
| π₀.₅-KI | 93.3 | 2.3 |
| **VLA-0** | **94.7** | **1.0** |

VLA-0 第一，而且平均 rank 是 1.0，意味着它在几乎所有 suite 上都是最好或并列最好。

### 不公平比较（对手有 large-scale action pretraining，VLA-0 没有）

| Model | Avg Success | Avg Rank |
|---|---|---|
| Octo | 75.1 | 8.8 |
| OpenVLA | 76.5 | 8.0 |
| π₀-FAST | 86.0 | 6.5 |
| MolmoAct | 86.8 | 6.5 |
| GR00T-N1 | 93.9 | 4.5 |
| π₀ | 94.2 | 3.3 |
| π₀.₅-KI | 94.3 | 3.0 |
| OpenVLA-OFT | **97.1** | **1.5** |
| VLA-0 | 94.7 | 2.8 |

VLA-0 打败了 GR00T-N1、π₀、π₀.₅-KI 这些有大数据加持的 model，仅次于 OpenVLA-OFT（一个 custom architecture，而且它这个版本也是 pretrained 的）。

**这意味着什么**：要么 large-scale action pretraining 的作用被严重高估了，要么现有 VLA 架构的 inductive bias 是错的。我赌是后者——大家加的 action head / custom tokenizer 反而限制了 VLM 的 reasoning 能力。pretrain 再多 data 也救不回架构本身的 bottleneck。

### Real-world：SO-100 robot + LeRobot

四个 task，每个 100 demos，VLA-0 from scratch vs SmolVLA（pretrained on large-scale SO-100 data）：

- Block reorient: VLA-0 ~85 vs SmolVLA ~75
- Push apple: ~95 vs ~85
- Pick-place banana: ~75 vs ~60
- Pick-place cupcake: ~70 vs ~55
- **Avg: VLA-0 比 SmolVLA 高 12.5 points**

SmolVLA 是专门在 SO-100 大规模数据上预训练的，VLA-0 从零开始训 100 个 demo 就反超 12.5 个点。这个 result 在 real world 比simulation 还 striking。

LeRobot: https://github.com/huggingface/lerobot

---

## 这对领域意味着什么

我觉得这篇 paper 的 meta-message 比 technical contribution 更重要：**过去两年 VLA 领域堆的复杂度可能是 over-engineering**。

大家默认"VLM 不能直接输出 action，因为 action 是连续的、高精度的、需要 special handling"。这个 assumption 没人认真检验过。VLA-0 检验了，发现 assumption 是错的。VLM 完全可以直接输出 action，而且比加了各种 special handling 的方法还好。

这让我想起你（Karpathy）之前在 nanoGPT [1] 里表达的哲学——先把 simplest thing that could possibly work 做透，再谈 complexity。VLA-0 把"simplest thing"做完了，结果是 SOTA。

参考链接：
- nanoGPT: https://github.com/karpathy/nanoGPT
- original blog "The simplest thing that could possibly work" 思路

---

## 我个人的延伸联想

### 1. Chain-of-thought + VLA-0 是 obvious next step

VLA-0 输出的是 text，那完全可以在 action 前面加 reasoning text。比如：

```
The banana is at approximately (300, 200) in the image. The plate is at 
(500, 400). I need to first move the gripper above the banana, then descend,
then grasp, then lift, then move to plate, then release.
Action: 300 200 150 0 1 ...
```

这种 CoT-augmented VLA 只有 VLA-0 这种 design 能做，因为其他 design 的 action head 是单独的 module，插不进 reasoning。这或许是 VLA-0 最大的 unexplored potential。

参考 CoT 在 LLM 上的 work：
- Chain-of-Thought: https://arxiv.org/abs/2201.11903
- 最近 reasoning model 的 trend（o1, R1 等）

### 2. Inference speed 是 VLA-0 的真实软肋

Output 是字符串，autoregressive decode 速度慢。每个 integer 3-4 characters，$H \times D$ 个 integer，总长度可能几百 characters。5090 上 4 Hz，real-time control 勉强够。

π₀-FAST [16] 用 DCT 压缩 action chunk 就是为了解决这个——把 100 维 action 压成 10 个 token。VLA-0 牺牲了速度换 simplicity。

可能的解法：
- Speculative decoding: https://arxiv.org/abs/2211.17192
- Distillation to smaller model
- Quantization（FP8, INT4）

### 3. Multi-modal action distribution 没讨论

Text generation 是 unimodal sampling。Diffusion head 可以 naturally 表达 multi-modal action distribution（同一个 observation 下多种 valid action）。当 task 有 ambiguity 时（比如可以从左边抓也可以从右边抓），VLA-0 怎么处理？可能它会 collapse 到 mode averaging，这是 diffusion policy 早期也被诟病的问题。Paper 没讨论这个，是个 open question。

### 4. 大规模 action pretraining + VLA-0 会怎样

作者承认没试。如果 VLA-0 + Open-X scale [2] 的 action data 预训练，会怎样？我赌会刷爆所有 benchmark，因为 VLA-0 的架构没有 bottleneck，pretrain 的收益能完整传导。这可能是下一篇 SOTA paper 的配方。

- Open-X: https://robotics-transformer-x.github.io/

### 5. 这和 LLM 算术的类比

你之前讲过 LLM 做 arithmetic 的 story——大家以为需要 calculator tool，结果 scale + 训练数据够多，autoregressive model 自己就学会了隐式算法。VLA-0 是 robotics 版的同一现象：大家以为需要 action head / special tokenizer，结果 VLM 直接输出数字就行。

### 6. 为什么 NVIDIA 的人能发现这个

我觉得是因为他们没有被"必须发 fancy method"的 incentive 绑架。大多数人发 paper 要 sell 新架构、新 module、新 loss。VLA-0 没有任何"新东西"可 sell，纯靠实验结果说话。这种 work 在顶会评审里其实吃亏，因为 reviewer 喜欢 novelty。NVIDIA 这个团队有 luxury 去 publish 一个"没 novelty 但 work"的结果，这是很多 academic group 没有的 freedom。

---

## 一句话总结

**VLA-0 揭穿了一个集体幻觉**：过去两年 VLA 领域堆的架构复杂度可能是错的 direction。最朴素的方法——让 VLM 把 action 当数字字符串直接输出——配上三个 cheap tricks（integer normalization + temporal ensemble + masked augmentation），就能 SOTA。这个 work 的价值不在"方法新"，而在"把 simplest baseline 做对，让所有人重新审视自己的复杂度是否必要"。

对你（Karpathy）来说，这篇 paper 应该是" vindication"——你一直说 simplest thing first，VLA-0 就是 robotics 版的 nanoGPT。下一步我最想看的是 VLA-0 + CoT reasoning，这会是 reasoning robotics policy 的 natural path。

---

### Key links

- VLA-0 project page: https://vla0.github.io/
- Qwen-VL-2.5: https://arxiv.org/abs/2412.15115
- LIBERO benchmark: https://arxiv.org/abs/2306.03310
- LeRobot: https://github.com/huggingface/lerobot
- ACT (ensemble source): https://arxiv.org/abs/2304.13705
- π₀: https://arxiv.org/abs/2410.24164
- π₀.₅ / Knowledge Insulation: https://arxiv.org/abs/2504.16054
- π₀-FAST (DCT tokenizer): https://arxiv.org/abs/2501.09747
- OpenVLA: https://arxiv.org/abs/2406.09246
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- SmolVLA: https://arxiv.org/abs/2506.01844
- GR00T-N1: https://arxiv.org/abs/2503.14734
- RT-2 (original discrete token VLA): https://arxiv.org/abs/2307.15818
- LLARVA (closest prior work): https://arxiv.org/abs/2406.11815
- Diffusion Policy: https://arxiv.org/abs/2303.04137

---

# VLA-0: 用最朴素的方式把 VLM 变成 VLA

Andrej, 这篇 paper 真的非常对你"simplest thing that could possibly work"的胃口。NVIDIA 的人做了一个几乎让人感到尴尬的实验：直接让 VLM 把 action 当成 text 输出（space-separated integers），零架构修改、零新 token、零 action head，结果在 LIBERO 上 SOTA，甚至吊打一堆带 large-scale action pretraining 的模型。这种"皇帝的新衣"式的工作正是 robotics + foundation model 这个领域最需要的。

项目主页：https://vla0.github.io/

---

## 1. 这篇 paper 在挑战什么 dogma

当前 VLA 领域基本默认了三条路线（论文 Fig. 2 给了清晰分类）：

| Family | 代表方法 | 关键操作 | 痛点 |
|---|---|---|---|
| Discrete Token VLA | RT-2, OpenVLA, MolmoAct | 把连续 action 分箱，每箱占用 VLM 词表里一个 token | 分辨率 vs. 词表冲突；语义污染 |
| Generative Action Head VLA | π₀, SmolVLA, GR00T-N1, Octo | VLM 输出 latent，再接 diffusion / flow matching head 解码 | 增加新网络，损害 VLM 原有 grounding |
| Custom Architecture VLA | OpenVLA-OFT (ACT head), π₀-FAST (DCT tokenizer) | 改架构或换 tokenizer | 训练 pipeline 复杂 |

VLA-0 的 claim 极其简单：以上三种都有 cost，而最 obvious 的方案——让 VLM 直接吐 integer string——居然没有人认真试过。LLARVA [15] 试过类似的，但它是 two-stage：先预测 2D trajectory plan，再 predict action。HAMSTER [13] 也是先 predict 2D trajectory text，再下游处理。VLA-0 是真正的 end-to-end direct action-as-text。

---

## 2. 方法：三个"小把戏"叠出 SOTA

base VLM 用的是 **Qwen-VL-2.5-3B** [18]（选择理由：3B 体积小、openweight、competitive）。

### 2.1 Action Decoding — 把连续 action 投到 [0, B] 整数

对每个 action 维度 $a \in \mathbb{R}$，做 normalization：

$$
a_{\text{int}} = \text{round}\left( \frac{a - a_{\min}}{a_{\max} - a_{\min}} \cdot B \right), \quad a_{\text{int}} \in \{0, 1, \dots, B\}
$$

- $a_{\min}, a_{\max}$：该维度在 dataset 里的 min/max（per-dim normalization）
- $B$：resolution hyperparameter，LIBERO 上 $B=1000$ 最佳
- 输出序列长度：$H \times D$，其中 $H$ 是 horizon（predict 多少 future steps），$D$ 是 action dimension

关键 insight：这里和 discrete token VLA 的本质区别在于——discrete token VLA 是把"bin index"映射到词表里某个 reserved token，所以分辨率受词表大小约束；VLA-0 是让 VLM 用**多 character 的十进制字符串**来表示这个 integer，比如 "847" 是三个 ASCII characters。分辨率 $B$ 可以任意大，因为只是 string 长度增加，词表完全不动。这点非常 elegant。

System prompt：
```
Analyze the input image and predict robot actions for the next H timesteps.
Each action has D dimensions. Output a single sequence of H × D integers (0 - B each),
representing the H timesteps sequentially. Provide only space-separated numbers.
Nothing else.
```

### 2.2 Ensemble Prediction — 借用 ACT 的 temporal smoothing

ACT [23] 和 OpenVLA-OFT [10] 都在用这个技巧。在 time step $t$，模型预测一个长度为 $n$ 的 action chunk $\hat{a}_{t:t+n}$。对于当前要执行的 action $\tilde{a}_t$，可以拿过去 $n$ 步里所有关于 $t$ 的预测做平均：

$$
\tilde{a}_t = \frac{1}{n} \sum_{i=0}^{n-1} \hat{a}_t^{(t-i)}
$$

- $\hat{a}_t^{(t-i)}$：在 time step $t-i$ 时模型预测的、对应未来第 $i$ 个位置的 action
- $n$：chunk size

这本质是个**temporal smoothing filter**，把 high-frequency jitter 抹掉。Diffusion policy 里类似的 multi-step denoising 也起到类似的稳定作用。Ablation 显示它单独贡献 +2.0 success rate，是三个 tricks 里最关键的。

### 2.3 Masked Action Augmentation — 这是我觉得最 clever 的部分

VLM 是 autoregressive decoder，训练时用 teacher forcing，每个 token 都看前面所有 token。问题是：action string 是一个高度结构化的数字序列（比如 `847 523 911 402 ...`），模型很可能学到的不是"从图像推理 action"，而是"我前面生成了 `847 523`，那下一个数字大概率是 911"——也就是 **copy distribution / shortcut learning**。

Masked Action Augmentation 的做法：训练时随机把 target string 里的 character 替换成 mask（或直接删掉），强迫模型不能依赖前面已生成的内容做 auto-completion，必须真正从 visual observation + instruction 推理。形式上，对 ground-truth string $s = (s_1, s_2, \dots, s_L)$，以概率 $p$ 把每个 character 替换成 `<mask>`：

$$
\tilde{s}_i = \begin{cases} \texttt{<mask>} & \text{w.p. } p \\ s_i & \text{otherwise} \end{cases}
$$

然后照常算 cross-entropy loss：

$$
\mathcal{L} = - \sum_{i=1}^{L} \log p(s_i \mid s_{<i}, I, T)
$$

- $s_i$：第 $i$ 个 character
- $s_{<i}$：前面所有 character（其中一部分被 mask 掉）
- $I$：image(s)
- $T$：task instruction text

Ablation：去掉 → -1.2 success rate。增量不大但 consistent。

**Intuition**：这个 trick 等于把 BERT-style 的 masking 思想搬到 autoregressive training 里，破坏 spurious correlation，强制 cross-modal grounding。这其实和 "drop previous tokens" / "prefix LM" 的一些 trick 有异曲同工之处。

### 2.4 Image Input 选项

两种都试了：
1. Multi-image：每张图作为独立 visual token 喂给 VLM
2. Tiled image：把多张图 tile 成一张 composite image 喂进去

Ablation 显示两者几乎没差（Row 5: -0.2）。这个结果其实有点 surprising，因为 tiling 通常会损害 spatial resolution。可能 Qwen-VL-2.5 对 tiling 已经 robust。

---

## 3. 实验结果：让人重新审视整个 VLA 领域

### 3.1 LIBERO 主表（Table I）

最 striking 的对比：

**无 large-scale action pretraining 这一组（公平比较）：**

| Model | Avg Succ | Avg Rank |
|---|---|---|
| Diffusion Policy | 72.4 | 6.5 |
| π₀-FAST (Paligemma) | 71.8 | 6.0 |
| SmolVLA (0.24B) | 82.8 | 5.3 |
| SmolVLA (2.25B) | 88.8 | 4.0 |
| OpenVLA-OFT | 91.9 | 2.8 |
| π₀.₅-KI | 93.3 | 2.3 |
| **VLA-0** | **94.7** | **1.0** |

VLA-0 在四个 suite（Spatial/Object/Goal/Long）上全面领先。

**有 large-scale action pretraining 这一组（不公平比较，VLA-0 仍无 pretraining）：**

| Model | Avg Succ | Avg Rank |
|---|---|---|
| Octo | 75.1 | 8.8 |
| OpenVLA | 76.5 | 8.0 |
| π₀-FAST | 86.0 | 6.5 |
| MolmoAct | 86.8 | 6.5 |
| GR00T-N1 | 93.9 | 4.5 |
| π₀ | 94.2 | 3.3 |
| π₀.₅-KI | 94.3 | 3.0 |
| OpenVLA-OFT | **97.1** | **1.5** |
| VLA-0 | 94.7 | 2.8 |

VLA-0 居然打败了 GR00T-N1、π₀、π₀.₅-KI 这些带 big data pretraining 的 model，仅次于 OpenVLA-OFT（一个 custom architecture）。

**这个结果的 implication**：要么 large-scale action pretraining 的收益被高估了，要么现有很多 VLA 架构本身的 inductive bias 是错的。我倾向于后者——这些 model 加的 action head / custom tokenizer 反而限制了 VLM 本身的 reasoning 能力。

参考链接：
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- π₀: https://arxiv.org/abs/2410.24164
- π₀.₅: https://arxiv.org/abs/2504.16054
- GR00T-N1: https://arxiv.org/abs/2503.14734
- SmolVLA: https://arxiv.org/abs/2506.01844

### 3.2 Real-World：SO-100 + LeRobot

四个任务：reorient block / push apple / pick-place banana / pick-place cupcake。每个任务 100 demos。

| Task | SmolVLA | VLA-0 |
|---|---|---|
| Block reorient | ~75 | ~85 |
| Push apple | ~85 | ~95 |
| Pick-place banana | ~60 | ~75 |
| Pick-place cupcake | ~55 | ~70 |
| **Avg** | ~70 | **+12.5 points** |

SmolVLA 是被 large-scale SO-100 data 预训练过的，VLA-0 是 from scratch。Inference 用 5090 GPU，4 Hz，没用 ensemble（要 8 个并行 instance）。

LeRobot repo: https://github.com/huggingface/lerobot

---

## 4. Ablation 深入解读（Table II）

| Row | Ensemble | Masked Aug | Tiled | Resolution | Avg Succ | Δ |
|---|---|---|---|---|---|---|
| 0 | ✓ | ✓ | ✓ | 1000 | 94.7 | baseline |
| 1 | ✗ | ✓ | ✓ | 1000 | 92.0 | -2.0 |
| 2 | ✓ | ✗ | ✓ | 1000 | 93.5 | -1.2 |
| 3 | ✓ | ✓ | ✓ | 4000 | 94.2 | -0.5 |
| 4 | ✓ | ✓ | ✓ | 250 | 93.2 | -1.5 |
| 5 | ✓ | ✓ | ✗ | 1000 | 94.5 | -0.2 |

观察：
- **Ensemble 是最大的 lever**（-2.0）：这暗示 VLA-0 单次预测其实噪声不小，靠 averaging over time 才稳定。这也是为什么 real-world 没用 ensemble 让我有点担心——4 Hz + no ensemble 在真实环境是否 robust 还需要更多验证。
- **Masked Aug 第二大**（-1.2）：说明 autoregressive shortcut learning 确实存在。
- **Resolution sweet spot 是 1000**：太低（250）丢精度，太高（4000）无收益甚至略降。1000 等于每个数字 3-4 characters，token 数量适中。
- **Tiling vs. multi-image 无差**：Qwen-VL-2.5 对输入 format 不敏感，这是好事。

---

## 5. 我的 Intuition 和为什么这能 work

你（Karpathy）应该会对这个有共鸣——这跟 LLM 刚开始做 math 时大家的惊讶很像：人们以为 LLM 算算术需要 calculator tool、需要 program-aided reasoning，结果发现 GPT-4 直接输出数字也能算 7 位数乘法（虽然有错）。原因：scale + 足够的 training data 让 autoregressive model 学到了隐式的算法。

VLA-0 在做类似的事：把 action prediction 重新 cast 成 VLM 已经被训练得极好的事情——生成结构化 text。任何 VLM 都已经在海量 web data 上见过数字、坐标、空间描述。当你说 "predict 7 integers"，它不需要新 capability，只需要 fine-tune mapping。

更深层的原因，我觉得有几条：

1. **VLM 的 text decoder 本身就是个超强的 sequence model**。Diffusion policy / flow matching head 在 action space 上做的 sequence modeling 工作，LLM decoder 早就做得很好了。加 head 反而是 redundant + 信息 bottleneck。
2. **词表污染是 VLA 表现差的重要原因**。OpenVLA 把 256 个 action bin 硬塞进 VLM 词表，破坏了原本的 token embedding 结构。VLA-0 完全不动词表，保留了 VLM 全部的 language grounding。
3. **Text representation 自带 hierarchical structure**。"847" 比 token `<act_847>` 更 informative——前者是 3 个 digit token，每个 token 都有 well-trained embedding，模型可以从十进制结构里推理；后者是一个 OOV 的孤立 token。
4. **Cross-entropy on text 是 well-understood loss**。Diffusion / flow matching 在 small data 上容易过拟合或训不稳，CE loss 在 fine-tune regime 下非常 stable。

---

## 6. 局限与未尽的 question

作者自己承认的 limitation：
- 没试 large-scale action pretraining（这是最大的 open question——如果 VLA-0 + big data 会怎样？）
- Inference 速度只有 4 Hz，没用 quantization / distillation

我额外想到的几个：
- **Ensemble 在 real world 用的 cost 太高**：要 8 个并行 instance，real-time 受限。这是 deployment 上的大问题。
- **String decoding 速度本身就是 bottleneck**：3-4 characters per integer × H × D 的输出长度，autoregressive 解码会很慢。这正是 π₀-FAST 用 DCT 想解决的问题——把 action chunk 压成几个 token。VLA-0 牺牲了速度换取 simplicity。
- **Long horizon 任务（Long suite）提升相对小**：87.6 vs OpenVLA-OFT 的 94.5（pretrained 版本）。这暗示 VLA-0 在 long-horizon planning 上可能弱，因为它本质是 reactive policy。
- **没有 failure mode analysis**：VLA-0 错的时候是错在哪？是 visual grounding 错还是 number formatting 错？这决定了下一步改进方向。
- **Action distribution 上的 multi-modality**：text generation 是 unimodal sampling（greedy/temperature），diffusion head 可以 naturally 表达 multi-modal action distribution。当 task 有多种 valid action 时，VLA-0 怎么处理？Paper 没讨论。

---

## 7. 相关延伸联想

- **π₀-FAST [16]**：用 DCT 把 action chunk 压成几个 token，再用离散化。其实和 VLA-0 思路相反——它想"压缩"，VLA-0 想"用最 native 的 representation"。两者可以结合：DCT 后的 coefficients 用 text 输出？
  - Link: https://arxiv.org/abs/2501.09747

- **CogACT / Emu3 / Show-o**：这些 work 也在探索"everything as token"，但更多是 image generation 方向。VLA-0 是 robotics 上的对应。

- **LLaRA / RT-2**：最早把 action 当 token 的尝试，但因为用 reserved token 牺牲了语义，效果一般。VLA-0 等于"用 character-level 而不是 token-level"避开了这个问题。
  - RT-2: https://arxiv.org/abs/2307.15818

- **你的 own work on nanoGPT / micrograd**：VLA-0 的哲学其实跟 nanoGPT 一样——把不必要的 complexity 删掉，看看 simplest architecture 能走多远。结果是 surprisingly far。

- **Reasoning + VLA**：如果 action 直接是 text，那能不能在 action 前面加 chain-of-thought？比如先输出 "the banana is at (300, 200), plate at (500, 400), I need to move gripper to banana first" 然后再 output action integers。这种 text-augmented VLA 可能是 VLA-0 的自然延伸，而且只有 VLA-0 这种 design 能做（其他 design 没法插 reasoning）。这一点 paper 没提，但我觉得是最大的潜力。

- **Knowledge Insulation [5]**：π₀.₅-KI 用的技巧，避免 action head 训练损害 VLM 的 language capability。VLA-0 天然没有这个问题，因为 action 就是 language。这或许是它能保留 VLM grounding 的根本原因。
  - Link: https://arxiv.org/abs/2505.23705

- **Open-world generalization**：π₀.₅ 主打 open-world。VLA-0 没测这个，但如果它的 VLM grounding 保留得更好，理论上 open-world 应该也强。

---

## 8. 一句话总结

VLA-0 用最 boring 的方法（让 VLM 直接吐整数）做 VLA，靠三个 cheap tricks（integer normalization + temporal ensemble + masked augmentation）达到 SOTA。这个 work 的真正贡献可能不是"方法新"，而是**揭穿了一个集体幻觉**：过去两年 VLA 领域堆的复杂度（action head, custom tokenizer, vocabulary surgery）可能都是 over-engineering，真正的 bottleneck 是 training recipe 和 representation choice，不是 architecture。

我觉得这正是你一直说的 "first do the simplest thing that works, then understand why" 的精神。VLA-0 把"first do the simplest thing"做完了，"understand why"还需要社区补完。

---

### 一些值得看的 related links

- 项目主页：https://vla0.github.io/
- LeRobot: https://github.com/huggingface/lerobot
- Qwen-VL: https://arxiv.org/abs/2412.15115
- LIBERO benchmark: https://arxiv.org/abs/2306.03310
- ACT (original ensembling): https://arxiv.org/abs/2304.13705
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- LLARVA (closest prior): https://arxiv.org/abs/2406.11815
- HAMSTER: https://arxiv.org/abs/2410.13736
- RT-2: https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246

如果你（Karpathy）接下来想 push 这个方向，我赌三件事会出 follow-up SOTA：(1) VLA-0 + CoT reasoning text before action，(2) VLA-0 + distillation 解决速度问题，(3) VLA-0 + big action pretraining 数据（Open-X scale）。第三件事最有可能直接刷爆所有 benchmark。
