---
source_pdf: FAST Efficient Action Tokenization for Vision-Language-Action Models.pdf
paper_sha256: 3739b31f5fecdde371509ff5bb13619979734e894a255a9b264253f4cc53934a
processed_at: '2026-08-04T06:49:53-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FAST 的人话版

## 一句话说清楚

机器人动作是一串连续数字，AI 想学会预测这些数字，得先把它们变成"词儿"。以前的做法太蠢了——把每个时间点的每个动作维度单独切成 256 个格子，就像把一句话拆成单个字母。高频控制下，相邻两个时间点几乎一样，模型学到的就是"抄上一步"，等于啥也没学。FAST 的办法是：先用 DCT 把动作信号变成频率域（大部分信息集中在低频），压缩掉冗余，再用 BPE 把高频那堆零压成短词。这样每个 token 都真有信息含量，模型才学得动。

---

## 为什么之前的做法不行

想象你在教一个学生画曲线。你给他的"训练数据"是：时间点 1 的 y 值是 0.10，时间点 2 是 0.11，时间点 3 是 0.12……采样率越高，相邻点差别越小。

考试题目：给前几个点，预测后面的点。

低频采样（每秒 5 个点）：相邻点差别大，学生得真理解曲线走向才能猜对。

高频采样（每秒 800 个点）：相邻点几乎一模一样。聪明学生直接"抄前一个答案"就能拿满分——但他其实根本没学会画曲线，只是在做复读机。

之前的 VLA action tokenization 就是这个困境。50Hz 双臂机器人，1 秒的动作变成 700 个 token，相邻 token 几乎重复。模型 next-token prediction loss 看着很低，其实啥也没学会。这就是为什么 OpenVLA 在低频数据上还行，跑到 DROID 15Hz 就废了。

---

## FAST 的核心直觉

关键洞察：**先压缩再 token化**。把冗余干掉，让每个 token 都"有话说"。

具体三步：

**第一步：DCT 变换。** 把时间域信号变成频率域。直觉上，机器人动作大多是平滑曲线偶尔有几个急转弯，这种信号在频率域里很稀疏——低频几个系数就能描述整体形状，高频系数基本是零。这跟 JPEG 压缩图像一个道理：照片里大部分区域颜色渐变，高频细节少，扔掉不影响视觉。

**第二步：量化。** 把 DCT 系数乘以 10 再取整。大部分高频系数本来就接近零，乘 10 取整后直接变 0。矩阵变得很稀疏。

**第三步：BPE 压缩。** 这步是精髓。BPE 本来是给文字用的：把经常一起出现的字母组合合并成一个词，比如 "the" 出现得多就合并成单 token。FAST 把它用在了频率系数上——高频位置那串连续的零被合并成短 token。最终输出从几百个整数压到约 30 个 token。

---

## 为什么这个思路对

之前每个 token 携带的信息约等于零，因为"知道第 t-1 步动作"几乎就能猜出第 t 步。FAST 之后，每个 token 携带的是"这段轨迹的整体形状是怎样的"、"高频细节如何"这类高阶信息。模型必须真理解动作的结构才能预测对。

一个微妙但重要的细节：**列优先 flatten**。把所有维度的最低频先放一起，再放次低频……预测顺序是从"整体形状"到"细节 refinement"。这跟人类画画先打草稿再描细节是一个道理，让 rollout 更稳。

---

## 实测数据说话

50Hz 双臂折衣服，naïve 要 700 token/chunk，FAST 只要 53 token，压缩 13 倍。

20Hz 单臂收餐桌，naïve 140 token，FAST 28 token，压缩 5 倍。

更狠的是：FAST 输出 token 数大概稳定在 30 per arm，几乎和频率无关。说明它压缩的是信号本身的复杂度，而不是被采样率绑架。

---

## 两个意外彩蛋

**第一，FAST+ 通用 tokenizer。** 在 1 百万条跨机器人动作轨迹上训练了一个 BPE 词表，覆盖单臂、双臂、移动机器人、不同控制频率、不同动作空间。拿来当黑盒用，没见过的机器人也能稳定压缩 2 倍以上。这意味着"机器人动作在频率域稀疏"是个跨 embodiment 的普适性质，未来所有 robot lab 都能直接用这一个 tokenizer，不用各自训 VQ-VAE。

**第二，干翻 diffusion VLA。** π₀ 用 flow matching + 专门 300M action expert，推理 100ms。π₀-FAST 用同一个 backbone 但 autoregressive 解码，推理 750ms 慢得多。但训练快了 5 倍，性能一样甚至更好。动辄几千 GPU 小时的大 VLA 训练里，训练快 5 倍是巨大节省。而且 autoregressive 版本更听话——diffusion 版本经常无视语言指令，autoregressive 版本更 follow。

---

## 这篇 paper 真正的 lesson

过去两年 VLA 领域普遍觉得：autoregressive 不行，dexterous control 得用 diffusion 或 flow matching 这种花哨 decoder。FAST 说：**问题不在 decoder，在你把 action 喂给 decoder 之前怎么表示它**。表示对了，最朴素的 next-token prediction 就够用。

这和 LLM 里 BPE 的故事一模一样。GPT-2 能 work 不只是因为 Transformer 架构，也因为 BPE 让 token 信息密度恰到好处。如果还停在 character-level，再大的 model 也难 scaling。

更深一层：**连续信号的 token化是个独立的研究问题，可以脱离具体 policy 单独优化**。FAST+ 作为通用 tokenizer 的意义，相当于 robot 领域的 GPT-2 BPE 词表——以后所有人都能 share 同一个 action vocabulary，跨 embodiment transfer 更容易。

---

## 几个我感兴趣的 open question

**推理慢的问题怎么解。** 750ms/chunk 在静态任务还能接受，动态任务（接球、抛接）就不行。LLM 领域有一堆现成技术：speculative decoding 用小模型起草大模型验证、quantization、KV cache 优化、parallel autoregressive。直接套应该能砍到 100ms 内。

**能不能把 FAST token 当 diffusion 的 target。** 现在扩散策略直接在 raw action 上加噪去噪。如果把 DCT 压缩表示当 latent space，可能比直接 diffuse raw action 更高效，因为噪声加在稀疏频率系数上比加在稠密时间序列上更"自然"。

**adaptive tokenization。** 复杂动作段（快速抓取）高频多，简单段（慢速移动）低频多。现在 FAST 每段固定长度，做成自适应分配应该更省。

**把"信息密度"formal 化成 loss。** FAST 其实是在隐式优化每个 token 的 marginal information content。能不能直接写成 $\sum_i I(T_i | T_{<i})$ 的 lower bound，end-to-end learn 出比 DCT 更好的 tokenizer？这是个很值得探索的方向。

---

参考链接：
- Paper: https://pi.website/research/fast
- Code: https://huggingface.co/physical-intelligence/fast
- π₀: https://arxiv.org/abs/2410.24164
- OpenVLA: https://arxiv.org/abs/2406.09246
- DROID: https://droid-dataset.github.io/

---

# FAST: 用时间序列压缩重新思考 Robot Action Tokenization

这篇paper来自Physical Intelligence（Pi），作者是Karl Pertsch和Kyle Stachowicz等人，paper website: https://pi.website/research/fast 。核心contribution非常focused：把robot action这个continuous time-series信号，先通过 **Discrete Cosine Transform (DCT)** 变到frequency domain做lossy compression，再用 **Byte Pair Encoding (BPE)** 做lossless compression，得到一组高信息密度的discrete action tokens，让autoregressive VLA（如PaliGemma-based π₀）能够在dexterous high-frequency control任务上训得动。

下面我尽量从first principles把intuition build up。

---

## 1. The Core Problem: 为什么 naïve binning 在高频下会崩

之前VLA（RT-1, RT-2, OpenVLA）的action tokenization非常朴素：对每个action dimension独立做256-bin uniform discretization，然后per-timestep flatten。形式化地：

$$\tau_a(a_{1:H}) = [T_{1,1}, \dots, T_{1,D}, \dots, T_{H,1}, \dots, T_{H,D}]$$

变量含义：
- $a_{1:H} \in \mathbb{R}^{H \times D}$：长度为 $H$ 的action chunk，每个timestep有 $D$ 维action
- $T_{t,d} \in \{1, \dots, 256\}$：第 $t$ timestep、第 $d$ 维action对应的bin index
- 总token数 = $H \times D$

高频控制下，比如50Hz bimanual（D=14），1秒chunk就是700个token。问题不止是token多，**真正的核心问题是marginal information坍缩**。

Autoregressive model训练目标是next token prediction，其learning signal正比于：

$$I(T_i \mid T_{1:i-1}) = H(T_i) - H(T_i \mid T_{1:i-1})$$

也就是给定前面所有tokens，第 $i$ 个token还剩多少不确定。对于smooth action trajectory，当control frequency $f \to \infty$ 时，$a_t \approx a_{t-1}$，于是 $T_{t,d} \approx T_{t-1,d}$，marginal information $\to 0$。

paper里Section IV那个toy experiment非常eloquent地展示了这点：他们用一个cubic spline interpolation任务（4个随机点插值出一条曲线），在不同sampling rate下训练autoregressive transformer预测curve。结果是 **naïve binning** 在低sampling rate下还能拟合，sampling rate一上去MSE就爆掉，最后模型干脆"copy第一个action"——进入了非常糟糕的local optimum。

这个observation其实和OpenVLA在DROID上失败的现象完全吻合：OpenVLA在低频BridgeV2/RT-1上还行，但跑到DROID（15Hz）就训不动。原因不在数据本身，**在tokenization让学习信号在marginal层面上消失了**。

---

## 2. The Key Insight: Action需要先被Compress

直觉上：如果连续token之间高度redundant，那做next token prediction就等价于让model学"复制粘贴"这个trivial mapping。我们需要的是把这种redundancy先compress掉，让每一个output token都携带**非平凡**的信息。

这其实就是language model里BPE的思想：常见sub-string合并成单token，让序列变短同时每个token携带更多information。FAST把这个思想扩展到continuous time-series。

关键选择：用DCT做compression。为什么是DCT？因为robot action trajectory在时间维度上通常是 **smooth with occasional sharp transitions**（比如抓取瞬间速度突变）。这种signal在frequency domain里是sparse的——绝大部分能量集中在低频。这和自然图像在JPEG里sparse是一个道理。

---

## 3. The FAST Pipeline: DCT + Quantize + BPE

### 3.1 Step-by-step walkthrough

参考paper的Algorithm 1和Figure 4：

**Step 1: Quantile Normalization**
对每个action dimension，把1st和99th quantile映射到 $[-1, 1]$。用quantile不用min/max是为了robust against outlier actions（大数据集里偶尔会有weird teleop inputs）。

**Step 2: Per-dimension DCT**
对每个action dimension $i$ 的sequence $a^i_{1:H}$ 独立做DCT，得到frequency coefficients：

$$C_j^i = \text{DCT}(a_{1:H}^i)_j$$

变量含义：
- 上标 $i \in \{1, \dots, D\}$：action dimension index
- 下标 $j \in \{0, \dots, H-1\}$：frequency index，$j=0$对应DC component（直流/均值），$j$ 越大频率越高

具体DCT-II公式（最常用variant，也是JPEG用的）：

$$C_j^i = \sum_{t=0}^{H-1} a_t^i \cdot \cos\!\left(\frac{\pi}{H}\left(t + \frac{1}{2}\right) j\right)$$

变量：
- $a_t^i$：第 $t$ timestep、第 $i$ 维的action value
- $H$：chunk长度
- $j$：frequency index
- 注意没有normalization系数（缩放在外面处理）

Inverse DCT (用于detokenize时)：

$$a_t^i = \sum_{j=0}^{H-1} w_j \, C_j^i \, \cos\!\left(\frac{\pi}{H}\left(t + \frac{1}{2}\right) j\right)$$

其中 $w_0 = \sqrt{1/H}$, $w_j = \sqrt{2/H}$ for $j > 0$。

**Step 3: Quantization (Scale-and-Round)**
$$\bar{C}_j^i = \text{round}(\gamma \cdot C_j^i)$$

变量：
- $\gamma$：scaling hyperparameter（默认10），控制fidelity vs compression的trade-off
- $\bar{C}_j^i \in \mathbb{Z}$：integer-valued quantized coefficient

$\gamma$ 大→更精细、压缩率低；$\gamma$ 小→更aggressive、有loss。Paper的实验中 $\gamma = 10$ 对所有dataset都用同一个值，说明这个hyperparameter不敏感。

**Step 4: Flatten (Column-first!)**
把 $D \times H$ 的quantized matrix展平成1D sequence。这里有个重要的design choice——**column-first**：

$$[\bar{T}_k] = [\bar{C}_1^1, \bar{C}_1^2, \dots, \bar{C}_1^D, \bar{C}_2^1, \dots, \bar{C}_H^D]$$

也就是先把所有dimension的最低频（$j=1$）放一起，再放所有dimension的次低频（$j=2$），依此类推。

为什么column-first而不是row-first？因为autoregressive prediction时，**先确定整体shape（低频），再refine细节（高频）** 这种coarse-to-fine的顺序让rollout更稳定。这很像图像生成里coarse-to-fine的多尺度生成思路（比如MaskGIT、或者DALL-E的从低分辨率到高分辨率）。

**Step 5: BPE Compression**
对flattened integer sequence训练BPE tokenizer，vocabulary size = 1024（默认）。BPE会merge频繁出现的integer组合，特别是大量0的runs（因为高频coefficient被quantize后基本都是0）。最终输出：

$$[\bar{T}_1, \dots, \bar{T}_{\bar{k}}] = \text{BPE}([T_1, \dots, T_k], \phi)$$

$\phi$ 是BPE dictionary，$\bar{k} \ll k$。这一步是 **lossless** 的，可以perfectly recover flatten的integer sequence。

### 3.2 Why this works intuitively

可以把FAST的composition看成：

| Stage | 操作 | 性质 |
|---|---|---|
| DCT | Time→Frequency domain | Lossless orthogonal transform |
| Quantize | Float→Int | Lossy，但sparse化 |
| Flatten | 2D→1D | Reorder，决定prediction order |
| BPE | Int seq→Token seq | Lossless compression |

最终每个output token携带的marginal information很高，因为redundancy被两层compression消掉了。Autoregressive model再也不是在学"复制上一帧"，而是在学"先选shape，再refine detail"这种语义上有意义的prediction order。

---

## 4. Compression Numbers: 实测对比

Table I给出了不同domain的压缩率（1-second chunk）：

| Dataset | Action Dim | Freq | Naïve tokens | FAST tokens | Compression ratio |
|---|---|---|---|---|---|
| BridgeV2 | 7 | 5Hz | 35 | 20 | 1.75x |
| DROID | 7 | 15Hz | 105 | 29 | 3.6x |
| Table Bussing | 7 | 20Hz | 140 | 28 | 5.0x |
| T-Shirt Fold | 14 | 50Hz | 700 | 53 | 13.2x |

非常醒目的pattern：**FAST output token数大致是 30 tokens per arm**，与频率几乎无关。这暗示FAST捕捉到了trajectory complexity本身，而非被采样率绑架。低频信号在naive下省token，FAST下也省；高频信号在naive下爆炸，FAST下还是30 tokens左右。

---

## 5. Universal Tokenizer: FAST+

这是个非常实用的副产品。他们在1M real robot action trajectories上训练了一个universal BPE vocabulary，覆盖：

- **Morphologies**: single-arm (Franka, UR5, WidowX), bi-manual (ARX, AgileX, Trossen, ALOHA), mobile (Fibocom, Mobile Trossen, ARX slate mobile)
- **Action spaces**: joint space, EE world frame, EE camera frame（同一robot多种parameterization）
- **Frequencies**: 5Hz ~ 50Hz
- **Open datasets**: DROID, BridgeV2, OpenX-Embodiment

具体数据mixture见paper Appendix A的表格，weight从0.9%到11.2%不等。所有action都先padding到32维以容纳不同dimensionality。

发布在HuggingFace上：https://huggingface.co/physical-intelligence/fast ，用起来3行代码：

```python
from transformers import AutoProcessor
tokenizer = AutoProcessor.from_pretrained(
    "physical-intelligence/fast", trust_remote_code=True
)
tokens = tokenizer(action_chunk)
```

测试结果（Figure 8）显示，即使在 **训练时没见过的robot setup** 上（dexterous hands、humanoids、autonomous driving、UMI等），FAST+也能稳定压缩2x以上。这暗示robot action的frequency-domain sparsity是个 **universal property**，跨embodiment都成立——这一点对未来的cross-embodiment generalist policy非常关键。

---

## 6. Experimental Results

### 6.1 Tokenizer对比（Figure 6）

7个评估环境：
- **Libero** (sim, 4 suites): 各方法都还行，FAST略好
- **Table Bussing** (20Hz UR5): naïve完全失败（学不动），FAST/FSQ都work
- **T-Shirt Folding** (50Hz bimanual ARX): naïve完全失败，FAST最佳
- **DROID zero-shot**: FAST首次让generalist policy work

FSQ baseline（Finite Scalar Quantization, Mentzer et al. https://arxiv.org/abs/2309.15505）是一个learned VQ-VAE的simpler alternative。FAST在dexterous任务上甚至优于FSQ，尽管FAST不需要训练任何neural network。

### 6.2 对比Diffusion π₀（Figure 9, 11）

这是最有意思的对比。Diffusion π₀（Black et al., https://arxiv.org/abs/2410.24164 ）用的是flow matching + 300M action expert + 10 denoise steps。π₀-FAST用同一backbone但autoregressive decoding 30-60 tokens。

| 指标 | π₀ (diffusion) | π₀-FAST |
|---|---|---|
| 小数据集（Libero, T-Shirt Fold）性能 | 相当 | 相当 |
| 大数据集（Table Bussing）收敛速度 | 慢 | **3x更快**达到同等性能 |
| **总训练compute** | 1x | **5x更少** |
| DROID语言follow能力 | 经常忽略指令 | 更follow指令 |
| Inference speed per chunk (4090) | **100ms** | 750ms |
| Action expert参数 | 300M | 用full 2B backbone |

Inference慢的原因是autoregressive要逐token decode（30-60步），且用full LM backbone而不是dedicated small action expert。但训练快了5x，这对于动辄thousands of GPU hours的大规模VLA training来说是巨大节省。

Figure 15做了compute-matched对比：同样GPU小时数下，π₀-FAST明显优于π₀（diffusion）。

### 6.3 Ablations

**BPE ablation**: 没有BPE只用DCT，policy性能下降但仍然优于naive。原因：DCT之后大量高频token是0，autoregressive model要浪费很多step预测这些0，dilute learning signal且拖慢inference。BPE把这些0 run压缩掉，让每步预测都有意义。

**Backbone generalization**: 在OpenVLA（Prismatic 7B, https://arxiv.org/abs/2406.09246 ）上做T-Shirt Folding，原版OpenVLA用naive binning失败，换成FAST+就work。说明FAST是 **architecture-agnostic** 的，任何pre-trained autoregressive transformer都能直接plug-in。

### 6.4 Scaling to 10k hours

π₀-FAST在cross-embodied mixture（903M timesteps自采集 + 9.1% BridgeV2/DROID/OXE）上训练，能匹配diffusion π₀的性能，包括最难的Laundry Folding（从basket取衣物→flatten→fold→stack，需要retry和correction）。

---

## 7. DROID Zero-Shot: A Telling Result

DROID dataset（Khazatsky et al., https://droid-dataset.github.io/ ）是大规模in-the-wild manipulation data。之前的works（包括原DROID paper和OpenVLA）都没展示真正的zero-shot unseen environment evaluation，都是co-training或fine-tune。

FAST训练的policy是**第一个能在完全unseen environment（新桌子、新背景、新物体、新视角、新桌子高度）下zero-shot perform manipulation task的DROID policy**。跨3个university campus测试（Berkeley, Stanford, UW），能做pick/place, open/close drawer, turn on faucet等任务。即使失败也是"sensible behavior"（比如approach了微波炉把手但没拉开）。

这个result在我看来比单纯的high-frequency dexterity更impressive，因为它说明action tokenization的改进**直接translate到generalization**。原因可能是：每个action token携带更多information → model capacity被更有效地用于learning task-relevant patterns而非记住redundant per-timestep action values。

---

## 8. Limitations & Open Questions

**Inference latency**: 750ms/chunk让autoregressive VLA在dynamic tasks上不可用（比如juggling, catching）。LLM literature有很多现成技术可以套用：
- Speculative decoding（Leviathan et al. https://arxiv.org/abs/2211.17192 ）：用小model起草，大model verify
- Quantization（GPTQ, AWQ, SmoothQuant）
- KV-cache + Flash Attention的custom kernels
- Medeed decoding / parallel autoregressive

**Universal tokenizer泛化到locomotion/humanoid**: paper的offline实验显示FAST+对humanoid (Unitree H1, 40-60Hz) 和dexterous hand都有不错压缩率，但实际policy performance没测。这个我很期待看到。

**BPE替代选择**: paper提到Huffman coding或Lempel-Ziv（gzip底层）也可以用，留作future work。BPE的好处是已经有highly optimized implementation，且能直接inject进VLM vocabulary。

**Combination with diffusion**: 一个非常自然的idea是——FAST token能不能作为diffusion policy的target？把DCT-compressed representation作为diffusion的"latent space"，可能得到比直接diffuse raw action更高效的diffusion VLA。

**Non-causal prediction**: 既然DCT coefficient matrix sparse且大部分是0，或许可以用masked prediction（mask掉一部分高频coefficient让model填）而不是left-to-right autoregressive。这能让inference从sequential变成parallel。

**Adaptive tokenization**: 类似ElasticTok（Yan et al., https://arxiv.org/abs/2410.08368 ）的思路——简单action segment少给token，复杂segment多给token。FAST目前每个chunk固定长度，但trajectory的frequency content分布是高度非uniform的（fast运动段高频多，slow段低频多），adaptive分配应该更efficient。

---

## 9. Big Picture: 这篇paper意味着什么

我觉得这篇paper其实揭示了关于VLA的一个deep lesson：**modeling的瓶颈往往不在architecture而在representation**。

过去两年VLA领域一直认为autoregressive VLA天然不适合dexterous control，需要diffusion或者flow matching这种expressive decoder。所以π₀用flow matching，OpenVLA-1.x探讨过diffusion head，等等。FAST说：**问题不在autoregressive decoding本身，而在你把action表征成什么token给它预测**。如果token信息密度太低，再fancy的decoder也救不了；如果token信息密度高，最朴素的next-token prediction就能work。

这和LLM里BPE的lesson是平行的——GPT-2能work不只是因为Transformer架构，也因为BPE让token vocabulary和information density恰到好处。如果LLM还在character-level（GPT-1 char-level），再大的model也难以scaling。

更深一层：这暗示**continuous signal的tokenization本身是一个独立的、可重用的、需要单独study的研究问题**，可以脱离任何具体policy architecture。FAST+作为universal tokenizer的意义类似GPT-2的BPE——以后所有robot lab都可以直接拿来用，而不用每个group自己train一个VQ-VAE。

我个人觉得最exciting的方向是：
1. FAST token作为**robot领域的universal vocabulary**，类似自然语言的BPE token，所有policy都能share同一个action vocabulary → 跨embodiment transfer更容易
2. 把FAST思想推到**video generation**（time-series的另一种形态），可能比当前VQ-based video tokenizer更简单有效
3. **Compression = learning signal density** 这个principle本身可以formal化成一个loss：比如直接优化 $\sum_i I(T_i | T_{<i})$ 的lower bound，可能end-to-end learn出一个比DCT更好的tokenizer

paper里有一个细节很值得关注：**column-first flatten比row-first好**。这个coarse-to-fine prediction order的发现其实和image generation（从low-res到high-res）以及LLM的planning（先plan skeleton再fill details）的精神是一致的。可以考虑把这种prediction order做成learned的，让model自己discover最优order。

---

## 10. Reference Links

- Paper website (with videos): https://pi.website/research/fast
- π₀ paper: https://arxiv.org/abs/2410.24164
- π₀-FAST code (HuggingFace): https://huggingface.co/physical-intelligence/fast
- OpenVLA: https://arxiv.org/abs/2406.09246
- DROID dataset: https://droid-dataset.github.io/
- DCT原始paper (Ahmed, Natarajan, Rao 1974): https://ieeexplore.ieee.org/document/1454974
- JPEG standard (Wallace 1992): IEEE Transactions on Consumer Electronics
- BPE (Sennrich, Haddow, Birch 2015): https://arxiv.org/abs/1508.07909
- Original BPE (Gage 1994): The C Users Journal
- VQ-VAE (van den Oord et al. 2017): https://arxiv.org/abs/1711.00937
- FSQ (Mentzer et al. 2023): https://arxiv.org/abs/2309.15505
- ALOHA / ACT (Zhao et al. 2023): https://arxiv.org/abs/2304.13705
- Diffusion Policy (Chi et al. 2023): https://arxiv.org/abs/2303.04137
- BridgeData V2: https://arxiv.org/abs/2308.12952
- OpenX-Embodiment: https://arxiv.org/abs/2310.08864
- ElasticTok (Yan et al. 2024): https://arxiv.org/abs/2410.08368
- PaliGemma (Beyer et al. 2024): https://arxiv.org/abs/2407.07726
- Speculative decoding: https://arxiv.org/abs/2211.17192

---

整体来看，FAST在我看来是2024-2025 VLA领域最重要的method contribution之一，因为它把一个被architecture-level思考主导的问题重新frame成representation-level问题，并且给出了一个shockingly simple的solution（DCT+BPE，两个上世纪的classic algorithm）。这种"用对的representation让问题消失"的taste，正是好的systems thinking的体现。
