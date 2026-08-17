---
source_pdf: Octopi Object Property Reasoning with Large Tactile-Language Models.pdf
paper_sha256: ccf1890b46458df796794be14bf789047f6032655ca680e4d0bac3045b3457be
processed_at: '2026-08-05T22:56:20-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 Octopi

## 一句话版本

让机器人装上"皮肤"，摸一摸东西，然后用大语言模型脑子里的常识去推理这玩意儿能干嘛。

## 为啥要做这件事

你现在让机器人挑个熟牛油果，它光靠摄像头看真看不出来——熟的和不熟的颜色长得差不多。但人一捏就知道了，熟的软。作者就想：**能不能给机器人也装上这个"捏一下"的能力，然后用LLM的常识把它串起来**。

这事的难点在于，现有的大模型(GPT-4V、Gemini、BLIP-2这些)全都是**视觉+语言**的，没人把**触觉**接进去过。视觉模态有歧义的时候，触觉能补上关键信息。

## 触觉传感器是啥

用的是 [GelSight](https://www.mdpi.com/1424-8220/17/12/2762) ——一块透明gel，表面有彩色涂层，里面有LED灯+一个小摄像头。你按到物体上，gel变形，摄像头拍到彩色变形图。本质上是把"摸"转换成"看一张变形图"。硬的东西按下去gel变形小，软的变形大；粗糙的表面会留下细密纹理，有凸起的会留下大块印记。

## 数据集 PHYSICLEAR

作者手工摸了74个日常物品（baseball、毛巾、牛油果、牙刷毛、剪刀柄……），每个物品采好几段视频，每段视频包含两种动作：
- **按压**(pressing) → 主要感受硬度
- **旋转**(rotation) → 主要感受粗糙度和凹凸

然后三个人独立标注三个property：
- **Hardness**: soft / moderately hard / hard
- **Roughness**: smooth / slightly rough / rough  
- **Bumpiness**: no bumps / small bumps / big bumps

总共408个video、1200多个标注。 annotator一致性(ICC)在0.79~0.98之间，bumpiness最低是因为它是从GelSight图像上**看**出来的，不是**摸**出来的——这是个annotation modality不一致的小坑。

跟之前的GelSight dataset比起来，PHYSICLEAR是唯一一个**既有property label又有property多样性又有object多样性**的，参考[Touch and Go](https://arxiv.org/abs/2211.12498)和[ObjectFolder 2.0](https://arxiv.org/abs/2204.02389)都没做到这三点。

## 模型架构 Octopi

整个pipeline分三步训：

### 第一步：教CLIP认识GelSight图

直接用预训练的[CLIP](https://arxiv.org/abs/2103.00020) ViT-L/14不行，因为CLIP是看natural image长大的，GelSight图长得完全不像——颜色诡异、纹理怪异。所以要在PHYSICLEAR上fine-tune。

为了不破坏CLIP学到的good representation，用[VPT (Visual Prompt Tuning)](https://arxiv.org/abs/2203.12119)——把transformer backbone全冻住，只在每一层前面插8个learnable prompt token，加3个分类头分别预测hardness/roughness/bumpiness，cross-entropy监督。

公式上每帧过encoder再average pool得到video embedding $\mathbf{v} = \frac{1}{N}\sum_n \text{CLIP}(\mathbf{X}_n)$，然后三个head分别softmax出3类。

### 第二步：把触觉embedding对齐到LLM词空间

CLIP输出是1024维，Vicuna词向量是4096或5120维。中间加个projection module（两个linear+GELU），仿照[LLaVA](https://arxiv.org/abs/2304.08485)：

$$\mathbf{z}_{\text{tact}} = W_2 \cdot \text{GELU}(W_1 \mathbf{v} + \mathbf{b}_1) + \mathbf{b}_2$$

这里encoder和LLM都冻住，只训projection + 两个新special token `<tact_start>` 和 `<tact_end>` 的word embedding。这两个token是告诉LLM"这里开始/结束一段触觉信号"。

### 第三步：端到端fine-tune

encoder继续冻住，projection + LLM用[LoRA](https://arxiv.org/abs/2106.09685)一起训。LoRA rank=128, alpha=256 (ratio=2)，dropout=0.05。LLM用[Vicuna v1.5](https://lmsys.org/blog/2023-03-30-vicuna/)，有7b和13b两个版本。

LoRA公式 $\Delta W = \frac{\alpha}{r}BA$，$r=128$算是比较高的rank，因为tactile modality和language modality差距太大，需要足够的adaptation capacity。

## 五个测试任务

| 任务 | 干啥 | 训练 | 测试 | Random基线 |
|---|---|---|---|---|
| OPD | 描述一段触觉视频的property | ✓ | ✓ | - |
| PC | 给两段视频，比较哪个更硬/更粗糙 | ✓ | ✓ | 33.33% |
| PSS | 给三段视频，选最硬/最光滑的 | ✓ | ✓ | 33.33% |
| POM | 给三段视频+三个物体名，匹配 | ✓ | ✓ | 16.67% |
| PSR | 给两段视频+一个真实场景问题，选合适的 | ✗ | ✓ | 50% |

PSR是最考验的——模型从没见过这种prompt格式，必须靠前面四个任务学到的property grounding + LLM自己的commonsense组合出来。比如：

> "Which object is most suitable for removing stains from a non-stick pan without scratching it?"

模型得自己想到——擦锅要不伤涂层，需要soft + smooth + no bumps，然后看两段视频的描述哪个符合。

## 最关键的发现：中间描述很重要

这是这篇paper最重要的实验结论。作者对比了两种模式：
- **with OPD**: 让模型先描述property再回答问题
- **without OPD**: 让模型直接回答

结果差距巨大。比如PSS任务上OCTOPI-7b从74.67%掉到39.88%，POM从44.39%掉到23.23%。13b在PSR上甚至从67.39%掉到39.13%，比random还低。

**直觉解释**：触觉信号本身在LLM脑子里是"模糊的"，必须先翻译成语言property（"hard, smooth, no bumps"）这个中间表示，LLM才能用它的commonsense去推理。如果直接让它从触觉embedding跳到结论，它的language prior会overpower触觉证据，特别是13b这种commonsense更强的LLM——它会"自信地"根据自己的prior乱猜，反而忽视触觉。

这种"perception → property-language → reasoning"的三段式其实是个通用template，可以推广到其他modality。

## 真机器人测试：挑牛油果

把两个GelSight装在[Franka Emika Panda](https://www.franka.de/)机械臂上，测10个牛油果，每个采20次触觉，100对pairwise比较"哪个更熟"。

结果OCTOPI-13b在ripeness pairwise上达到63%，property prediction(只按不转)上combined accuracy 35.5%（random 3.7%）。对比只用视觉的[PG-InstructBLIP](https://arxiv.org/abs/2309.02561)在avocado property prediction上几乎random(0%)——视觉上avocado真的差不多。

这里有个zero-shot reasoning prompt让LLM先自己rank哪个property重要：
> "Which of these properties help to determine avocado ripeness? Rank them."

OCTOPI-13b回答："hardness and bumpiness matter, roughness is not reliable. Ripe avocado will be moderately hard with small bumps, unripe will be hard with no/small bumps."

这说明LLM的commonsense被property grounding激活了——它本来就知道ripe avocado软，但只有在被prompt去考虑这些property时才会调出这个知识。

## 消融实验的两个关键发现

### 1. CLIP必须fine-tune

直接用base CLIP做encoder，roughness和bumpiness大幅下降，OCTOPI-7b的roughness从73.68%掉到52.63%，bumpiness从81.58%掉到55.26%。但hardness反而base CLIP更高（81.58% vs 71.05%）——因为CLIP的edge detector对硬度变形很敏感，但对GelSight特有的texture pattern不熟，必须学。

### 2. LoRA必须加

不加LoRA，13b的combined accuracy从55.26%掉到23.68%，比7b的39.47%还低。直觉：13b的language prior太强，会"压制"projection module提供的触觉信号。LoRA让LLM学会"听"触觉token，而不是只依赖自己的language prior。

7b不加LoRA只掉8个点，13b不加LoRA掉31个点——模型越大越需要LoRA去align。

## 弱点和开放问题

1. **数据集太小**：74个物品，7个test物品，PSS的84%其实就差一个样本就变71%，variance很大。
2. **Property太窄**：只有3个property × 3类=27种组合，实际只出现15种。没温度、没滑、没弹性、没重量。
3. **OPD的ground truth是ChatGPT生成的**，有弱监督bias风险。
4. **63%的avocado准确率实际部署不够用**，只是pairwise比较。
5. **没换sensor测generalization**：GelSight Mini换成Wedge还行不行？
6. **没做"纯LLM + 文字描述触觉"的baseline**——如果先让GPT-4V看GelSight图像描述一遍，再让LLM reason，能差多少？这能分离出"encoder学到了什么vision-language model没有的"。

## 最核心的intuition

这篇paper给我们的最大启发：

**物理property可以作为modality-specific perception和modality-agnostic reasoning之间的universal interface。**

任何一种新的sensing modality（触觉、本体感觉、温度、声学……），只要你能：
1. 找到一组离散的property category作为anchor
2. 这些category在LLM的language prior里有rich commonsense association

你就可以用同样的"perception → property-language → reasoning"三段式把它接进LVLM。触觉只是这个template的第一个instance。

代码和数据在 https://github.com/clear-nus/octopi 都开源了。

---

**一句话收尾**：Octopi不是发明了什么新架构，而是把已有的CLIP、LLaVA、LoRA这些积木拼起来，第一次认真回答了"机器人摸到东西之后，怎么用语言常识去推理"这个问题——答案是**先翻译成property语言，再让LLM干活**。这个simple but effective的设计思路值得借鉴。

---

# Octopi: Object Property Reasoning with Large Tactile-Language Models 深度解析

## 一、Big Picture 与动机直觉

这篇paper来自NUS的Harold Soh组，核心想做的事情是把**触觉(tactile modality)**真正地嫁接到**Large Vision-Language Model (LVLM)** backbone 上，让robot不仅能"看"，也能"摸"——并通过摸得到的physical property去驱动commonsense reasoning。这一点很关键，因为现有的LVLM (Gemini, Qwen-VL, BLIP-2, MiniGPT-v2, Flamingo) 全部只在vision+language上工作，而**visual modality是 ambiguous 的**：一颗ripe avocado和unripe avocado在RGB图像上几乎不可分，但在GelSight pressure map上一眼就能看出硬度差异。

直觉上，作者把整个pipeline切成两段：
- **底层 perception**：把GelSight输出的RGB-like tactile image映射成LLM能消化的token embedding sequence
- **上层 reasoning**：让LLM用其commonsense knowledge把physical property与scenario绑定起来（比如"ripe avocado → soft → 选这颗"）

整个工作的trickiness在于**domain gap**：CLIP是在natural image上pretrain的，GelSight image长得完全不一样（高对比度elastomer deformation map、光照场是mounted LEDs、颜色偏紫绿），所以直接用CLIP编码GelSight frame效果差，必须先做encoder fine-tuning。

Reference:
- Paper repo: https://github.com/clear-nus/octopi
- arXiv: https://arxiv.org/abs/2405.14051 (Octopi)
- GelSight原始paper: https://www.mdpi.com/1424-8220/17/12/2762
- LLaVA: https://llava-vl.github.io/
- CLIP: https://arxiv.org/abs/2103.00020
- ViFi-CLIP: https://arxiv.org/abs/2212.05283
- VPT (Visual Prompt Tuning): https://arxiv.org/abs/2203.12119
- LoRA: https://arxiv.org/abs/2106.09685
- Vicuna: https://lmsys.org/blog/2023-03-30-vicuna/
- Physically-Grounded VLMs (PG-InstructBLIP): https://arxiv.org/abs/2309.02561

---

## 二、PHYSICLEAR Dataset 细节

### 2.1 Property 选择动机

作者只选了三个property：**hardness, roughness, bumpiness**。直觉上这是gelSight这种vision-based tactile sensor能感受到的最robust的物理量：
- **Hardness**: 受压时gel elastomer的deformation depth + lateral spread
- **Roughness**: 高频spatial variation（摩擦系数相关，对应小尺度texture）
- **Bumpiness**: 低频spatial protrusion（大尺度geometry）

每个property有3个category（见Table I）：
| Property | Categories |
|---|---|
| Hardness | soft / moderately hard / hard |
| Roughness | smooth / slightly rough / rough |
| Bumpiness | no bumps / small bumps / big bumps |

为什么不做geometric property (size) 或affective property (comfort)？作者明确说：GelSight传感器 sensitivity & durability 不够。这里其实有一个值得深挖的intuition——gel thickness 决定了能感知的spatial frequency band，太细的纹理会被gel smoothing掉，太大的形状会被gel saturate掉。所以这套property选择其实是 sensor physics-aware 的。

### 2.2 数据采集

- **74 objects**, **408 tactile videos**, **1200+ annotations**
- 采集是 by-hand（避免损伤sensor + irregularly-shaped objects难夹持）
- 每个object最多采7个video，每个video对应一个distinct region
- 两种exploratory procedure：
  1. **Pressing** → 拿到normal force下的deformation map（hardness信息主导）
  2. **Rotation** → 拿到shear force下的lateral displacement（roughness/bumpiness信息主导）

这点很重要：later的avocado实验只用了pressing，没有rotation，仍然能work——说明模型学到的是robust representation，而不是过拟合到特定motion pattern。

### 2.3 Annotation 质量

Inter-annotator agreement用的是 **ICC(3,k)** (Intra-class Correlation, two-way fixed, average of k raters):
- Hardness: 0.894
- Roughness: 0.979
- Bumpiness: 0.792

> 0.75 above = good/excellent reliability

为什么bumpiness最低？因为bumpiness的guideline是**视觉从GelSight image判断**的（"bumps are less than 1/4 of the tactile image"），而hardness/roughness是用真实haptic feedback判断的——annotation modality不一致导致方差。

### 2.4 与其他GelSight dataset对比

| Dataset | Property Labels? | Property Diversity | Object Diversity | Material Diversity |
|---|---|---|---|---|
| Hardness Dataset (2016) | Yes (only hardness) | Yes | Yes | Medium |
| Clothing Dataset (2018) | Yes | Yes | No (only clothing) | Low |
| ObjectFolder 2.0 (2022) | No | No (only hard) | Yes | Medium |
| Touch and Go (2022) | No | Yes | Yes | High |
| ObjectFolder-Real (2023) | No | No (only hard) | Yes | Medium |
| **PHYSICLEAR** | **Yes** | **Yes** | **Yes** | Medium |

PHYSICLEAR的独特之处是同时拥有**property labels + property diversity + object diversity**——前人dataset要么没label，要么只covering一种property。

参考Touch and Go: https://arxiv.org/abs/2211.12498
参考ObjectFolder 2.0: https://arxiv.org/abs/2204.02389

---

## 三、OCTOPI 架构深度解析

### 3.1 整体pipeline

```
GelSight frames (X_1, ..., X_N)
       ↓
[CLIP ViT-L/14 visual encoder, fine-tuned]  ← frozen after stage 1
       ↓
video-level embedding (avg-pooled)
       ↓
[Projection: Linear → GELU → Linear]  ← trainable
       ↓
tactile tokens [T_1, ..., T_M]
       ↓
concat with text tokens [W_1, ..., W_K] + <tact_start> + <tact_end>
       ↓
[Vicuna v1.5 LLM with LoRA]  ← trainable (LoRA only)
       ↓
text response
```

新加的两个special token `<tact_start>` 和 `<tact_end>` 是要训练的word embedding——这是为了让LLM知道"这里开始/结束一段触觉信号"。

### 3.2 Encoder Fine-tuning 阶段

**Stage 1目标**：让CLIP能在GelSight tactile image上extract useful representation。

#### ViFi-CLIP 架构

ViFi-CLIP的核心idea是把video当成frame sequence，每帧过CLIP visual encoder，然后average-pool得到video-level representation：

$$\mathbf{v} = \frac{1}{N} \sum_{n=1}^{N} \text{CLIP}_{\text{visual}}(\mathbf{X}_n)$$

其中：
- $\mathbf{X}_n$ 是第 $n$ 帧tactile image
- $\text{CLIP}_{\text{visual}}(\cdot)$ 是CLIP visual encoder (ViT-L/14)
- $\mathbf{v} \in \mathbb{R}^{d}$ 是video-level embedding，$d=1024$ for ViT-L/14
- $N$ 是frame数

#### Visual Prompt Tuning (VPT)

为了不破坏CLIP的pretrained representation，作者用VPT来inject少量learnable参数。在每个transformer layer $l$ 的input sequence前插入8个learnable prompt tokens：

$$\text{input}^{(l)} = [\mathbf{p}_1^{(l)}, \mathbf{p}_2^{(l)}, \ldots, \mathbf{p}_8^{(l)}, \mathbf{x}_1^{(l)}, \ldots, \mathbf{x}_L^{(l)}]$$

其中：
- $\mathbf{p}_k^{(l)} \in \mathbb{R}^{d}$ 是layer $l$ 的第 $k$ 个prompt token的embedding
- $\mathbf{x}_j^{(l)}$ 是原始token序列
- $L$ 是原始token数（patch tokens + CLS token）

还有一个shared linear layer作用在prompts上。Transformer backbone完全frozen，只train prompts + shared linear layer + 3个classification head。

#### Multi-task classification heads

3个head分别预测3个property：

$$\hat{y}^{(h)} = \text{softmax}(W_h \mathbf{v} + b_h), \quad \hat{y}^{(r)} = \text{softmax}(W_r \mathbf{v} + b_r), \quad \hat{y}^{(b)} = \text{softmax}(W_b \mathbf{v} + b_b)$$

其中 $W_\cdot \in \mathbb{R}^{3 \times d}$（3 classes per property），上标 $h, r, b$ 分别表示hardness, roughness, bumpiness。

总loss：

$$\mathcal{L} = \mathcal{L}_h + \mathcal{L}_r + \mathcal{L}_b = -\sum_{i}\sum_{c \in \{h,r,b\}} \sum_{k=0}^{2} y_{i,c,k} \log \hat{y}_{i,c,k}$$

其中 $y_{i,c,k}$ 是sample $i$ 在property $c$ 上class $k$ 的one-hot label。

#### Hyperparameters
- 30 epochs, AdamW, **no weight decay**
- lr = $10^{-3}$, batch size = 32
- cosine annealing schedule
- **<5GB VRAM**, 6 hours on single GPU

为什么no weight decay？这是个小细节，可能是因为frozen backbone + 少量trainable params，regularization需求低。

### 3.3 Tactile Feature Alignment 阶段

**Stage 2目标**：把encoder的output embedding align到LLM的word embedding space。

具体做法：
- **Discard** stage 1的classification heads
- 取CLIP visual encoder的output
- Train **projection module**: 2个linear layer + intermediate GELU

Projection module公式（参照LLaVA）：

$$\mathbf{z}_{\text{tact}} = W_2 \cdot \text{GELU}(W_1 \mathbf{v} + \mathbf{b}_1) + \mathbf{b}_2$$

其中：
- $\mathbf{v}$ 是encoder output
- $W_1 \in \mathbb{R}^{d_{\text{hidden}} \times d}$, $W_2 \in \mathbb{R}^{d_{\text{word}} \times d_{\text{hidden}}}$
- $d_{\text{word}}$ 是LLM word embedding dimension（Vicuna-7b = 4096, Vicuna-13b = 5120）
- GELU: $\text{GELU}(x) = x \cdot \Phi(x)$, where $\Phi$ 是standard normal CDF

这个阶段：
- Encoder frozen
- LLM frozen
- Trainable: projection module + word embedding layer (因为新增了 `<tact_start>` 和 `<tact_end>` 两个token)
- 8k samples, AdamW, lr = $2 \times 10^{-5}$, batch = 16, cosine annealing

### 3.4 End-to-end Fine-tuning 阶段

**Stage 3目标**：让LLM能coherent地generate响应，并匹配language annotation style。

- Encoder frozen
- Trainable: projection module + LLM (via LoRA) + word embedding layer

#### LoRA 配置

LoRA的low-rank decomposition：

$$W = W_0 + \frac{\alpha}{r} B A$$

其中：
- $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 是frozen原始权重
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$ 是learnable
- $r = 128$ (rank)
- $\alpha = 256$ (scaling factor) → effective scaling $\alpha / r = 2$
- dropout = 0.05

直觉上，$r=128$ 是相对高的rank，意味着允许较大的adaptation capacity。这是因为tactile modality的representation shift比纯language task大很多。

#### Hyperparameters
- 3k samples
- AdamW, no weight decay, batch = 16, cosine annealing
- lr_projection = $2 \times 10^{-5}$
- lr_LoRA = $2 \times 10^{-4}$ (10x larger than projection，因为LLM需要更多adaptation)
- 5 hours for 7b, 6.5 hours for 13b (1-2 RTX A6000s)

---

## 四、5个Task详解

| Task | Type | Train? | Eval? | Random Baseline |
|---|---|---|---|---|
| OPD (Object Property Description) | 描述单个video | ✓ | ✓ | N/A (captioning) |
| PC (Property Comparison) | 二选一比较 | ✓ | ✓ | 33.33% |
| PSS (Property Superlative Selection) | 三选一最值 | ✓ | ✓ | 33.33% |
| POM (Property-Object Matching) | 三选一物体匹配 | ✓ | ✓ | 16.67% |
| PSR (Property Scenario Reasoning) | 场景推理 | ✗ | ✓ | 50.00% |

### 4.1 OPD (Object Property Description)

输入：一段tactile video序列 $T_1, \ldots, T_N$
输出：unstructured description (自由文本) + structured description ("Overall, it presents a {hardness} and {roughness} surface with {bumpiness}.")

unstructured description是用ChatGPT 3.5生成再人工清洗的——这点有点弱，因为ground truth本质是LLM-generated的，但作者说这样做能capture到更fine-grained的描述（比如toilet paper的"fibrous structure"，rice的"grains"）。

### 4.2 PC (Property Comparison)

输入：两段tactile video + 一个比较形容词（"bigger bumps than", "harder than"等）
模型先describe both，再给Yes/No结论。

这个task让模型 align "hardness = harder/softer" 这类comparative adjective 与 property category的ordering。

### 4.3 PSS (Property Superlative Selection)

输入：三段video + 一个最高级形容词（"smoothest", "hardest"）
模型describe each + 选出最匹配的。

注意NEWTON paper (https://arxiv.org/abs/2310.04298) 发现LLM在比较级polarity变化时性能不稳定，所以这个task专门针对这个鲁棒性。

### 4.4 POM (Property-Object Matching)

输入：三段video + 三个object name
模型需要根据haptic感觉 + LLM内嵌的object knowledge把video和object匹配。

这个task最考验模型把"tactile感受"和"object identity"联系起来的能力。

### 4.5 PSR (Property Scenario Reasoning) — **只evaluation不train**

输入：两段video + 一个real-world scenario question
例子（Table V）：
1. "Which object is most suitable for removing stains from a non-stick pan without scratching it?" → target: microfiber cloth (需要hardness + roughness都低)
2. "Which object would be most easily grippable when wet and slippery?" → target: rough + bumpy object

PSR是zero-shot generalization test——模型从没见过这种prompt format，必须靠在OPD/PC/PSS/POM上学的property grounding + LLM内嵌commonsense来组合出来。

---

## 五、实验结果深度分析

### 5.1 主结果（Table VI - 物理理解任务）

| Model | PC | PSS | POM |
|---|---|---|---|
| Random | 33.33 | 33.33 | 16.67 |
| OCTOPI-7b | 48.10 | 74.67 | 44.39 |
| OCTOPI-7b (no OPD) | 46.51 | 39.88 | 23.23 |
| OCTOPI-13b | 55.06 | 84.00 | 60.43 |
| OCTOPI-13b (no OPD) | 40.70 | 39.88 | 18.71 |

**Key insight**：去掉OPD（即让模型不先生成property description再回答，而是直接回答），性能大幅下降。比如PSS从74.67 → 39.88，POM从44.39 → 23.23。这说明**intermediate physical property prediction是核心**——它是把low-level tactile perception与high-level reasoning bridge起来的中间语言"platform"。这种"chain-of-thought但用property作为thought unit"的设计非常elegant。

**Scaling effect**：从7b → 13b在所有任务上都涨，POM涨最多（+16.04%）。直觉是POM需要更多object-level commonsense knowledge，13b的LLM比7b有更强的object knowledge prior。

### 5.2 场景推理结果（Table VII）

| Model | PSR |
|---|---|
| Random | 50.00 |
| OCTOPI-7b | 69.57 |
| OCTOPI-7b (w/o OPD) | 63.04 |
| OCTOPI-13b | 67.39 |
| OCTOPI-13b (w/o OPD) | 39.13 |

**Interesting finding**：7b在PSR上比13b略好（69.57 vs 67.39）。作者没给深入解释，但直觉猜测：13b可能overfitting到training prompt format上更严重，而PSR是zero-shot eval。这种7b outperforming 13b的case在LLM agent literature里其实也出现过，例如some tool-use benchmarks。

w/o OPD时13b掉到39.13，甚至低于random baseline 50.00，说明13b如果不被强制先生成property description，反而会被LLM prior误导（可能13b对自己的commonsense更自信，导致不依赖tactile evidence）。

### 5.3 Real Robot Avocado Ripeness Classification（Table VIII）

实验设置：
- 10个avocado，每个采20个tactile sample → 200 samples
- 100 pairs用于ripeness pairwise comparison
- Franka Emika Panda 7-DoF + 2个GelSight sensors
- 只用pressing，不用rotation

| Metric | Random | OCTOPI-13b | PG-InstructBLIP |
|---|---|---|---|
| Property Prediction (all 3 correct) | 3.70 | 35.50 | 0.00 |
| Hardness | 33.33 | 57.50 | 37.50 |
| Roughness | 33.33 | 71.00 | 3.00 |
| Bumpiness | 33.33 | 64.00 | 9.50 |
| Ripeness Classification | 50.00 | 63.00 | N/A |

**Key findings**：
1. OCTOPI-13b只用pressing就能达到不错的property prediction——说明representation对exploratory procedure是robust的
2. PG-InstructBLIP（vision-only baseline）在avocado property prediction上几乎random，因为视觉上avocado surface差异不显著
3. Ripeness classification 63% accuracy（pairwise comparison）——这个数字看起来不算特别高，但要注意是pairwise（random = 50%），且没有专门train过avocado

**Ripeness reasoning prompt**（这是PSR-style zero-shot的真正测试）：
```
"You will be given tactile descriptions that consist of three physical properties: 
hardness, roughness, bumpiness. ... Which of these properties help to determine 
avocado ripeness? Rank them."
```
OCTOPI-13b的回答："hardness and bumpiness. Roughness is not a reliable indicator... ripe avocado will be moderately hard with small bumps, unripe will be hard with no/small bumps."

这个回答展示了LLM的**commonsense能被property grounding激活**——它从language pretraining里知道ripe avocado soft，但只有在被prompt去consider properties时才调出这个知识。

### 5.4 OPD prediction accuracy（Table IX）

| Model | Combined | Hardness | Roughness | Bumpiness |
|---|---|---|---|---|
| Random | 3.70 | 33.33 | 33.33 | 33.33 |
| FT CLIP (encoder only) | 57.89 | 86.84 | 76.32 | 71.05 |
| OCTOPI-7b | 47.37 | 71.05 | 73.68 | 81.58 |
| OCTOPI-13b | 55.26 | 73.68 | 78.95 | 78.95 |

注意：FT CLIP是stage 1训练完的encoder + classification head，OCTOPI是end-to-end LVLM。

**Intuition**：OCTOPI的combined accuracy比pure CLIP classifier略低（55.26 vs 57.89），这是合理的——LVLM需要把representation压缩到language token空间，会有信息损失。但OCTOPI换来了language reasoning能力，这是纯classifier做不到的。

Hardness上OCTOPI明显比FT CLIP低（73.68 vs 86.84），可能因为hardness的"moderately hard"这个middle category在language space里最难表达——soft和hard有clear word anchor，moderately hard则需要更nuanced的compositional表达。

### 5.5 Ablation: Encoder fine-tuning（Table X, XI）

| Model | Combined | Hardness | Roughness | Bumpiness |
|---|---|---|---|---|
| OCTOPI-7b (FT CLIP) | 47.37 | 71.05 | 73.68 | 81.58 |
| OCTOPI-7b (base CLIP) | 39.47 | 81.58 | 52.63 | 55.26 |

Base CLIP在hardness上反而高（81.58 vs 71.05），但在roughness和bumpiness上大跌（52.63 vs 73.68，55.26 vs 81.58）。直觉：CLIP pretrain本来就对rigid object form factor敏感，hardness可以靠deformation edge detection做，base CLIP的edge detector已经够好；但roughness/bumpiness需要sensor-specific texture理解，必须fine-tune。

Table XI显示physical understanding task全面提升：
- PC: 48.10 → 30.38 (-17.72% w/o FT)
- PSS: 74.67 → 42.67 (-32% w/o FT)
- POM: 44.39 → 36.36 (-8.03% w/o FT)

PSS掉32%非常显著，说明superlative selection高度依赖encoder quality——base CLIP的embedding在roughness/bumpiness上区分不开，模型就只能猜。

### 5.6 Ablation: End-to-end fine-tuning with LoRA（Table XII, XIII）

| Model | Combined | Hardness | Roughness | Bumpiness |
|---|---|---|---|---|
| OCTOPI-7b (w/ LoRA) | 47.37 | 71.05 | 73.68 | 81.58 |
| OCTOPI-7b (w/o LoRA) | 39.47 | 65.79 | 76.32 | 71.05 |
| OCTOPI-13b (w/ LoRA) | 55.26 | 73.68 | 78.95 | 78.95 |
| OCTOPI-13b (w/o LoRA) | 23.68 | 36.84 | 73.68 | 71.05 |

OCTOPI-13b w/o LoRA combined accuracy只有23.68%——比random 3.70%高，但比7b w/o LoRA的39.47%低。直觉：13b LLM在没有LoRA adaptation时，会强烈pull representation toward its language prior，反而overpower了projection module提供的tactile evidence。7b的LLM prior较弱，所以即使没有LoRA，tactile projection还能占主导。

Table XIII显示PSS从84.00 → 77.33（w/o LoRA只掉7%），但POM从60.43 → 34.76（掉25.67%）。POM需要object-name匹配，这强依赖LLM的object knowledge被aligned到tactile representation，所以LoRA帮助巨大。

---

## 六、Appendix细节 & 实用insights

### 6.1 Annotation guidelines (Table A1)

具体rating：
- Hardness: Soft[0], Moderately hard[1], Hard[2]
- Roughness: Smooth[0], Slightly rough[1], Rough[2]
- Bumpiness: No bumps[0], Small bumps[1], Big bumps[2]

Hardness & Roughness用真实haptic judgment，Bumpiness用visual judgment from GelSight image（bumps size relative to tactile image area）。这种mixed modality annotation是个潜在confound。

### 6.2 Property distribution (Table A7, A8)

| Property | Class 0 | Class 1 | Class 2 |
|---|---|---|---|
| Hardness | 35.14% | 18.92% | 45.95% |
| Roughness | 48.65% | 21.62% | 29.73% |
| Bumpiness | 35.14% | 45.95% | 18.92% |

Moderately hard / Big bumps都是18.92%，是最minority class。Table A8显示joint distribution严重不均：[Hard, smooth, no bumps]有17个（典型rigid human-made object），而[Soft, slightly rough, no bumps], [Soft, rough, no bumps], [Moderately hard, slightly rough, no bumps]等都是0个。

这是dataset的局限——74个objects不够覆盖3^3=27个combinations，curse of dimensionality迅速显现。作者在future work里提到需要更好的representation learning来处理imbalance。

### 6.3 Sample video statistics (Table A9)

平均112.30 frames per video, min=50, max=126。但实际只sample 5 frames——通过top 30% pixel intensity difference筛选salient frames。

**Frame selection algorithm**:
1. 计算每帧与前帧的total pixel intensity difference
2. 选top 30%作为salient frames
3. Training时随机sample 5个；evaluation时uniform interval取5个

这个idea很合理：GelSight video里大部分information集中在contact开始/结束的deformation transient，middle段往往是steady state。

### 6.4 Encoder Analysis (Appendix E, Fig 6-7)

Confusion matrices显示：
- Hardness: Hard class识别最好，Moderately hard最差（经常被分到Hard）
- Roughness: Smooth和Slightly rough好，Rough差（与Smooth混淆）
- Bumpiness: No bumps和Small bumps好，Big bumps差（与No bumps混淆）

这种error pattern非常符合直觉——middle category总是最难，因为它是continuous spectrum的boundary。Appendix E的t-SNE/UMAP visualization也证实Moderately hard的embedding和Hard/Soft都重叠。

---

## 七、Critical Thoughts & Open Questions

### 7.1 强点

1. **Tactile-LVLM 是真的 underexplored**——这是第一批把GelSight接到LVLM的工作。同期有Letian Fu等人的Touch-Vision-Language dataset (https://arxiv.org/abs/2402.13232) 和Binding Touch to Everything (https://arxiv.org/abs/2401.18084)，但PHYSICLEAR的property labels + reasoning suite是unique的。

2. **Three-stage training**设计非常合理：
   - Stage 1 学 perceptual representation
   - Stage 2 学 cross-modal alignment
   - Stage 3 学 language coherence
   这种curriculum类似BLIP-2的optimal bootstrapping (https://arxiv.org/abs/2301.12597)。

3. **PSR task作为zero-shot reasoning eval** 是关键创新——它把physical reasoning test从"模仿training prompt"提升到"组合training学到的能力"。

### 7.2 弱点 & 可改进

1. **Dataset size 太小**：74 objects, 408 videos, 1200+ annotations——比ObjectFolder Real (https://arxiv.org/abs/2306.00956)等小一个数量级。这导致7个test objects的evaluation variance很大（PSS的84%在7个test sample上其实是约6/7 vs 7/7的区别）。

2. **Property set 太窄**：只有3个property × 3个category = 27个combinations，且实际只有15个combinations出现。没有temperature, slipperiness, elasticity, weight等关键physical property。GelSight其实也能感temperature（通过thermal conductance），paper没利用。

3. **OPD ground truth是ChatGPT-generated**：这是个weak supervision问题。如果ChatGPT生成的unstructured description有bias，模型会被带偏。

4. **Avocado ripeness 63% 不算很高**：在pairwise comparison上比random高13%，且只测了10个avocado（100 pairs）。如果用于实际robotics deployment，这个accuracy还远远不够。

5. **No comparison with pure LLM + text description of tactile**：可以加个baseline——把GelSight image送给GPT-4V让它直接describe，再让LLM reason。这能分离出"我们的encoder学到了什么LLM-vision没有的"。

6. **No test of generalization to new GelSight sensor**：如果换了GelSight Mini vs GelSight Wedge，encoder还能work吗？这是real deployment必须考虑的。

### 7.3 与相关工作的context

- **Physically Grounded VLMs (PG-InstructBLIP)** (https://arxiv.org/abs/2309.02561)：vision-only property prediction，paper里用了作为avocado baseline，惨败。
- **ObjectFolder Real** (https://arxiv.org/abs/2306.00956)：multisensory但没property label，只有reconstruction。
- **Touch and Go** (https://arxiv.org/abs/2211.12498)：vision-touch pairing但没reasoning benchmark。
- **Multiply (Hong et al. 2024)** (https://arxiv.org/abs/2401.08577)：simulated tactile + LLM，但sim-to-real gap未解决。
- **NEWTON** (https://arxiv.org/abs/2310.04298)：physical reasoning benchmark for LLMs，no tactile modality。
- **CLEVRER** (https://arxiv.org/abs/1910.01442)：physical reasoning from video，纯visual。
- **Meta-Transformer** (https://arxiv.org/abs/2307.10802)：unified multimodal learning framework，但没specifically handle tactile reasoning。
- **Video-ChatGPT** (https://arxiv.org/abs/2306.05424)：video understanding LVLM，OCTOPI的video-as-frames处理借鉴了它。
- **GelSight hardness estimation (Yuan et al. 2016)** (https://arxiv.org/abs/1704.03822)：早期工作，OCTOPI的hardness label选择参考了它。

### 7.4 Future directions（作者提到+我猜测）

1. **More exploratory procedures**: sliding, tapping, thermal contact（different motion → different property sensitivity）
2. **Combine with proprioception**: robot joint torque + tactile → richer physical estimate
3. **Cross-dataset training**: combine with ObjectFolder Real, Touch and Go → bigger pretraining
4. **Tactile encoder architecture**: 当前用CLIP ViT-L/14 + VPT，可以尝试MAE-style self-supervised (Tactile-MAE, https://arxiv.org/abs/2307.07358) 或vision transformer specifically pretrained on tactile
5. **Property compositionality**: 用structured property → 复合physical reasoning (e.g., "soft AND rough AND no bumps" → "stress ball")
6. **Active perception**: robot chooses where to touch based on current uncertainty about property
7. **LLM-driven exploratory procedure selection**: "I want to know if this avocado is ripe, what motion should I do?" — 这个其实是最exciting的方向，把language planning和tactile sensing闭环
8. **Tactile-conditioned manipulation policy**: 把property prediction作为input to a manipulation policy（类似RT-2 (https://arxiv.org/abs/2307.15818) but with tactile）

---

## 八、Reproduction 关键细节

如果你想reproduce或build on这个工作：

1. **Hardware**: GelSight Mini (sensor) + Franka Emika Panda + 1-2 RTX A6000 (48GB each)
2. **Software stack**: 
   - CLIP ViT-L/14 (https://github.com/openai/CLIP)
   - Vicuna v1.5 (https://github.com/lm-sys/FastChat)
   - LoRA via PEFT (https://github.com/huggingface/peft)
   - ViFi-CLIP (https://github.com/muzairkhan/ViFi-CLIP)
3. **Critical hyperparameters not to mess with**:
   - 5 frames per video
   - Top 30% pixel intensity difference threshold
   - LoRA rank 128, alpha 256 (ratio 2)
   - Stage 1: lr 1e-3 (high!), 30 epochs
   - Stage 2/3: lr 2e-5 to 2e-4
4. **Dataset access**: https://github.com/clear-nus/octopi

---

## 九、最核心的Intuition总结

如果你想用一句话grasp住Octopi的essence：

> **"Train a CLIP encoder to recognize hardness/roughness/bumpiness from GelSight frames → project those embeddings into Vicuna's word space → use language reasoning (with OPD as intermediate chain-of-thought) to bridge tactile perception and commonsense scenarios."**

整个pipeline的核心是**"language as the bridge between tactile perception and physical reasoning"**——触觉信号先被翻译成language-friendly的property description，再让LLM用它的commonsense去组合这些description做出scenario reasoning。

这种"perception → language-grounded property → reasoning"的三段式架构其实是一个通用template，可以推广到其他modality (e.g., proprioception, audio, thermal sensing) ——只要你能找到一组离散property category作为anchor，并且这些category在LLM的language prior里有rich commonsense association。

这是这篇paper给build embodied AI intuition的最大贡献：**物理property可以作为modality-specific perception和modality-agnostic reasoning之间的universal interface**。
