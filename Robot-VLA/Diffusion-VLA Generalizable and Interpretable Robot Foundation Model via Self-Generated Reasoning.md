---
source_pdf: Diffusion-VLA Generalizable and Interpretable Robot Foundation Model via
  Self-Generated Reasoning.pdf
paper_sha256: f036c44e762cd5fcdc0a32dd12f9294e390dafd7eb3233f839717c587ed0129d
processed_at: '2026-08-03T21:52:39-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 DiVLA

## 一句话版本

让大语言模型当"脑子"负责想，让diffusion model当"手"负责做，中间用一根管子（FiLM）把脑子的想法实时灌给手。

---

## 为什么要搞这个

现在robot foundation model 两条路，各有各的病：

**路线一：把action当文字来预测**（OpenVLA, RT-2这条路）

就好比让LLM去预测下一个字"是"还是"不是"，只不过让它预测下一个action是"往左0.3米"还是"往右0.5米"。把连续的action硬塞进离散的token里，精度会丢。而且一个token一个token往外吐，慢得要命，OpenVLA 7B只能跑5Hz——你眨一下眼它还没反应过来。

**路线二：纯diffusion policy**（Diffusion Policy这条路）

从噪声里一步步去噪，还原出action。好处是快，能建模多模态（同一场景下可能有好几种合理action），精度好。坏处是——它就是个肌肉记忆机器，没有"思考"。你给它一个复杂任务"先把杯子翻过来再放到盘子上"，它没法分解，只能端到端学整个trajectory，遇到没见过的东西就傻眼。

**DiVLA的想法**：这俩各干各擅长的。LLM负责reasoning（这是LLM的看家本领），diffusion负责action生成（这是它的强项），然后用FiLM把它们粘起来。

---

## 架构怎么搭的

想象一个流水线：

```
摄像头拍到的画面
    ↓
SigLIP 把图编码成visual features（每个camera view都过一遍，拼起来）
    ↓
Qwen2-VL 这个大脑接住visual features + 你的语言指令
    ↓
大脑分两路输出：
    路径A：输出reasoning text（"我要去抓那个toy car"）—— 用NTP loss训练
    路径B：输出action tokens（一组latent向量）—— 拿去给diffusion
    ↓
路径B经过一个MLP投影，对齐维度
    ↓
同时，路径A的reasoning embedding 通过 FiLM 注入 diffusion policy 的每一层
    ↓
Diffusion Policy 从噪声去噪出最终的joint space action
```

关键就两件事：

**1. Reasoning不是写完就完了，要"灌"进去**

老办法：LLM先吐一段reasoning text，再把这段text塞回input跑第二遍forward——慢，而且reasoning和action是两段分离的process。

DiVLA：reasoning tokens的hidden state直接通过FiLM调制的diffusion policy。FiLM说白了就是每一层都做：
```
feature = γ × feature + β
```
其中γ和β是从reasoning embedding算出来的。相当于reasoning在diffusion每一层都"提醒"一下网络："你现在的目标是在抓toy car，别忘了"。

这就是为什么ablation去掉reasoning injection后，多步任务（比如先开盖子再放cube）性能从90%崩到27%——没有持续的reasoning reminder，网络做着做着就忘了自己在干嘛。

**2. 换robot不用重新训整个action decoder**

不同robot的action维度不同（单臂7维，双臂14维）。Octo的做法是每个robot搞一个独立的action decoder，pre-trained知识全丢了。

DiVLA只换action decoder最底层的一个MLP head，其余部分保留。pre-trained的diffusion知识还在，只需小量数据快速适应新embodiment。

---

## 实验里几个让人眼前一亮的点

**1. 用1/25的数据打败对手**

Table 1里，DiVLA-2B只用39K trajectory就大幅超过用970K pretrain的OpenVLA和Octo。为什么？因为Qwen2-VL在pretrain时已经见过海量图文对，它知道什么是"cup"什么是"plate"什么是"toy car"。Robot数据只需要教它"怎么act on这些我已经认识的concept"。Pre-trained VLM是个巨大的视觉语义先验，DiVLA把这个先验用到了极致。

**2. 零样本抓102个没见过的东西**

Bin picking任务，102个训练时完全没见过的物体，DiVLA拿到63.7%，OpenVLA只有28.4%。Diffusion Policy只有8.9%（基本瞎抓）。

为什么DiVLA这么强？因为reasoning在做类比——它看到screwdriver没见过，但reasoning告诉自己"这玩意长得像hex key"，就归到hex key那类去抓了。纯diffusion policy没有这层语义类比，只能靠像素级别的pattern matching。

**3. 双臂任务OpenVLA直接挂零**

Table 2的table bussing任务（双臂AgileX），OpenVLA成功率0%。因为双臂action维度翻倍，NTP要把14维连续action tokenize成离散token序列，这个离散化在高维下信息损失太严重。

DiVLA的diffusion head天然处理高维continuous action，没问题。这其实印证了paper的核心论点：NTP不适合做action generation，它只适合做reasoning。

**4. 自我纠错的demo**

Figure 6那个实验特别有意思：模型打算抓toy car，reasoning输出"grabbing the toy car"，人突然把hex key塞进gripper，reasoning瞬间切换成"grabbing the hex key"，然后正确地把hex key分到对应区域。

这说明reasoning embedding在做实时的visual grounding——它不是一次性生成指令就走人，而是持续根据当前observation更新自己的内部状态。FiLM让这个动态更新直接传递给action生成。

**5. 8倍速度提升**

同样7B参数：
- OpenVLA: 5Hz
- DiVLA-7B: 42Hz

原因：autoregressive的action生成是一个token一个token串行吐，diffusion是并行去噪多步。Diffusion天生在action生成上比NTP快。

**6. 72B的scaling law成立**

| Size | Sorting | Bin Picking |
|---|---|---|
| 2B | 66.2% | 63.7% |
| 7B | 74.9% | 66.7% |
| 72B | 82.4% | 75.9% |

模型越大效果越好，这和LLM的scaling law一致。说明VLA也会从scale中受益，这个方向有继续投入的价值。

---

## 让你建立intuition的几个角度

### Intuition 1: System 1 和 System 2

Kahneman的dual process theory：人脑有System 1（快、自动、模式匹配）和System 2（慢、deliberate、语言推理）。

DiVLA正好是这个结构：
- Diffusion head = System 1：快速生成action，基于pattern
- VLM reasoning = System 2：慢思考，分解任务，用语言描述当前subgoal

FiLM就是prefrontal cortex对motor cortex的top-down modulation——脑子持续告诉手"我们现在在做什么"。

这比纯System 1（纯diffusion policy）或纯System 2（纯NTP VLA）都强，因为人脑也是两个系统协同。

参考：Kahneman的《Thinking, Fast and Slow》

### Intuition 2: 为什么FiLM比concat强

如果你只是把reasoning embedding和visual feature concat起来作为input，这个reasoning信号只在网络第一层出现。深层网络可能"忘记"它——想想resnet50，input信息经过50层residual block早被稀释了。

FiLM在每一层都重新注入 γ 和 β，相当于沿depth方向给reasoning一个持续的"skip connection"。每一层都在说"别忘了我们在抓toy car"。

这就是为什么ablation掉reasoning injection后，long-horizon task（Task 5，开盖子放cube）崩得最惨——这种多步任务最容易"做着做着忘了目标"，没有持续reminder就失败。

### Intuition 3: Pre-trained VLM是个隐式的world model

Qwen2-VL在互联网海量图文对上pretrain过，它已经知道：
- 什么是cup，什么是plate
- 红色是什么意思
- "on top of"是什么空间关系
- 一个screwdriver和hex key长得像

DiVLA把这部分知识"继承"过来。Robot data只教它"怎么act"，不教"什么是cup"。这就是为什么39K trajectory就能打败970K pretrain的OpenVLA——OpenVLA的pretrain数据也是robot数据，没有互联网级别的concept knowledge。

VLM作为frozen prior，是data efficiency的真正来源。

参考：LLaVA https://llava-vl.github.io/

### Intuition 4: Reasoning是task decomposition

Ablation Table 8显示，去掉reasoning injection后：
- Task 1（object selection，单步）：100 → 66.7（降33%）
- Task 5（开盖子放cube，多步）：90.9 → 27.3（降70%）

多步任务崩得更狠。为什么？因为reasoning的本质是task decomposition——把"开盖子然后放cube"分解成"先开盖子"和"再放cube"两个subgoal，每个subgoal对应一段action。

没有reasoning，网络要端到端学整个长trajectory，难度指数级上升。有reasoning，每个subgoal变成一个短trajectory学习问题，简单多了。

这就像你教小孩"先刷牙再洗脸再吃饭"，分解着教比一锅端教容易。

---

## 我对这篇paper的critique

**1. GPT-4o生成的reasoning数据靠谱吗**

Droid数据集只有action和observation，没有reasoning text。DiVLA用GPT-4o自动给每条trajectory配reasoning。但GPT-4o看的只是language instruction加几帧图像，它并不真正理解trajectory的subgoal结构。

生成的reasoning质量未必高，可能只是paraphrase了instruction。这是weak supervision，paper没validate reasoning的实际质量。

参考：Droid dataset https://droid-dataset.github.io/

**2. Reasoning到底是不是causal**

FiLM用的是reasoning tokens的final embedding，这个embedding学到的可能不只是text semantic，还包括某种latent code。你怎么知道是reasoning text的内容在起作用，还是只是个额外的learnable feature？

可以做permutation ablation：把reasoning text打乱（保持token但乱序），看性能是否下降。如果下降不明显，说明text内容不重要，FiLM只是个额外的capacity。paper没做这个实验。

**3. Action tokens的来源不清楚**

paper说"VLM最后一层生成固定数量action tokens"，但没说这些tokens是learned query（类似perceiver resampler的learnable query）还是VLM autoregressive输出（依赖前文）。如果是learned query，那action和reasoning是parallel output；如果是autoregressive，那action依赖reasoning先生成完。

这个engineering detail对理解data flow很重要，paper应该交代清楚。

参考：Perceiver Resampler https://arxiv.org/abs/2103.03206

**4. 72B的scale实验有混淆变量**

72B用OXE + Droid联合pretrain，2B和7B只用Droid。所以72B性能好可能来自两个因素：(a) 模型大，(b) 数据多。没控制变量，scale law的结论不够干净。

**5. Diffusion的推理step数没说**

82Hz这个数字很漂亮，但diffusion policy推理时去噪几步？如果是1步（consistency model风格），那82Hz很合理；如果是50步DDPM，82Hz就有点离谱了。paper缺这个关键engineering detail。

参考：Consistency Models https://arxiv.org/abs/2303.01469

---

## 联想到的方向

**1. 和π₀对比**

π₀也是VLM + action expert架构，用flow matching代替diffusion。DiVLA用reasoning injection把reasoning灌进action head，π₀没有显式reasoning——它的"reasoning"隐含在VLM的hidden state里。

DiVLA可以看作"explicit reasoning版的π₀"，π₀可以看作"implicit reasoning版的DiVLA"。两条路可以互相借鉴：DiVLA可以试flow matching加速，π₀可以加reasoning injection增强interpretability。

参考：π₀ https://www.physicalintelligence.company/blog/pi0

**2. 和CogACT对比**

CogACT（Li et al., 2024a）也是VLM + action head思路，也用了类似的设计。DiVLA的差异化在于reasoning injection module——CogACT没这个。这俩可以一起读，看同一思路的不同instantiation。

参考：CogACT https://arxiv.org/abs/2411.19650

**3. 和FAST对比**

FAST（Pertsch et al., 2025）走的是另一条路——改进action tokenization让NTP-style VLA更高效。用compression tokenizer把action压成更少的token。

FAST和DiVLA是正交的思路：FAST说"NTP也能做好，只要tokenization好"，DiVLA说"NTP别做action，让diffusion做"。可以想象一个hybrid：FAST的tokenizer + DiVLA的reasoning injection。

参考：FAST https://arxiv.org/abs/2501.09747

**4. 3D Diffusion Policy的启示**

DiVLA纯用2D visual feature，没用3D representation。3D Diffusion Policy系列（Ke et al. 2024, Ze et al. 2024）证明3D prior对manipulation很有帮助。DiVLA的FiLM injection可以很自然地扩展到3D diffusion policy上——把reasoning灌进3D denoising network。

参考：3D Diffuser Actor https://arxiv.org/abs/2402.10885

**5. 和Show-O / Transfusion的深层联系**

Show-O和Transfusion是统一understanding和generation的工作——用一个transformer既做NTP又做diffusion。DiVLA把这种"unifying NTP和diffusion"的思想从image generation迁移到action generation。

从更高视角看：未来可能有一个unified transformer，既做language reasoning（NTP），又做image generation（diffusion），还做action generation（diffusion），完全unified multimodal agent。

参考：Transfusion https://arxiv.org/abs/2408.11039
参考：Show-O https://arxiv.org/abs/2408.12528

**6. Reasoning数据可以更好**

DiVLA用GPT-4o生成reasoning是偷懒。更好的做法：
- 用VLM对trajectory做hindsight reasoning：先看完整trajectory，再生成"我刚才在做什么"的subgoal description
- 用人类标注的少量high-quality reasoning data做distillation
- 用RL让reasoning和action consistency对齐

参考：Hindsight Experience Replay https://arxiv.org/abs/1707.01495

---

## 总结

DiVLA的核心贡献：把robot foundation model从"VLM + tokenized action"的单一范式解放出来，证明**structured reasoning + diffusion action head + FiLM injection**这个组合work得很好。Conceptually clean（System 1/2分工），engineering简洁（FiLM一行公式），实验全面（multi-task、sorting、bin-picking、bimanual、scaling都覆盖了）。

对foundation model设计者的take-away：future VLA应该认真考虑reasoning-conditioned action generation这个范式。NTP擅长reasoning，diffusion擅长action，让它们各司其职比硬让一个范式做所有事强。

这paper给我最大的启发是：**hybrid architecture里两个组件怎么couple比各组件本身更重要**。FiLM这种"持续注入"的couple方式比"串行forward两次"优雅太多。这个insight可以迁移到很多其他hybrid model设计上。

参考汇总：
- DiVLA原文 ICML 2025: https://proceedings.mlr.press/v267/
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- SigLIP: https://arxiv.org/abs/2303.15343
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- FiLM原始paper: https://arxiv.org/abs/1709.07871
- OpenVLA: https://openvla.github.io/
- RT-2: https://robotics-transformer2.github.io/
- ECoT: https://embodiedcot.github.io/

---

# Diffusion-VLA (DiVLA) 深度解析

## 1. 核心Motivation与定位

这篇paper的核心洞察来自一个二分法：

- **Autoregressive VLA**（RT-2、OpenVLA、ECoT）: 把action tokenize成discrete token做next-token prediction。问题：(1) continuous action离散化损失precision；(2) autoregressive序列生成慢，实时性差。
- **Diffusion Policy**: 在action空间做denoising，能建模multimodal action distribution，速度快。问题：缺reasoning能力。

DiVLA的论点是：**让NTP做它擅长的reasoning，让diffusion做它擅长的action生成**，通过reasoning injection module把它们couple起来。这个动机很clean——分工合作，各取所长。

参考链接：
- OpenVLA: https://openvla.github.io/
- RT-2: https://robotics-transformer2.github.io/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- ECoT: https://embodiedcot.github.io/
- π₀: https://www.physicalintelligence.company/blog/pi0

## 2. Architecture详解

### 2.1 数据流

```
Image (multi-view) 
   ↓ SigLIP encoder (per-view shared weights)
Dense visual features
   ↓ Transformer projector
N visual embeddings (concatenated across views)
   ↓
Qwen2-VL (frozen visual encoder + LoRA fine-tuned LLM)
   ├── Output 1: text reasoning tokens (autoregressive, NTP loss)
   └── Output 2: action tokens (final embedding layer)
        ↓ Projection MLP (2 layers + LayerNorm)
        Action conditioning + reasoning embedding
        ↓ FiLM injection into Diffusion Policy
        Denoised action sequence (joint space)
```

### 2.2 关键设计点

**(a) Visual encoding**: 用SigLIP（Zhai et al., 2023）做vision encoder，每个camera view共享backbone然后concatenate visual tokens。这点比OpenVLA原始单view设计强很多——Table 7显示OpenVLA从3-view的45.3%降到1-view的12.7%。

**(b) VLM backbone**: 选Qwen2-VL，有2B/8B/72B三个size，可以做scaling law实验。Visual encoder和VLM都frozen，只对VLM做LoRA fine-tune（learning rate 2e-5）。

**(c) Action token projection**: VLM最后一层embedding出来固定数量的action tokens，经过2层MLP+LayerNorm的projector桥接到diffusion model。这个projector类似LLaVA的vision-to-LLM projector，只是这里方向反了（LLM→diffusion）。

**(d) Action decoder底部MLP**: 预测joint space。**多embodiment时只需新初始化一个MLP层**——这比Octo复制整个action decoder优雅得多，因为pre-trained knowledge不会丢失。

### 2.3 Reasoning Injection Module (核心创新)

这是paper的灵魂。关键问题是：怎么把VLM生成的reasoning真正"喂"给diffusion policy，而不只是输入端拼一下？

**Naive做法**：autoregressive先生成reasoning text → 把reasoning作为新input再forward一次 → 输出action。这是recursive setup，两次forward，慢。

**DiVLA做法**：reasoning tokens的final embedding直接通过FiLM注入diffusion policy的layers。

#### FiLM公式回顾

原始FiLM (Perez et al., 2018)：
$$\text{FiLM}(x_i) = \gamma_i(c) \odot x_i + \beta_i(c)$$

变量含义：
- $x_i \in \mathbb{R}^d$: 第 $i$ 层的feature vector
- $c$: conditioning signal（这里是reasoning embedding）
- $\gamma_i(c), \beta_i(c) \in \mathbb{R}^d$: 通过MLP从 $c$ 学到的第 $i$ 层的scale和shift参数
- $\odot$: 逐元素乘法

直觉：reasoning embedding $c$ 不是简单concat到input上，而是逐层调制diffusion network中间特征的"增益和偏置"。这相当于reasoning信号沿网络深度持续施加影响，而不是只在input处出现一次。

#### 为什么叫"injection"

paper明确说："policy network focuses primarily on action-specific tokens, while the reasoning module functions as an auxiliary enhancement"——reasoning不主导决策流，只是提供contextual depth。这个design choice很重要：避免reasoning text"劫持"action generation，但又要让它真正影响。

参考FiLM原始paper: https://arxiv.org/abs/1709.07871

## 3. Training Objective

$$L = L_{diff} + \alpha \cdot L_{ntp}$$

- $L_{diff}$: diffusion denoising loss
- $L_{ntp}$: next-token prediction loss (cross-entropy over reasoning tokens)
- $\alpha$: 平衡超参数，经验设 $\alpha = 10$

**为什么α=10**: paper观察 $L_{ntp}$ 的magnitude比 $L_{diff}$ 小约10倍。如果不加权，NTP loss会被diffusion loss淹没，reasoning能力学不出来。这暗示两个loss的gradient scale不平衡是训练这种hybrid model的关键engineering detail。

#### Diffusion loss的具体形式

标准DDPM目标（参考Ho et al., 2020）：
$$L_{diff} = \mathbb{E}_{t \sim \mathcal{U}(0,T),\, x_0 \sim q,\, \epsilon \sim \mathcal{N}(0, I)} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c_{reason}) \|^2 \right]$$

变量：
- $t$: diffusion timestep，均匀采样于 $[0, T]$
- $x_0$: ground-truth action sequence
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$: 加噪后的action
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$: 累积noise schedule
- $\epsilon$: 标准高斯噪声
- $\epsilon_\theta$: 神经网络预测的noise
- $c_{reason}$: reasoning embedding（通过FiLM注入）

关键：$c_{reason}$ 是conditioning signal，让diffusion不是随机生成action，而是"reasoning-conditioned"的action。

## 4. 实验数据深度解读

### 4.1 Multi-task Learning (Table 1)

| Model | Pre-train | In-Dist Avg | Visual Gen Avg |
|---|---|---|---|
| Diffusion Policy | - | 27.9 | 8.9 |
| TinyVLA | - | 45.5 | 28.9 |
| Octo | 970K | 24.3 | 17.8 |
| OpenVLA-7B | 970K | 39.4 | 26.7 |
| DiVLA-2B | **39K** | **83.6** | **57.8** |

惊人点：DiVLA-2B只用39K trajectory（Octo/OpenVLA的1/25）就大幅超越。这是data-efficiency的强证据，暗示reasoning prior弥补了数据量不足。

### 4.2 Factory Sorting (Table 4)

四个类别分拣：toy cars、knit gloves、stuffed toys、hex keys。Cluttered Mixed（最难场景）：
- DP: 6/65 = 9.2%
- OpenVLA: 22/65 = 33.8%
- DiVLA-2B: 40/65 = 61.5%

Diffusion Policy在clutter场景从66.7%崩到9.2%，说明它visually generalize弱。DiVLA保持60%+，说明reasoning提供了视觉鲁棒性。

### 4.3 Zero-shot Bin Picking (Figure 4)

102个完全unseen objects，各种尺寸/纹理/形变：
- Diffusion Policy: 8.9%
- TinyVLA: 23.5%
- OpenVLA: 28.4%
- **DiVLA-2B: 63.7%**

这是instance-level generalization的硬指标，DiVLA把SOTA翻倍以上。

### 4.4 Bimanual Table Bussing (Table 2)

| | Seen | Mixed |
|---|---|---|
| Diffusion Policy | 45.8 | 31.2 |
| OpenVLA | 0 | 0 |
| DiVLA-2B | 72.9 | 70.8 |

OpenVLA直接挂零——可能因为bimanual action dimension翻倍，NTP-style的token化对高维action不友好。DiVLA的diffusion head天然处理高维continuous action，所以没问题。这印证了paper的核心论点。

### 4.5 推理速度 (Table 5)

- DiVLA-2B: 82Hz (A6000)
- DiVLA-7B: 42Hz
- OpenVLA-7B: 5Hz

42Hz vs 5Hz，同尺寸8倍加速。原因：autoregressive逐token生成慢，diffusion可并行去噪多步。

注意：vLLM加速没那么显著（不用vLLM时2B 74Hz, 7B 30Hz），但已经远超OpenVLA。

### 4.6 Scaling Law (Table 10)

| Task | 2B | 7B | 72B |
|---|---|---|---|
| Sorting | 66.2 | 74.9 | 82.4 |
| Bin Picking | 63.7 | 66.7 | 75.9 |

72B在sorting上82.4%，zero-shot bin picking 75.9%。验证scaling law在VLA中成立。72B用了OXE + Droid联合pretrain（数据更多）。

### 4.7 Ablation: Reasoning Injection (Table 8)

去掉reasoning injection：
- Task 1 (Object Selection): 100 → 66.7
- Task 5 (Closed Lid Box): 90.9 → 27.3
- Avg: 83.6 → 50.3

Task 5崩塌最严重（multi-step操作），说明reasoning对long-horizon task最关键——它做了task decomposition。这印证了"reasoning让长任务被分解为子任务，简化learning"的假说。

## 5. Behavior Analysis亮点

### 5.1 类比泛化

未训练过的screwdriver被分类为"hex key"（视觉相似），绿色手套被识别为"green glove"，玩具猫被识别为"brown toy cat"。模型通过semantic feature类比而非直接识别来泛化。

### 5.2 Self-correction

最有趣的实验：模型打算抓toy car，reasoning输出"grabbing the toy car"，人为干预把hex key塞进gripper，reasoning瞬间切换为"grabbing the hex key"，并正确分拣。

这暗示reasoning embedding在做"实时状态估计"——它不只是一次性输出指令，而是持续地visual-grounding当前状态。这让action generation robust to perturbation。

## 6. 我的Intuition与批评

**Intuition 1: 双系统分工对应System 1/System 2**
DiVLA本质是Kahneman双系统理论的instance：
- Diffusion head = System 1（fast, automatic, pattern-matching）
- VLM reasoning = System 2（slow, deliberate, language-based）

reasoning injection让System 2的输出modulate System 1的执行，类似人脑prefrontal cortex对motor cortex的top-down control。

**Intuition 2: 为什么FiLM比concat强**
简单concat reasoning embedding到input只在第一层影响，深层网络可能"忘记"它。FiLM在每个layer都重新注入scaling/shifting，相当于沿depth轴的"skip connection for reasoning"。这解释了为什么ablation掉它性能崩塌——policy head缺了持续的reasoning guidance。

**Intuition 3: pre-trained VLM作为frozen prior**
DiVLA只LoRA fine-tune VLM，visual encoder完全冻结。这意味着所有visual concept knowledge（如识别"toy car"）来自pre-trained VLM，robot data只训练"如何act on这些concepts"。这是data efficiency的真正来源——76M的互联网图像-文本对（Qwen2-VL pretrain）隐式提供了object understanding。

**批评点**：
1. **GPT-4o生成reasoning data**: 用GPT-4o自动给Droid trajectory配reasoning text，这是weak supervision。GPT-4o看的只是language instruction + 可能的frame，不真正理解trajectory的subgoal结构。生成的reasoning质量未必高。
2. **Reasoning的"真假"**: paper没有验证生成的reasoning text真的causally影响action。FiLM用的是final embedding，可能text的semantic内容不重要，只是某种latent code。可以做permutation ablation打乱reasoning text看是否性能下降。
3. **72B实验细节缺失**: 72B在OXE+Droid上pretrain，数据更多，所以性能提升未必纯来自scale。需要控制变量。
4. **Action token的来源模糊**: paper说"VLM最后一层生成固定数量action tokens"，但没说这些tokens是learned query还是output sequence。如果是learned query，类似perceiver resampler；如果是output，则依赖VLM autoregressive生成。这里engineering detail缺失。
5. **Diffusion Policy的noise schedule和step数没给**: 推理时几步去噪？影响82Hz数字的实际意义。

## 7. 联想到的相关工作

1. **CogACT** (Li et al., 2024a): 也是VLM + action head，但用Flow Matching。DiVLA和CogACT是同一思路的不同instance。https://arxiv.org/abs/2411.19650

2. **FAST** (Pertsch et al., 2025): 用compression tokenizer让NTP-style VLA更高效，走的是另一条路——改进tokenization而非换action head。https://arxiv.org/abs/2501.09747

3. **YAY** (Shi et al., 2024): language correction通过FiLM-style injection，DiVLA的FiLM灵感来源之一。https://yell-at-your-robot.github.io/

4. **Show-O / Transfusion / Vila-U**: 统一understanding和generation的multimodal model，DiVLA把这种unifying思想从image generation移植到action generation。https://arxiv.org/abs/2408.12528

5. **π₀**: 同期工作，用flow matching代替diffusion，同样大VLM + action expert架构。值得对比——DiVLA的reasoning injection vs π₀的action expert design。https://arxiv.org/abs/2410.24164

6. **RDT-1B**: bimanual的diffusion foundation model。https://arxiv.org/abs/2410.07864

7. **3D Diffusion Policy / Equibot**: 3D representation + diffusion。DiVLA纯2D，可借鉴3D prior。https://arxiv.org/abs/2403.03954

## 8. 总结

DiVLA的真正贡献是把VLA领域从"VLM+tokenized action"的单一范式解放出来，证明**structured reasoning + diffusion action head + cross-modal injection**是更优design。Engineering上简洁（FiLM一行公式），conceptually clean（System 1/2分工），实验全面（multi-task, sorting, bin-picking, bimanual, scaling）。

对foundation model设计者：future VLA应该认真考虑reasoning-conditioned action generation这个范式，而非纯NTP或纯diffusion。

参考汇总：
- Paper PDF (ICML 2025): https://proceedings.mlr.press/v267/
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- SigLIP: https://arxiv.org/abs/2303.15343
- Droid dataset: https://droid-dataset.github.io/
- OXE: https://robotics-transformer-x.github.io/
- LoRA: https://arxiv.org/abs/2106.09685
- vLLM: https://arxiv.org/abs/2309.06180
