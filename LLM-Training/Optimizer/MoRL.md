---
source_pdf: MoRL.pdf
paper_sha256: 200eb56ff52c57a7a08e9fc2a8f2245588041d6e0a7ae0fea2d20be0c66094b9
processed_at: '2026-08-05T20:31:27-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好嘞，那我用大白话重新讲一遍，咱们不堆公式，先把intuition建立起来。

---

## 这篇paper到底想干嘛

一句话：**让模型在生成动作之前，先"想一想"。**

你想想现在的text-to-motion模型是怎么工作的——你给它一句话"一个人先向左看，然后右脚踢东西"，它直接一把梭，把motion tokens吐出来。没有任何planning，没有"让我先理清这段动作分几步"的过程。

这就像你让一个人写作文，不让他打草稿，直接一气呵成。简单句子没问题，复杂的长句子肯定写乱。

MoRL的作者就觉得：motion这东西，尤其是长motion、组合动作，本质上是个**规划问题**。你得先在脑子里把动作分解成几个phase，每个phase是啥动作，phase之间怎么过渡，然后才能去生成具体的pose。

所以核心idea就是：**把CoT reasoning搬到motion领域**。让模型先输出一段natural language的reasoning trace（"这个动作分两步，第一步head turn left，第二步right foot kick..."），然后再输出motion tokens。

---

## 为什么现有方法不行

paper里点出了两个问题：

**第一，模型不会reasoning。** 现在的方法把prompt当成一个whole thing喂进去，没有decompose成细粒度的步骤。比如"走路然后坐下然后挥手"，模型没有一个内部过程去理清"先走、再坐、再挥"这个顺序，它就硬生成。

**第二，test time没有planning。** 推理的时候就是single-pass decode，一把出结果，没有reflection，没有"让我重新想想"的机制。LLM明明有reasoning能力，但没被用上。

你看Figure 1那个backflip的例子就很直观：MotionLLM生成的后空翻，起飞位置就漂了，空中旋转方向乱，落地也没稳住。MoRL生成的就是完整的"准备-起跳-旋转-落地-恢复"一整套。差别就在于MoRL先在reasoning trace里把这几个phase规划好了。

---

## MoRL怎么做的：三个关键步骤

### 步骤一：造数据（教模型怎么"想"）

要让模型会reasoning，得先有reasoning的数据。但motion领域没有现成的CoT数据集。

于是作者用**Gemini-2.5-pro**当老师，基于MotionHubV2这个motion数据集，造了两个数据集：

**MoUnd-CoT-140K**（给understanding用的）：
- 输入：一段motion sequence
- 输出：先写reasoning（这段动作的因果、时序结构），再写caption

**MoGen-CoT-140K**（给generation用的）：
- 输入：一句text
- 输出：先写reasoning（这段文字应该分解成什么动作步骤），再写motion

格式长这样：
```
思考过程：这个动作分两步，第一步...
<answer>
最终答案（caption或motion）
</answer>
```

关键insight：understanding和generation是**互逆问题**，用同样的reasoning format来组织，让模型在同一个representation space里学双向mapping。

### 步骤二：Cold Start（先教会格式，再谈优化）

这里有个很重要的engineering insight。作者一开始想直接用RL让模型自己emerge出reasoning能力（学DeepSeek-R1的做法），但发现**根本训不动**。

为什么？因为math/code的reward很sharp——答案对就是对，错就是错。但motion的reward是continuous的、noisy的（cosine similarity、NLI probability），信号太弱。模型没有先验的reasoning format，就很容易reward hacking或者直接collapse。

所以得先用SFT把**reasoning的格式**教会。用上面造的CoT数据集做supervised fine-tuning，让模型学会"先输出think，再输出answer"这个输出格式。

这个insight其实很general：**RL需要一个稳定的starting point**。你不能指望RL从零开始emerge出structure，它只能refine已有的structure。

### 步骤三：RLVR（用reward去精调）

SFT之后，模型会了格式，但reasoning质量可能还不够好。这时候上RL。

作者用**GRPO**（来自DeepSeek）做优化。GRPO的核心思路：不用训value network（PPO那样），而是每个prompt sample一组candidates（K=8个），在group内做relative comparison。

然后设计了**四个task-specific的reward**，分两个任务：

**Understanding任务（motion→text）两个reward：**

1. **Semantic Alignment**：生成caption和reference caption的语义相似度。用text encoder算cosine similarity。就是看你生成的描述和ground truth语义上像不像。

2. **Reasoning Coherence**：reasoning trace逻辑上能不能支撑你的answer。用一个frozen的NLI模型（DeBERTa-v3-large）来判断"这段reasoning是否entail这个answer"。

这个第二个reward是关键创新点。它不看你reasoning写得漂不漂亮，而是看你的reasoning和你的结论之间**逻辑上是否成立**。防止模型写出一段看起来很合理但和answer无关的reasoning。

**Generation任务（text→motion）两个reward：**

3. **Physical Plausibility**：生成的motion物理上合不合理。具体惩罚两件事：
   - $L_{joint}$：关节角度违反生理限制（比如膝盖往反方向弯）
   - $L_{vel}$：速度突变（上一帧还在走，下一帧瞬移）
   - 权重 $\lambda_1=0.8$ 给关节限制，$\lambda_2=0.2$ 给速度平滑

4. **Text-Motion Consistency**：生成的motion和输入text的跨模态对齐度。用text encoder和motion encoder分别encode，算cosine similarity。

每个reward都先在group内做normalization（减均值除标准差），把absolute reward转成relative advantage。这个normalization很重要——简单prompt大家都做得好，复杂prompt大家都做得差，group内相对比较能cancel out这个baseline差异。

---

## Test Time：Chain-of-Motion (CoM)

训练完了，推理的时候怎么用reasoning能力？

作者提出**Chain-of-Motion** decoding strategy。核心流程：

1. 给一个prompt，模型先generate reasoning trace
2. 基于reasoning trace，generate motion candidates
3. 采样K=8个candidate（reasoning+motion pairs）
4. 用上面那几个reward给每个candidate打分
5. 低分的扔掉，高分的做T=2轮refinement（reflection）
6. 最后输出最好的那个

这其实就是把test-time search和reasoning结合了。不是一把decode完事，而是generate→evaluate→refine的循环。

代价是latency从8.7ms涨到18.4ms（大概2.1倍），但质量提升明显，作者觉得这个trade-off可以接受。

---

## 实验结果怎么样

### Understanding（motion→text）

在HumanML3D上，MoRL的CIDEr达到35.8，比Motion Agent的33.74高不少。BLEU@1、BLEU@4、ROUGE-L、BERTScore全面领先。

这说明semantic alignment reward（capturing语义）+ reasoning coherence reward（保证逻辑一致）的组合确实有效。

### Generation（text→motion）

R-Precision全面提升，MM Distance最低（2.790），说明text-motion对齐最好。

FID虽然不是最低（0.203，比diffusion-based的0.045差），但作者解释说这是因为RL优化的目标是alignment和plausibility，不是纯pixel-level reconstruction。这个trade-off是合理的——你更在乎语义对齐和物理合理，还是更在乎distribution match？MoRL选择了前者。

### Ablation Study

这个table很说明问题。从SFT only开始，逐步加东西：

- 去掉 $R_{sem}$：BERTScore和CIDEr明显掉（语义reward很重要）
- 去掉 $R_{coh}$：ROUGE-L和CIDEr掉（逻辑一致性reward很重要）
- 去掉 $R_{phys}$：FID从0.203涨到0.285（物理reward对生成质量关键）
- 去掉 $R_{align}$：R-Precision从0.527掉到0.492（跨模态对齐reward关键）
- 去掉CoM：所有指标中等下降（test-time reasoning确实有用）

每个组件都有贡献，没有一个是冗余的。

### Reward对比实验

这个实验很有意思。作者构造了一个Complex Motion Subset（CMS），专门挑那些长时序、组合性强的复杂prompt来测。

对比四种reward设计：
- Motion-R1 style（outcome-based）：R@1还行，R@2/R@3就掉，说明后期动作容易被忽略
- MotionRL reward：FID好一点，但对stage-level语义gap不敏感
- Process-aware reward：temporal coherence好一点，但linguistic alignment不足
- MoRL reward：R@2和R@3最好，MM Distance最低

这说明MoRL的reward设计在**长时序、组合动作**这个场景下确实有优势。

---

## 我的几点看法

**第一，cold start这个insight很valuable。** 不是所有domain都能像math/code那样直接RL emerge出reasoning。reward signal的性质决定了RL能不能work。motion这种continuous reward、noisy signal的domain，SFT前置是必要的。这个insight对未来其他domain做RLVR都有参考价值。

**第二，reasoning coherence reward用NLI模型是个巧思。** 它把"reasoning是否合理"这个看似主观的东西，转化成了一个verifiable的NLI任务。虽然NLI模型本身不完美，但它提供了一个stationary的、可计算的signal，足够给RL用。

**第三，CoM的test-time search思路和o1、R1的趋势是一致的。** 推理时多花点compute换质量，这个方向是对的。motion这个domain尤其适合——因为motion的output space巨大，single-pass decode很难考虑全局constraint。

**第四，limitation也很诚实。** Rule-based reward需要adapt到新domain，CoM有latency开销，discretized representation丢失了fine-grained contact dynamics。这些都是未来work的方向。

---

## 参考资料

- MoRL项目主页：https://aigeeksgroup.github.io/MoRL
- MoRL代码：https://github.com/AIGeeksGroup/MoRL
- DeepSeek-R1（RLVR范式来源）：https://arxiv.org/abs/2501.12948
- GRPO原始paper（DeepSeekMath）：https://arxiv.org/abs/2402.03300
- HumanML3D dataset：https://github.com/EricGuo5513/HumanML3D
- MotionGPT（baseline对比）：https://arxiv.org/abs/2306.14795
- Motion-R1（同期相关工作）：https://arxiv.org/abs/2506.10353
- Qwen3技术报告：https://arxiv.org/abs/2505.09388
- DeBERTa-v3（NLI模型）：https://huggingface.co/microsoft/deberta-v3-large
- VQ-VAE原始paper：https://arxiv.org/abs/1711.00937

---

总结一下，MoRL这个工作把"先想再做"的reasoning范式成功搬到了motion领域。核心技术贡献是：用SFT+RLVR的训练pipeline，配合task-specific的四重reward设计，加上test-time的Chain-of-Motion decoding。实验证明了在长时序、组合动作这个最难的场景下，reasoning确实能带来实质性的提升。

这个work对motion generation community的启示是：**不要只盯着architecture innovation，reasoning和planning可能是下一个breakthrough方向**。

---

# MoRL: Reinforced Reasoning for Unified Motion Understanding and Generation

很高兴和你聊聊这篇paper, Andrej。这个工作我看了之后很有启发，它把DeepSeek-R1的RLVR范式迁移到了motion-language这个领域，而且处理得相当细致。让我从intuition出发，把整个故事讲清楚。

## 1. 核心问题的intuition

先想想motion这个domain为什么难。Text-to-motion generation表面上和text-to-image很像，但本质上motion是一个**时序的、物理约束的、组合性的**信号。一段"先向左看然后用右脚踢东西"的motion，不是一个static的distribution可以sample出来的，它需要**temporal planning**——模型要先规划head turn这个phase，再规划kick这个phase，两个phase之间还要有physically plausible的transition。

现有的方法（MotionGPT、Motion Agent、MotionLLM等）大都把prompt当成一个整体喂进去，然后single-pass decode出motion tokens。问题就出在这里：LLM的reasoning能力没有被利用起来。模型没有"先想清楚要做几步，每步的action primitive是什么"这种planning过程。

MoRL的核心insight就是：**把motion generation/understanding重新formulate成一个reasoning问题**，让模型先输出一个natural language的reasoning trace，然后再输出motion tokens或者caption。这其实就是把CoT从text domain搬到motion domain，但关键在于怎么训练、怎么设计reward、怎么在test time利用这个reasoning。

## 2. 整体架构

### 2.1 模型backbone

MoRL基于**Qwen3-4B-Instruct**作为MLLM backbone。这个选择很有意思——4B的规模足够capturing language reasoning能力，但又不会太大导致训练cost爆炸。在这个backbone上，插入了两个modality-specific的tokenizer：

- **Text tokenizer**: 直接继承Qwen的原生tokenizer
- **Motion tokenizer**: VQ-VAE style，把continuous motion discretize成tokens

两个模态通过shared transformer layers做cross-attention融合。这是比较标准的MLLM设计，类似DeepSeek-VL那种思路。

### 2.2 Motion Tokenizer的技术细节

这块值得仔细讲。给定motion sequence：

$$m_{1:T} \in \mathbb{R}^{T \times D}$$

变量含义：
- $T$: 帧数
- $D$: 每帧的特征维度（HumanML3D是263维，KIT-ML是251维）

Encoder $E$ 把这个序列压缩成latent vectors：

$$z_{1:(T/l)} \in \mathbb{R}^{(T/l) \times d}$$

变量含义：
- $l$: 下采样因子，每 $l$ 帧压缩成一个latent
- $d$: latent dimension，paper里是128
- $T/l$: 压缩后的序列长度

然后每个latent $z_i$ 通过nearest neighbor lookup量化到codebook：

$$\hat{z}_i = \arg\min_{c_n \in \mathcal{C}} \|z_i - c_n\|_2^2$$

变量含义：
- $\mathcal{C} = \{c_n\}_{n=1}^N$: learnable codebook
- $N=512$: codebook size
- $c_n$: 第 $n$ 个codebook entry

量化后的 $\hat{z}_{1:(T/l)}$ 通过decoder $D$ 重构回motion $\hat{m}_{1:T}$。

VQ-VAE的training loss是组合形式：

$$\mathcal{L}_{vq} = \mathcal{L}_{reconstruct} + \mathcal{L}_{commit} + \mathcal{L}_{embed}$$

三个部分：
- $\mathcal{L}_{reconstruct}$: smoothed L1 loss + velocity regularization，保证重构精度和motion smoothness
- $\mathcal{L}_{commit}$: commitment loss，强迫encoder输出靠近codebook entries，防止codebook collapse
- $\mathcal{L}_{embed}$: embedding loss，stabilize codebook representation

这个设计的intuition是：motion是连续信号，但LLM只能处理discrete tokens。VQ-VAE把motion压缩成一个compact的token sequence，既减少了sequence length（$T \to T/l$），又和autoregressive generation paradigm无缝对齐。codebook size 512意味着每个motion token大约是9 bits的信息量，这在motion这种相对低熵的信号上是足够的。

### 2.3 LoRA配置

MoRL用LoRA做efficient fine-tuning：
- rank $r=16$
- dropout 0.1
- 插入到attention和feed-forward layers

这个rank选择对4B模型来说是合理的——足够expressive来学习motion-language alignment，又不会引入太多参数。

## 3. Cold Start Stage: 为什么不能直接RL

这是paper里一个很重要的engineering insight。作者一开始motivated by DeepSeek-R1，想直接用RL训练让模型self-emerge出CoT reasoning。但实验发现**highly unstable**：

> "the model rarely produced well-formed reasoning traces and even generated answers that deviated from the intended semantics"

这个观察和DeepSeek-R1原文的发现有点不同。R1在math/code这种domain可以直接用RL emerge出reasoning，因为math有明确的verifiable reward（答案对不对）。但motion domain的reward是**continuous and noisy**的（cosine similarity、NLI probability），没有那么sharp的signal。模型没有先验的reasoning format，就很容易陷入reward hacking或者直接collapse。

所以cold start stage用SFT来教模型**reasoning的格式**。用MoUnd-CoT-140K和MoGen-CoT-140K，每个样本是这种形式：

```
<motion sequence or text>
<answer>
final answer (caption or motion)
</answer>
```

SFT的作用是让模型学会这个output format，建立一个稳定的starting point给后续RL。这个insight其实和InstructGPT、Llama-2的训练pipeline是echo的——RLHF之前都需要SFT做alignment。

### 3.1 SFT超参
- Optimizer: AdamW
- Learning rate: $1 \times 10^{-5}$
- Batch size: 64
- Weight decay: 0.01
- Epochs: 5

lr比较保守，这符合SFT阶段想preserve base model capability的intuition。

## 4. 数据合成：MoUnd-CoT-140K 和 MoGen-CoT-140K

这是paper的一个核心贡献。作者构建了一个data engine，基于**Gemini-2.5-pro**来生成CoT数据。

### 4.1 Data Engine的设计

输入来自**MotionHubV2** dataset（Ling et al., 2024），这是一个aggregate了多个public motion capture dataset的corpus，涵盖dance、performance interaction、daily activities等场景。

两个branch：

**MoUnd-CoT-140K** (Motion Understanding):
- Input: motion sequence + caption
- Output: `<answer>caption</answer>`
- Motion用SMPL-X format，转换成HumanML3D的263维features
- 用Gemini生成gap-based reasoning chain（通过QA pair构造）

**MoGen-CoT-140K** (Motion Generation):
- Input: text caption
- Output: `<answer>motion sequence</answer>`
- Motion用SMPL-X format，normalize到HumanML3D feature space

这个设计的精妙之处在于**对称性**——understanding和generation是inverse problem，用同样的CoT format来组织，让模型在shared representation space里学习双向mapping。这和MotionGPT的"motion as a foreign language"思路有精神上的传承，但加了reasoning layer。

### 4.2 为什么要用Gemini-2.5-pro

作者选择Gemini-2.5-pro作为data generation的backbone，这是有考量的。Gemini-2.5在multimodal reasoning上很强，能够理解motion sequence（以某种textual description的形式）并generate合理的reasoning chain。这里有个subtle的点：motion本身是连续信号，Gemini并不能直接"看"motion，它看的是motion的textual representation（joint positions、velocities等的描述）。所以这个data engine本质上是在做**knowledge distillation**——把Gemini的reasoning ability蒸馏到MoRL里。

## 5. Reinforcement Learning: RLVR的设计

这是paper最核心的技术贡献。作者采用**GRPO**（Group Relative Policy Optimization，来自DeepSeek）作为优化算法，但reward design是task-specific的。

### 5.1 GRPO的intuition

GRPO和PPO的关键区别在于**不需要value network**。PPO需要训练一个critic来估计advantage，但critic本身很难训，尤其在高维output space里。GRPO的做法是：

1. 对每个prompt，sample一个group的 $K$ 个candidates
2. 用reward function给每个candidate打分
3. 在group内做normalization得到advantage
4. 用normalized advantage做policy gradient

paper里 $K=8$。

### 5.2 Group-wise Normalization

给定一个candidate group $\{r_1, r_2, \dots, r_K\}$，每个reward做normalization：

$$\tilde{r}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}$$

其中：

$$\mu_r = \frac{1}{K} \sum_{j=1}^{K} r_j$$

$$\sigma_r^2 = \frac{1}{K} \sum_{j=1}^{K} (r_j - \mu_r)^2$$

变量含义：
- $r_i$: 第 $i$ 个candidate的raw reward
- $\mu_r$: group内reward均值
- $\sigma_r$: group内reward标准差
- $\epsilon$: 数值稳定性的小常数

这个normalization的intuition是：不同prompt的absolute reward scale可能差别很大（简单prompt的reward普遍高，复杂prompt普遍低），但在group内做relative comparison可以cancel out这个baseline difference。这就是**relative advantage**的核心思想，和AlphaGo的self-play、RLHF里的preference learning有精神上的联系。

### 5.3 Motion Understanding的Rewards

两个reward：

**Semantic Alignment Reward**:

$$R_{sem} = \cos(E_{text}(\hat{a}), E_{text}(a))$$

变量含义：
- $\hat{a}$: 模型生成的caption
- $a$: reference caption
- $E_{text}$: pretrained text encoder
- $\cos$: cosine similarity

这个reward直接measure生成caption和reference caption的semantic similarity。用text encoder的embedding做cosine similarity，比BLEU这种surface metric更能capturing semantic equivalence。

**Reasoning Coherence Reward**:

$$R_{coh} = f_{NLI}(\hat{r}, \hat{a})$$

变量含义：
- $\hat{r}$: reasoning trace（`
