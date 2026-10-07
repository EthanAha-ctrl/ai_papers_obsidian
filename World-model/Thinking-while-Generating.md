---
source_pdf: Thinking-while-Generating.pdf
paper_sha256: c25ae2e97fcf876271378b0562dd973d321ce275ecd975a16776a664392fbc70
processed_at: '2026-08-12T15:43:14-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 TWIG 这篇 paper

好的 Andrej,我换个讲法,不搞那种 academic 的八股,直接讲核心 idea 和背后的直觉。

---

## 一句话概括

现在的 text-to-image model 画图就是 "闭着眼睛一口气画完",画得好不好全靠运气和 model 本身的 capability。TWIG 说:我们能不能让它**画几笔就停下来想想,想想再继续画,画完一块还能回头看看对不对、要不要改**——就像人画画的过程一样。

就这么个事儿。

---

## 现在的问题在哪

你用 Midjourney、DALL-E、SDXL 这些 model 画图,prompt 进去,图出来,中间发生什么你完全控制不了。model 内部是一个 black box 的 forward pass,一口气把所有 pixel / token 都 generate 出来。

这带来几个具体的问题:

**长 prompt 记不住**。你让它画 "一只红色的猫坐在蓝色椅子上,左边有个绿色的花瓶,右边窗外下着雨,地上有落叶"——它经常 color 错位、object 漏掉、spatial 关系混乱。因为整个 prompt 被一次性 encode,没有一个 mechanism 去逐步 check "我画到这里,前面那些东西都落实了吗"。

**多 object 关系容易崩**。"猫在椅子左边"这种 spatial relation,model 经常搞反或者干脆 ignore。

**没法 mid-course correction**。生成一旦开始,trajectory 就定了。哪怕 model 在第 30% 的时候其实已经走歪了,它也没有一个 mechanism 去说 "等等,这块画得不对,我重来"。

社区之前有两个方向的尝试:

**Think-before**:先生成一个 plan(caption 扩写、layout 规划),然后照着 plan 画。问题是 plan 是死的,画的时候如果发现 plan 本身有问题,没法改。就像你旅行前做了 detailed itinerary,但到了现场发现路况不对,你还得照着 itinerary 走。

**Think-after**:先画完整张图,然后让 model 自己 critique,发现问题再重画。问题是成本高——每次要重画整张图,而且 reasoning 和 generation 是 loosely coupled 的,critique 的信息很难 precise 地 map 回具体哪个 region 出了问题。

TWIG 说:这俩都不对。**应该边画边想**。

---

## TWIG 的核心 idea

直觉特别简单。你想想人怎么画一幅复杂的画:

1. 先有个 overall 构思(画什么、大致布局)
2. 画背景(天空、远山)
3. 停下来看看——背景画得对吗,颜色对不对,要不要调
4. 画中景(主体 object)
5. 再停下来看看——主体和背景搭不搭,位置对不对
6. 画前景(细节、阴影、texture)
7. 最后再看一眼——整体 coherent 吗,哪块需要 refine

TWIG 就是把这个过程 simulate 到 model 的 generation trajectory 里。

具体怎么做?它把图像分成三块:**upper background、central content、lower background**。然后:

- 生成第一块之前,先 think 一下("我要画天空,阴天的,灰蓝色调")
- 生成第一块
- think 一下这块画得怎么样("颜色有点偏紫,不对,应该是灰蓝"),如果不对就 local re-generate 这一小块
- think 一下接下来画什么("中间是一只猫,橘色,坐姿")
- 生成第二块
- 再 reflect 一下
- think 最后一块("下面是草地,绿色,有露水")
- 生成第三块
- 最后 reflect

关键在于:**整个过程是一个 single autoregressive trajectory**。不是 generate → 停 → 重新跑一个 forward pass → 再 generate。是 text token 和 image token 交替出现在同一个 sequence 里,model 一路 decode 下来。

---

## 为什么这个能 work

我觉得这篇 paper 最有意思的发现其实是:**zero-shot 就能 work**。

你不用 fine-tune,不用 RL,就是给 Janus-Pro [7] 一个精心设计的 prompt,告诉它 "你现在要边画边想,分成三块,每块画完要 critique,用这 5 个标准打分",它居然就照做了,而且效果比 baseline 大幅提升:

- Color binding 从 63.59 → 73.11(+9.5)
- Texture binding 从 49.36 → 64.77(+15.4)
- Complex composition 从 35.59 → 48.16(+12.6)

这说明什么?**ULM 的 pretraining 已经让它具备了 "在 visual generation 中插入 reasoning" 的 latent capability**,只是之前没人这么去 elicit 它。

这其实和你讲 LLM 的 in-context learning 时说的一个道理一样——model 在 pretraining 时见过大量 "文本中穿插思考" 的 pattern(Twitter、Reddit、blog、代码注释),所以它天然知道怎么 "think step by step"。Janus-Pro 这种 ULM 见过大量 interleaved image-text data,所以它也知道怎么在 image token 中间插入 text reasoning。你只要 prompt 对了,这个 capability 就出来了。

当然 zero-shot 有 instability 的问题——同一个 prompt 跑 5 次,std 比 SFT 后大不少。SFT 的价值主要不在于把 peak performance 推多高,而在于让 behavior 变 predictable。

---

## 三个组件,人话版

### When to Think:什么时候停下来想

就一个问题:画图过程中插几次 thinking。

实验发现 **3 次最好**。为什么 3 次?因为大多数图可以自然分成三块——上面、中间、下面。上面通常是天空/背景,中间是主体,下面是地面/前景细节。

试过 2 次和 4 次,2 次太少(覆盖不够),4 次没有额外收益(反而可能因为 partition 太碎导致 coherence 下降)。

也试过 adaptive schedule(让 model 自己决定每块大小),结果反而更差。因为现在的 ULM 还没强到能稳定地 follow "请输出每块的相对比例" 这种 instruction,schedule 一乱整个 generation trajectory 就崩了。

所以一个很朴素的 prior(uniform partition into 3)赢了花哨的 adaptive 方法。这种事情在 deep learning 里反复发生——inductive bias 有时候比 learned policy 更 work,especially 当 model 能力还不够的时候。

### What to Say:想什么内容

每次 think 的时候,model 输出一段 text,内容是关于**接下来要画的这块区域**的 sub-caption。

比如全局 prompt 是 "一只猫坐在窗台上看雨景",那三次 think 可能是:

- $\tau_1$:"上半部分是窗外的雨景,灰蒙蒙的天空,雨滴斜落,色调偏冷"
- $\tau_2$:"中间是一只橘猫,侧坐,看向窗外,毛色温暖,眼睛半眯"
- $\tau_3$:"下半部分是窗台,木质,深棕色,边缘有一盆绿植"

每段 thought 都只 focus 在 local region,但会参考之前已经生成的 region 的内容,保证 coherence。

这里有个 clever 的工程实现:$\mathrm{ULM}_g$ 只需要 text-to-image 能力,不需要 image-to-image。因为已经生成的 visual token 保持在 sequence 末尾不动,新加的 text thought append 在前面,然后 model 继续往后 decode 新的 visual token。整个过程没有 "拿生成的图再输入回去" 这种操作,全是 single-pass autoregressive。

### How to Refine:画完一块要不要改

每次画完一块,model 给自己打一个分,0 到 100,标准是 5 个维度:color accuracy、object completeness、detail richness、spatial relationships、visual coherence。

如果分数超过一个阈值(比如 80),这块就过了,直接进入下一块。
如果分数低,model 会输出一个 revised sub-caption("刚才画的天空太紫了,应该更灰蓝一点"),然后**只重新生成这一小块**,替换掉原来的。

注意是 **local re-generation**,不是整张图重画。这和 think-after-generation 的关键区别就在这里——cost 是局部的,不是全局的。

实验发现 **1-round reflection 最好,2-round 没有额外收益**。说明 zero-shot 下 model 的 critique-and-revise 能力有上限,一轮能抓到主要问题,再抓也抓不出更多了。要 unlock 更多,得靠 SFT 和 RL。

---

## SFT 和 RL 各自加了什么

### SFT:让 behavior 稳定下来

作者构造了一个 TWIG-50K 的数据集,用 GPT-4o 生成 sub-caption + GPT-4o-Image 生成对应的图,然后 filter 成 interleaved format。总共 9 个 subtask(think × 3 + reflect × 3 + generate × 3)。

SFT 的结果:

- Shape binding 从 41.55 → 52.42(+10.87,这是最大的 single gain)
- Spatial 从 21.98 → 27.02(+5.04)
- Complex 从 48.16 → 53.41(+5.25)

Shape 提升最大说明:SFT 帮 model 学到了"怎么用语言 precise 地描述 shape",这个在 zero-shot prompt 里很难 elicit。

一个意外发现:**加 reflection 数据反而变差**。Reflect-heavy 和 reflect-lite 都不如不加 reflection data。作者的解释是 zero-shot 已经把 reflection 能力 expose 得差不多了,硬加 reflection data 会让 thought 变长、over-correction 变多,反而干扰了 think 和 generate 的学习。

这其实是一个普遍现象:监督信号太强反而会 distort policy。就像你给学生太多 "正确答案" 的示范,他反而会 overfit 到 surface pattern,失去灵活性。

SFT 还有一个重要价值:**variance 降下来了**。5 个 random seed 的 std 在所有维度都降低了。SFT 给你的不是更高的 peak,是更 predictable 的 behavior。这个在 product 场景里其实更重要。

### RL:真正把上限推上去

用 GRPO(DeepSeekMath [41] 那个),基于 SFT model 继续 optimize。

关键设计:**joint reinforcement**。一个 rollout 里有 up to 9 个 subtask(think + reflect + generate 各 3 个),用一个 shared reward(基于 final image quality)同时 update 所有 subtask 的 policy。不分开学,一起学。

为什么 joint 比分开好?因为 interleaved reasoning 是一个 sequential decision process,局部 reward 很难定义——"这一步 think 得好不好"很难直接评估,但"最终图好不好"是 well-defined 的。用一个 final reward backpropagate 到整个 trajectory,相当于用 outcome 来 credit assignment 所有的中间 decision。

这个思路和 AlphaGo 一样——用最终胜负 reward 训练整盘棋的 trajectory,不需要每一步都有 immediate reward。

实验结果:

| Strategy | Color | Shape | Texture | Spatial | Non-Spatial | Complex |
|----------|-------|-------|---------|---------|-------------|---------|
| 只 reinforce generation | 80.12 | 59.87 | 72.01 | 32.47 | 31.30 | 54.02 |
| 只 reinforce understanding | 78.36 | 57.94 | 70.68 | 30.93 | 31.27 | 53.76 |
| **Joint (TWIG-GRPO)** | **82.49** | **61.28** | **73.19** | **34.06** | **31.99** | **54.45** |

Joint 全面领先。这证明 thinking 和 generation 是 **mutually reinforcing** 的——thinking 质量高了,generation 质量就高;generation 质量高了,下一轮 thinking 有更好的 visual context 可以 reference。分开学,两边都学不充分。

Reward model 用了 4 个,unweighted average:

- HPS v2 [51]:overall 美感和 style coherence
- GroundingDINO [31]:object detection,管 entity 是否存在、location 对不对
- GIT [48]:VQA consistency,管 prompt 里的 instruction 是否被遵守
- Fine-tuned ORM [17]:holistic text-image alignment

为什么要用多个 reward model 的 ensemble?因为每个 reward model 只 capture 一个维度,单一 reward model 会被 hack(比如 HPS 高但其实 object 都不对)。Ensemble 一下,各个维度互相约束,reward hacking 的问题就缓解了。

最终在 T2I-CompBench++ 上,TWIG-RL 在 Complex 类别拿到 53.56,比之前 SOTA(T2I-R1 [23] 的 39.93)高出 13.63 分。这是一个非常大的 gap,说明 TWIG 的 paradigm 在 complex composition 上确实有 structural advantage。

---

## 几个我觉得真正重要的 intuition

### 1. Reasoning 的时序结构很重要

以前大家觉得 reasoning 就是在 generation 前面加一段 CoT,或者在后面加一个 refine stage。TWIG 说:reasoning 应该 **woven into** generation trajectory 里,和 visual content co-evolve。

这个 insight 其实挺深刻的。因为 generation 是一个 sequential process,每一步的 output 都会影响后面的 decision。如果 reasoning 只在开头,它没法 adapt 到 generation 过程中出现的意外;如果只在结尾,它没法 intervene 到已经生成的 content 里。只有 interleaved,reasoning 才能 **react to intermediate state** 并且 **guide future generation**。

### 2. Zero-shot capability 是 latent 的

Janus-Pro 在 pretraining 时从来没被显式教过 "边画边想",但一个合适的 prompt 就把这个能力 elicit 出来了。这说明 ULM 的 pretraining data 里已经包含了足够多的 "interleaved text-image reasoning" 的 pattern,只是这个 capability 从来没被 probe 出来。

这个发现对未来的 implication 很大:可能很多我们觉得需要专门训练才能获得的 capability,其实在 pretraining 后已经存在,只是我们没找到合适的 "activation key"——prompt 或者 lightweight fine-tuning。

### 3. Simple prior 经常 beat fancy learned policy

K=3 uniform partition 干掉了 adaptive schedule。一个简单的 "图像分上中下三块" 的 prior,比让 model 自己学怎么 partition 更 work。

这不是因为 adaptive 本身不好,是因为 current ULM 的 planning 能力还撑不起 adaptive schedule。等 model 更强了,adaptive 应该会 win。但现在,一个手工的、简单的 inductive bias 更 reliable。

Deep learning 里这种事反复发生——residual connection、attention、layernorm,最开始都是手工设计的 inductive bias,然后才被理论 justify。

### 4. Joint reinforcement 比 separate 强

这是 paper 里我觉得对 RL community 最有信息量的发现。在 multi-step reasoning pipeline 里,用 final outcome reward joint reinforce 所有 step,比每个 step 单独 reinforce 更 work。

这暗示了一个更深的道理:**interleaved reasoning 的 value 不在于单个 step 的质量,而在于 step 之间的 coherence**。分开 reinforce,每个 step 都往自己的 local optimum 走,合起来不一定 globally optimal。Joint reinforce,所有 step 都被同一个 outcome signal 拉,它们会 naturally align 到 globally coherent 的 policy。

### 5. Visual generation 的新 axis:reasoning depth

以前我们 scale visual model 的两个 axis 是:model size 和 data size。TWIG 打开了第三个 axis:**reasoning depth**,也就是在 generation 过程中插入多少次 thinking。

K=3 已经大幅超过 K=1(就是 think-before-generation)。如果未来 K=10、K=50,reasoning 越来越 dense,会不会继续提升?这其实和 LLM 里的 test-time compute scaling 是同一个道理——给 model 更多 inference-time 的 "思考预算",performance 就能继续涨。

这可能是比单纯 scale model parameter 更 cost-effective 的方向。

---

## 和其他工作的关系

### 和 LLM reasoning 的对称性

LLM 里的 reasoning 已经很成熟了:CoT、Self-Refine、Reflexion、Tree-of-Thoughts。LLM 天然是 "think while generating"——每个 token 的生成都在利用前面所有 token 的 context,reasoning 是 dense 的。

Visual generation 之前没有这种 dense reasoning,因为 pixel/token 生成没有 explicit reasoning signal。TWIG 的做法是强行在 visual token stream 里插入 text token,给 visual generation 加一个 "reasoning scaffold"。

这是一个 modality transfer 的 insight:**text modality 的 reasoning capability 可以 cross-modality 地 inject 到 visual generation 里**,只要你的 model 是 unified 的(像 Janus-Pro 那样 text 和 image 在一个 backbone 里)。

### 和 o3 "thinking with images" 的对偶

OpenAI o3 [33] 在 reasoning 的时候会 generate / crop / zoom images 作为 scratchpad——thinking 时用 image 辅助。TWIG 在 generating images 的时候用 text reasoning 辅助——generating 时用 text 辅助。

这两个方向是 **dual** 的,最终 converge 到同一个终点:**modality-agnostic 的 reasoning-generation loop**。在这个终点上,model 可以在 reasoning 过程中自由地 generate 任何 modality 的 content,也可以在 generation 过程中自由地 reason。text、image、video、code 全部在一个 trajectory 里 interleave。

这可能是 AGI 的一个必要 component——reasoning 和 generation 不应该被 artificially 分开。

### 和 concurrent works 的区别

Paper 里提了两个 concurrent work:IRG [21] 和 Uni-CoT [36]。它们也叫 "interleave",但本质上是 think-before + think-after 的组合——把 generation 当成一个 monolithic block,thinking 只发生在 block 前面和后面,不 penetrate 进 generation 过程内部。

真正的 interleaving 应该是 token-level 的:生成 100 个 image token,think 一下,再生成 100 个,再 think 一下。TWIG 现在是 region-level(K=3),离 token-level 还远,但方向是对的。

Token-level TWIG 会面临 cost 问题——如果每 100 个 token 就 think 一次,thinking token 会远多于 generation token,推理成本爆炸。所以需要 sparse thinking 机制,比如让 model 自己 learn 什么时候值得 think(通过 RL 来学 schedule)。

---

## 局限和未来方向

Paper 自己承认的:

1. **Fixed K=3 schedule 不 optimal**。Adaptive schedule 需要 RL 来学,但现在 ULM 还不够强。未来更强的 model 可能能 learn 出比 "上中下三块" 更 smart 的 partition,比如按 object boundary 分、按 semantic complexity 分。

2. **用的 original GRPO**,没试更新的 RL variants。DAPO [55]、GSPO [58] 这些可能更 stable 或更 sample-efficient。

3. **只做了 T2I**,没扩展到 video、3D、image-to-image。Video 可能是最 exciting 的方向——因为 video 天然有 temporal structure,thinking 可以和 frame generation interleave(比如 "这段镜头的人物应该从左走到右,下一段切到特写")。

我额外想到的:

4. **Token-level TWIG**:现在是 region-level(K=3),往 token-level 走需要 sparse thinking mechanism。这可能需要新的 architecture design——比如一个 router network 决定 "接下来 N 个 token 要不要 think"。

5. **Continuous diffusion 上的 TWIG**:现在只在 autoregressive model 上做。Diffusion model 的 denoising step 之间也可以 interleave text reasoning——每 denoise 10 步 think 一次。这个方向完全 unexplored,但理论上可行,因为 LMM 可以读 partial denoised image 然后 reason。

6. **Reward hierarchy**:现在 4 个 reward model 是平权的。未来应该有 coarse-to-fine 的 reward structure——early thinking steps 用 coarse reward(overall composition 对不对),late steps 用 fine reward(color、texture、detail 对不对)。这和 hierarchical RL 的 idea 一致。

7. **Self-improvement loop**:现在 ZS → SFT → RL 是 linear pipeline。未来可以做成 iterative——RL 后的 model 重新 generate TWIG-50K,再 SFT,再 RL,形成一个 self-play 的 loop。这和 STaR [46] 的 idea 类似。

---

## 最后总结

TWIG 这篇 paper 做的事情,本质上就是给 visual generation 加了一个 "inner monologue"。以前 model 画图是 "闷头画",现在变成 "边画边自言自语"——自言自语的内容是 "接下来画什么、画得对不对、要不要改"。

这个 idea 本身很直觉,几乎 obvious。但论文的价值在于:

- **第一次把这个 idea formalize 成一个 complete framework**(when / what / how 三个 component)
- **第一次在 unified model 上实现并验证**(Janus-Pro,不用 pipeline)
- **第一次展示 ZS → SFT → RL 的完整 progression**,每一阶段都有 informative 的 finding
- **在 standard benchmark 上显著超过 SOTA**,证明这不只是 toy idea

我觉得这个方向才刚开始。K=3 是 baby step,真正的终态是 **token-level interleaved reasoning**,model 可以在 generation 的任何位置自由地 think,think 的内容由 RL 学出来,schedule 也是 learned 的。到那个程度,visual generation 就真的有了 "System 2" 的能力。

Paper 链接再放一下:
- Project: https://think-while-gen.github.io
- Code: https://github.com/ZiyuGuo99/Thinking-while-Generating

希望这次讲得够 "人话" 了,Andrej。

---

# Thinking-while-Generating (TWIG) 深度解读

非常高兴能和你聊这篇 paper，Andrej。这篇工作虽然标榜 "preliminary study"，但它触及了一个我觉得相当 fundamental 的问题：**生成过程中的 reasoning 应该以什么样的时序结构嵌入到生成 trajectory 里**。直觉上，这其实和你以前讲过的 "System 2 thinking" 在 LLM 里的展开是一回事——只是 modality 从 text-only 翻转到了 visual generation。下面我把整篇 paper 拆开来细讲，并尽量把每一步背后的 intuition 讲清楚。

Paper 链接：https://think-while-gen.github.io
Code: https://github.com/ZiyuGuo99/Thinking-while-Generating
arXiv (相关 reference): https://arxiv.org/abs/2501.17811 (Janus-Pro), https://arxiv.org/abs/2402.03300 (GRPO/DeepSeekMath)

---

## 1. 这篇 paper 想解决的真正问题

我们目前的 text-to-image generator（不管是 diffusion 如 SDXL [34]、PixArt-α [6]、FLUX.1 [24]，还是 autoregressive 如 Emu3 [49]、Janus-Pro [7]、Show-o [52]）有一个共同的毛病：**它们本质上是 "one-shot" 的 forward pass**。即使有 CFG、attention guidance 这些 trick，整个生成过程是一个 monolithic 的 mapping $T \to I$，中间没有显式的 "思考"。

社区尝试用 CoT 来补救，但按 reasoning 注入的位置可以分成两类：

| 类别 | 代表方法 | 问题 |
|------|----------|------|
| Think-before-Generation | GoT [11], ImageGen-CoT [27], T2I-R1 [23] | Plan 一旦生成就固定，没有 mid-course correction |
| Think-after-Generation | CoT-Gen [17], Reflect-DiT [26], Reflection-Tuning [61] | 等整张图都生成了才 critique，成本高、耦合松 |

TWIG 的核心 insight 来自一个**镜像对称**：visual understanding 里的 LMM（GPT-4o [22]、o3 [33]、DeepEyes [59]）已经在做 image-text interleaved reasoning——在 text CoT 中 weave 进 visual evidence（crop、zoom、detect）来 support reasoning。TWIG 把这个 flow 反过来：**在 visual generation 的 trajectory 中 weave 进 textual thought**，让 reasoning 和 pixel/token 生成 co-evolve。

这其实呼应了你之前在微博 / Twitter 上提过的观点——reasoning 不应该是 "前后夹击" 的两个独立 stage，而是应该和 perception/action loop 紧密耦合。这里 visual generation 就是一种 "action"，textual reasoning 就是 "perception + planning"，二者在一个 trajectory 里 interleave。

---

## 2. Framework 的三个核心问题

TWIG 把整个 interleaved process 拆成三个 design axes，作者起的名字很贴切：**When to Think, What to Say, How to Refine**。

### 2.1 When to Think — 调度问题

公式：

$$S = \mathrm{ULM}_u(T), \quad S = \{\nu_k\}_{k=1}^{K}$$

变量解释：
- $T$：input text prompt
- $\mathrm{ULM}_u$：Unified LM 的 understanding forward pass（这里是 Janus-Pro 的 understanding head）
- $S$：interleaved schedule，一组 reasoning 点
- $\nu_k$：第 $k$ 个 reasoning 点对应的 target visual region（在 autoregressive 模型里是 token span，在 continuous diffusion 里是 timestep window）
- $K$：总 reasoning 次数

直觉：这本质上是问"在生成过程中，要在哪些 spatial-temporal 锚点上插入一次思考"。可以是 **static**（fixed $K$、uniform spacing）或 **adaptive**（variable $K$、content-dependent $\nu_k$）。

实验结论（Table 1 panel b 和 c）非常有意思：

| Schedule | Color | Shape | Texture | Spatial | Non-Spatial | Complex |
|----------|-------|-------|---------|---------|-------------|---------|
| K=2 | 72.79 | 42.26 | 64.64 | 21.97 | 30.89 | 49.71 |
| K=3 | **73.11** | 41.55 | **64.77** | 21.98 | 30.90 | 48.16 |
| K=4 | 72.95 | 41.90 | 64.70 | **22.03** | **31.10** | 48.90 |
| Uniform | **73.11** | **41.55** | **64.77** | **21.98** | **30.90** | **48.16** |
| Adaptive | 72.43 | 40.88 | 63.92 | 21.67 | 30.88 | 47.39 |

**K=3 最好**，背后的 heuristic 是：大多数图像可以分解为 upper background / central content / lower background 三个 semantic components。这其实是一个非常 prior-based 的发现，让我想到你讲 nanoGPT 时常说的 "inductive bias is sometimes all you need"——一个简单的 K=3 + uniform partition 就打败了花哨的 adaptive schedule。

Adaptive 失败的原因是 current ULMs 没法稳定地 follow "请输出每块的相对比例" 这种 instruction，schedule 一乱整个 trajectory 就崩了。这点其实暗示了一个更深的 open problem：**schedule 本身应该被 RL 学出来，而不是 prompt 出来**。这正是 paper 在 Limitation 里承认的。

### 2.2 What to Say — 推理内容

公式：

$$\tau_k = \mathrm{ULM}_u\left(T, \{\tau_j\}_{j<k}, \{\mathcal{V}_j\}_{j<k}\right)$$

$$\mathcal{V}_k = \mathrm{ULM}_g\left(\{\tau_j\}_{j \le k}, \{\mathcal{V}_j\}_{j<k}\right)$$

变量解释：
- $\tau_k$：第 $k$ 步的 textual thought，作为针对 $\nu_k$ 的 **localized sub-prompt**
- $\{\tau_j\}_{j<k}$：之前所有 thoughts（累积的 textual context）
- $\{\mathcal{V}_j\}_{j<k}$：之前所有生成的 visual regions
- $\mathrm{ULM}_g$：generation forward pass

**这里有一个我觉得非常聪明的设计**：$\mathrm{ULM}_g$ 不需要 image-to-image 能力，只需要 text-to-image。怎么做？看 Figure 4(a)：

- 把 textual pre-context 从 $\{\tau_j\}_{j<k}$ 扩展到 $\{\tau_j\}_{j \le k}$（在 sequence 前部 append）
- 已生成的 visual content $\{\mathcal{V}_j\}_{j<k}$ 在 sequence 末尾**保持不变**（不重新生成）
- 这样整个生成过程仍然是 **single trajectory** 的 autoregressive decoding，没有 discontinuity

直觉上这相当于：模型 "看一眼" 自己已经画了什么（因为那些 token 已经在 context 里），然后 "想一下" 接下来要画什么（输出 $\tau_k$），再 "画" 下一段（输出 $\mathcal{V}_k$）。这和你在 "State of GPT" 里讲的 "autoregressive decoding as a Markov process with growing context" 完全一致——只是 context 里现在 mixed 了 text 和 image tokens。

### 2.3 How to Refine — 反思修正

公式：

$$c_k = \mathrm{ULM}_u\left(T, \{\tau_j\}_{j \le k}, \{\mathcal{V}_j\}_{j \le k}\right)$$

$$c_k = (r_k, \hat{\tau}_k), \quad r_k \in [0, 100]$$

$$\hat{\mathcal{V}}_k = \mathrm{ULM}_g\left(\{\tau_j\}_{j<k}, \hat{\tau}_k, \{\mathcal{V}_j\}_{j<k}\right) \quad \text{if } r_k < \theta$$

变量解释：
- $c_k$：reflection tuple
- $r_k$：critic score，整数，范围 0-100
- $\hat{\tau}_k$：revised sub-caption
- $\theta$：预设阈值，决定是否触发 local re-generation
- $\hat{\mathcal{V}}_k$：re-generated 的局部 region

5 个 critique criteria（zero-shot 时手工指定）：
1. color accuracy
2. object completeness  
3. detail richness
4. spatial relationships
5. visual coherence

**这个设计的精髓在于 locality**：不是像 Reflect-DiT 那样全图重新生成，而是只 re-generate 当前 $\nu_k$ 这一小块，replace 掉原来的 $\mathcal{V}_k$。这让我想到你在 CS231n 里讲过 "backprop through a small sub-graph" 的效率优势——local re-gen 的成本和 full re-gen 是天壤之别。

Table 1 panel (d) 的 ablation 验证了这一点：

| Reflection | Color | Shape | Texture | Spatial | Non-Spatial | Complex |
|------------|-------|-------|---------|---------|-------------|---------|
| w/o | 73.11 | 41.55 | 64.77 | 21.98 | 30.90 | 48.16 |
| 1-round | **73.90** | **46.02** | **66.10** | **24.50** | 30.81 | **51.97** |
| 2-round | 73.68 | 45.72 | 66.02 | 24.42 | 30.88 | 51.65 |

1-round reflection 几乎在所有维度都有提升（Spatial +2.52，Complex +3.81），但 2-round 没进一步收益——说明 zero-shot ULM 的 critique-and-revise 能力有上限，需要 SFT 或 RL 来 unlock。

---

## 3. 三种实现路径与逐步提升

这是 paper 最有意思的部分——它展示了一个清晰的 **ZS → SFT → RL** 的 progression，每一步都揭示了不同的现象。

### 3.1 TWIG-ZS：Zero-shot 的惊人表现

zero-shot 能 work 这件事本身就很有信息量。这说明 Janus-Pro 这类 ULM 的 pretraining 已经赋予了它 "在 visual generation 中插入 reasoning" 的 latent capability，只需要合适的 prompt 就能 elicit 出来。

Prompt 设计的三个 key point：
- **When to think**：让 model 采取 global view，step by step sketch high-level semantics
- **What to say**：focus 严格在 local region，**explicitly 禁止 spatial-anchor tokens**（比如 "左上角"、"中央"这种词），让模型只产生 descriptive content
- **How to refine**：用 5 个 criteria 强制一个一致的 critic 标准，且 revision 必须是 local 的、不能 contradict 之前的 validated regions

完整的数字对比（vs Janus-Pro-7B baseline）：

| Setting | Color | Shape | Texture | Spatial | Non-Spatial | Complex |
|---------|-------|-------|---------|---------|-------------|---------|
| Janus-Pro-7B | 63.59 | 35.28 | 49.36 | 20.61 | 30.85 | 35.59 |
| TWIG-ZS | 73.11 | 41.55 | 64.77 | 21.98 | 30.90 | 48.16 |
| Δ | **+9.52** | +6.27 | **+15.41** | +1.37 | +0.05 | **+12.57** |

**Texture +15.41 和 Complex +12.57 是 huge jumps**。这其实暗示了一个直觉：texture 和 complex composition 这种需要 "局部精细化" 的能力，最 benefit from on-the-fly 的 local guidance，而 spatial/non-spatial 这种关系类任务因为涉及全局布局，zero-shot 的提升就小很多。

### 3.2 TWIG-SFT：用 TWIG-50K 让模型学会结构化

作者把整个 TWIG 过程 decompose 成 **9 个 supervised tasks**：
- 3 个 thinking targets（upper / central / lower thoughts，对应 $\mathrm{ULM}_u$）
- 3 个 reflection targets（3 个 scores + revised thoughts，对应 $\mathrm{ULM}_u$）
- 3 个 generation targets（3 个 visual regions，对应 $\mathrm{ULM}_g$）

**TWIG-50K dataset 的构造**（这里我展开一点，因为这是 paper 里信息量很密的一段）：

| Subset | Size | Source | Pipeline |
|--------|------|--------|----------|
| What-to-say | ~17K | 5.5K prompts from T2I-CompBench train split | GPT-4o 生成 3-part sub-captions → GPT-4o-Image 合成图像 → quality filter → interleaved format |
| How-to-refine | ~17K | 基于 above interleaved samples | GPT-4o 按 5 criteria 评分 + 给出 revised sub-caption |
| Generation | ~16K | image-sub-caption pairs | 构造 $\mathcal{V}_k = \mathrm{ULM}_g(\{\tau_j\}_{j \le k}, \{\mathcal{V}_j\}_{j<k})$ 的训练样本 |

注意一个细节：**仍然保持 text-to-image 监督**（不是 image-to-image），通过 visual pre-context 来 preserve single trajectory。这是为了让 inference 时不需要 I2I 能力。

TWIG-SFT 的结果：

| Setting | Color | Shape | Texture | Spatial | Non-Spatial | Complex |
|---------|-------|-------|---------|---------|-------------|---------|
| TWIG-ZS | 73.11 | 41.55 | 64.77 | 21.98 | 30.90 | 48.16 |
| TWIG-SFT | 74.58 | 52.42 | 67.95 | 27.02 | 31.24 | 53.41 |
| Δ | +1.47 | **+10.87** | +3.18 | **+5.04** | +0.34 | +5.25 |

**Shape +10.87 和 Spatial +5.04 是 SFT 主要贡献的地方**——这些是 zero-shot prompt 难以稳定 elicit 的能力，需要 data-driven 的 supervision 来 lock in。

Data composition ablation（Table 2a）揭示了一个**反直觉**的发现：

| Mixture | Color | Shape | Texture | Spatial | Non-Spatial | Complex |
|---------|-------|-------|---------|---------|-------------|---------|
| Think-heavy | 73.38 | 50.92 | 66.47 | 26.08 | 30.97 | 51.86 |
| Gen-heavy | 74.12 | 51.77 | 67.28 | 26.58 | 31.09 | 52.83 |
| Think-Gen-equal | **74.58** | **52.42** | **67.95** | **27.02** | **31.24** | **53.41** |
| Reflect-lite | 72.76 | 49.75 | 65.93 | 26.36 | 30.92 | 51.17 |
| Reflect-heavy | 71.88 | 48.98 | 65.05 | 25.62 | 30.84 | 50.27 |

**加入 reflection 数据反而下降**。作者的解释是：TWIG-ZS 已经把 reflection 能力 expose 得差不多了，硬塞 R 数据会让 thoughts 变长、over-correction 频繁，反而削弱 T 和 G 的稳定学习。这其实呼应了 RL 里的 "reward hacking"——监督信号过强会扭曲 policy。

Stability（Table 2b）：

| Model | Color std | Shape std | Texture std | Spatial std | Non-Spatial std | Complex std |
|-------|-----------|-----------|-------------|-------------|-----------------|-------------|
| TWIG-ZS | 0.82 | 0.70 | 0.76 | 0.45 | 0.38 | 0.91 |
| TWIG-SFT | **0.65** | **0.59** | **0.61** | **0.40** | **0.36** | **0.80** |

SFT 把所有维度的 variance 都降下来了——这是 SFT 真正的价值，**不是 peak performance，而是 predictability**。这点和你常说的 "pretraining gives you the world model, SFT gives you the behavior distribution" 完全吻合。

### 3.3 TWIG-RL：TWIG-GRPO 的 joint reinforcement

这是 paper 里最有意思的算法部分。基于 DeepSeekMath 的 GRPO [41]（https://arxiv.org/abs/2402.03300）。

GRPO 的核心：对一个 prompt 采样 $G$ 个 responses，用 group-relative advantage 替代 PPO 的 critic：

$$A_i = \frac{r_i - \mathrm{mean}(\{r_j\}_{j=1}^G)}{\mathrm{std}(\{r_j\}_{j=1}^G)}$$

不需要训 value network，非常 memory-efficient。

**TWIG 的关键设计选择**：在一个 rollout 里有 up to 9 个 local visual subtasks（3 think + 3 reflect + 3 gen），**是分别 reinforce 还是 joint reinforce**？作者选择 **joint**：用一个 shared reward（基于 final image 和 input prompt）同时更新所有 pass 的 policy。

这背后的 intuition 是：interleaved reasoning 是一个 **sequential decision process**，每一步的局部 reward 很难定义，但 final image 的 quality 是 well-defined 的。用一个 scalar reward back-propagate 到整个 trajectory，本质上是把 visual generation 当成一个 **multi-step MDP** 来处理。这和 AlphaGo 的 "用一个 final win/loss reward 训练整个 game trajectory" 是同一个思路，也和 RLHF 里 "用 final human preference 训练整个 response" 是一回事。

Ablation（Table 3a）：

| Strategy | Color | Shape | Texture | Spatial | Non-Spatial | Complex |
|----------|-------|-------|---------|---------|-------------|---------|
| ULM_g-GRPO (only gen) | 80.12 | 59.87 | 72.01 | 32.47 | 31.30 | 54.02 |
| ULM-GRPO (only understanding) | 78.36 | 57.94 | 70.68 | 30.93 | 31.27 | 53.76 |
| TWIG-GRPO (joint) | **82.49** | **61.28** | **73.19** | **34.06** | **31.99** | **54.45** |

Joint 显著好——这说明 thinking 和 generation 是 **mutually reinforcing** 的，分开学两边都学不全。这个发现其实和最近 LLM reasoning 里的 "CoT + outcome reward" 比 "process reward" 更 work 的现象是同构的。

**Reward design**（Table 3b）也很有启发性，用了 4 个 reward model 的 unweighted average：

| Reward | Color | Shape | Texture | Spatial | Non-Spatial | Complex |
|--------|-------|-------|---------|---------|-------------|---------|
| HPS v2 only | 79.83 | 60.97 | 71.35 | 20.68 | 30.53 | 52.87 |
| + GroundingDINO | 80.44 | 60.01 | 73.79 | 25.84 | 31.15 | 54.03 |
| ++ GIT (VQA) | 80.87 | 59.29 | 74.26 | 30.05 | 31.41 | 53.64 |
| +++ LMM Alignment | **82.49** | **61.28** | 73.19 | **34.06** | **31.99** | **54.45** |

每个 reward 贡献不同维度：
- **HPS v2** [51] (https://arxiv.org/abs/2306.09301): global aesthetics 和 style coherence
- **GroundingDINO** [31] (https://arxiv.org/abs/2303.05499): entity presence 和 localization → 主要 boost Spatial
- **GIT** [48] (https://arxiv.org/abs/2205.14100): VQA consistency → curb instruction violation
- **Fine-tuned ORM** [17] (https://arxiv.org/abs/2501.13926): holistic text-image alignment

Unweighted average 看起来很简单，但其实是 anti-reward-hacking 的标准操作——你和你学生时代讨论过的 "multi-objective RL 用 Pareto frontier 而不是 weighted sum" 在工程上经常妥协成 simple average，因为 weighted sum 需要 tuning weights，且容易 overfit to one reward。

### 3.4 最终对比 T2I-CompBench++（Table 4）

| Model | Color | Shape | Texture | 2D-Spatial | 3D-Spatial | Non-Spatial | Numeracy | Complex |
|-------|-------|-------|---------|------------|------------|-------------|----------|---------|
| SDXL | 58.79 | 46.87 | 52.99 | 21.31 | 35.66 | 31.19 | 49.91 | 32.37 |
| PixArt-α | 66.90 | 49.27 | 64.77 | 20.64 | - | 31.97 | - | 34.33 |
| FLUX.1 | 74.07 | 57.18 | 69.22 | 28.63 | 38.66 | 31.27 | 61.85 | 37.03 |
| Emu3 | 75.44 | 57.06 | 71.64 | - | - | - | - | - |
| T2I-R1 | 81.30 | 58.52 | 72.43 | 33.78 | - | 30.90 | 60.97 | 39.93 |
| **TWIG-RL** | **82.49** | **61.28** | **73.19** | **34.06** | **38.87** | **31.99** | **61.93** | **53.56** |

TWIG-RL 在 7 个维度上达到 SOTA（除了 Non-Spatial 和 T2I-R1 持平），**Complex score 53.56 比 T2I-R1 的 39.93 高出 13.63 分**——这是很显著的 gap。Complex prompt 测的就是 long-horizon composition 和 nuanced instruction adherence，正好是 TWIG 设计来解决的 problem。

---

## 4. 我的几个直觉与扩展思考

### 4.1 为什么 K=3 uniform 这么强？

这让我想到视觉的 spatial prior。CNN 的 receptive field hierarchy、ViT 的 patch processing、Diffusion 的 coarse-to-fine denoising——所有这些都在隐式地利用 "natural images 有 multi-scale 结构" 这个 prior。K=3 uniform partition（upper/central/lower）其实是一个 coarsened version of this prior，且和人类 sketching 的顺序（先背景、再主体、再前景细节）对齐。**Adaptive schedule 失败不是因为 idea 不好，而是 because current ULMs 的 planning 能力还不足以 generate well-structured schedule**——这暗示了未来需要专门的 schedule RL。

### 4.2 Joint GRPO 为什么比 separate 好？

我觉得这背后是 **credit assignment 的不同 scale**。Separate GRPO 时，每个 subtask 的 advantage 是局部计算的，但 visual quality 是 global emergent property，局部的 reward signal 很弱。Joint GRPO 用一个 final scalar reward backpropagate 整个 trajectory，相当于让所有 subtask 共享一个 "global value" 信号。这和 HER (Hindsight Experience Replay) 的哲学一样：**当 reward sparse 时，用 final outcome 来 credit assignment 整个 trajectory**。

### 4.3 与 LLM reasoning 的类比

我觉得这篇 paper 其实是 LLM reasoning 里的 "thinking while speaking" vs "think before speak" 在 visual modality 的对照实验。在 LLM 里：
- **Think-before**：CoT prompt 让 model 先 reasoning 再 answer（经典 CoT）
- **Think-after**：Self-Refine / Reflexion 让 model 输出后 critique 再修正
- **Think-while**：在你的 micrograd / llm.c 直觉里其实就是 "token-by-token 的 hidden state 演化"——但 LLM 的 inference 是 dense 的，每个 token 都在 "think"

Visual generation 之前没有这种 dense 的 thinking，因为 pixel/token 生成没有 explicit reasoning 信号。TWIG 把 textual thought 强行 inject 到 visual token stream 中，相当于给 visual generation 加了一个 "reasoning scaffold"。

### 4.4 与 recent concurrent works 的区别

Paper 里提到了 IRG [21] (https://arxiv.org/abs/2509.06945) 和 Uni-CoT [36] (https://arxiv.org/abs/2508.05606)，它们也 "interleave"，但本质还是 think-before + think-after 的组合，把 synthesis 视为 monolithic block。真正的 interleaving 应该是：**生成 N 个 token → think → 生成 N 个 token → think → ...**，在 token level 而不是 region level。TWIG 现在是 region level（K=3），未来 token-level 的 TWIG 才是终态。

### 4.5 一个更大的 picture

我觉得 TWIG 指向一个更深的趋势：**unified generative model = unified reasoning model**。Janus-Pro、Show-o、Emu3、Transfusion [60] 已经把 understanding 和 generation 统一到一个 backbone。下一步是把 **reasoning** 也 unified 进去——不是把 reasoning 作为一个外挂 module，而是让 reasoning 自然 emerge 在 generation trajectory 里。

这个方向的极端版本就是 OpenAI o3 [33] 那种 "thinking with images" 的反演：o3 在 reasoning 时 generate images 作为 scratchpad；TWIG 在 generating images 时 reason 作为 scratchpad。**两个方向 converge 到同一个 endpoint：modality-agnostic 的 reasoning-generation loop**。

### 4.6 Limitations 和未来方向

Paper 自己承认的：
1. **Fixed K=3 schedule** 不是 optimal，adaptive schedule 需要 RL 学
2. 用 original GRPO，没试 DAPO [55] (https://arxiv.org/abs/2503.14476) 或 GSPO [58] (https://arxiv.org/abs/2507.18071) 等 variants
3. 没扩展到 video、3D、I2I

我额外想加的：
4. **Token-level TWIG**：现在是 region-level（K=3），token-level 的话每个 patch 都可以 think，但需要 sparse thinking 机制避免 cost 爆炸
5. **Continuous diffusion 上的 TWIG**：现在只在 autoregressive 上做，continuous diffusion 的 denoising step 间 interleave text 是 unexplored
6. **Reward model 的 hierarchy**：现在 4 个 reward 是平权的，未来应该有 coarse-to-fine 的 reward hierarchy 对应不同 reasoning step
7. **Curriculum**：现在 ZS → SFT → RL 是 linear pipeline，未来可以做 iterative 的 self-improvement loop

---

## 5. 总结

TWIG 这篇 paper 真正的贡献在我看来是三点：

1. **概念上**：第一次明确提出 "interleave text into visual generation trajectory" 这个 paradigm，并把它和 think-before / think-after 区分清楚
2. **工程上**：用 Janus-Pro 这种 single ULM 实现，避免了 pipeline coupling 的复杂性，且 ZS → SFT → RL 的三段式 progression 给了一个可复制的 recipe
3. **实验上**：在 T2I-CompBench(++) 上 beat 了 FLUX.1、T2I-R1 等 strong baseline，证明 paradigm 有效

但我觉得最大的价值是它**打开了一个新 axis**：visual generation 的 reasoning granularity。以前我们只在 "model size" 和 "data size" 上 scaling，现在多了一个 "reasoning depth" 的 axis。K 从 3 到 10 到 100，每一步都是更 fine-grained 的 thinking-while-generating。这条 axis 如果 work，可能比单纯 scale model 还要 impactful——就像 LLM 里 test-time compute scaling 比 pretraining scaling 更 cost-effective 一样。

希望这个解读对你 build intuition 有帮助，Andrej。如果你对 token-level TWIG 或 continuous diffusion 上的 TWIG 这两个方向有想法，我很想听听你的看法。
