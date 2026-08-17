---
source_pdf: CoT-VLA.pdf
paper_sha256: 20081e8c7efca4e4bf7f8d6e83ee69793a30961b5069975c0980dddb2cf211f0
processed_at: '2026-08-03T17:37:50-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 CoT-VLA

## 一句话说清楚

现在主流的 VLA model（比如 [OpenVLA](https://arxiv.org/abs/2406.09246)）就是给一张图 + 一句话，直接吐 action——本质上是个**条件反射**，没有"想一下再动手"的能力。这篇 paper 说：让 robot 先在脑子里**画一张未来某一帧的图**（subgoal image），然后照着这张图去 act。这就是所谓的 visual chain-of-thought——只不过 CoT 不是文字，是一张图片。

这玩意的核心 trick 在于：**subgoal image 的训练数据天然就在 video 里**。你不需要额外标 keypoint、bbox、trajectory，任何有 frame sequence 的 video 都能用来训"想象未来"这个能力。所以作者把 EPIC-KITCHENS、Something-Something V2 这种没 action label 的 human video 也塞进 pretraining 里——这是 OpenVLA 做不到的，因为它只能用有 action 的 robot demo。

---

## 这事为什么有意思

### 先讲讲 VLA 现在的尴尬

OpenVLA 这种 model 训练流程是：拿一个 pretrained VLM（懂图懂文字），fine-tune 它让它能输出 robot action。问题在于，VLM 本身是一个 **understanding model**，它只会"看图说话"，不会"看图想未来"。你让它直接从 $(s_t, l) \to a_t$，它学到的就是一个非常复杂的非线性 mapping，但这个 mapping 内部没有 explicit 的 "planning" 结构。

类比一下：你让一个人闭着眼睛抓桌上的杯子。一种做法是训练一个"视觉→肌肉指令"的端到端网络；另一种做法是让他先在脑子里想"杯子在哪儿、我的手要往哪儿伸、伸到之后要怎么样"，然后再 act。后者明显更 robust，因为它把问题分解了。

[Wei et al. 2022](https://arxiv.org/abs/2201.11903) 在 LLM 里证明了 CoT 有用——先写 reasoning 再写答案。那 robotics 的 CoT 应该长什么样？之前的工作试过：
- [EmbodiedGPT](https://arxiv.org/abs/2312.11085)：生成文字 plan
- [ECoT](https://arxiv.org/abs/2407.08693)：生成 bbox + rationale
- [ReKeP](https://arxiv.org/abs/2409.01652)：生成 keypoint constraint

这些都 work，但都需要额外的 annotation pipeline。CoT-VLA 的 insight 是：**video frame 本身就是最天然的 intermediate reasoning state**，不需要标任何额外东西。

### Subgoal image 作为 CoT 的另一个好处

它解锁了**无 action 标签的 video 数据**。OpenX 这种 robot dataset 撑死也就几十万条 trajectory，但 YouTube 上有海量的人类操作视频、ego-centric video。这些 video 没 action label，没法直接训 policy——但能训"从当前 frame + instruction 想象 future frame"。这个能力学会之后，就 transfer 到 robot policy 里当 reasoning step 用。

这跟 [Gen2act](https://arxiv.org/abs/2409.16283)、[GR-2](https://arxiv.org/abs/2410.06158)、[Video as the new language](https://arxiv.org/abs/2402.17139) 这些 work 的哲学是同源的：**video generation 就是 implicit world model**，能 predict future 就能 plan。

---

## 方法细节，但讲人话

### 两阶段的数学描述

Vanilla VLA：
$$\hat{\mathbf{a}}_t \sim P_\theta(\mathbf{a}_t | \mathbf{s}_t, l)$$

意思就是：在当前观察 $\mathbf{s}_t$ 和指令 $l$ 条件下，直接 sample 一个 action。

CoT-VLA 拆成两步：
$$\hat{\mathbf{s}}_{t+n} \sim P_\theta(\mathbf{s}_{t+n} | \mathbf{s}_t, l)$$
$$\{\hat{\mathbf{a}}_t, ..., \hat{\mathbf{a}}_{t+m}\} \sim P_\theta(\{\mathbf{a}_t, ..., \mathbf{a}_{t+m}\} | \mathbf{s}_t, l, \mathbf{s}_{t+n})$$

翻译成人话：
- 第一步：给我现在的画面 + 指令，让我想象未来第 $n$ 帧画面长啥样
- 第二步：给我现在的画面 + 指令 + 刚想象的未来画面，输出接下来 $m+1$ 步的 action

变量含义：
- $n$：subgoal 的"距离"——想象多远的未来。Bridge 数据集用 5-10 帧，TOTO 用 20-24 帧，因为不同 dataset frame rate 不同
- $m$：action chunk size，paper 里固定用 10
- $\hat{\mathbf{s}}_{t+n}$：生成的 subgoal image
- $\hat{\mathbf{a}}_t, ..., \hat{\mathbf{a}}_{t+m}$：长 m+1 的 action 序列

**关键点**：第二步同时 condition 在 $\mathbf{s}_t$ 和 $\hat{\mathbf{s}}_{t+n}$ 上。这意味着 policy 不是盲目"追目标"，而是知道"我现在在哪、我要去哪"——这跟 [Goal-Conditioned Imitation Learning](https://arxiv.org/abs/1903.08154) 的思路一脉相承，只不过 goal 是 model 自己 imagine 出来的。

### 为什么选 VILA-U 当 base model

这是整篇 paper 最关键的设计选择。绝大多数 VLA（OpenVLA、RT-2、PaLM-E）都 build 在**只能 understand 不能 generate** 的 VLM 上。你让它们输出一张图，它们做不到。

[VILA-U](https://arxiv.org/abs/2409.04429) 是个 unified multimodal model，既能看图说话，也能听话画图。技术上它用 **Residual Quantization (RQ-VAE)** 把 image 编码成 discrete token：

每张 256×256 的图被 encode 成 $16 \times 16 \times 4$ 个 token。意思是空间上 16×16=256 个位置，每个位置用 4 层 codebook entry 叠加表达。这 4 层 code 是通过一个 depth transformer autoregressively 预测的：

$$\mathcal{L}_{\text{visual}} = -\sum_j \sum_{d=1}^{D} \log P_\delta(k_{jd} | k_{j,<d})$$

变量含义：
- $j$：spatial position index（0 到 255）
- $d$：residual code 的 depth，从 1 到 D=4
- $k_{jd}$：第 j 个位置的第 d 层 code
- $k_{j,<d}$：第 j 个位置前 d-1 层 code（context）
- $P_\delta$：depth transformer 的参数

为什么用 residual 而不是 single-layer VQ？因为单 codebook 表达能力有限，多层 residual 能保留更多 visual detail。这很重要——[Table 3 的实验](https://cot-vla.github.io/)显示，把 generated goal image 换成 ground-truth goal image，success rate 能提升 40%。也就是说**视觉细节的 quality 直接决定 action 的 quality**。

### Hybrid attention：一个看起来细节但很关键的设计

看 Figure 3。CoT-VLA 在同一个 transformer 里用了两种 attention：

- **Image / text token**：用 causal attention（每个 token 只能看前面的，标准 next-token prediction）
- **Action token**：用 full attention（所有 action token 互相都能看到）

为什么 action 要 full attention？想想 action 是什么——7 维（xyz translation + rpy + gripper）× 10 步 chunk = 70 个 token。这 70 个 token 在物理上是高度耦合的：gripper close 的时机必须跟 z 轴下降同步，x y 移动必须跟最终 target 位置对齐。如果用 causal attention 让它一个个 token 顺序生成，前面的 token 看不到后面的，就会出现"先 commit 了 gripper open 然后发现 z 还没降下来"这种 inconsistency。

Full attention 等于把这 70 个 action token 当成一个集合同时 decode，类似 [Non-Autoregressive NMT](https://arxiv.org/abs/1711.03576) 的思路。[Figure 6 的 ablation](https://cot-vla.github.io/) 也证明了：单加 action chunking（causal attention）有提升，再换成 hybrid attention（full for action）又再提几个点。

### Action chunking 的 intuition

[Diffusion Policy](https://arxiv.org/abs/2303.04137)、[ACT](https://arxiv.org/abs/2304.13705) 都验证过 action chunking 有用。三个原因：

1. **减少 error accumulation**：每 10 步才重新观察世界一次，policy 网络的 inference 次数少了，compounding error 就小
2. **建模多模态 distribution**：人类 demo 里 action 经常是 multimodal 的（比如"先往左拿起来" vs "先往右拿起来"），单步预测会被 multimodality 弄糊涂，chunk prediction 能隐式表达 mode
3. **Inference 效率**：10 步一次 forward pass，吞吐量高 10 倍

但 chunking 有个副作用：chunk 之间可能出现 discontinuity（chunk 结尾的速度跟下一个 chunk 开头的速度不连续），paper 在 limitation 里也承认了这点。

---

## 实验结果，挑重点说

### LIBERO 上的结果（Table 1）

| Method | Avg | Spatial | Object | Goal | Long |
|---|---|---|---|---|---|
| Diffusion Policy | 72.4 | 78.3 | **92.5** | 68.3 | 50.5 |
| Octo | 75.1 | 78.9 | 85.7 | 84.6 | 51.1 |
| OpenVLA | 76.5 | 84.7 | 88.4 | 79.2 | 53.7 |
| **CoT-VLA** | **81.1** | **87.5** | 91.6 | **87.6** | **69.0** |

重点看 **Long horizon** 任务：CoT-VLA 比第二名高 15.3 个百分点。这完全 make sense——Long 任务需要多 stage 执行，subgoal image 提供了天然的 "break down" 机制。Spatial 任务上 OpenVLA 经常 "看错"（视觉相似但 instruction 不同），CoT-VLA 通过先 generate 一个 language-grounded subgoal 避免了这种 visual cue overfit。

### Bridge-V2 上（Table 2）

| Category | SUSIE | Octo | OpenVLA | CoT-VLA |
|---|---|---|---|---|
| Visual | 30 | 35 | **75** | 65 |
| Motion | 10 | 10 | 45 | **60** |
| Semantic | 20 | 0 | 40 | **50** |
| Language | 40 | 40 | **75** | 70 |

CoT-VLA 在 Visual 和 Language 略输 OpenVLA，paper 归因于 action chunking 的 grasping failure。但在 Motion 和 Semantic 上明显领先——这两个 task 正好需要"想象 future motion" 的能力，subgoal image 帮上忙了。

对比 [SUSIE](https://arxiv.org/abs/2310.10639) 很有意思：SUSIE 用 stable diffusion 做 image editing 生成 goal，视觉质量更高，但 success rate 反而最低。这说明 **goal image 的 task-relevance 比 visual quality 重要得多**。

### Table 3 是最关键的实验

|  | Sub-task 1 | Sub-task 2 |
|---|---|---|
| Generated Goal Images | 20% | 0% |
| Ground-truth Goal Images | 60% | 40% |

把 model 自己生成的 subgoal 换成 ground-truth subgoal，success rate 提升 40%。这说明：
1. 当前 model 的 action prediction 能力其实够用，**bottleneck 是 visual reasoning**
2. 如果未来 image generation model 更强（diffusion-based unified model 之类），action performance 会自然受益

这就是这篇 paper 给未来研究埋下的伏笔：**scale visual reasoning, action 自然提升**。

### Pretraining ablation（Figure 6b）

去掉 OpenX + video pretraining，直接 fine-tune VILA-U 在 Franka-Tabletop：53.7% → 加上 pretraining：78.8%。**相对提升 46.7%**。

这个数字非常 dramatic。它说明 foundation model 的 visual reasoning 能力需要大规模数据预热，narrow fine-tune 完全替代不了。这也间接回答了一个问题：为什么不用 OpenVLA + 一个单独的 image generator 拼？因为它们没有 shared representation，没法 transfer visual reasoning 能力。

---

## 跟其他思路的关系，build your intuition

### 跟 Model-based RL 的关系

CoT-VLA 本质上是 **implicit world model + goal-conditioned policy** 的组合：
- Phase 1 = "imagine future" = world model 的一步 rollout
- Phase 2 = "act toward goal" = goal-conditioned policy

这跟 [Dreamer](https://arxiv.org/abs/1912.01603)、[PlaNet](https://arxiv.org/abs/1811.04557) 的哲学一致，但区别在于：
- Dreamer 在 latent space rollout 多步然后做 trajectory optimization
- CoT-VLA 直接在 pixel token space 预测一步 subgoal，跳过 explicit planning loop

可以理解为 "shallow MCTS + closed-loop replan"。计算效率高，planning depth 浅。未来如果在 inference time 加 beam search / multiple subgoal sampling，能往真正的 multi-step planning 靠拢——这跟 [o1-style reasoning](https://openai.com/o1/) 的 "think longer" 是同一类思路。

### 跟 Hindsight Experience Replay 的关系

[HER](https://arxiv.org/abs/1707.01495) 在 RL 里用 future state 当 goal relabel，本质是 "假装我一开始就想去那儿"。CoT-VLA 训练时用真实 future frame 当 subgoal ground truth，思路有点像 HER，但它是把 "predict future" 显式建模成 generation task，而不是 relabeling。

### 跟 AlphaGo 的关系

AlphaGo 用 MCTS 从当前 state rollout 多个 future trajectory。CoT-VLA 等于一个**只有 depth=1 的 MCTS**——只 look ahead 一步到 subgoal，然后 react。如果未来能在 inference time 做 multiple subgoal sampling + scoring，就能逼近真正的 multi-step planning。

### 跟 Unified Multimodal Model trend 的关系

CoT-VLA 选 VILA-U 不是偶然，它押的是 unified multimodal model 这条路：
- [Chameleon](https://arxiv.org/abs/2405.09818)：Meta 的 mixed-modal early-fusion
- [Unified-IO 2](https://arxiv.org/abs/2312.17172)：统一 io
- [Emu3](https://arxiv.org/abs/2409.18869)：next-token prediction is all you need
- [Show-o](https://arxiv.org/abs/2408.12528)：single transformer
- [Transfusion](https://www.arxiv.org/abs/2408.11039)：next token + diffusion

这条路的 thesis 是：**理解 and 生成 share the same underlying representation learning mechanism**，所以应该用同一个 model 同时做。如果这条路 work，CoT-VLA 的 framework 天然可 scale——image generation 质量提升，action performance 就免费提升。

### 跟 Diffusion Policy 的关系

[Diffusion Policy](https://arxiv.org/abs/2303.04137) 用 diffusion 生成 action chunk，handle multi-modal action distribution。CoT-VLA 用 autoregressive next-token prediction + full attention 也能 handle multi-modal（不同 mode 对应不同 token sequence）。区别是 diffusion 在 continuous space，CoT-VLA 在 discrete token space。

Discretize action 的代价是精度损失（256 bin），但好处是能 seamlessly integrate 进 LLM framework，跟 image / text token 共享同一套 transformer。这是一个 architecture alignment 的 trade-off。

---

## Inference 的时候到底在干啥

Algorithm 1 翻译成人话：

```
循环直到任务完成:
    1. 拍一张当前照片 s_t
    2. 让 model 想象未来第 n 帧长啥样 → 生成 subgoal image
    3. 让 model 基于"当前照片 + 指令 + subgoal image"输出 10 个 action
    4. 顺序执行这 10 个 action
    5. 跳回第 1 步，重新观察、重新 imagine
```

这里有个 subtle 的点：每 10 步才重新 imagine 一次。这意味着 subgoal image 不是 per-step refinement，而是 per-chunk guidance。好处是 inference overhead 被 chunk size 摊薄了；坏处是 chunk 期间发生的意外（比如物体被碰飞）model 不会实时反应。

---

## Limitation 和 future

### 1. Inference 慢

要生成 256 个 image token + 70 个 action token，相比 OpenVLA 直接生成 7 个 action token 慢 7×。这是 autoregressive generation 的固有 cost。解决方向：
- [Speculative decoding](https://arxiv.org/abs/2211.17192) 用 small model draft + big model verify
- [Consistency models](https://arxiv.org/abs/2303.01469) 做 few-step generation
- [DC-AE](https://arxiv.org/abs/2410.10733) 把 image 压成更少 token
- [An image is worth 32 tokens](https://arxiv.org/abs/2406.07550) 这种 token reduction 工作

### 2. Image 质量不够

Autoregressive image generation 的 visual quality 目前还追不上 diffusion。未来用 [Emu3](https://arxiv.org/abs/2409.18869)、[Janus](https://arxiv.org/abs/2410.13848) 这种 hybrid 架构（understanding 用 discrete token, generation 用 diffusion decoder）应该能改善。

### 3. Action chunk discontinuity

Chunk 之间可能出现跳跃。Diffusion Policy 用 temporal smoothing 缓解，CoT-VLA 目前没处理。最近的 [RDT-1B](https://arxiv.org/abs/2410.07864) 用 diffusion-based VLA 也在解决这个问题。

### 4. OOD generalization 还不够

Table 3 显示在完全 OOD 的任务上，generated goal image 质量不行，导致 success rate 几乎为 0。这要等 video / world model scaling 上去才能解决——这跟 [Sora](https://openai.com/sora/)、[Pandora](https://arxiv.org/abs/2406.09455) 这些 world model work 的进展是绑定的。

---

## 给你的 intuition

如果让我总结这篇 paper 的 thesis，就一句话：

> **Reasoning 就是内部的 predictive simulation。LLM 的 CoT 是文字 simulation，AlphaGo 的 MCTS 是 game tree simulation，CoT-VLA 的 subgoal image 是 visual simulation。形式不同，本质相同。**

这个观点有几个推论：

1. **CoT 不必是文字**——任何能 predict future state 的 intermediate representation 都可以当 reasoning step
2. **Action prediction 和 world modeling 应该 couple**——分开训容易浪费 capacity，couple 训才能让 visual reasoning 直接服务 action
3. **Discrete token 是 unifying interface**——vision、language、action 都进 token space，统一的 autoregressive framework 才能 scale
4. **Foundation model 的 reasoning 能力靠 scale, fine-tune 只是 surface adaptation**——Figure 6b 的 46.7% 提升就是证据

如果你在思考下一代 agent design：CoT-VLA 给了一个很 actionable 的 recipe——让 model 在 act 之前先在内部"run forward"一个 model，无论是 language、visual、还是 3D representation，形式不重要，**有 explicit intermediate predictive step** 才重要。

---

## Reference Links

- [CoT-VLA 项目主页](https://cot-vla.github.io/)
- [VILA-U paper](https://arxiv.org/abs/2409.04429)
- [OpenVLA paper](https://arxiv.org/abs/2406.09246)
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [Chain-of-Thought prompting (Wei et al.)](https://arxiv.org/abs/2201.11903)
- [LIBERO benchmark](https://arxiv.org/abs/2306.03310)
- [Bridge-V2](https://arxiv.org/abs/2308.12952)
- [RQ-VAE](https://arxiv.org/abs/2203.01941)
- [ACT (Action Chunking Transformers)](https://arxiv.org/abs/2304.13705)
- [Embodied CoT (ECoT)](https://arxiv.org/abs/2407.08693)
- [3D-VLA](https://arxiv.org/abs/2403.09631)
- [Emu3](https://arxiv.org/abs/2409.18869)
- [Show-o](https://arxiv.org/abs/2408.12528)
- [Janus](https://arxiv.org/abs/2410.13848)
- [Transfusion](https://www.arxiv.org/abs/2408.11039)
- [Chameleon](https://arxiv.org/abs/2405.09818)
- [Unified-IO 2](https://arxiv.org/abs/2312.17172)
- [Gen2act](https://arxiv.org/abs/2409.16283)
- [GR-2](https://arxiv.org/abs/2410.06158)
- [SUSIE](https://arxiv.org/abs/2310.10639)
- [Video as the new language](https://arxiv.org/abs/2402.17139)
- [Dreamer V3](https://arxiv.org/abs/1912.01603)
- [HER](https://arxiv.org/abs/1707.01495)
- [Speculative decoding](https://arxiv.org/abs/2211.17192)
- [Consistency models](https://arxiv.org/abs/2303.01469)
- [DC-AE](https://arxiv.org/abs/2410.10733)
- [Token reduction (An image is worth 32 tokens)](https://arxiv.org/abs/2406.07550)
- [Goal-Conditioned Imitation Learning](https://arxiv.org/abs/1903.08154)
- [Non-Autoregressive NMT](https://arxiv.org/abs/1711.03576)
- [PaLM-E](https://arxiv.org/abs/2303.03378)
- [ReKeP](https://arxiv.org/abs/2409.01652)
- [EmbodiedGPT](https://arxiv.org/abs/2312.11085)
- [Pandora world model](https://arxiv.org/abs/2406.09455)
- [RDT-1B](https://arxiv.org/abs/2410.07864)

---

# CoT-VLA: Visual Chain-of-Thought Reasoning for VLA Models 深度解析

这篇paper由NVIDIA、Stanford、MIT合作完成（一作Qingqing Zhao），核心思想是把**visual chain-of-thought reasoning**引入Vision-Language-Action模型，让机器人policy在act之前先"think visually"——具体形式是先autoregressively生成一张未来n帧的subgoal image，再基于当前observation + language instruction + 生成的subgoal image输出一段action chunk。让我一层层剥开来讲，目标是让你建立起对这套系统的intuition。

---

## 1. Motivation：为什么vanilla VLA不够？

### 1.1 现有VLA的问题

现有VLA（OpenVLA、RT-1、PaLM-E、Octo）基本都是直接的input-output mapping：

$$\hat{\mathbf{a}}_t \sim P_\theta(\mathbf{a}_t | \mathbf{s}_t, l)$$

这里 $\mathbf{s}_t$ 是t时刻的visual observation（图像），$l$ 是language instruction，$\hat{\mathbf{a}}_t$ 是预测的action。问题在于：这种mapping没有intermediate reasoning step，相当于一个reflexive policy——你给它一个scene和指令，它直接吐出action，没有"想一想接下来世界会变成什么样"的过程。

在LLM领域，CoT已经被证明能大幅提升reasoning能力（参考[Wei et al. 2022](https://arxiv.org/abs/2201.11903)）。把CoT搬到robotics的方式有几种：
- Language-based planning：生成textual plan再执行（[EmbodiedGPT](https://arxiv.org/abs/2312.11085), [ProgPrompt](https://arxiv.org/abs/2209.11302)）
- Keypoint/bounding box预测（[ReKeP](https://arxiv.org/abs/2409.01652), [ECoT](https://arxiv.org/abs/2407.08693)）
- 3D representation（[3D-VLA](https://arxiv.org/abs/2403.09631)）

但这些intermediate representation需要额外annotation pipeline，不能scalable地利用海量无标注video。

### 1.2 Key insight：subgoal image就是天然的CoT

CoT-VLA的核心insight：robot demonstration数据本身就有subgoal frame，video数据本身也是frame sequence，所以"predict next frame"这个task不需要任何额外annotation。这意味着可以scale到EPIC-KITCHENS-100这种egocentric video dataset，把海量无action-label的video数据利用起来——这是OpenVLA做不到的，因为OpenVLA只能用有action annotation的$D_r$。

这本质上是一种**model-based planning in pixel space**的简化版：与其学一个精确的world dynamics model，不如直接让一个multimodal LLM在token space里autoregressively预测future frame作为goal，然后goal-conditioned policy去完成它。这跟[Yang et al., "Video as the new language for real-world decision making"](https://arxiv.org/abs/2402.17139)的vision一脉相承——video generation本质上就是latent dynamics model。

---

## 2. Method细节

### 2.1 Visual CoT的两阶段formulation

CoT-VLA把policy拆成两个sequential phase：

**Phase 1: Subgoal image generation** (visual reasoning)
$$\hat{\mathbf{s}}_{t+n} \sim P_\theta(\mathbf{s}_{t+n} | \mathbf{s}_t, l)$$

**Phase 2: Action chunk generation** (action execution)
$$\{\hat{\mathbf{a}}_t, ..., \hat{\mathbf{a}}_{t+m}\} \sim P_\theta(\{\mathbf{a}_t, ..., \mathbf{a}_{t+m}\} | \mathbf{s}_t, l, \mathbf{s}_{t+n})$$

变量含义：
- $n$: subgoal prediction horizon，预测未来第n帧
- $m$: action chunk size，paper中设为10
- $\hat{\mathbf{s}}_{t+n}$: 生成的subgoal image
- $\{\hat{\mathbf{a}}_t, ..., \hat{\mathbf{a}}_{t+m}\}$: 长度为m+1的action序列

这里有个细节值得注意：phase 2同时condition在$\mathbf{s}_t$（当前state）和$\hat{\mathbf{s}}_{t+n}$（未来goal）上，意味着policy需要做"从现在到goal"的path planning——这跟[Goal-Conditioned Imitation Learning](https://arxiv.org/abs/1903.08154)的idea是一致的，但这里goal是model自己imagined的而不是给定的。

注意phase 1的训练数据是$D_r \cup D_v$（有action的robot demo + 无action的video），phase 2只用$D_r$。这设计非常聪明：你不需要video有action annotation就能利用它训练"如何想象future"，因为这些video本身就有frame sequence可以学dynamics。

### 2.2 Base model: VILA-U架构

CoT-VLA建立在[VILA-U](https://arxiv.org/abs/2409.04429)上，这是关键选择。VILA-U是个unified multimodal foundation model，能同时做understanding和generation——既能"看图说话"也能"听话画图"。这跟[LlamaGen](https://arxiv.org/abs/2406.06525)、[Emu3](https://arxiv.org/abs/2409.18869)、[Show-o](https://arxiv.org/abs/2408.12528)、[Janus](https://arxiv.org/abs/2410.13848)、[Transfusion](https://www.arxiv.org/abs/2408.11039)这一波"unified next-token prediction for everything"的trend是同步的。

VILA-U的关键技术是**Residual Quantization (RQ-VAE)**：

- 每张256×256的image被encoded成$16 \times 16 \times 4$个token，即空间维度16×16=256个位置，每个位置用4个residual codebook entry组合表示
- 这4个code是通过一个depth transformer autoregressively预测的
- 第j个位置的D个residual code记作$(k_{j1}, ..., k_{jD})$，D=4
- Depth transformer $P_\delta$基于LLM输出的embedding $h_j$ autoregressively预测这D个code

为什么用RQ而不是普通的VQ？因为单个codebook entry表达能力有限，residual quantization通过多层codebook的残差叠加能保留更多visual detail，这对生成subgoal image的质量很重要——而subgoal image的质量直接决定下游action prediction的效果（Table 3证明：用ground-truth goal image代替generated goal image能提升40% success rate）。

### 2.3 Hybrid attention mechanism

这是论文的一个engineering贡献，参考Figure 3。设计动机：

- **Image/text generation**用causal attention（next-token prediction）：因为image和text token是autoregressive生成的，每个token只能看到前面的token
- **Action generation**用full attention：因为action chunk里所有维度（7-DoF × 10 steps = 70 tokens）需要mutually consistent，互相都要看到彼此

具体来说，每个action $\mathbf{a}_i$被表示成7个token（对应7-DoF），每个维度被independently discretize成256个bin，bin的边界由训练数据action distribution的1st和99th percentile确定。256个action bin token复用了text tokenizer里最少用的256个token。

训练loss：

**Visual loss** (Equation 4):
$$\mathcal{L}_{\text{visual}} = -\sum_j \sum_{d=1}^{D} \log P_\delta(k_{jd} | k_{j,<d})$$

- $j$: visual token的spatial position index
- $d$: residual code的depth index，从1到D=4
- $k_{jd}$: 第j个位置的第d层residual code
- $k_{j,<d}$: 第j个位置前d-1层residual code（context）
- $P_\delta$: depth transformer的参数

**Action loss** (Equation 5):
$$\mathcal{L}_{\text{action}} = -\sum_{i=1}^{m} \log P_\theta(\mathbf{a}_t ... \mathbf{a}_{t+m} | l, s_t, s_{t+n})$$

- $i$: action chunk内的index，从1到m=10
- $P_\theta$: 整个VLA模型的参数
- 条件信息包括language instruction $l$、当前state $s_t$、subgoal $s_{t+n}$

**Total loss**: $\mathcal{L} = \mathcal{L}_{\text{action}} + \mathcal{L}_{\text{visual}}$

### 2.4 为什么full attention对action重要？

想想vanilla OpenVLA用causal attention预测action，这意味着action dimension 0预测的时候看不到dimension 1, 2, ..., 6。但robot action的维度（x, y, z translation + roll, pitch, yaw + gripper）在物理上是高度耦合的——比如grasp的时候gripper close跟z轴下降必须同步。Hybrid attention让所有action token在生成时互相能看到，这相当于把action chunk prediction从autoregressive decoding变成了non-autoregressive decoding（类似[Non-Autoregressive Neural Machine Translation](https://arxiv.org/abs/1711.03576)的思路）。

参考Figure 6的ablation：单纯加action chunking（用causal attention）确实有提升，但再加hybrid attention（full attention for action）又进一步提升了几个点。这说明action内部的consistency是个实在的bottleneck。

### 2.5 Action chunking的intuition

[ACT (Action Chunking with Transformers)](https://arxiv.org/abs/2304.13705)、[Diffusion Policy](https://arxiv.org/abs/2303.04137)都验证过action chunking的好处。原因：
1. **降低error accumulation**：每m步才重新观察一次世界，相比每步都decide，减少了policy网络多次inference带来的compounding error
2. **建模temporal structure**：人类demonstration里action往往有smooth的multi-modal structure（比如"reach → grasp → lift"），单步预测会被这个multimodality弄糊涂
3. **Inference efficiency**：m步一次forward pass，吞吐量高m倍

但chunking也有代价：chunk之间可能出现discontinuity，且执行过程中没有高频feedback（paper在limitation里讲了这点）。

---

## 3. Training Pipeline

### 3.1 Two-stage training

**Pretraining stage**:
- 数据：OpenX subset（Bridge 24.14%, RT-1 6.90%, TOTO 10.34%, VIOLA 10.34%, RoboTurk 10.34%, Jaco Play 10.34%, Berkeley UR5 10.34%, Fanuc 10.34%）+ Something2Something V2 3.45% + EPIC-KITCHENS-100 3.45%
- 可训组件：LLM backbone、projector、depth transformer（vision tower frozen）
- Batch size: 2048, LR: 1e-4 with cosine decay, 10 epochs
- 总训练成本：11K A100 GPU hours on 96 GPUs
- Image resolution: 256×256
- Subgoal horizon: dataset-specific，比如Bridge用[5, 10]，TOTO用[20, 24]，EPIC-KITCHENS用[5, 7]
- Action chunk size: 10

注意每个dataset有不同的 $[n_l, n_u]$ 范围，这是为了适应不同dataset里frame rate和task duration的差异。Bridge的frame rate高、动作短，所以预测5-10帧；TOTO动作长所以预测20-24帧。

**Adaptation stage**:
- 在target robot setup上用少量demo（10-150 trajectories for Franka-Tabletop）fine-tune
- 优化同样三个组件（LLM + projector + depth transformer）
- LR: 1e-5 constant，150 epochs

### 3.2 Test-time closed-loop inference

Algorithm 1展示了一个闭环控制循环：
```
while not done:
    1. sample subgoal image ŝ_{t+n} ~ P(s_{t+n} | l, s_t^obs)
    2. sample action chunk [â_t, ..., â_{t+m}] ~ P(a_t...a_{t+m} | l, s_t^obs, s_{t+n})
    3. for j = 0 to m:
           execute â_{t+j}
    4. observe new state s_{t+m+1}^obs
    5. t ← t + m + 1
```

这里关键：每m+1步才重新visual reasoning一次，所以visual CoT的overhead被action chunking摊薄了（虽然paper里说还是有7× slowdown）。

---

## 4. Experiments深度解析

### 4.1 LIBERO benchmark (Table 1)

LIBERO有4个task suite：Spatial、Object、Goal、Long。每个suite 10 tasks × 50 demos × 3 seeds × 500 episodes = 评估很扎实。

| Method | Avg | Spatial | Object | Goal | Long |
|---|---|---|---|---|---|
| Diffusion Policy | 72.4 | 78.3 | 92.5 | 68.3 | 50.5 |
| Octo fine-tuned | 75.1 | 78.9 | 85.7 | 84.6 | 51.1 |
| OpenVLA fine-tuned | 76.5 | 84.7 | 88.4 | 79.2 | 53.7 |
| **CoT-VLA-7B** | **81.13** | 87.5 | **91.6** | **87.6** | **69.0** |

观察：
- **Long任务提升最大**（+15.3%）：LIBERO-Long是多stage任务，正好需要subgoal image来break down长horizon
- **Spatial任务** baseline有时候会"看错"——visual cue相似但language instruction不同，CoT-VLA通过先生成language-grounded的subgoal避免了这种visual cue overfit
- Diffusion Policy在Object任务（92.5%）反而最高，因为它对narrow distribution的 visuomotor mapping拟合得最好，但跨任务泛化差

### 4.2 Bridge-V2 (Table 2)

| Category | SUSIE | Octo | OpenVLA | CoT-VLA |
|---|---|---|---|---|
| Visual | 30% | 35% | 75% | 65% |
| Motion | 10% | 10% | 45% | 60% |
| Semantic | 20% | 0% | 40% | 50% |
| Language | 40% | 40% | 75% | 70% |

CoT-VLA在Visual和Language上略低于OpenVLA，paper归因于action chunking导致的grasping failure（chunk之间的不连续性）。但Motion和Semantic上有明显提升——这正好印证了CoT的intuition：semantic generalization和motion generalization需要"想象未来"的能力，subgoal image正好填补了这个gap。

对比SUSIE（two-stage: image editing for goal + goal-conditioned policy）：SUSIE生成的goal image视觉质量更高（因为用了stable diffusion），但success rate低很多——说明image quality不等于task usefulness。这跟"教biologist画细胞示意图 vs 教学生用自然语言描述细胞结构"的区别类似，前者更精细但后者更task-relevant。

### 4.3 Franka-Tabletop (Figure 4)

6个task：3 single-instruction + 3 multi-instruction。每个task只10-150 demos。这里CoT-VLA平均78.8%，比第二名高很多。Diffusion Policy在single-instruction narrow domain上能赢，因为可以从scratch学一个low-data regime的policy；但multi-instruction需要language grounding和generalization，pretrained model有优势。

### 4.4 Ablation: Better visual reasoning helps (Table 3)

这是最能说明visual CoT value的实验：

| | Sub-task 1 | Sub-task 2 |
|---|---|---|
| Generated Goal Images | 20% | 0% |
| Ground-truth Goal Images | 60% | 40% |

用GT goal image代替generated goal image能提升40% absolute success rate。这有两个interpretation：
1. **Visual reasoning是bottleneck**：当前autoregressive image generation还不足以生成OOD subgoal
2. **Action prediction本身是有能力的**：只要给它正确的goal，action prediction能work

这暗示了未来的研究方向：scale up visual reasoning capability（用diffusion-based generation或者larger video model），action prediction自然会受益。这跟paper Section 5展望的"video generation as world model"的future一致。

### 4.5 Pretraining ablation (Figure 6b)

去掉pretraining直接fine-tune VILA-U在Franka-Tabletop上：53.7% → 加上pretraining：78.8%，46.7% relative improvement。这印证了：foundation model的视觉reasoning能力需要大规模robot+video数据预热，narrow fine-tune无法替代。

---

## 5. Limitations与未来方向

### 5.1 Inference cost

生成256个image token + 70个action token，比OpenVLA直接生成7个action token慢7×。这是autoregressive generation的固有cost。潜在解决方向：
- [Speculative decoding](https://arxiv.org/abs/2211.17192)
- [Consistency models](https://arxiv.org/abs/2303.01469) for fast generation
- [DC-AE](https://arxiv.org/abs/2410.10733)把image压缩到更少token
- [Token reduction](https://arxiv.org/abs/2406.07550)：an image is worth 32 tokens

### 5.2 Image quality

Autoregressive generation的visual quality不如diffusion models（比如[Emu3](https://arxiv.org/abs/2409.18869), [Show-o](https://arxiv.org/abs/2408.12528), [Janus](https://arxiv.org/abs/2410.13848), [Transfusion](https://www.arxiv.org/abs/2408.11039)都在尝试弥合这个gap）。未来可以换成hybrid架构：understanding部分用discrete token，generation部分用diffusion decoder。

### 5.3 Action chunking的discontinuity

Chunk之间可能出现跳跃。解决方法：
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)用了temporal smoothing
- Per-step prediction with overlapping chunks
- 最近的[RDT-1B](https://arxiv.org/abs/2410.07864)等diffusion-based VLA

---

## 6. Broader context与intuition building

### 6.1 跟Model-based RL的connection

CoT-VLA本质上是一个**implicit world model + goal-conditioned policy**的组合：
- Phase 1 (subgoal generation) = world model的"imagine future"
- Phase 2 (action prediction) = planning/control

这跟[Dreamer](https://arxiv.org/abs/1912.01603)、[PlaNet](https://arxiv.org/abs/1811.04557)、[DayDreamer](https://arxiv.org/abs/2206.14176)的philosophy一致，但区别是：
- Dreamer在latent space学dynamics然后做planning，需要rollout多步
- CoT-VLA直接在pixel token space预测一步subgoal，跳过了explicit planning loop

可以理解为"single-step latent imagination + closed-loop re-plan"，这是计算效率vs planning depth的trade-off。

### 6.2 跟Hindsight Experience Replay的connection

Phase 1训练时用的是真实future frame作为subgoal ground truth，这跟[HER](https://arxiv.org/abs/1707.01495)的idea有相似性——都把future state当作goal来训练。但CoT-VLA不是goal relabeling，而是把"predict future"显式建模成generation task。

### 6.3 跟AlphaGo的look-ahead search

AlphaGo做MCTS：从当前state rollout多个future trajectory选最优。CoT-VLA相当于一个**very shallow MCTS**——只look ahead一步到subgoal，然后react。如果未来能在inference time做beam search或者multiple subgoal sampling + scoring，可能能逼近真正的multi-step planning。这跟[s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393)、[OpenAI o1-style reasoning](https://openai.com/o1/)的"think longer"思路是同一类思路的different manifestation。

### 6.4 跟Embodied foundation model trend

CoT-VLA在[VILA-U](https://arxiv.org/abs/2409.04429)上build，跟一波unified multimodal model trend同步：
- [Chameleon](https://arxiv.org/abs/2405.09818)：Meta的mixed-modal early-fusion
- [Unified-IO 2](https://arxiv.org/abs/2312.17172)：统一io for vision/language/audio/action
- [Emu3](https://arxiv.org/abs/2409.18869)：next-token prediction is all you need
- [Show-o](https://arxiv.org/abs/2408.12528)：single transformer for understanding + generation
- [Transfusion](https://www.arxiv.org/abs/2408.11039)：next token + diffusion在一个model

CoT-VLA的insight是：**unified understanding-generation model天然适合做visual CoT**，因为它能generate未来frame作为reasoning step，而不需要单独搭一个goal image generator。

### 6.5 跟OpenVLA的对比

[OpenVLA](https://arxiv.org/abs/2406.09246)是在Prismatic VLM上fine-tune的，只能understand不能generate。CoT-VLA选择VILA-U是deliberate trade-off：
- 代价：训练更复杂、inference更慢、image quality有限
- 收益：能做visual CoT、能利用action-less video

这是一个**architecture alignment**的赌注：如果未来是unified multimodal model的天下，那CoT-VLA的framework天然可scale；如果纯understanding model够用，那OpenVLA更轻量。

### 6.6 跟Diffusion Policy的关系

[Diffusion Policy](https://arxiv.org/abs/2303.04137)用diffusion生成action chunk，能handle multi-modal action distribution。CoT-VLA用autoregressive next-token prediction + full attention生成action chunk——本质上也是在handle multi-modal（不同mode对应不同的token sequence）。区别是diffusion在continuous space，CoT-VLA在discrete token space。Discretize的代价是精度损失（256 bin），收益是能seamlessly integrate进LLM framework。

### 6.7 跟RT-2 / PaLM-E的对比

- [RT-2](https://www.deepmind.com/blog/rt-2-new-cocktail-for-an-robot)和[PaLM-E](https://arxiv.org/abs/2303.03378)都是大VLM直接predict action
- 没有explicit reasoning step，纯end-to-end mapping
- CoT-VLA比这些"reflexive" VLA多了"imagination"的能力，bridge了VLA和world model

### 6.8 跟Gen2act、GR-2的对比

最近的[Gen2act](https://arxiv.org/abs/2409.16283)和[GR-2](https://arxiv.org/abs/2410.06158)也用video generation做robot learning，但侧重不同：
- Gen2act：用human video generation model给robot提供prior
- GR-2：大规模video-language-action pretraining
- CoT-VLA：明确把subgoal generation作为intermediate CoT step，强调closed-loop replanning

### 6.9 Subgoal horizon $n$ 的设计直觉

不同dataset的 $[n_l, n_u]$ 范围设置很有讲究：
- Bridge: [5, 10]——frame rate 5Hz，所以5-10帧是1-2秒
- TOTO: [20, 24]——TOTO帧率低，每帧跨越更长时间
- RoboTurk: [1, 2]——已经包含很多pause，预测太远没意义
- EPIC-KITCHENS: [5, 7]——egocentric human action，5-7帧是一段meaningful动作

这个range的选择反映了"predict enough future to inform planning, but not so far that subgoal becomes unpredictable"的直觉。太近=没有planning value，太远=distribution shift太大。

---

## 7. 关键take-away

### 7.1 核心thesis
Visual CoT = 让VLA在act之前先imagine future pixel state。这比language CoT更接近物理grounding，比keypoint/bbox CoT更scalable（不需要额外annotation），比world model dynamics更computationally cheap（只predict一步）。

### 7.2 工程贡献
1. **Hybrid attention**：causal for vision/text + full for action，让action chunk内部mutually consistent
2. **Two-phase training**：pretrain on $D_r \cup D_v$ 学visual reasoning，fine-tune on $D_r$ 学action grounding
3. **Closed-loop replanning**：每m步重新imagine一次，handle distribution shift

### 7.3 Limitations指向future
1. Inference speed：等fast generation技术成熟
2. Image quality：等unified multimodal model的generation quality追上diffusion
3. Action chunk discontinuity：等temporal smoothing技术整合
4. OOD generalization：等world model scaling law显现

### 7.4 对你的intuition启示

如果你在思考下一代的agent design：
- **CoT不一定要是language**——任何能predict future state的intermediate representation都可以作为"reasoning step"
- **Action prediction和world modeling应该couple**——单独学world model容易浪费capacity，单独学action prediction缺乏planning ability
- **Discrete token space是个unifying interface**——vision、language、action都能进token space，统一的autoregressive framework就能scale

CoT-VLA的视觉CoT本质上是在说：**reasoning就是内部的predictive simulation**，无论是LLM的language CoT、AlphaGo的MCTS、还是CoT-VLA的subgoal generation，本质上都是在act之前先在内部"run forward"一个model。

---

## References

- [CoT-VLA项目主页](https://cot-vla.github.io/)
- [VILA-U paper](https://arxiv.org/abs/2409.04429)
- [OpenVLA paper](https://arxiv.org/abs/2406.09246)
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [Chain-of-Thought prompting](https://arxiv.org/abs/2201.11903)
- [LIBERO benchmark](https://arxiv.org/abs/2306.03310)
- [Bridge-V2](https://arxiv.org/abs/2308.12952)
- [RQ-VAE](https://arxiv.org/abs/2203.01941)
- [ACT (Action Chunking Transformers)](https://arxiv.org/abs/2304.13705)
- [Embodied CoT (ECoT)](https://arxiv.org/abs/2407.08693)
- [3D-VLA](https://arxiv.org/abs/2403.09631)
- [Emu3](https://arxiv.org/abs/2409.18869)
- [Show-o](https://arxiv.org/abs/2408.12528)
- [Janus](https://arxiv.org/abs/2410.13848)
- [Transfusion](https://www.arxiv.org/abs/2408.11039)
- [Gen2act](https://arxiv.org/abs/2409.16283)
- [GR-2](https://arxiv.org/abs/2410.06158)
- [SUSIE](https://arxiv.org/abs/2310.10639)
- [Video as the new language](https://arxiv.org/abs/2402.17139)
- [Dreamer](https://arxiv.org/abs/1912.01603)
- [HER (Hindsight Experience Replay)](https://arxiv.org/abs/1707.01495)
- [Speculative decoding](https://arxiv.org/abs/2211.17192)
- [Consistency models](https://arxiv.org/abs/2303.01469)
- [Chameleon](https://arxiv.org/abs/2405.09818)
- [Unified-IO 2](https://arxiv.org/abs/2312.17172)
- [PaLM-E](https://arxiv.org/abs/2303.03378)
- [Goal-Conditioned Imitation Learning](https://arxiv.org/abs/1903.08154)
- [VideoPoet](https://arxiv.org/abs/2312.14125)
- [DC-AE](https://arxiv.org/abs/2410.10733)
- [Token reduction for images](https://arxiv.org/abs/2406.07550)
- [ReKeP](https://arxiv.org/abs/2409.01652)
- [ProgPrompt](https://arxiv.org/abs/2209.11302)
- [EmbodiedGPT](https://arxiv.org/abs/2312.11085)
