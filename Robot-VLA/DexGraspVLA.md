---
source_pdf: DexGraspVLA.pdf
paper_sha256: dbd34b755eb1fc2baa9bbfa06c746d234ec65469aa9b5c051cb29d723f6f109e
processed_at: '2026-08-03T20:21:54-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 DexGraspVLA

## 一句话版本

机器人抓东西这个事，你给它两千段人类示范，它就能在新环境里抓一千多种没见过的东西，成功率 90% 以上。秘诀在于：别让 policy 直接看 raw image，先用现成的大模型把 image "翻译"成一种固定格式的中间表示，policy 只学这个中间表示到 action 的 mapping。

## 问题到底有多难

Dexterous hand 抓东西和 parallel gripper 完全不是一个量级。Parallel gripper 就两个夹板，pinch 一下就完事。Dexterous hand 六个 finger，每个都要协调，抓一个水杯可能需要 thumb 在一边、index 在另一边、middle 托底部，每个 finger 的 contact point 和 force 都要讲究。

更头疼的是 real world 的 variation。你训练时候桌子是白的，测试时候桌子铺了花桌布；训练时候白光，测试时候 disco light；训练时候抓的是苹果，测试时候要抓一个没见过的透明水壶。Image 分布一变，policy 就 out of distribution，直接崩。

之前有几条路都走不通：

**Two-stage pipeline**：先用 network 预测一个 grasp pose，再 open-loop motion planning 过去。问题是你预测的时候 object 可能被遮住了，或者 robot 手有点偏差，open-loop 没法纠正，一抓就飞。

**Reinforcement learning in simulation**：在 Isaac Gym 里并行跑几万个 robot，学出很 fancy 的 in-hand rotation。问题是 sim 的 physics 和 real 不一样，rubber 的摩擦系数、gravity、joint backlash 都对不上，sim-to-real gap 能让成功率从 100% 掉到 10%。

**End-to-end VLA fine-tuning**：拿 OpenVLA 或 π0 这种 VLM，直接在 robot data 上 fine-tune。问题是 data 不够。Open X-Embodiment 攒了几百万条 demo，π0 用了几万条，但仍然 generalize 不好。你看 paper 里 Table 1a，π0 Full FT 在 seen object 上 75%，换 unseen background 直接掉到 0-20%。catastrophic forgetting 加上 data 不够，大模型的优势全没了。

**Plain imitation learning on raw image**：Diffusion Policy 用小 ViT 直接从 raw image 学。Paper 里 ViT-small ablation 在 single-object unseen 上只有 35%。35%！这说明 raw image 到 action 这个 mapping 在数据有限时根本学不好，因为 image 变化太大，policy 要同时学 "什么 object" + "object 在哪" + "怎么抓"，每个都难。

DexGraspVLA 的 insight 是：把这些 sub-problem 拆开，每个用合适的工具解决。

## 核心 idea：Domain-Invariant Representation

打比方。你训练一个小孩抓东西，你给他看一千张苹果照片，告诉他怎么抓。然后你测试时候给他一个梨，背景换成花园，光变成黄昏——小孩照样能抓。为什么？因为小孩脑子里处理的不是 pixel，是 "apple-like object，red，round，in this position"。他的 visual system 已经把 raw image 抽象成一个 stable 的 semantic representation，position 和 shape 信息保留了，无关的 background、lighting 被忽略了。

DexGraspVLA 想让 robot 也这样。具体做法是用一组 frozen foundation models 当这个 "visual system"：

1. **Qwen-VL 当眼睛和脑子**：看 scene，听 user 的 instruction "grasp the cookie"，决定先抓哪个，输出 bounding box。
2. **SAM + Cutie 当注意力**：bbox 告诉 SAM "这个位置是 target"，SAM 输出 mask。Cutie 接过来在后续帧里 track 这个 mask，object 移动或被遮挡也能跟上。
3. **DINOv2 当 visual cortex**：把 raw image 变成 patch features。DINOv2 是 self-supervised pretrain 的，features 对 lighting、background、texture 变化天然 robust——因为它在 LVD-142M 数据集上见过各种各样的 image，早就学会了什么 invariance 是重要的。

policy 看到的输入是：**DINOv2 features + mask + proprioception**，全是 high-level、domain-invariant 的。它不需要识别 object，不需要懂 language，不需要理解 background，只需要学 "这个 mask 对应的 object 在这个 relative position，我的手现在在这个 configuration，下一步 joint target 应该是什么"。

这个 mapping 简单多了。简单到 2000 条 demo 就能学好。简单到在新环境里 input 分布几乎不变，所以 zero-shot 直接 work。

## 架构拆解

### Planner：Qwen-VL 干的活

Planner 不是一直在线的，它是离散触发。流程是：

1. User 给一个 prompt p，可能是 "clear the table" 这种 long-horizon，也可能是 "grasp the toy" 这种 single。
2. Planner 看 head camera image，决定下一个要抓哪个 object，输出一句 natural language description，比如 "grasp the blue yogurt in the middle"。
3. Planner 把这句 description 和 head image 给自己，让它输出 bounding box，格式严格 JSON：`{"bbox_2d": [x1,y1,x2,y2], ...}`。
4. Planner 把 bbox 传给 controller，controller 开始干。
5. Controller 干完了（或者 fail 了），robot reset 回 initial pose。Planner 看 wrist + head image，判断 "这次抓成功了吗"。
6. 成功了就 update scene，再决定下一个 object。失败就 retry 同一个 object。
7. 每次 update 后，planner 还要判断 "整个 user prompt 完成了吗"——比较 initial image 和 current image，看 target 是不是都清掉了。

所有这些 sub-task 都是用 prompt engineering 实现的，Appendix A.1 把所有 system prompt 都贴出来了。比如 instruction proposal 的 prompt 里塞了 4 条 heuristic："prefer objects on the right"、"avoid blocked objects"、"avoid toppling others"、"match user prompt"。这些都是 hard-coded human knowledge，但很 reasonable，能让 VLM 输出更靠谱的选择。

Planner 还做了 image cropping：把 head image 桌面区域 crop 出来，桌外填白。这是为了减少 VLM 的 distraction，让它专注看桌面。

Key insight：**language 是 domain-variant 的**（不同人说法不同，"grasp the cookie" 和 "pick up the biscuit" 是一回事），**bounding box 是 domain-invariant 的**（永远是 4 个 pixel 坐标）。这一步把 language 的多样性吸收掉了，controller 完全不需要懂 language。这是个很漂亮的 abstraction boundary。

### Controller：四个 frozen model + 一个 trainable DiT

Controller 的 pipeline 是 paper 的核心。我画一下 data flow：

```
Head image  ──→ DINOv2 ViT-B/14 (frozen) ──→ z^h ∈ R^(1369×768)
                                                 │
BBox ──→ SAM (frozen) ──→ m_0                    │
              │                                  │
              ↓                                  │
         Cutie (frozen) ──→ m_t                   │
              │                                  ↓
              └─→ random ViT ──→ z^m ∈ R^(1369×768)
                                                 │
Wrist image ──→ DINOv2 ViT-L/14 (frozen) ──→ z^w ∈ R^(1369×1024)
                                                 │
Proprioception s_t ∈ R^13 ──→ MLP ──→ z^s ∈ R^(1×1024)
                                                 │
                                                 ↓
                              Concat: z^obs ∈ R^(2739×1024)
                                                 │
                                                 ↓
Noised action x_k ──→ project ──→ z^A ∈ R^(64×1024)
                                                 │
Diffusion step k ──→ embed ──→ z^d ∈ R^(1×1024)   │
                                                 │
                                                 ↓
                              Condition: [z^obs, z^d] ∈ R^(2740×1024)
                                                 │
                                                 ↓
                              DiT (12 layers, 8 heads, trainable)
                                                 │
                                                 ↓
                              Predict noise ε̂ ∈ R^(64×13)
                                                 │
                                                 ↓
                              DDIM denoise → action chunk A_t ∈ R^(64×13)
                                                 │
                                                 ↓
                              Execute first 6 actions, re-predict
```

几个细节解释：

**为什么 head 用 ViT-B，wrist 用 ViT-L？** Wrist camera 是 close-up，要 fine-grained 的 finger-object contact 信息，需要更大 capacity。Head camera 是全局 scene，patch 级 semantic 就够了。这是个 empirical 选择。

**为什么 mask 用 randomly initialized ViT？** Paper 没解释，我猜是因为 mask 信息 binary 且简单，random projection 把它从 H×W×1 拉到和 DINOv2 features 同样的 1369×768 就行，trainable 部分会学到一个合适的 mapping。如果 mask 信息很复杂，应该用预训练 ViT；但 mask 就是个 binary segmentation，random 起点足够。

**Patch-wise concat 怎么做？** Head image 被切成 37×37 = 1369 个 patch。每个 patch 有 768 维 DINOv2 feature 和 768 维 mask feature，concat 成 1536 维。这样每个 head patch 都知道"我这个位置是不是 target"。

**DiT 内部怎么工作？** 64 个 action token（每个 1024 维）做 self-attention，互相 attend，捕捉 temporal coherence——action chunk 里的 64 步要连贯。然后 action token 做 cross-attention 到 condition sequence（2740 个 token），让 action 知道当前 obs 是什么样的。每个 DiT layer 都这样，12 层堆叠，逐渐 refine noise prediction。

**为什么用 diffusion 而不是直接 regression？** Grasping 是 multi-modal 的——抓同一个杯子，你可以从左边抓也可以从右边抓，两种都对。Deterministic regression 会 average 这两种 mode，得到一个从中间穿过去的无效 action。Diffusion 能 model 一个分布，sample 出哪个 mode 都行。另外 diffusion 的 stochasticity 让 retry 有意义——同 obs 不同 sample 出不同 trajectory，第一次 fail 了第二次换个方式可能就成功了。这就是为什么 @1→@3 从 90.8% 提升到 96.9%。

**Receding horizon H_a=6 怎么定？** Diffusion policy 预测 64 步，但只执行前 6 步（0.3 秒），然后重新 observe + predict。为什么不全执行？因为 open-loop 64 步遇到扰动就崩。为什么不少于 6？因为每步都 predict 太慢，diffusion inference 有延迟。6 步是个 sweet spot，既享受 action chunk 的 temporal smoothing（avoid per-step jitter），又保持 responsiveness（每 0.3s 修正一次）。

## Frozen vs Trainable：最关键的 ablation

Table 1d 是这篇 paper 最重要的实验。我把它拎出来：

| 变体 | Seen | Unseen | Aggregated |
|------|------|--------|------------|
| ViT-small (trainable 小 ViT) | 60.0% | 35.0% | 50.5% |
| DINOv2-train (DINOv2 可训练) | 30.0% | 43.5% | 34.8% |
| **Ours (DINOv2 frozen)** | **98.5%** | **98.8%** | **98.6%** |

ViT-small 是 trainable 的小 ViT，相当于 Diffusion Policy 的标准配置。50.5% aggregated，不算太差，但 unseen 只有 35%，generalization 有限。

DINOv2-train 是 DexGraspVLA 完整架构，但 DINOv2 解冻了。结果 34.8%，比 ViT-small 还差！这非常反直觉——用了更大的预训练模型，怎么反而更差？

答案：**DINOv2 的大 capacity 在 trainable 状态下 overfit 训练 domain 的 pixel-level statistics**。2000 条 demo 对 ViT-L 来说太少，它会去记忆 training image 的 background、lighting、texture pattern，丢掉预训练时学到的 invariance。Frozen 状态下，DINOv2 保留预训练 invariance，policy 只学 invariant feature → action 的 mapping，generalize 自然好。

Unseen（98.8%）比 Seen（98.5%）还高一点，这有点诡异。Paper 解读为 "model 学的是 grasping task 本身，不是 overfit training data"。我同意，这进一步说明 frozen DINOv2 把 task-irrelevant 的 variation 全 filter 掉了，policy 看到的 training 和 testing 输入分布基本一致。

这个 ablation 给所有想 fine-tune VLM 做 robot 的人一个警示：**limited data 下，frozen + small trainable head > fine-tune everything**。大模型 fine-tune 需要大数据，否则 catastrophic forgetting + overfitting 双重打击。

## Figure 4：可视化证据

Figure 4 把 "domain-invariant representation 真的 invariant" 这件事可视化出来。同一个 9-object cluttered scene，目标 "grasp the blue yogurt in the middle"，四种环境：白桌、calibration board、彩色桌布、彩色桌布+disco light。

第一行 raw image：四个环境视觉上差得离谱，根本看不出是同一个 scene。

第二行 DINOv2 features：做 PCA 可视化，把前三个 principal component 映射到 RGB。四个环境下，yogurt 的 feature 颜色几乎一致，background 区域被 threshold 掉了。这说明 DINOv2 看到的"世界"在四种环境下是同一个世界。

第三行 Cutie mask：四种环境下 mask 都准确锁定 yogurt，没被 background 干扰。

第四行 DiT cross-attention：把所有 diffusion step × DiT layer × head × action token 的 attention weight 求和归一化，得到 37×37 attention map。四种环境下 attention 都聚焦在 yogurt 上，没被桌布花纹或 disco light 分散。

第五行 overlay：把 attention map 叠到 raw image 上，确认 attention 落在正确 object 上。

这个 figure 是 paper 的 mechanism-level proof。它不是只说 "我们 generalization 好"，而是展示了 generalization 的因果链：**环境变化 → DINOv2 features 不变 → mask 不变 → attention 不变 → action 不变**。这条链条上每一环都对，所以最终 policy behavior 跨环境一致。

## 大规模 generalization：Table 1c

这是 paper 的 headline result。1287 个 unseen combination（360 objects × 6 backgrounds × 3 lightings），"zero-shot"（训练环境和测试环境是不同房间，Figure 5 对比了 data collection site 和 test site，视觉上明显不同）：

- Ours@1: 90.8%
- Ours@2: 94.7%
- Ours@3: 96.9%

90.8% 是什么概念？这意味着你随便拿一个没见过的 object 放在没见过的桌布上打没见过的灯，robot 抓起来的概率超过 90%。这在 dexterous grasping 领域是前所未有的。

@2、@3 说明很多 failure 是 diffusion sampling 的 stochasticity 导致的，retry 就能 recover。这也意味着 policy 本身是 robust 的，fail 不是因为它 "不会抓"，是它这次 sample 出了个不太好的 trajectory。

耗时：平均 6 秒抓一个 object，接近人手速度，practical usability 没问题。

## Baseline 对比：Table 1a

| Method | Seen | Unseen Obj | Unseen Bg | Unseen Light | Aggr |
|--------|------|------------|-----------|--------------|------|
| OpenVLA (LoRA) | 33.3 | 16.7 | 14.6 | 4.2 | 12.9 |
| OpenVLA-OFT (LoRA) | 25.0 | 29.2 | 31.3 | 31.3 | 30.3 |
| RDT (Full FT) | 25.0 | 25.0 | 31.3 | 35.4 | 31.1 |
| π0 (LoRA) | 58.3 | 45.8 | 14.6 | 10.4 | 22.7 |
| π0 (Full FT) | 75.0 | 45.8 | 20.8 | 20.8 | 30.3 |
| **Ours** | **91.7** | **91.7** | **89.6** | **93.8** | **91.7** |

观察：

**OpenVLA** 是 7B VLM 直接 LoRA fine-tune，aggregated 12.9%。连 seen object 都只有 33%。这就是 end-to-end VLA fine-tune 在 limited data 下的惨状——VLM 预训练的 vision-language 知识没 transfer 到 action 上，反而被 robot data 破坏了。

**π0 Full FT** 在 seen object 上 75%，是所有 baseline 里最高的 seen performance。但 unseen background 直接掉到 0-20%。典型的 overfit training visual domain——它学到的是 training 环境的 pixel pattern，换 background 就 OOD。

**RDT** 比较特别，它也用 frozen vision + language foundation model（SigLIP）。所以它 cross-environment 比 π0 稳定（31.1% vs 22.7% aggregated）。但仍然远不如 Ours（91.7%）。Paper 解释两个原因：第一，RDT 用 SigLIP language encoding 做 condition，language embedding 太抽象，丢了 spatial grounding；第二，SigLIP 是 CLIP-style contrastive 训练，features 偏 semantic 抽象层，丢了 fine-grained visual detail。DINOv2 是 self-supervised dense feature，保留了 patch 级 spatial info。

这个对比说明：**frozen foundation model 是必要的，但不是充分的**。选哪个 frozen model、怎么用它，决定了 generalization 上限。DINOv2 + bbox + mask 这个组合比 SigLIP + language 这个组合强很多。

## Long-horizon：Table 1b

四种 long-horizon prompt：

- "Clear the table"：95.8%
- "Grasp all bottles"：91.7%
- "Grasp all green objects"：87.5%
- "Grasp all food"：83.3%
- Aggregated：89.6%

这个 89.6% 是 task success rate，定义为"整个 prompt 所有 stage 都完成"。比如 "grasp all food" 里有 3 个 food item，必须全抓完才算 success。这意味着每个 stage 都要成功，且 planner 要正确 decompose + 正确判断 completion。

Planner 的分项指标：
- Instruction Proposal：94.3%（94.3% 的时候 planner 选对了下一个要抓的 object）
- BBox Accuracy：98.4%（给定 instruction，98.4% 的时候 bbox 准确）
- Grasp Outcome Verification：paper 没单独报，但能推断很高
- Completion Check：96.3%（96.3% 的时候 planner 正确判断了"整个 prompt 完成了没"）
- Controller grasping：92.2%（controller 自己的 grasp 成功率）

Appendix C 给了一个完整 trace，prompt 是 "grasp all edible objects including food and drinks"。Planner 先识别 "green and orange snack bag"，抓完后判断 "还有 bottle 和另一个 food 没抓"，继续。整个过程 planner 展示了 commonsense reasoning（"edible" 这个词它要 infer 出 bottle、snack、food packet 都算），视觉 grounding（每次都能找到正确 object），和 state tracking（记住哪些抓了哪些没抓）。这让人看到 hierarchical VLA 的 reasoning 能力。

## Nonprehensile grasping：Table 1e

这个 experiment 展示了 framework 的 generality。任务是 push flat object（盘子、书、盒子）到桌边再 grasp。这个任务 parallel gripper 做不了（盘子太扁平，没地方 pinch），需要 dexterous hand 先 push 创造可抓的 pose，再 grasp。两阶段 skill 链。

数据：1029 demos，32 objects。Architecture 不变，只换数据。Controller 改 DINOv2-B/B（两个都用 ViT-B），action horizon 100（因为 push+grasp 链路长）。

结果：84.7% aggregated（unseen obj 88.9%，unseen bg 86.1%，unseen light 77.8%）。

ViT-small 在这个任务上只有 39.6%，DINOv2-train 66.0%。Gap 仍然巨大，证明 frozen DINOv2 + diffusion policy 这个 recipe 跨任务 transfer。

这个 experiment 的意义：**architectural invariance**。同一套 SAM+Cutie+DINOv2+DiT pipeline，只换数据就适应新任务，不需要重新设计 architecture。这是 generality 的第三层证据——第一层是 cross-environment，第二层是 cross-object，第三层是 cross-task。

## Data Collection 的 trick

Paper 用 **kinesthetic teaching** 收 data。Robot 设为 teaching mode（关节电机卸力），人手动引导 robot arm 到 target 位置，执行 grasp。记录每个 timestep 的 joint angles 作为 target。然后 reset 环境，用 PD control replay 这些 target 作为 action 标注。

这个 trick 好处：
1. **没有 teleoperation latency**：人直接 move robot，没有 master-slave 的延迟问题。
2. **Clean action labels**：action 就是人 move 到的 joint configuration，绝对准确。
3. **可以收集 hard example**：人可以 force robot 去抓 awkward pose 的 object，policy 也能学到。

每个 episode 75 timesteps @ 20Hz = 3.75 秒，接近人手速度。2094 条 demo，36 objects，cluttered scenes（每个 scene 约 6 个 object）。Data collection site 和 test site 是不同房间（Figure 5），确保 "zero-shot" 评估。

Mask 标注用和 inference 一样的 SAM + Cutie pipeline 生成，保证 train/test consistency。这点很重要——如果 training 用人工标注的 mask，inference 用 SAM 生成的 mask，两者分布有 shift，policy 会 fail。

## 对做 robot 的人的启发

### 启发 1：Foundation model 是 invariance extractor，不是 feature extractor

大家用 DINOv2 通常当 feature extractor，觉得它 features 好用。这篇 paper 指出更深一层：DINOv2 的价值在于它对 domain variation 的 invariance。Frozen 状态下这个 invariance 保留，trainable 状态下被破坏。这个 insight 适用于所有想用 foundation model 做 robot perception 的工作。

### 启发 2：Limited data regime 下，modular > end-to-end

End-to-end VLA 需要 millions of demos 才能 generalize（Open X-Embodiment 那个量级）。现实中很难收集这么多。Modular approach 把 task 拆成 perception + control，perception 用 frozen foundation model（不耗 data），control 用 small policy（少量 data 就够）。这个 recipe 在 data-limited regime 更实际。

### 启发 3：Hierarchical decomposition 要选对 abstraction boundary

DexGraspVLA 的 boundary 是 bbox。Planner 输出 bbox，controller 接收 bbox。bbox 是 spatial grounding，既保留了位置信息，又比 raw image 抽象，language 的多样性在 planner 那边吸收掉了。这个 boundary 选择是关键。

如果 boundary 选错了，比如 planner 直接输出 language embedding 给 controller（像 RDT），spatial info 就丢了。或者 planner 输出 pixel-level mask 给 controller，planner 就太重了（VLM 不擅长 pixel-level segmentation）。

### 启发 4：Diffusion policy 的三件套缺一不可

Diffusion policy 的威力来自三个东西的组合：
1. **Multi-modal action distribution**：能 model 多种合理 grasp 策略
2. **Action chunking**：64 步一起预测，temporal coherence
3. **Receding horizon control**：只执行前 6 步，保持 responsiveness

缺任何一个都会显著降级。Deterministic head 会 mode-collapse。Per-step prediction 会 jitter。Full open-loop 会 fragility。

### 启发 5：Retry 是 feature 不是 bug

@1→@3 从 90.8% 提升到 96.9%。这说明 diffusion policy 的 stochasticity 让 retry 有意义。Deterministic policy retry 也没用（每次输出一样）。这个 insight 对 real-world deployment 重要——robot 应该允许 retry，不该一次 fail 就 give up。

## Limitations 和 open questions

Paper 自己承认的：
1. **没做 functional grasping**：只 grasp 不做后续 manipulation（pour、twist、insert）。Functional grasping 需要更 fine-grained 的 affordance（contact point、grasp type）。
2. **没用 tactile sensing**：纯视觉，object 被手遮住后视觉就盲了。Tactile 能补这个 gap。
3. **Planner 是 black box**：VLM reasoning 不可控，有时 propose 怪 instruction。94.3% proposal success 意味着 5.7% long-horizon 因为 planner 错误而失败。
4. **Reset 之间是 discrete**：每次 grasp 后 robot reset 回 initial pose。Real task（bin picking）希望 continuous multi-grasp。

我补充的 open questions：

1. **bbox vs keypoint vs heatmap affordance**：bbox 粒度粗，只告诉 controller "object 在这"，不告诉 "抓哪里"。ReKep 用 keypoint constraints，VoxPoser 用 3D value map，能否做需要精细 contact 的 functional grasping？

2. **Trainable but invariance-preserving**：能否设计 contrastive loss，fine-tune DINOv2 同时约束 same object 不同环境 features 一致？这样能学 task-specific feature 又保 invariance。理论上应该比 frozen 更好，但实现 tricky。

3. **Tactile integration**：放在 controller input 层（concatenate 到 obs），还是 planner verification 层（用 tactile 判断 grasp stability）？See-to-Touch 的做法是前者。

4. **Planner 可解释性**：能否用 programmatic planner（Code as Policies）替代 VLM，获得可验证性？Trade-off 是 commonsense reasoning 能力会下降。

5. **Continuous multi-grasp**：能否 train 一个 policy 连续 grasp，grasp 后不 reset，直接 transition 到下一个 grasp pose？这需要 controller 学 "grasp 后 release + re-position" 的衔接 skill。

6. **Diffusion inference latency**：16 步 DDIM 在 A800 上要多久？如果用 consistency model 或 rectified flow 一步生成，能否做到 100Hz control？高频率对 fast dynamic 任务（in-hand rotation、catch）关键。

7. **Bimanual extension**：现在 single-arm，13 维 action。Bimanual 是 26 维，action space 爆炸。DiT 能 handle，但 data collection 成本翻倍。Planner 要协调两手分工，更复杂。

8. **Sim-to-real via foundation model invariance**：能否在 sim 里生成大量 data，用 frozen DINOv2 提取 features，real robot 上用同样 features 做 imitation？Foundation model 把 sim-real 的 visual gap 吸收掉。这是个 promising direction，DexGraspVLA 的 philosophy 可以延伸过去。

## 我的整体判断

DexGraspVLA 不是 architectural breakthrough。DiT 是 Peebles & Xie 2023 的，Diffusion Policy 是 Chi et al. 2023 的，SAM 是 Meta 2023 的，DINOv2 是 Meta 2023 的，Cutie 是 2024 的，Qwen-VL 是 2023 的。所有 component 都是现成的。

它的贡献是 **systematic engineering + 正确的 design philosophy**。把 frozen foundation models 串成一个 pipeline，每个负责一段 perception/reasoning，policy 只学简单的 invariant-to-action mapping。这个 recipe 在 dexterous grasping 这种 data-expensive task 上 reach 了 90%+ generalization，是之前没人做到的。

这个 philosophy 和当前 "train one giant VLA end-to-end" 的主流叙事形成对比。End-to-end VLA 需要 data scale，modular VLA 需要 careful design。在 data 有限的 mid-term，modular approach 可能更实际。在 data 充足的 long-term（如果能收集 millions of demos），end-to-end 可能胜出——但那是几年后的事。

DexGraspVLA 是一个 strong baseline 和 design template。未来 VLA 会继续往 end-to-end 走，但 frozen-foundation-model + imitation-on-invariant-representation 这个 recipe 在 dexterous、bimanual、long-horizon 这种 data-expensive task 上会持续有用。

## 参考链接汇总

- **DexGraspVLA**: https://dexgraspvla.github.io
- **Diffusion Policy**: https://diffusion-policy.cs.columbia.edu/
- **DiT**: https://www.wpeebles.com/DiT
- **DINOv2**: https://dinov2.meta.ai/
- **SAM**: https://segment-anything.com/
- **Cutie**: https://huggingface.co/papers/2403.16542
- **Qwen-VL**: https://github.com/QwenLM/Qwen-VL
- **Qwen2.5-VL**: https://arxiv.org/abs/2502.13923
- **OpenVLA**: https://openvla.github.io/
- **π0**: https://www.physicalintelligence.company/blog/pi0
- **RDT-1B**: https://thu-mlunidexterity.github.io/rdt-1b/
- **DDPM**: https://arxiv.org/abs/2006.11239
- **DDIM**: https://arxiv.org/abs/2010.02502
- **Immiscible Diffusion**: https://arxiv.org/abs/2406.12303
- **SayCan**: https://say-can.com/
- **Code as Policies**: https://code-as-policies.github.io/
- **VoxPoser**: https://voxposer.github.io/
- **ReKep**: https://rekep.github.io/
- **VLA Survey**: https://arxiv.org/abs/2507.01925
- **Open X-Embodiment**: https://robotics-transformer-x.github.io/
- **UniDexGrasp++**: https://arxiv.org/abs/2304.00564
- **DeXtreme**: https://dextreme-robot.github.io/
- **See-to-Touch**: https://arxiv.org/abs/2406.09429
- **Consistency Models**: https://arxiv.org/abs/2303.01469
- **Emergent Extrinsic Dexterity**: https://arxiv.org/abs/2310.16534
- **RoboPianist**: https://arxiv.org/abs/2304.04150
- **AnyRotate**: https://arxiv.org/abs/2405.07391

希望这个版本帮你 build 出 intuition。最值得 internalize 的是 **frozen foundation model 作为 invariance extractor 这个角度**，以及 **modular VLA 在 data-limited regime 的 practical advantage**。

---

# DexGraspVLA 深度解读

## 一、核心 intuition: Domain-Invariant Representation 作为 Imitation Learning 的"安全港"

这篇 paper 的灵魂就一句话:**imitation learning 在 raw pixel space 上做 generalization 是徒劳的,因为不同环境的 image 分布 shift 太大,policy 一旦 OOD 就崩**。但如果先用 foundation models 把 raw language + vision 迭代地翻译成一组 **domain-invariant 的 representation**(bounding box → mask → DINOv2 features),那 policy 看到的 input 在不同环境下几乎不变,于是 2000 条 demo 就能 generalize 到 1287 种 unseen combination。

这个思想本质上是一种 **representation-level domain adaptation**,只不过 adapter 是 frozen 的、预训练好的、internet-scale 的 foundation models,不需要训练。它和传统 domain adaptation 的区别在于:foundation model 本身就是"invariance extractor",你在它已经收敛的 feature space 上做 supervised learning,自然享受到它的 robustness。

Paper 里 Figure 4 把这件事可视化得很漂亮:同一个 cluttered scene 在白桌、calibration board、彩色桌布、disco light 下,raw image 看起来差异巨大,但 DINOv2 features 的 PCA 可视化几乎一致,Cutie 跟踪的 mask 一致,DiT 对 head image 的 cross-attention 都聚焦在 target object 上。这就解释了为什么 zero-shot 仍然 work——policy 内部看到的"世界"在四种环境下是同一个世界。

## 二、Hierarchical Architecture 详解

### 2.1 High-Level Planner (Purple in Figure 2)

Planner 用的是 off-the-shelf Qwen VLM,具体来说:
- 短任务(single grasp): **Qwen-VL-Chat** (Bai et al. 2023)
- Long-horizon 任务: **Qwen2.5-VL-72B-Instruct** (Team 2025),配 Qwen2.5-VL-7B 做 speculative decoding 加速

Planner 的职责被 paper 用 prompt engineering 拆成 4 个 sub-task,每个都有专门的 system prompt(见 Appendix A.1):

1. **Instruction Proposal**: 给定 user prompt p (e.g. "clear the table") + current head image,选下一个要 grasp 的 object,输出 natural language description,比如 "grasp the blue yogurt in the middle"。选 object 时遵循 4 条 heuristic:靠右优先、避免被遮挡的、避免 grasp 会引起其他物体倒塌的、最匹配 prompt 的。
2. **Bounding Box Prediction**: 给定 instruction + head image,输出 JSON `{"bbox_2d": [x1,y1,x2,y2], "label": ..., "description": ...}`。这是 domain-invariant affordance signal 的第一步——把 free-form language 压成 4 个数字。
3. **Grasp Outcome Verification**: 给定 head + wrist image + instruction,判断是否真的 grasp 成功,True/False。
4. **Prompt Completion Check**: 比较 initial image 和 current image,判断整个 user prompt 是否完成。

关键 insight: **language 是 domain-variant 的(不同人不同说法),bounding box 是 domain-invariant 的(永远 4 个 pixel 坐标)**。这一步 transformation 把 language 的多样性吸收掉了,controller 不需要懂 language。

Planner 还做 cropping:把 head image 的桌面区域 crop 出来,其余填白,减少 distraction。这是个简单但有效的 trick。

### 2.2 Low-Level Controller (Pink in Figure 2)

Controller 是 closed-loop policy,输入输出:

**输入**:
- Wrist image $I^w_t \in \mathbb{R}^{H \times W \times 3}$ (RealSense D405C, 640×480)
- Head image $I^h_t \in \mathbb{R}^{H \times W \times 3}$ (RealSense D435, 640×480)
- Proprioception $s_t \in \mathbb{R}^{13}$: arm 7-DoF + hand 6-DoF joint angles
- Mask $m_t \in \{0,1\}^{H \times W \times 1}$: 来自 SAM + Cutie

**输出**: action $a_t = (a^{arm}_t, a^{hand}_t) \in \mathbb{R}^{13}$,target joint angles(注意是 absolute target,不是 delta)。

#### 2.2.1 Mask Pipeline: SAM → Cutie

- **SAM** (Kirillov et al. 2023, https://segment-anything.com/) 用 planner 给的 bbox 作为 prompt,生成初始 binary mask $m_0$。SAM 是 zero-shot segmentation,frozen。
- **Cutie** (Cheng et al. 2024, https://huggingface.co/Papers/2403.16542) 是 video object segmentation 模型,用 $m_0$ 作为 first-frame annotation,后续每帧 track mask,产出 $m_t$。这步解决了一个关键问题:cluttered scene 中 target 会被遮挡或移动,光靠 bbox 不够,需要 temporal tracking。Cutie 也是 frozen 的。

这个 pipeline 的本质是: **把"哪个是 target"这个 semantic decision 留给 foundation models,把"如何 grasp"留给 imitation policy**。Policy 不需要学会分辨 object,只需要盯住 mask。

#### 2.2.2 Feature Extraction: Frozen DINOv2

$$z^h_t = \phi^h(I^h_t) \in \mathbb{R}^{L^h \times D^h}$$
$$z^w_t = \phi^w(I^w_t) \in \mathbb{R}^{L^w \times D^w}$$

- $\phi^h$: **DINOv2 ViT-B/14**,head image,resize 到 518×518,patch 14 → 37×37 = 1369 patches,$L^h = 1369$,$D^h = 768$
- $\phi^w$: **DINOv2 ViT-L/14**,wrist image,同样 resize 518×518,$L^w = 1369$,$D^w = 1024$

DINOv2 (Oquab et al. 2023, https://dinov2.metadem AI/) 是 self-supervised ViT,预训练于 LVD-142M,features 以 semantic consistency 闻名,是 paper 选它的核心理由——见 Figure 4 第二行,跨环境 features 几乎不变。

Frozen 是关键: ablation 里 DINOv2-train(解冻 DINOv2)在 single-object 只有 34.8%,而 frozen 98.6%,差 64 个点!这说明一旦 trainable,DINOv2 就会 overfit 到 training domain 的 pixel-level statistics,丢掉 invariance。这是个非常有力的证据,说明 paper 的核心 hypothesis 是对的。

为什么 wrist 用更大的 ViT-L 而 head 用 ViT-B? 我推测是因为 wrist image 是 close-up,detail 更重要(finger 要对准 object 的具体部位),需要更高 capacity;head image 是全局,patch 级 semantic 就够。

#### 2.2.3 Mask Fusion

把 $m_t$ 喂给一个 **randomly initialized ViT**(没说预训练,这点值得注意——也许 mask 信息简单,random projection 就够,或者作者懒得 train),输出 $z^m_t \in \mathbb{R}^{1369 \times 768}$,和 head feature 同形状。

Patch-wise concat:
$$\bar{z}^h_t = [z^h_t \oplus z^m_t] \in \mathbb{R}^{1369 \times 1536}$$

这里 $\oplus$ 是沿 feature dim 拼接。这样每个 head patch 都带上"这个位置是不是 target"的信号。

然后三个 MLP 投影到共同 dim $D=1024$:
$$\tilde{z}^h_t = \text{MLP}_h(\bar{z}^h_t) \in \mathbb{R}^{1369 \times 1024}$$
$$\tilde{z}^w_t = \text{MLP}_w(z^w_t) \in \mathbb{R}^{1369 \times 1024}$$
$$\tilde{z}^s_t = \text{MLP}_s(s_t) \in \mathbb{R}^{1 \times 1024}$$

Concat 成完整 obs sequence:
$$\tilde{z}^{obs}_t = (\tilde{z}^s_t, \tilde{z}^h_t, \tilde{z}^w_t) \in \mathbb{R}^{(1 + 1369 + 1369) \times 1024} = \mathbb{R}^{2739 \times 1024}$$

(Paper 写 $274 \times 1024$,应该是 typo,实际 2739;加上 diffusion embed 后 2740)

#### 2.2.4 Action Head: DiT-based Diffusion Policy

Action chunk 定义:
$$A_t = a_{t:t+H} = [a_t, a_{t+1}, \dots, a_{t+H-1}] \in \mathbb{R}^{H \times 13}$$

$H = 64$ 是 action horizon,每 chunk 64 步 × 13 dim = 832 维。在 20Hz control 下,一个 chunk 覆盖 3.2s,和 demo 平均时长 3.5s 接近,意味着一个 chunk 几乎就是一个完整 grasp。

Diffusion forward process (DDPM, https://arxiv.org/abs/2006.11239):
$$x_k = \alpha_k A_t + \sigma_k \varepsilon$$

- $k$: diffusion timestep,从 0(num_train_timesteps=50)到 50
- $\alpha_k$: signal retention coefficient,随 $k$ 增大单调减小($\alpha_k = \sqrt{\bar{\beta}_k}$, $\bar{\beta}_k$ 是 cumulative product of $1-\beta_k$)
- $\sigma_k$: noise std,随 $k$ 增大单调增大($\sigma_k = \sqrt{1-\bar{\beta}_k}$)
- $\varepsilon \sim \mathcal{N}(0, I_{H \times 13})$: standard Gaussian noise
- beta schedule: **squaredcos_cap_v2**(cosine schedule,variance preserved end 1.0),$\beta_{start}=0.0001$, $\beta_{end}=0.02$

DiT (Peebles & Xie 2023, https://www.wpeebles.com/DiT) 把 noised action chunk $x_k$ project 到 hidden space,得 $\tilde{z}^A_t \in \mathbb{R}^{64 \times 1024}$。

Diffusion timestep 也 embed 到 1024:$\tilde{z}^d_t \in \mathbb{R}^{1 \times 1024}$。

Condition sequence:
$$\tilde{z}_t = (\tilde{z}^{obs}_t, \tilde{z}^d_t) \in \mathbb{R}^{2740 \times 1024}$$

DiT 每个 layer(共 12 层,8 heads,attention dropout 0.1)做三件事:
1. **Bidirectional self-attention over action tokens**(64 个 action token 互相 attend,捕捉 action 之间的 temporal coherence)
2. **Cross-attention from action tokens to condition sequence**(action query,condition key/value,让 action 关注相关 obs patch)
3. **MLP** transform

输出 project 回 action space,预测 noise $\hat{\varepsilon}$。Loss:
$$\mathcal{L} = \mathbb{E}_{k, \varepsilon, A_t} \| \hat{\varepsilon}_\theta(x_k, \tilde{z}_t, k) - \varepsilon \|_2^2$$

训练时用 **Immiscible Diffusion** (Li et al. 2024, https://arxiv.org/abs/2406.12303) 改善 data-noise mapping——把不同 demo 分配到不同 noise level 区间,避免一个 noise level 同时要还原多种 mode,降低学习难度。这是个比较新的 trick,从 generative modeling 借来的。

Inference 用 **DDIM** (Song et al. 2020, https://arxiv.org/abs/2010.02502) sampling,**num_inference_steps=16**,从纯 Gaussian noise 出发,迭代 denoise 得到 action chunk。

Receding horizon control: 只执行前 $H_a = 6$ 个 action(0.3s),然后重新 observe + 重新 predict。这是 closed-loop 的关键——既享受 action chunk 的 temporal smoothing,又保持 responsiveness。这个设计直接来自 Diffusion Policy (Chi et al. 2023, https://diffusion-policy.cs.columbia.edu/)。

#### 2.2.5 Trainable Parameter Count

Controller 只有 **163M trainable parameters**(两个 MLP + mask ViT + DiT + projection),DINOv2 两个 ViT(B+L)和 SAM、Cutie 都 frozen。这比 OpenVLA 7B、π0 3B 小一个数量级,但效果更好——因为 trainable 部分只学 "invariant representation → action",问题简单了。

训练: 84 epochs,8×A800 GPU,<1 天。bf16 mixed precision,FusedAdamW optimizer,lr=1e-4,cosine schedule,warmup 2000 steps,weight decay 1e-4,batch size 48。Color jitter 做 domain randomization。

## 三、Data Collection

**2,094 demonstrations**,36 household objects,cluttered scenes(around 6 objects per scene)。每个 episode 75 timesteps @ 20Hz = 3.75s,接近人手速度。

收集方法: **kinesthetic teaching**——robot 设为 teaching mode,人手动引导 robot arm 到 target 位置执行 grasp,记录 joint angles 作为 target,然后 reset 后用 PD control replay 这些 target 作为 action 标注。这是个聪明的 trick: avoids teleoperation latency and gives clean action labels。

Mask 标注: training data 的 mask 用和 inference 一样的 SAM + Cutie pipeline 生成,保证 train/test consistency。这点很重要,否则 policy 学到的 mask-action mapping 在 inference 时会 shift。

Object 多样性: Figure 7 的 t-SNE 显示 360 unseen objects 在 length/width/height/mass/roughness/shape 6 维上分布广。

## 四、实验结果深度分析

### 4.1 Large-Scale Generalization (Table 1c)

- **Ours@1**: 90.8% aggregated (360 unseen objects × 6 bgs × 3 lightings = 1287 tests)
- **Ours@2**: 94.7% (允许 retry)
- **Ours@3**: 96.9%

@2/@3 的提升说明很多 failure 是 inference randomness 导致的,diffusion policy 每次 sample 不同,retry 能 recover。这也意味着 paper 的 model 本质上是 robust 的,fail 主要是 stochastic 的。

Table 3 里的细节:Pink towel 背景 84.5% 最低,可能因为粉色和某些 object 颜色接近,DINOv2 features 区分度下降。Disco light 92.2%,Dark light 91.2%,Lamp light 89.3%——lighting 整体很 robust。

### 4.2 Baseline Comparison (Table 1a)

| Method | Seen | Unseen Obj | Unseen Bg | Unseen Light | Aggr |
|---|---|---|---|---|---|
| OpenVLA (LoRA) | 33.3 | 16.7 | 14.6 | 4.2 | 12.9 |
| OpenVLA-OFT (LoRA) | 25.0 | 29.2 | 31.3 | 31.3 | 30.3 |
| RDT (Full FT) | 25.0 | 25.0 | 31.3 | 35.4 | 31.1 |
| π0 (LoRA) | 58.3 | 45.8 | 14.6 | 10.4 | 22.7 |
| π0 (Full FT) | 75.0 | 45.8 | 20.8 | 20.8 | 30.3 |
| **Ours** | **91.7** | **91.7** | **89.6** | **93.8** | **91.7** |

观察:
- **OpenVLA** (https://openvla.github.io/, Kim et al. 2024) 直接 fine-tune VLM end-to-end,连 seen object 都只有 33%,catastrophic forgetting 严重,language pretrain 的知识没 transfer 过来。
- **π0** (https://www.physicalintelligence.company/blog/pi0, Black et al. 2024) Full FT 在 seen object 上 75%,但 unseen bg/lighting 直接掉到 20%——典型的 overfit training visual domain。
- **RDT** (https://thu-mlunidexterity.github.io/rdt-1b/, Liu et al. 2024) 也用 frozen vision + language foundation model(SigLIP),所以 cross-environment 比 π0 稳定(31% vs 22.7% aggregated),但仍远不如 Ours(91.7%)。Paper 解释: bbox 比 language encoding 更 spatial grounding,DINOv2 比 SigLIP 保留更多 visual detail。这暗示 language embedding 太抽象,丢掉了 grasp 需要的几何信息。
- **OpenVLA-OFT** (Kim, Finn, Liang 2025, https://openvla.github.io/oft) 是 OpenVLA 的改进版,加 action chunking,unseen 上比原版好很多(30.3 vs 12.9),但仍远不如 Ours。

关键 takeaway: **end-to-end VLA fine-tuning 范式在 limited data 下不行**。需要数百万 demo 才可能 generalize(O'Neill et al. 2023 Open X-Embodiment),不实际。

### 4.3 Ablation (Table 1d) — 这是最关键的证据

Single-object grasping:
- **ViT-small** (trainable small ViT,相当于 Diffusion Policy): Seen 60%, Unseen 35%, Aggr 50.5%
- **DINOv2-train** (DexGraspVLA 但 DINOv2 trainable): Seen 30%, Unseen 43.5%, Aggr 34.8%
- **Ours** (frozen DINOv2): Seen 98.5%, Unseen 98.8%, Aggr 98.6%

注意 **DINOv2-train 比 ViT-small 还差**!这说明 frozen DINOv2 不是"用了大模型所以好",而是"frozen 这个状态本身重要"。一旦解冻,DINOv2 的预训练 invariance 被破坏,反而比小 ViT 还差——因为大模型 capacity 大,更容易 overfit training domain 的 pixel-level pattern,丢掉 generalization。这是个很强的反直觉证据。

Unseen(98.8%)比 Seen(98.5%)还高一点,paper 自己也觉得意外,解读为"model 学的是 grasping task 本身,不是 overfit training data"。这进一步印证 domain-invariant representation 的力量。

48% 的 ablation gap 是这篇 paper 的核心贡献量化证据。

### 4.4 Mechanism Analysis (Section 5.5, Figure 4)

实验设置: 同一个 9-object cluttered scene,目标 "grasp the blue yogurt in the middle",四种环境(白桌/calibration board/彩色桌布/同彩色桌布+disco light)。

可视化四件事:
1. **Raw head image**(第一行): 视觉上差异巨大
2. **DINOv2 features**(第二行): PCA 可视化,几乎一致。具体做法: 对 4 张 image 的所有 patch features 一起做 PCA,threshold 去 background,再对 foreground features 做 PCA,前 3 个 principal component 映射 RGB。
3. **Cutie mask**(第三行): 准确锁定 target
4. **DiT cross-attention to head image**(第四行): 平均所有 diffusion step × all DiT layer × all head × all action token 的 attention weight,得 37×37 map,upsample 到 518×518。所有环境下都聚焦在 target object 上,不被环境 distract。

这是 paper 的 mechanism-level explanation,把 "为什么 generalize" 的因果链完整展示出来:**环境变化 → DINOv2 features 不变 → mask 不变 → attention pattern 不变 → action 不变**。

### 4.5 Long-Horizon (Table 1b)

四种 prompt:
- "Clear the table": 95.8%
- "Grasp all bottles": 91.7%
- "Grasp all green objects": 87.5%
- "Grasp all food": 83.3%
- **Aggregated: 89.6%**

Planner 指标:
- Instruction Proposal 成功: 94.3%
- BBox Accuracy: 98.4% (Appendix D 单独验证 99.3%)
- Controller grasping: 92.2%
- Completion Check: 96.3%

Avg 1.12 attempts per grasp,说明 retry 不频繁。

Appendix C 展示了一个完整 trace,planner 用 commonsense 识别 "edible objects including food and drinks",分解成 "snack bag" → "brown snack packet" → "white bottle" 三步,每步都 verify 成功后才继续。这个 trace 让人看到 hierarchical VLA 的 reasoning 能力。

### 4.6 Nonprehensile Grasping (Table 1e, Section 5.7)

任务: 把 flat wide-surface object(plate、book、box)push 到桌边再 grasp。这是 Zhou & Held 2023 (https://arxiv.org/abs/2310.16534) 的 emergent extrinsic dexterity 思想——parallel gripper 做不了,需要 dexterous pre-grasp maneuver。

数据: 1029 demos,32 objects。Controller 改 DINOv2-B/B(都 ViT-B),action horizon 100(因为 pre-grasp + grasp 链路长)。106M trainable params,200 epochs,2 天。

结果: 84.7% aggregated(unseen obj 88.9%, unseen bg 86.1%, unseen light 77.8%)。ViT-small 39.6%,DINOv2-train 66.0%。Gap 仍然巨大,证明 framework 跨任务 transfer。

这个实验的意义: **architectural invariance**。同一套 SAM+Cutie+DINOv2+DiT pipeline,只换数据就适应新任务,不需要重新设计 architecture。这是 generality 的第三层证据。

## 五、关键设计决策的 intuition

我整理一下这篇 paper 几个 counter-intuitive 但 critical 的选择:

1. **Frozen > Trainable foundation model**: 直觉上 fine-tune 大模型应该更好,但实验证明 frozen 才保住 invariance。Trainable 让大模型 overfit training pixel stats,反而比小模型差。这个结论对所有想 fine-tune VLM 做 robot 的人是警示。

2. **Bounding box > Language embedding as affordance**: RDT 用 SigLIP language feature 做 condition,效果差。因为 language embedding 是高度抽象的 semantic vector,丢掉了 "object 在 image 哪个位置" 这个 spatial info。bbox 是 spatial grounding,经过 SAM→mask→DINOv2 patch-level fusion 后,spatial info 又回来了。

3. **Hierarchical > End-to-end**: π0/OpenVLA 想让一个 VLM 同时做 reasoning + control,数据需求爆炸。DexGraspVLA 让 VLM 只做 reasoning(它擅长的),让 small diffusion policy 做 control(数据 efficient),各司其职。

4. **Diffusion > Deterministic action head**: Diffusion policy 天然 multi-modal,能 model "多种合理 grasp 策略"的分布,且 action chunk 提供 temporal coherence。Deterministic head(MSE regression)会 average 多 mode 得到无效 action。Retry 也靠 diffusion 的 stochasticity——同一 obs 不同 sample 得不同 trajectory,提高 success rate(@1→@3 从 90.8→96.9)。

5. **Receding horizon with H_a=6**: 比 full open-loop(执行全部 64 步)responsive,比纯 step-by-step(每步 re-predict)efficient。6 步 = 0.3s 内不 re-predict,catch 快速 motion;超过 0.3s 重新 observe,适应 disturbance。这是 Diffusion Policy 的标准配方。

6. **Two cameras 分工**: Head camera 给全局 scene context(planner 用),wrist camera 给 close-up detail(controller 用,确保 finger 对准)。两个 DINOv2 不同 size(B vs L),因为 wrist 信息密度高需要更大 capacity。

7. **Cropping head image for planner**: 把桌面外填白,减少 VLM distraction。简单但有效,VLM 不容易被 irrelevant region 误导。

8. **Immiscible Diffusion**: 训练时给不同 demo 分配不同 noise level,降低学习难度。这是 generative model 的 trick 借到 robot learning。

## 六、Limitations 和 Future Work

Paper 自己列了:
1. 没做 functional grasping(grasp 后的后续 manipulation,比如 grasp 后 pour、twist)
2. 没用 tactile sensing,纯视觉。这对 in-hand manipulation 是短板,因为 occlusion 后 visual 就盲了
3. 未来: planner 生成更 fine-grained affordance(不只是 bbox,可能是 contact point、grasp type)
4. Task-oriented manipulation controller + tactile feedback

我自己补充几点 paper 没说但能看出来的:
- **Planner 是 black box**: VLM 的 reasoning 不可控,有时会 propose 怪 instruction。94.3% proposal success 意味着 5.7% 长任务因为 planner 错误而失败。
- **Reset 之间是 discrete**: 每次 grasp 后 robot reset 到 initial pose,不能连续 multi-grasp。真实任务(比如 bin picking)希望 continuous。
- **Mask tracking 在严重遮挡时会丢**: Cutie 不是万能,如果 target 完全被 hand 遮住,mask 就没了。Tactile 可以补。
- **20Hz control frequency**: 比较低,fast dynamic 任务(打乒乓球)做不了。Diffusion inference 16 步在 A800 上可能就是瓶颈。
- **Demo 还是需要 kinesthetic teaching**: 不能 scale 到 thousands of objects。未来可能要 teleoperation + VR + 大规模数据采集。
- **没有 force/torque feedback**: hard grasp(硬物)和 soft grasp(soft fruit)用同一 policy,可能 suboptimal。
- **Single-arm**: 没 bimanual,很多 real task 需要两手协作。

## 七、和 VLA 文献的关系

### 7.1 VLA 的两大流派

**End-to-end VLA**: OpenVLA, π0, π0.5, RDT, Octo, RT-2。直接把 VLM fine-tune 到输出 action token。优点是 architecture 统一,缺点是 data hungry + catastrophic forgetting。DexGraspVLA 的 baseline 实验直接证明这条路线在 limited demo 下不行。

**Modular/Hierarchical VLA**: SayCan (https://say-can.com/), Code as Policies (https://code-as-policies.github.io/), VoxPoser (https://voxposer.github.io/), ReKep (https://rekep.github.io/), DexGraspVLA。Foundation model 做 reasoning/grounding,separate policy 做 control。优点是 data efficient + interpretable + generalizable,缺点是 interface design 需要人投入。

DexGraspVLA 属于第二类,但它的贡献是证明了**第二类在 dexterous grasping 这种 high-DoF 任务上能 reach 90%+ generalization**,这是之前 modular 工作没做到的(之前多 parallel gripper 简单任务)。

### 7.2 Diffusion Policy 家族

- **Diffusion Policy** (Chi et al. 2023, https://diffusion-policy.cs.columbia.edu/): CNN-based UNet action head,parallel gripper 任务
- **RDT** (Liu et al. 2024): DiT-based,bimanual,用 SigLIP language condition
- **DexGraspVLA**: DiT-based + DINOv2 frozen features + bbox/mask affordance,dexterous hand

DexGraspVLA 可以看作 Diffusion Policy 的 "generalization-enhanced" 版本: 保留 action chunk + diffusion head,把 raw image encoder 换成 frozen DINOv2,把 language 换成 bbox+mask,大幅提升 generalization。

### 7.3 Foundation Models for Robotics

- **SAM** (https://segment-anything.com/): zero-shot segmentation,被很多 robot work 用(GraspNet, VIM)
- **DINOv2** (https://dinov2.meta.ai/): feature extractor,被 SparseDFF (Wang et al. 2023a) 等用
- **Cutie** (https://huggingface.co/Papers/2403.16542): video object segmentation,在 robot 上用得少,DexGraspVLA 找到了好用途
- **Qwen-VL** (https://github.com/QwenLM/Qwen-VL): planner

DexGraspVLA 把这四个 frozen model 串起来,每个负责一段 perception/reasoning,得到一个 robust pipeline。这种 "Lego 式"组合可能比追求一个 monolithic VLA 更实际。

### 7.4 Dexterous Grasping 文献

- **Two-stage**: SpringGrasp (Chen, Bohg, Liu 2024), GraspXL (Zhang et al. 2024a), DexGraspNet (Wang et al. 2023b), AnyDexGrasp (Fang et al. 2025)。Open-loop,脆弱。
- **RL in sim**: UniDexGrasp (Wan et al. 2023), DeXtreme (Handa et al. 2023), AnyRotate (Yang et al. 2024), DextrAH-RGB (Singh et al. 2024), Rubik's cube (Akkaya et al. 2019)。Sim-to-real gap。
- **Imitation**: Imitation learning from single-camera teleoperation (Qin, Su, Wang 2022), See-to-Touch (Guzey et al. 2024), Visuotactile (Lin et al. 2024b), Sequential Dexterity (Chen et al. 2023)。

DexGraspVLA 在 imitation 流派里第一个做到 thousands-of-conditions generalization with 2000 demos。这个 data efficiency 来自 foundation model invariance。

## 八、可以深挖的几个问题

如果你想 push 这个方向,我想到几个 research question:

1. **为什么 frozen DINOv2 features 是 invariant 的?** Figure 4 是 empirical 证据,但理论上 DINOv2 学到了什么 invariance?是 DINO objective(self-distillation + centering + sharpening)本身 induce 的,还是预训练数据 diversity 给的?如果在 narrow domain data 上 self-supervised pretrain 一个 ViT,能达到同样 invariance 吗?这关系到能不能 reduce 对 internet-scale data 的依赖。

2. **bbox vs keypoint vs heatmap affordance**: paper 用 bbox,粒度粗。如果用 keypoint(像 ReKep)或 heatmap(像 VoxPoser 的 3D value map),能不能做需要精细 contact 的 functional grasping?bbox 给 controller 的 spatial info 不够 fine-grained。

3. **End-to-end trainable but invariance-preserving**: 能否设计一个 trainable architecture,既学 task-specific feature 又保 invariance?比如 contrastive loss 约束 same object 不同环境 features 一致。这样能 fine-tune DINOv2 但不破坏 invariance。

4. **Tactile integration**: paper 没用 tactile。如果加 tactile,放在哪一层?是 controller 的额外 input(类似 See-to-Touch),还是 planner 用 tactile 验证 grasp stability?Tactile 对 occluded grasp 关键。

5. **Planner 的可解释性和可控性**: Qwen-VL 的 reasoning 是 black box。如果 planner 出错,如何 debug?能否用 programmatic planner(像 Code as Policies)替代,获得可验证性?

6. **Continuous multi-grasp without reset**: 现在 reset 之间是 discrete。能否 train 一个 policy 连续 grasp,不 reset?这需要 controller 学 "grasp 后 transition 到下一个 grasp pose"。

7. **Diffusion inference latency**: 16 步 DDIM 在 A800 上要多久?如果用 consistency model (https://arxiv.org/abs/2303.01469) 或 rectified flow 一步生成,能否做到 100Hz control?

## 九、总结

DexGraspVLA 的核心贡献可以浓缩成一句话:**在 dexterous grasping 上证明了 "frozen foundation models as invariance extractor + imitation learning on invariant representation" 这个 recipe 能用 2000 demos generalize 到 1287 unseen conditions,达到 90%+ success**。

它不是 architectural breakthrough(DiT、Diffusion Policy、SAM、DINOv2 都是现成的),而是 **systematic engineering + 正确的 design philosophy**: 让每个 component 做它最擅长的事(foundation model 做 invariance,imitation 做 task-specific mapping),不要让一个 model 试图学所有事。

这个 philosophy 其实和 LeCun 一直倡导的 "world model + planning"、和 Fodor 的 "modularity of mind" 都有共鸣。它也对当前 "train one giant VLA end-to-end on robot data" 的主流叙事提供了一个有力的 counter-example:在 data 有限的现实世界,modular + foundation model invariance 可能比 monolithic end-to-end 更可行。

我对这个工作的直觉是:**它不是终点,而是一个 strong baseline 和 design template**。未来 VLA 会继续往 end-to-end 走(因为 data scale 会增长),但 frozen-foundation-model + imitation-on-invariant-representation 这个 recipe 在 mid-term(数据仍然有限)会持续有用,特别是在 dexterous、bimanual、long-horizon 这种 data-expensive 的 task 上。

## 参考链接

- Paper project page: https://dexgraspvla.github.io
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- DiT: https://www.wpeebles.com/DiT
- DINOv2: https://dinov2.meta.ai/ / https://arxiv.org/abs/2304.07193
- SAM: https://segment-anything.com/ / https://arxiv.org/abs/2304.02643
- Cutie: https://huggingface.co/Papers/2403.16542
- Qwen-VL: https://github.com/QwenLM/Qwen-VL / https://arxiv.org/abs/2308.12966
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- OpenVLA: https://openvla.github.io/ / https://arxiv.org/abs/2406.09246
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- π0: https://www.physicalintelligence.company/blog/pi0 / https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- RDT-1B: https://thu-mlunidexterity.github.io/rdt-1b/ / https://arxiv.org/abs/2410.07864
- DDPM: https://arxiv.org/abs/2006.11239
- DDIM: https://arxiv.org/abs/2010.02502
- Immiscible Diffusion: https://arxiv.org/abs/2406.12303
- SayCan: https://say-can.com/
- Code as Policies: https://code-as-policies.github.io/
- VoxPoser: https://voxposer.github.io/
- ReKep: https://rekep.github.io/
- Emergent Extrinsic Dexterity (Zhou & Held): https://arxiv.org/abs/2310.16534
- VLA Survey (Zhong et al. 2025): https://arxiv.org/abs/2507.01925
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- Diffusion Policy (original): https://arxiv.org/abs/2303.04137
- UniDexGrasp++: https://arxiv.org/abs/2304.00564
- DeXtreme: https://dextreme-robot.github.io/
- RoboPianist: https://arxiv.org/abs/2304.04150
- AnyRotate: https://arxiv.org/abs/2405.07391
- See-to-Touch: https://arxiv.org/abs/2406.09429
- Consistency Models (Song et al.): https://arxiv.org/abs/2303.01469

希望这些细节帮你 build 出对 paper 的 intuition。最值得 internalize 的几点:**frozen foundation model 是 invariance extractor,不是 feature extractor 这个角度;modular VLA 在 data-limited regime 的优势;diffusion policy 的 multi-modal action distribution + action chunking + receding horizon 三件套**。如果之后想做 follow-up,我列的几个 research question 里 #2(bbox→keypoint affordance)和 #3(trainable but invariance-preserving)我觉得最有潜力。
