---
source_pdf: DreamDojo A Generalist Robot World Model from.pdf
paper_sha256: c8f02e22c8fbe98e0b50215e10e1f6baf2f589b1987e5c6dc07175bfe6fe8587
processed_at: '2026-08-03T23:20:30-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DreamDojo 人话版

Andrej，我把刚才那堆公式翻译成人话。

---

## 一句话总结

NVIDIA这帮人想造一个"机器人做梦的机器"——机器人闭着眼就能想象"如果我做这个动作，世界会变成什么样"。他们从4.4万小时的人类第一视角视频里偷学物理常识，再花一点点机器人数据微调，就能让world model泛化到从没见过的场景。

---

## 问题是什么

你想训练一个world model，输入"当前画面+一个动作"，输出"下一帧画面"。听起来简单，但robot data有三个要命的麻烦：

**第一，数据太少了。** 你看Table 1，最大的robot dataset也就两三千小时，skill几十种，场景一百来个。真实世界呢？杯子就有成千上万种，抽屉、门把手、工具更是数不清。你拿几千小时的数据训练，模型记住的就是那几个桌面、那几个玩具，换个厨房都不认识。

**第二，teleoperation太贵了。** 每一条轨迹都得人操作，一小时数据收集成本可能几百美元。你想要十万小时？破产都攒不够。

**第三，robot data缺乏多样性。** 都是expert demo，都是"最优动作"，都是"成功的轨迹"。但world model的职责是预测"各种动作的后果"——包括愚蠢的、失败的、counterfactual的动作。只见过成功，怎么预测失败？

---

## 为什么用人类视频

人类视频便宜、多样、量大。随便找一帮人戴GoPro拍日常活动，几十万小时都不难。DreamDojo-HV就是4.4万小时，覆盖家庭、零售、工业、教育等场景，6015种skill。

**关键的intuition**：杯子掉地上碎了，不管推它的是人手还是机械爪，碎法是一样的。World model学的本质是"世界怎么动"，不是"agent长什么样"。物理规律是embodiment-agnostic的。

人手和robot gripper之间确实有embodiment gap，但这个gap在world model层面被diluted了——因为你预测的是"世界变化"，不是"手本身怎么长"。

---

## 核心难题：没有action label

4.4万小时人类视频有一个大问题：**没有action标注**。你知道帧和帧之间发生了什么变化，但你不知道"人做了什么动作"导致这个变化。

有三个选项：

**选项A：不做action conditioning，就纯预测未来帧。** 听起来也行？但实验表明几乎没用。因为模型只学到了"视频统计规律"（比如手一般在画面中间），没学到causality——"因为手推了杯子，所以杯子动了"。这俩是fundamentally不同的东西。

**选项B：用hand pose estimator提取手部动作。** HaMeR这种模型可以提21个DOF的手部关节角度。但问题：遮挡时失败、手离开画面时失败、只提手不提手臂和身体移动、而且hand pose和gripper pose维度对不上，跨embodiment迁移效果差。

**选项C：latent action。** 这是DreamDojo选的路线，也是我觉得最优雅的。

---

## Latent Action 是什么

核心想法非常简单：**让模型自己从两帧之间"猜"出发生了什么动作。**

具体做法是一个VAE：
- Encoder看两帧连续画面 $f^t$ 和 $f^{t+1}$，压缩成一个32维向量 $\hat{a}_t$
- Decoder看 $f^t$ 和 $\hat{a}_t$，重建 $f^{t+1}$

为什么这能work？因为decoder已经有了 $f^t$（背景、光照、物体位置都在里面），要重建 $f^{t+1}$，它只需要知道"什么变了"。而 $\hat{a}_t$ 只有32维——容量很小，塞不下整张图。所以VAE被迫把32维全用来编码"变化的信息"，也就是motion/action。

这就是information bottleneck的魔力。你不用告诉模型"action是什么"，你只要创造一个信息瓶颈，模型自己就会disentangle出action。

**最妙的地方**：这个 $\hat{a}_t$ 是跨embodiment的。人抓杯子的latent action和robot抓杯子的latent action会非常接近，因为"抓"这个动作在视觉上的effect是类似的——杯子从A点移到B点。Figure 3右边那个retrieval实验就是这个意思。

所以latent action成了一个**universal action language**。不管你是什么embodiment——人手、机械爪、人形机器人——只要是"同类动作"，latent representation就接近。

---

## 怎么把latent action喂给world model

World model基于Cosmos-Predict2.5（一个latent video diffusion model）。视频先被WAN2.2 tokenizer压缩，4帧pixel压成1帧latent。也就是说1个latent frame对应4个pixel frame。

这里有两个设计trick：

**Trick 1：用relative action，不用absolute action。**

绝对关节角度的分布非常分散——起点不同，绝对值就差很多。但relative action（相对于chunk起点）分布窄得多。"向前伸5厘米"在任何起点都是"向前伸5厘米"。这降低了模型要学的复杂度。

**Trick 2：chunked injection。**

每个latent frame只喂它自己那4帧的action，不喂整条trajectory。为什么？因为diffusion model是bidirectional的，如果你让第5个latent frame看到第10帧的action，模型会学到"未来动作影响过去"这种spurious correlation。Chunked injection把causality硬编码进去——你只看属于你时间窗口的动作。

这两个trick合起来，让模型对action的control精度大幅提升。Table 5显示chunked injection单独就涨了1.1 PSNR。

---

## 训练流程三步走

**第一步：pretrain on human video。**

把44k hours人类视频用latent action VAE提取proxy action，然后在Cosmos-Predict2.5上继续训练。数据配比In-lab : EgoDex : DreamDojo-HV = 1 : 2 : 10。140k steps，256块H100。

这步的目的是让模型理解"物理世界怎么运作"——物体怎么形变、怎么受力、怎么消失再出现。

**第二步：post-train on target robot。**

拿少量目标机器人数据（比如GR-1的teleop数据），重置action MLP第一层（因为要从latent action空间切换到真实joint space），然后fine-tune所有参数。

这步的目的是让模型适应具体embodiment的"外观"和"动作空间"。

**第三步：distillation。**

为了实时推理。把bidirectional attention换成causal attention，50步denoising压到4步。用Self-Forcing方法分两阶段蒸馏：先warmup（student用teacher output做context），再distill（student用自己的output做context）。

结果：从2.72 FPS到10.81 FPS，4倍加速，质量损失可接受。

---

## 为什么distillation后反而有些地方更好

Table 6和Figure 11展示了一个反直觉的现象：distilled student model在长时间生成和遮挡恢复上反而比teacher好。

原因是架构差异。Teacher是bidirectional的，只能用单帧conditioning——生成一个chunk就忘掉前面的。Student是causal autoregressive的，天然有context window，能看到之前生成的帧。所以当物体被短暂遮挡又出现时，student能从历史context里"记住"物体，teacher做不到。

这暗示一个更深的结论：**也许world model从一开始就该是causal的**，而不是bidirectional然后再distill。Bidirectional架构虽然训练效率高，但丢了"时间因果"这个最重要的inductive bias。

---

## 实验证明了几件事

**Latent action有效。** Table 2显示latent action pretrain的效果接近"ideal setup"（有Manus手套精确标注action）。也就是说，self-supervised的latent action几乎赶上了有监督的精确action label。

**Data scale and diversity有效。** Table 3显示每加一个dataset，OOD性能都涨。说明human video pretrain确实在transfer physics knowledge。

**能做policy evaluation。** Figure 5a显示DreamDojo的success rate和real-world success rate的Pearson相关性达到0.995——你不用deploy到真实机器人，就能预测policy好不好用。

**能做model-based planning。** Figure 5b显示，让policy生成多个候选action，world model roll out，value model选最好的，成功率比baseline提升2倍。相当于test-time的MCTS简化版。

---

## 我的几点直觉

**第一，latent action这条路线是对的。** 它把"从unlabeled video学action-conditioned dynamics"这个fundamental problem给解决了。以前要么ignore action，要么hand-craft retargeting，都不scalable。Latent action是self-supervised的，可以scale到internet video。

**第二，world model学的不是agent，是world。** 这就是为什么human video能帮robot。杯子碎、球滚、纸皱——这些physics跟谁推的没关系。World model本质是learned physics engine。

**第三，distillation的bonus暗示了架构方向。** Causal model的context awareness是bidirectional model结构性无法拥有的。未来world model可能直接设计成causal的。

**第四，scale的故事还远没完。** 44k hours听起来大，但YouTube一天上传量就远超这个。如果能scale到million hours级别，可能涌现更强的physics reasoning，类似LLM的emergent ability。

**第五，limitation里提到的multi-view问题很关键。** 现在SOTA policy（比如GR00T N1.5）需要multi-view输入，而DreamDojo只支持单视角。要真正deploy到modern robot stack里，这个必须解决。

---

## 参考

- **项目主页**: https://dreamdojo-world.github.io
- **Cosmos-Predict2.5**: https://arxiv.org/abs/2511.00062
- **Self Forcing (NeurIPS 2025)**: https://arxiv.org/abs/2506.14218
- **AdaWorld (latent action前作, ICML 2025)**: https://arxiv.org/abs/2506.03865
- **EgoDex**: https://arxiv.org/abs/2505.11709
- **Flow Matching**: https://arxiv.org/abs/2210.02747
- **DiT**: https://arxiv.org/abs/2212.09748
- **WAN2.2**: https://arxiv.org/abs/2503.20314

---

一句话：DreamDojo证明了"先从海量人类视频偷学物理，再用少量机器人数据微调"这条路线work。它本身不是单一技术突破，而是把latent action VAE、relative/chunked action injection、temporal consistency loss、Self-Forcing distillation这些piece拼成一个完整pipeline，并且scale到44k hours验证了有效性。

---

# DreamDojo 深度讲解

Andrej，这篇paper来自NVIDIA的GEAR team（Linxi "Jim" Fan, Yuke Zhu, Joel Jang等），目标是build一个generalist robot world model。核心thesis很清晰：**用44k hours的人类视频做pretrain，让world model先理解"物理世界怎么运作"，再post-train到具体robot上**。下面我尽量build你的intuition。

---

## 1. 为什么需要从human video pretrain

先理解问题本质。现有的robot world model（IRASim, UniSim, Ctrl-World等）几乎都是直接在特定robot dataset上训练。但robot data有三个fundamental problems：

1. **Coverage极窄**：Table 1显示，最大的robot dataset（AgiBot-World）只有2.9k hours、87 skills、106 scenes。而真实世界interaction space几乎是infinite的。
2. **Cost高**：每条trajectory都需要teleoperation。
3. **Stochasticity不够**：绝大多数dataset是expert demonstration，缺少"试探性"、"失败"、"counterfactual action"的分布——而world model恰恰需要预测各种action的outcome。

Human video是天然的解：cheap（crowdsource即可）、diverse（household/retail/industrial等）、stochastic（人不会每次都做"最优"动作）。DreamDojo-HV有44k hours、6,015 skills、1,135k scenes——比之前最大的robot dataset大15×（duration）、96×（skills）、2000×（scenes）。

**Intuition**：尽管human hand和robot gripper长得很不一样（embodiment gap），但"杯子被推会倒"、"球会滚"、"纸会被揉皱"这些physics invariant。World model本质上要学的不是"我这个gripper怎么动"，而是"物体在接触力下如何形变和位移"。这个physics knowledge是embodiment-agnostic的。

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────┐
│  Phase 1: Pretrain from Human Videos                     │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────┐    │
│  │ Human    │ -> │ Latent      │ -> │ Cosmos-      │    │
│  │ Video    │    │ Action VAE  │    │ Predict2.5   │    │
│  │ (44k hr) │    │ (proxy â_t) │    │ (DiT 2B/14B) │    │
│  └──────────┘    └─────────────┘    └──────────────┘    │
│                         ↓ condition                        │
│  Phase 2: Post-train on Target Robot (G1/GR-1/AgiBot)    │
│  Reset action MLP first layer, finetune all weights       │
│                         ↓                                 │
│  Phase 3: Distillation (Self-Forcing)                     │
│  Bidirectional attention → Causal attention                │
│  50 denoising steps → 4 steps                             │
│  → 10.81 FPS real-time                                    │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Latent Action Model（核心创新点）

这是整篇paper最巧妙的部分。问题：44k hours的human video**没有action label**。HaMeR这类hand pose estimator可以提取，但有三个issue：
- 只capture hand，capture不到arm movement、locomotion
- Heavy occlusion下失败率高
- Low-level hand feature无法跨embodiment transfer（hand 21 DOF vs gripper 7 DOF gap太大）

**Latent action的解决方案**：用VAE从consecutive frames $f^{t:t+1}$ 自监督提取action embedding $\hat{a}_t$。

VAE的objective（Eq. 3）：
$$
\mathcal{L}_{\theta,\phi}^{pred}(f^{t+1}) = \mathbb{E}_{q_\phi(\hat{a}|f^{t:t+1})} \log p_\theta(f^{t+1}|\hat{a}, f^t) - \beta D_{KL}(q_\phi(\hat{a}|f^{t:t+1}) \| p(\hat{a}))
$$

变量含义：
- $\phi$：encoder参数，$\theta$：decoder参数
- $f^t, f^{t+1}$：consecutive video frames
- $\hat{a} \in \mathbb{R}^{32}$：latent action embedding（32维，bottleneck很紧）
- $\beta = 10^{-6}$：KL weight，故意设很小以保留representation capacity
- $p(\hat{a})$：prior，标准正态

**Information bottleneck的关键insight**：要让decoder只看 $f^t$ + $\hat{a}$ 就重建出 $f^{t+1}$，且 $\hat{a}$ 只有32维——VAE被迫disentangle出"帧间变化的最关键信息"，也就是motion/action。static context（背景、光照）已经在 $f^t$ 里，不需要塞进 $\hat{a}$。这就是为什么Figure 3右半部分能cross-embodiment retrieval：human抓杯子和robot抓杯子的latent action embedding会非常接近。

**架构**：spatiotemporal Transformer（700M params），24 encoder blocks + 24 decoder blocks，batch size 256，400k steps训练。

---

## 4. World Model架构设计

基于Cosmos-Predict2.5（latent video diffusion），用WAN2.2 tokenizer把pixel压成latent（temporal compression ratio = 4，即4 frames → 1 latent frame）。两个关键设计：

### 4.1 Relative Action Transformation

不用absolute joint pose，而是rebaseline到每个latent frame起点的pose：
$$
a_t^{rel} = a_t - a_{t_0^{chunk}}
$$

**Intuition**：不同trajectory的absolute pose差异巨大（起点位置不同），但relative action分布narrow得多——"往前伸5cm"在任何起点都是"往前伸5cm"。这降低了modeling complexity，让compositional generalization更容易。

### 4.2 Chunked Action Injection

由于WAN2.2 tokenizer temporal ratio = 4，每个latent frame对应4个pixel frames。作者把4个consecutive actions $a^{i:i+4}$ concatenate成chunk，inject到对应的latent frame，而不是把整条action trajectory作为global condition。

**Intuition**：causality问题。如果给第5帧latent看第10帧的action，模型会学到"未来action影响过去"这种spurious correlation，因为diffusion model是bidirectional的。Chunked injection强制每个latent frame只看属于它时间窗口的action，把causality作为inductive bias硬编码进去。

### 4.3 Temporal Consistency Loss（Eq. 4）

Standard flow matching loss逐帧独立supervise：
$$
\mathcal{L}_{flow}(\theta) = \mathbb{E}_{x,\epsilon,c,t} \|u(x_t, t, c; \theta) - v_t\|^2
$$

其中 $v_t = \epsilon - x$ 是target velocity（noise到clean sample的方向）。

但这样忽略了frame间的temporal correlation。作者加了temporal consistency loss：
$$
\mathcal{L}_{temporal}(\theta) = \mathbb{E}\left[\sum_{i=1}^{K-1} \|(z^{i+1} - z^i) - (v^{i+1} - v^i)\|^2\right]
$$

变量：
- $K$：video latent总长度
- $z^i = u(x_t, t, c; \theta)$：第 $i$ 帧的predicted velocity
- $v^i$：第 $i$ 帧的ground-truth velocity

**Intuition**：让predicted velocity的**frame间差分**等于ground-truth velocity的frame间差分。这相当于约束"运动趋势"——如果ground truth是匀速移动，prediction也要匀速。直接supervise derivative，比supervise absolute value更高效学dynamics。

Final loss：
$$
\mathcal{L}_{final} = \mathcal{L}_{flow} + \lambda \mathcal{L}_{temporal}, \quad \lambda = 0.1
$$

---

## 5. 训练细节

### 5.1 Pretrain
- 初始化：Cosmos-Predict2.5（已经在web-scale video上pretrain过）
- 数据mixture：In-lab : EgoDex : DreamDojo-HV = 1:2:10
- Resolution：640×480，sequence length 13
- 140k steps，effective batch 1024，256 H100 GPUs
- AdamW，weight decay 0.1，lr 1.6e-4
- EMA throughout training
- **关键trick**：action MLP last layer zero-init，避免训练初期perturb pretrained weights

### 5.2 Post-train
- Target robot video ~10Hz
- Sequence 13 frames，1 condition frame + 12 action chunks
- 50k steps，batch 512，128 H100
- Reset action MLP first layer（因为从latent action空间转到真实robot joint space）

---

## 6. Distillation Pipeline（Self-Forcing）

Diffusion model实时推理有两个瓶颈：
1. Bidirectional attention固定horizon
2. 50 denoising steps太慢

**解决方案**：两阶段distillation

### Stage 1: Warmup（Eq. 6）
$$
\mathcal{L}_{warmup}(G_{teacher}, G_{student}) = \mathbb{E}_{x,t} \|G_{student}(x_t, t) - x_0\|^2
$$

$x_0$ 来自teacher的ODE trajectory。Student用teacher forcing（context来自teacher生成的latent）。

### Stage 2: Distillation（Eq. 7-8）
关键：student用自己的previous output做context，不再用teacher forcing。这aligns训练分布和inference分布，减少compounding error。

用KL divergence-based distribution matching：
$$
\nabla \mathcal{L}_{distill} = -\mathbb{E}_{z,t}\left[(s_{real}(x_t,t) - s_{fake}(x_t,t)) \frac{dG_{student}}{d\theta}\right]
$$

变量：
- $s_{real}$：teacher作为real score estimator
- $s_{fake}$：在student output上训练的fake score estimator
- $z \sim \mathcal{N}(0, I)$：noise

**Long horizon trick**：让student生成 $N' > N$ 帧（N是teacher horizon），但只在随机选的N帧window上算loss。这simulates longer rollout，进一步缩小train-test gap。

**结果**（Table 6）：teacher 2.72 FPS → student 10.81 FPS（4×加速），PSNR从14.086降到13.146（acceptable degradation），且student的autoregressive architecture带来bonus：能处理occlusion和camera shift（teacher只能single-frame conditioning，做不到）。

---

## 7. 实验结果深度解析

### 7.1 Latent Action vs 其他conditioning（Table 2）

| Method | In-lab PSNR | EgoDex PSNR |
|--------|------------|-------------|
| w/o pretrain | 20.576 | 19.952 |
| action-free | 20.797 | 19.924 |
| **latent action** | 20.913 | 20.344 |
| retargeted action (ideal) | 20.960 | 20.474 (MANO) |

**Intuition**：action-free pretrain几乎没用（甚至EgoDex上比w/o pretrain还差！），因为passive prediction没学到causality。Latent action已经非常接近"ideal setup"（有Manus glove/Vision Pro精确action label）。这意味着latent action是**scalable的替代**——不需要额外硬件就能从internet video学到action knowledge。

### 7.2 Data mixture ablation（Table 3）

逐个加dataset：
- In-lab only: 18.621 PSNR (DreamDojo-HV eval)
- +EgoDex: 18.706
- +DreamDojo-HV: 18.724
- **DreamDojo-14B: 18.924**

每个增量都有positive contribution，验证scale + diversity的value。

### 7.3 Architecture ablation（Table 5）

| relative | chunked | temporal | GR-1 PSNR | Counterfactual PSNR |
|----------|---------|----------|-----------|---------------------|
| ✗ | ✗ | ✗ | 16.199 | 19.448 |
| ✓ | ✗ | ✗ | 16.522 | 19.482 |
| ✓ | ✓ | ✗ | 17.626 | 20.783 |
| ✓ | ✓ | ✓ | **17.630** | **20.980** |

Chunked injection贡献最大（+1.1 PSNR on GR-1）。Temporal loss在expert trajectory上提升不大，但在counterfactual上提升明显（+0.2）——因为它强化了dynamics learning，让模型对unseen action更robust。

### 7.4 Downstream: Policy Evaluation（Fig 5a）

AgiBot fruit packing task，20 scenes，每个scene 80秒rollout。
- Pearson r = 0.995（DreamDojo success rate vs real-world success rate）
- MMRV = 0.003（rank consistency）

**这是strong evidence**：DreamDojo可以作为reliable simulator，不需要real-world deployment就能评估policy quality。Limitation：absolute success rate在DreamDojo上偏高（model倾向于"乐观"生成成功trajectory），作者指出这是future work要解决的nuanced failure generation问题。

### 7.5 Downstream: Model-based Planning（Fig 5b）

Ensemble 5个policy checkpoint → generate action proposals → DreamDojo predict future → value model（DINOv2-based）选最优 → execute。

- 高variance policy group：比best checkpoint提升17%
- 比uniform sampling提升**~2×**

**Intuition**：world model让policy能"look ahead"。相当于policy在test time做Monte Carlo Tree Search的简化版——不需要real environment，用learned simulator roll out几个候选action，挑最有前景的。

### 7.6 Live Teleoperation（Fig 6）

Distilled DreamDojo-2B在RTX 5090上跑10.81 FPS，PICO VR controller input → virtual G1 robot。这是real-time interactive world model的demo。

---

## 8. Limitations & 我的思考

作者自己指出：
- Uncommon action（slapping、fast waving）效果不好——training data分布问题
- Policy evaluation时absolute success rate偏高——model对failure mode建模不够
- 不支持multi-view（GR00T N1.5等SOTA policy需要multi-view）
- Inference speed还有engineering优化空间

**我的intuition补充**：

1. **Latent action的真正威力**：它解决了"如何从unlabeled video学action-conditioned dynamics"这个fundamental problem。传统做法要么ignore action（action-free pretrain，效果差），要么hand-design retargeting（不scalable）。Latent action是self-supervised的，可以scale到internet video。这是这条路线的"正确性"。

2. **为什么human video有效**：不是因为human和robot长得像，而是因为**world model学的不是agent，而是world**。Cup掉到地上碎了，不管谁推的，碎的方式是一样的。World model的essence是physics engine，agent只是physics的perturbation source。

3. **Distillation的bonus**：autoregressive causal model除了speed，还有context modeling能力——能recover from occlusion。这是bidirectional model结构性无法做到的。这暗示future world model可能从一开始就该是causal的，而不是bidirectional然后distill。

4. **Potential direction**：value model用DINOv2 features有点ad hoc。如果world model本身能输出value（比如把value head attached到DiT），end-to-end训练可能更好。这也是JEPA-style world model的优势——直接在latent space预测，省去video generation的"rendering cost"。

5. **Scale的故事**：44k hours听起来大，但对比internet video（YouTube每天上传几十万hours），还差几个数量级。如果scale到million hours级别，可能涌现更强的physics reasoning能力——类似LLM的emergent ability。

---

## 9. Reference Links

- **Project page**: https://dreamdojo-world.github.io
- **Cosmos-Predict2.5**（base model）: https://arxiv.org/abs/2511.00062
- **WAN2.2 tokenizer**: https://arxiv.org/abs/2503.20314
- **Self Forcing**（distillation方法）: https://arxiv.org/abs/2506.14218（NeurIPS 2025）
- **AdaWorld**（latent action前作，同一作者Gao et al.）: https://arxiv.org/abs/2506.03865（ICML 2025）
- **EgoDex**: https://arxiv.org/abs/2505.11709
- **GR00T N1.5**（policy evaluation用的）: https://arxiv.org/abs/2503.14734
- **Flow Matching**（Lipman et al.）: https://arxiv.org/abs/2210.02747
- **DiT**（Peebles & Xie）: https://arxiv.org/abs/2212.09748
- **Distribution Matching Distillation**（Yin et al.）: https://arxiv.org/abs/2406.05981

---

总结一句：DreamDojo的核心贡献是把**"human video pretrain + latent action proxy + robot post-train"**这条路线work out了，并且scale到44k hours验证了effectiveness。它不是单一技术突破，而是把多个piece（latent action VAE、relative/chunked action injection、temporal consistency loss、Self-Forcing distillation）组合成一个完整pipeline。下一步的obvious方向是继续scale data、scale model、scale action diversity，看physics understanding能否涌现出zero-shot transfer到完全unseen的task family。
