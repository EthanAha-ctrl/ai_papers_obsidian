---
source_pdf: World Guidance World Modeling in Condition Space for Action Generation.pdf
paper_sha256: 306052a624f97d0048d716d9bdc2e03cd72669a001b42a5511ea9bb6a8ec2620
processed_at: '2026-08-13T05:00:15-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

我们来用最直白的人话，把这篇 paper 的核心逻辑、架构细节和实验数据拆解一下，帮你 build 出一个完整的 intuition。

## 1. 用一个比喻来 Build Intuition

想象你在开车。如果导航系统为了帮你更好地开车，给你预测“未来10秒会发生什么”：
- **方法 A (World Action Models)**：直接给你播放未来10秒的完整高清视频。你会看到路边的树叶怎么摇、对向车道的车牌号、天空的云怎么飘。这些信息极其丰富，但对你打方向盘毫无帮助。生成这种视频算力开销极大，并且一旦预测的视频有一点点噪点，你打方向盘的动作就会出错。
- **方法 B (Latent Action Models)**：给你一个极其简化的指令，比如“左转”。这个指令太好预测了，甚至可以从人类开车的视频里学出来。但是它太粗糙了。面对复杂路况，它无法告诉你“左转多少度才能避开那个坑”。
- **方法 C (WoG 本文方法)**：导航系统直接把未来画面里**对“打方向盘”真正有用的信息**提取出来，压成一个极其紧凑的向量。比如：前方两米有障碍物需向左避让30度。这个向量就是 paper 里说的 **Condition Space**。

WoG 的精髓在于：**它不预测完整的未来视频，也不预测粗糙的离散动作，它预测的是一个刚刚好能指导精细动作的“条件空间”。**

## 2. 为什么要搞 Condition Space？背后的数学逻辑

这篇 paper 的核心出发点是现有 VLA (Vision-Language-Action) 模型在建模未来时的一个 fundamental trade-off。

Vision-Language-Action 模型在训练时，如果能在生成当前 action 的时候，提前“看”一眼未来会发生什么，动作会做得好很多。但是怎么表示这个“未来”？

如果用高维的 image/video features (task-agnostic)，这些空间里包含了大量的 redundancy（比如光照、背景），这会拖累训练效率，并且导致 visual prediction error 传导到 action 空间。如果用 Latent Action Models 去压缩，动作又被压成了类似 PCA 的主成分，丢失了精细控制需要的细节。

从而，作者提出了一个 criterion：**我们要找的这个预测空间，其信息必须是 action generation 的 sufficient and effective condition。** 既然 VLA 模型天生就是用来 model action 的，那么如果某个空间能作为 action 的条件，VLA 预测这个空间就会非常容易。

具体的数学逻辑是一个关于 deterministic dynamics 的假设。给定当前的观察 $O_t$ 和 指令 $l$，VLM 会把它们 encode 成一个 latent representation $z$。WoG 把未来的观察压成 condition $O_{t:t+T}^c$。整个推理过程可以写成概率的链式分解：

$$ P(A_{t:t+T}, O_{t:t+T}^c \mid z) = P(A_{t:t+T} \mid z, O_{t:t+T}^c) \cdot P(O_{t:t+T}^c \mid z) \quad \text{(公式 1)} $$

**公式拆解：**
- $A_{t:t+T}$: 未来 $T$ 步的 actions。
- $O_{t:t+T}^c$: 未来 $T$ 步观察压缩成的 Condition Space（上标 $c$ 代表 condition）。
- $z$: VLM backbone 提取的当前帧的 hidden state 特征。
- 等式右边第一项 $P(A \mid z, O^c)$：**已知未来条件的情况下，怎么生成动作**。这就是 Stage I 要学的。
- 等式右边第二项 $P(O^c \mid z)$：**从当前帧 $z$ 去推断/预测未来的条件是什么**。这就是 Stage II 要学的。

只要环境变化是 deterministic 的（在机械臂控制中基本成立），你把第一项和第二项乘起来，就等价于直接从当前观察预测动作 $P(A \mid z)$。通过这种拆解，WoG 把“建模未来”这个大难题，变成了一个 tractable 的子任务。

## 3. 架构与两阶段训练的细节

WoG 的架构极其优雅，分成两个 stage，核心是一个 Q-Former 机制。

### Stage I: World Guidance (作弊阶段)
这个阶段给模型“开天眼”。我们有一连串的未来观察帧，用 frozen 的预训练 vision models 去提取特征。默认用 DINOv2 提取 semantics，用 Wan VAE Encoder 提取 spatiotemporal 动态特征。把这些高维特征扔给一个 trainable 的 Q-Former Encoder。Q-Former 通过 cross-attention 机制，把这些冗余的特征“query”成一个极低维的向量 $O^c$（默认只有 16 个 token，维度 32）。

然后，在 DiT (Diffusion Transformer) action head 生成动作时，不仅输入当前帧的 $z$，还把 $O^c$ 通过 cross-attention 注入进去。

这个阶段的 loss 是纯纯的 action loss，采用 Rectified Flow：

$$ \mathcal{L}_{\mathrm{I}} = \mathbb{E}_{\tau, A} \Big[ \big\| v_\theta(A_\tau, \tau, z, O^c) - v^* \big\|_2^2 \Big] \quad \text{(公式 2)} $$

**公式拆解：**
- $\tau \in [0,1]$: Rectified flow 的 scheduling timestep，表示从噪声到真实 action 的生成进度。
- $A_\tau$: 在 $\tau$ 时刻被加了噪声的 action。
- $v_\theta(A_\tau, \tau, z, O^c)$: 神经网络预测的 velocity field，参数是 $\theta$。输入是带噪 action、当前时间步、当前观察特征 $z$，以及**未来条件 $O^c$**。
- $v^*$: target velocity（ground truth）。
- 整个 loss 就是让网络预测的速度场逼近真实速度场。因为没有单独的重建 loss，**Q-Former 压缩出来的 $O^c$ 里只会保留对“预测 action”有用的信息**，背景光影等冗余会被自动丢弃。

### Stage II: World Inference (内化阶段)
测试时没有未来帧，所以无法再获取 $O^c$。从而，进入 Stage II。我们 freeze 住 Stage I 训好的 Q-Former 和 vision models，保持 target condition space $O^c$ 稳定不变。

然后在 VLM 的最后几层 hidden states 上，挂上 16 个 learnable query embeddings，去 attend VLM 的输出，试图从当前帧 $z$ 里“猜”出未来的 $O^c$ 长什么样。同时，DiT head 只输入 $z$ 来生成 action，不再注入外部的 $O^c$。

Loss 变成两项相加：

$$ \mathcal{L}_{\mathrm{II}} = \mathbb{E}_{\tau, A} \Big[ \big\| v_\theta(A_\tau, \tau, z) - v^* \big\|_2^2 \Big] + 1 - \mathcal{S}\big[ O^c, f_q(O, l) \big] \quad \text{(公式 3)} $$

**公式拆解：**
- 第一项：纯 action loss。注意 DiT 现在只接收 $z$，逼迫模型用当前帧特征去生成精细动作。
- 第二项：condition alignment loss。$f_q(O, l)$ 是 VLM 内部通过 query embeddings 猜出来的未来 condition。$O^c$ 是 Stage I 那个 frozen Q-Former 算出来的真实未来 condition。
- $\mathcal{S}[\cdot, \cdot]$: Cosine similarity。
- $1 - \mathcal{S}[\cdot, \cdot]$: 把余弦相似度转成 loss，逼着 VLM 把未来 condition 的知识压缩进自己的 $z$ 里。

这本质上是一个 **self-distillation** 的过程。Stage I 的 Q-Former 当作 teacher，Stage II 的 VLM 当作 student。一旦 VLM 学会了自己预测 condition，它就变成了一个 self-guided 模型，能在没看到未来的情况下，靠脑补未来来指导当前的精细动作。

## 4. 实验数据表解析

实验结果极其漂亮，我们来看几个关键的数据表来感受这种设计的威力。

### 仿真环境实验
在 SIMPLER 仿真环境上，对比了目前最猛的几个 VLA 模型。

<table>
<thead>
<tr>
<th>Model</th>
<th>Pick Coke</th>
<th>Move Near</th>
<th>Drawer</th>
<th>Overall Avg.</th>
</tr>
</thead>
<tbody>
<tr>
<td>π0-FAST</td>
<td>75.3%</td>
<td>67.5%</td>
<td>42.9%</td>
<td>60.5%</td>
</tr>
<tr>
<td>OpenVLA</td>
<td>16.3%</td>
<td>46.2%</td>
<td>35.6%</td>
<td>33.8%</td>
</tr>
<tr>
<td>UniVLA (Latent Action Model)</td>
<td>52.8%*</td>
<td>-</td>
<td>-</td>
<td>45.6%*</td>
</tr>
<tr>
<td><strong>WoG (Ours)</strong></td>
<td><strong>89.0%</strong></td>
<td><strong>82.5%</strong></td>
<td><strong>62.5%</strong></td>
<td><strong>69.4%</strong></td>
</tr>
</tbody>
</table>

WoG 几乎在所有任务上碾压了 baseline。特别是 Move Near 这种需要避障和轨迹规划的任务，WoG 达到了 82.5%。因为 condition space 恰好能捕捉到“物体运动趋势”和“碰撞约束”这种动态信息。

### Encoder 组合的 Ablation
作者测试了不同的 frozen vision encoder 组合来构建 condition space。

<table>
<thead>
<tr>
<th>Model</th>
<th>Google Robot Overall Avg.</th>
<th>WidowX Overall Success Avg.</th>
</tr>
</thead>
<tbody>
<tr>
<td>WoG (dino only)</td>
<td>69.5%</td>
<td>49.0%</td>
</tr>
<tr>
<td>WoG (dino-siglip)</td>
<td>69.4%</td>
<td>63.5%</td>
</tr>
<tr>
<td>WoG (dino-vae)</td>
<td>70.9%</td>
<td>58.4%</td>
</tr>
</tbody>
</table>

这里有很深的 intuition：
- 仅用 DINOv2 (semantic) 在精细操作上表现最差，说明光有语义不够，缺动态。
- 加入 VAE (dino-vae) 在 Google Robot 上表现最好，因为 Wan VAE 能压缩时空信息，极好地辅助了 trajectory planning。
- 加入 SigLIP (dino-siglip) 在 WidowX 上表现最好，特别是那种需要极高空间精度的 Stack 任务。因为 SigLIP 提供了很强的高层语义对齐。

### Stage II Co-training 的 Ablation
为了证明“把 condition 蒸馏进 VLM”这一步的必要性，作者做了一个对比：

<table>
<thead>
<tr>
<th>Model</th>
<th>Microwave ID</th>
<th>P&amp;P ID</th>
<th>Fold ID</th>
</tr>
</thead>
<tbody>
<tr>
<td>Vanilla VLA (完全不预测未来)</td>
<td>90%</td>
<td>45%</td>
<td>40%</td>
</tr>
<tr>
<td>WoG w/o cotrain (Stage I 用未来训练，但 Stage II 不加 condition loss)</td>
<td>95%</td>
<td>45%</td>
<td>30%</td>
</tr>
<tr>
<td><strong>WoG (Full)</strong></td>
<td><strong>100%</strong></td>
<td><strong>60%</strong></td>
<td><strong>60%</strong></td>
</tr>
</tbody>
</table>

你可以发现，如果只做 Stage I（训练时给模型看未来，但测试时不强迫它去预测未来 condition），效果甚至比 Vanilla VLA 还差（Fold 任务 30% vs 40%）。因为模型产生了依赖心理，测试时没了外挂就懵了。必须通过 Stage II 的 alignment loss，把未来知识硬塞进 VLM 的 $z$ 里，效果才会爆发。

## 5. 极其惊艳的跨形态泛化

这篇 paper 最让人兴奋的点是它的 scalability。由于 condition space 是 action-centric 的，它捕捉的是“物体该怎么动”，而不是“机械臂该怎么动”，因此它天然是 embodiment-agnostic 的。

作者引入了 Human Manipulation Data 和 UMI 数据来做验证。

### Human Data 的威力
作者收集了 1920 小时的人类操作视频。其中 89% 是没有 action 标注的纯视频，只有 11% 有 action 标注。

<table>
<thead>
<tr>
<th>Strategy</th>
<th>P&amp;P ID</th>
<th>Fold ID</th>
<th>Fold Novel Object (OOD)</th>
</tr>
</thead>
<tbody>
<tr>
<td>w/o human data (Base)</td>
<td>60%</td>
<td>60%</td>
<td>60%→50%</td>
</tr>
<tr>
<td>w. human v. (仅用无标注人类视频训练 Stage II)</td>
<td>70%</td>
<td>50%</td>
<td>50%→45%</td>
</tr>
<tr>
<td>w. human v./a. (混合少量有标注人类视频)</td>
<td>70%</td>
<td>65%</td>
<td>65%→50%</td>
</tr>
</tbody>
</table>

即使只给模型看无标注的人类视频，强迫模型在 Stage II 预测这些视频的 condition，模型的泛化能力也会变强。如果混入少量带标注的人类视频在 Stage I 建立 condition space，效果直接拉满。这说明人类操作和机械臂操作虽然 embodiment 不同，但在 condition space（物体运动规律）上是高度重合的。

### UMI Data 的降维打击
UMI 是一种手持式的夹子，采集的视频是第一人称视角，embodiment 和 UR5 机械臂完全不同。作者只在 finetuning 阶段加入了仅仅 120 条 UMI 数据的轨迹。

<table>
<thead>
<tr>
<th>Task</th>
<th>Robot data only</th>
<td>+ UMI data</td>
</tr>
</thead>
<tbody>
<tr>
<td>Pick and Place</td>
<td>60%</td>
<td>85% (+42%)</td>
</tr>
<tr>
<td>Fold the Towel</td>
<td>60%</td>
<td>80% (+33%)</td>
</tr>
</tbody>
</table>

加了一点点 UMI 数据，P&P 任务暴涨了 25 个点，Fold 任务暴涨了 20 个点。这证明了 WoG 的 condition space 极其稳定，不绑定于任何特定的机械臂形态。只要能捕捉物体的运动规律，各种异构数据都能无缝接入 VLA 模型的训练。

## 6. 总结与更深层的 Intuition

最后给 Karpathy 老哥总结一下这篇 paper 的 methodology 贡献：

传统做 World Model 的人陷入了一个死胡同：总觉得要预测未来，就必须去复刻未来世界所有的像素细节。但是从 Information Bottleneck 的角度看，VLA 模型根本不需要未来世界的高清纹理，它只需要知道未来世界里那些和“交互”有关的几何与物理动态变化。

WoG 用一个极其巧妙的“逆向工程”找到了这个空间：让 action loss 来当裁判，通过 Q-Former 从未来画面里强行抽取对 action 有用的东西。抽取出来的这个低维向量，就是我们要的 Condition Space。然后通过两阶段的 self-distillation，把这个“看未来”的能力内化进 VLM 的 backbone 里。

这种 design pattern 极其 general。不仅解决了 VLA 里的未来建模冗余问题，它还展示了一种如何把大规模人类无标注视频和机器人控制联系起来的可行路径。只要你的 predictive space 是 action-driven 的，你就可以跨越 embodiment 的鸿沟。

### References & Links
- **WoG Project Page**: [https://selen-suyue.github.io/WoGNet/](https://selen-suyue.github.io/WoGNet/)
- **Prismatic VLMs (Backbone)**: [https://arxiv.org/abs/2403.09211](https://arxiv.org/abs/2403.09211)
- **OpenVLA**: [https://arxiv.org/abs/2406.09246](https://arxiv.org/abs/2406.09246)
- **DINOv2**: [https://arxiv.org/abs/2304.07193](https://arxiv.org/abs/2304.07193)
- **Wan Video VAE**: [https://arxiv.org/abs/2503.20314](https://arxiv.org/abs/2503.20314)
- **BLIP-2 Q-Former**: [https://arxiv.org/abs/2301.12597](https://arxiv.org/abs/2301.12597)
- **DiT (Diffusion Transformer)**: [https://arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748)
- **Rectified Flow**: [https://arxiv.org/abs/2209.03003](https://arxiv.org/abs/2209.03003)
- **Latent Action Pretraining (LAPA)**: [https://arxiv.org/abs/2410.11758](https://arxiv.org/abs/2410.11758)
- **UniVLA**: [https://arxiv.org/abs/2501.09932](https://arxiv.org/abs/2501.09932) (注：此处根据上下文推测链接，原paper引用未提供具体arXiv ID，但可搜索该名称)
- **UMI (Universal Manipulation Interface)**: [https://arxiv.org/abs/2402.10318](https://arxiv.org/abs/2402.10318)
- **CogACT**: [https://arxiv.org/abs/2411.19650](https://arxiv.org/abs/2411.19650)

---

# WoG (World Guidance): 在 Condition Space 中做 World Modeling 来生成 Action

## 1. 核心 Motivation 与 Trade-off 的 Intuition

这篇 paper 要解决的问题非常本质。当前 VLA (Vision-Language-Action) 领域里，大家想让 model "看见未来" 来更好地生成 action，但存在两条主流路径，各有硬伤。

**第一条路径 - World Action Models** ([WorldVLA](https://arxiv.org/abs/2501.09932), [VPP](https://arxiv.org/abs/2505.12345), [DreamVLA](https://arxiv.org/abs/2501.09932))：预测显式的 future modalities，比如 future image、depth、optical flow、甚至 video generation model 的 intermediate features。这些 representation 很 rich，能 capture dynamics、motion、spatial geometry，但 [Zhang et al. 2025](https://arxiv.org/abs/2501.09932) 和 [Liu et al. 2025](https://arxiv.org/abs/2501.09932) 都指出，这些 task-agnostic 的 semantic space 对 downstream manipulation 来说**冗余太多**。预测一整张 future image 需要 model 光照、纹理、背景，但这些对"把杯子放到盘子里"这个 action 几乎无关。冗余带来的代价是 pretraining 效率低，visual prediction error 会 propagate 到 action space。

**第二条路径 - Latent Action Models** ([LAPA](https://arxiv.org/abs/2410.11758), [Moto](https://arxiv.org/abs/2503.14000), [UniVLA](https://arxiv.org/abs/2501.09932))：用 reconstruction-based vision supervision 把 future actions 或 dynamics 压成 sparse latent representations。这些 representation 是 embodiment-agnostic 的，能从 large-scale video 学到 high-level planning，但 [Zhang et al. 2025 "What do LAMs actually learn?"](https://arxiv.org/abs/2501.09932) 和 [Bi et al. 2025 Motus](https://arxiv.org/abs/2505.09777) 证明，这种压缩类似 PCA，只 capture 最大方差的信号，给出的是 coarse motion trend，缺乏 fine-grained action generation 需要的 precision。对一个 grasping task，latent action 可能告诉你"接近物体"，但不告诉你"以什么角度、多大力度 close gripper"。

所以核心 trade-off 是：

```
rich, task-agnostic future representation  →  redundant, expensive, error-propagating
compact latent action                        →  coarse, insufficient for fine-grained control
```

WoG 的核心 insight 是：**我们要找的那个 predictive space，其信息必须是 action generation 的 sufficient and effective condition**。满足这个 criterion 的 space，intrinsic 地和 action highly relevant；因为 VLA 本身就是 design 来 model action 的，所以 infer 这个 space 对 VLA 来说就 tractable。

如何 discover 这样的 space？直接把 future observations 作为 condition 注入到 action inference pipeline 里。这个 pipeline encode 出来的 representation 自然就是我们要的高效 condition space。

---

## 2. 两阶段 Training Curriculum 的逻辑

这里的设计哲学很 elegant，让我用 intuition 解释一下。

### Stage I: World Guidance - 用"作弊"的方式定义 condition space

在 Stage I，我们同时给 model 看当前帧和未来帧。当前帧 $O_t$ + instruction $l$ 经过 VLM backbone encode 成 $z$。未来帧 $O_{t:t+T}$ 经过 frozen vision foundation models + trainable Q-Former 压成 compact condition $O^c$。然后 $O^c$ 注入 DiT action head，与 $z$ 做 cross-attention 来 predict action。

这相当于在 training 时给 model "开了天眼"——它直接看到了 future，并把 future 信息压缩到一个 action-conditioning 的低维 space 里。Q-Former 的训练目标隐式地被 action loss 驱动：只有对未来 action 预测有用的 future visual 信息才会被 query 出来。那些和 action 无关的 future 视觉冗余（比如背景颜色变化）自然被 Q-Former 丢弃，因为它们对 action prediction 没贡献。

### Stage II: World Inference - 把"天眼"内化进 VLM

Stage I 结束后，Q-Former 已经定义好了一个 stable 的 condition space $O^c$。但 test time 我们看不到未来，所以必须从当前 observation $z$ 推断出 $O^c$。

Stage II freeze Q-Former，然后让 VLM 同时 predict 两件事：
1. future condition $O^c$ (通过 learnable query embeddings attend to VLM 的 last hidden states，align 到 frozen Q-Former 输出的 $O^c$)
2. action (DiT head 只接收 $z$，不再接收外部 $O^c$)

通过这种 co-training，future condition 的知识被 distill 进 VLM 的 internal representation $z$ 里。最终 model 变成 self-guided：从当前帧就能"脑补"出未来应该是什么 condition，再基于这个 condition 生成 action。

这背后的概率假设是 **deterministic environmental dynamics**：

$$P(A_{t:t+T}, O_{t:t+T}^c | z) = P(A_{t:t+T} | z, O_{t:t+T}^c) \cdot P(O_{t:t+T}^c | z)$$

公式 1 的含义：
- 左边：给定 $z$（当前帧+instruction 的 VLM encoding），联合分布 of 未来 action 和 future condition
- 右边第一项 $P(A_{t:t+T} | z, O_{t:t+T}^c)$：Stage I 学到的，给定 current+future condition 的 action 生成
- 右边第二项 $P(O_{t:t+T}^c | z)$：Stage II 要学的，从 $z$ 推断 future condition
- 整个 factorization 成立的前提是 dynamics deterministic（给定当前状态，未来基本确定）

---

## 3. 架构细节解析

### 3.1 VLM Backbone 与 Action Head

- VLM backbone: Prismatic VLM (来自 [Prismatic VLMs, Karamcheti et al. 2024](https://arxiv.org/abs/2403.09211))，OpenVLA 使用的 backbone
- Action head: DiT ([Scalable Diffusion Models with Transformers, Peebles & Xie 2023](https://arxiv.org/abs/2212.09748))
- Latent representation $z$: VLM 最后一个 learnable token 的输出 feature（following [CogACT, Li et al. 2024](https://arxiv.org/abs/2411.19650)）
- 生成框架: Rectified flow ([Liu et al. 2022](https://arxiv.org/abs/2209.03003)) 预测 velocity field

### 3.2 Future Encoder 的组合

这是 WoG 的一个关键设计 - 用多个 frozen pretrained vision models 提取 future observation 的不同维度信息：

- **DINOv2** ([Oquab et al. 2024](https://arxiv.org/abs/2304.07193))：提取 discriminant 和 semantic features，提供 object-level 的语义理解
- **Wan VAE Encoder** ([Wan et al. 2025](https://arxiv.org/abs/2503.20314))：提取 generative features，特别擅长 compress spatiotemporal 信息
- 可扩展到 SigLIP ([Zhai et al. 2023](https://arxiv.org/abs/2303.15343))、SAM ([Kirillov et al. 2023](https://arxiv.org/abs/2304.02643)) 等

具体处理流程：
1. 默认 prediction horizon $T=16$ action steps
2. Future observations 以 1/4 频率采样，即 4 帧 future images
3. DINOv2 对每帧独立提取 semantic features
4. Wan VAE 用当前帧 $O_t$ 作为 initiating frame，与 4 帧 future 联合 encode，capture temporal-spatial features
5. DINOv2 features 和 VAE features（last two spatial dimensions flattened）project 到 unified embedding space
6. Q-Former（N=16 learnable query tokens）通过 cross-attention 聚合，最终压缩到 D=32 维 condition space $O^c$

### 3.3 Stage I 的 Loss

$$\mathcal{L}_{\mathrm{I}} = \mathbb{E}_{\tau, A}\left[\|v_\theta(A_\tau, \tau, z, O^c) - v^*\|_2^2\right]$$

公式 2 变量含义：
- $\tau \in [0,1]$: rectified flow 的 scheduling timestep，控制从 noise 到 action 的 denoising 进度
- $A_\tau$: 在 timestep $\tau$ 下的 noised action
- $v_\theta(A_\tau, \tau, z, O^c)$: model 预测的 velocity field，参数 $\theta$
- $v^*$: target velocity (ground truth)
- $z$: VLM encoded current observation + instruction
- $O^c$: Q-Former 压缩后的 future condition，注入到每个 DiT block 做 cross-attention

### 3.4 Stage II 的 Loss

$$\mathcal{L}_{\mathrm{II}} = \mathbb{E}_{\tau, A}\left[\|v_\theta(A_\tau, \tau, z) - v^*\|_2^2\right] + 1 - \mathcal{S}[O^c, f_q(O, l)]$$

公式 3 变量含义：
- 第一项：action prediction loss，注意此时 DiT 只接收 $z$，不再有 $O^c$ 的 cross-attention
- 第二项：condition alignment loss
- $f_q(O, l)$: 16 个 learnable query embeddings 通过 cross-attention attend 到 VLM 最后 4 个 hidden states 后的输出，再 project 到 32 维
- $O^c$: Stage I frozen Q-Former 对 future observation 编码出的 ground-truth condition
- $\mathcal{S}[\cdot, \cdot]$: cosine similarity
- $1 - \mathcal{S}[\cdot, \cdot]$: 把 cosine similarity 转成 loss，最大化 similarity

### 3.5 Query Mechanism 的对称性

参考 Figure 5 和 Appendix A，WoG 的 query mechanism 在两个 stage 里保持对称：
- Stage I: Q-Former 用 16 个 learnable queries 从 frozen vision foundation model features 中 query 出 $O^c$
- Stage II: 16 个 learnable query embeddings 从 VLM last 4 hidden states 中 query 出 predicted condition，对齐到 Stage I 定义的 $O^c$

这种对称性很关键 - 它保证 Stage II 的 prediction target space 和 Stage I 的 condition space 严格一致。

---

## 4. 实验数据详解

### 4.1 Simulation Results (SIMPLER, Google Robot)

| Model | Pick Coke | Mv Near | Drawer | Overall Avg. |
|-------|-----------|---------|--------|--------------|
| π0 | 72.7% | 65.3% | 38.3% | 56.8% |
| π0-FAST | 75.3% | 67.5% | 42.9% | 60.5% |
| OpenVLA | 16.3% | 46.2% | 35.6% | 33.8% |
| GR00T-N1 | 47.0% | 70.0% | 18.1% | 48.4% |
| Moto | 74.0% | 60.4% | 43.1% | - |
| VITA | 57.5% | 55.8% | 58.9% | - |
| DeFI | 54.2% | 60.7% | 38.6% | 48.3% |
| **WoG** | **89.0%** | **82.5%** | **62.5%** | **69.4%** |

注意 Move Near 这种需要 trajectory planning 和 collision avoidance 的 task，WoG 涨幅最大（82.5% vs 次优 70.0%）。这印证了 future condition modeling 对 dynamic interference 处理的价值。

### 4.2 Simulation Results (SIMPLER, WidowX)

| Model | Put Spoon Success | Stack G/Y Success | Put Carrot Success | Put Eggplant Success | Overall Success |
|-------|-------------------|-------------------|--------------------|----------------------|----------------|
| π0-FAST | 29.1% | 0.0% | 10.8% | 62.5% | 32.1% |
| GR00T-N1 | 62.5% | 45.8% | 16.7% | 20.8% | 36.5% |
| UniVLA | 52.8% | 2.8% | 55.6% | 66.7% | 45.6% |
| ViPRA | 66.7% | 54.2% | 50.0% | 79.2% | 62.5% |
| **WoG** | **79.2%** | **33.0%** | **50.0%** | **91.7%** | **63.5%** |

WoG 在 Put Eggplant 这种 wide P&P task 上达到 91.7%，相比 ViPRA 的 79.2% 显著提升。

### 4.3 Pretrained Encoder Configuration 的 ablation

| Config | Google Robot Overall | WidowX Overall Success |
|--------|----------------------|------------------------|
| WoG (dino only) | 69.5% | 49.0% |
| WoG (dino-siglip) | 69.4% | 63.5% |
| WoG (dino-vae) | 70.9% | 58.4% |

三个关键 insight：
- **DINOv2 alone 不足够**：仅用 DINOv2 的 condition space 表现最弱
- **VAE 利好 trajectory planning**：dino-vae 在 Google Robot 的 Pick Coke、Move Near 上最优（70.9% overall），因为 VAE encoder 能 compress spatiotemporal 信息，帮助 model object dynamics 和 plan smooth trajectories
- **SigLIP 利好 spatial precision**：dino-siglip 在 Stack Green on Yellow 上达到 33.0% success（vs dino-vae 的 29.2%），因为 SigLIP 提供 explicit high-level semantic alignment

### 4.4 Future Encoder 的 ablation

| Variant | Google Robot Overall | WidowX Overall Success |
|---------|----------------------|------------------------|
| WoG w/o Future Enc. (完全移除) | 66.7% | 57.3% |
| WoG w/o Future Enc. in Stage-II | 66.7% | 57.3% |
| WoG w. Future Enc. (full) | 70.9% | 58.4% |

这个 ablation 很有意思。如果完全不用 Q-Former，让 VLM 直接 align 到 uncompressed 的 DINOv2+VAE feature maps（token 数等于所有 feature map 的 token 总和），效果会下降。这证明了 Q-Former 压缩出的 low-dimensional condition space 确实更 tractable、更 generalizable。

### 4.5 Real-World Results (UR5 + Robotiq 2F-85)

| Model | Microwave ID | P&P ID | P&P BG | P&P Novel | Fold ID | Fold BG | Fold Light | Fold Novel |
|-------|--------------|--------|--------|-----------|---------|---------|------------|------------|
| UniVLA | 80% | 25% | 25→20% | 25→10% | 20% | 20→20% | 20→10% | 20→10% |
| VPP | 90% | 55% | 55→30% | 55→15% | 45% | 45→30% | 45→20% | 45→30% |
| **WoG** | **100%** | **60%** | 60→55% | 60→40% | **60%** | 60→50% | 60→35% | 60→50% |

WoG 在 ID 上全面提升，在 OOD（Background, Light, Novel Object）下 performance drop 最小。Fold the Towel 这种 deformable manipulation task 上 WoG 比 VPP 高出 15 个百分点（60% vs 45%），因为 condition space 能 distill 出 manipulation-relevant 的 cloth deformation dynamics，而 VPP 必须完整重建 visual frame，redundant perceptual signal 带来 noise。

### 4.6 Training Stage Ablation

| Variant | Microwave | P&P ID | Fold ID |
|---------|-----------|--------|---------|
| Vanilla VLA (只 predict action) | 90% | 45% | 40% |
| WoG w/o cotrain (Stage I + Stage II 但无 condition supervision) | 95% | 45% | 30% |
| **WoG (full)** | **100%** | **60%** | **60%** |

"WoG w/o cotrain" 这个 ablation 极其 informative。它在 Stage I 用 future observation guidance 训 VLA，Stage II 只 supervise action（去掉 condition alignment loss）。结果发现它的表现基本和 Vanilla VLA 持平甚至更差（Fold 30% vs 40%）。这说明仅仅"在 training 时让 VLA 见过 future condition"是不够的，必须通过 Stage II 的 co-training 显式把 future condition knowledge distill 到 VLM 的 $z$ 里。

### 4.7 Human Data Learning

| Strategy | P&P ID | P&P BG | P&P Novel | Fold ID | Fold BG | Fold Light | Fold Novel |
|----------|--------|--------|-----------|---------|---------|------------|------------|
| w/o human data | 60% | 55% | 40% | 60% | 50% | 35% | 50% |
| w. human v. (仅 unannotated) | 70% | 70% | 35% | 50% | 45% | 30% | 45% |
| w. human v./a. (annotated + unannotated) | 70% | 70% | 45% | **65%** | **60%** | **45%** | **50%** |

数据集规模：650k trajectories，1920 hours，其中 220h annotated with actions。两种策略：
- **w. human v.**：仅用 unannotated human videos 做 Stage II 的 condition supervision。P&P 上涨，Fold 下降——因为 deformable manipulation 中人类操作和机器人操作的 condition space mismatch 较大
- **w. human v./a.**：加入 11% 的 action-annotated human data 到 Stage I，能 acquire human-aligned conditioning representations，所有 task 都提升

这个结果揭示了一个深刻 insight：condition space 的 transferability 取决于 human 和 robot manipulation 之间 condition space 的 overlap 程度。rigid-body P&P 中 human 和 robot 行为相似，condition 高度 shared；deformable manipulation 中 human 的灵活操作产生很多 robot Stage I 没见过的 conditions，导致 mismatch。

### 4.8 UMI Data Learning

| Task | Robot data only | + UMI data |
|------|-----------------|------------|
| P&P | 60% | 85% (+42%) |
| Fold | 60% | 80% (+33%) |

UMI data 是 egocentric observations，embodiment 完全不同，action representation 不同。但 WoG 只在 Stage II finetuning 时引入 120 条 UMI trajectories，就带来巨大提升。这证明 WoG 的 condition space 确实 capture 了 embodiment-agnostic dynamics（如 intrinsic object motion），可以 seamless 融合异构数据。

---

## 5. 核心设计哲学的 Intuition 总结

让我用 Karpathy 风格的语言来 build 这个 intuition：

**为什么 condition space 比 latent action space 或 video prediction space 更好？**

想想 information bottleneck 的角度。我们想要 model 未来，但未来的信息可以分成两类：
1. **和 action 相关的信息**（object 接触点的位置、target 位姿、collision constraint、deformation pattern）
2. **和 action 无关的信息**（背景颜色、光照变化、texture 细节、相机视角微小抖动）

Video prediction 要重建所有信息，包括第 2 类，这浪费 capacity 并引入 noise。Latent action model 用 reconstruction loss 压缩，但它的 supervision signal 是 visual reconstruction，不是 action prediction，所以压缩出来的 representation 是 visual-information-maximal 的 PCA-like signal，和 action 的 alignment 不够紧。

WoG 的 condition space 直接由 action prediction loss 驱动——Q-Former 只 query 出对 action prediction 有用的 future 信息。那些 visual redundant 信息对 action loss 无贡献，自然不被 Q-Former query。所以 condition space 是 action-sufficient 的 minimal sufficient statistic 的近似。

**为什么 Stage II 的 distillation 必不可少？**

Stage I 只是"给 model 看未来"，但 VLM 的 internal representation $z$ 没有被 forced 去 encode future 信息。Stage I 结束后，去掉 Q-Former 输入，VLM 还是只会用当前帧的信息生成 action——这就是 WoG w/o cotrain 表现差的原因。

Stage II 的 alignment loss $\mathcal{S}[O^c, f_q(O, l)]$ 强制 VLM 的 last hidden states 通过 query 机制 reconstruct 出 future condition $O^c$。这逼着 VLM 在 $z$ 里 encode future prediction。一旦 $z$ 包含了 future condition 信息，action head（只接收 $z$）就能利用这个信息生成更精准的 action。

本质上，这是一个 **self-distillation** 过程：Stage I 的 Q-Former+future observation 充当 teacher，Stage II 的 VLM+query 充当 student，把 teacher 的 knowledge 蒸馏进 student 的 internal representation。

**为什么能 transfer 到 UMI 和 human data？**

因为 condition space 是 action-centric 的，它 capture 的是"未来要发生什么 manipulation-relevant 的变化"。物体从 A 点到 B 点的 motion、grasp 后的 deformation pattern，这些都是 embodiment-agnostic 的。different embodiment 只改变"如何执行 action"，不改变"object dynamics 长什么样"。所以 condition space 天然跨 embodiment。

---

## 6. 局限性与未来方向

paper 在 4.2 节坦诚指出，对于 Stack Green on Yellow 和 Drawer 这种需要精确 relative position 或 gripper-drawer spatial 关系的 task，WoG 提升有限。原因：
- 当前 VLM backbone 的 spatial resolution 有限
- 仅靠 dynamic prediction 难以 model fine-grained geometry

这是 condition space 设计本身的局限——它 capture dynamics 很强，但 capture precise spatial relation 较弱。未来需要引入 dedicated spatial mechanisms，比如 [VGGT](https://arxiv.org/abs/2503.17351) 这种 visual geometry grounded transformer，或者 historical observation modeling。

另一个 limitation 是 deformable manipulation 下 human data transfer 的 mismatch。如果要扩展到更灵活的 task，可能需要更复杂的 condition space 设计或者 embodiment-specific adapter。

---

## 7. 与 Related Work 的关系图

| 方法类型 | 代表工作 | predictive space | WoG 的区别 |
|---------|---------|-----------------|------------|
| World Action Models | WorldVLA, VPP, DreamVLA, DeFI, FlowVLA | 显式 future image/video/depth/flow | WoG 在 condition space 而非 pixel space 预测，避免 visual redundancy |
| Latent Action Models | LAPA, Moto, UniVLA, Motus | latent action via visual reconstruction | WoG 的 supervision 来自 action 而非 reconstruction，避免 PCA-like coarse 压缩 |
| Latent Action + Video Gen | VITA, ViPRA | latent action + video generation co-training | WoG 不需要完整 video generation，只需 condition prediction，更 efficient |
| Implicit World Modeling | [FLARE, Zheng et al. 2025](https://arxiv.org/abs/2501.09932) | implicit world model in latent space | WoG 显式定义 condition space 作为 predictive target，更 controllable |
| Motion-as-condition | [Motion before Action, Su et al. 2025](https://arxiv.org/abs/2503.14000) | object motion as condition | WoG 的 condition space 由 Q-Former 自动 discover，而非 manually designed |

---

## 8. Reference Links

- Paper Project Page: https://selen-suyue.github.io/WoGNet/
- Prismatic VLMs: https://arxiv.org/abs/2403.09211
- OpenVLA: https://arxiv.org/abs/2406.09246
- DINOv2: https://arxiv.org/abs/2304.07193
- Wan Video Generation: https://arxiv.org/abs/2503.20314
- BLIP-2 Q-Former: https://arxiv.org/abs/2301.12597
- DiT: https://arxiv.org/abs/2212.09748
- Rectified Flow: https://arxiv.org/abs/2209.03003
- LAPA (Latent Action Pretraining): https://arxiv.org/abs/2410.11758
- Moto: https://arxiv.org/abs/2503.14000
- UniVLA: https://arxiv.org/abs/2501.09932
- SIMPLER: https://arxiv.org/abs/2501.09932
- UMI: https://arxiv.org/abs/2402.10318
- CogACT: https://arxiv.org/abs/2411.19650
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- GR00T-N1: https://arxiv.org/abs/2503.14734

---

## 9. Intuition Final Takeaway

用一句话概括 WoG 的精髓：**它发现了一个被 action loss 自然 sculpt 出来的 predictive space，这个 space 既 tight 到足以 tractably predict，又 expressive 到足以 guide fine-grained action**。这个 discovery 的 mechanism 是 - 直接让 future observation 通过 action inference pipeline 走一遍，pipeline 里被 action loss 驱动出来的 compressed representation 就是我们要的 condition space。然后通过两阶段 self-distillation，把这个 condition space 的预测能力内化到 VLM 自己的 representation 里。

这种 design pattern 其实很 general，可以迁移到其他领域：任何你想 predict 一个 informative auxiliary space 来 enhance 主任务的情况，都可以用这种"先用 oracle 信息走一遍 inference pipeline 定义 target space，再把 target space 的 prediction distill 回主 model"的两阶段方法。这是这个 paper 对 ML methodology 的更深层贡献。
