---
source_pdf: LaST0 Latent Spatio-Temporal Chain-of-Thought for Robotic Vision-Language-Action
  Model.pdf
paper_sha256: 6d5a954e08cba350515805501811771076d0714dad702304a57a13d98902f520
processed_at: '2026-08-05T12:05:18-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 LaST<sub>0</sub>

## 一句话说清楚

给机器人装了两个脑子：一个慢的负责"想"，一个快的负责"做"。想的时候不用语言、不用画图，在 latent space 里直接 roll out 未来几秒会发生什么物理变化，然后快的脑子每一步都偷偷瞄一眼这个 latent plan，闭着眼也能跟上。

---

## 1. 为什么要搞这个

先看之前的 VLA 怎么做 reasoning 的：

**第一派：让机器人说话**。CoT-VLA、ECoT 这些，先让 LLM 吐一段文字 "我应该先拿起铲子，再靠近鸡蛋，最后放到面包上"，然后根据这段话生成 action。

问题来了——"鸡蛋距离铲子尖端 3mm，法向力 0.5N，gripper 开口 2cm" 这种信息你用语言说得清楚吗？说不清楚。语言天生就是给人类交流用的 abstraction，物理世界的 fine-grained 细节它表达不了。这叫 **representational bottleneck**。

**第二派：让机器人脑补未来画面**。DreamVLA、WorldVLA 这些，让 model 先 generate 一张未来场景的 RGB image，再根据这张图做 action。

问题更直接——generate 一张图要 decode 几百个 pixel token，在 RTX 4090 上跑出 1.1 Hz 的频率。你 grab 一个鸡蛋的接触瞬间需要 20+ Hz 的 control，1.1 Hz 等于每秒只能决策一次，鸡蛋早飞了。这叫 **latency bottleneck**。

LaST<sub>0</sub> 的核心 insight 特别简单：**reasoning 和 action 本来就不该用一个频率跑**。Reasoning 是低频的、需要整合多模态信息、需要预测未来的；action 是高频的、需要对当前 observation 做快速反应的。这两个 computation pattern 在 frequency domain 上完全 separable，硬塞进一个 autoregressive stream 就是自找麻烦。

---

## 2. Latent CoT 到底是什么

想象你在开车，你不会每秒都在脑子里念叨 "前方 50 米有行人，我要减速，方向盘右转 3 度"。你的大脑在做的事情是：在某个 latent representation 里维持一个对未来几秒的物理 prediction——行人会走到哪、车会到哪、路会怎样——然后你的手脚每毫秒都在 respond 这个 latent prediction。

LaST<sub>0</sub> 就是让机器人这么干。

具体来说，对未来 4 个 keyframe，每个 keyframe 压三个东西成一个 token：

- **视觉 semantic**：SigLIP 编码 future image，average pooling 成 1 token
- **3D 几何**：Uni3D 编码 future point cloud，压成 1 token  
- **机器人本体感觉**：future robot state，压成 1 token

4 keyframes × 3 modalities = **12 个 token**。这就是整个 CoT。

对比一下：CoT-VLA 要 generate 几十个 image token，LaST<sub>0</sub> 只要 12 个 latent token。信息密度高了一个数量级，因为每个 token 都是 pretrained encoder 提取的 high-level feature，不是 raw pixel。

### 序列怎么排

$$\mathcal{Z}_{\mathrm{GT}} = [\mathbf{z}_1^v, \mathbf{z}_1^p, \mathbf{z}_1^s, \mathbf{z}_2^v, \mathbf{z}_2^p, \mathbf{z}_2^s, \ldots, \mathbf{z}_H^v, \mathbf{z}_H^p, \mathbf{z}_H^s]$$

- $\mathbf{z}_k^v$：第 $k$ 个 future keyframe 的视觉 latent token
- $\mathbf{z}_k^p$：第 $k$ 个 future keyframe 的 3D point cloud latent token  
- $\mathbf{z}_k^s$：第 $k$ 个 future keyframe 的 robot state latent token
- 下标 $k \in \{1, \ldots, H\}$，$H=4$ 是 temporal horizon

注意是 interleaved，不是 $[\mathbf{z}_1^v, \mathbf{z}_2^v, \ldots; \mathbf{z}_1^p, \ldots]$。这样排是为了让 positional encoding 上同一时刻的三个 modality 紧挨着，model 天然学到 cross-modal coupling——"这一刻的视觉变化对应这一刻的 3D 变化对应这一刻的 robot state 变化"。

### 为什么 1 个 token 就够

这是 ablation 里最 surprising 的发现：每个 modality 给 1 个 token，success rate 82%；给 2 个、4 个 token，还是 82%。

直觉上这反常识——多给 token 不是能编码更多信息吗？

但实际上 high-level reasoning 不需要 spatial detail。"鸡蛋在铲子上方 3cm" 这个信息，1 个 token 就能编码；你不需要 64 个 token 去描绘鸡蛋的每个像素。Spatial detail 是 fast expert 的 raw observation 该干的事，slow expert 只需要维持一个 **task-relevant 的状态摘要**。

这和 LLM 里 "reasoning token 越多越好" 完全相反。原因是 physical world 的 state space 比 language 的 token space compact 得多——物理世界就那么几个自由度，语言世界有无限的表达空间。

---

## 3. Dual-System MoT 怎么实现的

这是整个架构最聪明的部分。

### 3.1 不是两个分开的 transformer

LaST<sub>0</sub> 基座是 Janus-Pro（DeepSeek-LLM 1.5B, 24 层, $d=2048$）。改造方式是 [Mixture-of-Transformers](https://arxiv.org/abs/2505.14683)：

**所有 non-embedding 组件都分两套**：
- FFN：两套独立权重
- Attention projections $W_Q, W_K, W_V, W_O$：两套独立权重
- LayerNorm：两套独立参数

**但 self-attention 的 global context 是 shared**——所有 token 进同一个 attention 计算，不管它来自 slow 还是 fast expert。

这意味着两个 expert 不是物理分开的，而是 "同一个 transformer 里两套 FFN 和投影权重，token 通过 routing 决定走哪套"。参数效率很高，同时信息流通过 shared attention 保持畅通。

### 3.2 两个 expert 的分工

| | Slow Reasoning Expert | Fast Acting Expert |
|---|---|---|
| 吃什么 | Language + 低频 image | 高频 image |
| 吐什么 | 12 个 latent token | 7-26 DoF action |
| 跑多快 | 每 4 步跑一次 | 每步都跑 |
| 怎么生成 | Autoregressive latent regression | Flow matching |

### 3.3 KV Cache 是速度的关键

这个设计是 15.4 Hz 的秘密：

1. Slow expert 在 keyframe $t$ 跑一次，产生 12 个 latent token，**它的 Key-Value state 被缓存在内存里**。
2. 接下来 3 步，slow expert 完全 dormant，不消耗 compute。
3. Fast expert 每步编码当前 image，通过 shared self-attention **attend 到缓存的 latent CoT KV**——这是 $O(1)$ 检索，直接读 cache，不需要重新跑 slow expert。

实测：slow expert 12.7 Hz，fast expert 22.1 Hz，综合 15.4 Hz。CoT-VLA 是 1.1 Hz，**14× 加速**。

### 3.4 Asynchronous Frequency 训练

这里有个很 elegant 的 trick：SFT 时随机混合 1:1, 1:2, 1:4 的 fast-slow ratio。

效果：
- 固定 1:4 训练，test 1:4：76%
- Mixed ratio 训练，test 1:4：**82%**

为什么 mixed 更好？我的 hypothesis：这相当于 **temporal dropout**——随机让 fast expert 在 "stale latent"（latent CoT 更新延迟了几步）下训练，强迫它学到 robust 的 latent retrieval 而不是 over-fit 到 fixed update pattern。跟 ViT 的 [stochastic depth](https://arxiv.org/abs/1603.09382) 一个道理。

---

## 4. Loss 怎么设计的

两个 loss 联合优化：

### 4.1 Latent CoT loss

$$\mathcal{L}_{\mathrm{latent}} = \sum_{t=1}^{T} \left( 1 - \frac{\hat{\mathbf{z}}_t \cdot \mathbf{z}_t^{\mathrm{GT}}}{\|\hat{\mathbf{z}}_t\| \, \|\mathbf{z}_t^{\mathrm{GT}}\|} \right)$$

- $\hat{\mathbf{z}}_t$：slow expert 在序列位置 $t$ 预测的 latent embedding
- $\mathbf{z}_t^{\mathrm{GT}}$：frozen encoder（SigLIP/Uni3D/tokenizer）提取的 ground-truth latent
- $T = 3H = 12$ 是序列长度

用 cosine similarity 而不是 L2，因为 frozen encoder 的 embedding space 本来就是 contrastive pretrained 的，scale-agnostic。L2 会浪费 capacity 去匹配 embedding 的 norm，而 norm 在这里没物理意义。Cosine 只约束方向，让 model 学到 "未来物理状态在 representation manifold 上的方向"。

### 4.2 Flow Matching loss

Fast expert 用 [π0 style flow matching](https://arxiv.org/abs/2410.24164)：

$$\mathcal{L}_{\text{flow}} = \mathbb{E}_{t, \mathbf{a}_0, \epsilon} \left[ \| \mathbf{v}_\theta(\mathbf{a}_t, t, \text{context}) - (\mathbf{a}_1 - \mathbf{a}_0) \|^2 \right]$$

- $\mathbf{a}_0$：noise（起点）
- $\mathbf{a}_1$：ground-truth action（终点）
- $t \in [0, 1]$：flow matching timestep
- $\mathbf{a}_t = (1-t)\mathbf{a}_0 + t\mathbf{a}_1$：线性插值
- $\mathbf{v}_\theta$：model 预测的 flow velocity field
- context：通过 shared attention 检索到的 latent CoT tokens

总 loss：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{latent}} + \mathcal{L}_{\text{flow}}$$

---

## 5. 训练流程

### Stage 1: Pretraining
400K trajectories from [Open-X-Embodiment](https://arxiv.org/abs/2310.08864), [DROID](https://droid-dataset.github.io/), [RoboMIND](https://arxiv.org/abs/2512.24653)。

这里有个很 clever 的 trick：很多 pretraining 数据没有 depth sensor，没法提取 point cloud。他们用 [VGGT](https://arxiv.org/abs/2503.17351) 从 2D image 合成 3D point cloud，让 slow expert 从 pretraining 阶段就有 geometric awareness。这样 fine-tune 时切到真 point cloud 就 seamless transition。

### Stage 2: SFT
联合训练两个 expert，mixed fast-slow ratio。8×A800，300 epochs。

---

## 6. 实验数据说了什么

### 6.1 Simulation (RLBench, 10 tasks)

| Model | Success Rate | Speed |
|---|---|---|
| OpenVLA | 40% | 6.3 Hz |
| SpatialVLA | 46% | 7.9 Hz |
| CogACT | 61% | 9.8 Hz |
| CoT-VLA | 66% | **1.1 Hz** |
| π0.5 | 65% | 13.8 Hz |
| HybridVLA | 74% | 6.1 Hz |
| **LaST<sub>0</sub>** | **82%** | **15.4 Hz** |

通常 accuracy 和 speed 是 trade-off，LaST<sub>0</sub> 同时拿下两个第一。对比同 backbone 的 CoT-VLA：+16% accuracy，14× speed。这是纯架构红利。

### 6.2 Real-world

| Platform | LaST<sub>0</sub> | π0.5 | Δ |
|---|---|---|---|
| Franka (6 tasks) | 72% | 59% | +13% |
| Mobile (2 tasks) | 47% | 47% | +14%* |
| Dexterous (2 tasks) | 73% | 60% | +14% |

### 6.3 Long-horizon（最能说明问题的实验）

"放鸡蛋到面包上" 连续做 3 次：

| Step | LaST<sub>0</sub> | π0.5 |
|---|---|---|
| 1 | 66% | 47% |
| 2 | 47% | 20% |
| 3 | 33% | 7% |

**Gap 随 horizon 增大而扩大**。Explicit CoT 每次重新生成会 drift，latent CoT 通过 shared attention 持续 condition fast expert，task progress 的 representation 保持 coherent。这是 temporally consistent latent reasoning 的直接证据。

### 6.4 Attention heatmap

这是最能 build intuition 的可视化：

- **No CoT**：注意力散在背景纹理上
- **Explicit CoT (CoT-VLA)**：注意力在语言相关区域，但 miss 掉 robot-object interaction
- **LaST CoT**：高度集中在 robot arm 和 object 的接触区域

Latent space 学到的不是语言概念，是 **物理交互的几何焦点**。

---

## 7. 几个值得深挖的设计直觉

### 7.1 为什么 latent CoT 比 explicit CoT 适合 robotics

Language 是人类为了人类交流发明的 abstraction 层。物理世界的状态——3D 几何、接触力、关节角度、速度——本来就有更自然的 representation：连续向量。强行把它 tokenize 成语言，是在做一次有损翻译，翻译完还得翻译回来。

Latent CoT 跳过了这两次翻译，直接在物理状态的 native representation 里做 reasoning。

### 7.2 为什么 dual-system 打破了 accuracy-speed trade-off

传统 VLA 的 trade-off 长这样：

```
More reasoning tokens → higher accuracy but slower
Fewer reasoning tokens → faster but lower accuracy
```

LaST<sub>0</sub> 的解法：reasoning 和 action 用不同频率跑。Reasoning 慢没关系，因为它的输出被 cached 了，fast expert 每步都能 $O(1)$ 检索。相当于你花 4 步的时间想了一个 plan，接下来 4 步每步都能用到这个 plan，平均下来推理成本被摊薄了 4 倍。

### 7.3 为什么 1 token per modality 就饱和

这暗示了一个深层 intuition：**robotic reasoning 的 information bottleneck 远比想象中窄**。

Physical world 的 state space 是 compact 的——一个抓取任务的关键信息可能就几十个浮点数（物体位姿、接触状态、gripper 开口）。你不需要 language 那种 open-ended 的表达空间。

Capacity 应该花在 fast expert 的 closed-loop perception-action coupling 上，不是花在 slow expert 的 elaborate planning 上。这和 LLM 的 scaling intuition 完全不同。

### 7.4 Uni3D training-only 的 distillation 思路

Uni3D 只在训练时用，推理时不需要 point cloud input。这是个 distillation 设计：用 3D encoder 在训练时提供 supervision signal，让 slow expert 学会从 2D image 隐式 infer 3D 结构。

好处：部署时不需要 depth sensor，硬件成本低。  
风险：如果 deployment 的 3D 结构超出 pretraining distribution（透明物体、镜面），slow expert 可能 generate 错误的 $\mathbf{z}^p$。Paper 没 stress test 这一点。

### 7.5 Cosine loss 的潜在 collapse 风险

Cosine similarity loss 理论上允许 model collapse 到 trivial solution（所有预测指向同一方向）。Paper 没讨论防 collapse 机制。我怀疑之所以 work，是因为：
- Multi-modal interleaved 结构提供 strong positional constraint
- Frozen encoder 的 embedding space 有良好几何结构
- Autoregressive generation 强制 temporal dependency

Scale up 时可能需要加 contrastive 或 EMA teacher。

---

## 8. 一句话总结

LaST<sub>0</sub> 告诉我们：**robotic reasoning 应该在物理 grounding 的 latent space 里做 temporal rollout，配合 dual-system 把低频 reasoning 和高频 action 解耦**。1 token per modality 就够，说明 physical world 的 reasoning bottleneck 比 language 窄得多。Latent CoT + KV cache 让 reasoning 成本被摊薄到几乎免费，同时保持 temporally consistent 的 task representation——这就是 long-horizon task 上 gap 随 horizon 扩大的根本原因。

---

参考链接：
- [Project page](https://vla-last0.github.io/)
- [Janus-Pro](https://arxiv.org/abs/2501.17811)
- [π0 flow matching](https://arxiv.org/abs/2410.24164)
- [Coconut latent CoT](https://arxiv.org/abs/2412.06769)
- [Mixture-of-Transformers](https://arxiv.org/abs/2505.14683)
- [VGGT synthetic point cloud](https://arxiv.org/abs/2503.17351)
- [CoT-VLA baseline](https://arxiv.org/abs/2502.07643)

---

# LaST<sub>0</sub>: Latent Spatio-Temporal Chain-of-Thought for Robotic VLA — 深度技术解析

Andrej，这篇 paper 触及了几个我个人非常关心的设计张力：**reasoning 的表征瓶颈**、**inference latency 与 control frequency 的耦合**、以及 **dual-system 在 transformer 内部的实现方式**。我把核心直觉、公式细节、架构权衡和实验数据都拆开讲一下。

---

## 1. Paper 想解决的问题：为什么 explicit CoT 在 robotics 上是 "wrong abstraction"

现有 CoT-VLA 路线（CoT-VLA, VLA-R1, ECoT, DreamVLA, WorldVLA 等）本质上是把 LLM 里的 textual CoT 或 future-image generation 搬到 robotics。问题在于：

### 1.1 Latency 与 control loop 的不匹配
- Explicit CoT 要 autoregressive decode 语言 token 或 future image token。CoT-VLA 在 RTX 4090 上只有 **1.1 Hz**。
- 但 closed-loop manipulation 需要 ≥10 Hz 的 control frequency（抓取接触瞬间甚至要 30+ Hz）。
- 这导致一个根本矛盾：**reasoning 越深，control 越慢；control 越快，reasoning 越浅**。

### 1.2 Linguistic space 的 representational bottleneck
- 语言是 discrete、symbolic、人类-friendly 的，但 robotics 的物理状态是 continuous、geometric、dynamical 的。
- "把鸡蛋放到面包上" 这种 task，关键信息是 spatula tip 相对于 egg bottom 的毫米级 3D 偏移、接触瞬间的法向力方向、gripper 的开合角度——这些是 **ineffable**，语言根本无法 faithful 地表达。
- 用 future image token 也不行：pixel-level reconstruction 太贵，且把 reasoning 绑死在 RGB reconstruction 上，浪费 capacity。

### 1.3 LaST<sub>0</sub> 的核心 move
把 CoT 从 **离散 linguistic/pixel 空间** 移到 **连续 latent 空间**，并且这个 latent 空间是 **物理 grounding 的**：包含 2D semantic (SigLIP)、3D geometric (Uni3D)、proprioceptive state 三个 modality，沿时间轴展开。

参考 latent CoT 在 VLM 领域的工作（[Coconut (Hao et al. 2024)](https://arxiv.org/abs/2412.06769), [MONet (Wang et al. 2025)](https://arxiv.org/abs/2511.21395), [Machine Mental Imagery (Yang et al. 2025)](https://arxiv.org/abs/2506.17218)）以及 embodied 领域的 [LCDrive](https://arxiv.org/abs/2512.10226)、[ThinkAct](https://arxiv.org/abs/2507.16815)。

---

## 2. Latent Spatio-Temporal CoT 的构造细节

### 2.1 多模态 future latent 的提取
对 horizon $H$ 内的每个 future timestep $k \in \{1, \ldots, H\}$，提取三个 modality 的特征：

| Modality | Encoder | 输入 | 输出 token |
|---|---|---|---|
| Visual semantic | SigLIP-Large (frozen) | $I_{t+k} \in \mathbb{R}^{384 \times 384 \times 3}$ | $\mathbf{z}_k^v \in \mathbb{R}^{d_v}$ |
| 3D geometric | Uni3D (frozen, **training only**) | $P_{t+k}$ (1024 points) | $\mathbf{z}_k^p \in \mathbb{R}^{d_p}$ |
| Proprioceptive | Action tokenizer | $\mathbf{s}_{t+k}$ (robot state) | $\mathbf{z}_k^s \in \mathbb{R}^{d_s}$ |

**关键设计：average pooling 把每个 modality 的 feature map 压成 1 个 token**。这是个 aggressive 的信息瓶颈，但 ablation 显示 1 token 已经足够（2 tokens、4 tokens 收益 marginal）。这符合 [ThinkAct](https://arxiv.org/abs/2507.16815) 的直觉：high-level plan 不需要 spatial detail，detail 留给 fast expert 的 raw observation。

### 2.2 Interleaved 序列结构
$$\mathcal{Z}_{\mathrm{GT}} = [\mathbf{z}_1^v, \mathbf{z}_1^p, \mathbf{z}_1^s, \mathbf{z}_2^v, \mathbf{z}_2^p, \mathbf{z}_2^s, \ldots, \mathbf{z}_H^v, \mathbf{z}_H^p, \mathbf{z}_H^s]$$

- 总长度 $3 \times H$，文中 $H=4$，所以 latent CoT 只有 **12 tokens**。
- 对比 CoT-VLA 生成几十个 future image tokens，这是 **>10× 压缩**。
- Interleaved 而不是 grouped（即不是 $[\mathbf{z}_1^v, \mathbf{z}_2^v, \ldots; \mathbf{z}_1^p, \ldots]$），是为了在 positional encoding 上让 model 学到 **per-timestep 的 cross-modal coupling**——同一时刻的视觉、几何、本体感觉应该相互 predict。

### 2.3 训练目标：cosine similarity 而非 MSE
$$\mathcal{L}_{\mathrm{latent}} = \sum_{t=1}^{T} \left( 1 - \frac{\hat{\mathbf{z}}_t \cdot \mathbf{z}_t^{\mathrm{GT}}}{\|\hat{\mathbf{z}}_t\| \, \|\mathbf{z}_t^{\mathrm{GT}}\|} \right)$$

变量含义：
- $\hat{\mathbf{z}}_t$：slow expert 在位置 $t$ 预测的 latent embedding
- $\mathbf{z}_t^{\mathrm{GT}}$：由 frozen encoders 提取的 ground-truth latent（teacher signal）
- 求和 over sequence position $t \in \{1, \ldots, T\}$，$T = 3H$

**为什么用 cosine 而不是 L2？** 我的 intuition：frozen encoder 的 embedding space 是各向异性且 scale-agnostic 的（SigLIP/Uni3D 训练时本来就用 cosine/contrastive loss）。直接 L2 会让 model 浪费 capacity 去匹配 embedding 的 norm，而 norm 在这里没物理意义。Cosine 只约束方向，让 model 学到 "未来物理状态在 representation manifold 上的方向"，这正是 reasoning 需要的。

### 2.4 特殊 token 与 teacher forcing
- 训练时：`<latent_start> [z_GT tokens] <latent_end>`，teacher forcing。
- 推理时：`<latent_start> <latent_pad>×12 <latent_end>`，slow expert autoregressive 填充。
- 注意：这里的 "autoregressive" 是 **连续 latent regression**，不是 discrete token sampling——没有 softmax over vocabulary，直接输出 continuous vector。这点和 [Coconut](https://arxiv.org/abs/2412.06769) 一致。

---

## 3. Dual-System MoT 架构（这是最聪明的部分）

### 3.1 从 Janus-Pro 改造成 MoT
- 基座：[Janus-Pro](https://arxiv.org/abs/2501.17811)（DeepSeek-LLM 1.5B, 24 layers, $d=2048$）。
- 改造方式：参考 [Emerging Properties in Unified Multimodal Pretraining (Deng et al. 2025)](https://arxiv.org/abs/2505.14683) 的 MoT 思路——**所有 non-embedding 组件都 task-specific**：
  - FFN: 两套独立权重
  - Attention projections $W_Q, W_K, W_V, W_O$: 两套独立权重
  - LayerNorm: 两套独立参数
- **但 self-attention 的 global context 是 shared**：所有 token（无论来自 slow 还是 fast expert）进入同一个 attention 计算。

这一点很重要：两个 expert 不是 "两个分开的 transformer"，而是 "同一个 transformer 里两套 FFN/投影，token 通过 routing 决定走哪套"。这是参数效率与功能解耦的折中。

### 3.2 Reasoning expert vs Acting expert 的分工

| 维度 | Slow Reasoning Expert | Fast Acting Expert |
|---|---|---|
| 输入 | Language $l$ + low-freq image $I_{\text{slow}}$ | High-freq image $I_{\text{fast}}$ |
| 输出 | Latent CoT $\mathcal{Z}$ | Action $\mathbf{a}_t$ |
| 频率 | 每 $\kappa$ 步激活一次（$\kappa \in \{2,4,8\}$） | 每步激活 |
| 生成方式 | Autoregressive latent regression | Flow matching |
| 作用 | Capture spatio-temporal dynamics | Responsive closed-loop control |

### 3.3 Asynchronous Frequency 与 KV Cache
这是 inference 速度的关键：

- Slow expert 在 keyframe $t$ ($t \bmod \kappa = 0$) 运行一次，产生 latent CoT，**其 KV state 被缓存**。
- 在中间 $\kappa - 1$ 步，slow expert 完全 dormant。
- Fast expert 每步编码当前 $I_{\text{fast}}$，通过 shared self-attention **attend 到缓存的 latent CoT KV**——这是 $O(1)$ 检索，不需要重新跑 slow expert。

实测速度（RTX 4090, 1:4 ratio）：
- Slow expert: 12.7 Hz
- Fast expert: 22.1 Hz
- 综合: **15.4 Hz**
- 对比 CoT-VLA: 1.1 Hz → **14× speedup**

这个设计让我想到 Kahneman 的 System 1 / System 2，但实现上比 [OneTwoVLA](https://arxiv.org/abs/2505.11917) 或 [RT-H](https://arxiv.org/abs/2403.01823) 更优雅——它把两个 system 放在同一个 attention context 里，而不是用外部接口通信。

### 3.4 Action 生成：Flow Matching
Fast expert 用 [π0 style flow matching](https://arxiv.org/abs/2410.24164)：

输入侧需要两个 MLP：
- **Timestep MLP**：编码 continuous time $t \in [0, 1]$（sinusoidal embedding 初始化），表示 flow matching 的 diffusion timestep。
- **Noised-action MLP**：把 perturbed action $\mathbf{a}_t + \epsilon_t$ 投影到 LLM embedding space。

输出侧：
- **Projector MLP**：把 hidden state 转成 predicted flow velocity field $\mathbf{v}_\theta$。

Flow matching loss $\mathcal{L}_{\text{flow}}$ 是标准的：
$$\mathcal{L}_{\text{flow}} = \mathbb{E}_{t, \mathbf{a}_0, \epsilon} \left[ \| \mathbf{v}_\theta(\mathbf{a}_t, t, \text{context}) - (\mathbf{a}_1 - \mathbf{a}_0) \|^2 \right]$$

其中 $\mathbf{a}_0$ 是 noise，$\mathbf{a}_1$ 是 ground-truth action，$\mathbf{a}_t = (1-t)\mathbf{a}_0 + t\mathbf{a}_1$。Context 包含 latent CoT tokens（通过 shared attention 检索）。

### 3.5 Action space 定义
- Single-arm (Franka FR3): 7-DoF
  - $\Delta x, \Delta y, \Delta z \in \mathbb{R}^3$（relative position）
  - roll, pitch, yaw $\in \mathbb{R}^3$（Euler angles）
  - $g \in \mathbb{R}^1$（gripper open/closed）
- Dual-arm: 14-DoF（拼接两套 7-DoF）
- Mobile (AgileX): 20-DoF = $[\Delta\theta_{1:6}^R, g^R, \Delta\theta_{1:6}^L, g^L, \mathbf{v}_{lin}, \omega_{ang}]$
- Dexterous (TienKung): 26-DoF = $[\Delta\theta_{1:7}^R, \Delta\phi_{1:6}^R, \Delta\theta_{1:7}^L, \Delta\phi_{1:6}^L]$

注意：换 embodiment 时，**只需要重训 noised-action MLP 和 final projector MLP**——这是 MoT + flow matching 架构的可移植性优势。

---

## 4. 训练流程

### 4.1 Stage 1: Large-scale pretraining
- 400K trajectories (28M frames) from [Open-X-Embodiment](https://arxiv.org/abs/2310.08864), [DROID](https://droid-dataset.github.io/), [RoboMIND](https://arxiv.org/abs/2512.24653)。
- 关键 trick：用 [VGGT](https://arxiv.org/abs/2503.17351) 为所有 pretraining frame 合成 3D point cloud，让 slow expert 从一开始就有 geometric awareness，即使原数据没有 depth sensor。
- 这个 stage 只训 action expert（action loss），slow expert frozen（应该是初始化阶段）。

数据配比（Table 3）：
- BridgeV2: 20.93%
- Kuka: 20.22%
- Fractal: 13.67%
- Robo-Net: 11.53%
- Language Table: 7.72%
- BC-Z: 7.54%
- ManiSkill: 5.26%
- DROID: 4.82%
- 其余 < 4%

### 4.2 Stage 2: SFT
联合优化：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{latent}} + \mathcal{L}_{\text{flow}}$$

**关键：mixed fast-slow ratio training**。在 SFT 时，每个 batch 随机混合 1:1, 1:2, 1:4 的 ratio。这让 fast expert 学会处理 "latent CoT 更新延迟 $\kappa$ 步" 的情况，部署时可以自适应选 ratio。

Ablation (Fig. 5d)：
- 1:1: 75%
- 1:2: 79%
- 1:4: 76%
- 1:8: 74%（下降，延迟太大）
- Mixed (test at 1:4): **82%**

注意 mixed 训练比固定 1:4 训练高了 6 个点——这说明 **训练时 exposure diversity 比部署时单一 ratio 更重要**，类似 data augmentation 的效果。

---

## 5. 实验数据深度分析

### 5.1 RLBench Simulation (Table 1)
10 个 task，mean success rate：

| Model | Mean S.R. | Infer. Speed |
|---|---|---|
| OpenVLA | 0.40 ±0.02 | 6.3 Hz |
| SpatialVLA | 0.46 ±0.03 | 7.9 Hz |
| CogACT | 0.61 ±0.04 | 9.8 Hz |
| CoT-VLA | 0.66 ±0.03 | **1.1 Hz** |
| π0.5 | 0.65 ±0.04 | 13.8 Hz |
| HybridVLA | 0.74 ±0.04 | 6.1 Hz |
| **LaST<sub>0</sub>** | **0.82 ±0.03** | **15.4 Hz** |

观察：
1. LaST<sub>0</sub> 同时拿下了 **最高 accuracy + 最高 speed**——通常这是 trade-off，但 latent CoT + dual system 打破了这个 trade-off。
2. 对比同为 Janus-Pro 基座的 CoT-VLA：66% → 82% (+16%)，同时 1.1 Hz → 15.4 Hz (14×)。这是 **同 backbone 下的纯架构红利**，非常 clean 的 comparison。
3. 在 7/10 task 上 SOTA。弱项是 "Sweep to dustpan"（0.80 vs HybridVLA 0.90）和 "Close fridge"（0.85 vs π0.5 1.00）——可能是 fine-grained trajectory task 上 flow matching 不如 diffusion。

### 5.2 Real-world (Table 2)
分四类：

| Platform | LaST<sub>0</sub> | π0.5 | CoT-VLA | SpatialVLA | Δ vs SOTA |
|---|---|---|---|---|---|
| Franka (6 tasks) | 72% | 59% | 50% | 41% | **+13%** |
| Mobile (2 tasks) | 47% | 47% | 33% | - | **+14%** (vs π0.5 同分但其他更高) |
| Dexterous (2 tasks) | 60% / 87% | 53% / 67% | 40% / 53% | - | **+14%** |

Long-horizon "Place egg on bread" 三次连续执行：
- LaST<sub>0</sub>: 0.66 → 0.47 → 0.33
- π0.5: 0.47 → 0.20 → 0.07
- CoT-VLA: 0.33 → 0.13 → 0.07

**Gap 随 horizon 增大而扩大**——这是 temporally consistent latent CoT 的直接证据。Explicit CoT 每次重新生成会 drift，而 latent CoT 通过 shared attention 持续 condition fast expert，保持 task progress 的 coherent representation。

### 5.3 Ablation 核心发现

**(a) Modality 重要性** (Fig. 5a)：
- Image only: 74%
- Point cloud only: 76%
- State only: 75%
- Image + Point: ~78%
- Image + State: ~77%
- All three: **82%**

每个 modality 都贡献独特信息。Point cloud 单独比 image 略高（76 vs 74），验证了 3D geometric reasoning 对 manipulation 的重要性——这也是 [SpatialVLA](https://arxiv.org/abs/2501.15830) 的核心论点。

**(b) Token budget per modality** (Fig. 5b)：
- 0 tokens: 68%
- 1 token: **82%**
- 2 tokens: ~82%
- 4 tokens: ~82%

**1 token 就饱和**。这说明 high-level reasoning 不需要 spatial resolution，只需要 "状态摘要"。这是个很强的信息 bottleneck 发现，呼应了 [ThinkAct](https://arxiv.org/abs/2507.16815) 的 compact latent plan。

**(c) Temporal coverage** (Fig. 5c)：
- 0 steps: 68%
- 4 steps: **82%**
- 5, 6 steps: 平台期

预测 4 个 future keyframe 最优。再多没好处——可能是因为远端 future 不确定度太高，noise 大于 signal。

**(d) Collaboration frequency** (Fig. 5d)：见上文 mixed ratio 分析。

### 5.4 Attention heatmap (Fig. 4, Fig. 10)
- No CoT: 注意力分散到背景纹理
- Explicit CoT (CoT-VLA): 注意力在语言相关区域，但 miss 掉 robot-object interaction
- LaST CoT: **高度集中在 robot arm 与 manipulated object 的接触区域**

这是 latent CoT "physically grounded" 的直接可视化证据——latent space 学到的不是语言概念，而是 **物理交互的几何焦点**。

---

## 6. 我的几点 intuition 与 critique

### 6.1 为什么这个设计 work：信息流解耦
传统 VLA 把 perception → reasoning → action 全塞进一个 autoregressive stream，每一步都要承担所有 computation。LaST<sub>0</sub> 的核心 insight 是：
- **Reasoning 是低频的、deliberative 的**：需要整合多模态、预测未来、维护 task representation。
- **Action 是高频的、reactive 的**：需要 fast closed-loop、对当前 observation 响应。

这两个 computation pattern 在 frequency domain 上是 separable 的，所以架构上也应该 separate。MoT + KV cache 是实现这种 separation 的优雅方式。

### 6.2 Cosine loss 的潜在问题
Cosine similarity loss 有一个隐患：**它允许 model collapse 到 trivial solution**（所有预测都指向同一个方向，cosine 仍可高）。Paper 没讨论这个，但我怀疑之所以 work，是因为：
1. Multi-modal interleaved 结构提供了 strong positional constraint。
2. Frozen encoder 的 embedding space 已经有良好的几何结构（contrastive pretrained）。
3. Autoregressive generation 强制 temporal dependency。

如果未来 work 要 scale up，可能需要加 contrastive 或 EMA teacher 防止 collapse。

### 6.3 Uni3D 仅 training-only 的设计
这是一个 distillation 式设计：用 Uni3D 在训练时提供 3D supervision，但部署时不需 point cloud input。这意味着：
- 推理时 slow expert 必须从 2D image **隐式 infer 3D 结构**。
- 这依赖于 pretraining 时 VGGT 合成 point cloud 的 coverage。

风险：如果 deployment scenario 的 3D 结构超出 pretraining distribution（比如透明物体、镜面），slow expert 可能 generate 错误的 $\mathbf{z}^p$。Paper 的 real-world 实验没特别 stress test 这一点。

### 6.4 与 π0.5 的对比
π0.5 是 [Physical Intelligence 的工作](https://arxiv.org/abs/2504.16054)，也用了 flow matching + VLM backbone。LaST<sub>0</sub> 的优势主要来自：
1. Explicit latent CoT（π0.5 是 implicit 的，没有显式 future prediction）
2. Dual-system 的 frequency decoupling（π0.5 是 single-stream）

但 π0.5 在 "Close fridge" (1.00) 和 "Phone on base" 上更强，可能在某些 task 上 π0.5 的更大 pretraining scale 弥补了架构差异。

### 6.5 为什么 mixed ratio training > fixed ratio
我的 hypothesis：mixed ratio 相当于 **temporal dropout**——随机让 fast expert 在 "stale latent" 下训练，强迫它学到 robust 的 latent retrieval 而不是 over-fit 到 fixed update pattern。这和 [ViT 的 stochastic depth](https://arxiv.org/abs/1603.09382) 异曲同工。

---

## 7. 与相关工作的位置

| 维度 | Explicit CoT (CoT-VLA, ECoT) | Future Image (DreamVLA, WorldVLA) | Latent CoT VLM (Coconut, MONet) | **LaST<sub>0</sub>** |
|---|---|---|---|---|
| Reasoning space | Language | Pixel | Latent (single modality) | **Latent (multi-modal: 2D+3D+state)** |
| Temporal extension | Single step | Single future frame | Single step | **H-step temporal rollout** |
| Physical grounding | Weak | Medium (visual) | Weak | **Strong (geometric + proprio)** |
| Latency | High | High | Medium | **Low (KV cache)** |
| Architecture | Single transformer | Single transformer | Single transformer | **Dual-system MoT** |

LaST<sub>0</sub> 是第一个把 **latent CoT** + **temporal rollout** + **multi-modal physical grounding** + **dual-system frequency decoupling** 四件事同时做对的 robotics VLA。

---

## 8. 参考链接

- **Project page**: [https://vla-last0.github.io/](https://vla-last0.github.io/)
- **Janus-Pro (backbone)**: [https://arxiv.org/abs/2501.17811](https://arxiv.org/abs/2501.17811)
- **π0 / π0.5 (baseline + flow matching)**: [https://arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164), [https://arxiv.org/abs/2504.16054](https://arxiv.org/abs/2504.16054)
- **Coconut (latent CoT in LLM)**: [https://arxiv.org/abs/2412.06769](https://arxiv.org/abs/2412.06769)
- **MONet (latent visual reasoning)**: [https://arxiv.org/abs/2511.21395](https://arxiv.org/abs/2511.21395)
- **CoT-VLA (explicit CoT baseline)**: [https://arxiv.org/abs/2502.07643](https://arxiv.org/abs/2502.07643)
- **HybridVLA**: [https://arxiv.org/abs/2503.10631](https://arxiv.org/abs/2503.10631)
- **SpatialVLA**: [https://arxiv.org/abs/2501.15830](https://arxiv.org/abs/2501.15830)
- **Uni3D (3D encoder)**: [https://arxiv.org/abs/2310.06773](https://arxiv.org/abs/2310.06773)
- **VGGT (synthetic point cloud)**: [https://arxiv.org/abs/2503.17351](https://arxiv.org/abs/2503.17351)
- **LCDrive (latent CoT in driving)**: [https://arxiv.org/abs/2512.10226](https://arxiv.org/abs/2512.10226)
- **ThinkAct**: [https://arxiv.org/abs/2507.16815](https://arxiv.org/abs/2507.16815)
- **Open-X-Embodiment**: [https://arxiv.org/abs/2310.08864](https://arxiv.org/abs/2310.08864)
- **DROID**: [https://droid-dataset.github.io/](https://droid-dataset.github.io/)
- **RoboMIND**: [https://arxiv.org/abs/2512.24653](https://arxiv.org/abs/2512.24653)

---

## TL;DR for building intuition

LaST<sub>0</sub> 的核心 thesis 是：**robotic reasoning 不应该用语言表达，也不应该用像素重建，而应该在物理 grounding 的 latent space 里做 temporal rollout**。配合 dual-system MoT 把 low-freq reasoning 和 high-freq action 解耦，用 KV cache 让 fast expert $O(1)$ 检索 slow expert 的 latent output。结果是：在同 backbone (Janus-Pro) 下比 CoT-VLA 提升 16% success rate 同时 14× speedup，在 10 个 real-world task 上 mean 72%，long-horizon task 上 gap 随 horizon 扩大——这是 temporally consistent latent representation 的 smoking gun。

最值得深挖的 design choice 是：**1 token per modality 就饱和**。这暗示 robotic reasoning 的 information bottleneck 远比想象中窄，capacity 应该花在 fast expert 的 closed-loop 上，而不是 slow expert 的 elaborate planning。这和 LLM 里 "reasoning token 越多越好" 的直觉相反，可能是因为 physical world 的 state space 比 language 的 token space compact 得多。
