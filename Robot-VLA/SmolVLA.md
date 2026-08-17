---
source_pdf: SmolVLA.pdf
paper_sha256: cb1bb9a8f824187fcdc32af8c290214487c524f139a3c2def1c6d97adb1fa40c
processed_at: '2026-08-12T07:59:16-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SmolVLA 用人话讲

Andrej，这篇 paper 乍一看技术细节很多，剥开来其实就一句话：**怎么让机器人控制模型又小又快又好用**。

---

## 这篇 paper 在解决什么痛点

现在机器人圈有个尴尬：想让机器人聪明，就得用大模型；用了大模型，推理就慢；推理慢了，机器人动作就卡顿。

你看 OpenVLA 有 70 亿参数，π0 有 33 亿参数，训练烧几十万 GPU 小时，部署要高端显卡。普通研究者根本玩不起。

SmolVLA 就想说：我能不能用 4.5 亿参数（比 OpenVLA 小 15 倍），在一块普通 GPU 上训练，在 CPU 上跑，效果还能跟大模型打个平手甚至更好？

答案是能。靠三招：**架构瘦身、众包数据、异步推理**。

---

## 第一招：架构怎么瘦身的

先想象一个 VLA 模型要做的事：看到画面、听到指令、感知手臂位置，然后输出接下来 50 步该怎么动。

SmolVLA 把这个过程拆成两部分：**"看"的部分用现成的小 VLM，"动"的部分用一个专门的 action expert**。

### 看：VLM 只用一半层

SmolVLA 用 SmolVLM-2 做视觉理解。这模型有 30 多层 transformer，SmolVLA 直接砍掉后半截，只用前 16 层。

听起来很暴力，但道理是这样的：VLM 后面的层主要做"这是什么"的抽象判断，机器人不需要那么深的语义理解，它只需要知道"杯子在左边、手在右边"这种 dense 特征。前面的层就够用了，后面的层对它来说是冗余计算。

实测砍一半层，性能只掉 1.8%，但计算量直接减半。这笔买卖很划算。

### 看：图像只留 64 个 token

标准 VLM 处理高分辨率图像，会把图切成很多小块，每个小块一个 token，一张图能搞出几百个 token。

机器人场景没那么复杂，桌面、机械臂、一个方块，用 64 个 token 就够描述了。SmolVLA 用 pixel shuffle 把图压到 64 token/帧。视觉编码的计算量直接降一个数量级。

### 动：Action Expert 的设计

这是 SmolVLA 最有意思的部分。Action expert 是一个小 transformer，专门接收 VLM 的特征，输出 50 步连续动作。

它用 **flow matching** 训练，这个数学上很优雅：训练时把真实动作和噪声做线性混合，让网络学怎么从噪声"流"回真实动作。推理时从纯噪声出发，走 10 步就得到动作。

为什么用 flow matching 不用 regression？因为机器人的动作是**多模态**的——同一个"抓杯子"的指令，从左边抓和从右边抓都对。Regression 只能学一个平均值，flow matching 能学整个分布。

Action expert 内部还有一个设计：**cross-attention 和 self-attention 交替**。

- Cross-attention：让每个动作 token 都去看 VLM 的特征，知道环境长啥样
- Self-attention：让动作 token 之间互相看，保证 50 步动作连贯平滑

两个一起用效果最好，单独用任一个都差一截。这跟 π0 只用 cross-attention 不同，是 SmolVLA 的改进点。

---

## 第二招：数据怎么来的

机器人数据是出了名的难搞。OpenX-Embodiment 集合了 100 万条轨迹，已经是最大规模，但跟 NLP 的万亿 token 比差远了。

SmolVLA 走了条不同寻常路：**用社区数据**。

Hugging Face 的 LeRobot 平台上，有几百个爱好者用 3D 打印的 SO-100 机械臂（几百美元一台）采集数据上传。SmolVLA 挑了 481 个这样的数据集，加起来 2.3 万条轨迹，1060 万帧。

这比 OpenVLA 的数据少 50 倍。但有几个好处：

**多样性高**：不同人、不同房间、不同光照、不同任务。天然的数据增强。

**Embodiment 一致**：都是 SO-100，所以低层控制技能可以直接学，泛化到新任务时基础稳。

但社区数据有个大问题：标注乱七八糟。有的写 "task desc"，有的写 "Hold"，有的干脆没写。

SmolVLA 的解法很 pragmatic：**用另一个 VLM 自动写标注**。拿 Qwen2.5-VL 看几个关键帧，自动生成"Pick up the cube and place it in the box"这种动作描述。一个 VLM 帮另一个 VLM 训练，闭环了。

还有相机视角混乱的问题，不同数据集叫法不一样，SmolVLA 手动把所有相机映射到 top/wrist/side 三类。这种脏活看似 minor，但不做模型学不到稳定的视觉表征。

---

## 第三招：异步推理，最 engineering 的贡献

这部分其实跟模型架构无关，是**部署工程**的优化，但效果惊人。

### 问题背景

现在主流 policy 都输出 action chunk——一次生成 50 步动作。但怎么执行这些动作，有两种朴素做法：

**同步执行**：跑完 50 步再去看新画面。问题是中间 50 步机器人是"闭眼"的，如果中间环境变了，机器人不知道。而且 chunk 之间要等 inference 完成，有 idle lag。

**每步都 inference**：每个控制周期都看新画面、生成新 chunk。问题是计算太贵，每秒 30 次推理谁都受不了。

### SmolVLA 的解法

**解耦"执行"和"推理"**。机器人一边执行 queue 里的动作，一边在后台异步处理新画面。

具体逻辑：

1. 机器人从 queue 里取动作执行
2. 当 queue 剩余动作少于阈值（比如 30%）时，触发新观测
3. 如果新观测跟旧观测差不多（joint space 距离小），就跳过，避免冗余推理
4. 后台异步跑 inference，不阻塞当前执行
5. 新 chunk 生成后，跟旧 chunk 在重叠部分聚合

这个设计的关键是一个阈值 $g$。数学上，只要 $g \geq \frac{\mathbb{E}[\ell_S] / \Delta t}{n}$，队列就不会空，机器人不会卡顿。

意思是：当队列剩余动作能撑的时间 ≥ 一次推理时间，就不会 stall。

### 实测效果

成功率和同步差不多（78% vs 78%），但**速度快 30%**——因为消除了 chunk 之间的 idle lag。

更狠的指标：固定 60 秒内，同步能完成 9 个任务，异步能完成 19 个。**吞吐量翻倍**。

这个招是 model-agnostic 的，任何输出 chunk 的 policy 都能加。我觉得应该成为标配。

---

## 实验结果用一句话讲

SmolVLA 4.5 亿参数，没在机器人数据上预训练，在 LIBERO 上 87.3%，超过有机器人预训练的 π0 3.3B（86.0%）。

真实世界 SO-100 上三个任务平均 78.3%，超过 π0 的 61.7%。

Transfer 到从没见过的 SO-101 机器人上，依然比 ACT 强。

这些数字说明：**架构设计对了，小模型也能打大模型**。

---

## 我的几个直觉

**机器人不需要那么深的 reasoning**。LLM 那种深层次推理在机器人控制里用不上，机器人需要的是 tight perception-action loop。所以砍 layers、减 tokens、用小 action expert 都 work。

**Flow matching 是机器人动作建模的正确范式**。连续控制、多模态分布、推理步数少，三点都满足。Regression 太弱，DDPM 太慢，flow matching 刚好。

**社区数据是机器人研究的未来**。工业数据贵且集中，社区数据便宜且多样。SO-100 这种百美元平台 + LeRobot 平台 + 自动标注，是真正 scalable 的路径。这篇 paper 的 appendix 列了几百个贡献数据的用户，这才是 ecosystem 的力量。

**异步推理是 free lunch**。chunk-based policy 加 async 执行，30% 提速几乎没成本。没理由不用。

**VLM 后期层对机器人是冗余的**。这跟 Bolya 的 Perception Encoder 发现呼应——最好的视觉特征不一定在网络输出层。对机器人这种 dense prediction 任务，中间层反而更好用。

---

## 一句话总结

SmolVLA 证明了：用对架构（layer skipping + flow matching + interleaved attention）+ 用对数据（community-collected + auto-annotated）+ 用对部署（async inference），4.5 亿小模型能干翻 33 亿大模型，还能在 CPU 上跑。

它把 VLA 的门槛从"公司级"降到了"个人开发者级"，这才是这篇 paper 的真正价值。

---

# SmolVLA 深度解读

Andrej, 这篇 paper 我仔细读了, 内容很丰富, 涉及到 architecture 设计、训练 objective、inference 系统、data pipeline 多个层面。我会从直觉到细节逐层展开, 给你构建一个完整的 mental model。

Paper link: https://arxiv.org/abs/2506.08450 (Hugging Face 出品, 作者 Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Remi Cadene 等)

Code: https://github.com/huggingface/lerobot

---

## 1. 核心 motivation: 这篇 paper 想解决什么问题

当前 VLA 领域(OpenVLA, RT-2, π0, RDT-1B)有一个矛盾:
- 要 generalization 好, 就得用大的 pretrained VLM backbone
- 但 robotics 部署要求 fast inference, low latency, low compute

OpenVLA 7B 用 action tokenization(把 continuous action 离散化成 token), inference 慢且 continuous control 受限。π0 用 flow matching + 3.3B PaliGemma, 性能强但训练需要 10k 小时 cross-embodiment data, 工业级资源。SmolVLA 的 thesis 是: **通过 architectural efficiency + community data + async inference engineering, 用 0.45B 参数达到甚至超过 10x 大小的模型**。

这里关键 insight 是 robotics 不同于 LLM, 不需要 super deep reasoning, 而是需要 tight perception-action loop。所以 SmolVLA 大胆 skip layers, 减少 visual tokens, 用 small action expert。

---

## 2. 架构总览: 三段式设计

整体 data flow(对应 Figure 1):

```
[Image(s)] -> SigLIP encoder -> pixel shuffle -> 64 visual tokens/frame
[Language instruction] -> tokenizer -> text tokens
[Sensorimotor state] -> linear projection -> 1 state token
                                        |
                                        v
                          [concatenate] -> SmolVLM-2 decoder (first N=L/2 layers)
                                        |
                                        v
                              VLM features o_t (中间层输出)
                                        |
                                        v
                  Action Expert v_θ (interleaved CA/SA blocks)
                                        |
                                        v
                          Flow matching -> action chunk A_t = (a_t, ..., a_{t+n})
```

三个 input stream 在 VLM 内部 fuse, VLM 的中间层 features 作为 conditioning, 通过 cross-attention 喂给 action expert。Action expert 用 flow matching(类似 diffusion 的连续生成范式)生成 50 步 action chunk。

### 2.1 VLM Backbone: SmolVLM-2

SmolVLM-2 (Marafioti et al., 2025, https://arxiv.org/abs/2504.05299) 本身是一个高效的 multi-image VLM, 由:
- **Vision encoder**: SigLIP (Zhai et al., 2023, https://arxiv.org/abs/2303.15343) — 用 sigmoid loss 训练的 CLIP 变体
- **Language decoder**: SmolLM2 (Allal et al., 2025, https://arxiv.org/abs/2502.02737) — 1.7B params 的小 LLM

选择 SmolVLM-2 的 intuition 是它优化了 multi-image/video 输入, 且 token efficiency 高, 这对 robotics 多相机场景非常合适。

**Visual tokens reduction** 这个设计点很重要:
- 标准 SmolVLM-2 用 image tiling(把图切成多个 crop + global image), 高分辨率但 token 多
- SmolVLA 放弃 tiling, 只用 global image
- 加上 **pixel shuffle** 操作, 把 spatial pixel 重排成 channel, 把每帧压到 64 tokens

直觉上, robotics 场景的图像相对简单(桌面、机械臂、物体), 不需要 OCR/document 级别的细节, 64 tokens 足够表达 "cube 在哪, gripper 在哪"。这个设计直接把 vision encoder 的 FLOPs 降一个数量级。

### 2.2 Layer Skipping: 取前 N=L/2 层

这个设计来自一系列最近的研究(El-Nouby et al., 2024, https://arxiv.org/abs/2401.08541; Bolya et al., 2025, https://arxiv.org/abs/2504.13181; Rajasegaran et al., 2025, https://arxiv.org/abs/2501.05453): **VLM 的 best features for downstream tasks 不一定在最后一层**。

具体做法:
- SmolVLM-2 总共 L 层 LLM decoder
- SmolVLA 直接 **物理丢弃** 后 L-N 层(图里用剪刀 icon 表示)
- 默认 N = L/2, 即取前一半

Ablation Table 8 给出对比:
| 配置 | LIBERO Avg SR |
|---|---|
| N=8 (太浅) | 75.0% |
| N=16 (一半) | 78.5% |
| N=24 | 79.5% |
| N=32 (全部) | 80.3% |
| Skip every 2nd layer | 75.5% |
| 用更小的 VLM-256M | 75.8% |

直觉解读: 用 full layers 只比 half layers 高 1.8%, 但计算量翻倍。**Skip every 2nd layer(Shukor & Cord 2024, https://arxiv.org/abs/2410.09454)反而不如取前 N 层**, 因为 LLM 后期 layers 主要做 task-specific 抽象, 早期 layers 做 general feature extraction, 对 robotics 这种 dense prediction 任务, 早期 features 更通用。

更深层 insight: **用 large VLM 但 skip layers, 比直接用 small VLM 好**。因为大模型的 capacity 在前几层就足够, 而小模型 capacity 不足。这也呼应了 Bolya 2025 的 Perception Encoder 发现: "best visual embeddings are not at the output of the network"。

### 2.3 Action Expert: Flow Matching Transformer

这是 SmolVLA 与 π0 最相似的部分, 但有几个关键差异。

#### Architecture: Interleaved CA + SA

Action expert $\mathbf{v}_\theta$ 是一个 transformer, 包含交替的:
- **Cross-attention (CA) blocks**: action tokens 作为 query, VLM features $\mathbf{o}_t$ 作为 keys/values
- **Self-attention (SA) blocks**: action tokens 之间互相 attend, 用 causal mask

对比 π0(Black et al., 2024, https://arxiv.org/abs/2410.24164): π0 主要用 CA(VLM features 作为 KV)。对比 Bjorck et al. 2025 GR00T N1 (https://arxiv.org/abs/2503.14734): 主要用 SA(concat VLM features 后 self-attend)。

SmolVLA 的 ablation Table 6:
| Attention 机制 | LIBERO Avg SR |
|---|---|
| CA only | 79.0% |
| SA only | 74.5% |
| CA+SA interleaved (SmolVLA) | 85.5% |

CA 和 SA 有互补作用:
- **CA**: 让每个 action token 都能看到 VLM 提供的 environment/instruction context, 提供 grounding
- **SA (causal)**: 让 action chunk 内部有时间一致性, $a_{t+k}$ 可以参考 $a_t, ..., a_{t+k-1}$, 生成更平滑的轨迹

为什么用 **causal mask** 而不是 bidirectional? Table 7:
| Mask | LIBERO Avg SR |
|---|---|
| Bidirectional | 67.5% |
| Causal | 74.5% |

Bidirectional 让未来 action token 泄漏到现在, 训练时学不到真正的 sequential dependency, 推理时退化。

#### Hidden dimension: 0.75×d

Action expert 的 hidden size = 0.75 × d, 其中 d 是 VLM 的 hidden dim。Ablation Table 9:
| Expert width | LIBERO Avg SR |
|---|---|
| ×1.00 | 82.3% |
| ×0.75 (SmolVLA) | 77.5% |
| ×0.50 | 80.3% |
| ×0.25 | 73.8% |

注意 ×0.50 居然比 ×0.75 略好(80.3 vs 77.5), 但 ×0.75 是 production 选择, 推测是 robustness/real-world 上的 tradeoff, paper 里 LIBERO 数据有 noise。

#### Flow Matching 训练 objective

这是 SmolVLA 数学上最 interesting 的部分。Flow matching (Lipman et al., 2022, https://arxiv.org/abs/2210.02727; Liu, 2022, https://arxiv.org/abs/2209.14577; Esser et al., 2024, https://arxiv.org/abs/2406.06275 SD3) 是 diffusion 的表亲, 但更简洁。

**训练 loss**:

$$
\mathcal{L}^\tau(\theta) = \mathbb{E}_{p(\mathbf{A}_t | \mathbf{o}_t), q(\mathbf{A}_t^\tau | \mathbf{A}_t)} \left[ \big\| \mathbf{v}_\theta(\mathbf{A}_t^\tau, \mathbf{o}_t) - \mathbf{u}(\mathbf{A}_t^\tau | \mathbf{A}_t) \big\|^2 \right]
$$

变量逐一解析:
- $\theta$: 神经网络(action expert)参数
- $\mathbf{A}_t = (a_t, a_{t+1}, ..., a_{t+n})$: ground-truth action chunk, 时间步 t 到 t+n 的连续动作序列(n=50)
- $\mathbf{o}_t$: VLM features, 从 observation $o_t$ 在 VLM 第 N 层提取
- $p(\mathbf{A}_t | \mathbf{o}_t)$: 给定 observation 的真实 action 分布(data distribution)
- $q(\mathbf{A}_t^\tau | \mathbf{A}_t)$: 加噪分布
- $\mathbf{A}_t^\tau = \tau \mathbf{A}_t + (1-\tau) \epsilon$: noisy action chunk, 真实 action 和高斯噪声的线性插值
- $\tau \in [0, 1]$: 时间参数, 从 Beta 分布采样(与 π0 一致), 控制 noise level
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$: 标准高斯噪声
- $\mathbf{v}_\theta$: 神经网络预测的 vector field
- $\mathbf{u}(\mathbf{A}_t^\tau | \mathbf{A}_t) = \epsilon - \mathbf{A}_t$: 目标 vector field, conditional flow 的方向

**直觉解释**: Flow matching 训练网络预测一个 vector field, 这个 field 把噪声分布 $\mathcal{N}(0, \mathbf{I})$ 连续地"流"到 data 分布 $p(\mathbf{A}_t | \mathbf{o}_t)$。

- 当 $\tau = 1$: $\mathbf{A}_t^1 = \mathbf{A}_t$ (纯数据), 目标 $\mathbf{u} = \epsilon - \mathbf{A}_t$
- 当 $\tau = 0$: $\mathbf{A}_t^0 = \epsilon$ (纯噪声), 目标 $\mathbf{u} = \epsilon - \mathbf{A}_t$ (但 $\mathbf{A}_t$ 是 conditional 的 ground truth)

注意这里 conditional flow matching 与 DDPM 的关键区别: DDPM 学的是 $\epsilon$-prediction(预测噪声), flow matching 学的是 velocity $\mathbf{u} = \epsilon - \mathbf{A}_t$ (从 noise 到 data 的方向)。数学上, 这对应于概率路径上的 ODE:

$$
\frac{d\mathbf{A}_t^\tau}{d\tau} = \mathbf{u}(\mathbf{A}_t^\tau | \mathbf{A}_t) = \epsilon - \mathbf{A}_t
$$

求解: $\mathbf{A}_t^\tau(\tau) = \tau \mathbf{A}_t + (1-\tau) \epsilon$, 即线性插值路径。这条 path 是 deterministic 的 optimal transport, 比 DDPM 的 stochastic forward process 更直, 因此采样时需要的 step 更少(SmolVLA 用 10 步, DDPM 通常 50-1000 步)。

**推理时**: 从纯噪声 $\mathbf{A}_t^0 = \epsilon$ 出发, 用 Euler method 离散化 ODE:

$$
\mathbf{A}_t^{\tau + \Delta\tau} = \mathbf{A}_t^\tau + \Delta\tau \cdot \mathbf{v}_\theta(\mathbf{A}_t^\tau, \mathbf{o}_t)
$$

10 步即可生成 action chunk。

**为什么 flow matching 比 regression 好**? Table 10:
| Objective | LIBERO Avg SR |
|---|---|
| Flow matching | 80.25% |
| Regression (L1) | 75.25% |

Regression 假设 action distribution 是 unimodal 的(MSE), 但 robotics 中同一个 observation 可以对应多个合理 action(比如从左边或右边抓杯子)。Flow matching 显式建模 multimodal distribution, 通过 vector field 表达 "在每一点应该往哪个方向流"。

类似 ablation 也见于 π0 paper, diffusion policy (Chi et al., 2023, https://arxiv.org/abs/2307.01948)。**Multimodality 是 robotics action modeling 的核心难点**, 这也是为什么 OpenVLA 用 tokenization 的方式效果受限 — 离散 binning 平滑性差, 且无法表达 continuous multimodal。

---

## 3. 异步推理 Stack: 这篇 paper 最 engineering 的贡献

这部分是 SmolVLA 区别于纯 ML 论文的地方, 涉及 control system 设计。

### 3.1 同步 vs 异步: 为什么 chunk inference 有问题

现代 visuomotor policy(ACT, Diffusion Policy, π0)都输出 **action chunk**: 一次 forward pass 生成 $n$ 步 action, 而不是单步。这减少了 inference 频率, 但引入了 **open-loop control** 问题。

两种 naive 策略:
1. **Sync (sequential, $g=0$)**: 执行完整个 chunk 才发新 observation。问题: chunk 执行期间机器人 blind, chunk 间有 idle lag(等 inference 完成)
2. **Reactive ($g=1$)**: 每步都 inference(类似 ACT 原文)。问题: compute 极贵, 1 forward pass per control tick

SmolVLA 的 **async** 思路: **decouple action execution 和 observation processing**。机器人 client 持续从 queue 消费 action, 当 queue 低于阈值 $g$ 时, 异步触发新 observation → inference, 不阻塞当前 action 执行。

### 3.2 算法 1 详细解读

```
Algorithm 1 Asynchronous inference control-loop
Input: horizon T, chunk size n, threshold g ∈ [0,1]
Init: capture o_0; send o_0 to PolicyServer; receive A_0 = π(o_0)
for t to T do
    a_t ← PopFront(A_t)              // 从 queue 取一个 action
    Execute(a_t)                       // 机器人执行
    if |A_t|/n < g then                // queue 低于阈值
        capture new observation o_{t+1}
        if NeedsProcessing(o_{t+1}) then  // 相似性 filter
            async_handle ← AsyncInfer(o_{t+1})  // 非阻塞触发 inference
            Ã_{t+1} = π(o_{t+1})       // 新 chunk 生成
            A_{t+1} = f(A_t, Ã_{t+1})  // 聚合 overlap
        end
    end
    if NotCompleted(async_handle) then
        A_{t+1} ← A_t                  // inference 没完成, 用旧 queue
    end
end
```

关键设计:
- **Threshold $g$**: 当 queue 剩余 < $g \cdot n$ 时触发新 inference。$g=0$ 是 sync, $g=1$ 是 reactive
- **Similarity filter (NeedsProcessing)**: 在 joint-space 算 observation 距离, 距离 < $\epsilon$ 则 drop, 避免 redundant inference
- **非阻塞 AsyncInfer**: inference 在另一个 thread/server 跑, 主 control loop 不等
- **Aggregation function $f$**: 新 chunk 来了, 与旧 chunk 在 overlap 时间步聚合(可能用 weighted average, paper 没细说)

### 3.3 阈值 $g$ 的数学分析

Paper 里给了一个 clean 的分析:

设:
- $\ell$: 总 round-trip latency, $\ell = t_{CS} + \ell_S + t_{SC}$
  - $t_{CS}$: client → server 发 observation 的时间
  - $\ell_S$: server inference 时间
  - $t_{SC}$: server → client 发 action chunk 的时间
- $\Delta t$: 控制周期, 30 fps 时 $\Delta t = 33$ ms
- 假设 $t_{CS} \approx t_{SC}$ 且远小于 $\ell_S$, 则 $\mathbb{E}[\ell] \approx \mathbb{E}[\ell_S]$

**关键不等式**: 避免队列空闲(机器人 stall)的条件:
$$
g \geq \frac{\mathbb{E}[\ell_S] / \Delta t}{n}
$$

直觉: 当 queue 剩 $g \cdot n$ 步 action 时触发 inference, 这些 action 还能撑 $g \cdot n \cdot \Delta t$ 秒。如果这个时间 ≥ inference 时间 $\mathbb{E}[\ell_S]$, 就不会 stall。

例子: $n=50$, $\Delta t = 33$ ms, $\ell_S = 100$ ms
- 需要 $g \geq 100/(33 \times 50) = 0.06$
- 即只要 queue 剩余 ≥ 3 步就触发, 就能避免 stall

### 3.4 Figure 3 三个 regime 解读

**Panel A**: 不同 $g$ 下 queue size $|\mathbf{A}_t|$ 随时间演化

- **$g=0$ (sequential)**: queue 从 n 急剧降到 0, 然后 idle $\mathbb{E}[\ell_S]$ 秒, 然后跳回 n。锯齿状, 谷底为 0
- **$g=0.7$ (async)**: queue 在某个稳定区间波动, 不触底。每次消耗 30% chunk 触发新 inference, overlap 提供缓冲
- **$g=1$ (reactive)**: queue 几乎始终满, 只有小锯齿(因为 $\Delta t / \mathbb{E}[\ell_S] < 1$)。每 tick 一次 inference, 极贵

**Panel B**: 加 similarity filter 后
- Filter drops 近重复 observations, 避免 queue 被 nearly identical chunks 频繁打断
- 红色箭头标注一个特殊时刻: queue 空了, 即使 observation 几乎相同也强制处理

### 3.5 实验结果(Figure 5)

(a) Success rate:
| Inference | Pick-Place | Stacking | Sorting | Avg |
|---|---|---|---|---|
| Sync | 75 | 90 | 70 | 78.3 |
| Async | 80 | 90 | 50 | 73.3 |

Async 略低(可能 sorting 长 horizon 任务对 queue overlap 敏感), 但接近。

(b) Task completion time:
| Inference | Total (s) | Avg (s) | Std |
|---|---|---|---|
| Sync | 137.5 | 13.75 | 2.42 |
| Async | 97.0 | 9.70 | 2.95 |

**Async 快 30%**, 因为没有 chunk 间 idle lag。

(c) Fixed time (60s) 内完成的任务数:
| Inference | Total | Avg | Std |
|---|---|---|---|
| Sync | 9 | 1.8 | 0.45 |
| Async | 19 | 3.8 | 1.3 |

**Async 在固定时间内完成 2 倍多的任务**。这是实际部署最重要的指标 — 不是单次 success rate, 而是 throughput。

**这是 paper 最 strong 的 claim 之一**: async inference 是 model-agnostic 的, 可以加到任何 chunk-based policy 上, 大幅提升 throughput。

---

## 4. 数据集: Community-driven 的关键创新

### 4.1 为什么 community data

Robotics 数据相比 NLP/CV 数据少几个数量级。OpenX-Embodiment (O'Neill et al., 2024, https://openx-embodiment.github.io/) 集中了 1M+ trajectories, 但仍受限。

Hugging Face LeRobot 平台(Cadene et al., 2024, https://github.com/huggingface/lerobot)催生了 community data: 个人贡献者用便宜的 SO-100 arm(几百美元, 3D printed)采集数据上传。SmolVLA 用了 **481 个 community datasets**, 共 22.9K episodes, 10.6M frames(Table 1)。

这个数据量比 OpenVLA 的 1M trajectories 少两个数量级, 但 SmolVLA 仍然 work, 关键是:
- 多样性高(不同人、不同环境、不同 task)
- 噪声多(community 数据本身质量不一), 反而像 data augmentation
- Embodiment 一致(主要 SO-100), 让模型快速学到 low-level control

### 4.2 Task annotation 自动化

Community data 一个大问题: task description 质量差。Paper 提到看到 "task desc", "Hold", "Up" 这种无意义 placeholder。

解决方案: 用 Qwen2.5-VL-3B-Instruct (Bai et al., 2025, https://arxiv.org/abs/2502.13923) 自动生成 task description。Prompt:

```
Here is a current task description: {current_task}. Generate a very short, clear, and complete one-sentence describing the action performed by the robot arm (max 30 characters). Do not include unnecessary words. Be concise.
Here is some examples: Pick up the cube and place it in the box, open the drawer and so on. 
Start directly with an action verb like "Pick", "Place", "Open", etc.
Similar to the provided examples, what is the main action done by the robot arm?
```

这是 VLM 用 VLM 的有趣闭环: 一个 VLM 帮另一个 VLM 训练。

### 4.3 Camera viewpoint normalization

Community data 另一个问题: camera naming 不一致。`images.laptop` 在不同 dataset 可能是 top/side/wrist view。

SmolVLA 手动 mapping: 每个相机标准化为 `OBS_IMAGE_1` (top), `OBS_IMAGE_2` (wrist), `OBS_IMAGE_3` (side)。多余 view drop。

这个细节看似 minor, 但 ablation 暗示对训练至关重要 — 一致的 visual encoding 让 VLM 学到稳定的 spatial representation。

---

## 5. 实验结果深入分析

### 5.1 Simulation: LIBERO

LIBERO (Liu et al., 2023, https://arxiv.org/abs/2306.03310) 四个 task suite:
- **Spatial**: spatial generalization
- **Object**: novel object generalization  
- **Goal**: novel goal generalization
- **Long (10)**: long-horizon 10-step tasks

Table 2 结果:
| Policy | #Params | VLA Pt | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|---|
| Diffusion Policy | - | No | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| Octo | 0.09B | Yes | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| OpenVLA | 7B | Yes | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0 (PaliGemma-3B) | 3B | No | 87 | 63 | 89 | 48 | 71.8 |
| π0 (3.3B) | 3.3B | Yes | 90 | 86 | 95 | 73 | 86.0 |
| **SmolVLA** | 0.24B | No | 87 | 93 | 88 | 63 | 82.75 |
| **SmolVLA** | 0.45B | No | 90 | 96 | 92 | 71 | **87.3** |
| SmolVLA | 2.25B | No | 93 | 94 | 91 | 77 | 88.75 |

关键观察:
- **SmolVLA 0.45B > π0 3.3B (87.3 vs 86.0)**, 参数少 7.3x
- SmolVLA **没有 robotics pretraining** (VLA Pt = No), π0 3.3B 有 10k 小时 robotics pretraining
- π0 (PaliGemma-3B) 即 VLM-only 初始化, 只有 71.8%, 远低于 SmolVLA — 说明 SmolVLA 的 architecture 比 π0 更 sample efficient
- Long horizon (10) 是所有方法瓶颈, 但 SmolVLA 0.45B 的 71% 接近 π0 的 73%, 不输
- Scaling 到 2.25B 又涨 1.45%, 说明 SmolVLA 架构有 scaling potential

### 5.2 Simulation: Meta-World

Meta-World (Yu et al., 2020, https://arxiv.org/abs/1910.10897) 50 tasks, 难度递增:
| Policy | Easy | Medium | Hard | Very Hard | Avg |
|---|---|---|---|---|---|
| Diffusion Policy | 23.1 | 10.7 | 1.9 | 6.1 | 10.5 |
| TinyVLA | 77.6 | 21.5 | 11.4 | 15.8 | 31.6 |
| π0 (Paliigemma) | 80.4 | 40.9 | 36.7 | 44.0 | 50.5 |
| π0 (3.5B) | 71.8 | 48.2 | 41.7 | 30.0 | 47.9 |
| SmolVLA 0.24B | 86.43 | 46.36 | 35 | 60 | 56.95 |
| SmolVLA 0.45B | 82.5 | 41.8 | 45.0 | 60.0 | 57.3 |
| SmolVLA 2.25B | 87.14 | 51.82 | 70 | 64 | 68.24 |

SmolVLA 0.45B 比 π0 3.5B 高 9.4%, 再次胜出。Very Hard 任务上 SmolVLA 显著领先(60 vs 30), 推测因为 flow matching 对 complex multimodal action 更有优势。

### 5.3 Real-world SO-100

三个任务: Pick-Place, Stacking, Sorting。Table 3:
| Policy | Pick-Place | Stacking | Sorting | Avg |
|---|---|---|---|---|
| ACT (single-task) | 70 | 50 | 25 | 48.3 |
| π0 (3.5B, multi-task) | 100 | 40 | 45 | 61.7 |
| SmolVLA (0.45B, multi-task) | 75 | 90 | 70 | **78.3** |

SmolVLA 比 π0 高 16.6%, 即使 π0 在 Pick-Place 上 100% (但 stacking 仅 40%, 极不均衡)。SmolVLA 三任务均稳定。

### 5.4 Real-world SO-101 (新 embodiment)

SO-101 是 SO-100 升级版, SmolVLA **pretraining 时没见过 SO-101**, 测试 cross-embodiment generalization:
| Policy | In-Distribution | Out-of-Distribution |
|---|---|---|
| ACT (single-task) | 70 | 40 |
| SmolVLA (single-task) | 90 | 50 |

SmolVLA 比 ACT 高 20% (in-dist) 和 10% (OOD), 在 Lego precision task 上。这证明 community pretraining 学到的 features 可以 transfer。

### 5.5 Pretraining 的影响

Table 5:
| Setting | VLA Pt | Pick-Place | Stacking | Sorting | Avg |
|---|---|---|---|---|---|
| Single-task | No | 55 | 45 | 20 | 40 |
| Multi-task | No | 80 | 40 | 35 | 51.7 |
| Multi-task | Yes | 75 | 90 | 70 | **78.3** |

Pretraining 带来 26.6% 提升(51.7 → 78.3), multi-task vs single-task 也有 11.7% 提升。说明:
- Community data 的 general physical skills 可以 transfer
- Multi-task finetuning 提供正则化和 representation sharing

---

## 6. Ablation 综合解读: 设计 decision 的 reasoning

把所有 ablation 放一起, 可以看出 SmolVLA 的设计哲学:

### 6.1 Attention 设计

| 设计点 | 选择 | 直觉 |
|---|---|---|
| VLM-action 交互 | CA+SA interleaved | CA 提供 grounding, SA 提供 temporal consistency |
| SA 内部 mask | Causal | 避免 future action leakage, 训练-推理一致 |
| State 输入位置 | Prefix (to VLM) | 让 VLM 也看到 robot state, 编码 proprioception |

Table 11 (States as prefix vs suffix):
| States | Attention | LIBERO Avg |
|---|---|---|
| Prefix (to VLM) | CA | 80.3 |
| Suffix (to expert) | CA | 73.3 |
| Prefix | SA | 53.3 |
| Suffix | SA | 74.8 |

Interesting 发现:
- 用 CA 时, prefix 明显好(80.3 vs 73.3) — VLM 需要 state 来 ground visual features
- 用 SA 时, suffix 更好(74.8 vs 53.3) — SA 时 state 作为 action tokens 的一部分, 让 VLM 完全专注 vision
- 这暗示 **VLM 应该 multimodal fuse vision + state**, 而不是只处理 vision

### 6.2 Action chunk size

Table 12:
| Chunk size n | LIBERO Avg |
|---|---|
| 1 | 50.0 |
| 10 | **84.0** |
| 30 | 78.5 |
| 50 | 80.3 |
| 100 | 74.5 |

$n=1$ 单步预测效果差(没有 temporal smoothing), $n=100$ 太长(open-loop 太久), $n=10$ 最优, $n=50$ production 选择是 tradeoff(speed vs accuracy)。

### 6.3 Action execution steps

Table 13 (多久更新 observation):
| Steps | LIBERO Avg |
|---|---|
| 1 | 80.3 |
| 10 | **82.8** |
| 30 | 70.8 |
| 50 | 51.8 |

每 10 步更新 observation 最好(82.8)。每 1 步略低 — 可能因为过度 reactive 导致噪声影响。每 50 步(执行完整个 chunk)效果暴跌到 51.8%, 说明 **open-loop 时间长就失败**。

这印证 async inference 的价值: 频繁更新但非阻塞。

---

## 7. 与其他 VLA 的对比和我的思考

### 7.1 与 OpenVLA 的对比

OpenVLA (Kim et al., 2024, https://openvla.github.io/):
- 7B Llama-2 backbone
- Action tokenization(离散 binning)
- 1M trajectories OpenX pretraining

SmolVLA 优势:
- 0.45B vs 7B, 15x 小
- Flow matching vs tokenization, continuous control 自然
- 22K community episodes vs 1M OpenX, 数据效率高 50x
- 训练 single GPU vs 多节点

OpenVLA 优势:
- 1M data 覆盖 cross-embodiment 更广
- LLM reasoning 能力更强(7B vs 0.45B)

### 7.2 与 π0 的对比

π0 (Black et al., 2024, https://arxiv.org/abs/2410.24164):
- 3.3B PaliGemma + flow matching
- 10k 小时 robotics pretraining
- 主要 CA

SmolVLA 与 π0 共享 flow matching, 但:
- SmolVLA 用 SmolVLM-2(更 efficient) 而非 PaliGemma
- SmolVLA 加 SA layers(interleaved), π0 主要 CA
- SmolVLA 用 community data + 单 embodiment
- SmolVLA 加 layer skipping

Ablation 显示 SA 的加入对 real robot 平滑性很重要(paper 3.1 节提到 "self-attention to contribute to smoother action chunks")。这是 SmolVLA 相对 π0 的核心架构改进。

### 7.3 与 TinyVLA 的对比

TinyVLA (Wen et al., 2024, https://arxiv.org/abs/2409.12514):
- <1B, from scratch
- 多模态 pretraining + robotics finetuning

SmolVLA 优势在 Meta-World 56.95 vs TinyVLA 31.6 — flow matching 显著优于 TinyVLA 的 regression。

### 7.4 我的几个直觉

**Intuition 1**: SmolVLA 验证了 **architecture > scale** for robotics。Robotics 的 bottleneck 是 control frequency 和 embodiment-specific low-level skill, 而非 LLM-style reasoning。0.45B + flow matching + async 在 practical tasks 上超过 7B+ tokenization。

**Intuition 2**: Community data 是 future of robotics research。Industrial data(OpenX, DROID)集中、昂贵, 而 SO-100 这种便宜 platform + community 上传 + auto-annotation 是真正 scalable 的方向。这篇 paper 的数据 list(Appendix A.1)列了几百个 user, 体现了 Hugging Face 的 ecosystem thinking。

**Intuition 3**: Async inference 应该成为 robotics policy deployment 的标配。Chunk-based policy + async execution 几乎 free 提速 30%, 没理由不用。这个 engineering 贡献比 model 贡献更具有 immediate impact。

**Intuition 4**: Flow matching > diffusion > regression for action modeling。这与 π0、diffusion policy 结论一致。Conditional flow matching 的 rectified flow path 数学简洁, 推理步数少, 是 robotics action 生成的正确范式。

**Intuition 5**: Layer skipping 在 VLA 上 work, 因为 robotics 不需要 deep abstract reasoning, 只需要 dense feature。这与 Bolya 的 Perception Encoder 发现呼应 — visual representations 的 best layer 往往不在 output。

---

## 8. Limitations 和 future directions

Paper 自己列了几个, 我加上我的联想:

1. **Single embodiment pretraining**: SmolVLA 主要在 SO-100 上 pretrain, transfer 到 SO-101 work, 但更异构的 robot(Franka Panda, UR5)未测试。Cross-embodiment pretraining(类似 RT-X)是自然 next step。

2. **Dataset 规模**: 22K episodes vs OpenVLA 的 1M, 还有 50x 空间。如果 community data 继续增长, SmolVLA 架构能否 scale? Paper Table 2 显示 2.25B 版本在 LIBERO 88.75%, Meta-World 68.24%, 仍有提升空间。

3. **Long-horizon task**: LIBERO Long-horizon SmolVLA 71%, π0 73%。长 horizon 需要 hierarchical planning 或 subgoal decomposition, SmolVLA 当前是 flat policy。

4. **VLM backbone 选择**: Paper 5.1 提到 SmolVLM-2 主要 pretrain 在 OCR/document, 不一定 optimal for robotics。一个 robotics-specific pretrained VLM(比如在 manipulation video 上 pretrain)可能更好。

5. **RL finetuning**: 当前用 imitation learning。最近 ConRFT (Chen et al., 2025, https://arxiv.org/abs/2502.05450) 显示 RL finetune VLA 有效, SmolVLA + RL 是 obvious next step。

6. **Async inference 理论分析**: 当前分析假设 $\ell_S$ 稳定, 实际 server load 波动时 queue 动态更复杂。可以借鉴 queueing theory 分析。

7. **State input as single token**: SmolVLA 把 sensorimotor state 压成 1 个 token。对于高维 state(比如 force/torque, tactile)可能信息损失。Multi-token state encoding 可能更好。

8. **Cross-attention vs full self-attention on VLM+action tokens**: 最近 GR00T N1 用纯 SA(concat), SmolVLA 用 interleaved。哪个更好需要更大规模 ablation。

---

## 9. 相关 work 的 web links 合集

VLA foundation models:
- π0: https://arxiv.org/abs/2410.24164
- OpenVLA: https://openvla.github.io/
- RT-2: https://robotics-transformer2.github.io/
- RT-2-X (OpenX): https://openx-embodiment.github.io/
- Octo: https://octo-models.github.io/
- GR00T N1: https://arxiv.org/abs/2503.14734
- TinyVLA: https://arxiv.org/abs/2409.12514
- DexVLA: https://arxiv.org/abs/2502.05855
- FAST (Pertsch et al.): https://arxiv.org/abs/2501.09747

VLM backbones:
- SmolVLM-2: https://arxiv.org/abs/2504.05299
- PaliGemma: https://arxiv.org/abs/2407.07726
- SmolLM2: https://arxiv.org/abs/2502.02737
- SigLIP: https://arxiv.org/abs/2303.15343

Diffusion/Flow matching:
- Flow Matching (Lipman): https://arxiv.org/abs/2210.02727
- Rectified Flow (Liu): https://arxiv.org/abs/2209.14577
- SD3 (Esser): https://arxiv.org/abs/2406.06275
- Diffusion Policy: https://arxiv.org/abs/2307.01948

Layer skipping / early exit:
- Skip layers in multimodal LLM (Shukor & Cord): https://arxiv.org/abs/2410.09454
- Perception Encoder (Bolya): https://arxiv.org/abs/2504.13181
- Autoregressive pretraining from video (Rajasegaran): https://arxiv.org/abs/2501.05453
- AIM (El-Nouby): https://arxiv.org/abs/2401.08541

Benchmarks:
- LIBERO: https://arxiv.org/abs/2306.03310
- Meta-World: https://arxiv.org/abs/1910.10897
- Bridge Data: https://arxiv.org/abs/2109.13396
- DROID: https://arxiv.org/abs/2403.12945

Robots:
- SO-100 / SO-101: https://github.com/TheRobotStudio/SO-ARM100
- LeRobot: https://github.com/huggingface/lerobot
- ALOHA (ACT): https://arxiv.org/abs/2304.13705

RL finetuning for VLA:
- ConRFT: https://arxiv.org/abs/2502.05450

---

## 10. 总结

SmolVLA 的核心贡献是 **"如何让 VLA 真正 deployable"**, 通过三个 orthogonal 方向:

1. **Architecture efficiency**: layer skipping + visual token reduction + interleaved CA/SA + small action expert
2. **Data efficiency**: community-collected data + auto-annotation + camera normalization
3. **Inference efficiency**: async stack with threshold g + similarity filter

数学上的核心是 flow matching loss:

$$\mathcal{L}^\tau(\theta) = \mathbb{E}\left[\|\mathbf{v}_\theta(\tau\mathbf{A}_t + (1-\tau)\epsilon, \mathbf{o}_t) - (\epsilon - \mathbf{A}_t)\|^2\right]
$$

工程上的核心是 async inference 的 threshold:

$$
g \geq \frac{\mathbb{E}[\ell_S] / \Delta t}{n}
$$

两者结合, SmolVLA 用 0.45B 参数 + 单 GPU 训练 + consumer GPU/CPU 推理, 在 LIBERO、Meta-World、real-world SO-100/SO-101 上匹配或超过 10x 大小的 π0 和 OpenVLA。

这篇 paper 对 robotics community 的实际意义在于: 它把 VLA 的 entry barrier 从 "公司级资源" 降到 "个人开发者级", 同时开源了 code, models, data, hardware。这是 Hugging Face 一贯的 democratization 哲学在 robotics 上的延伸。

如果让我做 follow-up, 我会探索: (1) cross-embodiment pretraining 加多种 cheap arm, (2) RL finetune with sparse reward, (3) long-horizon hierarchical decomposition, (4) active perception(选择性地看哪里)。这些都是 SmolVLA 当前 architecture 的自然扩展。

Reference paper: https://arxiv.org/abs/2506.08450 (具体 final arxiv ID 待 paper 官方发布, 当前是 Hugging Face blog 形式)

希望这个讲解 build 了你完整的 mental model, Andrej。如果想深挖某个具体 part(比如 flow matching 的数学细节, 或者 async queue 的具体实现), 可以再细聊。
