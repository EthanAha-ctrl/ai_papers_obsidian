---
source_pdf: StarVLA A Lego-like Codebase for.pdf
paper_sha256: 88a0d78d9f310294e2733c29073b9edb974b8eb9ebcaa2104939773f9315ee14
processed_at: '2026-08-12T11:00:26-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# StarVLA 人话版

Andrej，我换个方式讲，尽量大白话，但技术细节照样给你。

---

## 1. 这篇 paper 到底干了啥

一句话：**VLA 这个圈子现在乱成一锅粥，有人用 VLM 出 action，有人用 world model 出 action，各有各的 codebase、各有各的 eval protocol，根本没法公平比。StarVLA 就是把这些乱七八糟的东西塞进一个统一框架里，像搭 Lego 一样随便拼。**

它不是提出一个新算法，它是在做 infrastructure。就像当年 HuggingFace 把 BERT / GPT / T5 全部塞进 `transformers` 库统一接口一样，StarVLA 想干的是同一件事，只不过对象是 VLA。

---

## 2. 为什么需要这东西

想象你是一个 researcher，想搞清楚一个问题：**π0 的 flow-matching action head 真的比 OpenVLA-OFT 的 MLP regression head 好吗？**

你打开 π0 的 codebase，发现：
- 它用自己一套 data pipeline
- 它用自己一套 image preprocessing
- 它用自己一套 action normalization
- 它用自己一套 eval harness

然后你打开 OpenVLA-OFT 的 codebase，发现另一套完全不同的东西。

你想做 controlled comparison，hold 其他变量不变只换 action head？**根本不可能**。因为换 action head 意味着换整个 codebase，data pipeline 变了，preprocessing 变了，eval 变了，你看到的 performance 差异根本不知道归因给谁。

这就是 paper Section 1 说的 fragmentation。作者把它分成三层：

**Architecture-level**：FAST 用离散 token autoregressive 出 action，OFT 用 MLP 直接回归连续值，π0 用 flow-matching 迭代 denoise，GR00T 用 dual-system（慢推理 + 快 action）。这四类方法的 action representation、loss、inference path 完全不同。

**System-level**：每个 release 把 model architecture + data processing + training pipeline 焊死在一起。OpenPI 的 backbone 没法直接接 π0 的 action head，因为两者之间的 representation interface 不兼容。

**Evaluation-level**：LIBERO 用 500 trials per suite，SimplerEnv 用 Visual Matching 和 Variant Aggregation 两种 protocol，RoboTwin 有 clean 和 randomized 两种 setup。不同 paper 报告不同 subset，SimplerEnv 自己 variance 都很大，作者跑了 5 次取均值才 honest。

这三层 fragmentation 叠在一起，VLA 文献的 "Tower of Babel" 就形成了。

---

## 3. 核心思路：一个公式统一所有 VLA

作者最 deep 的贡献是这个 formulation：

$$\pi(\mathbf{a}_{t:t+k}, \mathbf{y}_{\text{aux}} \mid \mathbf{x}_{\leq t}, \boldsymbol{\ell})
$$

翻译成人话：**一个 policy 接收历史观测 $\mathbf{x}_{\leq t}$ 和语言指令 $\boldsymbol{\ell}$，输出未来 k 步 action $\mathbf{a}_{t:t+k}$，可能还附带一些 auxiliary output $\mathbf{y}_{\text{aux}}$。**

变量细节：
- $\mathbf{x}_{\leq t}$：下标 $\leq t$ 表示到当前时刻为止的全部观测历史。上标 vis / depth / tactile 标注模态类型，所以 $\mathbf{x}$ 是多模态的
- $\boldsymbol{\ell}$：粗体表示它是 token sequence（一句话），不是单个 token
- $\mathbf{a}_{t:t+k}$：下标 $t:t+k$ 是 slicing notation，表示从时刻 $t$ 到 $t+k$ 的 action chunk。chunk-based control 是 diffusion policy 带火的方式
- $\mathbf{y}_{\text{aux}}$：auxiliary output。可以是 future observation $o^{\text{vis}}_{t+1:t+k}$（world model 干的事），也可以是 reasoning chain $\ell_{\text{plan}}$（VLM-based chain-of-thought 干的事）

然后训练 loss：

$$\mathcal{L} = \mathcal{L}_{\text{action}} + \mathcal{L}_{\text{aux}}
$$

- $\mathcal{L}_{\text{action}}$：监督 action 预测，所有 VLA 都有
- $\mathcal{L}_{\text{aux}}$：inductive bias，决定 paradigm 走向
  - Direct VLA：$\mathcal{L}_{\text{aux}} = 0$，纯 action supervision（OpenVLA, OFT）
  - VLM-based VLA：$\mathcal{L}_{\text{aux}}$ = language modeling loss on reasoning tokens（Magma, InstructVLA）
  - WM-based VLA：$\mathcal{L}_{\text{aux}}$ = future image/video prediction loss（Cosmos Policy, FLARE）

**关键 insight**：VLM-based 和 WM-based 表面上是两个 paradigm，实际上差别仅在 $\mathcal{L}_{\text{aux}}$ 的形式。一个加 language reasoning signal，一个加 future frame prediction signal。当你把 infrastructure 差异抹平，这个共性就显现了。作者叫它 "generalized VLA perspective"。

---

## 4. 工程实现：L Lego 怎么搭的

### 4.1 两个 contract

StarVLA 定义了两层标准化接口：

**Outer boundary**：raw observations → actions

**Inner boundary**：multimodal inputs → hidden states → actions

所有 framework module 继承同一个 base class，暴露两个方法：

```python
forward({raw images, str, ...}) → {loss_dict}
predict_action({raw images, str, ...}) → {normalized_actions}
```

注意 input 是 **raw environment-level observations**，不是 dataloader 里预处理好的 tensor。这是关键设计决策。

为什么这么重要？每个 VLM backbone 有自己的 image tokenization scheme。Qwen-VL 用 14×14 patch 加 window attention，Cosmos 用 video-native tokenization，InternVL 又是另一套。如果你训练时 preprocessing 在 dataloader 里做，部署时就得重新实现一遍，非常容易出 bug。train/test distribution mismatch 是 VLA 系统里 silent performance degradation 的主要来源。

把 preprocessing 放进 model 内部，让 raw observation 进 model、executable action 出 model，这个 contract 就 deployment-proof 了。

### 4.2 四种 action head 的本质

| Paradigm | Action 是什么 | 怎么 decode | Loss | 推理模式 |
|---|---|---|---|---|
| **FAST** | 离散 token | Autoregressive next-token | Cross-entropy | 顺序，单次 forward |
| **OFT** | 连续值 | MLP 并行回归 | L1 | 单次 forward |
| **π** | 连续值 | Flow-matching 迭代 denoise | Flow-matching objective | 迭代 10-20 步 |
| **GR00T** | 连续值 | Dual-system: System 2 慢推理 + System 1 DiT flow-matching | Flow-matching + LM | 两阶段推理 |

用大白话讲：

**FAST**：把 action chunk 通过 DCT 变换 + bpe-style tokenizer 离散化到 LLM vocabulary 空间，然后让 LLM autoregressive 地 next-token predict 出来。优点是直接复用 LLM inference，缺点是离散 quantization 损失精度。Ref: [FAST paper](https://arxiv.org/abs/2501.09747)

**OFT**：最朴素。在 VL backbone 输出里 predefined 几个 "action token positions"，读取这些位置的 hidden states，过一个 MLP 回归出连续 action。L1 loss。简单到令人发指，但效果惊人。Ref: [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)

**π**：flow-matching action expert。关键技术点：cross-attention 让 action DiT condition on **multi-layer** VL hidden states，而不是只取 last layer。last layer 主要是 semantic，low-layer 保留 spatial info，对 manipulation 很关键。Ref: [π0](https://arxiv.org/abs/2410.24164)

**GR00T**：dual-system，借用 Kahneman System 1/2 框架。VL backbone 当 System 2 慢 reasoning，DiT flow-matching 当 System 1 快 action。System 2 先产生 reasoning context（可能 implicit，通过 hidden states），System 1 快速生成 action。Ref: [GR00T N1](https://arxiv.org/abs/2503.14734)

这四类方法从 VLM-native（FAST, OFT）到 generative-model-based（π, GR00T），覆盖了当前 VLA 文献的主要 action decoding 家族。在 StarVLA 里，加新 paradigm 只需要实现并注册一个新的 action head module，backbone / training loop / eval pipeline 全部不动。

### 4.3 Backbone 也能换

这是 StarVLA 的另一层 modularity：

- **VLM backbone**：Qwen2.5-VL / Qwen3-VL ([Bai et al., 2025](https://arxiv.org/abs/2502.13923)), InternVL-A1 ([Cai et al., 2026](https://arxiv.org/abs/2601.02456))
- **World-Model backbone**：Cosmos-Predict2 ([Kim et al., 2026](https://arxiv.org/abs/2601.16163)), Dreamdojo ([Gao et al., 2026](https://arxiv.org/abs/2602.06949))

WM backbone 的本质区别：预训练时就 condition on action 预测未来帧，所以 hidden states 编码了 dynamics 信息。VLM backbone 只编码视觉语义，dynamics 要在 robot data 上从头学。

Table 2 里 Cosmos-Predict2-2B 在 LIBERO 上能 match Qwen3-VL-4B（95.8 vs 96.6），说明在 closed-world 简单任务上 dynamics prior 没那么关键。但在 long-horizon 或 OOD 场景下应该更有优势，paper 没直接 ablation 这点是 limitation 之一。

配置就是 YAML，像搭 Lego：

```yaml
backbone: qwen3-vl-4b  # or cosmos-predict2-2b, internvl-a1
action_head: oft        # or fast, pi, gr00t
```

---

## 5. 训练 pipeline 三种模式

### 5.1 纯 SFT

最直接的 behavior cloning。实现路径：`starVLA/training/train_starvla.py`。优化设置：full-parameter fine-tuning 或 selective freezing via `trainer.freeze_modules`。多 parameter group，不同 LR（qwen_vl_interface 与 action model 可以分开调）。bfloat16 autocast + gradient accumulation + gradient clipping + cosine schedule with min LR。

### 5.2 Multi-Modal Co-training

动机：robot-only SFT 会快速 catastrophic forgetting 掉 VLM backbone 的视觉语言能力。几千步内 RefCOCO-g grounding 性能就能掉到接近 random。

实现：`starVLA/training/train_starvla_cotrain.py`。双 dataloader，每步两次 forward/backward：
- VLA forward through `framework.forward()` → `action_loss`
- VLM forward through `qwen_vl_interface` → `language_modeling_loss`
- VLM loss 缩放系数：`trainer.loss_scale.vlm`

Section 6 的实验（Figure 4 + Table 8）做了三种对比：
1. Vanilla VLA：纯 action supervision，RefCOCO-g 20K 步掉到 random
2. Vanilla co-training：部分保留 perception，但震荡
3. ST4VLA spatially guided ([Ye et al., 2026a](https://arxiv.org/abs/2602.10109))：保留 ~70% grounding 性能的同时 manipulation 强

Figure 4c 里的 **Projection-Space Similarity (PSS)** 值得注意——它衡量 spatial grounding gradient 与 action objective gradient 在 subsspace 上的对齐程度。两个 objective gradient 对齐时 co-training 双赢，不对齐时互相干扰。这跟 Multi-Task Learning 里 [PCGrad](https://arxiv.org/abs/2001.06782) / [GradVac](https://arxiv.org/abs/2104.02291) 的 gradient surgery 思路是一脉的。

### 5.3 Cross-Embodiment Co-training

通过 LeRobot mixture dataset 接口，配置成 `(dataset_name, sampling_weight, robot_type)` tuples。运行时 `LeRobotMixtureDataset` 按 weight 采样并追踪 embodiment tag。

Generalist 实验的关键 trick（Section 7.1）：处理不同 embodiment 的 action space 差异（单臂 7-DoF，双臂 14-DoF，humanoid 更多），他们用 **unified padding 到 32-dim action vector** 的策略。低 DoF padding 上去，高 DoF 截断或映射。简单粗暴但 effective。

---

## 6. Eval：Server-Client 架构

StarVLA 用 **thin WebSocket policy server + 轻量 client wrapper**。

Check point 通过 `base_framework.from_pretrained()` 加载，hosted 在 StarVLA runtime 里。Benchmark evaluator 在自己的 conda env 里（LIBERO, SimplerEnv, RoboTwin 各自依赖不兼容），通过 client wrapper 通信。

Inference interface 统一：`Framework.predict_action()`。Client 把 observations 打包成 dict (`image`, `lang`, optional `state`, `timestamps`, `episode metadata`)，msgpack 序列化发到 server，server 返回 dict 包含 `normalized_actions`。

Benchmark-specific 差异隔离在 `model2libero_interface.py`, `model2simpler_interface.py`, `model2robotwin_interface.py` 等文件里。这些 adapter 干的事：image resize 到训练 resolution、读 `dataset_statistics.json` 做 action unnormalization、chunked normalized predictions → executable actions、action ensembling、sticky grippers、delta→absolute action 转换等。

**Sim-to-real 无缝的关键**：real robot controller 就是另一个 benchmark client。捕获 camera obs，组装 example dict，query remote server，执行 action。Robot-side 的 ROS node / vendor SDK 不进 StarVLA runtime。同一个 checkpoint sim 和 real 通用，只要 client 提供的 observation dict 格式对就行。

---

## 7. 实验结果重点解读

### 7.1 LIBERO（Table 2）

最 striking 的对比：

| Model | Steps | Epochs | Avg |
|---|---|---|---|
| OpenVLA-OFT | 175K | 223 | 97.1 |
| π0+FAST | - | - | 85.5 |
| GR00T-N1.5 | 20K | 203 | 86.5 |
| **StarVLA-OFT (Qwen3-VL-4B)** | **30K** | **9.54** | **96.6** |
| StarVLA-OFT (Cosmos-Predict2-2B) | 30K | 9.54 | 95.8 |

StarVLA-OFT 只用 30K steps (9.54 epochs) 就接近 OpenVLA-OFT 175K steps (223 epochs)。**6× fewer steps, 23× fewer epochs**。说明 Qwen3-VL-4B 作为 backbone 比 OpenVLA 用的 Prismatic-7B 在 LIBERO 上更 sample-efficient。30K 步只跑 ~10 epoch，backbone representation 几乎不需要大幅 shift，主要是 action head 对齐。

但 Long-horizon (LIBERO-Long) 上 StarVLA-OFT 是 93.8 vs OpenVLA-OFT 94.5，差距稍大。Long-horizon 还是需要更多训练或者更好 reasoning supervision。

### 7.2 SimplerEnv（Tables 3 & 4）

SimplerEnv 是 real-world proxy，方差大。作者跑 5 次取均值，honest。

**WidowX VM**：StarVLA-GR00T 65.3% (Qwen3-VL-4B)，对比 π0-FAST 48.3%, GR00T N1.5 35.8%, Magma 61.9%。

**Google Robot**：StarVLA-OFT: VM 76.0, VA 70.2，对比 CogACT 74.8 / 61.3，π0 58.8 / 54.8，GR00T N1.5 35.2 / 44.5。

Google Robot 上 StarVLA-OFT 几乎全面领先。"Open Top Drawer and Place Apple" 这个 long-horizon 任务上，StarVLA-OFT 是 66.1% (VM) / 59.4% (VA)，其他方法基本都 < 25%。Qwen3-VL 的 instruction following 能力在这里帮了大忙。

### 7.3 RoboCasa-GR1（Tables 5 & 6）

Humanoid-style manipulation，比 LIBERO 难得多：

StarVLA-OFT 48.8% > GR00T-N1.6 47.6% > StarVLA-GR00T 47.8% > StarVLA-π 43.9% > StarVLA-FAST 39.0% > π0.5 37.0%

**关键 insight**：在更难的任务上 **action head 选择 matters more**。Continuous action (OFT/π/GR00T) 都比 discrete (FAST) 显著好。FAST 落后 ~10 pt，high-precision bimanual 任务对 action quantization 误差更敏感。

Table 6 task-level 数据有意思：PnPCanToDrawerClose 上 StarVLA-GR00T 80% vs GR00T-N1.6 13%。但 PnPBottleToCabinetClose 上反过来。Task-level specialization 差异暗示 generalist 在某些 task 上反而比 specialist 好（cross-task transfer）。

### 7.4 RoboTwin 2.0（Table 7）

Bimanual 任务：StarVLA-π 88.1/88.8 (clean/random)，StarVLA-OFT 88.2/88.3，StarVLA-GR00T 88.0/88.5。对比 Lingbot-VLA 88.6/86.7，π0.5 82.7/76.8。

Random setup 上 StarVLA 比 π0.5 高 12 pt，**robustness 优势明显**。

### 7.5 Generalist vs Specialist（Table 9）

Hero result：

| Settings | LIBERO avg | SimplerEnv WidowX | SimplerEnv Google VA/VM | RoboTwin clean/random | RoboCasa |
|---|---|---|---|---|---|
| StarVLA-OFT (specialist) | 98.8 | 64.6 | 70.2/76.0 | 88.2/88.3 | 53.8 |
| **Generalist StarVLA** | 97.8 | **70.2** | **73.8/79.3** | 88.7/87.8 | **57.3** |

**最 striking 的发现**：Generalist 在 SimplerEnv 和 RoboCasa 上 **比所有 specialist 都强**！WidowX +5.6 pt, Google VA +3.6 pt, Google VM +3.3 pt, RoboCasa +3.5 pt。

Cross-benchmark joint training 帮助了 generalization，反过来也说明 specialist 容易 overfit 到自己 benchmark 的 distribution。LIBERO 上掉了 1 pt，但其他都涨了，非常 healthy 的 trade-off。

---

## 8. 计算效率（Section 8）

### 8.1 Single-node（8×A100, Table 10）

| Per-GPU batch | Global batch | s/step | samples/s | GPU util |
|---|---|---|---|---|
| 2 | 16 | 0.703 | 22.7 | 74% |
| 8 | 64 | 1.131 | 56.6 | 92% |
| 24 | 192 | 2.404 | 79.9 | 96% |

大 batch 提升 sample throughput 但 linearly 增加 step latency。batch 24 vs 2 在 sample throughput 上 3.5×，但 step latency 3.4×。GPU util 从 74% → 96% 几乎饱和。

Practical guidance：**batch 8 是 sweet spot**（92% util, 1.13s/step）。

### 8.2 Multi-node（Table 11）

| # GPUs | s/step | samples/s | Scaling eff |
|---|---|---|---|
| 8 | 0.735 | 87.0 | 100% |
| 32 | 0.899 | 284.7 | 81.9% |
| 64 | 0.925 | 553.8 | 79.6% |
| 128 | 0.921 | 1111.5 | 79.9% |
| 256 | 0.931 | 2200.0 | 79.1% |

**关键 insight**：inter-node communication 是 **one-time latency overhead**（0.735→0.93 s/step），过了 8 节点 (64 GPU) 后 plateau，再扩展不再降效。**可以放心 scale 到几百 GPU，scaling efficiency 稳定在 79-80%**。

估算：260M trajectories * 32-dim action * 8 chunk，256 GPU 跑 100K steps = 25.5 小时能处理 2200 samples/s × 91800s ≈ 2 亿样本。对 cross-embodiment large-scale pretraining 可行。

---

## 9. StarVLA 跟其他 codebase 比（Table 1）

| Framework | Modular Action Heads | Modular VLM | Modular WM | Mixture DS | Open Co-train MM | Open Co-train X-Emb | #Bench | Multi-Bench Co-train |
|---|---|---|---|---|---|---|---|---|
| OpenPI | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 2 | ✗ |
| Isaac-GR00T | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | 6 | ✗ |
| OpenVLA-OFT | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | 1 | ✗ |
| Dexbotic | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | 5 | ✗ |
| X-VLA | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | 5 | ✗ |
| **StarVLA** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **7** | **✓** |

StarVLA 在所有维度全 ✓，唯一同时支持 modular action heads + modular VLM + modular WM + open-source co-training + multi-benchmark joint training 的 codebase。

---

## 10. 跟 Karpathy 思想的关联

### 10.1 "Software 2.0" 哲学

你之前在 "Software 2.0" essay 里强调过：很多看起来不同的方法在 abstraction 上是同构的。StarVLA 的 generalized VLA perspective 正是这个哲学——把 paradigm-level 表象差异 abstract 掉，露出 loss formulation 的共性。

所有 VLA 都是 $\pi(\mathbf{a}, \mathbf{y}_{\text{aux}} | \mathbf{x}, \ell)$ 的 instantiation。FAST / OFT / π / GR00T 表面上千差万别，实际上都是同一个 equation 的不同 $\mathcal{L}_{\text{aux}}$ 形式。这跟 Software 2.0 把 "differentiable programming" 作为统一视角是同一思路。

### 10.2 "micrograd" / "makemore" 的 first-principles 精神

你讲过的 "micrograd" / "makemore" 系列强调 **从第一性原理理解 architecture**。StarVLA 的 unified formulation 给了 VLA 一个 first-principles 视角。当你把所有 paradigm 放进 Eq. (1)-(2) 的框架里，你会发现它们在 loss landscape 上是邻居，差别只在 auxiliary signal 的选择上。

### 10.3 HF transformers 的类比

StarVLA 的 backbone-head 解耦哲学跟 HF transformers 的 `PreTrainedModel` + task-specific head 一脉相承，但更严格——它要求 raw observation contract，强制 preprocessing 进 model 内部。这是 VLA 特殊需求，因为 robot deployment 时 sensor stream 必须 raw。

如果未来 StarVLA 被广泛采纳，VLA backbone 的发布格式可能标准化为 "preprocessed observation → hidden state" 接口，类似 BERT 的 "tokenized input → contextualized embedding"。这会进一步降低 method comparison 的 friction。

---

## 11. Limitations 与延伸方向

### 11.1 Paper 没明说的 limitations

1. **没做 backbone 的 fine-grained ablation**：Qwen3-VL-4B vs Cosmos-Predict2-2B 在 LIBERO 上 comparable，但在 long-horizon / OOD 上谁更强？没专门 ablation
2. **Unified padding 32-dim 的 information loss**：对真正高 DoF 人形（如 60-DoF full body）可能不够
3. **没报告 real-robot 实验**：所有结果都在 sim 上，尽管 architecture 支持 sim-to-real。Paper 自己说 "closing the gap between research exploration and practical deployment"，没给实证
4. **Co-training 用的是 ST4VLA 的研究**：spatially guided 的具体实现细节在另一篇 paper 里

### 11.2 可能的延伸方向

- **Action head 作为 RL policy**：paper 提到 RL fine-tuning 是 ongoing integration with [RLinf](https://github.com/RLinf/RLinf)，未来 action head 可能变成 RL-learned policy
- **Test-time scaling for action generation**：dual-system GR00T 的 reasoning + action 范式可以扩展成 test-time search，类似 OpenAI o1 在 VLM 上的 reasoning
- **WM backbone + Action head 的最佳 combination**：哪种 action head 最适合 WM backbone？Paper 没系统 ablation，但 Cosmos + OFT 在 LIBERO 上 95.8%，Cosmos + π 在 SimplerEnv WidowX 上 58.7%，看起来 OFT 在简单任务、π 在复杂任务占优
- **Active learning / DAgger integration**：paper 明确 avoid DAgger 等在线 refinement，作为 baseline 是 honest 的，未来可以加

---

## 12. 最终一句话总结

**StarVLA 是 VLA 领域的 HuggingFace transformers，用 backbone-action-head 解耦 + unified raw-observation I/O contract 把 VLM-based / WM-based / Direct policy 三大范式塞进一个框架，证明它们在 loss 层面是同构的（差别仅在 $\mathcal{L}_{\text{aux}}$ 形式），并用同一套 infrastructure 在 5 个 benchmark 上跑出 strong baseline，还顺便展示了 cross-benchmark generalist training 比 specialist 更强。**

---

## 13. Reference 汇总

**StarVLA 本身**:
- Codebase: https://github.com/starVLA/starVLA
- Project page: https://starvla.github.io

**Backbones**:
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Cosmos Policy: https://arxiv.org/abs/2601.16163
- InternVL-A1: https://arxiv.org/abs/2601.02456

**Action heads**:
- FAST tokenizer: https://arxiv.org/abs/2501.09747
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- π0: https://arxiv.org/abs/2410.24164
- GR00T N1: https://arxiv.org/abs/2503.14734

**Benchmarks**:
- LIBERO: https://arxiv.org/abs/2305.12248
- LIBERO-Plus: https://arxiv.org/abs/2505.24502
- SimplerEnv: https://arxiv.org/abs/2405.05941
- RoboCasa: https://arxiv.org/abs/2406.02523
- RoboTwin 2.0: https://arxiv.org/abs/2504.13059
- BEHAVIOR-1K: https://arxiv.org/abs/2403.09228
- CALVIN: https://arxiv.org/abs/2112.03227

**Co-training & related studies**:
- ST4VLA (spatially guided): https://arxiv.org/abs/2602.10109
- OpenVLA: https://arxiv.org/abs/2406.09246
- CogACT: https://arxiv.org/abs/2411.19650
- SpatialVLA: https://arxiv.org/abs/2501.15830
- Magma: https://arxiv.org/abs/2502.13130

**World model / VLA unification**:
- World Action Models: https://arxiv.org/abs/2602.15922
- Causal World Modeling: https://arxiv.org/abs/2601.21998
- Dreamdojo: https://arxiv.org/abs/2602.06949
- FLARE: https://arxiv.org/abs/2505.15659
- V-JEPA 2: https://arxiv.org/abs/2506.09985

**Cross-embodiment / Open X-Embodiment**:
- OXE: https://arxiv.org/abs/2310.08864
- DROID: https://arxiv.org/abs/2403.12945

**Gradient analysis for multi-task learning**:
- PCGrad: https://arxiv.org/abs/2001.06782
- GradVac: https://arxiv.org/abs/2104.02291

---

希望这版本 build 起你的 intuition 了。如果你要快速 grasp 核心思想，就记住两件事：**(1) 所有 VLA 都是 $\pi(\mathbf{a}, \mathbf{y}_{\text{aux}} | \mathbf{x}, \ell)$ 的 instantiation，paradigm 差别只在 $\mathcal{L}_{\text{aux}}$ 形式；(2) StarVLA 用 raw-observation I/O contract + backbone-head 解耦把这个 abstraction 工程化，让 Lego-style composition 成为可能。**

---

# StarVLA 深度技术解析

Andrej，这篇 paper 我反复读了几遍，本质上这是一篇 **infrastructure/system paper**，伪装成 method paper 的味道很重。它的真正贡献不在于某个新算法，而在于把整个 VLA 生态的"分裂问题"(fragmentation)用一个清晰的 abstraction 给解决掉了。我会把所有技术细节都摊开讲，并尽量 build 你的 intuition。

---

## 1. Paper 的一句话定位

StarVLA 是 HKUST Von Neumann Institute + 社区维护的 **开源 VLA codebase**，时间戳是 2026 年 4 月。核心命题是：现有的 VLA 方法 (π0, GR00T, OpenVLA-OFT, FAST, CogACT, SpatialVLA …) 在 architecture / codebase / evaluation 三个层面都互不兼容，形成 "Tower of Babel"，让 fair comparison 几乎不可能。StarVLA 用 **backbone-action-head 解耦 + unified I/O contract + unified server-client eval** 把这些统统拉到一个 abstraction 下。

GitHub: https://github.com/starVLA/starVLA
Project page: https://starvla.github.io

---

## 2. 核心问题诊断：为什么需要这个 codebase

Paper 在 Section 1 诊断了 fragmentation 的三个层次，我觉得这个 diagnosis 是整篇 paper 最有价值的部分：

**(1) Architecture-level fragmentation**：
- VLM-native 方法：autoregressive tokenization (FAST), parallel regression (OFT)
- Generative-model-based 方法：diffusion / flow matching (π0, GR00T)
- 这两类方法的 action representation、loss 形式、inference 路径完全不同，很难放在一起比

**(2) System-level fragmentation**：
- 每个 release 把 model architecture + data processing + training pipeline 焊死在一起
- 想复用 OpenPI 的 backbone + π0 的 action head 几乎不可能

**(3) Evaluation-level fragmentation**：
- LIBERO 用 500 trials，SimplerEnv 用 Visual Matching / Variant Aggregation，RoboTwin 用 clean + randomized
- 不同 paper 报告不同 subset，方差还很大（SimplerEnv 自己承认 variance 非平凡，作者跑了 5 次取均值）

这个 diagnosis 我觉得非常对。过去两年 VLA 文献最让我头疼的就是 **"这个数到底是怎么报出来的"** —— preprocessing pipeline 各家不一样，action normalization 各家不一样，连 chunking 策略都不一样。

---

## 3. 核心抽象：Equation (1) 与 (2)

这是整篇 paper 最 deep 的部分，作者把它叫 **"generalized VLA perspective"**。

### 3.1 Policy 定义 (Eq. 1)

$$
\pi(\mathbf{a}_{t:t+k}, \mathbf{y}_{\text{aux}} \mid \mathbf{x}_{\leq t}, \boldsymbol{\ell})
$$

变量含义：
- $\mathbf{x}_{\leq t} = \{o^{\text{vis}}_{<t}, o^{\text{depth}}_{<t}, o^{\text{tactile}}_{<t}, \ldots\}$：到时间 $t$ 为止的多模态观测历史。上标 vis/depth/tactile 标注模态类型，下标 $<t$ 表示历史窗口
- $\boldsymbol{\ell}$：language instruction（粗体表示它是 token sequence）
- $\mathbf{a}_{t:t+k}$：从 $t$ 到 $t+k$ 的 **k-step action chunk**（chunk-based control，diffusion policy 范式）
- $\mathbf{y}_{\text{aux}}$：optional auxiliary outputs，可以是 future observations $o^{\text{vis}}_{t+1:t+k}$、intermediate reasoning $\ell_{\text{plan}}$、sub-goal descriptions

关键 insight：这个 formulation **把 VLM-based 和 WM-based 统一了**。WM-based 方法（如 Cosmos Policy, Dreamdojo）预测 future observation，本质上就是 $\mathbf{y}_{\text{aux}} = o^{\text{vis}}_{t+1:t+k}$ 的特例。VLM-based 方法的 chain-of-thought reasoning 就是 $\mathbf{y}_{\text{aux}} = \ell_{\text{plan}}$。

### 3.2 Unified Loss (Eq. 2)

$$
\mathcal{L} = \mathcal{L}_{\text{action}} + \mathcal{L}_{\text{aux}}
$$

- $\mathcal{L}_{\text{action}}$：监督 action 预测，通常是 L1 (OFT), flow-matching objective (π0), cross-entropy on action tokens (FAST)
- $\mathcal{L}_{\text{aux}}$：inductive bias，决定 paradigm 走向
  - **Direct VLA**：$\mathcal{L}_{\text{aux}} = 0$，纯 action supervision（OpenVLA, OpenVLA-OFT）
  - **VLM-based VLA**：$\mathcal{L}_{\text{aux}}$ = language modeling loss on reasoning tokens（Magma, InstructVLA, Hi-Robot）
  - **WM-based VLA**：$\mathcal{L}_{\text{aux}}$ = future image/video prediction loss（Cosmos Policy, FLARE, GigaWorld）

这个视角的真正威力在于：**当你把 infrastructure 差异抹平后，VLM-based 和 WM-based 之间的差别不是 paradigm 级别的，而是 auxiliary signal 形式的差别**。这让我想起你之前在 Neural Nets: Zero to Hero 里讲的——很多看起来很不一样的架构，其实在 loss landscape 上是邻居。

---

## 4. 工程抽象：Unified I/O Contract

这是 StarVLA 真正的工程贡献。所有 framework module 继承自同一个 base class，暴露两个方法：

```
forward({raw images, str, ...}) → {loss_dict}
predict_action({raw images, str, ...}) → {normalized_actions}
```

**关键设计决策**：training input 是 raw environment-level observations（与 deployment 时 robot 收到的一致），而不是 heavily preprocessed dataloader tensors。

为什么这个 matters？因为 **train/test distribution mismatch** 是 VLA 系统里 silent performance degradation 的主要来源。每个 VLM backbone 有自己的 image tokenization scheme（Qwen-VL 用 14×14 patch + window attention，Cosmos 用 video-native tokenization），如果你训练时把 preprocessing 做成 dataloader 的一部分，部署时往往要重新实现一遍，非常容易出 bug。

这个 invariant-driven design 的哲学让我想起 HF transformers 的 `PreTrainedModel.forward()` 契约，但更严格——它要求 contract 是 raw sensor stream 到 executable motor command。

---

## 5. Compositional Architecture：Backbone-Head 解耦

### 5.1 两层 boundary

StarVLA 定义了两个标准化 contract：

**(1) Outer boundary**：raw observations → actions
**(2) Inner boundary**：multimodal inputs → hidden states → actions

Inner boundary 通过 standardized representation contract 连接 backbone 和 action head，所以两者可以独立替换。配置 declarative YAML，像搭 Lego 一样：

```yaml
backbone: qwen3-vl-4b  # or cosmos-predict2-2b, internvl-a1
action_head: oft        # or fast, pi, gr00t
```

### 5.2 四种 Action Head 的本质差异

| Paradigm | Action Rep | Decoding | Loss | Compute Pattern |
|---|---|---|---|---|
| **FAST** | Discrete tokens | Autoregressive next-token | Cross-entropy | Sequential, single-pass |
| **OFT** | Continuous | Parallel MLP regression | L1 | Single forward |
| **π** | Continuous | Iterative flow-matching denoising | Flow-matching objective | Iterative (10-20 steps) |
| **GR00T** | Continuous | Dual-system: System 2 reasoning + System 1 DiT flow-matching | Flow-matching + LM | Two-stage inference |

**FAST** ([Pertsch et al., 2025](https://arxiv.org/abs/2501.09747))：把 action chunk 通过 DCT + bpe-style tokenizer 离散化到 LLM 的 vocabulary 空间，然后用 next-token prediction 训练。优点是直接复用 LLM 的 autoregressive inference，缺点是 discrete quantization 损失精度。

**OFT** ([Kim et al., 2025](https://arxiv.org/abs/2502.19645))：最简单的 head。在 VL backbone 输出里 predefined 一些 "action token positions"，读取这些位置的 hidden states，过一个 MLP 回归 continuous action。L1 loss。这个设计有意思的地方在于它把 action prediction 完全解耦成 representation learning + regression 两个阶段。

**π** ([Black et al., 2024](https://arxiv.org/abs/2410.24164))：layer-wise cross-DiT flow-matching action expert。关键技术点：
- Flow matching 比 DDPM-style diffusion 更高效（更少 denoising step）
- Cross-attention 让 action DiT condition on **multi-layer** VL hidden states，而不是只取 last layer
- 这是 paper 里没明说但很重要的细节：multi-layer conditioning 保留 low-level spatial info，而 last layer 主要是 semantic

**GR00T** ([Bjorck et al., 2025](https://arxiv.org/abs/2503.14734))：dual-system 设计，VL backbone 当 System 2 (慢 reasoning)，DiT flow-matching 当 System 1 (快 action)。这直接借用了 Kahneman 的 System 1/2 框架。在 inference 时，System 2 先产生 reasoning context（可能是 implicit 的，通过 hidden states），System 1 fast generation。

### 5.3 Backbones：VLM vs World-Model

这是 StarVLA 的另一个 unification：**VLM backbone 和 World-Model backbone 可以互换**。

- **VLM backbone**：Qwen2.5-VL / Qwen3-VL ([Bai et al., 2025](https://arxiv.org/abs/2502.13923)), InternVL-A1 ([Cai et al., 2026](https://arxiv.org/abs/2601.02456))
- **World-Model backbone**：Cosmos-Predict2 ([Kim et al., 2026](https://arxiv.org/abs/2601.16163)), Dreamdojo ([Gao et al., 2026](https://arxiv.org/abs/2602.06949))

WM backbone 的本质区别：它预训练时就 condition on action，预测未来帧，所以 hidden states 里编码了 dynamics 信息。VLM backbone 只编码了视觉语义，dynamics 信息要在 robot data 上从头学。Table 2 里 Cosmos-Predict2-2B 在 LIBERO 上能 match Qwen3-VL-4B，说明在 LIBERO 这种 closed-world 简单任务上 dynamics prior 没那么关键，但在 long-horizon 或 OOD 场景下应该更有优势（paper 没直接 ablation 这一点，是 limitations 之一）。

---

## 6. 训练范式细节

### 6.1 SFT (Section 3.1.1)

实现路径：`starVLA/training/train_starvla.py`。优化设置：
- Full-parameter fine-tuning OR selective freezing via `trainer.freeze_modules` (comma-separated paths)
- 多 parameter group，不同 LR（如 qwen_vl_interface 与 action model 分开调 LR）
- bfloat16 autocast + gradient accumulation + gradient clipping + cosine schedule with min LR

### 6.2 Multi-Modal Co-training (Section 3.1.2)

实现：`starVLA/training/train_starvla_cotrain.py`。

**双 dataloader multi-objective**：每步做两次 forward/backward
- VLA forward through `framework.forward()` → `action_loss`
- VLM forward through `qwen_vl_interface` → `language_modeling_loss`
- VLM loss 缩放系数：`trainer.loss_scale.vlm`

**Section 6 的关键实验**（Figure 4 + Table 8）：
- Vanilla VLA：在 20K steps 内 RefCOCO-g 性能掉到接近 random！这是 catastrophic forgetting 的典型表现
- Vanilla co-training：部分保留 perception，但震荡
- ST4VLA spatially guided ([Ye et al., 2026a](https://arxiv.org/abs/2602.10109))：保留 ~70% grounding 性能的同时 manipulation 强

Figure 4c 里的 **Projection-Space Similarity (PSS)** 是个值得注意的指标——它衡量 spatial grounding gradient 与 action objective gradient 在 subspace 上的对齐程度。当两个 objective 的 gradient direction 对齐时，co-training 会双赢；不对齐时互相干扰。这个分析让我想起 Multi-Task Learning 里 gradient surgery 的工作（如 [GradVac](https://arxiv.org/abs/2104.02291), [PCGrad](https://arxiv.org/abs/2001.06782)）。

### 6.3 Cross-Embodiment Co-training (Section 3.1.3)

通过 LeRobot mixture dataset 接口，配置成 `(dataset_name, sampling_weight, robot_type)` tuples。运行时 `LeRobotMixtureDataset` 按 weight 采样并追踪 embodiment tag。

**Generalist 实验的关键 trick**（Section 7.1）：为了处理不同 embodiment 的 action space 差异（单臂 7-DoF，双臂 14-DoF，人形可能更多），他们用 **unified padding 到 32-dim action vector** 的策略。低 DoF action padding 到 32 维，高 DoF 截断或映射。这是个简单粗暴但 effective 的方法，让我想起 Decision Transformer 里对 return-to-go 的 tokenization。

Generalist 训练设置：LR = 1e-4, global batch = 256, joint training on LIBERO + SimplerEnv + RoboTwin 2.0 + RoboCasa-GR1。

---

## 7. Evaluation：Server-Client Architecture

### 7.1 设计哲学

StarVLA 用 **thin WebSocket policy server + 轻量 client wrapper**。Check point 通过 `base_framework.from_pretrained()` 加载，hosted 在 StarVLA runtime 里。Benchmark evaluator 在自己的 conda env 里（LIBERO, SimplerEnv, RoboTwin 各自依赖不兼容），通过 client wrapper 通信。

Inference interface 统一：`Framework.predict_action()`。Client 把 observations 打包成 dict (`image`, `lang`, optional `state`, `timestamps`, `episode metadata`)，msgpack 序列化发到 server，server返回 dict 包含 `normalized_actions`。

### 7.2 Benchmark-specific adapters

差异隔离在 `model2libero_interface.py`, `model2simpler_interface.py`, `model2robotwin_interface.py` 等文件里。这些 adapter 干的事：
- Image resize 到训练 resolution
- 读 `dataset_statistics.json` 做 action unnormalization
- Chunked normalized predictions → executable actions
- Action ensembling（diffusion policy 的 trick）
- Benchmark 特定 conventions：sticky grippers, delta→absolute action 转换

**这个设计让 sim-to-real 几乎无缝**：real robot controller 就是另一个 benchmark client，捕获 camera obs，组装 example dict，query remote server，执行 action。Robot-side 的 ROS node / vendor SDK 不需要进 StarVLA runtime。

---

## 8. 实验结果深度解读

### 8.1 LIBERO (Table 2)

最 striking 的对比：

| Model | Steps | Epochs | Avg |
|---|---|---|---|
| OpenVLA-OFT | 175K | 223 | 97.1 |
| π0+FAST | - | - | 85.5 |
| GR00T-N1.5 | 20K | 203 | 86.5 |
| **StarVLA-OFT (Qwen3-VL-4B)** | **30K** | **9.54** | **96.6** |
| StarVLA-OFT (Cosmos-Predict2-2B) | 30K | 9.54 | 95.8 |

**关键 insight**：StarVLA-OFT 只用 30K steps (9.54 epochs) 就接近 OpenVLA-OFT 175K steps (223 epochs) 的性能。这是 **6× fewer steps, 23× fewer epochs**。说明：
- Qwen3-VL-4B 作为 backbone 比 OpenVLA 用的 Prismatic-7B backbone 在 LIBERO 上更 sample-efficient
- 30K 步只跑了 ~10 epoch，意味着 backbone representation 几乎不需要大幅 shift，主要是 action head 的对齐
- Cosmos-Predict2-2B 同样能到 95.8%，说明 **backbone 的选择空间很大**

但要注意：Long-horizon (LIBERO-Long) 上 StarVLA-OFT 是 93.8 vs OpenVLA-OFT 94.5，差距稍大。说明 long-horizon 还是需要更多训练或者更好 reasoning supervision。

### 8.2 SimplerEnv (Tables 3 & 4)

SimplerEnv 是 real-world proxy，方差大。作者跑 5 次取均值，这个是 honest 的。

**WidowX VM**：
- StarVLA-GR00T 65.3% (Qwen3-VL-4B)
- StarVLA-OFT 61.6% (Cosmos-Predict2-2B)
- 对比 π0-FAST 48.3%, GR00T N1.5 35.8%, Magma 61.9%

**Google Robot**：
- StarVLA-OFT: VM 76.0, VA 70.2
- CogACT 74.8 / 61.3
- π0 58.8 / 54.8
- GR00T N1.5 35.2 / 44.5

Google Robot 上 StarVLA-OFT 几乎全面领先，VM 上比第二名 CogACT 高 1.2 pt。在 Put Coke Can 任务上达到 95.3%（VM）/ 91.3%（VA），比 RT-1 的 85.7% / 89.8% 都高。

注意 "Open Top Drawer and Place Apple" 这个 long-horizon 任务，StarVLA-OFT 是 66.1% (VM) / 59.4% (VA)，其他方法基本都 < 25%，这里 StarVLA 优势巨大。我推测是 Qwen3-VL 的 instruction following 能力在这里帮了大忙。

### 8.3 RoboCasa-GR1 (Tables 5 & 6)

这是 humanoid-style manipulation，比 LIBERO 难得多：

- StarVLA-OFT 48.8% > GR00T-N1.6 47.6% > StarVLA-GR00T 47.8% > StarVLA-π 43.9% > StarVLA-FAST 39.0% > π0.5 37.0%

**关键 insight**：在更难的任务上 **action head 选择 matters more**。Continuous action (OFT/π/GR00T) 都比 discrete (FAST) 显著好。FAST 落后 ~10 pt，我推测是因为 high-precision bimanual 任务对 action quantization 误差更敏感。

Table 6 的 task-level 数据有意思：PnPCanToDrawerClose 上 StarVLA-GR00T 80% vs GR00T-N1.6 13%。但 PnPBottleToCabinetClose 上反过来。这种 task-level 的 specialization 差异暗示 generalist 在某些 task 上反而比 specialist 好（可能因为 cross-task transfer）。

### 8.4 RoboTwin 2.0 (Table 7)

Bimanual 任务：
- StarVLA-π 88.1/88.8 (clean/random)
- StarVLA-OFT 88.2/88.3
- StarVLA-GR00T 88.0/88.5
- Lingbot-VLA 88.6/86.7（最强 baseline）
- π0.5 82.7/76.8

注意 random setup 上 StarVLA 比 π0.5 高 12 pt，**robustness 优势明显**。

### 8.5 Generalist vs Specialist (Table 9)

这是 paper 的 hero result：

| Settings | LIBERO avg | SimplerEnv WidowX | SimplerEnv Google VA/VM | RoboTwin clean/random | RoboCasa |
|---|---|---|---|---|---|
| StarVLA-OFT (specialist) | 98.8 | 64.6 | 70.2/76.0 | 88.2/88.3 | 53.8 |
| **Generalist StarVLA** | 97.8 | **70.2** | **73.8/79.3** | 88.7/87.8 | **57.3** |

**最 striking 的发现**：Generalist 在 SimplerEnv 和 RoboCasa 上 **比所有 specialist 都强**！WidowX +5.6 pt, Google VA +3.6 pt, Google VM +3.3 pt, RoboCasa +3.5 pt。

这暗示了：**cross-benchmark joint training 帮助了 generalization**，反过来也说明 specialist 容易 overfit 到自己 benchmark 的 distribution。LIBERO 上掉了 1 pt，但其他都涨了，这是非常 healthy 的 trade-off。

---

## 9. 计算效率分析 (Section 8)

这 section 对 practitioner 最实用。

### 9.1 Single-node (8×A100, Table 10)

| Per-GPU batch | Global batch | s/step | samples/s | GPU util |
|---|---|---|---|---|
| 2 | 16 | 0.703 | 22.7 | 74% |
| 8 | 64 | 1.131 | 56.6 | 92% |
| 24 | 192 | 2.404 | 79.9 | 96% |

**关键 trade-off**：大 batch 提升 sample throughput 但 linearly 增加 step latency。batch 24 vs 2 在 sample throughput 上 3.5×，但 step latency 3.4×。GPU util 从 74% → 96% 几乎饱和。

Practical guidance：**batch 8 是 sweet spot**（92% util, 1.13s/step）。

### 9.2 Multi-node (Table 11)

| # GPUs | s/step | samples/s | Scaling eff |
|---|---|---|---|
| 8 | 0.735 | 87.0 | 100% |
| 32 | 0.899 | 284.7 | 81.9% |
| 64 | 0.925 | 553.8 | 79.6% |
| 128 | 0.921 | 1111.5 | 79.9% |
| 256 | 0.931 | 2200.0 | 79.1% |

**关键 insight**：inter-node communication 是 **one-time latency overhead**（0.735→0.93 s/step），过了 8 节点 (64 GPU) 后 plateau，再扩展不再降效。这意味着 **可以放心 scale 到几百 GPU，scaling efficiency 稳定在 79-80%**。

让我做个估算：260M trajectories * 32-dim action * 8 chunk ≈ 看起来 256 GPU 跑 100K steps = 25.5 小时能处理 2200 samples/s × 91800s ≈ 2 亿样本。这对 cross-embodiment large-scale pretraining 是可行的。

---

## 10. 相关联想与 broader perspective

### 10.1 与 HF transformers 的关系

StarVLA 的 backbone-head 解耦哲学与 HF transformers 的 `PreTrainedModel` + task-specific head 一脉相承，但更严格——它要求 raw observation contract，强制 preprocessing 进 model 内部。这是 VLA 特殊的需求，因为 robot deployment 时 sensor stream 必须是 raw 的。

如果未来 StarVLA 被广泛采纳，VLA backbone 的发布格式可能标准化为 "preprocessed observation → hidden state" 接口，类似 BERT 的 "tokenized input → contextualized embedding"。这会进一步降低 method comparison 的 friction。

### 10.2 与 OpenPI / Isaac-GR00T / OpenVLA-OFT 的对比 (Table 1)

Table 1 的对比维度很关键：
- OpenPI：功能少，只支持 2 个 sim benchmarks
- Isaac-GR00T：6 个 benchmarks，但 no modular action heads（架构焊死）
- OpenVLA-OFT：modular VLM 但 no mixture dataloader
- Dexbotic：5 benchmarks 但 no open-source co-training
- X-VLA：modular action heads 但 no modular VLM
- **StarVLA**：所有维度全 ✓，7 个 benchmarks，multi-bench co-train

### 10.3 Generalized VLA perspective 的更深层含义

Paper 提的 "generalized VLA perspective" 暗示了一个 roadmap：
- VLM-based 和 WM-based 不是 fundamentally distinct paradigms
- 它们 differ 主要在 $\mathcal{L}_{\text{aux}}$ 的形式
- 未来可能出现 **hybrid auxiliary signals**：language reasoning + future frame prediction + action all in one loss

这让我想到 [World Action Models are Zero-Shot Policies](https://arxiv.org/abs/2602.15922) (Ye et al., 2026b) 和 [Causal World Modeling for Robot Control](https://arxiv.org/abs/2601.21998) 这些工作，它们已经在探索 joint video-action prediction 作为 policy learning 的 formulation。StarVLA 的 framework 抽象正好 accommodate 这些新方法。

### 10.4 Limitations（paper 没明说但我觉得重要的）

1. **没做 backbone 的 fine-grained ablation**：Qwen3-VL-4B vs Cosmos-Predict2-2B 在 LIBERO 上 comparable，但在 long-horizon / OOD 上谁更强？没有专门 ablation
2. **Unified padding 32-dim 的 information loss**：对真正高 DoF 人形（如 60-DoF full body）可能不够
3. **没报告 real-robot 实验**：所有结果都在 sim 上，尽管 architecture 支持 sim-to-real。但 paper 自己说"closing the gap between research exploration and practical deployment"，没给实证
4. **Co-training 用的是 ST4VLA 的研究**：spatially guided 的具体实现细节在另一篇 paper 里，StarVLA 这里只提供 infrastructure

### 10.5 与 Karpathy 自己工作的联想

你之前在 Eureka Labs / nanoGPT / "Software 2.0" essay 里强调过：很多看起来不同的方法在 abstraction 上是同构的。StarVLA 的 generalized VLA perspective 正是这个哲学——把 paradigm-level 表象差异 abstract 掉，露出 loss formulation 的共性。这跟 Software 2.0 把 "differentiable programming" 作为统一视角是同一思路。

另外，你讲过的 "micrograd" / "makemore" 系列强调 **从第一性原理理解 architecture**，StarVLA 的 unified formulation 给了 VLA 一个 first-principles 视角：所有 VLA 都是 $\pi(\mathbf{a}, \mathbf{y}_{\text{aux}} | \mathbf{x}, \ell)$ 的 instantiation。

### 10.6 可能的延伸方向

- **Action head 作为 RL policy**：paper 提到 RL fine-tuning 是 ongoing integration with [RLinf](https://github.com/RLinf/RLinf)，未来 action head 可能变成 RL-learned policy
- **Test-time scaling for action generation**：dual-system GR00T 的 reasoning + action 范式可以扩展成 test-time search，类似 OpenAI o1 在 VLM 上的 reasoning
- **WM backbone + Action head 的最佳 combination**：哪种 action head 最适合 WM backbone？paper 没系统 ablation，但 Cosmos + OFT 在 LIBERO 上 95.8%，Cosmos + π 在 SimplerEnv WidowX 上 58.7%，看起来 OFT 在简单任务、π 在复杂任务占优
- **Active learning / DAgger integration**：paper 明确 avoid DAgger 等在线 refinement，但作为 baseline 这是 honest 的，未来可以加

---

## 11. Reference 汇总

**StarVLA 本身**:
- Codebase: https://github.com/starVLA/starVLA
- Project page: https://starvla.github.io

**Backbones**:
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Cosmos Policy: https://arxiv.org/abs/2601.16163
- InternVL-A1: https://arxiv.org/abs/2601.02456

**Action heads**:
- FAST tokenizer: https://arxiv.org/abs/2501.09747
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- π0: https://arxiv.org/abs/2410.24164
- GR00T N1: https://arxiv.org/abs/2503.14734

**Benchmarks**:
- LIBERO: https://arxiv.org/abs/2305.12248 (Liu et al., 2024a)
- LIBERO-Plus: https://arxiv.org/abs/2505.24502 (Fei et al., 2025)
- SimplerEnv: https://arxiv.org/abs/2405.05941
- RoboCasa: https://arxiv.org/abs/2406.02523
- RoboTwin 2.0: https://arxiv.org/abs/2504.13059
- BEHAVIOR-1K: https://arxiv.org/abs/2403.09228
- CALVIN: https://arxiv.org/abs/2112.03227

**Co-training & related studies**:
- ST4VLA (spatially guided): https://arxiv.org/abs/2602.10109
- OpenVLA: https://arxiv.org/abs/2406.09246
- CogACT: https://arxiv.org/abs/2411.19650
- SpatialVLA: https://arxiv.org/abs/2501.15830
- Magma: https://arxiv.org/abs/2502.13130

**World model / VLA unification**:
- World Action Models: https://arxiv.org/abs/2602.15922
- Causal World Modeling: https://arxiv.org/abs/2601.21998
- Dreamdojo: https://arxiv.org/abs/2602.06949
- FLARE: https://arxiv.org/abs/2505.15659
- V-JEPA 2: https://arxiv.org/abs/2506.09985

**Cross-embodiment / Open X-Embodiment**:
- OXE: https://arxiv.org/abs/2310.08864
- DROID: https://arxiv.org/abs/2403.12945

**Gradient analysis for multi-task learning**:
- PCGrad: https://arxiv.org/abs/2001.06782
- GradVac: https://arxiv.org/abs/2104.02291

---

## 12. 最终总结

StarVLA 不是 method paper，而是 **infrastructure-as-research**。它真正的贡献：

1. **Conceptual**：证明 VLM-based / WM-based / Direct policy 可以放进一个 policy formulation (Eq. 1-2)，差别仅在 $\mathcal{L}_{\text{aux}}$ 形式
2. **Engineering**：raw-observation I/O contract + backbone-head 双层解耦，让 Lego-style composition 成为可能
3. **Empirical**：用同一套 infrastructure 跑出 5 个 benchmark 的 strong baseline，证明 framework 不引入性能 tax（甚至超过原方法）
4. **Generalist evidence**：cross-benchmark joint training 在 4 个 benchmark 上比 specialist 还强（Section 7），暗示 cross-task transfer > per-task overfit

如果你 2026 年要做 VLA 研究，StarVLA 几乎是必用的 codebase。它的 limitation 主要是 real-robot evidence 缺失，以及一些 fine-grained ablation 没做。但作为 community infrastructure，这是过去几年 VLA 领域最重要的工程贡献之一。
