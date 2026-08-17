---
source_pdf: VLA Foundry A Unified Framework for.pdf
paper_sha256: ccd7f4cde47f0ff00e0561ed895ac8af37f6b8b8d349c9a7e457913553924785
processed_at: '2026-08-13T02:40:12-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 VLA Foundry

## 这篇paper到底在干嘛

一句话：TRI做了一个训练框架，能让你用同一套代码，从语言模型一路训到机器人动作模型，中间不用换工具。

## 为什么这是个事儿

现在做VLA的人，workflow基本上是这样：去HuggingFace下载一个别人训好的VLM（比如Qwen-VL、PaliGemma），拿过来当backbone，然后在上面接一个action head，用机器人数据微调。代码也只覆盖这最后一阶段。

这有一个问题：你无法干预backbone是怎么训出来的。VLM用的什么image-text数据、语言模型阶段吃了什么corpus、lr schedule长什么样——全是黑盒。

当你想研究"如果在VLM pretraining阶段混入robotics-related的文本会不会提升下游VLA性能"这种问题时，你做不了实验。你得分别用三个不同的codebase：一个训LLM，一个训VLM，一个训VLA。每个codebase的数据格式、训练循环、分布式策略都不一样，拼起来很痛苦，也很难复现。

VLA Foundry就是把这个三段式的pipeline塞进一个codebase。同一个training loop、同一套dataloader、同一套config系统，从raw text一路训到flow matching action head。

## 他们怎么做到的

核心就几个设计决策：

**Config系统用frozen dataclass + YAML继承**。你写一个YAML，可以include另一个YAML，CLI参数再覆盖YAML。frozen的意思是config对象创建后不能改——避免你训练中途某个模块偷偷改了config，最后log和实际run对不上。

**Model和dataset按名字注册**。加新模型就是写个dataclass + 一个factory function，挂个装饰器，不用动中心配置文件。runtime根据`model.type`字符串dispatch。

**BatchHandler抽象**。LLM、VLM、VLA各自的batching、loss计算、output reduction逻辑封装在handler里，training loop本身跟model类型无关。同一个loop驱动三个阶段。

**数据存成WebDataset的tar shards**。一个tar里每个sample用unique prefix区分文件。加新modality（比如depth image）就加新文件，不改schema。

## Robotics数据处理里那些细节

这部分其实是最扎实的。讲几个关键的：

**Action可以选absolute或relative**。Absolute是world frame下的pose，relative是相对于当前时刻机器人pose的偏移。relative用SE(3)群运算算：$T_t^{\text{rel}} = T_{\text{ref}}^{-1} \cdot T_t$。直觉上relative更稳定，因为分布不依赖机器人初始位置。

**Rotation用6D连续表示**（Zhou et al. 2019）。网络输出6个数，用Gram-Schmidt正交化还原成3×3 rotation matrix。这避免了quaternion的double-cover不连续和Euler角的gimbal lock。

**Per-timestep normalization**。预测窗口里每个相对offset有自己的mean和scale。对relative action特别重要——预测"未来第1步"和"未来第10步"的位移分布差异很大。

**跨数据集合并统计量用t-digest**。Percentile无法从summary stats精确合并，所以每个数据集存一个t-digest sketch，训练时merge sketches恢复pooled distribution的近似分位数。这是个很优雅的工程细节。

## Action head用flow matching

不是diffusion，是flow matching。直觉区别：diffusion是加噪再去噪（SDE），flow matching是学一个vector field把Gaussian直接流到数据分布（ODE）。数学上相关但训练目标更简单——直接回归一个vector $v(x_t, t) = x_1 - x_0$，其中$x_t = (1-t)x_0 + t x_1$是线性插值。轨迹更直，采样步数更少。

Action head是个325M的transformer，输入三部分concat：VLM最后4层在observation token位置的hidden features + proprioception编码 + noised action编码。输出predicted denoising direction。

Observation token是个新加到LLM vocab的特殊token。VLM输入是`[image patches] + [task text] + [observation token]`，所有视觉和文本信息在LLM里处理完后，在observation token位置的特征被抽出来给action head做conditioning。用最后4层而不是最后1层——不同层提供不同抽象层级的representation。

## 他们实际训出来的两个模型

**Foundry-VLA-1.7B**：完全from scratch。先训1.2B LLM（DCLM数据，1T tokens），再接86M ViT训VLM（DataCompDR-1B，200M samples），再加observation token + 325M flow transformer训VLA。中间checkpoint全开源。

**Foundry-Qwen3VLA-2.1B-MT**：用预训练的Qwen3-VL 2B当backbone，action head架构跟上面一样，同样的数据recipe。

## 结果说明了什么

在LBM simulation benchmark（49个bimanual manipulation tasks，用Drake物理引擎）上：

1. **Qwen3-VLA比LBM（前作566M model）高20+个百分点**，统计显著。
2. **from scratch的1.7B model和LBM统计上on par**。
3. **Qwen3-VLA的multi-task → finetune比single-task好**；from scratch的1.7B没这个优势。
4. **小model同时训real+sim数据，反而比只训sim略差**。这有点反直觉，可能model capacity不够支撑多domain。

核心takeaway：**当前阶段，用强VLM backbone比从头训全pipeline更划算**。但paper强调from scratch pipeline的价值是可研究性，不是当下SOTA追求。

## 统计评估这块很讲究

用STEP工具做policy比较。不是简单报success rate，而是：
- Bayesian posterior估计每个policy的success rate分布（Beta分布，violin plot可视化）
- Compact Letter Display标注显著性（不共享字母的两组在5% FWER下显著不同）
- 允许中途peeking决定要不要继续collect rollout——这个在传统hypothesis testing里算p-hacking，但STEP的anytime-valid Bayesian framework允许

跨task聚合时强制平衡每个policy在每个task上的样本量。比如A在某task少一个rollout，就把所有task都截断到那个数量再聚合。保证aggregate是无偏的。paper说前作LBM没严格执行这个。

## 我的直觉理解

这篇paper的真正价值不在model性能本身，而在基础设施投资。几个值得关注的thread：

1. **End-to-end controllability**是稀缺品。现在能干预VLM pretraining recipe的团队极少。VLA Foundry让"VLM阶段数据里加robotics文本会怎样"这种问题变得可实验。

2. **当前阶段strong VLM backbone > from scratch全pipeline**。但from scratch的可控性是研究价值，不是工程效率。

3. **Flow matching + multi-layer hidden features做conditioning**是elegant的设计，类似ControlNet用multi-scale features的思路。

4. **统计rigor在robotics policy评估里被严重低估**。很多paper报的success rate差异在统计上根本不显著，但读者看不出来。STEP这套工具值得推广。

5. **小model上multi-domain数据可能有害**——real+sim混合反而比sim-only差。这和部分scaling研究的结论可能冲突，值得深究。

如果要我挑一个最值得follow的点：跨modality的data scaling law。VLA Foundry的multi-modal mixing + 全pipeline可控，正好是研究"LLM/VLM/VLA三阶段数据配比如何影响最终policy"的理想工具。这个方向目前基本是空白。

---

# VLA Foundry 深度技术讲解

## 1. Paper 的核心定位

VLA Foundry 是 Toyota Research Institute (TRI) 在 2026 年 4 月发布的一个开源训练框架，核心贡献在于**统一了 LLM、VLM、VLA 三个训练阶段到单一 codebase 中**。这一点在当前开源生态里相当独特，因为绝大多数 open-source VLA 工作（OpenVLA [30]、OpenPi [50]、GR00T [46]、LeRobot [10]）只专注于 action training stage，把上游的 pretraining recipe 当成固定输入。VLA Foundry 把从 backbone pretraining 到 action expert fine-tuning 的整条链路当成一个可控系统，研究者可以在任何阶段介入。

这种设计动机源于一个核心观察：**robotics 数据相对 language/vision 数据极度稀缺**，per-token 的 signal density 也低。随着 policy 规模扩大，non-robotics 数据的相对重要性只会增加，LLM/VLM pretraining 阶段做的数据决策会直接影响下游 robotics 性能。要研究这些 scaling 问题，就需要一个能贯穿全 pipeline 的框架。

项目主页：https://tri-ml.github.io/vla_foundry  
代码：https://github.com/TRI-ML/vla_foundry  
模型权重：https://huggingface.co/collections/TRI-ML/vla-foundry

## 2. 框架设计原则（Section 3.1）

VLA Foundry 明确提出了四条原则，下面我结合具体实现展开：

### 2.1 Modularity and Composability
配置系统基于 **Draccus** [42]（https://github.com/marin-community/draccus），每个参数声明在 frozen dataclass 中，可以被 YAML preset 或 CLI 参数覆盖，优先级递增：

```
CLI flag > YAML preset > nested include > dataclass default
```

关键设计是 **frozen dataclass**——配置对象一旦创建就不可变，避免 runtime 静默修改导致 config/log/runtime 三者不一致。YAML 支持 `include` 关键字做嵌套继承，一个实验通过组合 building blocks 表达，而非复制粘贴。共享参数（hidden_dim、seq_len 等）在 dataclass tree 中解析一次并向下传播，防止跨模块 silent mismatch。

### 2.2 Hackability and Interoperability
刻意避免重型 framework wrapper（PyTorch Lightning、HF Trainer），训练循环保持 thin。parallelism primitives（FSDP、gradient accumulation）暴露给用户而非隐藏。要添加新模型只需两步：定义 frozen dataclass 描述 hyperparameters，写一个 factory function 用 `@register_model` 装饰器注册。整个 registry 通过 `model.type` 字符串在 runtime dispatch。

### 2.3 Performance
基于 **FSDP2** 做分布式训练（注意是 FSDP2，PyTorch 较新的实现，比 FSDP1 更高效），支持 optional CPU offloading、mixed precision、gradient checkpointing、`torch.compile`、gradient accumulation、EMA。在 128 GPUs / 16 nodes 上 benchmark 过 LLM/VLM/VLA 三阶段的 throughput（Figure 2）。

### 2.4 Reproducibility
- Deterministic per-rank RNG seeding
- Dataloader state checkpointing 支持 exact restart
- Frozen dataclasses 防止 runtime config 漂移
- Training budget 用 **samples 数** 而非 steps 表达，这样不同 batch size / GPU count 的 run 可以直接对比

## 3. 框架架构细节（Section 3.2 + Appendix A）

### 3.1 四层架构
1. **YAML 配置系统**（Draccus + frozen dataclasses）
2. **Registry**（model / dataset / batch handler 都按名字注册）
3. **Modality-specific preprocessing + dataloading**（text / image-caption / robotics）
4. **Model-agnostic training loop**

`BatchHandler` 是一个关键抽象——它拥有 batching、loss construction、output reduction 三个职责。LLM/VLM/DP-VLA 各有共享的 handler，新增训练范式只需 `@register_batch_handler` 注册一个新的 handler。

### 3.2 数据管线（WebDataset）
数据存储为 **tar shards**，每个 sample 在 tar 内用 unique prefix 区分。目录结构：

```
dataset_name/
├── manifest.jsonl
├── shard_00000000.tar
│   ├── unique_name_1_camera1.jpg
│   ├── unique_name_1_camera2.jpg
│   ├── unique_name_1_meta.json
│   └── unique_name_1_actions.npz
├── shard_00000001.tar
└── ...
```

这种设计可扩展——加 depth image 就加 `unique_name_1_depth1.jpg`，加 video 就加对应文件，不需要改 schema。

Pipeline 用 WebDataset 的组合式 API 构造，每个 stage 是一个小函数，用户可以独立 extend 或 reorder：

```python
pipeline = [
    wds.SimpleShardList(datastring),
    deterministic_shuffle(bufsize=..., seed=..., epoch=checkpoint_num),
    wds.split_by_node,      # 多节点切分
    wds.split_by_worker,    # 多 worker 切分
    wds.tarfile_to_samples(handler=log_and_continue),
    wds.decode("pilrgb", handler=log_and_continue),
    wds.select(filter_no_caption_or_no_image),
    wds.map(augmentations.apply_transforms, ...),
    wds.rename(image="jpg;png;jpeg;webp", text="txt"),
    wds.map(lambda s: {**s, "text": "<image> " + s["text"]}),
    wds.batched(batch_size, partial=False),
    wds.map(processor, ...),  # tokenize / image preprocess
]
```

### 3.3 Dataset Mixing
通过 CLI list 参数天然支持多数据集混合。`--data.dataset_weighting` 控制每个数据集在 batch 中的比例，例如 `1:2:1` 对应 25%/50%/25%。这是个简单的 batch-level mixing，比 sample-level mixing 更可控。

### 3.4 预处理（Ray 并行）
预处理用 **Ray** [44] 并行化。Robotics 数据预处理分三阶段：
1. `frames/` —— 每个 sample 一个独立 tar
2. `episodes/` —— 按 episode 聚合 sample tar
3. `shards/` —— 随机聚合 sample tar，最终用于训练

`shards/` 下有 `manifest.jsonl`（shard 元信息）和 `stats.json`（per-dataset 统计量）。stats.json 的计算需要 worker 节点先在本地内存存统计量，再跨节点 gather。

## 4. Robotics 数据处理细节（Appendix A.2）—— 这部分是 robotics-specific 的精华

### 4.1 Normalization
`RoboticsNormalizer` 支持四种方法：
- **Standard deviation**：$\tilde{x} = (x - \mu) / \sigma$
- **Min-max**：$\tilde{x} = (x - x_{\min}) / (x_{\max} - x_{\min})$
- **Percentile 1-99**：用 1% 和 99% 分位数代替 min/max
- **Percentile 5-95**：用 5% 和 95% 分位数

Percentile-based normalization 对含 outlier 的 action field 很有用——避免大部分数据被压缩到窄带。

**Normalization scope** 有两种：
- **Global**：整个序列用同一组 mean/scale
- **Per-timestep**：window 内每个相对 offset 有自己的 mean/scale。对 relative action 特别有用，因为 prediction horizon 早期和晚期的 displacement 分布差异很大。

### 4.2 Statistics Merging 跨数据集合并
当同时训练多个数据集时，需要合并各自 stats.json。不同统计量的合并方式不同：

- **Means**：按 sample count 加权平均
  $$\bar{\mu}_{\text{overall}} = \frac{\sum_i n_i \mu_i}{\sum_i n_i}$$
  其中 $n_i$ 是第 $i$ 个数据集的样本数，$\mu_i$ 是其均值。

- **Standard deviations**：用 **law of total variance**
  $$\bar{\sigma}_{\text{overall}}^2 = \mathbb{E}[\sigma_i^2] + \text{Var}(\mu_i)$$
  paper 中写的是 $\bar{\sigma}_{\text{overall}}^2 = \mathbb{E}[\sigma_i^2]$，这其实是简化形式——严格的总方差分解应该是 within-group 方差均值加上 between-group 方差。完整公式：
  $$\text{Var}(X) = \mathbb{E}[\text{Var}(X|G)] + \text{Var}(\mathbb{E}[X|G])$$
  其中 $G$ 是数据集分组变量。$\mathbb{E}[\text{Var}(X|G)]$ 对应 $\mathbb{E}[\sigma_i^2]$，$\text{Var}(\mathbb{E}[X|G])$ 对应各数据集均值间的方差。paper 省略第二项可能是假设各数据集均值相近，或者把样本数加权写法隐含在 $\mathbb{E}$ 中。

- **Min/Max**：element-wise min/max

- **Percentiles**：无法从 summary statistics 精确合并，所以每个数据集保留一个序列化的 **t-digest sketch** [20]（https://arxiv.org/abs/1902.04023），训练时 merge sketch 恢复 pooled distribution 的近似分位数。t-digest 是一种 mergeable 的 streaming quantile 估计结构，精度高、内存小。

### 4.3 Absolute vs. Relative Actions
支持两种 action representation，作为独立 field 存储在 dataset 中。

- **Absolute**：world frame 下的 end-effector pose（XYZ + 6D rotation）
- **Relative**：相对于 anchor timestep 的 actual end-effector pose

形式化定义：设 $T_{\text{ref}} \in SE(3)$ 是 anchor timestep 的 actual end-effector pose，$T_t \in SE(3)$ 是未来 timestep $t$ 的 action pose，则 relative action 为：

$$T_t^{\text{rel}} = T_{\text{ref}}^{-1} \cdot T_t$$

这里 $T_{\text{ref}}^{-1}$ 是 $SE(3)$ 群上的逆，乘法是 $SE(3)$ 的群运算（齐次变换矩阵乘法）。这个公式本质上是把 $T_t$ 从 world frame 变换到 ref frame 下。物理含义：relative action 表示"从当前 pose 出发要到达的目标 pose"，对 policy 学习更友好，因为分布更稳定（不依赖 world frame 的绝对位置）。

**6D continuous rotation representation** [76]（https://arxiv.org/abs/1812.07035）：用 rotation matrix 的前两列（6 个数）表示 rotation，避免 Euler 角的 gimbal lock 问题和 quaternion 的 double-cover 不连续性。转换通过 **Gram-Schmidt orthogonalization** 完成——网络输出 6 维向量 $(a_1, a_2)$，Gram-Schmidt 得到正交的 $(\hat{r}_1, \hat{r}_2)$，第三列用叉积 $\hat{r}_3 = \hat{r}_1 \times \hat{r}_2$。这个表示在 neural network 连续性上证明比 quaternion 和 Euler 都好。

### 4.4 Past/Future Action Window
每个 training sample 围绕一个 anchor timestep $t$ 构造。低维 window 跨度 $[t - N_{\text{past}}, t + N_{\text{future}}]$，总长 $N_{\text{past}} + 1 + N_{\text{future}}$ 个 timestep。

- **Future portion**：监督信号（action chunking [75]，允许 temporal action chunking）
- **Past portion**：作为输入条件，让模型利用 recent action history

Episode 边界处用可配置 padding 策略（`copy` / `zero` / `reflect`）。若 padding 超过阈值（`max_padding_left`、`max_padding_right`）则丢弃该 sample。

Anchor 在 window 中的位置存为 `anchor_relative_idx`，让下游代码正确对齐 per-timestep 统计量、区分 past/future，无需重新解析 raw episode index。

**精妙的设计**：preprocessing 时的 window 不必和 training 时一致。可以预处理时存大 window，训练时用截断的子 window——给数据复用留了空间。

### 4.5 Proprioception
Proprioception 通过独立参数 `--proprioception_fields` 指定，与 `--action_fields` 分离。典型 fields：joint positions、joint velocities、actual end-effector pose（XYZ + 6D rotation）。

Batch 构造时各 field 被提取、normalize、沿 feature 维 concat，形成 shape $[B, T_{\text{prop}}, D_{\text{prop}}]$ 的 tensor。

**Causal constraint**：proprioception 只用 past + current timestep（indices $[0, t_{\text{anchor}}]$），而 action 跨整个 past+future window。这反映因果结构——past proprioception 是观测历史，future proprioception 在推理时不可用。

## 5. Flow Matching Action Head（Section 4.1 的核心）

VLA Foundry 的 action head 用 **flow matching** [37]（https://arxiv.org/abs/2210.02747）训练，不是 diffusion。这里我展开讲一下 flow matching 的直觉。

### 5.1 Flow Matching 基本原理
Flow matching 学习一个 vector field $v_t(x)$，使得 ODE $\frac{dx}{dt} = v_t(x)$ 能把简单先验（通常是 Gaussian $\mathcal{N}(0, I)$）变换到数据分布。

Conditional Flow Matching (CFM) 的训练目标：给定数据点 $x_1$ 和噪声 $x_0 \sim \mathcal{N}(0, I)$，定义条件路径
$$x_t = (1-t) x_0 + t x_1, \quad t \in [0, 1]$$
对应条件 vector field
$$u_t(x | x_1) = x_1 - x_0$$

训练损失（回归 vector field）：
$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, x_0, x_1} \| v_\theta(x_t, t) - (x_1 - x_0) \|^2$$

其中：
- $t \sim \mathcal{U}[0, 1]$ 是时间变量
- $x_0 \sim \mathcal{N}(0, I)$ 是噪声起点
- $x_1$ 是真实 action 数据
- $x_t = (1-t)x_0 + t x_1$ 是线性插值的中间状态
- $v_\theta(x_t, t)$ 是网络预测的 vector field
- $(x_1 - x_0)$ 是 ground truth 的 vector field（条件路径的导数）

推理时从 $x_0 \sim \mathcal{N}(0, I)$ 出发，用 Euler 或更高阶 ODE solver 积分 $\frac{dx}{dt} = v_\theta(x, t)$ 到 $t=1$ 得到 action。

### 5.2 Diffusion vs. Flow Matching
Diffusion 用的是 SDE（噪声 schedule + score function），flow matching 用的是 ODE（vector field）。两者数学上可以互转，但 flow matching 的训练目标更简单（直接回归 vector field，不需要 score 的对数导数），轨迹更直，采样步数更少。VLA Foundry 用 flow matching 是一个相对 modern 的选择。

### 5.3 Foundry-VLA-1.7B 的 action head 架构（Figure 4）
Action head 是一个 325M 参数的 transformer，结构和 LLM 一样（除了 vocabulary_size=0，没有 token embedding）。输入序列由三部分 concat：

1. **Conditioning tokens**：VLM 最后 $N=4$ 层 hidden features 中对应 observation token 的部分。这是关键设计——不是用最后一层的输出，而是用多层特征，让 action head 能访问不同抽象层级的 representation。
2. **Proprioception tokens**（可选）：proprioception 经过 linear layer 编码
3. **Noised action sequence tokens**：noised action 经过 linear layer 编码

顺序是 conditioning → proprioception → noised action。Flow transformer 输出 predicted denoising direction。

**Observation token** 是 VLA 训练时新加到 LLM vocabulary 的特殊 token。VLM 输入序列结构：`[image patches] + [task text tokens] + [observation token]`。这个 observation token 的作用是"汇总点"——所有视觉和文本信息在 LLM 中处理完后，最后 N 层的 hidden features 在 observation token 位置被抽取出来作为 action head 的 conditioning。

## 6. Pixel-Shuffle Pooling（Appendix C.4）

Image encoding 用 **pixel-shuffle** [41] [59] 做 patch pooling，减少传给 VLM 的 token 数。这里 paper 澄清了一个命名混乱：通常 "pixel-shuffle" [59]（https://arxiv.org/abs/1609.05158）是 super-resolution 中的上采样操作，VLA Foundry 实际用的是它的逆操作（"unshuffle"，下采样）。

直觉：把 $H \times W \times C$ 的 feature map 重排成 $\frac{H}{r} \times \frac{W}{r} \times (r^2 C)$，其中 $r$ 是 downsample factor。空间维度压缩 $r^2$ 倍，channel 维度膨胀 $r^2$ 倍。然后 linear projection 到 LLM embedding space。这样 ViT 输出的 patch tokens 被池化掉一部分，降低 VLA 序列长度（paper 中每个 image 64 tokens）。

## 7. FOUNDRY-VLA-1.7B 完整 from-scratch pipeline（Section 4.1）

### 7.1 LLM Stage
- **架构**：标准 transformer [23]，1.2B 参数，hidden_dim=2048，24 layers，16 heads
- **Embedding 参数**：约 200M（按惯例 [26] 不算入"有效"参数，所以叫 1.2B 而非 1.4B）
- **数据**：DCLM [33]（https://arxiv.org/abs/2406.11714），500M samples = 1T tokens
- **Sequence length**：2048 tokens
- **Tokenizer**：`HuggingFaceTB/SmolVLM2-256M-Video-Instruct`，vocab_size=49,280
- **LR schedule**：warmup-stable-decay [25]（https://arxiv.org/abs/2404.06395）

LLM benchmark 结果（Table 1）：

| Model | HS | MMLU | ARC-e | ARC-c | PIQA | WG | OBQA | BoolQ |
|-------|-----|------|-------|-------|------|-----|------|-------|
| 800B tokens | 64.3 | 26.0 | 70.3 | 37.0 | 75.8 | 60.9 | 40.0 | 63.2 |
| 1T tokens | 66.7 | 26.6 | 71.7 | 39.3 | 77.5 | 62.6 | 40.8 | 65.4 |

MMLU 接近 random（25%），因为模型小且没做 instruction tuning。但 HS、ARC-e、PIQA 等较简单的 benchmark 明显高于 random，说明基础语言能力在学习。

### 7.2 VLM Stage
- **Vision encoder**：86M 参数 ViT [19]，架构类似 CLIP [54]，随机初始化（不是用预训练 SigLIP/DINO）
- **Input**：224×224 image
- **Pooling**：pixel-shuffle
- **Init**：从 LLM 800B tokens checkpoint 加载（在 LR cooldown 之前，遵循 [29] 的建议）
- **数据**：DataCompDR-1B [22]
- **训练量**：200M samples

VLM captioning benchmark（Table 2，COCO_VAL）：

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE_L | CIDEr |
|-------|--------|--------|--------|--------|---------|-------|
| 165M | 57.25 | 37.12 | 23.23 | 14.44 | 37.13 | 50.17 |
| 200M | 58.64 | 38.62 | 24.49 | 15.57 | 38.17 | 55.14 |

paper 明确说这是 "end-to-end training functionality 的证据" 而非 "optimal performance 的主张"。换用 SigLIP/DINO 预训练 vision encoder 或 PaliGemma2/Qwen3-VL 这类预训练 VLM backbone 应该会有更好结果。

### 7.3 VLA Stage
在 VLM 基础上加 observation token + flow transformer action head（325M 参数）。

**模型参数分解**（Table 3）：

| Model | Embedding | LLM | Vision | Action head | Total | Non-embed |
|-------|-----------|-----|--------|-------------|-------|-----------|
| FOUNDRY-VLA-1.7B | 0.20 | 1.23 | 0.09 | 0.33 | 1.85 | 1.65 |
| FOUNDRY-QWEN3VLA-2.1B-MT | 0.62 | 1.41 | 0.41 | 0.31 | 2.75 | 2.13 |

**输入序列**：8 个 image（2 个 timestep × 4 个 camera）→ 512 tokens + task description text。VLA 序列平均 549 tokens，动态 padding。

**数据**：
- 42 个 simulation tasks，361 个 real-world tasks
- 39 个 tasks 在 real 和 sim 中都有对应
- 不用 OXE [16] 或 UMI [13] 数据（与 LBM [65] 不同）
- 总计 54,616 episodes，18.8M training samples（Table 7）

## 8. FOUNDRY-QWEN3VLA-2.1B-MT（Section 4.2）

为了证明 framework 的可替换性，用预训练的 **Qwen3-VL 2B** [4] 作为 VLM backbone，保持 action head 架构不变，用同样的数据 mixture 训练。Qwen3-VL 是 2025 年底发布的强 VLM，作为 backbone 自然比 from-scratch 的小 VLM 强很多。

结果（Figure 5）：在 lbm_eval_cs 上比 LBM-MT [65] 高出 **20+ 个百分点**，统计显著。同时证明传统 VLM→VLA recipe 在 VLA Foundry 内能高效复现——同一个 training loop、dataloader、preprocessing pipeline。

## 9. 评估方法论（Section 3.3）—— 这部分统计学处理很讲究

### 9.1 LBM Eval
评估在 **lbm_eval_oss** [66]（https://github.com/ToyotaResearchInstitute/lbm_eval）上做，是 LBM [65] 仿真 benchmark 的开源版。用 **Drake** [68]（https://drake.mit.edu）物理引擎，49 个 tabletop bimanual manipulation tasks，作为 Docker image 发布，避免平台依赖问题。

### 9.2 统计分析（STEP [63]）
这点是 paper 的隐藏亮点。用 **STEP** [63]（https://arxiv.org/abs/2506.14903，RSS 2025）做 rigorous 的 policy 比较。

核心工具：
- **Bayesian estimates of success rates**：用 Beta posterior（Beta(α, β)），rollout 结果（success/failure）更新 posterior，violin plot 显示分布
- **Compact Letter Display (CLD)** [53]：把多组比较结果压缩成字母标注，**不共享任何字母**的两组在 5% family-wise error rate (FWER) 下显著不同
- **Near-optimal stopping**：允许用户在中间结果基础上决策——可以提前停止 evaluation 省时间，或继续收集更多 rollout 提高统计 power

**关键反 p-hacking 设计**：标准统计检验（如 Barnard's test [6]）在 "peek + decide" 模式下会 p-hacking [64]。STEP 的 Bayesian + anytime-valid framework 允许这种 peeking 而不破坏 validity。

### 9.3 Unbiased Aggregation
跨 task 聚合时，对每个 policy 在每个 task 上**平衡样本量**。例如 Model A 在 4 个 task 上有 [50, 49, 50, 50] rollouts，第二个 task 少一个，则全部截断到 [49, 49, 49, 49]，用 196 而非 199 计算 aggregate。这保证 aggregate 是 equally-weighted multi-task performance 的无偏估计。paper 明确指出前作 [65] 没严格执行这点——所以新 dashboard 里 LBM 的数字可能和原 paper 略有差异。

## 10. 实验结果（Section 4.3）

### 10.1 与 LBM 比较（Figure 5）
四个模型在 lbm_eval_cs 上的 seen tasks：
- **Foundry-Qwen3VLA-2.1B-MT**：最强，统计上显著优于 LBM-MT 20+ 个百分点
- **LBM-MT** 与 **Foundry-VLA-1.7B-MT-sim**：统计上 on par
- **Foundry-VLA-1.7B**（full data）：最弱

注意只有 Foundry-VLA-1.7B-full 和 Foundry-Qwen3VLA-2.1B-MT 用完全相同的 robot training data，所以 LBM-MT 和 Foundry-VLA-1.7B-MT-sim 的比较要小心解读数据差异。

### 10.2 Training Stage Comparisons（Figure 7）
对两个模型系列分别比较 ST（single-task）、MT（multi-task）、FT（multi-task finetuned）：

**Foundry-Qwen3VLA-2.1B-MT 系列**：
- ST → MT → FT 性能单调提升
- MT 训练后比 ST 好
- FT 进一步提升

**Foundry-VLA-1.7B 系列**：
- 结果 mixed
- 某些 task（如 Apple:Bowl → Bin）FT 比 ST 好
- 某些 task（如 Stack Plates:Rack → Table）相反
- aggregate 上 MT 和 FT 统计上比 ST 差

**核心 hypothesis**：更强的 backbone（Qwen3-VL）→ 更好的 policy outcomes。from-scratch 的小 VLM backbone 不足以支撑多任务的 representation 需求。

### 10.3 Unseen Tasks（Figure 8）
3 个 held-out tasks：
- 两个 multi-task 模型都展现少量 zero-shot generalization
- Foundry-Qwen3VLA-2.1B-MT 的 FT 比 ST 好（aggregate）
- Foundry-VLA-1.7B 没有这个优势

证明 strong backbone 不仅在 seen tasks 上好，在 unseen tasks 上 zero-shot 和 FT 能力也更强。

### 10.4 Data Subset Ablation（Figure 9）
Foundry-VLA-1.7B 在三个数据子集上训练（相同 compute）：
- **Sim only**：sim 上最好
- **Real only**：sim 上接近 0%（分布外）
- **Sim + Real**：略差于 sim only

paper 给出两个 hypothesis：model undertraining，或 model 的 representational power 在 real 和 sim 任务间被 split。这个现象值得深入研究——是否多 embodiment/domain 数据在小 model 上反而有害？这与一些 recent scaling 研究 [35] 可能有不一致。

## 11. Training Hyperparameters（Table 4）

| Model | LR | Schedule | Warmup | Total samples | Batch |
|-------|-----|----------|--------|---------------|-------|
| Foundry-VLA-1.7B-full | 5e-5 | cosine | 1000 | 102.4M | 1024 |
| Foundry-VLA-1.7B-ST | 5e-5 | cosine | 1000 | 5.12M | 512 |
| Foundry-VLA-1.7B-FT | 5e-6 | cosine | 1000 | 1.024M | 512 |
| Foundry-Qwen3VLA-2.1B-MT | 5e-5 | cosine | 1000 | 100M | 1024 |
| Foundry-Qwen3VLA-2.1B-ST | 5e-5 | cosine | 1000 | 2M | 512 |
| Foundry-Qwen3VLA-2.1B-FT | 5e-6 | cosine | 1000 | 1.024M | 512 |

观察：
- MT 训练约 100M samples，batch 1024
- ST 训练 2-5M samples（per task），batch 512
- FT 从 MT checkpoint 出发，1M samples，LR 降 10×（5e-6 vs 5e-5）
- 都用 AdamW + cosine schedule

## 12. Throughput 性能（Figure 2）
在 AWS SageMaker P5 节点（8× NVIDIA H100 per node）上测了 LLM/VLM/VLA 的 throughput scaling：

- **LLM**：2048 tokens，padding
- **VLM**：64 tokens/image + 256 tokens text
- **VLA**：8 images = 512 tokens + task text，平均 549 tokens

关键观察：在 1.2B 这个 scale 下，单 GPU 能装下完整模型权重，FSDP 没优势，甚至在 VLM 上 scaling 比 DDP 弱。这说明 FSDP2 的 sharding overhead 在小 model 上不划算——只在 model > single GPU memory 时才该用。

## 13. 与 Related Work 的定位（Section 2）

### 13.1 LLM/VLM Frameworks
- **Megatron-LM** [60]、**DeepSpeed** [56]、**GPT-NeoX** [1]：大规模分布式训练
- **OpenLM** [23]（VLA Foundry 部分代码来源）、**OLMo** [67]、**LLM360** [40]、**K2** [39]：full-stack transparency
- **FastLLM** [58]、**nanoGPT** [28]（你的 repo！）、**LLMs from scratch** [55]：降低 reproduction 门槛
- **DCLM** [33]、**FineWeb** [48]：高质量 language dataset
- **OpenFlamingo** [2]、**LLaVA** [38]、**BLIP-2** [34]、**Prismatic** [27]、**InternVL** [12]、**Qwen** [3,4]：VLM 框架
- **DataComp** [22]：image-text dataset pipeline

### 13.2 VLA Frameworks
- **OpenVLA** [30]：7B 模型，基于 Prismatic
- **OpenPi** [50]：Physical Intelligence 的 π 模型开源，10k+ hours robot data
- **GR00T** [46]：NVIDIA，VLM backbone + diffusion transformer action head，dual-system architecture
- **MolmoAct** [32]：3D space reasoning with depth-aware perception tokens
- **LeRobot** [10]：community-first，SO-100/101 arms，SmolVLA [61] 450M
- **VLAb** [18]：HuggingFace 的 VLA pretraining 库
- **VLA-Scratch** [21,69]：FSDP2-based，多 VLM backbone 支持
- **StarVLA** [17]：Lego-like，decoupled backbone + action head
- **Dexbotic** [71]：unified PyTorch toolbox，跨 platform

VLA Foundry 的差异化：**唯一一个真正贯穿 LLM→VLM→VLA 全链路的单一 codebase**，其他都在 action training 阶段做文章。

## 14. Limitations（Section 5）

paper 明确说这是 deliberate scope：
1. 只评估了 LBM simulation，narrow embodiment，没 real hardware 数字（但 framework 支持扩展到 LIBERO、SimplerEnv、RoboCasa）
2. 只用 flow-matching action head（diffusion policy 已在 codebase 中实现但没评测，autoregressive discrete action tokenization 也能加）
3. 没刻画 optimal data recipe across stages
4. 没解决 safety、alignment、failure-mode detection

## 15. 我的整体 takeaways

这篇 paper 的真正价值不在 released models 的绝对性能（Qwen3-VLA 2.1B 比 LBM 高 20+ pp 当然不错，但 LBM 本身是 566M 的较小 model），而在**基础设施的投资**：

1. **End-to-end controllability** 是稀缺品——绝大多数 VLA 团队用 HF Hub 上的 VLM 黑盒，无法干预 pretraining。VLA Foundry 让 "如果 VLM pretraining 数据里加 robotics-related text 会怎样" 这种问题变得可实验。

2. **Statistics rigor** 是被严重低估的维度。STEP 的 near-optimal stopping + CLD + Bayesian violin plot + unbiased aggregation 这套组合拳，在 robotics policy 评估中罕见地严谨。很多 paper 报告的 success rate 差异在统计上根本不显著，但读者看不出来。

3. **Flow matching + multi-layer hidden features as conditioning** 是一个 elegant 的 action head 设计。多层 features 比 single layer 提供更丰富的 hierarchy 信息，类似 ControlNet 用 multi-scale features 的思路。

4. **Relative action via SE(3) + 6D rotation** 是 robotics 的正确做法，paper 处理得很规范。t-digest 跨数据集合并 percentiles 这个细节体现了工程素养。

5. **小 model + full pipeline vs. 大 VLM backbone** 的对比（Foundry-VLA-1.7B vs. Foundry-Qwen3VLA-2.1B-MT）清晰地展示了：**当前阶段，stronger VLM backbone 比端到端 from-scratch 更划算**。但 paper 也强调 from-scratch pipeline 的价值在于可控性和可研究性，不是当下的 SOTA 追求。

参考链接汇总：
- 项目主页：https://tri-ml.github.io/vla_foundry
- 代码：https://github.com/TRI-ML/vla_foundry
- 模型：https://huggingface.co/collections/TRI-ML/vla-foundry
- Draccus：https://github.com/marin-community/draccus
- WebDataset：https://github.com/webdataset/webdataset
- Flow Matching：https://arxiv.org/abs/2210.02747
- 6D Rotation：https://arxiv.org/abs/1812.07035
- t-digest：https://arxiv.org/abs/1902.04023
- ACT / Action Chunking：https://arxiv.org/abs/2304.13705
- LBM Eval：https://github.com/ToyotaResearchInstitute/lbm_eval
- Drake：https://drake.mit.edu
- STEP：https://arxiv.org/abs/2506.14903
- LBM 前作：https://arxiv.org/abs/2507.05331
- DCLM：https://arxiv.org/abs/2406.11714
- DataComp：https://arxiv.org/abs/2304.14108
- Warmup-Stable-Decay：https://arxiv.org/abs/2404.06395
- Qwen3-VL：https://arxiv.org/abs/2511.21631
- OpenVLA：https://arxiv.org/abs/2406.09246
- OpenPi：https://github.com/Physical-Intelligence/openpi
- GR00T N1：https://arxiv.org/abs/2503.14734
- LeRobot：https://github.com/huggingface/lerobot
- SmolVLA：https://arxiv.org/abs/2506.01844
- nanoGPT（你的）：https://github.com/karpathy/nanoGPT
- nanoVLM：https://github.com/huggingface/nanoVLM
- OpenLM：https://github.com/mlfoundations/open_lm
- Pixel-Shuffle (super-res)：https://arxiv.org/abs/1609.05158
- CLD：https://www.tandfonline.com/doi/abs/10.1198/1061860043515
- Barnard's test：https://doi.org/10.1093/biomet/34.1-2.123

如果你想 build intuition 关于"为什么统一 LLM/VLM/VLA pipeline 重要"，我建议关注三个 thread：
1. **Data scaling laws across modalities** 还不清晰——LLM 数据配比对 VLA 性能的影响是开放问题，VLA Foundry 的 multi-modal mixing 是研究这个的工具
2. **Backbone transfer** 的 mechanism——Qwen3-VL 到 VLA 的 20+ pp 提升里，多少来自 language understanding、多少来自 visual grounding、多少来自 general representation？需要 ablation
3. **Flow matching vs. diffusion vs. autoregressive action heads** 的 systematic 比较——paper 没做，但 framework 支持做

希望这个讲解对你 build intuition 有帮助。如果对某个具体模块（比如 FSDP2 实现细节、STEP 的 anytime-valid 统计原理、t-digest 的 merge 算法）想深入，可以继续问。
