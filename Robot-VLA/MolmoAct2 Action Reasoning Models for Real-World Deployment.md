---
source_pdf: MolmoAct2 Action Reasoning Models for Real-World Deployment.pdf
paper_sha256: 1734fb13d01e8145371f73d0b3b4ec4a55a1ff854720b800b27d7067f40796ce
processed_at: '2026-08-05T20:02:26-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 MolmoAct2

## 这论文到底在干啥

想象你想造一个**通用机器人保姆**——一个 model 就能做饭、洗碗、叠衣服、收拾房间，啥都会。现在学界都在往这个方向冲，叫 VLA (Vision-Language-Action) model，就是"看见啥+听懂人话→做出动作"。

但现在的问题是什么呢？

1. **顶尖的 model 都是黑盒**：Physical Intelligence 的 π0.5、Google 的 Gemini Robotics 这些 SOTA，数据和代码都不公开，你没法复现，也没法改
2. **会"思考"的 model 太慢**：让机器人先想想"这个杯子在哪儿、怎么抓"再动手，听起来不错，但这一想就是几百个 token，闭环控制根本来不及
3. **开源 model 都绑定贵机器人**：Franka 机械臂一台好几万刀，普通 lab 用不起
4. **实际成功率还是不够**：就算 fine-tune 了，成功率还是不够看，没法真部署

MolmoAct2 就是要同时把这四个坑都填了。

---

## 他们做了啥

### 1. 给"大脑"做专项训练

通用 VLM (Vision-Language Model) 像 GPT-5、Gemini 这些，虽然啥都见过，但它们对"空间"的理解是模糊的——能告诉你"这是厨房"，但说不准"杯子离桌边 30cm"。机器人需要的是后者。

所以他们搞了 **Molmo2-ER** (ER = Embodied Reasoning)：
- 不从头训，在已有的 Molmo2 上继续训
- 专门喂了 3.3M 个"空间推理"相关的 sample：哪有 free space、跨视角对应、depth 估计、点指哪
- 两阶段训练：先专攻空间能力，再混着复习通用能力防止"忘掉"
- 结果在 13 个 benchmark 上 9 个 SOTA，超过 GPT-5 和 Gemini Robotics ER

**人话**：就像给一个啥都懂一点的本科生，专门开一门"空间感知"的强化课，让他变成空间感知专家，但又不让他忘掉原来的通用知识。

---

### 2. 疯狂搞数据

这论文最实在的地方是数据。他们搞了三个数据集：

**MolmoAct2-BimanualYAM Dataset**：目前最大的开源双臂数据集，720 小时，34.5k 演示，28+ 种任务（叠衣服、收桌子、扫商品、装药...），整套硬件不到 6000 美金，2 个月采集完。

**MolmoAct2-SO100/101 Dataset**：Hugging Face 社区的便宜机器人，从 1,222 个公开 dataset 里筛。怎么筛呢？用 TOPReward 这个自动质量评估方法，过滤掉垃圾数据。

**MolmoAct2-DROID Dataset**：把已有的 DROID 数据集（Franka 机械臂的）用语言标注增强和空闲帧过滤，筛出干净的 74k episode。

还有个细节很妙：他们发现很多数据集的 language instruction 要么重复（"pick up the cup" 出现一万次），要么根本是乱写的（"lerobot_test"）。于是用 Qwen3.5-27B 重新注释了一遍，unique label 从 22% 翻到 46%。

**人话**：数据是王道。垃圾数据训不出好 model。他们花大量功夫做数据清洗和重注释，这比堆数量重要。

---

### 3. 怎么把连续动作塞进语言模型

机器人动作是连续的数字（关节角度、end-effector 位置），但语言模型只会预测离散 token。怎么办？

**OpenFAST Tokenizer**：把 1 秒的动作轨迹做傅里叶变换到频域，量化后用 BPE 压成离散 token。32 维 × 30Hz 的动作（960 个数字）能压缩到几十个 token。

这样整个训练就是标准的"预测下一个 token"，可以和 text、image 无缝混训。

---

### 4. 核心架构创新：让动作专家"逐层"读取大脑

这是论文最 tech 的创新。

传统做法：VLM 处理完图像和文字，把**最后一层**的 hidden state 喂给 action expert。问题是这样信息被压成单一表示。

MolmoAct2 的做法：让 action expert 的**每一层**都去读 VLM **对应层**的 KV cache。浅层读浅层（low-level 视觉特征），深层读深层（high-level 语义）。

这个叫 **per-layer KV conditioning**。

技术上：
- VLM 第 $\ell$ 层产生 key, value cache：$K_\ell^{\mathrm{vlm}}, V_\ell^{\mathrm{vlm}}$
- 用学习到的投影 $P_K, P_V$ 把它对齐到 action expert 的维度
- Action expert 第 $\ell$ 层的 cross-attention 用这个投影后的 cache

Ablation 显示：per-layer KV (95.9%) > per-head KV (94.8%) > hidden state (94.0%)。

**人话**：以前是只让动作专家看大脑最后的总结，现在是让它能看到大脑每一层的中间想法，所以它动作规划更精准。有点像 U-Net 的 skip connection，让浅层细节和高层语义都能传到动作生成那一步。

---

### 5. Flow matching 生成连续动作

Pre-training 出来的是离散 action token，但部署需要连续控制。所以 post-training 加了个 **DiT-style flow matching action expert**。

Flow matching 是啥？简单说：
1. 从纯 Gaussian noise 开始
2. 模型学一个 velocity field，告诉你怎么从 noise 走到真实 action
3. 推理时积分这个 velocity field，noise 就慢慢变成 action

公式核心：
$$x_t = (1-t)\epsilon + t \cdot a, \quad u^* = a - \epsilon$$

模型预测 $u^*$（target velocity），loss 是 MSE。

训练时每个 action chunk 采 K=4 个不同的 (noise, time) 对，让 expert 在 flow trajectory 上多点监督。Fine-tuning 时 K=8。

还有个 **knowledge insulation**：action expert 条件于 VLM KV cache，但梯度不回传到 VLM。这样 VLM 保持通用性，不被 flow loss "污染"。

---

### 6. MolmoAct2-Think：让机器人会"想"但不慢

前作 MolmoAct 让模型先预测 depth（场景深度），再基于 depth 生成动作。问题是每步都要预测 100 个 depth token，太慢。

MolmoAct2-Think 的 insight：**机器人轨迹中大部分场景是静态的**，机械臂没动到的区域 depth 没变，重新预测是浪费。

做法：
- 把 depth map 分成 10×10 grid（100 个 cell）
- 每帧比较当前 RGB patch 和上一帧的 cosine similarity
- 相似度 > 0.996 的 cell 复用上一帧的 depth token
- 只有变化的 cell 重新预测

公式：
$$m_{t,i} = \mathbf{1}[\cos(x_{t,i}, x_{t-1,i}) < 0.996]$$

如果 cosine < 0.996，标记为"变化"，重新预测；否则复用 buffer。

还有个 learned gate 控制每层 expert 用多少 depth 信息，初始化为接近 0，慢慢学。

**人话**：机器人动作其实大部分时候场景没啥变化，只在抓取瞬间物体位置变了。MolmoAct2-Think 让机器人只在"有变化"的地方重新"想"depth，静态部分直接用缓存。这样既保留了空间推理的好处，又不慢。

---

## 效果怎么样

### Out-of-the-box 部署（不 fine-tune 直接用）

- **MolmoSpaces** (simulation): 平均 20.6% vs π0.5 的 10.0%（+10.6%）
- **MolmoBot** (real-world DROID): 87.1% vs π0.5 的 45.2%
- **SO-100/101**: 56.7% vs π0 的 45.3% vs SmolVLA 的 2.3%

### Fine-tuning 后

- **LIBERO**: MolmoAct2 97.2%, MolmoAct2-Think 98.1%, π0.5 96.9%
- **Real-world YAM 8 tasks**: 平均 50.1%, 比 runner-up 高 15%
- **RoboEval**: 不光成功率高，trajectory 也更 smooth、更短、jerk 更小、self-collision 更少

### 推理速度

- **MolmoAct2**: CUDA Graph 优化后 55.79 Hz（远超 30Hz 控制频率）
- **MolmoAct2-Think**: 12.71 Hz（低频任务够用）

---

## 为什么这论文重要

1. **全开源**：权重 + 代码 + 完整训练数据，包括最大开源 bimanual dataset。VLA 研究不再被闭源 frontier model 垄断。

2. **不只是刷数字，是系统解决真实部署**：架构（per-layer KV）、数据（quality filtering）、推理优化（adaptive depth + CUDA Graph）协同。

3. **验证了"中间表示有用"**：per-layer KV 比 final hidden state 强，说明 VLM 的中间层信息不该被 bottleneck 掉。

4. **adaptive reasoning 是 latency 和 grounding 的折中**：让 reasoning cost 和 scene change 成正比，而不是固定开销。这个 idea 可以推广到其他 reasoning modality。

---

## 可能的不足

1. **per-layer KV 的 memory 开销**没仔细讨论，36 层 expert × 36 层 VLM 的 KV cache + projection 不小。

2. **adaptive depth 在高速场景退化**：如果机械臂快速运动，大部分 patch 都 change，加速失效。

3. **为什么是 depth 不是 surface normal 或 optical flow**？作者没 ablation 其他 reasoning modality。

4. **K=8 是 GPU memory 限制不是 accuracy 极限**，如果用 gradient checkpointing 能不能更大？收益是否饱和？

---

## 一句话总结

MolmoAct2 是一个全开源、能真部署、会"想"但不慢的机器人 foundation model，靠"per-layer KV conditioning 让动作专家看全大脑"、"OpenFAST tokenizer 把连续动作变成语言 token"、"adaptive depth 只在想变化的地方重新推理"三个核心创新，加上数据清洗和推理优化，把开源 VLA 推到了能跟闭源 frontier model 掰手腕的水平。

参考：
- 项目主页：https://allenai.org/blog/molmoact2
- 代码：https://github.com/allenai/molmoact2
- Paper PDF：https://arxiv.org/abs/2508.07917

---

# MolmoAct2: 一个面向真实世界部署的开放 Action Reasoning Model

作为长期关注 foundation model 在物理世界落地的研究者, 我读完这篇 paper 的第一反应是, 这是一份非常"工程实在"的论文。它不光是堆 benchmark 数字, 而是系统地针对真实部署的痛点做架构、数据和推理优化。让我从直觉开始, 逐层拆解。

---

## 1. 这篇论文到底要解决什么问题?

作者把当前 VLA (Vision-Language-Action) 模型的失败模式精炼为四个 axis:

1. **Frontier VLAs 几乎是 closed system**: π0.5, Gemini Robotics, GR-ER 1.5 Thinking 这些 SOTA 模型的 data, recipe,weights 都不公开。即便有的 release 了 weights, 也扣住训练 data 和 procedure。
2. **Reasoning-augmented policy 在推理 latency 上付出 prohibitive cost**: chain-of-thought trace, goal image prediction, point trajectory, world-model rollout 这些 reasoning 机制能在生成 action 之前要消耗 hundreds of tokens 或整帧 prediction, 在闭环控制下太慢。
3. **Open-weight VLA 绑定 expensive hardware**: 大部分能 out-of-the-box 跑的 VLA 都需要 Franka 这种高端平台, 普通学术 lab 和独立研究者用不起。
4. **Zero-shot 表现脆弱, fine-tuning 后的成功率仍然低于 dependable deployment threshold**。

MolmoAct2 同时对这四个 axis 做了响应, 沿着五个改进方向推进前作 MolmoAct: 更强的 embodied-reasoning VLM backbone (Molmo2-ER), 三个新数据集, 开源的 OpenFAST Tokenizer, 重新设计的 VLA 架构 (per-layer KV grafting), 以及新的 reasoning paradigm (MolmoAct2-Think 的 adaptive depth)。

参考链接:
- 项目主页: https://allenai.org/blog/molmoact2
- 代码: https://github.com/allenai/molmoact2

---

## 2. Molmo2-ER: 把通用 VLM 变成 embodied-reasoning specialist

### 2.1 直觉

通用 VLM (Molmo2, Qwen3-VL, GPT-5) 在 web-scale image-text 上 pretrain 出来的"空间感"其实是模糊的——它知道"这是个厨房", 但不一定能精确说出杯子离桌子边缘 30cm, 或者第二视角下的物体在第一视角的哪里。而 policy 需要的是 metric distance, free space affordance, cross-view correspondence, scene geometry 这些更"硬"的能力。

Molmo2-ER 的做法是: 不重训 Molmo2, 而是用一个**specialize-then-rehearse**的两阶段 recipe 在已有的 Molmo2 checkpoint 上继续训练。

### 2.2 训练数据: 六大 capability pillars

数据 corpus 大约 3.3M samples, 覆盖六个互补的能力维度:

| Pillar | 代表数据集 | 作用 |
|---|---|---|
| Image embodied QA | SAT, RoboPoint-QA, RefSpatial, VST-P, VSI-590K | 动态推理, 指代, 跨视图度量 |
| Video embodied QA | SIMS-VSI (203K), RoboVQA (200K) | 时间维度推理, planning, affordance |
| Pointing & detection | RoboPoint (700K), LVIS detection (100K) | pixel-accurate localization, 因为 pointing 是主要的 action interface |
| Multi-image / ego-exo | SenseNova-SI (500K), VST-P cross-view (200K) | 多相机, 第一/第三人称切换 |
| Abstract reasoning | CLEVR (50K), GRiD-3D (100K) | 组合推理, frame-of-reference |

整套 mixture (Molmo2 + Molmo2-ER) 约 12.51M samples, 非机器人部分中 Molmo2-ER 占 0.46 的 sampling weight。

### 2.3 Specialize-then-rehearse recipe

**Stage 1: Embodied specialization** (20K steps)
- 起点: Molmo2 (Qwen3-4B) mid-training checkpoint
- 数据: Molmo2-ER corpus + 8% Tulu-3 text-only (保语言能力)
- Sequence length 4,200, global batch 64, device batch 4 × 16 GPUs (2 nodes × 8 H100)
- 这一步迅速把 model 推到 embodied data manifold 上

**Stage 2: Joint refinement** (1.5K steps)
- 继续训 Stage 1 checkpoint, 把 embodied corpus 和 Molmo2 原始 multimodal mid-training data 混合
- 保持 NLP 8%, 剩余 92% 中 p 比例给 embodied, (1-p) 给 general
- 调参 sweep $p \in \{0.30, 0.50, 0.70, 0.90\}$, 发现 **p=0.5** 给出 best Pareto trade-off
- Sequence length 增到 16,384 (因为 multi-image / long-video example 长), per-device batch size 降到 1

### 2.4 结果: 13 个 benchmark 上的 embodied reasoning

Table 3 的结果非常 impressive:

- **Molmo2-ER 平均 63.8%**, 比 base Molmo2 (46.8%) 提升 +17 points
- 在 9 of 13 benchmarks 上 SOTA (在 open-weight 模型中)
- 超越 GR-ER 1.5 Thinking (61.3%), GPT-5 (57.9%), Gemini 2.5 Pro (57.1%)

特别值得注意的是 **RefSpatial** 上 Molmo2-ER 拿到 52.5, 而 GPT-5 只有 23.5——这是 chain-of-thought referring 的 benchmark, 说明专门训练对 spatial language → pointing action 的 bridge 非常有效。

**Intuition**: 这其实是一个 "数据 distribution 比 model size 更重要" 的实证。Molmo2-ER 用 4B 参数 + 针对性 3.3M 数据, 打败了远大于它的闭源模型。机器人需要的不是更多 web 知识, 而是更多 embodied 形态的 supervision。

---

## 3. 数据: 三个新数据集

### 3.1 MolmoAct2-BimanualYAM Dataset

这是**目前最大的开源 bimanual manipulation dataset**:
- 720 hours, 34.5k demonstrations
- 28+ unique real-world tasks (折叠衣服, 整理电缆, 收桌子, 扫商品, 分装药品)
- 双臂 YAM (Yet Another Manipulator) 平台, 整套 setup < $6,000 USD (Figure 3)
- 2 个月采集, Cortex AI 支持, 严格的 retry 和 no-op 协议保证质量

**Intuition**: 这个 dataset 之所以重要, 是因为双臂任务对空间推理的要求远高于单臂, 而开源的双臂数据一直非常稀缺。机器人 community 长期被"单臂 Franka pick-and-place"统治, 缺少 bimanual in-the-wild 数据阻碍了 generalist policy 的发展。

### 3.2 MolmoAct2-SO100/101 Dataset

SO-100/101 是 Hugging Face 的低成本机器人平台, 社区驱动。作者从 1,222 个公开 LeRobot dataset (377 个 user) 中 curate:
- 38,059 episodes, 19.8M frames, 184 hours
- 四阶段过滤: 结构合法性 → 移除 eval-style dataset → license/codebase 检查 → **TOPReward quality gate**

TOPReward (Chen et al., 2026) 是一个用 token probability 作为 hidden zero-shot reward 的方法。作者对 human-audited high-quality dataset 计算 TOPReward 平均值作为 threshold, 保留 mean TOPReward (last 3 sampled episodes) 高于此 threshold 的 dataset。

**Intuition**: community-sourced data 的 diversity 是双刃剑——broad coverage 但 quality 参差不齐。这里用的 TOPReward gate 是一个自动化的质量过滤, 避免人工审查数千个 dataset。

### 3.3 MolmoAct2-DROID Dataset

从原 DROID 用 extended language annotations (95% episode 有 3 条 instruction) 和 idle-frame filter 筛出:
- 74,604 valid episodes, 17.75M frames
- 每个 episode 都标记 successful, 至少 1 条有效 instruction, 无显著停顿
- 整个 filtered DROID 重新做了 language re-annotation

### 3.4 Language annotation pipeline

作者发现两个问题:
1. **重复 instruction**: 比如 BC-Z 104 条 unique instruction / 39,350 episodes = 0.26% unique rate
2. **不准确 / 无意义 instruction**: LeRobot 上常见 "lerobot_test", "Test run"

解决方法: 用 Qwen3.5-27B (open VLM) 对 dataset 重新注释。Prompt 给 VLM 一些 frame + 原 instruction, 让它生成新 instruction。为增加 diversity, 随机采样一个 word count 让 VLM 大致按那个长度生成。

效果: unique label 从 71,121 (22%) → 146,485 (46%), **翻倍**。

**Intuition**: 这个细节很重要, 常被忽视。Imitation learning 时, language 条件信号越 diverse, policy 的 conditioning 越鲁棒。重复 instruction 等于把 language 退化成 task ID, 失去 compositionality。

---

## 4. MolmoAct2 架构与训练: 三阶段 pipeline

这是论文最技术性的部分, 我会详细拆解。

### 4.1 Pre-training: discrete autoregressive policy

#### 4.1.1 OpenFAST Tokenizer

连续 action 不能直接塞进 language model pretraining stream。OpenFAST Tokenizer 是 FAST (Pertsch et al., 2025) 的开源 re-implementation, 同时 release 权重和训练数据。

**核心 pipeline**:
1. 把 1 秒的连续 action trajectory (32 维, 不同 embodiment 都 padding 到 32D) 做频域 transform (DCT)
2. 量化 frequency coefficient
3. 用 BPE (Byte-Pair Encoding) 生成 2048-token vocabulary 的离散 token sequence

**为什么要频域**? 因为 action trajectory 在时域上冗余度高 (相邻时刻 action 接近), 但在频域上能量集中低频。频域 transform + BPE 能极大压缩——1 秒 32-D action (比如 30 Hz × 32 = 960 个 number) 能被压缩到 ~几十个 token。

**训练 mixture** (Table 2):
- 30% MolmoAct2-BimanualYAM (YAM, absolute joint)
- 30% MolmoAct2-SO100/101 (SO-100/101, absolute joint)
- 30% MolmoAct2-DROID (Franka, absolute joint)
- 3.33% Fractal (Google Robot, delta end-effector)
- 3.33% BC-Z (Google Robot, delta end-effector)
- 3.33% Bridge (WidowX, delta end-effector)

总共 1M subsampled action sequences, 覆盖 5 个 embodiment 和两种 control mode (absolute joint 和 delta end-effector)。

**Normalization 细节**:
- 连续维度用 1-99 percentile 统计归一化 (限制 outlier 影响但保留 dynamic range)
- Gripper command 单独处理 (binary 或 narrow-range open/close)
- 每个 action padding 到 32 维, 让不同 action dimensionality 共享同一 tokenizer input space

#### 4.1.2 Pre-training recipe

**Model**:
- 初始化自 Molmo2-ER checkpoint
- 视觉编码器: SigLIP2 ViT
- Connector: 用 ViT 第三到最后和倒数第九层的 feature; image 用 2×2 pooling, video 用 3×3 pooling (减少 token); pooled feature 通过 MLP 投影到 LLM embedding space

**Data mixture**:
- 10% multimodal data (保留 VLM 能力)
- 90% robot trajectory
  - YAM, SO-100/101, DROID 各 30%
  - 剩 10% 分给 BC-Z, BridgeData V2, RT-1, MolmoAct Dataset

**关键 trick**:
- Single resized crop (不用 high-resolution tiled crop)
- Video 最多 sample 8 帧, 2 FPS
- Image augmentation: 几何扰动 + color jitter + occasional blur
- Multi-camera episode 随机化 camera 顺序 (episode level)
- 用 special token 标记 setup 和 control: `<setup_start>bimanual yam robotic arms in molmoact2<setup_end>`, `<control_start>absolute joint pose<control_end>`
- 末尾加 `<action_output>` 作为 explicit signal

**State tokenization**: normalize 后, 每个状态值 uniform discretize 到 256 个 state token, 加到 prompt 中 action target 之前

**Training**:
- 200K steps, max sequence length 4,200 tokens
- On-the-fly packing: 多个 short example pack 进一个 4,200-token sequence, 用 attention mask 隔开
- Vision encoder + connector LR: $5 \times 10^{-6}$
- LLM LR: $1 \times 10^{-5}$
- Global batch 128, 64 H100 GPUs, ~5,760 GPU hours

**输出**: 一个 discrete VLA checkpoint, 预测 OpenFAST action token, 保留 Molmo2 token interface。

**Intuition**: 这一步本质上是把"机器人控制"伪装成"语言建模"。通过 OpenFAST tokenizer 把连续 action 变成离散 token, 整个训练 pipeline 就是标准 next-token prediction, 可以无缝和 VLM data 共训。这是让 VLA 训练 stable 和 scalable 的关键。

---

### 4.2 Post-training: 加 flow-matching action expert

Pre-training 给的是 discrete action, 但部署需要 continuous control。Post-training 在 pretrain checkpoint 上接一个 **DiT-style action expert**, 用 flow matching 生成连续 action trajectory。

#### 4.2.1 Action expert 和 KV connection

**Flow matching 公式**:

给定 normalized target action chunk $a$, Gaussian noise $\epsilon$, 采样时间 $t \in [0, 1]$, 在 noise 和 data 间线性插值:

$$x_t = (1-t)\epsilon + t \cdot a$$

其中:
- $x_t$: noisy action chunk at time $t$
- $a$: 真实 action chunk (ground truth)
- $\epsilon$: 标准 Gaussian noise
- $t$: flow time, 0 = pure noise, 1 = pure data

Target velocity field:

$$u^* = a - \epsilon$$

Expert $f_\theta$ 预测这个 velocity, loss 为:

$$\mathcal{L}_{\mathrm{flow}} = \mathbb{E}_{a, \epsilon, t}\left[\left\| m \odot \left(f_\theta(x_t, t, c) - u^*\right)\right\|_2^2\right]$$

其中:
- $m$: mask, 屏蔽 padded time step 和 padded action dimension
- $c$: VLM context (task, observation, setup/control descriptor, state token)
- $f_\theta$: action expert network
- $u^*$: target velocity = $a - \epsilon$

**Inference**: 从 Gaussian noise 开始, 积分预测的 velocity field, 产生 continuous action trajectory。

**DiT-style expert block** (L=36 层, 与 VLM 同深度):

每个 block 做:
1. Action self-attention
2. Cross-attention to VLM
3. MLP

时间 embedding 产生 DiT-style shift, scale, gate 参数给三个 residual branch。Block $\ell$ 的计算:

$$h_\ell' = h_\ell + g_\ell^{\mathrm{sa}} \cdot \mathrm{SA}(\mathrm{AdaRMS}_\ell^{\mathrm{sa}}(h_\ell, t))$$

$$\bar{h}_\ell = h_\ell' + g_\ell^{\mathrm{ca}} \cdot \mathrm{CA}(\mathrm{AdaRMS}_\ell^{\mathrm{ca}}(h_\ell', t), \tilde{K}_\ell, \tilde{V}_\ell)$$

$$h_{\ell+1} = \bar{h}_\ell + g_\ell^{\mathrm{ff}} \cdot \mathrm{MLP}(\mathrm{AdaRMS}_\ell^{\mathrm{ff}}(\bar{h}_\ell, t))$$

其中:
- $h_\ell$: 第 $\ell$ 层的 hidden state
- $g_\ell^{\mathrm{sa}}, g_\ell^{\mathrm{ca}}, g_\ell^{\mathrm{ff}}$: 三个 residual branch 的 gate
- $\mathrm{AdaRMS}$: adaptive RMSNorm, 受 time embedding $t$ modulate
- $\tilde{K}_\ell, \tilde{V}_\ell$: 从 VLM KV cache 投影来的 cross-attention key, value

**关键创新: Per-layer KV connection**

不用 final hidden state, 而是用 VLM 每一层的 KV cache:

$$\tilde{K}_\ell = \mathrm{reshape}(P_K \cdot K_\ell^{\mathrm{vlm}}), \quad \tilde{V}_\ell = \mathrm{reshape}(P_V \cdot V_\ell^{\mathrm{vlm}})$$

其中:
- $K_\ell^{\mathrm{vlm}}, V_\ell^{\mathrm{vlm}}$: VLM 第 $\ell$ 层 self-attention 产生的 key, value cache
- $P_K, P_V$: 学习的线性投影, 把 VLM KV 维度对齐到 expert cross-attention 宽度
- reshape: 把投影后的 KV 组织成 expert attention head 的形状

Cross-attention:

$$\mathrm{CA}(Q_\ell, \tilde{K}_\ell, \tilde{V}_\ell) = \mathrm{softmax}\left(\frac{Q_\ell \tilde{K}_\ell^\top}{\sqrt{d_h}}\right) \tilde{V}_\ell$$

其中 $d_h$ 是 expert head dimension。

**Intuition**: 这是论文最重要的架构创新。传统做法是 action expert 只看 VLM 的最后一层 hidden state, 信息被压缩成单个 representation。Per-layer KV connection 让 expert 每一层都能 access 对应深度的 VLM attention state——早期层是 low-level visual feature, 深层是 high-level semantic, expert 能同时利用所有抽象层级的信息。这有点像 U-Net 的 skip connection 思想, 但用在 transformer 的 KV cache 上。

**Knowledge insulation** (Driess et al., 2025):

训练时, expert 条件于 VLM KV cache, 但这个 conditioning path 是 detached 的, flow loss 的梯度不回传到 VLM。这样 VLM 只被 $\mathcal{L}_{\mathrm{LM}}$ 更新, 不被 $\mathcal{L}_{\mathrm{flow}}$ 更新。

**为什么这样设计**? 因为 flow loss 的梯度会"污染" VLM 的 visual-language representation, 让它偏离原本的通用语义空间。Insulation 让 VLM 保持通用性, expert 学会"读懂" VLM context。

#### 4.2.2 Post-training recipe

**Multiple flow samples** (K=4):

每个 action chunk 在多个 noise level 上评估 flow objective:

$$\mathcal{L}_{\mathrm{flow}}(a, c) = \frac{1}{K} \sum_{i=1}^K \left\| m \odot \left(f_\theta(x_{t_i}, t_i, c) - (a - \epsilon_i)\right)\right\|_2^2$$

其中 $\{(\epsilon_i, t_i)\}_{i=1}^K$ 是 K 个独立的 (noise, time) 对。同一 visual-language context 复用 K 次, 每个样本贡献 flow trajectory 上的一个点。

K=4 在 post-training, K=8 在 fine-tuning (因 GPU memory 限制 post-training 没用 K=8)。

**Training objectives**:

$$\mathcal{L}_{\mathrm{post}} = \mathcal{L}_{\mathrm{LM}} + \mathcal{L}_{\mathrm{flow}}$$

- $\mathcal{L}_{\mathrm{LM}}$: next-token prediction, 包括 discrete action token (robot) 和 text token (VLM)
- $\mathcal{L}_{\mathrm{flow}}$: 只对 continuous robot action chunk

**Mask 设计**: target action-token span 从 expert 的 VLM conditioning path 中 mask 掉。Expert 能看到 task, observation, setup/control, state, 但看不到它要预测的 discrete action target。这样 expert 不能"作弊"。

**Padding**: action tensor 固定 shape——max horizon 30 steps, max width 32 dim。Horizon mask 移除 padded step 的 flow loss, dimension mask zero out padded dim。

**Training detail**:
- Robot batch: sequence length 2,100 (因为还要跑 action expert + 4 flow samples)
- VLM batch: 保持 4,200
- 100K updates, global batch 128, 64 H100, ~2,300 GPU hours
- VLM LR 同 pre-training, expert LR $5 \times 10^{-5}$ (更大)

---

### 4.3 Deployment: embodiment-specific fine-tuning

从 MolmoAct2-Post 起步, 用 same VLM-expert architecture。和 post-training 有四个区别:

1. **Robot-only**: 不混 VLM mixture
2. **K=8 flow samples** (从 4 升到 8)
3. **去掉 knowledge insulation**: 让 flow loss 梯度回传到 VLM (实验发现 fine-tuning 阶段 insulation 没有持续收益)
4. **冻结新加 token 的 input embedding**, 只 tune output head 和 final norm

**Bimanual YAM**: camera order 固定 (top, left, right), absolute joint pose, 30-step chunk (30 Hz), 100K updates, 2,300 GPU hours

**DROID**: 两个 exterior camera + 1 wrist; 训练时 loader 随机选一个 exterior + wrist; absolute joint pose, 15-step chunk (15 Hz); 不用 extended language annotation (fair comparison), 1,150 GPU hours

**SO-100/101**: camera order 随机化 (因为 internet data camera layout 不一致), absolute joint pose, 30-step chunk (30 Hz), 1,150 GPU hours

**LIBERO**: full LIBERO (Spatial+Object+Goal+Long) 一起训, 不分 suite; camera 固定 (front+wrist), 不用 language annotation; relative end-effector, 10 Hz, 10-step chunk; 50K updates, best ckpt at 40K

---

## 5. MolmoAct2-Think: adaptive depth reasoning

### 5.1 直觉

MolmoAct 引入了 depth-token prediction 作为中间 reasoning step——policy 先预测 depth, 再用 depth KV cache 条件 action expert, 这让 action 有 geometric grounding。

问题是: 每步都重新预测 100 个 depth token 太慢。但机器人 trajectory 有大量时间冗余——很多 depth cell 在相邻 control step 间根本不变。

MolmoAct2-Think 的核心 insight: **只对变化的 region 重新预测 depth token**, 静态部分直接复用 cache。

### 5.2 Depth 表示

- 每个观测的 depth map 量化为 10×10 grid = 100 个 spatial code position
- 每个 position 取 128 个 learned depth-code value 之一
- 用 `<depth_start>`, `<depth_end>` 包起来, 由 `<depth_output>` trigger 触发
- 用 Depth Anything V2 (Yang et al., 2024) 估 dense monocular depth
- VQ-VAE (从 MolmoAct 继承) 在 320×320 depth image 上, downsampling factor 32, 生成 10×10 codebook index (Ning et al., 2023 的 tokenization scheme)

### 5.3 Update mask 公式

$$m_{t,i} = \mathbf{1}\left[\cos(x_{t,i}, x_{t-1,i}) < 0.996\right]$$

$$b_{t,i} = \begin{cases} d_{t,i}, & m_{t,i} = 1 \\ b_{t-1,i}, & m_{t,i} = 0 \end{cases}$$

其中:
- $m_{t,i} \in \{0, 1\}^{100}$: frame $t$ position $i$ 的 update mask
- $x_{t,i}$: frame $t$ 第 $i$ 个 32×32 RGB patch (resize 到 320×320 后)
- $x_{t-1,i}$: 上一帧同位置 patch
- $\cos(\cdot, \cdot)$: cosine similarity
- $b_t$: carried-forward depth buffer
- $d_t$: 当前帧 full VQ depth codes

阈值 0.996: 如果两个 patch 的 cosine similarity 低于 0.996, 标记为"变化", 重新预测; 否则复用 cache。

### 5.4 训练 recipe

Post-training 时, robot data sample 三种 output style (各 1/3 概率):
1. **Action style**: `<action_output>`, 训 discrete + continuous action
2. **Depth style**: `<depth_output>`, 训 100 个 depth-buffer token 的 autoregressive prediction
3. **Depth-and-action style**: `<depth_output><action_output>`, 一起训。此时 action expert 条件于读入 depth token 之后的 VLM KV cache, 所以预测 depth state 对 continuous action 可用

**关键**: target action-token span 仍 mask 掉, expert 不能看到要预测的 discrete action。

Fine-tuning 时:
- 只 sample action 和 depth-and-action style, 去掉 pure depth-prediction
- **Noise injection**: 10% 的 depth-code input token 被替换为 uniform random depth code, 但 prediction target 不变 (模拟 inference 时 model 预测的 depth 不完美)
- **Learned per-layer depth gate**: 控制每个 expert layer 使用 depth prefix 的强度

Depth gate 公式:

$$c_\ell = \frac{\sum_t A_t(1 - M_t) V_{\ell,t}^{\mathrm{vlm}}}{\sum_t A_t(1 - M_t)}$$

$$g_\ell = \sigma(w_\ell^\top c_\ell + b_\ell)$$

其中:
- $M_t = 1$ 表示这个 position 是 depth-output trigger, depth delimiter, 或 depth-code token
- $A_t$: valid context position
- $c_\ell$: 第 $\ell$ 层 VLM 的 non-depth context 平均 (pooled)
- $g_\ell \in [0, 1]$: scalar gate, 决定该层 depth prefix 的 KV 贡献强度
- $w_\ell, b_\ell$: 学习的参数
- $\sigma$: sigmoid
- 初始化 bias $b_\ell = -4$, 所以 sigmoid(-4) ≈ 0.018, fine-tuning 开始时 gate 接近 0, 接近标准 action-conditioning path, 然后慢慢学

Gate 应用:

$$\bar{K}_{\ell,t}^{\mathrm{vlm}} = (1 - M_t + M_t \cdot g_\ell) K_{\ell,t}^{\mathrm{vlm}}$$

$$\bar{V}_{\ell,t}^{\mathrm{vlm}} = (1 - M_t + M_t \cdot g_\ell) V_{\ell,t}^{\mathrm{vlm}}$$

非 depth position ($M_t=0$): 系数 = 1, 不变
Depth position ($M_t=1$): 系数 = $g_\ell$, 由 gate 调制

### 5.5 Inference pipeline

输入: task, observation, proprioceptive state
Prompt: `<depth_output><action_output>`

**首帧 (无 cache)**: autoregressive 预测完整 depth sequence `<depth_start>` + 100 token + `<depth_end>`

**后续帧 (有 cache)**:
1. 比较当前 first observation image 与 cached previous image (10×10 RGB patch cosine, 阈值 0.996)
2. Updated cell: argmax decode from depth-token logits
3. Unchanged cell: replay from previous predicted depth buffer
4. 连续 unchanged span 一起 replay, changed span token-by-token decode
5. 填满 100 cell 后 emit `<depth_end>`, 存当前 image 和新 100-code buffer 作下一步 cache

**Action generation**: action expert 接收 prompt + filled depth prefix 的 VLM KV cache, 积分 flow velocity field 生成 action chunk

### 5.6 Inference optimization

Adaptive depth 引入了 systems challenge: 每帧 changed/unchanged cell 比例不同, decode schedule 是 data-dependent。

策略:
- Adaptive scheduler 保持 eager (包括 unchanged span 的 span-level replay)
- 但用 preallocated static KV cache 让 decode state 跨 step 稳定
- Regenerated depth token: 把 transformer 的 post-attention 到 next layer pre-attention 的固定 shape 部分 capture 成 CUDA Graph stage
- Attention 本身保持 eager (因为 KV length 在 decode 中变化)

**Intuition**: 这是 hybrid eager + CUDA Graph 的 trade-off。完全 capture CUDA Graph 要求 fixed shape, 但 adaptive depth 本质 data-dependent。所以只 capture 不依赖 KV length 的部分, attention 保持灵活。这种 partial graph capture 在 LLM serving 中也常见 (例如 vLLM 对 prefill/decode 的不同处理)。

---

## 6. Experiments: 全面而严谨

### 6.1 Molmo2-ER 在 embodied reasoning benchmark 上

Table 3 显示, Molmo2-ER 在 13 个 benchmark 上 9 个 SOTA, 平均 63.8%, 超过 GR-ER 1.5 Thinking (61.3%), GPT-5 (57.9%)。

**关键 takeaway**: backbone 的 embodied reasoning 能力直接迁移到下游 action learning。Table 9 显示, 用 Molmo2-ER 替换 Molmo2 (用相同 discrete action architecture fine-tune LIBERO Long), 成功率从 77.6% → 83.6%, **+6 个点**。这不是 VLM benchmark 上的虚高, 而是实打实的 policy 提升。

### 6.2 Out-of-the-box deployment

#### Simulation

Table 5 (MolmoSpaces) 和 Table 6 (MolmoBot) 显示:

- **MolmoSpaces**: MolmoAct2-DROID 平均 20.6%, vs π0.5-DROID 10.0% (+10.6%)
- **MolmoBot**: MolmoAct2-DROID 平均 87.1%, vs MolmoBot 48.4%, vs π0.5-DROID 45.2%

特别值得注意的是 Table 5 中 MolmoAct2-DROID 在 Pick & Place 任务上 oracle success vs success at end 的 delta 较小, 说明 policy 不会"反复 pick 已放置好的物体", 这对实际部署很重要。

#### Real-world

Table 4 (DROID) 和 Table 7 (SO-100/101) 显示:

- **DROID**: MolmoAct2 平均 37.7% (Table 4, MolmoSpace) vs π0.5-DROID 34.5%
- **SO-100/101**: MolmoAct2 平均 56.7% vs π0-SO100/101 45.3% vs SmolVLA 2.3%

SmolVLA 的 2.3% 显示了 community-sourced data 训出的 baseline 在 OOD 条件下脆弱, MolmoAct2 经过 quality filtering 显著提升。

### 6.3 Fine-tuning 效果

Table 8 (LIBERO) 显示:
- MolmoAct2 97.2% 平均
- MolmoAct2-Think 98.1% 平均
- π0.5 96.9%
- GR00T N1.7 97.0%
- NORA-1.5 94.5%

LIBERO-Object 上 MolmoAct2 达到 **100.0%**。

Figure 7 的 real-world YAM 评测 (8 tasks, 50 trials each):
- MolmoAct2 平均 50.1%
- 比 runner-up OpenVLA-OFT 高 15%
- 7/8 task 上最好

RoboEval (Figure 6):
- MolmoAct2 44.3%, 比 π0.5 高 3.8%
- 在 trajectory quality metric 上 (CT, TL, JPL, CPL, CJ, JJ, SC, SL) 全面领先

**Intuition**: RoboEval 的 trajectory quality metric 是 paper 的亮点之一。它回答了"成功率高但 path 是否 smooth / safe"这个问题。MolmoAct2 在 completion time, trajectory length, joint/Cartesian path length, jerk, self-collision, slip count 上都 best, 说明 policy 不只是"碰巧成功", 而是生成了高效稳定的 trajectory。

### 6.4 MolmoAct2-Think 的效果

Table 8:
- LIBERO Spatial: 98.8% (MolmoAct2-Think) vs 97.8% (MolmoAct2)
- LIBERO Object: 99.8% vs 100.0% (饱和, 略降)
- LIBERO Goal: 98.5% vs 97.8%
- LIBERO Long: 95.4% vs 93.2% (**+2.2%**)
- 平均: 98.1% vs 97.2%

最大 gain 在 Long (最难, baseline headroom 最大), 最小 gain 在 Object (饱和)。说明 adaptive depth 是真正的 geometric grounding, 而不是 noise。

### 6.5 Ablation

Table 10 (VLM-to-expert connection):
- Per-layer KV: 95.9%
- Per-head KV: 94.8%
- Hidden state: 94.0%

**Per-layer KV connection** 最好。验证了核心架构选择。

Table 11 (flow samples K):
- K=1: 94.15%
- K=2: 95.05%
- K=4: 95.15%
- K=8: 95.90%

更多 flow sample 沿同一 trajectory 提供更密 supervision, 单调改善。

Table 12 (fine-tuning design):
- Full FT + discrete co-train + no insulation: 97.20% (best)
- LoRA: 96.25%
- Action expert only: 93.05% (worst, 证明 VLM 也需要 tune)

Table 13 (depth fine-tuning):
- All enabled (mixed training + noise + depth gate): 98.10%
- 去掉 noise + gate: 97.65%
- 去掉 mixed training: 97.50%

证明三个组件都有贡献, mixed training 让 policy 同时保持 action-only path 的鲁棒性。

### 6.6 Inference speed

Figure 8 显示 (LIBERO, horizon 10, single H100):
- **MolmoAct2**: 23.02 Hz (original) → 27.39 Hz (cache) → **55.79 Hz** (CUDA Graph)
- **MolmoAct2-Think**: 8.04 Hz → 9.72 Hz → 12.71 Hz

MolmoAct2 CUDA Graph 加速 2.42×, 因为 flow matching 是 fixed-shape repeated compute, kernel launch overhead 是瓶颈, graph replay 几乎消除。

MolmoAct2-Think 加速 1.58×, 较小, 因为 adaptive depth 的 autoregressive decode 本质 sequential, variable-length, 不利于 graph capture。

**Intuition**: 55.79 Hz 已经远超 30 Hz 控制频率, 满足闭环控制需求。MolmoAct2-Think 12.71 Hz 对一些低频任务也够用, 但 high-frequency control 还需优化。

---

## 7. 这篇论文的真正贡献和启示

让我从更高层总结:

### 7.1 Per-layer KV conditioning 是核心架构创新

传统 VLA (π0, GR00T N1) 用 final hidden state condition action expert, 把 VLM 压成单 representation。MolmoAct2 让 expert 每层 access 对应深度的 VLM attention state——这本质上是 deep conditioning, 有点像 ControlNet 用 UNet 多层 feature condition diffusion。

Ablation 显示 +1.9% 提升 (95.9 vs 94.0), 不算巨大, 但概念上重要: 它说明 VLM 的"中间表示"对 action 生成有用, 不应被 final layer bottleneck。

### 7.2 OpenFAST tokenizer 让 VLA 训练变"语言建模"

把 continuous action 通过频域 + BPE 压成 discrete token, 让 pretraining 可以纯 next-token prediction 和 VLM data 混训。这极大简化了训练 pipeline, 让 VLA scalable。

而且作者开源了 tokenizer 权重和训练数据 (跨 5 个 embodiment), 这是 FAST 没做到的。

### 7.3 Adaptive depth 是 reasoning + efficiency 的折衷

MolmoAct 的 depth reasoning 提升了 grounding 但慢; MolmoAct2-Think 的 adaptive depth 让 cost 与 scene change 成正比, 而不是 grid size。这是个 elegant 的 idea: 机器人 trajectory 中大部分场景是静态的, 重新预测 depth 是浪费。

公式 $m_{t,i} = \mathbf{1}[\cos(x_{t,i}, x_{t-1,i}) < 0.996]$ 简单但有效。0.996 阈值对应约 5° 的 patch 角度变化, 对 robot 操作场景是合理 threshold。

### 7.4 Specialize-then-rehearse 避免 catastrophic forgetting

Stage 1 专攻 embodied, Stage 2 interleave with general data。这个 recipe 比"全量继续 pretrain"更高效, 比纯 fine-tune 更不易忘通用能力。p=0.5 的 sweep 结果也说明 50-50 是 Pareto frontier。

### 7.5 数据 quality 比 quantity 重要

- SO-100/101 dataset 经过 TOPReward quality gate 过滤
- DROID 用 idle-frame filter 和 language re-annotation
- YAM 严格 retry 协议
- 全量 re-annotation 让 unique label 翻倍

这些 quality control 是 MolmoAct2 outperform π0.5 (用更多但 noisy data) 的关键。

### 7.6 全开源是 community 贡献

权重 + 代码 + 完整训练数据, 包括最大开源 bimanual dataset (720 小时)。这对学术 lab 和独立研究者意义重大, 让 VLA 研究不再被闭源 frontier model 垄断。

参考:
- Blog: https://allenai.org/blog/molmoact2
- Code: https://github.com/allenai/molmoact2

---

## 8. 一些可以深挖的方向

读完 paper 我有几个想继续探索的点:

1. **Per-layer KV 的成本**: 36 层 expert × 36 层 VLM, 每层都要存 KV cache + adapter projection, memory 和 compute 开销如何? Ablation 没给推理 latency 对比 (Table 10 只给 accuracy)。

2. **Adaptive depth 在快速场景**: 如果机械臂高速运动, 大部分 patch 都 change, adaptive 退化成 full depth prediction, 加速失效。High-speed task 的极限是什么?

3. **Depth token 的 alternative**: 为什么是 depth 而不是 surface normal, semantic mask, optical flow? 作者选 depth 因为它最直接 grounding action, 但其他 modality 可能互补。

4. **Multiple flow sample 的 optimal K**: K=8 是 GPU memory 限制, 不是 accuracy 限制。如果用 gradient checkpointing 能不能上 K=16? 收益是否饱和?

5. **Language annotation 的 diversity vs accuracy**: 翻倍 unique label 提升 generalization, 但 Qwen3.5-27B 生成的 instruction 是否有 systematic bias? 是否需要 human verify?

6. **Knowledge insulation 何时该去掉**: Post-training 用 insulation, fine-tuning 去掉。这背后的 intuition 是什么? 是不是 fine-tuning 阶段 VLM 已稳定, 可以让 flow loss 微调?

总的来说, 这是一篇非常 solid 的论文, 它不追求单点 SOTA, 而是系统性地解决 VLA 真实部署的多个痛点。架构创新 (per-layer KV), 数据质量 (filtering, re-annotation), 推理优化 (adaptive depth, CUDA Graph) 协同工作, 把一个学术 foundation model 推到了可部署状态。开放数据 + 代码 + 权重让 community 能真正复现和扩展, 这是 VLA 研究民主化的重要一步。

参考:
- MolmoAct (前作): https://arxiv.org/abs/2508.07917
- FAST tokenizer: https://arxiv.org/abs/2501.09747
- π0.5: https://arxiv.org/abs/2504.16054
- Knowledge insulation: https://arxiv.org/abs/2505.23705
- Depth Anything V2: https://arxiv.org/abs/2406.09414
- DROID: https://arxiv.org/abs/2403.12945
- RoboEval: https://arxiv.org/abs/2507.00435
- LIBERO: https://arxiv.org/abs/2306.03310
- Molmo2: https://arxiv.org/abs/2601.10611
- Tulu 3: https://arxiv.org/abs/2411.15124
- MolmoBot: https://arxiv.org/abs/2603.16861
- MolmoSpaces: https://arxiv.org/abs/2602.11337
- Flow Matching: https://arxiv.org/abs/2210.02747
