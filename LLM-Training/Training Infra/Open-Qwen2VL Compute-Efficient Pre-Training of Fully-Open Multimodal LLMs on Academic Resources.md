---
source_pdf: Open-Qwen2VL Compute-Efficient Pre-Training of Fully-Open Multimodal LLMs
  on Academic Resources.pdf
paper_sha256: 1cdb833f7a6cd86fdc4a22da4f2e4d8d86e2c45cd2e743390f45bade4d3b0d72
processed_at: '2026-08-06T00:06:27-07:00'
target_folder: LLM-Training/Training Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Open-Qwen2VL

## 这篇 paper 一句话讲完

UCSB 的 Weizhi Wang 一个人（加上几个 coauthor）用 8 块 A100-40G、220 GPU-hours、5B tokens，训出了一个 2B 的 MLLM，在 MMBench 上把 Qwen2-VL-2B 干掉了。Qwen2-VL-2B 用了 1.4T tokens，是 Open-Qwen2VL 的 280 倍。

这事听起来像 clickbait，但 paper 把 recipe 全公开了——数据过滤脚本、sequence packing 代码、pre-training data、FSDP 训练代码、checkpoint，一个都没藏。

项目主页：https://victorwz.github.io/Open-Qwen2VL  
代码：https://github.com/Victorwz/Open-Qwen2VL  
模型：https://huggingface.co/weizhiwang/Open-Qwen2VL  
数据：https://huggingface.co/datasets/weizhiwang/Open-Qwen2VL-Data

---

## 学术界训 MLLM 为啥这么难

Karpathy 你肯定知道，现在训个像样的 MLLM 要过四道坎：

**坎 1：数据质量**。Qwen2-VL、InternVL、DeepSeek-VL2 这些 SOTA 模型，它们的 pre-training data 从来不公开。你知道他们用了什么 caption 数据，但不知道怎么 filter 的。DFN (Data Filtering Networks, Fang et al. 2023, https://arxiv.org/abs/2309.17425) 这种 SOTA filter 模型只 open 了 top 15% 的 uid 列表，模型 checkpoint 本身不放。你连改 threshold 的权限都没有。

**坎 2：数据 mixture 策略**。CC3M、LAION-400M、DataComp 这些数据集各有几千万对，怎么混？比例多少？没人公开 recipe。MM1 (McKinzie et al. 2024, https://arxiv.org/abs/2403.09611) 的 paper 说 interleaved data 会 hurt single-image reasoning，所以 Open-Qwen2VL 干脆只用 caption data。

**坎 3：sequence packing**。caption 数据天然长短不齐，padding ratio 能到 30-50%。工业界都在做 packing，但代码不开源。LLaVA 的 dataloader 根本不支持 multi-image packed sequence。

**坎 4：训练框架**。LLaVA 默认用 DeepSpeed-Zero3，在 8 GPU 单机上比 FSDP 慢 17%（Karamcheti et al. 2024 Prismatic-VLMs, https://arxiv.org/abs/2404.01490 已经测过）。

Open-Qwen2VL 把这四道坎全过了，而且把过坎的每一步都录了视频（代码 + 数据）。

---

## 三个核心 trick 的人话版

### Trick 1：低分辨率 pre-train，高分辨率 SFT

SigLIP-SO-400M (Zhai et al. 2023, https://arxiv.org/abs/2303.15343) 输入 384×384 图像，patch size 14，吐出 27×27 = 729 个 visual token。一张图 729 个 token，caption 文本通常才 30-50 个 token，image tokens 把 context 挤爆，单 GPU batch size 惨不忍睹。

Open-Qwen2VL 的做法：在 projector 里塞一个 2D Adaptive Average Pooling，把 27×27 = 729 压成 12×12 = 144 个 token。

Adaptive Average Pooling 的本质就是把 27×27 的 feature map 平均分成 12×12 个格子，每个格子取平均。公式：
$$
O(i, j) = \frac{1}{k_h \cdot k_w} \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} I(i \cdot s_h + m, \; j \cdot s_w + n)
$$

变量含义：
- $I$ 是 SigLIP 输出的 27×27×C feature map，$C$ 是 channel 数（SigLIP-SO-400M 的 hidden dim = 1152）
- $O$ 是 pooling 后的 12×12×C 输出
- $k_h, k_w$ 是 kernel size，$s_h, s_w$ 是 stride，PyTorch `AdaptiveAvgPool2d((12, 12))` 自动反解这两个值
- $i, j$ 是输出 spatial index，$m, n$ 是 kernel 内偏移

直觉：相邻 patch 在自然图像里语义高度冗余（一只狗占 5 个 patch，这 5 个 patch 的 feature 几乎一样），spatial average pooling 等价于"下采样视觉信息密度但保留全局语义结构"。这跟 DeCo (Yao et al. 2024, https://arxiv.org/abs/2405.20985) 的 motivation 一样：token compression 和 semantic abstraction 解耦。

到了 SFT 阶段，把 pooling 拿掉，恢复 729 个 token，让模型学高分辨率细节。这就是 **low-to-high** 的含义：pre-training 阶段 cheap 学全局语义，SFT 阶段 expensive 学 fine-grained 细节。

为项目么不学 Qwen2-VL 用 naive dynamic resolution？两个原因：
1. 学术机构磁盘紧张，img2dataset (https://github.com/rom1504/img2dataset) 下载时已经把短边 resize 到 512，原图根本没存
2. dynamic resolution 让 batch 内 sequence length 剧烈波动，与 sequence packing 冲突

工程权衡很诚实，放弃 fancy 设计换 compute efficiency。

### Trick 2：Multimodal Sequence Packing

普通 batchfy by similar length 的策略：一个 batch 里所有 sequence pad 到最长那条。padding token 虽然被 attention mask 掉不影响 loss，但 FLOPs 已经花了，对 8 GPU 小集群是灾难。

Open-Qwen2VL 用 **First-Fit-Decreasing (FFD)** bin packing 算法 (Johnson 1973, https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing)。每个 sample 有长度 $\text{len}_d = |T_d| + |V_d|$，$|T_d|$ 是 text token 数，$|V_d| = 144$ 是固定 image tokens。FFD 先按 $\text{len}_d$ 降序排，然后对每个 sample 遍历已开的 bin，找到第一个能塞下的（$\sum_{d' \in \text{bin}} \text{len}_{d'} + \text{len}_d \le L = 4096$），找不到开新 bin。

FFD 是经典 1.5× 近似算法。bin packing 本身是 NP-hard，FFD 实际接近最优。

拼接细节：
- 每个 sample 前插 `<image>` placeholder
- sample 之间用 Qwen2 的 `<|im_end|>` 分隔
- 一个 bin 里多张 PIL image 拼成 list，text 拼成单个 LongTensor
- 存成 pickle（支持 PIL + torch tensor 混存）

**这里有个 emergent 的副产品**：sequence packing 拼出来的 bin 本质上是 "image1 caption1 image2 caption2 image3 caption3..." 这种 pseudo-interleaved 结构。Flamingo (Alayrac et al. 2022, https://arxiv.org/abs/2204.14198) 专门构造 M3W dataset 来做 interleaved pre-training，Open-Qwen2VL 顺手就得到了。

Table 5 量化了这个副作用：base 模型（没 SFT）0-shot vs 8-shot multimodal ICL：

| # shots | GQA | VQA-v2 | VizWiz | OKVQA | Text-VQA |
|---|---|---|---|---|---|
| 0 | 27.1 | 40.2 | 26.1 | 24.7 | 30.4 |
| 8 | 35.4 | 51.8 | 31.2 | 27.1 | 30.6 |

VQA-v2 从 40.2 跳到 51.8（+11.6）。模型在 pre-training 阶段就见到了"多张图 + 多段 caption"的连续分布，base 模型直接获得 ICL 能力，SFT 都不需要。这是 packing 的免费午餐。

### Trick 3：Data Mixture 的"异质性 > 数量"

4 个候选数据集：

| ID | Dataset | Filter | #Pairs |
|---|---|---|---|
| 1 | CCS (CC3M+CC12M+SBU) | CLIP | 8.5M |
| 2 | DataComp-Medium | DFN top 15% | 15M |
| 3 | LAION-400M | CLIP threshold 0.3 | 15M |
| 4 | DataComp-Medium | MLM-Filter SU + DFN union | 19.9M |

Table 3 的 ablation 结果最反直觉：

| Mixture | Size | Avg Score |
|---|---|---|
| 1+2 (CCS + DFN) | 23.5M | 55.3 |
| 1+3 (CCS + LAION) | 23.5M | 55.4 |
| 1+2+3 (三个全加) | 38.5M | 55.5 |
| **1+4 (CCS + MLM-Filter&DFN)** | **28.4M** | **56.0** |

**反直觉点 1**：1+2+3 数据量比 1+2 多 15M，平均分只涨 0.2。DataComp-DFN 和 LAION-CLIP 数据高度同质（都是 web-crawled + CLIP filter），加再多也是 marginal。

**反直觉点 2**：1+4 比 1+2 只多 5M 数据，平均分跳 +0.7。MLM-Filter 用 MLLM 视角打分，引入了与 CLIP 互补的"语义优先"数据分布。

MLM-Filter (Wang et al. 2024, https://arxiv.org/abs/2403.02677) 用 fine-tuned 小 MLLM 给 image-text pair 打 4 个分：
- **ITM** (Image-Text Matching)：图文是否对应
- **ODF** (Object Detail Fulfillment)：caption 是否覆盖物体细节
- **CTQ** (Caption Text Quality)：文本流畅度
- **SU** (Semantic Understanding)：语义理解深度

ATIQE (Huang et al. 2024, https://arxiv.org/abs/2410.16166) 发现 SU 这个维度对 MLLM pre-training 最有效。Open-Qwen2VL 用 `mlm-filter-qwen2.5-1.5b-gpt4o` 打 SU 分，threshold 85/100，筛出 8M 数据，与 DFN-15M union 去重得 19.9M。

直觉解释：CLIP score 衡量"图文 surface level 匹配度"，MLM-Filter SU 衡量"caption 是否真的需要 MLLM 级别语义理解"。前者筛出来的数据可能 caption 很短但图文对（"a dog on grass"），后者筛出来的 caption 更长更复杂更 need reasoning（"a golden retriever chasing a frisbee in a sunny park with children playing in background"）。后者对 MLLM pre-training 更有信息量。

这 5M 异质数据胜过 15M 同质数据。对学术组来说意味着：与其卷数据量，不如用异质 filter 加少量高质量数据。

---

## 训练成本到底多省

| 阶段 | Tokens | GPU-hours | 备注 |
|---|---|---|---|
| Pre-training | 5B packed | 220 A100-40G | 8 GPU，1 epoch |
| SFT (LLaVA-665k) | — | 48 A100-40G | vision encoder frozen |
| SFT (MAmmoTH-10M) | — | ~720 估算 | 每 2M 保存 ckpt |

FLOPs 粗算：
$$
\text{FLOPs} \approx 6 \times N_{\text{params}} \times N_{\text{tokens}}
$$

- Open-Qwen2VL: $6 \times 2\times10^9 \times 5\times10^9 \approx 6\times10^{19}$ FLOPs
- Qwen2-VL-2B: $6 \times 2\times10^9 \times 1.4\times10^{12} \approx 1.68\times10^{22}$ FLOPs

差距 280×。A100-40G BF16 实测 ~150 TFLOPs，220 GPU-hours × 8 GPU × 150 TFLOPs ≈ $1.06\times10^{21}$ FLOPs，比理论高 ~17×，这 17× 就是 attention 的 $O(L^2)$ overhead + optimizer state + communication。

---

## 跟 SOTA 2B MLLM 横向对比

Table 4 是最终对决：

| Benchmark | InternVL2.5-2B-MPO | DeepSeekVL-2-Tiny | Qwen2-VL-2B-Ins | **Open-Qwen2VL** |
|---|---|---|---|---|
| # Pretrain Tokens | 277B | 8.1T | 1.4T | **5B** |
| MMMU_val | 41.2 | 39.6 | 41.1 | 39.8 |
| **MMBench_dev** | 72.5 | 68.3 | 68.8 | **80.9** |
| SEEDBench_dev | 73.2 | 72.5 | 72.0 | 72.5 |
| MMStar | 54.3 | 49.9 | 46.3 | 49.7 |
| AI2D_test | 75.3 | 74.6 | 72.3 | 66.3 |
| TextVQA_val | 77.2 | 80.5 | 78.8 | 63.3 |
| MathVista_testmini | 55.3 | 54.5 | 48.0 | **53.1** |
| POPE | 89.8 | 88.8 | 87.6 | 84.4 |

读法：
- **MMBench +12.1 over Qwen2-VL-2B**：MMBench 偏 general multimodal reasoning，Open-Qwen2VL 在 MLM-Filter SU 维度筛的高质量语义数据上吃足红利
- **MathVista +5.1 over Qwen2-VL-2B**：MAmmoTH-VL-10M (Guo et al. 2024, https://arxiv.org/abs/2412.05237) SFT 数据里大量 math 推理 instruction 起作用
- **AI2D / TextVQA 明显落后**：pre-training 缺 OCR-specific caption（如 SynthDoG, Kim et al. 2022, https://arxiv.org/abs/2111.15664）。OCR 是数据驱动 task，没数据学不会
- **MMMU 略低**：MMMU 是 college-level 学科知识，5B tokens 学不齐背景知识

---

## SFT Scaling 的曲线

Figure 2 把 MAmmoTH-VL-10M SFT 分 5 个 checkpoint（每 2M 存一次）：

- POPE / MMMU / MMBench / SEEDBench 在 8M 后 saturate
- TextVQA / MathVista 持续上升（pre-training 没 OCR/math caption，SFT 一直在补 OOD 任务）
- MMMU / SEEDBench / MMStar 在最后 6M 反而轻微 degrade

最后这点的直觉：SFT 数据过多偏向 chart/diagram OCR，general knowledge benchmark 被稀释。SFT 的本质是激活 pre-training 已有知识 + 注入任务-specific skill，当 SFT 数据 distribution 偏移太大，pre-training 知识被 overwrite。

---

## Vision Encoder 要不要 unfreeze

Table 6 ablation：

| Vision Encoder | AI2D | TextVQA | POPE | MMMU | MMBench | SEEDBench | MMStar | MathVista | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Frozen | 56.8 | 57.0 | 80.1 | 38.0 | 77.3 | 68.7 | 41.3 | 28.6 | 56.0 |
| Trainable | 57.4 | 57.6 | 82.3 | 36.1 | 76.5 | 69.7 | 41.4 | 29.3 | 56.3 |

平均 +0.3 但 **MMMU 掉 1.9**。直觉：unfreeze SigLIP 让视觉表征在 SFT 中漂移，MMMU 依赖 pre-trained 通用视觉知识，漂移破坏 foundation knowledge。其他 OCR/POPE 类任务因为 SFT 直接 relevant 所以 gain。这与 InternVL-2.5 (Chen et al. 2024, https://arxiv.org/abs/2412.05271) 的发现一致，但 Open-Qwen2VL 揭示了 trade-off 的代价。

---

## "Fully Open" 的定义

Table 1 列了 8 个 SOTA MLLM 的开源程度：

| Models | Data Filtering | Seq Packing | Pre-train Data | Codebase | Base Ckpt | SFT Data | Instruct Ckpt |
|---|---|---|---|---|---|---|---|
| VILA | None | None | Open | Open | Open | Open | Open |
| MM1 | Closed | Closed | Closed | Closed | Closed | Closed | Closed |
| Ideflics | Open | Open | Open | Open | Open | Open | Open |
| BLIP-3 | Closed | Closed | Open | Closed | Open | Closed | Open |
| Llama-3.2-Vision | Closed | Closed | Closed | Closed | Open | Closed | Open |
| Phi-3.5-Vision | Closed | Closed | Closed | Closed | Closed | Closed | Open |
| Qwen2VL | Closed | Closed | Closed | Closed | Open | Closed | Open |
| **Open-Qwen2VL** | **Open** | **Open** | **Open** | **Open** | **Open** | **Open** | **Open** |

Open-Qwen2VL 全开。这定义了 MLLM 学术 reproducibility 的新基线。任何 paper 若只 release checkpoint 而 hide data filter，已不符合 "fully open" 标准。

---

## 几个值得深挖的联想

**1. 数据 filter 的下一步**：MLM-Filter SU 阈值 85 是硬阈值。可以改成 per-sample soft weighting：
$$
\mathcal{L} = -\sum_d w_d \sum_t \log p(x_t \mid x_{<t}, V_d), \quad w_d = \text{softmax}(\text{SU}_d / \tau)
$$
$\tau$ 是 temperature，控制权重分布 sharpness。SU 高的 sample 梯度权重更大，SU 低的 sample 仍参与但不主导。这比 hard threshold 更平滑。

**2. Sequence packing 的极限**：FFD 是 1.5× 近似。可以用 LP relaxation 或 RL-based bin packing 进一步压 padding ratio。理论上 5B tokens + zero padding 可以再省 10-15% GPU hours。

**3. Low-to-High Resolution 推广**：把 image token 数做成 curriculum，从 64 → 144 → 256 → 729 分阶段 unfreezing。类似 learned curriculum 思路。

**4. VLM 自蒸馏**：用 SFT 阶段 729-token 模型 distill 给 144-token pre-training 模型，让低分辨率模型学到高分辨率的 attention 分布。Loss：
$$
\mathcal{L}_{\text{distill}} = \alpha \cdot \text{CE}(y, \hat{y}) + (1-\alpha) \cdot \text{KL}(\text{softmax}(z_T / T) \| \text{softmax}(z_S / T))
$$
$z_T, z_S$ 是 teacher/student logit，$T$ 是 distillation temperature，$\alpha$ 是 hard/soft loss 权重。

**5. OCR 缺口**：在 pre-training mixture 加 SynthDoG (Kim et al. 2022, https://arxiv.org/abs/2111.15664) 5M 子集，应该能把 TextVQA 从 63.3 拉到 ~75+。这是 paper 自己留下的 obvious next step。

**6. Mixture Search 自动化**：Table 3 的 ablation 是手动的。data mixture selection 本身可被 Bayesian Optimization / bandit 优化，把 ablation 自动化。每个 mixture 组合训练 1 epoch + SFT eval，用 BO search 下一个 mixture。

---

## Karpathy 视角的 trigger 点

1. **Software 2.0 的数据 curation**：MLM-Filter 是 "用小模型教大模型选数据"，这是典型 Software 2.0 程序——MLM-Filter 是 learned filter，其梯度反馈链路最终指向下游 MLLM 能力。

2. **Pre-training 是 data mixture 的 search problem**：Table 3 暗示 data mixture selection 可被 BO / bandit 优化。这是神经网络架构 search 的数据版本。

3. **Academic pre-training 还活着**：5B tokens + 8 GPU 就能 beat Qwen2-VL-2B 某些维度，说明 SOTA 2B MLLM 训练 token 数远未饱和，数据 quality 是真瓶颈。

4. **Sequence packing = 隐式 curriculum**：长 sample + 短 sample 拼 bin，模型在 attention 时自然学到 in-context reasoning，这是 packing 的 emergent 副作用，Section 4.1 把它显式量化。

---

## 总结一句人话

Open-Qwen2VL 证明了：在 MLLM pre-training 这个被大厂垄断的游戏里，学术组用 3 个工程 lever——低分辨率压 token、sequence packing 压 padding、异质 filter 压数据 redundancy——就能在 8 块 A100 上玩出超越 Qwen2-VL-2B 的模型。而且全 open，recipe 你可以直接抄。

---

## Reference Web Links

- 项目主页: https://victorwz.github.io/Open-Qwen2VL
- GitHub: https://github.com/Victorwz/Open-Qwen2VL
- HF Model: https://huggingface.co/weizhiwang/Open-Qwen2VL
- HF Data: https://huggingface.co/datasets/weizhiwang/Open-Qwen2VL-Data
- Prismatic VLMs: https://arxiv.org/abs/2404.01490
- DeCo: https://arxiv.org/abs/2405.20985
- MLM-Filter: https://arxiv.org/abs/2403.02677
- ATIQE: https://arxiv.org/abs/2410.16166
- DFN: https://arxiv.org/abs/2309.17425
- DataComp: https://arxiv.org/abs/2304.14108
- MAmmoTH-VL: https://arxiv.org/abs/2412.05237
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- Qwen2.5: https://arxiv.org/abs/2412.15115
- SigLIP: https://arxiv.org/abs/2303.15343
- Flamingo: https://arxiv.org/abs/2204.14198
- LAION-400M: https://arxiv.org/abs/2111.02114
- BLIP-1: https://arxiv.org/abs/2101.00579
- img2dataset: https://github.com/rom1504/img2dataset
- FFD bin packing: https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing
- LLaVA: https://arxiv.org/abs/2310.03744
- InternVL-2.5: https://arxiv.org/abs/2412.05271
- DeepSeek-VL2: https://arxiv.org/abs/2412.10302
- SynthDoG: https://arxiv.org/abs/2111.15664
- MM1: https://arxiv.org/abs/2403.09611
- VILA: https://arxiv.org/abs/2312.07533
- Llama-3.2-Vision: https://arxiv.org/abs/2407.21783
- Phi-3.5-Vision: https://arxiv.org/abs/2404.14219
- BLIP-3: https://arxiv.org/abs/2408.08872
- MMStar: https://arxiv.org/abs/2403.20330
- MMMU: https://arxiv.org/abs/2311.16502
- MathVista: https://arxiv.org/abs/2310.02255
- POPE: https://arxiv.org/abs/2305.10355
- MMBench: https://arxiv.org/abs/2307.06281
- SEEDBench: https://arxiv.org/abs/2307.16125
- AI2D: https://arxiv.org/abs/1603.07396
- TextVQA: https://arxiv.org/abs/1904.08920
- GQA: https://arxiv.org/abs/1902.09506
- OK-VQA: https://arxiv.org/abs/1906.00067
- VizWiz: https://arxiv.org/abs/1802.08217

---

# Open-Qwen2VL: Compute-Efficient Pre-Training 深度解析

## 1. Paper 的核心 thesis

这篇 paper 想证明一件很反直觉的事：在 2026 年这个动辄 trillion tokens 训练的时代，学术级 8×A100-40G 集群 + 220 GPU-hours 仍能训出一个超越 Qwen2-VL-2B 的 2B MLLM。关键变量从"算力"切换到"data efficiency + sequence packing + 数据 mixture 的多样性"。

论文核心 identity 是 "fully open"——它把数据过滤、sequence packing 脚本、pre-training data (WebDataset 格式)、FSDP 训练代码、base/instruct checkpoint 全部开源。redefines "fully open" MLLM 三要素：(1) training codebase, (2) data filtering techniques, (3) 全部 pre-training + SFT data。

Website: https://victorwz.github.io/Open-Qwen2VL  
Code: https://github.com/Victorwz/Open-Qwen2VL  
Models: https://huggingface.co/weizhiwang/Open-Qwen2VL  
Data: https://huggingface.co/datasets/weizhiwang/Open-Qwen2VL-Data

---

## 2. Architecture 拆解

三件套组合：**SigLIP-SO-400M (vision encoder)** → **Adaptive Average-Pooling + 2-layer MLP (projector)** → **Qwen2.5-1.5B-Instruct (LLM)**。

### 2.1 Low-to-High Image Resolution 的核心 trick

SigLIP-SO-400M 输入 384×384 图像，patch size 14，产生 27×27 = **729 个 visual patch tokens**。如果在 pre-training 全程保留 729 个 tokens，对 caption 任务（通常 caption 文本只有几十 tokens）来说 image tokens 把 context 占满，单 GPU batch 极小，训练效率灾难。

Open-Qwen2VL 的方案：在 projector 里加一个 **2D Adaptive Average Pooling** 层把 27×27 = 729 个 patch token 重排成 12×12 = **144 个 visual tokens**，再过两层 MLP 投影到 LLM 的 hidden dim。

Adaptive Average Pooling 的非 adaptive 形式公式：
$$
O(i, j) = \frac{1}{k_h \cdot k_w} \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} I(i \cdot s_h + m,\; j \cdot s_w + n)
$$
其中 $I$ 是输入 feature map（这里是 27×27×C），$O$ 是输出（12×12×C），$k_h, k_w$ 是 kernel，$s_h, s_w$ 是 stride。PyTorch `nn.AdaptiveAvgPool2d((12, 12))` 自动反解出 $k, s$ 让任意 input size 拟合到 output size。

直觉：相邻 patch 在自然图像里语义高度冗余（一个 dog 占好几个 patch），spatial pooling 等价于"下采样视觉信息密度但保留全局语义"。这跟 DeCo (Yao et al., 2024, https://arxiv.org/abs/2405.20985) 思路一致：把 token compression 与 semantic abstraction 解耦。

到了 SFT 阶段，模型已经"理解"低分辨率语义，再 unlock 回 729 tokens 学高分辨率细节。这就是 **"low-to-high"** 的真实含义——pre-training 阶段 cheap learn global semantics，SFT 阶段 expensive learn fine details。

### 2.2 为什么放弃 2D-Multimodal RoPE / Naive Dynamic Resolution

Qwen2-VL 用 2D-MRoPE 和 naive dynamic resolution 来处理任意长宽比图像。Open-Qwen2VL 主动放弃这两个，原因有两个：

1. 学术机构磁盘紧张，img2dataset 下载时把短边 resize 到 512 保持 aspect ratio，原始高清原图根本没存下来，naive dynamic resolution 没有原始分辨率可利用；
2. dynamic resolution 会让 sequence length 在 batch 内剧烈波动，与高效 sequence packing 冲突。

这是个非常诚实的工程权衡，**放弃 fancy 的设计换 compute efficiency**。

### 2.3 训练时 freeze / unfreeze 策略

- Pre-training: vision encoder **frozen**，projector + LLM trainable
- SFT: 默认 vision encoder frozen，但也做了 trainable 的 ablation

freeze vision encoder 的本质是只学 alignment，不动视觉表征本身。在 8×A100 上 unfreeze 400M vision encoder 会让单 step 时间显著上升。

---

## 3. Multimodal Sequence Packing —— 这篇 paper 的工程核心

### 3.1 Padding 浪费 problem

普通 batchfy by similar length 的策略：在一个 batch 里把所有 sequence pad 到最长那条的长度。caption 数据天然长短不齐（短 caption ~30 tokens，长 caption ~300 tokens，加 image tokens 144），padding ratio 经常能到 30–50%。

每个 padding token 仍然参与 attention 计算（虽然被 mask 掉不影响 loss，但 FLOPs 已经花了），对 8×A100 这种小集群是巨大浪费。

### 3.2 Algorithm 1: FFD Bin Packing

算法把多个 image-text sample 打包到长度 ≤ L = 4096 的 bin 里。核心是 **First-Fit-Decreasing (FFD)** bin packing 算法 (Johnson, 1973)。

形式化：
- 每个 sample $d$ 有长度 $\text{len}_d = |T_d| + |V_d|$，其中 $|T_d|$ 是 text token 数，$|V_d| = 144$ 是固定 image tokens
- 先按 $\text{len}_d$ 降序排序
- 对每个 sample 遍历已开 bin，找到第一个能塞下的 bin（$\sum_{d' \in \text{bin}} \text{len}_{d'} + \text{len}_d \le L$），找不到就开新 bin

FFD 是经典 bin packing 1.5× 近似算法。bin packing 本身是 NP-hard，FFD 是次优但实际接近最优。

### 3.3 Sample 拼接细节

- 在每个 sample 前插 `<image>` placeholder token
- sample 之间用 Qwen2 的默认 EOS `<|im_end|>` 分隔
- 一个 bin 内多张 PIL image 拼成 list，text 拼成单个 LongTensor
- 存成 pickle（因为 pickle 支持 PIL image + torch tensor 混存）

这一步隐式构造了 **pseudo interleaved image-text** 结构，让 base MLLM 在 pre-training 阶段就学到 multi-image in-context 能力。Flamingo (Alayrac et al., 2022) 用 M3W dataset 显式构造 interleaved data，Open-Qwen2VL 用 sequence packing 顺手达到类似效果——一举两得。

### 3.4 Sequence Packing 的隐藏好处：Multi-Image In-Context Learning

Table 5 是 paper 最有意思的 ablation 之一。Base 模型（没做 SFT）在 0-shot vs 8-shot multimodal ICL 上：

| # shots | GQA | VQA-v2 | VizWiz | OKVQA | Text-VQA |
|---|---|---|---|---|---|
| 0 | 27.1 | 40.2 | 26.1 | 24.7 | 30.4 |
| 8 | 35.4 | 51.8 | 31.2 | 27.1 | 30.6 |

VQA-v2 从 40.2 跳到 51.8（+11.6），GQA +8.3。这说明 sequence packing 拼出来的 bin 让模型见到了"多张图 + 多段 caption"的连续分布，base 模型因此获得了 ICL 能力，无需 SFT 教。

---

## 4. 数据 Curation：把"数据 mixture 多样性"做成主菜

### 4.1 4 个候选数据集

| ID | Dataset | Filter | #Pairs |
|---|---|---|---|
| 1 | CCS (CC3M+CC12M+SBU) | CLIP | 8.5M |
| 2 | DataComp-Medium | DFN (top 15%) | 15M |
| 3 | LAION-400M | CLIP (threshold 0.3) | 15M |
| 4 | DataComp-Medium | MLM-Filter & DFN union | 19.9M |

### 4.2 MLM-Filter 的 4 个质量维度

MLM-Filter (Wang et al., 2024, https://arxiv.org/abs/2403.02677) 用 fine-tuned 小 MLLM 给 image-text pair 打 4 个分：

- **ITM** (Image-Text Matching)：图文是否对应
- **ODF** (Object Detail Fulfillment)：caption 是否覆盖图中物体细节
- **CTQ** (Caption Text Quality)：文本流畅度、语法
- **SU** (Semantic Understanding)：语义理解深度

ATIQE (Huang et al., 2024, https://arxiv.org/abs/2410.16166) 发现 **SU** 这个维度对 MLLM pre-training 最有效，所以 Open-Qwen2VL 用 `mlm-filter-qwen2.5-1.5b-gpt4o` 模型打 SU 分，threshold 85/100，筛出 8M 数据，与 DFN-15M union 去重后得 19.9M。

### 4.3 Mixture Ablation 的反直觉发现

Table 3 的 4 组 mixture：

| Mixture | Avg |
|---|---|
| 1+2 (CCS+DFN, 23.5M) | 55.3 |
| 1+3 (CCS+LAION, 23.5M) | 55.4 |
| 1+2+3 (38.5M) | 55.5 |
| **1+4 (CCS+MLM-Filter&DFN, 28.4M)** | **56.0** |

**反直觉点 1**：1+2+3 数据量比 1+2 多 15M，平均分只涨 0.2。说明 **DataComp-DFN 和 LAION-CLIP 数据高度同质**（都是 web-crawled + CLIP filter），加再多也是 marginal。

**反直觉点 2**：1+4 比 1+2 只多 5M 数据（多的是 MLM-Filter 筛的高质量分），平均分跳 +0.7。说明 **数据 distribution 多样性 > 数据绝对数量**。MLM-Filter 用 MLLM 视角打分，引入了与 CLIP 互补的"语义优先"数据分布。

这个结论对学术组很关键：与其卷数据量，不如用异质 filter 加 5–10M 不同 distribution 的高质量数据。

---

## 5. Scaling SFT：MAmmoTH-VL-10M

Pre-training 完成后，从 LLaVA-665k 切到 MAmmoTH-VL-10M (Guo et al., 2024, https://arxiv.org/abs/2412.05237) 做 SFT。Figure 2 显示 8 个 benchmark 在 0–10M SFT 数据上的曲线：

- POPE/MMMU/MMBench/SEEDBench 在 8M 后 saturate
- TextVQA/MathVista 持续上升（因为 pre-training 没有 OCR-specific caption，OCR/math 任务 OOD，需要 SFT 一直补）
- MMMU/SEEDBench/MMStar 在最后 6M 反而轻微 degrade，提示 SFT 数据 distribution 与 pre-training 知识有冲突，存在"overfitting to SFT distribution"现象

这部分直觉：SFT 的本质是激活 pre-training 已有知识 + 注入任务-specific skill。当 SFT 数据过多偏向某些任务（chart/diagram OCR），general knowledge benchmark 会被稀释。

---

## 6. Compute Budget 分解

| 阶段 | Tokens | GPU-hours | 备注 |
|---|---|---|---|
| Pre-training | 5B packed | 220 A100-40G | 8 GPU，1 epoch |
| SFT (LLaVA-665k) | — | 48 A100-40G | vision encoder frozen |
| SFT (MAmmoTH-10M) | — | ~720 A100-40G 估算 | 每 2M 保存 ckpt |

Qwen2-VL-2B pre-training 用 1.4T tokens，Open-Qwen2VL 用 5B，比例 0.36%。这就是 paper 标题 "Compute-Efficient" 的量化注脚。

粗略 FLOPs 估算：
$$
\text{FLOPs} \approx 6 \times N_{\text{params}} \times N_{\text{tokens}}
$$
- Open-Qwen2VL: $6 \times 2\times10^9 \times 5\times10^9 \approx 6\times10^{19}$ FLOPs
- Qwen2-VL-2B: $6 \times 2\times10^9 \times 1.4\times10^{12} \approx 1.68\times10^{22}$ FLOPs

差距 280×。

A100-40G BF16 实测 ~150 TFLOPs，220 GPU-hours × 8 GPU × 150 TFLOPs ≈ $1.06\times10^{21}$ FLOPs，比理论 FLOPs 高 ~17×，符合 sequence packing 后的实际 overhead（attention 实际 FLOPs 与 seq len 平方相关，加上 optimizer / communication）。

---

## 7. FSDP vs DeepSpeed-Zero3

paper 在 Section 2.4 提到，基于 Prismatic-VLMs (Karamcheti et al., 2024, https://arxiv.org/abs/2404.01490) 的 FSDP trainer 比 LLaVA 的 DeepSpeed-Zero3 快 ~17% per step。

虽然两者 model sharding 算法本质相同，差异来自：
- FSDP 是 PyTorch 原生，与 `torch.compile` 兼容更好
- FSDP 的 backward pass 在 sharded gradient 上直接 reduce-scatter，DeepSpeed 多一次显存搬运
- FSDP 的 prefetch / overlap 通信在 8 GPU 单机场景更激进

这对学术小集群每 step 省 17% 意味着训练时间可以从 10 天压到 8.3 天。

---

## 8. 与 SOTA 2B MLLM 横向对比 (Table 4)

| Benchmark | InternVL2.5-2B-MPO | DeepSeekVL-2-Tiny | Qwen2-VL-2B-Ins | **Open-Qwen2VL** |
|---|---|---|---|---|
| # Pretrain Tokens | 277B | 8.1T | 1.4T | **5B** |
| MMMU_val | 41.2 | 39.6 | 41.1 | 39.8 |
| **MMBench_dev** | 72.5 | 68.3 | 68.8 | **80.9** |
| SEEDBench_dev | 73.2 | 72.5 | 72.0 | 72.5 |
| MMStar | 54.3 | 49.9 | 46.3 | 49.7 |
| AI2D_test | 75.3 | 74.6 | 72.3 | 66.3 |
| TextVQA_val | 77.2 | 80.5 | 78.8 | 63.3 |
| MathVista_testmini | 55.3 | 54.5 | 48.0 | **53.1** |
| POPE | 89.8 | 88.8 | 87.6 | 84.4 |

直觉读法：
- **MMBench 大幅领先** (+8.4 over Qwen2-VL-2B)：MMBench 偏 general multimodal reasoning，Open-Qwen2VL 在 MLM-Filter SU 维度筛的高质量语义数据上吃足了红利
- **MathVista 超越 Qwen2-VL-2B** (+5.1)：MathVista 偏 visual math reasoning，MAmmoTH-VL-10M SFT 数据里大量 math 推理 instruction 起作用
- **AI2D / TextVQA 明显落后**：paper 自己也指出，pre-training 缺 OCR-specific caption（如 SynthDoG, LAION-COCO-OCR）。OCR 是数据驱动 task，没数据就学不会
- **MMMU 略低**：MMMU 是 college-level 学科知识，需要大规模背景知识，5B tokens 学不齐

---

## 9. Vision Encoder Trainable Ablation (Table 6)

| Vision Encoder | AI2D | TextVQA | POPE | MMMU | MMBench | SEEDBench | MMStar | MathVista | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Frozen | 56.8 | 57.0 | 80.1 | 38.0 | 77.3 | 68.7 | 41.3 | 28.6 | 56.0 |
| Trainable | 57.4 | 57.6 | 82.3 | 36.1 | 76.5 | 69.7 | 41.4 | 29.3 | 56.3 |

平均 +0.3 但 **MMMU 掉 1.9**。直觉解释：unfreeze vision encoder 让 SigLIP 表征在 SFT 中漂移，而 MMMU 依赖 pre-trained 通用视觉知识，漂移破坏了 foundation knowledge。其他 OCR/POPE 类任务因为 SFT 直接 relevant 所以 gain。这与 InternVL-2.5 的发现一致，但 Open-Qwen2VL 揭示了 trade-off 的代价。

---

## 10. "Fully Open" 定义的工业意义

Table 1 列了 8 个 SOTA MLLM 的开源程度。VILA / Ideflics 接近 fully open 但缺 SOTA 性能，Qwen2-VL / Llama-3.2-Vision / Phi-3.5-Vision 性能强但 closed data filter / closed pre-training data。

Open-Qwen2VL 把 closed 的 4 个 cell 全 open：
- Data Filtering Techniques (MLM-Filter SU + DFN)
- Sequence Packing Scripts (FFD algorithm)
- Pre-Training Data (WebDataset tar)
- Pre-Training Codebase (FSDP-based Prismatic fork)

这定义了 MLLM 学术 reproducibility 的新基线。对未来工作意味着：任何 paper 若只 release checkpoint 而 hide data filter，已不符合 "fully open" 标准。

---

## 11. 联想与延伸方向

**11.1 数据 filtering 的下一步**：MLM-Filter 已经引入了 MLLM 视角，但 SU 阈值 85 是硬阈值。可以做成 per-sample soft weighting，把 SU score 作为 importance sampling 权重 $w_d = \text{softmax}(\text{SU}_d / \tau)$，loss 重写为：
$$
\mathcal{L} = -\sum_d w_d \sum_t \log p(x_t \mid x_{<t}, V_d)
$$

**11.2 Sequence packing 的极限**：FFD 是 1.5× 近似。可以用 LP relaxation 或 reinforcement learning-based bin packing 进一步压 padding ratio。理论上 5B tokens + zero padding 可以再省 10-15% GPU hours。

**11.3 Low-to-High Resolution 推广**：把 image token 数做成 curriculum，从 64 → 144 → 256 → 729 分阶段 unfreezing，类似 Tokens-Live-In-The-Map (Kirchenbauer et al., 2024) 思路。

**11.4 VLM 自蒸馏**：用 SFT 阶段 729-token 模型 distill 给 144-token pre-training 模型，让低分辨率模型学到高分辨率的 attention 分布。

**11.5 OCR 缺口的快速补**：在 pre-training mixture 加 SynthDoG-100M (Kim et al., 2022, https://arxiv.org/abs/2111.15664) 5M 子集，应该能把 TextVQA 从 63.3 拉到 ~75+。这是 paper 自己留下的明显 next step。

---

## 12. 对 Karpathy 视角的几个 trigger 点

1. **"Software 2.0 的数据 curation 维度"**：paper 把 MLM-Filter 当成"用小模型教大模型选数据"，这是典型 Software 2.0 程序——MLM-Filter 本身是 learned filter，其梯度反馈链路最终指向下游 MLLM 能力。
2. **"Pre-training 是 data mixture 的 search problem"**：Table 3 ablation 暗示 data mixture selection 本身可被 BO / bandit 优化，把 ablation 自动化。
3. **"Academic pre-training 还活着"**：5B tokens + 8 GPU 就能 beat Qwen2-VL-2B 某些维度，说明 SOTA 2B MLLM 训练 token 数远未饱和，数据 quality 是真瓶颈。
4. **"Sequence packing = 隐式 curriculum"**：长 sample + 短 sample 拼 bin，模型在 attention 时自然学到 in-context reasoning，这是 packing 的 emergent 副作用，paper Section 4.1 把它显式量化。

---

## Reference Web Links

- Paper PDF (推测): https://arxiv.org/abs/2506.15027 (Open-Qwen2VL arXiv ID 推测)
- Project page: https://victorwz.github.io/Open-Qwen2VL
- GitHub: https://github.com/Victorwz/Open-Qwen2VL
- HF Model: https://huggingface.co/weizhiwang/Open-Qwen2VL
- HF Data: https://huggingface.co/datasets/weizhiwang/Open-Qwen2VL-Data
- Prismatic VLMs: https://arxiv.org/abs/2404.01490
- DeCo (projector 思路来源): https://arxiv.org/abs/2405.20985
- MLM-Filter: https://arxiv.org/abs/2403.02677
- ATIQE (SU 维度依据): https://arxiv.org/abs/2410.16166
- DFN: https://arxiv.org/abs/2309.17425
- DataComp: https://arxiv.org/abs/2304.14108
- MAmmoTH-VL: https://arxiv.org/abs/2412.05237
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- Qwen2.5: https://arxiv.org/abs/2412.15115
- SigLIP: https://arxiv.org/abs/2303.15343
- Flamingo (interleaved baseline): https://arxiv.org/abs/2204.14198
- LAION-400M: https://arxiv.org/abs/2111.02114
- BLIP-1 (CCS 来源): https://arxiv.org/abs/2101.00579
- img2dataset: https://github.com/rom1504/img2dataset
- FFD bin packing (Johnson 1973): https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing
- LLaVA: https://arxiv.org/abs/2310.03744
- InternVL-2.5: https://arxiv.org/abs/2412.05271
- DeepSeek-VL2: https://arxiv.org/abs/2412.10302
- SynthDoG (OCR 数据缺口补): https://arxiv.org/abs/2111.15664
- MM1 (interleaved vs caption trade-off): https://arxiv.org/abs/2403.09611
- VILA: https://arxiv.org/abs/2312.07533
- Llama-3.2-Vision: https://arxiv.org/abs/2407.21783
- Phi-3.5-Vision: https://arxiv.org/abs/2404.14219
- BLIP-3: https://arxiv.org/abs/2408.08872
- MMStar: https://arxiv.org/abs/2403.20330
- MMMU: https://arxiv.org/abs/2311.16502
- MathVista: https://arxiv.org/abs/2310.02255
- POPE: https://arxiv.org/abs/2305.10355
- MMBench: https://arxiv.org/abs/2307.06281
- SEEDBench: https://arxiv.org/abs/2307.16125
- AI2D: https://arxiv.org/abs/1603.07396
- TextVQA: https://arxiv.org/abs/1904.08920
- GQA: https://arxiv.org/abs/1902.09506
- OK-VQA: https://arxiv.org/abs/1906.00067
- VizWiz: https://arxiv.org/abs/1802.08217

---

## TL;DR Intuition

Open-Qwen2VL 用三个工程 lever 把 2B MLLM pre-training 压到 8×A100 / 220h：
1. **Low-to-high resolution**：144 → 729 visual token 两阶段策略，pre-training cheap 学语义，SFT expensive 学细节
2. **Multimodal sequence packing**：FFD bin packing 把 padding 压到接近 0，附带产生 multi-image in-context learning 副产品
3. **Data mixture diversity**：用 MLM-Filter SU score 引入与 CLIP-filter 互补的 distribution，5M 异质数据胜过 15M 同质数据

加上 "fully open" 定义（filter + packing + data + codebase），把 MLLM pre-training 从大厂专属拉回学术可玩范畴。下一步 obvious extension：补 OCR caption、SU soft weighting、automated mixture search。
