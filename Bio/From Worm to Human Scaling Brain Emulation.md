---
source_pdf: From Worm to Human Scaling Brain Emulation.pdf
paper_sha256: 1c24fd248941baef174117d7ba1099a892d87d69e088cb595b52517a6de81093
processed_at: '2026-08-04T11:09:57-07:00'
target_folder: Bio
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲这篇论文

## 这篇论文到底在说什么

简单讲：作者 Isaak Freeman 在 MIT Boyden Lab 待了两年，访谈了 50 多个研究员，翻了几百篇 paper，然后写了一篇硕士论文，核心 argument 就一句话——

**"复制一个人类大脑，从工程角度看，这件事没那么科幻了。"**

他的 reasoning 很直接：brain emulation 这件事，需要三样东西同时 ready：
1. 知道大脑怎么连的（connectome）
2. 知道大脑在干什么（functional recording）
3. 有足够算力跑 simulation

这三样东西各自都在按 exponential curve 往上走，而且正在交汇。特别是第三样——AI 公司为了训练大模型砸了几百亿建 GPU cluster，这些算力恰好够跑 human-scale brain simulation。作者的意思是，硅谷在建的东西，附带地把 brain emulation 的算力门槛给铺平了。

---

## 先说 connectome：大脑的"接线图"

### 我们现在在哪

想象你有一台电脑，你想完全复制它。第一步是搞清楚每个零件连到哪个零件。大脑也一样——86 billion 个 neuron，每个平均连几千个 synapse，你需要一张完整的 wiring diagram。

这就是 connectome。progress 长这样：

- 1986 年：C. elegans 的 302 个 neuron 全部 mapped 完。花了 15 年。
- 2024 年：果蝇整个大脑 mapped 完，139,255 个 neuron，54.5 million synapse。这是目前最大的完整 connectome。
- 2024 年：一小块人类皮层，1 立方毫米，49,000 个 neuron，150 million synapse。

从 302 到 139,255 用了 38 年。听起来很慢，对吧？

### 但关键是成本在暴跌

这才是真正有意思的地方：

| 年份 | 生物 | 每个 neuron 的重建成本 |
|------|------|---------------------|
| 1980s | C. elegans | ~$16,500 |
| 2025 | Drosophila | ~$214 |
| 2025 | Zebrafish | ~$100 |
| 2025 | Mammalian | $500-1,000 |

38 年里降了 75 倍。这跟当年 genome sequencing 的曲线很像——一开始慢得要死，然后突然 super-exponential 起飞。

为什么？因为 proofreading（人工校对）占了 connectomics 成本的 90% 以上。果蝇的 connectome 花了 33 person-years 来 proofread。如果没有更好的 ML 来做这件事，mouse brain（比果蝇大 1000 倍）的 proofreading 成本会 astronomical。

但现在 flood-filling networks、SegCLR 这些 ML 方法已经在 proofreading 上实现 10 倍以上的 accuracy 提升。PRISM 那篇 2025 的 paper 把 binary protein barcoding + ExM + self-proofreading AI 三个技术合在一起，automatic tracing accuracy 比传统方法高 8 倍。

### 做一个人类 connectome 要多少台 microscope

作者算了一笔很直观的账：

人类大脑体积 1.4×10²⁴ nm³。用 16nm 分辨率的 voxel，10 GHz throughput 的 multi-beam SEM（这是 2019 年 331-beam 原型机的外推，业界专家认为 1000-beam 不夸张），50% 的 uptime（考虑维护、对准、样品移动），10 年完成。

需要多少台？

**200 台。**

每台假设 $1M（mass-produced 而非 handcrafted），总共 $200M 的 microscope 成本。

200 台多不多？想想 semiconductor fab——一台 EUV 光刻机就 $200M+，一个 fab 动辄 $10-20B。200 台电子显微镜在 mega-project 尺度下完全 feasible。

### Expansion Microscopy 的巧妙思路

传统 light microscope 因为光的 diffraction limit，分辨率卡在 250nm 左右。这远不够看 synapse（20-30nm 的 cleft）。

Boyden Lab 的 ExM 思路很 hacky：**不提高 microscope 分辨率，把样品物理放大**。

把组织锚定到 hydrogel 上，然后加水让 gel 膨胀。4 倍扩张 → 有效分辨率 62.5nm。10 倍 → 25nm。20 倍（在开发中）→ 12.5nm，够看 synapse 了。

而且 ExM 好处在于可以同时做 protein staining——你可以在 connectome 上标注哪些地方有 AMPAR、NMDAR、GABAR 这些 receptor。对 simulation 来说，这信息极其关键。

### Protein barcoding 的 combinatorics

这个 idea 我觉得特别 elegant。

传统 connectomics 需要把每个 axon 一路 trace 到底，不能断。一旦某个切片丢失或者 tracing 出错，后面全乱。

Barcoding 的思路：给每个 neuron 一个独特的 molecular barcode。这样即使你 trace 断了，只要读到 barcode，就知道这段是哪个 neuron 的。

用 25 个 protein，每个 binary（有/无），理论上是 2²⁵ ≈ 33M 种组合。但实际 AAV 的 expression probability 大概是 0.3，用 Shannon entropy 算 effective barcode space 大概 2²² ≈ 4.3M 种。

人类大脑有 86 billion neuron，每个平均 8,000-40,000 synapse。barcoding 不需要 globally unique——只需要在一个 neuron 的 synaptic neighborhood 内 unique 就行。4.3M 相对于 40,000 的 neighborhood 有 100 倍以上的 margin。

这就够用了。

---

## 再说 functional imaging：大脑"活着的时候在干什么"

### 光学成像的指数增长

Stevenson 和 Kording 2011 年发现，同时记录的 neuron 数量大约每 7.4 年翻倍。但自从 optical imaging 起来后，这个速度在加快。

现在你能同时记录多少 neuron：

| 生物 | 方法 | neuron 数 | 频率 |
|------|------|----------|------|
| C. elegans | SCAPE | whole-body | 25 Hz |
| 斑马鱼幼虫 | calcium | 70,000 | 1 Hz |
| 斑马鱼幼虫 | voltage | ~1/3 brain | - |
| 果蝇成体 | two-photon | near-whole-brain | 2 Hz |
| 小鼠皮层 | light beads | 1,000,000 | 2 Hz |

从小鼠皮层能同时记录 1 million neuron，这是用 light beads microscopy 实现的。配合 crystal skull 的弧形玻璃窗口，可以覆盖 30+ 个 neocortical area 的 80 万到 110 万 neuron。

### 但有一个 glass ceiling

这里作者提了一个很关键的 insight：**whole-brain single-neuron imaging 有一个光学穿透极限，大概 1-2mm**。

光在生物组织里会被 scatter 和 absorb。透明的生物可以 whole-brain imaging，不透明的做不到。

所以：

- C. elegans：天然透明，whole-body imaging 没问题
- 斑马鱼幼虫：透明，whole-brain imaging 没问题
- 果蝇成体：半透明，borderline
- 小鼠：有 skull，组织厚，whole-brain single-neuron imaging 做不到
- 人类：完全做不到

这意味着什么？

**斑马鱼幼虫是目前最适合做 end-to-end brain emulation 验证的 vertebrate model organism。** 它有 vertebrate 的脑结构，~100,000 个 neuron，whole-brain calcium imaging 已经实现（70K neuron at 1 Hz），voltage imaging 也到了 1/3 brain。如果斑马鱼的 connectome 也完成，你就有了一个完整的 structure + function + behavior 数据集来验证 simulation。

至于小鼠和人类——whole-brain single-neuron imaging 在 in vivo 条件下目前根本做不到，需要 tissue clearing 技术（比如 tartrazine）或者完全不同的 recording paradigm 的突破。

---

## 然后是 simulation：算力到底够不够

### 从最简单到最复杂的 neuron model

neuron model 复杂度是一个 spectrum：

**LIF（Leaky Integrate-and-Fire）**——最简单的 differential equation，大概 1907 年就提出来了。每个 neuron 就一个 scalar voltage variable，到了 threshold 就 spike，reset。

**Izhikevich**——稍微复杂一点，两个变量，可以 reproduce 很多 spiking pattern。

**Hodgkin-Huxley**——完整的 biophysical model，有 sodium/potassium/leak channel 的 gating variables。每个 compartment 一组 ODE。

**Multicompartmental HH**——每个 neuron 拆成 ~1000 个 compartment，每个有自己的 HH dynamics，加 7000 个 synapse。

### 算一下 human-scale 需要多少算力

**最乐观（LIF model）**：

每个 neuron 每毫秒 ~40 FLOP。86 billion neuron，real-time（1000 ms/s）：

40 × 8.6×10¹⁰ × 10³ = **3.4 petaFLOP/s**

单个 H100 GPU 的 FP16 dense tensor throughput 大约 1 petaFLOP/s，带 structured sparsity 接近 2 petaFLOP/s。所以 LIF 级别的 human brain simulation，**两三个 H100 就够了**。

**最悲观（multicompartmental HH）**：

每个 neuron 1000 compartment，每个 compartment 每毫秒 690 FLOP，加 7000 synapse 每个 spike 10⁴ FLOP at 10 Hz：

1000 × 690 × 10³ + 7000 × 10 × 10⁴ = 1.39×10⁹ FLOP/s per neuron

× 86 billion = **~10²⁰ FLOP/s**

这跟 xAI 的 Colossus cluster（100,000 H100）的总算力差不多。

所以 raw compute 的 range 是：**3.4 petaFLOP/s 到 10²⁰ FLOP/s**，取决于你用多详细的 model。前者一台 GPU 就够，后者需要一个巨型 AI cluster。两者都在当前技术的 reach 之内。

而且，关键 insight：synaptic modeling 的 compute cost 远大于 neuronal dynamics。LIF 和 HH 之间的 per-neuron gap 相对于 synaptic detail 来说是小头。

### 真正的 bottleneck：memory 和 interconnect

这里作者提了一个极其重要的 point。

过去 30 年，FLOPS 每 2 年涨 3 倍，但 memory bandwidth 只涨 1.6 倍，interconnect bandwidth 涨 1.4 倍。这就是所谓的 **memory wall**。

举例：
- 2004 年 Earth Simulator：41 TFLOPS，10 TB DRAM
- 2024 年 El Capitan：1.74 exaFLOPS，5.4 PB HBM3
- FLOPS 涨了 42,400 倍
- Memory 只涨了 540 倍

gap 是 78 倍。

具体到 brain simulation：

每个 synapse 64 bytes（ID、type、receptor state），每个 neuron 64 bytes，加 2 倍 HPC overhead。人类大脑 ~6×10¹⁴ synapse：

(64 × 6×10¹⁴ + 64 × 8.6×10¹⁰) × 2 ≈ **70 PB**

分摊到 100,000 GPU 上，每个 GPU 需要 **700 GB**。现在顶级 GPU 是 80-192 GB HBM。差不到一个数量级。

Interconnect 呢？假设 10% 的 synapse 跨 GPU 边界（大部分连接是 local 的），6×10¹³ cross-partition synapse，每个 spike event 4 bytes，10 Hz firing rate：

6×10¹³ × 4 × 10 / 100,000 = **24 GB/s per GPU**

当前 GPU interconnect（NVLink、InfiniBand）是 1.8 TB/s。所以 interconnect bandwidth 在 naive partitioning 下不是 binding constraint。但 memory bandwidth 和 latency 可能是。

**直觉：raw compute 不是瓶颈，memory wall 才是。** 这跟现在 AI training 遇到的问题一模一样。

### 数据存储

1 立方毫米脑组织，10nm 分辨率，1 byte/voxel = 1 PB。人类大脑 1.4 million mm³ = **1.4 zettabyte** raw data。

用 state-of-the-art 的 VAE compression（EM-Compressor, Li et al. 2024），大概 128 倍压缩 → 11 PB。

CERN 的 data center 存了 1 exabyte，建造成本 $93M，年运营 $100-200M。brain emulation 的存储需求比 CERN 小 10-100 倍。$10-50M 就能搞定初始建设。

但最终的 simulation 需要的 data 不是 raw image，是高度 annotated 的 connectome graph，那比 raw data 小 orders of magnitude。

---

## Structure-to-function：核心科学问题

### Shiu et al. 2024 的惊人结果

这是论文里最让我震惊的 empirical result。

他们做了什么？

1. 拿到完整的果蝇 connectome（FlyWire 2024）
2. 用最简单的 LIF neuron（1907 年的 differential equation）
3. **几乎没有用 functional data 来 tune**
4. **没有 molecular information**（EM 看不到 receptor protein）
5. 没有 compartment，没有 biophysical detail

结果：**成功 reproduce 了果蝇的一些行为**（feeding、grooming）。

这说明什么？**最简单的 connectome + 最简单的 neuron model，已经能产生 limited but real behavior。** 这是一个 proof-of-concept，说明 structure 本身蕴含了大量 functional information。

当然，这能不能 scale 到 mammalian brain，是未经验证的核心问题。但至少在 insect level，structure-to-function 的 mapping 是 work 的。

### Differentiable simulation 的思路

作者提出一个 proposal：在 synaptic compartment 加 voltage-dependent terms，每个 term 代表一类 receptor。这些 terms 的 parameter 可以被 ExM 的 protein staining data 约束（比如你知道某个 synapse 有多少 NMDAR），然后在 functional data 上用 gradient descent fine-tune。

整个系统是 fully differentiable 的，因为所有 voltage-dependent terms 和 probability curve 都是 smooth 的。这种方法已经被验证过对 point-neuron（Stanojevic et al. 2024）和 biophysically detailed model（Deistler et al. 2024, Chen et al. 2022）都 work。

直觉上这跟 differentiable physics engine（如 Brax、MJX）是一个思路——把 simulation 写成 differentiable 的，然后 leverage 整个 deep learning ecosystem 的 optimization infrastructure。

---

## Benchmarking：怎么知道你的 emulation 是"对的"

### 混沌系统的问题

brain dynamics 是 chaotic 的。两个初始条件几乎相同的 simulation，会很快 diverge——butterfly effect。

所以 deterministic metric（比如逐 neuron 逐时间步比 MAE）在 long time horizon 上根本不 work。

作者打了一个很好的比方：**climate model 可以预测 El Niño，即使 weather model 无法预测超过两周的具体天气。** 同理，brain emulation 的目标应该是在 distributional level 上 match dynamics——attractor structure、typical brain state、response to perturbation——而不是逐时刻完美复现某个 trajectory。

### Stochastic distribution matching

比 exact trajectory，应该比较 activity distribution：

- 单个 neuron 的 firing rate distribution
- neuron population 的 activity distribution
- conditional on brain state（比如在做某个 task 时的 distribution）

可以用 Total Variation Distance、KL Divergence、Jensen-Shannon Divergence 这些 metric。Allen Institute 的 V1 model（Billeh et al. 2020）就用 KS test 来比较 simulated 和 recorded neuron 的 distribution。

### Behavioral metrics

终极验证是 simulated organism 能在 virtual environment 里表现出真实行为。C. elegans 能 chemotaxis、thermotaxis，斑马鱼能 optomotor response。

Zador et al. 2023 提的 "embodied Turing test" 就是这个思路：simulated nervous system 在 simulated body 里能不能 replicate 原始 organism 的 behavioral repertoire。

但作者也警告：**behavioral equivalence 不等于 mechanistic accuracy。** 一个 lookup table 也能 pass behavioral test 而不 represent 任何 neural mechanism。所以 behavioral benchmark 必须跟 neural-level metric 结合。

### Benchmark suite 的思路

跟 AI 一样，不应该依赖 single benchmark。应该有 suite：neural predictivity + behavioral alignment + anatomical correspondence + perturbation response + out-of-distribution generalization。

Brain-Score（Schrimpf et al. 2020）是早期 example，评估 ANN 对 primate visual system 的 neural response predictivity。

---

## 从 worm 到 human 的路径

作者画了一条很清晰的 stepping stone：

**C. elegans（302 neuron）** → connectome 完整，whole-body calcium imaging 有了，但 graded non-spiking dynamics 难搞，voltage imaging 难。OpenWorm 和 BAAIWorm 在做。Creamer et al. 2024 用 connectome-constrained linear model 在 optogenetic perturbation data 上达到 r=0.92 的 prediction correlation。

**果蝇（140K neuron）** → connectome 完整，neurotransmitter classification 有了，Shiu et al. 的 simulation proof-of-concept 有了。但 whole-brain single-neuron imaging 还是 borderline。

**斑马鱼幼虫（~100K neuron）** → connectome 在做，whole-brain calcium imaging 到了 70K neuron at 1 Hz，voltage imaging 到了 1/3 brain。vertebrate brain structure，有 complex behavior 和 learning。**这是最有可能成为第一个 end-to-end emulated animal 的 model organism。**

**小鼠（75M neuron）** → connectome 的 10-15 mm³ 项目在进行，1M neuron cortical imaging 有了，10M neuron simulation on Fugaku 有了。但 whole-brain single-neuron imaging 超出了光学 frontier。

**人类（86B neuron）** → 没有 connectome，没有 whole-brain single-neuron imaging，有一个 86B neuron 的 crude simulation（Lu et al. 2024，60-120 倍慢于 real-time）。需要 mega-project scale 的投入。

---

## Mega-project 的 cost 和 timeline

作者的 estimate：**$5-50B，10-25 年**。

对比一下：
- Human Genome Project：~$5B，13 年
- Manhattan Project：~$30B，3 年
- Apollo Program：~$257B，8 年

Human brain emulation 介于 HGP 和 Manhattan 之间，低于 Apollo。

而且，**ML 是最大的 cost reduction driver**。SmartEM 用 ML 调 microscope 的 dwell time，EM 采集时间降了 7 倍。Flood-filling networks 把 proofreading accuracy 提升 10 倍。SegCLR 做 neuropil mapping。如果 AI scaling law 的类比成立——simulation accuracy 作为 input data quality 的 power-law function——那每个 incremental 的 connectomics throughput、functional recording density、molecular profiling 的提升都会 yield predictable 的 simulation fidelity gain。

---

## 给你 build intuition 的几个 takeaway

**1. 三条独立指数曲线在交汇**

Connectomics cost 在降，functional imaging neuron count 在升，AI cluster compute 在涨。这三条曲线各自独立，但 brain emulation 需要它们同时到达某个阈值。现在它们正在交汇。

**2. 算力不是瓶颈，memory wall 才是**

Raw FLOP/s 够了（LIF 只要几个 GPU，HH 要一个 100K GPU cluster）。真正的瓶颈是 memory bandwidth 和 interconnect latency。这跟 AI training 遇到的问题一模一样。

**3. Structure-to-function 是核心科学问题**

Shiu et al. 的果蝇实验证明，最简单的 connectome + LIF 就能产生 limited behavior。但能不能 scale 到 mammal，是未经验证的。斑马鱼是验证这个 mapping 的最佳平台。

**4. 斑马鱼是 key organism**

它是最大的、whole-brain single-neuron imaging 在 near-term feasible 的 vertebrate model。如果你想押注"第一个被完整 emulate 的动物"，斑马鱼幼虫是最可能的选择。

**5. Barcoding + ExM + AI proofreading 可能打破 cost bottleneck**

PRISM 已经 show 8 倍 accuracy 提升。如果这条路径 continue，proofreading cost（当前 >90% 预算）可能 collapse，human connectome 的 $0.01/neuron 目标可能达到。

**6. Benchmarking 严重 underdeveloped**

AI 有 MMLU、HELM、Chatbot Arena、ARC。Brain emulation 几乎什么都没有。ZAPBench 是早期 example。需要 stochastic distribution matching + behavioral + perturbation-based 的 composite benchmark suite。

**7. AI infrastructure 是 brain emulation 的 unintended beneficiary**

xAI 建 100K H100 cluster 是为了训 Grok。但这套基础设施的 raw compute 恰好够 human-scale HH simulation。AI 公司在无意间为 brain emulation 铺路。

**8. 这不是会不会的问题，是什么时候、怎么做的问题**

论文的 closing tone 很明确：field 的 trajectory 在从 "whether" 转向 "when and how"。答案取决于技术进步 + sustained large-scale coordinated effort。历史上人类做这种 mega-project 的 track record 不差——HGP、Manhattan、Apollo 都做到了。

---

参考链接汇总：
- 论文 companion report: https://brainemulation.mxschons.com
- State of Brain Emulation Report 2025: https://arxiv.org/abs/2510.15745
- Sandberg & Bostrom 2008 roadmap: https://www.fhi.ox.ac.uk/brain-emulation-roadmap-report.pdf
- FlyWire connectome: https://doi.org/10.1038/s41586-024-07558-0
- ZAPBench: https://doi.org/10.48550/arXiv.2503.02618
- Shiu et al. Drosophila simulation: https://doi.org/10.1038/s41586-024-07763-9
- PRISM barcoding: https://doi.org/10.1101/2025.09.26.678648
- Brain-Score: https://brain-score.org
- Allen Institute V1 model: https://doi.org/10.1016/j.neuron.2020.01.040
- Differentiable biophysical simulation: https://doi.org/10.1101/2024.08.21.608979
- Light beads microscopy: https://doi.org/10.1016/j.neuron.2024.02.011
- e11 bio (barcoding): https://e11.bio
- Boyden Lab: https://boydenlab.org

如果你想 enter 这个 field，作者留了邮箱：axon@mit.edu 和 i@isaak.net。他说 "the community is welcoming, and the potential impact is extraordinary"。

---

# From Worm to Human: Scaling Brain Emulation 深度技术解析

这篇论文是 Isaak Freeman 在 MIT Boyden Lab 完成的硕士论文（2026年3月），supervisor 是 Edward Boyden。核心 thesis 很清晰：**brain emulation 正在从 speculative aspiration 转变为 engineering problem**，而且多个底层技术的 scaling curves 正在交汇，使得 human-scale brain emulation 在本世纪内 plausible。

让我按照论文的逻辑架构，逐层深入讲解。

---

## 1. 核心论点：三条 scaling curves 的交汇

论文构建的核心 intuition 是这样的：brain emulation 的可行性取决于三个独立但 interdependent 的技术 pillar 同时成熟：

| Pillar | 关键 metric | 当前 state | Human-scale 需求 |
|--------|-----------|-----------|-----------------|
| Structural imaging (connectomics) | neurons reconstructed, cost/neuron | Drosophila complete (139K neurons) | 8.6e10 neurons |
| Functional imaging | simultaneously recorded neurons | ~1M (mouse cortex, light beads) | 8.6e10 neurons whole-brain |
| Simulation compute | FLOP/s, memory, interconnect | ~10^20 FLOP/s clusters | 3.4 petaFLOP/s (LIF) to 10^20 FLOP/s (HH) |

关键 insight 是：**AI infrastructure buildout 正在无意间为 brain emulation 铺路**。xAI 的 Colossus cluster（100K H100 GPUs）的 raw compute 已经接近 pessimistic 假设下 human brain emulation 的需求。

参考链接：
- 论文主页: https://brainemulation.mxschons.com
- State of Brain Emulation Report 2025: https://arxiv.org/abs/2510.15745
- Sandberg & Bostrom 2008 roadmap: https://www.fhi.ox.ac.uk/brain-emulation-roadmap-report.pdf

---

## 2. Structural Imaging: Connectomics 的成本下降曲线

### 2.1 EM Connectomics 的历史 trajectory

Connectomics 的 progress 呈现出一个非常清晰的 scaling pattern：

```
1986: C. elegans, 302 neurons (White et al.)
  ↓ 32 years gap
2018: Drosophila full brain imaged, 120 neurons reconstructed (Zheng et al.)
2020: Drosophila central brain, 22,594 neurons (Scheffer et al.)
2024: Drosophila complete, 139,255 neurons, 54.5M synapses (Dorkenwald et al., FlyWire)
2024: Human cortical fragment, 1 mm³, 49,000 neurons, 150M synapses (Shapson-Coe et al.)
```

这里的关键数据是 **per-neuron reconstruction cost 的下降**：

| Organism | Year | Cost/neuron |
|----------|------|-------------|
| C. elegans | 1980s | ~$16,500 |
| Drosophila | 2025 | ~$214 |
| Zebrafish | 2025 | ~$100 |
| Mammalian | 2025 | $500-1,000 |

目标 trajectory：
- Mouse connectome (~$1B budget): 需要 $10/neuron
- Human connectome (~$1B budget): 需要 $0.01/neuron

这是一个 **5 orders of magnitude** 的 gap，但 proofreading 仍然占 >90% 的成本，而 AI-assisted proofreading 正在 rapid improvement。

### 2.2 Multi-beam EM 的 throughput scaling

这是一个非常关键的 engineering bottleneck。让我详细讲解 napkin math：

**基础参数**：
- Mouse brain volume: V_mouse = 500 mm³ = 5×10²⁰ nm³
- Resolution: 10 nm isotropic → voxel size = 10×10×10 = 10³ nm³
- Total voxels: N_voxels = 5×10²⁰ / 10³ = 5×10¹⁷

**Single microscope throughput**：
- Current rate: ~1.5×10⁸ voxels/s (midpoint of 100-200M range)
- Time for single scope: T = 5×10¹⁷ / 1.5×10⁸ = 3.3×10⁹ s ≈ 104 years

**Multi-beam scaling**：
- 2015 Zeiss 61-beam: ~1 GHz peak (10⁹ voxels/s)
- 2019 extrapolation to 331 beams: 1 GHz × 331/61 ≈ 5.4 GHz
- Future 1000-beam (expert estimate): ~10 GHz plausible

**Human brain imaging calculation**：

$$T_{human} = \frac{V_{human}}{v_{size} \times r_{image} \times u \times N_{scopes} \times t_{years}}$$

其中：
- V_human = 1.4×10²⁴ nm³ (human brain volume)
- v_size = 16³ nm³ (using 16nm as minimum traceable voxel)
- r_image = 10¹⁰ voxels/s (10 GHz, 1000-beam scope)
- u = 0.5 (50% uptime, accounting for maintenance, alignment, stage movement)
- N_scopes = 200
- t_years = 10

代入：

$$T_{human} = \frac{1.4 \times 10^{24}}{16^3 \times 10^{10} \times 0.5 \times 200 \times 10 \times 365.25 \times 24 \times 3600} \approx 1$$

所以 **200台 10-GHz mass-produced SEMs, 10年内可以 image 一个 human brain**。Cost estimate: 200 × $1M = $200M for microscopes alone。

参考：
- FlyWire connectome: https://doi.org/10.1038/s41586-024-07558-0
- Wellcome Trust connectomics report: https://wellcome.org/reports/scaling-connectomics
- Princeton mouse connectome project: https://pni.princeton.edu/

### 2.3 Expansion Microscopy (ExM) 作为 alternative path

ExM 的核心 idea 非常 elegant：不提高 microscope 的分辨率，而是**物理放大 sample**。

**Resolution transformation**：
- Conventional light microscope lateral resolution: ~250 nm (diffraction limit)
- ExM 4x expansion: effective resolution = 250/4 ≈ 62.5 nm
- ExM 10x expansion: effective resolution = 250/10 = 25 nm
- ExM 20x expansion (in development): effective resolution = 250/20 = 12.5 nm

**Connectomic resolution requirement**：
- Synapse width: 200-800 nm
- Synaptic cleft: 20-30 nm
- Unmyelinated axons: 50-1,000 nm
- Fine spine necks: 10-30 nm (需要可靠检测)

所以 ExM 需要 ~20-24x expansion 才能 reach connectomic regime。

**umExM (ultrastructural membrane ExM)** 的 accuracy：
- Myelinated axons: Rand score 0.995 ± 0.004
- Unmyelinated axons: Rand score 0.993 ± 0.006
- Achieved at 4x expansion, effective ~60 nm resolution

ExM 的巨大优势是 **兼容 molecular annotation**：可以同时 stain proteins (ion channels, receptors, neuropeptides)，这对 simulation 的 parameterization 至关重要。

参考：
- Original ExM paper (Chen et al. 2015): https://doi.org/10.1126/science.1260088
- umExM (Shin et al. 2024): https://doi.org/10.1101/2024.03.07.583776
- ExA-SPIM (Glaser et al. 2024): https://doi.org/10.7554/eLife.91979

### 2.4 Protein Barcoding: Combinatorics 的威力

这是论文中最 mathematically elegant 的部分之一。核心 idea：如果每个 neuron 都有一个独特的 molecular barcode，就不需要 error-free tracing。

**Binary barcoding combinatorics**：

假设 25 个 randomly expressed proteins，每个 protein 的 presence/absence 是 binary signal：

**Ideal case (p=0.5)**：
$$N_{barcodes}^{ideal} = 2^{25} = 33,554,432 \approx 33M$$

**Realistic case (p=0.3, typical AAV expression)**：

这里需要用 Shannon entropy 来计算 effective barcode space。每个 protein 是一个 Bernoulli random variable with probability p。整个 barcode system 的 entropy 是：

$$H(p) = -p \log_2(p) - (1-p) \log_2(1-p)$$

其中：
- p = 单个 protein 的 expression probability
- (1-p) = 不表达的概率
- log₂(p), log₂(1-p) = 各自的信息量（以 bits 为单位）
- 负号确保 entropy 非负

Effective barcode space：

$$N_{barcodes}^{effective} = 2^{n \cdot H(p)}$$

其中 n = 25 (protein 数量)。

代入 p = 0.3：

$$H(0.3) = -0.3 \log_2(0.3) - 0.7 \log_2(0.7)$$
$$= -0.3 \times (-1.737) - 0.7 \times (-0.515)$$
$$= 0.521 + 0.360 = 0.881 \text{ bits}$$

$$N_{barcodes}^{effective} = 2^{25 \times 0.881} = 2^{22.0} \approx 4,300,000$$

**关键 comparison**：
- Human brain: 8×10¹⁰ neurons
- Average synapses per neuron: 8,000-40,000
- Effective barcodes: 4.3M

所以 **4.3M barcodes 相对于 single neuron 的 synaptic neighborhood (>100x margin)**。Barcoding 不需要 globally unique，只需要 locally unique（在一个 neuron 的 synaptic neighborhood 内 unique）。

**Binomial coefficient 验证**：

对于恰好 k = round(25 × 0.3) = 8 个 proteins 表达的情况：

$$C(n, k) = \binom{25}{8} = \frac{25!}{8! \times 17!} = 1,081,575$$

这大约是 theoretical 2²⁵ 的 3%，与 Shannon entropy 估计一致。

**PRISM (Park et al. 2025)** 是这个方向的 breakthrough：
- Binary protein barcoding + ExM + self-proofreading AI segmentation
- Accuracy 8x higher than conventional single-color methods
- 在 ~10M μm³ mouse hippocampus volume 中实现 automatic tracing + synapse molecular mapping

参考：
- PRISM (Park et al. 2025): https://doi.org/10.1101/2025.09.26.678648
- Bitbow (Li et al. 2021): https://doi.org/10.3389/fncir.2021.732183
- e11 bio barcoding: https://e11.bio

### 2.5 Protein Sequencing: 最高 upside 的 wild card

当前的 protein staining 受限于 imaging rounds：30,000 proteins × 10-fold multiplexing = 3,000 rounds。这是 years of imaging time。

**In-situ protein sequencing** 的 vision：

基于 Edman degradation (1950) 的原理，cyclically cleave N-terminal amino acid 并 identify。结合 ExM：

1. Anchor all proteins to hydrogel
2. Expand sample (10-20x)
3. Cyclically:
   a. Cleave N-terminal amino acid (Edman chemistry)
   b. Identify cleaved amino acid via binder/mass spec
   c. Repeat for sequence readout

**Challenges**：
- Binder specificity: antibodies typically need ~5 amino acid epitopes，single amino acid identification 很难
- Decrowding: proteins ~2-10 nm, packed at ~11 nm spacing at synapses。10x ExM gives 30 nm effective resolution，insufficient for decrowding
- 需要更高 expansion factors 或 alternative chemistry

如果突破，这会是一次 **orders of magnitude** 的 leap，类似于 genome sequencing 的 super-exponential cost decline。

参考：
- Edman degradation original (Edman 1950): Acta Chemica Scandinavica, 4, 283-293
- Nanopore protein sequencing (Motone et al. 2024): https://doi.org/10.1038/s41586-024-07935-7
- Protein shape imaging (Shaib et al. 2024): https://doi.org/10.1038/s41587-024-02431-9

---

## 3. Functional Imaging: The Optical Frontier

### 3.1 Recording neuron count 的 exponential growth

Stevenson & Kording (2011) 发现 simultaneously recorded neurons 大约每 7.4 年翻倍。但 optical imaging 加速了这个 trend。

当前 state-of-the-art：

| Method | Organism | Neurons | Hz | Reference |
|--------|----------|---------|-----|-----------|
| SCAPE microscopy | C. elegans | whole-body | 25.75 | Voleti et al. 2019 |
| Light-sheet calcium | Zebrafish larva | 70,000 | 1 | ZAPBench (Lueckmann et al. 2025) |
| Voltage imaging | Zebrafish larva | ~1/3 of brain | - | Wang et al. 2023 |
| Two-photon calcium | Drosophila adult | near-whole-brain | 1.95 | Mann et al. 2017 |
| Light beads microscopy | Mouse cortex | 1,000,000 | 2 | Manley et al. 2024 |

### 3.2 The "Glass Ceiling" concept

论文提出了一个关键 conceptual framework：**whole-brain single-neuron imaging 存在一个 optical frontier**。

```
Penetration depth limit: ~1-2 mm (scattering & absorption)

✓ Below frontier (transparent, small):
  - C. elegans (~302 neurons, naturally transparent)
  - Zebrafish larva (~100K neurons, transparent)
  - Potentially: Xenopus tadpole, Hydra, planarian

⚠️ Borderline:
  - Drosophila adult (~140K neurons, reduced transparency)

✗ Above frontier (opaque, large):
  - Mouse (7.5e7 neurons, skull + scattering)
  - Human (8.6e10 neurons, skull + thick tissue)
```

**Zebrafish larva 是 key stepping stone**，因为：
1. Vertebrate brain structure (mammal-like)
2. Whole-brain single-neuron imaging feasible
3. Complex behaviours, learning, plasticity
4. Connectome in progress (Lueckmann et al. 2025)

**Tissue clearing 的 hope**：tartrazine 等 absorbing molecules 可以改善 in vivo transparency (Ou et al. 2024)，暗示 deeper imaging without cranial windows 在某些 regime 下 physically plausible。

参考：
- ZAPBench: https://doi.org/10.48550/arXiv.2503.02618
- Light beads microscopy (Manley et al. 2024): https://doi.org/10.1016/j.neuron.2024.02.011
- Tartrazine transparency (Ou et al. 2024): https://doi.org/10.1126/science.adm6869
- Crystal skull windows (Kim et al. 2016): https://doi.org/10.1016/j.celrep.2016.12.004

---

## 4. Simulation: Structure-to-Function 的核心挑战

### 4.1 Neuron model complexity spectrum

论文讨论了从极简到极复杂的 neuron model spectrum：

**McCulloch-Pitts (1943)**：binary unit，无 dynamics

**Leaky Integrate-and-Fire (LIF)**：
$$\tau_m \frac{dV}{dt} = -V + R_m I_{syn}(t)$$

其中：
- τ_m = membrane time constant (ms)
- V = membrane voltage (mV)
- R_m = membrane resistance (MΩ)
- I_syn(t) = synaptic input current (nA)
- 当 V 达到 threshold V_th → spike, reset to V_reset

**Izhikevich (2003)**：
$$\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I$$
$$\frac{du}{dt} = a(bv - u)$$

其中：
- v = membrane potential
- u = recovery variable
- a, b = parameters controlling time scale and sensitivity
- if v ≥ 30: v ← c, u ← u + d

**Hodgkin-Huxley (1952)** — full biophysical：
$$C_m \frac{dV}{dt} = -g_{Na} m^3 h (V - E_{Na}) - g_K n^4 (V - E_K) - g_L (V - E_L) + I_{ext}$$

其中：
- C_m = membrane capacitance (μF/cm²)
- g_Na, g_K, g_L = maximal conductances for sodium, potassium, leak (mS/cm²)
- m, h, n = gating variables (0-1), each governed by own ODE
- E_Na, E_K, E_L = reversal potentials (mV)
- I_ext = external input current (μA/cm²)

**Multicompartmental HH**：每个 neuron 有 ~1,000 compartments，每个 compartment 有独立的 HH dynamics + 7,000 synapses。

### 4.2 Compute Fermi Estimates

这是论文中最关键的 quantitative contribution 之一。

**Lower bound (LIF neurons)**：

假设：
- ~40 FLOP per neuron per millisecond (Izhikevich 2004, Brette et al. 2007)
- N = 8.6×10¹⁰ neurons (human brain)
- Real-time simulation: 10³ ms/s

$$FLOP/s_{LIF} = 40 \times 8.6 \times 10^{10} \times 10^3 = 3.4 \times 10^{15} \text{ FLOP/s} = 3.4 \text{ petaFLOP/s}$$

这大约是 **单个 H100 GPU** 的 tensor throughput（~1 petaFLOP/s dense FP16, ~2 petaFLOP/s with structured sparsity）。

**Upper bound (multicompartmental HH)**：

假设：
- 1,000 compartments per neuron
- 690 FLOP per compartment per ms (Euler-exponential solver, Hines & Carnevale 2001)
- 7,000 synapses per neuron
- ~10⁴ operations per spike at 10 Hz average firing

Per neuron per second：

$$FLOP_{neuron} = 1000 \times 690 \times 10^3 + 7000 \times 10 \times 10^4$$
$$= 6.9 \times 10^8 + 7 \times 10^8 = 1.39 \times 10^9 \text{ FLOP/s/neuron}$$

Total：

$$FLOP/s_{HH} = 1.39 \times 10^9 \times 8.6 \times 10^{10} \approx 1.2 \times 10^{20} \text{ FLOP/s}$$

论文给出 ~10²⁰ FLOP/s，与 xAI Colossus cluster (~10²⁰ FLOP/s) 相当。

**关键 insight**：synaptic modeling 是 compute cost 的 dominant factor，不是 neuronal dynamics。LIF vs HH 的 per-neuron gap 相对于 synaptic detail costs 来说较小。

### 4.3 Memory & Interconnect: The Memory Wall

这是论文识别出的 **真正的 bottleneck**。

**Memory requirements napkin math**：

假设：
- 64 bytes per synapse (IDs, type, receptor states)
- 64 bytes per neuron
- 2x overhead for HPC bookkeeping
- Human brain: ~6×10¹⁴ synapses, 8.6×10¹⁰ neurons

$$Memory_{total} = (64 \times 6 \times 10^{14} + 64 \times 8.6 \times 10^{10}) \times 2$$
$$\approx 7.7 \times 10^{16} \text{ bytes} \approx 70 \text{ PB}$$

Distributed across 100,000 GPUs：

$$Memory_{per\_GPU} = \frac{70 \text{ PB}}{100,000} = 700 \text{ GB/GPU}$$

Current top-tier GPUs: 80-192 GB HBM。所以这个需求在 **<1 order of magnitude** 之内，tight but tractable。

**Interconnect bandwidth**：

假设：
- 10% of synapses cross GPU boundaries (大部分连接是 local)
- Cross-partition synapses: 6×10¹³
- 4 bytes per spike event (timing + neuron ID)
- 10 Hz firing rate

$$BW_{per\_GPU} = \frac{6 \times 10^{13} \times 4 \times 10}{100,000} = 24 \text{ GB/s per GPU}$$

Current GPU interconnect (NVLink, InfiniBand): 1.8 TB/s。所以 interconnect bandwidth 在 naive partitioning 下 **不是** binding constraint，但 memory bandwidth 和 latency 可能是。

**The Memory Wall trend**：

| Metric | 30-year improvement rate |
|--------|------------------------|
| Peak FLOPS | 3x every 2 years |
| Memory bandwidth | 1.6x every 2 years |
| Interconnect bandwidth | 1.4x every 2 years |

对比实例：
- 2004 Earth Simulator: 41 TFLOPS, 10 TB DRAM
- 2024 El Capitan: 1.74 exaFLOPS, 5.4 PB HBM3
- FLOPS 增长: 42,400x
- Memory 增长: 540x

这是一个 **78x 的 gap**，意味着 future brain simulations 将 increasingly bottlenecked by memory/interconnect，不是 raw compute。

参考：
- NVIDIA H100 whitepaper: https://resources.nvidia.com/en-us-tensor-core
- Memory wall analysis (Gholami et al. 2024): https://doi.org/10.1109/MM.2024.3373763
- GPU memory wall survey (An et al. 2024): https://arxiv.org/abs/2408.14158
- xAI Colossus cluster: https://x.com/elonmusk/status/1831089398592344424

### 4.4 Data Storage

**Raw imaging data**：
- 1 mm³ brain at 10 nm isotropic = 10¹⁵ voxels
- 1 byte/voxel → 1 PB/mm³
- Human brain: 1.4×10⁶ mm³ → **1.4 zettabytes** raw

**Compressed**：
- State-of-the-art compression: ~128x reduction (Li et al. 2024, EM-Compressor with VAEs)
- Compressed human brain: ~11 PB
- Cost at 2023 prices: ~$200M (Collins et al. 2025 estimate ~$2B for raw, 10x compression → $200M)

**Final connectome representation**：
高度 annotated connectome graph，orders of magnitude more storage-efficient than raw imaging。实际 simulation 需要的 data 是 manageable 的。

对比：CERN data center 存储超过 1 exabyte，throughput 100-200 GB/s，建造成本 $93M USD，年运营 $100-200M。Brain emulation 的存储需求 (10-100 PB compressed) 比 CERN 小 10-100x。

参考：
- EM-Compressor (Li et al. 2024): https://doi.org/10.1101/2024.07.07.601368
- CERN Data Centre: https://home.web.cern.ch/science/computing/data-centre
- Collins et al. 2025 imaging prospects: https://doi.org/10.1016/j.crmeth.2025.100988

### 4.5 Structure-to-Function: The Central Gap

论文的核心 thesis 之一是：**connectome 是 static snapshot，如何从 structure 推导 function 是关键 challenge**。

**Shiu et al. (2024) 的 Drosophila simulation** 是最重要的 proof-of-concept：

- 基于 complete fly connectome (Dorkenwald et al. 2024)
- 使用 LIF neurons（century-old differential equations!）
- **几乎没用 functional data** tuning
- **没有 molecular information**（EM 无法 capture membrane proteins）
- 没有 compartments，没有 biophysical detail
- 结果：**accurately reproduced some fly behaviours** (feeding, grooming)

这是一个极其 impressive 的 empirical result。最简单的 connectome + 最简单的 neuron model = limited but real behavior。

**Extended differential equations approach**：

论文提出一个 proposal：在 synaptic compartments 添加 voltage-dependent terms，每个 term 代表一类 receptor：

$$I_{syn} = \sum_{r \in \text{receptors}} g_r(V, t) \times (V - E_r)$$

其中：
- g_r(V, t) = receptor r 的 conductance，voltage- and time-dependent
- E_r = receptor r 的 reversal potential
- receptor types: AMPAR, NMDAR, GABAR, etc.

这些 g_r 的 parameters 可以被 ExM protein staining data 约束（receptor 密度 → parameter bounds），然后通过 **gradient descent** 在 functional data 上 fine-tune。

这是一个 **fully differentiable system**，因为所有 voltage-dependent terms 和 probability curves 都是 smooth and continuous。这种方法已被证明对 point-neuron (Stanojevic et al. 2024) 和 biophysically detailed models (Chen et al. 2022, Deistler et al. 2024) 都可行。

参考：
- Drosophila simulation (Shiu et al. 2024): https://doi.org/10.1038/s41586-024-07763-9
- Connectome-constrained fly visual system (Lappalainen et al. 2024): https://doi.org/10.1038/s41586-024-07939-3
- Differentiable biophysical simulation (Deistler et al. 2024): https://doi.org/10.1101/2024.08.21.608979
- Allen Institute V1 model (Billeh et al. 2020): https://doi.org/10.1016/j.neuron.2020.01.040

---

## 5. Benchmarking: 如何定义 "success"

### 5.1 The Stochastic Problem

论文非常 insightfully 指出：**deterministic metrics (如 MAE) 对 chaotic systems 不适用**。

Brain dynamics 是 chaotic + stochastic 的（thermal noise affecting ion channels, Faisal et al. 2008）。两个初始条件相似的 simulation 会 diverge（butterfly effect, Lorenz 1963）。

**类比**：climate models 可以预测 El Niño，即使 weather models 无法预测超过几周的 exact weather。同样，brain emulation 的目标应该是 **accurately model underlying dynamics (attractors, memory states)**，不是完美 predict 某个 trajectory。

### 5.2 Stochastic Distribution Matching

不是比较 exact activity trajectories，而是比较 activity distributions：

**Per-neuron distribution metrics**：
- Total Variation Distance: TVD(P, Q) = ½ Σ|P(x) - Q(x)|
- KL Divergence: D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))
- Jensen-Shannon Divergence: JSD(P, Q) = ½ D_KL(P||M) + ½ D_KL(Q||M), where M = ½(P+Q)

其中 P = biological neuron activity distribution, Q = simulated neuron activity distribution。

**Conditional on brain state**：neuronal activity 跨 brain states 变化很大，所以 distributional metrics 应该 **conditioned on brain state**（如 task being performed）。

### 5.3 Benchmark Suites

论文 advocates for **benchmark suites** 而不是 single benchmarks，类似 AI practice (MMLU, HELM, Chatbot Arena)。

**Brain-Score** (Schrimpf et al. 2020) 是一个早期 example：评估 ANN 对 primate visual system neural responses 的 predictivity，across multiple benchmarks simultaneously。

**"Successful mouse brain emulation" 的 concrete benchmarks**：
- Behavioral: maze navigation, fear conditioning, social interaction, circadian rhythms
- Neural: hippocampal place fields, V1 orientation tuning, sleep replay patterns
- Personality matching: individual simulated mice from different connectomes 是否 show behavioral variability？
- Novel predictions: simulation 预测特定 perturbation 的 behavioral deficit → experimental confirmation

参考：
- ZAPBench benchmark: https://doi.org/10.48550/arXiv.2503.02618
- Brain-Score: https://brain-score.org
- Embodied Turing Test (Zador et al. 2023): https://doi.org/10.1038/s41467-023-37180-x
- C. elegans I/O characterization call (Haspel et al. 2023): https://arxiv.org/abs/2308.06578

---

## 6. Model Organism Trajectory: From Worm to Human

论文构建了一个非常清晰的 **stepping stone strategy**：

```
C. elegans (302 neurons)
  ✓ Connectome complete (1986, improved 2019)
  ✓ Whole-body calcium imaging (Schrödel 2013, Voleti 2019)
  ⚠️ Voltage imaging difficult
  ⚠️ Graded, non-spiking dynamics
  → Current: OpenWorm, BAAIWorm, Creamer et al. (r=0.92 prediction)

Drosophila (140K neurons)
  ✓ Connectome complete (2024, FlyWire)
  ✓ Neurotransmitter classification (Eckstein et al. 2024)
  ✓ Simulation proof-of-concept (Shiu et al. 2024)
  ⚠️ Whole-brain single-neuron imaging borderline
  → Next: combine connectome + functional + molecular

Zebrafish larva (~100K neurons)
  ⚠️ Connectome in progress
  ✓ Whole-brain calcium imaging (70K neurons, 1 Hz, ZAPBench)
  ✓ Voltage imaging (~1/3 brain, Wang et al. 2023)
  ✓ Vertebrate brain structure
  → Key validation platform for end-to-end emulation

Mouse (7.5e7 neurons)
  ⚠️ Connectome: 10-15 mm³ projects underway
  ✓ 1M neuron cortical imaging (light beads microscopy)
  ✓ 10M neuron simulation on Fugaku (Kuriyama et al. 2025)
  ⚠️ Whole-brain single-neuron imaging beyond optical frontier
  → Requires tissue clearing or alternative approaches

Human (8.6e10 neurons)
  ✗ No connectome
  ✗ No whole-brain single-neuron imaging
  ✓ 86B neuron simulation (Lu et al. 2024, crude)
  → Mega-project scale ($5-50B, 10-25 years)
```

**C. elegans 的特殊挑战**：虽然只有 302 neurons，但 far from simple。Constrained by limited interneuronal communication capacity，worm brain 被迫 maximize computational processing within each neuron。Signaling proteins 占 over 20% of worm genome (Sterling & Laughlin 2015)。这意味着 single neuron modeling 需要 high detail，不能简单用 LIF。

**Zebrafish 作为关键 validation platform**：它是 largest organism where near-whole-brain single-neuron functional recording appears plausible with near-term technology。Combined with ongoing connectomics efforts，它可能是第一个 **end-to-end emulated animal** with structural + functional + behavioural validation。

参考：
- C. elegans connectome (Cook et al. 2019): https://doi.org/10.1038/s41586-019-1352-7
- C. elegans whole-body calcium (Nguyen et al. 2016): https://doi.org/10.1073/pnas.1507110112
- C. elegans neural signal propagation atlas (Randi et al. 2023): https://doi.org/10.1038/s41586-023-06683-4
- BAAIWorm (Zhao et al. 2024): https://doi.org/10.1038/s43588-024-00738-w
- Mouse cortex simulation on Fugaku (Kuriyama et al. 2025): https://doi.org/10.1145/3712285.3759819
- Human-scale simulation (Lu et al. 2024): https://doi.org/10.1038/s43588-024-00731-3

---

## 7. The Mega-Project: Cost & Timeline

### 7.1 Historical megaproject comparison

| Project | Cost (inflation-adjusted) | Duration | Key outcome |
|---------|--------------------------|----------|-------------|
| Human Genome Project | ~$5B | 13 years | Human genome sequenced |
| Manhattan Project | ~$30B | 3 years | Atomic bomb |
| Apollo Program | ~$257B | 8 years | Moon landing |

论文估计 human-scale connectomics + simulation 的范围：**$5-50B over 10-25 years**，介于 HGP 和 Manhattan Project 之间，低于 Apollo。

### 7.2 ML as cost reduction driver

**SmartEM** (Meirovitch et al. 2024)：ML-guided dwell time adjustment → 7x decrease in EM acquisition time。

**Flood-filling networks** (Januszewski et al. 2018)：automated segmentation accuracy 10x improvement vs prior methods。

**SegCLR** (Mu et al. 2023)：segmentation-guided contrastive learning for neuropil mapping。

**Connectome-constrained neural networks** (Lappalainen et al. 2024)：直接从 wiring diagram 预测 neural activity，可能部分 substitute functional recording data。

**AI scaling laws analogy**：如果 simulation accuracy 类似地 scales as power-law of compute/data/parameters (Kaplan et al. 2020)，那么 connectomics throughput、functional recording density、molecular profiling 的每个 incremental improvement 都会 yield predictable gains in simulation fidelity。

参考：
- SmartEM (Meirovitch et al. 2024): https://doi.org/10.1101/2023.10.05.561103
- Flood-filling networks (Januszewski et al. 2018): https://doi.org/10.1038/s41592-018-0049-4
- SegCLR (Mu et al. 2023): https://doi.org/10.1038/s41592-023-02059-8
- AI scaling laws (Kaplan et al. 2020): https://arxiv.org/abs/2001.08361
- Foundation models for connectomics (Januszewski & Jain 2024): https://doi.org/10.1101/2024.11.24.625067

---

## 8. Key Insights for Building Intuition

让我总结几个 core intuitions：

### 8.1 Three independent exponential curves converging

Connectomics cost decline + functional imaging neuron count growth + AI cluster compute growth = 三条独立指数曲线正在同时逼近 human-scale brain emulation 的需求阈值。这个 convergence 是历史性的。

### 8.2 The structure-to-function gap is the central scientific challenge

我们有 Drosophila connectome，有 LIF model，能 reproduce some behaviors (Shiu et al. 2024)。但 **从 static structure 到 dynamic function 的 mapping** 仍然是核心未解问题。Zebrafish 是验证这个 mapping 的最佳平台。

### 8.3 Memory wall, not compute, is the binding constraint

Raw FLOP/s 已经 sufficient（3.4 petaFLOP/s for LIF, 单个 H100 就够）。Memory bandwidth、interconnect latency、per-GPU memory capacity 是真正的 bottlenecks。这个 trend 会 worsen（FLOPS 3x/2yr vs memory 1.6x/2yr）。

### 8.4 Barcoding + ExM + AI proofreading = potential path to $0.01/neuron

PRISM 已经 show 8x accuracy improvement。如果 barcoding combinatorics (2^22 effective codes) + protein staining + self-proofreading AI 继续 improve，proofreading cost（当前 >90% of budget）可能 collapse。

### 8.5 Model organism staging is essential

Human brain emulation 不能 jump-start。需要 C. elegans → Drosophila → zebrafish → mouse → human 的 sequential validation，每个 stage 验证 structure-to-function mapping 的不同 aspect。

### 8.6 Benchmarking is severely underdeveloped

与 AI 的 rich benchmark ecosystem (MMLU, HELM, ARC, Chatbot Arena) 相比，brain emulation 的 benchmarking 几乎不存在。ZAPBench 是早期 example。需要 stochastic distribution matching + behavioral metrics + perturbation-based causality metrics 的 composite suites。

---

## 9. 个人补充联想

这篇论文让我联想到几个 broader patterns：

**1. Analogies to sequencing cost decline**：

Genome sequencing cost 从 2001 年的 ~$100M/genome 降到 2024 年的 ~$200/genome，super-exponential decline。Connectomics 正在 early stage of similar curve。Key driver：automation + ML + economies of scale。如果 connectomics cost decline 类似 sequencing，human connectome 的 $1B budget 可能在 2030s 就 feasible。

**2. Neural scaling laws 的潜在类比**：

Kaplan et al. 2020 发现 LM performance scales as power-law of compute/data/parameters。如果 brain simulation accuracy 也遵循类似 scaling law（with connectome resolution, functional data density, molecular annotation completeness 作为 "data"），那么我们可以 predict：每 doubling of input data quality → predictable gain in simulation fidelity。这需要 empirical validation，但如果成立，会极大地 de-risk the mega-project。

**3. Differentiable physics engines 的类比**：

Deistler et al. 2024 的 differentiable biophysical simulation 类似于 differentiable physics engines in robotics（如 Brax, MJX）。这允许 gradient-based optimization of neuron/synapse parameters against functional data。如果 combined with modern automatic differentiation frameworks (JAX, PyTorch)，可以 leverage 整个 deep learning ecosystem 的 tooling。

**4. Neuromorphic computing 的 potential role**：

论文 briefly 提到 neuromorphic systems（Intel Loihi, Hala Point with 1.15B neurons）。Neuromorphic 的 event-driven, asynchronous computation 天然适合 sparse, event-based neural simulation，可以 simultaneously address compute, memory, interconnect, energy bottlenecks。但当前 investment 远小于 AI accelerators，so it remains a promising but underfunded path。

**5. Cryopreservation + emulation 的 potential pathway**：

论文 mentions compatibility with cryopreserved tissue。如果 high-resolution structural preservation 可以在 cryopreserved brains 中 achieve，这开辟了一条 pathway：cryopreserve → wait for technology → vitrify → image → emulate。这对于 personal continuity 是一个 plausible（though highly speculative）longevity pathway。

**6. Consciousness research 的 empirical traction**：

如果 digital neural systems 可以在 varying levels of fidelity 运行，这为 consciousness 研究 (Tononi's IIT, integrated information theory) 提供了 novel experimental evidence。一个 digital system 是否 reproduce functional signatures of conscious states？这是一个 fundamentally empirical question that brain emulation 可以 address。

参考：
- Sequencing cost data (Wetterstrand): https://www.genome.gov/about-genomics/fact-sheets/DNA-Sequencing-Costs-Data
- Neural scaling laws (Kaplan et al. 2020): https://arxiv.org/abs/2001.08361
- Differentiable biophysical simulation (Deistler et al. 2024): https://doi.org/10.1101/2024.08.21.608979
- Intel Hala Point neuromorphic system: https://www.intel.com/content/www/us/en/newsroom/news/intel-builds-worlds-largest-neuromorphic-system.html
- IIT (Tononi 2004): https://doi.org/10.1186/1471-2202-5-42
- Sandberg & Bostrom 2008 roadmap: https://www.fhi.ox.ac.uk/brain-emulation-roadmap-report.pdf

---

## 10. Open Questions & Critical Gaps

论文坦诚地承认了多个 gaps：

1. **Molecular data**：connectome capture wiring，但 receptor types, ion channel densities, neuromodulator concentrations, gene expression profiles 仍然缺失。Spatial transcriptomics, in-situ sequencing, multiplexed protein imaging 正在 fill this gap for small volumes，但 scaling to whole brains 是 open challenge。

2. **Glia and astrocytes**：论文 focus on neurons，但 glial cells 占 brain cell count 的 ~50%，对 neural function 有重要 modulatory role。

3. **Plasticity and learning**：current simulations 主要是 feed-forward static models。Real brains 有 short-term depression, LTP, LTD, structural plasticity。如何 incorporate 这些 into emulation 是 major open question。

4. **Developmental data**：connectome 是 single timepoint snapshot。Real brain wiring 是 developmental process。Witvliet et al. 2021 的 C. elegans developmental connectomes 是 early effort，但 mammalian developmental connectomics 几乎不存在。

5. **Ethical/philosophical dimensions**：论文 explicitly excludes these，但 digital mind creation 的 implications 是 enormous。如果 high-fidelity emulation preserve causal structure underlying personal identity，这 raises questions about personhood, rights, continuity of consciousness。

6. **Non-connectomics approaches**：论文 focus on structure-dominant paradigm，但 function-to-function prediction（直接从 functional data 预测，bypass structural reconstruction）是 plausible alternative。Beiran & Litwin-Kumar 2024 show limited recording data may be sufficient to predict whole-brain activity。

参考：
- C. elegans developmental connectomes (Witvliet et al. 2021): https://doi.org/10.1038/s41586-021-03778-8
- Function-to-function prediction (Beiran & Litwin-Kumar 2024): https://doi.org/10.1101/2024.02.22.581667
- Neuropeptidergic connectome of C. elegans (Ripoll-Sanchez et al. 2023): https://doi.org/10.1016/j.neuron.2023.09.010
- Neurotransmitter atlas of C. elegans (Wang et al. 2024): https://doi.org/10.7554/eLife.95402.3

---

这篇论文的价值在于它 **translates cross-domain progress into common units**，使得 brain emulation 的 interdependencies 变得 visible。通过 Fermi estimates、cost trajectories、和 technology roadmaps，它把一个看似 speculative 的 aspiration 转化为 identifiable engineering bottlenecks。对于想要 enter 这个 field 的人来说，这 thesis 是一个极其 valuable 的 map of the terrain。
