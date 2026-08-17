---
source_pdf: Parallelized Autoregressive Visual Generation.pdf
paper_sha256: 8f24615ecd3070095274daf17b20c411745feaf4cd3ec3aeb294b5544bc600a7
processed_at: '2026-08-06T02:15:50-07:00'
target_folder: LLM-engine/Torch
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲PAR这篇paper

## 一句话总结

**Image里离得远的tokens互相不太care，可以同时生成；离得近的tokens互相紧密依赖，必须排队生成。利用这个property，把576步的sequential generation压缩到147步，速度快3.6倍，质量几乎不降。**

---

## 先说痛点在哪

LlamaGen这类AR visual model生成一张256×256的image要走576步forward pass，每步只吐一个token。A100上3B model要12秒一张image，这速度基本没法用。

你可能会想：那每步多吐几个token不就行了？比如每步吐4个token，576步变144步。

**这个naive想法直接死掉**。作者试了，FID从2.62暴涨到5.64，生成的老虎脸扭曲、斑马条纹断裂。

为什么死掉？因为autoregressive sampling有个根本要求：**每个token采样时必须知道之前所有token的实际取值**。这是conditional distribution的正确性要求。

当你parallel生成相邻的4个tokens $v_1, v_2, v_3, v_4$ 时，你实际上是各自独立地从marginal distribution采样：

$$v_i \sim \mathbb{P}(v_i | \text{context}), \quad i \in \{1,2,3,4\}$$

但正确的joint distribution应该是：

$$\mathbb{P}(v_1, v_2, v_3, v_4 | \text{context})$$

如果这4个tokens之间有强dependency，joint distribution是factorize不开的。你各自从marginal采样，相当于假设了：

$$\mathbb{P}(v_1, v_2, v_3, v_4 | \text{context}) \approx \prod_{i=1}^{4} \mathbb{P}(v_i | \text{context})$$

这个approximation对adjacent tokens来说是灾难性的。斑马的条纹方向、宽度、间距必须continuation，你各自marginal采样，每个token都"自由发挥"，条纹就乱了。

---

## Key Insight：Visual data有spatial locality

Image有个天然property：**spatial locality**。左上角的sky token和右下角的grass token，给定global context后，互相几乎不care。它们是conditionally independent的。

作者用conditional entropy $H(v_k | \{v_j\}_{j<k})$ 来quantify这个intuition。公式推导见Appendix D：

$$H(v_k | \{v_j\}_{j<k}) = H(\epsilon_k | \{v_j\}_{j<k}) \leq H(\epsilon_k) \leq \frac{1}{2}\log((2\pi e)^d |\Sigma|)$$

- $v_k$：要预测的token的continuous feature
- $\{v_j\}_{j<k}$：所有之前tokens的features
- $\epsilon_k = v_k - f(\{v_j\})$：residual error
- $\Sigma$：$\epsilon_k$ 的covariance matrix
- $d$：feature dimension
- $|\Sigma|$：covariance matrix的determinant，可以proxy conditional entropy

作者训了一个小model拟合 $f$，算residual的covariance。Fig.11的结果非常直观：对每个reference token位置，画其他所有位置的conditional entropy map。**Reference周围最深红（entropy最低，dependency最强），远处浅色（entropy高，dependency弱）**。

这quantitatively验证了spatial locality。

---

## 那怎么利用这个property？

### Naive方案：把image切成4块，每块内parallel生成

这个方案死掉了，原因刚才说了。Local region内tokens仍然adjacent，dependency强。

### PAR方案：跨region取对应位置parallel生成

这是关键创新。把image切成 $M \times M$ 个region（比如 $M=2$ 就是4个region），然后按"对应位置"跨regiongroup tokens：

```
Region 1 (左上)    Region 2 (右上)
[v_1^(1), v_2^(1), ...]    [v_1^(2), v_2^(2), ...]

Region 3 (左下)    Region 4 (右下)
[v_1^(3), v_2^(3), ...]    [v_1^(4), v_2^(4), ...]
```

每个parallel group是 $[v_j^{(1)}, v_j^{(2)}, v_j^{(3)}, v_j^{(4)}]$，即4个region中相同位置的tokens。

**为什么这样group work？** 因为这4个tokens虽然position相同（比如都是各自region的第5个token），但spatial距离远，dependency弱，conditional independence approximation tight。

### 但还有一个坑：Initial tokens必须sequential

作者发现如果4个region的initial tokens也parallel生成，会出问题。Fig.5 middle row展示了：dog会有重复的body parts，不同region的global structure互相不协调。

**原因**：Initial tokens决定global structure（object layout、scene composition）。每个region的initial token需要知道其他region的initial token是什么，才能coordinate。如果parallel生成，每个region"自说自话"，global就崩了。

所以PAR分两个stage：
- **Stage 1**：$M^2 = 4$ 个initial tokens sequential生成（只有4步，开销很小）
- **Stage 2**：剩余 $(576-4)/4 = 143$ 个parallel groups，每个group 4个tokens同时生成

Total：$4 + 143 = 147$ steps，相比原来576步，**3.9× reduction**。

---

## 画图直觉

把整个process visualize一下：

**传统AR（raster scan）**：
```
Step 1: [1][_][_][_][_][_][_][_]
Step 2: [1][2][_][_][_][_][_][_]
Step 3: [1][2][3][_][_][_][_][_]
...
Step 576: [1][2][3]...[576]
```
每步只填一个格子，576步。

**PAR-4×（M=2）**：
```
Stage 1 (sequential, 4 steps):
Step 1: [R1-1][_][_][_][_][_][_][_]
Step 2: [R1-1][R2-1][_][_][_][_][_][_]
Step 3: [R1-1][R2-1][R3-1][_][_][_][_][_]
Step 4: [R1-1][R2-1][R3-1][R4-1][_][_][_][_]

Stage 2 (parallel, 143 steps):
Step 5: [R1-1][R2-1][R3-1][R4-1][R1-2][R2-2][R3-2][R4-2]
                                                    ↑ 同一step生成4个tokens
Step 6: [R1-1][R2-1][R3-1][R4-1][R1-2][R2-2][R3-2][R4-2][R1-3][R2-3][R3-3][R4-3]
...
```

注意Stage 2每步生成的4个tokens来自4个不同的spatial region，spatial距离远，但position对齐。

---

## Architecture怎么实现：Sequence Reordering

这里有个工程上的elegant trick。作者没有改transformer architecture，只是reorder了input sequence。

传统LlamaGen的input sequence是raster scan：
```
[C, t1, t2, t3, t4, t5, t6, t7, t8, ...]
```
其中 $C$ 是class token，$t_i$ 是第 $i$ 个token（raster scan顺序）。

PAR的input sequence被reorder成：
```
[C, 1, 2, 3, 4, M1, M2, M3, 5a, 5b, 5c, 5d, 6a, 6b, 6c, 6d, ...]
```

- `[C]`：class token
- `[1, 2, 3, 4]`：4个sequential initial tokens（每个region一个）
- `[M1, M2, M3]`：3个learnable transition tokens，帮model从"single token mode"切换到"4-token parallel mode"
- `[5a, 5b, 5c, 5d]`：第一个parallel group，4个region的对应position tokens
- `[6a, 6b, 6c, 6d]`：第二个parallel group
- ...

**为什么需要transition tokens？** Model训练时一直在predict single next token，突然让它predict 4个tokens，prediction目标变了，attention pattern也变了。Learnable tokens相当于给model一个缓冲区，让它学会从sequential context中extract信息并prepare parallel prediction。这些tokens与regular tokens同维度，无缝整合。

**Position embedding怎么处理？** Tokens被reorder了，raster scan的sequential position不再反映spatial position。作者用2D RoPE，每个token的spatial coordinate $(x, y)$ 保持原始image中的位置，与sequence position解耦。

RoPE在attention计算时把position信息编码到query和key的rotation中：

$$\text{Attention}(q, k) = \text{softmax}\left(\frac{(R_{\text{pos}} q)^T (R_{\text{pos}} k)}{\sqrt{d}}\right)$$

其中 $R_{\text{pos}}$ 是position对应的rotation matrix。2D RoPE对 $x$ 和 $y$ 方向分别rotation。

RoPE的bonus：支持zero-shot high-resolution generation（Fig.6）。384×384训练的model可以zero-shot生成512×512，因为RoPE的rotation对位置外推友好。

---

## Group-wise Attention的Trick

这是最subtle的architecture design。Standard causal attention下，predict token $6d$ 时，可以看到所有先前tokens包括 $6a, 6b, 6c$。但如果naive地用causal mask parallel predict $[6a, 6b, 6c, 6d]$：

- $6a$ 只能看到 $5a$ 之前的tokens（看不到 $5b, 5c, 5d$）
- $6b$ 只能看到 $5a, 5b$ 之前的tokens（看不到 $5c, 5d$）
- $6c$ 只能看到 $5a, 5b, 5c$ 之前的tokens（看不到 $5d$）
- $6d$ 能看到 $5a, 5b, 5c, 5d$ 全部

这意味着 $6a$ 预测时丢失了上一组parallel group的大部分信息！

**Solution**：在parallel group内部用bi-directional attention，在group之间保持causal attention。

```
Group 5: [5a, 5b, 5c, 5d]  ← bi-directional within group
Group 6: [6a, 6b, 6c, 6d]  ← bi-directional within group
         ↓
Group 6 可以看到 Group 5 的所有tokens (causal across groups)
Group 5 看不到 Group 6 (future)
```

这样 $6a$ 预测时能看到 $5a, 5b, 5c, 5d$ 全部，context完整。

**为什么group内部bi-directional没问题？** 因为group内tokens是cross-region的distant tokens，dependency弱，互相看到也不会"作弊"或产生inconsistency。反而能利用彼此的信息（虽然弱）来improve prediction。

**KV-cache兼容性**：Group之间causal，之前group的KV可以cache。Group内部bi-directional只在当前forward pass计算，不影响cache。所以这个design完全兼容standard KV-cache optimization。

Tab.4c的ablation：causal → FID 3.64，full (bi-directional) → FID 2.61，**1.03 FID improvement**。

---

## 核心Ablation：Token Ordering决定一切

Tab.4d是最informative的ablation：

| Ordering | Prediction | FID | Steps |
|----------|------------|-----|-------|
| Raster scan | Single token | 2.62 | 576 |
| Distant (PAR) | Single token | 2.64 | 576 |
| Raster scan | Multi-token | 5.64 | 147 |
| Distant (PAR) | Multi-token | 2.61 | 147 |

**关键发现**：
- Single token下，raster和distant ordering性能相当（2.62 vs 2.64）
- Multi-token下，raster崩盘（5.64），distant保持（2.61）

这说明**parallel generation的质量完全取决于你选哪些tokens parallel**。选adjacent tokens（raster multi），dependency强，independent sampling崩盘。选distant tokens（PAR multi），dependency弱，quality保持。

Fig.5的visualization很直观：
- **Top row（PAR）**：sequential initial + parallel distant → 高质量coherent image
- **Middle row（parallel without sequential initial）**：global structure崩了，重复body parts
- **Bottom row（parallel adjacent tokens）**：local pattern崩了，distorted texture

---

## 实验结果

### ImageNet 256×256（Tab.2）

跟baseline LlamaGen比：

| Model | FID | Steps | Time | Speedup |
|-------|-----|-------|------|---------|
| LlamaGen-3B | 2.18 | 576 | 12.41s | 1× |
| PAR-3B-4× | 2.29 | 147 | 3.46s | 3.58× |
| PAR-3B-16× | 2.88 | 51 | 1.31s | 9.5× |

- PAR-4×：FID涨0.11，几乎free的3.6× speedup
- PAR-16×：FID涨0.7，9.5× speedup
- Scaling helps：PAR-XXL-4× (1.4B) FID 2.35 ≈ LlamaGen-XXL FID 2.34，larger model能更好model parallel tokens的joint distribution

### Video Generation UCF-101（Tab.3）

| Method | FVD | Steps | Time | Speedup |
|--------|-----|-------|------|---------|
| MAGVIT-v2-AR | 109 | 1280 | 336.7s | 1× |
| PAR-1× (baseline) | 94.1 | 1280 | 43.3s | 1× |
| PAR-4× | 99.5 | 323 | 11.27s | 3.8× |
| PAR-16× | 103.4 | 95 | 3.44s | 12.6× |

Video只在spatial dimension做parallelization，temporal保持sequential（因为temporal dependency强，parallel会导致motion inconsistency）。

### 与LLM Engineering Optimization正交（Tab.7）

| Model | Optimization | Latency |
|-------|-------------|---------|
| LlamaGen-3B | none | 12.41s |
| LlamaGen-3B | vLLM | 4.12s |
| PAR-3B-4× | none | 3.46s |
| PAR-3B-4× | PyTorch compile | 1.15s |
| PAR-3B-16× | PyTorch compile | 0.43s |

**PAR无优化就比vLLM优化的LlamaGen快**。PAR + compile达到0.43s，比vanilla LlamaGen快28.8×。Algorithm-level和system-level optimization是乘法关系。

---

## 为什么这个方法Work：Intuition Building

### 类比1：拼图游戏

想象你在拼一幅1000片的拼图：
- **先拼四个角**：需要看全局，决定整体layout（sequential initial tokens）
- **然后每个区域同时填**：每个区域的拼工可以独立工作，因为他们负责的region之间dependency弱（parallel cross-region tokens）
- **如果让相邻区域的拼工同时填**：他们会互相干扰，因为相邻pieces必须严丝合缝（parallel adjacent tokens，崩盘）

### 类比2：画画

油画家画画的过程：
- **先画构图**：用几条线确定object位置、scene layout（sequential initial tokens，建立global structure）
- **然后各区域同时铺色**：左上角的天空和右下角的草地可以同时画，因为它们color palette独立（parallel cross-region tokens）
- **细节需要refine**：每个region内部的texture需要根据邻居adjust（sequential within region）

### 类比3：分布式系统

把image generation看作一个distributed computing problem：
- **强dependency的tasks必须sequential执行**（adjacent tokens，类似critical path）
- **弱dependency的tasks可以parallel执行**（distant tokens，类似independent tasks）
- **关键是要正确identify dependency structure**

PAR做的事情就是**挖掘visual data的dependency structure**，然后据此schedule generation tasks。

### 与Diffusion的对比

Diffusion model用iterative denoising实现parallel generation：所有pixels同时更新，每步都看整个image来refine。这相当于"parallel but iterative"。

PAR用conditional independence实现parallel generation：选定distant tokens同时sample，每步都是closed-form的conditional distribution。这相当于"parallel and one-shot"。

两种parallelism的本质区别：
- **Diffusion**：不知道dependency structure，用iterative refinement来处理所有dependency
- **PAR**：知道dependency structure（spatial locality），直接exploit它来parallelize

### 与Speculative Decoding的对比

Speculative decoding用draft model生成候选，main model verify。这需要额外model且verification有cost。

PAR不需要draft model，直接利用visual data的structure property。但PAR的parallelism是**fixed pattern**的（cross-region aligned positions），而speculative decoding是adaptive的。

### 与VAR的对比

VAR用next-scale prediction：coarse scale所有tokens先生成，然后fine scale所有tokens再生成。这也是一种parallelism，但需要specialized multi-scale tokenizer。

PAR用cross-region parallelism：同一scale内，不同region的对应position tokens同时生成。不需要specialized tokenizer。

VAR的parallelism是**scale-wise**，PAR的parallelism是**spatial-wise**。两者可以结合：每个scale内用PAR的cross-region strategy。

---

## Limitations和我的思考

### 1. Fixed Parallel Group Size

当前PAR用fixed $M$，所有positions都用相同的parallel group size。但不同位置的prediction difficulty不同：
- Texture region（如zebra stripes）：local dependency强，应该小group
- Smooth region（如sky）：local dependency弱，可以大group

Adaptive方法可能根据conditional entropy动态调整group size。这类似diffusion的adaptive step size。

### 2. Long-range Dependency

某些visual elements有long-range dependency（Fig.7的deer antlers, vehicle wheels的symmetry）。这些elements需要global coordination。当前PAR依赖spatial locality假设，对这些cases处理suboptimal。

可能的改进：用attention map自动identify strong long-range dependency，将这些tokens也sequential生成。

### 3. Temporal Parallelization

作者提到temporal dimension的parallelization效果不好。这暗示video generation的bottleneck在temporal modeling。

可能的改进：
- **Hierarchical temporal parallelization**：coarse-to-fine temporal，先parallel生成keyframes，再interpolate中间frames
- **Motion-aware tokenization**：将motion信息与appearance分离，appearance可以parallel，motion保持sequential

### 4. Text-to-Image Extension

当前实验限于class-conditional。Text-to-image需要处理text-image alignment。Text tokens与image tokens的cross-attention可能影响parallelization strategy。

Text condition提供了额外的global guidance，可能允许更aggressive的parallelization（因为text已经约束了global structure）。但也可能导致新的dependency pattern（text token与特定image region的强dependency）。

### 5. 更激进的Parallelization

当前PAR-16×用 $M=4$，16个region。能否更激进？

极限情况：$M = H = 24$，每region 1个token，576个region，全parallel生成。这退化到non-autoregressive generation，需要像MaskGIT那样iterative refinement。

所以PAR的parallelism spectrum：
- $M=1$：fully sequential，576步
- $M=2$：4× parallel，147步
- $M=4$：16× parallel，51步
- $M=24$：fully parallel，1步（但quality崩盘）

Sweet spot在 $M=2$ 到 $M=4$ 之间，取决于quality-efficiency tradeoff。

---

## 对Future Research的启示

### 1. Dependency-aware Generation Scheduling

PAR的核心insight是**generation order应该reflect dependency structure**。这可以推广到：
- **Learned ordering**：用model自动learn最优generation order
- **Dynamic ordering**：根据当前generation state动态调整order
- **Content-aware ordering**：不同image类型用不同order

### 2. Visual Token的Conditional Independence Structure

Visual data有丰富的conditional independence structure，远不止spatial locality：
- **Symmetry**：左右对称的tokens应该coordinated生成
- **Repetition**：重复pattern的tokens可以share information
- **Hierarchical**：coarse-to-fine的structure可以exploit

这些structure可以被explicitly modeled来design更高效的generation algorithms。

### 3. Unified Multimodal Generation

PAR保持standard AR architecture，可以直接与language model整合。这对unified multimodal generation（如Chameleon、Emu3）有implications：
- **Modality-specific parallelization**：visual tokens用PAR，text tokens保持sequential
- **Cross-modal parallelization**：text tokens和对应image tokens能否parallel生成？

### 4. Inference System Co-design

Tab.7显示algorithm和system optimization是orthogonal的。Future work可以co-design：
- **KV-cache for group-wise attention**：optimize cache strategy for PAR's attention pattern
- **Batched parallel generation**：batch多个parallel groups的attention computation
- **Hardware-aware group size**：根据GPU memory hierarchy选最优 $M$

---

## 最后的Intuition

PAR这个工作让我想到一个deep principle：**efficiency的瓶颈往往不在model architecture，而在algorithm design对data structure的理解**。

Standard AR visual generation用raster scan order，这是最naive的order，完全没exploit visual data的structure。VAR用multi-scale order，exploit hierarchical structure。PAR用cross-region order，exploit spatial locality structure。

每种order都对应data的一种conditional independence structure。**最优generation order应该maximally exploit这种conditional independence**，让每步的parallel group尽可能conditionally independent。

这个principle不仅适用于visual generation，也适用于：
- **Audio generation**：temporal locality可以exploit
- **3D generation**：spatial+temporal structure
- **Molecule generation**：chemical bond structure
- **Code generation**：syntactic structure

甚至language generation本身也可能benefit：虽然language的dependency structure更complex（long-range syntax、semantic dependency），但某些locality property可能存在。比如adjacent tokens的dependency强，distant tokens的dependency弱，这跟visual类似。但language的"distance"不是spatial的，而是syntactic tree上的distance。

PAR给我最大的启示是：**当你想parallelize一个sequential process时，先understand process的dependency structure，然后parallelize那些conditionally independent的components**。这是一个非常general的principle，远超visual generation的scope。

---

# Parallelized Autoregressive Visual Generation (PAR) 详解

这篇paper由HKU和ByteDance合作，提出了一种**non-local parallel generation**策略，在保持autoregressive model架构不变的前提下，将visual generation的inference速度提升3.6×-9.5×，同时质量损失极小。核心insight非常elegant：**visual token的dependency与spatial distance强相关，远距离tokens之间几乎是conditionally independent的，可以并行采样**。

---

## 1. Motivation: 为什么standard AR visual generation慢

Standard autoregressive visual generation（如LlamaGen、VAR、Emu3等）遵循token-by-token的raster scan顺序：

$$v_1 \to v_2 \to \dots \to v_{HW}$$

每个token都通过sampling（如top-k）从conditional distribution $\mathbb{P}(v_k | v_{<k})$ 中采样得到。对于256×256 ImageNet image，使用16× downsampling的VQGAN tokenizer，token sequence长度为24×24=576，意味着576次sequential forward pass。在A100上LlamaGen-3B需要12.41秒/image，严重限制practical deployment。

加速的naive思路是每步predict多个token，但这里有一个**核心矛盾**：

**矛盾的本质**：autoregressive sampling要求每个token在采样时**已知之前所有token的取值**，这样才能从正确的conditional distribution采样。如果两个tokens $v_a, v_b$ 有strong dependency，他们的joint distribution $\mathbb{P}(v_a, v_b | \text{context})$ 无法factorize为 $\mathbb{P}(v_a | \text{context}) \cdot \mathbb{P}(v_b | \text{context})$。Independent sampling会违反这种dependency，导致inconsistent prediction。

在language domain，speculative decoding [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192) 和Jacobi decoding [Song et al. 2021](https://arxiv.org/abs/2105.13200) 通过draft model或iterative refinement实现parallel generation，但都需要额外model或多次refinement iteration。Visual domain中MaskGIT [Chang et al. 2022](https://arxiv.org/abs/2112.01526) 用masked prediction，VAR [Tian et al. 2024](https://arxiv.org/abs/2404.02905) 用next-scale prediction，但都需要specialized architecture。

---

## 2. Pilot Study: Adjacent tokens为什么不能parallel生成

作者做了一个pilot study（Fig.1b, Tab.4d）：将image分成local region，每个region内同时predict多个adjacent tokens。结果显示严重的quality degradation：
- 老虎的脸distorted
- 斑马的条纹fragmented
- FID从2.62涨到5.64（Tab.4d raster+multi）

**根本原因**：adjacent visual tokens的joint distribution高度correlated。例如zebra的条纹是有规律的纹理，相邻token必须保持stripe方向、宽度、相位的一致性。当独立采样时，每个token根据各自的marginal distribution采样，但marginal distribution远比joint distribution broad，因此无法保证邻接一致性。

可以用information theory精确刻画：对相邻token $v_k, v_{k+1}$，conditional entropy $H(v_{k+1} | v_k, \text{context}) \ll H(v_{k+1} | \text{context})$，即知道邻居后entropy大幅降低。Independent sampling相当于忽略了这种conditional信息。

---

## 3. Key Insight: Visual token dependency与spatial distance

Visual data有一个天然性质：**spatial locality**。Spatially distant的tokens之间，conditional dependency很弱。比如image左上角的sky token和右下角的grass token，给定global context后，几乎是conditionally independent的。

Appendix D给出了theoretical justification。作者用conditional entropy $H(v_k | \{v_j\}_{j<k})$ 度量token dependency：

**Model**: $v_k = f(\{v_j\}_{j<k}) + \epsilon_k$ （Eq.4）

其中 $f(\cdot)$ 是deterministic function，$\epsilon_k$ 是additive noise。

由于 $f$ 是deterministic的：

$$H(v_k | \{v_j\}_{j<k}) = H(\epsilon_k | \{v_j\}_{j<k}) \leq H(\epsilon_k) \leq \frac{1}{2}\log((2\pi e)^d |\Sigma|)$$ （Eq.5-6）

- $v_k \in \mathbb{R}^d$：target token的continuous feature（VQGAN codebook embedding）
- $\{v_j\}_{j<k}$：所有先前tokens的features
- $\Sigma$：residual error $\epsilon_k$ 的covariance matrix
- $d$：feature dimension
- 第二个不等号来自maximum entropy theorem：Gaussian distribution在固定covariance下entropy最大

因此 $|\Sigma|$ 可以作为conditional entropy的proxy。作者训练一个parameterized model $f_\theta$ 拟合 $f$，然后用residual $\epsilon_k = v_k - f_\theta(\{v_j\})$ 计算empirical covariance matrix。

**实验结果**（Fig.11）：对每个reference token位置 $v_j$，计算其他所有位置 $v_i$ 的 $H(v_i | v_j)$。结果发现：
- **Spatially adjacent tokens有最低conditional entropy**（图中最红的区域集中在reference周围）
- **Distant tokens的conditional entropy较高**（dependence弱）

这quantitatively验证了spatial locality假设。

---

## 4. Design Principles

基于以上分析，作者提出三条design principles：

1. **Initial tokens of each region必须sequential生成**，以建立global structure
2. **Local region内部保持sequential生成**，因为相邻tokens dependency强
3. **Cross-region对应位置的tokens可以parallel生成**，因为distant tokens dependency弱

为什么initial tokens必须sequential？作者在Fig.5 middle row展示了反面教材：如果4个region的initial tokens并行生成，每个region独立决定自己的global layout，会导致：
- Dog有重复的body parts
- 不同region的structure互相不consistent
- Global coherence被破坏

Initial tokens相当于image的"骨架"，必须由global coordinator（sequential AR）来决定。后续tokens填充细节时，依赖已经established的global structure，因此可以parallel化。

---

## 5. Method: Non-Local Parallel Generation

### 5.1 Cross-region Token Grouping

给定 $H \times W$ 的token grid（如24×24=576），将其划分为 $M \times M$ 个region（如 $M=2$ 则4个region，每个region 12×12=144 tokens）。

记 $v_j^{(r)}$ 为region $r$ 中第 $j$ 个token，$r \in \{1, ..., M^2\}$，$j \in \{1, ..., k\}$，$k = \frac{H}{M} \times \frac{W}{M}$。

将tokens按照**跨region的对应位置**重新组织（Eq.1）：

$$\{[v_1^{(1)}, v_1^{(2)}, ..., v_1^{(M^2)}], [v_2^{(1)}, ..., v_2^{(M^2)}], ..., [v_k^{(1)}, ..., v_k^{(M^2)}]\}$$

每个group $[v_j^{(1)}, ..., v_j^{(M^2)}]$ 包含 $M^2$ 个spatially distant但position-aligned的tokens，这些tokens将parallel生成。

**Intuition**：比如4个region中，每个region的左上角token被group在一起。这些token虽然position相同，但spatially相距很远，conditional dependency弱，可以同时采样。

### 5.2 Stage 1: Sequential Generation of Initial Tokens

第一个stage，每个region生成1个initial token，共 $M^2$ 个tokens，全部sequential（Eq.2）：

$$v_1^{(i)} \sim \mathbb{P}(v_1^{(i)} | v_1^{(<i)}), \quad i \in \{1, ..., M^2\}$$

- $v_1^{(i)}$：第 $i$ 个region的initial token
- $v_1^{(<i)}$：之前所有region的initial tokens
- 这一步只有 $M^2$ 步（如 $M=2$ 则4步），开销很小但建立了global structure

### 5.3 Stage 2: Parallel Generation of Cross-region Tokens

第二个stage，剩余 $k-1$ 个position，每个position同时生成 $M^2$ 个tokens（Eq.3）：

$$\{v_j^{(r)}\}_{r=1}^{M^2} \sim \mathbb{P}(\{v_j^{(r)}\}_{r=1}^{M^2} | v_{<j})$$

- $\{v_j^{(r)}\}_{r=1}^{M^2}$：当前步骤要parallel生成的 $M^2$ 个tokens
- $v_{<j}$：所有先前生成的tokens（包括initial tokens和之前的parallel groups）
- 关键：虽然group内tokens同时采样，但**每个token的conditional distribution都condition在所有先前tokens上**，因此仍然保持autoregressive property

**Total steps计算**：对24×24=576 tokens，$M=2$（4 regions）：
- Stage 1: 4 sequential steps
- Stage 2: $(576-4)/4 = 143$ parallel steps
- Total: $4 + 143 = 147$ steps（vs. 原始576 steps，3.9× reduction）

对 $M=4$（16 regions）：$16 + (576-16)/16 = 16 + 35 = 51$ steps（11.3× reduction）

### 5.4 公式的factorization intuition

Eq.3表面上写成了joint distribution $\mathbb{P}(\{v_j^{(r)}\} | v_{<j})$，但实际sampling时仍然independent：

$$\mathbb{P}(\{v_j^{(r)}\} | v_{<j}) \approx \prod_{r=1}^{M^2} \mathbb{P}(v_j^{(r)} | v_{<j})$$

这种factorization的approximation error取决于group内tokens的conditional mutual information。Cross-region对应位置tokens的mutual information低（Fig.11验证），因此approximation tight。

---

## 6. Model Architecture Details

### 6.1 Sequence Structure

作者通过**reorder input sequence**而非修改architecture来实现parallel prediction。Input sequence结构（Fig.4a）：

```
[C, 1, 2, 3, 4, M1, M2, M3, 5a, 5b, 5c, 5d, 6a, 6b, 6c, 6d, ...]
```

- `C`：class token（class-conditional generation）
- `[1, 2, 3, 4]`：initial sequential tokens（$M^2$ 个）
- `[M1, M2, M3]`：$M^2 - 1 = 3$ 个learnable transition tokens，帮助model从sequential mode切换到parallel mode
- `[5a, 5b, 5c, 5d]`：第一个parallel group（$M^2=4$ 个tokens）
- `[6a, 6b, 6c, 6d]`：第二个parallel group，以此类推

**为什么需要learnable transition tokens？** Model从predict单个token突然切换到predict 4个tokens，attention pattern和prediction目标都变了。Learnable tokens相当于给model一个"过渡区"，让它学会从sequential context中提取信息并准备parallel prediction。这些tokens的embedding与regular tokens同维度，无缝整合。

### 6.2 Position Embedding: 2D RoPE

由于tokens被reorder，raster scan的sequential position不再反映spatial position。作者用[2D Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864)保持每个token的原始spatial位置信息：

- 每个token有2D spatial coordinate $(x, y)$，与sequence position无关
- RoPE在attention计算时将spatial位置信息编码到query和key中
- 这使得即使token 5a在sequence中位于位置8，但它的spatial coordinate对应原image的某个具体位置

RoPE的另一个好处：支持**zero-shot high-resolution generation**（Fig.6）。384×384训练的model可以zero-shot生成512×512 image，因为RoPE的rotation matrix对位置外推比较友好。

### 6.3 Group-wise Bi-directional Attention with Global Autoregression

这是最tricky的architecture design。Standard causal attention下，predict token 6d时，可以看到所有先前tokens包括6a, 6b, 6c。但如果naive地parallel predict [6a, 6b, 6c, 6d]，causal mask会强制：
- 6a只能看到5a之前的tokens
- 6b只能看到5a, 5b之前的tokens
- 6c只能看到5a, 5b, 5c之前的tokens
- 6d可以看到5a, 5b, 5c, 5d

这意味着6a预测时**不知道5b, 5c, 5d**，丢失了上一组parallel group的完整信息。

**Solution**：在parallel group内部用**bi-directional attention**，在group之间保持**causal attention**。

形式化：设 $G_t = [v_t^{(1)}, ..., v_t^{(M^2)}]$ 是第 $t$ 个parallel group。Attention mask满足：
- 对 $s < t$：$G_t$ 中所有tokens可以attend $G_s$ 中所有tokens（causal across groups）
- 对 $s = t$：$G_t$ 内部tokens互相可以attend（bi-directional within group）
- 对 $s > t$：不能attend（future groups）

**Intuition**：当前parallel group的tokens之间虽然同时生成，但既然他们dependency弱（cross-region），互相看到也没问题。反而能看到上一组complete group信息，context更丰富。

**KV-cache compatibility**：这种group-wise attention仍然兼容KV-cache。因为group之间是causal的，之前group的KV可以cache。Group内部的bi-directional attention只需要在当前forward pass中计算，不影响cache。

### 6.4 Ablation: Attention Pattern的影响

Tab.4c对比了causal vs. full attention within group：
- Causal: FID 3.64
- Full (bi-directional): FID 2.61
- **1.03 FID improvement**，证明group-wise bi-directional attention的重要性

---

## 7. Extension to Video Generation

Video tokenization（MAGVIT-v2）：17 frames @ 128×128 → $T \times H \times W = 5 \times 16 \times 16 = 1280$ tokens（8× spatial, 4× temporal compression）。

Video extension的关键决策：**只在spatial dimension做parallelization，不在temporal dimension做**。

**为什么temporal不能parallel？** Video的temporal dependency有强sequential性质：frame $t$ 依赖frame $t-1$ 的motion、object position、background。Parallel predict多个frames会导致motion inconsistency和object jumping。

Spatial dimension则不同：同一frame内的不同spatial regions之间dependency弱，可以parallel。因此video generation用3D positional embedding，但parallelization策略与image相同。

---

## 8. Experimental Results

### 8.1 ImageNet 256×256 Class-Conditional Generation (Tab.2)

主要对比对象是LlamaGen（baseline，same architecture，same tokenizer）：

| Model | Params | FID↓ | IS↑ | Steps | Time(s) |
|-------|--------|------|-----|-------|---------|
| LlamaGen-3B | 3.1B | 2.18 | 263.3 | 576 | 12.41 |
| PAR-3B-4× | 3.1B | 2.29 | 255.5 | 147 | 3.46 |
| PAR-3B-16× | 3.1B | 2.88 | 262.5 | 51 | 1.31 |
| PAR-XXL-4× | 1.4B | 2.35 | 263.2 | 147 | 6.84 |
| PAR-XXL-16× | 1.4B | 3.02 | 270.6 | 51 | 2.28 |

**Key observations**：
- **PAR-3B-4× vs LlamaGen-3B**: FID 2.29 vs 2.18，仅涨0.11，但speedup 3.58×
- **PAR-3B-16×**: FID 2.88，涨0.7，但speedup 9.5×
- **Scaling helps**: PAR-XXL-4× (1.4B) FID 2.35 接近 LlamaGen-XXL (1.4B) FID 2.34，说明larger model能更好地model parallel tokens的joint distribution

与SOTA对比：
- **vs MaskGIT** (FID 6.18): PAR显著更优（2.29），虽然steps多147 vs 8
- **vs VAR** (FID 1.97): VAR略好但需要specialized multi-scale tokenizer
- **vs MAR** (FID 1.55): MAR用continuous mask token，最好但需要64步iterative denoising
- **vs DiT-XL/2** (FID 2.27): PAR-3B-4× 2.29相当，但速度快很多

### 8.2 UCF-101 Video Generation (Tab.3)

| Method | Params | FVD↓ | Steps | Time(s) |
|--------|--------|------|-------|---------|
| MAGVIT-v2 (mask) | 840M | 58 | - | - |
| MAGVIT-v2-AR | 840M | 109 | 1280 | 336.70 |
| PAR-1× (baseline) | 792M | 94.1 | 1280 | 43.30 |
| PAR-4× | 792M | 99.5 | 323 | 11.27 |
| PAR-16× | 792M | 103.4 | 95 | 3.44 |

- PAR-1× baseline就比MAGVIT-v2-AR好（94.1 vs 109 FVD）
- PAR-16×达到12.6× speedup，FVD仅涨9.3

### 8.3 Compatibility with LLM Engineering Optimizations (Tab.7)

这是非常实用的结果：algorithm-level parallelization与engineering-level optimization是**orthogonal**的。

| Model | Optimization | Latency |
|-------|-------------|---------|
| LlamaGen-3B | none | 12.41s |
| LlamaGen-3B | vLLM (PagedAttention + CUDA graph) | 4.12s |
| PAR-3B-4× | none | 3.46s |
| PAR-3B-4× | PyTorch compile (CUDA graph only) | 1.15s |
| PAR-3B-16× | PyTorch compile | 0.43s |

- **PAR-3B-4×无优化** (3.46s) 已优于 **LlamaGen-3B + vLLM** (4.12s)
- PAR + compile达到1.15s，比optimized LlamaGen快3.6×
- PAR-16× + compile达到0.43s，**28.8× speedup** over vanilla LlamaGen

这说明algorithm-level的sequential step减少与system-level的compute efficiency提升是乘法关系，应该结合使用。

---

## 9. Ablation Studies详解

### 9.1 Initial Sequential Token Generation (Tab.4a)

- Without: FID 3.67, 144 steps
- With: FID 2.61, 147 steps

仅多3步，FID降1.06。Fig.5可视化显示without版本有重复body parts和misaligned structure。

### 9.2 Parallel Group Size n (Tab.4b)

- n=1: FID 2.34, 576 steps (baseline)
- n=4: FID 2.35, 147 steps (4× reduction, FID涨0.01)
- n=16: FID 3.02, 51 steps (11.3× reduction, FID涨0.67)

这显示n=4几乎是"免费"的speedup，n=16才有显著quality drop。

### 9.3 Token Ordering (Tab.4d) - 最重要的ablation

| Order | Pattern | FID | Steps |
|-------|---------|-----|-------|
| Raster | single | 2.62 | 576 |
| Distant (ours) | single | 2.64 | 576 |
| Raster | multi | 5.64 | 147 |
| Distant (ours) | multi | 2.61 | 147 |

- **Single-token下**：raster和distant ordering相当（2.62 vs 2.64）
- **Multi-token下**：raster严重退化（5.64），distant保持（2.61）

这直接证明了**token selection比model architecture更重要**。Raster scan下parallel生成的tokens是adjacent的，dependency强，independent sampling导致distortion。Distant ordering下parallel生成的tokens跨region，dependency弱，quality保持。

### 9.4 Conditional Entropy Validation (Fig.12, Eq.7-8)

作者用conditional entropy quantitatively验证design：

对proposed order，定义entropy increase：

$$\Delta H_{\text{ours}} = H(v_k^{(r)} | \mathcal{V}_{k,r}^{\text{par}}) - H(v_k^{(r)} | \mathcal{V}_{k,r}^{\text{seq}})$$ （Eq.7）

对raster order：

$$\Delta H_{\text{raster}} = H(v_k | \mathcal{V}_k^{\text{par}}) - H(v_k | \mathcal{V}_k^{\text{seq}})$$ （Eq.8）

- $\mathcal{V}_{k,r}^{\text{seq}}$：sequential生成时 $v_k^{(r)}$ 的context
- $\mathcal{V}_{k,r}^{\text{par}}$：parallel生成时 $v_k^{(r)}$ 的context（缺少同组其他tokens）
- $\Delta H$：parallel化导致的prediction难度增加

**Fig.12结果**：proposed order的 $\Delta H$ 显著小于raster order，quantitatively证明cross-region parallelization引入的prediction difficulty远小于adjacent parallelization。

---

## 10. Intuition Building: 为什么这个方法work

让我从几个角度build intuition：

### 10.1 Visual Generation的层次性

Visual generation可以分解为两个level：
- **Global structure**：object layout, scene composition, color palette
- **Local detail**：texture, edges, fine patterns

Global structure需要**coordination** across整张image，必须由sequential process建立。Local detail则相对**modular**，不同region的细节可以独立rendering。

PAR的两阶段策略正好对应：Stage 1 sequential生成initial tokens（建立global structure），Stage 2 parallel生成remaining tokens（填充local detail）。

### 10.2 Conditional Independence的几何结构

考虑4个region的tokens $v_j^{(1)}, v_j^{(2)}, v_j^{(3)}, v_j^{(4)}$，给定global context $C$（之前所有tokens）：

如果 $\mathbb{P}(v_j^{(1)}, v_j^{(2)}, v_j^{(3)}, v_j^{(4)} | C) \approx \prod_r \mathbb{P}(v_j^{(r)} | C)$

那么parallel sampling与sequential sampling的distribution相同，quality保持。

Visual data的spatial locality使得这种factorization在cross-region setting下近似成立。Fig.11的conditional entropy map直接visualize了这一点：reference token周围的conditional entropy低（dependency强），远处高（dependency弱）。

### 10.3 与Diffusion Model的对比

Diffusion model通过iterative denoising实现parallel generation（所有pixels同时更新），但每步依赖model的denoising能力。PAR通过**exploiting visual structure的conditional independence**实现parallel generation，每步的conditional distribution是closed form的（autoregressive sampling），不需要iterative refinement。

这是两种不同的parallelism：
- **Diffusion**: spatial parallelism via iterative denoising
- **PAR**: spatial parallelism via conditional independence factorization

### 10.4 与Speculative Decoding的对比

Speculative decoding [Leviathan et al.](https://arxiv.org/abs/2211.17192) 用draft model快速生成多个候选tokens，main model验证。这需要额外draft model且verification有cost。

PAR不需要draft model，直接利用visual data的structure property。但PAR的parallelism是**fixed pattern**的（cross-region aligned positions），而speculative decoding是adaptive的。

### 10.5 与Blockwise Parallel Decoding的对比

[Blockwise parallel decoding (Stern et al. 2018)](https://arxiv.org/abs/1811.03115) 训练model预测未来多个tokens，然后verify。这与PAR类似但用于language，且没有考虑token之间的dependency structure。PAR的contribution是**identifying which tokens can be parallelized**，即dependency-aware grouping。

---

## 11. Limitations和Future Directions

### 11.1 Temporal Parallelization

作者提到temporal dimension的parallelization效果不好。这暗示video generation的bottleneck在temporal modeling。Future work可能需要：
- Hierarchical temporal parallelization（coarse-to-fine）
- Motion-aware tokenization（将motion信息与appearance分离）

### 11.2 Long-range Dependency

某些visual elements有long-range dependency（Fig.7的deer antlers, vehicle wheels）。这些elements的symmetry和coherence需要global coordination。当前PAR依赖spatial locality假设，对这些cases处理可能suboptimal。

可能的改进：用attention map自动identify strong long-range dependency，将这些tokens也sequential生成。

### 11.3 Adaptive Parallelization

当前PAR用fixed $M$，所有positions都用相同的parallel group size。Adaptive方法可能根据每步的conditional entropy动态调整parallel group size：

- 高confidence（低entropy）position：大parallel group
- 低confidence（高entropy）position：小parallel group or sequential

这类似[Diffusion的adaptive step size](https://arxiv.org/abs/2206.00364)。

### 11.4 与VAR的融合

VAR用next-scale prediction实现parallel，PAR用cross-region parallelization。两者可以结合：
- Coarse scale用sequential（建立global）
- Fine scale用cross-region parallel（填充detail）
- 每个scale内用PAR的strategy

### 11.5 Text-to-Image Generation

当前实验限于class-conditional。Text-to-image需要处理text-image alignment，text tokens与image tokens的cross-attention可能影响parallelization strategy。需要研究text conditioning如何与cross-region parallel generation交互。

---

## 12. 总结

PAR的核心贡献：

1. **Insight**: Visual token dependency与spatial distance强相关，distant tokens可以parallel生成
2. **Method**: 两阶段generation——sequential initial tokens建立global structure，parallel cross-region tokens填充local detail
3. **Architecture**: 通过sequence reordering和group-wise bi-directional attention实现，不修改standard transformer
4. **Theory**: Conditional entropy分析quantitatively验证设计选择
5. **Results**: 3.6×-9.5× speedup，quality损失minimal，与LLM engineering optimizations正交

这个工作的beauty在于**simplicity**——不需要新architecture、不需要新tokenizer、不需要draft model，仅通过reordering和attention mask调整就能实现significant speedup。这强烈暗示visual generation的efficiency bottleneck不在model architecture，而在**generation order的设计**。

对future research的启示：visual data有丰富的structure，这些structure可以被exploited来design更高效的generation algorithms。Conditional independence structure、hierarchical structure、symmetry structure都是可以exploit的方向。

---

## References

- [PAR Project Page](https://yuqingwang1029.github.io/PAR-project/)
- [LlamaGen: Autoregressive Model Beats Diffusion](https://arxiv.org/abs/2406.06525)
- [VAR: Visual Autoregressive Modeling via Next-Scale Prediction](https://arxiv.org/abs/2404.02905)
- [MaskGIT: Masked Generative Image Transformer](https://arxiv.org/abs/2112.01526)
- [Speculative Decoding (Leviathan et al.)](https://arxiv.org/abs/2211.17192)
- [Jacobi Decoding (Song et al.)](https://arxiv.org/abs/2105.13200)
- [Blockwise Parallel Decoding (Stern et al.)](https://arxiv.org/abs/1811.03115)
- [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [VQGAN: Taming Transformers](https://arxiv.org/abs/2012.09841)
- [MAGVIT-v2: Language Model Beats Diffusion](https://arxiv.org/abs/2408.15240)
- [MAR: Autoregressive Image Generation without VQ](https://arxiv.org/abs/2406.11838)
- [Emu3: Next-Token Prediction is All You Need](https://arxiv.org/abs/2409.18869)
- [vLLM: PagedAttention](https://arxiv.org/abs/2309.06180)
- [DiT: Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
