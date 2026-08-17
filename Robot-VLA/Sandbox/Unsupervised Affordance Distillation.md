---
source_pdf: Unsupervised Affordance Distillation.pdf
paper_sha256: bc979fb7ae0479b2f190f13d423a6b39a06013a6f4b3be8e519e202f4d5bdb9b
processed_at: '2026-08-12T20:24:55-07:00'
target_folder: Robot-VLA/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# UAD 用人话讲

Andrej，好，我用更直觉的方式重新给你讲一遍这篇 paper，去掉学术腔，重点 build 你的 intuition。

---

## 一句话版本

**让 GPT-4o 当"标注工人"，给 3D 渲染图打 affordance label，再用这些 label 训一个轻量小模型，最后这个小模型能 generalize 到 real-world 场景，还能当 policy 的"眼睛"用。**

核心 trick 是：VLM 知道"杯子把手用来抓"，DINOv2 知道"哪些 pixel 长得像把手"，把两者拼起来就得到了 pixel-level 的连续 affordance map，全程零人工标注。

---

## 为什么这事难？

Robot 要做 manipulation，给它一句话 "grasp the watering can"，它得先看图，找到 watering can 上具体哪个 pixel 该抓。这个 "哪里可以做什么" 就叫 affordance。

过去的做法有两条路：
1. **人工标 affordance mask**——贵、慢、标不了多少
2. **用 closed-set 分类**——"grasp / push / pour" 固定几个动作，新任务就废了

而且还要 generalize 到 real world、新物体、新 instruction，这在 pixel level 极难。

---

## Insight：VLM 和 LVM 各有短板

- **VLM（GPT-4o）**：知道 "handle is for grasping" 这种 world knowledge，但它只输出文字，没法说"具体哪个 pixel"
- **LVM（DINOv2）**：每个 pixel 都有 rich feature，知道"这块长得像那块"，但它 task-agnostic，不知道你想干啥

UAD 的核心 insight：**把 affordance 重写成 VQA 问题**。给 GPT-4o 看一张已经用 DINOv2 聚类涂好颜色的图，问它"哪个颜色区域用来倒水"，它就能回答。回答完之后，用 DINOv2 feature 的 cosine similarity 把这个离散决策"摊开"成连续 map。

---

## Pipeline 三步走

### Step 1: 用 DINOv2 把物体切成 functional regions

对每个 3D object（来自 BEHAVIOR-1K）：
1. 渲染 14 个 views 的 RGB + depth
2. 对每个 view 跑 DINOv2 ViT-L14 with registers，得到每个 pixel 的 1024 维 feature
3. Multi-view fusion：对每个 3D point，把所有 visible views 的 DINOv2 feature 平均，得到 global 3D feature field
   $$F_{\text{global}} \in \mathbb{R}^{N \times d}$$
   其中 $N$ 是 point cloud 数，$d=1024$ 是 DINOv2 feature 维度
4. PCA 降到 3 维：
   $$F_{\text{reduced}} = \text{PCA}(F_{\text{global}}) \in \mathbb{R}^{N \times 3}$$
   为啥降 3 维？DINOv2 feature 里有大量 texture 噪声，PCA 把主成分留下来，clustering 更稳。
5. Mean Shift clustering（自动选 cluster 数 M）。如果 M < 5 就 fallback 到 k-means k=5
6. Articulated objects（如 cabinet）按 link 分别 cluster，这样能找到 drawer knob 这种小 part

最终每个 3D point 有 region label $r_n \in \{1,...,M\}$。

**为什么 over-segmentation 更好**？如果一个 part 被切成两块，GPT-4o 仍能正确选其中一块；由于 DINOv2 feature 相似，另一块 cosine similarity 也会高，最终 affordance map 仍正确。但 under-segmentation 把功能不同的 parts 合并，会彻底丢信息。

### Step 2: 让 GPT-4o 提任务和配对 region

选最 natural 的 view（用 CLIP 算 object category name 与 image 的 similarity 选 view），把 regions 涂成不同颜色 overlay 到原图上。

Prompt 给 GPT-4o：原图 + region overlay 图 + color list + object category name。要求它输出形如：

```python
{
    "rim of the coffee mug -- region for drinking and pouring": "Red",
    "handle of the mug -- region for grasping": "Blue"
}
```

格式是 Python `ast.literal_eval()` 可 parse 的 dict。

GPT-4o 知道 mug 的语义结构，配合彩色 region 图就能配对。这是一个 VQA 形式的 prompting——把 region-instruction matching 变成可回答的问题。

### Step 3: 把离散 region 决策摊开成连续 affordance map

对每个 GPT-4o 选定的 region $r$：

1. **Reference feature**：region 内所有 3D points 的 DINOv2 feature 平均
   $$f_{\text{ref}} = \frac{1}{|r|} \sum_{p \in r} F_{\text{global}}(p) \in \mathbb{R}^d$$

2. **Per-point similarity**：对每个 3D point p 计算
   $$s(p) = \max\left(0, \cos(f_{\text{ref}}, F_{\text{global}}(p))\right) \in [0,1]$$
   $\cos(\cdot,\cdot)$ 是 cosine similarity，$\max(0, \cdot)$ 把负值 clip 掉。

3. **Project 回 2D**：对每个相机 view 的每个 pixel，通过 depth 反投影到 3D，找 P 中最近邻 point，assign 其 score。得到
   $$A \in [0,1]^{H \times W}$$

**为什么用 cosine similarity 而不是 binary mask？** Binary mask 假设 region 内所有 pixel 等价。但 handle 中间最适合抓，handle 末端次之。Cosine similarity 自动让"中心"区域得分高（feature 最接近 reference），"边缘"得分低——这是 continuous 的 affordance。

4. **Post-processing**：threshold 0.5 + Gaussian blur kernel=3，去掉硬边界。

最终得到 dataset 三元组 $(I, T, A)$：RGB image + free-form instruction + continuous affordance map。

---

## 训 Affordance Model：Frozen DINOv2 + FiLM 头

### 为什么 frozen？

DINOv2 在 LAION 上 self-supervised pretrain，见过海量 real-world image。它的 feature 已经 robust 到 real-world variations。如果 fine-tune，容易破坏这个 generalization，尤其训练数据只有 sim single object。

### FiLM 是什么？

FiLM (Feature-wise Linear Modulation, Perez et al. 2018)：

输入：
- Language embedding $e_T \in \mathbb{R}^{d_L}$（$d_L \approx 3072$，OpenAI embedding API）
- Pixel features $X \in \mathbb{R}^{H \times W \times C_{\text{in}}}$（DINOv2 输出）

对每个 output channel $c$：
$$\gamma_c = W_\gamma^c e_T + b_\gamma^c \in \mathbb{R}^{C_{\text{in}}}$$
$$\beta_c = W_\beta^c e_T + b_\beta^c \in \mathbb{R}$$

对每个 pixel $(h,w)$：
$$X'_{h,w,c} = \sum_{c'=1}^{C_{\text{in}}} \gamma_{c,c'} X_{h,w,c'} + \beta_c$$

变量含义：
- $\gamma_c \in \mathbb{R}^{C_{\text{in}}}$：channel-wise scale，由 language 控制
- $\beta_c \in \mathbb{R}$：channel-wise shift
- $W_\gamma^c \in \mathbb{R}^{C_{\text{in}} \times d_L}$, $W_\beta^c \in \mathbb{R}^{d_L}$：linear layer 参数

**关键性质**：$\gamma_c, \beta_c$ 只依赖 $e_T$，**与 pixel 位置无关**。整个 pixel space 共享同一组 transformation。

**直觉**：DINOv2 feature 已经告诉你 "这个 pixel 长啥样"，FiLM 用 language 选一组 channel weights，把 task-relevant 的 channel 放大、task-irrelevant 的压下去。整个图共享同样的 "放大模式"，类似 attention 但极轻量（只 ~1M 参数）。

**为什么用 FiLM 而非 cross-attention**？Cross-attention 每个像素与每个 language token 交互，参数多、容易 overfit、inference 慢。FiLM 是 location-agnostic 的 channel-wise projection，更 sample-efficient 且语义清晰。

### 完整架构

3 个 FiLM-conditioned conv layers：

| Layer | Input $C_{\text{in}}$ | Output $C_{\text{out}}$ |
|-------|------|--------|
| FiLM-1 | 1024 (DINOv2) | 256 |
| FiLM-2 | 256 | 64 |
| FiLM-3 | 64 | 1 |

最后一层经 sigmoid 输出 $\hat{A} \in [0,1]^{H \times W}$。

### Loss

Binary Cross-Entropy：
$$\mathcal{L} = -\frac{1}{HW} \sum_{h=1}^{H} \sum_{w=1}^{W} \left[ A_{h,w} \log \hat{A}_{h,w} + (1-A_{h,w}) \log(1-\hat{A}_{h,w}) \right]$$

变量：
- $A_{h,w}$：GT affordance，是 continuous 值 [0,1]
- $\hat{A}_{h,w}$：predicted affordance

**为啥 GT 是 continuous 还用 BCE 不用 MSE**？BCE 对中间值梯度更稳定，MSE 在 [0,1] 边界附近梯度小。

### Training 细节

- Linear layer 初始化 weight=1, bias=0（参考 RT-1）
- Adam, lr=0.001
- 30 epochs, batch size 8
- 12 hours on 1× A6000 GPU
- 只训 FiLM layers，DINOv2 300M params 完全 frozen

---

## Affordance 当 Policy 的 Observation Space

### 架构

基于 RVT (Goyal et al. 2023) multi-view transformer policy。每个 view 的输入通道：
1. **UAD affordance map** $A \in [0,1]^{H \times W}$（task-conditioned！）
2. Depth 值
3. $(x, y, z)$ world coords（per-pixel）
4. Global proprioception vector（拼接成单 vector）

Output：7-dim action = 6-DoF end-effector pose + binary gripper action。

### 关键设计

**Affordance 作 observation，不作 auxiliary loss 或 pretrain signal**。这把 affordance 直接变成 policy 的"眼睛"，让 policy 聚焦 task-relevant regions。

**Affordance model 在 policy training 时 frozen**。保留 UAD generalization，避免 10 demos 上 overfit 破坏 representation。

### 为什么 10 demos 够用？

通常 vision-based imitation learning 要大量 demos 才能学 visual representation。UAD 把 representation learning 解耦了——pre-trained affordance 已经 task-conditioned 且 fine-grained，policy 只需学 motion-level mapping (affordance map → action)。

对比 baseline（DINOv2 / CLIP / Voltron）：
- CLIP 是 image-level "bag-of-words"，pixel-level 表现差
- DINOv2 是 task-agnostic，policy 还得自己学会 attend
- Voltron 用 CLIP-like objective，也偏向 image-level

UAD 直接把 attention built-in 到 observation 里。

---

## 实验数据

### Affordance prediction

**Sim sanity check**（100 pairs per setting, AUC）：
| Setting | AUC |
|---------|-----|
| Train data | ≥0.92 |
| Novel instances | ≥0.92 |
| Novel categories | ≥0.92 |
| Novel instructions | ≥0.92 |

**DROID（real-world robotic scenes）**：
| Method | AUC |
|--------|-----|
| CLIP | 0.500 |
| OpenSeeD | 0.836 |
| **UAD** | **0.840** |

CLIP 0.5 接近 chance，证实 CLIP pixel-level 能力差。OpenSeeD 是 segmentation，binary 输出，对小 region 容易失败。UAD continuous 且 fine-grained，略胜。

**AGD20K（human activity affordances, zero-shot）**：
| Method | KLD ↓ | SIM ↑ | NSS ↑ | NSS-0.5 ↑ |
|--------|------|------|------|-----------|
| Cross-View-AG | 1.787 | 0.285 | 0.829 | - |
| LOCATE | 1.405 | 0.372 | 1.157 | 1.723 |
| 3DOI | 3.565 | 0.227 | 0.657 | - |
| AffordanceLLM | 1.463 | 0.377 | 1.070 | - |
| **UAD** | 1.878 | **0.407** | 1.092 | **2.050** |

UAD SIM 最高，NSS-0.5（更严 metric）也最高。KLD 偏高是因为 AGD20K GT 是 Gaussian mixture 扩散到 background，UAD 在 background 预测 0 被惩罚。

注意：UAD zero-shot generalize 到 "eating bananas"、"taking photos"、"sitting on bicycles"、"holding golf clubs"、"lying on bed"、"typing on computers" 这些完全 OOD 的 human activities，相当 impressive。

### Sim policy generalization

3 tasks（Pouring / Opening / Insertion）× 4 settings（new pose / instance / category / instruction）× 15 trials。

UAD 在所有 setting 都优于 baseline，特别是 fine-grained perception 的 Opening task（检测 thin drawer handles）。

### Real-world policy

10 demos via kinesthetic teaching, Franka Panda, 2 RGB-D cameras (Orbbec Femto Bolt)。

| Task | Success Rate |
|------|--------------|
| Watering plant | ~73% |
| Opening drawer | ~73% |
| Inserting pen | ~73% |
| **Average** | **73%** |

10 demos + zero-shot sim-to-real + novel objects + novel instructions，73% 已经相当 promising。

---

## Intuition 几个 Key Takeaways

### 1. Distillation as automatic labeling

用 VLM 当"老师"标 data，蒸馏到轻量专用模型。这个 pattern 在 robotics 越来越普遍，避免 inference 时调用慢且贵的 VLM。对比 CoPa / MOKA / PIVOT / RoboPoint 都是 inference 时调 VLM——慢且贵。UAD 推理快。

### 2. Frozen backbone + lightweight head

Data-scarce robotics 的 sample efficiency 关键。DINOv2 300M params frozen，只训 FiLM ~1M params。10 demos 就能 generalize 到 novel categories 说明 representation 占了大头，policy 学的是 motion-level mapping。

### 3. 3D consistency trick

虽然最终输出 2D，从 3D 渲染训练能 enforce view-consistency。学到的是 object-centric functional regions，避免 view-specific artifacts。Sim-to-real 中这个 trick 可能 underrated。

### 4. Feature similarity as soft label

把 VLM 的离散决策"摊开"成连续 map，用 DINOv2 feature similarity 作 smooth proxy。避免 binary mask 的硬边界，更符合 affordance 的 continuous nature。这个 trick 很 elegant，可推广到其他 "discrete VLM decision → continuous spatial map" 问题。

### 5. Over-segmentation > under-segmentation

Clustering 偏向多切一点，GPT-4o 仍能正确选 region，cosine similarity 让同 part 的其他 cluster 也得分高。Under-segmentation 会永久丢功能信息。这是个实用 engineering insight。

---

## Limitations

1. **Single static frame**：不考虑 video / multi-step affordance reasoning
2. **Single object rendering**：训练数据只有 single objects，多 object scene 是 zero-shot 泛化
3. **No motion-level generalization**：affordance 是 "where"，不解决 "how"
4. **依赖 GPT-4o 质量**：VLM hallucinate 会传播错误 label

---

## 潜在 Extension

- **Video affordance**：给 task，预测 affordance trajectory
- **Multi-object interaction**：场景级别 relational affordance
- **Affordance-based RL reward**：affordance map 作 dense reward signal
- **Bimanual manipulation**：两只手各自 affordance
- **Scaling to Objaverse-XL**：paper appendix 已经 case study 10,000+ object-instruction pairs

---

## Reference Links

- **UAD project page**: https://unsup-affordance.github.io/
- **DINOv2**: https://dinov2.metademolab.com/ | paper: https://arxiv.org/abs/2304.07193
- **DINOv2 with registers**: https://arxiv.org/abs/2309.16588
- **FiLM paper**: https://arxiv.org/abs/1709.07871
- **RVT**: https://arvt-2.github.io/ | paper: https://arxiv.org/abs/2306.13096
- **D3Fields (multi-view fusion)**: https://arxiv.org/abs/2309.16118
- **DROID dataset**: https://droid-dataset.github.io/
- **AGD20K**: https://mvig-rhos.com/affordance
- **LOCATE**: https://arxiv.org/abs/2305.11247
- **AffordanceLLM**: https://arxiv.org/abs/2402.16947
- **CLIP**: https://openai.com/research/clip
- **OpenSeeD**: https://github.com/IDEA-Research/OpenSeeD
- **BEHAVIOR-1K**: https://behavior.stanford.edu/
- **Objaverse-XL**: https://objaverse.allenai.org/
- **Voltron**: https://github.com/siddk/voltron
- **OmniGibson**: https://behavior.stanford.edu/omnigibson
- **RT-1 (FiLM init 参考)**: https://arxiv.org/abs/2212.06817
- **RoboPoint**: https://arxiv.org/abs/2406.10721
- **MOKA**: https://arxiv.org/abs/2403.03174
- **PIVOT**: https://arxiv.org/abs/2402.07872
- **CoPa**: https://arxiv.org/abs/2403.08248
- **R3M**: https://arxiv.org/abs/2203.12601

---

## TL;DR 给 Karpathy

最核心的 meta-pattern：**foundation model A 的知识 + foundation model B 的能力 = 自动标注 pipeline → 蒸馏到轻量专用模型**。

UAD 选 A=GPT-4o (semantic world knowledge)，B=DINOv2 (pixel-level features)，目标 = pixel-level task-conditioned affordance map。

工程上最 elegant 的两个 trick：
1. 把 affordance 重写成 VQA，让 VLM 能回答
2. 用 DINOv2 feature similarity 把离散决策摊开成连续 map

下游应用：把 affordance 当 observation space，10 demos 学个 policy 就能 generalize 到 novel objects / categories / instructions，sim 训练 real-world 部署，平均 73% success rate。

---

# UAD: Unsupervised Affordance Distillation 深度解析

Andrej，这篇 paper 的核心 insight 实际上是一个 **knowledge transfer** 的优雅设计：利用 VLM 的 semantic world knowledge 和 LVM 的 pixel-level feature 之间的互补性，把 affordance ground truth 从 foundation model 里"挤"出来，无需人工标注。下面我尽可能详细展开。

---

## 1. Motivation 和 Problem Setting

### 1.1 为什么 affordance prediction 难

Affordance 由 Gibson 在 1977 年提出，定义为 actor 在环境中可感知的 action possibilities。对 robotics manipulation 而言，robot 需要给定一个 free-form instruction（如"grasp the watering can"），输出一个 pixel-level 的连续 affordance map $A \in [0,1]^{H \times W}$，告诉 policy 每个 pixel 是否"affords"该 task。

核心难点在于：
- **标注成本极高**：fine-grained pixel-level affordance mask 极其昂贵
- **task vocabulary open-ended**：closed-set 分类无法 generalize 到 novel instructions
- **sim-to-real gap**：用 sim 数据训练需要 generalize 到 in-the-wild scenes

### 1.2 Insight: Foundation Model 的互补性

VLM（如 GPT-4o）和 LVM（如 DINOv2）各有短板：
- VLM 知道 "handle should be grasped for opening drawers" 这类 affordance knowledge，但 grounding 到 continuous spatial domain 不直接
- LVM 提供 general-purpose pixel-level feature（emergent from self-supervised learning on internet images），但 task-agnostic

UAD 的核心 trick：**把 affordance 重新 formulate 成 visual question answering 问题**——给 VLM 一张标好 colored regions 的图，让它把 region 和 task 关联起来，再用 DINOv2 的 feature similarity 把这个离散决策"摊开"成连续 map。

---

## 2. Affordance 提取 Pipeline 详解

### 2.1 Fine-Grained Region Proposal

**输入**：3D object asset（来自 BEHAVIOR-1K，206 objects / 76 categories；后续 case study 扩展到 Objaverse-XL）

**Step 1: Multi-view rendering**
- 对每个 3D object，在 empty scene 中渲染 K=14 个 views
- 得到 RGB images $I_{i=1}^{K} \in \mathbb{R}^{H \times W \times 3}$ 和 aggregated point cloud $P \in \mathbb{R}^{N \times 3}$（世界坐标系）

**Step 2: DINOv2 feature extraction**
- 用 DINOv2 ViT-L14 with registers（registers 解决了 ViT 的 artifact tokens 问题，参考 Darcet et al. 2023）
- 对每张 $I_i$ 提取 patch-wise features $F_i \in \mathbb{R}^{H \times W \times d}$，d=1024
- Bilinear interpolation upsample 到原图分辨率

**Step 3: Multi-view feature fusion**（参考 D3Fields, Wang et al. 2023）

对每个 3D point $p \in P$，对每个相机视图：
- 通过 camera projection 找到 p 的对应 pixel
- 检查 projection depth 与 depth image reading 是否接近（threshold），若是则视为 visible
- Fused feature = 所有 visible views 的 DINOv2 features 平均

得到 global 3D feature field：
$$F_{\text{global}} \in \mathbb{R}^{N \times d}$$

**Step 4: PCA 降维**
$$F_{\text{reduced}} = \text{PCA}(F_{\text{global}}) \in \mathbb{R}^{N \times 3}$$

为什么要降到 3D？作者 empirical 发现 PCA 让 features less sensitive to local texture，从而 clustering 更稳定。这点挺有意思——DINOv2 的 1024 维 feature 包含大量 texture 信息，对 region segmentation 是噪声。

**Step 5: Clustering**
- 默认 Mean Shift clustering（自动决定 cluster 数 M）
- 若 Mean Shift 找到 < 5 clusters，fallback 到 k-means with k=5
- 对 articulated objects（如 cabinet），对每个 link 单独跑 clustering pipeline，这样能找到 drawer knob 这种 fine-grained region

最终得到 region labels $r_{n=1}^{N} \in \{1, ..., M\}$

### 2.2 Task Instruction Proposal

**Step 1: 找 natural view**
- 用 CLIP 计算 object category name 与每个 view image 的 cosine similarity
- 选最 natural-looking 的 view 作为 VLM 的输入

**Step 2: Region visualization**
- 对选定 view 的每个 foreground pixel，通过 depth map 反投影到 3D，找 P 中最近邻 point，使用其 region label $r_p$
- 给每个 cluster 分配 unique color，overlay 到原图

**Step 3: VLM query**

给 GPT-4o 的输入包括：
1. 原始 RGB 图
2. Region overlay 图（每个 cluster 涂不同颜色）
3. Color list
4. Object category name

System prompt 要求 VLM：
- 提出 region description 和对应的 task instruction，格式如 `"handle of plastic bag -- region for agent to hold and lift the bag"`
- Match 每个 task 与一个 colored region
- 输出格式是 Python `ast.literal_eval()` 可 parse 的 dict

例如 coffee mug，GPT-4o 可能输出：
```python
{
    "rim of the coffee mug -- region for drinking and pouring": "Red",
    "handle of the mug -- region for grasping": "Blue"
}
```

### 2.3 Region-to-Instruction Mapping 到连续 Affordance

**关键 insight**：VLM 输出离散 region 决策，但 affordance 应该是 continuous。某些区域更 "tightly" 关联到 task（如 handle 中间 vs handle 末端），continuous 形式更好 capture。

**Step 1: Reference feature 计算**
对每个 VLM 选定的 region r：
$$f_{\text{ref}} = \frac{1}{|r|} \sum_{p \in r} F_{\text{global}}(p) \in \mathbb{R}^d$$

即该 region 内所有 3D points 的 DINOv2 features 平均。

**Step 2: Per-point similarity score**
对每个 3D point p：
$$s(p) = \max\left(0, \cos(f_{\text{ref}}, F_{\text{global}}(p))\right) \in [0,1]$$

其中 $\cos(\cdot, \cdot)$ 是 cosine similarity。这个操作的 intuition 是：与 reference region feature 越相似的 point，affordance 越高。由于 DINOv2 features 已经 semantic-aware，相似 region（如整个 handle）会得到高分，而 dissimilar region（如 cup body）得到低分。

**Step 3: Project 回 2D**
对每个相机 view 的每个 pixel，通过 depth 反投影到 3D，找 P 中最近邻 point，assign 其 score：
$$A_{h,w} = s(\arg\min_{p \in P} \| \text{proj}(h,w) - p \|)$$

得到 final affordance map $A \in [0,1]^{H \times W}$

**Step 4: Post-processing**
- Threshold 0.5：$A_{h,w} \leftarrow 0$ if $A_{h,w} < 0.5$
- Gaussian blur，kernel size = 3（smooth threshold 边界，稳定 training）

### 2.4 为什么 over-segmentation > under-segmentation

作者发现 clustering over-segment 比 under-segment 好。Intuition：
- Over-segment 时，GPT-4o 仍能正确选对应 region；由于 cosine similarity，同 part 的其他 cluster 仍会得到高分（因为 DINOv2 features 相似）
- Under-segment 时，可能把 functional 不同的 parts 合并，导致 affordance map 错误地 highlight 不相关 region

---

## 3. Task-conditioned Affordance Model 架构

### 3.1 整体设计

**设计理念**：
- **Frozen DINOv2 backbone**：保留 pre-trained feature 的 generalization 到 real-world scenes
- **Lightweight language-conditioned decoder**：只训练少量参数，避免 overfit 到 sim 渲染

### 3.2 FiLM (Feature-wise Linear Modulation) 详解

FiLM 是 UAD 的灵魂组件，来自 Perez et al. 2018 (AAAI)。

**输入**：
- Language embedding $e_T \in \mathbb{R}^{d_L}$（来自 OpenAI API，可能 text-embedding-3-large，$d_L \approx 3072$）
- Pixel-space features $\bar{X} \in \mathbb{R}^{H \times W \times C_{\text{in}}}$（来自 frozen DINOv2 或前一层）

**核心 operation**：

对每个 channel $c \in \{1, ..., C_{\text{out}}\}$：
$$\gamma_c = W_\gamma^c \cdot e_T + b_\gamma^c$$
$$\beta_c = W_\beta^c \cdot e_T + b_\beta^c$$

其中 $W_\gamma^c \in \mathbb{R}^{d_L \times C_{\text{in}}}$, $W_\beta^c \in \mathbb{R}^{d_L \times C_{\text{in}}}$ 是 linear layer 参数。

然后对每个 pixel $(h,w)$：
$$X'_{h,w,c} = \sum_{c'=1}^{C_{\text{in}}} \gamma_{c,c'} \cdot X_{h,w,c'} + \beta_c$$

注意：$\gamma_c, \beta_c$ 只 depend on $e_T$，**与 pixel 位置无关**。这就是 FiLM 的关键——transformation 是 location-agnostic 的，per-channel 共享。

**为什么 FiLM 适合这里**？

作者的解释是：FiLM 的 location-agnostic transformation 适合建立 DINOv2 feature 与 task instruction 的 association。直觉上：
- DINOv2 feature 已经 encode 了 "what's at this pixel"
- FiLM 通过 language-conditioned scaling/shifting 选择性地 amplify 与当前 task 相关的 feature channels
- 整个 pixel space 共享同样的 "selection pattern"，类似 attention 但更轻量

### 3.3 完整 Decoder 架构

3 个 FiLM-conditioned convolution layers：

| Layer | Input channels | Output channels |
|-------|----------------|-----------------|
| FiLM-1 | 1024 (DINOv2) | 256 |
| FiLM-2 | 256 | 64 |
| FiLM-3 | 64 | 1 |

最后一层输出 $\hat{A} \in [0,1]^{H \times W}$（logits，经 sigmoid）。

### 3.4 Loss Function

Binary Cross-Entropy：
$$\mathcal{L} = -\frac{1}{HW} \sum_{h=1}^{H} \sum_{w=1}^{W} \left[ A_{h,w} \log \hat{A}_{h,w} + (1 - A_{h,w}) \log(1 - \hat{A}_{h,w}) \right]$$

变量：
- $A_{h,w} \in [0,1]$：ground truth affordance at pixel (h,w)（注意：这里 GT 实际是 continuous，不是 binary，但用 BCE 而非 MSE 是因为 BCE 对 middle values 梯度更稳定）
- $\hat{A}_{h,w} \in [0,1]$：predicted affordance

### 3.5 Training 实现细节

- Linear layer 初始化：weight=1, bias=0（参考 RT-1）
- Optimizer: Adam, lr=0.001
- 30 epochs, batch size 8
- 12 hours on single NVIDIA A6000

**对比**：DINOv2 ViT-L14 有 300M parameters，但只训练 FiLM layers，参数量大约 1M 量级，efficient 且 less prone to overfitting。

---

## 4. Policy Learning with Affordance

### 4.1 架构

基于 RVT (Robotic View Transformer, Goyal et al. 2023)：
- Multi-view transformer policy
- 每个 view 的输入：UAD affordance map + depth + (x,y,z) world coords + global proprioception
- Output: 7-dim action = 6-DoF end-effector pose + binary gripper action

### 4.2 关键设计决策

- **Affordance 作为 observation space**：而非作为 auxiliary loss 或 pretraining signal。这把 affordance 直接变成 policy 的"眼睛"，使 policy 聚焦于 task-relevant regions。
- **不 finetune affordance model**：即便在 policy training 阶段也 frozen。这保留了 UAD 的 generalization 能力，避免 policy 在小数据（10 demos）上 overfit 破坏 affordance 表示。

### 4.3 为什么 10 demos 就够？

通常 vision-based imitation learning 需要大量 demos 来学习 visual representation。UAD 把 representation learning 解耦——pre-trained affordance 已经 task-conditioned 且 fine-grained，policy 只需学 motion-level mapping。

这与 R3M、Voltron、CLIP 等方法对比的优势：CLIP 是 "bag-of-words" 行为，对 fine-grained visual detail 不敏感；DINOv2 是 task-agnostic，policy 还需要 learn to attend。UAD 直接把 attention built-in 了。

---

## 5. 实验结果深度分析

### 5.1 Affordance Prediction 评估

#### Sanity check（sim 数据）

| Setting | AUC |
|---------|-----|
| Training data | ≥ 0.92 |
| Novel instances | ≥ 0.92 |
| Novel categories | ≥ 0.92 |
| Novel instructions | ≥ 0.92 |

100 个 <instruction, affordance> pairs per setting，MTurk 标注 ground truth。这个数字表明 UAD 学到的不是 dataset-specific 的 spurious correlation。

#### DROID dataset（real-world robotic scenes）

| Method | AUC |
|--------|-----|
| CLIP | 0.500 |
| OpenSeeD | 0.836 |
| **UAD (Ours)** | **0.840** |

关键观察：
- CLIP 表现差（0.5 接近 chance）——印证 CLIP 是 image-level "bag-of-words"，pixel-level 表现差
- OpenSeeD（open-vocabulary segmentation）有竞争力，但 UAD 略胜——因为 UAD 输出 continuous 表示，且对小 region 更稳健（segmentation model 在小 region 上常失败）

#### AGD20K (human activity affordances)

| Method | KLD ↓ | SIM ↑ | NSS ↑ | NSS-0.5 ↑ |
|--------|-------|-------|-------|-----------|
| Cross-View-AG | 1.787 | 0.285 | 0.829 | - |
| LOCATE | 1.405 | 0.372 | 1.157 | 1.723 |
| 3DOI | 3.565 | 0.227 | 0.657 | - |
| AffordanceLLM | 1.463 | 0.377 | 1.070 | - |
| **UAD (Ours)** | 1.878 | **0.407** | 1.092 | **2.050** |

Metric 解释：
- **KLD (KL Divergence)**：measure 分布差异。UAD 的 KLD 高（差），但作者解释是 AGD20K 的 GT 是 Gaussian mixture 围绕 keypoints，会 diffuse 到 background；UAD 是 fine-grained 在 background 预测 0，被惩罚。
- **SIM (Similarity)**：intersection over union 的 saliency 版本。UAD 最高。
- **NSS (Normalized Scanpath Saliency)**：在 GT > 0.1 的 fixation points 上的 mean predicted saliency（z-scored）。
- **NSS-0.5**：用 0.5 而非 0.1 作为 threshold，更严格。UAD 在此 metric 上 SOTA。

**Intuition**：UAD 在 AGD20K 上 competitive 但不是全面 SOTA。原因可能是 UAD 只在 sim 的 single objects 上训练，泛化到 human activities 是 zero-shot。但能 generalize 到 "eating bananas"、"taking photos" 这种完全 OOD 的 activities 已经很 impressive。

### 5.2 Simulation Policy Generalization

3 个 tasks × 4 个 generalization settings × 15 trials = 每个 task 60 trials。

**Baselines**: vanilla RGB, DINOv2, CLIP, Voltron

主要 takeaways：
1. **Object appearance 鲁棒**：例如只在黑色 marker 上训练，能成功 manipulate 白色 marker
2. **Fine-grained perception 优势**：在 Opening task（检测 thin drawer handles）上 UAD 明显胜出
3. **Instruction generalization**：能通过 language 控制 target object（如 "pouring fluid" vs "watering plants"）

### 5.3 Real-world Policy

| Task | Success Rate |
|------|--------------|
| Watering plant | (avg) |
| Opening drawer | 73% |
| Inserting pen | (avg) |
| **Average** | **73%** |

10 demos, kinesthetic teaching, Franka Emika Panda，2 个 RGB-D cameras (Orbbec Femto Bolt)。

---

## 6. Intuition 和 Insights

### 6.1 为什么从 sim single object 训练能 generalize 到 real-world multi-object？

1. **Frozen DINOv2 backbone**: DINOv2 在 LAION 等互联网数据集 self-supervised pretrain，features 已经 robust to real-world variations
2. **Lightweight decoder**: 只训练 FiLM layers（~1M params），不会破坏 pre-trained representations
3. **3D consistency 训练**: 通过 multi-view fusion + clustering 学到的是 functional regions，而非 view-specific artifacts
4. **Feature similarity-based label**: 不直接用 GPT-4o 输出作 label，而是用 DINOv2 feature similarity "spread" 出来，这让 label 与 DINOv2 feature space 对齐

### 6.2 为什么 cosine similarity 比 binary mask 好？

Binary mask 假设 region 内所有 pixel 等价。Cosine similarity 让"中心"区域（feature 最接近 reference）得分高，"边缘"得分低。这更符合 affordance 的 continuous nature——handle 中间最适合 grasp，handle 末端次之。

### 6.3 为什么 FiLM 比 cross-attention 更适合？

Cross-attention 会让每个 pixel 的 feature 与 language token 交互，参数多且容易 overfit。FiLM 是 location-agnostic 的 channel-wise modulation，相当于在 feature space 做一个 task-conditioned linear projection，更 parameter-efficient 且语义清晰。

### 6.4 与 LOCATE / AffordanceLLM 的本质区别

- LOCATE: weakly supervised，需要 image-level labels
- AffordanceLLM: 直接用 VLM 输出 affordance，inference 时仍依赖 VLM
- UAD: 用 VLM 做"老师"自动标注，蒸馏成轻量模型；inference 时不依赖 VLM，efficient

---

## 7. Limitations

1. **Single static frame**：不考虑 video 或 multi-step affordance reasoning
2. **Single object rendering**：training 数据只有 single objects，多 object scene 是 zero-shot 泛化
3. **No motion-level generalization**：affordance 是 where，不解决 how
4. **Dependence on GPT-4o**：data quality 受 VLM 能力限制；如果 VLM hallucinate region matching，会传播错误

---

## 8. 与其他工作的关系

### 8.1 同时代相关方法

- **CoPa** (Huang et al. 2024): spatial constraints of parts with foundation models
- **MOKA** (Liu et al. 2024): mark-based visual prompting for VLMs
- **PIVOT** (Nasiriany et al. 2024): iterative visual prompting
- **RoboPoint** (Yuan et al. 2024): VLM for spatial affordance prediction

这些方法都 inference 时调用 VLM，慢且贵。UAD 把 VLM 知识蒸馏到小模型，inference 快。

### 8.2 Pre-trained visual representation for manipulation

- **R3M, Voltron, LIV, CLIP** 等：都学 task-agnostic 或 task-conditioned representation
- **UAD 不同**：直接输出 task-conditioned affordance map，是 explicit 的 visual attention 而非 implicit feature

### 8.3 DINOv2 系列

- **D3Fields** (Wang et al. 2023): 同样用 multi-view DINOv2 fusion 到 3D
- **GenDP** (Wang et al. 2024): 3D semantic fields for diffusion policy
- **Find Any Part in 3D** (Ma et al. 2024): open-vocabulary 3D part segmentation

---

## 9. Reference Links

- **UAD project page**: https://unsup-affordance.github.io/
- **DINOv2**: https://dinov2.metademolab.com/ | paper: https://arxiv.org/abs/2304.07193
- **DINOv2 with registers**: https://arxiv.org/abs/2309.16588
- **GPT-4o**: https://openai.com/gpt-4o
- **FiLM paper**: https://arxiv.org/abs/1709.07871
- **RVT (Robotic View Transformer)**: https://arxiv.org/abs/2306.13096 | project: https://rvt-2.github.io/
- **R3Fields (multi-view fusion 参考)**: https://arxiv.org/abs/2309.16118
- **DROID dataset**: https://droid-dataset.github.io/
- **AGD20K / Affordance grounding**: https://mvig-rhos.com/affordance
- **LOCATE**: https://github.com/locate-parts/locate-parts | paper: https://arxiv.org/abs/2305.11247
- **AffordanceLLM**: https://arxiv.org/abs/2402.16947
- **CLIP**: https://openai.com/research/clip
- **OpenSeeD**: https://github.com/IDEA-Research/OpenSeeD
- **BEHAVIOR-1K**: https://behavior.stanford.edu/
- **Objaverse-XL**: https://objaverse.allenai.org/
- **Voltron**: https://github.com/siddk/voltron
- **OmniGibson**: https://behavior.stanford.edu/omnigibson
- **R3M**: https://arxiv.org/abs/2203.12601
- **RoboPoint**: https://arxiv.org/abs/2406.10721
- **MOKA**: https://arxiv.org/abs/2403.03174
- **PIVOT**: https://arxiv.org/abs/2402.07872
- **CoPa**: https://arxiv.org/abs/2403.08248
- **RT-1 (FiLM init 参考)**: https://arxiv.org/abs/2212.06817

---

## 10. Final Thoughts

Andrej，从你的视角看，UAD 最有意思的点可能是：

1. **Distillation as annotation**: 不用人工标注，用 foundation model 之间的互补性"互相教"——VLM 教 LVM semantic task knowledge，LVM 教 VLM continuous spatial structure。这是一个 general pattern，可以推广到很多类似问题。

2. **Frozen backbone + lightweight head**: 在数据稀缺的 robotics 领域，这是 sample efficiency 的关键。10 demos 能 generalize 到 novel categories 说明 representation 占了大头。

3. **3D consistency trick**: 即使最终是 2D 输出，从 3D 渲染训练能 enforce view-consistency，让 model 学到 object-centric 的 functional region 而非 view-specific patterns。这个 trick 在 sim-to-real 中可能 underrated。

4. **Feature similarity as soft label**: 把 VLM 的离散决策"摊开"成连续 map，用 DINOv2 feature similarity 作 smooth proxy——这是一个非常 elegant 的工程化 trick，避免了 binary mask 的硬边界问题。

潜在 extension 方向：
- Video affordance: 给定 task，预测 affordance trajectory
- Multi-object interaction: 在场景级别提取 relational affordance
- Affordance-based RL reward: 用 affordance map 作 dense reward signal
- Bimanual manipulation: 两只手各自的 affordance

这篇 paper 体现了一个有用的 meta-pattern：**用 foundation model 做 automatic labeling pipeline，然后蒸馏到轻量专用模型**。这个范式在 robotics、vision、language 各领域都越来越普遍。
