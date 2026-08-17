---
source_pdf: OPENTOUCH.pdf
paper_sha256: 1f262c3ca068996b46f09f31e7a56b06a50fa830b2035a5506a6c3b0e123d1c1
processed_at: '2026-08-06T01:00:26-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# OPENTOUCH 用人话说

## 一句话总结

一群人做了个手套，能感觉你摸东西的压力，又配了个眼镜摄像头，去各种真实场景里录人摸东西，然后说"大家来研究怎么把视觉和触觉对齐起来吧"。

---

## 背景为什么 important

你看 Ego4D 那些 egocentric dataset，几千小时的视频，人做饭、修东西、玩球，全都有。但有个根本问题：**视频里你根本看不出手用了多大力**。

举个 paper 里的例子 (Figure 3a)：一个手搭在椅子上，前几帧和后几帧手势几乎一模一样，但 tactile 显示第三帧手突然用力了——椅子被推动了。你光看 RGB 帧根本分不出来。

更狠的例子 (Figure 3d)：中指双击 button。这个动作太细微了，连 Rokoko 这种专业 motion capture glove 都抓不到，只有 tactile 能看到那一下压力尖峰。

所以这群人就想：**能不能去真实世界里，把手摸东西的感觉录下来，跟视觉对齐？**

---

## 硬件怎么做的

三件套：

### 手套本体 (tactile)

一个 16×16 的 electrode grid，包在 piezoresistive film 外面，一共 169 个 pressure sensor (taxel) 覆盖整个手掌和手指。

为什么这么设计？之前 STAG (Nature 2019) 用的是导电布料，得专门编织机器做，复现困难。这里改用 FPC (flexible printed circuit)，就是手机里那种柔性电路板，标准 PCB 工厂就能批量做，便宜、可复制。

压力范围 0.02–50 kPa，用 ESP-NOW 无线传，延迟 2ms。

### 手势手套 (pose)

直接买 Rokoko Smartgloves，商业产品，7 个 IMU per hand，精度 ±1°。30Hz streaming。这玩意儿不需要自己造，现成的够用。

### 眼镜摄像头 (vision)

Meta Project Aria，1408×1408 RGB，30fps，110° FOV。还带 eye tracking、IMU、麦克风，但 paper 主要用 RGB。

### 同步怎么搞

简单粗暴：terminal 打个 visual cue，眼镜拍到这一帧，拿这帧的时间戳当 reference point，其他 stream 往这个时间点对齐。不是 hardware trigger 那种精确同步，但 in-the-wild 够用。

---

## 数据怎么采的

**14 个场景**：厨房、车间、办公室、车库、浴室等等。让被试者去每个地方随便摸随便玩，没脚本。右手戴手套，左手不管。

最后得到：
- 5.1 小时同步录制
- 2900 个 clip，平均每个 57 帧
- 800+ 个 object，14 个 category
- 29 种 grasp type (GRASP taxonomy)

### 标注怎么 scale

这是关键问题。手动标 2900 个 clip 太贵，尤其在野外的场景里 object 五花八门。

他们的 trick：**用 GPT-5 自动标**。

具体做法很聪明。每个 clip 不送全部视频给 GPT，只送 3 帧：
1. **Onset frame**: 接触前压力最低那一帧 (手刚要碰到)
2. **Peak frame**: 压力最高那一帧 (用力最大)
3. **Post-peak frame**: 接触后压力最低那一帧 (松开)

每帧给 RGB + tactile heatmap 两个图。GPT 看 6 张图，输出 object name、category、environment、action、grasp type、自然语言描述。

为什么这 3 帧够了？因为 manipulation 的 essence 在 "approach → squeeze → release" 这个 arc 里。3 帧就是 arc 的三个 critical point。

人工验证后准确率大概 90%。失败的 case 主要是手离开视野、光线太差、3 帧上下文不够判断。

---

## Benchmark 设计了啥

两个 task：

### Task 1: Cross-modal retrieval

给一个 modality，能不能在数据库里 retrieve 出对应的另一个 modality。

比如 Video → Tactile: 给你 20 帧 RGB，你能不能在 2900 个 clip 里找到对应的 tactile sequence。

测了好多方向：
- Video ↔ Tactile (双向)
- Tactile ↔ Pose (双向)
- Video + Pose → Tactile (双模态 query 单模态)
- Tactile + Pose → Video
- Video + Tactile → Pose

Metric 用 Recall@1/5/10 和 mAP。

### Task 2: Tactile classification

两个独立任务：
1. **Action recognition**: 手在干啥 (picking up, pressing, rotating...)
2. **Grasp type recognition**: 手怎么握的 (Medium Wrap, Prismatic Two-Finger, Palmar...)

为什么分两个？因为 action "pointing" 对应 Index-Finger Extension grip，但 "picking up" 可能对应多种 grip。Action 和 grasp 是 decoupled 的 label space。

---

## 方法上几个有意思的发现

### 发现 1: Tactile 不需要大 model

这是个反直觉的结论。他们试了两个 tactile encoder：
- **Lite-CNN**: 3 层 CNN (32 filter, 5×5 kernel) + 2 层 BiGRU
- **ResNet-18**: 标准 ImageNet backbone，把 16×16 上采样到 224×224

结果 Lite-CNN 在 video→tactile retrieval 上 mAP 16.76%，ResNet-18 只有 6.27%。差 10 个点。

**为什么**？Tactile map 不是 natural image。它 sparse (大部分 taxel 是 0)、spatial structure 是 localized blob (不是 hierarchical texture)、没有 scale invariance 需求。ResNet 那套 multi-scale feature hierarchy 是为 ImageNet 设计的，套到 16×16 的 pressure map 上就是 over-parameterization，反而引入 noise。

这个结论对做 multimodal learning 的人有启发：**不同 modality 需要不同 inductive bias**。Vision backbone 不能无脑 port 到 tactile/audio/IMU。

### 发现 2: Temporal window 越长越好

5 帧 → 10 帧 → 20 帧，Video+Pose→Tactile 的 mAP 从 19.89% → 15.28% → 24.46% (中间 10 帧那个数字有点 weird，可能 noise，但 20 帧最好很清楚)。

为什么？Tactile contact 是个 dynamic process。单帧你只看到一个 static pressure snapshot，你不知道手是正在压下去还是正在松开。20 帧 (0.67 秒) 才能 capture 整个 grasp 的 temporal arc。

这和 GPT 标注那 3 帧 (onset/peak/post-peak) 的 intuition 一致：manipulation 的 semantics 在 trajectory 里，不在 snapshot 里。

### 发现 3: Tactile 对 grasp 最 informative，vision 对 action 最 informative

Table 3 数字很说明问题：

| 模态 | Action Acc | Grasp Acc |
|------|-----------|-----------|
| Vision only | **40.26%** | 57.45% |
| Pose only | 33.22% | 46.32% |  
| Tactile only | 29.95% | **60.23%** |

**Action 看 vision 最好**：因为 action (picking up, pressing, rotating) 依赖 global context — 你得看到 object 是什么、场景是啥、手整体轨迹。
**Grasp 看 tactile 最好**：因为 grasp 本质是 local contact geometry — 手指怎么分布、哪里用力、contact patch 在哪。这些信息 vision 看不到 (被手遮挡)。

### 发现 4: 三模态融合 > 双模态 > 单模态

Tri-modal retrieval (Table 2b) 的 mAP 在 23-27% 区间，比任何 bi-modal 都高。

为什么三者互补？
- **Vision** 给 global scene + object identity
- **Pose** 给 fine-grained kinematics (手指关节角度)
- **Tactile** 给 local contact + force magnitude

三者都有的话 ambiguity 最少。比如光看 video 你不知道手在推还是只是搭着，加上 tactile 就知道有没有力；光看 tactile 你不知道摸的是啥，加上 vision 就知道 object identity。

---

## InfoNCE loss 怎么 work 的

他们用的是 CLIP-style contrastive learning。公式 (Eq 1)：

$$\mathcal{L}_{ab} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\langle z_a^{(i)}, z_b^{(i)} \rangle / \tau)}{\sum_{j=1}^{B} \exp(\langle z_a^{(i)}, z_b^{(j)} \rangle / \tau)}$$

人话讲：
- Batch 里有 B 个样本，每个样本有 modality A 的 embedding $z_a^{(i)}$ 和 modality B 的 embedding $z_b^{(i)}$
- 分子：第 $i$ 个样本的 A 和它自己对应的 B 的相似度 (positive pair)
- 分母：第 $i$ 个样本的 A 和 batch 里所有 B 的相似度加起来 (1 个 positive + B-1 个 negative)
- $\tau = 0.07$ 是 temperature，控制 softmax 的 sharpness
- Loss 让 positive pair 的相似度相对所有 negative 高

Symmetric 就是两个方向都算：A→B 和 B→A，避免 embedding space 偏向某一模态。

Tri-modal 的时候 (Eq 3)，先把两个模态的 embedding concat 起来过个 linear head $\phi$：
$$z_f = \phi([z_a; z_b])$$

然后 $z_f$ 和第三个模态 $z_c$ 之间做 InfoNCE。

---

## 应用：给 Ego4D 加触觉

这个 application 我觉得很有 vision。Ego4D 有 3000 小时 egocentric video，但没有 tactile。

他们做了个 zero-shot experiment：拿 Ego4D 的 video query，从 OPENTOUCH 数据库里 retrieve 最相似的 tactile sequence。Figure 6 显示 retrieve 出来的 tactile 对应的 source video，hand motion 和 object geometry 跟 query 很像。

这意味着什么？**OPENTOUCH 可以当 tactile database**，给大规模 egocentric video "augment" 触觉信息。虽然不是真实的 tactile，但能 infer 出 "这段视频里手大概在用这种 grasp pattern，contact force 分布是这样"。

这个方向继续做下去，可能让 Ego4D 这种纯视频 dataset 变成 "video + pseudo-tactile" dataset，对机器人 learning from demonstration 很有用。

---

## 局限性 paper 自己承认的

1. **只测 normal pressure**：piezoresistive sensor 测不了 shear force、vibration、temperature。所以 slip detection、texture recognition、冷热感知都做不了。
2. **16×16 分辨率太粗**：fingertip 上 contact patch 小，16×16 grid 定位不够精确。
3. **只有右手**：bimanual manipulation (双手协作) 完全 missing。左手靠 mirroring 假设，但没验证过。
4. **Durability**：FPC 在 finger joint 处反复弯，加上汗、穿脱摩擦，铜线会断。Bench test 10k cycles，真实使用寿命更短。
5. **Scale 小**：5.1 小时 vs Ego4D 3000 小时。但这是 hardware-limited，手套贵且易坏。

---

## 我觉得对做 robotics / multimodal learning 的人有什么启发

### Tactile signal 的 "compact yet powerful" 特性

Paper 反复强调这个点。Tactile 只有 16×16 = 256 个 scalar per frame，但 grasp classification 准确率 60%，接近 vision 的 57%。而 tactile encoder 参数量比 DINOv3 ViT-B 小 1000 倍。

为什么？因为 tactile 是 physics 的 direct readout。你手压在物体上，压力分布就是 hand-object interaction 的 direct physical measurement。Vision 是这个物理过程的 indirect projection (光线反射 → camera sensor)。对于 "手怎么握的" 这种 local geometry 问题，direct signal 更 efficient。

这给 robotics 一个 hint：**don't over-rely on vision**。很多 manipulation task 里，一个 cheap tactile sensor 可能比 expensive camera + heavy vision model 更 effective。

### 不同 modality 需要不同 architecture

Lite-CNN (3 层) 击败 ResNet-18 (18 层) 这个结果很 striking。这说明 ImageNet-pretrained 的 inductive bias (hierarchical edge/texture detector) 对 tactile map 是有害的。

如果你做 multimodal learning，不要无脑把 vision backbone 套到所有 modality 上。Tactile、audio、IMU、EEG 这些 signal 的 statistics 各不相同，需要各自设计合适的 encoder。

### Temporal context 是 manipulation 的关键

20 帧 > 10 帧 > 5 帧。Manipulation 的 semantics 在 temporal trajectory 里，不在 single snapshot。这对你做 robotics policy 也有启发：single-frame observation 不够，需要给 policy 足够的 temporal history。

---

## 相关链接

**项目主页**: https://opentouch-tactile.github.io/

**硬件参考**:
- STAG (Nature 2019): https://www.nature.com/articles/s41586-019-1234-z  
- Conformal Tactile Textiles (Nature Electronics 2021): https://www.nature.com/articles/s41928-021-00545-0
- Rokoko Smartgloves: https://www.rokoko.com/products/smartgloves
- Meta Project Aria: https://www.projectaria.com/research-kit/

**Cross-modal learning**:
- CLIP: https://arxiv.org/abs/2103.00020
- ImageBind: https://arxiv.org/abs/2205.01377
- UniTouch: https://arxiv.org/abs/2402.14355

**Egocentric datasets**:
- Ego4D: https://ego4d-data.org/
- Ego-Exo4D: https://ego-exo4d-data.org/

**Grasp taxonomy**:
- GRASP Taxonomy paper: https://ieeexplore.ieee.org/document/7299224

---

## 最后吐槽一下

Paper 整体质量不错，硬件设计 practical，dataset 填补了真实空白，benchmark 设计合理。几个我觉得可以更好的地方：

**Baseline 太弱**。Retrieval 只对比 CCA/PLSCA 这种 2010 年代的 linear method。为什么不 finetune ImageBind 或 UniTouch 比？这两个都是 2023-2024 的 multimodal alignment SOTA，对比才有说服力。

**Scale 太小**。5.1 小时虽然够发 paper，但离 "foundation dataset" 差远了。作者也承认 hardware limited。下一步如果能把成本降到 $50/glove，采 500 小时，impact 会大得多。

**Single hand 是大问题**。真实 manipulation 里 bimanual 占比很高。只采右手然后说 "左手靠 mirroring" 是 hand-wave。bimanual coordination 的 tactile pattern 跟单手完全不同 (比如两只手配合拧瓶盖)。

**没有 dynamic tactile**。只有 normal pressure，没有 shear、vibration、temperature。这让 tactile 的 application scope 受限。 Slip detection 对 manipulation 太重要了，但这个 dataset 做不了。

但话说回来，作为第一个 in-the-wild full-hand tactile dataset，这个工作 opens up 了整个领域。硬件 open-source + 低成本 + 可复制，社区可以接着往下做。这种 "enabling" 类型的 paper 比那种 claim 解决所有问题的 paper 有价值得多。

---

# OPENTOUCH 深度讲解

## 1. Paper 的核心问题和动机

OPENTOUCH 要解决的核心问题：现有的 egocentric perception 缺少 tactile 这一个 modality。人类 manipulation 中，vision 提供全局 scene context，但 contact 时刻、contact location、force magnitude 这些信息 vision 基本无法获取，尤其是当 object 透明、反光、或者 hand 被 occlude 的时候。

现有 dataset 的 gap：
- **GRAB** (ECCV 2020) — MoCap + analytical contact，但 lab-only，51 objects
- **ContactDB** (CVPR 2019) — thermal imaging，50 objects，1 environment
- **STAG** (Nature 2019) — pressure glove，26 objects，lab setting
- **EgoPressure** — RGB pressure estimation，但 single environment
- **HOI4D** — in-the-wild but no tactile

OPENTOUCH 是第一个同时满足 in-the-wild + full-hand pressure + hand pose + egocentric video 的 dataset。Table 1 展示了与 12 个 prior dataset 的对比，OPENTOUCH 在所有维度上都打了勾。

项目主页：https://opentouch-tactile.github.io/

---

## 2. Hardware Setup — 为什么这套系统能 work

### 2.1 Tactile Sensing Glove (FPC-based)

这是 paper 的硬件核心创新。设计哲学：**PCB-level precision + wearable compliance**。

**5-layer 结构** (Figure 11)：
```
top silicone encapsulation
top FPC (16×16 electrode grid)
piezoresistive film (commercial)
bottom FPC
bottom silicone encapsulation
```

关键参数：
- **169 taxels** uniformly covering fingers + palmar surface
- **16×16 electrode grid** routed around piezoresistive film
- **Pressure range**: 0.02–50 kPa (calibrated)
- **Wireless**: ESP-NOW protocol，平均 latency 2ms
- **Zero-potential readout circuit** 减少寄生电容干扰

为什么 FPC 比 conductive textile 好：
- STAG (Nature 2019) 用 conductive textile，需要 specialized knitting machine，reproducibility 差
- FPC 用 standard PCB fabrication，可以 batch production，每个 glove 成本低
- 自动 layout 可以 adapt 不同 hand size

相关 prior work:
- FIT-Glove (CHI 2025): https://dl.acm.org/doi/10.1145/3706538.3713486
- Conformal Tactile Textiles (Nature Electronics 2021): https://www.nature.com/articles/s41928-021-00545-0

### 2.2 Hand-tracking Glove (Rokoko Smartgloves)

- 7 个 6-DOF sensors per glove (IMU + EMF fusion)
- 30 Hz streaming
- **Rotational accuracy**: ±1°
- Calibration pose: standing upright, elbows 90° bent

为什么选 Rokoko 而不是 optical MoCap：wearable form factor 支持 in-the-wild，不需要 external cameras。

参考：https://www.rokoko.com/products/smartgloves

### 2.3 Egocentric Video (Meta Project Aria)

Profile 28 配置 (Table 4)：
| Sensor | Resolution | FPS |
|--------|-----------|-----|
| RGB camera | 1408×1408 | 30 |
| SLAM cameras (×2) | 640×480 | 30 |
| Eye tracking | 320×240 | 60 |
| IMU | - | 800-1000 Hz |

Field of view: 110°，几乎覆盖整个 manipulation workspace。

参考：https://www.projectaria.com/research-kit/

### 2.4 Time Synchronization

方法很简洁：terminal 显示 visual cue → Aria RGB camera 记录 cue frame → 取该 frame 的 device-clock timestamp 作为 reference → 找到 tactile 和 pose stream 中最近 sample → 减去 matched sample time + 加 reference timestamp。

这种 soft-sync 方案虽然比 hardware trigger 精度低，但在 in-the-wild 场景下 practical。

---

## 3. Dataset 构建和 Annotation Pipeline

### 3.1 Collection Protocol

- **14 environments**: kitchen, workshop, office, garage, bathroom 等
- **800+ objects** across 14 categories
- **Right hand only** instrumented (simplify hardware + standardize annotation)
- Unscripted manipulation → 自然产生 power, precision, pinch, lateral, palmar grasps
- **5.1 hours** total recording，**2,900 curated clips** (平均 57 frames/clip)

### 3.2 GRASP Taxonomy

使用 Feix et al. 2015 的 33 类 grasp taxonomy，最终 dataset 覆盖了 29 类 (Figure 9)。每个 grasp type 的 accumulated tactile map (Supp C.1) 显示了非常强的 spatial pattern correlation，这验证了 tactile data 的 quality。

参考：https://ieeexplore.ieee.org/document/7299224

### 3.3 GPT-5 自动标注

这是 scaling 的关键。Manual annotation 在 in-the-wild 场景下 cost 太高。

**Sample strategy**: 基于 pressure dynamics 选取 3 帧
1. Frame 1 (Onset): pre-peak lowest pressure (approach)
2. Frame 2 (Peak): peak pressure (max manipulation force)
3. Frame 3 (Post-peak): post-peak lowest pressure (release)

这个 sampling strategy 的 intuition：manipulation 的 most informative moment 发生在 peak contact force 附近，3 帧足以 capture 整个 interaction arc。

**Prompt 设计** (Supp E.1)：
- 给 GPT-5 提供 3 个 RGB-tactile pairs
- Predefined label sets for object category, environment, action, grasp type
- 要求 GPT-5 先在 `<thinking>` tags 里做 frame-by-frame analysis
- Output JSON with 6 fields: object_name, object_category, environment, action, grip_type, description

**Accuracy**: ~90% after human verification。Failure cases 主要在 hand 离开视野、lighting 差、3 帧上下文不足。

---

## 4. Benchmark Tasks 设计哲学

### 4.1 Cross-Sensory Retrieval

**Task 1: Video ↔ Tactile**
测试 vision 和 touch 之间的 semantic alignment。两个方向都测 (Video→Tactile 和 Tactile→Video) 以验证 representation 的 symmetry。

**Task 2: Pose ↔ Tactile + Multimodal → Unimodal**
Pose-tactile retrieval 测试 geometric coupling。Multimodal→Unimodal (e.g., Video+Tactile→Pose) 测试 fusion 是否减少 ambiguity。

**Metrics**: Recall@1/5/10 + mAP，遵循 ObjectFolder protocol。

### 4.2 Tactile Pattern Classification

两个独立 task：
- **Action recognition**: 整体 hand-object motion intent
- **Grasp type recognition**: hand 的 contact configuration

为什么分开：action "pointing" 对应 Index-Finger Extension grip，action "picking up" 对应 Prismatic Two-Finger grip。这两个 label space 是 decoupled 的。

---

## 5. 方法架构详解

### 5.1 三个 modality-specific encoders

**Visual Encoder (f_V)**:
```
Input: N=20 frames at 30 Hz
→ DINOv3 ViT-B/16 (frozen)
→ per-frame features
→ temporal mean pooling
→ linear projection → z_v ∈ R^64
```

DINOv3 是 Meta 2025 的新版本 self-supervised ViT，pretrain on 1.689B images。Frozen 使用因为 dataset 规模 (5.1h) 不足以 fine-tune ViT-B。

参考 DINOv3: https://arxiv.org/abs/2508.10104

**Tactile Encoder (f_T)** — 这是 paper 的一个重要 insight：
```
Input: N=20 frames × 16×16 pressure maps
→ 3-layer CNN (32 filters, 5×5 kernel, ReLU, 2×2 maxpool)
→ flatten
→ 2-layer BiGRU (hidden dim=120)
→ concat forward + backward last hidden states
→ ReLU + linear projection → z_t ∈ R^64
```

**为什么不用 ResNet**：ablation (Table 6) 显示 Lite-CNN 在 video→tactile 上 mAP 16.76% vs ResNet-18 的 6.27%。Tactile signal 是 sparse + highly structured，不是 natural image。Vision backbone 的 inductive bias (hierarchical texture, edge detection) 与 tactile 的 spatial structure 不匹配。

**Pose Encoder (f_P)**:
```
Input: N=20 frames × 21 keypoints (3D)
→ geometric normalization (translation + scale invariance)
→ 4-layer MLP
→ adaptive temporal average pooling
→ linear projection → z_p ∈ R^64
```

21 keypoints 是标准 hand model (wrist + 5 fingers × 4 joints)。

### 5.2 Loss Function

使用 symmetric InfoNCE，CLIP-style contrastive learning。

**Bi-modal loss** (Eq 1):
$$\mathcal{L}_{ab} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\langle z_a^{(i)}, z_b^{(i)} \rangle / \tau)}{\sum_{j=1}^{B} \exp(\langle z_a^{(i)}, z_b^{(j)} \rangle / \tau)}$$

变量解释：
- $B$: batch size (256)
- $z_a^{(i)}, z_b^{(i)}$: 第 $i$ 个样本在 modality $a$ 和 $b$ 的 L2-normalized embedding
- $\langle \cdot, \cdot \rangle$: inner product (因为 L2-normalized，等价于 cosine similarity)
- $\tau = 0.07$: temperature parameter (from MoCo)
- 分子: positive pair (matching sample)
- 分母: sum over all $B$ candidates (1 positive + $B-1$ negatives)

**Total bi-modal loss** (Eq 2):
$$\mathcal{L} = \mathcal{L}_{ab} + \mathcal{L}_{ba}$$

Symmetric: 既学 $a \to b$ 也学 $b \to a$，避免 representation 偏向单一 modality。

**Tri-modal (fusion) loss** (Eq 3):
$$z_f = \phi([z_a; z_b]) \in \mathbb{R}^{64}$$

其中 $\phi(\cdot)$ 是 lightweight linear head，$[z_a; z_b]$ 是 concatenation。然后 $z_f$ 和 $z_c$ 之间应用同样的 InfoNCE。

**Training config**: 300 epochs, Adam, lr=1e-4, batch=256, cosine annealing + 5-epoch warmup。

---

## 6. 实验结果深度解读

### 6.1 Cross-Sensory Retrieval (Table 2)

**Bi-modal** (Table 2a) 关键数字：

| Task | Chance | CCA | PLSCA | Ours |
|------|--------|-----|-------|------|
| Video→Tactile R@1 | 0.07 | 0.50 | 0.21 | **7.15** |
| Tactile→Video R@1 | 0.07 | 0.71 | 0.64 | **7.15** |
| Tactile→Pose R@1 | 0.07 | 0.57 | 0.14 | **7.15** |
| Pose→Tactile R@1 | 0.07 | 0.64 | 0.14 | **6.93** |

Linear baselines (CCA, PLSCA) 基本接近 chance，说明 vision-tactile 之间是 highly non-linear relationship。Contrastive learning 能 learn 到这个 mapping。

**Tri-modal** (Table 2b) 关键 insight：

| Task | mAP |
|------|-----|
| Video+Pose→Tactile | **26.86** |
| Tactile+Pose→Video | **23.46** |
| Video+Tactile→Pose | **26.86** |

Multimodal query 显著高于 unimodal。Intuition：video 提供 global scene + object context，pose 提供 fine-grained kinematics，tactile 提供 local contact + force。三者组合减少 ambiguity。

### 6.2 Classification (Table 3)

| Modality | Action Acc | Grasp Acc |
|----------|-----------|-----------|
| Vision only | 40.26 | 57.45 |
| Pose only | 33.22 | 46.32 |
| Tactile only | 29.95-31.59 | **60.23-57.12** |
| T+P+V | **35.02-37.32** | **55.65-68.09** |

**关键 insight**：
1. **Tactile 对 grasp type 最 informative** (60.23% RN18) — 因为 grasp 本质是 local contact geometry
2. **Vision 对 action 最 informative** (40.26%) — 因为 action 依赖 global context (object, scene, trajectory)
3. **Tactile + Vision > Tactile + Pose** — vision 补充 object geometry 信息

### 6.3 Ablation Studies

**Window Size** (Table 5)：
| Window | Video→Tactile mAP | Video+Pose→Tactile mAP |
|--------|-------------------|------------------------|
| 5 | 12.57 | 19.89 |
| 10 | 15.44 | 15.28 |
| 20 | **16.76** | **24.46** |

20 帧 (0.67s) 最好。Tactile contact pattern 是 time-varying，短窗口 miss temporal evolution。

**Encoder Capacity** (Table 6)：
| Encoder | Video→Tactile mAP | Tactile→Pose mAP |
|---------|------------------|-------------------|
| ResNet-18 (224×224) | 6.27 | 6.53 |
| Lite-CNN (16×16) | **16.76** | **16.76** |

Lite-CNN 比 ResNet-18 高 10+ 个点。ResNet 的 deep hierarchy + ImageNet inductive bias 对 16×16 tactile map 是 over-parameterization，引入 noise。

**Discretization** (Table 7)：
| Method | V→T mAP | T→P mAP |
|--------|---------|---------|
| Log 3-level | 9.41 | 7.29 |
| Log 5-level | 12.98 | 10.04 |
| Log 7-level | 15.06 | 10.72 |
| Linear 5-level | 16.55 | **12.98** |
| Linear 7-level | 16.05 | 12.91 |
| Raw Continuous | **16.76** | 14.33 |

Discretization 作为 regularizer 减少 sensor noise，但 raw continuous 仍然是 strong default。Linear 5-7 level 在多数 task 上接近 raw。

---

## 7. 应用：Ego4D Zero-shot Retrieval

Figure 6 展示了在 Ego4D 上的 zero-shot tactile retrieval。给定 Ego4D video query，从 OPENTOUCH 数据库 retrieve 最相似的 tactile sequence。验证方法：看 retrieved tactile 对应的 source video，发现 manipulation primitive 高度相似。

这个 application 的意义：OPENTOUCH 可以作为 **tactile database**，为大规模 egocentric video (Ego4D 有 3000 小时) augment contact + force cues。这是 paper 的一个 big vision — 让 in-the-wild video 获得 tactile grounding。

参考 Ego4D: https://ego4d-data.org/

---

## 8. 局限性 (Supp D.6)

Paper 自己承认的 limitations：

1. **只测 normal pressure**: piezoresistive array 无法 capture shear, micro-vibration, temperature。Slip detection, texture recognition 受限。
2. **Spatial resolution 有限**: 16×16 grid 在 fingertip 区域的 resolution 不足以 localize 小 contact patch。
3. **Force range 0.02-50 kPa**: 超出范围会 saturate。
4. **Durability**: FPCB 在 finger joint 处反复 bending，real-world use (sweat, donning/doffing) 会 fracture copper traces。Bench test ~10k cycles。
5. **Wireless reliability**: ESP-NOW 有 packet loss, RF interference, clock drift 风险。
6. **Single glove size**: 改变 contact geometry 和 friction，影响 natural manipulation。
7. **Right hand only**: 左手 generalization 靠 mirroring，但没验证。

---

## 9. 对你的 Intuition Building

几个我觉得对你 (Karpathy) 特别 relevant 的点：

### 9.1 Tactile 作为 "Compact yet Powerful" Signal

Table 3 显示 tactile only 在 grasp type 上 60.23%，接近 vision only 的 57.45%。但 tactile encoder 只有 3-layer CNN + 2-layer BiGRU，参数量比 DINOv3 ViT-B 小几个数量级。这说明 tactile signal 的 information density 很高，不需要 heavy encoder。

Intuition：tactile 是 object + hand physics 的 direct sensor读数，而 vision 是这个 physics 的 indirect projection。对于 grasp 这种 local contact geometry task，direct 信号更 efficient。

### 9.2 为什么 Lite-CNN 击败 ResNet

这个结果对 general representation learning 有启示。Tactile map 的 statistics 和 natural image 完全不同：
- **Sparsity**: 大部分 taxel 是 0 或接近 0
- **Spatial structure**: contact patch 是 localized blob，不是 hierarchical texture
- **No scale invariance**: 16×16 就是 native resolution，没有 multi-scale 需求

ResNet 的 inductive bias (depth, multi-scale feature hierarchy, ImageNet texture bias) 与 tactile statistics 不匹配。Lite-CNN 的 shallow + local receptive field 反而 preserve 了 tactile 的 native structure。

这让人联想到: 对 different modality，我们需要 different inductive bias。直接 port vision backbone 到其他 modality 不一定 work。

### 9.3 Temporal Window 的重要性

从 5 帧到 20 帧，Video+Pose→Tactile mAP 从 19.89% 涨到 24.46%。这说明 tactile contact 是 dynamic process，single frame 不足以 capture grasp 的 "arc" (approach → peak → release)。

这和 GPT-5 annotation sampling strategy (3 frames: onset, peak, post-peak) 呼应。Manipulation 的 semantics 在 temporal trajectory 里，不在 single snapshot。

### 9.4 Multimodal Fusion 的 Complementarity

Table 2b 的 tri-modal 结果是最强的 evidence。Video+Tactile→Pose mAP 26.86% 说明 vision + tactile 能 supervise hand pose reconstruction。这个方向有 robotics 应用价值：如果 robot 有 camera + tactile sensor，可以 infer hand configuration，不需要 explicit pose sensor。

---

## 10. 相关工作链接汇总

**Tactile sensing hardware**:
- STAG (Nature 2019): https://www.nature.com/articles/s41586-019-1234-z
- Conformal Tactile Textiles (Nature Electronics 2021): https://www.nature.com/articles/s41928-021-00545-0
- Embroidered Smart Gloves (Nature Communications 2024): https://www.nature.com/articles/s41467-024-49007-8

**Cross-modal learning**:
- CLIP (ICML 2021): https://arxiv.org/abs/2103.00020
- ImageBind (CVPR 2023): https://arxiv.org/abs/2205.01377
- UniTouch (CVPR 2024): https://arxiv.org/abs/2402.14355
- Touch and Go (arXiv 2022): https://arxiv.org/abs/2211.12498

**Egocentric datasets**:
- Ego4D (CVPR 2022): https://ego4d-data.org/
- Ego-Exo4D (CVPR 2024): https://ego-exo4d-data.org/
- EPIC-KITCHENS: https://epic-kitchens.github.io/

**Hand-object interaction**:
- GRAB (ECCV 2020): https://grab.is.tue.mpg.de/
- ContactDB (CVPR 2019): https://contactdb.cc.gatech.edu/
- OakInk (CVPR 2022): https://www.oakink.org/
- ARCTIC (CVPR 2023): https://arctic.is.tue.mpg.de/

**Grasp taxonomy**:
- GRASP Taxonomy (IEEE THMS 2016): https://ieeexplore.ieee.org/document/7299224

---

## 11. 我的批判性思考

几个我觉得 paper 可以改进或值得讨论的点：

1. **Scale 仍然小**: 5.1 小时 / 2900 clips 对比 Ego4D 的 3000 小时，scale 差距巨大。但这是 hardware-limited，tactile glove 的 cost + durability 限制了 scaling。
2. **只有 right hand**: 真实 bimanual manipulation (cooking, opening jars) 完全 missing。Left hand 的 mirroring assumption 没有验证。
3. **Annotation accuracy 90%**: 对 grasp type 这种 fine-grained label，90% 意味着 10% noisy labels。Paper 没讨论这对 downstream training 的影响。
4. **No shear/tactile dynamics**: Piezoresistive 只测 normal pressure。Slip detection, texture, thermal 全部 missing。这限制了 tactile 的 application scope。
5. **Baseline 不够强**: Retrieval 用 CCA/PLSCA 作为 linear baseline，但没有对比近期 cross-modal 方法如 ImageBind, UniTouch 的 finetuned 版本。
6. **Single environment diversity**: 14 environments 听起来多，但每个 environment 的 recording time 不长。In-the-wild 的 environment diversity 和 Ego4D 的 global scale 比还是小。

但总体来说，OPENTOUCH 是一个重要的 dataset contribution，填补了 in-the-wild tactile sensing 的空白，硬件设计的 open-source + low-cost + reproducible 是 community 最需要的。Benchmark 的设计 (retrieval + classification) 也很合理，ablation 提供了 actionable insights (Lite-CNN > ResNet, 20 frames optimal, discretization as regularizer)。

我特别欣赏 paper 没有过度 claim — 它定位为 "foundation for community-driven progress"，而不是 claim 解决了所有问题。这种 positioning 对 dataset paper 是合适的。
