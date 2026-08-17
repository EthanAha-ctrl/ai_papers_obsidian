---
source_pdf: Human-level 3D shape perception emerges from multi-view learning.pdf
paper_sha256: c137cf2193c105f1fb0fafc739e525942756f77b5a3d763ac007448a7e413c84
processed_at: '2026-08-05T07:44:18-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：这篇paper到底在说什么

## 一句话版本

**之前所有AI模型看不懂3D形状，大家以为是因为缺"天赋"（inductive bias），其实是因为缺"经验"（multi-view data）——给它看足够多从不同角度拍的同一场景的照片，让它猜depth和camera位置，它自己就学会像人一样理解3D shape了。**

---

## 这个问题的背景：为什么3D perception这么难

你闭上眼睛摸一个杯子，再睁开眼看——你的大脑自动把触觉信息和视觉信息对上号，知道"这是一个3D的圆柱体"。

但AI做不到这件事。传统vision model（DINOv2, CLIP, MAE这些）看到一张椅子照片，它学到的是"这看起来像椅子"这种2D pattern，不是"这是一个有四条腿、有靠背、在3D空间里有体积的object"。

paper里的Figure S1把这件事量化了：DINOv2-giant在MOCHI benchmark上accuracy不到50%，MAE直接chance level。这些model在ImageNet上再强，对3D shape understanding基本是零。

## 人类怎么做到的

婴儿也不是生下来就懂3D。Long et al. 2024的BabyView dataset（https://arxiv.org/abs/2406.10447）显示：婴儿每天看到的是非常limited的几个object，但是从无数个angle看——因为他们会爬、会转头、会抓东西往嘴里塞。同时他们有stereo vision（两只眼睛）、vestibular system（感知自己头在动）、tactile feedback（摸到东西）。

这套multi-modal + multi-view的sensory data，就是婴儿学习3D perception的"training data"。

认知科学里两派吵了几十年：
- **Empiricist派**（Helmholtz, Gibson, Elman）：perception是从sensory experience里学出来的，general-purpose learning就够
- **Nativist派**（Spelke）：必须有built-in "core knowledge"，比如object permanence、3D structure这些是innate的

之前computational model一直帮不了nativist说话——因为所有model都做不到human-level 3D perception。这篇paper第一次让empiricist有了concrete computational evidence。

## VGGT做了什么

VGGT = Visual Geometry Grounded Transformer（J. Wang et al. CVPR 2025, https://arxiv.org/abs/2503.11651）

架构特别简单：

```
一堆照片（同一场景不同角度）
        ↓
DINOv2-Large encoder（frozen，提取patch features）
        ↓
24层transformer aggregator（做cross-attention，让照片之间互相"看"）
        ↓
输出：每张照片每个pixel的depth + camera pose + uncertainty
```

关键：**没有任何hand-coded geometric prior**。没有epipolar geometry，没有bundle adjustment，没有ICP。就是一个generic transformer，从data里学。

training objective更简单——给它一组同一场景的照片，让它猜：
1. 每张照片的camera在哪个位置（pose estimation）
2. 每张照片每个pixel离camera多远（depth estimation）
3. 自己对这个猜测有多confident（aleatoric uncertainty）

就这么train。没有object label，没有shape annotation，没有segmentation。训练data是DL3DV-10K这种大规模scene dataset（https://arxiv.org/abs/2312.16236）。

## 为什么这个objective能emerge 3D shape understanding

这是整篇paper最精妙的地方。直觉是这样的：

假设你给model看两张照片——一张椅子正面，一张椅子侧面。如果model要准确预测两张照片的depth和camera pose，它**必须**理解：
- 椅子正面照片里的"靠背顶部"和侧面照片里的"靠背顶部"是同一个3D point
- 这个3D point在两张照片里的2D projection位置不同，但corresponding to same physical location
- 椅子的3D structure决定了这个correspondence pattern

换句话说，**要predict depth和pose正确，model必须implicit地solve correspondence problem**——而correspondence正是3D shape understanding的核心。

传统vision model的问题是：它把每张照片当独立的image处理，features之间没有geometric relationship。VGGT的multi-view training force model在feature space里encode geometric correspondence。

## 怎么eval：zero-shot，没有任何task-specific training

这是paper最严谨的地方。他们没有fine-tune VGGT做oddity task，没有train linear decoder，什么都没做。纯粹用pre-trained VGGT的internal uncertainty signal。

具体protocol（Figure 2）：

每个trial有3张照片：$A, A', B$（$A$和$A'$是同一object不同角度，$B$是不同object）

1. 把3张照片两两组合：$(A, A'), (A, B), (A', B)$
2. 分别送进VGGT，提取每对的per-pixel uncertainty $\Sigma_i^D$
3. 对每对求average confidence
4. 逻辑：$(A, A')$是同一object，geometric correspondence高，confidence应该高；$(A, B)$是不同object，correspondence低，confidence应该低
5. 选confidence最低的pair对应的image作为"oddity"
6. 跟ground truth比对，correct or incorrect

这个approach的妙处在于：**aleatoric uncertainty本来是训练时用来weight depth loss的，paper直接repurpose它作为correspondence quality metric**。完全zero-shot。

## 结果：第一次match human

| Model | Accuracy |
|-------|----------|
| Human | 78.9% ± 3.0% |
| **VGGT** | **83.0% ± 3.7%** |
| DINOv2-L | 28.5% ± 4.9% |

t-test: VGGT vs Human, $p = 0.110$——**没有statistically significant difference**。

DINOv2-L就是VGGT的encoder本身。同一个encoder，加了multi-view trained aggregator，accuracy从28.5%跳到83%。这alone就是一个strong statement：single-view self-supervised learning**不够**emerge 3D shape understanding，必须加multi-view spatial objective。

其他multi-view model（DUSt3R, MASt3R, Pi3）也没达到human level（Figure S2），只有VGGT做到了。可能因为VGGT的aggregator更深（24 layers）且能处理arbitrary number of views。

## 更striking的两个result

### 1. Model confidence predicts human error patterns

把所有trial按VGGT的confidence margin分30个bin：

- VGGT confidence最低的bin → human accuracy只有63.0%
- VGGT confidence最高的bin → human accuracy 92.5%
- Pearson $r = 0.830$

也就是说：**VGGT觉得难的trial，人也觉得难；VGGT觉得容易的trial，人也觉得容易**。

这不是trivial的——VGGT从来没见过这些trial，没被train过判断"难度"。它的aleatoric uncertainty纯粹是training时predict depth用的，但这个internal variable天然correspond to human subjective difficulty。

### 2. Model solution layer predicts human reaction time

这个更crazy。VGGT的aggregator有24层transformer。对每个trial，他们找"最早的能correctly solve oddity且后续所有层都保持correct的layer"——叫"solution layer"。

结果：
- VGGT在early layer就solve的trial → human RT 3238ms（快）
- VGGT需要到late layer才solve的trial → human RT 4240ms（慢）
- Pearson $r = 0.796$
- Controlling for confidence: partial $r = 0.159, p = 2.78 \times 10^{-10}$（仍然significant）

这个partial correlation是关键——说明solution layer capture的不只是"难度"，而是**processing dynamics本身**。Feedforward depth天然map到processing time。

这跟Yamins & DiCarlo 2016（https://www.nature.com/articles/nn.4244）发现ventral stream的hierarchical processing time对应brain area sequence的logic完全一致。

## Qualitative evidence：attention visualization

Figure 4特别visual。他们在image $A$上手动选几个keypoint（不同颜色的点），然后看layer 15的cross-image attention在$A'$和$B$上的分布：

- 在$A'$上：$A$的每个keypoint attention集中到$A'$上**同一个object part**（即使2D位置不同，因为视角变了）
- 在$B$上：attention是分散的、不corresponding的

在layer 0（Figure S9），attention完全没分化。到layer 15才emerge出这种correspondence-based representation。

这说明VGGT内部真的emerged出implicit 3D correspondence estimation——这正是DUSt3R系列模型的核心思想（https://arxiv.org/abs/2312.14132）。

## 为什么这matters：三个层面

### Cognitive science层面

Spelke 1990说infant需要innate "core knowledge"才能理解object。这篇paper说：不需要。General-purpose transformer + multi-view naturalistic data + simple predictive objective就够。

Empiricist view第一次有了concrete computational instantiation。

### AI/ML层面

Bowers et al. 2023（https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/deep-problems-with-neural-network-models-of-human-vision/E5D5C20DF5C57C2E9B5C75A23F7A33D8）批评deep net model of human vision是"cherry-picked success"。这篇paper的response是：之前model失败不是因为deep learning不行，是**training objective选错了**。把vision当feature extraction problem是错的方向，当visual-spatial correspondence problem才对。

Rich Sutton的"Bitter Lesson"（http://www.incompleteideas.net/IncIdeas/BitterLesson.html）再次被验证：hand-coded geometric priors（Multiple View Geometry那套）被scalable learning超越。

### Neuroscience层面

VGGT的architecture跟brain有一种粗略correspondence：
- DINOv2 encoder ↔ ventral visual stream（V1 → IT）
- 24-layer aggregator ↔ Bonnen et al. 2021（https://www.cell.com/neuron/fulltext/S0896-6273(21)00494-9）说的medial temporal cortex在perception中的role——integrating visuospatial sequences

这种mapping很speculative，但作为hypothesis generator很有用。未来可以用VGGT的layer representations去fit human fMRI data。

## Limitations（paper自己承认的）

1. **No foveal/eye movement**：VGGT是single forward pass，human有sequential eye movements + active sampling
2. **Global coordinate frame**：VGGT用global frame，human用egocentric
3. **Depth supervision不faithful**：dense GT depth vs stereopsis
4. **Solution layer ↔ RT的mapping可能ambitious**：feedforward depth跟RT的correspondence可能反映shared difficulty，不是shared algorithm
5. **No scaling analysis**：只有VGGT-1B一个size

## 我的takeaway

这篇paper的核心message是：**human-level 3D perception不是magic，不需要innate core knowledge，不需要object-specific inductive bias。它就是multi-view visual-spatial learning的emergent property。**

这跟你自己一直push的philosophy完全对齐——Tesla vision stack的"photon-to-control"，end-to-end learning避免hand-engineered features。VGGT是同一个哲学在3D perception domain的instantiation。

如果进一步extend到temporal multi-view（video + self-motion + action），这就是world model的雏形。VGGT + temporal extension + action conditioning = LeCun JEPA的multi-view版本。这种model可能emerge出object permanence、occlusion reasoning、甚至intuitive physics。

---

所有reference链接在上一条回复里都有，这里不重复了。核心就这几个：
- VGGT: https://arxiv.org/abs/2503.11651
- Project page: https://vgg-t.github.io/
- MOCHI benchmark: Bonnen et al. NeurIPS 2024
- Sutton Bitter Lesson: http://www.incompleteideas.net/IncIdeas/BitterLesson.html

---

# 这篇paper的深度解读

Andrej，这篇paper做的工作非常漂亮，本质上是给"empiricist vs nativist"这场认知科学的老争论提供了一个新的computational证据。我从多个层面来讲。

## 1. Paper核心thesis

作者Tyler Bonnen, Jitendra Malik, Angjoo Kanazawa (Berkeley)想回答一个非常根本的问题：**human-level 3D shape perception不需要object-specific inductive bias，能从general-purpose learning + multi-view naturalistic data中emerge出来**。

这跟Spelke (1990)等nativist观点直接对着干——后者认为infant必须有built-in "core knowledge"才能理解object 3D structure。paper的证据是：用VGGT这种model，没有任何hand-coded geometric priors，zero-shot就能match human accuracy。

paper link: https://bonnen.ai/ (project page)
github: https://github.com/Tylerbonnen 
arXiv preprint: 应该会在arXiv上后续放出

## 2. Task Setup: MOCHI benchmark

他们用的是Bonnen et al. (NeurIPS 2024)的MOCHI benchmark，这是个oddity discrimination task：

- 给定3张image: $A, A', B$
- $A$和$A'$是同一个object的不同视角
- $B$是不同object
- 被试/model要找出B（odd one out）
- Difficulty parametrically varied via两个axis：
  - object similarity（A vs B的相似度）
  - viewpoint variation（A vs A'的角度差）

MOCHI有300+ human participants, 25K trials，对象包括real-world objects（chairs, tables）和procedurally generated abstract shapes（"nonsense" objects无semantic attributes）。

## 3. Model核心：VGGT (Visual Geometry Grounded Transformer)

VGGT来自J. Wang et al. CVPR 2025 (https://arxiv.org/abs/2503.11651)。

### 3.1 Architecture overview

```
[N images] → [DINOv2-Large encoder (frozen)] → [patch tokens per image]
                                                       ↓
                              [24-layer transformer aggregator]
                                                       ↓
                       predicts: depth, camera pose, point map, uncertainty
```

Key design features:
- Generic transformer架构（ViT style, Dosovitskiy 2020）
- **No hand-coded geometric priors** (no epipolar constraint baked in, no ICP)
- 任意数量image作为input（permutation-aware，actually Pi3是permutation-equivariant variant）
- 同时predict多个visual-spatial signals

### 3.2 Training objective

这是整个story最关键的部分。VGGT的训练target是visual-spatial prediction，**完全不是**object recognition或3D shape inference。

核心的depth loss（公式1）：

$$\mathcal{L}_{\text{depth}} = \sum_{i=1}^{N} \left\| \Sigma_i^D \odot (\hat{D}_i - D_i) \right\| + \left\| \Sigma_i^D \odot (\nabla \hat{D}_i - \nabla D_i) \right\| - \alpha \log \Sigma_i^D$$

逐项解释：
- $\mathcal{L}_{\text{depth}}$: 整个depth prediction的loss
- $i$: pixel index
- $N$: total number of pixels
- $\Sigma_i^D$: predicted precision (inverse variance) at pixel $i$，即aleatoric uncertainty的inverse；高precision意味着model对这点depth估计confident
- $\hat{D}_i$: predicted depth at pixel $i$
- $D_i$: ground truth depth at pixel $i$
- $\nabla \hat{D}_i$, $\nabla D_i$: predicted / ground truth depth gradient（spatial gradient，即$\partial D / \partial x$和$\partial D / \partial y$），enforce spatial smoothness
- $\odot$: Hadamard product (element-wise multiplication)，让precision高的地方penalty大
- $\alpha$: hyperparameter balancing confidence regularization
- $-\alpha \log \Sigma_i^D$: 这是log-likelihood term (Gaussian NLL的形式)，鼓励model别无限提高precision躲penalty

这套loss的形式来自Kendall & Gal (2017, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" https://arxiv.org/abs/1703.04977)的heteroscedastic aleatoric uncertainty framework。

training data是DL3DV-10K (Ling et al. 2024, https://arxiv.org/abs/2312.16236)、CO3D、ScanNet等大规模multi-view scene data。

### 3.3 为什么这个训练objective能emerge 3D shape understanding

我觉得这是paper最profound的地方。三个理由：

**a. 对应Helmholtz的unconscious inference**：human vision本质上是在solve一个inverse problem——从2D retina推断3D structure，通过motion parallax, stereopsis, proprioception等cues。VGGT训练object就**等同**这个inverse problem的computational instantiation：从image集合推断depth、camera pose。

**b. 把vision重新framing为correspondence problem**：传统CV把vision当成feature extraction（image → features）。multi-view framing则把vision当成correspondence problem（images + spatial context → geometric correspondences）。这种framing更接近Gibson的affordance view，把perception当成active process。

**c. No object-centric bias**：训练data是whole natural scenes，不是segmented objects。任何object perception ability是emerged，**没有**被explicitly supervised。

## 4. Evaluation：三个zero-shot metric

### 4.1 Accuracy metric

对每个trial $(A, A', B)$：
1. Encode所有pairwise combinations: $(A, A'), (A, B), (A', B)$
2. Extract per-pixel aleatoric uncertainty $\Sigma_i^D$ for each pair
3. 求pair-wise average confidence
4. 选择confidence最低的pair所对应的image作为"oddity"

直觉：matching pair $(A, A')$应该high confidence（同object有consistent geometry），non-matching pair $(A, B)$应该low confidence（无geometric correspondence）。

### 4.2 Confidence margin metric

$$\Delta = \text{conf}(A, A') - \frac{1}{2}[\text{conf}(A, B) + \text{conf}(A', B)]$$

这是trial-level的continuous confidence值，zero-shot预测human error patterns。

### 4.3 Solution layer metric

VGGT的aggregator有24个transformer layer。对每个trial：
- 从layer 1到layer 24分别extract patch tokens
- 在每层用3种相似度metric (mean patch-to-patch cosine, max patch-to-patch cosine, global pooling similarity)
- 找到**最早**的layer使得：oddity prediction正确，且所有后续layer都保持正确

这个layer index就作为"model solution time"，对应当human reaction time的computational proxy。

## 5. 主要结果

### 5.1 Accuracy

| Model | Normalized Accuracy (±SEM) |
|-------|----------------------------|
| **Human** | 78.9% ± 3.0% |
| **VGGT-1B** | **83.0% ± 3.7%** |
| DINOv2-L (baseline) | 28.5% ± 4.9% |

paired t-test: VGGT vs Human, $t(20) = -1.67, p = 0.110$，**no significant difference**！

VGGT vs DINOv2: $t(20) = 12.28, p < 0.001$, Cohen's $d = 2.68$ (massive effect)。

注意DINOv2-L就是VGGT的encoder本身——加一个multi-view trained aggregator就让accuracy从28.5%跃升到83%。这单独就是一个strong finding：**single-view self-supervised features不够emerge 3D shape understanding**，必须加上multi-view spatial objective。

其他multi-view model (DUST3R, MAST3R, Pi3) 也没达到human level，只有VGGT做到了（见Figure S2）。

### 5.2 Confidence predicts error patterns

按confidence margin分30个quantile bin：
- Pearson $r = 0.830, p < 0.001$
- Spearman $\rho = 0.932, p < 0.001$
- OLS regression $\beta = 0.01, F(1, 28) = 61.92, p = 1.43 \times 10^{-8}$
- Lowest confidence bin: human 63.0% accuracy
- Highest confidence bin: human 92.5% accuracy
- $R^2 = 0.689$

这是一个很strong的result：**模型uncertainty直接对应human subjective difficulty**。

### 5.3 Solution layer predicts reaction times

| Layer bin | Human RT (ms) |
|-----------|---------------|
| earliest bin (fastest) | 3238 |
| latest bin (slowest) | 4240 |
| range | 1002 ms |

- Pearson $r = 0.796, p < 0.001$
- Spearman $\rho = 0.804, p < 0.001$
- OLS $\beta = 30.91$ ms/layer, $F(1, 17) = 29.38, p = 4.59 \times 10^{-5}$
- $R^2 = 0.633$
- controlling for confidence: partial correlation $r = 0.159, p = 2.78 \times 10^{-10}$ (still significant)

最后这个partial correlation是关键，表明solution layer capture的是**processing dynamics独立于overall difficulty**。这跟Yamins & DiCarlo (2016, https://www.nature.com/articles/nn.4244)对ventral stream的"representational dynamics across cortical areas"工作的思路呼应——前馈layer的depth对应processing time。

## 6. Qualitative findings: Cross-image attention

Figure 4的visualization很striking：
- 在layer 15，对image $A$上某个keypoint，model在 $A'$ 上attention对应到**同一个object part**（即使xyz位置不同）
- 同一个keypoint在 $B$ 上的attention是分散的、不对应
- 在layer 0（Figure S9）attention几乎没分化

这说明VGGT内部emerged出**implicit correspondence estimation**——这也是DUSt3R / MASt3R系列模型的核心思想（Wang et al. 2024, https://arxiv.org/abs/2312.14132）。

## 7. 我觉得最深的insight

### 7.1 关于"neural network AGI"的debate

Bowers et al. (2023, https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/deep-problems-with-neural-network-models-of-human-vision/E5D5C20DF5C57C2E9B5C75A23F7A33D8)的target article主张：deep neural nets在human vision benchmark上的成功是"cherry-picked"，实际在很多core visual cognition任务（包括3D perception）远不如human。

Bonnen的MOCHI原始论文（Bonnen et al. 2024, https://proceedings.neurips.cc/paper_files/paper/2024/hash/43533)就是直接验证了human vs standard vision model的gap巨大。这篇新paper则show：换一种model class——multi-view trained——gap直接消失。

这对Bowers et al.的critique是个partial response：之前deep model失败，可能不是deep learning本身的问题，**而是训练objective的选择问题**。Perception不是image classification，是visual-spatial correspondence inference。

### 7.2 与neuroscience的connection

paper里提到Bonnen et al. (2021, Neuron, https://www.cell.com/neuron/fulltext/S0896-6273(21)00494-9)和Bonnen et al. (2025, Cognition, https://www.sciencedirect.com/science/article/pii/S0010027725000592)关于medial temporal cortex (MTC)在perception中的role的工作。

这与VGGT的architecture有个beautiful correspondence：
- DINOv2 encoder ↔ ventral visual stream (V1 → IT)
- 24-layer aggregator ↔ MTC's role as "integrating over visuospatial sequences"
- 这种mapping很粗，但作为hypothesis generator很有用

未来工作可以做的：把VGGT当成computational model fit human fMRI / neural data，看看aggregator layer能不能predict MTC activity patterns。

### 7.3 与infant development的parallel

paper里讨论infant visual experience（Long et al. 2024 BabyView, https://arxiv.org/abs/2406.10447; L.B. Smith et al. 2018）：
- infant密集visual experience，少量objects
- 有stereo vision, vestibular self-motion, tactile feedback

VGGT的training setup正是这种multi-view + spatial cue的computational analog。这给"empiricist"view一个具体computational implementation：human infant 3D perception的development可能就是类似multi-view learning objective的process。

### 7.4 "Bitter lesson"再次验证

Rich Sutton的"The Bitter Lesson" (http://www.incompleteideas.net/IncIdeas/BitterLesson.html)在CV里再次被验证：hand-coded geometric priors (Hartley & Zisserman 2003, Multiple View Geometry那套)被scalable general-purpose architecture + 大规模data + 简单objective超越。

### 7.5 跟你早期工作的潜在connection

你之前的work（如与Li Fei-Fei的vision work, Tesla的vision-only approach）一直强调end-to-end learning避免hand-engineered features。VGGT这个结果和Tesla vision stack的"photon-to-control"哲学非常对齐——都不multi-camera fusion（虽然Tesla是multi-camera single-time, VGGT是multi-camera multi-time）。

可以把VGGT这种架构推广到temporal multi-view setting（即video + self-motion），可能会进一步emerge出object permanence、3D scene reconstruction等更多cognitive abilities。

## 8. Limitations & Future directions

paper诚实承认的gaps：

1. **No foveal constraints**: VGGT是single forward pass，缺human vision的sequential eye movements + active sampling（Ullman 1979 sequence-seeking的critique）
2. **Global vs egocentric coordinate**: VGGT用global frame，human是egocentric self-motion
3. **Depth supervision形式不faithful**: 用dense ground truth depth，与stereopsis不完全一致
4. **Solution layer metric的歧义**: feedforward depth跟RT的correspondence可能反映shared difficulty，不是shared algorithm。Jacobs et al. (2025, https://arxiv.org/abs/2512.19941)的block-recurrent dynamics work可能提供更好的recurrent mapping
5. **没做大规模model scaling analysis**: VGGT-1B只有一个size，data scaling和model scaling曲线没系统探索

## 9. 一些技术细节的展开

### 9.1 Aleatoric vs Epistemic uncertainty

VGGT里用的是aleatoric uncertainty（data-inherent），不是epistemic uncertainty (model uncertainty)。这个区别来自Kendall & Gal 2017。Aleatoric noise有两类：
- **Homoscedastic**: task-level, 所有input同样noise level
- **Heteroscedastic**: data-dependent, 每个input自己的noise level

VGGT predict的$\Sigma_i^D$是heteroscedastic aleatoric——对应到geometric ambiguity（这块pixel depth估计多准）。Zero-shot accuracy metric本质上是用aleatoric uncertainty作为correspondence quality proxy。

### 9.2 VGGT的"permutation-equivariant"性质

VGGT虽然不是Pi3那种显式permutation-equivariant，但training过程是order-invariant的——aggregator里通过cross-attention让任何image order都能work。这个性质很重要，因为human vision也是order-invariant的——你看一组image不需要特定order就能infer 3D structure。

### 9.3 跟 DUSt3R / MASt3R / Pi3的关系

- **DUSt3R** (Wang et al. 2024): 2-view, predict dense 3D pointmap + confidence
- **MASt3R** (Leroy et al. 2024): DUSt3R + matching head (https://arxiv.org/abs/2406.09681)
- **VGGT** (Wang et al. 2025): N-view (arbitrary number), feedforward single pass
- **Pi3** (Wang et al. 2025, https://arxiv.org/abs/2507.13347): 显式permutation-equivariant variant

paper里测了所有这些，只有VGGT达到human level。Figure S2展示了其他都在gap内。为什么VGGT更优？可能原因：
- 训练data scale更大
- aggregator更deep（24 layers）
- 任意view number的灵活处理

## 10. 给Karpathy你的潜在延伸思考

如果让我猜你的反应，你可能会想到：

**a. World model connection**: VGGT本质上是predicting visual-spatial structure from image集合。如果扩展到temporal multi-view（video）+ action, 就成了LeCun JEPA或world model的multi-view版本。这种model可能emerge出object permanence、occlusion reasoning等更复杂cognition。

**b. Self-supervised scaling law**: VGGT的训练objective是自监督的（depth、pose从scene reconstruction中得到label，不需要human annotation）。这跟DINOv2、MAE、CLIP的self-supervised哲学一致，但加上了multi-view spatial structure。如果scaling laws predict：scale up VGGT到100B parameter + billion scene training data，应该能达到甚至超过human在更多3D cognition task上的能力。

**c. Token-level vs image-level**: VGGT用patch token，每个token在所有image间做cross-attention。这跟vision-language model里的visual token很类似。如果VGGT的aggregator layer的representations做probe，可能能decode出object-centric structure——这直接test Spelke的"core knowledge"是否emerge。

**d. Practical implications for robotics**: 这种model直接可以用作robot的visual perception backbone——zero-shot给出depth、correspondence、camera pose估计。比传统SLAM module简单得多。

---

## 关键reference链接

1. VGGT paper: https://arxiv.org/abs/2503.11651
2. VGGT project: https://vgg-t.github.io/
3. DUSt3R: https://arxiv.org/abs/2312.14132
4. MASt3R: https://arxiv.org/abs/2406.09681
5. Pi3: https://arxiv.org/abs/2507.13347
6. DINOv2: https://arxiv.org/abs/2304.07193
7. MOCHI benchmark (Bonnen et al. NeurIPS 2024): https://proceedings.neurips.cc/paper_files/paper/2024/hash/43533
8. Kendall & Gal uncertainty: https://arxiv.org/abs/1703.04977
9. Bowers et al. critique: https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/deep-problems-with-neural-network-models-of-human-vision/E5D5C20DF5C57C2E9B5C75A23F7A33D8
10. Yamins & DiCarlo ventral stream: https://www.nature.com/articles/nn.4244
11. Bonnen et al. 2021 MTC perception: https://www.cell.com/neuron/fulltext/S0896-6273(21)00494-9
12. Sutton "Bitter Lesson": http://www.incompleteideas.net/IncIdeas/BitterLesson.html
13. BabyView dataset: https://arxiv.org/abs/2406.10447
14. DL3DV-10K: https://arxiv.org/abs/2312.16236
15. Project page: https://bonnen.ai/

---

总结一句话：**这篇paper证明human 3D perception是emergent property of multi-view visual-spatial learning, 而不是需要预先wired的object-specific inductive bias**。技术上它用VGGT这种zero-shot eval framework，让"empiricist"哲学有了concrete computational instantiation。下一步自然会延伸到temporal multi-view (video + self-motion)和active vision (eye movements)，那会更接近human的真实perception process。
