---
source_pdf: A Taxonomy for Evaluating Generalist Robot Policies.pdf
paper_sha256: f039dc3712282422278212d11062524c08482824ea047047e037c9779bc91178
processed_at: '2026-07-17T22:09:31-07:00'
target_folder: Robot-VLA/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# ‹-Gen: A Taxonomy for Evaluating Generalist Robot Policies 深度解读

## 一、Paper 的核心 motivation

Robotics generalist policy 的研究目前处于一个尴尬的状态：大家都在 scaling data、scaling model，但 **generalization 的评估本身是 Wild West**。每篇 paper 自己定义一两个 generalization axis（比如 "robustness to distractors" 或 "novel object"），然后报告一些 cherry-picked 的成功案例。这导致两个问题：

1. 难以横向比较不同 model 的真实 generalization 能力
2. 看不到 model 真正的 failure mode 在哪

这篇 paper 的 insight 是：generalization 本质上是 **policy input/output modality 的 perturbation**。Visuo-lingual policy 有三个 modality——vision、language、action——所以所有 generalization 都可以沿着这三个 modality 的组合来分类。这就是 ‹-Gen (STAR-Gen = Systematic Taxonomy of the Axes of Robot Generalization) 的出发点。

参考链接：
- Paper: https://stargen-taxonomy.github.io/
- OpenVLA: https://openvla.github.io/
- Bridge V2: https://rail-berkeley.github.io/bridge_data_v2/
- Open X-Embodiment: https://robotics-transformer-x.github.io/

---

## 二、Formalism：把 generalization 严格定义出来

### 2.1 Environment 定义

$$E = (S, \mathcal{O}, \mathcal{A}, \mathcal{L}, f_o, f_t)$$

- $S$: state space（机器人物理状态 + 场景物体状态）
- $\mathcal{O}$: observation space（这里特指 third-person image）
- $\mathcal{A}$: action space（robot 7-DoF end-effector delta pose + gripper）
- $\mathcal{L}$: language instruction space
- $f_o: S \to \mathcal{O}$: observation function（rendering/camera projection）
- $f_t: S \times \mathcal{A} \to S$: transition function（physics）

### 2.2 Task 定义

$$\tau = (p_\tau(s_0), l_\tau, R_\tau)$$

- $p_\tau(s_0)$: 初始状态分布（场景 setup）
- $l_\tau \in \mathcal{L}$: language instruction（"put carrot on plate"）
- $R_\tau: (S \times \mathcal{A})^* \to \{0,1\}$: success function，把 state-action trajectory 映射成 0/1

### 2.3 Policy 定义

$$\pi(a \mid o^n, l)$$

输入 $n \geq 1$ 帧 observation（一般 $n=1$ 或 $n=2$ 用于 frame stacking）+ language instruction $l$，输出 action distribution。VLA model 通常用 autoregressive 或 flow matching 来建模这个 distribution。

### 2.4 Perturbation categorization（这是 paper 的核心 formal insight）

给定 perturbation function $P: T \to T$，把 base task $\tau$ 变成 $\tau_P$：

- **Visual**: $p_\tau(o_0) \neq p_{\tau_P}(o_0)$ — 初始 image 分布变了
- **Semantic**: $l_\tau \neq l_{\tau_P}$ — language instruction 变了
- **Behavioral**: $\pi_E$ 的 action distribution 改变 — 需要的最优行为变了

关键点：**categorization 不互斥**。比如把 "pick up carrot" 改成 "pick up zucchini"，同时改变了 vision（zucchini 的外观）、language（noun）、behavior（grasp strategy 不同），属于 Visual+Semantic+Behavioral。

还有一个微妙的点：同一个 perturbed task $\tau_P$ 相对于不同 base task 可以落入不同 category。比如 base task 是 "pick up carrot"，perturbed 为 "pick up orange object" 是 semantic only（behavior 不变）；但 base task 是 "pick up apple"，同样的 "pick up orange object" 就是 semantic+behavioral（因为现在要 pick up 一个不同的 object）。

### 2.5 Factor vs Axis

- **Factor**: human-interpretable 的 perturbation 分组，比如 "Lighting"（包含不同 light intensity 的 perturbation）
- **Axis**: 共享 modality 的 factor 集合，比如 "Image Augmentations" 包含 lighting、blur、contrast

---

## 三、‹-Gen 的 7 个 category 和 22 个 axis

基于 3 个 modality 的非空子集组合，得到 $2^3 - 1 = 7$ 个 category：

### 3.1 Visual only（4 个 axis）
- **V-AUG** (Image Augmentations): lighting, blur, contrast
- **V-SC** (Visual Scene): surface color, distractor appearance/placement, textures — 不影响 behavior
- **V-OBJ** (Visual Task Object): manipulated object color, container color
- **V-VIEW** (Viewpoint): camera pose, partial occlusion

### 3.2 Semantic only（5 个 axis）
- **S-PROP** (Object Properties): 用 color/mass/size 引用 object ("put the orange object")
- **S-LANG** (Language Rephrase): verb synonyms, 去掉 articles
- **S-MO** (Multi-Object Referencing): spatial relations like "in"/"left of"
- **S-AFF** (Human Affordances): "hand me something I can use to clean up"
- **S-INT** (Internet Knowledge): 名人、common object color（"basketball color"）

### 3.3 Behavioral only（2 个 axis）
- **B-HOBJ** (Hidden Object): object mass/friction/fragility — unobserved
- **B-HSC** (Hidden Scene): surface friction, temperature

注意 Behavioral-only 因子必然是 **unobserved** from single observation，这是为什么这类 axis 对 policy 极难——只能从 tactile/interaction 推断。

### 3.4 Visual + Behavioral（5 个 axis）
- **VB-POSE**: object 位置变化
- **VB-ISC** (Interacting Scene): clutter, surface height
- **VB-MOBJ** (Morphed Objects): size, shape 变化
- **VB-ROB** (Robot Embodiment): new arm/gripper
- **VB-SYM** (Symmetry): bimanual mirror motion

### 3.5 Semantic + Behavioral（4 个 axis）
- **SB-ADV** (Motion Adverbs): "quickly"/"slowly"
- **SB-SMO** (Spatial Multi-Object): "left of sink" → "right of sink"
- **SB-NOUN** (Noun Grounding): "pick carrot" → "pick knife"（场景里两者都在）
- **SB-VRB** (Action Verbs): "pick bottle" → "rotate bottle"

### 3.6 Visual + Semantic（1 个 axis）
- **VS-PROP**: object color 改变且 instruction 引用了 color（比如 base 是 "pick up the purple cup"，cup 变蓝后 instruction 必须改成 "pick up the blue cup"）

### 3.7 Visual + Semantic + Behavioral（1 个 axis）
- **VSB-NOBJ** (New Object): carrot → zucchini，三个 modality 全变

---

## 四、BridgeV2-‹ 案例研究

### 4.1 Base Tasks（4 个）

1. Put carrot on plate
2. Put knife on plate
3. Flip pot upright
4. Put plate in sink

选这些 task 的 rationale 是 Bridge V2 dataset 里有相似的 sink environment demonstrations，这样可以测出 pretraining 带来的 generalization。但又故意让 task 1/3 用 dataset 里出现过的 instruction，task 2/4 用没出现过的，形成 alignment 梯度。

### 4.2 评估的 13 个 axis

Table III 列出了实际评估的 axis：V-SC, V-OBJ, V-VIEW, S-PROP, S-LANG, S-MO, S-INT, VB-POSE, VB-ISC, VB-MOBJ, SB-SMO, SB-VRB, VSB-NOBJ。覆盖 5/7 category、13/22 axis。

**没覆盖的**：B-HOBJ/B-HSC（pure behavioral，难 instantiate，因为单 observation 看不到）、VS-PROP（base task 没引用 object property）、V-AUG/S-LANG/S-AFF/SB-ADV/SB-NOUN/VB-ROB/VB-SYM 等因为 base task 限制或 instantiation 复杂。

### 4.3 Co-fine-tuning 策略

这是个 methodologically 重要的设计：先在 Bridge V2 pretrain，再收集少量 in-domain demo（put carrot/knife 各 10 个，flip pot/put plate 各 50 个），然后 co-fine-tune。理由是：

1. **明确定义 in-distribution**：不 co-fine-tune 的话，base task 跟 pretraining data 之间的 distribution gap 会污染 generalization 测量
2. **灵活选择 base task**：可以选支持多 axis 的 task
3. **更贴近实际部署**：实际部署通常需要少量 in-domain data

Upsampling rate: put carrot/knife 用 100x，flip pot/put plate 用 50x；π0 用 1000x / 500x（因为 flow-based chunking 需要更多 in-domain signal）。

### 4.4 候选 Models

| Model | Backbone | Params | Action tokenization |
|-------|----------|--------|---------------------|
| OpenVLA | Prismatic-Llama 2 | 7B | Binning (256 bins per dim) |
| MiniVLA | Prismatic-Qwen 2.5 | 1B | Vector quantized chunking |
| π0 reimpl | PaliGemma | 3B | Flow matching action chunk |

技术细节：MiniVLA 用 vector quantization 把 action chunk 压成离散 token，π0 用 flow matching（continuous）生成 action chunk。这两个都比 OpenVLA 的单步 binning 更 expressive，能处理 multi-modal action distribution。

---

## 五、Main Results（Fig. 2）的关键 findings

### 5.1 整体 generalization 很弱

1600+ trials 中，多数 axis 上 SOTA model 的 success rate < 50%。这跟 paper 宣传里的 "generalist" 形成强烈对比。

### 5.2 Semantic generalization 是 systematic weakness

**尽管**所有 model 都用了 internet-scale pretraining 的 LLM backbone（Llama 2 7B, Qwen 2.5 0.5B, PaliGemma），semantic axis 仍然很差。Table VIII 里的 Carrot Counter (S-MO) 全部 0/5——所有 model 都不能理解 "put the object that is on the counter on the plate"。

这暗示：**LLM 的 language understanding 并没有 transfer 到 robotic control 上**。可能原因：
- VLA training 把 LLM 的 representation 破坏了
- Language grounding 需要的是 visual-language alignment，不是单纯 language modeling
- 数据集里的 language annotation 太模板化（参考 [STEER paper](https://steer-robot.github.io/)）

### 5.3 π0 大致最好，但绝对性能仍差

π0 在大部分 axis 上领先，可能因为：
- PaliGemma 是更强的 VLM backbone（siglip-based vision encoder）
- Flow matching action chunking 比 token binning 更适合 continuous control
- 但绝对成功率仍然只有 40-60% 在多数 axis 上

### 5.4 各模型的 weakness pattern

- **OpenVLA**: visual generalization 弱（V-SC, V-VIEW 都差），S-PROP 反而最好（可能 Llama 7B 大）
- **MiniVLA**: visual+behavioral 弱（VB-POSE, VB-ISC）
- **π0**: 最 balanced，但仍然在 S-MO, S-INT 上挣扎

---

## 六、VLA Design Decisions 实验（Fig. 3，390 trials）

### 6.1 Scaling robot data (Bridge vs OXE)

Fig. 3(a)：OpenVLA Bridge-only vs OpenVLA OXE（OXE 包含 Bridge + 20+ 其他 embodiment）。

**Finding**: OXE 改善多个 axis，**但 model 已经挣扎的 axis**（V-VIEW, VB-MOBJ, S-MO）**几乎不改善**。这是个重要 insight：scale data 不是万能药，有些 generalization 需要的不是 more data，而是 data 的特定 diversity（比如 OXE 没多少不同 camera view）。

### 6.2 Scaling LLM backbone

Fig. 3(b)：OpenVLA (Llama 2 7B) vs MiniVLA -VQ (Qwen 2.5 0.5B)，architecture 一致只换 LLM。

**Finding**: 7B 在 semantic axis 上略好，但绝对提升有限，其他 axis 几乎不动。说明 LLM scale 不是 semantic generalization 的瓶颈。

### 6.3 VQA co-training

Fig. 3(c)：OpenVLA (Bridge) vs OpenVLA (Bridge + 20% VQA from LLaVA-1.5)。

**Finding**: 出乎意料地 **mixed effect on semantic axes**：
- S-LANG, S-MO, S-INT 改善
- S-PROP **变差**

Hypothesis：general VQA 让 model 学到 general vision-language alignment，但可能"稀释"了 robot action 的 grounding。Targeted VQA data（比如专门做 object property referencing）可能更好。

### 6.4 Vector quantized action chunking

Fig. 3(d)：MiniVLA (VQ chunking) vs MiniVLA -VQ (binning single-step like OpenVLA)。

**Finding**: VQ chunking 在几乎所有 axis 上更好（除了 VB-ISC）。

**Intuition**: Action chunking 让 policy commit to a sequence，避免 step-by-step 的 multi-modal action uncertainty。公式上，单步 binning 是 $\pi(a_t | o, l)$ 的 categorical，chunking 是 $\pi(a_{t:t+k} | o, l)$ 的 joint，后者能 capture temporal correlation 和 multi-modality（同一 observation 下不同 chunk 路径）。

### 6.5 Compositional generalization（Table IV, 210 trials）

测试 axis 组合：S-PROP+S-LANG, V-SC+V-OBJ, VB-POSE+VB-ISC。

**Finding**: 组合后性能下降明显，但不同 model 的相对 ranking 跟 single axis 一致（π0 在 VB-POSE+VB-NOBJ 上 8/10 最好）。OXE 训练在 S-PROP+S-LANG 上提升明显（8/10 vs 4/10），可能因为 OXE 里 instruction 多样性更高。

---

## 七、对 robotics 社区的方法学启示

### 7.1 关于 generalization 的定义

Paper 主张用 **base task + perturbation** 而不是 "evaluate on task that's OOD from pretraining data"。原因：

- Pretraining data 太大没法精确知道哪些 "seen"，哪些 "unseen"
- Base task 是 controllable 的，perturbation 是 atomic 的，可以 isolate 单一 factor 的影响
- 这本质上是把 generalization 从 "out-of-distribution detection" 变成 "interventional" 评估

### 7.2 关于 benchmark design

Paper 提出的 desiderata：
- Base task 在 pretraining support 内（保证 generalization 而非 OOD failure）
- 一个 base task 支持 multiple axes（提高 ROI）
- 每 axis 1-2 个 factor（practical budget）
- Total ~13 axis x ~5 trials x ~7 model ≈ 1000+ trials

### 7.3 自动化 benchmark design（Appendix B）

Paper 提出用 VLM (Gemini 2.0 Flash) 自动 propose perturbation：
- Input: base scene image + base instruction
- Output: JSON 格式的 perturbed instruction + image edit prompt
- 用 SAM 2.1 做 segmentation 来 recolor（Carrot Red Sink 那种 condition 实际是 inference-time 用 SAM 2.1 Large mask + recolor）

这是个很有意思的方向——把 benchmark design 也变成 model-driven，减少 human bias。

参考：
- SAM 2: https://ai.meta.com/blog/segment-anything-2/
- Gemini 2.0 Flash: https://deepmind.google/technologies/gemini/flash/

---

## 八、Critical observations 和 future directions

### 8.1 局限性

- 只在 Bridge V2 上 instantiate，dataset 本身 diversity 有限（single sink environment）
- 4 个 base task 不够覆盖所有 axis（特别是 VS-PROP 没法 instantiate）
- 1600 trials 在 robotics 算多，但每个 condition 只有 5 trials，统计 confidence 不强
- Human evaluator 判断 success，可能有主观偏差

### 8.2 缺失但重要的 axis

Paper 没深入：
- **Long-horizon generalization**: 多 task sequencing，task reordering
- **Tactile modality**: 把 taxonomy 扩到 visuo-linguo-tactile
- **Cross-environment**: 不同厨房、不同光线条件
- **Failure recovery**: perturbation 导致中间 state 偏离时的 recovery

### 8.3 对 VLA model design 的启示

综合所有 finding：
1. **Semantic generalization 不能靠 LLM scale 解决** → 需要更好的 language grounding，可能需要 denser language annotation（STEER 方向）
2. **Targeted VQA 比 general VQA 好** → co-training data 要 task-aware
3. **Action chunking 是 free lunch** → 大部分 VLA 都该用
4. **Dataset diversity > dataset size** → OXE 在 bottleneck axis 上不 work，说明要 specifically collect camera view、object morphology 等多样性

### 8.4 跟相关工作对比的 framing

Table II 把 prior work 投影到 ‹-Gen 上看，发现：
- **Simulation benchmarks** (FactorWorld, Colosseum, CALVIN): 主要测 visual axes，semantic 少
- **Real-world benchmarks** (Bridge V2, DROID): dataset 本身，不是 benchmark，但被用来测有限 axis
- **Policy papers** (RT-1, RT-2, OpenVLA, π0): 各自测 3-5 个 axis，重复但不一致

‹-Gen 是 superset，包含 22 axis。BridgeV2-‹ 实测 13 axis，已经是 prior work 中最广的。

---

## 九、和 LLM/Vision scaling laws 的类比

Paper 里有个潜台词：robotics 不能简单照搬 LLM 的 scaling law 叙事。在 LLM 里，benchmark（MMLU, GSM8K）相对稳定，scale model 就能涨分。但 robotics 的 generalization 是 **structured** 的——不同 axis 之间几乎 orthogonal，scale data/model 对不同 axis 的效果完全不同。

这呼应了之前一些 robot learning scaling law 的工作：
- Data Scaling Laws in IL: https://arxiv.org/abs/2410.18647
- Efficient Data Collection (Eff-Comp): https://eff-comp.github.io/

Datta scaling 那篇发现 IL 有 log scaling law，但不同 task/scenario 的 coefficient 不同——跟 ‹-Gen 的发现一致：不同 axis 的 "data efficiency" 不一样。

---

## 十、可能的扩展方向（一些联想）

1. **Perturbation magnitude quantification**: Paper FAQ 里提到可以用 image/text embedding distance 或 dynamic time warping 量化 perturbation 的 "edit distance"。这跟 causal inference 里的 intervention strength 类似。

2. **Curriculum learning along axes**: 如果知道某 axis 是 bottleneck，可以在 data collection 时 targeted oversample 该 axis 的 perturbation。这就把 benchmark 直接变成 data collection guidance。

3. **Compositional generalization 的 scaling law**: Paper 测了 2-axis composition，但没测 3-axis。Composition 通常会 exponentially decay（跟 LLM 的 in-context composition 类似）。

4. **Cross-embodiment axis (VB-ROB)**: Paper 没测，但这是 Open X-Embodiment 的核心卖点。值得 instantiate 一个 benchmark 专门测这个。

5. **Active evaluation**: 自动化 benchmark design 那一节其实暗示了一个更激进的 idea——用 RL 或 bandit 算法 active select 哪个 axis 要测，efficiently find model 的 weakness。这跟 active learning 在 vision 里的应用类似。

6. **Generative model 作为 environment**: 用 video diffusion（Sora、Genie 等）生成 perturbed scene，省去物理 setup。这是 robotics sim2real 的 next frontier。

参考：
- Genie: https://arxiv.org/abs/2402.19459
- Sora: https://openai.com/sora

---

## 十一、Summary

这篇 paper 的核心贡献是 **conceptual** 而非 algorithmic：它给 robotics generalization 一个 vocabulary 和 structuring。22 个 axis 不是最终答案，而是一个起点。真正的价值在于：

1. 让 model comparison apples-to-apples
2. 暴露 systematic weakness（semantic generalization 在所有 SOTA VLA 上都很差）
3. 把 "scale data" 这个粗粒度建议细化成 "scale which axis of data"
4. 为 automated benchmark design 提供 schema

对于 build intuition：理解 VLA model 的 generalization 不能看成单一数字，而是 22 维向量。每个 axis 上 model 有不同的 "data efficiency" 和 "architectural inductive bias"。Improving VLA = 识别 bottleneck axis → 设计 targeted data/architecture intervention → 重新评估。这个 loop 比"加更多 data、加更多 param"要 productive 得多。
