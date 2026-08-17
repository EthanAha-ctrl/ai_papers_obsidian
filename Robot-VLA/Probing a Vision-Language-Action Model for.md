---
source_pdf: Probing a Vision-Language-Action Model for.pdf
paper_sha256: 0c7609203e6ba09435d32f8df589239e6e6f0ac9b44af20bad7ef78865d3be8f
processed_at: '2026-08-06T06:29:34-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲这篇paper

## 一句话说清楚

OpenVLA是个end-to-end的黑盒，你给它图和指令，它直接吐7D action。问题是你不知道它"脑子里"在想啥——它到底知不知道碗在哪个位置、有没有抓住、该往哪走？这篇paper的做法就是：**扒开它的中间层，用个linear classifier去"偷听"它内部到底编码了哪些symbolic信息**，然后把这些symbols喂给一个传统的cognitive architecture (DIARC)，让后者当一个"监工"。

## 为什么这事有意思

VLA model很强但不可靠。你让它去抓碗，它可能因为camera稍微挪了一下、灯光变了、或者出现了没见过的东西就翻车。而且它翻车之前你根本看不出来——它不会说"我confused了"，它就是继续输出action，直到撞墙。

Cognitive Architecture (CA)恰好相反：它非常可靠，能做逻辑推理、能检测矛盾、能监控state变化。问题在于它是个"死板"的系统，所有行为都得人手写rule，完全没法处理raw pixels。

所以就很自然想：**能不能让VLA干活，让CA盯着它？** CA需要symbols来reason，VLA不输出symbols，那symbols从哪来？这篇paper说：从VLA的hidden layers里probe出来。

## 具体怎么probe的

OpenVLA的backbone是Llama 2 7B，一共33层hidden states（32个transformer block + 1个embedding layer），每层是个4096维的向量。

做法很直白：
1. 跑LIBERO-spatial的10个pick-and-place task，每个收集5个成功episode
2. 每个timestep同时记下：(a) 33层各自的4096维activation，(b) 当前ground-truth的symbolic state（用detector function从simulator里读出来）
3. 对每一层，train一个linear probe——就是个logistic regression——把4096维向量map到一堆binary labels

他们定义了两类symbolic state：

**Object state**（224个atoms）：
- 关系类：`behind`, `in-front-of`, `inside`, `left-of`, `on`, `on-table`, `right-of`
- 属性类：`open(container)`, `turned-on(on-off-object)`
- 例如 `on(bowl1, plate1)=1` 表示碗在盘子上

**Action state**（12个atoms）：
- `grasped(obj)` — 有没有抓住
- `should-move-towards(obj)` — 该不该往这个物体移动

每个probe就是一个linear layer加sigmoid：

$$\hat{\mathbf{y}} = \sigma(\mathbf{W}\mathbf{h} + \mathbf{b})$$

- $\mathbf{h} \in \mathbb{R}^{4096}$ 是某一层的activation
- $\mathbf{W} \in \mathbb{R}^{n \times 4096}$ 是要学的weight，$n$ 是atoms数量
- $\mathbf{b} \in \mathbb{R}^n$ 是bias
- $\sigma$ 是sigmoid，每个atom独立出一个0到1的概率
- 用binary cross-entropy loss，Adam optimizer

注意这里用的是**multi-label**而非multi-class——每个atom独立预测，不考虑atom之间的互斥关系。这其实是个简化：现实中`on(bowl1, plate1)`和`on-table(bowl1)`不能同时为true，但probe不知道这个约束。好处是避免了 $2^{224}$ 的combinatorial explosion，坏处是可能输出矛盾state。不过矛盾检测恰好是DIARC的活，所以也算分工合理。

## Data preprocessing的几个关键决定

1. **Episode-level split**：整个episode要么在train要么在test，绝不能拆开。不然同一trajectory的frame既在train又在test，probe就只是在memorize trajectory pattern，没意义。

2. **Filter near-constant labels**：如果某个atom在所有frames里 >99% 或 <1% 是true，直接扔掉。比如`on-table(obj)`——大部分物体一直呆在桌上，几乎不变；`turned-on(stove)`——stove永远关着。这些labels如果留着，probe trivially predict "always 0"就能拿99% accuracy，完全没意义，还会artificially inflate整体数字。

3. **不做class balancing**：filter完near-constant之后，剩下的imbalance不算极端，就不额外oversample了。

4. **不standardize features**：直接用raw 4096维float，不做z-score normalization。作者说probe convergence很好不需要。这个我有点存疑——不同layer的activation magnitude差异可能挺大，但作者说works，那就works吧。

## 结果怎样

画了个heatmap：33行（layers）× 9列（7个object predicate + 2个action predicate）。

**主要发现**：

1. **Layer 0（embedding layer）明显差**，所有category都低。这个完全符合预期——embedding layer只是token embedding + positional encoding，还没经过任何transformer block的attention处理，本质上就是个lookup table，没有contextualization，也没有跨modal fusion。它不可能编码`on(bowl1, plate1)`这种high-level spatial relation。

2. **Layer 1-32几乎全都 >0.90 accuracy**。说明Llama backbone的大部分layer都linearly decodable地encode了这些symbolic states。

3. **没有看到object-early / action-late的分层pattern**。作者本来hypothesize：
   - H1: object state应该在early layer编码（因为early layer处理basic visual/spatial features）
   - H2: action state应该在later layer编码（因为action planning需要visual和language信息integrate之后才能形成）
   
   结果两个hypothesis都不支持——object和action的accuracy曲线基本平的，没有明显crossover。

## 为什么hypothesis失败

作者自己的解释：**data太simple了**。

LIBERO-spatial的10个task共享同一套objects，除了2个black bowls位置会变，其他所有objects在所有task里位置完全固定。而且robot永远pick black bowl，永远place到plate上。

后果就是：
- 大部分object relation永远是constant（`left-of(plate, ramekin)`在所有task所有时刻都一样）
- Action state也几乎constant（永远pick bowl，永远往plate走）
- 留下来的variable labels本来就少，pattern又简单
- 任何layer只要稍微encode了一点task-relevant信息，linear probe就能decode出来
- Layer-wise的细粒度differentiation被这种"太容易"的task wash out了

这就是典型的**ceiling effect**——task太简单，所有layer表现都好，你分不出谁更强、谁encode什么。要想真正test H1和H2，需要更多样的objects、更多变的layouts、更多样的goals。

## 还有一个我比较担心的点

>0.90 accuracy看起来很漂亮，但要想清楚这个数字意味着什么。

比如`grasped(bowl1)`这个atom：在一个episode里，grasped state大概只占5-10%的timestep（gripper闭合之后到放下之前那段时间）。如果probe永远predict "0"，能拿~90% accuracy。所以 >0.90 并不一定impressive——要看是否超过这种trivial baseline。

作者filter掉了 <1% / >99% frequency的labels，所以剩下的labels至少有一定fluctuation。但具体的class distribution没报告，positive class的precision/recall也没报告，只有accuracy。如果positive class很少（比如5%），那probe可能偏向predict 0，accuracy高但实际detect不出状态变化。这点上我觉得paper的evaluation还可以更严谨。

## DIARC集成长啥样

整个pipeline：

```
User在GUI里选一个自然语言指令
  → DIARC通过WebSocket发给server
  → Server跑LIBERO + OpenVLA
    → OpenVLA输出7D action（执行）
    → 同时hook出两个best layer的activation
      → Object probe → 224维0/1 array
      → Action probe → 12维0/1 array
  → VLAComponent把array转成DIARC的predicate格式
    → 例如 array[37]=1 → "on(bowl1, plate1)"
  → 存进DIARC的belief store
  → 同时stream到React UI显示
```

**两个best layer**是前面probing experiment里object state和action state各自accuracy最高的层。

DIARC拿到这些predicates之后能干啥：
- **矛盾检测**：如果probe同时输出`on(bowl1, plate1)=1`和`inside(bowl1, drawer1)=1`，DIARC知道这在物理上不可能，可以flag
- **Subgoal verification**：确认`grasped(bowl1)`从0变1了，说明pick subgoal完成
- **Task progress tracking**：跟踪整个trajectory的symbolic state evolution

**React UI**有个timeline slider功能挺酷的：task跑完之后可以scrub回看每个timestep的frame + 对应的symbolic states，看representation是怎么随时间演化的。对debugging和interpretability研究很有用。

## 整个系统的设计哲学

关键design choice是**minimal modification to OpenVLA**——不改policy本身，只是在inference时hook activation出来。这意味着：
- VLA policy保持原样，不degrade
- Probe是external add-on，可以随时加随时撤
- DIARC和VLA通过WebSocket解耦，模块化

目前的系统是**被动监控**——DIARC只observe，不intervene。作者提到future work想让DIARC主动干预，比如检测到矛盾state就override VLA action。这需要更多工程设计：override interface、safety guarantee、formal verification等等。

## 这篇paper的真正贡献

说到底，这篇是个**proof of concept**。它show了：

1. VLA的hidden layers确实linearly encode了symbolic states（至少在simple task上）
2. 可以用linear probe把这些states提取出来
3. 可以real-time集成进cognitive architecture做monitoring

但它也暴露了很多open question：
- Probe在complex task上还work吗？
- Probe能transfer到unseen task吗？
- Probe找到的representation是causal的还是correlational？(只有correlation的话，DIARC基于它reason可能不可靠)
- Linear probe够不够？nonlinear probe能decode更多吗？
- 能不能做到closed-loop intervention而不只是monitoring？

## 我觉得最值得follow-up的方向

如果让我做next step，我会：

1. **换更diverse的data**：LIBERO-spatial太窄了，用LIBERO-90或者Open X-Embodiment的real-world data，重新test H1/H2
2. **做causal probing**：不只看"能不能decode"，而是用activation patching / ablation看"如果改这个representation，VLA的action会不会变"。只有causal的representation才真正meaningingful。
3. **Probe attention pattern**：OpenVLA有visual tokens，可以看Llama的attention是否attended到正确的object region。这比probing activation更能直接说明模型"看哪里"。
4. **Failure prediction**：当probe confidence低（$\hat{y} \approx 0.5$）时，是否correlate with VLA即将fail？如果是，probe可以当early warning system。
5. **Closed-loop intervention**：DIARC检测到inconsistent state后，能不能halt VLA、replan、或者query human？这是真正让hybrid system有用的关键。

## 总结一句话

这篇paper告诉你：**VLA model不是完全的黑盒，它的中间层确实"知道"碗在哪、有没有抓住、该往哪走，只是它不告诉你。你用个linear probe就能把这些信息"撬"出来，喂给symbolic system做reasoning。** 当前是passive monitoring，但roadmap指向active intervention。

Limitations很transparent——data太窄、可能ceiling effect、没test transfer、没causal validation。是个early-stage exploration，但direction promising。

## References

- OpenVLA: https://openvla.github.io/ , https://arxiv.org/abs/2406.09246v3
- LIBERO benchmark: https://libero-project.github.io/ , https://arxiv.org/abs/2306.03310
- DIARC: https://link.springer.com/10.1007/978-3-319-97550-4_11
- Llama 2: https://arxiv.org/abs/2307.09288
- VLA testing (Wang et al.): http://arxiv.org/abs/2409.12894
- Li, Nye, Andreas (implicit meaning in LM): https://aclanthology.org/2021.acl-long.143
- Chen et al. (probing Llama across layers): https://arxiv.org/abs/2312.04333
- Wu et al. (ACT-R + LLM): http://arxiv.org/abs/2408.09176
- Kirk et al. (LLM as knowledge source): http://arxiv.org/abs/2310.06846
- ROME (causal intervention): https://arxiv.org/abs/2202.05462
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- Bommasani et al. (foundation models survey): https://arxiv.org/abs/2108.07258
- Probing survey (Belinkov): https://aclanthology.org/W19-4801/

---

# Probing OpenVLA for Symbolic States: A Detailed Walkthrough

Andrej, 这篇paper的核心问题非常interesting——能不能从一个end-to-end的VLA model内部"撬"出discrete symbolic states, 然后喂给cognitive architecture (DIARC)做reasoning和monitoring? 这其实是把subsymbolic和symbolic两个paradigm粘合起来的一种attempt, 在robotics safety和interpretability上有real value。

## 1. Motivation: 两个world的鸿沟

**VLA models (e.g., OpenVLA)**: end-to-end, 从pixels+language直接predict 7D action $\Delta x, \Delta \theta, \Delta \text{Grip}$, 7B params, 在~1M real-world demonstrations上训练。Strength是generalization, weakness是black-box + 对camera pose, lighting, unseen objects敏感 (Wang et al. 2024, http://arxiv.org/abs/2409.12894)。

**Cognitive Architectures (e.g., DIARC, ACT-R, SOAR)**: symbolic reasoning, predicate logic, 可以做inconsistency detection ("bowl不能同时在plate上和drawer里"), 但execution是rigid predefined的。

Ideal: hybrid——让VLA做low-level visuomotor policy, 让CA做high-level monitoring/reasoning。问题是VLA不输出symbols, 那symbols从哪来? Answer: **probe hidden layers**。

OpenVLA backbone是Llama 2 7B, 有32 transformer blocks + 1 embedding layer = 33 hidden states (indexed 0-32), 每层4096-dim vector。这个layer indexing是关键experimental axis。

References:
- OpenVLA: https://arxiv.org/abs/2406.09246v3
- DIARC: https://link.springer.com/10.1007/978-3-319-97550-4_11
- Octo (related VLA): https://arxiv.org/abs/2405.12213v2

## 2. 系统架构: DIARC-OpenVLA-Probe

Figure 1的整体flow:

```
User Command (NL) 
    → DIARC GUI 
    → VLAComponent (WebSocket) 
    → OpenVLA inference (Llama backbone)
        ├──→ 7D action → LIBERO simulator step
        └──→ hidden layer ℓ_obj, ℓ_act activations
                → Linear Probe_obj → object states (0/1 array, 224 atoms)
                → Linear Probe_act → action states (0/1 array, 12 atoms)
    → DIARC belief store (symbolic predicates)
    → React UI (color-coded state changes)
```

设计要点:
- **Minimal modification to OpenVLA**: 只在inference call时hook hidden layer activations, 不改policy本身
- **Two best layers**: ℓ_obj (object states最佳层) 和 ℓ_act (action states最佳层), 通过probing experiment确定
- **Decoupling via WebSocket**: OpenVLA policy和DIARC symbolic reasoning解耦, 模块化设计
- **Real-time**: ~5 Hz camera feed + streaming state updates

Symbolic predicate format:
- Object: `relation(obj1, obj2)`, `property(obj)`
- Action: `action(obj)`

例如: `on(bowl1, plate1)=1`, `grasped(bowl1)=0`, `should-move-towards(bowl1)=1`

DIARC可以做high-level checks: 例如检测 `on(bowl1, plate1) ∧ inside(bowl1, drawer1)` 这种矛盾state, 或者verify subgoal completion。

## 3. Probing Setup: 细节

### 3.1 Symbolic state定义

**Object state predicates (7类)**:
1. `behind(obj1, obj2)`
2. `in-front-of(obj1, obj2)`
3. `inside(obj, container)`
4. `left-of(obj1, obj2)`
5. `on(obj1, obj2)`
6. `on-table(obj)`
7. `right-of(obj1, obj2)`

Plus unary properties: `open(container)`, `turned-on(on-off-object)`

**Action state predicates (2类)**:
1. `grasped(obj)` — action status
2. `should-move-towards(obj)` — action subgoal

**Atoms计数**: object state有224 atoms (predicate × grounded object combinations), action state有12 atoms。224这个数字看起来大, 但大多是稀疏的——因为只有2个black bowls可pick, place target总是plate, 所以很多combinations是恒定的0。

### 3.2 Data collection

10个LIBERO-spatial tasks, form:
```
"pick up the black bowl {spatial relations identifier} and place it on the plate"
```
e.g., "between the plate and the ramekin", "next to the ramekin", "in the top drawer of the wooden cabinet"。

每个task收集5个成功episodes, 在每个timestep $t$ 同时记录:
- Hidden layer embeddings $\mathbf{h}_t^{(\ell)}$ for $\ell \in \{0, \ldots, 32\}$
- Ground-truth state $\mathbf{y}_t$ (通过detector functions从LIBERO environment提取)

关键: **temporal alignment**——$(\mathbf{h}_t, \mathbf{y}_t)$ 用同一时刻, 避免 $t+1$ 标签配 $t$ embedding的mismatch。

### 3.3 Preprocessing (这里很关键, 影响实验结论)

1. **Episode-level split**: 整个episode要么在train要么在test, 防止temporal leakage。如果同一trajectory的frames跨split, probe可能只是memorize trajectory pattern而非learn representation。

2. **Filtering near-constant labels**: 频率 <1% 或 >99% 的labels丢弃。具体被drop的:
   - `on-table(obj)`: 大部分object一直on table, 太constant
   - `turned-on(obj)`: stove一直off
   
   这点很重要, 否则probe trivially predict "always 0" 拿99% accuracy, 但没意义。

3. **No class balancing**: 不oversample/undersample, 作者认为filtering near-constant已经处理了最extreme的imbalance。

4. **No feature standardization**: 直接用raw 4096-dim float vectors, 不z-score normalize。理由是probe converge well without it。这点其实有点risky——Llama的hidden states在不同layer magnitude差异可能很大, 不normalize可能让某些layer的probe数值不稳定。但作者说converge well, 可以接受。

### 3.4 Probe model

Linear probe, multi-label classification:

$$\hat{\mathbf{y}} = \sigma(\mathbf{W}\mathbf{h} + \mathbf{b}) \tag{1}$$

变量含义:
- $\mathbf{h} \in \mathbb{R}^d$: hidden layer activation, $d = 4096$
- $\mathbf{W} \in \mathbb{R}^{n \times d}$: weight matrix, $n$ = number of tracked atoms (per probe)
- $\mathbf{b} \in \mathbb{R}^n$: bias
- $\sigma$: sigmoid, 给每个atom独立binary probability
- $\hat{\mathbf{y}} \in [0, 1]^n$: predicted probabilities

Loss: binary cross-entropy (每个atom独立, 因为multi-label而非multi-class)
Optimizer: Adam

注意: **single-label classification的combinatorial explosion**——如果有224个binary atoms, 完整state space是 $2^{224}$, 没法train classifier。Multi-label独立预测每个atom是合理的简化, 但牺牲了atom间的correlation (例如 `on(bowl1, plate1)` 和 `on-table(bowl1)` 应该互斥, 但probe不知道这个约束)。

### 3.5 Evaluation

Per-predicate averaged accuracy:

$$\text{acc}(\text{pred}) = \frac{1}{N_{\text{pred}}} \sum_{i=1}^{N_{\text{pred}}} \text{acc}(i) \tag{2}$$

- $N_{\text{pred}}$: 跟踪的该predicate的grounded instances数
- $\text{acc}(i)$: 第i个instance的binary accuracy

例如 `on` predicate可能跟踪 `on(bowl1, plate1)`, `on(plate1, table1)`, 等, 取平均。

## 4. Results & Discussion

### 4.1 Heatmap (Figure 4)

9个columns: 前7个object predicates, 后2个action predicates (`grasped`, `should-move-towards`)。Rows: 33 layers。

观察:
1. **Layer 0 (embedding layer)显著差**: 所有categories都低。符合预期——embedding layer只编码low-level syntactic/lexical features, 不编码visual-semantic relations。
2. **Layers 1-32: 大多 >0.90**: 这说明Llama backbone的中间/后段layers确实linearly decodable地encode了这些symbolic states。
3. **没有object-early vs action-late的pattern**: 这违反了H1和H2 hypotheses。

### 4.2 为什么hypotheses失败?

作者归因于**data diversity不足**:
- 10个tasks共享相同objects (black bowls, plate, ramekin, drawer, cabinet等)
- 除了2个black bowls, 其他objects初始位置across tasks完全一样
- Robot永远pick 1个black bowl, place到plate

后果:
- Object relations大部分constant (e.g., `left-of(plate, ramekin)` 在所有task所有timestep都一样)
- Action states也constant (永远pick bowl, 永远place on plate)
- Linear classification task变得trivial, 任何layer都能decode, wash out了layer-wise差异

这其实是**ceiling effect**——task太简单, probe太容易, 看不到representation的细粒度分化。需要更多variable objects, variable layouts, variable goals才能重新test H1/H2。

### 4.3 这是不是有点"too good to be true"?

我有点怀疑这个 >0.90 的accuracy。可能的原因:
1. Episode-level split但tasks有限——10个tasks可能still让probe学到task-specific spurious features
2. Train/test可能share same object layouts (因为tasks share objects), probe可能只是detect task identity而非真正decode state
3. Filter掉near-constant labels后, 剩下的labels可能集中在几个episode的关键transition frames, 数量少且pattern简单

作者也acknowledge "more and better data is needed"。

## 5. DIARC Integration Details

### 5.1 VLAComponent

每个timestep:
1. OpenVLA outputs 7D action → LIBERO simulator执行
2. 并行提取hidden layer activations (ℓ_obj, ℓ_act两层)
3. Probes输出 0/1 arrays (object 224-dim, action 12-dim)
4. VLAComponent转成DIARC predicate format, 更新belief store
5. DIARC可以做inconsistency detection, subgoal verification, task progress tracking

### 5.2 WebSocket server

轻量级real-time通信:
- User选command → DIARC → WebSocket → server
- Server runs LIBERO + OpenVLA inference → 7D action + hidden states
- Probe inference → symbolic states
- Bundle: base64 camera frame + states + timestep → stream back → DIARC + React UI

### 5.3 React UI (Figure 2)

- Live camera feed (~5 Hz), top-left
- Symbolic states panel, 右边, color-coded:
  - **绿色**: predicate newly activated (0→1)
  - **红色**: predicate deactivated (1→0)
- Timeline slider (task完成后可用): scrub回看每个timestep的frame + states, 分析model的representation evolution

这个timeline scrubbing功能很nice, 对debugging和interpretability研究有用——可以看到symbolic state是怎么随hidden representation演化的, 哪个时刻probe变confident, 是否和environment真实变化对齐。

## 6. Critical Thoughts

### 6.1 Probing的limitations

Linear probe只能capture linearly decodable的信息。如果symbolic state是以nonlinear way encode的 (例如在activation的某个manifold上), linear probe会miss。但linear probe是standard choice, 因为它trains fast, 不容易overfit, 容易interpret。

可能的improvement:
- Nonlinear probe (MLP), 但要小心overfitting
- Causal probing (intervention on activations) 验证causal而非correlation
- Attention pattern probing: 看Llama的attention是否指向relevant objects in image tokens

### 6.2 Generalization concerns

Probes在LIBERO-spatial上train的, 能不能transfer到其他tasks? 真实robot? 不同camera pose? 这些作者没test。如果probe是task-specific的, 那DIARC integration在new task上会fail。

### 6.3 Reactive vs proactive

当前架构是probe监控VLA的state, DIARC被动接收。作者提到future work: DIARC主动intervene when detect inconsistent states。这是关键——如果CA只能observe不能act, value有限。Active intervention需要:
- Detect error/inconsistency的mechanism
- Override或correct VLA action的interface
- Safety guarantee的formal verification

### 6.4 和interpretability literature的关系

这篇工作类似mechanistic interpretability的probing approach (Anthropic的circuits, https://transformer-circuits.pub/), 但更applied。Linear probe是"what's decodable"而非"what's causally used", 后者需要ablation/intervention experiments。

Reference: Belinkov & Glass 2019 "Analysis Methods in Neural Language Processing" https://aclanthology.org/W19-4801/

## 7. Related Work Context

### 7.1 CA + LLM integration (前人)
- **Wu et al. 2024** (http://arxiv.org/abs/2408.09176): Llama-2 13B上train linear classifier预测ACT-R expert decisions, 用Llama hidden representations做cognitive decision modeling。还试了fine-tune Llama来inject ACT-R knowledge。
- **Bajaj et al.**: NLP pipeline + LLM把unstructured text → structured knowledge for ACT-R analogical reasoning。
- **Kirk et al. 2023** (http://arxiv.org/abs/2310.06846): 3种LLM knowledge extraction方法——indirect (LLM response存knowledge store), direct (agent query LLM parse output), direct encoding (LLM生成programs直接跑)。

这篇paper是第一个CA + VLA集成 (作者claim "to the best of our knowledge")。

### 7.2 Probing literature
- **Prompt-based probing** (Petroni et al. 2019, https://arxiv.org/abs/1909.01066): LLM fill-in-blank, 但不适用VLA (VLA不output tokens)
- **Linear probing** (Chen et al. 2023, https://arxiv.org/abs/2312.04333): 跨layer/size evaluate Llama的高阶reasoning/calculation
- **Li, Nye, Andreas 2021** (https://aclanthology.org/2021.acl-long.143): 探索neural LM的implicit meaning representations, 这篇是methodology closest

### 7.3 VLA testing
Wang et al. 2024 (http://arxiv.org/abs/2409.12894) 对VLA做empirical evaluation, 发现对camera pose, lighting, unseen objects敏感。这motivates了why need CA monitoring。

## 8. 实验数据的Intuition Building

### 8.1 为什么layer 0差?

Llama 2的"layer 0"是embedding layer (token embedding + positional encoding, 没经过transformer block)。它主要编码lexical identity和position, 没有contextualization, 也几乎没有跨modal fusion (因为image tokens刚被project进去, 还没被attention整合)。所以semantic relation信息基本没有。

### 8.2 为什么layer 1-32都好?

可能解释:
1. Llama的residual stream会持续accumulate information, 任何layer都能linearly access到
2. Symbolic states其实在early layer就被encode了 (因为input image + instruction已经disambiguate了task), 中间layers只是refine
3. Data太简单, 任何non-trivial representation都sufficient to decode

如果第3点是对的, 那作者的方法在complex tasks上可能degrade, 才能真正test layer-wise differentiation。

### 8.3 Probe accuracy的ceiling

Consider `grasped(bowl1)`: 整个trajectory里大概5-10%的timestep是grasped state (after close gripper, before release)。如果probe只predict "0" 能拿~90% accuracy。所以 >0.90 的accuracy不一定impressive, 要看是否超过这种trivial baseline。

作者filtering了frequency <1% or >99%的labels, 所以remaining labels至少fluctuate。但具体distribution没报告, hard to judge。

## 9. Future Directions (作者提到 + 我的扩展)

### 9.1 作者提到的
- Scale up: 更多variable objects, variable layouts, variable goals
- DIARC主动intervene on contradictory states
- Explore CA reasoning如何enhance/monitor VLA performance

### 9.2 我想加的
1. **Causal probing**: 用activation patching (Meng et al. 2022 ROME, https://arxiv.org/abs/2202.05462) 验证probe找到的representation是否causally used by VLA
2. **Cross-task transfer**: 在LIBERO-spatial上train的probe能否transfer到LIBERO-object或LIBERO-goal?
3. **Failure detection**: 当probe confidence低 (e.g., $\hat{y} \approx 0.5$) 时, 是否correlate with VLA即将fail? 如果是, 可以做predictive safety monitor
4. **Hierarchical probing**: 不仅probe atomic predicates, 还probe higher-level task structure (subgoal sequence, phase transitions)
5. **Active learning**: 用probe的uncertainty guide数据收集, sample低confidence frames重新annotate
6. **Multimodal attention probing**: OpenVLA有visual tokens, 可以probe attention pattern看模型是否attended to correct object
7. **Comparison with VLM probing**: probe LLaVA, PaLI等pure VLM (no action)的相同layers, 看VLA training是否induces不同的symbolic representation

## 10. 总结

这篇paper是first step of一个potentially impactful方向: 把neural end-to-end policy和symbolic cognitive architecture桥接起来。Contribution是:

1. **Methodology**: linear probing on VLA hidden layers for symbolic state extraction
2. **System**: DIARC-OpenVLA WebSocket integration + React UI for real-time monitoring
3. **Empirical**: 33 layers × 9 predicates的heatmap, 发现>0.90 accuracy但layer-wise differentiation不显著
4. **Hypothesis testing**: H1 (object early) 和 H2 (action late) 都not supported, 归因data diversity不足

Limitations很transparent: data太narrow, 可能ceiling effect, 没test transfer, 没causal intervention。

Andrej, 如果要做follow-up, 我会建议先解决data diversity问题——用LIBERO-90或者real-world data (e.g., Open X-Embodiment, https://robotics-transformer-x.github.io/), 然后做causal probing确认representation的causal role, 最后build closed-loop intervention system。当前这篇是proof-of-concept, 真正的hybrid CA-VLA agent还需要很多work。

Reference links:
- OpenVLA: https://openvla.github.io/, https://arxiv.org/abs/2406.09246v3
- LIBERO benchmark: https://libero-project.github.io/, https://arxiv.org/abs/2306.03310
- DIARC: https://link.springer.com/10.1007/978-3-319-97550-4_11
- Llama 2: https://arxiv.org/abs/2307.09288
- Probing surveys: https://aclanthology.org/W19-4801/
- ROME (causal): https://arxiv.org/abs/2202.05462
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- VLA testing (Wang et al.): http://arxiv.org/abs/2409.12894
- Li, Nye, Andreas probing: https://aclanthology.org/2021.acl-long.143
- Wu et al. ACT-R+LLM: http://arxiv.org/abs/2408.09176
- Kirk et al. LLM as knowledge source: http://arxiv.org/abs/2310.06846
- Foundation models survey (Bommasani et al.): https://arxiv.org/abs/2108.07258
