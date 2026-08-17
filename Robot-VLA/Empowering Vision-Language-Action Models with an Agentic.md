---
source_pdf: Empowering Vision-Language-Action Models with an Agentic.pdf
paper_sha256: 7bd86141c839428aa7c1133b96df185656715326c8264161ed93471a8172a233
processed_at: '2026-08-04T04:11:53-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VLA² 用人话讲

## 一句话总结

**当VLA遇到没见过的东西（比如茅台酒、青花瓷碗），不要让它硬猜，而是在它动手之前，先帮它把这个陌生的东西"翻译"成它认识的东西。**

---

## 问题是什么

你训练了一个很厉害的robotic arm model叫OpenVLA。它见过black bowl、wine bottle、wooden cabinet这些常见物体，能按指令抓取放置。success rate 76.5%，还不错。

现在你给它一个新任务："把茅台酒放到白色柜子上"。

问题来了——它从来没见过茅台酒长什么样。它的反应就是：懵了，然后random action，success rate掉到32%。

这就尴尬了。你不能每次出现新东西都重新train model吧？Moutai今天火，明天可能又出来个新的什么酒，你train不完。

---

## 传统思路vs VLA²的思路

**传统思路**（RT-2那种）：把web data和robot data一起train，让model在pretraining阶段就见过世界上所有东西。问题是train一次要海量GPU，而且新概念出现了你没法及时更新。

**VLA²的思路**：我压根不改model。我在model外面包一层"翻译官"，遇到陌生概念时，翻译官先去web上搜图、搜描述，把这个陌生概念拆解成model认识的elemental attributes，然后告诉model"你就把它当成wine bottle来抓就行了"。

这个就是paper里说的**instant learning**——不重新训练，instant地把新概念assimilate进model的已知世界。

---

## 整个系统怎么运转

想象你在指挥一个没出过村的机器人干活。你说"把那个 sushi 拿过来"，它不知道sushi是什么。VLA²的做法是：

### Step 1: Planner拆任务
你说"把茅台酒放到柜子上"，Planner（GLM-4.1V）把这句话拆成：
```
1. pick up the moutai (object: moutai)
2. place the moutai on the cabinet (object: moutai, location: cabinet)
```

### Step 2: Vision模块辨认东西
系统看第一帧画面，让detector找moutai在哪。detector说"我找不到，我不认识moutai"。

这时候触发**web search**：
- 从Bing下载moutai的图片，拼成2×3的collage
- 把collage + 当前场景图 + "moutai"这个词一起喂给GLM，问它"这玩意儿有什么特征？"
- GLM返回5个keywords：`["bottle", "white", "red ribbon", "liquor", "glass"]`
- 把这些keywords当作enriched prompt重新让detector找
- 找到bbox了！用SAM2.1生成pixel-level mask
- 给mask染上红色（object的颜色），给stove染上蓝色（location的颜色）
- 把这些信息cache到memory里，下次再遇到moutai就不用重新搜了

### Step 3: Language模块改指令
现在图上moutai被红色mask盖住了，stove被蓝色mask盖住了。但文字指令里还是"moutai"，model不认识。

Language模块做一件事：把"moutai"替换成KnownList里model认识的词。GLM根据web collage、keywords、KnownList综合判断，决定把"moutai"映射成"wine bottle"。

最终指令变成：
```
1. pick up the red-mask wine bottle
2. place the red-mask wine bottle on the blue-mask cabinet
```

这下model完全认识每个词了，每个词对应的图上区域也用颜色标出来了。

### Step 4: VLA执行
OpenVLA看到的是一个"red-mask wine bottle on blue-mask cabinet"的任务——这对它来说是标准in-distribution task，正常执行就行。

### Step 5: Verifier监控
每执行20步，Qwen2.5-VL检查一下"wine bottle被抓起来了吗？""放到cabinet上了吗？"。如果检测到end-effector卡住了，强制让它先抬起gripper，再继续。

---

## 关键insight：为什么这个框架work

### Insight 1: OOD本质是representation mismatch
model失败不是因为不会抓东西，而是因为它不知道要抓什么。一旦你把"抓什么"这个问题翻译成它认识的representation，它原本的抓取技能就完全够用了。

这就像你让一个只会做中餐的厨师做日料——他不是不会切菜翻炒，他只是不知道"sushi"是什么。你告诉他"sushi就是米饭团子上面放鱼片"，他立马就能做。

### Insight 2: Vision mask解决"在哪"，Language replace解决"是什么"
两个模块各司其职：
- **Mask**告诉model"这个东西在图上的哪个位置"——解决localization问题
- **Replace**告诉model"你要抓的东西等价于你认识的什么"——解决semantic understanding问题

Ablation显示replace比mask更重要（-25.0 vs -11.4），因为如果model都不知道要抓什么，知道位置在哪也没用。

### Insight 3: Memory让"第一次贵，后续便宜"
第一次遇到moutai时，整套cognition pipeline要跑20秒（web search + GLM inference + SAM segmentation）。但一旦做完，moutai→wine bottle的mapping就存在memory里了。下次再遇到moutai，直接查memory，几乎0成本。

这就是Table IV里Vision/Language模块avg time只有0.5s的原因——大部分task都命中了memory cache。

### Insight 4: Prompt format本身就是OOD/ID的边界
Agentic Robot原版把"put the blue white porcelain bowl in the cabinet"分解后，subgoal偏离了训练分布，导致verifier一直过不了第一关，SR=0。

VLA²改成了：
```
"now do 'pick up the wine bottle', the whole task is 'put the wine bottle on the cabinet'"
```

这种**dual conditioning**让model既知道当前subtask，又知道整体goal，保持了和训练时的分布一致。单这一项prompt engineering就贡献了巨大gap。

---

## 三个难度等级的设计哲学

| Level | 修改 | 考验什么 |
|-------|------|---------|
| Easy | black bowl → orange bowl | 纯color shift，model应该能靠visual泛化 |
| Medium | 多个object改名改色 | 组合OOD，开始挑战model的concept边界 |
| Hard | 引入Moutai、青花瓷这种文化specific concept | 彻底OOD，pretraining data里根本不存在 |

Hard level的设计很巧妙——Moutai是真正的zero-shot concept，不是简单的color变种。这逼着framework必须真的去做web retrieval + semantic replacement，不能靠visual泛化蒙混过关。

---

## 结果有多惊艳

Hard level上：
- OpenVLA：32%
- OpenVLA-OFT（更强backbone）：47.4%
- π₀（更强backbone）：60%
- **VLA²：76.2%**

VLA²用的是OpenVLA做backbone，但Hard level SR比OpenVLA高44.2%，比π₀高16.2%。这说明**agentic framework的OOD处理能力 > backbone model的规模优势**。

更重要的是，in-domain task上VLA²没有degradation——Class 2 average 80.1%，比raw OpenVLA还高3.6%。这是很难做到的，因为加模块通常会引入noise。

---

## 我看到的一些有趣联想

### 联想1: 这本质是"工具调用替代权重更新"
传统ML的paradigm是：新概念→收集data→retrain→更新weight。VLA²的paradigm是：新概念→调用web工具→生成mapping→不改weight。

这和RAG对fine-tuning的替代是同一个pattern。未来可能越来越多的"knowledge"存在external tools和memory里，model weight只负责reasoning和execution skill。

### 联想2: "Instant Learning"和人类的类比
人类遇到没见过的东西（比如第一次见到durian），不会重新训练大脑神经元。我们会：
1. 看一眼它的形状、颜色、texture
2. 问别人"这是什么？像什么？"
3. 得到"它就像带刺的菠萝"这种analogy
4. 把"durian"这个concept锚定到已有的"pineapple"concept上

VLA²做的完全一样：web search = 问别人，keywords = 类比描述，replace = 锚定到已知concept。

### 联想3: System 1 vs System 2
VLA model本身是System 1——快速、自动、pattern-match。
VLA²的cognition module是System 2——慢、deliberate、reasoning。

遇到familiar task时，System 1直接处理。遇到OOD时，System 2介入，把问题转换成System 1能处理的形式，再交给System 1。这和Kahneman的双系统理论完美对应。

### 联想4: Token-level alignment的胜利
Ablation里replace（-25.0）> mask（-11.4）> web（-11.0）。这说明在VLA场景下，**language token的alignment比visual region的alignment更critical**。

这其实呼应了LLM领域的发现——token-level understanding往往是dominant的，visual feature更多是auxiliary。VLA²的实验数据给这个直觉提供了robotics场景的证据。

### 联想5: Memory hierarchy的设计
VLA²其实构建了一个简单的memory hierarchy：
- **短期memory**：当前task的vision memory + text memory（JSON replace map）
- **长期memory**：跨task的KnownList vocabulary

这和human memory的sensory memory → working memory → long-term memory hierarchy有结构上的相似性。未来可以扩展出更丰富的memory层级。

---

## 局限性和我的思考

paper自己承认几个问题：
1. Framework结构比较rigid，缺乏autonomy
2. 没做real-world experiment
3. 224×224分辨率有perception bottleneck

我看到的其他潜在问题：

1. **Web search的domain gap**：web上的moutai图片和simulator里的moutai可能差很远（光照、角度、背景）。这会让GLM生成的keywords有bias。paper在butter-bowl这个task上的失败（SR=22）可能就是这个原因——butter在simulator里是low-res的ambiguous blob，web image对不上。

2. **KnownList的coverage**：如果OOD concept实在无法映射到KnownList里的任何词（比如遇到"光剑"这种完全没analogy的），GLM会返回NONE，replace失败。paper没有讨论这种情况的fallback。

3. **Color coding的scalability**：如果场景里有很多objects，color palette会用完。paper用了两组palette（objects一组，locations一组），但没说每组多少颜色。scalability是个open question。

4. **Verifier的false negative**：如果Verifier误判subtask没完成，会一直卡在当前subtask。paper的recovery mechanism（lift gripper + resume）只是heuristic，没有learning component。

---

## 对你的启示

Karpathy，从你的视角我觉得有几个点特别值得琢磨：

**1. Agent是foundation model的"外挂大脑"**
与其让一个giant model装下所有knowledge，不如让model专注reasoning+execution，knowledge通过tool use动态获取。VLA²是这个paradigm在robotics的 instantiation。

**2. OOD问题在robotics比NLP更棘手**
NLP里OOD最多是回答不好，robotics里OOD直接导致物理failure。所以robotics对OOD handling的要求更高，VLA²这种explicit OOD→ID conversion可能是必要的，而不只是nice-to-have。

**3. Memory-based agent可能是方向**
当前VLA²的memory还很简单（JSON replace map），但如果扩展成更rich的episodic + semantic memory，可能实现真正的lifelong learning robot。这是[MemoryVLA](https://memoryvla.github.io/)和[RoboMemory](https://arxiv.org/abs/2502.20805)在探索的方向。

**4. Simulation evaluation的局限**
LIBERO simulation的success rate能说明问题，但real-world的noise、occlusion、deformation比simulation复杂得多。VLA²在simulation上work，不代表real-world能work。sim-to-real的gap仍然是个open challenge。

参考：
- [VLA² Project Page](https://vla-2.github.io)
- [OpenVLA GitHub](https://github.com/openvla/openvla)
- [LIBERO Benchmark](https://libero-project.github.io/)
- [RT-2 Paper](https://robotics-transformer2.github.io/)
- [Agentic Robot Paper](https://arxiv.org/abs/2501.13535)
- [MemoryVLA](https://memoryvla.github.io/)
- [SAM 2](https://ai.meta.com/sam2/)
- [Cutie: Video Object Segmentation](https://github.com/hkchengrex/Cutie)
- [π₀: Vision-Language-Action Flow Model](https://arxiv.org/abs/2410.24164)
- [GLM-4.1V](https://github.com/THUDM/GLM-4)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)

希望这个"人话版"能帮你建立起对VLA²的直觉——它本质上是一个**"OOD翻译官" + "memory cache" + "execution monitor"**的agent framework，把VLA model从"必须见过才能做"的局限中解放出来。

---

# VLA²: 通过Agentic Framework增强Vision-Language-Action Models的OOD能力

## 一、Motivation与核心问题

当前的VLA models（如OpenVLA、π₀、GR-3等）在大规模robotic data上pre-train后，在in-domain tasks上表现优异，但面对**完全unseen的object concepts**时，performance会急剧下降。这个问题的本质是：

$$\text{Performance gap} = \text{SR}_{\text{ID}} - \text{SR}_{\text{OOD}} \gg 0$$

其中 $\text{SR}_{\text{ID}}$ 是in-distribution success rate, $\text{SR}_{\text{OOD}}$ 是out-of-distribution success rate。例如OpenVLA在LIBERO-Hard上，ID时SR可达76.5%，但OOD时掉到32.0%，gap高达44.5%。

传统的解法是joint training robotic data with web-scale multimodal data（如RT-2的做法），但这样有两个痛点：(1) 训练资源开销极大；(2) 新概念出现时无法迭代更新model。VLA²换了个思路——**不改model，改输入**，把OOD的输入在送入VLA之前转换成ID的表示。

参考：
- [OpenVLA Project](https://openvla.github.io/)
- [RT-2 Paper](https://robotics-transformer2.github.io/)
- [LIBERO Benchmark](https://libero-project.github.io/)

---

## 二、核心Intuition：Instant Learning

VLA²借鉴了认知科学中的**prior knowledge reactivation**理论。paper引用了三个认知科学发现：

1. **Brod et al. (2013)**：accessible prior knowledge facilitates comprehension and memory of new information
2. **van Kesteren et al. (2018)**：successful knowledge construction happens through reactivation of previously learned information
3. **Bein et al. (2020)**：adaptive memory rarely learns tabula rasa, but builds on prior knowledge

VLA²把这套理论工程化为：遇到unseen concept（比如"Moutai"茅台酒），不去让model从头学，而是通过web retrieval把"Moutai"分解成elemental attributes（如"liquor bottle"、"white"、"red ribbon"），然后映射到model训练时见过的KnownList词汇（如"wine bottle"）。这就是"instant learning"。

---

## 三、System Architecture深度解析

整个framework分三大模块，对应Figure 2的三个block：

### A. Preliminary Information Processing

#### A.1 Planner

用**GLM-4.1V-9B-Thinking**（locally deployed）做task decomposition。输入是natural language instruction $T$，输出是subtask sequence $S = \{s_1, s_2, ..., s_n\}$，每个subtask $s_i = (v_i, o_i, l_i)$，其中：
- $v_i \in V$：action verb，$V = \{\text{pick up}, \text{place}, \text{open}, \text{close}, \text{turn on}, \text{turn off}\}$
- $o_i$：object name
- $l_i$：location name

Planner prompt有严格约束——每个subtask**只能包含一个action verb**，且必须explicitly specify objects和locations。post-processing有三层fallback：
1. 自动linguistic extraction
2. 检测到error时regeneration
3. 超过error tolerance时hard-coded parsing

这种设计保证无论GLM输出什么，downstream都能拿到valid的structured info。

#### A.2 Vision Pre-processing

用**MM-GroundingDINO**（在500张LIBERO渲染图像上fine-tuned）做object detection，输出bounding box list $B = \{b_1, b_2, ..., b_m\}$。由于recognition failure或post-processing不足，有些bbox可能为empty，需要后续Cognition模块处理。

参考：
- [MM-GroundingDINO](https://github.com/open-mmlab/mmdetection)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

---

### B. Cognition & Memory（核心创新）

这是整个framework的灵魂，负责把所有OOD信息转换成VLA能理解的ID表示。

#### B.1 Vision Module

**Double Judgment机制**：对每个word $w$（object或location），并行检查两个条件：

$$\text{Need search} = \begin{cases} \text{True}, & \text{if } b(w) = \emptyset \lor K(w) = \emptyset \\ \text{False}, & \text{otherwise} \end{cases}$$

其中 $b(w)$ 是bounding box，$K(w)$ 是auxiliary keywords。任一缺失就触发web search branch。

**Visual Search Pipeline**：

1. **Web Image Retrieval**：用[bid](https://github.com/ostrolucky/Bulk-Bing-Image-downloader)从Bing下载web images，组成2×3 collage
2. **GLM Understanding (Vision)**：输入 = first image + web collage + word $w$，输出 = 5个descriptive keywords $K(w) = \{k_1, k_2, k_3, k_4, k_5\}$，涵盖color、shape、function、size等elementary attributes
3. **Cache到Vision Memory**：$M_v(w) = \{K(w), \text{images}, \text{collage}\}$，供后续任务reuse
4. **Re-detection**：用enriched prompt（原文 + keywords）重新送给detector

**SAM2.1-L Segmentation**：拿到valid bbox后，用SAM2.1-L生成pixel-accurate mask $M = \text{SAM}(b)$，记录term-color assignment $\{(w_i, c_i)\}$。

**Color Encoding策略**：
- Objects用一组color palette：$C_{obj} = \{c_1^{obj}, c_2^{obj}, ...\}$
- Locations用另一组palette：$C_{loc} = \{c_1^{loc}, c_2^{loc}, ...\}$
- 这样downstream VLA能通过color cue区分object和location的role

**VOS (Video Object Segmentation)**：用[Cutie](https://github.com/hkchengrex/Cutie)做temporal-consistent mask propagation。第一帧由SAM生成mask，后续帧用Cutie的memory mechanism推断mask，形成连续的color-coded mask flow。这一步**每个timestep都执行**（其他cognition只在task开始时执行一次）。

参考：
- [SAM 2](https://ai.meta.com/sam2/)
- [Cutie Project](https://hkchengrex.github.io/Cutie/)

#### B.2 Language Module

**目标**：把prompt中所有OOD的object tokens替换成KnownList中的ID vocabulary。

**Double Judgment for Text**：
$$\text{Substitution} = \begin{cases} \text{KnownList}[w], & \text{if } w \in \text{KnownList} \\ \text{GLM}(w, \text{context}), & \text{if } w \notin \text{KnownList} \\ \text{NONE}, & \text{if no valid replacement} \end{cases}$$

**GLM Understanding (Text)的输入非常丰富**：
1. First image with cropped bbox regions + scores
2. Web collage (or NONE)
3. Original prompt
4. Web-derived keywords (or NONE)
5. Known vocabulary list
6. External API auxiliary info

GLM输出一个replacement word $\hat{w}$，如果valid，存入text memory $M_t(w) = \hat{w}$，下次直接reuse。

**Text Memory存储格式**：JSON file的replace map，例如：
```json
{
  "blue white porcelain bowl": "black bowl",
  "moutai": "wine bottle",
  "butter": "cream cheese"
}
```

这种design让OOD的"Moutai"→"wine bottle"，"blue white porcelain bowl"→"black bowl"的映射在第一次cognition后永久cache，后续相同concept的task直接复用。

---

### C. Judgment & Execution

#### C.1 Verifier (Judgment)

用**Qwen2.5-VL-3B-Instruct**（在LIBERO videos上fine-tuned），判断subtask是否完成。Verifier prompt根据verb $v$ 动态构造：

```python
if verb == "pick up":
    prompt = f"has '{object}' been grasped and lifted off any surface?"
elif verb == "place":
    prompt = f"has '{object}' been placed '{location}' and is gripper away?"
elif verb in ("turn on/off", "open/close"):
    prompt = f"has '{target}' been {action_text}?"
```

**Recovery Mechanism**：用dynamic threshold检测end-effector是否stuck。一旦flagged，强制设置current task为"lift the gripper"，经过fixed number of steps后resume原subtask。

#### C.2 Execution (VLA)

**OpenVLA Fine-tuning**的关键改动：
1. **Visual modality改造**：原始RGB videos替换为transparent colored masks augmented videos
2. **Task prompt reformulation**：从原始单一prompt改为temporally segmented plan-based prompts

训练时prompt格式：
$$P_t = \text{"now do } s_t\text{, the whole task is } \bigcup_{i=1}^{n} s_i\text{"}$$

其中 $s_t$ 是当前subtask，$\bigcup s_i$ 是全部subtask的joint。这种dual-context conditioning让VLA同时知道**当下要做什么**和**整体目标**。

**KnownList构建**：用NLTK对训练集做tokenization + POS tagging，聚合出JSON vocabulary，存到model里供inference时使用。

参考：
- [OpenVLA GitHub](https://github.com/openvla/openvla)
- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

---

## 四、Experimental Setup深度解读

### 4.1 Custom Environment设计

paper基于LIBERO-Spatial和LIBERO-Goal构造了三个OOD难度递增的环境：

| Level | 修改内容 | OOD类型 |
|-------|---------|---------|
| **Easy** | black bowl → orange bowl | Color shift |
| **Medium** | black bowl → white bowl; wine bottle → blue bottle; wooden cabinet → white cabinet | Multiple color+name changes |
| **Hard** | wine bottle → Moutai (茅台); black bowl → blue-white porcelain bowl; cream cheese → butter | Semantic reinterpretation + synonym substitution |

Hard level的设计很巧妙——Moutai和blue-white porcelain是中国文化specific的concept，在OpenVLA的训练数据中几乎不可能出现，是真正的zero-shot OOD。

每个task执行50次，report整体SR和每个task的SR。

### 4.2 Baselines

- **Class 1**（更强VLA backbone）：OpenVLA-OFT, π₀, π₀-FAST
- **Class 2**（OpenVLA family）：OpenVLA (FT), Agentic Robot, VLA²

这种分类很重要——VLA²用OpenVLA做backbone，所以最fair的对比是Class 2内部。

参考：
- [π₀ Paper](https://arxiv.org/abs/2410.24164)
- [π₀-FAST](https://www.physicalintelligence.company/blog/pi0)
- [Agentic Robot Paper](https://arxiv.org/abs/2501.13535)

---

## 五、Main Results深度分析

### 5.1 Table I：In-Domain Performance

| Method | Spatial | Object | Goal | Long | Average |
|--------|---------|--------|------|------|---------|
| OpenVLA-OFT (Class 1) | 97.6 | 98.4 | 97.9 | 94.5 | **97.1** |
| π₀ (Class 1) | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| Agentic Robot (Class 2) | 85.8 | 89.0 | 81.8 | 61.6 | 79.6 |
| OpenVLA (Class 2) | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| **VLA² (Class 2)** | 86.4 | 86.2 | 83.2 | 64.4 | **80.1** |

**关键观察**：
1. VLA²在Class 2中average最高（80.1%），且**没有性能退化**——这是非常难做到的，因为adding agentic modules通常会引入noise
2. Object suite上VLA²（86.2%）略低于Agentic Robot（89.0%）和OpenVLA（88.4%），原因paper分析是**perception bottleneck**：224×224 resolution + imprecise object names + web images和simulator views的domain gap
3. Long suite上VLA²（64.4%）显著优于Agentic Robot（61.6%）和OpenVLA（53.7%），说明plan-based prompt对long-horizon task特别有效

### 5.2 Table II：Out-of-Distribution Performance

| Method | Easy | Medium | Hard | Average |
|--------|------|--------|------|---------|
| OpenVLA-OFT | 98.8 | 95.4 | 47.4 | 80.5 |
| π₀ | 97.2 | 86.0 | 60.0 | 81.1 |
| π₀-FAST | 98.0 | 75.2 | 45.8 | 73.0 |
| Agentic Robot (RP) | 83.8 | 48.6 | 26.2 | 52.9 |
| OpenVLA | 85.0 | 66.7 | 32.0 | 61.2 |
| **VLA²** | **86.6** | **81.6** | **76.2** | **81.5** |

**惊人的发现**：
1. **Hard level上VLA²（76.2%）比π₀高16.2%，比OpenVLA-OFT高28.8%**——尽管π₀和OFT的backbone更强
2. **Average SR上VLA²（81.5%）甚至略高于π₀（81.1%）**——agentic framework弥补了backbone的劣势
3. **Degradation analysis**：从Easy→Hard，VLA²只掉10.4%，而OpenVLA掉53%，OpenVLA-OFT掉51.4%

这说明VLA²的OOD→ID转换pipeline确实有效。

### 5.3 Table III：Task-Level Deep Dive

Hard environment上的task-level breakdown非常informative：

| Task | New Items | VLA² | π₀ | OpenVLA | OpenVLA-OFT |
|------|-----------|------|-----|---------|-------------|
| stove | 0 | 96 | 98 | 96 | 100 |
| open-drawer | 0 | 78 | 94 | 40 | 100 |
| drawer-bowl | 1 | 62 | 66 | 14 | 92 |
| saucer-stove | 1 | 84 | 88 | 84 | 8 |
| bowl-stove | 1 | 86 | 92 | 52 | 88 |
| moutai-rack | 1 | **72** | 44 | 0 | 0 |
| bowl-saucer | 2 | **88** | 16 | 2 | 0 |
| bowl-cabinet | 2 | 86 | 68 | 30 | 82 |
| butter-bowl | 2 | 22 | 0 | 2 | 0 |
| moutai-cabinet | 2 | **88** | 34 | 0 | 4 |

**几个insight**：

1. **moutai-rack**（1 new item）：VLA²=72, π₀=44, 其他几乎为0。Moutai是纯OOD concept，只有VLA²能通过web retrieval把"Moutai"→"wine bottle"的mapping建立起来
2. **bowl-saucer**（2 new items）：VLA²=88, π₀=16。两个OOD token同时出现时，VLA²的replace机制优势巨大
3. **butter-bowl**（2 new items）：VLA²只有22，所有方法都低。Paper解释是butter在low resolution下视觉ambiguous，即使human都难verify
4. **stove**（0 new items）：所有方法都很高（96-100），说明ID task上VLA²不degradation

---

## 六、Ablation Study深度分析

### 6.1 w/o mask (-11.4)

| Task | VLA² | w/o mask | Δ |
|------|------|----------|---|
| open-drawer | 78 | 52 | -26 |
| bowl-cabinet | 86 | 64 | -22 |
| moutai-rack | 72 | 36 | **-36** |
| moutai-cabinet | 88 | 76 | -12 |

**Intuition**：mask overlay在container/occlusion场景最关键，因为raw RGB中object边界模糊。对于simple single-object placement（如stove），raw RGB已经足够。

### 6.2 w/o replace (-25.0) ← **最大影响**

| Task | VLA² | w/o replace | Δ |
|------|------|-------------|---|
| bowl-saucer | 88 | 16 | **-72** |
| moutai-rack | 72 | 16 | **-56** |
| moutai-cabinet | 88 | 42 | -46 |
| drawer-bowl | 62 | 26 | -36 |

**关键insight**：lexical replacement是bridging text OOD到model ID vocabulary的dominant lever。两个OOD token同时出现时，没有replace几乎完全fail。

### 6.3 w/o web (-11.0)

| Task | VLA² | w/o web | Δ |
|------|------|---------|---|
| moutai-rack | 72 | 24 | **-48** |
| moutai-cabinet | 88 | 36 | **-52** |

**Intuition**：web retrieval对novel-brand targets（如Moutai）decisive。但对trivially familiar scenes（如open-drawer +4, bowl-stove +6），retrieval反而inject noise。

### 6.4 Agentic Robot (RP) (-50.0) ← **全砍掉**

| Task | VLA² | RP | Δ |
|------|------|-----|---|
| drawer-bowl | 62 | 0 | -62 |
| saucer-stove | 84 | 0 | **-84** |
| bowl-saucer | 88 | 0 | **-88** |
| butter-bowl | 22 | 0 | -22 |
| moutai-cabinet | 88 | 20 | -68 |

**Critical finding**：去掉所有模块后，average SR崩到26.2%。paper分析原因不只是模块缺失，还有Agentic Robot原版的task-list prompt format引入了更多OOD（例如"put the blue white porcelain bowl in the cabinet"被分解成偏离训练分布的subgoals）。

这说明**prompt format设计本身就是OOD/ID的边界**——VLA²的"now do current subtask + full task context"的dual conditioning是关键创新。

---

## 七、Computational Efficiency分析

### Table IV: Average Computation Time (seconds)

| Module | Spatial | Goal | Object | Long | Easy | Medium | Hard | Avg |
|--------|---------|------|--------|------|------|--------|------|-----|
| Planner | 20.7 | 19.0 | 17.1 | 25.5 | 22.0 | 19.5 | 20.2 | 20.6 |
| Vision & Pre-Proc | 0.09 | 0.07 | 0.09 | 0.21 | 0.75 | 1.28 | 1.07 | 0.51 |
| Language | 0.02 | 0.02 | 0.05 | 0.04 | 0.26 | 0.58 | 0.78 | 0.25 |
| VOS | 8.91 | 8.70 | 9.02 | 12.08 | 7.95 | 9.11 | 9.19 | 9.28 |
| VLA | 72.95 | 73.10 | 79.78 | 131.35 | 69.71 | 82.76 | 99.02 | 86.83 |
| Verifier | 2.86 | 3.59 | 3.61 | 5.54 | 4.49 | 4.69 | 4.87 | 4.23 |
| **Total** | 105.56 | 104.49 | 109.67 | 174.75 | 105.13 | 117.87 | 135.13 | **121.66** |

**几个critical insights**：

1. **VLA占71.4%的时间**（86.83/121.66），这是backbone execution time，不可压缩
2. **Planner每次task只调用一次**（20.6s），但用thinking mode很慢
3. **Vision和Language的avg time极小**（0.51 + 0.25 = 0.76s），因为**"first cognition + memory reuse"**设计——第一次inference约20s（和Planner类似），后续直接从memory读取
4. **VOS是per-step执行**，平均9.28s，是Mask overlay的持续成本
5. **LIBERO-Long上VLA翻倍**（131.35s vs 72.95s），因为long task涉及两次pick-and-place

**Hard environment的额外开销**：
$$\Delta t_{\text{Hard}} = (1.07 - 0.09) + (0.78 - 0.02) = 1.74 \text{ s}$$

这是OOD cognition的额外成本，相对total 135.13s只占1.3%，**性价比极高**。

---

## 八、Information Flow深度解析（Figure 6）

paper的Figure 6展示了OOD→ID的完整transformation pipeline：

1. **Environment** 产生持续image flow
2. **Task Query** + first image进入系统（例："put the blue white porcelain bowl on the stove"）
3. **Planner输出**：subtask list + objects/locations extraction
4. **Vision Pre-processing**：MM-GroundingDINO尝试detect已知/未知
5. **Cognition阶段**：
   - 已知object（如stove）：直接生成bbox + mask
   - 未知object（如blue white porcelain bowl）：
     - 触发web image download
     - GLM生成5个keywords（如"round"、"white"、"blue pattern"、"porcelain"、"bowl-shaped"）
     - Cache到vision memory
     - 用keywords重新detect
6. **SAM2.1-L**：bbox → pixel-accurate mask
7. **Color Coding**：objects用红色系palette，locations用蓝色系palette
8. **Language Module**：
   - "blue white porcelain bowl" → "black bowl"（KnownList replacement）
   - 存入text memory的replace map
9. **Final Representation**：
   - Visual：mask overlay的image stream
   - Text：`"1) pick up the red-mask black bowl; 2) place the red-mask black bowl on the blue-mask stove"`
10. **Downstream VLA**接受这个ID representation执行

**关键insight**：整个cognition只在task开始时跑一次，之后只有VOS + VLA + Verifier per-step执行。这种**"one-time cognition + continuous execution"**的architecture是efficiency的关键。

---

## 九、Prompt Engineering深度剖析

### 9.1 Planner Prompt

Planner prompt非常细致，几个关键约束：
- 每个subtask只能有一个action verb
- Objects必须在`/()/`comment中explicitly标注
- 不允许`locate`、`move to`、`lift`等implicit actions
- Qualifying modifiers保留在subtask文本中，但`/()/`中只放largest specific item

例如：
```
1. pick up the yellow and white mug next to the cookie box /(yellow and white mug)/
2. place the yellow and white mug next to the cookie box on the plate /(yellow and white mug, plate)/
```

注意`/()/`中没有"next to the cookie box"这种position description——只有color和shape。

### 9.2 Verifier Prompt

Verifier prompt根据verb动态构造，四种情况：
- `pick up`：检查grasp + lift
- `place`：检查placed at location + gripper away
- `turn on/off`：检查device power state
- `open/close`：检查container state

### 9.3 GLM Understanding (Vision) Prompt

要求生成**5个keywords**，且强制"Think in ten sentences"。Keywords要求：
- 非常specific和descriptive
- 反映basic attributes
- 能让另一个VLM找到相同或similar subjects

输出格式严格JSON：`["female", "light-skinned", "doctor", "middle-aged", "smiling"]`

### 9.4 GLM Understanding (Text) Prompt

这个prompt是动态构造的，根据available evidence分三种case：

**Case A** (no comimage, no keywords, has boxes/scores)：提供evidence crop
**Case B** (nothing available)：只提供raw image
**Case C** (all available)：composite image + top crop + keywords都作为separate user turns

还有可选的web snippet（通过`fetch_snippets`函数从Wikipedia等获取）。

**Strict Constraints**：
- Output必须是Allowed vocabulary中的一个label，或者NONE
- 不允许任何analysis/explanation
- 格式严格：`<answer>LABEL_OR_NONE</answer>`

---

## 十、与其他工作的深度对比

### 10.1 VLA² vs RT-2

| 维度 | RT-2 | VLA² |
|------|------|------|
| Approach | Joint training with web data | Agentic wrapper around VLA |
| Concept update | 需要retrain | 通过web retrieval + memory instant |
| Resource cost | 极高 | 低（只fine-tune几个小模块） |
| OOD generalization | 通过pretraining knowledge | 通过explicit OOD→ID conversion |

### 10.2 VLA² vs Agentic Robot

VLA²相比[Agentic Robot](https://arxiv.org/abs/2501.13535)的核心区别：
1. **Web retrieval module**：Agentic Robot没有
2. **Mask + Color coding**：Agentic Robot用raw RGB
3. **Lexical replacement**：Agentic Robot没有KnownList substitution
4. **Prompt format**：VLA²用"now + whole"的dual conditioning
5. **Verifier recovery**：VLA²有dynamic threshold + lift gripper mechanism

Ablation Table III的Agentic Robot (RP)行显示，这些差异加起来有50.0%的SR gap。

### 10.3 VLA² vs MemoryVLA

[MemoryVLA](https://memoryvla.github.io/)也做perceptual-cognitive memory，但更侧重long-horizon task的memory，VLA²更侧重OOD concept的instant learning。

---

## 十一、Limitations与Future Directions

Paper明确提到几个limitation：

1. **Procedural rigidity**：当前framework是相对rigid的procedural structure，缺乏system autonomy
2. **No real-world experiments**：只在LIBERO simulation上验证
3. **Perception bottleneck**：224×224 resolution + imprecise object names限制fine-grained recognition

**潜在的extension方向**：

1. **Dynamic tool invocation**：让agent自己决定何时invoke web search、何时直接execute
2. **Real-world deployment**：在真机上验证，处理sim-to-real gap
3. **More tools**：加入tactile sensing、audio feedback等
4. **Larger KnownList**：用更大的pretrained vocabulary
5. **Hierarchical planning**：current planner是flat的，可以扩展为hierarchical
6. **Multi-agent collaboration**：不同VLA专长不同task

---

## 十二、对Andrej的思考

从你的视角看，这篇paper有几个特别值得关注的点：

1. **System2-like thinking**：VLA²本质上是给VLA加了System 2的slow thinking——遇到OOD先cognition、retrieval、reasoning，再execute。这和你常说的"System 1 vs System 2"框架吻合。

2. **Knowledge editing without retraining**：通过agentic framework实现concept的instant learning，绕过了retraining的cost。这是foundation model时代的重要pattern。

3. **Memory as first-class citizen**：Vision memory和Text memory的design让framework有state，可以跨task复用。这和transformer的in-context learning是互补的。

4. **Prompt engineering的精细度**：Planner、Verifier、GLM Understanding的prompt都极度structured，这种工程化是当前agentic system的核心竞争力。

5. **Ablation的insight**：replace > mask > web的importance ranking很有启发性——lexical alignment比visual mask更critical，这和LLM的token-level understanding优势呼应。

6. **OOD as a benchmark design**：三个难度递增的environment设计得很精巧，特别是Hard level引入文化specific concept（Moutai），这种OOD test design值得学习。

参考资源：
- [VLA² Project Page](https://vla-2.github.io)
- [OpenVLA GitHub](https://github.com/openvla/openvla)
- [LIBERO Benchmark](https://libero-project.github.io/)
- [SAM 2](https://ai.meta.com/sam2/)
- [Cutie: Video Object Segmentation](https://github.com/hkchengrex/Cutie)
- [MM-GroundingDINO](https://github.com/open-mmlab/mmdetection)
- [GLM-4.1V](https://github.com/THUDM/GLM-4)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [π₀: Vision-Language-Action Flow Model](https://arxiv.org/abs/2410.24164)
- [Agentic Robot Paper](https://arxiv.org/abs/2501.13535)
- [RT-2: Vision-Language-Action Models](https://robotics-transformer2.github.io/)
- [Bulk Bing Image Downloader](https://github.com/ostrolucky/Bulk-Bing-Image-downloader)

希望这个深度解析能帮你build起对VLA²的完整intuition！
