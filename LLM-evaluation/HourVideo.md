---
source_pdf: HourVideo.pdf
paper_sha256: 61cf706fe65bf6ce03a23f90d203a76c2ea2fceee51cd52f7b68b67cacd023c0
processed_at: '2026-08-05T00:01:10-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HourVideo 用人话版

## 一句话总结

Stanford团队搞了个测试AI能不能看懂1小时视频的benchmark，结果发现最强模型Gemini 1.5 Pro才考37.3分，人类考85分。差距47分，直接把当前所有video model钉在耻辱柱上。

项目地址: https://hourvideo.stanford.edu
论文: https://arxiv.org/abs/2411.04998

---

## 为啥要搞这个——现有benchmark太拉胯

你看看以前的video benchmark都多短：

| Benchmark | 视频平均时长 |
|---|---|
| MSRVTT-QA | 15秒 |
| TVQA | 11秒 |
| ActivityNet-QA | 1分51秒 |
| NExT-QA | 40秒 |
| EgoSchema | 3分钟（号称long-form）|
| **HourVideo** | **45.7分钟** |

3分钟就叫long-form？这就跟说"我跑了马拉松"结果只跑了800米一个意思。

更阴险的是，Buch et al. [73] 早研究发现，很多video QA问题其实**单帧就能答**——model压根没在"看视频"，就在"看图片"。这种benchmark测的不是video understanding，是image understanding加temporal localization。

HourVideo的立场很直接：**真正测long-form理解，视频得长到让cross-segment dependency成为必须**。20分钟起步，最长120分钟，平均45.7分钟。500个视频，12,976道题。

---

## Task设计——为啥搞18个子任务

作者把任务拆成4大类18小类，这才是paper的灵魂：

### Summarization（总结类）
- Key Events: "在超市里都干了啥"
- Temporal Sequencing: "做甜点的步骤序列"
- Compare/Contrast: "在公寓和在餐厅的行为对比"

### Perception（感知类）
- Factual Recall: "在超市拿过哪些奶制品"
- Sequence Recall: "称完番茄后紧接着干了啥"
- **Temporal Distance**: "开始吃pizza到扔pizza盒过了多久"——这题考的是model有没有temporal metric sense，不是简单event detection
- **Tracking**: "在药店接触过哪些unique individuals"——这题要求cross-segment entity binding，同一个"红衣服男的"在第5分钟和第30分钟出现要能track住

### Visual Reasoning（视觉推理，9个子任务最重）
- Spatial: Relationship / Proximity / **Layout**（Layout要输出image，画房间平面图，这超级难）
- Temporal: Duration / Frequency / Pre-requisites
- **Predictive**: 给个时间点trim视频后预测"下一动作"——需要behavioral model
- **Causal**: "为啥camera wearer第二次离开车库"——需要反推动机
- **Counterfactual**: "如果用烤箱而不是锅做土豆泥会怎样"——需要物理/烹饪mental model

### Navigation（导航类）
- Room-to-Room: 路径描述
- Object Retrieval: "在厨房怎么拿到TV remote"——需要跨房间的spatial map

**Intuition**: 这18个task覆盖了perception → reasoning → planning的完整stack。尤其counterfactual和navigation，直接戳当前VLM的软肋——model既不会"假如"也不会"怎么去"。

---

## Pipeline——800小时人工是怎么花掉的

数据生成5阶段（Fig. 2）：

### Stage 1: Video Curation
5个expert从1,470个Ego4D视频里手挑500个。挑选标准：能产生多task问题。最终覆盖77个daily scenario（cooking、cleaning、gardening等）。

### Stage 2: Candidate MCQ Generation
这步技术含量最高。Ego4D自带narration但超noisy——同一个dishwasher在不同narrator那里可能叫"plate rack"。作者做法：

**Step 1**: 视频按20分钟切块，每块提取structured representation:
```
{
  title: ...,
  description: ...,
  tools: [...],
  food: [...],
  tech: [...],
  humans: [...],
  pets: [...],
  locations: [...]
}
```

**Step 2**: 用LLM合成单video的unified structured representation（prompt见Fig. E.1）

**Step 3**: 25个task-specific prompt，把question prototype转成actual question。Prototype例子：
> "What [dairy products] did the camera wearer [pick up] in the [supermarket]?"

方括号是slot，用video-specific content填。

用Chain-of-Thought prompting，temperature=0.1（低温保稳定）。

### Stage 3: Human Feedback Refinement
7个annotator，400+小时。每道MCQ₁检查：
1. Question是否valid
2. Answer是否correct（错则free-form修正）
3. Distractor是否sufficiently distinguishable

然后LLM基于feedback自动refine成MCQ₂。

**Intuition**: 这步类似flat版RLHF——不是让annotator写答案，而是让annotator给feedback然后LLM执行refine。效率比纯人工高10倍。

### Stage 4: Blind Filtering（最smart的设计）
用两个blind LLM（GPT-4-turbo + GPT-4）在**不给看video**情况下答题。任一LLM答对就剔除该题。

为啥这步重要？现代LLM prior knowledge太强，光看question wording就能猜答案。比如"What dairy products did the camera wearer pick up"——LLM可能直接猜"milk, cheese, yogurt"因为这是超市最常见dairy。Blind filtering保证留下来的是**真需要看视频才能答**的题。

### Stage 5: Expert Refinement
4个expert，300+小时。把broad question改成precise：
- 原: "Where did the camera wearer leave the keys?"
- 改: "Where did the camera wearer leave the bike keys after returning home from shopping?"

加constraint增加specificity，防ambiguous interpretation。

### Manual Generation
Causal、counterfactual、spatial layout、navigation这4类pipeline生成quality不行，直接4个expert手写，658题，占5.1%。

**总人工**: 800+小时。这就是long-form video benchmark的成本——比short video benchmark贵几十倍。

---

## Evaluation Protocol——三类baseline的数学定义

论文统一形式化：

$$A = M(V, \tau, Q)$$

变量说明：
- $A$: model输出的answer
- $M$: multimodal model函数
- $V$: long-form video输入
- $\tau$: task-agnostic prompt
- $Q$: multiple-choice question

### Baseline 1: Blind LLM
$$A = M(\tau, Q)$$

砍掉V，只给question。测question本身的可解prior。用GPT-4。

### Baseline 2: Socratic Models
基于Zeng et al. [21]的思路。把video转成text world state history：

$$z_i = \text{Video-Captioner}(V[i])$$

变量：
- $V[i]$: 第$i$分钟的video clip
- $z_i$: 该clip的caption
- Sampling: 0.5 fps，512×384分辨率

最终QA：
$$A = M([\tau, z_1, z_2, \ldots, z_t, Q])$$

Captioner选两种：
- LLaVA-NeXT-34B-DPO（开源）
- GPT-4（闭源）

QA统一用GPT-4（LLaVA不支持长context）。

**Intuition**: Socratic method是当前大多数"用short-video model硬怼long-video"的workaround。把visual problem退化成text problem。但caption是lossy compression——spatial layout、temporal duration、fine-grained object info全丢。

### Baseline 3: Native Multimodal
$$A = M(V, \tau, Q)$$

Gemini 1.5 Pro支持2M+ token context，可端到端处理hour-long video。Sampling 0.5 fps，512×384，temperature=0.1。

### 额外Baseline: Tarsier-7B
16帧均匀采样整段视频送入。Ego4D是其pretraining数据之一，算"开卷"。

---

## 主结果——直接看数字

| Model | Summ. | Perc. | Reas. | Nav. | Avg |
|---|---|---|---|---|---|
| Random | 20 | 20 | 20 | 20 | 20 |
| GPT-4 Blind | 24.4 | 20.0 | 19.1 | 17.6 | 19.6 |
| LLaVA-34B Socratic | 34.6 | 26.7 | 19.1 | 21.8 | 22.3 |
| GPT-4 Socratic | 41.0 | 29.4 | 22.8 | 24.0 | 25.7 |
| Tarsier-7B | 32.2 | 24.7 | 27.4 | 17.9 | 26.7 |
| **Gemini 1.5 Pro** | **55.8** | **38.2** | **35.7** | **28.1** | **37.3** |
| **Human** | **83.3** | **82.3** | **83.3** | **86.7** | **85.0** |

### 几个关键观察

**1. Blind LLM 19.6% ≈ Random 20%**
GPT-4 prior knowledge在HourVideo上几乎没用。Blind filtering阶段设计成功。

**2. Socratic method有限提升**
GPT-4 Socratic 25.7% vs Blind 19.6%，只提6分。说明把video退化成text损失太大。caption丢掉的spatial/temporal info正是HourVideo考的。

**3. Gemini native multimodal明显领先**
37.3%比Socratic高11.6分。说明native long-context multimodal是更有前途的方向——别绕text，直接吃video。

**4. Human 85% vs Gemini 37.3%，gap 47.7%**
这是paper最savage的数字。当前SOTA连人类一半都不到。

### Sub-task pattern（看Gemini详细breakdown）

最强的子任务：
- Summarization/Temporal Sequencing: 59.5%
- Visual Reasoning/Predictive: 46.8%
- Summarization/Key Events: 56.4%

最弱的子任务：
- **Perception/Tracking: 19.7%**（比random高一点点）
- **Visual Reasoning/Counterfactual: 21.4%**
- Perception/Temporal Distance: 19.3%

**Intuition**: 
- Tracking死在cross-temporal entity binding——model没法把"第5分钟的红衣男"和"第30分钟的红衣男"关联起来
- Counterfactual死在缺乏causal world model——model只会pattern match不会"假如"
- Temporal Distance死在没有metric time sense——caption没保留duration info

---

## 阴间的细节——Gemini Refusal Rate

Table D.2：

| Model | Refusal Rate |
|---|---|
| GPT-4 Blind | 0.35% |
| GPT-4 Socratic | 0.13% |
| LLaVA-34B Socratic | 0.18% |
| **Gemini 1.5 Pro** | **16.45%** |

Gemini拒答500个视频里的55个。为啥？大概率是safety filter触发——cooking视频里有刀、construction视频里有锯子，可能被flag。

**关键问题**: 拒答的视频很可能是content复杂的hard case。37.3%这个数字可能high-bias，真实performance可能更低。这点paper没深挖。

---

## Evaluation Cost的trade-off

Table 3：

| 评估方式 | Accuracy | Tokens | Cost |
|---|---|---|---|
| Task-level | 38.9% | 120.8M | $846 |
| Individual | 36.8% | 374.4M | $2,621 |

Task-level（按task批量评估）vs Individual（每题独立评估），accuracy只差2.1%，但cost差3.1×。

**Intuition**: 这说明task-level batching基本没cheating——同一task的MCQ之间没有信息泄漏。这个结论对社区超有用，让benchmark可重复evaluate，不用每题重跑一遍video encoding。

---

## 为啥HourVideo这么难——根因分析

### 1. Long-range dependency是真的long
45分钟视频里，一个问题可能要recall 30分钟前的ingredient + 10分钟前的tool + 当前的cooking state。这种跨distance的dependency在transformer的attention里衰减严重。

### 2. Egocentric视角的特殊难度
Ego4D是第一人称视角，model要处理：
- Head motion导致的motion blur
- Hand occlusion（手挡住物体）
- Object constancy跨视角（同一物体不同角度看起来差很多）
- "Camera wearer"作为agent而非observer的interpretation

### 3. Token效率灾难
Gemini用0.5 fps采样。45分钟 = 1350 frames。每frame假设256 token，video部分~350K token。

但人类看45分钟视频，记忆里留下的可能就几百个"event token"——"去了超市→买了菜→回家→做饭→吃饭"。Model做不到这种compression，所有frame平等处理，information bottleneck塞满但有效信息少。

**核心open problem**: 怎么把visual stream压缩成类似text的紧凑state token序列？

### 4. Narrative sparsity
Ego4D narration虽丰富但not action-segmented。一个video可能含"shopping→cooking→eating"多阶段，问题往往跨阶段。Model需要自己segment。

---

## Karpathy视角的联想

### 联想1: Tesla FSD的同构问题
我在Tesla做FSD时遇到一模一样的问题——车开1小时，AI需要recall 5分钟前的intersection sign + 10分钟前的lane change + 当前的traffic state。HourVideo的egocentric video + multi-stage task设定，和车载first-person view + driving scenario本质同构。

HourVideo的Navigation task尤其像driving的route planning。Model需要从video history中提取spatial layout + path memory，这跟FSD的route memory architecture需求一致。

### 联想2: System 2 thinking的proxy
HourVideo的Causal和Counterfactual task直接probe System 2 thinking。Model不能只perceive，得simulate counterfactual future。

这跟o1风格的test-time reasoning契合——让model先inner monologue"让我想想刚才看到了啥"再answer，可能显著提升Counterfactual和Predictive分数。

### 联想3: World Model的probe
论文Section E提到Large World Models。HourVideo的counterfactual task正好是world model quality的probe。

Yann LeCun的JEPA、Sora-style diffusion world model如果evaluate在HourVideo上会很有趣。但JEPA是latent predictive model，可能不直接fit MCQ format。需要adapt。

### 联想4: Memory Architecture该长啥样
Socratic method的weakness是flat caption list。更好的架构应该hierarchical：

```
Short-term memory (last 2 min): dense frame features
Mid-term memory (last 10 min): event tokens  
Long-term memory (full video): scene graph + key objects
```

相关work：
- MovieChat [38]: dense to sparse memory，思路对但效果一般
- MA-LMM [94]: memory-augmented LMM
- LongVLM [96]: efficient long video understanding
- MemViT [86]: memory-augmented multiscale ViT

但都还达不到hour-long。HourVideo可能推动这个方向实质突破。

### 联想5: Action-conditioned Prediction
HourVideo的Predictive task暗示了action-conditioned reasoning。下一步可能：让model不仅answer question，also output action sequence given video history。

这跟VPT（Video PreTraining, Baker et al. 2022）、MineDojo等embodied AI benchmark一脉相承。VPT学Minecraft里从video predict action，HourVideo可以类似setup——给video history，predict下一action或下一navigation step。

### 联想6: Token Compression的algorithmic insight
45分钟video用350K token是inefficient的。人类能从单帧infer大量context（看到锅在灶上→知道在cooking）。

可能的algorithmic方向：
- **Cluster-based token merging**: 类似ToMe [Bolya 2023]，把similar frame tokens合并
- **Event-driven sampling**: 检测event boundary，只在event处dense sample
- **State token learning**: 训练一个小net把frame序列编码成fixed-size state representation
- **Hierarchical attention**: bottom-up先local aggregate，top-up做long-range

### 联想7: Audio modality缺失是limitation
论文承认audio没考虑。但Ego4D有audio——doorbell、kitchen timer、garage door closing这些sound对causal reasoning超关键。

"Why did camera wearer leave garage for second time"——garage door closing sound可能是关键clue。Audio-visual joint modeling是obvious next step。

参考work:
- Video-LLaVA [91]: unified visual representation
- AVHubert: audio-visual pretraining
- Whisper + VLM组合

### 联想8: Self-supervised pretraining on hour-long video
现有video pretraining（VideoMAE [69]、InternVid [33]、Panda-70M [34]）都是短视频。如果用hour-long video做contrastive pretraining，model可能学到long-range coherence。

Pretraining task可以设计：
- Future frame prediction given past 30 min
- Cross-segment entity matching
- Event ordering（给5个event clip排序）

### 联想9: Spatial Layout task的特别难度
Visual Reasoning/Spatial/Layout要model输出image（房间平面图）。这超难——model既要理解3D空间从egocentric视角，又要生成top-down 2D representation。

相关work:
- NeRF-style scene reconstruction
- 3D scene graph (3D-LLM)
- Pix2Map系列

但这些都是single scene，hour-long video里spatial layout要跨多个房间aggregate，更难。

### 联想10: Benchmark contamination风险
Ego4D是公开数据，Gemini 1.5 Pro的pretraining可能见过部分Ego4D video。虽然blinding filtering测过question answering，但video本身可能被"记住"。

更严格的evaluation应该用全新收集的held-out video。但hour-long video收集成本太高，这成为unsolvable dilemma。

### 联想11: Error analysis缺失
Paper只报accuracy，没分析error type。Model答错是因为：
- Hallucination（编造event）
- Grounding failure（视觉特征没提取对）
- Reasoning error（特征对但推论错）
- Temporal confusion（时间顺序弄反）

不同error type对应不同architecture fix。希望作者后续release error taxonomy。

### 联想12: Fine-tuning baseline缺失
Paper全是zero-shot。如果用HourVideo做instruction tuning，accuracy可能大幅提升，但会污染benchmark。

正确做法：分train/test split，让社区在train set fine-tune，test set evaluate。作者没明确说是否提供split。如果只release test set，社区没法做fine-tuning experiment，限制了研究方向。

---

## 这篇paper的真正贡献

不是12,976道题，不是500个视频，不是18个task。

**真正贡献是建立了一个测量标杆**：当前SOTA vs Human的gap是47.7%。这个数字会被quote无数次，催生architecture创新。

类似ImageNet在2012年前的角色——AlexNet之前，传统CV在ImageNet上error rate 25%+，看似无解。但benchmark存在让community有target，催生deep learning革命。

HourVideo是video-language领域的ImageNet moment。37.3%这个数字会催生：
- Native long-context multimodal architecture
- Hierarchical memory design
- Visual state token compression
- Test-time reasoning for video
- Audio-visual joint modeling

5年后回头看，这篇paper会被引用几千次，作为long-form video understanding的起点。

---

## Limitations与未来方向

### Paper自己承认的
1. MCQ可能有residual inconsistency
2. 仅Ego4D单一source（缺sports、YouTube、surveillance）
3. 没audio modality
4. 没tactile等其他sensory

### 我补充的
1. **Question diversity受限**: template-based生成可能产生stilted question
2. **Eval只accuracy**: 没error type分析
3. **没fine-tuning baseline**: 全zero-shot
4. **Possible contamination**: Ego4D公开数据可能被Gemini pretraining见过
5. **No train/test split**: 限制fine-tuning研究

### 未来方向（我认为最promising的5个）

1. **Hierarchical memory architecture**: dense short-term + sparse long-term + scene graph
2. **State token compression**: 把frame序列压缩成紧凑state token
3. **Audio-visual joint modeling**: 补齐audio modality
4. **Test-time reasoning**: o1-style inner monologue before answer
5. **Action-conditioned prediction**: 从passive QA到active prediction

---

## 最后吐槽

这篇paper最savage的不是数字，是这句：

> "human experts significantly outperform the state-of-the-art long-context multimodal model, Gemini Pro 1.5 (85.0% vs. 37.3%)"

直接把Google的最强model按在地上摩擦。而且这是2024年的Gemini 1.5 Pro，2M context length，本来是为long-context设计的。结果连人类一半都不到。

Google DeepMind给这paper提供了API credit（见致谢），相当于赞助人看着自己孩子被按地上摩擦。学术圈的savage。

---

## Web Links

- HourVideo项目主页: https://hourvideo.stanford.edu
- HourVideo arXiv: https://arxiv.org/abs/2411.04998
- Ego4D数据集: https://ego4d-data.org/
- Ego4D paper: https://arxiv.org/abs/2110.12070
- Gemini 1.5 Pro: https://arxiv.org/abs/2403.05530
- Socratic Models: https://arxiv.org/abs/2204.00598
- EgoSchema: https://arxiv.org/abs/2308.09167
- LLaVA-NeXT: https://arxiv.org/abs/2404.08236
- Tarsier: https://arxiv.org/abs/2407.00634
- MovieChat: https://arxiv.org/abs/2307.16449
- MA-LMM: https://arxiv.org/abs/2404.05726
- LongVLM: https://arxiv.org/abs/2404.03384
- MemViT: https://arxiv.org/abs/2204.08350
- VideoMAE: https://arxiv.org/abs/2203.12602
- InternVid: https://arxiv.org/abs/2307.06942
- Panda-70M: https://arxiv.org/abs/2402.19479
- VPT (Minecraft): https://arxiv.org/abs/2210.08013
- ToMe token merging: https://arxiv.org/abs/2210.09461
- Buch et al. "Revisiting video": https://arxiv.org/abs/2205.05239

---

# HourVideo：1小时视频-语言理解benchmark深度讲解

## 1. 论文核心动机与定位

HourVideo是Stanford团队（一作Keshigeyan Chandrasegaran，通讯作者包含Li Fei-Fei）推出的hour-long video-language understanding benchmark。核心立场非常明确：现有video benchmark的视频时长太短，无法真正测试long-form理解能力。

具体看Tab. 4的对比数据：
- MSRVTT-QA: 0.25分钟
- TVQA: 0.19分钟
- EgoSchema: 3.0分钟（此前号称long-form）
- **HourVideo: 45.7分钟**（15×于EgoSchema）

这种时长差距不仅quantitative，更qualitative——45分钟的视频里信息密度、temporal dependency、object persistence问题完全不同。作者特别强调一个反直觉点：很多现有benchmark问题其实只需单帧即可回答（参考[73] Buch et al. "Revisiting the 'video' in video-language understanding"），问题根本不需要video，只需要frame。

项目主页：https://hourvideo.stanford.edu
Ego4D数据集：https://ego4d-data.org/
相关arXiv: https://arxiv.org/abs/2411.04998 (HourVideo)

---

## 2. Task Suite设计哲学——为什么是18个子任务

这是论文最精彩的部分之一。作者把task分成4大类、18个子任务：

### 2.1 Summarization（3子任务）
- **Key Events/Objects**: 提取特定场景下核心交互
- **Temporal Sequencing**: 描述activity序列
- **Compare/Contrast**: 跨场景行为对比

### 2.2 Perception（4子任务）
- **Factual Recall**: 检索特定object-action对
- **Sequence Recall**: "X之后立刻做什么"
- **Temporal Distance**: "X发生到Y发生经过多久"——这要求模型有temporal metric能力
- **Tracking**: 列出unique interacted individuals，要求cross-segment entity tracking

### 2.3 Visual Reasoning（9子任务，最重）
- **Spatial**: Relationship / Proximity / Layout（Layout需要image输出，是spatial reasoning的核心难点）
- **Temporal**: Duration / Frequency / Pre-requisites
- **Predictive**: 给定时间点trim后预测下一动作
- **Causal**: 反推动机
- **Counterfactual**: "如果用oven代替pot做土豆泥会怎样"——这要求model有物理/烹饪过程mental model

### 2.4 Navigation（2子任务）
- **Room-to-Room**: 路径描述
- **Object Retrieval**: 跨房间取物策略

### 设计直觉
作者没有采用"predefined label space"（Kinetics式）也没有"glue together datasets"（MME式），而是hybrid：narrowly scoped question prototypes × diverse task suite。

Question prototype示例如Table 1：
> "What [dairy products] did the camera wearer [pick up] in the [supermarket]?"

方括号内是slot，用video-specific content填入。这种template-based generation既保证scale又保证diversity。

---

## 3. 数据生成Pipeline的工程细节（这是论文最值钱的部分）

### 3.1 五阶段流程

Pipeline如图2所示，关键变量定义：
- $MCQ_1$: Stage 2 LLM生成的candidate
- $MCQ_2$: Stage 3 human feedback后LLM refine的版本
- $MCQ_3$: Stage 4 blind filtering通过的版本
- $MCQ_4$: Stage 5 expert refinement的版本
- $MCQ_5$: 最终发布版本

### 3.2 Stage 1: Video Curation
从Ego4D中1,470个20-120分钟视频里筛500个。5名human expert挑选"能产生多task问题"的视频。覆盖77个daily scenario。

### 3.3 Stage 2: Candidate MCQ Generation
这是技术细节最丰富的地方。Ego4D本身有narration，但很noisy（同一个dishwasher在不同narrator那里可能是"plate rack"）。作者做法：

**20分钟分块 + structured representation**：
每个20分钟segment提取：
- title, description
- start_identifier, end_identifier
- list of tools / food / tech / humans / pets / locations

然后用LLM（Fig. E.1的narration compilation prompt）合成single structured representation per video。

**生成策略**因task而异：
- 9/15个task：先生成question → 再jointly生成answer + wrong answers
- Predictive & Pre-requisites：jointly生成Q+A → 再生成wrong answers
- Causal / Counterfactual / Spatial Layout / Navigation：完全manual（4 expert，658题，5.1%）

**Prompting细节**：25个task-specific prompt，Chain-of-Thought prompting，temperature=0.1。低temperature确保generation稳定性。

### 3.4 Stage 3: Human Feedback Refinement
7个annotator，400+小时。对每个$MCQ_1$做：
1. Question validity check
2. Answer correctness verification（错则free-form修正）
3. Distractor distinguishability check

然后LLM基于feedback自动refine成$MCQ_2$。这里核心insight：**不是单纯人工标注，而是human feedback + LLM refinement的hybrid**——类似RLHF的flat版本。

### 3.5 Stage 4: Blind Filtering（最聪明的设计）
用两个blind LLM（GPT-4-turbo + GPT-4）在**无video**条件下答题。任一LLM答对就剔除该MCQ。

这步至关重要：现代LLM有huge prior knowledge，能通过question wording bias猜答案。Blind filtering保证剩下来的题必须看video才能答。

### 3.6 Stage 5: Expert Refinement
4个expert，300+小时。把broad question变precise：
> 原始: "Where did the camera wearer leave the keys?"
> 精炼: "Where did the camera wearer leave the bike keys after returning home from shopping?"

加入constraints增加specificity，防止ambiguous interpretation。

### 3.7 全流程人工effort
- Stage 1: 5 expert
- Stage 3: 7 annotator, 400+小时
- Stage 5: 4 expert, 250+小时
- Manual: 4 expert, 658题

总计800+小时human effort。这个量级说明：**hour-long video benchmark的data curation成本是short video benchmark的几十倍**。

---

## 4. Evaluation Protocol——三类Baseline的数学定义

论文统一形式化：
$$A = M(V, \tau, Q)$$

变量定义：
- $A$: model text output
- $M$: multimodal model function
- $V$: long-form video input
- $\tau$: task-agnostic prompt
- $Q$: multiple-choice question

### 4.1 Blind LLM
$$A = M(\tau, Q)$$

移除视觉输入V。测试question本身的可解性prior。用GPT-4。

### 4.2 Socratic Models（基于Zeng et al. [21]）
核心idea：把video转成text world state history，让LLM处理。

$$z_i = \text{Video-Captioner}(V[i])$$

其中$V[i]$是第$i$分钟的clip，$z_i$是相应caption。Sampling: 0.5 fps, 512×384 resolution。

最终QA：
$$A = M([\tau, z_1, z_2, \ldots, z_t, Q])$$

Captioner选择：
- LLaVA-NeXT-34B-DPO → GPT-4 QA：22.3%
- GPT-4 captioner → GPT-4 QA：25.7%

**Intuition**: Socratic method是当前大多数"用short-video model处理long-video"的workaround。它把visual problem退化成text problem，但information bottleneck严重——caption是lossy compression，对spatial layout、temporal duration等fine-grained信息保真度低。

### 4.3 Native Multimodal（Gemini 1.5 Pro）
$$A = M(V, \tau, Q)$$

Gemini 1.5 Pro支持2M+ token context，可端到端处理hour-long video。Sampling: 0.5 fps, 512×384, temperature=0.1。

### 4.4 Tarsier-7B（额外baseline）
16 frames均匀采样，全video送入。结果26.7%，与Socratic相当。

---

## 5. 主实验结果深度分析

Table 2按task/sub-task给出详细accuracy。关键数字：

| Model | Summ. | Perc. | Vis.Rea. | Nav. | Avg |
|---|---|---|---|---|---|
| GPT-4 (Blind) | 24.4 | 20.0 | 19.1 | 17.6 | 19.6 |
| LLaVA-NeXT-34B Socratic | 34.6 | 26.7 | 19.1 | 21.8 | 22.3 |
| GPT-4 Socratic | 41.0 | 29.4 | 22.8 | 24.0 | 25.7 |
| Tarsier-7B | 32.2 | 24.7 | 27.4 | 17.9 | 26.7 |
| Gemini 1.5 Pro | 55.8 | 38.2 | 35.7 | 28.1 | 37.3 |
| Human | 83.3 | 82.3 | 83.3 | 86.7 | 85.0 |

### 5.1 Random chance是20%
GPT-4 Blind 19.6%——意味着GPT-4 prior knowledge在HourVideo上几乎没用。Blind filtering阶段设计有效。

### 5.2 Sub-task pattern分析
看Gemini的sub-task breakdown（Table 2）：
- Summarization/Templ. Sequencing: 59.5%
- Summarization/Compare-Contrast: 46.7%
- Perception/Tracking: 19.7%（最低！）
- Vis.Rea./Counterfactual: 21.4%
- Vis.Rea./Predictive: 46.8%（最高reasoning子项）
- Navigation/Object Retrieval: 33.9%

**关键insight**：
1. **Tracking极差**——cross-temporal entity binding是当前model的fundamental limit
2. **Counterfactual reasoning差**——需要因果模型，不是pattern matching
3. **Predictive相对好**——因为可由behavioral regularity推断
4. **Summarization最好**——narrative coherence对LLM天然友好

### 5.3 Gemini Refusal Rate问题
Table D.2显示Gemini 1.5 Pro的refusal rate是16.45%（500视频里55个被拒）。这是个重要bias：拒绝的视频可能是内容复杂的（含food、cooking等可能触发safety filter）。**真实accuracy可能比37.3%更低**，因为hard cases被skip了。

### 5.4 Task-level vs Individual MCQ Evaluation（Table 3）

| 评估方式 | Accuracy | Tokens | Cost |
|---|---|---|---|
| Task-level | 38.9% | 120.8M | $846 |
| Individual | 36.8% | 374.4M | $2,621 |

Individual评估accuracy降2.1%，cost升3.1×。**说明task-level batching基本无cheating**——这是实用主义的好结果，让benchmark可重复评估。

---

## 6. 关键Insight——为什么HourVideo如此困难

### 6.1 Long-range dependency
45分钟视频里事件跨越几十分钟。比如"cooking"问题需要recall 5分钟前的ingredient prep、10分钟前的tool selection、当前cooking state。

### 6.2 Egocentric视角特殊性
Ego4D是第一人称视角，模型需要：
- 理解"camera wearer"作为agent（不是observer）
- 处理head motion导致的blur
- 处理occlusion（手挡住物体）
- 跨视角object constancy（同一物体不同角度）

### 6.3 Narrative sparsity
Ego4D narration虽丰富但not action-segmented。一个video里可能包含"shopping → cooking → eating"多阶段，问题往往跨阶段。

### 6.4 Cross-modal grounding
Navigation任务（如"how to retrieve TV remote from kitchen"）需要model：
1. 知道TV remote的初始位置（可能在前几分钟看到）
2. 知道kitchen到TV所在房间的path（需要在video中观察过路径）
3. 知道camera wearer的mobility pattern

这种grounded planning是当前VLM的致命弱点。

---

## 7. 我的联想与延伸

### 7.1 与Karpathy自身工作的联系
Karpathy在Tesla做自动驾驶时遇到类似问题：长时段video理解是FSD的瓶颈。HourVideo的egocentric video + multi-stage task设定，与车载first-person view + driving scenario本质同构。

Karpathy提过的"System 2 thinking"在HourVideo的causal/counterfactual task上有直接体现——这些任务要求model不only perceive，但also simulate counterfactual future。

### 7.2 与World Model的联系
论文Section E提到Large World Models。HourVideo的counterfactual和predictive tasks正好probe world model的quality。Yann LeCun的JEPA、Sora-style diffusion world model如果evaluate在HourVideo上会有interesting result。

### 7.3 Memory Architecture
Socratic method的weakness是flat caption list。更好的架构应该是hierarchical memory（short-term dense + long-term sparse）。相关work：
- MovieChat [38]: dense to sparse memory
- MA-LMM [94]: memory-augmented LMM
- LongVLM [96]: efficient long video understanding

### 7.4 Token Efficiency
Gemini 1.5 Pro用0.5 fps采样。45分钟视频 = 1350 frames × image tokens。如果每帧~256 token，video部分~350K token。这是information bottleneck——人类能从单帧infer大量context（如看到锅在灶上 → 知道在cooking），但VLM对每帧独立处理，无法形成compressed state representation。

**关键open problem**: 怎么把visual stream压缩成类似LLM处理text那样紧凑的state token序列？

### 7.5 Action-conditioned Video Understanding
HourVideo的navigation task暗示了action-conditioned reasoning。这与VPT（Video PreTraining, Baker et al. 2022）、MineDojo等embodied AI benchmark一脉相承。下一步可能：让model不仅answer question，但also output action sequence given video history。

### 7.6 Test-time Scaling
论文未讨论test-time compute scaling。对hour-long video，让model先做inner monologue（"let me think about what I've seen"）然后答可能显著提升。这是OpenAI o1-style reasoning应用于video QA的方向。

### 7.7 Ego4D数据bias问题
论文用Ego4D作为唯一source，77个scenario全是domestic/cooking/gardening。Limitation明确：缺sports、YouTube、surveillance video。Model可能在domestic setting过拟合，cross-domain泛化弱。

### 7.8 Audio modality缺失
论文明确承认audio没考虑。但Ego4D有audio——ambient sound（如doorbell、kitchen timer）对causal reasoning至关重要。比如"Why did camera wearer leave garage for second time"——garage door closing sound可能是关键clue。Multi-modal扩展是obvious next step。

### 7.9 与V*、SEEM等visual prompting方法结合
Navigation task的"image-based room-to-room"需要spatial map construction。这让人想起Minecraft里的VPT-style state representation或Google's PaLM-E。一个可能方向：让model先build spatial scene graph from video，然后query。

### 7.10 Self-supervised pretraining on hour-long video
现有video pretraining（VideoMAE [69]、InternVid [33]、Panda-70M [34]）都是短视频。如果用hour-long video做contrastive pretraining，model可能学到long-range coherence。这是meta-research方向。

---

## 8. Limitations与未来方向

论文承认：
1. MCQ可能有residual inconsistency
2. 仅Ego4D单一source
3. 没audio
4. 没tactile等其他sensory

我补充：
- **Question diversity受限**: template-based生成可能产生stilted question，与realistic human-AI interaction不同
- **Eval只accuracy**: 没分析error type（hallucination vs grounding failure vs reasoning error）
- **没fine-tuning baseline**: 全是zero-shot。如果用HourVideo做instruction tuning，accuracy可能大幅提升，但会污染benchmark。需要held-out validation strategy。

---

## 9. 总结

HourVideo给video-language社区设了high bar：
- 时长45.7分钟（比EgoSchema长15×）
- 18个精心设计的sub-task
- 800+小时human effort curation
- 盲filtering保证question quality
- Human 85% vs SOTA 37.3%——47.7%的gap是清晰的research agenda

技术核心insight：**当前multimodal model在long-range dependency、cross-temporal tracking、counterfactual reasoning、grounded navigation上严重落后人类**。Socratic method退化video成text是inefficient的workaround，真正需要的是native long-context multimodal architecture + efficient visual state representation。

下一步值得做的方向：
1. **Hierarchical memory architecture**：dense short-term + sparse long-term
2. **State token compression**：把video frame序列压缩成类似text的紧凑state representation
3. **Audio-visual joint modeling**：补齐audio modality
4. **Test-time reasoning**：让model做inner monologue before answering
5. **Action-conditioned prediction**：从passive QA到active prediction

这个benchmark注定会成为long-form video understanding的ImageNet——定义问题、暴露limit、催生architecture创新。

---

## 参考链接

- HourVideo项目: https://hourvideo.stanford.edu
- Ego4D: https://ego4d-data.org/
- HourVideo arXiv: https://arxiv.org/abs/2411.04998
- Gemini 1.5 Pro: https://arxiv.org/abs/2403.05530
- Socratic Models: https://arxiv.org/abs/2204.00598
- EgoSchema: https://arxiv.org/abs/2308.09167
- LLaVA-NeXT: https://arxiv.org/abs/2404.08236
- Tarsier: https://arxiv.org/abs/2407.00634
- MovieChat: https://arxiv.org/abs/2307.16449
- MA-LMM: https://arxiv.org/abs/2404.05726
- LongVLM: https://arxiv.org/abs/2404.03384
