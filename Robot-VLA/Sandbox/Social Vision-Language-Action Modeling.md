---
source_pdf: Social Vision-Language-Action Modeling.pdf
paper_sha256: 8d23f832a48ee25a5eac684bb786fd0e74810558ce1999b16d47a2db8e0d1909
processed_at: '2026-08-12T08:19:01-07:00'
target_folder: Robot-VLA/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 SOLAMI

## 这 paper 到底想干啥

想象你戴上 VR 头显，面前站着 Batman 或者 Samantha，你能跟他说话，还能挥手、比手势、甚至模仿他的动作，他也能用 speech + body language 回应你。这就是 SOLAMI 想做的事——一个真正 immersive 的 3D character 对话系统。

听起来简单，做起来超难。难点在哪呢。

Reference: https://solami-ai.github.io/

---

## 之前的人怎么做的，为什么不行

之前 SenseTime 自己搞过一个叫 DLP (Digital Life Project) 的 system，思路是 LLM-Agent 那一套——把整个 system 拆成几个 sub-module 串起来：

user speech → ASR → text → LLM → text response → TTS → character speech
user motion → motion captioning → text → LLM → text instruction → text-to-motion → character motion

听起来 reasonable，但实际跑起来两个致命问题：

**问题一：信息丢了。** 你跟一个 character 聊天，你挥手的幅度、节奏、跟 speech 的配合，这些 subtle 的东西一旦压成 text caption 就没了。"他挥了挥手" 这句 text 描述，跟真实 motion 之间差了十万八千里。模型最后生成的 character motion 也只是 text caption 翻译回来的，生硬得很。

**问题二：太慢了。** 这套 pipeline 一跑下来 5 秒 latency 起步，你说完一句话等 5 秒 character 才回应，这对话没法进行下去。正常 human conversation 的 gap 是 200-500ms，你 5 秒就毁了。

Reference: DLP https://digital-life-project.com/

---

## 关键 insight：avatar 其实就是个 robot

作者抓到了一个很妙的类比。机器人领域前几年也走过同样的弯路：一开始 SayCan、Voyager 这些 LLM-Agent 把 planning 和 manipulation 串成 pipeline，后来发现 low-level manipulation 这种 task，end-to-end VLA 模型 (RT-1, RT-2, OpenVLA) 完爆 modular pipeline。原因一模一样——pipeline 丢信息、慢、不自然。

作者就说，digital avatar 本质就是个 virtual embodied 的 humanoid robot 嘛，它也要 perceive 环境、理解 user 意图、输出 coordinated 的 motor action。那干嘛不直接抄机器人的 end-to-end VLA 思路？把 speech 和 motion 都当作 LLM 的 "new languages"，全部 token 化，一个 decoder-only LLM 直接 autoregressive predict 输出。

这就是 SOLAMI 的核心 paradigm shift：从 modular LLM-Agent 变成 end-to-end VLA。

Reference:
- RT-2: https://robotics-transformer2.github.io/
- OpenVLA: https://openvla.github.io/

---

## 怎么把 motion 和 speech 塞进 LLM

**Motion 这边**：用 SMPL-X 的 joint rotation 表示 (不用 3D keypoints，因为 keypoints 还要拟合回 SMPL-X 才能驱动 Unity 里的 character，那一步又慢又有 artifact)。然后训练三个独立的 VQ-VAE，分别量化 body、hand、和两个 character 之间的 relative transform (相对位置和朝向)。为什么分开？因为 body 和 hand 的 motion 分布差很多，混在一起 codebook 学不好。Supplementary 里 ablation 证明了 separate 比 bind 在 PA-MPJPE 上一好就是 30mm。

**Speech 这边**：用 SpeechTokenizer，基于 RVQ-VAE，一秒 speech 编码成 8 层共 400 tokens，但只取第一层 semantic token (50 tokens/s) 送进 LLM。其他 7 层 acoustic token 留给 SoundStorm 做 voice cloning——character 想换声音只要换 4-6 秒的 voice prompt 就行，不用 retrain。

这样 user 的 speech 和 motion 都变成 token sequence，character 的 speech 和 motion 也是 token sequence，整个对话就是普通的 LLM multi-turn conversation：

```
System: <character description>
User: <motion tokens><speech tokens>
Character: <motion tokens><speech tokens>
...
```

LLM autoregressive 生成 character 的 motion tokens 先，再生成 speech tokens。这个顺序有讲究——先定 body language 骨架再填 speech 内容，保证两者 coordinated。DLP 那种 modular pipeline 里 motion 和 speech 是两个独立模块各跑各的，经常对不上号。

---

## 数据从哪来：最大的难题

End-to-end 模型最饿的就是数据。motion-text pair 有 HumanML3D、Inter-X 这些 dataset，speech-text pair 有 Common Voice 这些，但是 user 和 3D character 之间 multimodal 多轮 interaction 的 dataset？没有。

作者想了三条路：
1. 从 internet video 抓——video mocap 质量不行，occlusion 严重
2. 搭 VR 平台直接录——Quest 的 lower body tracking 是 estimated 的不准，而且真人 motion 和 animated character motion 分布有 gap
3. 用现有 motion dataset 合成——最可行

最后走第三条，搞了个 4 步 synthesis pipeline：

**Step 1**: 收集 5.3K 对话 topic，来源是 Google Trends、Zhihu、Jike 社区、加上 GPT-4o brainstorm。不同 character 的 topic 分布不同，Batman 聊 Gotham 的犯罪，Samantha 聊情感关系。

**Step 2**: GPT-4o 基于 topic 和 character setting，一轮一轮生成下一轮 dialogue 的 motion description + speech text。Supplementary 里对比了三种 generation 策略——round-by-round completion、两个 LLM 扮演两个角色对话、one-shot 整体生成后逐轮 refine。最后选前两种交替用，one-shot 那种多轮 refine 后质量反而下降。

**Step 3**: 用 text embedding 从 46K motion database (HumanML3D + Inter-X + DLP-MoCap) 里 retrieve 最匹配的 motion clip。Inter-X 原本只有 two-person 整体描述，用 GPT-4o 拆成 single-person motion-text pair。

**Step 4**: 用 LLM refine speech text 让它跟 retrieved motion 对齐。这一步关键——DLP 那种 modular 方法 speech 和 motion 各生成各的经常错位，这里强制对齐避免这个问题。

最后用 XTTS v2 / Azure TTS 做 voice cloning，保证 character 声音风格一致。最终 6.3K multimodal conversation items。

这个 dataset 还设计了 4 种 task type：
- `common`: 日常对话
- `motion understanding`: 用户做有语义的 motion，character 要理解并表达出来
- `instruction following`: 用户给 motion 指令，character 执行
- `imitation`: character 模仿用户 motion

这让 SOLAMI 不只是闲聊 bot，还能显式理解 body language、follow motion instruction。这是 LLM+Speech 类方法根本做不到的。

---

## 三阶段训练为什么必要

直接在 multimodal dialogue data 上 fine-tune 效果差 (ablation 显示 FID 5.052 vs full pipeline 3.443)。原因很 intuitive：motion data 太少，模型还没学会 motion 和 text 怎么对应，就被逼着学复杂的多轮对话生成，自然学不好。

所以搞三阶段：

**Stage 1**: 训 tokenizer。Motion VQ-VAE 用 reconstruction + embedding + commitment + velocity loss 联合训练，velocity loss 保证 motion temporal smoothness。Speech tokenizer 直接用 AnyGPT pre-trained checkpoint 冻起来。

**Stage 2**: modality alignment pre-training。用 46K motion-text pair 做 text-to-motion 和 motion captioning，用 410K speech-text pair 做 TTS 和 ASR，让模型先学会 motion/text 和 speech/text 的 grounded 对应关系。Motion 和 speech data 按 4:6 采样平衡数据量。

**Stage 3**: instruction tuning。用 5.7K multimodal conversation items 做 supervised fine-tuning，只 supervise character 的 response，不 supervise user input (不然模型会学会生成 user 行为)。这里有个坑：LoRA (rank 8) 效果很差，FID 15.729，远差于 full fine-tune 3.443。原因是 pre-training task 和 instruction tuning task 的 distribution shift 太大，LoRA 的 low-rank delta 容量不够。这个发现对未来 VLA 训练很重要——不能盲目套 NLP 的 LoRA 经验。

Reference:
- AnyGPT: https://arxiv.org/abs/2402.12226
- LoRA: https://arxiv.org/abs/2106.09685

---

## VR Interface 工程

Frontend: Oculus Quest 3
Backend: 2×H800 GPU

数据流：
1. Quest full-body tracking 拿 user pose → retarget 到 SMPL-X
2. Mic 录 speech → SpeechTokenizer
3. Backend LLM 推理生成 character motion + speech tokens
4. SoundStorm decode speech + voice cloning
5. UniTalker audio-to-face 生成 face blendshape
6. Body + face 参数 retarget 到 3D character (Unity Engine)

Character waiting 状态用 preset idle motion 保证视觉自然。3D character 资产覆盖 AI assistant、电影角色、internet meme、真人 celebrity，用 VRoid Studio 建模 + Unity 做 rigging、skinning、retargeting、texture。

这套工程最大的意义是让 user study 成为可能——60 个真实用户戴 VR 跟 character 对话 5 轮以上再填问卷，这比单纯的 quantitative metric 有说服力得多。

Reference:
- UniTalker: https://arxiv.org/abs/2408.09548
- QuestSim tracking: https://quest-sim.github.io/

---

## 实验结果最有意思的点

Table 1 的 quantitative 结果：

**Motion quality**: SOLAMI FID 3.443，DLP 4.254，w/o pretrain 5.052。End-to-end VLA + pre-training 在 motion distribution quality 上明显胜出。

**Latency**: SOLAMI 2.639s，DLP 5.518s。End-to-end 比 modular 快一倍多，因为省了 sub-module 串联。注意这里 SOLAMI 还有 motion 生成，DLP 用 MotionGPT 替代了原版 MoMat-MoGen (太慢 >5s)，已经是优化版 DLP 了。

**Character Consistency**: SOLAMI 3.824，反而比 LLM+Speech 3.859 略低。这有点反直觉——加 motion 模态后 character 一致性下降了。作者解释是 motion + speech training 冲击了原 LLM 内嵌的 character knowledge，类似 PaLM-E 和 VILA 里 modality conflict 现象。这其实是个 open problem——怎么在新 modality training 中保持 LLM 原有的 world knowledge。

**LoRA 完全失败**: FID 15.729，比 w/o pretrain 还差。这个结果对未来 VLA 训练很有启示意义——多模态 instruction tuning 的 distribution shift 比纯 NLP task 大得多，LoRA 不够。

User study 60 人 4 个维度 SOLAMI 全胜。但有个有意思的细节：DLP 的 speech consistency 比 LLM+Speech 还低，说明 modular pipeline 里 motion 和 speech 异步生成反而破坏了 speech 质量。但 DLP overall experience 仍高于 LLM+Speech，证明 **body language 的存在本身就大幅提升 user experience，即使 speech 质量降了**。这跟 psychology 研究里 immersion → presence → satisfaction 的链路完全一致。

---

## 我觉得这 paper 最 deep 的几个 insight

**Insight 1: Social interaction 也是 low-level motor task**。我们直觉觉得对话是 high-level reasoning task，但仔细想，natural conversation 里 timing、节奏、gesture-speech coordination、表情微调，这些都是 sub-second 的低 latency motor control，跟机器人 pick-and-place 本质类似。所以 end-to-end VLA 比 modular LLM-Agent 更适合。这个 paradigm shift 可能让 social AI 整个领域重新思考 architecture。

**Insight 2: Text 是 bottleneck modality**。NLP 一直把 text 当 lingua franca，但 human social signal 里 text 能承载的只是冰山一角。把 motion、speech、未来可能还有 facial expression、gaze、proxemics 都 token 化统一到 LLM 里，才是真正 capture human social signal 的正确路径。

**Insight 3: Synthetic data 是 pragmatic 的 bridge**。Real dyadic interaction data 极其昂贵且难收集，作者用现有 motion database + LLM 合成 6.3K conversation 已经能 beat DLP。但 synthetic data 的天花板也很明显——motion vocabulary 被限制在 database 里，character 签名动作 (比如 Batman 的 specific combat pose) 数据稀缺。未来真正的 breakthrough 可能要等 real data collection infrastructure 成熟，类似 robotics 领域 Open X-Embodiment 那种规模的数据集。

**Insight 4: Cross-embodiment 是 unsolved problem**。SMPL-X 统一表示所有 character，握手、object manipulation 这种 fine-grained task 就崩了。这跟机器人 cross-embodiment 是同一个问题——不同 body 的 motor 空间 mapping 怎么 generalize。Robotics 那边的 RT-2、π0、GR-2 在这个问题上有探索，3D character 这边几乎还没人做，是片 open territory。

Reference:
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- PaLM-E: https://palm-e.github.io/
- VILA: https://arxiv.org/abs/2312.07533

---

## 跟相关 work 的关系

SOLAMI 其实是几个 trend 的 convergence:

- AnyGPT (discrete token unify 多模态) → SOLAMI 继承 speech tokenizer
- MotionGPT (motion as foreign language) → SOLAMI 继承 motion VQ-VAE
- RT-2 / OpenVLA (end-to-end VLA) → SOLAMI 借鉴 paradigm
- DLP (modular 3D character) → SOLAMI 的主要 baseline 和 motivation
- Character-LLM (role-play LLM) → SOLAMI 加入 motion + speech grounding
- EgoLM / Ego4D (egocentric multimodal) → 未来多人交互方向

更广的 intuition：未来 social AI agent 一定是 end-to-end multimodal VLA，不是 modular LLM-Agent。Modular framework 在 planning/reasoning 这种 high-level task 够用，但凡涉及低 latency、subtle signal、fine-grained motor control 的 task，end-to-end 都更优。Robotics 已经证明了，digital character 现在也在证明，下一波可能是 embodied conversational agent (Siri/Assistant 加身体) 和 robot companion。

---

## Paper 没解决但我觉得重要的

1. **Full-duplex streaming**: 现在还是 turn-based，user 说完 character 才回应。真人对话是 overlap 的，你还没说完对方已经开始 respond (点头、嗯哼、partial gesture)。GLM-4-Voice、Body of Her 在探索这个，SOLAMI 还没碰。

2. **Long-term memory**: 长对话 forgetting 问题，SOLAMI 没显式处理。DLP 的 SocioMind、Talker-Reasoner 那种分层 memory 架构可能要整合进来。

3. **Multi-person interaction**: 现在只支持 1v1。多人场景下 proxemics、gaze allocation、turn-taking 全是新问题，input modality 也要扩展到 video 或 dynamic 3D scene。

4. **Emotion 和 personality modeling**: 现在 character setting 是 text description，emotion 是 implicit 的。如果有 explicit emotion state tracking + personality parameter，character 表现会更 consistent。

5. **Skill learning from few examples**: Character 的 signature action 数据少，怎么从 foundation model 已有的 knowledge 中 extract 或用 few-shot RLHF 学新 skill 是个 open direction。

Reference:
- GLM-4-Voice: https://github.com/THUDM/GLM-4-Voice
- Body of Her: https://arxiv.org/abs/2408.02879
- Talker-Reasoner: https://arxiv.org/abs/2410.08328

---

## 一句话总结

SOLAMI 把 "3D autonomous character" 这个问题从 modular LLM-Agent paradigm 重构成 end-to-end Social VLA paradigm，借鉴了 robotics VLA 的成功经验，用 discrete token 统一 motion 和 speech，用 synthetic data pipeline 解决数据稀缺，用 VR interface 证明 end-to-end 在 motion quality、latency、user experience 上全胜 modular pipeline。这是个 paradigm-setting 的工作，后续如果 data 规模和 backbone scale 跟上 LLM scaling law，可能会催生真正的 "Avatar GPT"——你戴上 VR 就能跟任何 character 面对面聊天的那个未来。

---

# SOLAMI: Social Vision-Language-Action Modeling 深度讲解

## 1. Problem Motivation 与核心 Insight

这篇 paper 来自 SenseTime Research 与 NTU S-Lab，目标非常 ambitious: 让用户通过 VR 设备与 3D autonomous characters 进行沉浸式 face-to-face 对话——不仅仅是 text 或 voice，而是一个能同时处理 speech + body motion + facial expression 的 end-to-end VLA model。

核心观察 (这也是 build intuition 的关键)：**digital avatars 本质上就是 robots with virtual humanoid embodiment**。机器人领域的经验告诉我们，LLM-Agent framework 在 planning 这类 high-level task 上表现好 (SayCan, Voyager)，但 low-level manipulation 上 end-to-end VLA 模型 (RT-1, RT-2, OpenVLA) 明显更优。把这个 insight 迁移到 social interaction 领域：previous work (Digital Life Project, DLP [17]) 用 text 作为 intermediate 把 motion captioning、text-to-motion、ASR、TTS 这些 sub-module 串接起来，存在两个致命问题：
1. **Information loss**：text 无法 capture motion 中的细微 nuance
2. **Latency**：模块串联的 pipeline latency 破坏 natural communication 的时效性

Paper 的 contribution 三件套：Social VLA Architecture + SynMSI Dataset + VR Interface。

Reference: 
- Digital Life Project: https://arxiv.org/abs/2402.07719
- RT-2: https://robotics-transformer2.github.io/
- OpenVLA: https://openvla.github.io/

---

## 2. Architecture 深度拆解

### 2.1 整体 Design philosophy

SOLAMI 把 speech 和 motion 都当作 LLM vocabulary 中的 "new languages"，用 discrete tokens 统一表达。这与 AnyGPT [81]、MotionGPT [38]、SpeechGPT [82] 的思想一脉相承——把 modality 离散化到 token space，让 LLM backbone 自然 next-token predict。

为什么是 decoder-only 而非 encoder-decoder? 因为多轮 social dialogue 本质就是 autoregressive sequence modeling，单 backbone 的 in-context learning 能力更适合处理 conversation history 与 character setting 共存的情况。

Reference: 
- AnyGPT: https://arxiv.org/abs/2402.12226
- MotionGPT: https://motion-gpt.github.io/

### 2.2 Motion Tokenizer (核心创新点之一)

**Motion representation 选择**：作者采用 SMPL-X [56] 的 joint rotations 而非 3D keypoints，原因有二：
1. SMPL-X rotation 直接兼容 Unity Engine 等工业 animation pipeline
2. 避免 keypoints → SMPL-X fitting 的 latency (SMPLify 单帧约 1s) 与 fitting artifacts

公式 (1) 是标准 VQ-VAE quantization:

$$\hat{m}_t^u = Q^u(\mathbf{m}_t^u) = \arg\min_{z_i \in \mathbb{Z}_u} \|\mathbf{m}_t^u - z_i\|_2$$

变量解释:
- $\mathbf{m}_t^u$: 第 $t$ 个时刻、motion part $u$ 的连续 motion feature
- $u \in \{b, h, t\}$: $b$=body, $h$=hand, $t$=inter-character relative transform (rotation + translation 两个 person 之间的相对位姿)
- $\mathbb{Z}_u$: motion part $u$ 对应的 codebook (一组可学习 codewords)
- $z_i$: codebook 中的第 $i$ 个 codeword
- $\hat{m}_t^u$: quantized 后的 discrete token

这里有个 **critical design choice**：body 和 hand 用 **separate VQ-VAE** ($Q^b$, $Q^h$) 而不是 bind 在一起。Supplementary Table 3-4 的 ablation 给出了原因：

| ID | Body & Hand | Repre | Backbone | Token Interleaved | FID↓ | Diversity↑ | PA-MPJPE↓ | Pred Valid↑ |
|---|---|---|---|---|---|---|---|---|
| 1 | bind | joints | GPT-2 | - | 1.48 | 9.03 | 148.00 | 0.836 |
| 4 | separate | rotation | GPT-2 | No | 2.72 | 14.05 | 112.53 | 0.638 |
| 5 | separate | rotation | Llama2 | No | 1.82 | 10.40 | 110.23 | 0.999 |

Separate representation 在 PA-MPJPE 上从 148 降到 112.53 ( keypoints-based 反而 87-80，但 visual quality 差，因为 fitting artifacts)。Separate 让 Pred Valid 从 0.836 掉到 0.638，但 Llama2 backbone 把它救回 0.999——这说明 strong LLM 可以更好地理解 separate sequence 的结构化输入。

Token 形式上：body 和 hand 各产出 $L_M$ 个 sequential tokens，用 1D conv 沿 temporal 维度下采样；transform token $\hat{m}^t$ 用 MLP 整体产出单个 token。

### 2.3 Speech Tokenizer

采用 SpeechTokenizer [84]，基于 RVQ-VAE [80] 结构，关键贡献是 disentangle 了 semantic 和 acoustic 信息。一秒 speech 编码到 8 层、共 400 tokens，SOLAMI 只用 first semantic layer (50 tokens/s) 送入 LLM，acoustic layers 在 SoundStorm [13] decoding 时通过 4-6 秒 character voice prompt 做 zero-shot voice cloning。

为什么不用所有 layer？因为：
1. Acoustic tokens 信息冗余，对 LLM 推理负担大
2. 用 short voice prompt 做 instance voice cloning 比 LLM 直接 generate acoustic tokens 更灵活，character switching 时不需要 re-train

### 2.4 Multi-modal Multi-round Interaction Template

整个 interaction 被建模成普通 LLM 对话格式：

```
System Prompt: <Character Placeholder>
User: <M Placeholder><S Placeholder>
Character: <M Placeholder><S Placeholder>
User: <M Placeholder><S Placeholder>
Output:
Character: <M Placeholder><S Placeholder>
```

这里 `<M Placeholder>` 是 motion token sequence，`<S Placeholder>` 是 speech token sequence。Special tokens 标记 modality 起止位置 (类似 LLaVA 的 `<image>` token 或 MotionGPT 的 `<motion>` special tokens)。

---

## 3. Training 三阶段策略

### Stage 1: Tokenizer Training

Motion VQ-VAE 训练 loss (公式 2)：

$$\mathcal{L}_m = \lambda_r \mathcal{L}_r + \lambda_e \mathcal{L}_e + \lambda_c \mathcal{L}_c + \lambda_v \mathcal{L}_v$$

- $\mathcal{L}_r$: reconstruction loss (decoder 输出 vs original motion)
- $\mathcal{L}_e$: embedding loss (codebook 更新，EMA)
- $\mathcal{L}_c$: commitment loss (encoder 输出 commit 到 codeword)
- $\mathcal{L}_v$: velocity loss (motion 的 temporal derivative 重建，保证 motion smoothness)
- $\lambda_*$: 手动调权的 weights

Speech tokenizer 直接用 AnyGPT [81] pre-trained checkpoint，全部冻结。

### Stage 2: Multi-task Pre-training for Modality Alignment

这是 **关键 stage**——直接在 multimodal interaction data 上 train 效果差 (见 ablation Tab. 1 中 SOLAMI w/o pretrain: FID 5.052 vs full pipeline 3.443)。原因：motion data 太少，模型没学会 motion 和 text 的 grounded alignment，就被迫学复杂的 dialogue 生成。

训练 task 清单：
- Motion-Text alignment: 46K motion-text pairs (text-to-motion + motion captioning) + 11K interactive motion pairs (two-person motion generation)
- Speech-Text alignment: 410K speech-text pairs (TTS + ASR) + 100K speech dialogue pairs (speech-to-speech)

Motion 和 speech data 按 4:6 比例 sampling，平衡模态数据量。

### Stage 3: Instruction Tuning for Multi-turn Conversation

公式 (3) 是 instruction tuning 的核心 loss：

$$\mathcal{L}_{IT} = -\sum_{r=1}^{R} \sum_{i=1}^{L_M^r} \log p_\Theta(\hat{m}_i^r | \hat{m}_{i-1}^r, ..., \hat{m}_1^r, \hat{S}_{<r}, \hat{M}_{<r})$$
$$\quad\quad\quad\quad\quad - \sum_{r=1}^{R} \sum_{i=1}^{L_S^r} \log p_\Theta(\hat{s}_i^r | \hat{s}_{i-1}^r, ..., \hat{s}_1^r, \hat{S}_{<r}, \hat{M}_{\le r})$$

变量解释:
- $\Theta$: LLM backbone parameters (或 LoRA weights)
- $R$: 一个 conversation 中的 round 数
- $\hat{S}_r$, $\hat{M}_r$: 第 $r$ round 的 speech 和 motion token sequence
- $L_M^r$, $L_S^r$: 第 $r$ round motion / speech 序列长度
- $\hat{S}_{<r}$, $\hat{M}_{<r}$: 前 $r-1$ round 所有 speech 和 motion tokens (历史 context)
- $\hat{M}_{\le r}$: 包含当前 round motion 的 prefix (motion 先于 speech 生成)

**Critical design**: 只 supervise character 的 response，不 supervise user input。这跟 instruction tuning 的标准做法一致——避免模型学会生成 user 的行为 (那不是它的 task)。

Motion 先于 speech 生成，这个顺序假设很关键：character 的 body language 和 speech 是 coordinated 的，先确定 motion 骨架再 generate speech 可以让 speech 内容跟 motion 语义对齐。Supplementary 里也提到 reverse 这个顺序会影响 coherence。

LoRA 实验 (rank 8, alpha 16) 效果不好 (Tab. 1: FID 15.729, 远差于 full params 3.443)，作者归因于 pre-training task distribution 和 instruction tuning task distribution 之间 gap 太大，LoRA 的 low-rank update 容量不够 absorb 这种 distribution shift。这与 VILA [46] 的观察类似。

---

## 4. SynMSI Dataset Synthesis Pipeline

数据是这种 multimodal social interaction 任务最稀缺的资源。Table 5 (supplementary) 对比了三种数据来源：

| Method | Cost | Quality | Limitations |
|---|---|---|---|
| Internet Videos | Low | Medium | Video mocap 难处理 occlusion, long-tail; 视角转换问题 |
| VR Platform | High | High | 设备 tracking 不准 (lower body 是 estimated); 真人 motion 与 animated character 分布 gap |
| Existing Datasets + Synthesis | Low | Medium-High | 需要解决 dialogue diversity、speech-motion alignment |

SOLAMI 选第三条路。Pipeline 4 步骤：

**Step 1 - Topic Collection (5.3K topics)**:
- Character-specific topics: GPT-4o brainstorm
- News topics: Google Trends
- Daily life topics: Jike community
- Curiosity topics: Quora, Zhihu

Supplementary Fig. 7 展示了 Samantha、K-VRC、Batman、Banaya 四个 character 的 topic word cloud，character-specific topic distribution 很不同，体现了 character setting 的多样性。

**Step 2 - Script Generation (Round-by-Round)**:
用 GPT-4o 基于 topic、character setting、前一轮 dialogue 生成下一轮的 (motion description, speech text, expression)。

Supplementary 提到对比了三种 script generation 方法：
- Method 1: Round-by-Round completion (主 paper 采用)
- Method 2: Character Agent Dialogue (DLP 的 SocioMind 思路，两个 LLM 各扮一个角色)
- Method 3: One-shot generation 整体生成后逐轮 refine

实验发现 Method 1 和 2 效果好，Method 3 多轮 refine 后质量下降。最终 SynMSI 随机交替使用 Method 1 和 2。

**Step 3 - Motion Retrieval**:
用 text embedding [50] (OpenAI text-embedding) 计算生成 motion description 与 motion database 语义相似度，retrieve 最匹配 motion clip。

Motion database: HumanML3D [29] (24K) + Inter-X [76] (20K motion-text + 10K two-person) + DLP-MoCap [17] (2K)，共 46K motion clips，全部 GPT-4o 合成详细 text annotation。

Inter-X 的特殊处理：用 GPT-4o 把 two-person action description 拆分成 single-person motion-text pair，保证每个 person 的 motion 都有 description。

**Step 4 - Speech-Motion Refinement**:
LLM 生成 speech text 与 retrieved motion 对齐 refine——这一步关键，避免 LLM-Agent 方法中 speech 与 motion misalignment 的问题 (DLP 用 text 中介，speech 和 motion 在不同模块独立生成，常常对不上)。

最终用 XTTS v2 [19] / Azure TTS 做 voice cloning，character 间声音风格保持一致。

**Task Type 设计** (Supplementary):
- `common`: 日常对话
- `motion understanding`: 用户 motion 有强语义，character 需理解并表达
- `instruction following`: 用户给出明确 motion 指令
- `imitation`: character 模仿用户 motion

这种 task 设计让 SOLAMI 不仅是闲聊 agent，还能显式理解 body language 和 follow motion instruction——这是 LLM+Speech 方法根本做不到的。

最终 SynMSI: 6.3K multimodal conversation items，9:1 train/test split，5.7K 用于 instruction tuning。

Reference:
- HumanML3D: https://github.com/EricGuo5513/HumanML3D
- Inter-X: https://inter-x.github.io/
- XTTS v2: https://arxiv.org/abs/2406.04904

---

## 5. VR Interface 工程

Frontend: Oculus Quest 3
Backend: 2× H800 GPU

数据采集链路:
1. Quest full-body tracking system [73] → pose parameters → SMPL-X retargeting
2. Microphone → speech → SpeechTokenizer
3. Backend LLM 推理 → motion + speech tokens
4. SoundStorm decode speech + voice cloning
5. UniTalker [25] audio-to-face 生成 face blendshape
6. Body + face 参数 joint retargeting → 3D character 驱动

LLM+Speech 或 character 等待状态时用 preset idle motion 保证视觉自然。

3D character 资产：AI assistant avatar、电影角色、internet memes、真人 celebrity。VRoid Studio [5] 制作 + Unity Engine facial rigging、skinning、bone chain simulation、retargeting、texture/material。

Reference:
- UniTalker: https://arxiv.org/abs/2408.09548
- QuestSim (Quest body tracking): https://quest-sim.github.io/

---

## 6. Experiments 深度分析

### 6.1 Baselines 设置

1. **LLM+Speech** (Llama2-7B + Whisper large-v3 ASR + XTTS v2 TTS)：纯语音对话 baseline
2. **AnyGPT (fine-tune)**：在 SynMSI speech data 上 fine-tune AnyGPT-base，验证 speech-only 端到端方案
3. **DLP (MotionGPT)** [17]：LLM-Agent modular pipeline，把 motion captioning 和 motion generation 都用 MotionGPT [38] 替代原版 MoMat-MoGen (后者 latency >5s 不能用)
4. **SOLAMI (w/o pretrain)**：验证 Stage 2 pre-training 的作用
5. **SOLAMI (LoRA)**：rank 8 alpha 16
6. **SOLAMI (full params)**：完整模型

全部用 vLLM [41] 加速，部署在 2× H800 GPU 上做 latency 评估。

### 6.2 Quantitative Results (Table 1) 解读

| Method | FID↓ | Diversity↑ | PA-MPJPE↓ | Angle Err↓ | VC Sim↑ | Context Rel↑ | Char Cons↑ | Latency↓ |
|---|---|---|---|---|---|---|---|---|
| LLM+Speech | - | - | - | - | 0.818 | 3.527 | 3.859 | 3.157 |
| AnyGPT (ft) | - | - | - | - | 0.819 | 3.502 | 3.803 | 2.588 |
| DLP (MotionGPT) | 4.254 | 8.259 | 165.053 | 0.495 | 0.812 | 3.577 | 3.785 | 5.518 |
| SOLAMI (w/o pretrain) | 5.052 | 8.558 | 159.709 | 0.387 | 0.820 | 3.541 | 3.461 | 2.657 |
| SOLAMI (LoRA) | 15.729 | 8.145 | 167.149 | 0.400 | 0.770 | 3.251 | 3.423 | 2.710 |
| **SOLAMI (full)** | **3.443** | 8.853 | **151.500** | **0.360** | **0.824** | **3.634** | 3.824 | 2.639 |

**几个有意思的观察**:

1. **FID**: SOLAMI full 3.443 vs DLP 4.254，端到端 VLA 在 motion quality distribution 上明显好。SynMSI 数据集本身 FID 9.136 (ground truth 自己——这反映 synthesis 数据本身的 distribution，不算特别高)。

2. **Diversity**: SOLAMI 8.853 最高，说明 end-to-end 不会 collapse 到 average motion，保留了 motion 的丰富性。

3. **PA-MPJPE**: SOLAMI 151.5 vs DLP 165.05——但注意这个数值绝对值很大 (151 mm)，远高于 HumanML3D 上 motion generation SOTA 的 ~40-50mm。这说明 social interaction task 的 motion 复杂度远高于 single-person generation task，ground truth 本身可能 noisy。

4. **Angle Error**: SOLAMI 0.360 < DLP 0.495，rotation-based representation + end-to-end 让 joint rotation 更准。

5. **VC Similarity**: SOLAMI 0.824 略高于 baselines，voice cloning 在 end-to-end 训练下学得更好。

6. **Context Relevance**: SOLAMI 3.634 最高——加入了 motion 输入让模型能感知 user body language，speech 内容更 relevant。

7. **Character Consistency**: SOLAMI 3.824 第二 (低于 LLM+Speech 3.859)。作者归因于 motion + speech modality 训练冲击了原 LLM 内嵌的 character knowledge。这跟 PaLM-E [24]、VILA [46] 的 modality conflict 现象类似。

8. **Latency**: SOLAMI 2.639s，仅次于 AnyGPT (ft) 2.588s，远好于 DLP 5.518s。LLM+Speech 3.157s 因为 Llama2-chat 倾向生成长回复 (paper 提到有时 >30s)，作者 truncate 到 3 句话降低 latency。

9. **LoRA 表现异常**: FID 15.729，比 w/o pretrain 还差。LoRA 容量不足以同时 align 新 modality distribution 和 instruction following。

### 6.3 User Study (Table 2, Fig. 5)

60 个参与者，每人 ≥5 轮对话，4 个维度 (1-5 Likert):

| Dimension | SOLAMI | DLP | LLM+Speech |
|---|---|---|---|
| Motion Coherence | 最高 | 中 | 最低 |
| Motion Interaction | 最高 | 中 | 最低 |
| Speech Consistency | 最高 | 最低 | 中 |
| Overall Experience | 最高 | 中 | 最低 |

关键 insight：DLP 的 speech consistency 反而比 LLM+Speech 低 (虽然 motion 相关 metric 高)。这说明 modular pipeline 中 motion 模块和 speech 模块异步工作，破坏了 speech-motion coordination。但 DLP overall 仍高于 LLM+Speech，说明 **body language 的存在本身就显著提升 user experience，即使 speech 质量下降**——这跟 psychology research 中 immersion → presence → satisfaction 的链路一致 [23, 34, 62, 66]。

### 6.4 Ablation 关键结论

1. **Pre-training 重要**: w/o pretrain FID 5.052 → full 3.443，character consistency 从 3.461 → 3.824。modality alignment 是 instruction tuning 的必要前置。

2. **Full params >> LoRA**: LoRA 在多模态 instruction tuning 下 insufficient，distribution shift 太大。

3. **Separate VQ-VAE >> Bind**: body 和 hand 分开 tokenize 在 PA-MPJPE 上明显好 (112 vs 148 在 supplementary Table 3)。

4. **Llama2 >> GPT-2 as backbone**: Llama2 的 Pred Valid 0.999 vs GPT-2 的 0.638。LLM 的 text understanding 能力直接决定 separate sequence 输入的 format compliance。

---

## 7. Future Work 与 Open Problems (Supplementary A)

作者自己列出了几个 limitation，对应未来方向：

1. **Input Modality 扩展**: dyadic interaction 用 user motion+speech 够用，但多人交互或 environment/object interaction 需要 video [24, 90] 或 dynamic 3D scene [57] 输入。

2. **Real Data Collection**: SynMSI 是 synthetic，真实 dyadic interaction data 能让模型更精确自然，且支持 full-duplex streaming conversation (类似 GLM-4-Voice [87]、Body of Her [9])。作者提到几个 promising data source:
   - Video mocap [16] 
   - 从 video 学习 human behavior [11] (Gen2Act 思路)
   - VR interaction platform [63]
   - Surrogate control [21] (Open-Television 思路)

3. **Cross Embodiment**: unified SMPL-X 表达对不同 character (尤其是非人形 character) 的 fine-grained task (握手、object manipulation) 仍有局限。机器人 cross-embodiment [90] 与 3D human retargeting 共享方法论。

4. **Long-Short Term Design**: 长时间交互下出现 forgetting、computational redundancy。Talker-Reasoner architecture [22]、DLP 的 SocioMind [17] 提供 long-term memory + short-term real-time 的分层思路。

5. **Few-shot Learning**: 人 motion 是 long-tail [85]，character signature action 数据稀缺。利用 foundation model 中已有的 character knowledge [75, 79] 或 RLHF [53] 引导 few-shot learning 是 promising 方向。

Reference:
- Body of Her: https://arxiv.org/abs/2408.02879
- GLM-4-Voice: https://github.com/THUDM/GLM-4-Voice
- Open-Television: https://open-tele-vision.github.io/
- Talker-Reasoner: https://arxiv.org/abs/2410.08328

---

## 8. 个人 Critical Analysis 与 Intuition Building

### 8.1 为什么 end-to-end VLA 在 social interaction 上更优？

直觉上理解：human 社交对话中 **60-70% 信息是非语言的** (body language、表情、tone)，把这些信号压成 text caption 会丢掉大量 nuance。DLP 类 pipeline 用 text 中介，等于强迫模型把"舞者用动作讲的故事"先翻译成文字再翻译回动作——两次损失。

End-to-end discrete token modeling 让 motion 和 speech 在同一个 latent space 中被 LLM 推理，model 可以直接学到 motion-motion 的 correlation (比如"我挥手" → "character 也挥手回应")，而无需经过 text 中介。

这与 RT-2 在 robotics 上的成功逻辑完全一致：vision 直接 → action token，跳过 symbolic 中介。

### 8.2 为什么 motion representation 选 rotation 而非 keypoints?

这是工程 vs 精度的 trade-off：
- Keypoints: motion metric (FID, PA-MPJPE) 数值好看，但需要 SMPLify fitting 才能驱动 character，fitting 1s/frame 的 latency 在 real-time interaction 不可接受
- Rotation: metric 数值稍差，但可以直接驱动 Unity Engine，无 fitting artifacts

这反映一个 deep insight：**research benchmark metric 与 real-world deployment quality 的 misalignment**。Paper 勇敢选 deployment-friendly 的 representation，即使 metric 牺牲一点。

### 8.3 三阶段训练的必然性

直觉：模态数量增加时，joint training 容易 underfit 各模态。Pre-training 阶段把每个 modality 与 text (最强 prior) 单独对齐，建立 grounded representation；instruction tuning 再用对齐好的 representation 学复杂的多模态 coordination。

这与 Flamingo、LLaVA 等 VLM 的两阶段训练思想一致：先 modality alignment 再 instruction tuning。

### 8.4 SynMSI 的 limitation 与未来改进

Synthetic data 的根本问题：retrieve 出来的 motion 不一定完全 match 生成的 speech text 描述。虽然有 Step 4 refine，但 refine 是 LLM 改写 speech text 适配 motion，无法改变 motion 本身——这意味着 character 的 motion vocabulary 被限制在 motion database 内。

未来方向可能是：
- 用 text-to-motion diffusion model (MotionDiffuse [83]) 生成 motion 而非 retrieve
- 或用 video-to-motion [16] 扩充 motion database
- 或直接收集 real dyadic interaction data (虽然昂贵)

### 8.5 LoRA 失败的启示

LoRA 在 LLM instruction tuning 上常见，但在 SOLAMI 失败 (FID 15.729 vs full 3.443)。原因：pre-training task distribution (single modality task) 和 instruction tuning task distribution (multi-turn multi-modal dialogue) 之间 shift 巨大，LoRA 的 low-rank delta 不够 absorb。

这对未来 VLA 训练有重要启示：**modality-injection 类训练可能需要 full fine-tuning 或 higher-rank LoRA**，不能直接套用 NLP 的 LoRA 经验。

---

## 9. 与相关 Work 的网络效应

SOLAMI 实际上是几个 trend 的 convergence point:

1. **AnyGPT lineage**: discrete token unify 多模态 → SOLAMI 继承 speech tokenizer
2. **MotionGPT lineage**: motion 作为 foreign language → SOLAMI 继承 motion VQ-VAE
3. **VLA in robotics** (RT-1/2, OpenVLA, GR-2 [20]): end-to-end vision→action → SOLAMI 借鉴到 social interaction
4. **DLP**: modular LLM-Agent for 3D character → SOLAMI 的主要 baseline 和 motivation
5. **Character-LLM** [64]: role-play LLM → SOLAMI 加入 motion 和 speech 的 grounding
6. **EgoLM** [31], **Ego4D/Ego-Exo4D** [27, 28]: egocentric multimodal LLM → 未来 SOLAMI 多人交互的方向

这给一个更广的 intuition：**未来 social AI agent 一定是 end-to-end multimodal VLA，而非 modular LLM-Agent**。Modular framework 在 planning/reasoning 这种 high-level task 上够用，但凡是涉及低 latency、subtle signal、fine-grained motor control 的 task，end-to-end VLA 都更优——robotics 已经证明，digital character 现在也在证明。

Reference:
- MotionDiffuse: https://motiondiffuse.github.io/
- GR-2: https://gr2-manipulation.github.io/
- Character-LLM: https://arxiv.org/abs/2310.10158
- EgoLM: https://arxiv.org/abs/2409.18127

---

## 10. 总结

SOLAMI 是 first end-to-end Social VLA framework，把"3D autonomous character" 这个问题从 modular LLM-Agent paradigm 重新定义成 end-to-end VLA paradigm。关键 contributions:

1. **Architecture**: decoder-only LLM + separate VQ-VAE for body/hand/transform + SpeechTokenizer semantic layer
2. **Data**: SynMSI 4-step synthesis pipeline 用 46K motion database + GPT-4o 生成 6.3K 多模态 conversation
3. **VR Interface**: Quest 3 + 2×H800 backend 演示端到端 immersive interaction
4. **Training**: 3-stage (tokenizer → modality alignment pre-training → instruction tuning)，full param fine-tuning 必要

实验证明 SOLAMI 在 motion quality、speech context relevance、latency 都显著优于 modular LLM-Agent (DLP) 和 speech-only end-to-end (AnyGPT fine-tune) baselines，user study 在 4 个维度全部领先。

未来方向聚焦：real data collection、cross-embodiment、long-short term memory、few-shot skill learning、扩展到 multi-person / environment interaction。

这个工作打开了一个新 paradigm——把 social intelligence 也归到 VLA framework 下处理。后续工作如果能在 data 规模和 backbone scale 上 follow LLM scaling law，可能会催生真正的 "Avatar GPT"。

Paper page: https://solami-ai.github.io/
DLP (前作): https://digital-life-project.com/
相关 VR 交互参考 Open-Television: https://open-tele-vision.github.io/
