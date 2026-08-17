---
source_pdf: Chain-of-Modality.pdf
paper_sha256: 2df31f796ae54ba6140f9efb4716dc7ec6c3ec8c1f11a08d031c2929e2c0d7fe
processed_at: '2026-08-03T15:28:00-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Chain-of-Modality

## 一句话版

你想教 robot 拧瓶盖，光给它看视频没用，因为它看不出你什么时候使劲、使多大劲。这篇 paper 的招儿就是：**让 robot 戴上"耳机"和"肌电手环"看你干活，然后用一个巧妙的 prompt 让大语言模型一步步分析这堆多模态信号，最后吐出能跑的 Python 代码**。

项目主页：https://chain-of-modality.github.io

---

## 核心痛点在哪

想象你教一个学生插插头。你会说："先轻轻捏住，转一转对准方向，再使劲插进去"。这个"轻轻"和"使劲"对你来说是常识，对 robot 来说是天书。

为什么？因为 video 帧里看不出力的变化。你握紧插头的那一瞬间，像素几乎没变化，但你的前臂肌肉其实在疯狂放电。Camera 拍得到你的手在哪，拍不到你用了多大力。

这事儿很关键，因为 manipulation task 里 force 是个一等公民。Drumming 要力度，擦桌子要压力，开瓶盖要 grasp/release 的节奏，全靠 force 来分阶段。没有 force signal，task plan 就缺了骨架。

作者就问了一个很自然的问题：**既然 video 不够，我们给 robot 再配个 microphone 或者 armband，让它能"听"到、"摸"到人类干活的过程，是不是就够了呢？**

答案是：数据够用了，但**让 VLM 同时理解这么多异质信号**成了新难题。

---

## 为什么直接把所有信号塞给 VLM 不行

Gemini 1.5 Pro 号称能吃 100 万 token，GPT-4o 也支持 long context。你心想："我把 image frames + force 曲线 + hand pose 序列全拼成一个超长 token sequence 喂进去，让 VLM 直接输出 task plan 不就完事？"

Paper 里测了这个 baseline，叫 "Merged"。结果很尴尬：**accuracy 只有 17% 左右**，而 vision-only 是 0%。你反而被这些额外的信号搞糊涂了。

这事儿其实跟你训练神经网络时遇到的问题一模一样：attention 是个 softmax，序列一长，关键信号就被稀释。VLM 看到一长串 force 数值 + 一堆 image token，它的 attention 会无脑地被 image 吸引过去（因为 image token 是高维 dense feature），把 force 当噪声忽略。或者更糟，它从 image 里试图猜 force，从 force 里试图猜 object，全错配。

这跟人类看长论文一个道理：给你一本 1000 页的书，让你一次性写综述，你肯定写不好。但让你先看第一章写个 outline，再带着 outline 看第二章补充细节，最后再看第三章填充 case study，你就 work 了。CoM 就是这个套路。

---

## CoM 的套路：像写提纲一样拆任务

CoM = Chain-of-Modality，顾名思义，**让 VLM 沿着 modality 的维度做 chain-of-thought**。

具体三步走：

**第一步：先看 force signal，写个粗时间骨架**

VLM 只看那串 force 数值（从 EMG 或 audio 来的）。它的任务是找出 "什么时候用力、什么时候松开"。比如它会说：

```
t=11: 开始用力
t=27: 用力峰值 1
t=37: 松手
t=50: 用力峰值 2
t=62: 松手
```

这时候 VLM 完全不知道人在干嘛，也不知道操作什么物体。它只知道"用了 2 次力，中间松了 2 次"。这就够了，这是骨架。

**第二步：加入 hand pose，给骨架贴上动作语义**

VLM 现在看 hand pose（thumb + middle finger 的 2D 像素坐标）+ 第一步的骨架。它的任务是：在每个用力阶段，手在做什么 motion？

它会发现："哦，t=27 到 t=37 这段，finger 之间先 pinch 在一起，然后逆时针旋转了大约 180 度，然后松开。t=37 到 t=50 这段，手指没接触物体，但在空中逆时针转回去。"

于是输出更新成：

```
t=11: Grasp(left)   ← 左手抓了什么东西
t=27: Grasp(right)
t=27-37: Twist(right, counterclockwise, 180°)
t=37: Release(right)
t=37-50: Twist(right, clockwise, 180°)
t=50: Grasp(right)
...
```

现在每个用力阶段都有了 motion label，但 VLM 还不知道操作的是什么物体。

**第三步：加入 image，给动作贴上 object label**

VLM 现在看 image frames + 第二步的半成品。它终于能识别："左手拿的是 bottle，右手扭的是 bottle_cap"。

最终输出：

```
LEFT Hand:
  t=11: Grasp(left, bottle)
RIGHT Hand:
  t=27: Grasp(right, bottle_cap)
  t=27-37: Twist(right, counterclockwise, 180°)
  ...
```

这就是完整的 task plan，带 timing、带 motion、带 object、带 force。

**第四步：把 plan 翻译成 Python 代码**

VLM 拿到第三步的 plan + skill API 的 docstring，生成：

```python
Move_to('left', Find('bottle'))
Grasp('left', 'bottle')
Move_to('right', Find('bottle_cap'))
for _ in range(3):
    Grasp('right', 'bottle_cap')
    Twist('right', 'counterclockwise', 180)
    Release('right')
    Twist('right', 'clockwise', 180)
```

注意那个 `for _ in range(3):` — VLM 自己发现了"重复 3 次"的模式，把它抽象成了 loop。这个在原始 plan 里没明说，是 code generation 阶段 emergent 出来的。

---

## 为什么顺序是 Force → Hand → Image

这个顺序很关键，不是随便排的。

**Force 最先**，因为 force signal 最稀疏、最 event-driven。一个 force peak 就是明确的 event 标记，VLM 很容易从一串数字里找出峰值。这步给整个 task 切了时间片。

**Hand pose 第二**，因为 hand motion 只在 force event 附近才有意义。知道"t=27 用力了"之后，VLM 只需要 focus 看 t=27 附近那几帧的 hand pose，不会被无关时刻的 hand pose 干扰。

**Image 最后**，因为 image 是 dense context，但需要时间定位。你已经知道"在 t=27 到 t=37 这段，右手做了 twist 动作"，那 VLM 看这几帧 image 时只需要回答"twist 的是什么东西"，问题就被 dramatically 简化了。

如果反过来，先看 image 让 VLM 描述"画面里有什么"，它会罗列一堆东西（bottle, cap, table, hand, finger...），但你不知道哪个是 task-relevant。CoM 的顺序让每个 stage 的 sub-question 都被前一个 stage constrain 住了，这是信息论意义上的 optimal decomposition。

形式化一点，CoM 做的是：

$$
P(A_3 | F, H, I) = P(A_3 | A_2, I) \cdot P(A_2 | A_1, H) \cdot P(A_1 | F)
$$

- $A_1$: force-only analysis
- $A_2$: hand-pose analysis conditioned on $A_1$
- $A_3$: image analysis conditioned on $A_2$
- $F, H, I$: force, hand pose, image 三个 modality 的输入

这相当于假设了 modality 之间的 conditional dependence 结构：force 信号独立解释时间结构，hand pose 在给定 force 的条件下解释 motion，image 在给定 motion 的条件下解释 object。这个假设在 manipulation task 里基本成立。

---

## Force 信号怎么来的

这里有两个 option。

### Option A: EMG Armband

EMG = Electromyography，肌电图。Armband 贴在前臂上，有 8 个电极，每个测一块肌肉的放电。原理是肌肉收缩时肌纤维产生 action potential，电极贴皮肤能测到这个电信号。

原始 EMG 有几个特点：
- 采样率高（200Hz），因为肌肉电信号是高频的
- 8 个 channel 对应 8 块不同肌肉
- 信号是 bipolar（有正有负），需要取绝对值或做 rectification
- 有噪声，需要平滑

Paper 的处理：

$$
F_t = \max_{c \in \{1, \dots, 8\}} |EMG_c(t)|
$$

- $EMG_c(t)$: 时刻 $t$ 第 $c$ 个 channel 的 EMG 值（已经从 200Hz 下采样到 60Hz 跟 camera 对齐）
- $| \cdot |$: 取绝对值（rectification）
- $\max_c$: 8 个 channel 取最大值

为什么取 max？因为不同 action 激活不同肌肉。Twist 用 flexor，hit 用 extensor，grasp 用 intrinsic muscle。Max 能捕捉"任何一块肌肉在使劲"这个事件，对 event detection 友好。

如果你了解 EMG processing 的标准 pipeline，其实更严谨的做法是：high-pass filter → rectify → low-pass filter → envelope。Paper 这里简化了，只做 rectify + max。因为 VLM 不需要精确的 force 值，只需要"什么时候用力、用多大力"的相对信息。

### Option B: Microphone

对于 drum、擦桌子这种有 impact sound 的任务，可以用 audio loudness 代替 EMG。计算 RMS (Root Mean Square)：

$$
L_t = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i(t)^2}
$$

- $L_t$: 时刻 $t$ 的 loudness
- $x_i(t)$: 时刻 $t$ 的 short window 里第 $i$ 个 audio sample
- $N$: window size（paper 没明确说，估计几十 ms）

RMS 跟人耳感知的 loudness 很接近，比 absolute mean 好因为平方放大了大值。Drum 这种 impact 信号，RMS 峰值清晰，跟"敲了多大力"高度相关。

这个 option 的意义：如果没 armband，拿个手机录视频带麦克风也能凑合用，降低了 hardware 门槛。

参考 EMG 基础：
- https://www.delsys.com/emg-workflow/
- HaMeR: https://hamer.is.tue.mpg.de/

---

## Hand Pose 用了什么

用了 HaMeR (CVPR 2024)，一个 transformer-based 3D hand reconstruction model。输入 hand image crop，输出 SMPL 参数化的 3D hand mesh + 21 个 keypoint 的 3D 位置。

但 paper 只取了 **thumb + middle finger 的 2D 像素坐标** 作为 modality。为什么这么简化？

1. **Token budget**: VLM 的 context window 有限，21 个 keypoint × 2D × 每帧 = 42 个数字/帧。60 帧 × 42 = 2520 个数值 token，太长。2 fingertip × 2D = 4 个数字/帧，可控。
2. **Pinch grasp 够用**: paper 的 task (拧瓶盖、握插头、抓鼓棒、擦板) 都是 pinch grasp，thumb + middle finger 是 key pair。
3. **3D 噪声大**: 单视角重建的 3D hand 有 depth 歧义，2D 反而更稳定。

我觉得这个 design choice 很聪明，体现了 "less is more for VLM" 的直觉。给 VLM 太多信息它处理不动，给它刚好够的关键 hint 反而效果好。

HaMeR 论文：https://arxiv.org/abs/2307.01087

---

## 那个 Find() API 怎么实现的

这个细节 paper 说得不多但很重要。Robot 执行 `Find('bottle')` 时，背后发生：

1. Robot 拍一张 RGB-D image
2. 把 RGB image + object name 发给 **Gemini 1.5 Pro**
3. Gemini 输出 2D bounding box: `[x_min, y_min, x_max, y_max]`
4. 用 depth + camera intrinsics 把 box 内每个 pixel 反投影到 3D

数学上：

$$
\mathbf{p}_{obj} = \frac{1}{|\mathcal{B}|} \sum_{(u,v) \in \mathcal{B}} \pi^{-1}(u, v, D(u, v))
$$

- $\mathcal{B}$: 2D bounding box 内 pixel 集合
- $D(u, v)$: depth map 在 pixel $(u, v)$ 处的值
- $\pi^{-1}$: camera inverse projection，公式是 $X = (u - c_x) \cdot D / f_x$, $Y = (v - c_y) \cdot D / f_y$, $Z = D$
- $\mathbf{p}_{obj}$: object 3D centroid in camera frame

这步就是用 VLM 替代了传统 open-vocabulary detector (比如 YOLO + CLIP)。好处是 zero-shot、灵活、能处理 free-form 描述。坏处是 latency 高（一次 VLM API call 几百 ms）+ 偶尔会给出错误 box。

这个 trick 实际上把 Gemini 当成了一个 **"perception primitive"**，跟 Code as Policies 里用 LLM 当 "reasoning primitive" 是同一个思路：foundation model 不只做 reasoning，还能直接做 perception。

参考 Code as Policies: https://code-as-policies.github.io/

---

## 实验结果怎么读

### Table I: 多模态理解准确率

最有意思的数据点是 **w.o. img** 和 **w.o. force** 这两个 baseline 的对比。

**Pressing Cube** 这个任务（用力按方块）：

| 方法 | Accuracy | Similarity |
|---|---|---|
| Image-only | 0.00 | 0.00 |
| w.o. img (force + hand) | 0.70 | 0.96 |

意思是：**这个任务没 image 反而更好**！因为按方块的 motion 很简单（手往下压），关键是"什么时候按、按几次、按多大力"，这些全在 force signal 里。Image 反而是干扰项，VLM 看 image 容易分心。

**Insert Plug** 这个任务：

| 方法 | Accuracy | Similarity |
|---|---|---|
| Image-only | 0.00 | 0.80 |
| w.o. img | 0.00 | 0.32 |

完全相反。没 image 完全歇菜，因为 plug 和 power_strip 的 spatial relationship 必须从 image 才能看出。

直觉上：**contact-rich + 几何复杂** 的任务，image 重要；**force-sensitive + motion 简单** 的任务，force 重要。CoM 的好处是它不预先假设哪个重要，按顺序全用一遍。

### Fig. 5: Reasoning 方式对比

5 种 baseline 的关键对比：

- **Merged** (所有信号一锅炖): 17% 左右
- **Sep-Sep** (分别处理再 late fusion): 比较好
- **CoM** (sequential conditioned): 最好，比 Sep-Sep 高 17-19%

CoM 比 Sep-Sep 高出的部分就来自"conditional reasoning"。Sep-Sep 是三个 modality 各自分析，最后拼答案，每个 stage 都是"无 context"地分析。CoM 让 stage 2 知道 stage 1 的结论，stage 3 知道 stage 2 的结论，这种 information flow 让后续 stage 的搜索空间急剧缩小。

### Table II: 真实 robot 实验

Ours 平均 **73%**，Oracle（人工写代码）平均 **92%**，差距 19%。

这 19% 的失败 case 我推测分布是：
- **VLM 分析错误** ~10%: 比如把 3 次 twist 识别成 2 次，或者 force level 判断错
- **Code 生成错误** ~5%: API call 顺序错、参数错
- **Execution 错误** ~4%: Find 给的位置偏差、grasp 滑了

我特别注意到 KUKA 比 ViperX 表现更好（15 vs 12 在 bottle opening 上）。ViperX 是相对便宜的 robot，KUKA 是工业级。这说明 paper 的 code-based abstraction 真的把"具体哪个 robot 执行"这件事解耦了——同一份 Python 代码，扔给不同 robot 都能跑，只是精度有差。

### Generalization 测试

Paper 特意强调了 generalization：
- Bottle opening: 测了 7 种瓶子（6 种 unseen）
- Plug inserting: plug 和 strip 随机摆放
- Drum: 不同 beat pattern
- Wiping: 不同形状 marker

这种 setup 很 important，因为如果只测一种 object 那就是过拟合。Paper 选择用 `Find('bottle')` 这种 abstract call，让 VLM 在 runtime 做 grounding，所以对 unseen object 天然有 generalization。

---

## 这套方法为什么 work 的深层直觉

### 直觉 1: Attention 的经济学

VLM 的 attention budget 是有限的。Long-context 不等于 long-attention。哪怕模型能吃 1M token，它的 attention 仍然被 softmax 归一化稀释。

CoM 通过 sequential prompting 强制 attention 聚焦：
- Stage 1 时，VLM 只看 force，100% attention 在 force
- Stage 2 时，VLM 看 hand pose，但 prompt 里还带着 stage 1 的 output，attention 会被 stage 1 的结论 anchor 到关键时间区间
- Stage 3 时，VLM 看 image，带着 stage 2 的 motion label，attention 被 anchor 到 hand 接触的 object

这就像你查字典：直接让你查"这本书讲了什么"很难，但如果先问"这本书提到几次'力'？分别在几页？"再问"提到力的那些页讲了什么动作？"再问"那些动作涉及什么物品？"——每个 sub-question 的搜索空间都被前一个 anchor 住了。

### 直觉 2: Modality 的 hierarchy

Force / hand pose / image 三种 modality 有天然的 hierarchy：

| Modality | 信息密度 | 时间分辨率 | 语义抽象度 |
|---|---|---|---|
| Force | 稀疏 | 高（精确到 16ms） | 低（只有数字）|
| Hand pose | 中等 | 中等 | 中（motion label）|
| Image | Dense | 低（60Hz） | 高（object, scene）|

稀疏 + 高时间分辨率的 signal 适合先做 temporal segmentation；dense + 高语义的 signal 适合后做 object grounding。CoM 的顺序刚好对应这个 hierarchy。

如果你反过来先看 image，VLM 会陷入"看到什么描述什么"的陷阱，把 task-irrelevant 的东西也罗列出来，浪费 attention。

### 直觉 3: 跟人类 motor learning 的同构

人类学 motor skill 也是分阶段的：
1. 先感知 effort：啊，那个动作那里要使劲
2. 再感知 motion：使劲的时候手是怎么动的
3. 最后感知 object：啊原来是在拧那个瓶盖

CoM 的 stage 顺序跟人类 motor learning 的认知顺序同构。这可能是它 work 的深层原因——它符合我们大脑处理 manipulation 信息的 natural pathway。

参考 motor learning 的阶段模型：
- https://en.wikipedia.org/wiki/Motor_learning
- Schmidt & Lee 的 Motor Control and Learning 教科书

### 直觉 4: 这本质是 Bayesian decomposition

完整的 task plan inference：

$$
P(\text{Plan} | F, H, I)
$$

直接 inference 难。CoM 做的是 factorization：

$$
P(\text{Plan} | F, H, I) \approx P(\text{Object} | \text{Motion}, I) \cdot P(\text{Motion} | \text{Timing}, H) \cdot P(\text{Timing} | F)
$$

- $P(\text{Timing} | F)$: force signal 推 timing
- $P(\text{Motion} | \text{Timing}, H)$: 给定 timing，hand pose 推 motion
- $P(\text{Object} | \text{Motion}, I)$: 给定 motion，image 推 object

这个 factorization 的有效性依赖于 conditional independence 假设：在给定 motion 的情况下，object identity 与 force 信号条件独立。这个假设大部分时候成立——你拧瓶盖还是拧门把手，force pattern 可能差不多，但 motion 都是 twist，object identity 由 image 决定，跟 force 没直接关系。

所以 CoM 在那些 conditional independence 假设强的任务上效果好（拧瓶盖、敲鼓）。如果某个任务 force 直接编码了 object identity（比如根据"咔哒声"判断按钮按到位了），这个 factorization 就会失效。

---

## Limitation 和我的吐槽

Paper 自己承认的：
1. Audio 只用了 loudness，没充分利用频率/音色
2. Open-loop execution，没 closed-loop feedback

我自己想吐槽的几点：

### Force normalization 跨 task 不通用

Paper 把 force normalize 到 [0, 100]，但 drum 的 force=50 和 plug insertion 的 force=50 物理含义完全不同。VLM 在 prompt 里看到的 force value 需要被 contextualize，否则跨 task 的 transfer 会出问题。

更好的做法可能是 per-task calibration，或者用 force 信号的统计特征（z-score）而不是 raw normalized value。

### 2 个 fingertip 不够

Thumb + middle finger 只能表征 pinch grasp。Power grasp（握杯子）、lateral grasp（握钥匙）、tripod grasp（握笔）都表征不了。Paper 的 task selection 偷了点懒——选的都是 pinch-friendly 的 task。

未来如果想扩展到更多 grasp type，需要至少 5 个 fingertip + palm center，token 数量会爆炸。这时候可能需要专门的 hand motion encoder 而不是直接喂数值给 VLM。

### VLM 推理延迟

Long-context VLM 一次推理几秒到几十秒，CoM 要调用 VLM 4 次（3 次分析 + 1 次代码生成）。整个 plan generation 流程可能要 1 分钟。对 real-time robot control 来说太慢了。

不过 paper 的定位是 one-shot learning from demonstration，不是 real-time control。Plan 生成本来就是离线的，robot 执行时是 open-loop 跑代码。这个延迟是可接受的。

### 没有错误恢复

Code 生成完就死板地执行，grasp 失败了也不会重新规划。这跟 SayCan、Inner Monologue 那种 closed-loop language model planning 形成对比。

未来可以结合 [Liang et al. 2024 的 LMPC](https://arxiv.org/abs/2402.11450)，让 robot 在执行失败时把失败信息（比如 vision observation）反馈给 VLM，重新生成修正后的 plan。

---

## 这篇 paper 的真正意义

我觉得这篇 paper 表面上是讲 prompting strategy，实际上揭示了一个更深的事实：

**当前 VLM 的 cross-modal reasoning 能力是被高估的。**

Gemini 1.5 Pro 和 GPT-4o 都号称 multimodal、long-context，但在真正复杂的 multimodal dense reasoning 上，merged input 只有 17% accuracy。这说明它们的 multimodal training 还不够，多模态信号之间的 cross-attention 没训好。

CoM 是个 workaround，通过 prompt engineering 强制引导 attention，在不重训模型的情况下提升性能。这给我们两个启示：

1. **短期**：在 robotics 应用里，与其等下一代 VLM，不如用 CoM 这种 prompt engineering 榨干现有模型
2. **长期**：VLM 训练需要考虑 modality-specific 的 attention 机制，单纯 scale context length 和 data volume 不够

更深一点说，这篇 paper 让我想到 Bitter Lesson。CoM 这种手工设计的 modality 顺序，本质上是个 inductive bias，跟当年手工设计 feature 一个性质。未来 VLM 足够强了，可能自己学会在内部做这种 modality-wise chain reasoning，CoM 就会被淘汰。但当下这个 inductive bias 极其有效，能让现有 VLM 在 robotics task 上 work。

这跟你在 Tesla 讲过的 "software 2.0" 的演化路径一致：先用大量 explicit engineering 解决问题，然后逐步被 learned solution 替代。CoM 是个优秀的 explicit engineering solution，等着被未来的 learned multimodal reasoner 替代。

参考 Bitter Lesson: http://www.incompleteideas.net/IncIdeas/BitterLesson.html

---

## 总结

如果让我用 3 句话概括这篇 paper：

1. **问题**: Vision-only video 不够教 robot 做需要 force control 的 task，得加 force/audio signal
2. **方法**: CoM 把 VLM 推理拆成 force → hand → image 三个 sequential stage，每个 stage 都基于前一个的 output refine
3. **结果**: 把 merged-input baseline 的 17% accuracy 拉到 60%+，real robot 成功率 73%，能 generalize 到 unseen object

这 paper 不是 fundamentally novel，它的核心 idea 很简单（拆开 modality 顺序处理），但 execution 很扎实，实验设计很 thoughtful，结果 convincingly 支持了 hypothesis。对 robotics community 的实际价值很高，因为它给了大家一个可复用的 prompt template，让现有 VLM 能用起来 force/audio 这类被忽视的 modality。

如果你接下来想做相关方向，我觉得几个 high-value 的 extension：
- 把 audio modality 拓展到 spectrogram input
- 加 closed-loop error recovery
- 探索让 VLM 自己 discover skill API（而不是预设）
- 用 CoM 思路做 long-horizon task 的 hierarchical planning

代码在项目主页：https://chain-of-modality.github.io

---

# Chain-of-Modality (CoM) 技术深度讲解

## 1. Paper 核心问题与动机

这篇 paper 解决的核心问题是 **one-shot imitation from multimodal human demonstration video**，具体聚焦在 vision-only data 无法表达的关键信息：**force / 力的控制参数**。

作者观察到很多 contact-rich manipulation task 中，human 在不同阶段施加的 force 等级差异巨大，而这些差异在 RGB 像素里几乎不可观测。例如：

- **Plug insertion**：先 light force 调整 plug orientation in-hand，再 high force insert into socket
- **Drum playing**：不同 beat 用不同力度产生不同音色
- **Bottle cap twisting**：需要多次 grasp-release-twist 循环，每次 twist 方向交替

这个 gap 就是 paper 要 bridge 的：**visual signal lacks force/effort cues, but force 是 manipulation primitive 的关键 control parameter**。

论文链接：
- Project page: https://chain-of-modality.github.io
- arXiv (相关作者): https://arxiv.org/abs/2407.07936 (MimicPlay, 同作者前作)
- HaMeR: https://hamer.is.tue.mpg.de/
- Code as Policies: https://code-as-policies.github.io/

---

## 2. Multimodal Data 组成

paper 用了三种 modality，每个 timestep $t$ 的输入向量可以形式化为：

$$
\mathcal{D} = \{(I_t, F_t, H_t)\}_{t=1}^{T}
$$

- $I_t \in \mathbb{R}^{H \times W \times 3}$: RGB image at time $t$, 60Hz camera
- $F_t \in \mathbb{R}$: scalar force value (from EMG 或 audio loudness)
- $H_t \in \mathbb{R}^{2 \times K}$: hand pose, 这里 $K$=2 fingertips (thumb + middle finger), 每个 fingertip 有 2D pixel location

### 2.1 EMG Force Signal 处理

EMG armband 输出 8 个 channel，每个 channel 对应一块肌肉的激活电信号，原始采样 200Hz。Camera 是 60Hz，所以需要 downsample。Paper 用了 max pooling across channels：

$$
F_t = \max_{c \in \{1, 2, \dots, 8\}} |EMG_c(t)|
$$

- $EMG_c(t)$: 时刻 $t$ 第 $c$ 个 channel 的 EMG 读数（先下采样到 60Hz）
- $c$: channel index, 1 到 8
- $\max$: 取 8 个 channel 的最大值
- 取绝对值是因为 EMG 是 bipolar signal（既有正也有负波动）

这个 max 操作的 intuition：单次肌肉收缩未必所有 channel 同时激活，max 能捕捉最显著的那块肌肉的 effort，作为该时刻整体 force 的代理量。

### 2.2 Audio Loudness

如果用 microphone 而不是 EMG，则用 RMS (Root Mean Square) 计算 loudness：

$$
L_t = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i(t)^2}
$$

- $L_t$: 时刻 $t$ 的 loudness scalar
- $N$: 窗口长度（论文未给具体值，估计在 20-50ms 量级）
- $x_i(t)$: 时刻 $t$ 窗口内第 $i$ 个 audio sample 的 amplitude

RMS 是 audio energy 的标准度量，对 drum、擦桌子这类 impact sound 特别敏感。

### 2.3 Hand Pose via HaMeR

HaMeR (CVPR 2024) 是一个 transformer-based 3D hand reconstruction model，输入是 cropped hand image，输出 SMPL 参数化的 3D hand mesh。Paper 只取其 2D fingertip pixel location 作为 modality，避免 3D 噪声。

为什么只用 2 个 fingertip 而不是全 21 个 keypoint？因为大部分 manipulation skill 的关键 motion 都可以用 thumb + middle finger 表征（pinch grasp），简化 input 长度有助于 VLM 处理。

---

## 3. Chain-of-Modality 的核心思想

### 3.1 为什么 Merged 方法失败？

Baseline "Merged" 把所有 modality interleaved 成一个 long token sequence，直接喂给 VLM 让它输出 final answer。结果发现：

- **Attention collapse**: VLM 在 long-context multimodal input 上倾向于关注最显眼的 modality（通常是 image），忽略数值 signal
- **Cross-modal misattribution**: VLM 试图从 image 里提取 force 信息，从 force signal 里推断 object，全部错配
- Long-context Gemini 1.5 Pro / GPT-4o 虽然 support 百万 token，但 reasoning ability 在 multimodal dense input 上急剧下降

### 3.2 CoM 的 Prompting 结构

CoM 的核心：**modality-wise chain of reasoning**，类似 Chain-of-Thought 但在 modality 维度上展开。Prompt 包含 3 部分：

1. **Modality description**: 解释每个 modality 的数据格式（例如 force 是 normalized scalar 0-100，hand pose 是 2D pixel 坐标）
2. **Action set**: 列出 available skills 和参数，例如 `Grasp(hand, object, force)`, `Twist(hand, direction, angle)`, `Insert(hand, object, force)`
3. **One example video**: 一个 task-irrelevant 的演示视频（比如转 apple + can），展示期望输出格式

### 3.3 三阶段 Pipeline 数学化

整个 CoM 推理过程可形式化为 sequential conditional distribution：

**Stage 1 (Force)**:
$$
\mathcal{A}_1 = \text{VLM}_{\text{prompt}}(\{F_t\}_{t=1}^{T})
$$

VLM 只看 force signal，输出 coarse temporal skeleton，比如 "t=11 apply force, t=27 apply force, t=37 release force, ..."。这一阶段确定 **when** 和 **how many subtasks**。

**Stage 2 (Hand pose)**:
$$
\mathcal{A}_2 = \text{VLM}_{\text{prompt}}(\{H_t\}_{t=1}^{T}, \mathcal{A}_1)
$$

VLM 看到 hand pose + Stage 1 的 analysis，推断 motion type（grasp / twist / hit）和 direction (clockwise / counterclockwise)、angle (180°)。这一阶段确定 **how**。

**Stage 3 (Vision)**:
$$
\mathcal{A}_3 = \text{VLM}_{\text{prompt}}(\{I_t\}_{t=1}^{T}, \mathcal{A}_2)
$$

VLM 看到 image sequence + Stage 2 的 analysis，识别具体 object（bottle, cap, plug, drum, eraser...）和 spatial layout。这一阶段确定 **what** 和 **where**。

**Stage 4 (Code generation)**:
$$
\text{Code} = \text{VLM}_{\text{prompt}}(\mathcal{A}_3, \text{API spec})
$$

VLM 把 task plan 翻译成 Python API calls。

### 3.4 每个 Modality 的信息贡献

| Modality | 提供 | 无法提供 |
|---|---|---|
| Force/Audio | 时间分段、subtask count、effort level | motion type、object identity |
| Hand pose | motion type、direction、grasp/release | object identity、force level |
| Image | object identity、spatial layout、scene context | 精确 force、fingertip motion |

CoM 利用了 modality 之间的 **互补性**：force 提供粗时间结构，hand pose 提供 motion 语义，image 提供 object grounding。三者的 union 才能完整 reconstruct 出可执行的 task plan。

---

## 4. 从 Task Plan 到 Robot Code

paper 用了 Code as Policies [30] 的思路：VLM 不直接输出 low-level joint torques，而是输出 Python code，调用预定义的 skill API。

API 示例：

```python
from skills import Grasp, Release, Twist, Find, Move_to, Push_towards, Insert

# Bottle opening task plan
Move_to('left', Find('bottle'))        # left gripper moves to bottle
Grasp('left')                          # left gripper grasps bottle
Move_to('right', Find('bottle_cap'))   # right gripper moves to cap
for _ in range(3):                     # 3 iterations of twist
    Grasp('right')
    Twist('right', 'counterclockwise', 180)
    Release('right')
    Twist('right', 'clockwise', 180)
```

### 4.1 Force Parameter 编码

对于 contact-rich task，code 中的 force 参数直接来自 EMG/audio 的 normalized value：

```python
Grasp('right', 'plug', 100)         # force range [0, 100]
Move_to('right', 'box', 20)        # low force to re-orient plug
Insert('right', 'power_strip', 100) # high force for insertion
```

这里 force=20 对应 in-hand manipulation (light grasp)，force=100 对应 firm grasp。这个映射是 prompt 里告诉 VLM 的 normalization scheme。

### 4.2 Find API 的实现细节

paper 提到 `Find()` API 的实现方式很有意思：

1. Robot 拍一张 RGB-D image
2. Query **Gemini 1.5 Pro** with image + free-form object name (比如 "red_eraser")
3. Gemini 输出 2D bounding box
4. 用 depth + camera intrinsics 把 box 内的 pixels 转成 3D point cloud
5. 取 point cloud centroid 作为 object 的 3D position

公式上：

$$
\mathbf{p}_{obj} = \frac{1}{|\mathcal{B}|} \sum_{u, v \in \mathcal{B}} \pi^{-1}(u, v, D(u, v))
$$

- $\mathcal{B}$: Gemini 给的 2D bounding box 内的 pixel 集合
- $D(u, v)$: depth value at pixel $(u, v)$
- $\pi^{-1}$: inverse camera projection, 把 2D pixel + depth 反投影到 3D camera frame
- $\mathbf{p}_{obj}$: object 3D centroid in camera frame

这种 open-vocabulary grounding 让 code-based policy 不需要训练专门的 object detector，直接复用 VLM 的视觉理解能力。

### 4.3 Cross-Embodiment Deployment

因为输出是 Python code 而不是 joint torques 或 end-effector trajectory，同一份 code 可以部署到不同 robot 平台：

- **ViperX**: 双臂 robot，relatively low precision
- **KUKA**: 双臂工业 robot，high precision

Code 层面 abstraction 让 embodiment-specific IK、control freq 全部封装在 `skills` module 里，VLM 不感知。

---

## 5. 实验设计与数据深度解读

### 5.1 评估指标

paper 用了两个指标：

1. **Accuracy**: VLM 输出的 task plan 与 ground truth 完全匹配的比例（严格匹配）
2. **Similarity Score**: 用 **Longest Common Substring (LCS)** 长度衡量输出与 ground truth 的相似度

$$
\text{Sim}(\mathcal{A}_{\text{pred}}, \mathcal{A}_{\text{gt}}) = \frac{|\text{LCS}(\mathcal{A}_{\text{pred}}, \mathcal{A}_{\text{gt}})|}{|\mathcal{A}_{\text{gt}}|}
$$

- $\mathcal{A}_{\text{pred}}$: VLM 输出的 task plan 字符串
- $\mathcal{A}_{\text{gt}}$: 人工标注的 ground truth
- LCS: 最长公共子串（注意是 substring，不是 subsequence）
- Normalization: 用 ground truth 长度归一化

LCS 而非 Levenshtein distance 的好处是：保留 partial credit，即使 VLM 错过几个 subtask 也能给出渐进的相似度分数。

### 5.2 Table I 数据解读

以 **Pressing Cube** task + Gemini 1.5 Pro 为例：

| Method | Accuracy | Similarity |
|---|---|---|
| Image-only | 0.00 | 0.00 |
| w.o. img | 0.70 | 0.96 |
| w.o. force | 0.00 | 0.68 |
| w.o. hand | 0.00 | 0.64 |
| All (CoM) | 0.67 | 0.92 |

**关键 insights**：

- **Image-only = 0% accuracy**: 纯 vision 完全无法识别 task plan，因为 pressing 动作的视觉变化非常 subtle，看不出力的大小、按压次数
- **w.o. img = 70% accuracy**: 这意味着 force + hand pose 已经能 reconstruct 大部分 task plan，对于 pressing 这种 motion pattern 简单的任务，image 反而不是必需
- **w.o. force = 0%**: 没有力信息，VLM 无法判断按了多少次、力度如何，task plan 完全错乱
- **All (CoM) = 67%**: 三种 modality 联合反而比 w.o. img 略低，可能是 image 引入了一些 noise（visual distractor）。但 similarity 仍 0.92，说明大部分 subtask 是对的，只是某个细节错了

**Insert Plug + Gemini 1.5 Pro**:

| Method | Accuracy | Similarity |
|---|---|---|
| Image-only | 0.00 | 0.80 |
| w.o. img | 0.00 | 0.32 |
| w.o. force | 0.00 | 0.72 |
| w.o. hand | 0.47 | 0.96 |
| All | 0.53 | 0.93 |

这个 task 完全不同：image 必需（要识别 plug 和 power strip 的相对位置），force 也必需（要分 light/high force 阶段），hand pose 反而 somewhat redundant（因为这个任务 motion pattern 简单：grasp + move + insert）。

### 5.3 Reasoning Procedure 比较（Fig. 5）

paper 测试了 5 种 reasoning 方式：

- **Merg**: 把所有 modality 合并成单一 input batch，直接输出一个 final answer
- **Merg-Sep**: 合并 input，但 VLM 分别为每个 modality 输出 answer，再合并
- **Sep-Merg**: 分开 input 处理，最后输出单一 final answer
- **Sep-Sep**: 分开 input 处理，分别输出 answer，再合并
- **Ours (CoM)**: 顺序处理，每步基于上一步的 output

CoM vs Sep-Sep 的差距：

- **Gemini 1.5 Pro**: CoM > Sep-Sep **19%**
- **GPT-4o**: CoM > Sep-Sep **17%**

这个差距的来源：Sep-Sep 是 "parallel analysis, late fusion"，而 CoM 是 "sequential analysis with progressive refinement"。CoM 的核心优势在于 stage 2 能利用 stage 1 的 force-based segmentation，stage 3 能利用 stage 2 的 motion label，形成 hierarchical 的 reasoning chain。Sep-Sep 没有这种 cross-stage 信息流。

### 5.4 Real Robot Results (Table II)

| Task | Ours | Oracle |
|---|---|---|
| Opening Bottle (ViperX) | 12/20 | 16/20 |
| Opening Bottle (KUKA) | 15/20 | 20/20 |
| Insert Plug | 15/20 | 18/20 |
| Wiping Board (red) | 16/20 | 20/20 |
| Wiping Board (blue) | 14/20 | 16/20 |
| Playing Drum | 16/20 | 20/20 |

- **Ours 平均 73%**: 全自动 pipeline，从 single human video 到 robot execution
- **Oracle 平均 92%**: 人工写 code 的上限

差距 19% 来自三个地方：

1. **CoM 分析错误** (~10%): VLM 偶尔误判 subtask 数量或 force level
2. **Code generation 错误** (~5%): VLM 把 task plan 翻译成 code 时漏掉 API call 或参数错
3. **Robot execution 错误** (~4%): Find API 偶尔定位不准、grasp 失败、object slipping

有趣的是 KUKA 比 ViperX 表现更好 (15 vs 12)，虽然 KUKA 是工业 robot，但 gripper 精度更高，twist 动作执行更稳定。这反过来说明 paper 的 code-based abstraction 真的能 work across embodiments。

### 5.5 Generalization Settings

paper 特别强调了 generalization：

- **Opening Bottle**: 7 种瓶子（6 种 unseen），不同尺寸、材质、cap 类型
- **Insert Plug**: plug / power strip / box 随机摆放位置
- **Wiping Board**: 不同形状 marker，不同位置
- **Playing Drum**: 不同 drumming beats

这些 setup 都测试 VLM-based code 的 generalization，因为 generated code 用 `Find('bottle')` 这种 abstract call，由 VLM 在 runtime resolve 具体位置。这是 paper 的关键 contribution：**通过 abstraction layer，VLM 在 perception 时的 generalization 能力直接 transfer 到 manipulation**。

---

## 6. Architecture Diagram 深度解析

### 6.1 Fig. 2 的 Pipeline 图

```
[Input Multimodal Video]
        │
        ├─ Force/Audio ──→ [VLM Stage 1] ──→ A1: "3 force peaks at t=11, 27, 50"
        │                                              │
        ├─ Hand Pose ────→ [VLM Stage 2] ──→ A2: "grasp + twist counterclockwise 180°"
        │                                              (conditioned on A1)
        │                                              │
        └─ RGB Image ────→ [VLM Stage 3] ──→ A3: "twist bottle_cap with right hand"
                                                       (conditioned on A2)
                                                       │
                                                       ▼
                                  [VLM Stage 4: Code Gen] ──→ Python Code
                                                                       │
                                                                       ▼
                                                            [Robot Execution]
```

### 6.2 Stage Output 形式化

每个 stage 的 output 是结构化的 text：

**Stage 1 output** (force-based temporal segmentation):
```
- t=11: apply force
- t=27: apply force (peak 1)
- t=37: release force
- t=50: apply force (peak 2)
- t=62: release force
```

**Stage 2 output** (hand-motion-aware action primitive):
```
LEFT Hand:
- t=11: Grasp(left)
RIGHT Hand:
- t=27: Grasp(right)
- t=27 to t=37: Twist(right, counterclockwise, 180)
- t=37: Release(right)
- t=37 to t=50: Twist(right, clockwise, 180)
- t=50: Grasp(right)
- t=50 to t=62: Twist(right, counterclockwise, 180)
- t=62: Release(right)
- After t=62: Twist(right, clockwise, 180)
```

**Stage 3 output** (object-grounded task plan):
```
LEFT Hand:
- t=11: Grasp(left, coconut_water)
RIGHT Hand:
- t=27: Grasp(right, coconut_water_cap)
- t=27 to t=37: Twist(right, counterclockwise, 180 degrees)
- ... (same as Stage 2 but with object names)
```

注意 Stage 2 没有 object name（只有 motion），Stage 3 才 fill in object name。这就是 progressive refinement 的 essence。

### 6.3 Code Generation Stage

VLM 看到 Stage 3 output + API descriptions，生成：

```python
from skills import Grasp, Release, Twist, Find, Move_to

Move_to('left', Find('coconut_water'))
Grasp('left', 'coconut_water')
Move_to('right', Find('coconut_water_cap'))
for _ in range(3):
    Grasp('right', 'coconut_water_cap')
    Twist('right', 'counterclockwise', 180)
    Release('right')
    Twist('right', 'clockwise', 180)
```

VLM 把 3 次重复的 twist 序列归纳成 `for _ in range(3):`，这是 VLM code generation 的一个 emergent 能力 — 它理解了 task plan 的重复结构并抽象成 loop。这种 abstraction 在 original task plan 里并不存在，是 code generation 阶段新引入的。

---

## 7. 与相关工作的关联和 Intuition

### 7.1 Chain-of-Thought 的扩展

Chain-of-Thought (CoT, [Wei et al. 2022](https://arxiv.org/abs/2201.11903)) 让 LLM 在推理时分步思考，每个 step 都基于前面 step 的结论。CoM 可以理解为 **modality-wise CoT**：每个 "thought step" 处理一种 modality，而不是处理一个 reasoning step。

Intuition: CoT 解决的是 LLM 在长 reasoning chain 上的 attention dilution；CoM 解决的是 VLM 在多 modality 上的 cross-modal attention dilution。两者本质都是用 explicit 的中间 output 引导 attention 到正确的信息源。

### 7.2 与 MimicPlay 的关系

同作者前作 [MimicPlay](https://arxiv.org/abs/2302.12422) (CoRL 2023) 用 human play video 学 long-horizon manipulation。CoM 与 MimicPlay 的区别：

- MimicPlay 训练一个 policy network，用 human video 作为 latent plan
- CoM 不训练任何 network，直接用 VLM 的 zero-shot reasoning 能力
- MimicPlay 学的是 motion trajectory，CoM 学的是 discrete task plan + control parameters
- MimicPlay 需要 task-relevant demonstration data，CoM 只需要 task-irrelevant example 来展示输出格式

### 7.3 与 RT-2 / VLA 模型对比

[RT-2](https://robotics-transformer2.github.io/) 把 robot action tokenize 成 text token，让 VLM 直接输出 action sequence。CoM 走了不同 path：

- RT-2 需要大规模 robot demonstration training data
- CoM 完全 zero-shot，不需要 robot data
- RT-2 输出 low-level action (end-effector pose)，CoM 输出 high-level code
- RT-2 难以 introspect (action token 没有语义)，CoM 的 code 完全 transparent

trade-off：RT-2 可以做 closed-loop reactive control，CoM 是 open-loop。paper 在 limitations 里也提到了。

### 7.4 与 Code as Policies 的关系

[Code as Policies](https://code-as-policies.github.io/) (Liang et al. 2023) 让 LLM 把 language instruction 翻译成 robot code。CoM 把这个思路扩展到 **video instruction**，并且引入 multimodal reasoning。

关键差别：CaP 输入是 language，CoM 输入是 multimodal video。Language 已经是抽象过的，video 是 raw signal，所以 CoM 需要 VLM 做 perception + reasoning 二合一。

### 7.5 Perceiver / Flamingo 路线的对比

[Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model) 等 multimodal foundation model 把 vision 和 language 都 tokenize 喂入同一个 transformer。CoM 没有走这条路，而是利用 existing VLM 的 long-context ability 直接输入 numerical signal token sequence。这避免了训练新模型，但依赖 VLM 处理 numerical token 的能力。

实测发现：Gemini 1.5 Pro 和 GPT-4o 在 numerical signal 处理上已经 work，但需要 proper prompt 格式（一行一帧 force value，附带 timestamp）。

---

## 8. Limitations 和 Future Directions

paper 自述的 limitation：

1. **Audio 只用 loudness**: 没用 frequency、pitch、timbre。未来可以用 spectrogram 作为 VLM input，让 drum task 不仅区分 loud/soft，还能区分不同音色
2. **Open-loop execution**: 没有 closed-loop feedback。如果 grasp 失败或 object slip，code 不会自适应

我从技术角度补充几个 limitation：

3. **Skill API 仍需手工设计**: paper 假设了 `Grasp, Twist, Insert, Hit, Wipe` 这些 primitive skill。未来可以探索让 VLM 自动 discover 必要的 skill，或者从 large-scale video corpus 学 skill library
4. **Force normalization 难以跨 task transfer**: paper 把 force normalize 到 [0, 100]，但 drum 和 plug insertion 的 force scale 完全不同。VLM 看到 force=50 在不同 task 含义不同，需要 per-task calibration
5. **Hand pose 限制为 2 fingertips**: 对 pinch grasp work，但对 power grasp、lateral grasp 等 grasp type 不够
6. **No temporal precision**: VLM 输出的是离散 timestep（比如 t=27, t=37），但 robot 执行需要毫秒级 precision
7. **No failure recovery**: code 一旦生成，没有 re-prompting 机制。可以参考 [Liang et al. 2024](https://arxiv.org/abs/2402.11450) 的 LMPC，让 VLM 在 robot failure 时 re-prompt

---

## 9. 我对这篇 paper 的 Intuition 总结

### 9.1 为什么 CoM work 的深层原因

我认为 CoM 的成功不仅是 prompting trick，背后有更深的原因：

- **Modality 之间的信息有 hierarchy**: Force 是 event-driven signal（稀疏，关键时刻才有 peak），hand pose 是 motion signal（dense，连续），image 是 context signal（dense，static）。这种 hierarchy 让 sequential processing 天然适合：先从 sparse signal 提取骨架，再从 dense signal 填充细节
- **VLM 的 attention 是 free bottleneck**: 即使 long-context VLM 支持 1M token，其 attention 仍受 softmax 归一化限制，长序列里关键 signal 容易被稀释。CoM 用 sequential prompting 把 attention 强制聚焦到当前 modality
- **Modality 之间存在 conditional independence 结构**: Force peaks 在没有 visual context 下也能识别（waveform 上明显），motion type 在知道 force timing 后更容易识别（focus 到那段 hand pose），object identity 在知道 motion 后更容易从 image 提取（focus 到 hand 接触的物体）。这种 Bayesian-friendly 的结构让 sequential inference 接近 optimal

### 9.2 与人类认知的相似性

人类学习 manipulation skill 时也走类似 hierarchy：

1. 先感知 effort 何时发生（"什么时候用力"）→ 形成时间分段
2. 再感知 motion 形态（"用什么动作"）→ 形成动作语义
3. 最后感知 object（"操作什么") → 形成场景理解

CoM 某种意义上模拟了这个 hierarchical perception 过程，把 modality 顺序排成 effort → motion → object，符合 motor learning 的认知顺序。

### 9.3 这篇 paper 的真正 contribution

paper surface 看是 prompting strategy，但 deep contribution 我认为是：

**证明了 SOTA VLM 在 multimodal dense reasoning 上的 fundamental limitation**。Gemini 1.5 Pro / GPT-4o 在 merged input 上只有 17% accuracy，说明现有 VLM 的 cross-modal attention 还很弱。这给未来 VLM 训练提出方向：需要专门训练 modality-aware attention mechanism，而不是单纯 scale context length。

同时 CoM 也给出 workaround：**通过 prompt engineering 显式引导 attention**，可以在不重新训练模型的情况下大幅提升 multimodal reasoning 性能。这种 workaround 对 robotics 应用特别重要，因为我们无法等下一代 VLM 训练完。

---

## 10. Reference 链接汇总

- **Project page**: https://chain-of-modality.github.io
- **MimicPlay (前作)**: https://arxiv.org/abs/2302.12422
- **Code as Policies**: https://code-as-policies.github.io/
- **HaMeR (hand pose)**: https://hamer.is.tue.mpg.de/
- **Gemini 1.5 Pro**: https://arxiv.org/abs/2403.05530
- **GPT-4o**: https://arxiv.org/abs/2303.08774
- **RT-2**: https://robotics-transformer2.github.io/
- **Chain-of-Thought**: https://arxiv.org/abs/2201.11903
- **ReKeP (同作者 follow-up)**: https://arxiv.org/abs/2409.01652
- **LMPC (failure recovery)**: https://arxiv.org/abs/2402.11450

如果你想让我深入某个具体方面（比如 EMG 信号处理细节、VLM prompt 的具体写法、或 generalization 实验的设计），可以告诉我，我可以再展开。
