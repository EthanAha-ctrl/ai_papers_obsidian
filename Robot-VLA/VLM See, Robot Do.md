---
source_pdf: VLM See, Robot Do.pdf
paper_sha256: 21b5bbb045a34581f6cdb14f29ee5b4cdd2e315e9e952ab740eb2340c23a13c8
processed_at: '2026-08-13T03:05:39-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

这篇 paper 的核心想法特别直觉：你让机器人看一段人类做饭、叠衣服的长视频，然后让机器人照着做。直接把视频塞给 GPT-4o 这种 Vision Language Model (VLM) 根本行不通，VLM 会看晕，它记不住那么多 frame，也分不清桌上长得一模一样的几个 wooden block 到底谁是谁，更别提判断谁在谁左边了。所以作者搞了个叫 SeeDo 的系统，把这个复杂任务拆成了几个步骤，用工程手段强行弥补了现在 VLM 的短板。

咱们顺着直觉走一遍这套系统是怎么运作的。

### 1. 痛点与 Keyframe Selection: 找准“关键时刻”

长视频动辄几百帧，全塞进去 VLM 的 context window 爆了，且信息冗余。大部分时间手都在移动，真正决定任务状态的只有抓起和放下那一瞬间。怎么找这一瞬间？SeeDo 用了一个特别聪明的物理先验：**手速**。

它用 MediaPipe 提取手部 21 个 keypoints，计算中心点速度。
公式长这样：
$$v_t = \frac{1}{K} \sum_{i=1}^{K} \| \mathbf{p}_t^i - \mathbf{p}_{t-1}^i \|_2$$
变量解释：$t$ 是当前 frame index，$t-1$ 是上一帧，$K$ 是 keypoint 总数 (MediaPipe 手部模型 $K=21$)，$\mathbf{p}_t^i$ 是第 $i$ 个 keypoint 在 frame $t$ 的 2D 坐标。$\| \cdot \|_2$ 是 L2 norm，算的是两点间的欧氏距离。这算出来就是一个时序速度波。

手在拿东西或者放东西时，必然减速停顿，对应速度波的 trough (波谷)。把这些波谷对应的 frame 抽出来当 keyframe，既省了 context，又精准捕捉了 state transition。这比那些 open-source VLM 傻乎乎地 uniform sampling 16 帧强太多了。你看实验数据表 (Table II)，把 SeeDo 换成 uniform sampling，Wooden Block 任务的 Step Success Rate (SSR) 直接掉到 0.00%，几乎全废。

### 2. Visual Prompting: 给物体发“身份证”

找到 keyframe 了，但 VLM 还是个近视眼加脸盲。特别是面对一堆没颜色的 wooden block，VLM 根本分不清。

SeeDo 的办法是给画面打上 "Visual Prompt"。首先让 VLM 看第一帧，列出桌上有哪些 object。接着用 Grounding DINO 去框出这些物体的 bounding box，再用 SAM 2 (Segment Anything Model 2) 对整个 video 进行 tracking。

最绝的一点，它没有直接把黑乎乎的 mask 盖在物体上（那样物体本身的颜色纹理就看不见了），它把 mask 的轮廓线，加上一个 tracking ID（比如 ID:0, ID:1）画在 keyframe 上。同时，把 mask 中心的 pixel 坐标 $(x, y)$ 贴到 text prompt 里。
这等于给每个物体发了身份证，还顺便告诉了 VLM 它们的坐标。

### 3. VLM Interpreter: 把空间感知变成读数字

到了 VLM Interpreter 这步，GPT-4o 看到的就是一系列带编号、带轮廓线的清晰图片。这里用了 Chain of Thought (CoT)。先判断这帧手有没有拿东西，滤掉无效 frame。然后一张张过：看 pick frame，问“手里拿的是哪个 ID？”，再看 place frame，问“放在哪个 ID 旁边了？什么空间关系？”。

这里定义了 6 种离散的空间关系：`in`, `on top of`, `at the back of`, `in front of`, `to the left`, `to the right`。
为什么这个模块能 work？因为之前的 visual prompt 把最难的“视觉识别”和“空间定位”变成了简单的“读数字”和“认标签”。VLM 只要看到 text prompt 里写着 ID:1 的坐标 $x=100$, ID:0 的坐标 $x=200$，它就能顺理成章地推断出 ID:1 在 ID:0 的左边。这就巧妙避开了 VLM 在 pixel-level 空间推理上的弱势，把空间关系 hack 成了 text 比大小。

### 4. 实验数据的直观解读

你看 Table I 的实验数据，非常直观。

在 Vegetable 任务里，SeeDo 的 Task Success Rate (TSR) 是 60.53%，而 LLaVA-OneVision 这些 open-source 模型基本是 0.00%。Gemini 1.5 Pro 凭借超长 context 勉强达到 39.47%。

最能体现这套系统工程价值的是 Wooden Block Stacking 任务。在这个任务里，所有长得一样的木块堆积，对 VLM 的空间和视觉能力要求极高。没有任何一个 baseline 能达到非零的 TSR，连 Gemini 也只有 0.00% 的 TSR 和 13.80% 的 SSR。SeeDo 硬是做到了 21.62% 的 TSR 和 52.48% 的 SSR。全靠 Visual Prompting 模块给木块发的 tracking ID。

你去看 Table III 的 ablation study，在 Block 任务里把 Visual Prompting 去掉，SeeDo 的 Spatial Error 直接从 64.10% 飙升到 87.50%，TSR 归零。这就证明了，现在的 VLM 离开这种外部辅助，根本处理不了精细的物体间关系。

### 5. Intuition 总结与联想

这套系统的本质，是把一个端到端难以优化的黑盒问题，通过引入人类物理常识（手速）和外部感知工具（SAM2, Grounding DINO），强行拆解成了几个 VLM 擅长的 sub-task。VLM 在这里只扮演了一个 reasoning engine 的角色，干着看图说话的活，真正的脏活累活（tracking, detection, temporal localization）都被工具包揽了。

这种 modular design 的思路在当前 foundation model 还不够完美的阶段极具实用价值。它揭示了一个事实：限制 VLM 在 robotics 落地的关键痛点是它的空间感知和时序记忆太差。与其等下一代 VLM 自己长出这些能力，用 pipeline 把它们补齐是更务实的做法。

往深了想，这里定义的 6 种离散空间关系其实是一个很强的 bottleneck。真实的机器人操作往往需要连续的 6D pose。未来的方向肯定是在这个 text plan 的基础上，引入更精细的 keypoint detection 或者 3D scene representation，把 "to the left of" 这种粗略描述，变成具体的 $(x, y, z, \text{roll}, \text{pitch}, \text{yaw})$ 坐标，这样才能真正释放机器人执行复杂操作的潜力。

### Web Links for Reference
*   **SeeDo Project Page**: [ai4ce.github.io/SeeDo](https://ai4ce.github.io/SeeDo/)
*   **SAM 2 (Segment Anything Model 2)**: [github.com/facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2)
*   **Grounding DINO 1.5**: [github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
*   **MediaPipe Hands Keypoint Model**: [google.github.io/mediapipe/solutions/hands.html](https://google.github.io/mediapipe/solutions/hands.html)
*   **Code as Policies**: [sites.google.com/view/codeaspolicies](https://sites.google.com/view/codeaspolicies)

---

这篇 paper 提出了一个名为 SeeDo 的 modularized agent pipeline, 旨在通过 Vision Language Model (VLM) 解读人类示范视频, 从而为机器人生成 long-horizon 的 task plan. 核心动机在于纯 language instruction 在描述 long-horizon task 时显得繁琐且低效, 而 video 蕴含了丰富的 temporal 和 spatial dependencies. 传统 imitation learning 直接从 video 映射到 robot action, 严重受限于 human-robot embodiment gap 以及 long-horizon task 对海量 demonstration data 的需求. SeeDo 选择了另一条路径: 利用 VLM 强大的 common sense reasoning 和 zero-shot generalization 能力, 将 video 翻译成自然语言描述的子任务序列, 再通过 Language Model Programs (LMPs) 调用底层 action primitives 执行.

### I. Architecture 解析

SeeDo 的核心架构由三个 module 组成: Keyframe Selection, Visual Prompting, 以及 VLM Interpreter. 这三个 module 串联解决了 VLM 在处理 video 时的三个核心痛点: context length limit, visual shortcoming (尤其是 spatial reasoning), 以及 temporal order confusion.

#### 1. Keyframe Selection Module
长视频包含大量冗余帧, 直接输入会超出 VLM 的 context window, 同时引入噪声. Open-source VLM 通常采用 uniform sampling, 这种策略极易遗漏关键的 pick-and-place 动作瞬间. 

SeeDo 采用 hand-speed heuristic. 人类在执行 pick 或 place 动作时, 手部速度会显著降低. 模块使用轻量级模型 MediaPipe 提取手部 21 个 3D keypoints. 手部中心点速度计算公式如下:
$$v_t = \frac{1}{K} \sum_{i=1}^{K} \| \mathbf{p}_t^i - \mathbf{p}_{t-1}^i \|_2$$
其中, $t$ 代表 frame index, $K$ 是手部 keypoints 的总数 (MediaPipe 中通常 $K=21$), $\mathbf{p}_t^i$ 是第 $i$ 个 keypoint 在 frame $t$ 的 2D 或 3D 坐标. 计算出的速度序列 $v_t$ 经过 linear interpolation 和 smoothing 处理后, 形成波形图. 波形的 troughs (极小值点) 对应着手部移动最慢的时刻, 即抓取或放置的关键瞬间, 这些帧被选为 keyframes.

#### 2. Visual Prompting Module
当前 VLM (如 GPT-4o) 在精确识别物体空间关系和区分视觉相似物体 (如 uncolored wooden blocks) 上存在严重的 visual shortcomings. 此模块旨在通过 external visual cues 增强 VLM 的 perception 能力.

流程如下:
1. VLM 首先被 prompt 去识别环境中的 object list.
2. 使用 open-vocabulary object detector Grounding DINO 1.5 在 first frame 中提取物体的 bounding boxes.
3. 将 bounding boxes 作为 prompt 输入给 SAM 2 (Segment Anything Model 2) 进行 video-level tracking. SAM 2 会为每个物体分配一个 tracking ID 并生成 mask.
4. 将 mask 的 contour (轮廓线) 和 tracking ID 渲染到之前选出的 keyframes 上. 使用 contour 而非 full mask 是为了不遮挡物体本身的 visual appearance, 保留 VLM 原生的视觉理解能力.
5. 将 mask center 的 pixel coordinates $(x, y)$ 结合 tracking ID append 到 text prompt 中, 隐式地向 VLM 注入 spatial relationship 信息.

#### 3. VLM Interpreter Module
有了带有 visual prompts 的 keyframes, VLM Interpreter 利用 Chain-of-Thought (CoT) 逐步生成 task plan. 采用 GPT-4o 作为核心 engine. CoT 设计将复杂的 long-horizon 推理拆解为局部决策:

*   **Step 1: Filter invalid keyframes.** 剔除手部未抓取任何物体的误选帧.
*   **Step 2: Identify object picked.** 基于当前的 pick frame, 结合 text prompt 中的 object list, 判断正在被抓取的物体.
*   **Step 3: Identify reference object.** 结合 place frame 和已确定的 object picked, 判断放置位置的 reference object (例如 container 或 另一个 block).
*   **Step 4: Reason spatial relationship.** 推理 object picked 相对于 reference object 的空间位置, 定义了 6 种 discrete spatial relations: `in`, `on top of`, `at the back of`, `in front of`, `to the left`, `to the right`.

最终输出格式为 "Drop [object picked] to the [spatial relation] of the [reference object]".

#### 4. Plan Execution
生成的 text plan 被输入给 Code as Policies 框架, 转化为 Language Model Programs (LMPs). LMPs 是一段段 Python code, 调用预定义的 action primitive functions (如 `pick(obj)`, `drop(loc)`). 在 real-world experiment 中, 使用 Intel RealSense 455 获取 RGB-D 数据, 结合 segmentation model 获取物体的 3D 坐标, 控制 UR10e 机械臂执行.

### II. Evaluation Metrics 设计

Paper 提出了三个 metric 来全面评估 long-horizon pick-and-place task. 假设 $P$ 为 predicted plan steps, $G$ 为 ground truth steps.

1.  **Task Success Rate (TSR):** 严格评估. 当且仅当 $P$ 与 $G$ 在 content 和 temporal order 上完全匹配时, $TSR=1$, 否则 $TSR=0$.
    $$TSR = \mathbb{I}(P \equiv G)$$
    其中 $\mathbb{I}$ 是 indicator function.

2.  **Final-state Success Rate (FSR):** 忽略 temporal order, 只关注最终状态. 将 plan 转化为物体间的 spatial relation pairs 集合 $S$. 如果 $S_{pred} == S_{gt}$, 则 $FSR=1$.
    $$FSR = \mathbb{I}(S_{pred} \equiv S_{gt})$$

3.  **Step Success Rate (SSR):** 评估 partial completeness. 使用双指针 algorithm 计算 aligned steps 数量.
    Algorithm 逻辑: 初始化 $\text{MATCH} \leftarrow 0$, $\text{ptr}_g \leftarrow 1$, $\text{ptr}_p \leftarrow 1$. 遍历 $P$, 对于每个 $p \in P$, 从 $\text{ptr}_g$ 开始遍历 $G$, 如果 $p$ matches $g$, 则 $\text{MATCH} \mathrel{+}= 1$, $\text{ptr}_g \mathrel{+}= 1$, break 并进入下一个 $p$.
    $$SSR = \frac{\text{MATCH}}{|G|}$$
    其中 $|G|$ 是 ground truth 的总步数. SSR 反映了模型在出错前能正确执行多少步.

### III. Experiments & Data Analysis

#### 1. Baselines 对比
对比了 SOTA video-input VLMs: LLaVA-OneVision, LLaVA-NeXT-Video-7B, VILA1.5-8B, Gemini 1.5 Pro, 以及 GPT-4o (Init+Final frames).

**Table I 数据解读:**
*   **Vegetable Organization:** SeeDo 达到 $TSR=60.53\%$, $SSR=80.40\%$. Gemini 1.5 Pro 仅为 $39.47\%$ 和 $70.00\%$. Open-source models 几乎全军覆没 ($TSR=0$).
*   **Wooden Block Stacking:** 这是极具挑战性的任务, 因为 blocks 视觉高度相似. 所有 baselines 的 $TSR$ 均为 $0.00\%$. Gemini 1.5 Pro 的 $SSR$ 仅为 $13.80\%$. SeeDo 达到 $TSR=21.62\%$, $SSR=52.48\%$. 这证明了 Visual Prompting Module 在区分相似物体时的决定性作用.
*   **GPT-4o (I+F):** 仅仅依赖 first and last frame, $TSR$ 很低, 但在 garment task 上 $FSR$ 略高, 体现了 GPT-4o 强大的 common sense (能猜出最终状态), 但缺乏 temporal reasoning 能力.

#### 2. Ablation Studies
**Table II: Keyframe Selection Ablation**
将 hand-speed heuristic 换成 uniform sampling (SeeDo Unif.), 性能断崖式下降. Vegetable task 的 $SSR$ 从 $80.40\%$ 降至 $1.32\%$, Block task $SSR$ 降为 $0.00\%$. Uniform sampling 极易采到过渡帧, 破坏了 VLM 对 action 边界的理解.

**Table III: Visual Prompting Ablation**
在 Wooden Block Stacking 任务上移除 Visual Prompting (SeeDo w/o V.P.).
*   $TSR$ 从 $21.62\%$ 降至 $0.00\%$.
*   Vision Error 从 $20.51\%$ 升至 $42.86\%$.
*   Spatial Error 从 $64.10\%$ 升至 $87.50\%$.
数据直接证明了 SAM2 tracking 提供的 contour 和 ID 极大缓解了 VLM 的视觉缺陷.

#### 3. Failure Case Analysis
Paper 将错误归为三类: Vision Error (识别错误), Spatial Error (空间关系错误), Temporal Error (时序错误). SeeDo 的 Temporal Error 显著低于其他 models, 得益于 keyframe 的时间序列化处理. 但 Spatial Error 依然是 SeeDo 的主要失败来源, 一方面因为 VLM 本身 spatial intelligence 有限, 另一方面 SAM2 的 tracking 偶尔会出现 ID switch, 导致 visual prompt 与 text 不一致.

### IV. Intuition & Insights

1.  **Modularity vs End-to-End:** SeeDo 展现了 modular design 在当前 AI 阶段的优势. 直接用 VLM 处理 video 失败率极高. 将 perception (SAM2, Grounding DINO), temporal localization (MediaPipe), 和 reasoning (GPT-4o) 解耦, 允许每个 module 发挥其最强性能. 这类似于人类认知中的 System 1 (fast perception) 和 System 2 (slow reasoning) 的分工.
2.  **Inductive Bias 的价值:** Hand-speed heuristic 是一个极强的 inductive bias. 在 pick-and-place task 中, 手部速度的极小值点几乎必然对应 state change. 这种物理先验比让 VLM 去 "理解" 整个视频的动态要高效得多.
3.  **Visual Prompting as Spatial Crutch:** VLM 在纯 pixel-level 的 spatial reasoning 上非常弱. 给出 mask center 的 $(x, y)$ 坐标, 本质上是将 2D spatial relationship 转化为 VLM 擅长的 text token 处理. VLM 通过比较数字大小来推断 "left/right", 通过 mask contour overlap 来推断 "in/on top of". 这是一种巧妙的 cross-modal grounding.
4.  **Discrete Action Space Limitation:** 当前 plan 局限于 6 种 discrete spatial relations. 这限制了机械臂执行需要 precise pose 的任务. 下一步自然是引入 continuous coordinate 或 6D pose estimation, 结合 VLM 生成 parametrized actions.

### V. Web Links for Reference

*   **Code & Demos:** [ai4ce.github.io/SeeDo](https://ai4ce.github.io/SeeDo/)
*   **Code as Policies:** [sites.google.com/view/codeaspolicies](https://sites.google.com/view/codeaspolicies)
*   **SAM 2 (Segment Anything Model 2):** [github.com/facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2)
*   **Grounding DINO 1.5:** [github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
*   **MediaPipe Hands:** [google.github.io/mediapipe/solutions/hands.html](https://google.github.io/mediapipe/solutions/hands.html)
*   **GPT-4o Technical Report:** [arxiv.org/abs/2303.08774](https://arxiv.org/abs/2303.08774)
*   **PyBullet:** [pybullet.org](https://pybullet.org/)
