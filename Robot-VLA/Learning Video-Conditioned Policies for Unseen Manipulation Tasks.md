---
source_pdf: Learning Video-Conditioned Policies for Unseen Manipulation Tasks.pdf
paper_sha256: f6f1fb6ca96d00424f5aca694792690aae02a802493d19172ec05341bdfa235e
processed_at: '2026-08-05T14:12:26-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

这 篇 paper 的 story 用 人话 讲 就是： 怎么 让 机器人 听懂 “ 人 做 事 的 视频指令 ”。

### 1. 要 solve 的 痛点
传统 robot learning 要么 需要 专家 手动 写 reward function， 要么 需要 给 机器人 演示 一遍 (teleoperation)， 成本 极 高 且 难以 scale。 我们 想 让 普通人 随手 拍 个 视频 说 “ 帮 我 推 一下 杯子 ”， 机器人 就 能 照做。 但 问题 是， 人 的 视频 和 机器人 的 视频 (pixel-level) 长得 完全 不一样， domain gap 巨大， 机器人 直接 看 不懂。

### 2. 核心 Intuition: 共享 “动作语义空间”
论文 的 core insight 极其 优雅。 如果 有 一个 神奇的 video encoder， 它 能 把 “ 人 推 杯子 ” 和 “ 机器人 推 杯子 ” 这 两个 视觉上 完全不同 的 video， map 到 embedding space 里 相邻 的 位置 (因为 它们 的 semantic action 相同)。 那么 domain gap 就 消失 了。

为了 得到 这个 encoder， 作者 利用了 大规模 的 human action recognition dataset (Something-Something-v2, SSv2)， 里面 全是 人 跟 物体 交互 的 视频。 用 Supervised Contrastive Learning (SupCon) 去 train 它。 SupCon 的 威力 在于， 它 强迫 同类 动作 的 video embedding 在 unit hypersphere 上 聚 成 一团， 异类 动作 互相 推开。 结果 就是， 这个 encoder 学会 了 抓住 动作 的 “ 本质 semantic”， 而 忽略 表面 的 visual appearance (比如 人手 还是 机械臂)。 论文 实验 发现， 这个 只 在 human video 上 train 的 encoder， 直接 拿来 encode robot video， 居然 也 能 很好 地 cluster 出 不同 的 robot task。 这 就是 zero-shot 跨域 的 基础。

### 3. 具体 Pipeline: 类似 Robotics 的 RAG
整个 pipeline 可以 拆 成 三步， 极其 类似 NLP 里 的 Retrieval-Augmented Generation (RAG)。

**Step 1: Build Robot Knowledge Base (Training phase)**
先 收集 一大堆 机器人 的 随机 轨迹 数据 (unlabelled， 不 需要 知道 在 干啥)。 用 刚才 那个 frozen 的 video encoder， 把 每条 轨迹 的 video encode 成 一个 embedding， 存 进 一个 library。 同时， train 一个 behavior cloning (BC) policy。 这个 policy 的 输入 是 当前 机器人 state $s$ 和 某条 轨迹 的 video embedding $e^r$， 输出 是 action $a$。 这 等于 让 policy 学会： “ 如果 你 要 执行 这个 embedding 代表 的 技能， 那么 在 这个 state 下 应该 发出 这个 action ”。 对应 公式 (1) 的 negative log-likelihood loss。 这里 的 intuition 是， video embedding 相当于 一个 极其 丰富 的 “ task token ”， 它 包含 了 整条 轨迹 的 future intent。

**Step 2: Translate Human Instruction (Inference phase)**
给 一个 人 的 视频 指令 $x^h$。 用 同一个 encoder 把 它 encode 成 human embedding $e^h$。 关键 来 了： 不 能 直接 把 $e^h$ 喂 给 policy！ 因为 policy 只 在 robot embedding 分布 上 train 过， $e^h$ 对 它 来说 是 out-of-distribution (OOD)， 会 直接 崩掉。 所以 需要 “翻译”。 怎么 翻译？ 用 k-Nearest Neighbors (kNN) regression。 去 Step 1 build 好 的 robot embedding library 里， 找 出 与 $e^h$ cosine similarity 最高 的 top-k 个 robot embeddings， 然后 取 平均， 得到 translated embedding $e^r$。 这个 $e^r$ 既 语义 上 贴近 人 的 指令， 又 依然 处 于 policy 的 训练 分布 内。

**Step 3: Rollout**
把 $e^r$ 喂 给 trained policy， 机器人 就 开始 一 步 步 rollout， 执行 技能。

### 4. 架构图 解析 (Figure 2)
- **Left (Training)**: Unlabelled robot trajectory $\rightarrow$ Video Encoder $f_\theta$ $\rightarrow$ Embedding $e^r$ $\rightarrow$ 存入 Library。 同时 $e^r$ 和 state $s$ 一起 喂给 Policy $\pi_\phi$ 预测 action， 算 BC loss。
- **Right (Inference)**: Human video $x^h$ $\rightarrow$ Video Encoder $f_\theta$ $\rightarrow$ Human embedding $e^h$。 在 Library 里 做 kNN， 取 top-k 平均 得到 $e^r$。 最后 $e^r$ condition policy 生成 action。

这个 架构 极其 简洁， 没有复杂的 video generation (像 baseline DVD 那样 去 predict future video frames)， 而是 全部 在 pre-computed embedding space 里 操作， 所以 推理 极快 (论文 提到 ViP <1s， DVD >16s)。

### 5. 公式 细节 剖析
**公式 (1)**: $\mathcal{L}_{\pi}(\phi) = - \mathbb{E}_{s,a,x^r \sim D^r} \log \pi_\phi(a | s, f_\theta(x^r))$
- $\phi$: Policy network 参数。
- $D^r$: Unlabelled robot dataset。
- $f_\theta(x^r)$: Frozen video encoder 提取 的 trajectory embedding， 相当于 task context。 这里 $f_\theta$ 参数 $\theta$ 被 frozen 掉， 只 train $\phi$。 这 就 把 高维 video-to-action mapping 拆解 成 了稳定的 embedding-to-action regression。

**公式 (2)**: SupCon loss。 这里 $\tilde{x}$ 是 augmented video， $P(i)$ 是 batch 内 与 anchor 同 label 的 positive index 集合， $A(i)$ 是 除 自己外 所有 index。 核心 是 分母 考虑 了 batch 内 所有 样本， 分子 只 累加 正 样本。 $\tau$ 是 temperature， 控制 softmax 的 sharpness。 训练 完 后， $f_\theta$ 就 成 了 一个 semantic similarity 度量 函数 $d(x^h, x^r) = \langle f_\theta(x^h), f_\theta(x^r) \rangle$。

### 6. 实验 数据 解读 Intuition
- **Table I**: 即使 在 seen robot demos setting， ViP 也 碾压 DVD。 这 说明 用 full trajectory embedding 做 planning， 比 DVD 用 sub-trajectory 做 video prediction 要 robust 得多。
- **Table II (Zero-shot)**: 这 是 最 亮 眼 的 结果。 当 完全 不 用 paired human-robot data 时， DVD 掉 到 36.8%， 而 ViP (Ours) 依然 有 71.3%。 “Human video as input” 这个 baseline 只有 43%， 证明 了 OOD 问题 的 严重性 以及 kNN translation 的 必要性。
- **Table III (Kitchen)**: ViP (Oracle) 接近 100%， 说明 policy 容量 足够。 但 ViP (Ours) 在 sliding door 和 microwave 上 跌得 厉害。 原因 很 直觉： SSv2 的 label 是 "Opening [something]"， 它 不 区分 开 的是 门 还是 微波炉。 所以 encoder 无法 提供 fine-grained object semantics， 导致 retrieve 错 了 robot embedding。 这 揭示 了 当前 semantic representation 的 bottleneck。

### 7. 脑洞 联想: 与 Foundation Model 的 结合
这个 方法 本质 上 是 机器人 版 的 CLIP retrieval。 如果 我们 把 SSv2 上 的 SupCon 换成 用 更 大规模 的 video-language model (比如 VideoCLIP, InternVideo) 去 对齐， 甚至 直接 用 VLM 的 text embedding 做 bridge， 也许 能解决 Kitchen 环境 里 的 object ambiguity 问题。 另外， kNN 是 一个 极其 simple 的 non-parametric memory， 完全 可以 升级 成 parametric 的 Transformer cross-attention， 让 policy 自己 学会 去 soft-retrieve， 这样 可能 能 处理 更 continuous 的 action space。

### Reference Web Links
1. **ViP Paper (arXiv)**: https://arxiv.org/abs/2202.10788
2. **Something-Something-v2 (SSv2) Dataset**: https://developer.qualcomm.com/software/ai-datasets/something-something
3. **Supervised Contrastive Learning Paper**: https://arxiv.org/abs/2004.11362
4. **DVD (Domain-agnostic Video Discriminator) Paper**: https://arxiv.org/abs/2103.16817
5. **R3M Representation Paper**: https://arxiv.org/abs/2203.12601
6. **Decision Transformer Paper**: https://arxiv.org/abs/2106.01345

---

这 篇 paper 的 core intuition 在于， 通过 Supervised Contrastive Learning (SupCon) training 得到 的 video encoder， 能够 将 human video 和 robot video map 到 同一个 semantic embedding space。 因为 SSv2 dataset 包含 了 massive 的 human-object interaction， 所以 其 learn 到 的 representation 天然 contain 了 task semantics。 从而， 我们 不 需要 在 training phase collect paired human-robot data。 在 inference phase， given 一个 human video instruction， 我们 先 use encoder extract embedding， 然后 在 预先 build 的 robot embedding library 中 perform kNN retrieval。 最后， 我们 用 retrieve 出 的 in-distribution robot embedding 去 condition behavior cloning policy， 从而 generate robot action。

### Method & Formula Breakdown

**公式 (1) Behavior Cloning Loss:**
$$
\mathcal { L } _ { \pi } ( \phi ) = - \mathbb { E } _ { s , a , x ^ { r } \sim D ^ { r } } \log \pi _ { \phi } ( a | s , f _ { \theta } ( x ^ { r } ) )
$$
- $\mathcal{L}_{\pi}(\phi)$: 代表 policy network 的 optimization target。
- $\phi$: 是 policy network 的 parameters。
- $s, a, x^r$: 分别 indicate 从 unlabelled robot dataset $D^r$ 中 sample 的 state, action， 以及 robot trajectory 的 video。
- $f_{\theta}$: 是 pre-trained 且 frozen 的 video encoder。
- $\pi_{\phi}(a|s, f_{\theta}(x^r))$: 表示 在 given 当前 state $s$ 以及 整个 trajectory 的 video embedding 作为 condition 时， predict action $a$ 的 probability density。 取 negative log-likelihood， 等价于 let policy 去 imitate dataset 里 的 action distribution。

**公式 (2) Supervised Contrastive Loss:**
$$
\mathcal { L } _ { S u p C o n } ( \theta ) = \sum _ { i \in I } \frac { - 1 } { | P ( i ) | } \sum _ { p \in P ( i ) } \log \frac { \exp \big ( \langle f _ { \theta } ( \tilde { x } _ { i } ^ { h } ) , f _ { \theta } ( \tilde { x } _ { p } ^ { h } ) \rangle / \tau \big ) } { \sum _ { a \in A ( i ) } \exp \big ( \langle f _ { \theta } ( \tilde { x } _ { i } ^ { h } ) , f _ { \theta } ( \tilde { x } _ { a } ^ { h } ) \rangle / \tau \big ) }
$$
- $\theta$: 是 projection network 的 parameters。
- $I$: 是 multi-view batch 的 index 集合。
- $A(i)$: represent 除 $i$ 之外 所有 index 的 集合。
- $P(i)$: 是 与 $i$ 同 class 的 positive sample index 集合。
- $\tilde{x}^h$: 表示 经过 random data augmentation 后 的 human video。
- $\langle . , . \rangle$: represent cosine similarity。
- $\tau$: 是 temperature hyperparameter， control softmax distribution 的 sharpness。
这个 loss 的 mechanism 是 在 unit hypersphere 上 pull closer 同类 video 的 embedding， 同时 push away 异类 video 的 embedding， 从而 build 一个 structured 的 semantic space。

### Experiment Data Parsing

**Table I**: 对比 了 ViP 和 baseline DVD 在 TableTop environment 上 的 performance。 我们 observe 到， 在 seen robot demos setting 下， ViP (DVD similarity) 在 env 1 achieve 了 97.1% 的 success rate， 而 DVD 仅为 65.2%。 这 prove 了 用 full trajectory embedding 去 condition policy， 相比 于 DVD 采用 的 sub-trajectory video prediction planning， 表现 更 outstanding 且 stable。

**Table II**: 展示 了 极具 challenging 的 zero-shot setting， 即 train similarity 时 完全 不 use 任何 paired human-robot data。 结果 indicate， ViP (Ours) achieve 了 71.3% 的 average success rate， 大幅 surpass 了 DVD 的 36.8% 和 Random baseline 的 22.8%。 特别 note "Human video as input" 这个 baseline 只有 43.0% 的 success rate， 这 说明 直接 用 human video embedding 去 condition policy 会 lead to 严重 的 out-of-distribution issue。 从而 凸显 了 kNN retrieval translation 步骤 的 necessity， 它 effective 地 bridge 了 domain gap。

**Table III**: 展示 了 Kitchen environment 的 结果。 ViP (Oracle) 和 single-task R3M performance 相当， 说明 policy 结构 本身 的 capacity 足以 handle precise manipulation。 但是 ViP (Ours) 在 sliding door 和 microwave task 上 performance 出现 drop。 原因 在于 SSv2 的 action label (如 "Opening something") granularity 过 粗， 导致 video encoder 无法 fine-grained distinguish 被 manipulate 的 object， 这 是 semantic representation 的 limitation。

### Intuition & Deep Association

这 篇 paper 的 core philosophy 其实 是 robotics field 的 Retrieval-Augmented Generation (RAG)。 我们 possess 一个 unlabelled 的 robot trajectory database (充当 external knowledge base)， 通过 human video 去 retrieve semantic related 的 robot skill embedding， 然后 用 这个 retrieved embedding 去 guide policy generate action。 这种 design 巧妙 地 将 high-dimensional 的 video-to-action mapping problem， decompose 成 video-to-video embedding 的 cross-modal retrieval problem， 以及 embedding-to-action 的 conditional control problem。

同时， 这 与 Decision Transformer (DT) 或 Reinforcement Learning via Supervised Learning (RvS) paradigm 有 异曲同工 之妙。 它们 都 belong to outcome-conditioned action regression， 都 是 将 reinforcement learning 转化 为 supervised learning。 区别 在于， DT condition on return-to-go， 而 ViP condition on video embedding of the full trajectory。 这 意味着 ViP 的 embedding contain 了 丰富 的 task semantics 和 desired outcome， 相当于 一个 极其 powerful 的 goal representation。

如果 要 进一步 improve， 可以 introduce 更 powerful 的 vision-language foundation model， 比如 CLIP 或 VideoCLIP， 用 language 作为 auxiliary supervision， 可能 能够 solve object distinguish 的 fine-grained issue。 此外， 探索 更 dynamic 的 memory mechanism， 比如 用 Transformer attention 替代 kNN， 也 是 一个 有 potential 的 direction。

### Reference Web Links
1. **ViP Paper (arXiv)**: https://arxiv.org/abs/2202.10788
2. **Something-Something-v2 (SSv2) Dataset**: https://developer.qualcomm.com/software/ai-datasets/something-something
3. **Supervised Contrastive Learning Paper**: https://arxiv.org/abs/2004.11362
4. **DVD (Domain-agnostic Video Discriminator) Paper**: https://arxiv.org/abs/2103.16817
5. **R3M Representation Paper**: https://arxiv.org/abs/2203.12601
6. **Decision Transformer Paper**: https://arxiv.org/abs/2106.01345
