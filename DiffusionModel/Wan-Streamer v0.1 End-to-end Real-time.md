---
source_pdf: Wan-Streamer v0.1 End-to-end Real-time.pdf
paper_sha256: 921ef5c7c4ce650c9da9c5c489ef946c01ad83eb12f110cf3a82af0c5cfde2cb
processed_at: '2026-08-13T03:35:47-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 Wan-Streamer

## 1. 先想一个生活场景

你跟朋友打 FaceTime。你说话的时候，朋友在点头；你突然停顿，朋友眉毛抬一下等你继续；你打断他，他立刻闭嘴；他说话时你皱眉，他看到你皱眉就改口。这一切发生在 **几百毫秒** 内，**视觉和声音是同步的**，**没有谁先听完整句话再开始反应**。

现在让 LLM 来做这件事。市面上大多数"AI 数字人"系统是这样的：

```
你的声音 → VAD 判断你说完了 → ASR 转文字 → LLM 想回答 → TTS 合成语音 → 驱动 avatar 嘴型 → 渲染视频 → 你看到
```

这条链路里每一环都要排队等上一环。ASR 等你说完一段才出文字，LLM 等文字才想回答，TTS 等 LLM 出文字才合成，avatar 等 TTS 出语音才动嘴。**每一级都有 buffer、都有延迟、都可能在错的基础上继续错**。ASR 把"我要苹果"听成"我要平板"，后面 LLM、TTS、avatar 全都按"平板"跑，错位滚雪球。

更糟的是，这套系统 **只会"轮流说话"**。你说话时它死机般地等你结束，它说话时你插话它根本听不见。你皱眉它没反应，你点头它不知道。它像一个对着提词器念稿的播音员，**不是一个真在场的人**。

Wan-Streamer 要解决的就是这件事：**让 AI 跟你视频聊天时，像一个真在场的人**。

---

## 2. 它到底干了什么

一句话：**把听、看、想、说、做表情，全部塞进一个模型，让它一边接收一边反应，像人一样**。

具体怎么塞？把所有东西排成一条时间线：

```
[时间轴 →]
user_text | user_audio | user_video | agent_text | agent_audio | agent_video | user_text | ...
```

这条时间线上 **每个 160 ms 是一格**，所有 modality 在这一格里同时存在。模型每走一格：

- 看到当前 160 ms 的用户音视频
- 想一下现在该说什么
- 生成自己接下来 160 ms 的语音 latent 和视频 latent
- 把生成的 latent **塞回历史**，成为下一格的"已经发生的事"

这就是论文里公式 (1) 的意思：

$$p_\theta(y_{1:K} \mid u_{1:K}) = \prod_{k=1}^{K} p_\theta\!\left(y_k^{\mathrm{t}}, y_k^{\mathrm{a}}, y_k^{\mathrm{v}} \mid u_{\le k}^{\mathrm{t}}, u_{\le k}^{\mathrm{a}}, u_{\le k}^{\mathrm{v}}, y_{<k}^{\mathrm{t}}, y_{<k}^{\mathrm{a}}, y_{<k}^{\mathrm{v}}\right)$$

翻译成人话：**第 k 格的 agent 行为，由"所有已经发生的 user 帧 + 所有已经发生的 agent 帧"决定**。注意条件里 user 用 $\le k$（user 的当前帧已经到达），agent 用 $<k$（agent 当前帧还在生成）。这种不对称恰恰是 **full-duplex 的形式化**——你说话的时候我也在听，我说话的时候你也还在说，只是我的"当前帧"还没生成完，所以不能拿自己当条件。

---

## 3. 为什么 audio 和 video 不用离散 token

如果你做过 LLM，第一反应可能是"全部 tokenize，next-token prediction 完事"。Wan-Streamer 在 text 上确实这么做，cross-entropy loss 一把梭。但 audio 和 video 它走了 **flow matching**，为什么？

因为 **离散化太费细节**。25 fps、160 ms 一个 chunk 的视频，嘴唇动作、微表情、呼吸声这些细节如果走 VQ-VAE 离散 codebook，codebook 一大就训不动，一小就糊。所以 Wan-Streamer 把 audio/video 都先经过一个 **causal VAE** 编到连续 latent 空间，然后用 flow matching 在 latent 上 denoise。

公式 (2) 是构造 noisy latent：

$$z_\tau^{m} = (1-\tau) z_0^{m} + \tau \epsilon^{m}$$

- $z_0^{m}$ 是 clean latent（m 取 a 或 v）
- $\epsilon^{m}$ 是高斯噪声
- $\tau$ 是 flow time，0 是干净，1 是纯噪声
- 这是 Rectified Flow 的标准写法，就是 noise 和 clean 之间的直线插值

公式 (3) 是训练目标：

$$\mathcal{L}_{\mathrm{FM}}^{m} = \mathbb{E}_{\epsilon^{m}} \left\| f_\theta(z_\tau^{\mathrm{a}}, z_\tau^{\mathrm{v}}, c_k, \tau) - (\epsilon^{m} - z_0^{m}) \right\|_2^2$$

关键点：**$f_\theta$ 同时接收 audio 的 noisy latent 和 video 的 noisy latent**。这意味着在 denoiser 内部，speech 和 motion 已经互相看到了。唇音同步是 **在生成时就耦合的**，不是事后再拿 speech 去驱动嘴型。这跟 VASA-1 那种"audio 驱动 face"的思路根本不同——VASA-1 是 audio 先有，再渲染 face；Wan-Streamer 是 audio 和 face **同时从噪声里长出来**，一起被 denoise。

$c_k$ 是 clean streaming context，也就是所有已经 commit 到 history 的 user 和 agent 帧。**因为 context 是 clean 的，模型生成当前帧时能看到历史全部细节**——这是 long-horizon identity preservation 的关键，agent 不会聊十分钟之后脸变样。

---

## 4. "全栈因果"到底在说什么

论文反复念"causal"，这词在这里非常具体：

**Encoder 必须因果**：普通 VAE encoder 看 future frame 来决定 current frame 的编码，流式时你拿不到 future，所以必须改成只用 past + current。论文说他们搞了 strictly causal audio VAE 和 strictly causal video VAE。

**Decoder 必须因果**：解码 latent 成 waveform 或 frame 时，不能依赖未来 latent，每 160 ms 必须能独立解码出来发给用户。

**Attention 必须 block-causal**：一个 160 ms unit 内部的 token 可以互相 attend（block 内双向，让帧内可以做唇音对齐），但跨 unit 严格只能看过去（block 间因果，保证 train/inference 一致）。

**生成的 latent 必须回填**：生成的 clean latent 直接 append 到 history，成为下一 unit 的 context。这是 **closed-loop**——模型自己生成的行为变成它未来的输入，跟 agent 进世界交互一个道理。

这一套下来，**streamability 就成了建模约束，而不是部署时再想办法打补丁**。这跟很多系统"先训个 offline 模型再改流式"的做法完全相反。

---

## 5. 训练三阶段，讲人话

**Stage 1：打底子**。拿 Qwen3 或 Qwen2.5 当底座（[Qwen3 技术报告](https://arxiv.org/abs/2505.09388)），喂一大堆 image understanding、video understanding、ASR、TTS、image generation、video generation、joint audio-visual generation 的混合数据。让模型先学会"看图、听音、说话、生成视频"这些原子能力。

**Stage 2：学真人对聊**。喂真双工对话数据，user 和 agent 的 text/audio/video 在同一条 causal 时间线上交错。模型从数据里学到：
- 什么时候该说话，什么时候该点头
- 用户插话时怎么停下来
- 一段长对话里怎么保持身份和场景一致

这些行为 **没有显式规则**，是从数据里涌现的。这跟 LLM 不显式做 NER 但 NER 能力涌现是一个味道。

**Stage 3：压延迟**。teacher 模型用 classifier-free guidance（CFG）+ 多步 flow matching solver，质量高但慢。通过 distillation 把 CFG 的效果 bake 进 student 权重，并把 solver 步数大幅减少。再加 **rolling distillation**——student 用自己生成的 history 滚动训练，配 distribution matching（[DMD](https://arxiv.org/abs/2311.18828)、[DMD2](https://arxiv.org/abs/2405.14867)）对齐 teacher。这一步解决两个问题：步数多导致延迟大，以及 student 训练时见的是 teacher 的 clean history、inference 时见的是自己的 noisy history 之间的 mismatch。

---

## 6. Thinker-Performer：一个模型拆两半跑

这部分是工程上最聪明的设计。模型本身是 end-to-end 一个，但 serving 时拆成两块 GPU：

- **Thinker**：跑 causal encoder、跑 token-causal Transformer（语言预测 + state update + 建 KV cache）、跑 causal decoder
- **Performer**：只跑 flow-matching solver，生成下一个 audio-video latent

每一步 k 的时间线：

```
Thinker：                       | Performer：
1. 收到 user 160ms 音视频        |
2. encoder + Transformer        |
3. 建 KV cache slice           |
4. 同时收到 performer 上一步的   |
   clean latent (y_{k-1})      |
5. 把 KV slice 发给 performer   | ← append KV，跑 flow matching
6. decode y_{k-1} → 输出给用户  |   生成 y_k latent 留在 performer
                                |
下一轮 k+1 开始 →               |
```

关键：**两个 GPU 同时干活**。Thinker 在解码上一帧，Performer 在生成下一帧，重叠起来。Throughput 瓶颈在 performer wall time，只要 performer 时间 + 通信开销 < 160 ms，就能实时跑。

**但 model-side latency 是另一回事**。它指 signal-to-signal 路径：从 user 信号到达到 agent 信号输出，要经过 encode → thinker state update → performer latent generation → decode，论文测约 **200 ms**。这是首字节延迟，跟 throughput 是两个概念——throughput 跟得上不代表响应快。

再加 350 ms 网络往返（远程用户），**total interaction latency 约 550 ms**。这是 sub-second 双工音视频通信。

工程上还用了 CUDA graph capture、kernel fusion、bf16/fp8 量化、KV cache 传输优化。CUDA graph 在 fixed-shape streaming step 上特别好用，因为每步 shape 都一样，可以整段 graph replay 省掉 kernel launch overhead。

---

## 7. 数字怎么读

Table 1 我帮你翻译成人话：

- **GPT-4o Realtime API**：音频响应 232/320 ms，API TTFB ~500 ms，但 **没有视频输出**
- **Moshi**：理论 160 ms，实际 200 ms model latency，但 **没有视频**
- **Hume EVI 3**：model response <300 ms，但 **没有视频**
- **Doubao Realtime Voice**：~700 ms bare model，纯语音
- **Qwen3-Omni / MiniCPM-o**：first-packet 234-651 ms，**没有同步 avatar 输出**
- **Wan-Streamer**：~200 ms model-side，~550 ms total，**25 FPS 视频输出**

所以 Wan-Streamer 的 550 ms 是 **买到了视频** 的价格。拿它跟纯语音系统比延迟不公平，纯语音系统要加一个 avatar renderer 才能做同样的事，那个 renderer 又依赖上游 stack，整体延迟反而看不见。

Table 2 那堆 visual agent 系统：
- **VASA-1**：40 FPS、170 ms 前置延迟，但只是 renderer，**没有对话推理**
- **StreamAvatar**：FFD 0.33-0.39 s，video latency ~1.20 s，**没有统一对话模型**
- **LiveTalk**：24.82 FPS，0.33 s first-frame，**用 Qwen3-Omni 做推理，video 延迟另算**
- **Hallo-Live**：20.38 FPS，0.94 s latency，**text-driven，不持续感知用户**

Wan-Streamer 是 **唯一把 perception + reasoning + speech + video + turn-taking 全压在一个 causal Transformer 里** 的，所以它的 550 ms 才是真正端到端可比的数字。

---

## 8. 自然性、打断、主动说话怎么来的

论文描述了几个 qualitative 行为：

- **Idle state**：agent 不冻结成肖像，保持 gaze、呼吸、微表情
- **Listening state**：用户说话时，agent 点头、目光跟随、姿势变化
- **Speaking state**：唇音同步在 denoise 前就耦合好
- **Interruption**：用户插话时 agent 能停、缩短、改口
- **Proactive speaking**：桌上出现新物体，agent 主动评论

这些 **没有一个是显式规则**。它们是 Stage 2 训练里从 interleaved interaction data 学出来的。本质上，**对话协议本身被当成了 in-context learning 的对象**，而不是 hardcoded 的 state machine。

这跟 LLM 时代一个核心洞察一脉相承：**你不需要给模型写"什么时候该做 NER"，喂足够多数据，NER 就涌现**；同样，**你不需要给模型写"什么时候该让话"，喂足够多双工对话数据，turn-taking 就涌现**。

---

## 9. 局限和我的联想

论文老实说 v0.1 只在 **192p** 验证，scaling 到更高分辨率"straightforward"。我几个直觉疑问：

**192p 够不够？** 高分辨率下 causal VAE 的 receptive field 是否够，是否必须引入 lookahead 才能保质量，这是 causal video VAE 的经典难题。Wan 2.1 原 VAE 是 bidirectional 的，改成 strictly causal 损失多少 PSNR，论文没给数字。

**Long-horizon drift**：rolling distillation 缓解了 exposure bias，但 session 跑几个小时后 identity 是否还稳？DMD 一步 distillation 在长视频上稳定性我个人持保留态度，[OmniForcing](https://arxiv.org/abs/2603.11647) 也报告了类似挑战。

**Block-causal 的 block 边界**：160 ms 内部双向，但用户在 block 中途插话怎么办？是否需要 sub-block causal，或者 block 内部也要 mask 掉 user→agent 的方向？这个论文没讲。

**KV cache 膨胀**：thinker → performer 每步传一个 KV slice，长 session 后 KV cache 会很大。是否需要 cache 压缩、sparsification、或者 chunk-level cache 回收？现在主流 LLM serving 都在做 PagedAttention、sliding window attention，Wan-Streamer 的双 GPU 架构里这些技术能不能直接套用，值得思考。

**Action stream 缺席**：[DuplexSLA](https://arxiv.org/abs/2605.20755) 在 full-duplex speech model 上加了 synchronized action stream 做 tool use，Wan-Streamer 当前没有 action channel。如果未来要做 embodied agent（机器人），这必须加。阿里自己的 [Body of Her](https://arxiv.org/abs/2408.02879) 已经探索过 humanoid agent 端到端，next frame within 42 ms @ 24 FPS，那个数字激进得多——但 Body of Her 不聚焦对话、不报 production latency。Wan-Streamer 是 Body of Her 的"对话特化版 + production serving 版"。

**DiT 架构细节缺失**：论文没展开 unified diffusion transformer 的具体结构。我猜是 MM-DiT（参考 [Wan 2.1](https://arxiv.org/abs/2503.20314)）加 causal mask，但 audio latent 和 video latent 在 attention 里是 joint attention 还是 cross attention，文本 token 和 latent token 怎么混，block-causal mask 怎么具体实现，这些都没写。复现上是大坑。

**数据配比黑盒**：Stage 1 understanding 数据 + Stage 2 interaction 数据的过渡怎么控制？太多 understanding 数据让模型变 turn-based，太多 interaction 数据会 lack grounding。这个 trade-off 我猜是实验堆出来的，没公式可循。

**多用户场景**：当前 formulation 假设一对一。多人会议需要扩展 user stream 成多个并行 stream，attention pattern 会复杂很多。这是 [Gemini Live](https://deepmind.google/technologies/gemini/live/) 这类商业系统已经在推的方向。

---

## 10. 跟 nanoGPT 哲学的对照

你训 nanoGPT 的核心信条是 **"tokenize 一切，train GPT，see what emerges"**。Wan-Streamer 在哲学上就是这个信条推到极致：

- Tokenize audio → causal VAE latent
- Tokenize video → causal VAE latent  
- 把 text token + audio latent + video latent 拼成 single causal stream
- Train one Transformer with mixed objectives（CE for text，flow matching for AV）
- See turn-taking / interruption / proactive speaking emerge

这就是 **GPT 在 multimodal streaming 上的最终形态**。跟 OpenAI Realtime API、Moshi 的方向一脉相承，但 Wan-Streamer 多了 video 通道——而 video 通道是真正让 agent "有身体" 的关键。没有 video，你永远只是电话里的声音；有 video，agent 才"在场"。

我个人觉得这篇 paper 最大的价值，是它 **诚实地把 production latency 写成 200/550 ms 这种工程师能对标的数字**，同时 **明确指出 cascaded pipeline 不可救药的根本原因是 streamability 必须是建模约束**。这两点合起来，给后续 real-time agent 的 serving 栈指了一条路：长得很像 nanoGPT + DiT + 分布式 KV cache，而不是 ASR + LLM + TTS + renderer 的全家桶。

---

## 11. 参考链接

核心论文：
- [Wan-Streamer 项目主页](https://wan-streamer.com/)
- [Wan 2.1 技术报告](https://arxiv.org/abs/2503.20314)
- [Moshi](https://arxiv.org/abs/2410.00037)
- [Body of Her](https://arxiv.org/abs/2408.02879)
- [Diffusion Forcing](https://arxiv.org/abs/2407.01392)
- [Self-Forcing](https://arxiv.org/abs/2506.08009)
- [Rolling Forcing](https://arxiv.org/abs/2509.25161)
- [DMD](https://arxiv.org/abs/2311.18828)
- [DMD2](https://arxiv.org/abs/2405.14867)

对比系统：
- [VASA-1](https://arxiv.org/abs/2404.10667)
- [OmniHuman-1](https://arxiv.org/abs/2502.01061)
- [Kling-Avatar](https://arxiv.org/abs/2509.09595)
- [OmniForcing](https://arxiv.org/abs/2603.11647)
- [StreamAvatar](https://arxiv.org/abs/2512.22065)
- [LiveTalk](https://arxiv.org/abs/2512.23576)
- [Hallo-Live](https://arxiv.org/abs/2604.23632)
- [TalkingMachines](https://arxiv.org/abs/2506.03099)
- [MIDAS](https://arxiv.org/abs/2508.19320)
- [X-Streamer](https://arxiv.org/abs/2509.21574)
- [LPM 1.0](https://arxiv.org/abs/2604.07823)
- [MAViD](https://arxiv.org/abs/2512.03034)
- [M.I.O](https://arxiv.org/abs/2512.13674)
- [U-Mind](https://arxiv.org/abs/2604.27393)

Full-duplex speech：
- [DuplexSLA](https://arxiv.org/abs/2605.20755)
- [SALM-Duplex](https://arxiv.org/abs/2505.15670)
- [OmniFlatten](https://arxiv.org/abs/2410.17799)
- [Seeduplex (ByteDance)](https://seed.bytedance.com/en/realtime_voice)

商业系统：
- [GPT-4o](https://openai.com/index/hello-gpt-4o/)
- [Realtime API missing manual](https://www.latent.space/p/realtime-api)
- [Hume EVI 3](https://www.hume.ai/blog/introducing-evi-3)
- [Doubao Realtime Voice](https://seed.bytedance.com/en/realtime_voice)
- [Qwen3-Omni](https://arxiv.org/abs/2509.17765)
- [Qwen3.5-Omni](https://arxiv.org/abs/2604.15804)
- [MiniCPM-o 4.5](https://arxiv.org/abs/2604.27393)

底座 LLM：
- [Qwen3](https://arxiv.org/abs/2505.09388)
- [Qwen2.5](https://arxiv.org/abs/2412.15115)

---

## 12. 最最后一句人话

Wan-Streamer 干的事，就是 **把"跟 AI 视频聊天"这件事，从"五个模块接力跑"改成了"一个模型在一条时间线上同时听说看做"**。代价是模型设计要从头改成 causal，训练要分三阶段，serving 要拆 thinker-performer 双 GPU。换来的，是 200 ms 模型延迟、550 ms 端到端延迟、25 FPS 同步音视频，以及一个能学会"什么时候该闭嘴什么时候该说话"的 agent。

这跟 LLM 之于 NLP 的关系完全一样：**没有 NER 模块、没有 parser、没有 coreference resolver，所有任务都涌现出来**；**没有 VAD、没有 turn-taking logic、没有 lip-sync repair，所有行为都涌现出来**。nanoGPT 的哲学，在 multimodal streaming 上开花了。

---

# Wan-Streamer v0.1 深度拆解：一个 native-streaming 全双工 multimodal foundation model 的工程哲学

## 1. 一句话直觉与定位

Wan-Streamer 把 **language、audio、video 同时作为 input 和 output**，全部塞进 **single causal Transformer**，用 **block-causal attention** 协调 incremental streaming，靠 **strictly causal VAE + causal encoder/decoder + flow-matching latent generation** 把 pipeline 拍平成一个 native-streaming 的 end-to-end 模型。它直面的核心矛盾是：**real-time audio-visual interaction 不是一个 "理解模块" 与 "生成模块" 的并集，而是一个 intrinsically full-duplex 的因果过程**。系统设计上，论文给出两个标志性数字：**~200 ms model-side latency**，**~550 ms total latency（含 350 ms 网络）**，对应 **160 ms streaming unit @ 25 fps**。

这与我之前给 nanoGPT 等讲过的"模型就是一切"的极端想法在哲学上非常接近——**streamability 是建模约束而非 serving 优化**。

---

## 2. 为什么 cascaded pipeline 不可救药（背景动机）

论文 Section 1 给了 cascaded 系统的"四宗罪"：

1. **module-boundary 等待**：VAD → ASR → LLM → TTS → audio-driven animation → video generation 每一级都有 buffer/queue，端到端延迟累加。
2. **error accumulation**：ASR 错一个词，LLM 拿错的上下文生成回复，TTS 又把错情感读出来，avatar 按错的情感驱动——错位滚雪球。
3. **post-hoc audio-visual alignment**：speech 已经合成完，再去驱动 lip-sync，本质上是修复而不是耦合，永远修不到无缝。
4. **turn-taking 难以学习**：双工行为（如 user 打断 agent、agent 在 user 说话时点头）需要写在 system logic 里，而不能从 interaction 数据里学出来。

Wan-Streamer 的破局点是 **single streaming contract**：

> Every component must operate causally, every newly observed unit must be usable immediately, and every generated unit must be emitted and committed back into the interaction history.

这句话决定了一切架构选择：encoder/decoder 必须因果、attention 必须 block-causal、生成的 latent 必须回填进历史成为下一个 unit 的 context。这是一个比"在线 inference"严格得多的约束——它要求 **train 和 inference 用同一个 causal 格式**，否则 train-test mismatch 会毁掉长程一致性。

---

## 3. 形式化建模（公式 1 解读）

论文把第 k 个 streaming unit 定义为：

$$u_k = (u_k^{\mathrm{t}}, u_k^{\mathrm{a}}, u_k^{\mathrm{v}}), \quad y_k = (y_k^{\mathrm{t}}, y_k^{\mathrm{a}}, y_k^{\mathrm{v}})$$

- $u$ = user observation（t/a/v 分别表示 text/audio/video）  
- $y$ = agent response  
- $k$ = streaming unit index（一个 unit 对应 160 ms 音视频块）  

**公式 (1)** 给出 full-history autoregressive 分解：

$$p_\theta(y_{1:K} \mid u_{1:K}) = \prod_{k=1}^{K} p_\theta\!\left(y_k^{\mathrm{t}}, y_k^{\mathrm{a}}, y_k^{\mathrm{v}} \mid u_{\le k}^{\mathrm{t}}, u_{\le k}^{\mathrm{a}}, u_{\le k}^{\mathrm{v}}, y_{<k}^{\mathrm{t}}, y_{<k}^{\mathrm{a}}, y_{<k}^{\mathrm{v}}\right)$$

直觉：

- 上标 $\mathrm{t/a/v}$ 是 modality 维度，下标 $k$ 是时间维度。
- $\le k$ 与 $< k$ 的不对称很关键：**user 的当前帧已经到达可以参与推理**，但 **agent 的当前帧还在生成中**——所以条件只放 $y_{<k}$。
- 这正是 full-duplex 的形式化：**user 的 stream 与 agent 的 stream 在时间上错开一个 unit**，但共用同一份 causal history。
- $K$ 个 unit 拼成一个 session，**history 是不断增长的上下文，不是对话回合列表**。

这个分解等价于一个 **infinite-context causal LM**，其中 token 类型跨 modalities。和 Moshi 的双流 formulation 比较：Moshi 只有 speech-text，Wan-Streamer 在 token 通道上多加了 video 与 text-input。

---

## 4. Audio/Video 的生成：Flow Matching 而非离散 token

### 4.1 为什么 audio/video 走 latent flow matching，text 走 discrete token？

- Text 是天然离散，next-token prediction + cross-entropy 是 SOTA。
- Audio/video 在 25 fps、160 ms unit 的情况下，**离散化损失太大**：VQ-VAE 类的 codebook 通常会丢掉 fine-grained 嘴唇细节、breath sound、微表情。flow matching 在连续 latent 上跑 solver 反而保留细节。
- 同时 flow matching 可被 distillation 到很少步（论文提到 distillation 后 solver steps 大幅减少），延迟可控。

### 4.2 公式 (2)：noisy latent 构造

$$z_\tau^{m} = (1-\tau)\, z_0^{m} + \tau\,\epsilon^{m}, \qquad \frac{\partial z_\tau^{m}}{\partial \tau} = \epsilon^{m} - z_0^{m}$$

变量：

- $m \in \{\mathrm{a}, \mathrm{v}\}$ 表示 modality。
- $z_0^{m}$ 是 clean target latent（VAE 编码后的 ground-truth latent）。
- $\epsilon^{m} \sim \mathcal{N}(0, I)$ 是 Gaussian noise。
- $\tau$ 是 flow time（flow matching 的伪时间），$\tau \in [0, 1]$：$\tau = 0$ 是 clean，$\tau = 1$ 是纯噪声。
- 第二式是 **conditional vector field**：从 noise 到 clean 的直线轨迹（optimal transport flow matching）。

注意这里选择 **linear interpolant** $(1-\tau) z_0 + \tau \epsilon$，等价于 Rectified Flow / Stochastic Interpolant 中的 $x_t = (1-t)x_0 + t x_1$ 写法（这里把 clean 放在 0 端、noise 放在 1 端，方向上和 Stable Diffusion 3 一致）。

### 4.3 公式 (3)：联合 flow matching loss

$$\mathcal{L}_{\mathrm{FM}}^{m} = \mathbb{E}_{\epsilon^{m}} \left\| f_\theta(z_\tau^{\mathrm{a}}, z_\tau^{\mathrm{v}}, c_k, \tau) - \frac{\partial z_\tau^{m}}{\partial \tau} \right\|_2^2$$

- $f_\theta$ 是 unified diffusion transformer，**audio 与 video 的 noisy latent 同时进入**，所以 speech 和 motion 在 denoiser 内部就已经耦合，避免了事后 align。
- $c_k = \{u_{\le k}^{\mathrm{t}}, u_{\le k}^{\mathrm{a}}, u_{\le k}^{\mathrm{v}}, y_{<k}^{\mathrm{t}}, y_{<k}^{\mathrm{a}}, y_{<k}^{\mathrm{v}}\}$ 是 clean streaming context——所有已到达的 user 帧和所有已 commit 的 agent 帧。
- $\tau$ 作为 noise level 条件注入（通常通过 AdaLN 或 FiLM）。
- 这个 loss 的精髓：**条件 $c_k$ 是 clean 的，所以模型在生成当前 noisy 帧时，能看到 history 全部细节**——这是 long-horizon identity preservation 的关键。已生成的 latent **回到 history 作为 clean context**，下一个 unit 拿它当条件，因此不会漂移。

这跟 **Diffusion Forcing**（[Boyuan Chen et al., 2024](https://arxiv.org/abs/2407.01392)）思路相通：把 next-token prediction 与 full-sequence diffusion 融合，clean frames 作为条件，noisy frames 作为生成目标。Wan-Streamer 把它从"视频生成"扩展到了"音视频对话生成"。

---

## 5. 全栈 causal 设计

论文反复强调"全栈因果"，这一段是工程上最硬核的部分：

| 组件 | 设计要点 | 为什么必须因果 |
|---|---|---|
| **Audio VAE** | strictly causal encoder/decoder，无 lookahead | 流式 audio 不能等未来帧 |
| **Video VAE** | strictly causal，3D 卷积只用过去+当前帧 | 流式 video 同理 |
| **Audio-Visual Encoders** | causal 适配层，将 VAE latent 喂进 Transformer | 保证 attention 可见性严格因果 |
| **Audio/Video Decoders** | causal 解码 latent → waveform / frames | 输出 unit 必须可独立解码 |
| **Transformer** | block-causal attention | block 内可双向，block 间因果 |
| **History commit** | 生成完的 latent 直接回填为 clean context | 维持 closed-loop 一致性 |

### 5.1 Block-causal attention 是什么？

我的理解（论文没明写细节，但这是当下 streaming DiT 通用做法）：

- 一个 **streaming unit（160 ms）内部的所有 token（text/audio/video 的 latent chunk）允许互相 attend**——这就是 block 内双向。
- **跨 unit，unit $i$ 可以 attend unit $j$ 当且仅当 $j \le i$**——这是 causal。
- 这样设计的好处：**一个 unit 内部能做 "帧内协同对齐"**（比如唇音同步在 160 ms 内消化），**跨 unit 严格因果**保证 train/inference 一致。

类似设计在 **OmniForcing**、**Self-Forcing**（[arXiv 2506.08009](https://arxiv.org/abs/2506.08009)）里都出现过。

---

## 6. 训练三阶段（Section 2.3）

| Stage | 任务 | 目的 |
|---|---|---|
| **Stage 1: Independent-task pretraining** | 用 image/audio/video understanding + text dialogue + ASR + TTS + audio dialogue + image/audio/video/joint AV generation 的混合数据 | 给 unified Transformer 打底能力，alignment 各 modality 接口。初始化自 Qwen3/Qwen2.5（[Qwen3 technical report](https://arxiv.org/abs/2505.09388)）。 |
| **Stage 2: End-to-end interaction training** | 在 duplex interaction data 上训练，user 和 agent 的 t/a/v 在同一 causal stream 上交错 | 学会 response timing、active listening、interruption、长程一致性——这些都是结构化涌现，而非 rule 写死。 |
| **Stage 3: Distillation for low-latency** | (a) CFG distillation：teacher 用 CFG + 多步 solver，student 吸收 CFG、减步数；(b) Rolling distillation：student 用自己生成的 history 滚动 rollout，加 **distribution matching**（DMD [Yin et al. 2023](https://arxiv.org/abs/2311.18828) / DMD2 [Yin et al. 2024](https://arxiv.org/abs/2405.14867)）对齐 teacher | 解决 inference 步数爆炸 + train-test mismatch，这是 self-forcing 思想在 streaming 上的延展。 |

**关键直觉**：Stage 2 才是 Wan-Streamer 真正区别于 Qwen-Omni、MiniCPM-o 的地方。前面那些模型虽然"接受 audio/video 输入，输出 speech/text"，但仍然 turn-based、缺乏 video 输出、turn-taking 用 VAD 写死。Stage 2 把"何时该说"也变成学习目标。

**Rolling distillation 的关键点**：student 在自己 rollout 出来的 latent 上继续训练，类似 **scheduled sampling** 或 **DAgger**，避免 exposure bias。Distribution matching 提供的 gradient 信号比简单 MSE 强，能逼 student 的 trajectory 分布贴合 teacher。

---

## 7. Thinker-Performer 推理架构（Section 2.4 + Figure 2）

这是这篇论文工程上最精彩的一节。模型本身是一个 end-to-end 模型，但 serving 时拆成两个 GPU 角色：

```
┌────────────────── Thinker (GPU A) ──────────────────┐
│ • causal audio/video encoders                       │
│ • short token-causal Transformer pass (language +    │
│   state update + KV cache slice construction)       │
│ • causal audio/video decoders                        │
└─────────────────────────────────────────────────────┘
              ▲ KV slice out    │ latent in (prev step)
              │                 ▼
┌────────────────── Performer (GPU B) ────────────────┐
│ • flow-matching solver only (next AV latent unit)   │
│ • 保留 KV cache，append 接收的 slice                │
└─────────────────────────────────────────────────────┘
```

### 7.1 单个 streaming step k 的时间线

1. Thinker 收到 $u_k$（160 ms 用户音视频）。
2. Thinker 跑 causal encoders + token-causal Transformer → 产生新的 **KV-cache slice**（语言预测 + state update）。
3. Thinker 同时收到 performer 上一步生成的 $y_{k-1}^{\mathrm{a}}, y_{k-1}^{\mathrm{v}}$ 的 clean latents。
4. Thinker 把新 KV slice 发给 performer，并 **decode** $y_{k-1}$ → 输出音频视频帧给用户。
5. Performer 把 KV slice append 进自己的 cache，**只跑 flow-matching solver**，生成 $y_k$ 的 audio+video latent，留在 performer 端。
6. 下一步 k+1 时回到第 3 步循环。

### 7.2 关键洞察

- **KV cache exchange 是统一状态的"传递介质"**。模型在数学上仍是一个 unified model，只是物理上分布在两块 GPU，KV cache 通过高速互联传过去。
- **Pipeline overlap**：thinker 在解码前一帧（4）、performer 在生成下一帧（5），两者并行。Throughput 的瓶颈是 performer 的 wall time，**只要 performer 时间 + 通信开销 < 160 ms**，系统就能实时。
- **Model-side latency** 是另一回事——指 **signal-to-signal** 路径：encode → thinker state update → performer latent generation → decode。论文测出约 200 ms，这意味着即便 throughput 跟得上，**首字节延迟仍有 200 ms**。Throughput vs latency 的区分在这里讲得非常清楚。

### 7.3 工程优化

论文提到 CUDA graph capture、kernel fusion、compilation、KV-cache exchange 优化。这些是生产级 streaming 推理的标配——CUDA graph 在 fixed-shape streaming step 上特别好用，因为每步 tensor shape 相同，可以整段 graph replay 避开 launch overhead。

---

## 8. 实验数据表解读

### Table 1：Speech / omni-modal 对话系统延迟对比

我把它整理成更可读的形式：

| 系统 | 交互类型 | User-visible latency | 其他指标 | 关键差异 |
|---|---|---|---|---|
| **Doubao Realtime Voice** | speech-to-speech | ~1 s overall | ~700 ms bare-model | 仅语音，无视觉 |
| **Seeduplex** | speech-to-speech | N/R | -250 ms endpoint vs 上代 | 全双工但纯语音 |
| **GPT-4o Realtime API** | speech + vision in, speech out | protocol-dependent | 232/320 ms 音频响应，~500 ms API TTFB | 混合多种延迟 |
| **Hume EVI 3** | speech-to-speech | 0.9–1.4 s web | <300 ms model | 无视觉输出 |
| **Gemini Live API** | speech-to-speech | 1.2–3.6 s | N/R | 无视觉 |
| **Sesame web app** | speech-to-speech | 0.8–1.2 s | N/R | 纯语音 |
| **Moshi** | speech-to-speech | N/R | 160 ms 理论 / 200 ms 实际 | 全双工鼻祖但无视觉 |
| **Qwen3/3.5-Omni** | AVT in, speech/text out | N/R | first-packet 234/547 ms；3.5 Flash 235/426 ms | first-packet，无同步 avatar |
| **MiniCPM-o 4.5** | AV in, speech/text out | N/R | first-token 0.58 s；RTF 0.20–0.27 | 无视觉 avatar |
| **Wan-Streamer** | text/audio/video in/out | ~550 ms total (含 350 ms 网络) | ~200 ms model-side；25 FPS | 全栈 end-to-end |

**关键读法**：论文明确警告"raw speed 单独看不公平"——

- 纯语音系统可以报极低 model latency，但没视觉响应；
- Avatar 渲染器可以跑 20-40 FPS，但依赖上游对话/语音模块，其延迟被隐藏；
- Wan-Streamer 的 550 ms **包含了完整 audio-visual response path**。

### Table 2：Visual agent / avatar 系统运行时对比

| 系统 | 覆盖范围 | 运行时指标 | 与 Wan-Streamer 的主要差异 |
|---|---|---|---|
| **Body of Her** | end-to-end humanoid agent | next frame within 42 ms @ 24 FPS | 概念验证，无 deployed signal-to-signal latency |
| **MIDAS** | 多模态数字人视频合成 | real-time frame-by-frame | 未披露绝对响应延迟 |
| **U-Mind** | text/speech/motion/video loop | real-time 渲染 | text-first pipeline，未公开延迟分解 |
| **X-Streamer** | 开放视频对话 | 25 FPS on 2×A100 | 未公开绝对响应延迟 |
| **LPM 1.0** | 在线角色性能引擎 | low-latency causal streaming | 引擎耦合外部 A2A |
| **MAViD** | audio-visual 对话框架 | 未披露 | 模块化框架，主要用于能力对比 |
| **M.I.O** | 交互式 omni-avatar | bounded-latency 设计 | 多模块系统，无公开信号延迟 |
| **VASA-1** | audio-driven talking face | 40 FPS, 170 ms preceding latency | 渲染器，无对话推理或视觉感知 |
| **TalkingMachines** | FaceTime 风格 audio-driven video | real-time chunk via TTBC | 依赖外部 audio LLM |
| **StreamAvatar** | streaming talking/listening avatar | FFD 0.33–0.39 s；video latency ~1.20 s | avatar renderer，无统一对话模型 |
| **AvatarForcing (Ki)** | 交互式头部 avatar | ~500 ms reaction；6.8× speedup | 不生成对话语音 |
| **AvatarForcing (Cui)** | one-step streaming talking avatar | 34 ms/frame；0.51 s audio-to-visual | 不是感知对话 |
| **LiveTalk** | 多模态交互 avatar 视频 | 24.82 FPS；0.33 s first-frame | 用 Qwen3-Omni 推理，video 延迟独立 |
| **Hallo-Live** | text-driven joint AV avatar | 20.38 FPS, 0.94 s latency | text-driven，不持续感知用户音视频 |
| **OmniForcing** | text-to-AV streaming generation | TTFC ~0.7 s；~25 FPS | first-chunk latency，非用户响应 |
| **Wan-Streamer** | 完整感知对话 + 同步 speech + video | 25 FPS；~550 ms total；~200 ms model-side | 单一 causal Transformer 学会所有 |

**读这张表的直觉**：现有 visual agent 系统大致分三类——

1. **Full-loop digital human**（如 Body of Her, MIDAS, X-Streamer）：覆盖广，但 latency 未标准化披露。
2. **Avatar renderer**（VASA-1, StreamAvatar, AvatarForcing, LiveTalk）：渲染极快但依赖上游 stack，整体延迟不可见。
3. **Joint AV generation**（OmniForcing, Hallo-Live）：first-chunk 延迟低，但脱离 perception/reasoning。

Wan-Streamer 是唯一同时把"perception + reasoning + 同步生成 + turn-taking"压在一个 causal Transformer 里的工作，因此 550 ms 才真正可比。

---

## 9. 与我个人关注点的串联（build intuition）

### 9.1 与 Moshi 的对比

[Moshi](https://arxiv.org/abs/2410.00037)（Kyutai）是全双工 speech-text foundation model 的标杆，提出 "parallel streams" 思路，理论 160 ms 实际 200 ms 模型延迟——Wan-Streamer 的数字几乎照搬 Moshi 的下限，但**多了 video 通道**。这意味着：

- Wan-Streamer 的 complexity ≈ Moshi + 1 个视频 latent stream + 1 个视频 VAE + 1 个 causal video decoder。
- 200 ms 模型延迟能维持，说明 video latent 的 denoising 没成为瓶颈——这只能靠 **flow matching 的步数大幅压缩（distillation）** + **performer/thinker 解耦并行** 才能实现。

### 9.2 与 Diffusion Forcing / Self-Forcing 的脉络

- **Diffusion Forcing**（[arXiv 2407.01392](https://arxiv.org/abs/2407.01392)）首次提出"每帧独立 noise level，clean frames 作 condition"的训练范式，把 next-token prediction 与 full-sequence diffusion 统一。
- **Self-Forcing**（[arXiv 2506.08009](https://arxiv.org/abs/2506.08009)）进一步让 student rollout 自己的 history，缓解 exposure bias。
- **Rolling Forcing**（[arXiv 2509.25161](https://arxiv.org/abs/2509.25161)）做 autoregressive long video diffusion。
- Wan-Streamer 把这条脉络推到 **multi-modal interactive** 场景，并加上 **DMD-style distribution matching** 做 distillation，思路完全自洽。

### 9.3 与 VASA-1 / OmniHuman-1 / Kling-Avatar 的差异

- [VASA-1](https://arxiv.org/abs/2404.10667)：audio-driven talking face，40 FPS、170 ms 前置延迟，但**不做对话推理**，相当于 Wan-Streamer 的"performer 子集"。
- [OmniHuman-1](https://arxiv.org/abs/2502.01061)：scaling one-stage 条件人像动画，单图驱动全身。
- [Kling-Avatar](https://arxiv.org/abs/2509.09595)：cascaded long-duration avatar，多模态 instruction grounding。
- 这些工作都假设"audio/text 已经准备好"，把 audio→video 的对齐留给 renderer。Wan-Streamer 把"准备 audio/text"这一步也吃进了模型。

### 9.4 与 GPT-4o Realtime API / Doubao Realtime / Hume EVI 3 的差异

- [GPT-4o Realtime API](https://openai.com/index/hello-gpt-4o/) 接受 audio/vision 输入，输出 speech，但**不输出 video**——avatar 渲染留给第三方。
- [Doubao Realtime Voice](https://seed.bytedance.com/en/realtime_voice) 是纯 speech-to-speech，~700 ms bare model latency。
- [Hume EVI 3](https://www.hume.ai/blog/introducing-evi-3) 纯 speech，model response <300 ms。
- 这些系统在 speech 维度延迟可比或更低，但在 video 输出维度完全缺失——所以 550 ms 对 Wan-Streamer 是 "买到了视觉"，而不是简单"慢一点"。

### 9.5 与 Body of Her 的精神传承

[Body of Her](https://arxiv.org/abs/2408.02879)（Tenglong Ao, 2024）是阿里探索 humanoid end-to-end agent 的初步研究，把 audio + visual 输入 + speech + full-body behavior + idling + response + manipulation 一起建模，**next frame within 42 ms @ 24 FPS**——这个数字非常激进。Wan-Streamer 在精神上是 Body of Her 的延续，但更聚焦"对话"而非"操作"，且把 production-grade serving latency 写得更清楚。两者共同点：**idling 也是一种生成行为**，模型在用户说话时也要继续 produce listening behavior。

---

## 10. 自然性、打断、主动说话（Section 3 后半）

论文用文字描述了几个 qualitative 行为，这些其实是 streaming 模型的 **emergent properties**：

1. **Idle state**：agent 不冻结成 portrait，维持 identity、gaze、breathing、micro-expression。
2. **Listening state**：对 user 的 speech/visual cue 产生 nod、gaze shift、posture change。
3. **Speaking state**：lip motion、facial dynamics、prosody 在 denoising 前耦合，无需事后 align。
4. **Interruption**：user 在 agent 说话时插话，agent 收到 user audio-video 后能 stop / shorten / redirect 自己的 speech。
5. **Proactive speaking**：user 桌上出现一个新物体，agent 主动评论。

这些行为不是显式规则，而是 Stage 2 训练中 **interleaved interaction data** 学出来的。**直觉**：这相当于把 "对话协议" 也变成了 in-context learning 的对象，而非 hardcoded state machine。

---

## 11. 局限与开放问题

论文 Section 5 老实交代：**当前 v0.1 验证在 192p 输出分辨率**。这是 proof of concept，scaling 到更高分辨率 "straightforward"。我个人的几个疑问：

1. **192p 是不是因果 VAE 的限速点？** 高分辨率下 causal VAE 的 receptive field 是否够？是否会引入 lookahead？
2. **long-horizon drift**：rolling distillation 缓解了，但 session 几小时后 identity 是否仍稳？DMD 的一步 distillation 在长视频上是否仍然稳定？[OmniForcing](https://arxiv.org/abs/2603.11647) 也报告了类似挑战。
3. **Block-causal attention 的 block 大小**：160 ms 内部双向——如果 user 在 block 中途插话呢？是否需要 sub-block causal？
4. **KV cache 通信带宽**：thinker → performer 每步传一个 KV slice，对于长 history（比如几千 streaming units 后），KV cache 会膨胀。是否需要 cache 压缩 / sparsification？
5. **Rollout error**：Stage 3 distillation 的 student trajectory 偏离 teacher 后，DMD 修正是否足够？我看到 [Self-Forcing](https://arxiv.org/abs/2506.08009) 在 long rollout 上仍有 quality drop。
6. **多用户场景**：current formulation 只有一对一。多人会议是否需要扩展 user stream？
7. **Tool use / action stream**：[DuplexSLA](https://arxiv.org/abs/2605.20755) 加了 synchronized action stream 给 tool use，Wan-Streamer 是否预留了 action channel？
8. **CFG distillation 的代价**：teacher CFG 蒸馏进 student，相当于把 CFG scale bake 进权重——如果 inference 时想动态调整多样性怎么办？
9. **DiT 架构细节**：论文没展开 unified diffusion transformer 的具体结构（MM-DiT? cross-attention? joint attention?）。我猜测借鉴了 Wan 2.1 的 [MM-DiT](https://arxiv.org/abs/2503.20314) 但加了 causal mask。
10. **数据配比**：Section 2.2 提到 understanding + generation + end-to-end interaction，但没给比例。Stage 1 与 Stage 2 的 transition 是关键——太多 understanding 数据会让模型变 turn-based，太多 interaction 数据会 lack grounding。

---

## 12. 与我自己的 nanoGPT 直觉的对照

nanoGPT 的核心信条是 " tokenize + train GPT + see what emerges"。Wan-Streamer 把这个信条推到极致：

- Tokenize audio → causal VAE latent；
- Tokenize video → causal VAE latent；
- 把 text token + audio latent + video latent 拼成 single causal stream；
- Train one Transformer with mixed objectives (CE for text, flow matching for AV)；
- See turn-taking / interruption / proactive speaking emerge。

**这正是 GPT 在 multimodal streaming 上的最终形态**。和 Musk 在 Grok 4 + Optimus 上押注的方向高度一致——**单一大模型 + causal stream 一切**。

---

## 13. 一段更工程化的"如果我要复现"清单

如果我要从零搭一个 mini-Wan-Streamer，我会按这个顺序：

1. **Causal audio VAE**：基于 EnCodec/SoundStream 改 causal convolution，禁用 future frames，验证 reconstruct SNR 不掉太多。
2. **Causal video VAE**：基于 Wan 2.1 VAE 改 3D causal conv，只看 past + current frame。
3. **Token 接口**：text 用 BPE，audio/video latent 用 patchify（2×2×2 patch）后线性投影。
4. **Transformer**：MM-DiT 骨架 + block-causal attention mask，block size = 160 ms 内所有 token。
5. **Training Stage 1**：mixed batch of (image understand, video understand, ASR, TTS, image gen, audio gen, video gen, joint AV gen)。
6. **Training Stage 2**：duplex interaction data——这个最难，需要真实人机对话录制或合成。开源数据如 [Common Voice](https://commonvoice.mozilla.org/) +合成 visual data 可能不够。
7. **Training Stage 3**：CFG teacher → student distillation + DMD + rolling rollout。
8. **Inference**：thinker + performer 双 GPU，KV cache 传输用 NVLink 或 RDMA。
9. **Serving**：CUDA graph + kernel fusion + bf16 / fp8 量化。
10. **Eval**：必须报 model-side + total latency 两条线，外加 naturalness MOS、interruption success rate、long-horizon identity FID。

---

## 14. 参考链接汇总

### 论文与项目主页
- Wan-Streamer 网站：https://wan-streamer.com/
- Wan 2.1（基础视频生成模型）：https://arxiv.org/abs/2503.20314
- Moshi（Kyutai）：https://arxiv.org/abs/2410.00037
- Body of Her：https://arxiv.org/abs/2408.02879
- Diffusion Forcing：https://arxiv.org/abs/2407.01392
- Self-Forcing：https://arxiv.org/abs/2506.08009
- Rolling Forcing：https://arxiv.org/abs/2509.25161
- DMD（Yin et al. 2023）：https://arxiv.org/abs/2311.18828
- DMD2：https://arxiv.org/abs/2405.14867
- VASA-1：https://arxiv.org/abs/2404.10667
- OmniHuman-1：https://arxiv.org/abs/2502.01061
- Kling-Avatar：https://arxiv.org/abs/2509.09595
- OmniForcing：https://arxiv.org/abs/2603.11647
- Qwen3：https://arxiv.org/abs/2505.09388
- Qwen2.5：https://arxiv.org/abs/2412.15115
- Qwen3-Omni：https://arxiv.org/abs/2509.17765
- MiniCPM-o 4.5：https://arxiv.org/abs/2604.27393
- StreamAvatar：https://arxiv.org/abs/2512.22065
- LiveTalk：https://arxiv.org/abs/2512.23576
- Hallo-Live：https://arxiv.org/abs/2604.23632
- TalkingMachines：https://arxiv.org/abs/2506.03099
- MIDAS：https://arxiv.org/abs/2508.19320
- U-Mind（CVPR 2026）：https://arxiv.org/abs/2604.27393
- X-Streamer：https://arxiv.org/abs/2509.21574
- LPM 1.0：https://arxiv.org/abs/2604.07823
- MAViD：https://arxiv.org/abs/2512.03034
- M.I.O：https://arxiv.org/abs/2512.13674
- DuplexSLA：https://arxiv.org/abs/2605.20755
- SALM-Duplex：https://arxiv.org/abs/2505.15670
- OmniFlatten：https://arxiv.org/abs/2410.17799
- FlowAct-R1：https://arxiv.org/abs/2601.10103
- Matrix-Game 3.0：https://arxiv.org/abs/2604.08995
- Causal World Modeling：https://arxiv.org/abs/2601.21998
- AvatarForcing (Ki)：https://arxiv.org/abs/2601.00664
- AvatarForcing (Cui)：https://arxiv.org/abs/2603.14331
- Seedance 2.0：https://arxiv.org/abs/2604.14148
- Recammaster：ICCV 2025
- GLM-4.5V / 4.1V-thinking：https://arxiv.org/abs/2507.01006
- Qwen2.5-VL：https://arxiv.org/abs/2502.13923
- Kimi-VL：https://arxiv.org/abs/2504.07491
- CogVideoX：ICLR 2025

### 商业系统
- GPT-4o：https://openai.com/index/hello-gpt-4o/
- Realtime API missing manual：https://www.latent.space/p/realtime-api
- Hume EVI 3：https://www.hume.ai/blog/introducing-evi-3
- Doubao Realtime Voice：https://seed.bytedance.com/en/realtime_voice
- Doubao Volcengine：https://www.volcengine.com/
- Mozilla Common Voice：https://commonvoice.mozilla.org/

---

## 15. 一段总结性的直觉

Wan-Streamer 的真正贡献**不在 200 ms 数字**，**也不在 25 FPS 视频生成**——这两个都能在单独模块里被堆出来。它的真正贡献是 **把 perception、reasoning、turn-taking、speech generation、visual generation、long-horizon identity 全部塞进同一个 causal state，并且让 train 和 inference 用同一个 causal 格式**。

这相当于把对话系统从"模块化软件"重新定义为"一个状态机的概率推断"。和 LLM 之于 NLP 的关系一样：**LLM 没有 NER 模块、没有 parser、没有 coreference resolver，但所有任务都涌现出来**；Wan-Streamer 没有 VAD、没有 turn-taking logic、没有 lip-sync repair，但所有行为都涌现出来。

这是真正的 "native-streaming"——一个模型、一个 causal stream、一个 history state。后面的工作很可能是继续 scaling：分辨率上 1080p / 4K、long-context 几小时 session、加入 action stream for embodied agent、加入多 user stream for 会议场景、加入 tool use for agentic task。这条路线如果走通，未来 real-time agent 的 serving 栈会长得非常像 nanoGPT + DiT + KV cache 分布式——而不是 ASR + LLM + TTS + renderer 的全家桶。

Andrej，从你训 nanoGPT 的角度，这个 paper 应该会让你有种"对，就该这么做"的会心一笑——它把 GPT 的 next-token philosophy 用 causal flow matching 推到了 audio-video 流式生成上，并且诚实地把 production latency 写成了 200/550 ms 这种工程师能直接对标的数字。
