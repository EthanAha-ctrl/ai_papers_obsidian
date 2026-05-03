



















Meetily.ai 是一个**隐私优先（privacy-first）的开源 AI 会议助手**，核心特点是**100%本地处理（100% local processing）**，不上传任何音频或数据到云端。

## 核心功能

- **实时转录（live transcription）**：捕获会议音频并实时转换为文字
- **说话人分离（speaker diarization）**：自动识别和区分不同说话人
- **AI 总结（summarization）**：使用本地 LLM 生成会议纪要
- **开源且可自托管**：支持 Windows 和 macOS，可完全在自有基础设施上运行

## 技术架构深度解析

### 1. 语音识别（ASR）模块

Meetily 使用两种 ASR 引擎：

#### Parakeet-RNNT (NVIDIA NeMo)
Parakeet 是基于**循环神经网络转录器（Recurrent Neural Network Transducer, RNNT）**架构的模型。RNNT 的数学表达式为：

$$
P(y|x) = \prod_{t=1}^{T} P(y_t | y_{<t}, x_{\leq t})
$$

其中：
- $x$：输入音频特征序列
- $y$：输出字符/词序列  
- $x_{\leq t}$：到时间步 t 的音频特征
- $y_{<t}$：到 t-1 的输出序列

Parakeet 的优势在于**流式处理能力**，可实现低延迟实时转录，比 Whisper 快约 **4倍**（根据项目描述）。

#### Whisper (OpenAI)
作为备选方案提供更广泛的语种支持。

### 2. 说话人分离（Speaker Diarization）

采用 **pyannote.audio 2.1 pipeline**，包含三个关键阶段：

1. **语音活动检测（Voice Activity Detection, VAD）**：使用基于Transformer的模型检测语音区域
2. **说话人分割（Speaker Segmentation）**：使用 $\text{SE-ResNet}^\star$ 嵌入模型提取说话人特征向量 $\mathbf{e}_i \in \mathbb{R}^d$（通常 d=256）
3. **聚类（Clustering）**：使用 **谱聚类（Spectral Clustering）** 或 **DBSCAN** 将相同说话人的片段分组

嵌入相似度计算使用**余弦相似度**：
$$
\text{sim}(\mathbf{e}_i, \mathbf{e}_j) = \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{\|\mathbf{e}_i\| \|\mathbf{e}_j\|}
$$

### 3. 文本摘要（Summarization）

支持多种本地 LLM 提供商：
- **Ollama**（推荐）：运行 Llama 3、Gemma 3、Mistral 等模型
- 云端备选：Claude、Groq、OpenRouter、OpenAI

系统提示词（system prompt）结构通常包含：
```
你是一个专业的会议纪要助手。请根据以下转录文本生成结构化摘要：

转录内容：
[INSERT TRANSCRIPT]

请输出：
- 会议主题
- 关键决策（3-5条）
- 行动项（包含负责人和截止时间）
- 未决议题
```

### 4. 性能优化

- **Rust 实现**：相比 Electron/JS 栈（如 Otter.ai），Rust 提供零成本抽象和内存安全，显著降低延迟和内存占用
- **实时处理能力**：通过流式 RNN-T 实现近实时转录（<1× 实时速度，视硬件而定）
- **本地硬件加速**：利用 CUDA（NVIDIA GPU）或 Metal（Apple Silicon）加速推理

### 5. 架构图概览

```
[音频输入] → [音频捕获] → [VAD] → [ASR Engine] → [原始转录]
                                      ↓
                              [说话人嵌入提取] → [聚类] → [说话人标识]
                                      ↓
                          [带说话人标签的文本] → [LLM 摘要] → [会议纪要]
```

All processing happens **locally** → 无数据离开设备

## 产品版本

- **免费版**：基础转录和总结功能
- **Pro 版**：增强准确度、自定义 AI 连接器、自动会议检测、优先支持
- **企业版**：支持合规部署，适用于 healthcare、legal、finance、government 等敏感行业

## 与商业解决方案对比

| Feature | Meetily | Otter.ai | Microsoft Teams Premium |
|---------|---------|----------|---------------------------|
| 本地处理 | ✅ | ❌ | ❌ |
| 开源 | ✅ | ❌ | ❌ |
| 隐私保护 | 最高（数据不出设备） | 依赖云端加密 | 企业级加密 |
| 成本 | 一次性付费/免费 | 订阅制 | 订阅制 |
| 可定制性 | 高（可自行修改源码） | 低 | 中 |

## 局限性

- **硬件要求**：本地 LLM 需要较强 GPU 才能流畅运行大模型（如 Llama 3 70B 需 >24GB VRAM）
- **初始设置复杂**：需要安装 Ollama、配置模型等
- **功能对比**：缺少云方案的高级功能如自动会议安排、第三方集成（需自行开发）

## 技术社区与参考

- GitHub 仓库：https://github.com/Zackriya-Solutions/meetily
- 官方博客技术文章详解：https://www.zackriya.com/meetily-building-an-open-source-ai-powered-meeting-assistant/
- 性能基准测试参考：https://developer.nvidia.com/blog/pushing-the-boundaries-of-speech-recognition-with-nemo-parakeet-asr-models/
- pyannote 论文：https://ar5iv.labs.arxiv.org/html/1911.01255

Meetily 代表了一个**以隐私为核心、利用现代开源 AI 工具链（Rust + PyTorch + Ollama）构建的本地化会议解决方案**，是技术团队追求数据主权和成本控制时的优秀选择。