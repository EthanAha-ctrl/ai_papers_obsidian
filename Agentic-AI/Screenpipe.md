Screenpipe是一个**开源的本地AI屏幕记忆系统**，它记录你的屏幕和音频，通过本地AI模型提取文本和转录，创建一个完全私有的、可搜索的数字记忆库。

## 核心技术架构分析

### 1. 事件驱动的屏幕捕获 (Event-Driven Capture)
Screenpipe采用Rust实现高性能捕获，不是简单地录制视频流，而是通过事件驱动机制：

- **主要捕获路径**：先尝试获取accessibility tree（可访问性树）文本，这是最准确的，因为它直接从操作系统UI层获取文字信息
- **回退机制**：当accessibility tree不可用时（如某些游戏或特殊应用），fallback到OCR提取
- **性能优化**：检测静态画面，避免重复处理相同内容

### 2. 多层次文本提取策略
```
屏幕数据 → [Accessibility API] → 结构化文本
         → [Tesseract/OCR] → 光学校正文本
         → [向量嵌入] → 语义搜索能力
```

Accessibility tree提取的变量：
- `text`: 提取的文本内容
- `app_name`: 当前应用名称
- `window_title`: 窗口标题
- `timestamp`: 时间戳

### 3. 音频处理管线 (Audio Pipeline)
音频处理采用分段转录：
```
原始音频 → 30秒片段切割 → 本地ASR模型 → 转录文本 + 时间戳
```

支持的本地ASR模型包括：Whisper、MLX-Audio、IBM-Granite、VibeVoice-ASR等。

采样率转换公式：`audio_samples_len = sample_rate * segment_duration`

### 4. 存储架构
- **SQLite数据库**：存储所有元数据、文本、嵌入向量
- **媒体文件**：原始截图和音频文件存储在本地文件系统
- **向量嵌入**：使用本地模型生成文本嵌入，实现语义搜索
- **存储占用**：约0.4 GB/月

数据库schema关键表：
- `screenshot_events`: 截图事件，包含时间戳、应用名、窗口信息
- `ocr_results`: OCR提取的文本结果
- `audio_transcription_results`: 音频转录结果
- `embeddings`: 向量嵌入用于相似性搜索

### 5. 隐私与安全性
- **100%本地默认**：所有数据存储在本地，不向云端发送
- **加密选项**：数据库支持加密（SQLCipher）
- **可选第三方集成**：仅在用户主动连接时才发送数据

### 6. "Pipes"自动化系统
Pipes是Screenpipe的AI代理系统：
- 每个pipe是一个`pipe.md`文件，包含提示词和调度计划
- 运行本地AI编码代理（如Claude Code）查询屏幕记忆
- 实现"自动化任何事"的能力

### 7. 性能优化技术
- **Rust高性能帧流API**：低延迟屏幕捕获
- **自适应捕获**：根据显示变化动态调整捕获频率
- **多监控器支持**：可配置每个显示器的捕获策略
- **并行处理**：OCR和转录工作线程池

## 解决的核心问题

1. **数字记忆增强**：人脑记不住所有看过的内容，Screenpipe提供第二记忆
2. **生产力工具**：快速搜索之前看过的任何信息
3. **AI训练数据**：为个人AI提供上下文
4. **隐私保护**：所有处理在本地完成，避免云服务隐私风险

## 参考链接
- [官方主页](https://screenpi.pe/)
- [GitHub仓库](https://github.com/screenpipe/screenpipe)
- [技术架构文档](https://docs.screenpi.pe/architecture)
- [隐私与安全](https://www.mintlify.com/screenpipe/screenpipe/advanced/privacy-security)
- [Reddit讨论](https://www.reddit.com/r/software/comments/1fjfwg0/screenpipe_open_source_247_screen_audio_capture/)

这是一个典型的"本地优先"AI应用，通过事件驱动架构和多种文本提取技术的组合，在保证性能的同时最大化准确性和隐私性。