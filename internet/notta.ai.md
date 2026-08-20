
AI meeting transcription（AI会议转录）和 note-taking（笔记工具）服务平台

Notta 采用 **deep learning-based ASR**（基于深度学习的自动语音识别）技术，能够将 audio/video（音频/视频）内容转换为文字。
AI-Powered Summary Generation（AI摘要生成）
- **Key discussion points extraction**：提取关键讨论点
- **Action items identification**：识别任务项和待办事项
- **Meeting insights generation**：生成会议洞察

集成- Zoom - Google Meet - Microsoft Teams - Cisco Webex

```
Audio Input → Pre-processing → Feature Extraction → Acoustic Model → Language Model → Post-processing → Text Output + NLP Processing → Summary Generation
```

#### 1. **Feature Extraction（特征提取）**
- **Mel-frequency cepstral coefficients (MFCC)**：提取音频频谱特征
- **Spectrogram analysis**：频谱图分析
- **Delta features**：一阶和二阶差分特征捕捉动态变化

#### 2. **Acoustic Model（声学模型）**
可能基于：
- **Convolutional Neural Networks (CNN)** 处理局部特征
- **Recurrent Neural Networks (RNN)** / **Long Short-Term Memory (LSTM)** 捕捉时序依赖
- **Transformer-based models**（如 Whisper 架构）进行序列建模

#### 3. **Language Model（语言模型）**
- **N-gram models**：传统统计方法
- **Neural language models**：基于 **BERT** 或 **GPT** 架构的上下文理解

#### 4. **Post-processing（后处理）**
- **Voice Activity Detection (VAD)**：过滤非语音段
- **Speaker diarization**：说话人分离和识别
- **Punctuation restoration**：标点符号恢复
- **Text normalization**：文本规范化

根据搜索结果：
- **Whisper API**：OpenAI Whisper 提供云端 API，定价基于使用量
- **Notta**：定价约 **$8.25-14/month**（月付）或 **$10-15 USD/month**，另外提供 **Free Plan**（免费计划）
- **Offline Whisper**：一次性付费 **$29**，完全离线运行

## 方案对比

| Feature       | Notta        | OpenAI Whisper | In-house Solutions  |
| ------------- | ------------ | -------------- | ------------------- |
| Deployment    | Cloud/SaaS   | API/Cloud      | On-premise          |
| Pricing       | Subscription | Pay-per-use    | Infrastructure cost |
| Languages     | 58+          | 100+           | Depends             |
| Customization | Limited      | Limited        | High                |
| Data Privacy  | Cloud        | Cloud          | On-premise          |
