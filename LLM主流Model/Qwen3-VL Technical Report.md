### 1. Text-based Video Timestamp (基于文本的视频时间戳) 详细解析

**Text-based Video Timestamp** 是 Qwen3-VL 在视频处理模块中的一项关键架构升级。它的核心思想是将视频的时间维度信息，从隐式的数学编码（如 T-RoPE）转变为**显式的、人类可读的文本 Token**，并作为前缀注入到视频特征流中。

这一改变极大地提升了模型在长视频理解和精确时间定位任务上的表现。

---

### 2. 背景：为什么要放弃 T-RoPE？

在 Qwen2.5-VL 中，视频的时间位置主要通过 **T-RoPE** (Time-synchronized MRoPE) 来编码。虽然有效，但在处理长视频时面临两大瓶颈：

#### 2.1 时间 ID 的稀疏性
在传统的 RoPE 或 MRoPE 机制中，时间 $t$ 被映射为一个离散的索引 ID（例如帧序号或秒数）。
*   **短视频**: ID 较小（如 0-100），旋转频率的变化平滑，模型容易捕捉相对时间关系。
*   **长视频**: 对于数小时的视频，时间 ID 可能会变得非常大（如 $t=180,000$ 秒）。在 RoPE 的频率公式 $\theta_i = \text{base}^{-2i/d} \times \text{pos}$ 中，当 $\text{pos}$ 极大时，导致某些维度的旋转值变得极其稀疏或剧烈震荡。
*   **后果**: 这种数值上的极端变化破坏了 RoPE 的平滑性，使得模型难以建立长距离的时间依赖关系。

#### 2.2 数据采样成本高
为了让 T-RoPE 学会适应不同的视频时长和帧率，训练数据需要覆盖极其广泛的 FPS 设置（从 15fps 到 60fps 甚至更高），并且在时间轴上分布均匀。构建这样庞大的高质量视频数据集成本极高。

---

### 3. Text-based Timestamp 的核心方案

Qwen3-VL 采用了一种非常直观且高效的策略：**让模型“读”时间，而不是“算”时间**。

#### 3.1 实现原理
对于视频的每一个 temporal patch（时间片的图像特征），在其特征序列的最前方**前缀**一个格式化的文本字符串 Token。

*   **输入**: 第 $i$ 帧图像，对应的视频时间戳为 $T$ 秒。
*   **处理**:
    1.  将 $T$ 转换为文本字符串，如 `<3.0>` 或 `<00:00:03>`。
    2.  将该字符串通过 LLM 的 Tokenizer 进行分词，得到文本 Token IDs。
    3.  将 Token IDs 嵌入为 Embedding。
    4.  将该 Embedding 作为序列的起始，拼接在该帧的 Visual Tokens 之前。

#### 3.2 双格式训练策略
为了增强模型的鲁棒性和泛化能力，Qwen3-VL 在训练阶段使用了两种时间格式，并要求模型学会对齐和理解它们：
1.  **秒数格式**: `<3.5>` 或 `<120.0>`。适合精确的时间描述。
2.  **HMS 格式**: `<00:00:03.5>` 或 `<00:02:00.0>` (Hours:Minutes:Seconds)。符合人类阅读习惯，非常适合长视频（如电影、长会议），避免出现巨大的数字导致数值预测失真。

**公式化表示**:
假设视频帧 $f_t$ 的视觉特征为 $V_t$，时间戳文本为 $S_t$，TokenEmbedding 函数为 $\mathcal{E}$，则输入序列 $I_t$ 构建如下：
$$ I_t = [ \mathcal{E}(S_t), V_t ] $$
整个视频的输入流为：
$$ I_{video} = Concat(I_0, I_1, ..., I_N) $$

---

### 4. 技术优势与深度解析

#### 4.1 利用 LLM 的先验知识
这是该方案最聪明的地方。大语言模型（LLM）在预训练阶段已经阅读了海量的文本，它天然理解**数字的大小关系**（10 大于 5）、**进制换算**（60秒=1分钟）以及**时间的语义**（After, Before, Duration）。
通过将时间转成文本，Qwen3-VL 直接复用了 LLM 强大的逻辑推理能力来处理时间，而不需要从头从视觉数据中学习这些常识。

#### 4.2 实现 "Time Grounding" (时间定位) 的零样本能力
当用户询问：“视频第 15 分 30 秒发生了什么？”时：
*   **T-RoPE 模型**: 需要将 "15:30" 内部映射到一个隐式的时间位置 ID，再去对齐注意力，这在长序列中容易漂移。
*   **Text-based 模型**: 模型只需要在输入流中找到前缀为 `<00:15:30>` 的那一段，并将注意力集中在该 Token 及其后续的 Visual Tokens 上。这本质上变成了一个 **Key-Value Retrieval** (键值检索) 问题，精度和鲁棒性大幅提升。

#### 4.3 缓解长序列外推问题
对于 256K 甚至更长的上下文，数值型位置编码容易出现外推能力下降。但文本Token 不受此影响，因为 `<01:00:00>` 无论放在序列的哪个位置，它所代表的语义是一致的。

---

### 5. 架构图示与伪代码

#### 5.1 架构图示

```text
Video Frame Index:    Frame 0           Frame 1          Frame 2 ...
                      (0.0s)            (0.5s)           (1.0s)
                         |                 |                 |
                         v                 v                 v
          +-------------------+  +-------------------+  +-------------------+
          | Text Token "<0.0>"|  | Text Token "<0.5>"|  | Text Token "<1.0>"|  <-- Timestamp Prefix
          +-------------------+  +-------------------+  +-------------------+
                         |                 |                 |
          +-------------------+  +-------------------+  +-------------------+
          | Visual Token [Patch Features V0]         | ...           |
          +-------------------+  +-------------------+  +-------------------+
                         |                 |                 |
                         v                 v                 v
        ==================================================================>
          LLM Input Token Sequence: [<0.0>, V0_0, V0_1, ..., <0.5>, V1_0, V1_1, ...]
```

#### 5.2 Python 代码实现

以下代码展示了如何生成这些时间戳 Token 并将其与视觉特征混合。

```python
import torch

class VideoTimestampProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def format_timestamp(self, seconds):
        """
        生成双格式时间戳文本。
        Args:
            seconds (float): 视频当前帧的起始时间（秒）
        Returns:
            list[str]: 包含秒数格式和 HMS 格式的字符串列表
        """
        # 1. 秒数格式 (e.g., <3.5>)
        sec_str = f"<{seconds:.1f}>"
        
        # 2. HMS 格式 (e.g., <00:00:03.5>)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        hms_str = f"<{hours:02d}:{minutes:02d}:{secs:04.1f}>"
        
        # 在训练时可以随机选择返回一个，或者返回拼接后的列表
        # 这里为了演示返回两种，实际使用时根据训练策略定
        return [sec_str, hms_str]

    def embed_timestamps(self, seconds_list, embedding_layer):
        """
        将时间戳字符串转换为 Embedding 并返回。
        Args:
            seconds_list: 每一帧的时间戳列表
            embedding_layer: LLM 的 Token Embedding 层
        Returns:
            Tensor: (Batch/SeqLen, TimestampLen, D_model)
        """
        all_tokens = []
        for sec in seconds_list:
            # 获取文本字符串
            text_strings = self.format_timestamp(sec)
            
            # 将字符串拼接成一个完整的字符串进行 Tokenize
            # 注意: 也可以分别 Tokenize 再 sum 或 concat，这里采用拼接更简单
            timestamp_text = " ".join(text_strings)
            
            # Tokenize
            token_ids = self.tokenizer.encode(timestamp_text, add_special_tokens=False)
            all_tokens.append(token_ids)
            
        # Padding 到最大长度
        max_len = max(len(t) for t in all_tokens)
        padded_tokens = torch.zeros(len(all_tokens), max_len, dtype=torch.long)
        for i, t in enumerate(all_tokens):
            padded_tokens[i, :len(t)] = torch.tensor(t)
            
        # 获取 Embeddings
        timestamp_embeddings = embedding_layer(padded_tokens)
        return timestamp_embeddings

# ==========================================
# 构建视频输入流 (Pseudocode)
# ==========================================

def build_video_input_stream(video_frames_features, video_seconds, tokenizer, llm_embed):
    """
    构建最终的 LLM 输入序列。
    """
    # 1. 生成时间戳 Embeddings
    ts_processor = VideoTimestampProcessor(tokenizer)
    ts_embeds = ts_processor.embed_timestamps(video_seconds, llm_embed) 
    # Shape: (Num_Frames, T_Len, D_Model)
    
    # 2. 视觉 Patch Embeddings (假设已经是 LLM 维度)
    # video_frames_features: (Num_Frames, Num_Patches_Per_Frame, D_Model)
    
    inputs = []
    for i in range(len(video_seconds)):
        # 获取当前帧的视觉特征
        frame_feats = video_frames_features[i] # (Patches, D)
        
        # 获取当前帧的时间戳嵌入
        timestamp_feats = ts_embeds[i]        # (T_Len, D)
        
        # 拼接: [Timestamp Tokens] + [Visual Patch Tokens]
        # 确保维度匹配: 拼接时通常沿着 Sequence 维度 (dim=0)
        
        timestamp_seq = timestamp_feats.view(-1, timestamp_feats.shape[-1]) # (T_Len, D)
        
        # 将视觉特征也展平
        visual_seq = frame_feats.view(-1, frame_feats.shape[-1])             # (Patches, D)
        
        # 拼接
        combined_seq = torch.cat([timestamp_seq, visual_seq], dim=0)
        
        inputs.append(combined_seq)
        
    # 3. 将所有帧序列拼接成最终的 Video Stream
    final_video_stream = torch.cat(inputs, dim=0) # (Total_Video_Tokens, D_Model)
    
    return final_video_stream

# Example Usage
# model_embed = ... (LLM's embedding layer)
# frames = ... (List of feature tensors)
# timestamps = [0.0, 0.5, 1.0, ...] (Corresponding to FPS)
# stream = build_video_input_stream(frames, timestamps, tokenizer, model_embed)
```

### 总结

Text-based Video Timestamp 是一种典型的 **"以自然理解代替数学拟合"** 的工程创新。

*   **旧方案 (T-RoPE)**: 把时间视为连续的几何位置，依赖模型从数据中拟合位置编码分布，对长视频（长尾分布）和复杂 FPS 不友好。
*   **新方案 (Text-based)**: 把时间视为离散的语义符号，直接写在视频流的每一帧开头。
    *   **代价**: 极小的上下文长度开销（通常只需 1-4 个 token）。
    *   **收益**: 模型瞬间获得了精确到秒级的时间感知能力，且具备了处理任意长度视频外推的鲁棒性。这使得 Qwen3-VL 能够胜任 Video Question Answering (VQA)、Temporal Grounding 和 Dense Captioning 等复杂视频任务。