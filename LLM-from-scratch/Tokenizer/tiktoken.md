









# tiktoken 深度解析：从第一性原理到实现细节

## 1. tiktoken 是什么？

**tiktoken** 是 OpenAI 开发的一个**高速 BPE (Byte Pair Encoding) tokenizer**，专门用于 OpenAI 的模型（GPT-4, GPT-3.5-turbo, text-embedding-ada-002 等）。它用 **Rust** 编写核心逻辑，通过 PyO3 暴露 Python 绑定，因此比纯 Python 实现快 **3-6 倍**。

GitHub: https://github.com/openai/tiktoken

---

## 2. 从第一性原理理解 BPE (Byte Pair Encoding)

### 2.1 核心问题

给定一段文本 $T$，我们想要找到一个 tokenization 方案，使得：
- **压缩率最大化**：用最少的 token 数量表示文本
- **泛化性好**：对未见过的文本也能合理分割
- **可逆性**：可以从 token 序列无歧义地恢复原文

### 2.2 BPE 的数学形式化

设初始字母表为 $\mathcal{V}_0$（对于 tiktoken，这是 UTF-8 的 256 个 byte），BPE 算法递归地构建词汇表：

**Step 1**: 初始化词汇表 $\mathcal{V}_0 = \{0, 1, 2, \ldots, 255\}$（所有可能的 byte values）

**Step 2**: 在当前 tokenization 的语料上，统计所有**相邻 token pair** 的频率：
$$\text{freq}(t_i, t_j) = \sum_{\text{all sequences } s \in \mathcal{D}} \text{count}_{s}(t_i, t_j)$$

其中：
- $t_i, t_j \in \mathcal{V}_k$ 是当前词汇表中的 token
- $\text{count}_{s}(t_i, t_j)$ 是 pair $(t_i, t_j)$ 在序列 $s$ 中出现的次数
- $\mathcal{D}$ 是训练语料

**Step 3**: 找到频率最高的 pair：
$$(t^*, t'^*) = \arg\max_{(t_i, t_j)} \text{freq}(t_i, t_j)$$

**Step 4**: 将该 pair 合并为新 token $t_{\text{new}}$，加入词汇表：
$$\mathcal{V}_{k+1} = \mathcal{V}_k \cup \{t_{\text{new}}\}$$

其中 $t_{\text{new}} = \text{concat}(t^*, t'^*)$，即两个 token 的字节拼接。

**Step 5**: 在语料中将所有 $(t^*, t'^*)$ 的出现替换为 $t_{\text{new}}$

**Step 6**: 重复 Step 2-5 直到词汇表大小达到目标 $|\mathcal{V}| = |\mathcal{V}_0| + N$，其中 $N$ 是 merge 次数。

### 2.3 编码（Encoding）的贪心策略

训练时学到的 merge 顺序决定了编码时的**优先级**。编码时，对于输入文本（转为 bytes 后），按 merge 顺序**从先到后**依次应用 merge 规则：

$$\text{encode}(s) = \text{apply\_merges}(\text{bytes}(s), \text{merge\_rules})$$

关键性质：**merge 规则的应用顺序是确定性的**（按训练时学到的 rank 排序），因此编码结果是唯一的。

---

## 3. tiktoken 的架构设计

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────┐
│                     Python Layer                         │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Encoding  │  │  Core API    │  │  Extensibility    │  │
│  │ Registry  │  │  encode()    │  │  Custom Encoding  │  │
│  │           │  │  decode()    │  │  (open() API)     │  │
│  └──────────┘  └──────┬───────┘  └───────────────────┘  │
│                        │  PyO3 binding                   │
├────────────────────────┼────────────────────────────────┤
│                  Rust Core (lib.rs)                      │
│  ┌──────────────────────┴────────────────────────────┐  │
│  │  1. Regex Pre-splitting (fancy-regex)              │  │
│  │  2. BPE Merge Engine (rank-based greedy)           │  │
│  │  3. Cache Layer (LRU cache for seen words)         │  │
│  │  4. Byte-level Unicode Handling                    │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 3.2 关键设计决策

#### 3.2.1 Regex Pre-splitting（正则预分割）

这是 tiktoken 与传统 BPE 实现最大的区别之一。在应用 BPE 之前，tiktoken 先用**正则表达式**将输入文本切分成"words"（或称为 chunks），然后对每个 chunk 独立进行 BPE。

**cl100k_base** 的正则模式：

```python
pat_str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

这个正则的含义逐段解析：

| Pattern | 含义 | 示例匹配 |
|---------|------|---------|
| `'(?i:[sdmt]\|ll\|ve\|re)` | 英文缩写：'s, 'd, 'm, 't, 'll, 've, 're | `don't` → `don`, `'t` |
| `[^\r\n\p{L}\p{N}]?+\p{L}+` | 可选的非字母数字前缀 + 字母序列 | ` hello`, `world` |
| `\p{N}{1,3}` | 1-3位数字 | `123`, `42` |
| ` ?[^\s\p{L}\p{N}]++[\r\n]*` | 可选空格 + 标点符号 + 可选换行 | ` !`, `, ` |
| `\s*[\r\n]` | 换行符（含前置空白） | `\n`, `  \n` |
| `\s+(?!\S)` | 末尾空白（negative lookahead） | trailing spaces |
| `\s+` | 其他空白 | `   ` |

**为什么这样做？**

1. **防止跨 word boundary 的 merge**：BPE 不会把不同单词的部分合并在一起
2. **提升效率**：每个 chunk 独立处理，便于并行和缓存
3. **确定性**：相同的 word 在不同上下文中 tokenization 结果一致

#### 3.2.2 Rank-based BPE Merge

tiktoken 不存储显式的 merge 规则列表，而是存储一个 **rank 字典**：

```python
# rank: bytes → int
# 每个 token 对应一个 rank，rank 越小表示合并优先级越高
encoder = {
    b' ': 1,
    b'e': 2,
    b' the': 3,
    b'ing': 4,
    ...
}
```

编码时，对每个 chunk 的 byte sequence，**贪心地找到 rank 最小（优先级最高）的相邻 pair 进行合并**，重复直到无法合并。

**算法伪代码：**

```
function bpe_encode(chunk_bytes, ranks):
    tokens = list(chunk_bytes)  # 初始化为单个 bytes
    while len(tokens) >= 2:
        # 找到所有相邻 pair 中 rank 最小的
        min_rank = infinity
        min_idx = -1
        for i in range(len(tokens) - 1):
            pair = tokens[i] + tokens[i+1]  # byte concatenation
            if pair in ranks and ranks[pair] < min_rank:
                min_rank = ranks[pair]
                min_idx = i
        if min_idx == -1:
            break  # 没有可合并的 pair
        # 执行合并
        tokens = tokens[:min_idx] + [tokens[min_idx] + tokens[min_idx+1]] + tokens[min_idx+2:]
    return tokens
```

**时间复杂度分析：**

设 chunk 长度为 $n$，最坏情况下需要 $O(n)$ 轮合并，每轮扫描 $O(n)$ 个 pair，总复杂度 $O(n^2)$。但实际中由于缓存和提前终止，远好于这个界。

tiktoken 的 Rust 实现做了进一步优化：
- **Early termination**：如果某个 pair 不在 rank 字典中，跳过
- **Batch merge**：同一轮中可以合并多个不重叠的 pair

#### 3.2.3 LRU Cache

tiktoken 对**已经见过的完整 word**（regex 分割后的 chunk）缓存其 BPE 结果：

```rust
// 简化的缓存结构
struct Tokenizer {
    cache: HashMap<Vec<u8>, Vec<TokenId>>,  // word bytes → token ids
    // LRU eviction policy
}
```

由于 regex 预分割的 chunk 通常是完整的 word（如 ` hello`），同一个 word 在不同文本中会被重复遇到，缓存命中率很高。这是 tiktoken 高速的关键原因之一。

---

## 4. Encoding 类型与词汇表

### 4.1 各 Encoding 对比

| Encoding | 模型 | 词汇表大小 | 特点 |
|----------|------|-----------|------|
| `r50k_base` | GPT-3 (davinci) | 50,257 | 最早的编码，基于 GPT-2 的 BPE |
| `p50k_base` | Code models (code-davinci) | 50,281 | 在 r50k 基础上增加了代码 token |
| `p50k_edit` | Edit models | 50,281 | 用于 edit 模式的特殊编码 |
| `cl100k_base` | GPT-4, GPT-3.5-turbo, text-embedding-ada-002 | 100,256 | 词汇量翻倍，更高效 |
| `o200k_base` | GPT-4o, GPT-4o-mini | 199,997 | 词汇量再翻倍，多语言优化 |

### 4.2 词汇表大小计算

词汇表大小 = **256** (base byte tokens) + **N** (merge tokens)

例如 `cl100k_base`：
$$|\mathcal{V}| = 256 + 100{,}000 = 100{,}256$$

### 4.3 词汇表文件格式

tiktoken 的词汇表存储在一个 JSON 文件中（或者是一个 B64 编码的文件），格式为：

```json
{
    "pat_str": "regex pattern string",
    "special_tokens": {
        "<|endoftext|>": 100257,
        "<|fim_prefix|>": 100258,
        ...
    },
    "bpe_ranks": "base64 encoded binary data"
}
```

**bpe_ranks** 的二进制格式：
- 每 4 bytes 为一个 `u32`，表示一个 rank 值
- 每个 token 的 byte representation 通过某种序列化方式存储
- 实际上，tiktoken 使用了一种紧凑的序列化：每行是一个 `(token_bytes, rank)` 对

---

## 5. Special Tokens

Special tokens 是不在正常 BPE 词汇表中的特殊标记，用于控制模型行为：

```python
# cl100k_base 的 special tokens
special_tokens = {
    "<|endoftext|>": 100257,    # 文本结束标记
    "<|fim_prefix|>": 100258,   # Fill-in-the-Middle 前缀
    "<|fim_middle|>": 100259,   # Fill-in-the-Middle 中间
    "<|fim_suffix|>": 100260,   # Fill-in-the-Middle 后缀
    "<|endofprompt|>": 100261,  # Prompt 结束标记
}
```

**编码时的处理**：

```python
enc = tiktoken.get_encoding("cl100k_base")

# 默认不允许 special tokens（抛异常）
enc.encode("<|endoftext|>")  # 会把 <|endoftext|> 当普通文本 tokenize

# 显式允许
enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})  # → [100257]
enc.encode("<|endoftext|>", allowed_special="all")  # 允许所有 special tokens
```

**为什么需要 `allowed_special` 参数？** 防止 injection attack——恶意用户在 prompt 中注入 special token 来操纵模型行为。

---

## 6. API 使用详解

### 6.1 基本 API

```python
import tiktoken

# 获取 encoding（按名称）
enc = tiktoken.get_encoding("cl100k_base")

# 获取 encoding（按模型名称，自动映射）
enc = tiktoken.encoding_for_model("gpt-4")
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

# 编码
tokens = enc.encode("Hello, world!")  
# → [9906, 11, 1917, 0]

# 解码
text = enc.decode(tokens)
# → "Hello, world!"

# 统计 token 数
num_tokens = len(enc.encode("Hello, world!"))  # → 4
```

### 6.2 编码选项

```python
# encode 的完整签名
enc.encode(
    text: str,
    *,
    allowed_special: set[str] | Literal["all"] = set(),  # 允许的 special tokens
    disallowed_special: set[str] | Literal["all"] = "all",  # 禁止的 special tokens
) -> list[int]
```

### 6.3 批量解码

```python
# decode 支持批量，但会拼接结果
enc.decode([9906, 11, 1917, 0])  # → "Hello, world!"

# decode_with_offsets 返回每个 token 对应的文本偏移量
enc.decode_with_offsets([9906, 11, 1917, 0])  
# → ("Hello, world!", [(0, 5), (5, 6), (6, 11), (11, 12)])
```

### 6.4 自定义 Encoding

```python
# 从 tiktoken 文件加载自定义 encoding
enc = tiktoken.Encoding(
    name="my_encoding",
    pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    mergeable_ranks={
        b" ": 1, b"a": 2, ...  # token bytes → rank 映射
    },
    special_tokens={
        "<|endoftext|>": 50257,
    },
)
```

---

## 7. 性能对比与基准测试

### 7.1 性能数据

根据 tiktoken README 提供的基准测试：

| Tokenizer | 速度 (tokens/sec) | 相对速度 |
|-----------|-------------------|---------|
| **tiktoken** | ~1.5M | **1.0x (baseline)** |
| HuggingFace Tokenizers (Rust) | ~1.0M | ~0.67x |
| SentencePiece (C++) | ~0.5M | ~0.33x |
| Python BPE (naive) | ~0.05M | ~0.03x |

### 7.2 tiktoken 高速的原因

1. **Rust 核心实现**：避免 Python 解释器开销，内存安全，零拷贝操作
2. **Regex 预分割**：将大文本切分成独立 chunks，减少 BPE 合并的搜索空间
3. **LRU Cache**：缓存常见 word 的 tokenization 结果，避免重复计算
4. **Rank-based 查找**：用 `HashMap<Vec<u8>, Rank>` 进行 O(1) 的 rank 查找，而非遍历 merge rules
5. **SIMD 优化**（部分操作）：利用 CPU 的 SIMD 指令加速 byte-level 操作

### 7.3 内存使用

tiktoken 将整个 rank 字典加载到内存中。对于 `cl100k_base`：
- 约 100K entries，每个 entry 是 `(Vec<u8>, u32)`
- 估计内存占用：约 **5-10 MB**

对于 `o200k_base`：
- 约 200K entries
- 估计内存占用：约 **10-20 MB**

---

## 8. BPE vs 其他 Tokenization 方法

### 8.1 方法对比

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **BPE** (tiktoken) | 贪心合并高频 byte pair | 确定性、可逆、高效 | 不考虑 token 语义 |
| **WordPiece** (BERT) | 类似 BPE，但用 likelihood 增益选 pair | 更好的语言学性质 | 更复杂 |
| **Unigram** (SentencePiece) | 从大词汇表剪枝到目标大小 | 概率模型、支持 subword regularization | 非确定性 |
| **SentencePiece** | 支持 BPE/Unigram，语言无关 | 不依赖预分词 | 速度较慢 |
| **Byte-level BPE** | BPE 在 byte 级别操作 | 无 OOV 问题 | 词汇表可能较大 |

### 8.2 tiktoken 的 BPE 与 GPT-2 BPE 的区别

| 特性 | GPT-2 BPE | tiktoken |
|------|-----------|----------|
| 预分词 | 简单 regex | 更复杂的 Unicode-aware regex |
| 词汇表 | 50,257 | 100,256 (cl100k) / 199,997 (o200k) |
| 实现 | Python (numpy) | Rust |
| 数字处理 | 按数字逐个 token | 按 1-3 位数字分组 |
| 缓存 | 无 | LRU cache |

**数字处理的关键区别**：

GPT-2 的 regex 会将数字逐个拆分：
```
"2023" → ["2", "0", "2", "3"]  (4 tokens)
```

cl100k_base 的 regex `\p{N}{1,3}` 会将数字按 1-3 位分组：
```
"2023" → ["202", "3"]  (2 tokens)
```

这对处理数字、日期等更高效。

---

## 9. 内部实现细节（Rust 核心代码解读）

### 9.1 核心结构体

```rust
// 简化的核心数据结构
pub struct CoreBPE {
    encoder: HashMap<Vec<u8>, Rank>,        // token bytes → rank
    decoder: HashMap<Rank, Vec<u8>>,        // rank → token bytes
    special_tokens: HashMap<String, Rank>,  // special token string → rank
    special_decoder: HashMap<Rank, String>,  // rank → special token string
    pattern: regex::Regex,                  // 预分词 regex
    cache: Mutex<LruCache<Vec<u8>, Vec<Rank>>>,  // LRU 缓存
}
```

### 9.2 编码流程

```rust
fn _encode(&self, text: &str) -> Vec<Rank> {
    let mut ret = vec![];
    
    // Step 1: 用 regex 将文本分割成 chunks
    for mat in self.pattern.find_iter(text) {
        let chunk = mat.as_str();
        let chunk_bytes = chunk.as_bytes();
        
        // Step 2: 检查缓存
        if let Some(cached) = self.cache.lock().get(chunk_bytes) {
            ret.extend(cached);
            continue;
        }
        
        // Step 3: 对 chunk 执行 BPE
        let tokens = self._bpe(chunk_bytes);
        
        // Step 4: 更新缓存
        self.cache.lock().put(chunk_bytes.to_vec(), tokens.clone());
        
        ret.extend(tokens);
    }
    
    ret
}
```

### 9.3 BPE 核心算法

```rust
fn _bpe(&self, piece: &[u8]) -> Vec<Rank> {
    // Step 1: 将每个 byte 映射为对应的 token rank
    let mut parts: Vec<(usize, Rank)> = (0..piece.len())
        .map(|i| {
            let byte_token = vec![piece[i]];
            (i, *self.encoder.get(&byte_token).unwrap_or(&0))
        })
        .collect();
    
    // Step 2: 贪心合并
    loop {
        if parts.len() == 1 {
            break;
        }
        
        // 找到 rank 最小的 adjacent pair
        let mut min_rank = Rank::MAX;
        let mut min_idx = 0;
        
        for i in 0..parts.len() - 1 {
            let pair_bytes = [&piece[parts[i].0..parts[i].0 + 1], 
                             &piece[parts[i+1].0..parts[i+1].0 + 1]].concat();
            // 实际实现中更复杂，这里简化了
            if let Some(&rank) = self.encoder.get(&pair_bytes) {
                if rank < min_rank {
                    min_rank = rank;
                    min_idx = i;
                }
            }
        }
        
        if min_rank == Rank::MAX {
            break;  // 没有可合并的 pair
        }
        
        // 执行合并
        parts.remove(min_idx + 1);
        // 更新 parts[min_idx] 的 token rank
    }
    
    parts.iter().map(|&(_, rank)| rank).collect()
}
```

### 9.4 实际的 BPE 优化（关键细节）

实际的 Rust 实现比上面更精巧。它使用了**两轮扫描**的策略：

**第一轮**：计算每个 adjacent pair 的 rank，找到所有可合并的 pair
**第二轮**：从左到右依次合并 rank 最小的 pair

```rust
fn _bpe_merge(piece: &[u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<Rank> {
    let mut parts: Vec<usize> = (0..piece.len()).collect();
    
    loop {
        if parts.len() == 1 {
            break;
        }
        
        // 计算 each adjacent pair 的 rank
        let mut min_rank: Option<(Rank, usize)> = None;
        for i in 0..parts.len() - 1 {
            // 构造 pair 的 byte representation
            let range_start = parts[i];
            let range_end = if i + 2 < parts.len() { parts[i + 2] } else { piece.len() };
            let pair_bytes = &piece[range_start..range_end];
            
            if let Some(&rank) = ranks.get(pair_bytes) {
                let rank_val = (rank, i);
                if min_rank.is_none() || rank_val < min_rank.unwrap() {
                    min_rank = Some(rank_val);
                }
            }
        }
        
        let (min_rank_val, min_idx) = match min_rank {
            Some(r) => r,
            None => break,  // 没有可合并的 pair
        };
        
        // 合并 parts[min_idx] 和 parts[min_idx+1]
        parts.remove(min_idx + 1);
    }
    
    // 将 parts 映射回 token ranks
    // ...
}
```

**一个关键的 insight**：在找 min rank 时，不是找所有 pair 中全局最小，而是找**第一个** rank 最小的 pair 然后合并，这和训练时的 merge 顺序一致。

---

## 10. 解码（Decoding）过程

解码是编码的逆过程，相对简单：

```python
def decode(tokens: list[int]) -> str:
    byte_parts = []
    for token in tokens:
        if token in decoder:  # 普通 token
            byte_parts.append(decoder[token])
        elif token in special_decoder:  # special token
            byte_parts.append(special_decoder[token].encode('utf-8'))
        else:
            raise ValueError(f"Unknown token: {token}")
    return b''.join(byte_parts).decode('utf-8', errors='replace')
```

**注意**：UTF-8 解码可能失败！因为 token boundary 不一定对齐 UTF-8 character boundary。tiktoken 使用 `errors='replace'` 处理这种情况。

**Batch 解码优化**：`decode(tokens)` 会将所有 token 的 bytes 先拼接，再一次性 UTF-8 解码，比逐个 decode 更快。

---

## 11. 计算 Token 数量的实用技巧

### 11.1 精确计数

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
```

### 11.2 估算公式

对于英文文本，粗略估算：
$$N_{\text{tokens}} \approx \frac{N_{\text{chars}}}{4}$$

对于中文文本，由于 UTF-8 编码每个汉字占 3 bytes，且 BPE 合并较少：
$$N_{\text{tokens}} \approx \frac{N_{\text{chars}} \times 3}{2} \approx 1.5 \times N_{\text{chars}}$$

即中文的 token 效率远低于英文——这是 cl100k_base 和 o200k_base 被诟病的主要原因之一，也是 o200k_base 大幅扩展词汇表的动机。

### 11.3 o200k_base 的改进

`o200k_base` (词汇量 ~200K) 相比 `cl100k_base` 的主要改进：
- 更多中文、日文、韩文 token
- 更多代码 token
- 更多 emoji token
- 平均 token 效率提升约 **15-20%**（针对非英文文本）

---

## 12. tiktoken 的局限性与替代方案

### 12.1 局限性

1. **只支持 OpenAI 模型的编码**：没有通用 tokenizer 的灵活性
2. **编码与解码不完全对称**：`encode(decode(tokens))` 不一定等于 `tokens`（因为 UTF-8 边界问题）
3. **不支持 subword regularization**：Unigram 模型支持的概率采样，BPE 不支持
4. **大词汇表的内存占用**：o200k_base 加载需要更多内存
5. **正则预分割的限制**：某些语言的书写系统（如 Thai、Lao）没有空格分词，regex 分割可能不理想

### 12.2 替代方案

| 库 | 特点 | 链接 |
|----|------|------|
| HuggingFace `tokenizers` | 多种算法、Rust 核心、支持训练 | https://github.com/huggingface/tokenizers |
| SentencePiece | 语言无关、支持 BPE/Unigram | https://github.com/google/sentencepiece |
| `tiktoken` 的 Python 纯实现 | 教学目的，速度慢 | 社区实现 |

---

## 13. 延伸阅读与参考

- **tiktoken GitHub**: https://github.com/openai/tiktoken
- **BPE 原始论文**: Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (ACL 2016) - https://arxiv.org/abs/1508.07909
- **GPT-2 Tokenizer**: https://github.com/openai/gpt-2/blob/master/src/encoder.py
- **OpenAI Tokenizer 可视化**: https://platform.openai.com/tokenizer
- **HuggingFace Tokenizers 文档**: https://huggingface.co/docs/tokenizers
- **SentencePiece 论文**: Kudo & Richardson, "SentencePiece: A simple and language independent subword tokenizer" (EMNLP 2018) - https://arxiv.org/abs/1808.06226
- **tiktoken PyPI**: https://pypi.org/project/tiktoken/

---

## 14. 总结：Key Takeaways

1. **tiktoken = Rust + BPE + Regex Pre-split + LRU Cache**，四个支柱撑起了它的速度优势
2. **BPE 的本质**：贪心地合并最高频的 byte pair，merge 顺序即优先级
3. **Rank 字典**是 tiktoken 的核心数据结构，key 是 token 的 byte representation，value 是 merge 优先级
4. **Regex 预分割**确保 BPE 不跨越 word boundary，同时提供缓存友好性
5. **Special tokens** 需要显式 `allowed_special` 才能编码，这是安全设计
6. **从 r50k → p50k → cl100k → o200k**，词汇表不断扩大，多语言支持持续改进
7. **中文 token 效率低**是当前 BPE tokenizer 的通病，需要更大的词汇表或专门的多语言训练来缓解