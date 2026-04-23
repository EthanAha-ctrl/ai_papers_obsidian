








## Guide Labs：构建可解释、可审计的 AI 系统

### 🏢 公司核心定位

Guide Labs 是一家专注于 **Interpretable and Auditable AI Systems（可解释且可审计的 AI 系统）** 的公司。其官网的标语非常明确：

> *"We are building a new class of interpretable AI systems and foundation models that humans can reliably debug, trust, and understand."*

简而言之，他们要解决的核心问题是：**当前大模型（如 GPT、LLaMA 等）是"黑盒"，人类无法理解其输出的因果链条**，而 Guide Labs 正在构建一类全新的、**inherently interpretable（本质可解释）** 的 AI 系统。

---

### 🔬 核心产品：Steerling-8B

他们的旗舰模型叫 **Steerling-8B**，号称是 **"The First Inherently Interpretable Language Model"**（首个本质可解释的语言模型），于 2026 年 2 月 23 日发布。

关键特性包括三大 Attribution（归因）能力：

| Feature | 功能 | 解决什么问题 |
|---------|------|-------------|
| **Prompt Attribution** | 追溯 output 的哪部分是由 prompt 的哪部分触发的 | "模型为什么这么说？是输入的哪句话导致的？" |
| **Concept Attribution** | 追溯 output 背后的"概念"来源 | "模型用的什么概念/语义在推理？" |
| **Training Attribution** | 追溯 output 的行为模式来自训练数据的哪部分 | "模型这种回答方式是从哪里学来的？" |

---

### 🧠 第一性原理分析：为什么这个问题重要？

从第一性原理出发，当前 AI 面临的根本矛盾是：

$$\text{Capability} \uparrow \quad \text{vs.} \quad \text{Interpretability} \downarrow$$

**越强大的模型，越不可解释。** 这是因为：

1. **Standard LLM Architecture（标准 LLM 架构）**：
   - 输入 $\mathbf{x} = (x_1, x_2, \ldots, x_n)$ 经过 Transformer 层：
   $$\mathbf{h}^{(l)} = \text{TransformerBlock}(\mathbf{h}^{(l-1)})$$
   - 其中 $\mathbf{h}^{(0)} = \text{Embed}(\mathbf{x})$，最终输出 $P(y \mid \mathbf{x})$
   - 中间表示 $\mathbf{h}^{(l)}$ 是高维连续向量，**不存在人类可读的语义结构**

2. **Post-hoc Interpretability（事后可解释性）的局限**：
   - SHAP、LIME、Attention 可视化等方法本质上是 **approximation（近似）**
   - 它们只能回答 "哪个输入特征对输出影响大"，但无法回答 **"模型的推理过程是什么"**
   - 形式化地，post-hoc 方法给出的是：
   $$\hat{\phi}_i \approx \phi_i = v(S \cup \{i\}) - v(S)$$
   - 这是 Shapley value 的估计，但 **不是模型的真实决策路径**

3. **Guide Labs 的思路——Inherent Interpretability（本质可解释性）**：
   - 模型的内部表征 **本身就是可解释的**，而非事后附加
   - 类比：如果一个模型的 hidden state 可以直接映射到人类概念（如"情感"、"主题"、"因果"），那就不需要 post-hoc 方法

---

### 🏗️ 技术架构推测

基于 "Steerling" 这个名字（Steer = 操控/引导），以及三个 Attribution 功能，可以推测其技术路线可能涉及：

#### 1. **Concept Bottleneck Architecture（概念瓶颈架构）**

$$\mathbf{x} \xrightarrow{f_{\theta}} \underbrace{\mathbf{c} \in \mathbb{R}^K}_{\text{Concept Space}} \xrightarrow{g_{\phi}} y$$

- 其中 $\mathbf{c}$ 是 **中间概念层**，每个维度 $c_k$ 对应一个人类可理解的概念
- 模型必须先预测概念 $\mathbf{c}$，再基于概念预测输出 $y$
- 这样就可以做到 **Concept Attribution**：看哪个 $c_k$ 对 $y$ 贡献最大

#### 2. **Sparse Autoencoder (SAE) 风格的可解释表示**

- Anthropic 的 SAE 研究表明，LLM 的内部表征可以被分解为 **sparse, monosemantic features**
- $$\mathbf{h}^{(l)} \approx \mathbf{D} \cdot \mathbf{z}, \quad \mathbf{z} \text{ is sparse}$$
- 其中 $\mathbf{D}$ 是 dictionary（字典），$\mathbf{z}$ 是 sparse code
- 每个 dictionary atom 可能对应一个可解释的概念
- Guide Labs 可能将这种 sparse decomposition **内建到模型架构中**，而非事后分析

#### 3. **Training Data Attribution（训练数据归因）**

- 这是最难的部分。相关技术包括：
  - **Influence Functions**：估计移除某个训练样本 $\mathbf{z}_j$ 对模型参数 $\theta$ 的影响
  $$\theta_{-j} - \theta \approx -\frac{1}{n} H_{\theta}^{-1} \nabla_{\theta} \ell(\mathbf{z}_j; \theta)$$
  其中 $H_{\theta} = \frac{1}{n}\sum_{i}\nabla_{\theta}^2 \ell(\mathbf{z}_i; \theta)$ 是 Hessian 矩阵
  
  - **Datamodels**（from Stanford SUML）：用线性模型预测 "哪些训练数据子集会导致某个测试输出"
  $$\hat{y}_{\text{test}} = \mathbf{w}^T \mathbf{1}_{S_{\text{train}}} + b$$
  其中 $\mathbf{1}_{S_{\text{train}}}$ 是指示向量，表示哪些训练数据被使用

#### 4. **"Steerling" 的可能含义——Steering Vectors**

- 在 Representation Engineering（表示工程）中，**steering vector** 是一种在 activation space 中控制模型行为的方法
- $$\mathbf{h}'^{(l)} = \mathbf{h}^{(l)} + \alpha \cdot \mathbf{v}_{\text{steer}}$$
- 其中 $\mathbf{v}_{\text{steer}}$ 是从对比数据中提取的方向向量（如 "truthful vs. hallucinated" 的方向）
- Steerling-8B 的名字暗示模型可能 **内置了可操控的概念方向**，使得用户可以 "steer" 模型的行为，同时理解为什么某个 steering 方向会产生某个输出

---

### 📊 与现有方法的对比

| 方法 | 类型 | Attribution 能力 | 精确度 | 计算开销 |
|------|------|-----------------|--------|---------|
| Attention Visualization | Post-hoc | Prompt-level only | 低 | 低 |
| SHAP/LIME | Post-hoc | Feature-level | 中（近似） | 中 |
| SAE (Anthropic) | Post-hoc | Concept-level | 中-高 | 高 |
| **Steerling-8B (Guide Labs)** | **Inherent** | **Prompt + Concept + Training** | **高（架构保证）** | **待验证** |

---

### 🌍 为什么现在？

1. **监管压力**：EU AI Act 要求高风险 AI 系统必须可解释
2. **安全需求**：AI Alignment 需要理解模型为什么做出某个决策
3. **企业信任**：金融、医疗等领域无法部署黑盒模型
4. **技术成熟**：Mechanistic Interpretability（机制可解释性）领域在 2023-2025 年取得了重大突破（Anthropic 的 SAE、OpenAI 的 superalignment 等）

---

### 🎯 总结

Guide Labs 做的事情本质上是 **将可解释性从"事后分析工具"提升为"架构级特性"**。他们的 Steerling-8B 模型通过内置的 Prompt/Concept/Training 三层 Attribution 机制，让模型的每一个输出都可以追溯到：

- **输入的哪部分**（Prompt Attribution）
- **模型使用了什么概念**（Concept Attribution）
- **训练数据的哪部分**（Training Attribution）

这是对当前 "黑盒 LLM + 事后解释" 范式的根本性挑战。

---

**参考链接：**
- Guide Labs 官网：https://www.guidelabs.ai/
- Concept Bottleneck Models 论文：https://arxiv.org/abs/2007.12599
- Anthropic SAE 研究：https://transformer-circuits.pub/
- Influence Functions 论文：https://arxiv.org/abs/1703.04730
- Representation Engineering：https://arxiv.org/abs/2310.01405