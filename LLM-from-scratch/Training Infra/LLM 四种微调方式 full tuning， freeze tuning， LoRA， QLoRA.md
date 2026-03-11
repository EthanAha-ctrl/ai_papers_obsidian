下面从**训练参数规模、计算与显存成本、效果与适用场景**三个维度，系统性说明 **4 种主流 LLM 微调方式**：**full tuning、freeze tuning、LoRA、QLoRA**。

---
![[Pasted image 20260118203220.png]]
## 1. Full Tuning（全参数微调）

### 核心机制

- **所有 model parameters 全部参与反向传播与更新**
    
- 包括 embedding layers、attention layers、FFN layers 等
    

### 特点

- 参数更新规模：**100%**
    
- 训练方式：standard fine-tuning
    
- 对 base model 权重直接修改
    

### 优点

- 表达能力最强
    
- 可最大程度适配新 domain / 新 task
    
- 对 distribution shift 适应性最好
    

### 缺点

- 计算成本极高
    
- 显存占用最大（parameters + gradients + optimizer states）
    
- 容易 overfitting（尤其是小数据）
    

### 适用场景

- 数据量大（≥ millions）
    
- 训练资源充足（multi-GPU / multi-node）
    
- 需要 **彻底改变模型行为**
    
- 典型场景：foundation model 二次训练、instruction model 从零构建
    

---

## 2. Freeze Tuning（冻结微调）

### 核心机制

- **冻结大部分 backbone parameters**
    
- 只训练：
    
    - top layers（如最后 N 层）
        
    - task-specific head（classifier / LM head）
        
    - 或新增 adapter layers
        

### 特点

- 参数更新规模：**通常 < 5%**
    
- Backbone 不发生权重变化
    
- 本质是 feature extractor + lightweight adaptation
    

### 优点

- 训练稳定
    
- 计算与显存成本低
    
- 不易破坏原有能力
    

### 缺点

- 表达能力受限
    
- 对复杂 reasoning / domain shift 效果有限
    
- 上限明显低于 full tuning
    

### 适用场景

- 数据规模小
    
- 任务相对简单（classification、slot filling）
    
- 希望快速验证想法
    
- 较早期 NLP fine-tuning 方案
    

---

## 3. LoRA（Low-Rank Adaptation）

### 核心机制

- **冻结原始权重 W**
    
- 在 attention / FFN 中引入低秩矩阵：
    
    ```
    W' = W + ΔW
    ΔW = A · B
    rank(A, B) = r << d
    ```
    
- 只训练 A、B 两个小矩阵
    

### 特点

- 参数更新规模：**~0.1%–1%**
    
- 通常插入位置：
    
    - Q / K / V / O projection
        
    - sometimes FFN
        

### 优点

- 极低显存与计算开销
    
- 效果接近 full tuning（在很多任务上）
    
- 可热插拔、多 task 共享同一 base model
    

### 缺点

- rank 选择需要经验
    
- 理论上表达能力仍受低秩限制
    
- 对极端 domain shift 不如 full tuning
    

### 适用场景

- 主流 instruction tuning
    
- 多任务 / 多客户模型部署
    
- GPU 资源有限但追求高效果
    
- 目前工业界最常用 PEFT 方法
    

---

## 4. QLoRA（Quantized LoRA）

### 核心机制

- **Base model 权重进行 4-bit quantization（如 NF4）**
    
- LoRA adapters 仍以 FP16 / BF16 训练
    
- 训练时：
    
    - quantized weights 只 forward
        
    - gradients 只流经 LoRA parameters
        

### 特点

- Base model：4-bit
    
- Trainable parameters：LoRA only
    
- 使用 paged optimizer（如 paged AdamW）
    

### 优点

- 显存占用极低（单卡可 fine-tune 30B+）
    
- 性能接近 LoRA / full tuning
    
- democratize LLM fine-tuning
    

### 缺点

- 实现复杂
    
- 对 quantization scheme 敏感
    
- 推理阶段通常仍需合并 / 特殊加载
    

### 适用场景

- 单张消费级 GPU（24GB / 16GB）
    
- 大模型 instruction tuning
    
- 开源模型社区与个人研究者
    

---

## 对比总结

|方法|Trainable Params|显存需求|性能上限|工业常用度|
|---|---|---|---|---|
|Full Tuning|100%|极高|⭐⭐⭐⭐⭐|低|
|Freeze Tuning|<5%|低|⭐⭐|很低|
|LoRA|~0.1–1%|很低|⭐⭐⭐⭐☆|很高|
|QLoRA|~0.1–1%|极低|⭐⭐⭐⭐☆|很高|

---

## 一句话决策建议

- **资源充足 + 强定制需求** → Full Tuning
    
- **轻量快速实验** → Freeze Tuning
    
- **工业级微调首选** → LoRA
    
- **单卡训练大模型** → QLoRA
    

如果你希望，我可以进一步：

- 从 **数学角度** 推导 LoRA
    
- 对比 **LoRA vs Adapter vs Prefix Tuning**
    
- 给出 **PyTorch / HuggingFace 实现示例**