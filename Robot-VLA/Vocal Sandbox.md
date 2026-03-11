# 《Vocal Sandbox：用于情境化人机协作的持续学习与适应》详细技术讲解

## 一、研究背景与核心问题

这篇文章来自Stanford大学，发表在CoRL 2024，提出了**Vocal Sandbox**框架，用于实现情境化人机协作中的无缝协作和实时学习。

### 核心问题：
现有的人机协作系统存在三个关键限制：
1. **技能库固定**：大多数系统使用静态预定义的技能库，无法在线学习新的低层技能
2. **抽象层次单一**：只能在单一抽象层次（通常是规划层）进行适应，无法同时学习高层行为和低层技能
3. **教学模式受限**：主要依赖单一模态反馈（如语言），无法利用多模态教学（语音、关键点、物理演示等）

### 研究动机：
论文中提供了一个极具启发性的**LEGO定格动画制作**场景作为motivation example（Fig. 1）：

**场景设定**：
- **用户角色**：动画师，负责摆放LEGO人偶和场景元素
- **机器人角色**：控制相机，执行精确的相机运动（如平滑追踪、变焦）

**协作流程**：
```
时间线t=1: 用户说"Let's get a tracking shot around the Hulk?"
         → LM尝试生成计划但失败（"tracking"概念不存在）
         → 机器人语音表达失败，请求教学
         
时间线t=2: 用户通过kinesthetic demonstration（物理引导演示）
         → 单次演示作为监督
         → 立即参数化新技能 β_track
         → 合成对应行为 track(loc: Location)
         
时间线t=3: 用户指令"push in on Loki and then track around the tower"
         → 机器人立即使用所学知识生成完整计划
         → 计划: zoom_in(Loki); track(Tower)
         → 通过GUI可视化运动轨迹
         → 用户确认后执行
```

## 二、Vocal Sandbox框架整体架构

Vocal Sandbox的核心思想是**多层次抽象的持续学习**，框架由两个核心组件构成：

### 2.1 架构图解析（Fig. 2）

```
┌─────────────────────────────────────────────────────────┐
│                    Vocal Sandbox 框架                      │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────────┐         ┌───────────────────┐  │
│  │   Language Model    │         │  Skill Policies    │  │
│  │      Planner        │◄────────┤  π_f (低层策略)    │  │
│  │  (高层规划器)        │  API    │                   │  │
│  │                     │ Λ_t    │ - Hand-coded       │  │
│  │  GPT-3.5 Turbo +   │        │ - Keypoint-Cond.  │  │
│  │  Function Calling  │        │ - DMPs            │  │
│  └─────────────────────┘         └───────────────────┘  │
│           │                              │               │
│           │ 计划 p_t                     │               │
│           │                              │               │
│  ┌──────────────────────────────────────────────┐      │
│  │         程序合成与教学模块 (§2.3)             │      │
│  │                                              │      │
│  │  • Argument Teaching (参数教学)              │      │
│  │  • Function Teaching (函数教学)              │      │
│  │  • 多模态反馈整合                            │      │
│  └──────────────────────────────────────────────┘      │
│           │                                              │
│           │ 用户反馈                                        │
│  ┌──────────────────────────────────────────────┐      │
│  │         多模态教学接口                         │      │
│  │  • Spoken Dialogue (语音对话)                │      │
│  │  • Object Keypoints (物体关键点)             │      │
│  │  • Kinesthetic Demo (物理演示)               │      │
│  └──────────────────────────────────────────────┘      │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### 2.2 核心数据流

**定义1：API规范 Λ_t**
```
Λ_t = (F_t, C_t)
```
其中：
- **F_t**: 函数集合（即"行为"集），时间步t时可用
- **C_t**: 参数字面量集合（typed literals）

每个函数 f ∈ F_t 定义为元组：
```
f = (n, σ, d, b)
```
- **n**: 语义化名称（如 `goto`）
- **σ**: 类型签名（如 `[ObjectRef] → None`）
- **d**: 人类可读的文档字符串（如 "Move above the specified obj"）
- **b**: 函数体（可组合的低层技能序列）

每个参数 c ∈ C_t 是类型化字面量，例如：
```
Location.HOME → 值: ([0.36, 0.00, 0.49]_pos, [1, 0, 0, 0]_rot)
```

## 三、高层规划：语言模型（§2.1）

### 3.1 规划数学形式化

**定义2：规划过程**
```
p_t = LM(· | u_t, Λ_t, h_t)
```
其中：
- **u_t**: 时间t时的用户语音
- **Λ_t**: 当前API规范（动态增长的）
- **h_t**: 完整交互历史
- **p_t**: 生成的计划（程序序列）

交互历史定义：
```
h_t = [(u_1, p_1), (u_2, p_2), ..., (u_{t-1}, p_{t-1})]
```

### 3.2 具体实现细节

**技术栈**：
- **模型选择**: GPT-3.5 Turbo with function calling [v11-06]
- **API编码**: Λ_t 编码为Python代码块（Markdown格式）
- **输出约束**: 使用OpenAI的function calling特性

**成本与延迟**（附录B.2）：
```
GPT-3.5 Turbo费用: $2.00 / 1M tokens
响应时间: 1-3秒 (平均)
整个礼品袋组装用户研究费用: $1.24
整个项目GPT-3.5 API总费用: $5.79
```

### 3.3 计划生成示例

用户指令: "place the candy in the gift bag"

生成的计划:
```python
p_t = pickup(ObjectRef.CANDY); 
      goto(ObjectRef.GIFT_BAG); 
      release()
```

**异常处理**：当LM无法生成有效计划时（如"pack a gift bag with three candies"），会：
1. 产生异常
2. 利用LM的常识理解生成有用的错误信息
3. 语音表达："I am not sure how to pack; could you teach me?"
4. 通过GUI展示给用户

## 四、低层技能策略（§2.2）

### 4.1 技能策略数学定义

**定义3：技能策略**
```
π_f(a | o_t, Jc₁, c₂, ...K)
```
映射：
- 输入：状态观察 o_t ∈ R^n + 解析后的参数 Jc₁, c₂, ...K
- 输出：机器人动作序列 a ∈ R^{T×D}（如末端执行器姿态）

### 4.2 解析操作符 J·K

**解析操作符的作用**：将LM生成的计划p_t映射为技能策略调用序列

```python
# 示例：展开高层函数
Jb_pickupK → goto(ObjectRef.Candy); release()

# 参数解析示例
JObjectRef.CandyK = "A gummy, sandwich-shaped candy"
JLocation.HOMEK = ([0.36, 0.00, 0.49]_pos, [1, 0, 0, 0]_rot)
```

### 4.3 三类技能策略

**类型1: Hand-coded Primitives（手写原始技能）**
- GO_HOME
- GRASP
- 等...

**类型2: Visual Keypoint-Conditioned Policies（视觉关键点条件策略）**
- 用于物体抓取和放置
- 输入：自然语言物体描述
- 输出：末端执行器姿态

**类型3: Dynamic Movement Primitives（动态运动原语）**
- 用于动态相机运动
- 学习方式：kinesthetic demonstration
- 特性：可泛化到新的起点/终点，支持时序缩放

## 五、教学方法：程序合成（§2.3）

这是Vocal Sandbox的核心创新，通过**程序合成**实现实时能力增长。

### 5.1 教学流程架构

```
┌─────────────────────────────────────────────────────┐
│           教学触发条件                                 │
│         • 计划生成失败                                 │
│         • 执行失败                                      │
│         • 用户主动教学请求                               │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│         LM分析"缺失"概念                                 │
│    → 语音表达需要学习的内容                              │
│    → 提示用户提供针对性的反馈                            │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│        用户提供多模态反馈                               │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│         LM合成新函数和参数                              │
│         → 更新API规范 Λ_t → Λ_{t+1}                      │
└─────────────────────────────────────────────────────┘
```

### 5.2 Argument Teaching（参数教学）

**目标**: 教授并落地新的参数字面量 ĉ ∈ C

**示例场景**：
- 用户指令: "Grab the green toy car"
- LM解析: "grab" → pickup（正确）
- 问题: "green toy car" 不在 ObjectRef 中

**教学过程**：

```python
# Step 1: LM类型推断
LM使用pickup的类型签名 [ObjectRef] → None
推断 "green toy car" 应该是 ObjectRef 类型

# Step 2: 自动合成API更新
自动添加新字面量: ObjectRef.TOY_CAR

# Step 3: 用户提供监督
用户通过GUI点击图像中"green toy car"的位置
提供关键点监督定位物体

# Step 4: 提交变更
Λ_t → Λ_{t+1}
```

**关键技术**：
- 利用类型签名自动推断
- GUI作为教学接口（用户只需点击）
- 实时参数落地

### 5.3 Function Teaching（函数教学）

**目标**: 教授新的函数 ̂f ∈ F

**示例场景**：
- 用户指令: "now can you pack the candy in the bag"
- LM解析失败: "pack" 没有关联函数
- 问题: 无法可靠推断类型签名

**教学过程**：

```python
# Step 1: LM识别新行为
LM识别 "pack" 为新行为
明确请求用户通过分解[23,24]教学

# Step 2: 用户分解行为
用户说: "Pick up the candy; go above the bag; drop it"
用户提供程序: pickup(ObjectRef.CANDY); goto(GIFT_BAG); release()

# Step 3: LM合成新函数
̂f = (̂n, ̂σ, ̂d)
其中:
    ̂n = pack (名称)
    ̂σ = obj: ObjectRef → None (类型签名)
    ̂d = "Retrieve the object and place it in the gift bag." (文档)

# Step 4: 生成提升后的函数体
通过一阶抽象[24,25]生成:
pickup(obj); goto(GIFT_BAG); release()
```

**关键技术**：
- 分解式学习（decomposition-based learning）
- 一阶抽象（first-order abstraction）
- 自动文档生成

## 六、实现细节与技术决策（§3）

### 6.1 硬件平台配置

```
机器人平台:
- Franka Emika Panda 7-DoF 机械臂
- Robotiq 2F-85 并行夹爪
- ZED 2 RGB-D相机（俯视）

计算平台:
- Alienware M16 笔记本电脑
- NVIDIA RTX 4080 GPU (12 GB VRAM)
- 符合 DROID 平台规范[26]

交互硬件:
- Anker PowerConf S3 USB 麦克风扬声器
- 27英寸外接显示器（用于GUI）
- 物理按钮或脚踏板（push-to-talk）
```

### 6.2 Visual Keypoint-Conditioned Policy（§3.2）

**架构设计**（Fig. 6）：

```
输入: RGB图像 I_t ∈ R^{H×W×3} + 自然语言 referent c_ref
      ↓
┌──────────────────────────────────────┐
│  Two-Stream Architecture             │
│  (双流架构，参考Shridhar et al.[57])  │
│                                      │
│  ┌──────────────┐  ┌───────────────┐ │
│  │ Text Stream  │  │ Image Stream  │ │
│  │ (文本流)      │  │ (图像流)       │ │
│  │              │  │               │ │
│  │ CLIP Text    │  │ Fully-Conv.   │ │
│  │ Embedding    │  │ Architecture   │ │
│  │              │  │               │ │
│  └──────────────┘  └───────────────┘ │
│         │                │           │
│         └──────┬─────────┘           │
│                │ Fusion              │
│                ▼                     │
└──────────────────────────────────────┘
      ↓
输出: 逐像素分数矩阵 H ∈ [0,1]^{H×W}
      ↓
预测关键点: (x, y) = argmax(H)
```

**训练数据**：
```
数据集规模:
- 25张独特图像
- 每张图像标注3个关键点
- 总计75个示例

训练损失: Binary Cross-Entropy

Heatmap生成:
围绕每个关键点中心放置2D高斯
标准差: σ = 6 像素

数据增强:
- 随机裁剪
- 剪切变换
- 旋转
（保持标签不变）
```

**静态评估结果**（§B.3, Table 1）：

| 方法 | Keypoint MSE (px) | Precision (%) |
|------|-------------------|---------------|
| OWLv2 Ensemble (ViT-L/14) | 35.3±1.0 | 11.83±0.91 |
| GPT-4-Turbo (w/ Vision) | 36.39±1.73 | 15.94±2.55 |
| Vocal Sandbox (Ours) | **30.46±3.61** | **69.41±3.12** |

**Precision定义**：在玩具车半径（14像素）内的预测成功计数

**模块化方法的优势**：
1. **数据效率高**：仅需几十个示例
2. **精确度高**：显著优于OWL-v2和GPT-4V
3. **可解释性强**：用户只需在GUI中点击即可指定新物体

### 6.3 Mask Propagation Model (XMem)

**XMem架构**（Cheng and Schwing [33]）：

```
组件组成:
1. Query Encoder (查询编码器) e
2. Decoder (解码器) d
3. Value Encoder (值编码器) v
4. Memory Modules (记忆模块):
   - Short-term sensory memory (短期感觉记忆) h_t
   - Working memory (工作记忆)
   - Long-term memory (长期记忆)

前向传播流程:
输入图像 I_t
    ↓
Query Encoder: q = e(I_t)
    ↓
Attention-based Memory Reading:
    从working memory和long-term memory读取
    ↓
提取特征 F (c_ref: 如"candy")
    ↓
Decoder: d(q, F, h_{t-1}) → 预测掩码 M_t
    ↓
Value Encoder: v(I_t, M_t) → 新特征
    ↓
添加到记忆历史 h_t
```

**网络架构细节**：
```
Query Encoder: ResNet-50
Value Encoder: ResNet-18 [59]

Decoder设计:
- 输入: [q, F, h_{t-1}] 拼接
- 上采样: 每次上采样2倍，直到stride=4
- Skip Connections: 从query encoder e的每一层融合
- 最终层: 3×3卷积 → 单通道logit → 上采样到图像尺寸
```

### 6.4 Point-Conditioned Segmentation (FastSAM)

**FastSAM架构**（[32]）：

```
组件:
1. YOLOv8分割模型 s
2. Point prompt引导选择

工作流程:
给定预测关键点 p
    ↓
分割模型输出: 包含p的掩码 M ∈ M_s
    ↓
将M添加到XMem记忆存储
```

### 6.5 Dynamic Movement Primitives (DMP)

**DMP数学形式化**（Ijspeert et al. [30], §B.4）：

离散DMP定义为二阶点动力学系统：

```
τÿ = α_y(γ_y(g - y) - ẏ) + f(x, g)
ẋ = -α_x x
```

其中：
- **y**: 系统状态（轨迹位置）
- **g**: 目标位置
- **x**: 相位变量（phase variable），控制时间进度
- **τ ∈ (0,1]**: 时间缩放因子
- **α_y, γ_y**: 增益项（gain terms）
- **f(x, g)**: 学习到的强制函数（forcing function）

**强制函数形式**：
```
f(x, g) = Σ_{j=1}^{J} ψ_j w_j / (Σ_{j=1}^{J} ψ_j) x (g - y_0)
```

其中径向基函数定义为：
```
ψ_j = exp(-h_j (x - c_j)^2)
```
- **c_j**: 基函数的中心（heuristically chosen）
- **h_j**: 基函数的高度（heuristically chosen）
- **w_j**: 需要学习的权重
- **y_0**: 初始位置
- **J**: 基函数数量

**参数设置**：
```
J = 32 (基函数数量)
α_y = 25
γ_y = 25/4
基函数参数: 按照前人工作[30]设定
```

**DMP学习**：
```
学习方法: Locally-Weighted Regression (LWR)
训练数据: Kinesthetic demonstration
```

**DMP的泛化特性**：

1. **新目标泛化**：通过指定新的g
   - 示例: "track(loc: Location)" 可应用于任意新位置

2. **时序缩放泛化**：通过重新缩放τ
   - 示例: "track around Loki in 30 frames" vs "track around the tower... let's try 2 seconds"

**策略参数化代数**：

```
π_d,β : (c_ref, l, N)

其中:
- c_ref: 学习的referent（学习的DMP标识）
- l: 新的目标位置（goal location）
- N: 路点数量（用于设置τ）
```

**DMP轨迹可视化**（Fig. 6 Right）：
```
GUI可视化内容:
- 原始kinesthetic演示轨迹（绿色，参考用）
- 计划的DMP轨迹（蓝色）
- 轨迹起点和终点的末端执行器姿态框架
- 用户可动态推进仿真以可视化完整rollout
```

### 6.6 语音识别与合成（§B.1）

**Speech-to-Text (Whisper [54])**：
```
模型: OpenAI Whisper API
延迟: < 0.5秒 (round-trip)
定价: $0.006 / 分钟转录
礼品袋组装用户研究总费用: < $0.50

接口设计: "Push-to-talk" (明确按下说话)
- 降低成本和词错误率
- 改善用户体验
- 用户更清楚系统状态
- 更快定位语音转录失败
```

**Text-to-Speech (OpenAI TTS [55])**：
```
定价: $15.00 / 1M 字符 (约200K词)
礼品袋组装用户研究费用: < $0.08

用途:
1. 确认提示: "does this plan look ok to you?"
2. 语音表达系统状态
3. 自适应查询用户教学: "I'm sorry, I'm not sure what the 'jelly-candy thing' looks like, could you teach me?"
```

### 6.7 控制器实现（§B.5）

**控制栈**：
```
修改版 DROID control stack (基于 Polymetis [61])

控制频率:
- 低层策略: 10 Hz (关节位置指令)
- 阻抗控制器: 1 kHz

两种compliance模式:
1. Stiff mode: 机器人执行低层技能时激活
2. Compliant mode: 用户提供kinesthetic demonstration时激活
```

**安全措施**：
```
1. Cancel按钮:
   - 当展示可解释轨迹时，用户可取消任何提议行为
   - 防止执行并立即回溯Vocal Sandbox系统

2. 执行中断:
   - 在执行任何低层技能期间，用户可用按钮中断机器人运动
   - 立即停止机器人运动并使其完全compliant

3. 紧急停止:
   - 用户和监督员都有硬件紧急停止按钮
   - 切断机器人电源并机械锁定机械臂
```

## 七、图形用户界面设计（§3.3）

### 7.1 设计动机

**问题场景**：
```
用户指令: "pack the candy"
失败模式: 机器人打包了错误物体（如ball）

困惑:
- 是LM规划器失败? (生成错误计划 pack(ball))
- 还是技能策略失败? (错误预测ball而不是candy的关键点)
```

### 7.2 GUI功能（Fig. 5 Right）

```
┌────────────────────────────────────────────────────┐
│                    Vocal Sandbox GUI                 │
├────────────────────────────────────────────────────┤
│                                                     │
│  1. 转录的用户语音                                  │
│     "pack the candy"                               │
│                                                     │
│  2. 当前模式                                        │
│     [Normal Execution | Teaching | ...]            │
│                                                     │
│  3. 生成的计划                                      │
│     pickup(ObjectRef.CANDY);                        │
│     goto(GIFT_BAG);                                 │
│     release()                                       │
│                                                     │
│  4. 可解释轨迹                                      │
│     • 预测的关键点和分割掩码                         │
│     • DMP输出的机器人预期路径                        │
│     • 始末姿态框架                                  │
│                                                     │
│  5. 交互历史                                        │
│     [(u_1, p_1), (u_2, p_2), ...]                   │
│                                                     │
│  6. 教学接口                                        │
│     • "点击"指定关键点                              │
│     • 物理按钮确认/取消                             │
│                                                     │
└────────────────────────────────────────────────────┘
```

### 7.3 GUI的双重作用

1. **透明度**：让用户了解系统状态
2. **教学接口**：可解释轨迹也作为提供教学反馈的接口（如"点击"指定关键点）

## 八、实验评估

### 8.1 实验1：用户研究——协作礼品袋组装（§4.1）

**任务描述**：
```
参与者和机器人协作组装4个礼品袋
固定物体集合: candy, Play-Doh, toy car
手写卡片: 从96词脚本转录

任务特点:
- 重复且耗时
- 研究用户如何教机器人最大程度地帮助性行为
- 并行化工作，最小化监督时间
```

**实验设计**：
```
被试: N=8 非专家用户 (3女/5男, 年龄 25.2±1.22)

设计: Within-subjects (被试内设计)
条件数: 3个系统
顺序: 随机化（跨用户）

准备阶段:
- 机器人能力说明表
- 教学接口使用说明（如适用）
- 练习任务: 扔不相干物体到垃圾桶（熟悉系统）
```

**自变量——机器人控制方法**（§4.1）：

```
1. Vocal Sandbox (VS) - 完整系统
   - 高层行为教学 + 低层技能教学

2. VS - (Low, High) - 静态基线
   - ablates both低层技能教学和高层计划教学
   - 代表如MOSAIC[8]的前人工作
   - 假设固定技能库和LM planner

3. VS - Low - 仅高层教学基线
   - ablates低层技能教学
   - 允许教学新的高层行为
```

**因变量**（§4.1）：

**客观指标**：
```
1. 机器人监督时间
   - 作为机器人能力的代理指标
   - 更能干的机器人应该需要更少监督

2. 指令语音数量

3. 每个指令执行的低层技能数量（行为复杂度）
   - 衡量高层行为的复杂度

4. 高层和低层的教学交互数量

5. 教学的高层行为数量
   - 平均16个新的低层技能

6. 低层技能执行错误数量
```

**主观指标**：
```
六维度定性调查:
1. Ease (易用性)
2. Helpfulness (帮助性)
3. Intuitiveness (直观性)
4. Willingness to use again (再次使用意愿)
5. Predictability (可预测性)
6. Trust (信任度)
```

### 8.2 客观结果（Fig. 4）

**监督时间**（Fig. 4 Left）：
```
观察:
- VS系统在监督时间上优于两个基线
- VS-(Low, High) 和 VS-Low 都需要更多监督

解释:
- 多层次抽象教学的能力
- VS的优势在时间分配可视化中更明显
  - instructing, confirming, parallelizing
```

**行为复杂度**（Fig. 4 Middle）：
```
指标: 每个语言指令执行的低层技能数量

观察:
- VS系统产生越来越复杂的行为
- 显著(p<0.05)比VS-(Low, High)基线更复杂

解释:
- 高层教学通过结构化函数合成的重要性
- 用户能快速教授和使用更复杂行为
```

**技能失败率**（Fig. 4 Right）：
```
观察:
- VS系统显示技能执行失败率下降
- 显著(p<0.05)比两个基线都少
- VS系统显示的技能失败数量是组装袋子数量的递减函数

解释:
- 低层教学独立重要性
- VS系统随时间改进的能力
```

### 8.3 主观结果（Fig. 5 Left）

```
六维度排名（1=最差, 5=最好）:

维度              VS    VS-(Low,High)  VS-Low
Ease            ★★★★☆      ★★☆☆☆        ★★★☆☆
Helpfulness     ★★★★☆      ★★★☆☆        ★★★☆☆
Intuitiveness   ★★★★☆      ★★☆☆☆        ★★★☆☆
Willingness     ★★★★☆      ★★★☆☆        ★★★☆☆
Predictability  ★★★☆☆      ★★☆☆☆        ★★★☆☆
Trust           ★★★☆☆      ★★☆☆☆        ★★★☆☆

显著性:
- VS在ease, helpfulness, intuitiveness, willingness显著优于两个基线 (p<0.05)
- 在predictability和trust上趋势明显但不显著
- 用户评论: "teaching is useful", "I loved how I was able to teach the robot certain skills"
```

### 8.4 实验2：扩展实验——LEGO定格动画（§4.2）

**实验设置**：
```
参与者: 有经验的系统用户（作者之一）
协作时长: 超过2小时的持续协作

用户角色:
- 导演
- 领导创意愿景
- 指导拍摄
- 安排LEGO场景

机器人角色:
- 控制相机
- 执行不同类型的动态拍摄

最终输出:
- 52秒定格电影
- 232个独立帧
```

**教学内容**：
```
电影概念通过kinesthetic demonstrations教学:
- "tracking" (追踪)
- "zooming" (变焦)
- "panning" (摇摄)

DMP学习:
- 不同的DMP对应不同的电影概念
- 通过物理引导演示拟合
```

**能力泛化**：

```
1. 位置泛化:
   - 教学技能能泛化到不同起点和终点位置

2. 时序泛化:
   - "pan around slowly" → pan_around运动，N=30帧 (8秒)
   - "pan around quickly" → 同样技能，N=8帧 (1.33秒)
```

**统计数据**：
```
总帧数: 232
完全自主动态相机运动帧数: 99帧 (43%)
这些帧中LEGO场景固定，相机运动由DMP rollouts控制

执行的新颖指令数量: 40
示例:
- "let's frame the tower in this shot"
- "zoom into Iron Man"
```

**时间线分析**：
```
2小时连续协作:

初始阶段:
- 用户教授基础电影概念
- 学习tracking, zooming, panning

中期阶段:
- 用户构建更复杂帧序列
- 迭代构建行为库

最终阶段:
- 复杂自主执行
- 最小监督
- 52秒电影完成
```

## 九、相关工作（§5）

### 9.1 Task Planning with Language Models

**相关工作对比**：
```
方法          LM使用  新技能生成  学习方式
Text2Motion[17]  GPT     ❌        离线
ProgPrompt[18]   GPT-4   ❌        离线
MOSAIC[8]        GPT     ❌        离线
InnerMonologue[9]  LM   ❌        离线
Voyager[41]      LLM    ✓        离线
RT-2[15]         多模态  ✓        预训练
RT-H[62]         多模态  ✓        预训练
Vocal Sandbox    GPT-3.5  ✓        在线
```

**关键区别**：
- **在线vs离线**: Vocal Sandbox设计轻量级学习算法在线从自然用户交互中学习新行为
- **技能生成方式**: 前人工作需要昂贵模拟和离线学习，或通过奖励函数参数化

### 9.2 Learning Generalizable Skills from Mixed-Modality Feedback

**反馈模态对比**：
```
工作          反馈模态          学习方式
KITE[31]      关键点           离线
MOKA[65]      标记提示         离线
GIRAF[12]     手势             离线
Yell at Your Robot[14]  语言纠正   在线
Vocal Sandbox  语言+关键点+物理演示  在线
```

**关键区别**：
- **多模态vs单模态**: 大多数方法限于特定类型的语言反馈（如目标规范或纠正）
- **语言不足**: Vocal Sandbox证明语言不足够教授新行为，特别是新物体落地或动态运动
- **多模态整合**: 框架同时利用多种反馈模态指导学习

## 十、讨论与未来工作（§6）

### 10.1 主要贡献总结

```
定量结果:
- 更少监督: -22.1% 主动监督时间
- 更复杂行为: +19.7% 自主性能复杂度
- 更少失败: -67.1% 技能失败
- 更高偏好: +13.9% 整体性能
  - 易用性: +20.6%
  - 帮助性: +10.8%
```

### 10.2 局限性

**局限性1: 灵巧性要求高的场景**
```
问题场景: 辅助洗澡（assistive bathing）
困难原因:
- 执行依赖从稀疏反馈快速适应的低层技能
- 需要更多数据捕获行为细微差别
```

**局限性2: 协作模式相对单一**
```
当前模式:
- 用户: 老师
- 机器人: 追随者

不适用的场景:
- 更平等的协作
- 多机器人团队
- 非教学型互动
```

### 10.3 未来研究方向

**方向1: 跨用户改进**
```
目标:
- 一个用户教学的能力可以转移到其他用户
- 知识共享和累积

挑战:
- 用户特定偏好
- 不同教学风格
```

**方向2: 样本高效的更具表现力技能学习**
```
目标:
- 学习需要更多数据的更表现力技能
- 灵巧操作技能

方法:
- 元学习
- 迁移学习
- 模拟到现实迁移
```

**方向3: 用户机器人能力模型**
```
研究问题:
- 人类对机器人能力的模型
- 信任建立
- 协作风格适配

应用:
- 个性化协作
- 自适应教学策略
```

**方向4: 扩展互动模态**
```
额外模态:
- 触觉反馈
- 非语言反馈
- 眼动追踪
- 肌电信号
```

**方向5: 多机器人团队**
```
场景:
- 多个机器人协作
- 复杂任务分配
- 团队协调

挑战:
- 通信协议
- 一致性维护
```

## 十一、附录补充（Appx. A-C）

### Appx. A: Motivating Questions

**Q1: 实现Vocal Sandbox的组件需求**
```
核心组件:
1. GPT-3.5 Turbo with function calling (任务规划)
2. 视觉关键点条件策略
3. FastSAM (点条件分割)
4. XMem (掩码传播)
5. Whisper (实时语音识别)
6. OpenAI TTS API (文本到语音)

硬件:
- 单台笔记本 (NVIDIA RTX 4080, 12 GB)
- USB麦克风扬声器
- 外接显示器
```

**Q2: 模块化 vs 端到端方法**
```
模块化优势:
1. 当前模型限制: 端到端模型在细粒度感知和落地方面仍有限制
2. 静态评估证明: 我们的简单CLIP-based关键点模型优于OWL-v2和GPT-4V
3. 失败隔离: 用户可系统性隔离故障并在正确的抽象层次解决

端到端限制:
- 视觉鲁棒性问题
- 分布偏移导致成功率下降
- 错误级联导致不可预测行为
```

### Appx. B: Implementing Vocal Sandbox

**系统架构细节**：
```
语音识别延迟: < 0.5秒 round-trip
语音合成成本: $15.00 / 1M 字符

硬件规格:
- Alienware M16 (DROID platform specification)
- RTX 4080 GPU, 12 GB VRAM
```

**成本统计**：
```
礼品袋组装用户研究 (N=8):
- Whisper: $0.47
- OpenAI TTS: $0.08
- GPT-3.5 Turbo: $1.24
总计: $1.79

整个项目:
- GPT-3.5 API: $5.79
- Whisper & TTS: ~$4.00
总计: <$10.00
```

### Appx. C: Extended Discussion & Future Work

**C.1 Modular vs End-to-End Approaches**
```
用户观察:
- 用户快速识别系统故障
- 用户co-adapt（共同适应）
- 示例: 用户发现关键点模型在Play-Doh上表现差
- 用户利用模块化隔离故障到特定模块
- 教授高层行为"assemble_bag()"总是最后打包Play-Doh
- 允许用户最大程度"脱离"积极监督

模块化优势:
- 给用户杠杆快速理解机器人能力
- 构建其优势
- 适应其弱点
```

**C.2 Baseline与前期工作的明确联系**
```
VS-(Low, High) → MOSAIC[8], Text2Motion[17], ProgPrompt[18]
VS-Low → InnerMonologue[9], Code as Policies[19], YAY[14]
VS → KITE[31], MOKA[65], GIRAF[12], DMPs[30]

评估设置差异:
- 前人工作: 二元成功率
- Vocal Sandbox: 情境化人机协作
- 角色分配: 明确角色（导演vs摄像机操作员）
- 任务成功: 必要条件（非充分条件）
```

**C.3 Broader Context and Future Work**
```
相关工作类别:
1. 情境化环境中明确人机团队[1,4,6,7,66]
2. 学习共享自主方法[67-69]
3. 辅助机器人平台[70-72]
4. 语言界面落地[8,73,74]

Vocal Sandbox定位:
- 建立在丰富历史工作之上
- 学习语言界面落地用户意图到低层机器人行为
- 未来: 构建这些互动和学习类型
```

## 十二、关键技术创新总结

### 12.1 技术创新点

1. **多层次抽象的持续学习**
   - 同时学习高层行为和低层技能
   - 在线实时能力增长

2. **多模态教学整合**
   - 语言 + 关键点 + 物理演示
   - 每种模态针对特定抽象层次

3. **程序合成作为教学机制**
   - Argument teaching
   - Function teaching
   - 一阶抽象

4. **可解释性和透明度**
   - GUI可视化计划
   - 可解释轨迹
   - 教学接口

5. **模块化架构**
   - 高层规划器与低层技能策略解耦
   - 故障隔离和定位
   - 用户co-adaptation

### 12.2 关键公式汇总

```
1. API规范:
   Λ_t = (F_t, C_t)

2. 函数定义:
   f = (n, σ, d, b)

3. 规划过程:
   p_t = LM(· | u_t, Λ_t, h_t)

4. 技能策略:
   π_f(a | o_t, Jc₁, c₂, ...K)

5. 交互历史:
   h_t = [(u_1, p_1), (u_2, p_2), ..., (u_{t-1}, p_{t-1})]

6. DMP动力学:
   τÿ = α_y(γ_y(g - y) - ẏ) + f(x, g)
   ẋ = -α_x x

7. DMP强制函数:
   f(x, g) = Σ_{j=1}^{J} ψ_j w_j / (Σ_{j=1}^{J} ψ_j) x (g - y_0)

8. 径向基函数:
   ψ_j = exp(-h_j (x - c_j)^2)
```

## 参考资源

**项目网站**：
- https://vocal-sandbox.github.io
- 语言提示词: https://vocal-sandbox.github.io/#language-prompts

**关键论文引用**：
1. MOSAIC: Wang et al. "Mosaic: A modular system for assistive and interactive cooking." arXiv:2402.18796, 2024
2. KITE: Sundaresan et al. "KITE: Keypoint-conditioned policies for semantic manipulation." CoRL 2023
3. DMP: Ijspeert et al. "Dynamical movement primitives: Learning attractor models for motor behaviors." Neural Computation 25, 2013
4. XMem: Cheng & Schwing. "XMem: Long-term video object segmentation with an atkinson-shiffrin memory model." ECCV 2022
5. FastSAM: Zhao et al. "Fast segment anything." arXiv:2306.12156, 2023
6. DROID: Khazatsky et al. "DROID: A large-scale in-the-wild robot manipulation dataset." RSS 2024
7. Yell at Your Robot: Shi et al. "Yell at your robot: Improving on-the-fly from language corrections." RSS 2024
8. Code as Policies: Liang et al. "Code as policies: Language model programs for embodied control." ICRA 2023