

## Prolific.com 是做什么的？

**Prolific** 是一个连接 **Researcher（研究者）** 和 **Participant（参与者）** 的在线研究平台，核心使命是构建世界上最大规模的高质量、**人类衍生数据（human-derived data）** 池，并为研究者提供访问这些数据的终极平台。

---

### 🧩 双边平台模型

Prolific 的商业模式是一个典型的 **two-sided marketplace**：

| 角色 | 功能 | 动机 |
|---|---|---|
| **Researcher**（研究者） | 发布 study（研究任务），设置筛选条件，招募参与者 | 获取高质量、可信的人类数据 |
| **Participant**（参与者） | 完成调查/实验任务，获得报酬 | 赚取公平酬劳（约 £6–12/小时） |

其网络效应可简化为：

$$\text{Platform Value} \propto N_{\text{researchers}} \times N_{\text{participants}}$$

其中 $N_{\text{researchers}}$ 为研究者数量，$N_{\text{participants}}$ 为参与者数量。双边越多，平台价值越大。

---

### 🔬 核心功能与技术细节

#### 1. **Participant Pool（参与者池）**
- 拥有 **200,000+** 经过验证的全球参与者
- 每 **2 分钟** 就有新任务发布
- 参与者 100% 为真人（非 bot），平台有 **identity verification** 机制
- 支持按人口统计学特征（demographics）精准筛选：年龄、性别、国籍、教育程度、政治倾向等

#### 2. **Prescreening（预筛选）系统**
研究者可以创建自定义的 prescreening questionnaire，其工作流程为：

$$\text{Study Requirement} \xrightarrow{\text{filter}} \text{Eligible Participants} \xrightarrow{\text{random assignment}} \text{Sampled Subset}$$

这确保了 **sample representativeness（样本代表性）**，解决了传统 MTurk 等 platform 上常见的 **WEIRD bias**（Western, Educated, Industrialized, Rich, Democratic）问题。

#### 3. **Fair Pay Model（公平薪酬模型）**
Prolific 强制研究者按最低时薪标准支付：

$$\text{Reward} \geq \frac{\text{Minimum Hourly Rate}}{3600} \times \text{Estimated Completion Time (seconds)}$$

默认最低时薪约为 **£6.00**（高于很多同类平台），研究者可以自愿提高。

#### 4. **Data Quality Control（数据质量控制）**
- **Attention checks**：内置注意力检测题
- **Study completion rate threshold**：参与者需维持一定完成率
- **Approval/rejection system**：研究者可 reject 低质量提交（但有严格限制防止滥用）

---

### 🔄 完整工作流程

```
Researcher Side:                          Participant Side:
┌─────────────────┐                      ┌─────────────────┐
│ Create Study     │                      │ Register Profile │
│ (survey/experiment)│                    │ Complete Prescreen│
└────────┬────────┘                      └────────┬────────┘
         ▼                                        ▼
┌─────────────────┐                      ┌─────────────────┐
│ Set Prescreeners │  ←─── Matching ────→  │ Eligible Study   │
│ & Requirements   │     Algorithm        │ Notification     │
└────────┬────────┘                      └────────┬────────┘
         ▼                                        ▼
┌─────────────────┐                      ┌─────────────────┐
│ Collect Data     │                      │ Complete Task    │
│ (auto-verified)  │                      │ & Get Paid       │
└────────┬────────┘                      └────────┬────────┘
         ▼                                        ▼
┌─────────────────┐                      ┌─────────────────┐
│ Export & Analyze │                      │ Receive Reward   │
│ (CSV/JSON API)   │                      │ via PayPal/Stripe│
└─────────────────┘                      └─────────────────┘
```

---

### 📊 与竞品对比

| 特性 | Prolific | Amazon MTurk | Qualtrics Panels |
|---|---|---|---|
| 参与者质量 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 公平薪酬 | 强制最低标准 | 无强制标准 | 商业定价 |
| 数据多样性 | 全球化 | 美国-centric | 取决于 panel |
| Bot 污染 | 极低 | 严重问题 | 低 |
| 学术适用性 | 高 | 中 | 高但昂贵 |

---

### 🎯 主要用户场景

1. **Academic Research（学术研究）**：心理学、社会学、经济学实验（Behavioral experiments, Survey research）
2. **Public Policy Research（公共政策研究）**：利用多样化样本获取政策洞察
3. **AI/ML Training Data（AI 训练数据）**：人类反馈数据（Human Feedback），用于 RLHF（Reinforcement Learning from Human Feedback）等
4. **Market Research（市场研究）**：品牌调研、消费者行为研究

---

### 💡 总结

用一句话概括：**Prolific 是一个以"公平对待参与者、保证数据质量"为核心理念的在线研究参与者招募平台，让研究者能快速获取高质量人类数据，让参与者通过参与科学研究获得合理报酬。**

它是学术研究从"线下实验室招募"到"线上大规模数据采集"这一范式转变的关键基础设施之一。

---

**参考链接：**
- [What is Prolific? | About Prolific](https://www.prolific.com/about)
- [Participant Pool | Prolific](https://www.prolific.com/participant-pool)
- [How Prolific Works for Participants](https://www.prolific.com/participants-how-it-works)
- [Prolific for Research](https://www.prolific.com/researchers)
- [Prolific Help - What is Prolific and how does it work](https://participant-help.prolific.com/en/articles/445003-what-is-prolific-and-how-does-it-work)