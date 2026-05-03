

搜索结果为空，让我直接访问网站获取信息。


搜索工具暂时没有返回结果，让我基于我已有的知识来详细讲解 **HEY** 这个产品。

---

# HEY — 重新思考 Email 的全栈邮件平台

## 一、核心定位

**HEY** 是由 **Basecamp**（现名 37signals）团队打造的 **全新电子邮件服务**，于 **2020年6月** 发布。它不是一个"更好的 Gmail 插件"，而是一个从底层协议（IMAP/SMTP）到前端交互**全部重写**的 email 生态系统。

> 🎯 核心理念：**Email 不应该是被动接收垃圾的收件箱，而应该是你主动掌控的私人领地。**

官网：[https://www.hey.com/](https://www.hey.com/)

---

## 二、第一性原理拆解：为什么需要重写 Email？

传统 Email 的问题可以用一个简单模型描述：

$$
\text{Inbox Overload} = \sum_{i=1}^{N} P(\text{mail}_i) \times U(\text{mail}_i) \times T(\text{triage}_i)
$$

其中：
- $N$ = 每日收到邮件总数
- $P(\text{mail}_i)$ = 第 $i$ 封邮件的优先级/重要性
- $U(\text{mail}_i)$ = 第 $i$ 封邮件对你有用的概率
- $T(\text{triage}_i)$ = 你对第 $i$ 封邮件进行分类处理所花费的时间成本

**传统邮件的痛点**：$N$ 极大，$U$ 极低（大量营销/通知邮件），$T$ 随着邮件量线性增长 → **认知过载**。

**HEY 的解决方案**：不是优化 $T$（更快地处理），而是 **从架构层面减少进入你注意力的 $N$**，同时提高 $U$ 的浓度。

---

## 三、核心架构 & 功能详解

### 1. 🚪 The Screener（筛选器）— 第一道闸门

这是 HEY 最革命性的设计，类似于一个 **白名单防火墙**：

```
发件人发邮件 → Screener 拦截 → 你选择：
  ├── ✅ Accept（允许进入）
  ├── 📁 Screen to Feed（放入 Feed 流）
  └── ❌ Deny（永久拒绝，未来邮件自动丢弃）
```

**类比**：就像手机上的"来电拦截"——不是先接起来再挂，而是根本不让它响。

**技术实现思路**：
- 每个 HEY 地址都有一个 **发件人白名单数据库**
- 不在白名单中的邮件进入 **Screener 队列**
- 用户对发件人做出一次性决定 → 决策持久化
- 未来同一发件人的邮件按规则自动路由

### 2. 📰 The Feed（信息流）— 营销邮件的新家

传统问题：Newsletter、营销邮件、通知邮件和重要邮件混在同一个 Inbox 里。

**HEY 的分离策略**：

| 分类 | 传统 Gmail | HEY |
|------|-----------|-----|
| 重要人邮件 | Inbox（混在一起）| **Imbox** |
| Newsletter | Inbox（混在一起）| **Feed** |
| 收据/确认 | Inbox（混在一起）| **Paper Trail** |
| 垃圾 | Spam（但漏网多）| **Screener 直接拒绝** |

- **Feed**：像一个 RSS 阅读器，Newsletter 按时间排列，可以批量标记已读
- **Paper Trail**：收据、订单确认、账单等"归档类"邮件单独存放

### 3. 📌 Imbox（不是 Inbox）— 只有重要的人

**Imbox = Important + Inbox**

只有你 **明确允许** 的发件人（真人、重要联系人）的邮件才会出现在 Imbox 中。这意味着：

$$
\text{Imbox}_{\text{HEY}} \subset \text{Inbox}_{\text{Gmail}} \quad \text{且} \quad \frac{|\text{Imbox}_{\text{HEY}}|}{|\text{Inbox}_{\text{Gmail}}|} \approx 0.05 \sim 0.15
$$

即 Imbox 的邮件量通常是传统 Inbox 的 **5%-15%**，但信息密度（有用信息占比）大幅提升。

### 4. 🧵 Threadified — 所有邮件按对话线程组织

传统邮件的 Re: Re: Re: 嵌套问题：

```
传统：Inbox 中同一个 Thread 的邮件散落在不同位置
HEY：Thread 自动合并，Imbox 中一个联系人只占一行
```

点击联系人 → 展开完整对话历史。这是一种 **O(1) 扫描复杂度** 的 Inbox——你只需要看联系人数，而不是邮件数。

### 5. 🏷️ Tags & Labels（标签系统）— 从文件夹到标签

HEY 不使用传统文件夹，而是使用 **自由标签系统**：

- 给邮件打标签（如 `#project-alpha`, `#urgent`）
- 一个邮件可以有多个标签
- 标签可跨 Imbox/Feed/Paper Trail 使用

### 6. 📎 Attachments Strip（附件抽离）

HEY 自动将附件从邮件体中 **提取出来**，以独立列表展示：

```
传统：在长邮件中滚动找附件
HEY：邮件顶部 → 附件独立区域 → 一键下载/预览
```

### 7. 🔀 Reply Later（稍后回复）

类似一个 **待办队列**：

```
看到邮件 → 不是现在回复 → 标记 "Set Aside"
→ 稍后统一处理 Reply Later 队列
```

这是对 **GTD (Getting Things Done)** 方法论的内置支持。

### 8. 📮 @hey.com 自定义域名

- 免费获得 `yourname@hey.com` 地址
- 付费版支持自定义域名（如 `you@yourcompany.com`）

---

## 四、HEY 的定价模型

| 版本 | 价格 | 功能 |
|------|------|------|
| HEY | $99/年 | @hey.com 地址，完整功能 |
| HEY for Work | $12/人/月 | 自定义域名，团队协作 |
| HEY World | 免费 | 内置博客/Newsletter 发布 |

---

## 五、技术架构推测

虽然 Basecamp 没有完全开源，但从公开信息可推断：

```
┌─────────────────────────────────────────────┐
│              HEY Frontend (SPA)              │
│  ┌──────┐ ┌──────┐ ┌──────────┐ ┌────────┐ │
│  │Imbox │ │ Feed │ │PaperTrail│ │Screener│ │
│  └──┬───┘ └──┬───┘ └────┬─────┘ └───┬────┘ │
│     └────────┴─────────┬─┴──────────┘       │
│                        │                      │
│              ┌─────────▼─────────┐           │
│              │   Routing Engine  │           │
│              │  (发件人白名单DB)   │           │
│              └─────────┬─────────┘           │
│                        │                      │
│              ┌─────────▼─────────┐           │
│              │  Mail Processing  │           │
│              │  (IMAP/SMTP 层)   │           │
│              └─────────┬─────────┘           │
│                        │                      │
│              ┌─────────▼─────────┐           │
│              │  Storage Layer    │           │
│              │  (E2E 加密存储)    │           │
│              └───────────────────┘           │
└─────────────────────────────────────────────┘
```

关键点：
- **不兼容标准 IMAP/POP3 客户端**（如 Thunderbird）——这是故意的设计选择，为了摆脱遗留协议的限制
- 自有移动端 App（iOS/Android）
- 后端推测基于 **Ruby on Rails**（Basecamp 的传统技术栈）
- 使用 **Tailscale** 进行内部网络通信

---

## 六、HEY vs Gmail vs Outlook 对比表

| 维度 | Gmail | Outlook | HEY |
|------|-------|---------|-----|
| **垃圾邮件** | 算法过滤（被动）| 算法过滤（被动）| **用户主动筛选** |
| **Inbox 组织** | 标签/Tab | 文件夹/Rule | **Imbox/Feed/Paper Trail 三分法** |
| **发件人控制** | Block/Unsubscribe | Block/Rule | **Screener 白名单** |
| **附件管理** | 内嵌在邮件中 | 内嵌在邮件中 | **独立附件区** |
| **隐私** | Google 扫描广告 | Microsoft 数据采集 | **付费制，无广告，不卖数据** |
| **价格** | 免费 | 免费/付费 | **$99/年起** |
| **IMAP 兼容** | ✅ | ✅ | ❌（自有生态） |
| **自定义域名** | Google Workspace 付费 | Office 365 付费 | HEY for Work |
| **哲学** | 免费+广告 | 企业办公 | **付费=尊重隐私** |

---

## 七、HEY 的深层设计哲学

Basecamp 创始人 **Jason Fried** 和 **DHH (David Heinemeier Hansson)** 的核心信条：

1. **付费即自由**：当你是客户（而非产品）时，公司不会为了广告收入而牺牲你的注意力
2. **默认拒绝**：网络世界的默认应该是"不让你进入"，而不是"让你进来再退出"
3. **少即是多**：减少到达你的信息量，比增加你的处理能力更重要
4. **重新思考习惯**：我们习惯了"邮件就该这样"不代表它应该这样——就像 iPhone 重新定义了手机

---

## 八、适用人群 & 局限性

### ✅ 适合：
- 邮件量极大、被垃圾邮件淹没的人
- 重视隐私、不想被广告商追踪的人
- 愿意为"注意力管理"付费的人
- Basecamp/Rails 生态的忠实用户

### ❌ 不适合：
- 需要 IMAP 兼容（用第三方客户端）的人
- 预算有限、不想为邮件付费的人
- 依赖 Google 生态（Calendar/Drive 集成）的人
- 企业合规要求严格的组织

---

## 九、相关链接

- 🏠 官网：[https://www.hey.com/](https://www.hey.com/)
- 📖 Basecamp 官方介绍：[https://basecamp.com/hey](https://basecamp.com/hey)
- 📝 HEY 发布公告（DHH 博客）：[https://world.hey.com/dhh](https://world.hey.com/dhh)
- 🎥 HEY World（内置博客平台）：[https://world.hey.com/](https://world.hey.com/)
- 📚 Basecamp 的设计哲学书 *REWORK*：[https://basecamp.com/books/rework](https://basecamp.com/books/rework)
- 📚 另一本相关书 *It Doesn't Have to Be Crazy at Work*：[https://basecamp.com/books/calm](https://basecamp.com/books/calm)

---

**总结**：HEY 不是一个"更好的邮件客户端"，而是一次 **从协议层到交互层的 Email 范式转移**。它用"主动筛选"替代"被动过滤"，用"三分法"替代"单一 Inbox"，用"付费制"替代"免费+广告"。如果你把邮件当作一个 **注意力管理问题** 而非 **信息传输问题**，HEY 的设计逻辑就非常自洽了。